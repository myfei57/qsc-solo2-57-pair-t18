"""热源调质调量、气象推送与停机复位。"""

from __future__ import annotations

import threading

import pytest

from heatnet.config import default_config
from heatnet.errors import ForecastError, InterlockError
from heatnet.source.adjust import apply_adjustment
from heatnet.source.joint import (
    BAND_KIND,
    FLOW_FIRST,
    HIGH_LOAD,
    LOW_LOAD,
    TEMP_FIRST,
    joint_rate_limit,
)
from heatnet.source.source import HeatSource
from heatnet.source.weather import WeatherPush, push_weather
from heatnet.temp.curve import HeatCurve


def _source() -> HeatSource:
    return HeatSource(default_config().source)


def _curve() -> HeatCurve:
    return HeatCurve(default_config().curve)


def test_adjust_registers_targets_and_rejects_same_step_overshoot():
    source = _source()
    # 冷态一步喊到 90℃/2000t/h 属抢跑：目标登记，本步拒绝，字段一个不写。
    result = apply_adjustment(source, 90.0, 2000.0, demand_pct=60.0, origin="manual")
    assert result.ok is False
    assert result.applied is False
    assert "joint_rate_band_exceeded" in result.violations
    assert result.temperature_c == pytest.approx(35.0)
    assert result.flow_t_h == pytest.approx(0.0)
    assert result.target_temperature_c == pytest.approx(90.0)
    assert result.target_flow_t_h == pytest.approx(2000.0)
    assert result.rate_limit["band_kind"] == "fixed"
    assert result.rate_limit["yield_to"] == "temperature"
    # 后续自动联动周期按限速逼近，不再需要重新下目标。
    for _ in range(600):
        move = source.joint_step(1.0, 60.0)
        if abs(source.flow_t_h() - 2000.0) < 1e-9 and abs(source.temperature_c() - 90.0) < 1e-9:
            assert move["ok"] is True
            break
    else:  # pragma: no cover - 收敛失败即用例失败
        raise AssertionError("联动未能在预算内收敛到目标")
    snapshot = source.snapshot()
    assert snapshot.target_c == pytest.approx(90.0)
    assert snapshot.target_flow_t_h == pytest.approx(2000.0)
    assert snapshot.state == "online"


def test_adjust_within_band_applies_immediately():
    source = _source()
    # 首步在温度速率内、流量保持零：整块在带内，当场生效。
    result = apply_adjustment(source, 35.35, 0.0, demand_pct=60.0, origin="manual")
    assert result.ok is True
    assert result.applied is True
    assert result.temperature_c == pytest.approx(35.35)
    assert result.flow_t_h == pytest.approx(0.0)
    snapshot = source.snapshot()
    assert snapshot.adjustments == 1


def test_invalid_target_is_rejected_without_touching_state():
    source = _source()
    apply_adjustment(source, 35.35, 0.0, demand_pct=60.0)
    rejected = apply_adjustment(source, 120.0, 2000.0, demand_pct=60.0)
    assert rejected.ok is False
    assert rejected.applied is False
    assert rejected.violations == ["调质目标越界"]
    snapshot = source.snapshot()
    assert snapshot.target_c == pytest.approx(35.35)
    assert snapshot.temperature_c == pytest.approx(35.35)
    assert snapshot.flow_t_h == pytest.approx(0.0)


def test_rate_band_rejects_overshoot_on_the_spot():
    source = _source()
    source.set_target(60.0)
    source.set_flow_target(2000.0)
    # 冷态供温未进固定温差带，提议直接把流量拉起必须当场拒绝，两个字段都不写。
    rejected = source.joint_step(1.0, 60.0, temperature_c=35.35, flow_t_h=2000.0)
    assert rejected["ok"] is False
    assert rejected["applied"] is False
    assert "joint_rate_band_exceeded" in rejected["violations"]
    assert source.temperature_c() == pytest.approx(35.0)
    assert source.flow_t_h() == pytest.approx(0.0)
    # 自动联动步只动温度、流量保持为零，目标仍在，后续周期按限速继续逼近。
    accepted = source.joint_step(1.0, 60.0)
    assert accepted["ok"] is True
    assert accepted["temperature_c"] == pytest.approx(35.35)
    assert accepted["flow_t_h"] == pytest.approx(0.0)
    assert source.target_c() == pytest.approx(60.0)
    assert source.target_flow_t_h() == pytest.approx(2000.0)


def test_concurrent_adjust_never_tears_the_pair():
    source = _source()
    pairs = ((95.0, 2000.0), (85.0, 3000.0))
    seed = source.snapshot()
    allowed = set(pairs) | {(seed.temperature_c, seed.flow_t_h)}
    observed: list[tuple[float, float]] = []
    stop = threading.Event()

    def writer(temperature_c: float, flow_t_h: float) -> None:
        for _ in range(300):
            source.adjust(temperature_c, flow_t_h)

    def sampler() -> None:
        while not stop.is_set():
            snapshot = source.snapshot()
            observed.append((snapshot.temperature_c, snapshot.flow_t_h))

    watcher = threading.Thread(target=sampler)
    watcher.start()
    workers = [threading.Thread(target=writer, args=pair) for pair in pairs]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    stop.set()
    watcher.join()

    mixed = [pair for pair in observed if pair not in allowed]
    assert mixed == []


def test_quality_must_be_confirmed_before_flow_ignites():
    source = _source()
    source.fire(90.0)
    # 固定温差带内保持不足确认时间，流量不许点火（允许矩形上界为零）。
    early = source.rate_limit_for(1.0, 60.0)
    assert early.flow_upper_t_h == pytest.approx(0.0)
    for _ in range(400):
        source.joint_step(1.0, 60.0)
        if source.quality_confirmed():
            break
    assert source.quality_confirmed() is True
    # 确认之后联动步一步点火到最小点火流量。
    source.set_flow_target(1500.0)
    ignited = source.joint_step(1.0, 60.0)
    assert ignited["ok"] is True
    assert ignited["flow_t_h"] == pytest.approx(source.limits.ignite_flow_t_h)
    # 继续联动爬坡到目标流量。
    for _ in range(400):
        source.joint_step(1.0, 60.0)
        if abs(source.flow_t_h() - 1500.0) < 1e-9:
            break
    assert source.flow_t_h() == pytest.approx(1500.0)
    assert source.state() == "online"


def test_joint_rate_band_is_fixed_and_yield_switches_by_load():
    limits = _source().limits
    # 低负荷：温度领先、流量让步；高负荷：流量领先、温度让步。
    low = joint_rate_limit(limits, 80.0, 2000.0, 90.0, 2600.0, 1.0, 60.0)
    assert low.band_kind == BAND_KIND
    assert low.load_segment == LOW_LOAD
    assert low.yield_to == TEMP_FIRST
    assert low.leader == FLOW_FIRST
    high = joint_rate_limit(limits, 80.0, 2000.0, 90.0, 2600.0, 1.0, 90.0)
    assert high.load_segment == HIGH_LOAD
    assert high.yield_to == FLOW_FIRST
    assert high.leader == TEMP_FIRST


def test_temp_gap_freezes_flow_then_releases_inside_band():
    limits = _source().limits
    # 温差在固定带外（>=6℃）流量跟随侧冻结。
    far = joint_rate_limit(limits, 80.0, 2000.0, 90.0, 2600.0, 1.0, 60.0)
    assert far.flow_upper_t_h == pytest.approx(2000.0)
    # 温差进带，流量跟随侧按 1 - 剩余温差/带宽 放开。
    near = joint_rate_limit(limits, 87.0, 2000.0, 90.0, 2600.0, 1.0, 60.0)
    gain = 1.0 - 3.0 / limits.joint_temp_band_c
    assert near.flow_upper_t_h == pytest.approx(2000.0 + limits.adjust_rate_t_h_s * gain)
    # 温差归零拿到全速。
    caught = joint_rate_limit(limits, 90.0, 2000.0, 90.0, 2600.0, 1.0, 60.0)
    assert caught.flow_upper_t_h == pytest.approx(2000.0 + limits.adjust_rate_t_h_s)


def test_cooling_withdraws_flow_at_full_speed_and_blocks_retreat():
    limits = _source().limits
    # 温度下行（温降）：流量回撤全速放开，逆行侧（升流量）封死。
    cooling = joint_rate_limit(limits, 90.0, 2600.0, 80.0, 2000.0, 1.0, 60.0)
    assert cooling.flow_upper_t_h == pytest.approx(2600.0)
    assert cooling.flow_lower_t_h == pytest.approx(2600.0 - limits.adjust_rate_t_h_s)
    assert cooling.allows(89.65, 2545.0) is True
    assert cooling.allows(89.65, 2601.0) is False


def test_readback_missing_is_partial_failure_down_only():
    limits = _source().limits
    limit = joint_rate_limit(
        limits, 88.0, 2000.0, 90.0, 2600.0, 1.0, 60.0, readback_missing=True
    )
    assert limit.degraded is True
    # 温度、流量都只许降不许升；流量可一路撤到零。
    assert limit.temp_upper_c == pytest.approx(88.0)
    assert limit.flow_upper_t_h == pytest.approx(2000.0)
    assert limit.flow_lower_t_h == pytest.approx(0.0)
    assert limit.allows(87.65, 1500.0) is True
    assert limit.allows(88.0, 2001.0) is False
    assert limit.allows(88.01, 1500.0) is False


def test_high_load_flow_leads_and_temp_gap_freezes_temp():
    limits = _source().limits
    # 高负荷流量领先：流量不动（流差为零）时温度拿全速（质调节）。
    quality_only = joint_rate_limit(limits, 80.0, 2000.0, 90.0, 2000.0, 1.0, 90.0)
    assert quality_only.temp_upper_c == pytest.approx(80.0 + limits.adjust_rate_c_s)
    # 流差在固定流差带外，温度跟随侧冻结。
    far = joint_rate_limit(limits, 80.0, 2000.0, 90.0, 3000.0, 1.0, 90.0)
    assert far.temp_upper_c == pytest.approx(80.0)


def test_weather_push_updates_curve_and_fires_source():
    source = _source()
    curve = _curve()
    report = push_weather(source, curve, WeatherPush(-12.0, 1000.0), 1000.0)
    assert report["ok"] is True
    assert report["epoch"] == 1
    assert curve.outdoor_c() == pytest.approx(-12.0)
    assert source.target_c() == pytest.approx(curve.supply_c())
    assert [step["action"] for step in report["steps"]] == ["apply_forecast", "fire_forecast"]


def test_out_of_range_forecast_is_rejected_without_touching_curve():
    source = _source()
    curve = _curve()
    push_weather(source, curve, WeatherPush(-8.0, 1000.0), 1000.0)
    before_epoch = curve.epoch()
    before_supply = curve.supply_c()

    with pytest.raises(ForecastError):
        curve.validate(60.0)
    assert curve.rejected() == 1

    report = push_weather(source, curve, WeatherPush(60.0, 1100.0), 1100.0)
    assert report["ok"] is False
    assert curve.epoch() == before_epoch
    assert curve.supply_c() == pytest.approx(before_supply)
    assert "超出曲线适用区间" in report["violations"][0]


def test_trip_and_reset():
    source = _source()
    source.trip("overtemperature")
    assert source.state() == "tripped"
    assert source.trips() == 1
    with pytest.raises(InterlockError):
        source.adjust(90.0, 1200.0)
    with pytest.raises(InterlockError):
        source.fire(90.0)
    assert source.reset() == "idle"
    assert source.trip_reason() == ""
    with pytest.raises(InterlockError):
        source.reset()
