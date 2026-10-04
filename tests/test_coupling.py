"""调质调量联动限速：固定温差带、负荷段让步、回读降级与时序。"""

from __future__ import annotations

import pytest

from heatnet.app import build_platform
from heatnet.clock import ManualClock
from heatnet.config import ConfigError, config_to_mapping, default_config
from heatnet.source.coupling import (
    BAND_MODE,
    HIGH_LOAD,
    LOW_LOAD,
    MID_LOAD,
    CouplingLimits,
    YIELD_FLOW,
    YIELD_NONE,
    YIELD_TEMP,
    evaluate,
    load_segment,
)


def _limits() -> CouplingLimits:
    return CouplingLimits(
        flow_per_c=40.0,
        band_c=3.0,
        temp_rate_c_s=0.35,
        flow_rate_t_h_s=30.0,
        low_load_pct=40.0,
        high_load_pct=75.0,
    )


def _eval(current=(80.0, 1200.0), proposed=(82.0, 1280.0), baseline=2400.0,
          origin=None, readback_ok=True, dt=None):
    origin = origin or current
    return evaluate(
        _limits(),
        origin[0], origin[1], current[0], current[1],
        proposed[0], proposed[1], baseline,
        readback_ok=readback_ok, dt=dt,
    )


def test_band_mode_is_fixed_and_exposed():
    assert BAND_MODE == "fixed"


def test_load_segments_by_flow_pct():
    limits = _limits()
    assert load_segment(800.0, 2400.0, limits) == LOW_LOAD
    assert load_segment(1200.0, 2400.0, limits) == MID_LOAD
    assert load_segment(1800.0, 2400.0, limits) == HIGH_LOAD


def test_paired_move_on_the_coupling_line_is_allowed():
    decision = _eval(current=(80.0, 1200.0), proposed=(82.0, 1280.0))
    assert decision.allowed is True
    assert decision.drift_c == pytest.approx(0.0)
    assert decision.yields == YIELD_NONE
    assert decision.violations == []


def test_temperature_runs_ahead_in_mid_load_is_rejected():
    # 温度 +4C 而流量不动 => drift=4C > band 3C
    decision = _eval(current=(80.0, 1200.0), proposed=(84.0, 1200.0))
    assert decision.allowed is False
    assert "coupling_band_exceeded_temp_lead" in decision.violations


def test_flow_pulled_ahead_in_mid_load_is_rejected():
    # 温度不动，流量 +300 => drift=-7.5C < -3C
    decision = _eval(current=(80.0, 1200.0), proposed=(80.0, 1500.0))
    assert decision.allowed is False
    assert "coupling_band_exceeded_flow_lead" in decision.violations


def test_inside_band_boundary_is_allowed():
    # drift 恰好 ±3C 落在带沿
    assert _eval(proposed=(83.0, 1200.0)).allowed is True
    assert _eval(proposed=(80.0, 1320.0)).allowed is True


def test_low_load_flow_yields_segment():
    # 低负荷段：流量让步（温度领步），但固定带两侧仍然硬拦。
    current = (80.0, 900.0)
    decision = _eval(current=current, proposed=(86.0, 900.0))
    assert decision.load_segment == LOW_LOAD
    assert decision.yields == YIELD_FLOW
    # 温度超前 6C > 3C 带：让步不豁免带，仍拒绝。
    assert decision.allowed is False
    assert "coupling_band_exceeded_temp_lead" in decision.violations
    # 带内（2C）放行。
    assert _eval(current=current, proposed=(82.0, 900.0)).allowed is True
    # 流量超前同样拒绝。
    lead_flow = _eval(current=current, proposed=(80.0, 1100.0))
    assert lead_flow.allowed is False
    assert "coupling_band_exceeded_flow_lead" in lead_flow.violations


def test_high_load_temperature_yields_segment():
    # 高负荷段（流量 >= 75% 基线）：温度让步（流量领步），固定带两侧仍硬拦。
    current = (80.0, 2000.0)  # 2000/2400 = 83% => 高负荷
    decision = _eval(current=current, proposed=(80.0, 2200.0))
    assert decision.load_segment == HIGH_LOAD
    assert decision.yields == YIELD_TEMP
    # 流量超前 5C > 3C 带：让步不豁免带，仍拒绝。
    assert decision.allowed is False
    assert "coupling_band_exceeded_flow_lead" in decision.violations
    # 带内（+120 t/h = 3C）放行。
    assert _eval(current=current, proposed=(80.0, 2120.0)).allowed is True
    # 温度掉 4C 而流量不退（末端过热）：同一侧硬拦。
    cooling = _eval(current=current, proposed=(76.0, 2000.0))
    assert cooling.allowed is False
    assert "coupling_band_exceeded_flow_lead" in cooling.violations


def test_opposed_move_is_rejected():
    decision = _eval(current=(80.0, 1200.0), proposed=(82.0, 1100.0))
    assert decision.allowed is False
    assert "opposed_move" in decision.violations


def test_rate_limits_only_apply_with_timestep():
    # 不给 dt：再大的整步也不受速率约束（只受固定带约束）。
    no_dt = _eval(current=(80.0, 1200.0), proposed=(80.5, 1220.0), dt=None)
    assert no_dt.temp_rate_limit_c_s is None
    assert no_dt.allowed is True
    # 给 dt=1：温度步 2C > 0.35C/s 限速
    stepped = _eval(current=(80.0, 1200.0), proposed=(82.0, 1280.0), dt=1.0)
    assert stepped.allowed is False
    assert "temp_rate_exceeded" in stepped.violations
    # 流量步 100 t/h > 30 t/h/s
    stepped_flow = _eval(current=(80.0, 1200.0), proposed=(80.3, 1300.0), dt=1.0)
    assert "flow_rate_exceeded" in stepped_flow.violations


def test_readback_missing_is_down_only():
    # 流量上升被拒
    up = _eval(current=(80.0, 1200.0), proposed=(81.0, 1240.0), readback_ok=False)
    assert up.allowed is False
    assert "flow_readback_missing_down_only" in up.violations
    assert up.readback_ok is False
    # 流量下降允许（即便温度同时下降）
    down = _eval(current=(80.0, 1200.0), proposed=(79.0, 1160.0), readback_ok=False)
    assert down.allowed is True
    # 纯温度动作也允许
    assert _eval(current=(80.0, 1200.0), proposed=(79.0, 1200.0), readback_ok=False).allowed is True


def test_origin_anchors_cumulative_drift():
    # 单步都在带内，但相对调作起点累计漂移超带也要拒绝。
    origin = (80.0, 1200.0)
    current = (82.0, 1200.0)  # 已经先爬了 2C
    decision = _eval(origin=origin, current=current, proposed=(83.5, 1200.0))
    assert decision.allowed is False
    assert "coupling_band_exceeded_temp_lead" in decision.violations


# ---- 平台集成 ---------------------------------------------------------

def _platform(tmp_path):
    return build_platform(default_config(), clock=ManualClock(), storage_root=str(tmp_path / "state"))


def test_platform_rejects_out_of_band_pair_atomically(tmp_path):
    platform = _platform(tmp_path)
    platform.start()
    platform.push_weather(-8.0)
    platform.startup(-8.0, 60.0)
    temp = platform.source.temperature_c()
    flow = platform.source.flow_t_h()

    rejected = platform.adjust_source(temp, flow + 300.0, "console")
    assert rejected["ok"] is False
    assert rejected["violations"]
    # 温度与流量一个都没写
    assert platform.source.temperature_c() == pytest.approx(temp)
    assert platform.source.flow_t_h() == pytest.approx(flow)
    # 一次网没有跟随到被拒值
    assert platform.network.flow_t_h() != pytest.approx(flow + 300.0)
    # 拒绝结论同时挂在返回值、控制台快照与审计里
    assert rejected["coupling"]["allowed"] is False
    assert rejected["coupling"]["band_mode"] == "fixed"
    status = platform.status()
    assert status["coupling"]["band_mode"] == "fixed"
    assert status["coupling"]["last_decision"]["violations"] == rejected["violations"]
    guards = [e for e in platform.audit.entries(20) if e["action"] == "coupling_guard"]
    assert guards
    assert guards[0]["payload"]["allowed"] is False
    assert guards[0]["severity"] == "alarm"


def test_platform_allows_in_band_pair_and_follows_primary(tmp_path):
    platform = _platform(tmp_path)
    platform.start()
    platform.push_weather(-8.0)
    platform.startup(-8.0, 60.0)
    temp = platform.source.temperature_c()
    flow = platform.source.flow_t_h()

    accepted = platform.adjust_source(temp + 2.0, flow + 80.0, "console")
    assert accepted["ok"] is True
    assert platform.source.temperature_c() == pytest.approx(temp + 2.0)
    assert platform.source.flow_t_h() == pytest.approx(flow + 80.0)
    # 时序：跟随发生在放行之后
    assert platform.network.supply_c() == pytest.approx(temp + 2.0)
    assert platform.network.flow_t_h() == pytest.approx(flow + 80.0)


def test_readback_missing_down_only_through_platform(tmp_path):
    platform = _platform(tmp_path)
    platform.start()
    platform.push_weather(-8.0)
    platform.startup(-8.0, 60.0)
    temp = platform.source.temperature_c()
    flow = platform.source.flow_t_h()

    platform.flow_readback(False)
    assert platform.coupling_report()["down_only"] is True

    up = platform.adjust_source(temp + 1.0, flow + 40.0, "console")
    assert up["ok"] is False
    assert "flow_readback_missing_down_only" in up["violations"]
    assert platform.source.flow_t_h() == pytest.approx(flow)

    down = platform.adjust_source(temp - 1.0, flow - 40.0, "console")
    assert down["ok"] is True
    assert platform.source.flow_t_h() == pytest.approx(flow - 40.0)

    platform.flow_readback(True)
    again = platform.adjust_source(temp - 1.0, flow, "console")
    assert again["ok"] is True


def test_cycle_carries_coupling_decision_and_audit(tmp_path):
    platform = _platform(tmp_path)
    platform.start()
    platform.push_weather(-8.0)
    platform.startup(-8.0, 60.0)
    report = platform.cycle(dt=1.0)
    assert report["coupling"]["band_mode"] == "fixed"
    assert report["coupling"]["yields"] in (YIELD_FLOW, YIELD_TEMP, YIELD_NONE)
    guards = [e for e in platform.audit.entries(10) if e["action"] == "coupling_guard"]
    assert any(e["payload"].get("origin") == "cycle" for e in guards)


def test_coupling_endpoint_and_readback_endpoint(tmp_path):
    from heatnet.console.api import Api

    platform = _platform(tmp_path)
    api = Api(platform)
    status, payload = api.handle("GET", "/api/coupling", {}, {})
    assert status == 200
    assert payload["band_mode"] == "fixed"
    assert payload["readback_ok"] is True

    status, payload = api.handle("POST", "/api/readback", {}, {"ok": False})
    assert status == 200
    assert payload["down_only"] is True


def test_config_carries_fixed_coupling_fields():
    config = default_config()
    assert config.source.coupling_band_c > 0
    restored = type(config).from_mapping(config_to_mapping(config))
    assert restored == config


def test_broken_load_segment_thresholds_rejected():
    mapping = config_to_mapping(default_config())
    mapping["limits"]["source"]["low_load_pct"] = 80.0
    mapping["limits"]["source"]["high_load_pct"] = 50.0
    with pytest.raises(ConfigError):
        type(default_config()).from_mapping(mapping)


# ---- 控制周期联动 ------------------------------------------------------

def test_cycle_coramps_flow_with_temperature_when_curve_cools(tmp_path):
    platform = _platform(tmp_path)
    platform.start()
    platform.push_weather(-5.0)
    platform.startup(-5.0, 60.0)
    temp0 = platform.source.temperature_c()
    flow0 = platform.source.flow_t_h()

    platform.push_weather(-24.0)  # 更冷 => 供温目标升高
    target = platform.curve.supply_c()
    per_c = default_config().source.coupling_flow_per_c
    expected_flow = flow0 + per_c * (target - temp0)

    for _ in range(1500):
        report = platform.cycle(dt=1.0)
        if abs(platform.source.temperature_c() - target) < 0.01:
            break
        assert report["ok"] is True, report.get("violations")

    assert platform.source.temperature_c() == pytest.approx(target, abs=0.02)
    # 流量是跟着温度一起上来的，不是温度先到位流量再拉。
    assert platform.source.flow_t_h() == pytest.approx(expected_flow, abs=2.0)
    assert platform.network.flow_t_h() == pytest.approx(platform.source.flow_t_h(), abs=1e-6)


def test_cycle_retreats_flow_when_temperature_drops(tmp_path):
    platform = _platform(tmp_path)
    platform.start()
    platform.push_weather(-5.0)
    platform.startup(-5.0, 60.0)
    platform.push_weather(-24.0)
    for _ in range(1500):
        platform.cycle(dt=1.0)
        if abs(platform.source.temperature_c() - platform.curve.supply_c()) < 0.01:
            break

    platform.push_weather(2.0)  # 转暖 => 供温目标降低，流量必须跟着退
    warm_target = platform.curve.supply_c()
    for _ in range(1500):
        report = platform.cycle(dt=1.0)
        if abs(platform.source.temperature_c() - warm_target) < 0.01:
            break
        assert report["ok"] is True, report.get("violations")

    # 温度掉下来时流量已同步退走，末端不会继续过热。
    assert platform.source.temperature_c() == pytest.approx(warm_target, abs=0.02)
    assert platform.source.flow_t_h() < default_config().source.flow_max_t_h
    drift = report["coupling"]["drift_c"]
    assert abs(drift) <= default_config().source.coupling_band_c + 0.01


def test_cycle_never_raises_flow_when_readback_missing(tmp_path):
    platform = _platform(tmp_path)
    platform.start()
    platform.push_weather(-5.0)
    platform.startup(-5.0, 60.0)
    platform.push_weather(-24.0)
    platform.flow_readback(False)
    flow_start = platform.source.flow_t_h()

    for _ in range(300):
        platform.cycle(dt=1.0)
        assert platform.source.flow_t_h() <= flow_start + 1e-9

    # 转暖退流在降级态下仍允许。
    platform.flow_readback(True)
    platform.push_weather(2.0)
    for _ in range(3):
        platform.cycle(dt=1.0)
    flow_before = platform.source.flow_t_h()
    platform.flow_readback(False)
    for _ in range(500):
        platform.cycle(dt=1.0)
        if abs(platform.source.temperature_c() - platform.curve.supply_c()) < 0.02:
            break
    assert platform.source.flow_t_h() < flow_before


def test_primary_network_does_not_follow_a_rejected_cycle(tmp_path):
    platform = _platform(tmp_path)
    platform.start()
    platform.push_weather(-5.0)
    platform.startup(-5.0, 60.0)
    # 回读缺失 + 要求升温：控制器不能靠抬流收敛，温度可走，一次网流量不得上升。
    platform.push_weather(-24.0)
    platform.flow_readback(False)
    net_flow = platform.network.flow_t_h()
    for _ in range(50):
        report = platform.cycle(dt=1.0)
        assert platform.network.flow_t_h() <= net_flow + 1e-9
    assert report["coupling"]["readback_ok"] is False
