"""调质调量联合限速：固定带、负荷段让步、回读降级、时序与审计一致性。"""

from __future__ import annotations

import pytest

from heatnet.app import build_platform
from heatnet.config import default_config
from heatnet.console.api import Api
from heatnet.source.joint import FLOW_FIRST, TEMP_FIRST, joint_rate_limit, load_segment


def _limits():
    return default_config().source


def test_band_kind_is_pinned_to_fixed():
    limit = joint_rate_limit(_limits(), 80.0, 2000.0, 90.0, 2600.0, 1.0, 60.0)
    assert limit.band_kind == "fixed"
    assert load_segment(74.9, 75.0) == "low"
    assert load_segment(75.0, 75.0) == "high"


def test_low_load_temperature_leads_high_load_flow_leads():
    low = joint_rate_limit(_limits(), 80.0, 2000.0, 90.0, 3000.0, 1.0, 60.0)
    high = joint_rate_limit(_limits(), 80.0, 2000.0, 90.0, 3000.0, 1.0, 90.0)
    assert low.yield_to == TEMP_FIRST
    assert high.yield_to == FLOW_FIRST
    # 低负荷：温差带外流量上升侧冻结；高负荷：流差带外温度上升侧冻结。
    assert low.flow_upper_t_h == pytest.approx(2000.0)
    assert high.temp_upper_c == pytest.approx(80.0)


def test_cold_ignition_always_temperature_first_even_at_high_load():
    limit = joint_rate_limit(_limits(), 35.0, 0.0, 90.0, 3000.0, 1.0, 90.0)
    assert limit.yield_to == TEMP_FIRST
    assert limit.flow_upper_t_h == pytest.approx(0.0)


def test_startup_runs_joint_ramp_and_reaches_both_targets(platform):
    platform.start()
    platform.push_weather(-12.0)
    report = platform.startup(-12.0, 60.0)
    assert report["ok"] is True
    assert report["stage"] == "done"
    # 联动到位后一次网跟随同一对温度/流量，没有撕裂。
    assert platform.network.supply_c() == pytest.approx(platform.source.temperature_c())
    assert platform.network.flow_t_h() == pytest.approx(platform.source.flow_t_h())
    # 爬坡过程审计是 joint_ramp，不是旧的两段式动作。
    actions = [step["action"] for step in report["steps"]]
    assert "joint_ramp" in actions
    assert "settle_quality" not in actions
    follow_at = actions.index("follow_source")
    assert follow_at > actions.index("joint_ramp")


def test_primary_follow_happens_in_same_cycle_as_source_move(online):
    before_t = online.source.temperature_c()
    online.cycle(dt=1.0)
    # 源与一次网在同一周期取到同一对值（时序：先联动、后跟随）。
    assert online.network.supply_c() == pytest.approx(online.source.temperature_c())
    assert online.network.flow_t_h() == pytest.approx(online.source.flow_t_h())
    assert online.source.temperature_c() >= before_t - 1e-9


def test_rate_limit_audit_matches_console_and_status(online):
    online.cycle(dt=1.0)
    audit = [e for e in online.bus.history("source") if e.action == "joint_rate_limit"][-1]
    status_band = online.status()["source"]["rate_limit"]
    assert audit.payload["band_kind"] == status_band["band_kind"] == "fixed"
    assert audit.payload["yield_to"] == status_band["yield_to"]
    assert audit.payload["allowed_band"] == status_band["allowed_band"]
    api = Api(online)
    _, preview = api.handle("GET", "/api/rate-limit", {"dt_s": "1"}, {})
    assert preview["allowed_band"] == online.source.snapshot().rate_limit["allowed_band"]


def test_readback_loss_is_partial_failure_down_only(online):
    api = Api(online)
    status, lost = api.handle("POST", "/api/flow-readback", {}, {"present": False})
    assert status == 200 and lost["readback_missing"] is True
    assert any(e.action == "flow_readback_lost" for e in online.bus.history("primary"))
    temp = online.source.temperature_c()
    flow = online.source.flow_t_h()
    # 升流量、升温都被拒绝。
    up = online.adjust_source(temp + 2.0, flow + 100.0)
    assert up["ok"] is False and "flow_readback_missing_down_only" in up["violations"]
    assert online.source.temperature_c() == pytest.approx(temp)
    assert online.source.flow_t_h() == pytest.approx(flow)
    # 降流量放行。
    down = online.adjust_source(temp, flow - 55.0)
    assert down["ok"] is True and down["flow_t_h"] == pytest.approx(flow - 55.0)
    # 降级期间控制周期也只许降不许升：保持不升即 ok。
    cycle_report = online.cycle(dt=1.0)
    assert cycle_report["degraded"] is True
    # readiness 反映回读缺失。
    _, ready = api.handle("GET", "/api/readiness", {}, {})
    assert ready["checks"]["flow_readback_live"] is False
    # 回读恢复。
    status, restored = api.handle(
        "POST", "/api/flow-readback", {}, {"present": True, "measured_t_h": down["flow_t_h"]}
    )
    assert status == 200 and restored["readback_missing"] is False
    assert any(e.action == "flow_readback_restored" for e in online.bus.history("primary"))


def test_high_load_startup_yields_temperature_to_flow(platform):
    platform.start()
    platform.push_weather(-20.0)
    report = platform.startup(-20.0, 90.0)
    assert report["ok"] is True
    # 高负荷段最终裁决为流量领先（温度让步给流量）。
    rate = platform.status()["source"]["rate_limit"]
    assert rate["load_segment"] == "high"
    assert rate["yield_to"] == FLOW_FIRST
    expected_flow = platform.network.baseline_t_h() * 0.9
    assert report["flow_t_h"] == pytest.approx(expected_flow)


def test_rate_limit_endpoint_exposes_fixed_band(online):
    api = Api(online)
    status, payload = api.handle("GET", "/api/rate-limit", {}, {})
    assert status == 200
    assert payload["band_kind"] == "fixed"
    for key in ("load_segment", "yield_to", "leader", "allowed_band", "targets", "gain"):
        assert key in payload


def test_status_exposes_fixed_band_config(online):
    band = online.status()["source"]["joint_band"]
    assert band == {
        "band_kind": "fixed",
        "temp_band_c": default_config().source.joint_temp_band_c,
        "flow_band_t_h": default_config().source.joint_flow_band_t_h,
        "ignite_flow_t_h": default_config().source.ignite_flow_t_h,
        "high_load_pct": default_config().source.high_load_pct,
    }


def test_flow_reduction_inside_trusted_range_is_not_clamped_back(platform):
    platform.start()
    platform.push_weather(-20.0)
    platform.startup(-20.0, 60.0)
    flow = platform.source.flow_t_h()
    report = platform.adjust_source(platform.source.temperature_c(), flow - 55.0)
    assert report["ok"] is True
    assert report["applied"] is True
    assert report["flow_t_h"] == pytest.approx(flow - 55.0)


def test_zero_startup_ignites_straight_to_minimum_flow(platform):
    limits = default_config().source
    platform.start()
    platform.push_weather(-12.0)
    # 点火前固定温差带内保持确认。
    for _ in range(200):
        platform.source.joint_step(1.0, 60.0)
        if platform.source.quality_confirmed():
            break
    platform.source.set_flow_target(2000.0)
    move = platform.source.joint_step(1.0, 60.0)
    assert move["ok"] is True
    assert move["flow_t_h"] == pytest.approx(limits.ignite_flow_t_h)
    assert limits.flow_min_t_h <= move["flow_t_h"] <= limits.flow_max_t_h


def test_degraded_auto_cycle_never_raises(platform):
    platform.start()
    platform.push_weather(-20.0)
    platform.startup(-20.0, 60.0)
    flow0 = platform.source.flow_t_h()
    temp0 = platform.source.temperature_c()
    platform.report_flow_readback(False)
    for _ in range(5):
        report = platform.cycle(dt=1.0)
        assert platform.source.flow_t_h() <= flow0 + 1e-9
        assert platform.source.temperature_c() <= temp0 + 1e-9
        assert report["degraded"] is True
