"""投运编排。

气象推送必须先更新曲线，热源再按新曲线点火；随后调质调量按固定联合限速带
联动爬坡——供温先进固定温差带并保持确认，流量才允许一步点火，之后两轴在
同一允许矩形里一起逼近目标；联动到位之后一次网才跟随。随后按"先补后定"把
压力拉回目标带，最后跑一个循环建立流量。任何一步失败都停在原地，不带着
半截状态继续往下走。
"""

from __future__ import annotations

from ..errors import PlantError
from ..primary.network import PrimaryNetwork
from ..pump.pump import Pump
from ..secondary.cycle import run_cycle
from ..secondary.loop import SecondaryLoop
from ..secondary.makeup import makeup_then_pressurise
from ..source.source import HeatSource
from ..source.weather import WeatherPush, push_weather
from ..store.state import DurableState
from ..temp.curve import HeatCurve
from ..valve.valve import Valve

MAX_STEPS = 1200


def startup_plant(
    source: HeatSource,
    curve: HeatCurve,
    network: PrimaryNetwork,
    loop: SecondaryLoop,
    pump: Pump,
    valve: Valve,
    durable: DurableState,
    outdoor_c: float,
    now: float,
    demand_pct: float = 60.0,
    dt: float = 1.0,
) -> dict:
    """把整条调度链从冷态带到带负荷。"""

    steps: list[dict] = []
    violations: list[str] = []

    weather = push_weather(source, curve, WeatherPush(outdoor_c, now), now)
    steps.extend(weather["steps"])
    violations.extend(weather["violations"])
    if not weather["ok"]:
        return {"ok": False, "steps": steps, "violations": violations, "stage": "weather"}

    demand = max(0.0, min(1.0, demand_pct / 100.0))
    flow_target = network.baseline_t_h() * demand
    try:
        source.set_flow_target(flow_target)
    except PlantError as error:
        violations.append(error.message)
        return {"ok": False, "steps": steps, "violations": violations, "stage": "quantity"}

    rate_limit = None
    for _ in range(MAX_STEPS):
        move = source.joint_step(dt, demand_pct, readback_missing=network.readback_missing())
        rate_limit = move["rate_limit"]
        violations.extend(move["violations"])
        if violations:
            steps.append({"component": "source", "action": "joint_ramp", "detail": rate_limit})
            return {"ok": False, "steps": steps, "violations": violations, "stage": "source"}
        if (
            abs(source.temperature_c() - source.target_c()) <= 1e-6
            and abs(source.flow_t_h() - flow_target) <= 1e-6
        ):
            break
    steps.append(
        {
            "component": "source",
            "action": "joint_ramp",
            "detail": {
                "temperature_c": round(source.temperature_c(), 2),
                "flow_t_h": round(source.flow_t_h(), 2),
                "hold_s": round(source.quality_hold_s(), 3),
                "confirmed": source.quality_confirmed(),
                "rate_limit": rate_limit,
            },
        }
    )
    if not source.quality_confirmed():
        violations.append("quality_not_confirmed")
        return {"ok": False, "steps": steps, "violations": violations, "stage": "quality"}
    if abs(source.flow_t_h() - flow_target) > 1e-6:
        violations.append("quantity_not_reached")
        return {"ok": False, "steps": steps, "violations": violations, "stage": "quantity"}

    # 联动到位之后一次网才跟随，顺序不能反过来。
    network.set_supply(source.temperature_c())
    network.set_return(loop.return_c())
    network.set_flow(source.flow_t_h())
    steps.append(
        {
            "component": "primary",
            "action": "follow_source",
            "detail": {"flow_t_h": round(source.flow_t_h(), 2), "supply_c": round(source.temperature_c(), 2)},
        }
    )

    makeup = makeup_then_pressurise(loop, dt)
    steps.extend(makeup["steps"])
    violations.extend(makeup["violations"])
    if not makeup["ok"]:
        return {"ok": False, "steps": steps, "violations": violations, "stage": "makeup"}

    cycle = run_cycle(loop, pump, valve, durable, dt)
    steps.extend(cycle["steps"])
    violations.extend(cycle["violations"])
    if not cycle["ok"]:
        return {"ok": False, "steps": steps, "violations": violations, "stage": "cycle"}

    return {
        "ok": not violations,
        "steps": steps,
        "violations": violations,
        "stage": "done",
        "flow_t_h": round(source.flow_t_h(), 2),
        "supply_c": round(source.temperature_c(), 2),
        "return_c": round(loop.return_c(), 2),
        "rate_limit": rate_limit,
    }
