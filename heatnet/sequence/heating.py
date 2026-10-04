"""控制周期编排。

一个控制步的时序定死为：先看曲线是否还在新鲜度窗口内，再让热源按固定联合
限速带动调质调量（温度与流量在同一个允许矩形里联动），随后一次网跟随热源
（源已动而网未跟随的撕裂态不允许流出本函数），最后二次网按跟随上来的流量
推进回水与补水。回读缺失属于部分失败：联动步进入只许降不许升的降级模式，
一次网跟随照做（只是不升），平台据此展示与落审计。任何一环越界都记进违约项
返回，由平台决定是否停机。
"""

from __future__ import annotations

from ..primary.network import PrimaryNetwork
from ..secondary.loop import SecondaryLoop
from ..source.source import HeatSource
from ..temp.curve import HeatCurve

MAX_STEPS = 600


def advance_heating(
    source: HeatSource,
    curve: HeatCurve,
    network: PrimaryNetwork,
    loop: SecondaryLoop,
    demand_pct: float,
    now: float,
    dt: float = 1.0,
) -> dict:
    """推进一个控制周期，返回本周期摘要。"""

    if dt <= 0:
        raise ValueError("步长必须为正")
    steps: list[dict] = []
    violations: list[str] = []

    if curve.is_stale(now):
        violations.append("curve_stale")
        steps.append(
            {"component": "temp", "action": "stale", "detail": {"updated_at": round(curve.updated_at(), 3)}}
        )
        return {"ok": False, "steps": steps, "violations": violations, "stage": "curve"}

    target = curve.supply_c()
    source.set_target(target)
    readback_missing = network.readback_missing()
    move = source.joint_step(dt, demand_pct, readback_missing=readback_missing)
    steps.append({"component": "source", "action": "joint_adjust", "detail": move["rate_limit"]})
    violations.extend(move["violations"])
    if violations:
        return {"ok": False, "steps": steps, "violations": violations, "stage": "source", "rate_limit": move["rate_limit"]}

    if source.limits.overtemperature(source.temperature_c()):
        source.trip("overtemperature")
        violations.append("overtemperature")
        return {"ok": False, "steps": steps, "violations": violations, "stage": "source"}

    # 调质调量与一次网跟随的时序不能乱：联动步写完，一次网立刻跟随同一对值。
    network.set_supply(source.temperature_c())
    network.set_return(loop.return_c())
    network.set_flow(source.flow_t_h())
    steps.append(
        {"component": "primary", "action": "follow_source", "detail": network.snapshot().as_dict()}
    )

    flow = network.flow_t_h()
    balance = loop.step(dt, flow)
    steps.append({"component": "secondary", "action": "step", "detail": balance})
    if balance["overheat"]:
        violations.append("return_overtemperature")
    if balance["needs_makeup"]:
        violations.append("makeup_required")

    for _ in range(MAX_STEPS):
        if not loop.needs_makeup():
            break
        loop.makeup_step(dt)
    steps.append(
        {
            "component": "secondary",
            "action": "makeup",
            "detail": {"pressure_kpa": round(loop.pressure_kpa(), 2)},
        }
    )

    return {
        "ok": not violations,
        "steps": steps,
        "violations": violations,
        "stage": "done",
        "temperature_c": round(source.temperature_c(), 2),
        "flow_t_h": round(source.flow_t_h(), 2),
        "degraded": move.get("degraded", False),
        "rate_limit": move["rate_limit"],
        "return_c": round(loop.return_c(), 2),
    }
