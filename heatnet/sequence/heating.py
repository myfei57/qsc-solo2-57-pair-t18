"""控制周期编排。

一个控制步里：先看曲线是否还在新鲜度窗口内，再按"温差 + 流量"过联动限速闸门，
把热源温度与流量作为一对一起推进；闸门拒绝时整步不写、一次网**不跟随**（调质
调量与一次网跟随的时序不能乱：闸门在前，跟随在后）。随后二次网按当前流量推进
回水。任何一环越界都记进违约项返回，由平台决定是否停机。
"""

from __future__ import annotations

from ..errors import PlantError
from ..primary.network import PrimaryNetwork
from ..secondary.loop import SecondaryLoop
from ..source.coupling import CouplingLimits, evaluate
from ..source.source import HeatSource
from ..temp.curve import HeatCurve

MAX_STEPS = 600


def propose_coupled_step(
    source: HeatSource,
    target_c: float,
    target_flow_t_h: float,
    baseline_t_h: float,
    readback_ok: bool,
    dt: float,
):
    """按各自速率与固定带算出本步拟议（温度, 流量），并给出联动裁决。

    控制器先让温度、流量各自按自身速率向配对目标取最大推进，再用固定带把越界
    的拟议收回带内：落后方在本步速率内能追多少追多少，仍追不上就把领先方收
    到带沿——因此每一步总有人在向目标推进，不会原地踏步。让步方向（负荷段）
    决定追不上时优先收谁；回读缺失时流量只能降，抬流追不上就一律收温度。
    """

    limits_cfg = CouplingLimits.from_source(source.limits)
    current_t = source.temperature_c()
    current_f = source.flow_t_h()
    origin_t, origin_f = source.origin()
    per_c = limits_cfg.flow_per_c
    band = limits_cfg.band_c
    max_t = limits_cfg.temp_rate_c_s * dt
    max_f = limits_cfg.flow_rate_t_h_s * dt

    def clamp_step(value: float, target: float, limit: float) -> float:
        return max(-limit, min(limit, target - value))

    def decide(t: float, f: float):
        return evaluate(
            limits_cfg, origin_t, origin_f, current_t, current_f,
            t, f, baseline_t_h, readback_ok=readback_ok, dt=dt,
        )

    temp_progress = clamp_step(current_t, target_c, max_t)
    flow_progress = clamp_step(current_f, target_flow_t_h, max_f)
    if not readback_ok and flow_progress > 0.0:
        flow_progress = 0.0
    proposed_t = current_t + temp_progress
    proposed_f = current_f + flow_progress

    decision = decide(proposed_t, proposed_f)
    if decision.allowed:
        return proposed_t, proposed_f, decision

    viol = set(decision.violations)
    if "coupling_band_exceeded_temp_lead" in viol:
        # 温度领先：流量需追到 f >= o_f + per_c*((t-o_t) - band)。
        need_f = origin_f + per_c * ((proposed_t - origin_t) - band)
        catch = clamp_step(current_f, need_f, max_f)
        if not readback_ok and catch > 0.0:
            catch = 0.0
        proposed_f = current_f + catch
        if proposed_f + 1e-9 < need_f:
            # 流量本步追不上：温度让步，收温度到流量可达的带沿。
            proposed_t = origin_t + (proposed_f - origin_f) / per_c + band
    elif "coupling_band_exceeded_flow_lead" in viol:
        # 流量领先：温度需追到 t >= o_t + (f-o_f)/per_c - band（降温时即温度
        # 必须随流量一起往下退，避免末端过热）。
        need_t = origin_t + (proposed_f - origin_f) / per_c - band
        catch = clamp_step(current_t, need_t, max_t)
        proposed_t = current_t + catch
        if proposed_t + 1e-9 < need_t:
            # 温度本步追不上：流量让步，收流量到温度可达的带沿。
            proposed_f = origin_f + per_c * ((proposed_t - origin_t) + band)

    decision = decide(proposed_t, proposed_f)
    return proposed_t, proposed_f, decision


def advance_heating(
    source: HeatSource,
    curve: HeatCurve,
    network: PrimaryNetwork,
    loop: SecondaryLoop,
    flow_t_h: float,
    now: float,
    dt: float = 1.0,
    readback_ok: bool = True,
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

    # 曲线换代（且与当前实际供温有温差）就是新调作：配出同一负荷段下的配对
    # 流量目标，并以当前位置为起点重新标记固定带。配对规则：流量按温差成比例
    # 联动（flow_per_c × ΔT），"温度爬、流量跟；温度退、流量退"。
    epoch = curve.epoch()
    if epoch != source.curve_epoch():
        # 新调作的配对流量目标是"调作起点在新供温下的联动线值"：从当前实际
        # 位置按 flow_per_c × 整条温差换算，并夹进流量硬边界。升温流量跟着
        # 加，降温流量跟着退——不会因为起点偏在线外而只动一点点。
        origin_t = source.temperature_c()
        origin_f = source.flow_t_h()
        paired_flow = origin_f + CouplingLimits.from_source(source.limits).flow_per_c * (
            target - origin_t
        )
        paired_flow = max(
            source.limits.flow_min_t_h, min(source.limits.flow_max_t_h, paired_flow)
        )
        source.begin_maneuver(target, paired_flow)
        source.note_curve_epoch(epoch)
    flow_target = source.target_flow_t_h()

    # 联动限速闸门：温差与流量一起算，裁决拒绝则整步不写，一次网不跟随。
    proposed_t, proposed_f, decision = propose_coupled_step(
        source, target, flow_target, network.baseline_t_h(), readback_ok, dt
    )
    if not decision.allowed:
        violations.extend(decision.violations)
        steps.append(
            {
                "component": "source",
                "action": "coupling_rejected",
                "detail": decision.as_dict(),
            }
        )
        return {
            "ok": False,
            "steps": steps,
            "violations": violations,
            "stage": "source",
            "coupling": decision.as_dict(),
        }

    try:
        temperature, flow = source.coupled_step(proposed_t, proposed_f, target, dt)
    except PlantError as error:
        violations.append(error.message)
        steps.append(
            {
                "component": "source",
                "action": "coupling_rejected",
                "detail": {"message": error.message, **decision.as_dict()},
            }
        )
        return {"ok": False, "steps": steps, "violations": violations, "stage": "source"}
    steps.append(
        {
            "component": "source",
            "action": "coupled_step",
            "detail": {
                "target_c": round(target, 2),
                "temperature_c": round(temperature, 2),
                "flow_t_h": round(flow, 2),
                "hold_s": round(source.quality_hold_s(), 3),
                "coupling": decision.as_dict(),
            },
        }
    )
    if source.limits.overtemperature(temperature):
        source.trip("overtemperature")
        violations.append("overtemperature")
        return {"ok": False, "steps": steps, "violations": violations, "stage": "source"}

    # 时序：调质调量放行并整体写入之后，一次网才跟随供温与流量。
    network.set_supply(temperature)
    network.set_return(loop.return_c())
    network.set_flow(flow)
    steps.append(
        {"component": "primary", "action": "follow_source", "detail": network.snapshot().as_dict()}
    )

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
        "return_c": round(loop.return_c(), 2),
        "coupling": decision.as_dict(),
    }
