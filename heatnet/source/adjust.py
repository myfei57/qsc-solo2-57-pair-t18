"""热源调整入口。

一次调整要么把温度与流量都写进去，要么一个都不写。这里把"联动限速/越界校验"和
"整体写入"放在同一个调用里：联动闸门拒绝时不触碰任何字段，一次网跟随也不会发生
（调质调量与一次网跟随的时序由调用方保证——闸门不过，跟随不启动），返回的
:class:`Adjustment` 里带出被拒原因与同一份联动限速结论，由平台写审计。并发调用
不会互相看见中间态。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import PlantError
from .coupling import CouplingDecision
from .source import HeatSource


@dataclass(frozen=True)
class Adjustment:
    """一次调质调量的结果。"""

    ok: bool
    temperature_c: float
    flow_t_h: float
    source: str
    violations: list[str]
    coupling: dict | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "temperature_c": round(self.temperature_c, 2),
            "flow_t_h": round(self.flow_t_h, 2),
            "source": self.source,
            "violations": list(self.violations),
            "coupling": self.coupling,
        }


def apply_adjustment(
    source: HeatSource,
    target_temp_c: float,
    target_flow_t_h: float,
    origin: str = "manual",
    decision: CouplingDecision | None = None,
) -> Adjustment:
    """原子地把供温与流量调整到目标值。

    ``decision`` 是调用方按当前回读与负荷段算好的联动限速结论。结论拒绝时本函数
    不触碰热源任何字段，直接把越界原因带回；给 ``None`` 表示只做绝对边界校验
    （装配/测试路径），联动限速由平台统一把关。
    """

    snapshot = source.snapshot()
    if decision is not None and not decision.allowed:
        return Adjustment(
            ok=False,
            temperature_c=snapshot.temperature_c,
            flow_t_h=snapshot.flow_t_h,
            source=origin,
            violations=list(decision.violations),
            coupling=decision.as_dict(),
        )

    try:
        temperature, flow = source.adjust(target_temp_c, target_flow_t_h)
    except PlantError as error:
        return Adjustment(
            ok=False,
            temperature_c=snapshot.temperature_c,
            flow_t_h=snapshot.flow_t_h,
            source=origin,
            violations=[error.message],
            coupling=decision.as_dict() if decision is not None else None,
        )
    return Adjustment(
        ok=True,
        temperature_c=temperature,
        flow_t_h=flow,
        source=origin,
        violations=[],
        coupling=decision.as_dict() if decision is not None else None,
    )
