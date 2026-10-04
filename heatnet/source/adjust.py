"""热源联动调整入口。

一次调质调量联动调整先登记温度目标与流量目标，再按固定联合限速带走一个
联动步：限速矩形来自 :mod:`~heatnet.source.joint`，控制台展示与审计落盘用的是
同一份裁决字典。提议越界时本周期不写温度与流量（目标仍保留，后续控制周期
按限速继续逼近）；目标本身越过物理区间时整次请求作废，连目标也不登记。

流量回读缺失属于部分失败：联动步进入只许降不许升的降级模式，升流量或升温度
的请求当场拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass

from .source import HeatSource


@dataclass(frozen=True)
class Adjustment:
    """一次调质调量联动调整的结果。"""

    ok: bool
    applied: bool
    temperature_c: float
    flow_t_h: float
    target_temperature_c: float
    target_flow_t_h: float
    degraded: bool
    source: str
    violations: list[str]
    rate_limit: dict | None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "applied": self.applied,
            "temperature_c": round(self.temperature_c, 2),
            "flow_t_h": round(self.flow_t_h, 2),
            "target_temperature_c": round(self.target_temperature_c, 2),
            "target_flow_t_h": round(self.target_flow_t_h, 2),
            "degraded": self.degraded,
            "source": self.source,
            "violations": list(self.violations),
            "rate_limit": self.rate_limit,
        }


def apply_adjustment(
    source: HeatSource,
    target_temp_c: float,
    target_flow_t_h: float,
    dt_s: float = 1.0,
    demand_pct: float = 60.0,
    *,
    origin: str = "manual",
    readback_missing: bool = False,
) -> Adjustment:
    """登记联动目标并按固定联合限速带走一个步长。"""

    snapshot = source.snapshot()
    if source.limits.overtemperature(target_temp_c) or float(target_temp_c) < source.limits.return_min_c:
        violations = ["调质目标越界"]
    elif float(target_flow_t_h) != 0.0 and not source.limits.flow_in_range(target_flow_t_h):
        violations = ["调量目标越界"]
    else:
        violations = []
    if violations:
        return Adjustment(
            ok=False,
            applied=False,
            temperature_c=snapshot.temperature_c,
            flow_t_h=snapshot.flow_t_h,
            target_temperature_c=snapshot.target_c,
            target_flow_t_h=snapshot.target_flow_t_h,
            degraded=bool(readback_missing),
            source=origin,
            violations=violations,
            rate_limit=snapshot.rate_limit,
        )

    source.set_target(target_temp_c)
    source.set_flow_target(target_flow_t_h)
    # 人工/外部请求按提议值原样走一个联动步：越界当场拒绝，不夹回带内。
    report = source.joint_step(
        dt_s,
        demand_pct,
        readback_missing=readback_missing,
        temperature_c=target_temp_c,
        flow_t_h=target_flow_t_h,
    )
    return Adjustment(
        ok=not report["violations"],
        applied=report["applied"],
        temperature_c=report["temperature_c"],
        flow_t_h=report["flow_t_h"],
        target_temperature_c=float(target_temp_c),
        target_flow_t_h=float(target_flow_t_h),
        degraded=report.get("degraded", False),
        source=origin,
        violations=report["violations"],
        rate_limit=report["rate_limit"],
    )
