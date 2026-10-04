"""调质调量联合限速。

调质（供温）与调量（流量）不允许再各按各的速率跑：温度还没到位流量就被拉起，
或者温度已经回落流量没跟着退，末端要么抢热要么过热。每次联动调整先由
:func:`joint_rate_limit` 按当前温度、流量与各自目标算出一条联合限速——本周期
温度允许落的区间与流量允许落的区间组成的矩形——提议的 ``(温度, 流量)`` 必须
整块落在矩形里，任何一个越界，整次调整当场拒绝，两个字段一个都不写。

限速带型一次定死为**固定带**，不做按负荷或行程缩放的动态带：

* 温度领先段（低负荷段，量调节，流量让步给温度）用固定温差带
  ``joint_temp_band_c``：温差还在带外时流量超前侧冻结，进带之后按
  ``1 - 剩余温差 / 固定带宽`` 线性放开，温差归零流量才拿到全速；温度下行时
  流量的跟随侧同样按这条增益放开、逆行侧（不降反升）当场封死，从根上消掉
  "温降量不退、末端过热"；
* 流量领先段（高负荷段，质调节，温度让步给流量）对称地用固定流差带
  ``joint_flow_band_t_h``，由流差给温度限速：流量不动（流差为零）时温度拿
  全速，正好是严寒段只调质、流量钉在设计值的质调节工况。

让步方向按负荷段切换，分界由配置 ``high_load_pct`` 定死：需求份额低于分界时
温度领先（流量让温度），达到或高于分界时流量领先（温度让流量）；冷态点火
（流量为零）一律按温度领先处理，先把供温托进固定温差带才允许点火流量。

跟随轴朝**加负荷方向**（升温时的流量上升、流量上升时的升温）受固定带增益
节流；朝**减负荷方向**（回撤）一律全速放开且不设逆行门槛——温度一旦下行，
流量在同一个联动步里就能全速退，从根上消掉"温降量不退、末端过热"。

流量回读缺失属于部分失败而不是停运：限速进入只许降不许升的降级模式，温度
同步只许降不许升，供控制台与审计照常工作，回读恢复后自动回到正常限速。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import SourceLimits

# 负荷段标识。
LOW_LOAD = "low"
HIGH_LOAD = "high"

# 让步方向标识。
TEMP_FIRST = "temperature"  # 温度领先，流量让步给温度（量调节）
FLOW_FIRST = "flow"  # 流量领先，温度让步给流量（质调节）

# 限速带型，定死为固定带。
BAND_KIND = "fixed"

_EPS = 1e-9


@dataclass(frozen=True)
class JointRateLimit:
    """一个联动步允许落入的 (温度区间 × 流量区间) 矩形。"""

    band_kind: str
    load_segment: str
    yield_to: str
    leader: str
    degraded: bool
    dt_s: float
    demand_pct: float
    high_load_pct: float
    temp_target_c: float
    flow_target_t_h: float
    temp_lower_c: float
    temp_upper_c: float
    flow_lower_t_h: float
    flow_upper_t_h: float
    temp_gain: float
    flow_gain: float
    ignite_flow_t_h: float
    violations: list[str]

    def allows(self, temperature_c: float, flow_t_h: float) -> bool:
        """提议的温度与流量是否整块落在允许矩形内。"""

        temperature = float(temperature_c)
        flow = float(flow_t_h)
        return (
            self.temp_lower_c - _EPS <= temperature <= self.temp_upper_c + _EPS
            and self.flow_lower_t_h - _EPS <= flow <= self.flow_upper_t_h + _EPS
        )

    def as_dict(self) -> dict:
        return {
            "band_kind": self.band_kind,
            "load_segment": self.load_segment,
            "yield_to": self.yield_to,
            "leader": self.leader,
            "degraded": self.degraded,
            "dt_s": round(self.dt_s, 3),
            "demand_pct": round(self.demand_pct, 2),
            "high_load_pct": round(self.high_load_pct, 2),
            "targets": {
                "temperature_c": round(self.temp_target_c, 2),
                "flow_t_h": round(self.flow_target_t_h, 2),
            },
            "allowed_band": {
                "temperature_c": [round(self.temp_lower_c, 2), round(self.temp_upper_c, 2)],
                "flow_t_h": [round(self.flow_lower_t_h, 2), round(self.flow_upper_t_h, 2)],
            },
            "gain": {
                "temperature": round(self.temp_gain, 4),
                "flow": round(self.flow_gain, 4),
            },
            "ignite_flow_t_h": round(self.ignite_flow_t_h, 2),
            "violations": list(self.violations),
        }


def load_segment(demand_pct: float, high_load_pct: float) -> str:
    """按需求份额确定负荷段。"""

    return HIGH_LOAD if float(demand_pct) >= float(high_load_pct) else LOW_LOAD


def _follower_band(
    current: float,
    target: float,
    step: float,
    gain: float,
    low: float,
    high: float,
) -> tuple[float, float]:
    """跟随轴的允许区间（温度轴与流量轴的加负荷方向都是数值增大）。

    朝加负荷方向（升到目标）按固定带增益节流，防超前抢热；朝减负荷方向
    （回撤、以及与领先轴反向的逆行）全速放开——温度一旦下行，流量立刻能
    全速退，从根上避免末端过热。目标一步可达时允许恰好落到目标。
    """

    increase = min(step * gain, max(0.0, target - current))
    decrease = step
    lower = max(low, current - decrease)
    upper = min(high, current + increase)
    if target - current <= step * gain + _EPS:
        upper = max(upper, min(high, max(low, target)))
    if current - target <= step + _EPS:
        lower = min(lower, max(low, target))
    return lower, upper


def joint_rate_limit(
    limits: SourceLimits,
    temperature_c: float,
    flow_t_h: float,
    target_temp_c: float,
    target_flow_t_h: float,
    dt_s: float,
    demand_pct: float,
    *,
    readback_missing: bool = False,
    flow_live: bool = True,
    can_ignite: bool = False,
) -> JointRateLimit:
    """按温差与流差算出本周期的联合限速矩形。

    ``readback_missing`` / ``flow_live`` 任一表示回读不可信即进入降级模式：
    温度与流量都只许降不许升。``can_ignite`` 表示供温已进固定温差带、允许
    流量从零一步点火到最小点火流量。
    """

    if dt_s <= 0:
        raise ValueError("步长必须为正")

    temperature = float(temperature_c)
    flow = float(flow_t_h)
    target_temp = float(target_temp_c)
    target_flow = float(target_flow_t_h)
    demand = float(demand_pct)
    high_pct = float(limits.high_load_pct)
    degraded = bool(readback_missing or not flow_live)

    violations: list[str] = []
    if limits.overtemperature(target_temp) or target_temp < limits.return_min_c:
        violations.append("target_temperature_out_of_range")
    if target_flow != 0.0 and not limits.flow_in_range(target_flow):
        violations.append("target_flow_out_of_range")

    segment = load_segment(demand, high_pct)
    # 冷态点火总是温度先行：没供温就没有点火流量。
    yield_to = TEMP_FIRST if (segment == LOW_LOAD or flow <= 0.0) else FLOW_FIRST
    leader = FLOW_FIRST if yield_to == TEMP_FIRST else TEMP_FIRST

    temp_step = limits.adjust_rate_c_s * dt_s
    flow_step = limits.adjust_rate_t_h_s * dt_s
    temp_gap = abs(target_temp - temperature)
    flow_gap = abs(target_flow - flow)

    # 固定带增益：差距在带外为零（超前侧冻结），归零拿到全速。
    temp_gain = max(0.0, 1.0 - flow_gap / limits.joint_flow_band_t_h)
    flow_gain = max(0.0, 1.0 - temp_gap / limits.joint_temp_band_c)

    temp_lower = max(limits.return_min_c, temperature - temp_step)
    temp_upper = min(limits.supply_max_c, temperature + temp_step)
    flow_lower = max(0.0, flow - flow_step)
    flow_upper = min(limits.flow_max_t_h, flow + flow_step)

    if degraded:
        # 部分失败：两轴都只许降不许升（流量可一直撤到零）。
        temp_upper = min(temp_upper, temperature)
        flow_upper = min(flow_upper, flow)
        flow_lower = 0.0
    elif flow <= 0.0:
        # 冷态：供温先进固定温差带，随后允许一步点火到最小点火流量。
        if can_ignite:
            flow_upper = max(0.0, min(limits.ignite_flow_t_h, target_flow))
        else:
            flow_upper = 0.0
        flow_lower = 0.0
    elif yield_to == TEMP_FIRST:
        # 温度领先：固定温差带给流量升侧节流，流量回撤全速以防末端过热。
        flow_lower, flow_upper = _follower_band(
            flow,
            target_flow,
            flow_step,
            flow_gain,
            limits.flow_min_t_h,
            limits.flow_max_t_h,
        )
    else:
        # 流量领先：固定流差带给温度升侧节流，降温全速（质调节）。
        temp_lower, temp_upper = _follower_band(
            temperature,
            target_temp,
            temp_step,
            temp_gain,
            limits.return_min_c,
            limits.supply_max_c,
        )
        flow_lower = max(flow_lower, limits.flow_min_t_h)

    # 非降级运行时流量只能是零或落在可信区间内；降级模式允许一路撤到零。
    if not degraded:
        if flow_upper > 0.0:
            flow_upper = max(flow_upper, limits.flow_min_t_h)
        # 下界只在"向上走不到可信区间"时抬到下限（如零起步点火），
        # 已在带内向减负荷方向回撤时不得抬回现值。
        if 0.0 < flow_upper < limits.flow_min_t_h:
            flow_lower = max(flow_lower, limits.flow_min_t_h)

    return JointRateLimit(
        band_kind=BAND_KIND,
        load_segment=segment,
        yield_to=yield_to,
        leader=leader,
        degraded=degraded,
        dt_s=float(dt_s),
        demand_pct=demand,
        high_load_pct=high_pct,
        temp_target_c=target_temp,
        flow_target_t_h=target_flow,
        temp_lower_c=temp_lower,
        temp_upper_c=temp_upper,
        flow_lower_t_h=flow_lower,
        flow_upper_t_h=flow_upper,
        temp_gain=temp_gain,
        flow_gain=flow_gain,
        ignite_flow_t_h=limits.ignite_flow_t_h,
        violations=violations,
    )
