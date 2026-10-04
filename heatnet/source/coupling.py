"""调质调量联动限速（固定温差带）。

温度与流量不再各跑各的：每一步都要过同一道联动闸门。闸门按"温差 + 流量"算一条
联合限速，双方相对**调作起点**的推进进度都必须待在一条**固定温差带**里——允许带
宽度取自配置（``coupling_band_c``，单位 °C），装配后定死，不随负荷、曲线或时间
移动，因此本模块只有 :data:`BAND_MODE` 声明的 ``"fixed"`` 一种制式，运行期不存在
第二条带，也没有动态带。

带的几何意义：以本次调作的起点（温度, 流量）为锚点，温度每推进 1 °C 允许流量至多
带走 ``flow_per_c`` 吨/小时。调作过程中累计漂移

    drift_c = (拟议温度 - 起点温度) - (拟议流量 - 起点流量) / flow_per_c

必须落在固定带内。温度先爬到目标而流量没跟上（drift > +band），或温度掉下来而
流量没跟着退（drift < -band）都会越界，越界即由调用方**当场拒绝**整步（温度、
流量一个都不写，一次网跟随也不得开始）。调作起点在目标改变时由热源重新标记，
因此拦的是"两段各跑各的"积累出来的错位，而不是单步抖动。

让步方向按负荷段定（负荷以流量相对一次网基线的百分比计），它决定联合限速预算
不够时**谁被收**（谁让步），而不是给谁豁免固定带：

* 低负荷段（流量让步，温度领步）：速率预算优先给温度，流量被收，跟着温度走；
* 高负荷段（温度让步，流量领步）：速率预算优先给流量，温度被收，跟着流量走；
* 中负荷段：双向对称，谁越界收谁。

无论哪个负荷段、升温还是降温，固定带两侧都硬拦：升温时"温度没到位流量就已经
拉起来"与降温时"温度掉了流量没跟着退"在联动线上是同一侧（流量相对温度偏高，
drift 小于 ``-band_c``），都当场拒绝；温度相对流量超前过多（小流量大温差）则
是另一侧（drift 大于 ``+band_c``），同样拒绝。

流量回读缺失属于**部分失败**而不是停机：此时闸门进入只降不升的降级态，任何要求
流量上升的步子一律拒绝（升温、降温方向都如此），降流量与纯降温仍放行。

限速与带形由本模块唯一的 :func:`evaluate` 计算，控制台展示、审计落盘和实际联锁
用的是同一个 :class:`CouplingDecision`，不存在两套口径。
"""

from __future__ import annotations

from dataclasses import dataclass

# 限速带制式：只此一种，写死。控制台与审计都引用这个常量，任何"动态带"都必须
# 先改这里、走配置装配，而不是在运行期临时放宽。
BAND_MODE = "fixed"

# 负荷段标签。
LOW_LOAD = "low"
MID_LOAD = "mid"
HIGH_LOAD = "high"

# 让步方。
YIELD_FLOW = "flow_yields"   # 低负荷：流量让步，温度领步
YIELD_TEMP = "temp_yields"  # 高负荷：温度让步，流量领步
YIELD_NONE = "symmetric"    # 中负荷：双向对称


@dataclass(frozen=True)
class CouplingLimits:
    """联动限速所需的固定约束（装配期定死）。"""

    flow_per_c: float
    band_c: float
    temp_rate_c_s: float
    flow_rate_t_h_s: float
    low_load_pct: float
    high_load_pct: float

    @classmethod
    def from_source(cls, source: object) -> "CouplingLimits":
        """从热源配置段取联动限速字段。"""

        return cls(
            flow_per_c=float(getattr(source, "coupling_flow_per_c")),
            band_c=float(getattr(source, "coupling_band_c")),
            temp_rate_c_s=float(getattr(source, "adjust_rate_c_s")),
            flow_rate_t_h_s=float(getattr(source, "coupling_flow_rate_t_h_s")),
            low_load_pct=float(getattr(source, "low_load_pct")),
            high_load_pct=float(getattr(source, "high_load_pct")),
        )


@dataclass(frozen=True)
class CouplingDecision:
    """一次联动限速评估的结论（联锁、控制台、审计共用同一份数据）。"""

    allowed: bool
    band_mode: str
    load_segment: str
    yields: str
    readback_ok: bool
    origin_temp_c: float
    origin_flow_t_h: float
    current_temp_c: float
    current_flow_t_h: float
    proposed_temp_c: float
    proposed_flow_t_h: float
    temp_step_c: float
    flow_step_t_h: float
    # 本步联合限速（供控制台展示与审计）。dt 为 None 时不做速率约束，取 None。
    temp_rate_limit_c_s: float | None
    flow_rate_limit_t_h_s: float | None
    # 拟议流量对应的固定带允许窗口（按拟议温度与调作起点折算）。
    allowed_flow_window_t_h: tuple[float, float]
    # 累计漂移（°C）：正为温度超前，负为流量超前；固定带即 [-band_c, +band_c]。
    drift_c: float
    band_c: float
    violations: list[str]

    def as_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "band_mode": self.band_mode,
            "load_segment": self.load_segment,
            "yields": self.yields,
            "readback_ok": self.readback_ok,
            "origin_temp_c": round(self.origin_temp_c, 2),
            "origin_flow_t_h": round(self.origin_flow_t_h, 2),
            "current_temp_c": round(self.current_temp_c, 2),
            "current_flow_t_h": round(self.current_flow_t_h, 2),
            "proposed_temp_c": round(self.proposed_temp_c, 2),
            "proposed_flow_t_h": round(self.proposed_flow_t_h, 2),
            "temp_step_c": round(self.temp_step_c, 3),
            "flow_step_t_h": round(self.flow_step_t_h, 3),
            "temp_rate_limit_c_s": (
                round(self.temp_rate_limit_c_s, 3) if self.temp_rate_limit_c_s is not None else None
            ),
            "flow_rate_limit_t_h_s": (
                round(self.flow_rate_limit_t_h_s, 3) if self.flow_rate_limit_t_h_s is not None else None
            ),
            "allowed_flow_window_t_h": [
                round(self.allowed_flow_window_t_h[0], 2),
                round(self.allowed_flow_window_t_h[1], 2),
            ],
            "drift_c": round(self.drift_c, 3),
            "band_c": round(self.band_c, 3),
            "violations": list(self.violations),
        }


def load_segment(flow_t_h: float, baseline_t_h: float, limits: CouplingLimits) -> str:
    """按流量相对基线的百分比划分负荷段。"""

    if baseline_t_h <= 0:
        return MID_LOAD
    pct = 100.0 * float(flow_t_h) / float(baseline_t_h)
    if pct < limits.low_load_pct:
        return LOW_LOAD
    if pct >= limits.high_load_pct:
        return HIGH_LOAD
    return MID_LOAD


def _segment_yields(segment: str) -> str:
    if segment == LOW_LOAD:
        return YIELD_FLOW
    if segment == HIGH_LOAD:
        return YIELD_TEMP
    return YIELD_NONE


def evaluate(
    limits: CouplingLimits,
    origin_temp_c: float,
    origin_flow_t_h: float,
    current_temp_c: float,
    current_flow_t_h: float,
    proposed_temp_c: float,
    proposed_flow_t_h: float,
    baseline_t_h: float,
    readback_ok: bool = True,
    dt: float | None = None,
) -> CouplingDecision:
    """评估一步调质调量是否通过联动限速，拒绝时带出全部越界原因。

    纯函数：不改任何设备状态。``origin_*`` 是本次调作起点（目标改变时由热源
    重新标记），``current_*`` 是当前回读，``proposed_*`` 是本步拟议值。调用方
    拿到 ``allowed=False`` 时必须整步拒绝，温度与流量一个都不许写，一次网跟随
    也不得开始。
    """

    o_t = float(origin_temp_c)
    o_f = float(origin_flow_t_h)
    cur_t = float(current_temp_c)
    cur_f = float(current_flow_t_h)
    new_t = float(proposed_temp_c)
    new_f = float(proposed_flow_t_h)
    temp_step = new_t - cur_t
    flow_step = new_f - cur_f

    segment = load_segment(cur_f, float(baseline_t_h), limits)
    yields = _segment_yields(segment)

    # 累计漂移：相对调作起点，温度进度与流量进度（折算成 °C）之差。
    drift = (new_t - o_t) - (new_f - o_f) / limits.flow_per_c

    # 拟议温度下流量应在的固定带窗口（供控制台展示"允许带"此刻落在哪）。
    coupled_flow = o_f + limits.flow_per_c * (new_t - o_t)
    window = (
        coupled_flow - limits.flow_per_c * limits.band_c,
        coupled_flow + limits.flow_per_c * limits.band_c,
    )

    violations: list[str] = []

    # 1) 回读部分失败：只降不升。流量要求上升一律拒绝（与升降温方向无关）。
    if not readback_ok and flow_step > 1e-9:
        violations.append("flow_readback_missing_down_only")

    # 2) 反向动作：不允许一边升温一边降流量，或一边降温一边升流量。
    if temp_step * flow_step < -1e-12:
        violations.append("opposed_move")

    # 3) 联合限速：本步速率不得越过配置斜率（只在控制周期给定时长时约束；
    #    人工一次性整步不做速率约束，只受固定带约束）。
    temp_rate_limit: float | None = None
    flow_rate_limit: float | None = None
    if dt is not None:
        if dt <= 0:
            raise ValueError("步长必须为正")
        temp_rate_limit = limits.temp_rate_c_s
        flow_rate_limit = limits.flow_rate_t_h_s
        if abs(temp_step) > temp_rate_limit * dt + 1e-9:
            violations.append("temp_rate_exceeded")
        if abs(flow_step) > flow_rate_limit * dt + 1e-9:
            violations.append("flow_rate_exceeded")

    # 4) 固定温差带：两侧硬拦，与负荷段、升降温方向无关。负荷段让步只决定速率
    #    预算不足时收谁（见控制器 propose_coupled_step），不豁免任何一侧的带。
    temp_ahead = drift > limits.band_c + 1e-9
    flow_ahead = drift < -limits.band_c - 1e-9
    if temp_ahead:
        violations.append("coupling_band_exceeded_temp_lead")
    if flow_ahead:
        violations.append("coupling_band_exceeded_flow_lead")

    return CouplingDecision(
        allowed=not violations,
        band_mode=BAND_MODE,
        load_segment=segment,
        yields=yields,
        readback_ok=bool(readback_ok),
        origin_temp_c=o_t,
        origin_flow_t_h=o_f,
        current_temp_c=cur_t,
        current_flow_t_h=cur_f,
        proposed_temp_c=new_t,
        proposed_flow_t_h=new_f,
        temp_step_c=temp_step,
        flow_step_t_h=flow_step,
        temp_rate_limit_c_s=temp_rate_limit,
        flow_rate_limit_t_h_s=flow_rate_limit,
        allowed_flow_window_t_h=window,
        drift_c=drift,
        band_c=limits.band_c,
        violations=violations,
    )
