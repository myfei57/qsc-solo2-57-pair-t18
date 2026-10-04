"""热源状态机与调质调量联动。

温度与流量是同一台机组的一次调质结果，任何一次调整都必须把两个字段当成一个
整体写入：并发的气象自动推送和人工调节同时改这两个字段时，只要中间态能被读到，
下游一次网就会拿到"上一次的温度配这一次的流量"。:meth:`HeatSource.adjust` 在
同一把可重入锁里写完两个字段，读方法与 :meth:`HeatSource.joint_step` 的
"校验—拒绝/写入"也都在同一把锁里完成，拒绝时不触碰任何字段。

联动调整不再分两段各跑各的：:meth:`HeatSource.joint_step` 每步先按
:mod:`~heatnet.source.joint` 给出的固定联合限速带算允许矩形，提议值整块在带内
才写；固定温差带、固定流差带与负荷段分界都取自配置，一次定死。
"""

from __future__ import annotations

import threading

from ..config import SourceLimits
from ..errors import InterlockError
from .joint import JointRateLimit, joint_rate_limit
from .state import SourceSnapshot

IDLE = "idle"
QUALITY = "quality"
QUANTITY = "quantity"
ONLINE = "online"
TRIPPED = "tripped"


class HeatSource:
    """热源：调质、调量与在线状态。"""

    def __init__(self, limits: SourceLimits) -> None:
        self._limits = limits
        self._guard = threading.RLock()
        self._state = IDLE
        self._temperature_c = limits.return_min_c
        self._flow_t_h = 0.0
        self._target_c = limits.supply_target_c
        self._target_flow_t_h = 0.0
        self._quality_hold_s = 0.0
        self._adjustments = 0
        self._trip_reason = ""
        self._trips = 0
        self._last_rate_limit: JointRateLimit | None = None

    @property
    def limits(self) -> SourceLimits:
        return self._limits

    def state(self) -> str:
        """返回当前状态。"""

        with self._guard:
            return self._state

    def temperature_c(self) -> float:
        """返回一次网供水温度。"""

        with self._guard:
            return self._temperature_c

    def flow_t_h(self) -> float:
        """返回一次网循环流量。"""

        with self._guard:
            return self._flow_t_h

    def target_c(self) -> float:
        """返回当前调质目标温度。"""

        with self._guard:
            return self._target_c

    def target_flow_t_h(self) -> float:
        """返回当前调量目标流量。"""

        with self._guard:
            return self._target_flow_t_h

    def quality_hold_s(self) -> float:
        """返回调质累计确认时间。"""

        with self._guard:
            return self._quality_hold_s

    def adjustments(self) -> int:
        """返回累计调整次数。"""

        with self._guard:
            return self._adjustments

    def trip_reason(self) -> str:
        """返回最近一次停机原因。"""

        with self._guard:
            return self._trip_reason

    def trips(self) -> int:
        """返回累计停机次数。"""

        with self._guard:
            return self._trips

    def set_target(self, temperature_c: float) -> float:
        """设定调质目标温度（越界即拒绝）。"""

        with self._guard:
            value = float(temperature_c)
            if value >= self._limits.supply_max_c:
                raise InterlockError("调质目标越上限", component="source")
            if value < self._limits.return_min_c:
                raise InterlockError("调质目标低于回水下限", component="source")
            self._target_c = value
            return self._target_c

    def set_flow_target(self, flow_t_h: float) -> float:
        """设定调量目标流量（目标允许为零，表示带量退出意向）。"""

        with self._guard:
            value = float(flow_t_h)
            if value != 0.0 and not self._limits.flow_in_range(value):
                raise InterlockError("调量目标越界", component="source")
            self._target_flow_t_h = value
            return self._target_flow_t_h

    def rate_limit_for(self, dt: float, demand_pct: float, *, readback_missing: bool = False) -> JointRateLimit:
        """按当前状态、目标与步长裁决联合限速矩形（不写任何字段）。"""

        with self._guard:
            can_ignite = self._flow_t_h <= 0.0 and self._quality_hold_s >= self._limits.quality_confirm_s
            limit = joint_rate_limit(
                self._limits,
                self._temperature_c,
                self._flow_t_h,
                self._target_c,
                self._target_flow_t_h,
                dt,
                demand_pct,
                readback_missing=readback_missing,
                can_ignite=can_ignite,
            )
            self._last_rate_limit = limit
            return limit

    def joint_step(
        self,
        dt: float,
        demand_pct: float,
        *,
        readback_missing: bool = False,
        temperature_c: float | None = None,
        flow_t_h: float | None = None,
    ) -> dict:
        """走一个联动步。

        不给定 ``temperature_c`` / ``flow_t_h`` 时按限速矩形自动向各自目标逼近
        （控制周期使用）；给定时按提议值原样校验，整块越出允许带则当场拒绝，
        温度与流量一个都不写、也绝不替调用方把提议夹回带内（人工与外部请求
        使用）。返回裁决与写后状态，裁决字典同时供控制台展示与审计落盘。
        """

        if dt <= 0:
            raise ValueError("步长必须为正")
        with self._guard:
            if self._state == TRIPPED:
                raise InterlockError(
                    f"热源已停机（{self._trip_reason}），禁止调质调量", component="source"
                )

            can_ignite = self._flow_t_h <= 0.0 and self._quality_hold_s >= self._limits.quality_confirm_s
            limit = joint_rate_limit(
                self._limits,
                self._temperature_c,
                self._flow_t_h,
                self._target_c,
                self._target_flow_t_h,
                dt,
                demand_pct,
                readback_missing=readback_missing,
                can_ignite=can_ignite,
            )
            self._last_rate_limit = limit

            auto = temperature_c is None and flow_t_h is None
            if auto:
                next_temp = min(max(self._target_c, limit.temp_lower_c), limit.temp_upper_c)
                next_flow = min(max(self._target_flow_t_h, limit.flow_lower_t_h), limit.flow_upper_t_h)
            else:
                next_temp = self._temperature_c if temperature_c is None else float(temperature_c)
                next_flow = self._flow_t_h if flow_t_h is None else float(flow_t_h)

            violations = list(limit.violations)
            if not limit.allows(next_temp, next_flow):
                violations.append("joint_rate_band_exceeded")
            if limit.degraded and (next_flow > self._flow_t_h or next_temp > self._temperature_c):
                violations.append("flow_readback_missing_down_only")
            if violations:
                return {
                    "ok": False,
                    "applied": False,
                    "violations": violations,
                    "rate_limit": limit.as_dict(),
                    "temperature_c": self._temperature_c,
                    "flow_t_h": self._flow_t_h,
                }

            changed = next_temp != self._temperature_c or next_flow != self._flow_t_h
            self._temperature_c = next_temp
            self._flow_t_h = next_flow
            if changed:
                self._adjustments += 1

            # 供温先在目标固定温差带内保持，保持时间够了才允许带量。
            if abs(self._target_c - self._temperature_c) <= self._limits.joint_temp_band_c:
                self._quality_hold_s += dt
            else:
                self._quality_hold_s = 0.0
            if self._flow_t_h <= 0.0:
                self._state = QUALITY if self._quality_hold_s < self._limits.quality_confirm_s else QUANTITY
            else:
                self._state = ONLINE

            return {
                "ok": True,
                "applied": True,
                "violations": [],
                "rate_limit": limit.as_dict(),
                "temperature_c": self._temperature_c,
                "flow_t_h": self._flow_t_h,
                "degraded": limit.degraded,
                "quality_confirmed": self.quality_confirmed(),
            }

    def adjust(self, temperature_c: float, flow_t_h: float) -> tuple[float, float]:
        """把温度与流量作为一个整体一次写入，返回写后的二元组。

        物理层原子写入口，只守绝对物理边界，不做联合限速；联动语义统一走
        :meth:`joint_step`。
        """

        with self._guard:
            if self._state == TRIPPED:
                raise InterlockError(f"热源已停机（{self._trip_reason}），禁止调整", component="source")
            temperature = float(temperature_c)
            flow = float(flow_t_h)
            if self._limits.overtemperature(temperature):
                raise InterlockError("调整后的供温越上限", component="source")
            if not self._limits.flow_in_range(flow):
                raise InterlockError("调整后的流量越界", component="source")
            self._temperature_c = temperature
            self._flow_t_h = flow
            self._adjustments += 1
            return self._temperature_c, self._flow_t_h

    def quality_confirmed(self) -> bool:
        """调质是否已确认（供带量点火放行）。"""

        with self._guard:
            return self._quality_hold_s >= self._limits.quality_confirm_s

    def fire(self, supply_c: float) -> float:
        """按供热曲线点火：把曲线给出的供水温度写入调质目标并立即到位。

        点火只是把供温立起来，带量必须等固定温差带内的调质确认，再由联动步
        放行点火流量。
        """

        with self._guard:
            if self._state == TRIPPED:
                raise InterlockError(f"热源已停机（{self._trip_reason}），禁止点火", component="source")
            value = float(supply_c)
            if self._limits.overtemperature(value):
                raise InterlockError("曲线供温越上限，拒绝点火", component="source")
            self._target_c = value
            self._temperature_c = value
            self._quality_hold_s = 0.0
            self._state = QUALITY
            return self._temperature_c

    def trip(self, reason: str) -> str:
        """停机并记录原因。"""

        with self._guard:
            self._state = TRIPPED
            self._trip_reason = str(reason)
            self._trips += 1
            self._flow_t_h = 0.0
            return self._state

    def reset(self) -> str:
        """复位停机状态。"""

        with self._guard:
            if self._state != TRIPPED:
                raise InterlockError("热源未停机，无需复位", component="source")
            self._trip_reason = ""
            self._quality_hold_s = 0.0
            self._target_flow_t_h = 0.0
            self._state = IDLE
            return self._state

    def snapshot(self) -> SourceSnapshot:
        """返回热源状态快照，温度与流量在同一把锁里一起取。"""

        with self._guard:
            return SourceSnapshot(
                state=self._state,
                temperature_c=self._temperature_c,
                flow_t_h=self._flow_t_h,
                target_c=self._target_c,
                target_flow_t_h=self._target_flow_t_h,
                quality_hold_s=self._quality_hold_s,
                adjustments=self._adjustments,
                trip_reason=self._trip_reason,
                rate_limit=self._last_rate_limit.as_dict() if self._last_rate_limit is not None else None,
            )
