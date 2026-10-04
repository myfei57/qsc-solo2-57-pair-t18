"""热源状态机与调质调量。

温度与流量是同一台机组的一次调质结果，任何一次调整都必须把两个字段当成一个
整体写入：并发的气象自动推送和人工调节同时改这两个字段时，只要中间态能被读到，
下游一次网就会拿到"上一次的温度配这一次的流量"。:meth:`HeatSource.adjust` 在
同一把可重入锁里写完两个字段，读方法也在同一把锁里取快照。
"""

from __future__ import annotations

import threading

from ..config import SourceLimits
from ..errors import InterlockError
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
        self._quality_hold_s = 0.0
        self._adjustments = 0
        self._trip_reason = ""
        self._trips = 0
        # 联动限速的"调作起点"：目标改变时重新标记，固定带按它量累计漂移。
        self._origin_temp_c = self._temperature_c
        self._origin_flow_t_h = self._flow_t_h
        # 调作的配对流量目标：与调质目标一起推进，没有配对目标时随当前流量。
        self._target_flow_t_h = self._flow_t_h
        # 最近一次按曲线开调作时看到的曲线代际（供控制周期判断是否要开新调作）。
        self._curve_epoch = 0

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

    def quality_hold_s(self) -> float:
        """返回调质累计确认时间。"""

        with self._guard:
            return self._quality_hold_s

    def adjustments(self) -> int:
        """返回累计调整次数。"""

        with self._guard:
            return self._adjustments

    def target_flow_t_h(self) -> float:
        """返回本次联动调作的配对流量目标。"""

        with self._guard:
            return self._target_flow_t_h

    def begin_maneuver(self, target_c: float, target_flow_t_h: float) -> tuple[float, float]:
        """开启一段联动调作：标记新的调质/调量目标并以当前位置为起点。"""

        with self._guard:
            self._target_c = float(target_c)
            self._target_flow_t_h = float(target_flow_t_h)
            self._origin_temp_c = self._temperature_c
            self._origin_flow_t_h = self._flow_t_h
            return self._target_c, self._target_flow_t_h

    def curve_epoch(self) -> int:
        """返回最近一次据此开调作的曲线代际。"""

        with self._guard:
            return self._curve_epoch

    def note_curve_epoch(self, epoch: int) -> int:
        """记录本周期已处理过的曲线代际。"""

        with self._guard:
            self._curve_epoch = int(epoch)
            return self._curve_epoch

    def origin(self) -> tuple[float, float]:
        """返回本次联动调作起点（供温, 流量）。"""

        with self._guard:
            return self._origin_temp_c, self._origin_flow_t_h

    def mark_origin(self) -> tuple[float, float]:
        """以当前温度与流量标记新的联动调作起点（目标改变时调用）。"""

        with self._guard:
            self._origin_temp_c = self._temperature_c
            self._origin_flow_t_h = self._flow_t_h
            return self._origin_temp_c, self._origin_flow_t_h

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
            if value != self._target_c:
                self._target_c = value
                # 新目标就是新调作：固定带从当前位置重新量，避免沿用上一段的漂移。
                self._origin_temp_c = self._temperature_c
                self._origin_flow_t_h = self._flow_t_h
            return self._target_c

    def adjust(self, temperature_c: float, flow_t_h: float) -> tuple[float, float]:
        """把温度与流量作为一个整体一次写入，返回写后的二元组。"""

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
            self._target_flow_t_h = flow
            self._adjustments += 1
            return self._temperature_c, self._flow_t_h

    def coupled_step(self, temperature_c: float, flow_t_h: float, target_c: float, dt: float = 1.0) -> tuple[float, float]:
        """联动限速放行后的一步调质调量：温度与流量在同一把锁里一起写。

        只负责原子写入与绝对边界；固定带、联合限速、回读降级由联动闸门
        （:mod:`heatnet.source.coupling`）在调用前裁决，裁决拒绝不得走到这里。
        写后按温度是否到位更新调质确认计时与状态机。
        """

        if dt <= 0:
            raise ValueError("步长必须为正")
        with self._guard:
            if self._state == TRIPPED:
                raise InterlockError(f"热源已停机（{self._trip_reason}），禁止调整", component="source")
            temperature = float(temperature_c)
            flow = float(flow_t_h)
            if self._limits.overtemperature(temperature):
                raise InterlockError("调整后的供温越上限", component="source")
            if not self._limits.flow_in_range(flow):
                raise InterlockError("调整后的流量越界", component="source")
            self._target_c = float(target_c)
            self._temperature_c = temperature
            self._flow_t_h = flow
            self._adjustments += 1
            if abs(self._temperature_c - self._target_c) <= 1e-9:
                self._quality_hold_s += dt
            else:
                self._quality_hold_s = 0.0
            if self._quality_hold_s < self._limits.quality_confirm_s:
                self._state = QUALITY
            elif self._state != ONLINE:
                self._state = QUANTITY
            return self._temperature_c, self._flow_t_h

    def settle_quality(self, target_c: float, dt: float) -> float:
        """向目标温度逼近一个步长，返回逼近后的温度。"""

        if dt <= 0:
            raise ValueError("步长必须为正")
        with self._guard:
            if self._state == TRIPPED:
                raise InterlockError(f"热源已停机（{self._trip_reason}），禁止调质", component="source")
            self._target_c = float(target_c)
            step = self._limits.adjust_rate_c_s * dt
            delta = self._target_c - self._temperature_c
            if abs(delta) <= step:
                self._temperature_c = self._target_c
                self._quality_hold_s += dt
            else:
                self._temperature_c += step if delta > 0 else -step
                self._quality_hold_s = 0.0
            if self._quality_hold_s < self._limits.quality_confirm_s:
                self._state = QUALITY
            elif self._state != ONLINE:
                self._state = QUANTITY
            return self._temperature_c

    def quality_confirmed(self) -> bool:
        """调质是否已确认（供调量放行）。"""

        with self._guard:
            return self._quality_hold_s >= self._limits.quality_confirm_s

    def set_flow(self, flow_t_h: float) -> float:
        """在调质确认之后调量。"""

        with self._guard:
            if self._state == TRIPPED:
                raise InterlockError(f"热源已停机（{self._trip_reason}），禁止调量", component="source")
            if not self.quality_confirmed():
                raise InterlockError("调质未确认，禁止调量", component="source")
            value = float(flow_t_h)
            if not self._limits.flow_in_range(value):
                raise InterlockError("调量目标越界", component="source")
            self._flow_t_h = value
            self._target_flow_t_h = value
            self._state = ONLINE
            return self._flow_t_h

    def fire(self, supply_c: float) -> float:
        """按供热曲线点火：把曲线给出的供水温度写入调质目标。"""

        with self._guard:
            if self._state == TRIPPED:
                raise InterlockError(f"热源已停机（{self._trip_reason}），禁止点火", component="source")
            value = float(supply_c)
            if self._limits.overtemperature(value):
                raise InterlockError("曲线供温越上限，拒绝点火", component="source")
            self._target_c = value
            # 点火只"装定"曲线目标，实际供温不允许脱离流量一步跳到位：温度与
            # 流量由联动限速一起带上去。新曲线即新调作，固定带从当前位置量起。
            self._origin_temp_c = self._temperature_c
            self._origin_flow_t_h = self._flow_t_h
            self._target_flow_t_h = self._flow_t_h
            self._state = QUANTITY if self.quality_confirmed() else QUALITY
            return self._target_c

    def trip(self, reason: str) -> str:
        """停机并记录原因。"""

        with self._guard:
            self._state = TRIPPED
            self._trip_reason = str(reason)
            self._trips += 1
            self._flow_t_h = 0.0
            self._target_flow_t_h = 0.0
            return self._state

    def reset(self) -> str:
        """复位停机状态。"""

        with self._guard:
            if self._state != TRIPPED:
                raise InterlockError("热源未停机，无需复位", component="source")
            self._trip_reason = ""
            self._quality_hold_s = 0.0
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
                quality_hold_s=self._quality_hold_s,
                adjustments=self._adjustments,
                trip_reason=self._trip_reason,
                origin_temp_c=self._origin_temp_c,
                origin_flow_t_h=self._origin_flow_t_h,
            )
