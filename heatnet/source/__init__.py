"""热源调质调量。

热源必须先调质（把一次网供水温度提到曲线给定的目标）再调量（把循环流量提到
目标），两步都由 :class:`~heatnet.source.source.HeatSource` 的状态机看守；调整
要么整体生效、要么整体不生效，中间态不允许被下游读到。
"""

from __future__ import annotations

from .adjust import Adjustment, apply_adjustment
from .coupling import (
    BAND_MODE,
    CouplingDecision,
    CouplingLimits,
    evaluate,
    load_segment,
)
from .source import HeatSource
from .state import SourceSnapshot
from .weather import WeatherPush, push_weather

__all__ = [
    "Adjustment",
    "BAND_MODE",
    "CouplingDecision",
    "CouplingLimits",
    "HeatSource",
    "SourceSnapshot",
    "WeatherPush",
    "apply_adjustment",
    "evaluate",
    "load_segment",
    "push_weather",
]
