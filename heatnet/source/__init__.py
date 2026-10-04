"""热源调质调量。

调质（供温）与调量（流量）联动：每个控制周期由
:mod:`~heatnet.source.joint` 的固定联合限速带给出允许矩形，温度与流量整块
在带内才允许写，越界当场拒绝；让步方向按负荷段切换，流量回读缺失时进入只许
降不许升的部分失败模式。
"""

from __future__ import annotations

from .adjust import Adjustment, apply_adjustment
from .joint import (
    BAND_KIND,
    FLOW_FIRST,
    HIGH_LOAD,
    LOW_LOAD,
    TEMP_FIRST,
    JointRateLimit,
    joint_rate_limit,
    load_segment,
)
from .source import HeatSource
from .state import SourceSnapshot
from .weather import WeatherPush, push_weather

__all__ = [
    "Adjustment",
    "BAND_KIND",
    "FLOW_FIRST",
    "HIGH_LOAD",
    "HeatSource",
    "JointRateLimit",
    "LOW_LOAD",
    "SourceSnapshot",
    "TEMP_FIRST",
    "WeatherPush",
    "apply_adjustment",
    "joint_rate_limit",
    "load_segment",
    "push_weather",
]
