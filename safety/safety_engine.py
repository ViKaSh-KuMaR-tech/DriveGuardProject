"""
Safety / risk engine.

HOW THE SCORE WORKS (edit the numbers in config.SafetyConfig, not here)
-----------------------------------------------------------------------
1. Each detector already filters noise with its own timer, so the engine
   receives five *persistent* booleans: drowsy, yawning, distracted, absent,
   phone.
2. Every active behaviour adds its weight in "risk points":
       drowsy 60, absent 60, phone 45, distracted 40, yawning 20   (defaults)
   Multiple behaviours add up (capped at 100).
3. The raw total is smoothed with an exponential filter (rises fast, falls
   slowly) so the displayed risk does not flicker and does not drop to SAFE
   the instant one behaviour stops.
4. Smoothed risk -> level:
       < 5   SAFE        5-25 LOW        25-50 MEDIUM        >= 50 HIGH
   Safety score shown to the user = 100 - smoothed risk.
5. If the level reaches `alert_min_level` (MEDIUM), the highest-priority active
   behaviour is returned as the alert to speak. Cooldowns are handled by the
   audio module.

Examples: yawning alone -> LOW (no alert). Phone alone -> MEDIUM (alert).
Drowsy alone -> HIGH. Yawning + distracted (20+40=60) -> HIGH.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple

from config import SafetyConfig

EVENT_KEYS = ("drowsy", "yawning", "distracted", "absent", "phone")


class RiskLevel(IntEnum):
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass(frozen=True)
class SafetyInputs:
    drowsy: bool = False
    yawning: bool = False
    distracted: bool = False
    driver_absent: bool = False
    phone_in_use: bool = False


@dataclass(frozen=True)
class SafetyAssessment:
    raw_risk: float
    smoothed_risk: float
    safety_score: int
    level: RiskLevel
    active_events: Tuple[str, ...]
    alert_event: Optional[str]
    alert_message: Optional[str]


class SafetyEngine:
    def __init__(self, cfg: SafetyConfig) -> None:
        missing = [k for k in EVENT_KEYS if k not in cfg.weights]
        if missing:
            raise ValueError(f"SafetyConfig.weights is missing keys: {missing}")
        if cfg.alert_min_level not in RiskLevel.__members__:
            raise ValueError(f"alert_min_level must be one of {list(RiskLevel.__members__)}")
        self._cfg = cfg
        self._alert_min = RiskLevel[cfg.alert_min_level]
        self._smoothed = 0.0
        self._last_time: Optional[float] = None

    def update(self, inputs: SafetyInputs, now: float) -> SafetyAssessment:
        flags: Dict[str, bool] = {
            "drowsy": inputs.drowsy,
            "yawning": inputs.yawning,
            "distracted": inputs.distracted,
            "absent": inputs.driver_absent,
            "phone": inputs.phone_in_use,
        }
        active = tuple(k for k in EVENT_KEYS if flags[k])
        raw = min(100.0, sum(self._cfg.weights[k] for k in active))

        # Time-based exponential smoothing: new = old + (target - old) * (1 - e^(-dt/tau))
        dt = 0.0 if self._last_time is None else max(0.0, now - self._last_time)
        self._last_time = now
        tau = self._cfg.rise_tau_s if raw > self._smoothed else self._cfg.fall_tau_s
        alpha = 1.0 if tau <= 0 else 1.0 - math.exp(-dt / tau)
        self._smoothed += (raw - self._smoothed) * alpha

        level = self._level_for(self._smoothed)
        alert_event = alert_message = None
        if level >= self._alert_min:
            for key in self._cfg.alert_priority:
                if key in active:
                    alert_event = key
                    alert_message = self._cfg.messages.get(key)
                    break

        return SafetyAssessment(
            raw_risk=raw,
            smoothed_risk=self._smoothed,
            safety_score=int(round(max(0.0, 100.0 - self._smoothed))),
            level=level,
            active_events=active,
            alert_event=alert_event,
            alert_message=alert_message,
        )

    def _level_for(self, risk: float) -> RiskLevel:
        c = self._cfg
        if risk >= c.high_at:
            return RiskLevel.HIGH
        if risk >= c.medium_at:
            return RiskLevel.MEDIUM
        if risk >= c.low_at:
            return RiskLevel.LOW
        return RiskLevel.SAFE
