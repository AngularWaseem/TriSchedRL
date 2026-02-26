from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SLAConfig:
    hard_deadline: bool = True

def violation_indicator(latency_s: float, deadline_s: float) -> int:
    return 1 if latency_s > deadline_s else 0

def sla_penalty(latency_s: float, deadline_s: float, hard_deadline: bool = True) -> float:
    if latency_s <= deadline_s:
        return 0.0
    tardiness = latency_s - deadline_s
    if hard_deadline:
        return 1.0 + tardiness
    return tardiness
