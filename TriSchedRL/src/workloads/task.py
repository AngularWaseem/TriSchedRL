from __future__ import annotations
from dataclasses import dataclass

@dataclass
class IoTTask:
    task_id: str
    c_mi: float
    d_s: float
    s_mb: float
    p: int
