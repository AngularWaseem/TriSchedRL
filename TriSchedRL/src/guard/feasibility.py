from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from src.workloads.task import IoTTask

@dataclass
class FeasibilityConfig:
    enforce_deadline: bool = True
    enforce_energy_budget: bool = True
    use_risk_threshold: bool = True
    risk_threshold_high_priority: float = 0.7
    high_priority_min_p: int = 2

def is_feasible(
    cfg: FeasibilityConfig,
    task: IoTTask,
    node_capacity_mi_step: float,
    energy_budget_j_step: Optional[float],
    L_hat_s: float,
    E_hat_j: float,
    R_hat: float,
) -> bool:
    if task.c_mi > node_capacity_mi_step:
        return False
    if cfg.enforce_deadline and (L_hat_s > task.d_s):
        return False
    if cfg.enforce_energy_budget and (energy_budget_j_step is not None):
        if energy_budget_j_step >= 0 and (E_hat_j > energy_budget_j_step):
            return False
    if cfg.use_risk_threshold and (task.p >= cfg.high_priority_min_p):
        if R_hat > cfg.risk_threshold_high_priority:
            return False
    return True
