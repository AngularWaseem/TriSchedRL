from __future__ import annotations
from typing import Dict, List, Optional
from src.workloads.task import IoTTask
from .feasibility import FeasibilityConfig, is_feasible

def feasible_set(
    feas_cfg: FeasibilityConfig,
    task: IoTTask,
    node_caps_mi_step: Dict[str, float],
    node_energy_budget: Dict[str, Optional[float]],
    L_hat: Dict[str, float],
    E_hat: Dict[str, float],
    R_hat: Dict[str, float],
) -> List[str]:
    nf: List[str] = []
    for nid in L_hat.keys():
        if is_feasible(
            cfg=feas_cfg,
            task=task,
            node_capacity_mi_step=float(node_caps_mi_step[nid]),
            energy_budget_j_step=node_energy_budget.get(nid, None),
            L_hat_s=float(L_hat[nid]),
            E_hat_j=float(E_hat[nid]),
            R_hat=float(R_hat[nid]),
        ):
            nf.append(nid)
    return nf

def repair_action(
    task: IoTTask,
    nf: List[str],
    L_hat: Dict[str, float],
    E_hat: Dict[str, float],
    R_hat: Dict[str, float],
    alpha: float,
    beta: float,
    gamma: float,
) -> str:
    best = None
    best_score = float("inf")
    for nid in nf:
        score = alpha * float(R_hat[nid]) + beta * float(L_hat[nid]) + gamma * float(E_hat[nid])
        if score < best_score:
            best_score = score
            best = nid
    if best is None:
        raise ValueError("repair_action called with empty feasible set.")
    return best
