from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from src.workloads.task import IoTTask

@dataclass
class FallbackConfig:
    mode: str = "eft"  # "eft" | "least_energy" | "weighted"
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0

def fallback_action(cfg: FallbackConfig, task: IoTTask, node_ids: List[str], L_hat: Dict[str, float], E_hat: Dict[str, float], R_hat: Dict[str, float]) -> str:
    if not node_ids:
        raise ValueError("No nodes available for fallback_action.")
    if cfg.mode == "eft":
        return min(node_ids, key=lambda nid: float(L_hat[nid]))
    if cfg.mode == "least_energy":
        return min(node_ids, key=lambda nid: float(E_hat[nid]))
    if cfg.mode == "weighted":
        a, b, g = cfg.alpha, cfg.beta, cfg.gamma
        return min(node_ids, key=lambda nid: a*float(R_hat[nid]) + b*float(L_hat[nid]) + g*float(E_hat[nid]))
    raise ValueError(f"Unknown fallback mode: {cfg.mode}")
