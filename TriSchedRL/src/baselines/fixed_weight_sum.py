from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass
class FixedWeights:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.5

def choose_fixed_weight_sum(L_hat: Dict[str, float], E_hat: Dict[str, float], R_hat: Dict[str, float], w: FixedWeights) -> str:
    return min(L_hat.keys(), key=lambda nid: (w.alpha*float(R_hat[nid]) + w.beta*float(L_hat[nid]) + w.gamma*float(E_hat[nid])))
