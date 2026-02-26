from __future__ import annotations
from typing import Dict

def choose_least_energy(E_hat: Dict[str, float]) -> str:
    return min(E_hat.keys(), key=lambda nid: float(E_hat[nid]))
