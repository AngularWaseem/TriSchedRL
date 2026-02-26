from __future__ import annotations
from typing import Dict

def choose_minmin(L_hat: Dict[str, float]) -> str:
    return min(L_hat.keys(), key=lambda nid: float(L_hat[nid]))

def choose_maxmin(L_hat: Dict[str, float]) -> str:
    return max(L_hat.keys(), key=lambda nid: float(L_hat[nid]))
