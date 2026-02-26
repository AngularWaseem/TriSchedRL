from __future__ import annotations
from typing import Dict

def choose_eft(L_hat: Dict[str, float]) -> str:
    return min(L_hat.keys(), key=lambda nid: float(L_hat[nid]))
