from __future__ import annotations
from typing import Dict
import math
from .feature_builder import CandidateFeatures

class SLARiskPredictor:
    def __init__(self, tau_s: float = 0.25):
        self.tau_s = float(max(1e-6, tau_s))

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def predict_from_latency(self, candidates: Dict[str, CandidateFeatures], latency_hat: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for nid, x in candidates.items():
            d = float(x.task["d_s"])
            L = float(latency_hat[nid])
            z = (L - d) / self.tau_s
            out[nid] = float(self._sigmoid(z))
        return out
