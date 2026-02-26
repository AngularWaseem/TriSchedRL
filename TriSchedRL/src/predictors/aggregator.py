from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class AggregationConfig:
    mode: str = "topk"  # "stats" or "topk"
    k: int = 3
    include_counts: bool = True

class PredictionAggregator:
    def __init__(self, cfg: AggregationConfig):
        self.cfg = cfg

    def aggregate_topk(self, L: Dict[str, float], E: Dict[str, float], R: Dict[str, float], alpha: float, beta: float, gamma: float) -> np.ndarray:
        items: List[Tuple[str, float]] = []
        for nid in L.keys():
            score = alpha * float(R[nid]) + beta * float(L[nid]) + gamma * float(E[nid])
            items.append((nid, score))
        items.sort(key=lambda x: x[1])
        k = max(1, int(self.cfg.k))
        top = items[:k]
        vec: List[float] = []
        for nid, _ in top:
            vec.extend([float(L[nid]), float(E[nid]), float(R[nid])])
        while len(vec) < 3 * k:
            vec.extend([0.0, 0.0, 0.0])
        if self.cfg.include_counts:
            vec.append(float(len(L)))
        return np.array(vec, dtype=np.float32)

    def aggregate(self, L: Dict[str, float], E: Dict[str, float], R: Dict[str, float], alpha: float, beta: float, gamma: float) -> np.ndarray:
        return self.aggregate_topk(L, E, R, alpha, beta, gamma)
