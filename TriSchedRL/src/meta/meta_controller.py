from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

@dataclass
class MetaConfig:
    w_min: float = 0.10
    w_max: float = 0.80
    ema: float = 0.85
    k_sla: float = 2.5
    k_lat: float = 1.5
    k_eng: float = 1.2
    base_alpha: float = 0.40
    base_beta: float = 0.35
    base_gamma: float = 0.25

class MetaController:
    def __init__(self, cfg: MetaConfig):
        self.cfg = cfg
        self.w = np.array([cfg.base_alpha, cfg.base_beta, cfg.base_gamma], dtype=float)

    def _clip_norm(self, w: np.ndarray) -> np.ndarray:
        w = np.clip(w, self.cfg.w_min, self.cfg.w_max)
        w = w / (w.sum() + 1e-12)
        return w

    def update(self, phi: Dict[str, float]) -> Tuple[float, float, float]:
        sla_u = self.cfg.k_sla * float(phi.get("viol_rate", 0.0)) + 0.5 * float(phi.get("lat_p90", 0.0))
        lat_u = self.cfg.k_lat * float(phi.get("congestion", 0.0)) + 0.7 * float(phi.get("lat_p90", 0.0))
        eng_u = self.cfg.k_eng * float(phi.get("energy_pressure", 0.0)) + 0.3 * float(phi.get("eng_p90", 0.0))

        u = np.array([sla_u, lat_u, eng_u], dtype=float)
        u = u - u.max()
        p = np.exp(u); p = p / (p.sum() + 1e-12)

        base = np.array([self.cfg.base_alpha, self.cfg.base_beta, self.cfg.base_gamma], dtype=float)
        target = self._clip_norm(0.6 * base + 0.4 * p)

        self.w = self.cfg.ema * self.w + (1.0 - self.cfg.ema) * target
        self.w = self._clip_norm(self.w)
        return float(self.w[0]), float(self.w[1]), float(self.w[2])
