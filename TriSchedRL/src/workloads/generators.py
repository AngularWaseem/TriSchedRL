from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from .task import IoTTask

@dataclass
class WorkloadConfig:
    seed: int = 123
    mode: str = "poisson"   # poisson | bursty
    horizon_s: float = 60.0
    lambda_per_s: float = 2.0
    burst_lambda_per_s: float = 8.0
    on_mean_s: float = 4.0
    off_mean_s: float = 3.0

class WorkloadGenerator:
    def __init__(self, cfg: WorkloadConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def _sample_task(self, idx: int) -> IoTTask:
        c_mi = float(self.rng.lognormal(mean=5.2, sigma=0.5))
        s_mb = float(self.rng.lognormal(mean=0.0, sigma=0.8))
        p = int(self.rng.integers(0, 3))
        base = 0.8 + 0.004 * c_mi + 0.03 * s_mb
        d_s = float(max(0.2, base - 0.1 * p))
        return IoTTask(task_id=f"t{idx}", c_mi=c_mi, d_s=d_s, s_mb=s_mb, p=p)

    def generate(self) -> List[Tuple[float, IoTTask]]:
        cfg = self.cfg
        arrivals: List[Tuple[float, IoTTask]] = []
        t = 0.0
        i = 0

        if cfg.mode == "poisson":
            while t < cfg.horizon_s:
                dt = float(self.rng.exponential(1.0 / max(1e-6, cfg.lambda_per_s)))
                t += dt
                if t > cfg.horizon_s:
                    break
                arrivals.append((t, self._sample_task(i)))
                i += 1
            return arrivals

        if cfg.mode == "bursty":
            on = True
            while t < cfg.horizon_s:
                seg = float(self.rng.exponential(cfg.on_mean_s if on else cfg.off_mean_s))
                seg_end = min(cfg.horizon_s, t + seg)
                lam = cfg.burst_lambda_per_s if on else max(1e-6, cfg.lambda_per_s * 0.2)

                while t < seg_end:
                    dt = float(self.rng.exponential(1.0 / max(1e-6, lam)))
                    t += dt
                    if t > seg_end or t > cfg.horizon_s:
                        break
                    arrivals.append((t, self._sample_task(i)))
                    i += 1

                on = not on
            return arrivals

        raise ValueError(f"Unknown workload mode: {cfg.mode}")
