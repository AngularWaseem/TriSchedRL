from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class EpisodeStats:
    n_tasks: int
    sla_viol_rate: float
    lat_mean: float
    lat_p95: float
    eng_mean: float
    eng_p95: float
    avg_reward: float

def summarize(latencies: List[float], energies: List[float], violations: List[int], rewards: List[float]) -> EpisodeStats:
    n = len(latencies)
    if n == 0:
        return EpisodeStats(0,0.0,0.0,0.0,0.0,0.0,0.0)
    lat = np.array(latencies, dtype=float)
    eng = np.array(energies, dtype=float)
    vio = np.array(violations, dtype=float)
    rew = np.array(rewards, dtype=float)
    return EpisodeStats(
        n_tasks=n,
        sla_viol_rate=float(vio.mean()),
        lat_mean=float(lat.mean()),
        lat_p95=float(np.quantile(lat, 0.95)),
        eng_mean=float(eng.mean()),
        eng_p95=float(np.quantile(eng, 0.95)),
        avg_reward=float(rew.mean()),
    )
