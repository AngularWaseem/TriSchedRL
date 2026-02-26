from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Any
import numpy as np

@dataclass
class SignalConfig:
    window: int = 50

class SignalTracker:
    def __init__(self, cfg: SignalConfig):
        w = int(cfg.window)
        self.lat_q: Deque[float] = deque(maxlen=w)
        self.eng_q: Deque[float] = deque(maxlen=w)
        self.viol_q: Deque[float] = deque(maxlen=w)
        self.cong_q: Deque[float] = deque(maxlen=w)
        self.energy_pressure_q: Deque[float] = deque(maxlen=w)

    def update_from_step(self, latency_s: float, energy_j: float, violation: int) -> None:
        self.lat_q.append(float(latency_s))
        self.eng_q.append(float(energy_j))
        self.viol_q.append(float(violation))

    def update_from_state(self, state: Dict[str, Any]) -> None:
        nodes = state["nodes"]
        rtts = [float(n["rtt_ms"]) for n in nodes]
        bws = [max(1e-6, float(n["bandwidth_mbps"])) for n in nodes]
        cong = (np.mean(rtts) / 1000.0) / (np.mean(bws))
        self.cong_q.append(float(cong))
        utils = [float(n["util"]) for n in nodes if n["energy_budget_j_step"] is not None]
        ep = float(np.mean(utils)) if utils else 0.0
        self.energy_pressure_q.append(ep)

    def phi(self) -> Dict[str, float]:
        def _mean(q): return float(np.mean(q)) if len(q) else 0.0
        def _p(q,p): return float(np.quantile(np.array(q,dtype=float), p)) if len(q) else 0.0
        return {
            "viol_rate": _mean(self.viol_q),
            "lat_p90": _p(self.lat_q, 0.90),
            "eng_p90": _p(self.eng_q, 0.90),
            "congestion": _mean(self.cong_q),
            "energy_pressure": _mean(self.energy_pressure_q),
        }
