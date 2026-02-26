from __future__ import annotations
from typing import Dict
from .feature_builder import CandidateFeatures

class LatencyPredictor:
    def __init__(self, overhead_ms: float = 1.0):
        self.overhead_ms = float(overhead_ms)

    @staticmethod
    def _t_exec(c_mi: float, f_mi_s: float) -> float:
        return float("inf") if f_mi_s <= 0 else (c_mi / f_mi_s)

    @staticmethod
    def _t_queue(queue_work_mi: float, f_mi_s: float) -> float:
        return float("inf") if f_mi_s <= 0 else (max(0.0, queue_work_mi) / f_mi_s)

    def _t_comm(self, s_mb: float, bandwidth_mbps: float, rtt_ms: float, loss: float) -> float:
        mbps = max(1e-6, bandwidth_mbps)
        rate_mb_s = mbps / 8.0
        tx_s = max(0.0, s_mb) / rate_mb_s
        one_way_s = (max(0.0, rtt_ms) / 1000.0) / 2.0
        overhead_s = self.overhead_ms / 1000.0
        loss = min(0.99, max(0.0, loss))
        inflation = 1.0 / (1.0 - loss)
        return float(inflation * (tx_s + one_way_s + overhead_s))

    def predict_one(self, x: CandidateFeatures) -> float:
        c_mi = x.task["c_mi"]; d_s = x.task["d_s"]; s_mb = x.task["s_mb"]
        f = x.node["f_mi_s"]; q = x.node["queue_work_mi"]
        bw = x.link["bandwidth_mbps"]; rtt = x.link["rtt_ms"]; loss = x.link["loss"]
        return float(self._t_exec(c_mi, f) + self._t_queue(q, f) + self._t_comm(s_mb, bw, rtt, loss))

    def predict(self, candidates: Dict[str, CandidateFeatures]) -> Dict[str, float]:
        return {nid: self.predict_one(x) for nid, x in candidates.items()}
