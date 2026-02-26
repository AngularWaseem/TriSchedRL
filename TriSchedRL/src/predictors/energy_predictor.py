from __future__ import annotations
from typing import Dict
from .feature_builder import CandidateFeatures

class EnergyPredictor:
    def __init__(
        self,
        power_idle_edge_w: float = 4.0,
        power_dyn_edge_w: float = 12.0,
        power_idle_cloud_w: float = 20.0,
        power_dyn_cloud_w: float = 60.0,
        nic_power_w: float = 1.5,
        comm_overhead_ms: float = 1.0,
    ):
        self.power_idle_edge_w = float(power_idle_edge_w)
        self.power_dyn_edge_w = float(power_dyn_edge_w)
        self.power_idle_cloud_w = float(power_idle_cloud_w)
        self.power_dyn_cloud_w = float(power_dyn_cloud_w)
        self.nic_power_w = float(nic_power_w)
        self.comm_overhead_ms = float(comm_overhead_ms)

    @staticmethod
    def _t_exec(c_mi: float, f_mi_s: float) -> float:
        return float("inf") if f_mi_s <= 0 else (c_mi / f_mi_s)

    def _t_comm(self, s_mb: float, bandwidth_mbps: float, rtt_ms: float, loss: float) -> float:
        mbps = max(1e-6, bandwidth_mbps)
        rate_mb_s = mbps / 8.0
        tx_s = max(0.0, s_mb) / rate_mb_s
        one_way_s = (max(0.0, rtt_ms) / 1000.0) / 2.0
        overhead_s = self.comm_overhead_ms / 1000.0
        loss = min(0.99, max(0.0, loss))
        inflation = 1.0 / (1.0 - loss)
        return float(inflation * (tx_s + one_way_s + overhead_s))

    def predict_one(self, x: CandidateFeatures) -> float:
        c_mi = x.task["c_mi"]; s_mb = x.task["s_mb"]
        f = x.node["f_mi_s"]; util = float(x.node["util"])
        is_cloud = bool(x.node["kind_is_cloud"] >= 0.5)
        bw = x.link["bandwidth_mbps"]; rtt = x.link["rtt_ms"]; loss = x.link["loss"]
        t_exec = self._t_exec(c_mi, f)
        t_comm = self._t_comm(s_mb, bw, rtt, loss)
        if is_cloud:
            p_idle, p_dyn = self.power_idle_cloud_w, self.power_dyn_cloud_w
        else:
            p_idle, p_dyn = self.power_idle_edge_w, self.power_dyn_edge_w
        p_w = p_idle + util * p_dyn
        return float(p_w * t_exec + self.nic_power_w * t_comm)

    def predict(self, candidates: Dict[str, CandidateFeatures]) -> Dict[str, float]:
        return {nid: self.predict_one(x) for nid, x in candidates.items()}
