from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from src.workloads.task import IoTTask

@dataclass
class CandidateFeatures:
    task: Dict[str, float]
    node: Dict[str, float]
    link: Dict[str, float]

class FeatureBuilder:
    def __init__(self, iot_src_id: str = "iot"):
        self.iot_src_id = iot_src_id

    def build_candidates(self, state: Dict[str, Any], task: IoTTask) -> Dict[str, CandidateFeatures]:
        feats: Dict[str, CandidateFeatures] = {}
        phi = {"c_mi": float(task.c_mi), "d_s": float(task.d_s), "s_mb": float(task.s_mb), "p": float(task.p)}
        for n in state["nodes"]:
            nid = str(n["node_id"])
            psi = {
                "f_mi_s": float(n["f_mi_s"]),
                "capacity_mi_step": float(n["capacity_mi_step"]),
                "queue_work_mi": float(n["queue_work_mi"]),
                "util": float(n["util"]),
                "energy_budget_j_step": float(n["energy_budget_j_step"]) if n["energy_budget_j_step"] is not None else -1.0,
                "kind_is_cloud": 1.0 if str(n["kind"]).lower() == "cloud" else 0.0,
            }
            link = {"bandwidth_mbps": float(n["bandwidth_mbps"]), "rtt_ms": float(n["rtt_ms"]), "loss": float(n["loss"])}
            feats[nid] = CandidateFeatures(task=phi, node=psi, link=link)
        return feats
