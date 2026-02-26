from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

@dataclass
class VectorizerConfig:
    use_node_kind: bool = True

class StateVectorizer:
    def __init__(self, node_ids: List[str], kappa_dim: int, cfg: VectorizerConfig = VectorizerConfig()):
        self.node_ids = list(node_ids)
        self.kappa_dim = int(kappa_dim)
        self.cfg = cfg
        self.per_node_dim = 8 + (1 if self.cfg.use_node_kind else 0)
        self.state_dim = self.per_node_dim * len(self.node_ids) + self.kappa_dim

    def _node_to_vec(self, n: Dict[str, Any]) -> np.ndarray:
        energy_budget = n["energy_budget_j_step"]
        energy_present = 0.0 if energy_budget is None else 1.0
        base = [
            float(n["util"]),
            float(n["queue_work_mi"]),
            float(n["f_mi_s"]),
            float(n["capacity_mi_step"]),
            float(n["bandwidth_mbps"]),
            float(n["rtt_ms"]),
            float(n["loss"]),
            float(energy_present),
        ]
        if self.cfg.use_node_kind:
            base.append(1.0 if str(n["kind"]).lower() == "cloud" else 0.0)
        return np.array(base, dtype=np.float32)

    def vectorize(self, state: Dict[str, Any], kappa: np.ndarray) -> np.ndarray:
        node_map = {str(n["node_id"]): n for n in state["nodes"]}
        parts = [self._node_to_vec(node_map[nid]) for nid in self.node_ids]
        s_flat = np.concatenate(parts, axis=0)
        kappa = np.asarray(kappa, dtype=np.float32).reshape(-1)
        if kappa.shape[0] != self.kappa_dim:
            raise ValueError(f"kappa_dim mismatch: expected {self.kappa_dim}, got {kappa.shape[0]}")
        return np.concatenate([s_flat, kappa], axis=0)
