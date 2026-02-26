from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class Node:
    node_id: str
    kind: str  # "edge" or "cloud"
    f: float   # MI/s
    capacity_mi_per_step: float
    power_idle_w: float
    power_dyn_w: float
    energy_budget_j_per_step: Optional[float] = None

@dataclass
class NodeRuntime:
    queue_work_mi: float = 0.0
    util: float = 0.0

class ResourcePool:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.runtime: Dict[str, NodeRuntime] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node
        self.runtime[node.node_id] = NodeRuntime()

    def all_ids(self) -> List[str]:
        return list(self.nodes.keys())

    def get(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def get_runtime(self, node_id: str) -> NodeRuntime:
        return self.runtime[node_id]

    def reset_runtime(self) -> None:
        for nid in self.nodes:
            self.runtime[nid] = NodeRuntime()

    def step_decay(self, dt_s: float) -> None:
        # Process queued work each step; update utilization proxy.
        for nid, node in self.nodes.items():
            rt = self.runtime[nid]
            processed = max(0.0, node.f) * max(0.0, dt_s)
            rt.queue_work_mi = max(0.0, rt.queue_work_mi - processed)
            cap = max(1e-6, node.capacity_mi_per_step)
            rt.util = float(min(1.0, rt.queue_work_mi / cap))
