from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .resources import ResourcePool
from .network import NetworkModel
from .sla import SLAConfig, violation_indicator, sla_penalty

@dataclass
class Task:
    task_id: str
    c_mi: float
    d_s: float
    s_mb: float
    p: int

@dataclass
class StepResult:
    node_id: str
    latency_s: float
    energy_j: float
    violation: int
    sla_penalty: float
    t_exec_s: float
    t_queue_s: float
    t_comm_s: float

class EdgeCloudEnv:
    def __init__(self, resources: ResourcePool, network: NetworkModel, sla: SLAConfig, dt_s: float = 1.0):
        self.resources = resources
        self.network = network
        self.sla = sla
        self.dt_s = float(dt_s)

    def reset(self, seed: int = 0) -> Dict[str, Any]:
        self.resources.reset_runtime()
        return self.observe_state()

    def observe_state(self) -> Dict[str, Any]:
        nodes = []
        for nid in self.resources.all_ids():
            node = self.resources.get(nid)
            rt = self.resources.get_runtime(nid)
            link = self.network.get_link("iot", nid)
            nodes.append({
                "node_id": nid,
                "kind": node.kind,
                "f_mi_s": float(node.f),
                "capacity_mi_step": float(node.capacity_mi_per_step),
                "queue_work_mi": float(rt.queue_work_mi),
                "util": float(rt.util),
                "energy_budget_j_step": node.energy_budget_j_per_step,
                "bandwidth_mbps": float(link.bandwidth_mbps),
                "rtt_ms": float(link.rtt_ms),
                "loss": float(link.loss),
            })
        return {"nodes": nodes}

    def _t_exec(self, task: Task, node_id: str) -> float:
        f = max(1e-9, float(self.resources.get(node_id).f))
        return float(task.c_mi) / f

    def _t_queue(self, node_id: str) -> float:
        rt = self.resources.get_runtime(node_id)
        f = max(1e-9, float(self.resources.get(node_id).f))
        return float(max(0.0, rt.queue_work_mi) / f)

    def _t_comm(self, task: Task, node_id: str) -> float:
        link = self.network.get_link("iot", node_id)
        mbps = max(1e-6, float(link.bandwidth_mbps))
        rate_mb_s = mbps / 8.0
        tx_s = max(0.0, float(task.s_mb)) / rate_mb_s
        one_way_s = (max(0.0, float(link.rtt_ms)) / 1000.0) / 2.0
        overhead_s = max(0.0, float(link.overhead_ms)) / 1000.0
        loss = min(0.99, max(0.0, float(link.loss)))
        inflation = 1.0 / (1.0 - loss)
        return float(inflation * (tx_s + one_way_s + overhead_s))

    def _energy(self, task: Task, node_id: str, t_exec: float, t_comm: float) -> float:
        node = self.resources.get(node_id)
        rt = self.resources.get_runtime(node_id)
        p_w = float(node.power_idle_w) + float(rt.util) * float(node.power_dyn_w)
        nic_w = 1.5
        return float(p_w * t_exec + nic_w * t_comm)

    def step(self, task: Task, node_id: str) -> Tuple[Dict[str, Any], StepResult, bool, Dict[str, Any]]:
        t_exec = self._t_exec(task, node_id)
        t_queue = self._t_queue(node_id)
        t_comm = self._t_comm(task, node_id)
        latency = t_exec + t_queue + t_comm

        vio = violation_indicator(latency, float(task.d_s))
        pen = sla_penalty(latency, float(task.d_s), hard_deadline=self.sla.hard_deadline)
        energy = self._energy(task, node_id, t_exec, t_comm)

        rt = self.resources.get_runtime(node_id)
        rt.queue_work_mi += float(task.c_mi)
        self.resources.step_decay(self.dt_s)

        res = StepResult(
            node_id=node_id,
            latency_s=float(latency),
            energy_j=float(energy),
            violation=int(vio),
            sla_penalty=float(pen),
            t_exec_s=float(t_exec),
            t_queue_s=float(t_queue),
            t_comm_s=float(t_comm),
        )
        return self.observe_state(), res, False, {}
