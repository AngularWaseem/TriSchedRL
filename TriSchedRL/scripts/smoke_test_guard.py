from __future__ import annotations
import numpy as np
from src.env.resources import ResourcePool, Node
from src.env.network import NetworkModel, Link
from src.env.sla import SLAConfig
from src.env.edgecloud_env import EdgeCloudEnv, Task
from src.workloads.generators import WorkloadConfig, WorkloadGenerator
from src.workloads.task import IoTTask
from src.predictors.feature_builder import FeatureBuilder
from src.predictors.latency_predictor import LatencyPredictor
from src.predictors.energy_predictor import EnergyPredictor
from src.predictors.sla_risk_predictor import SLARiskPredictor
from src.predictors.aggregator import AggregationConfig, PredictionAggregator
from src.guard.feasibility import FeasibilityConfig, is_feasible
from src.guard.repair import feasible_set, repair_action
from src.guard.fallback import FallbackConfig, fallback_action

def build_demo_env(seed: int = 7) -> EdgeCloudEnv:
    pool = ResourcePool()
    pool.add_node(Node("edge1","edge",800.0,1200.0,4.0,12.0,40.0))
    pool.add_node(Node("edge2","edge",600.0,900.0,4.0,10.0,35.0))
    pool.add_node(Node("edge3","edge",450.0,700.0,3.5,9.0,30.0))
    pool.add_node(Node("cloud1","cloud",2500.0,6000.0,20.0,60.0,None))
    net = NetworkModel()
    net.set_link("iot","edge1",Link(80.0,12.0,0.01,1.0))
    net.set_link("iot","edge2",Link(60.0,15.0,0.01,1.2))
    net.set_link("iot","edge3",Link(40.0,18.0,0.02,1.5))
    net.set_link("iot","cloud1",Link(30.0,60.0,0.01,2.0))
    env = EdgeCloudEnv(pool, net, SLAConfig(True))
    env.reset(seed)
    return env

def main():
    env = build_demo_env()
    arrivals = WorkloadGenerator(WorkloadConfig(seed=123, mode="poisson", horizon_s=5.0, lambda_per_s=3.0)).generate()
    fb = FeatureBuilder()
    lp = LatencyPredictor(1.0)
    ep = EnergyPredictor()
    rp = SLARiskPredictor(0.25)
    agg = PredictionAggregator(AggregationConfig(mode="topk", k=3, include_counts=True))
    feas_cfg = FeasibilityConfig()
    fall_cfg = FallbackConfig(mode="eft")
    rng = np.random.default_rng(999)

    for _, tsk in arrivals[:8]:
        state = env.observe_state()
        cand = fb.build_candidates(state, tsk)
        L = lp.predict(cand); E = ep.predict(cand); R = rp.predict_from_latency(cand, L)
        node_caps = {nid: float(cand[nid].node["capacity_mi_step"]) for nid in cand}
        node_budget = {nid: (None if float(cand[nid].node["energy_budget_j_step"]) < 0 else float(cand[nid].node["energy_budget_j_step"])) for nid in cand}

        proposed = str(rng.choice(list(cand.keys())))
        ok = is_feasible(feas_cfg, tsk, node_caps[proposed], node_budget[proposed], L[proposed], E[proposed], R[proposed])
        if ok:
            chosen = proposed; mode="accept"
        else:
            nf = feasible_set(feas_cfg, tsk, node_caps, node_budget, L, E, R)
            if nf:
                chosen = repair_action(tsk, nf, L, E, R, 1.0, 1.0, 1.0); mode="repair"
            else:
                chosen = fallback_action(fall_cfg, tsk, list(cand.keys()), L, E, R); mode="fallback"

        _, res, _, _ = env.step(Task(tsk.task_id, tsk.c_mi, tsk.d_s, tsk.s_mb, tsk.p), chosen)
        print(f"{tsk.task_id} -> {chosen} [{mode}] lat={res.latency_s:.3f}s E={res.energy_j:.2f}J viol={res.violation}")

        kappa = agg.aggregate(L, E, R, 1.0, 1.0, 1.0)
        print("kappa_dim:", kappa.shape[0])

if __name__ == "__main__":
    main()
