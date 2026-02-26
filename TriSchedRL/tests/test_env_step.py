from src.env.resources import ResourcePool, Node
from src.env.network import NetworkModel, Link
from src.env.sla import SLAConfig
from src.env.edgecloud_env import EdgeCloudEnv, Task

def test_env_runs_one_step():
    pool = ResourcePool()
    pool.add_node(Node("edge1","edge",f=500.0,capacity_mi_per_step=1000.0,power_idle_w=4.0,power_dyn_w=10.0,energy_budget_j_per_step=30.0))
    net = NetworkModel()
    net.set_link("iot","edge1",Link(bandwidth_mbps=50.0,rtt_ms=10.0,loss=0.0,overhead_ms=1.0))
    env = EdgeCloudEnv(pool, net, SLAConfig(hard_deadline=True), dt_s=1.0)
    env.reset(seed=1)
    _, res, _, _ = env.step(Task("t1", c_mi=100.0, d_s=1.0, s_mb=1.0, p=0), "edge1")
    assert res.latency_s >= 0.0
