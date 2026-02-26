from src.workloads.task import IoTTask
from src.guard.feasibility import FeasibilityConfig, is_feasible

def test_feasible_basic():
    cfg = FeasibilityConfig(enforce_deadline=True, enforce_energy_budget=True, use_risk_threshold=False)
    t = IoTTask("t1", c_mi=100.0, d_s=2.0, s_mb=1.0, p=0)
    ok = is_feasible(cfg, t, node_capacity_mi_step=200.0, energy_budget_j_step=50.0, L_hat_s=1.0, E_hat_j=10.0, R_hat=0.2)
    assert ok
