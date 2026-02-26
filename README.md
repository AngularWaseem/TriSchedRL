# TriSchedRL

TriSchedRL is a guarded deep reinforcement learning framework for SLA-, latency-, and energy-aware IoT task scheduling in edge–cloud environments.

## Install
```bash
pip install -r requirements.txt
```

## Smoke Test
```bash
python scripts/smoke_test_guard.py
```

## Notes
This repo contains a runnable environment, predictors, guard layer, and RL scaffolding. You can extend the placeholder scripts (`train_trischedrl.py`, `run_baselines.py`, `evaluate_and_plot.py`) with the full versions from the methodology implementation.
