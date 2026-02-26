# TriSchedRL

> **A Guarded Deep Reinforcement Learning Framework for SLA Latency and Energy-Aware IoT Task Scheduling in Edge–Cloud Environments**

---

## Overview

TriSchedRL is a research-driven implementation of a Guarded Deep Reinforcement Learning (DRL) framework designed to optimize SLA latency, energy consumption, and resource utilization for IoT task scheduling across heterogeneous edge–cloud infrastructures.

This repository contains the complete modular implementation, experimental pipeline, evaluation scripts, and reproducible configurations used in the associated SCI research paper.

---

## Key Features

-  Guarded Deep Reinforcement Learning Scheduler
-  SLA-Aware Policy Learning
-  Energy and Latency Joint Optimization
-  Edge–Cloud Continuum Simulation
-  Modular and Reproducible Research Pipeline
-  Ready for Academic Benchmarking and Extensions

---

## Research Motivation

Modern IoT environments generate heterogeneous workloads requiring real-time processing. Traditional schedulers struggle to:

- Guarantee SLA constraints
- Adapt to dynamic network states
- Balance energy vs. latency trade-offs

**TriSchedRL introduces:**

| Innovation | Description |
|---|---|
| **Guard Module** | Prevents unsafe scheduling decisions |
| **Tri-Objective Optimization** | Latency + Energy + Reliability |
| **Adaptive RL Policy** | Learns optimal offloading and scheduling behavior |

---

## System Architecture

| Module | Description |
|---|---|
| `src/env` | Edge–cloud environment simulation |
| `src/agents` | Actor-critic policy network and replay buffer |
| `src/guard` | SLA guard with feasibility checking and repair |
| `src/meta` | Meta-controller for adaptive policy signals |
| `src/predictors` | Latency, energy, and SLA risk predictors |
| `src/baselines` | Baseline scheduling algorithms for comparison |
| `src/evaluation` | Metrics logging and performance evaluation |
| `src/workloads` | IoT task generation and workload modeling |

---

## Repository Structure

```
TriSchedRL/
│
├── README.md
├── requirements.txt
│
├── configs/
│   ├── agent.yaml
│   ├── env.yaml
│   ├── meta.yaml
│   └── workload.yaml
│
├── scripts/
│   ├── train_trischedrl.py
│   ├── evaluate_and_plot.py
│   ├── run_baselines.py
│   ├── run_all.sh
│   └── smoke_test_guard.py
│
├── src/
│   ├── agents/
│   │   ├── actor_critic.py
│   │   ├── networks.py
│   │   ├── replay_buffer.py
│   │   ├── state_vectorizer.py
│   │   └── __init__.py
│   │
│   ├── baselines/
│   │   ├── eft.py
│   │   ├── fixed_weight_sum.py
│   │   ├── least_energy.py
│   │   ├── minmin_maxmin.py
│   │   └── __init__.py
│   │
│   ├── env/
│   │   ├── edgecloud_env.py
│   │   ├── network.py
│   │   ├── resources.py
│   │   ├── sla.py
│   │   └── __init__.py
│   │
│   ├── evaluation/
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   └── __init__.py
│   │
│   ├── guard/
│   │   ├── feasibility.py
│   │   ├── fallback.py
│   │   ├── repair.py
│   │   └── __init__.py
│   │
│   ├── meta/
│   │   ├── meta_controller.py
│   │   ├── signals.py
│   │   └── __init__.py
│   │
│   ├── predictors/
│   │   ├── aggregator.py
│   │   ├── energy_predictor.py
│   │   ├── feature_builder.py
│   │   ├── latency_predictor.py
│   │   ├── sla_risk_predictor.py
│   │   └── __init__.py
│   │
│   └── workloads/
│       ├── generators.py
│       ├── task.py
│       └── __init__.py
│
└── tests/
    ├── test_env_step.py
    └── test_guard.py
```

---

## Installation

### Step 1 — Clone Repository

```bash
git clone https://github.com/<your-username>/TriSchedRL.git
cd TriSchedRL
```

### Step 2 — Create Environment

```bash
conda create -n trischedrl python=3.10
conda activate trischedrl
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Framework

### Train the Guarded RL Scheduler

```bash
python scripts/train_trischedrl.py
```

### Evaluate and Plot Results

```bash
python scripts/evaluate_and_plot.py
```

### Run Baseline Comparisons

```bash
python scripts/run_baselines.py
```

### Run Full Pipeline

```bash
bash scripts/run_all.sh
```

### Smoke Test the Guard Module

```bash
python scripts/smoke_test_guard.py
```

---

## Configuration

All hyperparameters and settings are managed via YAML files in the `configs/` directory:

| File | Purpose |
|---|---|
| `agent.yaml` | RL agent hyperparameters (learning rate, batch size, etc.) |
| `env.yaml` | Edge–cloud environment settings |
| `meta.yaml` | Meta-controller configuration |
| `workload.yaml` | IoT workload generation parameters |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| SLA Latency Violation Rate | Percentage of tasks violating SLA deadlines |
| Energy Consumption | Total energy used across edge–cloud nodes |
| Task Completion Time | End-to-end task processing latency |
| Throughput | Number of tasks completed per time unit |
| Resource Utilization | CPU/memory usage across nodes |
| Reward Convergence | RL training stability over episodes |

---

## Baselines

| Baseline | File | Description |
|---|---|---|
| EFT | `eft.py` | Earliest Finish Time heuristic |
| Min-Min / Max-Min | `minmin_maxmin.py` | Classic task-to-resource mapping |
| Least Energy | `least_energy.py` | Energy-greedy assignment |
| Fixed Weight Sum | `fixed_weight_sum.py` | Static multi-objective weighted sum |

---

## Testing

```bash
python -m pytest tests/
```

| Test | Coverage |
|---|---|
| `test_env_step.py` | Environment step logic and state transitions |
| `test_guard.py` | Guard module feasibility and repair checks |


---

## Requirements

```
Python >= 3.10
torch
numpy
pandas
gymnasium
matplotlib
pyyaml
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

---

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

