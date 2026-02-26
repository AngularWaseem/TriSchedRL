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
| Environment Simulator | Models edge, fog, and cloud nodes |
| Guard Engine | Enforces SLA constraints during RL action selection |
| DRL Agent | Policy learning using deep neural networks |
| Scheduler Core | Executes decisions and monitors system metrics |
| Evaluation Engine | Computes performance indicators |

---

## Repository Structure

```
TriSchedRL/
│
├── data/
│   ├── workloads/
│   ├── configs/
│   └── synthetic_generator.py
│
├── env/
│   ├── edge_cloud_env.py
│   ├── resource_model.py
│   └── sla_constraints.py
│
├── agent/
│   ├── policy_network.py
│   ├── guarded_rl_agent.py
│   └── replay_buffer.py
│
├── scheduler/
│   ├── tri_scheduler.py
│   └── decision_engine.py
│
├── guard/
│   ├── sla_guard.py
│   └── safety_checker.py
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── hyperparams.yaml
│
├── utils/
│   ├── metrics.py
│   ├── logger.py
│   └── visualization.py
│
├── experiments/
│   ├── baseline_comparisons/
│   └── ablation_studies/
│
├── results/
│   ├── logs/
│   └── figures/
│
├── requirements.txt
├── README.md
└── LICENSE
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
python training/train.py
```

### Evaluate Performance

```bash
python training/evaluate.py
```

### Visualize Results

```bash
python utils/visualization.py
```

---

## Evaluation Metrics

The framework evaluates scheduling performance using:

| Metric | Description |
|---|---|
| SLA Latency Violation Rate | Percentage of tasks violating SLA deadlines |
| Energy Consumption | Total energy used across edge–cloud nodes |
| Task Completion Time | End-to-end task processing latency |
| Throughput | Number of tasks completed per time unit |
| Resource Utilization | CPU/memory usage across nodes |
| Reward Convergence | RL training stability over episodes |

---

## Experimental Design

TriSchedRL supports:

- Synthetic IoT workloads
- Edge–Cloud heterogeneous resources
- Dynamic arrival rates
- Baseline comparison experiments

**Baselines include:**

- Heuristic Scheduling
- Metaheuristic Optimization
- Standard DRL Scheduling

---

## Extending the Framework

You can extend TriSchedRL by:

- Adding new RL algorithms (PPO, SAC, DDPG)
- Integrating federated learning
- Introducing new SLA constraints
- Deploying on real edge devices

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

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{trischedrl,
  title={TriSchedRL: A Guarded Deep Reinforcement Learning Framework for SLA Latency and Energy Aware IoT Task Scheduling in Edge–Cloud Environments},
  journal={},
  year={2025}
}
```
