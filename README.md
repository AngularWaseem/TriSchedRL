TriSchedRL

TriSchedRL is a guarded deep reinforcement learning framework for SLA-, latency-, and energy-aware IoT task scheduling in edge–cloud environments.

It integrates:

A custom Edge–Cloud simulation environment

Predictive models for latency, energy, and SLA risk

A guard layer for feasibility and fallback control

Reinforcement learning agents for intelligent scheduling

Baseline schedulers for comparative evaluation

The framework is designed for research and experimentation in multi-objective, constraint-aware scheduling under realistic edge–cloud constraints.

Project Structure
TriSchedRL/
│
├── configs/              # Configuration files (agent, env, workload, meta)
├── scripts/              # Training, evaluation, smoke tests
├── src/
│   ├── agents/           # Actor-Critic and supporting components
│   ├── baselines/        # Heuristic schedulers (EFT, MinMin, etc.)
│   ├── env/              # Edge–Cloud simulation environment
│   ├── evaluation/       # Metrics and logging utilities
│   ├── guard/            # Feasibility, repair, fallback mechanisms
│   ├── meta/             # Meta-controller and signals
│   ├── predictors/       # Latency, energy, SLA risk predictors
│   └── workloads/        # Task generators and workload definitions
│
└── tests/                # Unit tests
Installation

Clone the repository and install dependencies:

pip install -r requirements.txt
Smoke Test

To verify that the environment and guard layer are functioning correctly:

python scripts/smoke_test_guard.py

If the smoke test runs successfully, the environment, guard logic, and core components are properly configured.

Running Experiments
Train TriSchedRL Agent
python scripts/train_trischedrl.py
Run Baselines
python scripts/run_baselines.py
Evaluate and Plot Results
python scripts/evaluate_and_plot.py
Core Components
Environment

Edge–cloud resource modeling

Network latency simulation

SLA constraints

Multi-objective reward shaping

Predictors

Energy consumption predictor

Latency predictor

SLA violation risk predictor

Feature aggregation pipeline

Guard Layer

Feasibility checks

Action repair

Fallback scheduling strategies

The guard ensures constraint-safe decisions before action execution.

Agents

Actor–Critic architecture

Replay buffer support

State vectorization

Baseline Algorithms

Included baselines for comparison:

Earliest Finish Time (EFT)

Least Energy

Fixed Weighted Sum

MinMin / MaxMin

Testing

Run unit tests using:

pytest tests/
