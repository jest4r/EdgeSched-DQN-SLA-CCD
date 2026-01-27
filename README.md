# EdgeSched-DQN-SLA
# SLA-Aware DRL Task Scheduler for Edge–Cloud Environments
This repository provides the Python implementation of the framework proposed in the paper
“SLA-Aware Deep Reinforcement Learning for Adaptive Edge–Cloud Task Scheduling.”
Overview
The code implements an SLA-aware, risk-sensitive deep reinforcement learning (DRL) scheduler for
dynamic task allocation in edge–cloud environments. The scheduling problem is formulated as a
sequential decision-making process, where the agent learns to minimise SLA violations while
optimising latency, task completion, and energy efficiency. A key contribution is the integration of an
explicit SLA Violation Risk Score (SVRS) into the DRL state and reward design, enabling proactive and
safe scheduling decisions.

# Key Features
 SLA Violation Risk Score (SVRS)–based state representation
 Tier-aware SLA handling (e.g., Gold/Silver/Bronze tasks)
 Hybrid Conv1D–GRU DRL architecture
 SLA-aware reward shaping and action pruning
 Closed-loop feedback using runtime SLA monitoring
 Simulation-based, reproducible evaluation


# Code Structure
DRL_SLA_Task_Scheduler_PY/
│── env/ # Edge–cloud environment and workload simulation

│── agent/ # DRL agent (Conv1D–GRU policy, training logic)

│── reward/ # SLA-aware reward and SVRS computation

│── scheduler/ # Task scheduling and action pruning logic

│── utils/ # Configuration, logging, metrics

│── main.py # Entry point for training and evaluation

│── requirements.txt # Python dependencies

# Usage
1. Install dependencies:
pip install -r requirements.txt
2. Run training or evaluation:
python main.py
Simulation parameters (workload intensity, SLA thresholds, service tiers) can be configured in the
provided config files.

Evaluation Scope
All results are obtained via high-fidelity simulation, as described in the paper. Real-world
deployment and testbed validation are considered future work.
Reference
If you use this code, please cite the corresponding research paper.
