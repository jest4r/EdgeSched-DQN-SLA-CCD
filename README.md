# SLA-Aware DRL Task Scheduler for Edge–Cloud Environments
This repository provides the Python implementation of the framework proposed in the paper
“SLA-Aware Deep Reinforcement Learning for Adaptive Edge–Cloud Task Scheduling.”

# Overview
The code implements an SLA-aware, risk-sensitive deep reinforcement learning (DRL) scheduler for
dynamic task allocation in edge–cloud environments. The scheduling problem is formulated as a
sequential decision-making process, where the agent learns to minimise SLA violations while
optimising latency, task completion, and energy efficiency. A key contribution is the integration of an
explicit SLA Violation Risk Score (SVRS) into the DRL state and reward design, enabling proactive and
safe scheduling decisions.

# 🔹 Key Features

SLA Violation Risk Score (SVRS)–based state representation


Tier-aware SLA handling (Gold / Silver / Bronze tasks)


Hybrid Conv1D–GRU DRL architecture


SLA-aware reward shaping and action pruning


Closed-loop feedback using runtime SLA monitoring


Simulation-based and reproducible evaluation


Jupyter Notebook for experiments and visualization


# 📁 Code Structure
DRL_SLA_Task_Scheduler/

│
├── env.py        # Edge–cloud environment and workload simulation

├── model.py      # DRL model (Conv1D–GRU network architecture)

├── train.py      # Training logic for the DRL agent

├── test.py       # Testing and evaluation script

├── utils.py      # Utility functions (config, logging, metrics)

│
├── DRL_SLA_Task_Scheduler_Notebook.ipynb  

# Notebook for:

│                 # - Running experiments

│                 # - Showing outputs

│                 # - Graphs and results

│                 # - Step-by-step explanation

│                 # - Paper figures


└── README.md          # Project documentation

└── requirements.txt          # Required packages



# ⚙️ Usage

1. Train the model
python train.py

2. Test / evaluate the trained model
python test.py

3. Run experiments and visualize results

Open the notebook:

DRL_SLA_Task_Scheduler_Notebook.ipynb


This notebook is used for:

Running experiments

Plotting graphs

Showing outputs

Step-by-step explanation

Generating figures for the research paper

# Evaluation Scope
All results are obtained via high-fidelity simulation, as described in the paper. Real-world
deployment and testbed validation are considered future work.

# Reference
If you use this code, please cite the corresponding research paper.

