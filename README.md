### EdgeSched-DQN-SLA
# “SLA-Aware Deep Reinforcement Learning for Adaptive Edge–Cloud Task Scheduling.”

The project implements an SLA-aware, risk-sensitive Deep Reinforcement Learning (DRL) scheduler for dynamic task allocation in edge–cloud environments. The scheduling problem is formulated as a sequential decision-making process where an agent learns to minimise SLA violations while optimising latency, task completion rate, and energy efficiency.

A key contribution is the integration of an explicit SLA Violation Risk Score (SVRS) into the DRL state and reward design, enabling proactive and safe scheduling decisions.

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

│                 # Notebook for:

│                 # - Running experiments

│                 # - Showing outputs

│                 # - Graphs and results

│                 # - Step-by-step explanation

│                 # - Paper figures

│
├── requirements.txt   # Python dependencies

└── README.md          # Project documentation


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


# 🧪 Simulation Scope

All results are obtained using a high-fidelity simulation environment as described in the paper.
The simulation models:

Edge–cloud nodes

Heterogeneous task workloads

SLA constraints (deadline, energy, priority tiers)

Dynamic task arrivals

Real-world deployment and testbed validation are considered future work.


# 📌 Configuration

Simulation parameters such as:

Workload intensity

SLA thresholds

Service tiers (Gold/Silver/Bronze)

Training hyperparameters

can be modified in:

utils.py / config section (if defined)
