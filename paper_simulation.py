from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from main import EdgeSchedDQNSLA, TaskSchedulingEnv, STATE_DIM


@dataclass
class SimulationConfig:
    seed: int = 42
    num_edge_nodes: int = 5
    num_cloud_nodes: int = 1
    max_queue_length: int = 100
    task_arrival_rate: float = 10.0
    episode_length: int = 500
    time_step: float = 100.0
    train_episodes: int = 50
    eval_episodes: int = 10
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 64
    target_update: int = 10

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BenchmarkPolicies:
    def __init__(self, env: TaskSchedulingEnv):
        self.env = env
        self.rr_counter = 0

    def fifo(self, state: np.ndarray) -> int:
        loads = [node.current_load for node in self.env.all_nodes]
        return int(np.argmin(loads))

    def round_robin(self, state: np.ndarray) -> int:
        action = self.rr_counter % len(self.env.all_nodes)
        self.rr_counter += 1
        return action

    def eag(self, state: np.ndarray) -> int:
        candidates = []
        for idx, node in enumerate(self.env.all_nodes):
            candidates.append((node.current_load, len(node.queue), idx))
        _, _, best = min(candidates, key=lambda x: (x[0], x[1], x[2]))
        return best

    def drl_nsla(self, state: np.ndarray, agent: EdgeSchedDQNSLA) -> int:
        with torch.no_grad():
            q = agent.policy_net(torch.FloatTensor(state).unsqueeze(0))
        return int(torch.argmax(q).item())

    def sla_drl(self, state: np.ndarray, agent: EdgeSchedDQNSLA) -> int:
        return agent.select_action(state, self.env)


def evaluate_policy(policy_name: str, env: TaskSchedulingEnv, agent: EdgeSchedDQNSLA | None = None, episodes: int = 5) -> Dict[str, float]:
    policies = BenchmarkPolicies(env)
    rewards: List[float] = []
    violation_rates: List[float] = []
    latencies: List[float] = []
    completion_rates: List[float] = []
    energy_usage: List[float] = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        violations = 0
        done = False
        while not done:
            if policy_name == "FIFO":
                action = policies.fifo(state)
            elif policy_name == "Round Robin":
                action = policies.round_robin(state)
            elif policy_name == "EAG":
                action = policies.eag(state)
            elif policy_name == "DRL-NSLA":
                action = policies.drl_nsla(state, agent)
            elif policy_name == "SLA-DRL":
                action = policies.sla_drl(state, agent)
            else:
                raise ValueError(f"Unsupported policy: {policy_name}")

            next_state, reward, done, info = env.step(action)
            total_reward += reward
            violations += info.get("violations", 0)
            state = next_state

        generated = info.get("tasks_generated", 0)
        completed = info.get("tasks_completed", 0)
        violation_rate = info.get("violation_rate", 0.0)
        avg_latency = info.get("avg_latency", 0.0)
        completion_rate = info.get("completion_rate", 0.0)

        rewards.append(total_reward)
        violation_rates.append(violation_rate * 100.0)
        latencies.append(avg_latency)
        completion_rates.append(completion_rate * 100.0)
        energy_usage.append(info.get("avg_energy", 0.0))

    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_violation_rate": float(np.mean(violation_rates)),
        "avg_latency": float(np.mean(latencies)),
        "avg_completion_rate": float(np.mean(completion_rates)),
        "avg_energy": float(np.mean(energy_usage)),
    }


def train_agent(env: TaskSchedulingEnv, config: SimulationConfig) -> EdgeSchedDQNSLA:
    agent = EdgeSchedDQNSLA(STATE_DIM, len(env.all_nodes))
    agent.gamma = config.gamma
    agent.epsilon_start = config.epsilon_start
    agent.epsilon_end = config.epsilon_end
    agent.epsilon_decay = config.epsilon_decay
    agent.batch_size = config.batch_size
    agent.target_update = config.target_update
    agent.epsilon = config.epsilon_start
    for episode in range(config.train_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, env)
            next_state, reward, done, info = env.step(action)
            violation_occurred = info.get("violations", 0) > 0
            agent.replay_buffer.push(state, action, reward, next_state, done, violation_occurred)
            _ = agent.update_model()
            state = next_state
        if episode % config.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
    return agent


def build_report_text(config: SimulationConfig, results: Dict[str, Dict[str, float]]) -> str:
    lines: List[str] = []
    lines.append("# Standardized simulation configuration for paper-aligned benchmark")
    lines.append("")
    lines.append("## Objective")
    lines.append("This benchmark reconstructs the paper-style edge-cloud scheduling evaluation using the simulator logic already present in the project, while fixing seeds and standardizing workload assumptions across all policies.")
    lines.append("")
    lines.append("## Simulation configuration")
    for key, value in config.to_dict().items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append("## Benchmark policies")
    for policy_name in ["FIFO", "Round Robin", "EAG", "DRL-NSLA", "SLA-DRL"]:
        lines.append(f"- **{policy_name}**: evaluated under identical task arrivals and node topology.")
    lines.append("")
    lines.append("## Results")
    lines.append("| Policy | Avg Reward | SLA Violation Rate (%) | Avg Latency (ms) | Completion Rate (%) | Avg Energy |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for policy_name in ["FIFO", "Round Robin", "EAG", "DRL-NSLA", "SLA-DRL"]:
        result = results[policy_name]
        lines.append(
            f"| {policy_name} | {result['avg_reward']:.2f} | {result['avg_violation_rate']:.2f} | {result['avg_latency']:.2f} | {result['avg_completion_rate']:.2f} | {result['avg_energy']:.3f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("The reported paper values are target ranges for the paper comparison, while the implementation here is deliberately built as a reproducible benchmark environment. The same workload, node topology, and action constraints are used across strategies so that the relative ordering is meaningful.")
    lines.append("")
    lines.append("## Notes")
    lines.append("- The default Gym stub in env.py is not used for the final benchmark; it is a placeholder.")
    lines.append("- The custom simulator in main.py is the source of the standardized experiment logic.")
    lines.append("- For publication-quality reproduction, the same seed, agent training schedule, and workload distribution should be logged for each run.")
    return "\n".join(lines) + "\n"


def main() -> None:
    config = SimulationConfig()
    set_seed(config.seed)

    env = TaskSchedulingEnv(
        episode_length=config.episode_length,
        time_step=config.time_step,
        task_arrival_rate=config.task_arrival_rate,
    )
    results: Dict[str, Dict[str, float]] = {}

    # Train the DRL agent once under the paper-style simulator configuration.
    agent = train_agent(env, config)

    for policy_name in ["FIFO", "Round Robin", "EAG", "DRL-NSLA", "SLA-DRL"]:
        if policy_name in {"DRL-NSLA", "SLA-DRL"}:
            results[policy_name] = evaluate_policy(policy_name, env, agent=agent, episodes=config.eval_episodes)
        else:
            results[policy_name] = evaluate_policy(policy_name, env, agent=None, episodes=config.eval_episodes)

    report_text = build_report_text(config, results)
    report_path = Path(__file__).with_name("paper_simulation_report.md")
    report_path.write_text(report_text, encoding="utf-8")

    print(json.dumps({"config": config.to_dict(), "results": results}, indent=2))
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
