import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from main import EdgeSchedDQNSLA, TaskSchedulingEnv, STATE_DIM

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CFG = {
    'episode_length': 500,
    'task_arrival_rate': 10,
    'train_episodes': 20,
    'eval_episodes': 5,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay': 0.995,
}


def run_policy(policy, env, agent=None, episodes=20):
    metrics = {'reward': [], 'violation': [], 'latency': [], 'completion': [], 'energy': []}
    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        violations = 0
        while not done:
            if policy == 'FIFO':
                action = min(range(len(env.all_nodes)), key=lambda i: env.all_nodes[i].current_load)
            elif policy == 'Round Robin':
                if not hasattr(env, '_rr_idx'):
                    env._rr_idx = 0
                action = env._rr_idx % len(env.all_nodes)
                env._rr_idx += 1
            elif policy == 'EAG':
                action = min(range(len(env.all_nodes)), key=lambda i: (env.all_nodes[i].current_load, len(env.all_nodes[i].queue)))
            elif policy == 'DRL-NSLA':
                with torch.no_grad():
                    q = agent.policy_net(torch.FloatTensor(state).unsqueeze(0))
                action = int(torch.argmax(q).item())
            elif policy == 'SLA-DRL':
                action = agent.select_action(state, env)
            else:
                raise ValueError(policy)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            violations += info.get('violations', 0)
            state = next_state

        completed = info.get('tasks_completed', 0)
        metrics['reward'].append(total_reward)
        metrics['violation'].append(info.get('violation_rate', 0.0) * 100.0)
        metrics['latency'].append(info.get('avg_latency', 0.0))
        metrics['completion'].append(info.get('completion_rate', 0.0) * 100.0)
        metrics['energy'].append(info.get('avg_energy', 0.0))

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def train_agent(env: TaskSchedulingEnv, cfg: dict) -> EdgeSchedDQNSLA:
    agent = EdgeSchedDQNSLA(STATE_DIM, len(env.all_nodes))
    agent.epsilon = cfg['epsilon_start']
    print('Training DRL agent...')
    for ep in range(cfg['train_episodes']):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, env)
            next_state, reward, done, info = env.step(action)
            violation_occurred = info.get('violations', 0) > 0
            agent.replay_buffer.push(state, action, reward, next_state, done, violation_occurred)
            agent.update_model()
            state = next_state
        agent.epsilon = max(cfg['epsilon_end'], agent.epsilon * agent.epsilon_decay)
        if ep % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        if (ep + 1) % 5 == 0:
            print(f'  trained episode {ep + 1}/{cfg["train_episodes"]}')
    return agent


def main() -> None:
    env = TaskSchedulingEnv()
    agent = train_agent(env, CFG)
    policy_names = ['FIFO', 'Round Robin', 'EAG', 'DRL-NSLA', 'SLA-DRL']
    results = {}
    print('Evaluating policies...')
    for name in policy_names:
        if name in {'DRL-NSLA', 'SLA-DRL'}:
            results[name] = run_policy(name, env, agent=agent, episodes=CFG['eval_episodes'])
        else:
            results[name] = run_policy(name, env, episodes=CFG['eval_episodes'])
        print(f'  {name}: {results[name]}')

    print('Generating figure...')

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    strategies = list(results.keys())
    viol = [results[p]['violation'] for p in strategies]
    lat = [results[p]['latency'] for p in strategies]
    comp = [results[p]['completion'] for p in strategies]
    reward = [results[p]['reward'] for p in strategies]

    axes[0, 0].bar(strategies, viol, color='gold')
    axes[0, 0].set_title('SLA Violation Rate (%)')
    axes[0, 0].set_ylabel('%')

    axes[0, 1].bar(strategies, lat, color='cornflowerblue')
    axes[0, 1].set_title('Average Latency (ms)')
    axes[0, 1].set_ylabel('ms')

    axes[1, 0].bar(strategies, comp, color='lightgreen')
    axes[1, 0].set_title('Task Completion Rate (%)')
    axes[1, 0].set_ylabel('%')

    axes[1, 1].bar(strategies, reward, color='lightcoral')
    axes[1, 1].set_title('Cumulative Reward')
    axes[1, 1].set_ylabel('Reward')

    for ax in axes.flat:
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.setp(ax.get_xticklabels(), rotation=20)

    fig.tight_layout()
    output_path = Path('paper_benchmark_figure.png')
    fig.savefig(output_path, dpi=200)
    print(f'Figure saved to: {output_path.resolve()}')
    print('Benchmark figure generation completed.')


if __name__ == '__main__':
    main()
