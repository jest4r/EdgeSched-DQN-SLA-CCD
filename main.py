import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, defaultdict
import random
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# Constants from your paper
NUM_EDGE_NODES = 5
NUM_CLOUD_NODES = 1
MAX_QUEUE_LENGTH = 100
TASK_TYPES = ['lightweight', 'moderate', 'heavy']
SLA_TIERS = ['gold', 'silver', 'bronze']
STATE_DIM = 15  # Based on your state representation in Eq. 2

class Task:
    def __init__(self, task_type: str, sla_tier: str):
        self.type = task_type
        self.sla_tier = sla_tier
        self.deadline = self._generate_deadline()
        self.priority = self._generate_priority()
        self.energy_budget = self._generate_energy_budget()
        self.estimated_exec_time = self._generate_estimated_time()
        self.assigned_node = None
        self.start_time = None
        self.completion_time = None
        self.energy_used = 0.0
        
    def _generate_deadline(self) -> float:
        """Generate deadline based on task type and SLA tier"""
        if self.sla_tier == 'gold':
            return np.random.uniform(20, 60)  # ms
        elif self.sla_tier == 'silver':
            return np.random.uniform(60, 100)
        else:  # bronze
            return np.random.uniform(100, 150)
            
    def _generate_priority(self) -> int:
        """Generate priority score (1-5) based on SLA tier"""
        if self.sla_tier == 'gold':
            return np.random.choice([4, 5])
        elif self.sla_tier == 'silver':
            return np.random.choice([2, 3])
        else:
            return 1
            
    def _generate_energy_budget(self) -> float:
        """Generate energy budget based on task type"""
        if self.type == 'lightweight':
            return np.random.normal(0.2, 0.05)
        elif self.type == 'moderate':
            return np.random.normal(0.3, 0.1)
        else:  # heavy
            return np.random.normal(0.5, 0.15)
            
    def _generate_estimated_time(self) -> float:
        """Generate estimated execution time based on task type"""
        if self.type == 'lightweight':
            return np.random.uniform(5, 20)
        elif self.type == 'moderate':
            return np.random.uniform(20, 50)
        else:  # heavy
            return np.random.uniform(50, 100)

class Node:
    def __init__(self, node_type: str, node_id: int):
        self.type = node_type  # 'edge' or 'cloud'
        self.id = node_id
        self.queue = deque(maxlen=MAX_QUEUE_LENGTH)
        self.current_load = 0.0  # 0-1 scale
        self.energy_efficiency = np.random.uniform(0.8, 1.2)  # Efficiency factor
        self.processing_power = 1.0 if node_type == 'cloud' else np.random.uniform(0.5, 0.8)
        self.violation_history = defaultdict(int)  # Track violations by task type
        self.total_tasks_processed = 0
        
    def add_task(self, task: Task) -> bool:
        """Add task to node queue if space available"""
        if len(self.queue) < MAX_QUEUE_LENGTH:
            self.queue.append(task)
            task.assigned_node = self
            self.current_load = min(1.0, self.current_load + 0.05)  # Simulate load increase
            return True
        return False
        
    def process_tasks(self, current_time: float) -> List[Task]:
        """Process tasks in queue and return completed tasks"""
        completed_tasks = []
        if not self.queue:
            self.current_load = max(0.0, self.current_load - 0.01)  # Decay load
            return completed_tasks
            
        # Process tasks based on node processing power
        processing_capacity = self.processing_power * (1 + random.uniform(-0.1, 0.1))
        
        for task in list(self.queue):
            # Calculate actual processing time considering node load
            actual_time = task.estimated_exec_time * (1 + self.current_load) / processing_capacity
            
            if task.start_time is None:
                task.start_time = current_time
                
            if current_time - task.start_time >= actual_time:
                # Task completed
                task.completion_time = current_time
                task.energy_used = task.energy_budget * self.energy_efficiency
                completed_tasks.append(task)
                self.queue.remove(task)
                self.total_tasks_processed += 1
                
                # Check for SLA violation
                if task.completion_time - task.start_time > task.deadline:
                    self.violation_history[task.type] += 1
            else:
                break
                
        return completed_tasks
        
    def get_violation_rate(self, task_type: str) -> float:
        """Get historical violation rate for a task type"""
        total = sum(self.violation_history.values())
        if total == 0:
            return 0.0
        return self.violation_history[task_type] / total
        
    def get_svrs(self, task: Task) -> float:
        """Calculate SLA Violation Risk Score (Eq. 1 in your paper)"""
        # Coefficients from your paper (α, β, γ)
        alpha, beta, gamma = 0.5, 0.3, 0.2
        
        # Time risk component
        time_risk = task.estimated_exec_time / task.deadline
        
        # Queue risk component
        queue_risk = len(self.queue) / MAX_QUEUE_LENGTH
        
        # Historical violation risk
        hist_violation = self.get_violation_rate(task.type)
        
        # Calculate SVRS (Eq. 1)
        svrs = alpha * time_risk + beta * queue_risk + gamma * hist_violation
        return min(max(svrs, 0.0), 1.0)  # Clamp between 0 and 1

class TaskSchedulingEnv:
    def __init__(self):
        self.edge_nodes = [Node('edge', i) for i in range(NUM_EDGE_NODES)]
        self.cloud_nodes = [Node('cloud', i) for i in range(NUM_CLOUD_NODES)]
        self.all_nodes = self.edge_nodes + self.cloud_nodes
        self.current_time = 0.0
        self.time_step = 1.0  # ms
        self.task_counter = 0
        self.completed_tasks = []
        self.violated_tasks = []
        self.total_reward = 0.0
        self.episode_length = 500  # Number of time steps per episode
        self.task_arrival_rate = 10  # Tasks per second (Poisson process)
        self.current_task = None  # Track the current task being scheduled
        
        # SLA reward weights from your paper (λ1, λ2, λ3)
        self.reward_weights = {
            'gold': [1.0, 2.0, 0.5],
            'silver': [0.8, 1.5, 0.7],
            'bronze': [0.5, 1.0, 1.0]
        }
        
    def reset(self) -> np.ndarray:
        """Reset environment for new episode"""
        self.edge_nodes = [Node('edge', i) for i in range(NUM_EDGE_NODES)]
        self.cloud_nodes = [Node('cloud', i) for i in range(NUM_CLOUD_NODES)]
        self.all_nodes = self.edge_nodes + self.cloud_nodes
        self.current_time = 0.0
        self.task_counter = 0
        self.completed_tasks = []
        self.violated_tasks = []
        self.total_reward = 0.0
        self.current_task = None
        
        # Return initial state
        return self._get_state()
        
    def _get_state(self) -> np.ndarray:
        """Construct SLA-aware state vector (Eq. 2 in your paper)"""
        state = []
        
        # System-level metrics
        for node in self.all_nodes:
            state.extend([
                node.current_load,  # L_t
                len(node.queue) / MAX_QUEUE_LENGTH,  # U_t
                node.processing_power  # Q_t
            ])
            
        # Current task information
        if self.current_task:
            state.extend([
                self.current_task.priority / 5.0,  # P_i (normalized priority)
                self.current_task.deadline / 150.0,  # D_i (normalized deadline)
                min(max(self.current_task.estimated_exec_time / 100.0, 0.0), 1.0),  # Normalized exec time
                1.0  # B_t (bandwidth placeholder)
            ])
        else:
            state.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(state, dtype=np.float32)
        
    def _generate_task(self) -> Task:
        """Generate a new task with random type and SLA tier"""
        task_type = random.choice(TASK_TYPES)
        sla_tier = random.choice(SLA_TIERS)
        return Task(task_type, sla_tier)
        
    def _calculate_reward(self, task: Task) -> float:
        """Calculate tiered reward based on SLA compliance (Eq. 4-5 in your paper)"""
        # Get weights based on SLA tier
        lambda1, lambda2, lambda3 = self.reward_weights[task.sla_tier]
        
        # Check for SLA violation
        completion_time = task.completion_time - task.start_time
        violated = (completion_time > task.deadline) or (task.energy_used > task.energy_budget)
        
        # Base reward for scheduling
        R_base = 1.0 if not violated else 0.0
        
        # Success indicator
        S_i = 1.0 if not violated else 0.0
        
        # Violation penalty
        V_i = 1.0 if violated else 0.0
        
        # Energy cost
        E_i = task.energy_used / task.energy_budget  # Normalized energy usage
        
        # Calculate reward (Eq. 4)
        reward = R_base + lambda1 * S_i - lambda2 * V_i - lambda3 * E_i
        
        # Scale reward based on priority
        reward *= (task.priority / 5.0)
        
        return reward
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one time step in the environment"""
        # Process node queues
        for node in self.all_nodes:
            completed = node.process_tasks(self.current_time)
            self.completed_tasks.extend(completed)
            
            # Check for SLA violations
            for task in completed:
                if (task.completion_time - task.start_time > task.deadline) or \
                   (task.energy_used > task.energy_budget):
                    self.violated_tasks.append(task)
                    
        # Generate new tasks (Poisson process)
        num_new_tasks = np.random.poisson(self.task_arrival_rate * (self.time_step / 1000))
        new_tasks = [self._generate_task() for _ in range(num_new_tasks)]
        
        # Assign tasks based on action
        # Action is an integer representing node selection (0-5 for 5 edge + 1 cloud)
        if 0 <= action < len(self.all_nodes) and new_tasks:
            selected_node = self.all_nodes[action]
            for task in new_tasks:
                self.current_task = task  # Track current task for state representation
                if not selected_node.add_task(task):
                    # If queue is full, task is dropped (count as violation)
                    self.violated_tasks.append(task)
        
        # Calculate reward based on completed tasks
        reward = 0.0
        for task in self.completed_tasks[-num_new_tasks:]:  # Only consider newly completed tasks
            reward += self._calculate_reward(task)
            
        self.total_reward += reward
        
        # Update time
        self.current_time += self.time_step
        
        # Check if episode is done
        done = self.current_time >= self.episode_length
        
        # Get next state
        next_state = self._get_state()
        
        # Info dictionary
        info = {
            'tasks_completed': len(self.completed_tasks),
            'violations': len(self.violated_tasks),
            'avg_latency': np.mean([t.completion_time - t.start_time for t in self.completed_tasks]) if self.completed_tasks else 0.0,
            'avg_energy': np.mean([t.energy_used for t in self.completed_tasks]) if self.completed_tasks else 0.0
        }
        
        return next_state, reward, done, info

class HybridConvGRUDQN(nn.Module):
    """SLA-aware DQN with Conv1D and GRU layers (Figure 2 in your paper)"""
    def __init__(self, state_size: int, action_size: int):
        super(HybridConvGRUDQN, self).__init__()
        
        # Conv1D block for local feature extraction (Eq. 6)
        self.conv1d = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv1d_bn = nn.BatchNorm1d(32)
        
        # GRU layer for temporal patterns (Eq. 7)
        self.gru = nn.GRU(32, 64, batch_first=True)
        
        # Shared fully connected layer
        self.shared_fc = nn.Linear(64, 128)
        
        # Actor branches (one for each action component)
        self.node_head = nn.Linear(128, action_size)  # Output size matches action space
        
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Reshape for Conv1D (batch_size, channels, sequence_length)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # Conv1D block
        x = self.relu(self.conv1d_bn(self.conv1d(x)))
        x = x.permute(0, 2, 1)  # Prepare for GRU
        
        # GRU layer
        _, h_n = self.gru(x)
        h_n = h_n.squeeze(0)
        
        # Shared fully connected
        shared = self.relu(self.shared_fc(h_n))
        
        # Node selection head
        node_logits = self.node_head(shared)
        
        return node_logits

class SLAwareReplayBuffer:
    """Enhanced replay buffer with priority sampling based on SLA violations"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool, violation: bool):
        """Add experience to buffer with priority"""
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        
        # Higher priority for experiences with SLA violations
        priority = self.max_priority * 2 if violation else self.max_priority
        
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
        
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch with priority"""
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return torch.stack(states), torch.LongTensor(actions), torch.FloatTensor(rewards), torch.stack(next_states), torch.FloatTensor(dones)
        
    def __len__(self) -> int:
        return len(self.buffer)

class EdgeSchedDQNSLA:
    """Complete SLA-aware DRL scheduler implementation"""
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        
        # Main and target networks
        self.policy_net = HybridConvGRUDQN(state_size, action_size)
        self.target_net = HybridConvGRUDQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # Replay buffer
        self.replay_buffer = SLAwareReplayBuffer(10000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        self.batch_size = 64
        self.target_update = 10
        
        # Training stats
        self.losses = []
        self.rewards = []
        self.epsilons = []
        self.violation_rates = []
        
    def select_action(self, state: np.ndarray, env: TaskSchedulingEnv) -> int:
        """Epsilon-greedy action selection with SLA-aware pruning"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        if random.random() < self.epsilon:
            # Random exploration
            return random.randint(0, self.action_size - 1)
        else:
            # Get Q-values from policy network
            with torch.no_grad():
                node_logits = self.policy_net(state_tensor)
                
            # SLA-aware action pruning (Eq. 3 in your paper)
            valid_actions = []
            q_values = []
            
            for action in range(self.action_size):
                node = env.all_nodes[action]
                
                # Get current task (last generated)
                if env.current_task:
                    task = env.current_task
                    svrs = node.get_svrs(task)
                    
                    # Check if action is valid (SVRS < threshold and load < threshold)
                    if svrs < 0.7 and node.current_load < 0.8:  # Thresholds from your paper
                        valid_actions.append(action)
                        q_values.append(node_logits[0, action].item())
            
            if valid_actions:
                # Select action with highest Q-value among valid actions
                return valid_actions[np.argmax(q_values)]
            else:
                # Fallback to least loaded node if no valid actions
                loads = [node.current_load for node in env.all_nodes]
                return np.argmin(loads)
                
    def update_model(self) -> float:
        """Update policy network using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
            
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Get current Q values
        node_logits = self.policy_net(states)
        current_q = node_logits.gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target network
        with torch.no_grad():
            next_node_logits = self.target_net(next_states)
            next_q = next_node_logits.max(1)[0].unsqueeze(1)
            expected_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))
        
        # Compute loss
        loss = nn.MSELoss()(current_q, expected_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def train(self, env: TaskSchedulingEnv, episodes: int = 500) -> None:
        """Train the DRL agent"""
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            violations = 0
            done = False
            
            while not done:
                # Select action
                action = self.select_action(state, env)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Track violations
                violations += info['violations']
                
                # Store experience in replay buffer
                violation_occurred = info['violations'] > 0
                self.replay_buffer.push(state, action, reward, next_state, done, violation_occurred)
                
                # Update model
                loss = self.update_model()
                if loss > 0:
                    self.losses.append(loss)
                
                total_reward += reward
                state = next_state
            
            # Update target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Save stats
            self.rewards.append(total_reward)
            self.epsilons.append(self.epsilon)
            violation_rate = violations / info['tasks_completed'] if info['tasks_completed'] > 0 else 0.0
            self.violation_rates.append(violation_rate)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.rewards[-10:]) if len(self.rewards) >= 10 else total_reward
                avg_violation = np.mean(self.violation_rates[-10:]) if len(self.violation_rates) >= 10 else violation_rate
                print(f"Episode {episode + 1}, Reward: {total_reward:.1f}, Avg Reward: {avg_reward:.1f}, "
                      f"Violation Rate: {violation_rate:.2%}, Epsilon: {self.epsilon:.3f}")
    
    def evaluate(self, env: TaskSchedulingEnv, episodes: int = 10) -> Dict:
        """Evaluate the trained agent"""
        results = {
            'rewards': [],
            'violation_rates': [],
            'latencies': [],
            'energy_usage': [],
            'task_completion': []
        }
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            violations = 0
            done = False
            
            while not done:
                # Select action (no exploration)
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    node_logits = self.policy_net(state_tensor)
                    action = torch.argmax(node_logits).item()
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                total_reward += reward
                violations += info['violations']
                state = next_state
            
            # Record results
            violation_rate = violations / info['tasks_completed'] if info['tasks_completed'] > 0 else 0.0
            results['rewards'].append(total_reward)
            results['violation_rates'].append(violation_rate)
            results['latencies'].append(info['avg_latency'])
            results['energy_usage'].append(info['avg_energy'])
            results['task_completion'].append(info['tasks_completed'])
            
            print(f"Evaluation Episode {episode + 1}, Reward: {total_reward:.1f}, "
                  f"Violation Rate: {violation_rate:.2%}, Latency: {info['avg_latency']:.1f}ms, "
                  f"Energy: {info['avg_energy']:.3f}J, Tasks: {info['tasks_completed']}")
        
        return results
        
    def plot_training(self) -> None:
        """Plot training statistics"""
        plt.figure(figsize=(15, 5))
        
        # Plot rewards
        plt.subplot(1, 3, 1)
        plt.plot(self.rewards, label='Episode Reward')
        plt.plot(np.convolve(self.rewards, np.ones(10)/10, mode='valid'), label='Moving Avg (10)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        plt.legend()
        plt.grid(True)
        
        # Plot violation rates
        plt.subplot(1, 3, 2)
        plt.plot(self.violation_rates, label='Violation Rate')
        plt.plot(np.convolve(self.violation_rates, np.ones(10)/10, mode='valid'), label='Moving Avg (10)')
        plt.xlabel('Episode')
        plt.ylabel('Violation Rate')
        plt.title('SLA Violation Rates')
        plt.legend()
        plt.grid(True)
        
        # Plot epsilon decay
        plt.subplot(1, 3, 3)
        plt.plot(self.epsilons)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate Decay')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    # Initialize environment and agent
    env = TaskSchedulingEnv()
    state_size = STATE_DIM
    action_size = len(env.all_nodes)  # Number of nodes to choose from
    
    agent = EdgeSchedDQNSLA(state_size, action_size)
    
    # Train the agent
    print("Training SLA-aware DQN scheduler...")
    agent.train(env, episodes=500)
    
    # Plot training results
    agent.plot_training()
    
    # Evaluate the agent
    print("\nEvaluating trained scheduler...")
    results = agent.evaluate(env, episodes=10)
    
    # Print summary statistics
    avg_violation = np.mean(results['violation_rates']) * 100
    avg_latency = np.mean(results['latencies'])
    avg_energy = np.mean(results['energy_usage'])
    avg_completion = np.mean(results['task_completion'])
    
    print("\nFinal Performance Summary:")
    print(f"- Average SLA Violation Rate: {avg_violation:.1f}%")
    print(f"- Average Task Latency: {avg_latency:.1f}ms")
    print(f"- Average Energy Usage: {avg_energy:.3f}J")
    print(f"- Average Tasks Completed: {avg_completion:.1f}")
    
    # Save the trained model
    torch.save(agent.policy_net.state_dict(), "edgesched_dqn_sla.pth")
    print("Model saved successfully!")

if __name__ == "__main__":
    main()