import torch
import numpy as np
from env import TaskSchedulingEnv
from model import DQN
from collections import deque
import random

def train_model(episodes=1000, batch_size=32, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
    env = TaskSchedulingEnv()
    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.n
    
    model = DQN(state_size, action_size)
    optimizer = torch.optim.Adam(model.parameters())
    memory = deque(maxlen=10000)
    
    rewards = []
    
    for episode in range(episodes):
        state = env.reset().flatten()
        state = torch.tensor(state, dtype=torch.float32)
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() <= epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = torch.argmax(q_values).item()
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            memory.append((state, action, reward, next_state, done))
            
            state = torch.tensor(next_state, dtype=torch.float32)
            total_reward += reward
            
            # Train on batch
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                # Training logic here...
                # (Add your DQN training code)
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")
    
    return model, rewards