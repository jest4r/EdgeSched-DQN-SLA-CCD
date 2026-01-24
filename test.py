import torch
from env import TaskSchedulingEnv
from model import DQN

def test_model(model, num_episodes=10, render=False):
    """
    Test the trained DQN model on the environment
    Args:
        model: Trained DQN model
        num_episodes: Number of episodes to run
        render: Whether to render the environment
    """
    env = TaskSchedulingEnv()
    total_reward = 0
    
    for episode in range(num_episodes):
        state = env.reset().flatten()
        state = torch.tensor(state, dtype=torch.float32)
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                q_values = model(state)
                action = torch.argmax(q_values).item()
            
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            next_state = torch.tensor(next_state, dtype=torch.float32)
            
            state = next_state
            episode_reward += reward
            
            if render:
                env.render()
        
        total_reward += episode_reward
        print(f"Episode {episode + 1}, Reward: {episode_reward}")
    
    avg_reward = total_reward / num_episodes
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward