import gymnasium as gym
from reinforce_agent import ReinforceAgent
from reinforce_trainer import ReinforceTrainer
import numpy as np
import matplotlib.pyplot as plt

# WRITE YOUR CODE HERE
# Hyperparameters for Walker2d
n_episodes = 5000
hidden_size1 = 64
hidden_size2 = 64
learning_rate = 1e-4
gamma = 0.99

# Environment setup
env = gym.make('Walker2d-v4')
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Agent instantiation
agent = ReinforceAgent(obs_dim, action_dim, hidden_size1, hidden_size2, learning_rate, gamma)

# Train the agent
trainer = ReinforceTrainer(env, agent, n_episodes)
episode_returns = trainer.train()

# Plot the learning curve
plt.plot(np.arange(len(episode_returns)), 
episode_returns)
plt.title('Episode returns')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.savefig('learning_curve.png')
plt.show()
