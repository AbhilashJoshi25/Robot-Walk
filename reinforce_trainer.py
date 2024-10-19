import numpy as np
import torch
from policy_network import PolicyNetwork
    
class ReinforceAgent:
    """An agent that learns a policy via the REINFORCE algorithm"""
    def __init__(
        self,
        obs_dim: int,           
        action_dim: int,    
        hidden_size1: int,    
        hidden_size2: int,
        learning_rate: float, 
        gamma: float
    ):
        """
        Args:
            obs_dim: Dimension of the observation space
            action_dim: Dimension of the action space
            hidden_size1: Size of the first hidden layer
            hidden_size2: Size of the second hidden layer            
            learning_rate: The learning rate
            gamma: The discount factor
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma        
        
        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_size1, hidden_size2)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=learning_rate)

    def get_action(self, obs: np.array) -> np.array:
        """Returns an action, conditioned on the policy and observation.
        Args:
            obs: Observation from the environment
        Returns:
            action: An action to be performed
            log_prob: The logarithm of the action probability
        """
        # WRITE YOUR CODE HERE
        x = torch.FloatTensor(obs).unsqueeze(0)  # Add the batch dimension     
        means, stddevs = self.policy(x)

        # Create a normal distribution from the predicted means and standard deviations    
        dist = torch.distributions.Normal(means, stddevs)

        # Sample an action and compute the mean of the action's log probabilities
        action = dist.sample()            
        log_prob = dist.log_prob(action).mean()

        return action.squeeze(0).numpy(), log_prob


    def update(self, log_probs, rewards):
        """Update the policy network's weights.
        Args:
            log_probs: Logarithms of the action probabilities 
            rewards: The rewards received for taking that actions
        """     
        # WRITE YOUR CODE HERE
        # Compute the discounted returns (Gt)   
        returns = self.compute_returns(rewards)

        # Compute the policy loss
        loss = torch.tensor(0.0)       
        for log_prob, Gt in zip(log_probs, returns):
            loss += -log_prob * Gt

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_returns(self, rewards):
        """Compute the returns Gt for all the episode steps."""
        returns = []       
        current_return = 0 

        for reward in reversed(rewards):           
            current_return = reward + self.gamma * current_return
            returns.insert(0, current_return)
        return returns
