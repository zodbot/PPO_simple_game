import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Solving Criteria: According to OpenAI Gym (now Gymnasium) documentation, CartPole-v1 is considered "solved" when the
# agent achieves an average reward of 475 or more over 100 consecutive episodes.

# Simple policy network
# This neural network takes a state as input and outputs a probability distribution over possible actions.
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)


# REINFORCE algorithm
def train_reinforce(env_name, policy, optimizer, num_episodes=1000, gamma=0.99):
    env = gym.make(env_name)

    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []

        # Collect trajectory
        done = False
        truncated = False
        while not (done or truncated):
            state_tensor = torch.FloatTensor(state)
            action_probs = policy(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()

            next_state, reward, done, truncated, _ = env.step(action.item())

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)

            state = next_state

        # Calculate returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        # Normalize returns
        if len(returns) > 1:  # Only normalize if we have more than one return
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)

        policy_loss = torch.stack(policy_loss).sum()

        # Update policy
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards)}")

    env.close()


# Actor-Critic model
class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )

        # Policy network (Actor)
        self.actor = nn.Sequential(
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )

        # Value network (Critic)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        shared_features = self.shared(x)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value


# Vanilla Policy Gradient with Baseline
def train_vpg_with_baseline(env_name, model, optimizer, num_episodes=1000, gamma=0.99):
    env = gym.make(env_name)

    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        values = []
        rewards = []

        # Collect trajectory
        done = False
        truncated = False
        while not (done or truncated):
            state_tensor = torch.FloatTensor(state)
            action_probs, state_value = model(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()

            next_state, reward, done, truncated, _ = env.step(action.item())

            log_probs.append(dist.log_prob(action))
            values.append(state_value)
            rewards.append(reward)

            state = next_state

        # Calculate returns and advantages
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        values = torch.cat(values)
        advantages = returns - values.detach()

        # Normalize advantages
        if len(advantages) > 1:  # Only normalize if we have more than one advantage
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        # Calculate losses
        actor_loss = [-log_prob * adv for log_prob, adv in zip(log_probs, advantages)]
        actor_loss = torch.stack(actor_loss).sum()

        critic_loss = 0.5 * ((returns - values) ** 2).sum()

        # Combined loss
        loss = actor_loss + critic_loss

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards)}")

    env.close()


# Example usage
if __name__ == "__main__":
    # Example for REINFORCE
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy = PolicyNetwork(input_dim, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    print("Training with REINFORCE...")
    train_reinforce(env_name, policy, optimizer, num_episodes=500)

    # Example for Actor-Critic
    model = ActorCritic(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("\nTraining with Actor-Critic...")
    train_vpg_with_baseline(env_name, model, optimizer, num_episodes=500)