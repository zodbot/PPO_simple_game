import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


# PPO networks with support for both discrete and continuous action spaces
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_space):
        super(ActorCritic, self).__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        # Determine if action space is discrete or continuous
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)

        # Actor head
        if self.is_discrete:
            n_actions = action_space.n
            self.actor = nn.Sequential(
                nn.Linear(128, n_actions),
                nn.Softmax(dim=-1)
            )
        else:  # Continuous actions
            n_actions = action_space.shape[0]
            self.action_mean = nn.Linear(128, n_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, n_actions))

        # Critic head
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        shared_features = self.shared(x)

        if self.is_discrete:
            action_probs = self.actor(shared_features)
            return action_probs, self.critic(shared_features)
        else:
            action_mean = self.action_mean(shared_features)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            return (action_mean, action_std), self.critic(shared_features)

    def get_action(self, state, action=None):
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension

        if self.is_discrete:
            action_probs, state_value = self(state)
            dist = Categorical(action_probs)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            return action.item(), log_prob, state_value
        else:
            (mean, std), state_value = self(state)
            dist = Normal(mean, std)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action).sum(dim=-1)

            # Convert to numpy for environment if needed
            if action.dim() > 1:
                action_np = action.squeeze().detach().numpy()
            else:
                action_np = action.detach().numpy()

            return action_np, log_prob, state_value


# PPO Agent
class PPOAgent:
    def __init__(self, env_name, learning_rate=3e-4, gamma=0.99, clip_ratio=0.2,
                 epochs=10, batch_size=64, lambda_gae=0.95):
        self.env = gym.make(env_name)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_gae = lambda_gae

        # Initialize model
        input_dim = self.env.observation_space.shape[0]
        self.model = ActorCritic(input_dim, self.env.action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Environment info
        self.is_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.env_name = env_name

        # Set default target rewards for different environments
        self.target_rewards = {
            "CartPole-v1": 475,
            "LunarLander-v2": 200,
            "Acrobot-v1": -100,  # Higher is better, but environment gives negative rewards
            "BipedalWalker-v3": 300,
            "MountainCar-v0": -110,  # Higher is better, but environment gives negative rewards
            "Pendulum-v1": -200,  # Higher is better, but environment gives negative rewards
        }

    def collect_trajectory(self, max_steps=1000):
        # Storage
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []

        # Collect experience
        state, _ = self.env.reset()
        done = False
        total_reward = 0

        for _ in range(max_steps):
            # Get action
            action, log_prob, value = self.model.get_action(state)

            # Execute action
            next_state, reward, done, truncated, _ = self.env.step(action)
            total_reward += reward

            # Store data
            states.append(state)

            # Handle different action types
            if self.is_discrete:
                actions.append(action)
            else:
                actions.append(action)

            log_probs.append(log_prob.detach())
            rewards.append(reward)
            values.append(value.squeeze().detach())
            dones.append(done or truncated)

            state = next_state

            if done or truncated:
                break

        # Convert to appropriate tensors
        states = np.array(states)

        if self.is_discrete:
            actions = np.array(actions)
            actions_tensor = torch.LongTensor(actions)
        else:
            actions = np.array(actions)
            actions_tensor = torch.FloatTensor(actions)

        return {
            'states': torch.FloatTensor(states),
            'actions': actions_tensor,
            'log_probs': torch.stack(log_probs),
            'rewards': torch.FloatTensor(rewards),
            'values': torch.stack(values),
            'dones': torch.FloatTensor(dones),
            'total_reward': total_reward
        }

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        next_value = 0
        next_advantage = 0

        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[t]

            # TD target
            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]

            # GAE
            advantages[t] = delta + self.gamma * self.lambda_gae * next_advantage * non_terminal
            returns[t] = advantages[t] + values[t]

            # Update for next iteration
            next_value = values[t]
            next_advantage = advantages[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def update_policy(self, trajectory):
        # Unpack trajectory
        states = trajectory['states']
        actions = trajectory['actions']
        old_log_probs = trajectory['log_probs']
        rewards = trajectory['rewards']
        values = trajectory['values']
        dones = trajectory['dones']

        # Compute returns and advantages using GAE
        returns, advantages = self.compute_gae(rewards, values, dones)

        # PPO update for multiple epochs
        for _ in range(self.epochs):
            # Get current policy distribution and values
            if self.is_discrete:
                action_probs, current_values = self.model(states)
                dist = Categorical(action_probs)
                current_log_probs = dist.log_prob(actions)
            else:
                (mean, std), current_values = self.model(states)
                dist = Normal(mean, std)
                current_log_probs = dist.log_prob(actions).sum(dim=-1)

            # Calculate ratios
            ratios = torch.exp(current_log_probs - old_log_probs)

            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

            # PPO's clipped objective function
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            critic_loss = F.mse_loss(current_values.squeeze(), returns)

            # Total loss
            loss = actor_loss + 0.5 * critic_loss

            # Update weights
            self.optimizer.zero_grad()
            loss.backward()

            # Optional: gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            self.optimizer.step()

    def train(self, max_episodes=1000, trajectory_length=1000, target_reward=None):
        # Set target reward based on environment if not specified
        if target_reward is None:
            if self.env_name in self.target_rewards:
                target_reward = self.target_rewards[self.env_name]
            else:
                target_reward = float('inf')  # No early stopping

        print(f"Training on {self.env_name} with target reward: {target_reward}")

        episode_rewards = []
        avg_rewards = []  # For plotting
        best_reward = float('-inf')

        # For tracking success
        reward_threshold = target_reward
        solved_episodes = 0
        required_solved = 10  # Number of consecutive episodes above threshold to consider solved

        # For plotting
        reward_history = deque(maxlen=100)

        for episode in range(max_episodes):
            # Collect trajectory
            trajectory = self.collect_trajectory(max_steps=trajectory_length)
            episode_reward = trajectory['total_reward']
            episode_rewards.append(episode_reward)
            reward_history.append(episode_reward)

            # Update policy
            self.update_policy(trajectory)

            # Calculate average reward over last 100 episodes (or as many as we have)
            avg_reward = np.mean(list(reward_history))
            avg_rewards.append(avg_reward)

            # Print progress
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Avg reward: {avg_reward:.2f}")

                # Save the model if it's the best so far
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    torch.save(self.model.state_dict(), f"ppo_{self.env_name}_best.pt")

            # Check if the environment is solved
            if avg_reward >= reward_threshold:
                solved_episodes += 1
                if solved_episodes >= required_solved:
                    print(f"Environment solved in {episode} episodes with average reward {avg_reward:.2f}")
                    break
            else:
                solved_episodes = 0

            # Plot learning progress every 100 episodes
            if episode % 100 == 0 and episode > 0:
                self.plot_learning_curve(episode_rewards, avg_rewards)

        self.env.close()

        # Final learning curve
        self.plot_learning_curve(episode_rewards, avg_rewards)

        return episode_rewards, avg_rewards

    def plot_learning_curve(self, rewards, avg_rewards):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.title(f'Episode Rewards - {self.env_name}')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        plt.subplot(1, 2, 2)
        plt.plot(avg_rewards)
        plt.axhline(y=self.target_rewards.get(self.env_name, 0), color='r', linestyle='--', label='Target')
        plt.title('Average Rewards (last 100 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Avg Reward')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"ppo_{self.env_name}_learning_curve.png")
        plt.close()

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def evaluate(self, num_episodes=10, render=False):
        render_mode = "human" if render else None
        eval_env = gym.make(self.env_name, render_mode=render_mode)

        eval_rewards = []

        for i in range(num_episodes):
            state, _ = eval_env.reset()
            done = False
            truncated = False
            total_reward = 0

            while not (done or truncated):
                # Select action (greedy for evaluation)
                with torch.no_grad():
                    if self.is_discrete:
                        action_dist, _ = self.model(torch.FloatTensor(state).unsqueeze(0))
                        action = torch.argmax(action_dist).item()
                    else:
                        (mean, _), _ = self.model(torch.FloatTensor(state).unsqueeze(0))
                        action = mean.squeeze().numpy()

                next_state, reward, done, truncated, _ = eval_env.step(action)
                total_reward += reward
                state = next_state

            eval_rewards.append(total_reward)
            print(f"Evaluation episode {i + 1}/{num_episodes}: Total reward: {total_reward:.2f}")

        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        print(f"Evaluation results: Avg reward: {avg_reward:.2f} ± {std_reward:.2f}")

        eval_env.close()
        return avg_reward, std_reward


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PPO for various gym environments')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        help='Gym environment name')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate a trained model')
    parser.add_argument('--render', action='store_true',
                        help='Render evaluation episodes')

    args = parser.parse_args()

    agent = PPOAgent(args.env, learning_rate=args.lr)

    if args.eval:
        try:
            agent.load_model(f"ppo_{args.env}_best.pt")
            print(f"Loaded model from ppo_{args.env}_best.pt")
            agent.evaluate(num_episodes=10, render=args.render)
        except FileNotFoundError:
            print(f"No trained model found for {args.env}. Please train first.")
    else:
        rewards, avg_rewards = agent.train(max_episodes=args.episodes)