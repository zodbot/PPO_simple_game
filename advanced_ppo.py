import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time
import os
from collections import deque


# Observation normalizer - helps the network learn more effectively
class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


# Advanced Actor-Critic Network with proper initialization
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_space, hidden_dim=64):
        super(ActorCritic, self).__init__()
        self.hidden_dim = hidden_dim

        # Determine if action space is discrete or continuous
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor head (policy)
        if self.is_discrete:
            n_actions = action_space.n
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, n_actions),
                nn.Softmax(dim=-1)
            )
        else:  # Continuous actions
            n_actions = action_space.shape[0]
            self.actor_mean = nn.Linear(hidden_dim, n_actions)
            self.actor_log_std = nn.Parameter(torch.zeros(1, n_actions))

        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize using orthogonal initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        shared_features = self.shared(x)

        if self.is_discrete:
            action_probs = self.actor(shared_features)
            return action_probs, self.critic(shared_features)
        else:
            action_mean = self.actor_mean(shared_features)
            action_log_std = self.actor_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            return (action_mean, action_std), self.critic(shared_features)

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            if self.is_discrete:
                action_probs, state_value = self(state)
                dist = Categorical(action_probs)

                if deterministic:
                    action = torch.argmax(action_probs, dim=1)
                else:
                    action = dist.sample()

                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                return action.item(), log_prob, state_value, entropy
            else:
                (mean, std), state_value = self(state)
                dist = Normal(mean, std)

                if deterministic:
                    action = mean
                else:
                    action = dist.sample()

                log_prob = dist.log_prob(action).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)

                return action.squeeze().numpy(), log_prob, state_value, entropy

    def evaluate_actions(self, states, actions):
        if self.is_discrete:
            action_probs, state_values = self(states)
            dist = Categorical(action_probs)

            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

        else:
            (mean, std), state_values = self(states)
            dist = Normal(mean, std)

            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        return log_probs, state_values, entropy


# Buffer for storing trajectories
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.returns = []
        self.advantages = []

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        # Convert to numpy arrays for easy operations
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values + [last_value])

        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)

        # GAE calculation
        gae = 0
        for t in reversed(range(len(rewards))):
            # If terminal state, next state value is 0
            next_non_terminal = 1.0 - dones[t]

            # Delta is the TD error
            delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]

            # Recursive advantage calculation with lambda smoothing
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + np.array(self.values)

        # Store calculated returns and advantages
        self.returns = returns.tolist()
        self.advantages = advantages.tolist()

    def get_batches(self, batch_size):
        indices = np.arange(len(self.states))
        np.random.shuffle(indices)

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        if isinstance(self.actions[0], (int, np.integer)):
            actions = torch.LongTensor(self.actions)
        else:
            actions = torch.FloatTensor(np.array(self.actions))
        log_probs = torch.FloatTensor(self.log_probs)
        returns = torch.FloatTensor(self.returns)
        advantages = torch.FloatTensor(self.advantages)

        # Normalize advantages (important for stable training)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Generate mini-batches
        for start_idx in range(0, len(self.states), batch_size):
            end_idx = min(start_idx + batch_size, len(self.states))
            batch_indices = indices[start_idx:end_idx]

            yield (
                states[batch_indices],
                actions[batch_indices],
                log_probs[batch_indices],
                returns[batch_indices],
                advantages[batch_indices]
            )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.returns.clear()
        self.advantages.clear()


# Advanced PPO Agent
class PPOAgent:
    def __init__(self, env_name, hidden_dim=64, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_ratio=0.2, target_kl=0.01, entropy_coef=0.01, value_coef=0.5,
                 ppo_epochs=10, batch_size=64, max_grad_norm=0.5,
                 normalize_observations=True, lr_annealing=True):

        # Environment
        self.env = gym.make(env_name)
        self.env_name = env_name

        # Check if action space is discrete or continuous
        self.is_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)

        # Observation normalization
        self.normalize_observations = normalize_observations
        if normalize_observations:
            self.obs_normalizer = RunningMeanStd(shape=self.env.observation_space.shape)

        # Model initialization
        input_dim = self.env.observation_space.shape[0]
        self.model = ActorCritic(input_dim, self.env.action_space, hidden_dim)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.lr_annealing = lr_annealing

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer()

        # Logging
        self.episode_rewards = []
        self.avg_rewards = []
        self.kls = []

        # Target rewards for different environments
        self.target_rewards = {
            "CartPole-v1": 475,
            "LunarLander-v2": 200,
            "Acrobot-v1": -100,
            "BipedalWalker-v3": 300,
            "MountainCar-v0": -110,
            "Pendulum-v1": -200,
        }

        # Create directory for saving results
        os.makedirs("results", exist_ok=True)

    def normalize_observation(self, obs):
        if self.normalize_observations:
            return self.obs_normalizer.normalize(obs)
        return obs

    def collect_rollouts(self, num_steps):
        self.rollout_buffer.clear()

        state, _ = self.env.reset()
        if self.normalize_observations:
            self.obs_normalizer.update(np.array([state]))

        # Collect trajectories
        episode_rewards = []
        total_reward = 0

        for _ in range(num_steps):
            # Normalize state
            norm_state = self.normalize_observation(state)

            # Get action
            action, log_prob, value, _ = self.model.get_action(norm_state)

            # Execute action
            next_state, reward, done, truncated, _ = self.env.step(action)
            terminal = done or truncated

            # Track rewards
            total_reward += reward

            # Store transition
            self.rollout_buffer.add(norm_state, action, reward, terminal, log_prob.item(), value.item())

            # Move to next state
            state = next_state
            if self.normalize_observations:
                self.obs_normalizer.update(np.array([state]))

            # Reset if episode ended
            if terminal:
                episode_rewards.append(total_reward)
                total_reward = 0
                state, _ = self.env.reset()
                if self.normalize_observations:
                    self.obs_normalizer.update(np.array([state]))

        # Get value of last state for bootstrapping
        _, _, last_value, _ = self.model.get_action(self.normalize_observation(state))

        # Compute returns and advantages
        self.rollout_buffer.compute_returns_and_advantages(last_value.item(), self.gamma, self.gae_lambda)

        return episode_rewards

    def update_policy(self, progress=None):
        # Apply learning rate annealing if enabled
        if self.lr_annealing and progress is not None:
            current_lr = self.learning_rate * (1 - progress)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

        # PPO update for specified number of epochs
        for _ in range(self.ppo_epochs):
            # Iterate over mini-batches
            total_kl = 0
            total_entropy = 0
            total_actor_loss = 0
            total_critic_loss = 0
            num_batches = 0

            for states, actions, old_log_probs, returns, advantages in self.rollout_buffer.get_batches(self.batch_size):
                # Get current policy outputs
                new_log_probs, state_values, entropy = self.model.evaluate_actions(states, actions)

                # Calculate policy ratio
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value function loss with clipping
                value_pred = state_values.squeeze()
                value_clipped = old_values = returns
                value_loss_unclipped = (returns - value_pred).pow(2)
                value_loss_clipped = (returns - value_clipped).pow(2)
                critic_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                # Calculate KL divergence for early stopping
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                total_kl += approx_kl
                total_entropy += entropy.mean().item()
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                num_batches += 1

                # Perform update
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()

                # Early stopping based on KL divergence
                if approx_kl > 1.5 * self.target_kl:
                    break

            # Calculate average metrics
            avg_kl = total_kl / num_batches if num_batches > 0 else 0
            self.kls.append(avg_kl)

            # Early stopping based on average KL
            if avg_kl > 1.5 * self.target_kl:
                break

    def train(self, total_steps=1000000, eval_interval=10000, save_interval=50000,
              log_interval=1000, rollout_length=2048, render_freq=0):

        # Set target reward based on environment
        if self.env_name in self.target_rewards:
            target_reward = self.target_rewards[self.env_name]
        else:
            target_reward = float('inf')  # No early stopping

        print(f"Starting training on {self.env_name} with target reward: {target_reward}")
        print(f"Model architecture: {self.model}")
        print(f"PPO hyperparameters: lr={self.learning_rate}, gamma={self.gamma}, " +
              f"gae_lambda={self.gae_lambda}, clip_ratio={self.clip_ratio}, " +
              f"ppo_epochs={self.ppo_epochs}, batch_size={self.batch_size}")

        # Setup tracking
        num_updates = total_steps // rollout_length
        start_time = time.time()
        running_reward = deque(maxlen=100)
        best_reward = float('-inf')

        # Create a render environment if rendering is requested
        if render_freq > 0:
            render_env = gym.make(self.env_name, render_mode="human")

        # Main training loop
        for update in range(1, num_updates + 1):
            # Collect rollouts
            episode_rewards = self.collect_rollouts(rollout_length)

            # If no complete episodes, continue
            if not episode_rewards:
                continue

            # Update policy
            progress = update / num_updates  # For learning rate annealing
            self.update_policy(progress)

            # Track rewards
            self.episode_rewards.extend(episode_rewards)
            running_reward.extend(episode_rewards)
            avg_reward = np.mean(list(running_reward)) if running_reward else 0
            self.avg_rewards.append(avg_reward)

            # Log progress
            if update % log_interval == 0:
                elapsed_time = time.time() - start_time
                steps = update * rollout_length
                fps = int(steps / elapsed_time)

                print(f"Update {update}/{num_updates}, Steps: {steps}, " +
                      f"FPS: {fps}, Episodes: {len(self.episode_rewards)}, " +
                      f"Mean reward: {avg_reward:.2f}, Recent rewards: {np.mean(episode_rewards):.2f}")

                # Save best model
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.save_model("best")
                    print(f"New best model with reward {best_reward:.2f}")

            # Render if requested
            if render_freq > 0 and update % render_freq == 0:
                self.render_episode(render_env)

            # Evaluate
            if update % eval_interval == 0:
                eval_reward = self.evaluate(num_episodes=5)
                print(f"Evaluation after {steps} steps: {eval_reward:.2f}")

                # Plot learning curves
                self.plot_learning_curves()

            # Save model checkpoint
            if update % save_interval == 0:
                self.save_model(f"checkpoint_{steps}")

            # Check if environment is solved
            if avg_reward >= target_reward:
                print(f"Environment solved after {steps} steps with average reward {avg_reward:.2f}!")
                self.save_model("solved")
                break

        # Final evaluation and save
        print("Training completed!")
        self.save_model("final")
        final_eval = self.evaluate(num_episodes=10)
        print(f"Final evaluation: {final_eval:.2f}")

        # Final plots
        self.plot_learning_curves()

        return self.episode_rewards, self.avg_rewards

    def evaluate(self, num_episodes=10, render=False):
        # Create evaluation environment
        render_mode = "human" if render else None
        if render_mode:
            eval_env = gym.make(self.env_name, render_mode=render_mode)
        else:
            eval_env = gym.make(self.env_name)

        eval_rewards = []

        for _ in range(num_episodes):
            state, _ = eval_env.reset()
            done = False
            truncated = False
            episode_reward = 0

            while not (done or truncated):
                # Normalize state
                norm_state = self.normalize_observation(state)

                # Get action (deterministic for evaluation)
                action, _, _, _ = self.model.get_action(norm_state, deterministic=True)

                # Execute action
                state, reward, done, truncated, _ = eval_env.step(action)
                episode_reward += reward

            eval_rewards.append(episode_reward)

        eval_env.close()
        return np.mean(eval_rewards)

    def render_episode(self, render_env=None):
        """Render a single episode with the current policy"""
        if render_env is None:
            render_env = gym.make(self.env_name, render_mode="human")

        state, _ = render_env.reset()
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            # Normalize state
            norm_state = self.normalize_observation(state)

            # Get action
            action, _, _, _ = self.model.get_action(norm_state, deterministic=True)

            # Execute action
            state, reward, done, truncated, _ = render_env.step(action)
            total_reward += reward

        print(f"Rendered episode finished with reward: {total_reward:.2f}")

    def save_model(self, tag="best"):
        """Save model with a specific tag"""
        os.makedirs("models", exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_mean': self.obs_normalizer.mean if self.normalize_observations else None,
            'obs_var': self.obs_normalizer.var if self.normalize_observations else None,
        }, f"models/ppo_{self.env_name}_{tag}.pt")

    def load_model(self, path):
        """Load a saved model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.normalize_observations and 'obs_mean' in checkpoint and checkpoint['obs_mean'] is not None:
            self.obs_normalizer.mean = checkpoint['obs_mean']
            self.obs_normalizer.var = checkpoint['obs_var']

    def plot_learning_curves(self):
        """Plot and save learning curves"""
        plt.figure(figsize=(15, 5))

        # Plot episode rewards
        plt.subplot(1, 3, 1)
        plt.plot(self.episode_rewards)
        plt.title(f'Episode Rewards - {self.env_name}')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        # Plot average rewards
        plt.subplot(1, 3, 2)
        plt.plot(self.avg_rewards)
        if self.env_name in self.target_rewards:
            plt.axhline(y=self.target_rewards[self.env_name], color='r', linestyle='--', label='Target')
        plt.title('Average Rewards (100 episodes)')
        plt.xlabel('Update')
        plt.ylabel('Avg Reward')
        plt.legend()

        # Plot KL divergence
        plt.subplot(1, 3, 3)
        plt.plot(self.kls)
        plt.axhline(y=self.target_kl, color='r', linestyle='--', label='Target KL')
        plt.title('Average KL Divergence')
        plt.xlabel('Update')
        plt.ylabel('KL Divergence')
        plt.legend()

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/ppo_{self.env_name}_learning_curve.png")
        plt.close()


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Advanced PPO')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Gym environment name')
    parser.add_argument('--steps', type=int, default=1000000, help='Total timesteps')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--clip_ratio', type=float, default=0.2, help='PPO clip ratio')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--render_freq', type=int, default=0, help='Render frequency (0=never)')
    parser.add_argument('--eval', action='store_true', help='Evaluate only')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model for evaluation')

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create agent
    agent = PPOAgent(
        env_name=args.env,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef
    )

    if args.eval:
        # Evaluation mode
        if args.model_path:
            agent.load_model(args.model_path)
        else:
            # Try to load the best model
            try:
                agent.load_model(f"models/ppo_{args.env}_best.pt")
                print(f"Loaded best model for {args.env}")
            except FileNotFoundError:
                print("No model found. Please specify a model path with --model_path.")
                exit(1)

        avg_reward = agent.evaluate(num_episodes=10, render=True)
        print(f"Average evaluation reward: {avg_reward:.2f}")
    else:
        # Training mode
        agent.train(
            total_steps=args.steps,
            render_freq=args.render_freq
        )