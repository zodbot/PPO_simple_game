# Reinforcement Learning Implementation

This repository contains a PyTorch implementation of two reinforcement learning algorithms:
1. REINFORCE (Monte Carlo Policy Gradient)
2. Actor-Critic with Baseline

## Training Analysis

### REINFORCE Training Results

The following analysis is based on training the REINFORCE algorithm on the CartPole-v1 environment for 500 episodes:

#### Learning Phases

1. **Initial Learning (Episodes 0-100)**
   - Agent begins with low rewards (11-25 steps)
   - First significant jump occurs around episode 30-40 (reaching 138)
   - Major breakthrough at episode 60 with reward 471
   - Performance fluctuates significantly during this exploratory phase

2. **Improving Performance (Episodes 100-300)**
   - More consistent improvements with rewards frequently above 200
   - First perfect score (500) achieved at episode 140
   - Additional perfect scores at episodes 220, 290
   - Policy begins to stabilize toward optimal behavior

3. **Peak Performance (Episodes 300-350)**
   - Episodes 300-340 show a remarkable streak of six consecutive perfect scores (500)
   - This indicates the policy successfully converged to an optimal solution for the environment

4. **Catastrophic Forgetting (Episodes 350-490)**
   - Sudden performance collapse at episode 350 (dropping to just 9)
   - Performance becomes erratic again
   - Never fully recovers to the consistent perfect scores seen in phase 3
   - Training ends with mediocre performance around 150-160

#### Observations and Challenges

The training results demonstrate a common challenge in policy gradient methods: **policy degradation** or **catastrophic forgetting**. Despite successfully learning to solve the environment (as evidenced by the consecutive perfect scores), the algorithm continues updating the policy, eventually moving away from the optimal solution.

This occurs because:
- REINFORCE continues to update the policy even after finding a good solution
- The stochastic nature of policy gradient methods can lead to harmful updates
- There's no explicit mechanism to preserve good policies once found


## Usage

Instructions for running the code:

```python
# Training with REINFORCE
python reinforce.py  --episodes 500 --eval_interval 100

# Training with Actor-Critic
python main.py --algorithm actor_critic --episodes 500 --eval_interval 100
```

## Dependencies

- PyTorch
- NumPy
- Gymnasium
- Matplotlib