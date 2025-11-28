# Reinforcement Learning Control of 2D Integrator - Project Report

## Overview

This module implements reinforcement learning control for a 2D discrete-time integrator system using Proximal Policy Optimization (PPO). The agent learns to navigate from a start position to a goal while avoiding obstacles through trial and error.

---

## System Model

The same discrete-time integrator model from the classical and symbolic control modules:

$$
\begin{cases}
x_1(t+1) = x_1(t) + \tau (u_1(t) + w_1(t)) \\
x_2(t+1) = x_2(t) + \tau (u_2(t) + w_2(t))
\end{cases}
$$

### Constraints

| Constraint | Symbol | Bounds |
|------------|--------|--------|
| State | $\mathbb{X}$ | $[-10, 10] \times [-10, 10]$ |
| Input | $\mathbb{U}$ | $[-1, 1] \times [-1, 1]$ |
| Disturbance | $\mathbb{W}$ | $[-0.05, 0.05] \times [-0.05, 0.05]$ |

---

## Reinforcement Learning Approach

### Problem Formulation

The navigation task is formulated as a Markov Decision Process (MDP):

| Component | Description |
|-----------|-------------|
| State | Robot position, goal direction, obstacle information |
| Action | Control input $u \in [-1, 1]^2$ |
| Reward | Goal reaching, obstacle avoidance, progress |
| Episode | Max 500 steps or terminal condition |

### Environment Setup

**Start Position:** $(-8, -8)$ - bottom-left corner

**Goal Position:** $(8, 8)$ - top-right corner

**Obstacle:** Center at $(0, 0)$ with radius $2.5$

The obstacle is placed directly in the path between start and goal, forcing the agent to learn avoidance behavior.

---

## Observation Space

The agent receives a 12-dimensional observation vector:

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0-1 | Position | $[-1, 1]$ | Normalized robot position $(x/10, y/10)$ |
| 2-3 | Goal vector | $[-2, 2]$ | Direction to goal (normalized) |
| 4 | Goal distance | $[0, 1]$ | Normalized distance to goal |
| 5-6 | Obstacle direction | $[-1, 1]$ | Unit vector toward obstacle |
| 7 | Surface distance | $[-1, 1]$ | Distance to obstacle surface (normalized) |
| 8 | Proximity | $[0, 1]$ | Clamped proximity (0=touching, 1=far) |
| 9 | Obstacle in path | $[-1, 1]$ | Dot product: positive if obstacle ahead |
| 10 | Danger flag | $\{0, 1\}$ | 1 if within 1.0 units of obstacle |
| 11 | Warning flag | $\{0, 1\}$ | 1 if within 3.0 units of obstacle |

---

## Action Space

Continuous 2D action space:

$$
a = [a_1, a_2] \in [-1, 1]^2
$$

Actions are directly mapped to control inputs $u = a$.

---

## Reward Function

The reward function is designed to encourage goal-reaching and obstacle avoidance:

### Terminal Rewards

| Condition | Reward | Purpose |
|-----------|--------|---------|
| Goal reached ($d < 1.0$) | $+100$ | Strong positive reinforcement |
| Collision with obstacle | $-100$ | Strong negative reinforcement |

### Shaping Rewards

$$
r_{\text{shaping}} = 5 \cdot \Delta d_{\text{goal}} - 3 \cdot e^{-d_{\text{obs}}} - 0.01
$$

Where:
- $\Delta d_{\text{goal}}$: Progress toward goal (positive = closer)
- $d_{\text{obs}}$: Distance to obstacle surface
- Exponential penalty activates when $d_{\text{obs}} < 3.0$
- Small time penalty ($-0.01$) encourages efficiency

---

## PPO Algorithm

### Overview

Proximal Policy Optimization (PPO) is a policy gradient method that:

1. Collects experience using current policy
2. Computes advantage estimates
3. Updates policy with clipped objective

### Objective Function

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio
- $\hat{A}_t$ is the advantage estimate
- $\epsilon = 0.2$ is the clip range

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning rate | $3 \times 10^{-4}$ | Adam optimizer step size |
| Steps per update | 2048 | Rollout buffer size |
| Batch size | 64 | Minibatch size for updates |
| Epochs | 10 | PPO update iterations |
| Gamma | 0.99 | Discount factor |
| GAE Lambda | 0.95 | Generalized advantage estimation |
| Clip range | 0.2 | Policy ratio clipping |
| Entropy coef. | 0.01 | Exploration bonus |
| Value coef. | 0.5 | Value loss weight |

### Network Architecture

```
Policy Network (Actor):
  Input (12) → Dense(64) → ReLU → Dense(64) → ReLU → Output (2)

Value Network (Critic):
  Input (12) → Dense(64) → ReLU → Dense(64) → ReLU → Output (1)
```

---

## Training Process

### Training Loop

1. **Collect rollouts**: Run policy in 4 parallel environments
2. **Compute advantages**: Use GAE with $\lambda = 0.95$
3. **Update policy**: PPO clipped objective for 10 epochs
4. **Evaluate**: Test on separate environment every 10k steps
5. **Save best**: Keep model with highest evaluation reward

### Monitoring

Training progress is logged to TensorBoard:
- Episode rewards
- Policy loss
- Value loss
- Entropy
- Explained variance

---

## Visualization

### PyBullet Scene Elements

| Element | Color | Description |
|---------|-------|-------------|
| Robot | Blue sphere | Current position |
| Goal | Green sphere | Target destination (radius 0.5) |
| Obstacle | Red cylinder | Forbidden zone (radius 2.5) |
| Trajectory | Red line | Actual path taken (drawn in real-time) |
| Workspace | Gray lines | State constraint boundary |
| Ground | White | Workspace floor |

---

## Usage

### Training

```bash
# Default training (300k timesteps)
python reinforcement_learning/integrator_rl_control.py --train

# Custom timesteps
python reinforcement_learning/integrator_rl_control.py --train --timesteps 500000
```

Training outputs (in `reinforcement_learning/` directory):
- `ppo_integrator_model.zip` - Final model
- `best_model/` - Best model during training
- `ppo_integrator_logs/` - TensorBoard logs
- `eval_logs/` - Evaluation metrics

### Evaluation

```bash
# Evaluate with visualization
python reinforcement_learning/integrator_rl_control.py --eval --episodes 5

# Use specific model
python reinforcement_learning/integrator_rl_control.py --eval --model my_model --episodes 10
```

### Random Baseline

```bash
# Run with random actions (no training)
python reinforcement_learning/integrator_rl_control.py --random --episodes 3
```

### TensorBoard Monitoring

```bash
tensorboard --logdir reinforcement_learning/ppo_integrator_logs
```

---

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--train` | - | Train a new PPO agent |
| `--eval` | - | Evaluate trained agent |
| `--random` | - | Run with random actions |
| `--timesteps` | 300000 | Training timesteps |
| `--episodes` | 5 | Evaluation episodes |
| `--model` | auto | Model path for evaluation |

---

## Expected Results

### Training Metrics

| Metric | Expected Range |
|--------|----------------|
| Final reward | $> 50$ per episode |
| Success rate | $> 80\%$ |
| Collision rate | $< 10\%$ |
| Episode length | 200-400 steps |

### Learned Behavior

The trained agent should:
1. Move away from start position
2. Navigate around the central obstacle
3. Approach goal from a safe angle
4. Avoid getting too close to obstacle boundary

---

## Comparison: Classical vs Symbolic vs RL Control

| Aspect | Classical | Symbolic | RL (PPO) |
|--------|-----------|----------|----------|
| Specification | Trajectory | LTL formula | Reward function |
| Design | Manual tuning | Formal synthesis | Learned |
| Optimality | Local (PID) | Satisficing | Optimal for reward |
| Adaptability | Fixed gains | Fixed regions | Generalizes |
| Training | None | None | Required |
| Guarantees | Stability | Satisfaction | Statistical |
| Obstacle handling | None | Potential fields | Learned |

---

## File Structure

```
integrator_robot_project/
├── classical_control/
│   ├── integrator_classical_control.py   # Classical control script
│   ├── classical_examples.txt            # Example commands
│   └── CLASSICAL_REPORT.md               # Classical control report
├── symbolic_control/
│   ├── integrator_symbolic_control.py    # Symbolic control script
│   ├── symbolic_examples.txt             # Example commands
│   └── SYMBOLIC_REPORT.md                # Symbolic control report
├── reinforcement_learning/
│   ├── integrator_rl_control.py          # Main RL/PPO control script
│   ├── rl_examples.txt                   # Example commands
│   ├── RL_REPORT.md                      # This report
│   ├── best_model/                       # Best model during training
│   ├── ppo_integrator_logs/              # TensorBoard logs
│   └── eval_logs/                        # Evaluation metrics
├── requirements.txt                      # Dependencies
└── Project.md                            # Original specification
```

---

## Dependencies

- **stable-baselines3[extra]**: PPO implementation
- **gymnasium**: RL environment interface
- **pybullet**: Physics simulation and visualization
- **numpy**: Numerical computations

---

## Conclusion

The reinforcement learning implementation demonstrates:

1. **Learning from Experience**: Agent discovers obstacle avoidance through trial and error
2. **End-to-End Control**: No hand-designed controller needed
3. **Optimal Behavior**: Learns efficient paths that balance speed and safety
4. **Generalization**: Can handle disturbances not seen during training
5. **Integration**: Uses same system model as classical and symbolic approaches

This approach shows how modern deep RL can solve navigation problems that would traditionally require careful controller design or complex motion planning algorithms.