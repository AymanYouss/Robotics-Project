# 2D Integrator Robot Control - Project Report

## Overview

This project implements a 2D discrete-time integrator control system using PyBullet for visualization. The robot navigates in a 2D plane while following various reference trajectories using a PID controller.

---

## System Model

### Discrete-Time Integrator Dynamics

The system follows the discrete-time integrator model:

$$
\begin{cases}
x_1(t+1) = x_1(t) + \tau (u_1(t) + w_1(t)) \\
x_2(t+1) = x_2(t) + \tau (u_2(t) + w_2(t))
\end{cases}
$$

Where:
- **$x(t) = [x_1(t), x_2(t)]^T$**: State vector (position in 2D plane)
- **$u(t) = [u_1(t), u_2(t)]^T$**: Control input vector
- **$w(t) = [w_1(t), w_2(t)]^T$**: Disturbance vector
- **$\tau = 0.1$**: Sampling period

### System Constraints

| Constraint | Symbol | Bounds |
|------------|--------|--------|
| State | $\mathbb{X}$ | $[-10, 10] \times [-10, 10]$ |
| Input | $\mathbb{U}$ | $[-1, 1] \times [-1, 1]$ |
| Disturbance | $\mathbb{W}$ | $[-0.05, 0.05] \times [-0.05, 0.05]$ |

---

## Control Strategy: PID Controller

### Controller Design

A PID (Proportional-Integral-Derivative) controller is implemented for trajectory tracking:

$$
u(t) = K_p \cdot e(t) + K_i \cdot \int_0^t e(\tau) d\tau + K_d \cdot \frac{de(t)}{dt}
$$

Where $e(t) = x_{ref}(t) - x(t)$ is the tracking error.

### Implementation Details

```python
class PIDController:
    def __init__(self, kp=2.0, ki=0.1, kd=0.5):
        self.kp = kp   # Proportional gain
        self.ki = ki   # Integral gain
        self.kd = kd   # Derivative gain
```

**Key Features:**
- **Proportional Term**: Corrects current error
- **Integral Term**: Eliminates steady-state error (with anti-windup clipping at ±5)
- **Derivative Term**: Dampens oscillations and improves response

### Default Gains
| Gain | Value | Purpose |
|------|-------|---------|
| $K_p$ | 2.0 | Primary error correction |
| $K_i$ | 0.1 | Steady-state error elimination |
| $K_d$ | 0.5 | Oscillation damping |

---

## Trajectory Types

### 1. Linear Trajectory

Straight-line path from start to goal:

$$
x_{ref}(t) = (1 - \alpha) \cdot x_{start} + \alpha \cdot x_{goal}, \quad \alpha = \frac{t}{T}
$$

**Use Case**: Point-to-point navigation

### 2. Circular Trajectory

Circular path around a center point:

$$
\begin{cases}
x_1(t) = c_1 + r \cos(\theta) \\
x_2(t) = c_2 + r \sin(\theta)
\end{cases}
\quad \theta = \frac{2\pi t}{T}
$$

**Use Case**: Patrol/surveillance patterns

**Note**: Uses minimum 500 trajectory points for smooth tracking performance.

### 3. Infinity (Lemniscate) Trajectory

Figure-eight pattern using Lemniscate of Bernoulli:

$$
\begin{cases}
x_1(t) = c_1 + \frac{s \cos(t)}{1 + \sin^2(t)} \\
x_2(t) = c_2 + \frac{s \sin(t) \cos(t)}{1 + \sin^2(t)}
\end{cases}
$$

**Use Case**: Complex maneuvers, testing controller agility

**Note**: Uses minimum 500 trajectory points for smooth tracking performance.

---

## Visualization

### PyBullet Scene Elements

| Element | Color | Description |
|---------|-------|-------------|
| Robot | Blue sphere | Current position of the integrator |
| Goal | Green sphere | Target position (linear trajectory only) |
| Reference Trajectory | Yellow line | Desired path |
| Actual Trajectory | Red line | Path actually followed by robot (drawn in real-time) |
| Workspace Bounds | Red box | State constraint boundary |

### Trajectory Comparison

During simulation, both trajectories are displayed in real-time:
- **Yellow**: Reference trajectory (what the robot should follow)
- **Red**: Actual trajectory (drawn incrementally as the robot moves)

This allows visual comparison of tracking performance and the effect of disturbances.

---

## Performance Metrics

The system computes and displays:

1. **Average Tracking Error**: Mean distance between actual and reference positions
2. **Maximum Tracking Error**: Worst-case deviation from reference
3. **Goal Distance** (linear only): Final distance to target

---

## Usage

### Setup

```bash
cd /home/zczak/Robotics-Project/integrator_robot_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the Demo

**Linear Trajectory:**
```bash
python classical_control/integrator_classical_control.py --gui --trajectory linear
```

**Circular Trajectory:**
```bash
python classical_control/integrator_classical_control.py --gui --trajectory circular --radius 5
```

**Infinity Trajectory:**
```bash
python classical_control/integrator_classical_control.py --gui --trajectory infinity --scale 6
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--gui` | False | Enable GUI visualization |
| `--trajectory` | linear | Trajectory type (linear/circular/infinity) |
| `--start` | -8 -8 | Start position (linear only) |
| `--goal` | 8 8 | Goal position (linear only) |
| `--center` | 0 0 | Center point (circular/infinity) |
| `--radius` | 5.0 | Circle radius |
| `--scale` | 6.0 | Infinity pattern scale |
| `--kp` | 2.0 | Proportional gain |
| `--ki` | 0.1 | Integral gain |
| `--kd` | 0.5 | Derivative gain |
| `--steps` | 600 | Simulation steps |
| `--horizon` | 500 | Trajectory points (min 500 for circular/infinity) |
| `--keep-running` | False | Don't stop at goal |

---

## File Structure

```
integrator_robot_project/
├── classical_control/
│   ├── integrator_classical_control.py   # Main simulation script
│   ├── classical_examples.txt            # Example commands
│   └── CLASSICAL_REPORT.md               # This report
├── symbolic_control/
│   ├── integrator_symbolic_control.py    # Symbolic control script
│   ├── symbolic_examples.txt             # Example commands
│   └── SYMBOLIC_REPORT.md                # Symbolic control report
├── reinforcement_learning/
│   ├── integrator_rl_control.py          # RL/PPO control script
│   ├── rl_examples.txt                   # Example commands
│   └── RL_REPORT.md                      # RL control report
├── requirements.txt                      # Python dependencies
└── Project.md                            # Original project specification
```

---

## Dependencies

- **pybullet**: Physics simulation and visualization
- **numpy**: Numerical computations

---

## Conclusion

This implementation demonstrates classical control of a 2D integrator system with:

1. **PID Control**: Robust trajectory tracking with tunable gains
2. **Multiple Trajectories**: Linear, circular, and infinity patterns
3. **Constraint Handling**: State, input, and disturbance bounds enforced
4. **Visual Comparison**: Reference vs actual trajectory visualization
5. **Performance Metrics**: Quantitative tracking error analysis

The system successfully tracks reference trajectories while handling random disturbances within the specified bounds.
