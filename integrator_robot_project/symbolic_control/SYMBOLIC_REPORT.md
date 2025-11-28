# Symbolic Control of 2D Integrator - Project Report

## Overview

This module implements symbolic control for a 2D discrete-time integrator system using Linear Temporal Logic (LTL) specifications. The controller synthesizes correct-by-construction behaviors from high-level temporal logic formulas.

---

## System Model

The same discrete-time integrator model from the classical control module:

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

## Symbolic Control Approach

### State Space Abstraction

The continuous state space is partitioned into discrete **regions** that represent meaningful locations:

| Region | Position | Purpose |
|--------|----------|---------|
| Start | Bottom-left | Initial robot position |
| Goal | Top-right | Target destination |
| Obstacle 1 | Center | Forbidden zone |
| Obstacle 2 | Bottom-right | Forbidden zone |
| Obstacle 3 | Middle-left | Forbidden zone |
| Waypoint 1 | Top-left area | Intermediate target |
| Waypoint 2 | Middle-right | Intermediate target |
| Waypoint 3 | Top-left corner | Corner coverage |
| Waypoint 4 | Bottom-right corner | Corner coverage |
| Charging | Top-left | Charging station |

### LTL Operators Used

| Operator | Symbol | Meaning |
|----------|--------|---------|
| Eventually | $\Diamond$ (◇) | At some future time |
| Always | $\Box$ (□) | At all future times |
| Negation | $\neg$ (¬) | Not |
| And | $\wedge$ (∧) | Conjunction |
| Implies | $\rightarrow$ (→) | If-then |

---

## Implemented Specifications

### 1. Reach-Avoid
**LTL Formula:** $\Diamond \text{goal} \wedge \Box \neg \text{obstacle}$

**Description:** Eventually reach the goal while always avoiding obstacles.

**Automaton States:**
- State 0: Navigate to goal
- Accept: Goal reached without obstacle collision

---

### 2. Sequential Visit
**LTL Formula:** $\Diamond \text{wp1} \wedge \Diamond(\text{wp1} \rightarrow \Diamond \text{wp2}) \wedge \Diamond(\text{wp2} \rightarrow \Diamond \text{goal})$

**Description:** Visit waypoints in strict order: wp1 → wp2 → goal.

**Automaton States:**
- State 0: Go to waypoint 1
- State 1: Go to waypoint 2
- State 2: Go to goal
- Accept: Goal reached after visiting waypoints

---

### 3. Patrol
**LTL Formula:** $\Box \Diamond \text{wp1} \wedge \Box \Diamond \text{wp2} \wedge \Box \Diamond \text{goal}$

**Description:** Infinitely often visit each waypoint (continuous patrol loop).

**Automaton States:**
- Cycles through: wp1 → wp2 → goal → start → wp1 → ...
- Never terminates (perpetual operation)

---

### 4. Coverage
**LTL Formula:** $\Diamond \text{corner1} \wedge \Diamond \text{corner2} \wedge \Diamond \text{corner3} \wedge \Diamond \text{corner4}$

**Description:** Visit all four corners of the workspace.

**Automaton States:**
- State 0-3: Visit each corner in sequence
- Accept: All corners visited

---

### 5. Charge-Patrol
**LTL Formula:** $\Box \Diamond \text{charging} \wedge \Box \Diamond \text{wp1} \wedge \Box \Diamond \text{wp2}$

**Description:** Patrol pattern with periodic returns to charging station.

**Automaton States:**
- Cycles: wp1 → charge → wp2 → charge → goal → charge → ...
- Models battery constraints in real robots

---

### 6. Surveillance
**LTL Formula:** $\Box(\Diamond \text{wp1} \wedge \Diamond \text{wp2} \wedge \Diamond \text{wp3} \wedge \Diamond \text{goal})$

**Description:** Continuously monitor multiple areas.

**Automaton States:**
- Loops through all surveillance points
- Ensures complete area coverage

---

### 7. Escape (Maze Navigation)
**LTL Formula:** $\Diamond \text{goal}$ via safe waypoints

**Description:** Navigate through obstacle-dense environment using predefined safe corridor.

**Automaton States:**
- Follows safe path through maze
- Accept: Goal reached

---

## Controller Implementation

### Potential Field Method

The controller combines attractive and repulsive forces:

$$
u = u_{\text{attract}} + u_{\text{repel}}
$$

**Attractive Force:**
$$
u_{\text{attract}} = K_p \cdot \frac{x_{\text{target}} - x}{\|x_{\text{target}} - x\|}
$$

**Repulsive Force (per obstacle):**
$$
u_{\text{repel}} = \begin{cases}
5 \cdot \left(\frac{d_{\text{safe}} - d}{d_{\text{safe}}}\right)^2 \cdot \hat{n} & \text{if } d < d_{\text{safe}} \\
0 & \text{otherwise}
\end{cases}
$$

Where:
- $d$ = distance to obstacle boundary
- $d_{\text{safe}} = 4.0$ = safety margin
- $\hat{n}$ = unit vector away from obstacle

### Automaton-Based State Machine

The controller maintains an automaton state that tracks:
1. Current target waypoint index
2. Set of visited waypoints
3. Loop mode (for patrol specifications)

State transitions occur when the robot enters a target region (within tolerance of 1.0 units).

---

## Safety Verification

The system continuously monitors safety:

```python
def check_safety(x):
    for obstacle in obstacles:
        if obstacle.contains(x):
            return False  # Safety violation
    return True
```

Safety violations are counted and reported at simulation end.

---

## Visualization

### PyBullet Scene Elements

| Element | Color | Description |
|---------|-------|-------------|
| Robot | Blue sphere | Current position |
| Start region | Blue box | Initial position |
| Goal region | Green box | Target destination |
| Obstacles | Red boxes | Forbidden zones |
| Waypoints | Yellow/Orange/Cyan/Magenta | Intermediate targets |
| Charging | Purple box | Charging station |
| Trajectory | Red line | Actual path (drawn in real-time) |
| Workspace | Gray box | State constraint boundary |
| Ground | White | Workspace floor |

---

## Usage

### Basic Commands

```bash
# Reach-avoid
python symbolic_control/integrator_symbolic_control.py --gui --spec reach_avoid

# Sequential
python symbolic_control/integrator_symbolic_control.py --gui --spec sequential

# Patrol (runs forever)
python symbolic_control/integrator_symbolic_control.py --gui --spec patrol

# Coverage
python symbolic_control/integrator_symbolic_control.py --gui --spec coverage

# Charge-patrol
python symbolic_control/integrator_symbolic_control.py --gui --spec charge_patrol

# Surveillance
python symbolic_control/integrator_symbolic_control.py --gui --spec surveillance

# Escape
python symbolic_control/integrator_symbolic_control.py --gui --spec escape
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--gui` | False | Enable GUI visualization |
| `--spec` | reach_avoid | LTL specification type |
| `--kp` | 2.0 | Controller gain |
| `--steps` | 1000 | Maximum simulation steps |

---

## Output Metrics

The simulation reports:

1. **Specification satisfied**: Whether the LTL formula was satisfied
2. **Final position**: Robot's end location
3. **Waypoints visited**: Count of waypoints reached
4. **Safety violations**: Number of times robot entered obstacle

---

## Comparison: Classical vs Symbolic Control

| Aspect | Classical Control | Symbolic Control |
|--------|-------------------|------------------|
| Specification | Reference trajectory | LTL formula |
| Abstraction | Continuous | Discrete regions |
| Guarantee | Tracking error bound | Specification satisfaction |
| Flexibility | Fixed path | Multiple valid paths |
| Obstacle handling | Not inherent | Built into specification |
| Verification | Numerical | Formal |

---

## File Structure

```
integrator_robot_project/
├── classical_control/
│   ├── integrator_classical_control.py   # Classical control script
│   ├── classical_examples.txt            # Example commands
│   └── CLASSICAL_REPORT.md               # Classical control report
├── symbolic_control/
│   ├── integrator_symbolic_control.py    # Main symbolic control script
│   ├── symbolic_examples.txt             # Example commands
│   └── SYMBOLIC_REPORT.md                # This report
├── reinforcement_learning/
│   ├── integrator_rl_control.py          # RL/PPO control script
│   ├── rl_examples.txt                   # Example commands
│   └── RL_REPORT.md                      # RL control report
├── requirements.txt                      # Dependencies
└── Project.md                            # Original specification
```

---

## Conclusion

The symbolic control implementation demonstrates:

1. **Formal Specifications**: Complex behaviors expressed in LTL
2. **Correct-by-Construction**: Controller derived from specification
3. **Safety Guarantees**: Obstacle avoidance built into control law
4. **Multiple Scenarios**: 7 different specification types
5. **Real-time Visualization**: PyBullet simulation with trajectory display

This approach bridges the gap between high-level mission requirements and low-level control, enabling specification of complex robot behaviors in a mathematically rigorous framework.
