#!/usr/bin/env python3
"""
Minimal PyBullet demo: 2D Integrator Control
Implements the discrete-time integrator model with constraints.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import argparse

# System parameters
TAU = 0.1  # Sampling period
X_BOUNDS = [-10, 10]  # State constraints
U_BOUNDS = [-1, 1]    # Input constraints  
W_BOUNDS = [-0.05, 0.05]  # Disturbance bounds

def clip_state(x):
    """Clip state to X bounds."""
    return np.clip(x, X_BOUNDS[0], X_BOUNDS[1])

def clip_input(u):
    """Clip control input to U bounds."""
    return np.clip(u, U_BOUNDS[0], U_BOUNDS[1])

def get_disturbance():
    """Random disturbance within W bounds."""
    return np.random.uniform(W_BOUNDS[0], W_BOUNDS[1], size=2)

class PIDController:
    """PID controller for trajectory tracking."""
    
    def __init__(self, kp=2.0, ki=0.1, kd=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
    
    def compute(self, x, x_ref):
        """Compute PID control signal."""
        error = x_ref - x
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (with anti-windup)
        self.integral += error * TAU
        self.integral = np.clip(self.integral, -5, 5)  # Anti-windup
        i_term = self.ki * self.integral
        
        # Derivative term
        d_term = self.kd * (error - self.prev_error) / TAU
        self.prev_error = error.copy()
        
        # Total control
        u = p_term + i_term + d_term
        return clip_input(u)
    
    def reset(self):
        """Reset controller state."""
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)

def integrator_dynamics(x, u, w):
    """
    Discrete-time integrator:
    x1(t+1) = x1(t) + tau*(u1(t) + w1(t))
    x2(t+1) = x2(t) + tau*(u2(t) + w2(t))
    """
    u_clipped = clip_input(u)
    x_next = x + TAU * (u_clipped + w)
    return clip_state(x_next)

def draw_actual_trajectory(actual_path):
    """Draw the actual trajectory followed by the robot (red color)."""
    for i in range(len(actual_path) - 1):
        p.addUserDebugLine(
            [actual_path[i][0], actual_path[i][1], 0.1],
            [actual_path[i+1][0], actual_path[i+1][1], 0.1],
            [1, 0, 0], lineWidth=3  # Red color
        )

def generate_linear_trajectory(start, goal, steps):
    """Linear trajectory from start to goal."""
    trajectory = []
    for i in range(steps):
        alpha = i / (steps - 1) if steps > 1 else 1.0
        point = (1 - alpha) * start + alpha * goal
        trajectory.append(point)
    return trajectory

def generate_circular_trajectory(center, radius, steps):
    """Circular trajectory around center point."""
    trajectory = []
    # Use more points for smoother circle
    num_points = max(steps, 500)
    for i in range(num_points):
        theta = 2 * np.pi * i / num_points
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        trajectory.append(np.array([x, y]))
    return trajectory

def generate_infinity_trajectory(center, scale, steps):
    """Infinity (lemniscate) shaped trajectory."""
    trajectory = []
    # Use more points for smoother infinity
    num_points = max(steps, 500)
    for i in range(num_points):
        t = 2 * np.pi * i / num_points
        # Lemniscate of Bernoulli parametric equations
        denom = 1 + np.sin(t)**2
        x = center[0] + scale * np.cos(t) / denom
        y = center[1] + scale * np.sin(t) * np.cos(t) / denom
        trajectory.append(np.array([x, y]))
    return trajectory

def generate_trajectory(traj_type, start, goal, steps, center, radius, scale):
    """Generate trajectory based on type."""
    if traj_type == "linear":
        return generate_linear_trajectory(start, goal, steps)
    elif traj_type == "circular":
        return generate_circular_trajectory(center, radius, steps)
    elif traj_type == "infinity":
        return generate_infinity_trajectory(center, scale, steps)
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

def main():
    parser = argparse.ArgumentParser(description="2D Integrator PyBullet Demo")
    parser.add_argument("--gui", action="store_true", help="Enable GUI mode")
    parser.add_argument("--start", nargs=2, type=float, default=[-8, -8], help="Start position")
    parser.add_argument("--goal", nargs=2, type=float, default=[8, 8], help="Goal position")
    parser.add_argument("--kp", type=float, default=2.0, help="Proportional gain")
    parser.add_argument("--ki", type=float, default=0.1, help="Integral gain")
    parser.add_argument("--kd", type=float, default=0.5, help="Derivative gain")
    parser.add_argument("--steps", type=int, default=600, help="Simulation steps")
    parser.add_argument("--horizon", type=int, default=500, help="Trajectory horizon")
    parser.add_argument("--keep-running", action="store_true", help="Don't stop at goal")
    parser.add_argument("--trajectory", type=str, default="linear", 
                        choices=["linear", "circular", "infinity"], help="Trajectory type")
    parser.add_argument("--center", nargs=2, type=float, default=[0, 0], help="Center for circular/infinity")
    parser.add_argument("--radius", type=float, default=5.0, help="Radius for circular trajectory")
    parser.add_argument("--scale", type=float, default=6.0, help="Scale for infinity trajectory")
    args = parser.parse_args()

    # Initialize PyBullet
    if args.gui:
        physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    else:
        physics_client = p.connect(p.DIRECT)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    
    # Load ground plane
    plane_id = p.loadURDF("plane.urdf")
    
    # Create robot (simple sphere)
    robot_radius = 0.3
    robot_visual = p.createVisualShape(p.GEOM_SPHERE, radius=robot_radius, rgbaColor=[0, 0.5, 1, 1])
    robot_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=robot_radius)
    
    start = np.array(args.start)
    goal = np.array(args.goal)
    center = np.array(args.center)
    
    # Set initial position based on trajectory type
    if args.trajectory == "circular":
        start = np.array([center[0] + args.radius, center[1]])
    elif args.trajectory == "infinity":
        start = np.array([center[0] + args.scale, center[1]])
    
    robot_id = p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=robot_collision,
        baseVisualShapeIndex=robot_visual,
        basePosition=[start[0], start[1], robot_radius]
    )
    
    # Create goal marker (green sphere) - only for linear trajectory
    if args.trajectory == "linear":
        goal_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.4, rgbaColor=[0, 1, 0, 0.5])
        goal_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=goal_visual,
            basePosition=[goal[0], goal[1], 0.4]
        )
    
    # Draw workspace bounds
    corners = [
        [X_BOUNDS[0], X_BOUNDS[0], 0.1],
        [X_BOUNDS[1], X_BOUNDS[0], 0.1],
        [X_BOUNDS[1], X_BOUNDS[1], 0.1],
        [X_BOUNDS[0], X_BOUNDS[1], 0.1],
    ]
    for i in range(4):
        p.addUserDebugLine(corners[i], corners[(i+1)%4], [1, 0, 0], lineWidth=2)
    
    # Generate reference trajectory
    trajectory = generate_trajectory(args.trajectory, start, goal, args.horizon, 
                                     center, args.radius, args.scale)
    
    # Draw reference trajectory (yellow)
    for i in range(len(trajectory) - 1):
        p.addUserDebugLine(
            [trajectory[i][0], trajectory[i][1], 0.05],
            [trajectory[i+1][0], trajectory[i+1][1], 0.05],
            [1, 1, 0], lineWidth=2
        )
    
    # Set camera
    if args.gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=25,
            cameraYaw=0,
            cameraPitch=-89,
            cameraTargetPosition=[0, 0, 0]
        )
    
    # Simulation state
    x = start.copy()
    tolerance = 0.5
    actual_path = [start.copy()]  # Store actual trajectory
    prev_pos = start.copy()  # For drawing trajectory incrementally
    
    # Initialize PID controller
    pid = PIDController(kp=args.kp, ki=args.ki, kd=args.kd)
    
    # For circular/infinity, stop after one complete loop
    if args.trajectory in ["circular", "infinity"]:
        max_steps = args.horizon  # One full loop
    else:
        max_steps = args.steps
    
    print(f"Trajectory: {args.trajectory}")
    print(f"Start: {start}")
    print(f"PID gains: Kp={args.kp}, Ki={args.ki}, Kd={args.kd}")
    if args.trajectory == "linear":
        print(f"Goal: {goal}")
    else:
        print(f"Center: {center}")
    print("Running simulation...")
    
    for step in range(max_steps):
        # Get reference point
        ref_idx = step % len(trajectory)
        x_ref = trajectory[ref_idx]
        
        # Compute PID control
        u = pid.compute(x, x_ref)
        
        # Get disturbance
        w = get_disturbance()
        
        # Apply dynamics
        x = integrator_dynamics(x, u, w)
        actual_path.append(x.copy())
        
        # Draw actual trajectory segment incrementally (red)
        p.addUserDebugLine(
            [prev_pos[0], prev_pos[1], 0.1],
            [x[0], x[1], 0.1],
            [1, 0, 0], lineWidth=3
        )
        prev_pos = x.copy()
        
        # Update robot position in PyBullet
        p.resetBasePositionAndOrientation(
            robot_id,
            [x[0], x[1], robot_radius],
            [0, 0, 0, 1]
        )
        
        # Check goal reached (only for linear trajectory)
        if args.trajectory == "linear":
            dist_to_goal = np.linalg.norm(x - goal)
            if dist_to_goal < tolerance and not args.keep_running:
                print(f"Goal reached at step {step}! Distance: {dist_to_goal:.3f}")
                break
        
        if args.gui:
            time.sleep(TAU)
        
        p.stepSimulation()
    
    print(f"Final position: {x}")
    if args.trajectory == "linear":
        print(f"Distance to goal: {np.linalg.norm(x - goal):.3f}")
    
    # Calculate tracking error
    tracking_errors = []
    for i, pos in enumerate(actual_path):
        ref_idx = i % len(trajectory)
        error = np.linalg.norm(pos - trajectory[ref_idx])
        tracking_errors.append(error)
    avg_error = np.mean(tracking_errors)
    max_error = np.max(tracking_errors)
    print(f"\nTrajectory complete!")
    print(f"Average tracking error: {avg_error:.4f}")
    print(f"Maximum tracking error: {max_error:.4f}")
    
    # Keep window open
    if args.gui:
        print("\nPress Ctrl+C to exit...")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    
    p.disconnect()

if __name__ == "__main__":
    main()
