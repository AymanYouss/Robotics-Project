"""
Test trajectory tracking in PyBullet simulation

This integrates:
    - Scenario (campus delivery)
    - Reference trajectory (waypoints)
    - Controller (feedback linearization)
    - Robot simulation (unicycle model)
    - Visualization (PyBullet)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt

from models.unicycle import UnicycleRobot
from controllers.trajectory_tracker import TrajectoryTracker
from utils.scenarios import CampusDeliveryScenario, WarehousePatrolScenario, AgriculturalFieldScenario

def create_environment_from_scenario(scenario):
    """Create PyBullet environment with scenario obstacles"""
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Ground
    planeId = p.loadURDF("plane.urdf")
    p.changeVisualShape(planeId, -1, rgbaColor=[0.9, 0.9, 0.9, 1])
    
    # Camera
    p.resetDebugVisualizerCamera(
        cameraDistance=15,
        cameraYaw=0,
        cameraPitch=-60,
        cameraTargetPosition=[5, 5, 0]
    )
    
    # Grid
    for x in range(11):
        p.addUserDebugLine([x, 0, 0.01], [x, 10, 0.01], [0.7, 0.7, 0.7], 1)
    for y in range(11):
        p.addUserDebugLine([0, y, 0.01], [10, y, 0.01], [0.7, 0.7, 0.7], 1)
    
    # Add obstacles from scenario
    for obs_pos, obs_size in scenario.get_obstacles_for_world():
        collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=obs_size, height=0.5)
        visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=obs_size, length=0.5,
                                          rgbaColor=[0.8, 0.2, 0.2, 1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape,
                         baseVisualShapeIndex=visual_shape,
                         basePosition=[obs_pos[0], obs_pos[1], 0.25])
    
    # Mark waypoints
    for i, (name, wp) in enumerate(zip(
        ['START', 'Bldg A', 'Library', 'Student Ctr', 'Engineering', 'END'],
        scenario.route
    )):
        # Small marker at each waypoint
        marker_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.15,
                                          rgbaColor=[0.2, 0.8, 0.2, 0.8])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=marker_shape,
                         basePosition=[wp[0], wp[1], 0.1])
        # Label
        p.addUserDebugText(name, [wp[0], wp[1], 0.5], textColorRGB=[0, 0, 0], textSize=1.2)
    
    print("Environment created with scenario elements")

def simulate_tracking(robot, controller, trajectory, duration=60.0, dt=0.1):
    """
    Simulate trajectory tracking
    
    Args:
        robot: UnicycleRobot instance
        controller: TrajectoryTracker instance
        trajectory: ReferenceTrajectory instance
        duration: Simulation duration (seconds)
        dt: Time step
    
    Returns:
        robot_trajectory, reference_trajectory, time_array
    """
    robot_traj = []
    ref_traj = []
    time_array = []
    
    t = 0.0
    steps = int(duration / dt)
    
    print(f"\nStarting simulation ({duration}s)...")
    
    for step in range(steps):
        # Get reference state
        x_ref, y_ref, theta_ref, v_ref, omega_ref = trajectory.get_reference(t)
        reference = [x_ref, y_ref, theta_ref, v_ref, omega_ref]
        
        # Get current robot state
        state = robot.get_state()
        
        # Compute control
        u1, u2 = controller.compute_control(state, reference, t)
        
        # Apply control to robot
        robot.step(u1, u2)
        
        # Step simulation
        p.stepSimulation()
        time.sleep(dt * 0.5)  # Half speed for visualization
        
        # Log data
        robot_traj.append([state[0], state[1]])
        ref_traj.append([x_ref, y_ref])
        time_array.append(t)
        
        # Progress printout
        if step % int(5.0/dt) == 0:  # Every 5 seconds
            pos_error = np.sqrt((x_ref - state[0])**2 + (y_ref - state[1])**2)
            print(f"t={t:5.1f}s | Pos: ({state[0]:.2f}, {state[1]:.2f}) | "
                  f"Ref: ({x_ref:.2f}, {y_ref:.2f}) | Error: {pos_error:.3f}m")
        
        t += dt
    
    # Draw final trajectory
    robot.draw_trajectory()
    
    print("\nSimulation complete!")
    
    return robot_traj, ref_traj, time_array

def plot_results(robot_traj, ref_traj, time_array, controller, scenario):
    """Plot tracking performance"""
    robot_traj = np.array(robot_traj)
    ref_traj = np.array(ref_traj)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Trajectories
    ax = axes[0, 0]
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], 'b--', linewidth=2, label='Reference', alpha=0.7)
    ax.plot(robot_traj[:, 0], robot_traj[:, 1], 'r-', linewidth=2, label='Actual')
    for wp in scenario.route:
        ax.plot(wp[0], wp[1], 'go', markersize=10)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Trajectories')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # Plot 2: Position errors over time
    ax = axes[0, 1]
    errors = controller.error_history
    ax.plot(errors['time'], errors['ex'], label='$e_x$ (longitudinal)')
    ax.plot(errors['time'], errors['ey'], label='$e_y$ (lateral)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (m)')
    ax.set_title('Position Tracking Errors')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Heading error over time
    ax = axes[1, 0]
    ax.plot(errors['time'], np.degrees(errors['etheta']), 'r-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading Error (degrees)')
    ax.set_title('Heading Tracking Error')
    ax.grid(True)
    
    # Plot 4: Total position error
    ax = axes[1, 1]
    pos_errors = np.sqrt(np.array(errors['ex'])**2 + np.array(errors['ey'])**2)
    ax.plot(errors['time'], pos_errors, 'g-', linewidth=2)
    ax.axhline(y=scenario.safe_distance, color='r', linestyle='--', label='Safe distance threshold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Total Position Error')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main simulation"""
    print("="*70)
    print("TRAJECTORY TRACKING TEST - CAMPUS DELIVERY SCENARIO")
    print("="*70)
    
    # Create scenario
    scenario = AgriculturalFieldScenario()
    
    # Get reference trajectory
    trajectory = scenario.get_reference_trajectory()
    
    # Create environment
    create_environment_from_scenario(scenario)
    
    # Create robot at starting position
    start_pos = scenario.route[0] + [0.0]  # [x, y, theta]
    robot = UnicycleRobot(start_pos=start_pos, tau=0.1, add_disturbance=True)
    
    # Create controller
    print("\nInitializing controller...")
    controller = TrajectoryTracker(kx=1.0, ky=1.5, ktheta=2.0)
    
    input("\nPress Enter to start tracking...")
    
    # Run simulation
    start_time = time.time()
    robot_traj, ref_traj, time_array = simulate_tracking(
        robot, controller, trajectory, duration=50.0, dt=0.1
    )
    end_time = time.time()
    
    # Print statistics
    controller.print_statistics()
    
    # Evaluate performance
    results = scenario.evaluate_performance(robot_traj, ref_traj, time_array[-1])
    scenario.print_performance_report(results)
    
    input("\nPress Enter to show plots...")
    
    # Plot results
    plot_results(robot_traj, ref_traj, time_array, controller, scenario)
    
    input("\nPress Enter to close...")
    p.disconnect()

if __name__ == "__main__":
    main()