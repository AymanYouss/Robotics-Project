"""
Unicycle robot model with discrete-time dynamics and disturbances
Implements the equations:
    x1(t+1) = x1(t) + τ(u1(t)cos(x3(t)) + w1(t))
    x2(t+1) = x2(t) + τ(u1(t)sin(x3(t)) + w2(t))
    x3(t+1) = x3(t) + τ(u2(t) + w3(t)) (mod 2π)
"""

import pybullet as p
import numpy as np

class UnicycleRobot:
    """
    A unicycle robot with discrete-time dynamics
    
    State: [x1, x2, x3] = [x_position, y_position, heading_angle]
    Control: [u1, u2] = [linear_velocity, angular_velocity]
    Disturbances: [w1, w2, w3] = position and angular disturbances
    """
    
    def __init__(self, start_pos=[1.0, 1.0, 0.0], tau=0.1, add_disturbance=True):
        """
        Initialize the unicycle robot
        
        Args:
            start_pos: Initial position [x, y, theta] where theta is in radians
            tau: Sampling period (time step) in seconds
            add_disturbance: Whether to add random disturbances
        """
        # Sampling period
        self.tau = tau
        
        # State variables
        self.x1 = start_pos[0]  # x position
        self.x2 = start_pos[1]  # y position
        self.x3 = start_pos[2]  # heading angle (radians)
        
        # State constraints: X = [0,10] × [0,10] × [-π, π]
        self.x_min, self.x_max = 0.0, 10.0
        self.y_min, self.y_max = 0.0, 10.0
        self.theta_min, self.theta_max = -np.pi, np.pi
        
        # Input constraints: U = [0.25, 1] × [-1, 1]
        self.u1_min, self.u1_max = 0.25, 1.0    # linear velocity bounds
        self.u2_min, self.u2_max = -1.0, 1.0    # angular velocity bounds
        
        # Disturbance bounds: W = [-0.05, 0.05]³
        self.w_min, self.w_max = -0.05, 0.05
        self.add_disturbance = add_disturbance
        
        # Robot physical properties
        self.radius = 0.2   # 20cm radius
        self.height = 0.3   # 30cm height
        
        # Create the robot in PyBullet
        self._create_robot()
        
        # Trajectory history for visualization
        self.trajectory = [[self.x1, self.x2]]
        
        print(f"Unicycle robot created at position ({self.x1:.2f}, {self.x2:.2f}, {np.degrees(self.x3):.1f}°)")
    
    def _create_robot(self):
        """
        Create the robot's visual and collision shapes in PyBullet
        """
        # Create collision shape (cylinder)
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=self.radius,
            height=self.height
        )
        
        # Create visual shape (green cylinder with a direction indicator)
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.radius,
            length=self.height,
            rgbaColor=[0.2, 0.8, 0.2, 1]  # Green
        )
        
        # Create the robot body
        self.robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[self.x1, self.x2, self.height/2]
        )
        
        # Add a direction indicator (a small red box pointing forward)
        self._add_direction_indicator()
        
        print("Robot body created in PyBullet")
    
    def _add_direction_indicator(self):
        """
        Add a small box on the robot to show which way it's facing
        """
        # Create a small box
        indicator_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.radius * 0.6, 0.05, 0.05],
            rgbaColor=[1, 0, 0, 1]  # Red
        )
        
        # Position it in front of the robot
        self.indicator_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=indicator_shape,
            basePosition=[
                self.x1 + self.radius * np.cos(self.x3),
                self.x2 + self.radius * np.sin(self.x3),
                self.height/2
            ]
        )
    
    def _clip_control(self, u1, u2):
        """
        Clip control inputs to satisfy constraints
        
        Args:
            u1: Linear velocity (may be outside bounds)
            u2: Angular velocity (may be outside bounds)
        
        Returns:
            u1_clipped, u2_clipped: Control inputs within bounds
        """
        u1_clipped = np.clip(u1, self.u1_min, self.u1_max)
        u2_clipped = np.clip(u2, self.u2_min, self.u2_max)
        return u1_clipped, u2_clipped
    
    def _generate_disturbance(self):
        """
        Generate random disturbances w = [w1, w2, w3]
        
        Returns:
            w1, w2, w3: Random disturbances in [-0.05, 0.05]
        """
        if self.add_disturbance:
            w1 = np.random.uniform(self.w_min, self.w_max)
            w2 = np.random.uniform(self.w_min, self.w_max)
            w3 = np.random.uniform(self.w_min, self.w_max)
        else:
            w1, w2, w3 = 0.0, 0.0, 0.0
        
        return w1, w2, w3
    
    def _normalize_angle(self, theta):
        """
        Normalize angle to [-π, π]
        
        Args:
            theta: Angle in radians
        
        Returns:
            Normalized angle in [-π, π]
        """
        # Map to [-π, π]
        return np.arctan2(np.sin(theta), np.cos(theta))
    
    def step(self, u1, u2):
        """
        Update robot state using discrete-time unicycle dynamics
        
        EQUATIONS:
            x1(t+1) = x1(t) + τ(u1(t)cos(x3(t)) + w1(t))
            x2(t+1) = x2(t) + τ(u1(t)sin(x3(t)) + w2(t))
            x3(t+1) = x3(t) + τ(u2(t) + w3(t)) (mod 2π)
        
        Args:
            u1: Linear velocity command
            u2: Angular velocity command
        
        Returns:
            state: New state [x1, x2, x3]
        """
        # Clip controls to satisfy input constraints
        u1, u2 = self._clip_control(u1, u2)
        
        # Generate disturbances
        w1, w2, w3 = self._generate_disturbance()
        
        # Apply discrete-time dynamics equations
        # x1(t+1) = x1(t) + τ(u1(t)cos(x3(t)) + w1(t))
        self.x1 = self.x1 + self.tau * (u1 * np.cos(self.x3) + w1)
        
        # x2(t+1) = x2(t) + τ(u1(t)sin(x3(t)) + w2(t))
        self.x2 = self.x2 + self.tau * (u1 * np.sin(self.x3) + w2)
        
        # x3(t+1) = x3(t) + τ(u2(t) + w3(t)) (mod 2π)
        self.x3 = self.x3 + self.tau * (u2 + w3)
        self.x3 = self._normalize_angle(self.x3)  # Keep in [-π, π]
        
        # Enforce state constraints (keep robot in bounds)
        self.x1 = np.clip(self.x1, self.x_min, self.x_max)
        self.x2 = np.clip(self.x2, self.y_min, self.y_max)
        
        # Update PyBullet visualization
        self._update_visualization()
        
        # Store trajectory
        self.trajectory.append([self.x1, self.x2])
        
        return self.get_state()
    
    def _update_visualization(self):
        """
        Update the robot's position and orientation in PyBullet
        """
        # Convert heading angle to quaternion for PyBullet
        quaternion = p.getQuaternionFromEuler([0, 0, self.x3])
        
        # Update robot position and orientation
        p.resetBasePositionAndOrientation(
            self.robot_id,
            [self.x1, self.x2, self.height/2],
            quaternion
        )
        
        # Update direction indicator
        p.resetBasePositionAndOrientation(
            self.indicator_id,
            [
                self.x1 + self.radius * 0.7 * np.cos(self.x3),
                self.x2 + self.radius * 0.7 * np.sin(self.x3),
                self.height/2
            ],
            quaternion
        )
    
    def get_state(self):
        """
        Get current state
        
        Returns:
            state: [x1, x2, x3] = [x_pos, y_pos, heading_angle]
        """
        return np.array([self.x1, self.x2, self.x3])
    
    def reset(self, pos=[1.0, 1.0, 0.0]):
        """
        Reset robot to a new position
        
        Args:
            pos: New position [x, y, theta]
        """
        self.x1, self.x2, self.x3 = pos
        self.trajectory = [[self.x1, self.x2]]
        self._update_visualization()
        print(f"Robot reset to ({self.x1:.2f}, {self.x2:.2f}, {np.degrees(self.x3):.1f}°)")
    
    def draw_trajectory(self):
        """
        Draw the robot's trajectory as a line in PyBullet
        """
        if len(self.trajectory) < 2:
            return
        
        # Draw lines between consecutive points
        for i in range(len(self.trajectory) - 1):
            p.addUserDebugLine(
                lineFromXYZ=[self.trajectory[i][0], self.trajectory[i][1], 0.05],
                lineToXYZ=[self.trajectory[i+1][0], self.trajectory[i+1][1], 0.05],
                lineColorRGB=[0, 0, 1],  # Blue trail
                lineWidth=2,
                lifeTime=0  # Permanent
            )


# Test the robot if this file is run directly
if __name__ == "__main__":
    import sys
    import os
    import time
    
    # Get the path to the src directory
    current_dir = os.path.dirname(os.path.abspath(__file__))  # models folder
    src_dir = os.path.dirname(current_dir)  # src folder
    
    # Add src directory to Python path
    sys.path.insert(0, src_dir)
    
    # Now import world
    from environment.world import RobotWorld
    import time
    
    print("Testing UnicycleRobot class...\n")
    
    # Create world
    world = RobotWorld(gui=True)
    world.add_grid_lines()
    
    # Create robot at position (2, 2) facing 45 degrees
    robot = UnicycleRobot(start_pos=[2.0, 2.0, np.pi/4], tau=0.1, add_disturbance=True)
    
    print("\nTest 1: Drive forward")
    print("Command: u1=0.5 (forward), u2=0 (no turn)")
    for i in range(50):  # 5 seconds
        robot.step(u1=0.5, u2=0.0)
        p.stepSimulation()
        time.sleep(0.1)
    
    print("\nTest 2: Turn left while moving")
    print("Command: u1=0.5 (forward), u2=0.5 (turn left)")
    for i in range(50):
        robot.step(u1=0.5, u2=0.5)
        p.stepSimulation()
        time.sleep(0.1)
    
    print("\nTest 3: Turn right while moving")
    print("Command: u1=0.5 (forward), u2=-0.5 (turn right)")
    for i in range(50):
        robot.step(u1=0.5, u2=-0.5)
        p.stepSimulation()
        time.sleep(0.1)
    
    # Draw the trajectory
    robot.draw_trajectory()
    
    print("\nTests complete!")
    print(f"Final position: ({robot.x1:.2f}, {robot.x2:.2f}, {np.degrees(robot.x3):.1f}°)")
    print("Blue line shows the robot's trajectory")
    print("\nWindow will stay open for 10 more seconds...")
    
    for i in range(10):
        p.stepSimulation()
        time.sleep(1)
    
    world.close()
    print("\nTest complete!")