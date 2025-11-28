"""
Reference trajectory generators for the unicycle robot
Generates desired paths: straight lines, circles, figure-8, etc.
"""

import numpy as np
import matplotlib.pyplot as plt

class ReferenceTrajectory:
    """Base class for reference trajectories"""
    
    def __init__(self, dt=0.1):
        """
        Args:
            dt: Time step for trajectory sampling
        """
        self.dt = dt
        self.t = 0.0
    
    def get_reference(self, t):
        """
        Get reference state at time t
        
        Returns:
            x_ref, y_ref, theta_ref, v_ref, omega_ref
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset trajectory to start"""
        self.t = 0.0

class StraightLine(ReferenceTrajectory):
    """
    Straight line trajectory
    
    Equation:
        x(t) = x₀ + v·t·cos(θ₀)
        y(t) = y₀ + v·t·sin(θ₀)
        θ(t) = θ₀
    """
    
    def __init__(self, start=[1.0, 1.0], angle=0.0, velocity=0.5, dt=0.1):
        """
        Args:
            start: Starting position [x, y]
            angle: Direction angle in radians
            velocity: Constant velocity
        """
        super().__init__(dt)
        self.x0, self.y0 = start
        self.theta0 = angle
        self.velocity = velocity
        
        print(f"Straight line: start=({self.x0}, {self.y0}), angle={np.degrees(angle):.1f}°, v={velocity}")
    
    def get_reference(self, t):
        """
        Get reference state at time t
        
        Returns:
            x_ref, y_ref, theta_ref, v_ref, omega_ref
        """
        x_ref = self.x0 + self.velocity * t * np.cos(self.theta0)
        y_ref = self.y0 + self.velocity * t * np.sin(self.theta0)
        theta_ref = self.theta0
        v_ref = self.velocity
        omega_ref = 0.0
        
        return x_ref, y_ref, theta_ref, v_ref, omega_ref

class CircularPath(ReferenceTrajectory):
    """
    Circular trajectory
    
    Equations:
        x(t) = xc + R·cos(ω·t + φ₀)
        y(t) = yc + R·sin(ω·t + φ₀)
        θ(t) = ω·t + φ₀ + π/2
        v(t) = R·ω
        ω(t) = ω
    
    Where:
        (xc, yc) = center of circle
        R = radius
        ω = angular velocity
        φ₀ = starting phase
    """
    
    def __init__(self, center=[5.0, 5.0], radius=2.0, angular_velocity=0.3, 
                 start_angle=0.0, dt=0.1):
        """
        Args:
            center: Center of circle [x, y]
            radius: Radius of circle
            angular_velocity: Angular velocity (rad/s)
            start_angle: Starting angle on circle
        """
        super().__init__(dt)
        self.xc, self.yc = center
        self.radius = radius
        self.omega = angular_velocity
        self.phi0 = start_angle
        
        print(f"Circular path: center=({self.xc}, {self.yc}), R={radius}, ω={angular_velocity}")
    
    def get_reference(self, t):
        """
        Get reference state at time t
        """
        phase = self.omega * t + self.phi0
        
        x_ref = self.xc + self.radius * np.cos(phase)
        y_ref = self.yc + self.radius * np.sin(phase)
        theta_ref = phase + np.pi/2  # Tangent to circle
        v_ref = self.radius * self.omega  # v = R·ω
        omega_ref = self.omega
        
        return x_ref, y_ref, theta_ref, v_ref, omega_ref

class Figure8(ReferenceTrajectory):
    """
    Figure-8 (lemniscate) trajectory
    
    Parametric equations:
        x(t) = xc + a·sin(ω·t)
        y(t) = yc + a·sin(ω·t)·cos(ω·t)
        
    This creates a figure-8 shape (∞)
    """
    
    def __init__(self, center=[5.0, 5.0], scale=2.0, angular_velocity=0.3, dt=0.1):
        """
        Args:
            center: Center of figure-8 [x, y]
            scale: Size scale factor
            angular_velocity: Speed of traversal
        """
        super().__init__(dt)
        self.xc, self.yc = center
        self.a = scale
        self.omega = angular_velocity
        
        print(f"Figure-8: center=({self.xc}, {self.yc}), scale={scale}, ω={angular_velocity}")
    
    def get_reference(self, t):
        """
        Get reference state at time t
        """
        phase = self.omega * t
        
        # Position
        x_ref = self.xc + self.a * np.sin(phase)
        y_ref = self.yc + self.a * np.sin(phase) * np.cos(phase)
        
        # Velocity (derivatives)
        dx_dt = self.a * self.omega * np.cos(phase)
        dy_dt = self.a * self.omega * (np.cos(2*phase))
        
        # Heading angle
        theta_ref = np.arctan2(dy_dt, dx_dt)
        
        # Linear velocity
        v_ref = np.sqrt(dx_dt**2 + dy_dt**2)
        
        # Angular velocity (derivative of theta)
        # Using finite difference approximation
        dt_small = 0.01
        theta_next = np.arctan2(
            self.a * self.omega * np.cos(2*(phase + dt_small)),
            self.a * self.omega * np.cos(phase + dt_small)
        )
        omega_ref = (theta_next - theta_ref) / dt_small
        
        return x_ref, y_ref, theta_ref, v_ref, omega_ref

class WaypointPath(ReferenceTrajectory):
    """
    Path through a series of waypoints with smooth interpolation
    """
    
    def __init__(self, waypoints, velocity=0.5, dt=0.1):
        """
        Args:
            waypoints: List of [x, y] waypoints
            velocity: Desired velocity between waypoints
        """
        super().__init__(dt)
        self.waypoints = np.array(waypoints)  # Convert to numpy array
        self.velocity = velocity
        self.current_segment = 0
        self.segment_start_time = 0.0
        
        # Calculate segment lengths and durations
        self.segment_lengths = []
        self.segment_durations = []
        
        for i in range(len(self.waypoints) - 1):
            p1 = np.array(self.waypoints[i])  # Convert to numpy array
            p2 = np.array(self.waypoints[i+1])  # Convert to numpy array
            length = np.linalg.norm(p2 - p1)
            duration = length / velocity
            self.segment_lengths.append(length)
            self.segment_durations.append(duration)
        
        print(f"Waypoint path: {len(waypoints)} waypoints, v={velocity}")
        
    def get_reference(self, t):
        """
        Get reference state at time t
        """
        # Find current segment
        cumulative_time = 0
        for i, duration in enumerate(self.segment_durations):
            if t < cumulative_time + duration:
                self.current_segment = i
                segment_time = t - cumulative_time
                break
            cumulative_time += duration
        else:
            # Reached end
            self.current_segment = len(self.segment_durations) - 1
            segment_time = self.segment_durations[-1]
        
        # Interpolate between waypoints
        p1 = self.waypoints[self.current_segment]
        p2 = self.waypoints[self.current_segment + 1]
        
        # Linear interpolation parameter
        alpha = segment_time / self.segment_durations[self.current_segment]
        alpha = np.clip(alpha, 0, 1)
        
        x_ref = p1[0] + alpha * (p2[0] - p1[0])
        y_ref = p1[1] + alpha * (p2[1] - p1[1])
        
        # Heading towards next waypoint
        theta_ref = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        
        v_ref = self.velocity
        omega_ref = 0.0  # Simplified: no turning within segments
        
        return x_ref, y_ref, theta_ref, v_ref, omega_ref


def plot_trajectory(trajectory, duration=20.0, dt=0.1):
    """
    Plot a reference trajectory
    
    Args:
        trajectory: ReferenceTrajectory object
        duration: Total time to plot
        dt: Time step
    """
    times = np.arange(0, duration, dt)
    x_vals, y_vals, theta_vals = [], [], []
    
    for t in times:
        x, y, theta, v, omega = trajectory.get_reference(t)
        x_vals.append(x)
        y_vals.append(y)
        theta_vals.append(theta)
    
    plt.figure(figsize=(10, 5))
    
    # Plot path
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, y_vals, 'b-', linewidth=2)
    plt.plot(x_vals[0], y_vals[0], 'go', markersize=10, label='Start')
    plt.plot(x_vals[-1], y_vals[-1], 'ro', markersize=10, label='End')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Reference Trajectory')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    # Plot heading
    plt.subplot(1, 2, 2)
    plt.plot(times, np.degrees(theta_vals), 'r-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Heading (degrees)')
    plt.title('Reference Heading')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# Test trajectories if run directly
if __name__ == "__main__":
    print("Testing trajectory generators...\n")
    
    # Test 1: Straight line
    print("1. Straight Line:")
    traj1 = StraightLine(start=[1, 1], angle=np.pi/4, velocity=0.5)
    plot_trajectory(traj1, duration=10)
    
    # Test 2: Circle
    print("\n2. Circular Path:")
    traj2 = CircularPath(center=[5, 5], radius=2, angular_velocity=0.3)
    plot_trajectory(traj2, duration=21)
    
    # Test 3: Figure-8
    print("\n3. Figure-8:")
    traj3 = Figure8(center=[5, 5], scale=2, angular_velocity=0.3)
    plot_trajectory(traj3, duration=21)
    
    # Test 4: Waypoints
    print("\n4. Waypoint Path:")
    waypoints = [[1, 1], [3, 4], [6, 4], [8, 7], [5, 8]]
    traj4 = WaypointPath(waypoints, velocity=0.5)
    plot_trajectory(traj4, duration=20)