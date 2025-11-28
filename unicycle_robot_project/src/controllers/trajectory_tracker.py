"""
Trajectory Tracking Controller for Unicycle Robot

Implements feedback linearization control with Lyapunov stability guarantees

MATHEMATICAL BACKGROUND:
========================

Error Dynamics in Robot Frame:
    ė_x = -v + v_r*cos(e_θ) + e_y*ω
    ė_y = -e_x*ω + v_r*sin(e_θ)  
    ė_θ = ω_r - ω

Control Law (with Lyapunov design):
    u_1 = v_r*cos(e_θ) + k_x*e_x
    u_2 = ω_r + k_y*v_r*e_y + k_θ*sin(e_θ)

Where:
    e_x, e_y, e_θ = tracking errors (longitudinal, lateral, heading)
    v_r, ω_r = reference velocities
    k_x, k_y, k_θ = positive control gains

LYAPUNOV STABILITY PROOF:
========================
Lyapunov function: V = (1/2)(e_x² + e_y² + e_θ²)

Time derivative: V̇ = -k_x*e_x² - k_y*v_r*e_y² - k_θ*sin²(e_θ)

Since k_x, k_y, k_θ > 0 and v_r > 0:
    V̇ < 0 for all e ≠ 0
    
Therefore: System is asymptotically stable → errors converge to zero!
"""

import numpy as np

class TrajectoryTracker:
    """
    Feedback controller for trajectory tracking
    
    Uses feedback linearization with Lyapunov-based design
    """
    
    def __init__(self, kx=1.0, ky=1.5, ktheta=2.0):
        """
        Initialize the trajectory tracker
        
        Args:
            kx: Longitudinal error gain (controls forward/backward error)
            ky: Lateral error gain (controls left/right error)
            ktheta: Heading error gain (controls orientation error)
            
        Tuning Guidelines:
            - Larger gains → faster convergence, but may cause oscillations
            - kx: Start with 1.0, increase for faster forward convergence
            - ky: Usually 1.5*kx for good lateral tracking
            - ktheta: Usually 2*kx for quick heading correction
        """
        self.kx = kx
        self.ky = ky
        self.ktheta = ktheta
        
        # For logging and analysis
        self.error_history = {
            'ex': [],
            'ey': [],
            'etheta': [],
            'time': []
        }
        
        print(f"Trajectory Tracker initialized:")
        print(f"  k_x = {kx} (longitudinal gain)")
        print(f"  k_y = {ky} (lateral gain)")
        print(f"  k_θ = {ktheta} (heading gain)")
    
    def compute_errors(self, state, reference):
        """
        Compute tracking errors in the robot's reference frame
        
        TRANSFORMATION:
        The errors are computed in the robot's local frame (not global frame)
        This makes the control problem simpler.
        
        Global errors:
            e_x_global = x_ref - x
            e_y_global = y_ref - y
            e_θ_global = θ_ref - θ
        
        Rotation to robot frame:
            [e_x]   [cos(θ)   sin(θ)  0] [x_ref - x]
            [e_y] = [-sin(θ)  cos(θ)  0] [y_ref - y]
            [e_θ]   [0        0       1] [θ_ref - θ]
        
        Args:
            state: [x, y, theta] - current robot state
            reference: [x_ref, y_ref, theta_ref, v_ref, omega_ref]
        
        Returns:
            ex, ey, etheta: Tracking errors in robot frame
        """
        x, y, theta = state
        x_ref, y_ref, theta_ref, v_ref, omega_ref = reference
        
        # Global position errors
        dx = x_ref - x
        dy = y_ref - y
        
        # Rotation matrix from global to robot frame
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Errors in robot frame
        ex = cos_theta * dx + sin_theta * dy    # Longitudinal error (forward/back)
        ey = -sin_theta * dx + cos_theta * dy   # Lateral error (left/right)
        
        # Heading error (normalized to [-π, π])
        etheta = theta_ref - theta
        etheta = np.arctan2(np.sin(etheta), np.cos(etheta))  # Wrap to [-π, π]
        
        return ex, ey, etheta
    
    def compute_control(self, state, reference, t=0.0):
        """
        Compute control inputs using feedback linearization
        
        CONTROL LAW DERIVATION:
        ======================
        Goal: Design u_1, u_2 such that errors → 0
        
        From error dynamics and Lyapunov analysis:
            u_1 = v_r*cos(e_θ) + k_x*e_x
            u_2 = ω_r + k_y*v_r*e_y + k_θ*sin(e_θ)
        
        Intuition:
            u_1: Base velocity from reference + correction proportional to forward error
            u_2: Base turning rate + corrections for lateral and heading errors
        
        Args:
            state: [x, y, theta] - current robot state
            reference: [x_ref, y_ref, theta_ref, v_ref, omega_ref]
            t: current time (for logging)
        
        Returns:
            u1, u2: Control inputs (linear velocity, angular velocity)
        """
        # Compute errors
        ex, ey, etheta = self.compute_errors(state, reference)
        
        # Extract reference velocities
        x_ref, y_ref, theta_ref, v_ref, omega_ref = reference
        
        # Control law (feedback linearization)
        u1 = v_ref * np.cos(etheta) + self.kx * ex
        u2 = omega_ref + self.ky * v_ref * ey + self.ktheta * np.sin(etheta)
        
        # Log errors for analysis
        self.error_history['ex'].append(ex)
        self.error_history['ey'].append(ey)
        self.error_history['etheta'].append(etheta)
        self.error_history['time'].append(t)
        
        return u1, u2
    
    def get_error_statistics(self):
        """
        Compute statistics of tracking errors
        
        Returns:
            Dictionary with mean, max, std of errors
        """
        if len(self.error_history['ex']) == 0:
            return None
        
        ex_arr = np.array(self.error_history['ex'])
        ey_arr = np.array(self.error_history['ey'])
        etheta_arr = np.array(self.error_history['etheta'])
        
        # Total position error
        pos_error = np.sqrt(ex_arr**2 + ey_arr**2)
        
        stats = {
            'mean_pos_error': np.mean(pos_error),
            'max_pos_error': np.max(pos_error),
            'std_pos_error': np.std(pos_error),
            'mean_heading_error': np.mean(np.abs(etheta_arr)),
            'max_heading_error': np.max(np.abs(etheta_arr)),
            'final_pos_error': pos_error[-1],
            'final_heading_error': np.abs(etheta_arr[-1])
        }
        
        return stats
    
    def reset(self):
        """Reset error history"""
        self.error_history = {
            'ex': [],
            'ey': [],
            'etheta': [],
            'time': []
        }
    
    def print_statistics(self):
        """Print error statistics"""
        stats = self.get_error_statistics()
        
        if stats is None:
            print("No tracking data available")
            return
        
        print("\n" + "="*60)
        print("TRACKING PERFORMANCE STATISTICS")
        print("="*60)
        print(f"Position Tracking:")
        print(f"  Mean error:   {stats['mean_pos_error']:.4f} m")
        print(f"  Max error:    {stats['max_pos_error']:.4f} m")
        print(f"  Std dev:      {stats['std_pos_error']:.4f} m")
        print(f"  Final error:  {stats['final_pos_error']:.4f} m")
        print(f"\nHeading Tracking:")
        print(f"  Mean error:   {np.degrees(stats['mean_heading_error']):.2f}°")
        print(f"  Max error:    {np.degrees(stats['max_heading_error']):.2f}°")
        print(f"  Final error:  {np.degrees(stats['final_heading_error']):.2f}°")
        print("="*60)


class PIDController:
    """
    PID Controller for trajectory tracking
    
    THEORY:
    =======
    PID = Proportional + Integral + Derivative
    
    u(t) = K_p*e(t) + K_i*∫e(τ)dτ + K_d*de(t)/dt
    
    Components:
        Proportional: Responds to current error
        Integral: Eliminates steady-state error
        Derivative: Dampens oscillations, predicts future error
    
    For unicycle robot:
        u_1 = PID on position error + v_ref
        u_2 = PID on heading error + ω_ref
    """
    
    def __init__(self, kp_pos=1.0, ki_pos=0.0, kd_pos=0.1,
                 kp_heading=2.0, ki_heading=0.0, kd_heading=0.5):
        """
        Initialize PID controller
        
        Args:
            kp_pos: Proportional gain for position
            ki_pos: Integral gain for position
            kd_pos: Derivative gain for position
            kp_heading: Proportional gain for heading
            ki_heading: Integral gain for heading
            kd_heading: Derivative gain for heading
        """
        # Position PID gains
        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kd_pos = kd_pos
        
        # Heading PID gains
        self.kp_heading = kp_heading
        self.ki_heading = ki_heading
        self.kd_heading = kd_heading
        
        # State variables
        self.integral_pos = 0.0
        self.integral_heading = 0.0
        self.prev_pos_error = 0.0
        self.prev_heading_error = 0.0
        self.dt = 0.1
        
        # History
        self.error_history = {
            'pos': [],
            'heading': [],
            'time': []
        }
        
        print(f"PID Controller initialized:")
        print(f"  Position: Kp={kp_pos}, Ki={ki_pos}, Kd={kd_pos}")
        print(f"  Heading:  Kp={kp_heading}, Ki={ki_heading}, Kd={kd_heading}")
    
    def compute_control(self, state, reference, t=0.0):
        """
        Compute control using PID
        
        Args:
            state: [x, y, theta]
            reference: [x_ref, y_ref, theta_ref, v_ref, omega_ref]
            t: current time
        
        Returns:
            u1, u2: Control inputs
        """
        x, y, theta = state
        x_ref, y_ref, theta_ref, v_ref, omega_ref = reference
        
        # Position error (Euclidean distance)
        pos_error = np.sqrt((x_ref - x)**2 + (y_ref - y)**2)
        
        # Heading error
        heading_error = theta_ref - theta
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Update integrals
        self.integral_pos += pos_error * self.dt
        self.integral_heading += heading_error * self.dt
        
        # Compute derivatives
        d_pos_error = (pos_error - self.prev_pos_error) / self.dt
        d_heading_error = (heading_error - self.prev_heading_error) / self.dt
        
        # PID control
        u1_correction = (self.kp_pos * pos_error + 
                        self.ki_pos * self.integral_pos + 
                        self.kd_pos * d_pos_error)
        
        u2_correction = (self.kp_heading * heading_error + 
                        self.ki_heading * self.integral_heading + 
                        self.kd_heading * d_heading_error)
        
        # Add reference feedforward
        u1 = v_ref + u1_correction
        u2 = omega_ref + u2_correction
        
        # Update previous errors
        self.prev_pos_error = pos_error
        self.prev_heading_error = heading_error
        
        # Log
        self.error_history['pos'].append(pos_error)
        self.error_history['heading'].append(heading_error)
        self.error_history['time'].append(t)
        
        return u1, u2
    
    def reset(self):
        """Reset controller state"""
        self.integral_pos = 0.0
        self.integral_heading = 0.0
        self.prev_pos_error = 0.0
        self.prev_heading_error = 0.0
        self.error_history = {'pos': [], 'heading': [], 'time': []}


# Test the controller
if __name__ == "__main__":
    print("Testing Trajectory Tracker...")
    
    # Create controller
    controller = TrajectoryTracker(kx=1.0, ky=1.5, ktheta=2.0)
    
    # Test scenario
    print("\nTest: Robot is off-path, controller should correct")
    
    # Current state: robot at (1, 1) facing 0°
    state = [1.0, 1.0, 0.0]
    
    # Reference: should be at (2, 1.5) facing 45°
    reference = [2.0, 1.5, np.pi/4, 0.5, 0.0]
    
    # Compute errors
    ex, ey, etheta = controller.compute_errors(state, reference)
    print(f"\nErrors:")
    print(f"  e_x (forward):  {ex:.3f} m")
    print(f"  e_y (lateral):  {ey:.3f} m")
    print(f"  e_θ (heading):  {np.degrees(etheta):.1f}°")
    
    # Compute control
    u1, u2 = controller.compute_control(state, reference)
    print(f"\nControl commands:")
    print(f"  u_1 (linear):   {u1:.3f} m/s")
    print(f"  u_2 (angular):  {u2:.3f} rad/s ({np.degrees(u2):.1f}°/s)")
    
    print("\nController test complete!")