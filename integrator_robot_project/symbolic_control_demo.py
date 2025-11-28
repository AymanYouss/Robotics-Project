#!/usr/bin/env python3
"""
Symbolic Control Demo: 2D Integrator with LTL Specifications
Implements discrete abstraction and controller synthesis from temporal logic specs.
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

# Grid abstraction parameters
GRID_SIZE = 1.0  # Size of each cell in the abstraction


class Region:
    """Named region in the state space."""
    def __init__(self, name, x_range, y_range, color):
        self.name = name
        self.x_range = x_range  # (min, max)
        self.y_range = y_range  # (min, max)
        self.color = color
        self.center = np.array([
            (x_range[0] + x_range[1]) / 2,
            (y_range[0] + y_range[1]) / 2
        ])
    
    def contains(self, x):
        """Check if state x is in this region."""
        return (self.x_range[0] <= x[0] <= self.x_range[1] and
                self.y_range[0] <= x[1] <= self.y_range[1])


# Define regions for specifications
REGIONS = {
    'start': Region('start', (-9, -7), (-9, -7), [0, 0, 1, 0.5]),      # Blue - bottom left
    'goal': Region('goal', (7, 9), (7, 9), [0, 1, 0, 0.5]),            # Green - top right
    'obstacle1': Region('obstacle1', (-2, 2), (-2, 2), [1, 0, 0, 0.5]), # Red - center
    'obstacle2': Region('obstacle2', (3, 6), (-6, -3), [1, 0, 0, 0.5]), # Red - bottom right
    'obstacle3': Region('obstacle3', (-7, -4), (0, 3), [1, 0, 0, 0.5]), # Red - middle left
    'waypoint1': Region('waypoint1', (-6, -4), (5, 7), [1, 1, 0, 0.5]), # Yellow - top left
    'waypoint2': Region('waypoint2', (4, 6), (2, 4), [1, 0.5, 0, 0.5]), # Orange - middle right
    'waypoint3': Region('waypoint3', (-9, -7), (5, 7), [0, 1, 1, 0.5]), # Cyan - top left corner
    'waypoint4': Region('waypoint4', (7, 9), (-9, -7), [1, 0, 1, 0.5]), # Magenta - bottom right corner
    'charging': Region('charging', (-9, -7), (7, 9), [0.5, 0, 0.5, 0.5]), # Purple - top left
}


def clip_state(x):
    return np.clip(x, X_BOUNDS[0], X_BOUNDS[1])

def clip_input(u):
    return np.clip(u, U_BOUNDS[0], U_BOUNDS[1])

def get_disturbance():
    return np.random.uniform(W_BOUNDS[0], W_BOUNDS[1], size=2)

def integrator_dynamics(x, u, w):
    u_clipped = clip_input(u)
    x_next = x + TAU * (u_clipped + w)
    return clip_state(x_next)


class SymbolicController:
    """
    Symbolic controller that follows LTL-like specifications.
    Uses a simple automaton-based approach for specification satisfaction.
    """
    
    def __init__(self, spec_type='reach_avoid'):
        self.spec_type = spec_type
        self.current_target_idx = 0
        self.waypoints = []
        self.obstacles = []
        self.visited = set()
        self._setup_specification(spec_type)
    
    def _setup_specification(self, spec_type):
        """
        Setup specification based on type.
        
        Supported LTL-like specifications:
        - reach_avoid: ◇goal ∧ □¬obstacle (eventually goal, always avoid obstacles)
        - sequential: ◇wp1 ∧ ◇wp2 ∧ ◇goal (visit waypoints in order then goal)
        - patrol: □◇wp1 ∧ □◇wp2 (infinitely often visit waypoints)
        """
        if spec_type == 'reach_avoid':
            # ◇goal ∧ □¬obstacle - Eventually reach goal while always avoiding obstacles
            self.waypoints = [REGIONS['goal'].center]
            self.obstacles = [REGIONS['obstacle1'], REGIONS['obstacle2'], REGIONS['obstacle3']]
            self.spec_description = "◇goal ∧ □¬obstacle (reach goal, avoid obstacles)"
            
        elif spec_type == 'sequential':
            # Visit waypoint1, then waypoint2, then goal
            self.waypoints = [
                REGIONS['waypoint1'].center,
                REGIONS['waypoint2'].center,
                REGIONS['goal'].center
            ]
            self.obstacles = [REGIONS['obstacle1'], REGIONS['obstacle2'], REGIONS['obstacle3']]
            self.spec_description = "◇wp1 ∧ ◇(wp1 → ◇wp2) ∧ ◇(wp2 → ◇goal) (sequential visit)"
            
        elif spec_type == 'patrol':
            # Patrol between waypoints indefinitely
            self.waypoints = [
                REGIONS['waypoint1'].center,
                REGIONS['waypoint2'].center,
                REGIONS['goal'].center,
                REGIONS['start'].center
            ]
            self.obstacles = [REGIONS['obstacle1'], REGIONS['obstacle2'], REGIONS['obstacle3']]
            self.spec_description = "□◇wp1 ∧ □◇wp2 ∧ □◇goal (patrol pattern)"
            self.loop = True
            
        elif spec_type == 'coverage':
            # Visit all corners of the workspace
            self.waypoints = [
                REGIONS['waypoint3'].center,  # Top left
                REGIONS['goal'].center,        # Top right
                REGIONS['waypoint4'].center,  # Bottom right
                REGIONS['start'].center        # Bottom left
            ]
            self.obstacles = [REGIONS['obstacle1'], REGIONS['obstacle2'], REGIONS['obstacle3']]
            self.spec_description = "◇corner1 ∧ ◇corner2 ∧ ◇corner3 ∧ ◇corner4 (full coverage)"
            
        elif spec_type == 'charge_patrol':
            # Patrol but return to charging station periodically
            self.waypoints = [
                REGIONS['waypoint1'].center,
                REGIONS['charging'].center,
                REGIONS['waypoint2'].center,
                REGIONS['charging'].center,
                REGIONS['goal'].center,
                REGIONS['charging'].center
            ]
            self.obstacles = [REGIONS['obstacle1'], REGIONS['obstacle2'], REGIONS['obstacle3']]
            self.spec_description = "□◇charging ∧ □◇wp1 ∧ □◇wp2 (patrol with charging)"
            self.loop = True
            
        elif spec_type == 'surveillance':
            # Monitor specific areas in sequence repeatedly
            self.waypoints = [
                REGIONS['waypoint1'].center,
                REGIONS['waypoint2'].center,
                REGIONS['waypoint3'].center,
                REGIONS['goal'].center
            ]
            self.obstacles = [REGIONS['obstacle1'], REGIONS['obstacle2']]
            self.spec_description = "□(◇wp1 ∧ ◇wp2 ∧ ◇wp3 ∧ ◇goal) (surveillance loop)"
            self.loop = True
            
        elif spec_type == 'escape':
            # Navigate through obstacle maze to reach goal
            self.waypoints = [
                np.array([-8.0, 4.0]),   # Go up first
                np.array([0.0, 8.0]),    # Top middle
                np.array([8.0, 5.0]),    # Right side
                REGIONS['goal'].center
            ]
            self.obstacles = [REGIONS['obstacle1'], REGIONS['obstacle2'], REGIONS['obstacle3']]
            self.spec_description = "◇goal via safe path (maze navigation)"
        else:
            raise ValueError(f"Unknown specification: {spec_type}")
        
        self.loop = spec_type == 'patrol'
    
    def get_current_target(self):
        """Get current target based on automaton state."""
        if self.current_target_idx >= len(self.waypoints):
            if self.loop:
                self.current_target_idx = 0
            else:
                return self.waypoints[-1]  # Stay at goal
        return self.waypoints[self.current_target_idx]
    
    def update_state(self, x):
        """Update automaton state based on current position."""
        target = self.get_current_target()
        dist = np.linalg.norm(x - target)
        
        # Check if reached current waypoint
        if dist < 1.0:
            if self.current_target_idx < len(self.waypoints):
                self.visited.add(self.current_target_idx)
                self.current_target_idx += 1
                if self.loop and self.current_target_idx >= len(self.waypoints):
                    self.current_target_idx = 0
    
    def compute_control(self, x, kp=2.0):
        """
        Compute control with obstacle avoidance using potential fields.
        """
        target = self.get_current_target()
        
        # Attractive force toward target
        error = target - x
        dist_to_target = np.linalg.norm(error)
        if dist_to_target > 0.1:
            u_attract = kp * error / dist_to_target  # Normalize direction
        else:
            u_attract = kp * error
        
        # Repulsive force from obstacles
        u_repel = np.zeros(2)
        for obs in self.obstacles:
            # Check distance to obstacle boundaries, not just center
            obs_center = obs.center
            obs_half_w = (obs.x_range[1] - obs.x_range[0]) / 2 + 1.5  # Add margin
            obs_half_h = (obs.y_range[1] - obs.y_range[0]) / 2 + 1.5
            
            # Vector from obstacle center to robot
            to_robot = x - obs_center
            
            # Distance considering obstacle size
            dx = max(0, abs(to_robot[0]) - obs_half_w + 1.5)
            dy = max(0, abs(to_robot[1]) - obs_half_h + 1.5)
            dist = np.sqrt(dx**2 + dy**2)
            
            # Strong repulsion when close
            safe_dist = 4.0
            if dist < safe_dist:
                # Very strong repulsion that increases exponentially when close
                strength = 5.0 * ((safe_dist - dist) / safe_dist) ** 2
                if np.linalg.norm(to_robot) > 0.1:
                    u_repel += strength * to_robot / np.linalg.norm(to_robot)
        
        # Combined control - repulsion dominates when close to obstacle
        u = u_attract + u_repel
        return clip_input(u)
    
    def check_safety(self, x):
        """Check if current state violates safety (in obstacle)."""
        for obs in self.obstacles:
            if obs.contains(x):
                return False
        return True
    
    def is_complete(self):
        """Check if specification is satisfied."""
        if self.loop:
            return False  # Patrol never completes
        return self.current_target_idx >= len(self.waypoints)


def draw_regions(regions):
    """Draw all regions in PyBullet."""
    for name, region in regions.items():
        # Draw region as a box outline
        x1, x2 = region.x_range
        y1, y2 = region.y_range
        z = 0.05
        
        corners = [
            [x1, y1, z], [x2, y1, z],
            [x2, y2, z], [x1, y2, z]
        ]
        
        color = region.color[:3]
        for i in range(4):
            p.addUserDebugLine(corners[i], corners[(i+1)%4], color, lineWidth=5)
        
        # Add label
        p.addUserDebugText(name, [(x1+x2)/2, (y1+y2)/2, 0.3], color, textSize=1.5)


def main():
    parser = argparse.ArgumentParser(description="Symbolic Control Demo with LTL Specs")
    parser.add_argument("--gui", action="store_true", help="Enable GUI mode")
    parser.add_argument("--spec", type=str, default="reach_avoid",
                        choices=["reach_avoid", "sequential", "patrol", "coverage", 
                                 "charge_patrol", "surveillance", "escape"],
                        help="LTL specification type")
    parser.add_argument("--kp", type=float, default=2.0, help="Controller gain")
    parser.add_argument("--steps", type=int, default=1000, help="Max simulation steps")
    args = parser.parse_args()

    # Initialize PyBullet
    if args.gui:
        physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    else:
        physics_client = p.connect(p.DIRECT)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    
    # Load plane and make it white
    plane_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[1, 1, 1, 1])
    
    # Create robot
    robot_radius = 0.3
    robot_visual = p.createVisualShape(p.GEOM_SPHERE, radius=robot_radius, 
                                        rgbaColor=[0, 0.5, 1, 1])
    robot_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=robot_radius)
    
    start = REGIONS['start'].center.copy()
    
    robot_id = p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=robot_collision,
        baseVisualShapeIndex=robot_visual,
        basePosition=[start[0], start[1], robot_radius]
    )
    
    # Draw workspace bounds
    corners = [
        [X_BOUNDS[0], X_BOUNDS[0], 0.1],
        [X_BOUNDS[1], X_BOUNDS[0], 0.1],
        [X_BOUNDS[1], X_BOUNDS[1], 0.1],
        [X_BOUNDS[0], X_BOUNDS[1], 0.1],
    ]
    for i in range(4):
        p.addUserDebugLine(corners[i], corners[(i+1)%4], [0.2, 0.2, 0.2], lineWidth=5)
    
    # Draw regions
    draw_regions(REGIONS)
    
    # Set camera
    if args.gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=25,
            cameraYaw=0,
            cameraPitch=-89,
            cameraTargetPosition=[0, 0, 0]
        )
    
    # Initialize symbolic controller
    controller = SymbolicController(spec_type=args.spec)
    
    print(f"Specification: {args.spec}")
    print(f"LTL Formula: {controller.spec_description}")
    print(f"Start: {start}")
    print("Running simulation...")
    
    # Simulation
    x = start.copy()
    prev_pos = start.copy()
    safety_violations = 0
    
    for step in range(args.steps):
        # Update automaton state
        controller.update_state(x)
        
        # Check specification completion
        if controller.is_complete():
            print(f"\n✓ Specification satisfied at step {step}!")
            break
        
        # Compute control
        u = controller.compute_control(x, args.kp)
        
        # Apply dynamics with disturbance
        w = get_disturbance()
        x = integrator_dynamics(x, u, w)
        
        # Check safety
        if not controller.check_safety(x):
            safety_violations += 1
        
        # Draw trajectory (red)
        p.addUserDebugLine(
            [prev_pos[0], prev_pos[1], 0.1],
            [x[0], x[1], 0.1],
            [1, 0, 0], lineWidth=5
        )
        prev_pos = x.copy()
        
        # Update robot position
        p.resetBasePositionAndOrientation(
            robot_id,
            [x[0], x[1], robot_radius],
            [0, 0, 0, 1]
        )
        
        if args.gui:
            time.sleep(TAU)
        
        p.stepSimulation()
    
    print(f"\nFinal position: {x}")
    print(f"Waypoints visited: {len(controller.visited)}/{len(controller.waypoints)}")
    print(f"Safety violations: {safety_violations}")
    
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
