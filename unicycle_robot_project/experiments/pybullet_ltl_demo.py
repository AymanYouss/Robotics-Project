"""
Complete PyBullet LTL Demonstration

Change ONLY the ltl_input string to see different robot behaviors!

WAYPOINTS AVAILABLE:
- start: (1, 1) - Starting position
- waypoint_A: (3, 8) - Top-left checkpoint
- waypoint_B: (8, 8) - Top-right checkpoint  
- waypoint_C: (8, 2) - Bottom-right checkpoint
- waypoint_D: (5, 5) - Center checkpoint
- goal: (9, 9) - Final goal

OBSTACLES:
- obstacle_center: (5, 5) - Center obstacle
- obstacle_1: (3, 5) - Left obstacle
- obstacle_2: (7, 5) - Right obstacle
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pybullet as p
import pybullet_data
import numpy as np
import time

from controllers.symbolic_controller import GridAbstraction, TransitionSystem
from controllers.unified_ltl_controller import UnifiedLTLController
from models.unicycle import UnicycleRobot


ltl_input = "F(goal) "

# Try these:
# "F(goal)"                                    - Just reach goal
# "F(goal) & G(!obstacle_center)"              - Reach goal, avoid center obstacle
# "F(goal) & G(!obstacle_1) & G(!obstacle_2)"  - Reach goal, avoid both side obstacles
# "F(waypoint_A & F(goal))"                    - Visit A then goal
# "F(waypoint_B & F(goal))"                    - Visit B then goal
# "F(waypoint_A & F(waypoint_B & F(goal)))"    - Visit A, then B, then goal
# "G(!obstacle_center)"                        - Just avoid center obstacle (stay safe)
# "F(waypoint_A) | F(waypoint_B)"              - Reach either A or B
# "F(waypoint_D)"                              - Reach center waypoint (near obstacle!)
# "F(waypoint_C & F(waypoint_B))"              - Visit C then B

# ============================================================================

def create_pybullet_environment():
    """Create PyBullet world with waypoints and obstacles"""
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
    
    # Define waypoint and obstacle positions
    locations = {
        'start': (0, 0),
        'waypoint_A': (3, 8),
        'waypoint_B': (8, 8),
        'waypoint_C': (8, 2),
        'waypoint_D': (5, 5),
        'goal': (9, 9),
        'obstacle_center': (5, 5),
        'obstacle_1': (3, 5),
        'obstacle_2': (7, 5)
    }
    
    # Visualize waypoints (green spheres)
    waypoint_names = ['start', 'waypoint_A', 'waypoint_B', 'waypoint_C', 'waypoint_D', 'goal']
    for name in waypoint_names:
        pos = locations[name]
        marker_shape = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=0.3,
            rgbaColor=[0.2, 0.8, 0.2, 0.8]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=marker_shape,
            basePosition=[pos[0], pos[1], 0.15]
        )
        # Label
        label = name.replace('waypoint_', '')
        p.addUserDebugText(
            label.upper(),
            [pos[0], pos[1], 0.6],
            textColorRGB=[0, 0, 0],
            textSize=1.5
        )
    
    # Visualize obstacles (red cylinders)
    obstacle_configs = [
        ('obstacle_center', 0.8),
        ('obstacle_1', 0.5),
        ('obstacle_2', 0.5)
    ]
    
    for name, radius in obstacle_configs:
        pos = locations[name]
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=0.5
        )
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=0.5,
            rgbaColor=[0.8, 0.2, 0.2, 0.7]
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[pos[0], pos[1], 0.25]
        )
    
    print("\n" + "="*70)
    print("PyBullet Environment Created")
    print("="*70)
    print("\nWaypoints:")
    for name in waypoint_names:
        pos = locations[name]
        print(f"  {name:15s}: ({pos[0]}, {pos[1]})")
    print("\nObstacles:")
    for name, radius in obstacle_configs:
        pos = locations[name]
        print(f"  {name:15s}: ({pos[0]}, {pos[1]}) radius={radius}")
    print("="*70)
    
    return locations

def create_symbolic_abstraction(locations):
    """Create grid abstraction and transition system"""
    print("\nCreating symbolic abstraction...")
    
    # Create grid
    grid = GridAbstraction(world_size=10.0, grid_size=20, num_headings=8)
    
    # Mark obstacles in grid
    grid.mark_obstacle_region(center=locations['obstacle_center'], radius=0.8)
    grid.mark_obstacle_region(center=locations['obstacle_1'], radius=0.5)
    grid.mark_obstacle_region(center=locations['obstacle_2'], radius=0.5)
    
    # Build transition system
    ts = TransitionSystem(grid)
    
    # Add all regions
    ts.add_region_from_continuous('start', center=locations['start'], radius=0.5)
    ts.add_region_from_continuous('waypoint_A', center=locations['waypoint_A'], radius=0.6)
    ts.add_region_from_continuous('waypoint_B', center=locations['waypoint_B'], radius=0.6)
    ts.add_region_from_continuous('waypoint_C', center=locations['waypoint_C'], radius=0.6)
    ts.add_region_from_continuous('waypoint_D', center=locations['waypoint_D'], radius=0.6)
    ts.add_region_from_continuous('goal', center=locations['goal'], radius=0.6)
    ts.add_region_from_continuous('obstacle_center', center=locations['obstacle_center'], radius=0.8)
    ts.add_region_from_continuous('obstacle_1', center=locations['obstacle_1'], radius=0.5)
    ts.add_region_from_continuous('obstacle_2', center=locations['obstacle_2'], radius=0.5)
    
    return grid, ts

def synthesize_controller(ts, ltl_formula):
    """Synthesize controller from LTL specification"""
    print("\n" + "="*70)
    print("LTL SPECIFICATION")
    print("="*70)
    print(f"Formula: {ltl_formula}")
    print("="*70)
    
    controller = UnifiedLTLController(ts)
    controller.set_specification(ltl_formula)
    controller.synthesize()
    
    return controller

def discrete_to_continuous_path(discrete_path, grid):
    """Convert discrete cell path to continuous waypoints"""
    continuous_path = []
    for state in discrete_path:
        i, j, h_idx = state
        x, y = grid.cell_to_continuous(i, j)
        theta = grid.discrete_heading_to_continuous(h_idx)
        continuous_path.append((x, y, theta))
    return continuous_path

def execute_in_pybullet(continuous_path, start_pos):
    """Execute path in PyBullet with unicycle robot"""
    print("\n" + "="*70)
    print("EXECUTING IN PYBULLET")
    print("="*70)
    
    # Create robot
    robot = UnicycleRobot(start_pos=start_pos, tau=0.1, add_disturbance=False)
    
    # Follow waypoints
    for i, (target_x, target_y, target_theta) in enumerate(continuous_path):
        print(f"\nWaypoint {i+1}/{len(continuous_path)}: ({target_x:.1f}, {target_y:.1f})")
        
        # Move towards waypoint
        max_steps = 50
        for step in range(max_steps):
            # Current state
            state = robot.get_state()
            x, y, theta = state
            
            # Distance to target
            dist = np.sqrt((target_x - x)**2 + (target_y - y)**2)
            
            if dist < 0.3:  # Close enough
                print(f"  Reached waypoint (distance: {dist:.2f}m)")
                break
            
            # Simple control: point towards target and move
            target_angle = np.arctan2(target_y - y, target_x - x)
            angle_error = target_angle - theta
            angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
            
            # Control inputs
            u1 = 0.5  # Forward velocity
            u2 = 2.0 * angle_error  # Proportional heading control
            
            # Apply control
            robot.step(u1, u2)
            p.stepSimulation()
            time.sleep(0.02)
    
    # Draw trajectory
    robot.draw_trajectory()
    
    print("\n" + "="*70)
    print("EXECUTION COMPLETE!")
    print("="*70)
    print(f"Robot visited {len(continuous_path)} waypoints")
    print(f"Final position: ({robot.x1:.2f}, {robot.x2:.2f})")
    
    return robot

def main():
    """Main simulation loop"""
    print("\n" + "="*70)
    print("PYBULLET LTL DEMONSTRATION")
    print("="*70)
    print(f"\nLTL Formula: {ltl_input}")
    print("="*70)
    
    # Step 1: Create PyBullet environment
    locations = create_pybullet_environment()
    
    # Step 2: Create symbolic abstraction
    grid, ts = create_symbolic_abstraction(locations)
    
    # Step 3: Synthesize controller
    controller = synthesize_controller(ts, ltl_input)
    
    # Step 4: Get discrete path
    start_states = ts.get_states_in_region('start')
    if not start_states:
        print("ERROR: No start states found!")
        return
    
    discrete_path, actions = controller.execute_strategy(start_states[0], max_steps=150)
    
    if len(discrete_path) < 2:
        print("\nWARNING: No valid path found for this specification!")
        print("The specification may be unsatisfiable or the start state is not in the winning region.")
        input("\nPress Enter to exit...")
        p.disconnect()
        return
    
    print(f"\nDiscrete path computed: {len(discrete_path)} states")
    
    # Step 5: Convert to continuous path
    continuous_path = discrete_to_continuous_path(discrete_path, grid)
    
    # Step 6: Execute in PyBullet
    start_pos = [locations['start'][0], locations['start'][1], 0.0]
    robot = execute_in_pybullet(continuous_path, start_pos)
    
    # Step 7: Keep simulation open
    print("\nSimulation complete!")
    print("The robot has executed the LTL specification.")
    print("\nClose the PyBullet window or press Ctrl+C to exit.")
    
    try:
        while True:
            p.stepSimulation()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")
    
    p.disconnect()

if __name__ == "__main__":
    main()