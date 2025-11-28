"""
Test open-loop control of the unicycle robot
This demonstrates manual control without feedback
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pybullet as p
import pybullet_data
import numpy as np
import time
from models.unicycle import UnicycleRobot

def create_simple_environment():
    """Create a simple test environment"""
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
    
    # Grid lines
    for x in range(11):
        p.addUserDebugLine([x, 0, 0.01], [x, 10, 0.01], [0.7, 0.7, 0.7], 1)
    for y in range(11):
        p.addUserDebugLine([0, y, 0.01], [10, y, 0.01], [0.7, 0.7, 0.7], 1)
    
    print("Environment created")

def test_straight_line():
    """Test 1: Drive in a straight line"""
    print("\n" + "="*60)
    print("TEST 1: STRAIGHT LINE")
    print("="*60)
    print("Command: u1=0.5 (constant forward), u2=0 (no turn)")
    print("Expected: Robot should move in a straight line")
    print("Reality: Small deviations due to disturbances\n")
    
    robot = UnicycleRobot(start_pos=[2.0, 2.0, 0.0], tau=0.1, add_disturbance=True)
    
    for i in range(80):  # 8 seconds
        state = robot.step(u1=0.5, u2=0.0)
        p.stepSimulation()
        time.sleep(0.05)
        
        if i % 20 == 0:
            print(f"t={i*0.1:.1f}s: pos=({state[0]:.2f}, {state[1]:.2f}), angle={np.degrees(state[2]):.1f}°")
    
    robot.draw_trajectory()
    print(f"\nFinal position: ({robot.x1:.2f}, {robot.x2:.2f})")
    print("Notice: Path is not perfectly straight due to disturbances!")
    return robot

def test_circle():
    """Test 2: Drive in a circle"""
    print("\n" + "="*60)
    print("TEST 2: CIRCULAR PATH")
    print("="*60)
    print("Command: u1=0.5 (constant forward), u2=0.3 (constant left turn)")
    print("Expected: Robot should move in a circular arc\n")
    
    robot = UnicycleRobot(start_pos=[5.0, 3.0, 0.0], tau=0.1, add_disturbance=True)
    
    for i in range(100):  # 10 seconds
        state = robot.step(u1=0.5, u2=0.3)
        p.stepSimulation()
        time.sleep(0.05)
        
        if i % 25 == 0:
            print(f"t={i*0.1:.1f}s: pos=({state[0]:.2f}, {state[1]:.2f}), angle={np.degrees(state[2]):.1f}°")
    
    robot.draw_trajectory()
    print(f"\nFinal position: ({robot.x1:.2f}, {robot.x2:.2f})")
    return robot

def test_square():
    """Test 3: Try to drive in a square (open-loop)"""
    print("\n" + "="*60)
    print("TEST 3: SQUARE PATH (OPEN-LOOP ATTEMPT)")
    print("="*60)
    print("Sequence: Forward → Turn 90° → Forward → Turn 90° → ...")
    print("Expected: Approximately square shape")
    print("Reality: Errors accumulate, shape becomes distorted!\n")
    
    robot = UnicycleRobot(start_pos=[3.0, 3.0, 0.0], tau=0.1, add_disturbance=True)
    
    # Define the sequence
    side_length = 3.0  # 3 meters per side
    forward_time = side_length / 0.5  # time = distance / speed
    turn_time = (np.pi/2) / 0.5  # time = angle / angular_velocity
    
    for side in range(4):
        print(f"\nSide {side+1}: Moving forward...")
        # Move forward
        for i in range(int(forward_time / 0.1)):
            robot.step(u1=0.5, u2=0.0)
            p.stepSimulation()
            time.sleep(0.05)
        
        print(f"Side {side+1}: Turning left 90°...")
        # Turn left 90 degrees
        for i in range(int(turn_time / 0.1)):
            robot.step(u1=0.25, u2=0.5)  # Slower forward while turning
            p.stepSimulation()
            time.sleep(0.05)
    
    robot.draw_trajectory()
    print(f"\nFinal position: ({robot.x1:.2f}, {robot.x2:.2f})")
    print("Notice: The 'square' is not perfect due to accumulated errors!")
    return robot

def test_with_without_disturbances():
    """Test 4: Compare with and without disturbances"""
    print("\n" + "="*60)
    print("TEST 4: DISTURBANCE COMPARISON")
    print("="*60)
    
    # Reset simulation
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")
    
    print("\n4a) WITHOUT disturbances:")
    robot1 = UnicycleRobot(start_pos=[2.0, 5.0, 0.0], tau=0.1, add_disturbance=False)
    
    for i in range(60):
        robot1.step(u1=0.5, u2=0.2)
        p.stepSimulation()
        time.sleep(0.05)
    
    robot1.draw_trajectory()
    print(f"Final position: ({robot1.x1:.2f}, {robot1.x2:.2f})")
    
    print("\n4b) WITH disturbances:")
    robot2 = UnicycleRobot(start_pos=[2.0, 3.0, 0.0], tau=0.1, add_disturbance=True)
    
    for i in range(60):
        robot2.step(u1=0.5, u2=0.2)
        p.stepSimulation()
        time.sleep(0.05)
    
    robot2.draw_trajectory()
    print(f"Final position: ({robot2.x1:.2f}, {robot2.x2:.2f})")
    
    print("\nCompare the two trajectories:")
    print("- Upper path (no disturbances): Smooth and predictable")
    print("- Lower path (with disturbances): Jittery and less predictable")

def main():
    """Run all open-loop tests"""
    print("="*60)
    print("OPEN-LOOP CONTROL TESTS")
    print("="*60)
    print("\nOpen-loop control means:")
    print("✗ No feedback from sensors")
    print("✗ No error correction")
    print("✗ Commands are sent blindly")
    print("✗ Disturbances cause cumulative errors")
    print("\nThis is why we need CLOSED-LOOP control!\n")
    
    input("Press Enter to start tests...")
    
    # Create environment
    create_simple_environment()
    
    # Test 1
    input("\nPress Enter for Test 1 (Straight Line)...")
    test_straight_line()
    
    # Test 2
    input("\nPress Enter for Test 2 (Circle)...")
    p.removeAllUserDebugItems()  # Clear previous trajectory
    test_circle()
    
    # Test 3
    input("\nPress Enter for Test 3 (Square)...")
    p.removeAllUserDebugItems()
    test_square()
    
    # Test 4
    input("\nPress Enter for Test 4 (Disturbance Comparison)...")
    test_with_without_disturbances()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE!")
    print("="*60)
    print("\nKey Observations:")
    print("1. Open-loop control cannot compensate for disturbances")
    print("2. Errors accumulate over time")
    print("3. Even simple shapes become distorted")
    print("4. We NEED feedback control for accurate tracking!")
    print("\nNext: We'll implement CLOSED-LOOP controllers that use feedback")
    
    input("\nPress Enter to close...")
    p.disconnect()

if __name__ == "__main__":
    main()