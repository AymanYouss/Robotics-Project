"""
World environment for the unicycle robot
Creates a 10x10 meter bounded area with walls
"""

import pybullet as p
import pybullet_data
import numpy as np

class RobotWorld:
    """
    A 10x10 meter world with boundaries for the unicycle robot
    """
    
    def __init__(self, gui=True):
        """
        Initialize the world
        
        Args:
            gui: If True, opens a GUI window. If False, runs headless (no window)
        """
        self.gui = gui
        self.world_size = 10  # 10x10 meters
        
        # Connect to PyBullet
        if self.gui:
            self.physicsClient = p.connect(p.GUI)
            print("PyBullet GUI connected!")
        else:
            self.physicsClient = p.connect(p.DIRECT)
            print("PyBullet running in DIRECT mode (no GUI)")
        
        # Set up the environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Configure the camera view
        self._setup_camera()
        
        # Create the ground and walls
        self._create_ground()
        self._create_walls()
        
        print(f"World created: {self.world_size}x{self.world_size} meters")
    
    def _setup_camera(self):
        """
        Set up a nice camera view looking down at the world
        """
        # Camera parameters
        camera_distance = 15  # How far the camera is
        camera_yaw = 0  # Rotation around z-axis (degrees)
        camera_pitch = -60  # Looking down angle (degrees)
        camera_target = [5, 5, 0]  # Looking at center of 10x10 world
        
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target
        )
        
        print("Camera set to bird's eye view")
    
    def _create_ground(self):
        """
        Create a textured ground plane
        """
        # Load the ground plane
        self.planeId = p.loadURDF("plane.urdf")
        
        # Change ground color to light gray
        p.changeVisualShape(self.planeId, -1, rgbaColor=[0.9, 0.9, 0.9, 1])
        
        print("Ground plane created")
    
    def _create_walls(self):
        """
        Create walls around the 10x10 meter boundary
        The walls are positioned at x=0, x=10, y=0, y=10
        """
        self.walls = []
        
        wall_height = 0.5  # 0.5 meters tall
        wall_thickness = 0.1  # 10 cm thick
        wall_color = [0.3, 0.3, 0.8, 1]  # Blue walls
        
        # Wall positions: [x_pos, y_pos, x_size, y_size]
        wall_configs = [
            # Bottom wall (along y=0)
            [5, 0, 5, wall_thickness],
            # Top wall (along y=10)
            [5, 10, 5, wall_thickness],
            # Left wall (along x=0)
            [0, 5, wall_thickness, 5],
            # Right wall (along x=10)
            [10, 5, wall_thickness, 5]
        ]
        
        for i, config in enumerate(wall_configs):
            x_pos, y_pos, x_size, y_size = config
            
            # Create collision shape (for physics)
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[x_size, y_size, wall_height/2]
            )
            
            # Create visual shape (what we see)
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[x_size, y_size, wall_height/2],
                rgbaColor=wall_color
            )
            
            # Create the wall body
            wall_id = p.createMultiBody(
                baseMass=0,  # Static wall (doesn't move)
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[x_pos, y_pos, wall_height/2]
            )
            
            self.walls.append(wall_id)
        
        print(f"Created {len(self.walls)} boundary walls")
    
    def add_grid_lines(self):
        """
        Add grid lines to visualize the 10x10 meter space
        This helps us see coordinates better
        """
        line_color = [0.7, 0.7, 0.7]  # Gray lines
        
        # Draw vertical lines (parallel to y-axis)
        for x in range(11):  # 0 to 10
            p.addUserDebugLine(
                lineFromXYZ=[x, 0, 0.01],
                lineToXYZ=[x, 10, 0.01],
                lineColorRGB=line_color,
                lineWidth=1
            )
        
        # Draw horizontal lines (parallel to x-axis)
        for y in range(11):  # 0 to 10
            p.addUserDebugLine(
                lineFromXYZ=[0, y, 0.01],
                lineToXYZ=[10, y, 0.01],
                lineColorRGB=line_color,
                lineWidth=1
            )
        
        print("Grid lines added")
    
    def add_obstacle(self, position, size=0.5):
        """
        Add a cylindrical obstacle to the world
        
        Args:
            position: [x, y] position of the obstacle
            size: radius of the obstacle (default 0.5m)
        
        Returns:
            obstacle_id: PyBullet body ID of the obstacle
        """
        x, y = position
        height = 0.5
        
        # Create collision shape
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=size,
            height=height
        )
        
        # Create visual shape (red obstacle)
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=size,
            length=height,
            rgbaColor=[0.8, 0.2, 0.2, 1]
        )
        
        # Create the obstacle
        obstacle_id = p.createMultiBody(
            baseMass=0,  # Static
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x, y, height/2]
        )
        
        print(f"Obstacle added at ({x}, {y})")
        return obstacle_id
    
    def reset(self):
        """
        Reset the simulation
        """
        p.resetSimulation()
        self._create_ground()
        self._create_walls()
        print("World reset")
    
    def close(self):
        """
        Close the PyBullet connection
        """
        p.disconnect()
        print("World closed")


# Test the world if this file is run directly
if __name__ == "__main__":
    import time
    
    print("Testing RobotWorld class...\n")
    
    # Create the world
    world = RobotWorld(gui=True)
    
    # Add grid lines
    world.add_grid_lines()
    
    # Add some obstacles for testing
    world.add_obstacle([3, 3], size=0.3)
    world.add_obstacle([7, 7], size=0.4)
    world.add_obstacle([5, 8], size=0.5)
    
    print("\nWorld created successfully!")
    print("You should see:")
    print("  - A 10x10 meter area with blue walls")
    print("  - Grid lines every 1 meter")
    print("  - Three red cylindrical obstacles")
    print("\nSimulation will run for 10 seconds...")
    
    # Keep the window open for 10 seconds
    for i in range(10):
        p.stepSimulation()
        time.sleep(1)
        print(f"Time: {i+1}s")
    
    # Close
    world.close()
    print("\nTest complete!")