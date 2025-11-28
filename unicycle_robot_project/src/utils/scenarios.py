"""
Real-world scenarios for the unicycle robot
Defines practical control objectives and reference trajectories
"""

import numpy as np
from utils.trajectories import WaypointPath

class CampusDeliveryScenario:
    """
    SCENARIO: Campus Package Delivery Robot
    
    Setting: University campus delivery robot
    
    Map Layout (10x10 meters represents a section of campus):
        - (1, 1): Warehouse (starting point)
        - (3, 2): Building A (Computer Science)
        - (7, 3): Building B (Library) 
        - (8, 7): Building C (Student Center)
        - (4, 8): Building D (Engineering)
        - (1, 6): Charging Station
    
    Obstacles:
        - (5, 5): Fountain (circular obstacle)
        - Walls at boundaries
    
    CONTROL OBJECTIVE:
    Deliver packages following the route:
    Warehouse → Building A → Library → Student Center → Engineering → Charging Station
    
    CONSTRAINTS:
    - Stay within campus boundaries [0, 10] × [0, 10]
    - Maintain safe distance from fountain
    - Linear velocity: 0.25-1.0 m/s (walking speed)
    - Angular velocity: -1 to 1 rad/s (smooth turns)
    - Must handle disturbances (uneven ground, wind)
    
    SUCCESS CRITERIA:
    - Visit all buildings in order
    - Stay within 0.3m of reference path
    - Arrive within 0.5m of each building
    - Complete route without collisions
    """
    
    def __init__(self):
        self.name = "Campus Package Delivery"
        
        # Define locations
        self.locations = {
            'warehouse': [1.0, 1.0],
            'building_a': [3.0, 2.0],
            'library': [7.0, 3.0],
            'student_center': [8.0, 7.0],
            'engineering': [4.0, 8.0],
            'charging_station': [1.0, 6.0]
        }
        
        # Define obstacles
        self.obstacles = [
            {'name': 'fountain', 'position': [5.0, 5.0], 'radius': 0.6}
        ]
        
        # Delivery route (waypoints)
        self.route = [
            self.locations['warehouse'],
            self.locations['building_a'],
            self.locations['library'],
            self.locations['student_center'],
            self.locations['engineering'],
            self.locations['charging_station']
        ]
        
        # Robot parameters
        self.max_velocity = 0.7  # m/s (comfortable walking speed)
        self.safe_distance = 0.3  # m (tracking tolerance)
        self.arrival_tolerance = 0.5  # m (reached waypoint threshold)
        self.waypoint_tolerance = 0.5  # m (for evaluation)
        
        print("="*70)
        print("SCENARIO: Campus Package Delivery Robot")
        print("="*70)
        print("\nMission: Deliver packages to 4 campus buildings")
        print("\nRoute:")
        for i, (name, loc) in enumerate(zip(
            ['Warehouse (START)', 'Building A (CS)', 'Library', 
             'Student Center', 'Engineering', 'Charging Station (END)'],
            self.route
        )):
            print(f"  {i+1}. {name:25s} at ({loc[0]:.1f}, {loc[1]:.1f})")
        
        print(f"\nConstraints:")
        print(f"  - Maximum velocity: {self.max_velocity} m/s")
        print(f"  - Path tracking tolerance: ±{self.safe_distance} m")
        print(f"  - Arrival threshold: {self.arrival_tolerance} m")
        print(f"  - Obstacles to avoid: {len(self.obstacles)}")
        print("="*70)
    
    def get_reference_trajectory(self):
        """
        Get the reference trajectory for this scenario
        
        Returns:
            WaypointPath object
        """
        return WaypointPath(self.route, velocity=self.max_velocity)
    
    def get_obstacles_for_world(self):
        """
        Get obstacle list formatted for the world environment
        
        Returns:
            List of (position, size) tuples
        """
        return [(obs['position'], obs['radius']) for obs in self.obstacles]
    
    def check_waypoint_reached(self, robot_pos, waypoint_idx):
        """
        Check if robot has reached a waypoint
        
        Args:
            robot_pos: [x, y] current robot position
            waypoint_idx: Index of waypoint to check
        
        Returns:
            True if within arrival tolerance
        """
        if waypoint_idx >= len(self.route):
            return False
        
        waypoint = self.route[waypoint_idx]
        distance = np.linalg.norm(np.array(robot_pos) - np.array(waypoint))
        return distance < self.arrival_tolerance
    
    def check_collision(self, robot_pos):
        """
        Check if robot collides with obstacles
        
        Args:
            robot_pos: [x, y] current robot position
        
        Returns:
            True if collision detected
        """
        for obs in self.obstacles:
            distance = np.linalg.norm(
                np.array(robot_pos) - np.array(obs['position'])
            )
            if distance < obs['radius']:
                return True
        return False
    
    def evaluate_performance(self, robot_trajectory, reference_trajectory, time_taken):
        """
        Evaluate how well the robot completed the mission
        
        Args:
            robot_trajectory: List of [x, y] actual positions
            reference_trajectory: List of [x, y] reference positions
            time_taken: Total mission time in seconds
        
        Returns:
            Dictionary of performance metrics
        """
        robot_traj = np.array(robot_trajectory)
        ref_traj = np.array(reference_trajectory)
        
        # Calculate tracking error
        min_len = min(len(robot_traj), len(ref_traj))
        tracking_errors = np.linalg.norm(
            robot_traj[:min_len] - ref_traj[:min_len], 
            axis=1
        )
        
        avg_error = np.mean(tracking_errors)
        max_error = np.max(tracking_errors)
        
        # Check if all waypoints were reached
        waypoints_reached = 0
        for i, waypoint in enumerate(self.route):
            for pos in robot_trajectory:
                if np.linalg.norm(np.array(pos) - np.array(waypoint)) < self.arrival_tolerance:
                    waypoints_reached += 1
                    break
        
        # Calculate path length
        path_length = 0
        for i in range(len(robot_trajectory) - 1):
            path_length += np.linalg.norm(
                np.array(robot_trajectory[i+1]) - np.array(robot_trajectory[i])
            )
        
        # Success criteria
        mission_success = (
            waypoints_reached == len(self.route) and
            avg_error < self.safe_distance and
            max_error < self.safe_distance * 2
        )
        
        results = {
            'mission_success': mission_success,
            'waypoints_reached': f"{waypoints_reached}/{len(self.route)}",
            'avg_tracking_error': avg_error,
            'max_tracking_error': max_error,
            'path_length': path_length,
            'time_taken': time_taken,
            'avg_velocity': path_length / time_taken if time_taken > 0 else 0
        }
        
        return results
    
    def print_performance_report(self, results):
        """Print a formatted performance report"""
        print("\n" + "="*70)
        print("MISSION PERFORMANCE REPORT")
        print("="*70)
        print(f"Mission Status: {'✓ SUCCESS' if results['mission_success'] else '✗ FAILED'}")
        print(f"\nWaypoints Reached: {results['waypoints_reached']}")
        print(f"Average Tracking Error: {results['avg_tracking_error']:.3f} m")
        print(f"Maximum Tracking Error: {results['max_tracking_error']:.3f} m")
        print(f"Path Length: {results['path_length']:.2f} m")
        print(f"Time Taken: {results['time_taken']:.1f} s")
        print(f"Average Velocity: {results['avg_velocity']:.2f} m/s")
        print("="*70)

class WarehousePatrolScenario:
    """
    SCENARIO: Warehouse Security Patrol Robot
    
    Setting: Automated security patrol in a warehouse
    
    Map Layout (10x10 meters = warehouse floor section):
        - Aisles between storage racks
        - Critical checkpoints to monitor
        - Obstacles: storage crates, forklifts
    
    Patrol Route:
        Corner 1 → Aisle A → Corner 2 → Aisle B → 
        Corner 3 → Aisle C → Corner 4 → Return to Corner 1
        (Continuous loop)
    
    CONTROL OBJECTIVE:
    Patrol the warehouse perimeter continuously, checking each corner
    for anomalies (simulated by reaching waypoints)
    
    CONSTRAINTS:
    - Must maintain precise path in narrow aisles (±0.2m tolerance)
    - Slow speed near corners for sensor scanning (0.3 m/s)
    - Faster speed in straight aisles (0.7 m/s)
    - Must avoid storage crates and equipment
    - Handle dynamic obstacles (forklifts moving - simulated as disturbances)
    
    SUCCESS CRITERIA:
    - Complete full patrol loop
    - Stay within aisle boundaries (0.2m tolerance)
    - Maintain constant patrol speed
    - No collisions with obstacles
    """
    
    def __init__(self):
        self.name = "Warehouse Security Patrol"
        
        # Patrol checkpoints (corners of warehouse section)
        self.checkpoints = {
            'corner_1': [1.5, 1.5],   # Southwest
            'aisle_a_mid': [5.0, 1.5],  # South aisle midpoint
            'corner_2': [8.5, 1.5],   # Southeast
            'aisle_b_mid': [8.5, 5.0],  # East aisle midpoint
            'corner_3': [8.5, 8.5],   # Northeast
            'aisle_c_mid': [5.0, 8.5],  # North aisle midpoint
            'corner_4': [1.5, 8.5],   # Northwest
            'aisle_d_mid': [1.5, 5.0],  # West aisle midpoint
        }
        
        # Obstacles (storage crates and equipment)
        self.obstacles = [
            {'name': 'crate_1', 'position': [3.5, 4.0], 'radius': 0.4},
            {'name': 'crate_2', 'position': [6.5, 6.0], 'radius': 0.4},
            {'name': 'forklift', 'position': [4.0, 7.0], 'radius': 0.5},
        ]
        
        # Patrol route (closed loop)
        self.route = [
            self.checkpoints['corner_1'],
            self.checkpoints['aisle_a_mid'],
            self.checkpoints['corner_2'],
            self.checkpoints['aisle_b_mid'],
            self.checkpoints['corner_3'],
            self.checkpoints['aisle_c_mid'],
            self.checkpoints['corner_4'],
            self.checkpoints['aisle_d_mid'],
            self.checkpoints['corner_1'],  # Return to start
        ]
        
        # Control parameters
        self.patrol_velocity = 0.5  # m/s (moderate patrol speed)
        self.aisle_tolerance = 0.2  # m (narrow aisles)
        self.checkpoint_tolerance = 0.3  # m
        self.waypoint_tolerance = 0.3  # m (for evaluation)
        self.safe_distance = 0.2  # m (for plotting)
        
        print("="*70)
        print("SCENARIO: Warehouse Security Patrol Robot")
        print("="*70)
        print("\nMission: Continuous perimeter patrol with checkpoint monitoring")
        print("\nPatrol Route (closed loop):")
        for i, checkpoint in enumerate(self.route):
            print(f"  {i+1}. Checkpoint at ({checkpoint[0]:.1f}, {checkpoint[1]:.1f})")
        
        print(f"\nConstraints:")
        print(f"  - Patrol velocity: {self.patrol_velocity} m/s")
        print(f"  - Aisle tolerance: ±{self.aisle_tolerance} m")
        print(f"  - Obstacles to avoid: {len(self.obstacles)}")
        print("="*70)
    
    def get_reference_trajectory(self):
        """Get patrol route trajectory"""
        return WaypointPath(self.route, velocity=self.patrol_velocity)
    
    def get_obstacles_for_world(self):
        """Get obstacles for PyBullet"""
        return [(obs['position'], obs['radius']) for obs in self.obstacles]
    
    def check_waypoint_reached(self, robot_pos, waypoint_idx):
        """Check if checkpoint reached"""
        if waypoint_idx >= len(self.route):
            return False
        waypoint = self.route[waypoint_idx]
        distance = np.linalg.norm(np.array(robot_pos) - np.array(waypoint))
        return distance < self.checkpoint_tolerance
    
    def check_collision(self, robot_pos):
        """Check collision with obstacles"""
        for obs in self.obstacles:
            distance = np.linalg.norm(
                np.array(robot_pos) - np.array(obs['position'])
            )
            if distance < obs['radius']:
                return True
        return False

    def evaluate_performance(self, robot_trajectory, reference_trajectory, time_taken):
        """
        Evaluate how well the robot completed the mission
        
        Args:
            robot_trajectory: List of [x, y] actual positions
            reference_trajectory: List of [x, y] reference positions
            time_taken: Total mission time in seconds
        
        Returns:
            Dictionary of performance metrics
        """
        robot_traj = np.array(robot_trajectory)
        ref_traj = np.array(reference_trajectory)
        
        # Calculate tracking error
        min_len = min(len(robot_traj), len(ref_traj))
        tracking_errors = np.linalg.norm(
            robot_traj[:min_len] - ref_traj[:min_len], 
            axis=1
        )
        
        avg_error = np.mean(tracking_errors)
        max_error = np.max(tracking_errors)
        
        # Check if all waypoints were reached
        waypoints_reached = 0
        for i, waypoint in enumerate(self.route):
            for pos in robot_trajectory:
                if np.linalg.norm(np.array(pos) - np.array(waypoint)) < self.waypoint_tolerance:
                    waypoints_reached += 1
                    break
        
        # Calculate path length
        path_length = 0
        for i in range(len(robot_trajectory) - 1):
            path_length += np.linalg.norm(
                np.array(robot_trajectory[i+1]) - np.array(robot_trajectory[i])
            )
        
        # Success criteria
        mission_success = (
            waypoints_reached >= len(self.route) - 1 and
            avg_error < self.aisle_tolerance and
            max_error < self.aisle_tolerance * 2
        )
        
        results = {
            'mission_success': mission_success,
            'waypoints_reached': f"{waypoints_reached}/{len(self.route)}",
            'avg_tracking_error': avg_error,
            'max_tracking_error': max_error,
            'path_length': path_length,
            'time_taken': time_taken,
            'avg_velocity': path_length / time_taken if time_taken > 0 else 0
        }
        
        return results

    def print_performance_report(self, results):
        """Print a formatted performance report"""
        print("\n" + "="*70)
        print("MISSION PERFORMANCE REPORT")
        print("="*70)
        print(f"Mission Status: {'✓ SUCCESS' if results['mission_success'] else '✗ FAILED'}")
        print(f"\nWaypoints Reached: {results['waypoints_reached']}")
        print(f"Average Tracking Error: {results['avg_tracking_error']:.3f} m")
        print(f"Maximum Tracking Error: {results['max_tracking_error']:.3f} m")
        print(f"Path Length: {results['path_length']:.2f} m")
        print(f"Time Taken: {results['time_taken']:.1f} s")
        print(f"Average Velocity: {results['avg_velocity']:.2f} m/s")
        print("="*70)
    
class AgriculturalFieldScenario:
    """
    SCENARIO: Agricultural Field Monitoring Robot
    
    Setting: Autonomous robot inspecting crop rows in a field
    
    Map Layout (10x10 meters = field section):
        - Parallel crop rows (N-S orientation)
        - Turn areas at row ends
        - Inspection points along each row
    
    Crop Rows Layout:
        Row 1: x=2.0, y=1→9
        Row 2: x=4.0, y=1→9
        Row 3: x=6.0, y=1→9
        Row 4: x=8.0, y=1→9
    
    CONTROL OBJECTIVE:
    Systematically inspect all crop rows in a "boustrophedon" 
    (lawnmower) pattern:
        - Move along Row 1 (south to north)
        - Turn at end, move to Row 2
        - Move along Row 2 (north to south)
        - Continue alternating until all rows covered
    
    CONSTRAINTS:
    - Must stay centered in crop rows (±0.15m)
    - Slow inspection speed (0.4 m/s) for camera sensors
    - Sharp 180° turns at row ends
    - Cannot damage crops (strict path following)
    - Uneven terrain (higher disturbances)
    
    SUCCESS CRITERIA:
    - Cover all crop rows completely
    - Maintain row centering (< 0.15m deviation)
    - Smooth transitions between rows
    - Complete pattern without crop damage
    """
    
    def __init__(self):
        self.name = "Agricultural Field Monitoring"
        
        # Field parameters
        self.row_x_positions = [2.0, 4.0, 6.0, 8.0]  # x-coordinates of rows
        self.row_start_y = 1.0
        self.row_end_y = 9.0
        self.row_width = 2.0  # spacing between rows
        
        # Generate boustrophedon pattern
        self.route = self._generate_boustrophedon_pattern()
        
        # Obstacles (field equipment, trees)
        self.obstacles = [
            {'name': 'equipment', 'position': [5.0, 0.5], 'radius': 0.3},
            {'name': 'tree', 'position': [9.5, 5.0], 'radius': 0.4},
        ]
        
        # Control parameters
        self.inspection_velocity = 0.4  # m/s (slow for sensors)
        self.row_tolerance = 0.15  # m (must stay in row)
        self.waypoint_tolerance = 0.3  # m
        self.safe_distance = 0.15  # m (for plotting)
        
        print("="*70)
        print("SCENARIO: Agricultural Field Monitoring Robot")
        print("="*70)
        print("\nMission: Systematic crop row inspection (boustrophedon pattern)")
        print(f"\nField Layout:")
        print(f"  - Number of rows: {len(self.row_x_positions)}")
        print(f"  - Row length: {self.row_end_y - self.row_start_y} m")
        print(f"  - Row spacing: {self.row_width} m")
        print(f"\nGenerated {len(self.route)} waypoints")
        
        print(f"\nConstraints:")
        print(f"  - Inspection speed: {self.inspection_velocity} m/s")
        print(f"  - Row centering tolerance: ±{self.row_tolerance} m")
        print(f"  - Terrain: Uneven (high disturbances)")
        print("="*70)
    
    def _generate_boustrophedon_pattern(self):
        """
        Generate lawnmower pattern waypoints
        
        Pattern:
            ↓ Row1   ↓ Row3
            →→→→→   →→→→→
            ↑ Row2   ↑ Row4
        """
        waypoints = []
        
        for i, x in enumerate(self.row_x_positions):
            if i % 2 == 0:
                # Even rows: south to north
                waypoints.append([x, self.row_start_y])
                waypoints.append([x, self.row_end_y])
            else:
                # Odd rows: north to south
                waypoints.append([x, self.row_end_y])
                waypoints.append([x, self.row_start_y])
        
        return waypoints
    
    def get_reference_trajectory(self):
        """Get field inspection trajectory"""
        return WaypointPath(self.route, velocity=self.inspection_velocity)
    
    def get_obstacles_for_world(self):
        """Get obstacles for PyBullet"""
        return [(obs['position'], obs['radius']) for obs in self.obstacles]
    
    def check_waypoint_reached(self, robot_pos, waypoint_idx):
        """Check if waypoint reached"""
        if waypoint_idx >= len(self.route):
            return False
        waypoint = self.route[waypoint_idx]
        distance = np.linalg.norm(np.array(robot_pos) - np.array(waypoint))
        return distance < self.waypoint_tolerance
    
    def check_collision(self, robot_pos):
        """Check collision"""
        for obs in self.obstacles:
            distance = np.linalg.norm(
                np.array(robot_pos) - np.array(obs['position'])
            )
            if distance < obs['radius']:
                return True
        return False

    def evaluate_performance(self, robot_trajectory, reference_trajectory, time_taken):
        """
        Evaluate how well the robot completed the mission
        
        Args:
            robot_trajectory: List of [x, y] actual positions
            reference_trajectory: List of [x, y] reference positions
            time_taken: Total mission time in seconds
        
        Returns:
            Dictionary of performance metrics
        """
        robot_traj = np.array(robot_trajectory)
        ref_traj = np.array(reference_trajectory)
        
        # Calculate tracking error
        min_len = min(len(robot_traj), len(ref_traj))
        tracking_errors = np.linalg.norm(
            robot_traj[:min_len] - ref_traj[:min_len], 
            axis=1
        )
        
        avg_error = np.mean(tracking_errors)
        max_error = np.max(tracking_errors)
        
        # Check if all waypoints were reached
        waypoints_reached = 0
        for i, waypoint in enumerate(self.route):
            for pos in robot_trajectory:
                if np.linalg.norm(np.array(pos) - np.array(waypoint)) < self.waypoint_tolerance:
                    waypoints_reached += 1
                    break
        
        # Calculate path length
        path_length = 0
        for i in range(len(robot_trajectory) - 1):
            path_length += np.linalg.norm(
                np.array(robot_trajectory[i+1]) - np.array(robot_trajectory[i])
            )
        
        # Success criteria
        mission_success = (
            waypoints_reached >= len(self.route) - 1 and
            avg_error < self.row_tolerance and
            max_error < self.row_tolerance * 2
        )
        
        results = {
            'mission_success': mission_success,
            'waypoints_reached': f"{waypoints_reached}/{len(self.route)}",
            'avg_tracking_error': avg_error,
            'max_tracking_error': max_error,
            'path_length': path_length,
            'time_taken': time_taken,
            'avg_velocity': path_length / time_taken if time_taken > 0 else 0
        }
        
        return results

    def print_performance_report(self, results):
        """Print a formatted performance report"""
        print("\n" + "="*70)
        print("MISSION PERFORMANCE REPORT")
        print("="*70)
        print(f"Mission Status: {'✓ SUCCESS' if results['mission_success'] else '✗ FAILED'}")
        print(f"\nWaypoints Reached: {results['waypoints_reached']}")
        print(f"Average Tracking Error: {results['avg_tracking_error']:.3f} m")
        print(f"Maximum Tracking Error: {results['max_tracking_error']:.3f} m")
        print(f"Path Length: {results['path_length']:.2f} m")
        print(f"Time Taken: {results['time_taken']:.1f} s")
        print(f"Average Velocity: {results['avg_velocity']:.2f} m/s")
        print("="*70)
    

# Test scenario
if __name__ == "__main__":
    scenario = AgriculturalFieldScenario()
    
    # Get reference trajectory
    traj = scenario.get_reference_trajectory()
    
    # Simulate checking waypoints
    print("\nSimulating waypoint checks:")
    test_positions = [
        [1.0, 1.0],  # At warehouse
        [3.1, 2.1],  # Near Building A
        [7.0, 3.0],  # At Library
    ]
    
    for i, pos in enumerate(test_positions):
        for wp_idx in range(len(scenario.route)):
            if scenario.check_waypoint_reached(pos, wp_idx):
                print(f"Position {pos} reached waypoint {wp_idx}")
    
    # Test collision detection
    print("\nTesting collision detection:")
    print(f"Position [5.0, 5.0] (fountain center): Collision = {scenario.check_collision([5.0, 5.0])}")
    print(f"Position [2.0, 2.0] (clear area): Collision = {scenario.check_collision([2.0, 2.0])}")