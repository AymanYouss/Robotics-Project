"""
Gymnasium Environment for Unicycle Robot

This wraps the PyBullet unicycle simulation in a Gym interface
for reinforcement learning training.

State Space:
    - Robot position: (x, y)
    - Robot heading: θ
    - Goal position: (goal_x, goal_y)
    - Distance to goal
    - Heading to goal
    - Obstacle distances (4 directions)

Action Space:
    - Linear velocity: u₁ ∈ [0.25, 1.0]
    - Angular velocity: u₂ ∈ [-1.0, 1.0]

Reward Function:
    - Goal reached: +100
    - Collision: -50
    - Progress: +10 * (dist_old - dist_new)
    - Heading alignment: -0.1 * |heading_error|
    - Time penalty: -0.1 per step
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.unicycle import UnicycleRobot

class UnicycleEnv(gym.Env):
    """
    Gymnasium environment for unicycle robot navigation
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None, scenario='simple', max_steps=None):
        """
        Initialize environment
        
        Args:
            render_mode: 'human' for GUI, None for headless
            scenario: 'simple', 'obstacles', 'waypoints'
            max_steps: Maximum steps per episode
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.scenario = scenario
        
        
        if max_steps is None:
            if scenario == 'waypoints':
                self.max_steps = 500  # Longer for waypoints
            elif scenario == 'obstacles':
                self.max_steps = 300
            else:
                self.max_steps = 200
        else:
            self.max_steps = max_steps
        self.current_step = 0
        
        # Action space: [u1, u2]
        self.action_space = spaces.Box(
            low=np.array([0.25, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: [x, y, theta, goal_x, goal_y, dist_to_goal, 
        #                     heading_to_goal, obs_dist_N, obs_dist_E, obs_dist_S, obs_dist_W]
        self.observation_space = spaces.Box(
            low=np.array([
                0, 0, -np.pi,           # Robot: x, y, theta
                0, 0,                   # Current target goal: x, y
                0, -np.pi,              # Distance, heading to current goal
                0, 0, 0, 0,             # Obstacle distances (N, E, S, W)
                0, 0,                   # Final goal position: x, y
                0, 0,                   # Waypoint position: x, y (0,0 if no waypoint)
                0, 0,                   # Obstacle 1 position: x, y
                0, 0,                   # Obstacle 2 position: x, y
                0, 0                    # Obstacle 3 position: x, y
            ]),
            high=np.array([
                10, 10, np.pi,          # Robot
                10, 10,                 # Current target
                15, np.pi,              # Distance, heading
                15, 15, 15, 15,         # Obstacle distances
                10, 10,                 # Final goal
                10, 10,                 # Waypoint
                10, 10,                 # Obstacle 1
                10, 10,                 # Obstacle 2
                10, 10                  # Obstacle 3
            ]),
            dtype=np.float32
        )
        
        # PyBullet setup
        self.physics_client = None
        self.robot = None
        self.obstacles = []
        
        # Scenario parameters
        self.goal_pos = None
        self.start_pos = None
        self.obstacle_positions = []
        
        self._setup_scenario()
        
        # Tracking
        self.prev_distance = None
        self.episode_reward = 0
        

    def _setup_scenario(self):
        """Setup scenario-specific parameters"""
        if self.scenario == 'simple':
            # No obstacles, fixed goal
            self.start_pos = [1.0, 1.0, 0.0]
            self.goal_pos = [9.0, 9.0]
            self.obstacle_positions = []
            self.at_waypoint = True  # No waypoint, so always "at" it
            
        elif self.scenario == 'obstacles':
            # Multiple obstacles
            self.start_pos = [1.0, 1.0, 0.0]
            self.goal_pos = [9.0, 9.0]
            self.obstacle_positions = [
                ([5.0, 5.0], 0.8),
                ([3.0, 6.0], 0.5),
                ([7.0, 4.0], 0.5)
            ]
            self.at_waypoint = True  # No waypoint, so always "at" it
            
        elif self.scenario == 'waypoints':
            # Visit waypoint then goal
            self.start_pos = [1.0, 1.0, 0.0]
            self.waypoint_pos = [3.0, 8.0]
            self.goal_pos = [9.0, 9.0]
            self.obstacle_positions = [([5.0, 5.0], 0.8), ([3.0, 5.0], 0.8)]
            self.at_waypoint = False  # Has waypoint
        
            
        elif self.scenario == 'random':
            # Random everything each episode (set in reset)
            self.start_pos = None
            self.goal_pos = None
            self.waypoint_pos = None
            self.obstacle_positions = []
            self.at_waypoint = True  # No waypoint in random
            
        elif self.scenario == 'random_waypoints':
            # Random waypoints scenario
            self.start_pos = None
            self.goal_pos = None
            self.waypoint_pos = None
            self.obstacle_positions = []
            self.at_waypoint = False  # Has waypoint


    def _init_pybullet(self):
        """Initialize PyBullet simulation"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        
        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Ground
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(self.plane_id, -1, rgbaColor=[0.9, 0.9, 0.9, 1])
        
        # Camera (if GUI)
        if self.render_mode == 'human':
            p.resetDebugVisualizerCamera(
                cameraDistance=15,
                cameraYaw=0,
                cameraPitch=-60,
                cameraTargetPosition=[5, 5, 0]
            )
        
        # Create obstacles
        self.obstacles = []
        for pos, radius in self.obstacle_positions:
            collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=0.5)
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=0.5,
                                            rgbaColor=[0.8, 0.2, 0.2, 0.7])
            obs_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[pos[0], pos[1], 0.25]
            )
            self.obstacles.append((obs_id, pos, radius))
        
        # Waypoint marker (yellow sphere for waypoints scenario)
        if self.scenario == 'waypoints' and hasattr(self, 'waypoint_pos'):
            waypoint_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, 
                                                rgbaColor=[1.0, 1.0, 0.0, 0.8])
            self.waypoint_marker = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=waypoint_shape,
                basePosition=[self.waypoint_pos[0], self.waypoint_pos[1], 0.15]
            )
            # Label
            if self.render_mode == 'human':
                p.addUserDebugText("WAYPOINT", 
                                [self.waypoint_pos[0], self.waypoint_pos[1], 0.6],
                                textColorRGB=[0, 0, 0], textSize=1.5)
        
        # Goal marker (green sphere)
        if self.goal_pos:
            goal_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, 
                                            rgbaColor=[0.2, 0.8, 0.2, 0.8])
            self.goal_marker = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=goal_shape,
                basePosition=[self.goal_pos[0], self.goal_pos[1], 0.15]
            )
            # Label
            if self.render_mode == 'human':
                p.addUserDebugText("GOAL", 
                                [self.goal_pos[0], self.goal_pos[1], 0.6],
                                textColorRGB=[0, 0, 0], textSize=1.5)


    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Random scenario setup
        if self.scenario == 'random':
            self.start_pos = [
                np.random.uniform(0.5, 2.0),
                np.random.uniform(0.5, 2.0),
                np.random.uniform(-np.pi, np.pi)
            ]
            self.goal_pos = [
                np.random.uniform(8.0, 9.5),
                np.random.uniform(8.0, 9.5)
            ]
            
            # Random obstacles (1-3 obstacles)
            n_obstacles = np.random.randint(1, 4)
            self.obstacle_positions = []
            for _ in range(n_obstacles):
                obs_x = np.random.uniform(3.0, 7.0)
                obs_y = np.random.uniform(3.0, 7.0)
                obs_radius = np.random.uniform(0.4, 0.8)
                self.obstacle_positions.append(([obs_x, obs_y], obs_radius))
        
        elif self.scenario == 'random_waypoints':
            # Random start, waypoint, goal, obstacles
            self.start_pos = [
                np.random.uniform(0.5, 2.0),
                np.random.uniform(0.5, 2.0),
                np.random.uniform(-np.pi, np.pi)
            ]
            
            # Waypoint in middle area
            self.waypoint_pos = [
                np.random.uniform(3.0, 7.0),
                np.random.uniform(3.0, 7.0)
            ]
            
            # Goal in far corner
            self.goal_pos = [
                np.random.uniform(8.0, 9.5),
                np.random.uniform(8.0, 9.5)
            ]
            
            # Random obstacles (1-2 obstacles for waypoints)
            n_obstacles = np.random.randint(1, 3)
            self.obstacle_positions = []
            for _ in range(n_obstacles):
                obs_x = np.random.uniform(3.0, 7.0)
                obs_y = np.random.uniform(3.0, 7.0)
                obs_radius = np.random.uniform(0.5, 0.9)
                # Make sure not too close to waypoint
                if np.sqrt((obs_x - self.waypoint_pos[0])**2 + 
                        (obs_y - self.waypoint_pos[1])**2) > 1.5:
                    self.obstacle_positions.append(([obs_x, obs_y], obs_radius))
        
        # Initialize waypoint tracking BEFORE creating robot
        if self.scenario in ['waypoints', 'random_waypoints']:
            self.at_waypoint = False
            self.current_goal = self.waypoint_pos
        else:
            self.current_goal = self.goal_pos
        
        # Initialize PyBullet if needed
        if self.physics_client is None:
            self._init_pybullet()
        
        # Create/reset robot
        if self.robot is None:
            self.robot = UnicycleRobot(
                start_pos=self.start_pos,
                tau=0.1,
                add_disturbance=True
            )
        else:
            self.robot.reset(self.start_pos)
        
        # Reset tracking
        self.current_step = 0
        self.episode_reward = 0
        self.prev_distance = self._distance_to_goal()
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info

    def _get_obs(self):
        """Get current observation with all dynamic positions"""
        state = self.robot.get_state()
        x, y, theta = state
        
        # Current goal (waypoint or final goal)
        if self.scenario == 'waypoints':
            current_goal = self.current_goal
            final_goal = self.goal_pos
            waypoint = self.waypoint_pos
        else:
            current_goal = self.goal_pos
            final_goal = self.goal_pos
            waypoint = [0.0, 0.0]  # No waypoint
        
        # Distance and heading to current goal
        dist_to_goal = np.sqrt((current_goal[0] - x)**2 + (current_goal[1] - y)**2)
        heading_to_goal = np.arctan2(current_goal[1] - y, current_goal[0] - x)
        
        # Obstacle distances (ray casting)
        obs_distances = self._get_obstacle_distances(x, y, theta)
        
        # Obstacle positions (pad with zeros if fewer than 3)
        obstacle_positions = []
        for i in range(3):
            if i < len(self.obstacle_positions):
                pos, radius = self.obstacle_positions[i]
                obstacle_positions.extend([pos[0], pos[1]])
            else:
                obstacle_positions.extend([0.0, 0.0])  # Padding
        
        # Build observation vector
        obs = np.array([
            # Robot state (3)
            x, y, theta,
            
            # Current target goal (2)
            current_goal[0], current_goal[1],
            
            # Distance and heading to current goal (2)
            dist_to_goal, heading_to_goal,
            
            # Obstacle distances in 4 directions (4)
            obs_distances[0],  # North
            obs_distances[1],  # East
            obs_distances[2],  # South
            obs_distances[3],  # West
            
            # Final goal position (2)
            final_goal[0], final_goal[1],
            
            # Waypoint position (2)
            waypoint[0], waypoint[1],
            
            # Obstacle positions (6 = 3 obstacles × 2)
            *obstacle_positions
        ], dtype=np.float32)
        
        return obs


    def _get_obstacle_distances(self, x, y, theta):
        """Get distances to nearest obstacles in 4 cardinal directions"""
        max_dist = 15.0
        directions = [
            (0, 1),   # North
            (1, 0),   # East
            (0, -1),  # South
            (-1, 0)   # West
        ]
        
        distances = []
        for dx, dy in directions:
            min_dist = max_dist
            
            # Check obstacles
            for _, obs_pos, obs_radius in self.obstacles:
                # Vector from robot to obstacle
                to_obs = np.array([obs_pos[0] - x, obs_pos[1] - y])
                direction = np.array([dx, dy])
                
                # Project onto direction
                projection = np.dot(to_obs, direction)
                
                if projection > 0:  # In front
                    # Perpendicular distance
                    perp_dist = abs(to_obs[1]*dx - to_obs[0]*dy)
                    
                    if perp_dist < obs_radius:
                        # Distance along ray
                        dist = projection - np.sqrt(obs_radius**2 - perp_dist**2)
                        min_dist = min(min_dist, max(0, dist))
            
            # Check boundaries
            if dx > 0:  # East
                min_dist = min(min_dist, 10.0 - x)
            elif dx < 0:  # West
                min_dist = min(min_dist, x)
            elif dy > 0:  # North
                min_dist = min(min_dist, 10.0 - y)
            elif dy < 0:  # South
                min_dist = min(min_dist, y)
            
            distances.append(min_dist)
        
        return distances
    
    def _distance_to_goal(self):
        """Calculate distance to current goal"""
        state = self.robot.get_state()
        x, y = state[0], state[1]
        
        if self.scenario == 'waypoints':
            goal = self.current_goal
        else:
            goal = self.goal_pos
        
        return np.sqrt((goal[0] - x)**2 + (goal[1] - y)**2)
    
    def _check_collision(self):
        """Check if robot collides with obstacles or boundaries"""
        state = self.robot.get_state()
        x, y = state[0], state[1]
        
        # Boundary collision
        if x < 0.2 or x > 9.8 or y < 0.2 or y > 9.8:
            return True
        
        # Obstacle collision
        robot_radius = self.robot.radius
        for _, obs_pos, obs_radius in self.obstacles:
            dist = np.sqrt((obs_pos[0] - x)**2 + (obs_pos[1] - y)**2)
            if dist < (robot_radius + obs_radius):
                return True
        
        return False
    
    def step(self, action):
        """Execute one step"""
        # Apply action
        u1, u2 = action
        self.robot.step(u1, u2)
        p.stepSimulation()
        
        self.current_step += 1
        
        obs = self._get_obs()
        state = self.robot.get_state()
        x, y, theta = state
        
        # Calculate reward
        reward = 0
        terminated = False
        truncated = False
        
        # 1. Distance-based reward (progress)
        current_distance = self._distance_to_goal()
        progress = self.prev_distance - current_distance
        reward += 10.0 * progress
        self.prev_distance = current_distance
        
        # 2. Goal reached
                # 2. Goal reached
        if current_distance < 0.5:
            # Simpler debug that works for all scenarios
            print(f"  [DEBUG] Distance < 0.5! scenario={self.scenario}")
            
            if hasattr(self, 'at_waypoint') and self.scenario in ['waypoints', 'random_waypoints'] and not self.at_waypoint:
                # Reached waypoint
                reward += 50
                self.at_waypoint = True
                self.current_goal = self.goal_pos
                self.prev_distance = self._distance_to_goal()
                print(f"  ✓ Waypoint reached! Switching to goal: {self.goal_pos}")
            else:
                # Reached final goal
                reward += 100
                terminated = True
                print(f"  ✓ Final goal reached! Terminating. Episode reward: {self.episode_reward + reward:.2f}")
        
        # 3. Collision penalty
        if self._check_collision():
            reward -= 50
            terminated = True
            print(f"  Collision! Episode reward: {self.episode_reward + reward:.2f}")
        
        # 4. Heading alignment reward
        if self.scenario == 'waypoints':
            goal = self.current_goal
        else:
            goal = self.goal_pos
        
        heading_to_goal = np.arctan2(goal[1] - y, goal[0] - x)
        heading_error = abs(theta - heading_to_goal)
        heading_error = min(heading_error, 2*np.pi - heading_error)  # Wrap
        reward += -0.1 * heading_error
        
        # 5. Time penalty (encourage efficiency)
        reward += -0.1
        
        # 6. Max steps
        if self.current_step >= self.max_steps:
            truncated = True
            print(f"  Max steps reached. Episode reward: {self.episode_reward + reward:.2f}")
        
        self.episode_reward += reward
        
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_info(self):
        """Get additional info"""
        state = self.robot.get_state()
        return {
            'position': state[:2],
            'heading': state[2],
            'distance_to_goal': self._distance_to_goal(),
            'step': self.current_step
        }
    
    def render(self):
        """Render environment (handled by PyBullet GUI)"""
        if self.render_mode == 'human':
            pass  # PyBullet handles rendering
    
    def close(self):
        """Close environment"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


# Test the environment
if __name__ == "__main__":
    print("Testing UnicycleEnv...")
    
    # Test with GUI
    env = UnicycleEnv(render_mode='human', scenario='obstacles', max_steps=200)
    
    print("\nRunning random policy for 3 episodes...")
    for episode in range(3):
        obs, info = env.reset()
        print(f"\nEpisode {episode + 1}")
        episode_reward = 0
        
        for step in range(200):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} finished: {step + 1} steps, reward: {episode_reward:.2f}")
    
    env.close()
    print("\nEnvironment test complete!")