#!/usr/bin/env python3
"""
Reinforcement Learning Control: 2D Integrator with PPO
Uses stable-baselines3 for PPO training on the integrator environment.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import argparse
import os
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# System parameters
TAU = 0.1  # Sampling period
X_BOUNDS = [-10, 10]  # State constraints
U_BOUNDS = [-1, 1]    # Input constraints
W_BOUNDS = [-0.05, 0.05]  # Disturbance bounds


class IntegratorEnv(gym.Env):
    """
    Gymnasium environment for 2D Integrator with obstacles.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    def __init__(self, render_mode=None, task='reach', difficulty=1.0):
        super().__init__()
        
        self.render_mode = render_mode
        self.task = task
        self.difficulty = difficulty
        
        # Action space: control inputs u1, u2 in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Observation space: position, goal direction, obstacle directions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        
        # Goal and obstacle - SIMPLER setup, obstacle is in the direct path
        self.goal = np.array([8.0, 8.0])
        self.obstacles = [
            {'center': np.array([0.0, 0.0]), 'radius': 2.5},  # Center obstacle
        ]
        
        # PyBullet
        self.physics_client = None
        self.robot_id = None
        self.plane_id = None
        
        # State
        self.state = None
        self.prev_state = None
        self.steps = 0
        self.max_steps = 500
        self.trajectory = []
        self.episode_count = 0
        
    def _get_obs(self):
        """Get observation vector with clear obstacle information."""
        # Goal info
        goal_vec = self.goal - self.state
        goal_dist = np.linalg.norm(goal_vec)
        
        # Obstacle info - direction and distance
        obs = self.obstacles[0]
        obs_vec = obs['center'] - self.state
        obs_dist = np.linalg.norm(obs_vec)
        obs_dir = obs_vec / (obs_dist + 1e-6)
        dist_to_surface = obs_dist - obs['radius']
        
        # Is obstacle between us and goal?
        goal_dir = goal_vec / (goal_dist + 1e-6)
        obstacle_in_path = np.dot(goal_dir, obs_dir)  # Positive = obstacle in front
        
        return np.array([
            self.state[0] / 10.0,
            self.state[1] / 10.0,
            goal_vec[0] / 10.0,
            goal_vec[1] / 10.0,
            goal_dist / 20.0,
            obs_dir[0],  # Direction TO obstacle
            obs_dir[1],
            dist_to_surface / 10.0,  # Distance to obstacle surface
            min(dist_to_surface, 5.0) / 5.0,  # Clamped proximity (0=touching, 1=far)
            obstacle_in_path,  # Is obstacle ahead?
            float(dist_to_surface < 1.0),  # Danger flag
            float(dist_to_surface < 3.0),  # Warning flag
        ], dtype=np.float32)
    
    def _get_info(self):
        return {
            'distance_to_goal': np.linalg.norm(self.state - self.goal),
            'in_obstacle': self._in_obstacle(),
        }
    
    def _in_obstacle(self):
        """Check if in any obstacle."""
        for obs in self.obstacles:
            if np.linalg.norm(self.state - obs['center']) < obs['radius']:
                return True
        return False
    
    def _dist_to_obstacle(self):
        """Get minimum distance to any obstacle surface."""
        min_dist = float('inf')
        for obs in self.obstacles:
            dist = np.linalg.norm(self.state - obs['center']) - obs['radius']
            min_dist = min(min_dist, dist)
        return min_dist
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.episode_count += 1
        
        # Start from fixed position
        self.state = np.array([-8.0, -8.0])
        self.prev_state = None
        
        self.steps = 0
        self.trajectory = [self.state.copy()]
        
        # Initialize PyBullet if rendering
        if self.render_mode == 'human' and self.physics_client is None:
            self._init_pybullet()
        
        if self.physics_client is not None:
            self._update_pybullet()
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # Clip action
        action = np.clip(action, -1.0, 1.0)
        
        # Store previous state
        self.prev_state = self.state.copy()
        
        # Disturbance
        w = np.random.uniform(W_BOUNDS[0], W_BOUNDS[1], size=2)
        
        # Dynamics
        self.state = self.state + TAU * (action + w)
        self.state = np.clip(self.state, X_BOUNDS[0], X_BOUNDS[1])
        
        self.steps += 1
        self.trajectory.append(self.state.copy())
        
        # Check conditions
        dist_to_goal = np.linalg.norm(self.state - self.goal)
        reached_goal = dist_to_goal < 1.0
        in_obstacle = self._in_obstacle()
        truncated = self.steps >= self.max_steps
        terminated = reached_goal or in_obstacle
        
        # Reward calculation
        reward = self._compute_reward(action, reached_goal, in_obstacle)
        
        # Update visualization
        if self.physics_client is not None:
            self._update_pybullet()
            if self.render_mode == 'human':
                time.sleep(TAU)
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _compute_reward(self, action, reached_goal, in_obstacle):
        """Simple, clear reward signal."""
        
        # TERMINAL REWARDS
        if reached_goal:
            return 100.0  # Big positive reward
        
        if in_obstacle:
            return -100.0  # Big negative reward - this is the key!
        
        # SHAPING REWARDS
        reward = 0.0
        
        # Progress toward goal
        dist_to_goal = np.linalg.norm(self.state - self.goal)
        if self.prev_state is not None:
            prev_dist = np.linalg.norm(self.prev_state - self.goal)
            progress = prev_dist - dist_to_goal
            reward += 5.0 * progress
        
        # Obstacle avoidance - STRONG penalty for getting close
        dist_to_obs = self._dist_to_obstacle()
        if dist_to_obs < 3.0:
            # Exponential penalty as we get closer
            penalty = 3.0 * np.exp(-dist_to_obs)
            reward -= penalty
        
        # Small time penalty to encourage efficiency
        reward -= 0.01
        
        return reward
    
    def _init_pybullet(self):
        """Initialize PyBullet visualization."""
        self.physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        
        # White ground
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(self.plane_id, -1, rgbaColor=[1, 1, 1, 1])
        
        # Robot
        robot_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, 
                                           rgbaColor=[0, 0.5, 1, 1])
        robot_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
        self.robot_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=robot_collision,
            baseVisualShapeIndex=robot_visual,
            basePosition=[self.state[0], self.state[1], 0.3]
        )
        
        # Goal marker
        goal_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, 
                                          rgbaColor=[0, 1, 0, 0.5])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=goal_visual,
                         basePosition=[self.goal[0], self.goal[1], 0.5])
        
        # Obstacles
        for obs in self.obstacles:
            obs_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=obs['radius'],
                                             length=0.5, rgbaColor=[1, 0, 0, 0.5])
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=obs_visual,
                             basePosition=[obs['center'][0], obs['center'][1], 0.25])
        
        # Workspace bounds
        corners = [[-10, -10, 0.1], [10, -10, 0.1], [10, 10, 0.1], [-10, 10, 0.1]]
        for i in range(4):
            p.addUserDebugLine(corners[i], corners[(i+1)%4], [0.2, 0.2, 0.2], lineWidth=5)
        
        # Camera
        p.resetDebugVisualizerCamera(
            cameraDistance=25, cameraYaw=0, cameraPitch=-89,
            cameraTargetPosition=[0, 0, 0]
        )
        
        self.prev_line_id = None
    
    def _update_pybullet(self):
        """Update PyBullet visualization."""
        if self.robot_id is not None:
            p.resetBasePositionAndOrientation(
                self.robot_id,
                [self.state[0], self.state[1], 0.3],
                [0, 0, 0, 1]
            )
            
            # Draw trajectory
            if len(self.trajectory) > 1:
                p.addUserDebugLine(
                    [self.trajectory[-2][0], self.trajectory[-2][1], 0.1],
                    [self.trajectory[-1][0], self.trajectory[-1][1], 0.1],
                    [1, 0, 0], lineWidth=5
                )
    
    def render(self):
        pass  # Rendering handled in step()
    
    def close(self):
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None


def train(args):
    """Train PPO agent."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    
    print("Creating training environment...")
    
    # Simple vectorized environment
    env = make_vec_env(lambda: IntegratorEnv(render_mode=None), n_envs=4)
    
    # Eval environment
    eval_env = Monitor(IntegratorEnv(render_mode=None))
    
    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(SCRIPT_DIR / "ppo_integrator_logs"),
        policy_kwargs=dict(
            net_arch=[dict(pi=[64, 64], vf=[64, 64])]
        )
    )
    
    # Eval callback to save best model
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(SCRIPT_DIR / "best_model"),
        log_path=str(SCRIPT_DIR / "eval_logs"),
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
    )
    
    print(f"Training for {args.timesteps} timesteps...")
    print("Obstacle is at center (0,0), goal is at (8,8)")
    print("Agent must learn to go AROUND the obstacle.\n")
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_cb,
        progress_bar=True
    )
    
    # Save model
    model_path = str(SCRIPT_DIR / "ppo_integrator_model")
    model.save(model_path)
    print(f"\nModel saved to {model_path}.zip")
    
    env.close()
    eval_env.close()
    return model


def evaluate(args):
    """Evaluate trained agent with visualization."""
    from stable_baselines3 import PPO
    
    # Try best model first
    best_model_path = SCRIPT_DIR / "best_model" / "best_model.zip"
    if best_model_path.exists():
        model_path = str(SCRIPT_DIR / "best_model" / "best_model")
        print("Using best model from training...")
    elif args.model:
        model_path = args.model
    else:
        model_path = str(SCRIPT_DIR / "ppo_integrator_model")
    
    if not os.path.exists(f"{model_path}.zip"):
        print(f"Model not found: {model_path}.zip")
        print("Train first with: python reinforcement_learning/integrator_rl_control.py --train")
        return
    
    print(f"Loading model from {model_path}.zip...")
    model = PPO.load(model_path)
    
    print("Creating environment with visualization...")
    env = IntegratorEnv(render_mode='human')
    
    successes = 0
    collisions = 0
    
    for episode in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        print(f"\nEpisode {episode + 1}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        print(f"Total reward: {total_reward:.2f}")
        print(f"Distance to goal: {info['distance_to_goal']:.2f}")
        print(f"Hit obstacle: {info['in_obstacle']}")
        
        if info['distance_to_goal'] < 1.0:
            successes += 1
            print(">>> SUCCESS!")
        elif info['in_obstacle']:
            collisions += 1
            print(">>> COLLISION!")
        
        # Pause between episodes
        if episode < args.episodes - 1:
            print("Starting next episode in 2 seconds...")
            time.sleep(2)
    
    print(f"\n=== SUMMARY ===")
    print(f"Successes: {successes}/{args.episodes}")
    print(f"Collisions: {collisions}/{args.episodes}")
    
    print("\nPress Ctrl+C to exit...")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    
    env.close()


def demo_random(args):
    """Run with random actions (no training required)."""
    print("Running with random actions...")
    env = IntegratorEnv(render_mode='human')
    
    for episode in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        print(f"\nEpisode {episode + 1} (Random Policy)")
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        print(f"Total reward: {total_reward:.2f}")
        print(f"Distance to goal: {info['distance_to_goal']:.2f}")
        
        if episode < args.episodes - 1:
            time.sleep(1)
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="RL PPO Control for 2D Integrator")
    parser.add_argument("--train", action="store_true", help="Train PPO agent")
    parser.add_argument("--eval", action="store_true", help="Evaluate trained agent")
    parser.add_argument("--random", action="store_true", help="Run with random actions")
    parser.add_argument("--timesteps", type=int, default=300000, help="Training timesteps")
    parser.add_argument("--episodes", type=int, default=5, help="Evaluation episodes")
    parser.add_argument("--model", type=str, default=None, help="Model path for evaluation")
    args = parser.parse_args()
    
    if args.train:
        train(args)
    elif args.eval:
        evaluate(args)
    elif args.random:
        demo_random(args)
    else:
        print("Usage:")
        print("  Train:    python reinforcement_learning/integrator_rl_control.py --train --timesteps 100000")
        print("  Evaluate: python reinforcement_learning/integrator_rl_control.py --eval --episodes 5")
        print("  Random:   python reinforcement_learning/integrator_rl_control.py --random --episodes 3")


if __name__ == "__main__":
    main()
