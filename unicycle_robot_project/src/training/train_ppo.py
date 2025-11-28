"""
Train PPO agent for unicycle robot navigation

Uses Stable-Baselines3 to train a PPO agent on the unicycle environment.
Includes curriculum learning and multiple scenarios.
"""

import sys
import os

# Fix path - go up to src directory
current_dir = os.path.dirname(os.path.abspath(__file__))  # training/
src_dir = os.path.dirname(current_dir)  # src/
sys.path.insert(0, src_dir)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np

from environment.unicycle_env import UnicycleEnv

def train_agent(scenario='simple', total_timesteps=100000, save_path='models/ppo_unicycle'):
    """Train PPO agent"""
    print("="*70)
    print(f"TRAINING PPO AGENT - Scenario: {scenario}")
    print("="*70)
    
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up to project root
    
    models_dir = os.path.join(project_root, save_path)
    logs_dir = os.path.join(project_root, 'logs')
    
    # Create directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(f'{models_dir}/best', exist_ok=True)
    os.makedirs(f'{models_dir}/checkpoints', exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"\nSaving models to: {models_dir}")
    print(f"Saving logs to: {logs_dir}")
    
    # Create vectorized environment
    n_envs = 4
    env = make_vec_env(
        lambda: UnicycleEnv(render_mode=None, scenario=scenario, max_steps=500),
        n_envs=n_envs
    )
    
    # Create evaluation environment
    eval_env = Monitor(UnicycleEnv(render_mode=None, scenario=scenario, max_steps=500))
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'{models_dir}/best',
        log_path=f'{models_dir}/logs',
        eval_freq=max(5000 // n_envs, 1),  # Adjust for parallel envs
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1),
        save_path=f'{models_dir}/checkpoints',
        name_prefix='ppo_unicycle',
        verbose=1
    )
    
    # Create PPO agent
    print("\nCreating PPO agent...")
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=512,  # Reduced for faster updates with small training
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=None
    )
    
    print(f"\nTraining for {total_timesteps} timesteps...")
    print(f"Using {n_envs} parallel environments")
    print(f"Models will be saved to: {models_dir}")
    print("\nTraining progress:")
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    final_path = f'{models_dir}/final_model'
    model.save(final_path)
    print(f"\n✓ Final model saved to: {final_path}.zip")
    
    env.close()
    eval_env.close()
    
    return model

def curriculum_training():
    """
    Train with curriculum learning: easy → medium → hard
    """
    print("\n" + "="*70)
    print("CURRICULUM TRAINING")
    print("="*70)
    
    # Stage 1: Simple (no obstacles)
    print("\n### STAGE 1: Simple Navigation (No Obstacles) ###")
    model = train_agent(scenario='simple', total_timesteps=50000, save_path='models/stage1_simple')
    
    # Stage 2: With obstacles (start from stage 1 model)
    print("\n### STAGE 2: Navigation with Obstacles ###")
    env2 = make_vec_env(
        lambda: UnicycleEnv(render_mode=None, scenario='obstacles', max_steps=200),
        n_envs=4
    )
    model.set_env(env2)
    model.learn(total_timesteps=100000, progress_bar=True)
    model.save('models/stage2_obstacles/final_model')
    env2.close()
    
    # Stage 3: Waypoints (start from stage 2 model)
    print("\n### STAGE 3: Waypoint Navigation ###")
    env3 = make_vec_env(
        lambda: UnicycleEnv(render_mode=None, scenario='waypoints', max_steps=500),
        n_envs=4
    )
    model.set_env(env3)
    model.learn(total_timesteps=100000, progress_bar=True)
    model.save('models/stage3_waypoints/final_model')
    env3.close()
    
    print("\n" + "="*70)
    print("CURRICULUM TRAINING COMPLETE!")
    print("="*70)
    print("\nModels saved:")
    print("  - models/stage1_simple/final_model")
    print("  - models/stage2_obstacles/final_model")
    print("  - models/stage3_waypoints/final_model")

def train_all_scenarios():
    """Train separate agents for each scenario"""
    scenarios = {
        'simple': 50000,
        'obstacles': 100000,
        'waypoints': 150000,
        'random': 100000
    }
    
    print("\n" + "="*70)
    print("TRAINING ALL SCENARIOS")
    print("="*70)
    
    for scenario, timesteps in scenarios.items():
        print(f"\n{'='*70}")
        print(f"Training {scenario} scenario ({timesteps} steps)")
        print(f"{'='*70}")
        
        train_agent(
            scenario=scenario,
            total_timesteps=timesteps,
            save_path=f'models/ppo_{scenario}'
        )
    
    print("\n" + "="*70)
    print("ALL SCENARIOS TRAINED!")
    print("="*70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO agent for unicycle robot')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'curriculum', 'all'],
                       help='Training mode: single scenario, curriculum, or all scenarios')
    parser.add_argument('--scenario', type=str, default='obstacles',
                       choices=['simple', 'obstacles', 'waypoints', 'random', 'random_waypoints'],  # Added random_waypoints
                       help='Scenario to train on (for single mode)')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Number of training timesteps')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # Train single scenario
        train_agent(
            scenario=args.scenario,
            total_timesteps=args.timesteps,
            save_path=f'models/ppo_{args.scenario}'
        )
    elif args.mode == 'curriculum':
        # Curriculum learning
        curriculum_training()
    elif args.mode == 'all':
        # Train all scenarios
        train_all_scenarios()
    
    print("\n✓ Training complete!")
    print("\nTo visualize training progress:")
    print("  tensorboard --logdir=logs")
    print("\nTo test the trained agent:")
    print("  python test_rl_agent.py")