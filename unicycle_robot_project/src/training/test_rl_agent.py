"""
Test trained RL agent in PyBullet with visualization
"""

import sys
import os

# Fix path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from stable_baselines3 import PPO
from environment.unicycle_env import UnicycleEnv
import time

def test_agent(model_path, scenario='obstacles', n_episodes=5, render=True):
    """
    Test trained agent
    
    Args:
        model_path: Path to trained model
        scenario: Which scenario to test on
        n_episodes: Number of test episodes
        render: Whether to show PyBullet GUI
    """
    print("="*70)
    print(f"TESTING RL AGENT - Scenario: {scenario}")
    print("="*70)
    
    # Load trained model
    print(f"\nLoading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    render_mode = 'human' if render else None
    env = UnicycleEnv(render_mode=render_mode, scenario=scenario, max_steps=500)
    
    # Test episodes
    print(f"\nRunning {n_episodes} test episodes...")
    
    results = {
        'success': 0,
        'collision': 0,
        'timeout': 0,
        'rewards': [],
        'steps': []
    }
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\n{'='*70}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"{'='*70}")
        
        while not done and step < 500:
            # Get action from trained policy
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated
            
            if render:
                time.sleep(0.02)  # Slow down for visualization
        
        # Record results
        results['rewards'].append(episode_reward)
        results['steps'].append(step)
        
        # Determine outcome
        if terminated and reward > 0:
            results['success'] += 1
            outcome = "✓ SUCCESS"
        elif terminated and reward < 0:
            results['collision'] += 1
            outcome = "✗ COLLISION"
        else:
            results['timeout'] += 1
            outcome = "⊗ TIMEOUT"
        
        print(f"\nOutcome: {outcome}")
        print(f"Steps: {step}, Reward: {episode_reward:.2f}")
    
    env.close()
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Episodes: {n_episodes}")
    print(f"Success: {results['success']} ({results['success']/n_episodes*100:.1f}%)")
    print(f"Collision: {results['collision']} ({results['collision']/n_episodes*100:.1f}%)")
    print(f"Timeout: {results['timeout']} ({results['timeout']/n_episodes*100:.1f}%)")
    print(f"Average reward: {sum(results['rewards'])/n_episodes:.2f}")
    print(f"Average steps: {sum(results['steps'])/n_episodes:.1f}")
    print("="*70)

    if render:
        print("\nPress Enter to close PyBullet window...")
        input()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained RL agent')
    
    # Get absolute path to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    default_model = os.path.join(project_root, 'models', 'ppo_obstacles', 'final_model')
    
    parser.add_argument('--model', type=str, default=default_model,
                       help='Path to trained model (without .zip extension)')
    parser.add_argument('--scenario', type=str, default='obstacles',
                   choices=['simple', 'obstacles', 'waypoints', 'random', 'random_waypoints'],  # Added random_waypoints
                   help='Scenario to test on')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of test episodes')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable visualization')
    
    
    args = parser.parse_args()
    
    print(f"Looking for model at: {args.model}.zip")
    
    test_agent(
        model_path=args.model,
        scenario=args.scenario,
        n_episodes=args.episodes,
        render=not args.no_render
    )