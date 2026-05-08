"""
Example: Training with multi-environment parallel collector.
Run this script to start parallel RL training.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from components.agents.a2c_agent import A2CAgent
from RL_training.runner.parallel_rl_train_runner import ParallelRLTrainRunner
from components.environments.precomputed_thor_env import PrecomputedThorEnv


def main():
    parser = argparse.ArgumentParser(description="Train RL agent with parallel environments")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--scenes", type=int, nargs="+", help="Scene numbers to use")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--num_steps", type=int, default=4000, help="Steps per rollout")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save frames")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Config
    navigation_config = {
        "rgb_dim": 512,
        "sg_dim": 256,
        "action_dim": 4,
        "policy_hidden": 256,
        "use_transformer": False,
        "policy_type": "LSTM",
    }
    
    agent_config = {
        "num_steps": args.num_steps,
        "gamma": 0.95,
        "alpha": 0.0002,
        "episodes": args.episodes,
        "value_coef": 0.5,
        "entropy_coef": 0.1,
    }
    
    # Dummy env for agent init
    dummy_env = PrecomputedThorEnv(render=False, scene_numbers=[1], record_dir=None)
    
    # Create agent
    print("[INFO] Initializing agent...")
    agent = A2CAgent(dummy_env, navigation_config, agent_config, device=device)
    
    # Environment kwargs
    env_kwargs = {
        "render": False,
        "record_dir": args.save_dir,
    }
    
    scene_numbers = args.scenes if args.scenes else list(range(1, 31))
    print(f"[INFO] Using scenes: {scene_numbers}")
    
    # Create parallel runner
    print(f"[INFO] Creating parallel runner with {args.num_workers} workers...")
    runner = ParallelRLTrainRunner(
        agent=agent,
        env_class="PrecomputedThorEnv",
        env_kwargs=env_kwargs,
        num_workers=args.num_workers,
        device=device,
        save_dir=args.save_dir,
        scene_numbers=scene_numbers,
    )
    
    # Train
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        runner.collector.close()


if __name__ == "__main__":
    main()
