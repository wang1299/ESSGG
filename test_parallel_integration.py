"""
Integration test: verify parallel collector and batch agent work together.
Run this before starting full training to catch config issues early.
"""

import sys
from pathlib import Path

import torch
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.agents.a2c_agent import A2CAgent
from components.utils.parallel_collector import ParallelEnvCollector
from components.environments.precomputed_thor_env import PrecomputedThorEnv


def test_single_env():
    """Test that a single env can be created and stepped."""
    print("[TEST 1] Single environment initialization...")
    try:
        env = PrecomputedThorEnv(render=False, scene_numbers=[1])
        obs = env.reset(scene_number=1)
        print(f"  ✓ Environment initialized")
        print(f"  ✓ Observation: state shape = {[s.shape if hasattr(s, 'shape') else len(s) for s in obs.state]}")
        
        for i in range(5):
            action = env.get_actions()[0]  # first action
            obs = env.step(action)
        print(f"  ✓ 5 steps executed successfully")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_creation():
    """Test that agent can be created with correct config."""
    print("\n[TEST 2] Agent initialization...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        dummy_env = PrecomputedThorEnv(render=False, scene_numbers=[1])
        
        navigation_config = {
            "rgb_dim": 512,
            "sg_dim": 256,
            "action_dim": 4,
            "policy_hidden": 256,
            "use_transformer": False,
            "policy_type": "LSTM",
        }
        
        agent_config = {
            "name": "a2c",
            "num_steps": 100,  # Small for testing
            "gamma": 0.95,
            "alpha": 0.0002,
            "episodes": 1,
            "value_coef": 0.5,
            "entropy_coef": 0.1,
        }
        
        agent = A2CAgent(dummy_env, navigation_config, agent_config, device=device)
        print(f"  ✓ Agent created on {device}")
        print(f"  ✓ Agent has get_batch_actions: {hasattr(agent, 'get_batch_actions')}")
        return True, agent, dummy_env
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_batch_actions(agent, dummy_env):
    """Test that agent can handle batch observations."""
    print("\n[TEST 3] Batch action generation...")
    try:
        # Create dummy observations
        obs_list = []
        for _ in range(3):
            obs = dummy_env.reset(scene_number=1)
            obs_list.append(obs)
        
        actions, values = agent.get_batch_actions(obs_list)
        print(f"  ✓ Got {len(actions)} actions for {len(obs_list)} observations")
        print(f"  ✓ Actions shape: {actions.shape}, dtype: {actions.dtype}")
        print(f"  ✓ Actions: {actions}")
        if values is not None:
            print(f"  ✓ Values shape: {values.shape}")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_collector():
    """Test parallel environment collector."""
    print("\n[TEST 4] Parallel environment collector...")
    try:
        env_kwargs = {"render": False}
        
        collector = ParallelEnvCollector(
            num_workers=2,
            env_class="PrecomputedThorEnv",
            env_kwargs=env_kwargs,
            scene_numbers=[1, 2, 3],
        )
        print(f"  ✓ Collector initialized with 2 workers")
        
        # Step a few times
        for step in range(3):
            actions = np.array([0, 1])  # different actions per worker
            transitions = collector.step(actions)
            print(f"  ✓ Step {step}: collected {len(transitions)} transitions")
            for trans in transitions:
                assert "obs" in trans
                assert "action" in trans
                assert "reward" in trans
                assert "done" in trans
        
        collector.close()
        print(f"  ✓ Collector closed cleanly")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test agent + collector together."""
    print("\n[TEST 5] Integration: agent + collector...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create agent
        dummy_env = PrecomputedThorEnv(render=False, scene_numbers=[1])
        navigation_config = {
            "rgb_dim": 512,
            "sg_dim": 256,
            "action_dim": 4,
            "policy_hidden": 256,
            "use_transformer": False,
            "policy_type": "LSTM",
        }
        agent_config = {
            "name": "a2c",
            "num_steps": 10,
            "gamma": 0.95,
            "alpha": 0.0002,
            "episodes": 1,
            "value_coef": 0.5,
            "entropy_coef": 0.1,
        }
        agent = A2CAgent(dummy_env, navigation_config, agent_config, device=device)
        
        # Create collector
        env_kwargs = {"render": False}
        collector = ParallelEnvCollector(
            num_workers=2,
            env_class="PrecomputedThorEnv",
            env_kwargs=env_kwargs,
            scene_numbers=[1, 2],
        )
        
        # Run interaction loop
        obs_batch = [dummy_env.reset(scene_number=1) for _ in range(2)]
        for step in range(3):
            actions, values = agent.get_batch_actions(obs_batch)
            print(f"  ✓ Step {step}: agent produced {len(actions)} actions")
            
            # Execute in collector
            transitions = collector.step(actions)
            print(f"  ✓ Step {step}: collector executed, got {len(transitions)} transitions")
        
        collector.close()
        print(f"  ✓ Integration test passed")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("PARALLEL RL TRAINING - INTEGRATION TESTS")
    print("=" * 80)
    
    results = []
    
    # Test 1: Single env
    results.append(("Single Environment", test_single_env()))
    
    # Test 2: Agent creation
    ok, agent, env = test_agent_creation()
    results.append(("Agent Creation", ok))
    
    if ok and agent is not None:
        # Test 3: Batch actions
        results.append(("Batch Actions", test_batch_actions(agent, env)))
    
    # Test 4: Parallel collector (can run independent of agent)
    results.append(("Parallel Collector", test_parallel_collector()))
    
    # Test 5: Integration
    results.append(("Integration", test_integration()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30s} {status}")
    
    all_passed = all(r[1] for r in results)
    print("=" * 80)
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to start training.")
        print("\nStart training with:")
        print("  python train_parallel.py --num_workers 4 --episodes 500")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix errors above before training.")
        return 1


if __name__ == "__main__":
    exit(main())
