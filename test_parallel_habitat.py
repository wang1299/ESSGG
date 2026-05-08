"""
Quick test of ParallelHabitatCollector to verify multi-process communication works.
Tests basic reset and step operations without training.
"""

import sys
import os
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from components.environments.parallel_habitat_collector import ParallelHabitatCollector


def test_basic_reset_step():
    """Test basic reset and step operations."""
    print("[TEST] Starting ParallelHabitatCollector test...")
    
    # Check if Habitat dataset exists
    dataset_root = "/home/wgy/data/versioned_data/hssd-hab"
    config_file = "/home/wgy/data/versioned_data/hssd-hab/hssd_hab.yaml"
    
    if not os.path.exists(dataset_root):
        print(f"[WARNING] Dataset root not found: {dataset_root}")
        print("Please ensure Habitat HSSD dataset is downloaded")
        return False
    
    if not os.path.exists(config_file):
        print(f"[WARNING] Config file not found: {config_file}")
        return False
    
    try:
        # Create collector with 2 workers
        num_workers = 2
        print(f"\n[INFO] Creating collector with {num_workers} workers...")
        
        collector = ParallelHabitatCollector(
            num_workers=num_workers,
            dataset_root=dataset_root,
            config_file=config_file,
            base_scene_ids=["00000-8SqXd7EB5V"],  # Sample scene
            env_kwargs={
                "render": False,
                "width": 300,
                "height": 300,
            },
            timeout=60.0,
        )
        
        # Test reset
        print("[INFO] Testing reset_all()...")
        obs_list = collector.reset_all(random_start=True)
        print(f"[OK] Got {len(obs_list)} observations from reset")
        
        if obs_list[0] is not None:
            print(f"[OK] First obs shape: {obs_list[0].state[0].shape if obs_list[0].state else 'None'}")
        
        # Test step
        print("\n[INFO] Testing step_all()...")
        actions = [0, 1]  # turn_left, turn_right
        obs_list = collector.step_all(actions)
        print(f"[OK] Got {len(obs_list)} observations from step")
        
        # Test multiple steps
        print("\n[INFO] Testing multiple steps...")
        for i in range(5):
            actions = [i % 3] * num_workers  # Cycle through actions
            obs_list = collector.step_all(actions)
            rewards = [obs.reward for obs in obs_list]
            print(f"  Step {i+1}: rewards={rewards}")
        
        # Cleanup
        print("\n[INFO] Closing collector...")
        collector.close()
        
        print("\n[SUCCESS] All tests passed!")
        return True
    
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            collector.close()
        except:
            pass
        return False


if __name__ == "__main__":
    success = test_basic_reset_step()
    sys.exit(0 if success else 1)
