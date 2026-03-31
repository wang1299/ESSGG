
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import magnum as mn
from habitat.utils.visualizations import maps
import cv2

sys.path.append(os.getcwd())

# Import our components
from components.environments.habitat_env import HabitatEnv
from components.models.feature_encoder import FeatureEncoder

def get_heading(agent_rot):
    # Convert numpy quaternion to Magnum quaternion if necessary
    # Habitat sim returns numpy-quaternion (w, x, y, z)
    if not hasattr(agent_rot, 'transform_vector'):
        # Create Magnum Quaternion from (imaginary_vector, real_scalar)
        # numpy-quaternion has attributes w (real), x, y, z (imag)
        agent_rot = mn.Quaternion(mn.Vector3(agent_rot.x, agent_rot.y, agent_rot.z), agent_rot.w)

    # Calculate heading relative to -Z axis (0 angle)
    # Transform forward vector (0,0,-1)
    # The agent faces -Z initially.
    vec = agent_rot.transform_vector(mn.Vector3(0, 0, -1))
    
    # We want 0 rad when pointing -Z (Up on map)
    # We want -pi/2 when pointing +X (Right on map)
    # arctan2(x, -z) gives +pi/2 for +X
    
    return np.arctan2(vec.x, -vec.z)

def test_exploration():
    print("=== Starting Habitat Exploration Test ===")
    
    # 1. Configuration
    dataset_root = "/home/wgy/hm3d"
    # Adjust this path if the user's structure is different, but based on context:
    # "hm3d/scene_datasets/hm3d" matches
    scene_dataset_config_file = f"{dataset_root}/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    
    # Using the specific scene 00855
    scene_id = "/home/wgy/hm3d/scene_datasets/hm3d/val/00855-c5eTyR3Rxyh/c5eTyR3Rxyh.basis.glb"
    
    print(f"Dataset Config: {scene_dataset_config_file}")
    print(f"Scene ID: {scene_id}")

    # 2. Check Paths
    if not os.path.exists(scene_dataset_config_file):
        print(f"Error: Config file not found {scene_dataset_config_file}")
        # Try finding it relative to current dir?
        # The user's workspace indicates 'hm3d' is at /home/wgy/hm3d
        pass 

    # 3. Initialize Environment
    print("Initializing HabitatEnv...")
    try:
        env = HabitatEnv(
            dataset_root=dataset_root,
            config_file=scene_dataset_config_file,
            scene_id=scene_id,
            render=False,
            width=224, # Match encoder expectation
            height=224,
            use_detector=False
        )
        print("HabitatEnv initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize HabitatEnv: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Initialize Feature Encoder (Simulate the Agent's Perception)
    print("Initializing FeatureEncoder...")
    try:
        # num_actions=8 is standard for HabitatEnv
        feature_encoder = FeatureEncoder(num_actions=8).to("cuda" if torch.cuda.is_available() else "cpu")
        feature_encoder.eval()
        print("FeatureEncoder initialized.")
    except Exception as e:
        print(f"Failed to initialize FeatureEncoder: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Run Exploration Loop
    print("\n=== Running Exploration Loop (10 Steps) ===")
    
    # Create output directory
    os.makedirs("test_vis", exist_ok=True)
    
    try:
        obs = env.reset()
        print("Reset successful. Initial Observation:")
        print(f"  State length: {len(obs.state)}")
        print(f"  RGB Shape: {obs.state[0].shape if obs.state[0] is not None else 'None'}")
        
        for step in range(10):
            print(f"\nStep {step+1}:")
            
            # Save RGB
            rgb_img = obs.state[0]
            if rgb_img is not None:
                img_path = f"test_vis/step_{step:03d}.png"
                Image.fromarray(rgb_img).save(img_path)
                print(f"  Saved view to {img_path}")
            
            # --- Generate TopDown Map ---
            try:
                # 1. Get raw map
                td_map = maps.get_topdown_map_from_sim(
                    env.sim, 
                    map_resolution=1024
                )
                # 2. Colorize
                recolor_map = maps.colorize_topdown_map(td_map)
                
                # 3. Calculate agent position on map
                agent_state = env.sim.get_agent(0).state
                pos = agent_state.position
                rot = agent_state.rotation
                
                # Careful with axes: Z -> grid_x (row), X -> grid_y (col)
                # Reverting to (Z, X) to ensure North-Up orientation (Z maps to vertical Row)
                grid_pos = maps.to_grid(
                    pos[2],  # realworld_x (Z)
                    pos[0],  # realworld_y (X)
                    (1024, 1024), 
                    sim=env.sim
                )
                
                # 4. Calculate heading
                heading = get_heading(rot)
                
                # Habitat maps usually expect 0 to be Right (+X).
                # Our heading is: 0 for North (-Z), pi/2 for West (-X), -pi/2 for East (+X).
                # Wait, arctan2(x, -z):
                # North (-Z): x=0, -z=1 -> 0
                # East (+X): x=1, -z=0 -> pi/2
                # West (-X): x=-1, -z=0 -> -pi/2
                # South (+Z): x=0, -z=-1 -> pi/-pi
                
                # We want North (-Z) to be Up (-pi/2 in image coords, or 3pi/2)
                # We want East (+X) to be Right (0 in image coords).
                # Current: North=0, East=pi/2.
                # Transform: angle_map = angle_habitat - pi/2 ?
                # 0 - pi/2 = -pi/2 (Up). Correct.
                # pi/2 - pi/2 = 0 (Right). Correct.
                
                heading_map = heading - np.pi/2

                # 5. Draw Agent
                maps.draw_agent(
                    recolor_map, 
                    grid_pos, 
                    heading_map, 
                    agent_radius_px=25
                )

                # Save map
                map_path = f"test_vis/step_{step:03d}_map.png"
                Image.fromarray(recolor_map).save(map_path)
                print(f"  Saved map to {map_path}")
                
            except Exception as e:
                print(f"  Failed to generate map: {e}")
                import traceback
                traceback.print_exc()

            # Pass through Encoder
            # Mock last action (0)
            with torch.no_grad():
                # feature_encoder expects obs, last_action
                # It handles turning obs into batch_dict
                features, _, _ = feature_encoder(obs, last_action=0)
                print(f"  Encoder Output Shape: {features.shape}")
            
            # Fixed Action Sequence for Testing Movement
            # 0: TurnR, 1: TurnL, 3: Fwd, 4: MoveR, 5: MoveL, 6: MoveB
            action_sequence = [0, 3, 3, 4, 4, 1, 5, 5, 6, 3] 
            
            if step < len(action_sequence):
                action_id = action_sequence[step]
            else:
                action_id = np.random.randint(0, 8)

            actions = env.get_actions()
            # Handle potential index error if get_actions returns fewer items
            if action_id < len(actions):
                action_name = actions[action_id]
            else:
                action_name = f"Unknown({action_id})"
            
            print(f"  Action: {action_name} ({action_id})")
            
            obs = env.step(action_id)
            
    except Exception as e:
        print(f"Exploration loop failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\nEnvironment closed.")

if __name__ == "__main__":
    test_exploration()
