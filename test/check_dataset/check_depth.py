import pickle
import sys
import os
import numpy as np

sys.path.append(os.getcwd())
# Ensure components can be imported
sys.path.append("/home/wgy/RL")

def check_depth():
    pkl_path = "/home/wgy/RL/components/data/il_dataset/FloorPlan1/FloorPlan1_px_-0.75_pz_-1.5_ry_270.0.pkl"
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print("Data loaded. Type:", type(data))
        if isinstance(data, list) and len(data) > 0:
            step0 = data[0]
            if 'obs' in step0:
                obs = step0['obs']
                state = obs.state
                # We know state[0] is RGB image
                print("State length:", len(state))
                for i, item in enumerate(state):
                    print(f"State[{i}] Type:", type(item))
                    if isinstance(item, np.ndarray):
                        print(f"  Shape: {item.shape}")
                    elif isinstance(item, dict):
                         print(f"  Keys: {list(item.keys())[:5]}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_depth()
