import pickle
import sys
import os
import numpy as np

sys.path.append(os.getcwd())

def check_transition_table():
    pkl_path = "/home/wgy/RL/components/data/transition_tables/FloorPlan1.pkl"
    if not os.path.exists(pkl_path):
        print(f"File {pkl_path} not found.")
        return

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print("Data keys:", data.keys())
        table = data.get("table")
        if table:
            print(f"Table size: {len(table)}")
            # Pick one event
            key = list(table.keys())[0]
            event = table[key]
            print(f"Sample Event Type: {type(event)}")
            print("Event attributes:", dir(event))
            
            if hasattr(event, "depth_frame"):
                d = event.depth_frame
                if d is not None:
                    print(f"Depth frame shape: {d.shape}")
                else:
                    print("Depth frame is None")
            else:
                print("Event has no depth_frame attribute")
                
            if hasattr(event, "metadata"):
                print("Metadata keys:", event.metadata.keys())

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_transition_table()
