import pickle
import sys
import os
import numpy as np

# Set path to a sample pickle file
sample_file = "/home/wgy/RL/components/data/il_dataset/FloorPlan1/FloorPlan1_px_-0.75_pz_-1.5_ry_270.0.pkl"

if not os.path.exists(sample_file):
    print(f"File not found: {sample_file}")
    sys.exit(1)

try:
    with open(sample_file, 'rb') as f:
        data = pickle.load(f)
    
    print("Keys in pickle file:", data.keys())
    
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f"Key: {key}, Type: {type(value)}, Shape: {value.shape}")
        else:
             print(f"Key: {key}, Type: {type(value)}")
             
except Exception as e:
    print(f"Error loading pickle: {e}")
