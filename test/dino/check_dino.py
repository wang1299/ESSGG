import os
import sys
import pickle
import numpy as np
import cv2
import torch
import json
import re
from PIL import Image

# Add potential paths to sys.path
sys.path.append(os.getcwd())

try:
    from components.detectors.grounding_dino_adapter import GroundingDINODetector
except ImportError as e:
    print(f"Error importing components: {e}")
    print("Please run this script from the project root (e.g. /home/wgy/RL)")
    sys.exit(1)

def load_objects_prompt(json_path):
    if not os.path.exists(json_path):
        print(f"Warning: Object types file not found at {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Keys are like "CounterTop", "microwave" etc.
    # Convert CamelCase to Spaced: CounterTop -> Counter Top
    objects = list(data.keys())
    
    processed_objects = []
    for obj in objects:
        # Split CamelCase: 'CounterTop' -> 'Counter Top'
        # Also handles 'TV' -> 'TV', 'SinkBasin' -> 'Sink Basin'
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', obj)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
        processed_objects.append(s2)
    
    # Join with " . " as required by GroundingDINO
    prompt = " . ".join(processed_objects) + " ."
    print(f"Generated prompt with {len(objects)} categories.")
    # Check length roughly
    if len(prompt.split()) > 200:
        print(f"Warning: Prompt length ({len(prompt.split())} words) is quite long. DINO might truncate.")
        
    return prompt

def load_pickle_data(pkl_path):
    if not os.path.exists(pkl_path):
        print(f"File not found: {pkl_path}")
        return None
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading pickle {pkl_path}: {e}")
        return None

def extract_image(data):
    # Try different keys or attributes
    image = None
    
    # 1. If data is a list (likely [step1, step2, ...])
    if isinstance(data, list) and len(data) > 0:
        # Check first element
        item = data[0]
        if isinstance(item, dict) and 'obs' in item:
            obs = item['obs']
            # Based on inspection, obs.state is a list and index 0 is image
            if hasattr(obs, 'state') and isinstance(obs.state, list):
                if len(obs.state) > 0 and isinstance(obs.state[0], np.ndarray):
                    image = obs.state[0]
    
    # 2. If date is a dict (fallback)
    elif isinstance(data, dict):
        keys = data.keys()
        # Common keys in AI2-THOR datasets: 'frame', 'rgb', 'image'
        for k in ['frame', 'rgb', 'image']:
            if k in keys:
                image = data[k]
                break
    
    return image

def extract_gt_objects(data):
    """Attempt to find ground truth objects in the data."""
    objects = []
    
    # If list, try to find in first element's observation info or similar
    if isinstance(data, list) and len(data) > 0:
        item = data[0]
        # Metadata might be in obs.info
        if 'obs' in item and hasattr(item['obs'], 'info'):
             info = item['obs'].info
             if info and 'objects' in info:
                 objects = info['objects']
    
    if not objects and isinstance(data, dict):
        if 'objects' in data:
            objects = data['objects']
        elif 'metadata' in data and 'objects' in data['metadata']:
            objects = data['metadata']['objects']
    elif hasattr(data, 'metadata') and 'objects' in data.metadata:
         objects = data.metadata['objects']
         
    return objects

def draw_detections(image, detections, output_path, gt_objects=None):
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    h, w, _ = img_bgr.shape
    
    # Draw DINO detections (Blue/Green)
    print("\n--- DINO Detections ---")
    for det in detections:
        label = det['label']
        score = det['score']
        bbox = det['bbox'] # [x1, y1, x2, y2]
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Color: Green
        color = (0, 255, 0)
        
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        
        text = f"{label} {score:.2f}"
        cv2.putText(img_bgr, text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"Detected: {label} ({score:.2f}) at {bbox}")

    # Draw Ground Truth if available (Red) but only checking rough visibility
    if gt_objects:
        print("\n--- GT Objects (Visible) ---")
        visible_gt_count = 0
        for obj in gt_objects:
            if isinstance(obj, dict) and obj.get('visible', False):
                print(f"GT Object (Visible): {obj.get('objectType')} (ID: {obj.get('objectId')})")
                visible_gt_count += 1
        
        if visible_gt_count == 0:
             print("No GT objects marked directly as 'visible' in metadata (or metadata format differs).")

    cv2.imwrite(output_path, img_bgr)
    print(f"\nVisualization saved to: {output_path}")

def main():
    # --- Configuration ---
    # 1. Select a sample file from the dataset
    pkl_file = "components/data/il_dataset/FloorPlan1/FloorPlan1_px_-0.75_pz_-1.5_ry_270.0.pkl"
    # Use absolute path just in case
    pkl_path = os.path.join("/home/wgy/RL", pkl_file)
    
    # 2. DINO Config (Assuming default paths from your project structure)
    grounding_dino_root = "/home/wgy/GroundingDINO"
    
    config_path = os.path.join(grounding_dino_root, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
    checkpoint_path = os.path.join(grounding_dino_root, "weights/groundingdino_swint_ogc.pth")
    
    # 3. Prompts
    # Load from object_types.json
    object_types_path = "/home/wgy/RL/components/data/scene_graph_mappings/default/object_types.json"
    full_prompt = load_objects_prompt(object_types_path)
    
    if full_prompt:
        prompt = full_prompt
    else:
        # Fallback if file not found
        print("Using fallback prompt.")
        prompt = "chair . table . sofa . bed . plant . tv . refrigerator . microwave . sink . toilet"
    
    # --- Execution ---
    print(f"Loading data from: {pkl_path}")
    data = load_pickle_data(pkl_path)
    if data is None:
        return

    image = extract_image(data)
    if image is None:
        print("Could not find 'frame' or 'rgb' or 'image' in the pickle data.")
        keys = list(data.keys()) if isinstance(data, dict) else dir(data)
        print(f"Available keys/attributes: {keys}")
        return
    
    print(f"Image loaded. Shape: {image.shape}")
    
    # GT Objects
    gt_objects = extract_gt_objects(data)
    
    # Initialize Detector
    print("Initializing GroundingDINO...")
    try:
        detector = GroundingDINODetector(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            text_prompt=prompt,
            box_threshold=0.20, # Following user's code in main.py
            text_threshold=0.20
        )
    except Exception as e:
        print(f"Failed to load detector: {e}")
        return

    # Run Detection
    print("Running detection...")
    try:
        detections = detector.detect(rgb_image=image)
    except Exception as e:
        print("Detection failed.")
        print(e)
        return

    # Visualize
    output_filename = "dino_visualization_test.png"
    draw_detections(image, detections, output_filename, gt_objects)

if __name__ == "__main__":
    main()
