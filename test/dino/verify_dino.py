import os
import sys
import numpy as np
from PIL import Image
import torch

# Ensure the current directory is in the path so we can import components
sys.path.append(os.getcwd())

try:
    from components.detectors.grounding_dino_adapter import GroundingDINODetector
except ImportError as e:
    print(f"Error importing GroundingDINODetector: {e}")
    sys.exit(1)

def test_dino():
    # Paths
    grounding_dino_root = "/home/wgy/GroundingDINO"
    config_path = os.path.join(grounding_dino_root, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
    checkpoint_path = os.path.join(grounding_dino_root, "weights/groundingdino_swint_ogc.pth")
    image_path = os.path.join(grounding_dino_root, ".asset/cats.png")

    if not os.path.exists(config_path):
        print(f"Config not found at: {config_path}")
        return
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at: {checkpoint_path}")
        return
    if not os.path.exists(image_path):
        print(f"Test image not found at: {image_path}")
        return

    print("Initializing GroundingDINODetector...")
    try:
        # Prompting for 'cat' as the test image is cats.png
        detector = GroundingDINODetector(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            text_prompt="cat",
            box_threshold=0.35,
            text_threshold=0.25
        )
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        return

    print(f"Loading image from {image_path}...")
    try:
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        print(f"Image shape: {image_np.shape}")
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    print("Running detection...")
    try:
        # Pass None for depth_image and agent_state as we only test 2D detection here
        detections = detector.detect(rgb_image=image_np, depth_image=None, agent_state=None)
        
        print(f"\nDetection Results (Count: {len(detections)}):")
        for i, det in enumerate(detections):
            print(f"  [{i}] Label: {det['label']}, Score: {det['score']:.4f}, Box: {det['bbox']}")
            
        if len(detections) > 0:
            print("\nSUCCESS: GroundingDINO successfully detected objects!")
        else:
            print("\nWARNING: No objects detected. This might be due to threshold or model issues.")

    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dino()
