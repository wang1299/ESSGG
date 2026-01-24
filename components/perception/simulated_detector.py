import random
from typing import List, Dict


class SimulatedDetector:
    """Simple simulated detector that converts AI2-THOR metadata into detection-like outputs.

    Behavior:
    - Uses objects from metadata to create detections.
    - Supports configurable false negative / false positive rates and score noise.
    - Adds an approx 2D bbox derived from object position for compatibility with downstream code.
    This is intentionally lightweight so it can be used for experiments without adding heavy deps.
    """

    def __init__(self, miss_rate: float = 0.1, false_pos_rate: float = 0.05, score_noise: float = 0.05, rng_seed: int = None):
        self.miss_rate = float(miss_rate)
        self.false_pos_rate = float(false_pos_rate)
        self.score_noise = float(score_noise)
        if rng_seed is not None:
            random.seed(rng_seed)

    def detect(self, frame, metadata: Dict) -> List[Dict]:
        """Return a list of detection dicts given a frame and AI2-THOR metadata.

        Each detection has fields: 'class', 'score', 'bbox' (x1,y1,x2,y2 in pixels or None), 'object_id'(optional), 'position' (dict with x,y,z)
        We build detections from metadata['objects'] if available.
        """
        detections = []

        objects = metadata.get("objects", []) if metadata else []

        # Create detections from true objects (allow missing ones according to miss_rate)
        for obj in objects:
            if random.random() < self.miss_rate:
                # simulate missed detection
                continue

            det = {
                "class": obj.get("objectType", "unknown"),
                "score": max(0.0, min(1.0, (obj.get("visible", False) and 0.9 or 0.2) + random.uniform(-self.score_noise, self.score_noise))),
                "bbox": None,  # we don't compute pixel bbox here (optional)
                "object_id": obj.get("objectId"),
                "position": obj.get("position", None),
            }

            detections.append(det)

        # Add a few false positives
        if random.random() < self.false_pos_rate:
            # create one synthetic false positive per trigger
            fp = {"class": "unknown_fp", "score": 0.4 + random.random() * 0.3, "bbox": None, "object_id": None, "position": None}
            detections.append(fp)

        return detections
