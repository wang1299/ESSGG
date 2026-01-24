"""Small demo showing how simulated detector output is converted to a SceneGraph.

This demo does not require AI2-THOR runtime — it uses GT graphs to synthesize metadata,
then runs the SimulatedDetector and LocalSceneGraphBuilder::build_from_detections.
"""
import json
import os

from components.perception.simulated_detector import SimulatedDetector
from components.graph.local_graph_builder import LocalSceneGraphBuilder


def synthesize_metadata_from_gt(gt_json_path):
    with open(gt_json_path, "r") as f:
        data = json.load(f)

    objects = []
    for n in data.get("nodes", []):
        pos = n.get("position", (0.0, 0.0, 0.0))
        obj = {
            "visible": True,
            "objectId": n.get("object_id"),
            "objectType": n.get("name"),
            "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
            "distance": 1.0,
            "axisAlignedBoundingBox": {"size": {"x": 0.5, "y": 0.5, "z": 0.5}},
        }
        objects.append(obj)
    return {"objects": objects}


def main():
    base = os.path.join(os.path.dirname(__file__), "..", "data", "gt_graphs")
    sample = os.path.join(base, "FloorPlan4.json")
    if not os.path.exists(sample):
        print("No sample GT file found at", sample)
        return

    metadata = synthesize_metadata_from_gt(sample)
    det = SimulatedDetector(miss_rate=0.15, false_pos_rate=0.1, score_noise=0.1, rng_seed=123)
    detections = det.detect(frame=None, metadata=metadata)

    print(f"Simulated {len(detections)} detections (including false positives):")
    for d in detections[:10]:
        print(d)

    builder = LocalSceneGraphBuilder()
    sg = builder.build_from_detections(detections)
    print("Constructed SceneGraph with nodes:")
    for k, v in sg.nodes.items():
        print(k, v.name, round(v.visibility, 3), v.position)


if __name__ == "__main__":
    main()
