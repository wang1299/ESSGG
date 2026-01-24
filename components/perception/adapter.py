# components/perception/adapter.py
import hashlib

def _make_id(cls, bbox, frame_idx=0):
    s = f"{cls}_{bbox}_{frame_idx}".encode("utf-8")
    return hashlib.md5(s).hexdigest()[:12]

def dets_to_thor_objects(dets, score_thr=0.3, frame_idx=0):
    """
    Convert detector outputs (class/score/bbox/position/object_id) into
    THOR-like object dicts used by local_graph_builder / RelationExtractor / exploration_map.
    """
    objs = []
    for i, d in enumerate(dets):
        score = float(d.get("score", 0.0))
        if score < score_thr:
            continue

        cls = d.get("class") or d.get("name") or "unknown"
        bbox = d.get("bbox", None)

        obj_id = d.get("object_id") or d.get("objectId")
        if obj_id is None:
            obj_id = _make_id(cls, bbox, frame_idx)

        # position: ideally fill with depth->3D later; for now allow None
        pos = d.get("position", None)  # can be dict {x,y,z} or None

        objs.append({
            "visible": True,           # visibility由 detector score 决定
            "objectId": obj_id,
            "objectType": cls,
            "position": pos,
            "bbox": bbox,
            "score": score,
        })
    return objs
