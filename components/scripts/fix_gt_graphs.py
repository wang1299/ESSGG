import glob
import json
import os


def fix_id_string(s):
    if isinstance(s, str) and "," in s:
        return s.replace(",", ".")
    return s


def fix_list_of_ids(lst):
    if lst is None:
        return None
    return [fix_id_string(item) for item in lst]


def fix_node(node):
    changed = False

    # Fix object_id
    if "object_id" in node:
        new_id = fix_id_string(node["object_id"])
        if new_id != node["object_id"]:
            node["object_id"] = new_id
            changed = True

    # Fix entries in properties
    props = node.get("properties", {})

    if "parentReceptacles" in props:
        original = props["parentReceptacles"]
        fixed = fix_list_of_ids(original)
        if fixed != original:
            props["parentReceptacles"] = fixed
            changed = True

    if "receptacleObjectIds" in props:
        original = props["receptacleObjectIds"]
        fixed = fix_list_of_ids(original)
        if fixed != original:
            props["receptacleObjectIds"] = fixed
            changed = True

    return changed


def fix_edges(edges):
    changed = False
    for edge in edges:
        new_source = fix_id_string(edge["source"])
        new_target = fix_id_string(edge["target"])
        if new_source != edge["source"]:
            edge["source"] = new_source
            changed = True
        if new_target != edge["target"]:
            edge["target"] = new_target
            changed = True
    return changed


def fix_gt_graph(file_path):
    with open(file_path, "r") as f:
        graph = json.load(f)

    changed = False
    for node in graph.get("nodes", []):
        changed = fix_node(node) or changed

    if "edges" in graph:
        changed = fix_edges(graph["edges"]) or changed

    if changed:
        with open(file_path, "w") as f:
            json.dump(graph, f, indent=2)
        print(f"✔ Updated: {file_path}")


def fix_all_gt_graphs(gt_path="data/gt_graphs"):
    json_files = glob.glob(os.path.join(gt_path, "FloorPlan*.json"))
    print(f"Found {len(json_files)} GT files.")
    for path in json_files:
        fix_gt_graph(path)


if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__), "..", "data", "gt_graphs")
    fix_all_gt_graphs(base_path)
