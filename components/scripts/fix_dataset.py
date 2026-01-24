import glob
import os
import pickle


def fix_id_string(s):
    """Replace commas with dots in an object identifier string."""
    if isinstance(s, str) and "," in s:
        return s.replace(",", ".")
    return s


def fix_edges(edge_list):
    """Fix source and target strings in a list of edge dictionaries."""
    for edge in edge_list:
        edge["source"] = fix_id_string(edge["source"])
        edge["target"] = fix_id_string(edge["target"])


def fix_graph(graph_dict):
    """Fix one SG dict by updating edge source/target."""
    if "edges" in graph_dict:
        fix_edges(graph_dict["edges"])


def fix_dataset(base_path="data/il_dataset"):
    pkl_files = glob.glob(os.path.join(base_path, "FloorPlan*", "*.pkl"), recursive=True)
    print(f"Found {len(pkl_files)} .pkl files")

    for file_path in pkl_files:
        with open(file_path, "rb") as f:
            sequence = pickle.load(f)

        changed = False

        for sample in sequence:
            if "lssg" in sample:
                fix_graph(sample["lssg"])
                changed = True
            if "gssg" in sample:
                fix_graph(sample["gssg"])
                changed = True

        if changed:
            with open(file_path, "wb") as f:
                pickle.dump(sequence, f)
            print(f"✔ Updated: {file_path}")


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "data", "il_dataset")
    fix_dataset(path)
