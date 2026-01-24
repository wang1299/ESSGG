import os
import pickle
from glob import glob


def list_all_pkl_files(data_dir):
    pattern = os.path.join(data_dir, "**", "*.pkl")
    return sorted(glob(pattern, recursive=True))


if __name__ == "__main__":
    data_dir = "../../components/data/il_dataset_mrunmai"
    for file_path in list_all_pkl_files(data_dir):
        with open(file_path, "rb") as f:
            ep = pickle.load(f)
