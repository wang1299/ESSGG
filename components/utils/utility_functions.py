import json
import torch
import random
import numpy as np


def print_dict(d, indent=0):
    """
    Recursively prints a nested dictionary with indentation.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_dict(value, indent + 2)
        else:
            print(" " * indent + f"{key}: {value}")


def read_config(file_name, use_print=True) -> dict:
    with open(file_name, "r", encoding="utf-8") as file:
        data = json.load(file)
    if use_print:
        print(f"\nLoaded {file_name}:")
        print_dict(data, indent=2)
    return data


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_seeds(num_seeds=30):
    random.seed(42)
    return [random.randint(0, 1000000) for _ in range(num_seeds)]


def log_mem(tag):
    import torch, psutil, os

    torch.cuda.synchronize()
    print(f"\n=== {tag} ===")
    print(f"CUDA allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
    print(f"CUDA reserved : {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")
    print(f"CPU RSS       : {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.1f} MB")


def log_summary(tag):
    import torch

    torch.cuda.synchronize()
    print(f"\n==== MEMORY SUMMARY: {tag} ====")
    print(torch.cuda.memory_summary())
