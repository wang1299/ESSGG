import json
from argparse import ArgumentParser
from pathlib import Path

import torch

from components.agents.imitation_agent import ImitationAgent
from ImitationLearning.runner.il_train_runner import ILTrainRunner
from ImitationLearning.dataset.il_dataset import ImitationLearningDataset


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--weights_save_folder", type=str, default=None, help="Directory for saving model weights. If None, a default folder is used."
    )
    parser.add_argument(
        "--checkpoint_path_load", type=str, default=None, help="Path to load model weights from. If None, training starts from scratch."
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    args = parser.parse_args()

    # --- Define all relevant paths ---
    project_root = Path(__file__).parent.parent.resolve()
    config_path = project_root / "config" / "navigation.json"
    data_dir = project_root / "components" / "data" / "il_dataset"
    default_weights_folder = project_root / "components" / "data" / "model_weights"
    weights_save_folder = Path(args.weights_save_folder) if args.weights_save_folder else default_weights_folder
    # args.checkpoint_path_load = (
    #     "/home/kueblero/PycharmProjects/embodied-scene-graph-navigation/data/model_weights/2025-06-03_14-31-23_imitation_agent_epoch100.pth"
    # )

    # --- Load navigation config ---
    with open(config_path, "r") as f:
        navigation_config = json.load(f)
    print("\nLoaded navigation config:")
    for k, v in navigation_config.items():
        print(f"  {k}: {v}")

    # --- Prepare dataset and agent ---
    dataset = ImitationLearningDataset(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = ImitationAgent(navigation_config=navigation_config, num_actions=dataset.num_actions, device=device)

    # --- Optionally load weights from checkpoint ---
    if args.checkpoint_path_load:
        checkpoint_path = Path(args.checkpoint_path_load)
        if checkpoint_path.exists():
            agent.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
            print(f"Weights loaded from: {checkpoint_path}")
        else:
            print(f"WARNING: Checkpoint not found at {checkpoint_path}. Training will start from scratch.")

    # --- Start training ---
    print(f"\n[INFO] Starting training for {args.epochs} epochs")
    print(f"[INFO] Weights will be saved in: {weights_save_folder}\n")

    runner = ILTrainRunner(agent, dataset, device=device, lr=0.0001, batch_size=args.batch_size)
    runner.run(num_epochs=args.epochs, save_folder=str(weights_save_folder))


if __name__ == "__main__":
    main()
