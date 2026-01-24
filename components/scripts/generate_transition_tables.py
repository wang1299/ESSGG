import os
import pickle
import platform

from tqdm import tqdm

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering


def generate_transition_tables(
    num_floorplans: int = 29, floorplans: list = None, transition_tables_path: str = "../data/transition_tables", grid_size: float = 0.25
):
    """
    Precompute and save transition tables for given AI2-THOR floorplans.
    Each table maps (x, z, rotation) -> event (or None if unreachable).
    """
    # Ensure output directory exists
    os.makedirs(transition_tables_path, exist_ok=True)

    if floorplans is None:
        floorplans = [f"FloorPlan{i}" for i in list(range(1, 6)) + list(range(7, 8)) + list(range(9, 31))][:num_floorplans]

    # Initialize AI2-THOR controller in cloud rendering mode
    if platform.system() == "Linux":
        controller = Controller(platform=CloudRendering, visibilityDistance=50.0)
    else:
        controller = Controller(visibilityDistance=50.0)
    rotations = [0, 90, 180, 270]

    for scene in floorplans:
        print(f"\nGenerating transition table for: {scene}")
        controller.reset(scene=scene)

        # Query reachable positions
        reachable = controller.step("GetReachablePositions").metadata["actionReturn"]
        total_tasks = len(reachable) * len(rotations)

        mapping = {}
        # Iterate over all reachable positions and orientations
        with tqdm(total=total_tasks, desc=f"Exploring {scene}") as pbar:
            for pos in reachable:
                x = round(pos["x"], 2)
                z = round(pos["z"], 2)
                for rot in rotations:
                    # Teleport agent to exact position and rotation
                    controller.step(
                        action="Teleport",
                        position={"x": pos["x"], "y": pos.get("y", 0), "z": pos["z"]},
                        rotation={"x": 0, "y": rot, "z": 0},
                        horizon=0,
                        standing=True,
                        forceAction=True,
                    )
                    # Perform a pass action to get the event
                    event = controller.step("Pass")
                    # Store event in mapping (key: world coords and rotation)
                    mapping[(x, z, rot)] = event
                    pbar.update(1)

        # Save mapping (and grid_size) to disk
        output_path = os.path.join(transition_tables_path, f"{scene}.pkl")
        data = {"table": mapping, "grid_size": grid_size}
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ Saved transition table: {output_path}")

    controller.stop()


if __name__ == "__main__":
    # generate_transition_tables(floorplans=["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan7", "FloorPlan9", "FloorPlan10", "FloorPlan11", "FloorPlan12", "FloorPlan13", "FloorPlan14", "FloorPlan15"])
    generate_transition_tables(num_floorplans=13)
