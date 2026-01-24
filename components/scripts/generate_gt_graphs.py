import os

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from tqdm import tqdm

from components.graph.gt_graph import GTGraph
from components.graph.local_graph_builder import LocalSceneGraphBuilder


def generate_gt_scene_graphs(num_floorplans: int = 29, floorplans: list = None):
    save_dir = os.path.join(os.path.dirname(__file__), "..", "data", "gt_graphs")
    os.makedirs(save_dir, exist_ok=True)
    if floorplans is None:
        floorplans = [f"FloorPlan{i}" for i in list(range(1, 6)) + list(range(7, 8)) + list(range(9, 31))][:num_floorplans]

    controller = Controller(platform=CloudRendering, visibilityDistance=50.0)
    builder = LocalSceneGraphBuilder()

    rotations = [0, 90, 180, 270]

    for scene in floorplans:
        print(f"\nCreating GT scene graph for: {scene}")
        controller.reset(scene=scene)

        # Get reachable positions
        reachable_positions = controller.step("GetReachablePositions").metadata["actionReturn"]

        total_steps = len(reachable_positions) * len(rotations)

        with tqdm(total=total_steps, desc=f"Exploring {scene}") as pbar:
            gt_graph = GTGraph()
            for pos in reachable_positions:
                for rot in rotations:
                    controller.step(
                        action="Teleport", position=pos, rotation={"x": 0, "y": rot, "z": 0}, horizon=0, standing=True, forceAction=True
                    )

                    vp = {"position": {"x": round(pos["x"], 2), "z": round(pos["z"], 2)}, "rotation": rot}
                    viewpoint = f"{vp['position']}_{vp['rotation']}"

                    event = controller.step("Pass")
                    if scene == "FloorPlan1":
                        objects = event.metadata["objects"]
                        for obj in objects:
                            name = obj.get("name", "")
                            visible = obj.get("visible", False)

                            if name.startswith("DishSponge") or name.startswith("Plate"):  # Those two objects are too hidden for GT graph
                                event.metadata["objects"].remove(obj)
                    local_sg = builder.build_from_metadata(event.metadata)
                    gt_graph.add_local_sg(local_sg)
                    gt_graph.add_viewpoint(viewpoint, local_sg.nodes)

                    pbar.update(1)

        # Save graph to file
        output_path = os.path.join(save_dir, f"{scene}.json")
        gt_graph.save_to_file(output_path)

        print(f"✅ Saved: {output_path}")

    controller.stop()


if __name__ == "__main__":
    generate_gt_scene_graphs(floorplans=["FloorPlan13", "FloorPlan14", "FloorPlan15", "FloorPlan16", "FloorPlan17"])
    # generate_gt_scene_graphs(num_floorplans=1)
