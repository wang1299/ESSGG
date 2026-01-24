import numpy as np
import matplotlib.pyplot as plt

UNKNOWN = 0
BLOCKED = -1

BLOCKING_CLASSES = {"CounterTop", "Fridge", "Sofa", "Table", "Cabinet", "Wall"}


class ExplorationMap:

    def __init__(self, grid_size: float, map_width: int, map_height: int, origin: tuple, vision_range: float = 10.0):
        self.grid_size = grid_size
        self.origin = origin
        self.visited = np.zeros((map_height, map_width), dtype=np.uint8)
        self.blocked = np.zeros((map_height, map_width), dtype=np.uint8)
        self.map = np.zeros((map_height, map_width), dtype=np.float32)
        self.agent_positions = []
        self.discovered_objects = {}
        self.vision_range = vision_range

    def reset(self):
        self.map.fill(0)
        self.agent_positions.clear()

    def world_to_map(self, x, z):
        dx = x - self.origin[0]
        dz = z - self.origin[1]

        i = self.map.shape[0] - 1 - int(dz / self.grid_size)
        j = int(dx / self.grid_size)

        return i, j

    def update_from_event(self, event):
        pos = event.metadata["agent"]["position"]
        x, z = pos["x"], pos["z"]
        i, j = self.world_to_map(x, z)

        if 0 <= i < self.visited.shape[0] and 0 <= j < self.visited.shape[1]:
            self.visited[i, j] = 1
            self.agent_positions.append((i, j))

        self.update_blocked_from_event(event)

    def update_blocked_from_event(self, event):
        """
        Marks all visible objects (regardless of type) as blocked, if within vision_range.
        """
        objects = event.metadata.get("objects", [])
        agent_pos = event.metadata["agent"]["position"]
        agent_x, agent_z = agent_pos["x"], agent_pos["z"]

        for obj in objects:
            if not obj.get("visible", False):
                continue

            pos = obj.get("position", {})
            x, z = pos.get("x"), pos.get("z")
            if x is None or z is None:
                continue

            dist = np.sqrt((agent_x - x) ** 2 + (agent_z - z) ** 2)
            if dist > self.vision_range:
                continue

            i, j = self.world_to_map(x, z)
            if 0 <= i < self.blocked.shape[0] and 0 <= j < self.blocked.shape[1]:
                self.blocked[i, j] = 1

    def update_occupancy_from_event(self, event):
        """
        Marks only nearby visible blocking objects as OCCUPIED.
        """
        objects = event.metadata.get("objects", [])
        agent_pos = event.metadata["agent"]["position"]
        agent_x, agent_z = agent_pos["x"], agent_pos["z"]

        for obj in objects:
            if not obj.get("visible", False):
                continue

            obj_type = obj.get("objectType")
            if obj_type not in BLOCKING_CLASSES:
                continue

            pos = obj.get("position", {})
            x, z = pos.get("x"), pos.get("z")
            if x is None or z is None:
                continue

            dist = np.sqrt((agent_x - x) ** 2 + (agent_z - z) ** 2)
            if dist > self.vision_range:
                continue

            i, j = self.world_to_map(x, z)
            if not (0 <= i < self.occupancy.shape[0] and 0 <= j < self.occupancy.shape[1]):
                continue

            # print(f"{obj_type}: found at {dist:.2f}m, marking cell ({i},{j}) as OCCUPIED")
            self.occupancy[i, j] = BLOCKED  # -1 = OCCUPIED

    def mark_blocked_in_front(self, event):
        agent = event.metadata["agent"]
        x, z = agent["position"]["x"], agent["position"]["z"]
        rotation = int(round(agent["rotation"]["y"])) % 360

        dx, dz = 0, 0
        if rotation == 0:
            dz = 1
        elif rotation == 90:
            dx = 1
        elif rotation == 180:
            dz = -1
        elif rotation == 270:
            dx = -1
        else:
            return

        x_front = x + dx * self.grid_size
        z_front = z + dz * self.grid_size

        i, j = self.world_to_map(x_front, z_front)
        if 0 <= i < self.blocked.shape[0] and 0 <= j < self.blocked.shape[1]:
            self.blocked[i, j] = 1

    def mark_discoveries(self, event, global_graph):
        """
        Checks which visible objects are newly discovered and stores their type in map cell.
        """
        objects = event.metadata.get("objects", [])
        for obj in objects:
            if not obj.get("visible", False):
                continue

            obj_id = obj.get("objectId")
            obj_type = obj.get("objectType")
            pos = obj.get("position", {})
            x, z = pos.get("x"), pos.get("z")
            if x is None or z is None:
                continue

            # ccheck if already seen
            if obj_id in global_graph.nodes:
                continue

            i, j = self.world_to_map(x, z)
            if 0 <= i < self.map.shape[0] and 0 <= j < self.map.shape[1]:
                if (i, j) not in self.discovered_objects:
                    self.discovered_objects[(i, j)] = set()
                self.discovered_objects[(i, j)].add(obj_type)

    def progress(self):
        visited = np.sum(self.map > 0.1)
        total = self.map.size
        return visited / total

    def render(self, show=True, save_path=None):
        img = np.copy(self.map)
        for i, j in self.agent_positions:
            img[i, j] = 2.0

        cmap = plt.cm.get_cmap("Greys", 3)
        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap=cmap, origin="upper")
        plt.title("Exploration Map")
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()

    def print_ascii(self):
        """
        ASCII grid: ⬛ = blocked, 🟥 = visited, ⬜ = unknown
        """
        for i in range(self.visited.shape[0]):
            line = ""
            for j in range(self.visited.shape[1]):
                if self.blocked[i, j] == 1:
                    line += "⬛"
                elif self.visited[i, j] == 1:
                    line += "🟥"
                else:
                    line += "⬜"
            print(line)

    def print_discoveries(self):
        for (i, j), object_types in self.discovered_objects.items():
            types = ", ".join(sorted(object_types))
            print(f"At ({i}, {j}) discovered {len(object_types)} item(s): {types}")
