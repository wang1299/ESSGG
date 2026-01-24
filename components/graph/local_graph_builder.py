import numpy as np
import math
from components.graph.RelationExtractor import RelationExtractor
from components.graph.scene_graph import Node, SceneGraph

class LocalSceneGraphBuilder:
    def __init__(self):
        self.relation_extractor = RelationExtractor()

    def build_from_metadata(self, metadata: dict) -> SceneGraph:
        sg = SceneGraph()
        objects = metadata.get("objects", [])

        for obj in objects:
            # 在某些情况下（如 GT 模式），我们可能想过滤不可见物体
            # 但检测器输出的物体通常默认是可见的
            if obj.get("visible", True): 
                object_id = obj.get("objectId", f"obj_{np.random.randint(100000)}")
                
                # 修复区域设置问题：如果 objectId 中的数字部分用逗号分隔，将其转换为点
                if "," in str(object_id) and "." not in str(object_id):
                    object_id = str(object_id).replace(",", ".")
                
                # 获取位置，兼容检测器可能不返回 position 的情况
                pos_dict = obj.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
                if pos_dict is None: 
                    pos_dict = {"x": 0.0, "y": 0.0, "z": 0.0}
                position = tuple(pos_dict.values())

                node = Node(
                    object_id=object_id,
                    name=obj.get("objectType", "Unknown"),
                    position=position,
                    visibility=self.compute_soft_visibility(obj),
                    properties={k: v for k, v in obj.items() if k not in ("objectId", "objectType", "position", "visible", "score", "distance")}
                )

                sg.add_node(node)

        # 仅在有完整元数据支持时提取关系
        # 如果是纯视觉检测，这里可能需要其他的关系提取逻辑
        edges = self.relation_extractor.extract_relations(objects)
        for edge in edges:
            sg.add_edge(edge)

        return sg

    def build_from_detections(self, detections: list) -> SceneGraph:
        """
        从检测器输出构建场景图的辅助方法。
        """
        sg = SceneGraph()

        for i, det in enumerate(detections):
            score = float(det.get("score", 0.0))
            if score <= 0.0:
                continue

            object_id = det.get("object_id") or f"det_{i}"
            name = det.get("class", "unknown")

            position_dict = det.get("position")
            position = tuple(position_dict.values()) if position_dict else (0.0, 0.0, 0.0)

            visibility = float(min(1.0, max(0.0, score)))

            node = Node(object_id=object_id, name=name, position=position, visibility=visibility, properties={"score": score})
            sg.add_node(node)

        return sg

    def compute_soft_visibility(self, obj):
        """
        计算物体的软可见度 (Soft Visibility)。
        逻辑升级：
        1. 真实感知模式：如果存在 'score' (置信度)，直接将其作为可见度。
        2. GT 模式：如果存在 'distance'，使用论文公式 13 基于距离计算。
        """
        
        # [新增] 1. 优先使用检测器的置信度 (Grounding DINO)
        if "score" in obj:
            # 确保返回浮点数，且在 0~1 之间
            return float(obj["score"])

        # [保留] 2. 原论文逻辑：基于距离的启发式计算 (公式 13)
        if "distance" in obj:
            d = obj["distance"]
            
            # 获取物体最大尺寸 s_max (用于公式 14)
            s_max = 0.5 # 默认参考值
            # 尝试从 objectBounds 提取尺寸
            if "objectBounds" in obj and isinstance(obj["objectBounds"], dict):
                bounds = obj["objectBounds"]
                if 'max' in bounds and 'min' in bounds:
                    dims = [
                        bounds['max']['x'] - bounds['min']['x'],
                        bounds['max']['y'] - bounds['min']['y'],
                        bounds['max']['z'] - bounds['min']['z']
                    ]
                    s_max = max(dims)

            # 论文 6.3.1 定义的超参数 
            sigma = 1.0
            c_base = 3.5
            k_size = 1.5
            s_ref = 0.5

            # 论文公式 14: 动态中心 c(s_max) [cite: 2798]
            c_s_max = c_base + k_size * (s_max - s_ref)

            # 论文公式 13: Sigmoid 计算 [cite: 2795]
            try:
                v_soft = 1.0 / (1.0 + math.exp(sigma * (d - c_s_max)))
            except OverflowError:
                v_soft = 0.0 # 距离过远
            return v_soft
            
        # [兜底] 如果什么都没有，默认设为可见 (1.0)，防止报错中断训练
        return 1.0