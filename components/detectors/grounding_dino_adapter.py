import torch
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict

class GroundingDINODetector:
    def __init__(self, config_path, checkpoint_path, text_prompt, box_threshold=0.35, text_threshold=0.25):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Perception] Loading Grounding DINO on {self.device}...")
        
        # 加载模型
        try:
            self.model = load_model(config_path, checkpoint_path)
        except Exception as e:
            print(f"[Error] Failed to load Grounding DINO model: {e}")
            raise e

        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # === 相机内参设置 (Camera Intrinsics) ===
        # 警告：必须根据你 config/env.json 或生成数据集时的设置修改这些值！
        # AI2-THOR/Habitat 默认通常是:
        self.img_width = 300   # 常见默认值，若是640x480请修改
        self.img_height = 300
        self.hfov = 90.0       # 水平视场角
        
        # 根据 FOV 计算焦距 (Pinhole Camera Model)
        self.fx = (self.img_width / 2.0) / np.tan(np.deg2rad(self.hfov / 2.0))
        self.fy = (self.img_height / 2.0) / np.tan(np.deg2rad(self.hfov / 2.0))
        self.cx = self.img_width / 2.0
        self.cy = self.img_height / 2.0

    @torch.no_grad()
    def detect(self, rgb_image, depth_image=None, agent_state=None):
        """
        Args:
            rgb_image: (H, W, 3) numpy array
            depth_image: (H, W) or (H, W, 1) numpy array, 深度值(米)
            agent_state: dict, {'position': {'x':.., 'y':.., 'z':..}, 'rotation': {'x':.., 'y':.., 'z':.., 'w':..}}
        """
        # 1. 图像预处理
        image_pil = Image.fromarray(rgb_image.astype(np.uint8))
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image_pil, None)

        # 2. 模型推理
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        
        # OOM Fix: Clean up
        image_tensor = image_tensor.cpu()
        # torch.cuda.empty_cache() # Removed for speed

        # === [DEBUG] 打印检测结果 ===
        if len(phrases) > 0:
            # 防止刷屏太快，只打印前几个
            # print(f"[DEBUG] DINO detected: {phrases} | Scores: {[round(s.item(), 2) for s in logits]}")
            pass
        else:
            # 只有在非常确信需要看空结果时才打印，否则会刷屏
            # print("[DEBUG] DINO detected NOTHING")
            pass

        detections = []
        H, W, _ = rgb_image.shape

        # 3. 处理检测结果
        for box, score, label in zip(boxes, logits, phrases):
            # 还原 bbox 到像素坐标
            box = box * torch.Tensor([W, H, W, H])
            cx, cy, w, h = box.numpy()
            
            # 这里的 bbox 格式是 [min_x, min_y, max_x, max_y]
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2

            # === 核心：2D -> 3D 投影 ===
            world_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            if depth_image is not None and agent_state is not None:
                world_pos = self._project_to_3d(cx, cy, depth_image, agent_state)
            
            # 构造符合 LocalGraphBuilder 接口的数据
            detections.append({
                "label": label,
                "score": float(score),
                "bbox": [x1, y1, x2, y2],
                "object_id": f"{label}_{np.random.randint(10000)}", # 临时ID
                "position": world_pos
            })
            
        return detections

    def _project_to_3d(self, u, v, depth_img, agent_state):
        # 1. 读取深度
        u_int, v_int = int(u), int(v)
        v_int = max(0, min(v_int, self.img_height - 1))
        u_int = max(0, min(u_int, self.img_width - 1))
        
        d = depth_img[v_int, u_int]
        if isinstance(d, np.ndarray): d = d.item()
        
        # 简单过滤无效深度
        if d <= 0.01 or d > 10.0:
            return {'x': 0.0, 'y': 0.0, 'z': 0.0}

        # 2. 图像坐标 -> 相机坐标 (Camera Space)
        # 假设 Habitat 相机: +X 右, +Y 上, -Z 前 (或者 Z 是深度)
        # 根据经验通常 Z 是深度方向
        z_c = d  # 有些坐标系是 -d，需要根据你的具体环境测试
        x_c = (u - self.cx) * d / self.fx
        y_c = -(v - self.cy) * d / self.fy # 图像Y向下，世界Y向上，通常取反

        camera_point = np.array([x_c, y_c, z_c])

        # 3. 相机坐标 -> 世界坐标 (World Space)
        # AI2-THOR Agent Position 通常在地面 (y=0 附近)，但相机有高度
        # 我们假设 agent_pos 是相机位置，如果不是，需要加 offset (但这取决于 upstream 传什么)
        # 通常 event.metadata['agent']['position'] 是 floor level
        # event.metadata['agent']['cameraHorizon'] 是俯仰角
        
        agent_pos = np.array([
            agent_state['position']['x'],
            agent_state['position']['y'],
            agent_state['position']['z']
        ])
        
        # 处理旋转 (支持 Quaternion 或 Euler)
        rot = agent_state['rotation']
        r = None
        
        if isinstance(rot, dict):
            if 'w' in rot:
                # Quaternion [x, y, z, w]
                r = R.from_quat([rot['x'], rot['y'], rot['z'], rot['w']])
            else:
                # AI2-THOR default: Euler degrees {x, y, z}. Usually only Y (yaw) changes for nav.
                # standard order for 'rotation' dict in ai2thor is likely (x, y, z) but usually y is yaw.
                # Assuming 'y' is yaw (rotation around vertical axis). 
                # 注意：AI2-THOR 坐标系转换可能需要更精细处理，这里做基础兼容
                r = R.from_euler('y', rot['y'], degrees=True)
        else:
            # 假设是 list/array
            if len(rot) == 4:
                r = R.from_quat(rot)
            else:
                 # 假设 Euler [x, y, z]
                r = R.from_euler('xyz', rot, degrees=True)

        if r is None:
             return {'x': 0.0, 'y': 0.0, 'z': 0.0}

        # 考虑相机俯仰角 (Camera Horizon / Pitch)
        # AI2-THOR: positive horizon looks down? No, usually positive is down/up?
        # Check docs: "cameraHorizon": 0 is straight. 30 is looking down 30 degrees?
        # 简单起见，如果不为0，叠加一个 x 轴旋转
        if 'cameraHorizon' in agent_state and abs(agent_state['cameraHorizon']) > 1e-3:
             r_pitch = R.from_euler('x', agent_state['cameraHorizon'], degrees=True)
             r = r * r_pitch

        # 应用旋转
        # 这里的相乘顺序取决于他是 局部旋转 还是 全局旋转
        # world_vec = R_agent * vec_camera
        world_point_relative = r.apply(camera_point)
        
        # 加上 Agent 位置 (注意：这里忽略了相机相对于 Agent 的高度偏移，如果 agent_pos 是脚底板)
        # 如果需要更精确，应该加 camera height，通常是 0.675 或类似
        # 临时修复：如果是 'y' ~ 0.9 这种，可能是脚底。
        # 但 depth 投影出来的 y_c 在相机系下。
        world_point = world_point_relative + agent_pos

        return {'x': world_point[0], 'y': world_point[1], 'z': world_point[2]}