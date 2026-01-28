import sys
import os
from PIL import Image

# 确保能找到项目根目录
sys.path.append("/home/wgy/RL")

from components.environments.thor_env import ThorEnv
from components.detectors.grounding_dino_adapter import GroundingDINODetector

def main():
    # 1. 配置 Grounding DINO 路径
    dino_root = "/home/wgy/GroundingDINO"
    config_file = os.path.join(dino_root, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
    checkpoint_file = os.path.join(dino_root, "weights/groundingdino_swint_ogc.pth")
    
    # 2. 初始化检测器
    # [关键修改] 使用针对厨房场景更全面的提示词，包含地板和墙壁，确保必有检测结果
    text_prompt = "cabinet . fridge . shelf . sink . stove . floor . wall . counter . window . microwave . pot . cup ."
    
    print(f"正在初始化 Grounding DINO 检测器...\n配置: {config_file}\n权重: {checkpoint_file}")
    
    detector = GroundingDINODetector(
        config_path=config_file, 
        checkpoint_path=checkpoint_file,
        text_prompt=text_prompt,
        box_threshold=0.25,   # [关键修改] 降低检测阈值
        text_threshold=0.20
    )

    # 3. 启动环境并注入检测器
    print("正在启动 ThorEnv (Perception Mode)...")
    env = ThorEnv(
        scene_number=1,             # FloorPlan1 是一个小厨房
        render=False,
        additional_images=False,
        use_detector=True,          # 开启检测器模式
        detector=detector,          # 注入真实检测器
        det_score_thr=0.20,         # [关键修改] 降低环境过滤阈值
        fill_position_from_gt=False # 纯视觉模式
    )
    
    print("正在重置环境...")
    obs = env.reset(random_start=True)

    # [调试] 保存当前视角图片，确认机器人是否面对墙壁
    try:
        debug_img = Image.fromarray(obs.info["event"].frame)
        debug_img.save("debug_view.jpg")
        print("📸 已保存当前视角图像到 RL/debug_view.jpg，请检查机器人视野。")
    except Exception as e:
        print(f"保存调试图片失败: {e}")

    ev = obs.info["event"]
    
    # 获取数据用于对比
    gt_objects = ev.metadata.get("_gt_objects", [])
    perceived_objects = ev.metadata.get("objects", [])
    
    gt_n = len(gt_objects)
    seen_n = len(perceived_objects)
    
    print(f"RESET 完成: GT物体数 (参考) = {gt_n}, 感知物体数 (Grounding DINO) = {seen_n}")
    
    # 打印前 5 个检测到的物体
    if seen_n > 0:
        print("\n--- 检测到的物体示例 (Top 5) ---")
        for i, obj in enumerate(perceived_objects[:5]):
            print(f"[{i}] 类型: {obj.get('objectType'):<15} 置信度: {obj.get('score'):.4f}")
            
        # 验证 graph builder 兼容性
        if "score" in perceived_objects[0]:
            print("\n✅ 数据格式验证通过：检测结果包含 'score' 字段。")
        else:
            print("\n❌ 警告：检测结果缺失 'score' 字段！")
    else:
        print("\n⚠️ 警告：当前视角未检测到任何物体！请检查生成的 debug_view.jpg。")

    # 简单运行几步测试稳定性
    print("\n开始步进测试...")
    for t in range(3):
        # 使用 Pass 动作保持原地，观察检测稳定性
        action = env.stop_index 
        obs = env.step(action)
        
        ev = obs.info["event"]
        curr_seen = len(ev.metadata.get("objects", []))
        print(f"STEP {t}: 当前感知物体数={curr_seen} \tReward={obs.reward:.4f}")

    env.close()
    print("\n✅ Perception seam test finished.")

if __name__ == "__main__":
    main()