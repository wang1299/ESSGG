import matplotlib
# 【关键】服务器没有屏幕，必须设置为 'Agg' 模式，否则 plt.show() 或绘图会报错
matplotlib.use('Agg') 

import habitat_sim
import numpy as np
import matplotlib.pyplot as plt
import os

# === 配置路径 ===
# 假设脚本和 .glb 文件在同一目录下
SCENE_FILE = "102344193.scene_instance.glb" 
OUTPUT_NAME = "hssd_topdown_map.png"

def main():
    if not os.path.exists(SCENE_FILE):
        print(f"错误：找不到文件 {SCENE_FILE}，请确认你已经下载了它！")
        return

    # 1. 配置模拟器
    print("正在初始化模拟器...")
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE_FILE
    sim_cfg.enable_physics = True
    
    # 就算不渲染画面，也需要配置一个 Agent 占位
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    try:
        sim = habitat_sim.Simulator(cfg)
    except Exception as e:
        print(f"\n初始化失败: {e}")
        print("提示：如果是 Display 相关错误，请检查是否安装了 habitat-sim-headless 版本")
        return

    # 2. 计算 NavMesh (导航网格)
    # HSSD 的 glb 文件只有模型，必须现场计算哪里能走
    print("正在计算 NavMesh (可能需要几秒钟)...")
    pf = sim.pathfinder
    nav_settings = habitat_sim.NavMeshSettings()
    nav_settings.set_defaults()
    nav_settings.agent_height = 1.5
    nav_settings.agent_radius = 0.1
    
    success = sim.recompute_navmesh(pf, nav_settings)
    if success:
        print(f"NavMesh 计算成功！可通行面积: {pf.navigable_area:.2f} m²")
    else:
        print("NavMesh 计算失败，无法生成地图。")
        sim.close()
        return

    # 3. 生成俯视图
    print("正在生成俯视图...")
    # height: 切片高度。HSSD 地板通常在 0，我们切 0.15m 的位置，既能看到墙，也能避开地板噪点
    # meters_per_pixel: 分辨率，越小越清晰
    td_map = pf.get_topdown_view(meters_per_pixel=0.02, height=0.15)
    
    if not np.any(td_map):
        print("警告：生成的地图是全黑的，可能是切片高度不对。")
    else:
        # 4. 保存图片
        plt.figure(figsize=(12, 12))
        # 翻转颜色：cmap='gray' 时 True(可通行)是白色，False(墙)是黑色
        plt.imshow(td_map, cmap="gray", origin="lower")
        plt.title(f"HSSD Scene: {SCENE_FILE}")
        plt.axis("off")
        
        plt.savefig(OUTPUT_NAME, bbox_inches='tight', pad_inches=0.1)
        print(f"\n成功！地图已保存为: {os.path.abspath(OUTPUT_NAME)}")

    sim.close()

if __name__ == "__main__":
    main()