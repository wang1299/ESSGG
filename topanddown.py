import matplotlib
matplotlib.use('Agg') # 服务器模式
import habitat_sim
import os
import matplotlib.pyplot as plt
import numpy as np

# === 配置 ===
HSSD_ROOT = "/home/wgy/RL/habitat_data" # 你的路径
DATASET_CONFIG = os.path.join(HSSD_ROOT, "hssd-hab.scene_dataset_config.json")
OUTPUT_DIR = "hssd_all_maps"
# ===========

def main():
    if not os.path.exists(DATASET_CONFIG):
        print("错误：找不到配置文件")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 初始化一个空模拟器来读取场景列表
    # (注意：这里必须先创建对象，再赋值)
    cfg_init = habitat_sim.SimulatorConfiguration()
    cfg_init.scene_dataset_config_file = DATASET_CONFIG
    cfg_init.scene_id = "NONE" # 占位符
    
    try:
        # 这里也不要直接在 Configuration 里传参，拆开写更稳
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg_init, [agent_cfg]))
    except Exception as e:
        print(f"模拟器初始化失败: {e}")
        return

    # 获取所有场景 ID
    all_scenes = sim.metadata_mediator.get_scene_handles()
    # 你的日志显示 169 个，这很正常（有些场景可能被过滤掉了，不影响使用）
    print(f"共发现 {len(all_scenes)} 个场景，准备开始批量生成...")
    
    # 2. 循环处理每一个场景
    for i, scene_handle in enumerate(all_scenes):
        scene_name = os.path.basename(scene_handle).split('.')[0]
        print(f"[{i+1}/{len(all_scenes)}] 正在处理: {scene_name} ...")
        
        # === 核心修复部分开始 ===
        # 错误写法: habitat_sim.SimulatorConfiguration(scene_id=...)
        # 正确写法: 先实例化，再赋值
        
        new_sim_cfg = habitat_sim.SimulatorConfiguration()
        new_sim_cfg.scene_dataset_config_file = DATASET_CONFIG
        new_sim_cfg.scene_id = scene_handle
        new_sim_cfg.enable_physics = True
        
        new_agent_cfg = habitat_sim.agent.AgentConfiguration()
        
        new_cfg = habitat_sim.Configuration(new_sim_cfg, [new_agent_cfg])
        
        sim.reconfigure(new_cfg)
        # === 核心修复部分结束 ===

        # 计算 NavMesh
        pf = sim.pathfinder
        nav_settings = habitat_sim.NavMeshSettings()
        nav_settings.set_defaults()
        nav_settings.include_static_objects = True # 包含家具
        # 强制重新计算，防止用缓存
        sim.recompute_navmesh(pf, nav_settings)
        
        # 生成地图 (切片高度 0.5米)
        td_map = pf.get_topdown_view(meters_per_pixel=0.05, height=0.5)
        
        # 保存图片
        if np.any(td_map):
            plt.figure(figsize=(10, 10))
            plt.imshow(td_map, cmap="gray", origin="lower")
            plt.axis("off")
            plt.title(scene_name)
            
            save_path = os.path.join(OUTPUT_DIR, f"{scene_name}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            print(f"   警告: 场景 {scene_name} 生成全黑，跳过。")

    sim.close()
    print("全部完成！")

if __name__ == "__main__":
    main()