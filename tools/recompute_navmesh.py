import habitat_sim
import os
import glob

def recompute_and_save_navmesh(scene_path, out_path, agent_radius=0.17, agent_height=1.5, agent_max_climb=0.1):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    
    with habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [])) as sim:
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_radius = agent_radius
        navmesh_settings.agent_height = agent_height
        navmesh_settings.agent_max_climb = agent_max_climb
        navmesh_settings.cell_height = 0.05
        navmesh_settings.cell_size = 0.03
        
        # 重新计算 navmesh，Habitat 默认保留最大连通区域
        success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings, include_static_objects=True)
        if success:
            sim.pathfinder.save_nav_mesh(out_path)
            print(f"[{scene_path}] 成功生成并保存 Navmesh: {out_path}")
        else:
            print(f"[{scene_path}] Navmesh 生成失败")

if __name__ == "__main__":
    # 示例调用：
    # 填入你自己的场景路径逻辑，或者后续遍历所有 .glb文件
    # example_scene = "/home/wgy/hm3d/scene_datasets/hm3d/train/00016-qk9eeNeR4vw/qk9eeNeR4vw.basis.glb"
    # recompute_and_save_navmesh(example_scene, example_scene.replace(".glb", ".navmesh"))
    pass