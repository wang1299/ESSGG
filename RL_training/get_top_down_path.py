import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from components.agents.a2c_agent import A2CAgent
from components.agents.reinforce_agent import ReinforceAgent
from components.environments.precomputed_thor_env import PrecomputedThorEnv
from components.environments.thor_env import ThorEnv
from components.environments.top_down_mapper import OrthoTopDownMapper
from components.utils.utility_functions import read_config


def _blend_line(img_bgr, p1, p2, color, thickness=3, alpha=0.75):
    """Draw one anti-aliased line segment with alpha blending."""
    overlay = img_bgr.copy()
    cv2.line(overlay, p1, p2, color, thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, dst=img_bgr)


def _draw_styled_polyline(img_bgr, pts, color, thickness=3, alpha=0.75, pattern="solid", dash_len=12, gap_len=8, dot_len=2):
    """
    Draw a polyline using a style: 'solid' | 'dashed' | 'dotted' | 'dashdot'.
    Lengths are in pixels along the polyline.
    """
    if len(pts) < 2:
        return

    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        vx, vy = x2 - x1, y2 - y1
        seg_len = int(np.hypot(vx, vy))
        if seg_len == 0:
            continue
        ux, uy = vx / seg_len, vy / seg_len

        if pattern == "solid":
            _blend_line(img_bgr, (x1, y1), (x2, y2), color, thickness, alpha)
            continue

        pos = 0
        cycle = None
        i = 0
        if pattern == "dashdot":
            # cycle: dash, gap, dot, gap
            cycle = [(dash_len, True), (gap_len, False), (dot_len, True), (gap_len, False)]

        while pos < seg_len:
            if pattern == "dashed":
                L_on, L_off = dash_len, gap_len
                on = True
            elif pattern == "dotted":
                L_on, L_off = dot_len, gap_len
                on = True
            elif pattern == "dashdot":
                L_on, on = cycle[i % 4]
                i += 1
                L_off = 0  # handled by next cycle step
            else:
                L_on, L_off, on = dash_len, gap_len, True  # fallback

            L = min(L_on, seg_len - pos)
            xs = int(round(x1 + ux * pos))
            ys = int(round(y1 + uy * pos))
            xe = int(round(x1 + ux * (pos + L)))
            ye = int(round(y1 + uy * (pos + L)))
            if on:
                _blend_line(img_bgr, (xs, ys), (xe, ye), color, thickness, alpha)
            pos += L if pattern == "dashdot" else (L_on + L_off)


def draw_path_on_topdown(img_rgb, traj_xz, mapper, color=(255, 0, 0), thickness=2):
    """
    Draw a polyline for the agent trajectory on the top-down image.
    - img_rgb: numpy array (H,W,3), RGB uint8
    - traj_xz: list of (x, z) world coordinates recorded during the episode
    """
    if img_rgb is None:
        raise ValueError("img_rgb is None")
    if not traj_xz or len(traj_xz) < 2:
        return img_rgb.copy()

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    pts = [mapper.world_to_pixel(x, z) for (x, z) in traj_xz]
    for (u1, v1), (u2, v2) in zip(pts[:-1], pts[1:]):
        cv2.line(img_bgr, (u1, v1), (u2, v2), color, thickness)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def draw_start_marker(img_rgb, mapper, world_xz, outer=9, inner=6):
    """Draw a single start marker (white disk with black outline) on top of the image."""
    u, v = mapper.world_to_pixel(world_xz[0], world_xz[1])
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.circle(bgr, (u, v), outer, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)  # black outline
    cv2.circle(bgr, (u, v), inner, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)  # white fill
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_goal_marker(img_rgb, mapper, world_xz, color_bgr, size=8, thickness=2):
    """Draw a colored X as goal marker in the agent's color."""
    u, v = mapper.world_to_pixel(world_xz[0], world_xz[1])
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    cv2.line(bgr, (u - size, v - size), (u + size, v + size), color_bgr, thickness, lineType=cv2.LINE_AA)
    cv2.line(bgr, (u - size, v + size), (u + size, v - size), color_bgr, thickness, lineType=cv2.LINE_AA)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_paths_on_topdown_multi(
    img_rgb, paths, mapper, colors, thickness=3, alpha=0.75, dash_len=14, gap_len=10, dot_len=2, start_marker=True, goal_marker_x=True
):
    """
    Draw multiple trajectories on one top-down image.
    - ≤2 agents: all solid
    - ≥3 agents: solid / dashed / dotted / dashdot cycling
    - same thickness & alpha for all
    - one shared white start marker; per-agent colored goal 'X'
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()

    # color map
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(paths.keys())} if isinstance(colors, list) else colors

    agent_names = list(paths.keys())
    n_agents = len(agent_names)

    # choose styles based on number of agents
    if n_agents <= 2:
        styles = ["solid"] * n_agents
    else:
        base_styles = ["solid", "dashed", "dotted", "dashdot"]
        styles = [base_styles[i % len(base_styles)] for i in range(n_agents)]

    # shared start marker
    if start_marker and agent_names:
        first_traj = paths[agent_names[0]]
        if first_traj:
            u, v = mapper.world_to_pixel(*first_traj[0])
            cv2.circle(img_bgr, (u, v), 6, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    # draw paths
    for i, name in enumerate(agent_names):
        traj = paths[name]
        if not traj or len(traj) < 2:
            continue
        pts = [mapper.world_to_pixel(x, z) for (x, z) in traj]
        _draw_styled_polyline(
            img_bgr,
            pts,
            color=color_map[name],
            thickness=thickness,
            alpha=alpha,
            pattern=styles[i],
            dash_len=dash_len,
            gap_len=gap_len,
            dot_len=dot_len,
        )
        # goal X
        if goal_marker_x:
            u, v = pts[-1]
            s = 7
            cv2.line(img_bgr, (u - s, v - s), (u + s, v + s), color_map[name], 2, cv2.LINE_AA)
            cv2.line(img_bgr, (u - s, v + s), (u + s, v - s), color_map[name], 2, cv2.LINE_AA)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def get_trajectory(agent, env, scene_number, start_pos=None, start_rot=None):
    """
    Run a single episode and return the trajectory of (x, z) positions.
    """
    if start_pos and start_rot:
        obs = env.reset(scene_number=scene_number, random_start=False, start_position=start_pos, start_rotation=start_rot)
    else:
        obs = env.reset(scene_number=scene_number, random_start=True)
    traj_xz = []
    traj_xz.append((start_pos["x"], start_pos["z"]))
    while not (obs.terminated or obs.truncated):
        action, _, _, _, _, _ = agent.get_action(obs)
        obs = env.step(action)
        pos = obs.info.get("agent_pos", None)
        if pos:
            traj_xz.append(pos)
    return traj_xz, obs.info["score"]


def add_bottom_legend(img_rgb, legend_items, row_height=30, bg_color=(240, 240, 240)):
    """
    Add a vertical list legend below the image.
    - img_rgb: (H,W,3) RGB numpy array
    - legend_items: list of (label, color_bgr)
    - row_height: pixel height per legend entry
    """
    H, W, _ = img_rgb.shape
    panel_h = row_height * len(legend_items) + 10  # +padding
    panel = np.full((panel_h, W, 3), bg_color, dtype=np.uint8)

    x0 = 20
    y = 10
    for label, color in legend_items:
        # color box
        cv2.rectangle(panel, (x0, y), (x0 + 25, y + 20), color, -1)
        # text (larger font)
        cv2.putText(panel, label, (x0 + 40, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        y += row_height

    # stack vertically: image above, legend panel below
    out_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out_bgr = np.vstack([out_bgr, panel])
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def get_initial_top_down_image(scene_number, start_pos, start_rot):
    """
    Get the initial top-down image from the environment.
    """
    env = ThorEnv()
    env.reset(scene_number=scene_number, start_position=start_pos, start_rotation=start_rot)
    image = env.get_top_down_view()
    center_x, center_z, ortho_size = env.td_center_x, env.td_center_z, env.td_ortho_size
    env.close()
    image_height, image_width = image.shape[:2]
    return image, image_height, image_width, center_x, center_z, ortho_size


def get_top_down_path(agents_config_and_weights, scene_numbers, num_starts_per_scene=3, out_dir=None, mode="train"):
    out_dir = f"components/data/topdowns/{mode}" if out_dir is None else out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = PrecomputedThorEnv()

    # Distinct BGR colors (cv2 draws in BGR)
    base_colors = [
        (0, 0, 255),  # red
        (0, 255, 0),  # green
        (255, 0, 0),  # blue
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
        (255, 255, 0),  # cyan
    ]

    for scene_number in scene_numbers:
        for start_idx in range(num_starts_per_scene):
            # Fix a random start and reuse for all agents
            obs = env.reset(scene_number=scene_number, random_start=True)
            start_pos = obs.info["event"].metadata["agent"]["position"]
            start_rot = obs.info["event"].metadata["agent"]["rotation"]

            # Base top-down and mapper from a clean ThorEnv
            top_down_img, img_h, img_w, cx, cz, ortho_size = get_initial_top_down_image(scene_number, start_pos, start_rot)
            mapper = OrthoTopDownMapper(cx, cz, ortho_size, img_h, img_w)

            # 1) run all agents and collect paths
            paths = {}
            colors = {}
            scores = {}
            for idx, agent_name in enumerate(agents_config_and_weights.keys()):
                agent_cfg, nav_cfg, weight_path = agents_config_and_weights[agent_name]

                if agent_cfg["name"].lower().startswith("reinforce"):
                    agent = ReinforceAgent(env=env, navigation_config=nav_cfg, agent_config=agent_cfg, device=device)
                elif agent_cfg["name"].lower().startswith("a2c"):
                    agent = A2CAgent(env=env, navigation_config=nav_cfg, agent_config=agent_cfg, device=device)
                else:
                    raise ValueError(f"Unknown agent type: {agent_cfg['name']}")

                agent.load_weights(model_path=weight_path, device=device)

                traj_xz, score = get_trajectory(agent, env, scene_number, start_pos, start_rot)
                if not traj_xz:
                    print(f"[warn] empty trajectory for {agent_name} (scene {scene_number}, start {start_idx})")

                paths[agent_name] = traj_xz
                colors[agent_name] = base_colors[idx % len(base_colors)]
                scores[agent_name] = score

            # 2) draw all trajectories with decreasing thickness (first agent thickest)
            combined = draw_paths_on_topdown_multi(top_down_img, paths, mapper, colors)

            # 3) one shared START marker (same for all)
            start_world_xz = (start_pos["x"], start_pos["z"])
            # combined = draw_start_marker(combined, mapper, start_world_xz, outer=9, inner=6)

            # 4) GOAL marker per agent in matching color (use last point of each trajectory)
            for agent_name, traj in paths.items():
                if traj:
                    goal_world_xz = traj[-1]
                    combined = draw_goal_marker(combined, mapper, goal_world_xz, color_bgr=colors[agent_name])

            legend_rows = [(agent_name, len(paths[agent_name]) - 1, scores[agent_name], colors[agent_name]) for agent_name in paths.keys()]
            combined_with_legend = add_bottom_legend_multiline(combined, legend_rows)

            if len(agents_config_and_weights.keys()) == 4:
                out_path = Path(out_dir) / "ALL" / f"topdown_scene{scene_number}_start{start_idx}.png"
            else:
                agent_names = "_".join(agents_config_and_weights.keys())
                out_path = Path(out_dir) / agent_names / f"topdown_scene{scene_number}_start{start_idx}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), cv2.cvtColor(combined_with_legend, cv2.COLOR_RGB2BGR))
            print(f"[info] wrote {out_path}")


import cv2
import numpy as np


def _truncate_to_width(text, max_px, font, scale, thick):
    """Truncate text with ellipsis so that rendered width <= max_px."""
    if max_px <= 0:
        return ""
    t = text
    (w, _), _ = cv2.getTextSize(t, font, scale, thick)
    if w <= max_px:
        return t
    ell = "…"
    while len(t) > 1:
        t = t[:-1]
        (w, _), _ = cv2.getTextSize(t + ell, font, scale, thick)
        if w <= max_px:
            return t + ell
    return ell


def add_bottom_legend_multiline(
    img_rgb,
    rows,  # list of (name, steps, score, color_bgr)
    bg_color=(255, 255, 255),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    name_scale=0.58,
    stats_scale=0.54,
    thickness=1,
    line_gap=4,  # gap between the two lines of one entry
    row_gap=12,  # <-- mehr Abstand zwischen Agenten
    left_pad=16,
    right_pad=16,
    top_pad=10,
    bottom_pad=10,
    swatch_w=22,
    swatch_h=12,
    swatch_gap=10,
):
    H, W, _ = img_rgb.shape
    text_x0 = left_pad + swatch_w + swatch_gap
    max_text_w = max(20, W - text_x0 - right_pad)

    total_h = top_pad + bottom_pad
    line_heights, rendered = [], []

    for name, steps, score, color in rows:
        # remove underscores
        clean_name = str(name).replace("_", " ")

        # truncate if too wide
        name_txt = _truncate_to_width(clean_name, max_text_w, font, name_scale, thickness)
        stats_txt = _truncate_to_width(f"Steps: {steps} | Score: {score:.2f}", max_text_w, font, stats_scale, thickness)

        (nw, nh), _ = cv2.getTextSize(name_txt, font, name_scale, thickness)
        (sw, sh), _ = cv2.getTextSize(stats_txt, font, stats_scale, thickness)

        entry_h = max(swatch_h + 4, nh + line_gap + sh)
        line_heights.append(entry_h)
        total_h += entry_h + row_gap
        rendered.append((name_txt, stats_txt, color))

    total_h -= row_gap
    panel = np.full((total_h, W, 3), bg_color, dtype=np.uint8)

    y = top_pad
    for (name_txt, stats_txt, color), entry_h in zip(rendered, line_heights):
        cv2.rectangle(panel, (left_pad, y + 2), (left_pad + swatch_w, y + 2 + swatch_h), color, -1)

        (nw, nh), _ = cv2.getTextSize(name_txt, font, name_scale, thickness)
        (sw, sh), _ = cv2.getTextSize(stats_txt, font, stats_scale, thickness)

        name_y = y + nh + 1
        stats_y = name_y + line_gap + sh

        cv2.putText(panel, name_txt, (text_x0, name_y), font, name_scale, (20, 20, 20), thickness, cv2.LINE_AA)
        cv2.putText(panel, stats_txt, (text_x0, stats_y), font, stats_scale, (30, 30, 30), thickness, cv2.LINE_AA)

        y += entry_h + row_gap

    out_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out_bgr = np.vstack([out_bgr, panel])
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def get_agent_config_and_weights(agents):
    agent_config_and_weights = {}

    # REINFORCE LSTM
    agent_config = read_config("RL_training/sbatch/configs/REINFORCE_LSTM/agent.json", use_print=False)
    navigation_config = read_config("RL_training/sbatch/configs/REINFORCE_LSTM/navigation.json", use_print=False)
    agent_weight_path = "RL_training/runs/model_weights/REINFORCE_Agent_LSTM/2025-08-19_23-04-16_reinforce_agent.pth"
    if "REINFORCE_LSTM" in agents:
        agent_config_and_weights["REINFORCE_LSTM"] = (agent_config, navigation_config, agent_weight_path)

    # REINFORCE Transformer
    agent_config = read_config("RL_training/sbatch/configs/REINFORCE_Transformer/agent.json", use_print=False)
    navigation_config = read_config("RL_training/sbatch/configs/REINFORCE_Transformer/navigation.json", use_print=False)
    agent_weight_path = "RL_training/runs/model_weights/REINFORCE_Agent_Transformer/2025-08-27_01-11-40_reinforce_agent.pth"
    if "REINFORCE_Transformer" in agents:
        agent_config_and_weights["REINFORCE_Transformer"] = (agent_config, navigation_config, agent_weight_path)

    # A2C LSTM
    agent_config = read_config("RL_training/sbatch/configs/A2C_LSTM/agent.json", use_print=False)
    navigation_config = read_config("RL_training/sbatch/configs/A2C_LSTM/navigation.json", use_print=False)
    agent_weight_path = "RL_training/runs/model_weights/A2C_Agent_LSTM/2025-08-22_22-18-44_a2c_agent.pth"
    if "A2C_LSTM" in agents:
        agent_config_and_weights["A2C_LSTM"] = (agent_config, navigation_config, agent_weight_path)

    # A2C Transformer
    agent_config = read_config("RL_training/sbatch/configs/A2C_Transformer/agent.json", use_print=False)
    navigation_config = read_config("RL_training/sbatch/configs/A2C_Transformer/navigation.json", use_print=False)
    agent_weight_path = "RL_training/runs/model_weights/A2C_Agent_Transformer/2025-08-29_20-18-21_a2c_agent.pth"
    if "A2C_Transformer" in agents:
        agent_config_and_weights["A2C_Transformer"] = (agent_config, navigation_config, agent_weight_path)

    return agent_config_and_weights


def set_working_directory():
    desired_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    current_directory = os.getcwd()

    if current_directory != desired_directory:
        os.chdir(desired_directory)
        sys.path.append(desired_directory)
        print(f"Current working director changed from '{current_directory}', to '{desired_directory}'")
        return

    print("Current working director:", os.getcwd())


if __name__ == "__main__":
    set_working_directory()
    mode = "eval"  # "train" | "eval"
    agents_to_use = [
        ["REINFORCE_LSTM", "REINFORCE_Transformer", "A2C_LSTM", "A2C_Transformer"],
        ["REINFORCE_LSTM", "A2C_LSTM"],
        ["REINFORCE_Transformer", "A2C_Transformer"],
    ]
    agents_to_use = [
        ["REINFORCE_Transformer", "A2C_Transformer"],
        ["REINFORCE_LSTM", "A2C_LSTM"],
        ["REINFORCE_LSTM", "REINFORCE_Transformer", "A2C_LSTM", "A2C_Transformer"],
    ]
    agents_to_use = [["REINFORCE_LSTM", "REINFORCE_Transformer", "A2C_LSTM", "A2C_Transformer"]]
    for agents in agents_to_use:
        print(f"\n=== Generating top-down paths for agents: {agents} ===")
        agent_config_and_weights = get_agent_config_and_weights(agents=agents)

        all_scene_numbers = list(range(1, 6)) + list(range(7, 8)) + list(range(9, 31))
        if mode == "train":
            scene_numbers = all_scene_numbers[:10]
        else:
            scene_numbers = all_scene_numbers[10:13]

        get_top_down_path(agent_config_and_weights, scene_numbers, num_starts_per_scene=40, mode=mode)
