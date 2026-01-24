import os
import json
import math
from components.utils.utility_functions import generate_seeds


def create_sl_configs(config_base_dir, num_seeds=20, episodes_override=1000, group_size=1):
    """
    For each agent config, generate N seeds, grouped into conf_0, conf_1, ... each holding up to 5 seeds,
    and write combined config files into sl_configs/<AgentName>/conf_<n>/config.json.

    Args:
        config_base_dir (str): Path to the base folder containing agent subdirectories with config files.
        num_seeds (int): Total number of seeds to generate.
        episodes_override (int): Value to overwrite the 'episodes' field in agent.json.
        group_size (int): Number of seeds per configuration group (default is 5).
    """
    seeds = generate_seeds(num_seeds)

    agent_dirs = [d for d in os.listdir(config_base_dir) if os.path.isdir(os.path.join(config_base_dir, d))]

    for agent_name in sorted(agent_dirs):
        agent_path = os.path.join(config_base_dir, agent_name)
        output_base = os.path.join("sl_configs", agent_name)

        try:
            with open(os.path.join(agent_path, "agent.json")) as f:
                agent_template = json.load(f)
            with open(os.path.join(agent_path, "env.json")) as f:
                env_template = json.load(f)
            with open(os.path.join(agent_path, "navigation.json")) as f:
                nav_template = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load configs for {agent_name}: {e}")
            continue

        num_groups = math.ceil(num_seeds / group_size)
        for group_idx in range(num_groups):
            group_seeds = seeds[group_idx * group_size : (group_idx + 1) * group_size]
            for seed in group_seeds:
                # Prepare full config
                agent_cfg = json.loads(json.dumps(agent_template))  # deep copy via serialization
                env_cfg = json.loads(json.dumps(env_template))
                nav_cfg = json.loads(json.dumps(nav_template))

                agent_cfg["seed"] = seed
                agent_cfg["episodes"] = episodes_override
                nav_cfg["seed"] = seed
                env_cfg["seed"] = seed

                full_cfg = {"seed": seed, "agent_config": agent_cfg, "env_config": env_cfg, "navigation_config": nav_cfg}

                # Write to sl_configs/AgentName/conf_<group_idx>/config_<seed>.json
                out_dir = os.path.join(output_base, f"conf_{group_idx}")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"config_{seed}.json")
                with open(out_path, "w") as out_file:
                    json.dump(full_cfg, out_file, indent=2)
        print(f"[INFO] Generated {num_seeds} configs for '{agent_name}' in {num_groups} groups.")


if __name__ == "__main__":
    create_sl_configs(config_base_dir="configs", num_seeds=10, episodes_override=1000)
