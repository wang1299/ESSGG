import os
import json
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


# ---- Matplotlib defaults for print-quality figures --------------------------------
plt.rcParams.update(
    {
        "figure.figsize": (10, 7),  # etwas höher als vorher (9x6)
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.35,
        "axes.titleweight": "bold",
        # Fonts deutlich größer
        "font.size": 16,  # baseline
        "axes.labelsize": 18,  # Achsenlabels
        "axes.titlesize": 20,  # Überschriften/Titel
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        # Legend
        "legend.frameon": False,
        "legend.handlelength": 2.5,
        "legend.handleheight": 1.2,
        # Export-Einstellungen
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


# ------------- Utils ---------------------------------------------------------


def _ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


def _nice_label(agent_dir_basename: str) -> str:
    """Optional: map folder names to pretty legend labels if needed."""
    # e.g., turn "A2C_LSTM" -> "A2C LSTM"
    return agent_dir_basename.replace("_", " ")


def moving_average(x, w, align="center", pad_mode="reflect"):
    """
    Moving average that returns the same number of values as the input.

    Args:
        x (array-like): Input series.
        w (int): Window size (>=1).
        align (str): 'center' (default), 'left' (causal), or 'right' (anti-causal).
        pad_mode (str): How to pad at the edges ('reflect', 'edge', 'constant', ...).

    Returns:
        np.ndarray: Smoothed series with len == len(x).
    """
    x = np.asarray(x, dtype=float)
    if w <= 1 or len(x) == 0:
        return x.copy()

    kernel = np.ones(w, dtype=float) / w

    if align == "center":
        pad_left = (w - 1) // 2
        pad_right = w - 1 - pad_left
        x_pad = np.pad(x, (pad_left, pad_right), mode=pad_mode)
        y = np.convolve(x_pad, kernel, mode="valid")
    elif align == "left":
        x_pad = np.pad(x, (w - 1, 0), mode=pad_mode)
        y = np.convolve(x_pad, kernel, mode="valid")
    elif align == "right":
        x_pad = np.pad(x, (0, w - 1), mode=pad_mode)
        y = np.convolve(x_pad, kernel, mode="valid")
    else:
        raise ValueError("align must be 'center', 'left', or 'right'.")

    return y[: len(x)]


# ------------- TensorBoard I/O ----------------------------------------------


def get_config_from_event(event_path):
    """Extracts the config from the correct tensor tag ('full_config/text_summary')."""
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    tags = ea.Tags()
    config_tag = "full_config/text_summary"
    if config_tag not in tags.get("tensors", []):
        print(f"[WARN] '{config_tag}' not found in {event_path}, skipping file.")
        return None
    try:
        tensor_events = ea.Tensors(config_tag)
        raw = tensor_events[0].tensor_proto.string_val[0]
        cfg_text = raw.decode("utf-8")
        config = json.loads(cfg_text)
        return config
    except Exception as e:
        print(f"Failed to extract config from {event_path}: {e}")
        return None


def config_without_seed(cfg):
    """Removes the seed from agent_config for config equality check."""
    cfg = deepcopy(cfg)
    if "agent_config" in cfg and "seed" in cfg["agent_config"]:
        del cfg["agent_config"]["seed"]
    if "navigation_config" in cfg and "seed" in cfg["navigation_config"]:
        del cfg["navigation_config"]["seed"]
    return cfg


def find_valid_event_files(event_files):
    """Returns a list of valid event files with matching config (ignoring seed)."""
    configs = []
    for f in event_files:
        cfg = get_config_from_event(f)
        if cfg is not None:
            configs.append((f, config_without_seed(cfg)))
    if not configs:
        return []
    ref_cfg = configs[0][1]
    filtered = [f for f, c in configs if c == ref_cfg]
    for f, c in configs:
        if c != ref_cfg:
            print(f"[WARN] Config mismatch in {f}, ignoring file.")
    return filtered


def collect_all_valid_event_files(agent_dir):
    """Collects all valid event files for an agent (subdir with seeds/runs)."""
    event_files = []
    for root, _, files in os.walk(agent_dir):
        for fname in files:
            if fname.startswith("events.out.tfevents."):
                event_files.append(os.path.join(root, fname))
    return find_valid_event_files(event_files) if event_files else []


def extract_scalar_from_event(event_path, tag):
    """Loads scalar values for the given tag from a TensorBoard event file."""
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    try:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    except KeyError:
        print(f"Tag {tag} not found in {event_path}")
        return [], []


def aggregate_seeds(event_files, tag):
    """
    Aggregates scalar values across seeds.

    Returns:
        steps_common: list[int]
        mean: np.ndarray
        std: np.ndarray              (across seeds, per step)
        all_trimmed: np.ndarray      shape (n_seeds, T)
    """
    all_seed_steps, all_seed_values = [], []
    for event_file in event_files:
        steps, values = extract_scalar_from_event(event_file, tag)
        if steps and values:
            all_seed_steps.append(steps)
            all_seed_values.append(values)

    if not all_seed_steps:
        return None, None, None, None

    min_length = min(len(s) for s in all_seed_steps)
    # trim to common length
    all_trimmed = np.array([np.asarray(vals[:min_length], dtype=float) for vals in all_seed_values])
    mean = np.mean(all_trimmed, axis=0)
    std = np.std(all_trimmed, axis=0)
    steps_common = all_seed_steps[0][:min_length]
    return steps_common, mean, std, all_trimmed


# ------------- Plotting ------------------------------------------------------


def plot_metric(
    base_dir,
    tag,
    ylabel,
    title,
    max_episodes=1000,
    ylim=None,
    smooth=100,
    save_path=None,
    show=True,
    plot_ci=True,
    plot_seeds=False,
    ci_level=0.95,
    band_type="ci",
    iqr=(0.25, 0.75),
):
    """
    Generic plotting helper for one metric across all agent folders in `base_dir`.
    - Plots mean over seeds (smoothed).
    - Optional: confidence/uncertainty band, controlled by `band_type`:
        - 'ci' (default): confidence interval (mean ± z * SE, with `ci_level`)
        - 'se': mean ± 1 standard error (SE)
        - 'iqr': quantile band using `iqr` tuple (e.g., (0.25, 0.75) for interquartile range)
    - Optional: thin per-seed curves.
    """
    if save_path:
        _ensure_dir(os.path.dirname(save_path))

    plt.figure(figsize=(10, 6))
    agent_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    agent_dirs = sorted(agent_dirs)

    plotted = 0
    for agent_dir in agent_dirs:
        event_files = collect_all_valid_event_files(agent_dir)
        if not event_files:
            print(f"[WARN] No valid event files found for {agent_dir}, skipping.")
            continue

        steps, mean, std, all_trimmed = aggregate_seeds(event_files, tag)
        if steps is None:
            print(f"[WARN] No data for {agent_dir}, skipping.")
            continue

        n = len(event_files)
        se = std / max(1, np.sqrt(n))
        label = _nice_label(os.path.basename(agent_dir))

        # optional per-seed lines (thin, semi-transparent)
        if plot_seeds and all_trimmed is not None:
            for seed_vals in all_trimmed:
                if smooth > 1:
                    seed_vals = moving_average(seed_vals, smooth)
                plt.plot(steps[: len(seed_vals)], seed_vals, alpha=0.15, linewidth=1)

        # smooth mean & SE (keep lengths aligned)
        if smooth > 1:
            mean_s = moving_average(mean, smooth)
            se_s = moving_average(se, smooth)
            x = steps[: len(mean_s)]
        else:
            mean_s, se_s, x = mean, se, steps

        plt.plot(x, mean_s, label=f"{label} ", linewidth=2)

        if plot_ci:
            lower_raw = upper_raw = None
            if band_type == "iqr":
                if all_trimmed is None:
                    pass
                else:
                    q_low, q_high = iqr
                    lower_raw = np.quantile(all_trimmed, q_low, axis=0)
                    upper_raw = np.quantile(all_trimmed, q_high, axis=0)
            else:
                from scipy.stats import norm

                z = 1.0 if band_type == "se" else norm.ppf(0.5 + ci_level / 2.0)
                lower_raw = mean - z * (std / max(1, np.sqrt(n)))
                upper_raw = mean + z * (std / max(1, np.sqrt(n)))
            if lower_raw is not None and upper_raw is not None:
                if smooth > 1:
                    lower = moving_average(lower_raw, smooth)
                    upper = moving_average(upper_raw, smooth)
                    x_band = steps[: len(lower)]
                else:
                    lower, upper, x_band = lower_raw, upper_raw, x
                plt.fill_between(x_band, lower, upper, alpha=0.15)

        plotted += 1

    if plotted == 0:
        print("No agent curves were plotted. Check folder structure and config tags.")
        return

    plt.legend()
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlim([0, max_episodes])
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Figure saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_score(base_dir, max_episodes=1000, ylim=(0.5, 1.0), smooth=100, save_path="abb/mean_score_per_agent.pdf", show=False, **kwargs):
    tag = "last_episode/Mean_Score"
    plot_metric(
        base_dir,
        tag,
        ylabel="Score",
        title="Mean Score per Agent",
        max_episodes=max_episodes,
        ylim=ylim,
        smooth=smooth,
        save_path=save_path,
        show=show,
        **kwargs,
    )


def plot_steps(base_dir, max_episodes=1000, ylim=(20, 40), smooth=100, save_path="abb/mean_steps_per_agent.pdf", show=False, **kwargs):
    tag = "last_episode/Mean_Steps"
    plot_metric(
        base_dir,
        tag,
        ylabel="Steps",
        title="Mean Steps per Agent",
        max_episodes=max_episodes,
        ylim=ylim,
        smooth=smooth,
        save_path=save_path,
        show=show,
        **kwargs,
    )


def plot_steps_for_score_1(
    base_dir, max_episodes=1000, ylim=(0, 50), smooth=100, save_path="abb/steps_per_score1_per_agent.pdf", show=False, **kwargs
):
    tag = "Rollout/Steps_for_score_1"
    plot_metric(
        base_dir,
        tag,
        ylabel="Steps for Score 1",
        title="Steps per Score 1",
        max_episodes=max_episodes,
        ylim=ylim,
        smooth=smooth,
        save_path=save_path,
        show=show,
        **kwargs,
    )


def plot_reward(base_dir, max_episodes=1000, ylim=(-2, 2), smooth=100, save_path="abb/mean_reward_per_agent.pdf", show=False, **kwargs):
    tag = "Rollout/Mean_Reward"
    plot_metric(
        base_dir,
        tag,
        ylabel="Reward",
        title="Mean Reward",
        max_episodes=max_episodes,
        ylim=ylim,
        smooth=smooth,
        save_path=save_path,
        show=show,
        **kwargs,
    )


def plot_score_and_steps_pair(
    base_dir,
    max_episodes=1000,
    smooth=100,
    ylim_score=(0.5, 1.0),
    ylim_steps=(20, 40),
    save_path="abb/main_learning_curves.pdf",
    show=False,
    **kwargs,
):
    """
    Erzeugt eine Doppel-Abbildung (Score links, Steps rechts) im Stil der Paper-Figur.
    The confidence/uncertainty band can be controlled via `band_type` ('ci', 'se', 'iqr') and `iqr=(0.25,0.75)`, passed via **kwargs.
    """
    from scipy.stats import norm

    if save_path:
        _ensure_dir(os.path.dirname(save_path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, constrained_layout=True)
    tags = [("last_episode/Mean_Score", "Score", "Mean Score (Node Recall)"), ("last_episode/Mean_Steps", "Steps", "Mean Steps")]
    ylims = [ylim_score, ylim_steps]

    agent_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    agent_dirs = sorted(agent_dirs)

    ci_level = kwargs.get("ci_level", 0.95)
    plot_ci = kwargs.get("plot_ci", True)
    plot_seeds = kwargs.get("plot_seeds", False)
    band_type = kwargs.get("band_type", "ci")
    iqr = kwargs.get("iqr", (0.25, 0.75))

    for ax, (tag, ylabel, title), ylim in zip(axes, tags, ylims):
        plotted = 0
        for agent_dir in agent_dirs:
            event_files = collect_all_valid_event_files(agent_dir)
            if not event_files:
                continue
            steps, mean, std, all_trimmed = aggregate_seeds(event_files, tag)
            if steps is None:
                continue
            n = len(event_files)
            se = std / max(1, np.sqrt(n))
            label = _nice_label(os.path.basename(agent_dir))

            if plot_seeds and all_trimmed is not None:
                for seed_vals in all_trimmed:
                    series = moving_average(seed_vals, smooth) if smooth > 1 else seed_vals
                    ax.plot(steps[: len(series)], series, alpha=0.15, linewidth=1)

            mean_s = moving_average(mean, smooth) if smooth > 1 else mean
            se_s = moving_average(se, smooth) if smooth > 1 else se
            x = steps[: len(mean_s)]

            ax.plot(x, mean_s, label=f"{label} ", linewidth=2)

            if plot_ci:
                lower_raw = upper_raw = None
                if band_type == "iqr":
                    if all_trimmed is not None:
                        q_low, q_high = iqr
                        lower_raw = np.quantile(all_trimmed, q_low, axis=0)
                        upper_raw = np.quantile(all_trimmed, q_high, axis=0)
                else:
                    z = 1.0 if band_type == "se" else norm.ppf(0.5 + ci_level / 2.0)
                    lower_raw = mean - z * se
                    upper_raw = mean + z * se
                if lower_raw is not None and upper_raw is not None:
                    if smooth > 1:
                        lower = moving_average(lower_raw, smooth)
                        upper = moving_average(upper_raw, smooth)
                        x_band = steps[: len(lower)]
                    else:
                        lower, upper, x_band = lower_raw, upper_raw, x
                    ax.fill_between(x_band, lower, upper, alpha=0.15)

            plotted += 1

        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlim([0, max_episodes])

        if plotted == 0:
            print(f"[WARN] No curves plotted for tag '{tag}'.")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=13)
    fig.tight_layout(rect=[0, 0.1, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Figure saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


# ------------- Tabellarische Auswertung -------------------------------------


def export_final_values(base_dir, tag, last_n=200, output_file=None):
    """
    Computes the mean of the last `last_n` values for each agent's tag.

    Returns:
        pd.DataFrame: columns = Agent, Mean_{last_n}, Std_{last_n}, Num_Seeds
    """
    results = []
    agent_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for agent_dir in sorted(agent_dirs):
        event_files = collect_all_valid_event_files(agent_dir)
        if not event_files:
            print(f"[WARN] No valid event files found for {agent_dir}, skipping.")
            continue

        steps, mean, std, _ = aggregate_seeds(event_files, tag)
        if mean is None or len(mean) < last_n:
            print(f"[WARN] Not enough data for {agent_dir}, skipping.")
            continue

        mean_last = float(np.mean(mean[-last_n:]))
        std_last = float(np.std(mean[-last_n:]))
        results.append(
            {
                "Agent": _nice_label(os.path.basename(agent_dir)),
                f"Mean_{last_n}": mean_last,
                f"Std_{last_n}": std_last,
                "Num_Seeds": len(event_files),
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values(by=f"Mean_{last_n}", ascending=False).reset_index(drop=True)
    if output_file:
        _ensure_dir(os.path.dirname(output_file))
        df.to_csv(output_file, index=False)
        print(f"[INFO] Results saved to {output_file}")
    return df


def export_all_tables(base_dir, last_n=200, out_dir="tables"):
    """Convenience: export all three tables as CSV (für LaTeX Import)."""
    _ensure_dir(out_dir)
    tables = {
        "scores.csv": "last_episode/Mean_Score",
        "steps.csv": "last_episode/Mean_Steps",
        "steps_for_score1.csv": "Rollout/Steps_for_score_1",
    }
    out = {}
    for fname, tag in tables.items():
        df = export_final_values(base_dir, tag=tag, last_n=last_n, output_file=os.path.join(out_dir, fname))
        out[fname] = df
    return out


# ------------- Main ---------------------------------------------------------

if __name__ == "__main__":
    mode = "training"  # "training" | "eval"
    base_dir = "runs/" + mode
    smooth = 10 if mode == "eval" else 100  # rolling window
    # Hauptfiguren (für den Paper-Hauptteil)
    max_episodes = 100 if mode == "eval" else 1000
    plots = True
    tables = False
    if plots:
        plot_score_and_steps_pair(
            base_dir,
            max_episodes=max_episodes,
            smooth=smooth,
            ylim_score=(0.4, 1.0),
            ylim_steps=(10, 35),
            save_path=f"abb/{mode}/main_learning_curves.pdf",
            show=True,
            plot_ci=True,
            band_type="se",  # "ci" | "se" | "iqr"
            plot_seeds=False,
        )

        # Zusatzfiguren (Appendix)
        plot_reward(
            base_dir,
            max_episodes=max_episodes,
            ylim=(-2, 2),
            smooth=smooth,
            save_path=f"abb/{mode}/mean_reward_per_agent.pdf",
            show=True,
            plot_ci=True,
            band_type="se",
            plot_seeds=False,
        )
        plot_steps_for_score_1(
            base_dir,
            max_episodes=max_episodes,
            ylim=(15, 45),
            smooth=smooth,
            save_path=f"abb/{mode}/steps_per_score1_per_agent.pdf",
            show=True,
            plot_ci=False,
            band_type="se",
            plot_seeds=False,
        )

    if tables:
        # Tabellen (Durchschnitt der letzten 200 Episoden)
        last_n = 100 if mode == "eval" else 200
        print("Scores:")
        df_scores = export_final_values(base_dir, tag="last_episode/Mean_Score", last_n=last_n)
        print(df_scores)

        print("\nSteps:")
        df_steps = export_final_values(base_dir, tag="last_episode/Mean_Steps", last_n=last_n)
        print(df_steps)

        print("\nSteps for Score 1:")
        df_eff = export_final_values(base_dir, tag="Rollout/Steps_for_score_1", last_n=last_n)
        print(df_eff)

    # CSV-Dateien schreiben (optional)
    # export_all_tables(base_dir, last_n=last_n, out_dir="tables")
