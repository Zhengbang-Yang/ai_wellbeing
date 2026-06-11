from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .config import DEFAULT_SEED, FIGURES_DIR, RESULTS_DIR
from .utils import bootstrap_mean_ci, ensure_dir, read_jsonl


CONTRAST_LABELS = {
    "difficulty_base": "Hard\nbase",
    "difficulty_praise": "Hard\n+praise",
    "praise_simple": "Praise\neasy",
    "praise_hard": "Praise\nhard",
}

MODEL_LABELS = {
    "qwen_qwen3_5_0_8b": "Qwen3.5-0.8B",
    "qwen_qwen3_5_2b": "Qwen3.5-2B",
    "qwen_qwen3_5_4b": "Qwen3.5-4B",
    "qwen_qwen3_5_9b": "Qwen3.5-9B",
    "qwen_qwen3_5_27b": "Qwen3.5-27B",
}

COLORS = ["#4F7AB0", "#D9828B", "#5E8C61", "#8A6FB0", "#C4904A"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make paper-style figures from completed runs.")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    return parser.parse_args()


def load_utility_raw_rows(utility_dir: Path) -> list[dict]:
    shard_paths = sorted(utility_dir.glob("utility_raw_shard_*_of_*.jsonl"))
    if shard_paths:
        rows = []
        for path in shard_paths:
            rows.extend(read_jsonl(path))
        return rows
    raw_path = utility_dir / "utility_raw.jsonl"
    return read_jsonl(raw_path) if raw_path.exists() else []


def order_averaged_preference_rows(rows: list[dict], model: str) -> list[dict]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    if "template_id" in df:
        experienced = df[df["template_id"] == "experienced"].copy()
        if not experienced.empty:
            df = experienced
    out = []
    for group_id, group in df.groupby("comparison_group_id", sort=False):
        first = group.iloc[0]
        prob = pd.to_numeric(group["prob_treatment"], errors="coerce").mean()
        if not np.isfinite(prob):
            continue
        out.append(
            {
                "model": model,
                "comparison_group_id": group_id,
                "contrast": first["contrast"],
                "prob_treatment": float(prob),
            }
        )
    return out


def load_figure_data(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    utility_pref_rows = []
    utility_score_rows = []
    downstream_effect_rows = []
    for model_dir in sorted(results_dir.iterdir()) if results_dir.exists() else []:
        util_dir = model_dir / "utility_test"
        utility_pref_rows.extend(order_averaged_preference_rows(load_utility_raw_rows(util_dir), model_dir.name))
        scores_path = model_dir / "utility_test" / "utility_scores_with_features.csv"
        if scores_path.exists():
            scores = pd.read_csv(scores_path)
            scores["model"] = model_dir.name
            scores["source"] = scores_path.parent.name
            utility_score_rows.extend(scores.to_dict(orient="records"))
        downstream_specs = [
            ("hard", model_dir / "downstream_hard" / "downstream_paired_by_task.csv"),
            ("easy", model_dir / "downstream_easy" / "downstream_paired_by_task.csv"),
        ]
        for task_set, paired_path in downstream_specs:
            if not paired_path.exists():
                continue
            paired = pd.read_csv(paired_path)
            if {"base_pass", "praise_pass"} <= set(paired.columns):
                diff = pd.to_numeric(paired["praise_pass"], errors="coerce") - pd.to_numeric(
                    paired["base_pass"], errors="coerce"
                )
                for task_id, value in zip(paired["task_id"], diff):
                    if np.isfinite(value):
                        downstream_effect_rows.append(
                            {
                                "model": model_dir.name,
                                "task_set": task_set,
                                "task_id": task_id,
                                "metric": "pass01",
                                "value": float(value),
                            }
                        )
            if {"base_reasoning_tokens", "praise_reasoning_tokens"} <= set(paired.columns):
                diff = pd.to_numeric(paired["praise_reasoning_tokens"], errors="coerce") - pd.to_numeric(
                    paired["base_reasoning_tokens"], errors="coerce"
                )
                for task_id, value in zip(paired["task_id"], diff):
                    if np.isfinite(value):
                        downstream_effect_rows.append(
                            {
                                "model": model_dir.name,
                                "task_set": task_set,
                                "task_id": task_id,
                                "metric": "reasoning_tokens_rough",
                                "value": float(value),
                            }
                        )
    return pd.DataFrame(utility_pref_rows), pd.DataFrame(downstream_effect_rows), pd.DataFrame(utility_score_rows)


def set_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ordered_models(*frames: pd.DataFrame) -> list[str]:
    models: set[str] = set()
    for frame in frames:
        if not frame.empty and "model" in frame.columns:
            models.update(str(model) for model in frame["model"].dropna().unique())
    return sorted(models, key=lambda model: MODEL_LABELS.get(model, model))


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout(pad=0.8)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def grouped_bar_width(n_models: int) -> float:
    return min(0.28, 0.72 / max(1, n_models))


def summarize_grouped_values(values_by_model: dict[str, list[list[float]]]) -> tuple[
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
]:
    means: dict[str, list[float]] = {}
    lows: dict[str, list[float]] = {}
    highs: dict[str, list[float]] = {}
    for model, grouped_values in values_by_model.items():
        means[model] = []
        lows[model] = []
        highs[model] = []
        for values in grouped_values:
            clean = [float(v) for v in values if np.isfinite(v)]
            if clean:
                mean, lo, hi = bootstrap_mean_ci(clean, seed=DEFAULT_SEED)
            else:
                mean = lo = hi = np.nan
            means[model].append(mean)
            lows[model].append(lo)
            highs[model].append(hi)
    return means, lows, highs


def plot_grouped_bars(
    ax: plt.Axes,
    *,
    groups: list[str],
    group_labels: list[str],
    models: list[str],
    values_by_model: dict[str, list[list[float]]],
) -> None:
    x = np.arange(len(groups))
    width = grouped_bar_width(len(models))
    means, lows, highs = summarize_grouped_values(values_by_model)
    for i, model in enumerate(models):
        vals = np.asarray(means.get(model, [np.nan] * len(groups)), dtype=float)
        los = np.asarray(lows.get(model, [np.nan] * len(groups)), dtype=float)
        his = np.asarray(highs.get(model, [np.nan] * len(groups)), dtype=float)
        err = np.vstack([vals - los, his - vals])
        offset = (i - (len(models) - 1) / 2) * width
        color = COLORS[i % len(COLORS)]
        ax.bar(x + offset, vals, width=width, color=color, label=MODEL_LABELS.get(model, model))
        ax.errorbar(x + offset, vals, yerr=err, fmt="none", ecolor="#222222", elinewidth=0.8, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.04)
    ax.legend(frameon=False, loc="best", ncol=min(3, max(1, len(models))))


def grouped_box_width(n_models: int) -> float:
    return min(0.28, 0.72 / max(1, n_models))


def plot_grouped_boxplots(
    ax: plt.Axes,
    *,
    groups: list[str],
    group_labels: list[str],
    models: list[str],
    values_by_model: dict[str, list[list[float]]],
) -> None:
    x = np.arange(len(groups))
    width = grouped_box_width(len(models))
    legend_handles = []
    for i, model in enumerate(models):
        grouped_values = values_by_model.get(model, [[] for _ in groups])
        offset = (i - (len(models) - 1) / 2) * width
        data = []
        positions = []
        for j, values in enumerate(grouped_values):
            clean = [float(v) for v in values if np.isfinite(v)]
            if not clean:
                continue
            data.append(clean)
            positions.append(x[j] + offset)
        color = COLORS[i % len(COLORS)]
        if data:
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=width * 0.78,
                patch_artist=True,
                manage_ticks=False,
                whis=1.5,
                showfliers=True,
                boxprops={"linewidth": 0.8, "edgecolor": "#222222"},
                medianprops={"linewidth": 1.0, "color": "#111111"},
                whiskerprops={"linewidth": 0.8, "color": "#222222"},
                capprops={"linewidth": 0.8, "color": "#222222"},
                flierprops={
                    "marker": "o",
                    "markersize": 2.2,
                    "markerfacecolor": color,
                    "markeredgecolor": "none",
                    "alpha": 0.55,
                },
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.78)
        legend_handles.append(Patch(facecolor=color, edgecolor="none", label=MODEL_LABELS.get(model, model)))
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.04)
    ax.legend(handles=legend_handles, frameon=False, loc="best", ncol=min(3, max(1, len(models))))


def set_bar_ylim(ax: plt.Axes, values_by_model: dict[str, list[list[float]]], *, baseline: float = 0.0) -> None:
    means, lows, highs = summarize_grouped_values(values_by_model)
    vals = [baseline]
    for grouped in (means, lows, highs):
        for grouped_values in grouped.values():
            vals.extend(float(v) for v in grouped_values if np.isfinite(v))
    lo = min(vals)
    hi = max(vals)
    span = max(hi - lo, 0.05)
    ax.set_ylim(lo - 0.18 * span, hi + 0.18 * span)


def set_value_ylim(ax: plt.Axes, values_by_model: dict[str, list[list[float]]], *, baseline: float = 0.0) -> None:
    vals = [baseline]
    for grouped_values in values_by_model.values():
        for values in grouped_values:
            vals.extend(float(v) for v in values if np.isfinite(v))
    lo = min(vals)
    hi = max(vals)
    span = max(hi - lo, 0.05)
    ax.set_ylim(lo - 0.18 * span, hi + 0.18 * span)


def make_utility_figure(utility: pd.DataFrame, out_dir: Path) -> None:
    if utility.empty:
        raise SystemExit("No utility summaries found. Run utility jobs before making figures.")

    models = ordered_models(utility)
    contrasts = ["difficulty_base", "difficulty_praise", "praise_simple", "praise_hard"]
    values: dict[str, list[list[float]]] = {}
    for model in models:
        sub = utility[utility["model"] == model]
        values[model] = [sub[sub["contrast"] == c]["prob_treatment"].astype(float).tolist() for c in contrasts]

    fig, ax = plt.subplots(figsize=(5.5, 2.7))
    plot_grouped_boxplots(
        ax,
        groups=contrasts,
        group_labels=[CONTRAST_LABELS[c] for c in contrasts],
        models=models,
        values_by_model=values,
    )
    ax.axhline(0.5, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Treatment chosen")
    ax.set_title("Utility preferences from forced choices")
    save_figure(fig, out_dir, "utility_preferences")


def make_latent_utility_figure(scores: pd.DataFrame, out_dir: Path) -> bool:
    if scores.empty:
        return False
    needed = {"model", "difficulty", "framing", "utility_mu"}
    if not needed <= set(scores.columns):
        return False

    scores = scores.copy()
    scores["utility_mu"] = pd.to_numeric(scores["utility_mu"], errors="coerce")
    scores = scores.dropna(subset=["utility_mu", "difficulty", "framing", "model"])
    if scores.empty:
        return False

    models = ordered_models(scores)
    settings = [
        ("hard", "base"),
        ("hard", "praise"),
        ("simple", "base"),
        ("simple", "praise"),
    ]
    labels = ["Hard\nbase", "Hard\n+praise", "Easy\nbase", "Easy\n+praise"]
    values: dict[str, list[list[float]]] = {}
    for model in models:
        values[model] = []
        for difficulty, framing in settings:
            sub = scores[
                (scores["model"] == model)
                & (scores["difficulty"] == difficulty)
                & (scores["framing"] == framing)
            ]
            values[model].append(sub["utility_mu"].astype(float).tolist())

    fig, ax = plt.subplots(figsize=(5.5, 2.7))
    plot_grouped_boxplots(
        ax,
        groups=[f"{difficulty}_{framing}" for difficulty, framing in settings],
        group_labels=labels,
        models=models,
        values_by_model=values,
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Normalized latent utility by task condition")
    ax.set_ylabel("Utility (no zero point)")
    set_value_ylim(ax, values, baseline=0.0)
    save_figure(fig, out_dir, "normalized_latent_utility")
    return True


def make_downstream_figure(
    downstream: pd.DataFrame,
    out_dir: Path,
    *,
    metric: str,
    stem: str,
    title: str,
    ylabel: str,
) -> bool:
    if downstream.empty:
        return False

    metric_df = downstream[downstream["metric"] == metric]
    if metric_df.empty:
        return False

    models = ordered_models(metric_df)
    task_sets = ["hard", "easy"]
    task_labels = ["Hard tasks", "Easy tasks"]
    values: dict[str, list[list[float]]] = {}
    for model in models:
        sub = metric_df[metric_df["model"] == model]
        values[model] = [sub[sub["task_set"] == t]["value"].astype(float).tolist() for t in task_sets]

    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    plot_fn = plot_grouped_bars if metric == "pass01" else plot_grouped_boxplots
    plot_fn(
        ax,
        groups=task_sets,
        group_labels=task_labels,
        models=models,
        values_by_model=values,
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if metric == "pass01":
        set_bar_ylim(ax, values, baseline=0.0)
    else:
        set_value_ylim(ax, values, baseline=0.0)
    save_figure(fig, out_dir, stem)
    return True


def make_figures(utility: pd.DataFrame, downstream: pd.DataFrame, utility_scores: pd.DataFrame, out_dir: Path) -> list[str]:
    set_figure_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = ["utility_preferences.pdf", "utility_preferences.png"]
    make_utility_figure(utility, out_dir)
    if make_latent_utility_figure(utility_scores, out_dir):
        written.extend(["normalized_latent_utility.pdf", "normalized_latent_utility.png"])
    if make_downstream_figure(
        downstream,
        out_dir,
        metric="pass01",
        stem="praise_pass1",
        title="Does appreciation change Pass@1?",
        ylabel="Praise - base Pass@1",
    ):
        written.extend(["praise_pass1.pdf", "praise_pass1.png"])
    if make_downstream_figure(
        downstream,
        out_dir,
        metric="reasoning_tokens_rough",
        stem="praise_effort",
        title="Does appreciation change planning effort?",
        ylabel="Praise - base planning tokens",
    ):
        written.extend(["praise_effort.pdf", "praise_effort.png"])
    return written


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    utility, downstream, utility_scores = load_figure_data(Path(args.results_dir))
    written = make_figures(utility, downstream, utility_scores, Path(out_dir))
    print("Wrote figures:")
    for name in written:
        print(f"- {Path(out_dir) / name}")


if __name__ == "__main__":
    main()
