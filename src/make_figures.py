from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch

from .config import FIGURES_DIR, RESULTS_DIR
from .utils import ensure_dir

MODEL_LABELS = {
    "qwen_qwen3_5_0_8b": "Qwen3.5-0.8B",
    "qwen_qwen3_5_2b": "Qwen3.5-2B",
    "qwen_qwen3_5_4b": "Qwen3.5-4B",
    "qwen_qwen3_5_9b": "Qwen3.5-9B",
    "qwen_qwen3_5_27b": "Qwen3.5-27B",
}

MODEL_ORDER = {
    "qwen_qwen3_5_0_8b": 0.8,
    "qwen_qwen3_5_2b": 2,
    "qwen_qwen3_5_4b": 4,
    "qwen_qwen3_5_9b": 9,
    "qwen_qwen3_5_27b": 27,
}

COLORS = ["#4F7AB0", "#D9828B", "#5E8C61", "#8A6FB0", "#C4904A"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make paper-style figures from completed runs.")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    return parser.parse_args()


def load_figure_data(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    utility_score_rows = []
    downstream_effect_rows = []
    for model_dir in sorted(results_dir.iterdir()) if results_dir.exists() else []:
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
    return pd.DataFrame(downstream_effect_rows), pd.DataFrame(utility_score_rows)


def set_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ordered_models(*frames: pd.DataFrame) -> list[str]:
    models: set[str] = set()
    for frame in frames:
        if not frame.empty and "model" in frame.columns:
            models.update(str(model) for model in frame["model"].dropna().unique())
    return sorted(models, key=lambda model: (MODEL_ORDER.get(model, float("inf")), MODEL_LABELS.get(model, model)))


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout(pad=0.8)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def grouped_bar_width(n_models: int) -> float:
    return min(0.28, 0.72 / max(1, n_models))


def mean_grouped_values(values_by_model: dict[str, list[list[float]]]) -> dict[str, list[float]]:
    means: dict[str, list[float]] = {}
    for model, grouped_values in values_by_model.items():
        means[model] = []
        for values in grouped_values:
            clean = [float(v) for v in values if np.isfinite(v)]
            if clean:
                mean = float(np.mean(clean))
            else:
                mean = np.nan
            means[model].append(mean)
    return means


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
    means = mean_grouped_values(values_by_model)
    for i, model in enumerate(models):
        vals = np.asarray(means.get(model, [np.nan] * len(groups)), dtype=float)
        offset = (i - (len(models) - 1) / 2) * width
        color = COLORS[i % len(COLORS)]
        ax.bar(x + offset, vals, width=width, color=color, label=MODEL_LABELS.get(model, model))
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.04)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(3, max(1, len(models))),
        handlelength=1.4,
        columnspacing=1.2,
    )


def grouped_box_width(n_models: int) -> float:
    return min(0.28, 0.72 / max(1, n_models))


def plot_grouped_boxplots(
    ax: plt.Axes,
    *,
    groups: list[str],
    group_labels: list[str],
    models: list[str],
    values_by_model: dict[str, list[list[float]]],
    showfliers: bool = True,
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
                showfliers=showfliers,
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
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(3, max(1, len(models))),
        handlelength=1.4,
        columnspacing=1.2,
    )


def set_bar_ylim(ax: plt.Axes, values_by_model: dict[str, list[list[float]]], *, baseline: float = 0.0) -> None:
    means = mean_grouped_values(values_by_model)
    vals = [baseline]
    for grouped_values in means.values():
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


def set_percentile_ylim(
    ax: plt.Axes,
    values_by_model: dict[str, list[list[float]]],
    *,
    baseline: float = 0.0,
    low_q: float = 1.0,
    high_q: float = 99.0,
) -> None:
    vals = [baseline]
    for grouped_values in values_by_model.values():
        for values in grouped_values:
            vals.extend(float(v) for v in values if np.isfinite(v))
    lo, hi = np.percentile(vals, [low_q, high_q])
    lo = min(float(lo), baseline)
    hi = max(float(hi), baseline)
    span = max(hi - lo, 0.05)
    ax.set_ylim(lo - 0.12 * span, hi + 0.12 * span)


def set_trimmed_symmetric_ylim(
    ax: plt.Axes,
    values_by_model: dict[str, list[list[float]]],
    *,
    baseline: float = 0.0,
    low_q: float = 2.0,
    high_q: float = 98.0,
    min_limit: float = 20.0,
) -> None:
    vals = [baseline]
    for grouped_values in values_by_model.values():
        for values in grouped_values:
            vals.extend(float(v) for v in values if np.isfinite(v))
    lo, hi = np.percentile(vals, [low_q, high_q])
    limit = max(abs(float(lo)), abs(float(hi)), min_limit)
    ax.set_ylim(-1.15 * limit, 1.15 * limit)


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
        ("simple", "base"),
        ("hard", "base"),
        ("simple", "praise"),
        ("hard", "praise"),
    ]
    labels = ["Easy\nbase", "Hard\nbase", "Easy\n+praise", "Hard\n+praise"]
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
        showfliers=False,
    )
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Utility")
    set_percentile_ylim(ax, values, baseline=0.0)
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
    showfliers: bool = True,
) -> bool:
    if downstream.empty:
        return False

    metric_df = downstream[downstream["metric"] == metric]
    if metric_df.empty:
        return False

    models = ordered_models(metric_df)
    task_sets = ["easy", "hard"]
    task_labels = ["Easy tasks", "Hard tasks"]
    values: dict[str, list[list[float]]] = {}
    for model in models:
        sub = metric_df[metric_df["model"] == model]
        values[model] = [sub[sub["task_set"] == t]["value"].astype(float).tolist() for t in task_sets]

    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    if metric == "pass01":
        plot_grouped_bars(
            ax,
            groups=task_sets,
            group_labels=task_labels,
            models=models,
            values_by_model=values,
        )
    else:
        plot_grouped_boxplots(
            ax,
            groups=task_sets,
            group_labels=task_labels,
            models=models,
            values_by_model=values,
            showfliers=showfliers,
        )
    zero_linestyle = "--" if stem == "praise_effort" else "-"
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle=zero_linestyle)
    ax.set_ylabel(ylabel)
    if metric == "pass01":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        set_bar_ylim(ax, values, baseline=0.0)
    elif not showfliers:
        if stem == "praise_effort":
            ax.set_ylim(-100, 100)
        else:
            set_trimmed_symmetric_ylim(ax, values, baseline=0.0)
    else:
        set_value_ylim(ax, values, baseline=0.0)
    save_figure(fig, out_dir, stem)
    return True


def make_figures(downstream: pd.DataFrame, utility_scores: pd.DataFrame, out_dir: Path) -> list[str]:
    set_figure_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
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
        showfliers=False,
    ):
        written.extend(["praise_effort.pdf", "praise_effort.png"])
    return written


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    downstream, utility_scores = load_figure_data(Path(args.results_dir))
    written = make_figures(downstream, utility_scores, Path(out_dir))
    print("Wrote figures:")
    for name in written:
        print(f"- {Path(out_dir) / name}")


if __name__ == "__main__":
    main()
