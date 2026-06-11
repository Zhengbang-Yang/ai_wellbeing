from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .config import DEFAULT_SEED, RESULTS_DIR
from .utils import bootstrap_mean_ci, ensure_dir, read_jsonl, slugify, write_json


FACTORIAL_CONTRASTS = [
    "difficulty_base",
    "difficulty_praise",
    "praise_hard",
    "praise_simple",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze utility comparison results.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--utility-raw", default=None)
    parser.add_argument("--items", default="data/utility_items.jsonl")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--result-subdir", default="utility")
    parser.add_argument(
        "--template-id",
        default="experienced",
        help="Preference template to analyze. Use 'all' to reproduce the legacy mixed-template analysis.",
    )
    parser.add_argument(
        "--no-order-average",
        action="store_true",
        help="Fit raw A/B rows directly instead of first averaging forward and reverse orderings.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fit-epochs", type=int, default=1000)
    return parser.parse_args()


def load_raw_rows(raw_path: Path) -> list[dict[str, Any]]:
    shard_paths = sorted(raw_path.parent.glob("utility_raw_shard_*_of_*.jsonl"))
    if shard_paths:
        rows: list[dict[str, Any]] = []
        for path in shard_paths:
            rows.extend(read_jsonl(path))
        return rows
    return read_jsonl(raw_path)


def filter_template(df: pd.DataFrame, template_id: str) -> pd.DataFrame:
    if template_id == "all":
        return df.copy()
    return df[df["template_id"] == template_id].copy()


def order_average_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_id, group in df.groupby("comparison_group_id", sort=False):
        first = group.iloc[0].to_dict()
        prob_treatment = pd.to_numeric(group["prob_treatment"], errors="coerce").mean()
        rows.append(
            {
                "comparison_id": group_id,
                "comparison_group_id": group_id,
                "contrast": first["contrast"],
                "unit_id": first["unit_id"],
                "template_id": first["template_id"],
                "order": "order_averaged",
                "item_a": first["treatment_item_id"],
                "item_b": first["control_item_id"],
                "treatment_item_id": first["treatment_item_id"],
                "control_item_id": first["control_item_id"],
                "model_id": first.get("model_id"),
                "backend": first.get("backend"),
                "seed": first.get("seed"),
                "logp_a": np.nan,
                "logp_b": np.nan,
                "prob_a": float(prob_treatment),
                "choice": "order_averaged",
                "prob_treatment": float(prob_treatment),
                "treatment_chosen": bool(prob_treatment >= 0.5),
                "raw_text": "",
                "n_order_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def summarize_contrasts(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for contrast in FACTORIAL_CONTRASTS:
        sub = df[df["contrast"] == contrast].copy()
        by_unit = sub.groupby(["unit_id", "template_id"])["prob_treatment"].mean().reset_index()
        values = by_unit["prob_treatment"].tolist()
        mean, lo, hi = bootstrap_mean_ci(values, seed=seed)
        rows.append(
            {
                "contrast": contrast,
                "n_units_x_templates": len(values),
                "mean_p_treatment": mean,
                "ci_low": lo,
                "ci_high": hi,
                "win_rate": float(np.mean(np.asarray(values) > 0.5)) if values else np.nan,
                "mean_margin_from_indifference": mean - 0.5,
            }
        )
    return pd.DataFrame(rows)


def order_consistency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for contrast, sub in df[df["contrast"].isin(FACTORIAL_CONTRASTS)].groupby("contrast"):
        decisions = []
        for _, group in sub.groupby("comparison_group_id"):
            if len(group) < 2:
                continue
            decisions.append(bool((group["prob_treatment"] > 0.5).all() or (group["prob_treatment"] < 0.5).all()))
        rows.append(
            {
                "contrast": contrast,
                "n_groups": len(decisions),
                "order_consistency": float(np.mean(decisions)) if decisions else np.nan,
            }
        )
    return pd.DataFrame(rows)


def fit_thurstonian(df: pd.DataFrame, item_ids: list[str], out_dir: Path, epochs: int) -> pd.DataFrame:
    import torch

    rng = np.random.default_rng(0)
    idx = {item_id: i for i, item_id in enumerate(item_ids)}
    a = torch.tensor([idx[x] for x in df["item_a"]], dtype=torch.long)
    b = torch.tensor([idx[x] for x in df["item_b"]], dtype=torch.long)
    y = torch.tensor(df["prob_a"].astype(float).to_numpy(), dtype=torch.float32)
    n = len(item_ids)
    mask = rng.random(len(df)) < 0.8
    train = torch.tensor(mask, dtype=torch.bool)
    holdout = ~train

    mu = torch.zeros(n, requires_grad=True)
    log_sigma = torch.zeros(n, requires_grad=True)
    opt = torch.optim.Adam([mu, log_sigma], lr=0.01)
    eps = 1e-6
    for _ in range(epochs):
        opt.zero_grad()
        sigma2 = torch.exp(2 * log_sigma)
        denom = torch.sqrt(sigma2[a] + sigma2[b] + eps)
        z = (mu[a] - mu[b]) / denom
        pred = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
        pred = torch.clamp(pred, eps, 1 - eps)
        loss = torch.nn.functional.binary_cross_entropy(pred[train], y[train])
        loss = loss + 1e-4 * (mu.pow(2).mean() + log_sigma.pow(2).mean())
        loss.backward()
        opt.step()
        with torch.no_grad():
            mu -= mu.mean()

    with torch.no_grad():
        sigma2 = torch.exp(2 * log_sigma)
        denom = torch.sqrt(sigma2[a] + sigma2[b] + eps)
        z = (mu[a] - mu[b]) / denom
        pred = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
        train_decidable = train & (torch.abs(y - 0.5) > 1e-6)
        holdout_decidable = holdout & (torch.abs(y - 0.5) > 1e-6)
        if holdout_decidable.any():
            acc = ((pred[holdout_decidable] > 0.5) == (y[holdout_decidable] > 0.5)).float().mean().item()
        else:
            acc = float("nan")
        if train_decidable.any():
            train_acc = ((pred[train_decidable] > 0.5) == (y[train_decidable] > 0.5)).float().mean().item()
        else:
            train_acc = float("nan")
        holdout_bce = torch.nn.functional.binary_cross_entropy(pred[holdout], y[holdout]).item() if holdout.any() else float("nan")
        train_bce = torch.nn.functional.binary_cross_entropy(pred[train], y[train]).item() if train.any() else float("nan")
        mu_np = mu.detach().cpu().numpy()
        sigma_np = torch.exp(log_sigma).detach().cpu().numpy()
        scale = np.std(mu_np) or 1.0
        mu_norm = (mu_np - np.mean(mu_np)) / scale
        sigma_norm = sigma_np / scale

    torch.save(
        {
            "item_ids": item_ids,
            "mu": mu_norm,
            "sigma": sigma_norm,
            "train_accuracy": train_acc,
            "holdout_accuracy": acc,
            "train_bce": train_bce,
            "holdout_bce": holdout_bce,
        },
        out_dir / "utility_fit.pt",
    )
    fit_df = pd.DataFrame({"item_id": item_ids, "utility_mu": mu_norm, "utility_sigma": sigma_norm})
    fit_df.to_csv(out_dir / "utility_scores.csv", index=False)
    write_json(
        out_dir / "utility_fit_metrics.json",
        {
            "train_accuracy": train_acc,
            "holdout_accuracy": acc,
            "train_bce": train_bce,
            "holdout_bce": holdout_bce,
            "n_fit_rows": int(len(df)),
            "n_train_rows": int(train.sum().item()),
            "n_holdout_rows": int(holdout.sum().item()),
            "n_train_decidable": int(train_decidable.sum().item()),
            "n_holdout_decidable": int(holdout_decidable.sum().item()),
        },
    )
    return fit_df


def bias_checks(fit_df: pd.DataFrame, items: list[dict[str, Any]], out_dir: Path) -> pd.DataFrame:
    item_df = pd.DataFrame(
        [
            {
                "item_id": item["item_id"],
                "task_id": item["task_id"],
                "difficulty": item["difficulty"],
                "framing": item["framing"],
                "prompt_chars": item["task"].get("prompt_chars", np.nan),
                "code_prompt_chars": item["task"].get("code_prompt_chars", np.nan),
                "test_chars": item["task"].get("test_chars", np.nan),
                "libs_count": item["task"].get("libs_count", np.nan),
            }
            for item in items
        ]
    )
    merged = fit_df.merge(item_df, on="item_id", how="left")
    merged.to_csv(out_dir / "utility_scores_with_features.csv", index=False)
    rows = []
    for feature in ["prompt_chars", "code_prompt_chars", "test_chars", "libs_count"]:
        clean = merged[["utility_mu", feature]].dropna()
        if len(clean) >= 3 and clean[feature].nunique() > 1:
            r, p = pearsonr(clean[feature], clean["utility_mu"])
        else:
            r, p = np.nan, np.nan
        rows.append({"feature": feature, "pearson_r_with_utility": r, "p_value": p, "n": len(clean)})
    checks = pd.DataFrame(rows)
    checks.to_csv(out_dir / "bias_checks.csv", index=False)
    return checks


def main() -> None:
    args = parse_args()
    model_slug = slugify(args.model_id)
    out_dir = ensure_dir(args.out_dir or (RESULTS_DIR / model_slug / args.result_subdir))
    raw_path = Path(args.utility_raw or (out_dir / "utility_raw.jsonl"))
    raw_df = pd.DataFrame(load_raw_rows(raw_path))
    raw_df = filter_template(raw_df, args.template_id)
    if raw_df.empty:
        raise SystemExit(f"No utility rows found for template_id={args.template_id!r}.")
    df = raw_df.copy() if args.no_order_average else order_average_comparisons(raw_df)
    items = read_jsonl(args.items)

    summary = summarize_contrasts(df, args.seed)
    consistency = order_consistency(raw_df)
    positional_bias = {
        "mean_prob_a": float(raw_df["prob_a"].mean()),
        "mean_a_margin_from_0_5": float(raw_df["prob_a"].mean() - 0.5),
        "n_raw_rows": int(len(raw_df)),
        "n_fit_rows": int(len(df)),
    }
    item_ids = sorted({row["item_id"] for row in items})
    fit_df = fit_thurstonian(df, item_ids, Path(out_dir), args.fit_epochs)
    checks = bias_checks(fit_df, items, Path(out_dir))

    summary.to_csv(Path(out_dir) / "utility_summary.csv", index=False)
    consistency.to_csv(Path(out_dir) / "order_consistency.csv", index=False)
    write_json(
        Path(out_dir) / "summary.json",
        {
            "model_id": args.model_id,
            "template_id": args.template_id,
            "order_averaged": not args.no_order_average,
            "contrasts": summary.to_dict(orient="records"),
            "order_consistency": consistency.to_dict(orient="records"),
            "positional_bias": positional_bias,
            "bias_checks": checks.to_dict(orient="records"),
        },
    )
    print(summary.to_string(index=False))
    print(f"Wrote utility analysis to {out_dir}")


if __name__ == "__main__":
    main()
