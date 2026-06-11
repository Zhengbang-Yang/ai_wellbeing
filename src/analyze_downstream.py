from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_SEED, RESULTS_DIR
from .utils import bootstrap_mean_ci, ensure_dir, read_jsonl, slugify, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze downstream base-vs-praise effects.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--eval-results", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--result-subdir", default="downstream")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def load_eval_rows(results_path: Path) -> list[dict]:
    shard_paths = sorted(results_path.parent.glob("eval_results_shard_*_of_*.jsonl"))
    if shard_paths:
        rows = []
        for path in shard_paths:
            rows.extend(read_jsonl(path))
        return rows
    return read_jsonl(results_path)


def paired_summary(df: pd.DataFrame, metric: str, seed: int) -> dict:
    pivot = df.pivot_table(index="task_id", columns="framing", values=metric, aggfunc="first").dropna()
    if pivot.empty:
        return {"metric": metric, "n_pairs": 0}
    diff = (pivot["praise"] - pivot["base"]).astype(float).tolist()
    mean, lo, hi = bootstrap_mean_ci(diff, seed=seed)
    return {
        "metric": metric,
        "n_pairs": len(diff),
        "base_mean": float(pivot["base"].mean()),
        "praise_mean": float(pivot["praise"].mean()),
        "paired_diff_praise_minus_base": mean,
        "ci_low": lo,
        "ci_high": hi,
    }


def main() -> None:
    args = parse_args()
    model_slug = slugify(args.model_id)
    out_dir = ensure_dir(args.out_dir or (RESULTS_DIR / model_slug / args.result_subdir))
    results_path = Path(args.eval_results or (Path(out_dir) / "eval_results.jsonl"))
    df = pd.DataFrame(load_eval_rows(results_path))
    df["pass01"] = df["passed"].astype(float)
    df["reasoning_tokens_rough"] = pd.to_numeric(df["reasoning_tokens_rough"], errors="coerce").fillna(0)
    df["completion_tokens"] = pd.to_numeric(df["completion_tokens"], errors="coerce")
    for cap_col in ["reasoning_hit_cap", "code_hit_cap"]:
        if cap_col in df:
            df[cap_col] = df[cap_col].fillna(False).astype(float)

    summaries = [
        paired_summary(df, "pass01", args.seed),
        paired_summary(df, "reasoning_tokens_rough", args.seed),
    ]
    if df["completion_tokens"].notna().any():
        summaries.append(paired_summary(df, "completion_tokens", args.seed))
    for cap_col in ["reasoning_hit_cap", "code_hit_cap"]:
        if cap_col in df:
            summaries.append(paired_summary(df, cap_col, args.seed))

    paired = []
    piv_pass = df.pivot_table(index="task_id", columns="framing", values="pass01", aggfunc="first")
    piv_effort = df.pivot_table(index="task_id", columns="framing", values="reasoning_tokens_rough", aggfunc="first")
    for task_id in sorted(set(piv_pass.index) | set(piv_effort.index)):
        row = {"task_id": task_id}
        if task_id in piv_pass.index and {"base", "praise"} <= set(piv_pass.columns):
            row["base_pass"] = piv_pass.loc[task_id].get("base")
            row["praise_pass"] = piv_pass.loc[task_id].get("praise")
        if task_id in piv_effort.index and {"base", "praise"} <= set(piv_effort.columns):
            row["base_reasoning_tokens"] = piv_effort.loc[task_id].get("base")
            row["praise_reasoning_tokens"] = piv_effort.loc[task_id].get("praise")
        paired.append(row)

    pd.DataFrame(summaries).to_csv(Path(out_dir) / "downstream_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(Path(out_dir) / "downstream_paired_by_task.csv", index=False)
    write_json(Path(out_dir) / "summary.json", {"model_id": args.model_id, "summaries": summaries})
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
