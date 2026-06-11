from __future__ import annotations

import argparse
import random
from collections import Counter

from datasets import load_dataset

from .config import (
    BIGCODE_SPLIT,
    DATA_DIR,
    DEFAULT_SEED,
    FULL_DATASET,
    HARD_DATASET,
)
from .tasks import compact_task, sort_tasks, task_id
from .utils import ensure_dir, set_hf_cache, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample BigCodeBench hard/simple tasks.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split", default=BIGCODE_SPLIT)
    parser.add_argument("--n-hard", type=int, default=100)
    parser.add_argument("--n-simple", type=int, default=100)
    parser.add_argument("--out-dir", default=str(DATA_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_hf_cache()
    out_dir = ensure_dir(args.out_dir)
    rng = random.Random(args.seed)

    full = [dict(row) for row in load_dataset(FULL_DATASET, split=args.split)]
    hard = [dict(row) for row in load_dataset(HARD_DATASET, split=args.split)]
    full_by_id = {task_id(row): row for row in full}
    hard_by_id = {task_id(row): row for row in hard}

    missing_hard = sorted(set(hard_by_id) - set(full_by_id))
    if missing_hard:
        raise ValueError(f"{len(missing_hard)} hard task ids are absent from the full dataset.")

    simple_pool = [row for row in full if task_id(row) not in hard_by_id]
    hard_pool = sort_tasks(hard)
    simple_pool = sort_tasks(simple_pool)
    if len(hard_pool) < args.n_hard:
        raise ValueError(f"Need {args.n_hard} hard tasks but only found {len(hard_pool)}.")
    if len(simple_pool) < args.n_simple:
        raise ValueError(f"Need {args.n_simple} simple tasks but only found {len(simple_pool)}.")

    hard_sample = rng.sample(hard_pool, args.n_hard)
    simple_sample = rng.sample(simple_pool, args.n_simple)
    hard_tasks = [compact_task(row, "hard") for row in hard_sample]
    simple_tasks = [compact_task(row, "simple") for row in simple_sample]

    hard_for_pairing = hard_tasks[:]
    simple_for_pairing = simple_tasks[:]
    rng.shuffle(hard_for_pairing)
    rng.shuffle(simple_for_pairing)
    pairs = []
    for idx, (hrow, srow) in enumerate(zip(hard_for_pairing, simple_for_pairing)):
        pairs.append(
            {
                "pair_index": idx,
                "hard_task_id": hrow["task_id"],
                "simple_task_id": srow["task_id"],
                "seed": args.seed,
            }
        )

    write_jsonl(out_dir / "tasks_hard_100.jsonl", hard_tasks)
    write_jsonl(out_dir / "tasks_simple_100.jsonl", simple_tasks)
    write_jsonl(out_dir / "task_pairs_difficulty.jsonl", pairs)
    write_json(
        out_dir / "sampling_metadata.json",
        {
            "seed": args.seed,
            "split": args.split,
            "full_dataset": FULL_DATASET,
            "hard_dataset": HARD_DATASET,
            "full_count": len(full),
            "hard_count": len(hard_pool),
            "simple_pool_count": len(simple_pool),
            "sampled_hard": len(hard_tasks),
            "sampled_simple": len(simple_tasks),
            "sampled_difficulty_counts": dict(Counter(t["difficulty"] for t in hard_tasks + simple_tasks)),
            "hard_ids": sorted(t["task_id"] for t in hard_tasks),
            "simple_ids": sorted(t["task_id"] for t in simple_tasks),
        },
    )
    print(f"Wrote sampled tasks and {len(pairs)} hard/simple pairs to {out_dir}")


if __name__ == "__main__":
    main()

