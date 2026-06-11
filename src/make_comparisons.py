from __future__ import annotations

import argparse
import itertools
import random
from typing import Any

from .config import DATA_DIR, DEFAULT_SEED
from .utils import read_jsonl, stable_hash, write_json, write_jsonl


FRAMINGS = ["base", "praise"]
TEMPLATE_IDS = ["experienced"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build utility items and comparisons.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--templates", nargs="+", default=TEMPLATE_IDS)
    parser.add_argument(
        "--random-edges",
        type=int,
        default=800,
        help="Extra cross-item edges to connect the Thurstonian graph.",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=100,
        help="Also write paired hard/easy test-set utility files using the first N fixed pairs.",
    )
    return parser.parse_args()


def item_id(task_id: str, difficulty: str, framing: str) -> str:
    return f"{difficulty}:{task_id}:{framing}"


def make_item(task: dict[str, Any], framing: str) -> dict[str, Any]:
    return {
        "item_id": item_id(task["task_id"], task["difficulty"], framing),
        "task_id": task["task_id"],
        "difficulty": task["difficulty"],
        "framing": framing,
        "task": task,
    }


def add_comparison(
    rows: list[dict[str, Any]],
    *,
    contrast: str,
    unit_id: str,
    template_id: str,
    treatment_item_id: str,
    control_item_id: str,
    seed: int,
) -> None:
    canonical = f"{contrast}|{unit_id}|{template_id}|{treatment_item_id}|{control_item_id}"
    for order_name, left, right in [
        ("treatment_left", treatment_item_id, control_item_id),
        ("treatment_right", control_item_id, treatment_item_id),
    ]:
        rows.append(
            {
                "comparison_id": stable_hash(f"{canonical}|{order_name}"),
                "comparison_group_id": stable_hash(canonical),
                "contrast": contrast,
                "unit_id": unit_id,
                "template_id": template_id,
                "order": order_name,
                "item_a": left,
                "item_b": right,
                "treatment_item_id": treatment_item_id,
                "control_item_id": control_item_id,
                "seed": seed,
            }
        )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    data_dir = DATA_DIR if args.data_dir == str(DATA_DIR) else args.data_dir
    hard_tasks = read_jsonl(f"{data_dir}/tasks_hard_100.jsonl")
    simple_tasks = read_jsonl(f"{data_dir}/tasks_simple_100.jsonl")
    pairs = read_jsonl(f"{data_dir}/task_pairs_difficulty.jsonl")
    by_id = {task["task_id"]: task for task in hard_tasks + simple_tasks}

    items = []
    for task in hard_tasks + simple_tasks:
        for framing in FRAMINGS:
            items.append(make_item(task, framing))
    valid_item_ids = {item["item_id"] for item in items}

    comparisons: list[dict[str, Any]] = []
    for pair in pairs:
        hard = by_id[pair["hard_task_id"]]
        simple = by_id[pair["simple_task_id"]]
        for template_id in args.templates:
            add_comparison(
                comparisons,
                contrast="difficulty_base",
                unit_id=f"pair:{pair['pair_index']}",
                template_id=template_id,
                treatment_item_id=item_id(hard["task_id"], "hard", "base"),
                control_item_id=item_id(simple["task_id"], "simple", "base"),
                seed=args.seed,
            )
            add_comparison(
                comparisons,
                contrast="difficulty_praise",
                unit_id=f"pair:{pair['pair_index']}",
                template_id=template_id,
                treatment_item_id=item_id(hard["task_id"], "hard", "praise"),
                control_item_id=item_id(simple["task_id"], "simple", "praise"),
                seed=args.seed,
            )
            add_comparison(
                comparisons,
                contrast="praise_hard",
                unit_id=f"task:{hard['task_id']}",
                template_id=template_id,
                treatment_item_id=item_id(hard["task_id"], "hard", "praise"),
                control_item_id=item_id(hard["task_id"], "hard", "base"),
                seed=args.seed,
            )
            add_comparison(
                comparisons,
                contrast="praise_simple",
                unit_id=f"task:{simple['task_id']}",
                template_id=template_id,
                treatment_item_id=item_id(simple["task_id"], "simple", "praise"),
                control_item_id=item_id(simple["task_id"], "simple", "base"),
                seed=args.seed,
            )

    all_pairs = list(itertools.combinations(sorted(valid_item_ids), 2))
    rng.shuffle(all_pairs)
    for idx, (left, right) in enumerate(all_pairs[: args.random_edges]):
        template_id = rng.choice(args.templates)
        add_comparison(
            comparisons,
            contrast="random_graph",
            unit_id=f"random:{idx}",
            template_id=template_id,
            treatment_item_id=left,
            control_item_id=right,
            seed=args.seed,
        )

    write_jsonl(f"{data_dir}/utility_items.jsonl", items)
    write_jsonl(f"{data_dir}/utility_comparisons.jsonl", comparisons)
    write_json(
        f"{data_dir}/comparison_metadata.json",
        {
            "seed": args.seed,
            "templates": args.templates,
            "n_items": len(items),
            "n_comparisons": len(comparisons),
            "n_factorial_comparisons": len(pairs) * len(args.templates) * 4 * 2,
            "n_random_comparison_rows": args.random_edges * 2,
            "framing_levels": FRAMINGS,
            "difficulty_levels": ["simple", "hard"],
        },
    )
    if args.test_limit and args.test_limit > 0:
        test_pairs = pairs[: args.test_limit]
        test_hard_tasks = [by_id[pair["hard_task_id"]] for pair in test_pairs]
        test_simple_tasks = [by_id[pair["simple_task_id"]] for pair in test_pairs]
        test_items = []
        for task in test_hard_tasks + test_simple_tasks:
            for framing in FRAMINGS:
                test_items.append(make_item(task, framing))

        test_comparisons: list[dict[str, Any]] = []
        for pair in test_pairs:
            hard = by_id[pair["hard_task_id"]]
            simple = by_id[pair["simple_task_id"]]
            for template_id in args.templates:
                add_comparison(
                    test_comparisons,
                    contrast="difficulty_base",
                    unit_id=f"pair:{pair['pair_index']}",
                    template_id=template_id,
                    treatment_item_id=item_id(hard["task_id"], "hard", "base"),
                    control_item_id=item_id(simple["task_id"], "simple", "base"),
                    seed=args.seed,
                )
                add_comparison(
                    test_comparisons,
                    contrast="difficulty_praise",
                    unit_id=f"pair:{pair['pair_index']}",
                    template_id=template_id,
                    treatment_item_id=item_id(hard["task_id"], "hard", "praise"),
                    control_item_id=item_id(simple["task_id"], "simple", "praise"),
                    seed=args.seed,
                )
                add_comparison(
                    test_comparisons,
                    contrast="praise_hard",
                    unit_id=f"task:{hard['task_id']}",
                    template_id=template_id,
                    treatment_item_id=item_id(hard["task_id"], "hard", "praise"),
                    control_item_id=item_id(hard["task_id"], "hard", "base"),
                    seed=args.seed,
                )
                add_comparison(
                    test_comparisons,
                    contrast="praise_simple",
                    unit_id=f"task:{simple['task_id']}",
                    template_id=template_id,
                    treatment_item_id=item_id(simple["task_id"], "simple", "praise"),
                    control_item_id=item_id(simple["task_id"], "simple", "base"),
                    seed=args.seed,
                )

        write_jsonl(f"{data_dir}/tasks_hard_test.jsonl", test_hard_tasks)
        write_jsonl(f"{data_dir}/tasks_simple_test.jsonl", test_simple_tasks)
        write_jsonl(f"{data_dir}/tasks_hard_test_{args.test_limit}.jsonl", test_hard_tasks)
        write_jsonl(f"{data_dir}/tasks_simple_test_{args.test_limit}.jsonl", test_simple_tasks)
        write_jsonl(f"{data_dir}/utility_items_test.jsonl", test_items)
        write_jsonl(f"{data_dir}/utility_comparisons_test.jsonl", test_comparisons)
        write_json(
            f"{data_dir}/comparison_metadata_test.json",
            {
                "seed": args.seed,
                "templates": args.templates,
                "test_limit_pairs": args.test_limit,
                "n_items": len(test_items),
                "n_comparisons": len(test_comparisons),
                "n_factorial_comparisons": len(test_pairs) * len(args.templates) * 4 * 2,
                "n_random_comparison_rows": 0,
                "framing_levels": FRAMINGS,
                "difficulty_levels": ["simple", "hard"],
                "note": "Test-set utility files use only the first fixed hard/simple pairs and omit random graph edges.",
            },
        )
    print(f"Wrote {len(items)} items and {len(comparisons)} comparison rows.")


if __name__ == "__main__":
    main()
