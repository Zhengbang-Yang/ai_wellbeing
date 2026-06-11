from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

from .config import DATA_DIR, RESULTS_DIR
from .utils import ensure_dir, read_jsonl, slugify, stable_hash, write_json, write_jsonl


SPECIAL_TOKEN_RE = re.compile(r"<\|[^>\n]+?\|>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated code on BigCodeBench tests.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tasks", default=str(DATA_DIR / "tasks_hard_100.jsonl"))
    parser.add_argument("--generations", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--result-subdir", default="downstream")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--tmp-dir", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def build_program(code: str, test: str, entry_point: str | None) -> str:
    code = SPECIAL_TOKEN_RE.sub("", code).rstrip()
    trailer = """

if __name__ == "__main__":
    import unittest
    if "check" in globals() and ENTRY_POINT_NAME in globals():
        check(globals()[ENTRY_POINT_NAME])
    else:
        unittest.main()
"""
    return (
        code.rstrip()
        + "\n\n"
        + str(test).strip()
        + "\n\nENTRY_POINT_NAME = "
        + repr(entry_point or "")
        + trailer
    )


def run_one(row: dict, task: dict, tmp_parent: Path, timeout: int) -> dict:
    program = build_program(row["extracted_code"], task.get("test") or "", task.get("entry_point"))
    tmp_parent = tmp_parent.resolve()
    tmp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bbc_", dir=str(tmp_parent)) as tmp:
        tmp_path = Path(tmp).resolve()
        path = tmp_path / f"{stable_hash(row['task_id'] + row['framing'])}.py"
        path.write_text(program, encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            passed = proc.returncode == 0
            status = "passed" if passed else "failed"
            stdout = proc.stdout[-4000:]
            stderr = proc.stderr[-4000:]
        except subprocess.TimeoutExpired as exc:
            passed = False
            status = "timeout"
            stdout = (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else ""
    return {
        "model_id": row["model_id"],
        "task_id": row["task_id"],
        "framing": row["framing"],
        "passed": passed,
        "status": status,
        "stdout_tail": stdout,
        "stderr_tail": stderr,
        "reasoning_tokens_rough": row.get("reasoning_tokens_rough", 0),
        "reasoning_completion_tokens": row.get("reasoning_completion_tokens"),
        "reasoning_hit_cap": row.get("reasoning_hit_cap"),
        "completion_tokens": row.get("completion_tokens"),
        "code_hit_cap": row.get("code_hit_cap"),
    }


def main() -> None:
    args = parse_args()
    model_slug = slugify(args.model_id)
    out_dir = ensure_dir(args.out_dir or (RESULTS_DIR / model_slug / args.result_subdir))
    suffix = "" if args.num_shards == 1 else f"_shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    generations_path = Path(args.generations or (Path(out_dir) / f"generations{suffix}.jsonl"))
    tasks = {row["task_id"]: row for row in read_jsonl(args.tasks)}
    generations = read_jsonl(generations_path)
    tmp_parent = Path(args.tmp_dir or (Path(out_dir) / "tmp"))

    results = []
    for row in tqdm(generations, desc="evaluate code"):
        task = tasks[row["task_id"]]
        results.append(run_one(row, task, tmp_parent, args.timeout))

    out_path = Path(out_dir) / f"eval_results{suffix}.jsonl"
    write_jsonl(out_path, results)
    write_json(
        Path(out_dir) / "eval_metadata.json",
        {
            "model_id": args.model_id,
            "tasks_path": str(args.tasks),
            "result_subdir": args.result_subdir,
            "n_generations": len(generations),
            "timeout_sec": args.timeout,
            "output": str(out_path),
            "note": "Local evaluator combines extracted code with BigCodeBench tests. Use official BigCodeBench evaluation for final paper numbers if cluster policy requires it.",
        },
    )
    print(f"Wrote evaluation results to {out_path}")


if __name__ == "__main__":
    main()
