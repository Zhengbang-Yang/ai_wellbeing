from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from tqdm import tqdm

from .config import DATA_DIR, DEFAULT_SEED, RESULTS_DIR
from .model_backends import load_backend
from .prompts import downstream_prompt, downstream_reasoning_prompt, strip_reasoning
from .utils import append_jsonl, count_tokens_rough, ensure_dir, read_jsonl, seed_everything, set_hf_cache, slugify, write_json


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
FINAL_CODE_RE = re.compile(r"final\s+code\s*:?", re.IGNORECASE)
REASONING_RE = re.compile(r"reasoning\s*:?", re.IGNORECASE)
END_PLAN_RE = re.compile(r"(?:^|\n)\s*END_PLAN\s*(?:\n|$)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BigCodeBench solutions for base/praise prompts.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--backend", choices=["hf", "openai"], default="hf")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--tasks", default=str(DATA_DIR / "tasks_hard_100.jsonl"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--reasoning-max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true", default=True)
    parser.add_argument("--disable-thinking", action="store_false", dest="enable_thinking")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--result-subdir", default="downstream")
    return parser.parse_args()


def extract_code(text: str) -> str:
    _, visible = strip_reasoning(text)
    parts = FINAL_CODE_RE.split(visible, maxsplit=1)
    if len(parts) == 2:
        code_part = parts[1].strip()
        matches = CODE_BLOCK_RE.findall(code_part)
        if matches:
            return max((match.strip() for match in matches), key=len)
        return code_part.replace("```python", "").replace("```", "").strip()
    matches = CODE_BLOCK_RE.findall(visible)
    if matches:
        return max((match.strip() for match in matches), key=len)
    cleaned = visible.strip()
    cleaned = cleaned.replace("```python", "").replace("```", "").strip()
    return cleaned


def extract_structured_reasoning(text: str) -> str:
    reasoning, visible = strip_reasoning(text)
    if reasoning:
        return END_PLAN_RE.split(reasoning, maxsplit=1)[0].strip()
    parts = FINAL_CODE_RE.split(visible, maxsplit=1)
    before_code = parts[0] if parts else visible
    before_code = REASONING_RE.sub("", before_code, count=1).strip()
    return END_PLAN_RE.split(before_code, maxsplit=1)[0].strip()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_hf_cache()
    model_slug = slugify(args.model_id)
    out_dir = ensure_dir(args.out_dir or (RESULTS_DIR / model_slug / args.result_subdir))
    tasks = read_jsonl(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.num_shards > 1:
        tasks = [task for idx, task in enumerate(tasks) if idx % args.num_shards == args.shard_index]

    backend = load_backend(
        args.backend,
        args.model_id,
        base_url=args.base_url,
        api_key=args.api_key,
        dtype=args.dtype,
        device_map=args.device_map,
    )
    suffix = "" if args.num_shards == 1 else f"_shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    out_path = Path(out_dir) / f"generations{suffix}.jsonl"
    if out_path.exists():
        out_path.unlink()
    rows_written = 0
    started = time.time()
    for task in tqdm(tasks, desc="downstream tasks"):
        for framing in ["base", "praise"]:
            messages = downstream_prompt(task, framing)
            starter = str(task.get("code_prompt") or "").strip()
            code_prefix = f"Final code:\n{starter}\n" if starter else "Final code:\n"
            reasoning_result = backend.generate(
                downstream_reasoning_prompt(task, framing),
                max_new_tokens=args.reasoning_max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                presence_penalty=args.presence_penalty,
                repetition_penalty=args.repetition_penalty,
                enable_thinking=args.enable_thinking,
                stop_regex=r"(?:^|\n)\s*END_PLAN\s*(?:\n|$)",
                assistant_prefix="Reasoning plan:\n",
            )
            result = backend.generate(
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                presence_penalty=args.presence_penalty,
                repetition_penalty=args.repetition_penalty,
                enable_thinking=args.enable_thinking,
                assistant_prefix=code_prefix,
            )
            text = result["text"]
            full_code_text = code_prefix + text
            reasoning_text = reasoning_result["text"]
            reasoning, visible = strip_reasoning(text)
            reasoning_content = (
                reasoning_result.get("reasoning_content")
                or strip_reasoning(reasoning_text)[0]
                or extract_structured_reasoning(reasoning_text)
            )
            extracted = extract_code(full_code_text)
            append_jsonl(
                out_path,
                [
                    {
                        "model_id": args.model_id,
                        "task_id": task["task_id"],
                        "framing": framing,
                        "raw_generation": text,
                        "prefilled_code_prefix": code_prefix,
                        "full_code_generation": full_code_text,
                        "visible_generation": visible,
                        "raw_reasoning_generation": reasoning_text,
                        "reasoning_content": reasoning_content,
                        "extracted_code": extracted,
                        "reasoning_tokens_rough": count_tokens_rough(reasoning_content) if reasoning_content else 0,
                        "reasoning_completion_tokens": reasoning_result.get("completion_tokens"),
                        "reasoning_hit_cap": reasoning_result.get("completion_tokens") == args.reasoning_max_new_tokens,
                        "prompt_tokens": result.get("prompt_tokens"),
                        "completion_tokens": result.get("completion_tokens"),
                        "code_hit_cap": result.get("completion_tokens") == args.max_new_tokens,
                        "seed": args.seed,
                    }
                ],
            )
            rows_written += 1
    write_json(
        Path(out_dir) / "generation_metadata.json",
        {
            "model_id": args.model_id,
            "backend": args.backend,
            "n_tasks": len(tasks),
            "tasks_path": str(args.tasks),
            "result_subdir": args.result_subdir,
            "n_generations": rows_written,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "max_new_tokens": args.max_new_tokens,
            "reasoning_max_new_tokens": args.reasoning_max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "enable_thinking": args.enable_thinking,
            "elapsed_sec": time.time() - started,
            "output": str(out_path),
        },
    )
    print(f"Wrote {rows_written} generations to {out_path}")


if __name__ == "__main__":
    main()
