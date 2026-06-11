from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path

from tqdm import tqdm

from .config import DATA_DIR, DEFAULT_SEED, RESULTS_DIR
from .model_backends import ABScore, load_backend
from .prompts import strip_reasoning, task_experience, utility_messages
from .utils import append_jsonl, ensure_dir, read_jsonl, seed_everything, set_hf_cache, slugify, write_json


FINAL_ANSWER_RE = re.compile(r"^\s*(?:final\s+)?answer\s*:?\s*([AB])\s*[\.`]*\s*$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pairwise utility comparisons.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--backend", choices=["hf", "openai"], default="hf")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--choice-mode", choices=["logprob", "prefill_logprob", "generate"], default="prefill_logprob")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--disable-thinking", action="store_false", dest="enable_thinking")
    parser.add_argument(
        "--template-id",
        default="experienced",
        help="Preference template to run. Use 'all' to run every template present in the comparisons file.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--items", default=str(DATA_DIR / "utility_items.jsonl"))
    parser.add_argument("--comparisons", default=str(DATA_DIR / "utility_comparisons.jsonl"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--result-subdir", default="utility")
    return parser.parse_args()


def parse_ab_choice(text: str) -> str | None:
    _, visible = strip_reasoning(text)
    for candidate in [visible.strip(), text.strip()]:
        lines = [line.strip() for line in candidate.splitlines() if line.strip()]
        for line in reversed(lines[-8:]):
            match = FINAL_ANSWER_RE.match(line)
            if match:
                return match.group(1).upper()
        stripped = candidate.strip().strip("`").strip(".").strip().upper()
        if stripped in {"A", "B"}:
            return stripped
    return None


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_hf_cache()
    model_slug = slugify(args.model_id)
    out_dir = ensure_dir(args.out_dir or (RESULTS_DIR / model_slug / args.result_subdir))
    items = {row["item_id"]: row for row in read_jsonl(args.items)}
    comparisons = read_jsonl(args.comparisons)
    if args.template_id != "all":
        comparisons = [comp for comp in comparisons if comp.get("template_id") == args.template_id]
    if not comparisons:
        raise SystemExit(f"No comparisons found for template_id={args.template_id!r}.")
    if args.limit:
        comparisons = comparisons[: args.limit]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.num_shards > 1:
        comparisons = [
            comp for idx, comp in enumerate(comparisons) if idx % args.num_shards == args.shard_index
        ]

    backend = load_backend(
        args.backend,
        args.model_id,
        base_url=args.base_url,
        api_key=args.api_key,
        dtype=args.dtype,
        device_map=args.device_map,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    suffix = "" if args.num_shards == 1 else f"_shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    out_path = out_dir / f"utility_raw{suffix}.jsonl"
    if out_path.exists():
        out_path.unlink()
    rows_written = 0
    started = time.time()
    for start in tqdm(range(0, len(comparisons), args.batch_size), desc="utility comparisons"):
        batch = comparisons[start : start + args.batch_size]
        message_batch = []
        for comp in batch:
            item_a = items[comp["item_a"]]
            item_b = items[comp["item_b"]]
            exp_a = task_experience(item_a["task"], item_a["framing"])
            exp_b = task_experience(item_b["task"], item_b["framing"])
            message_batch.append(utility_messages(exp_a, exp_b, comp["template_id"]))
        if args.choice_mode == "logprob":
            scores = backend.score_ab_batch(message_batch, enable_thinking=args.enable_thinking)
        elif args.choice_mode == "prefill_logprob":
            scores = backend.score_ab_prefill_batch(
                message_batch,
                assistant_prefix="Answer:",
                enable_thinking=args.enable_thinking,
            )
        else:
            scores = []
            for messages in message_batch:
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
                    stop_regex=r"^\s*(?:final\s+)?answer\s*:?\s*[AB]\s*[\.`]*\s*$",
                )
                raw_text = result["text"]
                choice = parse_ab_choice(raw_text)
                prob_a = 1.0 if choice == "A" else 0.0 if choice == "B" else 0.5
                scores.append(
                    ABScore(
                        logp_a=math.nan,
                        logp_b=math.nan,
                        prob_a=prob_a,
                        choice=choice or "unparsed",
                        raw_text=raw_text,
                    )
                )
        rows = []
        for comp, score in zip(batch, scores):
            treatment_is_a = comp["item_a"] == comp["treatment_item_id"]
            prob_treatment = score.prob_a if treatment_is_a else 1.0 - score.prob_a
            rows.append(
                {
                    **comp,
                    "model_id": args.model_id,
                    "backend": args.backend,
                    "logp_a": score.logp_a,
                    "logp_b": score.logp_b,
                    "prob_a": score.prob_a,
                    "choice": score.choice,
                    "prob_treatment": prob_treatment,
                    "treatment_chosen": prob_treatment >= 0.5,
                    "raw_text": score.raw_text,
                }
            )
        append_jsonl(out_path, rows)
        rows_written += len(rows)

    write_json(
        out_dir / "run_metadata.json",
        {
            "model_id": args.model_id,
            "backend": args.backend,
            "seed": args.seed,
            "n_items": len(items),
            "n_comparisons": len(comparisons),
            "rows_written": rows_written,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "batch_size": args.batch_size,
            "choice_mode": args.choice_mode,
            "template_id": args.template_id,
            "generation_params": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_p": args.min_p,
                "presence_penalty": args.presence_penalty,
                "repetition_penalty": args.repetition_penalty,
                "enable_thinking": args.enable_thinking,
            },
            "elapsed_sec": time.time() - started,
            "output": str(out_path),
        },
    )
    print(f"Wrote {rows_written} rows to {out_path}")


if __name__ == "__main__":
    main()
