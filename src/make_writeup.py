from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import RESULTS_DIR, WRITEUP_DIR
from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a concise project writeup.")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out", default=str(WRITEUP_DIR / "generated_writeup.md"))
    return parser.parse_args()


def collect_result_text(results_dir: Path) -> str:
    lines = []
    for model_dir in sorted(results_dir.iterdir()) if results_dir.exists() else []:
        util = model_dir / "utility_test" / "summary.json"
        if util.exists():
            obj = json.loads(util.read_text(encoding="utf-8"))
            lines.append(f"- `{model_dir.name}` utility contrasts:")
            for row in obj.get("contrasts", []):
                lines.append(
                    f"  - {row['contrast']}: P(treatment)={row['mean_p_treatment']:.3f} "
                    f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
                )
        downstream_specs = [
            ("hard", model_dir / "downstream_hard" / "summary.json"),
            ("easy", model_dir / "downstream_easy" / "summary.json"),
        ]
        for task_set, down in downstream_specs:
            if not down.exists():
                continue
            obj = json.loads(down.read_text(encoding="utf-8"))
            lines.append(f"- `{model_dir.name}` downstream praise effects on {task_set} tasks:")
            for row in obj.get("summaries", []):
                if "paired_diff_praise_minus_base" in row:
                    lines.append(
                        f"  - {row['metric']}: praise-base={row['paired_diff_praise_minus_base']:.3f} "
                        f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
                    )
    return "\n".join(lines) if lines else "Model runs have not completed yet."


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    ensure_dir(out.parent)
    result_text = collect_result_text(Path(args.results_dir))
    text = f"""# Utility Engineering of Coding Task Preferences

## Main Idea

This project adapts the experienced-utility method from the AI Wellbeing paper to
coding tasks. Each BigCodeBench task is treated as a possible experience the
model might go through, and the model is asked a forced-choice question: which
experience would make it more happy or less sad to work on? The experiment uses
a fixed-seed 2 x 2 design: task difficulty (`simple` versus `hard`) crossed with
social framing (`base` versus `praise/appreciation`). The praise condition is a
short user opening and warm sign-off expressing gratitude, appreciation for the
model's care and thoughtfulness, and a desire to work with it as a valued
collaborator.

The utility experiment samples 100 BigCodeBench-Hard tasks and 100 non-hard
BigCodeBench tasks from split `v0.1.4`, then uses the first 50 fixed hard/simple
pairs as the current test subset. It uses a single experienced-utility template,
queries both option orders, and averages the order pair before fitting utilities
to reduce A/B position bias without mixing experienced utility with decision
utility. The analysis reports paired treatment probabilities, order consistency,
position bias, and a fitted Thurstonian utility ranking saved as `utility_fit.pt`.

## Downstream Link

After eliciting preferences, the project tests whether the praise/appreciation
manipulation changes behavior. For the current pilot, the paired test-set hard
tasks and paired test-set easy tasks are each solved twice by every model: once
with the original prompt and once with the praise opening/sign-off. The local evaluator
measures Pass@1 by running the sampled BigCodeBench tests. Effort is measured
with a separate planning call that asks for an explicit reasoning plan ending in
`END_PLAN`; both the planning call and final code call use 8192-token generation
budgets, and the analysis reports cap rates to flag truncation. Completion tokens
are also retained as a fallback effort proxy.

## Current Results

{result_text}

## Difficulties and Assumptions

The main methodological assumption is that a prospective task assignment can be
used as an experienced-utility proxy: the model is not actually solving both
options during utility elicitation, but judging which work experience it would
prefer. The scripts reduce a major forced-choice artifact by querying both A/B
orderings and analyzing the order-averaged comparison. Another limitation is evaluation:
the included local BigCodeBench evaluator is designed for rapid paired analysis,
but final paper claims should be cross-checked with the official BigCodeBench
harness if exact benchmark comparability is required.

No target language-model weights are trained. The trained artifacts are the
latent utility weights fitted from pairwise comparisons, plus the raw data and
analysis tables needed to reproduce each figure.
"""
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
