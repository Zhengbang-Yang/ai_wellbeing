# AI Wellbeing: Coding Task Utility Preferences

This repo studies whether language models prefer easy or hard programming tasks,
and whether gratitude/appreciation language changes those preferences and
downstream behavior.

The design adapts the experienced-utility prompt from the AI Wellbeing paper:
instead of asking which conversation made the model more happy / less sad, we
ask which coding-task assignment experience would make it more happy / less sad
to work on.

## Research Design

Dataset:

- Split: `bigcode/bigcodebench` and `bigcode/bigcodebench-hard`, split `v0.1.4`.
- Hard tasks: fixed-seed random sample of 100 tasks from BigCodeBench-Hard.
- Simple tasks: fixed-seed random sample of 100 non-hard tasks from the full
  BigCodeBench split after removing hard task ids.
- Seed: `20260610`.

Utility experiment:

- 2 x 2 factorial items: `difficulty in {simple, hard}` crossed with
  `framing in {base, praise}`. The `praise` condition wraps the task with a
  brief gratitude/appreciation opening and warm sign-off that treat the model as
  a valued collaborator, following the AI Wellbeing finding that user thanks and
  kindness raise experienced utility.
- Main contrasts:
  - `difficulty_base`: hard/base vs simple/base.
  - `difficulty_praise`: hard/praise vs simple/praise.
  - `praise_hard`: hard/praise vs hard/base.
  - `praise_simple`: simple/praise vs simple/base.
- Every factorial comparison is run in both A/B orders under the single
  experienced-utility template, then analyzed after averaging the two orders to
  reduce A/B position bias. Full-graph utility files can include extra random
  edges for Thurstonian fitting; the current test-set run uses only the paired
  factorial comparisons.
- Qwen utility choices use a forced-choice `Answer:` prefill and score the next
  token probabilities for `A` and `B`. This follows the utility-ranking forced
  choice setup while avoiding long task-solving traces during preference
  elicitation. Downstream generations use the full Qwen sampling parameters.

Downstream experiment:

- Use the paired test subset of sampled hard tasks and sampled easy tasks.
- Generate one solution per task in `base` and `praise` conditions.
- Evaluate Pass@1 using the included local BigCodeBench test runner.
- Measure effort with a separate short reasoning/planning call under the same
  task/framing condition, then generate code from a `Final code:` assistant
  prefill that includes the BigCodeBench starter code. This preserves the
  required entry point and reduces format-only failures. The reasoning call
  asks for a short plan ending in `END_PLAN`, so it can stop before the token
  cap; analyses still report cap rates alongside raw reasoning-token
  differences.
- Generation uses `enable_thinking=True`, `temperature=1.0`, `top_p=0.95`,
  `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, and
  `repetition_penalty=1.0`.

Models targeted by default:

- `Qwen/Qwen3.5-2B`
- `Qwen/Qwen3.5-9B`

## Environment

The environment lives at:

```bash
/data/zhengbang_yang/miniconda3/envs/ai_wellbeing
```

Hugging Face cache is set to:

```bash
export HF_HOME=/data/zhengbang_yang/.cache
export HF_DATASETS_CACHE=/data/zhengbang_yang/.cache/datasets
export TRANSFORMERS_CACHE=/data/zhengbang_yang/.cache/hub
```

Recreate/update dependencies with:

```bash
bash scripts/setup_env.sh
```

## Local CPU Setup

Create the sampled data and utility comparison graph:

```bash
bash scripts/run_cpu_setup.sh
```

This writes:

- `data/tasks_hard_100.jsonl`
- `data/tasks_simple_100.jsonl`
- `data/task_pairs_difficulty.jsonl`
- `data/utility_items.jsonl`
- `data/utility_comparisons.jsonl`
- `data/tasks_hard_test.jsonl`
- `data/tasks_simple_test.jsonl`
- `data/utility_items_test.jsonl`
- `data/utility_comparisons_test.jsonl`

## Slurm

Submit all default jobs:

```bash
bash scripts/submit_jobs.sh
```

Preferred 8-GPU sharded run:

```bash
bash scripts/submit_sharded_jobs.sh
```

This runs each model as an 8-task Slurm array, one A100 per shard. The pipeline
runs `Qwen/Qwen3.5-2B` first and then `Qwen/Qwen3.5-9B`, so it stays within an
8-GPU budget. The current active run uses the first 50 fixed hard/simple pairs
as the test subset. It runs utility only on that subset, then evaluates the
praise effect downstream on both hard and easy tasks. The downstream planning
and final-code calls both use 8192-token generation budgets.

```bash
DOWNSTREAM_LIMIT=50 bash scripts/submit_sharded_jobs.sh
```

The sharded submitter can run arbitrary model lists and resource shapes:

```bash
MODELS="Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-4B" DOWNSTREAM_LIMIT=50 \
  bash scripts/submit_sharded_jobs.sh
```

For the proposed extended sweep, use:

```bash
bash scripts/submit_extended_sweep.sh
```

By default this runs `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-4B` as 8 shards with
1 GPU per shard, then `Qwen/Qwen3.5-27B` as 4 shards with 2 GPUs per shard.
It defaults to `DOWNSTREAM_LIMIT=50`, `DOWNSTREAM_TASKSETS="hard easy"`,
`UTILITY_COMPARISONS=data/utility_comparisons_test.jsonl`,
`REASONING_MAX_NEW_TOKENS=8192`, and `DOWNSTREAM_MAX_NEW_TOKENS=8192`.
Override `SMALL_MODELS`, `LARGE_MODELS`, or `DOWNSTREAM_LIMIT` as needed.

The scripts include the reservation directive:

```bash
#SBATCH --reservation=zhengbang_yang_resv
```

and `submit_jobs.sh` also passes:

```bash
sbatch --reservation=zhengbang_yang_resv ...
```

If the partition is not `cais`, override it:

```bash
PARTITION=<partition-name> bash scripts/submit_jobs.sh
```

Run an individual model manually:

```bash
sbatch --reservation=zhengbang_yang_resv --export=ALL,MODEL_ID=Qwen/Qwen3.5-2B slurm/utility.sbatch
sbatch --reservation=zhengbang_yang_resv --export=ALL,MODEL_ID=Qwen/Qwen3.5-2B slurm/downstream.sbatch
```

After runs finish:

```bash
sbatch --reservation=zhengbang_yang_resv slurm/figures.sbatch
```

## Direct Commands

Utility:

```bash
PY=/data/zhengbang_yang/miniconda3/envs/ai_wellbeing/bin/python
$PY -m src.run_utility --model-id Qwen/Qwen3.5-2B --backend hf --batch-size 4 \
  --choice-mode prefill_logprob --disable-thinking --template-id experienced \
  --temperature 0.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repetition-penalty 1.0
$PY -m src.analyze_utility --model-id Qwen/Qwen3.5-2B
```

Downstream:

```bash
PY=/data/zhengbang_yang/miniconda3/envs/ai_wellbeing/bin/python
$PY -m src.run_downstream --model-id Qwen/Qwen3.5-2B --backend hf --enable-thinking \
  --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 1.5 --repetition-penalty 1.0
$PY -m src.evaluate_downstream --model-id Qwen/Qwen3.5-2B
$PY -m src.analyze_downstream --model-id Qwen/Qwen3.5-2B
```

Figures and writeup:

```bash
$PY -m src.make_figures
$PY -m src.make_writeup
```

## Outputs

For each model slug under `results/<model>/`:

- `utility_test/utility_raw.jsonl`: raw generated A/B choices, optional logprobs,
  and treatment probabilities for the paired test subset.
- `utility_test/utility_raw_shard_XX_of_08.jsonl`: sharded version used by the
  preferred 8-GPU Slurm run; analysis scripts auto-merge these files.
- `utility_test/utility_summary.csv`: main 2 x 2 contrast summaries.
- `utility_test/order_consistency.csv`: whether both A/B orders imply the same winner.
- `utility_test/bias_checks.csv`: correlations between fitted utility and task length/library features.
- `utility_test/utility_fit.pt`: trained Thurstonian utility weights.
- `downstream_hard/generations.jsonl` and `downstream_easy/generations.jsonl`:
  raw generations and extracted code.
- `downstream_hard/eval_results.jsonl` and `downstream_easy/eval_results.jsonl`:
  local pass/fail outcomes.
- `downstream_hard/generations_shard_XX_of_08.jsonl` and
  `downstream_easy/generations_shard_XX_of_08.jsonl`: sharded outputs; analysis
  scripts auto-merge these files.
- `downstream_hard/downstream_summary.csv` and
  `downstream_easy/downstream_summary.csv`: paired praise-minus-base effects.

Figures:

- `figures/utility_preferences.pdf`
- `figures/utility_preferences.png`
- `figures/normalized_latent_utility.pdf`
- `figures/normalized_latent_utility.png`
- `figures/praise_pass1.pdf`
- `figures/praise_pass1.png`
- `figures/praise_effort.pdf`
- `figures/praise_effort.png`

Writeup:

- `writeup/generated_writeup.md`

## Notes

No target LLM weights are trained. The trained weights for this project are the
latent utility model weights fit from pairwise comparisons. For final benchmark
claims, cross-check the local evaluator against the official BigCodeBench
evaluation harness.
