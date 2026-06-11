# Data Directory

Generated files live here. The canonical first step is:

```bash
python -m src.make_dataset --seed 20260610
python -m src.make_comparisons --seed 20260610
```

Expected outputs:

- `tasks_hard_100.jsonl`: sampled BigCodeBench-Hard tasks.
- `tasks_simple_100.jsonl`: sampled non-hard BigCodeBench tasks.
- `task_pairs_difficulty.jsonl`: fixed random hard/simple pairings.
- `utility_items.jsonl`: utility-scored task/framing items.
- `utility_comparisons.jsonl`: pairwise comparisons for utility elicitation.
