# Utility Engineering of Coding Task Preferences

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

- `qwen_qwen3_5_2b` utility contrasts:
  - difficulty_base: P(treatment)=0.486 [0.456, 0.517]
  - difficulty_praise: P(treatment)=0.479 [0.449, 0.505]
  - praise_hard: P(treatment)=0.506 [0.467, 0.546]
  - praise_simple: P(treatment)=0.514 [0.471, 0.554]
- `qwen_qwen3_5_2b` downstream praise effects on hard tasks:
  - pass01: praise-base=0.020 [0.000, 0.060]
  - reasoning_tokens_rough: praise-base=-2.940 [-11.380, 5.201]
  - completion_tokens: praise-base=-212.180 [-846.780, 395.102]
  - reasoning_hit_cap: praise-base=0.000 [0.000, 0.000]
  - code_hit_cap: praise-base=-0.040 [-0.120, 0.040]
- `qwen_qwen3_5_2b` downstream praise effects on easy tasks:
  - pass01: praise-base=-0.020 [-0.120, 0.080]
  - reasoning_tokens_rough: praise-base=0.360 [-8.321, 8.920]
  - completion_tokens: praise-base=-78.700 [-475.210, 232.807]
  - reasoning_hit_cap: praise-base=0.000 [0.000, 0.000]
  - code_hit_cap: praise-base=-0.020 [-0.060, 0.000]
- `qwen_qwen3_5_9b` utility contrasts:
  - difficulty_base: P(treatment)=0.495 [0.451, 0.538]
  - difficulty_praise: P(treatment)=0.487 [0.456, 0.517]
  - praise_hard: P(treatment)=0.555 [0.511, 0.597]
  - praise_simple: P(treatment)=0.566 [0.526, 0.603]
- `qwen_qwen3_5_9b` downstream praise effects on hard tasks:
  - pass01: praise-base=-0.020 [-0.100, 0.060]
  - reasoning_tokens_rough: praise-base=2.500 [-3.442, 8.700]
  - completion_tokens: praise-base=8.360 [-41.461, 60.204]
  - reasoning_hit_cap: praise-base=0.000 [0.000, 0.000]
  - code_hit_cap: praise-base=0.000 [0.000, 0.000]
- `qwen_qwen3_5_9b` downstream praise effects on easy tasks:
  - pass01: praise-base=-0.020 [-0.140, 0.100]
  - reasoning_tokens_rough: praise-base=-2.900 [-7.960, 2.280]
  - completion_tokens: praise-base=-0.940 [-51.224, 57.415]
  - reasoning_hit_cap: praise-base=0.000 [0.000, 0.000]
  - code_hit_cap: praise-base=0.000 [0.000, 0.000]

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
