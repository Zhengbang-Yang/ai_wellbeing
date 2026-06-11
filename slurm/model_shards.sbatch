#!/usr/bin/env bash
#SBATCH --job-name=aiw_shard
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --reservation=zhengbang_yang_resv
#SBATCH --output=results/slurm/%x-%A_%a.out
#SBATCH --error=results/slurm/%x-%A_%a.err

set -euo pipefail

cd /data/zhengbang_yang/ai_wellbeing
mkdir -p results/slurm
export HF_HOME=/data/zhengbang_yang/.cache
export HF_DATASETS_CACHE=/data/zhengbang_yang/.cache/datasets
export TRANSFORMERS_CACHE=/data/zhengbang_yang/.cache/hub
export HF_HUB_ENABLE_HF_TRANSFER=1

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B}"
NUM_SHARDS="${NUM_SHARDS:-8}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
UTILITY_MAX_NEW_TOKENS="${UTILITY_MAX_NEW_TOKENS:-512}"
UTILITY_ITEMS="${UTILITY_ITEMS:-data/utility_items_test.jsonl}"
UTILITY_COMPARISONS="${UTILITY_COMPARISONS:-data/utility_comparisons_test.jsonl}"
UTILITY_RESULT_SUBDIR="${UTILITY_RESULT_SUBDIR:-utility_test}"
DOWNSTREAM_MAX_NEW_TOKENS="${DOWNSTREAM_MAX_NEW_TOKENS:-8192}"
REASONING_MAX_NEW_TOKENS="${REASONING_MAX_NEW_TOKENS:-8192}"
DOWNSTREAM_TASKSETS="${DOWNSTREAM_TASKSETS:-hard easy}"
DOWNSTREAM_HARD_TASKS="${DOWNSTREAM_HARD_TASKS:-data/tasks_hard_test.jsonl}"
DOWNSTREAM_EASY_TASKS="${DOWNSTREAM_EASY_TASKS:-data/tasks_simple_test.jsonl}"
DOWNSTREAM_LIMIT_ARG=()
if [[ -n "${DOWNSTREAM_LIMIT:-}" ]]; then
  DOWNSTREAM_LIMIT_ARG=(--limit "$DOWNSTREAM_LIMIT")
fi

PY=/data/zhengbang_yang/miniconda3/envs/ai_wellbeing/bin/python

nvidia-smi || true

$PY -m src.run_utility \
  --model-id "$MODEL_ID" \
  --backend hf \
  --batch-size "$BATCH_SIZE" \
  --dtype auto \
  --choice-mode prefill_logprob \
  --max-new-tokens "$UTILITY_MAX_NEW_TOKENS" \
  --disable-thinking \
  --template-id experienced \
  --temperature 0.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --repetition-penalty 1.0 \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --items "$UTILITY_ITEMS" \
  --comparisons "$UTILITY_COMPARISONS" \
  --result-subdir "$UTILITY_RESULT_SUBDIR"

for TASKSET in $DOWNSTREAM_TASKSETS; do
  case "$TASKSET" in
    hard)
      TASKS_PATH="$DOWNSTREAM_HARD_TASKS"
      RESULT_SUBDIR="downstream_hard"
      ;;
    easy|simple)
      TASKS_PATH="$DOWNSTREAM_EASY_TASKS"
      RESULT_SUBDIR="downstream_easy"
      ;;
    *)
      echo "Unknown downstream taskset: $TASKSET" >&2
      exit 2
      ;;
  esac

  $PY -m src.run_downstream \
    --model-id "$MODEL_ID" \
    --backend hf \
    --dtype auto \
    --tasks "$TASKS_PATH" \
    --result-subdir "$RESULT_SUBDIR" \
    --max-new-tokens "$DOWNSTREAM_MAX_NEW_TOKENS" \
    --reasoning-max-new-tokens "$REASONING_MAX_NEW_TOKENS" \
    --temperature 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.0 \
    --presence-penalty 1.5 \
    --repetition-penalty 1.0 \
    --enable-thinking \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    "${DOWNSTREAM_LIMIT_ARG[@]}"

  $PY -m src.evaluate_downstream \
    --model-id "$MODEL_ID" \
    --tasks "$TASKS_PATH" \
    --result-subdir "$RESULT_SUBDIR" \
    --timeout 30 \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX"
done
