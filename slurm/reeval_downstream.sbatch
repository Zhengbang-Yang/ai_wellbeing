#!/usr/bin/env bash
#SBATCH --job-name=aiw_reeval
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --reservation=zhengbang_yang_resv
#SBATCH --output=results/slurm/%x-%A_%a.out
#SBATCH --error=results/slurm/%x-%A_%a.err

set -euo pipefail

cd /data/zhengbang_yang/ai_wellbeing
mkdir -p results/slurm

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B}"
NUM_SHARDS="${NUM_SHARDS:-8}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
DOWNSTREAM_TASKSETS="${DOWNSTREAM_TASKSETS:-hard easy}"
DOWNSTREAM_TASKSETS="${DOWNSTREAM_TASKSETS//,/ }"
DOWNSTREAM_TASKSETS="${DOWNSTREAM_TASKSETS//:/ }"
DOWNSTREAM_HARD_TASKS="${DOWNSTREAM_HARD_TASKS:-data/tasks_hard_test.jsonl}"
DOWNSTREAM_EASY_TASKS="${DOWNSTREAM_EASY_TASKS:-data/tasks_simple_test.jsonl}"
TIMEOUT="${TIMEOUT:-240}"
MEMORY_LIMIT_MB="${MEMORY_LIMIT_MB:-8192}"

PY=/data/zhengbang_yang/miniconda3/envs/bigcodebench_eval/bin/python

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

  "$PY" -m src.evaluate_downstream \
    --model-id "$MODEL_ID" \
    --tasks "$TASKS_PATH" \
    --result-subdir "$RESULT_SUBDIR" \
    --timeout "$TIMEOUT" \
    --memory-limit-mb "$MEMORY_LIMIT_MB" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX"
done
