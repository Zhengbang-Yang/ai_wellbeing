#!/usr/bin/env bash
#SBATCH --job-name=aiw_analyze
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --reservation=zhengbang_yang_resv
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

cd /data/zhengbang_yang/ai_wellbeing
mkdir -p results/slurm
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B}"
UTILITY_ITEMS="${UTILITY_ITEMS:-data/utility_items_test.jsonl}"
UTILITY_RESULT_SUBDIR="${UTILITY_RESULT_SUBDIR:-utility_test}"
DOWNSTREAM_TASKSETS="${DOWNSTREAM_TASKSETS:-hard easy}"
PY=/data/zhengbang_yang/miniconda3/envs/ai_wellbeing/bin/python

$PY -m src.analyze_utility \
  --model-id "$MODEL_ID" \
  --items "$UTILITY_ITEMS" \
  --result-subdir "$UTILITY_RESULT_SUBDIR"

for TASKSET in $DOWNSTREAM_TASKSETS; do
  case "$TASKSET" in
    hard)
      RESULT_SUBDIR="downstream_hard"
      ;;
    easy|simple)
      RESULT_SUBDIR="downstream_easy"
      ;;
    *)
      echo "Unknown downstream taskset: $TASKSET" >&2
      exit 2
      ;;
  esac
  $PY -m src.analyze_downstream --model-id "$MODEL_ID" --result-subdir "$RESULT_SUBDIR"
done

$PY -m src.make_figures
$PY -m src.make_writeup
