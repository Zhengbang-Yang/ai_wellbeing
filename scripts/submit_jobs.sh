#!/usr/bin/env bash
set -euo pipefail

cd /data/zhengbang_yang/ai_wellbeing

RESERVATION="${RESERVATION:-zhengbang_yang_resv}"
PARTITION="${PARTITION:-cais}"
MODELS=(
  "Qwen/Qwen3.5-2B"
  "Qwen/Qwen3.5-9B"
)

sample_job=$(sbatch --parsable --partition="$PARTITION" --reservation="$RESERVATION" slurm/sample.sbatch)
echo "sample job: $sample_job"

for model in "${MODELS[@]}"; do
  util_job=$(sbatch --parsable --dependency=afterok:"$sample_job" --partition="$PARTITION" --reservation="$RESERVATION" --export=ALL,MODEL_ID="$model" slurm/utility.sbatch)
  echo "utility job for $model: $util_job"
  down_job=$(sbatch --parsable --dependency=afterok:"$sample_job" --partition="$PARTITION" --reservation="$RESERVATION" --export=ALL,MODEL_ID="$model" slurm/downstream.sbatch)
  echo "downstream job for $model: $down_job"
done
