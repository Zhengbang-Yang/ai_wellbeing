#!/usr/bin/env bash
set -euo pipefail

cd /data/zhengbang_yang/ai_wellbeing

RESERVATION="${RESERVATION:-zhengbang_yang_resv}"
PARTITION="${PARTITION:-cais}"
NUM_SHARDS="${NUM_SHARDS:-8}"
GPUS_PER_TASK="${GPUS_PER_TASK:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-128G}"
DOWNSTREAM_LIMIT="${DOWNSTREAM_LIMIT:-}"
MODELS="${MODELS:-Qwen/Qwen3.5-2B Qwen/Qwen3.5-9B}"

read -r -a MODEL_LIST <<< "$MODELS"

sample_job=$(sbatch --parsable --partition="$PARTITION" --reservation="$RESERVATION" slurm/sample.sbatch)
echo "sample job: $sample_job"

prev_dependency="$sample_job"
for model in "${MODEL_LIST[@]}"; do
  export_args="ALL,MODEL_ID=$model,NUM_SHARDS=$NUM_SHARDS"
  if [[ -n "$DOWNSTREAM_LIMIT" ]]; then
    export_args="$export_args,DOWNSTREAM_LIMIT=$DOWNSTREAM_LIMIT"
  fi
  shard_job=$(sbatch --parsable \
    --partition="$PARTITION" \
    --reservation="$RESERVATION" \
    --gpus="$GPUS_PER_TASK" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --array="0-$((NUM_SHARDS - 1))" \
    --dependency=afterok:"$prev_dependency" \
    --export="$export_args" \
    slurm/model_shards.sbatch)
  echo "model shard job for $model: $shard_job"

  analyze_job=$(sbatch --parsable \
    --partition="$PARTITION" \
    --reservation="$RESERVATION" \
    --dependency=afterok:"$shard_job" \
    --export=ALL,MODEL_ID="$model" \
    slurm/analyze_model.sbatch)
  echo "analysis job for $model: $analyze_job"
  prev_dependency="$analyze_job"
done
