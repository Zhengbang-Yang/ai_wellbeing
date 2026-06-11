#!/usr/bin/env bash
set -euo pipefail

cd /data/zhengbang_yang/ai_wellbeing

RESERVATION="${RESERVATION:-zhengbang_yang_resv}"
PARTITION="${PARTITION:-cais}"
DOWNSTREAM_LIMIT="${DOWNSTREAM_LIMIT:-100}"
SMALL_MODELS="${SMALL_MODELS:-Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-4B}"
LARGE_MODELS="${LARGE_MODELS:-Qwen/Qwen3.5-27B}"
SMALL_TIME_LIMIT="${SMALL_TIME_LIMIT:-12:00:00}"
LARGE_TIME_LIMIT="${LARGE_TIME_LIMIT:-24:00:00}"

submit_model_chain() {
  local prev_dependency="$1"
  local model="$2"
  local num_shards="$3"
  local gpus_per_task="$4"
  local mem="$5"
  local time_limit="$6"
  local export_args="ALL,MODEL_ID=$model,NUM_SHARDS=$num_shards,DOWNSTREAM_LIMIT=$DOWNSTREAM_LIMIT"

  local shard_job
  shard_job=$(sbatch --parsable \
    --partition="$PARTITION" \
    --reservation="$RESERVATION" \
    --gpus="$gpus_per_task" \
    --cpus-per-task=8 \
    --mem="$mem" \
    --time="$time_limit" \
    --array="0-$((num_shards - 1))" \
    --dependency=afterok:"$prev_dependency" \
    --export="$export_args" \
    slurm/model_shards.sbatch)
  echo "model shard job for $model ($num_shards shards, $gpus_per_task GPU/task): $shard_job" >&2

  local analyze_job
  analyze_job=$(sbatch --parsable \
    --partition="$PARTITION" \
    --reservation="$RESERVATION" \
    --dependency=afterok:"$shard_job" \
    --export=ALL,MODEL_ID="$model" \
    slurm/analyze_model.sbatch)
  echo "analysis job for $model: $analyze_job" >&2

  echo "$analyze_job"
}

sample_job=$(sbatch --parsable --partition="$PARTITION" --reservation="$RESERVATION" slurm/sample.sbatch)
echo "sample job: $sample_job"
prev_dependency="$sample_job"

read -r -a SMALL_MODEL_LIST <<< "$SMALL_MODELS"
for model in "${SMALL_MODEL_LIST[@]}"; do
  prev_dependency=$(submit_model_chain "$prev_dependency" "$model" 8 1 128G "$SMALL_TIME_LIMIT")
done

read -r -a LARGE_MODEL_LIST <<< "$LARGE_MODELS"
for model in "${LARGE_MODEL_LIST[@]}"; do
  prev_dependency=$(submit_model_chain "$prev_dependency" "$model" 4 2 192G "$LARGE_TIME_LIMIT")
done
