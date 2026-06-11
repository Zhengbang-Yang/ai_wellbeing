#!/usr/bin/env bash
set -euo pipefail

cd /data/zhengbang_yang/ai_wellbeing

RESERVATION="${RESERVATION:-zhengbang_yang_resv}"
PARTITION="${PARTITION:-cais}"
TIMEOUT="${TIMEOUT:-240}"
MEMORY_LIMIT_MB="${MEMORY_LIMIT_MB:-8192}"
REEVAL_MEM="${REEVAL_MEM:-64G}"
REEVAL_TIME_LIMIT="${REEVAL_TIME_LIMIT:-08:00:00}"
DOWNSTREAM_TASKSETS="${DOWNSTREAM_TASKSETS:-hard easy}"
TASKSETS_EXPORT="${DOWNSTREAM_TASKSETS// /:}"
MODELS="${MODELS:-Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B}"
SMALL_NUM_SHARDS="${SMALL_NUM_SHARDS:-8}"
LARGE_NUM_SHARDS="${LARGE_NUM_SHARDS:-4}"
MAKE_FIGURES_ON_EACH_MODEL="${MAKE_FIGURES_ON_EACH_MODEL:-0}"
FINAL_MAKE_FIGURES="${FINAL_MAKE_FIGURES:-1}"

# Optional source-generation dependencies, usually from the original generation chain.
DEPEND_QWEN_QWEN3_5_0_8B="${DEPEND_QWEN_QWEN3_5_0_8B:-}"
DEPEND_QWEN_QWEN3_5_2B="${DEPEND_QWEN_QWEN3_5_2B:-}"
DEPEND_QWEN_QWEN3_5_4B="${DEPEND_QWEN_QWEN3_5_4B:-}"
DEPEND_QWEN_QWEN3_5_9B="${DEPEND_QWEN_QWEN3_5_9B:-}"
DEPEND_QWEN_QWEN3_5_27B="${DEPEND_QWEN_QWEN3_5_27B:-}"

slugify_model() {
  local value="$1"
  value="${value//\//_}"
  value="${value//./_}"
  value="${value//-/_}"
  echo "${value,,}"
}

dependency_for_model() {
  local slug="$1"
  local var="DEPEND_${slug^^}"
  var="${var//[^A-Z0-9_]/_}"
  echo "${!var:-}"
}

submit_one_model() {
  local prev_dependency="$1"
  local model="$2"
  local make_figures="$3"
  local slug source_dependency dependency num_shards array_spec export_args reeval_job analyze_job

  slug=$(slugify_model "$model")
  source_dependency=$(dependency_for_model "$slug")
  dependency="$prev_dependency"
  if [[ -n "$source_dependency" ]]; then
    if [[ -n "$dependency" ]]; then
      dependency="${dependency}:${source_dependency}"
    else
      dependency="$source_dependency"
    fi
  fi

  if [[ "$model" == *"27B" ]]; then
    num_shards="$LARGE_NUM_SHARDS"
  else
    num_shards="$SMALL_NUM_SHARDS"
  fi
  array_spec="0-$((num_shards - 1))"
  export_args="ALL,MODEL_ID=$model,NUM_SHARDS=$num_shards,TIMEOUT=$TIMEOUT,MEMORY_LIMIT_MB=$MEMORY_LIMIT_MB,DOWNSTREAM_TASKSETS=$TASKSETS_EXPORT"

  local dependency_arg=()
  if [[ -n "$dependency" ]]; then
    dependency_arg=(--dependency=afterok:"$dependency")
  fi

  reeval_job=$(sbatch --parsable \
    --partition="$PARTITION" \
    --reservation="$RESERVATION" \
    --array="$array_spec" \
    --cpus-per-task=4 \
    --mem="$REEVAL_MEM" \
    --time="$REEVAL_TIME_LIMIT" \
    "${dependency_arg[@]}" \
    --export="$export_args" \
    slurm/reeval_downstream.sbatch)
  echo "reeval job for $model ($num_shards shards): $reeval_job" >&2

  analyze_job=$(sbatch --parsable \
    --partition="$PARTITION" \
    --reservation="$RESERVATION" \
    --dependency=afterok:"$reeval_job" \
    --export=ALL,MODEL_ID="$model",DOWNSTREAM_TASKSETS="$TASKSETS_EXPORT",MAKE_FIGURES="$make_figures" \
    slurm/analyze_reeval.sbatch)
  echo "reanalyze job for $model: $analyze_job" >&2

  echo "$analyze_job"
}

prev_dependency=""
read -r -a MODEL_LIST <<< "$MODELS"
for i in "${!MODEL_LIST[@]}"; do
  model="${MODEL_LIST[$i]}"
  make_figures="$MAKE_FIGURES_ON_EACH_MODEL"
  if [[ "$i" -eq "$((${#MODEL_LIST[@]} - 1))" ]]; then
    make_figures="$FINAL_MAKE_FIGURES"
  fi
  prev_dependency=$(submit_one_model "$prev_dependency" "$model" "$make_figures")
done
