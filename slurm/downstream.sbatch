#!/usr/bin/env bash
#SBATCH --job-name=aiw_down
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --reservation=zhengbang_yang_resv
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

cd /data/zhengbang_yang/ai_wellbeing
mkdir -p results/slurm
export HF_HOME=/data/zhengbang_yang/.cache
export HF_DATASETS_CACHE=/data/zhengbang_yang/.cache/datasets
export TRANSFORMERS_CACHE=/data/zhengbang_yang/.cache/hub
export HF_HUB_ENABLE_HF_TRANSFER=1

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
REASONING_MAX_NEW_TOKENS="${REASONING_MAX_NEW_TOKENS:-512}"
PY=/data/zhengbang_yang/miniconda3/envs/ai_wellbeing/bin/python

nvidia-smi || true
$PY -m src.run_downstream \
  --model-id "$MODEL_ID" \
  --backend hf \
  --dtype auto \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --reasoning-max-new-tokens "$REASONING_MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 1.5 \
  --repetition-penalty 1.0 \
  --enable-thinking
$PY -m src.evaluate_downstream --model-id "$MODEL_ID" --timeout 30
$PY -m src.analyze_downstream --model-id "$MODEL_ID"
