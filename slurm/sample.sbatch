#!/usr/bin/env bash
#SBATCH --job-name=aiw_sample
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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

PY=/data/zhengbang_yang/miniconda3/envs/ai_wellbeing/bin/python
$PY -m src.make_dataset --seed 20260610
$PY -m src.make_comparisons --seed 20260610

