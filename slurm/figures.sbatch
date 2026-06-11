#!/usr/bin/env bash
#SBATCH --job-name=aiw_figs
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --reservation=zhengbang_yang_resv
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

cd /data/zhengbang_yang/ai_wellbeing
mkdir -p results/slurm
PY=/data/zhengbang_yang/miniconda3/envs/ai_wellbeing/bin/python
$PY -m src.make_figures
$PY -m src.make_writeup

