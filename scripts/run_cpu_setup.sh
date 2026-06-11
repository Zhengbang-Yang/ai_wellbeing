#!/usr/bin/env bash
set -euo pipefail

cd /data/zhengbang_yang/ai_wellbeing
export HF_HOME=/data/zhengbang_yang/.cache
export HF_DATASETS_CACHE=/data/zhengbang_yang/.cache/datasets
export TRANSFORMERS_CACHE=/data/zhengbang_yang/.cache/hub
export HF_HUB_ENABLE_HF_TRANSFER=1

/data/zhengbang_yang/miniconda3/envs/ai_wellbeing/bin/python -m src.make_dataset --seed 20260610
/data/zhengbang_yang/miniconda3/envs/ai_wellbeing/bin/python -m src.make_comparisons --seed 20260610

