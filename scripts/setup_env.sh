#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/zhengbang_yang/ai_wellbeing"
CONDA="/data/zhengbang_yang/miniconda3/bin/conda"
ENV_PREFIX="/data/zhengbang_yang/miniconda3/envs/ai_wellbeing"

if [[ ! -x "$CONDA" ]]; then
  echo "Miniconda not found at $CONDA"
  exit 1
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA" create -y -p "$ENV_PREFIX" -c conda-forge --override-channels python=3.11 pip
fi

"$ENV_PREFIX/bin/python" -m pip install --force-reinstall torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
"$ENV_PREFIX/bin/python" -m pip install -r "$ROOT/requirements.txt"

echo "Environment ready: $ENV_PREFIX"
