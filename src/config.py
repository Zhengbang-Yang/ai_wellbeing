from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
WRITEUP_DIR = ROOT / "writeup"
REFS_DIR = ROOT / "refs"

HF_HOME = Path(os.environ.get("HF_HOME", "/data/zhengbang_yang/.cache"))
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

DEFAULT_SEED = 20260610
BIGCODE_SPLIT = "v0.1.4"
FULL_DATASET = "bigcode/bigcodebench"
HARD_DATASET = "bigcode/bigcodebench-hard"

DEFAULT_MODELS = [
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-9B",
]

