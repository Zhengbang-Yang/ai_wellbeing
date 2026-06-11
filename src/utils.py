from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import HF_HOME


def set_hf_cache() -> None:
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_HOME / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "hub"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_hash(value: str, digits: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:digits]


def slugify(value: str) -> str:
    keep = []
    for ch in value.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {"/", "-", "_", ".", " "}:
            keep.append("_")
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:100] or "model"


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def bootstrap_mean_ci(
    values: list[float],
    seed: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot = np.mean(arr[idx], axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return mean, float(lo), float(hi)


def count_tokens_rough(text: str) -> int:
    return max(1, len(text.split()))
