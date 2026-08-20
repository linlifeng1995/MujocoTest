from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def default_artifacts_root() -> Path:
    return Path(__file__).resolve().parents[4] / "Artifacts"


def new_run_directory(artifacts_root: Path, scenario: str, prefix: str) -> tuple[str, Path]:
    run_id = f"{prefix}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    run_directory = artifacts_root / scenario / run_id
    suffix = 1
    while run_directory.exists():
        run_directory = artifacts_root / scenario / f"{run_id}_{suffix:02d}"
        suffix += 1
    run_directory.mkdir(parents=True)
    return run_directory.name, run_directory


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("未检测到可用 CUDA，训练自动降级到 CPU。")
        return torch.device("cpu")
    return torch.device(requested)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def torch_load_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def batches(length: int, batch_size: int, *, shuffle: bool, seed: int) -> list[np.ndarray]:
    indices = np.arange(length)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, length, batch_size)]
