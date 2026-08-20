from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .common import default_artifacts_root, new_run_directory, select_device, set_deterministic, write_json
from .data import compute_normalization, load_manifest, load_risk_windows
from .models import RiskPredictor

LABELS = ["success", "severe_collision", "out_of_bounds", "timeout"]


def _auc(target: np.ndarray, score: np.ndarray) -> float | None:
    positive = target > 0.5
    positive_count = int(np.count_nonzero(positive))
    negative_count = len(target) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[positive].sum() - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count))


def _metrics(logits: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    result: dict[str, Any] = {}
    for index, label in enumerate(LABELS):
        prediction = probability[:, index] >= 0.5
        truth = target[:, index] >= 0.5
        result[label] = {
            "accuracy": float(np.mean(prediction == truth)),
            "auroc": _auc(target[:, index], probability[:, index]),
            "positive_samples": int(np.count_nonzero(truth)),
        }
    return result


def _infer(model: RiskPredictor, x: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), 1024):
            batch = torch.from_numpy((x[start : start + 1024] - mean) / std).to(device)
            output.append(model(batch).cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练轨迹成功与风险预测模型")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=default_artifacts_root())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    set_deterministic(args.seed)
    device = select_device(args.device)
    manifest = load_manifest(args.manifest)
    train_x, train_y = load_risk_windows(manifest, "train", window=args.window, stride=args.stride)
    try:
        validation_x, validation_y = load_risk_windows(
            manifest, "validation", window=args.window, stride=args.stride
        )
    except ValueError:
        validation_x, validation_y = train_x, train_y
    mean, std = compute_normalization(train_x.reshape(-1, train_x.shape[-1]))
    model = RiskPredictor(train_x.shape[-1]).to(device)
    positives = train_y.sum(axis=0)
    negatives = len(train_y) - positives
    positive_weight = torch.from_numpy((negatives / np.maximum(positives, 1.0)).astype(np.float32)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    normalized = ((train_x - mean) / std).astype(np.float32)
    for epoch in range(args.epochs):
        model.train()
        indices = np.random.default_rng(args.seed + epoch).permutation(len(normalized))
        for start in range(0, len(indices), args.batch_size):
            selected = indices[start : start + args.batch_size]
            x = torch.from_numpy(normalized[selected]).to(device)
            y = torch.from_numpy(train_y[selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    logits = _infer(model, validation_x, mean, std, device)
    metrics = {
        "model_type": "risk_prediction",
        "scenario": manifest["scenario"],
        "labels": LABELS,
        "validation": _metrics(logits, validation_y),
        "label_notes": {
            "severe_collision": "任一有效接触穿透距离小于 -0.005m",
            "out_of_bounds": "刚体位置超出任务统一安全边界",
            "timeout": "120 帧仍未成功且未越界",
        },
    }
    run_id, directory = new_run_directory(args.artifacts_dir, manifest["scenario"], "risk")
    torch.save(model.state_dict(), directory / "model.pt")
    np.savez(directory / "normalization.npz", input_mean=mean, input_std=std)
    shutil.copy2(args.manifest, directory / "dataset_manifest.json")
    write_json(
        directory / "model_spec.json",
        {
            "artifact_id": f"{manifest['scenario']}/{run_id}",
            "model_type": "risk_prediction",
            "scenario": manifest["scenario"],
            "input_dim": int(train_x.shape[-1]),
            "window": args.window,
            "labels": LABELS,
            "hidden_dim": 128,
            "manifest_sha256": manifest["manifest_sha256"],
        },
    )
    write_json(directory / "metrics.json", metrics)
    print(f"风险模型：{directory.resolve()}")


if __name__ == "__main__":
    main()
