from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import load_json, select_device, torch_load_state, write_json
from .data import load_bc_arrays, load_manifest
from .models import BehaviorCloningPolicy
from .policy_metrics import action_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="离线评估状态模仿学习模型")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = select_device(args.device)
    spec = load_json(args.artifact / "model_spec.json")
    manifest = load_manifest(args.artifact / "dataset_manifest.json")
    x, y, stages, trajectories = load_bc_arrays(
        manifest, args.split, policy="expert", schema_version=str(spec["schema_version"])
    )
    normalization = np.load(args.artifact / "normalization.npz")
    mean = normalization["input_mean"].astype(np.float32)
    std = normalization["input_std"].astype(np.float32)
    model = BehaviorCloningPolicy(
        int(spec["input_dim"]), int(spec["action_dim"]), tuple(spec["hidden_layers"])
    ).to(device)
    model.load_state_dict(torch_load_state(args.artifact / "model.pt", device))
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.from_numpy((x - mean) / std).to(device)).cpu().numpy()
    report = {"artifact_id": spec["artifact_id"], "split": args.split, **action_metrics(prediction, y, stages, trajectories)}
    output = args.artifact / f"offline_{args.split}_metrics.json"
    write_json(output, report)
    print(f"离线评估：MAE={report['mae']:.6f}，MSE={report['mse']:.6f}")
    print(f"报告：{output.resolve()}")


if __name__ == "__main__":
    main()
