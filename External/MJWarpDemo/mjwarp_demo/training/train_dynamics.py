from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common import default_artifacts_root, new_run_directory, select_device, set_deterministic, write_json
from .data import compute_normalization, load_dynamics_arrays, load_manifest
from .models import DynamicsModel


def _predict(
    model: DynamicsModel,
    values: np.ndarray,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    output_mean: np.ndarray,
    output_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), 4096):
            batch = torch.from_numpy((values[start : start + 4096] - input_mean) / input_std).to(device)
            normalized = model(batch).cpu().numpy()
            predictions.append(normalized * output_std + output_mean)
    return np.concatenate(predictions)


def _rollout_rmse(
    model: DynamicsModel,
    inputs: np.ndarray,
    targets: np.ndarray,
    trajectories: list[tuple[str, int]],
    state_dim: int,
    normalization: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
) -> float:
    input_mean, input_std, output_mean, output_std = normalization
    squared: list[np.ndarray] = []
    offset = 0
    for _, count in trajectories:
        state = inputs[offset, :state_dim].copy()
        for index in range(count):
            row = inputs[offset + index].copy()
            row[:state_dim] = state
            predicted = _predict(
                model, row[None], input_mean, input_std, output_mean, output_std, device
            )[0]
            truth = targets[offset + index, :state_dim]
            squared.append(np.square(predicted[:state_dim] - truth))
            state = predicted[:state_dim]
        offset += count
    return float(np.sqrt(np.mean(np.concatenate(squared))))


def main() -> None:
    parser = argparse.ArgumentParser(description="训练单步环境动力学模型")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=default_artifacts_root())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    set_deterministic(args.seed)
    device = select_device(args.device)
    manifest = load_manifest(args.manifest)
    train_x, train_y, _ = load_dynamics_arrays(manifest, "train")
    try:
        validation_x, validation_y, validation_trajectories = load_dynamics_arrays(manifest, "validation")
    except ValueError:
        validation_x, validation_y, validation_trajectories = load_dynamics_arrays(manifest, "train")
    input_mean, input_std = compute_normalization(train_x)
    output_mean, output_std = compute_normalization(train_y)
    normalized_x = ((train_x - input_mean) / input_std).astype(np.float32)
    normalized_y = ((train_y - output_mean) / output_std).astype(np.float32)
    model = DynamicsModel(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    for epoch in range(args.epochs):
        model.train()
        indices = np.random.default_rng(args.seed + epoch).permutation(len(normalized_x))
        for start in range(0, len(indices), args.batch_size):
            selected = indices[start : start + args.batch_size]
            x = torch.from_numpy(normalized_x[selected]).to(device)
            y = torch.from_numpy(normalized_y[selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    prediction = _predict(
        model, validation_x, input_mean, input_std, output_mean, output_std, device
    )
    state_dim = validation_y.shape[1] - 1
    normalization = (input_mean, input_std, output_mean, output_std)
    metrics = {
        "model_type": "dynamics_prediction",
        "scenario": manifest["scenario"],
        "one_step_state_rmse": float(np.sqrt(np.mean(np.square(prediction[:, :state_dim] - validation_y[:, :state_dim])))),
        "one_step_reward_rmse": float(np.sqrt(np.mean(np.square(prediction[:, -1] - validation_y[:, -1])))),
        "rollout_state_rmse": _rollout_rmse(
            model,
            validation_x,
            validation_y,
            validation_trajectories,
            state_dim,
            normalization,
            device,
        ),
    }
    run_id, directory = new_run_directory(args.artifacts_dir, manifest["scenario"], "dynamics")
    torch.save(model.state_dict(), directory / "model.pt")
    np.savez(
        directory / "normalization.npz",
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
    )
    shutil.copy2(args.manifest, directory / "dataset_manifest.json")
    write_json(
        directory / "model_spec.json",
        {
            "artifact_id": f"{manifest['scenario']}/{run_id}",
            "model_type": "dynamics_prediction",
            "scenario": manifest["scenario"],
            "input_dim": int(train_x.shape[1]),
            "output_dim": int(train_y.shape[1]),
            "state_dim": int(state_dim),
            "hidden_layers": [256, 256, 256],
            "manifest_sha256": manifest["manifest_sha256"],
            "transition_semantics": (
                "Schema 2.0 uses state[t] + action[t] -> state[t+1] + reward[t]; "
                "legacy schemas are aligned during loading"
            ),
        },
    )
    write_json(directory / "metrics.json", metrics)
    print(f"动力学模型：{directory.resolve()}")
    print(f"单步状态 RMSE：{metrics['one_step_state_rmse']:.6f}")


if __name__ == "__main__":
    main()
