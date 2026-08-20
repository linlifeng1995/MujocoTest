from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..scenarios import SCENARIOS
from .common import default_artifacts_root, new_run_directory, select_device, set_deterministic, write_json
from .data import FEATURE_FIELDS, compute_normalization, load_bc_arrays, load_manifest, records_for
from .models import BehaviorCloningPolicy
from .policy_metrics import action_metrics


def _predict(
    model: BehaviorCloningPolicy,
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), 4096):
            batch = torch.from_numpy((values[start : start + 4096] - mean) / std).to(device)
            predictions.append(model(batch).cpu().numpy())
    return np.concatenate(predictions)


def train(args: argparse.Namespace) -> Path:
    set_deterministic(args.seed)
    device = select_device(args.device)
    manifest = load_manifest(args.manifest)
    scenario = str(manifest["scenario"])
    if scenario not in SCENARIOS:
        raise ValueError(f"manifest must contain one supported scenario, got {scenario!r}")
    expert_train_records = records_for(manifest, split="train", policy="expert")
    if not expert_train_records:
        raise ValueError("training split contains no expert episodes")
    schema_priority = ("2.0", "1.1", "1.0")
    training_schema = next(
        (
            schema
            for schema in schema_priority
            if any(record["schema_version"] == schema for record in expert_train_records)
        ),
        None,
    )
    if training_schema is None:
        raise ValueError("training split contains no supported schema version")
    train_x, train_y, _, _ = load_bc_arrays(
        manifest, "train", policy="expert", schema_version=training_schema
    )
    try:
        validation_x, validation_y, validation_stages, validation_trajectories = load_bc_arrays(
            manifest, "validation", policy="expert", schema_version=training_schema
        )
    except ValueError:
        validation_x, validation_y, validation_stages, validation_trajectories = load_bc_arrays(
            manifest, "train", policy="expert", schema_version=training_schema
        )
    mean, std = compute_normalization(train_x)
    model = BehaviorCloningPolicy(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss_function = nn.MSELoss()
    normalized_train = ((train_x - mean) / std).astype(np.float32)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        permutation = np.random.default_rng(args.seed + epoch).permutation(len(normalized_train))
        total_loss = 0.0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            inputs = torch.from_numpy(normalized_train[indices]).to(device)
            targets = torch.from_numpy(train_y[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(indices)
        validation_prediction = _predict(model, validation_x, mean, std, device)
        validation_loss = float(np.mean(np.square(validation_prediction - validation_y)))
        history.append(
            {"epoch": epoch + 1, "train_mse": total_loss / len(train_x), "validation_mse": validation_loss}
        )
        if validation_loss + 1e-8 < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a valid model")
    model.load_state_dict(best_state)
    validation_prediction = _predict(model, validation_x, mean, std, device)
    metrics = {
        "model_type": "behavior_cloning",
        "scenario": scenario,
        "best_validation_mse": best_loss,
        "validation": action_metrics(
            validation_prediction, validation_y, validation_stages, validation_trajectories
        ),
        "history": history,
    }
    run_id, run_directory = new_run_directory(args.artifacts_dir, scenario, "bc")
    torch.save(best_state, run_directory / "model.pt")
    np.savez(run_directory / "normalization.npz", input_mean=mean, input_std=std)
    shutil.copy2(args.manifest, run_directory / "dataset_manifest.json")
    first_record = next(
        record
        for record in expert_train_records
        if record["schema_version"] == training_schema
    )
    model_spec = {
        "artifact_id": f"{scenario}/{run_id}",
        "model_type": "behavior_cloning",
        "scenario": scenario,
        "schema_version": first_record["schema_version"],
        "legacy_zero_goal_and_stage": training_schema == "1.0",
        "manifest_sha256": manifest["manifest_sha256"],
        "input_fields": FEATURE_FIELDS,
        "input_dim": int(train_x.shape[1]),
        "qpos_dim": int(first_record["qpos_dim"]),
        "qvel_dim": int(first_record["qvel_dim"]),
        "action_dim": int(train_y.shape[1]),
        "action_range": [-1.0, 1.0],
        "hidden_layers": [256, 256, 256],
        "transition_semantics": (
            "observation[t] -> action[t] -> observation[t+1]"
            if training_schema == "2.0"
            else "legacy observation[:-1] -> action[1:]"
        ),
        "created_utc": run_id.removeprefix("bc_"),
    }
    write_json(run_directory / "model_spec.json", model_spec)
    write_json(run_directory / "metrics.json", metrics)
    print(f"模型产物：{run_directory.resolve()}")
    print(f"最佳验证 MSE：{best_loss:.6f}")
    return run_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 MJWarp 状态模仿学习策略")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=default_artifacts_root())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
