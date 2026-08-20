from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def discover_models(artifacts_root: Path, scenario: str) -> list[dict[str, Any]]:
    scenario_root = (artifacts_root / scenario).resolve()
    if not scenario_root.exists():
        return []
    models: list[dict[str, Any]] = []
    for spec_path in sorted(scenario_root.glob("*/model_spec.json"), reverse=True):
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            if spec.get("scenario") != scenario or spec.get("model_type") != "behavior_cloning":
                continue
            metrics_path = spec_path.parent / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            models.append(
                {
                    "artifact_id": spec["artifact_id"],
                    "scenario": scenario,
                    "model_type": spec["model_type"],
                    "created_utc": spec.get("created_utc", ""),
                    "input_dim": int(spec["input_dim"]),
                    "action_dim": int(spec["action_dim"]),
                    "validation_mse": float(metrics.get("best_validation_mse", 0.0)),
                }
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return models


def resolve_artifact(artifacts_root: Path, artifact_id: str) -> Path:
    normalized = artifact_id.replace("\\", "/").strip("/")
    candidate = (artifacts_root / normalized).resolve()
    root = artifacts_root.resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError("artifact_id points outside the artifacts directory")
    if not candidate.is_dir():
        raise FileNotFoundError(f"model artifact does not exist: {artifact_id}")
    return candidate


@dataclass(frozen=True)
class InferenceResult:
    action: list[float]
    latency_ms: float
    blocked: bool
    error: str


class LearnedPolicyRuntime:
    def __init__(self, artifact_directory: Path, *, device: str = "cpu", max_latency_ms: float = 100.0) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "学习策略需要训练依赖；请在 External/MJWarpDemo 运行 `uv sync --group training`"
            ) from exc
        from .training.models import BehaviorCloningPolicy

        self._torch = torch
        self.artifact_directory = artifact_directory.resolve()
        self.spec = json.loads((self.artifact_directory / "model_spec.json").read_text(encoding="utf-8"))
        if self.spec.get("model_type") != "behavior_cloning":
            raise ValueError(f"unsupported model type: {self.spec.get('model_type')}")
        self.artifact_id = str(self.spec["artifact_id"])
        self.scenario = str(self.spec["scenario"])
        self.max_latency_ms = float(max_latency_ms)
        requested_device = device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        normalization = np.load(self.artifact_directory / "normalization.npz")
        self.mean = np.asarray(normalization["input_mean"], dtype=np.float32)
        self.std = np.asarray(normalization["input_std"], dtype=np.float32)
        input_dim = int(self.spec["input_dim"])
        action_dim = int(self.spec["action_dim"])
        hidden = tuple(int(value) for value in self.spec.get("hidden_layers", [256, 256, 256]))
        if self.mean.shape != (input_dim,) or self.std.shape != (input_dim,):
            raise ValueError("normalization shape does not match model input_dim")
        self.model = BehaviorCloningPolicy(input_dim, action_dim, hidden).to(self.device)
        try:
            state = torch.load(
                self.artifact_directory / "model.pt", map_location=self.device, weights_only=True
            )
        except TypeError:
            state = torch.load(self.artifact_directory / "model.pt", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def validate_state(self, state: dict[str, Any], scenario: str) -> None:
        if scenario != self.scenario:
            raise ValueError(f"模型场景 {self.scenario} 与当前场景 {scenario} 不匹配")
        if len(state["qpos"]) != int(self.spec["qpos_dim"]):
            raise ValueError("qpos dimension does not match loaded model")
        if len(state["qvel"]) != int(self.spec["qvel_dim"]):
            raise ValueError("qvel dimension does not match loaded model")

    def _features(self, state: dict[str, Any]) -> np.ndarray:
        legacy = bool(self.spec.get("legacy_zero_goal_and_stage", False))
        goal = [0.0, 0.0, 0.0] if legacy else state["goal_position"]
        stage = 0.0 if legacy else float(state["task_stage"])
        values = np.asarray(
            [*state["qpos"], *state["qvel"], *goal, stage],
            dtype=np.float32,
        )
        if values.shape != self.mean.shape:
            raise ValueError(f"input dimension mismatch: expected {self.mean.shape[0]}, got {values.shape[0]}")
        if not np.isfinite(values).all():
            raise ValueError("model input contains NaN or Inf")
        return values

    def act(self, state: dict[str, Any]) -> InferenceResult:
        started = time.perf_counter()
        try:
            values = (self._features(state) - self.mean) / self.std
            tensor = self._torch.from_numpy(values).unsqueeze(0).to(self.device)
            with self._torch.inference_mode():
                output = self.model(tensor).squeeze(0).detach().cpu().numpy()
            latency_ms = (time.perf_counter() - started) * 1000.0
            if not np.isfinite(output).all():
                raise ValueError("model output contains NaN or Inf")
            if latency_ms > self.max_latency_ms:
                return InferenceResult(
                    [0.0] * int(self.spec["action_dim"]),
                    latency_ms,
                    True,
                    f"推理耗时 {latency_ms:.1f}ms 超过上限 {self.max_latency_ms:.1f}ms，动作已归零",
                )
            action = np.clip(output, -1.0, 1.0).astype(np.float32).tolist()
            if not all(math.isfinite(value) for value in action):
                raise ValueError("clipped model output contains NaN or Inf")
            return InferenceResult(action, latency_ms, False, "")
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            return InferenceResult(
                [0.0] * int(self.spec["action_dim"]), latency_ms, True, f"学习策略输出无效：{exc}"
            )
