from __future__ import annotations

from typing import Any

import numpy as np


def action_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    stages: np.ndarray,
    trajectories: list[tuple[str, int]],
) -> dict[str, Any]:
    difference = predicted - target
    denominator = np.linalg.norm(predicted, axis=1) * np.linalg.norm(target, axis=1)
    cosine = np.sum(predicted * target, axis=1) / np.maximum(denominator, 1e-6)
    direction_match = np.mean(cosine > 0.0)
    stage_metrics: dict[str, Any] = {}
    for stage in sorted(np.unique(stages).tolist()):
        mask = stages == stage
        stage_metrics[str(int(stage))] = {
            "frames": int(np.count_nonzero(mask)),
            "mae": float(np.mean(np.abs(difference[mask]))),
            "mse": float(np.mean(np.square(difference[mask]))),
        }
    trajectory_errors: list[dict[str, Any]] = []
    offset = 0
    for path, count in trajectories:
        segment = difference[offset : offset + count]
        trajectory_errors.append(
            {
                "path": path,
                "frames": count,
                "mae": float(np.mean(np.abs(segment))),
            }
        )
        offset += count
    return {
        "frames": int(len(target)),
        "mae": float(np.mean(np.abs(difference))),
        "mse": float(np.mean(np.square(difference))),
        "direction_consistency": float(direction_match),
        "by_task_stage": stage_metrics,
        "by_trajectory": trajectory_errors,
    }
