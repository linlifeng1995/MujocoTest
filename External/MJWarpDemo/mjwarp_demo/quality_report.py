from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .validate_dataset import validate_file
from .training.data import load_manifest


def _json_attr(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return value if isinstance(value, dict) else {}


def _percentiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"p50": 0.0, "p95": 0.0, "maximum": 0.0}
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def _visibility_metrics(images: np.ndarray, instance_ids: list[int]) -> dict[str, float]:
    images = np.asarray(images, dtype=np.uint16)
    if images.ndim != 3 or not instance_ids:
        return {
            "visible_frame_rate": 0.0,
            "bbox_within_1_to_70_percent_rate": 0.0,
            "bbox_fraction_p50": 0.0,
            "mean_pixel_fraction": 0.0,
        }
    bbox_fractions: list[float] = []
    pixel_fractions: list[float] = []
    within = 0
    for frame in images:
        mask = np.isin(frame, instance_ids)
        pixel_fraction = float(np.mean(mask))
        pixel_fractions.append(pixel_fraction)
        if not np.any(mask):
            bbox_fractions.append(0.0)
            continue
        rows, columns = np.nonzero(mask)
        bbox_fraction = float(
            (rows.max() - rows.min() + 1)
            * (columns.max() - columns.min() + 1)
            / frame.size
        )
        bbox_fractions.append(bbox_fraction)
        within += int(0.01 <= bbox_fraction <= 0.70)
    bbox_array = np.asarray(bbox_fractions, dtype=np.float64)
    return {
        "visible_frame_rate": float(np.mean(bbox_array > 0.0)),
        "bbox_within_1_to_70_percent_rate": float(within / len(images)),
        "bbox_fraction_p50": float(np.percentile(bbox_array, 50)),
        "mean_pixel_fraction": float(np.mean(pixel_fractions)),
    }


def _visibility_by_stage(
    images: np.ndarray, instance_ids: list[int], stages: np.ndarray
) -> dict[str, dict[str, float]]:
    return {
        str(stage): _visibility_metrics(images[stages == stage], instance_ids)
        for stage in np.unique(stages).tolist()
    }


def _manipulation_visibility_passed(
    episode: dict[str, Any], *, minimum_visible_rate: float = 0.95
) -> bool:
    by_stage = episode.get("wrist_target_visibility_by_stage", {})
    manipulation_stages = [
        metrics
        for stage, metrics in by_stage.items()
        if int(stage) >= 2
    ]
    return bool(manipulation_stages) and all(
        metrics.get("visible_frame_rate", 0.0) >= minimum_visible_rate
        and metrics.get("bbox_within_1_to_70_percent_rate", 0.0) >= minimum_visible_rate
        for metrics in manipulation_stages
    )


def episode_metrics(path: Path) -> dict[str, Any]:
    errors = validate_file(path)
    with h5py.File(path, "r") as episode:
        timestamps = np.asarray(episode["timestamps"], dtype=np.float64)
        contacts = np.asarray(episode["contacts/count"], dtype=np.int32)
        valid = np.asarray(episode["contacts/valid"], dtype=np.bool_)
        penetration = np.asarray(episode["contacts/distance"], dtype=np.float32)
        contact_overflow = np.asarray(episode["contacts/overflow"], dtype=np.bool_)
        penetrations = np.maximum(0.0, -penetration[valid])
        target_penetrations = np.asarray([], dtype=np.float32)
        if "contacts/type_id" in episode:
            contact_types = np.asarray(episode["contacts/type_id"], dtype=np.int8)
            target_penetrations = np.maximum(0.0, -penetration[valid & (contact_types == 2)])
        actions = np.asarray(episode["actions/normalized"], dtype=np.float32)
        saturation_by_dimension = (
            np.mean(np.abs(actions) >= 0.999, axis=0) if len(actions) else np.zeros(0)
        )
        depth = np.asarray(episode["images/front_depth_m"], dtype=np.float32)
        depth_valid = (
            np.asarray(episode["images/front_depth_valid"], dtype=np.bool_)
            if "images/front_depth_valid" in episode
            else np.isfinite(depth) & (depth > 0.0)
        )
        contact_semantics = _json_attr(episode.attrs.get("contact_semantics", ""))
        object_instance_ids = [int(value) for value in contact_semantics.get("object_instance_ids", [])]
        front_instances = np.asarray(episode["images/front_instance_id"], dtype=np.uint16)
        wrist_instances = (
            np.asarray(episode["images/wrist_instance_id"], dtype=np.uint16)
            if "images/wrist_instance_id" in episode
            else None
        )
        front_visibility = _visibility_metrics(front_instances, object_instance_ids)
        wrist_visibility = (
            _visibility_metrics(wrist_instances, object_instance_ids)
            if wrist_instances is not None
            else {}
        )
        control_dt = float(episode.attrs.get("control_dt", 0.05))
        stages = np.asarray(episode["observations/task_stage"], dtype=np.int16)
        stage_durations = {
            str(stage): float(np.count_nonzero(stages == stage) * control_dt)
            for stage in np.unique(stages).tolist()
        }
        task_metrics = {}
        for name in (
            "insertion_depth_m",
            "axial_error_m",
            "object_contact_load_n",
            "maximum_target_penetration_m",
        ):
            dataset_name = f"task_metrics/{name}"
            if dataset_name in episode:
                values = np.asarray(episode[dataset_name], dtype=np.float32)
                task_metrics[name] = {
                    "final": float(values[-1]),
                    **_percentiles(values),
                }
        return {
            "path": str(path.resolve()),
            "schema_version": str(episode.attrs.get("schema_version", "")),
            "task_name": str(episode.attrs.get("task_name", "")),
            "seed": int(episode.attrs.get("seed", 0)),
            "policy": str(episode.attrs.get("policy", "")),
            "success": bool(episode.attrs.get("success_final", False)),
            "termination_reason": str(episode.attrs.get("termination_reason_final", "")),
            "observations": int(len(timestamps)),
            "transitions": int(episode.attrs.get("transition_count", 0)),
            "duration_seconds": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
            "max_contact_count": int(contacts.max(initial=0)),
            "contact_overflow_frames": int(np.count_nonzero(contact_overflow)),
            "contact_penetration_m": _percentiles(penetrations),
            "target_contact_penetration_m": _percentiles(target_penetrations),
            "maximum_penetration_m": float(penetrations.max(initial=0.0)),
            "action_saturation_rate_by_dimension": saturation_by_dimension.astype(float).tolist(),
            "arm_action_saturation_max": float(saturation_by_dimension[:-1].max(initial=0.0)),
            "depth_invalid_fraction": float(1.0 - np.mean(depth_valid)),
            "front_target_visibility": front_visibility,
            "wrist_target_visibility": wrist_visibility,
            "front_target_visibility_by_stage": _visibility_by_stage(
                front_instances, object_instance_ids, stages
            ),
            "wrist_target_visibility_by_stage": (
                _visibility_by_stage(wrist_instances, object_instance_ids, stages)
                if wrist_instances is not None
                else {}
            ),
            "stage_duration_seconds": stage_durations,
            "task_metrics": task_metrics,
            "file_size_bytes": int(path.stat().st_size),
            "validation_errors": errors,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


def _distribution(episodes: list[dict[str, Any]], field: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for episode in episodes:
        key = str(episode[field])
        values[key] = values.get(key, 0) + 1
    return dict(sorted(values.items()))


def _leakage_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    collisions: dict[str, dict[str, list[str]]] = {}
    for field in ("seed", "object_config_id", "scene_config_id", "randomization_group"):
        split_by_value: dict[str, set[str]] = {}
        for record in manifest["records"]:
            split_by_value.setdefault(str(record.get(field, "")), set()).add(str(record["split"]))
        field_collisions = {
            value: sorted(splits)
            for value, splits in split_by_value.items()
            if value and len(splits) > 1
        }
        if field_collisions:
            collisions[field] = field_collisions
    return {
        "passed": not collisions,
        "collisions": collisions,
        "split_counts": {
            split: sum(record["split"] == split for record in manifest["records"])
            for split in ("train", "validation", "test")
        },
        "manifest_sha256": manifest.get("manifest_sha256", ""),
    }


def build_quality_report(
    dataset_dir: Path, output_path: Path, manifest_path: Path | None = None
) -> dict[str, Any]:
    episodes = [episode_metrics(path) for path in sorted(dataset_dir.glob("*.h5"))]
    if not episodes:
        raise ValueError(f"no native HDF5 episodes in {dataset_dir}")
    success_count = sum(bool(item["success"]) for item in episodes)
    report = {
        "dataset_dir": str(dataset_dir.resolve()),
        "episode_count": len(episodes),
        "success_count": success_count,
        "failure_count": len(episodes) - success_count,
        "success_rate": success_count / len(episodes),
        "valid_episode_count": sum(not item["validation_errors"] for item in episodes),
        "invalid_episode_count": sum(bool(item["validation_errors"]) for item in episodes),
        "tasks": sorted({item["task_name"] for item in episodes}),
        "policies": sorted({item["policy"] for item in episodes}),
        "success_by_task": {
            task: {
                "episodes": sum(item["task_name"] == task for item in episodes),
                "successes": sum(item["task_name"] == task and item["success"] for item in episodes),
            }
            for task in sorted({item["task_name"] for item in episodes})
        },
        "policy_distribution": _distribution(episodes, "policy"),
        "termination_reason_distribution": _distribution(episodes, "termination_reason"),
        "quality_gates": {
            "validator_passed": all(not item["validation_errors"] for item in episodes),
            "contact_overflow_zero": all(item["contact_overflow_frames"] == 0 for item in episodes),
            "arm_action_saturation_below_5_percent": all(
                item["arm_action_saturation_max"] < 0.05 for item in episodes
            ),
            "insertion_target_penetration_below_3mm": all(
                item["task_name"] != "panda_peg_insert"
                or item["target_contact_penetration_m"]["maximum"] < 0.003
                for item in episodes
            ),
            "target_visible_in_both_cameras": all(
                item["front_target_visibility"]["visible_frame_rate"] > 0.0
                and item["wrist_target_visibility"].get("visible_frame_rate", 0.0) > 0.0
                for item in episodes
            ),
            "front_target_reviewable": all(
                item["front_target_visibility"]["visible_frame_rate"] >= 0.95
                and item["front_target_visibility"]["bbox_within_1_to_70_percent_rate"] >= 0.15
                for item in episodes
            ),
            "wrist_target_reviewable_during_manipulation": all(
                _manipulation_visibility_passed(item) for item in episodes
            ),
        },
        "episodes": episodes,
    }
    if manifest_path is not None:
        report["split_leakage_audit"] = _leakage_audit(load_manifest(manifest_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a vendor-review quality report")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("quality_report.json"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    report = build_quality_report(args.dataset_dir, args.output, args.manifest)
    print(json.dumps({key: report[key] for key in ("episode_count", "success_rate", "valid_episode_count")}, indent=2))


if __name__ == "__main__":
    main()
