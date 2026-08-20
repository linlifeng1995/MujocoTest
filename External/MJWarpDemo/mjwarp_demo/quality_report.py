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


def episode_metrics(path: Path) -> dict[str, Any]:
    errors = validate_file(path)
    with h5py.File(path, "r") as episode:
        timestamps = np.asarray(episode["timestamps"], dtype=np.float64)
        contacts = np.asarray(episode["contacts/count"], dtype=np.int32)
        valid = np.asarray(episode["contacts/valid"], dtype=np.bool_)
        penetration = np.asarray(episode["contacts/distance"], dtype=np.float32)
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
            "maximum_penetration_m": float(max(0.0, -penetration[valid].min(initial=0.0))),
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
