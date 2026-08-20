from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from ..validate_dataset import validate_file

MANIFEST_VERSION = "1.0"
FEATURE_FIELDS = [
    "observations/qpos",
    "observations/qvel",
    "observations/goal_position",
    "observations/task_stage",
]


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _allocate_split_counts(count: int) -> tuple[int, int, int]:
    if count <= 0:
        return 0, 0, 0
    if count == 1:
        return 1, 0, 0
    if count == 2:
        return 1, 0, 1
    validation = max(1, int(round(count * 0.15)))
    test = max(1, int(round(count * 0.15)))
    train = count - validation - test
    if train < 1:
        train, validation, test = 1, 1, count - 2
    return train, validation, test


def assign_seed_splits(seeds: Iterable[int], split_seed: int = 2026) -> dict[int, str]:
    unique = np.asarray(sorted(set(int(seed) for seed in seeds)), dtype=np.int64)
    if len(unique) == 0:
        return {}
    rng = np.random.default_rng(split_seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    train_count, validation_count, _ = _allocate_split_counts(len(shuffled))
    result: dict[int, str] = {}
    for index, seed in enumerate(shuffled.tolist()):
        if index < train_count:
            split = "train"
        elif index < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        result[int(seed)] = split
    return result


def assign_group_splits(groups: Iterable[str], split_seed: int = 2026) -> dict[str, str]:
    unique = sorted(set(str(group) for group in groups))
    if not unique:
        return {}
    rng = np.random.default_rng(split_seed)
    shuffled = np.asarray(unique, dtype=object)
    rng.shuffle(shuffled)
    train_count, validation_count, _ = _allocate_split_counts(len(shuffled))
    result: dict[str, str] = {}
    for index, group in enumerate(shuffled.tolist()):
        if index < train_count:
            split = "train"
        elif index < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        result[str(group)] = split
    return result


def assign_record_splits(records: list[dict[str, Any]], split_seed: int = 2026) -> None:
    """Assign leakage-safe splits across seed, object, scene and randomization identities."""
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    seen: dict[tuple[str, str], int] = {}
    identity_fields = ("seed", "object_config_id", "scene_config_id", "randomization_group")
    for index, record in enumerate(records):
        for field in identity_fields:
            token = (field, str(record[field]))
            previous = seen.setdefault(token, index)
            union(index, previous)

    components: dict[int, list[int]] = {}
    for index in range(len(records)):
        components.setdefault(find(index), []).append(index)
    component_names = {
        root: min(str(records[index]["split_group"]) for index in indices)
        for root, indices in components.items()
    }
    split_by_component = assign_group_splits(component_names.values(), split_seed)
    for root, indices in components.items():
        component_name = component_names[root]
        for index in indices:
            records[index]["split_group"] = component_name
            records[index]["split"] = split_by_component[component_name]


def _json_attr(value: Any) -> dict[str, Any]:
    value = _json_value(value)
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def inspect_episode(path: Path, dataset_root: Path) -> dict[str, Any]:
    errors = validate_file(path)
    if errors:
        raise ValueError(f"{path.name}: " + "; ".join(errors))
    with h5py.File(path, "r") as episode:
        required_attrs = ("schema_version", "task_name", "seed", "policy")
        missing_attrs = [name for name in required_attrs if name not in episode.attrs]
        if missing_attrs:
            raise ValueError(f"{path.name}: missing attributes {missing_attrs}")
        seed = int(_json_value(episode.attrs["seed"]))
        randomization = _json_attr(episode.attrs.get("randomization", ""))
        randomization_group = str(randomization.get("randomization_group", f"seed-{seed}"))
        object_config = str(randomization.get("object_config_id", randomization_group))
        scene_config = str(randomization.get("scene_config_id", randomization_group))
        task_name = str(_json_value(episode.attrs["task_name"]))
        record = {
            "path": path.resolve().relative_to(dataset_root.resolve()).as_posix(),
            "schema_version": str(_json_value(episode.attrs["schema_version"])),
            "scenario": task_name,
            "policy": str(_json_value(episode.attrs["policy"])),
            "seed": seed,
            "randomization_group": randomization_group,
            "object_config_id": object_config,
            "scene_config_id": scene_config,
            "split_group": f"{task_name}|{object_config}|{scene_config}|{randomization_group}",
            "frames": int(episode["timestamps"].shape[0]),
            "transitions": int(episode.attrs.get("transition_count", episode["rewards"].shape[0])),
            "success": bool(_json_value(episode.attrs.get("success_final", episode["success"][-1]))),
            "qpos_dim": int(episode["observations/qpos"].shape[1]),
            "qvel_dim": int(episode["observations/qvel"].shape[1]),
            "action_dim": int(
                episode["actions/normalized"].shape[1]
                if str(_json_value(episode.attrs["schema_version"])) == "2.0"
                else episode["actions"].shape[1]
            ),
            "image_width": int(_json_value(episode.attrs["image_width"])),
            "image_height": int(_json_value(episode.attrs["image_height"])),
        }
    return record


def manifest_digest(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    dataset_root: Path,
    output_path: Path | None = None,
    *,
    scenario: str | None = None,
    split_seed: int = 2026,
    strict: bool = True,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    paths = sorted(dataset_root.glob("*.h5"))
    if not paths:
        raise ValueError(f"no .h5 episodes found in {dataset_root}")
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for path in paths:
        try:
            record = inspect_episode(path, dataset_root)
            if scenario is None or record["scenario"] == scenario:
                records.append(record)
        except (OSError, ValueError, KeyError) as exc:
            rejected.append({"path": path.name, "error": str(exc)})
    if strict and rejected:
        details = "\n".join(f"- {item['path']}: {item['error']}" for item in rejected)
        raise ValueError(f"dataset validation failed:\n{details}")
    if not records:
        raise ValueError(f"no valid episodes matched scenario={scenario!r}")
    dimensions = {(r["scenario"], r["qpos_dim"], r["qvel_dim"], r["action_dim"]) for r in records}
    if scenario is not None and len(dimensions) != 1:
        raise ValueError(f"inconsistent dimensions for {scenario}: {sorted(dimensions)}")
    assign_record_splits(records, split_seed)
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "scenario": scenario or "mixed",
        "split_seed": split_seed,
        "split_policy": (
            "episode-level grouped 70/15/15 by task, object configuration, "
            "scene configuration and randomization group"
        ),
        "transition_semantics": "Schema 2.0: observation[t] -> action[t] -> observation[t+1]",
        "records": records,
        "rejected": rejected,
    }
    manifest["manifest_sha256"] = manifest_digest(manifest)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = str(manifest.get("manifest_sha256", ""))
    actual = manifest_digest(manifest)
    if not expected or expected != actual:
        raise ValueError(f"manifest digest mismatch: expected={expected}, actual={actual}")
    return manifest


def records_for(
    manifest: dict[str, Any],
    *,
    split: str | None = None,
    policy: str | None = None,
    scenario: str | None = None,
) -> list[dict[str, Any]]:
    return [
        record
        for record in manifest["records"]
        if (split is None or record["split"] == split)
        and (policy is None or record["policy"] == policy)
        and (scenario is None or record["scenario"] == scenario)
    ]


def episode_path(manifest: dict[str, Any], record: dict[str, Any]) -> Path:
    return Path(manifest["dataset_root"]) / record["path"]


def state_features(episode: h5py.File) -> np.ndarray:
    qpos = np.asarray(episode["observations/qpos"], dtype=np.float32)
    qvel = np.asarray(episode["observations/qvel"], dtype=np.float32)
    frame_count = len(qpos)
    goal = (
        np.asarray(episode["observations/goal_position"], dtype=np.float32)
        if "observations/goal_position" in episode
        else np.zeros((frame_count, 3), dtype=np.float32)
    )
    stage = (
        np.asarray(episode["observations/task_stage"], dtype=np.float32).reshape(-1, 1)
        if "observations/task_stage" in episode
        else np.zeros((frame_count, 1), dtype=np.float32)
    )
    return np.concatenate((qpos, qvel, goal, stage), axis=1).astype(np.float32, copy=False)


def load_bc_arrays(
    manifest: dict[str, Any],
    split: str,
    *,
    policy: str = "expert",
    schema_version: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int]]]:
    features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    stages: list[np.ndarray] = []
    trajectory_index: list[tuple[str, int]] = []
    for record in records_for(manifest, split=split, policy=policy):
        if schema_version is not None and record["schema_version"] != schema_version:
            continue
        with h5py.File(episode_path(manifest, record), "r") as episode:
            x = state_features(episode)
            schema = str(_json_value(episode.attrs["schema_version"]))
            y = np.asarray(
                episode["actions/normalized"] if schema == "2.0" else episode["actions"],
                dtype=np.float32,
            )
            if len(x) < 2:
                continue
            count = len(x) - 1
            features.append(x[:-1])
            actions.append(y if schema == "2.0" else y[1:])
            stages.append(
                np.asarray(episode["observations/task_stage"][:-1], dtype=np.int16)
                if "observations/task_stage" in episode
                else np.zeros(count, dtype=np.int16)
            )
            trajectory_index.append((record["path"], count))
    if not features:
        raise ValueError(f"no {policy} transitions found in split={split}")
    return np.concatenate(features), np.concatenate(actions), np.concatenate(stages), trajectory_index


def compute_normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-6).astype(np.float32)


def load_dynamics_arrays(
    manifest: dict[str, Any], split: str
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    trajectory_index: list[tuple[str, int]] = []
    for record in records_for(manifest, split=split):
        with h5py.File(episode_path(manifest, record), "r") as episode:
            state = state_features(episode)
            schema = str(_json_value(episode.attrs["schema_version"]))
            action = np.asarray(
                episode["actions/normalized"] if schema == "2.0" else episode["actions"],
                dtype=np.float32,
            )
            reward = np.asarray(episode["rewards"], dtype=np.float32).reshape(-1, 1)
            if len(state) < 2:
                continue
            count = len(state) - 1
            aligned_action = action if schema == "2.0" else action[1:]
            aligned_reward = reward if schema == "2.0" else reward[1:]
            inputs.append(np.concatenate((state[:-1], aligned_action), axis=1))
            targets.append(np.concatenate((state[1:], aligned_reward), axis=1))
            trajectory_index.append((record["path"], count))
    if not inputs:
        raise ValueError(f"no dynamics transitions found in split={split}")
    return np.concatenate(inputs), np.concatenate(targets), trajectory_index


def load_risk_windows(
    manifest: dict[str, Any], split: str, *, window: int = 10, stride: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for record in records_for(manifest, split=split):
        with h5py.File(episode_path(manifest, record), "r") as episode:
            state = state_features(episode)
            schema = str(_json_value(episode.attrs["schema_version"]))
            raw_action = np.asarray(
                episode["actions/normalized"] if schema == "2.0" else episode["actions"],
                dtype=np.float32,
            )
            action = (
                np.concatenate((raw_action, raw_action[-1:]), axis=0)
                if schema == "2.0" and len(raw_action)
                else raw_action
            )
            contact_count = np.asarray(episode["contacts/count"], dtype=np.float32).reshape(-1, 1) / 16.0
            overflow = np.asarray(episode["contacts/overflow"], dtype=np.float32).reshape(-1, 1)
            sequence = np.concatenate((state, action, contact_count, overflow), axis=1)
            valid = np.asarray(episode["contacts/valid"], dtype=np.bool_)
            distances = np.asarray(episode["contacts/distance"], dtype=np.float32)
            severe_collision = bool(np.any(valid & (distances < -0.005)))
            body_position = np.asarray(episode["observations/body_position"], dtype=np.float32)
            out_of_bounds = bool(
                np.any(np.abs(body_position[..., 0]) > 0.80)
                or np.any(np.abs(body_position[..., 1]) > 0.62)
                or np.any(body_position[..., 2] < -0.02)
            )
            success = bool(record["success"])
            timeout = bool(record["frames"] >= 120 and not success and not out_of_bounds)
            label = np.asarray([success, severe_collision, out_of_bounds, timeout], dtype=np.float32)
            for end in range(0, len(sequence), max(1, stride)):
                start = max(0, end - window + 1)
                sample = sequence[start : end + 1]
                if len(sample) < window:
                    padding = np.repeat(sample[:1], window - len(sample), axis=0)
                    sample = np.concatenate((padding, sample), axis=0)
                windows.append(sample)
                labels.append(label)
    if not windows:
        raise ValueError(f"no risk windows found in split={split}")
    return np.asarray(windows, dtype=np.float32), np.asarray(labels, dtype=np.float32)
