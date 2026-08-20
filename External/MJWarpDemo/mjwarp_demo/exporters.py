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


def _attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _robot_type(source: h5py.File) -> str:
    value = _attr(source.attrs.get("robot", "franka_panda"))
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(decoded, dict) and decoded.get("id"):
            return str(decoded["id"])
    if isinstance(value, dict) and value.get("id"):
        return str(value["id"])
    return "franka_panda"


def _require_schema_v2(source: h5py.File) -> None:
    if str(_attr(source.attrs.get("schema_version", ""))) != "2.0":
        raise ValueError("export requires native HDF5 Schema 2.0")


def _write_video(path: Path, frames: np.ndarray, fps: float) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("LeRobot video export requires imageio and imageio-ffmpeg") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_log_level="error",
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def export_lerobot_v3(
    native_path: Path, output_dir: Path, *, episode_index: int = 0, task_index: int = 0
) -> Path:
    errors = validate_file(native_path)
    if errors:
        raise ValueError(f"invalid source episode: {'; '.join(errors)}")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("LeRobot export requires pyarrow") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    data_dir = output_dir / "data" / "chunk-000"
    episode_meta_dir = meta_dir / "episodes" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    episode_meta_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(native_path, "r") as source:
        _require_schema_v2(source)
        action = np.asarray(source["actions/normalized"], dtype=np.float32)
        command = np.asarray(source["actions/command"], dtype=np.float32)
        joint_position = np.asarray(source["observations/joint_position"][:-1], dtype=np.float32)
        joint_velocity = np.asarray(source["observations/joint_velocity"][:-1], dtype=np.float32)
        end_effector = np.concatenate(
            (
                np.asarray(source["observations/end_effector_position"][:-1], dtype=np.float32),
                np.asarray(source["observations/end_effector_quaternion"][:-1], dtype=np.float32),
            ),
            axis=1,
        )
        state = np.concatenate(
            (
                joint_position,
                joint_velocity,
                end_effector,
                np.asarray(source["observations/gripper_width"][:-1], dtype=np.float32).reshape(-1, 1),
            ),
            axis=1,
        )
        timestamps = np.asarray(source["transition_timestamps"], dtype=np.float64)
        count = len(action)
        table = pa.table(
            {
                "timestamp": pa.array(timestamps),
                "frame_index": pa.array(np.arange(count, dtype=np.int64)),
                "episode_index": pa.array(np.full(count, episode_index, dtype=np.int64)),
                "index": pa.array(np.arange(count, dtype=np.int64)),
                "task_index": pa.array(np.full(count, task_index, dtype=np.int64)),
                "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), state.shape[1])),
                "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), action.shape[1])),
                "action.command": pa.array(command.tolist(), type=pa.list_(pa.float32(), command.shape[1])),
                "next.reward": pa.array(np.asarray(source["rewards"], dtype=np.float32)),
                "next.done": pa.array(np.asarray(source["terminated"], dtype=np.bool_)),
            }
        )
        parquet_path = data_dir / f"episode_{episode_index:06d}.parquet"
        pq.write_table(table, parquet_path, compression="zstd")

        fps = 1.0 / float(_attr(source.attrs.get("control_dt", 0.05)))
        front_video = output_dir / "videos" / "observation.images.front" / "chunk-000" / f"episode_{episode_index:06d}.mp4"
        wrist_video = output_dir / "videos" / "observation.images.wrist" / "chunk-000" / f"episode_{episode_index:06d}.mp4"
        _write_video(front_video, source["images/front_rgb"][:-1], fps)
        _write_video(wrist_video, source["images/wrist_rgb"][:-1], fps)

        task_name = str(_attr(source.attrs["task_name"]))
        info = {
            "codebase_version": "v3.0",
            "robot_type": _robot_type(source),
            "total_episodes": 1,
            "total_frames": count,
            "total_tasks": 1,
            "fps": fps,
            "splits": {"train": f"0:{1}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.state": {"dtype": "float32", "shape": [state.shape[1]]},
                "action": {"dtype": "float32", "shape": [action.shape[1]]},
                "observation.images.front": {"dtype": "video", "shape": [int(source.attrs["image_height"]), int(source.attrs["image_width"]), 3]},
                "observation.images.wrist": {"dtype": "video", "shape": [int(source.attrs["image_height"]), int(source.attrs["image_width"]), 3]},
            },
        }
        (meta_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        (meta_dir / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": task_name}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        episode_meta = pa.table(
            {
                "episode_index": [episode_index],
                "tasks": [[task_name]],
                "length": [count],
            }
        )
        pq.write_table(episode_meta, episode_meta_dir / f"episode_{episode_index:06d}.parquet")
        stats = {
            "observation.state": {
                "min": state.min(axis=0).tolist(),
                "max": state.max(axis=0).tolist(),
                "mean": state.mean(axis=0).tolist(),
                "std": state.std(axis=0).tolist(),
            },
            "action": {
                "min": action.min(axis=0).tolist(),
                "max": action.max(axis=0).tolist(),
                "mean": action.mean(axis=0).tolist(),
                "std": action.std(axis=0).tolist(),
            },
        }
        (meta_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


def export_lerobot_v3_dataset(
    native_paths: list[Path],
    output_dir: Path,
    *,
    split_by_path: dict[Path, str] | None = None,
) -> Path:
    if not native_paths:
        raise ValueError("no native HDF5 episodes to export")
    split_by_path = {path.resolve(): split for path, split in (split_by_path or {}).items()}
    split_order = {"train": 0, "validation": 1, "test": 2}
    paths = sorted(
        (path.resolve() for path in native_paths),
        key=lambda path: (split_order.get(split_by_path.get(path, "train"), 0), path.as_posix()),
    )
    task_names: list[str] = []
    for path in paths:
        errors = validate_file(path)
        if errors:
            raise ValueError(f"invalid source episode {path.name}: {'; '.join(errors)}")
        with h5py.File(path, "r") as source:
            task_names.append(str(_attr(source.attrs["task_name"])))
    unique_tasks = sorted(set(task_names))
    task_indices = {task: index for index, task in enumerate(unique_tasks)}

    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    lengths: list[int] = []
    for episode_index, (path, task_name) in enumerate(zip(paths, task_names, strict=True)):
        export_lerobot_v3(
            path,
            output_dir,
            episode_index=episode_index,
            task_index=task_indices[task_name],
        )
        with h5py.File(path, "r") as source:
            joint_position = np.asarray(source["observations/joint_position"][:-1], dtype=np.float32)
            joint_velocity = np.asarray(source["observations/joint_velocity"][:-1], dtype=np.float32)
            end_effector = np.concatenate(
                (
                    np.asarray(source["observations/end_effector_position"][:-1], dtype=np.float32),
                    np.asarray(source["observations/end_effector_quaternion"][:-1], dtype=np.float32),
                ),
                axis=1,
            )
            state = np.concatenate(
                (
                    joint_position,
                    joint_velocity,
                    end_effector,
                    np.asarray(source["observations/gripper_width"][:-1], dtype=np.float32).reshape(-1, 1),
                ),
                axis=1,
            )
            all_states.append(state)
            all_actions.append(np.asarray(source["actions/normalized"], dtype=np.float32))
            lengths.append(int(source.attrs["transition_count"]))

    meta_dir = output_dir / "meta"
    first_info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    split_ranges: dict[str, str] = {}
    cursor = 0
    for split in ("train", "validation", "test"):
        count = sum(split_by_path.get(path, "train") == split for path in paths)
        if count:
            split_ranges[split] = f"{cursor}:{cursor + count}"
            cursor += count
    first_info.update(
        total_episodes=len(paths),
        total_frames=sum(lengths),
        total_tasks=len(unique_tasks),
        splits=split_ranges,
    )
    (meta_dir / "info.json").write_text(
        json.dumps(first_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (meta_dir / "tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_index": task_indices[task], "task": task}, ensure_ascii=False) + "\n"
            for task in unique_tasks
        ),
        encoding="utf-8",
    )
    state = np.concatenate(all_states)
    action = np.concatenate(all_actions)
    stats = {
        "observation.state": {
            "min": state.min(axis=0).tolist(),
            "max": state.max(axis=0).tolist(),
            "mean": state.mean(axis=0).tolist(),
            "std": state.std(axis=0).tolist(),
        },
        "action": {
            "min": action.min(axis=0).tolist(),
            "max": action.max(axis=0).tolist(),
            "mean": action.mean(axis=0).tolist(),
            "std": action.std(axis=0).tolist(),
        },
    }
    (meta_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_dir


def export_robomimic(native_path: Path, output_path: Path, *, split: str = "train") -> Path:
    errors = validate_file(native_path)
    if errors:
        raise ValueError(f"invalid source episode: {'; '.join(errors)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(native_path, "r") as source, h5py.File(output_path, "w") as target:
        _require_schema_v2(source)
        data = target.create_group("data")
        demo = data.create_group("demo_0")
        count = int(source.attrs["transition_count"])
        demo.attrs["num_samples"] = count
        demo.attrs["task_name"] = str(_attr(source.attrs["task_name"]))
        demo.create_dataset("actions", data=source["actions/normalized"][...], compression="gzip")
        demo.create_dataset("rewards", data=source["rewards"][...])
        demo.create_dataset("dones", data=source["terminated"][...].astype(np.uint8))
        obs = demo.create_group("obs")
        next_obs = demo.create_group("next_obs")
        mappings = {
            "joint_position": "observations/joint_position",
            "joint_velocity": "observations/joint_velocity",
            "end_effector_position": "observations/end_effector_position",
            "end_effector_quaternion": "observations/end_effector_quaternion",
            "gripper_width": "observations/gripper_width",
            "front_image": "images/front_rgb",
            "wrist_image": "images/wrist_rgb",
        }
        for output_name, source_name in mappings.items():
            values = source[source_name]
            obs.create_dataset(output_name, data=values[:-1], compression="gzip")
            next_obs.create_dataset(output_name, data=values[1:], compression="gzip")
        data.attrs["total"] = count
        data.attrs["env_args"] = json.dumps(
            {
                "env_name": str(_attr(source.attrs["task_name"])),
                "type": 1,
                "env_kwargs": {"source": "MJWarp native Schema 2.0"},
            }
        )
        mask = target.create_group("mask")
        mask.create_dataset(split, data=np.asarray([b"demo_0"], dtype="S16"))
    return output_path


def export_robomimic_dataset(
    native_paths: list[Path],
    output_path: Path,
    *,
    split_by_path: dict[Path, str] | None = None,
) -> Path:
    if not native_paths:
        raise ValueError("no native HDF5 episodes to export")
    split_by_path = {path.resolve(): split for path, split in (split_by_path or {}).items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    masks: dict[str, list[bytes]] = {"train": [], "validation": [], "test": []}
    total = 0
    with h5py.File(output_path, "w") as target:
        data = target.create_group("data")
        task_names: set[str] = set()
        for episode_index, native_path in enumerate(path.resolve() for path in native_paths):
            errors = validate_file(native_path)
            if errors:
                raise ValueError(f"invalid source episode {native_path.name}: {'; '.join(errors)}")
            with h5py.File(native_path, "r") as source:
                _require_schema_v2(source)
                demo_name = f"demo_{episode_index}"
                demo = data.create_group(demo_name)
                count = int(source.attrs["transition_count"])
                total += count
                task_name = str(_attr(source.attrs["task_name"]))
                task_names.add(task_name)
                demo.attrs["num_samples"] = count
                demo.attrs["task_name"] = task_name
                demo.create_dataset("actions", data=source["actions/normalized"][...], compression="gzip")
                demo.create_dataset("rewards", data=source["rewards"][...])
                demo.create_dataset("dones", data=source["terminated"][...].astype(np.uint8))
                obs = demo.create_group("obs")
                next_obs = demo.create_group("next_obs")
                mappings = {
                    "joint_position": "observations/joint_position",
                    "joint_velocity": "observations/joint_velocity",
                    "end_effector_position": "observations/end_effector_position",
                    "end_effector_quaternion": "observations/end_effector_quaternion",
                    "gripper_width": "observations/gripper_width",
                    "front_image": "images/front_rgb",
                    "wrist_image": "images/wrist_rgb",
                }
                for output_name, source_name in mappings.items():
                    values = source[source_name]
                    obs.create_dataset(output_name, data=values[:-1], compression="gzip")
                    next_obs.create_dataset(output_name, data=values[1:], compression="gzip")
                split = split_by_path.get(native_path, "train")
                masks.setdefault(split, []).append(demo_name.encode("utf-8"))
        data.attrs["total"] = total
        data.attrs["env_args"] = json.dumps(
            {
                "env_name": sorted(task_names),
                "type": 1,
                "env_kwargs": {"source": "MJWarp native Schema 2.0"},
            }
        )
        mask = target.create_group("mask")
        for split, demo_names in masks.items():
            if demo_names:
                mask.create_dataset(split, data=np.asarray(demo_names, dtype="S32"))
                if split == "validation":
                    mask.create_dataset("valid", data=np.asarray(demo_names, dtype="S32"))
    return output_path


def _resolve_sources(
    source: Path, manifest_path: Path | None
) -> tuple[list[Path], dict[Path, str]]:
    if manifest_path is not None:
        manifest = load_manifest(manifest_path)
        root = Path(manifest["dataset_root"])
        records = manifest["records"]
        if source.is_file():
            records = [record for record in records if (root / record["path"]).resolve() == source.resolve()]
        paths = [(root / record["path"]).resolve() for record in records]
        return paths, {
            (root / record["path"]).resolve(): str(record["split"])
            for record in records
        }
    paths = sorted(source.glob("*.h5")) if source.is_dir() else [source]
    return [path.resolve() for path in paths], {}


def write_checksums(root: Path) -> Path:
    lines: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    output = root / "SHA256SUMS"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MJWarp Schema 2.0 data for vendor review")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--format", choices=("lerobot", "robomimic", "all"), default="all")
    parser.add_argument("--manifest", type=Path, help="optional split manifest for dataset-level export")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths, split_by_path = _resolve_sources(args.source, args.manifest)
    if not paths:
        raise SystemExit(f"no .h5 files found at {args.source}")
    if args.format in {"lerobot", "all"}:
        export_lerobot_v3_dataset(paths, args.output / "lerobot_v3", split_by_path=split_by_path)
    if args.format in {"robomimic", "all"}:
        export_robomimic_dataset(
            paths,
            args.output / "robomimic" / "dataset.hdf5",
            split_by_path=split_by_path,
        )
    write_checksums(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
