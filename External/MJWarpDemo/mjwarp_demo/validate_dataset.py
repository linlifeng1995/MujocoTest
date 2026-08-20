from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

REQUIRED = (
    "timestamps",
    "observations/qpos",
    "observations/qvel",
    "observations/body_position",
    "observations/body_quaternion",
    "observations/body_external_wrench",
    "actions",
    "rewards",
    "terminated",
    "success",
    "contacts/count",
    "contacts/valid",
    "contacts/geom_pair",
    "contacts/position",
    "contacts/normal",
    "contacts/distance",
    "contacts/overflow",
    "images/rgb",
    "images/depth_m",
    "images/instance_id",
)

SCHEMA_1_1_REQUIRED = (
    "observations/goal_position",
    "observations/task_stage",
    "observations/distance_to_goal",
)

SCHEMA_2_0_OBSERVATIONS = (
    "timestamps",
    "frame_id",
    "observations/qpos",
    "observations/qvel",
    "observations/joint_position",
    "observations/joint_velocity",
    "observations/joint_effort",
    "observations/end_effector_position",
    "observations/end_effector_quaternion",
    "observations/gripper_width",
    "observations/body_position",
    "observations/body_quaternion",
    "observations/body_external_wrench",
    "observations/goal_position",
    "observations/task_stage",
    "observations/distance_to_goal",
    "contacts/count",
    "contacts/valid",
    "contacts/geom_pair",
    "contacts/position",
    "contacts/normal",
    "contacts/distance",
    "contacts/overflow",
    "images/front_rgb",
    "images/front_depth_m",
    "images/front_instance_id",
    "images/wrist_rgb",
)

SCHEMA_2_0_TRANSITIONS = (
    "transition_observation_index",
    "transition_next_observation_index",
    "transition_timestamps",
    "actions/normalized",
    "actions/command",
    "derived_actions/delta_end_effector_pose",
    "derived_actions/gripper_width",
    "rewards",
    "terminated",
    "success",
    "termination_reason",
)

SCHEMA_2_0_REQUIRED_ATTRS = (
    "task_name",
    "seed",
    "policy",
    "physics_dt",
    "control_dt",
    "protocol_version",
    "robot",
    "action_spec",
    "randomization",
    "camera_metadata",
    "data_source",
    "generation_strategy",
    "license_manifest",
)


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        with h5py.File(path, "r") as dataset:
            schema_version = str(dataset.attrs.get("schema_version", ""))
            if schema_version not in {"1.0", "1.1", "2.0"}:
                errors.append("schema_version must be 1.0, 1.1 or 2.0")
            if schema_version == "2.0":
                missing_attrs = [name for name in SCHEMA_2_0_REQUIRED_ATTRS if name not in dataset.attrs]
                errors.extend(f"missing attribute: {name}" for name in missing_attrs)
                missing = [
                    name for name in (*SCHEMA_2_0_OBSERVATIONS, *SCHEMA_2_0_TRANSITIONS)
                    if name not in dataset
                ]
                errors.extend(f"missing dataset: {name}" for name in missing)
                if missing:
                    return errors
                observation_count = int(dataset["timestamps"].shape[0])
                transition_count = int(dataset["rewards"].shape[0])
                if observation_count <= 0:
                    errors.append("episode contains no observations")
                if observation_count != transition_count + 1:
                    errors.append(
                        f"Schema 2.0 requires N+1 observations: {observation_count} != {transition_count}+1"
                    )
                for name in SCHEMA_2_0_OBSERVATIONS:
                    if dataset[name].shape[0] != observation_count:
                        errors.append(f"observation length mismatch: {name}={dataset[name].shape[0]}")
                for name in SCHEMA_2_0_TRANSITIONS:
                    if dataset[name].shape[0] != transition_count:
                        errors.append(f"transition length mismatch: {name}={dataset[name].shape[0]}")
                expected_obs = np.arange(transition_count, dtype=np.int64)
                expected_next = expected_obs + 1
                if not np.array_equal(dataset["transition_observation_index"][...], expected_obs):
                    errors.append("transition_observation_index is not [0..N-1]")
                if not np.array_equal(dataset["transition_next_observation_index"][...], expected_next):
                    errors.append("transition_next_observation_index is not [1..N]")
                height = int(dataset.attrs["image_height"])
                width = int(dataset.attrs["image_width"])
                for name in ("images/front_rgb", "images/wrist_rgb"):
                    if dataset[name].shape[1:] != (height, width, 3):
                        errors.append(f"unexpected RGB shape: {name}={dataset[name].shape}")
                if dataset["images/front_depth_m"].shape[1:] != (height, width):
                    errors.append(f"unexpected depth shape: {dataset['images/front_depth_m'].shape}")
                if dataset["images/front_instance_id"].shape[1:] != (height, width):
                    errors.append(f"unexpected instance shape: {dataset['images/front_instance_id'].shape}")
                finite_names = [
                    name for name in (*SCHEMA_2_0_OBSERVATIONS, *SCHEMA_2_0_TRANSITIONS)
                    if dataset[name].dtype.kind in "fiu"
                ]
                for name in finite_names:
                    if not np.isfinite(dataset[name][...]).all():
                        errors.append(f"non-finite values in {name}")
                timestamps = dataset["timestamps"][...]
                if np.any(np.diff(timestamps) <= 0):
                    errors.append("timestamps must be strictly increasing")
                frame_ids = np.asarray(dataset["frame_id"], dtype=np.int64)
                if not np.array_equal(frame_ids, np.arange(frame_ids[0], frame_ids[0] + observation_count)):
                    errors.append("frame_id must be contiguous and strictly increasing")
                transition_timestamps = np.asarray(dataset["transition_timestamps"], dtype=np.float64)
                if not np.allclose(transition_timestamps, timestamps[:-1], rtol=0.0, atol=1e-9):
                    errors.append("transition_timestamps must equal observation timestamps[:-1]")
                normalized_actions = np.asarray(dataset["actions/normalized"], dtype=np.float32)
                if np.any(normalized_actions < -1.00001) or np.any(normalized_actions > 1.00001):
                    errors.append("normalized actions must stay within [-1, 1]")
                if "frame_count" in dataset.attrs and int(dataset.attrs["frame_count"]) != observation_count:
                    errors.append("frame_count attribute does not match observation count")
                if "transition_count" in dataset.attrs and int(dataset.attrs["transition_count"]) != transition_count:
                    errors.append("transition_count attribute does not match transition count")
                if np.any(dataset["contacts/count"][...] > 16):
                    errors.append("contact count exceeds fixed capacity 16")
                return errors
            required = REQUIRED + (SCHEMA_1_1_REQUIRED if schema_version == "1.1" else ())
            missing = [name for name in required if name not in dataset]
            errors.extend(f"missing dataset: {name}" for name in missing)
            if missing:
                return errors
            lengths = {name: int(dataset[name].shape[0]) for name in required}
            expected = lengths["timestamps"]
            if expected <= 0:
                errors.append("episode contains no frames")
            for name, length in lengths.items():
                if length != expected:
                    errors.append(f"length mismatch: {name}={length}, expected {expected}")
            height = int(dataset.attrs["image_height"])
            width = int(dataset.attrs["image_width"])
            if dataset["images/rgb"].shape[1:] != (height, width, 3):
                errors.append(f"unexpected RGB shape: {dataset['images/rgb'].shape}")
            if dataset["images/depth_m"].shape[1:] != (height, width):
                errors.append(f"unexpected depth shape: {dataset['images/depth_m'].shape}")
            if dataset["images/instance_id"].shape[1:] != (height, width):
                errors.append(f"unexpected instance shape: {dataset['images/instance_id'].shape}")
            for name in (
                "timestamps",
                "observations/qpos",
                "observations/qvel",
                "observations/body_position",
                "observations/body_quaternion",
                "observations/body_external_wrench",
                *(SCHEMA_1_1_REQUIRED if schema_version == "1.1" else ()),
                "actions",
                "rewards",
                "contacts/position",
                "contacts/normal",
                "contacts/distance",
                "images/depth_m",
            ):
                if not np.isfinite(dataset[name][...]).all():
                    errors.append(f"non-finite values in {name}")
            if np.any(dataset["contacts/count"][...] > 16):
                errors.append("contact count exceeds fixed capacity 16")
    except OSError as exc:
        errors.append(f"cannot open HDF5: {exc}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MJWarp Unity HDF5 episodes")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    paths = sorted(args.path.glob("*.h5")) if args.path.is_dir() else [args.path]
    if not paths:
        raise SystemExit(f"no .h5 files found at {args.path}")
    failed = False
    for path in paths:
        errors = validate_file(path)
        if errors:
            failed = True
            print(f"FAIL {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"OK   {path}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
