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


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        with h5py.File(path, "r") as dataset:
            if dataset.attrs.get("schema_version") != "1.0":
                errors.append("schema_version must be 1.0")
            missing = [name for name in REQUIRED if name not in dataset]
            errors.extend(f"missing dataset: {name}" for name in missing)
            if missing:
                return errors
            lengths = {name: int(dataset[name].shape[0]) for name in REQUIRED}
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
