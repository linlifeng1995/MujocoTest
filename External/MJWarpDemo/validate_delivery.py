from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import imageio.v3 as iio
import pyarrow.parquet as pq


def _validate_checksums(root: Path, errors: list[str]) -> int:
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.exists():
        errors.append("missing SHA256SUMS")
        return 0
    checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file():
            errors.append(f"checksum target missing: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            errors.append(f"checksum mismatch: {relative}")
        checked += 1
    return checked


def validate_delivery(root: Path) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    checksum_files = _validate_checksums(root, errors)

    manifest_path = root / "dataset_manifest.json"
    quality_path = root / "quality_report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
    if not manifest:
        errors.append("missing or empty dataset_manifest.json")
    if not quality:
        errors.append("missing or empty quality_report.json")
    for name, passed in quality.get("quality_gates", {}).items():
        if not passed:
            errors.append(f"quality gate failed: {name}")

    expected_episodes = len(manifest.get("records", []))
    native_paths = sorted((root / "native_hdf5").glob("*.h5"))
    if len(native_paths) != expected_episodes:
        errors.append(f"native episode count mismatch: {len(native_paths)} != {expected_episodes}")
    native_transitions = 0
    for path in native_paths:
        with h5py.File(path, "r") as episode:
            observations = int(episode["timestamps"].shape[0])
            transitions = int(episode["actions/normalized"].shape[0])
            if observations != transitions + 1:
                errors.append(f"native alignment mismatch: {path.name}")
            if str(episode.attrs.get("schema_version", "")) != "2.0":
                errors.append(f"unexpected native schema: {path.name}")
            native_transitions += transitions

    lerobot = root / "lerobot_v3"
    info_path = lerobot / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    parquet_paths = sorted((lerobot / "data").rglob("*.parquet"))
    front_videos = sorted((lerobot / "videos" / "observation.images.front").rglob("*.mp4"))
    wrist_videos = sorted((lerobot / "videos" / "observation.images.wrist").rglob("*.mp4"))
    if info.get("total_episodes") != expected_episodes:
        errors.append("LeRobot total_episodes mismatch")
    if len(parquet_paths) != expected_episodes:
        errors.append("LeRobot parquet episode count mismatch")
    if len(front_videos) != expected_episodes or len(wrist_videos) != expected_episodes:
        errors.append("LeRobot video episode count mismatch")
    parquet_rows = 0
    for path in parquet_paths:
        table = pq.read_table(path)
        parquet_rows += table.num_rows
        required = {"observation.state", "action", "next.reward", "next.done"}
        if not required.issubset(table.column_names):
            errors.append(f"LeRobot fields missing: {path.name}")
    if parquet_rows != native_transitions:
        errors.append(f"LeRobot/native transition mismatch: {parquet_rows} != {native_transitions}")
    for path in (*front_videos, *wrist_videos):
        frame = next(iio.imiter(path, plugin="FFMPEG"))
        if frame.shape[:2] != (240, 320):
            errors.append(f"unexpected video frame shape: {path.relative_to(root)}={frame.shape}")

    robomimic_path = root / "robomimic" / "dataset.hdf5"
    robomimic_demos = 0
    if not robomimic_path.exists():
        errors.append("missing robomimic/dataset.hdf5")
    else:
        with h5py.File(robomimic_path, "r") as dataset:
            demos = dataset.get("data", {})
            robomimic_demos = len(demos)
            if robomimic_demos != expected_episodes:
                errors.append("robomimic demo count mismatch")
            for name in demos:
                demo = demos[name]
                samples = int(demo.attrs["num_samples"])
                if demo["actions"].shape[0] != samples or demo["rewards"].shape[0] != samples:
                    errors.append(f"robomimic length mismatch: {name}")

    return {
        "root": str(root),
        "passed": not errors,
        "errors": errors,
        "checksum_files": checksum_files,
        "native_episodes": len(native_paths),
        "lerobot_episodes": len(parquet_paths),
        "lerobot_video_files": len(front_videos) + len(wrist_videos),
        "robomimic_demos": robomimic_demos,
        "transitions": native_transitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a packaged Panda Mini-Pilot delivery")
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args()
    report = validate_delivery(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
