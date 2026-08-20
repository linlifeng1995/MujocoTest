from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np


def write_episode_preview(path: Path, output_dir: Path) -> Path:
    with h5py.File(path, "r") as episode:
        front = np.asarray(episode["images/front_rgb"], dtype=np.uint8)
        wrist = np.asarray(episode["images/wrist_rgb"], dtype=np.uint8)
    indices = np.linspace(0, len(front) - 1, 3, dtype=np.int64)
    front_row = np.concatenate([front[index] for index in indices], axis=1)
    wrist_row = np.concatenate([wrist[index] for index in indices], axis=1)
    preview = np.concatenate((front_row, wrist_row), axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{path.stem}.png"
    iio.imwrite(output, preview)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate front/wrist episode review montages")
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    paths = sorted(args.source.glob("*.h5")) if args.source.is_dir() else [args.source]
    if not paths:
        raise SystemExit(f"no .h5 files found at {args.source}")
    for path in paths:
        print(write_episode_preview(path, args.output_dir).resolve())


if __name__ == "__main__":
    main()
