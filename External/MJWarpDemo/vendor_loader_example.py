from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import imageio.v3 as iio
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description="Read one episode from each vendor delivery format")
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()

    native_path = sorted((root / "native_hdf5").glob("*.h5"))[0]
    with h5py.File(native_path, "r") as episode:
        print("native", native_path.name, episode["observations/joint_position"].shape)
        print("native task", episode.attrs["task_name"], "success", bool(episode.attrs["success_final"]))

    parquet_path = sorted((root / "lerobot_v3" / "data").rglob("*.parquet"))[0]
    table = pq.read_table(parquet_path)
    print("lerobot", parquet_path.name, table.num_rows, table.column_names)
    video_path = sorted((root / "lerobot_v3" / "videos" / "observation.images.front").rglob("*.mp4"))[0]
    print("lerobot video first frame", next(iio.imiter(video_path, plugin="FFMPEG")).shape)

    with h5py.File(root / "robomimic" / "dataset.hdf5", "r") as dataset:
        demo_name = sorted(dataset["data"].keys())[0]
        demo = dataset["data"][demo_name]
        print("robomimic", demo_name, demo["actions"].shape, sorted(demo["obs"].keys()))

    info = json.loads((root / "lerobot_v3" / "meta" / "info.json").read_text(encoding="utf-8"))
    print("summary", {key: info[key] for key in ("total_episodes", "total_frames", "fps")})


if __name__ == "__main__":
    main()
