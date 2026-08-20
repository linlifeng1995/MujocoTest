from __future__ import annotations

import argparse
from pathlib import Path

from ..scenarios import SCENARIOS
from .data import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 HDF5 回合并生成可复现的数据清单")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="planar_push")
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--allow-invalid", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest(
        args.dataset_dir,
        args.output,
        scenario=args.scenario,
        split_seed=args.split_seed,
        strict=not args.allow_invalid,
    )
    counts = {
        split: sum(record["split"] == split for record in manifest["records"])
        for split in ("train", "validation", "test")
    }
    print(f"清单：{args.output.resolve()}")
    print(f"场景：{args.scenario}，回合：{len(manifest['records'])}，划分：{counts}")
    print(f"SHA256：{manifest['manifest_sha256']}")


if __name__ == "__main__":
    main()
