from __future__ import annotations

import argparse
import time
from pathlib import Path

from .task import PlanarPushTask


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate expert and random policies")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[1] / "model" / "planar_push.xml")
    args = parser.parse_args()
    task = PlanarPushTask(args.model, nworld=1)
    started = time.perf_counter()
    for policy in ("expert", "random"):
        wins = 0
        lengths: list[int] = []
        for seed in range(args.episodes):
            state = task.reset(seed, policy)
            while not state["terminated"]:
                state = task.step()
            wins += int(state["success"])
            lengths.append(int(state["frame_id"]))
        print(f"{policy}: {wins}/{args.episodes}, mean_frames={sum(lengths) / len(lengths):.1f}")
    print(f"elapsed_seconds={time.perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
