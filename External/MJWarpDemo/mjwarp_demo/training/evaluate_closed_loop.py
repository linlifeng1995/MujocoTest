from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from ..learned_policy import LearnedPolicyRuntime
from ..scenarios import SCENARIOS
from ..task import create_task
from .common import default_artifacts_root, write_json


def run_policy(task: Any, policy: str, seeds: list[int], runtime: LearnedPolicyRuntime | None) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        state = task.reset(seed, "learned" if policy == "learned" else policy)
        total_reward = 0.0
        blocked_actions = 0
        while not state["terminated"]:
            if policy == "learned":
                if runtime is None:
                    raise RuntimeError("learned policy runtime is missing")
                result = runtime.act(state)
                blocked_actions += int(result.blocked)
                state = task.step(result.action)
            else:
                state = task.step()
            total_reward += float(state["reward"])
        episodes.append(
            {
                "seed": seed,
                "success": bool(state["success"]),
                "frames": int(state["frame_id"]),
                "total_reward": total_reward,
                "blocked_actions": blocked_actions,
            }
        )
    return {
        "policy": policy,
        "episodes": len(episodes),
        "success_rate": mean(float(item["success"]) for item in episodes),
        "mean_frames": mean(item["frames"] for item in episodes),
        "mean_total_reward": mean(item["total_reward"] for item in episodes),
        "blocked_actions": sum(item["blocked_actions"] for item in episodes),
        "details": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="在未见随机种子上进行专家/随机/学习策略闭环评估")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime = LearnedPolicyRuntime(args.artifact, device="cpu")
    scenario = runtime.scenario
    if scenario not in SCENARIOS:
        raise ValueError(f"unsupported scenario in artifact: {scenario}")
    package_root = Path(__file__).resolve().parents[2]
    task = create_task(scenario, package_root, nworld=1, device=args.device)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    reports = {
        policy: run_policy(task, policy, seeds, runtime if policy == "learned" else None)
        for policy in ("expert", "random", "learned")
    }
    expert_rate = reports["expert"]["success_rate"]
    random_rate = reports["random"]["success_rate"]
    learned_rate = reports["learned"]["success_rate"]
    acceptance = {
        "learned_at_least_70_percent": learned_rate >= 0.70,
        "learned_beats_random_by_30_points": learned_rate - random_rate >= 0.30,
        "learned_within_15_points_of_expert": expert_rate - learned_rate <= 0.15,
    }
    report = {
        "artifact_id": runtime.artifact_id,
        "scenario": scenario,
        "seed_start": args.seed_start,
        "reports": reports,
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
    }
    output = args.output or (args.artifact / "closed_loop_metrics.json")
    write_json(output, report)
    print(
        f"专家={expert_rate:.1%}，随机={random_rate:.1%}，学习={learned_rate:.1%}，"
        f"验收={'通过' if report['passed'] else '未通过'}"
    )
    print(f"报告：{output.resolve()}")


if __name__ == "__main__":
    main()
