from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .exporters import export_lerobot_v3_dataset, export_robomimic_dataset, write_checksums
from .delivery_package import package_vendor_delivery
from .quality_report import build_quality_report
from .task import create_task
from .training.data import build_manifest

PANDA_SCENARIOS = ("panda_pick_place", "panda_peg_insert")


def run_physics_smoke(
    package_root: Path,
    scenarios: list[str],
    seeds: list[int],
    *,
    policy: str = "expert",
    device: str = "cuda:0",
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        task = create_task(scenario, package_root, device=device)
        try:
            for seed in seeds:
                state = task.reset(seed, policy)
                maximum_target_penetration = 0.0
                maximum_contact_count = 0
                contact_overflow = False
                actions: list[np.ndarray] = []
                while not state["terminated"]:
                    state = task.step()
                    actions.append(np.asarray(state["action"], dtype=np.float32))
                    maximum_target_penetration = max(
                        maximum_target_penetration,
                        float(state["task_metrics"]["maximum_target_penetration_m"]),
                    )
                    maximum_contact_count = max(maximum_contact_count, int(state["contacts"]["count"]))
                    contact_overflow |= bool(state["contacts"]["overflow"])
                action_array = np.stack(actions) if actions else np.zeros((0, task.action_dim), dtype=np.float32)
                saturation = (
                    np.mean(np.abs(action_array) >= 0.999, axis=0)
                    if len(action_array)
                    else np.zeros(task.action_dim, dtype=np.float32)
                )
                results.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "policy": policy,
                        "success": bool(state["success"]),
                        "termination_reason": str(state["termination_reason"]),
                        "frame_count": int(state["frame_id"]),
                        "maximum_target_penetration_m": maximum_target_penetration,
                        "maximum_contact_count": maximum_contact_count,
                        "contact_overflow": contact_overflow,
                        "arm_action_saturation_max": float(saturation[:-1].max(initial=0.0)),
                        "final_task_metrics": state["task_metrics"],
                        "randomization": task.randomization,
                    }
                )
        finally:
            del task
            gc.collect()

    gates: dict[str, Any] = {}
    for scenario in scenarios:
        scenario_results = [item for item in results if item["scenario"] == scenario]
        successes = sum(bool(item["success"]) for item in scenario_results)
        required_successes = math.ceil(0.8 * len(scenario_results))
        gates[scenario] = {
            "episodes": len(scenario_results),
            "successes": successes,
            "required_successes": required_successes,
            "success_gate_passed": successes >= required_successes,
            "contact_overflow_zero": all(not item["contact_overflow"] for item in scenario_results),
            "arm_action_saturation_below_5_percent": all(
                item["arm_action_saturation_max"] < 0.05 for item in scenario_results
            ),
            "insertion_target_penetration_below_3mm": (
                scenario != "panda_peg_insert"
                or all(item["maximum_target_penetration_m"] < 0.003 for item in scenario_results)
            ),
        }
        gates[scenario]["passed"] = all(
            value
            for key, value in gates[scenario].items()
            if key.endswith("_passed") or key.endswith("_zero") or key.endswith("_percent") or key.endswith("_3mm")
        )
    return {
        "profile": "panda-mini-pilot-v0.2-physics-smoke",
        "device": device,
        "policy": policy,
        "seeds": seeds,
        "gates": gates,
        "passed": all(gate["passed"] for gate in gates.values()),
        "episodes": results,
    }


def finalize_dataset(
    dataset_dir: Path,
    artifact_dir: Path,
    delivery_dir: Path,
    *,
    split_seed: int = 2026,
) -> dict[str, Any]:
    manifest_path = artifact_dir / "dataset_manifest.json"
    manifest = build_manifest(dataset_dir, manifest_path, split_seed=split_seed)
    paths = [(Path(manifest["dataset_root"]) / record["path"]).resolve() for record in manifest["records"]]
    split_by_path = {
        (Path(manifest["dataset_root"]) / record["path"]).resolve(): str(record["split"])
        for record in manifest["records"]
    }
    export_lerobot_v3_dataset(paths, delivery_dir / "lerobot_v3", split_by_path=split_by_path)
    export_robomimic_dataset(
        paths,
        delivery_dir / "robomimic" / "dataset.hdf5",
        split_by_path=split_by_path,
    )
    report = build_quality_report(dataset_dir, delivery_dir / "quality_report.json", manifest_path)
    package_vendor_delivery(paths, delivery_dir, manifest_path, report, Path(__file__).parents[1])
    write_checksums(delivery_dir)
    from validate_delivery import validate_delivery

    delivery_validation = validate_delivery(delivery_dir)
    (artifact_dir / "delivery_validation.json").write_text(
        json.dumps(delivery_validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not delivery_validation["passed"]:
        raise RuntimeError(f"delivery validation failed: {delivery_validation['errors']}")
    summary = {
        "manifest": str(manifest_path.resolve()),
        "delivery": str(delivery_dir.resolve()),
        "episode_count": report["episode_count"],
        "success_rate": report["success_rate"],
        "quality_gates": report["quality_gates"],
        "split_leakage_audit": report.get("split_leakage_audit", {}),
        "delivery_validation": delivery_validation,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "mini_pilot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Panda Data Mini-Pilot v0.2 orchestration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("physics-smoke", help="run multi-seed CUDA quality gates")
    smoke.add_argument("--package-root", type=Path, default=Path("."))
    smoke.add_argument("--scenario", action="append", choices=PANDA_SCENARIOS)
    smoke.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    smoke.add_argument(
        "--policy", choices=("expert", "recovery", "perturbed", "random"), default="expert"
    )
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--output", type=Path)

    finalize = subparsers.add_parser("finalize", help="index, export and report a captured batch")
    finalize.add_argument("dataset_dir", type=Path)
    finalize.add_argument("artifact_dir", type=Path)
    finalize.add_argument("delivery_dir", type=Path)
    finalize.add_argument("--split-seed", type=int, default=2026)

    args = parser.parse_args()
    if args.command == "physics-smoke":
        report = run_physics_smoke(
            args.package_root,
            args.scenario or list(PANDA_SCENARIOS),
            args.seeds,
            policy=args.policy,
            device=args.device,
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(0 if report["passed"] else 1)

    summary = finalize_dataset(
        args.dataset_dir,
        args.artifact_dir,
        args.delivery_dir,
        split_seed=args.split_seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
