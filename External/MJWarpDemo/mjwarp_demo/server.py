from __future__ import annotations

import argparse
import asyncio
import logging
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import mujoco_warp
import warp as wp

from .protocol import PROTOCOL_VERSION, read_message, response, write_message
from .learned_policy import LearnedPolicyRuntime, discover_models, resolve_artifact
from .recorder import EpisodeRecorder, IMAGE_HEIGHT, IMAGE_WIDTH
from .scenarios import DEFAULT_SCENARIO_ID, SCENARIOS, get_scenario, scenario_summaries
from .task import CONTROL_DT, EmbodiedTask, benchmark_sizes, create_task

LOGGER = logging.getLogger("mjwarp_demo")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
UNITY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASETS = UNITY_ROOT / "Datasets"
DEFAULT_ARTIFACTS = UNITY_ROOT / "Artifacts"


class DemoServer:
    def __init__(
        self,
        model_path: Path | None,
        dataset_dir: Path,
        device: str,
        default_scenario: str = DEFAULT_SCENARIO_ID,
        artifacts_dir: Path = DEFAULT_ARTIFACTS,
        inference_device: str = "cpu",
    ) -> None:
        self.model_override = model_path
        self.dataset_dir = dataset_dir
        self.device = device
        self.default_scenario = get_scenario(default_scenario).scenario_id
        self.artifacts_dir = artifacts_dir.resolve()
        self.inference_device = inference_device
        self.learned_policy: LearnedPolicyRuntime | None = None
        self.task: EmbodiedTask | None = None
        self.recorder: EpisodeRecorder | None = None
        self.pending_state: dict[str, Any] | None = None
        self.pending_initial = False
        self.shutdown_event = asyncio.Event()

    def ensure_task(self, nworld: int = 1, scenario_id: str | None = None) -> EmbodiedTask:
        requested_scenario = get_scenario(scenario_id or self.default_scenario).scenario_id
        if self.task is None or self.task.nworld != nworld or self.task.task_name != requested_scenario:
            if self.recorder is not None:
                self.recorder.abort("physics task was replaced while recording")
                self.recorder = None
            self.pending_state = None
            self.pending_initial = False
            model_override = self.model_override if requested_scenario == self.default_scenario else None
            self.task = create_task(
                requested_scenario,
                PACKAGE_ROOT,
                nworld=nworld,
                device=self.device,
                model_override=model_override,
            )
        return self.task

    async def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        request_id = int(message.get("request_id", 0))
        version = int(message.get("protocol_version", 0))
        if version != PROTOCOL_VERSION:
            raise ValueError(f"protocol mismatch: client={version}, server={PROTOCOL_VERSION}")
        message_type = str(message.get("type", ""))

        if message_type == "hello":
            scenario_id = str(message.get("scenario", self.default_scenario))
            task = self.ensure_task(1, scenario_id)
            return response(
                "hello",
                request_id,
                backend={
                    "name": "MJWarp",
                    "mujoco_version": mujoco.__version__,
                    "mujoco_warp_version": getattr(mujoco_warp, "__version__", "3.11.0"),
                    "warp_version": wp.__version__,
                    "gpu": task.gpu_name,
                },
                model_spec=task.model_spec(),
                scenarios=scenario_summaries(),
                models=discover_models(self.artifacts_dir, scenario_id),
            )

        if message_type == "model_list":
            scenario_id = str(message.get("scenario", self.task.task_name if self.task else self.default_scenario))
            get_scenario(scenario_id)
            return response(
                "model_list",
                request_id,
                models=discover_models(self.artifacts_dir, scenario_id),
                loaded_model=(self.learned_policy.artifact_id if self.learned_policy else ""),
            )

        if message_type == "model_load":
            scenario_id = str(message.get("scenario", self.task.task_name if self.task else self.default_scenario))
            artifact_id = str(message.get("artifact_id", ""))
            if not artifact_id:
                raise ValueError("artifact_id is required")
            task = self.ensure_task(1, scenario_id)
            runtime = LearnedPolicyRuntime(
                resolve_artifact(self.artifacts_dir, artifact_id), device=self.inference_device
            )
            runtime.validate_state(task.state_dict(), scenario_id)
            self.learned_policy = runtime
            return response(
                "model_load",
                request_id,
                loaded_model=runtime.artifact_id,
                model_info={
                    "artifact_id": runtime.artifact_id,
                    "scenario": runtime.scenario,
                    "device": str(runtime.device),
                    "input_dim": int(runtime.spec["input_dim"]),
                    "action_dim": int(runtime.spec["action_dim"]),
                },
            )

        if message_type == "model_unload":
            previous = self.learned_policy.artifact_id if self.learned_policy else ""
            self.learned_policy = None
            return response("model_unload", request_id, unloaded_model=previous)

        if message_type == "reset":
            nworld = int(message.get("nworld", 1))
            if nworld < 1 or nworld > 1024:
                raise ValueError("nworld must be in [1, 1024]")
            if self.recorder is not None:
                self.recorder.abort("reset occurred before record_stop")
                self.recorder = None
            self.pending_state = None
            self.pending_initial = False
            scenario_id = str(message.get("scenario", self.task.task_name if self.task else self.default_scenario))
            task = self.ensure_task(nworld, scenario_id)
            policy = str(message.get("policy", "expert"))
            if policy == "learned":
                if nworld != 1:
                    raise ValueError("learned policy preview currently supports nworld=1 only")
                if self.learned_policy is None:
                    raise RuntimeError("尚未加载当前场景的学习策略模型")
                if self.learned_policy.scenario != scenario_id:
                    raise ValueError(
                        f"已加载模型属于 {self.learned_policy.scenario}，不能用于 {scenario_id}"
                    )
            state = task.reset(seed=int(message.get("seed", 0)), policy=policy)
            if policy == "learned" and self.learned_policy is not None:
                self.learned_policy.validate_state(state, scenario_id)
            return response("reset", request_id, state=state)

        if message_type == "record_start":
            scenario_id = str(message.get("scenario", self.task.task_name if self.task else self.default_scenario))
            task = self.ensure_task(1, scenario_id)
            if self.recorder is not None:
                raise RuntimeError("a recording is already active")
            policy = str(message.get("policy", task.policy))
            seed = int(message.get("seed", task.seed))
            episode_id = str(
                message.get(
                    "episode_id",
                    f"{datetime.now():%Y%m%d_%H%M%S}_{task.task_name}_{policy}_seed{seed}_{uuid.uuid4().hex[:6]}",
                )
            )
            camera_metadata = message.get("camera_metadata", {})
            self.recorder = EpisodeRecorder(
                self.dataset_dir,
                episode_id,
                {
                    "task_name": task.task_name,
                    "task_display_name": task.definition.display_name,
                    "business_type": task.definition.business_type,
                    "official_reference": task.definition.official_reference,
                    "seed": seed,
                    "policy": policy,
                    "physics_dt": float(task.mj_model.opt.timestep),
                    "control_dt": CONTROL_DT,
                    "mujoco_version": mujoco.__version__,
                    "mujoco_warp_version": "3.11.0",
                    "warp_version": wp.__version__,
                    "gpu": task.gpu_name,
                    "protocol_version": PROTOCOL_VERSION,
                    "unity_version": str(message.get("unity_version", "unknown")),
                    "application_version": str(message.get("application_version", "unknown")),
                    "code_version": str(message.get("code_version", "pilot-v0.1")),
                    "data_source": str(message.get("data_source", "synthetic_simulation")),
                    "generation_strategy": str(message.get("generation_strategy", policy)),
                    "license_manifest": str(
                        message.get("license_manifest", "model/third_party/LICENSES.md")
                    ),
                    "camera_metadata": camera_metadata,
                    "asset_version": task.robot.model_source,
                    **task.episode_metadata(),
                },
                image_width=int(message.get("image_width", IMAGE_WIDTH)),
                image_height=int(message.get("image_height", IMAGE_HEIGHT)),
            )
            self.pending_state = task.state_dict()
            self.pending_initial = True
            return response("record_start", request_id, episode_id=episode_id)

        if message_type == "step":
            scenario_id = str(message.get("scenario", self.task.task_name if self.task else self.default_scenario))
            task = self.ensure_task(int(message.get("nworld", 1)), scenario_id)
            if self.recorder is not None and self.pending_state is not None:
                raise RuntimeError("previous physics frame has not received a matching capture")
            requested_action = message.get("action")
            inference: dict[str, Any] | None = None
            if task.policy == "learned" and requested_action is None:
                if self.learned_policy is None:
                    raise RuntimeError("学习策略回合缺少已加载模型")
                result = self.learned_policy.act(task.state_dict())
                requested_action = result.action
                inference = {
                    "artifact_id": self.learned_policy.artifact_id,
                    "latency_ms": result.latency_ms,
                    "blocked": result.blocked,
                    "error": result.error,
                    "action": result.action,
                }
            state = task.step(requested_action)
            if self.recorder is not None:
                self.pending_state = state
                self.pending_initial = False
            return response("step", request_id, state=state, inference=inference)

        if message_type == "capture":
            if self.recorder is None:
                return response("capture", request_id, recorded=False)
            if self.pending_state is None:
                raise RuntimeError("capture received without a pending physics state")
            try:
                if self.pending_initial:
                    self.recorder.append_initial(self.pending_state, message)
                else:
                    self.recorder.append_transition(self.pending_state, message)
            except Exception as exc:
                self.recorder.abort(str(exc))
                self.recorder = None
                self.pending_state = None
                self.pending_initial = False
                raise
            self.pending_state = None
            self.pending_initial = False
            return response("capture", request_id, recorded=True, frame_count=self.recorder.length)

        if message_type == "record_stop":
            if self.recorder is None:
                raise RuntimeError("no recording is active")
            if self.pending_state is not None:
                reason = "record_stop received before the last capture"
                self.recorder.abort(reason)
                self.recorder = None
                self.pending_state = None
                self.pending_initial = False
                raise RuntimeError(reason)
            path = self.recorder.close(bool(self.task and self.task.success[0]))
            frame_count = self.recorder.length
            self.recorder = None
            return response("record_stop", request_id, path=str(path), frame_count=frame_count)

        if message_type == "benchmark":
            sizes = [int(value) for value in message.get("sizes", [1, 64, 256, 1024])]
            scenario_id = str(message.get("scenario", self.task.task_name if self.task else self.default_scenario))
            task = self.ensure_task(1, scenario_id)
            results = benchmark_sizes(
                task.model_path,
                sizes=sizes,
                steps=int(message.get("steps", 300)),
                warmup=int(message.get("warmup", 30)),
                device=self.device,
            )
            return response("benchmark", request_id, gpu=task.gpu_name, results=results)

        if message_type == "shutdown":
            if self.recorder is not None:
                self.recorder.abort("server shutdown")
                self.recorder = None
            self.shutdown_event.set()
            return response("shutdown", request_id, stopped=True)

        raise ValueError(f"unsupported message type: {message_type}")

    async def client_connected(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        LOGGER.info("client connected: %s", peer)
        try:
            while not reader.at_eof() and not self.shutdown_event.is_set():
                try:
                    message = await read_message(reader)
                    reply = await self.handle(message)
                except asyncio.IncompleteReadError:
                    break
                except Exception as exc:
                    LOGGER.error("request failed: %s\n%s", exc, traceback.format_exc())
                    request_id = int(message.get("request_id", 0)) if "message" in locals() else 0
                    reply = response("error", request_id, error=str(exc))
                await write_message(writer, reply)
        finally:
            if self.recorder is not None:
                self.recorder.abort("client disconnected")
                self.recorder = None
                self.pending_state = None
            writer.close()
            await writer.wait_closed()
            LOGGER.info("client disconnected: %s", peer)


async def run_server(args: argparse.Namespace) -> None:
    demo = DemoServer(
        args.model,
        args.dataset_dir,
        args.device,
        args.scenario,
        args.artifacts_dir,
        args.inference_device,
    )
    server = await asyncio.start_server(demo.client_connected, args.host, args.port)
    addresses = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    print(f"MJWARP_DEMO_READY {addresses}", flush=True)
    async with server:
        serve_task = asyncio.create_task(server.serve_forever())
        await demo.shutdown_event.wait()
        serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MJWarp Unity demo backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default=DEFAULT_SCENARIO_ID)
    parser.add_argument("--model", type=Path, default=None, help="optional model override for the default scenario")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--inference-device", default="cpu")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
