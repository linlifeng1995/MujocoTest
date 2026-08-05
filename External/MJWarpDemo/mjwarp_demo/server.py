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
from .recorder import EpisodeRecorder, IMAGE_HEIGHT, IMAGE_WIDTH
from .scenarios import DEFAULT_SCENARIO_ID, SCENARIOS, get_scenario, scenario_summaries
from .task import EmbodiedTask, benchmark_sizes, create_task

LOGGER = logging.getLogger("mjwarp_demo")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
UNITY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASETS = UNITY_ROOT / "Datasets"


class DemoServer:
    def __init__(
        self,
        model_path: Path | None,
        dataset_dir: Path,
        device: str,
        default_scenario: str = DEFAULT_SCENARIO_ID,
    ) -> None:
        self.model_override = model_path
        self.dataset_dir = dataset_dir
        self.device = device
        self.default_scenario = get_scenario(default_scenario).scenario_id
        self.task: EmbodiedTask | None = None
        self.recorder: EpisodeRecorder | None = None
        self.pending_state: dict[str, Any] | None = None
        self.shutdown_event = asyncio.Event()

    def ensure_task(self, nworld: int = 1, scenario_id: str | None = None) -> EmbodiedTask:
        requested_scenario = get_scenario(scenario_id or self.default_scenario).scenario_id
        if self.task is None or self.task.nworld != nworld or self.task.task_name != requested_scenario:
            if self.recorder is not None:
                self.recorder.abort("physics task was replaced while recording")
                self.recorder = None
            self.pending_state = None
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
            )

        if message_type == "reset":
            nworld = int(message.get("nworld", 1))
            if nworld < 1 or nworld > 1024:
                raise ValueError("nworld must be in [1, 1024]")
            if self.recorder is not None:
                self.recorder.abort("reset occurred before record_stop")
                self.recorder = None
            self.pending_state = None
            scenario_id = str(message.get("scenario", self.task.task_name if self.task else self.default_scenario))
            task = self.ensure_task(nworld, scenario_id)
            state = task.reset(seed=int(message.get("seed", 0)), policy=str(message.get("policy", "expert")))
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
                    "control_dt": float(task.mj_model.opt.timestep * 10),
                    "mujoco_version": mujoco.__version__,
                    "mujoco_warp_version": "3.11.0",
                    "warp_version": wp.__version__,
                    "gpu": task.gpu_name,
                },
                image_width=int(message.get("image_width", IMAGE_WIDTH)),
                image_height=int(message.get("image_height", IMAGE_HEIGHT)),
            )
            self.pending_state = None
            return response("record_start", request_id, episode_id=episode_id)

        if message_type == "step":
            scenario_id = str(message.get("scenario", self.task.task_name if self.task else self.default_scenario))
            task = self.ensure_task(int(message.get("nworld", 1)), scenario_id)
            if self.recorder is not None and self.pending_state is not None:
                raise RuntimeError("previous physics frame has not received a matching capture")
            state = task.step(message.get("action"))
            if self.recorder is not None:
                self.pending_state = state
            return response("step", request_id, state=state)

        if message_type == "capture":
            if self.recorder is None:
                return response("capture", request_id, recorded=False)
            if self.pending_state is None:
                raise RuntimeError("capture received without a pending physics state")
            try:
                self.recorder.append_capture(self.pending_state, message)
            except Exception as exc:
                self.recorder.abort(str(exc))
                self.recorder = None
                self.pending_state = None
                raise
            self.pending_state = None
            return response("capture", request_id, recorded=True, frame_count=self.recorder.length)

        if message_type == "record_stop":
            if self.recorder is None:
                raise RuntimeError("no recording is active")
            if self.pending_state is not None:
                reason = "record_stop received before the last capture"
                self.recorder.abort(reason)
                self.recorder = None
                self.pending_state = None
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
    demo = DemoServer(args.model, args.dataset_dir, args.device, args.scenario)
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
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
