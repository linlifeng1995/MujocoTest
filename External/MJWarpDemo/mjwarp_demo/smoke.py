from __future__ import annotations

import argparse
import asyncio
import base64
from pathlib import Path

import numpy as np

from .protocol import PROTOCOL_VERSION
from .scenarios import DEFAULT_SCENARIO_ID, SCENARIOS
from .server import DemoServer
from .validate_dataset import validate_file


async def run(output_dir: Path, scenario: str = DEFAULT_SCENARIO_ID) -> Path:
    server = DemoServer(None, output_dir, "cuda:0", scenario)
    request_id = 0

    async def send(message_type: str, **payload):
        nonlocal request_id
        request_id += 1
        return await server.handle(
            {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "type": message_type,
                **payload,
            }
        )

    hello = await send("hello", scenario=scenario)
    assert hello["backend"]["name"] == "MJWarp"
    await send("reset", seed=7, policy="expert", nworld=1, scenario=scenario)
    await send(
        "record_start",
        episode_id=f"backend_smoke_{scenario}",
        seed=7,
        policy="expert",
        scenario=scenario,
        image_width=4,
        image_height=3,
    )
    stepped = await send("step", nworld=1, scenario=scenario)
    rgba = np.zeros((3, 4, 4), dtype=np.uint8).tobytes()
    depth = np.ones((3, 4), dtype="<f4").tobytes()
    await send(
        "capture",
        frame_id=stepped["state"]["frame_id"],
        rgb_b64=base64.b64encode(rgba).decode(),
        depth_b64=base64.b64encode(depth).decode(),
        instance_b64=base64.b64encode(rgba).decode(),
    )
    stopped = await send("record_stop")
    path = Path(stopped["path"])
    errors = validate_file(path)
    if errors:
        raise RuntimeError("; ".join(errors))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one end-to-end backend/HDF5 smoke test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default=DEFAULT_SCENARIO_ID)
    args = parser.parse_args()
    path = asyncio.run(run(args.output, args.scenario))
    print(f"OK {path}")


if __name__ == "__main__":
    main()
