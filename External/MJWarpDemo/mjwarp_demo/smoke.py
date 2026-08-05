from __future__ import annotations

import argparse
import asyncio
import base64
from pathlib import Path

import numpy as np

from .protocol import PROTOCOL_VERSION
from .server import DEFAULT_MODEL, DemoServer
from .validate_dataset import validate_file


async def run(output_dir: Path) -> Path:
    server = DemoServer(DEFAULT_MODEL, output_dir, "cuda:0")
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

    hello = await send("hello")
    assert hello["backend"]["name"] == "MJWarp"
    await send("reset", seed=7, policy="expert", nworld=1)
    await send("record_start", episode_id="backend_smoke", seed=7, policy="expert", image_width=4, image_height=3)
    stepped = await send("step", nworld=1)
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
    args = parser.parse_args()
    path = asyncio.run(run(args.output))
    print(f"OK {path}")


if __name__ == "__main__":
    main()
