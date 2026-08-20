import struct
import json
import asyncio
from pathlib import Path

import pytest

from mjwarp_demo.protocol import decode_message, encode_message
from mjwarp_demo.protocol import PROTOCOL_VERSION
from mjwarp_demo.server import DemoServer


def test_protocol_roundtrip() -> None:
    encoded = encode_message({"type": "hello", "request_id": 7, "values": [1, 2, 3]})
    (length,) = struct.unpack("<I", encoded[:4])
    assert length == len(encoded) - 4
    assert decode_message(encoded[4:]) == {"type": "hello", "request_id": 7, "values": [1, 2, 3]}


def test_protocol_rejects_nan() -> None:
    with pytest.raises(ValueError):
        encode_message({"bad": float("nan")})


def test_model_list_only_returns_matching_behavior_policies(tmp_path: Path) -> None:
    artifact = tmp_path / "Artifacts" / "planar_push" / "bc_test"
    artifact.mkdir(parents=True)
    (artifact / "model_spec.json").write_text(
        json.dumps(
            {
                "artifact_id": "planar_push/bc_test",
                "scenario": "planar_push",
                "model_type": "behavior_cloning",
                "input_dim": 8,
                "action_dim": 2,
            }
        ),
        encoding="utf-8",
    )
    server = DemoServer(None, tmp_path / "Datasets", "cuda:0", artifacts_dir=tmp_path / "Artifacts")
    reply = asyncio.run(
        server.handle(
            {
                "protocol_version": PROTOCOL_VERSION,
                "type": "model_list",
                "request_id": 9,
                "scenario": "planar_push",
            }
        )
    )
    assert reply["request_id"] == 9
    assert reply["models"][0]["artifact_id"] == "planar_push/bc_test"
