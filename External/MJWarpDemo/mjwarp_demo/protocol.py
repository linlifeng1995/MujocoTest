from __future__ import annotations

import asyncio
import json
import struct
from typing import Any

PROTOCOL_VERSION = 3
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def encode_message(message: dict[str, Any]) -> bytes:
    payload = json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"message is too large: {len(payload)} bytes")
    return struct.pack("<I", len(payload)) + payload


def decode_message(payload: bytes) -> dict[str, Any]:
    message = json.loads(payload.decode("utf-8"))
    if not isinstance(message, dict):
        raise ValueError("protocol payload must be a JSON object")
    return message


async def read_message(reader: asyncio.StreamReader) -> dict[str, Any]:
    header = await reader.readexactly(4)
    (length,) = struct.unpack("<I", header)
    if length <= 0 or length > MAX_MESSAGE_BYTES:
        raise ValueError(f"invalid message length: {length}")
    return decode_message(await reader.readexactly(length))


async def write_message(writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
    writer.write(encode_message(message))
    await writer.drain()


def response(message_type: str, request_id: int, **payload: Any) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": message_type,
        "request_id": request_id,
        **payload,
    }
