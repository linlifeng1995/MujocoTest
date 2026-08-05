import struct

import pytest

from mjwarp_demo.protocol import decode_message, encode_message


def test_protocol_roundtrip() -> None:
    encoded = encode_message({"type": "hello", "request_id": 7, "values": [1, 2, 3]})
    (length,) = struct.unpack("<I", encoded[:4])
    assert length == len(encoded) - 4
    assert decode_message(encoded[4:]) == {"type": "hello", "request_id": 7, "values": [1, 2, 3]}


def test_protocol_rejects_nan() -> None:
    with pytest.raises(ValueError):
        encode_message({"bad": float("nan")})
