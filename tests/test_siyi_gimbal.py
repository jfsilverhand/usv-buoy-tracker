"""SIYI A8 mini packet encoding/decoding, checked against SIYI SDK example frames.

Reference frames (seq = 0) are the fixed commands published with the SIYI gimbal SDK,
as reproduced in github.com/thiagolages/A8mini-gimbal-camera-control (A8miniControl.c).
"""
import struct

import pytest

from siyi_gimbal import SIYIGimbalController as Gimbal


def offline_gimbal():
    """Instance without opening a TCP connection."""
    g = Gimbal.__new__(Gimbal)
    g.seq = 65535  # next packet uses seq 0, like the reference frames
    return g


@pytest.mark.parametrize("cmd, data, expected", [
    (Gimbal.CMD_FIRMWARE_VER, b"", "55 66 01 00 00 00 00 01 64 c4"),
    (Gimbal.CMD_CENTER, b"\x01", "55 66 01 01 00 00 00 08 01 d1 12"),
    (Gimbal.CMD_GET_ATTITUDE, b"", "55 66 01 00 00 00 00 0d e8 05"),
    (Gimbal.CMD_GIMBAL_ROTATE, struct.pack("<bb", 0, 0), "55 66 01 02 00 00 00 07 00 00 f1 24"),
    (Gimbal.CMD_GIMBAL_ROTATE, struct.pack("<bb", 100, 100), "55 66 01 02 00 00 00 07 64 64 3d cf"),
    (Gimbal.CMD_GIMBAL_ROTATE, struct.pack("<bb", 45, 0), "55 66 01 02 00 00 00 07 2d 00 4b 54"),
])
def test_packets_match_sdk_reference(cmd, data, expected):
    assert offline_gimbal()._build_packet(cmd, data) == bytes.fromhex(expected)


def test_sequence_number_increments_and_wraps():
    g = offline_gimbal()
    first = g._build_packet(Gimbal.CMD_FIRMWARE_VER)
    second = g._build_packet(Gimbal.CMD_FIRMWARE_VER)
    assert struct.unpack_from("<H", first, 5)[0] == 0
    assert struct.unpack_from("<H", second, 5)[0] == 1


def test_rotation_speed_is_clamped(monkeypatch):
    g = offline_gimbal()
    sent = []
    monkeypatch.setattr(g, "_send_only", lambda cmd, data=b"": sent.append((cmd, data)))
    g.set_gimbal_rotation(250, -250.7)
    assert sent == [(Gimbal.CMD_GIMBAL_ROTATE, struct.pack("<bb", 100, -100))]


def _attitude_reply(yaw_ddeg, pitch_ddeg, roll_ddeg):
    g = offline_gimbal()
    payload = struct.pack("<hhhhhh", yaw_ddeg, pitch_ddeg, roll_ddeg, 0, 0, 0)
    frame = struct.pack("<HBHHB", Gimbal.HEADER, 0x02, len(payload), 7, Gimbal.CMD_GET_ATTITUDE) + payload
    return frame + struct.pack("<H", g._calc_crc(frame))


def test_parse_attitude_units_and_order():
    assert Gimbal.parse_attitude(_attitude_reply(123, -456, 7)) == pytest.approx((12.3, -45.6, 0.7))


def test_parse_attitude_skips_leading_garbage():
    assert Gimbal.parse_attitude(b"\x00\x01" + _attitude_reply(-900, 250, 0)) == pytest.approx((-90.0, 25.0, 0.0))


@pytest.mark.parametrize("resp", [None, b"", b"\x55\x66\x02", bytes.fromhex("55 66 01 00 00 00 00 01 64 c4")])
def test_parse_attitude_rejects_short_or_other_frames(resp):
    assert Gimbal.parse_attitude(resp) is None
