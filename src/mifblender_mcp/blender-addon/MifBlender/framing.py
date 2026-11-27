"""Length-prefixed JSON framing for the MifBlender TCP protocol.

Wire format, identical on both ends:

    [ 4 bytes, big-endian uint32 = body length ][ body: UTF-8 JSON ]

Why length-prefixed and not newline-delimited or "recv until it parses":
a newline framer breaks the moment a payload contains a raw newline, and the
recv-until-parses trick silently mis-frames when two messages arrive in one TCP
segment. A length prefix has neither failure mode and handles multi-megabyte
bodies without a special case.

Adapted from blender-mcp (github.com/MCPBlender/blender-mcp),
(c) 2025 Siddharth Ahuja, MIT licence. Its notice travels with the code: it is reproduced in
full in LICENSE beside this file, which also carries this add-on's own MIT licence.

This module deliberately imports NOTHING from bpy. It is safe to call from the
socket thread, and only from there -- see server.py for the threading contract.
"""

from __future__ import annotations

import json
import socket
import struct
import time

HEADER_SIZE = 4
HEADER_FORMAT = ">I"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024  # 64 MiB


def pack_json_message(payload) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    if len(body) > MAX_MESSAGE_BYTES:
        raise ValueError(
            "frame too large: %d bytes (cap %d)" % (len(body), MAX_MESSAGE_BYTES)
        )
    return struct.pack(HEADER_FORMAT, len(body)) + body


def send_json_message(sock: socket.socket, payload) -> None:
    sock.sendall(pack_json_message(payload))


def recv_exact(sock: socket.socket, n: int, deadline=None) -> bytes:
    """Read exactly n bytes or raise. A short recv is normal on TCP, not an error.

    Into one preallocated buffer (FOUND 2026-11-22: `buf += chunk` on bytes copied the whole buffer per chunk, so a
    64 MiB frame read in 64 KiB chunks moved about 32 GB). `deadline` is a time.monotonic() second the WHOLE read
    must end by - a per-recv timeout alone lets a client that drips one byte a second hold the socket forever, which
    is what a caller that has not shown its token yet must never be able to do. Past it, socket.timeout."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        if deadline is not None:
            left = deadline - time.monotonic()
            if left <= 0:
                raise socket.timeout("frame not read by its deadline (%d of %d bytes)" % (got, n))
            sock.settimeout(left)
        count = sock.recv_into(view[got:], n - got)
        if not count:
            raise ConnectionError("socket closed mid-frame (%d of %d bytes)" % (got, n))
        got += count
    return bytes(buf)


def receive_framed_bytes(sock: socket.socket, timeout=None, deadline=None, max_bytes=MAX_MESSAGE_BYTES) -> bytes:
    """One frame's body. `deadline` bounds the whole frame (see recv_exact); `max_bytes` lowers the cap for a frame
    read before its sender is known."""
    if timeout is not None:
        sock.settimeout(timeout)
    header = recv_exact(sock, HEADER_SIZE, deadline)
    (length,) = struct.unpack(HEADER_FORMAT, header)
    if length <= 0 or length > min(max_bytes, MAX_MESSAGE_BYTES):
        raise ValueError("invalid frame length: %d (cap %d)" % (length, min(max_bytes, MAX_MESSAGE_BYTES)))
    return recv_exact(sock, length, deadline)


def receive_framed_json(sock: socket.socket, timeout=None, deadline=None, max_bytes=MAX_MESSAGE_BYTES):
    return json.loads(receive_framed_bytes(sock, timeout, deadline, max_bytes).decode("utf-8"))
