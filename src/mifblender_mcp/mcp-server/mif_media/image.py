"""One picture fitted for a client's MCP image block: a long edge and a byte limit, standard library only (gif.py's reader).

WHY: a model sees a picture only through an image content block. Base64 inside a text block is read as text, and a client
caps text, so a render's preview either cost tens of thousands of tokens or was withheld. The limits, and why:
  DEFAULT_MAX_PX 1024    a vision model pays width x height / 750 tokens, so 1024 x 576 is about 790 tokens and a line of
                         text in a capture still reads. A longer edge than HARD_MAX_PX is scaled by the API before the model
                         sees it, so it would cost bytes and show nothing more.
  DEFAULT_MAX_BYTES      512 KB of PNG is 683 KB of base64. Every earlier image is sent again on every request and a request
                         tops out at 32 MB, so this leaves room for about 45 pictures in one conversation.
A picture over a limit is scaled down, never refused for its size. Refusal is left for what cannot be read here (JPG over
the limits, EXR, a broken file), and the caller reports that beside the file's path.
"""
import base64
import math
import os
import struct
import zlib

from .gif import Refusal, read_bmp, read_png_bytes

DEFAULT_MAX_PX = 1024
HARD_MAX_PX = 1568
DEFAULT_MAX_BYTES = 512 * 1024
MIN_PX = 32
# A decode takes about a second per megapixel here. 64 MP is an 8K x 8K sheet, about a minute with the server held.
MAX_SOURCE_PIXELS = 64_000_000
_PNG = b"\x89PNG\r\n\x1a\n"


def _png_size(data):
    return struct.unpack(">II", data[16:24]) if len(data) >= 24 else (0, 0)


def _jpeg_size(data):
    pos = 2
    while pos + 9 < len(data):
        if data[pos] != 0xFF:
            pos += 1
            continue
        kind = data[pos + 1]
        if kind in (0xFF, 0x00):
            pos += 1
        elif 0xC0 <= kind <= 0xCF and kind not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack(">HH", data[pos + 5:pos + 9])
            return w, h
        else:
            pos += 2 + struct.unpack(">H", data[pos + 2:pos + 4])[0]
    return 0, 0


def scale(w, h, rows, out_w, out_h):
    """Down to out_w x out_h, averaging four samples per pixel: nearest-pixel breaks the thin lines a capture is read for."""
    fx, fy = w / out_w, h / out_h
    x0 = [3 * min(w - 1, int((x + 0.25) * fx)) for x in range(out_w)]
    x1 = [3 * min(w - 1, int((x + 0.75) * fx)) for x in range(out_w)]
    out = []
    for y in range(out_h):
        a, b = rows[min(h - 1, int((y + 0.25) * fy))], rows[min(h - 1, int((y + 0.75) * fy))]
        line = bytearray(out_w * 3)
        at = 0
        for s, t in zip(x0, x1):
            line[at] = (a[s] + a[t] + b[s] + b[t]) >> 2
            line[at + 1] = (a[s + 1] + a[t + 1] + b[s + 1] + b[t + 1]) >> 2
            line[at + 2] = (a[s + 2] + a[t + 2] + b[s + 2] + b[t + 2]) >> 2
            at += 3
        out.append(line)
    return out


def encode_png(w, h, rows):
    """An 8-bit RGB PNG of rows, each line filtered against the one above (a capture's smooth areas compress far better)."""
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    raw = bytearray()
    prev = bytes(w * 3)
    for row in rows:
        raw.append(2)
        raw += bytes([(c - p) & 255 for c, p in zip(row, prev)])
        prev = row
    return (_PNG + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))


def _shrunk(w, h, longest):
    k = longest / max(w, h)
    return max(1, round(w * k)), max(1, round(h * k))


def fit(data, name="the picture", max_px=None, max_bytes=None):
    """{data, mime, width, height, sourceWidth, sourceHeight, scaled} - bytes a client can show, within both limits.

    max_px is capped at HARD_MAX_PX. Raises Refusal when the format cannot be read here, or when even MIN_PX is over
    max_bytes (only a caller's own tiny limit can do that)."""
    limit_px = min(max(int(max_px or DEFAULT_MAX_PX), MIN_PX), HARD_MAX_PX)
    limit_bytes = int(max_bytes or DEFAULT_MAX_BYTES)
    if data[:8] == _PNG:
        mime, size = "image/png", _png_size(data)
    elif data[:2] == b"\xff\xd8":
        mime, size = "image/jpeg", _jpeg_size(data)
    elif data[:2] == b"BM":
        mime, size = None, None
    else:
        raise Refusal("%s is not a PNG, JPG or BMP, so it cannot be shown here." % name)
    if size and max(size) <= limit_px and len(data) <= limit_bytes:
        return {"data": data, "mime": mime, "width": size[0], "height": size[1], "sourceWidth": size[0],
                "sourceHeight": size[1], "scaled": False}
    if mime == "image/jpeg":
        raise Refusal("%s is a JPG of %dx%d and %d bytes, over the %d px or %d byte limit, and JPG cannot be decoded here "
                      "to scale it." % (name, size[0], size[1], len(data), limit_px, limit_bytes))
    if size and size[0] * size[1] > MAX_SOURCE_PIXELS:
        raise Refusal("%s is %dx%d, over the %d megapixels this server decodes." % (name, size[0], size[1],
                                                                                    MAX_SOURCE_PIXELS // 1000000))
    if mime is None:
        w, h, rows = read_bmp(name) if isinstance(name, str) and os.path.isfile(name) else (0, 0, [])
        if not rows:
            raise Refusal("%s is a BMP that is not a file on disk." % name)
    else:
        w, h, rows = read_png_bytes(data, name)
    out_w, out_h = _shrunk(w, h, min(limit_px, max(w, h)))
    while True:
        scaled = rows if (out_w, out_h) == (w, h) else scale(w, h, rows, out_w, out_h)
        png = encode_png(out_w, out_h, scaled)
        if len(png) <= limit_bytes:
            return {"data": png, "mime": "image/png", "width": out_w, "height": out_h, "sourceWidth": w, "sourceHeight": h,
                    "scaled": (out_w, out_h) != (w, h)}
        if max(out_w, out_h) <= MIN_PX:
            raise Refusal("%s does not fit %d bytes even at %d px." % (name, limit_bytes, MIN_PX))
        # BYTES GO WITH THE AREA, so the square root aims at the limit in one step; the margin is for a noisy picture.
        next_edge = int(max(out_w, out_h) * math.sqrt(limit_bytes / len(png)) * 0.95)
        out_w, out_h = _shrunk(w, h, max(MIN_PX, min(next_edge, max(out_w, out_h) - 1)))


def fit_file(path, max_px=None, max_bytes=None):
    """fit() on a file. Raises Refusal for a file that is not there, which a server on another machine will meet."""
    if not os.path.isfile(path):
        raise Refusal("%s is not a file this server can read." % path)
    with open(path, "rb") as fh:
        return fit(fh.read(), path, max_px, max_bytes)


def as_base64(fitted):
    return base64.b64encode(fitted["data"]).decode("ascii")
