"""PNG in and out with the standard library, for rbx_screenshot: Studio hands back either PNG bytes or raw RGBA8, and the
answer must say whether the frame shows anything. Not a general PNG library: 8-bit RGB and RGBA, no interlace."""
import struct
import zlib

SIGNATURE = b"\x89PNG\r\n\x1a\n"


def encode_rgba(width, height, rgba):
    """Raw RGBA8 rows -> PNG bytes (filter 0 on every row)."""
    if len(rgba) != width * height * 4:
        raise ValueError("RGBA is %d bytes, %d x %d x 4 is %d" % (len(rgba), width, height, width * height * 4))
    stride = width * 4
    raw = b"".join(b"\x00" + rgba[y * stride:(y + 1) * stride] for y in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (SIGNATURE + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def header(data):
    """(width, height, bit depth, color type) of PNG bytes, or None when they are not a PNG."""
    if len(data) < 33 or not data.startswith(SIGNATURE) or data[12:16] != b"IHDR":
        return None
    w, h, depth, ctype = struct.unpack(">IIBB", data[16:26])
    return w, h, depth, ctype


def decode_rgba(data):
    """PNG bytes (8-bit RGB or RGBA, not interlaced) -> (width, height, RGBA bytes). A cut or damaged file raises
    ValueError, never struct.error, zlib.error or IndexError: the callers catch (OSError, ValueError)."""
    hd = header(data)
    if hd is None:
        raise ValueError("not a PNG")
    w, h, depth, ctype = hd
    if depth != 8 or ctype not in (2, 6) or data[28] != 0:
        raise ValueError("only 8-bit RGB/RGBA, not interlaced: depth %d, color type %d, interlace %d" % (depth, ctype, data[28]))
    try:
        return _decode(data, w, h, ctype)
    except (struct.error, zlib.error, IndexError) as exc:
        raise ValueError("the PNG is cut short or damaged (%d bytes): %s: %s" % (len(data), type(exc).__name__, exc)) from exc


def _decode(data, w, h, ctype):
    pos, idat = 8, b""
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        if kind == b"IDAT":
            idat += data[pos + 8:pos + 8 + n]
        pos += 12 + n
    raw = zlib.decompress(idat)
    bpp = 4 if ctype == 6 else 3
    stride = w * bpp
    out, prev = bytearray(), bytearray(stride)
    for y in range(h):
        f = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if f == 1:
                line[i] = (line[i] + a) & 255
            elif f == 2:
                line[i] = (line[i] + b) & 255
            elif f == 3:
                line[i] = (line[i] + (a + b) // 2) & 255
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        prev = line
        if bpp == 4:
            out += line
        else:
            for i in range(0, stride, 3):
                out += line[i:i + 3] + b"\xff"
    return w, h, bytes(out)


# Studio's empty viewport and pane color, exact RGB (dark theme, Studio 0.748): a minimized Studio's capture is this one
# color and nothing else (RB1037), and a play-test window that drew no game is this color around Studio's title bar,
# menu and empty panes. Measured 2026-11-23 over the 202 captures in the daemon's captures folder: the empty test
# windows (play_client 2026-11-22 10:52 and 20:03, 2026-11-23 11:02, 2576 x 1456) held it on 0.924 of their samples
# with 11 distinct colors and a luminance spread of 32.09 (the white title bar), so the color and spread floors passed
# them as frames; uncropped test windows that did draw a game held it on 0.232 at most (the Studio chrome around the
# game), cropped game frames and edit-mode captures on 0.000. The floor sits between, at 0.6. A frame mostly ONE other
# color is not judged by this: a deliberate white glyph probe was 0.992 white, a game's green sky 0.876.
STUDIO_EMPTY_RGB = (25, 26, 31)
STUDIO_EMPTY_FLOOR = 0.6


def frame_stats(width, height, rgba, samples=4096):
    """Is the frame blank? Distinct colors (quantized to 4 bits a channel) and luminance spread over a sample grid, and
    the share of samples that are exactly Studio's empty viewport color (an empty Studio window, not a game frame)."""
    n = width * height
    if n == 0:
        return {"samples": 0, "distinctColors": 0, "lumaStd": 0.0, "studioEmptyShare": 0.0, "blank": True,
                "blankBy": "empty"}
    step = max(1, n // samples)
    colors, lumas, empty = set(), [], 0
    for i in range(0, n, step):
        r, g, b = rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]
        colors.add((r >> 4, g >> 4, b >> 4))
        lumas.append(0.2126 * r + 0.7152 * g + 0.0722 * b)
        empty += (r, g, b) == STUDIO_EMPTY_RGB
    mean = sum(lumas) / len(lumas)
    std = (sum((x - mean) ** 2 for x in lumas) / len(lumas)) ** 0.5
    share = empty / len(lumas)
    by = ("colors" if len(colors) < 4 else "spread" if std < 2.0
          else "studio-empty" if share >= STUDIO_EMPTY_FLOOR else None)
    out = {"samples": len(lumas), "distinctColors": len(colors), "lumaMean": round(mean, 1), "lumaStd": round(std, 2),
           "studioEmptyShare": round(share, 3), "blank": by is not None}
    if by:
        out["blankBy"] = by
    return out


def blank_reason(stats):
    """The error words for a blank frame, naming which floor it fell under."""
    if stats.get("blankBy") == "studio-empty":
        return ("the frame is blank: %.0f%% of it is Studio's empty viewport color %s (floor %.0f%%), an empty Studio "
                "window rather than a game frame (%d distinct colors, luminance spread %s)" % (
                    100 * stats["studioEmptyShare"], STUDIO_EMPTY_RGB, 100 * STUDIO_EMPTY_FLOOR,
                    stats["distinctColors"], stats["lumaStd"]))
    return "the frame is blank: %d distinct colors, luminance spread %s" % (stats["distinctColors"], stats["lumaStd"])
