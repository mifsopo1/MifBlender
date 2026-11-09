"""An animated GIF from rendered frames - PNG or BMP files from a capture or a sequence render - standard library only.

WHY (2026-10-11, spec "An animated GIF from a capture or a sequence render"): render_movie_capture writes PNG, JPG, BMP,
EXR or AVI and render_sequence writes image sequences; a report or an issue thread wants one small moving picture. The
server encodes it itself, so nothing new is installed: the PNG reader (zlib and the five PNG filters), a nearest-pixel
downscale, a fixed 6 x 7 x 6 palette with a 4 x 4 ordered dither (one palette for every frame, so no frame flickers its
colours), and the GIF's own LZW. JPG and EXR are refused by name: their decoders are not in the standard library.

Limits, each a parameter with a ceiling: maxWidth (the frames are scaled down, never up), maxFrames, every (keep one frame
in N), fps (GIF delays are whole hundredths of a second, so the fps actually written is reported).
"""
import glob
import os
import struct
import time
import zlib

MAX_FRAMES_CAP = 600
MAX_WIDTH_CAP = 1280


class Refusal(Exception):
    pass


# ------------------------------------------------------------------------------------------------------------- reading
def read_png(path):
    """(width, height, rows) - rows a list of bytearrays of R, G, B per pixel, alpha composited over black."""
    with open(path, "rb") as fh:
        return read_png_bytes(fh.read(), path)


def read_png_bytes(data, path="the PNG"):
    """read_png on bytes already in memory (mif_terrain.elevation decodes downloaded tiles with it); path names it in a refusal."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise Refusal("%s is not a PNG (its signature is wrong)." % path)
    pos, idat, plte, trns = 8, [], None, None
    w = h = depth = ctype = interlace = None
    while pos + 8 <= len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + n]
        pos += 12 + n
        if kind == b"IHDR":
            w, h, depth, ctype, _c, _f, interlace = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"PLTE":
            plte = chunk
        elif kind == b"tRNS":
            trns = chunk
        elif kind == b"IDAT":
            idat.append(chunk)
        elif kind == b"IEND":
            break
    if w is None:
        raise Refusal("%s has no IHDR chunk." % path)
    if interlace:
        raise Refusal("%s is interlaced (Adam7), which this reader does not take - render it non-interlaced." % path)
    if depth not in (8, 16) or ctype not in (0, 2, 3, 4, 6) or (ctype == 3 and depth != 8):
        raise Refusal("%s is %d-bit colour type %d; this reader takes 8 or 16-bit grey, RGB, RGBA, grey+alpha and 8-bit "
                      "palette PNGs." % (path, depth, ctype))
    chans = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    bpp = chans * depth // 8
    stride = w * bpp
    raw = zlib.decompress(b"".join(idat))
    rows, prev, off = [], bytearray(stride), 0
    for _y in range(h):
        ft = raw[off]
        line = bytearray(raw[off + 1:off + 1 + stride])
        off += 1 + stride
        if ft == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 255
        elif ft == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 255
        elif ft == 3:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 255
        elif ft == 4:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 255
        elif ft != 0:
            raise Refusal("%s has an unknown PNG filter type %d." % (path, ft))
        prev = line
        rows.append(_to_rgb(line, w, ctype, depth, plte, trns))
    return w, h, rows


def _to_rgb(line, w, ctype, depth, plte, trns):
    out = bytearray(w * 3)
    step = depth // 8
    if ctype == 2:
        if step == 1:
            return line
        for x in range(w):
            out[3 * x:3 * x + 3] = bytes((line[6 * x], line[6 * x + 2], line[6 * x + 4]))
        return out
    if ctype == 6:
        n = 4 * step
        for x in range(w):
            a = line[n * x + 3 * step]
            r, g, b = line[n * x], line[n * x + step], line[n * x + 2 * step]
            out[3 * x:3 * x + 3] = bytes((r * a // 255, g * a // 255, b * a // 255))
        return out
    if ctype == 0 or ctype == 4:
        n = (2 if ctype == 4 else 1) * step
        for x in range(w):
            v = line[n * x]
            if ctype == 4:
                v = v * line[n * x + step] // 255
            out[3 * x:3 * x + 3] = bytes((v, v, v))
        return out
    # palette
    for x in range(w):
        k = line[x]
        r, g, b = plte[3 * k:3 * k + 3] if plte and 3 * k + 3 <= len(plte) else (0, 0, 0)
        a = trns[k] if trns and k < len(trns) else 255
        out[3 * x:3 * x + 3] = bytes((r * a // 255, g * a // 255, b * a // 255))
    return out


def read_bmp(path):
    """An uncompressed 24 or 32-bit BMP (what render_movie_capture's BMP writes): (width, height, rows)."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:2] != b"BM":
        raise Refusal("%s is not a BMP." % path)
    off = struct.unpack("<I", data[10:14])[0]
    w, h, _planes, bits, comp = struct.unpack("<iiHHI", data[18:34])
    if bits not in (24, 32) or comp not in (0, 3):
        raise Refusal("%s is a %d-bit BMP (compression %d); this reader takes uncompressed 24 and 32-bit." % (path, bits, comp))
    bpp = bits // 8
    stride = (w * bpp + 3) & ~3
    rows = []
    for y in range(abs(h)):
        src = off + (abs(h) - 1 - y if h > 0 else y) * stride
        out = bytearray(w * 3)
        for x in range(w):
            b, g, r = data[src + bpp * x:src + bpp * x + 3]
            out[3 * x:3 * x + 3] = bytes((r, g, b))
        rows.append(out)
    return w, abs(h), rows


def read_frame(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        return read_png(path)
    if ext == ".bmp":
        return read_bmp(path)
    raise Refusal("%s is %s; frames must be PNG or BMP (JPG and EXR decoders are not in the standard library - render "
                  "the capture as PNG)." % (path, ext or "a file with no extension"))


# --------------------------------------------------------------------------------------------------- colour and size
_BAYER = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]
_LEVELS = (6, 7, 6)
_PALETTE = bytearray()
for _r in range(6):
    for _g in range(7):
        for _b in range(6):
            _PALETTE += bytes((round(_r * 255 / 5), round(_g * 255 / 6), round(_b * 255 / 5)))
_PALETTE += bytes(3 * (256 - 252))


def _quant_table(levels):
    """[threshold][value] -> level index, so the dither is a table lookup per channel."""
    out = []
    for t in range(16):
        bias = (t + 0.5) / 16.0 - 0.5
        row = []
        for v in range(256):
            x = v / 255.0 * (levels - 1) + bias
            row.append(min(levels - 1, max(0, int(round(x)))))
        out.append(row)
    return out


_QR, _QG, _QB = (_quant_table(n) for n in _LEVELS)


def to_indices(w, h, rows, out_w, out_h):
    """Nearest-pixel scaled and dithered to the fixed palette: one bytearray of out_w * out_h indices."""
    idx = bytearray(out_w * out_h)
    xs = [min(w - 1, x * w // out_w) for x in range(out_w)]
    for y in range(out_h):
        src = rows[min(h - 1, y * h // out_h)]
        brow = _BAYER[y & 3]
        base = y * out_w
        for x in range(out_w):
            sx = 3 * xs[x]
            t = brow[x & 3]
            idx[base + x] = _QR[t][src[sx]] * 42 + _QG[t][src[sx + 1]] * 6 + _QB[t][src[sx + 2]]
    return idx


# ------------------------------------------------------------------------------------------------------------- writing
def _lzw(indices, min_code=8):
    """GIF LZW of a byte string of palette indices, as data sub-blocks."""
    clear, end = 1 << min_code, (1 << min_code) + 1
    size, next_code = min_code + 1, end + 1
    table = {bytes((i,)): i for i in range(clear)}
    out, buf, bits = bytearray(), 0, 0

    def emit(code):
        nonlocal buf, bits
        buf |= code << bits
        bits += size
        while bits >= 8:
            out.append(buf & 255)
            buf >>= 8
            bits -= 8
    emit(clear)
    w = b""
    for b in indices:
        wc = w + bytes((b,))
        if wc in table:
            w = wc
            continue
        emit(table[w])
        if next_code < 4096:
            table[wc] = next_code
            next_code += 1
            if next_code > (1 << size) and size < 12:
                size += 1
        else:
            emit(clear)
            table = {bytes((i,)): i for i in range(clear)}
            size, next_code = min_code + 1, end + 1
        w = bytes((b,))
    if w:
        emit(table[w])
    emit(end)
    if bits:
        out.append(buf & 255)
    blocks = bytearray()
    for i in range(0, len(out), 255):
        part = out[i:i + 255]
        blocks.append(len(part))
        blocks += part
    blocks.append(0)
    return bytes(blocks)


def write_gif(path, frames, width, height, delay_cs, loop=True):
    with open(path, "wb") as fh:
        fh.write(b"GIF89a" + struct.pack("<HHBBB", width, height, 0xF7, 0, 0) + bytes(_PALETTE))
        if loop:
            fh.write(b"\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00")
        for idx in frames:
            fh.write(b"\x21\xF9\x04\x04" + struct.pack("<H", delay_cs) + b"\x00\x00")
            fh.write(b"\x2C" + struct.pack("<HHHHB", 0, 0, width, height, 0))
            fh.write(b"\x08" + _lzw(idx))
        fh.write(b"\x3B")


# ----------------------------------------------------------------------------------------------------------------- run
def run(files=None, folder=None, pattern="*.png", out=None, fps=15.0, max_width=480, every=1, max_frames=150,
        loop=True):
    started = time.time()
    try:
        if (files is None) == (folder is None):
            raise Refusal("give files (a list of frame paths, in order) or folder (with pattern, default *.png), one "
                          "of the two.")
        if files is not None:
            if not isinstance(files, list) or not files or not all(isinstance(f, str) for f in files):
                raise Refusal("files is a non-empty list of frame paths, in order.")
            paths = list(files)
        else:
            if not isinstance(folder, str) or not os.path.isdir(folder):
                raise Refusal("folder %r is not a folder." % (folder,))
            paths = sorted(glob.glob(os.path.join(folder, pattern or "*.png")))
            if not paths:
                raise Refusal("no file in %s matches %r - nothing to encode." % (folder, pattern))
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise Refusal("%d frame file(s) are not there: %s" % (len(missing), ", ".join(missing[:5])))
        for name, v, lo, hi in (("fps", fps, 1, 50), ("max_width", max_width, 16, MAX_WIDTH_CAP),
                                ("every", every, 1, 1000), ("max_frames", max_frames, 1, MAX_FRAMES_CAP)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                raise Refusal("%s is %s to %s, not %r." % (name, lo, hi, v))
        if out is None:
            out = os.path.join(os.path.dirname(os.path.abspath(paths[0])), "capture.gif")
        if not isinstance(out, str) or not out.lower().endswith(".gif"):
            raise Refusal("out must be a path ending .gif, got %r." % (out,))
        if os.path.abspath(out) in {os.path.abspath(p) for p in paths}:
            raise Refusal("out is one of the frames.")
        picked = paths[::int(every)]
        dropped_by_cap = max(0, len(picked) - int(max_frames))
        picked = picked[:int(max_frames)]
        frames, size = [], None
        for p in picked:
            w, h, rows = read_frame(p)
            if size is None:
                out_w = min(w, int(max_width))
                out_h = max(1, round(h * out_w / float(w)))
                size = (w, h, out_w, out_h)
            elif (w, h) != size[:2]:
                raise Refusal("%s is %dx%d where the first frame is %dx%d - one capture's frames share a size."
                              % (p, w, h, size[0], size[1]))
            frames.append(to_indices(w, h, rows, size[2], size[3]))
        delay = max(2, round(100.0 / float(fps)))
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        write_gif(out, frames, size[2], size[3], delay, bool(loop))
    except Refusal as exc:
        return {"ok": False, "error": "%s Nothing was written." % exc}
    except (OSError, zlib.error, struct.error, IndexError) as exc:
        return {"ok": False, "error": "a frame could not be read (%s: %s). Nothing was written." % (type(exc).__name__, exc)}
    result = {"ok": True, "out": os.path.abspath(out), "frames": len(frames), "sourceFrames": len(paths),
              "every": int(every), "width": size[2], "height": size[3], "sourceSize": [size[0], size[1]],
              "delayCs": delay, "fpsWritten": round(100.0 / delay, 3), "loop": bool(loop),
              "bytes": os.path.getsize(out), "seconds": round(time.time() - started, 2),
              "palette": "fixed 6x7x6 (252 colours) with a 4x4 ordered dither, one palette for every frame"}
    if dropped_by_cap:
        result["droppedByMaxFrames"] = dropped_by_cap
    if result["fpsWritten"] != float(fps):
        result["fpsNote"] = "a GIF delay is whole hundredths of a second, so %s fps was written as %s" % (
            fps, result["fpsWritten"])
    return result
