"""Roblox's binary place format (.rbxl): just enough to put ServerStorage.MifBridgePort into a COPY of a binary place, so
a lane Studio can open a person's binary save on its own port. The person's file is only read.

The format (chunked, little-endian; the layout every open reader of it documents):
    header   "<roblox!" 89 FF 0D 0A 1A 0A, u16 version 0, i32 class count, i32 instance count, 8 reserved bytes
    chunk    4-byte name, u32 compressed length (0 = stored), u32 length, 4 reserved bytes, then the data, LZ4 block or a
             zstd frame (magic 28 B5 2F FD) when compressed
    INST     u32 class id, string class name, u8 format (1 = service, a marker byte per instance follows), u32 count,
             the referents (interleaved i32, transformed, delta coded)
    PROP     u32 class id, string property name, u8 type, one value per instance of the class, in INST order
    PRNT     u8 version 0, u32 count, child referents, parent referents (both interleaved i32, delta; -1 is no parent)
    END      "</roblox>", stored
A string is u32 length + bytes. An interleaved array of n values of width w holds byte 0 of every value, then byte 1, ...
(each value big-endian); an integer is "transformed" (zigzag: (x << 1) ^ (x >> bits-1)) before.

Only the chunks a marker changes are decoded: every INST (referents, class names), the IntValue class's PROPs and PRNT.
Those are written back stored (uncompressed); every other chunk is copied byte for byte. A chunk that needs decoding
and is zstd-compressed is refused when Python has no zstd (3.14's compression.zstd, or the zstandard package). An
IntValue property of a type this module cannot extend by one value is refused, never guessed. Standard library only.
"""
import itertools
import struct

MAGIC = b"<roblox!\x89\xff\r\n\x1a\n"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
MARKER_NAME = "MifBridgePort"

T_STRING, T_BOOL, T_INT32, T_FLOAT32, T_ENUM, T_REF, T_INT64, T_SHARED, T_UNIQUEID, T_CAPABILITIES = (
    0x01, 0x02, 0x03, 0x04, 0x12, 0x13, 0x1B, 0x1C, 0x1F, 0x21)
# Fixed-width interleaved types a new instance can be given a zero value in (the width is checked against the data).
FIXED = {T_INT32: 4, T_FLOAT32: 4, T_ENUM: 4, T_INT64: 8, T_UNIQUEID: 16, T_CAPABILITIES: 8}


class BinaryPlaceError(ValueError):
    """The file is not a binary place this module can mark: the message says why."""


# ---------------------------------------------------------------- compression
def lz4_decompress(src, size):
    """One LZ4 block (no frame) to exactly size bytes."""
    dst = bytearray()
    i, n = 0, len(src)
    try:
        while i < n:
            token = src[i]
            i += 1
            lit = token >> 4
            if lit == 15:
                while True:
                    b = src[i]
                    i += 1
                    lit += b
                    if b != 255:
                        break
            dst += src[i:i + lit]
            i += lit
            if len(dst) > size:
                raise BinaryPlaceError("an LZ4 block decodes past the %d bytes its header says" % size)
            if i >= n:
                break
            off = src[i] | (src[i + 1] << 8)
            i += 2
            ml = token & 15
            if ml == 15:
                while True:
                    b = src[i]
                    i += 1
                    ml += b
                    if b != 255:
                        break
            ml += 4
            start = len(dst) - off
            if off == 0 or start < 0:
                raise BinaryPlaceError("a broken LZ4 block (offset %d at %d)" % (off, i))
            if len(dst) + ml > size:
                # a damaged length would otherwise grow the output without bound before the size check below
                raise BinaryPlaceError("an LZ4 block decodes past the %d bytes its header says" % size)
            if off >= ml:
                dst += dst[start:start + ml]
            else:
                while ml > 0:                       # an overlapping match repeats the last off bytes
                    take = min(off, ml)
                    dst += dst[start:start + take]
                    start += take
                    ml -= take
    except IndexError as exc:
        raise BinaryPlaceError("a truncated LZ4 block") from exc
    if len(dst) != size:
        raise BinaryPlaceError("an LZ4 block decoded to %d bytes, its header says %d" % (len(dst), size))
    return bytes(dst)


def _zstd_decompress(src, size):
    try:
        from compression import zstd as _z          # Python 3.14
        out = _z.decompress(src)
    except ImportError:
        try:
            import zstandard as _z                   # the zstandard package, when installed
            out = _z.ZstdDecompressor().decompress(src, max_output_size=size)
        except ImportError as exc:
            raise BinaryPlaceError("a chunk the marker must change is zstd-compressed and this Python has no zstd "
                                   "(Python 3.14, or pip install zstandard)") from exc
    if len(out) != size:
        raise BinaryPlaceError("a zstd chunk decoded to %d bytes, its header says %d" % (len(out), size))
    return out


# ---------------------------------------------------------------- arrays
def _deinterleave(data, n, w):
    if n < 0 or len(data) != n * w:
        # checked before allocating: a damaged count (a u32) would otherwise ask for gigabytes
        raise BinaryPlaceError("an array of %d values of %d bytes holds %d bytes" % (n, w, len(data)))
    out = bytearray(n * w)
    for j in range(w):
        out[j::w] = data[j * n:(j + 1) * n]
    return bytes(out)


def _interleave(raw, n, w):
    out = bytearray(n * w)
    for j in range(w):
        out[j * n:(j + 1) * n] = raw[j::w]
    return bytes(out)


def read_i32s(data, n, delta=False):
    vals = [(u >> 1) ^ -(u & 1) for u in struct.unpack(">%dI" % n, _deinterleave(data, n, 4))]
    return list(itertools.accumulate(vals)) if delta else vals


def write_i32s(vals, delta=False):
    if delta:
        vals = [v - p for v, p in zip(vals, [0] + list(vals[:-1]))]
    raw = struct.pack(">%dI" % len(vals), *[((v << 1) ^ (v >> 31)) & 0xFFFFFFFF for v in vals])
    return _interleave(raw, len(vals), 4)


def read_i64s(data, n):
    return [(u >> 1) ^ -(u & 1) for u in struct.unpack(">%dQ" % n, _deinterleave(data, n, 8))]


def write_i64s(vals):
    raw = struct.pack(">%dQ" % len(vals), *[((v << 1) ^ (v >> 63)) & 0xFFFFFFFFFFFFFFFF for v in vals])
    return _interleave(raw, len(vals), 8)


def _string(data, pos):
    (n,) = struct.unpack_from("<I", data, pos)
    s = data[pos + 4:pos + 4 + n]
    if len(s) != n:
        raise BinaryPlaceError("a truncated string")
    return s.decode("utf-8", "replace"), pos + 4 + n


def _pstr(s):
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


# ---------------------------------------------------------------- chunks
class Chunk:
    __slots__ = ("name", "clen", "ulen", "raw", "_data")

    def __init__(self, name, clen, ulen, raw, data=None):
        self.name, self.clen, self.ulen, self.raw, self._data = name, clen, ulen, raw, data

    @classmethod
    def stored(cls, name, data):
        return cls(name, 0, len(data), data, data)

    def data(self):
        if self._data is None:
            if self.clen == 0:
                self._data = self.raw
            elif self.raw[:4] == ZSTD_MAGIC:
                self._data = _zstd_decompress(self.raw, self.ulen)
            else:
                self._data = lz4_decompress(self.raw, self.ulen)
        return self._data

    def to_bytes(self):
        return self.name.encode("latin-1").ljust(4, b"\0") + struct.pack("<III", self.clen, self.ulen, 0) + self.raw


def parse(blob):
    """(class count, instance count, [Chunk], trailing bytes) of a binary place, or BinaryPlaceError."""
    if not blob.startswith(MAGIC):
        raise BinaryPlaceError("not a binary Roblox place (no <roblox! signature)")
    if len(blob) < 32:
        raise BinaryPlaceError("a truncated header")
    version, ncls, ninst = struct.unpack_from("<Hii", blob, 14)
    if version != 0:
        raise BinaryPlaceError("binary format version %d (only 0 is known)" % version)
    pos, chunks = 32, []
    while True:
        if pos + 16 > len(blob):
            raise BinaryPlaceError("the file ends before its END chunk")
        name = blob[pos:pos + 4].rstrip(b"\0").decode("latin-1")
        clen, ulen, _r = struct.unpack_from("<III", blob, pos + 4)
        size = clen if clen else ulen
        raw = blob[pos + 16:pos + 16 + size]
        if len(raw) != size:
            raise BinaryPlaceError("chunk %s at %d is truncated" % (name, pos))
        chunks.append(Chunk(name, clen, ulen, raw))
        pos += 16 + size
        if name == "END":
            break
    return ncls, ninst, chunks, blob[pos:]


def serialize(ncls, ninst, chunks, trailing=b""):
    return (MAGIC + struct.pack("<Hii", 0, ncls, ninst) + b"\0" * 8 + b"".join(c.to_bytes() for c in chunks) + trailing)


def _inst(c):
    d = c.data()
    (cid,) = struct.unpack_from("<I", d, 0)
    name, p = _string(d, 4)
    fmt = d[p]
    (n,) = struct.unpack_from("<I", d, p + 1)
    refs = read_i32s(d[p + 5:p + 5 + 4 * n], n, delta=True)
    return {"id": cid, "name": name, "format": fmt, "refs": refs, "rest": d[p + 5 + 4 * n:]}


def _inst_bytes(i):
    return (struct.pack("<I", i["id"]) + _pstr(i["name"]) + bytes([i["format"]]) + struct.pack("<I", len(i["refs"]))
            + write_i32s(i["refs"], delta=True) + i["rest"])


def _prop(c):
    d = c.data()
    (cid,) = struct.unpack_from("<I", d, 0)
    name, p = _string(d, 4)
    return {"id": cid, "name": name, "type": d[p], "values": d[p + 1:]}


def _prop_bytes(cid, name, typ, values):
    return struct.pack("<I", cid) + _pstr(name) + bytes([typ]) + values


def _prop_class(c):
    """The class id of a PROP chunk; reads only the head, so a chunk of another class is never decoded past it."""
    return struct.unpack_from("<I", c.data(), 0)[0]


def _strings(values, n):
    out, p = [], 0
    for _ in range(n):
        s, p = _string(values, p)
        out.append(s)
    if p != len(values):
        raise BinaryPlaceError("a string property holds %d bytes past its %d values" % (len(values) - p, n))
    return out


def _prnt(c):
    d = c.data()
    if d[0] != 0:
        raise BinaryPlaceError("PRNT version %d (only 0 is known)" % d[0])
    (n,) = struct.unpack_from("<I", d, 1)
    kids = read_i32s(d[5:5 + 4 * n], n, delta=True)
    pars = read_i32s(d[5 + 4 * n:5 + 8 * n], n, delta=True)
    return kids, pars


def _prnt_bytes(kids, pars):
    return b"\0" + struct.pack("<I", len(kids)) + write_i32s(kids, delta=True) + write_i32s(pars, delta=True)


def _sstr(c):
    """[(hash, bytes)] of an SSTR chunk (u32 version 0, u32 count, then a 16-byte hash and a string per entry)."""
    d = c.data()
    ver, n = struct.unpack_from("<II", d, 0)
    if ver != 0:
        raise BinaryPlaceError("SSTR version %d (only 0 is known)" % ver)
    out, p = [], 8
    for _ in range(n):
        h = d[p:p + 16]
        (m,) = struct.unpack_from("<I", d, p + 16)
        out.append((h, d[p + 20:p + 20 + m]))
        p += 20 + m
    return out


def _sstr_bytes(entries):
    return struct.pack("<II", 0, len(entries)) + b"".join(h + struct.pack("<I", len(s)) + s for h, s in entries)


def _append_value(prop, n, value, empty_shared=None):
    """prop's values with one more, for a new instance: value for Name and Value, else the type's empty value
    (empty_shared: the SSTR index of an empty shared string, for a SharedString property such as Tags)."""
    t, v = prop["type"], prop["values"]
    if t == T_SHARED and empty_shared is not None:
        if len(v) != n * 4:
            raise BinaryPlaceError("shared-string property %s holds %d bytes for %d values" % (prop["name"], len(v), n))
        return _interleave(_deinterleave(v, n, 4) + struct.pack(">I", empty_shared), n + 1, 4)
    if t == T_STRING:
        return v + _pstr(value if isinstance(value, str) else "")
    if t == T_BOOL:
        if len(v) != n:
            raise BinaryPlaceError("bool property %s holds %d bytes for %d values" % (prop["name"], len(v), n))
        return v + (b"\x01" if value is True else b"\0")
    if t == T_INT64:
        old = read_i64s(v, n)
        if not isinstance(value, int):
            # no value of our own (SourceAssetId): the one every instance of the class holds, when they agree (its
            # default as this file stores it, -1 on 0.748), else 0
            value = old[0] if old and all(x == old[0] for x in old) else 0
        return write_i64s(old + [value])
    if t in FIXED:
        w = FIXED[t]
        if len(v) != n * w:
            raise BinaryPlaceError("property %s (type 0x%02X) holds %d bytes for %d values of %d" % (
                prop["name"], t, len(v), n, w))
        return _interleave(_deinterleave(v, n, w) + b"\0" * w, n + 1, w)
    raise BinaryPlaceError("IntValue property %s has type 0x%02X, which the bridge cannot extend by one value: save the "
                           "place as .rbxlx instead" % (prop["name"], t))


def _add_child(ncls, ninst, chunks, trailing, insts, by_name, kids, pars, prnt_at, ss_ref, cls, name, value, vtype):
    """(new bytes, how): one instance of class cls named name with Value value (vtype T_INT64 or T_BOOL) added under
    ServerStorage (made when ss_ref is None), to the file's own class when it has one, else a new class. Shared by the
    port marker (an IntValue) and the bridge-launched marker (a BoolValue)."""
    one = write_i64s([value]) if vtype == T_INT64 else (b"\x01" if value is True else b"\0")
    blank = (lambda n: write_i64s([0] * n)) if vtype == T_INT64 else (lambda n: b"\0" * n)  # noqa: E731
    cv = by_name.get(cls)
    props = {}
    if cv:
        for k, c in enumerate(chunks):
            if c.name == "PROP" and _prop_class(c) == cv[1]["id"]:
                pp = _prop(c)
                props[pp["name"]] = (k, pp)
    every = [r for v in insts.values() for r in v["refs"]]
    next_ref = max(every + [-1]) + 1
    next_cls = max([v["id"] for v in insts.values()] + [ncls - 1]) + 1
    add_inst, add_prop = [], []
    if ss_ref is None:
        ss_ref, next_ref = next_ref, next_ref + 1
        sid, next_cls = next_cls, next_cls + 1
        add_inst.append(Chunk.stored("INST", _inst_bytes({"id": sid, "name": "ServerStorage", "format": 1,
                                                          "refs": [ss_ref], "rest": b"\x01"})))
        add_prop.append(Chunk.stored("PROP", _prop_bytes(sid, "Name", T_STRING, _pstr("ServerStorage"))))
        kids, pars = kids + [ss_ref], pars + [-1]
        ncls, ninst = ncls + 1, ninst + 1
    ref = next_ref
    if cv:
        i, v = cv
        n = len(v["refs"])
        if v["format"] != 0:
            raise BinaryPlaceError("the %s class is stored as a service" % cls)
        if "Name" not in props:
            raise BinaryPlaceError("the %s class has no Name property" % cls)
        empty_shared = None
        if any(p["type"] == T_SHARED for _pi, p in props.values()):
            # a SharedString property (Tags, on 0.748) points into SSTR: the new value points at an empty entry,
            # the file's own when it has one, else one added (its hash the MD5 of nothing)
            s_at = [k for k, c in enumerate(chunks) if c.name == "SSTR"]
            if len(s_at) > 1:
                raise BinaryPlaceError("%d SSTR chunks (one at most is expected)" % len(s_at))
            entries = _sstr(chunks[s_at[0]]) if s_at else []
            empty_shared = next((k for k, (_h, s) in enumerate(entries) if s == b""), None)
            if empty_shared is None:
                import hashlib
                entries.append((hashlib.md5(b"").digest(), b""))
                empty_shared = len(entries) - 1
                if s_at:
                    chunks[s_at[0]] = Chunk.stored("SSTR", _sstr_bytes(entries))
                else:
                    first = next(k for k, c in enumerate(chunks) if c.name == "INST")
                    chunks.insert(first, Chunk.stored("SSTR", _sstr_bytes(entries)))
                    props = {nm: (pi + (pi >= first), p) for nm, (pi, p) in props.items()}
                    prnt_at = [k + (k >= first) for k in prnt_at]
                    i += i >= first
        for pname, (pi, p) in props.items():
            val = name if pname == "Name" else value if pname == "Value" else None
            chunks[pi] = Chunk.stored("PROP", _prop_bytes(p["id"], pname, p["type"], _append_value(p, n, val, empty_shared)))
        if "Value" not in props:
            add_prop.append(Chunk.stored("PROP", _prop_bytes(v["id"], "Value", vtype, blank(n) + one)))
        chunks[i] = Chunk.stored("INST", _inst_bytes(dict(v, refs=v["refs"] + [ref])))
        how = "a %s %s = %s added to the copy's %s class, under ServerStorage" % (cls, name, value, cls)
    else:
        cid, ncls = next_cls, ncls + 1
        add_inst.append(Chunk.stored("INST", _inst_bytes({"id": cid, "name": cls, "format": 0, "refs": [ref],
                                                          "rest": b""})))
        add_prop += [Chunk.stored("PROP", _prop_bytes(cid, "Name", T_STRING, _pstr(name))),
                     Chunk.stored("PROP", _prop_bytes(cid, "Value", vtype, one))]
        how = "a %s %s = %s (a new %s class) added to the copy, under ServerStorage" % (cls, name, value, cls)
    ninst += 1
    kids, pars = kids + [ref], pars + [ss_ref]
    chunks[prnt_at[0]] = Chunk.stored("PRNT", _prnt_bytes(kids, pars))
    # new INSTs after the last INST, new PROPs just before PRNT: every PROP follows its class's INST
    last_inst = max(i for i, c in enumerate(chunks) if c.name == "INST")
    chunks[last_inst + 1:last_inst + 1] = add_inst
    p_at = next(i for i, c in enumerate(chunks) if c.name == "PRNT")
    chunks[p_at:p_at] = add_prop
    return serialize(ncls, ninst, chunks, trailing), how


# ---------------------------------------------------------------- the marker
def read_marker(blob):
    """The value of ServerStorage's IntValue child named MifBridgePort, or None. BinaryPlaceError when unreadable."""
    return _mark(blob, None)[1]


def mark(blob, port):
    """(new bytes, how) with ServerStorage.MifBridgePort = port in the place: the file's own marker rewritten, or an
    IntValue added (to the file's IntValue class, or a new one), and ServerStorage added when the file has none.
    BinaryPlaceError when the place cannot be marked."""
    out, have, how = _mark(blob, int(port))
    if read_marker(out) != int(port):
        raise BinaryPlaceError("the marked copy does not read back MifBridgePort = %d" % port)
    return out, how


def _mark(blob, port):
    """_mark_chunks, with a damaged chunk (too short for its own fields: a 2-byte INST, an empty PRNT) answered as
    BinaryPlaceError, never struct.error or IndexError, so a caller's one except clause covers every unreadable file."""
    try:
        return _mark_chunks(blob, port)
    except BinaryPlaceError:
        raise
    except (struct.error, IndexError, UnicodeError, ValueError, OverflowError) as exc:
        raise BinaryPlaceError("a truncated or damaged chunk (%s: %s)" % (type(exc).__name__, exc)) from exc


def _mark_chunks(blob, port):
    ncls, ninst, chunks, trailing = parse(blob)
    insts = {i: _inst(c) for i, c in enumerate(chunks) if c.name == "INST"}
    by_name = {v["name"]: (i, v) for i, v in insts.items()}
    prnt_at = [i for i, c in enumerate(chunks) if c.name == "PRNT"]
    if len(prnt_at) != 1:
        raise BinaryPlaceError("%d PRNT chunks (one is expected)" % len(prnt_at))
    kids, pars = _prnt(chunks[prnt_at[0]])
    parent = dict(zip(kids, pars))
    ss = by_name.get("ServerStorage")
    ss_ref = ss[1]["refs"][0] if ss and ss[1]["refs"] else None
    iv = by_name.get("IntValue")
    props = {}
    if iv:
        for i, c in enumerate(chunks):
            if c.name == "PROP" and _prop_class(c) == iv[1]["id"]:
                p = _prop(c)
                props[p["name"]] = (i, p)
    have, at = None, None
    if iv and ss_ref is not None and "Name" in props:
        n = len(iv[1]["refs"])
        names = _strings(props["Name"][1]["values"], n)
        values = (read_i64s(props["Value"][1]["values"], n) if "Value" in props and props["Value"][1]["type"] == T_INT64
                  else [0] * n)
        for k, (ref, nm) in enumerate(zip(iv[1]["refs"], names)):
            if nm == MARKER_NAME and parent.get(ref) == ss_ref:
                have, at = values[k], k
                break
    if port is None or have == port:
        return blob, have, "the file already carries MifBridgePort = %s" % have
    if at is not None:
        n = len(iv[1]["refs"])
        if "Value" not in props or props["Value"][1]["type"] != T_INT64:
            raise BinaryPlaceError("the file's MifBridgePort has no Int64 Value to rewrite")
        values = read_i64s(props["Value"][1]["values"], n)
        values[at] = port
        i, p = props["Value"]
        chunks[i] = Chunk.stored("PROP", _prop_bytes(p["id"], p["name"], T_INT64, write_i64s(values)))
        return serialize(ncls, ninst, chunks, trailing), have, "MifBridgePort %s rewritten to %d in the copy" % (have, port)

    out, how = _add_child(ncls, ninst, chunks, trailing, insts, by_name, kids, pars, prnt_at, ss_ref, "IntValue",
                          MARKER_NAME, port, T_INT64)
    return out, have, how


# ---------------------------------------------------------------- the bridge-launched marker
LAUNCH_NAME = "MifBridgeLaunched"


def read_launched(blob):
    """True when ServerStorage has a BoolValue child named MifBridgeLaunched holding true. BinaryPlaceError when the
    file cannot be read."""
    return _launched(blob, False)[1]


def mark_launched(blob):
    """(new bytes, how) with ServerStorage.MifBridgeLaunched = true in the place: the file's own BoolValue set true, or
    one added (to the file's BoolValue class, or a new one), and ServerStorage added when the file has none. The plugin
    lets the agent raise the safety mode only in a place carrying it, and rbx_launch's copies count (Mifsopo,
    2026-11-22). BinaryPlaceError when the place cannot be marked."""
    out, _have, how = _launched(blob, True)
    if not read_launched(out):
        raise BinaryPlaceError("the copy does not read back ServerStorage.MifBridgeLaunched = true")
    return out, how


def _launched(blob, write):
    try:
        return _launched_chunks(blob, write)
    except BinaryPlaceError:
        raise
    except (struct.error, IndexError, UnicodeError, ValueError, OverflowError) as exc:
        raise BinaryPlaceError("a truncated or damaged chunk (%s: %s)" % (type(exc).__name__, exc)) from exc


def _launched_chunks(blob, write):
    ncls, ninst, chunks, trailing = parse(blob)
    insts = {i: _inst(c) for i, c in enumerate(chunks) if c.name == "INST"}
    by_name = {v["name"]: (i, v) for i, v in insts.items()}
    prnt_at = [i for i, c in enumerate(chunks) if c.name == "PRNT"]
    if len(prnt_at) != 1:
        raise BinaryPlaceError("%d PRNT chunks (one is expected)" % len(prnt_at))
    kids, pars = _prnt(chunks[prnt_at[0]])
    parent = dict(zip(kids, pars))
    ss = by_name.get("ServerStorage")
    ss_ref = ss[1]["refs"][0] if ss and ss[1]["refs"] else None
    bv = by_name.get("BoolValue")
    props = {}
    if bv:
        for i, c in enumerate(chunks):
            if c.name == "PROP" and _prop_class(c) == bv[1]["id"]:
                p = _prop(c)
                props[p["name"]] = (i, p)
    have, at = False, None
    if bv and ss_ref is not None and "Name" in props:
        n = len(bv[1]["refs"])
        names = _strings(props["Name"][1]["values"], n)
        vals = props["Value"][1]["values"] if "Value" in props and props["Value"][1]["type"] == T_BOOL else b"\0" * n
        for k, (ref, nm) in enumerate(zip(bv[1]["refs"], names)):
            if nm == LAUNCH_NAME and parent.get(ref) == ss_ref:
                have, at = vals[k:k + 1] == b"\x01", k
                break
    if not write or have:
        return blob, have, "the file already carries MifBridgeLaunched = true"
    if at is not None:
        if "Value" not in props or props["Value"][1]["type"] != T_BOOL:
            raise BinaryPlaceError("the file's MifBridgeLaunched has no bool Value to set")
        i, p = props["Value"]
        v = bytearray(p["values"])
        v[at] = 1
        chunks[i] = Chunk.stored("PROP", _prop_bytes(p["id"], p["name"], T_BOOL, bytes(v)))
        return serialize(ncls, ninst, chunks, trailing), have, "MifBridgeLaunched set true in the copy"
    out, how = _add_child(ncls, ninst, chunks, trailing, insts, by_name, kids, pars, prnt_at, ss_ref, "BoolValue",
                          LAUNCH_NAME, True, T_BOOL)
    return out, have, how


# ---------------------------------------------------------------- script sources (read only)
SCRIPT_CLASSES = ("Script", "LocalScript", "ModuleScript")


def instance_paths(blob):
    """[path (names from the root service)] of every instance in a binary place: INST, every PROP named Name, and PRNT;
    writes nothing. BinaryPlaceError when the file cannot be read."""
    _ncls, _ninst, chunks, _trailing = parse(blob)
    insts, names, parent = {}, {}, {}
    for c in chunks:
        if c.name == "INST":
            v = _inst(c)
            insts[v["id"]] = v
    for c in chunks:
        if c.name == "PROP":
            v = insts.get(_prop_class(c))
            if v is None:
                continue
            p = _prop(c)
            if p["type"] == T_STRING and p["name"] == "Name":
                names.update(zip(v["refs"], _strings(p["values"], len(v["refs"]))))
        elif c.name == "PRNT":
            kids, pars = _prnt(c)
            parent.update(zip(kids, pars))
    out = []
    for v in insts.values():
        for ref in v["refs"]:
            path, at, seen = [], ref, set()
            while at is not None and at != -1 and at not in seen:
                seen.add(at)
                path.append(names.get(at, "?"))
                at = parent.get(at)
            out.append(list(reversed(path)))
    return out


def value_at(blob, path):
    """(found, value, why) of the Value property of the instance at path (names from the root service) in a binary
    place: a bool, an Int64 or a string Value (BoolValue, IntValue, StringValue). found False when no instance is there;
    value None with why when its Value is of a type this reader does not decode. Writes nothing."""
    _ncls, _ninst, chunks, _trailing = parse(blob)
    insts = {}
    for c in chunks:
        if c.name == "INST":
            v = _inst(c)
            insts[v["id"]] = v
    names, parent, values, cls_of = {}, {}, {}, {}
    for v in insts.values():
        for ref in v["refs"]:
            cls_of[ref] = v["name"]
    for c in chunks:
        if c.name == "PROP":
            v = insts.get(_prop_class(c))
            if v is None:
                continue
            p = _prop(c)
            n = len(v["refs"])
            if p["name"] == "Name" and p["type"] == T_STRING:
                names.update(zip(v["refs"], _strings(p["values"], n)))
            elif p["name"] == "Value":
                if p["type"] == T_BOOL and len(p["values"]) == n:
                    values.update(zip(v["refs"], [b != 0 for b in p["values"]]))
                elif p["type"] == T_INT64:
                    values.update(zip(v["refs"], read_i64s(p["values"], n)))
                elif p["type"] == T_STRING:
                    values.update(zip(v["refs"], _strings(p["values"], n)))
                else:
                    values.update({r: ("type", p["type"]) for r in v["refs"]})
        elif c.name == "PRNT":
            kids, pars = _prnt(c)
            parent.update(zip(kids, pars))
    want = list(path)
    for ref in cls_of:
        at, here, seen = ref, [], set()
        while at is not None and at != -1 and at not in seen:
            seen.add(at)
            here.append(names.get(at))
            at = parent.get(at)
        if list(reversed(here)) != want:
            continue
        got = values.get(ref)
        if isinstance(got, tuple):
            return True, None, "%s's Value has binary type 0x%02X, which this reader does not decode" % (cls_of[ref], got[1])
        if ref not in values:
            return True, None, "%s at that path has no Value property" % cls_of[ref]
        return True, got, None
    return False, None, "no instance at %s in the file" % ".".join(want)


def script_sources(blob, text_classes=()):
    """[{path: [names from the root service], className, source}] for every Script, LocalScript and ModuleScript in a
    binary place, for the static save-layer check (rbx_save_audit). Decodes INST, every PROP named Name or Source, and
    PRNT; writes nothing. BinaryPlaceError when the file cannot be read (a zstd chunk with no zstd here, among others).
    With text_classes, each instance of those classes is answered too, as {path, className, text} with its Text."""
    _ncls, _ninst, chunks, _trailing = parse(blob)
    insts = {}
    for c in chunks:
        if c.name == "INST":
            v = _inst(c)
            insts[v["id"]] = v
    names, sources = {}, {}
    for c in chunks:
        if c.name != "PROP":
            continue
        v = insts.get(_prop_class(c))
        if v is None:
            continue
        p = _prop(c)
        if p["type"] != T_STRING or p["name"] not in ("Name", "Source", "Text"):
            continue
        if p["name"] == "Source" and v["name"] not in SCRIPT_CLASSES:
            continue
        if p["name"] == "Text" and v["name"] not in text_classes:
            continue
        vals = _strings(p["values"], len(v["refs"]))
        for ref, s in zip(v["refs"], vals):
            if p["name"] == "Name":
                names[ref] = s
            else:
                sources[ref] = (v["name"], "source" if p["name"] == "Source" else "text", s)
    prnt = [c for c in chunks if c.name == "PRNT"]
    parent = {}
    for c in prnt:
        kids, pars = _prnt(c)
        parent.update(zip(kids, pars))
    out = []
    for ref, (cls, field, src) in sources.items():
        path, at, seen = [], ref, set()
        while at is not None and at != -1 and at not in seen:
            seen.add(at)
            path.append(names.get(at, "?"))
            at = parent.get(at)
        out.append({"path": list(reversed(path)), "className": cls, field: src})
    out.sort(key=lambda s: s["path"])
    return out
