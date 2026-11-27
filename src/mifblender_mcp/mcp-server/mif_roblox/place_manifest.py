"""A canonical manifest of a saved Roblox place, and the diff of two (rbx_place_manifest). Offline: a file in, no Studio.

What it answers: did a rebuild make the same place, and what did a change touch. One row per instance - path, class,
the properties the file stores, attributes, tags, a script's sha256 - each row hashed, and the rows hashed in a stable
order into one sha256 for the place.

WHAT IS LEFT OUT, AND WHY (LEFT_OUT below, repeated in every answer). Measured 2026-11-22 on two Studio saves (.rbxlx)
of two builds of one place from the same build script, about 1,900 instances each:
    referent    the id a file gives each instance so Ref properties and parents can point at it. Of the 49 services
                both saves hold, 2 kept theirs; a binary save numbers them in file order. A Ref property is kept, as
                the PATH of the instance it points at.
    UniqueId    a per-instance id Studio assigns: no service kept its own between the two builds.
    HistoryId   an id stored beside it, all zero on most instances and an id of the same form on the rest.
    ScriptGuid  a per-script id Studio assigns: every script the two builds share (72) had a different one.
    MifBridgePort  the bridge's own port marker (ServerStorage.MifBridgePort): a copy of one place opened on another
                port differed in nothing else.
    file order  rows are sorted by path, so the order a file lists children in never reaches the hash.
Everything else a file stores is hashed, the edit camera's CFrame and Focus included: they are a person's view, not an
id, and a diff names them. No timestamp was found in either format. No default is filtered: a place file stores every serialized property and the
bridge's catalog holds no default values, so "properties that differ from default" cannot be answered offline - the
answer says so (defaults).

VALUES. A property is {"t": type, "v": value}. A string or binary value longer than INLINE bytes is given as its sha256
and length; a script's Source always is. A SharedString is resolved to its content first, so the key a file uses for it
never reaches the hash. Tags are split into names; attributes are decoded when they hold only strings, booleans and
numbers and are given as a sha256 otherwise (attributesAs says which).

TWO FORMATS. .rbxlx is read with ElementTree, .rbxl with rbxl.py's chunk reader and the decoders here. The two store
values differently (text against bits), so a manifest carries its format and a diff across formats is refused. A binary
property whose type this module does not decode, or whose bytes do not come out to exactly its values, is never
guessed: it is left out, named in the answer (undecoded), and the answer's complete is false. Measured on binary saves
of Studio 0.748: Content (type 0x22) is one source type per instance (interleaved i32: 0 none, 1 a URI), a count and
the URIs, then an object count and an external count (both 0 in every file read; anything else is undecoded);
PhysicalProperties is a flag byte per instance, 2 (0 in an older file) with nothing after it where the XML says
CustomPhysics false; AssetContentMap (0x23) is not decoded.
Standard library only. Raises ManifestError; the tool turns it into a refusal.
"""
import base64
import binascii
import hashlib
import json
import struct

from mif_roblox import rbxl as B

FORMAT = 1
KEY = "mifbridgePlaceManifest"
INLINE = 160
SCRIPT_CLASSES = ("Script", "LocalScript", "ModuleScript")
LEFT_OUT = [
    {"field": "referent", "why": "the id a file gives each instance; an XML save writes new ones on every save and a "
                                 "binary save numbers them in file order. A Ref property is kept as the path it points at"},
    {"field": "UniqueId", "why": "a per-instance id Studio assigns; a rebuild from the same inputs gets new ones"},
    {"field": "HistoryId", "why": "an id stored beside UniqueId, the same kind of field"},
    {"field": "ScriptGuid", "why": "a per-script id Studio assigns; a rebuild from the same inputs gets new ones"},
    {"field": "ServerStorage.MifBridgePort", "why": "the bridge's own port marker, which differs between two copies of "
                                                    "one place opened on two ports"},
    {"field": "file order", "why": "rows are sorted by path, so the order a file lists children in is not hashed"},
]
VOLATILE_PROPS = ("UniqueId", "HistoryId", "ScriptGuid")
MARKER_PATH = ["ServerStorage", B.MARKER_NAME]
DEFAULTS_NOTE = ("not filtered: a place file stores every serialized property and the bridge's catalog holds no default "
                 "values, so every stored property is in the manifest")


class ManifestError(ValueError):
    """The file is not a place this module can read, or holds nothing: the message says why."""


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


# ---------------------------------------------------------------- .rbxlx
def _xml_value(el):
    """A property element as plain data: its text when it has no children, else [[tag, value], ...] in order."""
    kids = list(el)
    if not kids:
        return (el.text or "").strip()
    return [[k.tag, _xml_value(k)] for k in kids]


def read_rbxlx(blob):
    """[{cls, parent (index or None), ref, props: {name: (kind, type, value)}}] from an XML place. kind is "str",
    "bytes", "ref" (value: the referent text, or None) or "val"."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(blob)
    except ET.ParseError as exc:
        raise ManifestError("not readable as XML: %s" % exc) from exc
    if root.tag != "roblox":
        raise ManifestError("not a Roblox place: the root element is <%s>, not <roblox>" % root.tag)
    shared = {}
    for holder in root.findall("SharedStrings"):
        for ss in holder.findall("SharedString"):
            try:
                shared[ss.get("md5")] = base64.b64decode("".join((ss.text or "").split()), validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ManifestError("SharedString %s is not base64: %s" % (ss.get("md5"), exc)) from exc
    out = []

    def walk(item, parent):
        props = {}
        holder = item.find("Properties")
        for p in (holder if holder is not None else ()):
            name, tag = p.get("name"), p.tag
            if name is None:
                raise ManifestError("a <%s> property with no name under a %s" % (tag, item.get("class")))
            if tag in ("string", "ProtectedString"):
                props[name] = ("str", tag, p.text or "")
            elif tag == "BinaryString":
                try:
                    props[name] = ("bytes", tag, base64.b64decode("".join((p.text or "").split()), validate=True))
                except (binascii.Error, ValueError) as exc:
                    raise ManifestError("BinaryString %s of a %s is not base64: %s" % (name, item.get("class"), exc)) from exc
            elif tag == "SharedString":
                key = (p.text or "").strip()
                if key not in shared:
                    raise ManifestError("SharedString %s of a %s names %r, which the file's SharedStrings do not hold"
                                        % (name, item.get("class"), key))
                props[name] = ("bytes", tag, shared[key])
            elif tag == "Ref":
                t = (p.text or "").strip()
                props[name] = ("ref", tag, None if t in ("", "null") else t)
            else:
                props[name] = ("val", tag, _xml_value(p))
        me = len(out)
        out.append({"cls": item.get("class") or "?", "parent": parent, "ref": item.get("referent"), "props": props})
        for child in item.findall("Item"):
            walk(child, me)
    for item in root.findall("Item"):
        walk(item, None)
    return out, []


# ---------------------------------------------------------------- .rbxl
_TYPE_NAMES = {0x01: "String", 0x02: "Bool", 0x03: "Int32", 0x04: "Float32", 0x05: "Float64", 0x06: "UDim", 0x07: "UDim2",
               0x08: "Ray", 0x09: "Faces", 0x0A: "Axes", 0x0B: "BrickColor", 0x0C: "Color3", 0x0D: "Vector2",
               0x0E: "Vector3", 0x10: "CFrame", 0x12: "Enum", 0x13: "Ref", 0x14: "Vector3int16", 0x15: "NumberSequence",
               0x16: "ColorSequence", 0x17: "NumberRange", 0x18: "Rect", 0x19: "PhysicalProperties", 0x1A: "Color3uint8",
               0x1B: "Int64", 0x1C: "SharedString", 0x1D: "Bytecode", 0x1E: "OptionalCFrame", 0x1F: "UniqueId",
               0x20: "Font", 0x21: "SecurityCapabilities", 0x22: "Content"}


class _Short(Exception):
    """A property's bytes are not what its type and count say: it is named undecoded, never guessed."""


def _need(d, size):
    if len(d) != size:
        raise _Short("%d bytes for %d expected" % (len(d), size))


def _u32s(d, n):
    return list(struct.unpack(">%dI" % n, B._deinterleave(d, n, 4)))


def _floats(d, n):
    """n Roblox floats: interleaved, big-endian, the sign bit rotated to the low end."""
    return [struct.unpack("<f", struct.pack("<I", (u >> 1) | ((u & 1) << 31)))[0] for u in _u32s(d, n)]


def _raw_floats(d, pos, k):
    if pos + 4 * k > len(d):
        raise _Short("floats run past the end")
    return list(struct.unpack_from("<%df" % k, d, pos)), pos + 4 * k


def _bstr(d, pos):
    if pos + 4 > len(d):
        raise _Short("a string length runs past the end")
    (m,) = struct.unpack_from("<I", d, pos)
    s = d[pos + 4:pos + 4 + m]
    if len(s) != m:
        raise _Short("a string runs past the end")
    return s, pos + 4 + m


def _text(b):
    """("str", text) for bytes that are UTF-8, else ("bytes", b): the binary format stores both as one type."""
    try:
        return "str", b.decode("utf-8")
    except UnicodeDecodeError:
        return "bytes", b


def _cframes(d, pos, n):
    rots = []
    for _ in range(n):
        if pos >= len(d):
            raise _Short("a CFrame rotation runs past the end")
        rid = d[pos]
        pos += 1
        if rid == 0:
            m, pos = _raw_floats(d, pos, 9)
            rots.append(m)
        else:
            rots.append({"rotationId": rid})
    if pos + 12 * n > len(d):
        raise _Short("CFrame positions run past the end")
    x, y, z = (_floats(d[pos + 4 * n * k:pos + 4 * n * (k + 1)], n) for k in range(3))
    return [{"position": [x[i], y[i], z[i]], "rotation": rots[i]} for i in range(n)], pos + 12 * n


def _decode(t, d, n, shared):
    """n values of binary type t from d as [(kind, value)], or _Short. Every branch accounts for every byte."""
    def val(rows):
        return [("val", r) for r in rows]

    def cols(k, f=_floats):
        _need(d, 4 * n * k)
        c = [f(d[4 * n * j:4 * n * (j + 1)], n) for j in range(k)]
        return [[c[j][i] for j in range(k)] for i in range(n)]

    if t in (0x01, 0x1D):
        out, pos = [], 0
        for _ in range(n):
            s, pos = _bstr(d, pos)
            out.append(_text(s))
        _need(d, pos)
        return out
    if t in (0x02, 0x09, 0x0A):
        _need(d, n)
        return val([bool(b) for b in d] if t == 0x02 else list(d))
    if t == 0x03:
        _need(d, 4 * n)
        return val(B.read_i32s(d, n))
    if t == 0x04:
        _need(d, 4 * n)
        return val(_floats(d, n))
    if t == 0x05:
        _need(d, 8 * n)
        return val(list(struct.unpack("<%dd" % n, d)))
    if t == 0x06:
        _need(d, 8 * n)
        return val([list(p) for p in zip(_floats(d[:4 * n], n), B.read_i32s(d[4 * n:], n))])
    if t == 0x07:
        _need(d, 16 * n)
        sx, sy = _floats(d[:4 * n], n), _floats(d[4 * n:8 * n], n)
        ox, oy = B.read_i32s(d[8 * n:12 * n], n), B.read_i32s(d[12 * n:], n)
        return val([[sx[i], ox[i], sy[i], oy[i]] for i in range(n)])
    if t == 0x08:
        _need(d, 24 * n)
        return val([list(struct.unpack_from("<6f", d, 24 * i)) for i in range(n)])
    if t in (0x0B, 0x12, 0x1C):
        _need(d, 4 * n)
        u = _u32s(d, n)
        if t != 0x1C:
            return val(u)
        if any(i >= len(shared) for i in u):
            raise _Short("a SharedString index past the file's %d shared strings" % len(shared))
        return [("bytes", shared[i]) for i in u]
    if t == 0x0C:
        return val(cols(3))
    if t == 0x0D:
        return val(cols(2))
    if t == 0x0E:
        return val(cols(3))
    if t == 0x10:
        rows, pos = _cframes(d, 0, n)
        _need(d, pos)
        return val(rows)
    if t == 0x13:
        _need(d, 4 * n)
        return [("ref", None if r == -1 else r) for r in B.read_i32s(d, n, delta=True)]
    if t == 0x14:
        _need(d, 6 * n)
        return val([list(struct.unpack_from("<3h", d, 6 * i)) for i in range(n)])
    if t in (0x15, 0x16):
        k = 3 if t == 0x15 else 5
        out, pos = [], 0
        for _ in range(n):
            if pos + 4 > len(d):
                raise _Short("a sequence count runs past the end")
            (m,) = struct.unpack_from("<I", d, pos)
            pos += 4
            keys = []
            for _j in range(m):
                f, pos = _raw_floats(d, pos, k)
                keys.append(f)
            out.append(keys)
        _need(d, pos)
        return val(out)
    if t == 0x17:
        _need(d, 8 * n)
        return val([list(struct.unpack_from("<2f", d, 8 * i)) for i in range(n)])
    if t == 0x18:
        return val(cols(4))
    if t == 0x19:
        out, pos = [], 0
        for _ in range(n):
            if pos >= len(d):
                raise _Short("a PhysicalProperties flag runs past the end")
            flag = d[pos]
            pos += 1
            if flag in (0, 2):                         # no custom values (measured against the XML form)
                out.append(None)
            elif flag == 1:
                f, pos = _raw_floats(d, pos, 5)
                out.append(f)
            else:
                raise _Short("PhysicalProperties flag %d is not one this reader knows" % flag)
        _need(d, pos)
        return val(out)
    if t == 0x1A:
        _need(d, 3 * n)
        return val([[d[i], d[n + i], d[2 * n + i]] for i in range(n)])
    if t == 0x1B:
        _need(d, 8 * n)
        return val(B.read_i64s(d, n))
    if t == 0x1E:
        if not d or d[0] != 0x10:
            raise _Short("an optional CFrame that does not start with the CFrame type")
        rows, pos = _cframes(d, 1, n)
        if d[pos:pos + 1] != b"\x02":
            raise _Short("an optional CFrame with no presence flags")
        flags = d[pos + 1:]
        _need(flags, n)
        return val([rows[i] if flags[i] else None for i in range(n)])
    if t == 0x1F:
        _need(d, 16 * n)
        raw = B._deinterleave(d, n, 16)
        return val([raw[16 * i:16 * i + 16].hex() for i in range(n)])
    if t == 0x20:
        out, pos = [], 0
        for _ in range(n):
            fam, pos = _bstr(d, pos)
            if pos + 3 > len(d):
                raise _Short("a Font weight runs past the end")
            weight, style = struct.unpack_from("<HB", d, pos)
            cached, pos = _bstr(d, pos + 3)
            out.append({"family": fam.decode("utf-8", "replace"), "weight": weight, "style": style,
                        "cachedFaceId": cached.decode("utf-8", "replace")})
        _need(d, pos)
        return val(out)
    if t == 0x21:
        _need(d, 8 * n)
        return val(list(struct.unpack(">%dQ" % n, B._deinterleave(d, n, 8))))
    if t == 0x22:
        if len(d) < 4 * n + 4:
            raise _Short("a Content property shorter than its source types")
        kinds = B.read_i32s(d[:4 * n], n)
        (count,) = struct.unpack_from("<I", d, 4 * n)
        pos, uris = 4 * n + 4, []
        for _ in range(count):
            s, pos = _bstr(d, pos)
            uris.append(s.decode("utf-8", "replace"))
        if pos + 8 > len(d):
            raise _Short("a Content property with no object and external counts")
        objects, external = struct.unpack_from("<II", d, pos)
        if objects or external or any(k not in (0, 1) for k in kinds) or sum(kinds) != count:
            raise _Short("a Content property that holds something other than URIs (source types %s, %d object(s), "
                         "%d external)" % (sorted(set(kinds)), objects, external))
        _need(d, pos + 8)
        left = iter(uris)
        return val([None if k == 0 else {"uri": next(left)} for k in kinds])
    raise _Short("type 0x%02X is not one this reader decodes" % t)


def read_rbxl(blob):
    """The same rows as read_rbxlx from a binary place, and [{className, property, type, why}] for each property left
    undecoded."""
    try:
        _ncls, _ninst, chunks, _trailing = B.parse(blob)
        classes, shared = {}, []
        for c in chunks:
            if c.name == "INST":
                v = B._inst(c)
                classes[v["id"]] = v
            elif c.name == "SSTR":
                shared = [s for _h, s in B._sstr(c)]
        index, out = {}, []
        for v in classes.values():
            for ref in v["refs"]:
                index[ref] = len(out)
                out.append({"cls": v["name"], "parent": None, "ref": ref, "props": {}})
        undecoded = []
        for c in chunks:
            if c.name == "PRNT":
                kids, pars = B._prnt(c)
                for k, p in zip(kids, pars):
                    if k in index:
                        out[index[k]]["parent"] = index.get(p) if p != -1 else None
            if c.name != "PROP":
                continue
            p = B._prop(c)
            v = classes.get(p["id"])
            if v is None:
                raise ManifestError("a PROP chunk for class id %d, which the file has no INST for" % p["id"])
            n, t = len(v["refs"]), p["type"]
            tname = _TYPE_NAMES.get(t, "0x%02X" % t)
            try:
                vals = _decode(t, p["values"], n, shared)
            except (_Short, struct.error) as exc:
                undecoded.append({"className": v["name"], "property": p["name"], "type": tname, "why": str(exc)})
                continue
            for ref, (kind, value) in zip(v["refs"], vals):
                out[index[ref]]["props"][p["name"]] = (kind, tname, value)
    except (B.BinaryPlaceError, struct.error, IndexError) as exc:
        raise ManifestError("not readable as a binary place: %s" % exc) from exc
    for row in out:                                    # a binary Ref is a referent number: point it at its row
        for name, (kind, tname, value) in list(row["props"].items()):
            if kind == "ref" and value is not None:
                row["props"][name] = (kind, tname, value if value in index else None)
    return out, undecoded


# ---------------------------------------------------------------- rows
def _attributes(b):
    """({name: value}, "decoded") when the blob holds only strings (type 0x02), booleans (0x03) and doubles (0x06) and
    comes out to its last byte; else ({"sha256", "bytes"}, "sha256")."""
    def fallback():
        return {"sha256": _sha(b), "bytes": len(b)}, "sha256"
    if not b:
        return {}, "decoded"
    try:
        (n,) = struct.unpack_from("<I", b, 0)
        pos, out = 4, {}
        for _ in range(n):
            name, pos = _bstr(b, pos)
            t = b[pos]
            pos += 1
            if t == 0x02:
                s, pos = _bstr(b, pos)
                value = _small("str", s.decode("utf-8"))
            elif t == 0x03:
                value, pos = bool(b[pos]), pos + 1
            elif t == 0x06:
                (value,), pos = struct.unpack_from("<d", b, pos), pos + 8
            else:
                return fallback()
            out[name.decode("utf-8")] = value
        if pos != len(b) or len(out) != n:
            return fallback()
        return out, "decoded"
    except (_Short, struct.error, IndexError, UnicodeDecodeError):
        return fallback()


def _small(kind, value):
    if kind == "str":
        b = value.encode("utf-8", "surrogatepass")
        return value if len(b) <= INLINE else {"sha256": _sha(b), "bytes": len(b)}
    if kind == "bytes":
        return "" if not value else {"sha256": _sha(value), "bytes": len(value)}
    return value


def _raw(kind, value):
    return value.encode("utf-8", "surrogatepass") if kind == "str" else value


def rows(instances):
    """The manifest rows of read_rbxlx's or read_rbxl's instances, sorted: {path, className, sha256, properties,
    attributes, attributesAs, tags, script?}."""
    names = []
    for r in instances:
        nm = r["props"].get("Name")
        names.append(nm[2] if nm and nm[0] == "str" else r["cls"])
    paths = [None] * len(instances)

    def path_of(i):
        chain, at, seen = [], i, set()
        while at is not None and paths[at] is None and at not in seen:
            seen.add(at)
            chain.append(at)
            at = instances[at]["parent"]
        base = list(paths[at]) if at is not None and paths[at] is not None else []
        for j in reversed(chain):
            base = base + [names[j]]
            paths[j] = base
        return paths[i]
    for i in range(len(instances)):
        path_of(i)
    by_ref = {}
    for i, r in enumerate(instances):
        if r["ref"] is not None:
            by_ref[r["ref"]] = i
    out = []
    for i, r in enumerate(instances):
        props, attrs, attrs_as, tags, script = {}, {}, "decoded", [], None
        for name, (kind, tname, value) in r["props"].items():
            if name == "Name" or name in VOLATILE_PROPS:
                continue
            if name == "AttributesSerialize" and kind in ("str", "bytes"):
                attrs, attrs_as = _attributes(_raw(kind, value))
            elif name == "Tags" and kind in ("str", "bytes"):
                tags = sorted(t.decode("utf-8", "replace") for t in _raw(kind, value).split(b"\0") if t)
            elif name == "Source" and r["cls"] in SCRIPT_CLASSES and kind in ("str", "bytes"):
                src = _raw(kind, value)
                script = {"sha256": _sha(src), "bytes": len(src), "lines": src.count(b"\n") + (1 if src else 0)}
            elif kind == "ref":
                j = by_ref.get(value) if value is not None else None
                props[name] = {"t": tname, "v": None if j is None else {"path": paths[j]}}
            else:
                props[name] = {"t": tname, "v": _small(kind, value)}
        row = {"path": paths[i], "className": r["cls"], "properties": props, "attributes": attrs, "tags": tags}
        if attrs_as != "decoded":
            row["attributesAs"] = attrs_as
        if script is not None:
            row["script"] = script
        if paths[i] == MARKER_PATH and r["cls"] == "IntValue":
            continue
        row["sha256"] = row_sha(row)
        out.append(row)
    out.sort(key=_order)
    return out


def _order(row):
    return (row["path"], row["className"], row["sha256"])


def row_sha(row):
    return _sha(_canon([row["className"], row["properties"], row["attributes"], row["tags"], row.get("script")]))


def place_sha(rows_):
    return _sha(_canon([[r["path"], r["className"], r["sha256"]] for r in sorted(rows_, key=_order)]))


def read(blob, ext):
    """(format, rows, undecoded) of a place file's bytes. ManifestError when it is not a place or holds no instance."""
    if not blob:
        raise ManifestError("the file is empty (0 bytes)")
    binary = blob.startswith(b"<roblox!")
    if not binary and ext == ".rbxl":
        raise ManifestError("a .rbxl with no <roblox! signature: not a binary place")
    instances, undecoded = read_rbxl(blob) if binary else read_rbxlx(blob)
    if not instances:
        raise ManifestError("the file holds no instance: nothing was read, so nothing can be hashed")
    return ("rbxl" if binary else "rbxlx"), rows(instances), undecoded


def manifest(fmt, rows_, undecoded, source=None, services=None):
    """The manifest document: what out= writes and against= reads back."""
    doc = {KEY: FORMAT, "format": fmt, "sha256": place_sha(rows_), "instances": rows_, "leftOut": LEFT_OUT,
           "undecoded": undecoded, "complete": not undecoded, "defaults": DEFAULTS_NOTE}
    if source:
        doc["source"] = source
    if services is not None:
        doc["services"] = sorted(services)
    return doc


def load_manifest(text):
    """(format, rows, undecoded, services) of a manifest JSON written by out=; its sha256 and every row's are
    recomputed, so an edited manifest is refused."""
    try:
        doc = json.loads(text)
    except ValueError as exc:
        raise ManifestError("not JSON: %s" % exc) from exc
    if not isinstance(doc, dict) or doc.get(KEY) != FORMAT:
        raise ManifestError("not a place manifest written by rbx_place_manifest (no %s: %d)" % (KEY, FORMAT))
    rows_ = doc.get("instances")
    if not isinstance(rows_, list) or not rows_:
        raise ManifestError("the manifest holds no instance")
    for r in rows_:
        if (not isinstance(r, dict) or not isinstance(r.get("path"), list) or not isinstance(r.get("properties"), dict)
                or "className" not in r or "attributes" not in r or "tags" not in r):
            raise ManifestError("a manifest row is not {path, className, properties, attributes, tags}")
        if row_sha(r) != r.get("sha256"):
            raise ManifestError("the row for %s does not hash to its sha256: the manifest was changed after it was "
                                "written" % ".".join(map(str, r["path"])))
    if place_sha(rows_) != doc.get("sha256"):
        raise ManifestError("the manifest's rows do not hash to its sha256: it was changed after it was written")
    return doc.get("format"), rows_, doc.get("undecoded") or [], doc.get("services")


def only_services(rows_, services):
    """(rows under the named top-level instances, names not in the file)."""
    have = {r["path"][0] for r in rows_}
    missing = sorted(set(services) - have)
    return [r for r in rows_ if r["path"][0] in services], missing


def per_service(rows_):
    by = {}
    for r in rows_:
        by.setdefault(r["path"][0], []).append(r)
    return {name: {"instances": len(rs), "sha256": place_sha(rs)} for name, rs in sorted(by.items())}


# ---------------------------------------------------------------- diff
_WHOLE = {"ClassName": "className", "Attributes": "attributes", "Tags": "tags", "Source": "script"}


def _changed_names(a, b):
    names = sorted(k for k in set(a["properties"]) | set(b["properties"])
                   if a["properties"].get(k) != b["properties"].get(k))
    if a["className"] != b["className"]:
        names.append("ClassName")
    if a["attributes"] != b["attributes"]:
        names.append("Attributes")
    if a["tags"] != b["tags"]:
        names.append("Tags")
    if a.get("script") != b.get("script"):
        names.append("Source")
    return names


def diff(old, new):
    """{added, removed, changed} between two manifests' rows. Instances are matched by path; among siblings of one
    path, equal rows are matched first, then the rest pair off in order as changed, and what is left is added or
    removed. An added or removed subtree is one row with its descendants counted."""
    by_old, by_new = {}, {}
    for rows_, by in ((old, by_old), (new, by_new)):
        for r in rows_:
            by.setdefault(tuple(r["path"]), []).append(r)
    added, removed, changed = [], [], []
    for key in sorted(set(by_old) | set(by_new)):
        a, b = list(by_old.get(key, [])), list(by_new.get(key, []))
        counts = {}
        for r in b:
            counts[r["sha256"]] = counts.get(r["sha256"], 0) + 1
        rest_a = []
        for r in a:
            if counts.get(r["sha256"]):
                counts[r["sha256"]] -= 1
            else:
                rest_a.append(r)
        rest_b = []
        for r in b:                                    # what the equal rows above did not use up
            if counts.get(r["sha256"]):
                counts[r["sha256"]] -= 1
                rest_b.append(r)
        for x, y in zip(rest_a, rest_b):
            names = _changed_names(x, y)
            row = {"path": list(key), "className": y["className"], "properties": names, "was": {}, "now": {}}
            for nm in names:
                if nm in x["properties"] or nm in y["properties"]:
                    row["was"][nm] = x["properties"].get(nm, {}).get("v")
                    row["now"][nm] = y["properties"].get(nm, {}).get("v")
                elif nm in _WHOLE:
                    row["was"][nm], row["now"][nm] = x.get(_WHOLE[nm]), y.get(_WHOLE[nm])
            changed.append(row)
        removed += [{"path": list(key), "className": r["className"]} for r in rest_a[len(rest_b):]]
        added += [{"path": list(key), "className": r["className"]} for r in rest_b[len(rest_a):]]

    def fold(rows_, other):
        gone = {tuple(r["path"]) for r in rows_ if tuple(r["path"]) not in other}
        tops = []
        for r in rows_:
            p = tuple(r["path"])
            if any(p[:k] in gone for k in range(1, len(p))):
                continue
            tops.append(r)
        for r in tops:
            p = tuple(r["path"])
            if p in gone:
                r["descendants"] = sum(1 for q in rows_ if len(q["path"]) > len(p) and tuple(q["path"][:len(p)]) == p)
        return tops
    return {"added": fold(added, by_old), "removed": fold(removed, by_new), "changed": changed}
