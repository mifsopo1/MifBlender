"""Terrain and materials, lighting and effects (section 5.4, batch 7): composites on the
MCP side over the generic plugin ops (call, get_props, set_props, create, tree), every write read back.

Loaded by tools.py with one star import, so these rbx_* tools join TOOLS and PACK like the rest. Nothing here needs a
plugin op of its own; what would help is filed as a plugin request.

THE TERRAIN WRITE GATE (decision 6, still open). Terrain is one object, Workspace.Terrain, so "under Workspace.MifScratch"
cannot hold it. Two answers are built and one switch picks:
- "full-only" (the default while decision 6 is open): a terrain write in scratch mode is refused before sending, unless
  Studio has MifBridge's own MifScratch.rbxlx open, where the whole place is scratch (the plugin's own rule).
- "box" (MIF_ROBLOX_TERRAIN_SCRATCH=box): in scratch mode a write whose cells all lie inside SCRATCH_BOX is sent, and one
  that leaves it is refused scratch-path naming the box. The plugin must hold the same box as a constant for its own gate
  to let it through; it is never sent or set over the wire, so nothing here can widen what the plugin allows.
Read mode refuses every terrain write either way. Full mode allows any region inside Terrain.MaxExtents.

VOXELS. Terrain:ReadVoxels and WriteVoxels at resolution 4 (one cell is 4 studs), chunked at CHUNK cells per axis: a
32-cell cube came back in 1.3 s and a 64-cell cube not within 135 s. Materials cross as EnumItems inside nested Arrays,
which values.encode refuses for a Variant (an EnumItem needs a typed member), so this module builds those wire values
itself; the plugin's decode takes an EnumItem anywhere. A grid is indexed [x][y][z] from the region's minimum cell.

NEVER RAISE: every failure is {"ok": False, ...}.
"""
import math
import os
import random
import struct
import time
import uuid
from typing import Any

from mif_roblox import tools as T  # noqa: E402 - tools.py imports this module last; T is complete by then
from mif_roblox import values as V  # noqa: E402

TERRAIN = ["Workspace", "Terrain"]
VOXEL = 4                                   # studs per cell; ReadVoxels and WriteVoxels take only resolution 4
CHUNK = int(os.environ.get("MIF_ROBLOX_VOXEL_CHUNK", "32"))   # cells per axis per call: 32 measured on 0.748 (RB301)
MAX_CELLS = 1 << 20                         # one tool call writes at most this many cells
MAX_RETURN_CELLS = 1 << 16                  # rbx_terrain_voxels get returns full grids up to this; above, a summary
SCRATCH_PLACE = "MifScratch.rbxlx"          # the plugin's own: in that place the whole of it is scratch
# Decision 6, the declared box: far from the origin so a person's build is never inside it. Studs, min inclusive, max
# exclusive, every face on the 4-stud grid. The plugin must hold the same numbers (PLUGIN_REQUESTS.md, lane 3).
SCRATCH_BOX = {"min": [8192, -256, 8192], "max": [8704, 256, 8704]}
COMPACT_CHUNK = 64                          # cells per axis per compact call (plugin 0.15.0: one byte a cell)
_COMPACT = {"ok": None}                     # None until measured: does this plugin have voxels_read and voxels_write
OCC_TOL = 0.01                              # Studio keeps solid occupancy in 1/256 steps, liquid in 1/255 (RB302)
THIN_DEFAULT = 0.5                          # the holes scan's thin: a top cell under this is judged (RB1046, RB1754)
# An Air fill and a thin cell it only partly reaches (RB1922, RB1046's replay at four turns: 0.298 cells beside a cut):
# emptied with its centre 0.112-0.823 studs outside the shape, kept as it was at 0.851 and beyond (by distance, not by
# the share of the cube inside: 0.234 inside was emptied at one turn and kept at another).
AIR_REACH_THIN, AIR_REACH_EMPTIES, AIR_REACH_KEEPS = 0.298, 0.823, 0.851
SLOT_MIN, SLOT_MAX = 2, 63                  # measured on 0.748: 0 Air and 1 Water refuse a set, 64 is out of range
# Smooth terrain draws a flat column's surface at 2 + 4 x (the column's solid cells summed) studs above its bottom cell:
# half a cell above the stack, at every top-cell occupancy from 0.05 to 1.0 (RB1150, RB1006). A heightmap therefore
# writes each column h - SURFACE_LIFT tall, and a column below the lift is drawn at the lift, the lowest it can be.
SURFACE_LIFT = 2.0
# WHICH SURFACE A TERRAIN WRITE PUT WHERE IT WAS TOLD: every terrain write's reply says it, so a caller reads
# it instead of guessing from the bridge's version. surface "drawn": the surface Studio draws lands on the height or top
# given (rbx_terrain_heightmap; rbx_terrain_fill surface true); "solid": the solid cells end where given and Studio draws
# the surface SURFACE_LIFT above them (rbx_terrain_fill without surface; rbx_terrain_voxels set). surfaceConvention is the
# number of this rule: 1 was the heightmap before RB1150 (solid to h, so drawn 2 high), 2 is this one. A caller that
# compensated for 1 with its own +2 drops it when it reads 2.
SURFACE_CONVENTION = 2
ROUGH_RELIEF = 0.1                          # nine rays across a top that spread more than this: a rough material (RB1158)
HEIGHT_TOL = 1.0                            # a sampled height passes within 1 stud of its 3x3 neighbourhood's range
MIN_OCC = 1 / 255                           # the least occupancy a cell keeps: a column's thinnest drawable top
CHANNELS = ["SolidMaterial", "SolidOccupancy", "LiquidOccupancy"]   # ReadVoxelChannels ids, Roblox's Terrain page
_CACHE = {}


def terrain_policy():
    p = os.environ.get("MIF_ROBLOX_TERRAIN_SCRATCH", "full-only")
    return p if p in ("full-only", "box") else "full-only"


# ---------------------------------------------------------------- wire helpers
def _w3(v, t="Vector3"):
    return {"$t": t, "v": [float(x) if t == "Vector3" else int(x) for x in v]}


def _wregion(c0, c1):
    """Region3 in studs from a cell box [c0, c1)."""
    return {"$t": "Region3", "v": [c * VOXEL for c in c0] + [c * VOXEL for c in c1]}


def _wmat(name):
    return {"$t": "EnumItem", "enum": "Material", "v": name}


def _wcf(pos, rot=None):
    return {"$t": "CFrame", "v": list(pos) + list(rot or [1, 0, 0, 0, 1, 0, 0, 0, 1])}


def _wgrid(grid, leaf):
    return {"$t": "Array", "v": [{"$t": "Array", "v": [{"$t": "Array", "v": [leaf(c) for c in col]} for col in plane]}
                                 for plane in grid]}


class _Cut(Exception):
    pass


def _unwire(w):
    """A plugin return to plain Python: EnumItems to names, nested Arrays to lists. ReadVoxels' grids carry a Size field,
    so the plugin encodes them as a Dictionary keyed "1".."n" plus "Size": that becomes a list again."""
    if isinstance(w, list):
        return [_unwire(x) for x in w]
    if not isinstance(w, dict) or "$t" not in w:
        return w
    t = w["$t"]
    if t == "EnumItem":
        return w.get("v")
    if t == "nil":
        return None
    if t in ("Array", "Instances"):
        if w.get("truncated"):
            raise _Cut("a list of %s was cut at 500 by the plugin" % w.get("total"))
        return [_unwire(x) for x in w.get("v") or []]
    if t == "Dictionary":
        v = w.get("v")
        if isinstance(v, list):
            return [_unwire(x) for x in v]
        v = v or {}
        nums = [k for k in v if k.isdigit()]
        if nums and all(k.isdigit() or k == "Size" for k in v):
            return [_unwire(v[str(i)]) for i in range(1, len(nums) + 1)]
        return {k: _unwire(x) for k, x in v.items()}
    return V.decode(w)


def _dims(grid):
    try:
        return [len(grid), len(grid[0]), len(grid[0][0])]
    except (TypeError, IndexError, KeyError):
        return None


def _blank(n, value):
    return [[[value] * n[2] for _ in range(n[1])] for _ in range(n[0])]


def _catalog_check(cls, method):
    cat, err = T._catalog()
    if err:
        return err
    return cat.read_refusal(cls, method, kind="F")


def _call(method, args, read_only, target=None, cls="Terrain", timeout=60):
    """One generic plugin call with wire args this module built. Catalog-checked; never raises."""
    bad = _catalog_check(cls, method)
    if bad:
        return bad
    return T._rbx("call", timeout=timeout + 15, target=target or TERRAIN, method=method, args=args, argc=len(args),
                  readOnly=read_only, yields=False, callTimeout=timeout)


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _vec(x, n=3):
    return isinstance(x, (list, tuple)) and len(x) == n and all(_num(v) for v in x)


def _region_cells(region):
    """{min, max} in studs -> (c0, c1) cells covering it (aligned outward to the 4-stud grid), or (None, None, why)."""
    if not isinstance(region, dict) or not _vec(region.get("min")) or not _vec(region.get("max")):
        return None, None, "region is {min: [x, y, z], max: [x, y, z]} in studs"
    lo, hi = region["min"], region["max"]
    if any(h <= low for low, h in zip(lo, hi)):
        return None, None, "region max must be above min on every axis"
    c0 = [int(math.floor(x / VOXEL)) for x in lo]
    c1 = [int(math.ceil(x / VOXEL)) for x in hi]
    return c0, c1, None


def _studs(c0, c1):
    return {"min": [c * VOXEL for c in c0], "max": [c * VOXEL for c in c1]}


def _count(c0, c1):
    return (c1[0] - c0[0]) * (c1[1] - c0[1]) * (c1[2] - c0[2])


# ---------------------------------------------------------------- the gate
def _extents():
    """Terrain.MaxExtents as a cell box, cached per process (it is a constant of the running Studio)."""
    if "extents" in _CACHE:
        return _CACHE["extents"], None
    r = T._rbx("get_props", target=TERRAIN, props=["MaxExtents"])
    if not r.get("ok"):
        return None, r
    w = T._obj(r.get("values")).get("MaxExtents")
    if not isinstance(w, dict) or not isinstance(w.get("v"), list) or len(w["v"]) != 6:
        return None, {"ok": False, "error": "Terrain.MaxExtents read back as %r" % (w,)}
    _CACHE["extents"] = (w["v"][:3], w["v"][3:])
    return _CACHE["extents"], None


def _inside(c0, c1, b0, b1):
    return all(b0[i] <= c0[i] and c1[i] <= b1[i] for i in range(3))


# How long the gate's mode ping waits. Found 2026-11-23: 15 s ran out right after a large rbx_terrain_smooth (about
# 140x24x140 studs, two iterations) while Studio was still busy, and the next fill failed on the ping, twice in two fresh
# Studios. The ping is a read, so waiting longer costs nothing when Studio answers at once.
GATE_PING_TIMEOUT = 60


def _terrain_gate(c0=None, c1=None, what="a terrain write", timeout=GATE_PING_TIMEOUT):
    """None when the write may be sent, else the refusal; nothing is sent on a refusal. c0/c1 None: a write with no
    region (material colors, water look), which a box cannot hold."""
    if c0 is not None:
        ext, err = _extents()
        if err:
            return dict(err, ok=False)
        if ext is not None:
            lo, hi = ext
            # MaxExtents is a Region3int16 of cells, its max inclusive.
            if not _inside(c0, c1, lo, [h + 1 for h in hi]):
                return {"ok": False, "ran": False, "refusedRule": "out-of-extents",
                        "maxExtents": _studs(lo, [h + 1 for h in hi]), "region": _studs(c0, c1),
                        "error": "%s reaches outside Terrain.MaxExtents" % what}
    ping = T._rbx("ping", nonce=uuid.uuid4().hex, timeout=timeout)
    if not ping.get("ok"):
        # The ping is the gate's read of the mode, sent before the write: whatever it says about itself, the write was
        # never sent (the daemon's "it may have run" is about the ping), so the answer says so.
        return dict(ping, ok=False, ran=False, sent=False, gate="mode-ping",
                    error="%s was not sent: the mode check before it got no answer (%s). Studio may still be busy with "
                          "an earlier call; send it again, or raise timeout" % (what, ping.get("error") or "no answer"))
    mode, place = ping.get("mode"), ping.get("placeName")
    unlock = ("Raise the mode in the MifBridge panel (Plugins > MifBridge), or rbx_set_mode on a place rbx_launch made "
              "where the person's own policy file allows it")
    if mode == "read":
        return {"ok": False, "ran": False, "refusedRule": "read-mode", "mode": mode, "unlock": unlock,
                "error": "%s is a write and the panel is in read mode: nothing was sent" % what}
    if mode == "scratch" and place != SCRATCH_PLACE:
        # The plugin's own panel toggle (0.16.0, decision 6's "box") is the person's answer; the env var still works for
        # an older plugin. Either way the plugin's gate holds the same box and is the real check.
        policy = "box" if ping.get("terrainBox") is True else terrain_policy()
        box0 = [x // VOXEL for x in SCRATCH_BOX["min"]]
        box1 = [x // VOXEL for x in SCRATCH_BOX["max"]]
        if policy == "box" and c0 is not None and _inside(c0, c1, box0, box1):
            return None
        why = ("Terrain is one object, so scratch mode cannot hold it; decision 6 is open and terrain writes need Full "
               "mode (or MifBridge's own %s place)" % SCRATCH_PLACE)
        if policy == "box":
            why = ("in scratch mode a terrain write must lie inside the declared box %s" % SCRATCH_BOX
                   if c0 is not None else "a write with no region (colors, water, slots) cannot be held to the "
                   "scratch box; it needs Full mode")
        return {"ok": False, "ran": False, "refusedRule": "scratch-path", "mode": mode, "policy": policy,
                "scratchBox": SCRATCH_BOX if policy == "box" else None, "unlock": unlock,
                "error": "%s: %s. Nothing was sent" % (what, why)}
    return None


# ---------------------------------------------------------------- voxel blocks, chunked
def _chunks(c0, c1, step):
    for x in range(c0[0], c1[0], step):
        for y in range(c0[1], c1[1], step):
            for z in range(c0[2], c1[2], step):
                yield [x, y, z], [min(x + step, c1[0]), min(y + step, c1[1]), min(z + step, c1[2])]


def _read_chunk(a, b, water):
    n = [b[i] - a[i] for i in range(3)]
    if water:
        r = _call("ReadVoxelChannels", [_wregion(a, b), VOXEL, {"$t": "Array", "v": list(CHANNELS)}], True)
        if not r.get("ok"):
            return None, r
        try:
            got = _unwire((r.get("returns") or [None])[0])
        except _Cut as exc:
            return None, {"ok": False, "error": str(exc)}
        if not isinstance(got, dict) or not all(k in got for k in CHANNELS):
            return None, {"ok": False, "error": "ReadVoxelChannels answered channels %s, asked %s" % (
                sorted(got) if isinstance(got, dict) else type(got).__name__, CHANNELS), "readBack": got}
        grids = [got["SolidMaterial"], got["SolidOccupancy"], got["LiquidOccupancy"]]
    else:
        r = _call("ReadVoxels", [_wregion(a, b), VOXEL], True)
        if not r.get("ok"):
            return None, r
        rets = r.get("returns") or []
        if len(rets) < 2:
            return None, {"ok": False, "error": "ReadVoxels returned %d values, expected materials and occupancy" % len(rets)}
        try:
            grids = [_unwire(rets[0]), _unwire(rets[1]), None]
        except _Cut as exc:
            return None, {"ok": False, "error": str(exc)}
    for g in grids:
        if g is not None and _dims(g) != n:
            return None, {"ok": False, "error": "Studio returned a %s grid for a %s cell region" % (_dims(g), n)}
    return grids, None


def _b64(raw):
    import base64
    return base64.b64encode(bytes(raw)).decode("ascii")


def _compact_read(a, b):
    """Plugin 0.15.0's voxels_read: a palette of Material names and one byte a cell (occupancy x 255), x, y, z with z
    fastest. (grids, None), (None, failure), or (None, None) when the plugin has no such op."""
    import base64
    if _COMPACT["ok"] is False:
        return None, None
    r = T._rbx("voxels_read", min=[c * VOXEL for c in a], max=[c * VOXEL for c in b], resolution=VOXEL, timeout=120)
    if r.get("refusedRule") == "unknown-op":
        _COMPACT["ok"] = False
        return None, None
    if not r.get("ok"):
        return None, r
    _COMPACT["ok"] = True
    n = [b[i] - a[i] for i in range(3)]
    if r.get("size") != n:
        return None, {"ok": False, "error": "voxels_read answered %s cells for a %s region" % (r.get("size"), n)}
    mats, occs = base64.b64decode(r["materials"]), base64.b64decode(r["occupancy"])
    pal = r.get("palette") or []
    mat = [[[None] * n[2] for _ in range(n[1])] for _ in range(n[0])]
    occ = [[[0.0] * n[2] for _ in range(n[1])] for _ in range(n[0])]
    k = 0
    for x in range(n[0]):
        for y in range(n[1]):
            mx, ox = mat[x][y], occ[x][y]
            for z in range(n[2]):
                mx[z] = pal[mats[k]]
                ox[z] = occs[k] / 255.0
                k += 1
    return [mat, occ, None], None


def _compact_write(a, mat, occ):
    """voxels_write of one chunk; the plugin reads it back cell by cell itself. None when the plugin has no such op."""
    if _COMPACT["ok"] is False:
        return None
    n = _dims(mat)
    pal, index, mb, ob = [], {}, bytearray(), bytearray()
    for x in range(n[0]):
        for y in range(n[1]):
            for z in range(n[2]):
                m = mat[x][y][z]
                if m not in index:
                    index[m] = len(pal)
                    pal.append(m)
                mb.append(index[m] % 256)
                ob.append(max(0, min(255, int(round(occ[x][y][z] * 255)))))
    if len(pal) > 256:
        return {"ok": False, "error": "more than 256 materials in one chunk"}
    b = [a[i] + n[i] for i in range(3)]
    r = T._rbx("voxels_write", min=[c * VOXEL for c in a], max=[c * VOXEL for c in b], resolution=VOXEL, palette=pal,
               materials=_b64(mb), occupancy=_b64(ob), timeout=120)
    if r.get("refusedRule") == "unknown-op":
        _COMPACT["ok"] = False
        return None
    if r.get("ok"):
        _COMPACT["ok"] = True
    return r


def _read_block(c0, c1, water=False):
    """{"mat", "occ", "liq"?} grids for the cell box [c0, c1), read in chunks. (block, None) or (None, failure)."""
    n = [c1[i] - c0[i] for i in range(3)]
    mat, occ = _blank(n, "Air"), _blank(n, 0.0)
    liq = _blank(n, 0.0) if water else None
    calls = 0
    if not water and _COMPACT["ok"] is not False:
        if _COMPACT["ok"] is None:
            _compact_read(c0, [min(c0[i] + 1, c1[i]) for i in range(3)])   # measures once whether the plugin has it
        if _COMPACT["ok"]:
            for a, b in _chunks(c0, c1, COMPACT_CHUNK):
                grids, err = _compact_read(a, b)
                calls += 1
                if err:
                    return None, dict(err, ok=False, chunk=_studs(a, b))
                for i in range(b[0] - a[0]):
                    for j in range(b[1] - a[1]):
                        X, Y = a[0] - c0[0] + i, a[1] - c0[1] + j
                        mat[X][Y][a[2] - c0[2]:b[2] - c0[2]] = grids[0][i][j]
                        occ[X][Y][a[2] - c0[2]:b[2] - c0[2]] = grids[1][i][j]
            return {"mat": mat, "occ": occ, "liq": None, "calls": calls, "route": "voxels_read"}, None
    for a, b in _chunks(c0, c1, CHUNK):
        grids, err = _read_chunk(a, b, water)
        calls += 1
        if err:
            return None, dict(err, ok=False, chunk=_studs(a, b))
        for i in range(b[0] - a[0]):
            for j in range(b[1] - a[1]):
                for k in range(b[2] - a[2]):
                    X, Y, Z = a[0] - c0[0] + i, a[1] - c0[1] + j, a[2] - c0[2] + k
                    mat[X][Y][Z] = grids[0][i][j][k]
                    occ[X][Y][Z] = grids[1][i][j][k]
                    if water:
                        liq[X][Y][Z] = grids[2][i][j][k]
    return {"mat": mat, "occ": occ, "liq": liq, "calls": calls}, None


def _sub(g, a, b, c0):
    return [[[g[x - c0[0]][y - c0[1]][z - c0[2]] for z in range(a[2], b[2])] for y in range(a[1], b[1])]
            for x in range(a[0], b[0])]


@T.ends_group_on_raise
def _write_block(c0, mat, occ, liq=None, label="terrain write"):
    """Write grids at cell c0, chunked, as one undo step when no other group is open. Gate first (the caller's)."""
    n = _dims(mat)
    c1 = [c0[i] + n[i] for i in range(3)]
    grouped = T.begin_group(label).get("ok") is True
    calls, t0 = 0, time.monotonic()
    if liq is None and _COMPACT["ok"] is not False:
        done, failed = [], None
        for a, b in _chunks(c0, c1, COMPACT_CHUNK):
            r = _compact_write(a, _sub(mat, a, b, c0), _sub(occ, a, b, c0))
            if r is None:
                break                                    # this plugin has no voxels_write: the generic route below
            calls += 1
            if not r.get("ok"):
                failed = (a, b, r)
                break
            done.append(r.get("readBack"))
        if failed:
            a, b, r = failed
            return _chunk_failed(r, a, b, calls, len(done), grouped)
        if _COMPACT["ok"] and calls:
            if grouped:
                T.rbx_undo_group("commit")
            return {"ok": True, "calls": calls, "ms": int((time.monotonic() - t0) * 1000), "undoGroup": grouped,
                    "route": "voxels_write", "pluginReadBack": done}
    for a, b in _chunks(c0, c1, CHUNK):
        m, o = _sub(mat, a, b, c0), _sub(occ, a, b, c0)
        if liq is not None:
            chans = {"$t": "Dictionary", "v": {"SolidMaterial": _wgrid(m, _wmat), "SolidOccupancy": _wgrid(o, float),
                                               "LiquidOccupancy": _wgrid(_sub(liq, a, b, c0), float)}}
            r = _call("WriteVoxelChannels", [_wregion(a, b), VOXEL, chans], False)
        else:
            r = _call("WriteVoxels", [_wregion(a, b), VOXEL, _wgrid(m, _wmat), _wgrid(o, float)], False)
        calls += 1
        if not r.get("ok"):
            return _chunk_failed(r, a, b, calls, calls - 1, grouped)
    if grouped:
        T.rbx_undo_group("commit")
    return {"ok": True, "calls": calls, "ms": int((time.monotonic() - t0) * 1000), "undoGroup": grouped}


def _chunk_failed(r, a, b, calls, written, grouped):
    """A chunk failed: cancel the group when there is one, and say what stays when the earlier chunks cannot be taken
    back (no group could be opened, or the cancel was refused)."""
    cancelled = bool(T.rbx_undo_group("cancel").get("ok")) if grouped else False
    if cancelled:
        tail = "; the undo group was cancelled"
    elif written:
        tail = ("; %d earlier chunk(s) stay written (%s), so the region is part-written" % (
            written, "the cancel was refused" if grouped else "no undo group could be opened"))
    else:
        tail = "; nothing was written before it" if not grouped else "; the cancel was refused, nothing was written before it"
    return dict(r, ok=False, chunk=_studs(a, b), calls=calls, undoGroup=grouped, cancelled=cancelled,
                chunksWritten=0 if cancelled else written, partWritten=bool(written) and not cancelled,
                error="writing cells %s: %s%s" % (_studs(a, b), r.get("error"), tail))


def _same_cell(m0, o0, m1, o1, l0=None, l1=None):
    """Did Studio keep (m0, o0, l0)? Air or zero occupancy reads back as Air or 0 (an empty cell has no material)."""
    if m0 == "Air" or o0 <= 0:
        ok = m1 == "Air" or (o1 or 0) <= OCC_TOL
    else:
        ok = m1 == m0 and _num(o1) and abs(o1 - o0) <= OCC_TOL
    if l0 is not None:
        ok = ok and _num(l1) and abs(l1 - l0) <= OCC_TOL
    return ok


def _compare(c0, sent, got, limit=20):
    bad, n = [], _dims(sent["mat"])
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                l0 = sent["liq"][i][j][k] if sent.get("liq") is not None else None
                l1 = got["liq"][i][j][k] if got.get("liq") is not None else None
                if not _same_cell(sent["mat"][i][j][k], sent["occ"][i][j][k], got["mat"][i][j][k], got["occ"][i][j][k],
                                  l0, l1):
                    if len(bad) < limit:
                        bad.append({"cell": [c0[0] + i, c0[1] + j, c0[2] + k],
                                    "sent": [sent["mat"][i][j][k], sent["occ"][i][j][k]] + ([l0] if l0 is not None else []),
                                    "readBack": [got["mat"][i][j][k], got["occ"][i][j][k]] + ([l1] if l1 is not None else [])})
                    else:
                        bad.append(None)
    return [b for b in bad if b], len(bad)


def _summary(block):
    counts, occ_sum, solid = {}, 0.0, 0
    water = 0.0
    for i, plane in enumerate(block["mat"]):
        for j, col in enumerate(plane):
            for k, m in enumerate(col):
                o = block["occ"][i][j][k] or 0
                counts[m] = counts.get(m, 0) + 1
                if m != "Air" and o > 0:
                    solid += 1
                    occ_sum += o
                if block.get("liq") is not None:
                    water += block["liq"][i][j][k] or 0
    out = {"materials": dict(sorted(counts.items(), key=lambda kv: -kv[1])), "nonEmptyCells": solid,
           "occupancySum": round(occ_sum, 4)}
    if block.get("liq") is not None:
        out["liquidSum"] = round(water, 4)
    return out


def _valid_material(name, allow_air=True):
    cat, err = T._catalog()
    if err:
        return err
    items = cat.enums.get("Material") or []
    if not isinstance(name, str) or name not in items:
        return {"ok": False, "ran": False, "refusedRule": "type", "error": "no Material %r; items: %s" % (name, ", ".join(items))}
    if not allow_air and name == "Air":
        return {"ok": False, "ran": False, "refusedRule": "type", "error": "Air has no color or slot: it is empty space"}
    return None


# ---------------------------------------------------------------- batch 6: terrain
def rbx_terrain_info() -> dict:
    "Roblox Studio terrain: Terrain.MaxExtents (cells and studs), the water look, Decoration and GrassLength, the number of non-empty cells (CountCells), and how terrain writes are gated here: the scratch policy (decision 6), the declared scratch box, the chunk size per voxel call. Read only."
    ext, err = _extents()
    props = T._rbx("get_props", target=TERRAIN, props=["WaterColor", "WaterReflectance", "WaterTransparency",
                                                      "WaterWaveSize", "WaterWaveSpeed", "Decoration", "GrassLength"])
    cells = _call("CountCells", [], True)
    out = {"ok": not err and props.get("ok") is True, "policy": terrain_policy(), "scratchBox": SCRATCH_BOX,
           "chunkCells": CHUNK, "voxelStuds": VOXEL,
           "maxExtents": None if err else {"cells": {"min": ext[0], "max": ext[1]}, "studs": _studs(ext[0], [h + 1 for h in ext[1]])},
           "props": {k: V.decode(v) for k, v in T._obj(props.get("values")).items()},
           "cellCount": (cells.get("returns") or [None])[0] if cells.get("ok") else None}
    if not out["ok"]:
        out["error"] = (err or props).get("error")
    return out


def _shape_box(shape, position, size, radius, height, rotation):
    """The shape's axis-aligned bounds in studs and its centre, or a reason."""
    rot = rotation or [1, 0, 0, 0, 1, 0, 0, 0, 1]
    if not _vec(rot, 9):
        return None, "rotation is 9 numbers, the CFrame's rotation matrix row by row"
    if shape in ("block", "wedge"):
        if not _vec(position) or not _vec(size) or min(size) <= 0:
            return None, "%s needs position [x,y,z] and size [x,y,z] above 0" % shape
        half = [sum(abs(rot[3 * i + j]) * size[j] / 2 for j in range(3)) for i in range(3)]
        return {"centre": list(position), "lo": [position[i] - half[i] for i in range(3)],
                "hi": [position[i] + half[i] for i in range(3)], "minDim": min(size)}, None
    if shape == "ball":
        if not _vec(position) or not _num(radius) or radius <= 0:
            return None, "ball needs position (its centre) and radius above 0"
        return {"centre": list(position), "lo": [p - radius for p in position], "hi": [p + radius for p in position],
                "minDim": 2 * radius}, None
    if shape == "cylinder":
        if not _vec(position) or not _num(radius) or radius <= 0 or not _num(height) or height <= 0:
            return None, "cylinder needs position, height and radius above 0 (its axis is the rotation's Y axis)"
        u = [rot[1], rot[4], rot[7]]
        half = [abs(u[i]) * height / 2 + radius * math.sqrt(max(0.0, 1 - u[i] * u[i])) for i in range(3)]
        return {"centre": list(position), "lo": [position[i] - half[i] for i in range(3)],
                "hi": [position[i] + half[i] for i in range(3)], "minDim": min(2 * radius, height)}, None
    return None, "shape is block, ball, cylinder, wedge or region"


def _cell_of(p):
    return [int(math.floor(x / VOXEL)) for x in p]


# A fill changes cells whose centre lies up to 2.48 studs outside the shape, some with no corner inside it (RB1153: a
# block tilted about two axes). A cell whose centre is at least half its space diagonal (2 x sqrt 3 = 3.46 studs) from the
# shape cannot overlap it at all, so the read-back's outside cell is the first one that far out.
OUTSIDE_CLEAR = 2 * math.sqrt(3) + 0.01


def _shape_distance(shape, box, p, size, radius, height, rot):
    """How far point p lies outside the shape (0 inside): a block or wedge by its rotated box, a ball, a cylinder along
    the rotation's Y axis, a region by its cells."""
    c = box["centre"]
    d = [p[i] - c[i] for i in range(3)]
    q = [sum(rot[3 * i + j] * d[i] for i in range(3)) for j in range(3)]     # in the shape's own axes (R^T d)
    if shape in ("block", "wedge"):
        return math.sqrt(sum(max(0.0, abs(q[j]) - size[j] / 2) ** 2 for j in range(3)))
    if shape == "ball":
        return max(0.0, math.sqrt(sum(x * x for x in d)) - radius)
    if shape == "cylinder":
        return math.hypot(max(0.0, math.hypot(q[0], q[2]) - radius), max(0.0, abs(q[1]) - height / 2))
    return math.sqrt(sum(max(0.0, box["lo"][i] - p[i], p[i] - box["hi"][i]) ** 2 for i in range(3)))


def _outside_cell(shape, box, size, radius, height, rot):
    """The read-back's cell beyond the shape: out along the shape's own +X axis (world +X for a ball or region) until
    the cell's centre is OUTSIDE_CLEAR from the shape. Answers the cell, its centre's distance and the axis used."""
    axis = [rot[0], rot[3], rot[6]] if shape in ("block", "wedge", "cylinder") else [1.0, 0.0, 0.0]
    reach = {"block": (size or [0])[0] / 2, "wedge": (size or [0])[0] / 2, "cylinder": radius or 0,
             "ball": radius or 0}.get(shape, (box["hi"][0] - box["lo"][0]) / 2)
    c = box["centre"]
    for step in range(0, 64):
        t = reach + VOXEL / 2 + step
        cell = _cell_of([c[i] + axis[i] * t for i in range(3)])
        dist = _shape_distance(shape, box, [(x + 0.5) * VOXEL for x in cell], size, radius, height, rot)
        if dist >= OUTSIDE_CLEAR:
            return cell, round(dist, 3), [round(a, 4) for a in axis]
    return cell, round(dist, 3), [round(a, 4) for a in axis]


# ---------------------------------------------------------------- dry run: the cells a fill reaches outside its shape
def _shape_poly(shape, box, size, radius, height, rot):
    """The shape as a convex polytope for a separating-axis test: (vertices, face normals, edge directions), world
    space. A cylinder is the 32-sided prism around it (at most 0.5% of the radius wider); a ball is answered apart."""
    c = box["centre"]
    ax = [[rot[0], rot[3], rot[6]], [rot[1], rot[4], rot[7]], [rot[2], rot[5], rot[8]]]   # the shape's own X, Y, Z

    def world(q):
        return [c[i] + sum(ax[j][i] * q[j] for j in range(3)) for i in range(3)]

    def dirn(q):
        return [sum(ax[j][i] * q[j] for j in range(3)) for i in range(3)]
    if shape == "block":
        h = [x / 2 for x in size]
        verts = [world([sx * h[0], sy * h[1], sz * h[2]]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        return verts, list(ax), list(ax)
    if shape == "wedge":
        # A WedgePart's shape: full height at +Z, its slope down to the bottom edge at -Z (the fake and FillWedge agree).
        h = [x / 2 for x in size]
        loc = [[sx * h[0], -h[1], -h[2]] for sx in (-1, 1)] + [[sx * h[0], -h[1], h[2]] for sx in (-1, 1)] + \
              [[sx * h[0], h[1], h[2]] for sx in (-1, 1)]
        slope = dirn([0, size[2], -size[1]])
        along = dirn([0, size[1], size[2]])
        return [world(q) for q in loc], list(ax) + [slope], list(ax) + [along]
    if shape == "cylinder":
        n = 32
        r = radius / math.cos(math.pi / n)
        verts, normals, edges = [], [ax[1]], [ax[1]]
        for i in range(n):
            a, b = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
            for y in (-height / 2, height / 2):
                verts.append(world([r * math.cos(a), y, r * math.sin(a)]))
            m = (a + b) / 2
            normals.append(dirn([math.cos(m), 0, math.sin(m)]))
            edges.append(dirn([math.cos(b) - math.cos(a), 0, math.sin(b) - math.sin(a)]))
        return verts, normals, edges
    return None


_AXES = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _overlaps_cell(poly, lo, eps=1e-6):
    """Does the 4-stud cell with minimum corner lo share volume with the convex polytope? Separating-axis test."""
    verts, normals, edges = poly
    cube = [[lo[0] + a * VOXEL, lo[1] + b * VOXEL, lo[2] + c * VOXEL] for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    axes = _AXES + list(normals) + [_cross(e, a) for e in edges for a in _AXES]
    for axis in axes:
        if sum(x * x for x in axis) < 1e-12:
            continue
        pa = [sum(v[i] * axis[i] for i in range(3)) for v in verts]
        pb = [sum(v[i] * axis[i] for i in range(3)) for v in cube]
        if min(max(pa), max(pb)) - max(min(pa), min(pb)) <= eps * math.sqrt(sum(x * x for x in axis)):
            return False
    return True


def _wedge_above(box, p, size, rot):
    """How far p, inside a wedge's box, lies above its slope (> 0: outside the wedge)."""
    q = [sum(rot[3 * a + b] * (p[a] - box["centre"][a]) for a in range(3)) for b in range(3)]
    return ((q[1] + size[1] / 2) * size[2] - (q[2] + size[2] / 2) * size[1]) / math.hypot(size[1], size[2])


def _centre_inside(shape, box, centre, size, radius, height, rot):
    """Does a cell centre lie inside the shape? Such a cell the fill sets whole (FOUND 2026-11-21: the cells a fill
    empties are the ones the dry run's thin-cell scan has to predict)."""
    if _shape_distance(shape, box, centre, size, radius, height, rot) > 0:
        return False
    return shape != "wedge" or _wedge_above(box, centre, size, rot) <= 0


def _fill_reach(shape, box, size, radius, height, rot, c0, c1):
    """Every cell whose centre lies outside the shape but whose cube shares volume with it: the cells a fill can change
    outside its shape (RB1153: up to 2.48 studs out, some with no corner inside). (cells, scanned)."""
    if shape == "region":
        return [], _count(c0, c1)
    poly = None if shape == "ball" else _shape_poly(shape, box, size, radius, height, rot)
    out, scanned = [], 0
    for i in range(c0[0] - 1, c1[0] + 1):
        for j in range(c0[1] - 1, c1[1] + 1):
            for k in range(c0[2] - 1, c1[2] + 1):
                scanned += 1
                lo = [i * VOXEL, j * VOXEL, k * VOXEL]
                centre = [x + VOXEL / 2 for x in lo]
                d = _shape_distance(shape, box, centre, size, radius, height, rot)
                if shape == "wedge" and d == 0:
                    above = _wedge_above(box, centre, size, rot)
                    if above > 0:
                        d = above                     # inside the wedge's box, above its slope: outside the wedge by this
                if d <= 0 or d >= OUTSIDE_CLEAR:
                    continue
                if shape == "ball":
                    near = [min(max(box["centre"][a], lo[a]), lo[a] + VOXEL) for a in range(3)]
                    hit = math.dist(near, box["centre"]) < radius - 1e-6
                else:
                    hit = _overlaps_cell(poly, lo)
                if hit:
                    out.append(([i, j, k], round(d, 3)))
    return out, scanned


def _dry_run(shape, material, box, size, radius, height, rot, c0, c1, asked_top):
    cells, scanned = _fill_reach(shape, box, size, radius, height, rot, c0, c1)
    if not scanned:
        return {"ok": False, "ran": False, "dryRun": True, "error": "the shape covers no cell: nothing was scanned"}
    r0 = [c - 1 for c in c0]
    r1 = [c + 1 for c in c1]
    if _count(r0, r1) > MAX_RETURN_CELLS:
        return {"ok": False, "ran": False, "dryRun": True, "refusedRule": "budget",
                "error": "a dry run reads the shape's box and one cell round it: %d cells is over %d" % (_count(r0, r1), MAX_RETURN_CELLS)}
    # THE THIN-CELL SCAN READS WIDER (FOUND 2026-11-21, RB1754): a thin cell becomes a hole when the fill empties a cell
    # beside it, or the one its open neighbour runs on to, or the one under it - so the read takes 4 cells round the
    # shape across, 2 above it and 8 under it (the depth to the ground under a void, RB1046's was about 20 studs), and
    # judges the cells up to 2 out.
    t0, t1 = [c0[0] - 4, c0[1] - 8, c0[2] - 4], [c1[0] + 4, c1[1] + 2, c1[2] + 4]
    wide = _count(t0, t1) <= MAX_RETURN_CELLS
    b0, b1 = (t0, t1) if wide else (r0, r1)
    now, err = _read_block(b0, b1)
    if err:
        return dict(err, ok=False, dryRun=True)
    rows, solid = [], 0
    for cell, d in cells:
        X, Y, Z = (cell[a] - b0[a] for a in range(3))
        m, o = now["mat"][X][Y][Z], now["occ"][X][Y][Z]
        holds = m != "Air" and (o or 0) > OCC_TOL
        solid += holds
        rows.append({"min": [x * VOXEL for x in cell], "centreOutside": d, "material": m, "occupancy": round(o or 0, 4),
                     "solidNow": holds})
    rows.sort(key=lambda r: (not r["solidNow"], -r["centreOutside"]))
    out = {"ok": True, "dryRun": True, "written": False, "shape": shape, "material": material, "region": _studs(c0, c1),
           "scannedCells": scanned, "outsideCells": len(rows), "outsideSolidNow": solid,
           "farthest": max((d for _c, d in cells), default=0.0), "cells": rows[:2000],
           "surface": "drawn" if asked_top is not None else "solid", "surfaceConvention": SURFACE_CONVENTION,
           "rule": "cells whose centre lies outside the shape and whose 4-stud cube shares volume with it (RB1153: a fill "
                   "changes cells up to 2.48 studs out, some with no corner inside; one 3.47 studs out or more is never touched)"}
    if len(rows) > 2000:
        out["note"] = "%d cells; the first 2000 are listed, those holding solid ground first" % len(rows)
    if material == "Air":
        out["wouldErase"] = solid
    if wide:
        out["thinOverVoid"] = _thin_after_fill(shape, material, box, size, radius, height, rot, now, b0, b1, c0, c1, cells)
        if out["thinOverVoid"]["new"]:
            out["warning"] = ("this fill would leave %d thin cells over a void at the edge of an opening, which draw and "
                              "raycast as holes up to %g studs past that edge (RB1046, RB1754): see thinOverVoid"
                              % (out["thinOverVoid"]["new"], out["thinOverVoid"]["widestNewBand"]))
    else:
        out["thinOverVoid"] = {"scanned": False, "why": "the scan reads 4 cells round the shape: %d cells is over %d, so "
                                                        "it was not run; fill in parts, or run rbx_terrain_voxels holes "
                                                        "after the write" % (_count(t0, t1), MAX_RETURN_CELLS)}
    return out


def _thin_after_fill(shape, material, box, size, radius, height, rot, now, b0, b1, c0, c1, reach):
    """The holes scan (rbx_terrain_voxels holes' own rule, _scan_holes) run on the terrain as the fill would leave it: a
    cell whose centre lies inside the shape set whole (material at 1, or empty for Air). A cell the fill only partly
    reaches (RB1153), by what was measured: an Air fill empties a thin cell of about 0.3 whose centre is up to
    AIR_REACH_EMPTIES outside and leaves one AIR_REACH_KEEPS or more out as it was (RB1922), and leaves a full cell solid
    (1.0 to 0.79-0.94, RB1153). Any other partly reached cell - a solid fill's, or a thin cell between those distances or
    of another occupancy - is kept as it is now and every thin cell beside, over or on it is marked uncertain."""
    n = [b1[a] - b0[a] for a in range(3)]
    after = {"mat": [[list(col) for col in plane] for plane in now["mat"]],
             "occ": [[list(col) for col in plane] for plane in now["occ"]]}
    inside = 0
    for X in range(n[0]):
        for Y in range(n[1]):
            for Z in range(n[2]):
                cell = [b0[0] + X, b0[1] + Y, b0[2] + Z]
                if shape == "region":
                    hit = all(c0[a] <= cell[a] < c1[a] for a in range(3))
                else:
                    hit = _centre_inside(shape, box, [(x + 0.5) * VOXEL for x in cell], size, radius, height, rot)
                if hit:
                    inside += 1
                    after["mat"][X][Y][Z], after["occ"][X][Y][Z] = (material, 1.0) if material != "Air" else ("Air", 0.0)
    partly, emptied = set(), 0
    for c, d in reach:
        X, Y, Z = (c[a] - b0[a] for a in range(3))
        m, o = now["mat"][X][Y][Z], now["occ"][X][Y][Z] or 0
        if m == "Air" or o <= OCC_TOL:
            if material != "Air":
                partly.add(tuple(c))               # a solid fill grows an empty cell by an amount not measured enough
            continue
        if material == "Air" and abs(o - AIR_REACH_THIN) <= 0.05 and d <= AIR_REACH_EMPTIES:
            after["mat"][X][Y][Z], after["occ"][X][Y][Z] = "Air", 0.0
            emptied += 1
        elif material == "Air" and ((abs(o - AIR_REACH_THIN) <= 0.05 and d >= AIR_REACH_KEEPS) or o >= 0.99):
            pass                                   # kept: as it was, or a full cell left solid
        else:
            partly.add(tuple(c))
    _hb, _cb, _colsb, _eb, before_cells = _scan_holes(now, b0, n, THIN_DEFAULT)
    holes, closed, columns, _eu, cells = _scan_holes(after, b0, n, THIN_DEFAULT)
    was = set(before_cells)
    listed, unknown_above, uncertain = [], 0, 0
    for h, (i, top, k) in zip(holes, cells):
        if not (2 <= i <= n[0] - 3 and 2 <= k <= n[2] - 3):
            continue                               # beyond 2 cells out: the fill changes nothing that judges this cell
        if top == n[1] - 1:
            unknown_above += 1                     # the read's top row: ground above it was not read
            continue
        row = {key: h[key] for key in ("column", "thinCell", "openFaces", "bandStuds", "emptyCellsUnder", "depth")
               if key in h}
        row["new"] = (i, top, k) not in was
        w = (b0[0] + i, b0[1] + top, b0[2] + k)
        near = [w, (w[0], w[1] - 1, w[2])] + [(w[0] + di, w[1], w[2] + dk) for di, dk in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        if any(c in partly for c in near):
            row["uncertain"] = ("the fill only partly changes a cell of it, under it or beside it (RB1153) in a way not "
                                "measured: read it with rbx_terrain_voxels holes after the write")
            uncertain += 1
        listed.append(row)
    listed.sort(key=lambda r: (not r["new"], r["column"]))
    return {"scanned": True, "thin": THIN_DEFAULT, "read": _studs(b0, b1), "columns": columns, "count": len(listed),
            "new": sum(1 for r in listed if r["new"]), "uncertain": uncertain, "cells": listed[:500],
            "widestNewBand": max((r["bandStuds"] for r in listed if r["new"]), default=0.0),
            "insideCells": inside, "reachEmptied": emptied, "reachNotPredicted": len(partly), "unknownAbove": unknown_above,
            "rule": "rbx_terrain_voxels holes on the terrain as this fill would leave it (cells whose centre is inside "
                    "set whole; an Air fill also empties a thin cell it reaches within %g studs of its centre, RB1922): "
                    "a top cell under %g occupancy over an empty cell, beside an opening in its own layer "
                    "that runs on along that face, lets rays through bandStuds past it (RB1754). new: not such a cell "
                    "now" % (AIR_REACH_EMPTIES, THIN_DEFAULT)}


def _read_cell(c):
    b, err = _read_block(c, [c[0] + 1, c[1] + 1, c[2] + 1])
    if err:
        return None, err
    return (b["mat"][0][0][0], b["occ"][0][0][0]), None


@T._reports
def rbx_terrain_fill(shape: str, material: str, position: list = None, size: list = None, radius: float = None,
                     height: float = None, rotation: list = None, region: dict = None, surface: bool = False,
                     dry_run: bool = False, timeout: float = 60) -> dict:
    "Roblox Studio terrain: fill a block, ball, cylinder or wedge (Terrain FillBlock, FillBall, FillCylinder, FillWedge) or an axis-aligned region (FillRegion, aligned outward to the 4-stud grid) with a material; Air erases. Read back by Terrain:ReadVoxels: the material and occupancy at the shape's centre (a wedge: the filled volume of its box, about half), and a cell outside it unchanged: out along the shape's own X axis, the first cell whose centre is 3.47 studs (half a cell's diagonal) or more from the shape, which no part of the fill can reach. A solid fill also answers drawnTop, where Studio draws its surface (the median of nine WorldRoot:Raycast hits across the middle; relief, their spread, which a rough material such as Rock or Slate makes about a stud): about 2 studs above the solid's top (RB1151). surface true (an upright block or a region, solid material) fills the block 2 studs shorter so the drawn top lands on the top asked, read back within 0.25 stud on a smooth material and inside the relief band on a rough one. Every reply that wrote says which convention it wrote: surface \"drawn\" (surface true: the drawn top is the top asked) or \"solid\" (the solid ends at the shape; Studio draws its top about 2 studs higher), and surfaceConvention, the number of that rule (2 since the heightmap fix; 1 was the heightmap writing solid to h). dry_run true writes nothing and answers the cells the fill would change outside its shape: every cell whose centre lies outside it and whose 4-stud cube shares volume with it (RB1153), each with its minimum corner, material and occupancy now, and how many hold solid ground now (wouldErase for Air); and thinOverVoid, rbx_terrain_voxels holes run on the terrain as the fill would leave it (cells whose centre is inside set whole, and a 0.3 cell an Air fill reaches with its centre up to 0.823 studs out emptied, RB1922; read 4 cells round the shape): every thin top cell over a void beside an opening running along its face, with bandStuds, the studs past that face rays fall through (RB1754), new when it is not such a cell now (a warning names them), uncertain when the fill partly changes a cell of it, under it or beside it in a way not measured (a solid fill's reach, another occupancy). Proven on RB1046's replay (RB1922): at each of eight turns the shaft's dry run named exactly the 11-13 holes its write made, and the room's named none and left none. Refused before sending: a shape outside Terrain.MaxExtents, and in scratch mode anything decision 6 does not allow. timeout (5-300 s, default 60) is how long the mode check before the write and the fill itself each wait; a mode check that gets no answer refuses with sent false - the fill was not sent - so it can be sent again."
    bad = _valid_material(material)
    if bad:
        return bad
    if not isinstance(surface, bool) or not isinstance(dry_run, bool):
        return {"ok": False, "ran": False, "error": "surface and dry_run are true or false"}
    if not _num(timeout) or not 5 <= timeout <= 300:
        return {"ok": False, "ran": False, "error": "timeout is 5-300 seconds"}
    asked_top = None
    if surface:
        # SMOOTH TERRAIN DRAWS A SOLID'S TOP 2 STUDS ABOVE IT (RB1150, RB1151): the block is filled SURFACE_LIFT shorter.
        if material == "Air":
            return {"ok": False, "ran": False, "error": "surface places a solid's drawn top; Air has none"}
        if shape == "region":
            c0r, c1r, why = _region_cells(region)
            if why:
                return {"ok": False, "ran": False, "error": why}
            lo, hi = [c * VOXEL for c in c0r], [c * VOXEL for c in c1r]
            shape, rotation = "block", None
            position, size = [(a + b) / 2 for a, b in zip(lo, hi)], [b - a for a, b in zip(lo, hi)]
        elif shape != "block":
            return {"ok": False, "ran": False, "error": "surface takes an upright block or a region: a %s has no flat top" % shape}
        if not _vec(position) or not _vec(size):
            return {"ok": False, "ran": False, "error": "block needs position [x,y,z] and size [x,y,z] above 0"}
        rr = rotation or [1, 0, 0, 0, 1, 0, 0, 0, 1]
        if not _vec(rr, 9) or abs(rr[4] - 1) > 1e-6:
            return {"ok": False, "ran": False, "error": "surface needs an upright block (the rotation's Y axis straight up): a "
                                                        "tilted block has no level top"}
        if size[1] <= SURFACE_LIFT:
            return {"ok": False, "ran": False, "error": "a block %g studs tall cannot be filled %g shorter" % (size[1], SURFACE_LIFT)}
        asked_top = position[1] + size[1] / 2
        size = [size[0], size[1] - SURFACE_LIFT, size[2]]
        position = [position[0], position[1] - SURFACE_LIFT / 2, position[2]]
    if shape == "region":
        c0, c1, why = _region_cells(region)
        if why:
            return {"ok": False, "ran": False, "error": why}
        lo, hi = [c * VOXEL for c in c0], [c * VOXEL for c in c1]
        box = {"centre": [(a + b) / 2 for a, b in zip(lo, hi)], "lo": lo, "hi": hi, "minDim": min(b - a for a, b in zip(lo, hi))}
    else:
        box, why = _shape_box(shape, position, size, radius, height, rotation)
        if why:
            return {"ok": False, "ran": False, "error": why}
        c0, c1 = _cell_of(box["lo"]), [int(math.ceil(x / VOXEL)) for x in box["hi"]]
    if _count(c0, c1) > MAX_CELLS:
        return {"ok": False, "ran": False, "refusedRule": "budget", "error": "%d cells is over %d a call" % (_count(c0, c1), MAX_CELLS)}
    rot = rotation or [1, 0, 0, 0, 1, 0, 0, 0, 1]
    if dry_run:
        return _dry_run(shape, material, box, size, radius, height, rot, c0, c1, asked_top)
    out_cell, out_dist, out_axis = _outside_cell(shape, box, size, radius, height, rot)
    gate = _terrain_gate(c0, c1, "rbx_terrain_fill", timeout=timeout)
    if gate:
        return gate
    before_out, err = _read_cell(out_cell)
    if err:
        return err
    wedge_before, centre_before = None, None
    centre_cell = _cell_of(box["centre"])
    # The centre cell must read full (or, for Air, empty) only when all of it lies inside the shape: a cell the shape's
    # centre sits in can still reach past a thin or turned shape (RB1154: a 3.5-stud Air slab, a radius-5 cylinder).
    whole = all(_shape_distance(shape, box, [(centre_cell[0] + a) * VOXEL, (centre_cell[1] + b) * VOXEL,
                                             (centre_cell[2] + c) * VOXEL], size, radius, height, rot) == 0
                for a in (0, 1) for b in (0, 1) for c in (0, 1))
    if shape != "wedge" and material == "Air" and not whole:
        centre_before, err = _read_cell(centre_cell)
        if err:
            return err
    if shape == "wedge":
        if _count(c0, c1) > MAX_RETURN_CELLS:
            return {"ok": False, "ran": False, "refusedRule": "budget",
                    "error": "a wedge is read back over its whole box; %d cells is over %d" % (_count(c0, c1), MAX_RETURN_CELLS)}
        wedge_before, err = _read_block(c0, c1)
        if err:
            return err
    m = _wmat(material)
    if shape == "block":
        r = _call("FillBlock", [_wcf(position, rot), _w3(size), m], False, timeout=timeout)
    elif shape == "wedge":
        r = _call("FillWedge", [_wcf(position, rot), _w3(size), m], False, timeout=timeout)
    elif shape == "ball":
        r = _call("FillBall", [_w3(position), float(radius), m], False, timeout=timeout)
    elif shape == "cylinder":
        r = _call("FillCylinder", [_wcf(position, rot), float(height), float(radius), m], False, timeout=timeout)
    else:
        r = _call("FillRegion", [_wregion(c0, c1), VOXEL, m], False, timeout=timeout)
    if not r.get("ok"):
        return r
    problems, centre, filled = [], None, None
    after_out, err = _read_cell(out_cell)
    if err:
        return dict(err, ok=False, error="filled, but the read-back failed: %s" % err.get("error"))
    if not _same_cell(before_out[0], before_out[1], after_out[0], after_out[1]):
        problems.append("the cell outside (%s, its centre %.2f studs from the shape along %s) changed from %s to %s"
                        % (out_cell, out_dist, out_axis, before_out, after_out))
    if shape == "wedge":
        after, err = _read_block(c0, c1)
        if err:
            return err
        gained = 0.0
        for i, plane in enumerate(after["mat"]):
            for j, col in enumerate(plane):
                for k, mm in enumerate(col):
                    if mm == material:
                        was = wedge_before["occ"][i][j][k] if wedge_before["mat"][i][j][k] == material else 0
                        gained += max(0.0, after["occ"][i][j][k] - was)
        vol = (box["hi"][0] - box["lo"][0]) * (box["hi"][1] - box["lo"][1]) * (box["hi"][2] - box["lo"][2]) / VOXEL ** 3
        # JUDGED BY WHAT THE BOX HOLDS AFTER, not by what it gained: the same wedge filled again gains nothing and is
        # right (FOUND 2026-11-20, a build run twice). A fill that did nothing to an empty box still holds nothing.
        held = sum(after["occ"][i][j][k] for i, plane in enumerate(after["mat"]) for j, col in enumerate(plane)
                   for k, mm in enumerate(col) if mm == material)
        filled = {"cells": round(held, 3), "gained": round(gained, 3), "boxCells": round(vol, 3),
                  "ratio": round(held / vol, 3) if vol else None}
        if material != "Air" and not (vol and 0.25 <= held / vol <= 0.75):
            problems.append("after the fill the wedge's box holds %.2f of its %.2f cells of %s (%.2f gained), not about "
                            "half" % (held, vol, material, gained))
    else:
        cm, co = _read_cell(centre_cell)[0] or (None, None)
        centre = {"cell": centre_cell, "material": cm, "occupancy": co, "wholeInside": whole}
        empty = cm == "Air" or (co or 0) <= OCC_TOL
        if material == "Air" and whole:
            if not empty:
                problems.append("the centre cell, wholly inside the shape, still holds %s at %s" % (cm, co))
        elif material == "Air":
            was = centre_before[1] if centre_before and centre_before[0] != "Air" else 0.0
            centre["before"] = list(centre_before) if centre_before else None
            if not empty and not (_num(co) and co < was - OCC_TOL):
                problems.append("the centre cell (partly inside the shape) still holds %s at %s, %s before: nothing erased"
                                % (cm, co, was))
        else:
            need = 0.99 if whole else OCC_TOL
            if cm != material or not _num(co) or co < need:
                problems.append("the centre reads %s at occupancy %s, expected %s at %s or more (%s)"
                                % (cm, co, material, need, "the cell lies wholly inside" if whole else "the cell is partly inside"))
    drawn = None
    if material != "Air":
        # NINE RAYS, THE MEDIAN COUNTS: a rough material draws relief about its level top (RB1158: Rock and Slate wander
        # up to about 1.3 studs, Grass, Sand and Concrete read flat to 0.001), so the top is the median of a 3 x 3 grid
        # across the middle, and relief is the spread of the nine.
        cx, cz = box["centre"][0], box["centre"][2]
        reach = max(0.0, min(4.0, (box["hi"][0] - box["lo"][0]) / 4, (box["hi"][2] - box["lo"][2]) / 4))
        ys = []
        for dx in (-reach, 0.0, reach):
            for dz in (-reach, 0.0, reach):
                hit, err = _raycast_down(cx + dx, cz + dz, box["hi"][1] + 16, box["hi"][1] - box["lo"][1] + 32)
                if err:
                    return dict(err, ok=False, error="filled, but the drawn-top Raycast failed: %s" % err.get("error"))
                if hit.get("hit"):
                    ys.append(hit["position"][1])
        ys.sort()
        drawn = {"at": [cx, cz], "spread": round(reach, 3), "rays": len(ys), "y": round(ys[len(ys) // 2], 3) if ys else None,
                 "lowest": round(ys[0], 3) if ys else None, "highest": round(ys[-1], 3) if ys else None,
                 "relief": round(ys[-1] - ys[0], 3) if ys else None}
        upright = shape == "region" or (shape == "block" and abs(rot[4] - 1) <= 1e-6)
        if upright:
            drawn["solidTop"] = round(box["hi"][1], 3)
            drawn["above"] = None if drawn["y"] is None else round(drawn["y"] - box["hi"][1], 3)
        if asked_top is not None:
            drawn["asked"] = round(asked_top, 3)
            if drawn["y"] is None:
                problems.append("surface: no drawn top was hit across the middle")
            elif drawn["relief"] <= ROUGH_RELIEF:
                if abs(drawn["y"] - asked_top) > 0.25:
                    problems.append("surface: the drawn top reads %s (median of %d rays), asked %.3f (tolerance 0.25)"
                                    % (drawn["y"], drawn["rays"], asked_top))
            else:
                # a rough material: the asked top must lie inside the band its relief draws
                drawn["rough"] = True
                if not drawn["lowest"] - 0.25 <= asked_top <= drawn["highest"] + 0.25:
                    problems.append("surface: %s draws its top between %s and %s, which does not hold the asked %.3f"
                                    % (material, drawn["lowest"], drawn["highest"], asked_top))
    out = {"ok": not problems, "verified": not problems, "shape": shape, "material": material, "region": _studs(c0, c1),
           "centre": centre, "outside": {"cell": out_cell, "distance": out_dist, "along": out_axis,
                                         "before": list(before_out), "after": list(after_out)},
           "requestId": r.get("requestId")}
    if filled:
        out["filled"] = filled
    if drawn:
        out["drawnTop"] = drawn
    out.update(surface="drawn" if asked_top is not None else "solid", surfaceConvention=SURFACE_CONVENTION)
    if asked_top is not None:
        out["surfaceAsked"] = {"askedTop": round(asked_top, 3), "solidTop": round(asked_top - SURFACE_LIFT, 3)}
    if problems:
        out["error"] = "; ".join(problems)
    return out


def _grid_ok(g, n, leaf_ok):
    if _dims(g) != n or len(g) != n[0]:
        return False
    for p in g:
        if not isinstance(p, list) or len(p) != n[1] or not all(isinstance(col, list) and len(col) == n[2] for col in p):
            return False
    return all(leaf_ok(c) for p in g for col in p for c in col)


@T._reports
def rbx_terrain_voxels(action: str, region: dict = None, origin: list = None, materials: list = None,
                       occupancy: list = None, water: list = None, summary: bool = False, with_water: bool = False,
                       thin: float = THIN_DEFAULT, rays: bool = True, columns: list = None, min_gap: float = VOXEL,
                       heightmap: dict = None, max_above: float = 4.0) -> dict:
    "Roblox Studio terrain voxels by region, chunked. action get: region {min, max} in studs (aligned outward to 4) -> grids [x][y][z] of material names and occupancy 0-1 (with_water adds LiquidOccupancy through ReadVoxelChannels); above 65,536 cells, or with summary, counts per material instead. action set: origin [x, y, z] (a cell corner, multiples of 4) with materials and occupancy grids of one shape (water adds LiquidOccupancy through WriteVoxelChannels), written as one undo step and read back cell by cell: Air or 0 reads back empty, occupancy to 0.01; the reply says surface \"solid\" (the cells are as given; Studio draws a surface 2 studs above the top solid) and surfaceConvention. action holes: every column of region whose top solid cell holds less than thin (default 0.5) occupancy with an empty cell under it AND a face beside an opening in its own layer that runs on along that face (the empty neighbour has an empty neighbour along the face) - where RB1046's hole came from: there smooth terrain lets a ray through a band past that face, 1.25 studs at occupancy 0.3 (bandStuds, from 1.75 at 0.1 to 0 at 1.0, RB1754), so the opening reads and shows that much wider down to the ground under the void. Each hole gives its column centre, the thin cell (bottom y, material, occupancy), openFaces, bandStuds, the empty cells under it and the depth to the next solid ground in the region. The same thin cell with all four neighbours solid, or beside a single empty cell, draws and stops every ray (RB1602, RB1755): it is listed under closed, not as a hole; one at the region's edge is counted in regionEdgeUnknown (scan one cell wider). With rays (default) a terrain-only WorldRoot:Raycast 0.25 stud inside each open face (up to 100) says where a ray stops, passesThrough when below the thin cell, and confirmedHoles counts the holes with one that did. A region with no terrain fails. Read only. action floating: terrain left floating over the ground, as a heightmap written under an older, taller build leaves it (arches, slabs, discs, RB3065). Over the whole region (at most 1,048,576 cells) or, with columns [[x, z], ...] (at most 400), those columns over the region's height (its x and z are then not used): in each column the lowest solid run is the ground and every solid run above an air gap of at least min_gap studs (default 4, one cell) is floating, with its column, bottom and top, thickness, gapStuds, the ground's top under it and its material (the run's commonest; topEdge when it reaches the region's top, so read taller). Floating columns that touch side by side are grouped into pieces (columns, box, the largest gap, materials, at: the column with the largest gap). heightmap {origin, heights} (as given to rbx_terrain_heightmap; one height per 4-stud column, null for none) adds the drawn surface: a terrain-only WorldRoot:Raycast down from the region's top at each scanned column inside it (up to 100, evenly spread) and every column drawn more than max_above studs (default 4) above its written height is listed under above (written, drawn, above, the hit's material, and the voxel gap when one is under it). clean is true when nothing is floating or above. A scan that read no terrain fails, and so does a heightmap that covers none of the columns scanned. Read only. With plugin 0.15.0 get and set go through its compact voxel ops (a palette and one byte a cell, 64 cells a side a call); water still takes the generic channel calls."
    if action == "get":
        c0, c1, why = _region_cells(region)
        if why:
            return {"ok": False, "ran": False, "error": why}
        n = _count(c0, c1)
        if n > MAX_CELLS:
            return {"ok": False, "ran": False, "refusedRule": "budget", "error": "%d cells is over %d a call" % (n, MAX_CELLS)}
        block, err = _read_block(c0, c1, water=bool(with_water))
        if err:
            return err
        out = {"ok": True, "region": _studs(c0, c1), "origin": [c * VOXEL for c in c0],
               "size": [c1[i] - c0[i] for i in range(3)], "calls": block["calls"], "summary": _summary(block)}
        if not summary and n <= MAX_RETURN_CELLS:
            out.update(materials=block["mat"], occupancy=block["occ"])
            if with_water:
                out["water"] = block["liq"]
        elif not summary:
            out["note"] = "%d cells is over %d: summary only; read a smaller region for the grids" % (n, MAX_RETURN_CELLS)
        return out
    if action == "holes":
        return _holes(region, thin, rays)
    if action == "floating":
        return _floating(region, columns, min_gap, heightmap, max_above)
    if action != "set":
        return {"ok": False, "ran": False, "error": "action is get, set, holes or floating"}
    if not _vec(origin) or any(float(x) % VOXEL for x in origin):
        return {"ok": False, "ran": False, "error": "origin is [x, y, z] in studs on the 4-stud grid (a cell's corner)"}
    n = _dims(materials)
    if n is None or min(n) < 1:
        return {"ok": False, "ran": False, "error": "materials is a non-empty grid [x][y][z] of Material names"}
    cat, err = T._catalog()
    if err:
        return err
    items = set(cat.enums.get("Material") or [])
    if not _grid_ok(materials, n, lambda c: c in items):
        return {"ok": False, "ran": False, "refusedRule": "type", "error": "materials must be a rectangular grid of Material names"}
    if not _grid_ok(occupancy, n, lambda c: _num(c) and 0 <= c <= 1):
        return {"ok": False, "ran": False, "refusedRule": "type", "error": "occupancy must be a grid the shape of materials (%s), values 0-1" % n}
    if water is not None and not _grid_ok(water, n, lambda c: _num(c) and 0 <= c <= 1):
        return {"ok": False, "ran": False, "refusedRule": "type", "error": "water must be a grid the shape of materials (%s), values 0-1" % n}
    if n[0] * n[1] * n[2] > MAX_CELLS:
        return {"ok": False, "ran": False, "refusedRule": "budget", "error": "%d cells is over %d a call" % (n[0] * n[1] * n[2], MAX_CELLS)}
    c0 = [int(x) // VOXEL for x in origin]
    c1 = [c0[i] + n[i] for i in range(3)]
    gate = _terrain_gate(c0, c1, "rbx_terrain_voxels set")
    if gate:
        return gate
    w = _write_block(c0, materials, occupancy, water, label="terrain voxels")
    if not w.get("ok"):
        return w
    got, err = _read_block(c0, c1, water=water is not None)
    if err:
        return dict(err, ok=False, written=w, error="written, but the read-back failed: %s" % err.get("error"))
    bad, count = _compare(c0, {"mat": materials, "occ": occupancy, "liq": water}, got)
    out = {"ok": not count, "verified": not count, "region": _studs(c0, c1), "cells": n[0] * n[1] * n[2],
           "writeCalls": w["calls"], "undoGroup": w["undoGroup"], "ms": w["ms"], "surface": "solid",
           "surfaceConvention": SURFACE_CONVENTION}
    if count:
        out.update(mismatched=count, firstMismatches=bad, error="%d cells read back different" % count)
    return out


MAX_HOLE_RAYS = 100
# Where a thin top cell over a void lets a terrain ray through (RB1754, RB1755): only within a band past a face it shares
# with an opening in its own layer that runs on along that face (an empty neighbour whose own neighbour along the face is
# empty too). Beside a single empty cell, or with all four neighbours solid, it draws and stops every ray. The band, in
# studs past the face, falls with the thin cell's occupancy (rays every 0.25 stud beside a strip opening; a 2 x 2 opening
# made 0.75 at 0.3 where a strip made 1.25, so these are the most measured). How deep the void is did not matter.
EDGE_BAND = ((0.1, 1.75), (0.2, 1.5), (0.3, 1.25), (0.45, 1.0), (0.5, 0.75), (0.6, 0.75), (0.8, 0.25), (1.0, 0.0))
EDGE_RAY_IN = 0.25                          # studs inside the face where the scan's ray goes down


def edge_band(occ):
    """Studs past an open face where a thin cell of this occupancy let rays through (EDGE_BAND, linear between rows)."""
    if occ <= EDGE_BAND[0][0]:
        return EDGE_BAND[0][1]
    for (o0, b0), (o1, b1) in zip(EDGE_BAND, EDGE_BAND[1:]):
        if occ <= o1:
            return round(b0 + (b1 - b0) * (occ - o0) / (o1 - o0), 3)
    return 0.0


def _holes(region, thin, rays):
    """rbx_terrain_voxels holes: thin top cells over a void at the edge of an opening (RB1046, RB1754). Read only."""
    if not (_num(thin) and 0 < thin <= 1) or not isinstance(rays, bool):
        return {"ok": False, "ran": False, "error": "thin is an occupancy above 0 and at most 1; rays true or false"}
    c0, c1, why = _region_cells(region)
    if why:
        return {"ok": False, "ran": False, "error": why}
    n = [c1[i] - c0[i] for i in range(3)]
    if _count(c0, c1) > MAX_CELLS:
        return {"ok": False, "ran": False, "refusedRule": "budget", "error": "%d cells is over %d a call" % (_count(c0, c1), MAX_CELLS)}
    if n[1] < 2:
        return {"ok": False, "ran": False, "error": "a hole is a cell over an empty one: the region must be at least 8 studs tall"}
    block, err = _read_block(c0, c1)
    if err:
        return err
    holes, closed, columns, edge_unknown, _cells = _scan_holes(block, c0, n, thin)
    if not columns:
        return {"ok": False, "region": _studs(c0, c1), "columns": 0,
                "error": "no terrain in the region: nothing was scanned, so no hole is not a clean answer"}
    out = {"ok": True, "region": _studs(c0, c1), "columns": columns, "thin": thin, "holes": holes[:2000],
           "holeCount": len(holes), "closedThinOverVoid": len(closed), "closed": closed[:200], "calls": block["calls"],
           "rule": "the top solid cell under %g occupancy with an empty cell under it, beside an opening in its own layer "
                   "that runs on along the shared face: rays pass a band past that face (bandStuds, by occupancy), so the "
                   "opening reads and shows that much wider over the void (RB1046, RB1754). The same cell with no such "
                   "face draws and stops every ray (RB1602, RB1755): it is listed under closed, not as a hole" % thin}
    if edge_unknown:
        out["regionEdgeUnknown"] = edge_unknown
        out["note"] = "%d thin cells over a void sit at the region's edge, where an opening outside it cannot be seen: " \
                      "scan a region one cell wider" % edge_unknown
    if rays and holes:
        top_y = c1[1] * VOXEL + 16
        through, cast = 0, 0
        for h in holes:
            if cast >= MAX_HOLE_RAYS:
                break
            cx, cz = h["column"]
            h["rays"] = []
            for side in h["openFaces"]:
                if cast >= MAX_HOLE_RAYS:
                    break
                sgn = 1 if side[0] == "+" else -1
                d = VOXEL / 2 - EDGE_RAY_IN
                x, z = (cx + sgn * d, cz) if side[1] == "x" else (cx, cz + sgn * d)
                hit, err = _raycast_down(x, z, top_y, top_y - c0[1] * VOXEL + 64)
                cast += 1
                if err:
                    return dict(err, ok=False, error="holes found, but the Raycast check failed: %s" % err.get("error"), holes=holes[:50])
                y = round(hit["position"][1], 3) if hit.get("hit") else None
                h["rays"].append({"side": side, "at": [round(x, 3), round(z, 3)], "hitY": y,
                                  "passesThrough": y is None or y < h["thinCell"]["minY"]})
            passed = [r_ for r_ in h["rays"] if r_["passesThrough"]]
            h["ray"] = passed[0] if passed else (h["rays"][0] if h["rays"] else None)
            through += bool(passed)
        out["rays"] = {"cast": cast, "passedThrough": through, "at": "%g stud inside each open face, at its middle" % EDGE_RAY_IN}
        out["confirmedHoles"] = through
    return out


def _scan_holes(block, c0, n, thin):
    """The holes rule over a block read at cell c0 with n cells a side: (holes, closed, columns, edgeUnknown, cells),
    cells[i] the hole's thin cell as (i, top, k) in the block. Shared by rbx_terrain_voxels holes and the fill's dry run,
    so the two can never judge a cell differently."""
    def solid(i, j, k):
        return block["mat"][i][j][k] != "Air" and (block["occ"][i][j][k] or 0) > OCC_TOL
    def open_faces(i, j, k):
        """The faces of cell (i, j, k) whose neighbour in its own layer is empty and runs on along that face. A face at
        the region's edge cannot be judged: it is listed apart."""
        faces, unknown = [], []
        for side, (di, dk) in (("+x", (1, 0)), ("-x", (-1, 0)), ("+z", (0, 1)), ("-z", (0, -1))):
            ni, nk = i + di, k + dk
            if not (0 <= ni < n[0] and 0 <= nk < n[2]):
                unknown.append(side)
                continue
            if solid(ni, j, nk):
                continue
            along = [(ni + dk * s_, nk + di * s_) for s_ in (-1, 1)]
            inside = [(a, b) for a, b in along if 0 <= a < n[0] and 0 <= b < n[2]]
            if any(not solid(a, j, b) for a, b in inside):
                faces.append(side)
            elif len(inside) < 2:
                unknown.append(side)
        return faces, unknown
    holes, closed, columns, edge_unknown, cells = [], [], 0, 0, []
    for i in range(n[0]):
        for k in range(n[2]):
            top = next((j for j in range(n[1] - 1, -1, -1) if solid(i, j, k)), None)
            if top is None:
                continue
            columns += 1
            occ = block["occ"][i][top][k]
            if occ >= thin or top == 0 or solid(i, top - 1, k):
                continue
            below = next((j for j in range(top - 2, -1, -1) if solid(i, j, k)), None)
            y_thin = (c0[1] + top) * VOXEL
            h = {"column": [(c0[0] + i + 0.5) * VOXEL, (c0[2] + k + 0.5) * VOXEL], "thinCell": {
                "minY": y_thin, "material": block["mat"][i][top][k], "occupancy": round(occ, 4)},
                "emptyCellsUnder": top - 1 - (below if below is not None else -1)}
            faces, unknown = open_faces(i, top, k)
            if not faces:
                if unknown:
                    edge_unknown += 1
                    h["regionEdge"] = unknown
                closed.append(h)
                continue
            h["openFaces"] = faces
            h["bandStuds"] = edge_band(occ)
            if below is not None:
                gtop = (c0[1] + below) * VOXEL + VOXEL
                h.update(nextSolid={"minY": gtop - VOXEL, "material": block["mat"][i][below][k],
                                    "occupancy": round(block["occ"][i][below][k], 4)},
                         depth=y_thin - gtop)
            else:
                h.update(nextSolid=None, depth=None, note="no solid ground under it inside the region: read deeper")
            holes.append(h)
            cells.append((i, top, k))
    return holes, closed, columns, edge_unknown, cells


MAX_FLOAT_COLUMNS = 400                     # columns one floating scan reads one by one
MAX_FLOAT_RAYS = 100                        # drawn-surface rays a floating scan casts: 373 took 57 s live, near the
                                            # plugin's 450 requests a minute (RB3068)


def _solid_runs(col, gap_cells):
    """The solid runs of a column [(material, occupancy)], bottom first, as [j0, j1, {material: cells}] (j1 exclusive);
    runs apart by fewer than gap_cells empty cells are one run."""
    runs = []
    for j, (m, o) in enumerate(col):
        if m == "Air" or (o or 0) <= OCC_TOL:
            continue
        if runs and j - runs[-1][1] < gap_cells:
            runs[-1][1] = j + 1
        else:
            runs.append([j, j + 1, {}])
        runs[-1][2][m] = runs[-1][2].get(m, 0) + 1
    return runs


def _floating_in(col, y0, x, z, gap_cells, top_edge):
    """Every run of a column above an air gap over the ground (its lowest run): [finding], and whether it holds terrain."""
    runs = _solid_runs(col, gap_cells)
    found = []
    for below, r in zip(runs, runs[1:]):
        found.append({"column": [x, z], "bottomY": (y0 + r[0]) * VOXEL, "topY": (y0 + r[1]) * VOXEL,
                      "thickness": (r[1] - r[0]) * VOXEL, "gapStuds": (r[0] - below[1]) * VOXEL,
                      "groundTop": (y0 + runs[0][1]) * VOXEL, "material": max(r[2], key=r[2].get)})
        if top_edge and r[1] == len(col):
            found[-1]["topEdge"] = True
    return found, bool(runs)


def _pieces(found):
    """Floating columns that touch side by side (4-stud grid neighbours), grouped."""
    by = {}
    for f in found:
        by.setdefault((int(math.floor(f["column"][0] / VOXEL)), int(math.floor(f["column"][1] / VOXEL))), []).append(f)
    seen, out = set(), []
    for start in by:
        if start in seen:
            continue
        seen.add(start)
        todo, cells = [start], []
        while todo:
            c = todo.pop()
            cells.append(c)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (c[0] + d[0], c[1] + d[1])
                if nb in by and nb not in seen:
                    seen.add(nb)
                    todo.append(nb)
        fs = [f for c in cells for f in by[c]]
        mats = {}
        for f in fs:
            mats[f["material"]] = mats.get(f["material"], 0) + 1
        best = max(fs, key=lambda f: f["gapStuds"])
        out.append({"columns": len(cells), "at": best["column"], "maxGap": best["gapStuds"],
                    "minGap": min(f["gapStuds"] for f in fs), "bottomY": min(f["bottomY"] for f in fs),
                    "topY": max(f["topY"] for f in fs), "materials": mats,
                    "box": {"min": [min(c[0] for c in cells) * VOXEL, min(c[1] for c in cells) * VOXEL],
                            "max": [(max(c[0] for c in cells) + 1) * VOXEL, (max(c[1] for c in cells) + 1) * VOXEL]}})
    out.sort(key=lambda p: -p["columns"])
    return out


def _heightmap_at(hm):
    """rbx_terrain_voxels floating's heightmap {origin, heights} -> (lookup(x, z) -> written height or None, None) or
    (None, why)."""
    if not isinstance(hm, dict) or not _vec(hm.get("origin")) or any(float(x) % VOXEL for x in hm["origin"]):
        return None, "heightmap is {origin: [x, y, z] on the 4-stud grid, heights: rows (z) of studs (x)}"
    rows = hm.get("heights")
    if (not isinstance(rows, list) or not rows or not all(isinstance(r, list) and r for r in rows)
            or not all(v is None or _num(v) for r in rows for v in r)):
        return None, "heightmap heights is a list of rows (z) of heights in studs (x), null for no column"
    ox, oz = hm["origin"][0], hm["origin"][2]

    def at(x, z):
        i, k = int(math.floor((x - ox) / VOXEL)), int(math.floor((z - oz) / VOXEL))
        if 0 <= k < len(rows) and 0 <= i < len(rows[k]):
            v = rows[k][i]
            return None if v is None or v <= 0 else max(float(v), SURFACE_LIFT)
        return None
    return at, None


def _floating(region, columns, min_gap, heightmap, max_above):
    """rbx_terrain_voxels floating: solid terrain over an air gap over the ground, and with a heightmap a drawn surface
    above the height written (RB3065). Read only."""
    if not (_num(min_gap) and min_gap > 0) or not (_num(max_above) and max_above >= 0):
        return {"ok": False, "ran": False, "error": "min_gap is studs above 0; max_above studs, 0 or more"}
    c0, c1, why = _region_cells(region)
    if why:
        return {"ok": False, "ran": False, "error": why}
    if c1[1] - c0[1] < 3:
        return {"ok": False, "ran": False, "error": "floating terrain is ground, a gap and a run above it: the region "
                                                    "must be at least 12 studs tall"}
    lookup = None
    if heightmap is not None:
        lookup, why = _heightmap_at(heightmap)
        if why:
            return {"ok": False, "ran": False, "error": why}
    gap_cells = max(1, int(math.ceil(min_gap / VOXEL)))
    ny, calls, cols = c1[1] - c0[1], 0, []         # cols: (x, z, cells bottom first)
    if columns is not None:
        if (not isinstance(columns, list) or not columns or len(columns) > MAX_FLOAT_COLUMNS
                or not all(_vec(c, 2) for c in columns)):
            return {"ok": False, "ran": False, "error": "columns is a list of 1-%d [x, z] in studs" % MAX_FLOAT_COLUMNS}
        if ny > COLUMN_SPAN:
            return {"ok": False, "ran": False, "refusedRule": "budget", "error": "%d cells a column is over %d" % (ny, COLUMN_SPAN)}
        done = set()
        for x, z in columns:
            i, k = int(math.floor(x / VOXEL)), int(math.floor(z / VOXEL))
            if (i, k) in done:
                continue
            done.add((i, k))
            got, err = _read_column(i, k, c0[1], c1[1])
            if err:
                return err
            calls += got[1]
            cols.append(((i + 0.5) * VOXEL, (k + 0.5) * VOXEL, got[0]))
    else:
        if _count(c0, c1) > MAX_CELLS:
            return {"ok": False, "ran": False, "refusedRule": "budget",
                    "error": "%d cells is over %d a call: scan a smaller region, or give columns" % (_count(c0, c1), MAX_CELLS)}
        block, err = _read_block(c0, c1)
        if err:
            return err
        calls = block["calls"]
        for i in range(c1[0] - c0[0]):
            for k in range(c1[2] - c0[2]):
                cols.append(((c0[0] + i + 0.5) * VOXEL, (c0[2] + k + 0.5) * VOXEL,
                             [(block["mat"][i][j][k], block["occ"][i][j][k]) for j in range(ny)]))
    found, with_terrain, by_col = [], 0, {}
    for x, z, col in cols:
        f, any_solid = _floating_in(col, c0[1], x, z, gap_cells, True)
        with_terrain += any_solid
        found.extend(f)
        if f:
            by_col[(x, z)] = f
    above, cast, hits, in_map = [], 0, 0, 0
    if lookup is not None:
        want = [(x, z, lookup(x, z)) for x, z, _c in cols]
        want = [w_ for w_ in want if w_[2] is not None]
        in_map = len(want)
        stride = max(1, int(math.ceil(len(want) / MAX_FLOAT_RAYS)))
        base, top_y = heightmap["origin"][1], c1[1] * VOXEL + 16
        for x, z, h in want[::stride]:
            hit, err = _raycast_down(x, z, top_y, top_y - c0[1] * VOXEL + 16)
            cast += 1
            if err:
                return dict(err, ok=False, error="floating: the drawn-surface Raycast failed: %s" % err.get("error"))
            if not hit.get("hit"):
                continue
            hits += 1
            drawn = hit["position"][1] - base
            if drawn - h > max_above:
                a = {"column": [x, z], "written": round(h, 3), "drawn": round(drawn, 3), "above": round(drawn - h, 3),
                     "material": hit.get("material")}
                under = by_col.get((x, z))
                if under:
                    a["gapStuds"] = max(f["gapStuds"] for f in under)
                above.append(a)
    out = {"region": _studs(c0, c1), "columns": len(cols), "withTerrain": with_terrain, "minGap": gap_cells * VOXEL,
           "calls": calls, "floatingColumns": len(by_col), "floatingRuns": len(found), "floating": found[:2000]}
    if found:
        ps = _pieces(found)
        out.update(pieces=ps[:200], pieceCount=len(ps))
    if lookup is not None:
        out.update(maxAbove=max_above, heightmapColumns=in_map, rays=cast, rayHits=hits, aboveCount=len(above),
                   above=sorted(above, key=lambda a: -a["above"])[:2000])
    if not with_terrain and not hits:
        return dict(out, ok=False, error="no terrain in the columns scanned: nothing was read, so nothing floating is "
                                         "not a clean answer")
    if lookup is not None and not in_map:
        return dict(out, ok=False, error="the heightmap covers none of the columns scanned: no drawn surface was compared")
    out.update(ok=True, clean=not found and not above,
               rule="in each column the lowest solid run is the ground; every run above an air gap of at least %g studs "
                    "is floating (solid tops; Studio draws each about 2 studs higher, RB1151)%s" % (
                        gap_cells * VOXEL, "; a column drawn more than %g studs above its written height is above "
                        "(a terrain-only ray down from the region's top)" % max_above if lookup is not None else ""))
    return out


# ---------------------------------------------------------------- heightmaps
def _heights_from(source, heights, width, height_scale, up, obj_scale):
    """A 2D list [z][x] of heights in studs (None where a mesh covers nothing), and what it came from."""
    if heights is not None:
        # null is "no column", as 0 is
        if not isinstance(heights, list) or not heights or not all(isinstance(r, list) and len(r) == len(heights[0])
                                                                 and all(v is None or _num(v) for v in r) for r in heights):
            return None, "heights is a rectangular list of rows (z) of numbers (x), null for no column"
        k = 1.0 if height_scale is None else height_scale
        return [[None if v is None else v * k for v in r] for r in heights], None
    if not isinstance(source, str) or not os.path.isfile(source):
        return None, "source is a .png, .raw/.r16 or .obj file on this machine, or give heights"
    ext = os.path.splitext(source)[1].lower()
    if ext in (".png", ".raw", ".r16") and not (_num(height_scale) and height_scale > 0):
        return None, "a PNG or RAW heightmap needs height_scale: the studs its brightest value stands for"
    try:
        data = open(source, "rb").read()
    except OSError as exc:
        return None, "the file could not be read: %s" % exc
    if ext == ".png":
        from mif_roblox import png as P
        try:
            w, h, rgba = P.decode_rgba(data)
        except ValueError as exc:
            return None, "the PNG could not be read: %s" % exc
        return [[rgba[(z * w + x) * 4] / 255.0 * height_scale for x in range(w)] for z in range(h)], None
    if ext in (".raw", ".r16"):
        if len(data) % 2:
            return None, "a RAW heightmap is 16-bit little-endian samples: an odd byte count"
        count = len(data) // 2
        w = width or int(round(math.sqrt(count)))
        if not w or count % w:
            return None, "%d samples do not make rows of %s; give width" % (count, w)
        vals = struct.unpack("<%dH" % count, data)
        return [[vals[z * w + x] / 65535.0 * height_scale for x in range(w)] for z in range(count // w)], None
    if ext == ".obj":
        return _obj_heights(data, up, obj_scale)
    return None, "source must be .png, .raw, .r16 or .obj"


def _obj_heights(data, up, scale):
    """A Blender OBJ (Y up by default, as Blender's exporter writes it; up='z' for Z-up files) rasterised to one height
    per 4-stud cell: each triangle's height at the cell centres it covers, the highest kept."""
    verts, faces = [], []
    for line in data.decode("utf-8", "replace").splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "v" and len(p) >= 4:
            x, y, z = (float(p[1]) * scale, float(p[2]) * scale, float(p[3]) * scale)
            verts.append((x, y, z) if up == "y" else (x, z, -y))
        elif p[0] == "f" and len(p) >= 4:
            idx = []
            for t in p[1:]:
                i = int(t.split("/")[0])
                idx.append(i - 1 if i > 0 else len(verts) + i)
            for j in range(1, len(idx) - 1):
                faces.append((idx[0], idx[j], idx[j + 1]))
    if not verts or not faces:
        return None, "the OBJ has no faces"
    xs, ys, zs = [v[0] for v in verts], [v[1] for v in verts], [v[2] for v in verts]
    x0, z0, y0 = min(xs), min(zs), min(ys)
    nx, nz = int(math.ceil((max(xs) - x0) / VOXEL)) or 1, int(math.ceil((max(zs) - z0) / VOXEL)) or 1
    if nx * nz > 512 * 512:
        return None, "the mesh spans %d x %d cells; at most 512 x 512" % (nx, nz)
    grid = [[None] * nx for _ in range(nz)]
    for a, b, c in faces:
        (ax, ay, az), (bx, by, bz), (cx, cy, cz) = verts[a], verts[b], verts[c]
        den = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz)
        if abs(den) < 1e-12:
            continue
        i0 = max(0, int((min(ax, bx, cx) - x0) / VOXEL))
        i1 = min(nx - 1, int((max(ax, bx, cx) - x0) / VOXEL))
        k0 = max(0, int((min(az, bz, cz) - z0) / VOXEL))
        k1 = min(nz - 1, int((max(az, bz, cz) - z0) / VOXEL))
        for k in range(k0, k1 + 1):
            pz = z0 + (k + 0.5) * VOXEL
            for i in range(i0, i1 + 1):
                px = x0 + (i + 0.5) * VOXEL
                w1 = ((bz - cz) * (px - cx) + (cx - bx) * (pz - cz)) / den
                w2 = ((cz - az) * (px - cx) + (ax - cx) * (pz - cz)) / den
                w3 = 1 - w1 - w2
                if min(w1, w2, w3) < -1e-9:
                    continue
                hgt = w1 * ay + w2 * by + w3 * cy - y0
                if grid[k][i] is None or hgt > grid[k][i]:
                    grid[k][i] = hgt
    return grid, None


def _resample(grid, nx, nz):
    """Bilinear resample of [z][x] to nz rows of nx; None cells stay None at their nearest sample."""
    h, w = len(grid), len(grid[0])
    if (w, h) == (nx, nz):
        return grid
    out = []
    for k in range(nz):
        fz = (k + 0.5) * h / nz - 0.5
        z0 = min(h - 1, max(0, int(math.floor(fz))))
        z1, tz = min(h - 1, z0 + 1), min(1.0, max(0.0, fz - z0))
        row = []
        for i in range(nx):
            fx = (i + 0.5) * w / nx - 0.5
            x0 = min(w - 1, max(0, int(math.floor(fx))))
            x1, tx = min(w - 1, x0 + 1), min(1.0, max(0.0, fx - x0))
            q = [grid[z0][x0], grid[z0][x1], grid[z1][x0], grid[z1][x1]]
            if any(v is None for v in q):
                row.append(grid[int(round(z0 + tz))][int(round(x0 + tx))])
            else:
                row.append((q[0] * (1 - tx) + q[1] * tx) * (1 - tz) + (q[2] * (1 - tx) + q[3] * tx) * tz)
        out.append(row)
    return out


def _slope_deg(hm, i, k):
    def at(ii, kk):
        ii, kk = min(len(hm[0]) - 1, max(0, ii)), min(len(hm) - 1, max(0, kk))
        v = hm[kk][ii]
        return hm[k][i] if v is None else v
    dx = (at(i + 1, k) - at(i - 1, k)) / (2 * VOXEL)
    dz = (at(i, k + 1) - at(i, k - 1)) / (2 * VOXEL)
    return math.degrees(math.atan(math.hypot(dx, dz)))


def _paint(rules, default, h, slope):
    for r in rules or []:
        if ("min_height" in r and h < r["min_height"]) or ("max_height" in r and h > r["max_height"]):
            continue
        if ("min_slope" in r and slope < r["min_slope"]) or ("max_slope" in r and slope > r["max_slope"]):
            continue
        return r["material"]
    return default


def _raycast_down(x, z, top, depth):
    params = {"$t": "RaycastParams", "filterType": "Include", "filter": [{"$t": "Instance", "path": TERRAIN}],
              "ignoreWater": True}
    r = _call("Raycast", [_w3([x, top, z]), _w3([0, -depth, 0]), params], True, target=["Workspace"], cls="Workspace")
    if not r.get("ok"):
        return None, r
    hit = (r.get("returns") or [None])[0]
    if not isinstance(hit, dict) or hit.get("$t") != "RaycastResult":
        return {"hit": False}, None
    pos, nor = V.decode(hit.get("position")) or {}, V.decode(hit.get("normal")) or {}
    return {"hit": True, "position": [pos.get("x"), pos.get("y"), pos.get("z")], "normal": [nor.get("x"), nor.get("y"), nor.get("z")],
            "material": _unwire(hit.get("material")), "instance": V.decode(hit.get("instance"))}, None


# ---------------------------------------------------------------- air above a heightmap tile (RB3065)
# One FillRegion carries at most 2^26 cells, whatever its shape: 64 x 16384 x 64, 128 x 8192 x 64, 8192 x 1 x 8192 and
# 256 x 1024 x 256 ran, one cell more on any axis answered "Region is too large"; an Air one over empty space or a solid
# cube answered in about 0.03 s (0.748, AIR_FILL_RB).
AIR_FILL_MAX_CELLS = 1 << 26
AIR_FILL_TIMEOUT = 60
AIR_FILL_RB = "RB3067"
AIR_SAMPLE_COLUMNS = 3                      # per axis: a 3 x 3 grid of the footprint's columns is read back
COLUMN_SPAN = 1 << 20                       # cells one compact column read carries (the plugin takes 2,985,984, RB809)


def _boxes(c0, c1, limit):
    """Cell boxes covering [c0, c1), each at most limit cells: the footprint halved until it fits, then layers of y."""
    n = [c1[i] - c0[i] for i in range(3)]
    sx, sz = n[0], n[2]
    while sx * sz > limit:
        if sx >= sz:
            sx = (sx + 1) // 2
        else:
            sz = (sz + 1) // 2
    sy = max(1, min(n[1], limit // (sx * sz)))
    for x in range(c0[0], c1[0], sx):
        for z in range(c0[2], c1[2], sz):
            for y in range(c0[1], c1[1], sy):
                yield [x, y, z], [min(x + sx, c1[0]), min(y + sy, c1[1]), min(z + sz, c1[2])]


def _read_column(i, k, y0, y1):
    """Cells (i, y0..y1-1, k) as [(material, occupancy)], bottom first: compact reads of COLUMN_SPAN cells, or the generic
    chunked read on a plugin without them. ((cells, calls), None) or (None, failure)."""
    out, calls = [], 0
    y = y0
    while y < y1 and _COMPACT["ok"] is not False:
        b = min(y1, y + COLUMN_SPAN)
        grids, err = _compact_read([i, y, k], [i + 1, b, k + 1])
        if err:
            return None, dict(err, ok=False, column=[(i + 0.5) * VOXEL, (k + 0.5) * VOXEL])
        if grids is None:
            break                                # no voxels_read on this plugin: the generic route below
        calls += 1
        out.extend((grids[0][0][j][0], grids[1][0][j][0]) for j in range(b - y))
        y = b
    if y < y1:
        block, err = _read_block([i, y, k], [i + 1, y1, k + 1])
        if err:
            return None, dict(err, ok=False, column=[(i + 0.5) * VOXEL, (k + 0.5) * VOXEL])
        calls += block["calls"]
        out.extend((block["mat"][0][j][0], block["occ"][0][j][0]) for j in range(y1 - y))
    return (out, calls), None


def _air_above_box(c0, c1, air_above):
    """The cells air_above empties over the block [c0, c1): its footprint from the block's top to the ceiling.
    ({c0, c1, ...}, None); c1 None when the ceiling is at or under the block's top (nothing to clear)."""
    ext, err = _extents()
    if err:
        return None, dict(err, ok=False)
    top_c = ext[1][1] + 1                       # MaxExtents' max is inclusive
    clamped = False
    if air_above is True:
        ceil_c = top_c
    else:
        ceil_c = int(math.ceil(float(air_above) / VOXEL))
        if ceil_c > top_c:
            ceil_c, clamped = top_c, True
    box = {"c0": [c0[0], c1[1], c0[2]], "c1": [c1[0], ceil_c, c1[2]], "ceiling": ceil_c * VOXEL,
           "asked": "maxExtents" if air_above is True else air_above, "clamped": clamped}
    if ceil_c <= c1[1]:
        box["c1"] = None
        box["reply"] = {"ceiling": ceil_c * VOXEL, "asked": box["asked"], "cells": 0, "calls": 0, "verified": True,
                        "note": "the ceiling is at or under the block's top (%g): nothing above it to clear" % (c1[1] * VOXEL)}
    return box, None


@T.ends_group_on_raise
def _write_and_clear(c0, mat, occ, clear):
    """The heightmap's block, then Air FillRegions over clear's cells, in one undo group; then the cleared cells read on a
    grid of sample columns. (write reply, airAbove reply). A failed fill cancels the group (Studio reverts both)."""
    grouped = T.begin_group("terrain heightmap").get("ok") is True
    w = _write_block(c0, mat, occ, label="terrain heightmap")    # joins the group open here: it begins none of its own
    if not w.get("ok"):
        cancelled = bool(T.rbx_undo_group("cancel").get("ok")) if grouped else False
        return dict(w, undoGroup=grouped, cancelled=cancelled), None
    a0, a1 = clear["c0"], clear["c1"]
    calls, t0, cells = 0, time.monotonic(), _count(a0, a1)
    for a, b in _boxes(a0, a1, AIR_FILL_MAX_CELLS):
        r = _call("FillRegion", [_wregion(a, b), VOXEL, _wmat("Air")], False, timeout=AIR_FILL_TIMEOUT)
        calls += 1
        if not r.get("ok"):
            cancelled = bool(T.rbx_undo_group("cancel").get("ok")) if grouped else False
            tail = ("; the undo group was cancelled, so the tile is not written either" if cancelled else
                    "; the tile and %d earlier fill(s) stay written (%s)" % (
                        calls - 1, "the cancel was refused" if grouped else "no undo group could be opened"))
            return dict(r, ok=False, chunk=_studs(a, b), airCalls=calls, undoGroup=grouped, cancelled=cancelled,
                        error="air_above: FillRegion Air over %s: %s%s" % (_studs(a, b), r.get("error"), tail)), None
    ms = int((time.monotonic() - t0) * 1000)
    if grouped:
        T.rbx_undo_group("commit")
    w = dict(w, undoGroup=grouped)
    n = [a1[0] - a0[0], a1[2] - a0[2]]
    picks = [sorted({min(n[d] - 1, (2 * s + 1) * n[d] // (2 * AIR_SAMPLE_COLUMNS)) for s in range(AIR_SAMPLE_COLUMNS)})
             for d in (0, 1)]
    read, solid, rcalls, cells_read = 0, [], 0, 0
    for i in picks[0]:
        for k in picks[1]:
            got, err = _read_column(a0[0] + i, a0[2] + k, a0[1], a1[1])
            if err:
                return w, {"ceiling": clear["ceiling"], "region": _studs(a0, a1), "cells": cells, "calls": calls, "ms": ms,
                           "verified": False, "error": "air_above: cleared, but the read-back failed: %s" % err.get("error")}
            col, c = got
            rcalls += c
            read += 1
            cells_read += len(col)
            for j, (m, o) in enumerate(col):
                if m != "Air" and (o or 0) > OCC_TOL:
                    solid.append({"column": [(a0[0] + i + 0.5) * VOXEL, (a0[2] + k + 0.5) * VOXEL],
                                  "minY": (a0[1] + j) * VOXEL, "material": m, "occupancy": round(o, 4)})
    out = {"ceiling": clear["ceiling"], "asked": clear["asked"], "region": _studs(a0, a1), "cells": cells, "calls": calls,
           "maxCellsPerCall": AIR_FILL_MAX_CELLS, "limitRow": AIR_FILL_RB, "ms": ms, "columnsRead": read,
           "cellsRead": cells_read, "readCalls": rcalls, "solidLeft": len(solid), "verified": not solid and read > 0}
    if clear["clamped"]:
        out["note"] = "the ceiling asked is above Terrain.MaxExtents: cleared to its top, %g" % clear["ceiling"]
    if solid:
        out.update(firstSolid=solid[:20], error="air_above: %d solid cell(s) left in the cleared region on %d sample "
                                                 "column(s)" % (len(solid), read))
    elif not read:
        out["error"] = "air_above: no sample column was read: nothing verified"
    return w, out


@T._reports
def rbx_terrain_heightmap(origin: list, source: str = None, heights: list = None, height_scale: float = None,
                          size: list = None, width: int = None, material: str = "Grass", paint: list = None,
                          up: str = "y", obj_scale: float = 1.0, verify_points: int = 64, air_above: Any = None) -> dict:
    "Roblox Studio terrain from heights: a PNG (red channel, 0-255 times height_scale), a 16-bit little-endian RAW or .r16 (0-65535 times height_scale; width when not square), an OBJ exported from Blender (Y up by default; heights in studs times obj_scale) or heights, a list of rows (z) of studs (x), null (like 0 or below) for a column left as it is - a round island's grid past its edge. origin [x, y, z] is the minimum corner on the 4-stud grid; one sample per 4-stud column, or size [x, z] in studs to resample to. Each column is written by WriteVoxels so the surface smooth terrain draws stands at its height: the voxels hold h - 2 studs, because Studio draws a surface half a cell (2 studs) above the solid it is given; a height between 0 and 2 is drawn at 2, the lowest a column on that origin can stand (lowered, counted). material, or paint rules [{material, min_height, max_height, min_slope, max_slope}] in studs and degrees, first match wins. Read back: heights sampled by WorldRoot:Raycast against Terrain only (a part over the ground is never hit) at up to verify_points columns, each within 1 stud of the range of its 3x3 neighbourhood (on flat ground, its own height; at a cliff smooth terrain blends a column toward its neighbours, and byNeighbourhood counts those); a neighbour past the block's edge is the terrain already standing there, read by a ray too (besideRead, besideTerrain), so a tile written beside an earlier one is judged against it. The reply says surface \"drawn\" and surfaceConvention 2: the drawn surface is the height given (convention 1, before the fix, wrote solid to h and drew it 2 studs high), so a caller never needs its own +2. The block written is the tile's footprint from origin up to its own highest column, so terrain standing above that (an older, taller build's hills) survives as arches, slabs and floating discs over the new ground (RB3065): air_above true also empties the footprint from the block's top up to the top of Terrain.MaxExtents, or a number up to that world height in studs (aligned up to the 4-stud grid, held inside MaxExtents), by Terrain:FillRegion with Air in as few calls as the measured per-call limit allows (airAbove.calls), in the same undo step as the write; read back by Terrain:ReadVoxels on up to 9 sample columns of the footprint over the whole cleared height, every cell of which must be empty (airAbove.columnsRead, solidLeft). The cleared region is gated like the write, so in scratch-box mode give a ceiling inside the box. rbx_terrain_voxels floating with this heightmap finds what is left standing above it."
    if not _vec(origin) or any(float(x) % VOXEL for x in origin):
        return {"ok": False, "ran": False, "error": "origin is [x, y, z] in studs on the 4-stud grid"}
    if air_above is not None and not isinstance(air_above, bool) and not _num(air_above):
        return {"ok": False, "ran": False, "error": "air_above is true (clear to the top of Terrain.MaxExtents), a world "
                                                    "height in studs to clear up to, or false"}
    if up not in ("y", "z") or not (_num(obj_scale) and obj_scale > 0):
        return {"ok": False, "ran": False, "error": "up is y or z; obj_scale a number above 0"}
    bad = _valid_material(material)
    if bad:
        return bad
    for r in paint or []:
        if not isinstance(r, dict) or set(r) - {"material", "min_height", "max_height", "min_slope", "max_slope"}:
            return {"ok": False, "ran": False, "error": "a paint rule is {material, min_height, max_height, min_slope, max_slope}"}
        bad = _valid_material(r.get("material"))
        if bad:
            return bad
    hm, why = _heights_from(source, heights, width, height_scale, up, obj_scale)
    if why:
        return {"ok": False, "ran": False, "error": why}
    if size is not None:
        if not _vec(size, 2) or min(size) < VOXEL:
            return {"ok": False, "ran": False, "error": "size is [x, z] in studs, at least 4 each"}
        hm = _resample(hm, int(round(size[0] / VOXEL)), int(round(size[1] / VOXEL)))
    nz, nx = len(hm), len(hm[0])
    top = max((v for r in hm for v in r if v is not None), default=0)
    if top <= 0:
        return {"ok": False, "ran": False, "error": "every height is 0 or below: nothing to build"}
    ny = max(1, int(math.ceil((top - SURFACE_LIFT) / VOXEL)))
    if nx * ny * nz > MAX_CELLS:
        return {"ok": False, "ran": False, "refusedRule": "budget",
                "error": "%d x %d x %d cells is over %d a call: resample with size" % (nx, ny, nz, MAX_CELLS)}
    c0 = [int(x) // VOXEL for x in origin]
    c1 = [c0[0] + nx, c0[1] + ny, c0[2] + nz]
    gate = _terrain_gate(c0, c1, "rbx_terrain_heightmap")
    if gate:
        return gate
    clear = None
    if air_above is not None and air_above is not False:
        clear, why = _air_above_box(c0, c1, air_above)
        if why:
            return why
        if clear["c1"] is not None:
            gate = _terrain_gate(clear["c0"], clear["c1"], "rbx_terrain_heightmap air_above")
            if gate:
                return gate
    mat, occ = _blank([nx, ny, nz], "Air"), _blank([nx, ny, nz], 0.0)
    painted, lowered = {}, 0
    for k in range(nz):
        for i in range(nx):
            h = hm[k][i]
            if h is None or h <= 0:
                continue
            m = _paint(paint, material, h, _slope_deg(hm, i, k))
            painted[m] = painted.get(m, 0) + 1
            solid = h - SURFACE_LIFT
            if solid < MIN_OCC * VOXEL:
                solid, lowered = MIN_OCC * VOXEL, lowered + 1
            for j in range(ny):
                o = min(1.0, max(0.0, (solid - j * VOXEL) / VOXEL))
                if o >= MIN_OCC:
                    mat[i][j][k], occ[i][j][k] = m, round(o, 4)
    if clear and clear["c1"] is not None:
        w, cleared = _write_and_clear(c0, mat, occ, clear)
    else:
        w, cleared = _write_block(c0, mat, occ, label="terrain heightmap"), (clear and clear["reply"])
    if not w.get("ok"):
        return w
    # Sample columns on an even grid, skipping the edges where a ray may graze the side.
    want = max(1, min(int(verify_points or 0), nx * nz, 400))
    step = max(1, int(math.sqrt(nx * nz / want)))
    samples, bad, by_nb = [], [], 0
    base = origin[1]

    beside = {}                 # a column just outside the block: the terrain already there, read once (None: none)

    def drawn(ii, kk):          # the height a column is drawn at: its own, or the lift when it is lower; None when empty
        if not (0 <= ii < nx and 0 <= kk < nz):
            # Smooth terrain blends a column toward its neighbours, and those past the block's edge are whatever terrain
            # stands there (an island's other quadrant, ground written before): read as they are, they bound the edge
            # column too
            if (ii, kk) not in beside:
                bx, bz = origin[0] + (ii + 0.5) * VOXEL, origin[2] + (kk + 0.5) * VOXEL
                hb, eb = _raycast_down(bx, bz, base + top + 512, top + 1024)
                beside[(ii, kk)] = (hb["position"][1] - base) if not eb and hb.get("hit") else None
            return beside[(ii, kk)]
        v = hm[kk][ii]
        return None if v is None or v <= 0 else max(v, SURFACE_LIFT)
    for k in range(step // 2, nz, step):
        for i in range(step // 2, nx, step):
            h = drawn(i, k)
            if h is None:
                continue
            x, z = origin[0] + (i + 0.5) * VOXEL, origin[2] + (k + 0.5) * VOXEL
            hit, err = _raycast_down(x, z, base + top + 64, top + 128)
            if err:
                return dict(err, ok=False, written=w, error="written, but the Raycast read-back failed: %s" % err.get("error"))
            got = hit["position"][1] - base if hit.get("hit") else None
            near = [d for d in (drawn(i + a, k + b) for a in (-1, 0, 1) for b in (-1, 0, 1)) if d is not None]
            lo, hi = min(near + [0.0] if len(near) < 9 else near), max(near)   # an empty neighbour pulls toward the base
            s = {"x": x, "z": z, "expected": round(h, 3), "sampled": None if got is None else round(got, 3),
                 "neighbourhood": [round(lo, 3), round(hi, 3)]}
            samples.append(s)
            if got is None:
                s["outside"] = None
                bad.append(s)
                continue
            s["outside"] = round(max(0.0, lo - got, got - hi), 3)
            if s["outside"] > HEIGHT_TOL:
                bad.append(s)
            elif abs(got - h) > HEIGHT_TOL:
                by_nb += 1
    out = {"ok": not bad and bool(samples), "verified": not bad and bool(samples), "region": _studs(c0, c1),
           "columns": [nx, nz], "maxHeight": round(top, 3), "painted": painted, "writeCalls": w["calls"],
           "undoGroup": w["undoGroup"], "samples": len(samples), "tolerance": HEIGHT_TOL, "byNeighbourhood": by_nb,
           "besideRead": len(beside), "besideTerrain": sum(1 for v in beside.values() if v is not None),
           "surfaceLift": SURFACE_LIFT, "surface": "drawn", "surfaceConvention": SURFACE_CONVENTION,
           "worst": max((abs((s["sampled"] or 0) - s["expected"]) for s in samples), default=None),
           "worstOutside": max((s["outside"] for s in samples if s["outside"] is not None), default=None)}
    if lowered:
        out["lowered"] = lowered
        out["note"] = ("%d column(s) between 0 and %g studs are drawn at %g, the lowest smooth terrain draws a column "
                       "standing on the origin's cell: lower origin by 4 to reach them" % (lowered, SURFACE_LIFT, SURFACE_LIFT))
    if bad:
        miss = [s["outside"] for s in bad if s["outside"] is not None]
        out.update(offBy=bad[:20], error="%d of %d sampled heights lie more than %g stud outside the range of their 3x3 "
                   "neighbourhood (worst %s outside%s)" % (len(bad), len(samples), HEIGHT_TOL,
                                                           max(miss) if miss else "n/a",
                                                           "; %d hit nothing" % (len(bad) - len(miss)) if len(miss) < len(bad) else ""))
    elif not samples:
        out["error"] = "no column was tall enough to sample: nothing verified"
    if cleared:
        out["airAbove"] = cleared
        if cleared.get("verified") is False:
            out.update(ok=False, verified=False)
            out["error"] = "; ".join(e for e in (out.get("error"), cleared.get("error")) if e)
    return out


# ---------------------------------------------------------------- materials, copy, smooth, sample
@T._reports
def rbx_terrain_material(region: dict = None, replace: dict = None, colors: dict = None, slots: list = None,
                         water: dict = None, decoration: bool = None, grass_length: float = None) -> dict:
    "Roblox Studio terrain materials and looks, each part read back. replace {from, to} inside region (Terrain:ReplaceMaterial; a voxel read before and after: no cell of from left, to gained them); colors {Material: [r, g, b] 0-1} (SetMaterialColor, read by GetMaterialColor to 1/255; Air refused); slots [{index, base_material, variant, color}] (SetMaterialSlot, read by GetMaterialSlot); water {WaterColor, WaterReflectance, WaterTransparency, WaterWaveSize, WaterWaveSpeed}; decoration (Terrain.Decoration); grass_length (Terrain.GrassLength). Only replace has a region, so only it can sit in the scratch box."
    if not any(x is not None for x in (replace, colors, slots, water, decoration, grass_length)):
        return {"ok": False, "ran": False, "error": "give at least one of replace, colors, slots, water, decoration, grass_length"}
    out, problems = {}, []
    if replace is not None:
        if not isinstance(replace, dict) or set(replace) != {"from", "to"}:
            return {"ok": False, "ran": False, "error": "replace is {from: Material, to: Material}"}
        for m in (replace["from"], replace["to"]):
            bad = _valid_material(m)
            if bad:
                return bad
        c0, c1, why = _region_cells(region)
        if why:
            return {"ok": False, "ran": False, "error": "replace needs region: " + why}
        if _count(c0, c1) > MAX_CELLS:
            return {"ok": False, "ran": False, "refusedRule": "budget", "error": "region over %d cells" % MAX_CELLS}
    if colors is not None:
        if not isinstance(colors, dict) or not colors:
            return {"ok": False, "ran": False, "error": "colors is {Material: [r, g, b]}"}
        for k, v in colors.items():
            bad = _valid_material(k, allow_air=False)
            if bad:
                return bad
            if not _vec(v) or not all(0 <= x <= 1 for x in v):
                return {"ok": False, "ran": False, "refusedRule": "type", "error": "%s's color is [r, g, b], each 0-1" % k}
    if slots is not None:
        if not isinstance(slots, list) or not slots:
            return {"ok": False, "ran": False, "error": "slots is a list of {index, base_material, variant, color}"}
        for s in slots:
            if not isinstance(s, dict) or not isinstance(s.get("index"), int) or not isinstance(s.get("variant", ""), str) \
                    or not _vec(s.get("color", [1, 1, 1])):
                return {"ok": False, "ran": False, "error": "a slot is {index: int, base_material, variant: name, color: [r,g,b]}"}
            # Measured on 0.748 (RB305): slots 0 and 1 are Air and Water and refuse a set; 64 and up are out of range.
            if not SLOT_MIN <= s["index"] <= SLOT_MAX:
                return {"ok": False, "ran": False, "refusedRule": "type",
                        "error": "slot index is %d-%d: 0 and 1 are Air and Water, which Studio will not set" % (SLOT_MIN, SLOT_MAX)}
            bad = _valid_material(s.get("base_material"), allow_air=False)
            if bad:
                return bad
    props = dict(water or {})
    if set(props) - {"WaterColor", "WaterReflectance", "WaterTransparency", "WaterWaveSize", "WaterWaveSpeed"}:
        return {"ok": False, "ran": False, "error": "water takes WaterColor, WaterReflectance, WaterTransparency, WaterWaveSize, WaterWaveSpeed"}
    if decoration is not None:
        props["Decoration"] = decoration
    if grass_length is not None:
        props["GrassLength"] = grass_length
    regionless = colors is not None or slots is not None or bool(props)
    gate = _terrain_gate(None, None, "rbx_terrain_material") if regionless else None
    if gate:
        return gate
    if replace is not None:
        gate = _terrain_gate(c0, c1, "rbx_terrain_material replace")
        if gate:
            return gate
        before, err = _read_block(c0, c1)
        if err:
            return err
        r = _call("ReplaceMaterial", [_wregion(c0, c1), VOXEL, _wmat(replace["from"]), _wmat(replace["to"])], False)
        if not r.get("ok"):
            return r
        after, err = _read_block(c0, c1)
        if err:
            return err
        sb, sa = _summary(before)["materials"], _summary(after)["materials"]
        left = sa.get(replace["from"], 0) if replace["from"] != replace["to"] else 0
        gained = sa.get(replace["to"], 0) - sb.get(replace["to"], 0)
        want = sb.get(replace["from"], 0) if replace["from"] != replace["to"] else 0
        out["replace"] = {"region": _studs(c0, c1), "before": sb, "after": sa, "replaced": gained}
        if left or gained != want:
            problems.append("replace: %d cells of %s left, %d of %s gained, %d expected" % (left, replace["from"], gained,
                                                                                           replace["to"], want))
    for k, v in (colors or {}).items():
        r = _call("SetMaterialColor", [_wmat(k), {"$t": "Color3", "v": list(v)}], False)
        g = _call("GetMaterialColor", [_wmat(k)], True) if r.get("ok") else r
        got = (g.get("returns") or [None])[0] if g.get("ok") else None
        same = isinstance(got, dict) and V.same({"$t": "Color3", "v": list(v)}, got)
        out.setdefault("colors", {})[k] = {"sent": list(v), "readBack": V.decode(got), "ok": same}
        if not same:
            problems.append("color %s: %s" % (k, g.get("error") or "reads back %s" % V.decode(got)))
    for s in slots or []:
        color = list(s.get("color", [1, 1, 1]))
        r = _call("SetMaterialSlot", [s["index"], _wmat(s["base_material"]), s.get("variant", ""),
                                      {"$t": "Color3", "v": color}], False)
        g = _call("GetMaterialSlot", [s["index"]], True) if r.get("ok") else r
        rets = (g.get("returns") or []) if g.get("ok") else []
        back = [_unwire(x) for x in rets] if g.get("ok") else None
        # GetMaterialSlot's Tuple, measured on 0.748: (Material, variant name, Color3).
        same = (len(rets) == 3 and back[0] == s["base_material"] and back[1] == s.get("variant", "")
                and V.same({"$t": "Color3", "v": color}, rets[2]))
        out.setdefault("slots", []).append({"index": s["index"], "readBack": back, "ok": same})
        if not same:
            problems.append("slot %s: %s" % (s["index"], g.get("error") or "GetMaterialSlot answered %s" % back))
    if props:
        r = T.rbx_set_props(TERRAIN, props)
        out["props"] = {k: r.get(k) for k in ("ok", "verified", "after", "mismatch", "error", "refused")}
        if not r.get("ok"):
            problems.append("properties: %s" % r.get("error"))
    out.update(ok=not problems, verified=not problems)
    if problems:
        out["error"] = "; ".join(problems)
    return out


@T._reports
def rbx_terrain_copy(region: dict, to: list, paste_empty: bool = True) -> dict:
    "Roblox Studio terrain: copy a region (Terrain:CopyRegion, a TerrainRegion held by the plugin, its SizeInCells read) and paste it with its minimum corner at to [x, y, z] studs on the 4-stud grid (Terrain:PasteRegion). paste_empty false keeps what is at the destination where the copy is empty. Read back: the pasted cells equal the source's. Only the destination is a write."
    c0, c1, why = _region_cells(region)
    if why:
        return {"ok": False, "ran": False, "error": why}
    if not _vec(to) or any(float(x) % VOXEL for x in to):
        return {"ok": False, "ran": False, "error": "to is [x, y, z] in studs on the 4-stud grid"}
    n = [c1[i] - c0[i] for i in range(3)]
    if _count(c0, c1) > MAX_RETURN_CELLS:
        return {"ok": False, "ran": False, "refusedRule": "budget",
                "error": "a copy is read back cell by cell: %d cells is over %d" % (_count(c0, c1), MAX_RETURN_CELLS)}
    d0 = [int(x) // VOXEL for x in to]
    d1 = [d0[i] + n[i] for i in range(3)]
    gate = _terrain_gate(d0, d1, "rbx_terrain_copy paste")
    if gate:
        return gate
    src, err = _read_block(c0, c1)
    if err:
        return err
    dst_before = None
    if not paste_empty:
        dst_before, err = _read_block(d0, d1)
        if err:
            return err
    # Region3int16's max is the last cell, inclusive, as Terrain.MaxExtents reads; SizeInCells below proves it.
    cp = _call("CopyRegion", [{"$t": "Region3int16", "v": c0 + [c - 1 for c in c1]}], True)
    if not cp.get("ok"):
        return cp
    h = (cp.get("returns") or [None])[0]
    handle = h.get("handle") if isinstance(h, dict) else None
    if not handle:
        return {"ok": False, "error": "CopyRegion returned %r, not an object handle" % (h,)}
    size = T._rbx("get_props", target={"handle": handle}, props=["SizeInCells"])
    sic = V.decode(T._obj(size.get("values")).get("SizeInCells")) if size.get("ok") else None
    got_n = [sic.get("x"), sic.get("y"), sic.get("z")] if isinstance(sic, dict) else None
    problems = []
    if got_n != n:
        problems.append("the copy's SizeInCells is %s, the region is %s cells" % (got_n, n))
    r = _call("PasteRegion", [{"$t": "Object", "handle": handle}, _w3(d0, "Vector3int16"), bool(paste_empty)], False)
    T._rbx("release_handle", handle=handle)
    if not r.get("ok"):
        return dict(r, sizeInCells=got_n)
    got, err = _read_block(d0, d1)
    if err:
        return dict(err, ok=False, error="pasted, but the read-back failed: %s" % err.get("error"))
    want = src
    if not paste_empty:
        want = {"mat": [[[src["mat"][i][j][k] if src["mat"][i][j][k] != "Air" else dst_before["mat"][i][j][k]
                          for k in range(n[2])] for j in range(n[1])] for i in range(n[0])],
                "occ": [[[src["occ"][i][j][k] if src["mat"][i][j][k] != "Air" else dst_before["occ"][i][j][k]
                          for k in range(n[2])] for j in range(n[1])] for i in range(n[0])]}
    bad, count = _compare(d0, want, got)
    if count:
        problems.append("%d pasted cells differ from the source" % count)
    out = {"ok": not problems, "verified": not problems, "from": _studs(c0, c1), "to": _studs(d0, d1),
           "sizeInCells": got_n, "source": _summary(src), "pasted": _summary(got), "requestId": r.get("requestId")}
    if bad:
        out["firstMismatches"] = bad
    if problems:
        out["error"] = "; ".join(problems)
    return out


def _roughness(occ, mat):
    """Mean squared occupancy step between neighbouring cells where either is solid: what smoothing lowers."""
    n = _dims(occ)
    tot, cnt = 0.0, 0
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    a, b, c = i + di, j + dj, k + dk
                    if a < n[0] and b < n[1] and c < n[2] and (mat[i][j][k] != "Air" or mat[a][b][c] != "Air"):
                        d = (occ[i][j][k] if mat[i][j][k] != "Air" else 0) - (occ[a][b][c] if mat[a][b][c] != "Air" else 0)
                        tot += d * d
                        cnt += 1
    return tot / cnt if cnt else 0.0


@T._reports
def rbx_terrain_smooth(region: dict, strength: float = 0.5, iterations: int = 1) -> dict:
    "Roblox Studio terrain smoothing without Studio's tool (Terrain:SmoothRegion is RobloxScriptSecurity): the region is read, each solid cell's occupancy blended toward its 26 neighbours' mean by strength (0-1) for iterations passes, and written back as one undo step. Materials never change: empty cells stay empty, water cells keep their water, a solid cell keeps at least 0.05. Read back: the occupancy roughness (mean squared step between neighbours) is lower and every cell's material is the same."
    c0, c1, why = _region_cells(region)
    if why:
        return {"ok": False, "ran": False, "error": why}
    if not _num(strength) or not 0 < strength <= 1 or not isinstance(iterations, int) or not 1 <= iterations <= 10:
        return {"ok": False, "ran": False, "error": "strength is above 0 and at most 1; iterations 1-10"}
    if _count(c0, c1) > MAX_RETURN_CELLS:
        return {"ok": False, "ran": False, "refusedRule": "budget", "error": "%d cells is over %d a call" % (_count(c0, c1), MAX_RETURN_CELLS)}
    gate = _terrain_gate(c0, c1, "rbx_terrain_smooth")
    if gate:
        return gate
    before, err = _read_block(c0, c1)
    if err:
        return err
    n = [c1[i] - c0[i] for i in range(3)]
    mat, occ = before["mat"], [[[x for x in col] for col in p] for p in before["occ"]]
    for _ in range(iterations):
        nxt = [[[x for x in col] for col in p] for p in occ]
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    if mat[i][j][k] in ("Air", "Water"):
                        continue
                    s, c = 0.0, 0
                    for a in range(max(0, i - 1), min(n[0], i + 2)):
                        for b in range(max(0, j - 1), min(n[1], j + 2)):
                            for d in range(max(0, k - 1), min(n[2], k + 2)):
                                if (a, b, d) != (i, j, k):
                                    s += occ[a][b][d] if mat[a][b][d] not in ("Air", "Water") else 0.0
                                    c += 1
                    mean = s / c if c else occ[i][j][k]
                    nxt[i][j][k] = round(min(1.0, max(0.05, (1 - strength) * occ[i][j][k] + strength * mean)), 4)
        occ = nxt
    r0 = _roughness(before["occ"], mat)
    changed = sum(1 for i in range(n[0]) for j in range(n[1]) for k in range(n[2]) if occ[i][j][k] != before["occ"][i][j][k])
    if not changed:
        return {"ok": False, "ran": False, "region": _studs(c0, c1), "roughness": round(r0, 6),
                "error": "nothing to smooth: no solid cell in the region would change"}
    w = _write_block(c0, mat, occ, label="terrain smooth")
    if not w.get("ok"):
        return w
    after, err = _read_block(c0, c1)
    if err:
        return dict(err, ok=False, error="written, but the read-back failed: %s" % err.get("error"))
    r1 = _roughness(after["occ"], after["mat"])
    moved = [[c0[0] + i, c0[1] + j, c0[2] + k] for i in range(n[0]) for j in range(n[1]) for k in range(n[2])
             if after["mat"][i][j][k] != mat[i][j][k]]
    problems = []
    if moved:
        problems.append("%d cells changed material, e.g. %s" % (len(moved), moved[:5]))
    if not r1 < r0:
        problems.append("roughness went from %.6f to %.6f, not lower" % (r0, r1))
    out = {"ok": not problems, "verified": not problems, "region": _studs(c0, c1), "cellsChanged": changed,
           "roughnessBefore": round(r0, 6), "roughnessAfter": round(r1, 6), "writeCalls": w["calls"], "undoGroup": w["undoGroup"]}
    if problems:
        out["error"] = "; ".join(problems)
    return out


# ---------------------------------------------------------------- a floating island
ISLAND_RETRY = ("outcomeUnknown", "retryable")   # a fill that timed out or was withdrawn is sent once more: it is idempotent


def _ring(n, radius, y, cx, cz, phase=0.0):
    return [[cx + radius * math.cos(phase + 2 * math.pi * i / n), y, cz + radius * math.sin(phase + 2 * math.pi * i / n)]
            for i in range(n)]


def _underside(fills, cx, cz, radius, y, underside, body):
    """Rings of shrinking body balls under y, narrowing to a tip; returns the y the tip reaches down to."""
    for k in range(1, underside + 1):
        share = 1 - k / (underside + 1.0)
        b = max(4.0, radius * 0.35 * share)
        rk = radius * share
        y -= b * 0.9
        n = max(1, int(math.ceil(2 * math.pi * rk / (1.2 * b)))) if rk >= b else 1
        for p in (_ring(n, rk, y, cx, cz, k * 0.37) if n > 1 else [[cx, y, cz]]):
            fills.append({"what": "underside", "shape": "ball", "material": body, "position": p, "radius": b})
    if underside:
        b = max(4.0, radius * 0.35 / (underside + 1.0))
        y -= b
        fills.append({"what": "underside", "shape": "ball", "material": body, "position": [cx, y, cz], "radius": b})


def _carves(fills, carves, carve_pad):
    """The carve list as Air regions appended to fills: (cuts, None) or (None, why)."""
    cuts = []
    for i, c in enumerate(carves or []):
        if not isinstance(c, dict) or not _vec(c.get("min")) or not _vec(c.get("max")):
            return None, "carves[%d] is {min: [x, y, z], max: [x, y, z]} in studs (the part's box)" % i
        lo = [c["min"][0] - carve_pad, c["min"][1], c["min"][2] - carve_pad]
        hi = [c["max"][0] + carve_pad, c["max"][1], c["max"][2] + carve_pad]
        c0, c1, why = _region_cells({"min": lo, "max": hi})
        if why:
            return None, "carves[%d]: %s" % (i, why)
        fills.append({"what": "carve", "shape": "region", "material": "Air", "region": _studs(c0, c1), "cells": [c0, c1],
                      "asked": {"min": list(c["min"]), "max": list(c["max"])}})
        cuts.append((c0, c1))
    return cuts, None


def _low(fills):
    return min(f["position"][1] - (f.get("radius") if f["shape"] == "ball" else f["height"] / 2)
               for f in fills if f["shape"] in ("ball", "cylinder"))


def island_plan(center, radius, top, depth=None, top_depth=4.0, cliff=True, underside=3, carves=None, carve_pad=4.0,
                material="Grass", body="Rock"):
    """The fills of one island, in order, and the box they reach: (plan, None) or (None, why). Pure arithmetic."""
    if not _vec(center, 2) or not _num(radius) or not 8 <= radius <= 1024 or not _num(top):
        return None, "center is [x, z], radius 8-1024 studs and top the height its surface is drawn at"
    depth = float(depth) if depth is not None else max(8.0, min(64.0, radius * 0.4))
    if not _num(depth) or not 4 <= depth <= 512 or not _num(top_depth) or not 2 <= top_depth <= depth:
        return None, "depth is 4-512 studs and top_depth 2 to depth"
    if not isinstance(underside, int) or isinstance(underside, bool) or not 0 <= underside <= 8:
        return None, "underside is 0-8 rings"
    if not isinstance(cliff, bool) or not _num(carve_pad) or not 0 <= carve_pad <= 64:
        return None, "cliff is true or false and carve_pad 0-64 studs"
    cx, cz = float(center[0]), float(center[1])
    solid = float(top) - SURFACE_LIFT         # smooth terrain draws a solid's top 2 studs above it (RB1151)
    fills = [{"what": "body", "shape": "cylinder", "material": body, "position": [cx, solid - depth / 2, cz],
              "height": depth, "radius": float(radius)}]
    reach = float(radius)
    if cliff:
        rc = max(4.0, min(16.0, radius * 0.18))
        n = max(6, int(math.ceil(2 * math.pi * radius / (1.2 * rc))))
        for i, p in enumerate(_ring(n, radius, solid - rc, cx, cz)):
            fills.append({"what": "cliff", "shape": "ball", "material": body, "position": p, "radius": rc})
        r2 = rc * 0.85
        for p in _ring(n, radius, solid - 2.5 * rc, cx, cz, math.pi / n):
            fills.append({"what": "cliff", "shape": "ball", "material": body, "position": p, "radius": r2})
        reach = radius + rc
    _underside(fills, cx, cz, radius, solid - depth, underside, body)
    low = _low(fills)
    fills.append({"what": "top", "shape": "cylinder", "material": material, "position": [cx, solid - top_depth / 2, cz],
                  "height": float(top_depth), "radius": float(radius)})
    cuts, why = _carves(fills, carves, carve_pad)
    if why:
        return None, why
    box = {"min": [cx - reach, low, cz - reach], "max": [cx + reach, float(top) + VOXEL, cz + reach]}
    for c0, c1 in cuts:
        st = _studs(c0, c1)
        box = {"min": [min(box["min"][i], st["min"][i]) for i in range(3)], "max": [max(box["max"][i], st["max"][i]) for i in range(3)]}
    return {"fills": fills, "box": box, "solidTop": solid, "drawnTop": float(top), "reach": reach, "bottom": low,
            "profile": "flat", "undersideBelow": solid - top_depth, "sampleRadius": 0.5 * float(radius)}, None


# ---------------------------------------------------------------- a bowl island
# A floating island hidden by its own ground: the playable middle stays level at top, an apron climbs `apron` studs from
# flat to rise, the ground rises to a crest at radius crest whose height above top varies round the ring (three sine
# harmonics, `variation` of it) and then falls away to the edge, `fall` studs under top. Value noise (`noise` of the crest
# height, on a `noise_cell`-stud lattice) roughs up everything past flat. Written as heightmap tiles standing on one base
# height, then the usual underside below the base. Shapes of the climb: smooth (a berm), steep (walls close in) and
# terraces (`steps` level benches).
BOWL_SHAPES = ("smooth", "steep", "terraces")
BOWL_KEYS = ("kind", "shape", "flat", "rise", "apron", "crest", "height", "variation", "noise", "noise_cell", "fall",
             "wobble", "steps", "seed", "paint")
HORIZON_DIRECTIONS, HORIZON_PITCHES = 16, (0.0, -5.0)     # level and slightly down, as the lesson's self-test cast them


def _smooth01(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def _lattice(i, j, seed):
    h = (i * 374761393 + j * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return (h & 0xFFFF) / 32767.5 - 1.0


def _value_noise(x, z, cell, seed):
    """Two octaves of smoothly interpolated value noise, about -1..1, the same for the same seed."""
    def octave(x, z, c):
        fx, fz = x / c, z / c
        i, j = math.floor(fx), math.floor(fz)
        u, v = _smooth01(fx - i), _smooth01(fz - j)
        a, b = _lattice(i, j, seed), _lattice(i + 1, j, seed)
        c0, d = _lattice(i, j + 1, seed), _lattice(i + 1, j + 1, seed)
        near = a + (b - a) * u
        return near + ((c0 + (d - c0) * u) - near) * v
    return 0.65 * octave(x, z, cell) + 0.35 * octave(x + 913, z - 377, cell / 2.3)


def bowl_profile(profile, radius, top):
    """A bowl profile with its defaults filled in: (p, None) or (None, why). radius is the island's edge."""
    if not isinstance(profile, dict) or profile.get("kind") != "bowl" or set(profile) - set(BOWL_KEYS):
        return None, "profile is {kind: 'bowl', height, and any of %s}" % ", ".join(k for k in BOWL_KEYS[2:] if k != "height")
    p = {"shape": "smooth", "flat": 0.45 * radius, "apron": 3.0, "crest": 0.78 * radius, "variation": 0.22,
         "noise": 0.07, "noise_cell": 48.0, "fall": min(30.0, 0.1 * radius), "wobble": min(10.0, 0.03 * radius),
         "steps": 5, "seed": 1, "paint": None}
    p.update({k: v for k, v in profile.items() if k != "kind" and v is not None})
    p.setdefault("rise", p["flat"] + 0.07 * radius)
    if p["shape"] not in BOWL_SHAPES:
        return None, "profile shape is one of %s" % ", ".join(BOWL_SHAPES)
    if not _num(p.get("height")) or not 4 <= p["height"] <= 512:
        return None, "profile height is the crest's height above top, 4-512 studs: more than a camera can climb to"
    for k in ("flat", "rise", "crest", "apron", "variation", "noise", "noise_cell", "fall", "wobble"):
        if not _num(p[k]) or p[k] < 0:
            return None, "profile %s is a number, 0 or more" % k
    if not 0 < p["flat"] < p["rise"] < p["crest"] < radius - p["wobble"]:
        return None, ("profile radii run 0 < flat < rise < crest < radius - wobble (flat %g, rise %g, crest %g, radius %g, "
                      "wobble %g)" % (p["flat"], p["rise"], p["crest"], radius, p["wobble"]))
    if p["variation"] >= 0.9 or p["noise"] >= 0.5 or p["noise_cell"] < 8:
        return None, "profile variation is under 0.9, noise under 0.5 and noise_cell 8 studs or more"
    if p["apron"] >= p["height"]:
        return None, "profile apron is under height"
    if not isinstance(p["steps"], int) or isinstance(p["steps"], bool) or not 2 <= p["steps"] <= 12 \
            or not isinstance(p["seed"], int) or isinstance(p["seed"], bool):
        return None, "profile steps is 2-12 and seed an integer"
    for r in p["paint"] or []:
        if not isinstance(r, dict) or set(r) - {"material", "min_height", "max_height", "min_slope", "max_slope"}:
            return None, "a profile paint rule is {material, min_height, max_height, min_slope, max_slope}"
    rng = random.Random(p["seed"])
    p["phases"] = [rng.uniform(0, 2 * math.pi) for _ in range(5)]
    p["radius"], p["top"] = float(radius), float(top)
    return p, None


def bowl_height(p, cx, cz, x, z):
    """The drawn ground height (world y) of a bowl at (x, z), or None past its edge. Pure."""
    dx, dz = x - cx, z - cz
    r, a = math.hypot(dx, dz), math.atan2(dz, dx)
    ph, top, hgt = p["phases"], p["top"], p["height"]
    edge = p["radius"] - p["wobble"] * (1 - math.sin(5 * a + ph[4])) / 2
    if r > edge:
        return None
    if r <= p["flat"]:
        return top
    hc = hgt * (1 + p["variation"] * (0.5 * math.sin(3 * a + ph[0]) + 0.3 * math.sin(7 * a + ph[1])
                                      + 0.2 * math.sin(13 * a + ph[2])))
    rc = p["crest"] + min(14.0, 0.25 * (p["crest"] - p["rise"]), 0.25 * (p["radius"] - p["crest"])) * math.sin(4 * a + ph[3])
    lump = _value_noise(x, z, p["noise_cell"], p["seed"]) * hgt * p["noise"] * _smooth01((r - p["flat"]) / 40.0)
    if r <= p["rise"]:
        return top + p["apron"] * _smooth01((r - p["flat"]) / (p["rise"] - p["flat"])) + lump * 0.3
    if r <= rc:
        t = (r - p["rise"]) / (rc - p["rise"])
        if p["shape"] == "steep":
            s = t ** 0.45
        elif p["shape"] == "terraces":
            n = p["steps"]
            i = min(n - 1, int(t * n))
            s = (i + _smooth01((t * n - i - 0.68) / 0.32)) / n
            lump *= 0.35                       # benches stay level enough to read as terraces
        else:
            s = _smooth01(t)
        return top + p["apron"] + (hc - p["apron"]) * s + lump
    t2 = (r - rc) / max(1.0, edge - rc)
    return top + hc * (1 - t2 ** 1.5) - p["fall"] * t2 + lump


def bowl_plan(center, radius, top, profile, depth=None, underside=3, carves=None, carve_pad=4.0, material="Grass",
              body="Rock"):
    """A bowl island's writes, in order: heightmap tiles standing on one base, the underside below it, the carves.
    (plan, None) or (None, why). Pure arithmetic."""
    if not _vec(center, 2) or not _num(radius) or not 32 <= radius <= 2048 or not _num(top):
        return None, "center is [x, z], radius 32-2048 studs (a bowl's edge) and top the height its middle is drawn at"
    p, why = bowl_profile(profile, float(radius), float(top))
    if why:
        return None, why
    if not isinstance(underside, int) or isinstance(underside, bool) or not 0 <= underside <= 8:
        return None, "underside is 0-8 rings"
    if not _num(carve_pad) or not 0 <= carve_pad <= 64:
        return None, "carve_pad is 0-64 studs"
    need = p["fall"] + p["height"] * p["noise"] + 8         # the edge, at its lowest lump, still stands on the base
    depth = float(depth) if depth is not None else need + 8
    if not _num(depth) or not need <= depth <= 1024:
        return None, "a bowl's depth (its base under top) is %g-1024 studs: fall plus the noise plus 8" % need
    cx, cz = float(center[0]), float(center[1])
    base = math.floor((float(top) - depth) / VOXEL) * VOXEL
    ox, oz = math.floor((cx - radius) / VOXEL) * VOXEL, math.floor((cz - radius) / VOXEL) * VOXEL
    nx = int(math.ceil((cx + radius - ox) / VOXEL))
    nz = int(math.ceil((cz + radius - oz) / VOXEL))
    grid, peak = [], 0.0
    for k in range(nz):
        row = []
        for i in range(nx):
            h = bowl_height(p, cx, cz, ox + (i + 0.5) * VOXEL, oz + (k + 0.5) * VOXEL)
            row.append(None if h is None else round(max(h - base, SURFACE_LIFT), 3))
            if h is not None:
                peak = max(peak, h)
        grid.append(row)
    lvl = float(top) - base
    paint = [{"material": body, "min_slope": 38}] if p["paint"] is None else [
        dict(r, **{k: r[k] + lvl for k in ("min_height", "max_height") if k in r}) for r in p["paint"]]
    ny = max(1, int(math.ceil((peak - base - SURFACE_LIFT) / VOXEL)))
    side = max(8, min(256, int(math.sqrt(MAX_CELLS / ny))))
    fills = []
    for k0 in range(0, nz, side):
        for i0 in range(0, nx, side):
            sub = [row[i0:i0 + side] for row in grid[k0:k0 + side]]
            if not any(v is not None for row in sub for v in row):
                continue
            fills.append({"what": "ground", "shape": "heightmap", "material": material, "paint": paint,
                          "origin": [ox + i0 * VOXEL, base, oz + k0 * VOXEL], "columns": [len(sub[0]), len(sub)],
                          "maxHeight": max(v for row in sub for v in row if v is not None) + base, "heights": sub})
    _underside(fills, cx, cz, float(radius) * 0.8, base, underside, body)
    balls = [f for f in fills if f["shape"] == "ball"]
    low = _low(balls) if balls else float(base)
    cuts, why = _carves(fills, carves, carve_pad)
    if why:
        return None, why
    box = {"min": [ox, low, oz], "max": [ox + nx * VOXEL, peak + VOXEL, oz + nz * VOXEL]}
    for c0, c1 in cuts:
        st = _studs(c0, c1)
        box = {"min": [min(box["min"][i], st["min"][i]) for i in range(3)], "max": [max(box["max"][i], st["max"][i]) for i in range(3)]}
    shape = {k: p[k] for k in ("shape", "flat", "rise", "apron", "crest", "height", "variation", "noise", "noise_cell",
                               "fall", "wobble", "steps", "seed")}
    return {"fills": fills, "box": box, "solidTop": float(top) - SURFACE_LIFT, "drawnTop": float(top),
            "reach": float(radius), "bottom": low, "profile": "bowl", "bowl": shape, "base": base, "peak": round(peak, 3),
            "undersideBelow": float(base), "sampleRadius": 0.5 * p["flat"]}, None


def _ray(origin, direction):
    params = {"$t": "RaycastParams", "filterType": "Include", "filter": [{"$t": "Instance", "path": TERRAIN}],
              "ignoreWater": True}
    r = _call("Raycast", [_w3(origin), _w3(direction), params], True, target=["Workspace"], cls="Workspace")
    if not r.get("ok"):
        return None, r
    hit = (r.get("returns") or [None])[0]
    if not isinstance(hit, dict) or hit.get("$t") != "RaycastResult":
        return {"hit": False}, None
    pos = V.decode(hit.get("position")) or {}
    return {"hit": True, "position": [pos.get("x"), pos.get("y"), pos.get("z")], "material": _unwire(hit.get("material"))}, None


def horizon_probe(center, radius, top, heights, directions=HORIZON_DIRECTIONS, pitches=HORIZON_PITCHES):
    """From the island's middle at each camera height (studs above top), rays outward in `directions` evenly spaced
    directions at each pitch (degrees, 0 level, negative down), terrain only: each must hit the island's own ground
    inside its edge (radius), or a camera there sees past the edge. (report, problems, error)."""
    cx, cz = float(center[0]), float(center[1])
    reach = 2.0 * float(radius) + 64
    rows, miss = [], []
    for h in heights:
        for d in range(directions):
            a = 2 * math.pi * d / directions
            for pt in pitches:
                pr = math.radians(pt)
                dirn = [math.cos(a) * math.cos(pr) * reach, math.sin(pr) * reach, math.sin(a) * math.cos(pr) * reach]
                o = [cx, float(top) + h, cz]
                hit, err = _ray(o, dirn)
                if err:
                    return None, None, err
                row = {"height": h, "direction": round(math.degrees(a), 2), "pitch": pt, "hit": hit["hit"]}
                if hit["hit"]:
                    hx, hy, hz = hit["position"]
                    row.update(distance=round(math.hypot(hx - cx, hz - cz), 2), y=round(hy, 2))
                    row["inside"] = row["distance"] <= float(radius)
                else:
                    row["inside"] = False
                rows.append(row)
                if not row["inside"]:
                    miss.append(row)
    hits = [x["distance"] for x in rows if x.get("hit")]
    rep = {"rays": len(rows), "heights": list(heights), "directions": directions, "pitches": list(pitches),
           "inside": len(rows) - len(miss), "farthest": max(hits) if hits else None, "seesPastEdge": miss[:32]}
    probs = []
    if not rows:
        probs.append("horizon: no ray was cast: nothing was checked")
    elif miss:
        probs.append("horizon: %d of %d rays from the middle see past the island's edge (no ground of its own inside "
                     "radius %g), e.g. at height %g toward %g degrees, pitch %g" % (
                         len(miss), len(rows), radius, miss[0]["height"], miss[0]["direction"], miss[0]["pitch"]))
    return rep, probs, None


def _horizon_args(horizon, plan):
    """(heights, directions, pitches) or why. A bowl's default: 6 studs and 0.6 of its crest height above top."""
    if horizon is None:
        if plan.get("profile") != "bowl":
            return None
        horizon = {}
    if not isinstance(horizon, dict) or set(horizon) - {"heights", "directions", "pitches"}:
        return "horizon is {heights: [studs above top], directions: 4-64, pitches: [degrees, 0 level, negative down]}"
    default = [6.0, round(0.6 * plan["bowl"]["height"], 2)] if plan.get("profile") == "bowl" else [6.0]
    hs = horizon.get("heights", default)
    ds = horizon.get("directions", HORIZON_DIRECTIONS)
    ps = horizon.get("pitches", list(HORIZON_PITCHES))
    if not isinstance(hs, list) or not hs or len(hs) > 8 or not all(_num(h) and 0 <= h <= 1024 for h in hs):
        return "horizon heights is a list of 1-8 heights above top, 0-1024 studs"
    if not isinstance(ds, int) or isinstance(ds, bool) or not 4 <= ds <= 64:
        return "horizon directions is 4-64"
    if not isinstance(ps, list) or not ps or len(ps) > 4 or not all(_num(x) and -45 <= x <= 10 for x in ps):
        return "horizon pitches is a list of 1-4 angles, -45 to 10 degrees"
    return [float(h) for h in hs], ds, [float(x) for x in ps]


def _island_fill(f, timeout):
    if f["shape"] == "heightmap":
        return rbx_terrain_heightmap(f["origin"], heights=f["heights"], material=f["material"], paint=f["paint"])
    m = _wmat(f["material"])
    if f["shape"] == "cylinder":
        return _call("FillCylinder", [_wcf(f["position"]), float(f["height"]), float(f["radius"]), m], False, timeout=timeout)
    if f["shape"] == "ball":
        return _call("FillBall", [_w3(f["position"]), float(f["radius"]), m], False, timeout=timeout)
    c0, c1 = f["cells"]
    return _call("FillRegion", [_wregion(c0, c1), VOXEL, m], False, timeout=timeout)


def _in_cut(x, z, cuts):
    return any(st["min"][0] - VOXEL <= x <= st["max"][0] + VOXEL and st["min"][2] - VOXEL <= z <= st["max"][2] + VOXEL
               for st in cuts)


@T._reports
def rbx_terrain_island(center: list, radius: float, top: float, material: str = "Grass", body: str = "Rock",
                       depth: float = None, top_depth: float = 4, cliff: bool = True, underside: int = 3,
                       carves: list = None, carve_pad: float = 4, fill_timeout: float = 60, dry_run: bool = False,
                       profile: dict = None, horizon: dict = None, check_only: bool = False) -> dict:
    "Roblox Studio terrain: a round floating island in one call. center [x, z] and radius in studs; top is where its surface is DRAWN (the solid ends 2 studs under it, RB1151). The fills, in order: a body cylinder of body (default Rock) depth studs tall (default 0.4 x radius, 8-64), two rows of rock balls bulging from the rim for a cliff (cliff), underside rings of shrinking balls narrowing to a tip (underside rings, 0-8), the top layer (a material cylinder top_depth studs thick, last, so the grass stays on top), then carves: each part box {min, max} you will build a flat floor or pit wall with is erased to Air, widened by carve_pad studs on X and Z and aligned outward to the 4-stud grid, because a terrain edge cannot sit flush against parts (a fill reaches 2.48 studs past its shape, RB1153). Every fill (Terrain FillCylinder, FillBall, FillRegion) is sent once more when it times out or is withdrawn - a fill is idempotent (one large ball did, and went through on the second send). profile {kind: 'bowl', height, ...} builds a BOWL instead, an island that hides that it floats: the middle stays level at top out to flat (default 0.45 x radius), an apron climbs apron studs (3) to rise (flat + 0.07 x radius), the ground rises to a crest at radius crest (0.78 x radius) height studs above top - give more than the camera can reach - varied by variation (0.22) round the ring, then falls away to the edge at radius (wobbled by wobble studs, min(10, 0.03 x radius)) fall studs (min(30, 0.1 x radius)) under top; value noise of noise (0.07) times height on a noise_cell (48) stud lattice, seeded by seed (1), roughs up all past flat; shape smooth (a berm), steep (walls close in) or terraces (steps level benches, 5). The ground is written as rbx_terrain_heightmap tiles standing on one base, depth studs under top (default fall + the noise + 16), each tile read back by its own rays, painted by profile paint (heights above top, slopes in degrees; default body on slopes over 38 degrees), then underside rings under the base and the carves; cliff and top_depth are not used. Read back: the drawn top by WorldRoot:Raycast at the centre and four points at half the radius (half of flat for a bowl), outside every carve, within 1 stud of top, every carve read empty by Terrain:ReadVoxels, and a ray up from below the centre finding the underside. horizon {heights, directions, pitches} (a bowl runs it by default: heights 6 and 0.6 x height above top, 16 directions, pitches 0 and -5 degrees) casts terrain-only rays outward from the middle at each camera height, level and slightly down: each must hit the island's own ground inside radius, or a camera there sees past the edge and the island fails (horizon.seesPastEdge, farthest). check_only true sends no fill and only runs the horizon probe on what stands there (read only; heights default 6 on a flat island). dry_run true sends nothing and answers the plan: every fill (a bowl's tiles without their heights), the box it reaches, the carve regions. Refused before sending: a flat island's top that is not of the form 4k + 2 (its fills end inside a cell the fill takes whole, so it would be drawn at the next such height, RB2885; nearestTops names the two that land), the box outside Terrain.MaxExtents, and in scratch mode anything decision 6 does not allow."
    for m, air in ((material, False), (body, False)):
        bad = _valid_material(m, allow_air=air)
        if bad:
            return bad
    if not isinstance(dry_run, bool) or not isinstance(check_only, bool) or not _num(fill_timeout) \
            or not 5 <= fill_timeout <= 300:
        return {"ok": False, "ran": False, "error": "dry_run and check_only are true or false and fill_timeout 5-300 seconds"}
    if profile is not None:
        plan, why = bowl_plan(center, radius, top, profile, depth, underside, carves, carve_pad, material, body)
    else:
        plan, why = island_plan(center, radius, top, depth, top_depth, cliff, underside, carves, carve_pad, material, body)
    if why:
        return {"ok": False, "ran": False, "error": why}
    if profile is None and not check_only:
        # the flat profile's fills end inside a cell the fill takes whole (RB2137) and the surface is drawn 2 above it
        # (RB1151), so only a top of the form 4k + 2 lands where asked: top 40 drew 42 and 44 drew 46 (RB2885)
        k = (float(top) - SURFACE_LIFT) / VOXEL
        if abs(k - round(k)) > 1e-6:
            lo = math.floor(k) * VOXEL + SURFACE_LIFT
            return {"ok": False, "ran": False, "refusedRule": "top-off-grid", "nearestTops": [lo, lo + VOXEL],
                    "error": "a flat island's top lands only at a height of the form 4k + 2 (the fill takes the cell its "
                             "solid ends in whole, RB2885): top %g would be drawn at %g. Ask for %g or %g. Nothing was sent"
                             % (float(top), lo + VOXEL, lo, lo + VOXEL)}
    hz = _horizon_args(horizon if horizon is not None or not check_only else {}, plan)
    if isinstance(hz, str):
        return {"ok": False, "ran": False, "error": hz}
    c0, c1, _w = _region_cells(plan["box"])
    counts = {}
    for f in plan["fills"]:
        counts[f["what"]] = counts.get(f["what"], 0) + 1
    out = {"plan": {"fills": len(plan["fills"]), "byPart": counts, "box": plan["box"], "solidTop": plan["solidTop"],
                    "drawnTop": plan["drawnTop"], "bottom": round(plan["bottom"], 3), "profile": plan["profile"]},
           "surface": "drawn", "surfaceConvention": SURFACE_CONVENTION}
    if plan["profile"] == "bowl":
        out["plan"].update(bowl=plan["bowl"], base=plan["base"], peak=plan["peak"])
    if dry_run:
        shown = [{k: v for k, v in f.items() if k != "heights"} for f in plan["fills"]]
        return dict(out, ok=True, ran=False, sent=False, dryRun=True, fills=shown)
    if check_only:
        rep, probs, err = horizon_probe(center, radius, top, *hz)
        if err:
            return dict(out, ok=False, verified=False, error="the horizon Raycast failed: %s" % err.get("error"))
        out.update(horizon=rep, ok=not probs, verified=not probs, sent=0, checkOnly=True)
        if probs:
            out["error"] = "; ".join(probs)
        return out
    gate = _terrain_gate(c0, c1, "rbx_terrain_island")
    if gate:
        return dict(gate, plan=out["plan"])
    t0 = time.time()
    sent, retried, tiles = 0, [], []
    for i, f in enumerate(plan["fills"]):
        r = _island_fill(f, fill_timeout)
        sent += 1
        if not r.get("ok") and any(r.get(k) for k in ISLAND_RETRY):
            retried.append({"fill": i, "what": f["what"], "first": r.get("error")})
            r = _island_fill(f, fill_timeout)
            sent += 1
        if f["shape"] == "heightmap":
            tiles.append({k: r.get(k) for k in ("ok", "region", "columns", "samples", "worst", "worstOutside",
                                                "byNeighbourhood", "besideTerrain", "error")})
        if not r.get("ok"):
            return dict(out, ok=False, verified=False, sent=sent, retried=retried, failedAt=i,
                        failedFill={k: v for k, v in f.items() if k != "heights"}, tiles=tiles or None,
                        seconds=round(time.time() - t0, 1),
                        error="fill %d (%s %s) failed%s: %s. Fills 0-%d were sent; a fill is idempotent, so the island "
                              "can be sent again whole" % (i, f["what"], f["shape"], " after a retry" if retried and
                                                           retried[-1]["fill"] == i else "", r.get("error"), i - 1))
    out.update(sent=sent, retried=retried, fillSeconds=round(time.time() - t0, 1))
    if tiles:
        out["tiles"] = tiles
    problems = []
    cuts = [f["region"] for f in plan["fills"] if f["what"] == "carve"]
    cx, cz = float(center[0]), float(center[1])
    tops = []
    sr = plan["sampleRadius"]
    for dx, dz in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        x, z = cx + dx * sr, cz + dz * sr
        if _in_cut(x, z, cuts):
            continue
        hit, err = _raycast_down(x, z, float(top) + 32, 64 + float(top_depth))
        if err:
            return dict(out, ok=False, verified=False, error="filled, but the drawn-top Raycast failed: %s" % err.get("error"))
        y = hit["position"][1] if hit.get("hit") else None
        tops.append({"at": [round(x, 3), round(z, 3)], "y": None if y is None else round(y, 3),
                     "material": hit.get("material")})
        if y is None or abs(y - float(top)) > HEIGHT_TOL:
            problems.append("the drawn top at %s reads %s, asked %s (tolerance %s)" % (tops[-1]["at"], y, top, HEIGHT_TOL))
    if not tops:
        problems.append("every top sample fell in a carve: nothing read the surface")
    out["drawnTop"] = tops
    under = _call("Raycast", [_w3([cx, plan["bottom"] - 32, cz]), _w3([0, plan["solidTop"] - plan["bottom"] + 32, 0]),
                              {"$t": "RaycastParams", "filterType": "Include",
                               "filter": [{"$t": "Instance", "path": TERRAIN}], "ignoreWater": True}],
                  True, target=["Workspace"], cls="Workspace")
    hit = (under.get("returns") or [None])[0] if under.get("ok") else None
    uy = (V.decode(hit.get("position")) or {}).get("y") if isinstance(hit, dict) and hit.get("$t") == "RaycastResult" else None
    out["underside"] = {"planned": round(plan["bottom"], 3), "hitY": None if uy is None else round(uy, 3)}
    if not under.get("ok"):
        problems.append("the underside Raycast failed: %s" % under.get("error"))
    elif uy is None or uy > plan["undersideBelow"]:
        problems.append("a ray up from under the centre found no underside below %s (hit %s)" % (
            "the base" if plan["profile"] == "bowl" else "the top layer", uy))
    carved = []
    for f in plan["fills"]:
        if f["what"] != "carve":
            continue
        a, b = f["cells"]
        if _count(a, b) > MAX_RETURN_CELLS:
            carved.append({"region": f["region"], "checked": False, "why": "over %d cells" % MAX_RETURN_CELLS})
            continue
        blk, err = _read_block(a, b)
        if err:
            return dict(out, ok=False, verified=False, error="filled, but a carve read-back failed: %s" % err.get("error"))
        left = sum(1 for pl, po in zip(blk["mat"], blk["occ"]) for cm, co in zip(pl, po) for mm, oo in zip(cm, co)
                   if mm != "Air" and oo > OCC_TOL)
        carved.append({"region": f["region"], "asked": f["asked"], "solidCellsLeft": left, "checked": True})
        if left:
            problems.append("carve %s still holds %d solid cells" % (f["region"], left))
    out["carves"] = carved
    if hz:
        rep, probs, err = horizon_probe(center, radius, top, *hz)
        if err:
            return dict(out, ok=False, verified=False, error="filled, but the horizon Raycast failed: %s" % err.get("error"))
        out["horizon"] = rep
        problems += probs
    unchecked = [c["region"] for c in carved if c.get("checked") is False]
    out.update(ok=not problems, verified=not problems and not unchecked, seconds=round(time.time() - t0, 1))
    if unchecked:
        out["unchecked"] = unchecked
        out["note"] = ("%d carve(s) over %d cells were not read back, so they are not verified: %s" % (
            len(unchecked), MAX_RETURN_CELLS, ", ".join(str(u) for u in unchecked)))
    if problems:
        out["error"] = "; ".join(problems)
    return out


def rbx_terrain_sample(points: list, ray_height: float = 512) -> dict:
    "Roblox Studio terrain at points [[x, y, z], ...] (at most 200): the cell's material and occupancy (Terrain:ReadVoxels), and straight down from ray_height above the point the terrain surface's height, material, normal and slope in degrees (WorldRoot:Raycast, terrain only, water ignored). Read only."
    if not isinstance(points, list) or not points or len(points) > 200 or not all(_vec(p) for p in points):
        return {"ok": False, "ran": False, "error": "points is a list of 1-200 [x, y, z]"}
    if not _num(ray_height) or not 0 < ray_height <= 4096:
        return {"ok": False, "ran": False, "error": "ray_height is above 0 and at most 4096"}
    out = []
    for p in points:
        cell, err = _read_cell(_cell_of(p))
        if err:
            return dict(err, ok=False, at=p)
        hit, err = _raycast_down(p[0], p[2], p[1] + ray_height, 2 * ray_height)
        if err:
            return dict(err, ok=False, at=p)
        s = {"point": p, "cell": _cell_of(p), "material": cell[0], "occupancy": cell[1], "surface": None}
        if hit.get("hit"):
            ny = hit["normal"][1]
            s["surface"] = {"height": hit["position"][1], "material": hit.get("material"), "normal": hit["normal"],
                            "slope": round(math.degrees(math.acos(max(-1.0, min(1.0, ny)))), 3) if _num(ny) else None}
        out.append(s)
    return {"ok": True, "samples": out}


def rbx_terrain_beta_probe(region: dict = None) -> dict:
    "Roblox Studio terrain: measure the beta voxel API before building on it. Says which of Terrain ReadVoxelsAsync_beta, WriteVoxelsAsync_beta, ModifyVoxelsAsync_beta, IterateVoxelsAsync_beta and ClearVoxelsAsync_beta this catalog has (refused beta-api when none), then calls ReadVoxelsAsync_beta once on region (default one cell at the origin) and returns what came back beside a ReadVoxels of the same region. Read only: no beta write is made."
    cat, err = T._catalog()
    if err:
        return err
    names = ["ReadVoxelsAsync_beta", "WriteVoxelsAsync_beta", "ModifyVoxelsAsync_beta", "IterateVoxelsAsync_beta",
             "ClearVoxelsAsync_beta"]
    have = {n: cat.read_refusal("Terrain", n, kind="F") is None for n in names}
    if not any(have.values()):
        return {"ok": False, "ran": False, "refusedRule": "beta-api", "catalog": have,
                "error": "this Studio's catalog has none of the beta voxel members (Studio %s)" % cat.studio.get("version")}
    c0, c1, why = _region_cells(region or {"min": [0, 0, 0], "max": [4, 4, 4]})
    if why:
        return {"ok": False, "ran": False, "error": why}
    out = {"ok": True, "catalog": have, "region": _studs(c0, c1)}
    if have["ReadVoxelsAsync_beta"]:
        r = _call("ReadVoxelsAsync_beta", [_wregion(c0, c1), VOXEL, {"$t": "Array", "v": list(CHANNELS)}], True)
        out["readVoxelsAsync"] = {"ok": r.get("ok"), "error": r.get("error"), "refusedRule": r.get("refusedRule"),
                                  "returns": [V.decode(x) for x in r.get("returns") or []]}
        for x in r.get("returns") or []:
            if isinstance(x, dict) and x.get("handle"):
                T._rbx("release_handle", handle=x["handle"])
    ref, err = _read_block(c0, c1)
    out["readVoxels"] = _summary(ref) if ref else {"error": err.get("error")}
    out["note"] = ("TerrainReadOperation delivers its block through its Ready event; rbx_watch subscribes by path, not "
                   "handle, so the block is not read yet (PLUGIN_REQUESTS.md, lane 3)")
    return out


@T._reports
def rbx_terrain_limit_probe(max_cells: int = 64, write: bool = False) -> dict:
    "Roblox Studio terrain: measure the largest cube one Terrain:ReadVoxels call accepts, doubling from 4 cells a side up to max_cells (at most 256), each read at the scratch box's corner; the first refusal's own words come back. With write, the same sizes are written as Air into the declared scratch box (empty first, read to prove it) to find the WriteVoxels limit; a write follows the terrain gate. Answers the per-call limits and reply sizes for the internal notes."
    if not isinstance(max_cells, int) or not 4 <= max_cells <= 256:
        return {"ok": False, "ran": False, "error": "max_cells is 4-256"}
    base = [x // VOXEL for x in SCRATCH_BOX["min"]]
    reads, writes, n = [], [], 4
    largest_read = largest_write = None
    while n <= max_cells:
        c1 = [base[0] + n, base[1] + n, base[2] + n]
        t0 = time.monotonic()
        r = _call("ReadVoxels", [_wregion(base, c1), VOXEL], True, timeout=120)
        row = {"side": n, "cells": n ** 3, "ok": bool(r.get("ok")), "ms": int((time.monotonic() - t0) * 1000)}
        if r.get("ok"):
            try:
                g = _unwire((r.get("returns") or [None])[0])
                row["dims"] = _dims(g)
            except _Cut as exc:
                row.update(ok=False, error=str(exc))
        else:
            row["error"] = r.get("error")
        reads.append(row)
        if not row["ok"]:
            break
        largest_read = n
        n *= 2
    if write:
        n = 4
        while n <= min(max_cells, largest_read or 0):
            c1 = [base[0] + n, base[1] + n, base[2] + n]
            gate = _terrain_gate(base, c1, "rbx_terrain_limit_probe write")
            if gate:
                writes.append({"side": n, "ok": False, "refused": gate.get("refusedRule"), "error": gate.get("error")})
                break
            blk, err = _read_block(base, c1)
            if err or _summary(blk)["nonEmptyCells"]:
                writes.append({"side": n, "ok": False, "error": "the box is not empty here; nothing written"
                               if not err else err.get("error")})
                break
            t0 = time.monotonic()
            r = _call("WriteVoxels", [_wregion(base, c1), VOXEL, _wgrid(_blank([n, n, n], "Air"), _wmat),
                                      _wgrid(_blank([n, n, n], 0.0), float)], False, timeout=120)
            writes.append({"side": n, "cells": n ** 3, "ok": bool(r.get("ok")), "ms": int((time.monotonic() - t0) * 1000),
                           "error": r.get("error")})
            if not r.get("ok"):
                break
            largest_write = n
            n *= 2
    return {"ok": largest_read is not None, "reads": reads, "writes": writes, "largestReadSide": largest_read,
            "largestWriteSide": largest_write, "chunkInUse": CHUNK, "at": _studs(base, base)["min"]}


# ---------------------------------------------------------------- batch 7: lighting and effects
def _children(parent):
    t = T.rbx_tree(parent, depth=1, limit=1000)
    if not t.get("ok"):
        return None, t
    return [(c.get("name"), c.get("className")) for c in (t.get("root") or {}).get("children") or []], None


def _find_or_create(parent, cls, props, name=None):
    """The first child of parent of class cls (named name, if given) set to props, or a new one; read back either way."""
    kids, err = _children(parent)
    if err:
        return err
    pp = V.split_path(parent)
    for n, c in kids:
        if c == cls and (name is None or n == name):
            if not props:
                return {"ok": True, "verified": True, "path": V.join_path(pp + [n]), "created": False}
            r = T.rbx_set_props(pp + [n], props)
            return dict(r, path=V.join_path(pp + [n]), created=False)
    r = T.rbx_create(cls, pp, name=T._unique_name(pp, name or cls), props=props or None)
    return dict(r, created=True)


PRESETS = {
    # Starting points, chosen by eye and documented as such: every value is an ordinary property a person could type in.
    "day": {"lighting": {"ClockTime": 14, "Brightness": 3, "Ambient": [0.27, 0.27, 0.27], "OutdoorAmbient": [0.5, 0.5, 0.5],
                         "ExposureCompensation": 0, "EnvironmentDiffuseScale": 1, "EnvironmentSpecularScale": 1,
                         "GlobalShadows": True},
            "atmosphere": {"Density": 0.3, "Offset": 0.25, "Haze": 0, "Glare": 0, "Color": [0.78, 0.84, 0.92],
                           "Decay": [0.42, 0.5, 0.6]},
            "clouds": {"Cover": 0.5, "Density": 0.5, "Color": [1, 1, 1], "Enabled": True},
            "effects": {"BloomEffect": {"Intensity": 0.6, "Size": 24, "Threshold": 2, "Enabled": True},
                        "ColorCorrectionEffect": {"Brightness": 0, "Contrast": 0.05, "Saturation": 0.05,
                                                  "TintColor": [1, 1, 1], "Enabled": True}}},
    "dusk": {"lighting": {"ClockTime": 17.8, "Brightness": 2, "Ambient": [0.3, 0.24, 0.24], "OutdoorAmbient": [0.55, 0.42, 0.38],
                          "ExposureCompensation": 0.2, "EnvironmentDiffuseScale": 1, "EnvironmentSpecularScale": 1,
                          "GlobalShadows": True},
             "atmosphere": {"Density": 0.35, "Offset": 0.1, "Haze": 1.5, "Glare": 0.6, "Color": [0.95, 0.66, 0.5],
                            "Decay": [0.6, 0.3, 0.25]},
             "clouds": {"Cover": 0.6, "Density": 0.4, "Color": [1, 0.8, 0.7], "Enabled": True},
             "effects": {"BloomEffect": {"Intensity": 0.8, "Size": 32, "Threshold": 1.6, "Enabled": True},
                         "ColorCorrectionEffect": {"Brightness": 0, "Contrast": 0.1, "Saturation": 0.1,
                                                   "TintColor": [1, 0.92, 0.85], "Enabled": True},
                         "SunRaysEffect": {"Intensity": 0.15, "Spread": 0.6, "Enabled": True}}},
    "night": {"lighting": {"ClockTime": 0, "Brightness": 1, "Ambient": [0.1, 0.1, 0.15], "OutdoorAmbient": [0.16, 0.18, 0.28],
                           "ExposureCompensation": 0, "EnvironmentDiffuseScale": 0.5, "EnvironmentSpecularScale": 0.5,
                           "GlobalShadows": True},
              "atmosphere": {"Density": 0.35, "Offset": 0, "Haze": 0.5, "Glare": 0, "Color": [0.2, 0.24, 0.35],
                             "Decay": [0.1, 0.12, 0.2]},
              "clouds": {"Cover": 0.3, "Density": 0.3, "Color": [0.5, 0.55, 0.7], "Enabled": True},
              "effects": {"BloomEffect": {"Intensity": 1, "Size": 24, "Threshold": 1.2, "Enabled": True},
                          "ColorCorrectionEffect": {"Brightness": -0.02, "Contrast": 0.1, "Saturation": -0.1,
                                                    "TintColor": [0.85, 0.9, 1], "Enabled": True}}},
    "interior": {"lighting": {"Brightness": 0, "Ambient": [0.35, 0.33, 0.3], "OutdoorAmbient": [0.2, 0.2, 0.2],
                              "ExposureCompensation": 0.3, "EnvironmentDiffuseScale": 0.2, "EnvironmentSpecularScale": 0.3,
                              "GlobalShadows": True},
                 "atmosphere": {"Density": 0, "Haze": 0, "Glare": 0},
                 "clouds": {"Enabled": False},
                 "effects": {"BloomEffect": {"Intensity": 0.4, "Size": 16, "Threshold": 2.2, "Enabled": True},
                             "ColorCorrectionEffect": {"Brightness": 0, "Contrast": 0.05, "Saturation": 0,
                                                       "TintColor": [1, 0.96, 0.9], "Enabled": True}}},
}


def _effect_check(classes):
    cat, err = T._catalog()
    if err:
        return err
    for cls in classes:
        if cls not in cat.classes:
            return cat.unknown_class(cls)
        if not cat.is_a(cls, "PostEffect"):
            return {"ok": False, "ran": False, "refusedRule": "type", "error": "%s is not a PostEffect" % cls}
        bad = cat.create_refusal(cls)
        if bad:
            return bad
    return None


SUCCESSORS = ("LightingStyle", "PrioritizeLightingQuality")   # Technology's replacements on Studio 0.748 (RB2112)


def technology_state():
    """What this Studio offers for the lighting technology, from the catalog: successors (LightingStyle
    and PrioritizeLightingQuality, plugin-writable) when it has them - Studio 0.748 shows no Technology in Properties at
    all (RB2112) - and otherwise Technology, RobloxScriptSecurity, which only the person can set."""
    cat, err = T._catalog()
    if err:
        return None, err
    have = [n for n in SUCCESSORS if (cat.find_member("Lighting", n)[1] or [None, None, None, None])[3] == "None"]
    row = cat.find_member("Lighting", "Technology")[1]
    out = {"successors": have == list(SUCCESSORS), "technologySecurity": row[2] if row else None,
           "technologyDeprecated": bool(row and "Deprecated" in (row[4] or []))}
    if out["successors"]:
        out["note"] = ("this Studio replaces Lighting.Technology with LightingStyle and PrioritizeLightingQuality, both "
                       "set and read back by tool (style, prioritize_quality); Technology is not a step for anyone (RB2112)")
    else:
        out["note"] = ("this Studio has no LightingStyle: Lighting.Technology is %s, so a plugin can neither read nor set "
                       "it" % out["technologySecurity"])
    return out, None


@T._reports
def rbx_lighting_set(preset: str = None, lighting: dict = None, atmosphere: dict = None, sky: dict = None,
                     clouds: dict = None, effects: dict = None, style: str = None, prioritize_quality: bool = None,
                     technology: str = None) -> dict:
    "Roblox Studio lighting as one described look: preset day, dusk, night or interior (starting values chosen by eye, every one an ordinary property; each sets LightingStyle Realistic and PrioritizeLightingQuality true), then any overrides: style (Lighting.LightingStyle, Realistic or Soft) and prioritize_quality (Lighting.PrioritizeLightingQuality), lighting {Lighting property: value - the scales EnvironmentDiffuseScale and EnvironmentSpecularScale, ExposureCompensation...}, atmosphere and sky (an Atmosphere and a Sky under Lighting, made where missing), clouds (a Clouds under Workspace.Terrain), effects {PostEffect class: props} under Lighting (BloomEffect, ColorCorrectionEffect, SunRaysEffect, DepthOfFieldEffect...). Every property read back; the answer lists each part's verdict and technology, what this Studio offers: on a Studio with LightingStyle and PrioritizeLightingQuality - 0.748 shows no Technology at all and a new place defaults to Realistic with quality prioritized (RB2112) - the look is set wholly by tool and technology (a Technology name) is refused with that route; only on an older Studio without them is technology named as the person's step (technology.personStep), since Lighting.Technology is RobloxScriptSecurity. Writes outside MifScratch: Full mode, or MifBridge's own scratch place."
    if preset is not None and preset not in PRESETS:
        return {"ok": False, "ran": False, "error": "preset is one of %s" % ", ".join(PRESETS)}
    state, err = technology_state()
    if err:
        return err
    cat = T._catalog()[0]
    if isinstance(lighting, dict) and "Technology" in lighting:
        return {"ok": False, "ran": False, "refusedRule": "security", "technology": state,
                "error": "Lighting.Technology is %s: no plugin can set it. %s" % (
                    state["technologySecurity"], "Use style and prioritize_quality" if state["successors"]
                    else "Pass technology and the answer names it as the person's step")}
    if technology is not None:
        items = cat.enums.get("Technology") or []
        if technology not in items:
            return {"ok": False, "ran": False, "error": "technology is one of %s (Enum.Technology)" % ", ".join(items)}
        if state["successors"]:
            return {"ok": False, "ran": False, "refusedRule": "replaced", "technology": state,
                    "error": "this Studio has no Lighting.Technology to set (RB2112): it is LightingStyle (style Realistic "
                             "or Soft) and PrioritizeLightingQuality (prioritize_quality) now, both set by this tool"}
        state = dict(state, want=technology, personStep="In Studio, select Lighting in the Explorer and set Technology "
                     "to %s in the Properties window (a person's step: no tool can do it on this Studio)" % technology)
    if style is not None and style not in (cat.enums.get("LightingStyle") or []):
        return {"ok": False, "ran": False, "error": "style is one of %s (Enum.LightingStyle)" % ", ".join(
            cat.enums.get("LightingStyle") or ["(none on this Studio)"])}
    if prioritize_quality is not None and not isinstance(prioritize_quality, bool):
        return {"ok": False, "ran": False, "error": "prioritize_quality is true or false"}
    if (style is not None or prioritize_quality is not None) and not state["successors"]:
        return {"ok": False, "ran": False, "refusedRule": "unknown-member", "technology": state,
                "error": "this Studio has no LightingStyle or PrioritizeLightingQuality: %s" % state["note"]}
    if technology is not None and preset is None and not any((lighting, atmosphere, sky, clouds, effects)):
        return {"ok": True, "ran": False, "sent": False, "technology": state,
                "note": "nothing to send: Technology is the person's step on this Studio"}
    look = {k: dict(v) for k, v in (PRESETS.get(preset) or {}).items() if k != "effects"}
    fx = {k: dict(v) for k, v in ((PRESETS.get(preset) or {}).get("effects") or {}).items()}
    if preset is not None and state["successors"]:
        look.setdefault("lighting", {}).update(LightingStyle="Realistic", PrioritizeLightingQuality=True)
    for key, extra in (("lighting", lighting), ("atmosphere", atmosphere), ("sky", sky), ("clouds", clouds)):
        if extra is not None:
            if not isinstance(extra, dict):
                return {"ok": False, "ran": False, "error": "%s is an object of property to value" % key}
            look.setdefault(key, {}).update(extra)
    if style is not None:
        look.setdefault("lighting", {})["LightingStyle"] = style
    if prioritize_quality is not None:
        look.setdefault("lighting", {})["PrioritizeLightingQuality"] = prioritize_quality
    for k, v in (effects or {}).items():
        if not isinstance(v, dict):
            return {"ok": False, "ran": False, "error": "effects is {PostEffect class: props}"}
        fx.setdefault(k, {}).update(v)
    bad = _effect_check(fx)
    if bad:
        return bad
    if not look and not fx:
        return {"ok": False, "ran": False, "error": "give a preset or something to set"}
    out, problems = {}, []
    if look.get("lighting"):
        r = T.rbx_set_props("Lighting", look["lighting"])
        out["lighting"] = {k: r.get(k) for k in ("ok", "verified", "mismatch", "error", "refused", "refusedRule")}
        if not r.get("ok"):
            problems.append("Lighting: %s" % r.get("error"))
    for key, cls, parent in (("atmosphere", "Atmosphere", "Lighting"), ("sky", "Sky", "Lighting"),
                             ("clouds", "Clouds", "Workspace.Terrain")):
        if key in look:
            r = _find_or_create(parent, cls, look[key])
            out[key] = {k: r.get(k) for k in ("ok", "verified", "path", "created", "mismatch", "error", "refused", "refusedRule")}
            if not r.get("ok"):
                problems.append("%s: %s" % (cls, r.get("error")))
    for cls, props in fx.items():
        r = _find_or_create("Lighting", cls, props)
        out.setdefault("effects", {})[cls] = {k: r.get(k) for k in ("ok", "verified", "path", "created", "mismatch", "error",
                                                                    "refused", "refusedRule")}
        if not r.get("ok"):
            problems.append("%s: %s" % (cls, r.get("error")))
    out.update(ok=not problems, verified=not problems, preset=preset, technology=state)
    if problems:
        out["error"] = "; ".join(problems)
    return out


def _vec_of(d):
    return [d.get("x"), d.get("y"), d.get("z")] if isinstance(d, dict) else None


@T._reports
def rbx_time_of_day(clock_time: float = None, minutes: float = None, latitude: float = None) -> dict:
    "Roblox Studio time of day: clock_time (hours, Lighting.ClockTime) or minutes after midnight (Lighting:SetMinutesAfterMidnight), and latitude (Lighting.GeographicLatitude, degrees). Read back: Lighting:GetMinutesAfterMidnight, and Lighting:GetSunDirection with the sun's elevation and its angle from straight up. With nothing given, reads only."
    if clock_time is not None and minutes is not None:
        return {"ok": False, "ran": False, "error": "give clock_time or minutes, not both"}
    if clock_time is not None and not (_num(clock_time) and 0 <= clock_time < 24):
        return {"ok": False, "ran": False, "error": "clock_time is hours, 0 to below 24"}
    if minutes is not None and not (_num(minutes) and 0 <= minutes < 1440):
        return {"ok": False, "ran": False, "error": "minutes is 0 to below 1440"}
    if latitude is not None and not (_num(latitude) and -90 <= latitude <= 90):
        return {"ok": False, "ran": False, "error": "latitude is -90 to 90 degrees"}
    out, problems = {}, []
    props = {}
    if clock_time is not None:
        props["ClockTime"] = clock_time
    if latitude is not None:
        props["GeographicLatitude"] = latitude
    if props:
        r = T.rbx_set_props("Lighting", props)
        out["set"] = {k: r.get(k) for k in ("ok", "verified", "mismatch", "error", "refusedRule")}
        if not r.get("ok"):
            return dict(out, ok=False, error=r.get("error"), refusedRule=r.get("refusedRule"))
    if minutes is not None:
        r = T.rbx_call("Lighting", "SetMinutesAfterMidnight", [minutes])
        if not r.get("ok"):
            return dict(r, ok=False)
    got = T.rbx_call("Lighting", "GetMinutesAfterMidnight")
    sun = T.rbx_call("Lighting", "GetSunDirection")
    if not got.get("ok") or not sun.get("ok"):
        return dict(out, ok=False, error=(got if not got.get("ok") else sun).get("error"))
    mins, d = got.get("value"), _vec_of(sun.get("value"))
    want = minutes if minutes is not None else (clock_time * 60 if clock_time is not None else None)
    if want is not None and not (_num(mins) and abs(mins - want) <= 0.01):
        problems.append("GetMinutesAfterMidnight reads %s, set %s" % (mins, want))
    elev = None
    if d and all(_num(x) for x in d):
        ln = math.sqrt(sum(x * x for x in d)) or 1
        elev = math.degrees(math.asin(max(-1.0, min(1.0, d[1] / ln))))
    out.update(ok=not problems, verified=bool(want is not None or props) and not problems, minutes=mins,
               clockTime=None if not _num(mins) else round(mins / 60, 4), sunDirection=d,
               sunElevation=None if elev is None else round(elev, 4),
               sunFromZenith=None if elev is None else round(90 - elev, 4))
    if problems:
        out["error"] = "; ".join(problems)
    return out


def _anchor(anchor):
    """The part attachments hang from: anchor, or Workspace.MifScratch.MifAnchor (made: anchored, invisible, no collision)."""
    if anchor is not None:
        return V.split_path(anchor), None
    folder = _find_or_create("Workspace", "Folder", None, name="MifScratch")
    if not folder.get("ok"):
        return None, folder
    a = _find_or_create("Workspace.MifScratch", "Part", {"Anchored": True, "Transparency": 1, "CanCollide": False,
                                                          "CFrame": {"position": [0, 0, 0]}, "Size": [1, 1, 1]},
                        name="MifAnchor")
    if not a.get("ok"):
        return None, a
    return ["Workspace", "MifScratch", "MifAnchor"], None


def _attachment_at(parent, point, name):
    """A new Attachment under parent with its WorldCFrame at point; read back."""
    a = T.rbx_create("Attachment", parent, name=T._unique_name(V.split_path(parent), name))
    if not a.get("ok"):
        return None, a
    s = T.rbx_set_props(a["path"], {"WorldCFrame": {"position": list(point)}})
    if not s.get("ok"):
        d = T.rbx_destroy(a["path"])          # not at the point: not left behind either
        return None, dict(s, path=a["path"], attachmentRemoved=bool(d.get("ok")),
                          error="%s; the Attachment %s" % (s.get("error"), "was removed" if d.get("ok") else
                                                           "stays at %s (%s)" % (a["path"], d.get("error"))))
    return a["path"], None


_LIGHTS = ("PointLight", "SpotLight", "SurfaceLight")


@T._reports
def rbx_light_place(lights: list, anchor: Any = None) -> dict:
    "Roblox Studio lights at world points: lights is a list of {kind: PointLight, SpotLight or SurfaceLight, position: [x, y, z], props: {Brightness, Color, Range, Angle, Face, Shadows...}, name}. Each gets a new Attachment under anchor (a BasePart; default Workspace.MifScratch.MifAnchor, made anchored and invisible) with its WorldCFrame at the point, and the light under it. Read back: the attachment's WorldCFrame and every light property. A light that cannot be made has its Attachment destroyed again; if that fails too, the error names the Attachment that stays."
    if not isinstance(lights, list) or not lights or len(lights) > 100:
        return {"ok": False, "ran": False, "error": "lights is a list of 1-100 {kind, position, props, name}"}
    for i, L in enumerate(lights):
        if not isinstance(L, dict) or L.get("kind") not in _LIGHTS or not _vec(L.get("position")):
            return {"ok": False, "ran": False, "error": "light %d needs kind (%s) and position [x, y, z]" % (i, ", ".join(_LIGHTS))}
    parent, err = _anchor(anchor)
    if err:
        return err
    out, problems = [], []
    for i, L in enumerate(lights):
        name = L.get("name") or "MifLight%d" % (i + 1)
        apath, err = _attachment_at(parent, L["position"], name)
        if err:
            problems.append("%s: %s" % (name, err.get("error")))
            out.append({"name": name, "ok": False, "error": err.get("error"), "refusedRule": err.get("refusedRule")})
            continue
        r = T.rbx_create(L["kind"], apath, name=L["kind"], props=L.get("props") or None)
        row = {"name": name, "attachment": apath, "light": r.get("path"), "ok": r.get("ok"), "error": r.get("error")}
        if not r.get("ok"):
            # The light was not made: its Attachment goes too, so a failed light leaves nothing behind.
            d = T.rbx_destroy(apath)
            row["attachmentRemoved"] = bool(d.get("ok"))
            if d.get("ok"):
                problems.append("%s: %s (its Attachment was removed)" % (name, r.get("error")))
            else:
                problems.append("%s: %s; its Attachment stays at %s (%s)" % (
                    name, r.get("error"), apath if isinstance(apath, str) else ".".join(map(str, apath or [])),
                    d.get("error")))
        out.append(row)
    res = {"ok": not problems, "verified": not problems, "anchor": V.join_path(parent), "lights": out}
    if problems:
        res["error"] = "; ".join(problems)
    return res


@T._reports
def rbx_particles_set(target: Any = None, parent: Any = None, name: str = "MifEmitter", props: dict = None,
                      burst: int = 0, clear: bool = False, capture: dict = None) -> dict:
    "Roblox Studio particles: set a ParticleEmitter's properties (target), or make one under parent (a BasePart or Attachment) - Size, Transparency and Squash as NumberSequence keys [[time, value, envelope?], ...] (2-20, times 0 to 1), Color as ColorSequence keys [[time, r, g, b], ...], Lifetime, Speed and Rotation as [min, max], Rate, Texture, the Flipbook settings. Every property read back, sequences key by key to 1e-4. Then clear (ParticleEmitter:Clear) and a preview burst of burst particles (ParticleEmitter:Emit), and with capture {position, look_at} a screenshot of it. FastForward is RobloxScriptSecurity, so a burst is the preview."
    if (target is None) == (parent is None):
        return {"ok": False, "ran": False, "error": "give target (an emitter) or parent (to make one), not both"}
    if not isinstance(burst, int) or isinstance(burst, bool) or not 0 <= burst <= 1000:
        return {"ok": False, "ran": False, "error": "burst is 0-1000 particles"}
    if target is not None:
        path = V.split_path(target)
        r = T.rbx_set_props(path, props) if props else {"ok": True, "verified": True}
    else:
        r = T.rbx_create("ParticleEmitter", parent, name=T._unique_name(V.split_path(parent), name), props=props or None)
        path = V.split_path(r.get("path")) if r.get("ok") else None
    out = {"ok": bool(r.get("ok")), "verified": bool(r.get("verified")), "path": V.join_path(path),
           "props": {k: r.get(k) for k in ("after", "mismatch", "error", "refused", "refusedRule") if r.get(k) is not None}}
    if not r.get("ok"):
        return dict(out, error=r.get("error"), refusedRule=r.get("refusedRule"))
    if clear:
        c = T.rbx_call(path, "Clear")
        out["cleared"] = bool(c.get("ok"))
    if burst:
        e = T.rbx_call(path, "Emit", [burst])
        out["burst"] = {"ok": bool(e.get("ok")), "count": burst, "error": e.get("error")}
        if not e.get("ok"):
            out.update(ok=False, error="Emit: %s" % e.get("error"))
    if capture is not None:
        s = T.rbx_screenshot(camera=capture, name="particles")
        out["screenshot"] = {k: s.get(k) for k in ("ok", "path", "file", "blank", "error") if k in s}
    return out


def _assembly_root(attachment):
    """The AssemblyRootPart of the part an attachment hangs from, or None when it is not under a BasePart."""
    parts = V.split_path(attachment)
    if not parts or len(parts) < 2:
        return None
    r = T.rbx_get_props(parts[:-1], ["AssemblyRootPart"])
    return (r.get("values") or {}).get("AssemblyRootPart") if r.get("ok") else None


@T._reports
def rbx_beam_trail(kind: str, parent: Any, points: list = None, attachments: list = None, name: str = None,
                   props: dict = None) -> dict:
    "Roblox Studio beams and trails between two attachments: kind beam or trail; attachments [path0, path1], or points [[x, y, z], [x, y, z]] for two new attachments under parent (a BasePart) at those world points. The Beam or Trail goes under parent with Attachment0 and Attachment1 and props (Beam CurveSize0/1, Width0/1, Segments, Color, Transparency, Texture...; Trail Lifetime, WidthScale, MinLength, MaxLength...). A trail whose attachments are in different assemblies (BasePart.AssemblyRootPart) is refused: it would stretch between bodies. Every property read back."
    cls = {"beam": "Beam", "trail": "Trail"}.get(kind)
    if cls is None:
        return {"ok": False, "ran": False, "error": "kind is beam or trail"}
    if (points is None) == (attachments is None):
        return {"ok": False, "ran": False, "error": "give points (two world positions) or attachments (two paths)"}
    pp, err = T._target(parent, "parent")
    if err:
        return err
    made = []
    if points is not None:
        if not isinstance(points, list) or len(points) != 2 or not all(_vec(p) for p in points):
            return {"ok": False, "ran": False, "error": "points is two [x, y, z]"}
        base = name or cls
        for i, p in enumerate(points):
            path, err = _attachment_at(pp, p, "%s_A%d" % (base, i))
            if err:
                return dict(err, ok=False, made=made)
            made.append(path)
        attachments = made
    if not isinstance(attachments, list) or len(attachments) != 2:
        return {"ok": False, "ran": False, "error": "attachments is two paths"}
    if cls == "Trail":
        r0, r1 = _assembly_root(attachments[0]), _assembly_root(attachments[1])
        if r0 is None or r0 != r1:
            return {"ok": False, "ran": False, "refusedRule": "assembly", "roots": [r0, r1], "made": made,
                    "error": "a trail's attachments must be in one assembly (AssemblyRootPart %s and %s)" % (r0, r1)}
    full = dict(props or {}, Attachment0=attachments[0], Attachment1=attachments[1])
    r = T.rbx_create(cls, pp, name=T._unique_name(pp, name or cls), props=full)
    out = {k: r.get(k) for k in ("ok", "verified", "path", "after", "mismatch", "error", "refused", "refusedRule")}
    out["attachments"] = [V.join_path(V.split_path(a)) for a in attachments]
    return out


@T._reports
def rbx_post_effects(effects: dict, parent: Any = "Lighting") -> dict:
    "Roblox Studio post effects: effects {class: props} for BloomEffect, BlurEffect, ColorCorrectionEffect, DepthOfFieldEffect, SunRaysEffect, ColorGradingEffect (TonemapperPreset) and any other PostEffect, under parent (Lighting, or a Camera). The first of each class there is set, or a new one made. Every property read back. A class that is not a PostEffect is refused."
    if not isinstance(effects, dict) or not effects or not all(isinstance(v, dict) for v in effects.values()):
        return {"ok": False, "ran": False, "error": "effects is {PostEffect class: props}"}
    bad = _effect_check(effects)
    if bad:
        return bad
    pp, err = T._target(parent, "parent")
    if err:
        return err
    out, problems = {}, []
    for cls, props in effects.items():
        r = _find_or_create(V.join_path(pp), cls, props)
        out[cls] = {k: r.get(k) for k in ("ok", "verified", "path", "created", "after", "mismatch", "error", "refusedRule")}
        if not r.get("ok"):
            problems.append("%s: %s" % (cls, r.get("error")))
    res = {"ok": not problems, "verified": not problems, "parent": V.join_path(pp), "effects": out}
    if problems:
        res["error"] = "; ".join(problems)
    return res


@T._reports
def rbx_surface_look(target: Any, highlight: dict = None, decals: list = None, textures: list = None) -> dict:
    "Roblox Studio looks on a part: highlight {FillColor, FillTransparency, OutlineColor, OutlineTransparency, DepthMode, Enabled} as a Highlight under target with Adornee target; decals and textures lists of {face: Right, Top, Back, Left, Bottom or Front, name, props: {ColorMapContent, NormalMapContent, MetalnessMapContent, RoughnessMapContent, Color3, Transparency, UVScale, UVOffset; Texture StudsPerTileU/V, OffsetStudsU/V}}. Content is an asset uri or {handle} of an EditableImage; an rbxassetid that names a Decal (what an image upload answers) is swapped for the image inside it. Every property read back."
    tp, err = T._target(target)
    if err:
        return err
    if highlight is None and not decals and not textures:
        return {"ok": False, "ran": False, "error": "give highlight, decals or textures"}
    out, problems = {}, []
    if highlight is not None:
        if not isinstance(highlight, dict):
            return {"ok": False, "ran": False, "error": "highlight is an object of Highlight properties"}
        r = T.rbx_create("Highlight", tp, name=T._unique_name(tp, "MifHighlight"), props=dict(highlight, Adornee=tp))
        out["highlight"] = {k: r.get(k) for k in ("ok", "verified", "path", "mismatch", "error", "refusedRule")}
        if not r.get("ok"):
            problems.append("Highlight: %s" % r.get("error"))
    for cls, items in (("Decal", decals), ("Texture", textures)):
        for i, d in enumerate(items or []):
            if not isinstance(d, dict) or d.get("face") not in ("Right", "Top", "Back", "Left", "Bottom", "Front"):
                return {"ok": False, "ran": False, "error": "a %s is {face, name, props}; face is a NormalId" % cls.lower()}
            dprops, swaps, ierr = T._image_props(d.get("props") or {})
            if ierr:
                return {"ok": False, "ran": False, "error": "%s %d: %s" % (cls, i + 1, ierr)}
            r = T.rbx_create(cls, tp, name=T._unique_name(tp, d.get("name") or "Mif%s%d" % (cls, i + 1)),
                             props=dict(dprops, Face=d["face"]))
            out.setdefault(cls.lower() + "s", []).append(dict({k: r.get(k) for k in ("ok", "verified", "path", "mismatch",
                                                                                      "error", "refusedRule")},
                                                              **({"decalSwaps": swaps} if swaps else {})))
            if not r.get("ok"):
                problems.append("%s %d: %s" % (cls, i + 1, r.get("error")))
    out.update(ok=not problems, verified=not problems, target=V.join_path(tp))
    if problems:
        out["error"] = "; ".join(problems)
    return out


RENDER_PROPS = ["QualityLevel", "EditQualityLevel", "ViewMode", "MeshPartDetailLevel", "GraphicsMode", "FrameRateManager",
                "ShowBoundingBoxes"]


def rbx_render_settings_get(set: dict = None) -> dict:  # noqa: A002 - the verb a caller writes, refused
    "Roblox Studio's own render settings, read: RenderSettings QualityLevel, EditQualityLevel, ViewMode, MeshPartDetailLevel, GraphicsMode, FrameRateManager, ShowBoundingBoxes and GetMaxQualityLevel. These are the person's Studio preferences, not the place: set is refused studio-preference until the person decides which Studio preferences an agent may change."
    if set is not None:
        return {"ok": False, "ran": False, "refusedRule": "studio-preference",
                "error": "RenderSettings change the person's Studio, not the place; writes wait on the person's decision about "
                         "Studio preferences. Nothing was sent"}
    # RenderSettings lives under settings(), not game (RB311): plugin 0.15.0 resolves a path rooted at "@settings".
    route = ["@settings", "RenderSettings"]
    r = T._rbx("get_props", target=route, props=RENDER_PROPS)
    if not r.get("ok"):
        route = ["RenderSettings"]
        r2 = T._rbx("get_props", target=route, props=RENDER_PROPS)
        if not r2.get("ok"):
            return dict(r, ok=False, route="RenderSettings lives under settings(), not game; the plugin resolves it from "
                        "the @settings root from 0.15.0", fromGame=r2.get("error"))
        r = r2
    mx = _call("GetMaxQualityLevel", [], True, target=route, cls="RenderSettings")
    return {"ok": True, "route": route, "values": {k: V.decode(v) for k, v in T._obj(r.get("values")).items()},
            "maxQualityLevel": (mx.get("returns") or [None])[0] if mx.get("ok") else None,
            "maxQualityError": None if mx.get("ok") else mx.get("error")}


__all__ = sorted(n for n in globals() if n.startswith("rbx_") and callable(globals()[n]))
