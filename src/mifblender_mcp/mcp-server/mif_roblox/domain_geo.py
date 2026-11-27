"""Instances and parts: pivots, model scale, CSG, welds, layout, spatial queries, bounds,
a place audit and fragments. Composites on the MCP side over the generic plugin ops (call, get_props, set_props,
reparent, destroy, batch, query), every write read back.

Loaded by tools.py with one star import, so these rbx_* tools join TOOLS and PACK like the rest. This module also
EXTENDS four tools.py tools to {handle} targets - rbx_reparent, rbx_set_props, rbx_destroy and rbx_release_handle. Each
extension handles the {handle} case itself and hands a path to the tools.py function unchanged (captured here as
_ORIG), so the path behaviour has one home. The star import rebinds the four names in tools.py to these.

WIRE VALUES BUILT HERE. GeometryService's part lists are typed `Array` in the dump, and the codec refuses an Instance
inside an Array or a Variant (FOUND 2026-11-19, sent to R2), so the GeometryService calls build their wire values here
and go through _wcall, which checks the member against the catalog first exactly as rbx_call does.

GEOMETRYSERVICE CALLS ARE SENT AS READS. Union, subtract, intersect, negate, separate, bake, fragment and the fragment
sites make new objects outside the place and change nothing in it (measured live on 0.748: the results' Parent is nil,
RB700), so they cross with readOnly true and are not recorded; what enters the place is the reparent of each result,
which is recorded in one undo group with any rename and with the inputs' removal when replace is asked.

READ MANY. A round trip costs about 200 ms (RB10). Where a tool reads many targets it uses the plugin's read_many op
(PLUGIN_REQUESTS R1-1) when the plugin has it and one call per target otherwise, and answers which ("route").

NEVER RAISE: every failure is {"ok": False, ...}.
"""
import functools
import math
import random
import re
from typing import Any

from mif_roblox import tools as T  # noqa: E402 - tools.py imports this module after its own tools; T has them by then
from mif_roblox import values as V  # noqa: E402

__all__ = []

# tools.py's own functions for a path target: the extensions below hand every path to these.
EXTENDED = ("rbx_reparent", "rbx_set_props", "rbx_destroy", "rbx_release_handle")
_ORIG = {n: getattr(T, n) for n in EXTENDED}

SCRATCH = ["Workspace", "MifScratch"]
SCRATCH_PLACE = "MifScratch.rbxlx"          # the plugin's own constant: in that place the whole of it is scratch
MAX_TARGETS = 2000                          # rbx_pivot, rbx_bounds
MAX_COPIES = 2000                           # rbx_layout
MAX_LINT = 20000                            # rbx_place_lint
MANY_CHUNK = 1000                           # items per read_many op
POS_TOL = 1e-4                              # pivots read back to 1e-4 (spec) near the origin; further out, F32_STEPS
F32_STEPS = 8                               # float steps a read-back may be off: Studio keeps CFrames in 32-bit floats,
                                            # 1.8e-4 at y 883 (3 steps, RB1605), half a step out at 24,576 and 100,000 (RB1750)
MIN_PART = 0.001                            # studs: the smallest side Studio shows; a smaller Size or ScaleTo stops there (RB1751)
MIN_EXTENT = 0.05                           # studs: Model:GetExtentsSize counts a part as at least this on each own axis (RB1751)
BOX_TOL = 0.01                              # studs: a CSG result's box against what its inputs allow
VOLUME_TOL = 0.01                           # rbx_fragment: summed volume within 1%
CSG_OPTIONS = {"CollisionFidelity": "Enum.CollisionFidelity", "RenderFidelity": "Enum.RenderFidelity",
               "FluidFidelity": "Enum.FluidFidelity", "SplitApart": "bool"}   # Roblox's GeometryService page
SITE_OPTIONS = {"SiteSpacing": "float", "Origin": "Vector3", "Radius": "float"}  # GenerateFragmentSites, same page
_MANY = {"ok": None}
# A part's world box is BasePart.ExtentsCFrame and ExtentsSize, not CFrame and Size: a GeometryService result keeps its
# first input's CFrame while its geometry is off centre (measured: a union spanning x -2..5 reads CFrame x 0, Size 7,
# ExtentsCFrame x 1.5, RB703).
EXTENTS = ["ExtentsCFrame", "ExtentsSize"]                        # None until measured: does this plugin have read_many


def _tool(fn):
    """Export fn, never raise, and post its verdict to the panel's journal as tools._reports does."""
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            out = fn(*a, **kw)
        except Exception as exc:  # noqa: BLE001 - never raise: a bug answers as a failure with its text
            out = {"ok": False, "error": "%s failed inside the bridge: %s: %s" % (fn.__name__, type(exc).__name__, exc)}
        T._post_verdict(fn.__name__, out)
        return out
    __all__.append(fn.__name__)
    return wrapper


# ---------------------------------------------------------------- small helpers
def _no(error, **kw):
    return dict({"ok": False, "ran": False, "error": error}, **kw)


def _num(x):
    return V._num(x)


def _vec(x, n=3):
    return isinstance(x, (list, tuple)) and len(x) == n and all(_num(v) for v in x)


def _handle(x):
    return isinstance(x, dict) and set(x) == {"handle"} and isinstance(x["handle"], str) and bool(x["handle"])


def _parts(x, what="target"):
    p = V.split_path(x)
    if not p:
        return None, _no("%s is a dot path from game ('Workspace.Part') or a list of names, got %r" % (what, x),
                         refusedRule="type")
    return p, None


def _paths(xs, what, lo=1, hi=MAX_TARGETS):
    if not isinstance(xs, list) or not lo <= len(xs) <= hi:
        return None, _no("%s is a list of %d-%d paths" % (what, lo, hi))
    out = []
    for x in xs:
        p, err = _parts(x, what)
        if err:
            return None, err
        out.append(p)
    if len({tuple(p) for p in out}) != len(out):
        return None, _no("%s names one instance twice" % what)
    return out, None


def _inst(parts):
    return {"$t": "Instance", "path": list(parts)}


def _wv3(v):
    return {"$t": "Vector3", "v": [float(x) for x in v]}


def _wcf(cf):
    return {"$t": "CFrame", "v": list(cf["position"]) + list(cf["rotation"])}


def _cf(w):
    """A wire or decoded CFrame -> {position, rotation}, or None."""
    if isinstance(w, dict) and w.get("$t") == "CFrame" and isinstance(w.get("v"), list) and len(w["v"]) == 12:
        return {"position": list(w["v"][:3]), "rotation": list(w["v"][3:])}
    if isinstance(w, dict) and _vec(w.get("position")) and _vec(w.get("rotation"), 9):
        return {"position": list(w["position"]), "rotation": list(w["rotation"])}
    return None


def _v3(w):
    if isinstance(w, dict) and w.get("$t") == "Vector3" and _vec(w.get("v")):
        return list(w["v"])
    if isinstance(w, dict) and all(k in w for k in "xyz"):
        return [w["x"], w["y"], w["z"]]
    return list(w) if _vec(w) else None


IDENT = [1, 0, 0, 0, 1, 0, 0, 0, 1]


def _mm(a, b):
    return [sum(a[3 * i + k] * b[3 * k + j] for k in range(3)) for i in range(3) for j in range(3)]


def _mv(r, v):
    return [r[3 * i] * v[0] + r[3 * i + 1] * v[1] + r[3 * i + 2] * v[2] for i in range(3)]


def _tr(r):
    return [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]]


def cf_mul(a, b):
    return {"position": [p + q for p, q in zip(a["position"], _mv(a["rotation"], b["position"]))],
            "rotation": _mm(a["rotation"], b["rotation"])}


def cf_inv(a):
    rt = _tr(a["rotation"])
    return {"position": [-x for x in _mv(rt, a["position"])], "rotation": rt}


def euler_xyz(deg):
    """CFrame.fromEulerAnglesXYZ(rx, ry, rz) in degrees: R = Rx * Ry * Rz (Roblox's CFrame.Angles)."""
    x, y, z = (math.radians(d) for d in deg)
    rx = [1, 0, 0, 0, math.cos(x), -math.sin(x), 0, math.sin(x), math.cos(x)]
    ry = [math.cos(y), 0, math.sin(y), 0, 1, 0, -math.sin(y), 0, math.cos(y)]
    rz = [math.cos(z), -math.sin(z), 0, math.sin(z), math.cos(z), 0, 0, 0, 1]
    return _mm(_mm(rx, ry), rz)


def cf_err(a, b):
    """The largest difference between two CFrames' twelve numbers."""
    return max(abs(x - y) for x, y in zip(a["position"] + a["rotation"], b["position"] + b["rotation"]))


def f32_step(x):
    """The gap between neighbouring 32-bit floats at |x| (1's gap below 1): the finest position Studio holds there."""
    m = abs(float(x))
    return 2.0 ** (max(math.floor(math.log2(m)), 0) - 23) if m >= 1 else 2.0 ** -23


def pivot_tol(*cfs):
    """A pivot's read-back tolerance: POS_TOL, or F32_STEPS float steps at the largest coordinate involved, whichever is
    more. At y 883 that is 4.9e-4 (RB1605 read back 1.8e-4 off); at 24,576 0.016, at 100,000 0.0625."""
    m = max([abs(v) for cf in cfs if cf for v in cf["position"]] + [0.0])
    return max(POS_TOL, F32_STEPS * f32_step(m))


def aabb(cf, size):
    """World axis-aligned box [min, max] of an oriented box."""
    r, p = cf["rotation"], cf["position"]
    half = [abs(r[3 * i]) * size[0] / 2 + abs(r[3 * i + 1]) * size[1] / 2 + abs(r[3 * i + 2]) * size[2] / 2
            for i in range(3)]
    return [[p[i] - half[i] for i in range(3)], [p[i] + half[i] for i in range(3)]]


def _box_union(boxes):
    return [[min(b[0][i] for b in boxes) for i in range(3)], [max(b[1][i] for b in boxes) for i in range(3)]]


def _box_inter(boxes):
    return [[max(b[0][i] for b in boxes) for i in range(3)], [min(b[1][i] for b in boxes) for i in range(3)]]


def _within(inner, outer, tol=BOX_TOL):
    return all(inner[0][i] >= outer[0][i] - tol and inner[1][i] <= outer[1][i] + tol for i in range(3))


def _same_box(a, b, tol=BOX_TOL):
    return all(abs(a[k][i] - b[k][i]) <= tol for k in range(2) for i in range(3))


def _r(x, n=4):
    if isinstance(x, list):
        return [_r(v, n) for v in x]
    return round(x, n) if _num(x) else x


def _cat():
    cat, err = T._catalog()
    return cat, err


def _is_a(cat, cls, base):
    return cat.is_a(cls, base)


def _ping():
    return T._rbx("ping", nonce="geo")


def _scratch_refusal(ping, paths, what):
    """In scratch mode (outside MifBridge's own scratch place) every path named must be under Workspace.MifScratch; read
    mode refuses any write. None when allowed."""
    mode = ping.get("mode")
    if mode == "full" or (mode == "scratch" and ping.get("placeName") == SCRATCH_PLACE):
        return None
    if mode == "read":
        return _no("%s is a write and the panel is in read mode: nothing was done" % what, refusedRule="read-mode",
                   mode=mode)
    outside = [V.path_text(p) for p in paths if list(p[:2]) != SCRATCH or len(p) < 2]
    if outside:
        return _no("%s would touch %s, outside Workspace.MifScratch, and the panel is in scratch mode: nothing was done. "
                   "Raise the mode in the MifBridge panel" % (what, ", ".join(outside[:10])),
                   refusedRule="scratch-path", mode=mode, outside=outside[:50])
    return None


def _wcall(target, cls, method, args, read_only=None, timeout=60):
    """A member call with wire arguments built here, checked against the catalog as rbx_call checks it: reachable,
    not needing a function, no more arguments than it takes. The plugin's raw reply (returns as wire values)."""
    cat, err = _cat()
    if err:
        return err
    bad = cat.read_refusal(cls, method, kind="F")
    if bad:
        return bad
    owner, row = cat.find_member(cls, method)
    if any(p[1] == "Function" for p in row[5]):
        return _no("%s:%s takes a Luau function" % (owner, method), refusedRule="needs-function")
    if len(args) > len(row[5]):
        return _no("%s:%s takes %d arguments, got %d" % (owner, method, len(row[5]), len(args)), refusedRule="type")
    ro = T._read_only(method, owner) if read_only is None else read_only
    return T._rbx("call", timeout=timeout + 15, target=target, method=method, args=args, argc=len(args), readOnly=ro,
                  yields="Yields" in row[4], callTimeout=timeout)


def _ret(r, i=0):
    rs = r.get("returns") or []
    return rs[i] if len(rs) > i else None


def _items(w):
    """An Array, Instances or Tuple wire value's elements."""
    if isinstance(w, dict) and isinstance(w.get("v"), list):
        return w["v"]
    return w if isinstance(w, list) else []


def _key(x):
    """A decoded path ("A.B", or the list join_path keeps for a dotted name) as a tuple, for a dict key or a set; None
    for anything that is not a path."""
    parts = V.split_path(x) if isinstance(x, (str, list)) else (list(x) if isinstance(x, tuple) else None)
    return tuple(parts) if parts is not None else None


def _many_row(x):
    """One read_many row in the per-target route's shape. The plugin (0.16.0) keeps a bad item's failure in the row: a
    property it could not read under errors, a call that failed or was fenced as a {"$t": "error"/"refused"} map in
    returns. The per-target route fails the whole row for either, so this does too."""
    if not isinstance(x, dict):
        return {"error": "read_many answered %r for an item" % (x,), "refusedRule": "failed"}
    if x.get("error") is not None or x.get("refusedRule") is not None:
        return {"error": x.get("error"), "refusedRule": x.get("refusedRule") or "failed"}
    errs = T._obj(x.get("errors"))
    if errs:
        name, e = sorted(errs.items())[0]
        e = T._obj(e)
        return {"error": "%s: %s" % (name, e.get("error")), "refusedRule": e.get("refusedRule") or "failed"}
    rets = x.get("returns") if isinstance(x.get("returns"), list) else []
    for ret in rets:
        if isinstance(ret, dict):
            return {"error": ret.get("error"), "refusedRule": ret.get("refusedRule") or "failed"}
    return {"className": x.get("className"), "values": T._obj(x.get("values")), "returns": rets}


def _many(items, props=None, calls=None):
    """Read many targets: items [{target, props?, calls?}], with props and calls added to every item. calls are
    [{method, args}] with wire args, read-only members only. -> (rows, route): each row {className, values, returns} or
    {error, refusedRule}. read_many when the plugin has it (R1-1), else one call per target."""
    props, calls = list(props or []), list(calls or [])
    full = [{"target": it["target"], "props": props + list(it.get("props") or []),
             "calls": [dict(c, argc=len(c.get("args") or [])) for c in calls + list(it.get("calls") or [])]}
            for it in items]
    if _MANY["ok"] is not False and full:
        rows = []
        for i in range(0, len(full), MANY_CHUNK):
            r = T._rbx("read_many", items=full[i:i + MANY_CHUNK], timeout=120)
            if not r.get("ok"):
                if r.get("refusedRule") == "unknown-op" and not rows:
                    _MANY["ok"] = False
                    break
                return None, r
            _MANY["ok"] = True
            got = r.get("items")
            if got is None:
                got = r.get("results")
            if not isinstance(got, list) or len(got) != len(full[i:i + MANY_CHUNK]):
                return None, _no("read_many answered %s rows for %d items" % (
                    len(got) if isinstance(got, list) else "no", len(full[i:i + MANY_CHUNK])), refusedRule="failed")
            rows.extend(_many_row(x) for x in got)
        else:
            return rows, "read_many"
    rows = []
    for it in full:
        row = {"values": {}, "returns": []}
        if it["props"]:
            g = T._rbx("get_props", target=it["target"], props=it["props"])
            if not g.get("ok"):
                rows.append({"error": g.get("error"), "refusedRule": g.get("refusedRule")})
                continue
            row["values"], row["className"] = T._obj(g.get("values")), g.get("className")
        bad = None
        for c in it["calls"]:
            r = T._rbx("call", target=it["target"], method=c["method"], args=c.get("args") or [], argc=c["argc"],
                       readOnly=True)
            if not r.get("ok"):
                bad = {"error": r.get("error"), "refusedRule": r.get("refusedRule")}
                break
            row["className"] = r.get("className")
            row["returns"].append(r.get("returns") or [])
        rows.append(bad or row)
    return rows, "per-target"


def _classes(paths):
    """{tuple(path): className} by one read_many when the plugin has it (a resolve per path cost 17.6 s of a 17.9 s
    rbx_pivot of 500 parts, measured live 2026-11-20, RB1181), else by resolve, or (None, the failure)."""
    out = {}
    if len(paths) > 1 and _MANY["ok"] is not False:
        rows, route = _many([{"target": p} for p in paths])
        if rows is None:
            return None, route
        if route == "read_many":
            for p, row in zip(paths, rows):
                if row.get("error") is not None or not row.get("className"):
                    return None, _no(row.get("error") or "read_many named no class", target=V.join_path(p),
                                     refusedRule=row.get("refusedRule") or "failed")
                out[tuple(p)] = row["className"]
            return out, None
    for p in paths:
        r = T._rbx("resolve", target=p)
        if not r.get("ok"):
            return None, dict(r, target=V.join_path(p))
        out[tuple(p)] = r.get("className")
    return out, None


class _Group:
    """One undo recording for a composite's writes: commit on success, cancel on failure."""

    def __init__(self, name):
        self.name, self.open = name, False

    def __enter__(self):
        b = T.begin_group(self.name)
        self.open = bool(b.get("ok"))
        self.began = b
        return self

    def end(self, ok):
        if not self.open:
            return None
        self.open = False
        return T.rbx_undo_group("commit" if ok else "cancel")

    def __exit__(self, et, ev, tb):
        if self.open:
            self.end(False)
        return False


def _children(parts):
    """({child names of parts}, None) or (None, the failure): tools._child_names, the one shared reader (D16)."""
    return T._child_names(parts)


def _free_names(parent_parts, base, n):
    """n names from base not in use under parent_parts: (names, None) or (None, the failure)."""
    have, err = _children(parent_parts)
    if err:
        return None, err
    out, i = [], 0
    while len(out) < n:
        i += 1
        name = base if (i == 1 and n == 1) else "%s_%d" % (base, i)
        if name not in have:
            out.append(name)
    return out, None


def _pivots(paths):
    """GetPivot of each path: ([cf or None], route, failure)."""
    rows, route = _many([{"target": p} for p in paths], calls=[{"method": "GetPivot", "args": []}])
    if rows is None:
        return None, None, route
    out = []
    for row in rows:
        rets = (row.get("returns") or [[]])
        out.append(_cf((rets[0] or [None])[0]) if rets and rets[0] else None)
    return out, route, None


# ---------------------------------------------------------------- the {handle} extensions of tools.py tools
def _handle_verdict(name, out):
    T._post_verdict(name, out)
    return out


def rbx_reparent(target: Any, parent: Any) -> dict:
    "Roblox Studio: move an instance under a new parent in one undo step. Read back: its Parent is the new parent. target is a path, or {handle} for an object a call returned outside the place (an InsertService:LoadAsset model, a GeometryService:SubtractAsync result); for a handle in scratch mode the destination is the check: it must be under Workspace.MifScratch."
    if not _handle(target):
        return _ORIG["rbx_reparent"](target, parent)
    pparts, err = _parts(parent, "parent")
    if err:
        return err
    ping = _ping()
    if not ping.get("ok"):
        return ping
    bad = _scratch_refusal(ping, [pparts], "reparenting a handle")
    if bad:
        return _handle_verdict("rbx_reparent", bad)
    r = T._rbx("reparent", target={"handle": target["handle"]}, parent=pparts)
    if not r.get("ok"):
        return _handle_verdict("rbx_reparent", r)
    ok = r.get("parentPath") == pparts
    out = {"ok": ok, "verified": ok, "path": V.join_path(r.get("path")), "handle": target["handle"],
           "requestId": r.get("requestId")}
    if not ok:
        out["error"] = "its parent is %s, not %s" % (V.join_path(r.get("parentPath")), V.join_path(pparts))
    return _handle_verdict("rbx_reparent", out)


def rbx_set_props(target: Any, props: dict) -> dict:
    "Roblox Studio: set properties of one instance in one undo step, then read each back. ok means Studio holds every value sent (verified: true); a clamp or rejection answers ok false with sent and readBack. Refused before sending: unreachable security, ReadOnly, an unknown name, a value of the wrong shape for the property's type. target is a path, or {handle} for an object a call returned (rename a loaded model before it is parented). An image Content property (a SurfaceAppearance or MaterialVariant map, a Decal or Texture, an image, a clothing template) given an rbxassetid that names a Decal - what an image upload answers - is set to the image inside it (decalSwaps says which), since a Decal id draws nothing. "" sent to a Content property clears it and reads back as the same value, the empty Content (plugin 0.17.16); a vector component may be inf or -inf where the type allows it (UISizeConstraint.MaxSize defaults to inf, inf)."
    # An image Content given a Decal id draws nothing - on a SurfaceAppearance it can drop the whole appearance ("Asset
    # type does not match requested type", the internal notes L85) - so a Decal id is swapped for the image inside it (RB1310).
    props, swaps, ierr = T._image_props(props)
    if ierr:
        return {"ok": False, "ran": False, "refusedRule": "decal-without-image", "error": ierr}
    if not _handle(target):
        out = _ORIG["rbx_set_props"](target, props)
        return dict(out, decalSwaps=swaps) if swaps and isinstance(out, dict) else out
    h = {"handle": target["handle"]}
    cat, err = _cat()
    if err:
        return err
    cls, err = T._resolve(h)
    if err:
        return _handle_verdict("rbx_set_props", err)
    wire, err = T._encode_props(cat, cls, props)
    if err:
        return _handle_verdict("rbx_set_props", err)
    r = T._rbx("set_props", target=h, props=wire)
    if not r.get("ok"):
        return _handle_verdict("rbx_set_props", r)
    verified, bad = T._judge(wire, r.get("after"))
    before_w, after_w = T._obj(r.get("before")), T._obj(r.get("after"))
    out = {"ok": verified, "verified": verified, "className": cls, "handle": target["handle"],
           "before": {k: V.decode(v) for k, v in before_w.items()},
           "after": {k: V.decode(v) for k, v in after_w.items()}, "requestId": r.get("requestId"),
           # changed, as for a path (the coverage record keeps "write" only for a changed property): without it every
           # write on a handle counted as write-same, a ChatWindowMessageProperties.TextSize 14 -> 15 included (RB2446)
           "changed": {k: k in before_w and not V.same(before_w[k], after_w[k], cat.property_type(cls, k))
                       for k in after_w}}
    if bad:
        out.update(mismatch=bad, error="Studio holds something else for %s" % ", ".join(sorted(bad)))
    if swaps:
        out["decalSwaps"] = swaps
    return _handle_verdict("rbx_set_props", out)


def rbx_destroy(target: Any, all: bool = False) -> dict:  # noqa: A002 - the parameter is the word a caller writes
    "Roblox Studio: remove an instance in one undo step. Read back: nothing is left at its path under its parent. Plugin 0.16.0 and later remove an instance in the place by Parent = nil inside the recording, so rbx_undo puts it back (removedBy, undoable); an older plugin calls Destroy, which no undo reverses. target is a path, or {handle} for an object a call returned: an object outside the place is destroyed and its handle released, so a later call with it is refused; one in the place removed by Parent = nil keeps its handle (released: false), because releasing an object outside the place destroys it and the undo could no longer bring it back - rbx_release_handle it when it is no longer needed. Several children with the same name: a path segment Name[n] reaches the nth (GetChildren order; the ambiguous refusal lists each one's class, position and indexed path), and all=true removes every child of the parent named the last segment in one undo step."
    if not _handle(target):
        return _ORIG["rbx_destroy"](target, all=all)
    if all:
        return {"ok": False, "ran": False, "refusedRule": "type", "error": "all=true takes a path, not a {handle}"}
    h = {"handle": target["handle"]}
    where = T._rbx("resolve", target=h)
    if not where.get("ok"):
        return _handle_verdict("rbx_destroy", where)
    r = T._rbx("destroy", target=h)
    if not r.get("ok"):
        return _handle_verdict("rbx_destroy", r)
    gone = r.get("stillThere") is False
    if gone and where.get("path"):
        gone = not T._rbx("resolve", target=where["path"]).get("ok") or False
    removed_by = r.get("removedBy") or "Destroy"
    undoable = removed_by == "Parent = nil"
    # The plugin's release_handle destroys an object outside the place, and after Parent = nil this one is outside it:
    # releasing it now would make the removal permanent, so the handle is kept for the undo.
    rel = {"ok": False} if undoable else T._rbx("release_handle", handle=target["handle"])
    out = {"ok": gone, "verified": gone, "handle": target["handle"], "className": r.get("className"),
           "wasAt": V.join_path(where.get("path")), "removedBy": removed_by, "undoable": undoable,
           "released": bool(rel.get("ok")), "requestId": r.get("requestId")}
    if undoable:
        out["note"] = "the handle is kept so rbx_undo can put the object back; rbx_release_handle it when done"
    if not gone:
        out["error"] = "the object is still in the place after Destroy"
    return _handle_verdict("rbx_destroy", out)


def rbx_release_handle(handle: str) -> dict:
    "Roblox Studio: release an object handle the plugin holds. An object still outside the place (an EditableImage, an unplaced result) is destroyed and its handle refused afterwards; an object that has since been put into the place is only forgotten, never destroyed (plugin 0.9.3 and later), and is read again by its path to prove it is still there."
    r = _ORIG["rbx_release_handle"](handle)
    if not r.get("ok"):
        return r
    out = dict(r)
    if r.get("inPlace"):
        still = T._rbx("resolve", target=r["inPlace"])
        out.update(inPlace=V.join_path(r["inPlace"]), stillInPlace=bool(still.get("ok")), verified=bool(still.get("ok")))
        if not still.get("ok"):
            out.update(ok=False, error="the object was in the place and is gone after the release")
    else:
        again = T._rbx("resolve", target={"handle": handle})
        out.update(handleRefused=not again.get("ok"), verified=not again.get("ok"))
        if again.get("ok"):
            out.update(ok=False, error="the handle still resolves after the release")
    return out


# ---------------------------------------------------------------- rbx_pivot
@_tool
def rbx_pivot(targets: list, pivots: list = None, translate: list = None, rotate: list = None, about: str = "each",
              event_mode: str = "FireCFrameChanged") -> dict:
    "Roblox Studio: move and rotate models and parts by their pivot, many at once (up to 2,000). Either pivots, one CFrame per target (12 numbers or {position, rotation}), or translate [x, y, z] and/or rotate [rx, ry, rz] degrees (CFrame.fromEulerAnglesXYZ) about each target's own pivot (about 'each') or about the centre of all their pivots (about 'center'). Parts move in one `WorldRoot:BulkMoveTo` (each part's CFrame is its new pivot times the inverse of `BasePart.PivotOffset`); models and other PVInstances by `PVInstance:PivotTo`; all in one undo step. In scratch mode, where BulkMoveTo on Workspace is refused, parts move by PivotTo instead (route says which). Read back: `PVInstance:GetPivot` of every target to 1e-4, or further from the origin to 8 steps of a 32-bit float at that position (Studio keeps CFrames in 32-bit floats: 4.9e-4 at y 883, 0.016 at 24,576; tolerance says which). A part with `BasePart.Locked` is moved and listed in locked."
    paths, err = _paths(targets, "targets")
    if err:
        return err
    if about not in ("each", "center"):
        return _no("about is 'each' or 'center'")
    if pivots is not None:
        if translate is not None or rotate is not None:
            return _no("give pivots, or translate and/or rotate, not both")
        if not isinstance(pivots, list) or len(pivots) != len(paths):
            return _no("pivots is a list with one CFrame per target")
        want = []
        for p in pivots:
            w, why = V.encode("CFrame", p)
            if why:
                return _no(why, refusedRule="type")
            want.append(_cf(w))
    else:
        if translate is None and rotate is None:
            return _no("give pivots, or translate and/or rotate")
        if translate is not None and not _vec(translate):
            return _no("translate is [x, y, z] studs")
        if rotate is not None and not _vec(rotate):
            return _no("rotate is [rx, ry, rz] degrees")
        want = None
    cat, err = _cat()
    if err:
        return err
    if event_mode not in cat.enums.get("BulkMoveMode", []):
        return _no("event_mode is one of %s" % ", ".join(cat.enums.get("BulkMoveMode", [])), refusedRule="type")
    classes, err = _classes(paths)
    if err:
        return err
    for p in paths:
        if not _is_a(cat, classes[tuple(p)], "PVInstance"):
            return _no("%s is a %s, which has no pivot" % (V.join_path(p), classes[tuple(p)]), refusedRule="type")
    before, route, err = _pivots(paths)
    if err:
        return err
    if any(b is None for b in before):
        return _no("GetPivot answered nothing for %s" % V.join_path(paths[before.index(None)]))
    if want is None:
        d = euler_xyz(rotate or [0, 0, 0])
        t = list(translate or [0, 0, 0])
        c = [sum(b["position"][i] for b in before) / len(before) for i in range(3)]
        want = []
        for b in before:
            if about == "each":
                pos = [b["position"][i] + t[i] for i in range(3)]
            else:
                off = _mv(d, [b["position"][i] - c[i] for i in range(3)])
                pos = [c[i] + off[i] + t[i] for i in range(3)]
            want.append({"position": pos, "rotation": _mm(d, b["rotation"])})
    part_idx = [i for i, p in enumerate(paths) if _is_a(cat, classes[tuple(p)], "BasePart")]
    other_idx = [i for i in range(len(paths)) if i not in set(part_idx)]
    extra, _r2 = _many([{"target": paths[i]} for i in part_idx], props=["PivotOffset", "Locked"]) if part_idx else ([], None)
    if extra is None:
        return _r2
    offsets, locked = {}, []
    for i, row in zip(part_idx, extra):
        po = _cf((row.get("values") or {}).get("PivotOffset"))
        if po is None:
            return _no("PivotOffset of %s did not read: %s" % (V.join_path(paths[i]), row.get("error")))
        offsets[i] = po
        if (row.get("values") or {}).get("Locked") is True:
            locked.append(V.join_path(paths[i]))
    move_route = "BulkMoveTo" if part_idx else "PivotTo"
    with _Group("MifBridge pivot") as g:
        if not g.open:
            return dict(g.began, ok=False, error="no undo group could be opened: %s" % g.began.get("error"))
        if part_idx:
            cfs = [_wcf(cf_mul(want[i], cf_inv(offsets[i]))) for i in part_idx]
            r = _wcall(["Workspace"], "Workspace", "BulkMoveTo",
                       [{"$t": "Instances", "v": [_inst(paths[i]) for i in part_idx]}, {"$t": "Array", "v": cfs},
                        {"$t": "EnumItem", "enum": "BulkMoveMode", "v": event_mode}], read_only=False)
            if not r.get("ok") and r.get("refusedRule") == "scratch-path":
                move_route, other_idx = "PivotTo", sorted(other_idx + part_idx)
            elif not r.get("ok"):
                g.end(False)
                return dict(r, ok=False, step="BulkMoveTo", cancelled=True)
        for i in other_idx:
            r = _wcall(paths[i], classes[tuple(paths[i])], "PivotTo", [_wcf(want[i])], read_only=False)
            if not r.get("ok"):
                g.end(False)
                return dict(r, ok=False, step="PivotTo", target=V.join_path(paths[i]), cancelled=True)
        g.end(True)
    after, route2, err = _pivots(paths)
    if err:
        return dict(err, ok=False, moved=True, error="moved, but the read-back failed: %s" % err.get("error"))
    errs = [cf_err(a, w) if a else float("inf") for a, w in zip(after, want)]
    tols = [pivot_tol(w, a) for w, a in zip(want, after)]
    bad = [{"target": V.join_path(paths[i]), "wanted": want[i], "readBack": after[i], "error": errs[i], "tolerance": tols[i]}
           for i in range(len(paths)) if errs[i] > tols[i]]
    out = {"ok": not bad, "verified": not bad, "moved": len(paths), "parts": len(part_idx),
           "models": len(paths) - len(part_idx), "moveRoute": move_route, "readRoute": route2,
           "maxError": max(errs), "tolerance": max(tols), "locked": locked, "undoSteps": 1,
           "pivots": [{"target": V.join_path(p), "pivot": a} for p, a in list(zip(paths, after))[:50]]}
    if bad:
        out.update(mismatch=bad[:20], error="%d of %d pivots read back further away than their tolerance (%g at most)"
                   % (len(bad), len(paths), max(b["tolerance"] for b in bad)))
    return out


# ---------------------------------------------------------------- rbx_scale_model
@_tool
def rbx_scale_model(target: Any, scale: float = None, factor: float = None) -> dict:
    "Roblox Studio: scale a Model about its pivot with `Model:ScaleTo` - scale is the new absolute scale factor, factor multiplies the current one (`Model:GetScale`). One undo step. Read back: GetScale equals the new factor, and `Model:GetExtentsSize` is what the parts predict: each BasePart's CFrame and Size are read first, its offset from the pivot scaled by the ratio and each side by the ratio but never below 0.001 (Studio's smallest side; a side shown at 0.001 may hold less, which Studio restores when the model grows again, so it is a range), each counted at least 0.05 on its own axes (what GetExtentsSize counts), boxed in the model's bounding-box frame; within 1e-4 relative or 8 steps of a 32-bit float at the model's position. So a tiny factor (0.000209, a rig to 1 stud) passes and an extents change the parts do not explain fails. When the prediction does not match the model before scaling (extents from something that is not a BasePart), the plain ratio is checked instead (within 1e-3 relative) and predicted says false. Scale at or below 0 is refused."
    parts, err = _parts(target)
    if err:
        return err
    if (scale is None) == (factor is None):
        return _no("give scale or factor, one of them")
    k = scale if scale is not None else factor
    if not _num(k) or k <= 0:
        return _no("scale and factor are above 0", refusedRule="type")
    cat, err = _cat()
    if err:
        return err
    cls, err = T._resolve(parts)
    if err:
        return err
    if not _is_a(cat, cls, "Model"):
        return _no("%s is a %s; ScaleTo is a Model's" % (V.join_path(parts), cls), refusedRule="type")
    g0 = T.rbx_call(parts, "GetScale")
    e0 = T.rbx_call(parts, "GetExtentsSize")
    if not (g0.get("ok") and e0.get("ok")):
        return dict(g0 if not g0.get("ok") else e0, ok=False)
    old, ext0 = g0.get("value"), _v3(e0.get("value"))
    new = k if scale is not None else old * k
    ratio = new / old
    frame, err = _scale_frame(parts)
    if err:
        return err
    pred0 = predict_extents(frame, 1.0)
    predicted = pred0 is not None and ext0 is not None and _extents_match(pred0, ext0, frame, 1.0)
    r = T.rbx_call(parts, "ScaleTo", [new])
    if not r.get("ok"):
        return r
    e1 = T.rbx_call(parts, "GetExtentsSize")
    ext1 = _v3(e1.get("value"))
    seen = [b / a if a else None for a, b in zip(ext0, ext1 or [0, 0, 0])]
    want = predict_extents(frame, ratio) if predicted else None
    if want is not None:
        good_ext = ext1 is not None and _extents_match(want, ext1, frame, ratio)
    else:
        good_ext = ext1 is not None and all(s is not None and abs(s - ratio) <= 1e-3 * ratio for s in seen)
    ok = bool(r.get("verified")) and good_ext
    out = {"ok": ok, "verified": ok, "target": V.join_path(parts), "scaleBefore": old, "scaleAfter": (r.get("readBack") or {}).get("value"),
           "extentsBefore": _r(ext0), "extentsAfter": _r(ext1), "ratio": ratio, "extentRatios": _r(seen, 6),
           "predicted": predicted, "extentsPredicted": [_r(w, 6) for w in want] if want else None, "parts": len(frame["parts"]),
           "requestId": r.get("requestId")}
    if not ok:
        out["error"] = r.get("error") or ("the extents read %s, the parts predict %s to %s" % (_r(ext1, 6), _r(want[0], 6), _r(want[1], 6)) if want
                                          else "the extents changed by %s, not %s" % (_r(seen, 6), ratio))
    return out


def _scale_frame(parts):
    """What rbx_scale_model predicts from: the model's pivot (ScaleTo's centre), its bounding box's rotation (the frame
    GetExtentsSize measures in) and every BasePart's CFrame and Size. -> (frame, failure)."""
    piv, _route, err = _pivots([parts])
    if err:
        return None, err
    bb = T.rbx_call(parts, "GetBoundingBox")
    found = T.rbx_find(class_name="BasePart", root=parts, limit=MAX_TARGETS)
    if not (bb.get("ok") and found.get("ok")) or not piv or piv[0] is None:
        return None, dict(bb if not bb.get("ok") else found, ok=False,
                          error="could not read the model's pivot, bounding box and parts: %s" % (bb.get("error") or found.get("error")))
    if len(found.get("matches") or []) >= MAX_TARGETS:
        piv_only = {"pivot": piv[0]["position"], "rotation": piv[0]["rotation"], "parts": []}
        return piv_only, None   # too many parts to read: the plain ratio is checked (predicted false)
    paths = [_parts(m["path"])[0] for m in found.get("matches") or []]
    if any(x is None for x in paths):
        return None, _no("a part's path did not parse: %s" % found.get("matches"))
    rows, route = _many([{"target": x} for x in paths], props=["CFrame", "Size"])
    if rows is None:
        return None, route
    ps = []
    for row in rows:
        cf, sz = _cf((row.get("values") or {}).get("CFrame")), _v3((row.get("values") or {}).get("Size"))
        if cf is None or sz is None:
            return None, _no("could not read a part's CFrame and Size: %s" % row.get("error"))
        ps.append((cf, sz))
    box = _cf((bb.get("values") or [None])[0])
    rot = box["rotation"] if box else piv[0]["rotation"]
    return {"pivot": piv[0]["position"], "rotation": rot, "parts": ps}, None


def predict_extents(frame, ratio):
    """Model:GetExtentsSize after ScaleTo(ratio) about the pivot, as [low, high] per axis: each part's offset from the
    pivot times ratio, each side times ratio but at least MIN_PART, each counted at least MIN_EXTENT on its own axes,
    boxed in the frame's rotation. A side that reads MIN_PART may hold less: Studio keeps each part's own size through
    ScaleTo and shows it no smaller than 0.001, so a side clamped on the way down comes back whole on the way up
    (measured, RB1752); such a side is anywhere from MIN_PART to MIN_PART times ratio. None for a model with no parts."""
    if not frame["parts"]:
        return None
    inv = _tr(frame["rotation"])
    lo_boxes, hi_boxes = [], []
    for cf, size in frame["parts"]:
        off = _mv(inv, [cf["position"][i] - frame["pivot"][i] for i in range(3)])
        rel = {"position": [x * ratio for x in off], "rotation": _mm(inv, cf["rotation"])}
        lo = [max(s * ratio, MIN_PART) if s > MIN_PART * 1.000001 else MIN_PART for s in size]
        hi = [max(s * ratio, MIN_PART) for s in size]
        lo_boxes.append(aabb(rel, [max(x, MIN_EXTENT) for x in lo]))
        hi_boxes.append(aabb(rel, [max(x, MIN_EXTENT) for x in hi]))
    a, b = _box_union(lo_boxes), _box_union(hi_boxes)
    return [[a[1][i] - a[0][i] for i in range(3)], [b[1][i] - b[0][i] for i in range(3)]]


def _extents_match(want, got, frame, ratio):
    """got inside [low, high] give or take float steps: F32_STEPS at the size read, plus the positions' own float steps
    before the scale times the ratio (a model shrunk at x 1,024 and grown 4,793 times again carries 1.2e-4 x 4,793)."""
    lo, hi = want
    before = max([abs(v) for v in frame["pivot"]] + [abs(v) for cf, _s in frame["parts"] for v in cf["position"]])
    tol = F32_STEPS * (f32_step(max(abs(v) for v in list(frame["pivot"]) + list(hi))) + ratio * f32_step(before))
    return all(lo[i] - max(1e-4 * abs(lo[i]), tol) <= got[i] <= hi[i] + max(1e-4 * abs(hi[i]), tol) for i in range(3))


# ---------------------------------------------------------------- rbx_csg
CSG_METHODS = {"union": "UnionAsync", "subtract": "SubtractAsync", "intersect": "IntersectAsync",
               "negate": "NegateAsync", "separate": "SeparateAsync", "bake": "BakeToMeshAsync"}


def _options(opts, allowed, cat):
    if opts is None:
        return None, None
    if not isinstance(opts, dict) or set(opts) - set(allowed):
        return None, _no("options takes %s (Roblox's GeometryService page names no others)" % ", ".join(allowed),
                         refusedRule="type")
    out = {}
    for k, v in opts.items():
        w, why = V.encode(allowed[k], v, cat.enums)
        if why:
            return None, _no("options.%s: %s" % (k, why), refusedRule="type")
        out[k] = w
    return {"$t": "Dictionary", "v": out}, None


def _boxes(paths):
    rows, _route = _many([{"target": p} for p in paths], props=EXTENTS)
    if rows is None:
        return None, _route
    out = []
    for p, row in zip(paths, rows):
        cf, sz = _cf((row.get("values") or {}).get(EXTENTS[0])), _v3((row.get("values") or {}).get(EXTENTS[1]))
        if cf is None or sz is None:
            return None, _no("could not read CFrame and Size of %s: %s" % (V.join_path(p), row.get("error")))
        out.append(aabb(cf, sz))
    return out, None


def _remove(parts):
    """Take an input out of the place so the undo step can put it back: Parent = nil, not Destroy. Measured on 0.748:
    an undo does not bring back an instance the plugin's destroy op removed (Instance:Destroy), and does bring back one
    whose Parent was set to nil (RB712). The path is read again: gone means gone."""
    r = T._rbx("set_props", target=parts, props={"Parent": {"$t": "nil"}})
    if r.get("ok") and T._rbx("resolve", target=parts).get("ok"):
        return dict(r, ok=False, error="%s is still in the place" % V.join_path(parts))
    return r


def _place_results(handles, pparts, base, replace_paths, group_name):
    """Name each result handle freely under pparts, parent it, and destroy replace_paths, in one undo group.
    -> (paths, failure)."""
    names, err = _free_names(pparts, base, len(handles))
    if err:
        return None, dict(err, step="name the results")
    placed = []
    with _Group(group_name) as g:
        if not g.open:
            return None, dict(g.began, ok=False, error="no undo group could be opened: %s" % g.began.get("error"))
        for h, name in zip(handles, names):
            nm = T._rbx("set_props", target={"handle": h}, props={"Name": name})
            mv = T._rbx("reparent", target={"handle": h}, parent=pparts) if nm.get("ok") else nm
            if not mv.get("ok") or not T.same_parent(mv.get("parentPath") or [], pparts):
                g.end(False)
                return None, dict(mv, ok=False, step="place", handle=h, cancelled=True,
                                  error="placing a result failed: %s" % (mv.get("error") or mv.get("parentPath")))
            placed.append(mv.get("path"))
        for p in replace_paths:
            d = _remove(p)
            if not d.get("ok"):
                g.end(False)
                return None, dict(d, ok=False, step="replace", target=V.join_path(p), cancelled=True)
        g.end(True)
    for h in handles:
        T._rbx("release_handle", handle=h)       # in the place now: forgotten, never destroyed (plugin 0.9.3+)
    return placed, None


@_tool
def rbx_csg(operation: str, target: Any = None, parts: list = None, parent: Any = None, name: str = None,
            replace: bool = False, options: dict = None) -> dict:
    "Roblox Studio: solid modelling with GeometryService. operation union, subtract or intersect: target and parts (`GeometryService:UnionAsync`, `GeometryService:SubtractAsync`, `GeometryService:IntersectAsync`, options CollisionFidelity, RenderFidelity, FluidFidelity, SplitApart); negate or separate: parts (`GeometryService:NegateAsync`, `GeometryService:SeparateAsync`); bake: target to a MeshPart (`GeometryService:BakeToMeshAsync`). On Studio 0.748 NegateAsync and SeparateAsync answer 'the function is not enabled yet' and BakeToMeshAsync 'This API is not available yet' (RB704); that answer is passed on. Results are made outside the place, then named (name, default the operation) and parented under parent (default the target's or first part's parent) in one undo step; replace also takes the inputs out of the place in that step (Parent = nil, so the one undo puts them back, RB712). Read back: each result by path, its class, `PartOperation.TriangleCount` above 0 for a PartOperation, and its world box (`BasePart.ExtentsCFrame` and `BasePart.ExtentsSize`; a result keeps its first input's CFrame while its geometry is off centre, RB703): a union's within the inputs' combined box (equal to it when every input is a Part block), a subtraction's within the target's, an intersection's within the inputs' overlap, a bake's equal to the input's. Scratch mode: inputs and parent must be under Workspace.MifScratch."
    if operation not in CSG_METHODS:
        return _no("operation is one of %s" % ", ".join(CSG_METHODS))
    method = CSG_METHODS[operation]
    tparts, plist = None, []
    if operation in ("union", "subtract", "intersect", "bake"):
        tparts, err = _parts(target)
        if err:
            return err
    if operation in ("union", "subtract", "intersect", "negate", "separate"):
        plist, err = _paths(parts, "parts", 1, 100)
        if err:
            return err
    elif parts is not None:
        return _no("bake takes target only")
    inputs = ([tparts] if tparts else []) + plist
    if len({tuple(p) for p in inputs}) != len(inputs):
        return _no("an input is named twice")
    if not isinstance(replace, bool):
        return _no("replace is true or false")
    pparts = inputs[0][:-1]
    if parent is not None:
        pparts, err = _parts(parent, "parent")
        if err:
            return err
    cat, err = _cat()
    if err:
        return err
    if operation in ("negate", "separate", "bake") and options is not None:
        return _no("%s takes no options" % operation, refusedRule="type")
    wopts, err = _options(options, CSG_OPTIONS, cat)
    if err:
        return err
    classes, err = _classes(inputs)
    if err:
        return err
    for p in inputs:
        if not _is_a(cat, classes[tuple(p)], "BasePart"):
            return _no("%s is a %s, not a part" % (V.join_path(p), classes[tuple(p)]), refusedRule="type")
    ping = _ping()
    if not ping.get("ok"):
        return ping
    bad = _scratch_refusal(ping, inputs + [pparts], "rbx_csg")
    if bad:
        return bad
    boxes, err = _boxes(inputs)
    if err:
        return err
    if operation in ("union", "subtract", "intersect"):
        args = [_inst(tparts), {"$t": "Array", "v": [_inst(p) for p in plist]}] + ([wopts] if wopts else [])
    elif operation == "bake":
        args = [_inst(tparts)]
    else:
        args = [{"$t": "Instances", "v": [_inst(p) for p in plist]}]
    r = _wcall(["GeometryService"], "GeometryService", method, args, read_only=True, timeout=120)
    if not r.get("ok"):
        return dict(r, step=method)
    ret = _ret(r)
    objs = [ret] if operation == "bake" else _items(ret)
    handles = [o.get("handle") for o in objs if isinstance(o, dict) and o.get("handle")]
    if not handles:
        return {"ok": False, "ran": True, "step": method, "error": "%s returned no result: %r" % (method, ret)}
    placed, err = _place_results(handles, pparts, name or operation.capitalize(),
                                 inputs if replace else [], "MifBridge " + operation)
    if err:
        for h in handles:
            T._rbx("release_handle", handle=h)
        return err
    rows, _route = _many([{"target": p} for p in placed], props=EXTENTS)
    if rows is None:
        return dict(_route, ok=False, error="placed, then the read-back failed")
    results, problems = [], []
    if operation == "union":
        expect = _box_union(boxes)
        blocks = all(classes[tuple(p)] == "Part" for p in inputs)
    elif operation == "subtract":
        expect, blocks = boxes[0], False
    elif operation == "intersect":
        expect, blocks = _box_inter(boxes), False
    elif operation == "bake":
        expect, blocks = boxes[0], True
    else:
        expect, blocks = None, False
    if blocks and operation == "union":
        shapes, _x = _many([{"target": p} for p in inputs], props=["Shape"])
        blocks = shapes is not None and all(((s.get("values") or {}).get("Shape") or {}).get("v") == "Block" for s in shapes)
    for p, row in zip(placed, rows):
        cls = row.get("className")
        cf, sz = _cf((row.get("values") or {}).get(EXTENTS[0])), _v3((row.get("values") or {}).get(EXTENTS[1]))
        box = aabb(cf, sz) if cf and sz else None
        res = {"path": V.join_path(p), "className": cls, "box": [_r(box[0]), _r(box[1])] if box else None}
        if cls and _is_a(cat, cls, "PartOperation"):
            tc = T._rbx("get_props", target=p, props=["TriangleCount"])
            res["triangleCount"] = T._obj(tc.get("values")).get("TriangleCount")
            if not (_num(res["triangleCount"]) and res["triangleCount"] > 0):
                problems.append("%s has TriangleCount %r" % (res["path"], res["triangleCount"]))
        if box is None:
            problems.append("%s: no box read back" % res["path"])
        elif expect is not None and len(placed) == 1 and blocks and not _same_box(box, expect):
            problems.append("%s's box is %s, expected %s" % (res["path"], _r(box), _r(expect)))
        elif expect is not None and not _within(box, expect):
            problems.append("%s's box %s leaves the allowed box %s" % (res["path"], _r(box), _r(expect)))
        results.append(res)
    gone = []
    if replace:
        gone = [V.path_text(p) for p in inputs if T._rbx("resolve", target=p).get("ok")]
        if gone:
            problems.append("inputs still there after replace: %s" % ", ".join(gone))
    out = {"ok": not problems, "verified": not problems, "operation": operation, "member": "GeometryService:" + method,
           "results": results, "count": len(results), "expectedBox": [_r(expect[0]), _r(expect[1])] if expect else None,
           "boxRule": None if expect is None else ("equal" if blocks and len(placed) == 1 else "within"),
           "replaced": [V.join_path(p) for p in inputs] if replace else [], "undoSteps": 1}
    if problems:
        out["error"] = "; ".join(problems)
    return out


# ---------------------------------------------------------------- rbx_weld
def _model_of(cat, parts, seen):
    """The nearest Model ancestor's path (a WorldRoot such as Workspace is not one), or None. seen caches classes."""
    for n in range(len(parts) - 1, 0, -1):
        pre = tuple(parts[:n])
        if pre not in seen:
            r = T._rbx("resolve", target=list(pre))
            seen[pre] = r.get("className") if r.get("ok") else None
        c = seen[pre]
        if c and _is_a(cat, c, "Model") and not _is_a(cat, c, "WorldRoot"):
            return pre
    return None


@_tool
def rbx_weld(action: str, parts: list = None, part0: Any = None, force: bool = False, recursive: bool = False,
             joint_type: str = "All") -> dict:
    "Roblox Studio: welds and what holds what. action weld: a WeldConstraint (`WeldConstraint.Part0`, `WeldConstraint.Part1`) from part0 to each of parts, parented under part0, all in one undo step; parts in different models are refused unless force. Read back: `BasePart:GetConnectedParts` of part0 lists each part and each part lists part0. action report: each part's `BasePart:GetJoints` and GetConnectedParts (recursive: the whole assembly). action join / unjoin: `Workspace:JoinToOutsiders` (joint_type an `Enum.JointCreationMode`) or `Workspace:UnjoinFromOutsiders` on parts, with GetJoints read before and after."
    if action not in ("weld", "report", "join", "unjoin"):
        return _no("action is weld, report, join or unjoin")
    plist, err = _paths(parts, "parts", 1, 100 if action == "weld" else MAX_TARGETS)
    if err:
        return err
    cat, err = _cat()
    if err:
        return err
    p0 = None
    if action == "weld":
        p0, err = _parts(part0, "part0")
        if err:
            return err
        if tuple(p0) in {tuple(p) for p in plist}:
            return _no("part0 is also in parts")
    every = ([p0] if p0 else []) + plist
    classes, err = _classes(every)
    if err:
        return err
    for p in every:
        if not _is_a(cat, classes[tuple(p)], "BasePart"):
            return _no("%s is a %s, not a part" % (V.join_path(p), classes[tuple(p)]), refusedRule="type")

    def joints_and_connected(paths):
        rows, route = _many([{"target": p} for p in paths], calls=[
            {"method": "GetJoints", "args": []}, {"method": "GetConnectedParts", "args": [bool(recursive)]}])
        if rows is None:
            return None, route
        # keyed by tuple(path): a name holding a dot ("Cube.001") makes join_path a list, which is no dict key (D5)
        out = {}
        for p, row in zip(paths, rows):
            rets = row.get("returns") or []
            j = [V.decode(x) for x in _items((rets[0] or [None])[0])] if len(rets) > 0 and rets[0] else []
            c = [V.decode(x) for x in _items((rets[1] or [None])[0])] if len(rets) > 1 and rets[1] else []
            out[tuple(p)] = {"joints": j, "connected": [x for x in c if _key(x) != tuple(p)], "error": row.get("error")}
        return out, route

    def shown(rep):
        return {V.path_text(k): v for k, v in rep.items()}

    if action == "report":
        rep, route = joints_and_connected(plist)
        if rep is None:
            return route
        return {"ok": True, "parts": shown(rep), "route": route, "recursive": bool(recursive),
                "free": [V.path_text(k) for k, v in rep.items() if not v["joints"]]}
    if action in ("join", "unjoin"):
        ping = _ping()
        if not ping.get("ok"):
            return ping
        before, _x = joints_and_connected(plist)
        if action == "join":
            if joint_type not in cat.enums.get("JointCreationMode", []):
                return _no("joint_type is one of %s" % ", ".join(cat.enums.get("JointCreationMode", [])), refusedRule="type")
            r = _wcall(["Workspace"], "Workspace", "JoinToOutsiders",
                       [{"$t": "Instances", "v": [_inst(p) for p in plist]},
                        {"$t": "EnumItem", "enum": "JointCreationMode", "v": joint_type}], read_only=False)
        else:
            r = _wcall(["Workspace"], "Workspace", "UnjoinFromOutsiders",
                       [{"$t": "Instances", "v": [_inst(p) for p in plist]}], read_only=False)
        if not r.get("ok"):
            return r
        after, route = joints_and_connected(plist)
        if after is None:
            return route
        changed = {k: {"before": len((before or {}).get(k, {}).get("joints", [])), "after": len(v["joints"])}
                   for k, v in after.items()}
        # verified needs the count to have moved the right way for some part: a join that made no joint, or an unjoin
        # that broke none, read back the same as before and proves nothing (D22f); so does a before that was not read
        moved = any((c["after"] > c["before"]) if action == "join" else (c["after"] < c["before"]) for c in changed.values())
        never_back = all((c["after"] >= c["before"]) if action == "join" else (c["after"] <= c["before"])
                         for c in changed.values())
        out = {"ok": True, "member": "Workspace:" + ("JoinToOutsiders" if action == "join" else "UnjoinFromOutsiders"),
               "joints": shown(changed), "parts": shown(after), "route": route,
               "verified": before is not None and moved and never_back}
        if before is None:
            out["note"] = "the joints before the call could not be read, so the change is not proved"
        elif not moved:
            out["note"] = "no part's joint count changed: nothing shows the call %s anything" % (
                "joined" if action == "join" else "unjoined")
        return out
    # weld
    if not force:
        seen = {}
        homes = {_model_of(cat, p, seen) for p in every}
        if len(homes) > 1:
            return _no("the parts are in different models (%s): a weld across models ties them together; pass force "
                       "true to do it anyway" % ", ".join(sorted(V.path_text(list(h)) if h else "(none)" for h in homes)),
                       refusedRule="cross-model")
    ping = _ping()
    if not ping.get("ok"):
        return ping
    bad = _scratch_refusal(ping, [p0], "rbx_weld")
    if bad:
        return bad
    names, nerr = _free_names(p0, "Weld", len(plist))
    if nerr:
        return dict(nerr, step="name the welds")
    ops = [{"tool": "rbx_create", "args": {"class_name": "WeldConstraint", "parent": p0, "name": n,
                                           "props": {"Part0": V.join_path(p0), "Part1": V.join_path(p)}}}
           for n, p in zip(names, plist)]
    b = T.rbx_batch(ops)
    if not b.get("ok"):
        return dict(b, step="create welds")
    rep, route = joints_and_connected([p0] + plist)
    if rep is None:
        return route
    k0 = tuple(p0)
    tied = {k: {_key(x) for x in v["connected"]} for k, v in rep.items()}
    missing = [V.path_text(p) for p in plist if tuple(p) not in tied.get(k0, set()) or k0 not in tied.get(tuple(p), set())]
    out = {"ok": not missing, "verified": not missing, "welds": [V.join_path(p0 + [n]) for n in names],
           "connected": shown(rep), "route": route, "undoSteps": 1}
    if missing:
        out["error"] = "GetConnectedParts does not tie %s to %s" % (", ".join(missing), V.path_text(p0))
    return out


# ---------------------------------------------------------------- rbx_layout
def _look_y(yaw_deg):
    return euler_xyz([0, yaw_deg, 0])


def _up_to(normal):
    """A rotation taking +Y to normal (unit), by the shortest arc."""
    n = normal
    ln = math.sqrt(sum(x * x for x in n)) or 1.0
    n = [x / ln for x in n]
    c = n[1]
    if c > 1 - 1e-9:
        return list(IDENT)
    if c < -1 + 1e-9:
        return [1, 0, 0, 0, -1, 0, 0, 0, -1]
    ax = [n[2], 0, -n[0]]                     # Y cross n
    s = math.sqrt(ax[0] ** 2 + ax[2] ** 2)
    k = [ax[0] / s, 0, ax[2] / s]
    a = math.acos(max(-1, min(1, c)))
    co, si, t = math.cos(a), math.sin(a), 1 - math.cos(a)
    x, y, z = k
    return [t * x * x + co, t * x * y - si * z, t * x * z + si * y,
            t * x * y + si * z, t * y * y + co, t * y * z - si * x,
            t * x * z - si * y, t * y * z + si * x, t * z * z + co]


@_tool
def rbx_layout(source: Any, mode: str, parent: Any = None, name: str = None, counts: list = None, spacing: list = None,
               count: int = None, radius: float = None, center: list = None, face_center: bool = False,
               points: list = None, region: dict = None, seed: int = None, random_yaw: bool = True,
               align_to_normal: bool = False, offset: float = 0.0, ignore: list = None) -> dict:
    "Roblox Studio: copies of a part or model (`Instance:Clone`) placed by pivot, at most 2,000. mode grid: counts [nx, ny, nz] and spacing [sx, sy, sz] from the source's pivot; circle: count copies on a radius around center (default the source's pivot) in the XZ plane, face_center turns each to look at it; points: a list of positions or CFrames; scatter: count copies dropped onto surfaces inside region {min, max} by `WorldRoot:Raycast` straight down from max y, with a seed so the same seed gives the same layout (random_yaw, align_to_normal, offset along the normal; the rays pass through the source and anything in ignore, e.g. an earlier scatter's copies). Copies are named name_1..name_n (default the source's name) under parent (default the source's parent); parts are moved by one `WorldRoot:BulkMoveTo`, models by `PVInstance:PivotTo`. Read back: the count found and every copy's `PVInstance:GetPivot` to 1e-4 (further out, rbx_pivot's 8 float steps)."
    sparts, err = _parts(source, "source")
    if err:
        return err
    if mode not in ("grid", "circle", "points", "scatter"):
        return _no("mode is grid, circle, points or scatter")
    pparts = sparts[:-1]
    if parent is not None:
        pparts, err = _parts(parent, "parent")
        if err:
            return err
    cat, err = _cat()
    if err:
        return err
    cls, err = T._resolve(sparts)
    if err:
        return err
    if not _is_a(cat, cls, "PVInstance"):
        return _no("%s is a %s, which has no pivot" % (V.join_path(sparts), cls), refusedRule="type")
    piv, _route, err = _pivots([sparts])
    if err:
        return err
    base = piv[0]
    want, hits = [], []
    if mode == "grid":
        if not (isinstance(counts, list) and len(counts) == 3 and all(isinstance(c, int) and c >= 1 for c in counts)):
            return _no("grid needs counts [nx, ny, nz], whole numbers 1 or more")
        if not _vec(spacing):
            return _no("grid needs spacing [sx, sy, sz] studs")
        n = counts[0] * counts[1] * counts[2]
        if n > MAX_COPIES:
            return _no("%d copies is more than %d" % (n, MAX_COPIES), refusedRule="too-many")
        for ix in range(counts[0]):
            for iy in range(counts[1]):
                for iz in range(counts[2]):
                    want.append({"position": [base["position"][0] + ix * spacing[0], base["position"][1] + iy * spacing[1],
                                              base["position"][2] + iz * spacing[2]], "rotation": list(base["rotation"])})
    elif mode == "circle":
        if not (isinstance(count, int) and 1 <= count <= MAX_COPIES):
            return _no("circle needs count 1-%d" % MAX_COPIES, refusedRule="too-many" if isinstance(count, int) else "type")
        if not (_num(radius) and radius > 0):
            return _no("circle needs radius above 0")
        c = list(center) if center is not None else list(base["position"])
        if not _vec(c):
            return _no("center is [x, y, z]")
        for i in range(count):
            a = 2 * math.pi * i / count
            pos = [c[0] + radius * math.cos(a), c[1], c[2] + radius * math.sin(a)]
            rot = list(base["rotation"])
            if face_center:
                yaw = math.degrees(math.atan2(pos[0] - c[0], pos[2] - c[2]))   # -Z of the copy points at the centre
                rot = _mm(_look_y(yaw), base["rotation"])
            want.append({"position": pos, "rotation": rot})
    elif mode == "points":
        if not isinstance(points, list) or not 1 <= len(points) <= MAX_COPIES:
            return _no("points is a list of 1-%d positions or CFrames" % MAX_COPIES,
                       refusedRule="too-many" if isinstance(points, list) and points else "type")
        for p in points:
            if _vec(p):
                want.append({"position": list(p), "rotation": list(base["rotation"])})
            else:
                w, why = V.encode("CFrame", p)
                if why:
                    return _no("a point is [x, y, z] or a CFrame: %s" % why, refusedRule="type")
                want.append(_cf(w))
    else:
        if not (isinstance(count, int) and 1 <= count <= MAX_COPIES):
            return _no("scatter needs count 1-%d" % MAX_COPIES, refusedRule="too-many" if isinstance(count, int) else "type")
        if not (isinstance(seed, int) and not isinstance(seed, bool)):
            return _no("scatter needs a whole-number seed, so a layout can be made again")
        if not (isinstance(region, dict) and _vec(region.get("min")) and _vec(region.get("max"))
                and all(region["max"][i] > region["min"][i] for i in range(3))):
            return _no("region is {min: [x, y, z], max: [x, y, z]} with max above min")
        if not _num(offset):
            return _no("offset is studs along the surface normal")
        rng = random.Random(seed)
        lo, hi = region["min"], region["max"]
        drops = [(rng.uniform(lo[0], hi[0]), rng.uniform(lo[2], hi[2]), rng.uniform(0, 360)) for _ in range(count)]
        skip, err = _paths(ignore, "ignore", 0, MAX_COPIES) if ignore is not None else ([], None)
        if err:
            return err
        params, why = V.encode("RaycastParams", {"filterType": "Exclude",
                                                 "filter": [V.join_path(sparts)] + [V.join_path(p) for p in skip]}, cat.enums)
        if why:
            return _no(why)
        items = [{"target": ["Workspace"], "calls": [{"method": "Raycast", "args": [
            _wv3([x, hi[1], z]), _wv3([0, -(hi[1] - lo[1]), 0]), params]}]} for x, z, _y in drops]
        rows, route = _many(items)
        if rows is None:
            return route
        for (x, z, yaw), row in zip(drops, rows):
            rets = row.get("returns") or []
            hit = rets[0][0] if rets and rets[0] else None
            if not isinstance(hit, dict) or hit.get("$t") == "nil" or row.get("error"):
                continue
            pos, nrm = _v3(hit.get("position")), _v3(hit.get("normal")) or [0, 1, 0]
            if pos is None:
                continue
            rot = list(base["rotation"])
            if random_yaw:
                rot = _mm(_look_y(yaw), rot)
            if align_to_normal:
                rot = _mm(_up_to(nrm), rot)
            pos = [pos[i] + nrm[i] * offset for i in range(3)]
            want.append({"position": pos, "rotation": rot})
            hits.append(V.path_text(V.decode(hit.get("instance"))) if hit.get("instance") is not None else None)
        if not want:
            return {"ok": False, "ran": True, "error": "no ray inside the region hit a surface: nothing was placed",
                    "seed": seed, "rays": count}
    ping = _ping()
    if not ping.get("ok"):
        return ping
    bad = _scratch_refusal(ping, [pparts], "rbx_layout")
    if bad:
        return bad
    names, nerr = _free_names(pparts, name or sparts[-1], len(want))
    if nerr:
        return dict(nerr, step="name the copies", made=0)
    copies, undo = [], 0
    for i in range(0, len(names), 100):
        b = T.rbx_batch([{"tool": "rbx_clone", "args": {"target": sparts, "parent": pparts, "name": n}}
                         for n in names[i:i + 100]])
        if not b.get("ok"):
            # a failed batch is cancelled, so it left no undo step: only the batches before it count (D6)
            return dict(b, ok=False, step="clone", made=len(copies), undoSteps=undo,
                        error="cloning failed after %d copies: %s" % (len(copies), b.get("error")))
        undo += 1
        copies.extend(pparts + [n] for n in names[i:i + 100])
    mv = rbx_pivot(copies, pivots=[{"position": w["position"], "rotation": w["rotation"]} for w in want])
    out = {"ok": bool(mv.get("ok")), "verified": bool(mv.get("verified")), "mode": mode, "count": len(copies),
           "copies": [V.join_path(c) for c in copies[:50]], "firstPivot": (mv.get("pivots") or [{}])[0].get("pivot"),
           "maxError": mv.get("maxError"), "moveRoute": mv.get("moveRoute"), "undoSteps": undo + 1,
           "seed": seed if mode == "scatter" else None}
    if mode == "scatter":
        out.update(rays=count, hitsOn=sorted({h for h in hits if h})[:20], missed=count - len(want))
    if not mv.get("ok"):
        out["error"] = "copies made, placing them failed: %s" % mv.get("error")
    return out


# ---------------------------------------------------------------- rbx_spatial_query
QUERY_KINDS = ("raycast", "blockcast", "spherecast", "shapecast", "box", "radius", "in_part", "closest_point")


@_tool
def rbx_spatial_query(kind: str, origin: list = None, direction: list = None, cframe: Any = None, size: list = None,
                      radius: float = None, part: Any = None, position: list = None, include: list = None,
                      exclude: list = None, ignore_water: bool = None, respect_can_collide: bool = None,
                      collision_group: str = None, max_parts: int = None) -> dict:
    "Roblox Studio: what is where. kind raycast (origin, direction: `WorldRoot:Raycast`), blockcast (cframe, size, direction: `WorldRoot:Blockcast`), spherecast (origin, radius, direction: `WorldRoot:Spherecast`), shapecast (part, direction: `WorldRoot:Shapecast`) answer the first hit {instance, position, normal, distance, material} or null; box (cframe, size: `WorldRoot:GetPartBoundsInBox`), radius (position, radius: `WorldRoot:GetPartBoundsInRadius`) and in_part (part: `WorldRoot:GetPartsInPart`) answer the parts found; closest_point (part, position: `BasePart:GetClosestPointOnSurface`) the nearest point on the part. include or exclude (paths) filter the casts and overlaps (RaycastParams / OverlapParams), with ignore_water, respect_can_collide, collision_group and max_parts. Read only."
    if kind not in QUERY_KINDS:
        return _no("kind is one of %s" % ", ".join(QUERY_KINDS))
    if include is not None and exclude is not None:
        return _no("give include or exclude, not both")
    filt = {}
    if include is not None or exclude is not None:
        filt = {"filterType": "Include" if include is not None else "Exclude", "filter": include if include is not None else exclude}
    for k, v in (("respectCanCollide", respect_can_collide), ("collisionGroup", collision_group)):
        if v is not None:
            filt[k] = v
    cast = kind in ("raycast", "blockcast", "spherecast", "shapecast")
    if ignore_water is not None:
        if not cast:
            return _no("ignore_water filters casts only")
        filt["ignoreWater"] = ignore_water
    if max_parts is not None:
        if cast:
            return _no("max_parts limits overlaps only")
        filt["maxParts"] = max_parts
    if kind == "closest_point":
        if filt:
            return _no("closest_point takes no filter")
        if not _vec(position):
            return _no("closest_point needs position [x, y, z]")
        r = T.rbx_call(part, "GetClosestPointOnSurface", [list(position)])
        return dict(r, kind=kind, point=r.get("value")) if r.get("ok") else r
    need = {"raycast": ("origin", "direction"), "blockcast": ("cframe", "size", "direction"),
            "spherecast": ("origin", "radius", "direction"), "shapecast": ("part", "direction"),
            "box": ("cframe", "size"), "radius": ("position", "radius"), "in_part": ("part",)}[kind]
    given = {"origin": origin, "direction": direction, "cframe": cframe, "size": size, "radius": radius, "part": part,
             "position": position}
    missing = [k for k in need if given[k] is None]
    if missing:
        return _no("%s needs %s" % (kind, ", ".join(missing)))
    pk = "RaycastParams" if cast else "OverlapParams"
    args = {"raycast": [origin, direction], "blockcast": [cframe, size, direction],
            "spherecast": [origin, radius, direction], "shapecast": [part, direction], "box": [cframe, size],
            "radius": [position, radius], "in_part": [part]}[kind]
    member = {"raycast": "Raycast", "blockcast": "Blockcast", "spherecast": "Spherecast", "shapecast": "Shapecast",
              "box": "GetPartBoundsInBox", "radius": "GetPartBoundsInRadius", "in_part": "GetPartsInPart"}[kind]
    r = T.rbx_call("Workspace", member, args + [filt] if filt else args)
    if not r.get("ok"):
        return r
    out = {"ok": True, "kind": kind, "member": r.get("member"), "params": pk if filt else None,
           "requestId": r.get("requestId")}
    if cast:
        out["hit"] = r.get("value")
    else:
        found = r.get("value") or []
        out.update(parts=found, count=len(found))
    return out


# ---------------------------------------------------------------- rbx_bounds
@_tool
def rbx_bounds(targets: list = None, root: Any = None, margin: float = 0.01, limit: int = MAX_TARGETS) -> dict:
    "Roblox Studio: bounding boxes and what rests on what. For each target (or every BasePart and Model under root, up to limit): a Model's `Model:GetBoundingBox` and `Model:GetExtentsSize`; a part's CFrame, Size, and world box from `BasePart.ExtentsCFrame` and `BasePart.ExtentsSize`, `BasePart:GetTouchingParts` and `WorldRoot:ArePartsTouchingOthers`, and supportedBy: the other parts inside its box grown by margin studs (`WorldRoot:GetPartBoundsInBox`). A part with nothing in that grown box is floating. Studio's own touching calls do not count two faces that only meet (measured: a part resting on the baseplate reads false, RB701), which is why supportedBy grows the box. Read only."
    if (targets is None) == (root is None):
        return _no("give targets or root, one of them")
    if not (_num(margin) and 0 <= margin <= 10):
        return _no("margin is 0-10 studs")
    if not (isinstance(limit, int) and 1 <= limit <= MAX_TARGETS):
        return _no("limit is 1-%d" % MAX_TARGETS)
    cat, err = _cat()
    if err:
        return err
    if targets is not None:
        paths, err = _paths(targets, "targets")
        if err:
            return err
        classes, err = _classes(paths)
        if err:
            return err
    else:
        rparts = [] if root in ("game",) else V.split_path(root)
        if rparts is None:
            return _no("root is a path", refusedRule="type")
        paths, classes, page, total = [], {}, 0, None
        while len(paths) < limit:
            q = T._rbx("query", root=rparts, page=page, pageSize=500)
            if not q.get("ok"):
                return q
            total = q.get("total")
            for m in q.get("matches") or []:
                c = m.get("className")
                if c and c != "Terrain" and (_is_a(cat, c, "BasePart") or c == "Model") and len(paths) < limit:
                    paths.append(m["path"])
                    classes[tuple(m["path"])] = c
            if not q.get("more"):
                break
            page += 1
        if not paths:
            return {"ok": False, "scanned": total or 0,
                    "error": "no part or model under %s: nothing measured is not nothing floating" % V.join_path(rparts)}
    items, kinds = [], []
    for p in paths:
        c = classes[tuple(p)]
        if _is_a(cat, c, "BasePart"):
            items.append({"target": p, "props": ["CFrame", "Size"] + EXTENTS,
                          "calls": [{"method": "GetTouchingParts", "args": []}]})
            kinds.append("part")
        elif _is_a(cat, c, "Model"):
            items.append({"target": p, "calls": [{"method": "GetBoundingBox", "args": []},
                                                 {"method": "GetExtentsSize", "args": []}]})
            kinds.append("model")
        else:
            return _no("%s is a %s: bounds are a part's or a model's" % (V.join_path(p), c), refusedRule="type")
    rows, route = _many(items)
    if rows is None:
        return route
    out_rows, boxes, second = [], [], []
    for p, k, row in zip(paths, kinds, rows):
        res = {"path": V.join_path(p), "className": classes[tuple(p)], "kind": k}
        if row.get("error"):
            res["error"] = row["error"]
            out_rows.append(res)
            continue
        rets = row.get("returns") or []
        if k == "part":
            vals = row.get("values") or {}
            ecf, esz = _cf(vals.get(EXTENTS[0])), _v3(vals.get(EXTENTS[1]))
            box = aabb(ecf, esz)
            res.update(cframe=_cf(vals.get("CFrame")), size=_r(_v3(vals.get("Size"))), extentsCFrame=ecf,
                       extentsSize=_r(esz), box=[_r(box[0]), _r(box[1])],
                       touching=[V.decode(x) for x in _items((rets[0] or [None])[0])] if rets and rets[0] else [])
            grow = [s + 2 * margin for s in esz]
            ov, why = V.encode("OverlapParams", {"filterType": "Exclude", "filter": [V.join_path(p)]}, cat.enums)
            second.append((len(out_rows), {"target": ["Workspace"], "calls": [
                {"method": "GetPartBoundsInBox", "args": [_wcf(ecf), _wv3(grow), ov]},
                {"method": "ArePartsTouchingOthers", "args": [{"$t": "Instances", "v": [_inst(p)]}]}]}))
        else:
            cf = _cf((rets[0] or [None])[0]) if rets and rets[0] else None
            sz = _v3(rets[0][1]) if rets and rets[0] and len(rets[0]) > 1 else None
            ext = _v3((rets[1] or [None])[0]) if len(rets) > 1 and rets[1] else None
            box = aabb(cf, sz) if cf and sz else None
            res.update(boundingBox={"cframe": cf, "size": _r(sz)}, extentsSize=_r(ext),
                       box=[_r(box[0]), _r(box[1])] if box else None)
        if box:
            boxes.append(box)
        out_rows.append(res)
    if second:
        rows2, _r2 = _many([it for _i, it in second])
        if rows2 is None:
            return _r2
        for (i, _it), row in zip(second, rows2):
            rets = row.get("returns") or []
            sup = [V.decode(x) for x in _items((rets[0] or [None])[0])] if rets and rets[0] else []
            out_rows[i]["supportedBy"] = sup
            out_rows[i]["touchingOthers"] = V.decode((rets[1] or [None])[0]) if len(rets) > 1 and rets[1] else None
            out_rows[i]["floating"] = not sup
    floating = [r["path"] for r in out_rows if r.get("floating")]
    whole = _box_union(boxes) if boxes else None
    return {"ok": True, "count": len(out_rows), "route": route, "margin": margin, "targets": out_rows,
            "floating": floating, "resting": [r["path"] for r in out_rows if r.get("floating") is False],
            "box": [_r(whole[0]), _r(whole[1])] if whole else None,
            "errors": [r for r in out_rows if r.get("error")]}


# ---------------------------------------------------------------- rbx_place_lint
STREAMING = ["StreamingEnabled", "StreamingTargetRadius", "StreamingMinRadius", "StreamingIntegrityMode"]
OVERLAP_TOL = 0.05                         # studs of interpenetration below which two parts count as touching
MAX_OVERLAP_PARTS = 5000                   # one GetPartsInPart per part


def _obb_depth(a, b):
    """How deep two oriented boxes ({cframe, size}) interpenetrate: the least overlap of their projections over the 15
    separating axes (0 when apart). Exact for two blocks; for other shapes, an upper bound from their extents boxes."""
    def axes(cf):
        r = cf["rotation"]
        return [[r[j], r[3 + j], r[6 + j]] for j in range(3)]
    aa, ba = axes(a["cframe"]), axes(b["cframe"])
    cands = aa + ba + [[x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]] for x in aa for y in ba]
    d = [b["cframe"]["position"][i] - a["cframe"]["position"][i] for i in range(3)]
    best = None
    for ax in cands:
        n = math.sqrt(sum(x * x for x in ax))
        if n < 1e-6:
            continue
        u = [x / n for x in ax]
        ra = sum(abs(sum(u[i] * e[i] for i in range(3))) * a["size"][k] / 2 for k, e in enumerate(aa))
        rb = sum(abs(sum(u[i] * e[i] for i in range(3))) * b["size"][k] / 2 for k, e in enumerate(ba))
        o = ra + rb - abs(sum(u[i] * d[i] for i in range(3)))
        if o <= 0:
            return 0.0
        best = o if best is None else min(best, o)
    return best or 0.0


def _excluded_part(rule, path, name, tags):
    import fnmatch
    if set(tags) & set(rule.get("tags") or []):
        return "tag"
    for pat in rule.get("names") or []:
        if fnmatch.fnmatchcase(name, pat) or fnmatch.fnmatchcase(path, pat):
            return "name %s" % pat
    return None


def _excluded_pair(rule, a, b):
    import fnmatch

    def m(p, pat):
        return fnmatch.fnmatchcase(p["path"], pat) or fnmatch.fnmatchcase(p["name"], pat)
    for pa, pb in rule.get("pairs") or []:
        if (m(a, pa) and m(b, pb)) or (m(a, pb) and m(b, pa)):
            return "pair %s / %s" % (pa, pb)
    if rule.get("same_model") and a.get("model") and a.get("model") == b.get("model"):
        return "same model %s" % a["model"]
    return None


def _check_exclude(exclude):
    if exclude is None:
        return {}, None
    if not isinstance(exclude, dict) or set(exclude) - {"tags", "names", "pairs", "same_model"}:
        return None, "exclude is {tags: [..], names: [glob..], pairs: [[globA, globB]..], same_model: bool}"
    for k in ("tags", "names"):
        if not isinstance(exclude.get(k, []), list) or not all(isinstance(x, str) and x for x in exclude.get(k, [])):
            return None, "exclude.%s is a list of strings" % k
    pairs = exclude.get("pairs", [])
    if not isinstance(pairs, list) or not all(isinstance(x, list) and len(x) == 2 and all(isinstance(y, str) and y for y in x)
                                              for x in pairs):
        return None, "exclude.pairs is a list of [globA, globB]"
    if not isinstance(exclude.get("same_model", False), bool):
        return None, "exclude.same_model is true or false"
    return exclude, None


def _overlap_audit(cat, rparts, parts, classes, tolerance, rule):
    """Each pair of parts under root that interpenetrates past tolerance: Workspace:GetPartsInPart per part (the
    exact geometry, held to root by an Include filter), the depth from the two parts' extents boxes."""
    root_path = V.join_path(rparts) or "game"
    props = ["CanQuery", "CanCollide"] + EXTENTS
    rows, route = _many([{"target": p, "props": props + (["Shape"] if classes.get(tuple(p)) == "Part" else [])}
                         for p in parts], calls=[{"method": "GetTags", "args": []}])
    if rows is None:
        return None, route
    info, skipped, unread, queryable = {}, [], [], []
    for p, row in zip(parts, rows):
        path = V.path_text(p)        # text: a dict key, a glob's subject and a pair's sort key (D5)
        if row.get("error"):
            unread.append({"path": path, "error": row["error"]})
            continue
        vals, rets = row.get("values") or {}, row.get("returns") or []
        tags = [V.decode(t) if isinstance(t, dict) else t for t in _items((rets[0] or [None])[0])] if rets and rets[0] else []
        shape = V.decode(vals.get("Shape")) if "Shape" in vals else None
        model = next((V.path_text(p[:i]) for i in range(len(p) - 1, len(rparts), -1)
                      if classes.get(tuple(p[:i])) == "Model"), None)
        e = {"path": path, "name": p[-1], "parts": p, "className": classes.get(tuple(p)), "tags": tags, "model": model,
             "cframe": _cf(vals.get(EXTENTS[0])), "size": _v3(vals.get(EXTENTS[1])),
             "block": classes.get(tuple(p)) == "Part" and shape in ("Block", "Enum.PartType.Block")}
        why = _excluded_part(rule, path, p[-1], tags)
        if why:
            skipped.append({"path": path, "by": why})
            continue
        info[tuple(p)] = e
        # CanQuery counts only on a part that does not collide: one with CanCollide true is still returned by
        # GetPartsInPart whatever CanQuery reads (measured live 2026-11-20, RB1604).
        if vals.get("CanQuery") is False and vals.get("CanCollide") is False:
            unread.append({"path": path, "error": "CanQuery and CanCollide are false: GetPartsInPart cannot see it, so its "
                                                  "overlaps with other parts are not checked"})
        else:
            queryable.append(e)
    if not info:
        return None, _no("every part under %s was excluded or unreadable: no overlap was checked" % V.path_text(rparts),
                         excluded=skipped[:50], unread=unread[:50])
    ov, why = V.encode("OverlapParams", {"filterType": "Include", "filter": [root_path]}, cat.enums)
    if why:
        return None, _no("OverlapParams: %s" % why)
    hits, hroute = _many([{"target": ["Workspace"], "calls": [{"method": "GetPartsInPart", "args": [_inst(e["parts"]), ov]}]}
                          for e in queryable])
    if hits is None:
        return None, hroute
    pairs, touching, excluded_pairs, seen = [], 0, [], set()
    for e, row in zip(queryable, hits):
        if row.get("error"):
            unread.append({"path": e["path"], "error": "GetPartsInPart: %s" % row["error"]})
            continue
        rets = row.get("returns") or []
        for h in _items((rets[0] or [None])[0]) if rets and rets[0] else []:
            hp = V.decode(h) if isinstance(h, dict) else h
            o = info.get(_key(hp))
            if o is None or o is e:
                continue
            key = tuple(sorted((e["path"], o["path"])))
            if key in seen:
                continue
            seen.add(key)
            why = _excluded_pair(rule, e, o)
            if why:
                excluded_pairs.append({"a": key[0], "b": key[1], "by": why})
                continue
            depth = _obb_depth(e, o) if e["cframe"] and e["size"] and o["cframe"] and o["size"] else None
            if depth is not None and depth <= tolerance:
                touching += 1
                continue
            pairs.append({"a": key[0], "b": key[1], "depth": None if depth is None else round(depth, 3),
                          "depthFrom": "blocks" if e["block"] and o["block"] else "extents boxes (an upper bound)"})
    pairs.sort(key=lambda x: -(x["depth"] or 0))
    return {"checked": len(queryable), "overlaps": pairs[:500], "overlapCount": len(pairs), "touchingWithinTolerance": touching,
            "tolerance": tolerance, "excludedParts": skipped[:200], "excludedPairs": excluded_pairs[:200],
            "unchecked": unread[:200], "route": hroute}, None


@_tool
def rbx_place_lint(root: Any = "Workspace", limit: int = MAX_LINT, overlaps: bool = False, tolerance: float = OVERLAP_TOL,
                   exclude: dict = None) -> dict:
    "Roblox Studio: a place audit under root (default Workspace), up to limit parts. Finds unanchored parts with no joints (`BasePart.Anchored` false and `BasePart:GetJoints` empty: they fall or drift when the game runs), parts whose position is below `Workspace.FallenPartsDestroyHeight` (destroyed at run time), and models with no `Model.PrimaryPart`; counts instances by class and reads the streaming settings (`Workspace.StreamingEnabled`, `Workspace.StreamingTargetRadius`, `Workspace.StreamingMinRadius`, `Workspace.StreamingIntegrityMode`). Each defect is reported once, with its kind and path; clean is true when there is none, every row read (unread empty) and limit cut nothing (truncated false); when no part row reads, ok is false. A root with no parts fails: nothing scanned is not clean. overlaps true (at most 5,000 parts) also lists each pair of parts under root that interpenetrates deeper than tolerance studs (default 0.05): `WorkspaceRoot:GetPartsInPart` per part, held to root by an Include filter, the depth the least overlap of the two parts' extents boxes over the separating axes (exact for two blocks, an upper bound otherwise); exclude {tags, names (globs on name or path), pairs [[globA, globB]], same_model} drops parts or pairs, each one counted; a part with CanQuery and CanCollide both false, which GetPartsInPart does not return, is listed unchecked; two parts that only touch faces are not returned (RB1604). Every overlap is a finding of kind part-overlap. Read only."
    rparts = [] if root == "game" else V.split_path(root)
    if rparts is None:
        return _no("root is a path", refusedRule="type")
    if not (isinstance(limit, int) and 1 <= limit <= MAX_LINT):
        return _no("limit is 1-%d" % MAX_LINT)
    if not isinstance(overlaps, bool) or not (isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool)
                                              and 0 <= tolerance <= 10):
        return _no("overlaps is true or false; tolerance 0-10 studs")
    rule, why = _check_exclude(exclude)
    if why:
        return _no(why, refusedRule="type")
    if overlaps and limit > MAX_OVERLAP_PARTS:
        limit = MAX_OVERLAP_PARTS
    cat, err = _cat()
    if err:
        return err
    ws = T._rbx("get_props", target=["Workspace"], props=["FallenPartsDestroyHeight"] + STREAMING)
    if not ws.get("ok"):
        return ws
    wv = {k: V.decode(v) for k, v in T._obj(ws.get("values")).items()}
    fall = wv.get("FallenPartsDestroyHeight")
    counts, parts, models, page, total, cut = {}, [], [], 0, 0, False
    classes = {}
    while True:
        q = T._rbx("query", root=rparts, page=page, pageSize=500)
        if not q.get("ok"):
            return q
        total = q.get("total") or 0
        for m in q.get("matches") or []:
            c = m.get("className")
            counts[c] = counts.get(c, 0) + 1
            classes[tuple(m["path"])] = c
            if c == "Terrain":
                continue
            if c and _is_a(cat, c, "BasePart"):
                if len(parts) < limit:
                    parts.append(m["path"])
                else:
                    cut = True
            elif c == "Model":
                models.append(m["path"])
        if not q.get("more"):
            break
        page += 1
    if not parts:
        return {"ok": False, "scanned": 0, "instances": total, "countsByClass": counts,
                "error": "no parts under %s: nothing was audited" % (V.join_path(rparts) or "game")}
    rows, route = _many([{"target": p} for p in parts], props=["Anchored", "CFrame"],
                        calls=[{"method": "GetJoints", "args": []}])
    if rows is None:
        return route
    findings, unread = [], []
    for p, row in zip(parts, rows):
        if row.get("error"):
            unread.append({"path": V.join_path(p), "error": row["error"]})
            continue
        vals, rets = row.get("values") or {}, row.get("returns") or []
        joints = _items((rets[0] or [None])[0]) if rets and rets[0] else []
        if vals.get("Anchored") is False and not joints:
            findings.append({"kind": "unanchored-no-joints", "path": V.join_path(p)})
        cf = _cf(vals.get("CFrame"))
        if cf and _num(fall) and cf["position"][1] < fall:
            findings.append({"kind": "below-destroy-height", "path": V.join_path(p), "y": round(cf["position"][1], 3),
                             "height": fall})
    if models:
        mrows, _mr = _many([{"target": m} for m in models], props=["PrimaryPart"])
        if mrows is None:
            return _mr
        for m, row in zip(models, mrows):
            pp = (row.get("values") or {}).get("PrimaryPart")
            if row.get("error"):
                unread.append({"path": V.join_path(m), "error": row["error"]})
            elif pp is None or (isinstance(pp, dict) and pp.get("$t") == "nil"):
                findings.append({"kind": "model-no-primary-part", "path": V.join_path(m)})
    audit = None
    if overlaps:
        audit, bad = _overlap_audit(cat, rparts, parts, classes, float(tolerance), rule)
        if bad:
            return bad
        findings += [{"kind": "part-overlap", "path": o["a"], "with": o["b"], "depth": o["depth"]} for o in audit["overlaps"]]
        if counts.get("MeshPart"):
            # what is inside one mesh is out of reach here: parts against parts only
            audit["insideMeshes"] = ("%d MeshPart(s) were checked as parts; the pieces modeled inside one mesh are not. "
                                     "Audit a kit piece's own parts in Blender before upload: bl_kit_audit." % counts["MeshPart"])
    by_kind = {}
    for f in findings:
        by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + 1
    if audit is not None:
        by_kind["part-overlap"] = audit["overlapCount"]
    # clean needs every row read and the whole root scanned: a row that did not read, or a part past limit, may hold
    # the defect (D18); a scan where no part row read audited nothing and fails
    parts_unread = sum(1 for row in rows if row.get("error"))
    out = {"overlapAudit": audit, "ok": parts_unread < len(parts), "clean": not findings and not unread and not cut,
           "scanned": len(parts), "models": len(models), "instances": total,
           "truncated": cut, "route": route, "findings": findings, "findingsByKind": by_kind,
           "countsByClass": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
           "fallenPartsDestroyHeight": fall, "streaming": {k: wv.get(k) for k in STREAMING}, "unread": unread}
    if not out["ok"]:
        out["error"] = "none of the %d parts under %s could be read: nothing was audited" % (
            len(parts), V.path_text(rparts))
    return out


# ---------------------------------------------------------------- rbx_fragment
@_tool
def rbx_fragment(target: Any, count: int = None, seed: int = None, site_spacing: float = None, origin: list = None,
                 radius: float = None, parent: Any = None, name: str = None, replace: bool = False,
                 options: dict = None) -> dict:
    "Roblox Studio: split a part into pieces with `GeometryService:FragmentAsync`. Sites come from count (that many points inside the part's extents box, seeded: the same seed gives the same pieces) or from `GeometryService:GenerateFragmentSites` (site_spacing, and origin with radius, as Roblox's page names them). options CollisionFidelity, RenderFidelity, FluidFidelity, SplitApart. The pieces go into a new Folder (name, default <part>_Fragments) under parent (default the part's parent) in one undo step; replace also takes the part out of the place in that step (Parent = nil, so the undo puts it back). Read back: every piece by path, and their summed volume - `BasePart:GetMass` over the part's density - within 1% of the part's (measured: 12 pieces of an 8 x 6 x 1 wall summed to 99.996%, RB702)."
    tparts, err = _parts(target)
    if err:
        return err
    if count is not None and (site_spacing is not None or origin is not None or radius is not None):
        return _no("give count (sites made here) or site_spacing / origin and radius (Studio's sites), not both")
    if count is not None and not (isinstance(count, int) and 2 <= count <= 200):
        return _no("count is 2-200 pieces")
    if count is not None and not (isinstance(seed, int) and not isinstance(seed, bool)):
        return _no("count needs a whole-number seed, so the pieces can be made again")
    if (origin is None) != (radius is None):
        return _no("give origin and radius together, or neither (Roblox's GeometryService page)")
    if not isinstance(replace, bool):
        return _no("replace is true or false")
    cat, err = _cat()
    if err:
        return err
    cls, err = T._resolve(tparts)
    if err:
        return err
    if not _is_a(cat, cls, "BasePart"):
        return _no("%s is a %s, not a part" % (V.join_path(tparts), cls), refusedRule="type")
    wopts, err = _options(options, CSG_OPTIONS, cat)
    if err:
        return err
    pparts = tparts[:-1]
    if parent is not None:
        pparts, err = _parts(parent, "parent")
        if err:
            return err
    ping = _ping()
    if not ping.get("ok"):
        return ping
    bad = _scratch_refusal(ping, [tparts, pparts], "rbx_fragment")
    if bad:
        return bad
    g = T._rbx("get_props", target=tparts, props=EXTENTS + ["CurrentPhysicalProperties"])
    if not g.get("ok"):
        return g
    vals = T._obj(g.get("values"))
    cf, sz = _cf(vals.get(EXTENTS[0])), _v3(vals.get(EXTENTS[1]))
    density = (V.decode(vals.get("CurrentPhysicalProperties")) or {}).get("density") \
        if isinstance(V.decode(vals.get("CurrentPhysicalProperties")), dict) else None
    m0 = _wcall(tparts, cls, "GetMass", [], read_only=True)
    mass0 = _ret(m0)
    if not (_num(mass0) and mass0 > 0):
        return dict(m0, ok=False, error="the part's mass did not read: %s" % (m0.get("error") or mass0))
    if count is not None:
        rng = random.Random(seed)
        sites = {"$t": "Array", "v": [_wv3(cf_mul(cf, {"position": [rng.uniform(-s / 2, s / 2) for s in sz],
                                                         "rotation": IDENT})["position"]) for _ in range(count)]}
        site_route = "seeded"
    else:
        sopts = {k: v for k, v in (("SiteSpacing", site_spacing), ("Origin", origin), ("Radius", radius)) if v is not None}
        w, err = _options(sopts or None, SITE_OPTIONS, cat)
        if err:
            return err
        s = _wcall(["GeometryService"], "GeometryService", "GenerateFragmentSites",
                   [_inst(tparts)] + ([w] if w else []), read_only=True)
        if not s.get("ok"):
            return dict(s, step="GenerateFragmentSites")
        sites, site_route = _ret(s), "GenerateFragmentSites"
    f = _wcall(["GeometryService"], "GeometryService", "FragmentAsync",
               [_inst(tparts), sites] + ([wopts] if wopts else []), read_only=True, timeout=120)
    if not f.get("ok"):
        return dict(f, step="FragmentAsync")
    handles = []
    for it in _items(_ret(f)):
        d = it.get("v") if isinstance(it, dict) and it.get("$t") == "Dictionary" else it
        h = ((d or {}).get("Instance") or {}).get("handle") if isinstance(d, dict) else None
        if h:
            handles.append(h)
    if not handles:
        return {"ok": False, "ran": True, "step": "FragmentAsync", "error": "FragmentAsync returned no piece: %r" % (_ret(f),)}
    masses = []
    for h in handles:
        m = _wcall({"handle": h}, "BasePart", "GetMass", [], read_only=True)
        masses.append(_ret(m) if m.get("ok") else None)
    folders, ferr = _free_names(pparts, name or (tparts[-1] + "_Fragments"), 1)
    if ferr:
        for h in handles:
            T._rbx("release_handle", handle=h)
        return dict(ferr, step="name the folder")
    folder = folders[0]
    fpath = pparts + [folder]
    placed = []
    with _Group("MifBridge fragment") as gr:
        if not gr.open:
            for h in handles:
                T._rbx("release_handle", handle=h)
            return dict(gr.began, ok=False, error="no undo group could be opened")
        mk = T._rbx("create", className="Folder", parent=pparts, name=folder, props={}, tags=[])
        if not mk.get("ok"):
            gr.end(False)
            return dict(mk, ok=False, step="folder")
        for i, h in enumerate(handles, 1):
            nm = T._rbx("set_props", target={"handle": h}, props={"Name": "Piece_%d" % i})
            mv = T._rbx("reparent", target={"handle": h}, parent=fpath) if nm.get("ok") else nm
            if not mv.get("ok") or not T.same_parent(mv.get("parentPath") or [], fpath):
                gr.end(False)
                for hh in handles:
                    T._rbx("release_handle", handle=hh)
                return dict(mv, ok=False, step="place", cancelled=True, error="placing a piece failed: %s" % mv.get("error"))
            placed.append(mv.get("path"))
        if replace:
            d = _remove(tparts)
            if not d.get("ok"):
                gr.end(False)
                return dict(d, ok=False, step="replace", cancelled=True)
        gr.end(True)
    for h in handles:
        T._rbx("release_handle", handle=h)
    found = [p for p in placed if T._rbx("resolve", target=p).get("ok")]
    total = sum(m for m in masses if _num(m))
    ratio = total / mass0
    problems = []
    if len(found) != len(handles):
        problems.append("%d of %d pieces found by path" % (len(found), len(handles)))
    if any(m is None for m in masses):
        problems.append("a piece's mass did not read")
    if abs(ratio - 1) > VOLUME_TOL:
        problems.append("the pieces hold %.3f%% of the part's volume" % (ratio * 100))
    out = {"ok": not problems, "verified": not problems, "pieces": len(found), "folder": V.join_path(fpath),
           "paths": [V.join_path(p) for p in found[:50]], "sites": site_route, "seed": seed if count else None,
           "massBefore": mass0, "massAfter": total, "volumeRatio": round(ratio, 6),
           "volumeBefore": round(mass0 / density, 4) if _num(density) and density > 0 else None,
           "volumeAfter": round(total / density, 4) if _num(density) and density > 0 else None,
           "replaced": V.join_path(tparts) if replace else None, "undoSteps": 1}
    if problems:
        out["error"] = "; ".join(problems)
    return out


# ---------------------------------------------------------------- greedy voxel merge
MERGE_MAX_CELLS = 200000
MERGE_MAX_BOXES = 2000
MERGE_BATCH = 100                               # rbx_batch takes at most 100 ops: each batch is one undo step
AXES = "xyz"
ORDERS = ("xyz", "xzy", "yxz", "yzx", "zxy", "zyx")


def _greedy(grid, order, flip):
    """One greedy pass: grow each seed along order[0], then order[1], then order[2], seeds taken lowest first after the
    axes in flip are mirrored (so a pass can also start from the top or the far side). [(lo, hi, texture)] in cells."""
    ax = [AXES.index(a) for a in order]
    sg = [-1 if f else 1 for f in flip]

    def q_of(c):
        m = [c[i] * sg[i] for i in range(3)]
        return (m[ax[0]], m[ax[1]], m[ax[2]])
    q = {q_of(c): t for c, t in grid.items()}
    used, boxes = set(), []
    for s in sorted(q, key=lambda c: (c[2], c[1], c[0])):
        if s in used:
            continue
        t = q[s]

        def free(c):
            return c not in used and q.get(c) == t
        n0 = 1
        while free((s[0] + n0, s[1], s[2])):
            n0 += 1
        n1 = 1
        while all(free((s[0] + i, s[1] + n1, s[2])) for i in range(n0)):
            n1 += 1
        n2 = 1
        while all(free((s[0] + i, s[1] + j, s[2] + n2)) for i in range(n0) for j in range(n1)):
            n2 += 1
        for i in range(n0):
            for j in range(n1):
                for k in range(n2):
                    used.add((s[0] + i, s[1] + j, s[2] + k))
        lo_q, hi_q = s, (s[0] + n0 - 1, s[1] + n1 - 1, s[2] + n2 - 1)
        a, b = [0, 0, 0], [0, 0, 0]
        for n in range(3):
            a[ax[n]], b[ax[n]] = lo_q[n] * sg[ax[n]], hi_q[n] * sg[ax[n]]
        lo = [min(a[i], b[i]) for i in range(3)]
        hi = [max(a[i], b[i]) for i in range(3)]
        boxes.append((lo, hi, t))
    return boxes


# Cells times passes one merge may run (D23g): a pass is pure Python over every cell, so 48 passes over 200,000 cells
# ran unbounded. 2,400,000 lets every pass run up to 50,000 cells; past that the passes are cut to fit, at least one.
MERGE_WORK = 2400000


def merge_cells(grid, orders=ORDERS, work=MERGE_WORK, info=None):
    """Greedy boxes for {(x, y, z): texture}: every growth order in orders, each from all eight corners, the fewest boxes
    kept (ties: the first). Answers (boxes, per-order counts, the winner {order, flip}). The passes are held to work
    (cells x passes, at least one pass): when they do not all fit, the first corner of every order runs first, then the
    next corner of each, so a cut run still tries every order. info (a dict) gets passes and passesPossible."""
    best, per, win = None, {}, None
    plan = [(order, f) for order in orders for f in range(8)]
    budget = max(1, int(work // max(1, len(grid))))
    if budget < len(plan):
        plan = [(order, f) for f in range(8) for order in orders][:budget]
    if isinstance(info, dict):
        info.update(passes=len(plan), passesPossible=8 * len(orders))
    for order, f in plan:
        flip = (bool(f & 1), bool(f & 2), bool(f & 4))
        boxes = _greedy(grid, order, flip)
        if order not in per or len(boxes) < per[order]:
            per[order] = len(boxes)
        if best is None or len(boxes) < len(best):
            best, win = boxes, {"order": order, "fromHigh": [AXES[i] for i in range(3) if flip[i]]}
    return best or [], per, win


def _merge_input(cells):
    """cells as [[x, y, z, texture], ...] or {texture: [[x, y, z], ...]} -> ({(x, y, z): texture}, None) or (None, why)."""
    grid = {}
    rows = []
    if isinstance(cells, dict):
        for t, pts in cells.items():
            if not isinstance(pts, list):
                return None, "cells as {texture: [[x, y, z], ...]}: %r is not a list" % (t,)
            rows += [list(p) + [t] for p in pts if isinstance(p, (list, tuple))]
            if len(rows) > MERGE_MAX_CELLS:
                break
    elif isinstance(cells, list):
        rows = cells
    else:
        return None, "cells is [[x, y, z, texture], ...] or {texture: [[x, y, z], ...]}"
    if not rows or len(rows) > MERGE_MAX_CELLS:
        return None, "cells holds 1-%d cells" % MERGE_MAX_CELLS
    for r in rows:
        if not (isinstance(r, (list, tuple)) and len(r) == 4 and all(isinstance(v, int) and not isinstance(v, bool)
                                                                       for v in r[:3])
                and isinstance(r[3], (str, int)) and not isinstance(r[3], bool) and str(r[3])):
            return None, "a cell is [x, y, z, texture] with whole-number x, y, z and a texture name or number: %r" % (r,)
        c = (r[0], r[1], r[2])
        if c in grid and grid[c] != str(r[3]):
            return None, "cell %s is given twice, as %s and %s" % (list(c), grid[c], r[3])
        grid[c] = str(r[3])
    return grid, None


@_tool
def rbx_voxel_merge(cells: Any, cell_size: Any = 4, origin: list = None, parent: Any = None, textures: dict = None,
                    name: str = "Box", orders: list = None, dry_run: bool = False) -> dict:
    "Roblox Studio scenery: cells on a grid in, axis-aligned boxes out, so a patterned wall or hall costs a part budget of boxes instead of a part per cell. cells is [[x, y, z, texture], ...] or {texture: [[x, y, z], ...]} (whole-number cells, at most 200,000); runs of one texture merge greedily - grow along one axis, then the second, then the third - under each of the six axis orders (orders narrows them), each started from all eight corners (passes says how many ran: above 50,000 cells they are cut to fit 2,400,000 cell-passes, every order's first corner first), and the fewest boxes win: one fixed order grew each pillar of a 26-portal hall up into the band over the openings and cut the band into 26 pieces (53 boxes) where the best gave 28. Answers boxes (cells lo and hi, size in cells, texture, and in studs position and size from origin and cell_size, a number or [x, y, z]), the count per order, the winner and the cell and box counts. Every cell is covered exactly once (checked). With parent (and not dry_run) each box becomes an anchored Part named <name>_<texture>_<n> under it with textures[texture] {props, tags} applied - Material, Color, CollectionService tags - made through rbx_batch, 100 boxes a batch, each batch one undo step and every Part read back."
    grid, why = _merge_input(cells)
    if why:
        return {"ok": False, "ran": False, "error": why}
    cs = [cell_size] * 3 if isinstance(cell_size, (int, float)) and not isinstance(cell_size, bool) else cell_size
    if not (isinstance(cs, list) and len(cs) == 3 and all(_num(v) and v > 0 for v in cs)):
        return {"ok": False, "ran": False, "error": "cell_size is a size in studs above 0, or [x, y, z]"}
    org = origin if origin is not None else [0, 0, 0]
    if not (isinstance(org, list) and len(org) == 3 and all(_num(v) for v in org)):
        return {"ok": False, "ran": False, "error": "origin is [x, y, z] in studs: the low corner of cell 0, 0, 0"}
    use = ORDERS if orders is None else orders
    if not (isinstance(use, list) or use is ORDERS) or not use or any(o not in ORDERS for o in use):
        return {"ok": False, "ran": False, "error": "orders is a list of %s" % ", ".join(ORDERS)}
    if not isinstance(dry_run, bool) or not isinstance(name, str) or not re.match(r"^[A-Za-z][A-Za-z0-9_]{0,40}$", name):
        return {"ok": False, "ran": False, "error": "dry_run is true or false; name is letters, digits and _"}
    textures = textures or {}
    if not isinstance(textures, dict) or not all(isinstance(v, dict) and set(v) <= {"props", "tags"} for v in textures.values()):
        return {"ok": False, "ran": False, "error": "textures is {texture: {props?, tags?}}"}
    passes = {}
    boxes, per, win = merge_cells(grid, tuple(use), info=passes)
    covered = {}
    for lo, hi, t in boxes:
        for x in range(lo[0], hi[0] + 1):
            for y in range(lo[1], hi[1] + 1):
                for z in range(lo[2], hi[2] + 1):
                    covered[(x, y, z)] = covered.get((x, y, z), 0) + (1 if grid.get((x, y, z)) == t else 1000)
    exact = len(covered) == len(grid) and all(v == 1 for v in covered.values())
    rows = []
    for lo, hi, t in boxes:
        n = [hi[i] - lo[i] + 1 for i in range(3)]
        rows.append({"texture": t, "lo": lo, "hi": hi, "cells": n,
                     "position": [round(org[i] + (lo[i] + n[i] / 2.0) * cs[i], 6) for i in range(3)],
                     "size": [round(n[i] * cs[i], 6) for i in range(3)]})
    out = {"ok": exact, "cells": len(grid), "boxes": len(boxes), "perOrder": per, "best": win,
           "textures": sorted(set(grid.values())), "exact": exact, "rows": rows if len(rows) <= 500 else rows[:500],
           "rowsShown": min(len(rows), 500), "passes": passes.get("passes"),
           "passesPossible": passes.get("passesPossible")}
    if not exact:
        return dict(out, error="the merge did not cover every cell exactly once (a bug): nothing was made")
    if parent is None or dry_run:
        return dict(out, sent=False)
    if len(boxes) > MERGE_MAX_BOXES:
        return dict(out, ok=False, ran=False, error="%d boxes is over %d a call" % (len(boxes), MERGE_MAX_BOXES))
    pparts, err = T._target(parent, "parent")
    if err:
        return err
    if not T._rbx("resolve", target=pparts).get("ok"):
        return dict(out, ok=False, ran=False, refusedRule="not-found", error="parent %s does not exist" % V.join_path(pparts))
    seq, ops = {}, []
    for r in rows:
        seq[r["texture"]] = seq.get(r["texture"], 0) + 1
        tx = textures.get(r["texture"]) or {}
        props = dict(tx.get("props") or {})
        props.update(Anchored=True, Size=r["size"], CFrame=r["position"] + [1, 0, 0, 0, 1, 0, 0, 0, 1])
        a = {"class_name": "Part", "parent": pparts, "props": props,
             "name": "%s_%s_%d" % (name, re.sub(r"[^A-Za-z0-9_]", "_", r["texture"]), seq[r["texture"]])}
        if tx.get("tags"):
            a["tags"] = list(tx["tags"])
        ops.append({"tool": "rbx_create", "args": a})
    made, problems = [], []
    for i in range(0, len(ops), MERGE_BATCH):
        b = T.rbx_batch(ops[i:i + MERGE_BATCH])
        if not b.get("ok"):
            problems.append("batch %d (boxes %d-%d): %s" % (i // MERGE_BATCH, i, min(i + MERGE_BATCH, len(ops)) - 1,
                                                            b.get("error") or b.get("problems")))
            break
        made += [x.get("path") for x in b.get("results") or []]
    out.update(ok=not problems, verified=not problems, sent=True, parent=V.join_path(pparts), made=len(made),
               batches=(len(ops) + MERGE_BATCH - 1) // MERGE_BATCH, paths=[V.join_path(p) if isinstance(p, list) else p
                                                                           for p in made[:50]])
    if problems:
        out["error"] = "; ".join(problems) + ("; %d boxes made before it, each batch its own undo step" % len(made))
    return out


# The four extensions join __all__ so tools.py's star import rebinds its own names to them.
__all__ += list(EXTENDED)

# Imported before tools.py finished (this module first, standalone): tools.py's star import ran while this module was
# still empty, so register here what it would have taken.
if hasattr(T, "TOOLS"):
    for _n in __all__:
        setattr(T, _n, globals()[_n])
        T.TOOLS[_n] = globals()[_n]
    T.PACK["tools"] = sorted(T.TOOLS)
