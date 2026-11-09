"""MifBlender ops: texel density set to a number, and the texture size that holds it.

THE GAP (spec, "Blender: images, texture density and painting", 2026-10-04): mesh_quality measured a density for one
square texture size and uv_pack brought every island to the MEDIAN density; nothing set a density to a value, said which
texture size holds it, or packed for a non-square texture. A target is a game-art choice (10 px/cm "decent", 10.24 for a
first-person prop, 20.48 for a held weapon, 5.12 third-person, 2.56 top-down), so it is always the caller's and never a
default.

DENSITY, the definition every number here uses: pixels per metre = sqrt(UV area x W x H / world area) for an island on a
W x H texture, with the UV area unsigned (a folded face counts whole, as uv_pack equalises) and the world area through the
object's rotation and scale. 1 px/cm is 100 px/m, the unit mesh_quality reports. A 2048 map on a 1 m plane and a 4096
map on a 2 m plane are both 2048 px/m = 20.48 px/cm.

THE LADDER IS A REAL PACK, not an area sum: at each texture size the islands are scaled to the target for that size and
Blender's pack_islands runs with scale off - every build has the option (measured 2026-10-04: pack_islands takes scale and
merge_overlap on 3.6.23, 4.4.0 and 5.2.1) - and the size fits when every island lands inside 0-1. The trial packs the
islands' bounding boxes, so a size it calls a fit holds the islands, and a tight layout may fit one size smaller once
packed by shape (pack: true). The UVs are put back after each trial.
"""
from __future__ import annotations

import array
import math

import bmesh
import bpy

from .ops_common import (MifOpError, finite_float, finite_floats, get_object, reject_unknown, require_editable,
                         select_only, selection_restore, selection_snapshot, shared_data_note, take, take_bool,
                         take_float, take_int)
from .ops_mesh import (_PACK_PINNED, _face_area_through, _face_uv_area, _run_pack_islands, _scale_layout_islands,
                       _uv_islands)

_DEFAULT_LADDER = (256, 512, 1024, 2048, 4096, 8192)


def _texture_size(raw, key="textureSize"):
    """(W, H) from a number or [w, h]."""
    if isinstance(raw, bool) or raw is None:
        raise MifOpError("%s is a size in pixels - a number or [w, h]. NOTHING was changed." % key)
    if isinstance(raw, (int, float)):
        w = h = finite_float(raw, key)
    elif isinstance(raw, (list, tuple)) and len(raw) == 2 and not any(isinstance(v, bool) for v in raw):
        w, h = finite_floats(list(raw), key)
    else:
        raise MifOpError("%s takes a number or [w, h], got %r. NOTHING was changed." % (key, raw))
    if w < 1 or h < 1 or w > 65536 or h > 65536:
        raise MifOpError("%s %s is outside 1-65536 pixels. NOTHING was changed." % (key, raw))
    return float(w), float(h)


def _target_px_per_m(params):
    """(target in px per metre, the unit asked, the target as given, derivedFrom) - from target or fromScreen."""
    unit = str(take(params, "unit", default="PX_PER_CM", kind=str)).upper()
    if unit not in ("PX_PER_CM", "PX_PER_M"):
        raise MifOpError("unit is PX_PER_CM or PX_PER_M, got %r. NOTHING was changed." % unit)
    target = take_float(params, "target", default=None)
    screen = params.get("fromScreen")
    if (target is None) == (screen is None):
        raise MifOpError("pass target (a density, %s) OR fromScreen {sizeCm, pixels} - a target is a game-art choice "
                         "and there is no default. NOTHING was changed." % unit)
    derived = None
    if screen is not None:
        if not isinstance(screen, dict) or set(screen) != {"sizeCm", "pixels"}:
            raise MifOpError("fromScreen is {sizeCm, pixels}: an object that size covers that many pixels of the frame at "
                             "its closest. Got %r. NOTHING was changed." % (screen,))
        size_cm, pixels = finite_floats([screen["sizeCm"], screen["pixels"]], "fromScreen")
        if size_cm <= 0 or pixels <= 0:
            raise MifOpError("fromScreen sizeCm and pixels must be above 0. NOTHING was changed.")
        per_cm = pixels / size_cm
        target = per_cm if unit == "PX_PER_CM" else per_cm * 100.0
        derived = {"sizeCm": size_cm, "pixels": pixels, "pxPerCm": round(per_cm, 6)}
    if target <= 0:
        raise MifOpError("target must be above 0, got %g. NOTHING was changed." % target)
    per_m = target * 100.0 if unit == "PX_PER_CM" else target
    return per_m, unit, target, derived


def _density(uva, a3, w, h):
    return (uva * w * h / a3) ** 0.5 if a3 > 1e-12 and uva > 1e-12 else None


def density_stats(values, unit):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    div = 100.0 if unit == "PX_PER_CM" else 1.0
    n = len(vals)
    med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    return {"min": round(vals[0] / div, 4), "median": round(med / div, 4), "max": round(vals[-1] / div, 4),
            "spread": round(vals[-1] / vals[0], 4) if vals[0] > 0 else None}


def _read_uvs(me, name):
    layer = me.uv_layers[name]
    buf = array.array("f", [0.0]) * (2 * len(layer.data))
    layer.data.foreach_get("uv", buf)
    return buf


def _write_uvs(me, name, buf):
    me.uv_layers[name].data.foreach_set("uv", buf)
    me.update()


def _measure(obj, name, wanted, w, h):
    """[(island faces' indices, uv area, world area, density)] for the islands this call works on."""
    me = obj.data
    m3 = obj.matrix_world.to_3x3()
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        uv = bm.loops.layers.uv[name]
        islands = _uv_islands(bm, uv)[0]
        rows = []
        for i, isl in enumerate(islands):
            if wanted is not None and i not in wanted:
                continue
            uva = sum(_face_uv_area(f, uv) for f in isl)
            a3 = sum(_face_area_through(f, m3) for f in isl)
            rows.append({"island": i, "faces": [f.index for f in isl], "uva": uva, "a3": a3,
                         "density": _density(uva, a3, w, h)})
        return rows, len(islands)
    finally:
        bm.free()


def _scale_islands(obj, name, factors):
    """Scale each island's UVs about its own bounding-box centre: factors {island index: k}."""
    me = obj.data
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        uv = bm.loops.layers.uv[name]
        islands = _uv_islands(bm, uv)[0]
        for i, k in factors.items():
            loops = [lp for f in islands[i] for lp in f.loops]
            us = [lp[uv].uv.x for lp in loops]
            vs = [lp[uv].uv.y for lp in loops]
            cx, cy = 0.5 * (min(us) + max(us)), 0.5 * (min(vs) + max(vs))
            for lp in loops:
                lp[uv].uv = (cx + (lp[uv].uv.x - cx) * k, cy + (lp[uv].uv.y - cy) * k)
        bm.to_mesh(me)
    finally:
        bm.free()
    me.update()


def texel_shape(obj, name, w, h):
    """How square the texels are on a w x h texture: per triangle, the longer side of a texel over the shorter (the
    singular values of the world-to-pixel map), as {median, max} - 1 is square. A layout unwrapped for a square tile
    reads w / h on a non-square one until V is stretched (uv_pack textureSize)."""
    me = obj.data
    m3 = obj.matrix_world.to_3x3()
    bm = bmesh.new()
    ratios = []
    try:
        bm.from_mesh(me)
        uv = bm.loops.layers.uv[name]
        for f in bm.faces:
            ls = f.loops
            p0, q0 = m3 @ ls[0].vert.co, ls[0][uv].uv
            for k in range(1, len(ls) - 1):
                e1, e2 = m3 @ ls[k].vert.co - p0, m3 @ ls[k + 1].vert.co - p0
                n = e1.cross(e2)
                if e1.length < 1e-9 or n.length < 1e-12:
                    continue
                x = e1.normalized()
                y = n.cross(e1).normalized()
                ax, bx, by = e1.length, e2.dot(x), e2.dot(y)
                q1, q2 = ls[k][uv].uv, ls[k + 1][uv].uv
                f1 = ((q1.x - q0.x) * w, (q1.y - q0.y) * h)
                f2 = ((q2.x - q0.x) * w, (q2.y - q0.y) * h)
                # J maps the triangle's own 2D frame to pixels: J [a b] = [f1 f2] with a = (ax, 0), b = (bx, by)
                j00, j10 = f1[0] / ax, f1[1] / ax
                j01, j11 = (f2[0] - j00 * bx) / by, (f2[1] - j10 * bx) / by
                t = j00 * j00 + j01 * j01 + j10 * j10 + j11 * j11
                d = abs(j00 * j11 - j01 * j10)
                disc = max(t * t - 4.0 * d * d, 0.0) ** 0.5
                s2 = max((t - disc) / 2.0, 0.0) ** 0.5
                if s2 > 1e-12:
                    ratios.append(((t + disc) / 2.0) ** 0.5 / s2)
    finally:
        bm.free()
    if not ratios:
        return None
    ratios.sort()
    return {"median": round(ratios[len(ratios) // 2], 4), "max": round(ratios[-1], 4)}


def pack_without_scaling(obj, name, margin, rotate=True, boxes=False):
    """Blender's pack_islands with scale OFF, so densities stay as they are; returns the islands left outside 0-1, by
    index. boxes packs bounding boxes (shape_method AABB): the ladder's fit test, because a concave pack with rotation
    took 15 to 23 s for a 6-island cube on 3.6.23 and under 0.01 s as boxes (2026-10-04), and boxes that fit mean the
    islands fit."""
    pinned = dict(_PACK_PINNED, rotate=rotate, scale=False)
    if boxes:
        pinned["shape_method"] = "AABB"
    if not rotate:
        pinned.pop("rotate_method")
    _run_pack_islands(obj, name, margin, pinned)
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        uv = bm.loops.layers.uv[name]
        outside = []
        for i, isl in enumerate(_uv_islands(bm, uv)[0]):
            us = [lp[uv].uv.x for f in isl for lp in f.loops]
            vs = [lp[uv].uv.y for f in isl for lp in f.loops]
            if min(us) < -1e-5 or min(vs) < -1e-5 or max(us) > 1 + 1e-5 or max(vs) > 1 + 1e-5:
                outside.append(i)
    finally:
        bm.free()
    return outside


_SET_TD_KEYS = ("object", "name", "uvLayer", "target", "unit", "textureSize", "fromScreen", "faces", "islands", "mode",
                "ladder", "pack", "margin")


def op_uv_set_texel_density(params):
    """Scale UV islands so they carry a target texel density on a texture size, and say which size holds the layout.

    params:
      object (str), uvLayer (str, default the active one)
      target (float) + unit PX_PER_CM (default) | PX_PER_M - OR fromScreen {sizeCm, pixels} (the density an object of
                   that size needs to fill that many pixels of the frame at its closest)
      textureSize (int | [w, h])   required: the texture the density is measured on
      faces [indices] | islands [indices]   which islands (default all); a face picks its island
      mode PER_ISLAND (default: each island to the target) | AVERAGE (the selection as one, relative sizes kept)
      ladder [sizes]   the texture sizes to test (default 256-8192, powers of two); a non-square textureSize keeps its
                       aspect on every rung
      pack (bool)      also pack the layout at the target (scale off) on textureSize; margin default 0.02
    Reply: density per island before and after (min, median, max, spread, in the unit asked), uvFillPercent at
    textureSize, each ladder rung's fill and whether a pack at the target fits in 0-1, smallestFittingSize, and with pack
    the islands left outside 0-1.
    """
    reject_unknown(params, _SET_TD_KEYS, "uv_set_texel_density")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    require_editable(obj, "set the texel density of")
    me = obj.data
    if not me.uv_layers:
        raise MifOpError("'%s' has no UV layer - unwrap it first (uv_unwrap). NOTHING was changed." % obj.name)
    names = [u.name for u in me.uv_layers]
    layer = take(params, "uvLayer", default=None, kind=str)
    if layer is not None and layer not in names:
        raise MifOpError("'%s' has no UV layer %r (it has %s). NOTHING was changed." % (obj.name, layer, ", ".join(names)))
    name = layer or (me.uv_layers.active or me.uv_layers[0]).name
    per_m, unit, target, derived = _target_px_per_m(params)
    if params.get("textureSize") is None:
        raise MifOpError("textureSize is required - a density means nothing without the texture it is on. NOTHING was "
                         "changed.")
    w, h = _texture_size(params.get("textureSize"))
    square = w == h
    mode = str(take(params, "mode", default="PER_ISLAND", kind=str)).upper()
    if mode not in ("PER_ISLAND", "AVERAGE"):
        raise MifOpError("mode is PER_ISLAND or AVERAGE, got %r. NOTHING was changed." % mode)
    ladder = params.get("ladder")
    if ladder is None:
        ladder = list(_DEFAULT_LADDER)
    elif not isinstance(ladder, (list, tuple)) or not ladder or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or v < 1 for v in ladder):
        raise MifOpError("ladder is a list of texture sizes in pixels, got %r. NOTHING was changed." % (ladder,))
    pack = take_bool(params, "pack", default=False)
    margin = take_float(params, "margin", default=0.02)
    if not 0.0 <= margin < 0.5:
        raise MifOpError("margin must be in [0, 0.5), got %g. NOTHING was changed." % margin)
    if params.get("margin") is not None and not pack:
        raise MifOpError("margin is the pack's; pass pack true, or drop it. NOTHING was changed.")

    all_rows, n_islands = _measure(obj, name, None, w, h)
    face_island = {f: r["island"] for r in all_rows for f in r["faces"]}
    wanted = None
    if params.get("faces") is not None and params.get("islands") is not None:
        raise MifOpError("pass faces OR islands, not both. NOTHING was changed.")
    if params.get("faces") is not None:
        raw = params["faces"]
        if not isinstance(raw, (list, tuple)) or not raw:
            raise MifOpError("faces is a list of face indices. NOTHING was changed.")
        bad = [f for f in raw if isinstance(f, bool) or not isinstance(f, int) or f not in face_island]
        if bad:
            raise MifOpError("faces %s are not face indices of '%s' (it has %d). NOTHING was changed."
                             % (bad[:10], obj.name, len(me.polygons)))
        wanted = {face_island[f] for f in raw}
    elif params.get("islands") is not None:
        raw = params["islands"]
        if not isinstance(raw, (list, tuple)) or not raw:
            raise MifOpError("islands is a list of island indices (uv_info lists them). NOTHING was changed.")
        bad = [i for i in raw if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n_islands]
        if bad:
            raise MifOpError("islands %s are not island indices of '%s' (it has %d). NOTHING was changed."
                             % (bad[:10], obj.name, n_islands))
        wanted = set(raw)
    rows = [r for r in all_rows if wanted is None or r["island"] in wanted]
    usable = [r for r in rows if r["density"] is not None]
    if not usable:
        raise MifOpError("none of the %d island(s) chosen has area both in UV and in the world, so none has a density. "
                         "NOTHING was changed." % len(rows))
    if mode == "PER_ISLAND":
        factors = {r["island"]: per_m / r["density"] for r in usable}
    else:
        avg = _density(sum(r["uva"] for r in usable), sum(r["a3"] for r in usable), w, h)
        factors = {r["island"]: per_m / avg for r in usable}
    before_stats = density_stats([r["density"] for r in usable], unit)

    saved = _read_uvs(me, name)
    try:
        _scale_islands(obj, name, factors)
        set_uvs = _read_uvs(me, name)
        after_rows = _measure(obj, name, wanted, w, h)[0]
        fill = sum(r["uva"] for r in _measure(obj, name, None, w, h)[0])
        # THE LADDER, each rung packed at the target with scale off and put back
        rungs = []
        aspect = h / w
        smallest = None
        for size in sorted(float(v) for v in ladder):
            sw, sh = size, size * aspect
            k = (w * h / (sw * sh)) ** 0.5
            row = {"size": [int(round(sw)), int(round(sh))],
                   "fillPercent": round(100.0 * fill * (w * h) / (sw * sh), 2)}
            if row["fillPercent"] > 100.0:
                row["fits"] = False
                row["why"] = "the islands at this density need more than the whole tile"
            else:
                _scale_layout_islands(me, name, k)
                outside = pack_without_scaling(obj, name, 0.02, rotate=square, boxes=True)
                row["fits"] = not outside
                if outside:
                    row["islandsOutside"] = len(outside)
                _write_uvs(me, name, set_uvs)
            if row["fits"] and smallest is None:
                smallest = row["size"]
            rungs.append(row)
        packed = None
        if pack:
            outside = pack_without_scaling(obj, name, margin, rotate=square)
            packed = {"islandsOutside": outside or None, "margin": margin, "rotated": square}
    except Exception:
        _write_uvs(me, name, saved)
        raise
    out = {
        "object": obj.name, "layer": name, "unit": unit, "target": round(target, 6),
        "targetPxPerMetre": round(per_m, 4), "derivedFrom": derived, "textureSize": [int(w), int(h)], "mode": mode,
        "islands": len(rows), "islandsScaled": len(factors),
        "islandsWithoutArea": [r["island"] for r in rows if r["density"] is None] or None,
        "densityBefore": before_stats,
        "densityAfter": density_stats([r["density"] for r in after_rows if r["island"] in factors], unit),
        "uvFillPercent": round(100.0 * fill, 2),
        "ladder": rungs, "smallestFittingSize": smallest,
        "packed": packed,
        "texelShape": texel_shape(obj, name, w, h),
    }
    if not square and (out["texelShape"] or {}).get("median", 1.0) > 1.05:
        out["texelShapeNote"] = ("texels on this %d x %d texture are %.2f times longer one way than the other - the "
                                 "layout was unwrapped for a square tile; uv_pack textureSize stretches V to square them"
                                 % (w, h, out["texelShape"]["median"]))
    if smallest is None:
        out["ladderNote"] = "no rung of the ladder holds the layout at this target - lower the target or add a larger size"
    out.update(shared_data_note(obj))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# ISLANDS ONE BY ONE (2026-10-06, spec "UV islands: transform one by one, straighten to a grid, match texel density"):
# uv_unwrap's uvTransform scales and offsets the WHOLE layer and offers no rotation, and uv_set_texel_density scales but
# never moves or turns an island. A handle's two halves resized to one density, an island parked on one tile of an atlas,
# an island turned to run with its neighbour, and an island laid out as a grid of squares so a straight pattern line
# runs across it - each was a hand edit in the UV editor.

def _uv_layer_name(obj, params, what):
    """The UV layer a call works on: uvLayer (or layer), else the active one. Refuses a mesh with none."""
    me = obj.data
    if not me.uv_layers:
        raise MifOpError("'%s' has no UV layer - unwrap it first (uv_unwrap). NOTHING was %s." % (obj.name, what))
    names = [u.name for u in me.uv_layers]
    layer = take(params, "uvLayer", "layer", default=None, kind=str)
    if layer is not None and layer not in names:
        raise MifOpError("'%s' has no UV layer %r (it has %s). NOTHING was %s."
                         % (obj.name, layer, ", ".join(names), what))
    return layer or (me.uv_layers.active or me.uv_layers[0]).name


def _chosen_islands(params, obj, rows, n_islands, what, required=True):
    """The island indices a call works on, from islands [i] or faces [f] (a face picks its island). None = all."""
    if params.get("faces") is not None and params.get("islands") is not None:
        raise MifOpError("pass faces OR islands, not both. NOTHING was %s." % what)
    if params.get("faces") is not None:
        raw = params["faces"]
        face_island = {f: r["island"] for r in rows for f in r["faces"]}
        if not isinstance(raw, (list, tuple)) or not raw:
            raise MifOpError("faces is a list of face indices. NOTHING was %s." % what)
        bad = [f for f in raw if isinstance(f, bool) or not isinstance(f, int) or f not in face_island]
        if bad:
            raise MifOpError("faces %s are not face indices of '%s' (it has %d). NOTHING was %s."
                             % (bad[:10], obj.name, len(obj.data.polygons), what))
        return sorted({face_island[f] for f in raw})
    if params.get("islands") is not None:
        raw = params["islands"]
        if not isinstance(raw, (list, tuple)) or not raw:
            raise MifOpError("islands is a list of island indices (uv_info islands:true lists them). NOTHING was %s."
                             % what)
        bad = [i for i in raw if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n_islands]
        if bad:
            raise MifOpError("islands %s are not island indices of '%s' (it has %d, 0-%d). NOTHING was %s."
                             % (bad[:10], obj.name, n_islands, n_islands - 1, what))
        return sorted(set(raw))
    if required:
        raise MifOpError("say which islands: islands [indices] (uv_info islands:true lists them) or faces [indices] (a "
                         "face picks its island). NOTHING was %s." % what)
    return list(range(n_islands))


def _island_loops(me, face_indices):
    return [li for f in face_indices for li in me.polygons[f].loop_indices]


def _island_box(buf, loops):
    us = [buf[2 * li] for li in loops]
    vs = [buf[2 * li + 1] for li in loops]
    return min(us), min(vs), max(us), max(vs)


def _island_median(buf, loops):
    """The mean of the island's distinct UV corners - Blender's Median Point over UVs, a shared corner counted once."""
    pts = {(round(buf[2 * li], 6), round(buf[2 * li + 1], 6)) for li in loops}
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def _px_per_cm(density):
    return round(density / 100.0, 4) if density is not None else None


def _island_rows_out(before, after, buf_before, buf_after, me):
    out = []
    for i in sorted(before):
        b, a = before[i], after[i]
        bb = _island_box(buf_before, _island_loops(me, b["faces"]))
        ab = _island_box(buf_after, _island_loops(me, a["faces"]))
        out.append({"island": i, "faces": len(b["faces"]),
                    "uvAreaBefore": round(b["uva"], 6), "uvAreaAfter": round(a["uva"], 6),
                    "densityBefore": _px_per_cm(b["density"]), "densityAfter": _px_per_cm(a["density"]),
                    "boxBefore": [round(v, 5) for v in bb], "boxAfter": [round(v, 5) for v in ab]})
    return out


def _uv_pair_or_number(raw, key, what):
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise MifOpError("%s takes a number or [u, v], not a boolean. NOTHING was %s." % (key, what))
    if isinstance(raw, (int, float)):
        v = finite_float(raw, key)
        return v, v
    if isinstance(raw, (list, tuple)) and len(raw) == 2 and not any(isinstance(x, bool) for x in raw):
        a, b = finite_floats(list(raw), key)
        return a, b
    raise MifOpError("%s takes a number or [u, v], got %r. NOTHING was %s." % (key, raw, what))


_TRANSFORM_KEYS = ("object", "name", "uvLayer", "layer", "islands", "faces", "translate", "rotate", "scale", "pivot",
                   "matchTexelDensity", "textureSize", "alignRotation", "randomize", "arrange")
_RANDOMIZE_KEYS = {"seed", "translate", "rotate", "scale"}
_ARRANGE_KEYS = {"columns", "margin"}


def _hull(pts):
    """Convex hull of 2D points (monotone chain), counter-clockwise."""
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _align_angle(buf, loops):
    """Degrees that turn an island so its smallest-area bounding box is axis-aligned (the hull edge that gives the
    least box area - uv.align_rotation AUTO's idea, 5.2 bl_operators/uvcalc_transform.py:332), the turn of smallest
    size among the four equivalent ones."""
    pts = [(round(buf[2 * li], 7), round(buf[2 * li + 1], 7)) for li in loops]
    hull = _hull(pts)
    if len(hull) < 3:
        return 0.0
    best, best_a = None, 0.0
    for i in range(len(hull)):
        (x0, y0), (x1, y1) = hull[i], hull[(i + 1) % len(hull)]
        a = math.atan2(y1 - y0, x1 - x0)
        c, s = math.cos(-a), math.sin(-a)
        xs = [x * c - y * s for x, y in hull]
        ys = [x * s + y * c for x, y in hull]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if best is None or area < best - 1e-12:
            best, best_a = area, -a
    deg = math.degrees(best_a) % 90.0
    return deg - 90.0 if deg > 45.0 else deg


def op_uv_transform_islands(params):
    """Move, turn and scale UV islands one by one, or bring them to a texel density - each about its own pivot.

    THE GAP: uv_unwrap uvTransform scales and offsets the whole layer (and offers no rotation); uv_set_texel_density
    scales islands but never moves or turns one. A handle's two halves resized to one density, an island scaled down
    over one tile of a nine-tile atlas and shifted onto it, a 30 cm detail texture scaled in a little - each was a hand
    edit in the UV editor.

    params:
      object (str), uvLayer (str, default the active one)
      islands [indices] (uv_info islands:true lists them) | faces [indices] (a face picks its island) - required
      translate [u, v]       added after the turn and the scale
      rotate (degrees)       counter-clockwise, about the pivot
      scale (number | [su, sv])  about the pivot
      pivot CENTER (default: each island's bounding-box centre) | MEDIAN (each island's mean UV corner) | [u, v] (one
                             point for every island)
      matchTexelDensity {target (px/cm), textureSize?} | {toIsland: i} - a uniform scale per island that brings it to
                             the target, or to island i's density (texture-independent). Excludes scale.
      textureSize (int | [w, h], default 1024)   what the reply's densities are measured on
      alignRotation (true)   turn each island so its smallest bounding box is axis-aligned (Blender's Align Rotation
                             AUTO); the turn is added to rotate
      randomize {seed, translate [du, dv] (largest offset), rotate (largest turn, degrees), scale (largest change, e.g.
                             0.1 = +-10%)} - a different value per island from the seed (Blender's Randomize
                             Transform), which breaks the repeat of a tiling material over a kit's identical pieces
      arrange {columns (default the square root, rounded up), margin (default 0.01)} - after everything else, the
                             islands laid out in a grid over the 0-1 tile in island order, one uniform scale for all of
                             them so their relative density holds (Blender 5.0's Arrange Islands, on every build)
    Applied in that order: the density scale, scale, rotate (with alignRotation and randomize), translate, arrange. Reply: per island its UV area, texel density
    (px/cm on textureSize) and box [umin, vmin, umax, vmax] before and after.
    """
    what = "changed"
    reject_unknown(params, _TRANSFORM_KEYS, "uv_transform_islands")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    require_editable(obj, "transform the UV islands of")
    me = obj.data
    name = _uv_layer_name(obj, params, what)
    w, h = _texture_size(params.get("textureSize", 1024))
    if params.get("translate") is not None and not isinstance(params.get("translate"), (list, tuple)):
        raise MifOpError("translate is [u, v], got %r. NOTHING was changed." % (params.get("translate"),))
    translate = _uv_pair_or_number(params.get("translate"), "translate", what)
    rotate = take_float(params, "rotate", default=None)
    scale = _uv_pair_or_number(params.get("scale"), "scale", what)
    if scale is not None and (scale[0] == 0.0 or scale[1] == 0.0):
        raise MifOpError("scale 0 collapses the island to a line or a point. NOTHING was changed.")
    pivot_raw = params.get("pivot", "CENTER")
    if isinstance(pivot_raw, str) and pivot_raw.upper() in ("CENTER", "MEDIAN"):
        pivot = pivot_raw.upper()
    elif isinstance(pivot_raw, (list, tuple)) and len(pivot_raw) == 2 and not any(isinstance(x, bool)
                                                                                  for x in pivot_raw):
        pivot = tuple(finite_floats(list(pivot_raw), "pivot"))
    else:
        raise MifOpError("pivot is CENTER, MEDIAN or [u, v], got %r. NOTHING was changed." % (pivot_raw,))
    match = params.get("matchTexelDensity")
    if match is not None:
        if not isinstance(match, dict) or not match or not set(match) <= {"target", "textureSize", "toIsland"} \
                or ("target" in match) == ("toIsland" in match):
            raise MifOpError("matchTexelDensity is {target (px/cm), textureSize?} or {toIsland: i}, got %r. NOTHING "
                             "was changed." % (match,))
        if "toIsland" in match and "textureSize" in match:
            raise MifOpError("matchTexelDensity toIsland is a ratio to another island and takes no textureSize. "
                             "NOTHING was changed.")
        if scale is not None:
            raise MifOpError("matchTexelDensity sets each island's scale; pass it OR scale, not both. NOTHING was "
                             "changed.")
    align = take_bool(params, "alignRotation", default=False)
    rnd_spec = params.get("randomize")
    if rnd_spec is not None:
        if not isinstance(rnd_spec, dict) or not rnd_spec or set(rnd_spec) - _RANDOMIZE_KEYS:
            raise MifOpError("randomize is {seed, translate [du, dv], rotate (degrees), scale}, got %r. NOTHING was "
                             "changed." % (rnd_spec,))
        r_t = _uv_pair_or_number(rnd_spec.get("translate"), "randomize.translate", what) or (0.0, 0.0)
        r_r = finite_float(rnd_spec.get("rotate", 0.0), "randomize.rotate")
        r_s = finite_float(rnd_spec.get("scale", 0.0), "randomize.scale")
        if not 0.0 <= r_s < 1.0:
            raise MifOpError("randomize.scale is the largest change, 0 to below 1 (0.1 = +-10%%), got %g. NOTHING was "
                             "changed." % r_s)
        seed = rnd_spec.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise MifOpError("randomize.seed is a whole number, got %r. NOTHING was changed." % (seed,))
    arrange = params.get("arrange")
    if arrange is not None:
        if arrange is True:
            arrange = {}
        if not isinstance(arrange, dict) or set(arrange) - _ARRANGE_KEYS:
            raise MifOpError("arrange is true or {columns, margin}, got %r. NOTHING was changed." % (arrange,))
        margin = finite_float(arrange.get("margin", 0.01), "arrange.margin")
        if not 0.0 <= margin < 0.25:
            raise MifOpError("arrange.margin is 0 to below 0.25 of the tile, got %g. NOTHING was changed." % margin)
    if (translate is None and rotate is None and scale is None and match is None and not align and rnd_spec is None
            and arrange is None):
        raise MifOpError("nothing to do: pass translate, rotate, scale, matchTexelDensity, alignRotation, randomize or "
                         "arrange. NOTHING was changed.")

    all_rows, n_islands = _measure(obj, name, None, w, h)
    chosen = _chosen_islands(params, obj, all_rows, n_islands, what)
    by_index = {r["island"]: r for r in all_rows}
    factors = {}
    match_out = None
    if match is not None:
        if "toIsland" in match:
            ref = match["toIsland"]
            if isinstance(ref, bool) or not isinstance(ref, int) or not 0 <= ref < n_islands:
                raise MifOpError("matchTexelDensity toIsland %r is not an island index of '%s' (it has %d). NOTHING "
                                 "was changed." % (ref, obj.name, n_islands))
            want = by_index[ref]["density"]
            if want is None:
                raise MifOpError("island %d has no area in UV or in the world, so it has no density to match. NOTHING "
                                 "was changed." % ref)
            match_out = {"toIsland": ref, "densityPxPerCm": _px_per_cm(want), "textureSize": [int(w), int(h)]}
        else:
            target = finite_float(match["target"], "matchTexelDensity.target")
            if target <= 0:
                raise MifOpError("matchTexelDensity target must be above 0 px/cm. NOTHING was changed.")
            mw, mh = _texture_size(match.get("textureSize", [w, h]), "matchTexelDensity.textureSize")
            # density on (mw, mh) is sqrt(uva * mw * mh / a3); restated on the reply's (w, h)
            want = target * 100.0 * ((w * h) / (mw * mh)) ** 0.5
            match_out = {"target": target, "textureSize": [int(mw), int(mh)]}
        flat = [i for i in chosen if by_index[i]["density"] is None]
        if flat:
            raise MifOpError("island(s) %s have no area in UV or in the world, so no density to match. NOTHING was "
                             "changed." % flat[:10])
        factors = {i: want / by_index[i]["density"] for i in chosen}

    # THE GRID IS JUDGED BEFORE THE MODE SWITCH: columns and the room the margin leaves depend only on the island count,
    # and both refusals sat after Object Mode was entered and the islands moved in memory.
    if arrange is not None:
        cols = arrange.get("columns") or max(1, math.ceil(math.sqrt(len(chosen))))
        if isinstance(cols, bool) or not isinstance(cols, int) or cols < 1:
            raise MifOpError("arrange.columns is a whole number 1 or more, got %r. NOTHING was changed." % (cols,))
        rows_n = max(1, math.ceil(len(chosen) / cols))
        cw, ch = (1.0 - margin * (cols + 1)) / cols, (1.0 - margin * (rows_n + 1)) / rows_n
        if cw <= 0 or ch <= 0:
            raise MifOpError("arrange: %d columns and %d rows with margin %g leave no room. NOTHING was changed."
                             % (cols, rows_n, margin))
    if obj.mode != "OBJECT" and bpy.context.object is not None:
        bpy.ops.object.mode_set(mode="OBJECT")     # an edit-mode bmesh would overwrite the write on the way out
    buf = _read_uvs(me, name)
    new = array.array("f", buf)
    extra = {}
    for i in chosen:
        loops = _island_loops(me, by_index[i]["faces"])
        turn, ks, dt = 0.0, 1.0, (0.0, 0.0)
        if align:
            turn += _align_angle(buf, loops)
        if rnd_spec is not None:
            import random                                            # noqa: PLC0415
            g = random.Random(seed * 1000003 + i)
            turn += g.uniform(-r_r, r_r)
            ks = 1.0 + g.uniform(-r_s, r_s)
            dt = (g.uniform(-r_t[0], r_t[0]), g.uniform(-r_t[1], r_t[1]))
        extra[i] = (turn, ks, dt)
    for i in chosen:
        loops = _island_loops(me, by_index[i]["faces"])
        turn, ks, dt = extra[i]
        total = (rotate or 0.0) + turn
        cos_r = sin_r = None
        if rotate is not None or turn:
            cos_r, sin_r = math.cos(math.radians(total)), math.sin(math.radians(total))
        if pivot == "CENTER":
            u0, v0, u1, v1 = _island_box(buf, loops)
            pu, pv = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        elif pivot == "MEDIAN":
            pu, pv = _island_median(buf, loops)
        else:
            pu, pv = pivot
        k = factors.get(i, 1.0) * ks
        su, sv = scale if scale is not None else (1.0, 1.0)
        for li in loops:
            x = (buf[2 * li] - pu) * k * su
            y = (buf[2 * li + 1] - pv) * k * sv
            if cos_r is not None:
                x, y = x * cos_r - y * sin_r, x * sin_r + y * cos_r
            x += pu
            y += pv
            if translate is not None:
                x += translate[0]
                y += translate[1]
            new[2 * li] = x + dt[0]
            new[2 * li + 1] = y + dt[1]
    arranged = None
    if arrange is not None:
        boxes = {i: _island_box(new, _island_loops(me, by_index[i]["faces"])) for i in chosen}
        mw_ = max(b[2] - b[0] for b in boxes.values()) or 1e-9
        mh_ = max(b[3] - b[1] for b in boxes.values()) or 1e-9
        s = min(cw / mw_, ch / mh_)
        for n_, i in enumerate(chosen):
            u0, v0, u1, v1 = boxes[i]
            c_, r_ = n_ % cols, n_ // cols
            cu = margin + c_ * (cw + margin) + cw / 2.0
            cv = 1.0 - (margin + r_ * (ch + margin) + ch / 2.0)
            pu, pv = (u0 + u1) / 2.0, (v0 + v1) / 2.0
            for li in _island_loops(me, by_index[i]["faces"]):
                new[2 * li] = (new[2 * li] - pu) * s + cu
                new[2 * li + 1] = (new[2 * li + 1] - pv) * s + cv
        arranged = {"columns": cols, "rows": rows_n, "margin": margin, "scale": round(s, 6)}
    _write_uvs(me, name, new)
    after_rows = {r["island"]: r for r in _measure(obj, name, set(chosen), w, h)[0]}
    out = {"object": obj.name, "layer": name, "textureSize": [int(w), int(h)],
           "pivot": pivot if isinstance(pivot, str) else list(pivot),
           "applied": {"translate": list(translate) if translate else None, "rotate": rotate,
                       "scale": list(scale) if scale else None, "matchTexelDensity": match_out},
           "islandCount": n_islands,
           "islands": _island_rows_out({i: by_index[i] for i in chosen}, after_rows, buf, new, me)}
    if factors:
        out["densityScale"] = {str(i): round(k, 6) for i, k in factors.items()}
    if align or rnd_spec is not None:
        out["perIsland"] = {str(i): {"turnDegrees": round(extra[i][0], 4), "scale": round(extra[i][1], 6),
                                     "offset": [round(extra[i][2][0], 6), round(extra[i][2][1], 6)]} for i in chosen}
    if arranged:
        out["arranged"] = arranged
    out.update(shared_data_note(obj))
    return out


_STITCH_KEYS = ("object", "name", "uvLayer", "layer", "mode", "islands", "faces", "threshold")


def _uv_splits(me, buf):
    """[edge keys] of the mesh edges whose two faces' UVs differ at either end - the layout's seams in UV."""
    edge_faces = {}
    for p in me.polygons:
        n = p.loop_total
        for k in range(n):
            li = p.loop_start + k
            lj = p.loop_start + (k + 1) % n
            a, b = me.loops[li].vertex_index, me.loops[lj].vertex_index
            edge_faces.setdefault((min(a, b), max(a, b)), []).append({a: li, b: lj})
    splits = []
    for key, sides in edge_faces.items():
        if len(sides) != 2:
            continue
        s0, s1 = sides
        if any(abs(buf[2 * s0[v]] - buf[2 * s1[v]]) > 1e-6 or abs(buf[2 * s0[v] + 1] - buf[2 * s1[v] + 1]) > 1e-6
               for v in key):
            splits.append(key)
    return splits


def op_uv_stitch(params):
    """Stitch two UV islands along the edges they share, weld UVs, or merge UVs by distance - and count the layout's
    split edges before and after.

    WHY (second 1.3 audit, 2026-10-25): fewer seams after an automatic unwrap, two halves unwrapped apart joined, a
    lightmap with fewer splits. uv.stitch, uv.weld and uv.remove_doubles are UV-editor operators; this is the data
    route, headless on every build.

    params:
      object, uvLayer (default the active one)
      mode      STITCH (default): islands [a, b] - island b is moved (turned, scaled and shifted, the best fit through
                the mesh vertices the two share) onto island a, and the shared UVs are made one; they must share
                mesh edges. WELD: the chosen faces' UVs at each mesh vertex set to their mean. MERGE_BY_DISTANCE: at each
                vertex, UVs closer than threshold (default 0.0001) become one.
      islands [indices] | faces [indices] - which (uv_info islands:true lists islands); WELD and MERGE default to all
      threshold - MERGE_BY_DISTANCE
    Reply: splitEdges and their world length before and after, islandCount before and after.
    """
    what = "changed"
    reject_unknown(params, _STITCH_KEYS, "uv_stitch")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    require_editable(obj, "stitch the UVs of")
    me = obj.data
    name = _uv_layer_name(obj, params, what)
    mode = str(take(params, "mode", default="STITCH", kind=str)).upper()
    if mode not in ("STITCH", "WELD", "MERGE_BY_DISTANCE"):
        raise MifOpError("mode is STITCH, WELD or MERGE_BY_DISTANCE, got %r. NOTHING was changed." % mode)
    thr = take_float(params, "threshold", default=1e-4)
    if params.get("threshold") is not None and mode != "MERGE_BY_DISTANCE":
        raise MifOpError("threshold is MERGE_BY_DISTANCE's. NOTHING was changed.")
    if not thr >= 0:
        raise MifOpError("threshold is 0 or more. NOTHING was changed.")
    if obj.mode != "OBJECT" and bpy.context.object is not None:
        bpy.ops.object.mode_set(mode="OBJECT")
    rows, n_isl = _measure(obj, name, None, 1024, 1024)
    by_index = {r["island"]: r for r in rows}
    chosen = _chosen_islands(params, obj, rows, n_isl, what, required=(mode == "STITCH"))
    buf = _read_uvs(me, name)
    mw = obj.matrix_world

    def world_len(keys):
        return sum((mw @ me.vertices[a].co - mw @ me.vertices[b].co).length for a, b in keys)
    split0 = _uv_splits(me, buf)
    new = array.array("f", buf)
    faces = [f for i in chosen for f in by_index[i]["faces"]]
    out = {"object": obj.name, "layer": name, "mode": mode, "islands": chosen}
    if mode == "STITCH":
        if len(chosen) != 2:
            raise MifOpError("STITCH takes two islands: [the one kept, the one moved onto it]. NOTHING was changed.")
        keep, move = (params.get("islands") or chosen)[:2]
        loops_of = {}
        for which in (keep, move):
            for f in by_index[which]["faces"]:
                p = me.polygons[f]
                for li in p.loop_indices:
                    loops_of.setdefault((which, me.loops[li].vertex_index), []).append(li)
        shared = sorted({v for (w, v) in loops_of if w == keep} & {v for (w, v) in loops_of if w == move})
        if len(shared) < 2:
            raise MifOpError("islands %d and %d share %d mesh vertex(es) - stitching needs an edge they share. NOTHING was "
                             "changed." % (keep, move, len(shared)))

        def uv(li):
            return complex(buf[2 * li], buf[2 * li + 1])
        src = [sum(uv(li) for li in loops_of[(move, v)]) / len(loops_of[(move, v)]) for v in shared]
        dst = [sum(uv(li) for li in loops_of[(keep, v)]) / len(loops_of[(keep, v)]) for v in shared]
        # BEST-FIT SIMILARITY (a turn, a uniform scale and a shift) taking the moved island's shared UVs onto the kept
        # island's: least squares in the complex plane, z -> a z + b.
        cs, cd = sum(src) / len(src), sum(dst) / len(dst)
        num = sum((d - cd) * (s - cs).conjugate() for s, d in zip(src, dst))
        den = sum(abs(s - cs) ** 2 for s in src) or 1e-12
        a = num / den
        b = cd - a * cs
        for f in by_index[move]["faces"]:
            for li in me.polygons[f].loop_indices:
                z = a * uv(li) + b
                new[2 * li], new[2 * li + 1] = z.real, z.imag
        for v, d in zip(shared, dst):
            for li in loops_of[(move, v)]:
                new[2 * li], new[2 * li + 1] = d.real, d.imag
        out.update(kept=keep, moved=move, sharedVertices=len(shared), scale=round(abs(a), 6),
                   turnDegrees=round(math.degrees(math.atan2(a.imag, a.real)), 4))
    else:
        per_v = {}
        for f in faces:
            for li in me.polygons[f].loop_indices:
                per_v.setdefault(me.loops[li].vertex_index, []).append(li)
        merged = 0
        for v, lis in per_v.items():
            if mode == "WELD":
                groups = [lis]
            else:
                groups = []
                for li in lis:
                    g = next((g for g in groups if abs(buf[2 * g[0]] - buf[2 * li]) <= thr
                              and abs(buf[2 * g[0] + 1] - buf[2 * li + 1]) <= thr), None)
                    if g is None:
                        groups.append([li])
                    else:
                        g.append(li)
            for g in groups:
                if len(g) < 2:
                    continue
                u = sum(buf[2 * li] for li in g) / len(g)
                w = sum(buf[2 * li + 1] for li in g) / len(g)
                for li in g:
                    if (new[2 * li], new[2 * li + 1]) != (u, w):
                        merged += 1
                    new[2 * li], new[2 * li + 1] = u, w
        out["uvsMoved"] = merged
    _write_uvs(me, name, new)
    split1 = _uv_splits(me, _read_uvs(me, name))
    _rows, n_after = _measure(obj, name, None, 1024, 1024)
    out.update(splitEdges={"before": len(split0), "after": len(split1)},
               splitLength={"before": round(world_len(split0), 6), "after": round(world_len(split1), 6)},
               islandCount={"before": n_isl, "after": n_after})
    out.update(shared_data_note(obj))
    return out


def _right_angle_error(buf, me, faces):
    """The largest departure from 90 degrees at any corner of the island's quads in UV, in degrees."""
    worst = 0.0
    for f in faces:
        lis = list(me.polygons[f].loop_indices)
        n = len(lis)
        for k in range(n):
            a, b, c = lis[k - 1], lis[k], lis[(k + 1) % n]
            ax, ay = buf[2 * a] - buf[2 * b], buf[2 * a + 1] - buf[2 * b + 1]
            cx, cy = buf[2 * c] - buf[2 * b], buf[2 * c + 1] - buf[2 * b + 1]
            la, lc = math.hypot(ax, ay), math.hypot(cx, cy)
            if la < 1e-12 or lc < 1e-12:
                return 90.0
            cosang = max(-1.0, min(1.0, (ax * cx + ay * cy) / (la * lc)))
            worst = max(worst, abs(math.degrees(math.acos(cosang)) - 90.0))
    return round(worst, 4)


def _edge_spread(buf, me, faces):
    """The longest UV edge of the island over its shortest - 1 when every quad is the same square."""
    lens = []
    for f in faces:
        lis = list(me.polygons[f].loop_indices)
        for k in range(len(lis)):
            a, b = lis[k], lis[(k + 1) % len(lis)]
            lens.append(math.hypot(buf[2 * a] - buf[2 * b], buf[2 * a + 1] - buf[2 * b + 1]))
    lo = min(lens)
    return round(max(lens) / lo, 4) if lo > 1e-12 else None


_STRAIGHTEN_KEYS = ("object", "name", "uvLayer", "layer", "islands", "faces", "mode", "activeFace", "textureSize")


def op_uv_straighten_islands(params):
    """Straighten quad UV islands into a grid: GRID (every quad a square) or RECTIFY (rectangles in world proportion).

    THE GAP: a curved or wavy quad island cannot carry a straight pattern line across it, and no op laid one out on a
    grid. Blender's Follow Active Quads does it from one face; this sets that face to a rectangle first (a square for
    GRID, its world edge lengths for RECTIFY), runs the operator on the island alone (EVEN for GRID, LENGTH_AVERAGE for
    RECTIFY), then scales the island back to its UV area and puts its box centre where it was.

    params:
      object (str), uvLayer (str, default the active one)
      islands [indices] | faces [indices] - required; every face of a chosen island must be a quad
      mode GRID (default) | RECTIFY
      activeFace (int)   the face the grid starts from - one island only; default the island's quad nearest square
      textureSize (int | [w, h], default 1024)   for the reply's densities
    Reply: per island the UV area, density and box before and after, rightAngleError (the worst corner's departure from
    90 degrees, in degrees) and edgeSpread (longest UV edge over shortest) before and after.
    """
    what = "changed"
    reject_unknown(params, _STRAIGHTEN_KEYS, "uv_straighten_islands")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    require_editable(obj, "straighten the UV islands of")
    me = obj.data
    name = _uv_layer_name(obj, params, what)
    w, h = _texture_size(params.get("textureSize", 1024))
    mode = str(take(params, "mode", default="GRID", kind=str)).upper()
    if mode not in ("GRID", "RECTIFY"):
        raise MifOpError("mode is GRID (every quad a square) or RECTIFY (rectangles in world proportion), got %r. "
                         "NOTHING was changed." % mode)
    active_face = take_int(params, "activeFace", default=None)
    all_rows, n_islands = _measure(obj, name, None, w, h)
    chosen = _chosen_islands(params, obj, all_rows, n_islands, what)
    by_index = {r["island"]: r for r in all_rows}
    for i in chosen:
        bad = [f for f in by_index[i]["faces"] if len(me.polygons[f].vertices) != 4]
        if bad:
            raise MifOpError("island %d has %d face(s) that are not quads (first: %s); a grid is laid quad by quad. "
                             "NOTHING was changed." % (i, len(bad), bad[:5]))
    if active_face is not None:
        if len(chosen) != 1:
            raise MifOpError("activeFace picks the start of ONE island's grid; this call names %d islands. NOTHING was "
                             "changed." % len(chosen))
        if active_face not in by_index[chosen[0]]["faces"]:
            raise MifOpError("activeFace %d is not a face of island %d. NOTHING was changed." % (active_face, chosen[0]))
    if obj.mode != "OBJECT" and bpy.context.object is not None:
        bpy.ops.object.mode_set(mode="OBJECT")

    buf = _read_uvs(me, name)
    starts = {}
    for i in chosen:
        faces = by_index[i]["faces"]
        if active_face is not None:
            starts[i] = active_face
        else:
            starts[i] = min(faces, key=lambda f: _right_angle_error(buf, me, [f]))
    m3 = obj.matrix_world.to_3x3()
    prev_layer = (me.uv_layers.active or me.uv_layers[0]).name
    snap = selection_snapshot()
    ts = bpy.context.scene.tool_settings
    sync_before = ts.use_uv_select_sync
    try:
        select_only([obj])
        bpy.context.view_layer.objects.active = obj
        me.uv_layers.active = me.uv_layers[name]
        bpy.ops.object.mode_set(mode="EDIT")
        ts.use_uv_select_sync = False
        for i in chosen:
            bm = bmesh.from_edit_mesh(me)
            bm.faces.ensure_lookup_table()
            uvl = bm.loops.layers.uv.active
            members = set(by_index[i]["faces"])
            for f in bm.faces:
                f.select_set(f.index in members)
            act = bm.faces[starts[i]]
            bm.faces.active = act
            ls = act.loops
            if mode == "GRID":
                a = b = 1.0
            else:
                a = 0.5 * ((m3 @ (ls[1].vert.co - ls[0].vert.co)).length + (m3 @ (ls[2].vert.co - ls[3].vert.co)).length)
                b = 0.5 * ((m3 @ (ls[2].vert.co - ls[1].vert.co)).length + (m3 @ (ls[3].vert.co - ls[0].vert.co)).length)
            q = [lp[uvl].uv.copy() for lp in ls]
            signed = sum(q[k].x * q[(k + 1) % 4].y - q[(k + 1) % 4].x * q[k].y for k in range(4))
            corners = ([(0, 0), (a, 0), (a, b), (0, b)] if signed >= 0 else [(0, 0), (0, a), (b, a), (b, 0)])
            for lp, c in zip(ls, corners):
                lp[uvl].uv = c
            bmesh.update_edit_mesh(me)
            bpy.ops.uv.select_all(action="SELECT")
            res = bpy.ops.uv.follow_active_quads(mode="EVEN" if mode == "GRID" else "LENGTH_AVERAGE")
            if "FINISHED" not in res:
                raise MifOpError("Follow Active Quads did not finish on island %d (%s)." % (i, sorted(res)))
        bpy.ops.object.mode_set(mode="OBJECT")
        # BACK TO ITS AREA AND ITS PLACE: the operator lays the grid out at the start face's size
        new = _read_uvs(me, name)
        for i in chosen:
            loops = _island_loops(me, by_index[i]["faces"])
            area_now = sum(_poly_uv_area(new, me, f) for f in by_index[i]["faces"])
            k = (by_index[i]["uva"] / area_now) ** 0.5 if area_now > 1e-15 else 1.0
            u0, v0, u1, v1 = _island_box(new, loops)
            cu, cv = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
            bu0, bv0, bu1, bv1 = _island_box(buf, loops)
            tu, tv = 0.5 * (bu0 + bu1), 0.5 * (bv0 + bv1)
            for li in loops:
                new[2 * li] = tu + (new[2 * li] - cu) * k
                new[2 * li + 1] = tv + (new[2 * li + 1] - cv) * k
        _write_uvs(me, name, new)
    except Exception:
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        _write_uvs(me, name, buf)
        raise
    finally:
        ts.use_uv_select_sync = sync_before
        if me.uv_layers.get(prev_layer) is not None:
            me.uv_layers.active = me.uv_layers[prev_layer]
        selection_restore(snap)
    after_rows = {r["island"]: r for r in _measure(obj, name, set(chosen), w, h)[0]}
    rows = _island_rows_out({i: by_index[i] for i in chosen}, after_rows, buf, new, me)
    for row in rows:
        faces = by_index[row["island"]]["faces"]
        row["startFace"] = starts[row["island"]]
        row["rightAngleErrorBefore"] = _right_angle_error(buf, me, faces)
        row["rightAngleErrorAfter"] = _right_angle_error(new, me, faces)
        row["edgeSpreadBefore"] = _edge_spread(buf, me, faces)
        row["edgeSpreadAfter"] = _edge_spread(new, me, faces)
    out = {"object": obj.name, "layer": name, "mode": mode, "textureSize": [int(w), int(h)],
           "islandCount": n_islands, "islands": rows}
    crooked = [r["island"] for r in rows if r["rightAngleErrorAfter"] > 1.0]
    if crooked:
        out["straightenNote"] = ("island(s) %s still have a corner more than 1 degree off square - Follow Active Quads "
                                 "walks quad loops, and a pole (a vertex with three or five quads) or a seam inside the "
                                 "island bends the grid" % crooked)
    out.update(shared_data_note(obj))
    return out


def _poly_uv_area(buf, me, f):
    lis = list(me.polygons[f].loop_indices)
    acc = 0.0
    for k in range(len(lis)):
        a, b = lis[k], lis[(k + 1) % len(lis)]
        acc += buf[2 * a] * buf[2 * b + 1] - buf[2 * b] * buf[2 * a + 1]
    return abs(acc) * 0.5


# ---------------------------------------------------------------------------------------------------------------------
# UV VERTICES WITH FALLOFF, AND RELAX (2026-10-06, spec "Move, scale and relax UV vertices with falloff, read a stretch
# measure, and unwrap with the other methods"): a scan texture fitted onto a head - all UVs scaled to cover the face,
# the nose pulled with proportional editing until the nostrils line up, Relax over the stretched and overlapping UVs,
# a strip's top edge flattened by scale Y 0 - is UV-editor work vertex by vertex. These work on the mesh's UV loops.

_FALLOFFS = {
    "SMOOTH": lambda t: t * t * (3.0 - 2.0 * t),
    "SPHERE": lambda t: math.sqrt(max(0.0, 2.0 * t - t * t)),
    "ROOT": lambda t: math.sqrt(t),
    "SHARP": lambda t: t * t,
    "LINEAR": lambda t: t,
    "CONSTANT": lambda t: 1.0,
}


def _loop_arrays(me):
    vi = array.array("i", [0]) * len(me.loops)
    me.loops.foreach_get("vertex_index", vi)
    face_of = array.array("i", [0]) * len(me.loops)
    for p in me.polygons:
        for li in p.loop_indices:
            face_of[li] = p.index
    return vi, face_of


def _uv_selection(params, obj, name, buf, what, default_all=False):
    """(loop indices, face indices fully selected, how) from select {faces | linkedFrom | vertices | uvBox}."""
    me = obj.data
    raw = params.get("select")
    if raw is None:
        if default_all:
            return set(range(len(me.loops))), set(range(len(me.polygons))), "all"
        raise MifOpError("select is required: {faces: [i]}, {linkedFrom: face} (its whole island), {vertices: [i]} or "
                         "{uvBox: [umin, vmin, umax, vmax]}. NOTHING was %s." % what)
    if not isinstance(raw, dict) or len(raw) != 1 or not set(raw) <= {"faces", "linkedFrom", "vertices", "uvBox"}:
        raise MifOpError("select takes exactly one of faces, linkedFrom, vertices or uvBox, got %r. NOTHING was %s."
                         % (raw, what))
    key, val = next(iter(raw.items()))
    n_f, n_v = len(me.polygons), len(me.vertices)
    vi, face_of = _loop_arrays(me)
    if key in ("faces", "vertices"):
        top = n_f if key == "faces" else n_v
        if not isinstance(val, (list, tuple)) or not val or any(
                isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < top for i in val):
            raise MifOpError("select.%s is a list of %s indices of '%s' (0-%d), got %r. NOTHING was %s."
                             % (key, key[:-1], obj.name, top - 1, val, what))
        want = set(val)
        if key == "faces":
            loops = {li for f in want for li in me.polygons[f].loop_indices}
        else:
            loops = {li for li in range(len(me.loops)) if vi[li] in want}
    elif key == "linkedFrom":
        if isinstance(val, bool) or not isinstance(val, int) or not 0 <= val < n_f:
            raise MifOpError("select.linkedFrom is one face index of '%s' (0-%d), got %r. NOTHING was %s."
                             % (obj.name, n_f - 1, val, what))
        rows = _measure(obj, name, None, 1, 1)[0]
        isl = next(r for r in rows if val in r["faces"])
        loops = {li for f in isl["faces"] for li in me.polygons[f].loop_indices}
    else:
        if not isinstance(val, (list, tuple)) or len(val) != 4 or any(isinstance(x, bool) for x in val):
            raise MifOpError("select.uvBox is [umin, vmin, umax, vmax], got %r. NOTHING was %s." % (val, what))
        u0, v0, u1, v1 = finite_floats(list(val), "select.uvBox")
        if u1 <= u0 or v1 <= v0:
            raise MifOpError("select.uvBox max must be above min, got %r. NOTHING was %s." % (val, what))
        loops = {li for li in range(len(me.loops))
                 if u0 <= buf[2 * li] <= u1 and v0 <= buf[2 * li + 1] <= v1}
    if not loops:
        raise MifOpError("select %r matched no UV of '%s' on layer %r. NOTHING was %s." % (raw, obj.name, name, what))
    faces = {p.index for p in me.polygons if all(li in loops for li in p.loop_indices)}
    return loops, faces, key


def _islands_touched(obj, name, loops, w, h):
    me = obj.data
    _vi, face_of = _loop_arrays(me)
    faces = {face_of[li] for li in loops}
    rows = _measure(obj, name, None, w, h)[0]
    return {r["island"] for r in rows if faces.intersection(r["faces"])}


def _stretch_of(obj, name, island_set, mode):
    from .ops_mesh import uv_stretch  # noqa: PLC0415 - deferred like the rest of the mesh reads
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        uv = bm.loops.layers.uv[name]
        islands = _uv_islands(bm, uv)[0]
        chosen = [islands[i] if i in island_set else [] for i in range(len(islands))]
        st = uv_stretch(chosen, uv, obj.matrix_world.to_3x3(), mode, 64)
        return {"mean": st["mean"], "worst": st["worst"]}
    finally:
        bm.free()


_MOVE_KEYS = ("object", "name", "uvLayer", "layer", "select", "translate", "scale", "rotate", "pivot", "falloff", "pin",
              "stretch")


def op_uv_move_vertices(params):
    """Move, scale and rotate UV vertices, with a proportional falloff onto their neighbours - the UV editor's G, S and R.

    THE GAP: every UV op here moved whole layers or whole islands; a nose pulled until the scan's nostrils line up, the
    ears grabbed with a large radius, a strip's top edge flattened by scale Y 0, were hand edits.

    params:
      object (str), uvLayer (str, default the active one)
      select {faces: [i]} | {linkedFrom: face} (its whole island) | {vertices: [i]} (every UV of those vertices) |
             {uvBox: [umin, vmin, umax, vmax]}                        required
      translate [u, v], scale (number | [su, sv]), rotate (degrees)   at least one; scale 0 on an axis flattens
      pivot CENTER (default, the selection's box centre) | MEDIAN | [u, v]
      falloff {type SMOOTH | SPHERE | ROOT | SHARP | LINEAR | CONSTANT, radius (UV units)} - proportional editing:
             an unselected UV within radius of the selection moves by the falloff of its distance to the nearest one
      pin [vertex indices]   their UVs never move
      stretch ANGLE | AREA (default AREA)   the measure reported before and after for the islands touched
    Reply: UVs selected, moved by falloff, pinned; the selection's box (pinned UVs in it too) before and after;
    stretch before and after.
    """
    what = "changed"
    reject_unknown(params, _MOVE_KEYS, "uv_move_vertices")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    require_editable(obj, "move the UVs of")
    me = obj.data
    name = _uv_layer_name(obj, params, what)
    if params.get("translate") is not None and not isinstance(params.get("translate"), (list, tuple)):
        raise MifOpError("translate is [u, v], got %r. NOTHING was changed." % (params.get("translate"),))
    translate = _uv_pair_or_number(params.get("translate"), "translate", what)
    scale = _uv_pair_or_number(params.get("scale"), "scale", what)
    rotate = take_float(params, "rotate", default=None)
    if translate is None and scale is None and rotate is None:
        raise MifOpError("nothing to do: pass translate, scale or rotate. NOTHING was changed.")
    pivot_raw = params.get("pivot", "CENTER")
    if isinstance(pivot_raw, str) and pivot_raw.upper() in ("CENTER", "MEDIAN"):
        pivot = pivot_raw.upper()
    elif isinstance(pivot_raw, (list, tuple)) and len(pivot_raw) == 2 and not any(isinstance(x, bool)
                                                                                  for x in pivot_raw):
        pivot = tuple(finite_floats(list(pivot_raw), "pivot"))
    else:
        raise MifOpError("pivot is CENTER, MEDIAN or [u, v], got %r. NOTHING was changed." % (pivot_raw,))
    fall = params.get("falloff")
    shape = radius = None
    if fall is not None:
        if not isinstance(fall, dict) or "radius" not in fall or not set(fall) <= {"type", "radius"}:
            raise MifOpError("falloff is {type, radius}, got %r. NOTHING was changed." % (fall,))
        ftype = str(fall.get("type", "SMOOTH")).upper()
        if ftype not in _FALLOFFS:
            raise MifOpError("falloff type is one of %s, got %r. NOTHING was changed."
                             % (", ".join(_FALLOFFS), fall.get("type")))
        radius = finite_float(fall["radius"], "falloff.radius")
        if radius <= 0:
            raise MifOpError("falloff radius must be above 0 (UV units). NOTHING was changed.")
        shape = ftype
    pin = params.get("pin")
    if pin is not None and (not isinstance(pin, (list, tuple)) or any(
            isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(me.vertices) for i in pin)):
        raise MifOpError("pin is a list of vertex indices of '%s' (0-%d), got %r. NOTHING was changed."
                         % (obj.name, len(me.vertices) - 1, pin))
    from .ops_mesh import uv_stretch_mode  # noqa: PLC0415
    mode = uv_stretch_mode(params.get("stretch", "AREA"))
    if obj.mode != "OBJECT" and bpy.context.object is not None:
        bpy.ops.object.mode_set(mode="OBJECT")
    buf = _read_uvs(me, name)
    loops, _faces, how = _uv_selection(params, obj, name, buf, what)
    vi, _face_of = _loop_arrays(me)
    pinned_v = set(pin or ())
    pinned = {li for li in range(len(me.loops)) if vi[li] in pinned_v}
    moving = loops - pinned
    if not moving:
        raise MifOpError("every selected UV is pinned, so nothing would move. NOTHING was changed.")
    sel_pts = [(buf[2 * li], buf[2 * li + 1]) for li in moving]
    us, vs = [p[0] for p in sel_pts], [p[1] for p in sel_pts]
    box_pts = [(buf[2 * li], buf[2 * li + 1]) for li in loops]      # the whole selection, pinned UVs too
    if pivot == "CENTER":
        pu, pv = 0.5 * (min(us) + max(us)), 0.5 * (min(vs) + max(vs))
    elif pivot == "MEDIAN":
        uniq = {(round(u, 6), round(v, 6)) for u, v in sel_pts}
        pu, pv = sum(p[0] for p in uniq) / len(uniq), sum(p[1] for p in uniq) / len(uniq)
    else:
        pu, pv = pivot
    weights = {li: 1.0 for li in moving}
    if shape is not None:
        # A GRID OF CELLS radius wide, so each UV looks only at the selected points in its own and the eight cells round
        # it - the nearest selected point within radius is always there.
        cells = {}
        for u, v in set(sel_pts):
            cells.setdefault((int(math.floor(u / radius)), int(math.floor(v / radius))), []).append((u, v))
        curve = _FALLOFFS[shape]
        for li in range(len(me.loops)):
            if li in weights or li in pinned:
                continue
            u, v = buf[2 * li], buf[2 * li + 1]
            cu, cv = int(math.floor(u / radius)), int(math.floor(v / radius))
            best = None
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    for su_, sv_ in cells.get((cu + du, cv + dv), ()):
                        d = math.hypot(u - su_, v - sv_)
                        if best is None or d < best:
                            best = d
            if best is not None and best < radius:
                weights[li] = curve(1.0 - best / radius)
    island_set = _islands_touched(obj, name, weights.keys(), 1, 1)
    stretch_before = _stretch_of(obj, name, island_set, mode)
    su, sv = scale if scale is not None else (1.0, 1.0)
    cos_r = math.cos(math.radians(rotate)) if rotate is not None else 1.0
    sin_r = math.sin(math.radians(rotate)) if rotate is not None else 0.0
    new = array.array("f", buf)
    for li, wgt in weights.items():
        x, y = (buf[2 * li] - pu) * su, (buf[2 * li + 1] - pv) * sv
        x, y = x * cos_r - y * sin_r + pu, x * sin_r + y * cos_r + pv
        if translate is not None:
            x += translate[0]
            y += translate[1]
        new[2 * li] = buf[2 * li] + (x - buf[2 * li]) * wgt
        new[2 * li + 1] = buf[2 * li + 1] + (y - buf[2 * li + 1]) * wgt
    _write_uvs(me, name, new)
    after_pts = [(new[2 * li], new[2 * li + 1]) for li in loops]
    out = {"object": obj.name, "layer": name, "selectedBy": how, "uvsSelected": len(loops),
           "uvsMoved": len(weights), "uvsMovedByFalloff": len(weights) - len(moving),
           "uvsPinned": len(pinned),
           "pivot": [round(pu, 6), round(pv, 6)],
           "falloff": {"type": shape, "radius": radius} if shape else None,
           "boxBefore": [round(min(p[0] for p in box_pts), 6), round(min(p[1] for p in box_pts), 6),
                         round(max(p[0] for p in box_pts), 6), round(max(p[1] for p in box_pts), 6)],
           "boxAfter": [round(min(p[0] for p in after_pts), 6), round(min(p[1] for p in after_pts), 6),
                        round(max(p[0] for p in after_pts), 6), round(max(p[1] for p in after_pts), 6)],
           "islandsTouched": sorted(island_set),
           "stretch": {"mode": mode, "before": stretch_before, "after": _stretch_of(obj, name, island_set, mode)}}
    out.update(shared_data_note(obj))
    return out


_RELAX_KEYS = ("object", "name", "uvLayer", "layer", "select", "iterations", "pinBoundary", "pin", "stretch")


def op_uv_relax(params):
    """Relax UVs: Blender's Minimize Stretch over the chosen faces, with the island borders or named vertices held.

    THE GAP: a layout pulled by hand (uv_move_vertices) or a scan fitted region by region is left stretched and folded,
    and the UV editor's Relax / Minimize Stretch was out of reach. bpy.ops.uv.minimize_stretch runs headless in edit
    mode with the UVs selected (measured on 4.2.23, 4.4.0, 5.0.1 and 5.2.2); it moves only the selected faces' UVs and
    leaves pinned UVs where they are, so pinBoundary pins each island's border for the run and unpins it after.

    params:
      object (str), uvLayer (str, default the active one)
      select  as uv_move_vertices (a face counts when all its UVs are chosen); default every face
      iterations (int, default 20, 1-10000)
      pinBoundary (bool, default false)   hold the island borders
      pin [vertex indices]                hold these vertices' UVs
      stretch ANGLE | AREA (default AREA) the measure reported before and after
    Reply: faces relaxed, UVs pinned for the run, the largest UV move, stretch before and after.
    """
    what = "changed"
    reject_unknown(params, _RELAX_KEYS, "uv_relax")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    require_editable(obj, "relax the UVs of")
    me = obj.data
    name = _uv_layer_name(obj, params, what)
    iterations = take_int(params, "iterations", default=20)
    if not 1 <= iterations <= 10000:
        raise MifOpError("iterations is 1-10000, got %d. NOTHING was changed." % iterations)
    pin_boundary = take_bool(params, "pinBoundary", default=False)
    pin = params.get("pin")
    if pin is not None and (not isinstance(pin, (list, tuple)) or any(
            isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(me.vertices) for i in pin)):
        raise MifOpError("pin is a list of vertex indices of '%s' (0-%d), got %r. NOTHING was changed."
                         % (obj.name, len(me.vertices) - 1, pin))
    from .ops_mesh import uv_stretch_mode  # noqa: PLC0415
    mode = uv_stretch_mode(params.get("stretch", "AREA"))
    if obj.mode != "OBJECT" and bpy.context.object is not None:
        bpy.ops.object.mode_set(mode="OBJECT")
    buf = _read_uvs(me, name)
    loops, faces, how = _uv_selection(params, obj, name, buf, what, default_all=True)
    if not faces:
        raise MifOpError("select chose UVs but no whole face, and a relax moves faces' UVs. NOTHING was changed.")
    island_set = _islands_touched(obj, name, {li for f in faces for li in me.polygons[f].loop_indices}, 1, 1)
    stretch_before = _stretch_of(obj, name, island_set, mode)
    pinned_v = set(pin or ())
    prev_layer = (me.uv_layers.active or me.uv_layers[0]).name
    snap = selection_snapshot()
    ts = bpy.context.scene.tool_settings
    sync_before = ts.use_uv_select_sync
    pinned_count = 0
    try:
        select_only([obj])
        bpy.context.view_layer.objects.active = obj
        me.uv_layers.active = me.uv_layers[name]
        bpy.ops.object.mode_set(mode="EDIT")
        ts.use_uv_select_sync = False
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()
        uvl = bm.loops.layers.uv.active
        was_pinned = {}
        boundary = set()
        if pin_boundary:
            # AN ISLAND BORDER: a loop whose UV edge is not shared, at the same two UV points, by the face across it
            for f in bm.faces:
                n = len(f.loops)
                for k, lp in enumerate(f.loops):
                    e = lp.edge
                    a = (round(lp[uvl].uv.x, 5), round(lp[uvl].uv.y, 5))
                    b = (round(lp.link_loop_next[uvl].uv.x, 5), round(lp.link_loop_next[uvl].uv.y, 5))
                    shared = False
                    for other in e.link_loops:
                        if other.face is f:
                            continue
                        oa = (round(other[uvl].uv.x, 5), round(other[uvl].uv.y, 5))
                        ob = (round(other.link_loop_next[uvl].uv.x, 5), round(other.link_loop_next[uvl].uv.y, 5))
                        if {oa, ob} == {a, b}:
                            shared = True
                    if not shared:
                        boundary.add((f.index, k))
                        boundary.add((f.index, (k + 1) % n))
        # BMLoop.index is not kept up to date by from_edit_mesh, so a loop is keyed by its face and its place in it
        for f in bm.faces:
            f.select_set(f.index in faces)
            for k, lp in enumerate(f.loops):
                hold = (f.index, k) in boundary or lp.vert.index in pinned_v
                was_pinned[(f.index, k)] = lp[uvl].pin_uv
                if hold and not lp[uvl].pin_uv:
                    lp[uvl].pin_uv = True
                if hold and f.index in faces:
                    pinned_count += 1
        bmesh.update_edit_mesh(me)
        bpy.ops.uv.select_all(action="SELECT")
        res = bpy.ops.uv.minimize_stretch(iterations=iterations, blend=0.0, fill_holes=True)
        if "FINISHED" not in res:
            raise MifOpError("Minimize Stretch did not finish (%s). NOTHING was changed." % sorted(res))
        bm = bmesh.from_edit_mesh(me)
        uvl = bm.loops.layers.uv.active
        for f in bm.faces:
            for k, lp in enumerate(f.loops):
                lp[uvl].pin_uv = was_pinned.get((f.index, k), lp[uvl].pin_uv)
        bmesh.update_edit_mesh(me)
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        _write_uvs(me, name, buf)
        raise
    finally:
        ts.use_uv_select_sync = sync_before
        if me.uv_layers.get(prev_layer) is not None:
            me.uv_layers.active = me.uv_layers[prev_layer]
        selection_restore(snap)
    new = _read_uvs(me, name)
    moved = max(math.hypot(new[2 * i] - buf[2 * i], new[2 * i + 1] - buf[2 * i + 1]) for i in range(len(me.loops)))
    outside = sum(1 for p in me.polygons if p.index not in faces for li in p.loop_indices
                  if new[2 * li] != buf[2 * li] or new[2 * li + 1] != buf[2 * li + 1])
    out = {"object": obj.name, "layer": name, "selectedBy": how, "facesRelaxed": len(faces), "iterations": iterations,
           "uvsPinnedForTheRun": pinned_count, "pinBoundary": pin_boundary, "largestMove": round(moved, 6),
           "uvsMovedOutsideSelection": outside, "islandsTouched": sorted(island_set),
           "stretch": {"mode": mode, "before": stretch_before, "after": _stretch_of(obj, name, island_set, mode)}}
    out.update(shared_data_note(obj))
    return out


OPS = {
    "uv_stitch": op_uv_stitch,
    "uv_set_texel_density": op_uv_set_texel_density,
    "uv_transform_islands": op_uv_transform_islands,
    "uv_straighten_islands": op_uv_straighten_islands,
    "uv_move_vertices": op_uv_move_vertices,
    "uv_relax": op_uv_relax,
}
