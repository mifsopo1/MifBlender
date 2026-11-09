"""MifBlender ops: a high-to-low detail bake in one call, with seams and stitching drawn into the maps (1.4, 2026-11-07).

THE GAP. A character built by rule held up on silhouette and fell short up close: cloth read as clean procedural waves
with no seams, boots as smooth blobs. The stage it skipped is the detailed high poly baked down to the game mesh. Every
piece existed - a Catmull-Clark subdivide, displace_by_rule's seams and stitches, sculpt_stroke, bake_texture_set's
high-to-low with a cage - and composing them meant a dozen calls, a cage pushed out by hand, and nothing that told a
flat or holed normal map from a good one.

recipe_detail_bake composes them: the detail mesh (given, or made from the game mesh), a cage pushed out by the measured
gap, NORMAL / AO / CURVATURE through bake_texture_set, the stitch layer, and the checks - ray misses counted against the
real high, black or untouched texels inside UV islands, the normal map's mean tilt and how much of it carries detail.

THE STITCH LAYER draws seams and stitches straight into the baked NORMAL and AO maps along the low's boundary loops, UV
seams, sharp edges or a list of edges, at a stitch length, spacing and offset in millimetres - detail displace_by_rule's
STITCH can only make on a mesh dense enough to hold a 1 mm thread. Each edge is drawn once per face it borders, in that
face's UV island and on that face's side, so a seam that splits two islands is drawn on both halves, and the dashes run
on one arc length per chain so they meet across the cut.
"""
from __future__ import annotations

import math
import os
import time

import bpy

from .ops_common import (MifOpError, caller_deadline, check_output_path, check_writable_dir, finite_float, get_object,
                         reject_unknown, take, take_bool, take_float, take_int)

_KEYS = ("object", "detail", "build", "types", "folder", "pattern", "width", "height", "samples", "margin", "uvLayer",
         "greenChannel", "cage", "cageDistance", "maxRayDistance", "occluders", "stitches", "checks", "outOfProcess",
         "jobId", "keepCage", "memoryLimitGB")
_BUILD_KEYS = {"levels", "rules", "folds", "sculpt", "name", "maxFaces"}
_FOLD_KEYS = {"axis", "period", "depth", "irregularity", "seed", "vertexGroup", "sharpness"}
_STITCH_KEYS = {"edges", "edgeIndices", "seamWidthMm", "seamDepthMm", "stitchLengthMm", "stitchSpacingMm",
                "stitchOffsetMm", "rows", "rowGapMm", "threadWidthMm", "threadHeightMm", "aoStrength"}
_CHECK_KEYS = {"maxMissPercent", "maxBlackPercent", "maxMeanTiltDeg", "minDetailPercent"}
_TYPES = ("NORMAL", "AO", "CURVATURE")
_EDGE_KINDS = ("BOUNDARY", "UV_SEAMS", "SHARP")
_JOBS = {}
TAIL = "NOTHING was baked."


# --------------------------------------------------------------------------------------------------------- parsing

def _f(d, key, default, lo=None, hi=None, where="", strict_lo=False):
    if d.get(key) is None:
        return default
    v = finite_float(d[key], where + key)
    if lo is not None and (v <= lo if strict_lo else v < lo):
        raise MifOpError("%s%s must be %s %g, got %g. %s" % (where, key, "above" if strict_lo else "at least", lo, v,
                                                             TAIL))
    if hi is not None and v > hi:
        raise MifOpError("%s%s must be at most %g, got %g. %s" % (where, key, hi, v, TAIL))
    return v


def _parse_folds(raw):
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > 16:
        raise MifOpError("build.folds is a list of up to 16 {axis, period, depth, irregularity, seed, vertexGroup, "
                         "sharpness}. %s" % TAIL)
    out = []
    for k, f in enumerate(raw):
        where = "build.folds[%d]." % k
        if not isinstance(f, dict) or set(f) - _FOLD_KEYS:
            raise MifOpError("%s takes %s, got %r. %s" % (where[:-1], ", ".join(sorted(_FOLD_KEYS)), f, TAIL))
        axis = f.get("axis", [0.0, 0.0, 1.0])
        if not isinstance(axis, (list, tuple)) or len(axis) != 3:
            raise MifOpError("%saxis is a world direction [x, y, z] the folds repeat along. %s" % (where, TAIL))
        ax = [finite_float(a, where + "axis") for a in axis]
        n = math.sqrt(sum(a * a for a in ax))
        if n < 1e-9:
            raise MifOpError("%saxis has no length. %s" % (where, TAIL))
        out.append({"axis": [a / n for a in ax],
                    "period": _f(f, "period", None, 0.0, where=where, strict_lo=True),
                    "depth": _f(f, "depth", None, where=where),
                    "irregularity": _f(f, "irregularity", 0.5, 0.0, 4.0, where=where),
                    "sharpness": _f(f, "sharpness", 2.0, 0.5, 8.0, where=where),
                    "seed": int(_f(f, "seed", 0, where=where)),
                    "vertexGroup": f.get("vertexGroup")})
        if out[-1]["period"] is None or out[-1]["depth"] is None:
            raise MifOpError("%speriod and depth (scene units) are required. %s" % (where, TAIL))
    return out


def _parse_stitches(raw):
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list) or not raw or len(raw) > 16:
        raise MifOpError("stitches is a list of 1-16 {edges | edgeIndices, seamWidthMm, seamDepthMm, stitchLengthMm, "
                         "stitchSpacingMm, stitchOffsetMm, rows, rowGapMm, threadWidthMm, threadHeightMm, aoStrength}. %s"
                         % TAIL)
    out = []
    for k, s in enumerate(raw):
        where = "stitches[%d]." % k
        if not isinstance(s, dict) or set(s) - _STITCH_KEYS:
            raise MifOpError("%s takes %s, got %r. %s" % (where[:-1], ", ".join(sorted(_STITCH_KEYS)), s, TAIL))
        if (s.get("edges") is None) == (s.get("edgeIndices") is None):
            raise MifOpError("%s takes edges (%s) OR edgeIndices [the low's edge indices]. %s"
                             % (where[:-1], ", ".join(_EDGE_KINDS), TAIL))
        row = {"edges": None, "edgeIndices": None}
        if s.get("edges") is not None:
            kind = str(s["edges"]).upper()
            if kind not in _EDGE_KINDS:
                raise MifOpError("%sedges must be one of %s, got %r. %s" % (where, ", ".join(_EDGE_KINDS), s["edges"],
                                                                         TAIL))
            row["edges"] = kind
        else:
            idx = s["edgeIndices"]
            if not isinstance(idx, list) or not idx or not all(isinstance(i, int) and not isinstance(i, bool)
                                                              for i in idx):
                raise MifOpError("%sedgeIndices is a non-empty list of the low's edge indices. %s" % (where, TAIL))
            row["edgeIndices"] = sorted(set(idx))
        row["seamWidthMm"] = _f(s, "seamWidthMm", 3.0, 0.0, 100.0, where)
        row["seamDepthMm"] = _f(s, "seamDepthMm", 0.6, 0.0, 20.0, where)
        row["stitchLengthMm"] = _f(s, "stitchLengthMm", 3.0, 0.0, 50.0, where)
        row["stitchSpacingMm"] = _f(s, "stitchSpacingMm", 1.5, 0.0, 50.0, where)
        row["stitchOffsetMm"] = _f(s, "stitchOffsetMm", 4.0, 0.0, 100.0, where)
        row["rows"] = int(_f(s, "rows", 1, 0, 4, where))
        row["rowGapMm"] = _f(s, "rowGapMm", 4.0, 0.0, 100.0, where)
        row["threadWidthMm"] = _f(s, "threadWidthMm", 1.0, 0.0, 20.0, where, strict_lo=True)
        row["threadHeightMm"] = _f(s, "threadHeightMm", 0.35, 0.0, 10.0, where)
        row["aoStrength"] = _f(s, "aoStrength", 0.6, 0.0, 1.0, where)
        if row["rows"] and not row["stitchLengthMm"] > 0:
            raise MifOpError("%sstitchLengthMm must be above 0 when rows is above 0 (rows 0 draws the seam alone). %s"
                             % (where, TAIL))
        if not row["rows"] and not (row["seamWidthMm"] > 0 and row["seamDepthMm"] > 0):
            raise MifOpError("%sdraws nothing: rows 0 and no seam (seamWidthMm and seamDepthMm above 0). %s"
                             % (where, TAIL))
        out.append(row)
    return out


# --------------------------------------------------------------------------------------------- the low's UV raster

def _np():
    import numpy as np                                           # noqa: PLC0415
    return np


def _island_of_faces(me, uv):
    """A UV island id per polygon: faces sharing an edge whose two corners match in UV on both sides are one island."""
    parent = list(range(len(me.polygons)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    seen = {}
    for p in me.polygons:
        ls = list(p.loop_indices)
        for i, li in enumerate(ls):
            lj = ls[(i + 1) % len(ls)]
            va, vb = me.loops[li].vertex_index, me.loops[lj].vertex_index
            ua, ub = tuple(round(c, 6) for c in uv[li].uv), tuple(round(c, 6) for c in uv[lj].uv)
            key = (va, vb) if va < vb else (vb, va)
            uvk = (ua, ub) if va < vb else (ub, ua)
            other = seen.get(key)
            if other is None:
                seen[key] = (p.index, uvk)
            elif other[1] == uvk:
                ra, rb = find(other[0]), find(p.index)
                if ra != rb:
                    parent[ra] = rb
    return [find(i) for i in range(len(parent))]


def _uv_raster(obj, uv_name, w, h):
    """(island id per texel, -1 outside every triangle; the polygon's island ids; the UV layer). Texel centres are tested
    against each loop triangle."""
    np = _np()
    me = obj.data
    uv = me.uv_layers[uv_name].data if uv_name else (me.uv_layers.active or me.uv_layers[0]).data
    isl = _island_of_faces(me, uv)
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    tl = np.empty(nt * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("loops", tl)
    tp = np.empty(nt, dtype=np.int32)
    me.loop_triangles.foreach_get("polygon_index", tp)
    uvs = np.empty(len(me.loops) * 2, dtype=np.float32)
    uv.foreach_get("uv", uvs)
    uvs = uvs.reshape(-1, 2)
    tri = uvs[tl.reshape(-1, 3)] * np.array([w, h], dtype=np.float32)
    ids = np.full((h, w), -1, dtype=np.int32)
    for t in range(nt):
        a, b, c = tri[t]
        x0, x1 = int(max(0, math.floor(min(a[0], b[0], c[0])))), int(min(w - 1, math.ceil(max(a[0], b[0], c[0]))))
        y0, y1 = int(max(0, math.floor(min(a[1], b[1], c[1])))), int(min(h - 1, math.ceil(max(a[1], b[1], c[1]))))
        if x1 < x0 or y1 < y0:
            continue
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-12:
            continue
        xs = np.arange(x0, x1 + 1, dtype=np.float32) + 0.5
        ys = np.arange(y0, y1 + 1, dtype=np.float32)[:, None] + 0.5
        l1 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / den
        l2 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / den
        inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (1.0 - l1 - l2 >= -1e-6)
        if inside.any():
            sub = ids[y0:y1 + 1, x0:x1 + 1]
            sub[inside] = isl[tp[t]]
    return ids, isl, uv


def _interior(ids):
    """Texels whose four neighbours are in the same island: the bake's margin and edge filtering never reach them."""
    np = _np()
    m = ids >= 0
    inner = m.copy()
    inner[1:, :] &= ids[1:, :] == ids[:-1, :]
    inner[:-1, :] &= ids[:-1, :] == ids[1:, :]
    inner[:, 1:] &= ids[:, 1:] == ids[:, :-1]
    inner[:, :-1] &= ids[:, :-1] == ids[:, 1:]
    inner[0, :] = inner[-1, :] = False
    inner[:, 0] = inner[:, -1] = False
    return inner & np.ones_like(m)


def _load(path):
    """(h, w, 4) float32 of a baked file as stored - loaded as data, so no transfer is applied - and its size."""
    np = _np()
    im = bpy.data.images.load(path, check_existing=False)
    try:
        try:
            im.colorspace_settings.name = "Non-Color"
        except TypeError:
            pass
        w, h = int(im.size[0]), int(im.size[1])
        a = np.empty(w * h * 4, dtype=np.float32)
        im.pixels.foreach_get(a)
    finally:
        bpy.data.images.remove(im)
    return a.reshape(h, w, 4)


def _save(rgba, path):
    """Write (h, w, 4) as a 16-bit Non-Color PNG through save_image, over the baked file."""
    from .ops_gamebake import _write_image                       # noqa: PLC0415
    return _write_image(_np(), "MifDetail_%s" % os.path.basename(path), rgba, path, "PNG")


# ------------------------------------------------------------------------------------------- building the detail

def _long_edge(obj):
    """The 90th-percentile edge length (world): a mesh with long edges one way is subdivided for those, not its median."""
    me = obj.data
    if not me.edges:
        return 0.0
    mw = obj.matrix_world
    ls = sorted(((mw @ me.vertices[e.vertices[0]].co) - (mw @ me.vertices[e.vertices[1]].co)).length for e in me.edges)
    return ls[min(len(ls) - 1, int(len(ls) * 0.9))]


def _subdivided_copy(low, levels, name):
    """The low's own object (transform, vertex groups, deform modifiers) with its mesh replaced by a Catmull-Clark
    subdivision `levels` deep - Blender's Subdivision Surface, evaluated and kept as new mesh data."""
    tmp = low.copy()
    tmp.data = low.data.copy()
    for c in low.users_collection:
        c.objects.link(tmp)
    try:
        tmp.modifiers.clear()
        if tmp.data.shape_keys is not None:
            tmp.shape_key_clear()
        sub = tmp.modifiers.new("MifDetailSubdiv", "SUBSURF")
        sub.levels = sub.render_levels = levels
        sub.subdivision_type = "CATMULL_CLARK"
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        ev = tmp.evaluated_get(dg)
        dense = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
    finally:
        old = tmp.data
        bpy.data.objects.remove(tmp)
        bpy.data.meshes.remove(old)
    det = low.copy()
    det.data = dense
    dense.name = name
    det.name = name
    for c in low.users_collection:
        c.objects.link(det)
    for m in list(det.modifiers):
        if m.type != "ARMATURE":
            det.modifiers.remove(m)
    return det


def _apply_folds(det, folds):
    """Folds by rule: each vertex moved along its normal by depth x a ridge profile of its position along `axis` - a
    sine sharpened by `sharpness`, its phase bent by `irregularity` x Blender's noise, its height varied by the same
    noise - optionally weighted by a vertex group. Procedural, repeatable: the same seed gives the same folds."""
    import mathutils                                             # noqa: PLC0415
    from mathutils import noise                                  # noqa: PLC0415
    me = det.data
    mw = det.matrix_world
    inv3 = mw.to_3x3().inverted_safe()
    nrm = [mw.to_3x3() @ v.normal for v in me.vertices]
    rows = []
    for f in folds:
        gi = None
        if f["vertexGroup"]:
            vg = det.vertex_groups.get(str(f["vertexGroup"]))
            if vg is None:
                raise MifOpError("folds vertexGroup %r is not a vertex group of '%s' (it has %s). %s"
                                 % (f["vertexGroup"], det.name, ", ".join(g.name for g in det.vertex_groups)[:400],
                                    TAIL))
            gi = vg.index
        ax = mathutils.Vector(f["axis"])
        off = mathutils.Vector((f["seed"] * 17.13, f["seed"] * 5.71, f["seed"] * 11.37))
        k = 1.0 / f["period"]
        moved = 0
        deepest = 0.0
        for v, n in zip(me.vertices, nrm):
            wgt = 1.0
            if gi is not None:
                wgt = next((g.weight for g in v.groups if g.group == gi), 0.0)
                if wgt <= 0.0:
                    continue
            p = mw @ v.co
            q = p * (k * 0.35) + off
            ph = 2.0 * math.pi * (p.dot(ax) * k + f["irregularity"] * noise.noise(q))
            ridge = (0.5 + 0.5 * math.sin(ph)) ** f["sharpness"]
            amp = 0.55 + 0.45 * noise.noise(q * 0.5 + off)
            d = f["depth"] * wgt * ridge * amp
            if d:
                v.co += inv3 @ (n * d)
                moved += 1
                deepest = max(deepest, abs(d))
        rows.append({"axis": [round(a, 4) for a in f["axis"]], "period": f["period"], "depth": f["depth"],
                     "verticesMoved": moved, "deepest": round(deepest, 6)})
    me.update()
    return rows


# ------------------------------------------------------------------------------------------------ cage and rays

def _bvh(obj):
    from mathutils.bvhtree import BVHTree                        # noqa: PLC0415
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        mw = ev.matrix_world
        verts = [mw @ v.co for v in me.vertices]
        polys = [tuple(p.vertices) for p in me.polygons]
        return BVHTree.FromPolygons(verts, polys), verts, me.polygons and [(mw.to_3x3() @ p.normal).normalized()
                                                                          for p in me.polygons]
    finally:
        ev.to_mesh_clear()


def _low_samples(low):
    """World points and smooth normals on the low: every vertex, face centre and edge midpoint (evaluated)."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = low.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        mw = ev.matrix_world
        m3 = mw.to_3x3()
        # INSIDE EACH FACE, as texel centres are: a point on an open rim casts a ray that grazes the high's own rim and
        # misses, where no texel's ray does (measured on an open cylinder: a third of the rays at its rim vertices)
        pts = []
        vn = [m3 @ v.normal for v in me.vertices]
        for p in me.polygons:
            c = p.center
            cn = sum((vn[i] for i in p.vertices), p.normal * 0.0)
            if cn.length < 1e-9:
                cn = m3 @ p.normal
            pts.append((mw @ c, cn.normalized()))
            for i in p.vertices:
                q = c.lerp(me.vertices[i].co, 0.8)
                n = cn.normalized().lerp(vn[i].normalized(), 0.8)
                if n.length > 1e-9:
                    pts.append((mw @ q, n.normalized()))
        return pts
    finally:
        ev.to_mesh_clear()


def _measure_gap(low, high):
    """How far the high sits out from and in under the low along the low's normals (world units): at each of the low's
    sample points, the NEAREST point of the high, its offset along the normal. The cage covers the outward gap; the
    ray reach the inward. Nearest, not a ray: a ray from a point lying on the high slips past that surface and finds
    the far side of a leg - measured on a garment, a 0.42 m "gap" and a cage half a metre out."""
    tree = _bvh(high)[0]
    out_max = in_max = 0.0
    for p, n in _low_samples(low):
        q = tree.find_nearest(p)[0]
        if q is None:
            continue
        d = (q - p).dot(n)
        if d > out_max:
            out_max = d
        elif -d > in_max:
            in_max = -d
    return out_max, in_max


def _make_cage(low, distance, name):
    """The low's own topology pushed out along its vertex normals by distance (world) - what bake_texture's cage
    takes. Same object data layout and deform modifiers, so it follows the low through an armature."""
    cage = low.copy()
    cage.data = low.data.copy()
    cage.data.name = name
    cage.name = name
    for c in low.users_collection:
        c.objects.link(cage)
    for m in list(cage.modifiers):
        if m.type != "ARMATURE":
            cage.modifiers.remove(m)
    if cage.data.shape_keys is not None:
        cage.shape_key_clear()
    mw = cage.matrix_world
    s = sum(mw.to_3x3().to_scale()) / 3.0 or 1.0
    for v in cage.data.vertices:
        v.co += v.normal * (distance / s)
    cage.data.update()
    cage.hide_render = True
    return cage


def _ray_misses(low, high, cage_d, reach):
    """Rays cast as the bake casts them - from the cage point in along the low's normal, cage_d + reach long - at every
    low vertex, face centre and edge midpoint: the share that strike no high."""
    tree = _bvh(high)[0]
    pts = _low_samples(low)
    miss = 0
    where = []
    for p, n in pts:
        o = p + n * cage_d
        hit = tree.ray_cast(o, -n, cage_d + reach)
        if hit[0] is None:
            miss += 1
            if len(where) < 12:
                where.append([round(c, 4) for c in p])
    return miss, len(pts), where


# ---------------------------------------------------------------------------------------------- the stitch layer

def _edge_chains(me, chosen):
    """Arc length (world mm) at each end of each chosen edge, along chains split where three or more meet."""
    adj = {}
    for e in chosen:
        for v in me.edges[e].vertices:
            adj.setdefault(v, []).append(e)
    used = set()
    s_at = {}
    chains = 0

    def walk(v, e, s):
        while e is not None and e not in used:
            used.add(e)
            a, b = me.edges[e].vertices
            o = b if a == v else a
            ln = _ED_LEN[e]
            s_at[e] = {v: s, o: s + ln}
            s += ln
            v = o
            nxt = [x for x in adj[v] if x not in used] if len(adj[v]) == 2 else []
            e = nxt[0] if nxt else None
    for v, es in adj.items():
        if len(es) != 2:
            for e in es:
                if e not in used:
                    chains += 1
                    walk(v, e, 0.0)
    for e in chosen:
        if e not in used:
            chains += 1
            walk(me.edges[e].vertices[0], e, 0.0)
    return s_at, chains


_ED_LEN = {}


def _stitch_layer(low, uv_name, ids, isl, uv, w, h, rules, mm_per_unit):
    """Height (mm) and px-per-mm per texel from the stitch rules, drawn per face-side of each chosen edge."""
    np = _np()
    me = low.data
    mw = low.matrix_world
    H = np.zeros((h, w), dtype=np.float32)
    K = np.zeros((h, w), dtype=np.float32)
    thread = np.zeros((h, w), dtype=np.float32)
    groove = np.zeros((h, w), dtype=np.float32)
    edge_faces = {}
    for p in me.polygons:
        ls = list(p.loop_indices)
        for i, li in enumerate(ls):
            lj = ls[(i + 1) % len(ls)]
            edge_faces.setdefault(me.loops[li].edge_index, []).append((p.index, li, lj))
    face_uv_centre = {}
    rows_out = []
    for k, r in enumerate(rules):
        if r["edges"] == "BOUNDARY":
            chosen = [e.index for e in me.edges if len(edge_faces.get(e.index, ())) == 1]
        elif r["edges"] == "UV_SEAMS":
            chosen = [e.index for e in me.edges if e.use_seam]
        elif r["edges"] == "SHARP":
            chosen = [e.index for e in me.edges if e.use_edge_sharp]
        else:
            bad = [i for i in r["edgeIndices"] if i < 0 or i >= len(me.edges)]
            if bad:
                raise MifOpError("stitches[%d].edgeIndices %s are not edges of '%s' (0..%d). %s"
                                 % (k, bad[:10], low.name, len(me.edges) - 1, TAIL))
            chosen = list(r["edgeIndices"])
        if not chosen:
            raise MifOpError("stitches[%d]: '%s' has no %s edge. %s" % (
                k, low.name, {"BOUNDARY": "open boundary", "UV_SEAMS": "UV seam", "SHARP": "sharp"}[r["edges"]], TAIL))
        for e in chosen:
            a, b = me.edges[e].vertices
            _ED_LEN[e] = ((mw @ me.vertices[a].co) - (mw @ me.vertices[b].co)).length * mm_per_unit
        s_at, chains = _edge_chains(me, chosen)
        seam_half = r["seamWidthMm"] * 0.5 if r["seamDepthMm"] > 0 else 0.0
        offs = [r["stitchOffsetMm"] + j * r["rowGapMm"] for j in range(r["rows"])]
        tw = r["threadWidthMm"] * 0.5
        reach = max([seam_half] + [o + tw for o in offs]) + 0.5
        period = r["stitchLengthMm"] + r["stitchSpacingMm"]
        drawn = 0
        for e in chosen:
            for f, li, lj in edge_faces.get(e, ()):
                va, vb = me.loops[li].vertex_index, me.loops[lj].vertex_index
                pa = np.array(uv[li].uv, dtype=np.float64) * (w, h)
                pb = np.array(uv[lj].uv, dtype=np.float64) * (w, h)
                L = float(np.linalg.norm(pb - pa))
                len_mm = _ED_LEN[e]
                if L < 1e-6 or len_mm < 1e-6:
                    continue
                kpx = L / len_mm                                 # texels per mm along this edge in this island
                d = (pb - pa) / L
                if f not in face_uv_centre:
                    face_uv_centre[f] = np.mean([np.array(uv[x].uv) for x in me.polygons[f].loop_indices], axis=0) * (w, h)
                c = face_uv_centre[f]
                side = 1.0 if (d[0] * (c[1] - pa[1]) - d[1] * (c[0] - pa[0])) > 0 else -1.0
                rp = reach * kpx + 1.5
                x0, x1 = int(max(0, math.floor(min(pa[0], pb[0]) - rp))), int(min(w - 1, math.ceil(max(pa[0], pb[0]) + rp)))
                y0, y1 = int(max(0, math.floor(min(pa[1], pb[1]) - rp))), int(min(h - 1, math.ceil(max(pa[1], pb[1]) + rp)))
                if x1 < x0 or y1 < y0:
                    continue
                xs = np.arange(x0, x1 + 1, dtype=np.float64) + 0.5
                ys = np.arange(y0, y1 + 1, dtype=np.float64)[:, None] + 0.5
                rx, ry = xs - pa[0], ys - pa[1]
                t = (rx * d[0] + ry * d[1]) / L
                dist = side * (d[0] * ry - d[1] * rx) / kpx      # mm from the edge into this face's side
                own = (ids[y0:y1 + 1, x0:x1 + 1] == isl[f]) & (t >= 0.0) & (t <= 1.0) & (dist >= 0.0) & (dist <= reach)
                if not own.any():
                    continue
                sa, sb = s_at[e][va], s_at[e][vb]
                s = sa + t * (sb - sa)
                hh = np.zeros(own.shape, dtype=np.float64)
                th = np.zeros(own.shape, dtype=np.float64)
                gr = np.zeros(own.shape, dtype=np.float64)
                if seam_half > 0:
                    x = np.clip(dist / seam_half, 0.0, 1.0)
                    g = r["seamDepthMm"] * (1.0 - x * x) * (dist < seam_half)
                    hh -= g
                    gr = np.maximum(gr, g / max(r["seamDepthMm"], 1e-6))
                if offs and period > 0:
                    ph = np.mod(s, period)
                    u = ph / max(r["stitchLengthMm"], 1e-6)
                    on = ph < r["stitchLengthMm"]
                    along = np.where(on, np.sqrt(np.clip(np.sin(np.pi * np.clip(u, 0, 1)), 0, 1)), 0.0)
                    for o in offs:
                        ac = np.clip(np.abs(dist - o) / tw, 0.0, 1.0)
                        across = np.sqrt(1.0 - ac * ac)
                        bump = r["threadHeightMm"] * across * along
                        th = np.maximum(th, bump / max(r["threadHeightMm"], 1e-6))
                        # where the thread goes through: a small pit at each dash end, on the gap side
                        hole_s = np.minimum(np.abs(ph - r["stitchLengthMm"]), np.abs(period - ph))
                        rr = np.sqrt((hole_s / tw) ** 2 + ac ** 2)
                        hh += bump - 0.35 * r["threadHeightMm"] * np.clip(1.0 - rr, 0.0, 1.0) * (~on)
                sub = (slice(y0, y1 + 1), slice(x0, x1 + 1))
                Hs, Ks, Ts, Gs = H[sub], K[sub], thread[sub], groove[sub]
                hh32 = hh.astype(np.float32)
                take_new = own & (np.abs(hh32) > np.abs(Hs))
                Hs[take_new] = hh32[take_new]
                Ks[own] = np.maximum(Ks[own], kpx)
                Ts[own] = np.maximum(Ts[own], (th * r["aoStrength"]).astype(np.float32)[own])
                Gs[own] = np.maximum(Gs[own], (gr * r["aoStrength"]).astype(np.float32)[own])
                drawn += 1
        rows_out.append({"edges": r["edges"] or "edgeIndices", "edgeCount": len(chosen), "chains": chains,
                         "faceSidesDrawn": drawn,
                         "lengthMm": round(sum(_ED_LEN[e] for e in chosen), 1),
                         "stitchesPerChain": (round(sum(_ED_LEN[e] for e in chosen) / period, 1)
                                              if offs and period > 0 else 0),
                         "rows": r["rows"], "stitchLengthMm": r["stitchLengthMm"],
                         "stitchSpacingMm": r["stitchSpacingMm"], "stitchOffsetMm": r["stitchOffsetMm"],
                         "seamWidthMm": r["seamWidthMm"], "seamDepthMm": r["seamDepthMm"]})
    return H, K, thread, groove, rows_out


def _apply_stitch_layer(files, layer, green, ids):
    """Compose the layer into the NORMAL (whiteout blend of the height's tangent normal) and AO (grooves and the
    ground beside each thread darkened) files, in place. Returns texels changed per map."""
    np = _np()
    H, K, thread, groove = layer
    out = {}
    drawn = (K > 0) & (ids >= 0)
    if "NORMAL" in files:
        a = _load(files["NORMAL"])
        gy, gx = np.gradient(H.astype(np.float64))                # mm per texel, rows run up the V axis
        kk = np.where(K > 0, K, 1.0)
        dx, dy = gx * kk, gy * kk                                  # mm per mm
        n2 = np.stack([-dx, -dy, np.ones_like(dx)], axis=-1)
        n2 /= np.linalg.norm(n2, axis=-1, keepdims=True)
        n1 = a[..., :3].astype(np.float64) * 2.0 - 1.0
        if green == "DIRECTX":
            n1[..., 1] *= -1.0
        n = np.stack([n1[..., 0] + n2[..., 0], n1[..., 1] + n2[..., 1], n1[..., 2] * n2[..., 2]], axis=-1)
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
        if green == "DIRECTX":
            n[..., 1] *= -1.0
        upd = drawn & ((np.abs(dx) > 1e-6) | (np.abs(dy) > 1e-6))
        res = a.copy()
        res[..., :3][upd] = (n[upd] * 0.5 + 0.5).astype(np.float32)
        _save(res, files["NORMAL"])
        out["NORMAL"] = int(upd.sum())
    if "AO" in files:
        a = _load(files["AO"])
        # the ground at each thread's foot: the thread mask spread 2 texels, less the thread itself
        sp = thread.copy()
        for _ in range(2):
            p = np.pad(sp, 1, mode="edge")
            sp = np.maximum.reduce([p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        foot = np.clip(sp - thread, 0.0, 1.0)
        mul = (1.0 - 0.55 * groove) * (1.0 - 0.45 * foot)
        upd = drawn & (mul < 0.999)
        res = a.copy()
        for c in range(3):
            res[..., c] = np.where(upd, a[..., c] * mul, a[..., c])
        _save(res, files["AO"])
        out["AO"] = int(upd.sum())
    return out


# ----------------------------------------------------------------------------------------------------- the checks

def _check_maps(files, ids, green, checks):
    np = _np()
    inner = _interior(ids)
    n_in = int(inner.sum())
    rows = {}
    for t, fp in files.items():
        a = _load(fp)
        if a.shape[:2] != ids.shape:
            raise MifOpError("the baked %s is %dx%d, the raster %dx%d. %s" % (t, a.shape[1], a.shape[0], ids.shape[1],
                                                                            ids.shape[0], TAIL))
        rgb = a[..., :3][inner]
        magenta = (rgb[:, 0] > 0.9) & (rgb[:, 1] < 0.1) & (rgb[:, 2] > 0.9)
        black = rgb.max(axis=1) < 0.004
        row = {"file": fp, "interiorTexels": n_in,
               "coveragePercent": round(100.0 * (1.0 - float((magenta | black).sum()) / max(n_in, 1)), 3),
               "blackTexels": int(black.sum()), "untouchedTexels": int(magenta.sum()),
               "blackPercent": round(100.0 * float((magenta | black).sum()) / max(n_in, 1), 3)}
        if t == "NORMAL":
            n = rgb.astype(np.float64) * 2.0 - 1.0
            if green == "DIRECTX":
                n[:, 1] *= -1.0
            ln = np.linalg.norm(n, axis=1)
            ok = ln > 0.5
            nn = n[ok] / ln[ok][:, None]
            mean = nn.mean(axis=0) if len(nn) else np.array([0.0, 0.0, 1.0])
            mlen = float(np.linalg.norm(mean))
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, mean[2] / mlen)))) if mlen > 1e-9 else 90.0
            ang = np.degrees(np.arccos(np.clip(nn[:, 2], -1.0, 1.0)))
            row.update({"meanNormal": [round(float(c), 4) for c in mean], "meanTiltDeg": round(tilt, 3),
                        "detailPercent": round(100.0 * float((ang > 3.0).sum()) / max(len(nn), 1), 3),
                        "strongDetailPercent": round(100.0 * float((ang > 10.0).sum()) / max(len(nn), 1), 3),
                        "invertedPercent": round(100.0 * float((nn[:, 2] < 0).sum()) / max(len(nn), 1), 3),
                        "angleP95Deg": round(float(np.percentile(ang, 95)), 2) if len(ang) else 0.0})
        else:
            v = rgb[:, 0]
            row.update({"mean": round(float(v.mean()), 4), "min": round(float(v.min()), 4),
                        "p05": round(float(np.percentile(v, 5)), 4), "p95": round(float(np.percentile(v, 95)), 4)})
        rows[t] = row
    return rows


def _verdict(rows, miss, checks, final):
    """Every failed check, worded with its fix. Misses and black texels are read off the plain bake; flatness off the
    final maps, so a seamed panel the stitch layer gave its detail passes."""
    why = []
    if miss["percent"] > checks["maxMissPercent"]:
        why.append("%.2f%% of the bake's rays miss the detail mesh (limit %.2f%%) - at %s; raise cageDistance or "
                   "maxRayDistance, or check the detail mesh covers the game mesh" % (
                       miss["percent"], checks["maxMissPercent"], miss["where"][:3]))
    for t, r in rows.items():
        if r["blackPercent"] > checks["maxBlackPercent"]:
            why.append("%s has %.2f%% black or unbaked texels inside UV islands (limit %.2f%%)"
                       % (t, r["blackPercent"], checks["maxBlackPercent"]))
    nr = rows.get("NORMAL")
    if nr is not None:
        if nr["meanTiltDeg"] > checks["maxMeanTiltDeg"]:
            why.append("the normal map's mean leans %.1f degrees off flat (limit %.1f) - a cage or ray reach that "
                       "lands on the wrong surface, or a green channel the other way" % (nr["meanTiltDeg"],
                                                                                       checks["maxMeanTiltDeg"]))
    fr = final.get("NORMAL")
    if fr is not None and fr["detailPercent"] < checks["minDetailPercent"]:
        why.append("the normal map is flat: %.2f%% of its texels lean more than 3 degrees (minimum %.2f%%) - the "
                   "detail mesh carries nothing the game mesh does not" % (fr["detailPercent"],
                                                                         checks["minDetailPercent"]))
    if nr is not None:
        if nr["invertedPercent"] > checks["maxBlackPercent"]:
            why.append("%.2f%% of the normal map points into the surface - rays from a cage inside the detail mesh"
                       % nr["invertedPercent"])
    return why


# --------------------------------------------------------------------------------------------------- the recipe

def op_recipe_detail_bake(params):
    """A high-to-low detail bake in one call: the detail mesh (detail, or build from the game mesh: a Catmull-Clark
    subdivide, then folds, displace_by_rule rules and sculpt_stroke strokes), a cage pushed out by the measured gap, NORMAL,
    AO and CURVATURE through bake_texture_set, seams and stitches drawn into the NORMAL and AO maps, and the checks before it
    answers: ray misses counted, black or unbaked texels inside UV islands, the normal map's mean tilt and its share of
    detail. Refused when a check fails: the meshes it made are removed, the maps stay on disk to look at.

    params:
      object (str)              the game mesh (UVs required) - the low
      detail (str)              an existing high poly; OR
      build {levels, maxFaces, name, folds, rules, sculpt}
                                make it from the low: a Catmull-Clark subdivide `levels` deep (default: enough that the
                                median edge is under the narrowest rule width, up to 4 and maxFaces, default 2,000,000),
                                then folds [{axis, period, depth, irregularity, sharpness, seed, vertexGroup}] (ridges across
                                axis, scene units), rules (displace_by_rule's: SEAM, STITCH, WEAVE, KNIT) and sculpt
                                (sculpt_stroke calls, object filled in). Kept as `name` (default <object>_detail).
      types (list)              NORMAL, AO, CURVATURE (default all three)
      folder, pattern, width, height, samples, margin, uvLayer, greenChannel   as bake_texture_set (width/height 2048,
                                samples 32, margin 8 by default)
      cage (str) | cageDistance (scene units)  an existing cage, or the low pushed out this far (default: the measured
                                outward gap x 1.25 + 1 mm, at least 2 mm); maxRayDistance the reach past the low (default
                                the measured inward gap x 1.25 + 2 mm). keepCage (bool) keeps the made cage.
      occluders (list)          objects left visible for the AO; every other geometry object is hidden for the bake
      stitches [{edges BOUNDARY | UV_SEAMS | SHARP, or edgeIndices, seamWidthMm, seamDepthMm, stitchLengthMm,
                stitchSpacingMm, stitchOffsetMm, rows, rowGapMm, threadWidthMm, threadHeightMm, aoStrength}]
                                seams and stitches drawn into the NORMAL and AO maps along the low's edges, in mm
      checks {maxMissPercent 1, maxBlackPercent 0.5, maxMeanTiltDeg 8, minDetailPercent 1}
      outOfProcess (bool)       bake in a second Blender (bake_texture_set's job); jobId alone then polls, and the
                                status call that sees it finish draws the stitches and runs the checks
    """
    reject_unknown(params, _KEYS, "recipe_detail_bake")
    if params.get("jobId") is not None:
        extra = [k for k in params if k != "jobId"]
        if extra:
            raise MifOpError("jobId asks a detail bake's status and takes nothing else (got %s). %s" % (extra, TAIL))
        return _status(take(params, "jobId", kind=str))
    low = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    if not low.data.uv_layers:
        raise MifOpError("'%s' has no UV layer - a bake writes through UVs; uv_unwrap makes one. %s" % (low.name, TAIL))
    uv_name = take(params, "uvLayer", default=None, kind=str)
    if uv_name is not None and low.data.uv_layers.get(uv_name) is None:
        raise MifOpError("'%s' has no UV layer %r (it has %s). %s" % (low.name, uv_name,
                                                                      ", ".join(u.name for u in low.data.uv_layers), TAIL))
    uv_name = uv_name or (low.data.uv_layers.active or low.data.uv_layers[0]).name
    detail_name = take(params, "detail", default=None, kind=str)
    build = params.get("build")
    if (detail_name is None) == (build is None):
        raise MifOpError("pass detail (an existing high poly) OR build {levels, folds, rules, sculpt} to make one from "
                         "'%s'. %s" % (low.name, TAIL))
    detail = None
    if detail_name is not None:
        detail = get_object(detail_name, want_mesh=True)
        if detail is low:
            raise MifOpError("detail is the game mesh itself - the detail is the high poly baked from. %s" % TAIL)
    else:
        if not isinstance(build, dict) or set(build) - _BUILD_KEYS:
            raise MifOpError("build takes %s, got %r. %s" % (", ".join(sorted(_BUILD_KEYS)), build, TAIL))
    types = params.get("types", list(_TYPES))
    if not isinstance(types, list) or not types:
        raise MifOpError("types is a non-empty list of %s. %s" % (", ".join(_TYPES), TAIL))
    types = [str(t).upper() for t in types]
    bad = [t for t in types if t not in _TYPES]
    if bad or len(set(types)) != len(types):
        raise MifOpError("types %s: each of %s at most once. %s" % (types, ", ".join(_TYPES), TAIL))
    raw_folder = take(params, "folder", required=True, kind=str)
    folder = os.path.abspath(bpy.path.abspath(raw_folder))
    try:
        check_writable_dir(folder)
    except (OSError, ValueError) as exc:
        raise MifOpError("the folder '%s' cannot be written (%s). %s" % (folder, exc, TAIL))
    pattern = take(params, "pattern", default="T_{object}_{suffix}", kind=str)
    width = take_int(params, "width", default=2048)
    height = take_int(params, "height", default=2048)
    if not (16 <= width <= 8192 and 16 <= height <= 8192):
        raise MifOpError("width/height must be 16-8192, got %dx%d. %s" % (width, height, TAIL))
    samples = take_int(params, "samples", default=32)
    margin = take_int(params, "margin", default=8)
    green = str(take(params, "greenChannel", default="OPENGL", kind=str)).upper()
    if green not in ("OPENGL", "DIRECTX"):
        raise MifOpError("greenChannel is OPENGL or DIRECTX, got %r. %s" % (green, TAIL))
    cage_name = take(params, "cage", default=None, kind=str)
    cage_d = take_float(params, "cageDistance", default=None)
    if cage_name is not None and cage_d is not None:
        raise MifOpError("cage names a cage; cageDistance pushes the low out to make one - pass one. %s" % TAIL)
    if cage_d is not None and not cage_d > 0:
        raise MifOpError("cageDistance must be above 0 (scene units), got %g. %s" % (cage_d, TAIL))
    reach = take_float(params, "maxRayDistance", default=None)
    if reach is not None and not reach > 0:
        raise MifOpError("maxRayDistance must be above 0 (scene units), got %g. %s" % (reach, TAIL))
    cage = None
    if cage_name is not None:
        cage = get_object(cage_name, want_mesh=True)
        if len(cage.data.vertices) != len(low.data.vertices) or len(cage.data.polygons) != len(low.data.polygons):
            raise MifOpError("cage '%s' is not the low's topology (%d/%d vertices, %d/%d faces). %s" % (
                cage.name, len(cage.data.vertices), len(low.data.vertices), len(cage.data.polygons),
                len(low.data.polygons), TAIL))
    occ = params.get("occluders") or []
    if not isinstance(occ, list) or not all(isinstance(n, str) for n in occ):
        raise MifOpError("occluders is a list of object names left visible for the AO. %s" % TAIL)
    occluders = [get_object(n) for n in dict.fromkeys(occ)]
    stitches = _parse_stitches(params.get("stitches"))
    if stitches and "NORMAL" not in types and "AO" not in types:
        raise MifOpError("stitches draw into the NORMAL and AO maps, and types has neither. %s" % TAIL)
    raw_checks = params.get("checks") or {}
    if not isinstance(raw_checks, dict) or set(raw_checks) - _CHECK_KEYS:
        raise MifOpError("checks takes %s. %s" % (", ".join(sorted(_CHECK_KEYS)), TAIL))
    checks = {"maxMissPercent": _f(raw_checks, "maxMissPercent", 1.0, 0.0, 100.0, "checks."),
              "maxBlackPercent": _f(raw_checks, "maxBlackPercent", 0.5, 0.0, 100.0, "checks."),
              "maxMeanTiltDeg": _f(raw_checks, "maxMeanTiltDeg", 8.0, 0.0, 90.0, "checks."),
              "minDetailPercent": _f(raw_checks, "minDetailPercent", 1.0, 0.0, 100.0, "checks.")}
    out_of_process = take_bool(params, "outOfProcess", default=False)
    keep_cage = take_bool(params, "keepCage", default=False)
    mem = take_float(params, "memoryLimitGB", default=None)
    if mem is not None and not out_of_process:
        raise MifOpError("memoryLimitGB caps the second Blender - pass outOfProcess true too. %s" % TAIL)
    files = {t: os.path.join(folder, pattern.replace("{object}", low.name).replace(
        "{suffix}", {"NORMAL": "N", "AO": "AO", "CURVATURE": "Curv"}[t]) + ".png") for t in types}
    for fp in files.values():
        check_output_path(fp, fp, "baked")

    # THE BUILD's own refusals, read before anything is made
    folds, rules, sculpt, levels, det_name, max_faces = [], [], [], None, None, 2000000
    if build is not None:
        from .ops_garment import _dr_rule                       # noqa: PLC0415
        folds = _parse_folds(build.get("folds"))
        raw_rules = build.get("rules") or []
        if not isinstance(raw_rules, list) or len(raw_rules) > 64:
            raise MifOpError("build.rules is a list of up to 64 displace_by_rule rules. %s" % TAIL)
        try:
            parsed = [_dr_rule(k, r, TAIL) for k, r in enumerate(raw_rules)]
        except MifOpError as exc:
            raise MifOpError("build.%s" % exc)
        rules = list(raw_rules)
        sculpt = build.get("sculpt") or []
        if not isinstance(sculpt, list) or not all(isinstance(s, dict) for s in sculpt) or len(sculpt) > 64:
            raise MifOpError("build.sculpt is a list of up to 64 sculpt_stroke calls (object is filled in). %s" % TAIL)
        if any("object" in s for s in sculpt):
            raise MifOpError("build.sculpt strokes go on the detail mesh this makes - leave object out of them. %s"
                             % TAIL)
        max_faces = int(_f(build, "maxFaces", 2000000, 1000, 20000000, "build."))
        det_name = str(build.get("name") or "%s_detail" % low.name)
        if bpy.data.objects.get(det_name) is not None:
            raise MifOpError("an object named '%s' already exists - build.name names the new detail mesh; pick another "
                             "or pass it as detail. %s" % (det_name, TAIL))
        if not folds and not rules and not sculpt:
            raise MifOpError("build makes a detail mesh from folds, rules or sculpt, and has none - a subdivided copy "
                             "alone bakes a flat map. %s" % TAIL)
        widths = [p["width"] for p in parsed if p.get("width")]
        med = _long_edge(low)
        nf = len(low.data.polygons)
        if build.get("levels") is not None:
            levels = int(_f(build, "levels", 2, 1, 6, "build."))
        else:
            levels = 1
            # THE FINEST DETAIL DECIDES: a fold sampled at three vertices a period aliases into hatching in the bake
            # (measured on a garment: an 18 mm fold on 6 mm edges came out as 4-texel ripples)
            target = min([w_ / 1.5 for w_ in widths] + [f["period"] / 8.0 for f in folds] or [med / 4.0])
            while levels < 4 and med / (2 ** levels) > target and nf * 4 ** (levels + 1) <= max_faces:
                levels += 1
        if nf * 4 ** levels > max_faces:
            raise MifOpError("levels %d makes about %d faces from %d, over maxFaces %d. %s"
                             % (levels, nf * 4 ** levels, nf, max_faces, TAIL))

    started = time.time()
    unit = bpy.context.scene.unit_settings
    mm_per_unit = 1000.0 * (unit.scale_length or 1.0)
    made = []
    report = {"object": low.name, "uvLayer": uv_name}
    try:
        if build is not None:
            detail = _subdivided_copy(low, levels, det_name)
            made.append(detail)
            b = {"name": detail.name, "levels": levels, "faces": len(detail.data.polygons),
                 "longEdgeLow": round(_long_edge(low), 6)}
            if folds:
                b["folds"] = _apply_folds(detail, folds)
            if rules:
                from .ops_garment import op_displace_by_rule     # noqa: PLC0415
                r = op_displace_by_rule({"object": detail.name, "rules": rules})
                b["rules"] = r.get("rules") or r
            if sculpt:
                from .ops_sculpt import op_sculpt_stroke         # noqa: PLC0415
                b["sculpt"] = []
                for k, s in enumerate(sculpt):
                    r = op_sculpt_stroke(dict(s, object=detail.name))
                    b["sculpt"].append({k_: r.get(k_) for k_ in ("brush", "verticesMoved", "moved", "maxMove") if k_ in r})
            report["build"] = b
        out_gap, in_gap = _measure_gap(low, detail)
        made_cage = cage is None
        if cage is None:
            cd = cage_d if cage_d is not None else max(out_gap * 1.25 + 0.001 / (mm_per_unit / 1000.0),
                                                       0.002 / (mm_per_unit / 1000.0))
            cage = _make_cage(low, cd, "%s_cage" % low.name)
            made.append(cage)
        else:
            cd = cage_d if cage_d is not None else out_gap * 1.25 + 0.001
        rr = reach if reach is not None else in_gap * 1.25 + 0.002 / (mm_per_unit / 1000.0)
        miss_n, miss_of, miss_where = _ray_misses(low, detail, cd, rr)
        miss = {"rays": miss_of, "missed": miss_n, "percent": round(100.0 * miss_n / max(miss_of, 1), 3),
                "where": miss_where}
        report["cage"] = {"name": cage.name, "made": made_cage, "distance": round(cd, 6),
                          "maxRayDistance": round(cd + rr, 6), "gapOut": round(out_gap, 6), "gapIn": round(in_gap, 6),
                          "kept": (keep_cage or not made_cage)}
        report["rays"] = miss
        if miss["percent"] > checks["maxMissPercent"]:
            raise MifOpError("%.2f%% of the bake's rays would miss the detail mesh (limit %.2f%%), first at %s - "
                             "raise cageDistance or maxRayDistance, or check the detail mesh covers '%s'. %s"
                             % (miss["percent"], checks["maxMissPercent"], miss_where[:3], low.name, TAIL))
        # THE AO SEES ONLY WHAT IT SHOULD: the low, the detail and the occluders named; the rest hidden for the bake
        keep = {low, detail, cage} | set(occluders)
        hidden = [o for o in bpy.context.scene.objects
                  if o not in keep and not o.hide_render and o.type in ("MESH", "CURVE", "SURFACE", "FONT", "META",
                                                                         "CURVES", "POINTCLOUD", "VOLUME")]
        for o in hidden:
            o.hide_render = True
        # THE DETAIL MUST RENDER FOR THE BAKE. 3.6.23 and 4.2.17 bake nothing from a source hidden from render (a flat
        # normal map, every ray missed); 4.4.0, 5.0.1 and 5.2.1 bake it. A detail this recipe built and hid is hidden,
        # so it is shown for the bake and put back after.
        shown = [o for o in (detail,) if o.hide_render]
        for o in shown:
            o.hide_render = False
        try:
            from .ops_gamebake import op_bake_texture_set        # noqa: PLC0415
            req = {"object": low.name, "types": types, "folder": folder, "pattern": pattern, "width": width,
                   "height": height, "samples": samples, "margin": margin, "uvLayer": uv_name,
                   "source": detail.name, "cage": cage.name, "maxRayDistance": cd + rr, "greenChannel": green,
                   "outOfProcess": out_of_process}
            if "NORMAL" not in types:
                req.pop("greenChannel")
            if mem is not None:
                req["memoryLimitGB"] = mem
            baked = op_bake_texture_set(req)
        finally:
            for o in hidden:
                o.hide_render = False
            for o in shown:
                o.hide_render = True
        report["hiddenForBake"] = len(hidden)
        report["occluders"] = [o.name for o in occluders] or None
        job = {"low": low.name, "uv": uv_name, "w": width, "h": height, "files": files, "green": green,
               "stitches": stitches, "checks": checks, "miss": miss, "mm": mm_per_unit, "report": report,
               "made": [o.name for o in made], "detail": detail.name, "keepCage": keep_cage, "cage": cage.name, "madeCage": made_cage,
               "started": started}
        if out_of_process:
            jid = "detail_" + baked["jobId"]
            job["bakeJob"] = baked["jobId"]
            _JOBS[jid] = job
            dl = caller_deadline()
            return dict(report, jobId=jid, bakeJobId=baked["jobId"], outOfProcess=True, maps=types, files=files,
                        snapshot=baked.get("snapshot"), pending="poll recipe_detail_bake {jobId}: the status call that "
                        "sees the bake finish draws the stitches and runs the checks",
                        callerDeadline=dl)
        return _finish(job, baked)
    except BaseException:
        for o in made:
            _remove(o)
        raise


def _remove(o):
    try:
        me = o.data
        bpy.data.objects.remove(o)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)
    except (ReferenceError, RuntimeError):
        pass


def _reload_images(paths):
    """Reload every image datablock of this session that reads one of these files; their names."""
    want = {os.path.normcase(os.path.abspath(p)) for p in paths}
    out = []
    for im in bpy.data.images:
        if im.filepath and os.path.normcase(os.path.abspath(bpy.path.abspath(im.filepath))) in want:
            im.reload()
            out.append(im.name)
    return out


def _finish(job, baked):
    """Stitch layer, checks, verdict - on the baked files, in this Blender."""
    low = bpy.data.objects.get(job["low"])
    if low is None:
        raise MifOpError("the game mesh '%s' is gone - the maps are on disk at %s, unchecked." % (job["low"],
                                                                                              job["files"]))
    ids, isl, uv = _uv_raster(low, job["uv"], job["w"], job["h"])
    if not (ids >= 0).any():
        raise MifOpError("no texel of '%s' lies inside its UV layer %s - the islands are outside 0-1 or degenerate. %s"
                         % (low.name, job["uv"], TAIL))
    report = dict(job["report"])
    rows_raw = _check_maps(job["files"], ids, job["green"], job["checks"])
    if job["stitches"]:
        H, K, thread, groove, srows = _stitch_layer(low, job["uv"], ids, isl, uv, job["w"], job["h"], job["stitches"],
                                                    job["mm"])
        changed = _apply_stitch_layer({t: f for t, f in job["files"].items() if t in ("NORMAL", "AO")},
                                      (H, K, thread, groove), job["green"], ids)
        # THE SESSION'S COPY OF EACH MAP IS READ AGAIN. An in-process bake leaves its image in bpy.data pointing at the
        # file, and a material wired from that file afterwards took the datablock in memory - the plain bake: the
        # test garment rendered with no stitches while the file on disk had them.
        reloaded = _reload_images([f for t, f in job["files"].items() if t in changed])
        report["stitchLayer"] = {"rules": srows, "texelsChanged": changed, "imagesReloaded": reloaded,
                                 "deepestMm": round(float(-H.min()), 3), "highestMm": round(float(H.max()), 3)}
        if not any(changed.values()):
            raise MifOpError("the stitch layer drew nothing into the maps - its edges fall outside the UV islands. The "
                             "maps on disk are the plain bake. %s" % TAIL)
    rows = _check_maps(job["files"], ids, job["green"], job["checks"])
    if job["stitches"] and "NORMAL" in rows:
        rows["NORMAL"]["detailPercentBeforeStitches"] = rows_raw["NORMAL"]["detailPercent"]
    why = _verdict(rows_raw, job["miss"], job["checks"], rows)
    report["maps"] = rows
    report["bake"] = {k: baked.get(k) for k in ("outOfProcess", "complete", "jobId") if k in baked}
    report["files"] = job["files"]
    report["elapsedSeconds"] = round(time.time() - job["started"], 1)
    if not job["keepCage"] and job["madeCage"]:
        c = bpy.data.objects.get(job["cage"])
        if c is not None:
            _remove(c)
            report["cage"]["kept"] = False
    if why:
        raise MifOpError("the detail bake failed its checks: %s. The maps are on disk (%s) for a look; nothing was "
                         "wired. %s" % ("; ".join(why), ", ".join(job["files"].values()), "Fix and bake again."))
    # THE DETAIL MESH IS KEPT, HIDDEN FROM RENDER: it sits on the game mesh, and a render showed both fighting through
    # each other in patches (found on the test garment's after-render). A detail the caller passed is theirs to hide.
    det = bpy.data.objects.get(job["detail"])
    if det is not None:
        if job["detail"] in job["made"]:
            det.hide_render = True
            report["detailMesh"] = {"name": det.name, "hiddenFromRender": True}
        elif not det.hide_render:
            report["detailMesh"] = {"name": det.name, "hiddenFromRender": False,
                                    "note": "it lies on the game mesh and shows through it in a render - hide it"}
    report["checksPassed"] = sorted(job["checks"])
    report["checks"] = job["checks"]
    return report


def _status(jid):
    job = _JOBS.get(jid)
    if job is None:
        return {"unknownJob": True, "jobId": jid, "knownJobs": sorted(_JOBS),
                "error": "no detail-bake job '%s' in this Blender (never started here, or Blender restarted)" % jid}
    if job.get("result") is not None:
        return job["result"]
    from .ops_gamebake import op_bake_texture_set               # noqa: PLC0415
    st = op_bake_texture_set({"jobId": job["bakeJob"]})
    if st.get("running"):
        return {"jobId": jid, "running": True, "bake": st}
    if not st.get("complete"):
        raise MifOpError("the bake job ended without every map: %s" % (st.get("error") or st))
    try:
        job["result"] = dict(_finish(job, st), jobId=jid, complete=True)
    except MifOpError as exc:
        job["result"] = {"jobId": jid, "complete": False, "ok": False, "error": str(exc)}
        for n in job["made"]:                                   # as in process: what it made goes with a refusal
            o = bpy.data.objects.get(n)
            if o is not None:
                _remove(o)
        raise
    return job["result"]


OPS = {
    "recipe_detail_bake": op_recipe_detail_bake,
}
