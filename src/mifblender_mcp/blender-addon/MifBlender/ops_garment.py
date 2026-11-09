"""MifBlender ops: garments and characters - measure a garment against a body, and how it deforms.

THE GAP (spec, "Blender: garments and characters", 2026-10-05): the garment passes measured a cloth against its body,
a cuff against its sleeve, a skinned garment over a run's frames - each in raw Python, by region, again and again. The
ops here are those measurements as typed calls. They READ: nothing in this module writes a mesh unless its docstring
says so.

SHARED CONTRACT, every op in this module:
  * world units, each mesh as it evaluates now (modifiers, shape keys and pose applied) unless the op says otherwise;
  * a region is a world box [[minX,minY,minZ],[maxX,maxY,maxZ]], a sphere {center, radius} or a vertex group
    {vertexGroup, minWeight}; a call whose region holds nothing is refused - a measurement of nothing is not clean;
  * statistics are min, median, the asked percentiles (linear between ranks, as numpy's default) and max.
"""
from __future__ import annotations

import bisect
import math

import bmesh
import bpy
from mathutils import Vector

from .ops_common import (remove_objects, MifOpError, finite_float, finite_floats, finite_int, get_object, mesh_counts, reject_unknown,
                         require_editable, rnd, shared_data_note, simulation_guard, take, take_bool, take_float,
                         take_int)


# --------------------------------------------------------------------------------------------- shared helpers

def percentile(sorted_vals, p):
    """The p-th percentile (0-100) of an ascending list, linear between ranks (numpy's default)."""
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def stats(values, pcts, places=6):
    """{count, min, median, p<N>..., max} of a list of numbers (None when empty)."""
    if not values:
        return None
    s = sorted(values)
    out = {"count": len(s), "min": round(s[0], places), "median": round(percentile(s, 50), places)}
    for p in pcts:
        out["p%s" % (int(p) if float(p).is_integer() else p)] = round(percentile(s, p), places)
    out["max"] = round(s[-1], places)
    return out


def parse_percentiles(params, default=(5, 95)):
    raw = params.get("percentiles")
    if raw is None:
        return list(default)
    if not isinstance(raw, (list, tuple)) or not raw or len(raw) > 10:
        raise MifOpError("percentiles must be a list of 1-10 numbers from 0 to 100, got %r. NOTHING was read." % (raw,))
    vals = finite_floats(list(raw), "percentiles")
    if any(not 0 <= v <= 100 for v in vals):
        raise MifOpError("percentiles must lie from 0 to 100, got %r. NOTHING was read." % (raw,))
    return vals


def parse_region(params, key="region", tail="NOTHING was read."):
    """A region test fn(world_point, vertex_index, obj) -> bool, and its description; (None, None) when not given."""
    raw = params.get(key)
    if raw is None:
        return None, None
    if not isinstance(raw, dict) or len([k for k in ("box", "sphere", "vertexGroup") if k in raw]) != 1:
        raise MifOpError("%s is one of {box: [[minX,minY,minZ],[maxX,maxY,maxZ]]}, {sphere: {center: [x,y,z], radius}} "
                         "or {vertexGroup: name, minWeight?}, got %r. %s" % (key, raw, tail))
    extra = set(raw) - {"box", "sphere", "vertexGroup", "minWeight"}
    if extra:
        raise MifOpError("%s has unknown key(s) %s. %s" % (key, ", ".join(sorted(extra)), tail))
    if "box" in raw:
        b = raw["box"]
        if not isinstance(b, (list, tuple)) or len(b) != 2 or any(not isinstance(c, (list, tuple)) or len(c) != 3
                                                                    for c in b):
            raise MifOpError("%s.box is [[minX,minY,minZ],[maxX,maxY,maxZ]] in world units, got %r. %s" % (key, b, tail))
        a, c = finite_floats(list(b[0]), key), finite_floats(list(b[1]), key)
        lo = [min(a[i], c[i]) for i in range(3)]
        hi = [max(a[i], c[i]) for i in range(3)]
        return (lambda p, i, o: all(lo[k] <= p[k] <= hi[k] for k in range(3))), {"box": [lo, hi]}
    if "sphere" in raw:
        s = raw["sphere"]
        if not isinstance(s, dict) or "center" not in s or "radius" not in s:
            raise MifOpError("%s.sphere is {center: [x,y,z], radius}, got %r. %s" % (key, s, tail))
        c = Vector(finite_floats(list(s["center"]), key))
        r = finite_float(s["radius"], "%s.sphere radius" % key)
        if not (math.isfinite(r) and r > 0):
            raise MifOpError("%s.sphere radius must be above 0, got %r. %s" % (key, s["radius"], tail))
        return (lambda p, i, o: (Vector(p) - c).length <= r), {"sphere": {"center": list(c), "radius": r}}
    name = raw["vertexGroup"]
    min_w = finite_float(raw.get("minWeight", 0.5), "%s.minWeight" % key)
    cache = {}

    def in_group(p, i, o):
        if o.name not in cache:
            g = o.vertex_groups.get(str(name))
            if g is None:
                raise MifOpError("'%s' has no vertex group '%s' (it has: %s). %s"
                                 % (o.name, name, ", ".join(x.name for x in o.vertex_groups) or "none", tail))
            w = {}
            for v in o.data.vertices:
                for ge in v.groups:
                    if ge.group == g.index:
                        w[v.index] = ge.weight
            cache[o.name] = w
        return cache[o.name].get(i, 0.0) >= min_w
    return in_group, {"vertexGroup": str(name), "minWeight": min_w}


def evaluated_world(obj, evaluated=True):
    """(bmesh in world space, its object) - the evaluated mesh (modifiers, keys, pose) or the base mesh. Caller frees.
    Vertex indices match the base mesh only when no modifier changes the topology."""
    bm = bmesh.new()
    if evaluated:
        ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = ev.to_mesh()
        try:
            bm.from_mesh(me)
        finally:
            ev.to_mesh_clear()
    else:
        bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    bm.normal_update()
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    return bm


# --------------------------------------------------------------------------------------------- mesh_distance

_DIST_KEYS = ("object", "target", "bins", "percentiles", "region", "evaluated", "maxDistance", "limit")


def op_mesh_distance(params):
    """How far one mesh stands from another, overall and by region - a garment from its body, a cuff from its sleeve.

    THE GAP: the garment passes measured the gap from a cloth to the body, binned by height and by angle round the body's
    axis (a hood stood 5-7 cm off at the neck base at rest), and the edge length and cross-section by region, each in
    raw Python. EACH VERTEX of `object` is measured to the nearest point of `target`'s surface (a BVH), SIGNED by the
    target's face normal there: positive outside, negative inside (`inside` counts them). A vertex farther than
    maxDistance (default 1 m) is left out of the statistics and counted as `far`.

    BINS {zStep, sectors, center}: height bands of zStep metres from the object's lowest vertex, and sectors equal
    angles round a vertical axis through center [x, y] (default the target's bounding-box centre), counted
    counter-clockwise from +X. Each bin reports gap stats, `radius` stats (the object's vertices' distance from the
    axis - its cross-section there) and `edgeLength` stats of the object's edges whose midpoint falls in it.

    params: object, target (meshes, not the same), bins, percentiles (default [5, 95]), region (box | sphere |
    vertexGroup on object), evaluated (default true), maxDistance, limit (bin rows, default 500).
    """
    from mathutils.bvhtree import BVHTree
    reject_unknown(params, _DIST_KEYS, "mesh_distance")
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    tgt = get_object(take(params, "target", required=True, kind=str), want_mesh=True)
    if obj is tgt:
        raise MifOpError("object and target are the same mesh - a mesh is 0 from itself. NOTHING was read.")
    pcts = parse_percentiles(params)
    region, region_desc = parse_region(params)
    evaluated = take_bool(params, "evaluated", default=True)
    max_d = take_float(params, "maxDistance", default=1.0)
    limit = take_int(params, "limit", default=500)
    if not (math.isfinite(max_d) and max_d > 0):
        raise MifOpError("maxDistance must be above 0, got %r. NOTHING was read." % max_d)
    if not 1 <= limit <= 5000:
        raise MifOpError("limit must be 1-5000 bin rows, got %d. NOTHING was read." % limit)
    bins = params.get("bins")
    z_step = sectors = center = None
    if bins is not None:
        if not isinstance(bins, dict) or not bins or set(bins) - {"zStep", "sectors", "center"}:
            raise MifOpError("bins is {zStep (metres), sectors (count), center [x, y]?}, got %r. NOTHING was read."
                             % (bins,))
        if bins.get("zStep") is not None:
            z_step = finite_float(bins["zStep"], "bins.zStep")
            if not (math.isfinite(z_step) and z_step > 0):
                raise MifOpError("bins.zStep must be above 0 metres, got %r. NOTHING was read." % bins["zStep"])
        if bins.get("sectors") is not None:
            sectors = bins["sectors"]
            if isinstance(sectors, bool) or not isinstance(sectors, int) or not 1 <= sectors <= 360:
                raise MifOpError("bins.sectors must be a whole number 1-360, got %r. NOTHING was read." % (sectors,))
        if bins.get("center") is not None:
            c = bins["center"]
            if not isinstance(c, (list, tuple)) or len(c) != 2:
                raise MifOpError("bins.center is [x, y] (world), got %r. NOTHING was read." % (c,))
            center = finite_floats(list(c), "bins.center")
        if z_step is None and sectors is None:
            raise MifOpError("bins needs zStep, sectors or both. NOTHING was read.")
    if not len(tgt.data.polygons) and not evaluated:
        raise MifOpError("'%s' has no faces to measure against. NOTHING was read." % tgt.name)

    tbm = evaluated_world(tgt, evaluated)
    obm = evaluated_world(obj, evaluated)
    try:
        if not len(tbm.faces):
            raise MifOpError("'%s' has no faces to measure against. NOTHING was read." % tgt.name)
        tree = BVHTree.FromBMesh(tbm)
        if center is None:
            xs = [v.co.x for v in tbm.verts]
            ys = [v.co.y for v in tbm.verts]
            center = [(min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0]
        cx, cy = center
        pts = []
        for v in obm.verts:
            if region is not None and not region(v.co, v.index, obj):
                continue
            pts.append(v)
        if not pts:
            raise MifOpError("no vertex of '%s' is in the region %s - a measurement of nothing is not clean. NOTHING was "
                             "read." % (obj.name, region_desc))
        z0 = min(v.co.z for v in pts)

        def key_of(p):
            zb = int(math.floor((p.z - z0) / z_step)) if z_step else 0
            if sectors:
                ang = math.degrees(math.atan2(p.y - cy, p.x - cx)) % 360.0
                sb = min(int(ang // (360.0 / sectors)), sectors - 1)
            else:
                sb = 0
            return zb, sb

        gaps_all, inside, far = [], 0, 0
        per_bin = {}
        for v in pts:
            loc, nrm, _i, d = tree.find_nearest(v.co, max_d)
            if loc is None:
                far += 1
                continue
            g = d if (v.co - loc).dot(nrm) >= 0 else -d
            if g < 0:
                inside += 1
            gaps_all.append(g)
            if bins is not None:
                row = per_bin.setdefault(key_of(v.co), {"gap": [], "radius": [], "edge": [], "inside": 0})
                row["gap"].append(g)
                row["radius"].append(math.hypot(v.co.x - cx, v.co.y - cy))
                row["inside"] += g < 0
        in_set = {v.index for v in pts}
        edges_all = []
        for e in obm.edges:
            a, b = e.verts
            if a.index not in in_set or b.index not in in_set:
                continue
            ln = (a.co - b.co).length
            edges_all.append(ln)
            if bins is not None:
                mid = (a.co + b.co) / 2.0
                row = per_bin.get(key_of(mid))
                if row is not None:
                    row["edge"].append(ln)
        if not gaps_all:
            raise MifOpError("every one of the %d vertices measured is farther than maxDistance (%g) from '%s' - nothing "
                             "to report. Raise maxDistance. NOTHING was read." % (len(pts), max_d, tgt.name))
        out = {"object": obj.name, "target": tgt.name, "evaluated": evaluated, "region": region_desc,
               "verticesMeasured": len(pts), "far": far, "maxDistance": max_d, "inside": inside,
               "gap": stats(gaps_all, pcts), "edgeLength": stats(edges_all, pcts),
               "signNote": ("gap is signed by the target's face normal at the nearest point: positive outside, "
                            "negative inside. Against a target with an inner layer (a lining) the nearest face can be "
                            "the inner one - measure against one layer at a time.")}
        if bins is not None:
            rows = []
            for (zb, sb), r in sorted(per_bin.items()):
                row = {}
                if z_step:
                    row["zFrom"] = round(z0 + zb * z_step, 6)
                    row["zTo"] = round(z0 + (zb + 1) * z_step, 6)
                if sectors:
                    row["sector"] = sb
                    row["angleFrom"] = round(sb * 360.0 / sectors, 4)
                    row["angleTo"] = round((sb + 1) * 360.0 / sectors, 4)
                row.update({"vertices": len(r["gap"]), "inside": r["inside"], "gap": stats(r["gap"], pcts),
                            "radius": stats(r["radius"], pcts), "edgeLength": stats(r["edge"], pcts)})
                rows.append(row)
            out["bins"] = rows[:limit]
            out["binCount"] = len(rows)
            out["binsTruncated"] = len(rows) > limit
            out["axis"] = {"center": [round(cx, 6), round(cy, 6)], "zFrom": round(z0, 6), "zStep": z_step,
                           "sectors": sectors, "angleNote": "degrees counter-clockwise from +X round the axis"}
        return out
    finally:
        tbm.free()
        obm.free()



# --------------------------------------------------------------------------------------------- deformation_stretch

_STRETCH_KEYS = ("object", "frames", "regions", "action", "threshold", "percentiles", "worst", "allowSimulation")


def _top_weights(obj, vi, n=3):
    names = {g.index: g.name for g in obj.vertex_groups}
    ws = sorted(((ge.weight, names.get(ge.group, "?")) for ge in obj.data.vertices[vi].groups if ge.weight > 0),
                reverse=True)[:n]
    return [[nm, round(w, 4)] for w, nm in ws]


def op_deformation_stretch(params):
    """How much a skinned mesh's edges stretch over animation frames, by region - a hard weight seam from cloth that moves.

    THE GAP: a garment's inner-sleeve vertices 100% on the upper arm beside torso vertices 100% on the spine tore 30-95x
    with the arm up, and only posed-over-rest edge length per region said so (the body's own armpit skin: p95 3.4x).
    EACH FRAME: the scene goes to the frame, the mesh is read as it evaluates (armature and every other modifier, the
    object's transform applied) and every edge's length is divided by its REST length - the base mesh in world scale.
    A region is a box, a sphere or a vertex group IN REST SPACE (the base mesh under the object's transform); an edge
    belongs to it when its rest midpoint is inside (both ends in the group for a vertex group). Without regions, one
    region 'all'.

    Per region and frame: edges, p05 / p50 / p95 (or the percentiles asked) / max of the ratio, and shareAbove - the
    share of edges above threshold (default 1.5). A sliver edge inflates max, so read p95 beside it. worst (default 10)
    lists the most-stretched edges over every frame and region: the edge, its frame, ratio, rest length, rest midpoint
    and each end's top three weights.

    action plays that action on the mesh's armature for the call and puts the old one back. The scene frame is put
    back. A frame change steps live simulations, so allowSimulation is asked as for every frame-moving op. Refused: a
    modifier that changes the mesh's topology (edges would not match their rest), no frames, more than 500 frames, a
    region with no edge.
    """
    reject_unknown(params, _STRETCH_KEYS, "deformation_stretch")
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    raw = params.get("frames")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw or len(raw) > 500:
        raise MifOpError("frames is a list of 1-500 frame numbers, got %r. NOTHING was read." % (raw,))
    frames = [finite_int(f, "frames") for f in raw]
    threshold = take_float(params, "threshold", default=1.5)
    if not (math.isfinite(threshold) and threshold > 0):
        raise MifOpError("threshold must be above 0, got %r. NOTHING was read." % threshold)
    pcts = parse_percentiles(params, default=(5, 50, 95))
    n_worst = take_int(params, "worst", default=10)
    if not 0 <= n_worst <= 200:
        raise MifOpError("worst must be 0-200, got %d. NOTHING was read." % n_worst)
    regions_raw = params.get("regions")
    if regions_raw is not None and (not isinstance(regions_raw, dict) or not regions_raw or len(regions_raw) > 50):
        raise MifOpError("regions is {name: {box | sphere | vertexGroup}} with 1-50 entries, got %r. NOTHING was read."
                         % (regions_raw,))
    regions = []
    for name, spec in (regions_raw or {}).items():
        test, desc = parse_region({"region": spec})
        regions.append((str(name), test, desc, spec))
    act_name = take(params, "action", default=None, kind=str)
    rig = next((m.object for m in obj.modifiers if m.type == "ARMATURE" and m.object is not None), None)
    action = None
    if act_name is not None:
        action = bpy.data.actions.get(act_name)
        if action is None:
            raise MifOpError("no action named '%s'. NOTHING was read." % act_name)
        if rig is None:
            raise MifOpError("action plays on the mesh's armature, and '%s' has no Armature modifier with an object. "
                             "NOTHING was read." % obj.name)

    me = obj.data
    mw = obj.matrix_world
    rest = [mw @ v.co for v in me.vertices]
    edges = [tuple(e.vertices) for e in me.edges]
    if not edges:
        raise MifOpError("'%s' has no edges. NOTHING was read." % obj.name)
    rest_len = [(rest[a] - rest[b]).length for a, b in edges]
    groups = []          # (name, desc, edge indices)
    if regions:
        for name, test, desc, spec in regions:
            if "vertexGroup" in spec:
                idx = [i for i, (a, b) in enumerate(edges) if test(rest[a], a, obj) and test(rest[b], b, obj)]
            else:
                idx = [i for i, (a, b) in enumerate(edges) if test((rest[a] + rest[b]) / 2.0, a, obj)]
            if not idx:
                raise MifOpError("region '%s' %s holds no edge of '%s' in rest space - a measurement of nothing is not "
                                 "clean. NOTHING was read." % (name, desc, obj.name))
            groups.append((name, desc, idx))
    else:
        groups.append(("all", None, list(range(len(edges)))))
    ev0 = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    em0 = ev0.to_mesh()
    try:
        counts = (len(em0.vertices), len(em0.edges))
    finally:
        ev0.to_mesh_clear()
    if counts != (len(me.vertices), len(edges)):
        raise MifOpError("'%s' evaluates to %d vertices and %d edges and its base mesh has %d and %d - a modifier changes "
                         "the topology, so no edge can be set against its rest length. Measure before that modifier "
                         "(its show_viewport off) or apply it. NOTHING was read."
                         % (obj.name, counts[0], counts[1], len(me.vertices), len(edges)))
    sim = simulation_guard(params, frames, "deformation_stretch", tail="NOTHING was read, and the frame did not move.")

    sc = bpy.context.scene
    started = sc.frame_current
    ad = rig.animation_data if rig is not None else None
    old_action = ad.action if ad is not None else None
    if action is not None:
        if rig.animation_data is None:
            rig.animation_data_create()
        rig.animation_data.action = action
    per = {name: [] for name, _d, _i in groups}
    worst, skipped = [], []
    try:
        for f in frames:
            sc.frame_set(f)
            ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            em = ev.to_mesh()
            try:
                # CHECKED BEFORE THE FRAME MOVED (above); a modifier animated into changing the topology at one frame
                # leaves that frame out and says so rather than refusing with the frame and action already moved.
                same = len(em.vertices) == len(me.vertices) and len(em.edges) == len(edges)
                ewm = ev.matrix_world
                pos = [ewm @ v.co for v in em.vertices] if same else None
            finally:
                ev.to_mesh_clear()
            if pos is None:
                skipped.append(f)
                continue
            ratio = [((pos[a] - pos[b]).length / rest_len[i]) if rest_len[i] > 1e-12 else 1.0
                     for i, (a, b) in enumerate(edges)]
            for name, _d, idx in groups:
                vals = [ratio[i] for i in idx]
                row = stats(vals, pcts, 4)
                row.pop("count", None)
                row = dict({"frame": f, "edges": len(idx)}, **row)
                row["shareAbove"] = round(sum(1 for v in vals if v > threshold) / len(vals), 6)
                per[name].append(row)
            if n_worst:
                tops = sorted(range(len(edges)), key=lambda i: ratio[i], reverse=True)[:n_worst]
                worst.extend((ratio[i], f, i) for i in tops)
    finally:
        if action is not None:
            rig.animation_data.action = old_action
        sc.frame_set(started)
    worst.sort(reverse=True)
    seen, rows = set(), []
    for r, f, i in worst:
        if i in seen:
            continue
        seen.add(i)
        a, b = edges[i]
        rows.append({"edge": i, "vertices": [a, b], "frame": f, "ratio": round(r, 4),
                     "restLength": round(rest_len[i], 6),
                     "restMidpoint": [round(c, 5) for c in (rest[a] + rest[b]) / 2.0],
                     "weights": [_top_weights(obj, a), _top_weights(obj, b)]})
        if len(rows) >= n_worst:
            break
    out = {"object": obj.name, "armature": rig.name if rig is not None else None, "action": act_name,
           "frames": frames, "threshold": threshold, "edges": len(edges),
           "regions": [{"name": name, "region": d, "edges": len(idx), "perFrame": per[name],
                        "maxRatio": max((r["max"] for r in per[name]), default=None),
                        "maxP95": max((r.get("p95", r["max"]) for r in per[name]), default=None)}
                       for name, d, idx in groups],
           "worst": rows, "frameRestored": sc.frame_current == started}
    if skipped:
        out["framesSkipped"] = skipped
        out["framesSkippedNote"] = ("at these frames a modifier changed the mesh's topology, so no edge could be set "
                                    "against its rest length; they are left out.")
    out.update(sim)
    return out



# --------------------------------------------------------------------------------------------- blend_weights

_BLEND_KEYS = ("object", "source", "mask", "smooth", "smoothFactor", "maxInfluences", "normalize", "groups", "dryRun",
               "previewId", "previewTtlSeconds")


def _mask_shapes(raw):
    """mask -> [fn(world point, vertex index, obj) -> 0..1]: a list (or one) of {box | sphere | band, falloff} or
    {vertexGroup}. The value is 1 inside a shape, falling linearly to 0 over `falloff` metres outside it; a vertex group's
    value is its weight. Several shapes take the largest value."""
    shapes = raw if isinstance(raw, (list, tuple)) else [raw]
    if not shapes or len(shapes) > 20:
        raise MifOpError("mask is one shape or a list of 1-20: {box, falloff} | {sphere, falloff} | {band: {zFrom, zTo}, "
                         "falloff} | {vertexGroup}. Got %r. NOTHING was changed." % (raw,))
    out, vg_cache = [], {}
    for sh in shapes:
        if not isinstance(sh, dict) or len([k for k in ("box", "sphere", "band", "vertexGroup") if k in sh]) != 1:
            raise MifOpError("each mask shape is exactly one of box, sphere, band or vertexGroup (with falloff for the "
                             "first three), got %r. NOTHING was changed." % (sh,))
        extra = set(sh) - {"box", "sphere", "band", "vertexGroup", "falloff"}
        if extra:
            raise MifOpError("a mask shape has unknown key(s) %s. NOTHING was changed." % ", ".join(sorted(extra)))
        if "vertexGroup" in sh:
            if "falloff" in sh:
                raise MifOpError("a vertexGroup mask is its own weights - it takes no falloff. NOTHING was changed.")
            name = str(sh["vertexGroup"])

            def vg(p, i, o, name=name, cache=vg_cache):
                key = (o.name, name)
                if key not in cache:
                    g = o.vertex_groups.get(name)
                    if g is None:
                        raise MifOpError("'%s' has no vertex group '%s' for the mask. NOTHING was changed." % (o.name, name))
                    cache[key] = {v.index: ge.weight for v in o.data.vertices for ge in v.groups if ge.group == g.index}
                return cache[key].get(i, 0.0)
            out.append(vg)
            continue
        fall = finite_float(sh.get("falloff", 0.0), "mask falloff")
        if fall < 0:
            raise MifOpError("mask falloff must be 0 or more metres, got %g. NOTHING was changed." % fall)
        if "band" in sh:
            b = sh["band"]
            if not isinstance(b, dict) or set(b) != {"zFrom", "zTo"}:
                raise MifOpError("a band is {zFrom, zTo} (world heights), got %r. NOTHING was changed." % (b,))
            z0, z1 = sorted((finite_float(b["zFrom"], "band zFrom"), finite_float(b["zTo"], "band zTo")))

            def dist(p, z0=z0, z1=z1):
                return max(z0 - p[2], 0.0, p[2] - z1)
        elif "sphere" in sh:
            test, desc = parse_region({"region": {"sphere": sh["sphere"]}}, tail="NOTHING was changed.")
            c, r = Vector(desc["sphere"]["center"]), desc["sphere"]["radius"]

            def dist(p, c=c, r=r):
                return max((Vector(p) - c).length - r, 0.0)
        else:
            test, desc = parse_region({"region": {"box": sh["box"]}}, tail="NOTHING was changed.")
            lo, hi = desc["box"]

            def dist(p, lo=lo, hi=hi):
                return math.sqrt(sum(max(lo[k] - p[k], 0.0, p[k] - hi[k]) ** 2 for k in range(3)))

        def fn(p, i, o, dist=dist, fall=fall):
            d = dist(p)
            if d <= 0.0:
                return 1.0
            return max(0.0, 1.0 - d / fall) if fall > 0 else 0.0
        out.append(fn)
    return out


def op_blend_weights(params):
    """Blend a source's skin weights into a garment under a soft mask, then smooth inside the mask.

    THE GAP: transfer_weights' destinationMask is in or out, so a region given the body's weights jumped from the
    garment's weights to the body's at its edge. The fix that worked, in raw Python: transfer the body's weights onto a
    copy, blend them in with a falloff (a neck band by height, spheres at the armpits), smooth only where the mask is
    above 0, cap the influences and normalize.

    THE SOURCE'S WEIGHTS at each garment vertex are interpolated over the nearest face of `source` (world space, its
    evaluated surface; the weights are its base mesh's, so a modifier that changes its topology is refused). mask is one
    shape or a list - {box, falloff} | {sphere: {center, radius}, falloff} | {band: {zFrom, zTo}, falloff} |
    {vertexGroup} - worth 1 inside a shape and falling linearly to 0 over falloff metres; the largest value counts. Each
    group: new = (1 - m) x garment + m x source. Then `smooth` passes (default 0) move each masked vertex's weights
    toward its neighbours' mean by smoothFactor (default 0.5) times m; maxInfluences keeps the largest that many groups
    per masked vertex; normalize (default true) makes a masked vertex's blended groups sum to 1.

    THE GROUPS blended are the garment's armature's deform bones when it has an Armature modifier, else every group of
    either object but a mask's own vertex group; `groups` names them instead (leave a pin group out that way). A group the source
    has and the garment lacks is made. dryRun reports and writes nothing. Refused: no vertex under the mask, a source
    with no faces or no weights in those groups, source = object.
    """
    reject_unknown(params, _BLEND_KEYS, "blend_weights")
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    src = get_object(take(params, "source", required=True, kind=str), want_mesh=True)
    if obj is src:
        raise MifOpError("object and source are the same mesh. NOTHING was changed.")
    if params.get("mask") is None:
        raise MifOpError("mask is required - where the source's weights come in. NOTHING was changed.")
    shapes = _mask_shapes(params.get("mask"))
    smooth = take_int(params, "smooth", default=0)
    factor = take_float(params, "smoothFactor", default=0.5)
    cap = take_int(params, "maxInfluences", default=None)
    normalize = take_bool(params, "normalize", default=True)
    dry = take_bool(params, "dryRun", default=False)
    if not 0 <= smooth <= 200:
        raise MifOpError("smooth must be 0-200 passes, got %d. NOTHING was changed." % smooth)
    if not (math.isfinite(factor) and 0 < factor <= 1):
        raise MifOpError("smoothFactor must be above 0 and at most 1, got %r. NOTHING was changed." % factor)
    if cap is not None and not 1 <= cap <= 32:
        raise MifOpError("maxInfluences must be 1-32, got %d. NOTHING was changed." % cap)
    raw_groups = params.get("groups")
    rig = next((m.object for m in obj.modifiers if m.type == "ARMATURE" and m.object is not None), None)
    src_names = [g.name for g in src.vertex_groups]
    if raw_groups is not None:
        if not isinstance(raw_groups, (list, tuple)) or not raw_groups or any(not isinstance(g, str) for g in raw_groups):
            raise MifOpError("groups is a list of group names, got %r. NOTHING was changed." % (raw_groups,))
        names = list(dict.fromkeys(raw_groups))
        missing = [g for g in names if g not in src_names and obj.vertex_groups.get(g) is None]
        if missing:
            raise MifOpError("neither '%s' nor '%s' has group(s) %s. NOTHING was changed."
                             % (obj.name, src.name, ", ".join(missing)))
        chosen_by = "groups"
    elif rig is not None and rig.type == "ARMATURE":
        names = [b.name for b in rig.data.bones if b.use_deform]
        chosen_by = "armature '%s' deform bones" % rig.name
    else:
        mask_groups = {str(sh.get("vertexGroup")) for sh in (params["mask"] if isinstance(params["mask"], (list, tuple))
                                                             else [params["mask"]]) if isinstance(sh, dict)}
        names = [g for g in dict.fromkeys(src_names + [g.name for g in obj.vertex_groups]) if g not in mask_groups]
        chosen_by = "both objects' groups but the mask's"
    if not names:
        raise MifOpError("no group to blend (%s gives none). NOTHING was changed." % chosen_by)

    # mathutils' submodules AFTER every refusal that needs no geometry: the offline refusal tests stub mathutils without
    # interpolate, and an import first turned an unknown-key or missing-object refusal into an ImportError.
    from mathutils.bvhtree import BVHTree
    from mathutils.interpolate import poly_3d_calc
    sbm = evaluated_world(src, True)
    try:
        if len(sbm.verts) != len(src.data.vertices):
            raise MifOpError("'%s' evaluates to %d vertices and its base mesh has %d - a modifier changes its topology, "
                             "so its weights cannot be read at the evaluated surface. NOTHING was changed."
                             % (src.name, len(sbm.verts), len(src.data.vertices)))
        if not len(sbm.faces):
            raise MifOpError("'%s' has no faces to take weights from. NOTHING was changed." % src.name)
        tree = BVHTree.FromBMesh(sbm)
        sidx = {g.index: g.name for g in src.vertex_groups if g.name in names}
        if not sidx:
            raise MifOpError("'%s' weighs none of the groups blended (%s: %s). NOTHING was changed."
                             % (src.name, chosen_by, ", ".join(names[:20])))
        sw = [{} for _ in src.data.vertices]
        for v in src.data.vertices:
            for ge in v.groups:
                nm = sidx.get(ge.group)
                if nm is not None and ge.weight > 0:
                    sw[v.index][nm] = ge.weight
        mw = obj.matrix_world
        oidx = {g.index: g.name for g in obj.vertex_groups}
        n = len(obj.data.vertices)
        cur = [{} for _ in range(n)]
        for v in obj.data.vertices:
            for ge in v.groups:
                nm = oidx.get(ge.group)
                if nm in names and ge.weight > 0:
                    cur[v.index][nm] = ge.weight
        mask = [0.0] * n
        world = [mw @ v.co for v in obj.data.vertices]
        for i in range(n):
            mask[i] = max(fn(world[i], i, obj) for fn in shapes)
        masked = [i for i in range(n) if mask[i] > 0.0]
        if not masked:
            raise MifOpError("no vertex of '%s' is under the mask - a blend of nothing is not clean. NOTHING was "
                             "changed." % obj.name)
        new = [dict(c) for c in cur]
        for i in masked:
            loc, _n, fi, _d = tree.find_nearest(world[i])
            face = sbm.faces[fi]
            bw = poly_3d_calc([lv.co for lv in face.verts], loc)
            got = {}
            for lv, b in zip(face.verts, bw):
                for nm, w in sw[lv.index].items():
                    got[nm] = got.get(nm, 0.0) + w * b
            m = mask[i]
            keys = set(cur[i]) | set(got)
            new[i] = {nm: (1.0 - m) * cur[i].get(nm, 0.0) + m * got.get(nm, 0.0) for nm in keys}
    finally:
        sbm.free()
    if smooth:
        nbr = [[] for _ in range(n)]
        for e in obj.data.edges:
            a, b = e.vertices
            nbr[a].append(b)
            nbr[b].append(a)
        for _ in range(smooth):
            nxt = list(new)
            for i in masked:
                if not nbr[i]:
                    continue
                keys = set(new[i]).union(*(new[j].keys() for j in nbr[i]))
                k = factor * mask[i]
                nxt[i] = {nm: (1 - k) * new[i].get(nm, 0.0) + k * sum(new[j].get(nm, 0.0) for j in nbr[i]) / len(nbr[i])
                          for nm in keys}
            new = nxt
    capped = 0
    for i in masked:
        row = {nm: w for nm, w in new[i].items() if w > 1e-6}
        if cap is not None and len(row) > cap:
            row = dict(sorted(row.items(), key=lambda kv: kv[1], reverse=True)[:cap])
            capped += 1
        if normalize:
            tot = sum(row.values())
            if tot > 0:
                row = {nm: w / tot for nm, w in row.items()}
        new[i] = row
    changes = [max([abs(new[i].get(nm, 0.0) - cur[i].get(nm, 0.0)) for nm in set(new[i]) | set(cur[i])] or [0.0])
               for i in masked]
    made = [nm for nm in names if obj.vertex_groups.get(nm) is None and any(nm in new[i] for i in masked)]
    out = {"object": obj.name, "source": src.name, "groupsBlended": chosen_by, "groups": len(names),
           "verticesMasked": len(masked), "verticesFullyMasked": sum(1 for i in masked if mask[i] >= 1.0),
           "smoothPasses": smooth, "maxInfluences": cap, "cappedVertices": capped, "normalized": normalize,
           "groupsCreated": made, "weightChange": stats(changes, [95], 4),
           "maxInfluencesAfter": max(len(new[i]) for i in masked), "dryRun": dry}
    if dry:
        return out
    require_editable(obj, "reweight")
    for nm in made:
        obj.vertex_groups.new(name=nm)
    groups = {g.name: g for g in obj.vertex_groups}
    for i in masked:
        for nm in names:
            g = groups.get(nm)
            if g is None:
                continue
            w = new[i].get(nm, 0.0)
            if w > 0:
                g.add([i], w, "REPLACE")
            elif nm in cur[i]:
                g.remove([i])
    obj.data.update()
    return out



# --------------------------------------------------------------------------------------------- facing a body

def parse_facing(raw, obj, tail="NOTHING was measured."):
    """mesh_quality facingAgainst -> (body object, bandStep, reach), checked before anything is measured."""
    if isinstance(raw, str):
        raw = {"body": raw}
    if not isinstance(raw, dict) or not raw.get("body") or set(raw) - {"body", "bandStep", "reach"}:
        raise MifOpError("facingAgainst is a body's name or {body, bandStep (m, default 0.1), reach (m, default 0.2)}, "
                         "got %r. %s" % (raw, tail))
    body = get_object(str(raw["body"]), want_mesh=True)
    if body is obj:
        raise MifOpError("facingAgainst names the mesh itself. %s" % tail)
    step = finite_float(raw.get("bandStep", 0.1), "facingAgainst.bandStep")
    reach = finite_float(raw.get("reach", 0.2), "facingAgainst.reach")
    if not step > 0 or not reach > 0:
        raise MifOpError("facingAgainst.bandStep and reach must be above 0, got %r and %r. %s" % (step, reach, tail))
    return body, step, reach


def facing_report(obj, body, step, reach):
    """Which faces of obj point INTO the body: the face normal against the direction from the nearest body point to the
    face's centre, for faces within reach. obj as its base mesh under its transform, the body as it evaluates."""
    from mathutils.bvhtree import BVHTree
    bbm = evaluated_world(body, True)
    try:
        if not len(bbm.faces):
            raise MifOpError("'%s' has no faces to face against. NOTHING was measured." % body.name)
        tree = BVHTree.FromBMesh(bbm)
    finally:
        bbm.free()
    mw = obj.matrix_world
    m3 = mw.to_3x3().inverted().transposed()
    rows, inward, measured, far = {}, [], 0, 0
    z0 = None
    hits = []
    for poly in obj.data.polygons:
        c = mw @ poly.center
        loc, _n, _i, d = tree.find_nearest(c, reach)
        if loc is None:
            far += 1
            continue
        nrm = (m3 @ poly.normal).normalized()
        into = d > 1e-9 and nrm.dot(c - loc) < 0
        hits.append((c.z, poly.index, into))
        z0 = c.z if z0 is None else min(z0, c.z)
    for z, idx, into in hits:
        measured += 1
        band = int(math.floor((z - z0) / step))
        row = rows.setdefault(band, [0, 0])
        row[0] += 1
        if into:
            row[1] += 1
            inward.append(idx)
    if not measured:
        raise MifOpError("no face of '%s' is within reach (%g) of '%s' - a facing check of nothing is not clean. Raise "
                         "reach. NOTHING was measured." % (obj.name, reach, body.name))
    bands = [{"zFrom": round(z0 + b * step, 5), "zTo": round(z0 + (b + 1) * step, 5), "faces": n, "inward": k,
              "share": round(k / n, 6)} for b, (n, k) in sorted(rows.items())]
    return {"body": body.name, "facesMeasured": measured, "facesBeyondReach": far, "inward": len(inward),
            "inwardShare": round(len(inward) / measured, 6), "inwardFaces": sorted(inward)[:50],
            "inwardFacesTruncated": len(inward) > 50, "bands": bands, "bandStep": step, "reach": reach,
            "note": ("a face is inward when its normal points toward the nearest point of the body. One-sided materials "
                     "hide such a face and show the hole; a folded-under edge faces the body by design.")}


# --------------------------------------------------------------------------------------------- coverage across poses

_COVER_KEYS = ("garment", "body", "frames", "compareTo", "region", "reach", "limit", "allowSimulation")
_PUSH_KEYS = ("garment", "body", "frames", "region", "reach", "margin", "spread", "maxPush", "passes", "dryRun",
              "allowSimulation", "previewId", "previewTtlSeconds")


def _parse_frames(params, tail):
    raw = params.get("frames")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw or len(raw) > 500:
        raise MifOpError("frames is a list of 1-500 frame numbers, got %r. %s" % (raw, tail))
    return list(dict.fromkeys(finite_int(f, "frames") for f in raw))


def _body_setup(params, body, tail):
    """(body vertex indices to test, by the rest-space region; region description), refused when it holds none."""
    region, desc = parse_region(params, tail=tail)
    mw = body.matrix_world
    idx = [v.index for v in body.data.vertices if region is None or region(mw @ v.co, v.index, body)]
    if not idx:
        raise MifOpError("no vertex of '%s' is in the region %s - a measurement of nothing is not clean. %s"
                         % (body.name, desc, tail))
    ev = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    em = ev.to_mesh()
    try:
        n = len(em.vertices)
    finally:
        ev.to_mesh_clear()
    if n != len(body.data.vertices):
        raise MifOpError("'%s' evaluates to %d vertices and its base mesh has %d - a modifier changes its topology, so "
                         "its rest region and its posed vertices cannot be matched. Turn that modifier's show_viewport "
                         "off or apply it. %s" % (body.name, n, len(body.data.vertices), tail))
    return idx, desc


def _posed_body(body, idx):
    """[(world position, world normal)] of the body vertices idx as the body evaluates now, or None when the topology
    changed at this frame."""
    bm = evaluated_world(body, True)
    try:
        if len(bm.verts) != len(body.data.vertices):
            return None
        return [(bm.verts[i].co.copy(), bm.verts[i].normal.copy()) for i in idx]
    finally:
        bm.free()


def _garment_tree(obj):
    """(BVHTree of the garment as it evaluates now in world space, [world vertex positions], [face vertex indices])."""
    from mathutils.bvhtree import BVHTree
    bm = evaluated_world(obj, True)
    try:
        if not len(bm.faces):
            return None, None, None
        return (BVHTree.FromBMesh(bm), [v.co.copy() for v in bm.verts],
                [[v.index for v in f.verts] for f in bm.faces])
    finally:
        bm.free()


def _cast(tree, p, n, reach):
    """(signed gap, hit point, face): the ray along the body normal n from p meets the garment at +gap; when it meets
    nothing, the ray back into the body at -gap (the garment is inside the skin there); (None, None, None) when
    neither meets the garment within reach - a hole."""
    if n.length < 1e-12:
        return None, None, None
    loc, _nrm, fi, d = tree.ray_cast(p, n, reach)
    if loc is not None:
        return d, loc, fi
    loc, _nrm, fi, d = tree.ray_cast(p, -n, reach)
    if loc is not None:
        return -d, loc, fi
    return None, None, None


def _cover_row(tree, posed, idx, reach, margin=0.0):
    """One frame's coverage: (uncovered vertex indices, inside count, hole count, closer-than-margin count, casts)."""
    unc, inside, holes, close, casts = [], 0, 0, 0, []
    for (p, n), vi in zip(posed, idx):
        g, loc, fi = _cast(tree, p, n, reach)
        casts.append((g, loc, fi))
        if g is None:
            holes += 1
            unc.append(vi)
        elif g < 0:
            inside += 1
            unc.append(vi)
        elif g < margin:
            close += 1
    return unc, inside, holes, close, casts


def _frame_loop(frames, fn):
    """fn(frame) at each frame; the scene frame is put back. Returns frameRestored."""
    sc = bpy.context.scene
    started = sc.frame_current
    try:
        for f in frames:
            sc.frame_set(f)
            fn(f)
    finally:
        sc.frame_set(started)
    return sc.frame_current == started


def _cover_objects(params, tail):
    g = get_object(take(params, "garment", required=True, kind=str), want_mesh=True)
    b = get_object(take(params, "body", required=True, kind=str), want_mesh=True)
    if g is b:
        raise MifOpError("garment and body are the same mesh. %s" % tail)
    reach = take_float(params, "reach", default=0.25)
    if not (math.isfinite(reach) and reach > 0):
        raise MifOpError("reach must be above 0 metres, got %r. %s" % (reach, tail))
    return g, b, reach


def op_pose_coverage(params):
    """Where a body shows through a garment, frame by frame - and, against an earlier version, only what a change broke.

    THE GAP: a weight change showed skin through the back of the shoulder only in a run's back-swing, found in raw
    Python; and the nearest-face sign test (mesh_distance) reads a body as OUTSIDE a garment with an inner layer, because
    the nearest face is often the inner one. EACH FRAME the scene goes to the frame and, from every body vertex (in
    region, tested in REST space), a ray along the vertex's POSED normal is cast against the garment as it evaluates:
    a hit within reach (default 0.25 m) covers it. A vertex not covered is `inside` when the ray back into the body
    meets the garment (the garment is under the skin) and a `hole` when neither does.

    compareTo: a second garment (a version before a change) cast at the same frames: per frame `shared` (uncovered by
    both), `newlyUncovered` (by the garment only - what the change broke) and `fixed` (by compareTo only).

    params: garment, body (meshes; the body's modifiers must keep its vertex count), frames (1-500), compareTo, region
    (box | sphere | vertexGroup on the body, rest space), reach, limit (vertex indices listed per row, default 200),
    allowSimulation. The scene frame is put back. Refused: no frames, compareTo = garment or body, a region with no body
    vertex, a garment with no faces.
    """
    tail = "NOTHING was read, and the frame did not move."
    reject_unknown(params, _COVER_KEYS, "pose_coverage")
    garment, body, reach = _cover_objects(params, tail)
    frames = _parse_frames(params, tail)
    cmp_name = take(params, "compareTo", default=None, kind=str)
    cmp_obj = None
    if cmp_name is not None:
        cmp_obj = get_object(cmp_name, want_mesh=True)
        if cmp_obj is garment or cmp_obj is body:
            raise MifOpError("compareTo names the %s itself - compare against another version of the garment. %s"
                             % ("garment" if cmp_obj is garment else "body", tail))
    limit = take_int(params, "limit", default=200)
    if not 1 <= limit <= 100000:
        raise MifOpError("limit must be 1-100000 vertex indices per row, got %d. %s" % (limit, tail))
    idx, desc = _body_setup(params, body, tail)
    for o in (garment, cmp_obj):
        if o is not None and not len(o.data.polygons) and not o.modifiers:
            raise MifOpError("'%s' has no faces for a ray to meet. %s" % (o.name, tail))
    sim = simulation_guard(params, frames, "pose_coverage", tail=tail)

    rows, skipped, ever, ever_cmp = [], [], set(), set()

    def one(f):
        posed = _posed_body(body, idx)
        tree, _p, _f = _garment_tree(garment)
        if posed is None or tree is None:
            skipped.append(f)
            return
        unc, inside, holes, _c, _x = _cover_row(tree, posed, idx, reach)
        row = {"frame": f, "uncovered": len(unc), "share": round(len(unc) / len(idx), 6), "inside": inside,
               "holes": holes, "uncoveredVertices": unc[:limit], "truncated": len(unc) > limit}
        ever.update(unc)
        if cmp_obj is not None:
            ctree, _p, _f = _garment_tree(cmp_obj)
            if ctree is None:
                skipped.append(f)
                return
            cunc = set(_cover_row(ctree, posed, idx, reach)[0])
            ever_cmp.update(cunc)
            mine = set(unc)
            row.update({"compareUncovered": len(cunc), "shared": len(mine & cunc),
                        "newlyUncovered": sorted(mine - cunc)[:limit], "newlyUncoveredCount": len(mine - cunc),
                        "fixed": sorted(cunc - mine)[:limit], "fixedCount": len(cunc - mine)})
        rows.append(row)

    restored = _frame_loop(frames, one)
    out = {"garment": garment.name, "body": body.name, "compareTo": cmp_name, "frames": frames, "region": desc,
           "reach": reach, "verticesTested": len(idx), "perFrame": rows, "uncoveredAnyFrame": len(ever),
           "uncoveredAnyFrameVertices": sorted(ever)[:limit],
           "worstFrame": max(rows, key=lambda r: r["uncovered"])["frame"] if rows else None,
           "frameRestored": restored,
           "note": ("a body vertex is covered when the ray along its posed normal meets the garment within reach; "
                    "'inside' means the garment lies under the skin there, 'holes' that no garment is near either way.")}
    if cmp_obj is not None:
        out["newlyUncoveredAnyFrame"] = len(ever - ever_cmp)
        out["fixedAnyFrame"] = len(ever_cmp - ever)
    if skipped:
        out["framesSkipped"] = skipped
        out["framesSkippedNote"] = "at these frames the body's topology changed or a garment had no faces; left out."
    out.update(sim)
    return out


def _falloff(d, spread):
    x = d / spread
    return (1.0 - x * x) ** 2 if x < 1.0 else 0.0


def op_push_out_for_poses(params):
    """Push a garment's REST vertices out along their normals where a body shows through it in any of the frames given.

    THE GAP: the fix after a coverage failure was raw Python - move the garment's rest vertices out near the failing
    points, enough for the worst pose plus a margin. EACH FRAME, as pose_coverage casts: a body vertex fails when the
    garment is under its skin (need = depth + margin) or covers it closer than margin (need = margin - gap); a hole (no
    garment near either way) is counted and not pushed. A need aims 10% (at least 0.5 mm) past the margin, so the
    passes converge. The garment face the ray met moves its vertices by the full need, and every garment vertex within
    spread (default 0.05 m) of the hit point by need x (1 - (d/spread)^2)^2; the largest need over frames and points
    counts. Rest vertices move along their rest normals (world), capped at maxPush (default
    0.1 m). Then it measures again and repeats, up to `passes` (default 4), until no frame fails.

    Reply: before and after per frame (uncovered, inside, holes, closerThanMargin), verticesMoved, push stats, capped,
    passes. dryRun reports the first pass's push and writes nothing. Refused: margin below 0, spread 0, a garment whose
    modifiers change its vertex count, a garment with shape keys (moving the basis would shift every key).
    """
    tail = "NOTHING was changed, and the frame did not move."
    reject_unknown(params, _PUSH_KEYS, "push_out_for_poses")
    garment, body, reach = _cover_objects(params, tail)
    frames = _parse_frames(params, tail)
    margin = take_float(params, "margin", default=0.005)
    spread = take_float(params, "spread", default=0.05)
    cap = take_float(params, "maxPush", default=0.1)
    passes = take_int(params, "passes", default=4)
    dry = take_bool(params, "dryRun", default=False)
    if not (math.isfinite(margin) and margin >= 0):
        raise MifOpError("margin must be 0 or more metres, got %r. %s" % (margin, tail))
    if not (math.isfinite(spread) and spread > 0):
        raise MifOpError("spread must be above 0 metres, got %r. %s" % (spread, tail))
    if not (math.isfinite(cap) and cap > 0):
        raise MifOpError("maxPush must be above 0 metres, got %r. %s" % (cap, tail))
    if not 1 <= passes <= 20:
        raise MifOpError("passes must be 1-20, got %d. %s" % (passes, tail))
    if garment.data.shape_keys is not None:
        raise MifOpError("'%s' has shape keys - moving its rest vertices would shift every key against the basis. Apply "
                         "or remove them first. %s" % (garment.name, tail))
    idx, desc = _body_setup(params, body, tail)
    me = garment.data
    if not len(me.polygons):
        raise MifOpError("'%s' has no faces to push. %s" % (garment.name, tail))
    ev = garment.evaluated_get(bpy.context.evaluated_depsgraph_get())
    em = ev.to_mesh()
    try:
        gn = len(em.vertices)
    finally:
        ev.to_mesh_clear()
    if gn != len(me.vertices):
        raise MifOpError("'%s' evaluates to %d vertices and its base mesh has %d - a modifier changes its topology, so a "
                         "posed point cannot be traced to a rest vertex. %s" % (garment.name, gn, len(me.vertices), tail))
    if not dry:
        require_editable(garment, "push out")
    sim = simulation_guard(params, frames, "push_out_for_poses", tail=tail)
    from mathutils.kdtree import KDTree

    mw = garment.matrix_world
    mwi = mw.inverted()
    m3 = mw.to_3x3().inverted().transposed()
    rest = [mw @ v.co for v in me.vertices]
    rnorm = [(m3 @ v.normal).normalized() for v in me.vertices]
    total = [0.0] * len(rest)
    # AIMED PAST THE MARGIN: a pushed face tilts, so a push of exactly the need leaves a residue that halves each pass
    # and never reaches the margin (measured: 3, 1, 1, 1 points still short after 1, 2, 4, 8 passes).
    aim = margin + max(0.1 * margin, 0.0005)

    def measure(collect):
        rows, need = [], {}

        def one(f):
            posed = _posed_body(body, idx)
            tree, gpos, gfaces = _garment_tree(garment)
            if posed is None or tree is None:
                rows.append({"frame": f, "skipped": True})
                return
            unc, inside, holes, close, casts = _cover_row(tree, posed, idx, reach, margin)
            rows.append({"frame": f, "uncovered": len(unc), "inside": inside, "holes": holes, "closerThanMargin": close})
            if not collect:
                return
            kd = None
            for g, loc, fi in casts:
                if g is None or g >= margin:
                    continue
                want = aim - g
                for vi in gfaces[fi]:
                    need[vi] = max(need.get(vi, 0.0), want)
                if kd is None:
                    kd = KDTree(len(gpos))
                    for i, p in enumerate(gpos):
                        kd.insert(p, i)
                    kd.balance()
                for _co, vi, d in kd.find_range(loc, spread):
                    w = _falloff(d, spread)
                    if w > 0:
                        need[vi] = max(need.get(vi, 0.0), want * w)
        restored = _frame_loop(frames, one)
        return rows, need, restored

    before, need, restored = measure(True)
    used, capped = 0, set()
    while need and used < passes:
        used += 1
        for vi, w in need.items():
            t = total[vi] + w
            if t > cap:
                t = cap
                capped.add(vi)
            total[vi] = t
        if dry:
            break
        for vi in need:
            me.vertices[vi].co = mwi @ (rest[vi] + rnorm[vi] * total[vi])
        me.update()
        after, need, restored = measure(used < passes)
        need = {vi: w for vi, w in need.items() if total[vi] < cap}
    moved = [w for w in total if w > 0]
    out = {"garment": garment.name, "body": body.name, "frames": frames, "region": desc, "reach": reach,
           "margin": margin, "spread": spread, "maxPush": cap, "verticesTested": len(idx), "before": before,
           "verticesMoved": len(moved), "push": stats(moved, [50, 95], 6), "capped": len(capped), "passes": used,
           "dryRun": dry, "frameRestored": restored}
    if not dry:
        out["after"] = after if used else before
    if not moved:
        out["note"] = "no body vertex in these frames is under the garment's skin or closer than margin - nothing moved."
    out.update(sim)
    return out


# --------------------------------------------------------------------------------------------- reshape toward a body

_RESHAPE_KEYS = ("object", "body", "mode", "mask", "axis", "target", "keep", "smooth", "reach", "dryRun",
                 "previewId", "previewTtlSeconds")


def _face_normals(polys, pos):
    """World normal of each polygon (Newell's method) from world vertex positions - the same sum before and after."""
    out = []
    for vs in polys:
        n = Vector((0.0, 0.0, 0.0))
        for a, b in zip(vs, vs[1:] + vs[:1]):
            pa, pb = pos[a], pos[b]
            n.x += (pa.y - pb.y) * (pa.z + pb.z)
            n.y += (pa.z - pb.z) * (pa.x + pb.x)
            n.z += (pa.x - pb.x) * (pa.y + pb.y)
        out.append(n)
    return out


def _flipped(before, after, faces=None):
    """Indices of faces whose normal turned over (dot below 0) from before to after; a degenerate face is not counted."""
    rng = range(len(before)) if faces is None else faces
    return sorted(i for i in rng if before[i].length > 1e-12 and after[i].length > 1e-12 and before[i].dot(after[i]) < 0)


def parse_flipped(raw, obj, tail="NOTHING was measured."):
    """mesh_quality flippedAgainst -> the reference mesh, refused unless its faces are obj's faces vertex for vertex."""
    if not isinstance(raw, str) or not raw:
        raise MifOpError("flippedAgainst is the name of a reference copy of the mesh (the copy before a reshape), got "
                         "%r. %s" % (raw, tail))
    ref = get_object(raw, want_mesh=True)
    if ref is obj:
        raise MifOpError("flippedAgainst names the mesh itself - name the copy kept before the change. %s" % tail)
    a, b = obj.data.polygons, ref.data.polygons
    if len(a) != len(b) or len(obj.data.vertices) != len(ref.data.vertices) or any(
            tuple(pa.vertices) != tuple(pb.vertices) for pa, pb in zip(a, b)):
        raise MifOpError("'%s' and '%s' do not share their faces vertex for vertex (%d and %d faces) - a fold check "
                         "compares the same face before and after, so it needs a copy with the same topology. %s"
                         % (obj.name, ref.name, len(a), len(b), tail))
    return ref


def flipped_report(obj, ref):
    """Faces of obj whose world normal turned over against the same face of ref (both base meshes under their
    transforms)."""
    polys = [list(p.vertices) for p in obj.data.polygons]
    pos_o = [obj.matrix_world @ v.co for v in obj.data.vertices]
    pos_r = [ref.matrix_world @ v.co for v in ref.data.vertices]
    flips = _flipped(_face_normals(polys, pos_r), _face_normals(polys, pos_o))
    return {"reference": ref.name, "faces": len(polys), "flipped": len(flips),
            "share": round(len(flips) / len(polys), 6) if polys else 0.0, "flippedFaces": flips[:50],
            "flippedFacesTruncated": len(flips) > 50,
            "note": "a face is flipped when its normal points against the same face's normal in the reference - a fold."}


def _parse_axis(raw, tail):
    if not isinstance(raw, dict) or "point" not in raw or set(raw) - {"point", "direction"}:
        raise MifOpError("axis is {point: [x,y,z], direction: [x,y,z] (default [0,0,1])} in world units, got %r. %s"
                         % (raw, tail))
    pt, d = raw["point"], raw.get("direction", [0, 0, 1])
    ok = all(isinstance(x, (list, tuple)) and len(x) == 3 for x in (pt, d))
    pv = Vector(finite_floats(list(pt), "axis.point")) if ok else None
    dv = Vector(finite_floats(list(d), "axis.direction")) if ok else None
    if not ok or dv.length < 1e-9:
        raise MifOpError("axis.point and axis.direction are [x,y,z], the direction not zero, got %r. %s" % (raw, tail))
    return pv, dv.normalized()


def _reshape_moves(params, world, masked, mask, skin, tree):
    """{vertex: world move} before smoothing, and the counts (insideBefore, noBody, alreadyWithinTarget, gap before)."""
    mode, target, keep, reach = params["mode"], params["target"], params["keep"], params["reach"]
    apt, adir = params.get("axisPoint"), params.get("axisDirection")
    move, gap0, inside0, nobody, close = {}, {}, 0, 0, 0
    for i in masked:
        p = world[i]
        s = skin(p)
        gap0[i] = s
        if s is not None and s < 0:
            inside0 += 1
            continue
        if mode == "pull":
            loc, _n, _f, gap = tree.find_nearest(p, reach)
            if loc is None or gap < 1e-12:
                nobody += 1
                continue
            direc = (loc - p) / gap
        else:
            r = (p - apt) - adir * (p - apt).dot(adir)
            if r.length < 1e-12:
                nobody += 1
                continue
            direc = -r.normalized()
            loc, _n, _f, gap = tree.ray_cast(p, direc, r.length)
            if loc is None:
                nobody += 1
                continue
        if gap <= target:
            close += 1
            continue
        step = (gap - (target + keep * (gap - target))) * mask[i]
        if step > 0:
            move[i] = direc * step
    return move, gap0, inside0, nobody, close


def _smooth_moves(me, n, masked, mask, move, passes, gap0):
    """Each masked vertex's move toward its neighbours' mean by half, times its mask value; inside vertices stay put."""
    nbr = [[] for _ in range(n)]
    for e in me.edges:
        a, b = e.vertices
        nbr[a].append(b)
        nbr[b].append(a)
    zero = Vector((0.0, 0.0, 0.0))
    for _ in range(passes):
        nxt = {}
        for i in masked:
            if gap0[i] is None or gap0[i] < 0 or not nbr[i]:
                continue
            mine = move.get(i, zero)
            avg = sum((move.get(j, zero) for j in nbr[i]), Vector((0.0, 0.0, 0.0))) / len(nbr[i])
            v = mine + (avg - mine) * 0.5 * mask[i]
            if v.length > 1e-12:
                nxt[i] = v
        move = nxt
    return move


def _clearance_floor(world, move, gap0, target, skin):
    """Cut each move back (bisection on its length) so the vertex ends no closer to the skin than target, nor closer
    than it began. Returns ({vertex: final move}, clamped count)."""
    final, clamped = {}, 0
    for i, dv in move.items():
        floor = min(target, gap0[i]) if gap0[i] is not None else target
        s = skin(world[i] + dv)
        t = 1.0
        if s is None or s < floor - 1e-9:
            clamped += 1
            lo, hi = 0.0, 1.0
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                sm = skin(world[i] + dv * mid)
                if sm is not None and sm >= floor - 1e-9:
                    lo = mid
                else:
                    hi = mid
            t = lo
        if dv.length * t > 1e-9:
            final[i] = dv * t
    return final, clamped


def _reshape_params(params, tail):
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    body = get_object(take(params, "body", required=True, kind=str), want_mesh=True)
    if obj is body:
        raise MifOpError("object and body are the same mesh. %s" % tail)
    mode = (take(params, "mode", default="pull", kind=str) or "").lower()
    if mode not in ("pull", "squeeze"):
        raise MifOpError("mode is pull (toward the nearest skin) or squeeze (toward an axis, against the body), got %r. "
                         "%s" % (params.get("mode"), tail))
    cfg = {"mode": mode}
    if params.get("axis") is not None and mode == "pull":
        raise MifOpError("axis is for mode squeeze - a pull goes to the nearest skin. %s" % tail)
    if mode == "squeeze":
        if params.get("axis") is None:
            raise MifOpError("mode squeeze needs axis {point, direction} - the line the region is squeezed toward. %s"
                             % tail)
        cfg["axisPoint"], cfg["axisDirection"] = _parse_axis(params["axis"], tail)
    if params.get("mask") is None:
        raise MifOpError("mask is required - the region that moves and its falloff. %s" % tail)
    cfg["shapes"] = _mask_shapes(params.get("mask"))
    cfg["target"] = take_float(params, "target", default=0.005)
    cfg["keep"] = take_float(params, "keep", default=0.1)
    cfg["smooth"] = take_int(params, "smooth", default=0)
    cfg["reach"] = take_float(params, "reach", default=0.25)
    cfg["dry"] = take_bool(params, "dryRun", default=False)
    if not (math.isfinite(cfg["target"]) and cfg["target"] >= 0):
        raise MifOpError("target must be 0 or more metres, got %r. %s" % (cfg["target"], tail))
    if not (math.isfinite(cfg["keep"]) and 0 <= cfg["keep"] <= 1):
        raise MifOpError("keep must be 0-1 (the share of the stand-off above target kept), got %r. %s"
                         % (cfg["keep"], tail))
    if not 0 <= cfg["smooth"] <= 200:
        raise MifOpError("smooth must be 0-200 passes, got %d. %s" % (cfg["smooth"], tail))
    if not (math.isfinite(cfg["reach"]) and cfg["reach"] > 0):
        raise MifOpError("reach must be above 0 metres, got %r. %s" % (cfg["reach"], tail))
    if obj.data.shape_keys is not None:
        raise MifOpError("'%s' has shape keys - moving its base vertices would shift every key against the basis. Apply "
                         "or remove them first. %s" % (obj.name, tail))
    return obj, body, cfg


def op_garment_reshape(params):
    """Pull a region of a garment toward a body's surface with a falloff, or squeeze it toward an axis against the body.

    THE GAP: a worn-up hood stood 3-5 cm off the head. The fix, in raw Python: keep only 10% of the stand-off above a
    target (so the folds survive), fade it with a soft mask, smooth the moves. Below the jaw the nearest skin is the
    shoulder, so the same pull went DOWN there; that band needed a horizontal squeeze toward the neck axis against the
    body. Each reshape then needed a clearance floor from the skin and a fold check.

    mode pull (default): each masked vertex's gap is its distance to the nearest point of the body (within reach,
    default 0.25 m); it moves toward that point. mode squeeze: the gap is along a ray from the vertex toward `axis`
    {point, direction (default [0,0,1])}, at right angles to it, to the body; it moves along that ray. Either way the
    new gap is target + keep x (gap - target) (target default 0.005 m, keep default 0.1), the move scaled by the mask
    value: one shape or a list of {box | sphere | band {zFrom, zTo}, falloff} or {vertexGroup}, as blend_weights takes
    it. `smooth` passes (default 0) move each masked vertex's move halfway to its neighbours' mean, times its mask value.
    THE CLEARANCE FLOOR: a vertex never ends closer to the skin (nearest point; negative inside) than target, nor closer
    than it began - a move that would is cut back. A vertex inside the body or within target of it does not move.

    The garment's REST mesh (base vertices, world space) is moved against the body as it evaluates now. Reply:
    verticesMasked, moved, move stats, insideBefore / insideAfter (masked vertices under the skin), noBody, clamped (cut
    back by the floor), flipped and flippedFaces (faces whose normal turned over - a fold), gap stats before and after.
    A lining against its outer shell is pose_coverage (body = lining). dryRun reports and writes nothing. Refused: an
    unknown mode, squeeze without axis or pull with one, keep outside 0-1, target below 0, no vertex under the mask,
    object = body, a body with no faces, a garment with shape keys.
    """
    tail = "NOTHING was changed."
    reject_unknown(params, _RESHAPE_KEYS, "garment_reshape")
    obj, body, cfg = _reshape_params(params, tail)
    if not cfg["dry"]:
        require_editable(obj, "reshape")   # before anything is computed, so a linked mesh is refused first
    me = obj.data
    mw = obj.matrix_world
    world = [mw @ v.co for v in me.vertices]
    mask = [max(fn(world[i], i, obj) for fn in cfg["shapes"]) for i in range(len(world))]
    masked = [i for i in range(len(world)) if mask[i] > 0.0]
    if not masked:
        raise MifOpError("no vertex of '%s' is under the mask - a reshape of nothing is not clean. %s" % (obj.name, tail))
    from mathutils.bvhtree import BVHTree
    bbm = evaluated_world(body, True)
    try:
        if not len(bbm.faces):
            raise MifOpError("'%s' has no faces to reshape against. %s" % (body.name, tail))
        tree = BVHTree.FromBMesh(bbm)
    finally:
        bbm.free()
    far = max(cfg["reach"], 1e3)

    def skin(p):
        loc, nrm, _i, d = tree.find_nearest(p, far)
        if loc is None:
            return None
        return d if nrm.dot(p - loc) >= 0 else -d

    move, gap0, inside0, nobody, close = _reshape_moves(cfg, world, masked, mask, skin, tree)
    if cfg["smooth"] and move:
        move = _smooth_moves(me, len(world), masked, mask, move, cfg["smooth"], gap0)
    final, clamped = _clearance_floor(world, move, gap0, cfg["target"], skin)
    new = list(world)
    for i, dv in final.items():
        new[i] = world[i] + dv
    polys = [list(p.vertices) for p in me.polygons]
    touched = [fi for fi, vs in enumerate(polys) if any(v in final for v in vs)]
    flips = _flipped(_face_normals(polys, world), _face_normals(polys, new), touched) if touched else []
    gap_after = {i: skin(new[i]) for i in masked}
    out = {"object": obj.name, "body": body.name, "mode": cfg["mode"], "target": cfg["target"], "keep": cfg["keep"],
           "smooth": cfg["smooth"], "verticesMasked": len(masked),
           "verticesFullyMasked": sum(1 for i in masked if mask[i] >= 1.0), "moved": len(final),
           "movedVertices": sorted(final)[:200], "move": stats([dv.length for dv in final.values()], [50, 95], 6),
           "insideBefore": inside0, "insideAfter": sum(1 for i in masked if (gap_after[i] or 0.0) < 0),
           "noBody": nobody, "alreadyWithinTarget": close, "clamped": clamped, "flipped": len(flips),
           "flippedFaces": flips[:50], "gapBefore": stats([gap0[i] for i in final], [50], 6),
           "gapAfter": stats([gap_after[i] for i in final if gap_after[i] is not None], [50], 6), "dryRun": cfg["dry"]}
    if cfg["mode"] == "squeeze":
        out["axis"] = {"point": [round(c, 6) for c in cfg["axisPoint"]],
                       "direction": [round(c, 6) for c in cfg["axisDirection"]]}
    if not final:
        out["note"] = "no masked vertex stands farther than target from the body where it was measured - nothing moved."
    if cfg["dry"] or not final:
        return out
    mwi = mw.inverted()
    for i in final:
        me.vertices[i].co = mwi @ new[i]
    me.update()
    return out


# --------------------------------------------------------------------------------------------- building from a rim

def _world_edit(obj, op_name, writes=True):
    """(bmesh of obj's base mesh in world space less its origin, the shift) for an op that writes. World lengths and
    rays without carrying a far-out mesh's float32 coordinates out and back (ops_mesh._shifted). _write_world puts it
    back. Shape keys, linked data and a mirrored or singular transform are refused first."""
    from .ops_mesh import _no_shape_keys, _world_bmesh, _world_matrix_or_refuse   # noqa: PLC0415
    if writes:
        require_editable(obj, "edit")
        _no_shape_keys(obj, op_name)
    mw = _world_matrix_or_refuse(obj, op_name)
    shift = mw.translation.copy()
    bm = _world_bmesh(obj, False, shift)
    bm.edges.ensure_lookup_table()
    return bm, shift


def _write_world(bm, obj, shift):
    from .ops_mesh import _shifted   # noqa: PLC0415
    from .ops_topology import _commit   # noqa: PLC0415
    bm.transform(_shifted(obj.matrix_world, shift).inverted())
    bm.normal_update()
    _commit(bm, obj)


def _pick_rim(bm, obj, spec, shift, key="loop"):
    """An ordered open boundary loop [verts] by _pick_loop's grammar ({near} in world units)."""
    from .ops_topology import _boundary_loops_of, _ordered, _pick_loop   # noqa: PLC0415
    loops = _boundary_loops_of(bm)
    if not loops:
        raise MifOpError("'%s' has no open boundary - nothing to grow from. NOTHING was changed." % obj.name)
    if spec is None:
        if len(loops) != 1:
            raise MifOpError("'%s' has %d open boundary loops; name one with %s {near: [x,y,z]} (world), {vertex}, "
                             "{edge} or {index}. NOTHING was changed." % (obj.name, len(loops), key))
        lp = loops[0]
    else:
        if isinstance(spec, dict) and "near" in spec:
            spec = {"near": list(Vector(finite_floats(list(spec["near"]), key + ".near")) - shift)} \
                if isinstance(spec["near"], (list, tuple)) and len(spec["near"]) == 3 else spec
        # _pick_loop reads {near} through the object's inverse matrix; this bmesh is already in shifted world space
        lp = _pick_loop(bm, _IdentityObj(obj.name), loops, spec, key)
    order = _ordered(lp)
    if order is None or len(order) < 3:
        raise MifOpError("the %s picked is not a clean ring (a bow-tie vertex joins it to another, or it has under three "
                         "vertices). NOTHING was changed." % key)
    return order


def _pick_loops(bm):
    from .ops_topology import _boundary_loops_of   # noqa: PLC0415
    return _boundary_loops_of(bm)


class _IdentityObj:
    """Stands in for an object whose mesh is already in the space the caller's points are in."""
    def __init__(self, name):
        from mathutils import Matrix   # noqa: PLC0415
        self.name = name
        self.matrix_world = Matrix.Identity(4)


def _newell(pts):
    n = Vector()
    for i, p in enumerate(pts):
        q = pts[(i + 1) % len(pts)]
        n += Vector(((p.y - q.y) * (p.z + q.z), (p.z - q.z) * (p.x + q.x), (p.x - q.x) * (p.y + q.y)))
    return n


def _outward_axis(order):
    """The loop's own normal (Newell), turned to point away from the faces the rim belongs to - the way the surface
    goes on past the cut."""
    pts = [v.co for v in order]
    n = _newell(pts)
    if n.length < 1e-12:
        raise MifOpError("the loop encloses no area along any axis, so it has no direction of its own - pass axis or "
                         "bone. NOTHING was changed.")
    n.normalize()
    c = sum(pts, Vector()) / len(pts)
    side = sum(((v.co - f.calc_center_median()).dot(n) for v in order for f in v.link_faces), 0.0)
    return (n if side >= 0 else -n), c


def _bone_line(raw, shift, key="bone"):
    """{armature, bone} -> (head, unit direction head to tail) in shifted world space, from the pose."""
    if not isinstance(raw, dict) or set(raw) != {"armature", "bone"}:
        raise MifOpError("%s is {armature, bone}, got %r. NOTHING was changed." % (key, raw))
    arm = get_object(str(raw["armature"]))
    if arm.type != "ARMATURE":
        raise MifOpError("'%s' is a %s, not an ARMATURE. NOTHING was changed." % (arm.name, arm.type))
    pb = arm.pose.bones.get(str(raw["bone"]))
    if pb is None:
        raise MifOpError("'%s' has no bone '%s'. NOTHING was changed." % (arm.name, raw["bone"]))
    head, tail = arm.matrix_world @ pb.head, arm.matrix_world @ pb.tail
    if (tail - head).length < 1e-9:
        raise MifOpError("bone '%s' has no length. NOTHING was changed." % pb.name)
    return head - shift, (tail - head).normalized()


def _winding(order):
    """Per rim edge i (order[i] -> order[i+1]): True when the face it bounds runs that way, so a face grown off it runs
    the other way and keeps the surface's winding."""
    out = []
    n = len(order)
    for i in range(n):
        a, b = order[i], order[(i + 1) % n]
        e = next(x for x in a.link_edges if x.other_vert(a) is b)
        f = e.link_faces[0]
        out.append(any(lp.vert is a and lp.link_loop_next.vert is b for lp in f.loops))
    return out


def _band(bm, rings, fwd, slot_of=None):
    """Quads between consecutive rings of equal length, wound by fwd; slot_of(k) -> material index or None."""
    made = []
    n = len(rings[0])
    for k in range(1, len(rings)):
        a_ring, b_ring = rings[k - 1], rings[k]
        for i in range(n):
            j = (i + 1) % n
            quad = ((a_ring[j], a_ring[i], b_ring[i], b_ring[j]) if fwd[i]
                    else (a_ring[i], a_ring[j], b_ring[j], b_ring[i]))
            f = bm.faces.new(quad)
            m = slot_of(k) if slot_of else None
            if m is not None:
                f.material_index = m
            made.append(f)
    return made


def _band_uvs(bm, rings, made):
    """UVs for a band grown off a rim: each column carries on from the face across its rim edge, away from that face, at
    that face's own UV size. New faces had every corner at (0, 0), which gameReady fails and a texture cannot use. A rim
    edge with no face across it starts from the band's own length. Nothing on a mesh with no UV layer."""
    layer = bm.loops.layers.uv.active
    if layer is None:
        return
    new = set(made)
    n = len(rings[0])
    for i in range(n):
        j = (i + 1) % n
        a, b = rings[0][i], rings[0][j]
        edge = bm.edges.get((a, b))
        old = next((f for f in edge.link_faces if f not in new), None) if edge is not None else None
        length = max((a.co - b.co).length, 1e-9)
        if old is not None:
            at = {lp.vert: lp[layer].uv.copy() for lp in old.loops}
            ua, ub = at[a], at[b]
            mid = sum(at.values(), type(ua)((0.0, 0.0))) / len(at)
        else:
            ua = type(next(iter(bm.faces)).loops[0][layer].uv)((i * length, 0.0))
            ub = type(ua)(((i + 1) * length, 0.0))
            mid = type(ua)(((i + 0.5) * length, -1.0))
        along = ub - ua
        size = along.length / length
        if size < 1e-9:
            along, size = type(ua)((length, 0.0)), 1.0
        away = type(ua)((-along.y, along.x)).normalized()
        if away.dot((ua + ub) / 2.0 - mid) < 0.0:
            away = -away
        da = db = 0.0
        for k in range(1, len(rings)):
            f = made[(k - 1) * n + i]
            p, q = rings[k - 1], rings[k]
            step_a, step_b = (p[i].co - q[i].co).length, (p[j].co - q[j].co).length
            spots = {p[i]: ua + away * (da * size), p[j]: ub + away * (db * size),
                     q[i]: ua + away * ((da + step_a) * size), q[j]: ub + away * ((db + step_b) * size)}
            for lp in f.loops:
                lp[layer].uv = spots[lp.vert]
            da, db = da + step_a, db + step_b


def _slot_index(obj, name, tail="NOTHING was changed."):
    """(index or None, append?) for a material name as a slot of obj; refused when no such material exists."""
    if name is None:
        return None, False
    if name not in bpy.data.materials:
        raise MifOpError("no material named %r for slot. %s" % (name, tail))
    mats = [m.name if m else None for m in obj.data.materials]
    if name in mats:
        return mats.index(name), False
    # a mesh with no slot draws every face from slot 0, so the new one goes in at 1 behind an empty slot 0
    return max(1, len(mats)), True


def _append_slot(obj, name):
    if not len(obj.data.materials):
        obj.data.materials.append(None)
    obj.data.materials.append(bpy.data.materials[name])


def _smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _ring_length(pts):
    return sum((pts[(i + 1) % len(pts)] - pts[i]).length for i in range(len(pts)))


_RIB_KEYS = ("object", "loop", "variant", "axis", "bone", "length", "rings", "radius", "under", "ease", "blend", "slot",
             "slotLength", "tiers", "scale", "fullness", "waves", "separateObjects")
_RIB_ONLY = ("radius", "under", "ease", "blend", "slotLength")
_FRILL_ONLY = ("tiers", "scale", "fullness", "waves", "separateObjects")


def op_build_rib(params):
    """Grow a band off an open boundary loop: a rib (cuff, hem band) to a target radius, or a frill in tiers.

    THE GAP (spec "Build a rib piece from a boundary loop, and its frill and gather variants", 2026-10-04): cuffs and a
    hem band were built in raw Python as rings from a cut edge along a bone (or straight down) to a radius taken from
    rays to the body plus ease; frills were not built at all. gather_loop is the third variant.

    loop picks the rim ({near: [x,y,z]} world, {vertex}, {edge} or {index}; may be left out when the mesh has one).
    The band runs along a line: bone {armature, bone} (the posed bone's head-to-tail line - the rings are centred on it),
    or axis [x,y,z] through the loop's centroid, or by default the loop's own normal turned away from the faces it
    bounds. Each rim vertex keeps its angle and its height along the line; ring k sits length x k / rings further on.
    WORLD units; the new faces wind like the surface they continue.

    variant RIB (default): each vertex's radius goes from the rim's own toward a target - radius (a number), or under
    [meshes]: a ray from the line out through the vertex to the nearest of their evaluated surfaces, plus ease (default
    0.005) - over blend (default length) with a smoothstep; no target keeps the rim's radius (a straight band). A ray
    that misses keeps the rim's radius and is counted. slot (a material) goes on the last slotLength (default all).
    variant FRILL: tiers (default 1) bands of length each, tier k+1 grown off tier k's hem; the hem's radius is the
    start's x scale (default 1.5), and fullness (hem length / start length, default the scale's own) is reached by
    `waves` folds (default 8) whose amplitude is solved for - at least 4 rim vertices per fold. separateObjects puts
    each tier in its own object (<object>_Frill<k>, its top ring a copy - join_objects and merge_vertices join them).

    Reply: rings with their radius stats (RIB), tiers with measured fullness and fold amplitude (FRILL), rimNear (a
    point on the new open hem, for fold_boundary_under loopNear), new face and vertex counts. Refused before any write:
    no or an ambiguous loop, a rim vertex on the line, length or rings out of range, radius and under together, an
    unreachable fullness, too few vertices for the folds, the other variant's keys.
    """
    reject_unknown(params, _RIB_KEYS, "build_rib")
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    variant = str(take(params, "variant", default="RIB", kind=str)).upper()
    if variant not in ("RIB", "FRILL"):
        raise MifOpError("variant must be RIB or FRILL (gather_loop pulls a rim in), got %r. NOTHING was changed."
                         % variant)
    wrong = [k for k in (_FRILL_ONLY if variant == "RIB" else _RIB_ONLY) if params.get(k) is not None]
    if wrong:
        raise MifOpError("%s belong to variant %s. NOTHING was changed." % (", ".join(wrong),
                                                                             "FRILL" if variant == "RIB" else "RIB"))
    if params.get("axis") is not None and params.get("bone") is not None:
        raise MifOpError("pass axis OR bone - each sets the line the band runs along. NOTHING was changed.")
    length = take_float(params, "length", required=True)
    if not (math.isfinite(length) and 0 < length <= 100):
        raise MifOpError("length must be above 0 and at most 100 (world units), got %r. NOTHING was changed." % length)
    rings = take_int(params, "rings", default=4)
    if not 1 <= rings <= 200:
        raise MifOpError("rings must be 1-200, got %d. NOTHING was changed." % rings)
    slot_name = take(params, "slot", default=None, kind=str)
    radius = take_float(params, "radius", default=None)
    under_raw = params.get("under")
    ease = take_float(params, "ease", default=0.005)
    blend = take_float(params, "blend", default=length)
    slot_len = take_float(params, "slotLength", default=length)
    tiers = take_int(params, "tiers", default=1)
    scale = take_float(params, "scale", default=1.5)
    fullness = take_float(params, "fullness", default=None)
    waves = take_int(params, "waves", default=8)
    separate = take_bool(params, "separateObjects", default=False)
    if radius is not None and under_raw is not None:
        raise MifOpError("pass radius OR under - each sets the target radius. NOTHING was changed.")
    if radius is not None and not (math.isfinite(radius) and radius > 0):
        raise MifOpError("radius must be above 0, got %r. NOTHING was changed." % radius)
    if not (math.isfinite(ease) and ease >= 0) or not (math.isfinite(blend) and blend > 0):
        raise MifOpError("ease must be 0 or more and blend above 0, got %r and %r. NOTHING was changed." % (ease, blend))
    if not (math.isfinite(slot_len) and slot_len > 0):
        raise MifOpError("slotLength must be above 0, got %r. NOTHING was changed." % slot_len)
    if not 1 <= tiers <= 10:
        raise MifOpError("tiers must be 1-10, got %d. NOTHING was changed." % tiers)
    if not (math.isfinite(scale) and 0.1 <= scale <= 10):
        raise MifOpError("scale must be 0.1-10, got %r. NOTHING was changed." % scale)
    if not 1 <= waves <= 500:
        raise MifOpError("waves must be 1-500, got %d. NOTHING was changed." % waves)
    if fullness is not None and not (math.isfinite(fullness) and fullness > 0):
        raise MifOpError("fullness must be above 0, got %r. NOTHING was changed." % fullness)
    unders = []
    if under_raw is not None:
        names = [under_raw] if isinstance(under_raw, str) else under_raw
        if not isinstance(names, (list, tuple)) or not names or any(not isinstance(x, str) for x in names):
            raise MifOpError("under is a mesh name or a list of them, got %r. NOTHING was changed." % (under_raw,))
        unders = [get_object(x, want_mesh=True) for x in dict.fromkeys(names)]
        if obj in unders:
            raise MifOpError("under names the mesh itself. NOTHING was changed.")
    slot_idx, slot_new = _slot_index(obj, slot_name)
    bm, shift = _world_edit(obj, "build_rib")
    trees = []
    extra_objs = []
    try:
        order = _pick_rim(bm, obj, params.get("loop"), shift)
        fwd = _winding(order)
        if params.get("bone") is not None:
            origin, axis = _bone_line(params["bone"], shift)
            line = "bone"
        elif params.get("axis") is not None:
            a = params["axis"]
            if not isinstance(a, (list, tuple)) or len(a) != 3:
                raise MifOpError("axis must be [x,y,z], got %r. NOTHING was changed." % (a,))
            axis = Vector(finite_floats(list(a), "axis"))
            if axis.length < 1e-12:
                raise MifOpError("axis must not be zero. NOTHING was changed.")
            axis.normalize()
            origin = sum((v.co for v in order), Vector()) / len(order)
            line = "axis"
        else:
            axis, origin = _outward_axis(order)
            line = "loop normal"

        def frame(pts):
            rows = []
            for p in pts:
                h = (p - origin).dot(axis)
                r = p - origin - axis * h
                if r.length < 1e-6:
                    raise MifOpError("a rim vertex lies on the band's line, so it has no direction to grow out along - "
                                     "move the line (axis, bone). NOTHING was changed.")
                rows.append((h, r.normalized(), r.length))
            return rows

        start = [v.co.copy() for v in order]
        base = frame(start)
        n = len(order)
        out = {"object": obj.name, "variant": variant, "line": line, "axis": [round(x, 6) for x in axis],
               "loopVertices": n, "length": length}
        if variant == "RIB":
            from mathutils.bvhtree import BVHTree   # noqa: PLC0415
            from .ops_mesh import _world_bmesh   # noqa: PLC0415
            for u in unders:
                ubm = _world_bmesh(u, True, shift)
                try:
                    if not len(ubm.faces):
                        raise MifOpError("'%s' has no faces to measure to. NOTHING was changed." % u.name)
                    trees.append(BVHTree.FromBMesh(ubm))
                finally:
                    ubm.free()
            misses = 0
            ring_pts, ring_rows = [], []
            for k in range(1, rings + 1):
                s = length * k / rings
                w = _smoothstep(s / blend)
                pts, radii = [], []
                for h, d, r0 in base:
                    c = origin + axis * (h + s)
                    target = r0
                    if radius is not None:
                        target = radius
                    elif trees:
                        hits = [t.ray_cast(c, d) for t in trees]
                        dists = [x[3] for x in hits if x[0] is not None]
                        if dists:
                            target = min(dists) + ease
                        else:
                            misses += 1
                    rr = r0 + (target - r0) * w
                    radii.append(rr)
                    pts.append(c + d * rr)
                ring_pts.append(pts)
                ring_rows.append({"ring": k, "distance": round(s, 6), "radius": stats(radii, [], 6)})
            vrings = [list(order)] + [[bm.verts.new(p) for p in pts] for pts in ring_pts]
            first_slot = length - slot_len
            faces = _band(bm, vrings, fwd, (lambda k: slot_idx if length * (k - 0.5) / rings >= first_slot - 1e-9
                                            else None) if slot_idx is not None else None)
            _band_uvs(bm, vrings, faces)
            out.update({"rings": ring_rows, "target": ("radius" if radius is not None else
                                                       "under" if trees else "rim radius"),
                        "ease": ease if trees else None, "blend": blend, "rayMisses": misses if trees else None,
                        "newFaces": len(faces), "newVertices": rings * n,
                        "rimNear": [round(x, 6) for x in vrings[-1][0].co + shift]})
        else:
            tier_rows = []
            cur = start
            cur_verts = list(order)
            new_faces = new_verts = 0
            if n < 4 * waves and fullness is not None:
                raise MifOpError("%d folds need at least %d rim vertices and the loop has %d - fewer waves, or "
                                 "subdivide_mesh the rim first. NOTHING was changed." % (waves, 4 * waves, n))
            for t_i in range(tiers):
                fr = frame(cur)
                arc = [0.0]
                for i in range(1, n):
                    arc.append(arc[-1] + (cur[i] - cur[i - 1]).length)
                total = arc[-1] + (cur[0] - cur[-1]).length
                theta = [2 * math.pi * a / total for a in arc]

                def ring(t, amp, fr=fr, theta=theta):
                    return [origin + axis * (h + length * t) + d * (r0 * (1 + (scale - 1) * t)
                                                                            + amp * t * math.sin(waves * th))
                            for (h, d, r0), th in zip(fr, theta)]
                plain = _ring_length(ring(1.0, 0.0)) / total
                amp = 0.0
                if fullness is not None:
                    if fullness < plain - 1e-9:
                        raise MifOpError("tier %d: fullness %g is under what scale %g gives alone (%.4f). NOTHING was "
                                         "changed." % (t_i + 1, fullness, scale, plain))
                    hi = min(r0 for _h, _d, r0 in fr) * scale * 0.95
                    if _ring_length(ring(1.0, hi)) / total < fullness:
                        raise MifOpError("tier %d: fullness %g is out of reach - %d folds that keep the hem off its line "
                                         "give at most %.4f. More waves or a larger scale. NOTHING was changed."
                                         % (t_i + 1, fullness, waves, _ring_length(ring(1.0, hi)) / total))
                    lo = 0.0
                    for _ in range(60):
                        mid = 0.5 * (lo + hi)
                        if _ring_length(ring(1.0, mid)) / total < fullness:
                            lo = mid
                        else:
                            hi = mid
                    amp = 0.5 * (lo + hi)
                pts_rings = [ring(k / rings, amp) for k in range(1, rings + 1)]
                if separate:
                    tbm = bmesh.new()
                    top = [tbm.verts.new(p) for p in cur]
                    vr = [top] + [[tbm.verts.new(p) for p in pts] for pts in pts_rings]
                    faces = _band(tbm, vr, fwd, (lambda k: 0) if slot_name else None)
                    tbm.normal_update()
                    me = bpy.data.meshes.new("%s_Frill%d" % (obj.name, t_i + 1))
                    tbm.transform(_translate(shift))
                    tbm.to_mesh(me)
                    tbm.free()
                    extra_objs.append((me, len(faces)))
                    hem = [p.copy() for p in pts_rings[-1]]
                else:
                    vr = [cur_verts] + [[bm.verts.new(p) for p in pts] for pts in pts_rings]
                    faces = _band(bm, vr, fwd, (lambda k: slot_idx) if slot_idx is not None else None)
                    _band_uvs(bm, vr, faces)
                    cur_verts = vr[-1]
                    hem = [v.co.copy() for v in cur_verts]
                new_faces += len(faces)
                new_verts += rings * n + (n if separate else 0)
                tier_rows.append({"tier": t_i + 1, "startLength": round(total, 6),
                                  "hemLength": round(_ring_length(hem), 6),
                                  "fullness": round(_ring_length(hem) / total, 6), "amplitude": round(amp, 6),
                                  "faces": len(faces)})
                cur = hem
            out.update({"tiers": tier_rows, "scale": scale, "waves": waves if fullness is not None else None,
                        "separateObjects": separate, "newFaces": new_faces, "newVertices": new_verts,
                        "rimNear": [round(x, 6) for x in cur[0] + shift]})
        if not separate:
            if slot_new:
                _append_slot(obj, slot_name)
            _write_world(bm, obj, shift)
    finally:
        bm.free()
    if extra_objs:
        made = []
        asked = []
        for me, _nf in extra_objs:
            asked.append(me.name)
            o = bpy.data.objects.new(me.name, me)
            if slot_name:
                me.materials.append(bpy.data.materials[slot_name])
            for c in obj.users_collection:
                c.objects.link(o)
            made.append(o.name)
        for row, nm, want in zip(out["tiers"], made, asked):
            row["object"] = nm
            row["requestedName"] = want
            row["nameWasSuffixed"] = nm != want
        out["objects"] = made
        bpy.context.view_layer.update()
    else:
        out["after"] = mesh_counts(obj)
        out.update(shared_data_note(obj))
    out["slot"] = slot_name
    return out


def _translate(v):
    from mathutils import Matrix   # noqa: PLC0415
    return Matrix.Translation(v)


_GATHER_KEYS = ("object", "loop", "strength", "group", "reach", "apply", "smoothIterations", "setCloth")


def op_gather_loop(params):
    """Gather an open boundary loop like elastic - a waist or an ankle: a shrink weight group falling off from the rim,
    the cloth set to shrink by it, and optionally the rim pulled in now with a smoothing pass.

    THE GAP (spec "Build a rib piece from a boundary loop, and its frill and gather variants", the gather variant): a
    gathered edge was pinned or shrunk vertex weights plus smoothing, written by hand. loop as build_rib's. group
    (default "Gather") is written over: 1 on the rim, falling linearly to 0 at reach (default 0.05, world units) measured
    along the mesh's edges. setCloth points the object's Cloth modifier's shrink group at it with shrink_min 0 and
    shrink_max strength - Blender scales a spring's rest length by 1 - shrink_min - (shrink_max - shrink_min) x weight,
    so a weight-1 rim rests at 1 - strength of its length. apply pulls each weighted vertex toward the loop's axis (its
    centroid along its normal) by strength x weight now, then smoothIterations (default 2) passes move the falloff
    vertices (not the rim) toward their neighbours' mean by half their weight.

    Reply: rim vertices, the group's vertex count and weight stats, the rim's length before and after (apply), and the
    cloth settings read back. Refused: strength outside 0-1 (exclusive), reach not above 0, smoothIterations without
    apply, setCloth on an object with no Cloth modifier, apply on a mesh with shape keys.
    """
    import heapq   # noqa: PLC0415
    reject_unknown(params, _GATHER_KEYS, "gather_loop")
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    strength = take_float(params, "strength", required=True)
    if not (math.isfinite(strength) and 0 < strength < 1):
        raise MifOpError("strength is the share of the rim's length taken in, above 0 and under 1, got %r. NOTHING was "
                         "changed." % strength)
    group = str(take(params, "group", default="Gather", kind=str))
    if not group.strip():
        raise MifOpError("group must be a non-empty name. NOTHING was changed.")
    reach = take_float(params, "reach", default=0.05)
    if not (math.isfinite(reach) and reach > 0):
        raise MifOpError("reach must be above 0 (world units), got %r. NOTHING was changed." % reach)
    apply = take_bool(params, "apply", default=False)
    if params.get("smoothIterations") is not None and not apply:
        raise MifOpError("smoothIterations smooths the pull apply makes; pass apply true. NOTHING was changed.")
    smooth = take_int(params, "smoothIterations", default=2)
    if not 0 <= smooth <= 100:
        raise MifOpError("smoothIterations must be 0-100, got %d. NOTHING was changed." % smooth)
    set_cloth = take_bool(params, "setCloth", default=False)
    cloth = next((m for m in obj.modifiers if m.type == "CLOTH"), None)
    if set_cloth and cloth is None:
        raise MifOpError("'%s' has no Cloth modifier to set the shrink group on. NOTHING was changed." % obj.name)
    require_editable(obj, "gather")
    bm, shift = _world_edit(obj, "gather_loop", writes=apply)
    try:
        order = _pick_rim(bm, obj, params.get("loop"), shift)
        dist = {v: 0.0 for v in order}
        heap = [(0.0, v.index, v) for v in order]
        heapq.heapify(heap)
        while heap:
            d, _i, v = heapq.heappop(heap)
            if d > dist.get(v, math.inf):
                continue
            for e in v.link_edges:
                w = e.other_vert(v)
                nd = d + e.calc_length()
                if nd < reach and nd < dist.get(w, math.inf):
                    dist[w] = nd
                    heapq.heappush(heap, (nd, w.index, w))
        weight = {v: 1.0 - d / reach for v, d in dist.items()}
        before = _ring_length([v.co for v in order])
        after = None
        if apply:
            axis, origin = _outward_axis(order)
            for v, w in weight.items():
                r = (v.co - origin) - axis * (v.co - origin).dot(axis)
                v.co = v.co - r * (strength * w)
            rim = set(order)
            soft = [v for v in weight if v not in rim and v.link_edges]
            for _ in range(smooth):
                nxt = {}
                for v in soft:
                    nb = [e.other_vert(v).co for e in v.link_edges]
                    mean = sum(nb, Vector()) / len(nb)
                    nxt[v] = v.co + (mean - v.co) * (0.5 * weight[v])
                for v, co in nxt.items():
                    v.co = co
            after = _ring_length([v.co for v in order])
            _write_world(bm, obj, shift)
        rows = {v.index: w for v, w in weight.items()}
        rim_n = len(order)
    finally:
        bm.free()
    vg = obj.vertex_groups.get(group)
    replaced = vg is not None
    if vg is None:
        vg = obj.vertex_groups.new(name=group)
    else:
        vg.remove(list(range(len(obj.data.vertices))))
    for i, w in rows.items():
        vg.add([i], w, "REPLACE")
    out = {"object": obj.name, "loopVertices": rim_n, "group": vg.name, "groupReplaced": replaced,
           "groupVertices": len(rows), "weights": stats(list(rows.values()), [], 4), "reach": reach,
           "strength": strength, "applied": apply, "rimLengthBefore": round(before, 6),
           "rimLengthAfter": round(after, 6) if after is not None else None,
           "rimRatio": round(after / before, 6) if after is not None else None,
           "smoothIterations": smooth if apply else None, "cloth": None}
    if set_cloth:
        st = cloth.settings
        st.vertex_group_shrink = vg.name
        st.shrink_min = 0.0
        st.shrink_max = strength
        out["cloth"] = {"modifier": cloth.name, "vertexGroupShrink": st.vertex_group_shrink,
                        "shrinkMin": round(st.shrink_min, 6), "shrinkMax": round(st.shrink_max, 6),
                        "rimRestScale": round(1.0 - st.shrink_min - (st.shrink_max - st.shrink_min), 6)}
    obj.data.update()
    if apply:
        out["after"] = mesh_counts(obj)
        out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- trim to an outline

_SNAP_PASSES = 30
_TRIM_KEYS = ("object", "center", "bone", "axis", "front", "radii", "keep", "relax")


def _polar_radius(raw):
    """radii -> (R(theta degrees), the keys): {front, side, back} (side both sides) or [[angleDeg, radius], ...] (3 or
    more), joined by a periodic cubic Hermite through the keys with finite-difference tangents."""
    if isinstance(raw, dict):
        if set(raw) != {"front", "side", "back"}:
            raise MifOpError("radii is {front, side, back} or [[angleDeg, radius], ...], got %r. NOTHING was changed."
                             % (raw,))
        f, s, b = (finite_float(raw[k], "radii." + k) for k in ("front", "side", "back"))
        keys = [(0.0, f), (90.0, s), (180.0, b), (270.0, s)]
    elif isinstance(raw, (list, tuple)) and len(raw) >= 3 and all(isinstance(k, (list, tuple)) and len(k) == 2
                                                                    for k in raw):
        keys = sorted(((finite_float(a, "radii angle") % 360.0, finite_float(r, "radii radius")) for a, r in raw))
        if any(keys[i + 1][0] - keys[i][0] < 1e-6 for i in range(len(keys) - 1)):
            raise MifOpError("radii repeats an angle. NOTHING was changed.")
    else:
        raise MifOpError("radii is {front, side, back} or [[angleDeg, radius], ...] with 3 or more keys, got %r. "
                         "NOTHING was changed." % (raw,))
    if any(not r > 0 for _a, r in keys):
        raise MifOpError("every radius must be above 0, got %r. NOTHING was changed." % (raw,))
    n = len(keys)
    ang = [a for a, _r in keys]
    rad = [r for _a, r in keys]

    def at(k):
        q, w = divmod(k, n)
        return ang[w] + 360.0 * q, rad[w]

    tan = []
    for k in range(n):
        a0, r0 = at(k - 1)
        a1, r1 = at(k + 1)
        tan.append((r1 - r0) / (a1 - a0))

    def radius(theta):
        th = theta % 360.0
        k = n - 1
        for i in range(n):
            if ang[i] > th:
                k = i - 1
                break
        a0, r0 = at(k)
        a1, r1 = at(k + 1)
        if th < a0:
            th += 360.0
        h = a1 - a0
        t = (th - a0) / h
        t2, t3 = t * t, t * t * t
        return ((2 * t3 - 3 * t2 + 1) * r0 + (t3 - 2 * t2 + t) * h * tan[k % n] + (-2 * t3 + 3 * t2) * r1
                + (t3 - t2) * h * tan[(k + 1) % n])
    low = min(radius(i * 0.5) for i in range(720))
    if not low > 0:
        raise MifOpError("the outline through these radii dips to %g between keys - space them more evenly. NOTHING "
                         "was changed." % low)
    return radius, [[round(a, 6), round(r, 6)] for a, r in keys]


def op_trim_to_outline(params):
    """Trim a sheet to a smooth outline - a radius by angle round an axis (front, side and back radii) - with the new
    edge cut along the curve, snapped onto it and relaxed on the original surface.

    THE GAP (spec "Rebuild a sheet's edge to a smooth outline", 2026-10-04): a hood's skirt was trimmed to a radius
    R(theta) round the neck bone and its new edge snapped onto that curve and relaxed, in raw Python. The axis is bone
    {armature, bone} (through the posed head along the bone) or center [x,y,z] and axis (default +Z), world units. The
    angle is measured round the axis from front (default -Y, laid square to the axis); radii is {front, side, back}
    (side at 90 degrees either way) or [[angleDeg, radius], ...], joined by a smooth periodic curve through the keys.
    A point's distance from the axis line is compared with R at its angle, at any height: keep INSIDE (default) keeps
    what lies within, OUTSIDE what lies beyond.

    Every edge that crosses the outline is split where it crosses, the faces across it are split between those
    points (a quad cut across a corner keeps a five-sided remainder), and the faces beyond are deleted. relax
    (default 10) passes then move each new rim vertex halfway to its two rim neighbours, back onto the original
    surface and out to the curve at its angle; 30 passes then snap without the move (measured on 5.2.2: on a
    32-segment sphere 4 left the rim 0.5 mm off the surface, 30 leave it under 0.1 micrometre). Reply: faces
    removed, rim vertices, curveError (the largest |distance - R|), surfaceDistance (the largest distance from the
    original surface), boundary loops before and after, the keys used. Refused: nothing inside or nothing beyond the
    outline, front along the axis, a radius at or under 0, shape keys.
    """
    from mathutils.bvhtree import BVHTree   # noqa: PLC0415
    reject_unknown(params, _TRIM_KEYS, "trim_to_outline")
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    if params.get("radii") is None:
        raise MifOpError("radii is required - {front, side, back} or [[angleDeg, radius], ...]. NOTHING was changed.")
    radius, keys = _polar_radius(params["radii"])
    keep = str(take(params, "keep", default="INSIDE", kind=str)).upper()
    if keep not in ("INSIDE", "OUTSIDE"):
        raise MifOpError("keep must be INSIDE or OUTSIDE, got %r. NOTHING was changed." % keep)
    relax = take_int(params, "relax", default=10)
    if not 0 <= relax <= 200:
        raise MifOpError("relax must be 0-200, got %d. NOTHING was changed." % relax)
    if params.get("bone") is not None and (params.get("center") is not None or params.get("axis") is not None):
        raise MifOpError("pass bone OR center and axis - each sets the axis. NOTHING was changed.")
    vecs = {}
    for k, dflt in (("center", None), ("axis", (0.0, 0.0, 1.0)), ("front", (0.0, -1.0, 0.0))):
        raw = params.get(k, dflt)
        if raw is None:
            continue
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise MifOpError("%s must be [x,y,z], got %r. NOTHING was changed." % (k, raw))
        vecs[k] = Vector(finite_floats(list(raw), k))
    if params.get("bone") is None and "center" not in vecs:
        raise MifOpError("center [x,y,z] or bone {armature, bone} is required. NOTHING was changed.")
    bm, shift = _world_edit(obj, "trim_to_outline")
    try:
        if params.get("bone") is not None:
            origin, axis = _bone_line(params["bone"], shift)
        else:
            origin, axis = vecs["center"] - shift, vecs["axis"]
            if axis.length < 1e-12:
                raise MifOpError("axis must not be zero. NOTHING was changed.")
            axis = axis.normalized()
        fu = vecs["front"] - axis * vecs["front"].dot(axis)
        if fu.length < 1e-6:
            raise MifOpError("front lies along the axis, so it names no direction round it. NOTHING was changed.")
        fu.normalize()
        su = axis.cross(fu)
        sign = 1.0 if keep == "INSIDE" else -1.0

        def polar(p):
            d = p - origin
            r = d - axis * d.dot(axis)
            return r, math.degrees(math.atan2(r.dot(su), r.dot(fu)))

        def field(p):
            r, th = polar(p)
            return sign * (r.length - radius(th))

        if not len(bm.faces):
            raise MifOpError("'%s' has no faces to trim. NOTHING was changed." % obj.name)
        src = bm.copy()
        tree = BVHTree.FromBMesh(src)
        src.free()
        loops_before = len(_pick_loops(bm))
        tol = 1e-7
        val = {v: field(v.co) for v in bm.verts}
        if not any(x < -tol for x in val.values()):
            raise MifOpError("no vertex of '%s' lies %s the outline - the trim would delete it all. NOTHING was changed."
                             % (obj.name, "within" if keep == "INSIDE" else "beyond"))
        if all(x <= tol for x in val.values()):
            raise MifOpError("every vertex of '%s' lies %s the outline - nothing to trim. NOTHING was changed."
                             % (obj.name, "within" if keep == "INSIDE" else "beyond"))
        cut = {v for v, x in val.items() if abs(x) <= tol}
        for e in list(bm.edges):
            a, b = e.verts
            fa, fb = val[a], val[b]
            if (fa > tol and fb < -tol) or (fa < -tol and fb > tol):
                lo, hi, pa, pb = 0.0, 1.0, a.co.copy(), b.co.copy()
                for _ in range(48):
                    mid = 0.5 * (lo + hi)
                    if (field(pa.lerp(pb, mid)) > 0) == (fa > 0):
                        lo = mid
                    else:
                        hi = mid
                t = 0.5 * (lo + hi)
                _ne, nv = bmesh.utils.edge_split(e, a, t)
                nv.co = pa.lerp(pb, t)
                val[nv] = 0.0
                cut.add(nv)
        bmesh.ops.connect_verts(bm, verts=list(cut), check_degenerate=True)
        doomed = [f for f in bm.faces if any(val.get(v, 0.0) > tol for v in f.verts)]
        removed = len(doomed)
        bmesh.ops.delete(bm, geom=doomed, context="FACES")
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")
        rim = [v for v in cut if v.is_valid and v.is_boundary]
        if len(rim) < 3:
            raise MifOpError("the cut left %d rim vertices on the outline - the mesh is too coarse there. NOTHING was "
                             "changed." % len(rim))
        rim_set = set(rim)
        nbr = {v: [e.other_vert(v) for e in v.link_edges if e.is_boundary and e.other_vert(v) in rim_set] for v in rim}

        def place(v):
            loc = tree.find_nearest(v.co)[0]
            if loc is None:
                return
            d = loc - origin
            h = d.dot(axis)
            r, th = polar(loc)
            if r.length > 1e-12:
                v.co = origin + axis * h + r.normalized() * radius(th)

        for _ in range(relax):
            moved = {v: v.co * 0.5 + (nbr[v][0].co + nbr[v][1].co) * 0.25 for v in rim if len(nbr[v]) == 2}
            for v, co in moved.items():
                v.co = co
            for v in rim:
                place(v)
        for _ in range(_SNAP_PASSES):
            for v in rim:
                place(v)
        curve_err = max(abs(polar(v.co)[0].length - radius(polar(v.co)[1])) for v in rim)
        surf = max(tree.find_nearest(v.co)[3] for v in rim)
        loops_after = len(_pick_loops(bm))
        _write_world(bm, obj, shift)
        rim_n = len(rim)
    finally:
        bm.free()
    out = {"object": obj.name, "keep": keep, "radii": keys, "axis": [round(x, 6) for x in axis],
           "front": [round(x, 6) for x in fu], "facesRemoved": removed, "rimVertices": rim_n,
           "curveError": round(curve_err, 9), "surfaceDistance": round(surf, 9), "relax": relax,
           "boundaryLoopsBefore": loops_before, "boundaryLoopsAfter": loops_after, "after": mesh_counts(obj)}
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- two-sided by geometry

_INNER_KEYS = ("object", "partNear", "nearOpenings", "openings", "offset", "group", "slot", "faces", "side", "join",
               "flipNormals", "internalLines", "linesGroup", "uvOffset", "name")
_INNER_SIDES = ("INSIDE", "OUTSIDE")
_INNER_JOINS = ("NONE", "RIM", "OBJECT")


def _parse_face_spec(raw, obj, tail="NOTHING was changed."):
    """faces -> ("ALL", None) | ("GROUP", (index, minWeight)) | ("SLOT", index) | ("INDICES", set); None when absent."""
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw.upper() != "ALL":
            raise MifOpError("faces as a word must be ALL, got %r. %s" % (raw, tail))
        return ("ALL", None)
    if isinstance(raw, (list, tuple)):
        raw = {"indices": list(raw)}
    if not isinstance(raw, dict) or not raw or not set(raw) <= {"vertexGroup", "minWeight", "slot", "indices"} \
            or len(set(raw) & {"vertexGroup", "slot", "indices"}) != 1 or ("minWeight" in raw and "vertexGroup" not in raw):
        raise MifOpError("faces is ALL, {vertexGroup, minWeight}, {slot: name or index} or {indices: [...]}, got %r. %s"
                         % (raw, tail))
    if "vertexGroup" in raw:
        vg = obj.vertex_groups.get(str(raw["vertexGroup"]))
        if vg is None:
            raise MifOpError("faces.vertexGroup %r is not a group of '%s' (it has: %s). %s" % (
                raw["vertexGroup"], obj.name, ", ".join(sorted(g.name for g in obj.vertex_groups)) or "none", tail))
        mw = finite_float(raw.get("minWeight", 0.001), "faces.minWeight")
        if not 0 < mw <= 1:
            raise MifOpError("faces.minWeight must be above 0 and at most 1, got %r. %s" % (mw, tail))
        return ("GROUP", (vg.index, mw))
    if "slot" in raw:
        s = raw["slot"]
        mats = [m.name if m else None for m in obj.data.materials]
        if isinstance(s, int) and not isinstance(s, bool):
            if not 0 <= s < max(1, len(mats)):
                raise MifOpError("faces.slot %d is out of range: '%s' has %d slots. %s" % (s, obj.name, len(mats), tail))
            return ("SLOT", s)
        if s not in mats:
            raise MifOpError("faces.slot %r is not a slot of '%s' (it has: %s). %s" % (
                s, obj.name, ", ".join(str(m) for m in mats) or "none", tail))
        return ("SLOT", mats.index(s))
    idx = raw["indices"]
    n = len(obj.data.polygons)
    if not isinstance(idx, (list, tuple)) or not idx or any(
            isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n for i in idx):
        raise MifOpError("faces.indices must be a non-empty list of face indices of '%s' (it has %d). %s"
                         % (obj.name, n, tail))
    return ("INDICES", set(idx))


def _faces_by_spec(bm, spec):
    kind, arg = spec
    if kind == "ALL":
        return set(bm.faces)
    if kind == "SLOT":
        return {f for f in bm.faces if f.material_index == arg}
    if kind == "INDICES":
        return {f for f in bm.faces if f.index in arg}
    gi, mw = arg
    dl = bm.verts.layers.deform.active
    if dl is None:
        return set()
    return {f for f in bm.faces if all(v[dl].get(gi, 0.0) >= mw for v in f.verts)}


def _inner_options(params, obj):
    """The lining options, read and checked before anything is written."""
    side = str(take(params, "side", default="INSIDE", kind=str)).upper()
    if side not in _INNER_SIDES:
        raise MifOpError("side must be one of %s, got %r. NOTHING was changed." % (", ".join(_INNER_SIDES), side))
    join = str(take(params, "join", default="NONE", kind=str)).upper()
    if join not in _INNER_JOINS:
        raise MifOpError("join must be one of %s, got %r. NOTHING was changed." % (", ".join(_INNER_JOINS), join))
    flip = take_bool(params, "flipNormals", default=True)
    if join == "RIM" and (side != "INSIDE" or not flip):
        raise MifOpError("join RIM closes one shell whose inner wall is the copy - it needs side INSIDE and flipNormals "
                         "true (got side %s, flipNormals %s). NOTHING was changed." % (side, flip))
    lines = take_int(params, "internalLines", default=0)
    if not 0 <= lines <= 50:
        raise MifOpError("internalLines must be 0-50, got %r. NOTHING was changed." % lines)
    uvo = params.get("uvOffset")
    if uvo is not None:
        if not isinstance(uvo, (list, tuple)) or len(uvo) != 2:
            raise MifOpError("uvOffset must be [u, v], got %r. NOTHING was changed." % (uvo,))
        uvo = finite_floats(list(uvo), "uvOffset")
    name = take(params, "name", default=None, kind=str)
    if name is not None and join != "OBJECT":
        raise MifOpError("name is the new object's name for join OBJECT; join is %s. NOTHING was changed." % join)
    if join == "OBJECT":
        name = name or obj.name + "Lining"
        if name in bpy.data.objects:
            raise MifOpError("an object named '%s' already exists; pass another name. NOTHING was changed." % name)
    lines_group = take(params, "linesGroup", default=None, kind=str)
    if lines_group is not None and not lines:
        raise MifOpError("linesGroup names the group internalLines writes; pass internalLines. NOTHING was changed.")
    return side, join, flip, lines, uvo, name, lines_group


def _band_sides(bm, faces):
    """The two edges of a band (sets of vertices) for internalLines: its two rims when the faces make a tube, or the
    two long sides between four corners when they make a strip. None when the faces are neither."""
    fset = set(faces)
    part, stack = {faces[0]}, [faces[0]]
    while stack:
        for e in stack.pop().edges:
            for g in e.link_faces:
                if g in fset and g not in part:
                    part.add(g)
                    stack.append(g)
    if len(part) != len(fset):
        return None
    bedges = [e for f in faces for e in f.edges if sum(1 for g in e.link_faces if g in fset) == 1]
    adj = {}
    for e in set(bedges):
        for v in e.verts:
            adj.setdefault(v, []).append(e)
    if not adj or any(len(x) != 2 for x in adj.values()):
        return None
    loops, seen = [], set()
    for start in adj:
        if start in seen:
            continue
        order, prev, v = [start], None, start
        seen.add(start)
        while True:
            nxt = [e for e in adj[v] if e is not prev][0]
            w = nxt.other_vert(v)
            if w is start:
                break
            order.append(w)
            seen.add(w)
            prev, v = nxt, w
        loops.append(order)
    if len(loops) == 2:
        return set(loops[0]), set(loops[1])
    if len(loops) != 1:
        return None
    ring = loops[0]
    corners = [i for i, v in enumerate(ring) if sum(1 for g in v.link_faces if g in fset) == 1]
    if len(corners) != 4:
        return None
    runs = []
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        idx = list(range(a, b + 1)) if b > a else list(range(a, len(ring))) + list(range(0, b + 1))
        pts = [ring[i] for i in idx]
        runs.append((sum((pts[i + 1].co - pts[i].co).length for i in range(len(pts) - 1)), pts))
    if runs[0][0] + runs[2][0] >= runs[1][0] + runs[3][0]:
        return set(runs[0][1]), set(runs[2][1])
    return set(runs[1][1]), set(runs[3][1])


def _graph_distance(verts, sources):
    import heapq   # noqa: PLC0415
    vs = set(verts)
    dist = {v: 0.0 for v in sources}
    heap = [(0.0, k, v) for k, v in enumerate(sources)]
    heapq.heapify(heap)
    tick = len(heap)
    while heap:
        d, _k, v = heapq.heappop(heap)
        if d > dist.get(v, math.inf):
            continue
        for e in v.link_edges:
            w = e.other_vert(v)
            if w not in vs:
                continue
            nd = d + e.calc_length()
            if nd < dist.get(w, math.inf) - 1e-12:
                dist[w] = nd
                tick += 1
                heapq.heappush(heap, (nd, tick, w))
    return dist


def _cut_lines(bm, faces, sides, n):
    """n evenly spaced lines across a band between its two edges: the level sets of t = dA / (dA + dB) (graph distance
    along the band's edges from each side), cut into its faces. Returns [[verts of line k], ...]."""
    verts = {v for f in faces for v in f.verts}
    da = _graph_distance(verts, list(sides[0]))
    db = _graph_distance(verts, list(sides[1]))
    t = {v: da.get(v, math.inf) / max(1e-12, da.get(v, math.inf) + db.get(v, math.inf)) for v in verts}
    fset = set(faces)
    lines = []
    for k in range(1, n + 1):
        lv = k / (n + 1.0)
        on = {v for v in verts if abs(t[v] - lv) < 1e-6}
        cross = {e for f in fset for e in f.edges if (t[e.verts[0]] - lv) * (t[e.verts[1]] - lv) < 0
                 and abs(t[e.verts[0]] - lv) >= 1e-6 and abs(t[e.verts[1]] - lv) >= 1e-6}
        for e in cross:
            a, b = e.verts
            fac = (lv - t[a]) / (t[b] - t[a])
            _ne, nv = bmesh.utils.edge_split(e, a, fac)
            t[nv] = lv
            verts.add(nv)
            on.add(nv)
        touched = {f for v in on for f in v.link_faces if f in fset}
        for f in touched:
            fv = [v for v in f.verts if v in on]
            if len(fv) == 2 and not any(e.other_vert(fv[0]) is fv[1] for e in fv[0].link_edges):
                r = bmesh.ops.connect_verts(bm, verts=fv)
                for e in r["edges"]:
                    fset.update(e.link_faces)
        lines.append(sorted(on, key=lambda v: (v.co.x, v.co.y, v.co.z)))
    faces[:] = [f for f in bm.faces if f in fset]
    return lines


def _inner_choose(bm, obj, shift, part_pt, near_open, raw_open, spec):
    """(chosen faces, by part, by openings, by faces)."""
    from mathutils.bvhtree import BVHTree   # noqa: PLC0415
    from mathutils.kdtree import KDTree   # noqa: PLC0415
    chosen = set()
    by_part = by_open = by_spec = 0
    if spec is not None:
        picked = _faces_by_spec(bm, spec)
        by_spec = len(picked)
        chosen |= picked
    if part_pt is not None:
        tree = BVHTree.FromBMesh(bm)
        _loc, _n, fi, _d = tree.find_nearest(part_pt - shift)
        seed = bm.faces[fi]
        part, stack = {seed}, [seed]
        while stack:
            f = stack.pop()
            for e in f.edges:
                for g in e.link_faces:
                    if g not in part:
                        part.add(g)
                        stack.append(g)
        by_part = len(part)
        chosen |= part
    if near_open is not None:
        loops = _pick_loops(bm)
        if not loops:
            raise MifOpError("'%s' has no opening (open boundary) to measure from. NOTHING was changed." % obj.name)
        if raw_open is not None:
            from .ops_topology import _pick_loop   # noqa: PLC0415
            picked = []
            for k, ospec in enumerate(raw_open):
                if isinstance(ospec, dict) and isinstance(ospec.get("near"), (list, tuple)) and len(ospec["near"]) == 3:
                    ospec = {"near": list(Vector(finite_floats(list(ospec["near"]), "openings.near")) - shift)}
                lp = _pick_loop(bm, _IdentityObj(obj.name), loops, ospec, "openings[%d]" % k)
                if lp not in picked:
                    picked.append(lp)
            loops = picked
        rim = [v for _e, vs in loops for v in vs]
        kd = KDTree(len(rim))
        for i, v in enumerate(rim):
            kd.insert(v.co, i)
        kd.balance()
        opened = {f for f in bm.faces if kd.find(f.calc_center_median())[2] <= near_open}
        by_open = len(opened)
        chosen |= opened
    return chosen, by_part, by_open, by_spec


def _shift_uvs(bm, faces, uvo):
    if uvo is None or (uvo[0] == 0 and uvo[1] == 0):
        return
    for layer in bm.loops.layers.uv.values():
        for f in faces:
            for lo in f.loops:
                lo[layer].uv = (lo[layer].uv[0] + uvo[0], lo[layer].uv[1] + uvo[1])


def _rim_join(bm, faces, vmap):
    """Quads from each rim edge of the chosen faces to its copy, wound so the outer wall, the rim and the reversed copy
    are one consistently wound shell. UVs: each rim face takes its outer face's UVs at the rim edge. Returns (rim faces,
    rim edges that also bound an unchosen face - three faces meet there)."""
    fset = set(faces)
    layers = list(bm.loops.layers.uv.values())
    rim, shared = [], 0
    for f in faces:
        for lo in f.loops:
            e = lo.edge
            if sum(1 for g in e.link_faces if g in fset) != 1:
                continue
            if len(e.link_faces) > 1:
                shared += 1
            a, b = lo.vert, lo.link_loop_next.vert   # f walks a -> b
            nf = bm.faces.new((b, a, vmap[a], vmap[b]))
            nf.material_index = f.material_index
            nf.smooth = f.smooth
            for layer in layers:
                ua, ub = lo[layer].uv.copy(), lo.link_loop_next[layer].uv.copy()
                for nl, uv in zip(nf.loops, (ub, ua, ua, ub)):
                    nl[layer].uv = uv
            rim.append(nf)
    return rim, shared


def op_add_inner_layer(params):
    """Make a garment two-sided or lined: flipped, inset copies of chosen faces - a loose part, every face near an
    opening, or faces by group, slot or index - kept loose, joined to the outside at the rims, or a new object.

    THE GAP (spec "Two-sided by geometry, by region", 2026-10-04): a garment seen without its body (an asset viewer)
    lets the background through a sleeve, hem or collar - the far wall faces away and is culled. Fixed by hand with
    flipped duplicates of the collar and of every face within 12 cm of the cuffs and hem. partNear [x,y,z] (world)
    takes the loose part (faces joined by edges) nearest the point; nearOpenings (world units) takes every face whose
    centre is that close (straight-line) to a rim vertex of the openings - every open boundary loop, or those named by
    openings [{near}|{vertex}|{edge}|{index}, ...]. Either or both. The copies are moved offset (default 0.002) against
    each vertex's normal, wound the other way, put in vertex group `group` (default "InnerLayer") and slot (a material)
    when given.

    THE LINING (spec "Clone a lining or second layer under chosen panels, joined at the rims", 2026-10-06): a firm hood
    or collar is a panel set cloned inward and joined at the rims into one closed shell - the closed volume cloth
    pressure needs - and a stretch tape is a band cloned under itself, loose, with evenly spaced lines in it.
    faces: ALL | {vertexGroup, minWeight} | {slot} | {indices} chooses faces too (a union with the two above).
    side INSIDE (against the normal, default) | OUTSIDE (along it); flipNormals (default true) winds the copies the
    other way. join NONE (copies in this mesh, loose) | RIM (a quad from every rim edge of the chosen faces to its
    copy: one closed, consistently wound shell; needs INSIDE and flipNormals; a rim edge that also bounds an unchosen
    face gets a third face and is counted in rimEdgesShared) | OBJECT (the copies as a new object `name`, default
    <object>Lining). internalLines n cuts n evenly spaced lines into the copies between the band's two edges (its two
    rims, or the long sides of a four-cornered strip), in group linesGroup (default <group>Lines). uvOffset [u, v]
    moves the copies' UVs (one tile over: [1, 0]).

    A BAKE MUST DROP THEM FIRST: the copies share the outside's UVs unless uvOffset moves them.
    ONE LAYER PER GROUP: a group that already holds vertices is refused - the copies' own rims are openings too, and
    a second pass would copy the copies.

    Reply: faces chosen by part, openings and faces, faces and vertices added, rim faces, the group, the copies'
    distance from the originals (min, max), how many face exactly opposite their original, and the lines. Refused:
    no selector, nothing chosen, offset outside 0-0.1, a group in use, openings without nearOpenings, shape keys, a
    bad side/join/faces, RIM with OUTSIDE or unflipped, internalLines on faces that are not a band, a taken name.
    """
    reject_unknown(params, _INNER_KEYS, "add_inner_layer")
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    near_raw = params.get("partNear")
    near_open = take_float(params, "nearOpenings", default=None)
    spec = _parse_face_spec(params.get("faces"), obj)
    if near_raw is None and near_open is None and spec is None:
        raise MifOpError("pass faces (ALL, {vertexGroup}, {slot} or {indices}), partNear [x,y,z], nearOpenings (a "
                         "distance), or several. NOTHING was changed.")
    if params.get("openings") is not None and near_open is None:
        raise MifOpError("openings names which openings nearOpenings measures from; pass nearOpenings. NOTHING was "
                         "changed.")
    if near_open is not None and not (math.isfinite(near_open) and near_open > 0):
        raise MifOpError("nearOpenings must be above 0, got %r. NOTHING was changed." % near_open)
    part_pt = None
    if near_raw is not None:
        if not isinstance(near_raw, (list, tuple)) or len(near_raw) != 3:
            raise MifOpError("partNear must be [x,y,z], got %r. NOTHING was changed." % (near_raw,))
        part_pt = Vector(finite_floats(list(near_raw), "partNear"))
    offset = take_float(params, "offset", default=0.002)
    if not (math.isfinite(offset) and 0 < offset <= 0.1):
        raise MifOpError("offset must be above 0 and at most 0.1, got %r. NOTHING was changed." % offset)
    group = str(take(params, "group", default="InnerLayer", kind=str))
    if not group.strip():
        raise MifOpError("group must be a non-empty name. NOTHING was changed.")
    side, join, flip, n_lines, uvo, new_name, lines_group = _inner_options(params, obj)
    lines_group = lines_group or group + "Lines"
    old = obj.vertex_groups.get(group)
    if join != "OBJECT" and old is not None and any(ge.group == old.index for v in obj.data.vertices
                                                    for ge in v.groups):
        raise MifOpError("'%s' already holds vertices in group '%s' - an inner layer is there; delete its faces or name "
                         "another group. NOTHING was changed." % (obj.name, group))
    raw_open = params.get("openings")
    if raw_open is not None and (not isinstance(raw_open, (list, tuple)) or not raw_open):
        raise MifOpError("openings is a list of loops ({near}|{vertex}|{edge}|{index}), got %r. NOTHING was changed."
                         % (raw_open,))
    slot_name = take(params, "slot", default=None, kind=str)
    slot_idx, slot_new = _slot_index(obj, slot_name)
    bm, shift = _world_edit(obj, "add_inner_layer")
    target = obj
    try:
        if not len(bm.faces):
            raise MifOpError("'%s' has no faces. NOTHING was changed." % obj.name)
        chosen, by_part, by_open, by_spec = _inner_choose(bm, obj, shift, part_pt, near_open, raw_open, spec)
        if not chosen:
            raise MifOpError("the selectors chose no face of '%s'. NOTHING was changed." % obj.name)
        faces = [f for f in bm.faces if f in chosen]
        if n_lines and _band_sides(bm, faces) is None:
            raise MifOpError("internalLines needs the chosen faces to be a band - two rims (a tube) or a strip with four "
                             "corners; these %d faces are neither. NOTHING was changed." % len(faces))
        sign = -1.0 if side == "INSIDE" else 1.0
        normals = {v: v.normal.copy() for f in faces for v in f.verts}
        orig_n = {f: f.normal.copy() for f in faces}
        if join == "OBJECT":
            for f in [f for f in bm.faces if f not in chosen]:
                bm.faces.remove(f)
            for v in [v for v in bm.verts if not v.link_faces]:
                bm.verts.remove(v)
            vmap = {v: v for v in normals}
            fmap = {f: f for f in faces}
            orig_co = {v: v.co.copy() for v in normals}
        else:
            r = bmesh.ops.duplicate(bm, geom=faces)
            vmap, fmap = r["vert_map"], r["face_map"]
            orig_co = {v: v.co.copy() for v in normals}
        new_verts = [vmap[v] for v in normals]
        for v, n in normals.items():
            vmap[v].co = orig_co[v] + n * (sign * offset)
        new_faces = [fmap[f] for f in faces]
        if flip:
            bmesh.ops.reverse_faces(bm, faces=new_faces)
        if slot_idx is not None:
            for f in new_faces:
                f.material_index = slot_idx
        _shift_uvs(bm, new_faces, uvo)
        bm.normal_update()
        gaps = [(vmap[v].co - orig_co[v]).length for v in normals]
        opposed = sum(1 for f in faces if fmap[f].normal.dot(orig_n[f]) < -0.999)
        rim, shared = _rim_join(bm, faces, vmap) if join == "RIM" else ([], 0)
        lines = []
        if n_lines:
            sides = _band_sides(bm, new_faces)
            lines = _cut_lines(bm, new_faces, sides, n_lines)
        bm.normal_update()
        if join == "OBJECT":
            target = obj.copy()
            target.data = obj.data.copy()
            target.name = new_name
            target.data.name = new_name
            for coll in obj.users_collection:
                coll.objects.link(target)
            if target.animation_data is not None:
                target.animation_data_clear()
        if slot_new:
            _append_slot(target, slot_name)
        layer_verts = {v for f in new_faces for v in f.verts} | set(new_verts)
        line_len = [_line_length(row) for row in lines]
        _write_world(bm, target, shift)
        new_idx = sorted(v.index for v in layer_verts)
        line_idx = [[v.index for v in row] for row in lines]
        n_faces = len(new_faces)
        n_rim = len(rim)
    finally:
        bm.free()
    if join == "OBJECT":
        for vg0 in list(target.vertex_groups):
            if vg0.name in (group, lines_group):
                target.vertex_groups.remove(vg0)
        vg = target.vertex_groups.new(name=group)
    else:
        vg = old if old is not None else obj.vertex_groups.new(name=group)
    vg.add(new_idx, 1.0, "REPLACE")
    lines_out = None
    if line_idx:
        lg = target.vertex_groups.get(lines_group) or target.vertex_groups.new(name=lines_group)
        lg.add(sorted({i for row in line_idx for i in row}), 1.0, "REPLACE")
        lines_out = {"group": lg.name, "count": len(line_idx), "vertices": [len(r) for r in line_idx],
                     "lengths": [round(x, 6) for x in line_len]}
    target.data.update()
    out = {"object": obj.name, "layerObject": target.name, "side": side, "join": join, "flipNormals": flip,
           "facesByPart": by_part, "facesByOpenings": by_open, "facesByFaces": by_spec, "facesAdded": n_faces,
           "verticesAdded": len(new_idx), "rimFaces": n_rim, "rimEdgesShared": shared, "group": vg.name,
           "slot": slot_name, "offset": offset, "uvOffset": uvo,
           "gap": {"min": round(min(gaps), 6), "max": round(max(gaps), 6)}, "facesOpposed": opposed,
           "internalLines": lines_out, "after": mesh_counts(target),
           "bakeNote": "the copies share the outside's UVs - delete group '%s''s faces on a bake copy first." % vg.name}
    if join == "RIM":
        out["pressureNote"] = ("the rim join makes one closed shell: add_cloth pressure (usePressure, pressure above 0) "
                               "pushes its two walls apart - Blender has one pressure per object, not one per wall.")
    out.update(shared_data_note(obj))
    return out


def _line_length(verts):
    """World length of a cut line: the edges whose two ends both lie on it."""
    on = set(verts)
    return sum(e.calc_length() for e in {e for v in verts for e in v.link_edges} if e.other_vert(e.verts[0]) in on
               and e.verts[0] in on)


# --------------------------------------------------------------------------------------------- measure_neckline

_NECK_KEYS = ("body", "center", "neckZ", "armature", "neckBone", "angles", "zStep", "searchDown", "tolerance", "clip",
              "smoothPasses", "reach", "evaluated")
_NECK_OP_KEYS = ("body", "center", "neckZ", "armature", "neckBone", "angles", "zStep", "searchDown", "tolerance",
                 "clip", "smoothPasses", "reach", "evaluated", "trim")
_NECK_BONES = ("neck_01", "neck", "Neck", "neck.001", "DEF-spine.004", "ORG-spine.004")


def _circ_at(values, angle_deg):
    """The value of an evenly spaced per-angle series (index 0 at 0 degrees, counter-clockwise from +X) at an angle,
    linear between its two neighbours."""
    n = len(values)
    t = (angle_deg % 360.0) / (360.0 / n)
    i = int(math.floor(t))
    f = t - i
    return values[i % n] * (1.0 - f) + values[(i + 1) % n] * f


def _neck_axis(params, tail):
    """(center [x, y], neckZ, bone row or None) from center/neckZ, or from an armature's neck bone (posed)."""
    center = params.get("center")
    neck_z = take_float(params, "neckZ", default=None)
    bone_row = None
    if params.get("armature") is not None:
        arm = get_object(take(params, "armature", kind=str))
        if arm.type != "ARMATURE":
            raise MifOpError("armature '%s' is a %s, not an ARMATURE. %s" % (arm.name, arm.type, tail))
        want = params.get("neckBone")
        names = [str(want)] if want is not None else list(_NECK_BONES)
        pb = next((arm.pose.bones.get(n) for n in names if arm.pose.bones.get(n) is not None), None)
        if pb is None:
            raise MifOpError("armature '%s' has no bone named %s - pass neckBone. %s"
                             % (arm.name, " or ".join("'%s'" % n for n in names), tail))
        head, tip = arm.matrix_world @ pb.head, arm.matrix_world @ pb.tail
        mid = (head + tip) / 2
        bone_row = {"armature": arm.name, "bone": pb.name, "head": rnd(head), "tail": rnd(tip)}
        if center is None:
            center = [mid.x, mid.y]
        if neck_z is None:
            neck_z = mid.z
    elif params.get("neckBone") is not None:
        raise MifOpError("neckBone names a bone of `armature`, and no armature was given. %s" % tail)
    if center is None or neck_z is None:
        raise MifOpError("pass center [x, y] and neckZ (a height at which every angle meets the neck column alone), "
                         "or armature (its neck bone's middle gives both). %s" % tail)
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        raise MifOpError("center is [x, y] (world) - the neck's vertical axis, got %r. %s" % (center, tail))
    return finite_floats(list(center), "center"), neck_z, bone_row


def measure_neck(params, tail="NOTHING was read."):
    """The neckline of a body: per angle, where the neck column meets chest and shoulders. Shared by measure_neckline
    and build_collar; refuses before it reads anything when a parameter is wrong."""
    from mathutils.bvhtree import BVHTree
    body = get_object(take(params, "body", required=True, kind=str), want_mesh=True)
    (cx, cy), neck_z, bone_row = _neck_axis(params, tail)
    n_ang = take_int(params, "angles", default=32)
    z_step = take_float(params, "zStep", default=0.0025)
    search = take_float(params, "searchDown", default=0.15)
    tol = take_float(params, "tolerance", default=0.004)
    clip = take_float(params, "clip", default=0.015)
    passes = take_int(params, "smoothPasses", default=1)
    reach = take_float(params, "reach", default=0.5)
    evaluated = take_bool(params, "evaluated", default=True)
    if not 8 <= n_ang <= 360:
        raise MifOpError("angles must be 8-360, got %d. %s" % (n_ang, tail))
    if not 0 < z_step <= 0.05:
        raise MifOpError("zStep must be above 0 and at most 0.05 m, got %g. %s" % (z_step, tail))
    if not z_step <= search <= 2.0 or search / z_step > 4000:
        raise MifOpError("searchDown must be from zStep to 2 m and at most 4000 steps, got %g. %s" % (search, tail))
    for key, val in (("tolerance", tol), ("clip", clip), ("reach", reach)):
        if not val > 0:
            raise MifOpError("%s must be above 0, got %g. %s" % (key, val, tail))
    if not 0 <= passes <= 20:
        raise MifOpError("smoothPasses must be 0-20, got %d. %s" % (passes, tail))
    steps = int(search / z_step + 1e-9)
    bm = evaluated_world(body, evaluated)
    try:
        if not len(bm.faces):
            raise MifOpError("'%s' has no faces to measure. %s" % (body.name, tail))
        tree = BVHTree.FromBMesh(bm)
        rows, misses, outside, rays = [], [], [], 0
        for k in range(n_ang):
            ang = 2 * math.pi * k / n_ang
            e = Vector((math.cos(ang), math.sin(ang), 0.0))
            loc, nrm, _i, r0 = tree.ray_cast(Vector((cx, cy, neck_z)), e, reach)
            rays += 1
            if loc is None:
                misses.append(round(math.degrees(ang), 3))
                continue
            if nrm.dot(e) < 0:
                outside.append(round(math.degrees(ang), 3))
                continue
            inside, base, prev_z, prev_r, lim = [r0], None, neck_z, r0, r0 + tol
            for s in range(1, steps + 1):
                z = neck_z - s * z_step
                hit = tree.ray_cast(Vector((cx, cy, z)), e, reach)
                rays += 1
                r = hit[3] if hit[0] is not None else None
                if r is None or r > lim:
                    base = prev_z - z_step / 2 if r is None else prev_z - (lim - prev_r) / (r - prev_r) * z_step
                    break
                inside.append(r)
                prev_z, prev_r = z, r
            rows.append({"angle": round(math.degrees(ang), 4), "rawBaseZ": base,
                         "columnRadius": sorted(inside)[len(inside) // 2], "baseRadius": prev_r, "e": e})
    finally:
        bm.free()
    if misses:
        raise MifOpError("from center %s at neckZ %g a ray outward met nothing of '%s' within reach %g at angle(s) %s - "
                         "the axis is not inside the neck there. %s" % (rnd([cx, cy], 5), neck_z, body.name, reach,
                                                                        misses[:12], tail))
    if outside:
        raise MifOpError("at neckZ %g the ray outward from center %s met the OUTSIDE of '%s' at angle(s) %s (a face "
                         "facing the axis) - the axis is outside the neck column there, or the mesh is inside out. %s"
                         % (neck_z, rnd([cx, cy], 5), body.name, outside[:12], tail))
    raw = [r["rawBaseZ"] for r in rows]
    found = [i for i, v in enumerate(raw) if v is not None]
    if len(found) * 4 < n_ang * 3:
        raise MifOpError("the neck column did not end within searchDown %g below neckZ %g at %d of %d angles - raise "
                         "searchDown, or lower tolerance (%g). %s" % (search, neck_z, n_ang - len(found), n_ang, tol,
                                                                       tail))
    filled = list(raw)
    for i in range(n_ang):
        if filled[i] is None:
            a = next(j for j in range(1, n_ang) if raw[(i - j) % n_ang] is not None)
            b = next(j for j in range(1, n_ang) if raw[(i + j) % n_ang] is not None)
            filled[i] = (raw[(i - a) % n_ang] * b + raw[(i + b) % n_ang] * a) / (a + b)
    med = [sorted(filled[(i + d) % n_ang] for d in range(-2, 3))[2] for i in range(n_ang)]
    clipped = [abs(filled[i] - med[i]) > clip for i in range(n_ang)]
    vals = [med[i] if clipped[i] else filled[i] for i in range(n_ang)]
    for _ in range(passes):
        vals = [(vals[i - 1] + 2 * vals[i] + vals[(i + 1) % n_ang]) / 4.0 for i in range(n_ang)]
    for i, r in enumerate(rows):
        r["baseZ"] = vals[i]
        r["clipped"] = clipped[i]
        r["filled"] = raw[i] is None
        r["point"] = [cx + r["e"].x * r["baseRadius"], cy + r["e"].y * r["baseRadius"], vals[i]]
    return {"body": body, "center": [cx, cy], "neckZ": neck_z, "bone": bone_row, "rows": rows,
            "settings": {"angles": n_ang, "zStep": z_step, "searchDown": search, "tolerance": tol, "clip": clip,
                         "smoothPasses": passes, "reach": reach, "evaluated": evaluated},
            "rays": rays}


def _neck_reply(m):
    rows, (cx, cy) = m["rows"], m["center"]
    bases = [r["baseZ"] for r in rows]
    lo = min(range(len(rows)), key=lambda i: bases[i])
    hi = max(range(len(rows)), key=lambda i: bases[i])
    at = {name: round(_circ_at(bases, a), 5) for name, a in (("+X", 0), ("+Y", 90), ("-X", 180), ("-Y", 270))}
    return {"center": rnd([cx, cy]), "neckZ": round(m["neckZ"], 6), "bone": m["bone"], "settings": m["settings"],
            "angleCount": len(rows), "rays": m["rays"],
            "baseZ": {"min": round(bases[lo], 5), "minAtAngle": rows[lo]["angle"], "max": round(bases[hi], 5),
                      "maxAtAngle": rows[hi]["angle"], "spread": round(bases[hi] - bases[lo], 5)},
            "baseZAtAxes": at,
            "clippedAngles": [r["angle"] for r in rows if r["clipped"]],
            "filledAngles": [r["angle"] for r in rows if r["filled"]],
            "angles": [{"angle": r["angle"], "baseZ": round(r["baseZ"], 5),
                        "rawBaseZ": None if r["rawBaseZ"] is None else round(r["rawBaseZ"], 5),
                        "columnRadius": round(r["columnRadius"], 5), "baseRadius": round(r["baseRadius"], 5),
                        "point": rnd(r["point"], 5), "clipped": r["clipped"], "filled": r["filled"]} for r in rows],
            "ring": [rnd(r["point"], 5) for r in rows]}


def _parse_trim(raw, body, tail):
    if not isinstance(raw, dict) or "garment" not in raw or set(raw) - {"garment", "distance", "margin"}:
        raise MifOpError("trim is {garment, distance (m, default 0.05), margin (m, default 0)}, got %r. %s" % (raw, tail))
    g = get_object(take(raw, "garment", kind=str), want_mesh=True)
    if g is body:
        raise MifOpError("trim.garment is the body itself - trim cuts a garment above the body's neckline. %s" % tail)
    require_editable(g, "trimmed")
    dist = take_float(raw, "distance", default=0.05)
    margin = take_float(raw, "margin", default=0.0)
    if not dist > 0:
        raise MifOpError("trim.distance must be above 0 (metres beyond the neck column), got %g. %s" % (dist, tail))
    return g, dist, margin


def _trim_garment(g, m, dist, margin, tail):
    """Delete the garment's faces whose centre stands above the neckline (+ margin) within dist of the neck column."""
    (cx, cy), rows = m["center"], m["rows"]
    bases = [r["baseZ"] for r in rows]
    radii = [r["columnRadius"] for r in rows]
    mw = g.matrix_world
    bm = bmesh.new()
    try:
        bm.from_mesh(g.data)
        doomed = []
        for f in bm.faces:
            c = mw @ f.calc_center_median()
            ang = math.degrees(math.atan2(c.y - cy, c.x - cx))
            if c.z > _circ_at(bases, ang) + margin and math.hypot(c.x - cx, c.y - cy) < _circ_at(radii, ang) + dist:
                doomed.append(f)
        total = len(bm.faces)
        if not doomed:
            raise MifOpError("no face of '%s' stands above the neckline within %g of the neck column - a trim of nothing "
                             "is refused. NOTHING was trimmed." % (g.name, dist))
        if len(doomed) == total:
            raise MifOpError("every face of '%s' (%d) stands above the neckline within %g of the neck - that deletes "
                             "the garment. NOTHING was trimmed." % (g.name, total, dist))
        nv = len(bm.verts)
        bmesh.ops.delete(bm, geom=doomed, context="FACES")
        edge = []
        for v in bm.verts:
            if v.is_boundary:
                w = mw @ v.co
                ang = math.degrees(math.atan2(w.y - cy, w.x - cx))
                if math.hypot(w.x - cx, w.y - cy) < _circ_at(radii, ang) + dist:
                    edge.append(w.z - _circ_at(bases, ang))
        bm.to_mesh(g.data)
        g.data.update()
        return {"garment": g.name, "distance": dist, "margin": margin, "facesDeleted": len(doomed),
                "facesLeft": total - len(doomed), "verticesRemoved": nv - len(bm.verts),
                "neckEdgeAboveNeckline": stats(edge, [50]) if edge else None,
                "note": "the garment's BASE mesh in world space; a face goes when its centre is above the neckline at "
                        "its angle (plus margin) and nearer the neck's axis than the column radius plus distance."}
    finally:
        bm.free()


def op_measure_neckline(params):
    """Where a body's neck column meets its chest and shoulders, per angle - a neckline no plane follows.

    THE GAP (spec "A neckline measured from the body", 2026-10-05): a shirt's neck edge and its collar both sit where
    the neck meets chest and shoulders. Measured on a stock mannequin that line is 1.50 m at the front, 1.545 m at the
    sides and 1.53 m at the back, so no plane follows it, and one sampling height fails both ways (high, a front ray from
    outside meets the jaw; low, a side ray meets the shoulder). Here the rays go OUTWARD FROM THE NECK'S AXIS, so jaw,
    chin and shoulders outside the column are never in the way.

    THE AXIS: center [x, y] and neckZ (a height where every angle meets the neck column alone), or armature (+ neckBone,
    default neck_01 / neck / Neck ...) whose posed bone's middle gives both; an explicit center or neckZ wins. At each of
    `angles` (default 32, counter-clockwise from +X; -Y is a character's front) the column radius at neckZ is read, then
    rays step down zStep (default 0.0025) up to searchDown (0.15) until the section leaves the column by more than
    tolerance (0.004): that height is the base. An angle that never leaves is filled from its neighbours; one more than
    clip (0.015) from the median of its five neighbours is clipped to it; smoothPasses (1) of [1, 2, 1] follow.

    trim {garment, distance (0.05), margin (0)} DELETES the garment's faces whose centre stands above the neckline
    within distance of the column - the one write here. Refused before anything: an axis outside the neck at neckZ, a
    column that does not end at a quarter of the angles, a trim that would delete nothing or everything.
    """
    trim_raw = params.get("trim")
    tail = "NOTHING was trimmed." if trim_raw is not None else "NOTHING was read."
    reject_unknown(params, _NECK_OP_KEYS, "measure_neckline")
    trim = None
    if trim_raw is not None:
        body = get_object(take(params, "body", required=True, kind=str), want_mesh=True)
        trim = _parse_trim(trim_raw, body, tail)
    m = measure_neck(params, tail)
    out = _neck_reply(m)
    out.update({"op": "measure_neckline", "body": m["body"].name,
                "note": "baseZ is where the section has left the neck column by `tolerance`; point is that base at "
                        "the column's radius there - the ring build_collar takes as its neckline."})
    if trim is not None:
        out["trim"] = _trim_garment(trim[0], m, trim[1], trim[2], tail)
    else:
        out["trim"] = None
    return out


# --------------------------------------------------------------------------------------------- build_collar

_PLACE_KEYS = ("name", "collection", "takeOverName")
_COLLAR_KEYS = ("neckline", "center", "segments", "standHeight", "ease", "standRows", "foldGap", "fallWidth",
                "fallAngle", "fallRows", "frontAngle", "frontGap", "pointDrop", "pointAngle", "pointSpan", "shirt",
                "layOffset", "layReach", "material", "smooth", "uvScale", "evaluated", "name", "collection",
                "takeOverName")


def build_part(params, verts, faces, uvs, smooth, default_name, op, materials=None, mat_idx=None, groups=None):
    """A mesh object through create_mesh (which owns naming, validation and the read-back), in world space."""
    from .ops_create import op_create_mesh   # noqa: PLC0415 - create_mesh owns naming and the read-back
    sub = {"name": params.get("name") or default_name, "vertices": verts, "faces": faces, "uvs": {"UVMap": uvs},
           "smooth": bool(smooth)}
    for k in _PLACE_KEYS[1:]:
        if params.get(k) is not None:
            sub[k] = params[k]
    if materials:
        sub["materials"] = list(materials)
        sub["materialIndices"] = list(mat_idx) if mat_idx is not None else [0] * len(faces)
    if groups:
        sub["vertexGroups"] = groups
    out = op_create_mesh(sub)
    out["op"] = op
    return out


def _ring_at(ring, angle_deg):
    """(radius, z) of a neckline ring [(angle_deg, r, z) ascending] at an angle, linear between its neighbours."""
    a = angle_deg % 360.0
    n = len(ring)
    for i in range(n):
        a0, r0, z0 = ring[i]
        a1, r1, z1 = ring[(i + 1) % n]
        span = (a1 - a0) % 360.0
        off = (a - a0) % 360.0
        if off <= span:
            f = off / span if span > 0 else 0.0
            return r0 + (r1 - r0) * f, z0 + (z1 - z0) * f
    return ring[0][1], ring[0][2]


def _neckline_ring(params, tail):
    """The neckline as [(angle, radius, z)] round its center, from points or from a measure_neckline request."""
    raw = params.get("neckline")
    source = None
    if isinstance(raw, dict):
        if set(raw) - set(_NECK_KEYS):
            raise MifOpError("neckline as a dict is a measure_neckline request (%s); it has unknown key(s) %s. %s"
                             % (", ".join(_NECK_KEYS), ", ".join(sorted(set(raw) - set(_NECK_KEYS))), tail))
        m = measure_neck(dict(raw), tail)
        pts = [r["point"] for r in m["rows"]]
        center = m["center"] if params.get("center") is None else None
        source = {"measured": m["body"].name, "angles": len(pts), "baseZ": [round(min(p[2] for p in pts), 5),
                                                                           round(max(p[2] for p in pts), 5)]}
    elif isinstance(raw, (list, tuple)):
        if len(raw) < 8 or any(not isinstance(p, (list, tuple)) or len(p) != 3 for p in raw):
            raise MifOpError("neckline is eight or more [x, y, z] points round the neck (measure_neckline's ring), "
                             "or a measure_neckline request {body, ...}. %s" % tail)
        pts = [finite_floats(list(p), "neckline") for p in raw]
        center = None
        source = {"points": len(pts)}
    else:
        raise MifOpError("neckline is required: [[x, y, z], ...] round the neck, or {body, center/neckZ or armature, "
                         "...} measured here. %s" % tail)
    if params.get("center") is not None:
        c = params["center"]
        if not isinstance(c, (list, tuple)) or len(c) != 2:
            raise MifOpError("center is [x, y] (world), got %r. %s" % (c, tail))
        center = finite_floats(list(c), "center")
    if center is None:
        center = [sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)]
    ring = []
    for p in pts:
        r = math.hypot(p[0] - center[0], p[1] - center[1])
        if r < 1e-6:
            raise MifOpError("a neckline point %s sits on the axis %s. %s" % (rnd(p, 5), rnd(center, 5), tail))
        ring.append((math.degrees(math.atan2(p[1] - center[1], p[0] - center[0])) % 360.0, r, p[2]))
    ring.sort()
    gaps = [(ring[(i + 1) % len(ring)][0] - ring[i][0]) % 360.0 for i in range(len(ring))]
    if min(gaps) < 1e-6 or max(gaps) > 90.0:
        raise MifOpError("the neckline's points must go round the axis %s with no two at one angle and no gap over 90 "
                         "degrees (widest gap %.1f). %s" % (rnd(center, 5), max(gaps), tail))
    return ring, center, source


def op_build_collar(params):
    """A shirt collar - stand, fold and fall - as one clean ring at even angles on a neckline. WRITES a new object.

    THE GAP (spec "A collar builder", 2026-10-05): a collar built from the shirt's own ragged neck edge came out jagged
    three times. Here it is built on a NECKLINE - measure_neckline's ring [[x, y, z], ...], or a measure_neckline
    request {body, center, neckZ | armature, ...} run first - resampled at `segments` (default 64) EVEN angles round
    center (default the neckline's own; frontAngle default 270 = -Y, a character's front).

    THE SECTION at each angle, in the plane through the axis: the STAND rises standHeight (0.03) straight up from the
    neckline, ease (0.005) out from it, in standRows (2); the FOLD turns over foldGap (0.004); the FALL runs down and
    out fallWidth (0.045) at fallAngle (20 degrees from vertical) in fallRows (3). frontGap (0.02 m, 0 = a closed band)
    is left open at the front; there COLLAR POINTS grow over pointSpan (40 degrees): the fall lengthens by pointDrop
    (0.02) and its tip swings pointAngle (15 degrees, + toward the gap) round the neck. shirt lays the fall onto that
    mesh: a fall point inside it, or within layReach (0.03) outside, goes to its surface plus layOffset (0.002).

    One sheet (give it thickness with a SOLIDIFY modifier), vertex groups CollarStand and CollarFall, u along the
    neckline and v down the section in metres / uvScale (1). material, smooth (default true), name, collection,
    takeOverName. Refused before anything exists: a neckline under 8 points or not round its axis, sizes out of range,
    a front gap wider than the neck, an unknown shirt or material.
    """
    from mathutils.bvhtree import BVHTree
    tail = "NOTHING was created."
    reject_unknown(params, _COLLAR_KEYS, "build_collar")
    segs = take_int(params, "segments", default=64)
    stand_h = take_float(params, "standHeight", default=0.03)
    ease = take_float(params, "ease", default=0.005)
    stand_rows = take_int(params, "standRows", default=2)
    fold = take_float(params, "foldGap", default=0.004)
    fall_w = take_float(params, "fallWidth", default=0.045)
    fall_ang = take_float(params, "fallAngle", default=20.0)
    fall_rows = take_int(params, "fallRows", default=3)
    front = take_float(params, "frontAngle", default=270.0)
    gap = take_float(params, "frontGap", default=0.02)
    p_drop = take_float(params, "pointDrop", default=0.02)
    p_ang = take_float(params, "pointAngle", default=15.0)
    p_span = take_float(params, "pointSpan", default=40.0)
    lay_off = take_float(params, "layOffset", default=0.002)
    lay_reach = take_float(params, "layReach", default=0.03)
    smooth = take_bool(params, "smooth", default=True)
    uv_scale = take_float(params, "uvScale", default=1.0)
    evaluated = take_bool(params, "evaluated", default=True)
    if not 8 <= segs <= 512:
        raise MifOpError("segments must be 8-512, got %d. %s" % (segs, tail))
    for key, val in (("standHeight", stand_h), ("foldGap", fold), ("uvScale", uv_scale)):
        if not val > 0:
            raise MifOpError("%s must be above 0, got %g. %s" % (key, val, tail))
    for key, val in (("ease", ease), ("fallWidth", fall_w), ("frontGap", gap), ("pointDrop", p_drop),
                     ("layOffset", lay_off), ("layReach", lay_reach)):
        if val < 0:
            raise MifOpError("%s must be 0 or more metres, got %g. %s" % (key, val, tail))
    if not 1 <= stand_rows <= 32 or not 1 <= fall_rows <= 32:
        raise MifOpError("standRows and fallRows must be 1-32, got %d and %d. %s" % (stand_rows, fall_rows, tail))
    if not 0 <= fall_ang < 90:
        raise MifOpError("fallAngle is degrees out from straight down, 0 to under 90, got %g. %s" % (fall_ang, tail))
    if not -80 <= p_ang <= 80 or not 0 < p_span <= 170:
        raise MifOpError("pointAngle must be -80 to 80 and pointSpan above 0 to 170 degrees, got %g and %g. %s"
                         % (p_ang, p_span, tail))
    if params.get("material") is not None and params["material"] not in bpy.data.materials:
        raise MifOpError("no material named %r. %s" % (params["material"], tail))
    shirt = None
    if params.get("shirt") is not None:
        shirt = get_object(take(params, "shirt", kind=str), want_mesh=True)
    ring, (cx, cy), source = _neckline_ring(params, tail)

    r_front = _ring_at(ring, front)[0] + ease
    half = 0.0
    if gap > 0:
        if gap / 2 >= r_front * 0.95:
            raise MifOpError("frontGap %g is as wide as the neck at the front (radius %.4f). %s" % (gap, r_front, tail))
        half = math.degrees(2 * math.asin(gap / 2 / r_front)) / 2
    closed = gap == 0
    span = 360.0 - 2 * half
    cols = segs if closed else segs + 1
    angles = [front + (360.0 * k / segs if closed else half + span * k / segs) for k in range(cols)]
    phi = math.radians(fall_ang)
    sec_n = stand_rows + 2 + fall_rows

    tree = None
    sbm = None
    if shirt is not None:
        sbm = evaluated_world(shirt, evaluated)
        if not len(sbm.faces):
            sbm.free()
            raise MifOpError("shirt '%s' has no faces to lay the fall on. %s" % (shirt.name, tail))
        tree = BVHTree.FromBMesh(sbm)
    try:
        verts, lay_gaps, moved = [], [], 0
        for k, a in enumerate(angles):
            r, zb = _ring_at(ring, a)
            r0, zt = r + ease, zb + stand_h
            w = 0.0
            if not closed:
                from_end = min(a - angles[0], angles[-1] - a)
                if from_end < p_span:
                    s = 1.0 - from_end / p_span
                    w = s * s * (3 - 2 * s)
            toward = -1.0 if (not closed and a - angles[0] < angles[-1] - a) else 1.0
            length = fall_w + p_drop * w
            sec = [(r0, zb + stand_h * j / stand_rows, 0.0) for j in range(stand_rows + 1)]
            sec.append((r0 + fold / 2, zt + fold / 2, 0.0))
            for j in range(1, fall_rows + 1):
                d = length * j / fall_rows
                rr = r0 + fold + d * math.sin(phi)
                sec.append((rr, zt - d * math.cos(phi), toward * math.degrees(d * math.tan(math.radians(p_ang)) * w / rr)))
            for j, (rr, z, da) in enumerate(sec):
                ang = math.radians(a + da)
                p = Vector((cx + rr * math.cos(ang), cy + rr * math.sin(ang), z))
                if tree is not None and j > stand_rows + 1:
                    loc, nrm, _i, _d = tree.find_nearest(p)
                    if loc is not None:
                        side = (p - loc).dot(nrm)
                        if side < max(lay_off, lay_reach):
                            p = loc + nrm * lay_off
                            moved += 1
                        lay_gaps.append((p - loc).dot(nrm))
                verts.append(list(p))
    finally:
        if sbm is not None:
            sbm.free()

    faces, uvs = [], []
    along = [0.0]
    for k in range(1, cols + (1 if closed else 0)):
        a0, a1 = math.radians(angles[k - 1]), math.radians(angles[k % cols] + (360.0 if closed and k == cols else 0))
        along.append(along[-1] + (_ring_at(ring, angles[k - 1])[0] + ease) * abs(a1 - a0))
    down = [0.0]
    for j in range(1, sec_n):
        down.append(down[-1] + (Vector(verts[j]) - Vector(verts[j - 1])).length)
    for k in range(cols - (0 if closed else 1)):
        k1 = (k + 1) % cols
        for j in range(sec_n - 1):
            faces.append([k * sec_n + j, k1 * sec_n + j, k1 * sec_n + j + 1, k * sec_n + j + 1])
            u0, u1 = along[k] / uv_scale, along[k + 1] / uv_scale
            v0, v1 = down[j] / uv_scale, down[j + 1] / uv_scale
            uvs.append([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
    groups = {"CollarStand": [k * sec_n + j for k in range(cols) for j in range(stand_rows + 1)],
              "CollarFall": [k * sec_n + j for k in range(cols) for j in range(stand_rows + 2, sec_n)]}
    mats = [params["material"]] if params.get("material") is not None else None
    out = build_part(params, verts, faces, uvs, smooth, "Collar", "build_collar", mats, None, groups)
    ends = None
    if not closed:
        b0, b1 = Vector(verts[0]), Vector(verts[(cols - 1) * sec_n])
        t0, t1 = Vector(verts[sec_n - 1]), Vector(verts[cols * sec_n - 1])
        ends = {"gapAtNeckline": round((b1 - b0).length, 6), "gapAtPoints": round((t1 - t0).length, 6),
                "fallAtPoints": round(fall_w + p_drop, 6)}
    out.update({"neckline": source, "center": rnd([cx, cy]), "columns": cols, "closedBand": closed,
                "angleStep": round(360.0 / segs if closed else span / segs, 6), "frontGapDegrees": round(2 * half, 4),
                "sectionPoints": sec_n, "standHeight": stand_h, "ease": ease, "fallWidth": fall_w, "front": ends,
                "laid": None if shirt is None else {"shirt": shirt.name, "fallPointsMoved": moved,
                                                     "gap": stats(lay_gaps, [5])},
                "note": "one sheet - its outer side is the stand's outside and the fall's underside; give it "
                        "thickness with a SOLIDIFY modifier, or render it two-sided."})
    return out


# --------------------------------------------------------------------------------------------- build_applique

_APPLIQUE_KEYS = ("target", "origin", "direction", "up", "outline", "spin", "resolution", "lift", "thickness",
                  "bottom", "reach", "material", "sideMaterial", "smooth", "evaluated", "name", "collection",
                  "takeOverName")


def _smooth_normal(bm, face_index, point):
    """The surface normal at a point on a face, interpolated from its corners' vertex normals."""
    from mathutils.interpolate import poly_3d_calc
    f = bm.faces[face_index]
    weights = poly_3d_calc([v.co for v in f.verts], point)
    n = Vector((0.0, 0.0, 0.0))
    for w, v in zip(weights, f.verts):
        n += v.normal * w
    return n.normalized() if n.length > 1e-12 else f.normal.copy()


def _star_outline(raw, res, tail):
    """A closed 2D outline -> (its points counter-clockwise and resampled to res, its area centroid, area); refused
    unless a fan from the centroid reaches every point without folding (star-shaped)."""
    if not isinstance(raw, (list, tuple)) or len(raw) < 3 or any(
            not isinstance(p, (list, tuple)) or len(p) != 2 for p in raw):
        raise MifOpError("outline is three or more [x, y] in metres on the surface (x right, y along up), round the "
                         "hit point. %s" % tail)
    pts = [finite_floats(list(p), "outline") for p in raw]
    if (Vector(pts[0]) - Vector(pts[-1])).length < 1e-9:
        pts = pts[:-1]
    for i in range(len(pts)):
        if (Vector(pts[i]) - Vector(pts[i - 1])).length < 1e-9:
            raise MifOpError("outline points %d and %d are on one spot. %s" % ((i - 1) % len(pts), i, tail))
    area = sum(pts[i - 1][0] * pts[i][1] - pts[i][0] * pts[i - 1][1] for i in range(len(pts))) / 2.0
    if abs(area) < 1e-10:
        raise MifOpError("the outline encloses no area. %s" % tail)
    if area < 0:
        pts = pts[::-1]
        area = -area
    cx = sum((pts[i - 1][0] + pts[i][0]) * (pts[i - 1][0] * pts[i][1] - pts[i][0] * pts[i - 1][1])
             for i in range(len(pts))) / (6 * area)
    cy = sum((pts[i - 1][1] + pts[i][1]) * (pts[i - 1][0] * pts[i][1] - pts[i][0] * pts[i - 1][1])
             for i in range(len(pts))) / (6 * area)
    out = []
    for i in range(len(pts)):
        a, b = Vector(pts[i]), Vector(pts[(i + 1) % len(pts)])
        n = max(1, int(math.ceil((b - a).length / res - 1e-6)))
        out.extend([list(a.lerp(b, k / n)) for k in range(n)])
    turn = 0.0
    for i in range(len(out)):
        a0 = math.atan2(out[i - 1][1] - cy, out[i - 1][0] - cx)
        a1 = math.atan2(out[i][1] - cy, out[i][0] - cx)
        d = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        if d <= 1e-9 or math.hypot(out[i][0] - cx, out[i][1] - cy) < 1e-9:
            raise MifOpError("the outline is not star-shaped round its centroid (%.4f, %.4f): at point %s a fan from the "
                             "centre would fold back. Split it into star-shaped pieces. %s"
                             % (cx, cy, rnd(out[i], 5), tail))
        turn += d
    if abs(turn - 2 * math.pi) > 1e-6:
        raise MifOpError("the outline winds round its centroid %.2f times, not once. %s" % (turn / (2 * math.pi), tail))
    return out, (cx, cy), area


_APPLIQUE_STEPS = (0.04, 0.02, 0.01)


def _applique_deviation(tree, hit, uu, vv, nrm, reach, outline, fx, fy, rings, stop=None):
    """How far straight spans between this fan's points leave the surface: the worst, over every radial and every round
    span, of the gap between the span middle's height (cast along -N, measured along the frame's normal) and the middle
    of its ends' heights. None when a ray misses (the fan cannot be laid at this step). Stops early past `stop`."""
    def height(x, y):
        h = tree.ray_cast(hit + uu * x + vv * y + nrm * reach, -nrm, 2 * reach)[0]
        return None if h is None else (h - hit).dot(nrm)

    def at(p, k):
        t = k / float(rings)
        return (fx + (p[0] - fx) * t, fy + (p[1] - fy) * t)

    worst = 0.0
    for i, p in enumerate(outline):
        q = outline[(i + 1) % len(outline)]
        for k in range(0, rings + 1):
            spans = [(at(p, k), at(p, k + 1))] if k < rings else []
            if k:
                spans.append((at(p, k), at(q, k)))
            for a, b in spans:
                ha, hb = height(*a), height(*b)
                hm = height((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
                if ha is None or hb is None or hm is None:
                    return None
                worst = max(worst, abs(hm - (ha + hb) / 2.0))
                if stop is not None and worst > stop:
                    return worst
    return worst


def op_build_applique(params):
    """A patch laid on a surface by rays - a pocket flap, a badge, a name plate, an epaulet. WRITES a new object.

    THE GAP (spec "Appliques on a surface", 2026-10-05): pocket flaps, a badge, a name plate, epaulets and shoulder
    patches were each laid on a shirt in raw Python by rays from a 2D outline, as a radial fan (a grid kept inside the
    outline came out stair-stepped), lifted and given side walls.

    WHERE: a ray from origin along direction meets target (world, evaluated); there the surface normal N and up
    (default world +Z, projected onto the surface; spin turns it, degrees) set the outline's frame - x right, y up.
    outline [[x, y], ...] in METRES round that hit point, closed implicitly, star-shaped round its centroid. Its edges
    are resampled to resolution and filled as a radial fan from the centroid at the same spacing; each fan point is cast
    back onto the target along -N from reach (0.05 m) in front of the frame.

    RESOLUTION IS THE FAN'S SPACING IN METRES, both along the rim and between the rings: bigger is coarser and makes
    fewer faces, about (outline perimeter / resolution) x (centroid-to-rim distance / resolution) quads a side - a
    12 x 14 cm pocket is 286 faces at 2 cm, 950 at 1 cm and 3,626 at 5 mm (measured on 5.0.1). Left out, it is the coarsest of 4, 2 and 1 cm whose spans
    follow the surface, measured by casting rays between them (resolutionFrom "surface"): within a quarter of the lift,
    or no worse than 1.25 times the finest step where the cloth's own grain is rougher than that - a finer fan that
    does not follow better only adds faces - and never a step whose spans dip a whole lift. When none qualifies it is
    1 cm (resolutionFrom "floor"), never finer. resolutionTried gives each step's spanDeviation (metres). A patch on flat cloth needs few faces: a 5 mm fan was
    3,072 triangles for a 10 x 6 cm patch, and the old 5 mm floor made 3,626 faces for a 12 cm pocket on folded denim
    (2026-11-07). An outline keeps every point it is given.

    THE PART: the underside sits lift (0.001 m) off the surface along its interpolated normal, the top thickness
    (0.002; 0 = one lifted sheet) above it, side walls round the rim, and bottom (default true) closes the underside
    so the part is a closed solid - checked by signed volume. Slots: material on top and bottom, sideMaterial on the
    walls (default material). Vertex groups AppliqueTop and AppliqueBottom; UVs: the outline's box on 0-1, the wall
    along the rim. Refused before anything exists: a ray that misses, a fan point that misses the target, an
    outline not star-shaped, up along the normal, sizes out of range, an unknown material.
    """
    from mathutils.bvhtree import BVHTree
    tail = "NOTHING was created."
    reject_unknown(params, _APPLIQUE_KEYS, "build_applique")
    target = get_object(take(params, "target", required=True, kind=str), want_mesh=True)
    vecs = {}
    for key, default in (("origin", None), ("direction", None), ("up", (0.0, 0.0, 1.0))):
        raw = params.get(key, default)
        if raw is None or not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise MifOpError("%s is [x, y, z] (world)%s, got %r. %s"
                             % (key, "" if default is None else ", default [0, 0, 1]", raw, tail))
        vecs[key] = Vector(finite_floats(list(raw), key))
    if vecs["direction"].length < 1e-9 or vecs["up"].length < 1e-9:
        raise MifOpError("direction and up must not be zero. %s" % tail)
    spin = take_float(params, "spin", default=0.0)
    res = take_float(params, "resolution", default=_APPLIQUE_STEPS[-1])
    res_given = params.get("resolution") is not None
    lift = take_float(params, "lift", default=0.001)
    thick = take_float(params, "thickness", default=0.002)
    reach = take_float(params, "reach", default=0.05)
    bottom = take_bool(params, "bottom", default=True)
    smooth = take_bool(params, "smooth", default=False)
    evaluated = take_bool(params, "evaluated", default=True)
    if not 0.0005 <= res <= 1.0:
        raise MifOpError("resolution must be 0.0005-1 m, got %g. %s" % (res, tail))
    if lift < 0 or thick < 0 or not reach > 0:
        raise MifOpError("lift and thickness must be 0 or more and reach above 0, got %g, %g and %g. %s"
                         % (lift, thick, reach, tail))
    mats = []
    for key in ("material", "sideMaterial"):
        if params.get(key) is not None:
            if bpy.data.materials.get(str(params[key])) is None:
                raise MifOpError("no material named %r (%s). %s" % (params[key], key, tail))
            mats.append(params[key])
    if params.get("sideMaterial") is not None and params.get("material") is None:
        raise MifOpError("sideMaterial needs material for the top and bottom too. %s" % tail)
    outline, (fx, fy), area = _star_outline(params.get("outline"), res, tail)
    far = max(math.hypot(p[0] - fx, p[1] - fy) for p in outline)
    rings = max(1, int(math.ceil(far / res - 1e-6)))
    if rings * len(outline) > 200000:
        raise MifOpError("%d rings of %d points is over 200,000 vertices - raise resolution. %s"
                         % (rings, len(outline), tail))

    bm = evaluated_world(target, evaluated)
    try:
        if not len(bm.faces):
            raise MifOpError("'%s' has no faces to lay on. %s" % (target.name, tail))
        tree = BVHTree.FromBMesh(bm)
        d = vecs["direction"].normalized()
        hit, _n, idx, _dist = tree.ray_cast(vecs["origin"], d)
        if hit is None:
            raise MifOpError("the ray from %s along %s meets nothing of '%s'. %s"
                             % (rnd(vecs["origin"], 4), rnd(d, 4), target.name, tail))
        nrm = _smooth_normal(bm, idx, hit)
        if nrm.dot(d) > 0:
            nrm = -nrm
        upv = vecs["up"].normalized()
        vv = upv - nrm * upv.dot(nrm)
        if vv.length < 0.1:
            raise MifOpError("up %s lies (nearly) along the surface normal %s there - pass another up. %s"
                             % (rnd(upv, 4), rnd(nrm, 4), tail))
        vv.normalize()
        uu = vv.cross(nrm).normalized()
        if spin:
            c, s = math.cos(math.radians(spin)), math.sin(math.radians(spin))
            uu, vv = uu * c + vv * s, vv * c - uu * s
        tried = []
        if not res_given:
            # EVERY STEP IS MEASURED, then the coarsest that follows is taken: within a quarter of the lift, or - on
            # cloth whose own grain is rougher than that - no worse than 1.25 times the finest step, since a finer fan
            # that does not follow better only adds faces. A step whose spans dip a whole lift would sink; it is never
            # taken, and when every step would, the finest is (resolutionFrom "floor").
            tol = max(lift, 1e-3) / 4.0
            fans = []
            for coarse in _APPLIQUE_STEPS:
                o2, (fx2, fy2), area2 = _star_outline(params.get("outline"), coarse, tail)
                r2 = max(1, int(math.ceil(max(math.hypot(p[0] - fx2, p[1] - fy2) for p in o2) / coarse - 1e-6)))
                dev = _applique_deviation(tree, hit, uu, vv, nrm, reach, o2, fx2, fy2, r2)
                fans.append((coarse, dev, (o2, fx2, fy2, area2, r2)))
            devs = [d for _c, d, _f in fans if d is not None]
            bound = max(tol, 1.25 * min(devs)) if devs else tol
            picked = None
            for coarse, dev, fan in fans:
                ok = dev is not None and dev <= bound and dev < max(lift, 1e-6)
                tried.append({"resolution": coarse, "spanDeviation": None if dev is None else round(dev, 6),
                              "follows": ok})
                if ok and picked is None:
                    picked = (coarse, fan)
            res_from = "surface"
            if picked is None:
                # NONE FOLLOWS: the finest step, never finer. It was 5 mm here, and on a test character's denim that
                # made 3,626 faces for one 12 cm pocket.
                picked = (fans[-1][0], fans[-1][2])
                res_from = "floor"
            res = picked[0]
            outline, fx, fy, area, rings = picked[1]
        else:
            res_from = "given"
        grid = [(fx, fy)] + [(fx + (p[0] - fx) * k / rings, fy + (p[1] - fy) * k / rings)
                             for k in range(1, rings + 1) for p in outline]
        under, normals, misses = [], [], []
        for x, y in grid:
            q = hit + uu * x + vv * y + nrm * reach
            h, _fn, fi, _fd = tree.ray_cast(q, -nrm, 2 * reach)
            if h is None:
                misses.append(rnd([x, y], 4))
                continue
            n = _smooth_normal(bm, fi, h)
            if n.dot(nrm) < 0:
                n = -n
            under.append(h + n * lift)
            normals.append(n)
        if misses:
            raise MifOpError("%d of %d fan points cast along -N miss '%s' within reach %g (first at outline %s) - the "
                             "outline runs off the surface, or raise reach. %s"
                             % (len(misses), len(grid), target.name, reach, misses[0], tail))
        gaps = [abs(tree.find_nearest(p)[3] - lift) for p in under]
    finally:
        bm.free()

    m = len(outline)
    count = 1 + rings * m

    def ring_i(k, i):
        return 0 if k == 0 else 1 + (k - 1) * m + (i % m)

    xs = [p[0] for p in outline]
    ys = [p[1] for p in outline]
    sx, sy = max(xs) - min(xs), max(ys) - min(ys)
    uv2 = [((x - min(xs)) / sx, (y - min(ys)) / sy) for x, y in grid]
    sheet = []
    for i in range(m):
        sheet.append([ring_i(0, 0), ring_i(1, i), ring_i(1, i + 1)])
        for k in range(1, rings):
            sheet.append([ring_i(k, i), ring_i(k + 1, i), ring_i(k + 1, i + 1), ring_i(k, i + 1)])
    if thick > 0:
        verts = [list(p + n * thick) for p, n in zip(under, normals)] + [list(p) for p in under]
    else:
        verts = [list(p) for p in under]
    faces = [list(f) for f in sheet]
    uvs = [[list(uv2[i]) for i in f] for f in sheet]
    slot = [0] * len(faces)
    if thick > 0:
        if bottom:
            for f in sheet:
                faces.append([count + i for i in reversed(f)])
                uvs.append([list(uv2[i]) for i in reversed(f)])
                slot.append(0)
        perim = [0.0]
        for i in range(1, m + 1):
            perim.append(perim[-1] + math.hypot(outline[i % m][0] - outline[i - 1][0],
                                                outline[i % m][1] - outline[i - 1][1]))
        for i in range(m):
            a, b = ring_i(rings, i), ring_i(rings, i + 1)
            faces.append([count + a, count + b, b, a])
            u0, u1 = perim[i] / max(perim[-1], 1e-12), perim[i + 1] / max(perim[-1], 1e-12)
            uvs.append([[u0, 0.0], [u1, 0.0], [u1, 1.0], [u0, 1.0]])
            slot.append(1 if len(mats) > 1 else 0)
    from .ops_kit import _signed_volume   # noqa: PLC0415 - one signed volume for every builder
    closed = thick > 0 and bottom
    vol = _signed_volume(verts, faces) if closed else None
    turned = bool(closed and vol < 0)
    if turned:
        faces = [list(reversed(f)) for f in faces]
        uvs = [list(reversed(u)) for u in uvs]
        vol = -vol
    groups = {"AppliqueTop": list(range(count))}
    if thick > 0:
        groups["AppliqueBottom"] = list(range(count, 2 * count))
    out = build_part(params, verts, faces, uvs, smooth, "Applique", "build_applique", mats or None,
                     slot if mats else None, groups)
    out.update({"target": target.name, "hitPoint": rnd(hit), "normal": rnd(nrm), "frame": {"x": rnd(uu), "y": rnd(vv)},
                "outlinePoints": m, "rings": rings, "rays": len(grid) + 1, "outlineArea": round(area, 8),
                "resolution": res, "resolutionFrom": res_from, "resolutionTried": tried or None,
                "fanCenter": rnd([fx, fy]), "lift": lift, "thickness": thick, "closed": closed,
                "liftError": round(max(gaps), 7),
                "outwardCheck": None if vol is None else {"signedVolume": round(vol, 10), "turnedOver": turned}})
    return out


# --------------------------------------------------------------------------------------------- build_band

_BAND_KEYS = ("around", "z0", "z1", "center", "segments", "samples", "offset", "thickness", "stiff", "smoothPasses",
              "items", "material", "itemMaterial", "smooth", "evaluated", "name", "collection", "takeOverName")


def _band_items(raw, tail):
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)) or len(raw) > 64:
        raise MifOpError("items is a list of up to 64 {angle, dz, size [w, h, d], material?}. %s" % tail)
    out = []
    for i, it in enumerate(raw):
        if not isinstance(it, dict) or "angle" not in it or "size" not in it or set(it) - {"angle", "dz", "size",
                                                                                            "material"}:
            raise MifOpError("items[%d] is {angle (degrees), dz (m from the band's middle, default 0), size [width, "
                             "height, depth] (m), material?}, got %r. %s" % (i, it, tail))
        size = it["size"]
        if not isinstance(size, (list, tuple)) or len(size) != 3:
            raise MifOpError("items[%d].size is [width, height, depth], got %r. %s" % (i, size, tail))
        size = finite_floats(list(size), "items[%d].size" % i)
        if min(size) <= 0:
            raise MifOpError("items[%d].size must be above 0 on every side, got %r. %s" % (i, size, tail))
        mat = it.get("material")
        if mat is not None and bpy.data.materials.get(str(mat)) is None:
            raise MifOpError("items[%d]: no material named %r. %s" % (i, mat, tail))
        out.append({"angle": finite_float(it["angle"], "items[%d].angle" % i),
                    "dz": finite_float(it.get("dz", 0.0), "items[%d].dz" % i), "size": size, "material": mat})
    return out


def op_build_band(params):
    """A stiff band round a body - a duty belt, a waistband, a strap - with boxes on it. WRITES a new object.

    THE GAP (spec "A stiff band round a body", 2026-10-05): a duty belt was built in raw Python from rings sampled by
    rays onto the garment underneath; following the hip's taper made it slant like a tray, and what worked was ONE
    radius per angle over the band's height - the widest - closed top and bottom, plus boxes placed on it by angle and
    height.

    RAYS from outside toward a vertical axis through center (default the middle of around's vertices between z0 and
    z1) meet `around` (world, evaluated) at `segments` (64) angles and `samples` (5) heights from z0 to z1. stiff
    (default true) takes the widest radius at each angle for the whole height - straight walls; false follows the
    surface at every sampled height. smoothPasses (2) of [1, 2, 1] round the band never go inside what was measured.
    The inner wall stands offset (0.003) out from that, the outer thickness (0.01) beyond, closed top and bottom - a
    closed solid checked by signed volume.

    items [{angle (degrees, counter-clockwise from +X; -Y = 270 is a character's front), dz (from the band's middle),
    size [width, height, depth], material?}] are boxes standing on the outer wall, their back flat at the widest outer
    radius they span. material on the band, itemMaterial (or an item's own) on the boxes. Refused before anything
    exists: a ray that misses, z1 not above z0, sizes out of range, an unknown material.
    """
    from mathutils.bvhtree import BVHTree
    tail = "NOTHING was created."
    reject_unknown(params, _BAND_KEYS, "build_band")
    around = get_object(take(params, "around", required=True, kind=str), want_mesh=True)
    z0 = take_float(params, "z0", required=True)
    z1 = take_float(params, "z1", required=True)
    segs = take_int(params, "segments", default=64)
    samples = take_int(params, "samples", default=5)
    offset = take_float(params, "offset", default=0.003)
    thick = take_float(params, "thickness", default=0.01)
    stiff = take_bool(params, "stiff", default=True)
    passes = take_int(params, "smoothPasses", default=2)
    smooth = take_bool(params, "smooth", default=False)
    evaluated = take_bool(params, "evaluated", default=True)
    if not z1 > z0:
        raise MifOpError("z1 must be above z0 (world heights of the band's bottom and top), got %g and %g. %s"
                         % (z0, z1, tail))
    if not 8 <= segs <= 512 or not 2 <= samples <= 64:
        raise MifOpError("segments must be 8-512 and samples 2-64, got %d and %d. %s" % (segs, samples, tail))
    if offset < 0 or not thick > 0:
        raise MifOpError("offset must be 0 or more and thickness above 0, got %g and %g. %s" % (offset, thick, tail))
    if not 0 <= passes <= 50:
        raise MifOpError("smoothPasses must be 0-50, got %d. %s" % (passes, tail))
    mats = []
    for key in ("material", "itemMaterial"):
        if params.get(key) is not None and bpy.data.materials.get(str(params[key])) is None:
            raise MifOpError("no material named %r (%s). %s" % (params[key], key, tail))
    items = _band_items(params.get("items"), tail)
    if params.get("material") is not None:
        mats.append(str(params["material"]))
    item_mats = [it["material"] or params.get("itemMaterial") for it in items]
    if any(item_mats) and not mats:
        raise MifOpError("an item has a material, so the band needs one too (material). %s" % tail)
    for mname in item_mats:
        if mname and mname not in mats:
            mats.append(mname)
    center = params.get("center")
    if center is not None and (not isinstance(center, (list, tuple)) or len(center) != 2):
        raise MifOpError("center is [x, y] (world), got %r. %s" % (center, tail))

    bm = evaluated_world(around, evaluated)
    try:
        if not len(bm.faces):
            raise MifOpError("'%s' has no faces to wrap. %s" % (around.name, tail))
        if center is None:
            inside = [v.co for v in bm.verts if z0 <= v.co.z <= z1]
            if not inside:
                raise MifOpError("'%s' has no vertex between z0 %g and z1 %g - pass center. %s"
                                 % (around.name, z0, z1, tail))
            cx = (min(p.x for p in inside) + max(p.x for p in inside)) / 2
            cy = (min(p.y for p in inside) + max(p.y for p in inside)) / 2
        else:
            cx, cy = finite_floats(list(center), "center")
        far = max(math.hypot(v.co.x - cx, v.co.y - cy) for v in bm.verts) + 0.1
        tree = BVHTree.FromBMesh(bm)
        heights = [z0 + (z1 - z0) * j / (samples - 1) for j in range(samples)]
        radii, misses = [], []
        for k in range(segs):
            a = 2 * math.pi * k / segs
            e = Vector((math.cos(a), math.sin(a), 0.0))
            col = []
            for z in heights:
                hit = tree.ray_cast(Vector((cx, cy, z)) + e * far, -e, far)
                if hit[0] is None:
                    misses.append((round(math.degrees(a), 3), round(z, 5)))
                    col.append(0.0)
                else:
                    col.append(far - hit[3])
            radii.append(col)
    finally:
        bm.free()
    if misses:
        raise MifOpError("%d of %d rays toward the axis (%.4f, %.4f) met nothing of '%s' (first at angle %s, z %s) - "
                         "the band's height leaves the surface there. %s"
                         % (len(misses), segs * samples, cx, cy, around.name, misses[0][0], misses[0][1], tail))
    rows = [z0, z1] if stiff else heights
    meas = [[max(col)] * 2 for col in radii] if stiff else [list(col) for col in radii]
    rad = [list(c) for c in meas]
    for _ in range(passes):
        rad = [[max(meas[k][j], (rad[k - 1][j] + 2 * rad[k][j] + rad[(k + 1) % segs][j]) / 4.0)
                for j in range(len(rows))] for k in range(segs)]
    nr = len(rows)
    verts, faces, uvs = [], [], []

    def vi(wall, j, k):
        return wall * nr * segs + j * segs + (k % segs)

    for wall, add in ((0, offset), (1, offset + thick)):
        for j, z in enumerate(rows):
            for k in range(segs):
                a = 2 * math.pi * k / segs
                r = rad[k][j] + add
                verts.append([cx + r * math.cos(a), cy + r * math.sin(a), z])
    circ = 2 * math.pi * (sum(rad[k][0] for k in range(segs)) / segs + offset + thick)
    for k in range(segs):
        u0, u1 = circ * k / segs, circ * (k + 1) / segs
        for j in range(nr - 1):
            v0, v1 = rows[j] - z0, rows[j + 1] - z0
            faces.append([vi(0, j, k), vi(0, j + 1, k), vi(0, j + 1, k + 1), vi(0, j, k + 1)])
            uvs.append([[u0, v0], [u0, v1], [u1, v1], [u1, v0]])
            faces.append([vi(1, j, k), vi(1, j, k + 1), vi(1, j + 1, k + 1), vi(1, j + 1, k)])
            uvs.append([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
        top, bot = nr - 1, 0
        faces.append([vi(0, top, k), vi(1, top, k), vi(1, top, k + 1), vi(0, top, k + 1)])
        uvs.append([[u0, 0], [u0, thick], [u1, thick], [u1, 0]])
        faces.append([vi(0, bot, k), vi(0, bot, k + 1), vi(1, bot, k + 1), vi(1, bot, k)])
        uvs.append([[u0, 0], [u1, 0], [u1, thick], [u0, thick]])
    from .ops_kit import _signed_volume   # noqa: PLC0415 - one signed volume for every builder
    vol = _signed_volume(verts, faces)
    turned = vol < 0
    if turned:
        faces = [list(reversed(f)) for f in faces]
        uvs = [list(reversed(u)) for u in uvs]
        vol = -vol
    slot = [0] * len(faces)
    placed = []
    zmid = (z0 + z1) / 2
    for it, mname in zip(items, item_mats):
        w, h, d = it["size"]
        zc = zmid + it["dz"]
        a = math.radians(it["angle"])
        span = math.degrees((w / 2) / max(_band_radius(rad, rows, it["angle"], zc) + offset + thick, 1e-6))
        step = 360.0 / segs
        angs = [it["angle"] - span, it["angle"] + span] + [
            step * kk for kk in range(int(math.floor((it["angle"] - span) / step)),
                                      int(math.ceil((it["angle"] + span) / step)) + 1)
            if it["angle"] - span <= step * kk <= it["angle"] + span]
        zs = [zc - h / 2, zc + h / 2] + [z for z in rows if zc - h / 2 <= z <= zc + h / 2]
        back = max(_band_radius(rad, rows, ang, z) for ang in angs for z in zs) + offset + thick
        e = Vector((math.cos(a), math.sin(a), 0.0))
        t = Vector((-math.sin(a), math.cos(a), 0.0))
        base = len(verts)
        for dr, dt, dzz in ((0, -1, -1), (0, 1, -1), (0, 1, 1), (0, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1),
                            (1, -1, 1)):
            p = Vector((cx, cy, zc)) + e * (back + dr * d) + t * (dt * w / 2) + Vector((0, 0, dzz * h / 2))
            verts.append(list(p))
        for q in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
            faces.append([base + i for i in q])
            uvs.append([[0, 0], [1, 0], [1, 1], [0, 1]])
            slot.append(mats.index(mname) if mname else 0)
        placed.append({"angle": it["angle"], "dz": it["dz"], "size": rnd(it["size"]), "center": rnd(
            Vector((cx, cy, zc)) + e * (back + d / 2)), "backRadius": round(back, 6), "material": mname})
    out = build_part(params, verts, faces, uvs, smooth, "Band", "build_band", mats or None, slot if mats else None)
    widest = [max(c) for c in radii]
    out.update({"around": around.name, "center": rnd([cx, cy]), "z0": z0, "z1": z1, "stiff": stiff, "rows": nr,
                "segments": segs, "rays": segs * samples,
                "surfaceRadius": {"min": round(min(widest), 6), "max": round(max(widest), 6)},
                "innerRadius": {"min": round(min(min(c) for c in rad) + offset, 6),
                                "max": round(max(max(c) for c in rad) + offset, 6)},
                "slant": round(max(abs(c[-1] - c[0]) for c in rad), 6),
                "taperFollowed": round(max(abs(c[-1] - c[0]) for c in radii), 6),
                "items": placed, "outwardCheck": {"signedVolume": round(vol, 10), "turnedOver": turned}})
    return out


def _band_radius(rad, rows, angle_deg, z):
    """The band's measured radius (before offset) at an angle and height, linear between angles and rows."""
    segs = len(rad)
    t = (angle_deg % 360.0) / (360.0 / segs)
    k = int(math.floor(t))
    f = t - k
    zc = min(max(z, rows[0]), rows[-1])
    j = 0
    while j < len(rows) - 2 and zc > rows[j + 1]:
        j += 1
    g = (zc - rows[j]) / (rows[j + 1] - rows[j])

    def at(kk):
        c = rad[kk % segs]
        return c[j] * (1 - g) + c[j + 1] * g
    return at(k) * (1 - f) + at(k + 1) * f


# --------------------------------------------------------------------------------------------- quilt_surface

_QUILT_KEYS = ("object", "mode", "axis", "across", "cellHeight", "cellWidth", "start", "startAcross", "pinch",
               "inflation", "group", "puffGroup", "pressure", "pinSeams")
_AXES = {"X": 0, "Y": 1, "Z": 2}


def _quilt_planes(lo, hi, step, start, key):
    first = lo + step if start is None else start
    k0 = math.ceil((lo + 1e-9 - first) / step)
    out = []
    k = k0
    while True:
        p = first + k * step
        if p >= hi - 1e-9:
            break
        if p > lo + 1e-9:
            out.append(p)
        k += 1
        if len(out) > 500:
            raise MifOpError("%s makes over 500 seam planes across the mesh. NOTHING was changed." % key)
    return out


def _quilt_params(params, obj):
    mode = str(take(params, "mode", default="BANDS", kind=str)).upper()
    if mode not in ("BANDS", "GRID"):
        raise MifOpError("mode must be BANDS or GRID, got %r. NOTHING was changed." % mode)
    axis = str(take(params, "axis", default="Z", kind=str)).upper()
    across = str(take(params, "across", default="X" if axis != "X" else "Y", kind=str)).upper()
    if axis not in _AXES or across not in _AXES or axis == across:
        raise MifOpError("axis and across are two different world axes of X, Y, Z (got %r, %r). NOTHING was changed."
                         % (axis, across))
    if mode == "BANDS" and ("cellWidth" in params or "startAcross" in params or "across" in params):
        raise MifOpError("cellWidth, across and startAcross shape GRID cells; mode is BANDS. NOTHING was changed.")
    ch = take_float(params, "cellHeight", required=True)
    cw = take_float(params, "cellWidth", default=ch if mode == "GRID" else None)
    for k, v in (("cellHeight", ch), ("cellWidth", cw)):
        if v is not None and not (math.isfinite(v) and v > 0):
            raise MifOpError("%s must be above 0, got %r. NOTHING was changed." % (k, v))
    start = take_float(params, "start", default=None)
    start_x = take_float(params, "startAcross", default=None)
    pinch = take_float(params, "pinch", default=0.0)
    infl = take_float(params, "inflation", default=0.0)
    for k, v, cap in (("pinch", pinch, ch / 2), ("inflation", infl, ch)):
        if not (math.isfinite(v) and 0 <= v <= cap):
            raise MifOpError("%s must be 0 to %g (half a cell for pinch, a cell for inflation), got %r. NOTHING was "
                             "changed." % (k, cap, v))
    group = str(take(params, "group", default="QuiltSeams", kind=str))
    puff = str(take(params, "puffGroup", default="QuiltPuff", kind=str))
    if not group.strip() or not puff.strip() or group == puff:
        raise MifOpError("group and puffGroup must be two different non-empty names. NOTHING was changed.")
    for g in (group, puff):
        if obj.vertex_groups.get(g) is not None:
            raise MifOpError("'%s' already has a vertex group '%s' - it is quilted already, or name other groups. "
                             "NOTHING was changed." % (obj.name, g))
    pressure = take_float(params, "pressure", default=None)
    pin = take_bool(params, "pinSeams", default=True)
    cloth = next((m for m in obj.modifiers if m.type == "CLOTH"), None)
    if pressure is not None:
        if not math.isfinite(pressure):
            raise MifOpError("pressure must be finite, got %r. NOTHING was changed." % pressure)
        if cloth is None:
            raise MifOpError("pressure is set on the object's cloth modifier and '%s' has none - add_cloth first. "
                             "NOTHING was changed." % obj.name)
    return mode, axis, across, ch, cw, start, start_x, pinch, infl, group, puff, pressure, pin, cloth


def op_quilt_surface(params):
    """Divide a garment into quilted cells by seam lines - horizontal BANDS or a GRID - pinch the seams in, puff the
    cells out, and hand the seams and the puff to the cloth as its pin and pressure groups.

    THE GAP (spec "Quilted puffs: divide a garment into cells by pinching seam lines", 2026-10-06): a puffer jacket's
    quilted cells are made by pinning or pinching seam lines and inflating each cell; vertex groups and add_cloth
    pressure existed, nothing divided the surface. Seams are cut as exact edge loops (bisect) on world planes across
    `axis` (default Z) every cellHeight from `start` (default one cell above the mesh's lowest point) and, for GRID,
    across `across` (default X) every cellWidth (default cellHeight) from startAcross. Seam vertices go in `group`
    (default QuiltSeams, weight 1) and move pinch (world units) against their normal; every other vertex goes in
    puffGroup (default QuiltPuff) at a dome weight - smoothstep of its distance to the nearest seam plane over half a
    cell, 0 at a seam and 1 mid-cell - and moves inflation x that weight along its normal. pressure, on an object
    that has a cloth modifier, turns pressure on at that force, scaled by puffGroup, with the seams as the pin group
    (pinSeams, default true).

    Reply: seam planes, seam vertices, cells (faces joined across non-seam edges) with their face counts and areas,
    cellsWithoutInterior (cells no vertex can puff - the mesh is too coarse), the largest pinch and puff moves, the
    groups and the cloth settings written. Refused: a bad mode, axis or size, no plane crossing the mesh, a group
    already on the object, pressure without a cloth modifier, shape keys.
    """
    reject_unknown(params, _QUILT_KEYS, "quilt_surface")
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    (mode, axis, across, ch, cw, start, start_x, pinch, infl, group, puff, pressure, pin,
     cloth) = _quilt_params(params, obj)
    ai, xi = _AXES[axis], _AXES[across]
    bm, shift = _world_edit(obj, "quilt_surface")
    try:
        if not len(bm.faces):
            raise MifOpError("'%s' has no faces. NOTHING was changed." % obj.name)
        cos = [v.co for v in bm.verts]
        sets = [(ai, _quilt_planes(min(c[ai] for c in cos), max(c[ai] for c in cos), ch,
                                   None if start is None else start - shift[ai], "cellHeight"))]
        if mode == "GRID":
            sets.append((xi, _quilt_planes(min(c[xi] for c in cos), max(c[xi] for c in cos), cw,
                                           None if start_x is None else start_x - shift[xi], "cellWidth")))
        if not sets[0][1] or (mode == "GRID" and not sets[1][1]):
            raise MifOpError("no seam plane crosses '%s' - its extent along %s is under one cell (cellHeight %g%s). "
                             "NOTHING was changed." % (obj.name, axis if not sets[0][1] else across, ch,
                                                       "" if mode == "BANDS" else ", cellWidth %g" % cw))
        normals_before = {}
        for idx, planes in sets:
            no = Vector((0, 0, 0))
            no[idx] = 1.0
            for p in planes:
                co = Vector((0, 0, 0))
                co[idx] = p
                bmesh.ops.bisect_plane(bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces), dist=1e-6,
                                       plane_co=co, plane_no=no)
        bm.normal_update()
        bm.verts.index_update()

        def plane_dist(v):
            return min(min(abs(v.co[idx] - p) for p in planes) for idx, planes in sets)

        seam = {v for v in bm.verts if plane_dist(v) < 1e-5}

        def on_same_plane(e):
            a, b = e.verts
            return any(abs(a.co[idx] - p) < 1e-5 and abs(b.co[idx] - p) < 1e-5 for idx, planes in sets for p in planes)
        seam_edges = {e for e in bm.edges if e.verts[0] in seam and e.verts[1] in seam and on_same_plane(e)}
        cells, seen = [], set()
        for f in bm.faces:
            if f in seen:
                continue
            part, stack = [f], [f]
            seen.add(f)
            while stack:
                for e in stack.pop().edges:
                    if e in seam_edges:
                        continue
                    for g in e.link_faces:
                        if g not in seen:
                            seen.add(g)
                            part.append(g)
                            stack.append(g)
            cells.append(part)
        half = {ai: ch / 2.0}
        if mode == "GRID":
            half[xi] = cw / 2.0
        weight = {}
        for v in bm.verts:
            if v in seam:
                weight[v] = 0.0
                continue
            w = 1.0
            for idx, planes in sets:
                d = min(abs(v.co[idx] - p) for p in planes)
                w = min(w, _smoothstep(d / half[idx]))
            weight[v] = w
        normals_before = {v: v.normal.copy() for v in bm.verts}
        for v in bm.verts:
            if v in seam:
                v.co -= normals_before[v] * pinch
            else:
                v.co += normals_before[v] * (infl * weight[v])
        no_interior = sum(1 for c in cells if not any(v not in seam for f in c for v in f.verts))
        cell_rows = [{"faces": len(c), "area": round(sum(f.calc_area() for f in c), 6)} for c in cells]
        _write_world(bm, obj, shift)
        seam_idx = sorted(v.index for v in seam)
        puff_rows = [(v.index, weight[v]) for v in bm.verts if v not in seam]
        n_planes = [len(planes) for _i, planes in sets]
    finally:
        bm.free()
    sg = obj.vertex_groups.new(name=group)
    sg.add(seam_idx, 1.0, "REPLACE")
    pg = obj.vertex_groups.new(name=puff)
    for i, w in puff_rows:
        if w > 0:
            pg.add([i], w, "REPLACE")
    cloth_out = None
    if pressure is not None:
        st = cloth.settings
        st.use_pressure = True
        st.uniform_pressure_force = pressure
        st.vertex_group_pressure = pg.name
        if pin:
            st.vertex_group_mass = sg.name
        cloth_out = {"modifier": cloth.name, "usePressure": bool(st.use_pressure),
                     "pressure": round(float(st.uniform_pressure_force), 6), "pressureGroup": st.vertex_group_pressure,
                     "pinGroup": st.vertex_group_mass or None}
    obj.data.update()
    out = {"object": obj.name, "mode": mode, "axis": axis, "across": across if mode == "GRID" else None,
           "cellHeight": ch, "cellWidth": cw if mode == "GRID" else None, "seamPlanes": n_planes,
           "seamVertices": len(seam_idx), "cellCount": len(cells), "cells": cell_rows[:200],
           "cellsWithoutInterior": no_interior, "pinch": pinch, "inflation": infl,
           "puffWeightMax": round(max([w for _i, w in puff_rows] or [0.0]), 6), "groups": [sg.name, pg.name],
           "cloth": cloth_out, "after": mesh_counts(obj)}
    if no_interior:
        out["coarseNote"] = ("%d cell(s) have no vertex off their seams, so nothing in them can puff - subdivide the "
                             "mesh to several rows per cell first." % no_interior)
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- add_accessory_physics

_ACC_KEYS = ("object", "body", "cageDensity", "margin", "pinNear", "pinRadius", "pinWithin", "collisionWith", "cloth",
             "name", "group")
_ACC_CLOTH_KEYS = ("quality", "mass", "stiffness", "damping", "gravity", "usePressure", "pressure", "collisionQuality",
                   "selfCollision", "pinStiffness", "preset", "bending", "bendingDamping", "bendingModel", "bendingMax",
                   "pressureFactor")


def _acc_params(params, obj):
    tail = "NOTHING was changed."
    body = get_object(take(params, "body", required=True, kind=str), want_mesh=True)
    if body is obj:
        raise MifOpError("body is the accessory itself; name the mesh it rests on. %s" % tail)
    dens = take_float(params, "cageDensity", default=0.02)
    if not (math.isfinite(dens) and 0.001 <= dens <= 1.0):
        raise MifOpError("cageDensity (the cage's voxel size, world units) must be 0.001-1, got %r. %s" % (dens, tail))
    margin = take_float(params, "margin", default=dens / 2.0)
    if not (math.isfinite(margin) and 0 <= margin <= 4 * dens):
        raise MifOpError("margin must be 0 to 4 x cageDensity (%g), got %r. %s" % (4 * dens, margin, tail))
    pin_r = take_float(params, "pinRadius", default=dens * 1.5)
    if not (math.isfinite(pin_r) and pin_r > 0):
        raise MifOpError("pinRadius must be above 0, got %r. %s" % (pin_r, tail))
    within = take_float(params, "pinWithin", default=None)
    if within is not None and not (math.isfinite(within) and within > 0):
        raise MifOpError("pinWithin (distance to the body's surface) must be above 0, got %r. %s" % (within, tail))
    raw = params.get("pinNear")
    points = []
    if raw is not None:
        if not isinstance(raw, (list, tuple)) or not raw or not all(isinstance(p, (list, tuple)) and len(p) == 3
                                                                    for p in raw):
            raise MifOpError("pinNear is a list of world points [[x,y,z], ...], got %r. %s" % (raw, tail))
        points = [Vector(finite_floats(list(p), "pinNear")) for p in raw]
    if not points and within is None:
        raise MifOpError("a cage with no pin falls off the body: pass pinNear [[x,y,z], ...], pinWithin (a distance to "
                         "the body), or both. %s" % tail)
    others = []
    for n in params.get("collisionWith") or []:
        o = get_object(str(n), want_mesh=True)
        if o is obj:
            raise MifOpError("collisionWith names the accessory itself; its cage collides through self-collision. %s"
                             % tail)
        others.append(o)
    if params.get("collisionWith") is not None and not isinstance(params.get("collisionWith"), (list, tuple)):
        raise MifOpError("collisionWith is a list of object names. %s" % tail)
    cloth = params.get("cloth") or {}
    if not isinstance(cloth, dict) or set(cloth) - set(_ACC_CLOTH_KEYS):
        raise MifOpError("cloth takes add_cloth's settings (%s); got %r. %s" % (", ".join(_ACC_CLOTH_KEYS), cloth, tail))
    name = str(take(params, "name", default=obj.name + "Cage", kind=str))
    if name in bpy.data.objects:
        raise MifOpError("an object named '%s' already exists; pass another name. %s" % (name, tail))
    group = str(take(params, "group", default="Pin", kind=str))
    if not group.strip():
        raise MifOpError("group must be a non-empty name. %s" % tail)
    if any(m.type == "SURFACE_DEFORM" for m in obj.modifiers):
        raise MifOpError("'%s' already has a Surface Deform modifier - it follows a cage already. %s" % (obj.name, tail))
    return body, dens, margin, pin_r, within, points, others, cloth, name, group


def _acc_cage(obj, dens, margin, name):
    """A closed low-poly cage round the accessory: its base mesh pushed margin out along the normals, voxel-remeshed
    at dens. Returns the linked cage object (world transform of the accessory)."""
    me = obj.data.copy()
    me.name = name
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bm.normal_update()
        sx = max(1e-9, min(abs(s) for s in obj.matrix_world.to_scale()))
        for v in bm.verts:
            v.co += v.normal * (margin / sx)
        bm.to_mesh(me)
    finally:
        bm.free()
    cage = bpy.data.objects.new(name, me)
    for coll in obj.users_collection:
        coll.objects.link(cage)
    cage.matrix_world = obj.matrix_world.copy()
    mod = cage.modifiers.new("Remesh", "REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = dens / sx
    mod.adaptivity = 0.0
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    new = bpy.data.meshes.new_from_object(cage.evaluated_get(dg))
    cage.modifiers.remove(mod)
    cage.data = new
    bpy.data.meshes.remove(me)
    new.name = name
    return cage


def _acc_rollback(cage, obj, deform_name):
    if deform_name and obj.modifiers.get(deform_name) is not None:
        obj.modifiers.remove(obj.modifiers[deform_name])
    if cage is not None:
        me = cage.data
        remove_objects([cage])
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)


def op_add_accessory_physics(params):
    """Simulate a chain, necklace or lace through a low-poly cage: the cage is built round it, pinned where it rests,
    run as cloth against the body, and the detailed accessory follows it through a bound Surface Deform.

    THE GAP (spec "A low-poly cage that simulates a chain, necklace or lace", 2026-10-06): a cage along the chain, run
    as cloth with pins where it rests on the body, drives the detailed accessory through a Surface Deform; two chains
    that cross need collision between them. Each piece existed (add_cloth, set_vertex_weights, add_collision,
    bind_modifier) and nothing built the cage. The cage is the accessory's base mesh pushed `margin` (default half of
    cageDensity) out along its normals and voxel-remeshed at cageDensity (world units) - one closed shell that fuses
    the links. Pins (vertex group `group`, default Pin): cage vertices within pinRadius (default 1.5 x cageDensity)
    of a pinNear point, and/or within pinWithin of the body's surface. The accessory gets a Surface Deform bound to the
    cage at rest; then the cage gets cloth (`cloth` takes add_cloth's settings) pinned by the group; the body, and
    each object in collisionWith (another chain's cage), get a Collision modifier where they have none.

    Reply: the cage (name, vertices, faces, voxel size, margin), pins (count, per point, by the body), the cloth, the
    bind and the collisions. Refused, with nothing left behind: no pin given, a pin point farther than pinRadius +
    margin + cageDensity from the accessory, a cage that holds no pin or comes out empty, a taken name, an accessory
    already on a Surface Deform, a cloth setting add_cloth refuses, a bind Blender refuses.
    """
    reject_unknown(params, _ACC_KEYS, "add_accessory_physics")
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    body, dens, margin, pin_r, within, points, others, cloth, name, group = _acc_params(params, obj)
    from mathutils.bvhtree import BVHTree   # noqa: PLC0415 - after the refusals: the offline sweep stubs mathutils
    require_editable(obj, "edit")
    bm = evaluated_world(obj, evaluated=False)
    try:
        if not len(bm.faces):
            raise MifOpError("'%s' has no faces to build a cage round. NOTHING was changed." % obj.name)
        src_tree = BVHTree.FromBMesh(bm)
    finally:
        bm.free()
    reach = pin_r + margin + dens
    for p in points:
        hit = src_tree.find_nearest(p)
        if hit[0] is None or hit[3] > reach:
            raise MifOpError("pinNear %s is %s from '%s' - farther than pinRadius + margin + cageDensity (%g), so it "
                             "would pin nothing. NOTHING was changed." % (
                                 [round(x, 4) for x in p], "unmeasured" if hit[0] is None else "%.4g" % hit[3],
                                 obj.name, reach))
    cage, deform = None, None
    try:
        cage = _acc_cage(obj, dens, margin, name)
        if len(cage.data.polygons) < 4:
            raise MifOpError("the cage of '%s' at cageDensity %g came out empty (%d faces) - the accessory is thinner "
                             "than a voxel; raise margin or lower cageDensity. NOTHING was changed."
                             % (obj.name, dens, len(cage.data.polygons)))
        mw = cage.matrix_world
        world = [mw @ v.co for v in cage.data.vertices]
        per_point = []
        pinned = set()
        for p in points:
            hit = {i for i, w in enumerate(world) if (w - p).length <= pin_r}
            per_point.append(len(hit))
            pinned |= hit
        by_body = 0
        if within is not None:
            bb = evaluated_world(body, evaluated=True)
            try:
                btree = BVHTree.FromBMesh(bb)
            finally:
                bb.free()
            near_body = set()
            for i, w in enumerate(world):
                h = btree.find_nearest(w)
                if h[0] is not None and h[3] <= within:
                    near_body.add(i)
            by_body = len(near_body)
            pinned |= near_body
        if not pinned:
            raise MifOpError("no cage vertex lies within pinRadius %g of a pinNear point%s - the cage would fall. "
                             "NOTHING was changed." % (pin_r, "" if within is None else
                                                       " or within pinWithin %g of the body" % within))
        vg = cage.vertex_groups.new(name=group)
        vg.add(sorted(pinned), 1.0, "REPLACE")
        deform = obj.modifiers.new("CageDeform", "SURFACE_DEFORM")
        deform.target = cage
        from .ops_rig import op_bind_modifier   # noqa: PLC0415
        bound = op_bind_modifier({"object": obj.name, "modifier": deform.name})
        from .ops_physics import op_add_cloth   # noqa: PLC0415
        q = dict(cloth)
        q.update({"object": cage.name, "pinGroup": group})
        cl = op_add_cloth(q)
    except MifOpError as exc:
        _acc_rollback(cage, obj, deform.name if deform is not None else None)
        msg = str(exc)
        raise MifOpError(msg if "NOTHING" in msg else msg.rstrip(".") + ". NOTHING was changed - the cage was removed.")
    collisions = {}
    for o in [body] + others:
        if any(m.type == "COLLISION" for m in o.modifiers):
            collisions[o.name] = "present"
        else:
            o.modifiers.new(name="Collision", type="COLLISION")
            collisions[o.name] = "added"
    out = {"object": obj.name, "body": body.name,
           "cage": {"name": cage.name, "vertices": len(cage.data.vertices), "faces": len(cage.data.polygons),
                    "voxelSize": dens, "margin": margin},
           "pins": {"group": group, "count": len(pinned), "perPoint": per_point, "byBody": by_body,
                    "pinRadius": pin_r, "pinWithin": within},
           "cloth": {k: cl.get(k) for k in ("modifier", "quality", "mass", "pinGroup", "bending", "selfCollision")},
           "deform": {"modifier": deform.name, "target": cage.name, "bound": bound.get("bound")},
           "collisions": collisions,
           "note": "step_simulation (or bake_physics) drives the cage; the accessory follows it through %s. Apply "
                   "the deform at a settled frame to keep that shape." % deform.name}
    if any(m.type == "CLOTH" for o in others for m in o.modifiers):
        out["clothCollisionNote"] = ("a cloth in collisionWith collides with this cage through its Collision "
                                     "modifier, as its evaluated surface - Blender has no cloth-to-cloth contact "
                                     "beyond that and self-collision.")
    return out


# --------------------------------------------------------------------------------------------- wrap_to_target

_WRAP_KEYS = ("object", "target", "wrapMethod", "offset", "pins", "landmarks", "landmarkRadius", "smoothIterations",
              "reach", "dryRun", "previewId", "previewTtlSeconds")
_WRAP_METHODS = ("NEAREST_SURFACEPOINT", "PROJECT", "NEAREST_VERTEX")


def _wrap_landmarks(raw, n_verts, tail):
    """[(vertex index or None, from point or None, to point)] from [{vertex | from, to}, ...]."""
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)) or not raw or len(raw) > 500:
        raise MifOpError("landmarks is a list of 1-500 {vertex: i | from: [x,y,z], to: [x,y,z]}. %s" % tail)
    out = []
    for k, lm in enumerate(raw):
        if not isinstance(lm, dict) or "to" not in lm or len({"vertex", "from"} & set(lm)) != 1 \
                or set(lm) - {"vertex", "from", "to"}:
            raise MifOpError("landmarks[%d] is {vertex: i, to: [x,y,z]} or {from: [x,y,z], to: [x,y,z]}, got %r. %s"
                             % (k, lm, tail))
        to = lm["to"]
        if not isinstance(to, (list, tuple)) or len(to) != 3:
            raise MifOpError("landmarks[%d].to must be [x,y,z]. %s" % (k, tail))
        to = Vector(finite_floats(list(to), "landmarks.to"))
        if "vertex" in lm:
            vi = lm["vertex"]
            if isinstance(vi, bool) or not isinstance(vi, int) or not 0 <= vi < n_verts:
                raise MifOpError("landmarks[%d].vertex %r is not a vertex index (the mesh has %d). %s"
                                 % (k, vi, n_verts, tail))
            out.append((vi, None, to))
        else:
            fr = lm["from"]
            if not isinstance(fr, (list, tuple)) or len(fr) != 3:
                raise MifOpError("landmarks[%d].from must be [x,y,z]. %s" % (k, tail))
            out.append((None, Vector(finite_floats(list(fr), "landmarks.from")), to))
    return out


def _edge_cv(bm):
    ls = [e.calc_length() for e in bm.edges]
    if not ls:
        return None
    m = sum(ls) / len(ls)
    return round(math.sqrt(sum((x - m) ** 2 for x in ls) / len(ls)) / m, 6) if m > 0 else None


def op_wrap_to_target(params):
    """Fit a mesh onto another surface while keeping its own topology - a template head or body wrapped onto a scan -
    with landmark pairs moved first, pinned vertices held, and the fit relaxed and re-projected.

    THE GAP (spec "Fit a mesh onto a template topology", 2026-10-06): a custom head or body is conformed by wrapping
    the template-topology mesh onto it; shrinkwrap plus apply with landmark pins reached it in several calls and none
    measured the fit. `object` keeps its vertices, edges and faces and takes `target`'s shape (both world space, base
    meshes, the target as it evaluates). In order: (1) landmarks [{vertex: i | from: [x,y,z], to: [x,y,z]}] move the
    vertex (or the one nearest `from`) to `to`, the move spread over landmarkRadius (world units, default none) with a
    smoothstep falloff; landmark vertices are pinned from then on; (2) every free vertex goes onto the target -
    wrapMethod NEAREST_SURFACEPOINT (default), PROJECT (along its normal, either way, within reach; a miss stays put
    and is counted) or NEAREST_VERTEX - plus offset along the target's normal there; (3) smoothIterations passes move
    each free vertex half way to its neighbours' mean and back onto the target. pins {vertexGroup, minWeight} holds a
    group's vertices where they are.

    Reply: the largest and mean distance to the target before and after, landmark errors, pinned and missed counts,
    the edge-length spread (coefficient of variation) before and after, and the counts (unchanged by construction).
    dryRun measures and writes nothing. Refused: the target being the object, a bad method, pins naming no group or
    holding nothing, a malformed landmark, offset or reach not finite, smoothIterations over 100, shape keys.
    """
    reject_unknown(params, _WRAP_KEYS, "wrap_to_target")
    tail = "NOTHING was changed."
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    tgt = get_object(take(params, "target", required=True, kind=str), want_mesh=True)
    if tgt is obj:
        raise MifOpError("target is the object itself; name the surface to fit onto. %s" % tail)
    method = str(take(params, "wrapMethod", default="NEAREST_SURFACEPOINT", kind=str)).upper()
    if method not in _WRAP_METHODS:
        raise MifOpError("wrapMethod must be one of %s, got %r. %s" % (", ".join(_WRAP_METHODS), method, tail))
    try:
        offset = take_float(params, "offset", default=0.0)
        reach = take_float(params, "reach", default=None)
        lm_r = take_float(params, "landmarkRadius", default=0.0)
        iters = take_int(params, "smoothIterations", default=0)
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
    for k, v in (("offset", offset), ("reach", reach), ("landmarkRadius", lm_r)):
        if v is not None and not math.isfinite(v):
            raise MifOpError("%s must be finite, got %r. %s" % (k, v, tail))
    if reach is not None and reach <= 0 or lm_r < 0:
        raise MifOpError("reach must be above 0 and landmarkRadius at least 0. %s" % tail)
    if not 0 <= iters <= 100:
        raise MifOpError("smoothIterations must be 0-100, got %r. %s" % (iters, tail))
    dry = take_bool(params, "dryRun", default=False)
    landmarks = _wrap_landmarks(params.get("landmarks"), len(obj.data.vertices), tail)
    pin_raw = params.get("pins")
    pin_idx = set()
    if pin_raw is not None:
        if not isinstance(pin_raw, dict) or "vertexGroup" not in pin_raw or set(pin_raw) - {"vertexGroup", "minWeight"}:
            raise MifOpError("pins is {vertexGroup, minWeight}, got %r. %s" % (pin_raw, tail))
        vg = obj.vertex_groups.get(str(pin_raw["vertexGroup"]))
        if vg is None:
            raise MifOpError("pins.vertexGroup %r is not a group of '%s'. %s" % (pin_raw["vertexGroup"], obj.name, tail))
        mw = finite_float(pin_raw.get("minWeight", 0.5), "pins.minWeight")
        pin_idx = {v.index for v in obj.data.vertices for g in v.groups if g.group == vg.index and g.weight >= mw}
        if not pin_idx:
            raise MifOpError("pins.vertexGroup '%s' holds no vertex at weight %g or more. %s" % (vg.name, mw, tail))
    from mathutils.bvhtree import BVHTree   # noqa: PLC0415 - after the refusals: the offline sweep stubs mathutils
    from mathutils.kdtree import KDTree   # noqa: PLC0415
    tb = evaluated_world(tgt, evaluated=True)
    try:
        if not len(tb.faces):
            raise MifOpError("'%s' has no faces to fit onto. %s" % (tgt.name, tail))
        tree = BVHTree.FromBMesh(tb)
        kd = KDTree(len(tb.verts))
        for v in tb.verts:
            kd.insert(v.co, v.index)
        kd.balance()
        tverts = [(v.co.copy(), v.normal.copy()) for v in tb.verts]
    finally:
        tb.free()
    bm, shift = _world_edit(obj, "wrap_to_target", writes=not dry)
    try:
        verts = list(bm.verts)
        pts = [v.co + shift for v in verts]   # world

        def dist(p):
            h = tree.find_nearest(p)
            return h[3] if h[0] is not None else math.inf
        before = [dist(p) for p in pts]
        cv0 = _edge_cv(bm)
        # (1) landmarks
        lm_rows, lm_pin = [], set()
        for vi, fr, to in landmarks:
            if vi is None:
                vi = min(range(len(pts)), key=lambda i, fr=fr: (pts[i] - fr).length)
            lm_rows.append((vi, to))
            lm_pin.add(vi)
        for vi, to in lm_rows:
            d = to - pts[vi]
            if lm_r > 0:
                src = pts[vi].copy()
                for i, p in enumerate(pts):
                    if i in lm_pin:
                        continue
                    r = (p - src).length
                    if r < lm_r:
                        pts[i] = p + d * (1.0 - _smoothstep(r / lm_r))
            pts[vi] = to.copy()
        held = pin_idx | lm_pin
        normals = [v.normal.copy() for v in verts]

        def onto(i, p):
            if method == "NEAREST_VERTEX":
                co, ti, _d = kd.find(p)
                return co + tverts[ti][1] * offset
            if method == "PROJECT":
                n = normals[i]
                hits = [tree.ray_cast(p, d, reach) if reach else tree.ray_cast(p, d) for d in (n, -n)]
                hits = [h for h in hits if h[0] is not None]
                if not hits:
                    return None
                h = min(hits, key=lambda h: h[3])
                return h[0] + h[1] * offset
            h = tree.find_nearest(p)
            return None if h[0] is None else h[0] + h[1] * offset
        missed = 0
        for i in range(len(pts)):
            if i in held:
                continue
            q = onto(i, pts[i])
            if q is None:
                missed += 1
            else:
                pts[i] = q
        nbrs = [[e.other_vert(v).index for e in v.link_edges] for v in verts]
        for _k in range(iters):
            new = list(pts)
            for i, nb in enumerate(nbrs):
                if i in held or not nb:
                    continue
                mean = sum((pts[j] for j in nb), Vector()) / len(nb)
                half = pts[i] + (mean - pts[i]) * 0.5
                if method == "NEAREST_VERTEX":
                    q = onto(i, half)
                else:   # back onto the nearest surface point - a relaxed vertex has no normal ray of its own
                    h = tree.find_nearest(half)
                    q = None if h[0] is None else h[0] + h[1] * offset
                if q is not None:
                    new[i] = q
            pts = new
        after = [dist(p) for p in pts]
        lm_err = [round((pts[vi] - to).length, 9) for vi, to in lm_rows]
        for v, p in zip(verts, pts):
            v.co = p - shift
        cv1 = _edge_cv(bm)
        if not dry:
            _write_world(bm, obj, shift)
    finally:
        bm.free()
    fin = [d for d in before if math.isfinite(d)]
    out = {"object": obj.name, "target": tgt.name, "wrapMethod": method, "offset": offset, "dryRun": dry,
           "before": {"maxDistance": round(max(fin), 6) if fin else None,
                      "meanDistance": round(sum(fin) / len(fin), 6) if fin else None},
           "after": {"maxDistance": round(max(after), 6), "meanDistance": round(sum(after) / len(after), 6)},
           "landmarks": [{"vertex": vi, "error": e} for (vi, _t), e in zip(lm_rows, lm_err)],
           "pinned": len(held), "missed": missed, "smoothIterations": iters,
           "edgeLengthCV": {"before": cv0, "after": cv1}, "counts": mesh_counts(obj)}
    if missed:
        out["missedNote"] = ("%d vertex/vertices found no target surface along their normal%s and stayed put."
                             % (missed, "" if reach is None else " within reach %g" % reach))
    if not dry:
        out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- retopo_panels

_RETOPO_KEYS = ("panels", "edgeLength", "seams", "rings", "uvMap", "name", "collection", "takeOverName")


def _rt_resample(pts, n):
    """n + 1 points along the polyline pts at equal arc-length steps (both ends kept)."""
    acc = [0.0]
    for a, b in zip(pts, pts[1:]):
        acc.append(acc[-1] + (b - a).length)
    total = acc[-1]
    out, k = [], 0
    for i in range(n + 1):
        t = total * i / n
        while k < len(pts) - 2 and acc[k + 1] < t:
            k += 1
        seg = acc[k + 1] - acc[k]
        f = (t - acc[k]) / seg if seg > 0 else 0.0
        out.append(pts[k].lerp(pts[k + 1], min(max(f, 0.0), 1.0)))
    out[0], out[-1] = pts[0].copy(), pts[-1].copy()
    return out


def _rt_poly_len(pts):
    return sum((b - a).length for a, b in zip(pts, pts[1:]))


def _rt_seg_dist(p, pts):
    from mathutils.geometry import intersect_point_line   # noqa: PLC0415
    best = math.inf
    for a, b in zip(pts, pts[1:]):
        q, f = intersect_point_line(p, a, b)
        q = a if f < 0 else (b if f > 1 else q)
        best = min(best, (p - q).length)
    return best


def _rt_panel_specs(raw, tail):
    """[(object, edgeLength or None, corners or None)] from panels [name | {object, edgeLength, corners}]."""
    if not isinstance(raw, (list, tuple)) or not raw or len(raw) > 64:
        raise MifOpError("panels is a list of 1-64 panel object names or {object, edgeLength, corners}. %s" % tail)
    out, seen = [], set()
    for k, row in enumerate(raw):
        el = corners = None
        if isinstance(row, dict):
            if "object" not in row or set(row) - {"object", "edgeLength", "corners"}:
                raise MifOpError("panels[%d] is {object, edgeLength, corners}, got %r. %s" % (k, row, tail))
            name = row["object"]
            if row.get("edgeLength") is not None:
                el = finite_float(row["edgeLength"], "panels[%d].edgeLength" % k)
                if not el > 0:
                    raise MifOpError("panels[%d].edgeLength must be above 0. %s" % (k, tail))
            if row.get("corners") is not None:
                c = row["corners"]
                if not isinstance(c, (list, tuple)) or len(c) != 4 or any(
                        not isinstance(x, (list, tuple)) or len(x) != 3 for x in c):
                    raise MifOpError("panels[%d].corners is four [x,y,z] points on the outline. %s" % (k, tail))
                corners = [Vector(finite_floats(list(x), "corners")) for x in c]
        else:
            name = row
        if not isinstance(name, str):
            raise MifOpError("panels[%d] names no object: %r. %s" % (k, row, tail))
        obj = get_object(name, want_mesh=True)
        if obj.name in seen:
            raise MifOpError("panel '%s' is listed twice. %s" % (obj.name, tail))
        seen.add(obj.name)
        out.append((obj, el, corners))
    return out


def _rt_outline(obj, corners_at, tail):
    """(four sides [[world Vector]] c0->c1->c2->c3->c0 wound with the panel's faces, corner points, holes [[world
    Vector]] wound the same way, plane normal) of a flat panel; corners are the four sharpest turns unless given."""
    from .ops_topology import _boundary_loops_of, _ordered   # noqa: PLC0415
    bm = evaluated_world(obj, evaluated=False)
    try:
        if not len(bm.faces):
            raise MifOpError("panel '%s' has no faces. %s" % (obj.name, tail))
        loops = [_ordered(lp) for lp in _boundary_loops_of(bm)]
        loops = [[v.co.copy() for v in lp] for lp in loops if lp and len(lp) >= 3]
        fn = sum((f.normal * f.calc_area() for f in bm.faces), Vector())
        area = sum(f.calc_area() for f in bm.faces)
    finally:
        bm.free()
    if not loops:
        raise MifOpError("panel '%s' has no open outline (a closed mesh is not a flat pattern panel). %s"
                         % (obj.name, tail))
    loop = max(loops, key=lambda lp: _rt_poly_len(lp + lp[:1]))
    n = fn.normalized() if fn.length > 0 else Vector((0, 0, 1))
    c = sum(loop, Vector()) / len(loop)
    size = max((p - c).length for p in loop)
    off = max(abs((p - c).dot(n)) for p in loop)
    if fn.length < 0.5 * area or off > 1e-4 * max(size, 1e-9):
        raise MifOpError("panel '%s' is not flat (its outline stands %.6g off its plane) - retopo_panels works on the "
                         "flat pattern; transfer_by_uv then moves the retopo onto the drape. %s" % (obj.name, off, tail))
    if _newell_signed(loop, n) < 0:
        loop.reverse()
    holes = [lp for lp in loops if lp is not loop]
    for h in holes:
        if _newell_signed(h, n) < 0:
            h.reverse()
    m = len(loop)
    if corners_at:
        idx = [min(range(m), key=lambda i, q=q: (loop[i] - q).length) for q in corners_at]
        if len(set(idx)) != 4:
            raise MifOpError("panel '%s': two corners land on the same outline vertex. %s" % (obj.name, tail))
    else:
        def turn(i):
            a, b = loop[i] - loop[i - 1], loop[(i + 1) % m] - loop[i]
            return a.angle(b, 0.0) if a.length and b.length else 0.0
        idx = sorted(range(m), key=turn, reverse=True)[:4]
        if len(idx) < 4 or turn(idx[3]) < math.radians(20):
            raise MifOpError("panel '%s' has fewer than four outline turns of 20 degrees or more - pass its corners "
                             "[[x,y,z] x4]. %s" % (obj.name, tail))
    idx.sort()
    sides = []
    for k in range(4):
        a, b = idx[k], idx[(k + 1) % 4]
        run = [loop[i % m] for i in range(a, b + 1 if b > a else b + m + 1)]
        sides.append(run)
    return sides, [loop[i] for i in idx], holes, n


def _newell_signed(loop, n):
    s = Vector()
    for a, b in zip(loop, loop[1:] + loop[:1]):
        s += a.cross(b)
    return s.dot(n)


def _rt_find(par, x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


def _rt_coons(bot, right, top, left):
    """Rows (bottom to top) of a Coons patch: bot and top run left to right, left and right bottom to top."""
    nu, nv = len(bot) - 1, len(left) - 1
    c0, c1, c2, c3 = bot[0], bot[-1], top[-1], top[0]
    rows = []
    for j in range(nv + 1):
        v = j / nv
        row = []
        for i in range(nu + 1):
            u = i / nu
            if j == 0:
                p = bot[i]
            elif j == nv:
                p = top[i]
            elif i == 0:
                p = left[j]
            elif i == nu:
                p = right[j]
            else:
                p = (bot[i] * (1 - v) + top[i] * v + left[j] * (1 - u) + right[j] * u
                     - (c0 * ((1 - u) * (1 - v)) + c1 * (u * (1 - v)) + c2 * (u * v) + c3 * ((1 - u) * v)))
            row.append(p)
        rows.append(row)
    return rows


def _rt_offset_dirs(pts, n, closed, sign):
    """Per point, the in-plane direction that moves it one unit off the polyline's segments (mitred): sign 1 is to
    the left of travel seen down n, -1 to the right."""
    m, out = len(pts), []
    for i in range(m):
        segs = []
        if closed or i > 0:
            segs.append(pts[i] - pts[i - 1])
        if closed or i < m - 1:
            segs.append(pts[(i + 1) % m] - pts[i])
        ns = [n.cross(s).normalized() * sign for s in segs if s.length > 1e-12]
        if not ns:
            out.append(Vector())
            continue
        avg = sum(ns, Vector())
        avg = avg.normalized() if avg.length > 1e-9 else ns[0]
        out.append(avg / max(min(x.dot(avg) for x in ns), 0.5))
    return out


def _rt_ease(x):
    """1 at 0 falling smoothly to 0 at 1 and beyond."""
    if x >= 1:
        return 0.0
    x = max(x, 0.0)
    return 1.0 - x * x * (3 - 2 * x)


def _rt_band_folds(row, base):
    """How many segments of an offset row run backwards against its base row."""
    return sum(1 for i in range(len(row) - 1) if (row[i + 1] - row[i]).dot(base[i + 1] - base[i]) <= 0)


def _rt_inside(p, poly, e1, e2):
    """Is p inside the closed polygon poly (all in one plane spanned by e1, e2)?"""
    x, y = p.dot(e1), p.dot(e2)
    q = [(a.dot(e1), a.dot(e2)) for a in poly]
    inside = False
    for (ax, ay), (bx, by) in zip(q, q[1:] + q[:1]):
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside


def _rt_side_band(sides, segs, s, loops, n, pname, tail):
    """(points, quads, step) of a panel whose side s carries `loops` rows of quads at a constant distance from it, the
    rest a Coons patch; points row-major with side s as the bottom row."""
    rs = [sides[(s + t) % 4] for t in range(4)]
    nu, nv = segs[s], segs[(s + 1) % 4]
    if loops >= nv:
        raise MifOpError("rings: %d loops along side %d of panel '%s' fill all %d rows across it - lower loops. %s"
                         % (loops, s, pname, nv, tail))
    bot, right = _rt_resample(rs[0], nu), _rt_resample(rs[1], nv)
    top, left = _rt_resample(rs[2], nu)[::-1], _rt_resample(rs[3], nv)[::-1]
    step = 0.5 * ((left[1] - left[0]).length + (right[1] - right[0]).length)
    dirs = _rt_offset_dirs(bot, n, False, 1)
    band = [bot]
    for j in range(1, loops + 1):
        raw = [p + d * (j * step) for p, d in zip(bot, dirs)]
        e0, e1 = left[j] - raw[0], right[j] - raw[-1]
        # the row's ends sit on the neighboring sides; the gap is eased out over a few vertices, so the middle of
        # the row keeps its constant distance from the side
        b = max(2, min(2 * j, nu // 2))
        row = [p + e0 * _rt_ease(i / b) + e1 * _rt_ease((nu - i) / b) for i, p in enumerate(raw)]
        if _rt_band_folds(row, bot):
            raise MifOpError("rings: the %d loops along side %d of panel '%s' fold - the side curves tighter than "
                             "%d x %.6g; lower loops or edgeLength. %s" % (loops, s, pname, loops, step, tail))
        band.append(row)
    rows = band[:-1] + _rt_coons(band[-1], right[loops:], top, left[loops:])
    pts = [p for row in rows for p in row]
    w = nu + 1
    quads = [[j * w + i, j * w + i + 1, (j + 1) * w + i + 1, (j + 1) * w + i]
             for j in range(len(rows) - 1) for i in range(nu)]
    return pts, quads, step


def _rt_hole_grid(sides, segs, hole, loops, el, n, pname, tail):
    """(points, quads, ring row) of a panel with one hole: the hole resampled to 2 x (nu + nv) points, `loops` rings
    at el apart round it, then four Coons patches from the last ring out to the four sides (an O-grid), so every
    ring is a closed loop of quads round the hole."""
    outer = []
    for k in range(4):
        outer.extend(_rt_resample(sides[k], segs[k])[:-1])
    big = len(outer)
    starts = [0, segs[0], segs[0] + segs[1], segs[0] + segs[1] + segs[2]]
    g = sum(hole, Vector()) / len(hole)
    e1 = n.orthogonal().normalized()
    e2 = n.cross(e1)
    closed = hole + hole[:1]
    cum = [0.0]
    for a, b in zip(closed, closed[1:]):
        cum.append(cum[-1] + (b - a).length)
    per = cum[-1]

    def to2(p):
        return (p - g).dot(e1), (p - g).dot(e2)

    def ray_hit(c):
        dx, dy = to2(c)
        best = None
        for k, (a, b) in enumerate(zip(closed, closed[1:])):
            ax, ay = to2(a)
            bx, by = to2(b)
            sx, sy = bx - ax, by - ay
            den = dx * sy - dy * sx
            if abs(den) < 1e-15:
                continue
            t = (ax * sy - ay * sx) / den
            f = (ax * dy - ay * dx) / den
            if t > 0 and -1e-9 <= f <= 1 + 1e-9 and (best is None or t < best[0]):
                best = (t, cum[k] + min(max(f, 0.0), 1.0) * (cum[k + 1] - cum[k]))
        return None if best is None else best[1]
    ts = [ray_hit(outer[a]) for a in starts]
    if any(t is None for t in ts):
        raise MifOpError("the hole of panel '%s' is not round its centre (a ray to a corner misses it) - only a hole "
                         "seen whole from its centre is ringed. %s" % (pname, tail))
    gaps = [(ts[(k + 1) % 4] - ts[k]) % per for k in range(4)]
    if abs(sum(gaps) - per) > 1e-6 * max(per, 1e-9) or min(gaps) <= 0:
        raise MifOpError("the hole of panel '%s' does not face the four corners in order - it is not ringed. %s"
                         % (pname, tail))

    def at(t):
        t %= per
        k = min(max(bisect.bisect_right(cum, t) - 1, 0), len(cum) - 2)
        seg = cum[k + 1] - cum[k]
        return closed[k].lerp(closed[k + 1], (t - cum[k]) / seg if seg > 0 else 0.0)
    ring0 = []
    for k in range(4):
        t0, t1 = ts[k], ts[k] + gaps[k]
        arc = [at(t0)]
        for lap in (0.0, per):
            arc.extend(closed[i].copy() for i in range(len(hole)) if t0 + 1e-12 < cum[i] + lap < t1 - 1e-12)
        arc.append(at(t1))
        ring0.extend(_rt_resample(arc, segs[k])[:-1])
    dirs = _rt_offset_dirs(ring0, n, True, -1)
    rings = [ring0]
    base = ring0 + ring0[:1]
    for r in range(1, loops + 1):
        ring = [p + d * (r * el) for p, d in zip(ring0, dirs)]
        if _rt_band_folds(ring + ring[:1], base):
            raise MifOpError("rings: the %d loops round the hole of panel '%s' fold - the hole turns tighter than "
                             "%d x %.6g; lower loops or edgeLength. %s" % (loops, pname, loops, el, tail))
        rings.append(ring)
    outline = [p for sd in sides for p in sd[:-1]]
    if any(not _rt_inside(p, outline, e1, e2) for p in rings[-1]):
        raise MifOpError("rings: %d loops at %.6g round the hole of panel '%s' reach past its outline - lower loops "
                         "or edgeLength. %s" % (loops, el, pname, tail))
    last = rings[-1]
    reach = sum((outer[a] - last[a]).length for a in starts) / 4
    m = max(1, int(round(reach / el)))
    levels = [[None] * big for _ in range(m + 1)]
    for k in range(4):
        a0, ns = starts[k], segs[k]
        idx = [(a0 + i) % big for i in range(ns + 1)]
        patch = _rt_coons([last[a] for a in idx],
                          [last[idx[-1]].lerp(outer[idx[-1]], j / m) for j in range(m + 1)],
                          [outer[a] for a in idx],
                          [last[idx[0]].lerp(outer[idx[0]], j / m) for j in range(m + 1)])
        for j in range(m + 1):
            for i in range(ns):
                levels[j][idx[i]] = patch[j][i]
    allrows = rings[:-1] + levels
    pts = [p for row in allrows for p in row]
    quads = [[r * big + a, (r + 1) * big + a, (r + 1) * big + (a + 1) % big, r * big + (a + 1) % big]
             for r in range(len(allrows) - 1) for a in range(big)]
    return pts, quads, {"kind": "hole", "loops": loops, "ringVertices": big, "step": round(el, 6),
                        "radialSegments": m, "holeLength": round(per, 6)}


def op_retopo_panels(params):
    """A quad grid over each flat pattern panel - the low-poly retopo of a garment made where the panels are flat -
    with the panel's UVs sampled on and the same vertex count on both sides of every seam named.

    THE GAP (spec "Retopologize flat pattern panels and move the retopo onto the drape by UV", 2026-10-06): the
    cleanest game mesh of a draped garment is the flat panels retopologized, their UVs copied on, then moved to the
    drape by UV (transfer_by_uv); the grid step had no typed call. panels [name | {object, edgeLength, corners}]: each
    a FLAT mesh (base mesh, world space); its outer outline is split at four corners (the four sharpest turns, or
    corners [[x,y,z] x4] - the nearest outline vertices) into four sides, and a Coons patch fills them with quads.
    Segments per side = its length / edgeLength (the panel's own, else the call's), opposite sides share a count and
    seams [[{panel, near: [x,y,z]}, {panel, near}], ...] tie the side nearest each point to its partner; every tied
    set takes its largest count, so a seam has the same vertex count on both panels. UVs come from uvMap (default:
    each panel's active map) at the same point of the panel; a grid point off the panel takes the nearest and is
    counted (uvOutside). One mesh object holds every panel (vertexStart per panel), UV map named uvMap or UVMap.

    LOOPS RINGING A HOLE OR A CURVE (2026-10-08, the same item's part (a)): a panel with a hole (an armhole cut out)
    gets the hole resampled to 2 x (nu + nv) vertices, `loops` closed rings at edgeLength apart round it (default
    1), then four Coons patches from the last ring out to the four sides - an O-grid, so every ring is a closed loop
    of quads and the seam counts still hold. rings [{panel, near: [x,y,z], loops 0-16}] sets a hole's loops (near
    nearest the hole) or, on a panel without a hole, lays `loops` rows at a constant distance (one side step) along
    the outline side nearest near - a concave armhole curve passed as one side by its corners - the rest a Coons
    patch. Each panel replies foldedQuads, holes and, when ringed, rings [{kind, loops, ...}].
    Refused: a panel not flat, closed or listed twice, no UV map, fewer than four 20-degree turns without corners,
    a seam on one side twice, edgeLength not above 0, over 250,000 faces; more than one hole in a panel, a hole a
    ray from its centre to a corner misses, loops past the outline or folding, loops filling every row, a side band
    on a holed panel or a second one on a panel, any quad of a ringed panel folded over.
    """
    reject_unknown(params, _RETOPO_KEYS, "retopo_panels")
    tail = "NOTHING was created."
    try:
        el0 = take_float(params, "edgeLength", default=None)
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
    if el0 is not None and not (math.isfinite(el0) and el0 > 0):
        raise MifOpError("edgeLength must be above 0, got %r. %s" % (el0, tail))
    specs = _rt_panel_specs(take(params, "panels", required=True), tail)
    if el0 is None and any(el is None for _o, el, _c in specs):
        raise MifOpError("edgeLength is required (for the call or on every panel). %s" % tail)
    uv_name = take(params, "uvMap", default=None, kind=str)
    for obj, _el, _c in specs:
        uvs = obj.data.uv_layers
        if not len(uvs) or (uv_name is not None and uvs.get(uv_name) is None):
            raise MifOpError("panel '%s' has no UV map %s - the retopo's UVs are sampled from it. %s"
                             % (obj.name, repr(uv_name) if uv_name else "at all", tail))
    seams_raw = params.get("seams")
    if seams_raw is not None and (not isinstance(seams_raw, (list, tuple)) or any(
            not isinstance(s, (list, tuple)) or len(s) != 2 or any(
                not isinstance(e, dict) or set(e) != {"panel", "near"} for e in s) for s in seams_raw)):
        raise MifOpError("seams is [[{panel, near: [x,y,z]}, {panel, near: [x,y,z]}], ...]. %s" % tail)
    names = {o.name: k for k, (o, _e, _c) in enumerate(specs)}
    outl = [_rt_outline(o, c, tail) for o, _e, c in specs]
    par = {(k, s): (k, s) for k in range(len(specs)) for s in range(4)}

    def union(a, b):
        ra, rb = _rt_find(par, a), _rt_find(par, b)
        par[ra] = rb
    for k in range(len(specs)):
        union((k, 0), (k, 2))
        union((k, 1), (k, 3))
    seam_of = {}
    for si, pair in enumerate(seams_raw or []):
        ends = []
        for e in pair:
            if e["panel"] not in names:
                raise MifOpError("seams[%d] names panel %r, which is not in panels. %s" % (si, e["panel"], tail))
            near = e["near"]
            if not isinstance(near, (list, tuple)) or len(near) != 3:
                raise MifOpError("seams[%d].near must be [x,y,z]. %s" % (si, tail))
            p = Vector(finite_floats(list(near), "seams.near"))
            k = names[e["panel"]]
            s = min(range(4), key=lambda s, k=k, p=p: _rt_seg_dist(p, outl[k][0][s]))
            if (k, s) in seam_of:
                raise MifOpError("seams[%d]: side %d of '%s' is already in a seam. %s" % (si, s, e["panel"], tail))
            ends.append((k, s))
        if ends[0] == ends[1]:
            raise MifOpError("seams[%d] ties a side to itself. %s" % (si, tail))
        seam_of[ends[0]], seam_of[ends[1]] = ends[1], ends[0]
        union(ends[0], ends[1])
    for k, (obj, _e, _c) in enumerate(specs):
        if len(outl[k][2]) > 1:
            raise MifOpError("panel '%s' has %d holes - one hole per panel is ringed; split the panel or fill the "
                             "other holes. %s" % (obj.name, len(outl[k][2]), tail))
    rings_raw = params.get("rings")
    if rings_raw is not None and (not isinstance(rings_raw, (list, tuple)) or len(rings_raw) > 64 or any(
            not isinstance(e, dict) or not {"panel", "near"} <= set(e) or set(e) - {"panel", "near", "loops"}
            for e in rings_raw)):
        raise MifOpError("rings is [{panel, near: [x,y,z], loops}, ...] (at most 64). %s" % tail)
    ring_of = {}
    for ri, e in enumerate(rings_raw or []):
        if e["panel"] not in names:
            raise MifOpError("rings[%d] names panel %r, which is not in panels. %s" % (ri, e["panel"], tail))
        near = e["near"]
        if not isinstance(near, (list, tuple)) or len(near) != 3:
            raise MifOpError("rings[%d].near must be [x,y,z]. %s" % (ri, tail))
        lps = e.get("loops", 1)
        if isinstance(lps, bool) or not isinstance(lps, int) or not 0 <= lps <= 16:
            raise MifOpError("rings[%d].loops must be a whole number 0-16, got %r. %s" % (ri, lps, tail))
        p = Vector(finite_floats(list(near), "rings.near"))
        k = names[e["panel"]]
        sides_k, _cs, holes_k, _n = outl[k]
        hole_d = min((_rt_seg_dist(p, h + h[:1]) for h in holes_k), default=math.inf)
        s = min(range(4), key=lambda s, k=k, p=p: _rt_seg_dist(p, sides_k[s]))
        if hole_d <= _rt_seg_dist(p, sides_k[s]):
            if (k, "hole") in ring_of:
                raise MifOpError("rings[%d]: the hole of '%s' is already ringed. %s" % (ri, e["panel"], tail))
            ring_of[(k, "hole")] = lps
            continue
        if holes_k:
            raise MifOpError("rings[%d]: '%s' has a hole, so its loops ring the hole - a band along an outline side "
                             "is made only on a panel without one (point near at the hole). %s"
                             % (ri, e["panel"], tail))
        if (k, "side") in ring_of:
            raise MifOpError("rings[%d]: '%s' already has a band along side %d - one outline side per panel. %s"
                             % (ri, e["panel"], ring_of[(k, "side")][0], tail))
        ring_of[(k, "side")] = (s, lps)
    want = {}
    for (k, s) in par:
        el = specs[k][1] or el0
        n = max(1, int(round(_rt_poly_len(outl[k][0][s]) / el)))
        r = _rt_find(par, (k, s))
        want[r] = max(want.get(r, 1), n)
    seg = {ks: want[_rt_find(par, ks)] for ks in par}
    total = sum(seg[(k, 0)] * seg[(k, 1)] for k in range(len(specs)))
    if total > 250000:
        raise MifOpError("the grids would hold %d faces (over 250,000) - raise edgeLength. %s" % (total, tail))
    from mathutils.bvhtree import BVHTree   # noqa: PLC0415 - after the refusals: the offline sweep stubs mathutils
    from mathutils.interpolate import poly_3d_calc   # noqa: PLC0415
    verts, faces, uvs, rows = [], [], [], []
    for k, (obj, _el, _c) in enumerate(specs):
        sides, cs, holes, pn = outl[k]
        nu, nv = seg[(k, 0)], seg[(k, 1)]
        segs = [seg[(k, s)] for s in range(4)]
        ring_rows = []
        if holes:
            grid, quads, info = _rt_hole_grid(sides, segs, holes[0], ring_of.get((k, "hole"), 1),
                                              specs[k][1] or el0, pn, obj.name, tail)
            ring_rows.append(info)
        elif (k, "side") in ring_of:
            s, lps = ring_of[(k, "side")]
            grid, quads, step = _rt_side_band(sides, segs, s, lps, pn, obj.name, tail)
            ring_rows.append({"kind": "side", "side": s, "loops": lps, "step": round(step, 6)})
        else:
            rws = _rt_coons(_rt_resample(sides[0], nu), _rt_resample(sides[1], nv),
                            _rt_resample(sides[2], nu)[::-1], _rt_resample(sides[3], nv)[::-1])
            grid = [p for rw in rws for p in rw]
            quads = [[j * (nu + 1) + i, j * (nu + 1) + i + 1, (j + 1) * (nu + 1) + i + 1, (j + 1) * (nu + 1) + i]
                     for j in range(nv) for i in range(nu)]
        folded = sum(1 for q in quads if (grid[q[2]] - grid[q[0]]).cross(grid[q[3]] - grid[q[1]]).dot(pn) <= 0)
        if folded and ring_rows:
            raise MifOpError("panel '%s': %d of its %d quads fold over with these rings - lower loops or edgeLength, "
                             "or pass corners nearer the hole's sides. %s" % (obj.name, folded, len(quads), tail))
        if len(faces) + len(quads) > 250000:
            raise MifOpError("the grids would hold over 250,000 faces - raise edgeLength. %s" % tail)
        start = len(verts)
        bm = evaluated_world(obj, evaluated=False)
        try:
            lay = bm.loops.layers.uv.get(uv_name) if uv_name else bm.loops.layers.uv.active
            if lay is None:
                lay = bm.loops.layers.uv[0]
            tree = BVHTree.FromBMesh(bm)
            diag = max((cs[0] - cs[2]).length, (cs[1] - cs[3]).length)
            vuv, outside = [], 0
            for p in grid:
                co, _n, fi, d = tree.find_nearest(p)
                if d > 1e-5 * diag:
                    outside += 1
                f = bm.faces[fi]
                w = poly_3d_calc([lv.co for lv in f.verts], co)
                vuv.append(sum((lp[lay].uv * wi for lp, wi in zip(f.loops, w)), Vector((0.0, 0.0))))
        finally:
            bm.free()
        verts.extend([list(p) for p in grid])
        for q in quads:
            faces.append([start + x for x in q])
            uvs.append([[round(vuv[x].x, 7), round(vuv[x].y, 7)] for x in q])
        row = {"panel": obj.name, "vertexStart": start, "vertexCount": len(grid), "faceCount": len(quads),
               "grid": [nu, nv], "uvOutside": outside, "foldedQuads": folded, "holes": len(holes),
               "corners": [[round(x, 6) for x in c] for c in cs],
               "sides": [dict({"length": round(_rt_poly_len(sides[s]), 6), "segments": seg[(k, s)],
                               "vertices": seg[(k, s)] + 1},
                              **({"seamWith": {"panel": specs[seam_of[(k, s)][0]][0].name,
                                               "side": seam_of[(k, s)][1]}} if (k, s) in seam_of else {}))
                         for s in range(4)]}
        if ring_rows:
            row["rings"] = ring_rows
        rows.append(row)
    from .ops_create import op_create_mesh   # noqa: PLC0415 - create_mesh owns naming and the read-back
    sub = {"name": params.get("name") or specs[0][0].name + "_retopo", "vertices": verts, "faces": faces,
           "uvs": {uv_name or "UVMap": uvs}}
    for kk in _PLACE_KEYS[1:]:
        if params.get(kk) is not None:
            sub[kk] = params[kk]
    out = op_create_mesh(sub)
    out["op"] = "retopo_panels"
    out["panels"] = rows
    out["uvMap"] = uv_name or "UVMap"
    out["quads"] = len(faces)
    return out


# --------------------------------------------------------------------------------------------- transfer_by_uv

_UVX_KEYS = ("object", "source", "uvMap", "sourceUvMap", "weldDistance", "levels", "evaluated")


def _uvx_source(src, name, evaluated, tail):
    """(BVHTree over the source's UV triangles as (u, v, 0), [(uv triangle, world triangle)]) of the source."""
    from mathutils.bvhtree import BVHTree   # noqa: PLC0415
    sb = evaluated_world(src, evaluated=evaluated)
    try:
        lay = sb.loops.layers.uv.get(name)
        if lay is None:
            raise MifOpError("source '%s' has no UV map %r%s. %s" % (src.name, name, " as it evaluates"
                                                                     if evaluated else "", tail))
        pts, polys, tris = [], [], []
        for tri in sb.calc_loop_triangles():
            uvt = [Vector((lp[lay].uv.x, lp[lay].uv.y, 0.0)) for lp in tri]
            k = len(pts)
            pts.extend(uvt)
            polys.append((k, k + 1, k + 2))
            tris.append((uvt, [lp.vert.co.copy() for lp in tri]))
    finally:
        sb.free()
    if not tris:
        raise MifOpError("source '%s' has no faces. %s" % (src.name, tail))
    return BVHTree.FromPolygons(pts, polys, all_triangles=True), tris


def _uvx_move(bm, lay, tree, tris, shift):
    """Each vertex of bm (shifted world) to the source point at its UV. Returns (outside, noUV, splitUV, max move)."""
    from mathutils.interpolate import poly_3d_calc   # noqa: PLC0415
    outside = no_uv = split = 0
    big = 0.0
    for v in bm.verts:
        lps = list(v.link_loops)
        if not lps:
            no_uv += 1
            continue
        uv = lps[0][lay].uv
        if any((lp[lay].uv - uv).length > 1e-6 for lp in lps[1:]):
            split += 1
        co, _n, ti, d = tree.find_nearest(Vector((uv.x, uv.y, 0.0)))
        if ti is None:
            no_uv += 1
            continue
        if d > 1e-6:
            outside += 1
        uvt, wt = tris[ti]
        w = poly_3d_calc(uvt, co)
        p = wt[0] * w[0] + wt[1] * w[1] + wt[2] * w[2] - shift
        big = max(big, (p - v.co).length)
        v.co = p
    return outside, no_uv, split, big


def _uvx_weld(bm, dist):
    """Boundary vertices closer than dist merged in groups at their centre. Returns (groups, vertices merged)."""
    from mathutils.kdtree import KDTree   # noqa: PLC0415
    bnd = [v for v in bm.verts if v.is_boundary]
    if len(bnd) < 2:
        return 0, 0
    kd = KDTree(len(bnd))
    for i, v in enumerate(bnd):
        kd.insert(v.co, i)
    kd.balance()
    par = list(range(len(bnd)))
    for i, v in enumerate(bnd):
        for _co, j, _d in kd.find_range(v.co, dist):
            if j != i:
                par[_rt_find(par, i)] = _rt_find(par, j)
    groups = {}
    for i in range(len(bnd)):
        groups.setdefault(_rt_find(par, i), []).append(bnd[i])
    tmap, n = {}, 0
    for g in groups.values():
        if len(g) < 2:
            continue
        n += 1
        c = sum((v.co for v in g), Vector()) / len(g)
        g[0].co = c
        for v in g[1:]:
            tmap[v] = g[0]
    if tmap:
        bmesh.ops.weld_verts(bm, targetmap=tmap)
    return n, len(tmap)


def op_transfer_by_uv(params):
    """Move a mesh's vertices to the point of another surface at the same UV - a flat retopo onto its draped
    garment - with the seam pairs welded at their centre and an upward LOD chain made the same way.

    THE GAP (spec "Retopologize flat pattern panels and move the retopo onto the drape by UV", 2026-10-06): a data
    transfer samples by position, not by UV, so the retopo of flat panels could not be put on the drape. Each vertex
    of `object` takes its UV in uvMap (default the active map; a vertex on a UV seam takes its first corner's, and is
    counted) and moves to the point of `source` (as it evaluates unless evaluated is false, world space) whose
    sourceUvMap triangle (default: the map of the same name) holds that UV, by barycentric weights; a UV outside every
    source triangle takes the nearest one and is counted (uvOutside). weldDistance then merges boundary vertices
    closer than it at their centre (the seam pairs). levels 1-4 also makes object_up1..N: the flat mesh subdivided
    once more each level (quads stay quads, UVs interpolated), moved and welded the same way. Reply: moved, uvOutside,
    largestMove, welded, each level's object and counts. Refused: the source being the object, a missing UV map,
    weldDistance not above 0, levels outside 0-4 or over 2,000,000 faces, shape keys.
    """
    reject_unknown(params, _UVX_KEYS, "transfer_by_uv")
    tail = "NOTHING was changed."
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    src = get_object(take(params, "source", required=True, kind=str), want_mesh=True)
    if src is obj:
        raise MifOpError("source is the object itself; name the draped surface to move onto. %s" % tail)
    try:
        weld = take_float(params, "weldDistance", default=None)
        levels = take_int(params, "levels", default=0)
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
    evaluated = take_bool(params, "evaluated", default=True)
    if weld is not None and not (math.isfinite(weld) and weld > 0):
        raise MifOpError("weldDistance must be above 0, got %r. %s" % (weld, tail))
    if not 0 <= levels <= 4:
        raise MifOpError("levels must be 0-4, got %r. %s" % (levels, tail))
    if len(obj.data.polygons) * 4 ** levels > 2000000:
        raise MifOpError("levels %d would make %d faces (over 2,000,000). %s"
                         % (levels, len(obj.data.polygons) * 4 ** levels, tail))
    uv_name = take(params, "uvMap", default=None, kind=str)
    lay_rna = obj.data.uv_layers.get(uv_name) if uv_name else obj.data.uv_layers.active
    if lay_rna is None:
        raise MifOpError("'%s' has no UV map %s. %s" % (obj.name, repr(uv_name) if uv_name else "at all", tail))
    uv_name = lay_rna.name
    s_name = take(params, "sourceUvMap", default=None, kind=str) or uv_name
    if src.data.uv_layers.get(s_name) is None and params.get("sourceUvMap") is None and src.data.uv_layers.active:
        s_name = src.data.uv_layers.active.name
    tree, tris = _uvx_source(src, s_name, evaluated, tail)
    bm, shift = _world_edit(obj, "transfer_by_uv")
    made = []
    try:
        lay = bm.loops.layers.uv.get(uv_name)
        flat = bm.copy() if levels else None
        outside, no_uv, split, big = _uvx_move(bm, lay, tree, tris, shift)
        moved = len(bm.verts) - no_uv
        welded = _uvx_weld(bm, weld) if weld else (0, 0)
        _write_world(bm, obj, shift)
        try:
            for lv in range(1, levels + 1):
                bmesh.ops.subdivide_edges(flat, edges=flat.edges[:], cuts=1, use_grid_fill=True)
                cp = flat.copy()
                try:
                    lay2 = cp.loops.layers.uv.get(uv_name)
                    o2, _n2, _s2, b2 = _uvx_move(cp, lay2, tree, tris, shift)
                    w2 = _uvx_weld(cp, weld) if weld else (0, 0)
                    me = bpy.data.meshes.new("%s_up%d" % (obj.name, lv))
                    for m in obj.data.materials:
                        me.materials.append(m)
                    ob = bpy.data.objects.new(me.name, me)
                    for col in obj.users_collection:
                        col.objects.link(ob)
                    ob.matrix_world = obj.matrix_world.copy()
                    _write_world(cp, ob, shift)
                finally:
                    cp.free()
                quads = sum(1 for p in me.polygons if len(p.vertices) == 4)
                want_name = "%s_up%d" % (obj.name, lv)
                made.append({"level": lv, "object": ob.name, "requestedName": want_name,
                             "nameWasSuffixed": ob.name != want_name, "counts": mesh_counts(ob), "quads": quads,
                             "uvOutside": o2, "largestMove": round(b2, 6), "weldedGroups": w2[0]})
        finally:
            if flat is not None:
                flat.free()
    finally:
        bm.free()
    out = {"object": obj.name, "source": src.name, "uvMap": uv_name, "sourceUvMap": s_name, "evaluated": evaluated,
           "moved": moved, "uvOutside": outside, "noUV": no_uv,
           "uvSeamVertices": split, "largestMove": round(big, 6), "counts": mesh_counts(obj),
           "quads": sum(1 for p in obj.data.polygons if len(p.vertices) == 4)}
    if weld:
        out["welded"] = {"distance": weld, "groups": welded[0], "verticesMerged": welded[1]}
    if made:
        out["levels"] = made
        out["created"] = [m["object"] for m in made]
    if outside:
        out["uvOutsideNote"] = ("%d vertex/vertices had a UV outside every source triangle and took the nearest "
                                "triangle's edge - the source's UV layout does not cover them." % outside)
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- loft_garment

_LOFT_KEYS = ("body", "armature", "rings", "segments", "ease", "offset", "edgeLength", "evaluated", "name",
              "collection", "takeOverName", "garment", "easeTable")

# MEASURED EASE TABLES, by garment and body region: a ring's ease (garment perimeter / body perimeter - 1 in the plane
# square to the bone, loft_garment's own measure) read off a professionally made garment on the body it was fitted to,
# with measure_garment_ease. Every number carries its source; a garment or a region nobody measured is absent, never
# guessed - a call naming one is refused and told to pass easeTable or a ring's own ease. Filled from measurement only.
#
# SOURCE of both tables (measured 2026-10-08): the crew-neck T-shirt and shorts Unreal Engine 5.8.2 ships in the
# MetaHumanCharacter plugin (Content/Optional/Clothing/DefaultGarment), each garment's imported mesh (ClothAssets/
# DF_mtd_<body>_Import/.../SM_Mesh) on the body it was fitted to (SourceBodies/archetype_SkelMesh, f_tall_ovw_SkelMesh,
# m_srt_unw_SkelMesh), exported to FBX and measured in Blender 5.2.1 with measure_garment_ease at 128 directions; the
# garment split into its panels, the top's and the bottom's measured apart. Rings (a region is a landmark, not a size):
# chest - the torso square to a vertical line through the spine at the spine_05 and spine_04 joint heights; waist -
# spine_03 and spine_02; hip - spine_01 and the thigh joint; upperArm - 0.1 to 0.5 of the shoulder-to-elbow line; thigh -
# 0.1 to 0.4 of the hip-to-knee line. Per body, the median of its complete rings; the value here is the median over the
# bodies, with every body's number beside it. These are fitted garments: the gap never fell under ~3.5 mm.
#
# SOURCE of hoodie and jeans (measured 2026-10-08): a bought MetaHuman outfit package (zip hoodie with the hood up,
# straight-leg jeans, sneakers) and the one body it was fitted to (its Meshes/Male_BodyShape, on the MetaHuman body
# skeleton), exported from UE 5.8.2 to FBX and measured in Blender 5.2.1 at 128 directions. Each garment is its shell:
# its loose parts of 150 faces or more, less the parts a ring would meet before the shell - for the hoodie the zipper's
# pieces, the hood cords, the inner collar facing (inside the chest at the chest ring) and the front pocket; for the jeans
# the button and the fly. Rings as above, plus forearm - 0.2 to 0.6 of the elbow-to-wrist line; knee - 0.95 of the
# hip-to-knee line and 0.05 of the knee-to-ankle line; calf - 0.2 to 0.5 of the knee-to-ankle line; hood - the hood round
# a vertical line through the head joint, from the head joint's height up 0.04, 0.08 and 0.12 (every ring complete). The
# hoodie's hip ring at the thigh joint is below its hem, so its hip is the one ring at spine_01 (the rib hem band). One
# body, so each region is the median of its complete rings on it, left and right together.
#
# SOURCE of dressShirt (measured 2026-10-09): the long-sleeve button shirt of a MetaHuman outfit package (Shirt_04), its
# shipped source meshes for the man and the woman (one topology, refitted for each) on the two bodies the package fits
# them to (its Meshes/SK_Shape_tall_mid_man and _woman, on the MetaHuman body skeleton), the bodies exported from UE
# 5.8.2 to FBX through a slot-padded copy (export_asset's own remedy) and measured in Blender 5.2.1 at 128 directions.
# The shirt is its shell: the one loose part of 9,029 faces, less the two chest pocket flaps and the buttons. Rings as
# above (chest, waist, hip, upperArm, forearm; the hem is below the thigh joint on both bodies, so both hip rings count),
# plus collar - 0.1 to 0.3 of the line from neck_01 to the head joint; cuff - 0.85 to 0.95 of the elbow-to-wrist line
# (the sleeve runs past the wrist on both). Every ring complete, none inside the body. Per body, the median of its rings;
# the value here is the median over the two bodies (their mean), each body's number beside it. The woman's fit is the
# looser one everywhere but the hip.
GARMENT_EASE = {
    "tshirt": {
        "regions": {"chest": 0.0484, "waist": 0.0938, "hip": 0.0686, "upperArm": 0.078},
        "bodies": {"chest": {"archetype": 0.0484, "f_tall": 0.1032, "m_srt": 0.0234},
                   "waist": {"archetype": 0.0938, "f_tall": 0.0456, "m_srt": 0.1286},
                   "hip": {"archetype": 0.0761, "m_srt": 0.061},          # f_tall: the hem rides above the hip ring
                   "upperArm": {"archetype": 0.078, "f_tall": 0.0725, "m_srt": 0.1043}},
        "source": "UE 5.8.2 MetaHumanCharacter DefaultGarment, the T-shirt panels, 3 bodies (measured 2026-10-08)",
    },
    "shorts": {
        "regions": {"hip": 0.0327, "thigh": 0.0528},
        "bodies": {"hip": {"archetype": 0.0327, "f_tall": 0.0067, "m_srt": 0.0365},
                   "thigh": {"archetype": 0.0528, "f_tall": 0.0318, "m_srt": 0.0749}},
        "source": "UE 5.8.2 MetaHumanCharacter DefaultGarment, the shorts panels, 3 bodies (measured 2026-10-08)",
    },
    "hoodie": {
        "regions": {"chest": 0.1139, "waist": 0.4097, "hip": 0.075, "upperArm": 0.1575, "forearm": 0.1414,
                    "hood": 0.2625},
        "bodies": {"chest": {"male_bodyshape": 0.1139}, "waist": {"male_bodyshape": 0.4097},
                   "hip": {"male_bodyshape": 0.075}, "upperArm": {"male_bodyshape": 0.1575},
                   "forearm": {"male_bodyshape": 0.1414}, "hood": {"male_bodyshape": 0.2625}},
        "source": "a bought MetaHuman outfit's zip hoodie (hood up), its shell, on the body it was fitted to "
                  "(measured 2026-10-08)",
    },
    "jeans": {
        "regions": {"hip": 0.0566, "thigh": 0.1503, "knee": 0.3319, "calf": 0.3842},
        "bodies": {"hip": {"male_bodyshape": 0.0566}, "thigh": {"male_bodyshape": 0.1503},
                   "knee": {"male_bodyshape": 0.3319}, "calf": {"male_bodyshape": 0.3842}},
        "source": "a bought MetaHuman outfit's straight-leg jeans, their shell, on the body they were fitted to "
                  "(measured 2026-10-08)",
    },
    "dressShirt": {
        "regions": {"chest": 0.1853, "waist": 0.2133, "hip": 0.0791, "collar": 0.2652, "upperArm": 0.3177,
                    "forearm": 0.2323, "cuff": 0.2905},
        "bodies": {"chest": {"tall_mid_man": 0.0726, "tall_mid_woman": 0.298},
                   "waist": {"tall_mid_man": 0.1778, "tall_mid_woman": 0.2489},
                   "hip": {"tall_mid_man": 0.0894, "tall_mid_woman": 0.0688},
                   "collar": {"tall_mid_man": 0.1925, "tall_mid_woman": 0.338},
                   "upperArm": {"tall_mid_man": 0.1588, "tall_mid_woman": 0.4767},
                   "forearm": {"tall_mid_man": 0.1517, "tall_mid_woman": 0.313},
                   "cuff": {"tall_mid_man": 0.2194, "tall_mid_woman": 0.3616}},
        "source": "a MetaHuman outfit package's long-sleeve button shirt (Shirt_04), its shell, on the two bodies it "
                  "was fitted to (measured 2026-10-09)",
    },
}

# MEASURED SHOE LASTS: a shoe's size against the foot it was fitted to, as shares of the foot, for build_sole (`last`)
# and a lofted upper. length, width: the shoe's over the foot's; height, collar: the shoe's top (the tongue) and its heel
# counter's top above its bottom, over the ankle joint's height above the foot's bottom; toe, heel, side: the room past
# the foot in front, behind and to each side (the mean of the two), soleHeel, soleBall: the sole under the foot at the
# heel and the ball - these as shares of the foot's length. SOURCE of sneaker (measured 2026-10-08): the same bought
# MetaHuman outfit's sneakers on the same body, the left shoe (every loose part) against the left foot, in sections
# across Z every 4 mm (the foot's below 0.06 m) and across Y at the heel and the ball, world axes for both (the foot
# points 6.5 degrees off Y; the ratios share that frame). Foot 0.2627 long, 0.1121 wide, ankle joint 0.0853 up; shoe
# 0.3182 x 0.1320 x 0.1227; room 0.0363 at the toe, 0.0192 at the heel, 0.0150 inside and 0.0049 outside; sole under
# the foot 0.0103 at the heel, 0.0110 at the ball (it does not rise toward the heel).
SHOE_LAST = {
    "sneaker": {
        "ratios": {"length": 1.2113, "width": 1.1775, "height": 1.4388, "collar": 1.2079, "toe": 0.1382,
                   "heel": 0.0731, "side": 0.0379, "soleHeel": 0.0392, "soleBall": 0.0419},
        "source": "a bought MetaHuman outfit's sneakers, the left shoe on the foot it was fitted to (measured "
                  "2026-10-08)",
    },
}


def _lg_ease_table(params, tail):
    """{region: ease} for the call: the named measured garment table (GARMENT_EASE), then easeTable over it. None when
    the call names neither."""
    name = params.get("garment")
    own = params.get("easeTable")
    if name is None and own is None:
        return None
    table = {}
    if name is not None:
        if not isinstance(name, str) or name not in GARMENT_EASE:
            raise MifOpError("garment %r has no measured ease table; the measured ones are: %s. Pass easeTable "
                             "{region: ease} or each ring's ease - no table is guessed. %s"
                             % (name, ", ".join(sorted(GARMENT_EASE)) or "none yet", tail))
        table.update(GARMENT_EASE[name]["regions"])
    if own is not None:
        if not isinstance(own, dict) or not own:
            raise MifOpError("easeTable is {region: ease}, got %r. %s" % (own, tail))
        for reg, val in own.items():
            e = finite_float(val, "easeTable[%r]" % reg)
            if e <= -1.0:
                raise MifOpError("easeTable[%r] must be above -1 (a share of the body's size), got %r. %s"
                                 % (reg, e, tail))
            table[str(reg)] = e
    return table


def _lg_rings(raw, arm, tail, least=2, params=None):
    """[(pose bone, at, ease or None, offset or None)] from rings [{bone, at, ease, offset, region}]. With params, a
    ring without its own ease takes its region's from the call's ease table (_lg_ease_table)."""
    if not isinstance(raw, (list, tuple)) or len(raw) < least or len(raw) > 200:
        raise MifOpError("rings is a list of %d-200 {bone, at (0 head - 1 tail), ease, offset}. %s" % (least, tail))
    table = _lg_ease_table(params, tail) if params is not None else None
    out = []
    for k, r in enumerate(raw):
        if not isinstance(r, dict) or "bone" not in r or set(r) - {"bone", "at", "ease", "offset", "region"}:
            raise MifOpError("rings[%d] is {bone, at, ease, offset, region}, got %r. %s" % (k, r, tail))
        if table is not None and r.get("ease") is None:
            reg = r.get("region")
            if reg is None:
                raise MifOpError("rings[%d] has no ease and no region - the ease table is read by region. %s"
                                 % (k, tail))
            if str(reg) not in table:
                raise MifOpError("rings[%d]: region %r is not in the ease table (%s) - pass its ease. %s"
                                 % (k, reg, ", ".join(sorted(table)), tail))
            r = dict(r, ease=table[str(reg)])
        pb = arm.pose.bones.get(str(r["bone"]))
        if pb is None:
            raise MifOpError("rings[%d]: '%s' has no bone %r. %s" % (k, arm.name, r["bone"], tail))
        at = finite_float(r.get("at", 0.0), "rings[%d].at" % k)
        if not 0.0 <= at <= 1.0:
            raise MifOpError("rings[%d].at is 0 (the head) to 1 (the tail), got %r. %s" % (k, at, tail))
        ease = None if r.get("ease") is None else finite_float(r["ease"], "rings[%d].ease" % k)
        off = None if r.get("offset") is None else finite_float(r["offset"], "rings[%d].offset" % k)
        if ease is not None and ease <= -1.0:
            raise MifOpError("rings[%d].ease must be above -1 (a share of the body's size), got %r. %s"
                             % (k, ease, tail))
        out.append((pb, at, ease, off))
    return out


def _lg_section(bm, p, n):
    """The body's cross-section in the plane (p, n): [(a, b)] world segments, one per face the plane cuts."""
    sd = [(v.co - p).dot(n) for v in bm.verts]
    segs = []
    for f in bm.faces:
        idx = [v.index for v in f.verts]
        ds = [sd[i] for i in idx]
        if min(ds) > 0.0 or max(ds) < 0.0:
            continue
        pts = []
        for a in range(len(idx)):
            i, j = idx[a], idx[(a + 1) % len(idx)]
            di, dj = sd[i], sd[j]
            if (di < 0.0) != (dj < 0.0):
                t = di / (di - dj)
                pts.append(bm.verts[i].co.lerp(bm.verts[j].co, t))
        if len(pts) >= 2:
            segs.append((pts[0], pts[1]))
    return segs


def _lg_frame(arm, pb, at, tail):
    """(centre, unit bone direction, unit X in the ring's plane on the bone's own Z - its roll) in world space."""
    mw = arm.matrix_world @ pb.matrix
    head, tl = arm.matrix_world @ pb.head, arm.matrix_world @ pb.tail
    if (tl - head).length < 1e-9:
        raise MifOpError("bone '%s' has no length. %s" % (pb.name, tail))
    n = (tl - head).normalized()
    zax = mw.to_3x3() @ Vector((0.0, 0.0, 1.0))
    ux = zax - n * zax.dot(n)
    if ux.length < 1e-9:
        ux = n.orthogonal()
    return head.lerp(tl, at), n, ux.normalized()


def _lg_cast(section, c, n, ux, angles, e, o, intersect_line_line_2d):
    """(garment ring, body points, rays missed): along each angle (0 on ux) in the plane (c, n), the first crossing of
    the body's section from c, times 1 + e plus o."""
    uy = n.cross(ux)
    segs = [((a - c).dot(ux), (a - c).dot(uy), (b - c).dot(ux), (b - c).dot(uy)) for a, b in section]
    far = 1e6
    ring, body_pts, missed = [], [], 0
    for ang in angles:
        d = (math.cos(ang), math.sin(ang))
        end = Vector((d[0] * far, d[1] * far))
        best = None
        for ax, ay, bx, by in segs:
            hit = intersect_line_line_2d(Vector((0.0, 0.0)), end, Vector((ax, ay)), Vector((bx, by)))
            if hit is not None:
                r = hit.length
                if best is None or r < best:
                    best = r
        if best is None:
            missed += 1
            continue
        dirw = ux * d[0] + uy * d[1]
        body_pts.append(c + dirw * best)
        ring.append(c + dirw * (best * (1.0 + e) + o))
    return ring, body_pts, missed


def _lg_row(pb, at, c, e, o, body_pts, ring):
    def perim(pts):
        return sum((pts[i] - pts[i - 1]).length for i in range(len(pts)))
    bp, gp = perim(body_pts), perim(ring)
    return {"bone": pb.name, "at": at, "center": [round(x, 6) for x in c], "ease": e, "offset": o,
            "bodyPerimeter": round(bp, 6), "garmentPerimeter": round(gp, 6),
            "ratio": round(gp / bp, 6) if bp > 0 else None}


def op_loft_garment(params):
    """Loft a garment tube round a body through rings placed at its bones - a sleeve, a trouser leg, a torso - each
    ring the body's own cross-section there, eased out.

    THE GAP (spec "A garment builder: patterns, cloth drape, weight transfer", route revised 2026-10-03, steps 1-3):
    the landmark-and-ring method places nothing by hand - landmarks are the skeleton's bones, a ring at each is the
    body's cross-section there plus an ease, and the tube is lofted through the rings on an even quad grid; the lofts
    were raw Python (pants_loft, hoodie_loft). rings [{bone, at, ease, offset}] (2 or more, in loft order) on
    `armature` (posed, world space): the ring's plane passes through the bone at `at` (0 head - 1 tail) square to the
    bone; along each of `segments` (default 32) directions in that plane, angle 0 on the bone's own Z axis (its roll),
    the ring vertex sits where the ray from the bone first meets the body's section (`body` as it evaluates unless
    evaluated is false), times 1 + ease (default the call's ease, 0.05) plus offset (world, default the call's, 0) - so
    an arm's ring is the arm's own section, not the torso's beside it. Between two rings, rows at edgeLength (default:
    the mean ring edge) interpolate them. UVs: u round the ring from angle 0, v the length down the tube in the same
    units. Reply per ring: the bone, its centre, the body's perimeter there and the garment's (their ratio is 1 + ease
    when offset is 0), and rays that missed. Refused: a ray from a ring missing the body (the ring is outside it), fewer
    than two rings, a bone not on the armature, segments outside 3-512, ease at or under -1, over 500,000 faces.
    """
    reject_unknown(params, _LOFT_KEYS, "loft_garment")
    tail = "NOTHING was created."
    body = get_object(take(params, "body", required=True, kind=str), want_mesh=True)
    arm = get_object(take(params, "armature", required=True, kind=str))
    if arm.type != "ARMATURE":
        raise MifOpError("'%s' is a %s, not an ARMATURE. %s" % (arm.name, arm.type, tail))
    try:
        segments = take_int(params, "segments", default=32)
        ease0 = take_float(params, "ease", default=0.05)
        off0 = take_float(params, "offset", default=0.0)
        el = take_float(params, "edgeLength", default=None)
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
    evaluated = take_bool(params, "evaluated", default=True)
    if not 3 <= segments <= 512:
        raise MifOpError("segments is 3-512, got %r. %s" % (segments, tail))
    if not (math.isfinite(ease0) and ease0 > -1.0) or not math.isfinite(off0):
        raise MifOpError("ease must be above -1 and offset finite. %s" % tail)
    if el is not None and not (math.isfinite(el) and el > 0):
        raise MifOpError("edgeLength must be above 0, got %r. %s" % (el, tail))
    rings = _lg_rings(params.get("rings"), arm, tail, params=params)
    from mathutils.geometry import intersect_line_line_2d   # noqa: PLC0415 - after the refusals (offline stub)
    bm = evaluated_world(body, evaluated=evaluated)
    try:
        if not len(bm.faces):
            raise MifOpError("body '%s' has no faces. %s" % (body.name, tail))
        built, rows = [], []
        angles = [2.0 * math.pi * s / segments for s in range(segments)]
        for k, (pb, at, ease, off) in enumerate(rings):
            c, n, ux = _lg_frame(arm, pb, at, tail)
            e = ease0 if ease is None else ease
            o = off0 if off is None else off
            ring, body_pts, missed = _lg_cast(_lg_section(bm, c, n), c, n, ux, angles, e, o, intersect_line_line_2d)
            if missed:
                raise MifOpError("ring %d (bone '%s' at %g): %d of %d rays met no body surface - the bone there is not "
                                 "inside '%s'. %s" % (k, pb.name, at, missed, segments, body.name, tail))
            built.append(ring)
            rows.append(_lg_row(pb, at, c, e, o, body_pts, ring))
    finally:
        bm.free()
    ring_edge = sum(sum((r[i] - r[i - 1]).length for i in range(segments)) / segments for r in built) / len(built)
    step = el or ring_edge
    full = [built[0]]
    for a, b in zip(built, built[1:]):
        gap = sum(((p - q).length for p, q in zip(a, b))) / segments
        nrow = max(1, int(round(gap / step)))
        for t in range(1, nrow + 1):
            full.append([p.lerp(q, t / nrow) for p, q in zip(a, b)])
    nfaces = (len(full) - 1) * segments
    if nfaces > 500000:
        raise MifOpError("the loft would hold %d faces (over 500,000) - raise edgeLength. %s" % (nfaces, tail))
    verts = [list(p) for row in full for p in row]
    faces, uvs = [], []
    vpos = [0.0]
    for a, b in zip(full, full[1:]):
        vpos.append(vpos[-1] + sum((p - q).length for p, q in zip(a, b)) / segments)
    circ = ring_edge * segments
    for r in range(len(full) - 1):
        for s in range(segments):
            s2 = (s + 1) % segments
            faces.append([r * segments + s, r * segments + s2, (r + 1) * segments + s2, (r + 1) * segments + s])
            u0, u1 = circ * s / segments, circ * (s + 1) / segments
            uvs.append([[u0, vpos[r]], [u1, vpos[r]], [u1, vpos[r + 1]], [u0, vpos[r + 1]]])
    from .ops_create import op_create_mesh   # noqa: PLC0415 - create_mesh owns naming and the read-back
    sub = {"name": params.get("name") or body.name + "_loft", "vertices": verts, "faces": faces, "uvs": {"UVMap": uvs}}
    for kk in _PLACE_KEYS[1:]:
        if params.get(kk) is not None:
            sub[kk] = params[kk]
    out = op_create_mesh(sub)
    out["op"] = "loft_garment"
    out["rings"] = rows
    out["rows"] = len(full)
    out["segments"] = segments
    out["edgeLength"] = round(step, 6)
    table = _lg_ease_table(params, tail)
    if table is not None:
        out["easeTable"] = table
        if params.get("garment") is not None:
            out["garment"] = params["garment"]
            out["easeSource"] = GARMENT_EASE[params["garment"]]["source"]
    return out


# --------------------------------------------------------------------------------------------- measure_garment_ease

_MGE_KEYS = ("body", "garment", "armature", "rings", "segments", "evaluated")


def _median(values):
    """The median: the middle value, or the mean of the middle two."""
    v = sorted(values)
    h = len(v) // 2
    return v[h] if len(v) % 2 else (v[h - 1] + v[h]) / 2.0


def _mge_dists(section, c, n, ux, angles, intersect_line_line_2d):
    """Per angle in the plane (c, n), angle 0 on ux: the distance from c to the section's first crossing, or None."""
    uy = n.cross(ux)
    segs = [((a - c).dot(ux), (a - c).dot(uy), (b - c).dot(ux), (b - c).dot(uy)) for a, b in section]
    out = []
    for ang in angles:
        end = Vector((math.cos(ang) * 1e6, math.sin(ang) * 1e6))
        best = None
        for ax, ay, bx, by in segs:
            hit = intersect_line_line_2d(Vector((0.0, 0.0)), end, Vector((ax, ay)), Vector((bx, by)))
            if hit is not None and (best is None or hit.length < best):
                best = hit.length
        out.append(best)
    return out


def op_measure_garment_ease(params):
    """Measure how far a garment stands off the body it was fitted to, ring by ring at the body's bones - the ease
    loft_garment builds with, read off a real garment rather than chosen.

    THE GAP (spec "A garment builder", LEFT 2026-10-07: per-garment ease tables): a table of eases per garment and region
    is only worth having when its numbers come from garments made by people who fit clothes for a living, and nothing
    measured one. rings [{bone, at, region}] (1 or more) on `armature`, as loft_garment's: each ring's plane passes
    through the bone at `at`, square to it; along each of `segments` (default 64) directions from the bone, angle 0 on
    the bone's roll, the ray's first crossing of the body's section and of the garment's. Per ring: bodyPerimeter and
    garmentPerimeter over the directions both meet, ease (their ratio - 1, the number loft_garment's ring ease
    reproduces), easeMedian / easeMin / easeMax per direction, gapMean / gapMin / gapMax (world units, garment minus
    body), inside (directions where the garment is inside the body), garmentMissed (directions that meet no garment -
    the ring is past a hem or opening) and complete (none missed). `table` is {region: ease} over the complete rings
    that name a region (the median when several share one) - an easeTable for loft_garment. Read-only. Refused: a ring
    whose rays miss the body, a garment that no ring meets, body and garment the same object, segments outside 8-512.
    """
    reject_unknown(params, _MGE_KEYS, "measure_garment_ease")
    tail = "Nothing was changed."
    body = get_object(take(params, "body", required=True, kind=str), want_mesh=True)
    gar = get_object(take(params, "garment", required=True, kind=str), want_mesh=True)
    arm = get_object(take(params, "armature", required=True, kind=str))
    if arm.type != "ARMATURE":
        raise MifOpError("'%s' is a %s, not an ARMATURE. %s" % (arm.name, arm.type, tail))
    if gar is body:
        raise MifOpError("body and garment are both '%s'. %s" % (body.name, tail))
    try:
        segments = take_int(params, "segments", default=64)
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
    if not 8 <= segments <= 512:
        raise MifOpError("segments is 8-512, got %r. %s" % (segments, tail))
    evaluated = take_bool(params, "evaluated", default=True)
    raw = params.get("rings")
    rings = _lg_rings(raw, arm, tail, least=1)
    regions = [r.get("region") for r in raw]
    from mathutils.geometry import intersect_line_line_2d   # noqa: PLC0415 - after the refusals (offline stub)
    angles = [2.0 * math.pi * s / segments for s in range(segments)]
    bm_b = evaluated_world(body, evaluated=evaluated)
    bm_g = evaluated_world(gar, evaluated=evaluated)
    rows = []
    try:
        for k, (pb, at, _e, _o) in enumerate(rings):
            c, n, ux = _lg_frame(arm, pb, at, tail)
            uy = n.cross(ux)
            db = _mge_dists(_lg_section(bm_b, c, n), c, n, ux, angles, intersect_line_line_2d)
            if any(d is None for d in db):
                raise MifOpError("ring %d (bone '%s' at %g): %d of %d rays met no body surface - the bone there is not "
                                 "inside '%s'. %s" % (k, pb.name, at, sum(d is None for d in db), segments, body.name,
                                                      tail))
            dg = _mge_dists(_lg_section(bm_g, c, n), c, n, ux, angles, intersect_line_line_2d)
            both = [(a, rb, rg) for a, rb, rg in zip(angles, db, dg) if rg is not None]
            row = {"bone": pb.name, "at": at, "region": regions[k], "center": [round(x, 6) for x in c],
                   "garmentMissed": segments - len(both), "complete": len(both) == segments}
            if both:
                def perim(pts):
                    return sum((pts[i] - pts[i - 1]).length for i in range(len(pts))) if len(pts) > 2 else 0.0
                bp = perim([c + (ux * math.cos(a) + uy * math.sin(a)) * rb for a, rb, _ in both])
                gp = perim([c + (ux * math.cos(a) + uy * math.sin(a)) * rg for a, _, rg in both])
                eases = sorted(rg / rb - 1.0 for _, rb, rg in both)
                gaps = [rg - rb for _, rb, rg in both]
                row.update({"bodyPerimeter": round(bp, 6), "garmentPerimeter": round(gp, 6),
                            "ease": round(gp / bp - 1.0, 6) if bp > 0 else None,
                            "easeMedian": round(_median(eases), 6), "easeMin": round(eases[0], 6),
                            "easeMax": round(eases[-1], 6), "gapMean": round(sum(gaps) / len(gaps), 6),
                            "gapMin": round(min(gaps), 6), "gapMax": round(max(gaps), 6),
                            "inside": sum(1 for g in gaps if g < 0.0)})
            else:
                row["ease"] = None
            rows.append(row)
    finally:
        bm_b.free()
        bm_g.free()
    if all(r["garmentMissed"] == segments for r in rows):
        raise MifOpError("no ring met the garment '%s' - it is not round these bones. %s" % (gar.name, tail))
    by_region = {}
    for r in rows:
        if r["region"] is not None and r["complete"] and r.get("ease") is not None:
            by_region.setdefault(str(r["region"]), []).append(r["ease"])
    table = {reg: round(_median(v), 6) for reg, v in by_region.items()}
    return {"ok": True, "op": "measure_garment_ease", "body": body.name, "garment": gar.name, "armature": arm.name,
            "segments": segments, "rings": rows, "table": table}


# --------------------------------------------------------------------------------------------- attach_sleeve

_SLEEVE_KEYS = ("object", "body", "armature", "rings", "ease", "offset", "edgeLength", "armholeRadius", "minFacing",
                "evaluated", "garment", "easeTable")


def _wrapped(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def op_attach_sleeve(params):
    """Cut a round armhole in a garment where an arm's bone line meets it, and loft the sleeve out of it - a torso
    tube and its sleeves as one garment, every new face a quad.

    THE GAP (spec "A garment builder", HALF 2026-10-06, LEFT "a torso tube with arm openings joined to the sleeves"):
    loft_garment lofts one tube; nothing cut an armhole in a torso or joined a sleeve to it, and a bridge between two
    loops of different counts is made of triangles.

    rings [{bone, at, ease, offset}] (1 or more, shoulder first) on `armature` are the sleeve's rings, as
    loft_garment's: each the body's own section square to its bone, eased out. The armhole sits on the FIRST ring's
    bone line: every face of `object` whose centre is within armholeRadius (default: the first ring's mean garment
    radius) of that line, on the near side of the first ring, and facing along the bone (normal . direction above
    minFacing, default 0.3 - so the torso's far side is never cut) is removed. The hole must be one disc: a cut that
    reaches a rim or makes two holes is refused. Its rim vertices are then put on the garment's surface (as it was
    before the cut) where a cylinder of armholeRadius round the line meets it, evenly round the line - a round
    armhole, as many vertices as the cut's rim had. The sleeve's rings take the armhole's angles (carried down the
    bones without twisting), so rows between them, at edgeLength (default the armhole's edge), are all quads; a denser
    garment round the arm gives a sleeve of more sides. Winding follows the garment's; UVs (the active map) for the
    new faces form one island to the right of the existing ones, u round the sleeve, v down it, in world units.
    Reply: armhole {vertices, radius, perimeter, facesRemoved, largestMove}, rings (as loft_garment's), rows,
    newFaces, counts. Refused, nothing changed: no face to cut, a cut that is not one hole or does not go round the
    line, a ray from a ring missing the body, a garment face turned over by the rounding, a bone not on the
    armature, armholeRadius not above 0, minFacing outside 0-1, shape keys.
    """
    reject_unknown(params, _SLEEVE_KEYS, "attach_sleeve")
    tail = "NOTHING was changed."
    try:
        obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
        body = get_object(take(params, "body", required=True, kind=str), want_mesh=True)
        arm = get_object(take(params, "armature", required=True, kind=str))
        ease0 = take_float(params, "ease", default=0.05)
        off0 = take_float(params, "offset", default=0.0)
        el = take_float(params, "edgeLength", default=None)
        rad = take_float(params, "armholeRadius", default=None)
        min_facing = take_float(params, "minFacing", default=0.3)
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
    evaluated = take_bool(params, "evaluated", default=True)
    if arm.type != "ARMATURE":
        raise MifOpError("'%s' is a %s, not an ARMATURE. %s" % (arm.name, arm.type, tail))
    if obj is body:
        raise MifOpError("object and body are both '%s'. %s" % (obj.name, tail))
    if not (math.isfinite(ease0) and ease0 > -1.0) or not math.isfinite(off0):
        raise MifOpError("ease must be above -1 and offset finite. %s" % tail)
    if el is not None and not (math.isfinite(el) and el > 0):
        raise MifOpError("edgeLength must be above 0, got %r. %s" % (el, tail))
    if rad is not None and not (math.isfinite(rad) and rad > 0):
        raise MifOpError("armholeRadius must be above 0 (world units), got %r. %s" % (rad, tail))
    if not (math.isfinite(min_facing) and 0.0 <= min_facing <= 1.0):
        raise MifOpError("minFacing is 0-1 (a face's normal . the bone's direction), got %r. %s" % (min_facing, tail))
    rings = _lg_rings(params.get("rings"), arm, tail, least=1, params=params)
    from mathutils.bvhtree import BVHTree   # noqa: PLC0415 - after the refusals (offline stub)
    from mathutils.geometry import intersect_line_line_2d   # noqa: PLC0415

    # ---- the sleeve's frames (absolute world), carried down the bones without twisting, and the body's sections
    frames, prev_ux = [], None
    for pb, at, ease, off in rings:
        c, n, ux = _lg_frame(arm, pb, at, tail)
        if prev_ux is not None:
            t = prev_ux - n * prev_ux.dot(n)
            ux = t.normalized() if t.length > 1e-9 else ux
        prev_ux = ux
        frames.append((pb, at, c, n, ux, ease0 if ease is None else ease, off0 if off is None else off))
    bmb = evaluated_world(body, evaluated=evaluated)
    try:
        if not len(bmb.faces):
            raise MifOpError("body '%s' has no faces. %s" % (body.name, tail))
        sections = [_lg_section(bmb, c, n) for _pb, _at, c, n, _ux, _e, _o in frames]
    finally:
        bmb.free()
    pb0, at0, c0w, n0, ux0, e0, o0 = frames[0]
    if rad is None:
        probe, _bp, missed = _lg_cast(sections[0], c0w, n0, ux0, [2.0 * math.pi * s / 64 for s in range(64)], e0, o0,
                                      intersect_line_line_2d)
        if missed:
            raise MifOpError("ring 0 (bone '%s' at %g): %d of 64 rays met no body surface - the bone there is not "
                             "inside '%s'. %s" % (pb0.name, at0, missed, body.name, tail))
        rad = sum((p - c0w).length for p in probe) / len(probe)
    uy0 = n0.cross(ux0)

    bm, shift = _world_edit(obj, "attach_sleeve")
    try:
        bm.faces.ensure_lookup_table()
        bm.normal_update()
        c0 = c0w - shift
        tree = BVHTree.FromBMesh(bm)             # the garment before the cut: where the armhole goes

        def off_line(p):
            d = p - c0
            return d - n0 * d.dot(n0)
        cut = [f for f in bm.faces if f.normal.dot(n0) > min_facing and (f.calc_center_median() - c0).dot(n0) < 0.0
               and off_line(f.calc_center_median()).length < rad]
        if not cut:
            raise MifOpError("no face of '%s' within armholeRadius %g of bone '%s''s line, before the first ring and "
                             "facing along the bone (minFacing %g) - the line does not meet the garment there. %s"
                             % (obj.name, rad, pb0.name, min_facing, tail))
        from .ops_topology import _ordered   # noqa: PLC0415
        before = {frozenset(vs) for _e, vs in _pick_loops(bm)}
        n_removed = len(cut)
        bmesh.ops.delete(bm, geom=cut, context="FACES_ONLY")
        loose = [e for e in bm.edges if not e.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="EDGES")
        lone = [v for v in bm.verts if not v.link_edges]
        if lone:
            bmesh.ops.delete(bm, geom=lone, context="VERTS")
        bm.verts.index_update()
        loops = _pick_loops(bm)
        new = [lp for lp in loops if frozenset(lp[1]) not in before]
        if len(new) != 1 or len(loops) != len(before) + 1:
            raise MifOpError("the armhole cut is not one hole in '%s' (%d new open loop(s), %d before, %d after) - "
                             "armholeRadius %g reaches a rim or another opening; make it smaller. %s"
                             % (obj.name, len(new), len(before), len(loops), rad, tail))
        order = _ordered(new[0])
        if order is None or len(order) < 3:
            raise MifOpError("the armhole's rim is not a clean ring (a bow-tie vertex). %s" % tail)
        angs = [math.atan2(off_line(v.co).dot(uy0), off_line(v.co).dot(ux0)) for v in order]
        turn = sum(_wrapped(angs[(i + 1) % len(angs)] - angs[i]) for i in range(len(angs)))
        if abs(abs(turn) - 2.0 * math.pi) > 0.5:
            raise MifOpError("the armhole's rim does not go round bone '%s''s line (it turns %.2f rad) - the line "
                             "passes beside the cut. %s" % (pb0.name, turn, tail))
        k = len(order)
        sgn = 1.0 if turn > 0 else -1.0
        targets = [angs[0] + sgn * 2.0 * math.pi * i / k for i in range(k)]
        # rounding: each rim vertex to where the cylinder meets the garment as it was, evenly round the line
        adj = {f for v in order for f in v.link_faces}
        normals0 = {f: f.normal.copy() for f in adj}
        big = 0.0
        for v, a in zip(order, targets):
            radial = ux0 * math.cos(a) + uy0 * math.sin(a)
            start = c0 + radial * rad
            hit = None
            for direction in (-n0, n0):
                h = tree.ray_cast(start - direction * 1e3, direction)
                if h[0] is not None and (hit is None or (h[0] - start).dot(n0) > (hit - start).dot(n0)):
                    hit = h[0]
            if hit is None:
                raise MifOpError("the armhole at angle %.3f rad does not meet '%s' at armholeRadius %g - make it "
                                 "smaller. %s" % (a, obj.name, rad, tail))
            big = max(big, (hit - v.co).length)
            v.co = hit
        for f in adj:
            f.normal_update()
        flipped = sum(1 for f in adj if f.normal.dot(normals0[f]) < 0.0)
        if flipped:
            raise MifOpError("rounding the armhole turned %d face(s) of '%s' over (largest move %.4g) - make the "
                             "garment denser round the arm or change armholeRadius. %s"
                             % (flipped, obj.name, big, tail))

        # ---- the sleeve's rings at the armhole's angles, in the shifted space
        built, rows = [], []
        for kk, ((pb, at, c, n, ux, e, o), sec) in enumerate(zip(frames, sections)):
            # the armhole's angles are measured on ux0; each ring's ux is ux0 carried down, so the same angle lines up
            ring, body_pts, missed = _lg_cast(sec, c, n, ux, targets, e, o, intersect_line_line_2d)
            if missed:
                raise MifOpError("ring %d (bone '%s' at %g): %d of %d rays met no body surface - the bone there is not "
                                 "inside '%s'. %s" % (kk, pb.name, at, missed, k, body.name, tail))
            built.append([p - shift for p in ring])
            rows.append(_lg_row(pb, at, c, e, o, body_pts, ring))
        hole = [v.co.copy() for v in order]
        perim = _ring_length(hole)
        step = el or perim / k
        full = [hole]
        for a, b in zip([hole] + built, built):
            gap = sum((p - q).length for p, q in zip(a, b)) / k
            nrow = max(1, int(round(gap / step)))
            for t in range(1, nrow + 1):
                full.append([p.lerp(q, t / nrow) for p, q in zip(a, b)])
        if len(bm.faces) + (len(full) - 1) * k > 500000:
            raise MifOpError("the sleeve would take '%s' over 500,000 faces - raise edgeLength. %s" % (obj.name, tail))
        fwd = _winding(order)
        vrows = [order] + [[bm.verts.new(p) for p in row] for row in full[1:]]
        made = _band(bm, vrows, fwd)
        uv = bm.loops.layers.uv.active
        if uv is not None:
            u_right = max((lp[uv].uv.x for f in bm.faces if f not in made for lp in f.loops), default=0.0)
            u0 = u_right + 0.05 * perim
            vpos = [0.0]
            for a, b in zip(full, full[1:]):
                vpos.append(vpos[-1] + sum((p - q).length for p, q in zip(a, b)) / k)
            at_v = {}
            for r_i, row in enumerate(vrows):
                for s, v in enumerate(row):
                    at_v[v] = (r_i, s)
            for f in made:
                ss = [at_v[lp.vert][1] for lp in f.loops]
                wrap = max(ss) - min(ss) > 1
                for lp in f.loops:
                    r_i, s = at_v[lp.vert]
                    if wrap and s == 0:
                        s = k
                    lp[uv].uv = (u0 + perim * s / k, vpos[r_i])
        _write_world(bm, obj, shift)
        new_faces = len(made)
        nrows = len(full)
    finally:
        bm.free()
    out = {"object": obj.name, "armhole": {"vertices": k, "radius": rad, "perimeter": round(perim, 6),
                                           "facesRemoved": n_removed, "largestMove": round(big, 6),
                                           "center": [round(x, 6) for x in c0w], "bone": pb0.name},
           "rings": rows, "rows": nrows, "segments": k, "edgeLength": round(step, 6), "newFaces": new_faces,
           "uvMap": obj.data.uv_layers.active.name if obj.data.uv_layers.active else None,
           "counts": mesh_counts(obj)}
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- displace_by_rule

_DISPLACE_KEYS = ("object", "rules", "name")
_DR_TYPES = ("SEAM", "STITCH", "WEAVE", "KNIT")
_DR_PATTERNS = {"WEAVE": ("PLAIN", "TWILL"), "KNIT": ("RIB",)}
_DR_SURFACE_KEYS = {"type", "pattern", "period", "depth", "uvMap"}
_DR_EDGES = ("BOUNDARY", "UV_SEAMS", "SHARP")
_DR_PROFILES = ("ROUND", "V", "FLAT")
_DR_RULE_KEYS = {"type", "polyline", "edges", "width", "depth", "profile", "offset", "dash", "gap", "side"}


def _dr_paths(bm, which, tail):
    """World polylines [[Vector, ...]] along the edges chosen: chains split where three or more meet."""
    if which == "BOUNDARY":
        chosen = [e for e in bm.edges if len(e.link_faces) == 1]
    elif which == "UV_SEAMS":
        chosen = [e for e in bm.edges if e.seam]
    else:
        chosen = [e for e in bm.edges if not e.smooth]
    if not chosen:
        raise MifOpError("edges %s: the mesh has no %s edge. %s" % (
            which, {"BOUNDARY": "open boundary", "UV_SEAMS": "UV seam", "SHARP": "sharp"}[which], tail))
    adj = {}
    for e in chosen:
        for v in e.verts:
            adj.setdefault(v, []).append(e)
    used, paths = set(), []

    def walk(v, e):
        out = [v.co.copy()]
        while e is not None and e not in used:
            used.add(e)
            v = e.other_vert(v)
            out.append(v.co.copy())
            nxt = [x for x in adj[v] if x not in used] if len(adj[v]) == 2 else []
            e = nxt[0] if nxt else None
        return out
    for v, es in adj.items():                    # open chains and branch points first
        if len(es) != 2:
            for e in es:
                if e not in used:
                    paths.append(walk(v, e))
    for e in chosen:                             # closed loops
        if e not in used:
            paths.append(walk(e.verts[0], e))
    return paths


def _dr_surface_rule(k, r, kind, tail):
    """A pattern over the whole surface by UV: {type WEAVE | KNIT, pattern, period (UV units), depth, uvMap}."""
    if set(r) - _DR_SURFACE_KEYS:
        raise MifOpError("rules[%d] (%s) takes %s, got %r. %s" % (k, kind, ", ".join(sorted(_DR_SURFACE_KEYS)),
                                                                  sorted(r), tail))
    pat = str(r.get("pattern", _DR_PATTERNS[kind][0])).upper()
    if pat not in _DR_PATTERNS[kind]:
        raise MifOpError("rules[%d].pattern for %s must be one of %s, got %r. %s" % (k, kind, ", ".join(_DR_PATTERNS[kind]),
                                                                                    r.get("pattern"), tail))
    out = {"type": kind, "pattern": pat, "uvMap": r.get("uvMap")}
    for key in ("period", "depth"):
        if r.get(key) is None:
            raise MifOpError("rules[%d].%s is required. %s" % (k, key, tail))
        out[key] = finite_float(r[key], "rules[%d].%s" % (k, key))
    if not out["period"] > 0:
        raise MifOpError("rules[%d].period must be above 0 (UV units), got %r. %s" % (k, out["period"], tail))
    return out


def _dr_pattern(pat, u, v, p):
    """The surface pattern's height -1..1 at (u, v): PLAIN a checker of bumps, TWILL diagonal ridges, RIB columns."""
    w = 2.0 * math.pi / p
    if pat == "PLAIN":
        return math.sin(w * u) * math.sin(w * v)
    if pat == "TWILL":
        return math.sin(w * (u + v))
    return math.cos(w * u)


def _dr_rule(k, r, tail):
    if isinstance(r, dict) and str(r.get("type", "")).upper() in _DR_PATTERNS:
        return _dr_surface_rule(k, r, str(r["type"]).upper(), tail)
    if not isinstance(r, dict) or set(r) - _DR_RULE_KEYS:
        raise MifOpError("rules[%d] takes %s, got %r. %s" % (k, ", ".join(sorted(_DR_RULE_KEYS)), r, tail))
    kind = str(r.get("type", "")).upper()
    if kind not in _DR_TYPES:
        raise MifOpError("rules[%d].type must be one of %s, got %r. %s" % (k, ", ".join(_DR_TYPES), r.get("type"),
                                                                          tail))
    if (r.get("polyline") is None) == (r.get("edges") is None):
        raise MifOpError("rules[%d] takes polyline [[x,y,z], ...] (world) OR edges (%s). %s"
                         % (k, ", ".join(_DR_EDGES), tail))
    out = {"type": kind}
    if r.get("polyline") is not None:
        pl = r["polyline"]
        if not isinstance(pl, (list, tuple)) or len(pl) < 2:
            raise MifOpError("rules[%d].polyline is 2 or more world points [x,y,z]. %s" % (k, tail))
        out["polyline"] = [Vector(finite_floats(list(p) if isinstance(p, (list, tuple)) else [],
                                                "rules[%d].polyline" % k)) for p in pl]
        if any(len(p) != 3 for p in out["polyline"]):
            raise MifOpError("rules[%d].polyline is 2 or more world points [x,y,z]. %s" % (k, tail))
    else:
        out["edges"] = str(r["edges"]).upper()
        if out["edges"] not in _DR_EDGES:
            raise MifOpError("rules[%d].edges must be one of %s, got %r. %s" % (k, ", ".join(_DR_EDGES), r["edges"],
                                                                                tail))
    for key in ("width", "depth"):
        if r.get(key) is None:
            raise MifOpError("rules[%d].%s is required (world units). %s" % (k, key, tail))
        out[key] = finite_float(r[key], "rules[%d].%s" % (k, key))
    if not out["width"] > 0:
        raise MifOpError("rules[%d].width must be above 0, got %r. %s" % (k, out["width"], tail))
    out["profile"] = str(r.get("profile", "ROUND")).upper()
    if out["profile"] not in _DR_PROFILES:
        raise MifOpError("rules[%d].profile must be one of %s. %s" % (k, ", ".join(_DR_PROFILES), tail))
    out["side"] = str(r.get("side", "BOTH")).upper()
    if out["side"] not in ("BOTH", "LEFT", "RIGHT"):
        raise MifOpError("rules[%d].side must be BOTH, LEFT or RIGHT. %s" % (k, tail))
    if kind == "STITCH":
        for key in ("offset", "dash", "gap"):
            if r.get(key) is None:
                raise MifOpError("rules[%d].%s is required for a STITCH. %s" % (k, key, tail))
            out[key] = finite_float(r[key], "rules[%d].%s" % (k, key))
        if not out["dash"] > 0 or out["gap"] < 0 or out["offset"] < 0:
            raise MifOpError("rules[%d]: dash must be above 0, gap and offset 0 or more. %s" % (k, tail))
    else:
        for key in ("offset", "dash", "gap"):
            if r.get(key) is not None:
                raise MifOpError("rules[%d].%s is a STITCH's; a SEAM is a groove on the line. %s" % (k, key, tail))
    return out


class _DrSegments:
    """The paths' segments with a KD tree of their ends and middles, for the nearest segment to a point."""
    def __init__(self, withacc, KDTree):
        self.segs = []
        for path, acc in withacc:
            for i in range(len(path) - 1):
                self.segs.append((path[i], path[i + 1], acc[i]))
        self.half = max(((b - a).length / 2.0 for a, b, _s in self.segs), default=0.0)
        self.tree = KDTree(3 * len(self.segs))
        for i, (a, b, _s) in enumerate(self.segs):
            for p in (a, (a + b) / 2.0, b):
                self.tree.insert(p, i)
        self.tree.balance()

    def nearest(self, p, n):
        """(distance, arclength along its path, side +1 left / -1 right of the path seen along the normal n)."""
        _co, _i, d0 = self.tree.find(p)
        if d0 is None:
            return None
        best = None
        for _co, i, _d in self.tree.find_range(p, d0 + self.half + 1e-12):
            a, b, acc = self.segs[i]
            ab = b - a
            ll = ab.length_squared
            t = 0.0 if ll < 1e-30 else max(0.0, min(1.0, (p - a).dot(ab) / ll))
            q = a + ab * t
            d = (p - q).length
            if best is None or d < best[0]:
                side = 1.0 if ab.cross(p - q).dot(n) >= 0.0 else -1.0
                best = (d, acc + t * math.sqrt(ll), side)
        return best


def _dr_profile(profile, x):
    """Depth share at x = distance / half width (0 at the centre, 1 at the edge)."""
    if x >= 1.0:
        return 0.0
    if profile == "V":
        return 1.0 - x
    if profile == "FLAT":
        return 1.0
    return 1.0 - x * x


def op_displace_by_rule(params):
    """A high poly's detail by rule: seam grooves and stitch dashes along lines, each vertex moved along its normal.

    THE GAP (spec "A garment builder", step 6 "a high-poly detail pass (seams, stitches, weave or knit) baked to
    tangent normal and AO maps on the low poly"; the raw-Python table's row 10, ~10 runs): the bake exists
    (bake_texture's high-to-low) and nothing made the high poly's detail.

    rules (applied in order, each on the positions the last left): {type, polyline | edges, width, depth, profile,
    side, offset, dash, gap}. The line is a world polyline [[x,y,z], ...] or the mesh's own edges - BOUNDARY (open
    rims), UV_SEAMS or SHARP - chained into paths. SEAM: a groove centred on the line, `width` wide, `depth` deep
    (positive sinks along the vertex normal, negative raises), profile ROUND (1 - x^2, default), V (1 - x) or FLAT, x
    the distance over half the width. STITCH: dashes on lines `offset` from the line, `dash` long with `gap`
    between along the line's length from its start, each `width` across with the profile. side (BOTH, LEFT or
    RIGHT, seen along the normal) keeps one side of the line for either type. Normals are the base mesh's vertex
    normals before any rule. Writes `object` in place, or a copy
    named `name` (the source untouched). Reply per rule: verticesMoved, deepest, paths and their length. Refused,
    nothing changed: an unknown type or edges, width not above 0, a polyline under two points, no edge of the kind
    asked, a STITCH without dash, gap or offset (or a SEAM with them), a width under the mesh's median edge (the
    detail would fall between vertices), shape keys.

    SURFACE PATTERNS (2026-10-07, the weave and knit of step 6): {type WEAVE | KNIT, pattern, period, depth, uvMap}
    moves every vertex along its normal by depth x a height -1..1 of its UV (the map named, or the active one; a seam
    vertex takes its first corner's): WEAVE PLAIN sin(2 pi u / period) x sin(2 pi v / period), a checker of bumps;
    WEAVE TWILL sin(2 pi (u + v) / period), diagonal ridges; KNIT RIB cos(2 pi u / period), columns down v. Positive
    depth raises the peaks. Refused: no UV map, period not above 0, a pattern the type lacks.
    """
    reject_unknown(params, _DISPLACE_KEYS, "displace_by_rule")
    tail = "NOTHING was changed."
    try:
        src = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
    raw = params.get("rules")
    if not isinstance(raw, (list, tuple)) or not raw or len(raw) > 64:
        raise MifOpError("rules is a list of 1-64 {type, polyline | edges, width, depth, ...}. %s" % tail)
    rules = [_dr_rule(k, r, tail) for k, r in enumerate(raw)]
    name = take(params, "name", default=None, kind=str)
    from mathutils.kdtree import KDTree   # noqa: PLC0415 - after the refusals (offline stub)
    bm, shift = _world_edit(src, "displace_by_rule")
    try:
        bm.verts.ensure_lookup_table()
        bm.normal_update()
        if not len(bm.verts):
            raise MifOpError("'%s' has no vertices. %s" % (src.name, tail))
        lens = sorted(e.calc_length() for e in bm.edges)
        med = lens[len(lens) // 2] if lens else 0.0
        for k, r in enumerate(rules):
            if r["type"] in _DR_PATTERNS:
                layer = (bm.loops.layers.uv.get(r["uvMap"]) if r["uvMap"] else bm.loops.layers.uv.active)
                if layer is None:
                    raise MifOpError("rules[%d] (%s) needs a UV map%s on '%s'. %s" % (
                        k, r["type"], " named %r" % r["uvMap"] if r["uvMap"] else "", src.name, tail))
                r["layer"] = layer
                continue
            if r["width"] < med:
                raise MifOpError("rules[%d].width %g is under the mesh's median edge %.4g - the detail would fall "
                                 "between vertices; subdivide first or widen it. %s" % (k, r["width"], med, tail))
        normals = [v.normal.copy() for v in bm.verts]
        tree = KDTree(len(bm.verts))
        for v in bm.verts:
            tree.insert(v.co, v.index)
        tree.balance()
        report = []
        for k, r in enumerate(rules):
            if r["type"] in _DR_PATTERNS:
                moved, deepest, seams = 0, 0.0, 0
                for v in bm.verts:
                    uvs = [tuple(lp[r["layer"]].uv) for lp in v.link_loops]
                    if not uvs:
                        continue
                    if any(abs(a[0] - uvs[0][0]) > 1e-6 or abs(a[1] - uvs[0][1]) > 1e-6 for a in uvs[1:]):
                        seams += 1
                    h = r["depth"] * _dr_pattern(r["pattern"], uvs[0][0], uvs[0][1], r["period"])
                    if abs(h) < 1e-12:
                        continue
                    v.co += normals[v.index] * h
                    moved += 1
                    deepest = max(deepest, abs(h))
                report.append({"type": r["type"], "pattern": r["pattern"], "period": r["period"],
                               "verticesMoved": moved, "deepest": round(deepest, 9), "uvSeamVertices": seams,
                               "uvSeamNote": "a vertex on a UV seam takes its first corner's UV" if seams else None})
                continue
            paths = ([[p - shift for p in r["polyline"]]] if "polyline" in r else _dr_paths(bm, r["edges"], tail))
            withacc = []
            for path in paths:
                acc = [0.0]
                for a, b in zip(path, path[1:]):
                    acc.append(acc[-1] + (b - a).length)
                withacc.append((path, acc))
            segs = _DrSegments(withacc, KDTree)
            reach = (r.get("offset", 0.0) + r["width"] / 2.0) if r["type"] == "STITCH" else r["width"] / 2.0
            near = set()
            for path, _acc in withacc:
                for a, b in zip(path, path[1:]):
                    steps = max(1, int(math.ceil((b - a).length / max(med, reach))))
                    for s in range(steps + 1):
                        c = a.lerp(b, s / steps)
                        for _co, idx, _d in tree.find_range(c, reach + max(med, reach)):
                            near.add(idx)
            moved, deepest = 0, 0.0
            period = r.get("dash", 0.0) + r.get("gap", 0.0)
            for idx in near:
                v = bm.verts[idx]
                hit = segs.nearest(v.co, normals[idx])
                if hit is None:
                    continue
                d, s, side = hit
                if (r["side"] == "LEFT" and side < 0) or (r["side"] == "RIGHT" and side > 0):
                    continue
                if r["type"] == "SEAM":
                    share = _dr_profile(r["profile"], d / (r["width"] / 2.0))
                else:
                    if period > 0 and math.fmod(s, period) >= r["dash"]:
                        continue
                    share = _dr_profile(r["profile"], abs(d - r["offset"]) / (r["width"] / 2.0))
                if share < 1e-6:     # the profile's rim: a float hair inside it would move ~1e-10 and count
                    continue
                v.co -= normals[idx] * (r["depth"] * share)
                moved += 1
                deepest = max(deepest, abs(r["depth"] * share))
            report.append({"type": r["type"], "line": "polyline" if "polyline" in r else r["edges"],
                           "paths": len(withacc), "length": round(sum(a[-1] for _p, a in withacc), 6),
                           "verticesMoved": moved, "deepest": round(deepest, 9)})
        target = src
        if name is not None:
            me = src.data.copy()
            target = bpy.data.objects.new(name, me)
            for col in src.users_collection:
                col.objects.link(target)
            target.matrix_world = src.matrix_world.copy()
        _write_world(bm, target, shift)
    finally:
        bm.free()
    out = {"object": target.name, "source": src.name, "copied": name is not None, "rules": report,
           "requestedName": name, "nameWasSuffixed": name is not None and target.name != name,
           "medianEdge": round(med, 9), "counts": mesh_counts(target)}
    out.update(shared_data_note(target))
    return out


# --------------------------------------------------------------------------------------------- attach_hood

_HOOD_KEYS = ("object", "body", "armature", "rings", "ease", "offset", "edgeLength", "neck", "front", "opening",
              "crownLift", "evaluated")


def _hood_point_at(order_pts, angs, a):
    """The point of a closed polyline (given its points' angles round an axis) where its angle is a."""
    n = len(order_pts)
    for j in range(n):
        d = _wrapped(angs[(j + 1) % n] - angs[j])
        t = _wrapped(a - angs[j])
        if abs(d) > 1e-12 and (t == 0.0 or (t > 0) == (d > 0)) and abs(t) <= abs(d) + 1e-12:
            return order_pts[j].lerp(order_pts[(j + 1) % n], min(1.0, abs(t) / abs(d)))
    return None


def _hood_opening(raw, tail):
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) - {"halfAngle", "bottom", "top", "bone"}:
        raise MifOpError("opening is {halfAngle (degrees round the front), bottom, top (0-1 along bone), bone "
                         "(default the last ring's)}, got %r. %s" % (raw, tail))
    ha = finite_float(raw.get("halfAngle", 60.0), "opening.halfAngle")
    bot = finite_float(raw.get("bottom", 0.1), "opening.bottom")
    top = finite_float(raw.get("top", 0.75), "opening.top")
    if not 0.0 < ha < 180.0:
        raise MifOpError("opening.halfAngle is above 0 and under 180 degrees, got %r. %s" % (ha, tail))
    if not 0.0 <= bot < top <= 1.0:
        raise MifOpError("opening.bottom and top are 0-1 along the bone with bottom under top, got %r and %r. %s"
                         % (bot, top, tail))
    return {"halfAngle": ha, "bottom": bot, "top": top, "bone": raw.get("bone")}


def op_attach_hood(params):
    """Loft a hood up from a garment's neckline round the head, close it over the crown with a centre seam, and open
    it at the face - every new face a quad.

    THE GAP (spec "A garment builder", LEFT "hood, pockets and drawstrings as parametric parts"; the raw-Python table's
    row 4, hoodie_loft x12 and hood_extend x2): loft_garment lofts open tubes and attach_sleeve joins a sleeve; nothing
    grew a hood from a neckline, closed it over the head or opened it at the face.

    `neck` {near: [x,y,z]} (world; default the open rim nearest the first ring's bone) picks the garment's neckline,
    which must have an even number of vertices. Its vertices are re-spaced along the rim evenly by angle round the
    first ring's bone line, vertex 0 at `front` (default [0, -1, 0], Blender's front view), so each vertex has a
    mirror partner across the sagittal plane. rings [{bone, at, ease, offset}] (1 or more, as loft_garment's: each the
    body's own section square to its bone, eased out by ease, default 0.25) are lofted at the same angles, rows at
    edgeLength (default the neckline's edge). Over the last ring the crown closes: each vertex meets its mirror partner
    halfway, raised along the bone by crownLift (default half the last ring's mean radius) - a centre seam of k/2 + 1
    vertices from front to back. `opening` {halfAngle (default 60 degrees either side of front), bottom, top (0-1 along
    `bone`, default the last ring's; defaults 0.1 and 0.75)} removes the hood faces inside it - the face opening, one
    hole or refused. Winding follows the garment's; the new faces' UVs are one island right of the others, in world
    units. Reply: neckline {vertices, perimeter, largestMove}, rings, crown {vertices, points, lift}, opening
    {facesRemoved}, newFaces, counts. Refused, nothing changed: no open rim near `neck`, an odd neckline, a neckline
    that does not go round the bone line, front along the bone, a ray from a ring missing the body, a garment face
    turned over by the re-spacing, an opening that is not one hole, a bone not on the armature, shape keys.
    """
    reject_unknown(params, _HOOD_KEYS, "attach_hood")
    tail = "NOTHING was changed."
    try:
        obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
        body = get_object(take(params, "body", required=True, kind=str), want_mesh=True)
        arm = get_object(take(params, "armature", required=True, kind=str))
        ease0 = take_float(params, "ease", default=0.25)
        off0 = take_float(params, "offset", default=0.0)
        el = take_float(params, "edgeLength", default=None)
        lift = take_float(params, "crownLift", default=None)
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
    evaluated = take_bool(params, "evaluated", default=True)
    if arm.type != "ARMATURE":
        raise MifOpError("'%s' is a %s, not an ARMATURE. %s" % (arm.name, arm.type, tail))
    if obj is body:
        raise MifOpError("object and body are both '%s'. %s" % (obj.name, tail))
    if not (math.isfinite(ease0) and ease0 > -1.0) or not math.isfinite(off0):
        raise MifOpError("ease must be above -1 and offset finite. %s" % tail)
    if el is not None and not (math.isfinite(el) and el > 0):
        raise MifOpError("edgeLength must be above 0, got %r. %s" % (el, tail))
    if lift is not None and not (math.isfinite(lift) and lift >= 0):
        raise MifOpError("crownLift must be 0 or more, got %r. %s" % (lift, tail))
    front = Vector(finite_floats(list(params.get("front", [0.0, -1.0, 0.0])), "front"))
    if len(front) != 3 or front.length < 1e-9:
        raise MifOpError("front is a world direction [x, y, z], not zero. %s" % tail)
    front.normalize()
    rings = _lg_rings(params.get("rings"), arm, tail, least=1)
    opening = _hood_opening(params.get("opening"), tail)
    neck = params.get("neck")
    if neck is not None and (not isinstance(neck, dict) or set(neck) != {"near"}):
        raise MifOpError("neck is {near: [x, y, z]} (world), got %r. %s" % (neck, tail))
    near = Vector(finite_floats(list(neck["near"]), "neck.near")) if neck else None
    if near is not None and len(near) != 3:
        raise MifOpError("neck.near is [x, y, z]. %s" % tail)
    if opening is not None:
        ob_name = opening["bone"] or rings[-1][0].name
        op_bone = arm.pose.bones.get(str(ob_name))
        if op_bone is None:
            raise MifOpError("opening.bone: '%s' has no bone %r. %s" % (arm.name, ob_name, tail))
    from mathutils.geometry import intersect_line_line_2d   # noqa: PLC0415 - after the refusals (offline stub)

    frames = []
    for pb, at, ease, off in rings:
        c, n, _ux = _lg_frame(arm, pb, at, tail)
        ux = front - n * front.dot(n)
        if ux.length < 1e-6:
            raise MifOpError("front %s runs along bone '%s' - it has no direction round the bone. %s"
                             % (list(front), pb.name, tail))
        frames.append((pb, at, c, n, ux.normalized(), ease0 if ease is None else ease, off0 if off is None else off))
    bmb = evaluated_world(body, evaluated=evaluated)
    try:
        if not len(bmb.faces):
            raise MifOpError("body '%s' has no faces. %s" % (body.name, tail))
        sections = [_lg_section(bmb, c, n) for _pb, _at, c, n, _ux, _e, _o in frames]
    finally:
        bmb.free()
    pb0, _at0, c0w, n0, ux0, _e0, _o0 = frames[0]
    uy0 = n0.cross(ux0)

    bm, shift = _world_edit(obj, "attach_hood")
    try:
        bm.faces.ensure_lookup_table()
        bm.normal_update()
        from .ops_topology import _ordered   # noqa: PLC0415
        loops = _pick_loops(bm)
        if not loops:
            raise MifOpError("'%s' has no open rim to grow a hood from. %s" % (obj.name, tail))
        target = (near if near is not None else (arm.matrix_world @ pb0.head)) - shift
        cents = [(sum((v.co for v in vs), Vector()) / len(vs), (e, vs)) for e, vs in loops]
        cen, lp = min(cents, key=lambda x: (x[0] - target).length)
        rad_loop = sum((v.co - cen).length for v in lp[1]) / len(lp[1])
        if near is not None and (cen - target).length > rad_loop:
            raise MifOpError("neck.near %s is %.4g from the nearest open rim's centre, beyond its radius %.4g - no "
                             "neckline there. %s" % (list(near), (cen - target).length, rad_loop, tail))
        order = _ordered(lp)
        if order is None or len(order) < 4:
            raise MifOpError("the neckline is not a clean ring (a bow-tie vertex, or under four vertices). %s" % tail)
        k = len(order)
        if k % 2:
            raise MifOpError("the neckline has %d vertices; a hood's centre seam pairs each with its mirror, so it "
                             "needs an even number - loft the garment with an even segment count. %s" % (k, tail))
        c0 = c0w - shift

        def ang(p):
            d = p - c0
            return math.atan2(d.dot(uy0), d.dot(ux0))
        pts0 = [v.co.copy() for v in order]
        angs = [ang(p) for p in pts0]
        turn = sum(_wrapped(angs[(i + 1) % k] - angs[i]) for i in range(k))
        if abs(abs(turn) - 2.0 * math.pi) > 0.5:
            raise MifOpError("the neckline does not go round bone '%s''s line (it turns %.2f rad). %s"
                             % (pb0.name, turn, tail))
        sgn = 1.0 if turn > 0 else -1.0
        # vertex 0 at the front: rotate the ring so the vertex nearest angle 0 leads
        lead = min(range(k), key=lambda i: abs(_wrapped(angs[i])))
        order = order[lead:] + order[:lead]
        pts0 = pts0[lead:] + pts0[:lead]
        angs = angs[lead:] + angs[:lead]
        targets = [_wrapped(sgn * 2.0 * math.pi * i / k) for i in range(k)]
        adj = {f for v in order for f in v.link_faces}
        normals0 = {f: f.normal.copy() for f in adj}
        big = 0.0
        new_pos = []
        for a in targets:
            p = _hood_point_at(pts0, angs, a)
            if p is None:
                raise MifOpError("the neckline has no point at angle %.3f round bone '%s'. %s" % (a, pb0.name, tail))
            new_pos.append(p)
        for v, p in zip(order, new_pos):
            big = max(big, (p - v.co).length)
            v.co = p
        for f in adj:
            f.normal_update()
        flipped = sum(1 for f in adj if f.normal.dot(normals0[f]) < 0.0)
        if flipped:
            raise MifOpError("re-spacing the neckline turned %d face(s) of '%s' over (largest move %.4g). %s"
                             % (flipped, obj.name, big, tail))

        built, rows = [], []
        for kk, ((pb, at, c, n, ux, e, o), sec) in enumerate(zip(frames, sections)):
            ring, body_pts, missed = _lg_cast(sec, c, n, ux, targets, e, o, intersect_line_line_2d)
            if missed:
                raise MifOpError("ring %d (bone '%s' at %g): %d of %d rays met no body surface - the bone there is not "
                                 "inside '%s'. %s" % (kk, pb.name, at, missed, k, body.name, tail))
            built.append([p - shift for p in ring])
            rows.append(_lg_row(pb, at, c, e, o, body_pts, ring))
        rim = [v.co.copy() for v in order]
        perim = _ring_length(rim)
        step = el or perim / k
        full = [rim]
        for a, b in zip([rim] + built, built):
            gap = sum((p - q).length for p, q in zip(a, b)) / k
            nrow = max(1, int(round(gap / step)))
            for t in range(1, nrow + 1):
                full.append([p.lerp(q, t / nrow) for p, q in zip(a, b)])
        last = built[-1]
        n_last = frames[-1][3]
        mean_r = sum((p - (frames[-1][2] - shift)).length for p in last) / k
        lift_v = mean_r * 0.5 if lift is None else lift
        seam = [((last[i] + last[(k - i) % k]) / 2.0) + n_last * lift_v for i in range(k // 2 + 1)]
        crown_gap = sum((last[i] - seam[min(i, k - i)]).length for i in range(k)) / k
        ncrown = max(1, int(round(crown_gap / step)))
        crown_rows = [[last[i].lerp(seam[min(i, k - i)], t / ncrown) for i in range(k)] for t in range(1, ncrown)]
        if len(bm.faces) + (len(full) + len(crown_rows)) * k > 500000:
            raise MifOpError("the hood would take '%s' over 500,000 faces - raise edgeLength. %s" % (obj.name, tail))
        fwd = _winding(order)
        vrows = [order] + [[bm.verts.new(p) for p in row] for row in full[1:] + crown_rows]
        seam_v = [bm.verts.new(p) for p in seam]
        vrows.append([seam_v[min(i, k - i)] for i in range(k)])
        made = _band(bm, vrows, fwd)
        uv = bm.loops.layers.uv.active
        if uv is not None:
            # _band makes the faces row by row, column by column: face q spans rows q // k and q // k + 1, columns
            # q % k and the next - a crown vertex shared by a mirror pair still sits in this face's own column
            made_set = set(made)
            u_right = max((lp_[uv].uv.x for f in bm.faces if f not in made_set for lp_ in f.loops), default=0.0)
            u0 = u_right + 0.05 * perim
            allrows = full + crown_rows + [[seam[min(i, k - i)] for i in range(k)]]
            vpos = [0.0]
            for a, b in zip(allrows, allrows[1:]):
                vpos.append(vpos[-1] + sum((p - q).length for p, q in zip(a, b)) / k)
            for q, f in enumerate(made):
                r_lo, i = q // k, q % k
                j = (i + 1) % k
                where = {}
                for r_i in (r_lo, r_lo + 1):
                    where.setdefault(vrows[r_i][i], (r_i, i))
                    where.setdefault(vrows[r_i][j], (r_i, i + 1))
                for lp_ in f.loops:
                    r_i, col = where[lp_.vert]
                    lp_[uv].uv = (u0 + perim * col / k, vpos[r_i])
        removed = 0
        if opening is not None:
            ob = arm.pose.bones.get(str(opening["bone"] or rings[-1][0].name))
            oh, ot = arm.matrix_world @ ob.head - shift, arm.matrix_world @ ob.tail - shift
            on = (ot - oh)
            olen = on.length
            on.normalize()
            oux = (front - on * front.dot(on)).normalized()
            ouy = on.cross(oux)
            cut = []
            for f in made:
                cc = f.calc_center_median()
                d = cc - oh
                s = d.dot(on) / olen
                a = math.degrees(abs(math.atan2(d.dot(ouy), d.dot(oux))))
                if opening["bottom"] <= s <= opening["top"] and a <= opening["halfAngle"]:
                    cut.append(f)
            if not cut:
                raise MifOpError("the opening holds no hood face - widen halfAngle or the bottom-top span. %s" % tail)
            before = {frozenset(vs) for _e, vs in _pick_loops(bm)}
            made = [f for f in made if f not in set(cut)]
            removed = len(cut)
            bmesh.ops.delete(bm, geom=cut, context="FACES_ONLY")
            loose = [e for e in bm.edges if not e.link_faces]
            if loose:
                bmesh.ops.delete(bm, geom=loose, context="EDGES")
            lone = [v for v in bm.verts if not v.link_edges]
            if lone:
                bmesh.ops.delete(bm, geom=lone, context="VERTS")
            after = _pick_loops(bm)
            new = [lp2 for lp2 in after if frozenset(lp2[1]) not in before]
            if len(new) != 1 or len(after) != len(before) + 1:
                raise MifOpError("the face opening is not one hole in the hood (%d new open loop(s)) - it reaches the "
                                 "neckline or the crown; narrow it. %s" % (len(new), tail))
        _write_world(bm, obj, shift)
        new_faces = len(made)
    finally:
        bm.free()
    out = {"object": obj.name, "neckline": {"vertices": k, "perimeter": round(perim, 6), "largestMove": round(big, 6)},
           "rings": rows, "rows": len(full) + len(crown_rows) + 1, "segments": k, "edgeLength": round(step, 6),
           "crown": {"vertices": len(seam), "lift": round(lift_v, 6),
                     "points": [[round(x, 6) for x in (p + shift)] for p in seam]},
           "opening": dict(opening or {}, facesRemoved=removed) if opening else None,
           "newFaces": new_faces,
           "uvMap": obj.data.uv_layers.active.name if obj.data.uv_layers.active else None,
           "counts": mesh_counts(obj)}
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- build_sole

_SOLE_KEYS = ("body", "armature", "bone", "margin", "thickness", "heelThickness", "segments", "ground", "sampleHeight",
              "evaluated", "name", "collection", "takeOverName", "last", "footprintTop", "heelLength")
_SOLE_LEVELS = 6


def _far_crossing(loop, cx, cy, dx, dy):
    """The farthest crossing of the ray from (cx, cy) along (dx, dy) with a closed loop, as a distance - or None."""
    best = None
    for i in range(len(loop)):
        (x1, y1), (x2, y2) = loop[i - 1], loop[i]
        ex, ey = x2 - x1, y2 - y1
        den = dx * ey - dy * ex
        if abs(den) < 1e-15:
            continue
        t = ((x1 - cx) * ey - (y1 - cy) * ex) / den
        u = ((x1 - cx) * dy - (y1 - cy) * dx) / den
        if t > 0 and -1e-9 <= u <= 1 + 1e-9 and (best is None or t > best):
            best = t
    return best


def _sole_heel_block(out, ground, thick, heel, head, dhn, heel_len, tail):
    """A sole whose heel is a BLOCK: the top flat at ground + heel from the footprint's back to heel_len along the bone,
    flat at ground + thick ahead of it, and a vertical breast between them across the outline. (verts, faces, uvs,
    breast {along, points}). The outline is cut where the breast line crosses it - twice, or it is refused."""
    from mathutils.geometry import tessellate_polygon   # noqa: PLC0415

    def along(p):
        return (p[0] - head.x) * dhn.x + (p[1] - head.y) * dhn.y
    al = [along(p) for p in out]
    back, front = min(al), max(al)
    cut = back + heel_len
    if not back < cut < front - 1e-4:
        raise MifOpError("heelLength %.4g reaches past the footprint, which is %.4g long along the bone - the heel block "
                         "must end before the toe. %s" % (heel_len, front - back, tail))
    ring = []                               # (x, y, label) - H heel side, F fore side, B on the breast line
    n = len(out)
    for i in range(n):
        a, b = out[i], out[(i + 1) % n]
        sa, sb = al[i] < cut, al[(i + 1) % n] < cut
        ring.append((a[0], a[1], "H" if sa else "F"))
        if sa != sb:
            t = (cut - al[i]) / (al[(i + 1) % n] - al[i])
            ring.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, "B"))
    bidx = [i for i, r in enumerate(ring) if r[2] == "B"]
    if len(bidx) != 2:
        raise MifOpError("the breast line at heelLength %.4g crosses the outline %d times, not twice - the footprint is "
                         "not convex across it. Pass another heelLength. %s" % (heel_len, len(bidx), tail))
    m = len(ring)
    verts = [[r[0], r[1], ground] for r in ring]              # bottom ring: index i
    top = {}
    for i, r in enumerate(ring):
        if r[2] == "H":
            top[(i, "H")] = len(verts)
            verts.append([r[0], r[1], ground + heel])
        elif r[2] == "F":
            top[(i, "F")] = len(verts)
            verts.append([r[0], r[1], ground + thick])
        else:
            top[(i, "F")] = len(verts)
            verts.append([r[0], r[1], ground + thick])
            top[(i, "H")] = len(verts)
            verts.append([r[0], r[1], ground + heel])

    faces, uvs = [], []

    def cap(idx, up):
        # A FAN FROM THE PIECE'S CENTRE, as the plain sole's caps: a breast point lies ON a straight edge of the
        # outline, and a fill that ends a triangle on it makes a sliver of no area. A piece the fan would fold over
        # (not star-shaped round its centre) is filled by Blender's polygon fill instead.
        cx_ = sum(verts[k][0] for k in idx) / len(idx)
        cy_ = sum(verts[k][1] for k in idx) / len(idx)

        def turn(a, b):
            return (a[0] - cx_) * (b[1] - cy_) - (a[1] - cy_) * (b[0] - cx_)
        if all(turn(verts[idx[q]], verts[idx[(q + 1) % len(idx)]]) > 1e-12 for q in range(len(idx))):
            c = len(verts)
            verts.append([cx_, cy_, verts[idx[0]][2]])
            for q in range(len(idx)):
                f = [c, idx[q], idx[(q + 1) % len(idx)]]
                faces.append(f if up else f[::-1])
                uvs.append([[verts[k][0], verts[k][1]] for k in faces[-1]])
            return
        pts = [Vector((verts[k][0], verts[k][1], 0.0)) for k in idx]
        for tri in tessellate_polygon([pts]):
            f = [idx[t] for t in tri]
            a, b, c = (verts[k] for k in f)
            area2 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if (area2 > 0) != up:
                f = f[::-1]
            faces.append(f)
            uvs.append([[verts[k][0], verts[k][1]] for k in f])

    cap(list(range(m)), up=False)
    # the two tops: each a run of the ring from one breast point to the other, closed by the breast chord
    for side in ("H", "F"):
        start = next(b for b in bidx if ring[(b + 1) % m][2] == side)
        run, i = [], start
        while True:
            run.append(top[(i, side)])
            i = (i + 1) % m
            if ring[i][2] == "B":
                run.append(top[(i, side)])
                break
        cap(run, up=True)
    run_len = [0.0]
    for i in range(m):
        j = (i + 1) % m
        run_len.append(run_len[-1] + math.hypot(ring[j][0] - ring[i][0], ring[j][1] - ring[i][1]))
    for i in range(m):
        j = (i + 1) % m
        side = ring[j][2] if ring[i][2] == "B" else ring[i][2]
        ti, tj = top[(i, side)], top[(j, side)]
        u0, u1 = run_len[i], run_len[i + 1]
        if side == "H" and ring[j][2] == "B":
            lo = top[(j, "F")]
            faces.append([i, j, lo])
            uvs.append([[u0, 0.0], [u1, 0.0], [u1, thick]])
            faces.append([i, lo, tj, ti])
            uvs.append([[u0, 0.0], [u1, thick], [u1, heel], [u0, heel]])
        elif side == "H" and ring[i][2] == "B":
            lo = top[(i, "F")]
            faces.append([i, j, lo])
            uvs.append([[u0, 0.0], [u1, 0.0], [u0, thick]])
            faces.append([lo, j, tj, ti])
            uvs.append([[u0, thick], [u1, 0.0], [u1, heel], [u0, heel]])
        else:
            faces.append([i, j, tj, ti])
            uvs.append([[u0, 0.0], [u1, 0.0], [u1, verts[tj][2] - ground], [u0, verts[ti][2] - ground]])
    b1, b2 = bidx
    breast = [top[(b1, "F")], top[(b2, "F")], top[(b2, "H")], top[(b1, "H")]]
    # FACING THE TOES: the breast is the heel block's front face
    e1 = Vector(verts[breast[1]]) - Vector(verts[breast[0]])
    e2 = Vector(verts[breast[3]]) - Vector(verts[breast[0]])
    if e1.cross(e2).dot(Vector((dhn.x, dhn.y, 0.0))) < 0:
        breast = breast[::-1]
    chord = math.hypot(ring[b2][0] - ring[b1][0], ring[b2][1] - ring[b1][1])
    faces.append(breast)
    uvs.append([[0.0, verts[k][2] - ground] if k in (top[(b1, "F")], top[(b1, "H")]) else
                [chord, verts[k][2] - ground] for k in breast])
    return verts, faces, uvs, {"along": round(cut - back, 6), "points": [[round(ring[b][0], 6), round(ring[b][1], 6)]
                                                                         for b in bidx]}


def _sole_loops(bm, z):
    """The body's closed section loops in the plane z: [[(x, y), ...]] from a bisect of a copy."""
    cp = bm.copy()
    try:
        res = bmesh.ops.bisect_plane(cp, geom=cp.verts[:] + cp.edges[:] + cp.faces[:], dist=1e-7,
                                     plane_co=Vector((0.0, 0.0, z)), plane_no=Vector((0.0, 0.0, 1.0)))
        cut = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
        adj = {}
        for e in cut:
            for v in e.verts:
                adj.setdefault(v, []).append(e)
        seen, loops = set(), []
        for e0 in cut:
            if e0 in seen:
                continue
            start = e0.verts[0]
            pts, v, e = [start], start, e0
            ok = True
            while True:
                seen.add(e)
                v = e.other_vert(v)
                if v is start:
                    break
                pts.append(v)
                nxt = [x for x in adj.get(v, []) if x is not e and x not in seen]
                if len(adj.get(v, [])) != 2 or not nxt:
                    ok = False
                    break
                e = nxt[0]
            if ok and len(pts) >= 3:
                loops.append([(p.co.x, p.co.y) for p in pts])
        return loops
    finally:
        cp.free()


def _poly_contains(pt, poly):
    x, y = pt
    c = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[i - 1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


def _poly_area(poly):
    return 0.5 * sum(poly[i - 1][0] * poly[i][1] - poly[i][0] * poly[i - 1][1] for i in range(len(poly)))


def op_build_sole(params):
    """A shoe sole round a foot's footprint: the body's section near the ground under a foot bone, eased out by a
    margin, from the ground up to a thickness that rises toward the heel - a closed solid.

    THE GAP (spec "A garment builder", LEFT "the footwear builder: a shoe last from foot bones ... lofted upper, sole,
    eyelets and laces"; raw-Python rows sneaker3, shoe_parts, outfit_sneaker): the upper is loft_garment round the
    foot bone, eyelets and laces are cut_eyelets and build_tube, and nothing made the sole from the foot.

    `bone` on `armature` (posed, world) is the foot, head at the heel and tail toward the toes (not vertical). ground
    (default the lowest point of `body` - as it evaluates unless evaluated is false - within the bone's length of the
    bone, seen from above) is where the bottom sits; the footprint is the body's closed section at sampleHeight (default
    ground + a tenth of the bone's length) that holds the bone's middle seen from above, or else the nearest one within
    the bone's length - so the other foot is never taken. Its outline, resampled to `segments` (default 64) points by
    angle round its centre (farthest crossing), is pushed out along its own normal by margin (default 0.005); the top
    sits at ground + thickness at the toe end and heelThickness (default thickness) at the heel end, linearly along the
    bone. Reply: ground, sampleHeight, footprint {area, perimeter}, outline {vertices, area, perimeter}, the made mesh's
    counts. Refused, nothing created: a bone the armature lacks or a vertical one, margin under 0, thickness or
    heelThickness not above 0, segments outside 8-1024, no part of the body near the bone at sampleHeight.
    last (2026-10-08) names a measured shoe last (SHOE_LAST, e.g. "sneaker"): margin, thickness and heelThickness
    default to its side room, sole under the ball and sole under the heel, each a share of the footprint's length along
    the bone - a value passed wins. thickness is required without a last. The reply's last gives the name, its source,
    footLength and the ratios. An unmeasured last is refused, naming the measured ones.
    """
    reject_unknown(params, _SOLE_KEYS, "build_sole")
    tail = "NOTHING was created."
    try:
        body = get_object(take(params, "body", required=True, kind=str), want_mesh=True)
        arm = get_object(take(params, "armature", required=True, kind=str))
        last_name = params.get("last")
        if last_name is not None and (not isinstance(last_name, str) or last_name not in SHOE_LAST):
            raise MifOpError("last %r has no measured table; the measured ones are: %s. Pass margin, thickness and "
                             "heelThickness instead - no last is guessed." % (last_name, ", ".join(sorted(SHOE_LAST))))
        margin = take_float(params, "margin", default=None)
        thick = take_float(params, "thickness", required=last_name is None)
        heel = take_float(params, "heelThickness", default=None)
        segments = take_int(params, "segments", default=64)
        ground = take_float(params, "ground", default=None)
        sample = take_float(params, "sampleHeight", default=None)
        top_z = take_float(params, "footprintTop", default=None)
        heel_len = take_float(params, "heelLength", default=None)
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
    if heel_len is not None and not heel_len > 0:
        raise MifOpError("heelLength must be above 0 (metres from the footprint's back along the bone), got %r. %s"
                         % (heel_len, tail))
    evaluated = take_bool(params, "evaluated", default=True)
    if arm.type != "ARMATURE":
        raise MifOpError("'%s' is a %s, not an ARMATURE. %s" % (arm.name, arm.type, tail))
    bname = take(params, "bone", required=True, kind=str)
    pb = arm.pose.bones.get(str(bname))
    if pb is None:
        raise MifOpError("'%s' has no bone %r. %s" % (arm.name, bname, tail))
    if last_name is None:
        margin = 0.005 if margin is None else margin
        heel = thick if heel is None else heel
    if margin is not None and not (math.isfinite(margin) and margin >= 0):
        raise MifOpError("margin must be 0 or more, got %r. %s" % (margin, tail))
    if any(v is not None and not (math.isfinite(v) and v > 0) for v in (thick, heel)):
        raise MifOpError("thickness and heelThickness must be above 0, got %r and %r. %s" % (thick, heel, tail))
    if not 8 <= segments <= 1024:
        raise MifOpError("segments is 8-1024, got %r. %s" % (segments, tail))
    head, tl = arm.matrix_world @ pb.head, arm.matrix_world @ pb.tail
    dh = Vector((tl.x - head.x, tl.y - head.y))
    blen = dh.length
    if blen < 1e-6:
        raise MifOpError("bone '%s' is vertical (or has no length) seen from above - a foot bone runs heel to toe. %s"
                         % (pb.name, tail))
    dhn = dh / blen
    mid = ((head.x + tl.x) / 2.0, (head.y + tl.y) / 2.0)

    def near_bone(x, y):
        rel = Vector((x - head.x, y - head.y))
        t = max(0.0, min(blen, rel.dot(dhn)))
        return (rel - dhn * t).length

    bm = evaluated_world(body, evaluated=evaluated)
    try:
        if ground is None:
            zs = [v.co.z for v in bm.verts if near_bone(v.co.x, v.co.y) <= blen]
            if not zs:
                raise MifOpError("no part of '%s' lies within %.4g of bone '%s' seen from above. %s"
                                 % (body.name, blen, pb.name, tail))
            ground = min(zs)
        sample = ground + 0.1 * blen if sample is None else sample
        # THE FOOTPRINT IS THE FOOT SEEN FROM ABOVE, from sampleHeight up to footprintTop (default the bone's higher
        # end, the ankle), not one section near the ground (2026-11-07). A boot's heel counter stands above its toe,
        # so the section near the ground held the forefoot alone, and the sole came out short, thin and slanted, with
        # no heel under the heel. Each level's section loop is picked as before - the one holding the bone's middle,
        # else the nearest within the bone's length - so the other foot is never taken.
        if top_z is None:
            top_z = max(head.z, tl.z)
        if top_z < sample:
            if params.get("footprintTop") is not None and top_z < sample - 1e-6:
                raise MifOpError("footprintTop %.4g is below sampleHeight %.4g - the footprint is taken from "
                                 "sampleHeight up. %s" % (top_z, sample, tail))
            top_z = sample
        n_lv = 1 if top_z - sample < 1e-6 else _SOLE_LEVELS
        levels = [sample + (top_z - sample) * j / max(1, n_lv - 1) for j in range(n_lv)]
        sections = []
        for z in levels:
            lps = _sole_loops(bm, z)
            got = [lp for lp in lps if _poly_contains(mid, lp)]
            if not got:
                scored = sorted(((min(near_bone(x, y) for x, y in lp), lp) for lp in lps), key=lambda s: s[0])
                got = [lp for d, lp in scored[:1] if d <= blen]
            sections.append((z, len(lps), got[0] if got else None))
    finally:
        bm.free()
    loops_at_sample = sections[0][1]
    if sections[0][2] is None:
        raise MifOpError("no part of '%s' is cut at sampleHeight %.4g near bone '%s' (%d section loop(s), none holding or "
                         "near the bone). %s" % (body.name, sample, pb.name, loops_at_sample, tail))
    picked = [sec[2] if _poly_area(sec[2]) >= 0 else sec[2][::-1] for sec in sections if sec[2] is not None]
    foot = picked[0]
    # THE CENTRE IS THE MEAN OF THE LEVELS' CENTRES, and the outline the farthest crossing of every level's loop by
    # angle round it - the union of the sections seen from above.
    cx = sum(sum(p[0] for p in lp) / len(lp) for lp in picked) / len(picked)
    cy = sum(sum(p[1] for p in lp) / len(lp) for lp in picked) / len(picked)
    ring = []
    for s in range(segments):
        a = 2.0 * math.pi * s / segments
        dx, dy = math.cos(a), math.sin(a)
        best = None
        for lp in picked:
            t = _far_crossing(lp, cx, cy, dx, dy)
            if t is not None and (best is None or t > best):
                best = t
        if best is None:
            raise MifOpError("the footprint's outline has no point at angle %.3f round its centre. %s" % (a, tail))
        ring.append((cx + dx * best, cy + dy * best))
    last_out = None
    if last_name is not None:
        # the measured last's shares of the foot's length: the footprint's extent along the bone (the union
        # seen from above, so the heel's back counts)
        along = [(x - head.x) * dhn.x + (y - head.y) * dhn.y for x, y in ring]
        flen = max(along) - min(along)
        rat = SHOE_LAST[last_name]["ratios"]
        margin = rat["side"] * flen if margin is None else margin
        thick = rat["soleBall"] * flen if thick is None else thick
        heel = rat["soleHeel"] * flen if heel is None else heel
        last_out = {"name": last_name, "source": SHOE_LAST[last_name]["source"], "footLength": round(flen, 6),
                    "ratios": dict(rat)}
    out = []
    for i in range(segments):
        (xa, ya), (xb, yb) = ring[i - 1], ring[(i + 1) % segments]
        nx, ny = yb - ya, -(xb - xa)            # right of the counter-clockwise tangent: outward
        ln = math.hypot(nx, ny) or 1.0
        out.append((ring[i][0] + nx / ln * margin, ring[i][1] + ny / ln * margin))
    if heel_len is not None:
        verts, faces, uvs, breast = _sole_heel_block(out, ground, thick, heel, head, dhn, heel_len, tail)
    else:
        verts, faces, uvs, breast = _sole_ramp(out, ground, thick, heel, head, dhn, blen)
    k = segments
    from .ops_create import op_create_mesh   # noqa: PLC0415 - create_mesh owns naming and the read-back
    sub = {"name": params.get("name") or body.name + "_sole", "vertices": verts, "faces": faces, "uvs": {"UVMap": uvs}}
    for kk in _PLACE_KEYS[1:]:
        if params.get(kk) is not None:
            sub[kk] = params[kk]
    res = op_create_mesh(sub)

    def perim(p):
        return sum(math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]) for i in range(len(p)))
    res["op"] = "build_sole"
    res["ground"] = round(ground, 6)
    res["sampleHeight"] = round(sample, 6)
    res["footprint"] = {"area": round(abs(_poly_area(ring)), 6), "perimeter": round(perim(ring), 6),
                        "loopsAtSample": loops_at_sample, "top": round(levels[-1], 6),
                        "levels": [{"z": round(z, 6), "loops": nl, "picked": lp is not None} for z, nl, lp in sections],
                        "areaAtSample": round(abs(_poly_area(foot)), 6)}
    along = [(x - head.x) * dhn.x + (y - head.y) * dhn.y for x, y in out]
    res["outline"] = {"vertices": k, "area": round(abs(_poly_area(out)), 6), "perimeter": round(perim(out), 6),
                      "lengthAlongBone": round(max(along) - min(along), 6)}
    res["margin"], res["thickness"], res["heelThickness"] = margin, thick, heel
    res["heel"] = ({"shape": "block", "length": heel_len, "height": heel, "breast": breast} if heel_len is not None
                   else {"shape": "ramp", "height": heel,
                         "note": "the top rises linearly from thickness at the toe end to heelThickness at the heel "
                                 "end; heelLength makes a heel block with a breast instead"})
    if last_out is not None:
        res["last"] = last_out
    return res


def _sole_ramp(out, ground, thick, heel, head, dhn, blen):
    """The plain sole: its top rising linearly along the bone from thick at the toe end to heel at the heel end, both
    caps fans from the outline's centre. (verts, faces, uvs, None)."""
    verts, faces = [], []
    for x, y in out:
        verts.append([x, y, ground])
    for x, y in out:
        t = max(0.0, min(1.0, Vector((x - head.x, y - head.y)).dot(dhn) / blen))
        verts.append([x, y, ground + heel + (thick - heel) * t])
    k = len(out)
    # FANS AND UVS, so the sole passes mesh_quality gameReady as it comes: its outline is star-shaped round (cx, cy) by
    # construction, so a fan from there holds, and UVs are in meters, the caps seen from above and the wall along the rim.
    mx = sum(p[0] for p in out) / k
    my = sum(p[1] for p in out) / k
    t_mid = max(0.0, min(1.0, Vector((mx - head.x, my - head.y)).dot(dhn) / blen))
    bottom_mid, top_mid = len(verts), len(verts) + 1
    verts.append([mx, my, ground])
    verts.append([mx, my, ground + heel + (thick - heel) * t_mid])
    uvs = []
    run = [0.0]
    for i in range(k):
        j = (i + 1) % k
        run.append(run[-1] + math.hypot(out[j][0] - out[i][0], out[j][1] - out[i][1]))
    for i in range(k):
        j = (i + 1) % k
        faces.append([bottom_mid, j, i])
        uvs.append([[mx, my], list(out[j]), list(out[i])])
        faces.append([top_mid, k + i, k + j])
        uvs.append([[mx, my], list(out[i]), list(out[j])])
        faces.append([i, j, k + j, k + i])
        uvs.append([[run[i], 0.0], [run[i + 1], 0.0], [run[i + 1], verts[k + j][2] - ground], [run[i], verts[k + i][2] - ground]])
    return verts, faces, uvs, None


# --------------------------------------------------------------------------------------------- envelopes, points, fits

def _world_tree(obj, tail):
    """BVHTree of a mesh as it evaluates now, in world space; refused when it has no faces."""
    from mathutils.bvhtree import BVHTree
    bm = evaluated_world(obj, True)
    try:
        if not len(bm.faces):
            raise MifOpError("'%s' has no faces. %s" % (obj.name, tail))
        return BVHTree.FromBMesh(bm)
    finally:
        bm.free()


_HIT_STEP = 1e-5


def _hits_along(tree, origin, d, reach, cap=256):
    """[(distance, entering)] of every crossing of the ray origin + t d (0 < t <= reach) with a world BVH, nearest first.
    Each cast restarts _HIT_STEP past the last hit, so a face is never met twice in one place."""
    out, start, gone = [], origin, 0.0
    while len(out) < cap and gone < reach:
        loc, nrm, _fi, dist = tree.ray_cast(start, d, reach - gone)
        if loc is None:
            break
        gone += dist
        out.append((gone, nrm.dot(d) < 0.0))
        start = loc + d * _HIT_STEP
        gone += _HIT_STEP
    return out


_POINT_SOURCES = ("CURVES", "POINTCLOUD", "MESH")
_MAX_POINTS = 500_000


def _source_points(name, tail):
    """(world points, strands [(first point, count)] for a Curves object else None, the object): a hair Curves object, a
    point cloud or a mesh's vertices, each as it evaluates now."""
    obj = get_object(name)
    if obj.type not in _POINT_SOURCES:
        raise MifOpError("'%s' is a %s - points come from a hair CURVES object, a POINTCLOUD or a MESH's vertices (an old "
                         "particle-system hair: convert it to a Curves object first). %s" % (obj.name, obj.type, tail))
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mw = obj.matrix_world
    strands = None
    if obj.type == "MESH":
        me = ev.to_mesh()
        try:
            pts = [mw @ v.co for v in me.vertices]
        finally:
            ev.to_mesh_clear()
    else:
        data = ev.data
        attr = data.attributes.get("position")
        n = len(attr.data) if attr is not None else 0
        if n > _MAX_POINTS:
            raise MifOpError("'%s' has %d points and one call reads at most %d. %s" % (obj.name, n, _MAX_POINTS, tail))
        flat = [0.0] * (3 * n)
        if n:
            attr.data.foreach_get("vector", flat)
        pts = [mw @ Vector(flat[3 * i:3 * i + 3]) for i in range(n)]
        if obj.type == "CURVES":
            strands = [(c.first_point_index, c.points_length) for c in data.curves]
    if not pts:
        raise MifOpError("'%s' has no points - a measurement of nothing is not clean. %s" % (obj.name, tail))
    if len(pts) > _MAX_POINTS:
        raise MifOpError("'%s' has %d points and one call reads at most %d. %s" % (obj.name, len(pts), _MAX_POINTS, tail))
    return pts, strands, obj


def _vertex_normals(polys, pos):
    """World vertex normals (the normalized sum of the adjacent faces' Newell normals) from world positions."""
    acc = [Vector((0.0, 0.0, 0.0)) for _ in pos]
    for vs, fn in zip(polys, _face_normals(polys, pos)):
        for v in vs:
            acc[v] += fn
    return [a.normalized() if a.length > 1e-12 else a for a in acc]


def _wall_thickness(polys, pos, idx, wall):
    """{vertex: distance} from each vertex in idx along its inward normal to the next surface of the same mesh, within
    wall - a lined garment's wall thickness there. A single layer has none."""
    from mathutils.bvhtree import BVHTree
    tree = BVHTree.FromPolygons(pos, polys)
    nrm = _vertex_normals(polys, pos)
    out = {}
    for i in idx:
        n = nrm[i]
        if n.length < 0.5:
            continue
        loc, _n, _f, d = tree.ray_cast(pos[i] - n * _HIT_STEP, -n, wall)
        if loc is not None:
            out[i] = d + _HIT_STEP
    return out


_ENV_KEYS = ("object", "envelope", "center", "margin", "coneAngle", "mask", "protect", "maxPush", "wall", "dryRun",
             "previewId", "previewTtlSeconds")


def _env_params(params, tail):
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    env = get_object(take(params, "envelope", required=True, kind=str))
    if env is obj:
        raise MifOpError("object and envelope are the same object - a garment cannot clear itself. %s" % tail)
    if env.type not in _POINT_SOURCES:
        raise MifOpError("envelope '%s' is a %s - an envelope is a MESH (a head, a body, a hull round the hair) or the "
                         "hair itself as a CURVES object or POINTCLOUD. %s" % (env.name, env.type, tail))
    cfg = {"margin": take_float(params, "margin", default=0.005), "cone": take_float(params, "coneAngle", default=10.0),
           "cap": take_float(params, "maxPush", default=0.1), "wall": take_float(params, "wall", default=0.03),
           "dry": take_bool(params, "dryRun", default=False)}
    if not (math.isfinite(cfg["margin"]) and cfg["margin"] >= 0):
        raise MifOpError("margin must be 0 or more metres, got %r. %s" % (cfg["margin"], tail))
    if not (math.isfinite(cfg["cone"]) and 0 < cfg["cone"] <= 90):
        raise MifOpError("coneAngle must be above 0 and at most 90 degrees, got %r. %s" % (cfg["cone"], tail))
    if not (math.isfinite(cfg["cap"]) and cfg["cap"] > 0):
        raise MifOpError("maxPush must be above 0 metres, got %r. %s" % (cfg["cap"], tail))
    if not (math.isfinite(cfg["wall"]) and cfg["wall"] > 0):
        raise MifOpError("wall must be above 0 metres, got %r. %s" % (cfg["wall"], tail))
    cfg["mask"] = _mask_shapes(params["mask"]) if params.get("mask") is not None else None
    cfg["protect"] = _mask_shapes(params["protect"]) if params.get("protect") is not None else None
    raw = params.get("center")
    if raw is not None:
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise MifOpError("center is [x, y, z] (world), got %r. %s" % (raw, tail))
        cfg["center"] = Vector(finite_floats(list(raw), "center"))
    else:
        corners = [env.matrix_world @ Vector(c) for c in env.bound_box]
        cfg["center"] = sum(corners, Vector((0.0, 0.0, 0.0))) / 8.0
    if obj.data.shape_keys is not None:
        raise MifOpError("'%s' has shape keys - moving its rest vertices would shift every key against the basis. Apply "
                         "or remove them first. %s" % (obj.name, tail))
    if not len(obj.data.polygons):
        raise MifOpError("'%s' has no faces to push. %s" % (obj.name, tail))
    return obj, env, cfg


def _env_radius(env, cfg, dirs, rads, idx, tail):
    """{vertex: the envelope's radius from center along the vertex's direction}: for a MESH the farthest crossing within
    the vertex's own radius + maxPush + margin; for points the farthest point within coneAngle of the direction."""
    c = cfg["center"]
    out = {}
    if env.type == "MESH":
        tree = _world_tree(env, tail)
        for i in idx:
            hits = _hits_along(tree, c, dirs[i], rads[i] + cfg["cap"] + cfg["margin"])
            if hits:
                out[i] = hits[-1][0]
        return out
    from mathutils.kdtree import KDTree
    pts, _strands, _o = _source_points(env.name, tail)
    keep = [(p - c) for p in pts if (p - c).length > 1e-9]
    kd = KDTree(len(keep))
    for j, v in enumerate(keep):
        kd.insert(v.normalized(), j)
    kd.balance()
    radii = [v.length for v in keep]
    chord = 2.0 * math.sin(math.radians(cfg["cone"]) / 2.0)
    for i in idx:
        found = kd.find_range(dirs[i], chord)
        if found:
            out[i] = max(radii[j] for _co, j, _d in found)
    return out


def op_clear_envelope(params):
    """Push a garment (shell and lining together) out from a centre until it clears an envelope - a head, a body, a hull
    round the hair, or the hair itself - by a margin, keeping its wall thickness, faded by a mask and kept off a
    protected region. WRITES the garment's rest mesh.

    THE GAP (spec 2026-10-08): a hood simulated on a bald head sits at the scalp, so any hair goes through it. The fix
    moved every hood vertex out from the head centre to clear the hair envelope by a margin, lining and shell together,
    faded toward the neck, the face opening untouched - raw Python.

    HOW: every vertex v of object (rest mesh, world) has a direction u from center (default the envelope's bounding-box
    centre) and a radius r. The envelope's radius R along u is the farthest crossing of the envelope MESH within
    r + maxPush + margin, or for a CURVES / POINTCLOUD envelope the farthest point within coneAngle (default 10 degrees)
    of u. need = R + margin - r (0 when it clears). THE WALL IS KEPT by moving shell and lining alike: each vertex takes
    the largest need of every vertex whose direction is within coneAngle of its own, weighted by (1 - (d/c)^2)^2 (d the
    chord between the directions, c the cone's) so the move is continuous, and the two layers at one place move the
    same distance along the same ray. The move is that, times mask (default 1 everywhere) times 1 - protect (each
    one shape or a list of {box | sphere | band {zFrom, zTo}, falloff} or {vertexGroup}, as blend_weights takes them),
    capped at maxPush (default 0.1 m). wall (default 0.03 m) bounds the thickness read: from each vertex along its inward
    normal to the next surface of the same mesh.

    Reply: center, verticesConsidered (mask x (1 - protect) above 0), withEnvelope / noEnvelope, verticesMoved, move,
    capped, clearanceBefore / clearanceAfter (r - R, metres; the min of the after is the clearance achieved), insideBefore
    / insideAfter, shortOfMargin {total, faded (weight under 1), capped}, wall {samples, before, after, maxChange}. dryRun
    writes nothing. Refused, nothing changed: object = envelope, an envelope that is not a mesh, curves or point cloud,
    margin below 0, coneAngle outside 0-90, maxPush or wall not above 0, a mask or protect holding no vertex, shape keys.
    """
    tail = "NOTHING was changed."
    reject_unknown(params, _ENV_KEYS, "clear_envelope")
    obj, env, cfg = _env_params(params, tail)
    if not cfg["dry"]:
        require_editable(obj, "push out")
    me = obj.data
    mw = obj.matrix_world
    world = [mw @ v.co for v in me.vertices]
    weight, protected = [], 0
    for i, p in enumerate(world):
        w = max(fn(p, i, obj) for fn in cfg["mask"]) if cfg["mask"] else 1.0
        if cfg["protect"]:
            pv = min(1.0, max(fn(p, i, obj) for fn in cfg["protect"]))
            protected += pv > 0.0
            w *= 1.0 - pv
        weight.append(w)
    if cfg["protect"] and not protected:
        raise MifOpError("protect holds no vertex of '%s' - a protection of nothing is a mistyped region. %s"
                         % (obj.name, tail))
    idx = [i for i, w in enumerate(weight) if w > 0.0]
    if not idx:
        raise MifOpError("no vertex of '%s' is under the mask and outside protect - a push of nothing is not clean. %s"
                         % (obj.name, tail))
    c = cfg["center"]
    dirs, rads = [None] * len(world), [0.0] * len(world)
    for i in idx:
        v = world[i] - c
        rads[i] = v.length
        dirs[i] = v.normalized() if v.length > 1e-9 else None
    idx = [i for i in idx if dirs[i] is not None]
    env_r = _env_radius(env, cfg, dirs, rads, idx, tail)
    need = {i: max(0.0, env_r[i] + cfg["margin"] - rads[i]) for i in env_r}
    from mathutils.kdtree import KDTree
    kd = KDTree(len(idx))
    for i in idx:
        kd.insert(dirs[i], i)
    kd.balance()
    chord = 2.0 * math.sin(math.radians(cfg["cone"]) / 2.0)
    move, capped = {}, set()
    for i in idx:
        # A SMOOTH WEIGHT, NOT A HARD CONE. A plain max over the cone jumps where a neighbour crosses its edge, and the
        # shell and lining at one place sit a hair apart in direction: on a dome whose rings were exactly one cone
        # apart, the lining took a neighbour the shell missed and the wall changed by 0.26 mm. Weighted by
        # (1 - (d/chord)^2)^2 the move is continuous in direction, so the two layers move alike.
        shared = max((need.get(j, 0.0) * _falloff(d, chord) for _co, j, d in kd.find_range(dirs[i], chord)),
                     default=0.0)
        m = shared * weight[i]
        if m > cfg["cap"]:
            m = cfg["cap"]
            capped.add(i)
        if m > 0.0:
            move[i] = m
    new = list(world)
    for i, m in move.items():
        new[i] = world[i] + dirs[i] * m
    before = {i: rads[i] - env_r[i] for i in env_r}
    after = {i: rads[i] + move.get(i, 0.0) - env_r[i] for i in env_r}
    short = [i for i in env_r if after[i] < cfg["margin"] - 1e-7]
    polys = [list(p.vertices) for p in me.polygons]
    t0 = _wall_thickness(polys, world, idx, cfg["wall"])
    t1 = _wall_thickness(polys, new, idx, cfg["wall"]) if move else t0
    both = [i for i in t0 if i in t1]
    out = {"object": obj.name, "envelope": env.name, "envelopeType": env.type, "center": rnd(list(c)),
           "margin": cfg["margin"], "coneAngle": cfg["cone"], "maxPush": cfg["cap"], "protected": protected,
           "verticesConsidered": len(idx), "withEnvelope": len(env_r), "noEnvelope": len(idx) - len(env_r),
           "verticesMoved": len(move), "move": stats(list(move.values()), [95], 6), "capped": len(capped),
           "clearanceBefore": stats(list(before.values()), [5], 6),
           "clearanceAfter": stats(list(after.values()), [5], 6),
           "insideBefore": sum(1 for g in before.values() if g < 0), "insideAfter": sum(1 for g in after.values() if g < 0),
           "shortOfMargin": {"total": len(short), "faded": sum(1 for i in short if weight[i] < 1.0),
                             "capped": sum(1 for i in short if i in capped)},
           "wall": {"samples": len(both), "before": stats([t0[i] for i in both], [], 6),
                    "after": stats([t1[i] for i in both], [], 6),
                    "maxChange": round(max((abs(t1[i] - t0[i]) for i in both), default=0.0), 6)},
           "dryRun": cfg["dry"]}
    if not env_r:
        out["note"] = ("no considered vertex has the envelope along its direction from center - check center (default "
                       "the envelope's bounding-box centre) and the envelope.")
    elif not move:
        out["note"] = "every considered vertex already clears the envelope by margin - nothing moved."
    out.update(shared_data_note(obj))
    if cfg["dry"] or not move:
        return out
    mwi = mw.inverted()
    for i in move:
        me.vertices[i].co = mwi @ new[i]
    me.update()
    return out


_PIN_KEYS = ("object", "points", "source", "center", "regions", "wall", "limit")
_WINDING_DIRS = ((0.0123, 0.0217, 1.0), (1.0, 0.0311, 0.0173), (0.0257, 1.0, 0.0419))


def _pin_points(params, tail):
    raw, src = params.get("points"), params.get("source")
    if (raw is None) == (src is None):
        raise MifOpError("pass points [[x,y,z], ...] OR source (a hair Curves object, a point cloud or a mesh), one of "
                         "them. %s" % tail)
    if src is not None:
        if not isinstance(src, str) or not src:
            raise MifOpError("source is an object name, got %r. %s" % (src, tail))
        return _source_points(src, tail)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise MifOpError("points is a non-empty list of [x, y, z] (world), got %r. %s" % (type(raw).__name__, tail))
    if len(raw) > _MAX_POINTS:
        raise MifOpError("points holds %d and one call takes at most %d. %s" % (len(raw), _MAX_POINTS, tail))
    pts = []
    for k, p in enumerate(raw):
        if not isinstance(p, (list, tuple)) or len(p) != 3:
            raise MifOpError("points[%d] must be [x, y, z], got %r. %s" % (k, p, tail))
        pts.append(Vector(finite_floats(list(p), "points[%d]" % k)))
    return pts, None, None


def _pin_regions(raw, src_obj, tail):
    if raw is None:
        return []
    if not isinstance(raw, dict) or not raw or len(raw) > 20:
        raise MifOpError("regions is {name: region} with 1-20 entries, each {box} | {sphere} | {vertexGroup} as "
                         "mesh_distance's region, got %r. %s" % (raw, tail))
    out = []
    for name, spec in raw.items():
        test, desc = parse_region({"region": spec}, tail=tail)
        if "vertexGroup" in desc and (src_obj is None or src_obj.type != "MESH"):
            raise MifOpError("region '%s' is a vertex group, which only a MESH source has. %s" % (name, tail))
        out.append((str(name), test, desc))
    return out


def op_points_inside(params):
    """Which points - hair strand points, a point cloud, a mesh's vertices - lie outside a garment, inside it, or in its
    wall. Read-only.

    THE GAP (spec 2026-10-08): measuring hair through a hood needed raw Python for inside / outside per strand point,
    and all-hits ray casts (ray_cast now takes allHits for those).

    TWO MODES. With center [x, y, z] (RADIAL - a hood round a head): each point casts away from center against object
    (as it evaluates now). No hit: outside (it pokes through, or nothing covers it). The first face met from its front:
    inside (under the garment). Met from its back - the point is behind a face - and a face of object within wall
    (default 0.03 m) back toward center, also met from its back: inWall (between a lined garment's shell and lining);
    otherwise inside (a one-layer garment's back). Without center (WINDING - a closed mesh): three skew rays count every
    crossing, odd is inside, the majority decides.

    Points: points [[x,y,z], ...] or source, a hair CURVES object, a POINTCLOUD or a MESH (its vertices), at most
    500,000. regions {name: {box} | {sphere} | {vertexGroup (mesh source)}} counts each. A Curves source also counts
    strands with any point outside (and in the wall). Reply: mode, points, counts {outside, inside, inWall, atCenter},
    regions {name: {points, ...counts}}, outsidePoints (first limit, default 200), strands {count, withOutside,
    withInWall, withOutsideIndices}. Refused: neither or both of points and source, an empty or malformed list, a
    source that is not curves, points or a mesh, object not a mesh or with no faces, a region holding no point.
    """
    tail = "NOTHING was read."
    reject_unknown(params, _PIN_KEYS, "points_inside")
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    pts, strands, src = _pin_points(params, tail)
    if src is obj:
        raise MifOpError("source and object are the same mesh - its vertices lie on its own surface. %s" % tail)
    wall = take_float(params, "wall", default=0.03)
    limit = take_int(params, "limit", default=200)
    if not (math.isfinite(wall) and wall > 0):
        raise MifOpError("wall must be above 0 metres, got %r. %s" % (wall, tail))
    if not 0 <= limit <= 100000:
        raise MifOpError("limit must be 0-100000, got %d. %s" % (limit, tail))
    raw_c = params.get("center")
    center = None
    if raw_c is not None:
        if not isinstance(raw_c, (list, tuple)) or len(raw_c) != 3:
            raise MifOpError("center is [x, y, z] (world), got %r. %s" % (raw_c, tail))
        center = Vector(finite_floats(list(raw_c), "center"))
    regions = _pin_regions(params.get("regions"), src, tail)
    tree = _world_tree(obj, tail)
    far = 1.0e6
    cls = []
    if center is None:
        dirs = [Vector(d).normalized() for d in _WINDING_DIRS]
        for p in pts:
            odd = sum(1 for d in dirs if len(_hits_along(tree, p, d, far, cap=4096)) % 2)
            cls.append("inside" if odd >= 2 else "outside")
    else:
        for p in pts:
            v = p - center
            if v.length < 1e-9:
                cls.append("atCenter")
                continue
            u = v.normalized()
            loc, nrm, _fi, _d = tree.ray_cast(p, u, far)
            if loc is None:
                cls.append("outside")
            elif nrm.dot(u) < 0.0:
                cls.append("inside")
            else:
                # BACK TOWARD THE CENTRE, a point in the wall leaves the material through the lining's back: the lining
                # faces the cavity, so its normal runs along -u.
                loc2, nrm2, _f2, _d2 = tree.ray_cast(p, -u, wall)
                cls.append("inWall" if loc2 is not None and nrm2.dot(-u) > 0.0 else "inside")
    kinds = ("outside", "inside", "inWall", "atCenter") if center is not None else ("outside", "inside")

    def counts(ix):
        row = {k: 0 for k in kinds}
        for i in ix:
            row[cls[i]] += 1
        return row
    out = {"object": obj.name, "mode": "RADIAL" if center is not None else "WINDING", "points": len(pts),
           "source": src.name if src is not None else None, "counts": counts(range(len(pts))),
           "outsidePoints": [i for i, k in enumerate(cls) if k == "outside"][:limit]}
    if center is not None:
        out["center"] = rnd(list(center))
        out["wall"] = wall
    if regions:
        rows = {}
        for name, test, desc in regions:
            ix = [i for i, p in enumerate(pts) if test(p, i, src)]
            if not ix:
                raise MifOpError("region '%s' %s holds no point - a measurement of nothing is not clean. %s"
                                 % (name, desc, tail))
            row = {"points": len(ix), "region": desc}
            row.update(counts(ix))
            rows[name] = row
        out["regions"] = rows
    if strands is not None:
        with_out, with_wall = [], 0
        for k, (first, n) in enumerate(strands):
            ks = cls[first:first + n]
            if "outside" in ks:
                with_out.append(k)
            if "inWall" in ks:
                with_wall += 1
        out["strands"] = {"count": len(strands), "withOutside": len(with_out), "withInWall": with_wall,
                          "withOutsideIndices": with_out[:limit]}
    return out


_FIT_KEYS = ("object", "target", "mode", "iterations", "samples", "reject", "init", "landmarks", "region", "carry",
             "tolerance", "dryRun", "previewId", "previewTtlSeconds")
_FIT_MODES = ("RIGID", "UNIFORM")
_FIT_INITS = ("CENTROID", "NONE")


def _umeyama(src, dst, with_scale):
    """4x4 numpy matrix T minimising sum |T src - dst|^2 over rotation, translation and (with_scale) one uniform scale
    (Umeyama 1991; reflections excluded)."""
    import numpy as np
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, dg, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1.0
    rot = u @ s @ vt
    var = (xs ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(dg) @ s) / var) if with_scale and var > 0 else 1.0
    t = np.eye(4)
    t[:3, :3] = scale * rot
    t[:3, 3] = mu_d - scale * rot @ mu_s
    return t


def _fit_dist(tree, pts, normals=False):
    """(distances, nearest points[, their face normals]) of each point to a world BVH."""
    import numpy as np
    out = np.empty(len(pts))
    near = np.empty((len(pts), 3))
    nrm = np.empty((len(pts), 3))
    for k, p in enumerate(pts):
        loc, n, _i, d = tree.find_nearest(Vector(p))
        out[k] = d
        near[k] = loc
        nrm[k] = n
    return (out, near, nrm) if normals else (out, near)


def _plane_step(x, q, n, with_scale):
    """4x4 step minimising sum ((T x - q) . n)^2 - point to plane, linearised: a small rotation w and scale s about the
    points' centroid, and a translation t (Chen and Medioni 1991). Converges far faster than point to point near the fit,
    where point to point crawls (measured: 50 point-to-point passes left a moved monkey 1.04e-4 off)."""
    import numpy as np
    c = x.mean(0)
    xc = x - c
    cols = [np.cross(xc, n), n]
    if with_scale:
        cols.append((xc * n).sum(1)[:, None])
    a = np.hstack(cols)
    b = -((x - q) * n).sum(1)
    sol = np.linalg.lstsq(a, b, rcond=None)[0]
    w, t = sol[:3], sol[3:6]
    s = 1.0 + (sol[6] if with_scale else 0.0)
    th = float(np.linalg.norm(w))
    rot = np.eye(3)
    if th > 1e-15:
        k = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / th
        rot = np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * (k @ k)
    m = np.eye(4)
    m[:3, :3] = s * rot
    m[:3, 3] = c + t - s * rot @ c
    return m


def _fit_err(d):
    import numpy as np
    return {"mean": round(float(d.mean()), 6), "rms": round(float(np.sqrt((d ** 2).mean())), 6),
            "median": round(float(np.median(d)), 6), "p95": round(float(np.percentile(d, 95)), 6),
            "max": round(float(d.max()), 6), "count": int(len(d))}


def _fit_params(params, tail):
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    tgt = get_object(take(params, "target", required=True, kind=str), want_mesh=True)
    if obj is tgt:
        raise MifOpError("object and target are the same mesh - it already fits itself. %s" % tail)
    cfg = {"mode": str(take(params, "mode", default="RIGID", kind=str)).upper(),
           "init": str(take(params, "init", default="CENTROID", kind=str)).upper(),
           "iters": take_int(params, "iterations", default=50), "samples": take_int(params, "samples", default=4000),
           "reject": take_float(params, "reject", default=0.1), "tol": take_float(params, "tolerance", default=1e-7),
           "dry": take_bool(params, "dryRun", default=False)}
    if cfg["mode"] not in _FIT_MODES:
        raise MifOpError("mode is RIGID (rotation and translation) or UNIFORM (and one scale), got %r. %s"
                         % (cfg["mode"], tail))
    if cfg["init"] not in _FIT_INITS:
        raise MifOpError("init is CENTROID or NONE (landmarks replace it), got %r. %s" % (cfg["init"], tail))
    if not 1 <= cfg["iters"] <= 500:
        raise MifOpError("iterations must be 1-500, got %d. %s" % (cfg["iters"], tail))
    if not 10 <= cfg["samples"] <= 200000:
        raise MifOpError("samples must be 10-200000, got %d. %s" % (cfg["samples"], tail))
    if not (math.isfinite(cfg["reject"]) and 0 <= cfg["reject"] < 0.9):
        raise MifOpError("reject (the share of worst matches left out each pass) must be 0 to under 0.9, got %r. %s"
                         % (cfg["reject"], tail))
    if not (math.isfinite(cfg["tol"]) and cfg["tol"] >= 0):
        raise MifOpError("tolerance must be 0 or more metres, got %r. %s" % (cfg["tol"], tail))
    lms = params.get("landmarks")
    cfg["landmarks"] = None
    if lms is not None:
        if not isinstance(lms, (list, tuple)) or len(lms) < 3 or len(lms) > 1000:
            raise MifOpError("landmarks is a list of 3-1000 {from: [x,y,z], to: [x,y,z]} (world; a point on object and "
                             "where it belongs on target), got %r. %s" % (lms, tail))
        pairs = []
        for k, lm in enumerate(lms):
            if not isinstance(lm, dict) or set(lm) != {"from", "to"}:
                raise MifOpError("landmarks[%d] is {from, to}, got %r. %s" % (k, lm, tail))
            a, b = lm["from"], lm["to"]
            if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == 3 and len(b) == 3):
                raise MifOpError("landmarks[%d] from and to are [x, y, z], got %r. %s" % (k, lm, tail))
            pairs.append((finite_floats(list(a), "landmarks.from"), finite_floats(list(b), "landmarks.to")))
        cfg["landmarks"] = pairs
    carry = params.get("carry")
    cfg["carry"] = []
    if carry is not None:
        if not isinstance(carry, (list, tuple)) or len(carry) > 100:
            raise MifOpError("carry is a list of up to 100 object names moved with object, got %r. %s" % (carry, tail))
        for nm in carry:
            o = get_object(nm)
            if o is obj or o is tgt:
                raise MifOpError("carry names '%s', which is object or target. %s" % (o.name, tail))
            cfg["carry"].append(o)
    return obj, tgt, cfg


def op_fit_mesh(params, _plan=None):
    """Fit one mesh onto another - rigid, or with one uniform scale - by iterative closest points, and report the fit
    error. WRITES object's transform (and carry's) unless dryRun.

    THE GAP (spec 2026-10-08): placing a groom on a different head needed an ICP-style fit in raw Python: fit the old
    head onto the new one, then move the groom by the same transform.

    HOW: object's vertices (as it evaluates now, world, within region when given) are sampled evenly to samples (default
    4000). The start is landmarks [{from, to}] (3 or more, the best transform taking each from to its to), else init
    CENTROID (default; centroid onto centroid, and for UNIFORM the spread's ratio) or NONE. Each of iterations (default
    50) passes: each sample's nearest point on target, the worst reject share (default 0.1) left out, the best rigid
    (mode RIGID, default) or rigid-and-scale (UNIFORM) transform for the rest composed on - point to point (Umeyama, no
    reflection) while a pass still gains 20% or more (at most 10 passes), then point to plane (the distance along target's normal there); it
    stops when the RMS changes by under tolerance (default 1e-7 m).

    Reply: transform (4x4 world, applied as transform @ matrix_world), translation, rotation {quaternion, eulerDegrees},
    scale, iterations, converged, before / after {mean, rms, median, p95, max, count} - every vertex of object (in
    region) to target's surface - and targetToObject {mean, max} after, the other way round; carried [names] (a child of
    object follows its parent and is reported in followsParent). Refused, nothing moved: object = target, a target with
    no faces, a mode or init it does not know, iterations outside 1-500, samples outside 10-200000, reject outside 0 to
    0.9, fewer than 3 landmarks or a malformed one, a carry naming object or target, a region holding no vertex.
    """
    import numpy as np
    from mathutils import Matrix
    tail = "NOTHING was moved."
    reject_unknown(params, _FIT_KEYS, "fit_mesh")
    obj, tgt, cfg = _fit_params(params, tail)
    region, region_desc = parse_region(params, tail=tail)
    if not cfg["dry"]:
        require_editable(obj, "move")
        for o in cfg["carry"]:
            require_editable(o, "move")
    tree = _world_tree(tgt, tail)
    obm = evaluated_world(obj, True)
    tbm = evaluated_world(tgt, True)
    try:
        allp = np.array([list(v.co) for v in obm.verts if region is None or region(v.co, v.index, obj)], dtype=float)
        tpts = np.array([list(v.co) for v in tbm.verts], dtype=float)
    finally:
        obm.free()
        tbm.free()
    if not len(allp):
        raise MifOpError("no vertex of '%s' is in the region %s - a fit of nothing is not clean. %s"
                         % (obj.name, region_desc, tail))
    take_n = min(cfg["samples"], len(allp))
    smp = allp[np.linspace(0, len(allp) - 1, take_n).round().astype(int)]
    if cfg["landmarks"]:
        a = np.array([p[0] for p in cfg["landmarks"]])
        b = np.array([p[1] for p in cfg["landmarks"]])
        t = _umeyama(a, b, cfg["mode"] == "UNIFORM")
        start = "landmarks"
    elif cfg["init"] == "CENTROID":
        mp, mq = allp.mean(0), tpts.mean(0)
        s = 1.0
        if cfg["mode"] == "UNIFORM":
            vp = ((allp - mp) ** 2).sum(1).mean()
            vq = ((tpts - mq) ** 2).sum(1).mean()
            s = math.sqrt(vq / vp) if vp > 0 else 1.0
        t = np.eye(4)
        t[:3, :3] *= s
        t[:3, 3] = mq - s * mp
        start = "centroid"
    else:
        t = np.eye(4)
        start = "none"

    def moved(pts, m):
        return pts @ m[:3, :3].T + m[:3, 3]
    before, _n = _fit_dist(tree, allp)
    # POINT TO POINT WHILE FAR, POINT TO PLANE ONCE CLOSE: point to point is robust from a rough start and crawls near
    # the fit; the plane step converges fast near it. The switch comes when a pass improves the RMS by under 20%, or
    # after 10 passes: point to point gains a near-constant share each pass (5-10% on a moved monkey), so a 2% rule
    # never switched in 50 passes.
    prev, used, converged, plane = None, 0, False, False
    scaled = cfg["mode"] == "UNIFORM"
    for _k in range(cfg["iters"]):
        used += 1
        x = moved(smp, t)
        d, near, nrm = _fit_dist(tree, x, normals=True)
        keep = d <= np.quantile(d, 1.0 - cfg["reject"]) if cfg["reject"] > 0 else np.ones(len(d), bool)
        if keep.sum() < 7:
            keep = np.ones(len(d), bool)
        rms = float(np.sqrt((d[keep] ** 2).mean()))
        if prev is not None and abs(prev - rms) <= cfg["tol"]:
            converged = True
            break
        if prev is not None and not plane and (prev - rms < 0.2 * prev or used > 10):
            plane = True
        step = (_plane_step(x[keep], near[keep], nrm[keep], scaled) if plane
                else _umeyama(x[keep], near[keep], scaled))
        t = step @ t
        prev = rms
    after, _n = _fit_dist(tree, moved(allp, t))
    mat = Matrix([list(r) for r in t])
    lin = mat.to_3x3()
    scale = lin.median_scale if hasattr(lin, "median_scale") else (abs(lin.determinant()) ** (1.0 / 3.0))
    rot = (lin * (1.0 / scale)).to_quaternion() if scale > 0 else lin.to_quaternion()
    from mathutils.bvhtree import BVHTree
    rbm = evaluated_world(obj, True)
    try:
        rbm.transform(mat)
        rtree = BVHTree.FromBMesh(rbm)
    finally:
        rbm.free()
    tk = tpts[np.linspace(0, len(tpts) - 1, min(cfg["samples"], len(tpts))).round().astype(int)]
    back, _n = _fit_dist(rtree, tk)
    out = {"object": obj.name, "target": tgt.name, "mode": cfg["mode"], "start": start, "region": region_desc,
           "transform": [[round(float(x), 8) for x in row] for row in t],
           "translation": [round(float(x), 6) for x in t[:3, 3]],
           "rotation": {"quaternion": [round(x, 6) for x in rot],
                        "eulerDegrees": [round(math.degrees(x), 4) for x in rot.to_euler()]},
           "scale": round(float(scale), 6), "iterations": used, "converged": converged, "samples": int(take_n),
           "reject": cfg["reject"], "before": _fit_err(before), "after": _fit_err(after),
           "targetToObject": {"mean": round(float(back.mean()), 6), "max": round(float(back.max()), 6)},
           "dryRun": cfg["dry"]}
    if _plan is not None:
        # THE BULK PREVIEW'S PLAN (_plan_fit_mesh): the transform itself, from this same fit.
        _plan.update(matrix=mat, carry=[o.name for o in cfg["carry"]])
    if cfg["dry"]:
        return out
    carried, follows = [], []
    for o in cfg["carry"]:
        p = o.parent
        while p is not None and p is not obj:
            p = p.parent
        if p is obj:
            follows.append(o.name)
        else:
            carried.append(o)
    obj.matrix_world = mat @ obj.matrix_world
    for o in carried:
        o.matrix_world = mat @ o.matrix_world
    bpy.context.view_layer.update()
    out["carried"] = [o.name for o in carried]
    out["followsParent"] = follows
    return out


# --------------------------------------------------------------------------------------------- bones by role

# A BONE'S ROLE READ OFF ITS NAME, its side off where it stands. The names cover the common skeletons (Unreal's
# mannequin and MetaHuman, Rigify's DEF bones, Mixamo, Blender's own samples); the side is the sign of the head's
# armature-space x, so a rig named in any language of left and right still splits. Order matters: a forearm also
# matches the arm pattern and an upper leg the leg pattern, so the narrower role is tried first.
_ROLE_ORDER = (
    ("lowerarm", r"lower_?arm|fore_?arm"),
    ("upperarm", r"upper_?arm|(?:^|[^a-z])arm(?:[^a-z]|$)|(?:left|right)arm$"),
    ("clavicle", r"clavicle|shoulder"),
    ("hand", r"hand"),
    ("thigh", r"thigh|up_?leg|upper_?leg"),
    ("calf", r"calf|shin|lower_?leg|(?:left|right)leg$|(?:^|[^a-z])leg(?:[^a-z]|$)"),
    ("foot", r"foot"),
    ("pelvis", r"pelvis|hips?(?:[^a-z]|$)"),
    ("spine", r"spine|chest"),
)
_ROLE_SKIP = r"twist|(?:^|[^a-z])ik(?:[^a-z]|$)|pole|ctrl|mch|org[-_.]|target|roll|tweak|socket|(?:^|[^a-z])end|helper|corrective|bend|_fk|fk_|vis_"


def role_bones(arm):
    """{(role, side): bone name} on an armature - side 'l' (+x), 'r' (-x) or 'c'. Of several candidates the one without
    constraints, deforming, with the shortest name wins (a Rigify DEF chain's first segment, not '.001')."""
    import re   # noqa: PLC0415
    skip = re.compile(_ROLE_SKIP)
    found = {}
    for b in arm.data.bones:
        low = b.name.lower()
        if skip.search(low):
            continue
        role = next((r for r, pat in _ROLE_ORDER if re.search(pat, low)), None)
        if role is None:
            continue
        x = b.head_local.x
        side = "l" if x > 1e-4 else "r" if x < -1e-4 else "c"
        pb = arm.pose.bones.get(b.name)
        rank = (bool(pb is not None and len(pb.constraints)), not b.use_deform, len(b.name), b.name)
        cur = found.get((role, side))
        if cur is None or rank < cur[0]:
            found[(role, side)] = (rank, b.name)
    return {k: v[1] for k, v in found.items()}


def nearest_bone_fn(arm):
    """fn(world point) -> the deform bone (rest pose) whose segment is nearest, for naming a place on a body."""
    from mathutils.geometry import intersect_point_line   # noqa: PLC0415
    mw = arm.matrix_world
    segs = [(b.name, mw @ b.head_local, mw @ b.tail_local) for b in arm.data.bones if b.use_deform] or \
           [(b.name, mw @ b.head_local, mw @ b.tail_local) for b in arm.data.bones]

    def near(p):
        best, name = None, None
        for nm, h, t in segs:
            q, f = intersect_point_line(p, h, t)
            q = h if f <= 0 else t if f >= 1 else q
            d = (p - q).length
            if best is None or d < best:
                best, name = d, nm
        return name
    return near


# --------------------------------------------------------------------------------------------- recipe_fit_outfit

_FO_KEYS = ("garment", "body", "sourceBody", "armature", "output", "method", "align", "ease", "landmarks", "falloff",
            "clearance", "smooth", "poses", "margin", "spread", "maxPush", "passes", "coverDistance", "edgeBand",
            "tolerance", "minPatch", "straps", "hideCovered", "keepOnRefusal", "keepPoses", "allowSimulation")
_FO_PRESETS = ("armsRaised", "stride", "crouch")
_FO_FRAME0 = 9001          # the scratch frames a recipe-keyed pose sits on, in a scratch action removed afterwards
_FO_GROUP = "MifFitCheck"  # the body's scratch group: the vertices the garment covers at rest, away from its openings
_FO_FOLD = 0.06            # a posed skin vertex whose normal meets the body again this near lies in a fold, unseen
_FO_POKE = 0.03            # skin this far through the garment at rest is still tested (and pushed), not left out


def _fo_preset(name, roles, arm, tail):
    """[(bone, ('rot', axis, degrees) | ('loc', Vector))] for a named test pose. Armature space: the character faces -y,
    z up, its left at +x - the convention Blender's front view and every skeleton named in _ROLE_ORDER use."""
    X, Y = Vector((1, 0, 0)), Vector((0, 1, 0))
    need = {"armsRaised": ("upperarm", "lowerarm"), "stride": ("thigh", "calf"),
            "crouch": ("thigh", "calf")}[name]
    missing = ["%s_%s" % (r, s) for r in need for s in "lr" if (r, s) not in roles]
    if missing:
        raise MifOpError("the %s preset turns the %s bones and '%s' has none named for %s (bones are read by name: "
                         "upperarm/upper_arm/arm, lowerarm/forearm, thigh/upleg, calf/shin/leg, each side by the head's "
                         "x). Pass that pose as {name, bones: {bone: {rotate: [{axis, degrees}], location}}} or "
                         "{name, frame}. %s" % (name, "/".join(need), arm.name, ", ".join(missing), tail))
    out = []

    def put(role, side, *entry):
        b = roles.get((role, side))
        if b is not None:
            out.append((b, entry))
    for side, s in (("l", 1.0), ("r", -1.0)):
        if name == "armsRaised":
            put("clavicle", side, "rot", Y, -14.0 * s)
            put("upperarm", side, "rot", Y, -100.0 * s)
            put("lowerarm", side, "rot", Y, -20.0 * s)
        elif name == "stride":
            fwd = side == "l"
            put("thigh", side, "rot", X, -29.0 if fwd else 20.0)
            put("calf", side, "rot", X, 10.0 if fwd else 32.0)
            put("foot", side, "rot", X, 9.0 if fwd else -14.0)
            put("upperarm", side, "rot", X, 23.0 if fwd else -23.0)
            put("lowerarm", side, "rot", X, -17.0 if fwd else -26.0)
        else:
            put("thigh", side, "rot", X, -60.0)
            put("calf", side, "rot", X, 112.0)
            put("foot", side, "rot", X, -52.0)
            put("upperarm", side, "rot", X, -31.0)
            put("lowerarm", side, "rot", X, -34.0)
    bones = arm.data.bones
    lt = bones[roles[("thigh", "l")]].length if name != "armsRaised" else 0.0
    lc = bones[roles[("calf", "l")]].length if name != "armsRaised" else 0.0
    if name == "crouch":
        spine = sorted((b for (r, s), b in roles.items() if r == "spine" and s == "c"),
                       key=lambda b: bones[b].head_local.z)
        if spine:
            out.append((spine[0], ("rot", X, 18.0)))
        if ("pelvis", "c") in roles:
            # THE FEET STAY PLANTED: a thigh at -60 and a shin at +52 net stand lt cos60 + lc cos52 tall and put the
            # foot lt sin60 forward and lc sin52 back - the pelvis drops and moves by the difference.
            a, b = math.radians(60.0), math.radians(52.0)
            drop = lt + lc - (lt * math.cos(a) + lc * math.cos(b))
            dy = -(-lt * math.sin(a) + lc * math.sin(b))
            out.insert(0, (roles[("pelvis", "c")], ("loc", Vector((0.0, dy, -drop)))))
    elif name == "stride" and ("pelvis", "c") in roles:
        out.insert(0, (roles[("pelvis", "c")], ("loc", Vector((0.0, 0.0, -0.03 * (lt + lc))))))
    return out


def _fo_custom_pose(raw, arm, tail):
    """{name, bones: {bone: {rotate: [{axis, degrees}], location: [x,y,z]}}} -> entries, armature space."""
    bones = raw.get("bones")
    if not isinstance(bones, dict) or not bones:
        raise MifOpError("pose '%s': bones is {bone: {rotate: [{axis: [x,y,z], degrees}], location: [x,y,z]}} in the "
                         "armature's space, got %r. %s" % (raw.get("name"), bones, tail))
    out = []
    for b, spec in bones.items():
        if arm.pose.bones.get(b) is None:
            raise MifOpError("pose '%s': '%s' has no bone '%s'. %s" % (raw.get("name"), arm.name, b, tail))
        if not isinstance(spec, dict) or not spec or set(spec) - {"rotate", "location"}:
            raise MifOpError("pose '%s': bones['%s'] is {rotate, location}, got %r. %s" % (raw.get("name"), b, spec, tail))
        if "location" in spec:
            loc = spec["location"]
            if not isinstance(loc, (list, tuple)) or len(loc) != 3:
                raise MifOpError("pose '%s': bones['%s'].location is [x, y, z]. %s" % (raw.get("name"), b, tail))
            out.append((b, ("loc", Vector(finite_floats(list(loc), "location")))))
        rot = spec.get("rotate") or []
        if isinstance(rot, dict):
            rot = [rot]
        for r in rot:
            if not isinstance(r, dict) or set(r) != {"axis", "degrees"} or not isinstance(r["axis"], (list, tuple)) \
                    or len(r["axis"]) != 3:
                raise MifOpError("pose '%s': bones['%s'].rotate is a list of {axis: [x,y,z], degrees}, got %r. %s"
                                 % (raw.get("name"), b, r, tail))
            ax = Vector(finite_floats(list(r["axis"]), "axis"))
            if ax.length < 1e-9:
                raise MifOpError("pose '%s': a rotate axis is zero. %s" % (raw.get("name"), tail))
            out.append((b, ("rot", ax.normalized(), finite_float(r["degrees"], "degrees"))))
    return out


def _fo_parse_poses(raw, arm, tail):
    """[(name, frame | None, entries | None)], refused unless every pose is keyed by the caller OR every one by the
    recipe - the recipe's scratch action replaces the armature's own while it runs."""
    if raw is None:
        raw = list(_FO_PRESETS)
    if not isinstance(raw, (list, tuple)) or not raw or len(raw) > 20:
        raise MifOpError("poses is a list of 1-20: a preset (%s), {name, frame} keyed by you, or {name, bones}. %s"
                         % (", ".join(_FO_PRESETS), tail))
    roles = role_bones(arm)
    out, names = [], set()
    for p in raw:
        if isinstance(p, str):
            if p not in _FO_PRESETS:
                raise MifOpError("'%s' is not a preset pose; the presets are %s. %s" % (p, ", ".join(_FO_PRESETS), tail))
            row = (p, None, _fo_preset(p, roles, arm, tail))
        elif isinstance(p, dict) and isinstance(p.get("name"), str) and p.get("name"):
            if "frame" in p and set(p) == {"name", "frame"}:
                row = (p["name"], finite_int(p["frame"], "frame"), None)
            elif "bones" in p and set(p) == {"name", "bones"}:
                row = (p["name"], None, _fo_custom_pose(p, arm, tail))
            else:
                raise MifOpError("a pose is {name, frame} or {name, bones}, got %r. %s" % (p, tail))
        else:
            raise MifOpError("a pose is a preset name or {name, frame} / {name, bones}, got %r. %s" % (p, tail))
        if row[0] in names:
            raise MifOpError("two poses are named '%s'. %s" % (row[0], tail))
        names.add(row[0])
        out.append(row)
    keyed = [r for r in out if r[1] is not None]
    if keyed and len(keyed) != len(out):
        raise MifOpError("poses mixes frames you keyed (%s) with poses the recipe keys - it keys its own on a scratch "
                         "action that replaces the armature's while it runs, so your frames would read its action. Key "
                         "every pose yourself, or give every one as a preset or {name, bones}. %s"
                         % (", ".join(r[0] for r in keyed), tail))
    for name, _f, entries in out:
        for b, _e in entries or ():
            pb = arm.pose.bones[b]
            if len(pb.constraints):
                raise MifOpError("pose '%s' turns '%s', and its constraints (%s) override a pose - a Rigify DEF bone "
                                 "follows its controls. Key the poses on the controls yourself and pass {name, frame}. %s"
                                 % (name, b, ", ".join(c.type for c in pb.constraints), tail))
    return out


def _fo_apply_pose(arm, entries):
    """Every pose bone to rest, then each entry in hierarchy order: a turn about an armature-space axis through the
    bone's posed head, or an armature-space move - so a child turns from where its parent put it."""
    from mathutils import Matrix   # noqa: PLC0415
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    depth = {pb.name: len(pb.parent_recursive) for pb in arm.pose.bones}
    for b, e in sorted(entries, key=lambda be: depth[be[0]]):
        pb = arm.pose.bones[b]
        m = pb.matrix.copy()
        if e[0] == "rot":
            head = m.translation.copy()
            m = Matrix.Translation(head) @ Matrix.Rotation(math.radians(e[2]), 4, e[1]) @ Matrix.Translation(-head) @ m
        else:
            m.translation = m.translation + e[1]
        pb.matrix = m
        bpy.context.view_layer.update()


def _fo_key_all(arm, frame):
    for pb in arm.pose.bones:
        pb.keyframe_insert("location", frame=frame)
        pb.keyframe_insert({"QUATERNION": "rotation_quaternion", "AXIS_ANGLE": "rotation_axis_angle"}.get(
            pb.rotation_mode, "rotation_euler"), frame=frame)
        pb.keyframe_insert("scale", frame=frame)


class _FoRig:
    """The armature's state the recipe borrows - pose position, action (and slot), each bone's basis - put back by
    restore() whatever happened."""

    def __init__(self, arms):
        self.arms = [a for a in dict.fromkeys(arms) if a is not None]
        self.pos = {a.name: a.data.pose_position for a in self.arms}
        self.anim = {}
        self.made = []
        self.keep_as = None      # keepPoses: the scratch action kept under this name, unassigned, with a fake user
        self.kept = None

    def rest(self, on=True):
        for a in self.arms:
            a.data.pose_position = "REST" if on else "POSE"
        bpy.context.view_layer.update()

    def key_poses(self, arm, poses):
        """Key every recipe pose on a scratch action; returns [(name, frame)]."""
        ad = arm.animation_data
        self.anim[arm.name] = (ad is not None, ad.action if ad else None,
                               getattr(ad, "action_slot", None) if ad else None,
                               {pb.name: pb.matrix_basis.copy() for pb in arm.pose.bones})
        if ad is not None:
            ad.action = None
        before = set(bpy.data.actions)
        frames = []
        for k, (name, _f, entries) in enumerate(poses):
            f = _FO_FRAME0 + k
            _fo_apply_pose(arm, entries)
            _fo_key_all(arm, f)
            frames.append((name, f))
        self.made += [a for a in bpy.data.actions if a not in before]
        return frames

    def restore(self):
        for name, (had, act, slot, basis) in self.anim.items():
            arm = bpy.data.objects.get(name)
            if arm is None:
                continue
            if arm.animation_data is not None:
                arm.animation_data.action = None
            for a in self.made:
                try:
                    if self.keep_as:
                        a.name = self.keep_as
                        a.use_fake_user = True
                        self.kept = a.name
                    else:
                        bpy.data.actions.remove(a)
                except ReferenceError:
                    pass
            if had:
                if arm.animation_data is None:
                    arm.animation_data_create()
                arm.animation_data.action = act
                if slot is not None and hasattr(arm.animation_data, "action_slot"):
                    try:
                        arm.animation_data.action_slot = slot
                    except (TypeError, RuntimeError, AttributeError):
                        pass
            elif arm.animation_data is not None:
                arm.animation_data_clear()
            for pb in arm.pose.bones:
                if pb.name in basis:
                    pb.matrix_basis = basis[pb.name]
        for a in self.arms:
            try:
                a.data.pose_position = self.pos[a.name]
            except ReferenceError:
                pass
        bpy.context.view_layer.update()


def _fo_covered(gtree, pts, cover, poke=0.0):
    """Indices i of (point, normal) pts under the garment: a ray along the normal leaves the garment through its INSIDE
    (the face it meets faces the same way) within cover. A ray that first meets the garment's outside - an arm beside a
    bib - reads as not under it; a lining's inward face is passed through to the shell behind it. With poke, a point
    that has come THROUGH the garment counts too: the ray back into the body meets the garment's outside within poke
    (a belly through a bib), so a fit that pokes at rest is measured rather than left out of the test."""
    out = []
    for i, (p, n) in enumerate(pts):
        if n.length < 1e-12:
            continue
        o, left = p + n * 1e-5, cover
        for _hit in range(4):
            loc, nrm, _f, d = gtree.ray_cast(o, n, left)
            if loc is None:
                break
            if nrm.dot(n) > 0:
                out.append(i)
                break
            left -= d + 1e-5
            if left <= 0:
                break
            o = loc + n * 1e-5
        else:
            continue
        if out and out[-1] == i or poke <= 0:
            continue
        loc, nrm, _f, _d = gtree.ray_cast(p - n * 1e-5, -n, poke)
        if loc is not None and nrm.dot(n) > 0:
            out.append(i)
    return out


def _fo_bvh(obj):
    from mathutils.bvhtree import BVHTree   # noqa: PLC0415
    bm = evaluated_world(obj, True)
    try:
        return BVHTree.FromBMesh(bm), [(v.co.copy(), v.normal.copy()) for v in bm.verts], bm_boundary(bm)
    finally:
        bm.free()


def bm_boundary(bm):
    return [v.co.copy() for v in bm.verts if any(len(e.link_faces) == 1 for e in v.link_edges)]


def _fo_gap_stats(gar, body, reach=0.15):
    """The garment's rest vertices' distance to the body's skin (both as they evaluate now), within reach - its ease."""
    from mathutils.bvhtree import BVHTree   # noqa: PLC0415
    bb = evaluated_world(body, True)
    gb = evaluated_world(gar, True)
    try:
        tree = BVHTree.FromBMesh(bb)
        d = []
        for v in gb.verts:
            loc, nrm, _i, dist = tree.find_nearest(v.co, reach)
            if loc is not None:
                d.append(dist if nrm.dot(v.co - loc) >= 0 else -dist)
    finally:
        bb.free()
        gb.free()
    s = stats(d, [5, 95], 4)
    if s:
        s["insideBody"] = sum(1 for x in d if x < 0)
    return s


def _fo_places(vidx, rest_pts, near, limit=6):
    """[{bone, vertices, at}] - the failing body vertices grouped by the rest bone nearest each, most first."""
    rows = {}
    for vi in vidx:
        p = rest_pts[vi]
        b = near(p)
        r = rows.setdefault(b, [0, Vector((0.0, 0.0, 0.0))])
        r[0] += 1
        r[1] += p
    out = [{"bone": b, "vertices": n, "at": [round(c, 3) for c in (s / n)]} for b, (n, s) in rows.items()]
    out.sort(key=lambda r: -r["vertices"])
    return out[:limit]


def op_recipe_fit_outfit(params):
    """Fit a bought garment onto a body in one call, check it in test poses, and refuse - naming the pose and the place -
    where skin shows through or a strap leaves the body.

    THE GAP (spec 1.4 "Fit a bought outfit to a body in one call", 2026-11-07): a bought outfit made for one body was
    refitted to another by hand through six tools in an order learned the hard way. This composes them, on a COPY
    (`output`, default '<garment>_fit'); the garment itself is never written:
      1. shape    DEFORM (default with sourceBody): the copy follows sourceBody's skin onto body's through a Surface
                  Deform binding (transfer_shape_keys), so its ease and folds come along - when the two bodies differ
                  in topology sourceBody is first wrapped onto body (wrap_to_target); align RIGID | UNIFORM first fits
                  sourceBody onto body (fit_mesh) and carries the garment. WRAP (no sourceBody): the copy itself is
                  wrapped onto body at `ease` (a skin-tight piece; a loose garment wants a sourceBody).
      2. clearance push_out_of, garment side, to `clearance` (default 0.004 m) smoothed `smooth` (8) rings, at rest.
      3. weights  transfer_weights from body, normalize_weights, an Armature modifier on body's armature.
      4. poses    preset armsRaised, stride and crouch (default) turned by role off the bone names, or {name, bones}
                  (keyed on a scratch action removed afterwards - keepPoses keeps it as 'MifFitPoses_<output>'), or
                  {name, frame} you keyed.
      5. push     push_out_for_poses at rest and then over the poses (margin 0.004, spread 0.05, maxPush 0.05, passes
                  4), against the body vertices the copy covers at rest (or that came up to _FO_POKE through it) with
                  the garment point nearest them more than edgeBand (0.05 m) from its openings.
      6. check    pose_coverage at rest and at each pose: skin in a fold and patches under minPatch (3) are left out
                  and counted; more than `tolerance` (0) vertices left REFUSES. straps {vertexGroup, maxGap 0.02} - a
                  strap vertex that rests on the body standing off it by more than maxGap in a pose REFUSES. The
                  refusal names the pose, the bones nearest the places with a count and a point, and deletes the copy
                  unless keepOnRefusal.
      7. hide     hideCovered: the body vertices covered in every pose (one ring in from the edge) go in a vertex group
                  'MifHide_<output>' and a Mask modifier hides them.
    Reply: each step and what it did, the gap to the skin before and after, per pose the uncovered counts before and
    after the push, strap gaps, hidden vertices. Refused before any write: a body with no armature or weights, a
    garment with shape keys, an output name that exists, a pose bone with constraints, presets on a rig whose bones
    are not named for them. The body, sourceBody and armature are shown while it runs (a hidden object is not
    evaluated) and put back.
    """
    tail = "NOTHING was changed."
    reject_unknown(params, _FO_KEYS, "recipe_fit_outfit")
    gar = get_object(take(params, "garment", required=True, kind=str), want_mesh=True)
    body = get_object(take(params, "body", required=True, kind=str), want_mesh=True)
    src_name = take(params, "sourceBody", default=None, kind=str)
    src = get_object(src_name, want_mesh=True) if src_name else None
    if gar is body or src is gar or src is body:
        raise MifOpError("garment, body and sourceBody must be three different meshes. %s" % tail)
    arm_name = take(params, "armature", default=None, kind=str)
    if arm_name:
        arm = get_object(arm_name)
    else:
        arm = next((m.object for m in body.modifiers if m.type == "ARMATURE" and m.object is not None), None)
        if arm is None and body.parent is not None and body.parent.type == "ARMATURE":
            arm = body.parent
    if arm is None or arm.type != "ARMATURE":
        raise MifOpError("'%s' is not skinned to an armature (no Armature modifier, no armature parent) and none was "
                         "named - the poses and the weights come from its rig. Pass armature, or skin the body first. %s"
                         % (body.name, tail))
    bone_names = {b.name for b in arm.data.bones}
    if not any(g.name in bone_names for g in body.vertex_groups):
        raise MifOpError("'%s' has no vertex group named for a bone of '%s', so there are no weights to give the "
                         "garment and the body would not pose. %s" % (body.name, arm.name, tail))
    if gar.data.shape_keys is not None:
        raise MifOpError("'%s' has shape keys - the fit writes the copy's rest shape and a key would shift against it. "
                         "Apply or remove them on a copy first (apply_shape_keys). %s" % (gar.name, tail))
    method = str(take(params, "method", default="DEFORM" if src else "WRAP", kind=str)).upper()
    if method not in ("DEFORM", "WRAP"):
        raise MifOpError("method is DEFORM (follow sourceBody's skin onto body's) or WRAP (wrap onto body at ease), got "
                         "%r. %s" % (method, tail))
    if method == "DEFORM" and src is None:
        raise MifOpError("method DEFORM follows the body the garment was made on onto the new one - pass sourceBody, "
                         "or method WRAP with an ease. %s" % tail)
    align = take(params, "align", default=None, kind=str)
    align = align.upper() if align else None
    if align not in (None, "NONE", "RIGID", "UNIFORM"):
        raise MifOpError("align is RIGID or UNIFORM (fit_mesh's modes), got %r. %s" % (align, tail))
    if align == "NONE":
        align = None
    if align and src is None:
        raise MifOpError("align fits sourceBody onto body and carries the garment - it needs sourceBody. %s" % tail)

    def num(key, default, lo, hi):
        v = take_float(params, key, default=default)
        if v is None:
            return None
        if not (math.isfinite(v) and lo <= v <= hi):
            raise MifOpError("%s runs %g to %g, got %r. %s" % (key, lo, hi, v, tail))
        return v
    ease = num("ease", None, 0.0, 0.5)
    if method == "WRAP" and ease is None:
        raise MifOpError("method WRAP puts the garment at `ease` metres off the skin - pass ease (a skin-tight piece "
                         "0.002-0.01). %s" % tail)
    if method == "DEFORM" and ease is not None:
        raise MifOpError("ease is WRAP's: DEFORM carries the garment's own ease from sourceBody. %s" % tail)
    falloff = num("falloff", 4.0, 2.0, 16.0)
    clearance = num("clearance", 0.004, 0.0, 0.1)
    margin = num("margin", 0.004, 0.0, 0.1)
    spread = num("spread", 0.05, 1e-4, 1.0)
    max_push = num("maxPush", 0.05, 1e-4, 0.5)
    cover = num("coverDistance", 0.15, 1e-3, 1.0)
    band = num("edgeBand", 0.05, 0.0, 1.0)
    smooth = take_int(params, "smooth", default=8)
    passes = take_int(params, "passes", default=4)
    tolerance = take_int(params, "tolerance", default=0)
    min_patch = take_int(params, "minPatch", default=3)
    if not 1 <= min_patch <= 1000:
        raise MifOpError("minPatch is 1-1000 body vertices, got %r. %s" % (min_patch, tail))
    if not 0 <= smooth <= 50 or not 1 <= passes <= 20 or tolerance < 0:
        raise MifOpError("smooth is 0-50, passes 1-20 and tolerance 0 or more, got %r, %r, %r. %s"
                         % (smooth, passes, tolerance, tail))
    landmarks = params.get("landmarks")
    if landmarks is not None and method != "WRAP" and (src is None or len(src.data.vertices) == len(body.data.vertices)):
        raise MifOpError("landmarks guide wrap_to_target, which runs for method WRAP or a sourceBody whose topology "
                         "differs from body's. %s" % tail)
    straps = params.get("straps")
    strap_gap = None
    if straps is not None:
        if not isinstance(straps, dict) or "vertexGroup" not in straps or set(straps) - {"vertexGroup", "maxGap"}:
            raise MifOpError("straps is {vertexGroup, maxGap (m, default 0.02)}, got %r. %s" % (straps, tail))
        sg = gar.vertex_groups.get(str(straps["vertexGroup"]))
        if sg is None:
            raise MifOpError("straps.vertexGroup '%s' is not a group of '%s' (it has: %s). %s"
                             % (straps["vertexGroup"], gar.name, ", ".join(g.name for g in gar.vertex_groups) or "none",
                                tail))
        strap_gap = finite_float(straps.get("maxGap", 0.02), "straps.maxGap")
        if not 0 < strap_gap <= 0.5:
            raise MifOpError("straps.maxGap is above 0 and at most 0.5 m, got %r. %s" % (strap_gap, tail))
    hide = take_bool(params, "hideCovered", default=False)
    keep_bad = take_bool(params, "keepOnRefusal", default=False)
    keep_poses = take_bool(params, "keepPoses", default=False)
    out_name = str(take(params, "output", default=gar.name + "_fit", kind=str)).strip()
    if not out_name or bpy.data.objects.get(out_name) is not None:
        raise MifOpError("output '%s' %s - the fit is written to a new object. Name another, or delete that one. %s"
                         % (out_name, "is empty" if not out_name else "already exists", tail))
    if hide and body.modifiers.get("MifHide " + out_name) is not None:
        raise MifOpError("'%s' already has a Mask modifier 'MifHide %s' from an earlier fit. Remove it or name another "
                         "output. %s" % (body.name, out_name, tail))
    poses = _fo_parse_poses(params.get("poses"), arm, tail)
    if keep_poses and poses[0][1] is not None:
        raise MifOpError("keepPoses keeps the action the recipe keys its poses on, and these poses are frames you keyed "
                         "- they are already yours to look at. %s" % tail)
    if keep_poses and bpy.data.actions.get("MifFitPoses_" + out_name) is not None:
        raise MifOpError("an action 'MifFitPoses_%s' already exists from an earlier fit; keepPoses would need its name. "
                         "%s" % (out_name, tail))
    require_editable(body, "hide faces on" if hide else "measure")
    sim_frames = [f for _n, f, _e in poses if f is not None] or [_FO_FRAME0 + k for k in range(len(poses))]
    sim = simulation_guard(params, sim_frames, "recipe_fit_outfit", tail=tail)
    allow_sim = take_bool(params, "allowSimulation", default=False)

    from .ops_meshdata import op_transfer_shape_keys   # noqa: PLC0415
    from .ops_mesh import op_push_out_of   # noqa: PLC0415
    from .ops_rig import op_normalize_weights, op_transfer_weights   # noqa: PLC0415
    from .ops_common import assign_id_name   # noqa: PLC0415

    steps = []
    temps = []
    # an earlier fit's hidden faces come back while this one measures - on the body, and on a sourceBody copied from
    # a body that had them (a source with holes cannot bind every garment vertex)
    masks_off = [m for o in (body, src) if o is not None for m in o.modifiers
                 if m.type == "MASK" and m.name.startswith("MifHide") and m.show_viewport]
    other_arms = [m.object for o in (gar, src) if o is not None for m in o.modifiers
                  if m.type == "ARMATURE" and m.object is not None]
    rig = _FoRig([arm] + other_arms)
    if keep_poses:
        rig.keep_as = "MifFitPoses_" + out_name
    copy = None
    refused = None
    done = False
    vis_borrowed = []
    try:
        for m in masks_off:
            m.show_viewport = False
        # HIDDEN OBJECTS ARE NOT EVALUATED: a garment hidden in the viewport gave a copy Surface Deform could not bind,
        # and a hidden body would not pose (2026-11-07, every fit after a render that hid the bought garment). The
        # body, sourceBody and armature are shown while the recipe runs and put back; the copy stays shown.
        for o in (body, src, arm, gar):
            if o is not None:
                vis_borrowed.append((o, o.hide_viewport, o.hide_get()))
                o.hide_viewport = False
                o.hide_set(False)
        bpy.context.view_layer.update()
        hid = [o.name for o in (body, src, arm) if o is not None and not o.visible_get()]
        if hid:
            raise MifOpError("%s cannot be shown in this view layer (a collection holding it is hidden or excluded), "
                             "so it would not evaluate - its pose and modifiers would be skipped. Show its collection. %s"
                             % (", ".join("'%s'" % n for n in hid), tail))
        rig.rest(True)
        near = nearest_bone_fn(arm)
        ref_body = src or body
        gap_before = _fo_gap_stats(gar, ref_body)

        # ---- the copy: the garment's data and modifiers, its Armature modifiers dropped (step 3 adds ours) --------
        copy = gar.copy()
        copy.data = gar.data.copy()
        for c in gar.users_collection:
            c.objects.link(copy)
        if not gar.users_collection:
            bpy.context.scene.collection.objects.link(copy)
        mw = gar.matrix_world.copy()
        copy.parent = None
        copy.matrix_world = mw
        for m in [m for m in copy.modifiers if m.type == "ARMATURE"]:
            copy.modifiers.remove(m)
        if copy.animation_data is not None:
            copy.animation_data_clear()
        assign_id_name(copy, out_name, bpy.data.objects)
        assign_id_name(copy.data, out_name, bpy.data.meshes)
        bpy.context.view_layer.update()
        if not copy.visible_get():
            bpy.context.scene.collection.objects.link(copy)
            bpy.context.view_layer.update()
        bpy.context.view_layer.update()
        ev = copy.evaluated_get(bpy.context.evaluated_depsgraph_get())
        em = ev.to_mesh()
        n_ev = len(em.vertices)
        ev.to_mesh_clear()
        if n_ev != len(copy.data.vertices):
            raise MifOpError("'%s' evaluates to %d vertices and its mesh has %d - a modifier (%s) changes its topology, so "
                             "the poses cannot be traced back to its rest vertices. Apply or remove it on the garment "
                             "first. %s" % (gar.name, n_ev, len(copy.data.vertices),
                                            ", ".join(m.name for m in copy.modifiers) or "none", tail))
        steps.append({"step": "copy", "detail": "'%s' copied to '%s'; the garment is not written" % (gar.name, copy.name)})

        # ---- 1. shape -------------------------------------------------------------------------------------------
        if method == "DEFORM":
            sbm = evaluated_world(src, True)
            # TRIANGLES, so no face is concave: Surface Deform refuses to bind to a concave polygon, and a body made
            # heavier along its normals has them - a heavy sourceBody never bound (2026-11-07). Triangulating adds
            # no vertex, so the key below still matches the body vertex for vertex.
            bmesh.ops.triangulate(sbm, faces=sbm.faces[:])
            tmp_me = bpy.data.meshes.new("MifFitSource")
            sbm.to_mesh(tmp_me)
            sbm.free()
            tmp = bpy.data.objects.new("MifFitSource", tmp_me)
            bpy.context.scene.collection.objects.link(tmp)
            temps.append(tmp)
            bpy.context.view_layer.update()
            if align:
                fit = op_fit_mesh({"object": tmp.name, "target": body.name, "mode": align, "carry": [copy.name]})
                from mathutils import Matrix   # noqa: PLC0415
                tmp.data.transform(tmp.matrix_world)
                tmp.matrix_world = Matrix.Identity(4)
                bpy.context.view_layer.update()
                steps.append({"step": "align", "detail": "fit_mesh %s: rms %s -> %s m" % (
                    align, fit["before"]["rms"], fit["after"]["rms"])})
            bbm = evaluated_world(body, True)
            try:
                same = len(bbm.verts) == len(tmp_me.vertices)
                target_co = [v.co.copy() for v in bbm.verts] if same else None
            finally:
                bbm.free()
            how = "the same topology, vertex for vertex"
            if not same:
                wrap_me = tmp_me.copy()
                wrap = bpy.data.objects.new("MifFitWrap", wrap_me)
                bpy.context.scene.collection.objects.link(wrap)
                temps.append(wrap)
                bpy.context.view_layer.update()
                wp = {"object": wrap.name, "target": body.name, "smoothIterations": 3}
                if landmarks is not None:
                    wp["landmarks"] = landmarks
                w = op_wrap_to_target(wp)
                target_co = [v.co.copy() for v in wrap_me.vertices]
                how = ("a different topology (%d and %d vertices): sourceBody wrapped onto body first, %s m off it at "
                       "most" % (len(tmp_me.vertices), len(body.data.vertices),
                                 (w.get("after") or {}).get("max", w.get("after"))))
            tmp.shape_key_add(name="Basis", from_mix=False)
            kb = tmp.shape_key_add(name="MifFit", from_mix=False)
            for i, co in enumerate(target_co):
                kb.data[i].co = co
            r = op_transfer_shape_keys({"source": tmp.name, "target": copy.name, "keys": ["MifFit"],
                                        "falloff": falloff})
            key = copy.data.shape_keys.key_blocks["MifFit"]
            flat = [0.0] * (3 * len(copy.data.vertices))
            key.data.foreach_get("co", flat)
            copy.shape_key_clear()
            copy.data.vertices.foreach_set("co", flat)
            copy.data.update()
            steps.append({"step": "shape", "method": "DEFORM",
                          "detail": "followed '%s' onto '%s' by Surface Deform (%s); the garment moved %s m at most"
                                    % (src.name, body.name, how, r["keys"][0]["targetMaxDisplacement"])})
        else:
            wp = {"object": copy.name, "target": body.name, "offset": ease, "smoothIterations": 3}
            if landmarks is not None:
                wp["landmarks"] = landmarks
            w = op_wrap_to_target(wp)
            steps.append({"step": "shape", "method": "WRAP",
                          "detail": "wrapped onto '%s' at %g m; distance to it %s -> %s" % (
                              body.name, ease, w.get("before"), w.get("after"))})

        # ---- 2. clearance at rest ---------------------------------------------------------------------------------
        if clearance > 0:
            try:
                p = op_push_out_of({"object": copy.name, "collider": body.name, "side": "garment",
                                    "offset": clearance, "smooth": smooth, "reach": max(0.03, 4 * clearance)})
                steps.append({"step": "clearance", "detail": "push_out_of to %g m: %s vertices pushed, %s still through"
                              % (clearance, p.get("pushed", p.get("verticesPushed")),
                                 p.get("remainingPenetration"))})
            except MifOpError as exc:
                if "nothing was tested" not in str(exc):
                    raise
                steps.append({"step": "clearance", "detail": "no garment vertex within reach of the skin - nothing "
                                                             "to push"})

        # ---- 3. weights -------------------------------------------------------------------------------------------
        op_transfer_weights({"source": body.name, "destination": copy.name})
        am = copy.modifiers.new("Armature", "ARMATURE")
        am.object = arm
        if len(copy.modifiers) > 1:
            with bpy.context.temp_override(object=copy, active_object=copy):
                bpy.ops.object.modifier_move_to_index(modifier=am.name, index=0)
        if body.parent is arm:
            # parented as the body is; the inverse keeps the copy where it stands
            copy.parent = arm
            copy.matrix_parent_inverse = arm.matrix_world.inverted()
        nw = op_normalize_weights({"object": copy.name, "maxInfluences": 4})
        steps.append({"step": "weights", "detail": "transfer_weights from '%s', normalized to 4 influences (%s "
                                                   "unweighted); Armature modifier on '%s'"
                                                   % (body.name, nw.get("unweightedCount", nw.get("unweighted", 0)),
                                                      arm.name)})
        bpy.context.view_layer.update()
        gap_after = _fo_gap_stats(copy, body)

        # ---- the body vertices the copy covers at rest, clear of its openings ----------------------------------------
        from mathutils.kdtree import KDTree   # noqa: PLC0415
        gtree, _gp, gbound = _fo_bvh(copy)
        bbm = evaluated_world(body, True)
        rest_pts = [(v.co.copy(), v.normal.copy()) for v in bbm.verts]
        body_edges = [(e.verts[0].index, e.verts[1].index) for e in bbm.edges]
        bbm.free()
        under = set(_fo_covered(gtree, rest_pts, cover, _FO_POKE))
        if gbound and band > 0:
            kd = KDTree(len(gbound))
            for i, p in enumerate(gbound):
                kd.insert(p, i)
            kd.balance()
            # MEASURED FROM THE GARMENT POINT NEAREST THE SKIN, not from the skin: an ankle under a wide trouser hem
            # stands more than a band from every hem vertex, and the hem is still the garment nearest it
            # (2026-11-07, a foot read as skin showing through a stride).
            kept = set()
            for i in under:
                q = gtree.find_nearest(rest_pts[i][0])[0]
                if q is not None and kd.find(q)[2] > band:
                    kept.add(i)
            under = kept
        if not under:
            raise MifOpError("no body vertex lies under '%s' at rest, %g m or more from its openings - nothing to check "
                             "in the poses. Its faces may point inward (recalculate normals outward), or it is not on "
                             "the body. %s" % (copy.name, band, tail))
        tg = body.vertex_groups.get(_FO_GROUP) or body.vertex_groups.new(name=_FO_GROUP)
        tg.add(sorted(under), 1.0, "REPLACE")
        rest_world = [p for p, _n in rest_pts]

        # ---- 4. poses ---------------------------------------------------------------------------------------------
        rig.rest(False)
        if poses[0][1] is None:
            named = rig.key_poses(arm, poses)
        else:
            named = [(n, f) for n, f, _e in poses]
        frames = [f for _n, f in named]
        steps.append({"step": "poses", "detail": ", ".join("%s (frame %d)" % nf for nf in named)})

        # ---- 5. push at rest (the pose position at rest, this frame), then across the poses -------------------------
        region = {"vertexGroup": _FO_GROUP, "minWeight": 0.5}
        cur = bpy.context.scene.frame_current
        push_args = {"garment": copy.name, "body": body.name, "region": region, "margin": margin, "spread": spread,
                     "maxPush": max_push, "passes": passes, "allowSimulation": allow_sim}
        rig.rest(True)
        pushed_rest = op_push_out_for_poses(dict(push_args, frames=[cur]))
        rig.rest(False)
        pushed = op_push_out_for_poses(dict(push_args, frames=frames))
        steps.append({"step": "push", "detail": "push_out_for_poses moved %d vertices at rest and %d in the poses (%s), "
                                                "%d capped at %g m"
                                                % (pushed_rest["verticesMoved"], pushed["verticesMoved"],
                                                   pushed.get("push"), pushed["capped"] + pushed_rest["capped"],
                                                   max_push)})

        # ---- 6. check: rest, then each pose -------------------------------------------------------------------------
        rig.rest(True)
        cov_rest = op_pose_coverage({"garment": copy.name, "body": body.name, "frames": [cur], "region": region,
                                     "limit": 100000, "allowSimulation": allow_sim})
        rig.rest(False)
        cov = op_pose_coverage({"garment": copy.name, "body": body.name, "frames": frames, "region": region,
                                "limit": 100000, "allowSimulation": allow_sim})
        by_frame = {r["frame"]: r for r in cov["perFrame"]}
        by_frame["rest"] = (cov_rest["perFrame"] or [{}])[0]
        before = {r.get("frame"): r for r in pushed.get("before") or []}
        before["rest"] = (pushed_rest.get("before") or [{}])[0]
        # SKIN IN A FOLD CANNOT SHOW: a crouch closes the hip crease, and a vertex there reads uncovered (its ray runs
        # into the thigh, not the garment) though the body itself hides it. A ray along the posed normal that meets
        # the body again within _FO_FOLD is left out and counted (2026-11-07, both hip creases refused a crouch).
        folded = {}

        def fold(f):
            unc = set((by_frame.get(f) or {}).get("uncoveredVertices") or ())
            if not unc:
                folded[f] = set()
                return
            from mathutils.bvhtree import BVHTree   # noqa: PLC0415
            bb = evaluated_world(body, True)
            try:
                t = BVHTree.FromBMesh(bb)
                folded[f] = {vi for vi in unc if bb.verts[vi].normal.length > 1e-12 and t.ray_cast(
                    bb.verts[vi].co + bb.verts[vi].normal * 2e-3, bb.verts[vi].normal, _FO_FOLD)[0] is not None}
            finally:
                bb.free()
        _frame_loop(frames, fold)
        rig.rest(True)
        fold("rest")
        rig.rest(False)
        # A PATCH, NOT A RAY: skin that shows is an area. A lone vertex whose ray grazes out through an opening (a hip
        # crease looking out of an overall's side slit in a crouch) is listed as isolated and not counted; minPatch
        # (default 3) connected uncovered vertices are (2026-11-07, two such single vertices refused a fit that showed
        # no skin in its render).
        nbr = {}
        for a, b2 in body_edges:
            nbr.setdefault(a, []).append(b2)
            nbr.setdefault(b2, []).append(a)

        def patches(vs):
            left, out_p = set(vs), []
            while left:
                seed = left.pop()
                comp, stack = [seed], [seed]
                while stack:
                    for j in nbr.get(stack.pop(), ()):
                        if j in left:
                            left.discard(j)
                            comp.append(j)
                            stack.append(j)
                out_p.append(comp)
            return out_p
        per_pose, failures = [], []
        for name, f in [("rest", "rest")] + named:
            row = by_frame.get(f) or {}
            fo = folded.get(f) or set()
            raw_unc = [vi for vi in row.get("uncoveredVertices") or [] if vi not in fo]
            comps = patches(raw_unc)
            unc = [vi for c in comps if len(c) >= min_patch for vi in c]
            isolated = sum(len(c) for c in comps if len(c) < min_patch)
            b = before.get(f) or {}
            pr = {"pose": name, "frame": f if f != "rest" else None, "tested": len(under), "uncoveredBeforePush": b.get("uncovered"),
                  "uncovered": len(unc) if row.get("uncovered") is not None else None, "inside": row.get("inside"),
                  "holes": row.get("holes"), "inFolds": len(fo), "isolated": isolated,
                  "patches": sorted((len(c) for c in comps if len(c) >= min_patch), reverse=True)[:10]}
            if row.get("uncovered") is None:
                pr["skipped"] = True
            elif len(unc) > tolerance:
                pr["places"] = _fo_places(unc, rest_world, near)
                failures.append(("skin", name, f, pr))
            per_pose.append(pr)

        # straps: a strap vertex resting on the body (within maxGap at rest) must stay within maxGap in every pose
        strap_rows = []
        if strap_gap is not None:
            from mathutils.bvhtree import BVHTree   # noqa: PLC0415
            sg = copy.vertex_groups.get(str(straps["vertexGroup"]))
            members = [v.index for v in copy.data.vertices for g in v.groups if g.group == sg.index and g.weight >= 0.5]
            rig.rest(True)
            gb = evaluated_world(copy, True)
            bb = evaluated_world(body, True)
            try:
                btree = BVHTree.FromBMesh(bb)
                resting, rest_g = [], {}
                for vi in members:
                    loc, _n, _i, d = btree.find_nearest(gb.verts[vi].co, strap_gap)
                    if loc is not None:
                        resting.append(vi)
                        rest_g[vi] = gb.verts[vi].co.copy()
            finally:
                gb.free()
                bb.free()
            rig.rest(False)
            if not resting:
                raise MifOpError("no vertex of straps group '%s' (%d at weight 0.5 or more) rests within maxGap %g m of "
                                 "the body after the fit - a strap check of nothing is not clean. %s"
                                 % (sg.name, len(members), strap_gap, tail))
            gaps = {}

            def one(f):
                gb2 = evaluated_world(copy, True)
                bb2 = evaluated_world(body, True)
                try:
                    t2 = BVHTree.FromBMesh(bb2)
                    off = []
                    worst = 0.0
                    for vi in resting:
                        _l, _n, _i, d = t2.find_nearest(gb2.verts[vi].co)
                        d = d if d is not None else math.inf
                        worst = max(worst, d)
                        if d > strap_gap:
                            off.append(vi)
                    gaps[f] = (worst, off)
                finally:
                    gb2.free()
                    bb2.free()
            _frame_loop(frames, one)
            for name, f in named:
                worst, off = gaps.get(f, (None, []))
                sr = {"pose": name, "frame": f, "strapVertices": len(resting), "maxGap": round(worst, 4)
                      if worst is not None else None, "off": len(off)}
                if off:
                    sr["places"] = _fo_places(off, {vi: rest_g[vi] for vi in off}, near)
                    failures.append(("strap", name, f, sr))
                strap_rows.append(sr)

        if failures:
            parts = []
            for kind, name, f, r in failures:
                pl = "; ".join("%s (%d, near %s)" % (p["bone"], p["vertices"], p["at"]) for p in r["places"][:3])
                where = "at rest" if f == "rest" else "in pose '%s' (frame %d)" % (name, f)
                if kind == "skin":
                    parts.append("%s skin shows through at %s - %d body vertices uncovered "
                                 "in %d patch(es) of %d or more (of all uncovered %d are under the garment's surface, %d "
                                 "in an opening; %d in a fold and %d isolated are left out)"
                                 % (where, pl, r["uncovered"], len(r["patches"]), min_patch, r["inside"] or 0,
                                    r["holes"] or 0, r["inFolds"], r["isolated"]))
                else:
                    parts.append("in pose '%s' (frame %d) the strap leaves the body at %s - %d strap vertices more than "
                                 "%g m off it, %s m at most" % (name, f, pl, r["off"], strap_gap, r["maxGap"]))
            refused = MifOpError(
                "the fit does not hold: %s. %s Raise maxPush or passes for skin under the surface, loosen the "
                "garment's ease (or fit from its own sourceBody) for an opening, and weight a strap to the bone it "
                "rides on. Within tolerance %d vertices per pose, patches of %d or more."
                % ("; ".join(parts), ("The fitted copy '%s' is kept to look at (keepOnRefusal)." % copy.name)
                                    if keep_bad else "The fitted copy was deleted; NOTHING was left behind.",
                   tolerance, min_patch))
            refused.payload = {"perPose": per_pose, "straps": strap_rows, "steps": steps,
                               "kept": copy.name if keep_bad else None}
            raise refused

        # ---- 7. hide the covered body faces ----------------------------------------------------------------------
        hidden = None
        if hide:
            covered_all = set(under)
            rig.rest(True)
            covered_all &= set(_fo_covered(_fo_bvh(copy)[0], _body_pts(body), cover))
            rig.rest(False)

            def keep_cov(f):
                covered_all.intersection_update(_fo_covered(_fo_bvh(copy)[0], _body_pts(body), cover))
            _frame_loop(frames, keep_cov)
            inner = {i for i in covered_all if all(j in covered_all for j in nbr.get(i, ()))}
            if inner:
                hg = body.vertex_groups.get("MifHide_" + copy.name) or body.vertex_groups.new(name="MifHide_" + copy.name)
                hg.add(sorted(inner), 1.0, "REPLACE")
                mm = body.modifiers.new("MifHide " + copy.name, "MASK")
                mm.vertex_group = hg.name
                mm.invert_vertex_group = True
                n_faces = len(body.data.polygons)
                bpy.context.view_layer.update()
                ev = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
                em = ev.to_mesh()
                shown = len(em.polygons)
                ev.to_mesh_clear()
                hidden = {"vertexGroup": hg.name, "modifier": mm.name, "vertices": len(inner),
                          "facesHidden": n_faces - shown, "facesShown": shown,
                          "note": "covered in every pose, one ring in from the edge; the Mask modifier is the last on "
                                  "the stack - turn it off (show_viewport) to measure the body again"}
            else:
                hidden = {"vertices": 0, "note": "no body vertex stays covered in every pose one ring in from the edge"}
            steps.append({"step": "hide", "detail": "%d body vertices hidden" % hidden["vertices"]})
        done = True
    except MifOpError as exc:
        if exc is refused:
            raise
        raise MifOpError("recipe_fit_outfit stopped: %s%s" % (str(exc).replace(" " + tail, ""),
                                                              " The fitted copy was deleted; the garment and body are "
                                                              "as they were."))
    finally:
        if not done and copy is not None and not (refused is not None and keep_bad):
            remove_objects([copy])
            copy = None
        rig.restore()
        if refused is not None and rig.kept:
            refused.payload["posesAction"] = rig.kept
        g = body.vertex_groups.get(_FO_GROUP)
        if g is not None:
            body.vertex_groups.remove(g)
        for m in masks_off:
            try:
                m.show_viewport = True
            except ReferenceError:
                pass
        for o, hv, hg in vis_borrowed:
            try:
                if o is copy:
                    continue
                o.hide_viewport = hv
                o.hide_set(hg)
            except (ReferenceError, RuntimeError):
                pass
        if temps:
            remove_objects(temps)
        for me in [m for m in bpy.data.meshes if m.name.startswith(("MifFitSource", "MifFitWrap")) and not m.users]:
            bpy.data.meshes.remove(me)
    out = {"garment": gar.name, "output": copy.name, "body": body.name, "sourceBody": src.name if src else None,
           "armature": arm.name, "method": method, "align": align, "steps": steps,
           "gapToSkin": {"before": gap_before, "after": gap_after,
                         "note": "each garment vertex's distance to the skin within 0.15 m (negative inside it): the "
                                 "garment on %s before, the fit on %s after" % (ref_body.name, body.name)},
           "testedBodyVertices": len(under), "perPose": per_pose, "tolerance": tolerance,
           "minPatch": min_patch}
    if strap_rows:
        out["straps"] = strap_rows
    if rig.kept:
        out["posesAction"] = rig.kept
        out["posesActionNote"] = ("the test poses, kept unassigned with a fake user: bl_assign_action puts them on '%s' "
                                  "to look at frames %s" % (arm.name, ", ".join(str(f) for f in frames)))
    if hide:
        out["hidden"] = hidden
    out.update(sim)
    return out


def _body_pts(body):
    bm = evaluated_world(body, True)
    try:
        return [(v.co.copy(), v.normal.copy()) for v in bm.verts]
    finally:
        bm.free()


# ---------------------------------------------------------------------------------------------------------------------
# THE BULK PREVIEWS (ops_preview.py; docs/15 section 5, 2026-10-09). fit_mesh moves objects, so its plan is the fit's own
# transform on each world matrix - the object, every object it carries and every child that follows it - read back and
# put back as transform_objects' are. The garment ops write a mesh and plan on a scratch copy (ops_preview.mesh_spec).
# ---------------------------------------------------------------------------------------------------------------------
def _plan_fit_mesh(request):
    from .ops_create import _world_transform   # noqa: PLC0415
    sink = {}
    resp = op_fit_mesh(dict(request, dryRun=True), _plan=sink)
    obj = bpy.data.objects[resp["object"]]
    carry = [bpy.data.objects[n] for n in sink["carry"]]
    for o in [obj] + carry:
        require_editable(o, "move")                       # the real call's refusals, which its dry run skips
    mat = sink["matrix"]
    movers = []
    for o in [obj] + carry + list(obj.children_recursive if hasattr(obj, "children_recursive") else ()):
        if o not in movers:
            movers.append(o)
    changes, unchanged = [], []
    for o in movers:
        row = {"kind": "transform", "op": "modify", "target": o.name, "field": "matrix_world",
               "before": _world_transform(o.matrix_world), "after": _world_transform(mat @ o.matrix_world)}
        if o is not obj and o not in carry:
            row["label"] = "follows %s" % obj.name
        (unchanged if row["before"] == row["after"] else changes).append(row)
    return {"changes": changes, "unchanged": unchanged, "context": [], "commit": dict(request), "response": resp}


def _preview_specs():
    from . import ops_create   # noqa: PLC0415
    from . import ops_preview as preview   # noqa: PLC0415
    return {
        "fit_mesh": {"plan": _plan_fit_mesh, "read": ops_create._read_transform_objects,
                     "capture": ops_create._capture_transform_objects, "restore": ops_create._restore_transform_objects},
        "blend_weights": preview.mesh_spec(op_blend_weights, keys=("object",)),
        "push_out_for_poses": preview.mesh_spec(op_push_out_for_poses, keys=("garment",)),
        "garment_reshape": preview.mesh_spec(op_garment_reshape, keys=("object",)),
        "wrap_to_target": preview.mesh_spec(op_wrap_to_target, keys=("object",)),
        "clear_envelope": preview.mesh_spec(op_clear_envelope, keys=("object",)),
    }


PREVIEW = _preview_specs()

OPS = {
    "clear_envelope": op_clear_envelope,
    "points_inside": op_points_inside,
    "fit_mesh": op_fit_mesh,
    "attach_hood": op_attach_hood,
    "build_sole": op_build_sole,
    "attach_sleeve": op_attach_sleeve,
    "displace_by_rule": op_displace_by_rule,
    "mesh_distance": op_mesh_distance,
    "deformation_stretch": op_deformation_stretch,
    "blend_weights": op_blend_weights,
    "pose_coverage": op_pose_coverage,
    "push_out_for_poses": op_push_out_for_poses,
    "garment_reshape": op_garment_reshape,
    "build_rib": op_build_rib,
    "gather_loop": op_gather_loop,
    "trim_to_outline": op_trim_to_outline,
    "add_inner_layer": op_add_inner_layer,
    "measure_neckline": op_measure_neckline,
    "build_collar": op_build_collar,
    "build_applique": op_build_applique,
    "build_band": op_build_band,
    "quilt_surface": op_quilt_surface,
    "add_accessory_physics": op_add_accessory_physics,
    "wrap_to_target": op_wrap_to_target,
    "retopo_panels": op_retopo_panels,
    "transfer_by_uv": op_transfer_by_uv,
    "loft_garment": op_loft_garment,
    "measure_garment_ease": op_measure_garment_ease,
    "recipe_fit_outfit": op_recipe_fit_outfit,
}
