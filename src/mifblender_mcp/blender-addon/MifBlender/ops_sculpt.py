"""MifBlender ops: the sculpt mask by rule, mask extract and slice, whole-mesh sculpt filters, symmetrize and symmetry
report, and smoothing a vertex region (second 1.3 audit, 2026-10-25).

WHY NATIVE, AND WHERE NOT. Sculpt-mode operators have failed headless here before: sculpt.mesh_filter segfaults in
background on 4.2.23 and 5.2.2 (ops_topology cloth_filter's set-up), and sculpt.cloth_filter answers PASS_THROUGH. So
the mask rules, the filters, the symmetry snap and the region smoothing are computed in mesh space here, the same on
every build. The ONE native operator used is the mask extract / slice, MEASURED to run headless on 3.6.23, 4.4.0 and
5.2.1 (2026-10-25: an upper-half mask of a 482-vertex sphere extracted to a 225-vertex shell with a Solidify) - under
its two names, mesh.paint_mask_extract / paint_mask_slice up to 4.4 and sculpt.paint_mask_extract / paint_mask_slice
from 5.0 (5.0 bl_ui/space_view3d.py:3892-3896, 5.2 :3975-3984; 4.4 :3865-3874).

THE MASK LIVES IN TWO PLACES (measured 2026-10-25): 3.6.23 keeps it in Mesh.vertex_paint_masks (bmesh's paint_mask
layer); 4.2.17 and 5.2.1 in the '.sculpt_mask' FLOAT attribute on points. Both are read and written here.
"""
from __future__ import annotations

import math

import bmesh
import bpy
from mathutils import Vector

from .ops_common import (MifOpError, finite_float, finite_floats, get_object, reject_unknown, require_editable, rnd,
                         selection_restore, selection_snapshot, shared_data_note, take, take_bool, take_float, take_int)

_MASK_ATTR = ".sculpt_mask"
_FACE_SET_ATTR = ".sculpt_face_set"


# ------------------------------------------------------------------------------------------------------- mask storage
def _mask_read(me):
    """The mask per vertex (0 where none was ever written), and where it was read from."""
    a = me.attributes.get(_MASK_ATTR)
    if a is not None and a.domain == "POINT" and a.data_type == "FLOAT":
        vals = [0.0] * len(me.vertices)
        a.data.foreach_get("value", vals)
        return vals, _MASK_ATTR
    vpm = getattr(me, "vertex_paint_masks", None)
    if vpm is not None and len(vpm):
        vals = [0.0] * len(me.vertices)
        vpm[0].data.foreach_get("value", vals)
        return vals, "vertex_paint_masks"
    return [0.0] * len(me.vertices), None


def _mask_write(me, weights):
    vals = [min(1.0, max(0.0, float(v))) for v in weights]
    if hasattr(me, "vertex_paint_masks"):
        bm = bmesh.new()
        try:
            bm.from_mesh(me)
            lay = bm.verts.layers.paint_mask.verify()
            bm.verts.ensure_lookup_table()
            for v, x in zip(bm.verts, vals):
                v[lay] = x
            bm.to_mesh(me)
        finally:
            bm.free()
        me.update()
        return "vertex_paint_masks"
    a = me.attributes.get(_MASK_ATTR)
    if a is None:
        a = me.attributes.new(_MASK_ATTR, "FLOAT", "POINT")
    a.data.foreach_set("value", vals)
    me.update()
    return _MASK_ATTR


def _mask_stats(vals):
    n = len(vals)
    return {"vertices": n, "masked": sum(1 for v in vals if v >= 0.5), "partly": sum(1 for v in vals if 0.0 < v < 0.5),
            "min": round(min(vals), 6) if n else None, "max": round(max(vals), 6) if n else None,
            "mean": round(sum(vals) / n, 6) if n else None}


def _neighbors(me):
    nb = [[] for _ in me.vertices]
    for e in me.edges:
        a, b = e.vertices
        nb[a].append(b)
        nb[b].append(a)
    return nb


class _ObjectMode:
    """OBJECT mode for the duration (mesh data written in sculpt or edit mode is overwritten on the way out), the mode
    put back after."""

    def __init__(self, obj):
        self.obj, self.mode = obj, obj.mode
        self.snap = None

    def __enter__(self):
        if self.mode != "OBJECT":
            self.snap = selection_snapshot()
            bpy.context.view_layer.objects.active = self.obj
            bpy.ops.object.mode_set(mode="OBJECT")
        return self

    def __exit__(self, *exc):
        if self.mode != "OBJECT":
            try:
                bpy.context.view_layer.objects.active = self.obj
                bpy.ops.object.mode_set(mode=self.mode)
            finally:
                selection_restore(self.snap)
        return False


def _mesh_object(name, what):
    obj = get_object(name, want_mesh=True)
    if obj.mode == "EDIT":
        raise MifOpError("'%s' is in EDIT mode, which keeps its own copy of the mesh and writes it back over this %s on "
                         "the way out. Leave edit mode first. NOTHING was changed." % (obj.name, what))
    return obj


def _start_vertex(obj, params, tail):
    if params.get("vertex") is not None:
        i = take_int(params, "vertex")
        if not 0 <= i < len(obj.data.vertices):
            raise MifOpError("vertex %d is out of range - '%s' has %d. %s" % (i, obj.name, len(obj.data.vertices), tail))
        return i
    near = params.get("near")
    if not isinstance(near, (list, tuple)) or len(near) != 3:
        raise MifOpError("EXPAND starts at vertex (an index) or near ([x, y, z] world). %s" % tail)
    p = obj.matrix_world.inverted_safe() @ Vector(finite_floats(list(near), "near"))
    return min(range(len(obj.data.vertices)), key=lambda k: (obj.data.vertices[k].co - p).length)


def _distances(me, mw, start, nb, by):
    """Edge steps (TOPOLOGY) or world edge-length distance (GEODESIC) from start to every vertex (None: unreached)."""
    import heapq
    n = len(me.vertices)
    dist = [None] * n
    dist[start] = 0.0
    if by == "TOPOLOGY":
        front = [start]
        while front:
            nxt = []
            for v in front:
                for w in nb[v]:
                    if dist[w] is None:
                        dist[w] = dist[v] + 1.0
                        nxt.append(w)
            front = nxt
        return dist
    world = [mw @ v.co for v in me.vertices]
    heap = [(0.0, start)]
    while heap:
        d, v = heapq.heappop(heap)
        if d > dist[v]:
            continue
        for w in nb[v]:
            nd = d + (world[v] - world[w]).length
            if dist[w] is None or nd < dist[w]:
                dist[w] = nd
                heapq.heappush(heap, (nd, w))
    return dist


def _falloff(t, kind):
    t = min(1.0, max(0.0, t))
    if kind == "CONSTANT":
        return 1.0
    if kind == "SMOOTH":
        return 1.0 - t * t * (3.0 - 2.0 * t)
    return 1.0 - t


# --------------------------------------------------------------------------------------------------- sculpt_mask
_SMASK_KEYS = {"object", "mode", "combine", "invert", "value", "group", "vertex", "near", "by", "radius", "falloff",
               "filter", "iterations", "steps", "boundary", "cavity", "minWeight"}
_SMASK_MODES = ("FROM_CAVITY", "FROM_BOUNDARY", "FROM_VERTEX_GROUP", "TO_VERTEX_GROUP", "EXPAND", "FILTER", "INVERT",
                "CLEAR", "SET", "READ")
_SMASK_COMBINE = ("REPLACE", "ADD", "SUBTRACT", "MULTIPLY", "MAX", "MIN")
_SMASK_FILTERS = ("SMOOTH", "SHARPEN", "GROW", "SHRINK", "CONTRAST_INCREASE", "CONTRAST_DECREASE")


def op_sculpt_mask(params):
    """Make, change or read an object's sculpt mask by rule - computed in mesh space, the same on every build.

    WHY (second 1.3 audit, 2026-10-25): sculpt_stroke masks only by vertex group or face sets; the cavity mask protects
    crevices before an inflate or a smooth, and the mask is what mask_extract extracts. Blender's own mask operators
    (sculpt.mask_from_cavity, mask_from_boundary from 4.4, mask_filter, expand - 5.2 bl_ui/properties_paint_common.py
    :1490, space_view3d.py:3938, :3960) need a sculpt session; these rules are written to the mask itself.

    params:
      object (str)       a mesh, not in edit mode (sculpt mode is left and re-entered around the write)
      mode               FROM_CAVITY | FROM_BOUNDARY | FROM_VERTEX_GROUP | TO_VERTEX_GROUP | EXPAND | FILTER | INVERT |
                         CLEAR | SET | READ
      combine            how a FROM_* / EXPAND result meets the mask there: REPLACE (default), ADD, SUBTRACT, MULTIPLY,
                         MAX, MIN
      invert (bool)      invert the computed mask before it is combined
      FROM_CAVITY        cavity {blurIterations (default 2), contrast (default 1)}: 1 in crevices, from the Dirty Vertex
                         Colors measure (the angle between a vertex's normal and its neighbours), normalized
      FROM_BOUNDARY      boundary MESH (default; open edges) or FACE_SETS (where face sets meet), steps (edge rings,
                         default 1), falloff LINEAR (default) | CONSTANT | SMOOTH
      FROM_VERTEX_GROUP / TO_VERTEX_GROUP   group: the vertex group read, or written (made when missing; minWeight,
                         default 0, leaves vertices below it out of the group)
      EXPAND             from vertex (index) or near ([x, y, z] world), by TOPOLOGY (edge steps, default) or GEODESIC
                         (world distance along edges), radius (steps or meters), falloff (default LINEAR)
      FILTER             filter SMOOTH | SHARPEN | GROW | SHRINK | CONTRAST_INCREASE | CONTRAST_DECREASE, iterations
                         (default 1)
      SET                value (0-1) everywhere; CLEAR is SET 0; INVERT is 1 - mask
      READ               the mask as it is, nothing written
    Reply: before and after - vertices, masked (0.5 or more), partly, min, max, mean - and storage (vertex_paint_masks on
    3.6, the .sculpt_mask attribute from 4.x).
    """
    reject_unknown(params, _SMASK_KEYS, "sculpt_mask")
    tail = "NOTHING was changed."
    mode = str(take(params, "mode", required=True, kind=str)).upper()
    if mode not in _SMASK_MODES:
        raise MifOpError("mode is one of %s, got %r. %s" % (", ".join(_SMASK_MODES), mode, tail))
    combine = str(take(params, "combine", default="REPLACE", kind=str)).upper()
    if combine not in _SMASK_COMBINE:
        raise MifOpError("combine is one of %s, got %r. %s" % (", ".join(_SMASK_COMBINE), combine, tail))
    inv = take_bool(params, "invert", default=False)
    obj = _mesh_object(take(params, "object", required=True, kind=str), "mask")
    me = obj.data
    if not len(me.vertices):
        raise MifOpError("'%s' has no vertices. %s" % (obj.name, tail))
    before, storage = _mask_read(me)
    out = {"object": obj.name, "mode": mode, "before": _mask_stats(before)}
    if mode == "READ":
        out.update(storage=storage, after=out["before"])
        if storage is None:
            out["note"] = "this mesh has no mask yet - every vertex reads 0"
        return out
    require_editable(obj, "mask")
    nb = _neighbors(me)
    new = None
    if mode in ("CLEAR", "SET", "INVERT"):
        if mode == "SET":
            val = finite_float(params.get("value"), "value") if params.get("value") is not None else None
            if val is None or not 0.0 <= val <= 1.0:
                raise MifOpError("SET needs value 0-1. %s" % tail)
            new = [val] * len(before)
        else:
            new = [0.0] * len(before) if mode == "CLEAR" else [1.0 - v for v in before]
        combine = "REPLACE"
    elif mode == "FROM_CAVITY":
        cav = params.get("cavity") or {}
        if not isinstance(cav, dict) or set(cav) - {"blurIterations", "contrast"}:
            raise MifOpError("cavity is {blurIterations, contrast}, got %r. %s" % (cav, tail))
        blur = take_int(cav, "blurIterations", default=2)
        contrast = take_float(cav, "contrast", default=1.0)
        if not 0 <= blur <= 40 or not 0.1 <= contrast <= 10.0:
            raise MifOpError("cavity.blurIterations is 0-40 and contrast 0.1-10. %s" % tail)
        from .ops_kit import _dirt_tones                          # noqa: PLC0415 - one dirt measure, ops_kit's
        tones = _dirt_tones(me, 1.0, blur, math.pi, 0.0, False, True)
        new = [min(1.0, max(0.0, (1.0 - t - 0.5) * contrast + 0.5)) for t in tones] if contrast != 1.0 else \
            [1.0 - t for t in tones]
    elif mode == "FROM_BOUNDARY":
        kind = str(take(params, "boundary", default="MESH", kind=str)).upper()
        steps = take_int(params, "steps", default=1)
        fall = str(take(params, "falloff", default="LINEAR", kind=str)).upper()
        if kind not in ("MESH", "FACE_SETS") or not 1 <= steps <= 1000 or fall not in ("LINEAR", "CONSTANT", "SMOOTH"):
            raise MifOpError("FROM_BOUNDARY: boundary MESH | FACE_SETS, steps 1-1000, falloff LINEAR | CONSTANT | "
                             "SMOOTH. %s" % tail)
        if kind == "MESH":
            count = [0] * len(me.edges)
            for lp in me.loops:
                count[lp.edge_index] += 1
            seeds = {v for i, e in enumerate(me.edges) if count[i] == 1 for v in e.vertices}
        else:
            fs = me.attributes.get(_FACE_SET_ATTR)
            if fs is None:
                raise MifOpError("'%s' has no face sets (%s). %s" % (obj.name, _FACE_SET_ATTR, tail))
            per_v = [set() for _ in me.vertices]
            for p in me.polygons:
                for v in p.vertices:
                    per_v[v].add(fs.data[p.index].value)
            seeds = {i for i, s in enumerate(per_v) if len(s) > 1}
        if not seeds:
            raise MifOpError("'%s' has no %s boundary to start from. %s"
                             % (obj.name, "open" if kind == "MESH" else "face-set", tail))
        dist = [None] * len(me.vertices)
        front = list(seeds)
        for s in seeds:
            dist[s] = 0
        while front:
            nxt = []
            for v in front:
                for w in nb[v]:
                    if dist[w] is None:
                        dist[w] = dist[v] + 1
                        nxt.append(w)
            front = nxt
        new = [0.0 if d is None or d > steps else _falloff(d / float(steps + 1) if fall != "CONSTANT" else 0.0, fall)
               for d in dist]
        out["boundaryVertices"] = len(seeds)
    elif mode in ("FROM_VERTEX_GROUP", "TO_VERTEX_GROUP"):
        gname = take(params, "group", required=True, kind=str)
        g = obj.vertex_groups.get(gname)
        if mode == "FROM_VERTEX_GROUP":
            if g is None:
                raise MifOpError("'%s' has no vertex group %r (it has %s). %s"
                                 % (obj.name, gname, ", ".join(x.name for x in obj.vertex_groups) or "none", tail))
            new = [next((x.weight for x in v.groups if x.group == g.index), 0.0) for v in me.vertices]
        else:
            low = take_float(params, "minWeight", default=0.0)
            vals = [1.0 - v for v in before] if inv else before
            if g is None:
                g = obj.vertex_groups.new(name=gname)
            g.remove(list(range(len(me.vertices))))
            for i, w in enumerate(vals):
                if w > low:
                    g.add([i], w, "REPLACE")
            out.update(group=g.name, groupVertices=sum(1 for w in vals if w > low), after=out["before"],
                       storage=storage)
            out.update(shared_data_note(obj))
            return out
    elif mode == "EXPAND":
        by = str(take(params, "by", default="TOPOLOGY", kind=str)).upper()
        if by not in ("TOPOLOGY", "GEODESIC"):
            raise MifOpError("by is TOPOLOGY or GEODESIC, got %r. %s" % (by, tail))
        radius = take_float(params, "radius", required=True)
        fall = str(take(params, "falloff", default="LINEAR", kind=str)).upper()
        if not radius > 0 or fall not in ("LINEAR", "CONSTANT", "SMOOTH"):
            raise MifOpError("EXPAND needs radius above 0 and falloff LINEAR | CONSTANT | SMOOTH. %s" % tail)
        start = _start_vertex(obj, params, tail)
        dist = _distances(me, obj.matrix_world, start, nb, by)
        new = [0.0 if d is None or d > radius else _falloff(d / radius, fall) for d in dist]
        out.update(startVertex=start, by=by, radius=radius)
    elif mode == "FILTER":
        kind = str(take(params, "filter", required=True, kind=str)).upper()
        its = take_int(params, "iterations", default=1)
        if kind not in _SMASK_FILTERS or not 1 <= its <= 100:
            raise MifOpError("filter is one of %s and iterations 1-100. %s" % (", ".join(_SMASK_FILTERS), tail))
        cur = list(before)
        for _ in range(its):
            nxt = list(cur)
            for i, ns in enumerate(nb):
                if not ns:
                    continue
                vals = [cur[j] for j in ns]
                if kind == "SMOOTH":
                    nxt[i] = (cur[i] + sum(vals)) / (len(vals) + 1)
                elif kind == "SHARPEN":
                    nxt[i] = cur[i] + (cur[i] - sum(vals) / len(vals))
                elif kind == "GROW":
                    nxt[i] = max([cur[i]] + vals)
                elif kind == "SHRINK":
                    nxt[i] = min([cur[i]] + vals)
                elif kind == "CONTRAST_INCREASE":
                    nxt[i] = (cur[i] - 0.5) * 1.1 + 0.5
                else:
                    nxt[i] = (cur[i] - 0.5) * 0.9 + 0.5
                nxt[i] = min(1.0, max(0.0, nxt[i]))
            cur = nxt
        new = cur
        combine = "REPLACE"
        out.update(filter=kind, iterations=its)
    if inv and mode not in ("INVERT", "FILTER", "CLEAR", "SET"):
        new = [1.0 - v for v in new]
    ops = {"REPLACE": lambda a, b: b, "ADD": lambda a, b: a + b, "SUBTRACT": lambda a, b: a - b,
           "MULTIPLY": lambda a, b: a * b, "MAX": max, "MIN": min}
    result = [min(1.0, max(0.0, ops[combine](a, b))) for a, b in zip(before, new)]
    with _ObjectMode(obj):
        where = _mask_write(me, result)
    after, _s = _mask_read(me)
    out.update(combine=combine, invert=inv, storage=where, after=_mask_stats(after),
               readBackMatches=max(abs(a - b) for a, b in zip(after, result)) <= 1e-5)
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------------- mask_extract
_MEXT_KEYS = {"object", "mode", "threshold", "addBoundaryLoop", "smoothIterations", "solidify", "thickness",
              "shrinkwrap", "fillHoles", "newObject", "name"}


def op_mask_extract(params):
    """Extract an object's masked region into a new object, or slice it out - one call across the 5.0 rename.

    WHY (second 1.3 audit, 2026-10-25): armor plates, a helmet shell or a strap from a body sculpt. Blender's operators
    are mesh.paint_mask_extract / paint_mask_slice up to 4.4 and sculpt.paint_mask_extract / paint_mask_slice from 5.0
    (the earlier record that they were gone in 5.0 was wrong - renamed); both MEASURED to run headless (module docstring).

    params:
      object (str)            a mesh with a sculpt mask (sculpt_mask makes one)
      mode                    EXTRACT (default) or SLICE
      threshold (0-1)         the mask value that counts (default 0.5)
      EXTRACT: addBoundaryLoop (default true), smoothIterations (default 4), solidify (default true) with thickness
                              (the Solidify modifier's, default Blender's), shrinkwrap (default true: the result is
                              shrink-wrapped back onto the source), name (the new object's)
      SLICE:   fillHoles (default true), newObject (default true: the sliced part also becomes an object)
    Reply: the operator that ran, the new object's counts and modifiers, the source's counts before and after.
    """
    reject_unknown(params, _MEXT_KEYS, "mask_extract")
    tail = "NOTHING was changed."
    mode = str(take(params, "mode", default="EXTRACT", kind=str)).upper()
    if mode not in ("EXTRACT", "SLICE"):
        raise MifOpError("mode is EXTRACT or SLICE, got %r. %s" % (mode, tail))
    thr = take_float(params, "threshold", default=0.5)
    if not 0.0 <= thr <= 1.0:
        raise MifOpError("threshold is 0-1, got %g. %s" % (thr, tail))
    ext_keys = ("addBoundaryLoop", "smoothIterations", "solidify", "thickness", "shrinkwrap", "name")
    sl_keys = ("fillHoles", "newObject")
    wrong = [k for k in (sl_keys if mode == "EXTRACT" else ext_keys) if params.get(k) is not None]
    if wrong:
        raise MifOpError("%s belong to %s, not %s. %s" % (", ".join(wrong), "SLICE" if mode == "EXTRACT" else "EXTRACT",
                                                         mode, tail))
    obj = _mesh_object(take(params, "object", required=True, kind=str), "extract")
    require_editable(obj, "extract from")
    me = obj.data
    mask, storage = _mask_read(me)
    if storage is None or not any(v >= thr for v in mask):
        raise MifOpError("'%s' has no vertex masked at %g or more - sculpt_mask makes the mask. %s" % (obj.name, thr, tail))
    # EVERY VALUE BEFORE THE OPERATOR (2026-10-28, found reading and measured by the version matrix): thickness was read
    # after the extract had made its object, so a bad thickness refused and left that object behind; name went through
    # str(), so a mapping or a list named it "{'mif': ...}"; the operator's own flags were read inside its call.
    if mode == "EXTRACT":
        op_kw = {"add_boundary_loop": take_bool(params, "addBoundaryLoop", default=True),
                 "smooth_iterations": take_int(params, "smoothIterations", default=4),
                 "apply_shrinkwrap": take_bool(params, "shrinkwrap", default=True),
                 "add_solidify": take_bool(params, "solidify", default=True)}
    else:
        op_kw = {"fill_holes": take_bool(params, "fillHoles", default=True),
                 "new_object": take_bool(params, "newObject", default=True)}
    thickness = take_float(params, "thickness") if params.get("thickness") is not None else None
    new_name = take(params, "name", kind=str)
    renamed = bpy.app.version >= (5, 0)
    module = bpy.ops.sculpt if renamed else bpy.ops.mesh
    opname = ("sculpt." if renamed else "mesh.") + ("paint_mask_extract" if mode == "EXTRACT"
                                                                       else "paint_mask_slice")
    counts0 = {"vertices": len(me.vertices), "faces": len(me.polygons)}
    snap = selection_snapshot()
    mode0 = obj.mode
    before = set(bpy.data.objects.keys())
    try:
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="SCULPT")
        if mode == "EXTRACT":
            r = module.paint_mask_extract(mask_threshold=thr, **op_kw)
        else:
            r = module.paint_mask_slice(mask_threshold=thr, **op_kw)
    finally:
        try:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode="OBJECT" if mode0 == "OBJECT" else mode0)
        except RuntimeError:
            pass
        selection_restore(snap)
    if "FINISHED" not in r:
        raise MifOpError("%s answered %s. %s" % (opname, sorted(r), tail))
    made = [bpy.data.objects[n] for n in sorted(set(bpy.data.objects.keys()) - before)]
    out = {"object": obj.name, "mode": mode, "operator": opname, "threshold": thr, "maskStorage": storage,
           "source": {"before": counts0, "after": {"vertices": len(obj.data.vertices), "faces": len(obj.data.polygons)}}}
    rows = []
    for o in made:
        if new_name and o.type == "MESH" and len(made) == 1:
            o.name = new_name
        sol = next((m for m in o.modifiers if m.type == "SOLIDIFY"), None)
        if sol is not None and thickness is not None:
            sol.thickness = thickness
        rows.append({"name": o.name, "type": o.type,
                     "vertices": len(o.data.vertices) if o.type == "MESH" else None,
                     "faces": len(o.data.polygons) if o.type == "MESH" else None,
                     "modifiers": [m.type for m in o.modifiers],
                     "thickness": round(sol.thickness, 6) if sol is not None else None})
    out["made"] = rows
    if mode == "EXTRACT" and not rows:
        raise MifOpError("%s finished but made no object. Nothing more was changed." % opname)
    return out


# --------------------------------------------------------------------------------------------------- mesh_filter
_MFILTER_KEYS = {"object", "filter", "strength", "iterations", "useMask", "vertexGroup", "center", "seed",
                 "shapePreservation", "currentVertex"}
_MFILTERS = ("SMOOTH", "SURFACE_SMOOTH", "INFLATE", "SCALE", "SPHERE", "RANDOM", "RELAX", "RELAX_FACE_SETS",
             "SHARPEN", "ENHANCE_DETAILS", "ERASE_DISPLACEMENT")


def op_mesh_filter(params):
    """A whole-mesh sculpt filter, computed in mesh space: SMOOTH, SURFACE_SMOOTH, INFLATE, SCALE, SPHERE, RANDOM, RELAX,
    RELAX_FACE_SETS, SHARPEN, ENHANCE_DETAILS.

    WHY NATIVE: sculpt.mesh_filter segfaults headless on 4.2.23 and 5.2.2 (ops_topology.py cloth_filter), so each filter
    is written out here as a displacement per vertex and iteration, weighted by 1 - the sculpt mask (useMask, default
    true: a masked vertex is protected, as in sculpt mode) and a vertex group:
      SMOOTH            toward the mean of the neighbours
      SURFACE_SMOOTH    the HC Laplacian (shapePreservation alpha, currentVertex beta, Blender's names and 0.5 / 0.5)
      INFLATE           along the vertex normal, strength x the mean edge length
      SCALE / SPHERE    out from center (default the bounding box's) / toward the sphere of the mean radius
      RANDOM            along the normal by a random amount (seed)
      RELAX             the smoothing move with its normal part taken off: spacing evens out, the surface stays
      RELAX_FACE_SETS   RELAX on the vertices where face sets meet only
      SHARPEN           away from the mean of the neighbours
      ENHANCE_DETAILS   away from a smoothed copy - detail amplified
    ERASE_DISPLACEMENT is refused: it erases toward a Multires level, which this mesh-space filter does not have.

    params: object, filter, strength (0-1, default 0.5), iterations (1-200, default 1), useMask, vertexGroup, center
    ([x, y, z] local), seed. Shape keys are refused (the keys would pull the base back). Reply: verticesMoved,
    maxDisplacement (world units), the world box before and after.
    """
    import random
    reject_unknown(params, _MFILTER_KEYS, "mesh_filter")
    tail = "NOTHING was changed."
    kind = str(take(params, "filter", required=True, kind=str)).upper()
    if kind not in _MFILTERS:
        raise MifOpError("filter is one of %s, got %r. %s" % (", ".join(_MFILTERS), kind, tail))
    if kind == "ERASE_DISPLACEMENT":
        raise MifOpError("ERASE_DISPLACEMENT erases sculpted detail toward a Multires level, and this filter runs on the "
                         "mesh itself, with no lower level to erase toward. Use a Multires modifier's levels "
                         "(subdivide_multires) instead. %s" % tail)
    strength = take_float(params, "strength", default=0.5)
    its = take_int(params, "iterations", default=1)
    if not 0.0 <= strength <= 1.0 or not 1 <= its <= 200:
        raise MifOpError("strength is 0-1 and iterations 1-200. %s" % tail)
    alpha = take_float(params, "shapePreservation", default=0.5)
    beta = take_float(params, "currentVertex", default=0.5)
    if not 0.0 <= alpha <= 1.0 or not 0.0 <= beta <= 1.0:
        raise MifOpError("shapePreservation and currentVertex are 0-1. %s" % tail)
    obj = _mesh_object(take(params, "object", required=True, kind=str), "filter")
    require_editable(obj, "filter")
    me = obj.data
    if getattr(me, "shape_keys", None) is not None:
        raise MifOpError("'%s' has shape keys, and the filter writes the base mesh only - the keys would pull it back. "
                         "Apply or remove them first. %s" % (obj.name, tail))
    n = len(me.vertices)
    if n < 2 or not len(me.edges):
        raise MifOpError("'%s' has no edges to filter over. %s" % (obj.name, tail))
    nb = _neighbors(me)
    w = [1.0] * n
    if take_bool(params, "useMask", default=True):
        mask, _s = _mask_read(me)
        w = [1.0 - m for m in mask]
    gname = take(params, "vertexGroup", kind=str)
    if gname is not None:
        g = obj.vertex_groups.get(gname)
        if g is None:
            raise MifOpError("'%s' has no vertex group %r. %s" % (obj.name, gname, tail))
        w = [wi * next((x.weight for x in v.groups if x.group == g.index), 0.0) for wi, v in zip(w, me.vertices)]
    if kind == "RELAX_FACE_SETS":
        fs = me.attributes.get(_FACE_SET_ATTR)
        if fs is None:
            raise MifOpError("'%s' has no face sets (%s), so RELAX_FACE_SETS has no border to relax. %s"
                             % (obj.name, _FACE_SET_ATTR, tail))
        per_v = [set() for _ in range(n)]
        for p in me.polygons:
            for v in p.vertices:
                per_v[v].add(fs.data[p.index].value)
        w = [wi if len(s) > 1 else 0.0 for wi, s in zip(w, per_v)]
    if not any(x > 0.0 for x in w):
        raise MifOpError("every vertex is masked (or outside the vertex group), so nothing would move. %s" % tail)
    co = [v.co.copy() for v in me.vertices]
    start = [c.copy() for c in co]
    unit = sum((co[e.vertices[0]] - co[e.vertices[1]]).length for e in me.edges) / len(me.edges)
    lo = Vector([min(c[k] for c in co) for k in range(3)])
    hi = Vector([max(c[k] for c in co) for k in range(3)])
    ctr = Vector(finite_floats(list(params["center"]), "center")) if params.get("center") is not None else (lo + hi) / 2
    radius = sum((c - ctr).length for c in co) / n
    rng = random.Random(take_int(params, "seed", default=1))

    def avg(cs, i):
        ns = nb[i]
        if not ns:
            return cs[i]
        s = Vector()
        for j in ns:
            s += cs[j]
        return s / len(ns)

    def normals(cs):
        acc = [Vector() for _ in range(n)]
        for p in me.polygons:
            vs = list(p.vertices)
            a, b, c = cs[vs[0]], cs[vs[1]], cs[vs[2]]
            fn = (b - a).cross(c - a)
            for v in vs:
                acc[v] += fn
        return [x.normalized() if x.length > 1e-12 else Vector((0.0, 0.0, 0.0)) for x in acc]
    for _ in range(its):
        if kind in ("SMOOTH", "SHARPEN", "RELAX", "RELAX_FACE_SETS"):
            av = [avg(co, i) for i in range(n)]
            nrm = normals(co) if kind.startswith("RELAX") else None
            for i in range(n):
                if w[i] <= 0.0 or not nb[i]:
                    continue
                d = av[i] - co[i]
                if nrm is not None:
                    d = d - nrm[i] * d.dot(nrm[i])
                co[i] = co[i] + d * (strength * w[i]) * (-1.0 if kind == "SHARPEN" else 1.0)
        elif kind == "SURFACE_SMOOTH":
            orig = [c.copy() for c in co]
            p = [orig[i] + (avg(orig, i) - orig[i]) * (strength * w[i]) for i in range(n)]
            b = [p[i] - (start[i] * alpha + orig[i] * (1.0 - alpha)) for i in range(n)]
            for i in range(n):
                if w[i] > 0.0 and nb[i]:
                    co[i] = p[i] - (b[i] * beta + avg(b, i) * (1.0 - beta))
        elif kind in ("INFLATE", "RANDOM"):
            nrm = normals(co)
            for i in range(n):
                if w[i] > 0.0:
                    k = strength * w[i] * unit * (1.0 if kind == "INFLATE" else rng.uniform(-1.0, 1.0))
                    co[i] = co[i] + nrm[i] * k
        elif kind == "SCALE":
            for i in range(n):
                co[i] = ctr + (co[i] - ctr) * (1.0 + 0.1 * strength * w[i])
        elif kind == "SPHERE":
            for i in range(n):
                d = co[i] - ctr
                if d.length > 1e-12:
                    co[i] = co[i] + (ctr + d.normalized() * radius - co[i]) * (strength * w[i])
        elif kind == "ENHANCE_DETAILS":
            sm = [c.copy() for c in co]
            for _k in range(3):
                sm = [avg(sm, i) for i in range(n)]
            for i in range(n):
                co[i] = co[i] + (co[i] - sm[i]) * (strength * w[i])
    mw = obj.matrix_world
    moved = [i for i in range(n) if (co[i] - start[i]).length > 1e-9]
    if not moved:
        raise MifOpError("the filter moved no vertex (a flat area under SMOOTH, or strength 0). %s" % tail)
    worst = max((mw.to_3x3() @ (co[i] - start[i])).length for i in moved)

    def box(cs):
        wpts = [mw @ c for c in cs]
        return {"min": rnd([min(p[k] for p in wpts) for k in range(3)]),
                "max": rnd([max(p[k] for p in wpts) for k in range(3)])}
    with _ObjectMode(obj):
        flat = [x for c in co for x in c]
        me.vertices.foreach_set("co", flat)
        me.update()
    out = {"object": obj.name, "filter": kind, "strength": strength, "iterations": its, "verticesMoved": len(moved),
           "maxDisplacement": round(worst, 6), "boxBefore": box(start), "boxAfter": box(co),
           "masked": sum(1 for x in w if x <= 0.0),
           "native": "mesh-space filter, not Blender's sculpt.mesh_filter (which segfaults headless)"}
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------------- symmetrize_mesh
_SYM_KEYS = {"object", "mode", "direction", "axis", "threshold", "factor", "limit"}
# mesh.symmetrize's NEGATIVE_X copies the -X side onto +X, and bmesh.ops.symmetrize's '-X' does the same (measured
# 2026-10-25 on 5.2.1: a +X corner raised 0.3 came back level; the cut's new center vertices sit where the old edge
# crossed the plane).
_SYM_DIRS = {"NEGATIVE_X": "-X", "POSITIVE_X": "X", "NEGATIVE_Y": "-Y", "POSITIVE_Y": "Y", "NEGATIVE_Z": "-Z",
             "POSITIVE_Z": "Z"}


def _mirror_pairs(co, axis, thr):
    """Mirror partners across the axis plane (local space) within thr: (pairs [(i, j)], centers [i], unmatched [i])."""
    from mathutils.kdtree import KDTree
    kd = KDTree(len(co))
    for i, c in enumerate(co):
        kd.insert(c, i)
    kd.balance()
    pairs, centers, unmatched, taken = [], [], [], set()
    for i, c in enumerate(co):
        if abs(c[axis]) <= thr:
            centers.append(i)
            continue
        if c[axis] < 0 or i in taken:
            continue
        m = c.copy()
        m[axis] = -m[axis]
        hit = kd.find(m)
        if hit[1] is not None and hit[2] <= thr and hit[1] not in taken and co[hit[1]][axis] < 0:
            pairs.append((i, hit[1]))
            taken.update((i, hit[1]))
    unmatched = [i for i in range(len(co)) if i not in taken and abs(co[i][axis]) > thr]
    return pairs, centers, unmatched


def op_symmetrize_mesh(params):
    """Symmetrize a mesh, report where it is not symmetric, or snap near-symmetric vertices to exact symmetry.

    WHY (second 1.3 audit, 2026-10-25): mirror_shape_key, paint_weights mirror and armature X-mirror all assume
    symmetric topology, and on a scan or a sculpt retopology that is only nearly symmetric every mirror silently picks a
    wrong partner. Blender's mesh.symmetrize and mesh.symmetry_snap (5.2 bl_ui/space_view3d.py:4544-4545) are edit-mode
    operators; here SYMMETRIZE is bmesh.ops.symmetrize and REPORT / SNAP pair vertices with a KD-tree, in local space.

    params:
      object (str)     a mesh in object mode
      mode             REPORT (default, reads), SNAP or SYMMETRIZE
      axis             X (default), Y or Z - the mirror plane is the object's own axis plane through its origin
      threshold        the largest distance at which two vertices are partners (default 0.0001 local units); a vertex
                       this close to the plane is a center vertex
      SNAP: factor     0-1 (default 1): how far each pair moves to exact symmetry; center vertices go onto the plane
      SYMMETRIZE: direction NEGATIVE_X (default: the -X side is copied onto +X) ... POSITIVE_Z - topology changes,
                       so shape keys are refused
      limit            unmatched vertices listed (default 100)
    Reply: pairs, centers, unmatched (count and indices) - before and, for a write, after.
    """
    reject_unknown(params, _SYM_KEYS, "symmetrize_mesh")
    tail = "NOTHING was changed."
    mode = str(take(params, "mode", default="REPORT", kind=str)).upper()
    if mode not in ("REPORT", "SNAP", "SYMMETRIZE"):
        raise MifOpError("mode is REPORT, SNAP or SYMMETRIZE, got %r. %s" % (mode, tail))
    axis_name = str(take(params, "axis", default="X", kind=str)).upper()
    if axis_name not in ("X", "Y", "Z"):
        raise MifOpError("axis is X, Y or Z, got %r. %s" % (axis_name, tail))
    axis = "XYZ".index(axis_name)
    thr = take_float(params, "threshold", default=1e-4)
    factor = take_float(params, "factor", default=1.0)
    limit = take_int(params, "limit", default=100)
    if not thr > 0 or not 0.0 <= factor <= 1.0 or limit < 0:
        raise MifOpError("threshold is above 0, factor 0-1 and limit 0 or more. %s" % tail)
    obj = _mesh_object(take(params, "object", required=True, kind=str), "symmetrize")
    me = obj.data
    co = [v.co.copy() for v in me.vertices]
    pairs, centers, unmatched = _mirror_pairs(co, axis, thr)

    def report(p, c, u):
        return {"pairs": len(p), "centers": len(c), "unmatched": len(u), "unmatchedVertices": u[:limit]}
    out = {"object": obj.name, "mode": mode, "axis": axis_name, "threshold": thr, "before": report(pairs, centers,
                                                                                                 unmatched)}
    if mode == "REPORT":
        out["symmetric"] = not unmatched
        return out
    require_editable(obj, "symmetrize")
    if getattr(me, "shape_keys", None) is not None:
        raise MifOpError("'%s' has shape keys; %s writes the base mesh only (and SYMMETRIZE changes topology). Apply or "
                         "remove them first. %s" % (obj.name, mode, tail))
    if mode == "SNAP":
        for i, j in pairs:
            a, b = co[i], co[j]
            mb = b.copy()
            mb[axis] = -mb[axis]
            mid = (a + mb) / 2
            mm = mid.copy()
            mm[axis] = -mm[axis]
            co[i] = a + (mid - a) * factor
            co[j] = b + (mm - b) * factor
        for i in centers:
            c = co[i]
            c[axis] = c[axis] * (1.0 - factor)
        with _ObjectMode(obj):
            me.vertices.foreach_set("co", [x for c in co for x in c])
            me.update()
        co2 = [v.co.copy() for v in me.vertices]
        out["after"] = report(*_mirror_pairs(co2, axis, thr))
        out["snapped"] = {"pairs": len(pairs), "centers": len(centers), "factor": factor}
    else:
        d = str(take(params, "direction", default="NEGATIVE_" + axis_name, kind=str)).upper()
        if d not in _SYM_DIRS or d[-1] != axis_name:
            raise MifOpError("direction is NEGATIVE_%s or POSITIVE_%s for axis %s, got %r. %s"
                             % (axis_name, axis_name, axis_name, d, tail))
        v0, f0 = len(me.vertices), len(me.polygons)
        bm = bmesh.new()
        try:
            bm.from_mesh(me)
            bmesh.ops.symmetrize(bm, input=bm.verts[:] + bm.edges[:] + bm.faces[:], direction=_SYM_DIRS[d], dist=thr)
            with _ObjectMode(obj):
                bm.to_mesh(me)
                me.update()
        finally:
            bm.free()
        co2 = [v.co.copy() for v in me.vertices]
        out.update(direction=d, after=report(*_mirror_pairs(co2, axis, thr)),
                   counts={"verticesBefore": v0, "verticesAfter": len(me.vertices), "facesBefore": f0,
                           "facesAfter": len(me.polygons)})
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------------- smooth_vertices
_SMV_KEYS = {"object", "vertices", "vertexGroup", "minWeight", "iterations", "factor", "axes", "keepBoundary",
             "laplacian", "preserveVolume"}


def op_smooth_vertices(params):
    """Smooth a region of a mesh's vertices - a list, a vertex group, or all - with bmesh's smooth or Laplacian smooth.

    WHY (second 1.3 audit, 2026-10-25): a pinch after a boolean or a knife cut; relax_loop takes one run of edges and
    clean_mesh's smoothing works only near boundaries. bmesh.ops.smooth_vert / smooth_laplacian_vert are the routes
    mesh.vertices_smooth and vertices_smooth_laplacian take, headless.

    params:
      object (str)          a mesh in object mode
      vertices [indices] | vertexGroup (with minWeight, default 0: every vertex with weight in it) - default all
      iterations (1-500, default 1), factor (0-1, default 0.5)
      axes                  'XYZ' (default) or any of its letters - the axes the smoothing moves along
      keepBoundary (bool)   default true: open-boundary vertices stay put
      laplacian (bool)      default false: smooth_laplacian_vert, with preserveVolume (default true)
    Reply: verticesMoved, maxDisplacement (world units), boundaryKept.
    """
    reject_unknown(params, _SMV_KEYS, "smooth_vertices")
    tail = "NOTHING was changed."
    its = take_int(params, "iterations", default=1)
    factor = take_float(params, "factor", default=0.5)
    axes = str(take(params, "axes", default="XYZ", kind=str)).upper()
    if not 1 <= its <= 500 or not 0.0 <= factor <= 1.0 or not axes or set(axes) - set("XYZ"):
        raise MifOpError("iterations 1-500, factor 0-1, axes letters of XYZ. %s" % tail)
    keep_b = take_bool(params, "keepBoundary", default=True)
    lap = take_bool(params, "laplacian", default=False)
    if params.get("preserveVolume") is not None and not lap:
        raise MifOpError("preserveVolume goes with laplacian:true. %s" % tail)
    obj = _mesh_object(take(params, "object", required=True, kind=str), "smooth")
    require_editable(obj, "smooth")
    me = obj.data
    if getattr(me, "shape_keys", None) is not None:
        raise MifOpError("'%s' has shape keys, and this smooths the base mesh only. %s" % (obj.name, tail))
    n = len(me.vertices)
    pick = None
    if params.get("vertices") is not None:
        raw = params["vertices"]
        if not isinstance(raw, list) or not raw or not all(isinstance(i, int) and not isinstance(i, bool) for i in raw):
            raise MifOpError("vertices is a non-empty list of vertex indices. %s" % tail)
        bad = [i for i in raw if not 0 <= i < n]
        if bad:
            raise MifOpError("vertex indices %s are out of range (%d vertices). %s" % (bad[:10], n, tail))
        pick = set(raw)
    gname = take(params, "vertexGroup", kind=str)
    if gname is not None:
        if pick is not None:
            raise MifOpError("pass vertices OR vertexGroup. %s" % tail)
        g = obj.vertex_groups.get(gname)
        if g is None:
            raise MifOpError("'%s' has no vertex group %r. %s" % (obj.name, gname, tail))
        low = take_float(params, "minWeight", default=0.0)
        pick = {v.index for v in me.vertices for x in v.groups if x.group == g.index and x.weight > low}
        if not pick:
            raise MifOpError("vertex group %r holds no vertex above %g. %s" % (gname, low, tail))
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        start = [v.co.copy() for v in bm.verts]
        verts = [v for v in bm.verts if pick is None or v.index in pick]
        kept = 0
        if keep_b:
            before = len(verts)
            verts = [v for v in verts if not v.is_boundary]
            kept = before - len(verts)
        if not verts:
            raise MifOpError("no vertex is left to smooth (all on the open boundary, and keepBoundary is on). %s" % tail)
        for _ in range(its):
            if lap:
                bmesh.ops.smooth_laplacian_vert(bm, verts=verts, lambda_factor=factor, lambda_border=0.0,
                                                use_x="X" in axes, use_y="Y" in axes, use_z="Z" in axes,
                                                preserve_volume=take_bool(params, "preserveVolume", default=True))
            else:
                bmesh.ops.smooth_vert(bm, verts=verts, factor=factor, use_axis_x="X" in axes, use_axis_y="Y" in axes,
                                      use_axis_z="Z" in axes)
        m3 = obj.matrix_world.to_3x3()
        moved = [v.index for v in bm.verts if (v.co - start[v.index]).length > 1e-9]
        worst = max(((m3 @ (bm.verts[i].co - start[i])).length for i in moved), default=0.0)
        with _ObjectMode(obj):
            bm.to_mesh(me)
            me.update()
    finally:
        bm.free()
    out = {"object": obj.name, "iterations": its, "factor": factor, "axes": axes, "laplacian": lap,
           "verticesChosen": len(pick) if pick is not None else n, "verticesMoved": len(moved),
           "maxDisplacement": round(worst, 6), "boundaryKept": kept}
    if not moved:
        out["warning"] = "no vertex moved - the region is already flat for this smoothing"
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------------- groom_attach
_GROOM_KEYS = {"object", "mode", "surface", "uvMap", "attachMode", "tolerance", "limit"}


def _root_distances(hair, surf):
    """Per curve: (index, world root, distance to the surface's evaluated mesh, nearest world point)."""
    from mathutils.bvhtree import BVHTree
    deps = bpy.context.evaluated_depsgraph_get()
    ev = surf.evaluated_get(deps)
    me = ev.to_mesh()
    try:
        mw = surf.matrix_world
        verts = [mw @ v.co for v in me.vertices]
        tree = BVHTree.FromPolygons(verts, [tuple(p.vertices) for p in me.polygons], epsilon=0.0)
    finally:
        ev.to_mesh_clear()
    cv = hair.data
    pos = cv.attributes["position"]
    hm = hair.matrix_world
    rows = []
    for c in cv.curves:
        root = hm @ Vector(pos.data[c.first_point_index].vector)
        hit, _n, _i, d = tree.find_nearest(root)
        rows.append((c.index, root, d if hit is not None else None, hit))
    return rows


def _groom_report(rows, tol, limit):
    ds = [d for _c, _r, d, _h in rows if d is not None]
    floating = sorted(((d, c) for c, _r, d, _h in rows if d is not None and d > tol), reverse=True)
    return {"curves": len(rows), "floating": len(floating), "tolerance": tol,
            "maxDistance": round(max(ds), 6) if ds else None, "meanDistance": round(sum(ds) / len(ds), 6) if ds else None,
            "worst": [{"curve": c, "distance": round(d, 6)} for d, c in floating[:limit]]}


def op_groom_attach(params):
    """Attach a hair groom (a Curves object) to its surface: set the surface and UV map, snap the roots onto it, or report
    the roots that float off it.

    WHY (second 1.3 audit, 2026-10-25): a groom moved onto another head with fit_mesh or wrap_to_target floats or
    sinks, and the surface binding (Curves.surface, surface_uv_map and the surface_uv_coordinate attribute) is what
    Unreal's groom binding and the Deform Curves on Surface node need. Blender's curves.snap_curves_to_surface (5.2
    bl_ui/space_view3d.py:3901) is MEASURED to run headless in sculpt-curves mode on 4.2.17 and 5.2.1: NEAREST moved
    each curve so its root sits on the surface and wrote surface_uv_coordinate (CURVE, FLOAT2).

    params:
      object (str)      a CURVES (hair) object
      mode              REPORT (default: writes nothing), SET_SURFACE or SNAP
      surface (str)     SET_SURFACE: the mesh to bind to (also accepted on SNAP, which sets it first)
      uvMap (str)       the surface's UV map (default its active one)
      attachMode        SNAP: NEAREST (default: roots to the nearest surface point) or DEFORM (re-attach by the
                        existing surface_uv_coordinate after the surface changed - refused when there is none)
      tolerance         world distance above which a root is floating (default 0.001)
      limit             worst roots listed (default 20)
    Reply: surface, uvMap, hasSurfaceUV, and the roots' distances before (and after a SNAP): curves, floating,
    maxDistance, meanDistance, worst.
    """
    reject_unknown(params, _GROOM_KEYS, "groom_attach")
    tail = "NOTHING was changed."
    mode = str(take(params, "mode", default="REPORT", kind=str)).upper()
    if mode not in ("REPORT", "SET_SURFACE", "SNAP"):
        raise MifOpError("mode is REPORT, SET_SURFACE or SNAP, got %r. %s" % (mode, tail))
    tol = take_float(params, "tolerance", default=0.001)
    limit = take_int(params, "limit", default=20)
    if not tol >= 0 or limit < 0:
        raise MifOpError("tolerance and limit are 0 or more. %s" % tail)
    hair = get_object(take(params, "object", required=True, kind=str))
    if hair.type != "CURVES":
        raise MifOpError("'%s' is a %s, not a CURVES (hair) object. %s" % (hair.name, hair.type, tail))
    cv = hair.data
    sname = take(params, "surface", kind=str)
    surf = get_object(sname, want_mesh=True) if sname else cv.surface
    if mode == "SET_SURFACE" and not sname:
        raise MifOpError("SET_SURFACE needs surface: the mesh to bind to. %s" % tail)
    am = str(take(params, "attachMode", default="NEAREST", kind=str)).upper()
    if mode != "SNAP" and params.get("attachMode") is not None:
        raise MifOpError("attachMode is SNAP's. %s" % tail)
    if am not in ("NEAREST", "DEFORM"):
        raise MifOpError("attachMode is NEAREST or DEFORM, got %r. %s" % (am, tail))
    if surf is None:
        raise MifOpError("'%s' has no surface - pass surface (SET_SURFACE, or SNAP sets it first). %s" % (hair.name, tail))
    if surf.type != "MESH":
        raise MifOpError("the surface '%s' is a %s; a groom attaches to a mesh. %s" % (surf.name, surf.type, tail))
    uv = take(params, "uvMap", kind=str)
    if uv is not None and surf.data.uv_layers.get(uv) is None:
        raise MifOpError("'%s' has no UV map %r (it has %s). %s"
                         % (surf.name, uv, ", ".join(u.name for u in surf.data.uv_layers) or "none", tail))
    if not len(cv.curves):
        raise MifOpError("'%s' has no curves. %s" % (hair.name, tail))
    before = _groom_report(_root_distances(hair, surf), tol, limit)
    out = {"object": hair.name, "mode": mode, "before": before}
    if mode == "REPORT":
        out.update(surface=cv.surface.name if cv.surface else None, uvMap=cv.surface_uv_map or None,
                   hasSurfaceUV="surface_uv_coordinate" in cv.attributes, reportedAgainst=surf.name)
        return out
    require_editable(hair, "attach")
    if sname or mode == "SET_SURFACE":
        cv.surface = surf
    if uv is not None or not cv.surface_uv_map:
        cv.surface_uv_map = uv or (surf.data.uv_layers.active.name if surf.data.uv_layers.active else "")
    if mode == "SNAP":
        if am == "DEFORM" and "surface_uv_coordinate" not in cv.attributes:
            raise MifOpError("DEFORM re-attaches by surface_uv_coordinate and '%s' has none - SNAP NEAREST first. The "
                             "surface and UV map were set; nothing moved." % hair.name)
        if am == "DEFORM" and not cv.surface_uv_map:
            raise MifOpError("DEFORM needs a UV map on the surface. The surface was set; nothing moved.")
        snap = selection_snapshot()
        mode0 = hair.mode
        try:
            for o in bpy.context.view_layer.objects:
                o.select_set(False)
            bpy.context.view_layer.objects.active = hair
            hair.select_set(True)
            bpy.ops.object.mode_set(mode="SCULPT_CURVES")
            r = bpy.ops.curves.snap_curves_to_surface(attach_mode=am)
        finally:
            try:
                bpy.context.view_layer.objects.active = hair
                bpy.ops.object.mode_set(mode=mode0)
            except RuntimeError:
                pass
            selection_restore(snap)
        if "FINISHED" not in r:
            raise MifOpError("curves.snap_curves_to_surface answered %s. The surface was set." % sorted(r))
        out["after"] = _groom_report(_root_distances(hair, surf), tol, limit)
        out["attachMode"] = am
        out["operator"] = "curves.snap_curves_to_surface"
    out.update(surface=cv.surface.name if cv.surface else None, uvMap=cv.surface_uv_map or None,
               hasSurfaceUV="surface_uv_coordinate" in cv.attributes)
    return out


# ------------------------------------------------------------------------------------------------- groom_stroke
# HAIR SCULPTING, NATIVE (second 1.3 audit, 2026-10-26): sculpt_curves.brush_stroke is keymap-only and region-bound, the
# same wall sculpt_stroke met on meshes, so the strokes are done here on the Curves points: the same on every build and
# repeatable. Roots never move (but SLIDE, whose job is moving them). A strand is in the stroke when its root is within
# radius of center (every strand without center); its weight is the falloff of that distance times strength.
# CUT, PUFF, PINCH, SLIDE AND DENSITY (1.3 wave 3, 2026-10-26), the rest of Blender's curves sculpt brushes. CUT and
# DENSITY change how many points or strands there are, which the data API can do only on the builds that have it: read
# against the bundled binaries, Curves.add_curves is on 4.2, 4.4 and 5.0 (not 3.6) and Curves.resize_curves and
# remove_curves on 4.4 and 5.0 (not 3.6 or 4.2) - each refused by name where it is missing. PUFF, SLIDE and DENSITY's
# new roots need the hair's surface (Curves.surface), as groom_attach reads it.
_GS_KEYS = {"object", "name", "mode", "center", "radius", "falloff", "strength", "vector", "factor", "iterations",
            "length", "count", "seed"}
_GS_MODES = ("COMB", "LENGTH", "SMOOTH", "CUT", "PUFF", "PINCH", "SLIDE", "DENSITY")
_GS_KEY_MODES = {"vector": ("COMB", "SLIDE"), "factor": ("LENGTH",), "iterations": ("SMOOTH",), "length": ("CUT",),
                 "count": ("DENSITY",), "seed": ("DENSITY",)}


def _gs_weight(d, radius, falloff):
    if radius is None:
        return 1.0
    t = min(1.0, d / radius) if radius > 0 else 1.0
    if falloff == "CONSTANT":
        return 1.0
    if falloff == "LINEAR":
        return 1.0 - t
    return 1.0 - (3 * t * t - 2 * t * t * t)                     # SMOOTH: smoothstep down to 0 at the edge


class _GsSurface:
    """The hair's surface in world space, triangulated: nearest point, its normal and its UV (surface_uv_map, else the
    active map). free() after use."""

    def __init__(self, cu, mode, tail):
        from mathutils.bvhtree import BVHTree                      # noqa: PLC0415
        surf = getattr(cu, "surface", None)
        if surf is None or surf.type != "MESH":
            raise MifOpError("%s needs the hair's surface mesh (Curves.surface - set by create_curves surface or "
                             "groom_attach), and this hair has %s. %s"
                             % (mode, "none" if surf is None else "a %s" % surf.type, tail))
        dg = bpy.context.evaluated_depsgraph_get()
        ev = surf.evaluated_get(dg)
        me = ev.to_mesh()
        self.bm = bmesh.new()
        try:
            self.bm.from_mesh(me)
        finally:
            ev.to_mesh_clear()
        self.bm.transform(surf.matrix_world)
        bmesh.ops.triangulate(self.bm, faces=list(self.bm.faces))
        self.bm.normal_update()
        self.bm.faces.ensure_lookup_table()
        if not self.bm.faces:
            self.bm.free()
            raise MifOpError("the surface '%s' has no faces. %s" % (surf.name, tail))
        layers = self.bm.loops.layers.uv
        want = getattr(cu, "surface_uv_map", "") or ""
        self.uv = layers.get(want) if want else layers.active
        self.bvh = BVHTree.FromBMesh(self.bm)
        self.name = surf.name

    def nearest(self, p):
        loc, nrm, idx, _d = self.bvh.find_nearest(p)
        return loc, nrm, idx

    def uv_at(self, loc, idx):
        if self.uv is None or idx is None:
            return None
        from mathutils.interpolate import poly_3d_calc             # noqa: PLC0415
        f = self.bm.faces[idx]
        w = poly_3d_calc([v.co for v in f.verts], loc)
        u = sum(wi * lp[self.uv].uv[0] for wi, lp in zip(w, f.loops))
        v = sum(wi * lp[self.uv].uv[1] for wi, lp in zip(w, f.loops))
        return (u, v)

    def free(self):
        self.bm.free()


def _gs_keep_lengths(new, seg):
    """Walk from the root and put each segment back to its length, keeping its new direction."""
    out = [new[0]]
    for i in range(1, len(new)):
        d = new[i] - out[-1]
        out.append(out[-1] + (d.normalized() * seg[i - 1] if d.length > 1e-12 else (new[i] - new[i - 1])))
    return out


def _gs_need(cu, fn, mode, tail):
    if fn not in cu.bl_rna.functions:
        raise MifOpError("%s needs Curves.%s, which this Blender (%s) does not have (read on the bundled builds: "
                         "add_curves from 4.2, resize_curves and remove_curves from 4.4). %s"
                         % (mode, fn, bpy.app.version_string, tail))


def op_groom_stroke(params):
    """Stroke a hair Curves object's strands natively on its points: comb, lengthen or shorten, smooth, cut, puff, pinch,
    slide, or add and remove strands.

    params:
      object (str, required)   a CURVES object (hair)
      mode                     COMB | LENGTH | SMOOTH | CUT | PUFF | PINCH | SLIDE | DENSITY
      center [x, y, z], radius (world metres)   the strands whose ROOTS lie within radius are stroked (both or neither;
                               neither strokes every strand). PINCH needs them.
      falloff                  SMOOTH (default) | LINEAR | CONSTANT, over the radius
      strength (0-1)           default 1
      vector [x, y, z]         COMB: world metres the TIP moves at full weight; each point moves by the square of its
                               share of the strand's length from the root, so the root stays and the strand bends.
                               SLIDE: world metres the ROOT moves at full weight, then put back on the surface; the
                               strand keeps its shape and its surface_uv_coordinate follows
      factor (> 0)             LENGTH: each strand's length times factor (1 + (factor - 1) x weight), every segment scaled
      iterations (1-100)       SMOOTH: Laplacian passes on the inner points, blended by weight; roots and tips held
      length (world m, > 0)    CUT: a strand longer than this is trimmed toward it by weight - points past the cut are
                               removed and the last one placed on it (Curves.resize_curves; 4.4 and later)
      PUFF                     raises strands toward the surface normal at the root by weight, lengths kept (needs the
                               surface)
      PINCH                    pulls strands toward the line through center, the tip most and the root not, lengths kept
      count (int), seed        DENSITY: count > 0 adds that many strands on the surface (in the region), each a copy of
                               the nearest strand's shape (Curves.add_curves; 4.2 and later; radius and
                               surface_uv_coordinate written when the hair has them); count < 0 removes that many of the
                               stroked strands, chosen by seed (Curves.remove_curves; 4.4 and later)
    Reply: strands stroked, mean length before and after, the largest root move (0 but SLIDE), and the tip moves; CUT
    and DENSITY add the point and strand counts before and after.
    """
    reject_unknown(params, _GS_KEYS, "groom_stroke")
    tail = "NOTHING was changed."
    obj = get_object(take(params, "object", "name", required=True, kind=str))
    if obj.type != "CURVES":
        raise MifOpError("'%s' is a %s; groom_stroke works on a hair CURVES object. %s" % (obj.name, obj.type, tail))
    require_editable(obj, "groom")
    mode = str(take(params, "mode", required=True, kind=str)).upper()
    if mode not in _GS_MODES:
        raise MifOpError("mode is %s, got %r. %s" % (", ".join(_GS_MODES), mode, tail))
    for k, modes in _GS_KEY_MODES.items():
        if params.get(k) is not None and mode not in modes:
            raise MifOpError("%s belongs to %s, not %s. %s" % (k, " / ".join(modes), mode, tail))
    has_c, has_r = params.get("center") is not None, params.get("radius") is not None
    if has_c != has_r:
        raise MifOpError("center and radius go together - give both to stroke a region, or neither for every strand. %s"
                         % tail)
    if mode == "PINCH" and not has_c:
        raise MifOpError("PINCH pulls strands toward center; give center and radius. %s" % tail)
    center = Vector(finite_floats(list(params["center"])[:3], "center")) if has_c else None
    radius = finite_float(params["radius"], "radius") if has_r else None
    if radius is not None and not radius > 0:
        raise MifOpError("radius is above 0. %s" % tail)
    falloff = str(params.get("falloff") or "SMOOTH").upper()
    if falloff not in ("SMOOTH", "LINEAR", "CONSTANT"):
        raise MifOpError("falloff is SMOOTH, LINEAR or CONSTANT. %s" % tail)
    strength = finite_float(params.get("strength", 1.0), "strength")
    if not 0.0 <= strength <= 1.0:
        raise MifOpError("strength is 0-1. %s" % tail)
    mw = obj.matrix_world
    mw_inv = mw.inverted()
    inv3 = mw.to_3x3().inverted()
    vec = factor = iters = cut_len = count = None
    seed = 0
    if mode in ("COMB", "SLIDE"):
        if params.get("vector") is None:
            raise MifOpError("%s needs vector [x, y, z] (world metres the %s moves). %s"
                             % (mode, "tip" if mode == "COMB" else "root", tail))
        vec = Vector(finite_floats(list(params["vector"])[:3], "vector"))
        if mode == "COMB":
            vec = inv3 @ vec
    elif mode == "LENGTH":
        factor = finite_float(params.get("factor", 1.0), "factor")
        if not factor > 0:
            raise MifOpError("factor is above 0. %s" % tail)
    elif mode == "SMOOTH":
        iters = take_int(params, "iterations", default=3)
        if not 1 <= iters <= 100:
            raise MifOpError("iterations is 1-100. %s" % tail)
    elif mode == "CUT":
        if params.get("length") is None:
            raise MifOpError("CUT needs length (world metres a strand is trimmed toward). %s" % tail)
        cut_len = finite_float(params["length"], "length")
        if not cut_len > 0:
            raise MifOpError("length is above 0. %s" % tail)
    elif mode == "DENSITY":
        count = take_int(params, "count", required=True)
        if count == 0 or abs(count) > 10000:
            raise MifOpError("count is -10000 to 10000 and not 0 (above 0 adds strands, below removes them). %s" % tail)
        seed = take_int(params, "seed", default=0)
        if seed < 0:
            raise MifOpError("seed is 0 or more. %s" % tail)
    cu = obj.data
    if mode == "CUT":
        _gs_need(cu, "resize_curves", mode, tail)
    elif mode == "DENSITY":
        _gs_need(cu, "add_curves" if count > 0 else "remove_curves", mode, tail)
    surface = _GsSurface(cu, mode, tail) if mode in ("PUFF", "SLIDE") or (mode == "DENSITY" and count > 0) else None
    try:
        return _groom_stroke_apply(obj, cu, mode, center, radius, falloff, strength, mw, mw_inv, inv3, vec, factor,
                                   iters, cut_len, count, seed, surface, tail)
    finally:
        if surface is not None:
            surface.free()


def _groom_stroke_apply(obj, cu, mode, center, radius, falloff, strength, mw, mw_inv, inv3, vec, factor, iters,
                        cut_len, count, seed, surface, tail):
    pos = [Vector(p.position) for p in cu.points]

    def strand_len(idx):
        return sum((pos[idx[i + 1]] - pos[idx[i]]).length for i in range(len(idx) - 1))
    strands = [list(range(c.first_point_index, c.first_point_index + c.points_length)) for c in cu.curves]
    before = [strand_len(s) for s in strands]
    roots0 = [Vector(pos[s[0]]) for s in strands if s]
    tips0 = [Vector(pos[s[-1]]) for s in strands if s]
    points0, curves0 = len(cu.points), len(cu.curves)
    weights = {}
    for ci, s in enumerate(strands):
        if len(s) < 2 and mode != "DENSITY":
            continue
        w = 1.0
        if center is not None:
            d = ((mw @ pos[s[0]]) - center).length
            if d > radius:
                continue
            w = _gs_weight(d, radius, falloff)
        w *= strength
        if w > 0.0:
            weights[ci] = w
    extra = {}
    resized = {}                                                    # CUT: curve index -> its new local points
    for ci, w in weights.items():
        s = strands[ci]
        if mode == "DENSITY":
            break
        seg = [(pos[s[i]] - pos[s[i - 1]]).length for i in range(1, len(s))]
        if mode == "COMB":
            total = strand_len(s) or 1.0
            acc = 0.0
            new = [Vector(pos[s[0]])]
            for i in range(1, len(s)):
                acc += (pos[s[i]] - pos[s[i - 1]]).length
                new.append(pos[s[i]] + vec * (w * (acc / total) ** 2))     # stiff at the root, as a comb bends hair
            for i, v in zip(s, new):
                pos[i] = v
        elif mode == "LENGTH":
            f = 1.0 + (factor - 1.0) * w
            root = Vector(pos[s[0]])
            segs = [pos[s[i]] - pos[s[i - 1]] for i in range(1, len(s))]
            cur = root
            for i, sg in zip(s[1:], segs):
                cur = cur + sg * f
                pos[i] = cur
        elif mode == "SMOOTH":
            for _ in range(iters):
                prev = [Vector(pos[i]) for i in s]
                for j in range(1, len(s) - 1):
                    avg = (prev[j - 1] + prev[j + 1]) * 0.5
                    pos[s[j]] = prev[j].lerp(avg, w)
        elif mode == "CUT":
            world = [mw @ pos[i] for i in s]
            wl = sum((world[i] - world[i - 1]).length for i in range(1, len(world)))
            if wl <= cut_len:
                continue
            target = wl - (wl - cut_len) * w
            acc = 0.0
            keep = [world[0]]
            for i in range(1, len(world)):
                step = (world[i] - world[i - 1]).length
                if acc + step >= target:
                    t = (target - acc) / step if step > 0 else 0.0
                    keep.append(world[i - 1].lerp(world[i], t))
                    break
                acc += step
                keep.append(world[i])
            if len(keep) < 2:
                continue
            resized[ci] = [mw_inv @ p for p in keep]
        elif mode == "PUFF":
            _loc, nrm, _idx = surface.nearest(mw @ pos[s[0]])
            if nrm is None:
                continue
            up = (inv3 @ nrm).normalized()
            acc = 0.0
            target = [Vector(pos[s[0]])]
            for L in seg:
                acc += L
                target.append(pos[s[0]] + up * acc)
            new = [Vector(pos[i]).lerp(t, w) for i, t in zip(s, target)]
            new[0] = Vector(pos[s[0]])
            for i, v in zip(s, _gs_keep_lengths(new, seg)):
                pos[i] = v
        elif mode == "PINCH":
            c_local = mw_inv @ center
            axis = pos[s[-1]] - pos[s[0]]
            axis = axis.normalized() if axis.length > 1e-12 else Vector((0.0, 0.0, 1.0))
            total = sum(seg) or 1.0
            acc = 0.0
            new = [Vector(pos[s[0]])]
            for j in range(1, len(s)):
                acc += seg[j - 1]
                p = pos[s[j]]
                d = c_local - p
                perp = d - axis * d.dot(axis)
                new.append(p + perp * (w * acc / total))
            for i, v in zip(s, _gs_keep_lengths(new, seg)):
                pos[i] = v
        elif mode == "SLIDE":
            root_w = mw @ pos[s[0]]
            loc, _nrm, idx = surface.nearest(root_w + vec * w)
            if loc is None:
                continue
            delta = (mw_inv @ loc) - pos[s[0]]
            for i in s:
                pos[i] = pos[i] + delta
            uv = surface.uv_at(loc, idx)
            if uv is not None:
                extra.setdefault("uv", {})[ci] = uv
    if mode in ("COMB", "LENGTH", "SMOOTH", "PUFF", "PINCH", "SLIDE"):
        for p, v in zip(cu.points, pos):
            p.position = v
        uvs = extra.get("uv")
        attr = cu.attributes.get("surface_uv_coordinate")
        if uvs and attr is not None and attr.domain == "CURVE":
            for ci, uv in uvs.items():
                attr.data[ci].vector = uv
            extra["uvUpdated"] = len(uvs)
    elif mode == "CUT":
        if resized:
            idx = sorted(resized)
            # indices BY KEYWORD: an optional RNA parameter is keyword-only, so 5.2.1 refuses it positionally (and
            # remove_curves' below the same way) - "required parameter "indices" to be a keyword argument".
            cu.resize_curves([len(resized[ci]) for ci in idx], indices=idx)
            for ci in idx:
                first = cu.curves[ci].first_point_index
                for j, v in enumerate(resized[ci]):
                    cu.points[first + j].position = v
        extra["trimmed"] = len(resized)
    else:
        extra.update(_groom_density(obj, cu, strands, pos, weights, count, seed, center, radius, mw, mw_inv, surface,
                                    tail))
    cu.update_tag()
    bpy.context.view_layer.update()
    pos1 = [Vector(p.position) for p in cu.points]
    strands1 = [list(range(c.first_point_index, c.first_point_index + c.points_length)) for c in cu.curves]
    after = [sum((pos1[s[i + 1]] - pos1[s[i]]).length for i in range(len(s) - 1)) for s in strands1]
    out = {"object": obj.name, "mode": mode, "strands": len(strands), "stroked": len(weights) if mode != "CUT"
           else extra.get("trimmed", 0),
           "meanLengthBefore": round(sum(before) / len(before), 6) if before else None,
           "meanLengthAfter": round(sum(after) / len(after), 6) if after else None,
           "units": "local lengths (the object's own units); tip moves in world metres"}
    if mode == "DENSITY":
        out["stroked"] = extra.get("added", 0) or extra.get("removed", 0)
    if len(strands1) == len(strands) or (mode == "DENSITY" and count > 0):
        n = len(roots0)
        roots1 = [Vector(pos1[s[0]]) for s in strands1[:n] if s]
        tips1 = [Vector(pos1[s[-1]]) for s in strands1[:n] if s]
        out["maxRootMove"] = round(max([(mw.to_3x3() @ (a - b)).length for a, b in zip(roots1, roots0)] or [0.0]), 9)
        out["maxTipMove"] = round(max([(mw.to_3x3() @ (a - b)).length for a, b in zip(tips1, tips0)] or [0.0]), 6)
    else:
        out["maxRootMove"] = out["maxTipMove"] = None
    if mode in ("CUT", "DENSITY"):
        out.update(pointsBefore=points0, pointsAfter=len(cu.points), strandsBefore=curves0, strandsAfter=len(cu.curves))
    if mode in ("PUFF", "PINCH"):
        out["lengthsKept"] = True
    for k in ("trimmed", "added", "removed", "uvUpdated", "templateNote", "uvNote"):
        if k in extra:
            out[k] = extra[k]
    if surface is not None:
        out["surface"] = surface.name
    return out


def _groom_density(obj, cu, strands, pos, weights, count, seed, center, radius, mw, mw_inv, surface, tail):
    """DENSITY: add count strands on the surface (copies of the nearest strand's shape) or remove -count stroked ones."""
    import random                                                  # noqa: PLC0415
    rng = random.Random(seed)
    if count < 0:
        pool = sorted(weights)
        if not pool:
            return {"removed": 0}
        gone = sorted(rng.sample(pool, min(-count, len(pool))))
        cu.remove_curves(indices=gone)
        return {"removed": len(gone)}
    live = [s for s in strands if len(s) >= 2]
    if not live:
        raise MifOpError("DENSITY adds copies of the nearest strand's shape, and '%s' has no strand of 2 or more points "
                         "to copy. %s" % (obj.name, tail))
    faces = surface.bm.faces
    cand, areas = [], []
    for f in faces:
        if center is not None and (f.calc_center_median() - center).length > radius + max(
                (v.co - f.calc_center_median()).length for v in f.verts):
            continue
        a = f.calc_area()
        if a > 0:
            cand.append(f)
            areas.append(a)
    if not cand:
        raise MifOpError("no face of the surface '%s' lies in the region. %s" % (surface.name, tail))
    import itertools                                               # noqa: PLC0415
    cum = list(itertools.accumulate(areas))                        # once, not per draw: a dense surface has many faces
    roots = []
    tries = 0
    while len(roots) < count and tries < count * 50:
        tries += 1
        f = rng.choices(cand, cum_weights=cum)[0]
        a, b, c = (v.co for v in list(f.verts)[:3])
        r1, r2 = rng.random(), rng.random()
        s1 = math.sqrt(r1)
        p = a * (1 - s1) + b * (s1 * (1 - r2)) + c * (s1 * r2)
        if center is not None and (p - center).length > radius:
            continue
        roots.append((p, f.index))
    if not roots:
        raise MifOpError("no root could be placed on the surface within the region. %s" % tail)
    roots_w = [(mw @ pos[s[0]], s) for s in live]
    shapes = []
    for p, fidx in roots:
        _rw, tmpl = min(roots_w, key=lambda r: (r[0] - p).length)
        base = mw_inv @ p
        shapes.append(([base + (pos[i] - pos[tmpl[0]]) for i in tmpl], tmpl, surface.uv_at(p, fidx)))
    n0 = len(cu.curves)
    cu.add_curves([len(sh[0]) for sh in shapes])
    rad = cu.attributes.get("radius")
    rad_vals = [d.value for d in rad.data] if rad is not None and rad.domain == "POINT" else None
    uv_attr = cu.attributes.get("surface_uv_coordinate")
    wrote_uv = 0
    for k, (pts, tmpl, uv) in enumerate(shapes):
        c = cu.curves[n0 + k]
        first = c.first_point_index
        for j, v in enumerate(pts):
            cu.points[first + j].position = v
            if rad_vals is not None:
                rad.data[first + j].value = rad_vals[tmpl[j]]
        if uv_attr is not None and uv_attr.domain == "CURVE" and uv is not None:
            uv_attr.data[n0 + k].vector = uv
            wrote_uv += 1
    out = {"added": len(shapes), "templateNote": "each new strand is the nearest existing strand's shape, moved to its "
                                                 "root (not turned to the surface there)"}
    if len(shapes) < count:
        out["templateNote"] += "; %d of %d roots fit the region" % (len(shapes), count)
    if uv_attr is not None:
        out["uvUpdated"] = wrote_uv
    return out

OPS = {
    "groom_stroke": op_groom_stroke,
    "groom_attach": op_groom_attach,
    "sculpt_mask": op_sculpt_mask,
    "mask_extract": op_mask_extract,
    "mesh_filter": op_mesh_filter,
    "symmetrize_mesh": op_symmetrize_mesh,
    "smooth_vertices": op_smooth_vertices,
}
