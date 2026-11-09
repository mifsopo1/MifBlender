"""MifBlender ops: hand-modelling topology edits - extrude, loop cut, edge slide, inset, knife, bridge, dissolve, fill.

THE GAP (spec, "Everyday modeling, sculpt and garment ops", 2026-10-04): the steps every modelling session repeats had
no typed call. bevel_edges and extrude_skirt were the only edits, and the rest was raw Python or nothing. Each op here
is a bmesh operator, so it runs headless like bevel_edges - no edit mode, no selection state, no view.

SHARED CONTRACT, every op in this module:
  * the elements are named by INDEX (from select_faces, select_edges or vertex_info), validated before anything is
    written; a list that names nothing, or an index past the end, is refused with NOTHING changed;
  * the work happens on a scratch bmesh and is written once at the end, so a refusal anywhere leaves the mesh as it was;
  * the reply carries the counts before and after and the INDICES of what was made - read after the write, in the order
    the mesh now holds them - so the next call needs no fresh selection;
  * lengths and offsets are in the mesh's own units (`space` LOCAL, the default) or in world units (`space` WORLD, the
    object's scale and rotation applied to the vector only: a big translation never touches a coordinate - float32 at
    240 m from the origin moves a vertex by micrometres, docs/02_GOTCHAS.md);
  * a mesh with shape keys is refused: these ops write the base mesh, and the keys would pull it back.
"""
from __future__ import annotations

import math

import bmesh
import bpy
from mathutils import Matrix, Vector

from .ops_common import (remove_objects, MifOpError, finite_floats, get_object, mesh_counts, reject_unknown, require_editable, rnd,
                         shared_data_note, take, take_bool, take_float, take_int, _open_collapsed_uvs, triangulate_ngons)


# --------------------------------------------------------------------------------------------- shared helpers

def _target(params, op_name):
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    require_editable(obj, "edit")
    if getattr(obj.data, "shape_keys", None) is not None:
        raise MifOpError("'%s' has shape keys, and %s writes the base mesh only - the keys would pull the edit back. "
                         "Apply or remove them first. NOTHING was changed." % (obj.name, op_name))
    return obj


def _open(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.normal_update()
    return bm


def _commit(bm, obj):
    """Write the scratch bmesh; indices are refreshed first, so an element's .index is its index in the mesh after."""
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()
    bm.to_mesh(obj.data)
    obj.data.update()
    bpy.context.view_layer.update()


def _elements(params, key, table, kind, obj, required=True):
    raw = params.get(key)
    if raw is None:
        if required:
            raise MifOpError("'%s' is required: the %s indices to act on (select_faces, select_edges or vertex_info "
                             "give them). NOTHING was changed." % (key, kind))
        return None
    if not isinstance(raw, (list, tuple)) or not raw:
        raise MifOpError("'%s' must be a non-empty list of %s indices. NOTHING was changed." % (key, kind))
    bad = [i for i in raw if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(table)]
    if bad:
        raise MifOpError("%s %s are not %s indices of '%s' (it has %d). NOTHING was changed."
                         % (key, bad[:10], kind, obj.name, len(table)))
    return [table[i] for i in dict.fromkeys(raw)]


def _vec3(params, key, default=None):
    raw = params.get(key)
    if raw is None:
        return default
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise MifOpError("'%s' must be [x,y,z], got %r. NOTHING was changed." % (key, raw))
    return Vector(finite_floats(list(raw), key))


def _space(params):
    space = str(take(params, "space", default="LOCAL", kind=str)).upper()
    if space not in ("LOCAL", "WORLD"):
        raise MifOpError("space must be LOCAL or WORLD, got %r. NOTHING was changed." % space)
    return space


def _uniform_scale(obj):
    s = [abs(x) for x in obj.matrix_world.to_scale()]
    if max(s) <= 0.0 or max(s) - min(s) > 1e-6 * max(s):
        return None
    return s[0]


def _length(obj, value, space, key):
    """A length in the op's space, as mesh units. WORLD needs a uniform scale: one world length is one mesh length."""
    if space == "LOCAL" or value is None:
        return value
    s = _uniform_scale(obj)
    if s is None:
        raise MifOpError("'%s' has a non-uniform or zero world scale, so %s in WORLD units is a different mesh length "
                         "along each axis. Pass space LOCAL, or apply the scale. NOTHING was changed." % (obj.name, key))
    return value / s


def _direction(obj, vec, space):
    """A world vector (a move, an axis) as a mesh-space vector: the 3x3 part of the inverse only, no translation."""
    if space == "LOCAL" or vec is None:
        return vec
    return obj.matrix_world.to_3x3().inverted_safe() @ vec


def _point(obj, p, space):
    if space == "LOCAL" or p is None:
        return p
    return obj.matrix_world.inverted_safe() @ p


def _idx(items):
    return [x.index for x in items if x.is_valid]


# --------------------------------------------------------------------------------------------- extrude

_EXTRUDE_KEYS = ("object", "name", "elements", "indices", "mode", "offset", "distance", "scale", "rotate", "steps",
                 "space", "flipNormals")


def _median(verts):
    return sum((v.co for v in verts), Vector()) / len(verts)


def op_extrude_mesh(params):
    """Extrude faces, edges or vertices along their normal or a vector, with a scale and a rotation per step.

    elements FACES (default), EDGES or VERTS, named by indices. mode REGION (faces extruded together, their shared edges
    kept - default) or INDIVIDUAL (each face on its own). Each step extrudes the previous step's new geometry and then,
    in order: moves it by `offset` [x,y,z] or `distance` along the normal (the faces' area-weighted normal for a region,
    each face's own for INDIVIDUAL, the vertices' averaged normal for edges and vertices), scales it about its own
    median by `scale` (a number or [x,y,z]) and rotates it about its median by `rotate` {angle (degrees), axis [x,y,z]}.
    steps (default 1, at most 100) repeats the three - a handle bent out of a cup wall step by step. space LOCAL
    (default) or WORLD (vectors and lengths in world units, the object's rotation and scale applied).

    A step that would neither move, scale nor rotate is refused: the new geometry would sit on the old as coincident
    vertices. The originals of a face extrude are removed (the region's bottom is inside the new walls), as Blender's
    own extrude does. Reply: per step the new vertex, edge and face indices; the last step's are top*, ready for the
    next call; counts before and after.
    """
    reject_unknown(params, _EXTRUDE_KEYS, "extrude_mesh")
    obj = _target(params, "extrude_mesh")
    kind = str(take(params, "elements", default="FACES", kind=str)).upper()
    if kind not in ("FACES", "EDGES", "VERTS"):
        raise MifOpError("elements must be FACES, EDGES or VERTS, got %r. NOTHING was changed." % kind)
    mode = str(take(params, "mode", default="REGION", kind=str)).upper()
    if mode not in ("REGION", "INDIVIDUAL"):
        raise MifOpError("mode must be REGION or INDIVIDUAL, got %r. NOTHING was changed." % mode)
    if mode == "INDIVIDUAL" and kind != "FACES":
        raise MifOpError("mode INDIVIDUAL is for FACES; edges and vertices always extrude on their own. NOTHING was "
                         "changed.")
    space = _space(params)
    offset = _vec3(params, "offset")
    distance = take_float(params, "distance", default=None)
    if offset is not None and distance is not None:
        raise MifOpError("pass offset OR distance - each says how far a step moves. NOTHING was changed.")
    raw_scale = params.get("scale")
    if raw_scale is None:
        scale = None
    elif isinstance(raw_scale, (int, float)) and not isinstance(raw_scale, bool):
        scale = Vector(finite_floats([raw_scale] * 3, "scale"))
    else:
        scale = _vec3(params, "scale")
    if scale is not None and any(abs(c) < 1e-9 for c in scale):
        raise MifOpError("scale must not be 0 on any axis - it would collapse the step onto a point or a line. NOTHING "
                         "was changed.")
    rot = params.get("rotate")
    angle = axis = None
    if rot is not None:
        if not isinstance(rot, dict) or set(rot) - {"angle", "axis"}:
            raise MifOpError("rotate must be {angle: degrees, axis: [x,y,z]}, got %r. NOTHING was changed." % (rot,))
        angle = take_float(rot, "angle", required=True)
        axis = _vec3(rot, "axis")
        if axis is None or axis.length < 1e-12:
            raise MifOpError("rotate needs a non-zero axis [x,y,z]. NOTHING was changed.")
        if space == "WORLD" and _uniform_scale(obj) is None:
            raise MifOpError("a rotation given in WORLD space on '%s', whose scale is not uniform, is not a rotation in "
                             "its mesh. Pass space LOCAL, or apply the scale. NOTHING was changed." % obj.name)
    steps = take_int(params, "steps", default=1)
    if not 1 <= steps <= 100:
        raise MifOpError("steps must be 1-100, got %d. NOTHING was changed." % steps)
    flip = take_bool(params, "flipNormals", default=False)
    moves = (offset is not None and offset.length > 0) or (distance is not None and distance != 0.0)
    if not moves and (scale is None or all(abs(c - 1.0) < 1e-9 for c in scale)) and not angle:
        raise MifOpError("this step neither moves, scales nor rotates, so the extruded geometry would sit exactly on the "
                         "old - coincident vertices. Give offset, distance, scale or rotate. NOTHING was changed.")
    offset_l = _direction(obj, offset, space)
    dist_l = _length(obj, distance, space, "distance")
    axis_l = _direction(obj, axis, space).normalized() if axis is not None else None
    before = mesh_counts(obj)
    bm = _open(obj)
    try:
        table = {"FACES": bm.faces, "EDGES": bm.edges, "VERTS": bm.verts}[kind]
        current = _elements(params, "indices", table, kind.lower()[:-1], obj)
        if kind == "EDGES" and dist_l is not None and any(len(e.link_faces) == 0 for e in current):
            raise MifOpError("a wire edge has no faces and so no normal to extrude along - pass offset. NOTHING was "
                             "changed.")
        per_step = []
        for _ in range(steps):
            if kind == "FACES" and mode == "REGION":
                normal = sum((f.normal * f.calc_area() for f in current), Vector())
                r = bmesh.ops.extrude_face_region(bm, geom=list(current), use_normal_flip=flip)
                bmesh.ops.delete(bm, geom=list(current), context="FACES_ONLY")
                new_v = [g for g in r["geom"] if isinstance(g, bmesh.types.BMVert)]
                new_e = [g for g in r["geom"] if isinstance(g, bmesh.types.BMEdge)]
                new_f = [g for g in r["geom"] if isinstance(g, bmesh.types.BMFace)]
                groups = [(new_v, normal)]
                current = new_f
            elif kind == "FACES":
                r = bmesh.ops.extrude_discrete_faces(bm, faces=list(current), use_normal_flip=flip)
                new_f = list(r["faces"])
                new_v = list(dict.fromkeys(v for f in new_f for v in f.verts))
                new_e = list(dict.fromkeys(e for f in new_f for e in f.edges))
                groups = [(list(f.verts), f.normal.copy()) for f in new_f]
                current = new_f
            elif kind == "EDGES":
                normal = sum((v.normal for e in current for v in e.verts), Vector())
                r = bmesh.ops.extrude_edge_only(bm, edges=list(current), use_normal_flip=flip)
                new_v = [g for g in r["geom"] if isinstance(g, bmesh.types.BMVert)]
                vs = set(new_v)
                new_e = [g for g in r["geom"] if isinstance(g, bmesh.types.BMEdge)]
                new_f = [g for g in r["geom"] if isinstance(g, bmesh.types.BMFace)]
                groups = [(new_v, normal)]
                current = [e for e in new_e if e.verts[0] in vs and e.verts[1] in vs]
            else:
                normal = sum((v.normal for v in current), Vector())
                r = bmesh.ops.extrude_vert_indiv(bm, verts=list(current))
                new_v, new_e, new_f = list(r["verts"]), list(r["edges"]), []
                groups = [(new_v, normal)]
                current = new_v
            for verts, n in groups:
                if dist_l is not None:
                    if n.length < 1e-12:
                        raise MifOpError("the extruded geometry has no normal to move along (its normals cancel) - "
                                         "pass offset. NOTHING was changed.")
                    move = n.normalized() * dist_l
                else:
                    move = offset_l if offset_l is not None else Vector()
                for v in verts:
                    v.co += move
                if scale is not None or angle:
                    c = _median(verts)
                    rm = Matrix.Rotation(math.radians(angle), 3, axis_l) if angle else Matrix.Identity(3)
                    for v in verts:
                        d = v.co - c
                        if scale is not None:
                            d = Vector((d.x * scale.x, d.y * scale.y, d.z * scale.z))
                        v.co = c + rm @ d
            per_step.append((new_v, new_e, new_f))
        bm.normal_update()
        _commit(bm, obj)
        steps_out = [{"vertices": _idx(v), "edges": _idx(e), "faces": _idx(f)} for v, e, f in per_step]
    finally:
        bm.free()
    last = steps_out[-1]
    out = {"object": obj.name, "elements": kind, "mode": mode if kind == "FACES" else None, "steps": steps,
           "space": space, "before": before, "after": mesh_counts(obj), "perStep": steps_out,
           "topVertices": last["vertices"], "topEdges": last["edges"] if kind != "FACES" else None,
           "topFaces": last["faces"] if kind == "FACES" else None}
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- loops and rings

def _marked(e, stop_at):
    """True when edge e carries a mark stop_at names: SEAM (UV seam) or SHARP (not smooth)."""
    return ("SEAM" in stop_at and e.seam) or ("SHARP" in stop_at and not e.smooth)


def edge_ring(seed, stop_at=()):
    """The edge ring through a seed edge: across each quad to the opposite edge, both ways, until a boundary, a face that
    is not a quad, or the start again. Returns ([(v0, v1) oriented edges...], closed). Oriented so each pair's v0 is on
    the same side of the ring as the previous pair's v0 - what a slide needs to move every cut the same way.

    stop_at (2026-10-06) also ends a direction at a SEAM or SHARP ring edge (that edge kept, nothing crossed past it) and,
    with MATERIAL, before a face whose material differs from the first face that direction crossed."""
    def opposite(face, a, b):
        loops = list(face.loops)
        if len(loops) != 4:
            return None
        for i, lp in enumerate(loops):
            if {lp.vert, loops[(i + 1) % 4].vert} == {a, b}:
                p, q = loops[(i + 2) % 4].vert, loops[(i + 3) % 4].vert
                # the edge p-q: the vertex next to a across the quad is the one joined to a by an edge
                return (q, p) if lp.vert is a else (p, q)
        return None

    def walk(a, b, face_from):
        out = []
        e = next(x for x in a.link_edges if x.other_vert(a) is b)
        prev_face = face_from
        material = None
        while True:
            nxt = [f for f in e.link_faces if f is not prev_face]
            if not nxt:
                return out, False
            face = nxt[0]
            if "MATERIAL" in stop_at:
                if material is None:
                    material = face.material_index
                elif face.material_index != material:
                    return out, False
            pair = opposite(face, a, b)
            if pair is None:
                return out, False
            a, b = pair
            e = next(x for x in a.link_edges if x.other_vert(a) is b)
            if e is seed:
                return out, True
            out.append((a, b))
            if _marked(e, stop_at):
                return out, False
            prev_face = face

    a, b = seed.verts
    faces = list(seed.link_faces)
    fwd, closed = walk(a, b, faces[1] if len(faces) > 1 else None)
    if closed:
        return [(a, b)] + fwd, True
    back = []
    if len(faces) > 1:
        back, _c = walk(a, b, faces[0])
    return list(reversed(back)) + [(a, b)] + fwd, False


def edge_loop(seed, stop_at=()):
    """The edge loop through a seed edge: at each vertex with exactly four edges, the edge that shares no face with the
    one arriving; on a boundary, the next boundary edge where a vertex has exactly two. Both ways, until neither holds or
    the start comes round again. Returns (edges, closed).

    stop_at (2026-10-06, the stops 5.1's loop select offers) also ends a direction at a vertex where: SEAM or SHARP - a
    marked edge other than the two the loop runs through meets it (a loop up to a sleeve's seam); MATERIAL - the next
    edge's faces carry other materials than the arriving edge's; and on a boundary loop, OUTER_CORNER - the vertex has
    one face, INNER_CORNER - three or more, NGON - the boundary passes there between an n-gon's side and another face's."""
    boundary = len(seed.link_faces) == 1

    def stops(e, v, n):
        if any(_marked(x, stop_at) for x in v.link_edges if x is not e and x is not n):
            return True
        if "MATERIAL" in stop_at and ({f.material_index for f in e.link_faces}
                                      != {f.material_index for f in n.link_faces}):
            return True
        if boundary:
            nf = len(v.link_faces)
            if ("OUTER_CORNER" in stop_at and nf == 1) or ("INNER_CORNER" in stop_at and nf >= 3):
                return True
            if "NGON" in stop_at and (any(len(f.verts) > 4 for f in e.link_faces)
                                      != any(len(f.verts) > 4 for f in n.link_faces)):
                return True
        return False

    def step(e, v):
        if boundary:
            bes = [x for x in v.link_edges if len(x.link_faces) == 1]
            if len(bes) != 2:
                return None
            n = bes[0] if bes[1] is e else bes[1]
        else:
            if len(v.link_edges) != 4:
                return None
            fs = set(e.link_faces)
            cand = [x for x in v.link_edges if x is not e and not (set(x.link_faces) & fs)]
            if len(cand) != 1:
                return None
            n = cand[0]
        return None if stop_at and stops(e, v, n) else n

    out = [seed]
    seen = {seed}
    e, v = seed, seed.verts[1]
    while True:
        n = step(e, v)
        if n is None:
            break
        if n is seed:
            return out, True
        if n in seen:
            break
        out.append(n)
        seen.add(n)
        e, v = n, n.other_vert(v)
    e, v = seed, seed.verts[0]
    back = []
    while True:
        n = step(e, v)
        if n is None or n in seen:
            break
        back.append(n)
        seen.add(n)
        e, v = n, n.other_vert(v)
    return list(reversed(back)) + out, False


def _seed_edge(params, bm, obj):
    idx = params.get("edge")
    near = _vec3(params, "edgeNear")
    if (idx is None) == (near is None):
        raise MifOpError("pass edge (an index) OR edgeNear [x,y,z] (world) - the seed edge. NOTHING was changed.")
    if idx is not None:
        if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(bm.edges):
            raise MifOpError("edge %r is not an edge index of '%s' (it has %d). NOTHING was changed."
                             % (idx, obj.name, len(bm.edges)))
        return bm.edges[idx]
    local = obj.matrix_world.inverted_safe() @ near
    return min(bm.edges, key=lambda e: ((e.verts[0].co + e.verts[1].co) * 0.5 - local).length)


_LOOP_CUT_KEYS = ("object", "name", "edge", "edgeNear", "count", "slide", "even", "smoothness")


def _hermite(a, b, na, nb, t):
    """A point t of the way along a-b on the cubic Hermite curve whose end tangents lie in each end's tangent plane
    (the chord with the vertex normal taken off, at the chord's length): on a circle it bows out toward the arc."""
    chord = b - a
    length = chord.length

    def tangent(n):
        tv = chord - n * chord.dot(n) if n.length > 1e-12 else chord.copy()
        return tv.normalized() * length if tv.length > 1e-12 else chord.copy()
    ta, tb = tangent(na), tangent(nb)
    t2, t3 = t * t, t * t * t
    return (a * (2 * t3 - 3 * t2 + 1) + ta * (t3 - 2 * t2 + t) + b * (-2 * t3 + 3 * t2) + tb * (t3 - t2))


def op_loop_cut(params):
    """Cut edge loops across the ring of quads through a seed edge, with a count and a slide.

    The seed is an edge index or edgeNear [x,y,z] (world: the edge whose middle is nearest). The ring runs across each
    quad to its opposite edge, both ways, until a boundary, a face that is not a quad, or the start again (a closed ring
    round a cylinder). count (default 1, at most 64) loops are cut, evenly spaced along every ring edge; slide (-1 to 1
    exclusive, default 0) moves them all toward one end by that share of the spacing - +1 would reach the seed edge's
    second vertex - the way a support loop is slid toward a rim to tighten its rounding under subdivision.

    even (default false) puts each cut at the same DISTANCE from the ring's first side - the seed edge's share times the
    seed edge's length - instead of the same share of every ring edge, so the loop runs parallel to that side (the
    "straight and even" loops a shirt's seams need); a cut that would pass the end of a shorter ring edge is refused.
    smoothness (-1 to 1, default 0) moves each new vertex that share of the way from the straight edge to a cubic curve
    whose end tangents lie in each end's tangent plane - on a curved surface the loop follows the curve instead of
    cutting a chord across it (1 on a radius-1 octagon's 45-degree edge: the midpoint at radius 0.997103, against the
    chord's 0.92388).

    Reply: loops - for each cut, its vertex indices in ring order and its edge indices - and the ring's edge count and
    whether it closed. select_edges {seedEdge, loop: true} on a reply edge finds the same loop.
    """
    reject_unknown(params, _LOOP_CUT_KEYS, "loop_cut")
    obj = _target(params, "loop_cut")
    count = take_int(params, "count", default=1)
    slide = take_float(params, "slide", default=0.0)
    if not 1 <= count <= 64:
        raise MifOpError("count must be 1-64, got %d. NOTHING was changed." % count)
    if not -1.0 < slide < 1.0:
        raise MifOpError("slide must be between -1 and 1, exclusive (at 1 a loop would land on the ring edge's end), got "
                         "%r. NOTHING was changed." % slide)
    even = take_bool(params, "even", default=False)
    smooth = take_float(params, "smoothness", default=0.0)
    if not -1.0 <= smooth <= 1.0:
        raise MifOpError("smoothness must be -1 to 1, got %r. NOTHING was changed." % smooth)
    before = mesh_counts(obj)
    bm = _open(obj)
    try:
        seed = _seed_edge(params, bm, obj)
        ring, closed = edge_ring(seed)
        if len(ring) < 2:
            raise MifOpError("edge %d has no ring to cut - neither of its faces is a quad. NOTHING was changed."
                             % seed.index)
        pairs = list(range(len(ring))) if closed else list(range(len(ring) - 1))
        # EVERY QUAD THE LOOPS CROSS, found before anything is split: two consecutive ring edges share one
        for i in pairs:
            j = (i + 1) % len(ring)
            ea = {ring[i][0], ring[i][1]}
            eb = {ring[j][0], ring[j][1]}
            if not any(ea <= set(f.verts) and eb <= set(f.verts) for f in ring[i][0].link_faces):
                raise MifOpError("ring edges %d and %d share no quad - the mesh is not a clean quad strip there. "
                                 "NOTHING was changed." % (i, j))
        fracs = [(k + 1) / (count + 1) + slide / (count + 1) for k in range(count)]
        # EVERY CUT'S SHARE AND POSITION, computed before anything is split, so a cut past an edge's end refuses first
        seed_len = seed.calc_length()
        plan = []
        for a, b in ring:
            pa, pb = a.co.copy(), b.co.copy()
            length = (pb - pa).length
            shares = [f * seed_len / length for f in fracs] if even else list(fracs)
            if shares and shares[-1] >= 1.0 - 1e-9:
                raise MifOpError("even: a cut %.6g from the ring's first side passes the end of a ring edge %.6g long. "
                                 "Use fewer cuts, a smaller slide, or even false. NOTHING was changed."
                                 % (fracs[-1] * seed_len, length))
            na, nb = a.normal.copy(), b.normal.copy()
            pos = []
            for t in shares:
                p = pa.lerp(pb, t)
                if smooth:
                    p = p + (_hermite(pa, pb, na, nb, t) - p) * smooth
                pos.append(p)
            plan.append((shares, pos))
        cuts = []          # per ring edge, the new vertices in order of fracs
        for (a, b), (shares, pos) in zip(ring, plan):
            e = next(x for x in a.link_edges if x.other_vert(a) is b)
            made, cur_e, cur_v, prev = [], e, a, 0.0
            for t, p in zip(shares, pos):
                ne, nv = bmesh.utils.edge_split(cur_e, cur_v, (t - prev) / (1.0 - prev))
                nv.co = p
                made.append(nv)
                cur_e = ne if b in ne.verts else cur_e
                cur_v, prev = nv, t
            cuts.append(made)
        loop_edges = [[] for _ in range(count)]
        for i in pairs:
            j = (i + 1) % len(ring)
            for k in range(count):
                va, vb = cuts[i][k], cuts[j][k]
                common = set(va.link_faces) & set(vb.link_faces)    # the quad checked above, or a part of it
                _f, lp = bmesh.utils.face_split(next(iter(common)), va, vb)
                loop_edges[k].append(lp.edge)
        bm.normal_update()
        _commit(bm, obj)
        loops = [{"vertices": [cuts[i][k].index for i in range(len(ring))], "edges": _idx(loop_edges[k])}
                 for k in range(count)]
        seed_i = seed.index
    finally:
        bm.free()
    out = {"object": obj.name, "seedEdge": seed_i, "ringEdges": len(ring), "closed": closed, "count": count,
           "slide": slide, "even": even, "smoothness": smooth, "loops": loops, "before": before,
           "after": mesh_counts(obj)}
    out.update(shared_data_note(obj))
    return out


_SLIDE_KEYS = ("object", "name", "edges", "factor", "clamp")


def op_slide_edges(params):
    """Slide a chain of edges (a loop) along the edges beside it, toward one side or the other.

    edges is a connected chain or closed loop of edge indices (a loop_cut reply, or select_edges seedEdge + loop).
    factor -1 to 1: each vertex moves along its rail - the edge at it, not in the chain, that lies in the face on the
    chain's side - by that share of the rail's length; + is one side and - the other, consistent along the chain
    because the side is read off the faces' winding. clamp (default true) refuses |factor| over 1; clamp false lets a
    loop slide past the rail's end along its line - how a turned-back cuff or hem is pushed past the original edge.

    A vertex with no rail on the asked side (a chain end on a boundary) is left and counted. Reply: each moved vertex's
    index, the rail lengths' range, and counts.
    """
    reject_unknown(params, _SLIDE_KEYS, "slide_edges")
    obj = _target(params, "slide_edges")
    factor = take_float(params, "factor", required=True)
    clamp = take_bool(params, "clamp", default=True)
    if clamp and not -1.0 <= factor <= 1.0:
        raise MifOpError("factor must be -1 to 1 with clamp on, got %r; clamp false slides past a rail's end. NOTHING "
                         "was changed." % factor)
    if factor == 0.0:
        raise MifOpError("factor 0 moves nothing. NOTHING was changed.")
    bm = _open(obj)
    try:
        chain = _elements(params, "edges", bm.edges, "edge", obj)
        cset = set(chain)
        # ORDER THE CHAIN, so each edge has a direction and its 'left' face is the one whose loop runs that way
        adj = {}
        for e in chain:
            for v in e.verts:
                adj.setdefault(v, []).append(e)
        if any(len(es) > 2 for es in adj.values()):
            raise MifOpError("the edges branch (a vertex holds three or more of them) - slide takes one chain or loop. "
                             "NOTHING was changed.")
        ends = [v for v, es in adj.items() if len(es) == 1]
        start = ends[0] if ends else chain[0].verts[0]
        order, v, used = [], start, set()
        while True:
            nxt = [e for e in adj[v] if e not in used]
            if not nxt:
                break
            e = nxt[0]
            used.add(e)
            order.append((e, v))
            v = e.other_vert(v)
        if len(used) != len(chain):
            raise MifOpError("the edges are not one connected chain (%d of %d reached from one end). NOTHING was changed."
                             % (len(used), len(chain)))
        side = 1 if factor > 0 else -1
        rails = {}
        for e, v0 in order:
            for lp in e.link_loops:
                running = lp.vert is v0           # this face's loop runs v0 -> v1 along the edge: the + side
                if (side > 0) != running:
                    continue
                for vert in e.verts:
                    other = [x for x in vert.link_edges if x not in cset and lp.face in x.link_faces]
                    if other and vert not in rails:
                        rails[vert] = other[0]
        verts = list(adj)
        moved, skipped, lengths = [], 0, []
        targets = {}
        for vert in verts:
            rail = rails.get(vert)
            if rail is None:
                skipped += 1
                continue
            far = rail.other_vert(vert).co
            lengths.append((far - vert.co).length)
            targets[vert] = vert.co + (far - vert.co) * abs(factor)
        if not targets:
            raise MifOpError("no vertex of the chain has a rail on that side - every face beside it is on the other side "
                             "or there is none. Try the other sign. NOTHING was changed.")
        for vert, p in targets.items():
            vert.co = p
            moved.append(vert)
        bm.normal_update()
        _commit(bm, obj)
        moved_i = _idx(moved)
    finally:
        bm.free()
    out = {"object": obj.name, "factor": factor, "clamp": clamp, "verticesMoved": len(moved_i), "vertices": moved_i,
           "verticesWithoutRail": skipped,
           "railLength": {"min": round(min(lengths), 6), "max": round(max(lengths), 6)} if lengths else None}
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- inset

_INSET_KEYS = ("object", "name", "faces", "thickness", "depth", "individual", "boundary", "even", "outset",
               "relativeOffset", "space")


def op_inset_faces(params):
    """Inset faces: a ring of new faces inside each region's edge, `thickness` wide, the inner face moved `depth` along
    the normal (negative pushes it in).

    individual (default false) insets each face on its own instead of the region as one. boundary (default true) insets
    a region's open edges too; even (default true) keeps the ring's width even at corners; outset grows outward
    instead; relativeOffset scales the width by the size of the neighbouring faces. Lengths in the mesh's units, or
    world units with space WORLD (uniform scale only). Reply: innerFaces (the faces given, now smaller) and ringFaces
    (new), counts before and after.
    """
    reject_unknown(params, _INSET_KEYS, "inset_faces")
    obj = _target(params, "inset_faces")
    space = _space(params)
    thickness = take_float(params, "thickness", required=True)
    depth = take_float(params, "depth", default=0.0)
    individual = take_bool(params, "individual", default=False)
    boundary = take_bool(params, "boundary", default=True)
    even = take_bool(params, "even", default=True)
    outset = take_bool(params, "outset", default=False)
    relative = take_bool(params, "relativeOffset", default=False)
    if not thickness > 0.0:
        raise MifOpError("thickness must be above 0, got %r. NOTHING was changed." % thickness)
    if individual and (outset or params.get("boundary") is not None):
        raise MifOpError("outset and boundary are for a region inset; individual insets each face on its own. NOTHING "
                         "was changed.")
    th = _length(obj, thickness, space, "thickness")
    dp = _length(obj, depth, space, "depth")
    before = mesh_counts(obj)
    bm = _open(obj)
    try:
        faces = _elements(params, "faces", bm.faces, "face", obj)
        if individual:
            r = bmesh.ops.inset_individual(bm, faces=faces, thickness=th, depth=dp, use_even_offset=even,
                                           use_interpolate=True, use_relative_offset=relative)
        else:
            r = bmesh.ops.inset_region(bm, faces=faces, thickness=th, depth=dp, use_boundary=boundary,
                                       use_even_offset=even, use_interpolate=True, use_relative_offset=relative,
                                       use_outset=outset)
        ring = list(r["faces"])
        if not ring:
            raise MifOpError("the inset made no faces. NOTHING was changed.")
        bm.normal_update()
        _commit(bm, obj)
        inner, ring_i = _idx(faces), _idx(ring)
    finally:
        bm.free()
    out = {"object": obj.name, "thickness": thickness, "depth": depth, "space": space, "individual": individual,
           "innerFaces": inner, "ringFaces": ring_i, "before": before, "after": mesh_counts(obj)}
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- knife and connect

_KNIFE_KEYS = ("object", "name", "points", "vertexPairs", "vertexPath", "space", "tolerance")


def _common_face(a, b):
    common = set(a.link_faces) & set(b.link_faces)
    return next(iter(common)) if common else None


def op_knife_cut(params):
    """Cut across faces between points, or connect vertices.

    points [[x,y,z], ...] (a polyline; world with space WORLD, the default here, or mesh space): each point lands on the
    mesh - on the vertex nearest it when one is within `tolerance` (default 1e-4 in the mesh's units), otherwise on the
    nearest point of the nearest edge, which is split there - and consecutive points are joined by a cut across the
    faces between them. vertexPairs [[a, b], ...] or vertexPath [a, b, c, ...] join existing vertices the same way: a
    pair in one face splits that face, a pair further apart is cut across the faces between (bmesh connect_vert_pair).
    A pair that cannot be joined (separate pieces, or the same vertex) is refused, naming it.

    Reply: pointsLanded (the vertex each point became), the cut's edge and vertex indices - a cut across several faces
    adds a vertex on every edge it crosses - and counts.
    """
    reject_unknown(params, _KNIFE_KEYS, "knife_cut")
    obj = _target(params, "knife_cut")
    space = str(take(params, "space", default="WORLD", kind=str)).upper()
    if space not in ("LOCAL", "WORLD"):
        raise MifOpError("space must be LOCAL or WORLD, got %r. NOTHING was changed." % space)
    tol = take_float(params, "tolerance", default=1e-4)
    if tol < 0:
        raise MifOpError("tolerance must be 0 or more. NOTHING was changed.")
    given = [k for k in ("points", "vertexPairs", "vertexPath") if params.get(k) is not None]
    if len(given) != 1:
        raise MifOpError("pass exactly one of points, vertexPairs or vertexPath (got %s). NOTHING was changed."
                         % (", ".join(given) or "none"))
    before = mesh_counts(obj)
    bm = _open(obj)
    try:
        if given[0] == "points":
            raw = params["points"]
            if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                raise MifOpError("points must be at least two [x,y,z] points. NOTHING was changed.")
            pts = [_point(obj, _vec3({"p": p}, "p"), space) for p in raw]
            chain = []
            for p in pts:
                v = min(bm.verts, key=lambda x: (x.co - p).length)
                if (v.co - p).length > tol:
                    best, best_d, best_t = None, None, None
                    for e in bm.edges:
                        a, b = e.verts[0].co, e.verts[1].co
                        ab = b - a
                        t = max(0.0, min(1.0, (p - a).dot(ab) / ab.length_squared)) if ab.length_squared else 0.0
                        d = (a + ab * t - p).length
                        if best_d is None or d < best_d:
                            best, best_d, best_t = e, d, t
                    if best_t <= 1e-6 or best_t >= 1 - 1e-6:
                        v = best.verts[0] if best_t <= 1e-6 else best.verts[1]
                    else:
                        a0 = best.verts[0]
                        _ne, v = bmesh.utils.edge_split(best, a0, best_t)
                chain.append(v)
            pairs = list(zip(chain, chain[1:]))
        else:
            if given[0] == "vertexPairs":
                raw = params["vertexPairs"]
                if not isinstance(raw, (list, tuple)) or not raw or any(
                        not isinstance(p, (list, tuple)) or len(p) != 2 for p in raw):
                    raise MifOpError("vertexPairs must be [[a, b], ...]. NOTHING was changed.")
                flat = [i for p in raw for i in p]
            else:
                flat = params["vertexPath"]
                if not isinstance(flat, (list, tuple)) or len(flat) < 2:
                    raise MifOpError("vertexPath must list at least two vertex indices. NOTHING was changed.")
            bad = [i for i in flat if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(bm.verts)]
            if bad:
                raise MifOpError("%s are not vertex indices of '%s' (it has %d). NOTHING was changed."
                                 % (bad[:10], obj.name, len(bm.verts)))
            vs = [bm.verts[i] for i in flat]
            pairs = list(zip(vs[0::2], vs[1::2])) if given[0] == "vertexPairs" else list(zip(vs, vs[1:]))
        cut_edges = []
        for a, b in pairs:
            if a is b:
                raise MifOpError("a pair joins vertex %d to itself. NOTHING was changed." % a.index)
            if any(e.other_vert(a) is b for e in a.link_edges):
                continue                                   # already joined by an edge: nothing to cut
            if _common_face(a, b) is not None:
                r = bmesh.ops.connect_verts(bm, verts=[a, b])
            else:
                r = bmesh.ops.connect_vert_pair(bm, verts=[a, b])
            if not r["edges"]:
                raise MifOpError("vertices %d and %d could not be joined - no faces run between them (separate pieces?). "
                                 "NOTHING was changed." % (a.index, b.index))
            cut_edges += list(r["edges"])
        if not cut_edges:
            raise MifOpError("every pair is already joined by an edge, so there is nothing to cut. NOTHING was changed.")
        bm.normal_update()
        _commit(bm, obj)
        landed = _idx(chain) if given[0] == "points" else None
        ne = _idx(cut_edges)
        cut_v = sorted({v.index for e in cut_edges if e.is_valid for v in e.verts})
    finally:
        bm.free()
    out = {"object": obj.name, "pointsLanded": landed, "cutEdges": ne, "cutVertices": cut_v, "before": before,
           "after": mesh_counts(obj)}
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- join two loops

_JOIN_KEYS = ("object", "name", "loopA", "loopB", "mode", "twist", "match", "cuts", "interpolation", "runA", "runB",
              "runsB", "reportOnly", "tolerance")
_RUN_KEYS = {"from", "to", "way", "along"}


def _run_vertex(bm, obj, spec, key):
    """One end of a run: a vertex index or {near: [x,y,z]} (world) - the nearest vertex."""
    if isinstance(spec, dict) and set(spec) == {"near"}:
        p = obj.matrix_world.inverted_safe() @ _vec3(spec, "near")
        return min(bm.verts, key=lambda v: (v.co - p).length)
    if isinstance(spec, bool) or not isinstance(spec, int) or not 0 <= spec < len(bm.verts):
        raise MifOpError("%s must be a vertex index of '%s' (it has %d) or {near: [x,y,z]}; got %r. NOTHING was "
                         "changed." % (key, obj.name, len(bm.verts), spec))
    return bm.verts[spec]


def _edge_path(a, b):
    """The cheapest edge path a -> b by length (Dijkstra over the mesh's edges), as vertices, or None."""
    import heapq   # noqa: PLC0415
    dist, prev, done = {a: 0.0}, {}, set()
    heap, n = [(0.0, 0, a)], 1
    while heap:
        d, _k, v = heapq.heappop(heap)
        if v in done:
            continue
        if v is b:
            break
        done.add(v)
        for e in v.link_edges:
            w = e.other_vert(v)
            nd = d + e.calc_length()
            if w not in dist or nd < dist[w]:
                dist[w], prev[w] = nd, v
                heapq.heappush(heap, (nd, n, w))
                n += 1
    if b not in dist:
        return None
    out = [b]
    while out[-1] is not a:
        out.append(prev[out[-1]])
    return out[::-1]


def _open_run(bm, obj, loops, spec, key):
    """An ordered run of vertices from `from` to `to`: along the boundary loop holding both (the shorter way round, or
    `way` LONG), or along the mesh's cheapest edge path (`along` EDGES - an internal line). Refuses before any write."""
    if not isinstance(spec, dict) or set(spec) - _RUN_KEYS or "from" not in spec or "to" not in spec:
        raise MifOpError("%s is {from, to, way SHORT|LONG, along BOUNDARY|EDGES}, each end a vertex index or {near: "
                         "[x,y,z]}; got %r. NOTHING was changed." % (key, spec))
    a = _run_vertex(bm, obj, spec["from"], key + ".from")
    b = _run_vertex(bm, obj, spec["to"], key + ".to")
    if a is b:
        raise MifOpError("%s starts and ends on vertex %d - a run needs two ends. NOTHING was changed." % (key, a.index))
    along = str(spec.get("along") or "BOUNDARY").upper()
    way = str(spec.get("way") or "SHORT").upper()
    if along not in ("BOUNDARY", "EDGES") or way not in ("SHORT", "LONG"):
        raise MifOpError("%s: along is BOUNDARY or EDGES and way SHORT or LONG. NOTHING was changed." % key)
    if along == "EDGES":
        if spec.get("way") is not None:
            raise MifOpError("%s: way picks a side of a boundary loop; an EDGES run is the cheapest path. NOTHING was "
                             "changed." % key)
        path = _edge_path(a, b)
        if path is None:
            raise MifOpError("%s: vertices %d and %d are not connected by edges. NOTHING was changed."
                             % (key, a.index, b.index))
        return path
    ring = None
    for lp in loops:
        if a in lp[1] and b in lp[1]:
            ring = _ordered(lp)
            break
    if ring is None:
        raise MifOpError("%s: vertices %d and %d are not on one open boundary loop (along EDGES follows an internal "
                         "line). NOTHING was changed." % (key, a.index, b.index))
    n, i, j = len(ring), ring.index(a), ring.index(b)
    fwd = [ring[(i + k) % n] for k in range((j - i) % n + 1)]
    back = [ring[(i - k) % n] for k in range((i - j) % n + 1)]

    def length(seq):
        return sum((p.co - q.co).length for p, q in zip(seq, seq[1:]))
    short, long_ = (fwd, back) if length(fwd) <= length(back) else (back, fwd)
    return short if way == "SHORT" else long_


def _resample_open(bm, order, want):
    """Split the longest edges of an ordered OPEN run at their middles until it has `want` vertices."""
    order = list(order)
    while len(order) < want:
        i = max(range(len(order) - 1), key=lambda k: (order[k + 1].co - order[k].co).length)
        a, b = order[i], order[i + 1]
        e = next(x for x in a.link_edges if x.other_vert(a) is b)
        _ne, nv = bmesh.utils.edge_split(e, a, 0.5)
        nv.co = (a.co + b.co) * 0.5
        order.insert(i + 1, nv)
    return order


def _world_length(obj, seq):
    mw = obj.matrix_world
    return sum(((mw @ p.co) - (mw @ q.co)).length for p, q in zip(seq, seq[1:]))


def _join_runs(params, obj, mode, match, tol, report_only):
    """join_boundary_loops with runs - see its docstring."""
    for key in ("loopA", "loopB", "twist", "cuts", "interpolation"):
        if params.get(key) is not None:
            raise MifOpError("%s is for whole loops; runs sew or fuse open runs. NOTHING was changed." % key)
    if mode == "BRIDGE" and not report_only:
        raise MifOpError("runs SEW or FUSE (or reportOnly); BRIDGE joins whole loops. NOTHING was changed.")
    if (params.get("runB") is None) == (params.get("runsB") is None):
        raise MifOpError("pass runB (one run) or runsB (runs sewn to runA in order). NOTHING was changed.")
    specs_b = [params["runB"]] if params.get("runB") is not None else params["runsB"]
    if not isinstance(specs_b, (list, tuple)) or not specs_b:
        raise MifOpError("runsB is a list of runs. NOTHING was changed.")
    before = mesh_counts(obj)
    bm = _open(obj)
    try:
        loops = _boundary_loops_of(bm)
        ra = _open_run(bm, obj, loops, params["runA"], "runA")
        rbs = [_open_run(bm, obj, loops, s, "runsB[%d]" % i if params.get("runsB") is not None else "runB")
               for i, s in enumerate(specs_b)]
        rb = []
        for r in rbs:
            rb.extend(r[1:] if rb and r[0] is rb[-1] else r)
        if set(ra) & set(rb):
            raise MifOpError("runA and the B run(s) share vertices %s - a seam joins two different edges. NOTHING was "
                             "changed." % sorted(v.index for v in set(ra) & set(rb))[:10])
        if len(set(rb)) != len(rb):
            raise MifOpError("the B runs cross themselves (a vertex twice). NOTHING was changed.")
        len_a = _world_length(obj, ra)
        len_bs = [_world_length(obj, r) for r in rbs]
        # THE SUM OF THE B RUNS, not the concatenated path: a band equals the edges it joins, and the step from one
        # B run to the next is no part of any seam.
        len_b = sum(len_bs)
        seam = {"lengthA": round(len_a, 6), "lengthB": round(len_b, 6), "lengthsB": [round(x, 6) for x in len_bs],
                "difference": round(len_a - len_b, 6), "tolerance": tol, "flagged": abs(len_a - len_b) > tol,
                "verticesA": len(ra), "verticesB": len(rb)}
        if report_only:
            return {"object": obj.name, "reportOnly": True, "seam": seam}
        if len(ra) != len(rb) and match == "EQUAL":
            raise MifOpError("%s pairs vertex for vertex, and the runs have %d and %d - pass match RESAMPLE to split "
                             "the shorter one's edges. NOTHING was changed." % (mode, len(ra), len(rb)))
        resampled = 0
        if len(ra) < len(rb):
            resampled = len(rb) - len(ra)
            ra = _resample_open(bm, ra, len(rb))
        elif len(rb) < len(ra):
            resampled = len(ra) - len(rb)
            rb = _resample_open(bm, rb, len(ra))
        straight = sum((p.co - q.co).length for p, q in zip(ra, rb))
        flipped = sum((p.co - q.co).length for p, q in zip(ra, rb[::-1]))
        reversed_b = flipped < straight
        if reversed_b:
            rb = rb[::-1]
        pairs = list(zip(ra, rb))
        dists = [(p.co - q.co).length for p, q in pairs]
        new_edges = []
        if mode == "SEW":
            for p, q in pairs:
                if not any(e.other_vert(p) is q for e in p.link_edges):
                    new_edges.append(bm.edges.new((p, q)))
        else:
            for p, q in pairs:
                p.co = (p.co + q.co) * 0.5
            bmesh.ops.weld_verts(bm, targetmap={q: p for p, q in pairs})
        bm.normal_update()
        _commit(bm, obj)
        ne = _idx(new_edges)
    finally:
        bm.free()
    out = {"object": obj.name, "mode": mode, "runs": True, "match": match, "verticesAddedByResample": resampled,
           "pairs": len(pairs), "reversedB": reversed_b,
           "pairDistance": {"min": round(min(dists), 6), "max": round(max(dists), 6)}, "newEdges": ne,
           "seam": seam, "before": before, "after": mesh_counts(obj)}
    out.update(shared_data_note(obj))
    return out


def _resample(bm, order, want):
    """Split the longest edges of an ordered closed loop at their middles until it has `want` vertices; the loop order
    with the new vertices in place."""
    order = list(order)
    while len(order) < want:
        n = len(order)
        i = max(range(n), key=lambda k: (order[(k + 1) % n].co - order[k].co).length)
        a, b = order[i], order[(i + 1) % n]
        e = next(x for x in a.link_edges if x.other_vert(a) is b)
        _ne, nv = bmesh.utils.edge_split(e, a, 0.5)
        nv.co = (a.co + b.co) * 0.5
        order.insert(i + 1, nv)
    return order


def _inner(v, loop_set):
    """The vertex across v's one edge that is not on its loop - where the surface goes on from the rim - or None."""
    others = [e.other_vert(v) for e in v.link_edges if e.other_vert(v) not in loop_set and e.link_faces]
    return others[0] if len(others) == 1 else None


def _boundary_loops_of(bm):
    from .ops_mesh import _boundary_loops   # noqa: PLC0415 - one loop finder for the add-on
    return _boundary_loops(bm)


def _pick_loop(bm, obj, loops, spec, key):
    """A boundary loop by {near: [x,y,z]} (world), {vertex: i}, {edge: i} or {index: i} (mesh_quality's order)."""
    if not isinstance(spec, dict) or len(spec) != 1 or next(iter(spec)) not in ("near", "vertex", "edge", "index"):
        raise MifOpError("%s must be one of {near: [x,y,z]}, {vertex: i}, {edge: i} or {index: i} (mesh_quality "
                         "boundaryLoops' order); got %r. NOTHING was changed." % (key, spec))
    k, v = next(iter(spec.items()))
    if k == "index":
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < len(loops):
            raise MifOpError("%s index %r: '%s' has %d boundary loops. NOTHING was changed." % (key, v, obj.name,
                                                                                               len(loops)))
        return loops[v]
    if k == "near":
        p = obj.matrix_world.inverted_safe() @ _vec3({"near": v}, "near")
        return min(loops, key=lambda lp: min((x.co - p).length for x in lp[1]))
    table = bm.verts if k == "vertex" else bm.edges
    if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < len(table):
        raise MifOpError("%s %s %r is out of range. NOTHING was changed." % (key, k, v))
    el = table[v]
    for lp in loops:
        if (el in lp[1]) if k == "vertex" else (el in lp[0]):
            return lp
    raise MifOpError("%s %s %d is not on an open boundary. NOTHING was changed." % (key, k, v))


def _ordered(loop):
    edges, verts = loop
    adj = {}
    for e in edges:
        for v in e.verts:
            adj.setdefault(v, []).append(e)
    if any(len(x) != 2 for x in adj.values()):
        return None
    start = verts[0]
    out, prev, v = [start], None, start
    while True:
        nxt = [e for e in adj[v] if e is not prev][0]
        w = nxt.other_vert(v)
        if w is start:
            return out
        out.append(w)
        prev, v = nxt, w


def op_join_boundary_loops(params):
    """Join two open boundary loops of one mesh: BRIDGE them with a band of faces, FUSE them into one seam, or SEW them
    with loose edges a cloth's sewing springs pull together.

    loopA and loopB each name a loop: {near: [x,y,z]} (world), {vertex: i}, {edge: i} or {index: i} (mesh_quality
    boundaryLoops' order). Both must be closed (no bow-tie) and distinct. Vertices are paired around the loops by the
    rotation and direction that make the pairs closest in total (twist adds a further rotation); FUSE and SEW need the
    same vertex count, BRIDGE runs bmesh bridge_loops, which pairs them itself. FUSE welds each B vertex onto its A
    partner at their midpoint. Two meshes are joined first with join_objects - this works inside one object.

    match EQUAL (default) refuses loops of different vertex counts for FUSE and SEW; RESAMPLE first splits the shorter
    loop's longest edges at their middles until the counts match (a 4-vertex rim fused to an 8-vertex one gains 4).
    BRIDGE with equal loops takes cuts (0-64): that many rings across the band, each rung split at even shares
    (interpolation LINEAR) or along a cubic curve leaving each rim the way its surface runs (SMOOTH) - two offset tubes
    bridged SMOOTH meet them square instead of in a straight slant.

    Reply: the new faces (BRIDGE) or loose edges (SEW), the per-pair distances (min, max), and counts.

    OPEN RUNS (2026-10-04) - a side or shoulder seam is a stretch of one panel's outline sewn to a stretch of another's,
    and a flat panel has ONE boundary loop: runA and runB (or runsB, several runs sewn to runA in order - one hood edge
    to both neckline halves) are each {from, to} by vertex index or {near: [x,y,z]}; the run follows the boundary loop
    holding both ends the shorter way round (way LONG for the other), or the cheapest edge path for an internal line
    (along EDGES - a drawcord's centerline). SEW adds loose edges pair by pair, FUSE welds at the midpoints; match
    RESAMPLE splits the shorter run's longest edges until the counts meet; B is paired in the direction that puts the
    pairs closest (reversedB). seam reports both run lengths in world units, each B run's length, the difference and
    whether it passes tolerance (0.001); reportOnly measures and writes nothing.
    """
    reject_unknown(params, _JOIN_KEYS, "join_boundary_loops")
    if params.get("runA") is not None or params.get("runB") is not None or params.get("runsB") is not None:
        if params.get("runA") is None:
            raise MifOpError("runB and runsB are sewn to runA, which is missing. NOTHING was changed.")
        obj_r = _target(params, "join_boundary_loops")
        mode_r = str(take(params, "mode", default="SEW", kind=str)).upper()
        if mode_r not in ("BRIDGE", "FUSE", "SEW"):
            raise MifOpError("mode must be BRIDGE, FUSE or SEW, got %r. NOTHING was changed." % mode_r)
        match_r = str(take(params, "match", default="EQUAL", kind=str)).upper()
        if match_r not in ("EQUAL", "RESAMPLE"):
            raise MifOpError("match must be EQUAL or RESAMPLE, got %r. NOTHING was changed." % match_r)
        tol = take_float(params, "tolerance", default=0.001)
        if not tol >= 0.0:
            raise MifOpError("tolerance must be 0 or more. NOTHING was changed.")
        return _join_runs(params, obj_r, mode_r, match_r, tol, take_bool(params, "reportOnly", default=False))
    for key in ("reportOnly", "tolerance"):
        if params.get(key) is not None:
            raise MifOpError("%s goes with runA. NOTHING was changed." % key)
    obj = _target(params, "join_boundary_loops")
    mode = str(take(params, "mode", default="BRIDGE", kind=str)).upper()
    if mode not in ("BRIDGE", "FUSE", "SEW"):
        raise MifOpError("mode must be BRIDGE, FUSE or SEW, got %r. NOTHING was changed." % mode)
    twist = take_int(params, "twist", default=0)
    match = str(take(params, "match", default="EQUAL", kind=str)).upper()
    if match not in ("EQUAL", "RESAMPLE"):
        raise MifOpError("match must be EQUAL or RESAMPLE, got %r. NOTHING was changed." % match)
    cuts = take_int(params, "cuts", default=0)
    interp = str(take(params, "interpolation", default="LINEAR", kind=str)).upper()
    if interp not in ("LINEAR", "SMOOTH"):
        raise MifOpError("interpolation must be LINEAR or SMOOTH, got %r. NOTHING was changed." % interp)
    if (cuts or params.get("interpolation") is not None) and mode != "BRIDGE":
        raise MifOpError("cuts and interpolation shape a BRIDGE's band. NOTHING was changed.")
    if not 0 <= cuts <= 64:
        raise MifOpError("cuts must be 0-64, got %d. NOTHING was changed." % cuts)
    if params.get("loopA") is None or params.get("loopB") is None:
        raise MifOpError("loopA and loopB are both required. NOTHING was changed.")
    before = mesh_counts(obj)
    bm = _open(obj)
    try:
        loops = _boundary_loops_of(bm)
        if len(loops) < 2:
            raise MifOpError("'%s' has %d open boundary loop(s); joining needs two. NOTHING was changed."
                             % (obj.name, len(loops)))
        la = _pick_loop(bm, obj, loops, params["loopA"], "loopA")
        lb = _pick_loop(bm, obj, loops, params["loopB"], "loopB")
        if la is lb:
            raise MifOpError("loopA and loopB are the same loop. NOTHING was changed.")
        oa, ob = _ordered(la), _ordered(lb)
        if oa is None or ob is None:
            raise MifOpError("a loop is not a clean ring (a bow-tie vertex joins it to another). NOTHING was changed.")
        if mode != "BRIDGE" and len(oa) != len(ob) and match == "EQUAL":
            raise MifOpError("%s needs loops of equal vertex count, and these have %d and %d - pass match RESAMPLE to "
                             "split the shorter one's edges, or BRIDGE. NOTHING was changed." % (mode, len(oa), len(ob)))
        if cuts and len(oa) != len(ob) and match == "EQUAL":
            raise MifOpError("cuts need a band of quads, and loops of %d and %d vertices bridge with triangles - pass "
                             "match RESAMPLE. NOTHING was changed." % (len(oa), len(ob)))
        resampled = 0
        if match == "RESAMPLE" and len(oa) != len(ob):
            if len(oa) < len(ob):
                resampled = len(ob) - len(oa)
                oa = _resample(bm, oa, len(ob))
            else:
                resampled = len(oa) - len(ob)
                ob = _resample(bm, ob, len(oa))
            la = ([next(x for x in a.link_edges if x.other_vert(a) is b) for a, b in zip(oa, oa[1:] + oa[:1])], oa)
            lb = ([next(x for x in a.link_edges if x.other_vert(a) is b) for a, b in zip(ob, ob[1:] + ob[:1])], ob)
        n = len(oa)
        best = None
        if len(oa) == len(ob):
            for rev in (False, True):
                seq = list(reversed(ob)) if rev else ob
                for off in range(n):
                    cost = sum((oa[i].co - seq[(i + off) % n].co).length for i in range(n))
                    if best is None or cost < best[0]:
                        best = (cost, rev, off)
            _c, rev, off = best
            seq = list(reversed(ob)) if rev else ob
            pairs = [(oa[i], seq[(i + off + twist) % n]) for i in range(n)]
            dists = [(a.co - b.co).length for a, b in pairs]
        else:
            pairs, dists = [], []
        new_faces, new_edges = [], []
        if mode == "BRIDGE":
            set_a, set_b = set(oa), set(ob)
            inner = {v: _inner(v, set_a if v in set_a else set_b) for v in oa + ob}
            r = bmesh.ops.bridge_loops(bm, edges=list(la[0]) + list(lb[0]), twist_offset=twist)
            new_faces = list(r["faces"])
            if not new_faces:
                raise MifOpError("bridge_loops made no faces; the mesh was not written.")
            if cuts:
                rungs = []
                for a in oa:
                    rung = [e for e in a.link_edges if e.other_vert(a) in set_b]
                    if len(rung) != 1:
                        raise MifOpError("the band is not one rung per vertex, so it cannot be cut evenly; the mesh was "
                                         "not written.")
                    rungs.append((a, rung[0].other_vert(a), rung[0]))
                rings = []
                for a, b, e in rungs:
                    pa, pb = a.co.copy(), b.co.copy()
                    length = (pb - pa).length
                    ia, ib = inner[a], inner[b]
                    ta = (pa - ia.co).normalized() * length if ia is not None else pb - pa
                    tb = (ib.co - pb).normalized() * length if ib is not None else pb - pa
                    made, cur_e, cur_v, prev = [], e, a, 0.0
                    for k in range(cuts):
                        t = (k + 1) / (cuts + 1)
                        ne, nv = bmesh.utils.edge_split(cur_e, cur_v, (t - prev) / (1.0 - prev))
                        if interp == "SMOOTH":
                            t2, t3 = t * t, t * t * t
                            nv.co = (pa * (2 * t3 - 3 * t2 + 1) + ta * (t3 - 2 * t2 + t) + pb * (-2 * t3 + 3 * t2)
                                     + tb * (t3 - t2))
                        else:
                            nv.co = pa.lerp(pb, t)
                        made.append(nv)
                        cur_e = ne if b in ne.verts else cur_e
                        cur_v, prev = nv, t
                    rings.append(made)
                for i in range(len(rungs)):
                    j = (i + 1) % len(rungs)
                    for k in range(cuts):
                        va, vb = rings[i][k], rings[j][k]
                        common = set(va.link_faces) & set(vb.link_faces)
                        if common:
                            bmesh.utils.face_split(next(iter(common)), va, vb)
                new_faces = list(dict.fromkeys(f for ring_v in rings for v in ring_v for f in v.link_faces))
        elif mode == "SEW":
            for a, b in pairs:
                new_edges.append(bm.edges.new((a, b)))
        else:
            for a, b in pairs:
                a.co = (a.co + b.co) * 0.5
            bmesh.ops.weld_verts(bm, targetmap={b: a for a, b in pairs})
        bm.normal_update()
        _commit(bm, obj)
        nf, ne = _idx(new_faces), _idx(new_edges)
    finally:
        bm.free()
    out = {"object": obj.name, "mode": mode, "loopVertices": [len(oa), len(ob)], "twist": twist, "match": match,
           "verticesAddedByResample": resampled, "cuts": cuts, "interpolation": interp if mode == "BRIDGE" else None,
           "pairDistance": {"min": round(min(dists), 6), "max": round(max(dists), 6)} if dists else None,
           "newFaces": nf, "newEdges": ne, "before": before, "after": mesh_counts(obj)}
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- dissolve and fill

_DISSOLVE_KEYS = ("object", "name", "elements", "indices", "useFaceSplit", "useBoundaryTear", "alsoVerts", "mode", "angle",
                  "delimit", "dissolveBoundaries", "keepNgons")
_LIMITED_ONLY = ("angle", "delimit", "dissolveBoundaries", "keepNgons")
_NAMED_ONLY = ("useFaceSplit", "useBoundaryTear", "alsoVerts")


def _delimit(params):
    """delimit for a Limited Dissolve: the names this build's own dissolve_limited lists, default its own default."""
    prop = bpy.ops.mesh.dissolve_limited.get_rna_type().properties["delimit"]
    known = [i.identifier for i in prop.enum_items]
    raw = params.get("delimit")
    if raw is None:
        return sorted(prop.default_flag)
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise MifOpError("delimit is a list of %s (an empty list delimits nothing), got %r. NOTHING was changed."
                         % (", ".join(known), raw))
    bad = [d for d in raw if not isinstance(d, str) or d.upper() not in known]
    if bad:
        raise MifOpError("delimit %s are not among this Blender's %s. NOTHING was changed." % (bad, ", ".join(known)))
    return sorted({d.upper() for d in raw})


def _dissolve_limited(params, obj, kind):
    """mode LIMITED: bmesh.ops.dissolve_limit over the chosen elements (all of them without indices)."""
    angle = take_float(params, "angle", default=5.0)
    if not 0.0 < angle <= 180.0:
        raise MifOpError("angle is in DEGREES, above 0 and at most 180 (default 5), got %g. NOTHING was changed." % angle)
    delimit = _delimit(params)
    boundaries = take_bool(params, "dissolveBoundaries", default=False)
    keep_ngons = take_bool(params, "keepNgons", default=False)
    if params.get("elements") is not None and params.get("indices") is None:
        raise MifOpError("elements says what indices are; without indices a Limited Dissolve works on the whole mesh - "
                         "pass indices, or drop elements. NOTHING was changed.")
    before = mesh_counts(obj)
    bm = _open(obj)
    try:
        if params.get("indices") is None:
            verts, edges = bm.verts[:], bm.edges[:]
        else:
            table = {"VERTS": bm.verts, "EDGES": bm.edges, "FACES": bm.faces}[kind]
            items = _elements(params, "indices", table, kind.lower()[:-1], obj)
            if kind == "FACES":
                edges = list({e for f in items for e in f.edges})
                verts = list({v for f in items for v in f.verts})
            elif kind == "EDGES":
                edges = items
                verts = list({v for e in items for v in e.verts})
            else:
                verts, edges = items, []
        start = (len(bm.verts), len(bm.edges), len(bm.faces))
        region = list(bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(angle), use_dissolve_boundaries=boundaries,
                                               verts=verts, edges=edges, delimit=set(delimit))["region"])
        end = (len(bm.verts), len(bm.edges), len(bm.faces))
        if end == start:
            raise MifOpError("a Limited Dissolve at %g degrees changed nothing - no two faces meet flatter than that "
                             "among the elements given (delimit %s). NOTHING was changed." % (angle, delimit or "none"))
        if end[2] == 0:
            # MEASURED 2026-10-05 on 3.6.23, 4.4.0 and 5.2.1: a flat 5 x 5 grid with dissolveBoundaries true comes out as
            # 2 vertices, 1 edge and no face.
            raise MifOpError("this Limited Dissolve would leave '%s' with no face at all (%d vertices, %d edges) - "
                             "dissolveBoundaries dissolves an open mesh's rim too. NOTHING was changed."
                             % (obj.name, end[0], end[1]))
        # N-GONS SPLIT, THE SURFACE UNCHANGED: this is the pass gameReady names for triangles over a budget, and what it
        # left failed gameReady's ngons. keepNgons leaves them, for a mesh still being modeled.
        split = 0
        if not keep_ngons:
            split, born = triangulate_ngons(bm)
            kept = {f for f in region if f.is_valid}
            region = list(kept) + [new for new, old in born.items() if old in kept and new not in kept]
        uv0 = bm.loops.layers.uv.get(obj.data.uv_layers[0].name) if len(obj.data.uv_layers) else None
        opened = _open_collapsed_uvs(bm, uv0.name) if uv0 is not None else 0
        bm.normal_update()
        _commit(bm, obj)
        reg = _idx(region)
    finally:
        bm.free()
    after = mesh_counts(obj)
    out = {"object": obj.name, "mode": "LIMITED", "elements": kind if params.get("indices") is not None else "ALL",
           "angle": angle, "angleRadians": round(math.radians(angle), 6), "delimit": delimit,
           "dissolveBoundaries": boundaries, "regionFaces": reg, "ngonsSplit": split, "uvFacesOpened": opened,
           "facesDissolved": start[2] - end[2], "verticesDissolved": start[0] - end[0],
           "before": before, "after": after}
    out.update(shared_data_note(obj))
    return out


def op_dissolve_mesh_elements(params):
    """Dissolve vertices, edges or faces and keep the surface: the faces around them merge into one, where deleting
    would leave a hole.

    elements VERTS, EDGES or FACES with indices. useFaceSplit (vertices, edges) keeps the merged face split where it
    would be concave; useBoundaryTear (vertices) tears a boundary vertex out instead of merging along it; alsoVerts
    (edges, faces; default true) dissolves the vertices left with two edges. Reply: regionFaces (the merged faces, for
    edges and faces) and counts before and after - a dissolve that changes nothing is refused, and so is an edge
    without two faces (a boundary or wire edge): it has nothing to merge, and 4.4 dissolves a corner vertex in its place.

    mode LIMITED is Blender's Limited Dissolve (bmesh.ops.dissolve_limit): every edge between two faces that meet
    flatter than angle (DEGREES, default 5) and every vertex between two edges that run straighter than it go, over the
    whole mesh or the elements given (indices with elements; faces bring their edges and vertices). delimit keeps
    edges that separate NORMAL, MATERIAL, SEAM, SHARP or UV regions (this build's list; default the build's own,
    NORMAL), dissolveBoundaries (default false) dissolves an open rim too. The n-gons it merges are then split into
    triangles, the surface unchanged, so the result passes mesh_quality gameReady; keepNgons true leaves them, for a mesh
    still being modeled. Reply: facesDissolved, verticesDissolved, regionFaces, ngonsSplit, angleRadians and the counts;
    a result with no face left is refused.
    """
    reject_unknown(params, _DISSOLVE_KEYS, "dissolve_mesh_elements")
    obj = _target(params, "dissolve_mesh_elements")
    kind = str(take(params, "elements", default="EDGES", kind=str)).upper()
    if kind not in ("VERTS", "EDGES", "FACES"):
        raise MifOpError("elements must be VERTS, EDGES or FACES, got %r. NOTHING was changed." % kind)
    mode = str(take(params, "mode", default="NAMED", kind=str)).upper()
    if mode not in ("NAMED", "LIMITED"):
        raise MifOpError("mode is NAMED (the indices given) or LIMITED (by angle), got %r. NOTHING was changed." % mode)
    wrong = [k for k in (_NAMED_ONLY if mode == "LIMITED" else _LIMITED_ONLY) if params.get(k) is not None]
    if wrong:
        raise MifOpError("%s %s for mode %s. NOTHING was changed."
                         % (", ".join(wrong), "are" if len(wrong) > 1 else "is",
                            "NAMED, not LIMITED" if mode == "LIMITED" else "LIMITED, not NAMED"))
    if mode == "LIMITED":
        return _dissolve_limited(params, obj, kind)
    split = take_bool(params, "useFaceSplit", default=False)
    tear = take_bool(params, "useBoundaryTear", default=False)
    also = take_bool(params, "alsoVerts", default=True)
    if tear and kind != "VERTS":
        raise MifOpError("useBoundaryTear is for VERTS. NOTHING was changed.")
    if split and kind == "FACES":
        raise MifOpError("useFaceSplit is for VERTS and EDGES. NOTHING was changed.")
    if params.get("alsoVerts") is not None and kind == "VERTS":
        raise MifOpError("alsoVerts is for EDGES and FACES. NOTHING was changed.")
    before = mesh_counts(obj)
    bm = _open(obj)
    try:
        table = {"VERTS": bm.verts, "EDGES": bm.edges, "FACES": bm.faces}[kind]
        items = _elements(params, "indices", table, kind.lower()[:-1], obj)
        start = (len(bm.verts), len(bm.edges), len(bm.faces))    # off the scratch bmesh, not the reply's counts
        if kind == "EDGES":
            # AN EDGE WITHOUT TWO FACES HAS NOTHING TO MERGE, and what Blender does with one differs by build - measured
            # 2026-10-04 on a 3x3 sheet's corner rim edge: 3.6.23 and 5.2.1 change nothing; 4.4.0 dissolves the corner
            # vertex instead (16 -> 15 vertices, the quad a triangle) and reports success. So it is refused up front.
            single = [e.index for e in items if len(e.link_faces) != 2]
            if single:
                raise MifOpError("edges %s do not join two faces (a boundary or a wire edge), so dissolving them merges "
                                 "nothing - and Blender 4.4 would dissolve a corner vertex instead. NOTHING was changed."
                                 % single[:10])
        region = []
        if kind == "VERTS":
            bmesh.ops.dissolve_verts(bm, verts=items, use_face_split=split, use_boundary_tear=tear)
        elif kind == "EDGES":
            region = list(bmesh.ops.dissolve_edges(bm, edges=items, use_verts=also, use_face_split=split)["region"])
        else:
            region = list(bmesh.ops.dissolve_faces(bm, faces=items, use_verts=also)["region"])
        if (len(bm.verts), len(bm.edges), len(bm.faces)) == start:
            raise MifOpError("the dissolve changed nothing - Blender keeps an edge between two faces that cannot merge "
                             "(a boundary edge, or one on the mesh's only face). NOTHING was changed.")
        bm.normal_update()
        _commit(bm, obj)
        reg = _idx(region)
    finally:
        bm.free()
    out = {"object": obj.name, "elements": kind, "regionFaces": reg if kind != "VERTS" else None,
           "before": before, "after": mesh_counts(obj)}
    out.update(shared_data_note(obj))
    return out


_FILL_KEYS = ("object", "name", "vertices", "edges", "loopNear", "mode", "span", "offset")


def op_fill_mesh_loop(params):
    """Make a face from chosen vertices, or fill an open loop: one n-gon (NGON), triangles (TRIANGLES, beauty-shaped) or
    a grid of quads (GRID - a subdivision-surface model stays quads).

    Name what to fill by vertices (indices: NGON makes one face through them, in the order the mesh connects them),
    edges (the loop's edge indices) or loopNear [x,y,z] (world: the open boundary loop nearest). GRID splits a closed
    loop of an even vertex count into two opposite rails of `span` edges (default a quarter of the loop, rounded) from
    vertex `offset` along it and fills between them with quads; a loop of 2 x (span + m) vertices gets span x m quads.
    The new faces take their material and smooth flag from the neighbouring faces, and each corner's UV from that
    vertex's UV in a neighbouring face. Reply: the new face indices and the boundary edge count before and after.
    """
    reject_unknown(params, _FILL_KEYS, "fill_mesh_loop")
    obj = _target(params, "fill_mesh_loop")
    mode = str(take(params, "mode", default="NGON", kind=str)).upper()
    if mode not in ("NGON", "TRIANGLES", "GRID"):
        raise MifOpError("mode must be NGON, TRIANGLES or GRID, got %r. NOTHING was changed." % mode)
    given = [k for k in ("vertices", "edges", "loopNear") if params.get(k) is not None]
    if len(given) != 1:
        raise MifOpError("pass exactly one of vertices, edges or loopNear (got %s). NOTHING was changed."
                         % (", ".join(given) or "none"))
    if given[0] == "vertices" and mode != "NGON":
        raise MifOpError("vertices make one NGON face; TRIANGLES and GRID fill a loop (edges or loopNear). NOTHING was "
                         "changed.")
    span = take_int(params, "span", default=None)
    offset = take_int(params, "offset", default=0)
    if mode != "GRID" and (params.get("span") is not None or params.get("offset") is not None):
        raise MifOpError("span and offset are for GRID. NOTHING was changed.")
    before = mesh_counts(obj)
    bm = _open(obj)
    try:
        bnd_before = sum(1 for e in bm.edges if len(e.link_faces) == 1)
        if given[0] == "vertices":
            verts = _elements(params, "vertices", bm.verts, "vertex", obj)
            if len(verts) < 3:
                raise MifOpError("a face needs at least 3 vertices. NOTHING was changed.")
            edges = None
        else:
            if given[0] == "edges":
                edges = _elements(params, "edges", bm.edges, "edge", obj)
            else:
                loops = _boundary_loops_of(bm)
                if not loops:
                    raise MifOpError("'%s' has no open boundary loop to fill. NOTHING was changed." % obj.name)
                p = obj.matrix_world.inverted_safe() @ _vec3(params, "loopNear")
                edges = list(min(loops, key=lambda lp: min((x.co - p).length for x in lp[1]))[0])
            if any(len(e.link_faces) > 1 for e in edges):
                raise MifOpError("an edge given already has two faces - a filled loop's edges are open. NOTHING was "
                                 "changed.")
        if mode == "NGON":
            geom = verts if edges is None else edges
            r = bmesh.ops.contextual_create(bm, geom=list(geom))
            made = list(r["faces"])
        elif mode == "TRIANGLES":
            r = bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=False, edges=edges)
            made = [g for g in r["geom"] if isinstance(g, bmesh.types.BMFace)]
        else:
            ring = _ordered((edges, list(dict.fromkeys(v for e in edges for v in e.verts))))
            if ring is None:
                raise MifOpError("GRID needs one closed loop. NOTHING was changed.")
            n = len(ring)
            if n % 2 or n < 4:
                raise MifOpError("GRID needs a loop with an even vertex count of at least 4; this one has %d. NOTHING was "
                                 "changed." % n)
            if span is None:
                span = max(1, round(n / 4))
            m = n // 2 - span
            if not 1 <= span or m < 1:
                raise MifOpError("span %d leaves no room for the other two sides on a loop of %d vertices (span must be "
                                 "1 to %d). NOTHING was changed." % (span, n, n // 2 - 1))
            o = offset % n

            def rail(start, length):
                vs = [ring[(start + i) % n] for i in range(length + 1)]
                return [next(e for e in a.link_edges if e.other_vert(a) is b) for a, b in zip(vs, vs[1:])]
            r = bmesh.ops.grid_fill(bm, edges=rail(o, span) + rail(o + span + m, span), use_interp_simple=True)
            made = list(r["faces"])
        if not made:
            raise MifOpError("the fill made no face (the loop may be open or the vertices share no plane Blender can "
                             "fill). NOTHING was changed.")
        uv = bm.loops.layers.uv.active
        for f in made:
            nbr = [g for e in f.edges for g in e.link_faces if g not in made]
            if nbr:
                f.material_index = nbr[0].material_index
                f.smooth = nbr[0].smooth
            if uv is not None and mode != "GRID":
                for lp in f.loops:
                    other = [x for x in lp.vert.link_loops if x.face not in made]
                    if other:
                        lp[uv].uv = other[0][uv].uv.copy()
        bm.normal_update()
        bnd_after = sum(1 for e in bm.edges if len(e.link_faces) == 1)
        _commit(bm, obj)
        new_f = _idx(made)
    finally:
        bm.free()
    out = {"object": obj.name, "mode": mode, "newFaces": new_f, "boundaryEdgesBefore": bnd_before,
           "boundaryEdgesAfter": bnd_after, "before": before, "after": mesh_counts(obj)}
    if mode == "GRID":
        out["span"] = span
        out["offset"] = offset
    out.update(shared_data_note(obj))
    return out


_MERGE_KEYS = ("object", "name", "vertices", "pairs", "at", "distance")


def op_merge_vertices(params):
    """Merge chosen vertices: all of them into one, pair by pair, or those closer than a distance.

    vertices [i, ...] alone collapses them into ONE vertex; pairs [[a, b], ...] merges each pair; vertices with
    distance merges, among those vertices only, every two closer than distance (mesh units) - a merge by distance kept to
    a selection, where clean_mesh mergeDistance welds the whole mesh. at says where a merged vertex lands: CENTER (the
    default: the mean of what merged), FIRST or LAST (the first or last vertex named; for a pair, a or b); a merge by
    distance keeps Blender's own choice and takes no `at`. Faces that collapse to nothing are removed by the merge.
    Reply: for each group (or pair) the vertex it became and where it landed, counts before and after.
    """
    reject_unknown(params, _MERGE_KEYS, "merge_vertices")
    obj = _target(params, "merge_vertices")
    at = str(take(params, "at", default="CENTER", kind=str)).upper()
    if at not in ("CENTER", "FIRST", "LAST"):
        raise MifOpError("at must be CENTER, FIRST or LAST, got %r. NOTHING was changed." % at)
    dist = take_float(params, "distance", default=None)
    given = [k for k in ("vertices", "pairs") if params.get(k) is not None]
    if len(given) != 1:
        raise MifOpError("pass vertices OR pairs (got %s). NOTHING was changed." % (", ".join(given) or "none"))
    if dist is not None and given[0] != "vertices":
        raise MifOpError("distance merges among vertices; pairs say exactly what merges. NOTHING was changed.")
    if dist is not None and params.get("at") is not None:
        raise MifOpError("a merge by distance keeps Blender's choice of vertex; at is for vertices or pairs. NOTHING "
                         "was changed.")
    if dist is not None and not dist > 0.0:
        raise MifOpError("distance must be above 0, got %r. NOTHING was changed." % dist)
    before = mesh_counts(obj)
    bm = _open(obj)
    try:
        groups = []
        if given[0] == "vertices":
            verts = _elements(params, "vertices", bm.verts, "vertex", obj)
            if len(verts) < 2:
                raise MifOpError("merging needs at least two vertices. NOTHING was changed.")
            groups = [verts]
        else:
            raw = params["pairs"]
            if not isinstance(raw, (list, tuple)) or not raw or any(
                    not isinstance(p, (list, tuple)) or len(p) != 2 for p in raw):
                raise MifOpError("pairs must be [[a, b], ...]. NOTHING was changed.")
            flat = [i for p in raw for i in p]
            bad = [i for i in flat if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(bm.verts)]
            if bad:
                raise MifOpError("%s are not vertex indices of '%s' (it has %d). NOTHING was changed."
                                 % (bad[:10], obj.name, len(bm.verts)))
            if len(set(flat)) != len(flat):
                raise MifOpError("a vertex is in two pairs (or paired with itself) - merge it once. NOTHING was "
                                 "changed.")
            groups = [[bm.verts[a], bm.verts[b]] for a, b in raw]
        landed = []
        if dist is not None:
            v0 = len(bm.verts)
            bmesh.ops.remove_doubles(bm, verts=groups[0], dist=dist)
            merged = v0 - len(bm.verts)
            if not merged:
                raise MifOpError("no two of those vertices are closer than %g, so nothing merged; the mesh was not written."
                                 % dist)
        else:
            for g in groups:
                if at == "CENTER":
                    target = sum((v.co for v in g), Vector()) / len(g)
                else:
                    target = (g[0] if at == "FIRST" else g[-1]).co.copy()
                keep = g[0] if at != "LAST" else g[-1]
                keep.co = target
                bmesh.ops.weld_verts(bm, targetmap={v: keep for v in g if v is not keep})
                landed.append((keep, target))
            merged = sum(len(g) - 1 for g in groups)
        bm.normal_update()
        _commit(bm, obj)
        rows = [{"vertex": v.index, "at": [round(x, 6) for x in p]} for v, p in landed if v.is_valid]
    finally:
        bm.free()
    out = {"object": obj.name, "mode": "distance" if dist is not None else given[0], "at": None if dist else at,
           "verticesMerged": merged, "merged": rows, "before": before, "after": mesh_counts(obj)}
    out.update(shared_data_note(obj))
    return out


_RELAX_KEYS = ("object", "name", "vertices", "mode", "iterations", "interpolation", "onSurface")


def _ordered_run(verts):
    """The vertices as one run along the mesh's edges: (ordered list, closed). Refused unless they form one simple
    path or one simple loop."""
    members = {v.index: v for v in verts}
    nbrs = {i: [e.other_vert(v).index for e in v.link_edges if e.other_vert(v).index in members]
            for i, v in members.items()}
    if any(len(n) > 2 for n in nbrs.values()):
        bad = sorted(i for i, n in nbrs.items() if len(n) > 2)
        raise MifOpError("vertices %s join more than two others of the run, so it branches - name one path or one loop. "
                         "NOTHING was changed." % bad[:10])
    ends = [i for i, n in nbrs.items() if len(n) < 2]
    closed = not ends
    start = ends[0] if ends else min(members)
    order, seen, cur = [start], {start}, start
    while True:
        nxt = sorted(n for n in nbrs[cur] if n not in seen)
        if not nxt:
            break
        cur = nxt[0]
        order.append(cur)
        seen.add(cur)
    if len(order) != len(members):
        raise MifOpError("the vertices are not one connected run along edges (%d of %d reached from vertex %d). NOTHING "
                         "was changed." % (len(order), len(members), start))
    return [members[i] for i in order], closed


def _run_lengths(pts, closed):
    pairs = list(zip(pts, pts[1:] + (pts[:1] if closed else [])))
    lens = [(a - b).length for a, b in pairs]
    return {"max": round(max(lens), 6), "mean": round(sum(lens) / len(lens), 6), "min": round(min(lens), 6)}


def _catmull(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2.0 * p1) + (p2 - p0) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                  + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3)


def _spaced(pts, closed, cubic):
    """The run's points moved to equal arc lengths along it - along the polyline (LINEAR) or along a Catmull-Rom curve
    through the points (CUBIC, sampled 64 times a span). Ends of an open run stay."""
    n = len(pts)
    ring = pts + [pts[0]] if closed else list(pts)
    if cubic:
        dense = []
        spans = len(ring) - 1
        for i in range(spans):
            if closed:
                p0, p1, p2, p3 = pts[(i - 1) % n], pts[i % n], pts[(i + 1) % n], pts[(i + 2) % n]
            else:
                p0 = pts[i - 1] if i > 0 else 2.0 * pts[0] - pts[1]
                p1, p2 = pts[i], pts[i + 1]
                p3 = pts[i + 2] if i + 2 < n else 2.0 * pts[-1] - pts[-2]
            for k in range(64):
                dense.append(_catmull(p0, p1, p2, p3, k / 64.0))
        dense.append(ring[-1].copy())
    else:
        dense = ring
    cum = [0.0]
    for a, b in zip(dense, dense[1:]):
        cum.append(cum[-1] + (b - a).length)
    total = cum[-1]
    count = n if closed else n - 1
    out, j = [], 0
    for i in range(n):
        s = total * i / count
        while j < len(cum) - 2 and cum[j + 1] < s:
            j += 1
        seg = cum[j + 1] - cum[j]
        t = (s - cum[j]) / seg if seg > 0.0 else 0.0
        out.append(dense[j].lerp(dense[j + 1], min(1.0, max(0.0, t))))
    if not closed:
        out[0], out[-1] = pts[0].copy(), pts[-1].copy()
    return out


def _flatten(params, obj):
    import numpy as np   # noqa: PLC0415
    for key in ("interpolation", "iterations"):
        if params.get(key) is not None:
            raise MifOpError("%s shapes SPACE or RELAX; FLATTEN projects onto one plane. NOTHING was changed." % key)
    on_surface = take_bool(params, "onSurface", default=False)
    if on_surface:
        raise MifOpError("onSurface puts a run back on the surface; FLATTEN takes the vertices off it on purpose. "
                         "NOTHING was changed.")
    bm = _open(obj)
    try:
        verts = _elements(params, "vertices", bm.verts, "vertex", obj)
        if len(verts) < 3:
            raise MifOpError("a plane needs at least three vertices. NOTHING was changed.")
        pts = np.array([list(v.co) for v in verts])
        c = pts.mean(axis=0)
        _w, vecs = np.linalg.eigh(np.cov((pts - c).T))
        nrm = Vector(vecs[:, 0]).normalized()
        cen = Vector(c)
        disp = 0.0
        for v in verts:
            d = (v.co - cen).dot(nrm)
            disp = max(disp, abs(d))
            v.co = v.co - nrm * d
        bm.normal_update()
        _commit(bm, obj)
    finally:
        bm.free()
    out = {"object": obj.name, "mode": "FLATTEN", "verticesMoved": len(verts), "maxDisplacement": round(disp, 9),
           "plane": {"point": [round(x, 6) for x in cen], "normal": [round(x, 6) for x in nrm]},
           "space": "LOCAL"}
    out.update(shared_data_note(obj))
    return out


def op_relax_loop(params):
    """Even out the spacing of a run of vertices, or relax it, along the mesh's edges.

    vertices: the run's indices in any order - they must form ONE path or ONE closed loop along edges. mode SPACE
    (default) moves them to equal distances along the run: interpolation LINEAR keeps them on the run's own segments,
    CUBIC on a smooth curve through them (Catmull-Rom); an open run's two ends stay. mode RELAX moves each vertex
    halfway to the midpoint of its two neighbours on the run, `iterations` times (default 1; an open run's ends stay).
    onSurface (default false) puts every moved vertex back on the mesh's surface as it was before the call (the nearest
    point), so a run relaxed across a curved panel stays on it. Reply: the order walked, closed, and the max, mean and
    min edge length along the run before and after.

    mode FLATTEN (2026-10-06; 5.2's mesh.flatten, native here on every build) projects the vertices - any set, not a
    run - onto their least-squares plane: reply plane {point, normal} and maxDisplacement. It takes no interpolation
    or iterations.
    """
    reject_unknown(params, _RELAX_KEYS, "relax_loop")
    obj = _target(params, "relax_loop")
    mode = str(take(params, "mode", default="SPACE", kind=str)).upper()
    if mode not in ("SPACE", "RELAX", "FLATTEN"):
        raise MifOpError("mode must be SPACE, RELAX or FLATTEN, got %r. NOTHING was changed." % mode)
    if mode == "FLATTEN":
        return _flatten(params, obj)
    interp = str(take(params, "interpolation", default="LINEAR", kind=str)).upper()
    if interp not in ("LINEAR", "CUBIC"):
        raise MifOpError("interpolation must be LINEAR or CUBIC, got %r. NOTHING was changed." % interp)
    if mode == "RELAX" and params.get("interpolation") is not None:
        raise MifOpError("interpolation shapes SPACE; RELAX averages neighbours. NOTHING was changed.")
    iters = take_int(params, "iterations", default=1)
    if mode == "SPACE" and params.get("iterations") is not None:
        raise MifOpError("iterations repeats RELAX; SPACE places the vertices once. NOTHING was changed.")
    if not 1 <= iters <= 500:
        raise MifOpError("iterations must be 1-500, got %d. NOTHING was changed." % iters)
    on_surface = take_bool(params, "onSurface", default=False)
    bm = _open(obj)
    try:
        verts = _elements(params, "vertices", bm.verts, "vertex", obj)
        if len(verts) < 3:
            raise MifOpError("a run needs at least three vertices - two ends and one to move. NOTHING was changed.")
        run, closed = _ordered_run(verts)
        pts = [v.co.copy() for v in run]
        before = _run_lengths(pts, closed)
        if mode == "SPACE":
            new = _spaced(pts, closed, interp == "CUBIC")
        else:
            new = [p.copy() for p in pts]
            n = len(new)
            for _ in range(iters):
                nxt = [p.copy() for p in new]
                for i in range(n):
                    if not closed and i in (0, n - 1):
                        continue
                    a, b = new[(i - 1) % n], new[(i + 1) % n]
                    nxt[i] = new[i] * 0.5 + (a + b) * 0.25
                new = nxt
        if on_surface:
            from mathutils.bvhtree import BVHTree   # noqa: PLC0415 - the offline suites stub mathutils
            tree = BVHTree.FromBMesh(bm)
            for i, p in enumerate(new):
                hit = tree.find_nearest(p)
                if hit[0] is not None:
                    new[i] = hit[0]
        moved = 0
        for v, p in zip(run, new):
            if (v.co - p).length > 1e-9:
                moved += 1
            v.co = p
        bm.normal_update()
        order = [v.index for v in run]
        _commit(bm, obj)
        after = _run_lengths([v.co.copy() for v in run], closed)
    finally:
        bm.free()
    out = {"object": obj.name, "mode": mode, "interpolation": interp if mode == "SPACE" else None,
           "iterations": iters if mode == "RELAX" else None, "onSurface": on_surface, "order": order, "closed": closed,
           "verticesMoved": moved, "edgeLengthBefore": before, "edgeLengthAfter": after}
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- offset edges

_OFFSET_KEYS = ("object", "name", "edges", "distance", "sideFace", "space", "clampOverlap")


def _edge_walk(edges):
    """The edges as one walk: (vertices in order, closed). Refused when they branch or fall in pieces."""
    adj = {}
    for e in edges:
        a, b = e.verts
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    branch = [v.index for v, n in adj.items() if len(n) > 2]
    if branch:
        raise MifOpError("the edges branch at vertices %s - offset_edges takes one loop or one open chain. NOTHING was "
                         "changed." % branch[:10])
    ends = [v for v, n in adj.items() if len(n) == 1]
    start = ends[0] if ends else min(adj, key=lambda v: v.index)
    order, prev, cur, closed = [start], None, start, False
    while True:
        nxt = [n for n in adj[cur] if n is not prev]
        if not nxt:
            break
        if nxt[0] is start:
            closed = True
            break
        order.append(nxt[0])
        prev, cur = cur, nxt[0]
    if len(order) != len(adj):
        raise MifOpError("the edges are not one connected loop or chain (%d of their %d vertices are reached from vertex "
                         "%d). NOTHING was changed." % (len(order), len(adj), start.index))
    return order, closed


class _LoopAt:
    """Where a position sits on the loop: its walk index, or None. By nearest within tol, not by exact position - the
    bevel rebuilds the loop's vertices, and a rebuilt one can differ from the old in the last bits."""

    def __init__(self, order, tol):
        from mathutils.kdtree import KDTree   # noqa: PLC0415 - the offline suites stub mathutils
        self.tol = tol
        self.kd = KDTree(len(order))
        for i, v in enumerate(order):
            self.kd.insert(v.co, i)
        self.kd.balance()

    def get(self, co):
        _co, i, dist = self.kd.find(co)
        return i if i is not None and dist <= self.tol else None

    def twins(self, co):
        return [i for _co, i, _d in self.kd.find_range(co, self.tol)]


def _side(face_loops, at, n, closed):
    """LEFT when the face walks its edge on the loop in the loop's own order (vertex k to k + 1), else RIGHT."""
    for lp in face_loops:
        a, b = at.get(lp.vert.co), at.get(lp.link_loop_next.vert.co)
        if a is None or b is None:
            continue
        if b == a + 1 or (closed and a == n - 1 and b == 0):
            return "LEFT"
        if a == b + 1 or (closed and b == n - 1 and a == 0):
            return "RIGHT"
    return None


def op_offset_edges(params):
    """A buffer loop beside an edge loop, `distance` away across each face it touches - the room a bevel needs, carved
    round a junction a Boolean left in n-gons and triangles.

    edges is one closed loop or one open chain (select_edges seedEdge gives one). The route is Blender's bevel on those
    edges with two segments and a square profile, so the middle ring IS the loop, in place, and the new rings sit
    distance from it along the surface: measured 2026-10-05 on 3.6.23, 4.4.0 and 5.2.1 round the 56-edge junction of a
    cylinder unioned onto a sphere (104 n-gon sides), every junction vertex kept and both rings 0.050 from it at
    distance 0.05. A round profile moved the junction; bmesh's offset_edgeloops only stacks new loops on the old one;
    the editor's Offset Edge Slide needs a 3D view, and crashed 4.4.0 run without one. clampOverlap (default true) is
    the bevel's: a distance the neighbouring faces cannot hold is shortened for the whole loop, and each ring's distance
    and clampNote say by how much - on that junction, merged by distance first, 0.05 came out 0.034. Off, the rings sit
    at the distance asked and may cross the faces beyond.

    sideFace (a face touching the loop, closed loops only) keeps the buffer on that face's side alone: the other ring is
    welded back onto the loop, which put that side back on the surface to 1e-6. Lengths in mesh units, or world units
    with space WORLD (uniform scale only). Reply: loopVertices in walk order, rings (side LEFT or RIGHT of that walk,
    their edges and distance min, median, max), bufferFaces, maxOffSurface (how far any vertex now sits from the mesh
    as it was) and counts before and after.
    """
    reject_unknown(params, _OFFSET_KEYS, "offset_edges")
    obj = _target(params, "offset_edges")
    space = _space(params)
    distance = take_float(params, "distance", required=True)
    if not distance > 0.0:
        raise MifOpError("distance must be above 0, got %r. NOTHING was changed." % distance)
    d = _length(obj, distance, space, "distance")
    unit = 1.0 if space == "LOCAL" else _uniform_scale(obj)
    side_face = take_int(params, "sideFace", default=None)
    clamp = take_bool(params, "clampOverlap", default=True)
    before = mesh_counts(obj)
    bm = _open(obj)
    try:
        edges = _elements(params, "edges", bm.edges, "edge", obj)
        lone = [e.index for e in edges if len(e.link_faces) != 2]
        if lone:
            raise MifOpError("edges %s do not join two faces (a boundary or a wire edge) - a buffer goes into the faces "
                             "on both sides of the loop. NOTHING was changed." % lone[:10])
        order, closed = _edge_walk(edges)
        at = _LoopAt(order, d * 1e-3)
        for i, v in enumerate(order):
            twins = [j for j in at.twins(v.co) if j != i]
            if twins:
                raise MifOpError("vertices %d and %d of the loop share a position (closer than %g), so the loop cannot be "
                                 "followed through the bevel - clean_mesh mergeDistance or merge_vertices first. NOTHING "
                                 "was changed." % (order[twins[0]].index, v.index, d * 1e-3))
        keep = None
        if side_face is not None:
            if not closed:
                raise MifOpError("sideFace is for a closed loop - keeping one side was measured on closed loops only, "
                                 "and an open chain's two sides meet round its ends. NOTHING was changed.")
            if not 0 <= side_face < len(bm.faces):
                raise MifOpError("sideFace %d is not a face index of '%s' (it has %d). NOTHING was changed."
                                 % (side_face, obj.name, len(bm.faces)))
            f = bm.faces[side_face]
            if not set(f.edges) & set(edges):
                raise MifOpError("face %d does not touch the loop - sideFace names a face on one side of it. NOTHING "
                                 "was changed." % side_face)
            keep = _side(f.loops, at, len(order), closed)
        from mathutils.bvhtree import BVHTree   # noqa: PLC0415 - the offline suites stub mathutils
        bvh = BVHTree.FromBMesh(bm)
        r = bmesh.ops.bevel(bm, geom=list(edges) + list(order), offset=d, offset_type="OFFSET", segments=2, profile=1.0,
                            affect="EDGES", clamp_overlap=clamp, loop_slide=True)
        strip = [f for f in r["faces"] if f.is_valid]
        mid = {v for f in strip for v in f.verts if at.get(v.co) is not None}
        if not strip or len(mid) != len(order):
            raise MifOpError("the bevel did not keep the loop in place (%d of its %d vertices found after it) - an "
                             "open chain's ends move. NOTHING was changed." % (len(mid), len(order)))
        rung, side_of = {}, {}
        for f in strip:
            side = _side(f.loops, at, len(order), closed)
            for e in f.edges:
                a, b = e.verts
                if (a in mid) != (b in mid):
                    ring, j = (b, a) if a in mid else (a, b)
                    rung[ring] = j
                    if side is not None:
                        side_of[ring] = side
        if keep is not None:
            bmesh.ops.weld_verts(bm, targetmap={v: j for v, j in rung.items() if side_of.get(v) != keep})
        rings = {}
        for v, j in rung.items():
            if v.is_valid:
                rings.setdefault(side_of.get(v, "END"), []).append((v, (v.co - j.co).length))
        ring_verts = {v for rows in rings.values() for v, _l in rows}
        off = max((bvh.find_nearest(v.co)[3] or 0.0) for v in ring_verts) if ring_verts else 0.0
        bm.normal_update()
        _commit(bm, obj)
        out_rings = []
        for side in sorted(rings):
            vs = {v for v, _l in rings[side]}
            ls = sorted(length * unit for _v, length in rings[side])
            out_rings.append({"side": side, "edges": sorted(e.index for e in bm.edges if e.verts[0] in vs
                                                             and e.verts[1] in vs),
                              "distance": {"min": round(ls[0], 6), "median": round(ls[len(ls) // 2], 6),
                                           "max": round(ls[-1], 6)}})
        buffer_faces = _idx(f for f in strip if f.is_valid)
        loop_after = [v.index for v in sorted(mid, key=lambda v: at.get(v.co))]
    finally:
        bm.free()
    out = {"object": obj.name, "distance": distance, "space": space, "closed": closed, "clampOverlap": clamp,
           "keptSide": keep or "BOTH", "loopVertices": loop_after, "rings": out_rings,
           "bufferFaces": buffer_faces, "maxOffSurface": round(off * unit, 6),
           "before": before, "after": mesh_counts(obj)}
    short = min((r_["distance"]["min"] for r_ in out_rings), default=None)
    if short is not None and short < 0.98 * distance:
        out["clampNote"] = ("the neighbouring faces could not hold %g everywhere: the nearest ring vertex is %g from the "
                            "loop (bevel clamp_overlap)" % (distance, short))
    out.update(shared_data_note(obj))
    return out


_SUBD_KEYS = ("object", "name", "faces", "edges", "all", "cuts", "smoothness", "quadCorner", "fractal", "seed")
# Blender's Subdivide operator spells one corner type INNERVERT; bmesh.ops.subdivide_edges spells it INNER_VERT.
_SUBD_CORNERS = {"STRAIGHT_CUT": "STRAIGHT_CUT", "INNER_VERT": "INNER_VERT", "INNERVERT": "INNER_VERT", "PATH": "PATH",
                 "FAN": "FAN"}


def op_subdivide_mesh(params):
    """Cut chosen faces or edges - or the whole mesh - into `cuts` + 1 pieces per edge: Blender's Subdivide.

    THE GAP, 2026-10-04: a SIMPLE Subdivision Surface plus apply_modifier doubles every edge of the whole mesh and
    nothing else, so a plane cut 25 times for a half pattern (26 segments) or a region cut finer had no call.

    faces (indices: every edge of those faces is cut), edges (indices) or all true pick what is cut; cuts (1-100,
    default 1) is how many new vertices go on each edge. bmesh.ops.subdivide_edges runs with the settings Blender's own
    Subdivide operator (mesh.subdivide) uses: grid fill on, a quad with two neighbouring edges cut split by quadCorner
    (STRAIGHT_CUT, the operator's default, INNER_VERT - also spelled INNERVERT - PATH or FAN), linear smoothing falloff.
    smoothness (0-1) moves the new vertices toward the surface the old ones imply - a cube rounds out - and fractal
    (with seed) displaces them at random. A face next to the cut region gains the new vertex on its shared edge (it is
    not split: it gains the new vertex on its edge). Reply: counts before and after, edgesCut, newVertices and the faces
    of the cut region (regionFaces, in the mesh's order after the write).
    """
    reject_unknown(params, _SUBD_KEYS, "subdivide_mesh")
    obj = _target(params, "subdivide_mesh")
    given = [k for k in ("faces", "edges", "all") if params.get(k) is not None]
    if len(given) != 1:
        raise MifOpError("pass exactly one of faces, edges or all (got %s). NOTHING was changed."
                         % (", ".join(given) or "none"))
    want_all = take_bool(params, "all") if params.get("all") is not None else None   # read before any condition
    if given[0] == "all" and want_all is not True:
        raise MifOpError("all takes true; pick faces or edges instead. NOTHING was changed.")
    cuts = take_int(params, "cuts", default=1)
    if not 1 <= cuts <= 100:
        raise MifOpError("cuts must be 1-100 (new vertices per edge), got %d. NOTHING was changed." % cuts)
    smooth = take_float(params, "smoothness", default=0.0)
    if not 0.0 <= smooth <= 1.0:
        raise MifOpError("smoothness must be 0-1, got %r. NOTHING was changed." % smooth)
    fractal = take_float(params, "fractal", default=0.0)
    if fractal < 0.0:
        raise MifOpError("fractal must be 0 or more, got %r. NOTHING was changed." % fractal)
    seed = take_int(params, "seed", default=0)
    corner_raw = str(take(params, "quadCorner", default="STRAIGHT_CUT", kind=str)).upper()
    if corner_raw not in _SUBD_CORNERS:
        raise MifOpError("quadCorner must be one of STRAIGHT_CUT, INNER_VERT (INNERVERT), PATH or FAN, got %r. NOTHING "
                         "was changed." % corner_raw)
    before = mesh_counts(obj)
    bm = _open(obj)
    try:
        if given[0] == "faces":
            faces = _elements(params, "faces", bm.faces, "face", obj)
            edges = list(dict.fromkeys(e for f in faces for e in f.edges))
        elif given[0] == "edges":
            edges = _elements(params, "edges", bm.edges, "edge", obj)
        else:
            edges = list(bm.edges)
        if not edges:
            raise MifOpError("'%s' has no edge to cut. NOTHING was changed." % obj.name)
        cut = set(edges)
        full = [f for f in bm.faces if all(e in cut for e in f.edges)]   # faces wholly inside the cut region
        v0 = len(bm.verts)
        r = bmesh.ops.subdivide_edges(bm, edges=edges, cuts=cuts, smooth=smooth, smooth_falloff="LINEAR",
                                      fractal=fractal, seed=seed, quad_corner_type=_SUBD_CORNERS[corner_raw],
                                      use_grid_fill=True, use_single_edge=False, use_only_quads=False,
                                      use_sphere=False, use_smooth_even=False)
        # THE REGION: the faces subdivide_edges made plus the ones it reshaped in place - it keeps each old face as
        # one of its pieces, so geom_inner alone was 3 of a quad's 4 (measured on 5.2.2).
        region = list(dict.fromkeys([g for g in r["geom_inner"] if isinstance(g, bmesh.types.BMFace)]
                                    + [f for f in full if f.is_valid]))
        new_v = len(bm.verts) - v0
        if not new_v:
            raise MifOpError("the subdivide added no vertex. NOTHING was changed.")
        bm.normal_update()
        _commit(bm, obj)
        region_idx = _idx(region)
    finally:
        bm.free()
    out = {"object": obj.name, "selection": given[0], "cuts": cuts, "smoothness": smooth, "quadCorner":
           _SUBD_CORNERS[corner_raw], "fractal": fractal, "edgesCut": len(edges), "newVertices": new_v,
           "regionFaces": region_idx[:200], "regionFacesTruncated": len(region_idx) > 200, "before": before,
           "after": mesh_counts(obj)}
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- sculpt_stroke
_SCULPT_KEYS = ("object", "name", "brush", "path", "radius", "strength", "falloff", "invert", "symmetry", "mask",
                "shapeKey", "spacing", "stabilize", "boundaryDeform", "dyntopo", "multiresLevel", "clothDeform",
                "deformTarget", "clothArea")
_SCULPT_BRUSHES = ("DRAW", "CLAY", "INFLATE", "SMOOTH", "FLATTEN", "CREASE", "GRAB", "SNAKE_HOOK", "BOUNDARY", "CLOTH",
                   # 1.3 gap audit (2026-10-19): the most-used form and character brushes
                   "CLAY_STRIPS", "SCRAPE", "FILL", "PINCH", "LAYER", "BLOB", "DRAW_SHARP", "NUDGE", "ELASTIC", "POSE")
_PATH_BRUSHES = ("NUDGE", "ELASTIC")     # move along the path, so two points at least; POSE takes three
_CLOTH_DEFORMS = ("DRAG", "PUSH", "GRAB", "PINCH_POINT", "INFLATE", "EXPAND")
_CLOTH_PASSES = 20      # edge-length constraint passes after each cloth dab (Gauss-Seidel, stiffness 1)
_BOUNDARY_DEFORMS = ("BEND", "EXPAND", "INFLATE", "GRAB", "SMOOTH")
_FACE_SET_ATTR = ".sculpt_face_set"      # Blender's own face-set attribute (INT, FACE domain)
_DYNTOPO_MAX_VERTS = 400000
_MULTIRES_MAX_VERTS = 2000000
# Blender's brush falloff curves as formulas of t = distance / radius (0 at the centre, 1 at the rim).
_SCULPT_FALLOFF = {
    "SMOOTH": lambda t: 1.0 - t * t * (3.0 - 2.0 * t),
    "SPHERE": lambda t: math.sqrt(max(0.0, 1.0 - t * t)),
    "ROOT": lambda t: 1.0 - math.sqrt(t),
    "SHARP": lambda t: (1.0 - t) ** 2,
    "LINEAR": lambda t: 1.0 - t,
    "CONSTANT": lambda t: 1.0,
}


def _sculpt_stabilize(pts, length):
    """Blender's stroke stabilization (lazy mouse) along a path: the dab trails the pointer on a string of `length` and
    moves only when the string is taut, so a dab is laid where the trailing point stops, never at a corner it cut."""
    d = pts[0].copy()
    out = [d.copy()]
    for p in pts[1:]:
        v = p - d
        if v.length > length:
            d = p - v.normalized() * length
            out.append(d.copy())
    return out


def _sculpt_dabs(path, symmetry, spacing, radius):
    """Every dab position (mesh space), the path resampled at spacing x radius when spacing is given, then mirrored."""
    pts = list(path)
    if spacing is not None and len(pts) > 1:
        step = max(spacing * radius, 1e-6)
        out = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            seg = (b - a).length
            n = max(1, int(math.ceil(seg / step)))
            out.extend(a.lerp(b, k / n) for k in range(1, n + 1))
        pts = out
    mirrored = []
    for axes in [(0, 0, 0)] + [m for m in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
                               if all(symmetry[i] or not m[i] for i in range(3))]:
        sign = Vector([-1.0 if axes[i] else 1.0 for i in range(3)])
        mirrored.append([Vector((q.x * sign.x, q.y * sign.y, q.z * sign.z)) for q in pts])
    return mirrored


def _dyntopo_refine(bm, centres, r_loc, detail):
    """Dynamic topology's detail half, in mesh space: the faces touching the brush are triangulated and every edge
    there longer than `detail` is split, pass after pass, until none is. Works on the scratch bmesh only; a refusal
    here leaves the mesh untouched. Returns (passes, the longest edge left in the region)."""
    r2 = r_loc * r_loc

    def near(v):
        return any((v.co - c).length_squared < r2 for c in centres)
    tri = [f for f in bm.faces if len(f.verts) > 3 and any(near(v) for v in f.verts)]
    if tri:
        bmesh.ops.triangulate(bm, faces=tri)
    passes = 0
    while True:
        long_edges = [e for e in bm.edges if e.calc_length() > detail and (near(e.verts[0]) or near(e.verts[1]))]
        if not long_edges:
            break
        passes += 1
        if passes > 24:
            raise MifOpError("dyntopo detailSize %g still had edges to split after 24 passes. NOTHING was changed."
                             % detail)
        r = bmesh.ops.subdivide_edges(bm, edges=long_edges, cuts=1, use_grid_fill=True, use_single_edge=False,
                                      use_only_quads=False, smooth=0.0)
        new_v = [g for g in r["geom_inner"] + r["geom_split"] if isinstance(g, bmesh.types.BMVert)]
        ngons = list({f for v in new_v for f in v.link_faces if len(f.verts) > 3})
        if ngons:
            bmesh.ops.triangulate(bm, faces=ngons)
        if len(bm.verts) > _DYNTOPO_MAX_VERTS:
            raise MifOpError("dyntopo detailSize %g would grow the mesh past %d vertices - raise detailSize or lower the "
                             "radius. NOTHING was changed." % (detail, _DYNTOPO_MAX_VERTS))
    longest = max([e.calc_length() for e in bm.edges if near(e.verts[0]) or near(e.verts[1])] or [0.0])
    return passes, longest


def _boundary_region(bm, co, centre, r_loc):
    """BOUNDARY brush's region: the open-boundary vertex nearest the dab (within the radius), the boundary vertices
    within the radius of it ALONG the boundary, and every vertex within the radius of those along the mesh's edges.
    Returns ({vertex: (distance from the boundary, its boundary origin)}, the boundary vertices) or (None, None)."""
    import heapq
    lim = r_loc * (1.0 + 1e-5)     # float32 grid steps sum a hair past the radius
    on_b = [v for v in bm.verts if any(e.is_boundary for e in v.link_edges)]
    on_b_idx = {v.index for v in on_b}
    near = [(((co[v.index] - centre).length), v.index) for v in on_b]
    near = [x for x in near if x[0] <= lim]
    if not near:
        return None, None
    start = min(near)[1]
    vs = bm.verts
    bdist = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        d, i = heapq.heappop(heap)
        if d > bdist.get(i, 1e30):
            continue
        for e in vs[i].link_edges:
            if not e.is_boundary:
                continue
            j = e.other_vert(vs[i]).index
            nd = d + (co[i] - co[j]).length
            if nd <= lim and nd < bdist.get(j, 1e30):
                bdist[j] = nd
                heapq.heappush(heap, (nd, j))
    region = {i: (0.0, i) for i in bdist}
    heap = [(0.0, i, i) for i in bdist]
    heapq.heapify(heap)
    while heap:
        d, i, o = heapq.heappop(heap)
        if d > region[i][0]:
            continue
        for e in vs[i].link_edges:
            j = e.other_vert(vs[i]).index
            if j in on_b_idx and j not in bdist:     # the boundary past the radius is not the region's to move
                continue
            nd = d + (co[i] - co[j]).length
            if nd <= lim and nd < region.get(j, (1e30,))[0]:
                region[j] = (nd, o)
                heapq.heappush(heap, (nd, j, o))
    return region, set(bdist)


def _boundary_dab(bm, co, vn, nbrs, centre, end, r_loc, mode, k_strength, sign, fall, mask_w):
    """One BOUNDARY dab: {vertex: new position}. GRAB moves by end - centre; the rest act at the dab."""
    region, bverts = _boundary_region(bm, co, centre, r_loc)
    if not region:
        return {}
    w = {}
    for i, (d, _o) in region.items():
        wi = fall(min(1.0, d / r_loc)) * mask_w(i)
        if wi > 0.0:
            w[i] = wi
    step = 0.25 * r_loc
    vs = bm.verts
    upd = {}
    if mode == "GRAB":
        return {i: co[i] + (end - centre) * wi for i, wi in w.items()}
    if mode == "INFLATE":
        return {i: co[i] + vn[i] * (sign * step * wi * k_strength) for i, wi in w.items()}
    if mode == "SMOOTH":
        return {i: co[i].lerp(sum((co[j] for j in nbrs[i]), Vector()) / len(nbrs[i]), min(1.0, wi * k_strength))
                for i, wi in w.items() if nbrs[i]}
    # EXPAND and BEND need each boundary vertex's frame: its outward direction in the surface, its tangent along the
    # boundary, and (BEND) the pivot - the farthest vertex its propagation reached, where the weight falls to 0.
    frame = {}
    for b in bverts:
        inner = [j for j in nbrs[b] if j not in bverts]
        mean_in = sum((co[j] for j in inner), Vector()) / len(inner) if inner else None
        out_dir = (co[b] - mean_in) if mean_in is not None else Vector()
        out_dir = out_dir - vn[b] * out_dir.dot(vn[b])
        out_dir = out_dir.normalized() if out_dir.length > 1e-12 else Vector()
        bn = [e.other_vert(vs[b]).index for e in vs[b].link_edges if e.is_boundary]
        tan = (co[bn[-1]] - co[bn[0]]) if len(bn) >= 2 else ((co[bn[0]] - co[b]) if bn else Vector())
        frame[b] = [out_dir, tan.normalized() if tan.length > 1e-12 else Vector(), co[b], -1.0]
    for i, (d, o) in region.items():
        if d > frame[o][3]:
            frame[o][2], frame[o][3] = co[i], d
    if mode == "EXPAND":
        return {i: co[i] + frame[region[i][1]][0] * (sign * step * wi * k_strength) for i, wi in w.items()}
    angle = sign * k_strength * math.pi / 2.0          # BEND: a full-strength dab turns the boundary 90 degrees
    axes = {}
    for b, (_out, tan, pivot, _d) in frame.items():
        if tan.length < 0.5:
            continue
        probe = Matrix.Rotation(angle, 3, tan) @ (co[b] - pivot) + pivot - co[b]
        axes[b] = tan if probe.dot(vn[b]) * sign >= 0.0 else -tan    # the boundary bends toward its normal
    for i, wi in w.items():
        o = region[i][1]
        if o not in axes:
            continue
        pivot = frame[o][2]
        upd[i] = Matrix.Rotation(angle * wi, 3, axes[o]) @ (co[i] - pivot) + pivot
    return upd


def _cloth_edges(bm):
    """The cloth's edges as (i, j) pairs, i < j, in an order read off the faces rather than Blender's edge table.

    WHY (found 2026-10-10 on 3.6.23): _cloth_relax is Gauss-Seidel, so its result depends on the order it visits the
    edges, and Blender's own edge order differs by build - a grid built from the same faces lists its edges row by row
    on 3.6.23 and face by face on 4.2.17 and 4.4.0 - so the same filter on the same grid grew a free grid by 2.26 on 3.6
    and 2.55 (the 1.05 a step it is asked for) on 4.x. Each face in order gives its closing edge (last corner to first),
    then its other edges corner by corner, each edge once: the order 4.x's own edge build gives a mesh made from faces,
    so a result measured there holds, and now on every build. Edges on no face follow in Blender's order."""
    out, seen = [], set()
    for f in bm.faces:
        vs = [v.index for v in f.verts]
        n = len(vs)
        for k in [n - 1] + list(range(n - 1)):
            a, b = vs[k], vs[(k + 1) % n]
            key = (a, b) if a < b else (b, a)
            if key not in seen:
                seen.add(key)
                out.append(key)
    for e in bm.edges:
        a, b = e.verts[0].index, e.verts[1].index
        key = (a, b) if a < b else (b, a)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _cloth_relax(co, edges, rest, free, passes):
    """Position-based cloth: each edge (i, j) pulled back toward its rest length, the correction split by the free
    weights (0 = pinned, 1 = free), passes times over the edges in order. co is changed in place."""
    live = [(i, j, r) for (i, j), r in zip(edges, rest) if free[i] > 0.0 or free[j] > 0.0]
    for _p in range(passes):
        for i, j, r in live:
            wi, wj = free[i], free[j]
            d = co[j] - co[i]
            ln = d.length
            if ln < 1e-12:
                continue
            corr = d * ((ln - r) / (ln * (wi + wj)))
            if wi > 0.0:
                co[i] = co[i] + corr * wi
            if wj > 0.0:
                co[j] = co[j] - corr * wj


def _cloth_dab(co, rest, edges, c, nxt, w, area, mask_w, vn_fn, cfg):
    """One CLOTH brush dab: the brush's displacement on the vertices under it (held by their weight), then the
    edge-length constraint over the simulation area (vertices within clothArea x radius; outside it, and where the
    mask is 0, they are pinned). EXPAND grows the rest lengths under the brush instead."""
    mode, r_loc, strength, sign = cfg["clothDeform"], cfg["r_loc"], cfg["strength"], cfg["sign"]
    steps, d = 1, None
    if mode in ("DRAG", "GRAB"):
        d = nxt - c
        steps = max(1, min(40, int(math.ceil(d.length / (0.1 * r_loc)))))
    if mode == "EXPAND":
        for e, (i, j) in enumerate(edges):
            if i in w and j in w:
                rest[e] *= 1.0 + 0.25 * strength * sign * 0.5 * (w[i] + w[j])
    vn = vn_fn() if mode in ("PUSH", "INFLATE") else None
    if mode == "PUSH":
        area_n = sum((vn[i] * wi for i, wi in w.items()), Vector())
        area_n = area_n.normalized() if area_n.length > 1e-12 else Vector((0.0, 0.0, 1.0))
    free = [0.0] * len(co)
    for i in area:
        free[i] = 1.0 if mask_w(i) > 0.0 else 0.0
    for i, wi in w.items():
        free[i] = max(0.0, 1.0 - wi)
    for _s in range(steps):
        for i, wi in w.items():
            k = wi * strength
            if mode in ("DRAG", "GRAB"):
                co[i] = co[i] + d * (wi / steps)
            elif mode == "PUSH":
                co[i] = co[i] + area_n * (sign * 0.25 * r_loc * k)
            elif mode == "INFLATE":
                co[i] = co[i] + vn[i] * (sign * 0.25 * r_loc * k)
            elif mode == "PINCH_POINT":
                co[i] = co[i] + (c - co[i]) * (0.5 * k * sign)
        _cloth_relax(co, edges, rest, free, _CLOTH_PASSES)


def _sculpt_apply(bm, cfg):
    """Run the brush over the scratch bmesh's vertex positions (or a shape-key layer). Returns (start, co, moved, n_dabs,
    the bmesh verts, the layer written). Nothing is written to the mesh here."""
    brush, r_loc, strength, fall, sign = cfg["brush"], cfg["r_loc"], cfg["strength"], cfg["fall"], cfg["sign"]
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    layer = bm.verts.layers.shape.get(cfg["shapeKey"]) if cfg["shapeKey"] else None
    deform = bm.verts.layers.deform.active
    verts = list(bm.verts)
    co = [Vector(v[layer]) if layer is not None else v.co.copy() for v in verts]
    start = [c.copy() for c in co]
    mw = None
    if cfg["groupIndex"] is not None:
        mw = [v[deform].get(cfg["groupIndex"], 0.0) if deform is not None else 0.0 for v in verts]
    if cfg["faceSets"] is not None:
        fs_layer = bm.faces.layers.int.get(_FACE_SET_ATTR)
        if fs_layer is None:
            raise MifOpError("the mesh sculpted has no face sets (%s). NOTHING was changed." % _FACE_SET_ATTR)
        want = set(cfg["faceSets"])
        in_set = [1.0 if any(f[fs_layer] in want for f in v.link_faces) else 0.0 for v in verts]
        mw = in_set if mw is None else [a * b for a, b in zip(mw, in_set)]

    def mask_w(i):
        return 1.0 if mw is None else mw[i]
    nbrs = [[e.other_vert(v).index for e in v.link_edges] for v in verts]

    def normals():
        out = []
        for v in verts:
            n = Vector()
            for f in v.link_faces:
                vs = [co[x.index] for x in f.verts]
                n += (vs[1] - vs[0]).cross(vs[2] - vs[0]) if len(vs) >= 3 else Vector()
            out.append(n.normalized() if n.length > 1e-12 else Vector((0.0, 0.0, 1.0)))
        return out

    def weights(c):
        out = {}
        for i, p in enumerate(co):
            d = (p - c).length
            if d < r_loc:
                w = fall(d / r_loc) * mask_w(i)
                if w > 0.0:
                    out[i] = w
        return out
    n_dabs = 0
    vn0 = normals() if brush == "DRAW_SHARP" else None
    edges = _cloth_edges(bm)
    rest = [(co[i] - co[j]).length for i, j in edges]
    area_r = cfg["clothArea"] * r_loc

    def area_of(cs):
        return {i for i, p in enumerate(co) if any((p - c).length < area_r for c in cs)}
    for dabs in cfg["dabSets"]:
        if brush == "CLOTH":
            mode = cfg["clothDeform"]
            pairs = [(dabs[0], dabs[-1])] if mode == "GRAB" else (
                list(zip(dabs, dabs[1:])) if mode == "DRAG" else [(c, c) for c in dabs])
            for c, nxt in pairs:
                n_dabs += 1
                w = weights(c)
                if w:
                    _cloth_dab(co, rest, edges, c, nxt, w, area_of([c, nxt]), mask_w, normals, cfg)
            continue
        if brush == "BOUNDARY":
            if cfg["boundaryDeform"] == "GRAB":
                pairs = [(dabs[0], dabs[-1])]
            else:
                pairs = [(c, c) for c in dabs]
            for c, end in pairs:
                n_dabs += 1
                upd = _boundary_dab(bm, co, normals(), nbrs, c, end, r_loc, cfg["boundaryDeform"], strength, sign,
                                    fall, mask_w)
                for i, q in upd.items():
                    co[i] = q
            continue
        if brush == "ELASTIC":
            # A KELVINLET-LIKE GRAB: no hard edge - every vertex moves by (last - first) x 1 / (1 + (d / r)^2)^1.5 from the
            # first point, so the surface stretches like rubber instead of tearing at the radius (strength scales it)
            d = dabs[-1] - dabs[0]
            for i, p in enumerate(co):
                t = (p - dabs[0]).length / r_loc
                wi = mask_w(i) * strength / (1.0 + t * t) ** 1.5
                if wi > 1e-6:
                    co[i] = co[i] + d * wi
            n_dabs += 1
            continue
        if brush == "POSE":
            # path [pivot, from, to]: the part beyond the pivot (along from - pivot, within |from - pivot| + radius of
            # the pivot) turns about the pivot by the rotation taking from to to; the weight rises from 0 at the pivot's
            # plane to 1 half a radius past it (smoothstep), so the joint bends instead of shearing
            pivot, frm, to = dabs[0], dabs[1], dabs[2]
            a, b = frm - pivot, to - pivot
            if a.length < 1e-9 or b.length < 1e-9:
                n_dabs += 1
                continue
            rot = a.rotation_difference(b)
            axis_dir = a.normalized()
            reach = a.length + r_loc
            for i, p in enumerate(co):
                rel = p - pivot
                h = rel.dot(axis_dir)
                if h <= 0.0 or rel.length > reach:
                    continue
                s = min(1.0, h / (0.5 * r_loc))
                wi = s * s * (3.0 - 2.0 * s) * mask_w(i)
                if wi <= 0.0:
                    continue
                target = rot @ rel + pivot
                co[i] = p.lerp(target, wi)
            n_dabs += 1
            continue
        if brush == "NUDGE":
            # each segment pushes the surface along the stroke, in the surface's own plane
            for a, b in zip(dabs, dabs[1:]):
                w = weights(a)
                if w:
                    vn = normals()
                    for i, wi in w.items():
                        d = b - a
                        d = d - vn[i] * d.dot(vn[i])
                        co[i] = co[i] + d * (wi * strength)
                n_dabs += 1
            continue
        if brush in ("GRAB", "SNAKE_HOOK"):
            if brush == "GRAB":
                w = weights(dabs[0])
                d = dabs[-1] - dabs[0]
                for i, wi in w.items():
                    co[i] = co[i] + d * wi
                n_dabs += 1
            else:
                for a, b in zip(dabs, dabs[1:]):
                    w = weights(a)
                    for i, wi in w.items():
                        co[i] = co[i] + (b - a) * wi
                    n_dabs += 1
            continue
        for c in dabs:
            n_dabs += 1
            w = weights(c)
            if not w:
                continue
            vn = normals() if brush in ("DRAW", "INFLATE", "FLATTEN", "CLAY", "CREASE", "CLAY_STRIPS", "SCRAPE", "FILL",
                                        "PINCH", "LAYER", "BLOB", "DRAW_SHARP") else None
            if vn is not None:
                area_n = sum((vn[i] * wi for i, wi in w.items()), Vector())
                area_n = area_n.normalized() if area_n.length > 1e-12 else Vector((0.0, 0.0, 1.0))
                tw = sum(w.values())
                area_p = sum((co[i] * wi for i, wi in w.items()), Vector()) / tw
            step = 0.25 * r_loc
            if brush == "SMOOTH":
                co_upd = {i: co[i].lerp(sum((co[j] for j in nbrs[i]), Vector()) / len(nbrs[i]), min(1.0, wi * strength))
                          for i, wi in w.items() if nbrs[i]}
            else:
                co_upd = {}
                for i, wi in w.items():
                    k = wi * strength
                    if brush == "DRAW":
                        co_upd[i] = co[i] + area_n * (sign * step * k)
                    elif brush == "INFLATE":
                        co_upd[i] = co[i] + vn[i] * (sign * step * k)
                    elif brush == "FLATTEN":
                        co_upd[i] = co[i] - area_n * ((co[i] - area_p).dot(area_n) * k)
                    elif brush == "CLAY":
                        plane = area_p + area_n * (sign * step * strength)
                        h = (co[i] - plane).dot(area_n) * sign
                        co_upd[i] = co[i] - area_n * ((co[i] - plane).dot(area_n) * k) if h < 0 else co[i]
                    elif brush == "CREASE":
                        pinch = (c - co[i]) - area_n * (c - co[i]).dot(area_n)
                        co_upd[i] = co[i] - area_n * (sign * step * k) + pinch * (0.5 * k)
                    elif brush == "CLAY_STRIPS":
                        # toward the raised area plane from BOTH sides - it fills and cuts at once, a flat strip
                        plane = area_p + area_n * (sign * step * strength)
                        co_upd[i] = co[i] - area_n * ((co[i] - plane).dot(area_n) * k)
                    elif brush in ("SCRAPE", "FILL"):
                        # toward the area plane, SCRAPE from above only (cuts peaks), FILL from below only (fills
                        # dips); invert swaps the sides
                        h = (co[i] - area_p).dot(area_n)
                        above = (h > 0) == ((brush == "SCRAPE") == (sign > 0))
                        co_upd[i] = co[i] - area_n * (h * k) if above else co[i]
                    elif brush == "PINCH":
                        # toward the dab centre in the surface's plane (invert: away - magnify)
                        pinch = (c - co[i]) - area_n * (c - co[i]).dot(area_n)
                        co_upd[i] = co[i] + pinch * (0.5 * k * sign)
                    elif brush == "LAYER":
                        # out to ONE height: 0.25 x radius x strength above where the vertex began this stroke, along
                        # the area normal, however many dabs pass - a layer, not a pile
                        target_h = sign * step * strength * fall(min(1.0, (co[i] - c).length / r_loc)) * mask_w(i)
                        cur_h = (co[i] - start[i]).dot(area_n)
                        if abs(target_h) > abs(cur_h):
                            co_upd[i] = co[i] + area_n * (target_h - cur_h)
                    elif brush == "BLOB":
                        # outward along each normal and drawn in toward the dab: a rounded blob
                        pinch = (c - co[i]) - vn[i] * (c - co[i]).dot(vn[i])
                        co_upd[i] = co[i] + vn[i] * (sign * step * k) + pinch * (0.25 * k)
                    elif brush == "DRAW_SHARP":
                        # along the vertex's normal at the start of the stroke, with the falloff squared: a sharp line
                        n0 = vn0[i] if vn0 is not None else vn[i]
                        co_upd[i] = co[i] + n0 * (sign * step * strength * (wi * wi) * 2.0)
            for i, q in co_upd.items():
                co[i] = q
    if cfg["deformTarget"] == "CLOTH_SIM" and brush != "CLOTH":
        # the brush's result is the goal of the vertices it moved; the rest of the simulation area follows as cloth
        moves = {i: (co[i] - start[i]).length for i in range(len(co))}
        top = max(moves.values())
        area = area_of([c for dabs in cfg["dabSets"] for c in dabs])
        free = [1.0 if i in area and mask_w(i) > 0.0 else 0.0 for i in range(len(co))]
        for i, m in moves.items():
            if m > 1e-9:      # held by how far the brush moved it: the most-moved vertex stays at the brush's result
                free[i] = min(free[i], 1.0 - m / top)
        _cloth_relax(co, edges, rest, free, _CLOTH_PASSES * 2)
    moved = [i for i in range(len(co)) if (co[i] - start[i]).length > 1e-9]
    return start, co, moved, n_dabs, verts, layer


def _multires_mesh(obj, mres):
    """The mesh at the Multires modifier's top level, evaluated with the modifiers after it switched off. A new mesh
    datablock the caller removes."""
    others = [(m, m.show_viewport) for m in obj.modifiers if m is not mres]
    keep = (mres.levels, mres.show_viewport)
    try:
        for m, _s in others:
            m.show_viewport = False
        mres.levels = mres.total_levels
        mres.show_viewport = True
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        return bpy.data.meshes.new_from_object(obj.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    finally:
        for m, s in others:
            m.show_viewport = s
        mres.levels, mres.show_viewport = keep
        bpy.context.view_layer.update()


def _multires_drop_levels(obj, mres, keep_total):
    """Delete the levels a refused stroke added, so a refusal leaves the modifier as it was."""
    if mres.total_levels <= keep_total:
        return
    mres.levels = mres.sculpt_levels = keep_total
    mres.render_levels = min(mres.render_levels, keep_total)
    with bpy.context.temp_override(object=obj, active_object=obj):
        bpy.ops.object.multires_higher_levels_delete(modifier=mres.name)


def _sculpt_parse(params):
    """Read and check every parameter of sculpt_stroke (its keys are checked by the op); refuses before any write."""
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    require_editable(obj, "sculpt")
    brush = str(take(params, "brush", required=True, kind=str)).upper()
    if brush not in _SCULPT_BRUSHES:
        raise MifOpError("brush must be one of %s, got %r. NOTHING was changed." % (", ".join(_SCULPT_BRUSHES), brush))
    deform = take(params, "boundaryDeform", default=None, kind=str)
    if deform is not None and brush != "BOUNDARY":
        raise MifOpError("boundaryDeform is the BOUNDARY brush's mode; brush is %s. NOTHING was changed." % brush)
    deform = str(deform or "BEND").upper()
    if deform not in _BOUNDARY_DEFORMS:
        raise MifOpError("boundaryDeform must be one of %s, got %r. NOTHING was changed."
                         % (", ".join(_BOUNDARY_DEFORMS), deform))
    cloth_deform = take(params, "clothDeform", default=None, kind=str)
    if cloth_deform is not None and brush != "CLOTH":
        raise MifOpError("clothDeform is the CLOTH brush's mode; brush is %s. NOTHING was changed." % brush)
    cloth_deform = str(cloth_deform or "DRAG").upper()
    if cloth_deform not in _CLOTH_DEFORMS:
        raise MifOpError("clothDeform must be one of %s, got %r. NOTHING was changed."
                         % (", ".join(_CLOTH_DEFORMS), cloth_deform))
    target = str(take(params, "deformTarget", default="CLOTH_SIM" if brush == "CLOTH" else "GEOMETRY",
                      kind=str)).upper()
    if target not in ("GEOMETRY", "CLOTH_SIM"):
        raise MifOpError("deformTarget must be GEOMETRY or CLOTH_SIM, got %r. NOTHING was changed." % target)
    if brush == "CLOTH" and target != "CLOTH_SIM":
        raise MifOpError("the CLOTH brush is a cloth simulation; deformTarget GEOMETRY is for the other brushes. "
                         "NOTHING was changed.")
    cloth_area = take_float(params, "clothArea", default=2.5)
    if not 1.0 <= cloth_area <= 20.0:
        raise MifOpError("clothArea is the simulation area as a multiple of the radius, 1-20, got %r. NOTHING was "
                         "changed." % cloth_area)
    raw = params.get("path")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise MifOpError("path is a non-empty list of [x, y, z] world points - one point is a dab. NOTHING was changed.")
    pts = []
    for i, pt in enumerate(raw):
        if not isinstance(pt, (list, tuple)) or len(pt) != 3:
            raise MifOpError("path[%d] must be [x, y, z], got %r. NOTHING was changed." % (i, pt))
        pts.append(Vector(finite_floats(list(pt), "path")))
    grabs = brush in ("GRAB", "SNAKE_HOOK") + _PATH_BRUSHES or (brush == "BOUNDARY" and deform == "GRAB") or (
        brush == "CLOTH" and cloth_deform in ("GRAB", "DRAG"))
    if brush == "POSE" and len(pts) != 3:
        raise MifOpError("POSE turns the part past a pivot: path is [pivot, from, to] - three points. NOTHING was "
                         "changed.")
    if grabs and len(pts) < 2:
        raise MifOpError("%s moves from the first point to the last, so path needs at least two points. NOTHING was "
                         "changed." % (brush if brush not in ("BOUNDARY", "CLOTH") else brush + " " + (
                             deform if brush == "BOUNDARY" else cloth_deform)))
    radius = take_float(params, "radius", required=True)
    if not radius > 0.0:
        raise MifOpError("radius must be above 0 (world units), got %r. NOTHING was changed." % radius)
    strength = take_float(params, "strength", default=0.5)
    if not 0.0 <= strength <= 1.0:
        raise MifOpError("strength is 0-1, got %r. NOTHING was changed." % strength)
    falloff = str(take(params, "falloff", default="SMOOTH", kind=str)).upper()
    if falloff not in _SCULPT_FALLOFF:
        raise MifOpError("falloff must be one of %s, got %r. NOTHING was changed." % (", ".join(_SCULPT_FALLOFF), falloff))
    invert = take_bool(params, "invert", default=False)
    sym = params.get("symmetry") or [False, False, False]
    if not isinstance(sym, (list, tuple)) or len(sym) != 3 or not all(isinstance(x, bool) for x in sym):
        raise MifOpError("symmetry is [x, y, z] booleans (the object's own axes), got %r. NOTHING was changed." % (sym,))
    spacing = take_float(params, "spacing", default=None)
    if spacing is not None and not spacing > 0.0:
        raise MifOpError("spacing is a fraction of the radius, above 0. NOTHING was changed.")
    stabilize = take_float(params, "stabilize", default=None)
    if stabilize is not None and not stabilize > 0.0:
        raise MifOpError("stabilize is the string length in world units, above 0. NOTHING was changed.")
    if brush == "POSE" and (spacing is not None or stabilize is not None):
        raise MifOpError("POSE's path is [pivot, from, to] as given - spacing and stabilize would resample it. NOTHING was "
                         "changed.")
    group, face_sets = _sculpt_mask(obj, params.get("mask"))
    key_name = take(params, "shapeKey", default=None, kind=str)
    keys = obj.data.shape_keys
    kb = None
    if key_name is not None:
        kb = keys.key_blocks.get(key_name) if keys is not None else None
        if kb is None:
            raise MifOpError("'%s' has no shape key %r (add_shape_key makes one). NOTHING was changed."
                             % (obj.name, key_name))
    elif keys is not None:
        raise MifOpError("'%s' has shape keys, and the Basis holds its rest shape - name the key to sculpt with "
                         "shapeKey. NOTHING was changed." % obj.name)
    detail = _sculpt_dyntopo_arg(params.get("dyntopo"), kb, obj)
    mres, level = _sculpt_multires_arg(obj, params, kb, detail)
    scale = _uniform_scale(obj)
    if scale is None:
        raise MifOpError("'%s' has a non-uniform or zero world scale, so a world radius is no one mesh distance. Apply "
                         "the scale first. NOTHING was changed." % obj.name)
    inv = obj.matrix_world.inverted_safe()
    local = [inv @ q for q in pts]
    if stabilize is not None:
        local = _sculpt_stabilize(local, stabilize / scale)
    r_loc = radius / scale
    return obj, {"brush": brush, "boundaryDeform": deform if brush == "BOUNDARY" else None, "radius": radius,
                 "r_loc": r_loc, "scale": scale, "strength": strength, "falloff": falloff,
                 "fall": _SCULPT_FALLOFF[falloff], "invert": invert, "sign": -1.0 if invert else 1.0,
                 "symmetry": list(sym), "stabilize": stabilize, "faceSets": face_sets,
                 "groupIndex": group.index if group is not None else None, "shapeKey": kb.name if kb else None,
                 "detail": detail, "mres": mres, "level": level, "clothDeform": cloth_deform if brush == "CLOTH"
                 else None, "deformTarget": target, "clothArea": cloth_area,
                 "dabSets": _sculpt_dabs(local, list(sym), spacing, r_loc)}


def _sculpt_mask(obj, mask):
    """mask {vertexGroup, faceSets [ids]} - either or both (multiplied). Returns (group, face set ids)."""
    if mask is None:
        return None, None
    if not isinstance(mask, dict) or not mask or set(mask) - {"vertexGroup", "faceSets"}:
        raise MifOpError("mask is {vertexGroup: name, faceSets: [ids]} (either or both), got %r. NOTHING was changed."
                         % (mask,))
    group = None
    if "vertexGroup" in mask:
        group = obj.vertex_groups.get(str(mask["vertexGroup"]))
        if group is None:
            raise MifOpError("'%s' has no vertex group %r. NOTHING was changed." % (obj.name, mask["vertexGroup"]))
    face_sets = None
    if "faceSets" in mask:
        ids = mask["faceSets"]
        if not isinstance(ids, (list, tuple)) or not ids or not all(isinstance(x, int) and not isinstance(x, bool)
                                                                    for x in ids):
            raise MifOpError("mask.faceSets is a non-empty list of face set ids (integers), got %r. NOTHING was "
                             "changed." % (ids,))
        attr = obj.data.attributes.get(_FACE_SET_ATTR)
        if attr is None or attr.domain != "FACE" or attr.data_type != "INT":
            raise MifOpError("'%s' has no face sets (no %s face attribute) - a mesh gets them in sculpt mode or by "
                             "writing that INT face attribute. NOTHING was changed." % (obj.name, _FACE_SET_ATTR))
        have = {d.value for d in attr.data}
        missing = sorted(set(ids) - have)
        if missing:
            raise MifOpError("'%s' has no face set %s (it has %s). NOTHING was changed."
                             % (obj.name, missing, sorted(have)[:40]))
        face_sets = list(ids)
    return group, face_sets


def _sculpt_dyntopo_arg(raw, kb, obj):
    """dyntopo {detailSize} (world units) - or None."""
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {"detailSize"}:
        raise MifOpError("dyntopo is {detailSize: the longest edge left under the brush, in world units}, got %r. "
                         "NOTHING was changed." % (raw,))
    detail = raw["detailSize"]
    if isinstance(detail, bool) or not isinstance(detail, (int, float)) or not math.isfinite(detail) or detail <= 0:
        raise MifOpError("dyntopo.detailSize is a world length above 0 (Blender's relative detail in screen pixels "
                         "has no meaning without a view), got %r. NOTHING was changed." % (detail,))
    if kb is not None:
        raise MifOpError("dyntopo changes the topology, and shape key %r needs the base mesh's vertex count. NOTHING "
                         "was changed." % kb.name)
    if any(m.type == "MULTIRES" for m in obj.modifiers):
        raise MifOpError("'%s' has a Multires modifier, whose levels are bound to the base topology - dyntopo would "
                         "break them (Blender refuses the same). NOTHING was changed." % obj.name)
    s = _uniform_scale(obj) or 1.0
    return float(detail) / s


def _sculpt_multires_arg(obj, params, kb, detail):
    """multiresLevel: the Multires modifier's level to sculpt - its top level, or above it (levels are added)."""
    raw = params.get("multiresLevel")
    if raw is None:
        return None, None
    level = take_int(params, "multiresLevel")
    mres = next((m for m in obj.modifiers if m.type == "MULTIRES"), None)
    if mres is None:
        raise MifOpError("'%s' has no Multires modifier (add_modifier type MULTIRES adds one). NOTHING was changed."
                         % obj.name)
    if kb is not None:
        raise MifOpError("multiresLevel sculpts the Multires displacement, not shape key %r. NOTHING was changed."
                         % kb.name)
    if detail is not None:
        raise MifOpError("dyntopo and multiresLevel do not mix. NOTHING was changed.")
    if level < 1:
        raise MifOpError("multiresLevel is 1 or more, got %d. NOTHING was changed." % level)
    if level < mres.total_levels:
        raise MifOpError("multiresLevel %d is below the modifier's top level %d: Blender's reshape writes the top level "
                         "only, so sculpt at %d, or delete the higher levels first. NOTHING was changed."
                         % (level, mres.total_levels, mres.total_levels))
    if len(obj.data.vertices) * (4 ** level) > _MULTIRES_MAX_VERTS:
        raise MifOpError("multiresLevel %d on %d base vertices would be about %d vertices (cap %d). NOTHING was changed."
                         % (level, len(obj.data.vertices), len(obj.data.vertices) * 4 ** level, _MULTIRES_MAX_VERTS))
    return mres, level


def op_sculpt_stroke(params):
    """Sculpt a mesh along a stroke in mesh space: DRAW, CLAY, INFLATE, SMOOTH, FLATTEN, CREASE, GRAB, SNAKE_HOOK or
    BOUNDARY, with dynamic-topology detail, on a Multires level, masked by a vertex group or face sets.

    THE ROUTE, MEASURED FIRST (2026-10-05, the item's own design check): Blender's sculpt.brush_stroke poll fails with
    no 3D view, and passes once the startup screen's VIEW_3D area and window region are overridden - but the stroke
    SEGFAULTED Blender 4.4.0 (exit 139) and 5.0.1 headless, and on 5.2.2 answered FINISHED and moved no vertex (its dabs
    are screen positions ray-cast through a view not looking at the mesh). So the brushes are done here, on the vertex
    positions: the same on every build and exactly repeatable, with Blender's falloff curves by formula - not its brush
    engine, its spacing or its pressure.

    path [[x,y,z], ...] (WORLD; one point is a dab) is where the dabs land, resampled every spacing x radius when spacing
    is given; stabilize (world length) is Blender's stroke stabilization - the dab trails the path on a string that long.
    radius (world units, uniform scale), strength 0-1 (default 0.5), falloff SMOOTH (default) | SPHERE | ROOT
    | SHARP | LINEAR | CONSTANT over distance / radius. Per dab, with w = falloff x strength x mask weight: DRAW moves
    along the area normal (the weighted mean of the vertex normals inside the radius) by 0.25 x radius x w; INFLATE
    along each vertex's own normal by the same; SMOOTH toward the mean of the vertex's neighbours by w; FLATTEN toward
    the area plane (through the weighted mean position) by w; CLAY toward that plane raised 0.25 x radius x strength,
    from below only; CREASE draws inward (invert: outward) and pinches toward the dab by w / 2; GRAB moves the vertices
    inside the radius of the first point by (last - first) x falloff (strength not applied); SNAKE_HOOK does that segment
    by segment, re-picking around the moved point. BOUNDARY (added 2026-10-06) works on the open boundary nearest the
    dab: the boundary vertices within the radius of it along the boundary, and from them every vertex within the radius
    along the edges (never the boundary past them), weighted by the falloff over that distance; boundaryDeform BEND
    (default: the boundary turns about the far edge of its region, 90 degrees at strength 1, toward its normal - invert
    away), EXPAND (out of the surface's edge by 0.25 x radius x w), INFLATE (along each normal), GRAB (first point to
    last, strength not applied) or SMOOTH.
    THE 1.3 BRUSHES (2026-10-19), with w as above: CLAY_STRIPS moves toward CLAY's raised plane from both sides (fills and
    cuts); SCRAPE toward the area plane from above only and FILL from below only (invert swaps them); PINCH toward the dab
    in the surface's plane by w / 2 (invert: away); LAYER out to one height, 0.25 x radius x strength x falloff above
    where each vertex began, however many dabs pass; BLOB along each normal by 0.25 x radius x w and drawn in toward the
    dab by w / 4; DRAW_SHARP along each vertex's normal at the stroke's start, falloff squared; NUDGE pushes along each
    segment in the surface's plane (two points at least); ELASTIC moves every vertex by (last - first) x strength /
    (1 + (d / radius)^2)^1.5 from the first point - no hard edge; POSE takes path [pivot, from, to] and turns the part
    past the pivot (within |from - pivot| + radius) by the rotation from from to to, blended over half a radius at the
    pivot.
    CLOTH (added 2026-10-06) is a cloth simulation under the brush, NATIVE in mesh space - not Blender's cloth brush
    engine, whose stroke cannot run headless: per dab the vertices under it take the clothDeform displacement - DRAG
    (default: along the path, segment by segment), GRAB (first point to last, in steps), PUSH (along the area normal,
    as DRAW), PINCH_POINT (toward the dab), INFLATE (each normal) or EXPAND (the rest lengths under it grow by
    0.25 x w) - and are held by their weight while every edge in the simulation area (clothArea x radius, default 2.5)
    is pulled back to its rest length by position-based constraints; outside the area, and where the mask is 0,
    vertices are pinned. So the cloth folds and drags instead of stretching. deformTarget CLOTH_SIM on the other
    brushes holds each vertex the brush moved by how far it moved it (the furthest stays at the brush's result) and
    lets the rest of the area follow as cloth (GEOMETRY, the default, moves only what the brush moved).
    invert flips DRAW, CLAY, INFLATE, CREASE and the BOUNDARY modes. symmetry [x, y, z] mirrors every dab across the
    object's own axes. mask {vertexGroup, faceSets [ids]} multiplies by the group's weight and / or keeps the vertices
    on a face of those sets (Blender's .sculpt_face_set attribute). shapeKey writes the stroke into that key.
    dyntopo {detailSize} (world length) triangulates the faces under the brush and splits every edge there longer than
    detailSize before the stroke - detail is added, never collapsed. multiresLevel sculpts the Multires modifier's top
    level (levels are added with Catmull-Clark subdivision up to it) and writes it back with Blender's own reshape: the
    base mesh does not move. Reply: dabs, verticesMoved, maxDisplacement, the world box before and after, and dyntopo /
    multires blocks with their counts.
    """
    reject_unknown(params, _SCULPT_KEYS, "sculpt_stroke")
    obj, cfg = _sculpt_parse(params)
    me = obj.data
    mres, level = cfg["mres"], cfg["level"]
    target_mesh, temp_obj, levels_added, keep_total = me, None, 0, None
    multires_out = None
    try:
        if mres is not None:
            keep_total = mres.total_levels
            with bpy.context.temp_override(object=obj, active_object=obj):
                while mres.total_levels < level:
                    r = bpy.ops.object.multires_subdivide(modifier=mres.name, mode="CATMULL_CLARK")
                    if "FINISHED" not in r:
                        raise MifOpError("Multires subdivide answered %s." % sorted(r))
                    levels_added += 1
            target_mesh = _multires_mesh(obj, mres)
        bm = bmesh.new()
        try:
            bm.from_mesh(target_mesh)
            bm.normal_update()
            v0 = len(bm.verts)
            dyn = None
            if cfg["detail"] is not None:
                centres = [c for dabs in cfg["dabSets"] for c in dabs]
                passes, longest = _dyntopo_refine(bm, centres, cfg["r_loc"], cfg["detail"])
                dyn = {"detailSize": round(cfg["detail"] * cfg["scale"], 6), "passes": passes, "verticesBefore": v0,
                       "verticesAfter": len(bm.verts), "longestEdgeUnderBrush": round(longest * cfg["scale"], 6),
                       "collapse": "not done: detail is added, never removed"}
            start, co, moved, n_dabs, verts, layer = _sculpt_apply(bm, cfg)
            if not moved:
                raise MifOpError("the stroke moved no vertex - no vertex lies within %g of the path%s%s. NOTHING was "
                                 "changed." % (cfg["radius"], " with weight in the mask" if (
                                     cfg["groupIndex"] is not None or cfg["faceSets"]) else "",
                                     " (BOUNDARY needs an open boundary within the radius of a dab)"
                                     if cfg["brush"] == "BOUNDARY" else ""))
            max_d = max((co[i] - start[i]).length for i in moved)
            mwld = obj.matrix_world

            def box(cs):
                w = [mwld @ c for c in cs]
                return {"min": [round(min(p[k] for p in w), 6) for k in range(3)],
                        "max": [round(max(p[k] for p in w), 6) for k in range(3)]}
            box_before, box_after = box(start), box(co)
            for i in moved:
                if layer is not None:
                    verts[i][layer] = co[i]
                else:
                    verts[i].co = co[i]
            bm.normal_update()
            if mres is None:
                _commit(bm, obj)
            else:
                bm.to_mesh(target_mesh)
                temp_obj = bpy.data.objects.new("_mif_sculpt_level", target_mesh)
                temp_obj.matrix_world = obj.matrix_world.copy()
                bpy.context.scene.collection.objects.link(temp_obj)
                with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj, temp_obj],
                                               selected_editable_objects=[obj, temp_obj]):
                    r = bpy.ops.object.multires_reshape(modifier=mres.name)
                if "FINISHED" not in r:
                    raise MifOpError("Multires reshape answered %s." % sorted(r))
                multires_out = {"modifier": mres.name, "level": level, "levelsAdded": levels_added,
                                "verticesAtLevel": len(verts), "baseVertices": len(me.vertices)}
        finally:
            bm.free()
    except MifOpError as exc:
        if levels_added:
            _multires_drop_levels(obj, mres, keep_total)
            raise MifOpError("%s (the %d Multires level(s) it added were deleted again)" % (exc, levels_added))
        raise
    finally:
        if temp_obj is not None:
            remove_objects([temp_obj])
        if target_mesh is not me and target_mesh is not None:
            bpy.data.meshes.remove(target_mesh)
    bpy.context.view_layer.update()
    out = {"object": obj.name, "brush": cfg["brush"], "dabs": n_dabs, "verticesMoved": len(moved),
           "maxDisplacement": round(max_d * cfg["scale"], 6), "radius": cfg["radius"], "strength": cfg["strength"],
           "falloff": cfg["falloff"], "invert": cfg["invert"], "symmetry": cfg["symmetry"],
           "shapeKey": cfg["shapeKey"], "boxBefore": box_before, "boxAfter": box_after}
    if cfg["boundaryDeform"]:
        out["boundaryDeform"] = cfg["boundaryDeform"]
    if cfg["stabilize"] is not None:
        out["stabilize"] = cfg["stabilize"]
    if cfg["deformTarget"] == "CLOTH_SIM":
        out["deformTarget"] = "CLOTH_SIM"
        out["clothArea"] = cfg["clothArea"]
        out["clothDeform"] = cfg["clothDeform"]
        out["clothNote"] = ("native cloth in mesh space (edge lengths held by position-based constraints), not "
                            "Blender's cloth brush engine")
    if cfg["faceSets"] is not None:
        out["maskFaceSets"] = cfg["faceSets"]
    if dyn is not None:
        out["dyntopo"] = dyn
        out["after"] = mesh_counts(obj)
    if multires_out is not None:
        out["multires"] = multires_out
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- cloth_filter
_CFILTER_KEYS = ("object", "filterType", "strength", "steps", "direction", "center", "mask", "colliders",
                 "collisionOffset", "friction", "dryRun", "previewId", "previewTtlSeconds")
_CFILTER_TYPES = ("GRAVITY", "INFLATE", "EXPAND", "PINCH", "SCALE")
_CFILTER_PASSES = 40    # constraint passes per filter step, a collision projection after every 5
_CFILTER_SETTLE = 200   # constraint passes after the last step, with no force


def _cfilter_vec(params, key, tail):
    raw = params.get(key)
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise MifOpError("%s must be [x, y, z], got %r. %s" % (key, raw, tail))
    return Vector(finite_floats(list(raw), key))


def op_cloth_filter(params):
    """Run a cloth filter over a whole mesh in mesh space - GRAVITY, INFLATE, EXPAND, PINCH or SCALE - with its edge
    lengths held as cloth, masked vertices pinned, and colliders pushed out of.

    THE GAP (spec "Cloth-style sculpt brushes and the cloth filter", 2026-10-06): the PROBE FAILED - Blender's
    sculpt.cloth_filter answers PASS_THROUGH headless and moves nothing (4.2.23, 5.2.2), and sculpt.mesh_filter in the
    same set-up segfaults both. So this is NATIVE, not Blender's filter: per step every free vertex moves by
    strength x the mean edge length - GRAVITY along direction (world, default [0,0,-1]), INFLATE along its normal,
    PINCH toward center (world, default the object's origin) - or, for EXPAND, every rest length grows by
    strength x 5 % (SCALE grows them the same and moves the vertices out from center with them); then the edges are
    pulled back to their rest lengths by position-based constraints (40 passes; 200 more after the last step), and
    every vertex inside a collider (or closer than collisionOffset, default 0.005 world) goes to its surface plus the
    offset along the normal, its
    slide along the surface that step cut back by friction (0-1, default 0.8), and a vertex in contact is that much
    heavier against the edges pulling it on the next step. steps
    (default 20, 1-500). mask {vertexGroup, faceSets [ids]} weights the vertices (Blender's face-set attribute); a
    vertex at weight 0 is pinned. colliders [object names] are read as they evaluate, world space. Reply: steps,
    verticesMoved, maxDisplacement, edge strain (largest |length / rest - 1| against the rest lengths it started
    with) before and after, collision pushes, the world box before and after. dryRun writes nothing. Refused: shape
    keys, a mask holding nothing, a collider that is the object or no mesh, strength outside 0-10, steps outside
    1-500, a zero direction.
    """
    reject_unknown(params, _CFILTER_KEYS, "cloth_filter")
    tail = "NOTHING was changed."
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    kind = str(take(params, "filterType", required=True, kind=str)).upper()
    if kind not in _CFILTER_TYPES:
        raise MifOpError("filterType must be one of %s, got %r. %s" % (", ".join(_CFILTER_TYPES), kind, tail))
    strength = take_float(params, "strength", default=0.5)
    steps = take_int(params, "steps", default=20)
    offset = take_float(params, "collisionOffset", default=0.005)
    friction = take_float(params, "friction", default=0.8)
    dry = take_bool(params, "dryRun", default=False)
    if not (math.isfinite(strength) and 0.0 <= strength <= 10.0):
        raise MifOpError("strength is 0-10, got %r. %s" % (strength, tail))
    if not 1 <= steps <= 500:
        raise MifOpError("steps is 1-500, got %r. %s" % (steps, tail))
    if not (math.isfinite(offset) and offset >= 0.0):
        raise MifOpError("collisionOffset is 0 or more (world units), got %r. %s" % (offset, tail))
    if not (math.isfinite(friction) and 0.0 <= friction <= 1.0):
        raise MifOpError("friction is 0-1, got %r. %s" % (friction, tail))
    direction = _cfilter_vec(params, "direction", tail)
    if direction is not None and direction.length < 1e-12:
        raise MifOpError("direction is a zero vector. %s" % tail)
    direction = (direction or Vector((0.0, 0.0, -1.0))).normalized()
    center = _cfilter_vec(params, "center", tail)
    group, face_sets = _sculpt_mask(obj, params.get("mask"))
    raw_c = params.get("colliders")
    colliders = []
    if raw_c is not None:
        if not isinstance(raw_c, (list, tuple)) or not raw_c or not all(isinstance(x, str) for x in raw_c):
            raise MifOpError("colliders is a list of mesh object names. %s" % tail)
        for nm in raw_c:
            c = get_object(nm, want_mesh=True)
            if c is obj:
                raise MifOpError("'%s' cannot collide with itself. %s" % (nm, tail))
            colliders.append(c)
    from .ops_garment import _world_edit, _write_world, evaluated_world   # noqa: PLC0415
    bm, shift = _world_edit(obj, "cloth_filter", writes=not dry)
    try:
        verts = list(bm.verts)
        if len(verts) < 2 or not len(bm.edges):
            raise MifOpError("'%s' has no edges to hold as cloth. %s" % (obj.name, tail))
        deform = bm.verts.layers.deform.active
        mw = [1.0] * len(verts)
        if group is not None:
            mw = [v[deform].get(group.index, 0.0) if deform is not None else 0.0 for v in verts]
        if face_sets is not None:
            fs = bm.faces.layers.int.get(_FACE_SET_ATTR)
            want = set(face_sets)
            mw = [m * (1.0 if any(f[fs] in want for f in v.link_faces) else 0.0) for m, v in zip(mw, verts)]
        free = [min(1.0, max(0.0, m)) for m in mw]
        if not any(free):
            raise MifOpError("the mask leaves no vertex free to move. %s" % tail)
        from mathutils.bvhtree import BVHTree   # noqa: PLC0415 - after the refusals: the offline sweep stubs mathutils
        trees = []
        for c in colliders:
            cb = evaluated_world(c, evaluated=True)
            try:
                cb.transform(Matrix.Translation(-shift))
                if len(cb.faces):
                    trees.append(BVHTree.FromBMesh(cb))
            finally:
                cb.free()
        co = [v.co.copy() for v in verts]
        start = [p.copy() for p in co]
        edges = _cloth_edges(bm)
        rest0 =[(co[i] - co[j]).length for i, j in edges]
        rest = list(rest0)
        unit = sum(rest0) / len(rest0)
        ctr = (center - shift) if center is not None else (obj.matrix_world.translation - shift)

        def strain(cs):
            return max(abs((cs[i] - cs[j]).length / r - 1.0) for (i, j), r in zip(edges, rest0) if r > 1e-12)
        strain0 = strain(co)
        pushes = 0
        contact = set()
        faces_of = [[f.index for f in v.link_faces] for v in verts]
        poly = [[x.index for x in f.verts] for f in bm.faces]
        for _s in range(steps):
            prev = [p.copy() for p in co]
            if kind in ("EXPAND", "SCALE"):
                g = 1.0 + 0.05 * strength
                for e, (i, j) in enumerate(edges):
                    if free[i] > 0.0 or free[j] > 0.0:
                        rest[e] *= 1.0 + (g - 1.0) * max(free[i], free[j])
            if kind == "INFLATE":
                vn = []
                for i in range(len(co)):
                    n = Vector()
                    for fi in faces_of[i]:
                        ps = [co[x] for x in poly[fi]]
                        n += (ps[1] - ps[0]).cross(ps[2] - ps[0])
                    vn.append(n.normalized() if n.length > 1e-12 else Vector())
            for i in range(len(co)):
                if free[i] <= 0.0:
                    continue
                k = strength * unit * free[i]
                if kind == "GRAVITY":
                    co[i] = co[i] + direction * k
                elif kind == "INFLATE":
                    co[i] = co[i] + vn[i] * k
                elif kind == "PINCH":
                    d = ctr - co[i]
                    if d.length > k:
                        co[i] = co[i] + d.normalized() * k
                elif kind == "SCALE":
                    co[i] = ctr + (co[i] - ctr) * (1.0 + 0.05 * strength * free[i])
            heavy = [f * (1.0 - friction) if i in contact else f for i, f in enumerate(free)]
            contact = set()
            for _chunk in range(_CFILTER_PASSES // 5):
                _cloth_relax(co, edges, rest, heavy if trees else free, 5)
                for t in trees:
                    for i in range(len(co)):
                        if free[i] <= 0.0:
                            continue
                        hit, n, _fi, _d = t.find_nearest(co[i])
                        if hit is not None and (co[i] - hit).dot(n) < offset:
                            q = hit + n * offset
                            d = q - prev[i]      # friction: the step's slide along the surface is cut back
                            q = q - (d - n * d.dot(n)) * friction
                            hit, n, _fi, _d = t.find_nearest(q)
                            co[i] = q if hit is None or (q - hit).dot(n) >= offset else hit + n * offset
                            contact.add(i)
                            pushes += 1
        for _chunk in range(_CFILTER_SETTLE // 5):      # settle: the constraints alone, no force
            _cloth_relax(co, edges, rest, free, 5)
            for t in trees:
                for i in range(len(co)):
                    if free[i] > 0.0:
                        hit, n, _fi, _d = t.find_nearest(co[i])
                        if hit is not None and (co[i] - hit).dot(n) < offset:
                            co[i] = hit + n * offset
        moved = [i for i in range(len(co)) if (co[i] - start[i]).length > 1e-9]
        if not moved:
            raise MifOpError("the filter moved no vertex. %s" % tail)
        max_d = max((co[i] - start[i]).length for i in moved)

        def box(cs):
            return {"min": [round(min(p[k] for p in cs) + shift[k], 6) for k in range(3)],
                    "max": [round(max(p[k] for p in cs) + shift[k], 6) for k in range(3)]}
        out = {"object": obj.name, "filterType": kind, "strength": strength, "steps": steps, "dryRun": dry,
               "verticesMoved": len(moved), "pinned": sum(1 for f in free if f <= 0.0),
               "maxDisplacement": round(max_d, 6), "strain": {"before": round(strain0, 6), "after": round(strain(co), 6)},
               "colliders": [c.name for c in colliders], "collisionPushes": pushes, "boxBefore": box(start),
               "boxAfter": box(co), "native": "mesh-space position-based cloth, not Blender's cloth filter (which "
                                               "does not run headless)"}
        if not dry:
            for v, p in zip(verts, co):
                v.co = p
            _write_world(bm, obj, shift)
    finally:
        bm.free()
    if not dry:
        out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- mesh_from_outline

_OUTLINE_KEYS = ("name", "outline", "holes", "height", "plane", "edgeLength", "materials", "uvScale", "boundaryGroup",
                 "location", "rotation", "scale", "collection", "takeOverName")
# The 2D outline's (u, v) in the plane, the plane's normal (the extrusion direction) - u x v = normal, so an outline
# drawn counter-clockwise faces along the normal.
_PLANES = {"XY": ((1, 0, 0), (0, 1, 0), (0, 0, 1)), "XZ": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
           "YZ": ((0, 1, 0), (0, 0, 1), (1, 0, 0))}


def _poly2d(raw, key, tail):
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        raise MifOpError("'%s' must be at least three [x, y] points. %s" % (key, tail))
    pts = []
    for p in raw:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise MifOpError("each point of '%s' must be [x, y], got %r. %s" % (key, p, tail))
        pts.append(tuple(finite_floats(list(p), key)))
    if len(pts) > 3 and math.dist(pts[0], pts[-1]) < 1e-9:
        pts.pop()                       # a closing point that repeats the first
    for i, p in enumerate(pts):
        if math.dist(p, pts[(i + 1) % len(pts)]) < 1e-9:
            raise MifOpError("'%s' repeats point %d (%r) - two points on one spot make a zero-length edge. %s"
                             % (key, i, p, tail))
    return pts


def _signed_area(pts):
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))


def _seg_cross(p, q, r, s):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2, d3, d4 = orient(r, s, p), orient(r, s, q), orient(p, q, r), orient(p, q, s)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and 0 not in (d1, d2, d3, d4)


def _crossings(rings):
    """The first pair of edges, over every ring, that cross each other - None when no two do."""
    segs = [(ri, i, ring[i], ring[(i + 1) % len(ring)]) for ri, ring in enumerate(rings) for i in range(len(ring))]
    for a in range(len(segs)):
        for b in range(a + 1, len(segs)):
            ra, ia, p, q = segs[a]
            rb, ib, r, s = segs[b]
            if ra == rb and (abs(ia - ib) == 1 or abs(ia - ib) == len(rings[ra]) - 1):
                continue
            if _seg_cross(p, q, r, s):
                return segs[a][:2], segs[b][:2]
    return None


def _inside(pt, ring):
    x, y = pt
    hit = False
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


def _seg_dist(pt, a, b):
    ax, ay = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((pt[0] - a[0]) * ax + (pt[1] - a[1]) * ay) / (ax * ax + ay * ay)))
    return math.hypot(pt[0] - a[0] - t * ax, pt[1] - a[1] - t * ay)


def _resample_ring(ring, step):
    """Every corner kept, each edge split into ceil(length / step) equal pieces."""
    if not step:
        return list(ring)
    out = []
    for a, b in zip(ring, ring[1:] + ring[:1]):
        n = max(1, int(math.ceil(math.dist(a, b) / step - 1e-9)))
        out.extend((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(n))
    return out


def _fill_rings(rings, step):
    """Triangles (or one n-gon) filling the outline rings[0] minus the holes rings[1:]. Returns (points, faces,
    boundary point indices in ring order per ring). With step, the rings are resampled to step and the inside carries
    a triangular lattice of that spacing (kept half a step clear of every edge), so a cloth panel has inner vertices."""
    from mathutils.geometry import delaunay_2d_cdt
    rings = [_resample_ring(r, step) for r in rings]
    if len(rings) == 1 and not step:
        n = len(rings[0])
        return rings[0], [list(range(n))], [list(range(n))]
    pts, edges, ring_idx = [], [], []
    for r in rings:
        base = len(pts)
        pts.extend(r)
        ring_idx.append(list(range(base, base + len(r))))
        edges.extend((base + i, base + (i + 1) % len(r)) for i in range(len(r)))
    segs = [(r[i], r[(i + 1) % len(r)]) for r in rings for i in range(len(r))]
    if step:
        xs = [p[0] for p in rings[0]]
        ys = [p[1] for p in rings[0]]
        dy = step * math.sqrt(3.0) / 2.0
        row, y = 0, min(ys) + dy * 0.5
        while y < max(ys):
            x = min(xs) + (step * 0.5 if row % 2 else 0.0)
            while x < max(xs):
                p = (x, y)
                if (_inside(p, rings[0]) and not any(_inside(p, h) for h in rings[1:])
                        and min(_seg_dist(p, a, b) for a, b in segs) > 0.5 * step):
                    pts.append(p)
                x += step
            y += dy
            row += 1
    vs, _e, fs, orig_v, _oe, _of = delaunay_2d_cdt([Vector(p) for p in pts], edges, [], 1, 1e-9)
    remap = {}
    for out_i, origs in enumerate(orig_v):
        for o in origs:
            remap[o] = out_i
    faces = []
    for f in fs:
        c = (sum(vs[i][0] for i in f) / len(f), sum(vs[i][1] for i in f) / len(f))
        if _inside(c, rings[0]) and not any(_inside(c, h) for h in rings[1:]):
            faces.append(list(f))
    used = sorted({i for f in faces for i in f})
    keep = {old: new for new, old in enumerate(used)}
    out_pts = [(vs[i][0], vs[i][1]) for i in used]
    faces = [[keep[i] for i in f] for f in faces]
    for f in faces:                     # counter-clockwise in the plane
        if _signed_area([out_pts[i] for i in f]) < 0:
            f.reverse()
    bound = [[keep[remap[i]] for i in ri] for ri in ring_idx]
    return out_pts, faces, bound


def op_mesh_from_outline(params):
    """A mesh from a 2D outline: a flat panel, or a slab extruded to a height with top, side and bottom material slots
    and UVs at a world density.

    THE GAP (2026-10-02): street blocks and kerbs were extruded from their plan outlines, and garment panels (a hood
    as a rounded rectangle, a sleeve from an armhole offset) were drawn as outlines and filled - each in raw Python
    around mesh.from_pydata. outline [[x, y], ...] is one closed ring in the plane (the last point need not repeat the
    first; either winding - it is made counter-clockwise, so the panel or the top faces along the plane's normal).
    holes are rings inside it. plane XY (default; normal +Z), XZ (a front view: x right, y up, normal -Y) or YZ
    (normal +X).

    height 0 (default) makes a flat panel; above 0 a closed slab: the outline's face at the plane (slot bottom), one
    at height along the normal (slot top) and the walls (slot side). edgeLength resamples every edge to at most that
    length and fills the inside with a triangular lattice of that spacing, so a cloth panel has inner vertices (cloth
    wants an edge near 10-20 mm); without it a ring with no holes is one n-gon, and holes are filled with triangles.
    Walls are cut every edgeLength up the height too.

    materials {top, side, bottom} names existing materials for the three slots (a slab always has the three slots, in
    that order; missing names leave a slot empty). uvScale (default 1) is metres per UV unit: caps are mapped flat in
    the plane's (x, y), walls along the perimeter by height, so a texture tiles at one density on every face.
    boundaryGroup (default "Boundary") names a vertex group holding the outline's vertices - a panel's sewing edge;
    null leaves it out. name, location, rotation, scale, collection and takeOverName are create_mesh's.

    Refused before anything is made: fewer than three points, a repeated point, edges that cross, a hole outside the
    outline, a negative height, an edgeLength not above 0 or one that would make more than 200,000 vertices.
    """
    tail = "NOTHING was created."
    reject_unknown(params, _OUTLINE_KEYS, "mesh_from_outline")
    outline = _poly2d(params.get("outline"), "outline", tail)
    holes_raw = params.get("holes") or []
    if not isinstance(holes_raw, (list, tuple)):
        raise MifOpError("'holes' must be a list of rings. %s" % tail)
    holes = [_poly2d(h, "holes[%d]" % i, tail) for i, h in enumerate(holes_raw)]
    height = take_float(params, "height", default=0.0)
    step = take_float(params, "edgeLength", default=None)
    uv_scale = take_float(params, "uvScale", default=1.0)
    plane = str(take(params, "plane", default="XY", kind=str)).upper()
    group = params.get("boundaryGroup", "Boundary")
    mats = params.get("materials")
    if plane not in _PLANES:
        raise MifOpError("plane must be XY, XZ or YZ, got %r. %s" % (plane, tail))
    if height < 0:
        raise MifOpError("height must be 0 (a flat panel) or above, got %g. %s" % (height, tail))
    if step is not None and not step > 0:
        raise MifOpError("edgeLength must be above 0, got %g. %s" % (step, tail))
    if not uv_scale > 0:
        raise MifOpError("uvScale must be above 0 (metres per UV unit), got %g. %s" % (uv_scale, tail))
    if group is not None and (not isinstance(group, str) or not group.strip()):
        raise MifOpError("boundaryGroup must be a vertex group name or null, got %r. %s" % (group, tail))
    if mats is not None:
        if not isinstance(mats, dict) or set(mats) - {"top", "side", "bottom"}:
            raise MifOpError("materials is {top, side, bottom} (existing material names), got %r. %s" % (mats, tail))
        have = set(bpy.data.materials.keys())
        missing = [m for m in mats.values() if m is not None and m not in have]
        if missing:
            raise MifOpError("no material named %s. %s" % (", ".join(repr(m) for m in missing), tail))
        if not height:
            raise MifOpError("materials name a slab's top, side and bottom slots; a flat panel (height 0) has one face "
                             "set - assign_material sets it. %s" % tail)
    if _signed_area(outline) < 0:
        outline.reverse()
    holes = [h if _signed_area(h) < 0 else h[::-1] for h in holes]
    for i, h in enumerate(holes):
        if not all(_inside(p, outline) for p in h):
            raise MifOpError("holes[%d] is not inside the outline. %s" % (i, tail))
    hit = _crossings([outline] + holes)
    if hit:
        raise MifOpError("edges cross: ring %d edge %d and ring %d edge %d (ring 0 is the outline, 1+ the holes) - an "
                         "outline must not cross itself or a hole. %s" % (hit[0] + hit[1] + (tail,)))
    if step:
        area = abs(_signed_area(outline))
        perim = sum(math.dist(a, b) for r in [outline] + holes for a, b in zip(r, r[1:] + r[:1]))
        est = 1.2 * area / (step * step * 0.866) + perim / step
        if height:
            est = 2 * est + perim / step * math.ceil(height / step)
        if est > 200000:
            raise MifOpError("edgeLength %g would make about %d vertices (more than 200,000). %s" % (step, est, tail))
    pts, cap, bound = _fill_rings([outline] + holes, step)
    u_ax, v_ax, n_ax = (Vector(a) for a in _PLANES[plane])

    def at(p, h=0.0):
        return list(u_ax * p[0] + v_ax * p[1] + n_ax * h)

    verts = [at(p) for p in pts]
    faces, mat_idx, uvs = [], [], []
    if not height:
        faces = [list(f) for f in cap]
        uvs = [[[pts[i][0] / uv_scale, pts[i][1] / uv_scale] for i in f] for f in faces]
    else:
        rows = max(1, int(math.ceil(height / step - 1e-9))) if step else 1
        n = len(pts)
        top = [[i + n for i in f] for f in cap]
        verts += [at(p, height) for p in pts]
        faces += top
        mat_idx += [0] * len(top)
        uvs += [[[pts[i - n][0] / uv_scale, pts[i - n][1] / uv_scale] for i in f] for f in top]
        # Walls: each boundary ring walked in order (outline counter-clockwise, holes clockwise), so a wall's outward
        # side is the right of the walk; inner rows of rings between bottom (row 0) and top (row `rows`).
        ring_rows = {}
        for ring in bound:
            for k in range(1, rows):
                for i in ring:
                    ring_rows[(i, k)] = len(verts)
                    verts.append(at(pts[i], height * k / rows))

        def vid(i, k):
            return i if k == 0 else (i + n if k == rows else ring_rows[(i, k)])
        for ring in bound:
            dist = [0.0]
            for a, b in zip(ring, ring[1:] + ring[:1]):
                dist.append(dist[-1] + math.dist(pts[a], pts[b]))
            for j, (a, b) in enumerate(zip(ring, ring[1:] + ring[:1])):
                for k in range(rows):
                    faces.append([vid(a, k), vid(b, k), vid(b, k + 1), vid(a, k + 1)])
                    mat_idx.append(1)
                    z0, z1 = height * k / rows / uv_scale, height * (k + 1) / rows / uv_scale
                    u0, u1 = dist[j] / uv_scale, dist[j + 1] / uv_scale
                    uvs.append([[u0, z0], [u1, z0], [u1, z1], [u0, z1]])
        bottom = [list(reversed(f)) for f in cap]
        faces += bottom
        mat_idx += [2] * len(bottom)
        uvs += [[[-pts[i][0] / uv_scale, pts[i][1] / uv_scale] for i in f] for f in bottom]
    sub = {"name": params.get("name") or "Outline", "vertices": verts, "faces": faces, "uvs": {"UVMap": uvs}}
    for k in ("location", "rotation", "scale", "collection", "takeOverName"):
        if params.get(k) is not None:
            sub[k] = params[k]
    if height:
        sub["materialIndices"] = mat_idx
        names = [(mats or {}).get(s) for s in ("top", "side", "bottom")]
        if all(names):
            sub["materials"] = names
    if group:
        sub["vertexGroups"] = {group: sorted({i for r in bound for i in r})}
    from .ops_create import op_create_mesh
    out = op_create_mesh(sub)
    obj = bpy.data.objects[out["name"]]
    if height and mats and not all(names):
        for slot, m in zip(obj.material_slots, names):
            if m:
                slot.material = bpy.data.materials[m]
        out["materialSlots"] = [s.material.name if s.material else None for s in obj.material_slots]
    out.update({
        "op": "mesh_from_outline", "plane": plane, "height": height, "edgeLength": step, "uvScale": uv_scale,
        "outlineArea": round(abs(_signed_area(outline)) - sum(abs(_signed_area(h)) for h in holes), 9),
        "outlinePoints": len(outline), "holes": len(holes),
        "boundaryVertices": [len(r) for r in bound], "boundaryGroup": group or None,
        "slots": ["top", "side", "bottom"] if height else None,
    })
    return out


# --------------------------------------------------------------------------------------------- mesh_sections

_SECTION_KEYS = ("object", "name", "axis", "heights", "step", "space", "evaluated", "limit")


def _chain(segments):
    """Join (a, b) point-key segments into runs: [(keys in order, closed)]."""
    nbr = {}
    for a, b in segments:
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    seen, runs = set(), []
    starts = [k for k, v in nbr.items() if len(v) == 1] + list(nbr)
    for s in starts:
        if s in seen:
            continue
        run, cur, prev = [s], s, None
        seen.add(s)
        while True:
            nxt = [k for k in nbr[cur] if k != prev and k not in seen]
            if not nxt:
                closed = len(run) > 2 and s in nbr[cur] and prev is not None
                break
            prev, cur = cur, nxt[0]
            run.append(cur)
            seen.add(cur)
        runs.append((run, closed))
    return runs


def op_mesh_sections(params):
    """Slice a mesh by planes across an axis and report each cross-section: its loops, their extents, length and area.

    THE GAP (2026-10-02): a body's and a garment's widths at the chest, waist and hip, and a car's profile at a few
    heights, were read by slicing in raw Python. heights [h, ...] or step (every step from the lowest point to the
    highest) along axis X, Y or Z (default Z), in WORLD space (default; the evaluated mesh - modifiers and the armature
    pose applied) or LOCAL. Each slice lists its loops - closed or open (an open mesh edge) - with vertex count,
    length, area (closed loops; shoelace in the slice plane), min / max / center in the two other axes, and the
    slice's overall extents and size across all its loops. A plane that meets nothing is a slice with no loops and
    empty true. Read-only.

    Refused: neither or both of heights and step, a step not above 0, more than limit (default 200, at most 2000)
    slices, and a mesh with no faces.
    """
    reject_unknown(params, _SECTION_KEYS, "mesh_sections")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    axis = str(take(params, "axis", default="Z", kind=str)).upper()
    space = str(take(params, "space", default="WORLD", kind=str)).upper()
    evaluated = take_bool(params, "evaluated", default=True)
    step = take_float(params, "step", default=None)
    limit = take_int(params, "limit", default=200)
    heights = params.get("heights")
    if axis not in ("X", "Y", "Z"):
        raise MifOpError("axis must be X, Y or Z, got %r. NOTHING was read." % axis)
    if space not in ("WORLD", "LOCAL"):
        raise MifOpError("space must be WORLD or LOCAL, got %r. NOTHING was read." % space)
    if (heights is None) == (step is None):
        raise MifOpError("pass heights [h, ...] OR step. NOTHING was read.")
    if not 1 <= limit <= 2000:
        raise MifOpError("limit must be 1-2000 slices, got %d. NOTHING was read." % limit)
    if step is not None and not step > 0:
        raise MifOpError("step must be above 0, got %g. NOTHING was read." % step)
    if heights is not None:
        if not isinstance(heights, (list, tuple)) or not heights:
            raise MifOpError("heights must be a non-empty list of numbers. NOTHING was read.")
        heights = finite_floats(list(heights), "heights")
    ai = "XYZ".index(axis)
    others = [i for i in range(3) if i != ai]
    bm = bmesh.new()
    try:
        if evaluated:
            dg = bpy.context.evaluated_depsgraph_get()
            ev = obj.evaluated_get(dg)
            me = ev.to_mesh()
            bm.from_mesh(me)
            ev.to_mesh_clear()
        else:
            bm.from_mesh(obj.data)
        if space == "WORLD":
            bm.transform(obj.matrix_world)
        if not bm.faces:
            raise MifOpError("'%s' has no faces to slice. NOTHING was read." % obj.name)
        lo = min(v.co[ai] for v in bm.verts)
        hi = max(v.co[ai] for v in bm.verts)
        if step is not None:
            n = int(math.floor((hi - lo) / step + 1e-9)) + 1
            if n > limit:
                raise MifOpError("step %g makes %d slices from %g to %g, over limit %d. NOTHING was read."
                                 % (step, n, lo, hi, limit))
            heights = [lo + step * k for k in range(n)]
        if len(heights) > limit:
            raise MifOpError("%d heights is over limit %d. NOTHING was read." % (len(heights), limit))
        slices = []
        for h in heights:
            segs, pos = [], {}
            for f in bm.faces:
                cut = []
                for lp in f.loops:
                    a, b = lp.vert.co, lp.link_loop_next.vert.co
                    da, db = a[ai] - h, b[ai] - h
                    if da == 0.0:
                        key = ("v", lp.vert.index)
                        pos[key] = a.copy()
                        cut.append(key)
                    elif (da < 0) != (db < 0) and db != 0.0:
                        t = da / (da - db)
                        key = ("e", lp.edge.index)
                        pos[key] = a.lerp(b, t)
                        cut.append(key)
                cut = list(dict.fromkeys(cut))
                if len(cut) == 2:
                    segs.append(tuple(cut))
            loops = []
            for run, closed in _chain(dict.fromkeys(tuple(sorted(s)) for s in segs)):
                if len(run) < 2:
                    continue
                p2 = [(pos[k][others[0]], pos[k][others[1]]) for k in run]
                length = sum(math.dist(a, b) for a, b in zip(p2, p2[1:] + (p2[:1] if closed else [])))
                loops.append({"closed": closed, "vertexCount": len(run), "length": round(length, 6),
                              "area": round(abs(_signed_area(p2)), 6) if closed else None,
                              "min": rnd([min(p[0] for p in p2), min(p[1] for p in p2)]),
                              "max": rnd([max(p[0] for p in p2), max(p[1] for p in p2)]),
                              "center": rnd([sum(p[0] for p in p2) / len(p2), sum(p[1] for p in p2) / len(p2)])})
            loops.sort(key=lambda r: -(r["area"] or 0.0))
            row = {"at": round(h, 6), "loops": loops, "empty": not loops}
            if loops:
                mn = [min(r["min"][i] for r in loops) for i in (0, 1)]
                mx = [max(r["max"][i] for r in loops) for i in (0, 1)]
                row.update({"min": rnd(mn), "max": rnd(mx), "size": rnd([mx[0] - mn[0], mx[1] - mn[1]])})
            slices.append(row)
    finally:
        bm.free()
    return {"object": obj.name, "axis": axis, "space": space, "evaluated": evaluated,
            "axes": ["XYZ"[i] for i in others], "range": [round(lo, 6), round(hi, 6)], "slices": slices,
            "note": "min, max, center and size are in the two other axes, %s and %s, in that order."
                    % ("XYZ"[others[0]], "XYZ"[others[1]])}


# --------------------------------------------------------------------------------------------- cut_eyelets

_EYELET_KEYS = ("object", "name", "at", "shape", "innerRadius", "size", "axis", "ringWidth", "slot", "minBoundary",
                "layers", "lining", "layerDepth", "mirror")


def _eyelet_plan(bm, bvh, p_local, axis_local, a, b, ring_w, k, taken):
    """One eyelet's hole and ring faces on a bmesh, centred on the surface point nearest p_local. Writes nothing."""
    loc, nrm, _fi, _dist = bvh.find_nearest(p_local)
    nrm = nrm.normalized()
    u = axis_local - nrm * axis_local.dot(nrm)
    if u.length < 1e-9:
        u = nrm.orthogonal()
    u.normalize()
    v = nrm.cross(u).normalized()
    # ANOTHER LAYER OF THE SAME MESH under or over this one is not part of this hole: faces are taken only within half
    # the distance to the nearest layer along the normal
    near = max(a, b) + ring_w
    for sign in (1, -1):
        hit = bvh.ray_cast(loc + nrm * sign * 1e-6, nrm * sign, near * 2)
        if hit[0] is not None:
            near = min(near, (hit[0] - loc).length / 2)

    def inside(co, grow=0.0):
        d = co - loc
        return (d.dot(u) / (a + grow)) ** 2 + (d.dot(v) / (b + grow)) ** 2 <= 1.0 and abs(d.dot(nrm)) < near
    hole = [f for f in bm.faces if inside(f.calc_center_median())]
    if not hole:
        raise MifOpError("eyelet %s deletes no face - the hole is smaller than the faces there; subdivide_mesh the "
                         "region or make it larger. NOTHING was changed." % k)
    if len(hole) == len(bm.faces):
        raise MifOpError("eyelet %s would delete every face. NOTHING was changed." % k)
    if taken & {f.index for f in hole}:
        raise MifOpError("eyelet %s overlaps another. NOTHING was changed." % k)
    taken |= {f.index for f in hole}
    ring = [f for f in bm.faces if f not in hole and inside(f.calc_center_median(), grow=ring_w)]
    return (k, loc, nrm, u, v, a, b, hole, ring)


def _eyelet_cut(bm, plans, mw, s):
    rows = []
    for k, loc, nrm, u, v, a, b, hole, ring in plans:
        hole_set = set(hole)
        edge_verts = [vv for vv in {vv for f in hole for vv in f.verts}
                      if any(f not in hole_set for f in vv.link_faces)]
        bmesh.ops.delete(bm, geom=hole, context="FACES_ONLY")
        loose = [vv for vv in bm.verts if not vv.link_faces]
        radii = []
        for vv in edge_verts:
            if not vv.is_valid:
                continue
            d = vv.co - loc
            du, dv = d.dot(u), d.dot(v)
            ang = math.atan2(dv / b, du / a)
            vv.co = loc + u * (a * math.cos(ang)) + v * (b * math.sin(ang)) + nrm * d.dot(nrm)
            radii.append(math.hypot((vv.co - loc).dot(u), (vv.co - loc).dot(v)) * s)
        if loose:
            bmesh.ops.delete(bm, geom=[vv for vv in loose if vv.is_valid], context="VERTS")
        rows.append({"eyelet": k, "center": [round(x, 6) for x in mw @ loc],
                     "normal": [round(x, 6) for x in (mw.to_3x3() @ nrm).normalized()],
                     "facesDeleted": len(hole), "boundaryVertices": len(radii),
                     "radius": {"min": round(min(radii), 6), "max": round(max(radii), 6)} if radii else None,
                     "ringFaces": len([f for f in ring if f.is_valid]), "_ring": ring})
    return rows


def op_cut_eyelets(params):
    """Cut eyelets - round or oval holes, each with a ring of faces round it in its own material slot - through a sheet.

    THE GAP (2026-10-02): eyelets on a hood band and the holes a drawcord exits by were cut with a circle primitive,
    an inset and a knife, or in raw Python; no op cut one. at [[x, y, z], ...] (world; each is moved to the nearest
    point of the surface) gives one eyelet per point. shape RING (innerRadius) or ELLIPSE (size [w, h], w along axis
    - a world direction laid into the surface, default +Z - and h across it). Faces whose centre lies inside the
    hole are deleted and the new boundary's vertices are moved onto the circle or ellipse in the surface's tangent
    plane, so the hole is round at the mesh's own density; faces within ringWidth (default 4 mm) of the hole's edge
    go to slot (a material name, added as a slot when the mesh lacks it). minBoundary (default 8) refuses a hole whose
    edge would have fewer vertices - the mesh is too coarse there; subdivide_mesh it first.

    at entries may also be vertex indices (that vertex's world position); mirror X | Y | Z adds each eyelet's twin
    across the plane through the mesh's centre (the middle of its world bounds) normal to that world axis (refused for
    a point on the plane).

    LAYERS (2026-10-06): layers OUTER (default) cuts the sheet the point is on. layers ALL also cuts every layer under
    it - along the eyelet's inward normal, within layerDepth (default 3 cm) - in the same mesh (a garment's lining
    shell) and in each object named in lining (a separate lining), the same shape round the point the line meets
    each, its own ring and slot. A lining that lies under none of the eyelets, or ALL finding no layer at all, is
    refused.

    Reply per eyelet: center (on the surface), normal, facesDeleted, boundaryVertices, radius {min, max} of the edge
    as cut, ringFaces, object and layer (OUTER, or INNER for a cut under it); boundaryLoopsBefore/After for the
    object, and linings [{object, boundaryLoopsBefore, boundaryLoopsAfter}]. Refused before the write: a point that
    is not near the surface (farther than 10 cm), two eyelets that overlap, a hole that would delete nothing or
    everything, shape keys.
    """
    from mathutils.bvhtree import BVHTree   # noqa: PLC0415
    reject_unknown(params, _EYELET_KEYS, "cut_eyelets")
    obj = _target(params, "cut_eyelets")
    raw = params.get("at")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise MifOpError("at must be a list of [x, y, z] world points or vertex indices, one per eyelet. NOTHING was "
                         "changed.")
    pts = []
    for p in raw:
        if isinstance(p, int) and not isinstance(p, bool):
            if not 0 <= p < len(obj.data.vertices):
                raise MifOpError("at names vertex %d, and '%s' has %d. NOTHING was changed."
                                 % (p, obj.name, len(obj.data.vertices)))
            pts.append(obj.matrix_world @ obj.data.vertices[p].co)
        elif isinstance(p, (list, tuple)) and len(p) == 3:
            pts.append(Vector(finite_floats(list(p), "at")))
        else:
            raise MifOpError("each entry of at must be [x, y, z] or a vertex index, got %r. NOTHING was changed." % (p,))
    # MIRRORED (2026-10-06): mirror X | Y | Z adds each eyelet's twin across the plane through the mesh's centre (the
    # middle of its world bounds) normal to that world axis - a pair either side of a hood's centre at an offset.
    mirror = take(params, "mirror", default=None, kind=str)
    if mirror is not None:
        mirror = str(mirror).upper()
        if mirror not in ("X", "Y", "Z"):
            raise MifOpError("mirror must be X, Y or Z (the world axis mirrored across the object's origin), got %r. "
                             "NOTHING was changed." % mirror)
        ax = "XYZ".index(mirror)
        world = [obj.matrix_world @ v.co for v in obj.data.vertices]
        c = (min(w[ax] for w in world) + max(w[ax] for w in world)) / 2
        twins = []
        for p in pts:
            q = p.copy()
            q[ax] = 2 * c - q[ax]
            if (q - p).length < 1e-9:
                raise MifOpError("an eyelet at %s lies on the mirror plane, so its twin is itself. NOTHING was changed."
                                 % [round(x, 4) for x in p])
            twins.append(q)
        pts += twins
    shape = str(take(params, "shape", default="RING", kind=str)).upper()
    if shape not in ("RING", "ELLIPSE"):
        raise MifOpError("shape must be RING or ELLIPSE, got %r. NOTHING was changed." % shape)
    if shape == "RING":
        if params.get("size") is not None:
            raise MifOpError("size [w, h] is an ELLIPSE's; a RING takes innerRadius. NOTHING was changed.")
        r = take_float(params, "innerRadius", default=None)
        if r is None or not r > 0:
            raise MifOpError("a RING needs innerRadius above 0 (metres). NOTHING was changed.")
        half = (r, r)
    else:
        if params.get("innerRadius") is not None:
            raise MifOpError("innerRadius is a RING's; an ELLIPSE takes size [w, h]. NOTHING was changed.")
        size = params.get("size")
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise MifOpError("an ELLIPSE needs size [w, h] (metres). NOTHING was changed.")
        w, h = finite_floats(list(size), "size")
        if not (w > 0 and h > 0):
            raise MifOpError("size must be above 0 both ways, got %r. NOTHING was changed." % (size,))
        half = (w / 2.0, h / 2.0)
    axis_w = _vec3(params, "axis", Vector((0, 0, 1)))
    if axis_w.length < 1e-9:
        raise MifOpError("axis must not be zero. NOTHING was changed.")
    ring_w = take_float(params, "ringWidth", default=0.004)
    if ring_w < 0:
        raise MifOpError("ringWidth must be 0 or above, got %g. NOTHING was changed." % ring_w)
    min_b = take_int(params, "minBoundary", default=8)
    if min_b < 3:
        raise MifOpError("minBoundary must be at least 3, got %d. NOTHING was changed." % min_b)
    slot_name = take(params, "slot", default=None, kind=str)
    if slot_name is not None and slot_name not in bpy.data.materials:
        raise MifOpError("no material named %r for the ring's slot. NOTHING was changed." % slot_name)
    layers = str(take(params, "layers", default="OUTER", kind=str)).upper()
    if layers not in ("OUTER", "ALL"):
        raise MifOpError("layers must be OUTER or ALL, got %r. NOTHING was changed." % layers)
    depth = take_float(params, "layerDepth", default=None)
    raw_lining = params.get("lining")
    if layers == "OUTER" and (depth is not None or raw_lining is not None):
        raise MifOpError("lining and layerDepth find the layers under an eyelet - pass layers ALL. NOTHING was "
                         "changed.")
    depth = 0.03 if depth is None else depth
    if not depth > 0:
        raise MifOpError("layerDepth must be above 0, got %g. NOTHING was changed." % depth)
    names = [] if raw_lining is None else [raw_lining] if isinstance(raw_lining, str) else raw_lining
    if not isinstance(names, (list, tuple)) or not all(isinstance(n, str) for n in names):
        raise MifOpError("lining is an object name or a list of them. NOTHING was changed.")
    linings = []
    for n in dict.fromkeys(names):
        lo = _target({"object": n}, "cut_eyelets")
        if lo is obj:
            raise MifOpError("'%s' is the object cut - a lining is another object; layers ALL already cuts the "
                             "layers of the same mesh. NOTHING was changed." % n)
        linings.append(lo)
    targets = [obj] + linings
    for t in targets:
        if _uniform_scale(t) is None:
            raise MifOpError("'%s' has a non-uniform or zero world scale, so a round hole in world units is an ellipse "
                             "in the mesh - apply the scale first. NOTHING was changed." % t.name)
    bms = {t: _open(t) for t in targets}
    try:
        bvhs = {t: BVHTree.FromBMesh(bm) for t, bm in bms.items()}
        before = {t: len(_boundary_loops_of(bm)) for t, bm in bms.items()}
        plans = {t: [] for t in targets}
        taken = {t: set() for t in targets}
        layer_of = {}
        mw = obj.matrix_world
        s = _uniform_scale(obj)
        inv = mw.inverted_safe()
        for k, p in enumerate(pts):
            loc, nrm, _fi, dist = bvhs[obj].find_nearest(inv @ p)
            if loc is None or dist * s > 0.1:
                raise MifOpError("at[%d] %s is not within 10 cm of '%s'. NOTHING was changed."
                                 % (k, [round(x, 4) for x in p], obj.name))
            plan = _eyelet_plan(bms[obj], bvhs[obj], inv @ p, inv.to_3x3() @ axis_w, half[0] / s, half[1] / s,
                                ring_w / s, k, taken[obj])
            plans[obj].append(plan)
            layer_of[(obj, k)] = "OUTER"
            if layers != "ALL":
                continue
            c_w, n_w = mw @ plan[1], (mw.to_3x3() @ plan[2]).normalized()
            for t in targets:
                tm = t.matrix_world
                ti = tm.inverted_safe()
                ts = _uniform_scale(t)
                eps = 1e-4 if t is obj else 0.0
                origin = ti @ (c_w - n_w * eps)
                direction = (ti.to_3x3() @ -n_w).normalized()
                hit = bvhs[t].ray_cast(origin, direction, (depth - eps) / ts)
                if hit[0] is None:
                    continue
                key = "%d.%s" % (k, t.name)
                plans[t].append(_eyelet_plan(bms[t], bvhs[t], hit[0], ti.to_3x3() @ axis_w, half[0] / ts,
                                             half[1] / ts, ring_w / ts, key, taken[t]))
                layer_of[(t, key)] = "INNER"
        if layers == "ALL":
            for t in linings:
                if not plans[t]:
                    raise MifOpError("lining '%s' lies under none of the eyelets (within layerDepth %g). NOTHING was "
                                     "changed." % (t.name, depth))
            if len(plans[obj]) == len(pts) and not linings:
                raise MifOpError("layers ALL found no layer under any eyelet within layerDepth %g - pass lining for a "
                                 "separate lining, or layers OUTER. NOTHING was changed." % depth)
        for t in targets:
            for k, loc, nrm, u, v, a, b, hole, ring in plans[t]:
                hole_verts = {vv for f in hole for vv in f.verts}
                edge_verts = [vv for vv in hole_verts if any(f not in hole for f in vv.link_faces)]
                if len(edge_verts) < min_b:
                    raise MifOpError("eyelet %s's edge on '%s' would have %d vertices (under minBoundary %d) - the mesh "
                                     "is too coarse there for a round hole; subdivide_mesh the region first. NOTHING "
                                     "was changed." % (k, t.name, len(edge_verts), min_b))
        rows, lining_rows, slot_index = [], [], None
        for t in targets:
            bm = bms[t]
            t_rows = _eyelet_cut(bm, plans[t], t.matrix_world, _uniform_scale(t))
            idx = None
            if slot_name is not None:
                mats = [m.name if m else None for m in t.data.materials]
                if slot_name not in mats:
                    t.data.materials.append(bpy.data.materials[slot_name])
                    mats.append(slot_name)
                idx = mats.index(slot_name)
                for row in t_rows:
                    for f in row["_ring"]:
                        if f.is_valid:
                            f.material_index = idx
            for row in t_rows:
                del row["_ring"]
                row["object"] = t.name
                row["layer"] = layer_of[(t, row["eyelet"])]
            if t is obj:
                slot_index = idx
            bm.normal_update()
            after = len(_boundary_loops_of(bm))
            _commit(bm, t)
            if t is obj:
                after_loops = after
            else:
                lining_rows.append({"object": t.name, "boundaryLoopsBefore": before[t], "boundaryLoopsAfter": after})
            rows += t_rows
    finally:
        for bm in bms.values():
            bm.free()
    out = {"object": obj.name, "shape": shape, "layers": layers, "eyelets": rows, "slot": slot_name,
           "slotIndex": slot_index, "boundaryLoopsBefore": before[obj], "boundaryLoopsAfter": after_loops,
           "linings": lining_rows or None, "after": mesh_counts(obj)}
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- rip

_RIP_KEYS = ("object", "name", "edges", "selector", "fill")


def _islands(bm):
    """Connected parts of the mesh: vertices joined by edges, a lone vertex its own part."""
    parent = {v: v for v in bm.verts}

    def find(v):
        while parent[v] is not v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    for e in bm.edges:
        a, b = find(e.verts[0]), find(e.verts[1])
        if a is not b:
            parent[a] = b
    return len({find(v) for v in bm.verts})


def _rip_fill_face(bm, f1, f2, cos):
    """The face across one ripped edge: f1's copy of the edge walked backwards, then f2's - a quad, or a triangle where
    one end did not split (the end of an open rip). None when neither end split."""
    def side(face):
        for loop in face.loops:
            a, b = loop.vert, loop.link_loop_next.vert
            if {tuple(a.co), tuple(b.co)} == cos:
                return a, b
        return None
    s1, s2 = side(f1), side(f2)
    if s1 is None or s2 is None:
        return None
    a1, b1 = s1                       # f1 walks a1 -> b1
    a2 = s2[0] if tuple(s2[0].co) == tuple(a1.co) else s2[1]
    b2 = s2[1] if a2 is s2[0] else s2[0]
    ring = [b1, a1]
    ring += [v for v in (a2, b2) if v not in ring]
    if len(ring) < 3:
        return None
    return bm.faces.new(ring)


def op_rip_edges(params):
    """Rip (split) a mesh along chosen edges: each edge becomes two, one on each face, and the vertices along the cut
    split with it - a chain that runs boundary to boundary, or a closed loop, cuts the mesh into separate parts.

    THE GAP: the EDGE_SPLIT modifier splits by angle or by sharp marks, never by a chosen set, so cutting a shirt into
    panels along its seams (collar, sleeve openings, hem, back pieces) - each rip a UV island boundary - had no call.
    bmesh.ops.split_edges, headless, no edit mode.

    edges [i, ...] (select_edges gives them) OR selector {select_edges' grammar: seedEdge with loop, minAngleDeg,
    axis and side, ...}. Every edge must have exactly two faces - a boundary or wire edge has nothing to rip. fill true
    (default false) bridges each ripped edge with a face, as Blender's Rip Fill does: the faces have no area until one
    side is moved. Reply: the new vertex indices, the parts (islands) before and after, counts before and after.
    """
    reject_unknown(params, _RIP_KEYS, "rip_edges")
    obj = _target(params, "rip_edges")
    fill = take_bool(params, "fill", default=False)
    given = [k for k in ("edges", "selector") if params.get(k) is not None]
    if len(given) != 1:
        raise MifOpError("pass edges OR selector (got %s). NOTHING was changed." % (", ".join(given) or "none"))
    sel = params.get("selector")
    if sel is not None and not isinstance(sel, dict):
        raise MifOpError("selector must be an object in select_edges' grammar, e.g. {\"seedEdge\": 4, \"loop\": true}. "
                         "NOTHING was changed.")
    before = mesh_counts(obj)
    bm = _open(obj)
    try:
        if sel is not None:
            from .ops_mesh import _SELECT_KEYS, _select_edges   # noqa: PLC0415 - select_edges' own resolver
            bad = sorted(set(sel) - (set(_SELECT_KEYS) - {"object", "name", "maxReported"}))
            if bad:
                raise MifOpError("selector keys %s are not select_edges selectors. NOTHING was changed." % bad)
            edges, _crit = _select_edges(bm, obj, dict(sel), "rip_edges")
            edges = list(dict.fromkeys(edges))
            if not edges:
                raise MifOpError("the selector matched no edge of '%s'. NOTHING was changed." % obj.name)
        else:
            edges = _elements(params, "edges", bm.edges, "edge", obj)
        open_edges = [e.index for e in edges if len(e.link_faces) != 2]
        if open_edges:
            raise MifOpError("edges %s do not have exactly two faces (a boundary, wire or non-manifold edge) - a rip "
                             "parts two faces. NOTHING was changed." % open_edges[:10])
        islands_before = _islands(bm)
        old = set(bm.verts)
        plan = [(e.link_faces[0], e.link_faces[1], {tuple(e.verts[0].co), tuple(e.verts[1].co)}) for e in edges]
        bmesh.ops.split_edges(bm, edges=edges)
        new_verts = [v for v in bm.verts if v not in old]
        if not new_verts:
            raise MifOpError("splitting those %d edge(s) split no vertex - a cut that starts and ends inside the "
                             "surface needs at least two edges in a chain. NOTHING was changed." % len(edges))
        filled = []
        if fill:
            for f1, f2, cos in plan:
                face = _rip_fill_face(bm, f1, f2, cos)
                if face is not None:
                    filled.append(face)
        islands_after = _islands(bm)
        bm.normal_update()
        _commit(bm, obj)
        new_idx = sorted(v.index for v in new_verts)
        fill_idx = sorted(f.index for f in filled)
    finally:
        bm.free()
    out = {"object": obj.name, "edgesRipped": len(edges), "newVertices": new_idx, "newVertexCount": len(new_idx),
           "fill": fill, "fillFaces": fill_idx if fill else None, "islandsBefore": islands_before,
           "islandsAfter": islands_after, "before": before, "after": mesh_counts(obj)}
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- circularize

_CIRCLE_KEYS = ("object", "name", "faces", "loop", "flatten", "radius", "regularize", "space")


def _region_loop(faces, obj):
    """The one boundary loop of a face region, in order, walked along the region's outer edges."""
    region = set(faces)
    rim = [e for f in faces for e in f.edges if sum(1 for g in e.link_faces if g in region) == 1]
    rim = list(dict.fromkeys(rim))
    if not rim:
        raise MifOpError("those faces have no rim - they close on themselves (a whole closed mesh), so there is no loop "
                         "to make round. NOTHING was changed.")
    nbrs = {}
    for e in rim:
        a, b = e.verts
        nbrs.setdefault(a, []).append(b)
        nbrs.setdefault(b, []).append(a)
    pinched = sorted(v.index for v, n in nbrs.items() if len(n) != 2)
    if pinched:
        raise MifOpError("the region's rim touches itself at vertices %s (two regions meeting at a corner) - name one "
                         "region with one rim. NOTHING was changed." % pinched[:10])
    start = min(nbrs, key=lambda v: v.index)
    order, prev, cur = [start], None, start
    while True:
        nxt = [v for v in nbrs[cur] if v is not prev]
        nxt = nxt[0] if prev is not None else min(nbrs[cur], key=lambda v: v.index)
        if nxt is start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
        if len(order) > len(nbrs):
            break
    if len(order) != len(nbrs):
        raise MifOpError("the faces' rim is more than one loop (%d of %d rim vertices on the first) - a region with a "
                         "hole, or two regions. Name one region with one rim. NOTHING was changed."
                         % (len(order), len(nbrs)))
    return order


def _circle_fit(pts):
    """(centre, normal, u, v) for an ordered closed polygon: Newell's normal, so the winding gives its sign."""
    c = sum(pts, Vector()) / len(pts)
    n = Vector()
    for a, b in zip(pts, pts[1:] + pts[:1]):
        n.x += (a.y - b.y) * (a.z + b.z)
        n.y += (a.z - b.z) * (a.x + b.x)
        n.z += (a.x - b.x) * (a.y + b.y)
    if n.length < 1e-12:
        raise MifOpError("the loop encloses no area (its points are on a line), so it has no plane to make a circle in. "
                         "NOTHING was changed.")
    n.normalize()
    u = (pts[0] - c) - n * (pts[0] - c).dot(n)
    if u.length < 1e-12:
        u = n.orthogonal()
    u.normalize()
    return c, n, u, n.cross(u)


def _circle_measure(pts, c, n, u, v, scale):
    """Radial deviation from the mean-radius circle, height off its plane, and the spread of the angular gaps."""
    rad, off, ang = [], [], []
    for p in pts:
        d = p - c
        h = d.dot(n)
        q = d - n * h
        rad.append(q.length)
        off.append(abs(h))
        ang.append(math.atan2(q.dot(v), q.dot(u)))
    mean_r = sum(rad) / len(rad)
    gaps = [math.degrees((b - a) % (2.0 * math.pi)) for a, b in zip(ang, ang[1:] + ang[:1])]
    return {"meanRadius": round(mean_r * scale, 6), "maxRadialDeviation": round(max(abs(r - mean_r) for r in rad)
                                                                                * scale, 6),
            "maxOffPlane": round(max(off) * scale, 6), "gapDegrees": {"min": round(min(gaps), 4),
                                                                      "max": round(max(gaps), 4)}}


def op_circularize(params):
    """Make a closed loop, or the rim of a face region, a circle - LoopTools' Circle, headless.

    THE GAP: a round boss grown out of a curved surface needs its inset region turned into a circle with the host's
    curvature kept; create_primitive circle only makes a new circle, and nothing reshaped an existing loop.

    faces [i...] (a region with one rim; its rim is made round, the faces inside are left where they are) OR loop
    [vertex i...] (a closed loop along edges, in any order). The circle lies in the loop's best-fit plane round its
    centre. flatten (default true) puts the loop in that plane; false keeps each vertex's height off it (the host's
    curvature). radius (default: the loop's mean distance from the centre; in `space` units, LOCAL or WORLD) sets the
    size. regularize (default true) spaces the vertices evenly round the circle; false keeps each one's angle.
    Reply: the circle's centre, normal and radius, and maxRadialDeviation, maxOffPlane and the angular gaps before and
    after.
    """
    reject_unknown(params, _CIRCLE_KEYS, "circularize")
    obj = _target(params, "circularize")
    space = _space(params)
    flatten = take_bool(params, "flatten", default=True)
    regular = take_bool(params, "regularize", default=True)
    radius = take_float(params, "radius", default=None)
    if radius is not None and not radius > 0.0:
        raise MifOpError("radius must be above 0, got %r. NOTHING was changed." % radius)
    radius = _length(obj, radius, space, "radius")
    given = [k for k in ("faces", "loop") if params.get(k) is not None]
    if len(given) != 1:
        raise MifOpError("pass faces OR loop (got %s). NOTHING was changed." % (", ".join(given) or "none"))
    scale = 1.0 if space == "LOCAL" else (_uniform_scale(obj) or 1.0)
    bm = _open(obj)
    try:
        if given[0] == "faces":
            loop = _region_loop(_elements(params, "faces", bm.faces, "face", obj), obj)
        else:
            verts = _elements(params, "loop", bm.verts, "vertex", obj)
            loop, closed = _ordered_run(verts)
            if not closed:
                raise MifOpError("loop's vertices are an open run along the edges, not a closed loop - a circle needs "
                                 "the loop to close. NOTHING was changed.")
        if len(loop) < 3:
            raise MifOpError("a circle needs at least 3 vertices on the loop (got %d). NOTHING was changed." % len(loop))
        pts = [v.co.copy() for v in loop]
        c, n, u, v = _circle_fit(pts)
        before = _circle_measure(pts, c, n, u, v, scale)
        heights = [(p - c).dot(n) for p in pts]
        angles = [math.atan2((p - c).dot(v), (p - c).dot(u)) for p in pts]
        r = radius if radius is not None else before["meanRadius"] / scale
        if regular:
            # Newell's normal follows the walk, so the angle climbs along it; the phase is the circular mean of each
            # vertex's offset from its even slot, which moves the loop as little as a rotation can.
            step = 2.0 * math.pi / len(loop)
            sx = sum(math.cos(a - k * step) for k, a in enumerate(angles))
            sy = sum(math.sin(a - k * step) for k, a in enumerate(angles))
            phase = math.atan2(sy, sx)
            angles = [phase + k * step for k in range(len(loop))]
        for vert, a, h in zip(loop, angles, heights):
            vert.co = c + u * (r * math.cos(a)) + v * (r * math.sin(a)) + (Vector() if flatten else n * h)
        after_pts = [vert.co.copy() for vert in loop]
        after = _circle_measure(after_pts, c, n, u, v, scale)
        moved = max((a - b).length for a, b in zip(after_pts, pts)) * scale
        bm.normal_update()
        _commit(bm, obj)
        idx = [vert.index for vert in loop]
    finally:
        bm.free()
    mw = obj.matrix_world
    out = {"object": obj.name, "source": given[0], "loop": idx, "vertexCount": len(idx), "space": space,
           "flatten": flatten, "regularize": regular, "radius": round(r * scale, 6),
           "center": [round(x, 6) for x in (c if space == "LOCAL" else mw @ c)],
           "normal": [round(x, 6) for x in (n if space == "LOCAL" else (mw.to_3x3() @ n).normalized())],
           "before": before, "after": after, "maxMoved": round(moved, 6)}
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- surface snapping
_SNAP_KEYS = ("object", "name", "target", "vertices", "vertexGroup", "mode", "axis", "offset", "outsideOnly",
              "maxDistance")
_SNAP_MODES = ("NEAREST", "PROJECT_NORMAL", "PROJECT_AXIS")
_QUAD_KEYS = ("object", "name", "target", "fromEdges", "quads", "distance", "direction", "steps", "snap", "offset",
              "mirror", "weldDistance")
_QUAD_SNAPS = ("FACE", "FACE_NEAREST", "NONE")
_AXIS_VECTORS = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
_FAR = 1.0e30


def _surface_target(params, obj):
    tgt = get_object(take(params, "target", required=True, kind=str), want_mesh=True)
    if tgt is obj:
        raise MifOpError("target is '%s' itself - a mesh is snapped onto ANOTHER mesh's surface. NOTHING was changed."
                         % obj.name)
    return tgt


def _surface_tree(tgt):
    """The target's evaluated surface (its modifiers and pose applied) in WORLD space, as a BVH tree."""
    from mathutils.bvhtree import BVHTree     # noqa: PLC0415 - after every refusal (the refusal sweep runs on a stub)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = tgt.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        mw = tgt.matrix_world
        verts = [mw @ v.co for v in me.vertices]
        polys = [tuple(p.vertices) for p in me.polygons]
    finally:
        ev.to_mesh_clear()
    if not polys:
        raise MifOpError("target '%s' has no faces to snap onto. NOTHING was changed." % tgt.name)
    return BVHTree.FromPolygons(verts, polys)


def _snap_point(tree, p, mode, direction, offset, outside_only, max_d):
    """One world point onto the surface: (the new point, or None when no surface was found; True when it was already
    outside and outsideOnly kept it). The offset is along the surface normal at the hit."""
    if mode == "NEAREST":
        loc, n, _i, _d = tree.find_nearest(p, max_d)
        if loc is None:
            return None, False
    else:
        best = None
        for s in (1.0, -1.0):
            loc, n, _i, d = tree.ray_cast(p, direction * s, max_d)
            if loc is not None and (best is None or d < best[2]):
                best = (loc, n, d)
        if best is None:
            return None, False
        loc, n = best[0], best[1]
    n = n.normalized()
    if outside_only and (p - loc).dot(n) >= offset - 1e-9:
        return p.copy(), True
    return loc + n * offset, False


def _snap_parse(params, obj):
    mode = str(take(params, "mode", default="NEAREST", kind=str)).upper()
    if mode not in _SNAP_MODES:
        raise MifOpError("mode must be one of %s, got %r. NOTHING was changed." % (", ".join(_SNAP_MODES), mode))
    axis = take(params, "axis", default=None, kind=str)
    if (axis is not None) != (mode == "PROJECT_AXIS"):
        raise MifOpError("axis (X, Y or Z, world) goes with mode PROJECT_AXIS and only with it; mode is %s, axis %r. "
                         "NOTHING was changed." % (mode, axis))
    if axis is not None:
        axis = axis.strip().upper().lstrip("+-")
        if axis not in _AXIS_VECTORS:
            raise MifOpError("axis must be X, Y or Z (the ray is cast both ways along it), got %r. NOTHING was "
                             "changed." % params.get("axis"))
    offset = take_float(params, "offset", default=0.0)
    max_d = take_float(params, "maxDistance", default=None)
    if max_d is not None and not max_d > 0.0:
        raise MifOpError("maxDistance is a world length above 0, got %r. NOTHING was changed." % max_d)
    outside = take_bool(params, "outsideOnly", default=False)
    me = obj.data
    raw, group_name = params.get("vertices"), take(params, "vertexGroup", default=None, kind=str)
    if raw is not None and group_name is not None:
        raise MifOpError("give vertices or vertexGroup, not both. NOTHING was changed.")
    if raw is not None:
        if not isinstance(raw, (list, tuple)) or not raw:
            raise MifOpError("vertices must be a non-empty list of vertex indices. NOTHING was changed.")
        bad = [i for i in raw if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(me.vertices)]
        if bad:
            raise MifOpError("vertices %s are not vertex indices of '%s' (it has %d). NOTHING was changed."
                             % (bad[:10], obj.name, len(me.vertices)))
        idx = list(dict.fromkeys(raw))
    elif group_name is not None:
        g = obj.vertex_groups.get(group_name)
        if g is None:
            raise MifOpError("'%s' has no vertex group %r. NOTHING was changed." % (obj.name, group_name))
        idx = [v.index for v in me.vertices if any(e.group == g.index and e.weight > 0.0 for e in v.groups)]
        if not idx:
            raise MifOpError("vertex group %r has no vertex with weight above 0. NOTHING was changed." % group_name)
    else:
        idx = list(range(len(me.vertices)))
        if not idx:
            raise MifOpError("'%s' has no vertices. NOTHING was changed." % obj.name)
    return {"mode": mode, "axis": axis, "offset": offset, "maxDistance": max_d, "outsideOnly": outside,
            "indices": idx, "selection": "vertices" if raw is not None else ("vertexGroup" if group_name else "all")}


def op_snap_to_surface(params):
    """Snap a mesh's vertices onto another mesh's surface: nearest point, or projected along each vertex normal or a
    world axis, lifted by an offset, optionally only the ones that sit inside it.

    THE GAP (spec "Draw quads onto a body surface and snap vertices to it", 2026-10-06): a shell traced over a body is
    snapped to it vertex by vertex; the only routes were ray_cast or closest_point_on_mesh per position plus a write, or
    a Shrinkwrap modifier applied - which moves every vertex and cannot say which ones found no surface.

    target the mesh snapped onto, evaluated (its modifiers and pose applied). vertices [indices] | vertexGroup (weight
    above 0) | neither (every vertex). mode NEAREST (default: the closest surface point, which wraps round a limb),
    PROJECT_NORMAL (a ray along the vertex normal, both ways, the nearer hit) or PROJECT_AXIS with axis X|Y|Z (world,
    both ways). offset (world, default 0) lifts the point along the surface normal at the hit. outsideOnly moves only the
    vertices below that lifted surface (a push out, never a pull in). maxDistance (world) - farther counts as no surface.
    A vertex on a clipping Mirror modifier's plane stays on that plane.
    Reply: moved, alreadyOutside, missed (the indices that found no surface - push them out by hand), maxMove. Every
    vertex missing is refused.
    """
    reject_unknown(params, _SNAP_KEYS, "snap_to_surface")
    obj = _target(params, "snap_to_surface")
    tgt = _surface_target(params, obj)
    cfg = _snap_parse(params, obj)
    tree = _surface_tree(tgt)
    me = obj.data
    mw = obj.matrix_world
    inv = mw.inverted_safe()
    nmat = mw.to_3x3().inverted_safe().transposed()
    planes = _clip_planes(obj)
    axis_v = Vector(_AXIS_VECTORS[cfg["axis"]]) if cfg["axis"] else None
    new, missed, kept, max_move = {}, [], 0, 0.0
    for i in cfg["indices"]:
        v = me.vertices[i]
        p = mw @ v.co
        d = axis_v if axis_v is not None else (nmat @ v.normal)
        if d.length < 1e-12:
            missed.append(i)
            continue
        q, outside = _snap_point(tree, p, cfg["mode"], d.normalized(), cfg["offset"], cfg["outsideOnly"],
                                 cfg["maxDistance"] if cfg["maxDistance"] is not None else _FAR)
        if q is None:
            missed.append(i)
            continue
        if outside:
            kept += 1
            continue
        q = _keep_on_planes(p, q, planes)
        if (q - p).length > 1e-9:
            new[i] = q
            max_move = max(max_move, (q - p).length)
    if len(missed) == len(cfg["indices"]):
        raise MifOpError("none of the %d vertices found the surface of '%s'%s. NOTHING was changed."
                         % (len(missed), tgt.name, " within maxDistance %g" % cfg["maxDistance"]
                            if cfg["maxDistance"] is not None else " (no ray met it)"
                            if cfg["mode"] != "NEAREST" else ""))
    for i, q in new.items():
        me.vertices[i].co = inv @ q
    me.update()
    bpy.context.view_layer.update()
    out = {"object": obj.name, "target": tgt.name, "mode": cfg["mode"], "axis": cfg["axis"], "offset": cfg["offset"],
           "outsideOnly": cfg["outsideOnly"], "selection": cfg["selection"], "considered": len(cfg["indices"]),
           "moved": len(new), "alreadyOutside": kept, "missedCount": len(missed), "missed": missed[:200],
           "missedTruncated": len(missed) > 200, "maxMove": round(max_move, 6)}
    out.update(shared_data_note(obj))
    return out


def _clip_planes(obj, extra=None):
    """The world planes of obj's clipping Mirror modifiers (plus `extra`, one about to be added): [(point, normal)]."""
    out = []
    mods = [(m.mirror_object, list(m.use_axis)) for m in obj.modifiers if m.type == "MIRROR" and m.use_clip]
    if extra is not None:
        mods.append(extra)
    for mobj, axes in mods:
        frame = mobj.matrix_world if mobj is not None else obj.matrix_world
        for k in range(3):
            if axes[k]:
                n = (frame.to_3x3() @ Vector(_AXIS_VECTORS["XYZ"[k]]))
                if n.length > 1e-12:
                    out.append((frame.translation.copy(), n.normalized()))
    return out


def _keep_on_planes(p, q, planes, tol=1e-5):
    """A point that sat on a clipping plane before the snap is put back on it."""
    for pt, n in planes:
        if abs((p - pt).dot(n)) <= tol:
            q = q - n * (q - pt).dot(n)
    return q


def _outward_dirs(bm, edges):
    """Each vertex of the boundary edges: the mean direction out of its faces, in the faces' planes (mesh space)."""
    acc = {}
    for e in edges:
        f = e.link_faces[0]
        a, b = e.verts[0].co, e.verts[1].co
        t = (b - a).normalized()
        out = (a + b) / 2.0 - f.calc_center_median()
        out = out - t * out.dot(t)
        out = out - f.normal * out.dot(f.normal)
        if out.length < 1e-12:
            continue
        for v in e.verts:
            acc[v] = acc.get(v, Vector()) + out.normalized()
    return {v: d.normalized() for v, d in acc.items() if d.length > 1e-12}


def _quads_parse(params, obj, bm):
    edges_raw, quads_raw = params.get("fromEdges"), params.get("quads")
    if (edges_raw is None) == (quads_raw is None):
        raise MifOpError("give fromEdges (boundary edge indices to extrude) or quads ([[x,y,z] x 4] world corners), "
                         "one of them. NOTHING was changed.")
    snap = str(take(params, "snap", default="FACE", kind=str)).upper()
    if snap not in _QUAD_SNAPS:
        raise MifOpError("snap must be one of %s, got %r. NOTHING was changed." % (", ".join(_QUAD_SNAPS), snap))
    offset = take_float(params, "offset", default=0.0)
    weld = take_float(params, "weldDistance", default=0.0001)
    if not weld >= 0.0:
        raise MifOpError("weldDistance is a world length, 0 or more. NOTHING was changed.")
    mirror = take(params, "mirror", default=None, kind=str)
    if mirror is not None:
        mirror = mirror.strip().upper()
        if mirror not in _AXIS_VECTORS:
            raise MifOpError("mirror must be X, Y or Z (a clipping Mirror modifier about the target), got %r. NOTHING "
                             "was changed." % params.get("mirror"))
    cfg = {"snap": snap, "offset": offset, "weld": weld, "mirror": mirror, "edges": None, "quads": None}
    if edges_raw is not None:
        cfg["edges"] = _elements(params, "fromEdges", bm.edges, "edge", obj)
        inner = [e.index for e in cfg["edges"] if len(e.link_faces) != 1]
        if inner:
            raise MifOpError("fromEdges %s are not boundary edges (each needs exactly one face, so it has an open side "
                             "to grow from). NOTHING was changed." % inner[:10])
        dist = take_float(params, "distance", required=True)
        if not dist > 0.0:
            raise MifOpError("distance is the world length of each row, above 0, got %r. NOTHING was changed." % dist)
        steps = take_int(params, "steps", default=1)
        if not 1 <= steps <= 100:
            raise MifOpError("steps is 1-100, got %r. NOTHING was changed." % steps)
        cfg.update(distance=dist, steps=steps, direction=_vec3(params, "direction"))
        if cfg["direction"] is not None and cfg["direction"].length < 1e-12:
            raise MifOpError("direction is a non-zero world vector. NOTHING was changed.")
    else:
        for k in ("distance", "direction", "steps"):
            if params.get(k) is not None:
                raise MifOpError("%s goes with fromEdges; quads gives every corner. NOTHING was changed." % k)
        if not isinstance(quads_raw, (list, tuple)) or not quads_raw:
            raise MifOpError("quads is a non-empty list of quads, each four [x, y, z] world corners. NOTHING was "
                             "changed.")
        quads = []
        for qi, q in enumerate(quads_raw):
            if not isinstance(q, (list, tuple)) or len(q) != 4 or not all(isinstance(c, (list, tuple)) and len(c) == 3
                                                                           for c in q):
                raise MifOpError("quads[%d] must be four [x, y, z] corners, got %r. NOTHING was changed." % (qi, q))
            quads.append([Vector(finite_floats(list(c), "quads")) for c in q])
        cfg["quads"] = quads
    return cfg


def _orient_out(faces, tree, mw):
    """Turn each new face to face away from the target surface (along the surface normal at its nearest point)."""
    flipped = 0
    for f in faces:
        c = mw @ f.calc_center_median()
        loc, n, _i, _d = tree.find_nearest(c)
        fn = mw.to_3x3().inverted_safe().transposed() @ f.normal
        if loc is not None and fn.dot(n) < 0.0:
            f.normal_flip()
            flipped += 1
    return flipped


def op_add_quads_on_surface(params):
    """Draw quads onto a body: extrude a strip from boundary edges, or add quads by their world corners, each new
    vertex snapped onto the target's surface, optionally half-mirrored with clipping.

    THE GAP (spec "Draw quads onto a body surface and snap vertices to it", 2026-10-06): a garment shell or a retopology
    is traced over a body by dragging quads out of boundary edges with face snapping. There was no call for it -
    ray_cast per point and create_mesh, quad by quad, which cannot grow from the edges of an existing mesh.

    target the surface (evaluated). fromEdges [boundary edge indices] grows rows: steps (default 1) rows of distance
    (world) each, along direction [x,y,z] (world) or, by default, out of the faces in their own plane; each row's
    vertices are snapped. Or quads [[[x,y,z] x 4], ...] (world corners), corners within weldDistance (world, default
    0.0001) of each other or of an existing vertex welded. snap FACE (default: a ray along the strip's own normal, both
    ways - projects like a view), FACE_NEAREST (the closest surface point - wraps round a limb) or NONE; offset (world)
    lifts along the surface normal. Every new face is turned to face away from the target. mirror X|Y|Z adds (or reuses)
    a clipping Mirror modifier whose mirror object is the target, and puts new vertices that cross or touch its plane
    on it. Reply: the new vertex and face indices, snapped, missed (no surface found - left where drawn), clipped.
    """
    reject_unknown(params, _QUAD_KEYS, "add_quads_on_surface")
    obj = _target(params, "add_quads_on_surface")
    tgt = _surface_target(params, obj)
    before = mesh_counts(obj)
    bm = _open(obj)
    try:
        cfg = _quads_parse(params, obj, bm)
        mirror_mod = None
        if cfg["mirror"] is not None:
            k = "XYZ".index(cfg["mirror"])
            mirror_mod = next((m for m in obj.modifiers if m.type == "MIRROR" and m.use_axis[k]), None)
        tree = _surface_tree(tgt)
        mw = obj.matrix_world
        inv = mw.inverted_safe()
        planes = _clip_planes(obj)
        if cfg["mirror"] is not None and (mirror_mod is None or not mirror_mod.use_clip):
            axes = [a == cfg["mirror"] for a in "XYZ"]
            mobj = mirror_mod.mirror_object if mirror_mod is not None else tgt
            planes = _clip_planes(obj, (mobj, axes))
        ref_side = {}
        for pi, (pt, n) in enumerate(planes):
            ss = [(mw @ v.co - pt).dot(n) for v in bm.verts]
            if cfg["quads"] is not None:
                ss += [(sum(q, Vector()) / 4.0 - pt).dot(n) for q in cfg["quads"]]
            mean = sum(ss) / len(ss) if ss else 0.0
            ref_side[pi] = 1.0 if mean >= 0.0 else -1.0
        stats = {"snapped": 0, "missed": [], "clipped": 0}

        def place(p, direction):
            """p (world) snapped and clipped -> (local point, found)."""
            found = True
            if cfg["snap"] != "NONE":
                q, _o = _snap_point(tree, p, "NEAREST" if cfg["snap"] == "FACE_NEAREST" else "PROJECT",
                                    direction, cfg["offset"], False, _FAR)
                if q is None or (cfg["snap"] == "FACE" and direction.length < 1e-12):
                    found = False
                else:
                    p = q
            for pi, (pt, n) in enumerate(planes):
                s = (p - pt).dot(n)
                if s * ref_side[pi] < 0.0 or abs(s) <= max(cfg["weld"], 1e-6):
                    p = p - n * s
                    stats["clipped"] += 1
            return inv @ p, found
        nmat = mw.to_3x3().inverted_safe().transposed()
        new_verts, new_faces = [], []
        if cfg["edges"] is not None:
            cur = cfg["edges"]
            for _step in range(cfg["steps"]):
                bm.normal_update()
                dirs = _outward_dirs(bm, cur)
                face_n = {}
                for e in cur:
                    for v in e.verts:
                        face_n[v] = face_n.get(v, Vector()) + e.link_faces[0].normal
                ret = bmesh.ops.extrude_edge_only(bm, edges=cur)
                nv = [g for g in ret["geom"] if isinstance(g, bmesh.types.BMVert)]
                nset = set(nv)
                for v in nv:
                    src = next(e.other_vert(v) for e in v.link_edges if e.other_vert(v) not in nset)
                    if cfg["direction"] is not None:
                        step_w = cfg["direction"].normalized() * cfg["distance"]
                    else:
                        d_loc = dirs.get(src)
                        if d_loc is None:
                            raise MifOpError("boundary vertex %d has no outward direction (its faces are degenerate) "
                                             "- pass direction. NOTHING was changed." % src.index)
                        step_w = (mw.to_3x3() @ d_loc).normalized() * cfg["distance"]
                    proj = (nmat @ face_n.get(src, Vector())).normalized() if face_n.get(src) is not None \
                        else Vector()
                    loc, found = place(mw @ src.co + step_w, proj)
                    v.co = loc
                    if not found:
                        stats["missed"].append(v)
                    elif cfg["snap"] != "NONE":
                        stats["snapped"] += 1
                new_verts += nv
                faces = [g for g in ret["geom"] if isinstance(g, bmesh.types.BMFace)]
                new_faces += faces
                cur = [g for g in ret["geom"] if isinstance(g, bmesh.types.BMEdge)
                       and g.verts[0] in nset and g.verts[1] in nset]
        else:
            existing = [(v, mw @ v.co) for v in bm.verts]
            uniq, corner_of = [], []           # drawn corners welded to each other first, on their DRAWN positions
            for qi, q in enumerate(cfg["quads"]):
                qn = (q[1] - q[0]).cross(q[3] - q[0]) + (q[3] - q[2]).cross(q[1] - q[2])
                if qn.length < 1e-12:
                    raise MifOpError("quads[%d] has no area. NOTHING was changed." % qi)
                row = []
                for c in q:
                    k = next((j for j, u in enumerate(uniq) if (u[0] - c).length <= cfg["weld"]), None)
                    if k is None:
                        uniq.append([c, Vector()])
                        k = len(uniq) - 1
                    uniq[k][1] += qn.normalized()
                    row.append(k)
                if len(set(row)) != 4:
                    raise MifOpError("quads[%d]: two corners welded into one vertex (weldDistance %g) - it would be a "
                                     "triangle or nothing. NOTHING was changed." % (qi, cfg["weld"]))
                corner_of.append(row)
            made, vert_of = [], []
            for c, n in uniq:
                loc, found = place(c, n.normalized() if n.length > 1e-12 else n)
                w = mw @ loc
                hit = next((v for v, vw in existing if (vw - c).length <= cfg["weld"] or (vw - w).length <= cfg["weld"]),
                           None)
                if hit is None:
                    hit = bm.verts.new(loc)
                    made.append(hit)
                    if not found:
                        stats["missed"].append(hit)
                    elif cfg["snap"] != "NONE":
                        stats["snapped"] += 1
                vert_of.append(hit)
            for qi, row in enumerate(corner_of):
                corners = [vert_of[k] for k in row]
                if len(set(corners)) != 4:
                    raise MifOpError("quads[%d]: two corners welded onto one existing vertex (weldDistance %g). NOTHING "
                                     "was changed." % (qi, cfg["weld"]))
                if bm.faces.get(corners) is not None:
                    raise MifOpError("quads[%d] is already a face of '%s'. NOTHING was changed." % (qi, obj.name))
                new_faces.append(bm.faces.new(corners))
            new_verts = made
        bm.normal_update()
        flipped = _orient_out(new_faces, tree, mw)
        bm.normal_update()
        if mirror_mod is None and cfg["mirror"] is not None:
            mirror_mod = obj.modifiers.new("Mirror", "MIRROR")
            mirror_mod.use_axis = [a == cfg["mirror"] for a in "XYZ"]
            mirror_mod.use_clip = True
            mirror_mod.mirror_object = tgt
            mirror_added = True
        else:
            mirror_added = False
            if mirror_mod is not None:
                mirror_mod.use_clip = True
        _commit(bm, obj)
        nv_idx, nf_idx, miss_idx = _idx(new_verts), _idx(new_faces), _idx(stats["missed"])
    finally:
        bm.free()
    out = {"object": obj.name, "target": tgt.name, "snap": cfg["snap"], "offset": cfg["offset"],
           "newVertices": nv_idx[:500], "newFaces": nf_idx[:500], "newVertexCount": len(nv_idx),
           "newFaceCount": len(nf_idx), "snapped": stats["snapped"], "missedCount": len(miss_idx),
           "missed": miss_idx[:200], "clipped": stats["clipped"], "facesTurnedOut": flipped, "before": before,
           "after": mesh_counts(obj)}
    if cfg["edges"] is not None:
        out.update(steps=cfg["steps"], distance=cfg["distance"])
    if cfg["mirror"] is not None:
        out["mirror"] = {"modifier": mirror_mod.name, "axis": cfg["mirror"], "added": mirror_added,
                         "mirrorObject": mirror_mod.mirror_object.name if mirror_mod.mirror_object else None}
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- subdivide_multires
_MRES_KEYS = ("object", "name", "modifier", "mode", "levels", "sculptLevels", "viewportLevels", "renderLevels")
_MRES_MODES = ("CATMULL_CLARK", "SIMPLE", "LINEAR", "UNSUBDIVIDE", "REBUILD")


def _mres_levels(m):
    return {"viewport": m.levels, "sculpt": m.sculpt_levels, "render": m.render_levels, "total": m.total_levels}


def op_subdivide_multires(params):
    """Run a Multires modifier's Subdivide (Catmull-Clark, Simple, Linear), Unsubdivide or Rebuild Subdivisions, and
    set its viewport, sculpt and render levels.

    THE GAP (spec "Run Multires Subdivide and Unsubdivide, and set Multires levels", 2026-10-06): add_modifier makes a
    Multires and set_modifier writes its levels, but total_levels is read-only and nothing ran the subdivide operator,
    so the levels could never rise past 0 - and nothing rebuilt the lower levels of a dense sculpt for a light proxy.

    MEASURED FIRST (2026-10-06, 5.2.2 headless): object.multires_subdivide, multires_unsubdivide and
    multires_rebuild_subdiv all run under a context override with no window; Unsubdivide on a cube subdivided 3 times and
    applied went 386 -> 98 -> 26 -> 8 base vertices, then "No valid subdivisions found to rebuild a lower level".

    modifier (default: the object's first Multires). mode CATMULL_CLARK | SIMPLE | LINEAR adds `levels` (default 1)
    levels; UNSUBDIVIDE rebuilds one lower level per pass, `levels` passes (default 1) or fewer when Blender reports the
    lowest; REBUILD rebuilds every lower level at once. mode may be left out to set levels only. sculptLevels,
    viewportLevels, renderLevels (0 - the total after) are written after the run; after an UNSUBDIVIDE or REBUILD a value
    above the new total is clamped to it and reported. Reply: total levels before and after, the three levels, base
    vertices before and after, and the vertex count at the top level.
    """
    reject_unknown(params, _MRES_KEYS, "subdivide_multires")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    require_editable(obj, "subdivide")
    mname = take(params, "modifier", default=None, kind=str)
    if mname is not None:
        m = obj.modifiers.get(mname)
        if m is None or m.type != "MULTIRES":
            raise MifOpError("'%s' has no Multires modifier named %r (it has %s). NOTHING was changed."
                             % (obj.name, mname, [x.name for x in obj.modifiers if x.type == "MULTIRES"]))
    else:
        m = next((x for x in obj.modifiers if x.type == "MULTIRES"), None)
        if m is None:
            raise MifOpError("'%s' has no Multires modifier (add_modifier type MULTIRES adds one). NOTHING was changed."
                             % obj.name)
    mode = take(params, "mode", default=None, kind=str)
    mode = mode.upper() if mode is not None else None
    if mode is not None and mode not in _MRES_MODES:
        raise MifOpError("mode must be one of %s, got %r. NOTHING was changed." % (", ".join(_MRES_MODES), mode))
    n = take_int(params, "levels", default=None)
    if n is not None and (mode is None or mode == "REBUILD"):
        raise MifOpError("levels counts the passes of a subdivide or UNSUBDIVIDE; mode is %s. NOTHING was changed."
                         % (mode or "not given"))
    n = 1 if n is None else n
    if not 1 <= n <= 10:
        raise MifOpError("levels is 1-10, got %d. NOTHING was changed." % n)
    want = {}
    for key, attr in (("viewportLevels", "levels"), ("sculptLevels", "sculpt_levels"), ("renderLevels", "render_levels")):
        v = take_int(params, key, default=None)
        if v is not None:
            want[key] = (attr, v)
    if mode is None and not want:
        raise MifOpError("nothing to do: give mode (%s) and / or sculptLevels, viewportLevels, renderLevels. NOTHING was "
                         "changed." % ", ".join(_MRES_MODES))
    total0 = m.total_levels
    base0 = len(obj.data.vertices)
    adds = mode in ("CATMULL_CLARK", "SIMPLE", "LINEAR")
    top = total0 + n if adds else total0
    if adds and base0 * 4 ** top > _MULTIRES_MAX_VERTS:
        raise MifOpError("%d levels on %d base vertices would be about %d vertices at the top (cap %d). NOTHING was "
                         "changed." % (top, base0, base0 * 4 ** top, _MULTIRES_MAX_VERTS))
    cap = total0 + n if mode == "UNSUBDIVIDE" else (total0 + 10 if mode == "REBUILD" else top)
    for key, (_attr, v) in want.items():
        if not 0 <= v <= cap:
            raise MifOpError("%s %d is outside 0-%d (the total levels %s). NOTHING was changed."
                             % (key, v, cap, "after this call" if mode else "now"))
    if mode in ("UNSUBDIVIDE", "REBUILD") and obj.data.shape_keys is not None:
        raise MifOpError("'%s' has shape keys, and %s rebuilds the base mesh with fewer vertices - the keys cannot "
                         "follow. NOTHING was changed." % (obj.name, mode))
    before = _mres_levels(m)
    passes, lowest = 0, False
    with bpy.context.temp_override(object=obj, active_object=obj):
        if adds:
            for _k in range(n):
                r = bpy.ops.object.multires_subdivide(modifier=m.name, mode=mode)
                if "FINISHED" not in r:
                    raise MifOpError("Multires subdivide answered %s after %d level(s)." % (sorted(r), passes))
                passes += 1
        elif mode in ("UNSUBDIVIDE", "REBUILD"):
            op = bpy.ops.object.multires_unsubdivide if mode == "UNSUBDIVIDE" else bpy.ops.object.multires_rebuild_subdiv
            for _k in range(n if mode == "UNSUBDIVIDE" else 1):
                try:
                    r = op(modifier=m.name)
                except RuntimeError as exc:
                    if passes == 0:
                        raise MifOpError("Blender: %s - the base mesh is not a regular subdivision (a grid of quads "
                                         "from one level up). NOTHING was changed." % str(exc).strip()) from None
                    lowest = True
                    break
                if "FINISHED" not in r:
                    raise MifOpError("Multires %s answered %s after %d pass(es)." % (mode, sorted(r), passes))
                passes += 1
    clamped = {}
    for key, (attr, v) in want.items():
        val = min(v, m.total_levels)
        if val != v:
            clamped[key] = {"asked": v, "set": val}
        setattr(m, attr, val)
    bpy.context.view_layer.update()
    top_mesh = _multires_mesh(obj, m)
    try:
        top_verts = len(top_mesh.vertices)
    finally:
        bpy.data.meshes.remove(top_mesh)
    out = {"object": obj.name, "modifier": m.name, "mode": mode, "passes": passes, "before": before,
           "after": _mres_levels(m), "baseVerticesBefore": base0, "baseVerticesAfter": len(obj.data.vertices),
           "topLevelVertices": top_verts}
    if mode == "UNSUBDIVIDE":
        out["lowestReached"] = lowest
    if clamped:
        out["clamped"] = clamped
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- remesh_mesh
_REMESH_KEYS = ("object", "name", "mode", "targetFaces", "targetRatio", "targetEdgeLength", "symmetry", "preserveSharp",
                "preserveBoundary", "preserveAttributes", "smoothNormals", "seed", "voxelSize", "adaptivity",
                "fixPoles", "preserveVolume", "preserveVertexColors", "preserveFaceSets", "preserveMask")
_QUADRIFLOW_ONLY = ("targetFaces", "targetRatio", "targetEdgeLength", "symmetry", "preserveSharp", "preserveBoundary",
                    "smoothNormals", "seed")
_VOXEL_ONLY = ("voxelSize", "adaptivity", "fixPoles", "preserveVolume", "preserveVertexColors", "preserveFaceSets",
               "preserveMask")
_VOXEL_MAX_CELLS = 200_000_000
# The voxel remesh's three keep-flags became one (use_remesh_preserve_attributes) in Blender 4.1.
_VOXEL_LEGACY = {"preserveVertexColors": "use_remesh_preserve_vertex_colors",
                 "preserveFaceSets": "use_remesh_preserve_sculpt_face_sets",
                 "preserveMask": "use_remesh_preserve_paint_mask"}


def _mesh_shape_counts(me):
    """faces, vertices, triangles/quads/ngons, poles (valence 3 or 5+ off the boundary) and non-manifold edges."""
    edge_faces = [0] * len(me.edges)
    key_index = {e.key: e.index for e in me.edges}
    for p in me.polygons:
        for k in p.edge_keys:
            edge_faces[key_index[k]] += 1
    valence = [0] * len(me.vertices)
    on_rim = [False] * len(me.vertices)
    for e in me.edges:
        for v in e.vertices:
            valence[v] += 1
            if edge_faces[e.index] != 2:
                on_rim[v] = True
    sides = [p.loop_total for p in me.polygons]
    return {"faces": len(me.polygons), "vertices": len(me.vertices), "triangles": sides.count(3),
            "quads": sides.count(4), "ngons": len(sides) - sides.count(3) - sides.count(4),
            "poles": sum(1 for i, n in enumerate(valence) if not on_rim[i] and n != 4 and n > 0),
            "nonManifoldEdges": sum(1 for n in edge_faces if n != 2)}


def _weld_symmetry_seam(obj, sym):
    """QuadriFlow's symmetric result leaves its halves unwelded (measured 2026-10-06 on 5.2.2: a closed 32 x 16 sphere
    came back with 56 open edges at x = 0, each seam vertex within 0.0023 of its twin, the median edge 0.22). The open
    vertices within 5 % of the median edge of a symmetry plane are put on it and welded. Returns the vertices removed."""
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        lens = sorted(e.calc_length() for e in bm.edges)
        tol = 0.05 * lens[len(lens) // 2] if lens else 0.0
        rim = [v for v in bm.verts if any(len(e.link_faces) != 2 for e in v.link_edges)]
        seam = []
        for v in rim:
            axes = [k for k in range(3) if sym[k] and abs(v.co[k]) <= tol]
            for k in axes:
                v.co[k] = 0.0
            if axes:
                seam.append(v)
        n0 = len(bm.verts)
        if seam:
            bmesh.ops.remove_doubles(bm, verts=seam, dist=tol)
        removed = n0 - len(bm.verts)
        if removed:
            bm.normal_update()
            bm.to_mesh(obj.data)
            obj.data.update()
        return removed
    finally:
        bm.free()


def _remesh_parse(params, obj):
    mode = str(take(params, "mode", default="", kind=str)).upper()
    if mode not in ("QUADRIFLOW", "VOXEL"):
        raise MifOpError("mode must be QUADRIFLOW or VOXEL, got %r. NOTHING was changed." % mode)
    wrong = [k for k in (_VOXEL_ONLY if mode == "QUADRIFLOW" else _QUADRIFLOW_ONLY) if params.get(k) is not None]
    if wrong:
        raise MifOpError("%s belong to the other mode; mode is %s. NOTHING was changed." % (", ".join(wrong), mode))
    cfg = {"mode": mode}
    if mode == "QUADRIFLOW" and tuple(bpy.app.version[:2]) == (4, 4):
        # MEASURED 2026-10-06: quadriflow_remesh on a 32 x 16 UV sphere ABORTS Blender 4.4.0 (std::bad_alloc, the
        # process dies) headless, every run, with and without symmetry; 4.2.23, 5.0.1 and 5.2.2 finish it.
        raise MifOpError("QuadriFlow aborts Blender 4.4 (std::bad_alloc - the whole process dies, measured on 4.4.0), "
                         "so it is not run here; use 4.2 LTS or 5.x, or mode VOXEL. NOTHING was changed.")
    if mode == "QUADRIFLOW":
        given = [k for k in ("targetFaces", "targetRatio", "targetEdgeLength") if params.get(k) is not None]
        if len(given) != 1:
            raise MifOpError("QUADRIFLOW takes one target: targetFaces, targetRatio or targetEdgeLength (got %s). "
                             "NOTHING was changed." % (given or "none"))
        if given[0] == "targetFaces":
            v = take_int(params, "targetFaces")
            if not 4 <= v <= 500_000:
                raise MifOpError("targetFaces is 4-500000, got %d. NOTHING was changed." % v)
            cfg["qf"] = {"mode": "FACES", "target_faces": v}
        else:
            # QuadriFlow's exec reads target_faces only: RATIO and EDGE are turned into it by the panel's update
            # callbacks, which a call never runs (measured 2026-10-06 on 5.2.2: ratio 0.5 and edge 0.4 and 0.2 on a
            # 512-face sphere all gave 3,962 faces - the 4,000 default). So the count is derived here as the panel
            # does: faces x ratio, or the mesh's area / edge length squared (object space).
            key = given[0]
            v = take_float(params, key)
            if key == "targetRatio" and not 0.0 < v <= 10.0:
                raise MifOpError("targetRatio is above 0 and at most 10, got %r. NOTHING was changed." % v)
            if key == "targetEdgeLength" and not v > 0.0:
                raise MifOpError("targetEdgeLength is above 0, got %r. NOTHING was changed." % v)
            me = obj.data
            n = round(len(me.polygons) * v) if key == "targetRatio" else round(
                sum(p.area for p in me.polygons) / (v * v))
            if not 4 <= n <= 500_000:
                raise MifOpError("%s %g comes to %d faces on '%s' - outside 4-500000. NOTHING was changed."
                                 % (key, v, n, obj.name))
            cfg["qf"] = {"mode": "FACES", "target_faces": n}
            cfg["derived"] = {key: v, "targetFaces": n}
        sym = params.get("symmetry")
        if sym is not None and (not isinstance(sym, (list, tuple)) or len(sym) != 3
                                or not all(isinstance(x, bool) for x in sym)):
            raise MifOpError("symmetry is [x, y, z] booleans (the mesh's own axes), got %r. NOTHING was changed."
                             % (sym,))
        cfg["symmetry"] = list(sym) if sym is not None else None
        seed = take_int(params, "seed", default=0)
        if seed < 0:
            raise MifOpError("seed is 0 or more, got %d. NOTHING was changed." % seed)
        cfg["qf"].update(use_preserve_sharp=take_bool(params, "preserveSharp", default=False),
                         use_preserve_boundary=take_bool(params, "preserveBoundary", default=False),
                         smooth_normals=take_bool(params, "smoothNormals", default=False), seed=seed,
                         use_mesh_symmetry=sym is not None and any(sym))
        # preserve_attributes IS NOT ON EVERY BUILD: 3.6.23's quadriflow_remesh has preserve_paint_mask in its place
        # (read 2026-10-10; 4.2.17 has preserve_attributes), and passing it raised a TypeError out of the op on every
        # QuadriFlow call. Read off the operator: written where it exists, refused by name where asked and absent.
        keep_attrs = take_bool(params, "preserveAttributes", default=False)
        qf_props = [p.identifier for p in bpy.ops.object.quadriflow_remesh.get_rna_type().properties]
        if "preserve_attributes" in qf_props:
            cfg["qf"]["preserve_attributes"] = keep_attrs
        elif keep_attrs:
            raise MifOpError("this Blender's (%s) QuadriFlow has no preserve_attributes (its keep flag is %s). NOTHING "
                             "was changed." % (bpy.app.version_string,
                                               ", ".join(p for p in qf_props if p.startswith("preserve")) or "none"))
        return cfg
    me = obj.data
    size = take_float(params, "voxelSize", default=None)
    size = me.remesh_voxel_size if size is None else size
    if not size > 0.0:
        raise MifOpError("voxelSize is a length above 0 (object space), got %r. NOTHING was changed." % size)
    dims = [max(v.co[k] for v in me.vertices) - min(v.co[k] for v in me.vertices) for k in range(3)] \
        if len(me.vertices) else [0.0, 0.0, 0.0]
    cells = 1.0
    for d in dims:
        cells *= max(1.0, d / size + 2.0)
    if cells > _VOXEL_MAX_CELLS:
        raise MifOpError("voxelSize %g over a %s box is about %d voxels (cap %d) - raise voxelSize. NOTHING was "
                         "changed." % (size, [round(d, 4) for d in dims], cells, _VOXEL_MAX_CELLS))
    adapt = take_float(params, "adaptivity", default=None)
    if adapt is not None and not 0.0 <= adapt <= 1.0:
        raise MifOpError("adaptivity is 0-1, got %r. NOTHING was changed." % adapt)
    props = {"remesh_voxel_size": size}
    if adapt is not None:
        props["remesh_voxel_adaptivity"] = adapt
    for key, attr in (("fixPoles", "use_remesh_fix_poles"), ("preserveVolume", "use_remesh_preserve_volume")):
        if params.get(key) is not None:
            props[attr] = take_bool(params, key)
    keep = {k: take_bool(params, k) for k in ("preserveAttributes",) + tuple(_VOXEL_LEGACY) if params.get(k) is not None}
    if hasattr(me, "use_remesh_preserve_attributes"):
        vals = set(keep.values())
        if len(vals) > 1:
            raise MifOpError("this Blender keeps vertex colors, face sets and the mask with ONE flag "
                             "(use_remesh_preserve_attributes, since 4.1), and %s disagree. NOTHING was changed." % keep)
        if vals:
            props["use_remesh_preserve_attributes"] = vals.pop()
    else:
        if "preserveAttributes" in keep:
            raise MifOpError("this Blender has no single preserve flag: give preserveVertexColors, preserveFaceSets "
                             "and preserveMask. NOTHING was changed.")
        for k, attr in _VOXEL_LEGACY.items():
            if k in keep:
                props[attr] = keep[k]
    cfg["props"] = props
    return cfg


def op_remesh_mesh(params):
    """Remesh a mesh in place: QUADRIFLOW to a target face count, ratio or edge length, or VOXEL with the voxel
    operator's own options - fix poles, preserve volume, keep vertex colors, face sets and mask.

    THE GAP (spec "Quad remesh to a target count, and the voxel-remesh options", 2026-10-06): a voxel remesh was only
    the Remesh modifier applied; nothing ran QuadriFlow (a scan to about 1,500 vertices before UVs and a bake, a sculpt
    to 3,500 quads) and the voxel operator's extras are MESH properties the modifier does not have.

    MEASURED FIRST (2026-10-06, 4.2.23 and 5.2.2 headless): object.quadriflow_remesh(mode FACES, target_faces 300) on a
    32 x 16 UV sphere finished in the call, 333-335 faces; object.voxel_remesh ran with no window. The mesh's remesh
    properties on 4.2-5.2 are remesh_voxel_size, remesh_voxel_adaptivity, use_remesh_fix_poles,
    use_remesh_preserve_volume and use_remesh_preserve_attributes (one flag since 4.1 for colors, face sets and mask).

    QUADRIFLOW: one of targetFaces | targetRatio | targetEdgeLength (the last two turned into a face count here, as
    Blender's panel does - its exec reads the count only); symmetry [x, y, z] (the mesh's own axes, kept
    symmetric; the open seam QuadriFlow leaves is welded), preserveSharp, preserveBoundary, preserveAttributes, smoothNormals, seed. VOXEL: voxelSize (object
    space; default the mesh's own), adaptivity 0-1, fixPoles, preserveVolume, preserveAttributes or
    preserveVertexColors / preserveFaceSets / preserveMask (one flag on 4.1+, so they must agree there); the values are
    left on the mesh, as Blender's own panel leaves them. A mesh with non-manifold edges (open, or three faces on an
    edge) is refused with the count - close it first (fill_mesh_loop); so is one with shape keys or a Multires.
    QUADRIFLOW is refused on Blender 4.4: it aborted 4.4.0 (std::bad_alloc) on a plain sphere, every run.
    Reply: faces, vertices, quads, triangles and poles before and after.
    """
    reject_unknown(params, _REMESH_KEYS, "remesh_mesh")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    require_editable(obj, "remesh")
    me = obj.data
    if me.shape_keys is not None:
        raise MifOpError("'%s' has shape keys, and a remesh makes new topology they cannot follow. NOTHING was changed."
                         % obj.name)
    if any(m.type == "MULTIRES" for m in obj.modifiers):
        raise MifOpError("'%s' has a Multires modifier, whose levels are bound to the base topology. NOTHING was "
                         "changed." % obj.name)
    cfg = _remesh_parse(params, obj)
    before = _mesh_shape_counts(me)
    if not before["faces"]:
        raise MifOpError("'%s' has no faces to remesh. NOTHING was changed." % obj.name)
    if before["nonManifoldEdges"]:
        raise MifOpError("'%s' has %d non-manifold edges (open or shared by 3+ faces) - %s needs a closed surface; close "
                         "the holes first (fill_mesh_loop). NOTHING was changed."
                         % (obj.name, before["nonManifoldEdges"], cfg["mode"]))
    ctx = dict(object=obj, active_object=obj, selected_objects=[obj], selected_editable_objects=[obj])
    if cfg["mode"] == "QUADRIFLOW":
        mirror_before = (me.use_mirror_x, me.use_mirror_y, me.use_mirror_z)
        if cfg["symmetry"] is not None:
            me.use_mirror_x, me.use_mirror_y, me.use_mirror_z = cfg["symmetry"]
        try:
            with bpy.context.temp_override(**ctx):
                r = bpy.ops.object.quadriflow_remesh(**cfg["qf"])
        except RuntimeError as exc:
            me.use_mirror_x, me.use_mirror_y, me.use_mirror_z = mirror_before
            raise MifOpError("QuadriFlow: %s. NOTHING was changed." % str(exc).strip()) from None
        finally:
            if cfg["symmetry"] is not None:
                me.use_mirror_x, me.use_mirror_y, me.use_mirror_z = mirror_before
        if "FINISHED" not in r:
            raise MifOpError("QuadriFlow answered %s. NOTHING was changed." % sorted(r))
        settings = dict(cfg["qf"], symmetry=cfg["symmetry"])
        if cfg.get("derived"):
            settings["derivedFrom"] = cfg["derived"]
        if cfg["symmetry"] is not None and any(cfg["symmetry"]):
            settings["seamWelded"] = _weld_symmetry_seam(obj, cfg["symmetry"])
    else:
        old = {k: getattr(me, k) for k in cfg["props"]}
        for k, v in cfg["props"].items():
            setattr(me, k, v)
        try:
            with bpy.context.temp_override(**ctx):
                r = bpy.ops.object.voxel_remesh()
        except RuntimeError as exc:
            for k, v in old.items():
                setattr(me, k, v)
            raise MifOpError("voxel remesh: %s. NOTHING was changed." % str(exc).strip()) from None
        if "FINISHED" not in r:
            for k, v in old.items():
                setattr(me, k, v)
            raise MifOpError("voxel remesh answered %s. NOTHING was changed." % sorted(r))
        # read back off the mesh after the run - the values the next voxel remesh in Blender's panel starts from
        settings = {k: (round(getattr(obj.data, k), 6) if isinstance(getattr(obj.data, k), float)
                        else getattr(obj.data, k)) for k in cfg["props"]}
    bpy.context.view_layer.update()
    after = _mesh_shape_counts(obj.data)
    out = {"object": obj.name, "mode": cfg["mode"], "settings": settings, "before": before, "after": after}
    out.update(shared_data_note(obj))
    return out


# --------------------------------------------------------------------------------------------- set_face_sets
_FSET_KEYS = ("object", "name", "mode", "faces", "box", "setId", "iterations", "strength")
_FSET_MODES = ("FROM_SEAMS", "FROM_MATERIAL", "FROM_LOOSE_PARTS", "FROM_VISIBLE", "FROM_FACES", "FROM_BOX", "CLEAR",
               "RELAX")


def _face_set_values(me):
    attr = me.attributes.get(_FACE_SET_ATTR)
    if attr is None or attr.domain != "FACE" or attr.data_type != "INT":
        return None
    vals = [0] * len(me.polygons)
    attr.data.foreach_get("value", vals)
    return vals


def _face_set_rows(vals):
    if vals is None:
        return []
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return [{"id": k, "faces": counts[k]} for k in sorted(counts)]


def _face_regions(bm, cut):
    """Connected face regions across edges where cut(edge) is False. Returns a region number per face index."""
    region = [-1] * len(bm.faces)
    n = 0
    for f0 in bm.faces:
        if region[f0.index] >= 0:
            continue
        region[f0.index] = n
        stack = [f0]
        while stack:
            f = stack.pop()
            for e in f.edges:
                if cut(e):
                    continue
                for g in e.link_faces:
                    if region[g.index] < 0:
                        region[g.index] = n
                        stack.append(g)
        n += 1
    return region


def _relax_face_set_borders(obj, vals, iterations, strength):
    """The Relax Face Sets mesh filter, in mesh space: every vertex on a border between two sets moves toward the mean
    of its border neighbours, `iterations` times by `strength` - a jagged border becomes a smooth line on the surface.
    Returns (vertices moved, the largest move in world units)."""
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        def on_border(v):
            return len({vals[f.index] for f in v.link_faces}) > 1
        border = [v for v in bm.verts if on_border(v) and not any(e.is_boundary for e in v.link_edges)]
        if not border:
            return 0, 0.0, False
        bset = set(border)
        start = {v: v.co.copy() for v in border}
        for _i in range(iterations):
            new = {}
            for v in border:
                nb = [e.other_vert(v) for e in v.link_edges if e.other_vert(v) in bset]
                if len(nb) < 2:
                    continue
                mean = sum((w.co for w in nb), Vector()) / len(nb)
                new[v] = v.co.lerp(mean, strength)
            for v, c in new.items():
                v.co = c
        scale = _uniform_scale(obj) or 1.0
        moved = [v for v in border if (v.co - start[v]).length > 1e-9]
        max_d = max([(v.co - start[v]).length for v in moved] or [0.0]) * scale
        if moved:
            bm.normal_update()
            bm.to_mesh(obj.data)
            obj.data.update()
        return len(moved), max_d, True
    finally:
        bm.free()


def op_set_face_sets(params):
    """Make, change or clear a mesh's sculpt face sets - from UV seams, materials, loose parts, the visible faces, a
    face list or a world box - or relax the borders between them.

    THE GAP (spec "Create and use sculpt face sets", 2026-10-06): face sets split a mesh into regions without separating
    it, and they are the mask of sculpt_stroke (mask.faceSets) and of every cloth-style brush - but nothing made them,
    so a mesh had them only if it came from Blender's sculpt mode. A face set is Blender's own INT face attribute
    .sculpt_face_set, written here directly, so it works headless where the sculpt operators need a view.

    mode FROM_SEAMS (each region bounded by UV seams is a set), FROM_MATERIAL (set = material slot + 1),
    FROM_LOOSE_PARTS (each connected part a set), FROM_VISIBLE (the faces not hidden become one new set, the hidden keep
    theirs), FROM_FACES (faces [indices]) or FROM_BOX (box {min, max} world: the faces whose centre is inside) into
    setId - a new set (one above the highest) by default, or an existing id to EXTEND that set - CLEAR (the attribute
    removed: no face sets) or RELAX (the Relax Face Sets filter: each vertex on a border between sets moves toward its
    border neighbours, iterations (default 5) x strength (default 0.5); open-boundary vertices stay). The FROM_ modes
    over the whole mesh (seams, material, loose parts) replace every set; FROM_FACES / FROM_BOX / FROM_VISIBLE start a
    mesh with no face sets at set 1. Reply: setCount and every set's id and face count, before and after.
    """
    reject_unknown(params, _FSET_KEYS, "set_face_sets")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    require_editable(obj, "change")
    me = obj.data
    mode = str(take(params, "mode", default="", kind=str)).upper()
    if mode not in _FSET_MODES:
        raise MifOpError("mode must be one of %s, got %r. NOTHING was changed." % (", ".join(_FSET_MODES), mode))
    if not len(me.polygons):
        raise MifOpError("'%s' has no faces. NOTHING was changed." % obj.name)
    allowed = {"FROM_FACES": {"faces", "setId"}, "FROM_BOX": {"box", "setId"}, "FROM_VISIBLE": {"setId"},
               "RELAX": {"iterations", "strength"}}.get(mode, set())
    extra = [k for k in ("faces", "box", "setId", "iterations", "strength") if params.get(k) is not None
             and k not in allowed]
    if extra:
        raise MifOpError("%s do not go with mode %s. NOTHING was changed." % (", ".join(extra), mode))
    vals = _face_set_values(me)
    before = _face_set_rows(vals)
    out = {"object": obj.name, "mode": mode, "before": before}
    if mode == "CLEAR":
        if vals is None:
            raise MifOpError("'%s' has no face sets to clear. NOTHING was changed." % obj.name)
        me.attributes.remove(me.attributes[_FACE_SET_ATTR])
        me.update()
        out.update(after=[], setCount=0)
        return out
    if mode == "RELAX":
        if vals is None or len(set(vals)) < 2:
            raise MifOpError("'%s' has %s - RELAX smooths the borders BETWEEN sets. NOTHING was changed."
                             % (obj.name, "no face sets" if vals is None else "one face set only"))
        if me.shape_keys is not None:
            raise MifOpError("'%s' has shape keys, and RELAX moves the base mesh's vertices. NOTHING was changed."
                             % obj.name)
        it = take_int(params, "iterations", default=5)
        if not 1 <= it <= 100:
            raise MifOpError("iterations is 1-100, got %d. NOTHING was changed." % it)
        k = take_float(params, "strength", default=0.5)
        if not 0.0 < k <= 1.0:
            raise MifOpError("strength is above 0 and at most 1, got %r. NOTHING was changed." % k)
        moved, max_d, had = _relax_face_set_borders(obj, vals, it, k)
        if not had:
            raise MifOpError("every border vertex of '%s' lies on an open boundary, which RELAX leaves. NOTHING was "
                             "changed." % obj.name)
        bpy.context.view_layer.update()
        out.update(iterations=it, strength=k, verticesMoved=moved, maxMove=round(max_d, 6), after=before,
                   setCount=len(before))
        return out
    n = len(me.polygons)
    new = list(vals) if vals is not None else [1] * n
    targets = None
    if mode in ("FROM_FACES", "FROM_BOX", "FROM_VISIBLE"):
        set_id = take_int(params, "setId", default=None)
        if set_id is not None and set_id < 1:
            raise MifOpError("setId is 1 or more (Blender's face sets count from 1), got %d. NOTHING was changed."
                             % set_id)
        if mode == "FROM_FACES":
            raw = params.get("faces")
            if not isinstance(raw, (list, tuple)) or not raw:
                raise MifOpError("FROM_FACES needs faces, a non-empty list of face indices. NOTHING was changed.")
            bad = [i for i in raw if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n]
            if bad:
                raise MifOpError("faces %s are not face indices of '%s' (it has %d). NOTHING was changed."
                                 % (bad[:10], obj.name, n))
            targets = list(dict.fromkeys(raw))
        elif mode == "FROM_BOX":
            box = params.get("box")
            if not isinstance(box, dict) or set(box) != {"min", "max"}:
                raise MifOpError("box is {min: [x,y,z], max: [x,y,z]} in world units, got %r. NOTHING was changed."
                                 % (box,))
            lo = Vector(finite_floats(list(box["min"]), "box.min")) if isinstance(box["min"], (list, tuple)) \
                and len(box["min"]) == 3 else None
            hi = Vector(finite_floats(list(box["max"]), "box.max")) if isinstance(box["max"], (list, tuple)) \
                and len(box["max"]) == 3 else None
            if lo is None or hi is None or any(lo[k] > hi[k] for k in range(3)):
                raise MifOpError("box.min and box.max are [x,y,z] with min <= max on every axis, got %r. NOTHING was "
                                 "changed." % (box,))
            mw = obj.matrix_world
            targets = [p.index for p in me.polygons
                       if all(lo[k] <= (mw @ p.center)[k] <= hi[k] for k in range(3))]
            if not targets:
                raise MifOpError("no face of '%s' has its centre inside the box %s - %s. NOTHING was changed."
                                 % (obj.name, [list(lo), list(hi)], "nothing to put in a set"))
        else:
            targets = [p.index for p in me.polygons if not p.hide]
            if not targets:
                raise MifOpError("every face of '%s' is hidden - FROM_VISIBLE has nothing to put in a set. NOTHING was "
                                 "changed." % obj.name)
        have = set(new) if vals is not None else set()
        extended = set_id is not None and set_id in have
        if set_id is None:
            set_id = (max(have) + 1) if have else (2 if len(targets) < n else 1)
        for i in targets:
            new[i] = set_id
        out.update(setId=set_id, extended=extended, facesAssigned=len(targets))
    else:
        bm = bmesh.new()
        try:
            bm.from_mesh(me)
            bm.faces.ensure_lookup_table()
            bm.faces.index_update()
            if mode == "FROM_MATERIAL":
                new = [f.material_index + 1 for f in bm.faces]
            elif mode == "FROM_LOOSE_PARTS":
                new = [r + 1 for r in _face_regions(bm, lambda e: False)]
            else:
                if not any(e.seam for e in bm.edges):
                    raise MifOpError("'%s' has no UV seam (mark_seams marks them). NOTHING was changed." % obj.name)
                new = [r + 1 for r in _face_regions(bm, lambda e: e.seam)]
        finally:
            bm.free()
    attr = me.attributes.get(_FACE_SET_ATTR)
    if attr is not None and (attr.domain != "FACE" or attr.data_type != "INT"):
        raise MifOpError("'%s' has a %s attribute that is not an INT face attribute. NOTHING was changed."
                         % (obj.name, _FACE_SET_ATTR))
    if attr is None:
        attr = me.attributes.new(_FACE_SET_ATTR, "INT", "FACE")
    attr.data.foreach_set("value", new)
    me.update()
    after = _face_set_rows(_face_set_values(me))
    out.update(after=after, setCount=len(after))
    return out


def _preview_specs():
    # THE BULK PREVIEW of cloth_filter (ops_preview.py; docs/15 section 5, 2026-10-09): planned on a scratch copy, as the
    # other mesh writes are - the filter's settle passes are the result, and they run in the pass that writes it.
    from . import ops_preview as preview   # noqa: PLC0415
    return {"cloth_filter": preview.mesh_spec(op_cloth_filter, keys=("object",))}


PREVIEW = _preview_specs()

OPS = {
    "relax_loop": op_relax_loop,
    "extrude_mesh": op_extrude_mesh,
    "loop_cut": op_loop_cut,
    "slide_edges": op_slide_edges,
    "inset_faces": op_inset_faces,
    "knife_cut": op_knife_cut,
    "join_boundary_loops": op_join_boundary_loops,
    "dissolve_mesh_elements": op_dissolve_mesh_elements,
    "fill_mesh_loop": op_fill_mesh_loop,
    "merge_vertices": op_merge_vertices,
    "offset_edges": op_offset_edges,
    "subdivide_mesh": op_subdivide_mesh,
    "sculpt_stroke": op_sculpt_stroke,
    "cloth_filter": op_cloth_filter,
    "mesh_from_outline": op_mesh_from_outline,
    "mesh_sections": op_mesh_sections,
    "cut_eyelets": op_cut_eyelets,
    "rip_edges": op_rip_edges,
    "circularize": op_circularize,
    "snap_to_surface": op_snap_to_surface,
    "add_quads_on_surface": op_add_quads_on_surface,
    "subdivide_multires": op_subdivide_multires,
    "remesh_mesh": op_remesh_mesh,
    "set_face_sets": op_set_face_sets,
}
