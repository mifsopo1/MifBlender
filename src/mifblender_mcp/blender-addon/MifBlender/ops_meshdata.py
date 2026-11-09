"""MifBlender ops: a mesh's data written or carried across - custom normals, data transfer of UVs / colors / normals /
flags / attributes, shape keys onto another topology, generic attributes, hooks and spinning part of a mesh (1.3 gap
audit, 2026-10-19).

THE GAP. set_shading wrote smooth, flat, angle and WEIGHTED normals only, so foliage normals from a hemisphere and
hard-surface normals from a smooth proxy had no route; the Data Transfer modifier, the hand route for UVs and colors onto
a retopology, transfers NOTHING into a layer the target lacks and still answers ok (object.datalayout_transfer must run
first); add_shape_key fromObject needs the same vertex count; nothing wrote a float, int or vector attribute; add_modifier
HOOK never set the inverse matrix, so a hook away from the origin threw the vertices across the scene; and nothing spun
part of an existing mesh.
"""
from __future__ import annotations

import math

import bpy

from .ops_common import (MifOpError, finite_floats, finite_int, get_object, reject_unknown, require_editable, take,
                         take_bool, take_float, take_int)


def _vec3(params, key, tail, required=False, default=None):
    raw = params.get(key)
    if raw is None:
        if required:
            raise MifOpError("%s is required ([x, y, z]). %s" % (key, tail))
        return default
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise MifOpError("%s is [x, y, z], got %r. %s" % (key, raw, tail))
    return finite_floats(list(raw), key)


def _corner_normals(mesh):
    """One normal per corner (loop), in object space, as Blender shades them now."""
    if "calc_normals_split" in mesh.bl_rna.functions:          # 3.6 - 4.0
        mesh.calc_normals_split()
        return [tuple(lp.normal) for lp in mesh.loops]
    return [tuple(n.vector) for n in mesh.corner_normals]


def _angle_deg(a, b):
    d = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
    return math.degrees(math.acos(d))


def _norm(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / n, v[1] / n, v[2] / n) if n > 1e-12 else None


def _vgroup_weights(obj, name, tail):
    if name is None:
        return None
    vg = obj.vertex_groups.get(name)
    if vg is None:
        raise MifOpError("'%s' has no vertex group '%s'. It has: %s. %s"
                         % (obj.name, name, ", ".join(g.name for g in obj.vertex_groups) or "(none)", tail))
    w = [0.0] * len(obj.data.vertices)
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == vg.index:
                w[v.index] = g.weight
    return w


# --------------------------------------------------------------------------------------------- set_custom_normals

_CN_KEYS = ("object", "source", "proxy", "center", "direction", "vertexGroup", "mix")
_CN_SOURCES = ("PROXY", "SPHERE", "DIRECTION", "FROM_FACES", "SMOOTH", "CLEAR")


def op_set_custom_normals(params):
    """Write custom split normals by a rule, mixed under an optional vertex-group mask, and read back how far each
    corner's normal moved.

    params:
      object (str)          a mesh in object mode
      source (str)          PROXY (the nearest point of another mesh's smooth surface - a hemisphere over a canopy, a
                            smooth proxy round a hard-surface part) | SPHERE (away from center) | DIRECTION (one vector)
                            | FROM_FACES (each corner its face's normal - flat) | SMOOTH (each corner its vertex's
                            normal, averaged across sharp edges) | CLEAR (remove custom normals)
      proxy (str)           PROXY: the mesh read
      center ([x,y,z])      SPHERE: world-space center (default the object's bounding-box center)
      direction ([x,y,z])   DIRECTION: world-space vector (e.g. [0, 0, 1])
      vertexGroup (str)     only these vertices, by weight (a weight of 0.5 moves half way)
      mix (float)           0-1, default 1: how far toward the rule's normal each corner moves
    3.6 shows custom normals only with Auto Smooth on, so it is turned on there and the reply says so (4.1 and later
    have no such switch). FBX carries custom normals; Unreal's importer reads them with Normal Import Method Import
    Normals.
    Reply: corners, corners changed, the angle moved (min / mean / max degrees).
    """
    reject_unknown(params, _CN_KEYS, "set_custom_normals")
    tail = "NOTHING was changed."
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    require_editable(obj, "changed")
    mesh = obj.data
    if not mesh.polygons:
        raise MifOpError("'%s' has no faces - nothing to shade. %s" % (obj.name, tail))
    src = str(take(params, "source", required=True, kind=str)).upper()
    if src not in _CN_SOURCES:
        raise MifOpError("source must be one of %s, got %r. %s" % (", ".join(_CN_SOURCES), params.get("source"), tail))
    for key, owner in (("proxy", "PROXY"), ("center", "SPHERE"), ("direction", "DIRECTION")):
        if params.get(key) is not None and src != owner:
            raise MifOpError("%s is %s's; this is %s. %s" % (key, owner, src, tail))
    mix = take_float(params, "mix", default=1.0)
    if not 0.0 <= mix <= 1.0:
        raise MifOpError("mix is 0-1, got %g. %s" % (mix, tail))
    weights = _vgroup_weights(obj, take(params, "vertexGroup", default=None, kind=str), tail)
    auto_smooth_set = False
    if "use_auto_smooth" in mesh.bl_rna.properties and src != "CLEAR" and not mesh.use_auto_smooth:
        mesh.use_auto_smooth = True
        auto_smooth_set = True
    before = _corner_normals(mesh)
    if src == "CLEAR":
        had = bool(getattr(mesh, "has_custom_normals", False))
        with bpy.context.temp_override(object=obj, active_object=obj):
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
        after = _corner_normals(mesh)
        moved = [_angle_deg(a, b) for a, b in zip(before, after)]
        return {"object": obj.name, "source": src, "hadCustomNormals": had,
                "hasCustomNormals": bool(getattr(mesh, "has_custom_normals", False)), "corners": len(after),
                "cornersChanged": sum(1 for m in moved if m > 0.01),
                "angleMovedDeg": {"max": round(max(moved), 4), "mean": round(sum(moved) / len(moved), 4)}}
    mw = obj.matrix_world
    nmat = mw.to_3x3().inverted_safe().transposed()            # world normal -> object normal is its inverse
    to_obj = nmat.inverted_safe()
    loops_v = [lp.vertex_index for lp in mesh.loops]
    loop_poly = [0] * len(mesh.loops)
    for p in mesh.polygons:
        for li in range(p.loop_start, p.loop_start + p.loop_total):
            loop_poly[li] = p.index
    target = [None] * len(mesh.loops)
    if src == "FROM_FACES":
        fn = [tuple(p.normal) for p in mesh.polygons]
        target = [fn[loop_poly[i]] for i in range(len(target))]
    elif src == "SMOOTH":
        vn = [tuple(v.normal) for v in mesh.vertices]
        target = [vn[loops_v[i]] for i in range(len(target))]
    elif src in ("SPHERE", "DIRECTION"):
        from mathutils import Vector                           # noqa: PLC0415
        if src == "DIRECTION":
            d = _norm(_vec3(params, "direction", tail, required=True))
            if d is None:
                raise MifOpError("direction has zero length. %s" % tail)
            dn = _norm(tuple(to_obj @ Vector(d)))
            target = [dn] * len(target)
        else:
            c = _vec3(params, "center", tail)
            if c is None:
                corners = [mw @ Vector(b) for b in obj.bound_box]
                c = tuple(sum(p[i] for p in corners) / 8.0 for i in range(3))
            cw = Vector(c)
            vdir = []
            for v in mesh.vertices:
                w = _norm(tuple((mw @ v.co) - cw))
                vdir.append(_norm(tuple(to_obj @ Vector(w))) if w else None)
            target = [vdir[loops_v[i]] for i in range(len(target))]
    elif src == "PROXY":
        from mathutils import Vector                           # noqa: PLC0415
        from mathutils.bvhtree import BVHTree                  # noqa: PLC0415
        prox = get_object(take(params, "proxy", required=True, kind=str), want_mesh=True)
        if prox is obj:
            raise MifOpError("the proxy is the object itself - SMOOTH averages its own normals. %s" % tail)
        dg = bpy.context.evaluated_depsgraph_get()
        pe = prox.evaluated_get(dg)
        pm = pe.to_mesh()
        try:
            if not pm.polygons:
                raise MifOpError("proxy '%s' has no faces. %s" % (prox.name, tail))
            pmw = prox.matrix_world
            pn = pmw.to_3x3().inverted_safe().transposed()
            verts = [pmw @ v.co for v in pm.vertices]
            vnorm = [(pn @ v.normal).normalized() for v in pm.vertices]
            pm.calc_loop_triangles()
            tris = [tuple(t.vertices) for t in pm.loop_triangles]
            tree = BVHTree.FromPolygons(verts, tris)
            vdir = []
            for v in mesh.vertices:
                p = mw @ v.co
                hit = tree.find_nearest(p)
                if hit[0] is None:
                    vdir.append(None)
                    continue
                loc, fnorm, ti, _d = hit
                a, b, cc = (verts[k] for k in tris[ti])
                # barycentric weights of the hit point, then the proxy's smooth (vertex) normal there
                v0, v1, v2 = b - a, cc - a, loc - a
                d00, d01, d11 = v0.dot(v0), v0.dot(v1), v1.dot(v1)
                d20, d21 = v2.dot(v0), v2.dot(v1)
                den = d00 * d11 - d01 * d01
                if abs(den) < 1e-18:
                    n = fnorm
                else:
                    wv = (d11 * d20 - d01 * d21) / den
                    ww = (d00 * d21 - d01 * d20) / den
                    wu = 1.0 - wv - ww
                    na, nb, nc = (vnorm[k] for k in tris[ti])
                    n = na * wu + nb * wv + nc * ww
                vdir.append(_norm(tuple(to_obj @ Vector(tuple(n)))))
            target = [vdir[loops_v[i]] for i in range(len(target))]
        finally:
            pe.to_mesh_clear()
    out_n = []
    for i, cur in enumerate(before):
        t = target[i]
        f = mix * (weights[loops_v[i]] if weights is not None else 1.0)
        if t is None or f <= 0.0:
            out_n.append(cur)
            continue
        out_n.append(_norm(tuple(cur[k] * (1.0 - f) + t[k] * f for k in range(3))) or cur)
    mesh.normals_split_custom_set(out_n)
    mesh.update()
    after = _corner_normals(mesh)
    moved = [_angle_deg(a, b) for a, b in zip(before, after)]
    err = [_angle_deg(a, b) for a, b in zip(out_n, after)]
    out = {"object": obj.name, "source": src, "mix": mix, "corners": len(after),
           "cornersChanged": sum(1 for m in moved if m > 0.01),
           "angleMovedDeg": {"min": round(min(moved), 4), "mean": round(sum(moved) / len(moved), 4),
                             "max": round(max(moved), 4)},
           "readBackErrorDeg": round(max(err), 4), "hasCustomNormals": bool(getattr(mesh, "has_custom_normals", True)),
           "autoSmoothSet": auto_smooth_set}
    if auto_smooth_set:
        out["note"] = ("Blender %s shows custom normals only with Auto Smooth on, so it was turned on for '%s'"
                       % (bpy.app.version_string, mesh.name))
    if out["readBackErrorDeg"] > 1.0:
        out["warning"] = ("the normals read back differ from the ones written by up to %.2f degrees - Blender keeps "
                          "custom normals per smooth fan, so corners on one fan share one normal"
                          % out["readBackErrorDeg"])
    return out


# --------------------------------------------------------------------------------------------- transfer_mesh_data

_TM_KEYS = ("source", "targets", "target", "data", "mapping", "vertexGroup", "mix", "attributes")
_TM_DATA = ("UV", "COLOR", "CUSTOM_NORMAL", "SEAM", "SHARP_EDGE", "CREASE", "BEVEL_WEIGHT_EDGE", "SMOOTH",
            "FACE_SETS", "ATTRIBUTE")
_TM_MAPPING = {
    #          loops                  verts                   edges                 polys
    "NEAREST_FACE": ("POLYINTERP_NEAREST", "POLYINTERP_NEAREST", "NEAREST", "NEAREST"),
    "POLYINTERP": ("POLYINTERP_LNORPROJ", "POLYINTERP_VNORPROJ", "EDGEINTERP_VNORPROJ", "POLYINTERP_PNORPROJ"),
    "NEAREST": ("NEAREST_POLY", "NEAREST", "NEAREST", "NEAREST"),
    "TOPOLOGY": ("TOPOLOGY", "TOPOLOGY", "TOPOLOGY", "TOPOLOGY"),
}
_TM_NOTES = {"weights": "vertex weights have their own op, transfer_weights (group choice, a destination mask)",
             "VGROUP_WEIGHTS": "vertex weights have their own op, transfer_weights"}


def _layer_snapshot(mesh):
    import numpy as np                                          # noqa: PLC0415
    snap = {}
    for uv in mesh.uv_layers:
        a = np.empty(len(mesh.loops) * 2, dtype=np.float32)
        uv.data.foreach_get("uv", a)
        snap["UV:" + uv.name] = a
    for at in getattr(mesh, "color_attributes", []):
        a = np.empty(len(at.data) * 4, dtype=np.float32)
        at.data.foreach_get("color", a)
        snap["COLOR:" + at.name] = a
    return snap


def _flag_counts(mesh):
    out = {"SEAM": sum(1 for e in mesh.edges if e.use_seam),
           "SHARP_EDGE": sum(1 for e in mesh.edges if e.use_edge_sharp),
           "SMOOTH": sum(1 for p in mesh.polygons if p.use_smooth)}
    for key, attr in (("CREASE", "crease_edge"), ("BEVEL_WEIGHT_EDGE", "bevel_weight_edge")):
        at = mesh.attributes.get(attr)
        out[key] = sum(1 for d in at.data if d.value > 0) if at is not None else 0
    return out


def _native_attribute_transfer(src, tgt, name, tail):
    """Copy one generic attribute by the nearest source element in world space (POINT: vertex, FACE: face center,
    EDGE: edge middle, CORNER: the nearest corner of the nearest face). Returns (created, elements, maxDistance)."""
    from mathutils.kdtree import KDTree                         # noqa: PLC0415
    sm, tm = src.data, tgt.data
    sa = sm.attributes.get(name)
    if sa is None:
        raise MifOpError("source '%s' has no attribute '%s'. It has: %s. %s"
                         % (src.name, name, ", ".join(a.name for a in sm.attributes if not a.name.startswith("."))
                            or "(none)", tail))
    smw, tmw = src.matrix_world, tgt.matrix_world

    def points(mesh, mw, domain):
        if domain == "POINT":
            return [mw @ v.co for v in mesh.vertices]
        if domain == "EDGE":
            return [mw @ ((mesh.vertices[e.vertices[0]].co + mesh.vertices[e.vertices[1]].co) * 0.5) for e in mesh.edges]
        if domain == "FACE":
            return [mw @ p.center for p in mesh.polygons]
        return [mw @ mesh.vertices[lp.vertex_index].co for lp in mesh.loops]

    sp = points(sm, smw, sa.domain)
    if not sp:
        raise MifOpError("source attribute '%s' has no elements. %s" % (name, tail))
    tree = KDTree(len(sp))
    for i, p in enumerate(sp):
        tree.insert(p, i)
    tree.balance()
    ta = tm.attributes.get(name)
    created = ta is None
    if ta is not None and (ta.domain != sa.domain or ta.data_type != sa.data_type):
        raise MifOpError("target '%s' has '%s' as %s %s and the source as %s %s - convert one with set_mesh_attribute "
                         "first. %s" % (tgt.name, name, ta.domain, ta.data_type, sa.domain, sa.data_type, tail))
    if created:
        ta = tm.attributes.new(name, sa.data_type, sa.domain)
    field = _ATTR_FIELD.get(sa.data_type, "value")
    vals = [getattr(d, field) for d in sa.data]
    tp = points(tm, tmw, sa.domain)
    worst = 0.0
    for i, p in enumerate(tp):
        _co, j, dist = tree.find(p)
        worst = max(worst, dist)
        v = vals[j]
        setattr(ta.data[i], field, tuple(v) if hasattr(v, "__len__") and not isinstance(v, str) else v)
    return created, len(tp), worst


_ATTR_FIELD = {"FLOAT": "value", "INT": "value", "BOOLEAN": "value", "INT8": "value", "STRING": "value",
               "FLOAT_VECTOR": "vector", "FLOAT2": "vector", "FLOAT_COLOR": "color", "BYTE_COLOR": "color",
               "QUATERNION": "value", "INT32_2D": "value", "INT16_2D": "value", "FLOAT4X4": "value", "FLOAT4": "value"}


def op_transfer_mesh_data(params):
    """Copy UVs, colors, custom normals, edge and face flags, face sets or any named attribute from a source mesh onto
    one or more targets - the destination layers made FIRST, so nothing transfers into a layer that is not there.

    params:
      source (str)          the mesh read
      targets / target      the mesh(es) written
      data (list)           UV | COLOR | CUSTOM_NORMAL | SEAM | SHARP_EDGE | CREASE | BEVEL_WEIGHT_EDGE | SMOOTH |
                            FACE_SETS | ATTRIBUTE (with attributes [...]); default [UV]
      attributes (list)     ATTRIBUTE: the generic attribute names copied (nearest element in world space)
      mapping (str)         NEAREST_FACE (default: interpolated on the nearest face) | POLYINTERP (projected along the
                            normal) | NEAREST (nearest element, no interpolation) | TOPOLOGY (same topology, by index)
      vertexGroup (str)     a vertex group on each target limiting where data lands
      mix (float)           0-1 (default 1)
    Route: a Data Transfer modifier per target, object.datalayout_transfer to create the layers, applied; FACE_SETS and
    ATTRIBUTE natively by nearest element. Weights have their own op (transfer_weights).
    Reply per target: the layers made, per UV / color layer the corners whose value changed, flag counts before and
    after, and the farthest any target vertex is from the source surface (how far the mapping reached).
    """
    reject_unknown(params, _TM_KEYS, "transfer_mesh_data", notes=_TM_NOTES)
    tail = "NOTHING was changed."
    src = get_object(take(params, "source", required=True, kind=str), want_mesh=True)
    raw = params.get("targets", params.get("target"))
    names = [raw] if isinstance(raw, str) else raw
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise MifOpError("targets is a mesh name or a list of them. %s" % tail)
    tgts = [get_object(n, want_mesh=True) for n in dict.fromkeys(names)]
    if src in tgts:
        raise MifOpError("'%s' is the source and a target. %s" % (src.name, tail))
    for t in tgts:
        require_editable(t, "changed")
    data = params.get("data") or ["UV"]
    data = [data] if isinstance(data, str) else data
    data = [str(d).upper() for d in data]
    bad = [d for d in data if d not in _TM_DATA]
    if bad:
        raise MifOpError("data is a list of %s, got %s. %s" % (", ".join(_TM_DATA), bad, tail))
    attrs = params.get("attributes")
    if attrs is not None and (not isinstance(attrs, list) or not all(isinstance(a, str) for a in attrs)):
        raise MifOpError("attributes is a list of attribute names. %s" % tail)
    if ("ATTRIBUTE" in data) != (attrs is not None and len(attrs) > 0):
        raise MifOpError("ATTRIBUTE in data and attributes [names] go together. %s" % tail)
    mapping = str(take(params, "mapping", default="NEAREST_FACE", kind=str)).upper()
    if mapping not in _TM_MAPPING:
        raise MifOpError("mapping must be one of %s, got %r. %s" % (", ".join(_TM_MAPPING), mapping, tail))
    if mapping == "TOPOLOGY":
        wrong = [t.name for t in tgts if (len(t.data.vertices), len(t.data.polygons))
                 != (len(src.data.vertices), len(src.data.polygons))]
        if wrong:
            raise MifOpError("TOPOLOGY maps by index and %s do not have the source's vertex and face counts. %s"
                             % (", ".join(wrong), tail))
    mix = take_float(params, "mix", default=1.0)
    if not 0.0 <= mix <= 1.0:
        raise MifOpError("mix is 0-1, got %g. %s" % (mix, tail))
    vgroup = take(params, "vertexGroup", default=None, kind=str)
    if vgroup is not None:
        for t in tgts:
            _vgroup_weights(t, vgroup, tail)
    sm = src.data
    if "UV" in data and not sm.uv_layers:
        raise MifOpError("source '%s' has no UV layer to transfer. %s" % (src.name, tail))
    if "COLOR" in data and not len(getattr(sm, "color_attributes", [])):
        raise MifOpError("source '%s' has no color attribute to transfer. %s" % (src.name, tail))
    if "FACE_SETS" in data and sm.attributes.get(".sculpt_face_set") is None:
        raise MifOpError("source '%s' has no face sets. %s" % (src.name, tail))
    color_domains = {a.domain for a in getattr(sm, "color_attributes", [])}
    loops_t = [x for x in (("UV", "UV"), ("COLOR", "COLOR_CORNER"), ("CUSTOM_NORMAL", "CUSTOM_NORMAL"))
               if x[0] in data and (x[0] != "COLOR" or "CORNER" in color_domains)]
    verts_t = ["COLOR_VERTEX"] if "COLOR" in data and "POINT" in color_domains else []
    edges_t = [d for d in ("SEAM", "SHARP_EDGE", "CREASE", "BEVEL_WEIGHT_EDGE") if d in data]
    polys_t = ["SMOOTH"] if "SMOOTH" in data else []
    lm, vm, em, pm = _TM_MAPPING[mapping]
    from mathutils.bvhtree import BVHTree                        # noqa: PLC0415
    sw = src.matrix_world
    stree = BVHTree.FromPolygons([sw @ v.co for v in sm.vertices], [tuple(p.vertices) for p in sm.polygons])
    rows = []
    for t in tgts:
        tmesh = t.data
        row = {"object": t.name}
        if "CUSTOM_NORMAL" in data and "use_auto_smooth" in tmesh.bl_rna.properties and not tmesh.use_auto_smooth:
            tmesh.use_auto_smooth = True
            row["autoSmoothSet"] = True
        layers_before = set(_layer_snapshot(tmesh))
        snap = _layer_snapshot(tmesh)
        flags_before = _flag_counts(tmesh)
        if loops_t or verts_t or edges_t or polys_t:
            mod = t.modifiers.new("MifTransfer", "DATA_TRANSFER")
            try:
                mod.object = src
                mod.use_object_transform = True
                mod.mix_factor = mix
                if vgroup:
                    mod.vertex_group = vgroup
                if loops_t:
                    mod.use_loop_data = True
                    mod.data_types_loops = {x[1] for x in loops_t}
                    mod.loop_mapping = lm
                    mod.layers_uv_select_src = "ALL"
                    mod.layers_uv_select_dst = "NAME"
                    mod.layers_vcol_loop_select_src = "ALL"
                    mod.layers_vcol_loop_select_dst = "NAME"
                if verts_t:
                    mod.use_vert_data = True
                    mod.data_types_verts = set(verts_t)
                    mod.vert_mapping = vm
                    mod.layers_vcol_vert_select_src = "ALL"
                    mod.layers_vcol_vert_select_dst = "NAME"
                if edges_t:
                    mod.use_edge_data = True
                    mod.data_types_edges = set(edges_t)
                    mod.edge_mapping = em
                if polys_t:
                    mod.use_poly_data = True
                    mod.data_types_polys = set(polys_t)
                    mod.poly_mapping = pm
                with bpy.context.temp_override(object=t, active_object=t, selected_objects=[t],
                                               selected_editable_objects=[t]):
                    # THE STEP THE HAND ROUTE MISSES: the modifier writes only into layers that exist on the target
                    bpy.ops.object.datalayout_transfer(modifier=mod.name)
                    while t.modifiers[0] != mod:
                        bpy.ops.object.modifier_move_up(modifier=mod.name)
                    bpy.ops.object.modifier_apply(modifier=mod.name)
            except Exception as exc:
                if mod.name in t.modifiers:
                    t.modifiers.remove(mod)
                raise MifOpError("the transfer onto '%s' failed: %s. %s" % (t.name, str(exc).strip(), tail))
        created_attrs = []
        if "FACE_SETS" in data:
            made, n, _w = _native_attribute_transfer(src, t, ".sculpt_face_set", tail)
            row["faceSets"] = {"created": made, "faces": n}
        for a in attrs or []:
            made, n, w = _native_attribute_transfer(src, t, a, tail)
            created_attrs.append({"attribute": a, "created": made, "elements": n, "maxDistance": round(w, 6)})
        after = _layer_snapshot(tmesh)
        row["layersCreated"] = sorted(set(after) - layers_before)
        changed = {}
        for key, arr in after.items():
            old = snap.get(key)
            width = 2 if key.startswith("UV:") else 4
            if old is None or len(old) != len(arr):
                changed[key] = len(arr) // width
            else:
                import numpy as np                              # noqa: PLC0415
                changed[key] = int(np.any(np.abs(old.reshape(-1, width) - arr.reshape(-1, width)) > 1e-6, axis=1).sum())
        row["changed"] = changed
        if edges_t or polys_t:
            fa = _flag_counts(tmesh)
            row["flags"] = {k: {"before": flags_before[k], "after": fa[k]} for k in edges_t + polys_t}
        if attrs:
            row["attributes"] = created_attrs
        tw = t.matrix_world
        far = 0.0
        for v in tmesh.vertices:
            hit = stree.find_nearest(tw @ v.co)
            if hit[0] is not None:
                far = max(far, hit[3])
        row["maxDistanceToSource"] = round(far, 6)
        rows.append(row)
    idle = [r["object"] for r in rows if not any(r.get("changed", {}).values()) and not r.get("flags")
            and not r.get("attributes") and not r.get("faceSets") and "CUSTOM_NORMAL" not in data]
    out = {"source": src.name, "data": data, "mapping": mapping, "mix": mix, "targets": rows}
    if idle:
        out["warning"] = "nothing changed on %s - the data was already equal, or the mapping reached nothing" % idle
    return out


# --------------------------------------------------------------------------------------------- transfer_shape_keys

_TS_KEYS = ("source", "target", "keys", "falloff", "strength", "replace")


def op_transfer_shape_keys(params):
    """Carry shape keys from a source mesh onto a target of DIFFERENT topology through a Surface Deform binding, one
    key at a time - a facial set or a fitting morph made on a template moved onto a custom head or a garment.

    params:
      source (str)       the mesh whose shape keys are read
      target (str)       the mesh that receives them (it gets a Basis if it has none)
      keys (list)        which keys (default every key but the Basis)
      falloff (float)    Surface Deform's interpolation falloff (default 4; 2-16)
      strength (float)   1 by default
      replace (bool)     replace a target key of the same name (default false: refused)
    Per key: slider range, relative key and the source key's max displacement are carried; the target key's max
    displacement is measured and a key that lands under 1% of the source's is flagged.
    """
    reject_unknown(params, _TS_KEYS, "transfer_shape_keys")
    tail = "NOTHING was changed."
    src = get_object(take(params, "source", required=True, kind=str), want_mesh=True)
    tgt = get_object(take(params, "target", required=True, kind=str), want_mesh=True)
    if src is tgt:
        raise MifOpError("source and target are one object. %s" % tail)
    require_editable(tgt, "changed")
    sk = src.data.shape_keys
    if sk is None or len(sk.key_blocks) < 2:
        raise MifOpError("source '%s' has no shape keys besides its Basis. %s" % (src.name, tail))
    basis = sk.reference_key
    want = params.get("keys")
    blocks = [kb for kb in sk.key_blocks if kb != basis]
    if want is not None:
        if not isinstance(want, list) or not all(isinstance(k, str) for k in want):
            raise MifOpError("keys is a list of shape key names. %s" % tail)
        absent = [k for k in want if sk.key_blocks.get(k) is None or sk.key_blocks.get(k) == basis]
        if absent:
            raise MifOpError("source has no shape key(s) %s (the Basis is not transferred). %s" % (absent, tail))
        blocks = [sk.key_blocks[k] for k in dict.fromkeys(want)]
    replace = take_bool(params, "replace", default=False)
    tk = tgt.data.shape_keys
    if tk is not None and not replace:
        clash = [kb.name for kb in blocks if tk.key_blocks.get(kb.name) is not None]
        if clash:
            raise MifOpError("target already has shape key(s) %s - replace:true overwrites them. %s" % (clash, tail))
    falloff = take_float(params, "falloff", default=4.0)
    strength = take_float(params, "strength", default=1.0)
    if not 2.0 <= falloff <= 16.0:
        raise MifOpError("falloff is 2-16, got %g. %s" % (falloff, tail))
    if tgt.data.shape_keys is None:
        tgt.shape_key_add(name="Basis", from_mix=False)
    # the source shows its Basis while binding: every key value 0, nothing pinned
    saved = [(kb, kb.value, kb.mute) for kb in sk.key_blocks]
    pin = src.show_only_shape_key
    src.show_only_shape_key = False
    for kb in sk.key_blocks:
        kb.value = 0.0
    mod = tgt.modifiers.new("MifShapeTransfer", "SURFACE_DEFORM")
    rows = []
    try:
        mod.target = src
        mod.falloff = falloff
        mod.strength = strength
        with bpy.context.temp_override(object=tgt, active_object=tgt):
            bpy.context.view_layer.update()
            bpy.ops.object.surfacedeform_bind(modifier=mod.name)
            if not mod.is_bound:
                raise MifOpError("Surface Deform could not bind '%s' to '%s' (the source needs faces near every target "
                                 "vertex, and no concave polygons). %s" % (tgt.name, src.name, tail))
            import numpy as np                                  # noqa: PLC0415
            nb = len(src.data.vertices)
            bco = np.empty(nb * 3, dtype=np.float32)
            basis.data.foreach_get("co", bco)
            tb = tgt.data.shape_keys.reference_key
            nt = len(tgt.data.vertices)
            tbco = np.empty(nt * 3, dtype=np.float32)
            tb.data.foreach_get("co", tbco)
            for kb in blocks:
                for other in sk.key_blocks:
                    other.value = 0.0
                kb.value = 1.0
                kco = np.empty(nb * 3, dtype=np.float32)
                kb.data.foreach_get("co", kco)
                rel = kb.relative_key
                rco = bco
                if rel is not None and rel != basis:
                    rco = np.empty(nb * 3, dtype=np.float32)
                    rel.data.foreach_get("co", rco)
                sdisp = float(np.max(np.linalg.norm((kco - rco).reshape(-1, 3), axis=1)))
                bpy.context.view_layer.update()
                old = tgt.data.shape_keys.key_blocks.get(kb.name)
                if old is not None:
                    tgt.shape_key_remove(old)
                bpy.ops.object.modifier_apply_as_shapekey(keep_modifier=True, modifier=mod.name)
                new = tgt.data.shape_keys.key_blocks[-1]
                new.name = kb.name
                new.slider_min, new.slider_max = kb.slider_min, kb.slider_max
                new.value = 0.0
                nco = np.empty(nt * 3, dtype=np.float32)
                new.data.foreach_get("co", nco)
                tdisp = float(np.max(np.linalg.norm((nco - tbco).reshape(-1, 3), axis=1)))
                row = {"key": kb.name, "sourceMaxDisplacement": round(sdisp, 6),
                       "targetMaxDisplacement": round(tdisp, 6),
                       "relativeKey": rel.name if rel is not None else None,
                       "slider": [kb.slider_min, kb.slider_max]}
                if sdisp > 1e-6 and tdisp < 0.01 * sdisp:
                    row["warning"] = "landed near zero on the target - the moved region is far from the target surface"
                rows.append(row)
            # relative keys that were transferred too are re-pointed by name
            for kb in blocks:
                if kb.relative_key is not None and kb.relative_key != basis:
                    tr = tgt.data.shape_keys.key_blocks.get(kb.relative_key.name)
                    if tr is not None:
                        tgt.data.shape_keys.key_blocks[kb.name].relative_key = tr
    finally:
        if mod.name in tgt.modifiers:
            tgt.modifiers.remove(mod)
        for kb, val, mute in saved:
            kb.value, kb.mute = val, mute
        src.show_only_shape_key = pin
    flagged = [r["key"] for r in rows if r.get("warning")]
    out = {"source": src.name, "target": tgt.name, "transferred": len(rows), "keys": rows,
           "sourceVertices": len(src.data.vertices), "targetVertices": len(tgt.data.vertices)}
    if flagged:
        out["warning"] = "%d key(s) landed near zero: %s" % (len(flagged), flagged)
    return out


# --------------------------------------------------------------------------------------------- set_mesh_attribute

_MA_KEYS = ("object", "action", "name", "newName", "domain", "dataType", "value", "values", "indices",
            "fromVertexGroup", "sample")
_MA_ACTIONS = ("CREATE", "WRITE", "RENAME", "CONVERT", "READ")
_DOMAINS = ("POINT", "EDGE", "FACE", "CORNER")
_WIDTH = {"FLOAT": 1, "INT": 1, "BOOLEAN": 1, "INT8": 1, "FLOAT_VECTOR": 3, "FLOAT2": 2, "FLOAT_COLOR": 4,
          "BYTE_COLOR": 4, "QUATERNION": 4, "INT32_2D": 2, "INT16_2D": 2}


def _attr_types():
    return [i.identifier for i in bpy.types.Attribute.bl_rna.properties["data_type"].enum_items]


def _domain_len(mesh, domain):
    return {"POINT": len(mesh.vertices), "EDGE": len(mesh.edges), "FACE": len(mesh.polygons),
            "CORNER": len(mesh.loops)}[domain]


def _attr_read(at, sample):
    field = _ATTR_FIELD.get(at.data_type, "value")
    n = len(at.data)
    rows = []
    for i in range(min(n, sample)):
        v = getattr(at.data[i], field)
        rows.append(round(v, 6) if isinstance(v, float) else (list(round(float(x), 6) for x in v)
                                                               if hasattr(v, "__len__") and not isinstance(v, str) else v))
    out = {"name": at.name, "domain": at.domain, "dataType": at.data_type, "elements": n, "sample": rows}
    if at.data_type in ("FLOAT", "INT", "INT8") and n:
        held = [getattr(d, field) for d in at.data]
        out["min"], out["max"] = round(float(min(held)), 6), round(float(max(held)), 6)
    return out


def _coerce(dtype, v, where, tail):
    w = _WIDTH.get(dtype)
    if w is None:
        raise MifOpError("writing %s values is not supported here (STRING, FLOAT4X4). %s" % (dtype, tail))
    if dtype == "BOOLEAN":
        if not isinstance(v, bool):
            raise MifOpError("%s: a BOOLEAN value is true or false, got %r. %s" % (where, v, tail))
        return v
    if w == 1:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise MifOpError("%s: a %s value is a number, got %r. %s" % (where, dtype, v, tail))
        if dtype in ("INT", "INT8"):
            if float(v) != int(v):
                raise MifOpError("%s: %s takes whole numbers, got %r. %s" % (where, dtype, v, tail))
            return int(v)
        return finite_floats([v], where)[0]
    if not isinstance(v, (list, tuple)) or len(v) != w:
        raise MifOpError("%s: a %s value is a list of %d numbers, got %r. %s" % (where, dtype, w, v, tail))
    if dtype.startswith("INT"):
        return tuple(finite_int(x, where) for x in v)
    return tuple(finite_floats(list(v), where))


def op_set_mesh_attribute(params):
    """Create, write, rename, convert or read a GENERIC mesh attribute - the named data geometry nodes read (a scatter
    density, a wind mask, a per-piece id) and that FLOAT2 attributes carry as extra UV channels on export.

    params:
      object (str)            a mesh in object mode
      action (str)            CREATE | WRITE | RENAME | CONVERT | READ
      name (str)              the attribute
      domain (str)            POINT | EDGE | FACE | CORNER (CREATE; CONVERT's new domain)
      dataType (str)          FLOAT | INT | BOOLEAN | INT8 | FLOAT_VECTOR | FLOAT2 | FLOAT_COLOR | BYTE_COLOR |
                              QUATERNION (4.2+; refused by name below) | ... the build's list (CREATE; CONVERT's type)
      value                   CREATE / WRITE: one value for every element
      values (list)           CREATE / WRITE: one value per element, or per index with indices
      indices (list)          WRITE: the elements written
      fromVertexGroup (str)   CREATE / WRITE on a FLOAT POINT attribute: the group's weights (0 where unweighted)
      newName (str)           RENAME
      sample (int)            elements read back (default 8)
    """
    reject_unknown(params, _MA_KEYS, "set_mesh_attribute")
    tail = "NOTHING was changed."
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    mesh = obj.data
    action = str(take(params, "action", required=True, kind=str)).upper()
    if action not in _MA_ACTIONS:
        raise MifOpError("action must be one of %s, got %r. %s" % (", ".join(_MA_ACTIONS), params.get("action"), tail))
    name = take(params, "name", required=True, kind=str)
    sample = take_int(params, "sample", default=8)
    at = mesh.attributes.get(name)
    if action != "CREATE" and at is None:
        raise MifOpError("'%s' has no attribute '%s'. It has: %s. %s"
                         % (obj.name, name, ", ".join(a.name for a in mesh.attributes if not a.name.startswith("."))
                            or "(none)", tail))
    if action == "READ":
        return {"object": obj.name, "action": action, "attribute": _attr_read(at, sample)}
    if action != "READ":
        require_editable(obj, "changed")
    if action == "RENAME":
        new = take(params, "newName", required=True, kind=str)
        if mesh.attributes.get(new) is not None:
            raise MifOpError("'%s' already has an attribute '%s'. %s" % (obj.name, new, tail))
        if name in ("position", ".select_vert") or name.startswith("."):
            raise MifOpError("'%s' is Blender's own attribute and is not renamed. %s" % (name, tail))
        at.name = new
        return {"object": obj.name, "action": action, "attribute": _attr_read(mesh.attributes[new], sample),
                "oldName": name}
    types = _attr_types()
    if action == "CONVERT":
        domain = str(take(params, "domain", default=at.domain, kind=str)).upper()
        dtype = str(take(params, "dataType", default=at.data_type, kind=str)).upper()
        if domain not in _DOMAINS or dtype not in types:
            raise MifOpError("domain is one of %s and dataType one of %s on Blender %s. %s"
                             % (", ".join(_DOMAINS), ", ".join(types), bpy.app.version_string, tail))
        if (domain, dtype) == (at.domain, at.data_type):
            raise MifOpError("'%s' is already %s %s. %s" % (name, domain, dtype, tail))
        was = [at.domain, at.data_type]
        mesh.attributes.active = at
        try:
            with bpy.context.temp_override(object=obj, active_object=obj):
                bpy.ops.geometry.attribute_convert(mode="GENERIC", domain=domain, data_type=dtype)
        except RuntimeError as exc:
            raise MifOpError("Blender refused the conversion: %s. %s" % (str(exc).strip(), tail))
        nat = mesh.attributes.get(name)
        return {"object": obj.name, "action": action, "from": was, "attribute": _attr_read(nat, sample)}
    if action == "CREATE":
        if at is not None:
            raise MifOpError("'%s' already has an attribute '%s' (%s %s) - WRITE writes it. %s"
                             % (obj.name, name, at.domain, at.data_type, tail))
        domain = str(take(params, "domain", default="POINT", kind=str)).upper()
        dtype = str(take(params, "dataType", default="FLOAT", kind=str)).upper()
        if domain not in _DOMAINS:
            raise MifOpError("domain must be one of %s, got %r. %s" % (", ".join(_DOMAINS), domain, tail))
        if dtype not in types:
            extra = " (QUATERNION arrived in 4.2's list)" if dtype == "QUATERNION" else ""
            raise MifOpError("dataType %r is not one of Blender %s's: %s%s. %s"
                             % (dtype, bpy.app.version_string, ", ".join(types), extra, tail))
        if name.startswith(".") or name == "position":
            raise MifOpError("'%s' is a reserved name. %s" % (name, tail))
    else:
        domain, dtype = at.domain, at.data_type
    n = _domain_len(mesh, domain)
    plan = None
    vg = take(params, "fromVertexGroup", default=None, kind=str)
    given = [k for k in ("value", "values", "fromVertexGroup") if params.get(k) is not None]
    if len(given) > 1:
        raise MifOpError("give one of value, values or fromVertexGroup. %s" % tail)
    if action == "WRITE" and not given:
        raise MifOpError("WRITE needs value, values or fromVertexGroup. %s" % tail)
    idx = params.get("indices")
    if idx is not None and (params.get("values") is None):
        raise MifOpError("indices go with values (one value per index). %s" % tail)
    if vg is not None:
        if (domain, dtype) != ("POINT", "FLOAT"):
            raise MifOpError("fromVertexGroup writes a FLOAT POINT attribute; this one is %s %s. %s"
                             % (domain, dtype, tail))
        plan = [(i, w) for i, w in enumerate(_vgroup_weights(obj, vg, tail))]
    elif params.get("value") is not None:
        v = _coerce(dtype, params["value"], "value", tail)
        plan = [(i, v) for i in range(n)]
    elif params.get("values") is not None:
        vals = params["values"]
        if not isinstance(vals, list):
            raise MifOpError("values is a list. %s" % tail)
        if idx is None:
            if len(vals) != n:
                raise MifOpError("values has %d entries and the %s domain %d elements - give one per element, or "
                                 "indices. %s" % (len(vals), domain, n, tail))
            idx = list(range(n))
        if not isinstance(idx, list) or len(idx) != len(vals):
            raise MifOpError("indices and values must be lists of one length. %s" % tail)
        bad = [i for i in idx if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n]
        if bad:
            raise MifOpError("indices %s are outside the %s domain's 0-%d. %s" % (bad[:10], domain, n - 1, tail))
        plan = [(i, _coerce(dtype, v, "values[%d]" % k, tail)) for k, (i, v) in enumerate(zip(idx, vals))]
    if action == "CREATE":
        at = mesh.attributes.new(name, dtype, domain)
    field = _ATTR_FIELD.get(dtype, "value")
    for i, v in plan or []:
        setattr(at.data[i], field, v)
    mesh.update()
    at = mesh.attributes[name]
    return {"object": obj.name, "action": action, "written": len(plan or []), "attribute": _attr_read(at, sample)}


# --------------------------------------------------------------------------------------------- add_hook

_HOOK_KEYS = ("object", "vertices", "vertexGroup", "hookObject", "armature", "bone", "name", "strength", "falloff",
              "falloffRadius", "emptyName", "emptySize")


def _hook_points(obj, idx):
    """World positions of the chosen points, and the HOOK modifier's index list for them."""
    mw = obj.matrix_world
    if obj.type == "MESH":
        n = len(obj.data.vertices)
        return [mw @ obj.data.vertices[i].co for i in idx], list(idx), n
    if obj.type == "LATTICE":
        n = len(obj.data.points)
        return [mw @ obj.data.points[i].co for i in idx], list(idx), n
    # CURVE: Blender counts a Bezier point as three (left handle, point, right handle), a poly / NURBS point as one
    flat, pos, k = [], {}, 0
    for sp in obj.data.splines:
        if sp.type == "BEZIER":
            for bp in sp.bezier_points:
                pos[len(pos)] = ([k, k + 1, k + 2], bp.co)
                k += 3
        else:
            for p in sp.points:
                pos[len(pos)] = ([k], p.co.xyz)
                k += 1
    pts = []
    for i in idx:
        ids, co = pos[i]
        flat += ids
        pts.append(mw @ co)
    return pts, flat, len(pos)


def op_add_hook(params):
    """Hook vertices (or curve / lattice points) to an object or a bone, with the inverse matrix RESET so nothing moves
    when the hook is added - the vertices' world positions before and after are read back and must match.

    params:
      object (str)          a mesh, curve or lattice in object mode
      vertices (list)       point indices (a curve's points counted per spline in order; a Bezier point brings its
                            handles), or
      vertexGroup (str)     a mesh's vertex group (weights over 0)
      hookObject (str)      the object the points follow; default an Empty made at their centroid (emptyName,
                            emptySize)
      armature, bone (str)  or a bone of an armature object
      name (str)            the modifier's name (default Hook-<target>)
      strength (float)      0-1 (default 1); falloff NONE | CURVE | SMOOTH | SPHERE | ROOT | INVERSE_SQUARE | SHARP |
                            LINEAR | CONSTANT with falloffRadius
    The modifier is placed before every non-hook modifier, as Blender's own Hook to Selected Object does.
    """
    reject_unknown(params, _HOOK_KEYS, "add_hook", notes={"inverse": "the inverse matrix is always reset - that is the "
                                                                     "point of this op"})
    tail = "NOTHING was changed."
    obj = get_object(take(params, "object", required=True, kind=str))
    if obj.type not in ("MESH", "CURVE", "LATTICE"):
        raise MifOpError("'%s' is a %s; hooks move a MESH's, CURVE's or LATTICE's points. %s" % (obj.name, obj.type, tail))
    if obj.mode != "OBJECT":
        raise MifOpError("'%s' is in %s mode - leave it first. %s" % (obj.name, obj.mode, tail))
    require_editable(obj, "changed")
    raw = params.get("vertices")
    vgname = take(params, "vertexGroup", default=None, kind=str)
    if (raw is None) == (vgname is None):
        raise MifOpError("give vertices (indices) or vertexGroup. %s" % tail)
    if vgname is not None:
        if obj.type != "MESH":
            raise MifOpError("vertexGroup selects a mesh's vertices. %s" % tail)
        w = _vgroup_weights(obj, vgname, tail)
        idx = [i for i, x in enumerate(w) if x > 0]
    else:
        if not isinstance(raw, list) or not raw or any(isinstance(i, bool) or not isinstance(i, int) for i in raw):
            raise MifOpError("vertices is a non-empty list of point indices. %s" % tail)
        idx = list(dict.fromkeys(raw))
    if not idx:
        raise MifOpError("no point selected (the vertex group has no weight above 0). %s" % tail)
    try:
        pts, flat, n = _hook_points(obj, idx)
    except (IndexError, KeyError):
        raise MifOpError("an index in vertices is outside '%s''s points. %s" % (obj.name, tail))
    hook_name = take(params, "hookObject", default=None, kind=str)
    arm_name = take(params, "armature", default=None, kind=str)
    bone = take(params, "bone", default=None, kind=str)
    if hook_name is not None and (arm_name or bone):
        raise MifOpError("hook to hookObject or to armature + bone, not both. %s" % tail)
    if bool(arm_name) != bool(bone):
        raise MifOpError("armature and bone go together. %s" % tail)
    falloff = str(take(params, "falloff", default="NONE", kind=str)).upper()
    fall_items = [i.identifier for i in bpy.types.HookModifier.bl_rna.properties["falloff_type"].enum_items]
    if falloff not in fall_items:
        raise MifOpError("falloff must be one of %s, got %r. %s" % (", ".join(fall_items), falloff, tail))
    strength = take_float(params, "strength", default=1.0)
    if not 0.0 <= strength <= 1.0:
        raise MifOpError("strength is 0-1, got %g. %s" % (strength, tail))
    # EVERY VALUE THAT CAN REFUSE IS READ HERE, before the Empty is made: emptySize, the modifier's name and
    # falloffRadius were read after objects.new, so a malformed one left a stray Empty behind "NOTHING was changed".
    empty_req = take(params, "emptyName", default="Hook_%s" % obj.name, kind=str)
    empty_size = take_float(params, "emptySize", default=0.1)
    mod_name = take(params, "name", default=None, kind=str)
    radius = take_float(params, "falloffRadius") if params.get("falloffRadius") is not None else None
    from mathutils import Vector                                # noqa: PLC0415
    centroid = sum(pts, Vector()) / len(pts)
    made_empty = None
    if arm_name:
        arm = get_object(arm_name)
        if arm.type != "ARMATURE" or arm.pose.bones.get(bone) is None:
            raise MifOpError("'%s' is not an armature with a bone '%s'. %s" % (arm_name, bone, tail))
        hook = arm
        hook_mw = arm.matrix_world @ arm.pose.bones[bone].matrix
    elif hook_name is not None:
        hook = get_object(hook_name)
        if hook is obj:
            raise MifOpError("an object cannot hook to itself. %s" % tail)
        hook_mw = hook.matrix_world
    else:
        made_empty = hook = bpy.data.objects.new(empty_req, None)
        hook.empty_display_size = empty_size
        hook.location = centroid
        for c in obj.users_collection:
            c.objects.link(hook)
        bpy.context.view_layer.update()
        hook_mw = hook.matrix_world
    deps = bpy.context.evaluated_depsgraph_get()

    def eval_points():
        ev = obj.evaluated_get(deps)
        if obj.type == "MESH":
            me = ev.to_mesh()
            try:
                return [ev.matrix_world @ me.vertices[i].co for i in idx]
            finally:
                ev.to_mesh_clear()
        return None

    before = eval_points()
    mod = obj.modifiers.new(mod_name if mod_name is not None else "Hook-%s" % (bone or hook.name), "HOOK")
    mod.object = hook
    if bone:
        mod.subtarget = bone
    mod.strength = strength
    mod.falloff_type = falloff
    if radius is not None:
        mod.falloff_radius = radius
    mod.vertex_indices_set(flat)
    # THE RESET: the hook's transform at bind time, relative to the object, is undone by matrix_inverse (Blender's
    # hook_reset: parentinv = inverse(obmat^-1 * hookmat)); center is the points' centroid in object space (falloff)
    mod.matrix_inverse = (obj.matrix_world.inverted_safe() @ hook_mw).inverted_safe()
    mod.center = obj.matrix_world.inverted_safe() @ centroid
    hooks_first = sum(1 for m in obj.modifiers if m.type == "HOOK" and m != mod)
    if "move" in obj.modifiers.bl_rna.functions:
        obj.modifiers.move(len(obj.modifiers) - 1, hooks_first)
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    after = eval_points()
    err = None
    if before is not None and after is not None:
        err = max((a - b).length for a, b in zip(before, after))
    out = {"object": obj.name, "modifier": mod.name, "hook": hook.name, "bone": bone, "points": len(idx),
           "indicesWritten": len(flat), "centroid": [round(c, 6) for c in centroid],
           "emptyMade": made_empty.name if made_empty is not None else None,
           # the Empty's name as asked and whether Blender suffixed it (a retry after a timeout makes Hook_X.001)
           "requestedName": empty_req if made_empty is not None else None,
           "nameWasSuffixed": made_empty is not None and made_empty.name != empty_req,
           "stackIndex": list(obj.modifiers).index(mod), "positionsMovedMax": None if err is None else round(err, 7)}
    if err is not None and err > 1e-4:
        out["warning"] = ("the hooked points moved %.6f when the hook was added - a modifier below the hook changes "
                          "them, or the hook target is animated" % err)
    return out


# --------------------------------------------------------------------------------------------- spin_mesh

_SPIN_KEYS = ("object", "vertices", "edges", "faces", "axis", "center", "angle", "steps", "dupli", "mergeSeam",
              "flipNormals", "space")


def op_spin_mesh(params):
    """Spin chosen edges or faces of an existing mesh about an axis - a lip on a bowl, a flange on a pipe end, a radial
    array of bolt heads (dupli) - without making a new object (build_lathe revolves a profile into one).

    params:
      object (str)        a mesh in object mode
      edges / faces / vertices (list)   the indices spun (one kind)
      axis ([x,y,z])      the spin axis (default [0, 0, 1]); center ([x,y,z]) a point on it (default the origin)
      space (str)         WORLD (default) | LOCAL - the frame axis and center are given in
      angle (float)       degrees (default 360)
      steps (int)         segments (default 12)
      dupli (bool)        copies at each step instead of a swept surface
      mergeSeam (bool)    at 360 degrees, weld the last ring to the first (default true)
      flipNormals (bool)  flip the new faces (default: they are made consistent with the existing ones)
    Reply: new vertices, edges and faces, and whether the result is closed (no boundary edge among the new faces).
    """
    reject_unknown(params, _SPIN_KEYS, "spin_mesh")
    tail = "NOTHING was changed."
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    require_editable(obj, "changed")
    given = [k for k in ("vertices", "edges", "faces") if params.get(k) is not None]
    if len(given) != 1:
        raise MifOpError("give one of vertices, edges or faces (indices). %s" % tail)
    kind = given[0]
    idx = params[kind]
    if not isinstance(idx, list) or not idx or any(isinstance(i, bool) or not isinstance(i, int) for i in idx):
        raise MifOpError("%s is a non-empty list of indices. %s" % (kind, tail))
    mesh = obj.data
    pool = {"vertices": mesh.vertices, "edges": mesh.edges, "faces": mesh.polygons}[kind]
    bad = [i for i in idx if not 0 <= i < len(pool)]
    if bad:
        raise MifOpError("%s %s are outside 0-%d. %s" % (kind, bad[:10], len(pool) - 1, tail))
    angle = take_float(params, "angle", default=360.0)
    steps = take_int(params, "steps", default=12)
    if not 1 <= steps <= 1024:
        raise MifOpError("steps is 1-1024, got %d. %s" % (steps, tail))
    if angle == 0 or abs(angle) > 360.0 * 16:
        raise MifOpError("angle is a non-zero number of degrees, got %g. %s" % (angle, tail))
    space = str(take(params, "space", default="WORLD", kind=str)).upper()
    if space not in ("WORLD", "LOCAL"):
        raise MifOpError("space is WORLD or LOCAL, got %r. %s" % (space, tail))
    from mathutils import Matrix, Vector                        # noqa: PLC0415
    import bmesh                                                # noqa: PLC0415
    axis = Vector(_vec3(params, "axis", tail, default=[0.0, 0.0, 1.0]))
    center = Vector(_vec3(params, "center", tail, default=[0.0, 0.0, 0.0]))
    if axis.length < 1e-9:
        raise MifOpError("axis has zero length. %s" % tail)
    # READ BEFORE THE SPIN: mergeSeam was read only at 360 degrees, so one malformed value passed at 90
    merge_seam = take_bool(params, "mergeSeam", default=True)
    flip = take_bool(params, "flipNormals", default=False)
    dupli = take_bool(params, "dupli", default=False)
    if space == "WORLD":
        inv = obj.matrix_world.inverted_safe()
        center = inv @ center
        axis = (inv.to_3x3() @ axis)
    axis.normalize()
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        if kind == "faces":
            faces = [bm.faces[i] for i in idx]
            geom = list({v for f in faces for v in f.verts}) + list({e for f in faces for e in f.edges}) + faces
        elif kind == "edges":
            edges = [bm.edges[i] for i in idx]
            geom = list({v for e in edges for v in e.verts}) + edges
        else:
            geom = [bm.verts[i] for i in idx]
        n0 = (len(bm.verts), len(bm.edges), len(bm.faces))
        full = abs(abs(angle) - 360.0) < 1e-6
        res = bmesh.ops.spin(bm, geom=geom, cent=center, axis=axis, dvec=Vector((0, 0, 0)),
                             angle=math.radians(angle), space=Matrix.Identity(4), steps=steps,
                             use_merge=full and merge_seam, use_normal_flip=flip, use_duplicate=dupli)
        new_faces = [g for g in res.get("geom_last", []) if isinstance(g, bmesh.types.BMFace)]
        bm.faces.ensure_lookup_table()
        made_faces = list(bm.faces)[n0[2]:]          # bmesh appends what an op makes
        boundary = sum(1 for f in made_faces for e in f.edges if len(e.link_faces) < 2)
        n1 = (len(bm.verts), len(bm.edges), len(bm.faces))
        profile = len({v for g in geom for v in (g.verts if hasattr(g, "verts") else [g])})
        merged = full and merge_seam
        bm.to_mesh(mesh)
    finally:
        bm.free()
    mesh.update()
    return {"object": obj.name, "spun": kind, "count": len(idx), "angle": angle, "steps": steps,
            "axisLocal": [round(a, 6) for a in axis], "centerLocal": [round(c, 6) for c in center], "space": space,
            "newVertices": n1[0] - n0[0], "newEdges": n1[1] - n0[1], "newFaces": n1[2] - n0[2],
            "lastRingFaces": len(new_faces), "boundaryEdgesOnNewFaces": boundary,
            # the last ring welded onto the first: one ring fewer of new vertices than steps
            "seamWelded": (merged and not dupli and n1[0] - n0[0] == profile * (steps - 1)) if full else None,
            "vertices": n1[0], "faces": n1[2]}


OPS = {
    "set_custom_normals": op_set_custom_normals,
    "transfer_mesh_data": op_transfer_mesh_data,
    "transfer_shape_keys": op_transfer_shape_keys,
    "set_mesh_attribute": op_set_mesh_attribute,
    "add_hook": op_add_hook,
    "spin_mesh": op_spin_mesh,
}
