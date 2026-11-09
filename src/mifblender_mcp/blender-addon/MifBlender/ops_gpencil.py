"""Grease Pencil: objects, layers, strokes on frames, and stroke / fill materials - across its API generations.

THE GAP (spec "Grease Pencil objects, strokes, layers, frames, materials, modifiers and effects across the three API
generations", approved 2026-10-04): nothing in the add-on wrote Grease Pencil, so a 2D shot on a camera
canvas, outlines and drawn strokes on a model, and strokes laid on a surface were raw Python. An empty Grease Pencil
object was declined earlier for being a container with no way to fill it; draw_grease_pencil_strokes is that way.

THE THREE GENERATIONS, read off the running builds 2026-10-08 (3.6.23, 4.4.0, 5.2.1):
  3.6, 4.2   bpy.data.grease_pencils -> GreasePencil, object type GPENCIL. layer.frames.new(n) is a frame of strokes;
             frame.strokes.new(), stroke.points.add(n), point.co / pressure / strength, stroke.line_width (pixels),
             material_index, use_cyclic.
  4.3 - 4.5  bpy.data.grease_pencils_v3 -> GreasePencilv3, object type GREASEPENCIL. layer.frames.new(n).drawing;
             drawing.add_strokes([sizes]); stroke.points[i].position / radius (metres) / opacity; stroke.cyclic,
             material_index. Strokes and points are Python wrappers with no bl_rna.
  5.0+       bpy.data.grease_pencils -> GreasePencil, the v3 data under the old collection name.
Materials on every build: bpy.data.materials.create_gpencil_data(mat), then mat.grease_pencil (show_stroke, show_fill,
color, fill_color, mode, stroke_style, fill_style, ...; 5.2 adds placement_* and random_*).

Every reply names the generation it wrote ("GPENCIL" or "GREASEPENCIL"), because a value means something different in
each: a legacy stroke's width is pixels, a v3 point's radius is metres.
"""

import bpy
from mathutils import Vector

from .ops_common import (remove_objects, MifOpError, assign_id_name, finite_floats, get_object, reject_unknown, rna_properties, rnd, take,
                         take_bool, take_int, write_rna_property)

_ADDED = "NOTHING was added."
_CHANGED = "NOTHING was changed."
_MAX_POINTS = 100000


def _v3_collection():
    """(bpy.data collection for new Grease Pencil data, 'GPENCIL' | 'GREASEPENCIL') on this build."""
    v3 = getattr(bpy.data, "grease_pencils_v3", None)
    if v3 is not None:
        return v3, "GREASEPENCIL"
    return bpy.data.grease_pencils, ("GREASEPENCIL" if bpy.app.version >= (5, 0, 0) else "GPENCIL")


def _gp_object(name, tail):
    obj = get_object(name)
    if obj.type not in ("GPENCIL", "GREASEPENCIL"):
        raise MifOpError("'%s' is a %s, not a Grease Pencil object (create_grease_pencil makes one). %s"
                         % (obj.name, obj.type, tail))
    return obj


def _is_v3(obj):
    return obj.type == "GREASEPENCIL"


def _gp_material(name, create, tail, color=None):
    mat = bpy.data.materials.get(name)
    if mat is None:
        if not create:
            raise MifOpError("no material named '%s'. %s" % (name, tail))
        mat = bpy.data.materials.new(name)
        assign_id_name(mat, name, bpy.data.materials)
    if mat.grease_pencil is None:
        bpy.data.materials.create_gpencil_data(mat)
    if color is not None:
        mat.grease_pencil.color = color
    return mat


def _slot_of(obj, mat):
    """The object's material index for mat, adding it to the object's materials when it is not there."""
    mats = list(obj.data.materials)
    if mat.name not in [m.name for m in mats if m is not None]:
        obj.data.materials.append(mat)
        mats = list(obj.data.materials)
    return [m.name if m is not None else None for m in mats].index(mat.name)


def _frame_row(obj, fr):
    if _is_v3(obj):
        d = fr.drawing
        strokes = len(d.strokes)
        points = sum(len(s.points) for s in d.strokes)
    else:
        strokes = len(fr.strokes)
        points = sum(len(s.points) for s in fr.strokes)
    return {"frame": int(fr.frame_number), "strokes": strokes, "points": points, "keyframeType": fr.keyframe_type}


def _layer_row(obj, layer):
    row = {"name": layer.name if _is_v3(obj) else layer.info, "opacity": round(float(layer.opacity), 6),
           "hide": bool(layer.hide), "lock": bool(layer.lock), "blendMode": layer.blend_mode,
           "useLights": bool(layer.use_lights), "frames": [_frame_row(obj, f) for f in layer.frames]}
    # MASKS AND GROUP, read back for MASK / UNMASK / GROUP / UNGROUP. use_mask_layer on 3.6 / 4.2, use_masks on 4.3+.
    toggle = "use_masks" if "use_masks" in layer.bl_rna.properties else "use_mask_layer"
    row["useMasks"] = bool(getattr(layer, toggle, False))
    row["masks"] = [m.name for m in getattr(layer, "mask_layers", [])]
    if _is_v3(obj):
        pg = getattr(layer, "parent_group", None)
        row["group"] = pg.name if pg is not None else None
    return row


def _layers(obj):
    return list(obj.data.layers)


def _find_layer(obj, name, tail, create=False):
    for layer in _layers(obj):
        if (layer.name if _is_v3(obj) else layer.info) == name:
            return layer, False
    if not create:
        have = [(lay.name if _is_v3(obj) else lay.info) for lay in _layers(obj)]
        raise MifOpError("'%s' has no layer '%s' (it has: %s). %s" % (obj.name, name, ", ".join(have) or "none", tail))
    return obj.data.layers.new(name), True


def gp_summary(obj):
    return {"object": obj.name, "generation": obj.type, "data": obj.data.name,
            "layers": [_layer_row(obj, lay) for lay in _layers(obj)],
            "materials": [m.name if m is not None else None for m in obj.data.materials]}


# ------------------------------------------------------------------------------------------------ create_grease_pencil

_CREATE_KEYS = ("name", "layer", "material", "color", "location", "collection")


def op_create_grease_pencil(params):
    """Create a Grease Pencil object with a layer and a stroke material, on whichever API generation this build has.

    params:
      name (str)          default "GPencil"; Blender's suffix on a clash is reported
      layer (str)         the first layer, default "Lines"
      material (str)      a stroke material to give it - made (black stroke, no fill) when no material has that name;
                          default "<name> Stroke"
      color [r,g,b,a]     the stroke colour of a material this call makes (default black)
      location [x,y,z]    world location; collection (str) an existing collection, default the scene's

    Reply: the object, generation GPENCIL (3.6, 4.2) or GREASEPENCIL (4.3+), its layers and materials.
    """
    reject_unknown(params, _CREATE_KEYS, "create_grease_pencil")
    name = take(params, "name", default="GPencil", kind=str)
    if not name.strip():
        raise MifOpError("name is blank. %s" % _ADDED)
    layer = take(params, "layer", default="Lines", kind=str)
    mat_name = take(params, "material", default=None, kind=str) or name + " Stroke"
    color = params.get("color")
    if color is not None:
        if not isinstance(color, (list, tuple)) or len(color) not in (3, 4):
            raise MifOpError("color takes [r, g, b] or [r, g, b, a], got %r. %s" % (color, _ADDED))
        color = finite_floats(list(color) + ([1.0] if len(color) == 3 else []), "color")
    loc = params.get("location")
    if loc is not None:
        if not isinstance(loc, (list, tuple)) or len(loc) != 3:
            raise MifOpError("location takes [x, y, z], got %r. %s" % (loc, _ADDED))
        loc = finite_floats(list(loc), "location")
    coll_name = take(params, "collection", default=None, kind=str)
    coll = bpy.context.scene.collection
    if coll_name is not None:
        coll = bpy.data.collections.get(coll_name)
        if coll is None:
            raise MifOpError("no collection named '%s'. %s" % (coll_name, _ADDED))
    if bpy.data.materials.get(mat_name) is not None and color is not None:
        raise MifOpError("material '%s' exists, and color is only for a material this call makes - set it with "
                         "set_grease_pencil_material. %s" % (mat_name, _ADDED))
    holder, gen = _v3_collection()
    data = holder.new(name)
    obj = bpy.data.objects.new(name, data)
    coll.objects.link(obj)
    if obj.type not in ("GPENCIL", "GREASEPENCIL"):
        remove_objects([obj])
        holder.remove(data)
        raise MifOpError("Blender %s made a %s from its Grease Pencil data - not a Grease Pencil object. %s"
                         % (bpy.app.version_string, obj.type, _ADDED))
    if loc is not None:
        obj.location = loc
    made_mat = bpy.data.materials.get(mat_name) is None
    mat = _gp_material(mat_name, True, _ADDED, color=color or ([0.0, 0.0, 0.0, 1.0] if made_mat else None))
    if made_mat:
        mat.grease_pencil.show_fill = False
        mat.grease_pencil.show_stroke = True
    obj.data.materials.append(mat)
    data.layers.new(layer)
    out = gp_summary(obj)
    out.update({"requestedName": name, "nameWasSuffixed": obj.name != name, "materialMade": made_mat,
                "collection": coll.name, "location": rnd(list(obj.location))})
    return out


# ------------------------------------------------------------------------------------------ draw_grease_pencil_strokes

_DRAW_KEYS = ("object", "layer", "frame", "strokes", "space", "surface", "replace")
_STROKE_KEYS = {"points", "radius", "radii", "lineWidth", "strength", "strengths", "material", "cyclic", "curveType",
                "caps", "fill", "hideStroke"}
# STROKE SHAPE AND FILL (2026-10-11, measured 3.6.23 to 5.2.1): curveType through drawing.set_types (POLY, CATMULL_ROM,
# BEZIER, NURBS; 4.3+ only - a legacy stroke is a polyline); caps start / end ROUND | FLAT on both generations
# (start_cap_mode on 3.6 / 4.2, start_cap 0 | 1 on 4.3+); a per-stroke fill colour and opacity (fill_color /
# fill_opacity on 4.3+, the legacy vertex_color_fill with the opacity in its alpha); hideStroke - draw the fill without
# the line - is 5.2+. No build has a corner-type property on a stroke (searched every Grease Pencil RNA struct).
_CURVE_TYPES = ("POLY", "CATMULL_ROM", "BEZIER", "NURBS")
# A v3 stroke's curve_type is the int index of set_types' enum, in its order (POLY reads 1 - measured).
_V3_CURVE_TYPES = ("CATMULL_ROM", "POLY", "BEZIER", "NURBS")
_CAPS = ("ROUND", "FLAT")


def _check_strokes(obj, raw, tail):
    if not isinstance(raw, list) or not raw:
        raise MifOpError("strokes takes a list of {points [[x,y,z], ...], radius | radii, lineWidth, strength | "
                         "strengths, material, cyclic}. %s" % tail)
    v3 = _is_v3(obj)
    total, out = 0, []
    for k, s in enumerate(raw):
        if not isinstance(s, dict) or set(s) - _STROKE_KEYS:
            raise MifOpError("strokes[%d] takes {%s}, got %r. %s" % (k, ", ".join(sorted(_STROKE_KEYS)), s, tail))
        pts = s.get("points")
        if not isinstance(pts, list) or len(pts) < 1:
            raise MifOpError("strokes[%d].points is a list of [x, y, z]. %s" % (k, tail))
        coords = []
        for i, p in enumerate(pts):
            if not isinstance(p, (list, tuple)) or len(p) != 3:
                raise MifOpError("strokes[%d].points[%d] must be [x, y, z], got %r. %s" % (k, i, p, tail))
            coords.append(Vector(finite_floats(list(p), "strokes[%d].points[%d]" % (k, i))))
        total += len(coords)
        n = len(coords)
        # bool() TOOK ANYTHING: a mapping or a list closed the stroke (2026-10-28, the version matrix's corrupted-value pass).
        if s.get("cyclic") is not None and not isinstance(s["cyclic"], bool):
            raise MifOpError("strokes[%d].cyclic is true or false, got %r. %s" % (k, s["cyclic"], tail))
        row = {"points": coords, "cyclic": bool(s.get("cyclic", False))}
        for one, many, key, lo, hi in (("radius", "radii", "radius", 0.0, 1000.0),
                                       ("strength", "strengths", "strength", 0.0, 1.0)):
            if s.get(one) is not None and s.get(many) is not None:
                raise MifOpError("strokes[%d]: pass %s OR %s. %s" % (k, one, many, tail))
            vals = None
            if s.get(many) is not None:
                if not isinstance(s[many], list) or len(s[many]) != n:
                    raise MifOpError("strokes[%d].%s takes one number per point (%d). %s" % (k, many, n, tail))
                vals = finite_floats(s[many], "strokes[%d].%s" % (k, many))
            elif s.get(one) is not None:
                vals = [float(finite_floats([s[one]], "strokes[%d].%s" % (k, one))[0])] * n
            if vals is not None and any(not lo <= v <= hi for v in vals):
                raise MifOpError("strokes[%d].%s must be %g-%g. %s" % (k, one, lo, hi, tail))
            row[key] = vals
        if row.get("radius") is not None and not v3:
            raise MifOpError("strokes[%d]: radius is a v3 point radius in metres (Blender 4.3+); this build's legacy "
                             "Grease Pencil draws a stroke lineWidth in pixels - pass lineWidth. %s" % (k, tail))
        if s.get("lineWidth") is not None:
            if v3:
                raise MifOpError("strokes[%d]: lineWidth is the legacy (3.6, 4.2) stroke width in pixels; Blender %s "
                                 "draws a point radius in metres - pass radius. %s" % (k, bpy.app.version_string, tail))
            lw = take_int(s, "lineWidth")
            if not 1 <= lw <= 1000:
                raise MifOpError("strokes[%d].lineWidth is 1-1000 pixels. %s" % (k, tail))
            row["lineWidth"] = lw
        if s.get("material") is not None:
            mat = bpy.data.materials.get(str(s["material"]))
            if mat is None or mat.grease_pencil is None:
                raise MifOpError("strokes[%d].material '%s' is not a Grease Pencil material (set_grease_pencil_material "
                                 "makes one). %s" % (k, s["material"], tail))
            row["material"] = mat
        if s.get("curveType") is not None:
            ct = str(s["curveType"]).upper()
            if ct not in _CURVE_TYPES:
                raise MifOpError("strokes[%d].curveType is one of %s, got %r. %s" % (k, ", ".join(_CURVE_TYPES),
                                                                                    s["curveType"], tail))
            if not v3:
                raise MifOpError("strokes[%d].curveType: a legacy Grease Pencil stroke (Blender %s) is a polyline - "
                                 "curve types are 4.3+. %s" % (k, bpy.app.version_string, tail))
            row["curveType"] = ct
        if s.get("caps") is not None:
            caps = s["caps"]
            if not isinstance(caps, dict) or not caps or set(caps) - {"start", "end"} \
                    or any(str(v).upper() not in _CAPS for v in caps.values()):
                raise MifOpError("strokes[%d].caps is {start, end}, each ROUND or FLAT, got %r. %s" % (k, caps, tail))
            row["caps"] = {e: str(v).upper() for e, v in caps.items()}
        if s.get("fill") is not None:
            fl = s["fill"]
            if not isinstance(fl, dict) or not fl or set(fl) - {"color", "opacity"}:
                raise MifOpError("strokes[%d].fill is {color [r,g,b(,a)], opacity 0-1}, got %r. %s" % (k, fl, tail))
            col = fl.get("color")
            if col is not None and (not isinstance(col, (list, tuple)) or len(col) not in (3, 4)):
                raise MifOpError("strokes[%d].fill.color takes [r, g, b] or [r, g, b, a]. %s" % (k, tail))
            col = finite_floats(list(col)[:3], "strokes[%d].fill.color" % k) if col is not None else None
            op = float(finite_floats([fl["opacity"]], "strokes[%d].fill.opacity" % k)[0]) \
                if fl.get("opacity") is not None else None
            if op is not None and not 0 <= op <= 1:
                raise MifOpError("strokes[%d].fill.opacity is 0-1. %s" % (k, tail))
            row["fill"] = {"color": col, "opacity": op}
        if s.get("hideStroke") is not None:
            hide = take_bool(s, "hideStroke")
            if hide and bpy.app.version < (5, 2, 0):
                raise MifOpError("strokes[%d].hideStroke (a fill with no line) is Blender 5.2+; this is %s. %s"
                                 % (k, bpy.app.version_string, tail))
            row["hideStroke"] = hide
        out.append(row)
    if total > _MAX_POINTS:
        raise MifOpError("%d points; one call draws at most %d. %s" % (total, _MAX_POINTS, tail))
    return out


def op_draw_grease_pencil_strokes(params):
    """Draw strokes on a Grease Pencil layer at a frame, from points - in world space, on a surface, or local.

    params:
      object (str, required), layer (str; made when missing), frame (int, default the current frame; a keyframe made
      when missing)
      strokes [{points [[x,y,z], ...], radius | radii (v3 point radius in metres), lineWidth (legacy pixels),
               strength | strengths (0-1: a v3 point's opacity, a legacy point's strength), material (a Grease Pencil
               material; added to the object), cyclic}]
      space WORLD (default) | LOCAL - the points' space
      surface {target (mesh), offset (default 0.01 m)} - each point moved to the nearest point of the target's surface
              and lifted along its normal by offset (strokes that sit ON a model without flickering into it)
      replace (bool) - clear that layer's frame first

    Reply: the generation, per stroke its point count and first and last points (world), the layer's frames with stroke
    and point counts, and with surface the largest difference between a point's distance from the target and offset.
    """
    tail = _CHANGED
    reject_unknown(params, _DRAW_KEYS, "draw_grease_pencil_strokes")
    obj = _gp_object(take(params, "object", required=True, kind=str), tail)
    layer_name = take(params, "layer", default=None, kind=str) or (
        (_layers(obj)[0].name if _is_v3(obj) else _layers(obj)[0].info) if _layers(obj) else "Lines")
    frame = take_int(params, "frame", default=int(bpy.context.scene.frame_current))
    space = str(params.get("space") or "WORLD").upper()
    if space not in ("WORLD", "LOCAL"):
        raise MifOpError("space is WORLD or LOCAL. %s" % tail)
    replace = take_bool(params, "replace", default=False)
    strokes = _check_strokes(obj, params.get("strokes"), tail)
    surface = params.get("surface")
    tree = None
    offset = 0.0
    if surface is not None:
        if not isinstance(surface, dict) or set(surface) - {"target", "offset"} or not surface.get("target"):
            raise MifOpError("surface takes {target (a mesh), offset}. %s" % tail)
        if space != "WORLD":
            raise MifOpError("surface places points in world space - leave space WORLD. %s" % tail)
        tgt = get_object(str(surface["target"]), want_mesh=True)
        offset = float(finite_floats([surface.get("offset", 0.01)], "surface.offset")[0])
        if not 0 <= offset <= 10:
            raise MifOpError("surface.offset must be 0-10 m. %s" % tail)
        from mathutils.bvhtree import BVHTree
        import bmesh
        bm = bmesh.new()
        try:
            ev = tgt.evaluated_get(bpy.context.evaluated_depsgraph_get())
            me = ev.to_mesh()
            bm.from_mesh(me)
            ev.to_mesh_clear()
            bm.transform(tgt.matrix_world)
            if not len(bm.faces):
                raise MifOpError("surface.target '%s' has no faces. %s" % (tgt.name, tail))
            tree = BVHTree.FromBMesh(bm)
        finally:
            bm.free()
    for s in strokes:
        if tree is not None:
            moved = []
            for p in s["points"]:
                loc, nrm, _i, _d = tree.find_nearest(p)
                moved.append(loc + nrm.normalized() * offset)
            s["points"] = moved
    # ------------------------------------------------------------------ writes
    layer, layer_made = _find_layer(obj, layer_name, tail, create=True)
    inv = obj.matrix_world.inverted_safe()
    fr = next((f for f in layer.frames if int(f.frame_number) == frame), None)
    frame_made = fr is None
    if fr is None:
        fr = layer.frames.new(frame)
    v3 = _is_v3(obj)
    if replace:
        if v3:
            d = fr.drawing
            if len(d.strokes):
                d.remove_strokes()
        else:
            for st in list(fr.strokes):
                fr.strokes.remove(st)
    rows = []
    for s in strokes:
        pts = [p if space == "LOCAL" else inv @ p for p in s["points"]]
        n = len(pts)
        mat_index = _slot_of(obj, s["material"]) if s.get("material") is not None else 0
        if v3:
            d = fr.drawing
            before = len(d.strokes)
            d.add_strokes([n])
            st = d.strokes[before]
            for i, p in enumerate(pts):
                pt = st.points[i]
                pt.position = p
                if s.get("radius") is not None:
                    pt.radius = s["radius"][i]
                if s.get("strength") is not None:
                    pt.opacity = s["strength"][i]
            st.cyclic = s["cyclic"]
            st.material_index = mat_index
            if s.get("curveType") is not None:
                # set_types takes the CURVE INDEX within the drawing; the stroke is looked up again after it.
                d.set_types(type=s["curveType"], indices=[before])
                st = d.strokes[before]
            for end, val in (s.get("caps") or {}).items():
                setattr(st, end + "_cap", _CAPS.index(val))
            if s.get("fill") is not None:
                if s["fill"]["color"] is not None:
                    st.fill_color = list(s["fill"]["color"]) + [1.0]
                if s["fill"]["opacity"] is not None:
                    st.fill_opacity = s["fill"]["opacity"]
            if s.get("hideStroke") is not None and hasattr(st, "hide_stroke"):
                st.hide_stroke = s["hideStroke"]
            back = [obj.matrix_world @ Vector(st.points[i].position) for i in (0, n - 1)]
            width = {"radius": [round(float(st.points[0].radius), 6), round(float(st.points[n - 1].radius), 6)],
                     "curveType": _V3_CURVE_TYPES[int(st.curve_type)] if 0 <= int(st.curve_type) < 4 else
                     int(st.curve_type),
                     "caps": {"start": _CAPS[int(st.start_cap)], "end": _CAPS[int(st.end_cap)]},
                     "fill": {"color": rnd(list(st.fill_color)[:3], 6), "opacity": round(float(st.fill_opacity), 6)}}
            if hasattr(st, "hide_stroke"):
                width["hideStroke"] = bool(st.hide_stroke)
        else:
            st = fr.strokes.new()
            st.points.add(n)
            for i, p in enumerate(pts):
                st.points[i].co = p
                st.points[i].pressure = 1.0
                if s.get("strength") is not None:
                    st.points[i].strength = s["strength"][i]
            st.use_cyclic = s["cyclic"]
            st.material_index = mat_index
            if s.get("lineWidth") is not None:
                st.line_width = s["lineWidth"]
            for end, val in (s.get("caps") or {}).items():
                setattr(st, end + "_cap_mode", val)
            if s.get("fill") is not None:
                old = list(st.vertex_color_fill)
                col = s["fill"]["color"] if s["fill"]["color"] is not None else old[:3]
                alpha = s["fill"]["opacity"] if s["fill"]["opacity"] is not None else 1.0
                st.vertex_color_fill = list(col) + [alpha]
            back = [obj.matrix_world @ Vector(st.points[i].co) for i in (0, n - 1)]
            vcf = list(st.vertex_color_fill)
            width = {"lineWidth": int(st.line_width), "curveType": "POLY",
                     "caps": {"start": st.start_cap_mode, "end": st.end_cap_mode},
                     "fill": {"color": rnd(vcf[:3], 6), "opacity": round(float(vcf[3]), 6)}}
        row = {"points": n, "first": rnd(list(back[0]), 6), "last": rnd(list(back[1]), 6), "cyclic": s["cyclic"],
               "materialIndex": mat_index}
        row.update(width)
        rows.append(row)
    out = {"object": obj.name, "generation": obj.type, "layer": layer_name, "layerMade": layer_made, "frame": frame,
           "frameMade": frame_made, "replaced": replace, "strokes": rows, "layerFrames": _layer_row(obj, layer)["frames"]}
    if tree is not None:
        worst = 0.0
        for s in strokes:
            for p in s["points"]:
                loc, _n, _i, d = tree.find_nearest(p)
                worst = max(worst, abs(d - offset))
        out["surface"] = {"target": surface["target"], "offset": offset, "largestOffsetError": round(worst, 6)}
    return out


# ------------------------------------------------------------------------------------------ set_grease_pencil_layers

_LAYER_KEYS = ("object", "action", "layer", "newName", "settings", "direction", "group", "mask")
_LAYER_ACTIONS = ("LIST", "ADD", "REMOVE", "RENAME", "SET", "MOVE", "DATA", "GROUP", "UNGROUP", "MASK", "UNMASK")
# DATA writes the Grease Pencil DATA's own settings - onion skin (onion_mode, onion_factor, ghost_before_range,
# ghost_after_range, before_color, after_color, use_ghost_custom_colors, onion_keyframe_type, use_onion_fade,
# use_onion_loop) and use_autolock_layers, the same names on both generations (measured 3.6.23 to 5.2.1; legacy also has
# use_onion_skinning there, v3 has it per layer). Not these: the collections and the name are other ops' business.
_DATA_DENY = frozenset({"name", "layers", "materials", "layer_groups", "rna_type", "users", "use_fake_user"})
_LAYER_MOVES = ("UP", "DOWN", "TOP", "BOTTOM")


def op_set_grease_pencil_layers(params):
    """List, add, remove, rename a Grease Pencil object's layers, or set a layer's properties by their RNA names
    (opacity, blend_mode, use_lights, hide, lock, tint_color, tint_factor, use_onion_skinning, ...).

    params: object (required), action LIST (default) | ADD | REMOVE | RENAME | SET | MOVE, layer, newName (RENAME),
    settings {property: value} (SET, and ADD for the new layer) - checked against the layer's own properties, read back;
    direction UP | DOWN | TOP | BOTTOM (MOVE) - the drawing order, the last layer drawn on top; a move that would leave
    the layer where it is is refused. DATA (no layer) writes settings on the Grease Pencil data instead - onion skin
    (onion_mode, onion_factor, ghost_before_range / ghost_after_range, before_color / after_color with
    use_ghost_custom_colors) and use_autolock_layers - all or nothing, read back. GROUP puts the layer in the layer group
    `group` (made when missing; 4.3+, where layer groups exist) and UNGROUP takes it out; MASK makes layer `mask` a mask
    of the layer (turning its masks on) and UNMASK removes it - not on 4.3-5.0, whose layers have no mask_layers.add.
    Reply: the object's layers with opacity, hide, lock, blend mode, lights and their frames' stroke counts.
    """
    tail = _CHANGED
    reject_unknown(params, _LAYER_KEYS, "set_grease_pencil_layers")
    obj = _gp_object(take(params, "object", required=True, kind=str), tail)
    action = str(params.get("action") or "LIST").upper()
    if action not in _LAYER_ACTIONS:
        raise MifOpError("action is one of %s. %s" % (", ".join(_LAYER_ACTIONS), tail))
    name = take(params, "layer", default=None, kind=str)
    settings = params.get("settings")
    if settings is not None and (not isinstance(settings, dict) or not settings):
        raise MifOpError("settings takes {property: value} of the layer's own properties. %s" % tail)
    if action == "DATA":
        if name:
            raise MifOpError("DATA writes the Grease Pencil data's own settings, not a layer's - drop layer, or use SET "
                             "for the layer. %s" % tail)
        if not settings:
            raise MifOpError("DATA needs settings {property: value} of the Grease Pencil data. %s" % tail)
    elif action != "LIST" and not name:
        raise MifOpError("%s needs layer. %s" % (action, tail))
    if settings is not None and action not in ("SET", "ADD", "DATA"):
        raise MifOpError("settings is for SET, ADD and DATA. %s" % tail)
    direction = take(params, "direction", default=None, kind=str)
    if (direction is not None) != (action == "MOVE"):
        raise MifOpError("MOVE takes direction (%s), and direction goes only with MOVE. %s"
                         % (" | ".join(_LAYER_MOVES), tail))
    group = take(params, "group", default=None, kind=str)
    if (group is not None) != (action == "GROUP"):
        raise MifOpError("GROUP takes group (the layer group's name), and group goes only with GROUP. %s" % tail)
    mask = take(params, "mask", default=None, kind=str)
    if (mask is not None) != (action in ("MASK", "UNMASK")):
        raise MifOpError("MASK and UNMASK take mask (the masking layer's name), and mask goes only with them. %s" % tail)
    out = {"object": obj.name, "generation": obj.type, "action": action}
    if action == "ADD":
        for lay in _layers(obj):
            if (lay.name if _is_v3(obj) else lay.info) == name:
                raise MifOpError("'%s' already has a layer '%s'. %s" % (obj.name, name, tail))
        layer = obj.data.layers.new(name)
        try:
            out["settingsApplied"] = _write_layer(layer, settings or {}, obj)
        except MifOpError:
            obj.data.layers.remove(layer)
            raise
    elif action == "REMOVE":
        layer, _m = _find_layer(obj, name, tail)
        obj.data.layers.remove(layer)
    elif action == "RENAME":
        new = take(params, "newName", required=True, kind=str)
        layer, _m = _find_layer(obj, name, tail)
        if _is_v3(obj):
            layer.name = new
        else:
            layer.info = new
    elif action == "SET":
        layer, _m = _find_layer(obj, name, tail)
        out["settingsApplied"] = _write_layer(layer, settings or {}, obj)
    elif action == "MOVE":
        out["moved"] = _move_layer(obj, name, direction.upper(), tail)
    elif action == "DATA":
        out["dataSettingsApplied"] = _write_data(obj, settings, tail)
    elif action in ("GROUP", "UNGROUP"):
        out["grouped"] = _group_layer(obj, name, group, tail)
    elif action in ("MASK", "UNMASK"):
        out["masked"] = _mask_layer(obj, name, mask, action == "MASK", tail)
    out.update(gp_summary(obj))
    return out


def _group_layer(obj, name, group, tail):
    """GROUP / UNGROUP through layers.move_to_layer_group (4.3+; a group of None is the root, measured 4.4.0-5.2.1)."""
    if not _is_v3(obj):
        raise MifOpError("layer groups are Grease Pencil 4.3+; '%s' is a legacy Grease Pencil object (Blender %s), "
                         "whose layers have no groups. %s" % (obj.name, bpy.app.version_string, tail))
    layer, _m = _find_layer(obj, name, tail)
    before = layer.parent_group.name if layer.parent_group is not None else None
    if group is None:
        if before is None:
            raise MifOpError("'%s' is in no layer group. %s" % (name, tail))
        obj.data.layers.move_to_layer_group(layer, None)
        made = False
    else:
        if before == group:
            raise MifOpError("'%s' is already in the layer group '%s'. %s" % (name, group, tail))
        g = next((x for x in obj.data.layer_groups if x.name == group), None)
        made = g is None
        if made:
            g = obj.data.layer_groups.new(group)
        obj.data.layers.move_to_layer_group(layer, g)
    layer, _m = _find_layer(obj, name, tail)
    after = layer.parent_group.name if layer.parent_group is not None else None
    if after != group:
        raise MifOpError("move_to_layer_group left '%s' in %r, not %r." % (name, after, group))
    return {"layer": name, "from": before, "to": after, "groupMade": made,
            "groups": [x.name for x in obj.data.layer_groups]}


def _mask_layer(obj, name, mask, add, tail):
    """MASK / UNMASK through layer.mask_layers - add exists on 3.6, 4.2 and 5.2.1 and not on 4.4.0 or 5.0.1 (measured
    2026-10-11), so those are refused by name. MASK turns the layer's masks on (use_mask_layer / use_masks)."""
    layer, _m = _find_layer(obj, name, tail)
    other, _m = _find_layer(obj, mask, tail)
    if mask == name:
        raise MifOpError("a layer cannot mask itself. %s" % tail)
    coll = layer.mask_layers
    have = [m.name for m in coll]
    toggle = "use_masks" if "use_masks" in layer.bl_rna.properties else "use_mask_layer"
    if add:
        if mask in have:
            raise MifOpError("'%s' already masks '%s'. %s" % (mask, name, tail))
        if not hasattr(coll, "add"):
            raise MifOpError("Blender %s's Grease Pencil layers have no mask_layers.add (measured missing on 4.4.0 and "
                             "5.0.1; 3.6, 4.2 and 5.2 have it), so a mask cannot be added here. %s"
                             % (bpy.app.version_string, tail))
        coll.add(other)
        setattr(layer, toggle, True)
    else:
        if mask not in have:
            raise MifOpError("'%s' does not mask '%s' (its masks: %s). %s" % (mask, name, ", ".join(have) or "none",
                                                                             tail))
        coll.remove(next(m for m in coll if m.name == mask))
    layer, _m = _find_layer(obj, name, tail)
    now = [m.name for m in layer.mask_layers]
    if (mask in now) != add:
        raise MifOpError("mask_layers did not %s '%s' (masks now: %s)." % ("take" if add else "drop", mask,
                                                                         ", ".join(now) or "none"))
    return {"layer": name, "mask": mask, "masks": now, "useMasks": bool(getattr(layer, toggle))}


def _write_data(obj, settings, tail):
    """Settings on the Grease Pencil data by RNA name, all or nothing: what was there is put back on a refusal."""
    from .ops_rig import _restore_settings, _rna_copy
    data = obj.data
    held, got = {}, {}
    try:
        for k, v in settings.items():
            rna = data.bl_rna.properties.get(k)
            if rna is not None and not rna.is_readonly and k not in _DATA_DENY:
                held[k] = _rna_copy(getattr(data, k))
            got[k] = write_rna_property(data, k, v, tail, "Grease Pencil data '%s'" % data.name, "Grease Pencil data",
                                        _DATA_DENY)
    except MifOpError:
        _restore_settings(data, held)
        raise
    return got


def _layer_names(obj):
    return [(lay.name if _is_v3(obj) else lay.info) for lay in _layers(obj)]


def _move_layer(obj, name, direction, tail):
    """Move a layer in the drawing order with layers.move(UP | DOWN) - the one call both generations have (3.6 to 5.2) -
    stepped to an end for TOP and BOTTOM. Read back; a move that changes nothing is refused before any write."""
    if direction not in _LAYER_MOVES:
        raise MifOpError("direction is one of %s, got %r. %s" % (", ".join(_LAYER_MOVES), direction, tail))
    layer, _m = _find_layer(obj, name, tail)
    before = _layer_names(obj)
    at, last = before.index(name), len(before) - 1
    want = {"UP": at + 1, "DOWN": at - 1, "TOP": last, "BOTTOM": 0}[direction]
    if not 0 <= want <= last or want == at:
        raise MifOpError("'%s' is layer %d of %d (%s, bottom first), so %s leaves it where it is. %s"
                         % (name, at, len(before), ", ".join(before), direction, tail))
    step = "UP" if want > at else "DOWN"
    for _i in range(abs(want - at)):
        obj.data.layers.move(layer, type=step)
    after = _layer_names(obj)
    if after.index(name) != want:
        raise MifOpError("layers.move left '%s' at %d, not %d (%s). The order is now %s."
                         % (name, after.index(name), want, direction, ", ".join(after)))
    return {"layer": name, "from": at, "to": want, "orderBefore": before, "orderAfter": after}


def _write_layer(layer, settings, obj):
    got = {}
    for k, v in settings.items():
        if k in ("name", "info", "frames"):
            raise MifOpError("'%s' is not set here - RENAME renames a layer, draw_grease_pencil_strokes makes frames. %s"
                             % (k, _CHANGED))
        got[k] = write_rna_property(layer, k, v, "Settings before it in this call were applied.",
                                    "layer of '%s'" % obj.name, "Grease Pencil layer", frozenset({"name", "info"}))
    return got


# ---------------------------------------------------------------------------------------- set_grease_pencil_material

_MAT_KEYS = ("material", "create", "object", "stroke", "fill", "settings")


def op_set_grease_pencil_material(params):
    """Make or edit a Grease Pencil material - its stroke and fill - and give it to an object.

    params:
      material (str, required), create (bool, default true: made when missing; an existing ordinary material gets
      Grease Pencil data)
      stroke {show, color [r,g,b,a], style SOLID | TEXTURE, mode LINE | DOTS | BOX}
      fill {show, color [r,g,b,a], style SOLID | GRADIENT | TEXTURE}
      settings {property: value} - any other mat.grease_pencil property by its RNA name (5.2's placement_mode and
      random_*, holdouts, pass_index, ...)
      object (str) - a Grease Pencil object to add the material to

    Reply: the material's Grease Pencil settings read back, and the object's materials.
    """
    tail = _CHANGED
    reject_unknown(params, _MAT_KEYS, "set_grease_pencil_material")
    name = take(params, "material", required=True, kind=str)
    create = take_bool(params, "create", default=True)
    obj = _gp_object(take(params, "object", kind=str), tail) if params.get("object") is not None else None
    writes = {}
    for group, mapping in (("stroke", {"show": "show_stroke", "color": "color", "style": "stroke_style",
                                       "mode": "mode"}),
                           ("fill", {"show": "show_fill", "color": "fill_color", "style": "fill_style"})):
        raw = params.get(group)
        if raw is None:
            continue
        if not isinstance(raw, dict) or not raw or set(raw) - set(mapping):
            raise MifOpError("%s takes {%s}, got %r. %s" % (group, ", ".join(mapping), raw, tail))
        for k, v in raw.items():
            writes[mapping[k]] = v
    extra = params.get("settings")
    if extra is not None:
        if not isinstance(extra, dict) or not extra:
            raise MifOpError("settings takes {property: value} of mat.grease_pencil. %s" % tail)
        writes.update(extra)
    existed = bpy.data.materials.get(name) is not None
    mat = _gp_material(name, create, tail)
    gp = mat.grease_pencil
    done = {}
    for k, v in writes.items():
        if isinstance(v, (list, tuple)) and k in ("color", "fill_color") and len(v) == 3:
            v = list(v) + [1.0]
        done[k] = write_rna_property(gp, k, v, "Writes before it in this call were applied.",
                                     "Grease Pencil material '%s'" % mat.name, "Grease Pencil material",
                                     frozenset({"rna_type"}))
    if obj is not None:
        _slot_of(obj, mat)
    props = ("show_stroke", "show_fill", "color", "fill_color", "mode", "stroke_style", "fill_style")
    out = {"material": mat.name, "created": not existed, "applied": done,
           "settings": {p: (rnd(list(getattr(gp, p)), 6) if p in ("color", "fill_color") else getattr(gp, p))
                        for p in props if p in gp.bl_rna.properties}}
    if obj is not None:
        out["object"] = obj.name
        out["objectMaterials"] = [m.name if m is not None else None for m in obj.data.materials]
    return out


# ----------------------------------------------------------------------------------------- set_grease_pencil_frames

_FRAME_KEYS = ("object", "layer", "action", "frames", "to", "offset", "factor", "pivot", "step", "keyframeType",
               "detail", "space", "limit")
_FRAME_ACTIONS = ("LIST", "INSERT", "DELETE", "MOVE", "DUPLICATE", "SCALE_TIMING", "INTERPOLATE", "SET_TYPE")


def _frames_by_number(layer):
    return {int(f.frame_number): f for f in layer.frames}


def _frame_ints(raw, key, tail, need=None):
    if not isinstance(raw, list) or not raw:
        raise MifOpError("%s is a list of frame numbers. %s" % (key, tail))
    out = []
    for v in raw:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v):
            raise MifOpError("%s takes whole frame numbers, got %r. %s" % (key, v, tail))
        out.append(int(v))
    if len(set(out)) != len(out):
        raise MifOpError("%s names a frame twice. %s" % (key, tail))
    if need is not None and len(out) != need:
        raise MifOpError("%s takes exactly %d frame number(s) here, got %d. %s" % (key, need, len(out), tail))
    return out


def _renumber(obj, layer, mapping):
    """Move frames old -> new, never onto a frame that is still occupied: each pass moves the frames whose target is free
    (a shift or a scale always leaves one), so no two frames share a number at any step. Checked before any write."""
    v3 = _is_v3(obj)
    todo = dict(mapping)
    while todo:
        have = _frames_by_number(layer)
        ready = [(o, n) for o, n in todo.items() if n not in have or n == o]
        if not ready:
            raise MifOpError("frames %s cannot be renumbered one at a time without a collision. The layer's frames are "
                             "now %s." % (sorted(todo), sorted(have)))
        for o, n in ready:
            if o != n:
                if v3:
                    layer.frames.move(o, n)
                else:
                    have[o].frame_number = n
            del todo[o]


def _blend_frame(obj, layer, a, b, num, t, ktype):
    """A new frame at num, each stroke the point-by-point blend of frame a's and frame b's (t from a) - the native
    in-between. The frames are looked up by number AFTER the new one is made: adding a frame can move the others in
    memory, and a wrapper taken before it would point at the old place."""
    layer.frames.new(num)
    by = _frames_by_number(layer)
    fa, fb, new = by[a], by[b], by[num]
    if _is_v3(obj):
        da, db, d = fa.drawing, fb.drawing, new.drawing
        d.add_strokes([len(s.points) for s in da.strokes])
        for sa, sb, s in zip(da.strokes, db.strokes, d.strokes):
            for pa, pb, p in zip(sa.points, sb.points, s.points):
                p.position = Vector(pa.position).lerp(Vector(pb.position), t)
                p.radius = pa.radius + (pb.radius - pa.radius) * t
                p.opacity = pa.opacity + (pb.opacity - pa.opacity) * t
            s.cyclic = sa.cyclic
            s.material_index = sa.material_index
    else:
        for sa, sb in zip(fa.strokes, fb.strokes):
            s = new.strokes.new()
            s.points.add(len(sa.points))
            for pa, pb, p in zip(sa.points, sb.points, s.points):
                p.co = Vector(pa.co).lerp(Vector(pb.co), t)
                p.pressure = pa.pressure + (pb.pressure - pa.pressure) * t
                p.strength = pa.strength + (pb.strength - pa.strength) * t
            s.use_cyclic = sa.use_cyclic
            s.material_index = sa.material_index
            s.line_width = int(round(sa.line_width + (sb.line_width - sa.line_width) * t))
    new.keyframe_type = ktype
    return new


def _stroke_shape(obj, fr):
    strokes = fr.drawing.strokes if _is_v3(obj) else fr.strokes
    return [len(s.points) for s in strokes]


def _stroke_detail(obj, fr, mw, limit, used):
    """Each stroke of a keyframe with its points - the read half of draw_grease_pencil_strokes (1.3 gap audit)."""
    rows = []
    strokes = fr.drawing.strokes if _is_v3(obj) else fr.strokes
    for si, s in enumerate(strokes):
        pts = []
        for p in s.points:
            if used[0] >= limit:
                break
            used[0] += 1
            if _is_v3(obj):
                co = p.position
                row = {"co": rnd(list(mw @ co if mw is not None else co), 6), "radius": round(float(p.radius), 6),
                       "opacity": round(float(p.opacity), 6), "vertexColor": rnd(list(p.vertex_color), 6)}
            else:
                co = p.co
                row = {"co": rnd(list(mw @ co if mw is not None else co), 6), "pressure": round(float(p.pressure), 6),
                       "strength": round(float(p.strength), 6), "vertexColor": rnd(list(p.vertex_color), 6)}
            pts.append(row)
        mats = obj.data.materials
        mi = int(s.material_index)
        rows.append({"index": si, "material": mats[mi].name if mi < len(mats) and mats[mi] is not None else None,
                     "cyclic": bool(s.cyclic if _is_v3(obj) else s.use_cyclic), "pointCount": len(s.points),
                     "points": pts})
    return rows


def op_set_grease_pencil_frames(params):
    """List and edit the keyframes on a Grease Pencil layer: insert blank ones, delete, move, duplicate, scale their
    timing, interpolate in-betweens, or set their keyframe type - on either API generation.

    params:
      object (required), layer (default the first layer), action LIST (default) | INSERT | DELETE | MOVE | DUPLICATE |
      SCALE_TIMING | INTERPOLATE | SET_TYPE
      frames [n, ...]   INSERT, DELETE, SET_TYPE: the frames; MOVE: the frames to move (one with `to`, any with `offset`);
                        DUPLICATE: the one frame copied; SCALE_TIMING: the frames retimed (default all);
                        INTERPOLATE: the two keyframes [a, b] to fill between
      to (int)          MOVE one frame, or DUPLICATE, to this frame number
      offset (int)      MOVE every frame named by this many frames
      factor (number)   SCALE_TIMING: each frame's distance from pivot times this (rounded to a whole frame)
      pivot (int)       SCALE_TIMING: the frame that stays put (default the first of frames)
      step (int)        INTERPOLATE: an in-between every step frames (default 1)
      keyframeType      INSERT and INTERPOLATE (default KEYFRAME and BREAKDOWN), SET_TYPE: KEYFRAME | BREAKDOWN |
                        MOVING_HOLD | EXTREME | JITTER (| GENERATED where the build has it)

    INTERPOLATE is a native point-by-point blend (position, radius or pressure, opacity or strength; each stroke's
    material and cyclic from the first keyframe): Blender's own interpolate operators need an edit-mode 3D view. It
    needs the same stroke and point counts on both keyframes, no frame between them, and neither bounded by a breakdown.

    Every check runs before the first write; a frame number two frames would share is refused. Reply: the layer's
    frames after, each with its keyframe type and stroke and point counts.

    LIST detail:true (1.3) adds strokes: per keyframe (frames, default all) every stroke's material, cyclic and points -
    position (space LOCAL | WORLD), radius and opacity (4.3+) or pressure and strength (3.6, 4.2), vertex color - capped
    at limit points (default 5000). The read half of draw_grease_pencil_strokes.
    """
    tail = _CHANGED
    reject_unknown(params, _FRAME_KEYS, "set_grease_pencil_frames")
    obj = _gp_object(take(params, "object", required=True, kind=str), tail)
    names = _layer_names(obj)
    if not names:
        raise MifOpError("'%s' has no layers. %s" % (obj.name, tail))
    layer_name = take(params, "layer", default=None, kind=str) or names[0]
    layer, _m = _find_layer(obj, layer_name, tail)
    action = str(params.get("action") or "LIST").upper()
    if action not in _FRAME_ACTIONS:
        raise MifOpError("action is one of %s. %s" % (", ".join(_FRAME_ACTIONS), tail))
    allowed = {"LIST": {"frames", "detail", "space", "limit"}, "INSERT": {"frames", "keyframeType"}, "DELETE": {"frames"},
               "MOVE": {"frames", "to", "offset"}, "DUPLICATE": {"frames", "to"},
               "SCALE_TIMING": {"frames", "factor", "pivot"}, "INTERPOLATE": {"frames", "step", "keyframeType"},
               "SET_TYPE": {"frames", "keyframeType"}}[action]
    stray = [k for k in ("frames", "to", "offset", "factor", "pivot", "step", "keyframeType", "detail", "space", "limit")
             if params.get(k) is not None and k not in allowed]
    if stray:
        raise MifOpError("%s does not take %s. %s" % (action, ", ".join(stray), tail))
    types = [i.identifier for i in layer.frames[0].bl_rna.properties["keyframe_type"].enum_items] if len(layer.frames) \
        else None
    ktype = take(params, "keyframeType", default=None, kind=str)
    if ktype is not None:
        ktype = ktype.upper()
        if types is None:
            types = ["KEYFRAME", "BREAKDOWN", "MOVING_HOLD", "EXTREME", "JITTER"]
        if ktype not in types:
            raise MifOpError("keyframeType is one of %s, got %r. %s" % (", ".join(types), ktype, tail))
    have = _frames_by_number(layer)
    detail = take_bool(params, "detail", default=False)
    out = {"object": obj.name, "generation": obj.type, "layer": layer_name, "action": action}

    def missing(nums):
        gone = [n for n in nums if n not in have]
        if gone:
            raise MifOpError("layer '%s' has no keyframe at %s (it has %s). %s"
                             % (layer_name, ", ".join(map(str, gone)), sorted(have) or "none", tail))

    if action == "INSERT":
        nums = _frame_ints(params.get("frames"), "frames", tail)
        taken = [n for n in nums if n in have]
        if taken:
            raise MifOpError("layer '%s' already has a keyframe at %s. %s" % (layer_name, taken, tail))
        for n in nums:
            layer.frames.new(n).keyframe_type = ktype or "KEYFRAME"
        out["inserted"] = nums
    elif action == "DELETE":
        nums = _frame_ints(params.get("frames"), "frames", tail)
        missing(nums)
        for n in nums:
            if _is_v3(obj):
                layer.frames.remove(n)
            else:
                layer.frames.remove(_frames_by_number(layer)[n])
        out["deleted"] = nums
    elif action == "SET_TYPE":
        nums = _frame_ints(params.get("frames"), "frames", tail)
        missing(nums)
        if ktype is None:
            raise MifOpError("SET_TYPE needs keyframeType. %s" % tail)
        for n in nums:
            have[n].keyframe_type = ktype
        out["set"] = nums
    elif action in ("MOVE", "SCALE_TIMING"):
        if action == "MOVE":
            nums = _frame_ints(params.get("frames"), "frames", tail)
            if (params.get("to") is None) == (params.get("offset") is None):
                raise MifOpError("MOVE takes to (one frame) or offset (any number of frames). %s" % tail)
            if params.get("to") is not None:
                if len(nums) != 1:
                    raise MifOpError("to moves one frame - pass offset to move %d together. %s" % (len(nums), tail))
                mapping = {nums[0]: take_int(params, "to")}
            else:
                off = take_int(params, "offset")
                if off == 0:
                    raise MifOpError("offset 0 moves nothing. %s" % tail)
                mapping = {n: n + off for n in nums}
        else:
            nums = (_frame_ints(params.get("frames"), "frames", tail) if params.get("frames") is not None
                    else sorted(have))
            factor = float(finite_floats([params.get("factor")], "factor")[0]) if params.get("factor") is not None \
                else None
            if factor is None or not factor > 0:
                raise MifOpError("SCALE_TIMING needs factor above 0. %s" % tail)
            pivot = take_int(params, "pivot", default=min(nums))
            mapping = {n: pivot + int(round((n - pivot) * factor)) for n in nums}
        missing(list(mapping))
        targets = list(mapping.values())
        if len(set(targets)) != len(targets):
            raise MifOpError("%s would put two frames on one number (%s). %s"
                             % (action, ", ".join("%d->%d" % kv for kv in sorted(mapping.items())), tail))
        clash = sorted(n for n in targets if n in have and n not in mapping)
        if clash:
            raise MifOpError("%s would land on keyframe(s) %s, which stay where they are. %s" % (action, clash, tail))
        if all(o == n for o, n in mapping.items()):
            raise MifOpError("%s leaves every frame where it is. %s" % (action, tail))
        _renumber(obj, layer, mapping)
        out["moved"] = [{"from": o, "to": n} for o, n in sorted(mapping.items()) if o != n]
    elif action == "DUPLICATE":
        src = _frame_ints(params.get("frames"), "frames", tail, need=1)[0]
        missing([src])
        if params.get("to") is None:
            raise MifOpError("DUPLICATE needs to (the new frame's number). %s" % tail)
        dst = take_int(params, "to")
        if dst in have:
            raise MifOpError("layer '%s' already has a keyframe at %d. %s" % (layer_name, dst, tail))
        if _is_v3(obj):
            layer.frames.copy(src, dst)
        else:
            # copy() places the copy on the next free number (measured 3.6.23, 4.2.17); it is then renumbered.
            new = layer.frames.copy(have[src])
            new.frame_number = dst
        out["duplicated"] = {"from": src, "to": dst}
    elif action == "INTERPOLATE":
        a, b = sorted(_frame_ints(params.get("frames"), "frames", tail, need=2))
        missing([a, b])
        step = take_int(params, "step", default=1)
        if step < 1:
            raise MifOpError("step is 1 or more. %s" % tail)
        between = [n for n in have if a < n < b]
        if between:
            raise MifOpError("layer '%s' already has keyframe(s) %s between %d and %d - interpolate between neighbours. "
                             "%s" % (layer_name, sorted(between), a, b, tail))
        if b - a <= step:
            raise MifOpError("%d and %d have no frame between them at step %d. %s" % (a, b, step, tail))
        bad = [n for n in (a, b) if have[n].keyframe_type == "BREAKDOWN"]
        if bad:
            raise MifOpError("keyframe %s is a breakdown - interpolation runs between key poses, not from an in-between. "
                             "%s" % (bad, tail))
        sa, sb = _stroke_shape(obj, have[a]), _stroke_shape(obj, have[b])
        if sa != sb:
            raise MifOpError("frames %d and %d differ in strokes or points (%s vs %s) - a point-by-point blend needs the "
                             "same counts on both. %s" % (a, b, sa, sb, tail))
        made = []
        for n in range(a + step, b, step):
            _blend_frame(obj, layer, a, b, n, (n - a) / float(b - a), ktype or "BREAKDOWN")
            made.append(n)
        out["interpolated"] = made
        # READ BACK: each in-between's first stroke's first point in world space, off the frame as written.
        by = _frames_by_number(layer)
        samples = []
        for n in made:
            strokes = by[n].drawing.strokes if _is_v3(obj) else by[n].strokes
            if len(strokes) and len(strokes[0].points):
                p0 = strokes[0].points[0]
                local = Vector(p0.position if _is_v3(obj) else p0.co)
                samples.append({"frame": n, "firstPoint": rnd(list(obj.matrix_world @ local), 6)})
        out["firstPoints"] = samples
    out["frames"] = [_frame_row(obj, f) for f in sorted(layer.frames, key=lambda f: int(f.frame_number))]
    if action == "LIST" and (params.get("detail") is not None or params.get("frames") is not None):
        # STROKE POINTS (1.3): detail true lists every stroke of the keyframes named (frames, default all) with its
        # points - position (space LOCAL default | WORLD), radius and opacity (4.3+) or pressure and strength (3.6 / 4.2),
        # vertex color - capped at limit points (default 5000) with truncated said
        want = _frame_ints(params.get("frames"), "frames", tail) if params.get("frames") is not None else sorted(have)
        missing(want)
        space = str(params.get("space") or "LOCAL").upper()
        if space not in ("LOCAL", "WORLD"):
            raise MifOpError("space is LOCAL or WORLD, got %r. %s" % (params.get("space"), tail))
        limit = int(params.get("limit") or 5000)
        if detail:
            used = [0]
            mw = obj.matrix_world if space == "WORLD" else None
            out["strokes"] = [{"frame": n, "strokes": _stroke_detail(obj, have[n], mw, limit, used)} for n in want]
            total = sum(r["points"] for f in want for r in [_frame_row(obj, have[f])])
            out["pointsListed"] = used[0]
            out["truncated"] = used[0] < total
            out["space"] = space
        out["frames"] = [r for r in out["frames"] if r["frame"] in set(want)]
    return out


# ------------------------------------------------------------------------------------ transform_grease_pencil_points

_XFORM_KEYS = ("object", "layer", "frame", "strokes", "translate", "rotate", "scale", "pivot")


def _vec3(raw, key, tail):
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise MifOpError("%s takes [x, y, z], got %r. %s" % (key, raw, tail))
    return Vector(finite_floats(list(raw), key))


def _bounds(pts):
    if not pts:
        return None
    return {"min": rnd([min(p[i] for p in pts) for i in range(3)], 6), "max": rnd([max(p[i] for p in pts) for i in range(3)],
                                                                                   6)}


def op_transform_grease_pencil_points(params):
    """Translate, rotate and scale the points of a Grease Pencil layer's strokes at one keyframe - written to the stroke
    data, no edit mode - for a head turn on its own layer, a drawing nudged or resized in place.

    params:
      object (required), layer (required), frame (the keyframe to edit; default the one shown at the scene's frame -
      a frame that is not a keyframe is refused, because its drawing is a hold of an earlier one)
      strokes [i, ...]     stroke indices on that keyframe (default every stroke)
      translate [x,y,z]    metres, world space
      rotate [x,y,z]       degrees, an XYZ Euler rotation in world space about pivot
      scale s | [x,y,z]    about pivot
      pivot MEDIAN (default: the median of the points moved) | ORIGIN (the object's) | CURSOR | [x,y,z] (world)

    Order: scale, then rotate, about pivot, then translate. Reply: the points moved, the pivot used and their world bounds
    before and after, read back off the strokes.
    """
    tail = _CHANGED
    reject_unknown(params, _XFORM_KEYS, "transform_grease_pencil_points")
    obj = _gp_object(take(params, "object", required=True, kind=str), tail)
    layer_name = take(params, "layer", required=True, kind=str)
    layer, _m = _find_layer(obj, layer_name, tail)
    have = _frames_by_number(layer)
    if params.get("frame") is not None:
        frame = take_int(params, "frame")
        if frame not in have:
            raise MifOpError("layer '%s' has no keyframe at %d (it has %s) - a frame between keyframes holds an earlier "
                             "drawing. %s" % (layer_name, frame, sorted(have) or "none", tail))
    else:
        now = int(bpy.context.scene.frame_current)
        shown = [n for n in have if n <= now]
        if not shown:
            raise MifOpError("layer '%s' shows no drawing at the scene's frame %d (keyframes: %s) - pass frame. %s"
                             % (layer_name, now, sorted(have) or "none", tail))
        frame = max(shown)
    fr = have[frame]
    strokes = list(fr.drawing.strokes if _is_v3(obj) else fr.strokes)
    if params.get("strokes") is not None:
        idx = _frame_ints(params.get("strokes"), "strokes", tail)
        bad = [i for i in idx if not 0 <= i < len(strokes)]
        if bad:
            raise MifOpError("stroke index %s is outside 0-%d on frame %d. %s" % (bad, len(strokes) - 1, frame, tail))
    else:
        idx = list(range(len(strokes)))
    if not idx:
        raise MifOpError("frame %d of layer '%s' has no strokes. %s" % (frame, layer_name, tail))
    if all(params.get(k) is None for k in ("translate", "rotate", "scale")):
        raise MifOpError("pass translate, rotate or scale - nothing to do. %s" % tail)
    move = _vec3(params["translate"], "translate", tail) if params.get("translate") is not None else Vector((0, 0, 0))
    rot = None
    if params.get("rotate") is not None:
        from math import radians
        from mathutils import Euler
        r = _vec3(params["rotate"], "rotate", tail)
        rot = Euler((radians(r.x), radians(r.y), radians(r.z)), "XYZ").to_matrix()
    sc = Vector((1, 1, 1))
    if params.get("scale") is not None:
        raw = params["scale"]
        sc = (Vector([finite_floats([raw], "scale")[0]] * 3) if isinstance(raw, (int, float))
              and not isinstance(raw, bool) else _vec3(raw, "scale", tail))
        if any(abs(v) < 1e-9 for v in sc):
            raise MifOpError("scale 0 collapses the points onto the pivot. %s" % tail)
    mw, inv = obj.matrix_world, obj.matrix_world.inverted_safe()
    v3 = _is_v3(obj)

    def world_points():
        out = []
        for i in idx:
            for p in strokes[i].points:
                out.append(mw @ Vector(p.position if v3 else p.co))
        return out

    before = world_points()
    piv = params.get("pivot", "MEDIAN")
    if isinstance(piv, str):
        piv_u = piv.upper()
        if piv_u == "MEDIAN":
            pivot = sum(before, Vector((0, 0, 0))) / len(before)
        elif piv_u == "ORIGIN":
            pivot = mw.translation.copy()
        elif piv_u == "CURSOR":
            pivot = bpy.context.scene.cursor.location.copy()
        else:
            raise MifOpError("pivot is MEDIAN, ORIGIN, CURSOR or [x, y, z], got %r. %s" % (piv, tail))
    else:
        pivot = _vec3(piv, "pivot", tail)
    # ------------------------------------------------------------------ writes
    for i in idx:
        for p in strokes[i].points:
            w = mw @ Vector(p.position if v3 else p.co) - pivot
            w = Vector((w.x * sc.x, w.y * sc.y, w.z * sc.z))
            if rot is not None:
                w = rot @ w
            loc = inv @ (w + pivot + move)
            if v3:
                p.position = loc
            else:
                p.co = loc
    strokes = list(have[frame].drawing.strokes if v3 else have[frame].strokes)
    after = world_points()
    return {"object": obj.name, "generation": obj.type, "layer": layer_name, "frame": frame, "strokes": idx,
            "pointsMoved": len(after), "pivot": rnd(list(pivot), 6), "boundsBefore": _bounds(before),
            "boundsAfter": _bounds(after)}


# ---------------------------------------------------------------------------------------- set_grease_pencil_effects

_FX_KEYS = ("object", "action", "effect", "type", "settings")
_FX_ACTIONS = ("LIST", "ADD", "REMOVE", "SET")
# Not settings: name is how every action finds an effect, type is fixed for its life, show_expanded is panel state.
_FX_DENY = frozenset({"name", "type", "show_expanded", "rna_type"})


def _fx_row(fx):
    return {"name": fx.name, "type": fx.type, "showViewport": bool(fx.show_viewport),
            "showRender": bool(fx.show_render), "settings": rna_properties(fx, skip=_FX_DENY)}


def op_set_grease_pencil_effects(params):
    """List, add, remove or set a Grease Pencil object's visual effects (Blur, Colorize, Flip, Glow, Pixelate, Rim,
    Shadow, Swirl, Wave) - obj.shader_effects, the same nine FX_* types on 3.6 to 5.2.

    params: object (required), action LIST (default) | ADD | REMOVE | SET, effect (its name; ADD defaults to the type's
    title), type (ADD: FX_GLOW and the rest, or GLOW without the prefix), settings {property: value} by RNA name (ADD
    and SET) - checked against the effect's own properties, read back; a refused setting on ADD removes the new effect.
    Reply: the object's effects in stack order, each with its settings read back.
    """
    tail = _CHANGED
    reject_unknown(params, _FX_KEYS, "set_grease_pencil_effects")
    obj = _gp_object(take(params, "object", required=True, kind=str), tail)
    action = str(params.get("action") or "LIST").upper()
    if action not in _FX_ACTIONS:
        raise MifOpError("action is one of %s. %s" % (", ".join(_FX_ACTIONS), tail))
    name = take(params, "effect", default=None, kind=str)
    fx_type = take(params, "type", default=None, kind=str)
    settings = params.get("settings")
    if settings is not None and (not isinstance(settings, dict) or not settings):
        raise MifOpError("settings takes {property: value} of the effect's own properties. %s" % tail)
    if settings is not None and action not in ("ADD", "SET"):
        raise MifOpError("settings is for ADD and SET. %s" % tail)
    if fx_type is not None and action != "ADD":
        raise MifOpError("type is for ADD - an effect's type is fixed once it exists. %s" % tail)
    if action in ("REMOVE", "SET") and not name:
        raise MifOpError("%s needs effect (the effect's name). %s" % (action, tail))
    stack = obj.shader_effects
    out = {"object": obj.name, "generation": obj.type, "action": action}
    if action == "ADD":
        if not fx_type:
            raise MifOpError("ADD needs type. %s" % tail)
        valid = [t.identifier for t in bpy.types.ShaderFx.bl_rna.properties["type"].enum_items]
        want = fx_type.upper()
        if want not in valid and "FX_" + want in valid:
            want = "FX_" + want
        if want not in valid:
            raise MifOpError("type %r is not an effect on Blender %s - it takes %s. %s"
                             % (fx_type, bpy.app.version_string, ", ".join(valid), tail))
        name = name or want[3:].title()
        if stack.get(name) is not None:
            raise MifOpError("'%s' already has an effect named '%s'. %s" % (obj.name, name, tail))
        fx = stack.new(name, want)
        if fx is None:
            raise MifOpError("shader_effects.new returned none for %s on '%s'. %s" % (want, obj.name, tail))
        try:
            out["settingsApplied"] = _write_fx(fx, settings or {}, obj, "NOTHING was added.")
        except MifOpError:
            stack.remove(fx)
            raise
        out["effect"] = fx.name
    elif action in ("REMOVE", "SET"):
        fx = stack.get(name)
        if fx is None:
            raise MifOpError("'%s' has no effect named '%s' (it has: %s). %s"
                             % (obj.name, name, ", ".join(f.name for f in stack) or "none", tail))
        if action == "REMOVE":
            stack.remove(fx)
        else:
            if not settings:
                raise MifOpError("SET needs settings. %s" % tail)
            out["settingsApplied"] = _write_fx(fx, settings, obj, tail)
        out["effect"] = name
    out["effects"] = [_fx_row(f) for f in stack]
    return out


def _write_fx(fx, settings, obj, tail):
    """Every setting written by RNA name, all or nothing: what was there is put back on a refusal."""
    from .ops_rig import _restore_settings, _rna_copy
    held, got = {}, {}
    try:
        for k, v in settings.items():
            rna = fx.bl_rna.properties.get(k)
            if rna is not None and not rna.is_readonly and k not in _FX_DENY:
                held[k] = _rna_copy(getattr(fx, k))
            got[k] = write_rna_property(fx, k, v, tail, "effect '%s' on '%s'" % (fx.name, obj.name),
                                        "%s effect" % fx.type, _FX_DENY)
    except MifOpError:
        _restore_settings(fx, held)
        raise
    return got


# ---------------------------------------------------------------------------------------- edit_grease_pencil_strokes
_EDIT_KEYS = ("object", "layer", "frame", "strokes", "action", "settings")
_EDIT_ACTIONS = {"SIMPLIFY": ("distance",), "SMOOTH": ("iterations", "factor"), "SUBDIVIDE": ("cuts",),
                 "MERGE_BY_DISTANCE": ("threshold",), "SET_THICKNESS": ("value",), "SET_OPACITY": ("value",),
                 "SET_CAPS": ("start", "end"), "CYCLICAL": ("value",), "SWITCH_DIRECTION": (), "JOIN": (),
                 "REPROJECT": ("surface", "offset"), "SEPARATE": (), "OUTLINE": ("radius", "offsetFactor")}


def _read_stroke(s, v3):
    if v3:
        return [(Vector(p.position), float(p.radius), float(p.opacity)) for p in s.points]
    return [(Vector(p.co), float(p.pressure), float(p.strength)) for p in s.points]


def _write_stroke(s, pts, v3):
    """Resize a stroke to len(pts) and write each point's position, size (v3 radius / legacy pressure) and opacity."""
    have = len(s.points)
    if v3:
        if len(pts) > have:
            s.add_points(len(pts) - have)
        elif len(pts) < have:
            s.remove_points(have - len(pts))
        for p, (co, size, op) in zip(s.points, pts):
            p.position, p.radius, p.opacity = co, size, op
    else:
        if len(pts) > have:
            s.points.add(len(pts) - have)
        while len(s.points) > len(pts):
            s.points.pop(index=len(s.points) - 1)
        for p, (co, size, op) in zip(s.points, pts):
            p.co, p.pressure, p.strength = co, size, op


def _rdp(pts, eps):
    """Ramer-Douglas-Peucker on (co, size, opacity) rows: the ends kept, a point dropped where the line misses it by
    no more than eps."""
    if len(pts) < 3:
        return list(pts)
    a, b = pts[0][0], pts[-1][0]
    ab = b - a
    worst, at = -1.0, 0
    for i in range(1, len(pts) - 1):
        p = pts[i][0]
        d = (p - a).length if ab.length < 1e-12 else (ab.cross(p - a)).length / ab.length
        if d > worst:
            worst, at = d, i
    if worst <= eps:
        return [pts[0], pts[-1]]
    return _rdp(pts[:at + 1], eps)[:-1] + _rdp(pts[at:], eps)


def _cyclic_of(s, v3):
    return bool(s.cyclic if v3 else s.use_cyclic)


def op_edit_grease_pencil_strokes(params):
    """Clean up Grease Pencil strokes on one keyframe: simplify, smooth, subdivide, merge by distance, a uniform thickness
    or opacity, caps, cyclic, direction, join, reproject onto a surface, separate (4.3+) or outline (5.0+).

    WHY (second 1.3 audit, 2026-10-25): the Grease Pencil ops drew, listed and moved strokes and edited none. Blender's
    edit-mode operators are MEASURED to change nothing on 3.6.23 in a background session (stroke_subdivide and
    stroke_simplify answered FINISHED and the point counts stayed 20 and 10), while the v3 ones work on 4.4 and 5.2 - so
    the geometry verbs are written on the stroke data here, the same on both API generations, and only SEPARATE
    (grease_pencil.separate, 4.3+) and OUTLINE (grease_pencil.outline, 5.0+) run Blender's operator.

    params:
      object, layer (required), frame (a keyframe; default the one shown at the scene's frame), strokes [indices]
      (default every stroke on the frame), action, settings:
        SIMPLIFY {distance: local units, default 0.01} - Ramer-Douglas-Peucker, the ends kept
        SMOOTH {iterations 1-100 (default 2), factor 0-1 (default 0.5)} - each inner point toward its neighbours
        SUBDIVIDE {cuts 1-10 (default 1)} - points added between each pair, size and opacity interpolated
        MERGE_BY_DISTANCE {threshold, default 0.001} - a point closer than that to the last kept one goes
        SET_THICKNESS {value} - v3: every point's radius (metres); legacy: the stroke's line_width (pixels)
        SET_OPACITY {value 0-1} - every point's opacity (legacy strength)
        SET_CAPS {start, end: ROUND | FLAT}
        CYCLICAL {value: true | false | 'TOGGLE'}
        SWITCH_DIRECTION - the point order reversed
        JOIN - the chosen strokes (two or more) appended into the first, in order; the rest removed
        REPROJECT {surface (mesh), offset (default 0)} - every point to the nearest point of the surface, offset along its
                  normal
        SEPARATE - the chosen strokes to a new object (4.3+)
        OUTLINE {radius, offsetFactor} - Blender's outline (5.0+)
    Reply: generation, per stroke the points before and after, and the frame's strokes and points before and after.
    """
    tail = _CHANGED
    reject_unknown(params, _EDIT_KEYS, "edit_grease_pencil_strokes")
    action = str(take(params, "action", required=True, kind=str)).upper()
    if action not in _EDIT_ACTIONS:
        raise MifOpError("action is one of %s, got %r. %s" % (", ".join(_EDIT_ACTIONS), action, tail))
    st = params.get("settings") or {}
    if not isinstance(st, dict) or set(st) - set(_EDIT_ACTIONS[action]):
        raise MifOpError("%s takes settings %s, got %r. %s" % (action, ", ".join(_EDIT_ACTIONS[action]) or "none", st, tail))
    obj = _gp_object(take(params, "object", required=True, kind=str), tail)
    v3 = _is_v3(obj)
    if action == "SEPARATE" and not v3:
        raise MifOpError("SEPARATE runs grease_pencil.separate, which is in the Grease Pencil 3 API (4.3 and later); this "
                         "is %s. %s" % (bpy.app.version_string, tail))
    if action == "OUTLINE" and not (v3 and hasattr(bpy.ops.grease_pencil, "outline") and bpy.app.version >= (5, 0)):
        raise MifOpError("OUTLINE runs grease_pencil.outline, which arrived in Blender 5.0; this is %s. %s"
                         % (bpy.app.version_string, tail))
    if action == "OUTLINE" and tuple(bpy.app.version[:2]) == (5, 0):
        # MEASURED 2026-10-26 by the certification: grease_pencil.outline here takes Blender 5.0.1 down (the session
        # exits); 5.2.1 runs it
        raise MifOpError("OUTLINE crashes Blender %s (measured on 5.0.1); 5.2 runs it. %s" % (bpy.app.version_string, tail))
    layer_name = take(params, "layer", required=True, kind=str)
    layer, _m = _find_layer(obj, layer_name, tail)
    have = _frames_by_number(layer)
    if params.get("frame") is not None:
        frame = take_int(params, "frame")
        if frame not in have:
            raise MifOpError("layer '%s' has no keyframe at %d (it has %s). %s" % (layer_name, frame, sorted(have), tail))
    else:
        now = int(bpy.context.scene.frame_current)
        shown = [n for n in have if n <= now]
        if not shown:
            raise MifOpError("layer '%s' shows no drawing at the scene's frame %d - pass frame. %s" % (layer_name, now, tail))
        frame = max(shown)
    fr = have[frame]
    holder = fr.drawing if v3 else fr
    strokes = list(holder.strokes)
    if params.get("strokes") is not None:
        idx = _frame_ints(params.get("strokes"), "strokes", tail)
        bad = [i for i in idx if not 0 <= i < len(strokes)]
        if bad:
            raise MifOpError("stroke index %s is outside 0-%d on frame %d. %s" % (bad, len(strokes) - 1, frame, tail))
    else:
        idx = list(range(len(strokes)))
    if not idx:
        raise MifOpError("frame %d of layer '%s' has no strokes. %s" % (frame, layer_name, tail))
    if action == "JOIN" and len(idx) < 2:
        raise MifOpError("JOIN needs two or more strokes. %s" % tail)
    before_frame = _frame_row(obj, fr)
    before = {i: len(strokes[i].points) for i in idx}
    out = {"object": obj.name, "generation": obj.type, "layer": layer_name, "frame": frame, "action": action,
           "strokes": idx}

    def num(key, default, lo, hi):
        v = finite_floats([st.get(key, default)], "settings." + key)[0]
        if not lo <= v <= hi:
            raise MifOpError("settings.%s is %g to %g, got %g. %s" % (key, lo, hi, v, tail))
        return v
    if action in ("SIMPLIFY", "SMOOTH", "SUBDIVIDE", "MERGE_BY_DISTANCE", "SET_OPACITY", "REPROJECT"):
        tree = mwi = None
        if action == "REPROJECT":
            from mathutils.bvhtree import BVHTree                  # noqa: PLC0415
            surf = get_object(st.get("surface") or "", want_mesh=True) if st.get("surface") else None
            if surf is None:
                raise MifOpError("REPROJECT needs settings.surface: the mesh to project onto. %s" % tail)
            offset = num("offset", 0.0, -100.0, 100.0)
            ev = surf.evaluated_get(bpy.context.evaluated_depsgraph_get())
            me = ev.to_mesh()
            try:
                smw = surf.matrix_world
                tree = BVHTree.FromPolygons([smw @ v.co for v in me.vertices], [tuple(p.vertices) for p in me.polygons])
            finally:
                ev.to_mesh_clear()
            mw, mwi = obj.matrix_world, obj.matrix_world.inverted_safe()
        dist = num("distance", 0.01, 0.0, 1e6) if action == "SIMPLIFY" else None
        its = int(num("iterations", 2, 1, 100)) if action == "SMOOTH" else None
        fac = num("factor", 0.5, 0.0, 1.0) if action == "SMOOTH" else None
        cuts = int(num("cuts", 1, 1, 10)) if action == "SUBDIVIDE" else None
        thr = num("threshold", 0.001, 0.0, 1e6) if action == "MERGE_BY_DISTANCE" else None
        opac = num("value", 1.0, 0.0, 1.0) if action == "SET_OPACITY" else None
        for i in idx:
            s = strokes[i]
            pts = _read_stroke(s, v3)
            if action == "SIMPLIFY":
                pts = _rdp(pts, dist)
            elif action == "SMOOTH":
                cyc = _cyclic_of(s, v3)
                for _ in range(its):
                    nxt = list(pts)
                    n = len(pts)
                    for k in range(n):
                        if not cyc and k in (0, n - 1):
                            continue
                        a, b = pts[(k - 1) % n][0], pts[(k + 1) % n][0]
                        nxt[k] = (pts[k][0] + ((a + b) / 2 - pts[k][0]) * fac, pts[k][1], pts[k][2])
                    pts = nxt
            elif action == "SUBDIVIDE":
                new = []
                for k in range(len(pts) - 1):
                    a, b = pts[k], pts[k + 1]
                    new.append(a)
                    for c in range(1, cuts + 1):
                        t = c / float(cuts + 1)
                        new.append((a[0].lerp(b[0], t), a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t))
                new.append(pts[-1])
                pts = new
            elif action == "MERGE_BY_DISTANCE":
                keep = [pts[0]]
                for p in pts[1:]:
                    if (p[0] - keep[-1][0]).length > thr:
                        keep.append(p)
                if len(keep) == 1 and len(pts) > 1:
                    keep.append(pts[-1])
                pts = keep
            elif action == "SET_OPACITY":
                pts = [(c, sz, opac) for c, sz, _o in pts]
            else:
                moved = []
                for c, sz, o in pts:
                    hit, n, _fi, _d = tree.find_nearest(mw @ c)
                    moved.append((mwi @ (hit + n * offset) if hit is not None else c, sz, o))
                pts = moved
            _write_stroke(s, pts, v3)
    elif action == "SET_THICKNESS":
        if "value" not in st:
            raise MifOpError("SET_THICKNESS needs settings.value (%s). %s" % ("metres, a point radius" if v3 else
                                                                               "pixels, the line width", tail))
        val = num("value", None, 0.0, 10000.0)
        for i in idx:
            s = strokes[i]
            if v3:
                for p in s.points:
                    p.radius = val
            else:
                s.line_width = int(round(val))
        out["thickness"] = {"value": val, "unit": "metres (point radius)" if v3 else "pixels (line_width)"}
    elif action == "SET_CAPS":
        want = {}
        for k in ("start", "end"):
            if k in st:
                c = str(st[k]).upper()
                if c not in ("ROUND", "FLAT"):
                    raise MifOpError("settings.%s is ROUND or FLAT, got %r. %s" % (k, st[k], tail))
                want[k] = c
        if not want:
            raise MifOpError("SET_CAPS needs settings.start and / or settings.end. %s" % tail)
        for i in idx:
            s = strokes[i]
            for k, c in want.items():
                if v3:
                    setattr(s, k + "_cap", 0 if c == "ROUND" else 1)
                else:
                    setattr(s, k + "_cap_mode", c)
    elif action == "CYCLICAL":
        val = st.get("value", "TOGGLE")
        if not (isinstance(val, bool) or val == "TOGGLE"):
            raise MifOpError("settings.value is true, false or 'TOGGLE', got %r. %s" % (val, tail))
        for i in idx:
            s = strokes[i]
            new = (not _cyclic_of(s, v3)) if val == "TOGGLE" else val
            if v3:
                s.cyclic = new
            else:
                s.use_cyclic = new
    elif action == "SWITCH_DIRECTION":
        for i in idx:
            pts = _read_stroke(strokes[i], v3)
            _write_stroke(strokes[i], pts[::-1], v3)
    elif action == "JOIN":
        first = strokes[idx[0]]
        pts = []
        for i in idx:
            pts += _read_stroke(strokes[i], v3)
        _write_stroke(first, pts, v3)
        drop = sorted(idx[1:], reverse=True)
        if v3:
            holder.remove_strokes(indices=drop)
        else:
            for i in drop:
                holder.strokes.remove(holder.strokes[i])
        out["joinedInto"] = idx[0]
    else:   # SEPARATE / OUTLINE - Blender's operators, on the chosen strokes selected
        from .ops_common import selection_restore, selection_snapshot   # noqa: PLC0415
        snap = selection_snapshot()
        before_objs = set(bpy.data.objects.keys())
        sc = bpy.context.scene
        frame0 = sc.frame_current
        try:
            sc.frame_set(frame)
            for k, s in enumerate(holder.strokes):
                s.select = k in idx
                for p in s.points:
                    p.select = k in idx
            for o in bpy.context.view_layer.objects:
                o.select_set(o == obj)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode="EDIT")
            if action == "SEPARATE":
                r = bpy.ops.grease_pencil.separate(mode="SELECTED")
            else:
                kw = {}
                if "radius" in st:
                    kw["radius"] = num("radius", None, 0.0, 1000.0)
                if "offsetFactor" in st:
                    kw["offset_factor"] = num("offsetFactor", None, -1.0, 1.0)
                r = bpy.ops.grease_pencil.outline(**kw)
        finally:
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                pass
            sc.frame_set(frame0)
            selection_restore(snap)
        if "FINISHED" not in r:
            raise MifOpError("grease_pencil.%s answered %s. %s" % (action.lower(), sorted(r), tail))
        out["made"] = sorted(set(bpy.data.objects.keys()) - before_objs)
        out["operator"] = "grease_pencil." + action.lower()
    if v3:
        holder.tag_positions_changed()
    obj.data.update_tag()
    now_strokes = list(holder.strokes)
    out["points"] = [{"stroke": i, "before": before[i], "after": len(now_strokes[i].points) if i < len(now_strokes)
                      else None} for i in idx]
    out["frameBefore"] = before_frame
    out["frameAfter"] = _frame_row(obj, have[frame] if frame in _frames_by_number(layer) else fr)
    return out


# ---------------------------------------------------------------------------------------- trace_image_to_strokes
_TRACE_KEYS = ("image", "file", "name", "threshold", "turnPolicy", "thickness", "resolution", "frame", "convertTo")
_TURN_V3 = {"BLACK": "FOREGROUND", "WHITE": "BACKGROUND"}
_TURN_LEGACY = {"FOREGROUND": "BLACK", "BACKGROUND": "WHITE"}


def op_trace_image_to_strokes(params):
    """Trace a black-and-white image - a logo, a sketch, a scan - into a new Grease Pencil object's strokes.

    WHY (second 1.3 audit, 2026-10-25): a sketch or logo made editable, then a mesh for a decal or an extrusion. The
    route is Blender's trace (gpencil.trace_image to 4.2, grease_pencil.trace_image from 4.3) on an image Empty; it is
    MEASURED to run headless on 3.6.23, 4.2.17, 4.4.0 and 5.2.1 - 4.2 has it although its UI does not list it.

    params:
      image (str) | file (str)  the image (one in the file, or a path loaded for the trace and removed after)
      name (str)                the new Grease Pencil object's name
      threshold (0-1, default 0.5), turnPolicy (FOREGROUND | BACKGROUND | LEFT | RIGHT | MINORITY | MAJORITY | RANDOM -
                                BLACK / WHITE, the pre-4.3 names, are accepted too), frame (the keyframe written, default
                                the scene's), and on 3.6 / 4.2 only thickness (pixels) and resolution - the 4.3+ trace
                                has neither and refuses them by name
      convertTo                 MESH: convert_object's conversion of the result, after the trace - 4.3 and later only
                                (on 3.6.23 object.convert adds no object for a Grease Pencil, and CURVE adds none on
                                5.2.1 either - measured), refused before the trace otherwise
    Reply: the object, its generation, strokes and points, and the conversion's result.
    """
    tail = "NOTHING was created."
    reject_unknown(params, _TRACE_KEYS, "trace_image_to_strokes")
    name = take(params, "image", kind=str)
    path = take(params, "file", kind=str)
    if (name is None) == (path is None):
        raise MifOpError("trace_image_to_strokes takes image (a name in this file) OR file (a path). %s" % tail)
    new_api = bpy.app.version >= (4, 3)
    for k in ("thickness", "resolution"):
        if params.get(k) is not None and new_api:
            raise MifOpError("%s is the pre-4.3 trace's; Blender %s's grease_pencil.trace_image has no %s. %s"
                             % (k, bpy.app.version_string, k, tail))
    conv = take(params, "convertTo", kind=str)
    if conv is not None and (conv.upper() != "MESH" or not new_api):
        raise MifOpError("convertTo takes MESH, from Blender 4.3 (this is %s): object.convert makes no object from a "
                         "Grease Pencil before that, and none as a CURVE on any build measured. %s"
                         % (bpy.app.version_string, tail))
    thr = finite_floats([params.get("threshold", 0.5)], "threshold")[0]
    if not 0.0 <= thr <= 1.0:
        raise MifOpError("threshold is 0-1. %s" % tail)
    pol = str(params.get("turnPolicy", "MINORITY")).upper()
    pol = (_TURN_V3 if new_api else _TURN_LEGACY).get(pol, pol)
    valid = ("FOREGROUND", "BACKGROUND") if new_api else ("BLACK", "WHITE")
    if pol not in valid + ("LEFT", "RIGHT", "MINORITY", "MAJORITY", "RANDOM"):
        raise MifOpError("turnPolicy is FOREGROUND, BACKGROUND, LEFT, RIGHT, MINORITY, MAJORITY or RANDOM, got %r. %s"
                         % (params.get("turnPolicy"), tail))
    # EVERY VALUE BEFORE THE FIRST WRITE (2026-10-28, found reading and measured by the version matrix): frame was read
    # after the image was loaded and the source empty linked, so a bad frame refused and left `_mif_trace_source` and
    # the loaded image in the file; thickness / resolution went through take_int inside the operator call's try; and
    # name went through str(), so a mapping or a list named the object "{'mif': ...}".
    frame = take_int(params, "frame", default=int(bpy.context.scene.frame_current))
    legacy_kw = {}
    if not new_api:
        for k in ("thickness", "resolution"):
            if params.get(k) is not None:
                legacy_kw[k] = take_int(params, k)
    new_name = take(params, "name", kind=str)
    loaded = None
    if path is not None:
        import os                                                    # noqa: PLC0415
        full = os.path.abspath(bpy.path.abspath(path))
        if not os.path.isfile(full):
            raise MifOpError("no file '%s'. %s" % (full, tail))
        img = loaded = bpy.data.images.load(full, check_existing=False)
    else:
        img = bpy.data.images.get(name)
        if img is None:
            raise MifOpError("no image named %r. %s" % (name, tail))
    from .ops_common import selection_restore, selection_snapshot       # noqa: PLC0415
    snap = selection_snapshot()
    emp = bpy.data.objects.new("_mif_trace_source", None)
    emp.empty_display_type = "IMAGE"
    emp.data = img
    bpy.context.scene.collection.objects.link(emp)
    before = set(bpy.data.objects.keys())
    try:
        for o in bpy.context.view_layer.objects:
            o.select_set(o == emp)
        bpy.context.view_layer.objects.active = emp
        kw = {"target": "NEW", "threshold": thr, "turnpolicy": pol, "mode": "SINGLE", "use_current_frame": False,
              "frame_number": frame}
        kw.update(legacy_kw)
        op = bpy.ops.grease_pencil.trace_image if new_api else bpy.ops.gpencil.trace_image
        r = op(**kw)
    finally:
        remove_objects([emp])
        if loaded is not None and not loaded.users:
            bpy.data.images.remove(loaded)
        selection_restore(snap)
    made = sorted((o for o in bpy.data.objects if o.name not in before and o.type in ("GPENCIL", "GREASEPENCIL")),
                  key=lambda o: o.name)
    if "FINISHED" not in r or not made:
        n_made = len(made)
        remove_objects(made)
        raise MifOpError("the trace answered %s and made %d Grease Pencil object(s) - a blank or all-black image traces "
                         "to nothing. Nothing was kept." % (sorted(r), n_made))
    gp = made[0]
    if new_name:
        gp.name = new_name
    summary = gp_summary(gp)
    strokes = sum(f["strokes"] for lay in summary["layers"] for f in lay["frames"])
    points = sum(f["points"] for lay in summary["layers"] for f in lay["frames"])
    out = {"object": gp.name, "generation": gp.type, "strokes": strokes, "points": points,
           "operator": "grease_pencil.trace_image" if new_api else "gpencil.trace_image", "layers": summary["layers"]}
    if conv:
        from .ops_create import op_convert_object                       # noqa: PLC0415
        out["converted"] = op_convert_object({"object": gp.name, "target": conv.upper(), "keepOriginal": True})
    return out


# ---------------------------------------------------------------------------------------- bake_grease_pencil
_BAKE_KEYS = ("objects", "frameStart", "frameEnd", "step", "frameTarget", "project", "name")


def op_bake_grease_pencil(params):
    """Bake Grease Pencil objects' modifiers and animation - Line Art included - into a new object of plain strokes, one
    keyframe per baked frame.

    WHY (second 1.3 audit, 2026-10-25): Line Art's lines can only be edited, exported frame by frame or converted once
    baked. On 3.6 and 4.2 this is Blender's gpencil.bake_grease_pencil_animation, which polls for a 3D View and so runs
    inside one (ops_common.editor_override) - MEASURED on 3.6.23: four keyframes from an object keyed over 1-4. The 4.3+
    operator, grease_pencil.bake_grease_pencil_animation, answered FINISHED in the same set-up on 5.2.1 and made an object
    with NO layers, so from 4.3 the bake is written here: at each frame the object's EVALUATED drawings (modifiers applied
    - read through Object.evaluated_get, measured on 4.4.0 and 5.2.1) are copied, in world space, into a new object's
    keyframes. That route bakes project KEEP only; FRONT, SIDE, TOP, VIEW and CURSOR are refused there.

    params: objects [Grease Pencil objects] (required), frameStart / frameEnd (default the scene's), step (default 1),
    frameTarget (the first frame written, default frameStart), project KEEP (default) | FRONT | SIDE | TOP | VIEW |
    CURSOR, name (the baked object's). Reply: the new object, its keyframes, strokes and points.
    """
    tail = "NOTHING was created."
    reject_unknown(params, _BAKE_KEYS, "bake_grease_pencil")
    names = params.get("objects")
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, list) or not names:
        raise MifOpError("objects is a list of Grease Pencil objects to bake. %s" % tail)
    objs = [_gp_object(n, tail) for n in names]
    sc = bpy.context.scene
    fs = take_int(params, "frameStart", default=sc.frame_start)
    fe = take_int(params, "frameEnd", default=sc.frame_end)
    step = take_int(params, "step", default=1)
    if fe < fs or step < 1:
        raise MifOpError("frameEnd must not be before frameStart, and step is 1 or more. %s" % tail)
    target = take_int(params, "frameTarget", default=fs)
    # A STRING, checked here for both routes (2026-10-28, the version matrix): to 4.2 str() named the baked object
    # "{'mif': ...}" from a mapping, and from 4.3 the mapping reached greasepencils.new and raised a raw TypeError.
    new_name = take(params, "name", kind=str)
    proj = str(params.get("project", "KEEP")).upper()
    renamed = bpy.app.version >= (4, 3)
    mod = bpy.ops.grease_pencil if renamed else bpy.ops.gpencil
    opname = ("grease_pencil" if renamed else "gpencil") + ".bake_grease_pencil_animation"
    projs = [i.identifier for i in mod.bake_grease_pencil_animation.get_rna_type().properties["project_type"].enum_items]
    if proj not in projs:
        raise MifOpError("project is one of %s, got %r. %s" % (", ".join(projs), proj, tail))
    if renamed:
        if proj != "KEEP":
            raise MifOpError("from 4.3 the bake is written by this op (Blender's own makes an empty object - see the "
                             "docstring), and it keeps the strokes where they are: project %s is refused. %s" % (proj, tail))
        return _bake_v3(objs, fs, fe, step, target, new_name, opname)
    from .ops_common import editor_override, selection_restore, selection_snapshot   # noqa: PLC0415
    snap = selection_snapshot()
    before = set(bpy.data.objects.keys())
    f0 = sc.frame_current
    try:
        for o in bpy.context.view_layer.objects:
            o.select_set(o in objs)
        bpy.context.view_layer.objects.active = objs[0]
        # THE OPERATOR POLLS FOR A 3D VIEW (measured on 5.2.1: 'context is incorrect' with none), so it runs in the
        # window's first area turned into one for the call (ops_common.editor_override), as the sequencer ops do.
        with editor_override("VIEW_3D", sc):
            r = mod.bake_grease_pencil_animation(frame_start=fs, frame_end=fe, step=step, only_selected=True,
                                                 frame_target=target, project_type=proj)
    finally:
        sc.frame_set(f0)
        selection_restore(snap)
    made = [bpy.data.objects[n] for n in sorted(set(bpy.data.objects.keys()) - before)]
    if "FINISHED" not in r or not made:
        raise MifOpError("%s answered %s and made %d object(s). Nothing was kept." % (opname, sorted(r), len(made)))
    gp = made[0]
    summary = gp_summary(gp)
    if not any(f["strokes"] for lay in summary["layers"] for f in lay["frames"]):
        # THE PRE-4.3 OPERATOR BAKES MOTION, NOT DRAWINGS. Measured 2026-10-28 on 3.6.23 and 4.2.17 (found by the version
        # matrix): an object whose only animation is its drawing keyframes bakes to an object with no layers, frames 1-2
        # and 1-6 alike, and the op answered ok with nothing in it; keyed over 1-4 by its location the same object bakes
        # two strokes. The 4.3+ route copies the drawings themselves.
        datas = [o.data for o in made]
        remove_objects(made)
        for d in datas:
            if d is not None and not d.users:
                try:
                    bpy.data.grease_pencils.remove(d)
                except (ReferenceError, RuntimeError, TypeError):
                    pass
        raise MifOpError("%s made an object with no strokes: before 4.3 Blender's bake reads an object's own animation "
                         "(its transform or its modifiers over the range), and %s %s none over %d-%d - its drawing "
                         "keyframes alone bake to nothing on %s. Nothing was kept." % (
                             opname, ", ".join(o.name for o in objs), "has" if len(objs) == 1 else "have", fs, fe,
                             bpy.app.version_string))
    if new_name:
        gp.name = new_name
        summary = gp_summary(gp)
    frames = sorted({f["frame"] for lay in summary["layers"] for f in lay["frames"]})
    return {"object": gp.name, "generation": gp.type, "operator": opname, "sources": [o.name for o in objs],
            "range": [fs, fe], "step": step, "keyframes": frames,
            "strokes": sum(f["strokes"] for lay in summary["layers"] for f in lay["frames"]),
            "points": sum(f["points"] for lay in summary["layers"] for f in lay["frames"]), "layers": summary["layers"]}


def _bake_v3(objs, fs, fe, step, target, name, opname):
    """bake_grease_pencil from 4.3: each frame's evaluated drawings copied into a new object's keyframes, world space."""
    sc = bpy.context.scene
    coll, _gen = _v3_collection()
    data = coll.new(name or "GreasePencil")
    new = bpy.data.objects.new(data.name, data)
    sc.collection.objects.link(new)
    for o in objs:
        for m in o.data.materials:
            if m is not None and m.name not in [x.name for x in data.materials if x is not None]:
                data.materials.append(m)
    mat_index = {m.name: i for i, m in enumerate(data.materials) if m is not None}
    f0 = sc.frame_current
    layers = {}
    try:
        for f in range(fs, fe + 1, step):
            sc.frame_set(f)
            deps = bpy.context.evaluated_depsgraph_get()
            for o in objs:
                ev = o.evaluated_get(deps)
                mw = ev.matrix_world
                src_mats = [m.name if m is not None else None for m in o.data.materials]
                for lay in ev.data.layers:
                    cf = lay.current_frame()
                    if cf is None or cf.drawing is None or not len(cf.drawing.strokes):
                        continue
                    key = "%s_%s" % (o.name, lay.name)
                    dst = layers.get(key)
                    if dst is None:
                        dst = layers[key] = data.layers.new(key)
                    dfr = dst.frames.new(target + (f - fs))
                    strokes = list(cf.drawing.strokes)
                    dfr.drawing.add_strokes([len(s.points) for s in strokes])
                    for s, d in zip(strokes, dfr.drawing.strokes):
                        for p, q in zip(s.points, d.points):
                            q.position = mw @ Vector(p.position)
                            q.radius = p.radius
                            q.opacity = p.opacity
                        d.cyclic = s.cyclic
                        mn = src_mats[s.material_index] if 0 <= s.material_index < len(src_mats) else None
                        d.material_index = mat_index.get(mn, 0)
                    dfr.drawing.tag_positions_changed()
    finally:
        sc.frame_set(f0)
    if not layers:
        name_ = new.name
        remove_objects([new])
        coll.remove(data)
        raise MifOpError("no frame of %s from %d to %d has a drawing to bake. '%s' was removed: NOTHING was created."
                         % (", ".join(o.name for o in objs), fs, fe, name_))
    summary = gp_summary(new)
    frames = sorted({fr["frame"] for lay in summary["layers"] for fr in lay["frames"]})
    return {"object": new.name, "generation": new.type, "operator": None,
            "route": "evaluated drawings copied per frame (%s makes an empty object here)" % opname,
            "sources": [o.name for o in objs], "range": [fs, fe], "step": step, "keyframes": frames,
            "strokes": sum(fr["strokes"] for lay in summary["layers"] for fr in lay["frames"]),
            "points": sum(fr["points"] for lay in summary["layers"] for fr in lay["frames"]), "layers": summary["layers"]}


OPS = {
    "edit_grease_pencil_strokes": op_edit_grease_pencil_strokes,
    "trace_image_to_strokes": op_trace_image_to_strokes,
    "bake_grease_pencil": op_bake_grease_pencil,
    "create_grease_pencil": op_create_grease_pencil,
    "draw_grease_pencil_strokes": op_draw_grease_pencil_strokes,
    "set_grease_pencil_layers": op_set_grease_pencil_layers,
    "set_grease_pencil_material": op_set_grease_pencil_material,
    "set_grease_pencil_effects": op_set_grease_pencil_effects,
    "set_grease_pencil_frames": op_set_grease_pencil_frames,
    "transform_grease_pencil_points": op_transform_grease_pencil_points,
}
