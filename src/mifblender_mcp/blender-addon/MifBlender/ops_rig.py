"""MifBlender ops: armatures, shape keys, vertex groups - the character-rigging data the
addon had ZERO ops for before this file.

WHY THIS EXISTS. object_info() (ops_common.py) reports transform, bounds, materials and UVs for
a MESH, and NOTHING for an ARMATURE object beyond its bare transform - `if obj.type != "MESH":
return info` skips straight past it. Shape keys and vertex groups are absent even for a mesh.
That is a real gap for a game-asset pipeline: character-driven games are the common case, the UE side of
this bridge can already read a skeleton's bones, virtual bones and morph targets
(MifBridgeSkeleton.cpp, list_bones/list_virtual_bones/list_morph_targets), and until now nothing
on the BLENDER side - where a rigger actually AUTHORS that data before export - could read any
of it back.

NAMED TO MATCH THE UE SIDE ON PURPOSE. list_bones here reads an Armature's REST pose the same
way UE's list_bones reads a Skeleton's ReferenceSkeleton; list_shape_keys is Blender's name for
what UE calls morph targets, cross-referenced in both docstrings so a caller working across the
pipeline recognises the pairing instead of having to know that "shape key" and "morph target"
are the same thing under two names.

READ-ONLY WHEN THIS FILE WAS WRITTEN, AND NO LONGER - normalize_weights and
transfer_weights were added 2026-08-30 as its write half, for the reason this paragraph
itself makes: the reads could report a problem nobody could act on. The four list_* ops
below are still read-only. The original note, kept because the reasoning behind the reads
still stands: nothing here creates a bone, key or group, or changes scene data -
matching this addon's existing op_object_info / list_objects, and the read-before-you-can-fix-it
shape UE's own list_bones/list_morph_targets already established.

VERIFIED against a live Blender 4.4 instance before being trusted, not assumed from API docs -
see the commit this file was added in for the exact live-call transcript.
"""

from __future__ import annotations

import array
import hashlib
import math

import bpy
from mathutils import Vector

import bmesh

from .ops_common import (MifOpError, simulation_guard, check_axis_dict, exclude_rows, finite_float, finite_floats,
                         finite_int, get_object, limit_rows, mesh_counts, refuse_override_data, require_editable,
                         reject_unknown, rnd, select_only, selection_restore, selection_snapshot, shared_data_note,
                         rna_properties, take, take_bool, take_float, take_int, write_rna_property, drop_doubled_and_fin_faces,
                         open_collapsed_uvs_of, project_flat_uvs)


def _bone_dict(bone):
    return {
        "name": bone.name,
        "parent": bone.parent.name if bone.parent else None,
        # ARMATURE-SPACE, not parent-relative - head_local/tail_local, not head/tail, which are in
        # the bone's OWN local space and would need every ancestor's transform composed to mean
        # anything on their own. Matches UE list_bones reporting refPose as parent-relative but
        # DOCUMENTING it as such; here the armature-space form is the one riggers actually compare
        # against a reference skeleton, so it is the default rather than an opt-in.
        "headArmatureSpaceBU": rnd(bone.head_local),
        "tailArmatureSpaceBU": rnd(bone.tail_local),
        "length": round(float(bone.length), 6),
        "useDeform": bool(bone.use_deform),
        # WHETHER THE HEAD IS WELDED TO THE PARENT'S TAIL - create_armature and add_bones write it (`connect`)
        # and nothing read it back, so a connect that moved a head could only be inferred from coordinates.
        "connected": bool(bone.use_connect),
        "isRoot": bone.parent is None,
        "childCount": len(bone.children),
        # ROLL (2026-10-06): radians, as add_bones takes it, and degrees - from matrix_local, so a bone set read here
        # and written back through add_bones keeps its twist. Nothing read it before.
        "roll": round(_bone_roll(bone), 6),
        "rollDeg": round(math.degrees(_bone_roll(bone)), 4),
    }


def _bone_roll(bone):
    """A bone's edit-mode roll in radians, from its rest matrix (Blender's own Bone.AxisRollFromMatrix)."""
    return float(bpy.types.Bone.AxisRollFromMatrix(bone.matrix_local.to_3x3())[1])


# Blender truncates ID and group names here on 3.6 through 4.4. Measured on all four builds:
# a 100-character vertex group name comes back at 63 characters on every one of them.
_ID_NAME_LIMIT = 63


def op_list_bones(params):
    """The rest-pose bone hierarchy of an Armature object - Blender's name for what UE calls a
    Skeleton's ReferenceSkeleton."""
    reject_unknown(params, {"object", "name", "nameContains", "nameExcludes", "limit", "posed"}, "list_bones")
    obj = get_object(take(params, "object", "name", required=True, kind=str))
    if obj.type != "ARMATURE":
        raise MifOpError("object '%s' is a %s, not an ARMATURE. list_bones reads bone rest pose; "
                         "list_objects with type:'ARMATURE' finds one if you are not sure of the "
                         "name." % (obj.name, obj.type))
    # POSED (1.4, 2026-11-07): the rest pose alone meant a pose could be checked only by rendering it.
    posed = take_bool(params, "posed", default=False)

    filt = take(params, "nameContains", default=None)
    bones = obj.data.bones
    rows = []
    for bone in bones:
        if filt and str(filt) not in bone.name:
            continue
        rows.append(_bone_dict(bone))

    # nameExcludes BEFORE the count, so `count` is the number that MATCHED the request as asked - the
    # same order every other bounded listing uses. It was missing here while the MCP wrapper already
    # offered name_excludes, so through MCP the call came back "list_bones received unknown
    # parameter(s): nameExcludes" - found by parity_check 2026-09-14, which reads the wrapper's payload
    # against the handler's accepted set. Rigs are the case that wants it most: a control rig's DEF-
    # and MCH- bones are most of the list and almost never what is being asked about.
    # EACH BONE'S COLLECTIONS (layer indices on 3.6) AND RIGIFY TYPE (2026-10-04) - neither was read anywhere.
    from .ops_rigify import rigify_bone_fields
    rigify_bone_fields(obj, rows)
    rows = exclude_rows(rows, params, name_of=lambda r: r.get("name", ""))
    matched = len(rows)
    rows, extra = limit_rows(rows, params, "bone(s)")
    if posed:
        extra["posedNote"] = posed_bone_fields(obj, rows)
    return dict({
        "armature": obj.name,
        "boneCount": len(bones),
        "count": matched,
        "bones": rows,
    }, **extra)


def _mat_rows(m):
    return [[round(float(m[i][j]), 6) for j in range(4)] for i in range(4)]


def posed_bone_fields(obj, rows):
    """Add each row's `posed` block - head, tail and matrix as the bone stands now, read through the depsgraph (every
    constraint, driver and action applied), in world space and armature space - beside the rest pose the row already
    holds. tailMovedFromRest is world units between the posed and rest tail."""
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    mw = ev.matrix_world
    for row in rows:
        epb = ev.pose.bones.get(row["name"]) if ev.pose is not None else None
        bone = obj.data.bones.get(row["name"])
        if epb is None or bone is None:
            row["posed"] = None
            continue
        head_w, tail_w = mw @ epb.head, mw @ epb.tail
        rest_t = obj.matrix_world @ bone.tail_local
        row["posed"] = {"headWorld": rnd(list(head_w)), "tailWorld": rnd(list(tail_w)),
                        "headArmature": rnd(list(epb.head)), "tailArmature": rnd(list(epb.tail)),
                        "matrixArmature": _mat_rows(epb.matrix), "matrixWorld": _mat_rows(mw @ epb.matrix),
                        "restHeadWorld": rnd(list(obj.matrix_world @ bone.head_local)), "restTailWorld": rnd(list(rest_t)),
                        "tailMovedFromRest": round(float((tail_w - rest_t).length), 6)}
    return ("posed is read through the depsgraph at frame %d: where each bone really is, constraints and actions "
            "applied; matrixArmature is the pose bone's armature-space matrix, matrixWorld the same times the "
            "armature's world matrix" % bpy.context.scene.frame_current)


def op_list_shape_keys(params):
    """Shape keys on a mesh object - Blender's name for what UE calls morph targets. Names,
    value, slider range, mute state and which key each is relative to."""
    reject_unknown(params, {"object", "name", "limit", "nameExcludes"}, "list_shape_keys")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    keys = obj.data.shape_keys
    if keys is None:
        # A real, common state - most game meshes have none - not a failure. Matches UE
        # list_morph_targets' own "this mesh has no morph targets" note rather than an error.
        return {"object": obj.name, "hasShapeKeys": False, "count": 0, "shapeKeys": [],
               "note": "this mesh has no shape keys. That is normal for most meshes - only ones "
                       "authored for facial or blend-shape animation need them."}

    blocks = keys.key_blocks
    rows = []
    for kb in blocks:
        rows.append({
            "name": kb.name,
            "value": round(float(kb.value), 6),
            "sliderMin": round(float(kb.slider_min), 6),
            "sliderMax": round(float(kb.slider_max), 6),
            "mute": bool(kb.mute),
            # The BASIS key (index 0) is relative to itself and drives nothing - reported so a
            # caller can tell "the neutral pose" from "an actual pose target" without guessing
            # from index 0 by convention.
            "isBasis": kb.name == blocks[0].name,
            "relativeTo": kb.relative_key.name if kb.relative_key else None,
        })

    # nameExcludes BEFORE the count, so `count` is the number that MATCHED the request as asked. A rig's
    # corrective shape keys are mostly DEF-, MCH- and ORG- families, and a limit alone hands a caller twenty rows of the
    # family they were trying to look past.
    rows = exclude_rows(rows, params)
    matched = len(rows)
    rows, extra = limit_rows(rows, params, "shape key(s)")
    return dict({
        "object": obj.name,
        "hasShapeKeys": True,
        "count": matched,
        "shapeKeys": rows,
    }, **extra)


def _group_weight_scan(obj):
    """{group index: [assigned, weighted, min, max, total]} in ONE pass over the mesh's vertices - O(verts) rather than
    O(verts * groups), which matters once a character mesh with 30+ groups reaches tens of thousands of vertices.
    assigned counts every member; weighted, min, max and total only the vertices whose weight is above zero.
    list_vertex_groups and transfer_weights both read it."""
    acc = {}
    for v in obj.data.vertices:
        for ge in v.groups:
            entry = acc.get(ge.group)
            if entry is None:
                entry = acc[ge.group] = [0, 0, None, None, 0.0]
            entry[0] += 1
            weight = ge.weight
            if weight > 0.0:
                entry[1] += 1
                entry[2] = weight if entry[2] is None else min(entry[2], weight)
                entry[3] = weight if entry[3] is None else max(entry[3], weight)
                entry[4] += weight
    return acc


def _weight_stats(entry):
    """One group's scan entry as {weightedVerts, min, max, mean, total} - None-safe for a group nothing is in."""
    weighted = entry[1] if entry else 0
    return {"weightedVerts": weighted,
            "min": round(entry[2], 6) if weighted else None,
            "max": round(entry[3], 6) if weighted else None,
            "mean": round(entry[4] / weighted, 6) if weighted else None,
            "total": round(entry[4], 6) if entry else 0.0}


def op_list_vertex_groups(params):
    """Vertex groups on a mesh object - the bone-weight assignment groups a skinned mesh needs
    one per deforming bone. Reports which groups actually have any vertex assigned, since a
    group with ZERO weighted vertices is a rig that will not deform on that bone at all - the
    same class of bug UE's analyze_skeletal_split flags with `influencesGeometry:false`.

    REGION (2026-10-05): region {box | sphere} in rest space (the base mesh under the object's transform) adds each
    group's regionMean - its mean weight over every vertex inside, an unassigned vertex counting 0 - and regionWeighted,
    and the reply's regionDominant lists the groups by that mean, largest first. It found a hood's back going straight
    from spine_04 to head with almost no spine_05 or neck between. A region holding no vertex is refused."""
    reject_unknown(params, {"object", "name", "limit", "nameExcludes", "region"}, "list_vertex_groups")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    from .ops_garment import parse_region    # noqa: PLC0415 - one region grammar for the garment measurements
    if isinstance(params.get("region"), dict) and "vertexGroup" in params["region"]:
        raise MifOpError("region here is a box or a sphere - a vertex group is what this lists. NOTHING was read.")
    region, region_desc = parse_region(params)
    groups = obj.vertex_groups
    if not groups:
        return {"object": obj.name, "count": 0, "vertexGroups": [],
               "note": "this mesh has no vertex groups. If it is meant to be skinned to an "
                       "armature, this mesh cannot deform on any bone yet."}

    # ONE PASS over the mesh rather than one pass PER GROUP - see _group_weight_scan.
    #
    # THE WEIGHTS, NOT ONLY HOW MANY. A count said a group influenced geometry and nothing said with what: the second
    # seat's report 87 found no op that could read a weight value back, so a group weighted 0.01 everywhere read the
    # same as one weighted 1.0. min / max / mean over the vertices that carry weight - the vertices counted by
    # weightedVertexCount - and assignedVertexCount for every member, zero weights included. vertex_info reads one
    # group's weight per vertex.
    scan = _group_weight_scan(obj)
    inside, sums, hits = None, {}, {}
    if region is not None:
        mw = obj.matrix_world
        inside = [v for v in obj.data.vertices if region(mw @ v.co, v.index, obj)]
        if not inside:
            raise MifOpError("no vertex of '%s' is in the region %s (rest space) - a measurement of nothing is not "
                             "clean. NOTHING was read." % (obj.name, region_desc))
        for v in inside:
            for ge in v.groups:
                sums[ge.group] = sums.get(ge.group, 0.0) + ge.weight
                if ge.weight > 0:
                    hits[ge.group] = hits.get(ge.group, 0) + 1
    rows = []
    for g in groups:
        entry = scan.get(g.index)
        stats = _weight_stats(entry)
        row = {"name": g.name, "index": g.index, "weightedVertexCount": stats["weightedVerts"],
               "influencesGeometry": stats["weightedVerts"] > 0,
               "assignedVertexCount": entry[0] if entry else 0,
               "weightMin": stats["min"], "weightMax": stats["max"], "weightMean": stats["mean"]}
        if inside is not None:
            row["regionMean"] = round(sums.get(g.index, 0.0) / len(inside), 6)
            row["regionWeighted"] = hits.get(g.index, 0)
        rows.append(row)
    region_reply = {}
    if inside is not None:
        dominant = sorted((r for r in rows if r["regionMean"] > 0), key=lambda r: r["regionMean"], reverse=True)
        region_reply = {"region": region_desc, "regionVertices": len(inside),
                        "regionDominant": [{"name": r["name"], "mean": r["regionMean"]} for r in dominant[:20]]}

    # nameExcludes BEFORE the count, so `count` is the number that MATCHED the request as asked. A rig's
    # vertex groups are mostly DEF-, MCH- and ORG- families, and a limit alone hands a caller twenty rows of the
    # family they were trying to look past.
    rows = exclude_rows(rows, params)
    matched = len(rows)
    rows, extra = limit_rows(rows, params, "vertex group(s)")
    return dict({
        "object": obj.name,
        "count": matched,
        "vertexGroups": rows,
    }, **extra, **region_reply)


# A handful of settings PER TYPE that actually decide what export produces, not every property
# Blender exposes for that type - e.g. Mirror's use_axis/merge_threshold change the resulting
# geometry, its vertex_group/bisect settings are refinements on top. Curated rather than
# exhaustive on purpose: Blender ships 100+ modifier types, and hand-describing all of them for
# one addon nobody has asked to author modifiers through (only READ what is already there) would
# be effort spent on the wrong problem. A type not listed here still reports name/type/visibility
# - never silently dropped, just without a decoded settings dict.
_MODIFIER_FIELDS = {
    "ARMATURE": (("object", lambda m: m.object.name if m.object else None),),
    "MIRROR": (("axisX", lambda m: bool(m.use_axis[0])), ("axisY", lambda m: bool(m.use_axis[1])),
              ("axisZ", lambda m: bool(m.use_axis[2])),
              ("mergeThreshold", lambda m: round(float(m.merge_threshold), 6))),
    "SOLIDIFY": (("thickness", lambda m: round(float(m.thickness), 6)),),
    "BEVEL": (("width", lambda m: round(float(m.width), 6)), ("segments", lambda m: int(m.segments))),
    "SUBSURF": (("levels", lambda m: int(m.levels)),
               ("renderLevels", lambda m: int(m.render_levels))),
    "DECIMATE": (("decimateType", lambda m: str(m.decimate_type)),
                ("ratio", lambda m: round(float(m.ratio), 6))),
    "TRIANGULATE": (("quadMethod", lambda m: str(m.quad_method)),),
}


def _read_modifier_attr(mod, attr):
    """One modifier property as something JSON can carry, whatever kind it is."""
    value = getattr(mod, attr)
    if value is None:
        return None
    if hasattr(value, "name") and not isinstance(value, str):
        return value.name                       # a datablock pointer - report what it points AT
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(float(value), 6)
    if isinstance(value, int):
        return int(value)
    if hasattr(value, "__len__") and not isinstance(value, str):
        return [round(float(v), 6) for v in value]
    return str(value)


def _generated_fields(mod_type):
    """The read side, DERIVED from the write table rather than typed out again.

    WHY THIS IS GENERATED. _MODIFIER_FIELDS was a second hand-written table listing the same seven
    types _MODIFIER_WRITES did, and the moment fourteen types were added to the write side the two
    disagreed - every new setting was writable and unreadable, which is precisely the read/write
    asymmetry this repo treats as a defect class. Two tables that must agree will not, so there is
    one table and the other is computed from it.

    Only entries whose target is a plain RNA name are generated; the handful with callable setters
    (MIRROR's use_axis indices) keep their hand-written getters in _MODIFIER_FIELDS, which wins.
    """
    out = []
    for key, (target, _coerce) in sorted(_MODIFIER_WRITES.get(mod_type, {}).items()):
        if isinstance(target, str):
            out.append((key, lambda m, a=target: _read_modifier_attr(m, a)))
    return tuple(out)


def _modifier_dict(mod):
    row = {
        "name": mod.name,
        "type": mod.type,
        "showViewport": bool(mod.show_viewport),
        "showRender": bool(mod.show_render),
    }
    fields = _MODIFIER_FIELDS.get(mod.type) or _generated_fields(mod.type)
    if fields:
        settings = {}
        for key, getter in fields:
            try:
                settings[key] = getter(mod)
            except Exception:  # noqa: BLE001 - a field this addon's own curation got wrong for
                                # some modifier sub-state must not take the whole row down with it
                settings[key] = None
        row["settings"] = settings

    # NODES IS NOT IN EITHER TABLE AND CANNOT BE, because its settings are not RNA properties -
    # they are per-group inputs addressed by IDENTIFIER (Socket_2) inside the modifier itself.
    #
    # Until this was added, list_modifiers reported a geometry-nodes modifier as a bare name and
    # type: which GROUP it held was invisible, so two assign_node_group calls stacking two
    # modifiers were indistinguishable from one, and there was no way to see what any of their
    # inputs had been set to. assign_node_group could write them and nothing could read them back.
    if mod.type == "NODES":
        tree = getattr(mod, "node_group", None)
        row["group"] = tree.name if tree is not None else None
        inputs = {}
        if tree is not None:
            if hasattr(tree, "interface"):
                items = [it for it in tree.interface.items_tree
                         if getattr(it, "item_type", "SOCKET") == "SOCKET" and it.in_out == "INPUT"]
            else:
                items = list(tree.inputs)
            for it in items:
                # BLENDER 5.2 MOVED THESE OUT FROM UNDER THE MODIFIER. Up to 5.0 an input was a flat
                # id-property (`mod["Socket_2"]`); on 5.2 the NodesModifier supports no id-properties
                # at all and the value lives in mod.properties.inputs[identifier]["value"].
                #
                # Without this, every socket on 5.2 read back None while still being NAMED - so this
                # op, whose own comment above says "assign_node_group could write them and nothing
                # could read them back", had quietly gone back to not reading them back. ops_nodes
                # carries the five-build measurement and is imported rather than copied; ops_anim got
                # the same treatment the same day. Three call sites, one table.
                group = _modifier_socket_group(mod, it.identifier)
                if group is not None:
                    held = group.get(_MOD_SOCKET_VALUE_KEY)
                    if held is None:
                        inputs[it.name] = None
                        continue
                else:
                    try:
                        held = mod[it.identifier]
                    except (KeyError, TypeError):
                        inputs[it.name] = None
                        continue
                inputs[it.name] = (held.name if hasattr(held, "name") and not isinstance(held, str)
                                   else (round(float(held), 6) if isinstance(held, float)
                                         else (list(held) if hasattr(held, "__len__")
                                               and not isinstance(held, str) else held)))
        row["inputs"] = inputs
        row["outputConnected"] = (
            any(l.to_node.bl_idname == "NodeGroupOutput" for l in tree.links)
            if tree is not None else None)
    return row


# The 5.2 socket home, measured across all five builds in ops_nodes. Imported, never copied.
from .ops_nodes import _modifier_socket_group, _MOD_SOCKET_VALUE_KEY  # noqa: E402

def op_list_modifiers(params):
    """The modifier stack on an object - a mesh, curve, text, surface or lattice - in EVALUATION ORDER (top
    to bottom in the Modifier Properties panel, which is also the order they apply in). Answers "what will
    export_mesh actually produce" before spending an export on finding out - a Mirror or Subsurf still in the
    stack changes the exported geometry; a disabled one (showViewport/showRender both false)
    does not, and both states are reported rather than only "is it there".

    `rna: true` adds each modifier's EVERY readable RNA property by identifier (ops_common.rna_properties) - the
    read side of what add_modifier and set_modifier write generically. `settings` stays the curated camelCase
    table, which is all this op could read before and is nothing at all for a type outside it (a WAVE).

    A FLUID's rna carries its live role's settings too, under that struct's identifier (domain_settings,
    flow_settings or effector_settings) - where set_modifier writes a fluid's keys, so a write there has a reader."""
    reject_unknown(params, {"object", "name", "limit", "nameExcludes", "rna"}, "list_modifiers")
    obj = _modifier_host(take(params, "object", "name", required=True, kind=str))
    want_rna = take_bool(params, "rna", default=False)
    rows = [_modifier_dict(m) for m in _stack(obj)]
    if want_rna:
        for row, mod in zip(rows, _stack(obj)):
            row["rna"] = rna_properties(mod)
            role_attr, role = _fluid_role(mod)
            if role is not None:
                row["rna"][role_attr] = rna_properties(role, skip=_FLUID_DATA_PROPS)
            # AND A CLOTH'S, COLLISION'S OR SOFT BODY'S STRUCTS, where set_modifier writes their keys.
            for attr in _MODIFIER_SUBSTRUCTS.get(mod.type, ()):
                struct = getattr(mod, attr, None)
                if struct is not None:
                    row["rna"][attr] = rna_properties(struct)
    # nameExcludes BEFORE the count, so `count` is the number that MATCHED the request as asked. A rig's
    # modifier names are mostly DEF-, MCH- and ORG- families, and a limit alone hands a caller twenty rows of the
    # family they were trying to look past.
    rows = exclude_rows(rows, params)
    matched = len(rows)
    rows, extra = limit_rows(rows, params, "modifier(s)")
    return dict({
        "object": obj.name,
        "count": matched,
        "modifiers": rows,
    }, **extra)


_PAINTW_KEYS = {"object", "group", "path", "radius", "weight", "strength", "falloff", "falloffShape", "mode",
                "spacing", "accumulate", "frontFacesOnly", "viewDirection", "mask", "autoNormalize", "createGroup"}
_PAINTW_MODES = ("MIX", "ADD", "SUBTRACT", "REPLACE", "LIGHTEN", "DARKEN")


def _paintw_mask(obj, mask):
    """The vertex indices a mask lets the brush reach, or None for every vertex. Read only."""
    if mask is None:
        return None
    if not isinstance(mask, dict) or len(mask) != 1 or not set(mask) <= {"faces", "vertices", "linked"}:
        raise MifOpError("mask is one of {faces: [i...]}, {vertices: [i...]} or {linked: vertex i} (the piece that "
                         "vertex is part of), got %r. NOTHING was changed." % (mask,))
    me = obj.data
    (kind, val), = mask.items()
    if kind == "linked":
        if isinstance(val, bool) or not isinstance(val, int) or not 0 <= val < len(me.vertices):
            raise MifOpError("mask.linked is a vertex index 0-%d, got %r. NOTHING was changed."
                             % (len(me.vertices) - 1, val))
        nbr = {}
        for e in me.edges:
            a, b = e.vertices
            nbr.setdefault(a, []).append(b)
            nbr.setdefault(b, []).append(a)
        seen, todo = {val}, [val]
        while todo:
            for n in nbr.get(todo.pop(), ()):
                if n not in seen:
                    seen.add(n)
                    todo.append(n)
        return seen
    top = len(me.polygons) if kind == "faces" else len(me.vertices)
    if not isinstance(val, list) or not val or not all(isinstance(i, int) and not isinstance(i, bool)
                                                        and 0 <= i < top for i in val):
        raise MifOpError("mask.%s is a non-empty list of indices 0-%d, got %r. NOTHING was changed."
                         % (kind, top - 1, val if not isinstance(val, list) else val[:8]))
    if kind == "vertices":
        return set(val)
    return {v for i in val for v in me.polygons[i].vertices}


def op_paint_weights(params):
    """Paint a vertex group along a stroke the way the weight brush does - in mesh data, with no viewport.

    THE GAP. set_vertex_weights writes listed weights and gradients by axis; nothing painted a stroke - a radius with
    falloff, a strength that builds up, front faces only, a mask limiting the brush to one piece. Blender's own
    paint.weight_paint needs a 3D view (the same reason sculpt_stroke works on the vertex positions), so the brush is
    done here: every dab is a sphere (or, falloffShape PROJECTED, a cylinder along viewDirection) round a path point.

    params:
      object, group (created when missing unless createGroup false)
      path [[x,y,z]...] WORLD points, one point is a dab; spacing resamples it every spacing x radius
      radius (world), weight 0-1 (default 1), strength 0-1 (default 1)
      falloff SMOOTH | SPHERE | ROOT | SHARP | LINEAR | CONSTANT (sculpt_stroke's curves), falloffShape SPHERE | PROJECTED
      mode MIX (toward weight, default) | ADD | SUBTRACT | REPLACE | LIGHTEN | DARKEN, each by f = falloff x strength
      accumulate false (default): one stroke changes a vertex at most by its strongest dab, as Blender's brush without
                 Accumulate; true: every dab applies on top of the last
      frontFacesOnly: a vertex is painted only when its normal faces the viewer - viewDirection [x,y,z] (where the view
                 looks), or each dab's area normal when none is given
      mask {faces [i] | vertices [i] | linked vertex i}
      autoNormalize: the other groups (the deform bones' with an Armature modifier, else every other group) are scaled
                 on each painted vertex so its weights sum to 1, the painted weight kept
    Reply: dabs, verticesTouched, weightRange written, meanBefore and meanAfter on the touched vertices.
    """
    reject_unknown(params, _PAINTW_KEYS, "paint_weights", {
        "brush": "weight painting has one brush here; mode picks how it blends (MIX, ADD, SUBTRACT, REPLACE, LIGHTEN, "
                 "DARKEN).",
        "symmetry": "not offered: paint the mirrored path in a second call, or mirror the group with "
                    "set_vertex_weights.",
    })
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    require_editable(obj, "painted")
    if obj.mode == "EDIT":
        raise MifOpError("'%s' is in Edit Mode, where vertex group writes do not reach the mesh until it leaves. "
                         "Leave Edit Mode first. NOTHING was changed." % obj.name)
    gname = take(params, "group", required=True, kind=str)
    create = take_bool(params, "createGroup", default=True)
    vg = obj.vertex_groups.get(gname)
    if vg is None and not create:
        raise MifOpError("'%s' has no vertex group '%s' and createGroup is false. Groups: %s. NOTHING was changed."
                         % (obj.name, gname, ", ".join(g.name for g in obj.vertex_groups)[:300] or "<none>"))
    raw = params.get("path")
    if not isinstance(raw, list) or not raw or not all(isinstance(p, list) and len(p) == 3 for p in raw):
        raise MifOpError("path is a non-empty list of [x, y, z] world points - one point is a dab. NOTHING was "
                         "changed.")
    path = [finite_floats(p, "path") for p in raw]
    radius = take_float(params, "radius", required=True)
    weight = take_float(params, "weight", default=1.0)
    strength = take_float(params, "strength", default=1.0)
    if not radius > 0.0 or not 0.0 <= weight <= 1.0 or not 0.0 < strength <= 1.0:
        raise MifOpError("radius must be above 0 (world), weight 0-1 and strength above 0 up to 1 - got %r, %r, %r. "
                         "NOTHING was changed." % (radius, weight, strength))
    from .ops_topology import _SCULPT_FALLOFF   # noqa: PLC0415
    falloff = str(take(params, "falloff", default="SMOOTH", kind=str)).upper()
    shape = str(take(params, "falloffShape", default="SPHERE", kind=str)).upper()
    mode = str(take(params, "mode", default="MIX", kind=str)).upper()
    if falloff not in _SCULPT_FALLOFF or shape not in ("SPHERE", "PROJECTED") or mode not in _PAINTW_MODES:
        raise MifOpError("falloff is one of %s, falloffShape SPHERE or PROJECTED, mode one of %s - got %r, %r, %r. "
                         "NOTHING was changed." % (", ".join(_SCULPT_FALLOFF), ", ".join(_PAINTW_MODES),
                                                   falloff, shape, mode))
    spacing = take_float(params, "spacing", default=None)
    if spacing is not None and not spacing > 0.0:
        raise MifOpError("spacing is a fraction of the radius, above 0. NOTHING was changed.")
    accumulate = take_bool(params, "accumulate", default=False)
    front = take_bool(params, "frontFacesOnly", default=False)
    view = params.get("viewDirection")
    if view is not None:
        view = finite_floats(view, "viewDirection") if isinstance(view, list) and len(view) == 3 else None
        if not view or sum(c * c for c in view) < 1e-18:
            raise MifOpError("viewDirection is [x, y, z], the direction the view looks along, not zero. NOTHING was "
                             "changed.")
    if shape == "PROJECTED" and view is None:
        raise MifOpError("falloffShape PROJECTED paints through the mesh along a view - pass viewDirection. NOTHING "
                         "was changed.")
    reach = _paintw_mask(obj, params.get("mask"))
    auto = take_bool(params, "autoNormalize", default=False)
    arm_mod = next((m for m in obj.modifiers if m.type == "ARMATURE" and m.object is not None
                    and m.object.type == "ARMATURE" and m.show_viewport), None)
    deform = {b.name for b in arm_mod.object.data.bones if b.use_deform} if arm_mod is not None else None
    if auto and deform is not None and gname not in deform:
        raise MifOpError("autoNormalize balances bone weights, and '%s' is not a deform bone of '%s'. Paint a bone's "
                         "group, or drop autoNormalize. NOTHING was changed." % (gname, arm_mod.object.name))

    # ---- measure: every new weight is computed before anything is written
    from mathutils import Vector                # noqa: PLC0415
    from mathutils.kdtree import KDTree         # noqa: PLC0415
    from .ops_topology import _sculpt_dabs      # noqa: PLC0415
    me = obj.data
    mw = obj.matrix_world
    nmat = mw.to_3x3().inverted_safe().transposed()
    world = [mw @ v.co for v in me.vertices]
    normals = [(nmat @ v.normal).normalized() for v in me.vertices]
    dabs = _sculpt_dabs([Vector(p) for p in path], [False, False, False], spacing, radius)[0]
    vdir = Vector(view).normalized() if view is not None else None

    def flat(co):
        return co - vdir * co.dot(vdir) if shape == "PROJECTED" else co
    tree = KDTree(len(world))
    for i, co in enumerate(world):
        tree.insert(flat(co), i)
    tree.balance()
    fall = _SCULPT_FALLOFF[falloff]
    gi = vg.index if vg is not None else None
    old = {}
    if gi is not None:
        for v in me.vertices:
            for g in v.groups:
                if g.group == gi:
                    old[v.index] = g.weight

    def blend(w, f):
        if mode == "MIX":
            return w + (weight - w) * f
        if mode == "ADD":
            return min(1.0, w + weight * f)
        if mode == "SUBTRACT":
            return max(0.0, w - weight * f)
        if mode == "REPLACE":
            return weight
        if mode == "LIGHTEN":
            return max(w, w + (weight - w) * f) if weight > w else w
        return min(w, w + (weight - w) * f) if weight < w else w

    new, strongest = {}, {}
    for c in dabs:
        hits = tree.find_range(flat(c), radius)
        if front:
            if vdir is not None:
                facing = -vdir
            else:
                area = sum((normals[i] for _co, i, _d in hits), Vector())
                facing = area.normalized() if area.length > 1e-12 else None
        for _co, i, d in hits:
            if reach is not None and i not in reach:
                continue
            if front and (facing is None or normals[i].dot(facing) <= 0.0):
                continue
            f = fall(min(d / radius, 1.0)) * strength
            if f <= 0.0:
                continue
            if accumulate:
                new[i] = blend(new.get(i, old.get(i, 0.0)), f)
            elif f > strongest.get(i, 0.0):
                strongest[i] = f
    if not accumulate:
        new = {i: blend(old.get(i, 0.0), f) for i, f in strongest.items()}
    new = {i: min(1.0, max(0.0, w)) for i, w in new.items()}
    touched = sorted(i for i, w in new.items() if (i in old and abs(w - old[i]) > 1e-9) or (i not in old and w > 0.0))
    if not touched:
        raise MifOpError("the stroke changed no weight - %s. NOTHING was changed."
                         % ("it reached %d vertex(es), and %s leaves each where it is" % (len(new), mode) if new else
                            "no vertex lies within %g of the path%s%s" % (
                                radius, " inside the mask" if reach is not None else "",
                                " facing the view" if front else "")))

    # ---- write
    made = vg is None
    if made:
        vg = obj.vertex_groups.new(name=gname)
    for i in touched:
        vg.add([i], new[i], "REPLACE")
    balanced, unbalanced = 0, 0
    if auto:
        others = {g.index for g in obj.vertex_groups if g.index != vg.index
                  and (deform is None or g.name in deform)}
        for i in touched:
            rows = [g for g in me.vertices[i].groups if g.group in others]
            total = sum(g.weight for g in rows)
            if total <= 0.0:
                unbalanced += 1 if new[i] < 1.0 else 0
                continue
            k = max(0.0, 1.0 - new[i]) / total
            for g in rows:
                g.weight = g.weight * k
            balanced += 1
    me.update()
    before = [old.get(i, 0.0) for i in touched]
    after = [new[i] for i in touched]
    out = {"object": obj.name, "group": vg.name, "groupCreated": made, "dabs": len(dabs), "mode": mode,
           "accumulate": accumulate, "verticesTouched": len(touched),
           "weightRange": [round(min(after), 6), round(max(after), 6)],
           "meanBefore": round(sum(before) / len(before), 6), "meanAfter": round(sum(after) / len(after), 6),
           "masked": len(reach) if reach is not None else None, "frontFacesOnly": front}
    if made:
        out.update(requestedName=gname, nameWasSuffixed=vg.name != gname)
    if auto:
        out["autoNormalize"] = {"balancedVertices": balanced, "groups": "deform bones" if deform is not None
                                else "every other group", "leftUnderOne": unbalanced}
    return out


def op_normalize_weights(params):
    """Make every vertex's bone weights sum to 1, and cap how many bones influence one vertex.

    THE WRITE HALF this file did not have. list_vertex_groups could report that a
    mesh has 40 groups and that some vertex is influenced by 11 of them, and nothing
    could act on it - the same detect-but-cannot-fix shape uv_unwrap closed for UVs
    and apply_transform closed for pivots.

    WHY THE LIMIT MATTERS, and it is not a Blender concern at all. Unreal's GPU skin
    cache supports a bounded number of influences per vertex - 4 by default, 8 or 12
    with the project setting raised - and the FBX importer DROPS the smallest weights
    past that limit and renormalises silently. So a mesh that deforms correctly in
    Blender can deform differently in Unreal, and nothing in either tool says why.
    Limiting here, deliberately, means the weights you tested are the weights that
    ship.

    ORDER IS LOAD-BEARING: limit first, THEN normalise. Limiting drops the smallest
    influences, which leaves the remaining weights summing to less than 1; normalise
    afterwards restores the sum. Doing it the other way round renormalises and then
    throws part of the result away, so the sum ends up wrong - which is exactly the
    bug the importer produces and this op exists to pre-empt.

    UNWEIGHTED VERTICES ARE REPORTED, NEVER INVENTED. A vertex in no group at all
    cannot be normalised - there is nothing to scale - and guessing a bone for it
    would be a silent wrong answer of the worst kind, because it deforms plausibly.
    They are counted and named in the response instead.

    ONLY BONE GROUPS ON A RIGGED MESH. A group no deform bone is named after - a cloth pin group, a
    panel or region group - is not an influence: Unreal's importer drops it and renormalises what is
    left. Counting it made pins and panels share the sum with the bones (2026-10-03, the garment work:
    a cuff rim vertex kept 0.227 bone weight, and every shipped skin lost an influence slot to its
    P_* panel group). So with an enabled Armature modifier bound to an armature the default scope is
    the armature's deform bones, and the groups left out are named; scope "all" is the old behaviour,
    and a mesh with no armature has nothing to tell bones from, so it keeps every group.
    """
    reject_unknown(params, ("object", "name", "maxInfluences", "normalize", "groups", "scope", "respectLocked"),
                   "normalize_weights")
    obj = get_object(take(params, "object", "name", required=True), want_mesh=True)
    # LOCKED GROUPS (2026-10-06): respectLocked keeps every group whose lock_weight is set as it is - not limited, not
    # scaled - and scales the rest to fill what is left of 1, as Blender's own Normalize does from 4.4. Default true on
    # 4.4 and later, false before (the build's own behaviour); lock_weight was never read here.
    respect = take_bool(params, "respectLocked", default=bpy.app.version >= (4, 4, 0))
    locked = {vg.index for vg in obj.vertex_groups if vg.lock_weight} if respect else set()

    max_inf = take_int(params, "maxInfluences", default=0)
    do_norm = take_bool(params, "normalize", default=True)
    only = take(params, "groups", default=None)
    scope_given = params.get("scope") is not None
    scope = str(take(params, "scope", default="bones", kind=str)).lower()
    if scope not in ("bones", "all"):
        raise MifOpError("scope must be 'bones' or 'all' (got %r). NOTHING was changed." % scope)
    if scope_given and only is not None:
        raise MifOpError("pass groups OR scope - groups names the exact groups to count, scope picks them. "
                         "NOTHING was changed.")

    if not obj.vertex_groups:
        raise MifOpError(
            "'%s' has no vertex groups, so there are no weights to normalise. NOTHING was changed."
            % obj.name)
    if max_inf and max_inf < 1:
        raise MifOpError("maxInfluences must be at least 1 - a vertex with zero influences is not "
                         "skinned at all. NOTHING was changed.")
    if not do_norm and not max_inf:
        raise MifOpError(
            "normalize_weights was asked to do nothing - normalize is false and maxInfluences is "
            "unset. NOTHING was changed.")

    if only is not None:
        if not isinstance(only, (list, tuple)):
            raise MifOpError("groups must be an array of vertex group names. NOTHING was changed.")
        missing = [g for g in only if g not in obj.vertex_groups]
        if missing:
            raise MifOpError(
                "no vertex group named %s on '%s' - %d group(s) exist. NOTHING was changed."
                % (", ".join("'%s'" % m for m in missing), obj.name, len(obj.vertex_groups)))
    allowed = None if only is None else {obj.vertex_groups[g].index for g in only}

    # The armature the mesh deforms by: the first enabled Armature modifier with an armature object.
    arm_mod = next((m for m in obj.modifiers if m.type == "ARMATURE" and m.object is not None
                    and m.object.type == "ARMATURE" and m.show_viewport), None)
    scope_used = "groups" if only is not None else scope
    skipped = []
    if only is None and scope == "bones":
        if arm_mod is None:
            if scope_given:
                raise MifOpError(
                    "scope 'bones' needs an enabled Armature modifier bound to an armature, and '%s' has none, "
                    "so no group can be told to be a bone. Pass scope 'all' or name the groups. NOTHING was "
                    "changed." % obj.name)
            scope_used = "all (no armature)"
        else:
            deform = {b.name for b in arm_mod.object.data.bones if b.use_deform}
            allowed = {vg.index for vg in obj.vertex_groups if vg.name in deform}
            skipped = sorted(vg.name for vg in obj.vertex_groups if vg.name not in deform)
            if not allowed:
                raise MifOpError(
                    "none of '%s''s %d vertex group(s) is named after a deform bone of '%s', so there are no "
                    "bone weights to normalise. Pass scope 'all' to treat every group as an influence. NOTHING "
                    "was changed." % (obj.name, len(obj.vertex_groups), arm_mod.object.name))

    me = obj.data
    unweighted = 0
    limited = 0
    normalised = 0
    dropped_total = 0
    max_seen_before = 0

    for v in me.vertices:
        # A ZERO WEIGHT IS NOT AN INFLUENCE, and counting one is what made this op report a full
        # round of work on a mesh it had already capped. v.groups is MEMBERSHIP: the trim below
        # zeroes a weight rather than removing the group - deliberately, because "removing while
        # iterating a vertex's own group list is what corrupts the mesh" - so after one run every
        # vertex is still IN all its original groups at weight 0.
        #
        # Measured 2026-08-31 on a cube with 8 groups at 0.125 each, maxInfluences 4:
        #
        #     before        64 influences (list_vertex_groups)
        #     run 1         influencesDropped 32, verticesLimited 8, maxSeenBefore 8   -> 32 left
        #     run 2         influencesDropped 32, verticesLimited 8, maxSeenBefore 8   -> 32 left
        #
        # The second call changed NOTHING and reported the same work as the first. A caller who
        # normalises twice is told twice that weights were thrown away, and there is no other field
        # that would tell them otherwise.
        #
        # Filtering here also settles a disagreement between two ops about what an "influence" is:
        # list_vertex_groups' weightedVertexCount already counts only NONZERO weights, so the two
        # were describing the same mesh with different numbers. The op's own comment on the trim -
        # "a zero weight is equivalent to absent everywhere Unreal reads it" - is the argument for
        # this being the right side of that disagreement.
        elems = [g for g in v.groups
                 if (allowed is None or g.group in allowed) and g.weight > 0.0]
        if not elems:
            unweighted += 1
            continue
        max_seen_before = max(max_seen_before, len(elems))

        if max_inf and len(elems) > max_inf:
            elems.sort(key=lambda g: (g.group in locked, g.weight), reverse=True)
            for g in elems[max_inf:]:
                # Zeroing rather than removing: removing while iterating a vertex's
                # own group list is what corrupts the mesh, and a zero weight is
                # equivalent to absent everywhere Unreal reads it.
                dropped_total += 1
                g.weight = 0.0
            elems = elems[:max_inf]
            limited += 1

        if do_norm:
            fixed = sum(g.weight for g in elems if g.group in locked)
            free = [g for g in elems if g.group not in locked]
            total = sum(g.weight for g in free)
            room = max(0.0, 1.0 - fixed) if locked else 1.0
            if total > 0.0 and abs(total - room) > 1e-6:
                for g in free:
                    g.weight = g.weight * room / total
                normalised += 1

    me.update()

    out = {
        "object": obj.name,
        "vertices": len(me.vertices),
        "groupsConsidered": len(obj.vertex_groups) if allowed is None else len(allowed),
        "scope": scope_used,
        "respectLocked": respect,
        "lockedGroups": sorted(vg.name for vg in obj.vertex_groups if vg.index in locked),
        "armature": arm_mod.object.name if arm_mod is not None else None,
        "groupsLeftOut": skipped,
        "maxInfluencesRequested": max_inf or None,
        "maxInfluencesSeenBefore": max_seen_before,
        "verticesLimited": limited,
        "influencesDropped": dropped_total,
        "verticesNormalized": normalised,
        "unweightedVertices": unweighted,
        "changedAnything": bool(limited or normalised),
    }
    if unweighted:
        out["unweightedNote"] = (
            "%d vertex/vertices belong to NO vertex group and were left alone. They cannot be "
            "normalised - there is nothing to scale - and assigning them a bone would be a guess "
            "that deforms plausibly and wrongly. In Unreal these bind to the root bone." % unweighted)
    if max_inf and max_seen_before <= max_inf:
        out["limitNote"] = (
            "no vertex exceeded %d influences (the most any had was %d), so the limit changed "
            "nothing. Reported rather than returned as work performed." % (max_inf, max_seen_before))
    if not out["changedAnything"]:
        out["note"] = ("the weights were already normalised and within the influence limit - "
                       "counts are unchanged.")
    return out


def op_vertex_groups_from_bones(params):
    """Make vertex groups from bone weights: each new group holds the vertices whose summed weight on chosen bones
    reaches a threshold.

    THE GAP, 2026-10-03: garment work needs REGION groups - the hood, the cuffs, the hem, a self-collision band, a pin
    rim - and a skinned mesh already says where each region is, in its bone weights. The garment and outfit agents built
    them in raw Python twenty times (selfcol_group, rib_group, outfit_shell_weights); set_vertex_weights takes explicit
    vertex lists, and nothing computed one from the weights.

    groups: {name: {bones: [bone group names], threshold (default 0.5), value ("sum" - the summed weight, clamped to
    1 - or "one", default)}}. A bone named that has no vertex group on the mesh is refused with the groups it does
    have; a name that already exists is refused unless replace:true. Every group is computed before any is written, so
    a refusal leaves the mesh as it was. Each group's vertex count is reported, and a group that would come out empty
    is refused rather than created empty.
    """
    reject_unknown(params, ("object", "name", "groups", "replace"), "vertex_groups_from_bones")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    spec = params.get("groups")
    replace = take_bool(params, "replace", default=False)
    if not isinstance(spec, dict) or not spec:
        raise MifOpError("groups must be {name: {bones: [...], threshold, value}}. NOTHING was changed.")
    existing = {vg.name: vg.index for vg in obj.vertex_groups}
    plans = {}
    for gname, g in spec.items():
        if not isinstance(g, dict) or not isinstance(g.get("bones"), list) or not g["bones"]:
            raise MifOpError("groups['%s'] needs bones: a non-empty list of vertex group names. NOTHING was changed."
                             % gname)
        unknown = set(g) - {"bones", "threshold", "value"}
        if unknown:
            raise MifOpError("groups['%s'] has unknown key(s) %s - it takes bones, threshold, value. NOTHING was "
                             "changed." % (gname, ", ".join(sorted(unknown))))
        missing = [b for b in g["bones"] if b not in existing]
        if missing:
            raise MifOpError("groups['%s']: '%s' has no vertex group %s. Its groups: %s. NOTHING was changed."
                             % (gname, obj.name, ", ".join("'%s'" % m for m in missing),
                                ", ".join(sorted(existing)[:40])))
        thr = finite_float(g.get("threshold", 0.5), "groups['%s'].threshold" % gname)
        if not 0.0 < thr <= 1.0:
            raise MifOpError("groups['%s'].threshold must be above 0 and at most 1, got %r. NOTHING was changed."
                             % (gname, thr))
        mode = str(g.get("value", "one")).lower()
        if mode not in ("one", "sum"):
            raise MifOpError("groups['%s'].value must be 'one' or 'sum', got %r. NOTHING was changed." % (gname, mode))
        if gname in existing and not replace:
            raise MifOpError("'%s' already has a vertex group '%s' - pass replace:true to rebuild it. NOTHING was "
                             "changed." % (obj.name, gname))
        if gname in g["bones"]:
            raise MifOpError("groups['%s'] cannot be built from itself. NOTHING was changed." % gname)
        idx = {existing[b] for b in g["bones"]}
        rows = []
        for v in obj.data.vertices:
            total = sum(e.weight for e in v.groups if e.group in idx)
            if total >= thr:
                rows.append((v.index, min(total, 1.0) if mode == "sum" else 1.0))
        if not rows:
            raise MifOpError("groups['%s'] would be EMPTY: no vertex of '%s' has a summed weight of %g or more on %s. "
                             "NOTHING was changed." % (gname, obj.name, thr, ", ".join(g["bones"])))
        plans[gname] = rows
    made = {}
    for gname, rows in plans.items():
        vg = obj.vertex_groups.get(gname)
        if vg is not None:
            obj.vertex_groups.remove(vg)
        vg = obj.vertex_groups.new(name=gname)
        for vi, w in rows:
            vg.add([vi], w, "REPLACE")
        made[gname] = {"vertices": len(rows), "replaced": gname in existing,
                       "minWeight": round(min(w for _v, w in rows), 6)}
    obj.data.update()
    return {"object": obj.name, "groups": made}


def op_transfer_weights(params):
    """Copy vertex weights from one mesh onto another by proximity.

    The op a retopology or LOD pass needs and this addon had no answer for: a
    decimated or rebuilt mesh comes out of clean_mesh/decimate_mesh with its
    topology changed and its skinning destroyed, and re-rigging by hand is the
    expensive part. Blender's data transfer maps weights from the original by
    nearest-surface, which is exactly the operation, and nothing exposed it.

    IT REFUSES RATHER THAN GUESSES in the two cases where a plausible-looking
    result would be wrong:
      - the source has no vertex groups: there is nothing to transfer, and
        producing an unskinned mesh while reporting success is the silent-success
        shape this project keeps finding.
      - source and destination are the same object: the operator would happily run
        and achieve nothing.

    WHAT IT DOES NOT DO: it does not normalise. Transferred weights routinely do not
    sum to 1, because a nearest-surface mapping interpolates between source vertices
    with different totals. Call normalize_weights afterwards - separately and
    visibly, rather than folded in here, so the caller knows it happened. The
    response says so when the result needs it.

    ONLY SOME GROUPS, OR ONLY SOME VERTICES (2026-10-04). onlyGroups / excludeGroups pick the source groups that
    are transferred; destinationMask (a vertex group on the destination) limits which destination vertices receive
    them. Blender's data transfer cannot choose groups by name, so the destination's weights are snapshotted, the
    full transfer runs, and every group outside the choice - and every vertex outside the mask - is put back exactly
    as it was; a group the transfer created that was not chosen is removed. The garment work needed this per region
    (a hood takes the head and neck bones, a sleeve the arm's) and did it in raw Python.
    """
    reject_unknown(params, ("source", "from", "destination", "to", "object", "mapping", "onlyGroups",
                            "excludeGroups", "destinationMask"),
                   "transfer_weights", {
        "normalize": "this op does NOT normalise, deliberately. Transferred weights routinely do "
                     "not sum to 1 because a nearest-surface copy takes each group independently, "
                     "and normalising inside the transfer would hide that. Call normalize_weights "
                     "afterwards, as a separate step you can see.",
        "normalise": "see 'normalize'.",
        "maxInfluences": "that is set_vertex_weights' - a limit applied while transferring would "
                         "silently drop the influences this op exists to copy.",
    })
    src = get_object(take(params, "source", "from", required=True), want_mesh=True)
    dst = get_object(take(params, "destination", "to", "object", required=True), want_mesh=True)

    if src.name == dst.name:
        raise MifOpError(
            "source and destination are the same object ('%s') - there is nothing to transfer. "
            "NOTHING was changed." % src.name)
    if not src.vertex_groups:
        raise MifOpError(
            "source '%s' has no vertex groups, so there are no weights to transfer. Producing an "
            "unskinned result and reporting success would be worse than refusing. NOTHING was "
            "changed." % src.name)

    mapping = (take(params, "mapping", default="POLYINTERP_NEAREST") or "POLYINTERP_NEAREST").upper()
    valid = ("NEAREST", "EDGE_NEAREST", "EDGEINTERP_NEAREST", "POLY_NEAREST",
             "POLYINTERP_NEAREST", "POLYINTERP_VNORPROJ")
    if mapping not in valid:
        raise MifOpError("unknown mapping '%s' for transfer_weights. Accepted: %s. NOTHING was "
                         "changed." % (mapping, ", ".join(valid)))

    only = params.get("onlyGroups")
    exclude = params.get("excludeGroups")
    mask = take(params, "destinationMask", default=None, kind=str)
    if only is not None and exclude is not None:
        raise MifOpError("pass onlyGroups OR excludeGroups, not both. NOTHING was changed.")
    for key, val in (("onlyGroups", only), ("excludeGroups", exclude)):
        if val is not None and (not isinstance(val, (list, tuple)) or not val
                                or not all(isinstance(n, str) for n in val)):
            raise MifOpError("%s must be a non-empty list of source vertex group names. NOTHING was changed." % key)
    src_names = {g.name for g in src.vertex_groups}
    named = list(only or []) + list(exclude or [])
    missing = [n for n in named if n not in src_names]
    if missing:
        raise MifOpError("source '%s' has no vertex group %s. NOTHING was changed."
                         % (src.name, ", ".join("'%s'" % m for m in missing)))
    if mask is not None and mask not in {g.name for g in dst.vertex_groups}:
        raise MifOpError("destinationMask: '%s' has no vertex group '%s'. NOTHING was changed." % (dst.name, mask))
    chosen = (set(only) if only is not None else src_names - set(exclude or [])) if (only or exclude) else None
    if chosen is not None and not chosen:
        raise MifOpError("excludeGroups leaves no source group to transfer. NOTHING was changed.")
    filtered = chosen is not None or mask is not None
    # THE SNAPSHOT a filtered transfer restores from: {group name: {vertex: weight}} and which vertices the mask holds.
    before_w, mask_verts = {}, None
    if filtered:
        names = {vg.index: vg.name for vg in dst.vertex_groups}
        for v in dst.data.vertices:
            for e in v.groups:
                before_w.setdefault(names[e.group], {})[v.index] = e.weight
        if mask is not None:
            mask_verts = {vi for vi, w in before_w.get(mask, {}).items() if w > 0.0}
            if not mask_verts:
                raise MifOpError("destinationMask '%s' holds no weighted vertex on '%s'. NOTHING was changed."
                                 % (mask, dst.name))
    existed = {vg.name for vg in dst.vertex_groups}

    groups_before = len(dst.vertex_groups)
    snap = selection_snapshot()
    try:
        select_only([src, dst])
        bpy.context.view_layer.objects.active = dst
        bpy.ops.object.data_transfer(
            use_reverse_transfer=True,
            data_type="VGROUP_WEIGHTS",
            vert_mapping=mapping,
            # SWAPPED, BECAUSE use_reverse_transfer SWAPS WHICH ENUM VALIDATES WHICH ARGUMENT.
            # layers_select_src normally takes ACTIVE|ALL|BONE_SELECT|BONE_DEFORM and
            # layers_select_dst takes ACTIVE|NAME|INDEX - but with reverse on, "ALL" passed to
            # _src is checked against the DST enum and raises
            #   TypeError: enum "ALL" not found in ('ACTIVE', 'NAME', 'INDEX')
            # so this op had NEVER WORKED on any Blender. Measured on 3.6, 4.2, 4.4 and 5.0: as
            # shipped it raises on all four; swapped it returns FINISHED and the destination gains
            # the group on all four.
            #
            # Found by blender_version_matrix once set_vertex_weights existed to give the source a
            # group - before that this op refused for "no vertex groups" and never reached the call.
            layers_select_src="NAME",
            layers_select_dst="ALL",
            mix_mode="REPLACE",
        )
    finally:
        selection_restore(snap)

    kept_as_before, removed_new, restored_outside_mask = [], [], 0
    if filtered:
        all_idx = list(range(len(dst.data.vertices)))
        for vg in list(dst.vertex_groups):
            name = vg.name
            if name == mask:
                continue
            transfer_it = name in src_names and (chosen is None or name in chosen)
            old = before_w.get(name, {})
            if not transfer_it:
                if name not in existed:
                    dst.vertex_groups.remove(vg)
                    removed_new.append(name)
                    continue
                vg.remove(all_idx)
                for vi, w in old.items():
                    vg.add([vi], w, "REPLACE")
                kept_as_before.append(name)
            elif mask_verts is not None:
                outside = [vi for vi in all_idx if vi not in mask_verts]
                vg.remove(outside)
                for vi in outside:
                    if vi in old:
                        vg.add([vi], old[vi], "REPLACE")
                restored_outside_mask += 1

    # READ BACK. data_transfer reports nothing useful, so the postcondition is
    # measured: how many groups the destination has now, and how many of its
    # vertices actually carry a weight.
    #
    # NONZERO WEIGHT, NOT MEMBERSHIP - the same distinction op_normalize_weights documents at length
    # 120 lines up, and the 2026-08-31 fix there corrected only that site. `v.groups` is MEMBERSHIP:
    # a vertex trimmed to weight 0 is still IN the group, and layers_select_dst="NAME" reuses groups
    # that already exist rather than adding any. So on a SECOND transfer into the same destination,
    # `len(dst.vertex_groups) > groups_before` is False by construction and `weighted > 0` was
    # guaranteed True by the memberships the first run left behind - which made the refusal below
    # unreachable in exactly the iterate-the-rig case it exists for.
    #
    # It also made two numbers wrong rather than merely weak: destinationVerticesWeighted reported
    # the vertex COUNT, and coverageNote tested the same measure, so a destination whose every
    # weight is 0.0 was reported as fully covered with no note. This file's own argument for the
    # filter - "a zero weight is equivalent to absent everywhere Unreal reads it" - applies here
    # unchanged, and list_vertex_groups' weightedVertexCount has always counted it this way.
    weighted = sum(1 for v in dst.data.vertices if any(g.weight > 0.0 for g in v.groups))
    out = {
        "source": src.name,
        "destination": dst.name,
        "mapping": mapping,
        "sourceGroups": len(src.vertex_groups),
        "destinationGroupsBefore": groups_before,
        "destinationGroupsAfter": len(dst.vertex_groups),
        "destinationVertices": len(dst.data.vertices),
        "destinationVerticesWeighted": weighted,
        # A NON-ZERO WEIGHT, AND NOTHING ELSE. This was
        #     len(dst.vertex_groups) > groups_before or weighted > 0
        # and the OR made the group COUNT sufficient - so a first transfer that created the group
        # and wrote every weight as 0.0 reported transferred:True with weighted 0 of 8. Measured on
        # 4.4.0 and 5.0.1 with a half-weighted source: exactly the unskinned result the docstring
        # above says would be worse than refusing, reported as a success.
        #
        # The comment above added `weighted > 0` to rescue the SECOND-transfer case, where the group
        # already exists so the count cannot move. That was right; keeping the count as an
        # alternative was not. A group is not evidence - a group whose every weight is zero exists,
        # reads back perfectly in list_vertex_groups, and deforms nothing.
        "transferred": weighted > 0,
        "destinationGroupCountMoved": len(dst.vertex_groups) > groups_before,
    }
    if filtered:
        out["groupsTransferred"] = sorted(n for n in src_names if chosen is None or n in chosen)
        out["groupsKeptAsBefore"] = sorted(kept_as_before)
        out["groupsNotCreated"] = sorted(removed_new)
        if mask is not None:
            out["destinationMask"] = {"group": mask, "vertices": len(mask_verts),
                                      "groupsRestoredOutsideMask": restored_outside_mask}
    if not out["transferred"]:
        raise MifOpError(
            "the transfer ran and '%s' still has NO WEIGHTED VERTICES (%d groups before, %d "
            "after%s). Two usual causes: the meshes are far apart in world space, since a "
            "nearest-surface mapping needs them roughly coincident; or the source vertices nearest "
            "the destination are themselves unweighted, which a partially weighted source does "
            "quietly. Either way the group may now EXIST on '%s' and deform nothing, which is why "
            "this refuses instead of reporting the group it made. NOTHING usable was produced."
            % (dst.name, groups_before, len(dst.vertex_groups),
               " - a group WAS created" if len(dst.vertex_groups) > groups_before else "",
               dst.name))
    # THE WEIGHTS THEMSELVES, per transferred group - not only whether a vertex got one. The second seat
    # rendered all six mappings on 2026-09-16, saw six identical pictures, and suspected use_reverse_transfer
    # was swallowing vert_mapping. MEASURED on 4.4.0 with no addon: it is not - interpolating mappings move
    # the weights 0.025-0.113 mean-abs from NEAREST, identically reversed and forward, while EDGE_NEAREST
    # and POLY_NEAREST equal NEAREST on a regular grid. A render that does not visualise weights cannot show
    # any of that, and the response gave a count and nothing else, so there was no way to see a mapping's
    # effect through the bridge. min / max / mean / total over the vertices that carry weight now say it.
    # Capped at 64 groups, because a full rig has hundreds and this is a check, not a dump.
    # ONE name -> index map, not `g.name in dst.vertex_groups` per source group: that test walks the bpy
    # collection each time, which is quadratic on a rig (audit_quadratic_membership caught it, 2026-09-16).
    dst_index = {vg.name: vg.index for vg in dst.vertex_groups}
    wanted = {dst_index[g.name]: g.name for g in src.vertex_groups if g.name in dst_index}
    scan = _group_weight_scan(dst)                       # the scan list_vertex_groups reads, so the two cannot disagree
    by_name = {name: scan.get(index) for index, name in wanted.items()}
    shown = sorted(by_name)[:64]
    out["weightStats"] = {name: _weight_stats(by_name[name]) for name in shown}
    if len(by_name) > len(shown):
        out["weightStatsTruncated"] = {"groups": len(by_name), "shown": len(shown)}
    if weighted < len(dst.data.vertices):
        out["coverageNote"] = (
            "%d of %d destination vertices carry no weight - the mapping found nothing near them. "
            "In Unreal those bind to the root bone."
            % (len(dst.data.vertices) - weighted, len(dst.data.vertices)))
    out["normalizeNote"] = (
        "transferred weights are NOT normalised - a nearest-surface mapping interpolates between "
        "source vertices whose totals differ, so sums drift from 1. Call normalize_weights on "
        "'%s' before exporting." % dst.name)
    return out


# The WRITE side of _MODIFIER_FIELDS. Two tables, not one, and that is deliberate rather than
# duplication left unresolved: the read table is getter lambdas, and a write table needs setters,
# type coercion and a per-version presence check, so "share one description" is not achievable
# without making both halves worse. What IS enforced is that the two describe the same TYPES -
# test_blender_rig asserts the key sets match, so a type added to one and forgotten in the other
# is a failing test rather than a silent asymmetry.
#
# Each entry: param name -> (attribute, coercion). Coercion is applied before assignment so a JSON
# number arriving as a float for an int property is a clean error rather than a Blender exception
# surfacing as a stack trace.
def _idx_setter(i):
    def setter(mod, value):
        axes = list(mod.use_axis)
        axes[i] = bool(value)
        mod.use_axis = axes
    # WHAT IT WRITES, declared. Every other entry in _MODIFIER_WRITES holds an attribute name and
    # can be snapshotted by it; a callable cannot. _apply_modifier_settings reads this to roll the
    # value back when a later setting refuses. Hard-coding "use_axis" at the snapshot site would be
    # correct today and silently wrong the first time a second callable setter is added.
    setter.writes = "use_axis"
    return setter


_MODIFIER_WRITES = {
    "ARMATURE": {
        "object": ("object", "object"),
    },
    "MIRROR": {
        "axisX": (_idx_setter(0), bool),
        "axisY": (_idx_setter(1), bool),
        "axisZ": (_idx_setter(2), bool),
        "mergeThreshold": ("merge_threshold", float),
    },
    "SOLIDIFY": {"thickness": ("thickness", float)},
    "BEVEL": {"width": ("width", float), "segments": ("segments", int)},
    "SUBSURF": {"levels": ("levels", int), "renderLevels": ("render_levels", int)},
    "DECIMATE": {"decimateType": ("decimate_type", "enum"), "ratio": ("ratio", float)},
    "TRIANGULATE": {"quadMethod": ("quad_method", "enum")},
    # OCEAN, 2026-09-12. Fourteen of its twenty settings cannot be keyed (Blender marks them not
    # animatable), so until this entry no typed op could set an ocean's resolution, domain, wind or
    # spectrum at all - the ocean showcase measured every one. list_modifiers reads them back through
    # _generated_fields, so the write and the read are one table.
    "OCEAN": {
        "geometryMode": ("geometry_mode", "enum"), "size": ("size", float),
        "spatialSize": ("spatial_size", int), "resolution": ("resolution", int),
        "waveScale": ("wave_scale", float), "waveScaleMin": ("wave_scale_min", float),
        "choppiness": ("choppiness", float), "windVelocity": ("wind_velocity", float),
        "waveAlignment": ("wave_alignment", float), "waveDirection": ("wave_direction", float),
        "damping": ("damping", float), "depth": ("depth", float), "time": ("time", float),
        "repeatX": ("repeat_x", int), "repeatY": ("repeat_y", int), "randomSeed": ("random_seed", int),
        "spectrum": ("spectrum", "enum"), "useFoam": ("use_foam", bool),
        "foamCoverage": ("foam_coverage", float), "useNormals": ("use_normals", bool),
    },

    # EVERY IDENTIFIER BELOW WAS READ OFF bl_rna ON 3.6.23, 4.2.17, 4.4.0 AND 5.0.1 and is present
    # on all four - not taken from documentation, and not guessed from the UI labels, which differ.
    #
    # WHY THESE FOURTEEN. add_modifier could point exactly ONE modifier at an object (ARMATURE), and
    # a modifier that cannot be pointed at anything sits in the stack doing nothing while reading
    # back as a perfectly healthy modifier. That closed retopo (shrinkwrap), lattice/hook/mesh-deform
    # rigging, arrays along a curve, mirror-across-object, boolean operands and displacement - and
    # every one of them looked like a separate missing feature when the blocker was a single missing
    # idea: a NAME becoming a POINTER.
    #
    # vertex_group is a STRING on the modifier and is validated against the OBJECT's groups anyway,
    # because Blender accepts any string and a name that matches nothing is not an error. On a MASK
    # that is the difference between masking by a group and masking EVERYTHING.
    "SHRINKWRAP": {
        "target": ("target", "object"), "offset": ("offset", float),
        "vertexGroup": ("vertex_group", "vgroup"),
        "wrapMethod": ("wrap_method", "enum"), "wrapMode": ("wrap_mode", "enum"),
    },
    "LATTICE": {
        "object": ("object", "object"), "strength": ("strength", float),
        "vertexGroup": ("vertex_group", "vgroup"),
    },
    "HOOK": {
        "object": ("object", "object"), "strength": ("strength", float),
        "falloffRadius": ("falloff_radius", float),
        "vertexGroup": ("vertex_group", "vgroup"),
    },
    "MASK": {
        "vertexGroup": ("vertex_group", "vgroup"),
        "invert": ("invert_vertex_group", bool), "threshold": ("threshold", float),
        "armature": ("armature", "object"),
    },
    "CAST": {
        "object": ("object", "object"), "factor": ("factor", float),
        "radius": ("radius", float), "size": ("size", float),
        "castType": ("cast_type", "enum"),
    },
    "ARRAY": {
        "count": ("count", int), "curve": ("curve", "object"),
        "offsetObject": ("offset_object", "object"),
        "startCap": ("start_cap", "object"), "endCap": ("end_cap", "object"),
        "fitType": ("fit_type", "enum"), "mergeThreshold": ("merge_threshold", float),
        "mergeVertices": ("use_merge_vertices", bool),
        # THE TWO SWITCHES THAT DECIDE WHICH OFFSETS APPLY. offset_object is read ONLY while
        # use_object_offset is on, and Blender's default is relative offset alone - so without these
        # offsetObject set a pointer that did nothing. See the implication in _apply_modifier_settings.
        "useObjectOffset": ("use_object_offset", bool),
        "useRelativeOffset": ("use_relative_offset", bool),
    },
    "CURVE": {
        "object": ("object", "object"), "deformAxis": ("deform_axis", "enum"),
        "vertexGroup": ("vertex_group", "vgroup"),
    },
    "BOOLEAN": {
        "object": ("object", "object"), "collection": ("collection", "collection"),
        "operation": ("operation", "enum"), "solver": ("solver", "enum"),
        "doubleThreshold": ("double_threshold", float),
    },
    "DISPLACE": {
        "strength": ("strength", float), "midLevel": ("mid_level", float),
        "direction": ("direction", "enum"), "texture": ("texture", "texture"),
        "textureCoordsObject": ("texture_coords_object", "object"),
        "vertexGroup": ("vertex_group", "vgroup"),
    },
    "SIMPLE_DEFORM": {
        "deformMethod": ("deform_method", "enum"), "angle": ("angle", float),
        "factor": ("factor", float), "deformAxis": ("deform_axis", "enum"),
        "origin": ("origin", "object"), "vertexGroup": ("vertex_group", "vgroup"),
    },
    "MESH_DEFORM": {
        "object": ("object", "object"), "precision": ("precision", int),
        "vertexGroup": ("vertex_group", "vgroup"),
    },
    "SMOOTH": {
        "factor": ("factor", float), "iterations": ("iterations", int),
        "vertexGroup": ("vertex_group", "vgroup"),
    },
    "WELD": {
        "mergeThreshold": ("merge_threshold", float), "mode": ("mode", "enum"),
        "vertexGroup": ("vertex_group", "vgroup"),
    },
    "WIREFRAME": {
        "thickness": ("thickness", float), "offset": ("offset", float),
        "boundary": ("use_boundary", bool), "evenOffset": ("use_even_offset", bool),
        "creaseWeight": ("crease_weight", float),
    },
}


# WHICH bpy.data COLLECTION EACH POINTER KIND DRAWS FROM. Same idea as ops_nodes' _POINTER_SOCKETS,
# kept separate because a modifier field is addressed by RNA name rather than by socket type.
_DATABLOCK_COERCE = {
    "object": ("objects", "object"),
    "collection": ("collections", "collection"),
    "texture": ("textures", "texture"),
}


def _resolve_datablock(kind, value, key):
    """A NAME becoming a POINTER, refusing with what is actually in the file."""
    attr, noun = _DATABLOCK_COERCE[kind]
    if value is None:
        return None
    coll = getattr(bpy.data, attr)
    found = coll.get(str(value))
    if found is None:
        have = sorted(d.name for d in coll)[:25]
        raise MifOpError("no %s named '%s' for setting '%s'. This file has: %s. NOTHING was "
                         "changed." % (noun, value, key, ", ".join(have) if have else "(none)"))
    return found


def _apply_modifier_settings(mod, settings, mod_type):
    """Assign a settings dict onto a modifier. Returns the names actually applied."""
    table = _MODIFIER_WRITES.get(mod_type, {})
    if not isinstance(settings, dict):
        raise MifOpError("settings must be an object of {name: value}. NOTHING was changed.")
    unknown = [k for k in settings if k not in table]
    if unknown:
        raise MifOpError(
            "%s has no writable setting named %s here. This addon curates a handful of settings per "
            "type - the ones that decide what an export produces - rather than exposing every "
            "property Blender has. Accepted for %s: %s. NOTHING was changed."
            % (mod_type, ", ".join("'%s'" % u for u in unknown), mod_type,
               ", ".join(sorted(table)) or "(none yet)"))

    # SNAPSHOT BEFORE THE FIRST WRITE, so a refusal on the fifth setting does not leave the first
    # four standing under a message that says NOTHING was changed. That was true before the pointer
    # check below existed - the loop has always walked one key at a time - it simply had no refusal
    # that could fire after a write in the same breath, so nothing made it visible.
    restore = {}
    for key in settings:
        target = table[key][0]
        attr = target if isinstance(target, str) else getattr(target, "writes", None)
        if attr and attr not in restore:
            try:
                restore[attr] = getattr(mod, attr)
            except Exception:                                       # noqa: BLE001
                pass

    applied = []
    for key, value in settings.items():
        target, coerce = table[key]
        try:
            if coerce in _DATABLOCK_COERCE:
                wanted = _resolve_datablock(coerce, value, key)
                setattr(mod, target, wanted)
                # DID IT TAKE? Blender puts a poll on most datablock pointers and a value the poll
                # rejects is dropped in SILENCE - no exception, no warning, the pointer simply
                # stays None. Measured on 5.0.1: a CURVE modifier's object accepts only a curve and
                # a LATTICE's only a lattice, while HOOK and MESH_DEFORM take a mesh. So this is
                # per-property and cannot be answered by a hand-written table of what goes where;
                # asking the pointer afterwards is the only reliable question.
                #
                # Without this the response said ok:true with settingsApplied ["object"] and the
                # modifier sat inert - settingsApplied listing what was ASKED FOR as though it were
                # what happened.
                if wanted is not None and getattr(mod, target, None) is None:
                    kind = getattr(wanted, "type", None) or type(wanted).__name__
                    raise MifOpError(
                        "Blender refused '%s' (a %s) as '%s' for a %s modifier. That pointer only "
                        "accepts certain kinds, and an assignment it rejects raises NOTHING - the "
                        "pointer is simply left empty and the modifier sits in the stack doing "
                        "nothing. NOTHING was changed."
                        % (getattr(wanted, "name", value), kind, key, mod_type))
            elif coerce == "vgroup":
                # A STRING PROPERTY THAT MUST NAME A REAL GROUP. Blender takes any string here and
                # a name matching nothing is not an error - it just selects no vertices. On a MASK
                # that is the difference between masking by a group and masking the whole object.
                owner = mod.id_data
                if value is not None and str(value) not in owner.vertex_groups:
                    have = [g.name for g in owner.vertex_groups][:25]
                    raise MifOpError(
                        "'%s' has no vertex group named '%s'. Blender accepts any string here and "
                        "a name that matches nothing is NOT an error - it simply selects no "
                        "vertices, which on a MASK empties the object. Groups: %s. NOTHING was "
                        "changed." % (owner.name, value, ", ".join(have) if have else "(none)"))
                setattr(mod, target, str(value))
            elif coerce == "enum":
                # VALIDATED AGAINST THIS BUILD'S OWN ENUM, so the refusal names what is available
                # rather than letting RNA raise a TypeError that names the type and not the fix.
                valid = [i.identifier for i in mod.bl_rna.properties[target].enum_items]
                want = str(value).upper()
                match = [v for v in valid if v.upper() == want]
                if not match:
                    raise MifOpError(
                        "'%s' is not a valid %s for a %s modifier on Blender %s. Available: %s. "
                        "NOTHING was changed."
                        % (value, key, mod_type, bpy.app.version_string, ", ".join(valid)))
                setattr(mod, target, match[0])
            else:
                # THE THREE PLAIN COERCIONS WERE BARE BUILTINS, and each let a wrong value through as a
                # different right-looking one - measured on 5.0.1, 2026-09-20, through add_modifier:
                #   float: {"choppiness": NaN} stored nan and read it back as nan, under ok:true;
                #   bool:  {"axisY": {"mif": "x"}} switched the axis ON - bool(dict) is True;
                #   int:   {"levels": 2.7} became 2 in silence.
                # The same guards every other writer in this addon uses: finite_float, finite_int (which
                # refuses a fraction rather than truncating it), and a real boolean or 0/1.
                if coerce is float:
                    cast = finite_float(value, key)
                elif coerce is int:
                    cast = finite_int(value, key)
                elif coerce is bool:
                    if not isinstance(value, bool) and value not in (0, 1):
                        raise MifOpError("'%s' on a %s modifier is true or false, got %r. NOTHING was changed."
                                         % (key, mod_type, value))
                    cast = bool(value)
                else:
                    cast = coerce(value)
                if callable(target):
                    target(mod, cast)
                else:
                    setattr(mod, target, cast)
        except MifOpError:
            _restore_settings(mod, restore)
            raise
        except Exception as exc:  # noqa: BLE001
            _restore_settings(mod, restore)
            raise MifOpError("could not set '%s' on the %s modifier: %s. NOTHING was changed."
                             % (key, mod_type, exc))
        applied.append(key)

    # AN OFFSET OBJECT IMPLIES OBJECT OFFSET. Blender's Array reads offset_object only while
    # use_object_offset is on, and its default is relative offset alone (offset_type =
    # MOD_ARR_OFF_RELATIVE in DNA_modifier_defaults.h, v4.4.0). So offsetObject set a pointer that did
    # nothing: measured 2026-09-11, an ARRAY with offsetObject evaluated to the SAME shape hash as one
    # without it, while readBack reported the pointer as set. Asking for an offset object is asking for
    # object offset; an explicit useObjectOffset:false still wins, because it is in `settings`.
    if mod_type == "ARRAY" and settings.get("offsetObject") and "useObjectOffset" not in settings:
        mod.use_object_offset = True
        applied.append("useObjectOffset")
    return sorted(applied)


def _restore_settings(mod, restore):
    """Put every attribute back the way it was found. Never raises.

    A rollback that throws on its way out replaces a clear refusal with an obscure one and leaves
    the object half-written anyway, so each attribute is restored independently and a failure on one
    does not abandon the rest.
    """
    for attr, old in restore.items():
        try:
            setattr(mod, attr, old)
        except Exception:                                           # noqa: BLE001
            pass


# OBJECT TYPES WITH A MODIFIER STACK, and not only meshes. Measured 2026-09-16 on 3.6.23, 4.2.17, 4.4.0, 5.0.1
# and 5.2.1 by calling modifiers.new for every type on a fresh object of each kind, no addon loaded: a MESH takes
# 55; a CURVE and a FONT take the same 28 (ARRAY, BEVEL, SOLIDIFY, SCREW, SUBSURF, SIMPLE_DEFORM, NODES, WELD
# and twenty more); a SURFACE 26-27; a LATTICE the 12 deforming ones. On every build an unsupported pairing makes
# modifiers.new RETURN None rather than raise - which op_add_modifier already turns into a refusal.
#
# The three ops below required a MESH until then, so a chain of curve links could not be arrayed, a text could
# not be bent and a profile could not be solidified through the bridge, though Blender does all three. Found
# building the pcbuild hero scene, whose fan blades are filled 2D curves round a radial ARRAY.
#
# AND CURVES, POINTCLOUD AND VOLUME (2026-10-04): assign_node_group already put a NODES modifier on them, and these four
# ops refused them as having "no modifier stack". Measured on 3.6.23, 4.4.0 and 5.2.1: a Curves and a point cloud take
# NODES (and MESH_SEQUENCE_CACHE from 4.2 - measured 2026-10-08, 3.6 refuses); a volume takes NODES, MESH_TO_VOLUME and VOLUME_DISPLACE - and from 4.4
# modifiers.new also hands a volume every GREASE_PENCIL_* type and LINEART, which act on grease pencil only, so those
# are refused by name on anything that is not grease pencil.
#
# AND GREASE PENCIL (2026-10-11), both generations, measured on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1. A legacy GPENCIL
# object (3.6, 4.2) keeps its modifiers in obj.grease_pencil_modifiers - GpencilModifier, types GP_* plus SHRINKWRAP - and
# its obj.modifiers stays empty; a GREASEPENCIL object (4.3+) keeps GREASE_PENCIL_* and LINEART in obj.modifiers, where
# modifiers.new returns None for a mesh type (SUBSURF). _stack picks the collection, and these ops read and write it.
_MODIFIER_OBJECT_TYPES = ("MESH", "CURVE", "FONT", "SURFACE", "LATTICE", "CURVES", "POINTCLOUD", "VOLUME", "GPENCIL",
                          "GREASEPENCIL")

# The same modifier under each generation's name, for a refusal that names the one this build takes.
_GP_LEGACY_TO_V3 = {"GP_THICK": "GREASE_PENCIL_THICKNESS", "GP_LINEART": "LINEART", "SHRINKWRAP": "GREASE_PENCIL_SHRINKWRAP",
                    "GP_WEIGHT_ANGLE": "GREASE_PENCIL_VERTEX_WEIGHT_ANGLE",
                    "GP_WEIGHT_PROXIMITY": "GREASE_PENCIL_VERTEX_WEIGHT_PROXIMITY"}


def _gp_v3_name(legacy):
    return _GP_LEGACY_TO_V3.get(legacy) or ("GREASE_PENCIL_" + legacy[3:] if legacy.startswith("GP_") else None)


def _stack(obj):
    """The collection an object's modifiers live in: grease_pencil_modifiers on a legacy GPENCIL object (3.6, 4.2),
    obj.modifiers on everything else."""
    return obj.grease_pencil_modifiers if obj.type == "GPENCIL" else obj.modifiers


def _modifier_types(obj):
    """The modifier type identifiers this object's stack takes on this build."""
    base = bpy.types.GpencilModifier if obj.type == "GPENCIL" else bpy.types.Modifier
    return {t.identifier for t in base.bl_rna.properties["type"].enum_items}


def _modifier_move_op(obj):
    return bpy.ops.object.gpencil_modifier_move_to_index if obj.type == "GPENCIL" else \
        bpy.ops.object.modifier_move_to_index


def _modifier_host(name):
    """The object a modifier op works on - a mesh, or any other object Blender gives a modifier stack."""
    obj = get_object(name)
    if obj.type not in _MODIFIER_OBJECT_TYPES:
        raise MifOpError("object '%s' is a %s, which has no modifier stack - modifiers go on a %s. NOTHING was "
                         "changed." % (obj.name, obj.type, ", ".join(_MODIFIER_OBJECT_TYPES)))
    if obj.mode != "OBJECT":
        raise MifOpError("object '%s' is in %s mode - leave it first, because what the stack evaluates to is read "
                         "from object mode. NOTHING was changed." % (obj.name, obj.mode))
    return obj


def _triangle_overlap(obj):
    """How much more area the evaluated mesh's TRIANGLES cover than its FACES do. 1.0 for anything that renders as modelled.

    A FOLD HIDES INSIDE ONE FACE, which is why neither counts nor area can see it. A bevel wider than half a thin
    wall folds the concave n-gons a boolean_op leaves - the pcbuild fans' "half boxes", 2026-09-16 - and the folded
    face keeps its vertex count and its own area, because polygon.area is taken from the outline. Only when the face
    is split into triangles, as every renderer and exporter does, do the triangles overlap and cover the opening.
    MEASURED on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1: a fan frame with a 4.2 mm wall reads 1.000 bevelled at 0.5
    and 2 mm and when triangulated first, 2.201 at 4 mm and 2.486 at 5.5 mm; cubes bevelled up to 0.45 and cylinders
    read 1.000. A TOTAL, not the worst face: per-face ratios drift on small non-planar faces while the total holds.
    foreach_get into arrays, so a large mesh costs a buffer rather than a Python loop. None when there is nothing to
    measure.
    """
    try:
        deps = bpy.context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(deps)
        mesh = ev.to_mesh()
    except Exception:  # noqa: BLE001
        return None
    try:
        mesh.calc_loop_triangles()
        n_tri, n_poly = len(mesh.loop_triangles), len(mesh.polygons)
        if not n_tri or not n_poly:
            return None
        tri = array.array("f", [0.0]) * n_tri
        poly = array.array("f", [0.0]) * n_poly
        mesh.loop_triangles.foreach_get("area", tri)
        mesh.polygons.foreach_get("area", poly)
        total = sum(poly)
        return round(sum(tri) / total, 4) if total > 0 else None
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            ev.to_mesh_clear()
        except Exception:  # noqa: BLE001
            pass


# ABOVE THIS, a face's triangles overlap. Clean meshes measured exactly 1.000 on every build; folds measured above 2.
_FOLD_RATIO = 1.02


def _gp_evaluated(ev):
    """The evaluated Grease Pencil at the scene's frame: every layer's shown drawing, strokes, points and a position
    fingerprint. to_mesh() raises on a Grease Pencil object on every build ("Object does not have geometry data"), and
    the evaluated data carries what a GP modifier did - a Noise moves the points there in both generations (measured
    3.6.23 to 5.2.1, 2026-10-11)."""
    strokes, coords = 0, array.array("f")
    for layer in ev.data.layers:
        if ev.type == "GREASEPENCIL":
            fr = layer.current_frame()
            if fr is None or fr.drawing is None:
                continue
            attr = fr.drawing.attributes.get("position")
            strokes += len(fr.drawing.strokes)
            if attr is not None and len(attr.data):
                buf = array.array("f", [0.0]) * (3 * len(attr.data))
                attr.data.foreach_get("vector", buf)
                coords.extend(buf)
        else:
            fr = layer.active_frame
            if fr is None:
                continue
            for st in fr.strokes:
                strokes += 1
                buf = array.array("f", [0.0]) * (3 * len(st.points))
                st.points.foreach_get("co", buf)
                coords.extend(buf)
    return {"strokes": strokes, "points": len(coords) // 3, "shape": hashlib.sha256(coords.tobytes()).hexdigest()[:16]}


def _evaluated_counts(obj):
    """What the modifier stack currently PRODUCES, not what the mesh data holds.

    THE TWO ARE DIFFERENT AND THAT IS THE WHOLE POINT. obj.data is the mesh before any modifier;
    the depsgraph result is what renders and what exports. A modifier that empties the object
    changes the second and not the first.
    """
    try:
        deps = bpy.context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(deps)
        if obj.type in ("GPENCIL", "GREASEPENCIL"):
            return _gp_evaluated(ev)
        if obj.type in ("CURVES", "POINTCLOUD"):
            # POINTS, NOT A MESH (2026-10-08). to_mesh() raises on a Curves object, so both sides of the comparison
            # came back {} and a MESH_SEQUENCE_CACHE that filled an empty Curves with twelve cached points was
            # reported "changed nothing about the evaluated mesh" (measured on 4.4.0). The evaluated data's
            # position attribute is what renders; it carries the same fingerprint.
            data = ev.data
            attr = data.attributes.get("position")
            n = len(attr.data) if attr is not None else 0
            buf = array.array("f", [0.0]) * (3 * n)
            if n:
                attr.data.foreach_get("vector", buf)
            out = {"points": n, "shape": hashlib.sha256(buf.tobytes()).hexdigest()[:16]}
            if obj.type == "CURVES":
                out["curves"] = len(data.curves)
            return out
        mesh = ev.to_mesh()
        n = len(mesh.vertices)
        # A COORDINATE FINGERPRINT, NOT JUST COUNTS, and this was got wrong here FIRST.
        # The version of this helper written an hour before apply_modifier was fixed for
        # exactly this compared counts alone, so a SHRINKWRAP pointed at a real target
        # reported evaluatedUnchanged:true - a deforming modifier moves vertices and
        # changes no count. Knowing about the trap did not stop me walking into it in the
        # next function, which is the argument for the fingerprint being in the helper
        # rather than remembered at each call site.
        buf = array.array("f", [0.0]) * (3 * n)
        mesh.vertices.foreach_get("co", buf)
        out = {"verts": n, "edges": len(mesh.edges), "faces": len(mesh.polygons),
               "shape": hashlib.sha256(buf.tobytes()).hexdigest()[:16]}
        ev.to_mesh_clear()
        return out
    except Exception:  # noqa: BLE001
        # A REPORTING FIELD MUST NOT BE ABLE TO BREAK THE OP. If the depsgraph cannot be read here,
        # the answer is "unknown" rather than a raised exception from a diagnostic.
        return {}


# COPY A MODIFIER, OR A STACK, TO OTHER OBJECTS (2026-10-05). A 3-modifier stack onto 6 parts was 1 read and 18 adds,
# the caller dropping read-only fields, pointers, vertex groups the target lacks and bind-type modifiers by hand. Native
# RNA copying, so no operator and no context override: every writable property, in the source's stack order.
_COPY_SKIP = {"rna_type", "name", "type", "is_active", "is_override_data"}
# their state lives in a struct of their own (cloth settings, a fluid role, a particle system) - each has its own op
_COPY_REFUSED_TYPES = {"CLOTH", "SOFT_BODY", "COLLISION", "FLUID", "DYNAMIC_PAINT", "PARTICLE_SYSTEM",
                       "PARTICLE_INSTANCE", "EXPLODE"}
# a bind made on the source is data of the source - the copy starts unbound
_COPY_BIND_TYPES = {"SURFACE_DEFORM", "MESH_DEFORM", "LAPLACIANDEFORM"}


def _copy_modifier_props(src, dst, target):
    """Every writable property of modifier src onto dst (on object target) -> [{property, reason}] it left."""
    skipped = []
    for p in src.bl_rna.properties:
        ident = p.identifier
        if ident in _COPY_SKIP or p.is_readonly:
            continue
        val = getattr(src, ident)
        if p.type == "POINTER" and val is not None and val == target:
            skipped.append({"property": ident, "reason": "it points at '%s' itself, which a modifier on '%s' cannot "
                            "(left empty)" % (target.name, target.name)})
            continue
        if p.type == "STRING" and val and "vertex_group" in ident and target.vertex_groups.get(val) is None:
            skipped.append({"property": ident, "reason": "'%s' has no vertex group '%s' - make it, then set_modifier"
                            % (target.name, val)})
            continue
        if p.type == "STRING" and val and ident in ("layer", "layer_filter") \
                and target.type in ("GPENCIL", "GREASEPENCIL") and val not in [
                    (lay.name if target.type == "GREASEPENCIL" else lay.info) for lay in target.data.layers]:
            skipped.append({"property": ident, "reason": "'%s' has no Grease Pencil layer '%s' - the filter would match "
                            "nothing (left empty)" % (target.name, val)})
            continue
        if p.type == "STRING" and val and ident == "uv_layer" and target.type == "MESH" \
                and target.data.uv_layers.get(val) is None:
            skipped.append({"property": ident, "reason": "'%s' has no UV layer '%s'" % (target.name, val)})
            continue
        try:
            setattr(dst, ident, tuple(val) if getattr(p, "array_length", 0) > 0 else val)
        except Exception as exc:                                       # noqa: BLE001 - named, never silent
            skipped.append({"property": ident, "reason": str(exc)[:200]})
    # THE CURVE STRUCTS ARE READ-ONLY POINTERS - a custom bevel profile, a falloff curve - named when they are in use
    if getattr(src, "profile_type", None) == "CUSTOM":
        skipped.append({"property": "custom_profile", "reason": "the custom profile's points are not copied"})
    if getattr(src, "falloff_type", None) == "CURVE" and hasattr(src, "falloff_curve"):
        skipped.append({"property": "falloff_curve", "reason": "the falloff curve's points are not copied"})
    if src.type == "NODES":
        sin = getattr(getattr(src, "properties", None), "inputs", None)
        din = getattr(getattr(dst, "properties", None), "inputs", None)
        if sin is not None and din is not None:
            for key in sin.keys():
                for k in sin[key].keys():
                    din[key][k] = sin[key][k]
        else:
            for k in src.keys():
                dst[k] = src[k]
        target.update_tag()
    return skipped


# A SOLIDIFY ON A CLOSED MESH (decided 2026-11-02; measured on 5.2.1, ledger). It has no edge to thicken from, so it
# makes a second closed shell inside the first, facing inward, where no outside view reaches it. A cube of 12 triangles
# becomes 24 and looks exactly the same at offset -1, and mesh_quality gameReady fails it as a closed part facing inward. At
# offset 1 the surface grows, and SHRINK_FATTEN makes that same surface (a 2 m cube: x 1.0289, volume 8.71301, both ways)
# at 12 triangles, passing. So what a person gets by accident is doubled triangles, and it needs saying before it lands:
# refused with both ways on. The cavity is a real thing to want, a wall to cut a window through, a shell to print, and a cut
# through it joins the two shells into one that passes, so it stays one explicit word away. Dropping the inner shell
# instead would hand back the input at offset -1 and what SHRINK_FATTEN already does at offset 1. Measured on what the stack
# evaluates to rather than asked of the object, so an open sheet, use_rim_only and a part that was inside out already
# are not caught.
_SOLIDIFY_CLOSED = (
    "'%s' is closed, so a Solidify has no edge to thicken from: it makes %d second shell(s) inside the surface, facing inward, "
    "where nothing outside can see them, %d triangles become %d, and mesh_quality gameReady fails the result as a closed "
    "part facing inward. To make it bigger, use transform_selection with operation SHRINK_FATTEN and distance: the same "
    "outer surface as offset 1, with no second shell. To make a hollow you will cut open or print, pass hollow true: a "
    "boolean_op through the wall then joins the shells into one, which passes gameReady. NOTHING was changed.")
_SOLIDIFY_HOLLOW_NOTE = (
    "hollow: %d inner shell(s) that face inward. mesh_quality gameReady fails this as a closed part facing inward until a "
    "boolean_op through the wall joins the shells. Do not recalculate its normals: that points the inner shell outward, "
    "passes, and the cavity is gone for any later cut.")


def _inward_parts(obj):
    """(closed parts facing inward, triangles) in what obj's stack evaluates to now; (0, 0) when it evaluates to no mesh."""
    from .ops_mesh import _winding_scan   # noqa: PLC0415 - gameReady's own measure, so this cannot disagree with it
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    try:
        mesh = ev.to_mesh()
    except RuntimeError:
        return 0, 0
    if mesh is None:
        return 0, 0
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.edges.index_update()
        bm.faces.index_update()
        tris = sum(len(f.verts) - 2 for f in bm.faces)
        return len(_winding_scan(bm)[1]), tris
    finally:
        bm.free()
        ev.to_mesh_clear()


def _copy_modifiers(params):
    """add_modifier's copyFrom / to path - see op_add_modifier."""
    tail = "NOTHING was changed."
    extra = [k for k in ("object", "name", "type", "modifier", "settings") if params.get(k) is not None]
    if extra:
        raise MifOpError("copyFrom copies modifiers that exist; %s belong to adding a new one - drop them. %s"
                         % (", ".join(extra), tail))
    spec = params.get("copyFrom")
    if not isinstance(spec, dict) or set(spec) - {"object", "modifier", "all"} or not spec.get("object"):
        raise MifOpError("copyFrom is {object, modifier | all: true}, got %r. %s" % (spec, tail))
    if (spec.get("modifier") is None) == (not spec.get("all")):
        raise MifOpError("copyFrom takes one of modifier (a name) or all: true. %s" % tail)
    src = _modifier_host(str(spec["object"]))
    if spec.get("modifier") is not None:
        mod = _stack(src).get(str(spec["modifier"]))
        if mod is None:
            raise MifOpError("'%s' has no modifier '%s' (it has %s). %s" % (src.name, spec["modifier"], ", ".join(
                m.name for m in _stack(src)) or "none", tail))
        mods = [mod]
    else:
        mods = list(_stack(src))
        if not mods:
            raise MifOpError("'%s' has no modifiers to copy. %s" % (src.name, tail))
    bad = sorted({m.type for m in mods} & _COPY_REFUSED_TYPES)
    if bad:
        raise MifOpError("%s keep their state in a struct of their own (the simulation's settings) and are made by "
                         "their own ops (add_cloth, add_particles and the rest), not copied here. %s"
                         % (", ".join(bad), tail))
    names = params.get("to")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise MifOpError("to is the list of objects to copy onto. %s" % tail)
    if len(set(names)) != len(names):
        raise MifOpError("to names an object twice. %s" % tail)
    targets = [_modifier_host(n) for n in names]
    # A GREASE PENCIL STACK GOES ONLY TO GREASE PENCIL OF THE SAME GENERATION (2026-10-11): a legacy GPENCIL object's
    # GP_* modifiers live in grease_pencil_modifiers and a 4.3+ object's GREASE_PENCIL_* ones in obj.modifiers, and
    # neither type name exists on the other stack (or on a mesh).
    gp = ("GPENCIL", "GREASEPENCIL")
    mixed = [t.name for t in targets if (t.type in gp or src.type in gp) and t.type != src.type]
    if mixed:
        raise MifOpError("'%s' is a %s, and a Grease Pencil modifier stack copies only between Grease Pencil objects "
                         "of one generation - %s %s not. %s" % (src.name, src.type, ", ".join(mixed),
                                                               "is" if len(mixed) == 1 else "are", tail))
    if any(t == src for t in targets):
        raise MifOpError("'%s' is the source - copying its stack onto itself is not a copy. %s" % (src.name, tail))
    for t in targets:
        require_editable(t, "copy modifiers onto")
    index = take_int(params, "index", default=None)
    if index is not None:
        worst = min(len(_stack(t)) for t in targets)
        if not 0 <= index <= worst:
            raise MifOpError("index %d is outside a target's stack (0-%d). %s" % (index, worst, tail))
    made, rows = [], []
    hollow = take_bool(params, "hollow", default=False)
    if hollow and not any(m.type == "SOLIDIFY" for m in mods):
        raise MifOpError("hollow says what a Solidify may make on a closed mesh, and this copies %s. %s"
                         % (", ".join(sorted({m.type for m in mods})), tail))
    try:
        for t in targets:
            row = {"object": t.name, "made": [], "skipped": [], "needsBind": []}
            inward_before = _inward_parts(t) if any(m.type == "SOLIDIFY" for m in mods) else None
            for k, m in enumerate(mods):
                new = _stack(t).new(name=m.name, type=m.type)
                if new is None:
                    raise MifOpError("Blender does not take a %s modifier on '%s' (a %s). %s"
                                     % (m.type, t.name, t.type, tail))
                made.append((t, new.name))
                for s in _copy_modifier_props(m, new, t):
                    row["skipped"].append(dict(s, modifier=new.name))
                if m.type in _COPY_BIND_TYPES:
                    row["needsBind"].append(new.name)
                if index is not None:
                    snap = selection_snapshot()
                    try:
                        select_only([t])
                        _modifier_move_op(t)(modifier=new.name, index=index + k)
                    finally:
                        selection_restore(snap)
            if inward_before is not None:
                inward_after = _inward_parts(t)
                row["innerShells"] = max(0, inward_after[0] - inward_before[0]) or None
                if row["innerShells"] and not hollow:
                    raise MifOpError(_SOLIDIFY_CLOSED % (t.name, row["innerShells"], inward_before[1], inward_after[1]))
            for nm in [n for tt, n in made if tt == t]:
                row["made"].append({"name": nm, "type": _stack(t)[nm].type,
                                    "index": list(_stack(t)).index(_stack(t)[nm])})
            rows.append(row)
    except Exception:
        for t, nm in made:
            m = _stack(t).get(nm)
            if m is not None:
                _stack(t).remove(m)
        raise
    for row in rows:
        row["skipped"] = row["skipped"] or None
        row["needsBind"] = row["needsBind"] or None
        row["stack"] = [m.name for m in _stack(bpy.data.objects[row["object"]])]
    return {"copyFrom": {"object": src.name, "modifiers": [m.name for m in mods]}, "targets": rows,
            "modifiersMade": len(made),
            "bindNote": ("a bound modifier's bind is data of its own object: bind_modifier binds each copy named in "
                         "needsBind." if any(r["needsBind"] for r in rows) else None)}


def op_add_modifier(params):
    """Add a modifier to an object's stack - a mesh, curve, text, surface or lattice - the write half of list_modifiers.

    THE GENERAL FORM of the edits this addon otherwise hardcodes one at a time.
    decimate_mesh already builds, configures and applies a DECIMATE modifier
    internally; this exposes the same mechanism for the other types, so mirror,
    solidify, subsurf, weighted-normal and the rest stop each needing their own op.

    IT DOES NOT APPLY. Adding and applying are separate on purpose: a modifier left
    in the stack still changes what export_mesh writes (use_mesh_modifiers is on),
    so the caller may well want it live and unapplied. apply_modifier is the
    separate, destructive step, and list_modifiers is how you check what is stacked
    before spending an export finding out.

    SETTINGS TAKE BOTH SPELLINGS, since 2026-09-21. The curated camelCase names - the
    handful that decide what an export produces, mirroring _MODIFIER_FIELDS on the read
    side - and ANY writable RNA property by its identifier (foam_layer_name, limit_method,
    use_axis), through _write_modifier_settings, the path set_modifier uses. This op used
    to curate only, on the grounds that describing every property of 100+ types by hand
    was effort spent on the wrong problem; write_rna_property reads bl_rna at call time
    and describes nothing by hand, so the grounds went, and add-then-set is one call.
    A setting that refuses removes the new modifier again - NOTHING was changed.

    A SOLIDIFY THAT WOULD MAKE A SECOND SHELL INSIDE A CLOSED MESH IS REFUSED unless hollow is true (_SOLIDIFY_CLOSED says why).
    """
    reject_unknown(params, ("object", "name", "type", "modifier", "settings", "index", "copyFrom", "to", "hollow"),
                   "add_modifier", {
        "apply": "adding and applying are SEPARATE on purpose. A modifier left in the stack still "
                 "changes what export_mesh writes, so adding one is usually all you want; "
                 "apply_modifier is the destructive step and it is a different call.",
        "applyNow": "see 'apply'.",
        "show": "the visibility toggles are settings - pass them inside `settings`, which is "
                "checked against the modifier type and refuses a name that type does not have.",
        "value": "modifier settings go in `settings` as an object, because which names are valid "
                 "depends on the modifier TYPE and the refusal lists that type's own.",
    })
    if params.get("copyFrom") is not None or params.get("to") is not None:
        if params.get("copyFrom") is None:
            raise MifOpError("to names the objects copyFrom copies onto - pass copyFrom {object, modifier | all}. "
                             "NOTHING was changed.")
        return _copy_modifiers(params)
    obj = _modifier_host(take(params, "object", "name", required=True, kind=str))
    mod_type = str(take(params, "type", required=True, kind=str)).upper()

    valid = _modifier_types(obj)
    # THE OTHER GENERATION'S NAME, named back in this build's spelling: a legacy GPENCIL object takes GP_NOISE where a
    # 4.3+ GREASEPENCIL takes GREASE_PENCIL_NOISE, and 4.2's Modifier enum already lists the GREASE_PENCIL_* names
    # while its Grease Pencil objects are still legacy.
    if obj.type == "GPENCIL" and mod_type not in valid:
        legacy = next((k for k in sorted(valid) if _gp_v3_name(k) == mod_type), None)
        if legacy is not None:
            raise MifOpError("'%s' is a legacy Grease Pencil object (Blender %s), whose modifiers are named %s here - %s "
                             "is the 4.3+ name. NOTHING was changed." % (obj.name, bpy.app.version_string, legacy,
                                                                         mod_type))
    if obj.type == "GREASEPENCIL" and (mod_type.startswith("GP_") or mod_type == "SHRINKWRAP"):
        v3 = _gp_v3_name(mod_type)
        if v3 in valid:
            raise MifOpError("%s is the legacy (3.6, 4.2) Grease Pencil name; on Blender %s '%s' takes %s. NOTHING was "
                             "changed." % (mod_type, bpy.app.version_string, obj.name, v3))
    if mod_type not in valid:
        raise MifOpError(
            "unknown modifier type '%s' for this Blender. NOTHING was changed. Valid types: %s. Every "
            "type takes its settings by RNA name; these also take curated camelCase names: %s."
            % (mod_type, ", ".join(sorted(valid)), ", ".join(sorted(_MODIFIER_WRITES))))

    # AN UNCURATED KEY IS WRITTEN NOW, NOT REFUSED (2026-09-21). Until then any name outside _MODIFIER_WRITES was
    # refused here, before the modifier existed, pointing at set_modifier. The recommendation was taken to let
    # this op take the generic path too: the names are checked against the new modifier's own bl_rna by
    # write_rna_property, and a refusal removes the modifier in the except below - so it still leaves nothing.
    # Only the SHAPE is checked here, where a refusal costs nothing.
    asked_settings = params.get("settings")
    if asked_settings is not None and not isinstance(asked_settings, dict):
        raise MifOpError("settings must be an object of {name: value} - curated camelCase names or the modifier's "
                         "own RNA identifiers. NOTHING was changed.")

    mod_name = take(params, "modifier", default=None) or mod_type.title()
    if mod_name in _stack(obj):
        raise MifOpError("'%s' already has a modifier named '%s'. Names must be unique on an "
                         "object. NOTHING was changed." % (obj.name, mod_name))

    hollow = take_bool(params, "hollow", default=False)
    if hollow and mod_type != "SOLIDIFY":
        raise MifOpError("hollow says what a Solidify may make on a closed mesh, and this is a %s. NOTHING was changed."
                         % mod_type)
    inward_before = _inward_parts(obj) if mod_type == "SOLIDIFY" else None

    before = [m.name for m in _stack(obj)]
    # WHAT THE STACK PRODUCES NOW, so what it produces after is comparable. See the postcondition
    # at the end of this function for why a modifier's own fields cannot answer this.
    if (mod_type.startswith("GREASE_PENCIL_") or mod_type == "LINEART") and obj.type not in ("GREASEPENCIL",
                                                                                              "GPENCIL"):
        raise MifOpError("a %s modifier acts on grease pencil strokes, and '%s' is a %s - Blender %s may accept it there "
                         "and it does nothing. NOTHING was changed." % (mod_type, obj.name, obj.type,
                                                                       bpy.app.version_string))
    evaluated_before = _evaluated_counts(obj)
    overlap_before = _triangle_overlap(obj)
    mod = _stack(obj).new(name=mod_name, type=mod_type)
    if mod is None:
        # WHAT None MEANS DEPENDS ON THE OBJECT. On a mesh it is rare; on a curve or text it is the ordinary answer
        # for the 27 types that need mesh data - DATA_TRANSFER, the vertex-weight family, WEIGHTED_NORMAL - and
        # "refused" alone reads like a fault in the call rather than a limit of the object.
        raise MifOpError("Blender does not support a %s modifier on a %s object ('%s') - modifiers.new returned "
                         "none.%s NOTHING was changed."
                         % (mod_type, obj.type, obj.name,
                            "" if obj.type == "MESH" else
                            " A curve or text takes 28 types - generating ones like ARRAY, BEVEL, SOLIDIFY, SCREW, "
                            "SUBSURF, WELD and deforming ones like SIMPLE_DEFORM, CAST, CURVE, LATTICE - but not "
                            "the ones that need mesh data (vertex groups, data transfer, custom normals); a "
                            "lattice takes only the deforming ones; a Curves or point cloud object takes NODES (and "
                            "MESH_SEQUENCE_CACHE from 4.2), a volume NODES, MESH_TO_VOLUME and VOLUME_DISPLACE; a "
                            "Grease Pencil object takes the GREASE_PENCIL_* types and LINEART (GP_* on 3.6 and 4.2)."))

    applied = []
    applied_values = {}
    try:
        settings = take(params, "settings", default=None)
        if settings:
            # settingsApplied stays a LIST OF NAMES, as it has always been. The VALUES, read back off the modifier
            # by _write_modifier_settings, go out as settingsReadBack: readBack below is list_modifiers' curated
            # row, which shows no setting outside the curated table - so a generic one (a BEVEL's limit_method, a
            # WAVE's height) was written and its value reported nowhere (measured on 5.0.1, 2026-09-21).
            applied_values = _write_modifier_settings(obj, mod, settings)
            applied = list(applied_values)

        index = take_int(params, "index", default=None)
        if index is not None:
            if index < 0 or index >= len(_stack(obj)):
                raise MifOpError("index %d is outside the stack (0-%d). NOTHING was changed."
                                 % (index, len(_stack(obj)) - 1))
            snap = selection_snapshot()
            try:
                select_only([obj])
                _modifier_move_op(obj)(modifier=mod.name, index=index)
            finally:
                selection_restore(snap)
        inner_shells = 0
        if inward_before is not None:
            inward_after = _inward_parts(obj)
            inner_shells = max(0, inward_after[0] - inward_before[0])
            if inner_shells and not hollow:
                raise MifOpError(_SOLIDIFY_CLOSED % (obj.name, inner_shells, inward_before[1], inward_after[1]))
    except Exception:
        # CLEAN UP ON FAILURE, the same discipline decimate_mesh uses: a half-configured modifier
        # left in the stack silently changes every later export, and the caller was told NOTHING
        # was changed.
        try:
            _stack(obj).remove(mod)
        except Exception:  # noqa: BLE001
            pass
        raise

    # READ BACK through the same describer list_modifiers uses, so add and read speak one
    # vocabulary and a settings value that did not take is visible rather than assumed.
    row = _modifier_dict(_stack(obj)[mod.name])

    # WHAT THE MODIFIER ACTUALLY DOES TO THE OBJECT, evaluated rather than inferred from its fields.
    #
    # A MASK modifier added with no vertex group masks EVERYTHING: the object evaluates to zero
    # vertices and disappears from the viewport and from every export, while this op returned a
    # clean success. Measured on 4.4 - a default cube, 8 vertices, evaluates to 0 the moment a bare
    # MASK is added. Nothing in the modifier's own properties says so, which is why reading them
    # back through _modifier_dict was not enough.
    #
    # The same measurement answers the quieter half. Most modifier types need to be POINTED at
    # something - a shrinkwrap target, a lattice, a hook, a curve - and until they are they sit in
    # the stack doing nothing. An inert modifier and a working one are identical in every field.
    evaluated_after = _evaluated_counts(obj)
    count_key = "points" if "points" in evaluated_after else "verts"
    became_empty = (evaluated_before.get(count_key, 0) > 0
                    and evaluated_after.get(count_key, 1) == 0)
    # UNKNOWN IS NOT UNCHANGED: an evaluation that could not be read on both sides ({} == {}) proves nothing.
    inert = evaluated_before == evaluated_after if evaluated_after else None
    overlap_after = _triangle_overlap(obj)
    # ONLY A FOLD THIS MODIFIER MADE. A mesh that already overlapped is not news about this call.
    folded = (overlap_after is not None and overlap_after > _FOLD_RATIO
              and (overlap_before is None or overlap_before <= _FOLD_RATIO))

    return {
        "object": obj.name,
        "modifier": mod.name,
        "type": mod_type,
        "settingsApplied": applied,
        "stackBefore": before,
        "stackAfter": [m.name for m in _stack(obj)],
        "evaluatedBefore": evaluated_before,
        "evaluatedAfter": evaluated_after,
        # THE FIELD THAT MATTERS. True means the object is GONE from every render and every export
        # while still sitting in the outliner looking normal.
        "evaluatesToEmpty": became_empty,
        "evaluatedUnchanged": inert,
        # 1.0 renders as modelled; above it, a face's triangles overlap. See _triangle_overlap.
        "triangleOverlapBefore": overlap_before,
        "triangleOverlapAfter": overlap_after,
        "facesFold": folded,
        "effectWarning": (
            "'%s' evaluates to ZERO vertices with this modifier on it - the object will not render "
            "and will export as an empty mesh, while still looking present in the outliner. A MASK "
            "with no vertex group does exactly this. Configure it or remove it."
            % obj.name) if became_empty else (
            "with this modifier on, '%s' has faces that FOLD over themselves: their triangles cover %.2fx the "
            "faces' own area, which renders as geometry across an opening while every count reads normal. %s"
            % (obj.name, overlap_after,
               "A BEVEL wider than half a thin wall does this to the concave faces boolean_op leaves - measured "
               "clean at 2 mm and folded at 4 mm on a 4.2 mm wall. Add a TRIANGULATE modifier before the bevel, "
               "or narrow its width." if mod_type == "BEVEL" else
               "Check the settings - a face this modifier bent or offset now crosses itself.")) if folded else (
            "this modifier changed nothing about the evaluated mesh. Most types have to be POINTED "
            "at something - a target, a lattice, a curve, a vertex group - before they do anything, "
            "and an unconfigured one is indistinguishable from a working one in every field here."
            if inert else None),
        "index": list(_stack(obj)).index(_stack(obj)[mod.name]),
        "settingsReadBack": applied_values,
        "readBack": row,
        "note": ("added to the stack, NOT applied - it will still affect what export_mesh writes "
                 "unless useMeshModifiers is off. Call apply_modifier to bake it into the mesh."),
        "innerShells": inner_shells or None,
        "hollowNote": (_SOLIDIFY_HOLLOW_NOTE % inner_shells) if inner_shells else None,
    }


def op_remove_modifier(params):
    """Remove a modifier from the stack WITHOUT applying it - the geometry is untouched."""
    reject_unknown(params, ("object", "name", "modifier"), "remove_modifier")
    obj = _modifier_host(take(params, "object", "name", required=True, kind=str))
    mod_name = take(params, "modifier", required=True, kind=str)

    mod = _stack(obj).get(mod_name)
    if mod is None:
        raise MifOpError(
            "'%s' has no modifier named '%s'. It has %d: %s. NOTHING was changed."
            % (obj.name, mod_name, len(_stack(obj)),
               ", ".join(m.name for m in _stack(obj)) or "(none)"))

    # A LIBRARY MODIFIER ON AN OVERRIDE cannot be removed - it reads gone and returns on reload (see
    # ops_common.refuse_override_data) - and a linked object cannot be edited at all.
    refuse_override_data(mod, obj, "modifier")
    require_editable(obj, "remove a modifier from")
    before = [m.name for m in _stack(obj)]
    # MESH DATA ONLY ON A MESH. A curve or text keeps splines or characters, and mesh_counts reads vertices.
    counts_before = mesh_counts(obj) if obj.type == "MESH" else None
    # WHAT THE STACK PRODUCED, which is the half that can actually change here.
    evaluated_before = _evaluated_counts(obj)
    _stack(obj).remove(mod)
    after = [m.name for m in _stack(obj)]

    # POSTCONDITION, read back rather than assumed - modifiers.remove returns nothing.
    if mod_name in after:
        raise MifOpError("Blender reported no error but '%s' is still on the stack." % mod_name)

    evaluated_after = _evaluated_counts(obj)

    return {
        "object": obj.name,
        "modifier": mod_name,
        "removed": True,
        "stackBefore": before,
        "stackAfter": after,
        # TRUE BY CONSTRUCTION, and kept only because it is documented. modifiers.remove
        # never touches obj.data, so this compares the original mesh against itself and
        # cannot report anything else - it is a restatement of the note below, not
        # evidence. The field that can actually be wrong is the evaluated one.
        # None on a curve, text, surface or lattice: there is no mesh data to compare, and False would claim a change.
        "meshUnchanged": (mesh_counts(obj) == counts_before) if counts_before is not None else None,
        "evaluatedBefore": evaluated_before,
        "evaluatedAfter": evaluated_after,
        # WHAT THE REMOVAL ACTUALLY DID. The mesh data is untouched either way; what moved
        # is what RENDERS and what export_mesh writes. False here means the modifier was
        # contributing nothing - it was inert, or already disabled - which is worth knowing
        # before concluding that removing it fixed something.
        "evaluatedChanged": bool(evaluated_before) and evaluated_before != evaluated_after,
        "note": ("removed WITHOUT applying - the mesh data is untouched, and whatever this modifier "
                 "was contributing to the viewport and to export_mesh is gone."),
    }


_SETMOD_KEYS = {"object", "name", "modifier", "settings", "index", "move", "adaptive"}
_MOVES = ("UP", "DOWN", "TOP", "BOTTOM")


def _move_modifier(obj, name, index):
    """Move modifier `name` to `index` with Blender's own operator (as add_modifier places one), and say whether it went:
    the operator can decline a move - a modifier that needs original data cannot sit below one that changes it - and
    then reports CANCELLED without raising, so the stack is read back."""
    snap = selection_snapshot()
    try:
        select_only([obj])
        try:
            _modifier_move_op(obj)(modifier=name, index=index)
        except RuntimeError as exc:
            return str(exc).strip()
    finally:
        selection_restore(snap)
    at = [m.name for m in _stack(obj)].index(name)
    return None if at == index else "Blender kept it at %d" % at

# NOT SETTINGS, or not this op's to set. name is the key every other modifier op finds it by; type is
# fixed for a modifier's life; is_active, show_expanded and the override/uid bookkeeping are UI and file
# state. node_group belongs to assign_node_group, which also carries the group's inputs.
_MODIFIER_DENY = {"name", "type", "is_active", "show_expanded", "is_override_data", "persistent_uid",
                  "node_group"}

# STRING SETTINGS THAT NAME SOMETHING ON THE OBJECT. Blender stores any text in these and a name that
# matches nothing is not an error - it simply selects nothing, which on a MASK empties the object. The
# same check _apply_modifier_settings makes for the curated vertexGroup keys, applied to the RNA names.
_NAMES_A_UV_MAP = ("uv_layer", "uvlayer_name")


def _read_rna_value(value):
    """A setting as JSON: a datablock by name, an array as a list, a float rounded.

    NAMED _read_ ON PURPOSE: it formats a value Blender handed back, and audit_unguarded_numbers exempts that
    prefix by convention - as _rna_readback it read `value` as caller input and turned the gate red.
    """
    if hasattr(value, "bl_rna") and hasattr(value, "name"):
        return value.name
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "__len__") and not isinstance(value, str):
        return [round(float(v), 6) if isinstance(v, float) else v for v in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def _rna_copy(value):
    """A detached copy - an array read off a modifier is a LIVE view, and restoring from it restores nothing."""
    if isinstance(value, (set, frozenset)):
        return set(value)
    if hasattr(value, "__len__") and not isinstance(value, str) and not hasattr(value, "bl_rna"):
        return tuple(value)
    return value


# A FLUID MODIFIER'S SETTINGS ARE ONE LEVEL DOWN. Its one writable property of its own is fluid_type, a role selector;
# everything that decides what a smoke, fire or liquid does - whether smoke dissolves, which domain walls it collides
# with, a flow's fuel and temperature, the flame's vorticity, the time scale - hangs off domain_settings, flow_settings
# or effector_settings, whichever the role is, and only that one exists. add_fluid writes a curated few at creation
# and nothing reached the rest: measured 2026-09-21 building a fire showcase, a GAS domain whose smoke cannot dissolve
# fills from its ceiling down and renders as a grey block, with no op able to say otherwise.
_FLUID_ROLE_ATTR = {"DOMAIN": "domain_settings", "FLOW": "flow_settings", "EFFECTOR": "effector_settings"}
_FLUID_ROLE_TYPE = {"DOMAIN": "FluidDomainSettings", "FLOW": "FluidFlowSettings", "EFFECTOR": "FluidEffectorSettings"}
# THE SIMULATION'S OWN GRIDS, NEVER READ by list_modifiers {rna: true}: on a domain with no simulation data the getter
# is an access violation that takes Blender down (density_grid, 4.4.0, measured 2026-09-21), and with data each is a
# resolution-cubed array. They are the simulation's output, not its settings. See ops_common.rna_properties' `skip`.
_FLUID_DATA_PROPS = frozenset({"density_grid", "velocity_grid", "flame_grid", "color_grid", "heat_grid",
                               "temperature_grid"})


def _fluid_role(mod):
    """(identifier, struct) of a FLUID modifier's live settings - domain_settings, flow_settings or effector_settings,
    whichever its fluid_type is - or (None, None) for any other modifier and for a fluid whose type is NONE."""
    if mod.type != "FLUID":
        return None, None
    attr = _FLUID_ROLE_ATTR.get(getattr(mod, "fluid_type", None))
    struct = getattr(mod, attr, None) if attr else None
    return (attr, struct) if struct is not None else (None, None)


def _fluid_split(obj, mod, generic):
    """Take a FLUID modifier's role-struct keys out of `generic` -> (role identifier, role struct, {key: value}).

    A key the modifier itself has stays in `generic` and is written there as before. A key that only the live role's
    struct has is moved out, to be written on that struct. Everything that cannot be written is refused HERE, before
    the first write: a key another role owns (a flow's fuel_amount sent to a DOMAIN) names that role, as add_fluid's
    wrong-role refusal does; fluid_type beside role keys is refused, because changing the role replaces the struct the
    other keys were meant for; and a key on neither says where a fluid's settings live."""
    role_attr, role = _fluid_role(mod)
    if role is None:
        return None, None, {}
    on_role = {k: generic.pop(k) for k in list(generic)
               if mod.bl_rna.properties.get(k) is None and role.bl_rna.properties.get(k) is not None}
    if on_role and "fluid_type" in generic:
        raise MifOpError("fluid_type and %s's own settings (%s) in one call - changing the role replaces %s, so the "
                         "other keys would land on a struct that no longer exists. Change the role first, then its "
                         "settings. NOTHING was changed." % (mod.fluid_type, ", ".join(sorted(on_role)), role_attr))
    for key in generic:
        if mod.bl_rna.properties.get(key) is not None:
            continue
        owner = next((t for t, cls in _FLUID_ROLE_TYPE.items() if t != mod.fluid_type
                      and getattr(bpy.types, cls, None) is not None
                      and getattr(bpy.types, cls).bl_rna.properties.get(key) is not None), None)
        if owner:
            raise MifOpError("'%s' is a fluid %s setting (%s), and '%s' on '%s' is a %s - its settings are %s. "
                             "NOTHING was changed." % (key, owner, _FLUID_ROLE_ATTR[owner], mod.name, obj.name,
                                                       mod.fluid_type, role_attr))
        writable = sorted(p.identifier for p in role.bl_rna.properties
                          if not p.is_readonly and not p.identifier.startswith("bl_") and p.identifier != "rna_type")
        raise MifOpError("modifier '%s' (FLUID, a %s) has no setting '%s' - not on the modifier and not on its %s, "
                         "which has: %s. NOTHING was changed." % (mod.name, mod.fluid_type, key, role_attr,
                                                                  ", ".join(writable) or "(none)"))
    return role_attr, role, on_role


# A CLOTH, COLLISION OR SOFT BODY MODIFIER'S SETTINGS ARE ONE LEVEL DOWN TOO, like a fluid's - but on fixed structs, not
# a role. A cloth's own writable properties are the stack toggles; its stiffness, sewing, shrink, quality and air damping
# are on `settings` (ClothSettings), its collision distances and self collision on `collision_settings`, and its bake
# range on `point_cache`. A collision modifier's thickness and friction are on its `settings` (CollisionSettings); a
# soft body's are on `settings` (SoftBodySettings, the same struct as obj.soft_body). set_modifier reached the
# modifier's own properties only, so every one of these needed raw Python while dressing a short film's garments
# (2026-10-02). Read off each build's RNA: on 5.2.1 no key is on two of a type's structs, and _substruct_split refuses
# one that is, naming both.
_MODIFIER_SUBSTRUCTS = {
    "CLOTH": ("settings", "collision_settings", "point_cache"),
    "COLLISION": ("settings",),
    "SOFT_BODY": ("settings", "point_cache"),
}


def _struct_keys(struct):
    return sorted(p.identifier for p in struct.bl_rna.properties
                  if not p.is_readonly and not p.identifier.startswith("bl_") and p.identifier != "rna_type")


def _substruct_split(obj, mod, generic):
    """Take a CLOTH, COLLISION or SOFT_BODY modifier's struct keys out of `generic` -> [(identifier, struct,
    {key: value})].

    Two spellings, both refused before the first write when they cannot land: NESTED, {"collision_settings":
    {"distance_min": 0.01}} - the struct named by its RNA identifier - and FLAT, {"distance_min": 0.01}, looked up on
    the type's structs when the modifier itself has no such property. A flat key on two structs is refused naming both
    (send it nested); a key on none lists every struct's writable settings; one setting named both ways is two answers
    to one question."""
    names = _MODIFIER_SUBSTRUCTS.get(mod.type, ())
    structs = {a: getattr(mod, a, None) for a in names}
    structs = {a: s for a, s in structs.items() if s is not None}
    if not structs:
        return []
    picked = {a: {} for a in structs}

    def put(attr, key, value):
        if key in picked[attr]:
            raise MifOpError("%s.%s is named twice in one call - nested and flat - which is two answers to one "
                             "question. Pass one. NOTHING was changed." % (attr, key))
        picked[attr][key] = value

    for key in list(generic):
        value = generic[key]
        if key in structs and isinstance(value, dict):
            if not value:
                raise MifOpError("'%s' is an empty object: name its settings, as {\"%s\": {\"<setting>\": value}}. "
                                 "NOTHING was changed." % (key, key))
            struct = structs[key]
            for sub, sub_value in value.items():
                if struct.bl_rna.properties.get(sub) is None:
                    raise MifOpError("modifier '%s' (%s) has no setting '%s' on its %s, which has: %s. NOTHING was "
                                     "changed." % (mod.name, mod.type, sub, key, ", ".join(_struct_keys(struct))))
                put(key, sub, sub_value)
            generic.pop(key)
            continue
        if mod.bl_rna.properties.get(key) is not None:
            continue
        owners = [a for a, s in structs.items() if s.bl_rna.properties.get(key) is not None]
        if len(owners) > 1:
            raise MifOpError("'%s' is a setting on both %s of modifier '%s' (%s) - send it nested, as {\"%s\": {\"%s\": "
                             "...}}. NOTHING was changed." % (key, " and ".join(owners), mod.name, mod.type,
                                                             owners[0], key))
        if not owners:
            listing = "; ".join("%s: %s" % (a, ", ".join(_struct_keys(s))) for a, s in structs.items())
            raise MifOpError("modifier '%s' (%s) has no setting '%s' - not on the modifier and not on its %s. "
                             "Its settings are %s. NOTHING was changed."
                             % (mod.name, mod.type, key, " or ".join(structs), listing))
        put(owners[0], key, generic.pop(key))

    # THE NAME-BEARING STRINGS, before the first write, as _write_modifier_settings checks the modifier's own: a
    # cloth's vertex_group_mass (its pin group), vertex_group_shrink and the rest store any text, and a name that
    # matches nothing pins or shrinks nothing.
    groups = {g.name for g in obj.vertex_groups} if hasattr(obj, "vertex_groups") else set()
    for attr, keys in picked.items():
        for key, value in keys.items():
            rna = structs[attr].bl_rna.properties.get(key)
            if (rna is not None and rna.type == "STRING" and "vertex_group" in key and isinstance(value, str)
                    and value and value not in groups):
                raise MifOpError("'%s' has no vertex group named '%s' for %s.%s - Blender stores any text there and a "
                                 "name that matches nothing selects no vertices. Groups: %s. NOTHING was changed."
                                 % (obj.name, value, attr, key, ", ".join(sorted(groups)[:25]) or "(none)"))
    return [(a, structs[a], keys) for a, keys in picked.items() if keys]


def _write_modifier_settings(obj, mod, settings):
    """Write `settings` onto a modifier that is on the stack - curated camelCase names through
    _apply_modifier_settings, any other writable RNA property through ops_common.write_rna_property. ALL OR
    NOTHING: every attribute it can touch is snapshotted first and put back on any refusal. Returns
    {name: value read back}.

    ONE PATH FOR BOTH CALLERS since 2026-09-21: set_modifier on a modifier that was already there, and
    add_modifier on the one it has just made - so a setting accepted at birth is written exactly the way it
    would be afterwards, and add-then-set is one call. Moved out of set_modifier unchanged.
    """
    mtype = mod.type
    table = _MODIFIER_WRITES.get(mtype, {})
    curated = {k: v for k, v in settings.items() if k in table}
    generic = {k: v for k, v in settings.items() if k not in table}

    def _attr_of(key):
        target = table[key][0]
        return target if isinstance(target, str) else getattr(target, "writes", None)

    curated_attrs = {k: _attr_of(k) for k in curated}
    twice = sorted(k for k in generic if k in curated_attrs.values())
    if twice:
        raise MifOpError("%s named twice - once by the curated name and once by its RNA identifier (%s), which "
                         "is two answers to one question. Pass one. NOTHING was changed."
                         % (", ".join("'%s'" % k for k in twice),
                            ", ".join("%s = %s" % (k, a) for k, a in sorted(curated_attrs.items())
                                      if a in twice)))
    role_attr, role, on_role = _fluid_split(obj, mod, generic)
    sub_writes = _substruct_split(obj, mod, generic)
    sub_snaps = []
    for _attr, struct, keys in sub_writes:
        held = {}
        for key in keys:
            rna = struct.bl_rna.properties.get(key)
            if rna is None or rna.is_readonly:
                continue
            try:
                held[key] = _rna_copy(getattr(struct, key))
            except AttributeError:
                continue
        sub_snaps.append((struct, held))
    role_snap = {}
    for key in on_role:
        rna = role.bl_rna.properties.get(key)
        if rna is None or rna.is_readonly:
            continue
        try:
            role_snap[key] = _rna_copy(getattr(role, key))
        except AttributeError:
            continue

    # SNAPSHOT EVERY ATTRIBUTE THIS CALL CAN TOUCH, before the first write. The ARRAY implication in
    # _apply_modifier_settings (offsetObject switches use_object_offset on) writes an attribute nobody
    # named, so it is snapshotted too.
    touched = [a for a in curated_attrs.values() if a] + [k for k in generic
                                                          if mod.bl_rna.properties.get(k) is not None]
    if mtype == "ARRAY":
        touched.append("use_object_offset")
    snap = {}
    for attr in touched:
        rna = mod.bl_rna.properties.get(attr)
        if rna is None or rna.is_readonly:
            continue
        try:
            snap[attr] = _rna_copy(getattr(mod, attr))
        except AttributeError:
            continue

    # THE NAME-BEARING STRINGS ARE CHECKED BEFORE THE FIRST WRITE. They need nothing but the object, and
    # below the curated writes a refusal here would be a promise the snapshot has to keep rather than one
    # that is simply true (audit_mutate_then_deny). Name sets built once (audit_quadratic_membership).
    vgroup_names = {g.name for g in obj.vertex_groups} if hasattr(obj, "vertex_groups") else set()
    uv_names = {u.name for u in obj.data.uv_layers} if obj.type == "MESH" else set()
    for key, value in generic.items():
        rna = mod.bl_rna.properties.get(key)
        if rna is None or rna.type != "STRING" or not isinstance(value, str) or not value:
            continue
        if "vertex_group" in key and value not in vgroup_names:
            have = sorted(vgroup_names)[:25]
            raise MifOpError(
                "'%s' has no vertex group named '%s'. Blender stores any text in %s and a name that "
                "matches nothing selects no vertices - on a MASK that empties the object. Groups: %s. "
                "NOTHING was changed." % (obj.name, value, key, ", ".join(have) or "(none)"))
        if key in _NAMES_A_UV_MAP and obj.type == "MESH" and value not in uv_names:
            have = sorted(uv_names)
            raise MifOpError(
                "'%s' has no UV map named '%s'. Blender stores any text in %s and a name that "
                "matches nothing falls back silently. UV maps: %s. NOTHING was changed."
                % (obj.name, value, key, ", ".join(have) or "(none)"))

    who = "modifier '%s' (%s)" % (mod.name, mtype)
    kind = "%s modifier" % mtype
    applied = {}
    try:
        if curated:
            written = _apply_modifier_settings(mod, curated, mtype)
            for key, attr in curated_attrs.items():
                applied[key] = _read_rna_value(getattr(mod, attr)) if attr else True
            # A WRITE THE CALLER DID NOT NAME IS REPORTED TOO. offsetObject switches use_object_offset on for an
            # ARRAY, and _apply_modifier_settings names it in what it returns - add_modifier's settingsApplied carried
            # it until the two ops shared this path, and the first certification after the move caught it going
            # missing (test_blender_authoring A702b, all five builds, 2026-09-21).
            for key in written:
                if key not in applied:
                    attr = _attr_of(key) if key in table else None
                    applied[key] = _read_rna_value(getattr(mod, attr)) if attr else True
        for key, value in generic.items():
            applied[key] = write_rna_property(
                mod, key, value, "NOTHING was changed.", who, kind, _MODIFIER_DENY,
                extra=(" A geometry-nodes modifier's group and its inputs belong to assign_node_group."
                       if key == "node_group" else
                       " Rename is not a setting - the name is how every modifier op finds it."
                       if key == "name" else ""))
        for key, value in on_role.items():
            applied[key] = write_rna_property(
                role, key, value, "NOTHING was changed.", "%s's %s" % (who, role_attr),
                "fluid %s's %s" % (mod.fluid_type, role_attr), frozenset())
        for attr, struct, keys in sub_writes:
            for key, value in keys.items():
                # DOTTED IN THE REPLY, collision_settings.distance_min, whichever spelling was sent: the struct it
                # landed on is the fact a caller reads back.
                applied["%s.%s" % (attr, key)] = write_rna_property(
                    struct, key, value, "NOTHING was changed.", "%s's %s" % (who, attr), "%s %s" % (mtype, attr),
                    frozenset())
        if sub_writes:
            obj.update_tag()
        if on_role:
            # SOME OF A FLUID'S SETTINGS HAVE NO UPDATE CALLBACK - cache_frame_start/end among them (rna_fluid.cc) - so a
            # write reached the original and not the evaluated copy that simulates. See ops_common.tag_fluid_domain.
            obj.update_tag()
    except Exception:
        # THE SAME RESTORE _apply_modifier_settings uses, not a second copy of it - and a call named
        # *restore* is what audit_mutate_then_deny recognises as an except handler keeping the promise.
        _restore_settings(mod, snap)
        _restore_settings(role, role_snap)
        for struct, held in sub_snaps:
            _restore_settings(struct, held)
        raise
    return applied


# WHAT BAKED GUIDES FREEZE. FOUND 2026-10-04 building the staged fluid bake: Blender's fluid panel greys Resolution
# Divisions and Time Scale once a domain's guides are baked, and the properties do not refuse - measured on 3.6.23,
# 4.4.0 and 5.2.1, a MODULAR gas domain with 5 baked guide frames took resolution_max 24 -> 28 and time_scale 1.0 -> 0.5
# through RNA; each write freed the data (rna_Fluid_datacache_reset) and KEPT the guiding folder and
# has_cache_baked_guide, so the next data bake follows guides made at the old resolution or time scale.
_GUIDE_FROZEN = ("resolution_max", "time_scale")


def _refuse_frozen_by_guides(obj, mod, settings):
    """Refuse resolution_max or time_scale on a fluid domain whose guides are baked, naming the free that unlocks them."""
    if getattr(mod, "type", None) != "FLUID" or getattr(mod, "fluid_type", None) != "DOMAIN":
        return
    ds = getattr(mod, "domain_settings", None)
    if ds is None or not bool(getattr(ds, "has_cache_baked_guide", False)):
        return
    asked = [k for k in _GUIDE_FROZEN if k in settings]
    nested = settings.get("domain_settings")
    if isinstance(nested, dict):
        asked += [k for k in _GUIDE_FROZEN if k in nested and k not in asked]
    if not asked:
        return
    raise MifOpError("'%s' on '%s' has baked guides, which freeze resolution_max (now %s) and time_scale (now %s) - "
                     "Blender's fluid panel greys both. A write here would free the baked data and KEEP the guides, so "
                     "the next data bake would follow guides made at the old value. To change %s: bake_physics "
                     "{object: '%s', freeStages: ['GUIDES']}, set it, then bake the guides again. NOTHING was changed."
                     % (mod.name, obj.name, getattr(ds, "resolution_max", "?"), getattr(ds, "time_scale", "?"),
                        " and ".join(asked), obj.name))


def op_set_modifier(params):
    """Change the settings of a modifier that is ALREADY on the stack - the half add_modifier left write-once.

    FOUND 2026-09-20, dogfooding an ocean through the typed ops on 5.0.1. add_modifier writes a curated
    handful of settings AT CREATION and nothing wrote them afterwards: re-tuning an Ocean's choppiness or a
    Bevel's width meant remove_modifier and add_modifier again, and any setting outside the curated table
    was unreachable for ever. The one that stopped the ocean: `foam_layer_name`. With useFoam on, the
    evaluated mesh carries the foam as a BYTE_COLOR corner attribute - named "Col" while the name is empty,
    "foam" once it is set (measured, read through the evaluated mesh) - and a material's Attribute node
    reads it BY NAME. No typed op could set the name, so no material could find the foam.

    GENERIC, LIKE set_constraint, and add_modifier's curation is left exactly as it is. Its docstring gives
    the reason it curates - describing every property of 100+ types by hand is effort on the wrong problem -
    and reading bl_rna at call time needs no description at all. So `settings` takes EITHER the curated
    camelCase names add_modifier and list_modifiers use (foamCoverage, vertexGroup, offsetObject - validated
    exactly as add_modifier validates them, through the same _apply_modifier_settings) OR any writable RNA
    property by its identifier (foam_layer_name, spectrum, limit_method, show_render), written through
    ops_common.write_rna_property, the dispatcher set_node_property uses: enums against their own items,
    finite numbers, strict booleans, a datablock pointer by name with its poll read back. Naming one
    attribute twice - `foamCoverage` and `foam_coverage` - is refused as two answers to one question.

    ALL OR NOTHING. Every attribute the call can touch is snapshotted first and restored on any refusal.

    A FLUID'S SETTINGS ARE ON ITS ROLE (2026-09-21). The modifier's own property is fluid_type; a key it lacks is
    looked up on the live role's struct - domain_settings, flow_settings or effector_settings - and written there by
    the same dispatcher: use_dissolve_smoke, dissolve_speed, use_collision_border_top, vorticity or time_scale on a
    domain, fuel_amount or temperature on a flow. A key another role owns is refused naming that role, and
    list_modifiers {rna: true} reads the role's settings back under its identifier.

    A CLOTH'S, COLLISION'S AND SOFT BODY'S SETTINGS ARE ON THEIR STRUCTS (2026-10-03, a short film's garments):
    a cloth's settings (use_sewing_springs, sewing_force_max, shrink_min, quality, air_damping, vertex_group_mass) and
    collision_settings (distance_min, use_self_collision, self_distance_min) and its point_cache (frame_start,
    frame_end); a collision modifier's settings (thickness_outer, cloth_friction); a soft body's settings and
    point_cache. Sent nested - {"collision_settings": {"distance_min": 0.01}} - or flat when one struct has the key;
    a flat key two structs share is refused naming both. Written by the same dispatcher, snapshotted and restored the
    same way, read back as "collision_settings.distance_min", and listed by list_modifiers {rna: true}.

    THE EFFECT IS MEASURED, not inferred from the fields: the evaluated mesh is fingerprinted before and
    after, as add_modifier does. An unchanged fingerprint is right for a setting that moves no vertex - a
    render toggle, a layer name - and is said as that rather than as a failure.

    params:
      object (str, required)     alias name
      modifier (str, required)   the modifier's name on that object
      settings (dict)            curated name or RNA identifier -> value; required unless moving
      index (int)                move the modifier to this place in the stack (0 = top, evaluated first)
      move (str)                 UP, DOWN, TOP or BOTTOM - one step or to an end; index or move, not both
      adaptive (dict)            Subdivision Surface only (2026-10-06): {use, dicingRate, space PIXEL | OBJECT (5.0+)} -
                                 Cycles' render-time dicing, on the modifier from 5.0 and on the object's Cycles
                                 settings before it (the scene's feature set made EXPERIMENTAL then, and said)

    MOVING (2026-10-04): the stack evaluates top to bottom, so order is behaviour - an Armature above a Collision or
    the cloth simulates the rest pose; a Shrinkwrap, then a Solidify, then a Subdivision Surface; a Subdivision Surface
    before a Cloth sets the simulation's detail and one after only smooths it. Before this the only route was remove
    and re-add, which lost the settings. The move uses Blender's own operator and is read back: a move Blender
    declines, and one that would leave the modifier where it is, are refused. With settings as well, the move happens
    first and is undone if a setting refuses. stackBefore and stackAfter name the order.
    """
    reject_unknown(params, _SETMOD_KEYS, "set_modifier", {
        "value": "settings go in `settings` as {name: value} - which names are valid depends on the "
                 "modifier's type, and a refusal lists that type's own.",
        "inputs": "a geometry-nodes modifier's inputs are its group's sockets, set with assign_node_group, "
                  "which resolves input names to the identifiers the modifier stores them under.",
        "apply": "setting and applying are separate on purpose - apply_modifier is the destructive step.",
    })
    obj = _modifier_host(take(params, "object", "name", required=True, kind=str))
    mod_name = take(params, "modifier", required=True, kind=str)
    mod = _stack(obj).get(str(mod_name))
    if mod is None:
        raise MifOpError("'%s' has no modifier named '%s'. It has %d: %s. NOTHING was changed."
                         % (obj.name, mod_name, len(_stack(obj)),
                            ", ".join(m.name for m in _stack(obj)) or "(none)"))
    settings = take(params, "settings", default=None)
    index = take_int(params, "index", default=None)
    move = take(params, "move", default=None, kind=str)
    if index is not None and move is not None:
        raise MifOpError("pass index or move, not both - each says where the modifier goes. NOTHING was changed.")
    moving = index is not None or move is not None
    if settings is not None and (not isinstance(settings, dict) or not settings):
        raise MifOpError("settings must be a non-empty object of name -> value. NOTHING was changed.")
    adaptive = _adaptive_plan(obj, mod, params.get("adaptive")) if params.get("adaptive") is not None else None
    if settings is None and not moving and adaptive is None:
        # A CALL THAT CHANGES NOTHING MUST NOT REPORT SUCCESS - set_constraint's rule, same reason.
        raise MifOpError("pass settings (name -> value), or index or move to place the modifier in the stack. Nothing "
                         "to do, so NOTHING was changed.")
    if settings is not None:
        _refuse_frozen_by_guides(obj, mod, settings)
    stack_before = [m.name for m in _stack(obj)]
    here = stack_before.index(mod.name)
    target = None
    if move is not None:
        move = move.upper()
        if move not in _MOVES:
            raise MifOpError("move must be UP, DOWN, TOP or BOTTOM (got %r). NOTHING was changed." % move)
        target = {"UP": here - 1, "DOWN": here + 1, "TOP": 0, "BOTTOM": len(stack_before) - 1}[move]
    elif index is not None:
        target = index
    if moving:
        if not 0 <= target < len(stack_before):
            raise MifOpError("'%s' is at %d of %d (%s), so it cannot go to %d. NOTHING was changed."
                             % (mod.name, here, len(stack_before), ", ".join(stack_before), target))
        if target == here:
            raise MifOpError("'%s' is already at %d (%s) - the move would change nothing. NOTHING was changed."
                             % (mod.name, here, ", ".join(stack_before)))

    mtype = mod.type
    mod_name_now = mod.name
    evaluated_before = _evaluated_counts(obj)
    if moving:
        declined = _move_modifier(obj, mod_name_now, target)
        if declined:
            _move_modifier(obj, mod_name_now, here)
            now = [m.name for m in _stack(obj)]
            if now != stack_before:
                raise MifOpError("Blender would not move '%s' from %d to %d (%s): %s, and putting it back left the "
                                 "stack as %s." % (mod_name_now, here, target, ", ".join(stack_before), declined,
                                                   ", ".join(now)))
            raise MifOpError("Blender would not move '%s' from %d to %d (%s): %s. A modifier that needs the original "
                             "mesh cannot sit below one that changes it. NOTHING was changed."
                             % (mod_name_now, here, target, ", ".join(stack_before), declined))
        mod = _stack(obj)[mod_name_now]
    applied = []
    if settings is not None:
        try:
            applied = _write_modifier_settings(obj, mod, settings)
        except Exception:
            # THE MOVE IS UNDONE, so the settings' own all-or-nothing refusal stays true for the whole call.
            if moving:
                _move_modifier(obj, mod_name_now, here)
            raise

    adaptive_out = _adaptive_write(obj, mod, adaptive) if adaptive is not None else None
    evaluated_after = _evaluated_counts(obj)
    changed = (bool(evaluated_before) and bool(evaluated_after) and evaluated_before != evaluated_after)
    return {
        "object": obj.name,
        "modifier": mod.name,
        "type": mtype,
        "adaptive": adaptive_out,
        # READ BACK off the modifier, never echoed - a clamped value or a case-folded enum is reported as
        # what Blender kept.
        "applied": applied,
        "appliedCount": len(applied),
        "stackBefore": stack_before,
        "stackAfter": [m.name for m in _stack(obj)],
        "moved": {"from": here, "to": target} if moving else None,
        "evaluatedBefore": evaluated_before,
        "evaluatedAfter": evaluated_after,
        "evaluatedChanged": changed,
        "evaluatedNote": (None if changed else
                          "the evaluated mesh has the same vertices, counts and positions as before this call. "
                          "That is right for a setting that moves no geometry - a render or viewport toggle, a "
                          "layer or attribute name, a bake path - and is worth a second look for one that "
                          "should: a modifier that is not pointed at anything evaluates to its input."),
        "readBack": _modifier_dict(mod),
    }


def _adaptive_plan(obj, mod, spec):
    """set_modifier adaptive {use, dicingRate, space}: checked in full, nothing written. 5.0+ keeps adaptive subdivision
    on the Subdivision Surface modifier (use_adaptive_subdivision, adaptive_space PIXEL | OBJECT, adaptive_pixel_size or
    adaptive_object_edge_length); before 5.0 it is the object's Cycles settings (cycles.use_adaptive_subdivision,
    cycles.dicing_rate) and needs the scene's Cycles feature set EXPERIMENTAL."""
    tail = "NOTHING was changed."
    if mod.type != "SUBSURF":
        raise MifOpError("adaptive is a Subdivision Surface's render-time dicing; '%s' is %s. %s" % (mod.name, mod.type,
                                                                                                   tail))
    if not isinstance(spec, dict) or not spec or any(k not in ("use", "dicingRate", "space") for k in spec):
        raise MifOpError("adaptive is {use, dicingRate, space}, got %r. %s" % (spec, tail))
    on_mod = "use_adaptive_subdivision" in mod.bl_rna.properties
    plan = {}
    if spec.get("use") is not None:
        plan["use"] = take_bool(spec, "use")
    if spec.get("space") is not None:
        space = str(take(spec, "space", kind=str)).upper()
        if not on_mod:
            raise MifOpError("adaptive space needs Blender 5.0 or later, where the modifier carries it (this is %s). %s"
                             % (bpy.app.version_string, tail))
        spaces = [i.identifier for i in mod.bl_rna.properties["adaptive_space"].enum_items]
        if space not in spaces:
            raise MifOpError("adaptive space %r is not one of %s. %s" % (space, ", ".join(spaces), tail))
        plan["space"] = space
    if spec.get("dicingRate") is not None:
        rate = take_float(spec, "dicingRate")
        if not rate > 0:
            raise MifOpError("dicingRate must be above 0, got %g. %s" % (rate, tail))
        plan["dicingRate"] = rate
    if not on_mod and getattr(getattr(obj, "cycles", None), "use_adaptive_subdivision", None) is None:
        raise MifOpError("this Blender has no Cycles adaptive subdivision setting on objects (is Cycles enabled?). %s"
                         % tail)
    return plan


def _adaptive_write(obj, mod, plan):
    """Write an _adaptive_plan where this build keeps it, and read it back."""
    out = {"route": "modifier" if "use_adaptive_subdivision" in mod.bl_rna.properties else "object.cycles"}
    if out["route"] == "modifier":
        if "use" in plan:
            mod.use_adaptive_subdivision = plan["use"]
        if "space" in plan:
            mod.adaptive_space = plan["space"]
        if "dicingRate" in plan:
            if mod.adaptive_space == "OBJECT":
                mod.adaptive_object_edge_length = plan["dicingRate"]
            else:
                mod.adaptive_pixel_size = plan["dicingRate"]
        out.update({"use": bool(mod.use_adaptive_subdivision), "space": mod.adaptive_space,
                    "dicingRate": round(float(mod.adaptive_object_edge_length if mod.adaptive_space == "OBJECT"
                                              else mod.adaptive_pixel_size), 6)})
        return out
    cyc = obj.cycles
    if "use" in plan:
        cyc.use_adaptive_subdivision = plan["use"]
    if "dicingRate" in plan:
        cyc.dicing_rate = plan["dicingRate"]
    scene_cycles = bpy.context.scene.cycles
    if cyc.use_adaptive_subdivision and scene_cycles.feature_set != "EXPERIMENTAL":
        out["featureSetBefore"] = scene_cycles.feature_set
        scene_cycles.feature_set = "EXPERIMENTAL"
        out["featureSetNote"] = ("before 5.0 Cycles dices adaptively only with the scene's feature set EXPERIMENTAL, "
                                 "so it was set from %s." % out["featureSetBefore"])
    out.update({"use": bool(cyc.use_adaptive_subdivision), "space": "PIXEL", "dicingRate": round(float(cyc.dicing_rate),
                                                                                                6),
                "featureSet": scene_cycles.feature_set})
    return out


def _vertex_coords(mesh):
    """Every vertex position as a flat array, for comparing a mesh against itself.

    foreach_get RATHER THAN A PYTHON LOOP, the same technique _uv_fingerprint uses: a hundred
    thousand vertices through attribute access is seconds, and this runs on the main thread while
    Blender is blocked.
    """
    n = len(mesh.vertices)
    buf = array.array("f", [0.0]) * (3 * n)
    mesh.vertices.foreach_get("co", buf)
    return buf


# Their state is a simulation or a particle system, not a deformation of the mesh as it is: refused with keepShapeKeys.
_KEEP_KEYS_REFUSED = ("CLOTH", "SOFT_BODY", "FLUID", "DYNAMIC_PAINT", "OCEAN", "EXPLODE", "PARTICLE_SYSTEM",
                      "PARTICLE_INSTANCE", "COLLISION")


def _apply_keeping_keys(obj, mod):
    """apply_modifier keepShapeKeys: bake a DEFORMING modifier into the basis and every shape key, keeping the keys.

    BLENDER HAS NO OPERATOR FOR IT - modifier_apply refuses any modifier on a keyed mesh (measured 3.6.23-5.2.1) - so
    each key's shape is captured with the modifier alone on the stack: the other modifiers bypassed, the shape-key lock
    (show_only_shape_key) on that key, its vertex group and mute cleared so the key's full shape is read, the evaluated
    mesh's positions taken. A topology-changing modifier is refused by name when any key's evaluated vertex count
    differs, before anything is written. Then the basis and each key take their captured positions, the modifier is
    removed, and everything set for the capture is put back. Reply: each key's largest vertex move, read back."""
    tail = "NOTHING was changed."
    if mod.type in _KEEP_KEYS_REFUSED:
        raise MifOpError("'%s' is a %s modifier - its result is a simulation or a particle system, not a deformation "
                         "keepShapeKeys can bake into each key. %s" % (mod.name, mod.type, tail))
    me = obj.data
    keys = me.shape_keys.key_blocks
    n = len(me.vertices)
    held_flags = (obj.show_only_shape_key, obj.active_shape_key_index)
    held_mods = [(m, m.show_viewport) for m in obj.modifiers]
    held_keys = [(k, k.vertex_group, k.mute) for k in keys]
    captured = []
    try:
        for m in obj.modifiers:
            m.show_viewport = (m == mod)
        for k, _vg, _mu in held_keys:
            k.vertex_group = ""
            k.mute = False
        obj.show_only_shape_key = True
        for i, k in enumerate(keys):
            obj.active_shape_key_index = i
            bpy.context.view_layer.update()
            ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            emesh = ev.to_mesh()
            try:
                if len(emesh.vertices) != n:
                    raise MifOpError("'%s' (%s) changes the topology - key '%s' evaluates to %d vertices, the mesh has %d "
                                     "- so it cannot be baked into the keys; keepShapeKeys takes a DEFORMING modifier. %s"
                                     % (mod.name, mod.type, k.name, len(emesh.vertices), n, tail))
                buf = array.array("f", [0.0]) * (3 * n)
                emesh.vertices.foreach_get("co", buf)
            finally:
                ev.to_mesh_clear()
            captured.append(buf)
    finally:
        obj.show_only_shape_key, obj.active_shape_key_index = held_flags
        for m, vis in held_mods:
            m.show_viewport = vis
        for k, vg, mu in held_keys:
            k.vertex_group = vg
            k.mute = mu
    # ------------------------------------------------------------------ writes
    ref = me.shape_keys.reference_key
    rows = []
    for i, k in enumerate(keys):
        before = array.array("f", [0.0]) * (3 * n)
        k.data.foreach_get("co", before)
        k.data.foreach_set("co", captured[i])
        after = array.array("f", [0.0]) * (3 * n)
        k.data.foreach_get("co", after)
        moved = max((abs(after[j] - before[j]) for j in range(3 * n)), default=0.0)
        rows.append({"key": k.name, "maxVertexDelta": round(moved, 6),
                     "readBack": all(abs(after[j] - captured[i][j]) < 1e-6 for j in range(3 * n))})
        if k == ref:
            me.vertices.foreach_set("co", captured[i])
    mod_name, mod_type = mod.name, mod.type
    obj.modifiers.remove(mod)
    me.update()
    return {"object": obj.name, "modifier": mod_name, "type": mod_type, "applied": True,
            "keepShapeKeys": True, "keys": rows, "stackAfter": [m.name for m in obj.modifiers],
            "note": None if any(r["maxVertexDelta"] > 1e-6 for r in rows) else
            "the modifier moved no vertex of any key - it was inert on this mesh."}


def _apply_modifier_curves(obj, params):
    """apply_modifier on a hair CURVES object (2026-10-15, the rounds 11-12 settle pass): Blender applies a modifier on
    Curves from 3.5 - a Geometry Nodes modifier is the only kind a Curves object takes - and this op refused it as "not
    a MESH". Measured on the settle pass's builds: bpy.ops.object.modifier_apply on a Curves object with a pass-through
    NODES modifier leaves the curve count and removes the modifier. The proof here is the curve and point counts and
    the point positions, before and after; the mesh-only modes are refused by name."""
    for key in ("asShapeKey", "keepModifier", "keyName", "keepShapeKeys", "previewId", "previewTtlSeconds"):
        if params.get(key) is not None:
            raise MifOpError("%s is a mesh mode of apply_modifier; '%s' is a CURVES object, whose modifier applies into "
                             "its curves. NOTHING was changed." % (key, obj.name))
    mod_name = take(params, "modifier", required=True, kind=str)
    if obj.modifiers.get(mod_name) is None:
        raise MifOpError("'%s' has no modifier named '%s'. It has %d: %s. NOTHING was changed."
                         % (obj.name, mod_name, len(obj.modifiers), ", ".join(m.name for m in obj.modifiers) or "(none)"))

    def counts():
        d = obj.data
        return {"curves": len(d.curves), "points": len(d.points)}

    def positions():
        return [tuple(p.position) for p in obj.data.points]

    before, pos_before = counts(), positions()
    if take_bool(params, "dryRun", default=False):
        return {"object": obj.name, "type": "CURVES", "modifier": mod_name, "dryRun": True, "applied": False,
                "counts": before, "dataUsers": obj.data.users}
    if obj.data.users > 1:
        raise MifOpError("'%s' shares its curves data with %d other object(s), and applying rewrites it for every one of "
                         "them. NOTHING was changed." % (obj.name, obj.data.users - 1))
    snap = selection_snapshot()
    try:
        select_only([obj])
        try:
            with bpy.context.temp_override(object=obj, active_object=obj):
                bpy.ops.object.modifier_apply(modifier=mod_name)
        except RuntimeError as exc:
            raise MifOpError("Blender refused to apply '%s' on '%s': %s. NOTHING was changed."
                             % (mod_name, obj.name, str(exc).strip()))
    finally:
        selection_restore(snap)
    after = counts()
    if mod_name in obj.modifiers:
        raise MifOpError("modifier_apply reported no error but '%s' is still on '%s' - counts before %s, after %s."
                         % (mod_name, obj.name, before, after))
    pos_after = positions()
    moved = sum(1 for a, b in zip(pos_before, pos_after) if max(abs(x - y) for x, y in zip(a, b)) > 1e-6) \
        if len(pos_before) == len(pos_after) else None
    return {"object": obj.name, "type": "CURVES", "modifier": mod_name, "applied": True, "countsBefore": before,
            "countsAfter": after, "changedTopology": before != after, "pointsMoved": moved,
            "stackAfter": [m.name for m in obj.modifiers]}


def op_apply_modifier(params):
    """Bake a modifier into the mesh data. Destructive, and it says what it cost.

    THE COUNTS ARE THE POINT. modifier_apply reports {'FINISHED'} whether it changed
    the mesh or not, so this reads vertex and face counts before and after and
    reports both. A Mirror that doubled the mesh and a disabled modifier that did
    nothing are indistinguishable from the operator's return value alone.

    IT REFUSES ON MULTI-USER DATA, for the same reason apply_transform does:
    applying would rewrite a mesh other objects share, and Blender's own operator
    fails there anyway - refusing first with the sharing count is a better error
    than the operator's.

    dryRun reports what is on the stack and what applying would touch, and changes
    nothing. It is also the BULK PREVIEW (docs/15 section 5): the apply run on a scratch copy of the object, every
    change it makes (each moved vertex, or the geometry when the topology changes, and the stack), and a previewId that
    {previewId} alone commits - planned again, refused on any difference, read back, and the mesh and the modifier put
    back on a mismatch. A simulation modifier cannot be previewed: the copy does not share its cache.
    """
    reject_unknown(params, ("object", "name", "modifier", "dryRun", "asShapeKey", "keepModifier", "keyName", "previewId",
                            "previewTtlSeconds", "keepShapeKeys"), "apply_modifier")
    obj_name = take(params, "object", "name", required=True, kind=str)
    host = get_object(obj_name)
    if host.type in ("CURVE", "FONT", "SURFACE", "LATTICE"):
        # THE REASON, not only "not a MESH", now that add_modifier puts modifiers on these. Blender itself refuses
        # a constructive modifier on a curve - "Cannot apply constructive modifiers on curve. Convert curve to mesh
        # in order to apply", measured on 3.6 to 5.2 - and this op proves what applying changed by comparing vertex
        # data, which only a mesh has.
        raise MifOpError("'%s' is a %s. apply_modifier bakes into MESH data and proves the result by its vertices; "
                         "Blender itself will not apply a constructive modifier to a curve without converting it "
                         "to a mesh first. The modifier still evaluates and renders where it is. NOTHING was changed."
                         % (host.name, host.type))
    if host.type == "CURVES":
        return _apply_modifier_curves(host, params)
    obj = get_object(obj_name, want_mesh=True)
    mod_name = take(params, "modifier", required=True, kind=str)

    mod = obj.modifiers.get(mod_name)
    if mod is None:
        raise MifOpError(
            "'%s' has no modifier named '%s'. It has %d: %s. NOTHING was changed."
            % (obj.name, mod_name, len(obj.modifiers),
               ", ".join(m.name for m in obj.modifiers) or "(none)"))

    # AS A SHAPE KEY (2026-10-04, the garment work): a lattice or surface-deform result captured as a key on the base
    # mesh instead of baked into it - how a garment morph is made to follow a body morph. And a plain apply that
    # Blender refuses because the mesh HAS keys is refused with the ways through named (below, at the apply). Blender
    # applies NO modifier into a keyed mesh, deforming or constructive: Displace, Simple Deform, Smooth and Solidify
    # were each refused "Modifier cannot be applied to a mesh with shape keys" by a bare modifier_apply on 3.6.23,
    # 4.2.17, 4.4.0, 5.0.1 and 5.2.1 (2026-10-11). The refusal said a deform-only one "applies to every key; this one
    # changes the topology", which told a caller applying a Displace something false about it.
    as_key = take_bool(params, "asShapeKey", default=False)
    keep_mod = take_bool(params, "keepModifier", default=False)
    key_name = take(params, "keyName", default=None, kind=str)
    if not as_key and (params.get("keepModifier") is not None or key_name is not None):
        raise MifOpError("keepModifier and keyName go with asShapeKey:true. NOTHING was changed.")
    has_keys = bool(obj.data.shape_keys and len(obj.data.shape_keys.key_blocks))
    # KEEP SHAPE KEYS (2026-10-11): a deforming modifier applied to the basis AND every key, the keys
    # kept - what Blender's own Apply refuses on a keyed mesh. See _apply_keeping_keys.
    keep_keys = take_bool(params, "keepShapeKeys", default=False)
    if keep_keys:
        if as_key:
            raise MifOpError("keepShapeKeys applies the modifier into the mesh and its keys; asShapeKey captures it as a "
                             "new key instead - pass one. NOTHING was changed.")
        if not has_keys:
            raise MifOpError("'%s' has no shape keys, so keepShapeKeys would keep nothing - drop it and the modifier is "
                             "applied directly. NOTHING was changed." % obj.name)
        if take_bool(params, "dryRun", default=False):
            raise MifOpError("dryRun is not offered with keepShapeKeys - it would preview the plain apply, which Blender "
                             "refuses on a keyed mesh. NOTHING was changed.")
        if obj.data.users > 1:
            raise MifOpError("'%s' shares its mesh data with %d other object(s), and applying a modifier rewrites that "
                             "data for every one of them. Make it single-user in Blender first. NOTHING was changed."
                             % (obj.name, obj.data.users - 1))
        return _apply_keeping_keys(obj, mod)
    if as_key:
        if take_bool(params, "dryRun", default=False):
            raise MifOpError("dryRun does not apply to asShapeKey - it adds a key and keeps the mesh. NOTHING was changed.")
        keys_before = list(obj.data.shape_keys.key_blocks.keys()) if has_keys else []
        if key_name is not None and key_name in keys_before:
            raise MifOpError("'%s' already has a shape key '%s'. NOTHING was changed." % (obj.name, key_name))
        snap = selection_snapshot()
        try:
            select_only([obj])
            try:
                bpy.ops.object.modifier_apply_as_shapekey(modifier=mod_name, keep_modifier=keep_mod)
            except RuntimeError as exc:
                raise MifOpError("Blender would not apply '%s' as a shape key: %s. Only a DEFORMING modifier (one that "
                                 "moves vertices without changing the topology) can be. NOTHING was changed."
                                 % (mod_name, str(exc).strip()))
        finally:
            selection_restore(snap)
        blocks = obj.data.shape_keys.key_blocks if obj.data.shape_keys else []
        # On a mesh with no keys Blender adds the Basis (the reference key) and the result; the result is the last new
        # block that is not the reference.
        ref = obj.data.shape_keys.reference_key if obj.data.shape_keys else None
        new = [k for k in blocks if k.name not in keys_before and k != ref]
        made = new[-1] if new else None
        if made is None:
            raise MifOpError("modifier_apply_as_shapekey reported no error and '%s' gained no shape key. NOTHING usable "
                             "was produced." % obj.name)
        if key_name is not None:
            made.name = key_name
        basis = made.relative_key
        offset = max(((made.data[i].co - basis.data[i].co).length for i in range(len(made.data))), default=0.0)
        return {"object": obj.name, "modifier": mod_name, "asShapeKey": True, "key": made.name,
                "keys": [k.name for k in obj.data.shape_keys.key_blocks], "modifierKept": mod_name in obj.modifiers,
                "maxOffsetFromBasis": round(offset, 6),
                "note": None if offset > 1e-6 else ("the new key moves nothing - the modifier deformed nothing at the "
                                                    "current frame and settings.")}

    counts_before = mesh_counts(obj)
    uvs_before = len(obj.data.uv_layers)
    # POSITIONS TOO. See the postcondition below for why counts were not enough.
    coords_before = _vertex_coords(obj.data)
    row = _modifier_dict(mod)

    if take_bool(params, "dryRun", default=False):
        return {
            "object": obj.name,
            "modifier": mod_name,
            "dryRun": True,
            "applied": False,
            "wouldApply": row,
            "counts": counts_before,
            "meshDataUsers": obj.data.users,
        }

    if obj.data.users > 1:
        raise MifOpError(
            "'%s' shares its mesh data with %d other object(s), and applying a modifier rewrites "
            "that data for every one of them. Make it single-user in Blender first. NOTHING was "
            "changed." % (obj.name, obj.data.users - 1))

    snap = selection_snapshot()
    try:
        select_only([obj])
        try:
            bpy.ops.object.modifier_apply(modifier=mod_name)
        except RuntimeError as exc:
            text = str(exc).strip()
            if "shape key" in text.lower():
                raise MifOpError("Blender will not apply '%s' into '%s', which has %d shape key(s): %s. Blender applies "
                                 "no modifier into a mesh with shape keys, deforming or constructive. Ways through: "
                                 "keepShapeKeys:true applies a DEFORMING modifier to the basis and each key and keeps "
                                 "them; asShapeKey:true captures its result as a new key and leaves the mesh and its "
                                 "keys as they are; or apply_shape_keys first, which bakes the keys into the mesh and "
                                 "removes them, then apply. NOTHING was changed."
                                 % (mod_name, obj.name, len(obj.data.shape_keys.key_blocks), text))
            raise MifOpError("Blender refused to apply '%s': %s. NOTHING was changed." % (mod_name, text))
    finally:
        selection_restore(snap)

    still_there = mod_name in obj.modifiers
    counts_after = mesh_counts(obj)
    if still_there:
        raise MifOpError(
            "modifier_apply reported no error but '%s' is still on '%s'. The mesh may or may not "
            "have been changed - counts before %s, after %s." % (mod_name, obj.name,
                                                                 counts_before, counts_after))

    # TOPOLOGY AND MOVEMENT ARE DIFFERENT QUESTIONS, and this used to ask only the first.
    #
    # changedGeometry was `counts_before != counts_after`, and mesh_counts is vertex/edge/face
    # COUNTS. Every DEFORMING modifier - SHRINKWRAP, LATTICE, CAST, DISPLACE, SIMPLE_DEFORM, WAVE,
    # WARP, HOOK, SMOOTH, ARMATURE - moves vertices and changes no count, so all of them read as
    # unchanged. The note then stated a conclusion: "it was either disabled, or its settings
    # amounted to a no-op on this mesh". A shrinkwrap that had just conformed an entire mesh to
    # terrain was reported as having done nothing, in the confident register the rest of this addon
    # earns - which is worse than saying nothing, because somebody acts on it.
    changed_topology = counts_before != counts_after
    # THE FACES A CONSTRUCTIVE MODIFIER MADE, with UVs Unreal can build tangents from: a Solidify's rim and a Wireframe's
    # struts come out collapsed to a line, 16 vertex instances on a plane's rim (measured on 5.2.1, 2026-11-01).
    # A Remesh drops the UV layers altogether, so the mesh gets a box projection in their place; an Array that merges two
    # touching pieces keeps the wall between them doubled, 8 edges on four faces for three cubes in a row.
    uv_projected = (project_flat_uvs(obj, only_collapsed=False)
                    if changed_topology and uvs_before and not len(obj.data.uv_layers) else 0)
    uv_opened = open_collapsed_uvs_of(obj) if changed_topology else 0
    walls_removed = 0
    if changed_topology and row.get("type") in ("ARRAY", "MIRROR"):
        bm = bmesh.new()
        try:
            bm.from_mesh(obj.data)
            walls_removed = drop_doubled_and_fin_faces(bm)
            if walls_removed:
                bm.to_mesh(obj.data)
                obj.data.update()
                counts_after = mesh_counts(obj)
        finally:
            bm.free()
    coords_after = _vertex_coords(obj.data)
    moved, max_delta = 0, 0.0
    if not changed_topology and len(coords_before) == len(coords_after):
        for i in range(0, len(coords_before), 3):
            dx = abs(coords_after[i] - coords_before[i])
            dy = abs(coords_after[i + 1] - coords_before[i + 1])
            dz = abs(coords_after[i + 2] - coords_before[i + 2])
            worst = max(dx, dy, dz)
            if worst > 1e-6:
                moved += 1
                if worst > max_delta:
                    max_delta = worst

    out = {
        "object": obj.name,
        "modifier": mod_name,
        "applied": True,
        "wasApplied": row,
        "countsBefore": counts_before,
        "countsAfter": counts_after,
        "changedTopology": changed_topology,
        "uvFacesOpened": uv_opened,
        "uvFacesProjected": uv_projected,
        "doubledFacesRemoved": walls_removed,
        # HOW MANY VERTICES ACTUALLY MOVED. None when the topology changed, because the two meshes
        # have no vertex-to-vertex correspondence to compare and a number there would be invented.
        "movedVertices": None if changed_topology else moved,
        "maxVertexDelta": None if changed_topology else round(max_delta, 6),
        # KEPT, and now true. It was False for every deforming modifier ever applied through here.
        "changedGeometry": changed_topology or moved > 0,
        "stackAfter": [m.name for m in obj.modifiers],
    }
    # A SIMULATION BAKES THE FRAME THE SCENE IS ON (2026-10-05, measured on 4.2.23-5.2.2): a cloth stepped to frame 10
    # with step_simulation and applied here leaves the base mesh exactly at the shape it evaluated to at frame 10 -
    # the route for a settled drape. Said, with the frame, because "applied" alone does not say which frame's shape.
    if row.get("type") in ("CLOTH", "SOFT_BODY", "FLUID", "DYNAMIC_PAINT", "OCEAN", "EXPLODE", "PARTICLE_INSTANCE"):
        out["simulationFrame"] = int(bpy.context.scene.frame_current)
        out["simulationNote"] = ("a %s modifier bakes the shape it evaluates to at the frame the scene is on - frame "
                                 "%d here. To keep a settled shape, step_simulation to that frame first (it steps "
                                 "every frame, which a simulation needs, and leaves the scene there), then apply."
                                 % (row.get("type"), out["simulationFrame"]))
    if not out["changedGeometry"]:
        out["note"] = (
            "the modifier applied cleanly and NOTHING changed - same vertex/face counts and not one "
            "vertex moved by more than 1e-6. It was either disabled, or its settings amounted to a "
            "no-op on this mesh. Said in words rather than returned as a bare ok, because the "
            "operator reports FINISHED either way.")
    elif not changed_topology:
        out["note"] = (
            "the counts are identical and %d vertex/vertices MOVED, by up to %.6f - a deforming "
            "modifier does exactly this, and judging by counts alone would have called it a no-op."
            % (moved, max_delta))
    return out


# Modifier type -> (bind operator, the property that says it is bound, the property that must name its driver).
_BINDABLE = {
    "SURFACE_DEFORM": ("surfacedeform_bind", "is_bound", "target"),
    "MESH_DEFORM": ("meshdeform_bind", "is_bound", "object"),
    "CORRECTIVE_SMOOTH": ("correctivesmooth_bind", "is_bind", None),
    "LAPLACIANDEFORM": ("laplaciandeform_bind", "is_bind", None),
}


def op_bind_modifier(params):
    """Bind (or unbind) a deform modifier that does nothing until it is bound.

    THE GAP, 2026-10-04: Surface Deform, Mesh Deform, Corrective Smooth (rest source Bind) and Laplacian Deform deform
    nothing until their Bind button is pressed, and nothing typed pressed it - so the bind-and-bake route to garment
    morphs that follow a body morph stopped at the first step.

    BLENDER'S BIND IS A TOGGLE: the same operator unbinds a bound modifier. So the state is read first and the operator
    runs only when it would change it - bind on a bound modifier answers alreadyInState rather than unbinding it - and
    the state is read back after. A Surface Deform with no target or a Mesh Deform with no cage object is refused, as is
    a Corrective Smooth whose rest source is not BIND (binding it would change nothing). A bind that Blender refuses
    (Surface Deform: "target has edges with more than two polygons", or concave polygons) is passed on with its reason.
    """
    reject_unknown(params, ("object", "name", "modifier", "unbind"), "bind_modifier")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    mod_name = take(params, "modifier", required=True, kind=str)
    unbind = take_bool(params, "unbind", default=False)
    mod = obj.modifiers.get(mod_name)
    if mod is None:
        raise MifOpError("'%s' has no modifier named '%s'. It has: %s. NOTHING was changed."
                         % (obj.name, mod_name, ", ".join(m.name for m in obj.modifiers) or "(none)"))
    if mod.type not in _BINDABLE:
        raise MifOpError("'%s' is a %s modifier, which has no bind. Bindable: %s. NOTHING was changed."
                         % (mod_name, mod.type, ", ".join(sorted(_BINDABLE))))
    op_name, state_attr, needs = _BINDABLE[mod.type]
    if needs and getattr(mod, needs, None) is None and not unbind:
        raise MifOpError("'%s' (%s) has no %s set, so there is nothing to bind to - set it with set_modifier first. "
                         "NOTHING was changed." % (mod_name, mod.type, needs))
    if mod.type == "CORRECTIVE_SMOOTH" and getattr(mod, "rest_source", "BIND") != "BIND":
        raise MifOpError("'%s' takes its rest shape from '%s', not BIND, so binding changes nothing - set its rest_source "
                         "to BIND first. NOTHING was changed." % (mod_name, mod.rest_source))
    before = bool(getattr(mod, state_attr))
    want = not unbind
    if before != want:
        snap = selection_snapshot()
        try:
            select_only([obj])
            with bpy.context.temp_override(object=obj, active_object=obj):
                getattr(bpy.ops.object, op_name)(modifier=mod_name)
        except RuntimeError as exc:
            raise MifOpError("Blender would not %s '%s': %s" % ("unbind" if unbind else "bind", mod_name,
                                                                str(exc).strip()))
        finally:
            selection_restore(snap)
    after = bool(getattr(mod, state_attr))
    if after != want:
        raise MifOpError("the %s ran and '%s' reads %s %s - Blender reported no error and did not %s it."
                         % (op_name, mod_name, state_attr, after, "unbind" if unbind else "bind"))
    return {"object": obj.name, "modifier": mod_name, "type": mod.type, "bound": after, "boundBefore": before,
            "alreadyInState": before == want}


def op_rename_bones(params):
    """Rename armature bones through a map, refusing the collision that silently breaks skinning.

    BLENDER ALREADY SYNCS THE EASY PART, verified against 4.4 rather than assumed: setting
    bone.name renames the matching vertex group on every skinned mesh, and updates constraint
    subtargets and driver bone targets. This op does not re-implement any of that.

    WHAT IT EXISTS FOR is the case where that sync silently fails. Renaming a bone to a name another
    bone already holds gives you 'Hips.001' - a name nobody asked for - and leaves the vertex group
    under its OLD name, matching no bone. That part of the mesh then stops deforming, and nothing
    says so. Measured:

        bones ['Hips','Spine'] vgroups ['Hips','Spine']  ->  bones['Spine'].name = 'Hips'
        bones ['Hips','Hips.001'] vgroups ['Hips','Spine']

    So collisions are refused before anything is written, and every rename is READ BACK - a bone
    whose name is not what was asked for means Blender suffixed it, which is the failure.

    SWAPS ARE SUPPORTED because a retarget map is full of them (L/R mix-ups especially). A->B and
    B->A cannot be done in either order directly - whichever runs first collides - so the batch goes
    through unique temporary names first. That is invisible in the result and stated here because a
    reader will otherwise wonder why the two-pass exists.

    dryRun:true is the BULK PREVIEW (docs/15 section 5): one change per bone, its name before and after, the skinned
    meshes as context, and a previewId that {previewId} alone commits - planned again, refused on any difference, read
    back by the bone's index, renamed back on a mismatch.
    """
    reject_unknown(params, ("object", "name", "renames", "map", "dryRun", "previewId", "previewTtlSeconds"),
                   "rename_bones")
    obj = get_object(take(params, "object", "name", required=True, kind=str))
    if obj.type != "ARMATURE":
        raise MifOpError("object '%s' is a %s, not an ARMATURE. list_objects with type:'ARMATURE' "
                         "finds one. NOTHING was changed." % (obj.name, obj.type))

    renames = take(params, "renames", "map", required=True)
    if not isinstance(renames, dict) or not renames:
        raise MifOpError("renames must be a non-empty object of {oldName: newName}. NOTHING was "
                         "changed.")

    bones = obj.data.bones
    existing = [b.name for b in bones]
    # take_bool: bool("false") is True, so {dryRun: "false"} renamed NOTHING and answered dryRun:true under
    # ok:true (measured on 5.0.1, 2026-09-21).
    try:
        dry_run = take_bool(params, "dryRun", default=False)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was changed." % exc)

    # ---- every check BEFORE any write, so a bad batch changes nothing at all ----------------
    missing = [old for old in renames if old not in existing]
    if missing:
        raise MifOpError("no bone named %s on '%s'. It has %d: %s. NOTHING was changed."
                         % (", ".join("'%s'" % m for m in sorted(missing)), obj.name,
                            len(existing), ", ".join(existing[:24])))

    bad = [(o, n) for o, n in renames.items() if not isinstance(n, str) or not n.strip()]
    if bad:
        raise MifOpError("every new name must be a non-empty string; %s is not. NOTHING was changed."
                         % ", ".join("'%s' -> %r" % (o, n) for o, n in bad))

    # A target is only a collision if the name is held by a bone that is NOT being renamed away in
    # this same batch - otherwise a swap or a chain would be refused for no reason.
    freed = set(renames.keys())
    taken = {b for b in existing if b not in freed}
    collisions = {o: n for o, n in renames.items() if n in taken}
    if collisions:
        raise MifOpError(
            "these targets are already used by a bone this batch does NOT rename away: %s. Blender "
            "would not fail - it would silently suffix the bone ('Hips' becomes 'Hips.001') AND "
            "leave the vertex group under its old name, matching no bone, so that part of the mesh "
            "stops deforming with nothing to say so. Refused before anything was written. NOTHING "
            "was changed."
            % ", ".join("'%s' -> '%s'" % (o, n) for o, n in sorted(collisions.items())))

    dupes = [n for n in set(renames.values()) if list(renames.values()).count(n) > 1]
    if dupes:
        raise MifOpError("two bones are being renamed to the same name (%s), which collides with "
                         "itself whichever order it runs in. NOTHING was changed."
                         % ", ".join("'%s'" % d for d in sorted(dupes)))

    meshes = [o for o in bpy.data.objects
              if o.type == "MESH" and any(m.type == "ARMATURE" and m.object is obj
                                          for m in o.modifiers)]
    groups_before = {o.name: [g.name for g in o.vertex_groups] for o in meshes}

    if dry_run:
        return {
            "armature": obj.name,
            "dryRun": True,
            "changed": False,
            "wouldRename": dict(sorted(renames.items())),
            "skinnedMeshes": [o.name for o in meshes],
            "note": "DRY RUN - nothing was written. Every name checked: all sources exist, no "
                    "target collides with a bone kept by this batch, and no two bones want the "
                    "same name.",
        }

    # ---- TWO PASSES, so a swap is expressible ------------------------------------------------
    # A->B with B->A collides whichever runs first. Parking every source on a name nothing else can
    # hold makes the order irrelevant.
    temp = {}
    for i, old in enumerate(renames):
        tmp = "__mif_rn_%d__" % i
        bones[old].name = tmp
        temp[tmp] = renames[old]
    for tmp, new in temp.items():
        bones[tmp].name = new

    # ---- POSTCONDITION: read every name back -------------------------------------------------
    now = [b.name for b in bones]
    not_applied = {o: n for o, n in renames.items() if n not in now}
    if not_applied:
        raise MifOpError(
            "renamed, but %s are not present afterwards - Blender suffixed at least one name, which "
            "is the collision this op checks for and means the check missed a case. The armature is "
            "now: %s"
            % (", ".join("'%s'" % n for n in sorted(not_applied.values())), ", ".join(now[:24])))

    # Vertex groups matching NO bone. Reported whether or not this call caused them: an orphan is
    # the thing that makes a mesh stop deforming, and a caller renaming bones is the one person
    # certain to care.
    bone_names = set(now)
    orphans = {}
    for o in meshes:
        stray = [g.name for g in o.vertex_groups if g.name not in bone_names]
        if stray:
            orphans[o.name] = stray

    result = {
        "armature": obj.name,
        "renamed": dict(sorted(renames.items())),
        "renamedCount": len(renames),
        "changed": True,
        "boneNames": now,
        "skinnedMeshes": [o.name for o in meshes],
        "vertexGroupsBefore": groups_before,
        "vertexGroupsAfter": {o.name: [g.name for g in o.vertex_groups] for o in meshes},
    }
    if orphans:
        result["orphanedVertexGroups"] = orphans
        result["note"] = (
            "these vertex groups match NO bone on '%s' and so deform nothing: %s. Blender renames a "
            "vertex group along with its bone, so an orphan here was already orphaned before this "
            "call - reported because a caller renaming bones is the one person certain to care."
            % (obj.name, "; ".join("%s: %s" % (k, ", ".join(v)) for k, v in sorted(orphans.items()))))
    else:
        result["orphanedVertexGroups"] = {}
        result["note"] = ("every vertex group on every skinned mesh still matches a bone. Blender "
                          "renames groups along with their bones; this confirms it happened rather "
                          "than assuming it.")
    return result


# ---------------------------------------------------------------------------------------------------------------------
# THE BULK PREVIEW of rename_bones (ops_preview.py; docs/15 section 5). A bone is named by its INDEX in the armature, which a
# rename does not move, so the read-back finds it whatever it is called; the inverse renames back through temporary
# names, as the op itself does, so a swap goes back too.
# ---------------------------------------------------------------------------------------------------------------------
def _plan_rename_bones(request):
    resp = op_rename_bones(dict(request, dryRun=True))          # every refusal, and writes nothing
    obj = bpy.data.objects[resp["armature"]]
    names = [b.name for b in obj.data.bones]
    renames = take(request, "renames", "map")
    changes, unchanged = [], []
    for old, new in renames.items():
        index = names.index(old)
        row = {"kind": "bone", "op": "modify", "target": obj.name, "field": "bones[%d].name" % index,
               "label": old, "before": old, "after": new, "_index": index}
        (unchanged if old == new else changes).append(row)
    commit = {"object": obj.name, "renames": {c["before"]: c["after"] for c in changes}}
    return {"changes": changes, "unchanged": unchanged, "commit": commit, "response": resp,
            "context": ["armature %s: %d bones" % (obj.name, len(names)),
                        "skinnedMeshes: %s" % ", ".join(resp.get("skinnedMeshes") or [])]}


def _read_rename_bones(change, _kept):
    obj = bpy.data.objects.get(change["target"])
    bones = obj.data.bones if obj is not None and obj.type == "ARMATURE" else None
    return bones[change["_index"]].name if bones is not None and change["_index"] < len(bones) else None


def _capture_rename_bones(plan):
    obj = bpy.data.objects[plan["commit"]["object"]]
    return {"object": obj.name, "names": [b.name for b in obj.data.bones]}


def _restore_rename_bones(kept, _plan):
    obj = bpy.data.objects.get(kept["object"])
    if obj is None:
        raise MifOpError("armature '%s' is gone, so its bones cannot be renamed back" % kept["object"])
    bones = obj.data.bones
    wrong = [i for i, name in enumerate(kept["names"]) if i < len(bones) and bones[i].name != name]
    for i in wrong:
        bones[i].name = "__mif_rb_%d__" % i
    for i in wrong:
        bones[i].name = kept["names"][i]


PREVIEW = {"rename_bones": {"plan": _plan_rename_bones, "read": _read_rename_bones,
                            "capture": _capture_rename_bones, "restore": _restore_rename_bones}}


def op_set_bone_pose(params):
    """Pose a bone, and read the result back through the DEPSGRAPH.

    Character animation had zero coverage here: bones could be listed and renamed and nothing
    else. This was also unreachable through set_keyframe until the same day, because its dotted-
    path walk stripped subscripts and pose.bones["x"].location resolved to the bone COLLECTION.

    params:
      object (str, required)   the ARMATURE object
      bone (str, required)     pose bone name
      location [x,y,z]         in the bone's own space, as Blender stores it
      rotation [x,y,z]         euler radians, for a bone in an Euler mode - refused on a QUATERNION or AXIS_ANGLE
                               bone, which would ignore it
      quaternion [w,x,y,z]     for a bone in QUATERNION mode - refused together with rotation
      scale [x,y,z]
      rotationMode (str)       QUATERNION | XYZ | XZY | YXZ | YZX | ZXY | ZYX | AXIS_ANGLE, applied BEFORE the values.
                               A new bone is QUATERNION, which ignores a euler; changing the mode keeps the bone's
                               current pose (Blender converts the rotation) - measured on every build.

      mirror (dict)            {pair: 'FLIP_NAME' | [a, b] | {name: partner}, axis X, verify true} - then write the
                               partner bone with the same values flipped across X (ops_pose's mirror rule) and, with
                               verify, report the mirror error of the two evaluated bones. Refused before any write: no
                               partner under the pair, a partner with another rotation mode.

      turn (dict)              {axis: [x,y,z] | 'X' | 'Y' | 'Z' (WORLD), degrees, pivot 'head' (default) | 'tail' | [x,y,z]}
                               (1.4) - turn the bone about a world axis through its posed head, composed onto the pose
                               as it stands (after any location, rotation or scale in the same call), its children
                               carried with it. The reply measures the turn back off the evaluated bone.
      ik (dict)                {target: [x,y,z] (WORLD) | 'rest', pole: [x,y,z] (WORLD), keepRotation 'current'
                               (default) | 'rest' | 'none'} (1.4) - a two-bone IK solve written as plain FK pose values:
                               `bone` is the end (a foot), its parent and grandparent the chain (shin and thigh). The
                               end's head is put on target ('rest' = where it stands in the rest pose: the foot planted),
                               the middle joint bends toward pole (default: the way the chain bends now, else in the
                               rest pose), and the end keeps its world rotation. Refused before any write: a target out
                               of reach (the distance and the reach named), a chain whose joints do not meet, a straight
                               chain with no pole, a constrained chain bone.

    THE READ-BACK IS EVALUATED. A pose bone with a constraint on it - an IK chain, a copy
    rotation, a limit - does not end up where you put it, and pose_bone.matrix is the raw value
    rather than the result. Reporting what you wrote back at you would be a proxy that cannot fail,
    so this reports both and says which is which.
    """
    reject_unknown(params, {"object", "name", "bone", "location", "rotation", "quaternion",
                            "scale", "rotationMode", "mirror", "turn", "ik"}, "set_bone_pose")
    obj = get_object(take(params, "object", "name", required=True))
    if obj.type != "ARMATURE":
        raise MifOpError("'%s' is a %s, not an ARMATURE. NOTHING was changed."
                         % (obj.name, obj.type))
    bone_name = take(params, "bone", default=None, kind=str)
    if not bone_name:
        raise MifOpError("'bone' is required - which pose bone to move. NOTHING was changed.")
    if obj.pose is None:
        raise MifOpError("'%s' has no pose data. NOTHING was changed." % obj.name)
    pb = obj.pose.bones.get(str(bone_name))
    if pb is None:
        known = [b.name for b in obj.pose.bones][:25]
        raise MifOpError("no bone named '%s' on '%s'. Present: %s. NOTHING was changed."
                         % (bone_name, obj.name, ", ".join(known) if known else "<none>"))

    if "rotation" in params and "quaternion" in params:
        raise MifOpError("pass rotation OR quaternion, not both - they are two answers to the same "
                         "question. NOTHING was changed.")

    def _vec(key, n):
        v = params.get(key)
        if v is None:
            return None
        if isinstance(v, dict):
            order = ("w", "x", "y", "z") if n == 4 else ("x", "y", "z")
            # SIXTH COPY OF THIS PARSER IN THE ADDON, and the defect was in all six: a dict read
            # with .get(axis, 0.0) turns {"mif":"typo"} into a zero vector - or a zero QUATERNION,
            # which is not even a rotation - and reports success.
            check_axis_dict(v, key, order)
            # FINITE TOO. A NaN quaternion is not a rotation any more than a zero one is, and it
            # reads back as nan while every field in the response agrees the pose was set.
            return finite_floats([v.get(k, 0.0) for k in order], key)
        if isinstance(v, (list, tuple)) and len(v) == n:
            return finite_floats(v, key)
        raise MifOpError("'%s' must be a %d-list%s, got %r. NOTHING was changed."
                         % (key, n, " or {x,y,z}" if n == 3 else " or {w,x,y,z}", v))

    loc = _vec("location", 3)
    rot = _vec("rotation", 3)
    quat = _vec("quaternion", 4)
    scl = _vec("scale", 3)
    # THE MODE IS VALIDATED AND DECIDED FIRST, and the value checks below read the mode the bone WILL have - see
    # ops_create's note above _BONE_KEYS for the measurement.
    want_mode = take(params, "rotationMode", default=None, kind=str)
    modes = [m.identifier for m in bpy.types.PoseBone.bl_rna.properties["rotation_mode"].enum_items]
    if want_mode is not None and str(want_mode).upper() not in modes:
        raise MifOpError("rotationMode '%s' is not one this Blender offers: %s. NOTHING was changed."
                         % (want_mode, ", ".join(modes)))
    mode = str(want_mode).upper() if want_mode is not None else pb.rotation_mode
    if quat is not None and mode != "QUATERNION":
        raise MifOpError("'%s' is in %s rotation mode, so a quaternion would be ignored - set "
                         "rotation instead, or pass rotationMode QUATERNION with it. NOTHING was changed."
                         % (bone_name, mode))
    if rot is not None and mode == "QUATERNION":
        raise MifOpError("'%s' is in QUATERNION rotation mode, so a euler would be ignored - pass "
                         "quaternion instead, or pass rotationMode XYZ (or another euler order) with it. "
                         "NOTHING was changed." % bone_name)
    # AND AXIS_ANGLE, which ignores rotation_euler exactly as QUATERNION does and was let through: measured on 5.2.1
    # (2026-09-28, test_blender_review_defects RD122), rotation [1, 0, 0] on an AXIS_ANGLE bone answered ok, `written`
    # showed the euler, and the bone's tail did not move. Refused the same way - this op takes no axis-angle value, so the
    # way in is an Euler rotationMode.
    if rot is not None and mode == "AXIS_ANGLE":
        raise MifOpError("'%s' is in AXIS_ANGLE rotation mode, so a euler would be ignored - pass rotationMode XYZ (or "
                         "another euler order) with it, or QUATERNION with a quaternion. NOTHING was changed."
                         % bone_name)
    turn = _parse_turn(params.get("turn")) if params.get("turn") is not None else None
    # A CONNECTED BONE'S HEAD IS WELDED TO ITS PARENT'S TAIL, so a turn about any other pivot would have to move it -
    # Blender drops the translation and the bone ends up turned about its head instead, under ok (measured 2026-11-07:
    # a forearm turned about its tail kept its head and moved its tail).
    if turn is not None and turn["pivot"] != "head" and pb.bone.use_connect:
        raise MifOpError("'%s' is connected to '%s', so its head cannot leave that bone's tail and a turn about another "
                         "pivot cannot be made - pivot 'head', or turn '%s' instead. NOTHING was changed."
                         % (pb.name, pb.parent.name if pb.parent else "its parent",
                            pb.parent.name if pb.parent else "its parent"))
    ik = _parse_ik(obj, pb, params.get("ik")) if params.get("ik") is not None else None
    if ik is not None and (turn is not None or loc is not None or rot is not None or quat is not None
                           or scl is not None or params.get("mirror") is not None):
        raise MifOpError("ik writes the whole chain's pose itself - pass it alone (with rotationMode at most); a turn, "
                         "a mirror or channel values go in a call of their own. NOTHING was changed.")
    if loc is None and rot is None and quat is None and scl is None and want_mode is None and turn is None \
            and ik is None:
        raise MifOpError("nothing to set - pass location, rotation, quaternion, scale, rotationMode, turn or ik. "
                         "NOTHING was changed.")
    # THE MIRROR IS CHECKED BEFORE THE FIRST WRITE (2026-10-04): its partner, and that partner's rotation mode against
    # the mode this bone will have.
    mirror = None
    if params.get("mirror") is not None:
        from .ops_pose import check_mirror_spec   # noqa: PLC0415 - the op modules import ops_common, not each other
        mirror = check_mirror_spec(obj, pb, params["mirror"], mode)

    mode_before = pb.rotation_mode
    if want_mode is not None:
        pb.rotation_mode = mode

    if loc is not None:
        pb.location = loc
    if rot is not None:
        pb.rotation_euler = rot
    if quat is not None:
        pb.rotation_quaternion = quat
    if scl is not None:
        pb.scale = scl
    turned = _apply_turn(obj, pb, turn) if turn is not None else None
    solved = _apply_ik(obj, pb, ik) if ik is not None else None
    mirrored = None
    if mirror is not None:
        from .ops_pose import mirror_bone_pose   # noqa: PLC0415
        mirrored = mirror_bone_pose(obj, pb, mirror)

    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    epb = ev.pose.bones.get(str(bone_name)) if ev.pose is not None else None
    reply = {
        "object": obj.name,
        "bone": pb.name,
        "rotationMode": pb.rotation_mode,
        "rotationModeBefore": mode_before,
        # WHAT WAS WRITTEN, off the raw pose bone.
        "written": {
            "location": rnd(list(pb.location)),
            "rotationEuler": rnd(list(pb.rotation_euler)),
            "rotationQuaternion": rnd(list(pb.rotation_quaternion)),
            "scale": rnd(list(pb.scale)),
        },
        # WHERE IT ACTUALLY ENDED UP, after constraints. An IK chain, a Copy Rotation or a Limit
        # will move it somewhere else entirely, and pose_bone.matrix is the raw value.
        "evaluatedHeadWorld": rnd(list((ev.matrix_world @ epb.head))) if epb else None,
        "evaluatedTailWorld": rnd(list((ev.matrix_world @ epb.tail))) if epb else None,
        "constraints": [c.type for c in pb.constraints],
        "constrainedNote": ("evaluated* is read through the depsgraph and is where the bone really "
                            "is. With a constraint on the bone it will NOT match `written`, and "
                            "that difference is the constraint working rather than a fault."),
    }
    if mirrored is not None:
        reply["mirror"] = mirrored
    if turned is not None:
        reply["turn"] = turned
    if solved is not None:
        reply["ik"] = solved
    return reply


# ---------------------------------------------------------------------------------------------------------------------
# set_bone_pose turn and ik (1.4, 2026-11-07). A test character's poses were written as quaternions worked out by hand
# from each bone's roll-0 rest frame, and a crouch's feet floated because nothing planted them. Both write the pose
# bone's armature-space matrix (PoseBone.matrix), which Blender turns back into the bone's own channels in whatever
# rotation mode it has; every bone written is followed by a view-layer update, so a child reads its parent's new pose.
# ---------------------------------------------------------------------------------------------------------------------
_UNCHANGED = "NOTHING was changed."


def _world_vec(raw, key, axis_names=False):
    if axis_names and isinstance(raw, str) and raw.upper() in ("X", "Y", "Z"):
        return Vector([1.0 if raw.upper() == a else 0.0 for a in "XYZ"])
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise MifOpError("%s is [x, y, z] in world space%s, got %r. %s"
                         % (key, " or 'X' / 'Y' / 'Z'" if axis_names else "", raw, _UNCHANGED))
    return Vector(finite_floats(raw, key))


def _arm_point(obj, p):
    return obj.matrix_world.inverted() @ p


def _arm_dir(obj, d):
    return (obj.matrix_world.to_3x3().inverted() @ d).normalized()


def _evaluated_bone(obj, name):
    bpy.context.view_layer.update()
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return ev, ev.pose.bones[name]


def _parse_turn(raw):
    if not isinstance(raw, dict):
        raise MifOpError("turn is {axis, degrees, pivot}, got %r. %s" % (raw, _UNCHANGED))
    unknown = sorted(str(k) for k in raw if k not in ("axis", "degrees", "pivot"))
    if unknown:
        raise MifOpError("turn takes axis, degrees and pivot; %s %s not among them. %s"
                         % (", ".join(unknown), "is" if len(unknown) == 1 else "are", _UNCHANGED))
    if "axis" not in raw or "degrees" not in raw:
        raise MifOpError("turn needs axis (world, e.g. [0, 1, 0] or 'Y') and degrees. %s" % _UNCHANGED)
    axis = _world_vec(raw["axis"], "turn.axis", axis_names=True)
    if axis.length < 1e-9:
        raise MifOpError("turn.axis is a zero vector, which turns about nothing. %s" % _UNCHANGED)
    deg = finite_float(raw["degrees"], "turn.degrees")
    pivot = raw.get("pivot", "head")
    if isinstance(pivot, str):
        if pivot not in ("head", "tail"):
            raise MifOpError("turn.pivot is 'head', 'tail' or a world point [x, y, z], got %r. %s" % (pivot, _UNCHANGED))
    else:
        pivot = _world_vec(pivot, "turn.pivot")
    return {"axis": axis.normalized(), "degrees": deg, "pivot": pivot}


def _apply_turn(obj, pb, turn):
    from mathutils import Matrix   # noqa: PLC0415
    _, epb = _evaluated_bone(obj, pb.name)
    mw = obj.matrix_world
    head_before, tail_before = mw @ epb.head, mw @ epb.tail
    pv = turn["pivot"]
    pivot_w = head_before if pv == "head" else (tail_before if pv == "tail" else pv)
    pa = _arm_point(obj, pivot_w)
    rot = (Matrix.Translation(pa) @ Matrix.Rotation(math.radians(turn["degrees"]), 4, _arm_dir(obj, turn["axis"]))
           @ Matrix.Translation(-pa))
    pb.matrix = rot @ epb.matrix
    _, epb = _evaluated_bone(obj, pb.name)
    head_after, tail_after = mw @ epb.head, mw @ epb.tail
    # the turn measured back: the angle between the tail's arm from the pivot before and after, seen along the axis
    ax = turn["axis"]
    a = (tail_before - pivot_w) - ax * (tail_before - pivot_w).dot(ax)
    b = (tail_after - pivot_w) - ax * (tail_after - pivot_w).dot(ax)
    measured = math.degrees(math.atan2(ax.dot(a.cross(b)), a.dot(b))) if a.length > 1e-7 and b.length > 1e-7 else None
    out = {"axisWorld": rnd(list(ax)), "degrees": turn["degrees"], "pivotWorld": rnd(list(pivot_w)),
           "headBefore": rnd(list(head_before)), "tailBefore": rnd(list(tail_before)),
           "headAfter": rnd(list(head_after)), "tailAfter": rnd(list(tail_after)),
           "measuredDegrees": None if measured is None else round(measured, 4)}
    if measured is None:
        out["note"] = ("the tail lies on the axis, so the turn spun the bone about its own length and the angle cannot "
                       "be read off the tail")
    return out


def _parse_ik(obj, pb, raw):
    if not isinstance(raw, dict):
        raise MifOpError("ik is {target, pole, keepRotation}, got %r. %s" % (raw, _UNCHANGED))
    unknown = sorted(str(k) for k in raw if k not in ("target", "pole", "keepRotation"))
    if unknown:
        raise MifOpError("ik takes target, pole and keepRotation; %s %s not among them. %s"
                         % (", ".join(unknown), "is" if len(unknown) == 1 else "are", _UNCHANGED))
    mid = pb.parent
    root = mid.parent if mid is not None else None
    if mid is None or root is None:
        raise MifOpError("ik solves the two bones above '%s' (its parent and grandparent), and it has %s. Name the end "
                         "of a chain - the foot below a shin and a thigh. %s"
                         % (pb.name, "no parent" if mid is None else "no grandparent", _UNCHANGED))
    b_end, b_mid, b_root = pb.bone, mid.bone, root.bone
    for a, b, what in ((b_root.tail_local, b_mid.head_local, "'%s' does not start where '%s' ends" % (mid.name, root.name)),
                       (b_mid.tail_local, b_end.head_local, "'%s' does not start where '%s' ends" % (pb.name, mid.name))):
        if (a - b).length > 1e-3:
            raise MifOpError("ik needs a chain whose joints meet, and %s (%.4f apart in the rest pose). %s"
                             % (what, (a - b).length, _UNCHANGED))
    busy = [b.name for b in (root, mid, pb) if len(b.constraints)]
    if busy:
        raise MifOpError("%s carr%s constraints, which would move the bones after this solve writes them - mute them "
                         "first, or use Blender's own IK constraint. %s"
                         % (", ".join(busy), "ies" if len(busy) == 1 else "y", _UNCHANGED))
    target = raw.get("target")
    if target is None:
        raise MifOpError("ik needs target: a world point [x, y, z] for '%s's head, or 'rest' to plant it where it stands "
                         "in the rest pose. %s" % (pb.name, _UNCHANGED))
    tw = obj.matrix_world @ b_end.head_local if target == "rest" else _world_vec(target, "ik.target")
    pole = raw.get("pole")
    pw = _world_vec(pole, "ik.pole") if pole is not None else None
    keep = raw.get("keepRotation", "current")
    if keep not in ("current", "rest", "none"):
        raise MifOpError("ik.keepRotation is 'current' (the end keeps its world rotation), 'rest' (its rest rotation) "
                         "or 'none', got %r. %s" % (keep, _UNCHANGED))
    # reach and bend, checked on the pose as it stands, before anything is written
    _, er = _evaluated_bone(obj, root.name)
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    em = ev.pose.bones[mid.name]
    t = _arm_point(obj, tw)
    p0 = Vector(er.head)
    la, lb = (er.tail - er.head).length, (em.tail - em.head).length
    dist = (t - p0).length
    sc = obj.matrix_world.to_scale()[0]
    if dist > la + lb - 1e-6 or dist < abs(la - lb) + 1e-6:
        raise MifOpError("the target is %.4f from '%s's head and the chain reaches %.4f to %.4f (%s %.4f, %s %.4f), so "
                         "it cannot get there%s. Move the target, or the bone above the chain (a pelvis lower for a "
                         "crouch). %s" % (dist * sc, root.name, abs(la - lb) * sc, (la + lb) * sc, root.name, la * sc,
                                          mid.name, lb * sc,
                                          " without stretching" if dist > la + lb - 1e-6 else " - it is too near",
                                          _UNCHANGED))
    d = (t - p0).normalized()
    bend, bend_from = None, None
    if pw is not None:
        bend, bend_from = _arm_point(obj, pw) - p0, "pole"
    else:
        for knee, src in ((Vector(er.tail), "the chain as it stands"), (b_root.tail_local, "the rest pose")):
            v = (knee - p0) - d * (knee - p0).dot(d)
            if v.length > 1e-4 * max(la, 1e-6):
                bend, bend_from = knee - p0, src
                break
    if bend is None:
        raise MifOpError("the chain is straight now and in its rest pose, so '%s' could bend either way - pass pole, a "
                         "world point the joint should bend toward (in front of a knee). %s" % (mid.name, _UNCHANGED))
    side = bend - d * bend.dot(d)
    if side.length < 1e-9:
        raise MifOpError("ik.pole lies on the line from '%s' to the target, so it gives no direction to bend - move it "
                         "off that line. %s" % (root.name, _UNCHANGED))
    return {"target": tw, "keep": keep, "root": root, "mid": mid, "side": side.normalized(), "bendFrom": bend_from}


def _aim(obj, pb, head, to):
    """Turn pose bone pb about `head` (armature space) by the smallest rotation that points its tail at `to`."""
    from mathutils import Matrix   # noqa: PLC0415
    _, epb = _evaluated_bone(obj, pb.name)
    q = (epb.tail - epb.head).normalized().rotation_difference((to - head).normalized())
    pb.matrix = Matrix.Translation(head) @ q.to_matrix().to_4x4() @ Matrix.Translation(-head) @ epb.matrix


def _apply_ik(obj, pb, ik):
    from mathutils import Matrix   # noqa: PLC0415
    root, mid = ik["root"], ik["mid"]
    mw = obj.matrix_world
    chain = (root, mid, pb)
    ev, e_end = _evaluated_bone(obj, pb.name)
    before = {b.name: (mw @ ev.pose.bones[b.name].head, mw @ ev.pose.bones[b.name].tail) for b in chain}
    end_rot = {"current": e_end.matrix.to_3x3().normalized(), "rest": pb.bone.matrix_local.to_3x3().normalized(),
               "none": None}[ik["keep"]]
    end_rot_world = None if end_rot is None else (mw.to_3x3() @ end_rot).normalized()
    er, em = ev.pose.bones[root.name], ev.pose.bones[mid.name]
    p0 = Vector(er.head)
    t = _arm_point(obj, ik["target"])
    la, lb = (er.tail - er.head).length, (em.tail - em.head).length
    dist = (t - p0).length
    d = (t - p0).normalized()
    cos_a = max(-1.0, min(1.0, (la * la + dist * dist - lb * lb) / (2 * la * dist)))
    knee = p0 + d * la * cos_a + ik["side"] * la * math.sqrt(max(0.0, 1 - cos_a * cos_a))
    _aim(obj, root, p0, knee)
    _, em = _evaluated_bone(obj, mid.name)
    _aim(obj, mid, Vector(em.head), t)
    if end_rot is not None:
        _, e_end = _evaluated_bone(obj, pb.name)
        m = end_rot.to_4x4() @ Matrix.Diagonal(e_end.matrix.to_scale()).to_4x4()
        m.translation = e_end.head
        pb.matrix = m
    ev, e_end = _evaluated_bone(obj, pb.name)
    rot_err = None
    if end_rot_world is not None:
        now = (mw.to_3x3() @ e_end.matrix.to_3x3()).normalized()
        rot_err = math.degrees(end_rot_world.to_quaternion().rotation_difference(now.to_quaternion()).angle)
        rot_err = min(rot_err, 360.0 - rot_err)
    end_w = mw @ e_end.head
    sc = mw.to_scale()[0]
    return {"chain": [b.name for b in chain], "targetWorld": rnd(list(ik["target"])),
            "endHeadWorld": rnd(list(end_w)), "endErrorM": round(float((end_w - ik["target"]).length), 6),
            "jointWorld": rnd(list(mw @ ev.pose.bones[mid.name].head)), "bendFrom": ik["bendFrom"],
            "reachM": round(float((la + lb) * sc), 6), "distanceM": round(float(dist * sc), 6),
            "keepRotation": ik["keep"], "endRotationErrorDeg": None if rot_err is None else round(rot_err, 4),
            "before": {k: {"head": rnd(list(v[0])), "tail": rnd(list(v[1]))} for k, v in before.items()},
            "after": {b.name: {"head": rnd(list(mw @ ev.pose.bones[b.name].head)),
                               "tail": rnd(list(mw @ ev.pose.bones[b.name].tail))} for b in chain}}


def _remove_shape_key(obj, kb, params):
    """set_shape_key remove: drop one key, re-pointing keys relative to it at the basis first."""
    others = sorted(set(params) - {"object", "name", "key", "shapeKey", "remove"})
    if others:
        raise MifOpError("remove takes the key alone; drop %s. NOTHING was changed." % ", ".join(others))
    require_editable(obj, "remove a shape key from")
    sk = obj.data.shape_keys
    ref = sk.reference_key
    if kb == ref and len(sk.key_blocks) > 1:
        raise MifOpError("'%s' is the basis, and %d other key(s) are measured from it - removing it would make '%s' the "
                         "basis and move the mesh to that shape. Remove the others first, or apply_shape_keys. NOTHING "
                         "was changed." % (kb.name, len(sk.key_blocks) - 1, sk.key_blocks[1].name))
    repointed = [b.name for b in sk.key_blocks if b != kb and b.relative_key == kb]
    for b in sk.key_blocks:
        if b.name in repointed:
            b.relative_key = ref
    name = kb.name
    obj.shape_key_remove(kb)
    left = [b.name for b in obj.data.shape_keys.key_blocks] if obj.data.shape_keys else []
    return {"object": obj.name, "removed": name, "repointedToBasis": repointed, "keysAfter": left}


def op_set_shape_key(params):
    """Set a shape key's value, and optionally its range. The other half of list_shape_keys.

    params:
      object (str, required)
      key (str, required)      the shape key name
      value (float)            the influence, normally 0..1
      sliderMin / sliderMax    the allowed range
      mute (bool)
      remove (bool)            remove the key (2026-10-06). Keys relative to it are pointed at the basis first; the
                               basis itself is refused while other keys remain (removing it would make the next key
                               the basis and move the mesh) - apply_shape_keys bakes it instead
      moveTo (int) | makeBasis (true) | applyToBasis (true)   (2026-10-06) one, alone: the key moved to that
                               index (1 or more - the basis stays), made the basis (5.0+), or its offsets
                               at its value (and vertex group) added into the basis and every other key,
                               then the key removed (5.2's Apply to Basis, natively on every build)

    A value outside the slider range is CLAMPED BY BLENDER SILENTLY, so this reports what the key
    actually holds afterwards rather than what was asked for, and says when they differ.
    """
    reject_unknown(params, {"object", "name", "key", "shapeKey", "value", "sliderMin",
                            "sliderMax", "mute", "vertexGroup", "relativeKey", "remove", "moveTo", "makeBasis",
                            "applyToBasis"}, "set_shape_key")
    obj = get_object(take(params, "object", "name", required=True))
    if obj.data is None or getattr(obj.data, "shape_keys", None) is None:
        raise MifOpError("'%s' has no shape keys. NOTHING was changed." % obj.name)
    key_name = take(params, "key", "shapeKey", default=None, kind=str)
    if not key_name:
        raise MifOpError("'key' is required - which shape key. NOTHING was changed.")
    blocks = obj.data.shape_keys.key_blocks
    kb = blocks.get(str(key_name))
    if kb is None:
        raise MifOpError("no shape key named '%s' on '%s'. Present: %s. NOTHING was changed."
                         % (key_name, obj.name, ", ".join(b.name for b in blocks) or "<none>"))
    if any(params.get(k) is not None and params.get(k) is not False for k in ("moveTo", "makeBasis", "applyToBasis")):
        return _shape_key_stack(obj, kb, params)

    if take_bool(params, "remove", default=False):
        return _remove_shape_key(obj, kb, params)

    smin = take_float(params, "sliderMin", default=None)
    smax = take_float(params, "sliderMax", default=None)
    if smin is not None and smax is not None and smin > smax:
        raise MifOpError("sliderMin %g is above sliderMax %g. NOTHING was changed." % (smin, smax))

    requested = take_float(params, "value", default=None)
    # READ BEFORE THE FIRST WRITE (2026-09-21): mute was read after the range and value were written, so {value: 0.7,
    # mute: "flase"} changed the key and then refused.
    want_mute = take_bool(params, "mute", default=False) if "mute" in params else None
    vg, rel = _key_links(obj, params, kb.name, list(blocks.keys()), "NOTHING was changed.")
    if vg is not None:
        kb.vertex_group = vg
    if rel is not None:
        kb.relative_key = blocks[rel]
    # RANGE FIRST, then the value: setting a value outside the OLD range would be clamped to it
    # and then look wrong even though the new range would have allowed it.
    if smin is not None:
        kb.slider_min = smin
    if smax is not None:
        kb.slider_max = smax
    if requested is not None:
        kb.value = requested
    if want_mute is not None:
        kb.mute = want_mute

    actual = round(float(kb.value), 6)
    return {
        "object": obj.name,
        "key": kb.name,
        "requestedValue": requested,
        "value": actual,
        # BLENDER CLAMPS SILENTLY to the slider range. Asking for 2.0 on a 0..1 key leaves 1.0 and
        # reports nothing, so the difference is named rather than left to be discovered in a render.
        "clamped": requested is not None and abs(actual - requested) > 1e-6,
        "sliderMin": round(float(kb.slider_min), 6),
        "sliderMax": round(float(kb.slider_max), 6),
        "mute": bool(kb.mute),
        "vertexGroup": kb.vertex_group or None,
        "relativeKey": kb.relative_key.name if kb.relative_key is not None else None,
    }


_APPLY_SK_KEYS = {"object", "name", "mix", "keys", "keepBasis"}


def _key_coords(kb, n):
    flat = array.array("f", [0.0]) * (n * 3)
    kb.data.foreach_get("co", flat)
    return flat


def op_apply_shape_keys(params):
    """Bake shape keys into the basis: the whole current mix (mix true, the default) or the named keys at their current
    values, then remove what was applied.

    THE GAP: add_shape_key only adds; nothing baked the keys or removed one. A morphed head and body joined for a bake
    needed every shape key applied first (a mesh with keys refuses Armature apply and join), then the Armature
    modifier applied, then join, merge by distance and subdivision.

    mix true bakes what the keys show now - each key at its value, mute, vertex group and relative key honoured, read
    through Blender's own from-mix - into the base mesh, and removes every key (keepBasis true keeps a Basis holding
    the new shape). keys [names] bakes only those keys at their current values into the basis and removes them; every
    key left keeps its offset from the basis (the applied displacement is added to it as well), and keepBasis false
    clears the Basis too when no other key is left. Reply: the keys applied (value each), the keys before and after,
    maxDisplacement (how far the base mesh moved) and mixDeviation (the base mesh after against the shape the applied
    keys showed before: 0 when the bake is faithful).
    """
    reject_unknown(params, _APPLY_SK_KEYS, "apply_shape_keys")
    obj = get_object(take(params, "object", "name", required=True, kind=str))
    if obj.type != "MESH":
        raise MifOpError("'%s' is a %s; apply_shape_keys bakes a mesh's keys. NOTHING was changed." % (obj.name, obj.type))
    require_editable(obj, "apply shape keys on")
    sk = obj.data.shape_keys
    if sk is None or not sk.key_blocks:
        raise MifOpError("'%s' has no shape keys to apply. NOTHING was changed." % obj.name)
    if not sk.use_relative:
        raise MifOpError("'%s' uses absolute shape keys (evaluated by time, not by value); there is no mix of values to "
                         "bake. NOTHING was changed." % obj.name)
    names = params.get("keys")
    mix = take_bool(params, "mix", default=names is None)
    keep_basis = take_bool(params, "keepBasis", default=False)
    blocks = sk.key_blocks
    ref = sk.reference_key
    if names is not None:
        if mix:
            raise MifOpError("pass mix true (every key, as shown) OR keys [names] - not both. NOTHING was changed.")
        if not isinstance(names, list) or not names or not all(isinstance(k, str) for k in names):
            raise MifOpError("keys must be a non-empty list of shape key names. NOTHING was changed.")
        missing = [k for k in names if blocks.get(k) is None]
        if missing:
            raise MifOpError("no shape keys %s on '%s' (it has %s). NOTHING was changed."
                             % (missing, obj.name, ", ".join(b.name for b in blocks)))
        if ref.name in names:
            raise MifOpError("'%s' is the basis itself - name the keys to bake INTO it. NOTHING was changed." % ref.name)
        names = list(dict.fromkeys(names))
    elif not mix:
        raise MifOpError("mix false with no keys names nothing to apply. NOTHING was changed.")
    before_keys = [b.name for b in blocks]
    n = len(obj.data.vertices)
    base = array.array("f", [0.0]) * (n * 3)
    obj.data.vertices.foreach_get("co", base)
    basis = _key_coords(ref, n)
    applied = [{"key": b.name, "value": round(float(b.value), 6), "mute": bool(b.mute)}
               for b in blocks if b != ref and (names is None or b.name in names)]
    # THE MIX THROUGH BLENDER'S OWN EVALUATION: a temporary key from_mix holds exactly what the viewport shows, vertex
    # groups and relative keys included. For named keys every other key is muted for that read and put back.
    mutes = {b.name: b.mute for b in blocks}
    active = obj.active_shape_key_index
    try:
        if names is not None:
            for b in blocks:
                if b != ref and b.name not in names:
                    b.mute = True
        tmp = obj.shape_key_add(name="__mif_apply_mix", from_mix=True)
        target = _key_coords(tmp, n)
        obj.shape_key_remove(tmp)
    finally:
        for b in obj.data.shape_keys.key_blocks:
            if b.name in mutes:
                b.mute = mutes[b.name]
        obj.active_shape_key_index = min(active, len(obj.data.shape_keys.key_blocks) - 1)
    delta = [t - b for t, b in zip(target, basis)]
    sk = obj.data.shape_keys
    if mix:
        for b in list(sk.key_blocks)[::-1]:
            if b != sk.reference_key:
                obj.shape_key_remove(b)
        sk.reference_key.data.foreach_set("co", target)
        obj.data.vertices.foreach_set("co", target)
        if not keep_basis:
            obj.shape_key_clear()
    else:
        for name in names:
            obj.shape_key_remove(sk.key_blocks[name])
        for b in sk.key_blocks:
            co = _key_coords(b, n)
            b.data.foreach_set("co", array.array("f", [c + d for c, d in zip(co, delta)]))
        obj.data.vertices.foreach_set("co", target)
        if not keep_basis and len(sk.key_blocks) == 1:
            obj.shape_key_clear()
    obj.data.update()
    bpy.context.view_layer.update()
    after = array.array("f", [0.0]) * (n * 3)
    obj.data.vertices.foreach_get("co", after)

    def worst(a, b):
        return max((math.dist(a[i:i + 3], b[i:i + 3]) for i in range(0, len(a), 3)), default=0.0)
    keys_after = [b.name for b in obj.data.shape_keys.key_blocks] if obj.data.shape_keys else []
    out = {"object": obj.name, "mode": "mix" if mix else "keys", "applied": applied, "keepBasis": keep_basis,
           "keysBefore": before_keys, "keysAfter": keys_after, "maxDisplacement": round(worst(after, base), 6),
           "mixDeviation": round(worst(after, target), 6)}
    out.update(shared_data_note(obj))
    return out


def _shape_key_stack(obj, kb, params):
    """set_shape_key moveTo | makeBasis | applyToBasis - see its docstring. Every refusal before the first write."""
    asked = [k for k in ("moveTo", "makeBasis", "applyToBasis") if params.get(k) is not None
             and params.get(k) is not False]
    others = [k for k in ("value", "sliderMin", "sliderMax", "mute", "vertexGroup", "relativeKey", "remove")
              if params.get(k) is not None]
    if len(asked) > 1 or others:
        raise MifOpError("%s acts alone; got %s. NOTHING was changed." % (asked[0], ", ".join(asked[1:] + others)))
    require_editable(obj, "reorder shape keys on")
    key = obj.data.shape_keys
    blocks = key.key_blocks
    names = [b.name for b in blocks]
    at = names.index(kb.name)
    out = {"object": obj.name, "key": kb.name, "orderBefore": names}
    if asked[0] == "moveTo":
        to = take_int(params, "moveTo")
        if not 1 <= to < len(names):
            raise MifOpError("moveTo %d: a key moves to 1 to %d - index 0 is the basis (makeBasis replaces it). NOTHING "
                             "was changed." % (to, len(names) - 1))
        if at == 0:
            raise MifOpError("'%s' is the basis; makeBasis on another key replaces it. NOTHING was changed." % kb.name)
        obj.active_shape_key_index = at
        with bpy.context.temp_override(object=obj, active_object=obj):
            for _step in range(abs(to - at)):
                bpy.ops.object.shape_key_move(type="DOWN" if to > at else "UP")
    elif asked[0] == "makeBasis":
        take_bool(params, "makeBasis")
        if "shape_key_make_basis" not in dir(bpy.ops.object):
            raise MifOpError("makeBasis needs Blender 5.0 or later (this is %s). NOTHING was changed."
                             % bpy.app.version_string)
        if at == 0:
            raise MifOpError("'%s' is already the basis. NOTHING was changed." % kb.name)
        obj.active_shape_key_index = at
        with bpy.context.temp_override(object=obj, active_object=obj):
            bpy.ops.object.shape_key_make_basis()
        # THE MESH FOLLOWS ITS NEW BASIS: 5.2.2 writes it, 5.0.1 left the old basis in the mesh (measured).
        co = [0.0] * (len(obj.data.vertices) * 3)
        key.reference_key.data.foreach_get("co", co)
        obj.data.vertices.foreach_set("co", co)
        obj.data.update()
    else:
        take_bool(params, "applyToBasis")
        if at == 0:
            raise MifOpError("'%s' is the basis; applyToBasis takes another key into it. NOTHING was changed."
                             % kb.name)
        n = len(obj.data.vertices)
        mine, rel = [0.0] * (n * 3), [0.0] * (n * 3)
        kb.data.foreach_get("co", mine)
        kb.relative_key.data.foreach_get("co", rel)
        weights = [kb.value] * n
        if kb.vertex_group:
            gi = obj.vertex_groups[kb.vertex_group].index
            weights = [kb.value * next((g.weight for g in v.groups if g.group == gi), 0.0) for v in obj.data.vertices]
        delta = [(mine[i] - rel[i]) * weights[i // 3] for i in range(n * 3)]
        for other in blocks:
            if other == kb:
                continue
            co = [0.0] * (n * 3)
            other.data.foreach_get("co", co)
            other.data.foreach_set("co", [a + d for a, d in zip(co, delta)])
        co = [0.0] * (n * 3)
        obj.data.vertices.foreach_get("co", co)
        obj.data.vertices.foreach_set("co", [a + d for a, d in zip(co, delta)])
        out["appliedValue"] = round(float(kb.value), 6)
        out["maxOffset"] = round(max((abs(d) for d in delta), default=0.0), 6)
        obj.shape_key_remove(kb)
        obj.data.update()
    out["orderAfter"] = [b.name for b in obj.data.shape_keys.key_blocks] if obj.data.shape_keys else []
    out["basis"] = out["orderAfter"][0] if out["orderAfter"] else None
    return out


_VGROUP_KEYS = {"object", "name", "group", "vertices", "weight", "weights", "mode", "create",
                "remove", "byAxis", "allVertices", "exclusive", "byRadius",
                # modes MIRROR, SMOOTH and QUANTIZE (2026-10-04) - read by _mirror_or_smooth
                "mirrorAxis", "mirrorPair", "tolerance", "factor", "repeat", "expand", "steps",
                # group admin (2026-10-04) - read by _group_admin
                "deleteGroups", "armature", "lock", "groups",
                # modes CLEAN, INVERT and LEVELS (1.3) - read by _clean_invert_levels
                "threshold", "keepSingle", "offset", "gain", "autoAssign", "autoRemove"}
_BYRADIUS_KEYS = {"center", "radius", "falloff", "weight", "space"}
_BYAXIS_KEYS = {"axis", "from", "to", "weightFrom", "weightTo", "space"}
_SHAPEKEY_KEYS = {"object", "name", "key", "fromMix", "fromObject", "value", "sliderMin", "sliderMax", "vertexGroup",
                  "relativeKey"}


def _key_links(obj, params, own_name, keys_now, tail):
    """(vertex group name or "" to clear or None, relative key name or None) for a shape key, checked before any write.
    A shape key has no invert flag on its vertex group in Blender (ShapeKey carries vertex_group and relative_key only),
    so none is offered."""
    vg = params.get("vertexGroup")
    if vg is not None:
        if not isinstance(vg, str):
            raise MifOpError("vertexGroup is a vertex group's name (\"\" clears it), got %r. %s" % (vg, tail))
        if vg and obj.vertex_groups.get(vg) is None:
            raise MifOpError("'%s' has no vertex group %r (it has %s). %s"
                             % (obj.name, vg, ", ".join(g.name for g in obj.vertex_groups) or "none", tail))
    rel = params.get("relativeKey")
    if rel is not None:
        if not isinstance(rel, str) or rel not in keys_now:
            raise MifOpError("relativeKey %r is not a shape key of '%s' (it has %s). %s"
                             % (rel, obj.name, ", ".join(keys_now) or "none", tail))
        if rel == own_name:
            raise MifOpError("a key cannot be relative to itself. %s" % tail)
    return vg, rel

_VGROUP_MODES = ("REPLACE", "ADD", "SUBTRACT")


_VG_MIRROR_SMOOTH_KEYS = {"object", "name", "group", "mode", "mirrorAxis", "mirrorPair", "tolerance", "factor",
                          "repeat", "expand", "vertices", "steps"}
_RADIAL_CURVES = {
    "SMOOTH": lambda t: 3.0 * t * t - 2.0 * t * t * t,
    "SPHERE": lambda t: math.sqrt(max(0.0, 2.0 * t - t * t)),
    "ROOT": lambda t: math.sqrt(t),
    "SHARP": lambda t: t * t,
    "LINEAR": lambda t: t,
    "CONSTANT": lambda t: 1.0,
}


_GROUP_ADMIN_OTHERS = ("vertices", "weight", "weights", "mode", "create", "remove", "byAxis", "allVertices",
                       "exclusive", "byRadius")


def _group_admin(params):
    """set_vertex_weights deleteGroups / lock - see its docstring. Every name resolved before the first write."""
    for key in _GROUP_ADMIN_OTHERS:
        if params.get(key) not in (None, False):
            raise MifOpError("%s writes weights; deleteGroups and lock act on whole groups. Make two calls. NOTHING was "
                             "changed." % key)
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    require_editable(obj, "edit vertex groups on")
    have = [g.name for g in obj.vertex_groups]
    lock = take_bool(params, "lock") if params.get("lock") is not None else None
    lock_names = []
    if lock is not None:
        one, many = params.get("group") or params.get("name"), params.get("groups")
        if (one is None) == (many is None):
            raise MifOpError("lock needs group (one name) or groups (a list). NOTHING was changed.")
        lock_names = [one] if many is None else list(many) if isinstance(many, (list, tuple)) else None
        if lock_names is None or any(not isinstance(n, str) for n in lock_names):
            raise MifOpError("groups is a list of vertex group names. NOTHING was changed.")
        missing = [n for n in lock_names if n not in have]
        if missing:
            raise MifOpError("'%s' has no vertex group %s. NOTHING was changed." % (obj.name, ", ".join(missing)))
    elif params.get("groups") is not None:
        raise MifOpError("groups goes with lock. NOTHING was changed.")
    raw = params.get("deleteGroups")
    doomed = []
    mode = None
    if raw is not None:
        if isinstance(raw, str) and raw.upper() in ("ALL", "UNLOCKED", "NON_BONE"):
            mode = raw.upper()
        elif isinstance(raw, str):
            mode, raw = "NAMES", [raw]
        elif isinstance(raw, (list, tuple)) and raw and all(isinstance(n, str) for n in raw):
            mode = "NAMES"
        else:
            raise MifOpError("deleteGroups is a name, a list of names, ALL, UNLOCKED or NON_BONE; got %r. NOTHING was "
                             "changed." % (raw,))
        if params.get("armature") is not None and mode != "NON_BONE":
            raise MifOpError("armature goes with deleteGroups NON_BONE. NOTHING was changed.")
        locked_after = {g.name: (lock if g.name in lock_names else g.lock_weight) for g in obj.vertex_groups} \
            if lock is not None else {g.name: g.lock_weight for g in obj.vertex_groups}
        if mode == "NAMES":
            missing = [n for n in raw if n not in have]
            if missing:
                raise MifOpError("'%s' has no vertex group %s (it has %s). NOTHING was changed."
                                 % (obj.name, ", ".join(repr(n) for n in missing), ", ".join(have) or "none"))
            doomed = list(dict.fromkeys(raw))
        elif mode == "ALL":
            doomed = list(have)
        elif mode == "UNLOCKED":
            doomed = [n for n in have if not locked_after[n]]
        else:
            arm_name = params.get("armature")
            if arm_name is not None:
                arm = bpy.data.objects.get(str(arm_name))
            else:
                mod = next((m for m in obj.modifiers if m.type == "ARMATURE" and m.object is not None), None)
                arm = mod.object if mod is not None else None
            if arm is None or arm.type != "ARMATURE":
                raise MifOpError("NON_BONE keeps the groups named for a bone of an armature: pass armature, or give "
                                 "'%s' an Armature modifier. NOTHING was changed." % obj.name)
            bones = {b.name for b in arm.data.bones}
            doomed = [n for n in have if n not in bones]
        if not doomed:
            raise MifOpError("deleteGroups %s matches no vertex group of '%s', so nothing would be deleted - and that is "
                             "not a success. NOTHING was changed." % (mode, obj.name))
    before = [(g.name, bool(g.lock_weight)) for g in obj.vertex_groups]
    for n in lock_names:
        obj.vertex_groups[n].lock_weight = lock
    for n in doomed:
        obj.vertex_groups.remove(obj.vertex_groups[n])
    after = [(g.name, bool(g.lock_weight)) for g in obj.vertex_groups]
    return {"object": obj.name, "deleteMode": mode, "deleted": doomed, "locked": lock_names if lock else [],
            "unlocked": lock_names if lock is False else [], "groupsBefore": len(before), "groupsAfter": len(after),
            "groups": [{"name": n, "locked": lk} for n, lk in after]}


def op_set_vertex_weights(params):
    """Create a vertex group and put weights in it - which nothing in this addon could do.

    THE HOLE THIS CLOSES, found by asking what the addon consumes and cannot produce - the fourth
    time that question has paid, after collections, empties and armatures. list_vertex_groups,
    normalize_weights and transfer_weights all operate on vertex groups, and grepping the whole
    addon for vertex_groups.new returned NOTHING. So a rig imported with weights could be
    normalised and transferred, and a rig built here could not be weighted at all. transfer_weights
    refuses rather than producing an unskinned result, which is that op being honest about a hole it
    could not fill.

    THE POSTCONDITION IS THE WEIGHTS, NOT THE GROUP. A vertex group that exists with every weight at
    zero deforms nothing and reads back perfectly - it has a name, an index, and it appears in
    list_vertex_groups exactly like a working one. So this reads the stored weight back for every
    vertex it touched and reports the range, and reports how many vertices ended up with a NON-ZERO
    weight, which is the number that decides whether anything will actually move.

    vg.weight(i) RAISES RuntimeError for a vertex that is not in the group rather than returning 0,
    on every build, so the read-back catches per vertex rather than assuming membership.

    params:
      object (str)              required, must be a MESH
      group / name (str)        the vertex group. Created if absent unless create:false.
      vertices (list[int])      which vertices. Omit to create an empty group.
      weight (float)            one weight for all of them. Default 1.0.
      weights (list[float])     a weight per vertex, parallel to `vertices`. Excludes `weight`.
      mode (str)                REPLACE (default) | ADD | SUBTRACT
      create (bool)             create the group if it does not exist. Default true.
      remove (bool)             REMOVE the listed vertices from the group instead of weighting them
      byAxis (dict)             a weight per vertex FROM WHERE IT SITS: {axis X|Y|Z, from, to, weightFrom (default 1),
                                weightTo (default 0), space WORLD (default) | LOCAL}. A vertex at `from` gets weightFrom,
                                one at `to` gets weightTo, linearly between, clamped beyond. Every vertex unless
                                `vertices` narrows it. Excludes weight, weights and remove.
      allVertices (bool)        every vertex of the mesh. Excludes `vertices` and `weights`.
      mode MIRROR               copy this group's weights across mirrorAxis (X default) onto the vertex mirrored there
                                (nearest within tolerance, 1e-4 mesh units) in the group with the other side's suffix
                                (mirrorPair, default ['_l', '_r'] - upperarm_l writes upperarm_r, created if missing); a
                                name with neither suffix has its + half copied onto its - half. verticesWithoutPartner counts
                                vertices with no mirror vertex (an asymmetric mesh).
      mode SMOOTH               each vertex in the group (or `vertices`) moves `factor` (0.5) of the way to the mean of
                                its edge neighbours' weights (0 for a neighbour outside the group), `repeat` times (1), all
                                at once per pass; expand also smooths into the vertices next to the group.
      byRadius (dict)           a radial ramp: {center [x,y,z], radius, falloff SMOOTH|SPHERE|ROOT|SHARP|LINEAR|CONSTANT
                                (LINEAR default), weight (1), space WORLD|LOCAL}: a vertex at distance d < radius gets
                                weight x falloff(1 - d / radius) with this call's mode (ADD adds, SUBTRACT takes away);
                                vertices outside the radius are not touched.
      deleteGroups              DELETE whole groups (remove takes vertices out of one): a name, a list of names,
                                "ALL", "UNLOCKED" (every group not locked) or "NON_BONE" (every group that is not a
                                bone of `armature`, default the mesh's Armature modifier's object). Takes no weights.
      lock (bool) + group|groups lock or unlock groups (lock_weight), so a later UNLOCKED delete keeps them; with
                                deleteGroups the locks are written first. (2026-10-04)
      exclusive (bool)          then take the weighted vertices OUT of every other vertex group, so this group is the
                                only one they are in - a rigid piece (a belt, a holster) on one bone after a weight
                                transfer gave it the thigh's. REPLACE only. otherGroupsCleared names each group and how
                                many vertices left it; non-bone groups (a cloth pin group) are cleared too, so read it.

    CLEAN / INVERT / LEVELS (1.3): mode CLEAN removes weights under threshold (keepSingle leaves no vertex unweighted),
    INVERT writes 1 - w (autoAssign, autoRemove), LEVELS writes (w + offset) * gain clamped - on group, or groups
    [names] | "DEFORM" | "ALL". See _clean_invert_levels.

    BY POSITION, since 2026-09-13. No op returns vertex positions, so a weight per INDEX could not say "1 at the bottom
    fading to 0 halfway up" - and that is exactly what pins a soft body's base under a loose top.
    """
    reject_unknown(params, _VGROUP_KEYS, "set_vertex_weights")
    if params.get("deleteGroups") is not None or params.get("lock") is not None:
        return _group_admin(params)
    if str(params.get("mode") or "").upper() in ("CLEAN", "INVERT", "LEVELS"):
        return _clean_invert_levels(params)
    for key in ("threshold", "keepSingle", "offset", "gain", "autoAssign", "autoRemove"):
        if params.get(key) is not None:
            raise MifOpError("%s is for mode CLEAN, INVERT or LEVELS. NOTHING was changed." % key)
    for key in ("armature", "groups"):
        if params.get(key) is not None:
            raise MifOpError("%s goes with deleteGroups or lock. NOTHING was changed." % key)
    if str(params.get("mode") or "").upper() in ("MIRROR", "SMOOTH", "QUANTIZE"):
        return _mirror_or_smooth(params)
    for key in ("mirrorAxis", "mirrorPair", "tolerance", "factor", "repeat", "expand", "steps"):
        if params.get(key) is not None:
            # "MIRROR or SMOOTH" is pinned by test_blender_rigging_weights' refusal of factor.
            raise MifOpError("%s is for mode %s. NOTHING was changed."
                             % (key, "QUANTIZE" if key == "steps" else "MIRROR or SMOOTH"))
    src = take(params, "object", required=True, kind=str)
    group_name = take(params, "group", "name", required=True, kind=str)
    if len(str(group_name)) > _ID_NAME_LIMIT:
        raise MifOpError("the group name is %d characters and Blender truncates at %d on every "
                         "build, so the group you get would not be the one you named. NOTHING was "
                         "changed." % (len(str(group_name)), _ID_NAME_LIMIT))
    mode = str(take(params, "mode", default="REPLACE", kind=str)).upper()
    if mode not in _VGROUP_MODES:
        raise MifOpError("mode must be one of %s, MIRROR, SMOOTH, QUANTIZE, CLEAN, INVERT or LEVELS, got '%s'. NOTHING "
                         "was changed."
                         % (", ".join(_VGROUP_MODES), mode))
    create = take_bool(params, "create", default=True)
    remove = take_bool(params, "remove", default=False)
    all_vertices = take_bool(params, "allVertices", default=False)
    exclusive = take_bool(params, "exclusive", default=False)

    raw_verts = params.get("vertices")
    if raw_verts is not None and not isinstance(raw_verts, (list, tuple)):
        raise MifOpError("'vertices' must be a list of vertex indices, got %s. NOTHING was changed."
                         % type(raw_verts).__name__)
    raw_weights = params.get("weights")
    single = take_float(params, "weight", default=None)
    if raw_weights is not None and single is not None:
        raise MifOpError("pass weight OR weights, not both - one is a value for every vertex and "
                         "the other is a value per vertex, and they cannot both win. NOTHING was "
                         "changed.")
    if raw_weights is not None:
        if not isinstance(raw_weights, (list, tuple)):
            raise MifOpError("'weights' must be a list of numbers, got %s. NOTHING was changed."
                             % type(raw_weights).__name__)
        if raw_verts is None or len(raw_weights) != len(raw_verts):
            raise MifOpError("'weights' has %d entries and 'vertices' has %s - they are parallel "
                             "lists and must match. NOTHING was changed."
                             % (len(raw_weights), len(raw_verts) if raw_verts is not None else 0))
        # FINITE, HERE - the write further down used a bare float(w), and a NaN weight was stored in the
        # group under ok:true (5.0.1, 2026-09-20; audit_unguarded_numbers' provenance pass).
        try:
            raw_weights = finite_floats(raw_weights, "weights")
        except MifOpError as exc:
            raise MifOpError("%s NOTHING was changed." % exc)
    if remove and (raw_weights is not None or single is not None):
        raise MifOpError("remove:true takes vertices OUT of the group, so a weight has nothing to "
                         "apply to. NOTHING was changed.")
    if all_vertices and (raw_verts is not None or raw_weights is not None):
        raise MifOpError("allVertices:true weights every vertex, so it takes neither 'vertices' nor a per-vertex "
                         "'weights' list. NOTHING was changed.")
    # A RIGID PIECE ON ONE BONE (2026-10-04, the police uniform's duty belt and holster): a weight transfer gave them
    # the thigh's weights, and the fix was a raw-Python loop emptying every other group.
    if exclusive and (remove or mode != "REPLACE"):
        raise MifOpError("exclusive:true makes this group the only one the vertices are in, at the weight given - it "
                         "takes mode REPLACE and no remove. NOTHING was changed.")
    if exclusive and raw_verts is None and not all_vertices and params.get("byAxis") is None:
        raise MifOpError("exclusive:true needs the vertices it applies to: vertices, allVertices:true or byAxis. "
                         "NOTHING was changed.")

    # THE RAMP IS VALIDATED IN FULL HERE, before the object is even resolved - so every refusal changes nothing.
    by_axis = params.get("byAxis")
    by_axis_report = None
    if by_axis is not None:
        if not isinstance(by_axis, dict):
            raise MifOpError("'byAxis' must be an object like {axis: 'Z', from: 0.0, to: 1.0, weightFrom: 1.0, "
                             "weightTo: 0.0}, got %s. NOTHING was changed." % type(by_axis).__name__)
        unknown = sorted(set(by_axis) - _BYAXIS_KEYS)
        if unknown:
            raise MifOpError("byAxis does not take %s - it takes %s. NOTHING was changed."
                             % (", ".join(unknown), ", ".join(sorted(_BYAXIS_KEYS))))
        if raw_weights is not None or single is not None or remove:
            raise MifOpError("byAxis COMPUTES a weight per vertex from where it sits, so it cannot be combined with "
                             "weight, weights or remove. NOTHING was changed.")

        def _ramp_num(key, default):
            value = by_axis.get(key, default)
            if value is None:
                raise MifOpError("byAxis.%s is required. NOTHING was changed." % key)
            # finite_float, the addon's one scalar parser, rather than a bare float() and its own copy of the
            # NaN/Infinity test (audit_unguarded_numbers, 2026-09-13). Its refusal gains this op's promise.
            try:
                return finite_float(value, "byAxis." + key)
            except MifOpError as exc:
                raise MifOpError("%s NOTHING was changed." % exc)

        axis = str(by_axis.get("axis", "Z")).upper()
        if axis not in ("X", "Y", "Z"):
            raise MifOpError("byAxis.axis must be X, Y or Z, got %r. NOTHING was changed." % by_axis.get("axis"))
        space = str(by_axis.get("space", "WORLD")).upper()
        if space not in ("WORLD", "LOCAL"):
            raise MifOpError("byAxis.space must be WORLD or LOCAL, got %r. NOTHING was changed."
                             % by_axis.get("space"))
        a_from, a_to = _ramp_num("from", None), _ramp_num("to", None)
        w_from, w_to = _ramp_num("weightFrom", 1.0), _ramp_num("weightTo", 0.0)
        if a_from == a_to:
            raise MifOpError("byAxis.from and byAxis.to are both %g - a ramp needs a length. NOTHING was changed."
                             % a_from)
        if not (0.0 <= w_from <= 1.0 and 0.0 <= w_to <= 1.0):
            raise MifOpError("byAxis weights must lie in 0-1 (a vertex group weight is clamped there), got "
                             "weightFrom %g and weightTo %g. NOTHING was changed." % (w_from, w_to))
        by_axis_report = {"axis": axis, "from": a_from, "to": a_to, "weightFrom": w_from, "weightTo": w_to,
                          "space": space}

    by_radius = params.get("byRadius")
    by_radius_report = None
    if by_radius is not None:
        if not isinstance(by_radius, dict) or set(by_radius) - _BYRADIUS_KEYS:
            raise MifOpError("byRadius is {center [x,y,z], radius, falloff, weight, space}; got %r. NOTHING was changed."
                             % (by_radius,))
        if by_axis is not None or raw_weights is not None or single is not None or remove or all_vertices:
            raise MifOpError("byRadius computes each weight from the distance to its centre, so it takes no byAxis, "
                             "weight, weights, remove or allVertices. NOTHING was changed.")
        c = by_radius.get("center")
        if not isinstance(c, (list, tuple)) or len(c) != 3:
            raise MifOpError("byRadius.center must be [x,y,z]. NOTHING was changed.")
        try:
            r_center = finite_floats(list(c), "byRadius.center")
            r_radius = finite_float(by_radius.get("radius"), "byRadius.radius")
            r_weight = finite_float(by_radius.get("weight", 1.0), "byRadius.weight")
        except MifOpError as exc:
            raise MifOpError("%s NOTHING was changed." % exc)
        if not r_radius > 0.0 or not 0.0 <= r_weight <= 1.0:
            raise MifOpError("byRadius.radius must be above 0 and weight 0-1. NOTHING was changed.")
        r_fall = str(by_radius.get("falloff", "LINEAR")).upper()
        if r_fall not in _RADIAL_CURVES:
            raise MifOpError("byRadius.falloff must be one of %s, got %r. NOTHING was changed."
                             % (", ".join(_RADIAL_CURVES), by_radius.get("falloff")))
        r_space = str(by_radius.get("space", "WORLD")).upper()
        if r_space not in ("WORLD", "LOCAL"):
            raise MifOpError("byRadius.space must be WORLD or LOCAL. NOTHING was changed.")
        by_radius_report = {"center": r_center, "radius": r_radius, "falloff": r_fall, "weight": r_weight,
                            "space": r_space}

    obj = get_object(src, want_mesh=True)
    mesh = obj.data
    verts = list(range(len(mesh.vertices))) if all_vertices else []
    if raw_verts is not None:
        try:
            # finite_int per element: a vertex index of 2**40 is not a mesh error, it is a
            # raw ValueError out of Blender with no sentence explaining it.
            verts = [finite_int(i, "vertices") for i in raw_verts]
        except (TypeError, ValueError):
            raise MifOpError("'vertices' must be whole numbers. NOTHING was changed.")
        highest = len(mesh.vertices) - 1
        bad = sorted(i for i in verts if i < 0 or i > highest)
        if bad:
            raise MifOpError("vertex index %d is out of range - '%s' has %d vertice(s), so the "
                             "highest index is %d. NOTHING was changed."
                             % (bad[0], obj.name, len(mesh.vertices), highest))
    if by_axis_report is not None:
        # A WEIGHT PER VERTEX FROM ITS POSITION, then the same write and read-back as a weights list.
        if raw_verts is None:
            verts = list(range(len(mesh.vertices)))
        which = "XYZ".index(by_axis_report["axis"])
        world = by_axis_report["space"] == "WORLD"
        mw = obj.matrix_world
        raw_weights = []
        for i in verts:
            co = mesh.vertices[i].co
            coord = (mw @ co)[which] if world else co[which]
            t = min(1.0, max(0.0, (coord - a_from) / (a_to - a_from)))
            raw_weights.append(w_from + t * (w_to - w_from))

    if by_radius_report is not None:
        # A WEIGHT PER VERTEX FROM ITS DISTANCE TO THE CENTRE, inside the radius only; then the usual write and read-back.
        pool = verts if raw_verts is not None else list(range(len(mesh.vertices)))
        mw = obj.matrix_world
        cen = by_radius_report["center"]
        curve = _RADIAL_CURVES[by_radius_report["falloff"]]
        verts, raw_weights = [], []
        for i in pool:
            co = mesh.vertices[i].co
            p = (mw @ co) if by_radius_report["space"] == "WORLD" else co
            d = math.sqrt(sum((p[k] - cen[k]) ** 2 for k in range(3)))
            if d < r_radius:
                verts.append(i)
                raw_weights.append(r_weight * curve(1.0 - d / r_radius))
        if not verts:
            raise MifOpError("no vertex of '%s' lies within byRadius.radius %g of %s. NOTHING was changed."
                             % (obj.name, r_radius, cen))

    group = obj.vertex_groups.get(group_name)
    created = False
    if group is None:
        if not create:
            known = [g.name for g in obj.vertex_groups][:25]
            raise MifOpError("'%s' has no vertex group named '%s' and create:false was given. "
                             "Groups: %s. NOTHING was changed."
                             % (obj.name, group_name, ", ".join(known) if known else "<none>"))
        if remove:
            raise MifOpError("'%s' has no vertex group named '%s', so there is nothing to remove "
                             "vertices from. NOTHING was changed." % (obj.name, group_name))
        group = obj.vertex_groups.new(name=str(group_name))
        created = True
        # A DUPLICATE NAME GIVES THE NEW GROUP A .001 SUFFIX and leaves the incumbent alone -
        # uniform on 3.6, 4.4 and 5.0, unlike OBJECT renaming which reverses across the 3.6/4.x
        # line. It is reported rather than refused because a second group is a legitimate thing to
        # want, but a caller who then looks for their name by string needs to know.
        if group.name != str(group_name):
            created = True

    weights_before = {}
    for i in verts:
        try:
            weights_before[i] = round(group.weight(i), 6)
        except RuntimeError:
            weights_before[i] = None      # not in the group at all - a distinct state from 0.0

    if verts:
        if remove:
            group.remove(verts)
        elif raw_weights is not None:
            for index, w in zip(verts, raw_weights):
                group.add([index], w, mode)       # finite already - parsed above, or computed by byAxis
        else:
            group.add(verts, 1.0 if single is None else single, mode)

    # READ BACK PER VERTEX. vg.weight() raises for a vertex outside the group rather than returning
    # zero, so membership and a zero weight are distinguishable - and they are different states: one
    # deforms nothing, the other is not in the group to be normalised or transferred at all.
    after, nonzero, missing = {}, 0, []
    for i in verts:
        try:
            value = round(group.weight(i), 6)
            after[i] = value
            if value > 0.0:
                nonzero += 1
        except RuntimeError:
            after[i] = None
            missing.append(i)
    # SUBTRACT TAKES A VERTEX OUT OF THE GROUP when its weight reaches zero - Blender's own rule, and the
    # right result, since zero is what was asked for - and a vertex that was never in the group has nothing
    # to subtract from, so it stays out. This check was written for REPLACE and ADD, where a vertex missing
    # afterwards IS a failure, and it refused every subtract that zeroed anything - after the write, so the
    # group had changed under a refusal. Found 2026-09-13 by the tentacles showcase's tent weights ("1346 of
    # 3074 vertices are NOT in group 'seg1'"). Only a miss that neither case explains is refused now.
    zeroed, never_in = [], []
    if verts and not remove and missing and mode == "SUBTRACT":
        amount = dict(zip(verts, raw_weights)) if raw_weights is not None else {}
        flat = 1.0 if single is None else single
        for i in missing:
            before = weights_before.get(i)
            if before is None:
                never_in.append(i)
            elif before - amount.get(i, flat) <= 1e-6:
                zeroed.append(i)
        explained = set(zeroed) | set(never_in)
        missing = [i for i in missing if i not in explained]
    if verts and not remove and missing:
        raise MifOpError("%d of %d vertices are NOT in group '%s' after the write: %s. The group "
                         "exists and those vertices are outside it, which deforms nothing."
                         % (len(missing), len(verts), group.name, missing[:10]))

    # EXCLUSIVE: out of every other group, read from each vertex's own memberships (one walk, not groups x vertices),
    # then counted again from the data rather than trusted.
    others_cleared, left_in_others = None, None
    if exclusive and verts:
        by_group = {}
        for i in verts:
            for member in mesh.vertices[i].groups:
                if member.group != group.index:
                    by_group.setdefault(member.group, []).append(i)
        others_cleared = {}
        for gi in sorted(by_group):
            other = obj.vertex_groups[gi]
            other.remove(by_group[gi])
            others_cleared[other.name] = len(by_group[gi])
        left_in_others = sum(1 for i in verts for member in mesh.vertices[i].groups if member.group != group.index)

    total_in_group = 0
    for v in mesh.vertices:
        try:
            group.weight(v.index)
            total_in_group += 1
        except RuntimeError:
            pass
    values = [w for w in after.values() if w is not None]
    return {
        "ok": True,
        "object": obj.name,
        "group": group.name,
        "groupIndex": group.index,
        "created": created,
        "requestedName": str(group_name),
        "nameWasSuffixed": group.name != str(group_name),
        "mode": mode if not remove else "REMOVE",
        "byAxis": by_axis_report,
        "byRadius": by_radius_report,
        "verticesTouched": len(verts),
        "allVertices": all_vertices,
        "exclusive": exclusive,
        # {group: vertices taken out of it}; null unless exclusive. verticesInOtherGroups is the read-back: 0.
        "otherGroupsCleared": others_cleared,
        "verticesInOtherGroups": left_in_others,
        # THE NUMBER THAT DECIDES WHETHER ANYTHING MOVES. A group full of zero weights is a group
        # that exists and deforms nothing - and it is DETECTABLE, which this comment and the note
        # below both used to deny. list_vertex_groups counts `if g.weight > 0.0`, so such a group
        # comes back with weightedVertexCount 0 and influencesGeometry false while a working one
        # comes back positive and true.
        "verticesWithNonZeroWeight": nonzero,
        "verticesInGroup": total_in_group,
        # SUBTRACT's two quiet outcomes, counted rather than refused: taken out at zero, or never in.
        "removedAtZero": len(zeroed),
        "subtractedFromNonMembers": len(never_in),
        "weightRange": [min(values), max(values)] if values else None,
        "weightsBefore": weights_before if len(weights_before) <= 32 else None,
        # SAY WHERE TO LOOK, not that looking is pointless. This note fires exactly when the
        # caller's rig is broken, and the old wording - "reads back identically to a working one in
        # list_vertex_groups" - was false about the one op that finds it, at the only moment
        # somebody was going to check.
        "note": ("every weight written is 0, so this group exists and deforms NOTHING. It is "
                 "detectable: list_vertex_groups counts only weights above zero, so this group "
                 "comes back with weightedVertexCount 0 and influencesGeometry false, where a "
                 "working one comes back positive and true.")
        if (values and not nonzero and not remove) else
        (("the group name was taken, so Blender created '%s' instead of '%s' - the existing group "
          "was left alone. Anything looking this up by name needs the new one."
          % (group.name, group_name)) if group.name != str(group_name) else None),
    }


def _side_partner(name, pair):
    """The group name on the other side by the suffix pair ('_l', '_r'), or None for a name with neither suffix."""
    a, b = pair
    if name.endswith(a):
        return name[: -len(a)] + b
    if name.endswith(b):
        return name[: -len(b)] + a
    return None


def _group_weights(obj, group):
    """{vertex index: weight} for one vertex group, read off each vertex's memberships in one walk."""
    out = {}
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == group.index:
                out[v.index] = g.weight
    return out


_VG_CIL_KEYS = {"object", "name", "group", "groups", "mode", "vertices", "threshold", "keepSingle", "offset", "gain",
                "autoAssign", "autoRemove", "armature"}


def _clean_invert_levels(params):
    """set_vertex_weights mode CLEAN, INVERT or LEVELS (1.3 gap audit, 2026-10-19) - Blender's vertex_group_clean /
    _invert / _levels, written natively (as QUANTIZE is) so no active group or mode is needed:
      CLEAN   remove a vertex from a group where its weight is under threshold (default 0.01); keepSingle (default true)
              keeps a vertex whose every considered weight is under it in its strongest group, so none is left unweighted
      INVERT  w -> 1 - w; autoAssign (default true) adds the group's missing vertices at 1, autoRemove (default true)
              takes vertices that land on 0 out of the group (Blender's defaults)
      LEVELS  w -> clamp((w + offset) * gain, 0, 1) (Blender's formula; offset default 0, gain default 1)
    group, or groups [names], or groups "DEFORM" (every group named after a bone of `armature`, default the mesh's
    Armature modifier's object) or "ALL"; vertices narrows INVERT and LEVELS. Counts per group are returned."""
    reject_unknown(params, _VG_CIL_KEYS, "set_vertex_weights")
    tail = "NOTHING was changed."
    mode = str(take(params, "mode", kind=str)).upper()
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    require_editable(obj, "change the weights of")
    one = take(params, "group", "name", default=None, kind=str)
    many = params.get("groups")
    if (one is None) == (many is None):
        raise MifOpError("give group (one name) or groups (names, \"DEFORM\" or \"ALL\"). %s" % tail)
    if one is not None:
        names = [one]
    elif isinstance(many, str) and many.upper() in ("ALL", "DEFORM"):
        if many.upper() == "ALL":
            names = [g.name for g in obj.vertex_groups]
        else:
            arm_name = take(params, "armature", default=None, kind=str)
            arm = get_object(arm_name) if arm_name else next(
                (m.object for m in obj.modifiers if m.type == "ARMATURE" and m.object is not None), None)
            if arm is None or arm.type != "ARMATURE":
                raise MifOpError("groups DEFORM needs an armature (the mesh has no Armature modifier; pass armature). %s"
                                 % tail)
            bones = {b.name for b in arm.data.bones if b.use_deform}
            names = [g.name for g in obj.vertex_groups if g.name in bones]
    elif isinstance(many, list) and many and all(isinstance(n, str) for n in many):
        names = list(dict.fromkeys(many))
    else:
        raise MifOpError("groups is a list of names, \"DEFORM\" or \"ALL\". %s" % tail)
    missing = [n for n in names if obj.vertex_groups.get(n) is None]
    if missing:
        raise MifOpError("'%s' has no vertex group(s) %s. %s" % (obj.name, missing[:20], tail))
    if not names:
        raise MifOpError("no vertex group matched. %s" % tail)
    allowed = {"CLEAN": {"threshold", "keepSingle"}, "INVERT": {"autoAssign", "autoRemove", "vertices"},
               "LEVELS": {"offset", "gain", "vertices"}}[mode]
    stray = sorted(k for k in ("threshold", "keepSingle", "offset", "gain", "autoAssign", "autoRemove", "vertices")
                   if params.get(k) is not None and k not in allowed)
    if stray:
        raise MifOpError("%s are not mode %s's. %s" % (stray, mode, tail))
    nverts = len(obj.data.vertices)
    region = None
    if params.get("vertices") is not None:
        raw = params["vertices"]
        if not isinstance(raw, list) or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < nverts
                                            for i in raw):
            raise MifOpError("vertices must be vertex indices of '%s'. %s" % (obj.name, tail))
        region = set(raw)
    groups = [obj.vertex_groups[n] for n in names]
    weights = {g.name: _group_weights(obj, g) for g in groups}
    rows = []
    if mode == "CLEAN":
        thr = take_float(params, "threshold", default=0.01)
        if not 0.0 <= thr <= 1.0:
            raise MifOpError("threshold is 0-1, got %g. %s" % (thr, tail))
        keep = take_bool(params, "keepSingle", default=True)
        drop = {g.name: [i for i, w in weights[g.name].items() if w < thr] for g in groups}
        kept_single = 0
        if keep:
            # a vertex every one of whose considered weights is under the threshold keeps its strongest
            for v in range(nverts):
                mine = [(weights[n][v], n) for n in names if v in weights[n]]
                if mine and all(w < thr for w, _n in mine):
                    best = max(mine)[1]
                    if v in drop[best]:
                        drop[best].remove(v)
                        kept_single += 1
        for g in groups:
            if drop[g.name]:
                g.remove(drop[g.name])
            rows.append({"group": g.name, "removed": len(drop[g.name]), "remaining": len(weights[g.name]) -
                         len(drop[g.name])})
        return {"object": obj.name, "mode": mode, "threshold": thr, "keepSingle": keep, "keptSingle": kept_single,
                "groups": rows, "removed": sum(r["removed"] for r in rows)}
    if mode == "INVERT":
        assign = take_bool(params, "autoAssign", default=True)
        autorm = take_bool(params, "autoRemove", default=True)
        for g in groups:
            cur = weights[g.name]
            pool = (region if region is not None else set(range(nverts)) if assign else set(cur))
            added = removed = 0
            for v in sorted(pool):
                if v not in cur and not assign:
                    continue
                w = 1.0 - cur.get(v, 0.0)
                if autorm and w <= 0.0:
                    if v in cur:
                        g.remove([v])
                        removed += 1
                    continue
                g.add([v], w, "REPLACE")
                added += v not in cur
            rows.append({"group": g.name, "inverted": len(pool), "added": added, "removed": removed})
        return {"object": obj.name, "mode": mode, "autoAssign": assign, "autoRemove": autorm, "groups": rows}
    offset = take_float(params, "offset", default=0.0)
    gain = take_float(params, "gain", default=1.0)
    if not -1.0 <= offset <= 1.0 or not 0.0 <= gain <= 10.0:
        raise MifOpError("offset is -1 to 1 and gain 0 to 10 (Blender's ranges), got %g and %g. %s" % (offset, gain, tail))
    for g in groups:
        cur = weights[g.name]
        changed = 0
        for v, w in cur.items():
            if region is not None and v not in region:
                continue
            nw = min(1.0, max(0.0, (w + offset) * gain))
            if abs(nw - w) > 1e-9:
                g.add([v], nw, "REPLACE")
                changed += 1
        after = _group_weights(obj, g)
        rows.append({"group": g.name, "changed": changed,
                     "range": [round(min(after.values()), 6), round(max(after.values()), 6)] if after else None})
    return {"object": obj.name, "mode": mode, "offset": offset, "gain": gain, "groups": rows}


def _mirror_or_smooth(params):
    """set_vertex_weights mode MIRROR or SMOOTH - see that op's docstring. Everything is computed, then written once."""
    reject_unknown(params, _VG_MIRROR_SMOOTH_KEYS, "set_vertex_weights")
    mode = str(take(params, "mode", kind=str)).upper()
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    name = take(params, "group", "name", required=True, kind=str)
    group = obj.vertex_groups.get(name)
    if group is None:
        raise MifOpError("'%s' has no vertex group '%s' to %s. NOTHING was changed." % (obj.name, name, mode.lower()))
    mesh = obj.data
    before = _group_weights(obj, group)
    if mode == "QUANTIZE":
        # BLENDER'S OWN ROUNDING (object.vertex_group_quantize: floor(w * steps + 0.5) / steps), written natively so no
        # active group or mode is needed - the operator measured identically on 3.6.23, 4.4.0 and 5.2.1 (0.49 -> 0,
        # 0.51 -> 1 at steps 1, a vertex rounded to 0 kept in the group).
        for key in ("mirrorAxis", "mirrorPair", "tolerance", "factor", "repeat", "expand"):
            if params.get(key) is not None:
                raise MifOpError("%s is for mode %s. NOTHING was changed."
                                 % (key, "MIRROR" if key.startswith(("mirror", "tol")) else "SMOOTH"))
        steps = take_int(params, "steps", default=4)
        if not 1 <= steps <= 1000:
            raise MifOpError("steps must be 1-1000 (Blender's range), got %d. NOTHING was changed." % steps)
        raw = params.get("vertices")
        if raw is not None:
            if not isinstance(raw, (list, tuple)) or any(isinstance(i, bool) or not isinstance(i, int)
                                                        or not 0 <= i < len(mesh.vertices) for i in raw):
                raise MifOpError("vertices must be vertex indices of '%s'. NOTHING was changed." % obj.name)
            region = [i for i in raw if i in before]
        else:
            region = sorted(before)
        if not region:
            raise MifOpError("group '%s' holds none of those vertices, so there is nothing to quantize. NOTHING was "
                             "changed." % name)
        changed = 0
        for i in region:
            w = math.floor(before[i] * steps + 0.5) / steps
            if abs(w - before[i]) > 1e-9:
                group.add([i], w, "REPLACE")
                changed += 1
        after = _group_weights(obj, group)
        levels = sorted({round(after[i], 6) for i in region})
        return {"object": obj.name, "mode": "QUANTIZE", "group": name, "steps": steps, "verticesQuantized": len(region),
                "verticesChanged": changed, "levels": levels[:20],
                "weightRangeBefore": [round(min(before[i] for i in region), 6), round(max(before[i] for i in region), 6)],
                "verticesAtZero": sum(1 for i in region if after[i] == 0.0)}
    if params.get("steps") is not None:
        raise MifOpError("steps is for mode QUANTIZE. NOTHING was changed.")
    if mode == "MIRROR":
        for key in ("factor", "repeat", "expand"):
            if params.get(key) is not None:
                raise MifOpError("%s is for mode SMOOTH. NOTHING was changed." % key)
        axis = str(take(params, "mirrorAxis", default="X", kind=str)).upper()
        if axis not in ("X", "Y", "Z"):
            raise MifOpError("mirrorAxis must be X, Y or Z, got %r. NOTHING was changed." % axis)
        pair = params.get("mirrorPair", ["_l", "_r"])
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(not isinstance(s, str) or not s for s in pair) \
                or pair[0] == pair[1]:
            raise MifOpError("mirrorPair is two different suffixes, like ['_l', '_r']; got %r. NOTHING was changed."
                             % (pair,))
        tol = take_float(params, "tolerance", default=1e-4)
        if not tol > 0.0:
            raise MifOpError("tolerance must be above 0. NOTHING was changed.")
        if not before:
            raise MifOpError("group '%s' holds no vertex, so there is nothing to mirror. NOTHING was changed." % name)
        target_name = _side_partner(name, pair) or name
        k = "XYZ".index(axis)
        from mathutils.kdtree import KDTree   # noqa: PLC0415 - the offline suites stub mathutils
        tree = KDTree(len(mesh.vertices))
        for v in mesh.vertices:
            tree.insert(v.co, v.index)
        tree.balance()
        plan, unpaired = {}, []
        in_place = target_name == name
        for i, w in sorted(before.items()):
            p = mesh.vertices[i].co.copy()
            if in_place and p[k] <= tol:
                continue   # IN PLACE the + half writes the - half; both halves at once would swap the sides

            p[k] = -p[k]
            _co, j, d = tree.find(p)
            if j is None or d > tol:
                unpaired.append(i)
                continue
            plan[j] = w
        if not plan:
            raise MifOpError("no vertex of '%s' has a partner across %s within %g - the mesh is not symmetric there. "
                             "NOTHING was changed." % (name, axis, tol))
        target = obj.vertex_groups.get(target_name)
        created = target is None
        if created:
            target = obj.vertex_groups.new(name=target_name)
        for j, w in plan.items():
            target.add([j], w, "REPLACE")
        after = _group_weights(obj, target)
        vals = [after[j] for j in plan]
        return {"object": obj.name, "mode": "MIRROR", "group": name, "target": target.name, "targetCreated": created,
                "axis": axis, "verticesWritten": len(plan), "verticesWithoutPartner": len(unpaired),
                "unpairedVertices": unpaired[:50], "weightRange": [round(min(vals), 6), round(max(vals), 6)]}
    for key in ("mirrorAxis", "mirrorPair", "tolerance"):
        if params.get(key) is not None:
            raise MifOpError("%s is for mode MIRROR. NOTHING was changed." % key)
    factor = take_float(params, "factor", default=0.5)
    repeat = take_int(params, "repeat", default=1)
    expand = take_bool(params, "expand", default=False)
    if not 0.0 < factor <= 1.0:
        raise MifOpError("factor must be above 0 and at most 1, got %r. NOTHING was changed." % factor)
    if not 1 <= repeat <= 100:
        raise MifOpError("repeat must be 1-100, got %d. NOTHING was changed." % repeat)
    raw = params.get("vertices")
    if raw is not None:
        if not isinstance(raw, (list, tuple)) or any(isinstance(i, bool) or not isinstance(i, int)
                                                    or not 0 <= i < len(mesh.vertices) for i in raw):
            raise MifOpError("vertices must be vertex indices of '%s'. NOTHING was changed." % obj.name)
        region = set(raw)
    else:
        region = set(before)
    if not region:
        raise MifOpError("group '%s' holds no vertex, so there is nothing to smooth. NOTHING was changed." % name)
    nbrs = {i: [] for i in range(len(mesh.vertices))}
    for e in mesh.edges:
        a, b = e.vertices
        nbrs[a].append(b)
        nbrs[b].append(a)
    if expand:
        region |= {n for i in list(region) for n in nbrs[i]}
    cur = dict(before)
    for _ in range(repeat):
        nxt = dict(cur)
        for i in region:
            if not nbrs[i]:
                continue
            avg = sum(cur.get(n, 0.0) for n in nbrs[i]) / len(nbrs[i])
            nxt[i] = (1.0 - factor) * cur.get(i, 0.0) + factor * avg
        cur = nxt
    changed = [i for i in region if abs(cur.get(i, 0.0) - before.get(i, 0.0)) > 1e-9]
    for i in changed:
        if i in before or cur[i] > 0.0:
            group.add([i], min(1.0, max(0.0, cur[i])), "REPLACE")
    after = _group_weights(obj, group)
    bvals = [before[i] for i in region if i in before]
    avals = [after[i] for i in region if i in after]
    return {"object": obj.name, "mode": "SMOOTH", "group": name, "factor": factor, "repeat": repeat, "expand": expand,
            "verticesChanged": len(changed), "weightRangeBefore": [round(min(bvals), 6), round(max(bvals), 6)]
            if bvals else None, "weightRangeAfter": [round(min(avals), 6), round(max(avals), 6)] if avals else None}


_MIRROR_KEY_KEYS = ("object", "name", "key", "keys", "prefix", "axis", "newName", "pairs", "tolerance", "overwrite")
_SIDE_PAIRS = (("_l", "_r"), ("_L", "_R"), (".l", ".r"), (".L", ".R"), ("Left", "Right"), ("left", "right"))


def _swap_side(name, pairs):
    """The key name with its side swapped at the END (smile_L -> smile_R), or None when it carries no side."""
    for a, b in pairs:
        if name.endswith(a):
            return name[: -len(a)] + b
        if name.endswith(b):
            return name[: -len(b)] + a
    return None


def op_mirror_shape_key(params):
    """Mirror shape keys across an axis onto new keys with the other side's name.

    One of key, keys [...] or prefix (every key whose name starts with it; the basis never). Each key's offsets from the
    basis are carried to the mirrored vertex - the nearest basis vertex to the mirrored position within tolerance (1e-4
    mesh units) - with the axis component negated. The new key is named by swapping the side at the end of the name
    (pairs, default _l/_r, _L/_R, .l/.r, .L/.R, Left/Right) or newName (one key only); an existing key of that name is
    refused unless overwrite is true. Reply: the new keys, the vertex pairs matched, the unmatched vertices with
    indices, and the matched share - under 100 percent the mesh is not symmetric there and the reply says so. Drivers
    are not carried.
    """
    reject_unknown(params, _MIRROR_KEY_KEYS, "mirror_shape_key")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    require_editable(obj, "mirror shape keys on")
    sk = obj.data.shape_keys
    if sk is None or len(sk.key_blocks) < 2:
        raise MifOpError("'%s' has no shape key beside its basis to mirror. NOTHING was created." % obj.name)
    basis = sk.reference_key
    blocks = [kb for kb in sk.key_blocks if kb.name != basis.name]
    given = [k for k in ("key", "keys", "prefix") if params.get(k) is not None]
    if len(given) != 1:
        raise MifOpError("pass one of key, keys or prefix (got %s). NOTHING was created." % (", ".join(given) or "none"))
    if given[0] == "key":
        names = [str(params["key"])]
    elif given[0] == "keys":
        raw = params["keys"]
        if not isinstance(raw, (list, tuple)) or not raw or any(not isinstance(n, str) for n in raw):
            raise MifOpError("keys is a list of shape key names. NOTHING was created.")
        names = list(dict.fromkeys(raw))
    else:
        names = [kb.name for kb in blocks if kb.name.startswith(str(params["prefix"]))]
        if not names:
            raise MifOpError("no shape key of '%s' starts with %r. NOTHING was created." % (obj.name, params["prefix"]))
    have = {kb.name for kb in blocks}
    missing = [n for n in names if n not in have]
    if missing:
        raise MifOpError("'%s' has no shape key %s (the basis cannot be mirrored). NOTHING was created."
                         % (obj.name, ", ".join(repr(n) for n in missing)))
    axis = str(take(params, "axis", default="X", kind=str)).upper()
    if axis not in ("X", "Y", "Z"):
        raise MifOpError("axis must be X, Y or Z, got %r. NOTHING was created." % axis)
    pairs = params.get("pairs")
    if pairs is None:
        pairs = _SIDE_PAIRS
    elif not isinstance(pairs, (list, tuple)) or any(not isinstance(p, (list, tuple)) or len(p) != 2
                                                     or not all(isinstance(s, str) and s for s in p) for p in pairs):
        raise MifOpError("pairs is [[left, right], ...] of non-empty suffixes. NOTHING was created.")
    new_name = take(params, "newName", default=None, kind=str)
    if new_name is not None and len(names) != 1:
        raise MifOpError("newName names one key; %d were asked. Leave it out to swap each name's side. NOTHING was "
                         "created." % len(names))
    tol = take_float(params, "tolerance", default=1e-4)
    if not tol > 0.0:
        raise MifOpError("tolerance must be above 0. NOTHING was created.")
    overwrite = take_bool(params, "overwrite", default=False)
    plan = []
    for n in names:
        target = new_name or _swap_side(n, pairs)
        if not target:
            raise MifOpError("%r carries no side suffix (%s) to swap - pass newName. NOTHING was created."
                             % (n, ", ".join("%s/%s" % tuple(p) for p in pairs)))
        if target == n:
            raise MifOpError("%r would mirror onto itself. NOTHING was created." % n)
        if target in have and not overwrite:
            raise MifOpError("'%s' already has a key %r - pass overwrite:true to replace it. NOTHING was created."
                             % (obj.name, target))
        plan.append((n, target))
    k = "XYZ".index(axis)
    from mathutils.kdtree import KDTree   # noqa: PLC0415 - the offline suites stub mathutils
    base_co = [v.co.copy() for v in basis.data]
    tree = KDTree(len(base_co))
    for i, co in enumerate(base_co):
        tree.insert(co, i)
    tree.balance()
    partner = {}
    for i, co in enumerate(base_co):
        p = co.copy()
        p[k] = -p[k]
        _c, j, d = tree.find(p)
        if j is not None and d <= tol:
            partner[j] = i          # vertex j takes vertex i's offset, mirrored
    unmatched = sorted(set(range(len(base_co))) - set(partner))
    made = []
    for src_name, target in plan:
        src = sk.key_blocks[src_name]
        offs = [src.data[i].co - base_co[i] for i in range(len(base_co))]
        tgt = sk.key_blocks.get(target)
        created = tgt is None
        if created:
            tgt = obj.shape_key_add(name=target, from_mix=False)
            # 0, AS add_shape_key PINS IT: Blender 5.x starts a new key at 1.0 and 3.6-4.4 at 0.0.
            tgt.value = 0.0
        tgt.relative_key = basis
        for j in range(len(base_co)):
            i = partner.get(j)
            off = offs[i].copy() if i is not None else offs[j] * 0.0
            if i is not None:
                off[k] = -off[k]
            tgt.data[j].co = base_co[j] + off
        made.append({"from": src_name, "key": tgt.name, "created": created, "value": round(tgt.value, 6),
                     "verticesMoved": sum(1 for j in range(len(base_co))
                                          if (tgt.data[j].co - base_co[j]).length > 1e-9)})
    obj.data.update()
    share = round(len(partner) / float(len(base_co)), 6) if base_co else 0.0
    out = {"object": obj.name, "axis": axis, "keys": made, "pairsMatched": len(partner),
           "unmatchedVertices": len(unmatched), "unmatchedIndices": unmatched[:100], "matchedShare": share}
    if share < 1.0:
        out["symmetryWarning"] = ("%d of %d vertices have no mirror vertex within %g, so their offsets were not carried "
                                  "- the mesh is not symmetric there." % (len(unmatched), len(base_co), tol))
    return out


def op_add_shape_key(params):
    """Add a shape key - the other thing five ops could consume and nothing could produce.

    set_shape_key drives a key's value and list_shape_keys reports them, and nothing anywhere could
    CREATE one, so a mesh without shape keys could never be given any.

    THE FIRST KEY IS THE BASIS AND IT IS NOT A SHAPE. Blender names the first key added on a mesh
    'Basis' and every later key is stored RELATIVE to it - verified on 3.6, 4.4 and 5.0, where the
    second key came back with relative_key 'Basis'. A caller adding one key to a bare mesh gets the
    rest position and nothing that can move, which is why this reports basisCreated separately
    rather than counting it as the key that was asked for.

    fromMix TAKES THE CURRENT EVALUATED MIX rather than the rest shape, which is how a corrective
    key is made. Off by default: a key silently capturing whatever the sliders happened to be set
    to is a surprise, not a convenience.

    params:
      object (str)         required, must be a MESH
      name / key (str)     the key's name. Default 'Key'.
      fromMix (bool)       capture the current mix instead of the rest shape. Default false.
      value (float)        the key's influence, 0-1. Default 0.0 on EVERY build - Blender 5.0 and later would start a
                           new key at 1.0, and 3.6-4.4 at 0.0; see the pin below and the response's valueNote.
      sliderMin / sliderMax (float)
      fromObject (str)     Join as Shapes: the key takes ANOTHER mesh's evaluated vertex positions - a regenerated
                           garment or a sculpted copy with the same vertex count - carried from its world space into
                           this object's. A mesh with no keys gets a 'Basis' first, so the named key is the shape.
                           Refused on a different vertex count, and with fromMix (2026-10-04, the garment work).
    """
    reject_unknown(params, _SHAPEKEY_KEYS, "add_shape_key")
    src = take(params, "object", required=True, kind=str)
    name = str(take(params, "name", "key", default="Key", kind=str))
    if len(name) > _ID_NAME_LIMIT:
        raise MifOpError("the key name is %d characters and Blender truncates at %d. NOTHING was "
                         "created." % (len(name), _ID_NAME_LIMIT))
    from_mix = take_bool(params, "fromMix", default=False)
    value = take_float(params, "value", default=None)
    smin = take_float(params, "sliderMin", default=None)
    smax = take_float(params, "sliderMax", default=None)
    if smin is not None and smax is not None and smax < smin:
        raise MifOpError("sliderMax %g is below sliderMin %g. NOTHING was created." % (smax, smin))

    obj = get_object(src, want_mesh=True)
    from_name = take(params, "fromObject", default=None, kind=str)
    shape_co = None
    if from_name is not None:
        if from_mix:
            raise MifOpError("pass fromObject OR fromMix - one says where the shape comes from. NOTHING was created.")
        other = get_object(from_name, want_mesh=True)
        if other is obj:
            raise MifOpError("fromObject is '%s' itself, which has no other shape to give. NOTHING was created."
                             % obj.name)
        ev = other.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = ev.to_mesh()
        try:
            if len(me.vertices) != len(obj.data.vertices):
                raise MifOpError("'%s' has %d vertices (evaluated) and '%s' has %d - Join as Shapes needs the same "
                                 "topology, vertex for vertex. NOTHING was created."
                                 % (other.name, len(me.vertices), obj.name, len(obj.data.vertices)))
            to_local = obj.matrix_world.inverted() @ other.matrix_world
            shape_co = [to_local @ v.co for v in me.vertices]
        finally:
            ev.to_mesh_clear()
    existing = obj.data.shape_keys
    had = list(existing.key_blocks.keys()) if existing else []
    vg, rel = _key_links(obj, params, name, had, "NOTHING was created.")
    if name in had:
        raise MifOpError("'%s' already has a shape key named '%s'. Blender would create '%s.001' "
                         "beside it, which is rarely what somebody adding a named key wants - pick "
                         "another name or drive the existing one with set_shape_key. NOTHING was "
                         "created." % (obj.name, name, name))

    basis_created = not had
    if shape_co is not None and basis_created:
        if name == "Basis":
            raise MifOpError("the first key on a mesh is its 'Basis', so fromObject's key needs another name. NOTHING "
                             "was created.")
        obj.shape_key_add(name="Basis", from_mix=False)
    key = obj.shape_key_add(name=name, from_mix=from_mix)
    if key is None:
        raise MifOpError("shape_key_add returned None for '%s' - Blender reports failure that way "
                         "rather than raising. NOTHING was created." % name)
    # A NEW KEY STARTS AT 0.0 ON EVERY BUILD, unless value says otherwise. Blender changed its own default at 5.0:
    # shape_key_add gives a new key value 0.0 on 3.6.23, 4.2.17 and 4.4.0 and 1.0 on 5.0.1 and 5.2.1 (measured
    # 2026-09-21), so the same two calls - add a key, move its points with set_point_positions - left the mesh
    # unchanged on 4.4 ("at value 0, so its new positions show only once set_shape_key gives it influence") and
    # deformed it at full strength on 5.0. Pinned to the value every build before 5.0 gives, the way uv_unwrap
    # pins fill_holes: the op behaves one way wherever it runs, and the response says what the key starts at.
    max_offset = None
    if shape_co is not None:
        basis = key.relative_key
        for i, co in enumerate(shape_co):
            key.data[i].co = co
        max_offset = max(((key.data[i].co - basis.data[i].co).length for i in range(len(shape_co))), default=0.0)
    native_value = float(key.value)
    # THE BASIS TOO: its value moves nothing (every relative key is measured from it), but the response and every
    # listing read it, and 1.0 on one build and 0.0 on another is a difference with no meaning to report.
    value_pinned = value is None
    if value_pinned:
        key.value = 0.0
    if value is not None:
        key.value = value
    if smin is not None:
        key.slider_min = smin
    if smax is not None:
        key.slider_max = smax
    if vg:
        key.vertex_group = vg
    if rel is not None:
        key.relative_key = obj.data.shape_keys.key_blocks[rel]

    blocks = obj.data.shape_keys.key_blocks
    return {
        "ok": True,
        "object": obj.name,
        "key": key.name,
        "keyCount": len(blocks),
        "keys": list(blocks.keys()),
        # THE FIRST KEY ON A BARE MESH IS THE REST POSITION. Reported so a caller who asked for one
        # key and got 'Basis' can see why nothing moves.
        "basisCreated": basis_created,
        "relativeTo": key.relative_key.name if getattr(key, "relative_key", None) else None,
        "vertexGroup": key.vertex_group or None,
        "value": round(float(key.value), 6),
        "valueNote": ("no value was given, so this key starts at 0.0 - what every build before 5.0 gives a new key. "
                      "Blender %s would have started it at %g, applying it at full strength the moment its points "
                      "move. Pass value to choose; set_shape_key changes it later."
                      % (bpy.app.version_string, native_value))
                     if value_pinned and native_value != 0.0 and not basis_created else None,
        "sliderRange": [round(float(key.slider_min), 6), round(float(key.slider_max), 6)],
        "fromMix": from_mix,
        "fromObject": from_name,
        "maxOffsetFromBasis": round(max_offset, 6) if max_offset is not None else None,
        "note": ("this was the FIRST key on the mesh, so Blender made it the BASIS - the rest "
                 "position. It holds the shape the mesh already has and moves nothing. Add a "
                 "second key and edit its data to get something that deforms.")
        if basis_created else None,
    }


_POINTS_KEYS = {"object", "name", "points", "shapeKey", "key", "tilt", "handleType", "radius", "handles"}
_READ_POINTS_CAP = 10000


def _read_curve_points(obj, params):
    """set_point_positions with nothing to write on a CURVE: every control point read back, local and world - the read
    form a cable or road curve is checked or handed off with (second 1.3 audit, 2026-10-25)."""
    cu = obj.data
    start, limit = 0, _READ_POINTS_CAP
    mw = obj.matrix_world
    pts = _curve_point_list(cu)
    rows = []
    for i, (si, pt, bez) in enumerate(pts):
        if i < start or i >= start + limit:
            continue
        co = tuple(pt.co)[:3]
        row = {"index": i, "spline": si, "co": [round(float(x), 6) for x in co],
               "world": [round(float(x), 6) for x in mw @ Vector(co)],
               "tilt": round(float(pt.tilt), 6), "radius": round(float(pt.radius), 6)}
        if bez:
            row.update({"handleLeft": [round(float(x), 6) for x in pt.handle_left],
                        "handleRight": [round(float(x), 6) for x in pt.handle_right],
                        "handleLeftWorld": [round(float(x), 6) for x in mw @ pt.handle_left],
                        "handleRightWorld": [round(float(x), 6) for x in mw @ pt.handle_right],
                        "handleLeftType": pt.handle_left_type, "handleRightType": pt.handle_right_type})
        else:
            row["weight"] = round(float(pt.co[3]), 6)
        rows.append(row)
    return {"ok": True, "object": obj.name, "read": True, "pointCount": len(pts),
            "splines": [{"index": si, "type": sp.type, "points": len(sp.bezier_points) if sp.type == "BEZIER"
                         else len(sp.points), "cyclic": bool(sp.use_cyclic_u), "resolution": int(sp.resolution_u)}
                        for si, sp in enumerate(cu.splines)],
            "curvePoints": rows, "truncated": len(pts) > limit,
            "units": "co and handles in the curve's local space, world after its matrix_world; tilt in radians; weight "
                     "is a poly or NURBS point's w"}


def _curve_point_list(cu):
    """[(spline index, point, is_bezier)] over every spline of a curve, in order - the index set_point_positions uses."""
    out = []
    for si, sp in enumerate(cu.splines):
        if sp.type == "BEZIER":
            out.extend((si, bp, True) for bp in sp.bezier_points)
        else:
            out.extend((si, pt, False) for pt in sp.points)
    return out


def _curve_point_rows(pts, limit=200):
    rows = []
    for i, (si, pt, bez) in enumerate(pts[:limit]):
        row = {"index": i, "spline": si, "co": [round(float(x), 6) for x in tuple(pt.co)[:3]],
               "tilt": round(float(pt.tilt), 6), "radius": round(float(pt.radius), 6)}
        if bez:
            row.update({"handleLeft": [round(float(x), 6) for x in pt.handle_left],
                        "handleRight": [round(float(x), 6) for x in pt.handle_right],
                        "handleLeftType": pt.handle_left_type, "handleRightType": pt.handle_right_type})
        rows.append(row)
    return rows


def _set_curve_points(obj, params):
    """set_point_positions on a CURVE: positions, tilt and handle types of its control points - see the op's docstring.
    Every row is checked before any point is written."""
    cu = obj.data
    if getattr(cu, "shape_keys", None) is not None or params.get("shapeKey") is not None or params.get("key") is not None:
        raise MifOpError("'%s' is a curve with shape keys or a shapeKey was named - curve shape keys are not written "
                         "here. NOTHING was changed." % obj.name)
    pts = _curve_point_list(cu)
    n = len(pts)

    def idx(i, what):
        i = finite_int(i, what)
        if not 0 <= i < n:
            raise MifOpError("%s %d is out of range - '%s' has %d control points. NOTHING was changed."
                             % (what, i, obj.name, n))
        return i
    raw = params.get("points")
    tilt_raw = params.get("tilt")
    ht = params.get("handleType")
    rad_raw = params.get("radius")
    hd_raw = params.get("handles")
    if raw is None and tilt_raw is None and ht is None and rad_raw is None and hd_raw is None:
        raise MifOpError("pass points ([[index, x, y, z], ...]), tilt ([[index, radians], ...]), radius ([[index, r], "
                         "...]), handles or handleType. NOTHING was changed.")
    moves, tilts, radii, handles = {}, {}, {}, {}
    # HANDLES WRITTEN WHERE THEY ARE GIVEN (second 1.3 audit, 2026-10-25): handles [[index, [lx, ly, lz], [rx, ry, rz]],
    # ...] in the curve's local space, each point's two handles made FREE first - an AUTO or ALIGNED handle is moved
    # again by Blender the moment its partner or point changes. What an Unreal spline's arrive and leave tangents need:
    # left = point - arrive / 3, right = point + leave / 3.
    if hd_raw is not None:
        if not isinstance(hd_raw, list) or not hd_raw:
            raise MifOpError("handles is [[index, [lx, ly, lz], [rx, ry, rz]], ...]. NOTHING was changed.")
        if ht is not None:
            raise MifOpError("handles makes the points' handles FREE; pass it without handleType. NOTHING was changed.")
        for row in hd_raw:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise MifOpError("every entry in handles is [index, [lx, ly, lz], [rx, ry, rz]]; got %r. NOTHING was "
                                 "changed." % (row,))
            i = idx(row[0], "handles index")
            if not pts[i][2]:
                raise MifOpError("point %d is on a %s spline, which has no handles. NOTHING was changed."
                                 % (i, cu.splines[pts[i][0]].type))
            handles[i] = (tuple(finite_floats(list(row[1])[:3], "handles left")),
                          tuple(finite_floats(list(row[2])[:3], "handles right")))
    if rad_raw is not None:
        if not isinstance(rad_raw, list) or not rad_raw:
            raise MifOpError("radius is [[index, r], ...]. NOTHING was changed.")
        for row in rad_raw:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise MifOpError("every entry in radius is [index, r]; got %r. NOTHING was changed." % (row,))
            r = finite_floats([row[1]], "radius")[0]
            if r < 0:
                raise MifOpError("a radius is 0 or more, got %g. NOTHING was changed." % r)
            radii[idx(row[0], "radius index")] = r
    if raw is not None:
        if not isinstance(raw, list) or not raw:
            raise MifOpError("points is [[index, x, y, z], ...] in the curve's local space. NOTHING was changed.")
        for row in raw:
            if not isinstance(row, (list, tuple)) or len(row) != 4:
                raise MifOpError("every entry in points is [index, x, y, z]; got %r. NOTHING was changed." % (row,))
            i = idx(row[0], "point index")
            if i in moves:
                raise MifOpError("point %d is given twice. NOTHING was changed." % i)
            moves[i] = tuple(finite_floats(row[1:], "points"))
    if tilt_raw is not None:
        if not isinstance(tilt_raw, list) or not tilt_raw:
            raise MifOpError("tilt is [[index, radians], ...]. NOTHING was changed.")
        for row in tilt_raw:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise MifOpError("every entry in tilt is [index, radians]; got %r. NOTHING was changed." % (row,))
            tilts[idx(row[0], "tilt index")] = finite_floats([row[1]], "tilt")[0]
    handle = None
    if ht is not None:
        if not isinstance(ht, dict) or "type" not in ht or set(ht) - {"type", "indices", "side"}:
            raise MifOpError("handleType is {type, indices (default every Bezier point), side LEFT | RIGHT | BOTH}; got "
                             "%r. NOTHING was changed." % (ht,))
        valid = [e.identifier for e in bpy.types.BezierSplinePoint.bl_rna.properties["handle_left_type"].enum_items]
        kind = str(ht["type"]).upper()
        if kind not in valid:
            raise MifOpError("handleType.type %r is not a handle type this Blender has (%s). NOTHING was changed."
                             % (ht["type"], ", ".join(valid)))
        side = str(ht.get("side", "BOTH")).upper()
        if side not in ("LEFT", "RIGHT", "BOTH"):
            raise MifOpError("handleType.side is LEFT, RIGHT or BOTH. NOTHING was changed.")
        if ht.get("indices") is None:
            which = [i for i, (_s, _p, bez) in enumerate(pts) if bez]
            if not which:
                raise MifOpError("'%s' has no Bezier point, so it has no handles. NOTHING was changed." % obj.name)
        else:
            if not isinstance(ht["indices"], list) or not ht["indices"]:
                raise MifOpError("handleType.indices is a non-empty list of point indices. NOTHING was changed.")
            which = [idx(i, "handleType index") for i in ht["indices"]]
            not_bez = [i for i in which if not pts[i][2]]
            if not_bez:
                raise MifOpError("points %s are on a %s spline, which has no handles. NOTHING was changed."
                                 % (not_bez[:10], cu.splines[pts[not_bez[0]][0]].type))
        handle = (kind, side, which)
    before = _evaluated_counts(obj)
    for i, xyz in moves.items():
        _si, pt, bez = pts[i]
        if bez:
            d = (xyz[0] - pt.co[0], xyz[1] - pt.co[1], xyz[2] - pt.co[2])
            pt.handle_left = tuple(pt.handle_left[k] + d[k] for k in range(3))
            pt.handle_right = tuple(pt.handle_right[k] + d[k] for k in range(3))
            pt.co = xyz
        else:
            pt.co = (xyz[0], xyz[1], xyz[2], pt.co[3])
    for i, (hl, hr) in handles.items():
        bp = pts[i][1]
        bp.handle_left_type = bp.handle_right_type = "FREE"
        bp.handle_left, bp.handle_right = hl, hr
    for i, t in tilts.items():
        pts[i][1].tilt = t
    for i, r in radii.items():
        pts[i][1].radius = r
    if handle is not None:
        kind, side, which = handle
        for i in which:
            bp = pts[i][1]
            if side in ("LEFT", "BOTH"):
                bp.handle_left_type = kind
            if side in ("RIGHT", "BOTH"):
                bp.handle_right_type = kind
    cu.update_tag()
    bpy.context.view_layer.update()
    worst = max([max(abs(pts[i][1].co[k] - xyz[k]) for k in range(3)) for i, xyz in moves.items()]
                + [max(abs(pts[i][1].handle_left[k] - h[0][k]) for k in range(3)) for i, h in handles.items()]
                + [max(abs(pts[i][1].handle_right[k] - h[1][k]) for k in range(3)) for i, h in handles.items()] or [0.0])
    after = _evaluated_counts(obj)
    return {"ok": True, "object": obj.name, "target": "curve control points", "pointCount": n,
            "written": len(moves), "tiltsWritten": len(tilts), "radiiWritten": len(radii), "handlesWritten": len(handles),
            "handlesSet": len(handle[2]) if handle else 0, "handleType": handle[0] if handle else None,
            "maxReadBackError": round(float(worst), 9), "readBackMatches": worst <= 1e-4,
            "curvePoints": _curve_point_rows(pts), "curvePointsTruncated": n > 200,
            "evaluatedChanged": [obj.name] if after.get("shape") != before.get("shape") else [],
            "note": ("a Bezier point's handles move with it; an AUTO or VECTOR handle is then placed by Blender, which "
                     "is why handleLeft / handleRight are read back rather than assumed.") if moves else None}


def _deformed_by(obj):
    """The objects whose evaluated shape obj's points decide: a mesh is moved by its own keys, and a
    lattice moves every object with a LATTICE modifier aimed at it or lattice-parented to it."""
    if obj.type != "LATTICE":
        return [obj]
    return [o for o in bpy.data.objects
            if (o.parent == obj and o.parent_type == "LATTICE")
            or any(m.type == "LATTICE" and getattr(m, "object", None) == obj for m in o.modifiers)]


def op_set_point_positions(params):
    """Move a lattice's points or a shape key's vertices, by index - the edit that makes either deform.

    BOTH COULD BE MADE AND NEITHER COULD MOVE. create_lattice makes the lattice and add_modifier binds
    it, but a lattice deforms only by its points and no op wrote them - scaling the lattice OBJECT,
    which the lattice showcase keyed, moves the mesh and the lattice together and deforms nothing.
    add_shape_key is the same one level down: nothing wrote a key's vertices, so every key it made
    was a copy of the basis. Found 2026-09-11 by the lattice showcase's perfectly still head.

    THE CONSEQUENCE IS MEASURED, NOT ASSUMED. The written positions are read back, and every object
    the points deform - the mesh itself, or each object bound to the lattice - has its evaluated shape
    fingerprinted before and after, the same fingerprint add_modifier reports. A lattice nothing is
    bound to, or a key at value 0, moves points and changes nothing you can see; both are said.

    params:
      object (str, required)   a LATTICE or a MESH, in OBJECT mode
      points (list, required)  [[index, x, y, z], ...] in the object's LOCAL space. A lattice's point
                               index is u + v*pointsU + w*pointsU*pointsV, so a 3x3x3's centre is 13.
      shapeKey / key (str)     write that shape key's positions. REQUIRED on an object that has shape
                               keys, where the Basis key holds the rest shape and moving the base
                               points underneath it changes nothing.

    A CURVE (2026-10-05): points index every control point of every spline in order (a Bezier spline's
    bezier_points, a poly or NURBS spline's points); a Bezier point's handles move with it, as dragging it does.
    tilt [[index, radians], ...] writes the twist, and handleType {type FREE | VECTOR | ALIGNED | AUTO (this
    build's list), indices (default every Bezier point), side LEFT | RIGHT | BOTH} sets handle types after creation -
    a zip or thread curve smoothed by moving and twisting its points, then set to automatic handles. Either may come
    without points; the reply's curvePoints reads every point back (co, handles and their types, tilt, radius).
    radius [[index, r], ...] writes a point's radius (what a bevel and a curve-to-mesh scale by). handles [[index,
    [lx, ly, lz], [rx, ry, rz]], ...] places a Bezier point's two handles exactly, made FREE first. With NOTHING to write
    a CURVE is READ (second 1.3 audit, 2026-10-25): every control point's local and world position, handles, tilt,
    radius and weight, and each spline's type, count and cyclic flag (the first 10,000 points; truncated says so).
    """
    reject_unknown(params, _POINTS_KEYS, "set_point_positions")
    obj = get_object(take(params, "object", "name", required=True, kind=str))
    writes = [k for k in ("points", "tilt", "handleType", "radius", "shapeKey", "key", "handles")
              if params.get(k) is not None]
    if obj.type == "CURVE" and not writes:
        return _read_curve_points(obj, params)
    if obj.type != "CURVE" and params.get("radius") is not None:
        raise MifOpError("radius is a CURVE point's; '%s' is a %s. NOTHING was changed." % (obj.name, obj.type))
    if obj.type not in ("LATTICE", "MESH", "CURVE"):
        raise MifOpError("'%s' is a %s - set_point_positions moves a LATTICE's points, a MESH's "
                         "vertices and shape keys, or a CURVE's control points. NOTHING was changed." % (obj.name, obj.type))
    if obj.mode != "OBJECT":
        raise MifOpError("'%s' is in %s mode, which keeps its own copy of the points and writes it "
                         "back on exit, over this edit. Leave edit mode first. NOTHING was changed."
                         % (obj.name, obj.mode))
    require_editable(obj, "move points on")
    if obj.type == "CURVE":
        return _set_curve_points(obj, params)
    if params.get("tilt") is not None or params.get("handleType") is not None or params.get("handles") is not None:
        raise MifOpError("tilt, handles and handleType are a CURVE's; '%s' is a %s. NOTHING was changed." % (obj.name, obj.type))

    # EVERY ROW PARSED BEFORE ANY IS WRITTEN, so a bad tenth row cannot leave nine moved.
    raw = params.get("points")
    if not isinstance(raw, list) or not raw:
        raise MifOpError("'points' is required: [[index, x, y, z], ...] in the object's local space. "
                         "NOTHING was changed.")
    want = {}
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise MifOpError("every entry in 'points' is [index, x, y, z]; got %r. NOTHING was "
                             "changed." % (row,))
        i = finite_int(row[0], "points index")
        if i in want:
            raise MifOpError("point %d is given twice, and which position wins would be an accident "
                             "of order. NOTHING was changed." % i)
        want[i] = tuple(finite_floats(row[1:], "points"))

    keys = getattr(obj.data, "shape_keys", None)
    blocks = list(keys.key_blocks) if keys is not None else []
    key_name = take(params, "shapeKey", "key", default=None, kind=str)
    kb = None
    if key_name:
        kb = keys.key_blocks.get(key_name) if keys is not None else None
        if kb is None:
            raise MifOpError("no shape key named '%s' on '%s'. Present: %s. NOTHING was changed."
                             % (key_name, obj.name,
                                ", ".join(b.name for b in blocks) or "<none> - add_shape_key makes them"))
        coords, attr, target = kb.data, "co", "shape key '%s'" % kb.name
    elif blocks:
        raise MifOpError("'%s' has shape keys (%s), and the Basis key holds its rest shape - moving the "
                         "base points underneath it changes nothing. Pass shapeKey: '%s' to move the "
                         "rest shape, or name the key to edit. NOTHING was changed."
                         % (obj.name, ", ".join(b.name for b in blocks), blocks[0].name))
    elif obj.type == "LATTICE":
        coords, attr, target = obj.data.points, "co_deform", "lattice points"
    else:
        coords, attr, target = obj.data.vertices, "co", "mesh vertices"

    count = len(coords)
    bad = sorted(i for i in want if i < 0 or i >= count)
    if bad:
        raise MifOpError("point index %s is out of range - %s on '%s' has %d. NOTHING was changed."
                         % (bad[:8], target, obj.name, count))

    deformed = _deformed_by(obj)
    before = {o.name: _evaluated_counts(o) for o in deformed}
    for i, xyz in want.items():
        setattr(coords[i], attr, xyz)
    obj.data.update_tag()
    if keys is not None:
        keys.update_tag()

    worst = max(max(abs(getattr(coords[i], attr)[k] - xyz[k]) for k in range(3))
                for i, xyz in want.items())
    rows, changed = [], []
    for o in deformed:
        after = _evaluated_counts(o)
        moved = bool(after.get("shape")) and after.get("shape") != before[o.name].get("shape")
        rows.append({"object": o.name, "shapeBefore": before[o.name].get("shape"),
                     "shapeAfter": after.get("shape"), "changed": moved})
        if moved:
            changed.append(o.name)

    note = None
    if obj.type == "LATTICE" and not deformed:
        note = ("nothing is bound to this lattice, so these points deform nothing yet - give a mesh "
                "inside it add_modifier {type: LATTICE, settings: {object: '%s'}}." % obj.name)
    elif kb is not None and kb != keys.reference_key and (kb.mute or kb.value == 0.0):
        note = ("shape key '%s' is %s, so its new positions show only once set_shape_key gives it "
                "influence." % (kb.name, "muted" if kb.mute else "at value 0"))
    elif deformed and not changed:
        note = ("the points were written and read back, and no evaluated shape changed - a lattice "
                "reaches only geometry inside its own volume.")
    return {
        "ok": True,
        "object": obj.name,
        "target": target,
        "pointCount": count,
        "resolution": ([obj.data.points_u, obj.data.points_v, obj.data.points_w]
                       if obj.type == "LATTICE" else None),
        "written": len(want),
        "maxReadBackError": round(float(worst), 9),
        "readBackMatches": worst <= 1e-4,
        "deforms": rows,
        "evaluatedChanged": changed,
        "note": note,
    }


# sample and enableRigify: the Rigify sample form (2026-10-04, ops_rigify.add_rig_sample)
_ADDBONES_KEYS = {"object", "name", "armature", "bones", "replaceExisting", "sample", "enableRigify", "renameFrom",
                  "remove", "mergeWeights"}


def _skinned_meshes(obj):
    return [m for m in bpy.data.objects if m.type == "MESH"
            and any(md.type == "ARMATURE" and md.object == obj for md in m.modifiers)]


def _remove_bones(obj, params):
    """add_bones remove: delete bones, their children reparented to the nearest bone that stays, weights optionally
    merged into it. See op_add_bones."""
    tail = "NOTHING was changed."
    if params.get("bones") is not None or params.get("renameFrom") is not None or params.get("sample") is not None:
        raise MifOpError("remove deletes bones; pass it alone, not with bones, renameFrom or sample. %s" % tail)
    names = params.get("remove")
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names):
        raise MifOpError("remove is a non-empty list of bone names. %s" % tail)
    data = obj.data
    missing = [n for n in names if data.bones.get(n) is None]
    if missing:
        raise MifOpError("'%s' has no bone(s) %s (it has %d). %s" % (obj.name, ", ".join(missing), len(data.bones), tail))
    merge = take_bool(params, "mergeWeights", default=False)
    doomed = set(names)

    def keeper(bone):
        p = bone.parent
        while p is not None and p.name in doomed:
            p = p.parent
        return p.name if p is not None else None
    reparent = {b.name: keeper(b) for b in data.bones if b.name not in doomed and b.parent is not None
                and b.parent.name in doomed}
    target_of = {n: keeper(data.bones[n]) for n in names}
    count_before = len(data.bones)
    meshes = _skinned_meshes(obj)
    # Curves in actions that key a removed bone are left in the action (Blender does not touch them); named in the reply.
    from .ops_anim import _action_curves                       # noqa: PLC0415
    keyed = {}
    for act in bpy.data.actions:
        hit = sorted({n for fc in _action_curves(act) for n in doomed if fc.data_path.startswith('pose.bones["%s"]' % n)})
        if hit:
            keyed[act.name] = hit
    snap = selection_snapshot()
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        ebs = data.edit_bones
        for child, new_parent in reparent.items():
            eb = ebs[child]
            eb.use_connect = False
            eb.parent = ebs[new_parent] if new_parent else None
        for n in names:
            ebs.remove(ebs[n])
    finally:
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass
        selection_restore(snap)
    mode_after = bpy.context.object.mode if bpy.context.object else "OBJECT"
    left = [n for n in names if data.bones.get(n) is not None]
    if left or len(data.bones) != count_before - len(names):
        raise MifOpError("asked to remove %d bone(s); %d are still on '%s' (%s) and the count went %d -> %d."
                         % (len(names), len(left), obj.name, ", ".join(left), count_before, len(data.bones)))
    weights = []
    for m in meshes:
        for n in names:
            vg = m.vertex_groups.get(n)
            if vg is None:
                continue
            row = {"mesh": m.name, "group": n}
            if merge and target_of[n]:
                dst = m.vertex_groups.get(target_of[n]) or m.vertex_groups.new(name=target_of[n])
                moved = 0
                for v in m.data.vertices:
                    for g in v.groups:
                        if g.group == vg.index and g.weight > 0.0:
                            dst.add([v.index], g.weight, "ADD")
                            moved += 1
                m.vertex_groups.remove(vg)
                row.update(mergedInto=dst.name, vertices=moved)
            elif merge:
                m.vertex_groups.remove(vg)
                row.update(mergedInto=None, dropped=True)
            else:
                row["left"] = True
            weights.append(row)
    out = {"ok": True, "object": obj.name, "removed": names, "boneCountBefore": count_before,
           "boneCountAfter": len(data.bones), "reparented": reparent or None, "modeAfter": mode_after,
           "weights": weights, "mergeWeights": merge}
    left_groups = [w for w in weights if w.get("left")]
    if left_groups:
        out["warning"] = ("%d vertex group(s) named after a removed bone are left and now skin to nothing: %s. "
                          "mergeWeights:true moves their weights into the bone that takes over."
                          % (len(left_groups), ", ".join("%s.%s" % (w["mesh"], w["group"]) for w in left_groups[:10])))
    if keyed:
        out["actionsKeyingRemoved"] = keyed
    return out


def _mirror_bone_chain(obj, spec):
    """set_bone_rest mirror - see that op's docstring. Validated in full before edit mode is entered."""
    if not isinstance(spec, dict) or set(spec) - {"from", "axis", "createMissing", "roll", "pair"}:
        raise MifOpError("mirror is {from '_l'|'_r', axis 'X', createMissing, roll NEGATE|KEEP}; got %r. NOTHING was "
                         "changed." % (spec,))
    pair = spec.get("pair", ["_l", "_r"])
    if not isinstance(pair, (list, tuple)) or len(pair) != 2 or pair[0] == pair[1] \
            or any(not isinstance(s, str) or not s for s in pair):
        raise MifOpError("mirror.pair is two different suffixes. NOTHING was changed.")
    src = spec.get("from", pair[0])
    if src not in pair:
        raise MifOpError("mirror.from must be one of %s, got %r. NOTHING was changed." % (list(pair), src))
    if str(spec.get("axis", "X")).upper() != "X":
        raise MifOpError("mirror.axis is X (an armature's left and right). NOTHING was changed.")
    roll_mode = str(spec.get("roll", "NEGATE")).upper()
    if roll_mode not in ("NEGATE", "KEEP"):
        raise MifOpError("mirror.roll must be NEGATE or KEEP. NOTHING was changed.")
    create = bool(spec.get("createMissing", False))
    data = obj.data
    names = [b.name for b in data.bones]
    known = set(names)
    sources = [n for n in names if n.endswith(src)]
    if not sources:
        raise MifOpError("'%s' has no bone ending in %r to mirror from. NOTHING was changed." % (obj.name, src))
    plan, unpaired = [], []
    for n in sources:
        partner = _side_partner(n, pair)
        if partner not in known and not create:
            unpaired.append(n)
            continue
        plan.append((n, partner))
    if not plan:
        raise MifOpError("no bone ending in %r has a partner, and createMissing is off. NOTHING was changed." % src)
    planned = {p for _n, p in plan}
    snap = selection_snapshot()
    created = []
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        ebs = data.edit_bones
        for n, partner in plan:
            s = ebs[n]
            t = ebs.get(partner)
            if t is None:
                t = ebs.new(partner)
                created.append(partner)
            t.head = (-s.head.x, s.head.y, s.head.z)
            t.tail = (-s.tail.x, s.tail.y, s.tail.z)
            t.roll = -s.roll if roll_mode == "NEGATE" else s.roll
        for n, partner in plan:
            if partner in created:
                s = ebs[n]
                sp = s.parent.name if s.parent is not None else None
                mp = _side_partner(sp, pair) if sp else None
                target_parent = mp if (mp and (mp in known or mp in planned)) else sp
                if target_parent is not None:
                    ebs[partner].parent = ebs[target_parent]
                    ebs[partner].use_connect = s.use_connect
    finally:
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass
        selection_restore(snap)
    rows, off = [], []
    for n, partner in plan:
        a, b = data.bones[n], data.bones.get(partner)
        if b is None:
            off.append(partner)
            continue
        err = max(abs(-a.head_local.x - b.head_local.x), abs(a.head_local.y - b.head_local.y),
                  abs(a.head_local.z - b.head_local.z), abs(-a.tail_local.x - b.tail_local.x),
                  abs(a.tail_local.y - b.tail_local.y), abs(a.tail_local.z - b.tail_local.z))
        if err > 1e-4:
            off.append(partner)
        rows.append({"from": n, "to": partner, "head": rnd(list(b.head_local)), "tail": rnd(list(b.tail_local))})
    if off:
        raise MifOpError("bone(s) %s do not hold the mirrored rest after edit mode (a connected child follows its "
                         "parent's tail)." % ", ".join(off[:10]))
    return {"bonesWritten": rows, "created": created, "withoutPartner": unpaired}


_ROLL_TYPES = ("POS_X", "POS_Z", "GLOBAL_POS_X", "GLOBAL_POS_Y", "GLOBAL_POS_Z", "NEG_X", "NEG_Z", "GLOBAL_NEG_X",
               "GLOBAL_NEG_Y", "GLOBAL_NEG_Z", "ACTIVE", "VIEW", "CURSOR")


def _recalc_roll(obj, spec):
    """set_bone_rest recalcRoll (1.3 gap audit, 2026-10-19): Blender's Recalculate Roll (armature.calculate_roll) on a
    bone list or a name prefix, with each bone's roll before and after."""
    tail = "NOTHING was changed."
    if not isinstance(spec, dict):
        raise MifOpError("recalcRoll is {type, bones | prefix, axisFlip, axisOnly, active}. %s" % tail)
    bad = sorted(set(spec) - {"type", "bones", "prefix", "axisFlip", "axisOnly", "active"})
    if bad:
        raise MifOpError("recalcRoll has unknown key(s) %s. %s" % (bad, tail))
    kind = str(spec.get("type", "GLOBAL_POS_Z")).upper()
    if kind not in _ROLL_TYPES or kind == "VIEW":
        raise MifOpError("recalcRoll.type is one of %s (VIEW needs a viewport), got %r. %s"
                         % (", ".join(t for t in _ROLL_TYPES if t != "VIEW"), spec.get("type"), tail))
    names = spec.get("bones")
    prefix = spec.get("prefix")
    if (names is None) == (prefix is None):
        raise MifOpError("recalcRoll takes bones [names] or prefix. %s" % tail)
    data = obj.data
    if names is not None:
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            raise MifOpError("recalcRoll.bones is a list of bone names. %s" % tail)
        missing = [n for n in names if data.bones.get(n) is None]
        if missing:
            raise MifOpError("'%s' has no bone(s) %s. %s" % (obj.name, missing[:20], tail))
        names = list(dict.fromkeys(names))
    else:
        names = [b.name for b in data.bones if b.name.startswith(str(prefix))]
        if not names:
            raise MifOpError("no bone of '%s' starts with %r. %s" % (obj.name, prefix, tail))
    active = spec.get("active")
    if kind == "ACTIVE":
        if not isinstance(active, str) or data.bones.get(active) is None:
            raise MifOpError("recalcRoll type ACTIVE needs active: a bone of '%s' to align to. %s" % (obj.name, tail))
    elif active is not None:
        raise MifOpError("recalcRoll.active goes with type ACTIVE. %s" % tail)
    before = {n: _bone_roll(data.bones[n]) for n in names}
    snap = selection_snapshot()
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        for eb in data.edit_bones:
            sel = eb.name in before
            eb.select = eb.select_head = eb.select_tail = sel
        if kind == "ACTIVE":
            data.edit_bones.active = data.edit_bones[active]
            data.edit_bones[active].select = True
        bpy.ops.armature.calculate_roll(type=kind, axis_flip=bool(spec.get("axisFlip", False)),
                                        axis_only=bool(spec.get("axisOnly", False)))
    finally:
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass
        selection_restore(snap)
    rows = [{"name": n, "rollBefore": round(before[n], 6), "rollAfter": round(_bone_roll(data.bones[n]), 6)}
            for n in names]
    return {"type": kind, "bones": rows, "changed": sum(1 for r in rows if abs(r["rollAfter"] - r["rollBefore"]) > 1e-6)}


def op_set_bone_rest(params):
    """Move existing bones' REST positions - by head/tail/roll, or by applying the current pose as the rest.

    THE GAP, 2026-10-04: proportion variants of a character that keep Manny's bone names (a taller, broader sprayer)
    need the rest skeleton moved, and nothing typed could - add_bones adds, set_bone_pose poses. bones:
    [{name, head?, tail?, roll?}] in the ARMATURE's space (what edit mode shows), each one refused if it does not exist,
    if a value is not finite, or if it would leave a zero-length bone. applyPoseAsRest:true makes the current pose the
    rest (Blender's Apply Pose as Rest Pose) instead.

    READ BACK from data.bones after leaving edit mode - head_local/tail_local - against what was written. A mesh
    skinned to this armature is NOT updated: its Armature modifier now deforms it from a different rest, so every such
    mesh is named in skinnedMeshes with a warning (re-bind its weights, or apply its Armature modifier first).

    MIRROR A CHAIN (2026-10-04): mirror {from '_l' (default) | '_r', axis X, createMissing, roll NEGATE (default) | KEEP,
    pair} writes every bone ending in `from` onto its partner across X - head and tail with x negated, the roll negated
    or kept; createMissing makes a missing partner (parented to the mirrored parent, connected as its source). Bones
    with no partner (createMissing off) come back in withoutPartner.

    APPLY SELECTED (2026-10-04): applyPoseAsRest with bones [names] applies only those bones' pose (Blender's Apply
    Selected as Rest Pose, pose.armature_apply(selected=True), measured headless on 3.6.23, 4.4.0 and 5.2.1); the other
    bones keep their pose, and otherBonesMoved names any whose evaluated head moved because a parent's rest changed.

    RECALCULATE ROLL (1.3): recalcRoll {type POS_X ... GLOBAL_NEG_Z | ACTIVE (with active: a bone) | CURSOR, bones
    [names] or prefix, axisFlip, axisOnly} - Blender's Recalculate Roll on those bones, each one's roll before and after.
    """
    reject_unknown(params, ("object", "name", "armature", "bones", "applyPoseAsRest", "mirror", "recalcRoll"),
                   "set_bone_rest")
    obj = get_object(take(params, "object", "name", "armature", required=True, kind=str))
    if obj.type != "ARMATURE":
        raise MifOpError("'%s' is %s, not an ARMATURE. NOTHING was changed." % (obj.name, obj.type))
    require_editable(obj, "edit the rest bones of")
    if params.get("recalcRoll") is not None:
        if any(params.get(k) is not None for k in ("bones", "applyPoseAsRest", "mirror")):
            raise MifOpError("recalcRoll names its own bones; pass it alone. NOTHING was changed.")
        skinned_r = sorted(o.name for o in bpy.data.objects if o.type == "MESH"
                           and any(m.type == "ARMATURE" and m.object is obj for m in o.modifiers))
        out = {"object": obj.name, "mode": "recalcRoll", "skinnedMeshes": skinned_r}
        out.update(_recalc_roll(obj, params["recalcRoll"]))
        if skinned_r and out["changed"]:
            out["warning"] = ("%d mesh(es) skinned to '%s' deform from the rest as it was: a roll change turns the bones' "
                              "axes, which their weights do not depend on, but a pose keyed in local axes will play "
                              "differently: %s." % (len(skinned_r), obj.name, ", ".join(skinned_r[:10])))
        return out
    bones = params.get("bones")
    apply_pose = take_bool(params, "applyPoseAsRest", default=False)
    mirror = params.get("mirror")
    apply_names = None
    if apply_pose and bones is not None:
        if not isinstance(bones, list) or not bones or any(not isinstance(b, str) or not b for b in bones):
            # "bones OR" is pinned by test_blender_garment_ops GO192 (rest rows with applyPoseAsRest).
            raise MifOpError("with applyPoseAsRest, bones is a list of bone NAMES to apply - pass rest rows "
                             "{name, head, tail} as bones OR applyPoseAsRest, not both. NOTHING was changed.")
        missing = [b for b in bones if obj.pose.bones.get(b) is None]
        if missing:
            raise MifOpError("'%s' has no bone(s) %s. NOTHING was changed." % (obj.name, ", ".join(missing[:20])))
        apply_names, bones = list(bones), None
    if sum(1 for x in (bones is not None, apply_pose, mirror is not None) if x) != 1:
        raise MifOpError("pass one of bones, applyPoseAsRest:true (with bones [names] to apply only those) or mirror. "
                         "NOTHING was changed.")
    if mirror is not None:
        skinned_m = sorted(o.name for o in bpy.data.objects if o.type == "MESH"
                           and any(m.type == "ARMATURE" and m.object is obj for m in o.modifiers))
        out = {"object": obj.name, "mode": "mirror", "skinnedMeshes": skinned_m}
        out.update(_mirror_bone_chain(obj, mirror))
        if skinned_m:
            out["warning"] = ("%d mesh(es) skinned to '%s' now deform from a different rest: %s."
                              % (len(skinned_m), obj.name, ", ".join(skinned_m[:10])))
        return out
    data = obj.data
    plan = []
    if bones is not None:
        if not isinstance(bones, list) or not bones:
            raise MifOpError("bones must be a non-empty list of {name, head?, tail?, roll?}. NOTHING was changed.")
        bone_names = set(data.bones.keys())   # once - bpy collection membership walks the collection
        for i, b in enumerate(bones):
            if not isinstance(b, dict) or not isinstance(b.get("name"), str):
                raise MifOpError("bones[%d] needs a name. NOTHING was changed." % i)
            unknown = set(b) - {"name", "head", "tail", "roll"}
            if unknown:
                raise MifOpError("bones[%d] has unknown key(s) %s. NOTHING was changed." % (i, ", ".join(sorted(unknown))))
            if b["name"] not in bone_names:
                raise MifOpError("bones[%d]: '%s' has no bone '%s'. NOTHING was changed." % (i, obj.name, b["name"]))
            bone = data.bones[b["name"]]
            head = finite_floats(b["head"], "bones[%d].head" % i) if "head" in b else list(bone.head_local)
            tail = finite_floats(b["tail"], "bones[%d].tail" % i) if "tail" in b else list(bone.tail_local)
            if len(head) != 3 or len(tail) != 3:
                raise MifOpError("bones[%d]: head and tail are [x, y, z]. NOTHING was changed." % i)
            if sum((head[k] - tail[k]) ** 2 for k in range(3)) < 1e-12:
                raise MifOpError("bones[%d] ('%s'): head and tail are the same point - Blender deletes a zero-length "
                                 "bone when edit mode is left. NOTHING was changed." % (i, b["name"]))
            roll = finite_float(b["roll"], "bones[%d].roll" % i) if "roll" in b else None
            plan.append((b["name"], head, tail, roll))
    skinned = sorted(o.name for o in bpy.data.objects if o.type == "MESH"
                     and any(m.type == "ARMATURE" and m.object is obj for m in o.modifiers))
    snap = selection_snapshot()
    kept_basis, heads_before, bone_sel = {}, {}, None
    if apply_names is not None:
        from .ops_pose import evaluated_pose, select_bones   # noqa: PLC0415
        kept_basis = {pb.name: pb.matrix_basis.copy() for pb in obj.pose.bones if pb.name not in apply_names}
        heads_before = {n: m.to_translation() for n, m in evaluated_pose(obj).items() if n not in apply_names}
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        if apply_pose:
            bpy.ops.object.mode_set(mode="POSE")
            if apply_names is not None:
                bone_sel = select_bones(obj, set(apply_names))
                bpy.ops.pose.armature_apply(selected=True)
            else:
                bpy.ops.pose.armature_apply(selected=False)
        else:
            bpy.ops.object.mode_set(mode="EDIT")
            ebs = data.edit_bones
            for name, head, tail, roll in plan:
                eb = ebs[name]
                eb.head = head
                eb.tail = tail
                if roll is not None:
                    eb.roll = roll
    finally:
        if bone_sel is not None:
            from .ops_pose import restore_bone_selection   # noqa: PLC0415
            restore_bone_selection(obj, bone_sel)
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass
        selection_restore(snap)
    out = {"object": obj.name, "mode": "applyPoseAsRest" if apply_pose else "bones", "skinnedMeshes": skinned}
    if apply_pose:
        pool = [obj.pose.bones[n] for n in apply_names] if apply_names is not None else list(obj.pose.bones)
        moved = [pb.name for pb in pool
                 if any(abs(pb.matrix_basis[r][c] - (1.0 if r == c else 0.0)) > 1e-5 for r in range(4) for c in range(4))]
        if moved:
            raise MifOpError("Apply Pose as Rest ran and %d bone(s) still hold a pose: %s." % (len(moved), moved[:10]))
        out["poseCleared"] = True
        if apply_names is not None:
            from .ops_pose import evaluated_pose   # noqa: PLC0415
            changed = [n for n, m in kept_basis.items()
                       if any(abs(obj.pose.bones[n].matrix_basis[r][c] - m[r][c]) > 1e-5 for r in range(4)
                              for c in range(4))]
            if changed:
                raise MifOpError("Apply Selected as Rest ran and changed the pose of bone(s) it was not given: %s."
                                 % changed[:10])
            now = evaluated_pose(obj)
            out["bonesApplied"] = apply_names
            out["otherBonesKeptPose"] = len(kept_basis)
            out["otherBonesMoved"] = sorted(n for n, h in heads_before.items()
                                            if (now[n].to_translation() - h).length > 1e-4)[:50]
    else:
        rows, off = [], []
        for name, head, tail, _roll in plan:
            b = data.bones.get(name)
            if b is None:
                off.append(name)
                continue
            dh = max(abs(b.head_local[k] - head[k]) for k in range(3))
            dt = max(abs(b.tail_local[k] - tail[k]) for k in range(3))
            if dh > 1e-4 or dt > 1e-4:
                off.append(name)
            rows.append({"name": name, "head": rnd(list(b.head_local)), "tail": rnd(list(b.tail_local))})
        if off:
            raise MifOpError("bone(s) %s do not hold the rest written to them after edit mode (a connected child "
                             "follows its parent's tail)." % ", ".join(off))
        out["bones"] = rows
    if skinned:
        out["warning"] = ("%d mesh(es) skinned to '%s' were not updated and now deform from a different rest: %s. "
                          "Re-bind their weights, or apply their Armature modifier first."
                          % (len(skinned), obj.name, ", ".join(skinned[:10])))
    return out


def _renamed_copies(obj, params):
    """add_bones renameFrom: ([bone dicts for the copies, parents first], {copy name: use_deform}). Refuses before any
    write - bones beside it, a find that matches nothing, or a new name that is taken."""
    rename = params.get("renameFrom")
    if (not isinstance(rename, dict) or not rename
            or any(not isinstance(k, str) or not k or not isinstance(v, str) for k, v in rename.items())):
        raise MifOpError("renameFrom must be {find: replace} with text on both sides (got %r). NOTHING was changed."
                         % (rename,))
    if params.get("bones") is not None:
        raise MifOpError("renameFrom copies existing bones; pass it OR bones. NOTHING was changed.")

    def renamed(nm):
        for find, repl in rename.items():
            if find in nm:
                return nm.replace(find, repl)
        return None
    data = obj.data
    picked = [b for b in data.bones if renamed(b.name) is not None]
    if not picked:
        raise MifOpError("no bone of '%s' holds %s. NOTHING was changed." % (obj.name, ", ".join(map(repr, rename))))
    new_names = {b.name: renamed(b.name) for b in picked}
    have = {b.name for b in data.bones}
    clash = sorted(n for n in new_names.values() if n in have or n == "")
    if clash or len(set(new_names.values())) != len(new_names):
        raise MifOpError("the copies' names %s are taken on '%s' or repeat; Blender would add .001. NOTHING was "
                         "changed." % (", ".join(clash) or "repeat", obj.name))
    picked.sort(key=lambda b: len(b.parent_recursive))
    out, deform = [], {}
    for b in picked:
        parent = b.parent.name if b.parent else None
        row = {"name": new_names[b.name], "head": list(b.head_local), "tail": list(b.tail_local),
               "roll": _bone_roll(b), "connect": bool(b.use_connect)}
        if parent is not None:
            row["parent"] = new_names.get(parent, parent)
        else:
            row["connect"] = False
        out.append(row)
        deform[row["name"]] = bool(b.use_deform)
    return out, deform


def op_add_bones(params):
    """Add bones to an armature that already exists - which nothing could do.

    THE HOLE THIS CLOSES. create_armature builds bones and every other rigging op edits what is
    already there: list_bones, rename_bones, set_bone_pose, the weight ops. edit_bones appears in
    exactly ONE place in the whole addon, inside create_armature, so a skeleton that arrived through
    import_mesh could be renamed, posed and weighted and never given another bone. Adding a socket
    bone, or an ik_hand_gun beside the hand, is ordinary skeletal work for a game engine and was
    impossible through the typed path.

    BONES ONLY EXIST IN EDIT MODE, which is the whole difficulty, and RESTORING THE MODE IS A
    POSTCONDITION rather than a courtesy - being left in edit mode strands every op that follows.
    Same discipline as create_armature, and the mode is asserted afterwards rather than assumed.

    A PARENT MAY BE A BONE THAT WAS ALREADY THERE or one created in this same call, which is the
    difference from create_armature: it resolves against the live edit_bones list, so an imported
    skeleton's existing bone is a legal parent.

    A ZERO-LENGTH BONE IS REFUSED. Blender DELETES a bone whose head equals its tail when edit mode
    is left - silently, with no error - so the count check at the end would catch it and the caller
    would get a confusing failure instead of the real reason.

    params:
      object / name / armature (str)  the armature object. Required.
      bones (list)   [{"name","head":[x,y,z],"tail":[x,y,z],"parent":"...","connect":bool,
                       "roll":float}]   Parents must exist already or appear earlier in the list.
      replaceExisting (bool)   default false. Adding a bone whose name is taken gives it a .001
                               suffix, and anything looking it up by name afterwards finds the wrong
                               one - so it is refused unless this says otherwise.
      renameFrom ({find: replace})   instead of bones (2026-10-06, 5.2's Duplicate and Rename, on every
                               build): every bone whose name holds a `find` is copied under the name with
                               it replaced (DEF- to CTRL-) - head, tail, roll, connect and deform kept, a
                               copy parented to its parent's copy when that was copied too. A new name
                               that is taken is refused, so no .001 appears.
      remove [names]           instead of bones (second 1.3 audit, 2026-10-25): delete these bones. Each child of a
                               removed bone is reparented to the nearest ancestor that stays (unconnected, so its head
                               does not move); the mode is restored and the count asserted. Vertex groups named after a
                               removed bone are left and warned about, or with mergeWeights:true added into the group of
                               the bone that takes over (dropped for a removed root). Actions still keying a removed
                               bone are named (actionsKeyingRemoved).
    """
    reject_unknown(params, _ADDBONES_KEYS, "add_bones")
    if params.get("remove") is not None:
        obj = get_object(take(params, "object", "name", "armature", required=True, kind=str))
        if obj.type != "ARMATURE":
            raise MifOpError("'%s' is a %s, not an ARMATURE. NOTHING was changed." % (obj.name, obj.type))
        return _remove_bones(obj, params)
    if params.get("mergeWeights") is not None:
        raise MifOpError("mergeWeights is for remove. NOTHING was changed.")
    if params.get("sample") is not None:
        from .ops_rigify import add_rig_sample
        return add_rig_sample(params)
    if params.get("enableRigify") is not None:
        raise MifOpError("enableRigify is for the sample form - pass sample too. NOTHING was changed.")
    obj = get_object(take(params, "object", "name", "armature", required=True, kind=str))
    if obj.type != "ARMATURE":
        raise MifOpError("'%s' is a %s, not an ARMATURE. NOTHING was changed."
                         % (obj.name, obj.type))
    data = obj.data
    deform = {}
    if params.get("renameFrom") is not None:
        bones, deform = _renamed_copies(obj, params)
    else:
        bones = params.get("bones")
    if not isinstance(bones, (list, tuple)) or not bones:
        raise MifOpError("'bones' must be a non-empty list of {name, head, tail}. NOTHING was "
                         "changed.")

    # VALIDATED IN FULL BEFORE ANY MODE CHANGE, for create_armature's reason: a refusal partway
    # through would leave the armature half-built AND Blender sitting in edit mode.
    replace = take_bool(params, "replaceExisting", default=False)
    existing = {b.name for b in data.bones}
    planned = set()
    ends = {}          # i -> (head, tail, roll, connect), checked - parsed once, written by the loop further down
    for i, b in enumerate(bones):
        # A STRING, not merely truthy - and str() would have hidden it. A dict name passed
        # this guard and str({...}) turned it into a bone literally called
        # "{'mif': 'not-a-value'}", ACCEPTED and reported as created. My own code from the
        # same day, found by the matrix pass once it learned to corrupt list-of-dict entries.
        if not isinstance(b, dict) or not isinstance(b.get("name"), str) or not b["name"]:
            raise MifOpError("bones[%d] needs a 'name' as text, got %r. NOTHING was changed."
                             % (i, b.get("name") if isinstance(b, dict) else b))
        nm = b["name"]
        if len(nm) > _ID_NAME_LIMIT:
            raise MifOpError("bones[%d] name is %d characters and Blender truncates at %d, so the "
                             "bone you get would not be the one you named. NOTHING was changed."
                             % (i, len(nm), _ID_NAME_LIMIT))
        if nm in existing and not replace:
            raise MifOpError(
                "'%s' already has a bone named '%s'. Blender would add '%s.001' beside it and "
                "anything looking the name up afterwards would find the wrong bone - including the "
                "vertex groups that skin it. Pass replaceExisting:true to add it anyway, or pick "
                "another name. NOTHING was changed." % (obj.name, nm, nm))
        if nm in planned:
            raise MifOpError("bones[%d] repeats the name '%s' inside this call. NOTHING was "
                             "changed." % (i, nm))
        planned.add(nm)
        for end in ("head", "tail"):
            v = b.get(end)
            if not isinstance(v, (list, tuple)) or len(v) < 3:
                raise MifOpError("bones[%d] needs '%s' as [x,y,z]. NOTHING was changed." % (i, end))
        # FINITE, and kept for the write loop below rather than converted a second time there. A bare
        # float() put a bone head at nan under ok:true, took a NaN roll, and raised a raw ValueError on a
        # string (5.0.1, 2026-09-20; audit_unguarded_numbers' provenance pass).
        try:
            head = finite_floats(list(b["head"])[:3], "bones[%d].head" % i)
            tail = finite_floats(list(b["tail"])[:3], "bones[%d].tail" % i)
            roll = finite_float(b["roll"], "bones[%d].roll" % i) if b.get("roll") is not None else None
        except MifOpError as exc:
            raise MifOpError("%s NOTHING was changed." % exc)
        # take_bool: the write loop used bool(b.get("connect")), and bool("false") is True - {connect: "false"}
        # CONNECTED the bone and moved its head onto the parent's tail (measured on 5.0.1, 2026-09-21).
        try:
            connect = take_bool(b, "connect", default=False)
        except MifOpError as exc:
            raise MifOpError("bones[%d] ('%s'): %s NOTHING was changed." % (i, nm, exc))
        ends[i] = (head, tail, roll, connect)
        if head == tail:
            raise MifOpError(
                "bones[%d] ('%s') has head == tail, so it has ZERO length. Blender DELETES a "
                "zero-length bone when edit mode is left, silently and with no error, so this would "
                "report a confusing count mismatch instead of the real reason. NOTHING was changed."
                % (i, nm))
        parent = b.get("parent")
        if parent is not None and str(parent) not in existing and str(parent) not in planned:
            known = sorted(existing)[:25]
            raise MifOpError(
                "bones[%d] ('%s') names parent '%s', which is neither on '%s' already nor earlier "
                "in this list. Existing bones: %s. NOTHING was changed."
                % (i, nm, parent, obj.name, ", ".join(known) if known else "(none)"))

    count_before = len(data.bones)
    snap = selection_snapshot()
    made = []
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        ebs = data.edit_bones
        for i, b in enumerate(bones):
            eb = ebs.new(str(b["name"]))
            head, tail, roll, connect = ends[i]
            eb.head = head
            eb.tail = tail
            if roll is not None:
                eb.roll = roll
            if b.get("parent"):
                eb.parent = ebs[str(b["parent"])]
                # use_connect MOVES THE CHILD'S HEAD onto the parent's tail. Off by default, because
                # a socket bone placed at a deliberate offset would silently jump.
                eb.use_connect = connect
            made.append(eb.name)
    finally:
        # ALWAYS. Left in edit mode, every op after this one fails on an editor nobody asked for.
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass
        selection_restore(snap)

    mode_after = bpy.context.object.mode if bpy.context.object else "OBJECT"
    if mode_after != "OBJECT":
        raise MifOpError("added bones to '%s' but Blender is left in %s mode, which strands every "
                         "op after this one." % (obj.name, mode_after))

    # COUNTED FROM data.bones, NOT from the edit_bones list above. edit_bones exist only in edit
    # mode; the real bones appear when it is left, and one that did not survive the transition
    # would still be in the list this op built.
    now = {b.name for b in data.bones}
    landed = [n for n in made if n in now]
    if len(landed) != len(bones):
        lost = [n for n in made if n not in now]
        raise MifOpError("asked for %d bone(s); %d survived leaving edit mode. Missing: %s."
                         % (len(bones), len(landed), ", ".join(lost) or "(renamed)"))
    for nm, flag in deform.items():
        data.bones[nm].use_deform = flag

    return {
        "ok": True,
        "object": obj.name,
        "added": landed,
        "boneCountBefore": count_before,
        "boneCountAfter": len(data.bones),
        "bones": sorted(now),
        # THE MODE IS REPORTED because it is the postcondition that matters to the NEXT call, not
        # to this one.
        "modeAfter": mode_after,
        "parentedTo": {str(b["name"]): str(b["parent"]) for b in bones if b.get("parent")} or None,
        "note": ("these bones deform NOTHING until a mesh is weighted to them - add_bones only "
                 "builds the skeleton. set_vertex_weights puts weights on the groups that match "
                 "them, and a vertex group whose name matches no bone is skinned to nothing."),
    }

_BONEPROP_KEYS = ("object", "name", "armature", "bones", "namePrefix", "all", "useDeform", "parent", "clearParent",
                  "connected", "bboneSegments", "inheritScale", "dryRun", "customShape", "customShapeTranslation",
                  "customShapeRotation", "customShapeScale", "color", "colorPalette", "previewId", "previewTtlSeconds")
_SHAPE_VECS = (("customShapeTranslation", "custom_shape_translation", False),
               ("customShapeRotation", "custom_shape_rotation_euler", True),
               ("customShapeScale", "custom_shape_scale_xyz", False))


def _bone_props(obj, n):
    """One bone's properties as set_bone_properties writes and reports them - the op's read, and its preview's."""
    b = obj.data.bones[n]
    pb = obj.pose.bones[n]
    row = {"useDeform": bool(b.use_deform), "parent": b.parent.name if b.parent else None,
           "connected": bool(b.use_connect), "bboneSegments": int(b.bbone_segments),
           "inheritScale": b.inherit_scale,
           "customShape": pb.custom_shape.name if pb.custom_shape is not None else None}
    for key, attr, deg in _SHAPE_VECS:
        if hasattr(pb, attr):
            v = list(getattr(pb, attr))
            row[key] = [round(math.degrees(x), 4) if deg else round(x, 6) for x in v]
    if hasattr(pb, "color"):
        row["colorPalette"] = pb.color.palette
        row["color"] = {s: [round(x, 6) for x in getattr(pb.color.custom, s)] for s in ("normal", "select", "active")}
    return row


def op_set_bone_properties(params):
    """Write existing bones' properties: deform, parent, connected, bendy segments, inherit scale.

    THE GAP (spec, "Blender: rigging and weights"): list_bones reads useDeform and connected, add_bones and set_bone_rest
    write positions, and nothing wrote the rest - Deform off on a root and the IK targets before automatic weights,
    bendy segments back to 1 (the engine importer ignores them), a hierarchy reparented to its deform bones.

    Bones by `bones` (names), `namePrefix` or all:true - exactly one. useDeform (bool), parent (a bone name - not itself or one of its
    own descendants) or clearParent:true, connected (bool: on puts the head on
    the parent's tail, as Blender does), bboneSegments (1-32), inheritScale (one of Blender's own Bone.inherit_scale enum
    items, read from this build). DISPLAY (2026-10-04): customShape (an object; "" clears it), customShapeTranslation,
    customShapeRotation (degrees) and customShapeScale ([x, y, z]), color {normal, select, active} ([r, g, b] 0-1, the
    palette set to CUSTOM) or colorPalette (a theme set, THEME01 ... from this build's list) - bone colours are per
    bone from Blender 4.0 and refused before it. dryRun reports what would change. Each bone's previous and new values come back, read
    off the data after the write. dryRun is also the BULK PREVIEW (docs/15 section 5): one change per bone per property,
    and a previewId that {previewId} alone commits - planned again, refused on any difference, read back, put back on a
    mismatch.
    """
    reject_unknown(params, _BONEPROP_KEYS, "set_bone_properties")
    obj = get_object(take(params, "object", "name", "armature", required=True, kind=str))
    if obj.type != "ARMATURE":
        raise MifOpError("'%s' is %s, not an ARMATURE. NOTHING was changed." % (obj.name, obj.type))
    require_editable(obj, "set bone properties on")
    data = obj.data
    names_all = [b.name for b in data.bones]
    by = [k for k in ("bones", "namePrefix", "all") if params.get(k) not in (None, False)]
    if len(by) != 1:
        raise MifOpError("pass exactly one of bones, namePrefix or all:true (got %s). NOTHING was changed."
                         % (", ".join(by) or "none"))
    if by[0] == "bones":
        raw = params["bones"]
        if not isinstance(raw, (list, tuple)) or not raw or any(not isinstance(n, str) for n in raw):
            raise MifOpError("bones must be a list of bone names. NOTHING was changed.")
        known = set(names_all)
        missing = [n for n in raw if n not in known]
        if missing:
            raise MifOpError("'%s' has no bone %s. NOTHING was changed." % (obj.name, ", ".join(missing[:10])))
        targets = list(dict.fromkeys(raw))
    elif by[0] == "namePrefix":
        prefix = take(params, "namePrefix", kind=str)
        targets = [n for n in names_all if n.startswith(prefix)]
        if not targets:
            raise MifOpError("no bone of '%s' starts with %r. NOTHING was changed." % (obj.name, prefix))
    else:
        take_bool(params, "all")
        targets = list(names_all)
    want = {}
    if params.get("useDeform") is not None:
        want["useDeform"] = take_bool(params, "useDeform")
    clear_parent = take_bool(params, "clearParent", default=False)
    if clear_parent and params.get("parent") is not None:
        raise MifOpError("pass parent OR clearParent:true. NOTHING was changed.")
    if clear_parent:
        want["parent"] = None
    elif params.get("parent") is not None:
        p = params.get("parent")
        if not isinstance(p, str) or p not in set(names_all):
            raise MifOpError("parent %r is not a bone of '%s' (clearParent:true for no parent). NOTHING was changed."
                             % (p, obj.name))
        want["parent"] = p
    if params.get("connected") is not None:
        want["connected"] = take_bool(params, "connected")
    if params.get("bboneSegments") is not None:
        seg = take_int(params, "bboneSegments")
        if not 1 <= seg <= 32:
            raise MifOpError("bboneSegments must be 1-32, got %d. NOTHING was changed." % seg)
        want["bboneSegments"] = seg
    if params.get("inheritScale") is not None:
        mode = take(params, "inheritScale", kind=str)
        items = [i.identifier for i in bpy.types.Bone.bl_rna.properties["inherit_scale"].enum_items]
        if mode not in items:
            raise MifOpError("inheritScale must be one of this Blender's %s, got %r. NOTHING was changed."
                             % (", ".join(items), mode))
        want["inheritScale"] = mode
    # DISPLAY (2026-10-04): the pose bone's custom shape and its transform, and the bone colour (4.0+).
    if params.get("customShape") is not None:
        cs = params.get("customShape")
        if cs == "":
            want["customShape"] = None
        else:
            ob = bpy.data.objects.get(str(cs))
            if ob is None or ob.name == obj.name:
                raise MifOpError("customShape %r is %s. NOTHING was changed."
                                 % (cs, "the armature itself" if ob is not None else "not an object"))
            want["customShape"] = ob.name
    for key, attr, deg in _SHAPE_VECS:
        if params.get(key) is not None:
            v = params.get(key)
            if not isinstance(v, (list, tuple)) or len(v) != 3:
                raise MifOpError("%s is [x, y, z]%s. NOTHING was changed." % (key, " in degrees" if deg else ""))
            want[key] = [round(x, 6) for x in finite_floats(list(v), key)]
    palette_given = params.get("colorPalette") is not None
    if params.get("color") is not None or palette_given:
        if not hasattr(bpy.types, "BoneColor"):
            raise MifOpError("bone colours are per bone from Blender 4.0 (BoneColor); %s keeps them in bone groups. "
                             "NOTHING was changed." % bpy.app.version_string)
        items = [e.identifier for e in bpy.types.BoneColor.bl_rna.properties["palette"].enum_items]
        if palette_given:
            pal = str(params.get("colorPalette")).upper()
            if pal not in items or pal == "CUSTOM":
                raise MifOpError("colorPalette must be one of this Blender's %s (CUSTOM comes with color). NOTHING was "
                                 "changed." % ", ".join(i for i in items if i != "CUSTOM"))
            if params.get("color") is not None:
                raise MifOpError("pass color (custom colours) OR colorPalette (a theme set). NOTHING was changed.")
            want["colorPalette"] = pal
        else:
            col = params.get("color")
            if not isinstance(col, dict) or set(col) - {"normal", "select", "active"} or not col:
                raise MifOpError("color is {normal, select, active}, each [r, g, b] 0-1. NOTHING was changed.")
            cw = {}
            for slot, rgb in col.items():
                if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
                    raise MifOpError("color.%s is [r, g, b]. NOTHING was changed." % slot)
                vals = finite_floats(list(rgb), "color.%s" % slot)
                if any(not 0.0 <= x <= 1.0 for x in vals):
                    raise MifOpError("color.%s values must be 0-1, got %r. NOTHING was changed." % (slot, rgb))
                cw[slot] = [round(x, 6) for x in vals]
            want["color"] = cw
    if not want:
        raise MifOpError("say what to set: useDeform, parent, connected, bboneSegments or inheritScale. NOTHING was "
                         "changed.")
    if want.get("connected") and want.get("parent", "x") is None:
        raise MifOpError("a bone with no parent cannot be connected. NOTHING was changed.")
    if "parent" in want and want["parent"] is not None:
        for n in targets:
            if n == want["parent"]:
                raise MifOpError("bone '%s' cannot be its own parent. NOTHING was changed." % n)
            anc = data.bones[want["parent"]]
            while anc is not None:
                if anc.name == n:
                    raise MifOpError("'%s' is a descendant of '%s', so making it the parent would make a loop. NOTHING "
                                     "was changed." % (want["parent"], n))
                anc = anc.parent
    if want.get("connected") and "parent" not in want:
        orphans = [n for n in targets if data.bones[n].parent is None]
        if orphans:
            raise MifOpError("bone(s) %s have no parent, so they cannot be connected. NOTHING was changed."
                             % ", ".join(orphans[:10]))

    def read(n):
        return _bone_props(obj, n)
    before = {n: read(n) for n in targets}
    if take_bool(params, "dryRun", default=False):
        return {"object": obj.name, "dryRun": True, "bones": targets, "set": want, "before": before}
    if "parent" in want or "connected" in want:
        snap = selection_snapshot()
        try:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.mode_set(mode="EDIT")
            ebs = data.edit_bones
            for n in targets:
                eb = ebs[n]
                if "parent" in want:
                    eb.parent = ebs[want["parent"]] if want["parent"] is not None else None
                if "connected" in want:
                    eb.use_connect = want["connected"] and eb.parent is not None
        finally:
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                pass
            selection_restore(snap)
    for n in targets:
        b = data.bones[n]
        if "useDeform" in want:
            b.use_deform = want["useDeform"]
        if "bboneSegments" in want:
            b.bbone_segments = want["bboneSegments"]
        if "inheritScale" in want:
            b.inherit_scale = want["inheritScale"]
        pb = obj.pose.bones[n]
        if "customShape" in want:
            pb.custom_shape = bpy.data.objects[want["customShape"]] if want["customShape"] else None
        for key, attr, deg in _SHAPE_VECS:
            if key in want:
                setattr(pb, attr, [math.radians(x) if deg else x for x in want[key]])
        if "colorPalette" in want:
            pb.color.palette = want["colorPalette"]
        if "color" in want:
            pb.color.palette = "CUSTOM"
            for slot, rgb in want["color"].items():
                setattr(pb.color.custom, slot, rgb)
    after = {n: read(n) for n in targets}
    def held(row, k, v):
        if k == "color":
            return row.get("colorPalette") == "CUSTOM" and all(
                max(abs(a - c) for a, c in zip(row["color"][s], rgb)) < 1e-5 for s, rgb in v.items())
        if isinstance(v, list):
            return max(abs(a - c) for a, c in zip(row.get(k) or [0, 0, 0], v)) < 1e-4
        return row.get(k) == v
    off = [n for n in targets if not all(held(after[n], k, v) for k, v in want.items())]
    if off:
        raise MifOpError("bone(s) %s do not hold what was written to them: %s." % (", ".join(off[:10]),
                                                                                 [after[n] for n in off[:3]]))
    rows = [{"name": n, "before": before[n], "after": after[n]} for n in targets if before[n] != after[n]]
    return {"object": obj.name, "set": want, "bonesNamed": len(targets), "bonesChanged": len(rows), "changes": rows}


# ---------------------------------------------------------------------------------------------------------------------
# THE BULK PREVIEW of set_bone_properties (ops_preview.py; docs/15 section 5, 2026-10-09). One change per bone per property
# set, before as the op reads it and after as it will write it (connected only where the bone has a parent then; color
# with the palette CUSTOM). A parent or connected change is undone in edit mode - parent, then each bone's matrix and
# length, then connected - because connecting moved the head onto the parent's tail.
# ---------------------------------------------------------------------------------------------------------------------
def _plan_set_bone_properties(request):
    from .ops_preview import _near   # noqa: PLC0415
    resp = op_set_bone_properties(dict(request, dryRun=True))          # every refusal, and writes nothing
    obj = bpy.data.objects[resp["object"]]
    want, before = resp["set"], resp["before"]
    changes, unchanged = [], []
    for n in resp["bones"]:
        b = before[n]
        after = {}
        for field, value in want.items():
            if field == "connected":
                after[field] = bool(value) and (want["parent"] if "parent" in want else b["parent"]) is not None
            elif field == "color":
                after["colorPalette"] = "CUSTOM"
                after["color"] = dict(b.get("color") or {}, **value)
            else:
                after[field] = value
        for field, value in after.items():
            row = {"kind": "bone", "op": "modify", "target": obj.name, "field": "bones['%s'].%s" % (n, field),
                   "label": n, "before": b.get(field), "after": value, "_bone": n, "_field": field}
            (unchanged if _near(row["before"], value) else changes).append(row)
    return {"changes": changes, "unchanged": unchanged, "context": [], "commit": dict(request), "response": resp}


def _read_set_bone_properties(change, _kept):
    obj = bpy.data.objects.get(change["target"])
    if obj is None or obj.type != "ARMATURE" or obj.data.bones.get(change["_bone"]) is None:
        return None
    return _bone_props(obj, change["_bone"]).get(change["_field"])


def _capture_set_bone_properties(plan):
    obj = bpy.data.objects[plan["response"]["object"]]
    kept = {"object": obj.name, "props": {n: _bone_props(obj, n) for n in plan["response"]["bones"]}, "edit": None}
    if any(c["_field"] in ("parent", "connected") for c in plan["changes"]):
        kept["edit"] = {b.name: {"parent": b.parent.name if b.parent else None, "connect": bool(b.use_connect),
                                 "matrix": b.matrix_local.copy(), "length": b.length} for b in obj.data.bones}
    return kept


def _restore_set_bone_properties(kept, _plan):
    obj = bpy.data.objects.get(kept["object"])
    if obj is None or obj.type != "ARMATURE":
        raise MifOpError("armature '%s' is gone, so its bones cannot be put back" % kept["object"])
    if kept["edit"]:
        snap = selection_snapshot()
        try:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.mode_set(mode="EDIT")
            ebs = obj.data.edit_bones
            rows = [(ebs.get(n), e) for n, e in kept["edit"].items() if ebs.get(n) is not None]
            for eb, _e in rows:
                eb.use_connect = False
            for eb, e in rows:
                eb.parent = ebs.get(e["parent"]) if e["parent"] else None
            for eb, e in rows:
                eb.matrix = e["matrix"]
                eb.length = e["length"]
            for eb, e in rows:
                eb.use_connect = e["connect"]
        finally:
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                pass
            selection_restore(snap)
    for n, p in kept["props"].items():
        b, pb = obj.data.bones.get(n), obj.pose.bones.get(n)
        if b is None or pb is None:
            continue
        b.use_deform = p["useDeform"]
        b.bbone_segments = p["bboneSegments"]
        b.inherit_scale = p["inheritScale"]
        pb.custom_shape = bpy.data.objects.get(p["customShape"]) if p["customShape"] else None
        for key, attr, deg in _SHAPE_VECS:
            if key in p and hasattr(pb, attr):
                setattr(pb, attr, [math.radians(x) if deg else x for x in p[key]])
        if "color" in p and hasattr(pb, "color"):
            for slot, rgb in p["color"].items():
                setattr(pb.color.custom, slot, rgb)
            pb.color.palette = p["colorPalette"]
    bpy.context.view_layer.update()


# apply_modifier plans on a scratch copy (ops_preview.mesh_spec); these are the refusals the copy cannot make.
_SIM_MODIFIERS = ("CLOTH", "SOFT_BODY", "FLUID", "DYNAMIC_PAINT", "OCEAN", "EXPLODE", "PARTICLE_INSTANCE",
                  "PARTICLE_SYSTEM", "COLLISION")


def _check_apply_modifier(_request, resp, obj):
    if obj.data.users > 1:
        raise MifOpError("'%s' shares its mesh data with %d other object(s), and applying a modifier rewrites that data "
                         "for every one of them. Make it single-user in Blender first. Nothing was written."
                         % (obj.name, obj.data.users - 1))
    mod = obj.modifiers.get(resp["modifier"])
    if mod is not None and mod.type in _SIM_MODIFIERS:
        raise MifOpError("a %s modifier's shape at this frame comes from its simulation and its cache, which the "
                         "preview's scratch copy does not share - so no preview can show what applying it writes. Apply "
                         "it without dryRun; undo takes it back. Nothing was written." % mod.type)


def _more_previews():
    from . import ops_preview as preview   # noqa: PLC0415
    return {"set_bone_properties": {"plan": _plan_set_bone_properties, "read": _read_set_bone_properties,
                                    "capture": _capture_set_bone_properties,
                                    "restore": _restore_set_bone_properties},
            "apply_modifier": preview.mesh_spec(op_apply_modifier, check=_check_apply_modifier)}


PREVIEW.update(_more_previews())


def _ue_quat(pitch, yaw, roll):
    """FRotator::Quaternion (UE 5.3 Runtime/Core/Private/Math/UnrealMath.cpp, the scalar branch) - (x, y, z, w)."""
    p, y, r = (math.radians(a) / 2.0 for a in (pitch, yaw, roll))
    sp, cp, sy, cy, sr, cr = math.sin(p), math.cos(p), math.sin(y), math.cos(y), math.sin(r), math.cos(r)
    return (cr * sp * sy - sr * cp * cy, -cr * sp * cy - sr * cp * sy, cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def _qmul(a, b):
    """a * b (Hamilton product, UE's operator*: b applied first)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def _qrot(q, v):
    """TQuat::RotateVector: V + w(2 Q x V) + Q x (2 Q x V)."""
    qx, qy, qz, w = q

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    t = tuple(2.0 * c for c in cross((qx, qy, qz), v))
    u = cross((qx, qy, qz), t)
    return tuple(v[i] + w * t[i] + u[i] for i in range(3))


_FROM_SKEL_KEYS = ("name", "bones", "unitScale", "heightM", "leafLength", "collection", "location")


def op_create_armature_from_skeleton(params):
    """Build an armature from an engine skeleton's bone list - names, hierarchy and rest positions - in one call.

    bones is list_bones' answer from Unreal with include_transforms (the `bones` array: name, parent, refPose {location,
    rotation {x: pitch, y: yaw, z: roll}, scale}, PARENT-RELATIVE, centimetres). Each bone's component-space transform is
    composed down the hierarchy the way FTransform does (child local x parent: the rotation FRotator::Quaternion, the
    location the parent's rotation of the parent's scale times the child's location, plus the parent's), then mapped to
    Blender: (x, -y, z) x unitScale (0.01: centimetres to metres) - the axes Blender's FBX exporter and the engine's
    importer agree on. A bone's head is that position; its tail is its only child's head, or (no child, or several) its
    head plus its own forward axis (the engine's X) at leafLength (default half its parent's length, at least 1 cm).
    heightM scales the whole rig so its heads span that height. The armature is made by create_armature, named `name`
    (default 'root', the engine's convention). Roll is not carried: Blender rebuilds a bone's roll from its own axis.
    Reply: the bone count, the root, the scale applied and the heads' height.
    """
    reject_unknown(params, _FROM_SKEL_KEYS, "create_armature_from_skeleton")
    raw = params.get("bones")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise MifOpError("bones is list_bones' bones array (with include_transforms). NOTHING was created.")
    scale = take_float(params, "unitScale", default=0.01)
    height = take_float(params, "heightM", default=None)
    if not scale > 0.0 or (height is not None and not height > 0.0):
        raise MifOpError("unitScale and heightM must be above 0. NOTHING was created.")
    leaf = take_float(params, "leafLength", default=None)
    rows = {}
    order = []
    for i, b in enumerate(raw):
        if not isinstance(b, dict) or not isinstance(b.get("name"), str) or not isinstance(b.get("refPose"), dict):
            raise MifOpError("bones[%d] needs name and refPose (list_bones with include_transforms). NOTHING was "
                             "created." % i)
        rp = b["refPose"]
        try:
            loc = [finite_float(rp["location"][k], "location") for k in "xyz"]
            rot = [finite_float(rp["rotation"][k], "rotation") for k in "xyz"]
            sc = [finite_float((rp.get("scale") or {"x": 1, "y": 1, "z": 1})[k], "scale") for k in "xyz"]
        except (KeyError, TypeError):
            raise MifOpError("bones[%d] refPose needs location, rotation and scale as {x, y, z}. NOTHING was created."
                             % i)
        if b["name"] in rows:
            raise MifOpError("bone '%s' appears twice. NOTHING was created." % b["name"])
        rows[b["name"]] = (b.get("parent") or None, loc, rot, sc)
        order.append(b["name"])
    roots = [n for n in order if rows[n][0] is None]
    if len(roots) != 1:
        raise MifOpError("the list has %d root bones (no parent); a skeleton has one. NOTHING was created." % len(roots))
    unknown = sorted({p for p, *_ in rows.values() if p is not None and p not in rows})
    if unknown:
        raise MifOpError("parent bone(s) %s are not in the list - pass the whole skeleton. NOTHING was created."
                         % ", ".join(unknown[:10]))
    comp = {}

    def solve(n, depth=0):
        if n in comp:
            return comp[n]
        if depth > len(rows):
            raise MifOpError("the hierarchy loops at '%s'. NOTHING was created." % n)
        parent, loc, rot, sc = rows[n]
        q = _ue_quat(rot[0], rot[1], rot[2])
        if parent is None:
            comp[n] = (q, tuple(loc), tuple(sc))
        else:
            pq, pl, ps = solve(parent, depth + 1)
            moved = _qrot(pq, tuple(loc[k] * ps[k] for k in range(3)))
            comp[n] = (_qmul(pq, q), tuple(moved[k] + pl[k] for k in range(3)), tuple(sc[k] * ps[k] for k in range(3)))
        return comp[n]
    for n in order:
        solve(n)

    def to_bl(v):
        return [v[0] * scale, -v[1] * scale, v[2] * scale]
    heads = {n: to_bl(comp[n][1]) for n in order}
    zs = [h[2] for h in heads.values()]
    span = max(zs) - min(zs)
    k = 1.0
    if height is not None:
        if span <= 1e-9:
            raise MifOpError("the bones' heads have no height to scale. NOTHING was created.")
        k = height / span
        heads = {n: [c * k for c in h] for n, h in heads.items()}
    children = {n: [c for c in order if rows[c][0] == n] for n in order}
    bones_out = []
    for n in order:
        h = heads[n]
        kids = children[n]
        if len(kids) == 1 and sum((heads[kids[0]][i] - h[i]) ** 2 for i in range(3)) > 1e-12:
            t = heads[kids[0]]
        else:
            parent = rows[n][0]
            plen = (math.dist(heads[parent], h) / 2.0) if parent is not None else 0.0
            length = leaf if leaf is not None else max(plen, 0.01)
            fwd = _qrot(comp[n][0], (1.0, 0.0, 0.0))
            t = [h[0] + fwd[0] * length, h[1] - fwd[1] * length, h[2] + fwd[2] * length]
        row = {"name": n, "head": h, "tail": t}
        if rows[n][0] is not None:
            row["parent"] = rows[n][0]
        bones_out.append(row)
    from .ops_create import op_create_armature   # noqa: PLC0415 - one way to make an armature
    made_params = {"name": take(params, "name", default="root", kind=str), "bones": bones_out}
    for key in ("collection", "location"):
        if params.get(key) is not None:
            made_params[key] = params[key]
    made = op_create_armature(made_params)
    zsn = [hh[2] for hh in heads.values()]
    made.update({"boneCount": len(order), "root": roots[0], "heightScale": round(k, 6),
                 "headsHeight": round(max(zsn) - min(zsn), 6), "unitScale": scale,
                 "fromSkeletonNote": "heads from the engine's parent-relative reference pose, composed and mapped "
                                     "(x, -y, z) x unitScale; roll is not carried."})
    return made


_FIT_KEYS = ("object", "name", "armature", "landmarks", "symmetry", "pair")


def op_fit_armature(params):
    """Move an armature's bones onto a character from landmarks, keeping the names and the hierarchy.

    landmarks {bone name: [x, y, z]} (the armature's own space) puts each named bone's HEAD there. A bone not named moves
    with its parent: rigidly, by the parent's head move, so a hand's fingers follow the wrist. Every bone's tail goes to
    its only child's head when that child was moved, and otherwise keeps its offset from its own head. symmetry 'X'
    fills each missing partner of a named bone ('_l' and '_r', or `pair`) with the landmark mirrored across X.
    A mesh skinned to the armature is named in a warning (its Armature modifier deforms from the old rest). Reply: each
    bone's head displacement, and the length of every bone before and after.
    """
    reject_unknown(params, _FIT_KEYS, "fit_armature")
    obj = get_object(take(params, "object", "name", "armature", required=True, kind=str))
    if obj.type != "ARMATURE":
        raise MifOpError("'%s' is %s, not an ARMATURE. NOTHING was changed." % (obj.name, obj.type))
    marks = params.get("landmarks")
    if not isinstance(marks, dict) or not marks:
        raise MifOpError("landmarks is {bone name: [x, y, z]}. NOTHING was changed.")
    data = obj.data
    names = [b.name for b in data.bones]
    known = set(names)
    pts = {}
    for n, p in marks.items():
        if n not in known:
            raise MifOpError("'%s' has no bone '%s'. NOTHING was changed." % (obj.name, n))
        if not isinstance(p, (list, tuple)) or len(p) != 3:
            raise MifOpError("landmark '%s' must be [x, y, z]. NOTHING was changed." % n)
        pts[n] = Vector(finite_floats(list(p), "landmarks.%s" % n))
    sym = take(params, "symmetry", default=None, kind=str)
    pair = params.get("pair", ["_l", "_r"])
    mirrored = []
    if sym is not None:
        if str(sym).upper() != "X":
            raise MifOpError("symmetry is 'X' (mirror across the armature's X). NOTHING was changed.")
        for n in list(pts):
            other = _side_partner(n, pair)
            if other and other in known and other not in pts:
                pts[other] = Vector((-pts[n].x, pts[n].y, pts[n].z))
                mirrored.append(other)
    old = {b.name: (b.head_local.copy(), b.tail_local.copy()) for b in data.bones}
    new_head = {}
    for b in data.bones:          # parents come before children in data.bones
        if b.name in pts:
            new_head[b.name] = pts[b.name]
        elif b.parent is not None:
            new_head[b.name] = old[b.name][0] + (new_head[b.parent.name] - old[b.parent.name][0])
        else:
            new_head[b.name] = old[b.name][0].copy()
    new_tail = {}
    for b in data.bones:
        kids = [c for c in b.children]
        # A CONNECTED CHILD'S HEAD IS ITS PARENT'S TAIL - Blender enforces it in edit mode - so a bone with one takes
        # that head as its tail; a landmark on the child that the old offset ignored was moved back by edit mode
        # after the write (measured 2026-10-04 on 5.2.1) and set_bone_rest's read-back refused a half-written rest.
        tied = sorted({tuple(round(x, 9) for x in new_head[c.name]) for c in kids if c.use_connect})
        if len(tied) > 1:
            raise MifOpError("the connected children of '%s' would need different tails (%s). NOTHING was changed."
                             % (b.name, ", ".join(c.name for c in kids if c.use_connect)))
        if tied:
            new_tail[b.name] = Vector(tied[0])
        elif len(kids) == 1 and kids[0].name in pts and (new_head[kids[0].name] - new_head[b.name]).length > 1e-9:
            new_tail[b.name] = new_head[kids[0].name].copy()
        else:
            new_tail[b.name] = new_head[b.name] + (old[b.name][1] - old[b.name][0])
    bad = [n for n in names if (new_tail[n] - new_head[n]).length < 1e-9]
    if bad:
        raise MifOpError("the landmarks would give bone(s) %s zero length. NOTHING was changed." % ", ".join(bad[:10]))
    plan = [{"name": n, "head": list(new_head[n]), "tail": list(new_tail[n])} for n in names]
    out = op_set_bone_rest({"object": obj.name, "bones": plan})
    out.update({"landmarksUsed": sorted(marks), "mirroredLandmarks": sorted(mirrored),
                "displacement": {n: round((new_head[n] - old[n][0]).length, 6) for n in names},
                "lengthsBefore": {n: round((old[n][1] - old[n][0]).length, 6) for n in names},
                "lengthsAfter": {n: round((new_tail[n] - new_head[n]).length, 6) for n in names}})
    return out


_DEFORM_KEYS = ("rig", "object", "name", "disconnect", "actions", "clearConstraints", "suffix", "allowSimulation")


def _deform_motion_error(rig, arm, act, baked_act, frames):
    """Largest world-space bone head distance and rotation angle between `arm` playing `baked_act` with its constraints
    muted and `rig` playing `act`, over `frames`. The constraints are unmuted and the frame restored afterwards."""
    sc = bpy.context.scene
    keep = sc.frame_current
    muted = [con for pb in arm.pose.bones for con in pb.constraints if not con.mute]
    pos = rot = 0.0
    try:
        for con in muted:
            con.mute = True
        rig.animation_data.action = act
        arm.animation_data.action = baked_act
        for f in frames:
            sc.frame_set(f)
            for pb in arm.pose.bones:
                ma = arm.matrix_world @ pb.matrix
                mr = rig.matrix_world @ rig.pose.bones[pb.name].matrix
                pos = max(pos, (ma.translation - mr.translation).length)
                ang = ma.to_quaternion().rotation_difference(mr.to_quaternion()).angle
                rot = max(rot, min(ang, 2.0 * math.pi - ang))
    finally:
        for con in muted:
            con.mute = False
        sc.frame_set(keep)
    return pos, rot


def op_extract_deform_skeleton(params):
    """A deform-only skeleton built from a control rig, with the rig's actions baked onto it - what a game engine keeps.

    An engine keeps only deform bones and baked keys, so a control rig (IK targets, poles, mechanism bones) is exported
    through a flat copy of its deform bones. This makes that copy as Blender's own steps would: a new armature `name`
    with every deform bone of `rig` at its rest (each parented to its nearest DEFORM ancestor; disconnect true leaves
    every bone unconnected, for squash and stretch), a Copy Transforms constraint on each bone to the rig's bone of the
    same name, then for each of `actions` (default: the rig's current action) the action put on the rig and the copy
    baked over that action's frame range by bake_to_keyframes (visual keying). Per action the copy then plays its baked
    keys with every constraint muted against the rig at the first, middle and last frame, and the largest bone position
    and rotation error comes back with motionPreserved. Each baked action is named <action><suffix> (suffix '_deform') and kept with a fake user. clearConstraints
    (default true) removes the Copy Transforms afterwards. The rig gets its own action back.
    """
    reject_unknown(params, _DEFORM_KEYS, "extract_deform_skeleton")
    rig = get_object(take(params, "rig", "object", required=True, kind=str))
    if rig.type != "ARMATURE":
        raise MifOpError("'%s' is %s, not an ARMATURE. NOTHING was created." % (rig.name, rig.type))
    name = take(params, "name", default=rig.name + "_deform", kind=str)
    if name in {o.name for o in bpy.data.objects}:
        raise MifOpError("an object named '%s' exists. NOTHING was created." % name)
    disconnect = take_bool(params, "disconnect", default=False)
    clear = take_bool(params, "clearConstraints", default=True)
    suffix = take(params, "suffix", default="_deform", kind=str)
    deform = [b for b in rig.data.bones if b.use_deform]
    if not deform:
        raise MifOpError("'%s' has no deform bone. NOTHING was created." % rig.name)
    raw = params.get("actions")
    ad = rig.animation_data
    current = ad.action if ad is not None else None
    if raw is None:
        acts = [current] if current is not None else []
    else:
        if not isinstance(raw, (list, tuple)) or any(not isinstance(a, str) for a in raw):
            raise MifOpError("actions must be a list of action names. NOTHING was created.")
        missing = [a for a in raw if bpy.data.actions.get(a) is None]
        if missing:
            raise MifOpError("no action named %s. NOTHING was created." % ", ".join(missing))
        acts = [bpy.data.actions[a] for a in raw]
    taken = {a.name for a in bpy.data.actions}
    clash = [a.name + suffix for a in acts if a.name + suffix in taken]
    if clash:
        raise MifOpError("action(s) %s exist already. NOTHING was created." % ", ".join(clash))
    # EACH ACTION IS BAKED OVER ITS RANGE, which steps every live simulation (ops_common.simulation_guard) - asked
    # here, before the copy is made, and the answer handed to each bake.
    allow_sim = take_bool(params, "allowSimulation", default=False)
    visits = [int(round(x)) for a in acts for x in a.frame_range]
    sim_reply = simulation_guard(params, visits, "extract_deform_skeleton", tail="NOTHING was created.")
    names = {b.name for b in deform}
    bones = []
    for b in deform:
        p = b.parent
        while p is not None and p.name not in names:
            p = p.parent
        row = {"name": b.name, "head": list(b.head_local), "tail": list(b.tail_local)}
        if p is not None:
            row["parent"] = p.name
            # A NAME COMPARE, not `is`: bpy hands out a fresh wrapper per access, so `b.parent is p` was never true
            # and every connected bone came out unconnected (measured 2026-10-04 on 5.2.1).
            if not disconnect and b.use_connect and b.parent is not None and b.parent.name == p.name:
                row["connect"] = True
        bones.append(row)
    from .ops_create import op_create_armature   # noqa: PLC0415 - one way to make an armature
    from .ops_anim import op_bake_to_keyframes   # noqa: PLC0415 - one way to bake
    made = op_create_armature({"name": name, "bones": bones})
    arm = bpy.data.objects[made["name"]]
    arm.matrix_world = rig.matrix_world.copy()
    for pb in arm.pose.bones:
        con = pb.constraints.new("COPY_TRANSFORMS")
        con.target = rig
        con.subtarget = pb.name
    baked = []
    try:
        for act in acts:
            if rig.animation_data is None:
                rig.animation_data_create()
            rig.animation_data.action = act
            f0, f1 = (int(round(x)) for x in act.frame_range)
            if arm.animation_data is not None:
                arm.animation_data.action = None
            op_bake_to_keyframes({"object": arm.name, "frameStart": f0, "frameEnd": f1, "allowSimulation": allow_sim})
            new_act = arm.animation_data.action if arm.animation_data else None
            if new_act is None:
                raise MifOpError("baking '%s' left '%s' with no action." % (act.name, arm.name))
            new_act.name = act.name + suffix
            new_act.use_fake_user = True
            baked.append((act, new_act, f0, f1))
        if clear:
            for pb in arm.pose.bones:
                for con in list(pb.constraints):
                    pb.constraints.remove(con)
        # THE PROOF IS TAKEN WITH THE CONSTRAINTS OFF. bake_to_keyframes' own motion check compares the copy before and
        # after its bake, and the Copy Transforms still drive the copy at both ends, so it reads 0 whatever the keys
        # hold. Here each baked action plays on the copy with every constraint muted, against its source on the rig.
        report = []
        for act, new_act, f0, f1 in baked:
            pos, rot = _deform_motion_error(rig, arm, act, new_act, sorted({f0, (f0 + f1) // 2, f1}))
            report.append({"action": new_act.name, "from": act.name, "frames": [f0, f1],
                           "motionPreserved": pos < 1e-4 and rot < 1e-4, "maxBonePositionError": round(pos, 7),
                           "maxBoneRotationErrorRadians": round(rot, 7)})
    finally:
        if rig.animation_data is not None:
            rig.animation_data.action = current
    baked = report
    return dict({"object": arm.name, "rig": rig.name, "deformBones": len(bones), "bonesOfRig": len(rig.data.bones),
                 "disconnected": disconnect, "actions": baked,
                 "constraintsLeft": sum(len(pb.constraints) for pb in arm.pose.bones)}, **sim_reply)


_CONTROL_KEYS = ("object", "name", "armature", "preset", "chains", "switchProperty", "switchDirection", "fkPrefix",
                 "ikPrefix", "targetParent")
# THE ENGINE'S LIMB NAMES (Manny, Quinn and a MetaHuman body share them): each limb is three bones, upper - lower - end.
_LIMB_PRESET = (("arm_l", ("upperarm_l", "lowerarm_l", "hand_l"), "FK"),
                ("arm_r", ("upperarm_r", "lowerarm_r", "hand_r"), "FK"),
                ("leg_l", ("thigh_l", "calf_l", "foot_l"), "IK"),
                ("leg_r", ("thigh_r", "calf_r", "foot_r"), "IK"))


def _signed_angle(u, v, normal):
    """The IK pole angle's sign convention: NEGATIVE when u x v points along the bone. The opposite sign put a leg's
    IK knee 0.1 behind the deform knee at rest (measured 2026-10-04 on 5.2.1) - the chain bent backwards."""
    a = u.angle(v) if u.length and v.length else 0.0
    return -a if u.cross(v).dot(normal) > 0 else a


def _control_chains(data, params):
    """[(chain name, (upper, lower, end), default side)] - from chains [...] or the preset - every bone checked."""
    raw = params.get("chains")
    preset = params.get("preset")
    if (raw is None) == (preset is None):
        raise MifOpError("pass preset LIMBS_FK_IK (the engine's limb names) OR chains [{name, bones [upper, lower, end], "
                         "default FK|IK}]. NOTHING was changed.")
    if preset is not None:
        if str(preset).upper() != "LIMBS_FK_IK":
            raise MifOpError("preset must be LIMBS_FK_IK, got %r. NOTHING was changed." % preset)
        have = {b.name for b in data.bones}
        found = [(n, b, d) for n, b, d in _LIMB_PRESET if all(x in have for x in b)]
        if not found:
            raise MifOpError("no limb of the engine's naming (upperarm_l/lowerarm_l/hand_l, thigh_l/calf_l/foot_l and "
                             "the _r side) is complete on this armature - pass chains. NOTHING was changed.")
        return found
    if not isinstance(raw, (list, tuple)) or not raw:
        raise MifOpError("chains is a list of {name, bones [upper, lower, end], default}. NOTHING was changed.")
    out = []
    have = {b.name for b in data.bones}
    for i, ch in enumerate(raw):
        if not isinstance(ch, dict) or set(ch) - {"name", "bones", "default"} or not isinstance(ch.get("name"), str):
            raise MifOpError("chains[%d] is {name, bones, default}; got %r. NOTHING was changed." % (i, ch))
        bones = ch.get("bones")
        if not isinstance(bones, (list, tuple)) or len(bones) != 3 or any(b not in have for b in bones):
            raise MifOpError("chains[%d].bones must be three bones of the armature, upper - lower - end; got %r. NOTHING "
                             "was changed." % (i, bones))
        if data.bones[bones[1]].parent is None or data.bones[bones[1]].parent.name != bones[0] or \
                data.bones[bones[2]].parent is None or data.bones[bones[2]].parent.name != bones[1]:
            raise MifOpError("chains[%d]: %s must each be the parent of the next. NOTHING was changed."
                             % (i, " - ".join(bones)))
        d = str(ch.get("default") or "FK").upper()
        if d not in ("FK", "IK"):
            raise MifOpError("chains[%d].default must be FK or IK. NOTHING was changed." % i)
        out.append((ch["name"], tuple(bones), d))
    names = [c[0] for c in out]
    if len(set(names)) != len(names):
        raise MifOpError("two chains share a name. NOTHING was changed.")
    return out


def op_add_control_rig(params):
    """FK and IK control bones over an existing deform skeleton, its names and hierarchy untouched.

    For each limb (preset LIMBS_FK_IK: the engine's upperarm/lowerarm/hand and thigh/calf/foot, both sides; or chains
    [{name, bones [upper, lower, end], default FK|IK}]) this adds an FK copy of the three bones (fkPrefix 'FK_'), an IK
    copy ('IK_') with an IK constraint on its lower bone (chain 2) aimed at an IK target (a copy of the end bone,
    parented to targetParent - default the bone named 'root' when there is one) with a pole bone in front of the middle
    joint, the pole angle computed from the rest pose so the IK chain lies on the deform chain. The controls do not
    deform. Each deform bone gets two Copy Transforms - from its FK copy, then its IK copy - and the IK one's influence
    is driven by an object property switchProperty_<chain> ('ik_fk_arm_l'): switchDirection IK_AT_ZERO (default: 0 = IK,
    1 = FK) or FK_AT_ZERO. Each chain starts on its default (arms FK, legs IK). A deform bone that already has a
    constraint is refused. Reply: the controls made, the constraints on each deform bone, each switch's data path and
    value, and restError - how far the IK chain sits from the deform chain at rest (with the switch on IK).
    """
    reject_unknown(params, _CONTROL_KEYS, "add_control_rig")
    obj = get_object(take(params, "object", "name", "armature", required=True, kind=str))
    if obj.type != "ARMATURE":
        raise MifOpError("'%s' is %s, not an ARMATURE. NOTHING was changed." % (obj.name, obj.type))
    require_editable(obj, "add controls to")
    data = obj.data
    chains = _control_chains(data, params)
    fk_p = take(params, "fkPrefix", default="FK_", kind=str)
    ik_p = take(params, "ikPrefix", default="IK_", kind=str)
    switch = take(params, "switchProperty", default="ik_fk", kind=str)
    direction = str(take(params, "switchDirection", default="IK_AT_ZERO", kind=str)).upper()
    if direction not in ("IK_AT_ZERO", "FK_AT_ZERO"):
        raise MifOpError("switchDirection must be IK_AT_ZERO or FK_AT_ZERO. NOTHING was changed.")
    if not fk_p or not ik_p or fk_p == ik_p:
        raise MifOpError("fkPrefix and ikPrefix must be two different, non-empty prefixes. NOTHING was changed.")
    tp = params.get("targetParent")
    if tp is None and "root" in data.bones:
        tp = "root"
    if tp is not None and tp not in data.bones:
        raise MifOpError("targetParent %r is not a bone of '%s'. NOTHING was changed." % (tp, obj.name))
    all_deform = [b for _n, bs, _d in chains for b in bs]
    busy = [b for b in all_deform if len(obj.pose.bones[b].constraints)]
    if busy:
        raise MifOpError("deform bone(s) %s already have constraints - two control rigs would fight. NOTHING was "
                         "changed." % ", ".join(busy))
    new_names = [p + b for _n, bs, _d in chains for b in bs for p in (fk_p, ik_p)]
    new_names += ["%s%s_target" % (ik_p, n) for n, _b, _d in chains] + ["%s%s_pole" % (ik_p, n) for n, _b, _d in chains]
    have = {b.name for b in data.bones}
    clash = [n for n in new_names if n in have]
    if clash:
        raise MifOpError("bone(s) %s exist already. NOTHING was changed." % ", ".join(clash[:10]))
    if any(("%s_%s" % (switch, n)) in obj.keys() for n, _b, _d in chains):
        raise MifOpError("'%s' already has a %s_<chain> property. NOTHING was changed." % (obj.name, switch))

    snap = selection_snapshot()
    poles = {}
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        ebs = data.edit_bones
        for name, bones, _d in chains:
            for pref in (fk_p, ik_p):
                prev = None
                for b in bones:
                    src = ebs[b]
                    nb = ebs.new(pref + b)
                    nb.head, nb.tail, nb.roll = src.head.copy(), src.tail.copy(), src.roll
                    nb.use_deform = False
                    nb.parent = prev if prev is not None else src.parent
                    nb.use_connect = src.use_connect if prev is not None else False
                    prev = nb
            up, lo, end = (ebs[b] for b in bones)
            tgt = ebs.new("%s%s_target" % (ik_p, name))
            tgt.head, tgt.tail, tgt.roll = end.head.copy(), end.tail.copy(), end.roll
            tgt.use_deform = False
            tgt.parent = ebs[tp] if tp is not None else None
            mid = (up.head + end.head) * 0.5
            out_dir = lo.head - mid
            length = (up.tail - up.head).length
            if out_dir.length < 1e-6 * max(length, 1e-6):
                out_dir = Vector((0.0, -1.0, 0.0))
            pole_at = lo.head + out_dir.normalized() * length
            pole = ebs.new("%s%s_pole" % (ik_p, name))
            pole.head, pole.tail = pole_at, pole_at + Vector((0.0, 0.0, 0.1 * max(length, 0.01)))
            pole.use_deform = False
            pole.parent = ebs[tp] if tp is not None else None
            poles[name] = pole_at.copy()
    finally:
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass
        selection_restore(snap)

    from .ops_anim import op_add_driver   # noqa: PLC0415 - one way to make a driver
    pbs = obj.pose.bones
    switches, cons = {}, {}
    for name, bones, default in chains:
        up, lo, end = bones
        ik_lo = pbs[ik_p + lo]
        ik = ik_lo.constraints.new("IK")
        ik.name = "Mif IK"
        ik.target, ik.subtarget = obj, "%s%s_target" % (ik_p, name)
        ik.pole_target, ik.pole_subtarget = obj, "%s%s_pole" % (ik_p, name)
        ik.chain_count = 2
        base, lower = data.bones[ik_p + up], data.bones[ik_p + lo]
        pole_normal = (lower.tail_local - base.head_local).cross(poles[name] - base.head_local)
        projected = pole_normal.cross(base.tail_local - base.head_local)
        x_axis = base.matrix_local.to_3x3().col[0]
        ik.pole_angle = _signed_angle(x_axis, projected, base.tail_local - base.head_local)
        rot = pbs[ik_p + end].constraints.new("COPY_ROTATION")
        rot.name = "Mif IK end"
        rot.target, rot.subtarget = obj, "%s%s_target" % (ik_p, name)
        prop = "%s_%s" % (switch, name)
        ik_value, fk_value = (0.0, 1.0) if direction == "IK_AT_ZERO" else (1.0, 0.0)
        obj[prop] = ik_value if default == "IK" else fk_value
        switches[name] = {"property": prop, "dataPath": '["%s"]' % prop, "value": obj[prop], "default": default}
        for b in bones:
            pb = pbs[b]
            c_fk = pb.constraints.new("COPY_TRANSFORMS")
            c_fk.name = "Mif FK"
            c_fk.target, c_fk.subtarget = obj, fk_p + b
            c_ik = pb.constraints.new("COPY_TRANSFORMS")
            c_ik.name = "Mif IK"
            c_ik.target, c_ik.subtarget = obj, ik_p + b
            op_add_driver({"object": obj.name, "dataPath": 'pose.bones["%s"].constraints["Mif IK"].influence' % b,
                           "expression": "1 - var" if direction == "IK_AT_ZERO" else "var",
                           "variables": [{"name": "var", "object": obj.name, "dataPath": '["%s"]' % prop}]})
            cons[b] = [c.name for c in pb.constraints]
    # THE REST CHECK: every chain on IK, how far each IK control sits from its deform bone at rest.
    saved = {s["property"]: obj[s["property"]] for s in switches.values()}
    for s in switches.values():
        obj[s["property"]] = 0.0 if direction == "IK_AT_ZERO" else 1.0
    obj.update_tag()
    bpy.context.view_layer.update()
    errs = {}
    for cname, bones, _d in chains:
        e = 0.0
        for b in bones:
            e = max(e, (pbs[ik_p + b].head - data.bones[b].head_local).length,
                    (pbs[ik_p + b].tail - data.bones[b].tail_local).length)
        errs[cname] = round(e, 6)
    err = max(errs.values())
    for k, v in saved.items():
        obj[k] = v
    obj.update_tag()
    bpy.context.view_layer.update()
    return {"object": obj.name, "chains": [c[0] for c in chains], "controlsMade": len(new_names),
            "controls": new_names, "constraints": cons, "switches": switches, "switchDirection": direction,
            "restError": round(err, 6), "restErrorByChain": errs}


OPS = {
    "set_bone_properties": op_set_bone_properties,
    "create_armature_from_skeleton": op_create_armature_from_skeleton,
    "fit_armature": op_fit_armature,
    "extract_deform_skeleton": op_extract_deform_skeleton,
    "add_control_rig": op_add_control_rig,
    "set_bone_pose": op_set_bone_pose,
    "set_shape_key": op_set_shape_key,
    "apply_shape_keys": op_apply_shape_keys,
    "mirror_shape_key": op_mirror_shape_key,
    "list_bones": op_list_bones,
    "list_shape_keys": op_list_shape_keys,
    "list_vertex_groups": op_list_vertex_groups,
    "list_modifiers": op_list_modifiers,
    "normalize_weights": op_normalize_weights,
    "paint_weights": op_paint_weights,
    "vertex_groups_from_bones": op_vertex_groups_from_bones,
    "bind_modifier": op_bind_modifier,
    "set_bone_rest": op_set_bone_rest,
    "set_vertex_weights": op_set_vertex_weights,
    "add_shape_key": op_add_shape_key,
    "set_point_positions": op_set_point_positions,
    "add_bones": op_add_bones,
    "transfer_weights": op_transfer_weights,
    "add_modifier": op_add_modifier,
    "remove_modifier": op_remove_modifier,
    "set_modifier": op_set_modifier,
    "apply_modifier": op_apply_modifier,
    "rename_bones": op_rename_bones,
}
