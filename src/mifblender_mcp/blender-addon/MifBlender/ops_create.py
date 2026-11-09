"""Creating geometry, and placing it without baking.

WHAT WAS MISSING. Every mesh in this addon entered through import_mesh. The bridge could decimate,
bevel, unwrap, skirt and export - and could not originate a single vertex. That is the right shape
for a round trip and the wrong one for building an asset IN Blender, which is what was asked for
on 2026-08-30.

=============================================================================
EVERY PRIMITIVE TAKES A DIFFERENT SIZING PARAMETER, AND THEY ARE NOT INTERCHANGEABLE
=============================================================================
Read off the real bpy.ops RNA rather than assumed, and it is not uniform:

    cube / plane / grid / monkey        size
    sphere / icosphere / cylinder /
      circle                            radius        (a radius is HALF a size, not a synonym)
    cone                                radius1, radius2, depth   - and NO `radius` at all
    torus                               major_radius, minor_radius - and NO size, NO radius,
                                        and NO enter_editmode either

The first version of this file remapped `size` onto `radius` for everything outside the size group.
That is a factor-of-two error on four kinds, a TypeError on cone, and a TypeError on torus - and it
also means a blanket enter_editmode=False would have broken torus on its own. So the table below
records what each primitive ACTUALLY accepts, size and radius are never silently swapped, and asking
for the wrong one is refused with the reason rather than reinterpreted.

=============================================================================
TWO THINGS THE bpy.ops PRIMITIVE CALLS DO THAT HAVE TO BE HANDLED
=============================================================================

1. THEY NAME THE OBJECT THEMSELVES, AND RENAME ON COLLISION. primitive_cube_add makes "Cube"; the
   second is "Cube.001". Blender never fails and never overwrites - it appends a number. So a caller
   who asked for "Crate" can get "Crate.003", and every op here echoes the name the object ACTUALLY
   has.

2. THEY OPERATE ON CONTEXT - the active collection and the current mode - so they depend on state a
   previous op left behind. Each op takes a selection snapshot and restores it on every path, and
   create_primitive refuses outright unless Blender is in OBJECT mode, because in EDIT mode
   primitive_*_add WELDS its geometry into the mesh being edited instead of creating an object.

WHY transform_object IS HERE AND NOT IN ops_mesh. apply_transform and set_origin both BAKE: they
write the transform into the mesh data and leave the object at identity. That is what an export
pipeline wants and is NOT how you place a second object next to a first.
"""
import math
import os
import re

import bmesh
import bpy

from .ops_common import (remove_objects, MifOpError, assign_id_name, check_axis_dict, finite_float, finite_floats,
                         finite_int, get_object, mesh_counts, mesh_shape, object_info,
                         reject_unknown, require_editable, rnd, selection_restore,
                         selection_snapshot, select_only, take, take_bool, take_float, take_int, drawn_instances,
                         rotation_mode_note,
                         write_rotation, _open_collapsed_uvs, open_collapsed_uvs_of, project_flat_uvs, triangulate_ngons)

# What each primitive really accepts. `size` is that operator's own size-like kwarg, or None when it
# has none at all. `extras` is every other per-kind kwarg. Verified against bpy.ops RNA.
PRIMITIVES = {
    "cube":      {"fn": "primitive_cube_add",       "size": "size",   "extras": ()},
    "plane":     {"fn": "primitive_plane_add",      "size": "size",   "extras": ()},
    "grid":      {"fn": "primitive_grid_add",       "size": "size",
                  "extras": ("x_subdivisions", "y_subdivisions")},
    "monkey":    {"fn": "primitive_monkey_add",     "size": "size",   "extras": ()},
    "sphere":    {"fn": "primitive_uv_sphere_add",  "size": "radius",
                  "extras": ("segments", "ring_count")},
    "uvsphere":  {"fn": "primitive_uv_sphere_add",  "size": "radius",
                  "extras": ("segments", "ring_count")},
    "icosphere": {"fn": "primitive_ico_sphere_add", "size": "radius",
                  "extras": ("subdivisions",)},
    # capKw: fillType's operator property on a closed primitive. Blender's own default, an n-gon on each end, fails
    # gameReady on the plainest cylinder there is, and a fan shades the same, so a fan is the default here.
    "cylinder":  {"fn": "primitive_cylinder_add",   "size": "radius", "capKw": "end_fill_type",
                  "extras": ("vertices", "depth", "fill_type")},
    "circle":    {"fn": "primitive_circle_add",     "size": "radius",
                  "extras": ("vertices", "fill_type")},
    "cone":      {"fn": "primitive_cone_add",       "size": None, "capKw": "end_fill_type",
                  "extras": ("vertices", "depth", "radius1", "radius2", "fill_type")},
    # NO enter_editmode on this operator - passing one is a TypeError, hence the opt-out flag.
    "torus":     {"fn": "primitive_torus_add",      "size": None, "noEnterEditmode": True,
                  "extras": ("major_radius", "minor_radius", "major_segments", "minor_segments")},
}

# Accepted request key -> the Blender kwarg it maps to. Only these are ever forwarded.
EXTRA_KEYS = {
    "segments": "segments", "ringCount": "ring_count", "subdivisions": "subdivisions",
    "vertices": "vertices", "depth": "depth", "radius1": "radius1", "radius2": "radius2",
    "xSubdivisions": "x_subdivisions", "ySubdivisions": "y_subdivisions",
    "fillType": "fill_type", "majorRadius": "major_radius", "minorRadius": "minor_radius",
    # A DONUT'S SEGMENTS (2026-10-05): primitive_torus_add's own major_segments / minor_segments (48 x 12 by default;
    # the donut case wanted 48 x 18) - the other torus kwargs were reachable and these two were not.
    "majorSegments": "major_segments", "minorSegments": "minor_segments",
}
INT_KWARGS = ("segments", "ring_count", "subdivisions", "vertices",
              "x_subdivisions", "y_subdivisions", "major_segments", "minor_segments")
ALIGN_VALUES = ("WORLD", "VIEW", "CURSOR")

# MODULE-LEVEL AND A PLAIN LITERAL on purpose. parity_check.py resolves an accepted-key expression
# statically and REFUSES to guess at one it cannot - a parameter list it cannot read is exactly
# where a docstring and the code drift apart unnoticed.
CREATE_PRIMITIVE_PARAMS = (
    "kind", "type", "name", "size", "radius", "location", "rotation", "align",
    "segments", "ringCount", "subdivisions", "vertices", "depth",
    "radius1", "radius2", "xSubdivisions", "ySubdivisions", "fillType",
    "majorRadius", "minorRadius", "majorSegments", "minorSegments",
)


def _vec3(params, key, default):
    val = params.get(key)
    if val is None:
        return list(default)
    if isinstance(val, dict):
        # AT LEAST ONE OF x/y/z, AND NOTHING ELSE. A dict was read with .get(..., default) for
        # each axis, so {"mif":"typo"} returned the DEFAULT vector and the call reported
        # success - a misspelled key silently placed the object at the origin, or left it where
        # it was, and every field in the response agreed. Partial dicts stay legal ({"z": 2} is
        # a useful thing to write); a dict that names none of them is a typo, not a request.
        check_axis_dict(val, key, ("x", "y", "z"))
        return finite_floats([val.get("x", 0.0), val.get("y", 0.0), val.get("z", 0.0)], key)
    if hasattr(val, "__len__") and not isinstance(val, str):
        if len(val) != 3:
            raise MifOpError("'%s' takes [x,y,z] or {x,y,z}, got %d component(s)" % (key, len(val)))
        return finite_floats(val, key)
    raise MifOpError("'%s' takes [x,y,z] or {x,y,z}, got %r" % (key, val))


def _require_object_mode(op):
    """EDIT mode makes several of these operators destructive rather than additive."""
    mode = getattr(bpy.context, "mode", "OBJECT")
    if mode != "OBJECT":
        raise MifOpError(
            "Blender is in %s mode and %s cannot run there. bpy.ops.mesh.primitive_*_add WELDS its "
            "geometry into the mesh being edited instead of creating an object, which changes that "
            "mesh in a way this endpoint cannot undo. Leave edit mode first. NOTHING was changed."
            % (mode, op))


def op_create_primitive(params):
    """Create a primitive mesh object: cube, sphere, icosphere, cylinder, cone, torus, plane, grid,
    circle or monkey.

    THE FOUNDATIONAL GAP - without this the addon can only edit meshes that came from a file.

    OUTPUT THAT PASSES mesh_quality gameReady (2026-11-01): a cylinder's and a cone's ends are triangle fans by default,
    where Blender's operator leaves an n-gon; fillType NGON, TRIFAN or NOTHING picks another. A monkey's collapsed UV
    faces are opened (clean_mesh openCollapsedUvs), which Blender's operator leaves for Unreal to warn about. A circle
    is still an outline with no face unless fillType says otherwise.

    `size` AND `radius` ARE NOT SYNONYMS and are never swapped for one another. A cube takes size; a
    sphere takes radius, which is half of it. Passing the wrong one is refused naming which that kind
    takes, rather than silently reinterpreted - an earlier version remapped one onto the other and
    made four kinds come out at twice the requested size with nothing in the response to say so. A
    cone takes radius1/radius2 and a torus majorRadius/minorRadius; neither takes size or radius.

    REFUSED IN EDIT MODE, because primitive_*_add welds into the edited mesh there - a destructive
    outcome dressed up as a failure.

    enter_editmode IS FORCED OFF. Blender defaults that operator property from a USER PREFERENCE
    (Editing > Objects > New Objects > Enter Edit Mode), so on a machine with it turned on the call
    would leave Blender in edit mode and strand every op after it. Torus is the exception: its
    operator has no such property and passing one raises.
    """
    reject_unknown(params, CREATE_PRIMITIVE_PARAMS, "create_primitive")
    kind = (take(params, "kind", "type", required=True, kind=str) or "").lower()
    if kind not in PRIMITIVES:
        raise MifOpError("unknown primitive kind '%s' - accepted: %s. Refused rather than "
                         "defaulted, because a substituted shape looks like success."
                         % (kind, ", ".join(sorted(PRIMITIVES))))
    spec = PRIMITIVES[kind]

    # EVERYTHING IS VALIDATED BEFORE ANYTHING IS CREATED. A refusal must leave nothing behind, and
    # an earlier version checked the name's type only after bpy.ops had already made the object.
    name = take(params, "name", kind=str)
    align = take(params, "align", kind=str)
    if align is not None:
        align = align.upper()
        if align not in ALIGN_VALUES:
            raise MifOpError("align must be one of %s, got %r. NOTHING was created."
                             % (", ".join(ALIGN_VALUES), align))

    kwargs = {}
    for key, blender_key in EXTRA_KEYS.items():
        if key not in params:
            continue
        if blender_key not in spec["extras"]:
            accepted = sorted(k for k, v in EXTRA_KEYS.items() if v in spec["extras"])
            raise MifOpError(
                "'%s' does not apply to a %s - it accepts: %s. NOTHING was created."
                % (key, kind, ", ".join(accepted) or "(no extra parameters)"))
        val = params[key]
        # BOUNDED AND FINITE, because these go straight into a bpy.ops operator's properties and it
        # converts them itself. segments: 2**40 came back as a raw ValueError - "Converting py args
        # to operator properties" - from inside primitive_uv_sphere_add, escaping this op's refusal
        # contract entirely. The shared helpers say the same thing as a sentence.
        kwargs[blender_key] = (val if isinstance(val, str)
                               else (finite_int(val, key) if blender_key in INT_KWARGS
                                     else finite_float(val, key)))

    size = take_float(params, "size", default=None)
    radius = take_float(params, "radius", default=None)
    if size is not None and radius is not None:
        raise MifOpError("pass size OR radius, not both. NOTHING was created.")
    given, given_name = ((size, "size") if size is not None else (radius, "radius"))
    if given is not None:
        if spec["size"] is None:
            accepted = sorted(k for k, v in EXTRA_KEYS.items() if v in spec["extras"])
            raise MifOpError(
                "a %s takes neither size nor radius - it is sized by %s. NOTHING was created."
                % (kind, ", ".join(accepted) or "(nothing)"))
        if spec["size"] != given_name:
            raise MifOpError(
                "a %s is sized by `%s`, not `%s`, and those are DIFFERENT dimensions - a radius is "
                "half of a size, so substituting one would quietly give you an object at the wrong "
                "scale. Pass %s instead. NOTHING was created."
                % (kind, spec["size"], given_name, spec["size"]))
        kwargs[spec["size"]] = given

    if spec.get("capKw"):
        kwargs[spec["capKw"]] = kwargs.pop("fill_type", "TRIFAN")
    kwargs["location"] = _vec3(params, "location", (0.0, 0.0, 0.0))
    kwargs["rotation"] = _vec3(params, "rotation", (0.0, 0.0, 0.0))
    if align is not None:
        kwargs["align"] = align
    if not spec.get("noEnterEditmode"):
        kwargs["enter_editmode"] = False

    _require_object_mode("create_primitive")

    snapshot = selection_snapshot()
    before = set(bpy.data.objects.keys())
    try:
        getattr(bpy.ops.mesh, spec["fn"])(**kwargs)
        made = [n for n in bpy.data.objects.keys() if n not in before]
        if len(made) != 1:
            raise MifOpError("expected exactly one new object, got %d (%s). NOTHING usable was "
                             "produced." % (len(made), ", ".join(made)))
        obj = bpy.data.objects[made[0]]
        if name:
            # THROUGH THE HELPER, not a bare assignment. On 3.6 `obj.name = name` renames whichever
            # of the two colliding objects sorts first, so when the operator's default name sorts
            # after the caller's - "Torus" against "MifT_Clash" - the object that ALREADY had the
            # name is the one silently renamed, and that is the one direction the nameNote below
            # cannot see. See ops_common.assign_id_name and ledger B12. The DATA name goes the same
            # way for the same reason: meshes are IDs too, and a displaced mesh is just as invisible.
            assign_id_name(obj, name, bpy.data.objects)
            assign_id_name(obj.data, obj.name, bpy.data.meshes)
        if kind == "monkey":
            open_collapsed_uvs_of(obj)
    except TypeError as exc:
        raise MifOpError("Blender refused the %s parameters: %s" % (kind, exc))
    finally:
        # ON EVERY PATH, including the raises above. An earlier version had `finally: pass`, which
        # restored nothing and could leave the selection pointing at a half-made object.
        selection_restore(snapshot)

    # NESTED UNDER "object", matching op_object_info (ops_scene.py:177-180). The same payload flat
    # here and nested there would be two shapes for one thing, and an agent that creates a
    # primitive and then re-reads it would get different keys for identical data.
    out = {"object": object_info(obj), "created": True, "kind": kind}
    # ECHOED FROM THE OBJECT. Blender appends .001 on a collision and never says so. Duplicated at
    # top level because the name is the identity every op reports, not just a geometry fact.
    out["name"] = obj.name
    out["verts"] = out["object"].get("verts")
    # THE CANONICAL SPELLING, alongside the note. Ten other ops report requestedName and
    # nameWasSuffixed, and a caller that has to look for a differently-named field on every creation
    # op cannot write one check. nameWasSuffixed is now answerable on all five builds: before
    # assign_id_name it was silently False on 3.6 whatever happened (ledger B12).
    out["requestedName"] = name or None
    out["nameWasSuffixed"] = bool(name and obj.name != name)
    if name and obj.name != name:
        out["nameNote"] = ("Blender renamed this to '%s' because '%s' was already taken - it "
                           "appends a number rather than failing or overwriting."
                           % (obj.name, name))
    return out


def op_transform_object(params):
    """Move, rotate or scale an object WITHOUT baking it into the mesh data.

    apply_transform and set_origin both bake; there was no way to simply place something.

    ROTATION GOES TO THE FIELD THE OBJECT ACTUALLY USES. Blender evaluates rotation_euler ONLY when
    rotation_mode is one of the Euler orders; a QUATERNION or AXIS_ANGLE object ignores it entirely.
    An earlier version wrote rotation_euler unconditionally, so on a quaternion object - which is
    what glTF import produces for every node, and what constraint-driven objects tend to be - the
    object did not move AND the response read that same dead field back, reporting the requested
    rotation as fact. A silent no-op that confirms itself is the worst shape available. The euler is
    now converted into whichever representation the object is in.

    `after` IS DERIVED FROM matrix_world, not from the field that was written, so it cannot agree
    with the request by construction.
    """
    reject_unknown(params, ("object", "name", "location", "rotation", "scale", "relative"),
                   "transform_object")
    obj = get_object(take(params, "object", "name", required=True))
    # A LINKED OBJECT CANNOT BE MOVED BY THIS FILE. Measured on 5.0.1: this accepted the move,
    # reported the new location as fact, and the change is unsaveable - gone on the next reload.
    require_editable(obj, "move")
    relative = take_bool(params, "relative", default=False)
    if not any(k in params for k in ("location", "rotation", "scale")):
        raise MifOpError("nothing to set - pass location, rotation and/or scale. NOTHING was "
                         "changed.")

    before = {
        "location": rnd(list(obj.matrix_world.to_translation())),
        "rotationEuler": rnd(list(obj.matrix_world.to_euler())),
        "scale": rnd(list(obj.matrix_world.to_scale())),
        "rotationMode": obj.rotation_mode,
    }

    # ALL THREE PARSED BEFORE ANY IS WRITTEN. Each was parsed on the line above its own write, so
    # {location: [1, 2, 3], scale: [NaN, 1, 1]} MOVED the object and then refused the scale - ok:false with
    # the location changed (measured on 5.0.1, 2026-09-21). A caller reading a refusal re-sends the whole call.
    try:
        want_loc = _vec3(params, "location", (0, 0, 0)) if "location" in params else None
        want_scale = _vec3(params, "scale", (1, 1, 1)) if "scale" in params else None
        want_rot = _vec3(params, "rotation", (0, 0, 0)) if "rotation" in params else None
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was changed." % str(exc).rstrip("."))

    if want_loc is not None:
        v = want_loc
        obj.location = [a + b for a, b in zip(obj.location, v)] if relative else v
    if want_scale is not None:
        v = want_scale
        obj.scale = [a * b for a, b in zip(obj.scale, v)] if relative else v
    if want_rot is not None:
        # ops_common.write_rotation since 2026-09-28 - this block's own rule, moved there when set_light, set_camera and
        # aim_object were found writing rotation_euler whatever the mode. Same semantics, one implementation.
        write_rotation(obj, want_rot, relative)

    # Blender does not refresh matrix_world until the depsgraph runs, and a caller reading the world
    # transform straight afterwards would get the stale one.
    bpy.context.view_layer.update()

    out = {"object": object_info(obj), "name": obj.name, "before": before, "relative": relative}
    # READ BACK OFF matrix_world, so this reports where the object IS rather than what was asked
    # for. That is the whole difference between the two claims the docstring distinguishes.
    out["after"] = {
        "location": rnd(list(obj.matrix_world.to_translation())),
        "rotationEuler": rnd(list(obj.matrix_world.to_euler())),
        "scale": rnd(list(obj.matrix_world.to_scale())),
        "rotationMode": obj.rotation_mode,
    }
    out["bakedNote"] = ("this changed the OBJECT transform only - the mesh data is untouched. "
                        "apply_transform is what bakes it in, and export writes the object "
                        "transform unless you do.")
    # A KEYED CHANNEL TAKES THE WRITE AND GIVES IT BACK AT THE NEXT FRAME CHANGE, while the read-back above agrees with
    # it - measured on all five builds, see ops_anim.animated_channels. So the channels written that animation drives
    # are named, with what drives them.
    from .ops_anim import animated_channels   # local: the op modules import ops_common, not each other, at load
    written = [p for p, key in (("location", "location"), ("scale", "scale")) if key in params]
    if "rotation" in params:
        written.append({"QUATERNION": "rotation_quaternion", "AXIS_ANGLE": "rotation_axis_angle"}.get(
            obj.rotation_mode, "rotation_euler"))
    animated = animated_channels(obj, set(written))
    if animated:
        out["animatedChannels"] = sorted(animated)
        out["animatedNote"] = (
            "%s %s animated (%s): the value was written and reads back now, and the animation puts its own back at "
            "the next frame change. Key it with set_keyframe, or edit the curve, to keep it."
            % (", ".join(sorted(animated)), "is" if len(animated) == 1 else "are",
               "; ".join(sorted({s for sources in animated.values() for s in sources}))))
    if "rotation" in params and obj.rotation_mode != "XYZ":
        # The shared sentence (ops_common.rotation_mode_note): CONVERTED for QUATERNION and AXIS_ANGLE, and for another
        # Euler order the truth - the angles went to rotation_euler as given, read in that order. This said CONVERTED
        # for every mode but XYZ.
        out["rotationModeNote"] = rotation_mode_note(obj, written[-1])
    return out


_TRANSFORM_OBJECTS_KEYS = ("objects", "pivot", "rotation", "translation", "dryRun", "previewId", "previewTtlSeconds")


def op_transform_objects(params):
    """Rotate and translate several independent objects about one WORLD-space pivot.

    The matrix is applied to each object's evaluated world transform, not its mesh data. All
    inputs and all target objects are checked before the first write. Parented objects are
    refused because changing a parent's matrix would also move unlisted children.

    dryRun:true is the BULK PREVIEW (docs/15 section 5): every object's world transform before and after, and a
    previewId that {previewId} alone commits - planned again, refused on any difference, read back, rolled back.
    """
    reject_unknown(params, _TRANSFORM_OBJECTS_KEYS, "transform_objects")
    objects, before, after, dry_run, shown = _transform_objects_resolve(params)
    if not dry_run:
        for obj in objects:
            obj.matrix_world = after[obj.name]
        bpy.context.view_layer.update()
    return _transform_objects_answer(objects, before, after, dry_run, shown)


def _transform_objects_answer(objects, before, after, dry_run, shown):
    names = [obj.name for obj in objects]
    return {
        "objects": names, "count": len(names), "dryRun": dry_run,
        "pivot": shown["pivot"], "rotation": shown["rotation"], "translation": shown["translation"],
        "before": {name: rnd(list(matrix.to_translation())) for name, matrix in before.items()},
        "after": {obj.name: rnd(list((after[obj.name] if dry_run else obj.matrix_world).to_translation()))
                  for obj in objects},
        "rotationBefore": {name: rnd(list(matrix.to_euler())) for name, matrix in before.items()},
        "rotationAfter": {obj.name: rnd(list((after[obj.name] if dry_run else obj.matrix_world).to_euler()))
                          for obj in objects},
        "note": "world-space object transforms only; mesh data unchanged",
    }


def _transform_objects_resolve(params):
    """transform_objects' every check and its prediction, writing nothing: (objects, {name: world matrix before},
    {name: after}, dryRun, the echoed inputs). The op and its bulk preview plan (below) both run this."""
    reject_unknown(params, _TRANSFORM_OBJECTS_KEYS, "transform_objects")
    names = params.get("objects")
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
        raise MifOpError("'objects' must be a non-empty list of object names. NOTHING was changed.")
    if len(names) != len(set(names)):
        raise MifOpError("'objects' contains duplicate names. NOTHING was changed.")
    if len(names) > 500:
        raise MifOpError("'objects' is limited to 500 per call. NOTHING was changed.")
    if "rotation" not in params and "translation" not in params:
        raise MifOpError("pass rotation and/or translation. NOTHING was changed.")
    pivot = _vec3(params, "pivot", (0, 0, 0))
    rotation = _vec3(params, "rotation", (0, 0, 0))
    translation = _vec3(params, "translation", (0, 0, 0))
    dry_run = take_bool(params, "dryRun", default=False)
    objects = [get_object(name) for name in names]
    for obj in objects:
        require_editable(obj, "move")
        if obj.parent or obj.children:
            raise MifOpError("'%s' belongs to a parent hierarchy; transform_objects only accepts "
                             "independent objects so unlisted children cannot move. NOTHING was changed."
                             % obj.name)

    # mathutils ONLY NOW, after every refusal above has had its turn. reject_unknown is the first
    # statement of every op because test_blender_refusals sweeps the whole table with a bogus key
    # (B110) and with a missing object (B111) where no real mathutils exists; importing it first
    # answered both with an ImportError instead of the refusal.
    from mathutils import Euler, Matrix, Vector
    pivot = Vector(pivot)
    translation = Vector(translation)

    bpy.context.view_layer.update()
    turn = Euler(rotation, "XYZ").to_matrix().to_4x4()
    change = (Matrix.Translation(translation) @ Matrix.Translation(pivot) @
              turn @ Matrix.Translation(-pivot))
    before = {obj.name: obj.matrix_world.copy() for obj in objects}
    after = {name: change @ matrix for name, matrix in before.items()}
    return objects, before, after, dry_run, {"pivot": rnd(list(pivot)), "rotation": rnd(rotation),
                                             "translation": rnd(list(translation))}


# ---------------------------------------------------------------------------------------------------------------------
# THE BULK PREVIEW of transform_objects (ops_preview.py; docs/15 section 5): one change per object, its world transform
# before and after; the inverse sets the matrix back.
# ---------------------------------------------------------------------------------------------------------------------
def _world_transform(matrix):
    return {"location": rnd(list(matrix.to_translation()), 5), "rotationEuler": rnd(list(matrix.to_euler()), 5),
            "scale": rnd(list(matrix.to_scale()), 5)}


def _plan_transform_objects(request):
    objects, before, after, _dry, shown = _transform_objects_resolve(dict(request, dryRun=True))
    changes, unchanged, moving = [], [], []
    for obj in objects:
        row = {"kind": "transform", "op": "modify", "target": obj.name, "field": "matrix_world",
               "before": _world_transform(before[obj.name]), "after": _world_transform(after[obj.name])}
        if row["before"] == row["after"]:
            unchanged.append(row)
        else:
            changes.append(row)
            moving.append(obj.name)
    commit = {k: v for k, v in request.items() if k != "dryRun"}
    commit["objects"] = moving
    return {"changes": changes, "unchanged": unchanged, "context": [], "commit": commit,
            "response": _transform_objects_answer(objects, before, after, True, shown)}


def _read_transform_objects(change, _kept):
    obj = bpy.data.objects.get(change["target"])
    return None if obj is None else _world_transform(obj.matrix_world)


def _capture_transform_objects(plan):
    return {c["target"]: bpy.data.objects[c["target"]].matrix_world.copy() for c in plan["changes"]}


def _restore_transform_objects(kept, _plan):
    for name, matrix in kept.items():
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.matrix_world = matrix
    bpy.context.view_layer.update()


PREVIEW = {"transform_objects": {"plan": _plan_transform_objects, "read": _read_transform_objects,
                                 "capture": _capture_transform_objects, "restore": _restore_transform_objects}}


def op_set_parent(params):
    """Parent one object to another - or to a bone of an armature - or clear its parent, without moving it in the world
    unless told to.

    NO OP COULD BUILD A HIERARCHY until 2026-09-13 (spec 31470, found by the inert-parameter probe): the addon set a
    parent only on edit bones and collections. A prop on an empty, a camera on a rig, a lamp riding a moving object and
    every hierarchy an FBX carries into Unreal were out of reach, and bake_to_keyframes' clearParents had nothing to
    clear.

    KEEP TRANSFORM IS MEASURED, not trusted: the world matrix is read before and after, and worldMoved says whether the
    object moved. With keepTransform (the default) it must not; with keepTransform:false the object's CURRENT local
    transform is taken into the parent's space, which is Blender's "without inverse". The parent inverse is left as
    identity either way, so the local transform IS the offset from the parent - what an exporter writes.

    A CYCLE IS REFUSED BEFORE ANYTHING IS WRITTEN: an object as its own parent, or parented to its own descendant.

    params:
      object (str, required)
      parent (str)           the new parent
      clear (bool)           clear the parent instead - refused together with parent
      keepTransform (bool)   default true
      type (str)             OBJECT (default) | BONE | ARMATURE_AUTO (a mesh skinned to an armature parent with
                             automatic heat weights and an Armature modifier) | ARMATURE_NAME (the modifier and one empty
                             group per bone)
                             | VERTEX | VERTEX_3 (2026-10-06: the object follows one vertex of a mesh parent, or a
                             triangle of three - a button riding a cloth, the alternative to binding a deform modifier)
      vertices               with VERTEX or VERTEX_3: [i] or [i, j, k] of the parent mesh, or {near: [x,y,z]} (world) -
                             the nearest vertex, or the three corners of the nearest face nearest the point
      bone (str)             with type BONE: a bone of the parent armature
      fallback (str)         with ARMATURE_AUTO: NONE (default) | NEAREST | DIFFUSE - fill the vertices heat weighting
                             left with no bone weight (it fails per bone, silently, on open or overlapping shells).
                             NEAREST weighs each such vertex by 1 / distance^2 to the nearest maxInfluences deform bone
                             segments; DIFFUSE spreads the weights of the weighted vertices along the mesh's edges into
                             them (one ring per pass), and gives NEAREST's to any piece with no weighted vertex at all.
                             Only the empty vertices are written; the rest keep heat's weights.
      maxInfluences (int)    with a fallback: bones per filled vertex, 1-8 (default 4), weights normalised to 1
      bones (list)           with ARMATURE_AUTO or ARMATURE_NAME: weight these bones only and leave every other
                             group as it is - Blender's Assign Automatic from Bones (paint.weight_from_bones over the
                             selected pose bones, run headless in weight paint mode and the modes put back) for AUTO, an
                             empty group per named bone for NAME. The mesh is parented and given an Armature modifier
                             first if it has neither
      headTail (float)       with type BONE: 0 the bone's head to 1 its tail (default, Blender's), where the parent space
                             sits
      includeParents (bool)  with bones; default true for ARMATURE_AUTO, false for ARMATURE_NAME: each named bone's
                             deform parent joins the run, so a finger added to a weighted hand takes its share FROM the
                             hand rather than doubling it
    """
    reject_unknown(params, ("object", "name", "parent", "clear", "keepTransform", "type", "bone", "fallback",
                            "maxInfluences", "bones", "includeParents", "headTail", "vertices"), "set_parent")
    # HEAD TO TAIL (2026-10-06; 5.2 adds the slider): a bone parent's space sits at the bone's tail; headTail 0 puts
    # it at the head, 0.5 at the middle - through the parent inverse, on every build.
    head_tail = take_float(params, "headTail", default=None)
    if head_tail is not None and str(params.get("type") or "").upper() != "BONE":
        raise MifOpError("headTail places a BONE parent's space along the bone - pass type BONE. NOTHING was changed.")
    if head_tail is not None and not 0.0 <= head_tail <= 1.0:
        raise MifOpError("headTail must be 0 (head) to 1 (tail), got %g. NOTHING was changed." % head_tail)
    obj = get_object(take(params, "object", "name", required=True))
    require_editable(obj, "parent")
    parent_name = take(params, "parent", default=None, kind=str)
    clear = take_bool(params, "clear", default=False)
    keep = take_bool(params, "keepTransform", default=True)
    ptype = str(take(params, "type", default="OBJECT", kind=str) or "OBJECT").upper()
    bone = take(params, "bone", default=None, kind=str)
    fallback = str(take(params, "fallback", default="NONE", kind=str)).upper()
    max_inf = take_int(params, "maxInfluences", default=4)
    if fallback not in ("NONE", "NEAREST", "DIFFUSE"):
        raise MifOpError("fallback must be NONE, NEAREST or DIFFUSE, got %r. NOTHING was changed." % fallback)
    if (fallback != "NONE" or params.get("maxInfluences") is not None) and str(
            params.get("type") or "").upper() != "ARMATURE_AUTO":
        raise MifOpError("fallback and maxInfluences fill what automatic weights left empty - they go with type "
                         "ARMATURE_AUTO. NOTHING was changed.")
    if params.get("maxInfluences") is not None and fallback == "NONE":
        raise MifOpError("maxInfluences caps the fallback's bones per vertex; pass fallback NEAREST or DIFFUSE. NOTHING "
                         "was changed.")
    if not 1 <= max_inf <= 8:
        raise MifOpError("maxInfluences must be 1-8, got %d. NOTHING was changed." % max_inf)
    if clear and parent_name:
        raise MifOpError("pass parent OR clear:true, not both. NOTHING was changed.")
    if not clear and not parent_name:
        raise MifOpError("pass parent (an object name) or clear:true. NOTHING was changed.")
    if ptype not in ("OBJECT", "BONE", "ARMATURE_AUTO", "ARMATURE_NAME", "VERTEX", "VERTEX_3"):
        raise MifOpError("type must be OBJECT, BONE, ARMATURE_AUTO, ARMATURE_NAME, VERTEX or VERTEX_3, got %r. NOTHING "
                         "was changed." % ptype)
    if params.get("vertices") is not None and (clear or ptype not in ("VERTEX", "VERTEX_3")):
        raise MifOpError("vertices names the parent mesh's vertices for type VERTEX or VERTEX_3. NOTHING was changed.")
    if clear and ptype in ("VERTEX", "VERTEX_3"):
        raise MifOpError("type describes a NEW parent and means nothing with clear:true. NOTHING was changed.")
    if clear and (bone or ptype == "BONE"):
        raise MifOpError("type and bone describe a NEW parent and mean nothing with clear:true. NOTHING was changed.")
    if bone and ptype != "BONE":
        raise MifOpError("bone needs type:'BONE'. NOTHING was changed.")
    new_parent = None
    if parent_name:
        new_parent = get_object(parent_name)
        if new_parent == obj:
            raise MifOpError("an object cannot be its own parent. NOTHING was changed.")
        walk = new_parent
        while walk is not None:
            if walk == obj:
                raise MifOpError("'%s' is a descendant of '%s', so this would make a cycle. NOTHING was changed."
                                 % (new_parent.name, obj.name))
            walk = walk.parent
        if ptype == "BONE":
            if new_parent.type != "ARMATURE":
                raise MifOpError("type BONE needs an armature parent; '%s' is %s. NOTHING was changed."
                                 % (new_parent.name, new_parent.type))
            names = [b.name for b in new_parent.data.bones]
            if not bone or bone not in names:
                raise MifOpError("type BONE needs bone - one of %s on '%s'%s. NOTHING was changed."
                                 % (", ".join(names[:20]) or "(no bones)", new_parent.name,
                                    "" if not bone else ", and '%s' is not one" % bone))
    parent_verts = None
    if new_parent is not None and ptype in ("VERTEX", "VERTEX_3"):
        parent_verts = _parent_vertices(new_parent, ptype, params.get("vertices"))

    def world():
        m = obj.matrix_world
        return {"location": rnd(list(m.to_translation())), "rotationEuler": rnd(list(m.to_euler())),
                "scale": rnd(list(m.to_scale()))}

    def parent_of(o):
        if o.parent is None:
            return None
        row = {"name": o.parent.name, "type": o.parent_type,
               "bone": (o.parent_bone or None) if o.parent_type == "BONE" else None}
        if o.parent_type in ("VERTEX", "VERTEX_3"):
            row["vertices"] = list(o.parent_vertices)[:1 if o.parent_type == "VERTEX" else 3]
        return row

    some_bones = params.get("bones")
    if params.get("includeParents") is not None and some_bones is None:
        raise MifOpError("includeParents goes with bones. NOTHING was changed.")
    if some_bones is not None:
        if ptype not in ("ARMATURE_AUTO", "ARMATURE_NAME"):
            raise MifOpError("bones weights chosen bones of an armature parent - it goes with type ARMATURE_AUTO or "
                             "ARMATURE_NAME. NOTHING was changed.")
        if fallback != "NONE":
            raise MifOpError("fallback fills what a whole-rig heat run left empty; with bones only the named bones are "
                             "weighted - pass one or the other. NOTHING was changed.")
        include = take_bool(params, "includeParents", default=ptype == "ARMATURE_AUTO")
        parent_before = parent_of(obj)
        return _weights_from_bones(obj, new_parent, ptype, some_bones, include, keep, parent_before)
    import mathutils
    bpy.context.view_layer.update()
    if ptype in ("ARMATURE_AUTO", "ARMATURE_NAME"):
        parent_before = parent_of(obj)
        out = _parent_to_armature(obj, new_parent, ptype, keep, parent_before)
        if fallback != "NONE":
            out.update(_fill_unweighted(obj, new_parent, fallback, max_inf))
        return out
    before_parent, before_world = parent_of(obj), world()
    kept = obj.matrix_world.copy()
    if clear:
        obj.parent = None
        obj.parent_type = "OBJECT"
    else:
        obj.parent = new_parent
        obj.parent_type = ptype
        if ptype == "BONE":
            obj.parent_bone = bone
        if parent_verts is not None:
            obj.parent_vertices = (parent_verts + [0, 0])[:3]
    obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    if not clear and ptype == "BONE" and head_tail is not None:
        length = new_parent.data.bones[bone].length
        obj.matrix_parent_inverse = mathutils.Matrix.Translation((0.0, -(1.0 - head_tail) * length, 0.0))
    bpy.context.view_layer.update()
    if keep and parent_verts is not None:
        # A VERTEX PARENT'S SPACE IS READ FROM THE EVALUATED PARENT, and the matrix_world setter does not: MEASURED on
        # 5.2.2, it solved against the parent object's own matrix and put a cube parented to vertex 4 one vertex
        # position away. So the parent space is measured off the evaluated child and the local transform solved in it.
        space = obj.matrix_world @ obj.matrix_basis.inverted_safe()
        obj.matrix_basis = space.inverted_safe() @ kept
        bpy.context.view_layer.update()
    elif keep:
        obj.matrix_world = kept
        bpy.context.view_layer.update()
    now = obj.matrix_world
    moved = max(abs(kept[i][j] - now[i][j]) for i in range(4) for j in range(4)) > 1e-4
    out = {"object": obj.name, "parentBefore": before_parent, "parentAfter": parent_of(obj),
           "keepTransform": keep, "worldBefore": before_world, "worldAfter": world(), "worldMoved": moved,
           "localLocation": rnd(list(obj.location))}
    if keep and moved:
        out["warning"] = ("keepTransform was asked for and the object MOVED in the world anyway - measured, not "
                          "assumed. A parent with non-uniform scale under rotation shears its children, and a "
                          "child's own transform cannot always undo that.")
    if clear and before_parent is None:
        out["note"] = "'%s' had no parent, so there was nothing to clear." % obj.name
    return out


def _parent_vertices(parent, ptype, raw):
    """set_parent VERTEX / VERTEX_3: the parent mesh's vertex indices, from [i] / [i, j, k] or {near: [x,y,z]} (world):
    the nearest vertex, or the three corners nearest the point of the face nearest it. Checked before any write."""
    want = 1 if ptype == "VERTEX" else 3
    if parent.type != "MESH":
        raise MifOpError("type %s follows a MESH parent's vertices; '%s' is %s. NOTHING was changed."
                         % (ptype, parent.name, parent.type))
    mesh = parent.data
    if raw is None:
        raise MifOpError("type %s needs vertices: [%s] of '%s', or {near: [x,y,z]}. NOTHING was changed."
                         % (ptype, "i" if want == 1 else "i, j, k", parent.name))
    if isinstance(raw, dict):
        if set(raw) != {"near"}:
            raise MifOpError("vertices as an object takes near [x,y,z] alone, got %s. NOTHING was changed."
                             % sorted(raw))
        import mathutils
        point = mathutils.Vector(finite_floats(raw["near"], "vertices.near")) if isinstance(
            raw["near"], (list, tuple)) and len(raw["near"]) == 3 else None
        if point is None:
            raise MifOpError("vertices.near must be [x,y,z] (world). NOTHING was changed.")
        local = parent.matrix_world.inverted_safe() @ point
        if not mesh.vertices:
            raise MifOpError("'%s' has no vertices to follow. NOTHING was changed." % parent.name)
        if want == 1:
            return [min(mesh.vertices, key=lambda v: (v.co - local).length_squared).index]
        if not mesh.polygons:
            raise MifOpError("'%s' has no faces, so there is no triangle to follow. NOTHING was changed." % parent.name)
        hit, _co, _no, face = parent.closest_point_on_mesh(local)
        if not hit or face < 0:
            raise MifOpError("no face of '%s' was found near %s. NOTHING was changed." % (parent.name, list(point)))
        corners = sorted(mesh.polygons[face].vertices, key=lambda i: (mesh.vertices[i].co - local).length_squared)
        return sorted(corners[:3])
    if not isinstance(raw, (list, tuple)) or len(raw) != want:
        raise MifOpError("type %s takes %d vertex index(es), got %r. NOTHING was changed." % (ptype, want, raw))
    bad = [i for i in raw if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(mesh.vertices)]
    if bad:
        raise MifOpError("%s are not vertex indices of '%s' (it has %d). NOTHING was changed."
                         % (bad, parent.name, len(mesh.vertices)))
    if len(set(raw)) != len(raw):
        raise MifOpError("VERTEX_3 needs three different vertices, got %r. NOTHING was changed." % (raw,))
    return list(raw)


def _weights_from_bones(obj, rig, ptype, names, include_parents, keep, before_parent):
    """set_parent bones: weights for the named deform bones only (and their deform parents), the rest left alone.

    MEASURED 2026-10-04 on 3.6.23, 4.4.0 and 5.2.1 (scratchpad probe p2_ops): paint.weight_from_bones runs headless with
    the mesh active in WEIGHT_PAINT and the armature selected; it solves heat over every deform bone and writes the
    SELECTED bones' groups only - a hand bone alone on a mesh nearer the forearm got no vertex. Bone selection is
    Bone.select on 3.6 and 4.4 and PoseBone.select on 5.2 (ops_pose.select_bones)."""
    if rig is None or rig.type != "ARMATURE":
        raise MifOpError("%s needs an armature parent; '%s' is %s. NOTHING was changed."
                         % (ptype, getattr(rig, "name", None), getattr(rig, "type", None)))
    if obj.type != "MESH":
        raise MifOpError("%s skins a MESH; '%s' is %s. NOTHING was changed." % (ptype, obj.name, obj.type))
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
        raise MifOpError("bones is a non-empty list of bone names. NOTHING was changed.")
    bones = rig.data.bones
    missing = [n for n in names if bones.get(n) is None]
    if missing:
        raise MifOpError("'%s' has no bone(s) %s. NOTHING was changed." % (rig.name, ", ".join(missing[:20])))
    named = [n for n in names if bones[n].use_deform]
    if not named:
        raise MifOpError("none of the named bones deforms (%s), so none can carry weights. NOTHING was changed."
                         % ", ".join(names[:20]))
    run = list(named)
    parents_added = []
    if include_parents:
        for n in named:
            par = bones[n].parent
            while par is not None and not par.use_deform:
                par = par.parent
            if par is not None and par.name not in run:
                run.append(par.name)
                parents_added.append(par.name)
    groups_before = {vg.name for vg in obj.vertex_groups}

    def weights_of(group_names):
        idx = {vg.index: vg.name for vg in obj.vertex_groups if vg.name in group_names}
        return [{idx[g.group]: g.weight for g in v.groups if g.group in idx} for v in obj.data.vertices]
    others = [vg.name for vg in obj.vertex_groups if vg.name not in run]
    before_run, before_others = weights_of(set(run)), weights_of(set(others))
    from .ops_pose import restore_bone_selection, select_bones   # noqa: PLC0415
    snap = selection_snapshot()
    # THE RIG'S OWN MODE IS PUT BACK TOO (settled 2026-10-18 on 3.6-5.2, test_blender_settle_routes BS607): an armature
    # left in POSE mode came back in OBJECT mode, because the weight-paint switch and the return to object mode take every
    # object in the mode with them, and selection_restore puts back the selection and the active object, not modes.
    rig_mode = rig.mode
    sel = None
    made_mod = False
    try:
        if not any(m.type == "ARMATURE" and m.object is rig for m in obj.modifiers):
            mod = obj.modifiers.new(name="Armature", type="ARMATURE")
            mod.object = rig
            made_mod = True
        if obj.parent is not rig:
            kept = obj.matrix_world.copy()
            obj.parent = rig
            obj.parent_type = "OBJECT"
            import mathutils   # noqa: PLC0415
            obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            bpy.context.view_layer.update()
            if keep:
                obj.matrix_world = kept
        for n in run:
            if obj.vertex_groups.get(n) is None:
                obj.vertex_groups.new(name=n)
        if ptype == "ARMATURE_AUTO":
            sel = select_bones(rig, set(run))
            for o in bpy.context.view_layer.objects:
                o.select_set(o is obj or o is rig)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
            try:
                bpy.ops.paint.weight_from_bones(type="AUTOMATIC")
            finally:
                bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError as exc:
        raise MifOpError("Blender would not weight '%s' from bones %s: %s" % (obj.name, run, str(exc).strip()))
    finally:
        if sel is not None:
            restore_bone_selection(rig, sel)
        selection_restore(snap)
        if rig_mode == "POSE" and rig.mode != "POSE":
            vl = bpy.context.view_layer
            prev = vl.objects.active
            vl.objects.active = rig
            try:
                bpy.ops.object.mode_set(mode="POSE")
            except RuntimeError:
                pass
            vl.objects.active = prev
    after_run, after_others = weights_of(set(run)), weights_of(set(others))
    changed = [i for i, (a, b) in enumerate(zip(before_run, after_run))
               if set(a) != set(b) or any(abs(a[k] - b[k]) > 1e-6 for k in a)]
    deform = {b.name for b in bones if b.use_deform}
    dg = {vg.index for vg in obj.vertex_groups if vg.name in deform}
    unweighted = sum(1 for v in obj.data.vertices if not any(g.group in dg and g.weight > 0.0 for g in v.groups))
    no_weights = [n for n in run if not any(row.get(n, 0.0) > 0.0 for row in after_run)]

    def wrange(rows):
        vals = [w for i in changed for w in rows[i].values()]
        return [round(min(vals), 6), round(max(vals), 6)] if vals else None
    if before_others != after_others:
        raise MifOpError("weighting %s changed the weights of other groups - Blender's run did not stay on the selected "
                         "bones. The mesh now holds that result." % run)
    out = {"object": obj.name, "parentBefore": before_parent, "parentAfter": {"name": rig.name, "type": obj.parent_type},
           "type": ptype, "bones": named, "parentsIncluded": parents_added, "groupsWritten": run,
           "groupsCreated": sorted({vg.name for vg in obj.vertex_groups} - groups_before),
           "armatureModifierAdded": made_mod, "verticesChanged": len(changed),
           "weightRangeBefore": wrange(before_run), "weightRangeAfter": wrange(after_run),
           "verticesUnweighted": unweighted, "bonesWithNoWeights": no_weights,
           "otherGroupsUntouched": len(others)}
    skipped = [n for n in names if n not in named]
    if skipped:
        out["notDeforming"] = skipped
    if ptype == "ARMATURE_AUTO" and no_weights:
        out["warning"] = ("%s got no vertex: heat weighting gives a vertex to the bone nearest it among ALL deform bones, "
                          "so a bone with nothing of the mesh nearer to it than to another bone stays empty."
                          % ", ".join(no_weights))
    if ptype == "ARMATURE_AUTO" and not include_parents:
        out["parentsNote"] = ("includeParents was off: the parents keep their old weights, and where the new bones took "
                              "weight the vertex now answers to both - normalised, each moves it half as far.")
    return out


def _fill_unweighted(obj, rig, how, max_inf):
    """set_parent's fallback: weights for the vertices automatic weighting left with no deform-bone weight."""
    deform = [b for b in rig.data.bones if b.use_deform]
    names = {b.name for b in deform}
    gidx = {vg.name: vg.index for vg in obj.vertex_groups}
    me = obj.data
    have = {}
    for v in me.vertices:
        row = {}
        for e in v.groups:
            nm = obj.vertex_groups[e.group].name
            if nm in names and e.weight > 0.0:
                row[nm] = e.weight
        have[v.index] = row
    empty = [i for i, row in have.items() if not row]
    if not empty:
        return {"fallback": how, "verticesFilled": 0, "fallbackNote": "automatic weights reached every vertex."}
    inv = obj.matrix_world.inverted_safe() @ rig.matrix_world
    segs = [(b.name, inv @ b.head_local, inv @ b.tail_local) for b in deform]

    def nearest(p):
        ds = []
        for nm, a, b in segs:
            ab = b - a
            t = max(0.0, min(1.0, (p - a).dot(ab) / ab.length_squared)) if ab.length_squared else 0.0
            ds.append(((a + ab * t - p).length, nm))
        ds.sort()
        picked = ds[:max_inf]
        if picked[0][0] < 1e-9:
            return {picked[0][1]: 1.0}
        raw = {nm: 1.0 / (d * d) for d, nm in picked}
        s = sum(raw.values())
        return {nm: w / s for nm, w in raw.items()}
    filled = {}
    if how == "DIFFUSE":
        nbrs = {i: [] for i in range(len(me.vertices))}
        for e in me.edges:
            a, b = e.vertices
            nbrs[a].append(b)
            nbrs[b].append(a)
        known = {i: row for i, row in have.items() if row}
        todo = set(empty)
        while todo:
            ring = {}
            for i in todo:
                got = [known[n] for n in nbrs[i] if n in known]
                if got:
                    acc = {}
                    for row in got:
                        for nm, w in row.items():
                            acc[nm] = acc.get(nm, 0.0) + w / len(got)
                    ring[i] = acc
            if not ring:
                break
            known.update(ring)
            filled.update(ring)
            todo -= set(ring)
        for i in todo:                       # a piece with no weighted vertex at all
            filled[i] = nearest(me.vertices[i].co)
    else:
        for i in empty:
            filled[i] = nearest(me.vertices[i].co)
    spread = []
    for i, row in filled.items():
        top = sorted(row.items(), key=lambda kv: -kv[1])[:max_inf]
        s = sum(w for _n, w in top) or 1.0
        for nm, w in top:
            if nm not in gidx:
                gidx[nm] = obj.vertex_groups.new(name=nm).index
            obj.vertex_groups[gidx[nm]].add([i], w / s, "REPLACE")
        spread.append(len(top))
    left = sum(1 for v in me.vertices if not any(obj.vertex_groups[e.group].name in names and e.weight > 0.0
                                                for e in v.groups))
    return {"fallback": how, "maxInfluences": max_inf, "verticesFilled": len(filled),
            "influencesPerFilledVertex": [min(spread), max(spread)] if spread else None,
            "verticesUnweightedAfter": left}


def _parent_to_armature(obj, rig, ptype, keep, before_parent):
    """set_parent's skinning branch: Blender's own parent_set, then the weights it produced MEASURED.

    THE GAP, 2026-10-04: a body too far from Manny's shape for a nearest-surface weight copy needs automatic (heat)
    weights, and nothing typed made them. Heat weighting fails per bone without raising - Blender reports "failed to
    find solution for one or more bones" and leaves those groups empty - so the reply lists every deform bone whose
    group came out with no weighted vertex, and the vertices no bone weights at all.
    """
    if rig is None or rig.type != "ARMATURE":
        raise MifOpError("%s needs an armature parent; '%s' is %s. NOTHING was changed."
                         % (ptype, getattr(rig, "name", None), getattr(rig, "type", None)))
    if obj.type != "MESH":
        raise MifOpError("%s skins a MESH; '%s' is %s. NOTHING was changed." % (ptype, obj.name, obj.type))
    groups_before = {vg.name for vg in obj.vertex_groups}
    snap = selection_snapshot()
    try:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        rig.select_set(True)
        bpy.context.view_layer.objects.active = rig
        with bpy.context.temp_override(active_object=rig, object=rig,
                                       selected_objects=[obj, rig], selected_editable_objects=[obj, rig]):
            bpy.ops.object.parent_set(type=ptype, keep_transform=keep)
    except RuntimeError as exc:
        raise MifOpError("Blender would not parent '%s' to '%s' with %s: %s" % (obj.name, rig.name, ptype,
                                                                              str(exc).strip()))
    finally:
        selection_restore(snap)
    bpy.context.view_layer.update()
    mods = [m for m in obj.modifiers if m.type == "ARMATURE" and m.object is rig]
    if obj.parent is not rig or not mods:
        raise MifOpError("parent_set reported no error and '%s' is %s, with %d Armature modifier(s) on '%s' - the skin "
                         "was not made." % (obj.name, "parented to it" if obj.parent is rig else "not parented to it",
                                            len(mods), rig.name))
    deform = [b.name for b in rig.data.bones if b.use_deform]
    idx = {vg.index: vg.name for vg in obj.vertex_groups}
    weighted_by = {}
    unweighted = 0
    for v in obj.data.vertices:
        any_w = False
        for e in v.groups:
            if e.weight > 0.0 and idx.get(e.group) in deform:
                weighted_by[idx[e.group]] = weighted_by.get(idx[e.group], 0) + 1
                any_w = True
        if not any_w:
            unweighted += 1
    empty = sorted(b for b in deform if b not in weighted_by)
    out = {"object": obj.name, "parentBefore": before_parent, "parentAfter": {"name": rig.name, "type": obj.parent_type},
           "type": ptype, "armatureModifier": mods[0].name, "deformBones": len(deform),
           "groupsCreated": sorted({vg.name for vg in obj.vertex_groups} - groups_before),
           "bonesWithWeights": len(weighted_by), "verticesUnweighted": unweighted,
           "vertices": len(obj.data.vertices)}
    if ptype == "ARMATURE_AUTO":
        out["bonesWithNoWeights"] = empty[:100]
        if unweighted:
            out["warning"] = ("%d of %d vertices got no bone weight - heat weighting found no solution there (a "
                              "non-manifold or self-intersecting mesh, or one far from the bones). They bind to "
                              "nothing." % (unweighted, len(obj.data.vertices)))
    return out


def _boolean_mesh_health(obj):
    """Small pre/post snapshot that can reveal a boolean damaging a previously healthy mesh."""
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        boundary = sum(1 for edge in bm.edges if len(edge.link_faces) == 1)
        loose = sum(1 for edge in bm.edges if len(edge.link_faces) == 0)
        non_manifold = sum(1 for edge in bm.edges if len(edge.link_faces) > 2)
        if bm.verts:
            lo = [min(vert.co[i] for vert in bm.verts) for i in range(3)]
            hi = [max(vert.co[i] for vert in bm.verts) for i in range(3)]
        else:
            lo = hi = [0.0, 0.0, 0.0]
        return {
            "boundaryEdges": boundary,
            "looseEdges": loose,
            "nonManifoldEdges": non_manifold,
            "boundsLocalMinBU": rnd(lo),
            "boundsLocalMaxBU": rnd(hi),
            "boundsLocalSizeBU": rnd([hi[i] - lo[i] for i in range(3)]),
        }
    finally:
        bm.free()


def _cutter_mesh(cutter):
    """A new mesh of the cutter as the boolean sees it - its modifiers applied, every data layer kept. The caller
    removes it."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return bpy.data.meshes.new_from_object(cutter, preserve_all_data_layers=True, depsgraph=depsgraph)


def _mesh_shells(mesh):
    """The mesh's shells: one vertex-index list per set of faces joined through shared vertices, in order of each
    shell's lowest vertex. Vertices on no face belong to no shell."""
    parent = list(range(len(mesh.vertices)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    used = set()
    for poly in mesh.polygons:
        vs = list(poly.vertices)
        used.update(vs)
        a = root(vs[0])
        for v in vs[1:]:
            b = root(v)
            if b != a:
                parent[b] = a
    shells = {}
    for v in sorted(used):
        shells.setdefault(root(v), []).append(v)
    return sorted(shells.values(), key=lambda s: s[0])


def _cutter_shell_count(cutter):
    mesh = _cutter_mesh(cutter)
    try:
        return len(_mesh_shells(mesh))
    finally:
        bpy.data.meshes.remove(mesh)


def _solver_is_exact(params):
    """True when this cut runs the EXACT solver: asked for, or Blender's default (EXACT on 3.6 to 5.0)."""
    solver = params.get("solver")
    return solver is None or str(solver).strip().upper() == "EXACT"


def _boolean_many_cutters(params):
    """(cutter names, listed) when this boolean_op call is cut one cutter or shell at a time, else None.

    A LIST of cutters always is. ONE cutter is when the EXACT solver would read several shells in it and the operation is
    a DIFFERENCE or a UNION: cut shell by shell, those give what the whole cutter means. An INTERSECT is not cut in turn
    - the target met with each shell in turn is the target inside EVERY shell, not inside any - so op_boolean_op runs it
    as one cut with the modifier's self-intersection on, which measured right on 5.0.1. Anything this cannot read (a
    missing object, a wrong type) returns None and the one-cutter path refuses it in its own words."""
    raw = params.get("cutter")
    if isinstance(raw, (list, tuple)):
        return list(raw), True
    if not isinstance(raw, str) or not _solver_is_exact(params):
        return None
    operation = str(params.get("operation") or params.get("op") or "difference").strip().upper()
    if operation not in ("DIFFERENCE", "UNION"):
        return None
    cutter = bpy.data.objects.get(raw)
    if cutter is None or cutter.type != "MESH" or cutter.name == params.get("target"):
        return None
    return ([raw], False) if _cutter_shell_count(cutter) > 1 else None


def _shell_objects(cutter):
    """One temporary object per shell of `cutter`, each standing where the cutter stands, with the cutter's materials
    and data layers. Returns [] for a single-shell cutter (none are made)."""
    whole = _cutter_mesh(cutter)
    made = []
    try:
        shells = _mesh_shells(whole)
        if len(shells) < 2:
            return []
        for i, keep in enumerate(shells):
            mesh = whole.copy()
            bm = bmesh.new()
            try:
                bm.from_mesh(mesh)
                keep_set = set(keep)
                bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.index not in keep_set], context="VERTS")
                bm.to_mesh(mesh)
            finally:
                bm.free()
            obj = bpy.data.objects.new("%s.MifShell%d" % (cutter.name, i + 1), mesh)
            obj.matrix_world = cutter.matrix_world.copy()
            bpy.context.scene.collection.objects.link(obj)
            made.append(obj)
        return made
    except Exception:
        _remove_objects_and_meshes(made)
        raise
    finally:
        bpy.data.meshes.remove(whole)


def _remove_objects_and_meshes(objects):
    """Remove temporary objects and then their meshes, through remove_objects (which updates the view layer)."""
    meshes = []
    for obj in objects:
        try:
            meshes.append(obj.data)
        except ReferenceError:
            pass
    remove_objects(objects)
    for mesh in meshes:
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _boolean_in_turn(params, names, listed, cleanup=None):
    """boolean_op for several cutters, or for one cutter of several shells: each single-shell cut made in turn by
    op_boolean_op itself, so every refusal and postcondition of one cut holds for each. A step that refuses puts the
    target's mesh back as it was before the first cut - NOTHING was changed - and the temporary shell objects are
    removed whatever happens."""
    target = get_object(take(params, "target", required=True), want_mesh=True)
    operation = str(take(params, "operation", "op") or "difference").strip().upper()
    if not names or not all(isinstance(n, str) and n for n in names):
        raise MifOpError("cutter takes one object name or a list of names - got %r. NOTHING was changed."
                         % (params.get("cutter"),))
    if len(set(names)) != len(names):
        raise MifOpError("cutter names %s more than once; each cutter cuts once. NOTHING was changed."
                         % sorted({n for n in names if names.count(n) > 1}))
    if operation == "INTERSECT" and len(names) > 1:
        raise MifOpError(
            "intersect takes one cutter. With several, 'in turn' keeps only what is inside EVERY cutter and 'together' "
            "is the trap this list exists to avoid (an EXACT boolean with touching or overlapping cutters in one "
            "object came back empty on 5.0.1). Union the cutters first (boolean_op with operation union, one at a "
            "time), then intersect with the result. NOTHING was changed.")
    try:
        delete_cutter = take_bool(params, "deleteCutter", default=False)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was changed." % exc)
    cutters = [get_object(n, want_mesh=True) for n in names]
    if any(c is target for c in cutters):
        raise MifOpError("'%s' is both the target and one of its cutters - a boolean cannot cut an object with "
                         "itself. NOTHING was changed." % target.name)

    exact = _solver_is_exact(params)
    before = mesh_counts(target)
    topology_before = _boolean_mesh_health(target)
    shape_before = mesh_shape(target.data.vertices)
    mesh_name = target.data.name
    backup = target.data.copy()
    temps, steps, shells_by_cutter = [], [], {}
    sub = {k: v for k, v in params.items() if k not in ("cutter", "deleteCutter", "cleanup")}
    # THE CLEANUP RUNS ONCE, AFTER THE LAST CUT, over every face any of the cuts made or touched.
    keys_before = _boolean_face_keys(target) if cleanup is not None else None
    try:
        for cutter in cutters:
            shells = _shell_objects(cutter) if exact and operation != "INTERSECT" else []
            temps.extend(shells)
            shells_by_cutter[cutter.name] = max(1, len(shells))
            plan = [(s.name, i + 1) for i, s in enumerate(shells)] or [(cutter.name, None)]
            for use, shell in plan:
                step = op_boolean_op(dict(sub, cutter=use, cleanup=False))
                steps.append({"cutter": cutter.name, "shell": shell, "changed": step.get("changed"),
                              "after": step.get("after"), "note": step.get("note")})
                last = step
    except Exception as exc:
        if steps:
            # PUT THE MESH BACK. Earlier cuts were applied to target.data in place; the copy taken before them is the
            # target as the caller left it.
            cut = target.data
            target.data = backup
            if cut.users == 0:
                bpy.data.meshes.remove(cut)
            backup.name = mesh_name
            backup = None
        where = "cut %d of %s" % (len(steps) + 1, ", ".join(names))
        _remove_objects_and_meshes(temps)
        if backup is not None:
            bpy.data.meshes.remove(backup)
        if isinstance(exc, MifOpError):
            raise MifOpError("%s refused: %s%s" % (where, str(exc).rstrip(),
                                                  " The cuts before it were undone." if steps else ""))
        raise
    _remove_objects_and_meshes(temps)
    bpy.data.meshes.remove(backup)
    cleaned = _boolean_cleanup(target, keys_before, cleanup) if cleanup is not None else None

    deleted = []
    if delete_cutter:
        removed = remove_objects(cutters)
        present = set(bpy.data.objects.keys())
        deleted = [n for n in removed if n not in present]

    after = mesh_counts(target)
    topology_after = _boolean_mesh_health(target) if cleaned is not None else (last.get("topologyAfter")
                                                                                 or _boolean_mesh_health(target))
    result = {
        "target": target.name,
        "operation": operation,
        "solver": last.get("solver"),
        "materialMode": last.get("materialMode"),
        "before": before,
        "after": after,
        "topologyBefore": topology_before,
        "topologyAfter": topology_after,
        "cutters": [c for c in names],
        "cutterShells": shells_by_cutter,
        "cutsApplied": len(steps),
        "appliedOneAtATime": True,
        "steps": steps,
        "cutterDeleted": bool(delete_cutter) and len(deleted) == len(names),
        "cuttersDeleted": deleted,
        "why": ("each cutter and each shell of a cutter is cut on its own: an EXACT boolean whose cutters touch or "
                "overlap inside one object can come back empty or as a sliver (measured on 5.0.1), and one at a "
                "time it does not"),
    }
    if not listed:
        result["cutter"] = names[0]
    if cleaned is not None:
        result["cleanup"] = cleaned
    degraded = {key: {"before": topology_before[key], "after": topology_after[key],
                      "added": topology_after[key] - topology_before[key]}
                for key in ("boundaryEdges", "looseEdges", "nonManifoldEdges")
                if topology_after[key] - topology_before[key] > 0}
    result["topologyDegraded"] = bool(degraded)
    if degraded:
        result["topologyDegradation"] = degraded
        result["topologyWarning"] = (
            "the cuts introduced new open, loose or non-manifold edges. Blender applied them and changed the mesh, "
            "but the result may be structurally damaged. Compare topologyBefore/topologyAfter before export.")
    if after["faces"] == before["faces"] and after["verts"] == before["verts"] \
            and mesh_shape(target.data.vertices) == shape_before:
        result["changed"] = False
        result["note"] = ("the mesh is UNCHANGED after every cut - for DIFFERENCE the cutters almost always miss '%s', "
                          "for UNION they sit inside it. Check their positions before exporting this." % target.name)
    elif after["faces"] == 0:
        result["changed"] = True
        result["note"] = ("the mesh is now EMPTY - every face was removed. Almost certainly not what was wanted.")
    else:
        result["changed"] = True
    return result


# THE CLEANUP AFTER A CUT (1.4, 2026-10-30). Found on a one-prompt prop: an EXACT difference of twelve joined slit
# boxes from a 48-sided cylinder left 48 new n-gons round the slits, and the run never looked at them.
# ON UNLESS REFUSED SINCE 2026-11-01: it began opt-in, so every existing caller got the cut it always got, and every plain
# boolean_op then made a mesh mesh_quality gameReady fails (all 11 tried). cleanup false gets the raw cut. Called with no
# cleanup, the dissolve runs at _BOOLEAN_AUTO_ANGLE, not 5 degrees: at 5 a fine curved cut (128 sides, 2.8 degrees) would
# lose its silhouette, and only flat faces should go when nobody asked.
_BOOLEAN_CLEANUP_KEYS = ("angle", "minFaceArea", "ngons")
_BOOLEAN_CLEANUP_NGONS = ("keep", "triangulate", "quads")
_BOOLEAN_CLEANUP_ANGLE = 5.0
_BOOLEAN_AUTO_ANGLE = 0.5
# How thin a triangle the split of an n-gon with near-straight corners may leave before it is dissolved (m), and how
# short an edge the near-coincident corners of two aligned meshes: under gameReady's sliver width, which is 1e-5 at most.
_BOOLEAN_DEGENERATE_DIST = 2e-5
# A SLIVER IS A FACE UNDER ONE SQUARE MILLIMETER at a meter per unit - in the mesh's own units squared, as clean_mesh's
# minFaceArea is.
_BOOLEAN_SLIVER_AREA = 1e-6
# Edges between regions a game mesh keeps apart are never dissolved: a material change (materialMode TRANSFER gives the
# cutter's faces their own), a seam, a sharp edge, or faces that point opposite ways. Only the names this build has.
_BOOLEAN_CLEANUP_DELIMIT = ("NORMAL", "MATERIAL", "SEAM", "SHARP")
# THE N-GONS ARE SPLIT BY DEFAULT, because the dissolve alone measured as nothing. On 5.2.1, 2026-10-30, an EXACT cut
# of twelve slits from a 48-sided cylinder, a door and window from a box wall, and a box intersected with a UV sphere
# each left n-gons (48, 4 and 54) and NO two cut faces that a 5-degree Limited Dissolve could merge: the EXACT solver
# already leaves the fewest faces its outline allows, so what a game mesh objects to is the n-gons themselves. quads
# triangulates the cut's n-gons (BEAUTY) and joins the triangles back into quads where two meet flat enough (40
# degrees, seams, sharp edges and materials kept apart); triangulate stops after the split; keep leaves them. The
# triangles a renderer draws are the same number either way - an engine splits an n-gon on import - and only how is
# then decided here, where it can be seen.
_BOOLEAN_CLEANUP_NGONS_DEFAULT = "quads"


def _boolean_cleanup_spec(params):
    """None when cleanup is false, else {angle, minFaceArea, ngons}. cleanup is true / false or {angle, minFaceArea,
    ngons}, and left out it runs at _BOOLEAN_AUTO_ANGLE. Read and checked before anything is cut, so a bad value changes
    nothing."""
    raw = params.get("cleanup")
    if raw is None:
        return {"angle": _BOOLEAN_AUTO_ANGLE, "minFaceArea": _BOOLEAN_SLIVER_AREA, "ngons": _BOOLEAN_CLEANUP_NGONS_DEFAULT}
    if isinstance(raw, dict):
        unknown = sorted(str(k) for k in raw if k not in _BOOLEAN_CLEANUP_KEYS)
        if unknown:
            raise MifOpError("cleanup takes %s; %s %s not among them. NOTHING was changed."
                             % (", ".join(_BOOLEAN_CLEANUP_KEYS), ", ".join(unknown),
                                "is" if len(unknown) == 1 else "are"))
        if any(isinstance(raw.get(k), bool) for k in _BOOLEAN_CLEANUP_KEYS):
            raise MifOpError("cleanup.angle and cleanup.minFaceArea are numbers, not true or false. NOTHING was "
                             "changed.")
        try:
            angle = take_float(raw, "angle", default=_BOOLEAN_CLEANUP_ANGLE)
            area = take_float(raw, "minFaceArea", default=_BOOLEAN_SLIVER_AREA)
        except MifOpError as exc:
            raise MifOpError("cleanup: %s. NOTHING was changed." % exc)
        ngons = raw.get("ngons", _BOOLEAN_CLEANUP_NGONS_DEFAULT)
        if not isinstance(ngons, str) or ngons.strip().lower() not in _BOOLEAN_CLEANUP_NGONS:
            raise MifOpError("cleanup.ngons is %s, got %r. NOTHING was changed."
                             % (", ".join(_BOOLEAN_CLEANUP_NGONS), ngons))
        ngons = ngons.strip().lower()
    else:
        try:
            on = take_bool(params, "cleanup")
        except MifOpError as exc:
            raise MifOpError("%s Or pass {angle, minFaceArea}. NOTHING was changed." % exc)
        if not on:
            return None
        angle, area, ngons = _BOOLEAN_CLEANUP_ANGLE, _BOOLEAN_SLIVER_AREA, _BOOLEAN_CLEANUP_NGONS_DEFAULT
    if not (math.isfinite(angle) and 0.0 < angle <= 180.0):
        raise MifOpError("cleanup.angle is in DEGREES, above 0 and at most 180 (default %g), got %r. NOTHING was "
                         "changed." % (_BOOLEAN_CLEANUP_ANGLE, angle))
    if not (math.isfinite(area) and area >= 0.0):
        raise MifOpError("cleanup.minFaceArea is a face area in the mesh's units squared, 0 or more (default %g), got "
                         "%r. NOTHING was changed." % (_BOOLEAN_SLIVER_AREA, area))
    return {"angle": angle, "minFaceArea": area, "ngons": ngons}


def _face_key(verts):
    """A face as the set of its corner positions, rounded: equal before and after a cut only when the cut left it alone
    (a vertex added on one of its edges, or a corner moved, makes it a different face)."""
    return tuple(sorted(tuple(round(c, 5) for c in v.co) for v in verts))


def _boolean_face_keys(obj):
    """Every face of obj's mesh as _face_key - taken before a cut, so the cleanup knows which faces the cut reached."""
    co = obj.data.vertices
    return {_face_key([co[i] for i in p.vertices]) for p in obj.data.polygons}


def _face_kinds(faces):
    """{faces, tris, quads, ngons} over an iterable of BMFaces or MeshPolygons."""
    out = {"faces": 0, "tris": 0, "quads": 0, "ngons": 0}
    for f in faces:
        n = len(f.verts) if hasattr(f, "verts") else f.loop_total
        out["faces"] += 1
        out["tris" if n == 3 else "quads" if n == 4 else "ngons"] += 1
    return out


def _open_edge_count(bm):
    """Edges on one face, or on three or more - what a cleanup must never add."""
    return sum(1 for e in bm.edges if len(e.link_faces) != 2)


def _merge_slivers(bm, untouched, area):
    """Collapse each cut face under `area` by welding the ends of its short edges (under the square root of `area`)
    until it is gone. An end on a face the cut did not reach stays where it is and takes the other; an edge with both
    ends on such faces is not welded. Returns (merged, left)."""
    edge_max = math.sqrt(area)

    def pinned(v):
        return any(g in untouched for g in v.link_faces)
    slivers = [f for f in bm.faces if f not in untouched and f.calc_area() < area]
    for f in slivers:
        for _ in range(len(f.verts) if f.is_valid else 0):
            if f.calc_area() >= area:
                break
            short = [e for e in f.edges
                     if e.calc_length() < edge_max and not (pinned(e.verts[0]) and pinned(e.verts[1]))]
            if not short:
                break
            a, b = min(short, key=lambda e: e.calc_length()).verts
            keep = a if pinned(a) else b if pinned(b) else None
            bmesh.ops.pointmerge(bm, verts=[a, b], merge_co=keep.co.copy() if keep else (a.co + b.co) * 0.5)
            if not f.is_valid:
                break
    # COUNTED AT THE END: welding one side of a thin slab takes its other slivers with it.
    left = sum(1 for f in slivers if f.is_valid and f.calc_area() < area)
    return len(slivers) - left, left


def _boolean_cleanup(target, keys_before, spec):
    """Clean what one boolean_op call cut, and only that: the faces whose corners differ from every face the target had
    before the cut (the ones it made or touched). Slivers under spec minFaceArea among them are collapsed by welding
    their short edges (_merge_slivers) - deleting one would open a hole in a closed mesh - and the weld is undone if it
    left more open or non-manifold edges than the cut did (a sliver that is the side of a thin slab cannot go without
    folding the slab). Then a Limited Dissolve at spec angle runs over the edges with cut faces on both sides and the
    vertices with cut faces all round, and the cut's n-gons are split as spec ngons says. A face the cut did not reach
    is never an argument to any step, and is checked unchanged before the mesh is written; if one is not, nothing is
    written and the answer says so."""
    mesh = target.data
    known = [i.identifier for i in bpy.ops.mesh.dissolve_limited.get_rna_type().properties["delimit"].enum_items]
    delimit = sorted(d for d in _BOOLEAN_CLEANUP_DELIMIT if d in known)
    whole_before = _face_kinds(mesh.polygons)
    area = spec["minFaceArea"]
    bm = None
    try:
        # SLIVERS FIRST, so the dissolve after can take the near-straight vertex a weld leaves - and on a mesh read
        # afresh when the weld is undone.
        sliver_note = None
        for attempt in (0, 1):
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bm.normal_update()
            untouched = {f for f in bm.faces if _face_key(f.verts) in keys_before}
            untouched_keys = sorted(_face_key(f.verts) for f in untouched)
            cut_before = _face_kinds(f for f in bm.faces if f not in untouched)
            small = sum(1 for f in bm.faces if f not in untouched and f.calc_area() < area)
            if attempt == 1:
                merged, left = 0, small
                break
            open_before = _open_edge_count(bm)
            merged, left = _merge_slivers(bm, untouched, area) if small else (0, 0)
            if _open_edge_count(bm) <= open_before:
                break
            sliver_note = ("welding the %d sliver(s) would have added %d open or non-manifold edge(s), so they were "
                           "left as the cut made them" % (small, _open_edge_count(bm) - open_before))
            bm.free()
            bm = None

        verts = [v for v in bm.verts if v.link_faces and all(g not in untouched for g in v.link_faces)]
        edges = [e for e in bm.edges if len(e.link_faces) == 2 and all(g not in untouched for g in e.link_faces)]
        f0 = len(bm.faces)
        if verts or edges:
            bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(spec["angle"]), use_dissolve_boundaries=False,
                                     verts=verts, edges=edges, delimit=set(delimit))
        dissolved = f0 - len(bm.faces)
        split = joined = 0
        if spec["ngons"] != "keep":
            ngons = [f for f in bm.faces if f not in untouched and len(f.verts) > 4]
            split = len(ngons)
            if ngons:
                tris = bmesh.ops.triangulate(bm, faces=ngons, quad_method="BEAUTY", ngon_method="BEAUTY")["faces"]
                if spec["ngons"] == "quads":
                    t0 = len(bm.faces)
                    bmesh.ops.join_triangles(bm, faces=[f for f in tris if f.is_valid and len(f.verts) == 3],
                                             angle_face_threshold=math.radians(40.0),
                                             angle_shape_threshold=math.radians(40.0), cmp_seam=True, cmp_sharp=True,
                                             cmp_materials=True, cmp_uvs=False, cmp_vcols=False)
                    joined = t0 - len(bm.faces)
        # THE NEEDLES THE SPLIT LEFT, and the UVs the cut collapsed. A cylinder's corners on a sphere's meridians put a
        # vertex a hair off each sphere edge: 30 zero-area faces and 57 slivers after the split, measured on 5.2.1.
        # Only edges with nothing but cut faces on them, and the untouched-face check below still rules.
        thin = [e for e in bm.edges if e.link_faces and all(g not in untouched for g in e.link_faces)]
        f1 = len(bm.faces)
        bmesh.ops.dissolve_degenerate(bm, dist=_BOOLEAN_DEGENERATE_DIST, edges=thin)
        degenerate = f1 - len(bm.faces)
        left_ngons = (triangulate_ngons(bm, [f for f in bm.faces if f not in untouched])[0]
                      if spec["ngons"] != "keep" else 0)
        uv_opened = _open_collapsed_uvs(bm, mesh.uv_layers[0].name) if len(mesh.uv_layers) else 0
        # THE PROMISE, MEASURED: every face the cut did not reach is still there, corner for corner.
        kept = sorted(_face_key(f.verts) for f in bm.faces if f in untouched and f.is_valid)
        if kept != untouched_keys:
            return {"applied": False, "angle": spec["angle"], "minFaceArea": spec["minFaceArea"],
                    "reason": "the cleanup would have changed %d face(s) the cut did not reach, so it was not written - "
                              "the cut is as the boolean left it" % len(set(untouched_keys) - set(kept))}
        cut_after = _face_kinds(f for f in bm.faces if f not in untouched)
        bm.normal_update()
        bm.to_mesh(mesh)
        mesh.update()
    finally:
        if bm is not None:
            bm.free()
    out = {"applied": True, "angle": spec["angle"], "minFaceArea": spec["minFaceArea"],
           "sliverEdgeMax": round(math.sqrt(area), 9), "delimit": delimit,
           "cutFaces": {"before": cut_before, "after": cut_after},
           "mesh": {"before": whole_before, "after": _face_kinds(mesh.polygons)},
           "untouchedFaces": len(untouched_keys), "untouchedFacesKept": True,
           "sliversMerged": merged, "sliversLeft": left, "facesDissolved": dissolved, "ngons": spec["ngons"],
           "ngonsSplit": split + left_ngons, "trianglePairsJoined": joined, "degenerateFacesDissolved": degenerate,
           "uvFacesOpened": uv_opened,
           "note": "cutFaces are the faces the cut made or touched - the only faces cleaned. mesh is the whole mesh "
                   "after the cut and after the cleanup; the counts before the cut are the answer's before."}
    if sliver_note:
        out["sliverNote"] = sliver_note
    return out


def op_boolean_op(params):
    """Cut, merge or intersect one mesh with another, and prove it by the resulting mesh.

    THIS IS A MODIFIER, NOT AN OPERATOR, which is what makes it the awkward one of the three
    mesh-combining verbs. Three steps fail independently: add the BOOLEAN modifier naming the
    cutter, APPLY it, and dispose of the cutter. A modifier added and never applied is the worst
    failure available here - the viewport shows the cut, and the exported FBX does not, so the 3D
    view agrees with the request while the deliverable disagrees.

    So success is measured from the mesh afterwards. modifier_add returns a status about the
    OPERATOR, not about the geometry.

    AN UNCHANGED MESH IS REPORTED, not swallowed. A DIFFERENCE whose cutter misses, or an INTERSECT
    with no overlap, legitimately changes nothing or empties the mesh - and both are almost always a
    modelling mistake the caller wants to hear about at once rather than discover in Unreal.

    The cutter is KEPT unless deleteCutter is asked for: it is usually reusable, and deleting
    someone's object as a side effect is not a default worth having.

    SEVERAL CUTTERS ARE CUT ONE AT A TIME. cutter takes a list of names, and a DIFFERENCE or UNION with the EXACT
    solver (Blender's default) cuts one cutter of several shells shell by shell: an exact boolean whose cutters touch
    or overlap inside one object came back empty or as a sliver on 5.0.1. The answer lists each cut in steps; a cut
    that refuses undoes the ones before it. An INTERSECT takes one cutter, and one of several shells is cut whole with
    the modifier's self-intersection on.

    CLEANUP (1.4, on unless cleanup is false since 2026-11-01, so the result passes mesh_quality gameReady): cleanup true
    or {angle (5 degrees; left out altogether, 0.5, so only flat faces go), minFaceArea (1e-6, the mesh's units squared),
    ngons (quads, triangulate or keep; default quads)} cleans the faces the cut made or touched and no other: slivers
    under minFaceArea are merged into a cut neighbor, a Limited Dissolve at angle runs over the cut faces (material,
    seam, sharp and normal borders kept), then the cut's n-gons are split - into triangles joined back into quads where
    they meet flat, or into triangles - and the needles that leaves are dissolved and collapsed UVs opened. The answer's
    cleanup gives faces, triangles, quads and n-gons of the cut faces and of the whole mesh before and after, and checks
    every face the cut did not reach is unchanged; with several cutters it runs once, after the last cut.
    """
    reject_unknown(params, ("target", "cutter", "operation", "op", "deleteCutter", "solver",
                            "materialMode", "cleanup"), "boolean_op")
    _require_object_mode("boolean_op")
    cleanup = _boolean_cleanup_spec(params)

    # SEVERAL CUTTERS, OR ONE CUTTER OF SEVERAL SHELLS, ARE CUT ONE AT A TIME. Measured on 5.0.1: an EXACT difference
    # with one cutter object holding a door box and a window box that share a face left a 9-vertex sliver of a
    # 4 x 3 m wall, and two overlapping boxes left it EMPTY; the same two boxes as two cuts in turn gave the right
    # wall. _boolean_in_turn makes those single-shell cuts through this same function.
    many = _boolean_many_cutters(params)
    if many is not None:
        return _boolean_in_turn(params, *many, cleanup=cleanup)

    target = get_object(take(params, "target", required=True), want_mesh=True)
    cutter = get_object(take(params, "cutter", required=True), want_mesh=True)
    if target is cutter:
        raise MifOpError("target and cutter are the same object ('%s') - a boolean cannot cut an "
                         "object with itself. NOTHING was changed." % target.name)

    raw = take(params, "operation", "op") or "difference"
    operation = str(raw).strip().upper()
    valid = ("DIFFERENCE", "UNION", "INTERSECT")
    if operation not in valid:
        raise MifOpError("operation '%s' is not one of difference, union, intersect. NOTHING was "
                         "changed." % raw)
    # take_bool, AND HERE rather than after the cut. It was bool(take(...)) at the end: bool("false") is True,
    # so {deleteCutter: "false"} DELETED the cutter, and so did the typo "flase" (measured on 5.0.1,
    # 2026-09-21). Parsed before the modifier exists, so a word it cannot read changes nothing.
    try:
        delete_cutter = take_bool(params, "deleteCutter", default=False)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was changed." % exc)

    # A hidden object is not a boolean problem the way it is a join problem - the modifier reads the
    # cutter's mesh data directly rather than going through selection - but applying the modifier
    # DOES need the target reachable and active, so that is checked.
    view_layer = bpy.context.view_layer
    if target.name not in view_layer.objects:
        raise MifOpError("'%s' is not in the active view layer, so its modifier cannot be applied. "
                         "NOTHING was changed." % target.name)

    before = mesh_counts(target)
    topology_before = _boolean_mesh_health(target)
    cutter_counts = mesh_counts(cutter)
    if cutter_counts["faces"] == 0:
        raise MifOpError("the cutter '%s' has no faces, so a boolean against it is undefined - "
                         "DIFFERENCE would remove nothing and INTERSECT would empty '%s'. NOTHING "
                         "was changed." % (cutter.name, target.name))
    # THE SHAPE, TAKEN NOW because the apply below replaces it. The postcondition at the end compares
    # against it when the counts tie.
    shape_before = mesh_shape(target.data.vertices)
    keys_before = _boolean_face_keys(target) if cleanup is not None else None

    # AN INTERSECT WITH A CUTTER OF SEVERAL SHELLS. Measured on 5.0.1: with a door box and a window box in one cutter
    # object, touching or overlapping, the EXACT intersect of a wall came back EMPTY; with the modifier's
    # self-intersection on it gave what the two boxes' union gives. A difference or a union is cut shell by shell
    # instead (_boolean_in_turn); an intersect cannot be, so it is cut whole with use_self.
    cutter_shells = _cutter_shell_count(cutter) if operation == "INTERSECT" else 1

    modifier = target.modifiers.new(name="MifBoolean", type="BOOLEAN")
    modifier.object = cutter
    modifier.operation = operation
    solver = take(params, "solver")
    if solver:
        # READ OFF THE LIVE ENUM, not a hardcoded pair, and this is the third place in the addon
        # that has had to learn it - _valid_light_types and the render-engine alias came first.
        #
        # WHAT MOVED, measured on 3.6.23, 4.2.17, 4.4.0 and 5.0.1:
        #   3.6 / 4.2 / 4.4   FAST, EXACT
        #   5.0               FLOAT, EXACT, MANIFOLD
        # FAST was RENAMED to FLOAT and MANIFOLD was added. The old pair was stale in both
        # directions: solver:'fast' raised TypeError from RNA on 5.0, and solver:'manifold' was
        # refused on the only build that has it.
        #
        # THE ASSIGNMENT USED TO SIT OUTSIDE THE try BELOW, so that TypeError escaped as a raw
        # exception with the BOOLEAN modifier still on the target - an unapplied boolean shows the
        # cut in the viewport and exports the original, which is the exact failure the apply path
        # further down was written to prevent. It is refused up front now, and the modifier removed.
        valid = [i.identifier
                 for i in modifier.bl_rna.properties["solver"].enum_items]
        wanted = str(solver).strip().upper()
        if wanted not in valid:
            # FAST AND FLOAT ARE THE SAME SOLVER under two names, so a caller written against
            # either spelling keeps working on every build rather than breaking at 5.0.
            alias = {"FAST": "FLOAT", "FLOAT": "FAST"}.get(wanted)
            if alias in valid:
                wanted = alias
            else:
                target.modifiers.remove(modifier)
                raise MifOpError(
                    "solver '%s' does not exist on Blender %s, which has %s. FAST was renamed FLOAT "
                    "at 5.0 and MANIFOLD was added there, so a solver name is not portable across "
                    "builds - fast and float are accepted as each other wherever one of them "
                    "exists. NOTHING was changed."
                    % (solver, bpy.app.version_string, ", ".join(valid)))
        modifier.solver = wanted

    # WHICH MATERIAL THE CUTTER'S NEW FACES WEAR. The modifier's default, INDEX, gives each new face the
    # target's slot with the cutter face's slot NUMBER - so a dark cutter carving a wheel arch into a
    # painted body leaves the arch's inside painted (slot 0 of the target), however the cutter was
    # dressed. TRANSFER gives them the cutter's own material, adding a slot to the target when it has
    # no slot with that material. Only the EXACT solver reads it, so it is refused with any other.
    material_mode = take(params, "materialMode")
    if material_mode is not None:
        modes = [i.identifier for i in modifier.bl_rna.properties["material_mode"].enum_items]             if "material_mode" in modifier.bl_rna.properties else []
        wanted_mode = str(material_mode).strip().upper()
        if wanted_mode not in modes:
            target.modifiers.remove(modifier)
            raise MifOpError("materialMode '%s' is not one of %s on Blender %s. NOTHING was changed."
                             % (material_mode, ", ".join(m.lower() for m in modes) or "(none)",
                                bpy.app.version_string))
        if str(modifier.solver) != "EXACT":
            solver_now = str(modifier.solver)
            target.modifiers.remove(modifier)
            raise MifOpError("materialMode is read by the EXACT solver only, and this cut would use %s, which "
                             "ignores it - the cutter's faces would take the target's slots by index whatever was "
                             "asked. Pass solver:'exact' with it. NOTHING was changed." % solver_now)
        modifier.material_mode = wanted_mode

    # TAKE WHAT IS NEEDED AS PYTHON VALUES NOW, because modifier_apply FREES the modifier and every
    # attribute read after it is a read of released RNA memory. That is undefined behaviour, and it
    # behaved differently per version rather than failing honestly: 3.6, 4.2 and 4.4 returned the
    # old string, 5.0.1 returned bytes that are not UTF-8 and raised UnicodeDecodeError from inside
    # a plain `.name`. Passing on three versions was luck, not correctness.
    self_intersection = False
    if cutter_shells > 1 and str(modifier.solver) == "EXACT" and hasattr(modifier, "use_self"):
        modifier.use_self = True
        self_intersection = True
    mod_name = str(modifier.name)
    mod_solver = str(modifier.solver) if solver else "default"
    mod_material_mode = str(modifier.material_mode) if material_mode is not None else "default"

    # APPLY IT. Everything above only describes the cut; this is the step that performs it, and
    # leaving it out is the failure this whole function is arranged around.
    view_layer.objects.active = target
    try:
        bpy.ops.object.modifier_apply(modifier=mod_name)
    except Exception as exc:
        # Leave nothing half-done: an unapplied modifier would render as a cut that does not export.
        # Looked up by NAME rather than through the handle - apply may have freed it even on the
        # path that then raised, and removing a freed modifier is its own crash.
        stale = target.modifiers.get(mod_name)
        if stale is not None:
            target.modifiers.remove(stale)
        raise MifOpError("the boolean modifier could not be applied to '%s' (%s). The modifier was "
                         "removed rather than left in place, because an unapplied one shows the cut "
                         "in the viewport and exports the original. NOTHING was changed."
                         % (target.name, exc))

    # THE postcondition, and it must be asked of data that is still alive. `modifier` is freed by
    # now; asking IT whether it still exists is the bug this line used to be.
    leftover = target.modifiers.get(mod_name)
    if leftover is not None:
        target.modifiers.remove(leftover)
        raise MifOpError("the boolean modifier is still on '%s' after apply, so the geometry was "
                         "NOT changed - it has been removed. An unapplied modifier renders as a cut "
                         "and exports as the original." % target.name)

    cleaned = _boolean_cleanup(target, keys_before, cleanup) if cleanup is not None else None
    after = mesh_counts(target)
    topology_after = _boolean_mesh_health(target)

    # THE CUTTER'S NAME IS TAKEN AS A PYTHON STRING BEFORE IT CAN BE DELETED, for the same reason as
    # mod_name above. The "unchanged" note below names the cutter, and it used to read `cutter.name`
    # AFTER deleteCutter had removed the object: ReferenceError "StructRNA of type Object has been
    # removed". That escaped as a refusal of a call that had ALREADY applied the cut and deleted the
    # cutter - the caller was told nothing happened while the scene had changed twice. Found by the
    # boolean showcase on 2026-09-11, whose later cylinders sat inside the hole the sphere had made.
    cutter_name = str(cutter.name)
    deleted = False
    if delete_cutter:
        remove_objects([cutter])
        deleted = cutter_name not in bpy.data.objects

    result = {
        "target": target.name,
        "operation": operation,
        "solver": mod_solver,
        "materialMode": mod_material_mode,
        "before": before,
        "after": after,
        "topologyBefore": topology_before,
        "topologyAfter": topology_after,
        "cutterDeleted": deleted,
    }
    if cleaned is not None:
        result["cleanup"] = cleaned
    if self_intersection:
        result["cutterShells"] = cutter_shells
        result["selfIntersection"] = True
        result["why"] = ("the cutter holds %d shells; an EXACT intersect reads touching or overlapping shells in one "
                         "object wrongly (empty on 5.0.1), so the cut ran with the modifier's self-intersection on"
                         % cutter_shells)
    degraded = {}
    for key in ("boundaryEdges", "looseEdges", "nonManifoldEdges"):
        delta = topology_after[key] - topology_before[key]
        if delta > 0:
            degraded[key] = {
                "before": topology_before[key],
                "after": topology_after[key],
                "added": delta,
            }
    result["topologyDegraded"] = bool(degraded)
    if degraded:
        result["topologyDegradation"] = degraded
        result["topologyWarning"] = (
            "the boolean introduced new open, loose or non-manifold edges. Blender applied the "
            "modifier and changed the mesh, but the result may be structurally damaged. Compare "
            "topologyBefore/topologyAfter and inspect or retry with solver:'exact' before export."
        )
    # THE POSTCONDITION, and it is reported rather than asserted, because "no change" is a legal
    # outcome that is nearly always a mistake.
    #
    # DECIDED BY THE SHAPE, not the counts. A cutter that takes a whole slab off a box leaves 8 verts
    # and 6 faces, the box it started as, and that was reported changed:false - "they do not
    # overlap" - about a cut that had worked. bisect_plane's flaw exactly, found by looking for it.
    if after["faces"] == before["faces"] and after["verts"] == before["verts"] \
            and mesh_shape(target.data.vertices) == shape_before:
        result["changed"] = False
        result["note"] = (
            "the mesh is UNCHANGED. The modifier applied cleanly, so this is geometry rather than "
            "failure: for DIFFERENCE it almost always means '%s' and '%s' do not overlap, and for "
            "UNION it means the cutter is entirely inside the target. Check their positions before "
            "exporting this." % (target.name, cutter_name))
    elif after["faces"] == 0:
        result["changed"] = True
        result["note"] = (
            "the mesh is now EMPTY - every face was removed. For INTERSECT that means the two "
            "objects share no volume; for DIFFERENCE it means the cutter completely enclosed the "
            "target. Almost certainly not what was wanted.")
    else:
        result["changed"] = True
    return result


def op_join_objects(params):
    """Join several mesh objects into one, keeping every material slot.

    JOINING IS DESTRUCTIVE AND ASYMMETRIC: the sources are DELETED and everything lands in the
    target. The target is explicit here because bpy.ops.object.join() otherwise silently uses
    whatever happened to be active.

    A SOURCE THAT CANNOT BE SELECTED IS REFUSED UP FRONT. join() only consumes objects in
    context.selected_objects, so a hidden one - hide_get(), hide_viewport, or outside the view layer
    - is silently skipped. An earlier version echoed the REQUEST back as `joined` and reported
    success having merged nothing. Each source is now checked for selectability by name first, the
    operator's return value is checked, and `consumed` is measured from which objects actually
    disappeared.

    MATERIAL SLOTS ARE THE THING TO WATCH. Join merges the slot lists and remaps face material_index
    values, so the result's slot ORDER is not either input's order - and slot order is what decides
    which Unreal material lands on which face.

    ARMATURES (second 1.3 audit, 2026-10-25): a target ARMATURE joins armature sources - a garment or prop rig into the
    body rig before an export as one skeleton. Blender renames a clashing bone to bone.001 silently, and the joined
    meshes' vertex groups then point at nothing; here each clashing source bone is renamed FIRST (rename is a bone's
    name write, which Blender carries into the vertex groups of the meshes it skins), the meshes skinned to a source
    get their Armature modifier repointed at the target, and the rename map is REPORTED. renameSuffix (default '.001',
    then .002 ...) is how a clash is renamed.

    keepSources (2026-11-07): join COPIES of the sources and leave the sources as they are. The change cap then counts
    the join as the one object it writes, the target; a consuming join (the default) also removes every source, and
    counts each as a removal - joining 29 parts at once was refused by the 25-change cap and had to be batched. The
    Unreal plugin's merge_static_mesh_actors counts the same way: the merged mesh and its actor, plus each piece only
    when keepInputs false destroys it. kept names the sources still there. Mesh targets only.
    """
    reject_unknown(params, ("target", "objects", "sources", "renameSuffix", "keepSources"), "join_objects")
    _require_object_mode("join_objects")
    keep = take_bool(params, "keepSources", default=False)
    probe = get_object(take(params, "target", required=True))
    if probe.type == "ARMATURE":
        if keep:
            raise MifOpError("keepSources joins copies of MESH sources; an armature join renames and repoints bones, so "
                             "it consumes its sources. NOTHING was changed.")
        return _join_armatures(probe, params)
    if params.get("renameSuffix") is not None:
        raise MifOpError("renameSuffix is for joining armatures. NOTHING was changed.")
    target = get_object(take(params, "target", required=True), want_mesh=True)
    names = take(params, "objects", "sources", required=True)
    if not hasattr(names, "__len__") or isinstance(names, str):
        raise MifOpError("objects must be a list of object names to join INTO target")
    sources = [get_object(n, want_mesh=True) for n in names]
    if target in sources:
        raise MifOpError("'%s' is both the target and a source - a join cannot consume its own "
                         "target. NOTHING was changed." % target.name)
    if not sources:
        raise MifOpError("objects[] is empty - nothing to join. NOTHING was changed.")

    view_layer = bpy.context.view_layer
    # One pass, not one linear scan per source - a join of many objects made this quadratic. Same
    # shape as the list_objects note in ops_scene.
    vl_names = set(view_layer.objects.keys())
    unreachable = []
    for obj in [target] + sources:
        if obj.name not in vl_names:
            unreachable.append("%s (not in the active view layer)" % obj.name)
        elif obj.hide_get() or obj.hide_viewport:
            unreachable.append("%s (hidden)" % obj.name)
    if unreachable:
        raise MifOpError(
            "these cannot take part in a join, because bpy.ops.object.join() only consumes objects "
            "it can SELECT and skips the rest silently: %s. Unhide them first. NOTHING was changed."
            % ", ".join(unreachable))

    slots_before = [s.material.name if s.material else None for s in target.material_slots]
    counts_before = mesh_counts(target)

    # THE COPIES JOIN, NOT THE SOURCES, with keepSources: each a copy of the object and of its mesh, in the source's
    # collections, so join() takes them exactly as it would have taken the source (transform, modifiers dropped alike).
    joining, copy_names = sources, None
    if keep:
        joining = []
        for src in sources:
            dup = src.copy()
            dup.data = src.data.copy()
            for coll in src.users_collection:
                coll.objects.link(dup)
            joining.append(dup)
        copy_names = [d.name for d in joining]
        copy_meshes = [d.data for d in joining]

    snapshot = selection_snapshot()
    try:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in joining:
            obj.select_set(True)
        target.select_set(True)
        view_layer.objects.active = target
        result = bpy.ops.object.join()
        if "FINISHED" not in result:
            if copy_names:
                remove_objects([bpy.data.objects[n] for n in copy_names if n in bpy.data.objects])
            raise MifOpError("bpy.ops.object.join() returned %s rather than FINISHED - nothing was "
                             "merged. NOTHING usable was produced." % set(result))
    finally:
        selection_restore(snapshot)

    present = set(bpy.data.objects.keys())
    if keep:
        # THE COPIES ARE WHAT MUST BE GONE, and every source must still be there - both measured.
        left_copies = [n for n in copy_names if n in present]
        missing = [n for n in names if n not in present]
        if left_copies or missing:
            raise MifOpError("keepSources: the join consumed %d of %d copies (%s still exist) and %d source(s) are "
                             "missing (%s). The target holds whatever WAS merged; re-read it before continuing."
                             % (len(copy_names) - len(left_copies), len(copy_names), ", ".join(left_copies) or "none",
                                len(missing), ", ".join(missing) or "none"))
        # The copies' meshes have no user left once joined; they go now rather than wait for a save to purge them.
        for m in copy_meshes:
            try:
                if m.users == 0:
                    bpy.data.meshes.remove(m)
            except ReferenceError:
                pass        # already freed by the join
    gone = [o for o in names if o not in present] if not keep else list(names)
    if not keep and len(gone) != len(names):
        left = [o for o in names if o in present]
        raise MifOpError(
            "the join consumed %d of %d source(s) - %s still exist. join() skips what it cannot "
            "select and reports nothing, so this is measured rather than assumed. The target holds "
            "whatever WAS merged; re-read it before continuing."
            % (len(gone), len(names), ", ".join(left)))

    slots_after = [s.material.name if s.material else None for s in target.material_slots]
    return {
        "target": target.name,
        "joined": list(names),
        # MEASURED from which objects actually disappeared, never echoed from the request. With keepSources nothing
        # named was consumed: the copies were, and kept lists the sources, every one measured as still there.
        "consumed": [] if keep else gone,
        "consumedCount": 0 if keep else len(gone),
        "keepSources": keep,
        "kept": list(names) if keep else None,
        "vertsBefore": counts_before.get("verts"),
        "verts": mesh_counts(target).get("verts"),
        "slotsBefore": slots_before,
        "slots": slots_after,
        "slotNote": ("join MERGES the material slot lists and remaps every face's slot index, so "
                     "the result's slot ORDER is not either input's order. Slot order is what "
                     "lines up against Unreal's material array - check it before export."),
    }


def _join_armatures(target, params):
    """join_objects on armatures - see op_join_objects."""
    tail = "NOTHING was changed."
    names = take(params, "objects", "sources", required=True)
    if not isinstance(names, list) or not names:
        raise MifOpError("objects must be a list of armature names to join INTO target. %s" % tail)
    sources = [get_object(n) for n in names]
    bad = [o.name for o in sources if o.type != "ARMATURE"]
    if bad:
        raise MifOpError("an armature joins only armatures; %s %s not. %s" % (", ".join(bad), "is" if len(bad) == 1
                                                                              else "are", tail))
    if target in sources:
        raise MifOpError("'%s' is both the target and a source. %s" % (target.name, tail))
    suffix = take(params, "renameSuffix", default=".001", kind=str)
    m = re.match(r"^(.*?)(\d+)$", suffix)
    stem, width, start = (m.group(1), len(m.group(2)), int(m.group(2))) if m else (suffix, 0, 0)
    vl = bpy.context.view_layer
    vl_names = set(vl.objects.keys())
    hidden = [o.name for o in [target] + sources if o.name not in vl_names or o.hide_get() or o.hide_viewport]
    if hidden:
        raise MifOpError("these cannot take part in a join (hidden or outside the view layer): %s. %s"
                         % (", ".join(hidden), tail))
    taken = {b.name for b in target.data.bones}
    plan = {}
    for src in sources:
        own = {b.name for b in src.data.bones}
        ren = {}
        for b in src.data.bones:
            if b.name in taken:
                k = start
                while True:
                    new = ("%s%s%0*d" % (b.name, stem, width, k)) if width else "%s%s" % (b.name, suffix) + (
                        "" if k == start else str(k))
                    if new not in taken and new not in own:
                        break
                    k += 1
                ren[b.name] = new
                taken.add(new)
            else:
                taken.add(b.name)
        plan[src.name] = ren
    skinned = {}
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        for md in o.modifiers:
            if md.type == "ARMATURE" and md.object in sources:
                skinned.setdefault(md.object.name, []).append((o.name, md.name))
    children = {src.name: [(c.name, c.matrix_world.copy(), c.parent_type, c.parent_bone) for c in src.children]
                for src in sources}
    bones_before = len(target.data.bones) + sum(len(s.data.bones) for s in sources)
    # RENAMED BEFORE THE JOIN, through the bone's name, which Blender carries into the skinned meshes' vertex groups
    for src in sources:
        for old, new in plan[src.name].items():
            src.data.bones[old].name = new
    group_misses = []
    for src in sources:
        for mesh_name, _md in skinned.get(src.name, []):
            mo = bpy.data.objects[mesh_name]
            for old, new in plan[src.name].items():
                if mo.vertex_groups.get(old) is not None and mo.vertex_groups.get(new) is None:
                    group_misses.append("%s.%s" % (mesh_name, old))
    snapshot = selection_snapshot()
    src_names = [s.name for s in sources]
    try:
        for o in vl.objects:
            o.select_set(False)
        for s in sources:
            s.select_set(True)
        target.select_set(True)
        vl.objects.active = target
        result = bpy.ops.object.join()
        if "FINISHED" not in result:
            raise MifOpError("bpy.ops.object.join() returned %s. The clashing bones of the sources were already renamed "
                             "(see the map); nothing was joined." % sorted(result))
    finally:
        selection_restore(snapshot)
    present = set(bpy.data.objects.keys())
    gone = [n for n in src_names if n not in present]
    repointed, reparented = [], []
    for src_name in src_names:
        for mesh_name, md_name in skinned.get(src_name, []):
            mo = bpy.data.objects.get(mesh_name)
            md = mo.modifiers.get(md_name) if mo is not None else None
            if md is not None and md.object is not target:
                md.object = target
                repointed.append(mesh_name)
        for cname, mw, ptype, pbone in children.get(src_name, []):
            c = bpy.data.objects.get(cname)
            if c is None or c.parent is target:
                continue
            c.parent = target
            if ptype == "BONE" and pbone:
                c.parent_type = "BONE"
                c.parent_bone = plan[src_name].get(pbone, pbone)
            c.matrix_world = mw
            reparented.append(cname)
    out = {"target": target.name, "type": "ARMATURE", "joined": src_names, "consumed": gone,
           "renamed": {k: v for k, v in plan.items() if v}, "boneCountBefore": bones_before,
           "boneCount": len(target.data.bones), "meshesRepointed": sorted(set(repointed)),
           "childrenReparented": reparented}
    if len(target.data.bones) != bones_before:
        out["warning"] = "the joined armature has %d bones where %d went in" % (len(target.data.bones), bones_before)
    if group_misses:
        out["groupWarning"] = ("vertex groups still under the old name after the rename (rename them to match): %s"
                               % ", ".join(group_misses[:20]))
    return out


_SEP_SELECT_KEYS = ("faces", "vertexGroup", "minWeight", "loop", "duplicate", "newName")


def _sep_pick(params, obj, bm):
    """(faces, edges) the selection names: faces by index (or a select_faces reply), by vertex group, or an edge loop."""
    given = [k for k in ("faces", "vertexGroup", "loop") if params.get(k) is not None]
    if len(given) != 1:
        raise MifOpError("mode selection takes exactly one of faces, vertexGroup or loop (got %s). NOTHING was changed."
                         % (", ".join(given) or "none"))
    if params.get("minWeight") is not None and given[0] != "vertexGroup":
        raise MifOpError("minWeight goes with vertexGroup. NOTHING was changed.")
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    if given[0] == "faces":
        raw = params["faces"]
        if isinstance(raw, dict):
            # A select_faces REPLY: its indices, unless it listed fewer than it matched.
            if (raw.get("indicesReturned") is not None and raw.get("matched") is not None
                    and raw["indicesReturned"] < raw["matched"]):
                raise MifOpError("that select_faces reply is truncated - it matched %s faces and listed %s. Raise its "
                                 "limit, or pass the indices. NOTHING was changed."
                                 % (raw["matched"], raw["indicesReturned"]))
            raw = raw.get("indices")
        if not isinstance(raw, (list, tuple)) or not raw:
            raise MifOpError("faces is a non-empty list of face indices, or a select_faces reply. NOTHING was changed.")
        n = len(bm.faces)
        bad = [i for i in raw if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n]
        if bad:
            raise MifOpError("faces %s are not face indices of '%s' (it has %d). NOTHING was changed."
                             % (bad[:10], obj.name, n))
        return [bm.faces[i] for i in dict.fromkeys(raw)], []
    if given[0] == "vertexGroup":
        gname = take(params, "vertexGroup", kind=str)
        vg = obj.vertex_groups.get(gname)
        if vg is None:
            raise MifOpError("'%s' has no vertex group %r (it has %s). NOTHING was changed."
                             % (obj.name, gname, ", ".join(g.name for g in obj.vertex_groups) or "none"))
        min_w = take_float(params, "minWeight", default=0.5)
        deform = bm.verts.layers.deform.active
        weights = {v: (v[deform].get(vg.index, 0.0) if deform is not None else 0.0) for v in bm.verts}
        faces = [f for f in bm.faces if all(weights[v] >= min_w for v in f.verts)]
        if not faces:
            raise MifOpError("no face of '%s' has every vertex at or above %g in %r. NOTHING was changed."
                             % (obj.name, min_w, gname))
        return faces, []
    spec = params["loop"]
    if isinstance(spec, (list, tuple)):
        n = len(bm.edges)
        bad = [i for i in spec if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n]
        if not spec or bad:
            raise MifOpError("loop as a list holds edge indices of '%s' (it has %d); bad: %s. NOTHING was changed."
                             % (obj.name, n, bad[:10]))
        return [], [bm.edges[i] for i in dict.fromkeys(spec)]
    from .ops_topology import _boundary_loops_of, _pick_loop      # noqa: PLC0415 - one loop picker for the add-on
    loops = _boundary_loops_of(bm)
    if not loops:
        raise MifOpError("'%s' has no open boundary loop. NOTHING was changed." % obj.name)
    return [], list(_pick_loop(bm, obj, loops, spec, "loop")[0])


def _separate_selection(params, obj):
    """separate_mesh mode selection - see op_separate_mesh. bmesh on a copy of the data; no edit mode."""
    require_editable(obj, "separate part of")
    dup = take_bool(params, "duplicate", default=False)
    new_name = take(params, "newName", default=None, kind=str)
    if new_name is not None and (not new_name.strip() or bpy.data.objects.get(new_name) is not None):
        raise MifOpError("newName %r is empty or already an object's name. NOTHING was changed." % new_name)
    if obj.data.shape_keys is not None:
        raise MifOpError("'%s' has shape keys; a selection separate writes the base mesh only. NOTHING was changed."
                         % obj.name)
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        faces, edges = _sep_pick(params, obj, bm)
        if edges and not dup and any(e.link_faces for e in edges):
            raise MifOpError("those loop edges border faces, and taking them out would delete those faces too - pass "
                             "duplicate true to copy the loop out. NOTHING was changed.")
        if faces and len(faces) == len(bm.faces) and not dup:
            raise MifOpError("the selection is every face of '%s', which would leave it empty - pass duplicate true to "
                             "copy them, or rename the object. NOTHING was changed." % obj.name)
        keep_f = {f.index for f in faces}
        keep_e = {e.index for e in edges} | {e.index for f in faces for e in f.edges}
        keep_v = {v.index for e in edges for v in e.verts} | {v.index for f in faces for v in f.verts}
    finally:
        bm.free()
    # THE NEW OBJECT: a copy of the object and its data, cut down to the selection.
    new_obj = obj.copy()
    new_obj.data = obj.data.copy()
    for coll in obj.users_collection:
        coll.objects.link(new_obj)
    if new_name:
        assign_id_name(new_obj, new_name, bpy.data.objects)
        assign_id_name(new_obj.data, new_name, bpy.data.meshes)
    nb = bmesh.new()
    try:
        nb.from_mesh(new_obj.data)
        nb.faces.ensure_lookup_table()
        nb.edges.ensure_lookup_table()
        nb.verts.ensure_lookup_table()
        # COLLECTED BEFORE ANY DELETE: a delete leaves the survivors' indices stale.
        kill_f = [f for f in nb.faces if f.index not in keep_f]
        kill_e = [e for e in nb.edges if e.index not in keep_e]
        kill_v = [v for v in nb.verts if v.index not in keep_v]
        bmesh.ops.delete(nb, geom=kill_f, context="FACES_ONLY")
        bmesh.ops.delete(nb, geom=[e for e in kill_e if e.is_valid], context="EDGES")
        bmesh.ops.delete(nb, geom=[v for v in kill_v if v.is_valid], context="VERTS")
        nb.to_mesh(new_obj.data)
    finally:
        nb.free()
    if not dup:
        ob = bmesh.new()
        try:
            ob.from_mesh(obj.data)
            ob.faces.ensure_lookup_table()
            ob.edges.ensure_lookup_table()
            if faces:
                bmesh.ops.delete(ob, geom=[ob.faces[i] for i in keep_f], context="FACES")
            else:
                bmesh.ops.delete(ob, geom=[ob.edges[i] for i in sorted(keep_e)], context="EDGES")
            ob.to_mesh(obj.data)
        finally:
            ob.free()
    obj.data.update()
    new_obj.data.update()
    bpy.context.view_layer.update()
    return new_obj, dup


def op_separate_mesh(params):
    """Split a mesh into separate objects, by loose parts or by material - or chosen faces or a loop into one.

    The counterpart to join. `mode` is "loose" (each disconnected island becomes its own object) or
    "material" (one object per material slot in use).

    THE NEW OBJECTS ARE NAMED BY BLENDER as <source>.001, .002, so the response lists what actually
    appeared rather than predicting names. A separate that produces nothing means the mesh had
    nothing to split on, reported as a measured zero rather than as bare success.

    MODE "selection" (2026-10-05): faces (indices, or a select_faces reply - refused when it listed fewer than it
    matched), vertexGroup (the faces whose every vertex weighs minWeight or more, default 0.5) or loop (edge indices,
    or {near: [x,y,z]} | {vertex: i} | {edge: i} | {index: i}: an open boundary loop as fill_mesh_loop names one) go to
    ONE new object - a sleeve's lower band split off as a cuff, a cup's rim loop copied out for the coffee surface,
    which went through a material slot and mode material before. duplicate true copies them and leaves the source
    whole; newName names the new object. The new object is a copy of the source - transform, material slots, UV layers,
    vertex groups, modifiers - cut down to the selection; without duplicate the source loses those faces (and the
    vertices only they used); a loop whose edges border faces is copied only (duplicate true), since taking it out
    would delete those faces. Done in bmesh, so no edit mode; a mesh with shape keys is refused. Reply: new and
    source counts.
    """
    # A LITERAL TUPLE: parity_check reads the accepted keys statically and refuses an expression it cannot read.
    reject_unknown(params, ("object", "name", "mode", "faces", "vertexGroup", "minWeight", "loop", "duplicate",
                            "newName"), "separate_mesh")
    _require_object_mode("separate_mesh")
    obj = get_object(take(params, "object", "name", required=True), want_mesh=True)
    mode = (take(params, "mode", default="loose", kind=str) or "loose").lower()
    if mode not in ("loose", "material", "selection"):
        raise MifOpError("mode must be \"loose\", \"material\" or \"selection\", got '%s'" % mode)
    extra = [k for k in _SEP_SELECT_KEYS if params.get(k) is not None]
    if mode != "selection" and extra:
        raise MifOpError("%s %s for mode selection. NOTHING was changed." % (", ".join(extra),
                                                                            "are" if len(extra) > 1 else "is"))
    if mode == "selection":
        new_obj, dup = _separate_selection(params, obj)
        nc, sc = mesh_counts(new_obj), mesh_counts(obj)
        out = {"object": obj.name, "mode": mode, "created": [new_obj.name], "createdCount": 1, "duplicate": dup,
               "new": {"name": new_obj.name, "verts": nc.get("verts"), "edges": nc.get("edges"),
                       "faces": nc.get("faces")},
               "source": {"name": obj.name, "verts": sc.get("verts"), "edges": sc.get("edges"),
                          "faces": sc.get("faces")},
               "verts": sc.get("verts")}
        if new_obj.name != (params.get("newName") or new_obj.name):
            out["nameWasSuffixed"] = True
        return out

    before = set(bpy.data.objects.keys())
    snapshot = selection_snapshot()
    try:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.separate(type="LOOSE" if mode == "loose" else "MATERIAL")
    except RuntimeError as exc:
        raise MifOpError("Blender refused the separate: %s" % exc)
    finally:
        # Leaving Blender in EDIT mode would strand every later op, so this runs on every path.
        try:
            if bpy.context.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:  # noqa: BLE001
            pass
        selection_restore(snapshot)

    made = sorted(n for n in bpy.data.objects.keys() if n not in before)
    out = {
        "object": obj.name,
        "mode": mode,
        "created": made,
        "createdCount": len(made),
        "verts": mesh_counts(obj).get("verts"),
    }
    if not made:
        out["note"] = ("nothing was separated - the mesh has only one %s, so there was nothing to "
                       "split on. createdCount:0 is the measured result, not a failure."
                       % ("loose part" if mode == "loose" else "material in use"))
    return out


_EMPTY_KEYS = {"name", "location", "rotation", "displayType", "displaySize", "collection",
               # AN IMAGE BACKDROP (2026-10-05): the image, how it draws, and a set of them placed in their views
               "image", "images", "opacity", "depth", "side", "showPerspective", "showOrthographic", "offset",
               "selectable"}
# THE ROTATION THAT TURNS AN IMAGE EMPTY (drawn in its local XY, facing +Z) TO FACE A VIEW: FRONT looks along +Y, so the
# drawing stands in XZ facing -Y; RIGHT looks along -X; TOP looks down -Z - Blender's numpad views.
_VIEW_ROTATION = {"FRONT": (90.0, 0.0, 0.0), "BACK": (90.0, 0.0, 180.0), "RIGHT": (90.0, 0.0, 90.0),
                  "LEFT": (90.0, 0.0, -90.0), "TOP": (0.0, 0.0, 0.0), "BOTTOM": (180.0, 0.0, 0.0)}
_IMAGE_EMPTY_KEYS = ("opacity", "depth", "side", "showPerspective", "showOrthographic", "offset", "selectable")


def _image_empty_settings(params):
    """The image-drawing settings, checked before anything exists: {opacity, depth, side, showPerspective,
    showOrthographic, offset, selectable} -> {object attribute: value}."""
    out = {}
    if params.get("opacity") is not None:
        op = take_float(params, "opacity")
        if not 0.0 <= op <= 1.0:
            raise MifOpError("opacity is 0-1, got %g. NOTHING was created." % op)
        out["opacity"] = op
    for key, attr in (("depth", "empty_image_depth"), ("side", "empty_image_side")):
        raw = take(params, key, default=None, kind=str)
        if raw is not None:
            valid = [e.identifier for e in bpy.types.Object.bl_rna.properties[attr].enum_items]
            if str(raw).upper() not in valid:
                raise MifOpError("%s must be one of %s, got %r. NOTHING was created." % (key, ", ".join(valid), raw))
            out[attr] = str(raw).upper()
    for key, attr in (("showPerspective", "show_empty_image_perspective"),
                      ("showOrthographic", "show_empty_image_orthographic")):
        if params.get(key) is not None:
            out[attr] = take_bool(params, key)
    if params.get("offset") is not None:
        raw = params["offset"]
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise MifOpError("offset is [x, y] (the image's corner in its own size; [-0.5, -0.5] centres it), got %r. "
                             "NOTHING was created." % (raw,))
        out["empty_image_offset"] = tuple(finite_floats(list(raw), "offset"))
    if params.get("selectable") is not None:
        out["hide_select"] = not take_bool(params, "selectable")
    return out


def _image_for_empty(spec, where):
    """An image from {file} (loaded, refused if missing or empty) or {name} (already in the file)."""
    if not isinstance(spec, dict) or len(spec) != 1 or next(iter(spec)) not in ("file", "name"):
        raise MifOpError("%s is {file: path} or {name: an image already in the file}, got %r. NOTHING was created."
                         % (where, spec))
    if "name" in spec:
        img = bpy.data.images.get(str(spec["name"]))
        if img is None:
            raise MifOpError("%s names image %r, and there is none. NOTHING was created." % (where, spec["name"]))
        return img
    from .ops_material import load_image_file   # noqa: PLC0415 - the one loader for every op that takes an image file
    return load_image_file(spec["file"], tail="NOTHING was created.")[0]


def _apply_image_empty(obj, img, settings):
    obj.empty_display_type = "IMAGE"
    obj.data = img
    for attr, val in settings.items():
        if attr == "opacity":
            obj.use_empty_image_alpha = True
            obj.color = (obj.color[0], obj.color[1], obj.color[2], val)
        else:
            setattr(obj, attr, val)
_FIELD_KEYS = {"name", "type", "location", "rotation", "collection", "strength", "flow", "noise",
               "seed", "falloffPower", "maxDistance", "shape"}
_CURVE_KEYS = {"name", "location", "rotation", "points", "splineType", "cyclic", "bevelDepth",
               "bevelResolution", "extrude", "resolution", "usePath", "collection", "dimensions", "radii",
               "bevelObject", "taperObject"}
_SETCURVE_KEYS = {"object", "name", "objects", "collection", "bevelMode", "bevelObject", "taperObject",
                  "taperRadiusMode", "fillMode", "resolutionU", "bevelResolution", "bevelDepth", "extrude", "offset",
                  "triangles"}
# (key, Curve attribute, kind): kind "enum" is checked against this build's own list, "object" names a CURVE object.
_CURVE_DATA = (
    ("bevelMode", "bevel_mode", "enum"),
    ("bevelObject", "bevel_object", "object"),
    ("taperObject", "taper_object", "object"),
    ("taperRadiusMode", "taper_radius_mode", "enum"),
    ("fillMode", "fill_mode", "enum"),
    ("resolutionU", "resolution_u", int),
    ("bevelResolution", "bevel_resolution", int),
    ("bevelDepth", "bevel_depth", float),
    ("extrude", "extrude", float),
    ("offset", "offset", float),
)
_TEXT_KEYS = {"name", "location", "rotation", "body", "size", "extrude", "align", "alignY",
              "collection", "bevelDepth", "font"}
_SETTEXT_KEYS = {"object", "name", "body", "size", "extrude", "bevelDepth", "align", "alignY",
                 "font"}

# THE bpy.data COLLECTION holding each object type's data, for removing the datablock a conversion orphans. A type not
# listed here is simply not cleaned - the conversion still happened and says so.
_DATA_COLLECTIONS = {"MESH": "meshes", "CURVE": "curves", "SURFACE": "curves", "FONT": "curves",
                     "CURVES": "hair_curves", "POINTCLOUD": "pointclouds", "GPENCIL": "grease_pencils",
                     "GREASEPENCIL": "grease_pencils_v3"}


def _posed_armatures(obj):
    """[(armature name, bones moved)] for each enabled Armature modifier on obj whose rig is not at rest now.

    Read off the EVALUATED rig, because an action's pose at the current frame lives there; a rig in REST position
    deforms nothing whatever its bones hold. A bone counts as moved when its matrix_basis is not the identity.
    """
    out = []
    if obj.type != "MESH":
        return out
    deps = bpy.context.evaluated_depsgraph_get()
    for m in obj.modifiers:
        rig = m.object if m.type == "ARMATURE" and m.show_viewport else None
        if rig is None or rig.type != "ARMATURE" or rig.data.pose_position == "REST" or rig.pose is None:
            continue
        ev = rig.evaluated_get(deps)
        moved = 0
        for pb in ev.pose.bones:
            mb = pb.matrix_basis
            if any(abs(mb[r][c] - (1.0 if r == c else 0.0)) > 1e-5 for r in range(4) for c in range(4)):
                moved += 1
        if moved:
            out.append((rig.name, moved))
    return out


_HAIR_TARGETS = ("HAIR_PARTICLES", "HAIR_CURVES")


def _hair_selectable(obj, what):
    view_layer = bpy.context.view_layer
    if obj.name not in view_layer.objects.keys():
        raise MifOpError("%s '%s' is not in the active view layer, and Blender's hair conversion acts on the selected "
                         "object. NOTHING was changed." % (what, obj.name))
    if obj.hide_get() or obj.hide_viewport:
        raise MifOpError("%s '%s' is hidden, and Blender's hair conversion acts on the selected object. Unhide it "
                         "first. NOTHING was changed." % (what, obj.name))


def _strands_below(surf, psys):
    """How many strands have a point (after the root) below the surface, against its normal - inside it for a closed
    mesh. Hair keys are in the surface's object space, as closest_point_on_mesh is."""
    below = 0
    for pa in psys.particles:
        for key in list(pa.hair_keys)[1:]:
            hit, loc, nor, _ = surf.closest_point_on_mesh(key.co)
            if hit and (key.co - loc).dot(nor) < -1e-6:
                below += 1
                break
    return below


def _convert_hair(params, obj, target):
    """convert_object HAIR_PARTICLES / HAIR_CURVES, through Blender's curves.convert_to_particle_system and
    curves.convert_from_particle_system.

    MEASURED on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1 (2026-10-06), identical on all five:
      to particles    needs Curves.surface (the operator's poll says so). The system is named after the Curves object
                      and holds one strand per STORED curve - a fur object's 11-16 guides, not the ~1000 curves its
                      modifiers evaluate to. A system of that name already on the surface is REWRITTEN in place
                      (FINISHED, no new system), so the reply says updatedExisting.
      from particles  the ACTIVE system of the active mesh; a new Curves object named after the system, at the
                      emitter's world matrix, not parented, Curves.surface unset; points follow the display path
                      (2^display_step + 1 per strand), not the hair keys; with interpolated children it holds one curve
                      per CHILD and no parents. An EMITTER system is CANCELLED with nothing made.
    The source is never removed, so keepOriginal, dropInstances and pose do not apply."""
    # false is no ask: the MCP wrapper sends keepOriginal and dropInstances false by default
    given = [k for k in ("keepOriginal", "dropInstances", "pose") if params.get(k) not in (None, False)]
    if given:
        raise MifOpError("%s do not apply to %s: Blender's hair conversion always keeps the source. NOTHING was "
                         "changed." % (", ".join(given), target))
    if target == "HAIR_PARTICLES":
        return _curves_to_particles(params, obj)
    return _particles_to_curves(params, obj)


def _curves_to_particles(params, obj):
    if params.get("particleSystem") is not None:
        raise MifOpError("particleSystem picks the system HAIR_CURVES reads; HAIR_PARTICLES names its system after the "
                         "Curves object. NOTHING was changed.")
    if obj.type != "CURVES":
        raise MifOpError("'%s' is a %s; HAIR_PARTICLES converts a CURVES (hair) object. NOTHING was changed."
                         % (obj.name, obj.type))
    surf = obj.data.surface
    if surf is None or surf.type != "MESH":
        raise MifOpError("'%s' has no mesh surface (Curves.surface), and its strands become a particle system ON the "
                         "surface. Set one with create_curves surface, or make it with surfaceDeform. NOTHING was "
                         "changed." % obj.name)
    stored = len(obj.data.curves)
    if stored == 0:
        raise MifOpError("'%s' stores no curves, and a hair system with no strands is nothing - one already named '%s' "
                         "on '%s' would be emptied. NOTHING was changed." % (obj.name, obj.name, surf.name))
    _hair_selectable(obj, "Curves object")
    deps = bpy.context.evaluated_depsgraph_get()
    evaluated = len(obj.evaluated_get(deps).data.curves)
    name = obj.name
    existing = surf.particle_systems.get(name) is not None
    systems_before = len(surf.particle_systems)
    snapshot = selection_snapshot()
    try:
        select_only([obj])
        result = bpy.ops.curves.convert_to_particle_system()
    except RuntimeError as exc:
        raise MifOpError("Blender refused to convert '%s' to particles: %s. NOTHING was changed."
                         % (name, str(exc).strip()))
    finally:
        selection_restore(snapshot)
    psys = surf.particle_systems.get(name)
    if "FINISHED" not in result or psys is None:
        raise MifOpError("curves.convert_to_particle_system returned %s for '%s' and '%s' has no system '%s'; read the "
                         "scene before continuing." % (sorted(result), name, surf.name, name))
    deps = bpy.context.evaluated_depsgraph_get()
    ev_psys = surf.evaluated_get(deps).particle_systems.get(name)
    strands = len(ev_psys.particles) if ev_psys is not None else 0
    out = {"object": name, "type": obj.type, "target": "HAIR_PARTICLES", "surface": surf.name,
           "particleSystem": psys.name, "settings": psys.settings.name, "updatedExisting": existing,
           "systemsBefore": systems_before, "systemsAfter": len(surf.particle_systems), "strands": strands,
           "keysPerStrand": len(ev_psys.particles[0].hair_keys) if strands else 0, "guideCurves": stored,
           "evaluatedCurves": evaluated,
           "strandsBelowSurface": _strands_below(surf, ev_psys) if strands else 0,
           "evaluatedNote": None}
    if evaluated != stored:
        out["evaluatedNote"] = ("Blender converts the %d curve(s) STORED on '%s', not the %d its modifiers evaluate to; "
                                "apply the modifiers first to convert the groom as drawn" % (stored, name, evaluated))
    out["note"] = ("%s the hair system '%s' on '%s' with %d strand(s); '%s' is unchanged"
                   % ("REWROTE" if existing else "made", psys.name, surf.name, strands, name))
    return out


def _particles_to_curves(params, obj):
    if obj.type != "MESH":
        raise MifOpError("'%s' is a %s; HAIR_CURVES converts a hair particle system on a MESH. NOTHING was changed."
                         % (obj.name, obj.type))
    systems = obj.particle_systems
    if len(systems) == 0:
        raise MifOpError("'%s' has no particle system to convert. NOTHING was changed." % obj.name)
    want = take(params, "particleSystem", default=None, kind=str)
    if want is None:
        index = systems.active_index
    else:
        index = systems.find(want)
        if index < 0:
            raise MifOpError("'%s' has no particle system '%s'. Systems: %s. NOTHING was changed."
                             % (obj.name, want, ", ".join(p.name for p in systems)))
    psys = systems[index]
    if psys.settings.type != "HAIR":
        raise MifOpError("particle system '%s' on '%s' is %s; only a HAIR system has strands to convert (Blender "
                         "cancels an emitter). NOTHING was changed." % (psys.name, obj.name, psys.settings.type))
    _hair_selectable(obj, "mesh")
    deps = bpy.context.evaluated_depsgraph_get()
    ev_psys = obj.evaluated_get(deps).particle_systems[index]
    strands, children = len(ev_psys.particles), len(ev_psys.child_particles)
    before = set(bpy.data.objects.keys())
    was_active = systems.active_index
    snapshot = selection_snapshot()
    try:
        systems.active_index = index
        select_only([obj])
        result = bpy.ops.curves.convert_from_particle_system()
    except RuntimeError as exc:
        raise MifOpError("Blender refused to convert '%s' on '%s' to curves: %s. NOTHING was changed."
                         % (psys.name, obj.name, str(exc).strip()))
    finally:
        systems.active_index = was_active
        selection_restore(snapshot)
    made = [bpy.data.objects[n] for n in sorted(set(bpy.data.objects.keys()) - before)]
    hair = [o for o in made if o.type == "CURVES"]
    if "FINISHED" not in result or len(hair) != 1:
        raise MifOpError("curves.convert_from_particle_system returned %s for '%s' on '%s' and made %d Curves "
                         "object(s) (%s); read the scene before continuing."
                         % (sorted(result), psys.name, obj.name, len(hair), ", ".join(o.name for o in made) or "none"))
    new = hair[0]
    # SET ON ITS SURFACE BY THE BRIDGE, which Blender leaves unset: without it the object cannot go back through
    # HAIR_PARTICLES, and a hair node group that attaches to a surface has none to read.
    new.data.surface = obj
    uv = obj.data.uv_layers.active
    if uv is not None:
        new.data.surface_uv_map = uv.name
    sizes = sorted(set(cv.points_length for cv in new.data.curves))
    out = {"object": new.name, "created": new.name, "requested": obj.name, "type": new.type, "target": "HAIR_CURVES",
           "particleSystem": psys.name, "strands": strands, "childStrands": children,
           "curveCount": len(new.data.curves), "pointCount": len(new.data.points), "pointsPerCurve": sizes,
           "surface": obj.name, "surfaceUvMap": new.data.surface_uv_map or None,
           "parent": new.parent.name if new.parent else None, "childrenNote": None}
    if children:
        out["childrenNote"] = ("'%s' has %d interpolated child strand(s), and Blender converts the children's paths: "
                               "%d curve(s), one per child, and the %d parent strand(s) are not among them"
                               % (psys.name, children, out["curveCount"], strands))
    out["note"] = ("'%s' holds the strands of '%s' at the emitter's world matrix, not parented; Curves.surface is set "
                   "to '%s' by the bridge (Blender leaves it unset). Points follow the system's display path "
                   "(2^display_step + 1 per strand), not its hair keys; '%s' and its system are unchanged"
                   % (new.name, psys.name, obj.name, obj.name))
    return out


def op_convert_object(params):
    """Convert an object to another type - a curve or text to a MESH, a mesh to a CURVE - baking its modifiers in.

    THE GAP IT CLOSES, 2026-09-16. add_modifier puts ARRAY, SOLIDIFY, BEVEL and 25 more types on curves and text, and
    apply_modifier cannot bake them: Blender refuses a constructive modifier on a curve ("Convert curve to mesh in
    order to apply"). With no convert op, a modifier on a curve could be added and never applied.

    WHAT BLENDER DOES, MEASURED on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1 with no addon (bpy.ops.object.convert):
      in place       the SAME object - name, parent, children and material slots intact - becomes the new type with
                     the WHOLE modifier stack baked in (a bevelled circle arrayed x3: 1728 evaluated verts before,
                     1728 real verts after) and the stack emptied.
      keepOriginal   the original is untouched, modifiers and all; a NEW object, which Blender names, holds the result.
      mesh -> MESH   applies the entire modifier stack - the whole-stack form of apply_modifier.
      shared data    every object using the same datablock converts with it, so they are named in the response.
      the old data   is left with no users. It is REMOVED here when nothing else holds it: create_curve names a curve's
                     data after its object, so an orphan keeping that name pushes the next curve of the same name to
                     "<name>.001" - the name-reservation trap ledger B6 records.
    TARGETS ARE READ OFF THE OPERATOR, never listed: 3.6 and 4.2 offer GPENCIL, 4.4 onward GREASEPENCIL, 5.0 onward
    POINTCLOUD as well.

    SUCCESS IS MEASURED ON THE RESULT - its type, and for a MESH its real vertex count against what the stack evaluated
    to before - not taken from the operator's FINISHED.

    AN ARMATURE MODIFIER IS BAKED IN THE POSE IT HOLDS NOW. A skinned garment converted while its rig's action sat at
    frame 30 came out in the frame-30 pose, and nothing said so (2026-10-03, the garment sim mesh). So when an enabled
    Armature modifier's rig is not at rest the convert is refused unless `pose` says which: "rest" converts with the
    rig switched to its rest position and puts it back, "current" bakes the pose as it stands.

    HAIR, BETWEEN ITS TWO SYSTEMS (2026-10-06): target HAIR_PARTICLES turns a Curves object's stored curves into a hair
    particle system on its surface; HAIR_CURVES turns a mesh's hair particle system (particleSystem, default the active
    one) into a new Curves object. See _convert_hair for what Blender does, measured.
    """
    reject_unknown(params, ("object", "name", "target", "keepOriginal", "dropInstances", "pose", "particleSystem"),
                   "convert_object")
    _require_object_mode("convert_object")
    obj = get_object(take(params, "object", "name", required=True, kind=str))
    requested = obj.name
    target = str(take(params, "target", default="MESH", kind=str)).upper()
    if target in _HAIR_TARGETS:
        return _convert_hair(params, obj, target)
    if params.get("particleSystem") is not None:
        raise MifOpError("particleSystem picks the system a HAIR_CURVES conversion reads, and target is %s. NOTHING "
                         "was changed." % target)
    targets = [i.identifier for i in bpy.ops.object.convert.get_rna_type().properties["target"].enum_items]
    if target not in targets:
        raise MifOpError("target '%s' is not a type Blender %s converts to. It offers: %s; and for hair, %s. NOTHING "
                         "was changed." % (target, bpy.app.version_string, ", ".join(targets),
                                           " and ".join(_HAIR_TARGETS)))
    keep = take_bool(params, "keepOriginal", default=False)
    drop = take_bool(params, "dropInstances", default=False)
    pose = take(params, "pose", default=None, kind=str)
    if pose is not None:
        pose = str(pose).lower()
        if pose not in ("rest", "current"):
            raise MifOpError("pose must be 'rest' or 'current' (got %r). NOTHING was changed." % pose)
    posed_rigs = _posed_armatures(obj)
    if posed_rigs and pose is None:
        raise MifOpError(
            "'%s' is deformed by %s, which %s not at rest at frame %d, and converting bakes that pose into the mesh. "
            "Pass pose 'rest' to convert in the rest position (the rig is put back afterwards) or pose 'current' to "
            "bake the pose as it stands. NOTHING was changed."
            % (requested, ", ".join("'%s' (%d bone(s) moved)" % (n, c) for n, c in posed_rigs),
               "is" if len(posed_rigs) == 1 else "are", bpy.context.scene.frame_current))
    if pose is not None and not any(m.type == "ARMATURE" and m.show_viewport and m.object is not None
                                    for m in obj.modifiers):
        raise MifOpError("pose says which pose an Armature modifier is baked in, and '%s' has no enabled Armature "
                         "modifier. NOTHING was changed." % requested)

    view_layer = bpy.context.view_layer
    if requested not in view_layer.objects.keys():
        raise MifOpError("'%s' is not in the active view layer, so the convert operator cannot select it. NOTHING was "
                         "changed." % requested)
    if obj.hide_get() or obj.hide_viewport:
        raise MifOpError("'%s' is hidden, and the convert operator only acts on what it can select. Unhide it first. "
                         "NOTHING was changed." % requested)

    type_before = obj.type
    data_before = obj.data
    data_name = data_before.name if data_before is not None else None
    sharers = sorted(o.name for o in bpy.data.objects
                     if data_before is not None and o.data is data_before and o.name != requested)
    modifiers_before = [m.name for m in obj.modifiers]
    slots_before = [s.material.name if s.material else None for s in obj.material_slots]
    evaluated_verts = None
    try:
        deps = bpy.context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(deps)
        me = ev.to_mesh()
        evaluated_verts = len(me.vertices)
        ev.to_mesh_clear()
    except Exception:                                                         # noqa: BLE001
        pass                     # an object with no mesh form (an empty) has nothing to compare, and says so below
    # WHAT THE CONVERSION DESTROYS AND THE BAKE COUNT CANNOT SEE. MEASURED 2026-09-16 on 3.6.23, 4.2.17, 4.4.0, 5.0.1
    # and 5.2.1 through this op (scratchpad measure_instances_through_ops.py, measure_instance_sources_convert.py): a
    # sphere whose Geometry Nodes tree scattered 482 cube instances converted in place to its 482 vertices and 512 faces
    # - every cube gone - and a sphere emitting 50 particles rendered as a cube lost its particle system (50 instances
    # to 0), both answering bakedMatchesEvaluated:true, because to_mesh() never saw them either. Vertex instancing of a
    # child survived (114 instances before and after, the child still parented). In place there is no undo through the
    # bridge, so it is refused before anything changes unless the caller says to drop them.
    gn_instances = (drawn_instances(obj) if obj.instance_type == "NONE"
                    and any(m.show_viewport and m.type == "NODES" for m in obj.modifiers) else 0)
    particle_systems = len(getattr(obj, "particle_systems", ()) or ())
    if (gn_instances or particle_systems) and not keep and not drop:
        lost = []
        if gn_instances:
            lost.append("the %d instance(s) its Geometry Nodes tree draws" % gn_instances)
        if particle_systems:
            lost.append("its %d particle system(s)" % particle_systems)
        raise MifOpError("converting '%s' in place would DESTROY %s - the conversion keeps only the evaluated mesh "
                         "(measured on every build). Add a Realize Instances node at the end of the tree to keep "
                         "instances as geometry, pass keepOriginal:true to leave the original intact, or pass "
                         "dropInstances:true to convert anyway. NOTHING was changed."
                         % (requested, " and ".join(lost)))
    names_before = set(bpy.data.objects.keys())

    snapshot = selection_snapshot()
    # pose "rest": each rig switched to its REST position for the convert, and put back by the finally whatever happens.
    rest_switched = []
    try:
        if pose == "rest":
            for m in obj.modifiers:
                if (m.type == "ARMATURE" and m.show_viewport and m.object is not None and m.object.type == "ARMATURE"
                        and m.object.data.pose_position != "REST"):
                    rest_switched.append((m.object.data, m.object.data.pose_position))
                    m.object.data.pose_position = "REST"
            if rest_switched:
                view_layer.update()
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        view_layer.objects.active = obj
        result = bpy.ops.object.convert(target=target, keep_original=keep)
    except RuntimeError as exc:
        raise MifOpError("Blender refused to convert '%s' (%s) to %s: %s. NOTHING was changed."
                         % (requested, type_before, target, str(exc).strip()))
    finally:
        selection_restore(snapshot)
        for arm_data, was in rest_switched:
            arm_data.pose_position = was
        if rest_switched:
            view_layer.update()
    if "FINISHED" not in result:
        raise MifOpError("bpy.ops.object.convert returned %s for '%s' (%s) -> %s, so nothing was converted. NOTHING "
                         "was changed." % (sorted(result), requested, type_before, target))

    if keep:
        made = sorted(set(bpy.data.objects.keys()) - names_before)
        if len(made) != 1:
            raise MifOpError("converting '%s' with keepOriginal should add exactly one object and added %d (%s). The "
                             "original is untouched; read the scene before continuing."
                             % (requested, len(made), ", ".join(made) or "none"))
        result_obj = bpy.data.objects[made[0]]
    else:
        result_obj = bpy.data.objects.get(requested)
        if result_obj is None and type_before == "META":
            # A METABALL LEAVES ITS NAME BEHIND: Blender writes the surface to a new object (M.001) and removes M, so this
            # refused a convert that had happened, under "NOTHING was changed" (measured on 5.2.1, 2026-11-01). The mesh
            # takes the name it was asked for - below, once its type has passed, so a refusal never follows a rename.
            made = sorted(set(bpy.data.objects.keys()) - names_before)
            if len(made) == 1:
                result_obj = bpy.data.objects[made[0]]
    if result_obj is None or result_obj.type != target:
        raise MifOpError("'%s' is %s after a FINISHED convert to %s - Blender reported success and did not convert it. "
                         "NOTHING was changed." % (requested, result_obj.type if result_obj else "gone", target))
    if result_obj.name != requested and not keep:
        assign_id_name(result_obj, requested, bpy.data.objects)

    removed = None
    if not keep and data_name is not None:
        coll = getattr(bpy.data, _DATA_COLLECTIONS.get(type_before, ""), None)
        old = coll.get(data_name) if coll is not None else None
        if old is not None and old.users == 0 and old is not result_obj.data:
            coll.remove(old)
            removed = data_name
    counts = mesh_counts(result_obj) if result_obj.type == "MESH" else None
    # UVS A TEXTURE CAN USE. A font's fill comes out with v 0 on every corner (all 111 faces of "Ab", measured on 5.2.1)
    # and a metaball's surface with no layer at all; both fail gameReady on a mesh nobody has touched yet. A Solidify's rim
    # baked in comes out collapsed too, and those keep where they are, opened a fraction of a texel.
    uv_written = uv_opened = 0
    if result_obj.type == "MESH":
        if type_before != "MESH":
            # A glyph's fill can leave needles too, on "Mif 8" on 3.6 to 5.0 and not on 5.2, which shade as streaks.
            bm = bmesh.new()
            try:
                bm.from_mesh(result_obj.data)
                # The needles stand on points in a straight line along the outline, so those points go first, or the
                # split only makes the same needles again; only on a mesh that has one, at a hair's angle.
                needles = any(f.calc_area() < _BOOLEAN_DEGENERATE_DIST * max(e.calc_length() for e in f.edges) / 2.0
                              for f in bm.faces)
                if needles:
                    bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(0.1), use_dissolve_boundaries=False,
                                             verts=bm.verts[:], edges=bm.edges[:], delimit={"MATERIAL"})
                    straight = []
                    for v in bm.verts:
                        rim = [e for e in v.link_edges if len(e.link_faces) == 1]
                        if len(rim) == 2 and len(v.link_faces) == 1:
                            a, b = (e.other_vert(v).co - v.co for e in rim)
                            if a.length > 0.0 and b.length > 0.0 and a.angle(-b, 1.0) < math.radians(0.1):
                                straight.append(v)
                    if straight:
                        bmesh.ops.dissolve_verts(bm, verts=straight)
                    for _ in range(4):
                        bmesh.ops.dissolve_degenerate(bm, dist=_BOOLEAN_DEGENERATE_DIST, edges=bm.edges[:])
                        triangulate_ngons(bm)
                        if not any(f.calc_area() < 1e-10 for f in bm.faces):
                            break
                    bm.to_mesh(result_obj.data)
                    result_obj.data.update()
            finally:
                bm.free()
            uv_written = project_flat_uvs(result_obj, only_collapsed=bool(len(result_obj.data.uv_layers)))
        uv_opened = open_collapsed_uvs_of(result_obj)
    also = [n for n in sharers if bpy.data.objects.get(n) is not None and bpy.data.objects[n].type == target] \
        if not keep else []
    return {
        "object": result_obj.name,
        "requested": requested,
        "created": result_obj.name if keep else None,
        "keptOriginal": requested if keep else None,
        "typeBefore": type_before,
        "type": result_obj.type,
        "uvFacesProjected": uv_written,
        "uvFacesOpened": uv_opened,
        "modifiersBefore": modifiers_before,
        "modifiersAfter": [m.name for m in result_obj.modifiers],
        "evaluatedVertsBefore": evaluated_verts,
        "meshCounts": counts,
        # THE BAKE, MEASURED: the real mesh holds what the stack evaluated to. None when there is no mesh to compare.
        "bakedMatchesEvaluated": (counts["verts"] == evaluated_verts
                                  if counts is not None and evaluated_verts is not None else None),
        "slotsBefore": slots_before,
        "slots": [s.material.name if s.material else None for s in result_obj.material_slots],
        "alsoConverted": also,
        "removedOrphanData": removed,
        "pose": pose,
        "posedRigs": [{"armature": n, "bonesMoved": c} for n, c in posed_rigs],
        "droppedInstances": gn_instances if (gn_instances and not keep) else None,
        "droppedParticleSystems": particle_systems if (particle_systems and not keep) else None,
        "copyLacks": (("the copy does not carry %s - the original keeps them."
                       % " or ".join(x for x in (("the %d Geometry Nodes instance(s)" % gn_instances) if gn_instances
                                                 else "", ("the %d particle system(s)" % particle_systems)
                                                 if particle_systems else "") if x))
                      if keep and (gn_instances or particle_systems) else None),
        "note": ("keepOriginal: '%s' is unchanged, modifiers and all, and '%s' holds the converted copy."
                 % (requested, result_obj.name)) if keep else (
                 "converted IN PLACE - same object, same name, parent and children; the modifier stack is baked "
                 "in and gone, and there is no undo through the bridge." + (
                     " %d other object(s) shared its data and converted with it: %s."
                     % (len(also), ", ".join(also)) if also else "")),
    }


# THE ONE FONT EVERY BLENDER HAS, and the reason `font` exists at all. bpy.data.fonts starts with a
# single entry - the built-in - and every text object made without asking gets it, so a scene full
# of labels was a scene in one typeface with no way to change that. Its name differs across builds
# ("Bfont" on 3.6, "Bfont Regular" from 4.0), so it is matched by PREFIX rather than by equality:
# comparing against one spelling would report the builtin as a custom font on half the builds.
_BUILTIN_FONT_PREFIX = "Bfont"


def _resolve_font(params, verb):
    """A typeface from a file path, or None to leave the text on the built-in.

    LOADED WITH check_existing, so ten labels sharing one .ttf share one datablock rather than
    stacking ten copies of the same font in the file. That is the same argument assign_node_group's
    `reuse` makes and the same one _link_new makes about collections: the obvious API call is the
    one that quietly accumulates.

    REFUSED BY PATH BEFORE BLENDER IS ASKED, because bpy.data.fonts.load raises a bare RuntimeError
    for a missing file and for a file that is not a font, and those are different problems with
    different fixes - one is a typo and the other is a .png.
    """
    raw = take(params, "font", default=None, kind=str)
    if raw is None:
        return None
    path = os.path.abspath(os.path.expanduser(str(raw)))
    if not os.path.isfile(path):
        raise MifOpError("no font file at '%s'. `font` takes a PATH to a .ttf/.otf - Blender has no "
                         "font library to name into, and the only face it ships with is the "
                         "built-in one you get by leaving this out. NOTHING was changed." % path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".ttf", ".otf", ".ttc", ".pfb", ".woff", ".woff2"):
        raise MifOpError("'%s' is not a font file Blender reads (%s). It loads .ttf, .otf, .ttc, .pfb, .woff and "
                         ".woff2. NOTHING was changed." % (path, ext or "no extension"))
    try:
        return bpy.data.fonts.load(path, check_existing=True)
    except Exception as exc:                                       # noqa: BLE001
        raise MifOpError("Blender refused '%s' as a font (%s). NOTHING was changed." % (path, exc))


def _font_fields(data):
    """What typeface this text is actually in, said the same way by both ops."""
    font = getattr(data, "font", None)
    name = getattr(font, "name", None)
    builtin = bool(name) and str(name).startswith(_BUILTIN_FONT_PREFIX)
    out = {"font": name, "fontIsBuiltIn": builtin,
           "fontPath": (getattr(font, "filepath", None) or None) if font is not None else None}
    if builtin:
        out["fontNote"] = ("this is Blender's BUILT-IN face, which is what a text object gets when "
                           "no font is asked for. Pass `font` with a path to a .ttf or .otf to use "
                           "anything else - there is no font library to name into.")
    return out
_ARM_KEYS = {"name", "location", "rotation", "bones", "displayType", "showInFront", "collection",
             # THE METARIG FORM (2026-10-04, ops_rigify.create_metarig): a bundled Rigify metarig, its size baked in
             "metarig", "height", "scale", "enableRigify"}
# A POSE BONE'S ROTATION MODE, MEASURED 2026-09-16 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1, identically (scratchpad
# measure_posebone_rotation_mode3.py): a new bone's pose starts in QUATERNION, so a euler written to it moves nothing (the
# posed tail stays at rest); with rotation_mode set to XYZ FIRST the same 90 degree euler turns it; and a bone already
# posed by quaternion keeps its pose when the mode changes - Blender CONVERTS the rotation into the new mode (a 90 degree
# quaternion became the euler (1.5708, 0, 0), tail unmoved). Which is also why the order matters: a euler written before
# the mode change is overwritten by the conversion. The second seat found create_armature accepting
# {'rotationMode': 'XYZ', 'bogusKey': 5} inside a bone with ok:true and the bone left QUATERNION (mailbox id 145).
_BONE_KEYS = ("name", "head", "tail", "parent", "connect", "rotationMode")

_EMPTY_TYPES = ("PLAIN_AXES", "ARROWS", "SINGLE_ARROW", "CIRCLE", "CUBE", "SPHERE", "CONE",
                "IMAGE")


def _resolve_collection(params):
    """Where a new object will be linked - RESOLVED BEFORE ANYTHING IS CREATED.

    Split out from the linking on 2026-09-03 because the first version did the lookup after
    bpy.data.objects.new(), so naming a collection that does not exist left the object BEHIND: in
    bpy.data, in no collection, therefore in no scene - invisible, unrendered, absent from the
    outliner, surviving the save with nothing to warn anybody. Precisely the state _link_new's own
    comment describes as the thing to avoid, produced by the refusal meant to prevent it.

    Found by a check asserting the refusal, not by reading. Every op in this module now resolves
    first and creates second, which is the same rule stated everywhere else here: a refusal fires
    BEFORE a mutation.
    """
    name = take(params, "collection", kind=str)
    if not name:
        return bpy.context.scene.collection
    coll = bpy.data.collections.get(name)
    if coll is None:
        known = sorted(c.name for c in bpy.data.collections)[:25]
        raise MifOpError("no collection named '%s'. Present: %s. Make one with create_collection. "
                         "NOTHING was created."
                         % (name, ", ".join(known) if known else "<none>"))
    return coll


def _link_new(obj, coll):
    """Link a freshly made object into an ALREADY-RESOLVED collection.

    LINKED SOMEWHERE ALWAYS. bpy.data.objects.new() creates an object that belongs to NO collection,
    which means it is in no scene: invisible in the viewport, absent from the render, missing from
    the outliner, and it survives the save with nothing to warn anybody. The same inert shape
    ops_collection was written around, and the default outcome of the obvious API call.
    """
    coll.objects.link(obj)
    return coll


def _place_values(params):
    """Parse location and rotation BEFORE anything is created. Returns what _place will write.

    _place runs after bpy.data.*.new and after the object is LINKED, so a malformed vector refused
    with the object already sitting in the caller's file. Same defect fixed in ops_lightcam's _vec3
    earlier on 2026-09-04 and still present here, because the parse lived inside the writer rather
    than beside the caller - five ops share _place, so five shared the defect.

    Found by the matrix's corrupted-payload pass once it learned to take parameters from FIXTURES:
    create_lattice with a bad location left the lattice behind.

    SPLIT RATHER THAN GUARDED. "Remember to validate first" is the kind of rule this file keeps
    removing; making _place take the parsed values means the parse cannot be skipped.
    """
    return (_vec3(params, "location", (0.0, 0.0, 0.0)),
            _vec3(params, "rotation", (0.0, 0.0, 0.0)))


def _place(obj, placed):
    obj.location, obj.rotation_euler = placed


def _created(obj, coll, want_type, verb, requested=None):
    """The shared postcondition: it exists, it is the RIGHT TYPE, and it is in a scene.

    AND WHETHER IT GOT THE NAME IT ASKED FOR. Blender renames on collision rather than failing, so
    a caller retrying a timed-out call gets "Foo.001" while believing it has "Foo" - and every
    later lookup by the name it asked for finds nothing, or worse, the OTHER object. Reported here
    rather than in each op because all five of these build a name the same way and got it wrong the
    same way.

    Type is checked because these ops build objects out of datablocks rather than through
    bpy.ops.*_add, so a wrong datablock class produces an object that exists and is not what was
    asked for - and every other field on it would read back fine.
    """
    if obj.name not in bpy.data.objects:
        raise MifOpError("%s reported success and '%s' is not in bpy.data.objects." % (verb, obj.name))
    if obj.type != want_type:
        raise MifOpError("%s made '%s' but its type is %s, not %s." % (verb, obj.name, obj.type,
                                                                       want_type))
    if not any(obj.name in c.objects for c in
               list(bpy.data.collections) + [bpy.context.scene.collection]):
        raise MifOpError("%s made '%s' and it is in NO collection, so it is in no scene - "
                         "invisible, unrendered and absent from the outliner." % (verb, obj.name))
    return {"ok": True, "name": obj.name, "type": obj.type, "collection": coll.name,
            "requestedName": requested,
            "nameWasSuffixed": bool(requested is not None and obj.name != requested),
            "location": rnd(list(obj.location)), "rotation": rnd(list(obj.rotation_euler))}


def op_create_empty(params):
    """An Empty - the most-used object in Blender that this addon could not make.

    WHY IT WAS THE FIRST GAP WORTH CLOSING in object creation. An Empty is what a Track To or Copy
    Location constraint points AT, what a rig is controlled by, what a camera is aimed at, and what
    a set of objects is parented to for one shared pivot. add_constraint and aim_object both take a
    target and neither could create the object people overwhelmingly use as one, so the typed path
    could set up a constraint only against something that already existed.

    params:
      name (str)             default "Empty"
      location / rotation    3-lists
      displayType (str)      PLAIN_AXES | ARROWS | SINGLE_ARROW | CIRCLE | CUBE | SPHERE | CONE
      displaySize (float)    viewport size only - an Empty has no geometry and renders nothing
      collection (str)       link into this collection instead of the scene root
    """
    # PARSED BEFORE ANYTHING IS CREATED. _place used to do this itself, after bpy.data.*.new
    # and after the object was linked, so a malformed vector refused with the object already in
    # the caller's file. Five ops shared _place and so shared the defect.
    _placed = _place_values(params)
    reject_unknown(params, _EMPTY_KEYS, "create_empty")
    if params.get("images") is not None:
        return _create_image_set(params)
    image_spec = params.get("image")
    if image_spec is not None and params.get("displayType") is not None \
            and str(params.get("displayType")).upper() != "IMAGE":
        raise MifOpError("an image goes on an IMAGE empty, and displayType is %r. NOTHING was created."
                         % params.get("displayType"))
    if image_spec is None and any(params.get(k) is not None for k in _IMAGE_EMPTY_KEYS):
        raise MifOpError("%s describe an image empty's drawing; pass image (or images). NOTHING was created."
                         % ", ".join(k for k in _IMAGE_EMPTY_KEYS if params.get(k) is not None))
    img_settings = _image_empty_settings(params) if image_spec is not None else {}
    img = _image_for_empty(image_spec, "image") if image_spec is not None else None
    kind = str(take(params, "displayType", default="IMAGE" if img is not None else "PLAIN_AXES", kind=str)).upper()
    if kind not in _EMPTY_TYPES:
        raise MifOpError("displayType '%s' is not one Blender offers. Valid: %s. NOTHING was "
                         "created." % (kind, ", ".join(_EMPTY_TYPES)))
    size = take_float(params, "displaySize", default=None)

    # AN EMPTY IS AN OBJECT WITH NO DATA - objects.new(name, None). There is no "empty datablock",
    # which is why this cannot follow the create_light shape of make-data-then-object.
    coll = _resolve_collection(params)
    wanted = str(take(params, "name", default="Empty", kind=str))
    obj = bpy.data.objects.new(wanted, None)
    _link_new(obj, coll)
    _place(obj, _placed)
    obj.empty_display_type = kind
    if size is not None:
        obj.empty_display_size = size
    if img is not None:
        _apply_image_empty(obj, img, img_settings)

    out = _created(obj, coll, "EMPTY", "create_empty", wanted)
    out.update({"displayType": obj.empty_display_type,
                "displaySize": round(float(obj.empty_display_size), 6),
                "note": "an Empty has no geometry and renders nothing - displaySize is viewport "
                        "only. It exists to be a target, a parent or a pivot."})
    return out


def _create_image_set(params):
    """create_empty images [{view, file | name, scaleToCube, location}]: one image empty per entry, each rotated to face
    its view, the shared drawing settings on all - front, side, top and back drawings in one call."""
    raw = params.get("images")
    if params.get("image") is not None:
        raise MifOpError("pass image (one empty) OR images (a set placed in their views), not both. NOTHING was created.")
    if not isinstance(raw, list) or not raw:
        raise MifOpError("images is a list of {view, file | name, scaleToCube, location}. NOTHING was created.")
    if params.get("rotation") is not None or params.get("location") is not None:
        raise MifOpError("images places each empty in its view; give a location per entry, not rotation or location "
                         "for the set. NOTHING was created.")
    settings = _image_empty_settings(params)
    size = take_float(params, "displaySize", default=None)
    plan = []
    for i, ent in enumerate(raw):
        if not isinstance(ent, dict) or set(ent) - {"view", "file", "name", "scaleToCube", "location"}:
            raise MifOpError("images[%d] is {view, file | name, scaleToCube, location}, got %r. NOTHING was created."
                             % (i, ent))
        view = str(ent.get("view", "")).upper()
        if view not in _VIEW_ROTATION:
            raise MifOpError("images[%d].view must be one of %s, got %r. NOTHING was created."
                             % (i, ", ".join(_VIEW_ROTATION), ent.get("view")))
        src = {k: ent[k] for k in ("file", "name") if ent.get(k) is not None}
        if len(src) != 1:
            raise MifOpError("images[%d] needs a file (or name) - exactly one. NOTHING was created." % i)
        loc = _vec3({"location": ent.get("location")}, "location", (0.0, 0.0, 0.0))
        plan.append((view, src, take_bool(ent, "scaleToCube", default=False), loc))
    coll = _resolve_collection(params)
    base = str(take(params, "name", default="Reference", kind=str))
    imgs = [_image_for_empty(src, "images[%d]" % i) for i, (_v, src, _s, _l) in enumerate(plan)]
    made = []
    for (view, _src, to_cube, loc), img in zip(plan, imgs):
        wanted = "%s_%s" % (base, view.title())
        obj = bpy.data.objects.new(wanted, None)
        _link_new(obj, coll)
        obj.location = loc
        obj.rotation_euler = tuple(math.radians(a) for a in _VIEW_ROTATION[view])
        _apply_image_empty(obj, img, settings)
        if to_cube:
            obj.empty_display_size = 2.0          # the default cube's edge: the image's longer side spans it
        elif size is not None:
            obj.empty_display_size = size
        row = _created(obj, coll, "EMPTY", "create_empty", wanted)
        row.update({"view": view, "image": img.name, "displaySize": round(float(obj.empty_display_size), 6)})
        made.append(row)
    return {"created": made, "count": len(made),
            "note": ("each drawing faces its view (Blender's numpad FRONT, RIGHT, TOP ...); scaleToCube sizes its "
                     "longer side to 2, the default cube.")}


# WHAT BLENDER'S ADD > FORCE FIELD OPERATOR GIVES A TYPE AND THIS OP'S ROUTE DID NOT. effector_add calls
# BKE_partdeflect_new(type), which sets these per type; forcefield_toggle makes a FORCE field and setting
# .type afterwards changes only the type. So a VORTEX came out POINT-shaped - measured to run away, its
# motes reaching 67 km from the axis by frame 250 of the vortex showcase - and a WIND came out with no
# flow, blowing a constant acceleration instead of a wind. MEASURED 2026-09-12 on 3.6.23, 4.2.17,
# 4.4.0, 5.0.1 and 5.2.1 by diffing an operator-made field of every type against one made this way:
# identical on all five, and every other type already matched.
_FIELD_TYPE_DEFAULTS = {
    "VORTEX": {"shape": "PLANE"},
    "WIND": {"shape": "PLANE", "flow": 1.0, "inflow": 1.0, "wind_factor": 1.0},
    "FLUID_FLOW": {"flow": 1.0, "inflow": 1.0},
    "TEXTURE": {"size": 1.0, "rest_length": 1.0},
}


def op_create_force_field(params):
    """A force field - wind, turbulence, vortex - the thing every simulation here could be scaled
    against and none could create.

    WHY THIS WAS A REAL GAP. `add_cloth`, `add_soft_body` and `set_particles` all read and write
    `effector_weights`, which SCALES how much the fields in a scene affect that simulation. With no
    field in the scene the only effector is gravity, so every one of those parameters was a dial on
    something that could not exist. Found 2026-09-07 by surveying the addon against Blender's
    feature surface: physics had six ops and particles three, and both were thin in the same place.

    A FIELD IS NOT A DATABLOCK AND CANNOT BE MADE BY bpy.data ALONE. `object.field` is None until
    Blender attaches PartialSettings to the object, and there is no `bpy.data.fields.new()`. The
    obvious route, `bpy.ops.object.effector_add`, works headless but names the object after the type
    ("Wind") and places it at the 3D cursor - so the caller's `name` and `location` would be
    suggestions. This creates the object the way create_empty does, with bpy.data and a deterministic
    name, and only then turns it into a field through `forcefield_toggle` under a temp_override.
    Verified on 3.6.23, 4.2.17 LTS, 4.4.0 and 5.0.1: all four attach the field and take every
    setting below, and all four offer the same 14 types.

    THE TYPE LIST IS READ OFF THE RUNNING BLENDER, never quoted from here. A hardcoded list is a
    second source of truth that goes stale the first time Blender adds a field type, and this file
    has already paid for that shape elsewhere.

    params:
      name (str)          default "Field"
      type (str)          WIND | TURBULENCE | VORTEX | FORCE | DRAG | MAGNET | HARMONIC | CHARGE |
                          LENNARDJ | TEXTURE | GUIDE | BOID | FLUID_FLOW - refused if this Blender
                          does not offer it, naming what it does
      location / rotation 3-lists
      collection (str)    link here instead of the scene root
      strength (float)    the field's force. NEGATIVE PULLS instead of pushing.
      flow / noise / seed / falloffPower / maxDistance
    """
    # EVERY PARSE BEFORE THE FIRST WRITE, and this ordering is not stylistic. The first version of
    # this op read its floats AFTER creating and linking the object, so `strength: "abc"` refused
    # with the object already in the caller's file - and blender_version_matrix caught it on the
    # first run: 48 rows of "REFUSALS THAT LEFT SOMETHING BEHIND", twelve per build.
    #
    # It is the exact defect the comment at the top of op_create_empty records - "_place used to do
    # this itself, after bpy.data.*.new and after the object was linked, so a malformed vector
    # refused with the object already in the caller's file. Five ops shared _place and so shared the
    # defect" - written in this file, above this one, and repeated anyway. Reading a comment about a
    # trap is not the same as not falling into it.
    _placed = _place_values(params)
    reject_unknown(params, _FIELD_KEYS, "create_force_field", {
        "object": "this CREATES an empty carrying a field - it does not turn an object you already "
                  "have into one. `object.field` is None until Blender's own forcefield_toggle "
                  "operator runs, which is what this op does for the empty it makes; nothing here "
                  "adds a field to an existing object.",
        "target": "a force field acts on whatever is simulated near it, so there is nothing to point "
                  "it at. Its reach is `maxDistance` and `falloffPower` shapes how it fades.",
        "radius": "spelt maxDistance. A field with no distance reaches forever, so passing one turns "
                  "useMaxDistance on for you rather than storing a limit that does nothing.",
    })
    wanted_type = str(take(params, "type", default="WIND", kind=str)).upper()

    # THE TYPE LIST OFF THE CLASS, not off an instance, which is what lets this validate before
    # anything exists. bpy.types.FieldSettings.bl_rna answers on 3.6.23 and 5.0.1 alike, so the
    # refusal costs no object and the list still comes from the running Blender rather than from a
    # literal here that would go stale the first time Blender adds a field type.
    offered = [i.identifier for i in
               bpy.types.FieldSettings.bl_rna.properties["type"].enum_items]
    if wanted_type not in offered:
        raise MifOpError("field type '%s' is not one this Blender (%s) offers. Valid: %s. NOTHING "
                         "was created." % (wanted_type, bpy.app.version_string, ", ".join(offered)))

    st = take_float(params, "strength", default=None)
    fl = take_float(params, "flow", default=None)
    nz = take_float(params, "noise", default=None)
    # take_int: FieldSettings.seed is an INT, and take_float then int() floored {seed: 3.5} to 3 under ok:true (the
    # 2026-09-21 pack-rule review; measured on 5.2.1). A fraction is refused, as take_int refuses it everywhere.
    sd = take_int(params, "seed", default=None)
    fp = take_float(params, "falloffPower", default=None)
    md = take_float(params, "maxDistance", default=None)

    # THE SHAPE, settable. WIND and VORTEX default to PLANE (see _FIELD_TYPE_DEFAULTS), and a PLANE field
    # measures maxDistance PERPENDICULAR to its own plane - it never stops at its sides. The second seat's
    # row of three wind fields 2.85 apart became three infinite slabs with every subject inside all of
    # them (2026-09-16: guard 1.376 as a row against 14.900 one subject at a time). Reported in the response
    # since the shape field existed; not choosable until now. Read off the class, like the type list, and
    # refused before anything exists. SURFACE and POINTS take their shape from a MESH, and this op makes
    # an empty, so they are refused by name rather than accepted into a field shaped by nothing.
    shape = take(params, "shape", default=None, kind=str)
    if shape is not None:
        shape = str(shape).upper()
        shapes = [i.identifier for i in bpy.types.FieldSettings.bl_rna.properties["shape"].enum_items]
        if shape not in shapes:
            raise MifOpError("field shape '%s' is not one this Blender offers. Valid: %s. NOTHING was "
                             "created." % (shape, ", ".join(shapes)))
        if shape in ("SURFACE", "POINTS"):
            raise MifOpError("field shape '%s' is taken from a mesh's surface or vertices, and this op makes "
                             "an EMPTY - there is no geometry to shape it. Use POINT, LINE or PLANE. NOTHING "
                             "was created." % shape)

    coll = _resolve_collection(params)
    wanted = str(take(params, "name", default="Field", kind=str))
    obj = bpy.data.objects.new(wanted, None)
    _link_new(obj, coll)
    _place(obj, _placed)
    obj.empty_display_type = "SINGLE_ARROW"

    try:
        with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
            bpy.ops.object.forcefield_toggle()
    except Exception as exc:                      # noqa: BLE001
        remove_objects([obj])
        raise MifOpError("could not attach a force field to a new object (%s). NOTHING was left "
                         "behind." % exc)
    if obj.field is None:
        remove_objects([obj])
        raise MifOpError("forcefield_toggle reported no error and left object.field None, so there "
                         "is nothing to configure. NOTHING was left behind.")

    fld = obj.field
    fld.type = wanted_type
    # BLENDER'S OWN DEFAULTS FOR THE TYPE, before the caller's values so an explicit flow still wins.
    applied_defaults = {}
    for attr, value in _FIELD_TYPE_DEFAULTS.get(wanted_type, {}).items():
        if attr == "shape" and shape is not None:
            continue      # an explicit shape wins over the type's default, and is not reported as one
        setattr(fld, attr, value)
        applied_defaults[attr] = value
    if shape is not None:
        fld.shape = shape
    if st is not None:
        fld.strength = st
    if fl is not None:
        fld.flow = fl
    if nz is not None:
        fld.noise = nz
    if sd is not None:
        fld.seed = int(sd)
    if fp is not None:
        fld.falloff_power = fp
    if md is not None:
        # THE FLAG AND THE NUMBER MOVE TOGETHER. distance_max is ignored unless use_max_distance is
        # on, so setting the number alone reads back as the value asked for and affects nothing -
        # the silent no-op this codebase refuses. One parameter, both writes.
        fld.use_max_distance = True
        fld.distance_max = md

    out = _created(obj, coll, "EMPTY", "create_force_field", wanted)
    out.update({"fieldType": fld.type,
                # THE SHAPE DECIDES HOW A VORTEX OR WIND BEHAVES, and it was the field that went wrong.
                "shape": fld.shape,
                "typeDefaultsApplied": applied_defaults or None,
                "strength": round(float(fld.strength), 6),
                "flow": round(float(fld.flow), 6),
                "noise": round(float(fld.noise), 6),
                "seed": int(fld.seed),
                "falloffPower": round(float(fld.falloff_power), 6),
                "useMaxDistance": bool(fld.use_max_distance),
                "maxDistance": round(float(fld.distance_max), 6),
                # RIGID BODIES ARE NAMED because this note used to list only the three ops that carry their
                # own weights, and the second seat read that as the list of listeners - a baked rigid-body
                # scene then measured unmoved and was declined (2026-09-16). They do listen; what hid it was
                # a bake made before the field existed, which bake_physics now reports and rebake clears.
                "note": "a field affects a simulation only as far as that simulation's "
                        "effectorWeights allow - add_cloth, add_soft_body and set_particles all "
                        "carry a gravity weight and the same scaling applies to fields. RIGID BODIES "
                        "listen too, through the rigid body world's own effector weights. A simulation "
                        "baked BEFORE this field existed ignores it - bake_physics with rebake:true. "
                        "Negative strength PULLS."})
    if fld.shape == "PLANE":
        out["shapeNote"] = ("this field is a PLANE: it reaches everything on either side of its own plane, and "
                            "maxDistance (when on) is measured PERPENDICULAR to that plane - it does not stop the "
                            "field at its sides, so fields in a row overlap completely. Pass shape: POINT for a "
                            "field that ends in every direction.")
    return out


def _curve_pointer(params, key, targets, tail):
    """A bevel or taper object by name: a CURVE object, not one of the curves it is set on. None clears it."""
    name = params.get(key)
    if name is None or name == "":
        return None
    if not isinstance(name, str):
        raise MifOpError("%s is an object name, got %r. %s" % (key, name, tail))
    ob = bpy.data.objects.get(name)
    if ob is None or ob.type != "CURVE":
        raise MifOpError("%s %r is %s - it must be a CURVE object. %s"
                         % (key, name, "not an object" if ob is None else "a " + ob.type, tail))
    if any(ob.name == t for t in targets):
        raise MifOpError("%s %r is one of the curves being set - a curve cannot be its own %s. %s"
                         % (key, name, key, tail))
    return ob


def _curve_plan(params, targets, tail):
    """[(attribute, value)] for every curve data key given, all checked before any is written."""
    plan = []
    for key, attr, kind in _CURVE_DATA:
        if key not in params:
            continue
        if attr not in bpy.types.Curve.bl_rna.properties:
            raise MifOpError("this Blender's curves have no %s, so %s would be ignored. %s" % (attr, key, tail))
        if kind == "enum":
            items = [e.identifier for e in bpy.types.Curve.bl_rna.properties[attr].enum_items]
            val = str(params.get(key) or "").upper()
            if val not in items:
                raise MifOpError("%s must be one of this Blender's %s, got %r. %s"
                                 % (key, ", ".join(items), params.get(key), tail))
        elif kind == "object":
            val = _curve_pointer(params, key, targets, tail)
        elif kind is int:
            val = take_int(params, key)
            if not 1 <= val <= 64:
                raise MifOpError("%s must be 1-64, got %d. %s" % (key, val, tail))
        else:
            val = take_float(params, key)
        plan.append((attr, val))
    return plan


def _curve_row(cu):
    row = {}
    for key, attr, kind in _CURVE_DATA:
        if hasattr(cu, attr):
            v = getattr(cu, attr)
            row[key] = (v.name if v is not None else None) if kind == "object" else (
                round(float(v), 6) if kind is float else v)
    return row


def _evaluated_triangles(ob):
    """Triangles in the curve's evaluated mesh - what a convert to mesh would give."""
    deps = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(deps)
    me = ev.to_mesh()
    try:
        if me is None:
            return 0
        me.calc_loop_triangles()
        return len(me.loop_triangles)
    finally:
        ev.to_mesh_clear()


def op_set_curve(params):
    """Write a curve's data settings - bevel and taper objects, fill, resolution - on one curve or many at once.

    The curve counterpart of set_camera and set_light. One of object, objects [...] or collection (every CURVE object
    in it). bevelMode ROUND | OBJECT | PROFILE, bevelObject and taperObject (a CURVE object's name; null clears it),
    taperRadiusMode, fillMode (enum names from this Blender's own lists), resolutionU and bevelResolution (1-64),
    bevelDepth, extrude, offset. Every value is checked before anything is written, and a pointer to an object that
    does not exist, is not a curve, or is one of the curves being set is refused. Reply: each curve's values before and
    after; triangles true adds the evaluated triangle count before and after (what converting to a mesh would give).
    """
    reject_unknown(params, _SETCURVE_KEYS, "set_curve")
    given = [k for k in ("object", "objects", "collection") if params.get(k) is not None]
    if params.get("name") is not None and "object" not in given:
        given.append("object")
    if len(given) != 1:
        raise MifOpError("pass one of object, objects or collection (got %s). NOTHING was changed."
                         % (", ".join(given) or "none"))
    tail = "NOTHING was changed."
    if given[0] == "object":
        obs = [get_object(take(params, "object", "name", kind=str))]
    elif given[0] == "objects":
        raw = params["objects"]
        if not isinstance(raw, (list, tuple)) or not raw or any(not isinstance(n, str) for n in raw):
            raise MifOpError("objects is a list of curve object names. NOTHING was changed.")
        obs = [get_object(n) for n in dict.fromkeys(raw)]
    else:
        coll = bpy.data.collections.get(str(params["collection"]))
        if coll is None:
            raise MifOpError("no collection named %r. NOTHING was changed." % params["collection"])
        obs = [o for o in coll.all_objects if o.type == "CURVE"]
        if not obs:
            raise MifOpError("collection %r holds no CURVE object. NOTHING was changed." % coll.name)
    bad = [o.name for o in obs if o.type != "CURVE"]
    if bad:
        raise MifOpError("%s %s not a CURVE object. NOTHING was changed." % (", ".join(bad), "is" if len(bad) == 1
                                                                            else "are"))
    for o in obs:
        require_editable(o, "set curve settings on")
    plan = _curve_plan(params, [o.name for o in obs], tail)
    tris = take_bool(params, "triangles", default=False)
    if not plan:
        raise MifOpError("nothing to set: pass at least one of %s. NOTHING was changed."
                         % ", ".join(k for k, _a, _k in _CURVE_DATA))
    rows = []
    shared = {}
    for o in obs:
        shared.setdefault(o.data.name, []).append(o.name)
    for o in obs:
        row = {"object": o.name, "before": _curve_row(o.data)}
        if tris:
            row["trianglesBefore"] = _evaluated_triangles(o)
        rows.append(row)
    for o in obs:
        for attr, val in plan:
            setattr(o.data, attr, val)
    bpy.context.view_layer.update()
    for row, o in zip(rows, obs):
        row["after"] = _curve_row(o.data)
        if tris:
            row["trianglesAfter"] = _evaluated_triangles(o)
    out = {"curves": len(rows), "set": sorted(k for k, a, _kd in _CURVE_DATA if a in {a2 for a2, _v in plan}),
           "rows": rows}
    multi = {d: n for d, n in shared.items() if len(n) > 1}
    if multi:
        out["sharedDataNote"] = ("these objects share one curve datablock, so one write reached all of them: %s"
                                 % "; ".join("%s (%s)" % (d, ", ".join(n)) for d, n in multi.items()))
    return out


def op_create_curve(params):
    """A curve - a path to follow, a profile to bevel, or a cable.

    A curve is what a Follow Path constraint needs, what a bevel turns into a pipe or a cable, and
    what text can be laid along. add_constraint accepts FOLLOW_PATH and nothing here could make the
    one object it requires.

    THE SPLINE TYPE DECIDES WHAT THE POINTS MEAN. POLY and NURBS points live in spline.points with
    a 4th weight component; BEZIER points live in spline.bezier_points and have handles instead.
    They are different collections with different lengths, so the type is chosen first and the
    points are added to the right one - mixing them up produces a curve with no points and no error.

    params:
      name (str)
      points (list)          [[x,y,z], ...] - at least 2. Required.
      splineType (str)       POLY | BEZIER | NURBS. Default POLY.
      cyclic (bool)          close the loop
      bevelDepth (float)     round profile radius - this is what makes a cable out of a line
      bevelResolution (int)
      extrude (float)        flat extrusion, an alternative to bevel
      resolution (int)       preview resolution per segment
      usePath (bool)         default true - a Follow Path constraint does NOTHING without it
      radii (list)           one radius per point, scaling the bevel there - a stem or branch that
                             TAPERS is radii [1.0, ..., 0.1]. Read back as `radii`.
      bevelObject / taperObject (str)   a CURVE object's name: its shape is the cross-section (bevel_mode OBJECT is
                             set with it) or the taper along the length (2026-10-04)
      location / rotation / collection
    """
    # PARSED BEFORE ANYTHING IS CREATED. _place used to do this itself, after bpy.data.*.new
    # and after the object was linked, so a malformed vector refused with the object already in
    # the caller's file. Five ops shared _place and so shared the defect.
    _placed = _place_values(params)
    reject_unknown(params, _CURVE_KEYS, "create_curve")
    raw = params.get("points")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise MifOpError("'points' is required and needs at least 2 points, each [x,y,z]. A curve "
                         "with fewer has no length and nothing can follow it. NOTHING was created.")
    pts = []
    for i, p in enumerate(raw):
        if not isinstance(p, (list, tuple)) or len(p) < 3:
            raise MifOpError("points[%d] must be [x,y,z], got %r. NOTHING was created." % (i, p))
        # finite_floats, NOT a bare float(). Measured on 5.0.1, 2026-09-20: [[NaN, 0, 0], [1, 0, 0]] made a
        # curve with a NaN point under ok:true, and [["abc", 0, 0], ...] died as a raw "ValueError: could
        # not convert string to float" with a traceback - the one vector parser in this file the finite
        # guard never reached.
        try:
            pts.append(finite_floats(p[:3], "points[%d]" % i))
        except MifOpError as exc:
            raise MifOpError("%s NOTHING was created." % exc)

    # PER-POINT RADIUS - what makes a stem, a branch, a horn or a cable TAPER. Blender scales the bevel by
    # each point's radius and nothing here could set one, so every curve was a tube of one width; the only
    # route was a geometry-nodes group. Validated before anything exists: one finite, non-negative number
    # per point.
    radii = params.get("radii")
    if radii is not None:
        if not isinstance(radii, (list, tuple)) or len(radii) != len(pts):
            raise MifOpError("radii is one number per point - %d points were given, and radii was %r. NOTHING "
                             "was created." % (len(pts), radii))
        try:
            radii = finite_floats(radii, "radii")
        except MifOpError as exc:
            raise MifOpError("%s NOTHING was created." % exc)
        if any(r < 0 for r in radii):
            raise MifOpError("radii must be 0 or more - a negative radius turns the bevel inside out. Got %r. "
                             "NOTHING was created." % (radii,))

    stype = str(take(params, "splineType", default="POLY", kind=str)).upper()
    if stype not in ("POLY", "BEZIER", "NURBS"):
        raise MifOpError("splineType must be POLY, BEZIER or NURBS, got '%s'. NOTHING was created."
                         % stype)

    # DIMENSIONS IS VALIDATED BEFORE THE DATABLOCK EXISTS, and this is the third time this file has
    # learnt the same lesson - see the align block in op_create_text below, which records it for
    # `align` and again for the font path. The assignment used to sit one line AFTER
    # bpy.data.curves.new, so Blender's own enum check fired on an object that already existed and
    # the op refused with a TypeError having left an orphaned curve behind.
    #
    # WHY THAT IS WORSE THAN AN UNTIDY FILE: by ledger B6 an orphan keeps its NAME reserved. A
    # caller who mistypes, reads the error and retries gets "<name>.001" with nameWasSuffixed:true,
    # and every later lookup by the name they actually asked for finds nothing. The old refusal did
    # not even say what had happened, because the message was Blender's rather than ours.
    #
    # Found 2026-09-15 by the second seat, reproduced here on all five builds - 3.6, 4.2, 4.4, 5.0
    # and 5.2 - which is what distinguishes it from a build-specific quirk. splineType and align
    # were already pre-validated and leak nothing; dimensions was the one that was missed.
    dims = str(take(params, "dimensions", default="3D", kind=str)).upper()
    valid_dims = {i.identifier for i in bpy.types.Curve.bl_rna.properties["dimensions"].enum_items}
    if dims not in valid_dims:
        raise MifOpError("dimensions must be one of %s, got '%s'. NOTHING was created."
                         % (", ".join(sorted(valid_dims)), dims))
    # AND THE REST OF THE SETTINGS, FOR THE SAME REASON - the fourth time. cyclic, bevelDepth, extrude,
    # bevelResolution, resolution and usePath were read AFTER bpy.data.curves.new: {cyclic: "flase"} was
    # refused with the curve datablock left behind holding the name (measured on 5.0.1, 2026-09-21), so the
    # retry the refusal invites came back as "<name>.001" - ledger B6.
    try:
        cyclic = take_bool(params, "cyclic", default=False)
        use_path = take_bool(params, "usePath", default=True)
        shape_floats = [(attr, take_float(params, key, default=None))
                        for key, attr in (("bevelDepth", "bevel_depth"), ("extrude", "extrude"))]
        shape_ints = [(attr, take_int(params, key, default=None))
                      for key, attr in (("bevelResolution", "bevel_resolution"), ("resolution", "resolution_u"))]
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was created." % str(exc).rstrip("."))
    bevel_ob = _curve_pointer(params, "bevelObject", [], "NOTHING was created.")
    taper_ob = _curve_pointer(params, "taperObject", [], "NOTHING was created.")

    coll = _resolve_collection(params)
    wanted = str(take(params, "name", default="Curve", kind=str))
    data = bpy.data.curves.new(wanted, type="CURVE")
    if bevel_ob is not None:
        if hasattr(data, "bevel_mode"):
            data.bevel_mode = "OBJECT"
        data.bevel_object = bevel_ob
    if taper_ob is not None:
        data.taper_object = taper_ob
    data.dimensions = dims
    spline = data.splines.new(stype)
    if stype == "BEZIER":
        # add() is relative to the ONE point a new spline already has, for both collections - so
        # this adds len-1 and then writes all of them, rather than adding len and leaving a stray.
        spline.bezier_points.add(len(pts) - 1)
        for i, (bp, xyz) in enumerate(zip(spline.bezier_points, pts)):
            bp.co = xyz
            bp.handle_left_type = bp.handle_right_type = "AUTO"
            if radii is not None:
                bp.radius = radii[i]
    else:
        spline.points.add(len(pts) - 1)
        for i, (sp, xyz) in enumerate(zip(spline.points, pts)):
            sp.co = (xyz[0], xyz[1], xyz[2], 1.0)      # POLY/NURBS points are 4-component
            if radii is not None:
                sp.radius = radii[i]
    spline.use_cyclic_u = cyclic

    for attr, v in shape_floats + shape_ints:
        if v is not None:
            setattr(data, attr, v)
    # DEFAULT TRUE, and not a detail: a Follow Path constraint evaluates to NOTHING when use_path is
    # off, with the constraint and the curve both reading back perfectly.
    data.use_path = use_path

    obj = bpy.data.objects.new(data.name, data)
    _link_new(obj, coll)
    _place(obj, _placed)

    # THE POINT COUNT IS THE POSTCONDITION, taken from the collection the type actually uses. A
    # spline built into the wrong collection has zero points and raises nothing.
    made = len(spline.bezier_points) if stype == "BEZIER" else len(spline.points)
    if made != len(pts):
        raise MifOpError("asked for %d points and the %s spline holds %d afterwards."
                         % (len(pts), stype, made))
    out = _created(obj, coll, "CURVE", "create_curve", wanted)
    if radii is not None:
        # READ BACK off the points, not echoed.
        held = spline.bezier_points if stype == "BEZIER" else spline.points
        out["radii"] = [round(float(p.radius), 6) for p in held]
    out.update({"splineType": stype, "pointCount": made, "cyclic": bool(spline.use_cyclic_u),
                "bevelDepth": round(float(data.bevel_depth), 6),
                "bevelObject": data.bevel_object.name if data.bevel_object is not None else None,
                "taperObject": data.taper_object.name if data.taper_object is not None else None,
                "usePath": bool(data.use_path),
                "note": (None if data.use_path else
                         "usePath is OFF, so a Follow Path constraint against this curve will "
                         "evaluate to nothing while the constraint and the curve both read back "
                         "correctly.")})
    return out


_TEXT_RESOLUTION, _TEXT_BEVEL_RESOLUTION = 6, 1


def op_create_text(params):
    """A text object - titles, labels, mograph, and anything with words in the render.

    params:
      name (str)
      body (str)            the text itself. Required - an empty text object renders nothing.
      size (float)          font size
      extrude (float)       depth, which is what turns flat text into 3D
      bevelDepth (float)
      align (str)           horizontal: LEFT | CENTER | RIGHT | JUSTIFY | FLUSH
      alignY (str)          vertical: TOP | TOP_BASELINE | CENTER | BOTTOM | BOTTOM_BASELINE
      font (str)            PATH to a .ttf/.otf. Left out, the text is in Blender's built-in face -
                            which is what every text object this addon made before 2026-09-07 was
                            stuck in, since nothing could set one.
      location / rotation / collection
    """
    # PARSED BEFORE ANYTHING IS CREATED. _place used to do this itself, after bpy.data.*.new
    # and after the object was linked, so a malformed vector refused with the object already in
    # the caller's file. Five ops shared _place and so shared the defect.
    _placed = _place_values(params)
    reject_unknown(params, _TEXT_KEYS, "create_text")
    body = take(params, "body", required=True, kind=str)
    if not str(body):
        raise MifOpError("'body' is empty, and a text object with no body renders nothing while "
                         "existing perfectly. NOTHING was created.")

    # VALIDATED OFF THE TYPE, BEFORE THE DATABLOCK EXISTS. This read the enum off the instance, so
    # it could only run after bpy.data.curves.new - and a bad `align` then refused with "NOTHING was
    # created" having already left an orphaned FONT curve in the file. bl_rna is on the type as well
    # as the instance, and asking the type costs nothing and needs nothing to exist.
    aligns = {}
    for key, attr in (("align", "align_x"), ("alignY", "align_y")):
        v = take(params, key, kind=str)
        if v is None:
            continue
        valid = {i.identifier for i in bpy.types.TextCurve.bl_rna.properties[attr].enum_items}
        if str(v).upper() not in valid:
            raise MifOpError("%s '%s' is not one this Blender offers. Valid: %s. NOTHING was "
                             "created." % (key, v, ", ".join(sorted(valid))))
        aligns[attr] = str(v).upper()

    # THE NUMBERS, the one read the align lesson above did not reach: size, extrude and bevelDepth were
    # parsed after bpy.data.curves.new, so {size: NaN} was refused with the FONT curve left behind holding
    # its name (measured on 5.0.1, 2026-09-21).
    try:
        sizes = [(attr, take_float(params, key, default=None))
                 for key, attr in (("size", "size"), ("extrude", "extrude"), ("bevelDepth", "bevel_depth"))]
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was created." % str(exc).rstrip("."))
    coll = _resolve_collection(params)

    # THE FONT IS LOADED LAST OF ALL THE CHECKS, before the curve exists and after every refusal that can still come:
    # loading it is itself a write (a font datablock), and run before the number parse and the collection lookup it
    # was left in the file by a later refusal that said NOTHING was created (found 2026-10-25).
    font = _resolve_font(params, "create_text")
    wanted = str(take(params, "name", default="Text", kind=str))
    data = bpy.data.curves.new(wanted, type="FONT")
    data.body = str(body)
    if font is not None:
        data.font = font
    for attr, v in sizes:
        if v is not None:
            setattr(data, attr, v)
    for attr, value in aligns.items():
        setattr(data, attr, value)
    # A GLYPH'S CURVES AT A QUARTER TURN IN 6 STEPS, a bevel in one: Blender's 12 and 4 put 8,180 triangles on "Mud 8" at
    # 3 m, past gameReady's number for the size, for 1.7 mm of sag on a letter 1 m high. set_curve raises either.
    data.resolution_u = _TEXT_RESOLUTION
    data.bevel_resolution = _TEXT_BEVEL_RESOLUTION

    obj = bpy.data.objects.new(data.name, data)
    _link_new(obj, coll)
    _place(obj, _placed)

    out = _created(obj, coll, "FONT", "create_text", wanted)
    out.update(_font_fields(data))
    out.update({"body": data.body, "size": round(float(data.size), 6),
                "extrude": round(float(data.extrude), 6),
                "bevelDepth": round(float(data.bevel_depth), 6),
                "align": data.align_x, "alignY": data.align_y,
                "note": ("extrude is 0, so this text is a flat plane with no thickness - which is "
                         "correct for a 2D title and wrong for anything meant to catch a light.")
                if not data.extrude else None})
    return out


def op_set_text(params):
    """Change an existing text object - the half that did not exist, so text was WRITE-ONCE.

    create_text could set a body, a size and an alignment and nothing could change any of them
    afterwards. object_info REPORTS `body` for a FONT object, so the string was readable and
    unwritable - the plain read/write asymmetry this repo chases everywhere else, sitting on the one
    object type whose whole purpose is the words on it. Re-labelling a sign meant deleting the
    object and rebuilding it, which loses its transform, its material and anything parented to it.

    EVERY FIELD IS OPTIONAL AND ONLY WHAT WAS NAMED IS TOUCHED, so this is safe to call with one
    key. `changed` names exactly which fields moved, and `changedAnything` is false when a call
    asked for values the object already had - a caller polling this in a loop can tell a no-op from
    a write, which a bare ok:true cannot.

    params:
      object (str, required)   the FONT object
      body / size / extrude / bevelDepth / align / alignY / font   as create_text
    """
    reject_unknown(params, _SETTEXT_KEYS, "set_text", {
        "text": "spelt body - it is the Blender field name, and create_text uses the same word.",
        "content": "see 'text'.",
        "fontSize": "spelt size.",
        "material": "assign it with set_material_slots; a text object takes material slots like any "
                    "other object.",
        "location": "text is an OBJECT like any other - move it with transform_object rather than "
                    "through the thing that edits its letters.",
        "rotation": "see 'location'.",
    })
    obj = get_object(take(params, "object", "name", required=True))
    if obj.type != "FONT":
        raise MifOpError("'%s' is a %s, not a text object. create_text makes FONT objects; this "
                         "edits them. NOTHING was changed." % (obj.name, obj.type))
    data = obj.data

    # EVERY VALUE PARSED AND VALIDATED BEFORE THE FIRST WRITE, so a call naming a good body and a
    # bad align does not leave the body changed and then refuse. create_force_field paid for this
    # ordering on 48 matrix rows.
    aligns = {}
    for key, attr in (("align", "align_x"), ("alignY", "align_y")):
        v = take(params, key, default=None, kind=str)
        if v is None:
            continue
        valid = {i.identifier for i in bpy.types.TextCurve.bl_rna.properties[attr].enum_items}
        if str(v).upper() not in valid:
            raise MifOpError("%s '%s' is not one this Blender offers. Valid: %s. NOTHING was "
                             "changed." % (key, v, ", ".join(sorted(valid))))
        aligns[attr] = str(v).upper()
    body = take(params, "body", default=None, kind=str)
    if body is not None and not str(body):
        raise MifOpError("'body' is empty, and a text object with no body renders nothing while "
                         "existing perfectly. NOTHING was changed.")
    floats = {}
    for key, attr in (("size", "size"), ("extrude", "extrude"), ("bevelDepth", "bevel_depth")):
        v = take_float(params, key, default=None)
        if v is not None:
            floats[attr] = v
    font = _resolve_font(params, "set_text")

    before = {"body": data.body, "size": round(float(data.size), 6),
              "extrude": round(float(data.extrude), 6),
              "bevelDepth": round(float(data.bevel_depth), 6),
              "align": data.align_x, "alignY": data.align_y,
              "font": getattr(getattr(data, "font", None), "name", None)}

    if body is not None:
        data.body = str(body)
    for attr, v in floats.items():
        setattr(data, attr, v)
    for attr, v in aligns.items():
        setattr(data, attr, v)
    if font is not None:
        data.font = font

    after = {"body": data.body, "size": round(float(data.size), 6),
             "extrude": round(float(data.extrude), 6),
             "bevelDepth": round(float(data.bevel_depth), 6),
             "align": data.align_x, "alignY": data.align_y,
             "font": getattr(getattr(data, "font", None), "name", None)}
    changed = sorted(k for k in after if before[k] != after[k])

    out = {"object": obj.name, "dataName": data.name,
           "before": before, "after": after,
           "changed": changed, "changedAnything": bool(changed)}
    out.update(_font_fields(data))
    if not data.extrude:
        out["note"] = ("extrude is 0, so this text is a flat plane with no thickness - correct for "
                       "a 2D title and wrong for anything meant to catch a light.")
    return out


def op_create_armature(params):
    """An armature, with its bones - without which the whole rigging family could only EDIT.

    ops_rig has twelve ops and not one of them creates an armature: list_bones, rename_bones,
    set_bone_pose, the vertex-group and weight ops all operate on a rig that already exists. So
    nothing could be rigged from scratch through the typed path.

    BONES CAN ONLY BE CREATED IN EDIT MODE, which is the whole difficulty. armature.edit_bones does
    not exist outside it, so this switches mode, builds, and switches back - and RESTORING THE MODE
    IS A POSTCONDITION, not a courtesy: being left in edit mode strands every op that follows,
    which is the same failure create_primitive forces enter_editmode off to avoid.

    params:
      name (str)
      bones (list)          [{"name","head":[x,y,z],"tail":[x,y,z],"parent":"...","connect":bool,
                              "rotationMode":"XYZ"}] - Parents must appear BEFORE their children. A key a bone
                              does not take is refused. rotationMode is the POSE bone's (default QUATERNION,
                              which ignores a euler).
      displayType (str)     OCTAHEDRAL | STICK | BBONE | ENVELOPE | WIRE
      showInFront (bool)    draw the rig over the mesh - default true, because a rig inside a body
                            is invisible and looks like it was never created
      location / rotation / collection
    """
    # PARSED BEFORE ANYTHING IS CREATED. _place used to do this itself, after bpy.data.*.new
    # and after the object was linked, so a malformed vector refused with the object already in
    # the caller's file. Five ops shared _place and so shared the defect.
    _placed = _place_values(params)
    reject_unknown(params, _ARM_KEYS, "create_armature")
    if params.get("metarig") is not None:
        from .ops_rigify import create_metarig
        return create_metarig(params)
    for key in ("height", "scale", "enableRigify"):
        if params.get(key) is not None:
            raise MifOpError("%s sizes or enables a METARIG - pass metarig too. NOTHING was created." % key)
    bones = params.get("bones") or []
    if not isinstance(bones, (list, tuple)):
        raise MifOpError("'bones' must be a list of {name, head, tail}. NOTHING was created.")
    # VALIDATED IN FULL BEFORE ANY MODE CHANGE. A refusal partway through would leave an armature
    # with half its bones AND Blender in edit mode.
    seen = set()
    # EVERY END PARSED HERE, FINITE, and the write loop below uses these - not a second float() of the raw
    # value. It used a bare float(): [NaN, 0, 0] made an armature whose bone head is nan under ok:true, and
    # ["abc", 0, 0] raised a raw ValueError AFTER the armature object existed, leaving it in the file
    # (measured on 5.0.1, 2026-09-20; audit_unguarded_numbers' provenance pass).
    ends = {}
    connects = {}
    for i, b in enumerate(bones):
        # A STRING, not merely truthy. A dict name passed this guard and then reached a set()
        # of seen names, raising TypeError: unhashable type - a raw exception out of the op.
        if not isinstance(b, dict) or not isinstance(b.get("name"), str) or not b["name"]:
            raise MifOpError("bones[%d] needs a 'name' as text, got %r. NOTHING was created."
                             % (i, b.get("name") if isinstance(b, dict) else b))
        for end in ("head", "tail"):
            v = b.get(end)
            if not isinstance(v, (list, tuple)) or len(v) < 3:
                raise MifOpError("bones[%d]['%s'] must be [x,y,z], got %r. NOTHING was created."
                                 % (i, end, v))
            try:
                ends[(i, end)] = finite_floats(list(v)[:3], "bones[%d].%s" % (i, end))
            except MifOpError as exc:
                raise MifOpError("%s NOTHING was created." % exc)
        # EVERY KEY IN A BONE IS ONE THIS OP READS. Anything else used to be dropped without a word - see _BONE_KEYS.
        extra = sorted(k for k in b if k not in _BONE_KEYS)
        if extra:
            raise MifOpError("bones[%d] ('%s') has %s, which a bone here does not take - it takes %s. NOTHING was "
                             "created." % (i, b["name"], ", ".join(repr(k) for k in extra), ", ".join(_BONE_KEYS)))
        if b.get("rotationMode") is not None:
            modes = [m.identifier for m in bpy.types.PoseBone.bl_rna.properties["rotation_mode"].enum_items]
            if not isinstance(b["rotationMode"], str) or b["rotationMode"].upper() not in modes:
                raise MifOpError("bones[%d] ('%s') rotationMode %r is not one this Blender offers: %s. NOTHING was "
                                 "created." % (i, b["name"], b["rotationMode"], ", ".join(modes)))
        if b["name"] in seen:
            raise MifOpError("bones[%d] repeats the name '%s'. NOTHING was created."
                             % (i, b["name"]))
        parent = b.get("parent")
        if parent and parent not in seen:
            raise MifOpError("bones[%d] ('%s') names parent '%s', which is not defined ABOVE it. "
                             "Parents must come first. NOTHING was created."
                             % (i, b["name"], parent))
        # take_bool, HERE. The write loop used bool(b.get("connect")), and bool("false") is True: a child
        # placed at [1, 0, 0] under {connect: "false"} was CONNECTED, its head moved onto the parent's tail
        # (measured on 5.0.1, 2026-09-21). A word take_bool cannot read is refused before anything exists.
        try:
            connects[i] = take_bool(b, "connect", default=False)
        except MifOpError as exc:
            raise MifOpError("bones[%d] ('%s'): %s NOTHING was created." % (i, b["name"], exc))
        seen.add(b["name"])

    mode_before = bpy.context.object.mode if bpy.context.object else "OBJECT"
    if mode_before != "OBJECT":
        raise MifOpError("Blender is in %s mode. Creating an armature has to switch modes, and "
                         "doing that from anything but OBJECT mode would drop whatever is being "
                         "edited. NOTHING was created." % mode_before)

    # BEFORE THE ARMATURE EXISTS. It was read on the line after _place, so {showInFront: "flase"} was refused
    # with the armature object already linked into the scene (measured on 5.0.1, 2026-09-21).
    try:
        show_front = take_bool(params, "showInFront", default=True)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was created." % exc)
    # displayType TOO, and against the TYPE's enum so no armature has to exist to ask. It was read after
    # bpy.data.armatures.new: the enum refusal removed the data again, but take(kind=str) refusing a
    # non-string ({displayType: 5}) left the armature datablock behind (audit_mutate_then_deny's DEBRIS rule,
    # measured on 5.0.1, 2026-09-21).
    disp = take(params, "displayType", kind=str)
    if disp:
        valid = {i.identifier for i in bpy.types.Armature.bl_rna.properties["display_type"].enum_items}
        if str(disp).upper() not in valid:
            raise MifOpError("displayType '%s' is not one this Blender offers. Valid: %s. NOTHING "
                             "was created." % (disp, ", ".join(sorted(valid))))
    coll = _resolve_collection(params)
    wanted = str(take(params, "name", default="Armature", kind=str))
    data = bpy.data.armatures.new(wanted)
    if disp:
        data.display_type = str(disp).upper()
    obj = bpy.data.objects.new(data.name, data)
    _link_new(obj, coll)
    _place(obj, _placed)
    obj.show_in_front = show_front

    made = []
    if bones:
        snap = selection_snapshot()
        try:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.mode_set(mode="EDIT")
            ebs = data.edit_bones
            for i, b in enumerate(bones):
                eb = ebs.new(str(b["name"]))
                eb.head = ends[(i, "head")]
                eb.tail = ends[(i, "tail")]
                if b.get("parent"):
                    eb.parent = ebs[str(b["parent"])]
                    eb.use_connect = connects[i]
                made.append(eb.name)
        finally:
            # ALWAYS, and this is the postcondition rather than the tidy-up. Left in edit mode,
            # every op after this one fails on an editor nobody asked to be in.
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                pass
            selection_restore(snap)

    mode_after = bpy.context.object.mode if bpy.context.object else "OBJECT"
    if mode_after != "OBJECT":
        raise MifOpError("created '%s' but Blender is left in %s mode, which strands every op "
                         "after this one." % (obj.name, mode_after))
    # BONES COUNTED FROM data.bones, NOT from the edit_bones list built above. edit_bones only
    # exist in edit mode; the real bones appear when it is left, and a bone that failed to survive
    # that transition would still be in the list this op made.
    if len(data.bones) != len(bones):
        raise MifOpError("asked for %d bone(s) and the armature holds %d after leaving edit mode."
                         % (len(bones), len(data.bones)))
    # THE POSE BONES EXIST ONLY NOW, out of edit mode - so the rotation modes are written here, and read back below.
    for b in bones:
        if b.get("rotationMode") is not None and obj.pose is not None:
            obj.pose.bones[str(b["name"])].rotation_mode = str(b["rotationMode"]).upper()

    out = _created(obj, coll, "ARMATURE", "create_armature", wanted)
    out.update({"bones": [b.name for b in data.bones], "boneCount": len(data.bones),
                "rotationModes": ({pb.name: pb.rotation_mode for pb in obj.pose.bones} if obj.pose is not None else {}),
                "displayType": data.display_type, "showInFront": bool(obj.show_in_front),
                "modeAfter": mode_after,
                "note": (None if bones else
                         "created with NO bones - an armature with none deforms nothing and shows "
                         "nothing in the viewport. Add them with the bones parameter.")})
    return out


_LATTICE_KEYS = {"name", "resolution", "pointsU", "pointsV", "pointsW", "interpolation",
                 "useOutside", "location", "rotation", "scale", "collection", "fitTo", "margin"}
# THE TYPES A LATTICE MODIFIER DEFORMS - what fitTo accepts, since it adds one to each.
_LATTICE_DEFORMABLE = ("MESH", "CURVE", "SURFACE", "FONT", "LATTICE")


def _lattice_fit(params):
    """fitTo / margin: (objects, location, scale, bounds) of the world box round them, or None. Parsed before anything
    exists - see op_create_lattice."""
    raw = params.get("fitTo")
    if raw is None:
        if params.get("margin") is not None:
            raise MifOpError("margin goes with fitTo. NOTHING was created.")
        return None
    if not isinstance(raw, (list, tuple)) or not raw or not all(isinstance(n, str) for n in raw):
        raise MifOpError("fitTo is a non-empty list of object names. NOTHING was created.")
    placed = [k for k in ("location", "rotation", "scale") if params.get(k) is not None]
    if placed:
        raise MifOpError("fitTo places and sizes the lattice from the objects' box, so %s cannot go with it. NOTHING "
                         "was created." % ", ".join(placed))
    margin = take_float(params, "margin", default=0.0)
    if margin < 0.0:
        raise MifOpError("margin is world units added on every side and must be 0 or more, got %g. NOTHING was "
                         "created." % margin)
    objs = []
    for n in dict.fromkeys(raw):
        ob = bpy.data.objects.get(n)
        if ob is None:
            raise MifOpError("fitTo names %r, and there is no object of that name. NOTHING was created." % n)
        if ob.type not in _LATTICE_DEFORMABLE:
            raise MifOpError("fitTo names '%s', a %s - a Lattice modifier deforms %s only. NOTHING was created."
                             % (n, ob.type, ", ".join(_LATTICE_DEFORMABLE)))
        objs.append(ob)
    from mathutils import Vector   # noqa: PLC0415 - local, as op_transform_objects imports it
    corners = [ob.matrix_world @ Vector(c) for ob in objs for c in ob.bound_box]
    lo = [min(c[a] for c in corners) - margin for a in range(3)]
    hi = [max(c[a] for c in corners) + margin for a in range(3)]
    size = [hi[a] - lo[a] for a in range(3)]
    if min(size) <= 1e-9:
        raise MifOpError("the box round %s is flat on %s and margin is %g - a lattice with no thickness deforms nothing. "
                         "Pass a margin. NOTHING was created."
                         % (", ".join(o.name for o in objs), "XYZ"[size.index(min(size))], margin))
    return objs, [(lo[a] + hi[a]) / 2.0 for a in range(3)], size, (lo, hi, margin)
# "useColorRamp" IS DELIBERATELY ABSENT. It was listed here while the handler ignored it, so
# reject_unknown ACCEPTED it and nothing happened - the same declared-and-ignored mistake as
# set_material_texture's "slot" a few hours earlier, caught by the same audit. A caller
# passing it now gets a refusal naming what this op does take.
_TEXTURE_KEYS = {"name", "type", "image", "intensity", "contrast",
                 # an image FILE loaded as the datablock (2026-10-04), and the colour space it is read in
                 "file", "colorspace"}


_GN_KEYS = {"name", "location", "rotation", "scale", "collection"}

# THE POPULATING OP ALREADY EXISTS, which is the entire reason these two are built and the other six
# object types are not. Eight of Blender's seventeen object types were declined on 2026-09-07 because
# each creates an EMPTY CONTAINER whose populating op is missing - a metaball with no elements, a
# volume with no grids - and shipping one would be a capability that looks real and is not.
#
# CURVES and POINTCLOUD are the exception, and it was MEASURED rather than argued. On 3.6.23, 4.4.0
# and 5.0.1 alike, an empty hair_curves or pointcloud object accepts a NODES modifier, accepts a
# generating node group, and evaluates to real geometry - 32 points and 1 curve, and 16 points,
# read off the EVALUATED object through the depsgraph rather than off the node tree, because a tree
# agreeing with itself is not evidence. create_node_group and assign_node_group have been in this
# addon since before either of these existed, so the thing that fills them is already here.
_GN_NOTE = ("this object is EMPTY and renders NOTHING until a geometry node group generates into "
            "it - that is what this object type IS, not a failure of this op. Build one with "
            "create_node_group and attach it with assign_node_group; a group whose output comes "
            "from a primitive (Mesh Grid into Mesh to Points, Curve Circle) produces geometry on "
            "3.6, 4.4 and 5.0 alike, measured.")


_GN_NOTES = {
    "points": "there is no way to give this object points directly, and that is deliberate. "
              "Points come from a geometry node group - see the note on the response.",
    "count": "see 'points'.",
    "radius": "a hair curve's radius is a point ATTRIBUTE written by the node group, not a "
              "property of the object.",
    "nodeGroup": "attach one with assign_node_group after this returns. Doing it here would mean "
                 "this op either created the group too or failed on a missing one, and both are "
                 "jobs that already have their own op.",
    "material": "assign it with set_material_slots once the object exists.",
}


def _gn_object(params, kind, verb, default_name):
    """Shared body for the two node-group-driven object types.

    ONE FUNCTION BECAUSE THEY DIFFER IN A DATABLOCK COLLECTION AND NOTHING ELSE. Writing them
    separately would be the parallel-systems mistake this repo collapses everywhere - six vector
    parsers, twenty-two scratch filters - and the two would drift the first time either was fixed.
    """
    # PARSED BEFORE ANYTHING IS CREATED, the ordering create_force_field had to learn the hard way
    # after leaving objects behind on a refusal in 48 matrix rows.
    _placed = _place_values(params)
    # SCALE TOO, which the line above does not parse: it was read after the object was linked, so
    # {scale: [NaN, 1, 1]} was refused with the object in the scene (measured on 5.0.1, 2026-09-21).
    try:
        want_scale = _vec3(params, "scale", (1.0, 1.0, 1.0)) if "scale" in params else None
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was created." % str(exc).rstrip("."))
    coll = _resolve_collection(params)
    collection = getattr(bpy.data, kind, None)
    if collection is None:
        # A BUILD THAT LACKS THE DATABLOCK ENTIRELY. Not reachable on 3.6 through 5.0, which were
        # all measured - but refusing by name beats an AttributeError if a future build moves it,
        # which is exactly what happened to grease_pencils between 3.6 and 4.4.
        raise MifOpError("this Blender (%s) has no bpy.data.%s, so %s cannot make one. NOTHING "
                         "was created." % (bpy.app.version_string, kind, verb))

    wanted = str(take(params, "name", default=default_name, kind=str))
    data = collection.new(name=wanted)
    obj = bpy.data.objects.new(data.name, data)
    _link_new(obj, coll)
    _place(obj, _placed)
    if want_scale is not None:
        obj.scale = want_scale

    want_type = {"hair_curves": "CURVES", "pointclouds": "POINTCLOUD",
                 "volumes": "VOLUME"}[kind]
    out = _created(obj, coll, want_type, verb, wanted)
    # THE COUNT IS REPORTED AND IT IS ZERO. Saying so beats omitting it: a caller who sees no count
    # field can believe the object came with something in it, and the whole risk with these two
    # types is a capability that looks real and is not.
    # SCALE READ BACK like location and rotation - it was written and not reported - and pointCount only where the type
    # HAS points: a volume holds grids, and its pointCount 0 was a field borrowed from the point cloud (found 2026-10-25).
    out.update({"dataName": data.name, "scale": rnd(list(obj.scale)), "isEmpty": True, "note": _GN_NOTE})
    if kind != "volumes":
        out["pointCount"] = 0
    return out


# MODULE-LEVEL AND A PLAIN LITERAL: parity_check.py resolves an accepted-key expression statically.
CREATE_CURVES_PARAMS = (
    "name", "location", "rotation", "scale", "collection",
    "surface", "uvMap", "count", "length", "pointsPerCurve", "direction", "customDirection", "densityGroup", "seed",
    # Blender's own Empty Hair, and its Quick Fur preset (2026-10-06)
    "surfaceDeform", "preset", "density", "radius", "viewPercentage", "applyHairGuides", "useNoise", "useFrizz",
)
_CURVES_NOTES = dict(_GN_NOTES, points="a Curves object's points come from a geometry node group, or from surface "
                                       "and count here: guide curves spawned on the surface mesh.")
del _CURVES_NOTES["count"]
del _CURVES_NOTES["radius"]          # taken since 2026-10-06: the fur preset's hair radius
_GUIDE_DIRECTIONS = ("NORMAL", "UP", "CUSTOM")
_GUIDE_MAX = 100000
# QUICK FUR'S OWN DENSITIES, read off object_quick_effects.py (3.6 and 5.2 alike): the curve count it aims for over
# the whole surface. Measured at LOW on all five builds: 997 to 1124 evaluated curves.
_FUR_DENSITY = {"LOW": 1000, "MEDIUM": 10000, "HIGH": 100000}
_FUR_KEYS = ("density", "radius", "viewPercentage", "applyHairGuides", "useNoise", "useFrizz")
# What fur cannot take: quick_fur grows its own guides from Blender's hair assets on the surface's active UV map, and it
# is built on Empty Hair, so the deform modifier comes with it.
_FUR_REFUSES = ("count", "pointsPerCurve", "direction", "customDirection", "densityGroup", "seed", "uvMap",
                "surfaceDeform")


def _guide_request(params):
    """The guide-spawning keys, read and checked before anything is created: None when no surface was asked for.
    Returns a dict the spawning step reads; every refusal ends 'NOTHING was created.'"""
    guide_keys = ("count", "length", "pointsPerCurve", "direction", "customDirection", "densityGroup", "seed", "uvMap",
                  "surfaceDeform", "preset") + _FUR_KEYS
    surface_name = take(params, "surface", default=None, kind=str)
    if surface_name is None:
        stray = [k for k in guide_keys if params.get(k) is not None]
        if stray:
            raise MifOpError("%s %s read only with surface (the mesh the guides grow from) - pass surface, or leave "
                             "%s out. NOTHING was created." % (", ".join(stray), "is" if len(stray) == 1 else "are",
                                                               "it" if len(stray) == 1 else "them"))
        return None
    placed = [k for k in ("location", "rotation", "scale") if params.get(k) is not None]
    if placed:
        raise MifOpError("%s cannot be set with surface: hair curves live in their surface's space, so the object is "
                         "parented to the surface at its origin, as Blender's own Empty Hair does. Move the surface "
                         "instead. NOTHING was created." % ", ".join(placed))
    surf = bpy.data.objects.get(surface_name)
    if surf is None:
        raise MifOpError("no object named '%s' to grow the guides from. NOTHING was created." % surface_name)
    if surf.type != "MESH":
        raise MifOpError("surface '%s' is a %s; guides grow from a MESH. NOTHING was created." % (surf.name, surf.type))
    mesh = surf.data
    if len(mesh.polygons) == 0:
        raise MifOpError("surface '%s' has no faces to grow guides from. NOTHING was created." % surf.name)
    preset = take(params, "preset", default=None, kind=str)
    fur_given = [k for k in _FUR_KEYS if params.get(k) is not None]
    if preset is None and fur_given:
        raise MifOpError("%s %s read only with preset 'fur'. NOTHING was created."
                         % (", ".join(fur_given), "is" if len(fur_given) == 1 else "are"))
    deform = take_bool(params, "surfaceDeform", default=False)
    if preset is not None or deform:
        _hair_operator_target(surf)
    if preset is not None:
        return _fur_request(params, surf, str(preset).lower())
    uv_name = take(params, "uvMap", default=None, kind=str)
    if uv_name is not None and mesh.uv_layers.get(uv_name) is None:
        raise MifOpError("surface '%s' has no UV map '%s'. UV maps: %s. NOTHING was created."
                         % (surf.name, uv_name, ", ".join(u.name for u in mesh.uv_layers) or "(none)"))
    if uv_name is None and mesh.uv_layers.active is not None:
        uv_name = mesh.uv_layers.active.name
    count = take_int(params, "count", default=0)
    if not 0 <= count <= _GUIDE_MAX:
        raise MifOpError("count is %d; it must be 0 (an empty groom on the surface) to %d. NOTHING was created."
                         % (count, _GUIDE_MAX))
    length = finite_float(take_float(params, "length", default=0.1), "length")
    if not 0.0 < length <= 10.0:
        raise MifOpError("length is %g m; a guide is over 0 and at most 10 m long. NOTHING was created." % length)
    per = take_int(params, "pointsPerCurve", default=8)
    if not 2 <= per <= 128:
        raise MifOpError("pointsPerCurve is %d; it must be 2 to 128. NOTHING was created." % per)
    direction = str(take(params, "direction", default="NORMAL", kind=str)).upper()
    if direction not in _GUIDE_DIRECTIONS:
        raise MifOpError("direction must be NORMAL, UP or CUSTOM, got %r. NOTHING was created." % direction)
    custom = params.get("customDirection")
    if (custom is not None) != (direction == "CUSTOM"):
        raise MifOpError("customDirection [x, y, z] (world) goes with direction CUSTOM and only with it. NOTHING was "
                         "created.")
    if custom is not None:
        if not isinstance(custom, (list, tuple)) or len(custom) != 3:
            raise MifOpError("customDirection takes [x, y, z], got %r. NOTHING was created." % (custom,))
        custom = finite_floats(list(custom), "customDirection")
        if sum(c * c for c in custom) < 1e-12:
            raise MifOpError("customDirection is a zero vector - it points nowhere. NOTHING was created.")
    group = take(params, "densityGroup", default=None, kind=str)
    if group is not None and surf.vertex_groups.get(group) is None:
        raise MifOpError("surface '%s' has no vertex group '%s'. Groups: %s. NOTHING was created."
                         % (surf.name, group, ", ".join(g.name for g in surf.vertex_groups) or "(none)"))
    seed = take_int(params, "seed", default=0)
    if seed < 0:
        raise MifOpError("seed must be 0 or more, got %d. NOTHING was created." % seed)
    if count and "add_curves" not in bpy.types.Curves.bl_rna.functions:
        raise MifOpError("this Blender (%s) has no Curves.add_curves, which writes the guides (it arrived in 4.2). "
                         "NOTHING was created." % bpy.app.version_string)
    return {"surface": surf, "uvMap": uv_name, "count": count, "length": length, "per": per, "direction": direction,
            "custom": custom, "group": group, "seed": seed, "deform": deform, "fur": None}


def _hair_operator_target(surf):
    """Refuse, before anything is created, a surface Blender's hair operators cannot act on: they run on the selected,
    active object, so it has to be selectable in object mode."""
    _require_object_mode("create_curves")
    view_layer = bpy.context.view_layer
    if surf.name not in view_layer.objects.keys():
        raise MifOpError("surface '%s' is not in the active view layer, and Blender's Empty Hair and Quick Fur act on "
                         "the selected object. NOTHING was created." % surf.name)
    if surf.hide_get() or surf.hide_viewport:
        raise MifOpError("surface '%s' is hidden, and Blender's Empty Hair and Quick Fur act on the selected object. "
                         "Unhide it first. NOTHING was created." % surf.name)


def _fur_request(params, surf, preset):
    """The fur preset's keys, read and checked before anything is created (quick_fur's own ranges)."""
    if preset != "fur":
        raise MifOpError("preset '%s' is not one this op makes; it makes 'fur' (Blender's Quick Fur). NOTHING was "
                         "created." % preset)
    clash = [k for k in _FUR_REFUSES if params.get(k) is not None]
    if clash:
        raise MifOpError("%s cannot go with preset 'fur': Quick Fur grows its own guides from Blender's hair assets on "
                         "the surface's active UV map, and builds on Empty Hair, so its deform modifier comes with it. "
                         "NOTHING was created." % ", ".join(clash))
    if len(surf.data.uv_layers) == 0:
        # QUICK FUR SKIPS A MESH WITH NO UV MAP SILENTLY - FINISHED, a warning, no object - and still leaves a new
        # "Fur Material" behind (measured on all five builds, 2026-10-06), so it is refused before it runs.
        raise MifOpError("surface '%s' has no UV map, and Quick Fur needs one to place the hair (it would skip the "
                         "mesh and report success). Unwrap it first. NOTHING was created." % surf.name)
    density = str(take(params, "density", default="MEDIUM", kind=str)).upper()
    if density not in _FUR_DENSITY:
        raise MifOpError("density must be LOW (1000 curves), MEDIUM (10000) or HIGH (100000), got %r. NOTHING was "
                         "created." % density)
    length = finite_float(take_float(params, "length", default=0.1), "length")
    if not 0.001 <= length <= 10.0:
        raise MifOpError("length is %g m; fur is 0.001 to 10 m long. NOTHING was created." % length)
    radius = finite_float(take_float(params, "radius", default=0.001), "radius")
    if not 0.0 <= radius <= 10.0:
        raise MifOpError("radius is %g m; a hair radius is 0 to 10 m. NOTHING was created." % radius)
    view = finite_float(take_float(params, "viewPercentage", default=1.0), "viewPercentage")
    if not 0.0 <= view <= 1.0:
        raise MifOpError("viewPercentage is %g; it is the share of curves drawn in the viewport, 0 to 1. NOTHING was "
                         "created." % view)
    fur = {"density": density, "length": length, "radius": radius, "view_percentage": view,
           "apply_hair_guides": take_bool(params, "applyHairGuides", default=True),
           "use_noise": take_bool(params, "useNoise", default=True),
           "use_frizz": take_bool(params, "useFrizz", default=True)}
    return {"surface": surf, "uvMap": surf.data.uv_layers.active.name if surf.data.uv_layers.active else None,
            "count": 0, "deform": False, "fur": fur}


def _blender_hair_object(req, wanted, coll):
    """Make the Curves object with Blender's own operator on the selected, active surface - object.quick_fur for the
    fur preset, object.curves_empty_hair_add otherwise - and adopt it under the caller's name and collection.

    REAL SELECTION, NOT A CONTEXT OVERRIDE. Quick Fur runs Empty Hair inside itself and then reads
    context.active_object for the new hair; under an outer temp_override(active_object=surface) that read returns the
    SURFACE, and every hair modifier lands on the mesh (measured on 5.2.1, 2026-10-06).

    Returns (object, modifiers the operator added to the surface)."""
    surf = req["surface"]
    before = set(bpy.data.objects.keys())
    surf_mods = [m.name for m in surf.modifiers]
    snapshot = selection_snapshot()
    try:
        select_only([surf])
        if req["fur"] is not None:
            result = bpy.ops.object.quick_fur(**req["fur"])
        else:
            result = bpy.ops.object.curves_empty_hair_add()
    except RuntimeError as exc:
        raise MifOpError("Blender refused to make hair on '%s': %s. NOTHING was created." % (surf.name,
                                                                                            str(exc).strip()))
    finally:
        selection_restore(snapshot)
    made = [bpy.data.objects[n] for n in sorted(set(bpy.data.objects.keys()) - before)]
    hair = [o for o in made if o.type == "CURVES"]
    if "FINISHED" not in result or len(hair) != 1:
        raise MifOpError("Blender's %s returned %s on '%s' and made %d Curves object(s) (%s); read the scene before "
                         "continuing." % ("quick_fur" if req["fur"] else "curves_empty_hair_add", sorted(result),
                                         surf.name, len(hair), ", ".join(o.name for o in made) or "none"))
    obj = hair[0]
    assign_id_name(obj, wanted, bpy.data.objects)
    assign_id_name(obj.data, wanted, bpy.data.hair_curves)
    if coll is not None and obj.name not in coll.objects:
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        coll.objects.link(obj)
    added = [m.name for m in surf.modifiers if m.name not in surf_mods]
    return obj, added


def _fur_reply(obj, req, added):
    """What Quick Fur made, read off the object: its modifiers, the stored guides, and the curves the groom evaluates
    to - the count the density aims for."""
    fur = req["fur"]
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    mods = [m.name for m in obj.modifiers]
    applied = fur["apply_hair_guides"] and not any(m.type == "NODES" and m.node_group is not None
                                                   and m.node_group.name.startswith("Generate Hair Curves")
                                                   for m in obj.modifiers)
    out = {"preset": "fur", "surface": req["surface"].name, "surfaceUvMap": obj.data.surface_uv_map or None,
           "parent": obj.parent.name if obj.parent else None, "density": fur["density"],
           "targetCurves": _FUR_DENSITY[fur["density"]], "curveCount": len(obj.data.curves),
           "pointCount": len(obj.data.points), "evaluatedCurveCount": len(ev.data.curves),
           "evaluatedPointCount": len(ev.data.points), "lengthM": fur["length"], "radius": fur["radius"],
           "viewPercentage": fur["view_percentage"], "applyHairGuides": fur["apply_hair_guides"],
           "guidesApplied": applied, "useNoise": fur["use_noise"], "useFrizz": fur["use_frizz"], "modifiers": mods,
           "material": obj.data.materials[0].name if len(obj.data.materials) and obj.data.materials[0] else None,
           "surfaceModifiersAdded": added, "isEmpty": len(ev.data.curves) == 0}
    out["note"] = ("Blender's Quick Fur: %d stored guide curve(s) on the object, interpolated by its modifiers into "
                   "%d evaluated curve(s) (the density aims for %d over the surface)"
                   % (out["curveCount"], out["evaluatedCurveCount"], out["targetCurves"]))
    if fur["apply_hair_guides"] and not applied:
        out["note"] += "; Blender could not apply the Generate modifier, so the guides are still generated by it"
    return out


def _guide_weights(surf, group, tri_verts):
    """Per triangle, the density group's mean weight over its corners (0 where a vertex is not in the group)."""
    import numpy as np                                   # noqa: PLC0415 - after every refusal
    gi = surf.vertex_groups[group].index
    w = np.zeros(len(surf.data.vertices))
    for v in surf.data.vertices:
        for g in v.groups:
            if g.group == gi:
                w[v.index] = g.weight
    return w[tri_verts].mean(axis=1)


def _spawn_guides(obj, req):
    """Write `count` guide curves into the Curves object `obj`, rooted on the surface by area (times the density
    group's weight), each `per` points long, straight along its direction for `length` world meters. Positions are
    in the surface's space (the object is parented to it at its origin). Writes surface_uv_coordinate per curve from
    the surface UV map. Returns the reply's guide block."""
    import numpy as np                                   # noqa: PLC0415 - after every refusal
    surf, curves = req["surface"], obj.data
    from mathutils import Matrix                          # noqa: PLC0415 - after every refusal
    obj.parent = surf
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = Matrix.Identity(4)
    curves.surface = surf
    if req["uvMap"]:
        curves.surface_uv_map = req["uvMap"]
    out = {"surface": surf.name, "surfaceUvMap": req["uvMap"], "curveCount": 0, "pointCount": 0,
           "pointsPerCurve": req["per"], "lengthM": req["length"], "direction": req["direction"],
           "densityGroup": req["group"], "seed": req["seed"], "surfaceUvAttribute": None}
    if not req["count"]:
        out["note"] = "count 0: the Curves object is set on its surface and holds no curves yet"
        return out
    mesh = surf.data
    mesh.calc_loop_triangles()
    n_tri = len(mesh.loop_triangles)
    tri_v = np.empty(n_tri * 3, np.int64)
    mesh.loop_triangles.foreach_get("vertices", tri_v)
    tri_v = tri_v.reshape(-1, 3)
    tri_l = np.empty(n_tri * 3, np.int64)
    mesh.loop_triangles.foreach_get("loops", tri_l)
    tri_l = tri_l.reshape(-1, 3)
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    mw = np.array(surf.matrix_world)
    m3 = mw[:3, :3]
    world = co @ m3.T + mw[:3, 3]
    a, b, c = world[tri_v[:, 0]], world[tri_v[:, 1]], world[tri_v[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    weight = area * (_guide_weights(surf, req["group"], tri_v) if req["group"] else 1.0)
    if not weight.sum() > 0.0:
        raise MifOpError("density group '%s' weighs 0 over every face of '%s' (or the faces have no area), so no "
                         "guide has anywhere to grow." % (req["group"], surf.name))
    rng = np.random.default_rng(req["seed"])
    n = req["count"]
    tri = rng.choice(n_tri, size=n, p=weight / weight.sum())
    r1, r2 = rng.random(n), rng.random(n)
    flip = r1 + r2 > 1.0
    r1[flip], r2[flip] = 1.0 - r1[flip], 1.0 - r2[flip]
    bary = np.stack([1.0 - r1 - r2, r1, r2], axis=1)
    root_local = np.einsum("nk,nkj->nj", bary, co[tri_v[tri]])
    if req["direction"] == "NORMAL":
        d_world = cross[tri] / np.maximum(1e-30, np.linalg.norm(cross[tri], axis=1, keepdims=True))
    elif req["direction"] == "UP":
        d_world = np.tile(np.array((0.0, 0.0, 1.0)), (n, 1))
    else:
        cv = np.array(req["custom"], float)
        d_world = np.tile(cv / np.linalg.norm(cv), (n, 1))
    # A WORLD LENGTH, whatever the surface's scale: the step is taken in world space and carried back into the
    # surface's space through its inverse matrix
    step_local = (d_world * req["length"]) @ np.linalg.inv(m3).T
    t = np.linspace(0.0, 1.0, req["per"])
    pts = root_local[:, None, :] + t[None, :, None] * step_local[:, None, :]
    curves.add_curves([req["per"]] * n)
    curves.attributes["position"].data.foreach_set("vector", pts.astype(np.float32).ravel())
    out.update({"curveCount": len(curves.curves), "pointCount": len(curves.points)})
    if req["uvMap"]:
        uv = np.empty(len(mesh.loops) * 2)
        mesh.uv_layers[req["uvMap"]].data.foreach_get("uv", uv)
        uv = uv.reshape(-1, 2)
        root_uv = np.einsum("nk,nkj->nj", bary, uv[tri_l[tri]])
        attr = curves.attributes.get("surface_uv_coordinate") or curves.attributes.new("surface_uv_coordinate",
                                                                                         "FLOAT2", "CURVE")
        attr.data.foreach_set("vector", root_uv.astype(np.float32).ravel())
        out["surfaceUvAttribute"] = attr.name
    else:
        out["uvNote"] = ("the surface has no UV map, so no surface_uv_coordinate was written - a shader reading the "
                         "surface's texture by the root's UV has nothing to read")
    curves.update_tag()
    return out


def op_create_curves(params):
    """A CURVES object - Blender's hair/curves type, driven by geometry nodes, or grown as guides on a surface.

    NOT create_curve, WHICH ALREADY EXISTS AND IS A DIFFERENT THING. That one makes a legacy CURVE
    object - a Bezier or NURBS path with control points you set directly, the thing you extrude a
    profile along. This makes the modern CURVES type, which holds no editable control points of its
    own and is filled by a geometry node group. Both are real and neither replaces the other; the
    names are Blender's and the one-letter difference is unfortunate.

    WHY THIS ONE AND NOT THE OTHER SIX unbuilt object types: measured on 3.6.23, 4.4.0 and 5.0.1,
    an empty one accepts a node group and evaluates to real geometry. The ops that fill it -
    create_node_group, assign_node_group - are already here. A metaball, a volume and a grease
    pencil have no such op, so they would be empty containers with no way to fill them.

    THE GAP (2026-10-06): a groom started with nothing - no op could put a guide curve on a surface, so a
    hair-card or groom node tree had no guides to read. With surface, real guides are written through the build's
    own Curves.add_curves (4.2 and later; measured: add_curves([4, 4, 4]) gives 3 curves and 12 points), the object
    parented to the surface at its origin with Curves.surface and surface_uv_map set, as Empty Hair leaves them.

    params:
      name (str)                 Blender appends .001 on a collision; the response says so
      location / rotation / scale / collection   (not with surface: the curves live in its space)
      surface (str)              a mesh to grow guides on; the object is set on it (surface, surface_uv_map)
      uvMap (str)                the surface's UV map for surface_uv_coordinate; default its active one
      count (int)                guides to spawn, 0 (default: set on the surface, no curves) to 100000
      length (m)                 each guide's world length, default 0.1
      pointsPerCurve (int)       2 to 128, default 8
      direction (str)            NORMAL (default: the face's), UP (world +Z) or CUSTOM with customDirection [x,y,z]
      densityGroup (str)         a vertex group on the surface; roots spread by face area times its weight
      seed (int)                 0 or more, default 0 - the same seed, the same roots
      surfaceDeform (bool)       with surface: the object Blender's own Empty Hair makes (object.curves_empty_hair_add)
                                 - parented, surface and surface_uv_map set, and the build's deform modifier: measured
                                 "Surface Deform" on 3.6-5.0, "Hair Dynamics" on 5.2, which also adds "Capture Rest
                                 Geometry" to the surface. Guides (count) are then written into it as above.
      preset (str)               "fur": Blender's Quick Fur on the surface (it needs a UV map), with
        density (str)            LOW (1000 curves) | MEDIUM (default, 10000) | HIGH (100000)
        length (m)               default 0.1; radius (m) default 0.001; viewPercentage 0-1, default 1
        applyHairGuides / useNoise / useFrizz (bool)   default true
    """
    reject_unknown(params, CREATE_CURVES_PARAMS, "create_curves", _CURVES_NOTES)
    req = _guide_request(params)
    if req is not None and (req["deform"] or req["fur"] is not None):
        return _create_blender_hair(params, req)
    out = _gn_object(params, "hair_curves", "create_curves", "Curves")
    if req is None:
        return out
    obj = bpy.data.objects.get(out["name"])
    try:
        guides = _spawn_guides(obj, req)
    except MifOpError as exc:
        # THE ONLY REFUSAL AFTER THE OBJECT EXISTS (a density group of all zeros is known only once the faces are
        # weighed): the object goes again, so the promise holds.
        data = obj.data
        remove_objects([obj])
        bpy.data.hair_curves.remove(data)
        raise MifOpError("%s NOTHING was created." % str(exc).rstrip("."))
    out.update(guides)
    out["location"], out["rotation"] = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    out.update({"pointCount": guides["pointCount"], "isEmpty": guides["curveCount"] == 0,
                "parent": req["surface"].name})
    if guides["curveCount"]:
        out["note"] = ("the guides are stored on the object itself (object_info generated.stored counts them); a "
                       "geometry node group on it reads them as its input curves")
    return out


def _create_blender_hair(params, req):
    """create_curves with surfaceDeform or preset 'fur': the object Blender's own operator makes, adopted."""
    coll = _resolve_collection(params) if params.get("collection") else None
    wanted = str(take(params, "name", default="Curves", kind=str))
    obj, added = _blender_hair_object(req, wanted, coll)
    out = _created(obj, coll or obj.users_collection[0], "CURVES", "create_curves", wanted)
    out["dataName"] = obj.data.name
    if req["fur"] is not None:
        out.update(_fur_reply(obj, req, added))
        return out
    try:
        guides = _spawn_guides(obj, req)
    except MifOpError as exc:
        # A density group weighing 0 everywhere is known only once the faces are weighed: the object, its data and
        # what Empty Hair put on the surface go again, so the promise holds.
        surf = req["surface"]
        data = obj.data
        remove_objects([obj])
        bpy.data.hair_curves.remove(data)
        for name in added:
            mod = surf.modifiers.get(name)
            if mod is not None:
                surf.modifiers.remove(mod)
        raise MifOpError("%s NOTHING was created." % str(exc).rstrip("."))
    out.update(guides)
    out["location"], out["rotation"] = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    out.update({"surfaceDeform": True, "madeBy": "object.curves_empty_hair_add",
                "deformModifiers": [m.name for m in obj.modifiers if m.type == "NODES"],
                "surfaceModifiersAdded": added, "pointCount": guides["pointCount"],
                "isEmpty": guides["curveCount"] == 0, "parent": req["surface"].name})
    out["note"] = ("Blender's Empty Hair: parented to '%s' with Curves.surface and surface_uv_map set and the build's "
                   "deform modifier (%s)%s; %s"
                   % (req["surface"].name, ", ".join(out["deformModifiers"]) or "none",
                      "; it also added %s to the surface" % ", ".join(added) if added else "",
                      "%d guide curve(s) written" % guides["curveCount"] if guides["curveCount"]
                      else "no curves yet"))
    return out


def op_create_point_cloud(params):
    """A POINTCLOUD object - points with attributes, driven by geometry nodes.

    THE SAME ARGUMENT AS create_curves, and the same measurement: on 3.6.23, 4.4.0 and 5.0.1 an
    empty pointcloud accepts a NODES modifier and a generating group evaluates to 16 real points.

    IT ARRIVES EMPTY AND THAT IS THE OBJECT TYPE, not a defect. Blender has no UI for placing points
    in a pointcloud by hand either - the type exists to be generated. The response says so outright,
    with pointCount 0 and isEmpty true, because the failure worth avoiding here is a caller
    believing an object is populated when nothing has filled it.

    params:
      name (str)
      location / rotation / scale / collection
    """
    reject_unknown(params, _GN_KEYS, "create_point_cloud", _GN_NOTES)
    return _gn_object(params, "pointclouds", "create_point_cloud", "PointCloud")


_VOLUME_KEYS = {"name", "location", "rotation", "scale", "collection", "filepath", "file", "sequence"}
_VDB_SEQ_KEYS = {"frameStart", "frameOffset", "mode"}
_VDB_SEQ_RE = re.compile(r"^(.*?)(\d+)$")


def _vdb_request(params):
    """None for a node-driven volume, else the checked file request - before anything is created."""
    path = take(params, "filepath", "file", kind=str)
    seq = params.get("sequence")
    if path is None:
        if seq is not None:
            raise MifOpError("sequence describes a VDB sequence; pass filepath too. NOTHING was created.")
        return None
    if seq is not None and not isinstance(seq, (bool, dict)):
        raise MifOpError("sequence is true or {frameStart, frameOffset, mode}, got %r. NOTHING was created." % (seq,))
    sopt = seq if isinstance(seq, dict) else {}
    if set(sopt) - _VDB_SEQ_KEYS:
        raise MifOpError("sequence takes %s; %s is not one. NOTHING was created."
                         % (", ".join(sorted(_VDB_SEQ_KEYS)), ", ".join(sorted(set(sopt) - _VDB_SEQ_KEYS))))
    full = os.path.abspath(bpy.path.abspath(path))
    if not os.path.isfile(full):
        raise MifOpError("no file '%s'. NOTHING was created." % full)
    if not full.lower().endswith(".vdb"):
        raise MifOpError("'%s' is not a .vdb file. NOTHING was created." % full)
    req = {"path": full, "sequence": seq is True or isinstance(seq, dict),
           "frameStart": take_int(sopt, "frameStart", default=1), "frameOffset": take_int(sopt, "frameOffset", default=0),
           "mode": None, "files": [full]}
    mode = take(sopt, "mode", kind=str)
    if mode is not None:
        modes = [i.identifier for i in bpy.types.Volume.bl_rna.properties["sequence_mode"].enum_items]
        if mode.upper() not in modes:
            raise MifOpError("sequence.mode is one of %s on this build, got %r. NOTHING was created." % (modes, mode))
        req["mode"] = mode.upper()
    if req["sequence"]:
        stem, ext = os.path.splitext(os.path.basename(full))
        m = _VDB_SEQ_RE.match(stem)
        if m is None:
            raise MifOpError("'%s' has no frame number at the end of its name, so it is not part of a sequence. NOTHING "
                             "was created." % os.path.basename(full))
        head, digits = m.group(1), m.group(2)
        folder = os.path.dirname(full)
        nums = sorted(int(mm.group(2)) for f in os.listdir(folder)
                      for mm in [_VDB_SEQ_RE.match(os.path.splitext(f)[0])]
                      if f.lower().endswith(ext.lower()) and mm is not None and mm.group(1) == head
                      and len(mm.group(2)) == len(digits))
        req["numbers"] = nums
        req["files"] = [os.path.join(folder, "%s%0*d%s" % (head, len(digits), n, ext)) for n in nums]
        gaps = [n for n in range(nums[0], nums[-1] + 1) if n not in set(nums)] if nums else []
        if gaps:
            req["gaps"] = gaps[:50]
    return req


def _vdb_load(obj, req):
    """Point the volume at the file (or sequence), load its grids and describe them; a failure removes the object."""
    import mathutils                                                 # noqa: PLC0415
    vol = obj.data
    vol.filepath = req["path"]
    if req["sequence"]:
        vol.is_sequence = True
        vol.frame_duration = len(req["files"])
        vol.frame_start = req["frameStart"]
        vol.frame_offset = req["frameOffset"]
        if req["mode"]:
            vol.sequence_mode = req["mode"]
    loaded = vol.grids.load()
    if not loaded or not len(vol.grids):
        why = getattr(vol.grids, "error_message", "") or "no grids"
        name = obj.name
        remove_objects([obj])
        bpy.data.volumes.remove(vol)
        raise MifOpError("Blender could not read grids from '%s' (%s). '%s' was removed: NOTHING was created."
                         % (req["path"], why, name))
    grids = []
    for g in vol.grids:
        mx = g.matrix_object
        grids.append({"name": g.name, "dataType": g.data_type, "channels": g.channels,
                      "voxelSize": [round(mx.col[i].to_3d().length, 6) for i in range(3)]})
    bpy.context.view_layer.update()
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    corners = [obj.matrix_world @ mathutils.Vector(c) for c in ev.bound_box]
    lo = [min(c[i] for c in corners) for i in range(3)]
    hi = [max(c[i] for c in corners) for i in range(3)]
    out = {"filepath": req["path"], "grids": grids, "gridCount": len(grids),
           "activeGrid": (vol.grids[vol.grids.active_index].name
                          if 0 <= getattr(vol.grids, "active_index", -1) < len(vol.grids) else None),
           "bboxWorld": {"min": [round(x, 6) for x in lo], "max": [round(x, 6) for x in hi],
                         "size": [round(hi[i] - lo[i], 6) for i in range(3)]},
           "isEmpty": False,
           "note": "an OpenVDB volume: its grids come from the file. It renders through a material whose output is fed by "
                   "a volume shader (Principled Volume reads the density grid by name)."}
    if req["sequence"]:
        out["sequence"] = {"frames": len(req["files"]), "first": req["numbers"][0], "last": req["numbers"][-1],
                           "frameStart": vol.frame_start, "frameOffset": vol.frame_offset,
                           "sequenceMode": vol.sequence_mode, **({"missingNumbers": req["gaps"]} if req.get("gaps") else {})}
    return out


def op_create_volume(params):
    """A VOLUME object - the container for OpenVDB grids, filled by geometry nodes.

    THE THIRD ON THE SAME MEASURED ARGUMENT as create_curves and create_point_cloud, and the
    argument was checked for this type specifically rather than assumed from the other two: with a
    volume-OUTPUTTING group (Mesh Grid into Mesh to Volume), an empty volume object evaluates to a
    grid on 3.6.23, 4.4.0 and 5.0.1 alike. assign_node_group reaches it, since that op stopped being
    mesh-only in the same change.

    AN OPENVDB FILE OR SEQUENCE (second 1.3 audit, 2026-10-25). This used to refuse `filepath` with "there is no VDB
    reader in this addon" - but Blender has one (object.volume_import, 5.2 bl_ui/space_view3d.py:2618, 3.6 :2288),
    and its data route is Volume.filepath + grids.load(), which works headless. With filepath the volume holds that
    file (sequence: the numbered files beside it, frame_duration set from how many there are), and the reply lists
    its grids - name, data type, channels, voxel size - and the world box Blender computes from them. A file Blender
    cannot read is refused and the object removed.

    A VOLUME WITH GRIDS AND NO VOLUME SHADER RENDERS NOTHING, which is the way this most often looks
    broken. A fresh volume object has NO material at all - reported as materialSlots so it is a
    number rather than a warning to be believed - and Blender's default material is a SURFACE
    shader, which a volume does not use. It needs a material whose output is fed by a volume shader.
    Not created here: choosing somebody's shader is not this op's job, and the addon's material ops
    already exist.

    params:
      name (str)                 Blender appends .001 on a collision; the response says so
      location / rotation / scale / collection
      filepath (or file)         an OpenVDB file to load
      sequence (true | dict)     the file is one of a numbered sequence (name_0001.vdb ...): every number is found.
                                 {frameStart (the scene frame it starts on, default 1), frameOffset (added to the file
                                 number, default 0), mode (CLIP, EXTEND, REPEAT or PING_PONG, this build's enum)}
    """
    reject_unknown(params, _VOLUME_KEYS, "create_volume", dict(_GN_NOTES, **{
        "resolution": "voxel size is decided by the node that generates the volume (Mesh to Volume "
                      "takes a Voxel Amount or Voxel Size), or by the VDB file, not by the object.",
        "density": "an attribute on the generated grid, written by the node group.",
    }))
    vdb = _vdb_request(params)
    out = _gn_object(params, "volumes", "create_volume", "Volume")
    obj = bpy.data.objects.get(out["name"])
    if vdb is not None:
        out.update(_vdb_load(obj, vdb))
    out["materialSlots"] = len(obj.material_slots) if obj is not None else 0
    out["renderNote"] = (
        "a volume renders through a VOLUME shader, and this object has %d material slot(s). "
        "Blender's default material is a SURFACE shader, which a volume does not use - so a volume "
        "full of grids and no volume shader renders as nothing at all, which is the usual reason "
        "one looks broken." % out["materialSlots"])
    return out


# ------------------------------------------------------------------------------------------- SURFACE and META creators
# THE LAST TWO OBJECT TYPES (1.3, 2026-10-26). SURFACE and META were left out on 2026-09-07 because bpy.data alone makes
# EMPTY CONTAINERS of both - a curve datablock of type SURFACE with no spline, a metaball with no element - and an empty
# container is a capability that looks real and is not. So neither op here returns one: create_nurbs_surface makes a
# primitive with real control points (Blender's own Surface primitives, since a Spline's point_count_u / point_count_v
# are read-only and a U x V grid cannot be built point by point through the data API), and create_metaball always has
# at least one element. Both report what the object EVALUATES to (faces and vertices of its mesh), so "it has geometry"
# is a measurement in the reply rather than a claim.
_NURBS_KINDS = {
    "PATCH": "primitive_nurbs_surface_surface_add",
    "CYLINDER": "primitive_nurbs_surface_cylinder_add",
    "SPHERE": "primitive_nurbs_surface_sphere_add",
    "TORUS": "primitive_nurbs_surface_torus_add",
}
_NURBS_KEYS = {"name", "kind", "radius", "location", "rotation", "scale", "collection", "controlPoints", "orderU",
               "orderV", "resolutionU", "resolutionV", "endpointU", "endpointV", "cyclicU", "cyclicV"}
_NURBS_NOTES = {
    "points": "a surface's points are a U x V grid whose size Blender's data API cannot set (Spline.point_count_u / "
              "point_count_v are read-only); pick a kind and move its points with controlPoints.",
    "splineType": "a surface's splines are NURBS; there is no POLY or BEZIER surface.",
    "type": "see kind (PATCH, CYLINDER, SPHERE or TORUS).",
}


def _evaluated_counts(obj):
    """Faces and vertices the object evaluates to, or None when this view layer does not evaluate it.

    THE VIEW LAYER IS SYNCED FIRST. An object linked by bpy.data (not by an operator) is missing from
    view_layer.objects until the layer resyncs, so the membership test alone answered "not in this view layer"
    for every metaball create_metaball had just linked - measured on 5.2.1: False before the update, True after."""
    bpy.context.view_layer.update()
    if obj.name not in bpy.context.view_layer.objects:
        return None
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    try:
        me = ev.to_mesh()
    except RuntimeError:
        return None
    try:
        return {"faces": len(me.polygons), "vertices": len(me.vertices)} if me is not None else {"faces": 0,
                                                                                              "vertices": 0}
    finally:
        ev.to_mesh_clear()


def _move_to_collection(obj, coll):
    """An operator-made object links into the active collection; put it in the resolved one instead."""
    if obj.name not in coll.objects:
        coll.objects.link(obj)
    for other in list(obj.users_collection):
        if other != coll:
            other.objects.unlink(obj)


def _nurbs_request(params):
    """Every key read and checked BEFORE the operator runs - a refusal leaves nothing behind."""
    tail = "NOTHING was created."
    kind = str(take(params, "kind", default="PATCH", kind=str)).upper()
    if kind not in _NURBS_KINDS:
        raise MifOpError("kind is one of %s, got %r. %s" % (", ".join(sorted(_NURBS_KINDS)), kind, tail))
    radius = finite_float(take_float(params, "radius", default=1.0), "radius")
    if not radius > 0:
        raise MifOpError("radius is above 0, got %g. %s" % (radius, tail))
    try:
        placed = _place_values(params)
        scale = _vec3(params, "scale", (1.0, 1.0, 1.0)) if "scale" in params else None
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
    cps = params.get("controlPoints")
    if cps is not None:
        if not isinstance(cps, list) or not cps:
            raise MifOpError("controlPoints is a list of [x, y, z] or [x, y, z, weight], one per control point in "
                             "the primitive's order (U fastest). %s" % tail)
        rows = []
        for i, p in enumerate(cps):
            if not isinstance(p, (list, tuple)) or len(p) not in (3, 4):
                raise MifOpError("controlPoints[%d] is [x, y, z] or [x, y, z, weight], got %r. %s" % (i, p, tail))
            row = finite_floats(list(p), "controlPoints[%d]" % i)
            if len(row) == 4 and not row[3] > 0:
                raise MifOpError("controlPoints[%d]'s weight is above 0, got %g. %s" % (i, row[3], tail))
            rows.append(row)
        cps = rows
    ints = {}
    for key, lo, hi in (("orderU", 2, 6), ("orderV", 2, 6), ("resolutionU", 1, 1024), ("resolutionV", 1, 1024)):
        v = take_int(params, key, default=None)
        if v is not None and not lo <= v <= hi:
            raise MifOpError("%s is %d to %d, got %d. %s" % (key, lo, hi, v, tail))
        ints[key] = v
    bools = {}
    for key in ("endpointU", "endpointV", "cyclicU", "cyclicV"):
        bools[key] = take_bool(params, key) if params.get(key) is not None else None
    return {"kind": kind, "radius": radius, "placed": placed, "scale": scale, "cps": cps, "ints": ints,
            "bools": bools}


def op_create_nurbs_surface(params):
    """A SURFACE object - a NURBS surface with real control points (Blender's Surface primitives).

    params:
      name (str)                 Blender appends .001 on a collision; the response says so
      kind (str)                 PATCH (default, a 4 x 4 grid), CYLINDER, SPHERE or TORUS - Add > Surface's shapes
      radius (m)                 the primitive's size, default 1
      location / rotation / scale / collection
      controlPoints              [[x, y, z] or [x, y, z, weight], ...] in the object's space, one per control point,
                                 U fastest - exactly the primitive's count (the reply's controlPointCount); refused,
                                 with the object removed, when it differs
      orderU / orderV (2-6)      the spline's NURBS order (Blender clamps it to the point count; read back)
      resolutionU / resolutionV (1-1024)   evaluated segments per span
      endpointU / endpointV / cyclicU / cyclicV (bool)
    Reply: the spline's grid (pointCountU x pointCountV), orders, resolutions and flags read back, and the faces and
    vertices the surface evaluates to.
    """
    reject_unknown(params, _NURBS_KEYS, "create_nurbs_surface", _NURBS_NOTES)
    req = _nurbs_request(params)
    coll = _resolve_collection(params)
    _require_object_mode("create_nurbs_surface")
    surface_ops = getattr(bpy.ops, "surface", None)
    if surface_ops is None or not hasattr(surface_ops, _NURBS_KINDS[req["kind"]]):
        raise MifOpError("this Blender (%s) has no surface.%s, which makes the %s. NOTHING was created."
                         % (bpy.app.version_string, _NURBS_KINDS[req["kind"]], req["kind"]))
    wanted = take(params, "name", default="Surface", kind=str)
    loc, rot = req["placed"]
    snapshot = selection_snapshot()
    before = set(bpy.data.objects.keys())
    try:
        res = getattr(surface_ops, _NURBS_KINDS[req["kind"]])(radius=req["radius"], enter_editmode=False,
                                                              align="WORLD", location=loc, rotation=rot)
        made = [n for n in bpy.data.objects.keys() if n not in before]
    except (TypeError, RuntimeError) as exc:
        raise MifOpError("Blender refused the %s surface: %s. NOTHING was created." % (req["kind"], exc))
    finally:
        selection_restore(snapshot)
    if "FINISHED" not in res or len(made) != 1:
        for n in made:
            o = bpy.data.objects.get(n)
            if o is not None:
                remove_objects([o])
        raise MifOpError("surface.%s answered %s and made %d object(s); expected one. NOTHING was created."
                         % (_NURBS_KINDS[req["kind"]], sorted(res), len(made)))
    obj = bpy.data.objects[made[0]]
    data = obj.data

    def undo(msg):
        remove_objects([obj])
        if data.users == 0:
            bpy.data.curves.remove(data)
        raise MifOpError("%s NOTHING was created." % msg)

    if obj.mode != "OBJECT":
        # enter_editmode=False leaves object mode on every build this was read against; a build that does not is
        # refused rather than left in edit mode for the next op.
        undo("the %s surface was left in %s mode." % (req["kind"], obj.mode))
    assign_id_name(obj, wanted, bpy.data.objects)
    assign_id_name(data, obj.name, bpy.data.curves)
    _move_to_collection(obj, coll)
    if req["scale"] is not None:
        obj.scale = req["scale"]
    points = [p for s in data.splines for p in s.points]
    if req["cps"] is not None:
        if len(req["cps"]) != len(points):
            undo("controlPoints has %d entries and the %s has %d control points (%s); give exactly one per point."
                 % (len(req["cps"]), req["kind"], len(points),
                    " + ".join("%d x %d" % (s.point_count_u, s.point_count_v) for s in data.splines)))
        for p, row in zip(points, req["cps"]):
            p.co = (row[0], row[1], row[2], row[3] if len(row) == 4 else p.co[3])
    for s in data.splines:
        for key, attr in (("orderU", "order_u"), ("orderV", "order_v"), ("resolutionU", "resolution_u"),
                          ("resolutionV", "resolution_v")):
            if req["ints"][key] is not None:
                setattr(s, attr, req["ints"][key])
        for key, attr in (("endpointU", "use_endpoint_u"), ("endpointV", "use_endpoint_v"),
                          ("cyclicU", "use_cyclic_u"), ("cyclicV", "use_cyclic_v")):
            if req["bools"][key] is not None:
                setattr(s, attr, bool(req["bools"][key]))
    data.update_tag()
    out = _created(obj, coll, "SURFACE", "create_nurbs_surface", wanted)
    ev = _evaluated_counts(obj)
    out.update({
        "kind": req["kind"], "dataName": data.name, "scale": rnd(list(obj.scale)),
        "controlPointCount": len(points),
        "splines": [{"pointCountU": s.point_count_u, "pointCountV": s.point_count_v, "orderU": s.order_u,
                     "orderV": s.order_v, "resolutionU": s.resolution_u, "resolutionV": s.resolution_v,
                     "cyclicU": s.use_cyclic_u, "cyclicV": s.use_cyclic_v, "endpointU": s.use_endpoint_u,
                     "endpointV": s.use_endpoint_v} for s in data.splines],
        "controlPoints": [rnd(list(p.co)) for p in points] if len(points) <= 256 else None,
        "evaluated": ev, "isEmpty": (ev["faces"] == 0) if ev is not None else None,
    })
    if ev is None:
        out["evaluatedNote"] = "the object is not in this view layer, so it was not evaluated here"
    return out


_META_KEYS = {"name", "location", "rotation", "scale", "collection", "elements", "resolution", "renderResolution",
              "threshold", "updateMethod"}
_META_ELEMENT_KEYS = {"type", "co", "radius", "stiffness", "size", "rotation", "negative"}
_META_NOTES = {
    "points": "a metaball is made of elements, not points: elements [{type, co, radius, stiffness, ...}].",
    "material": "assign it with set_material_slots once the object exists.",
}


def _meta_elements(params, tail):
    """The element list, read and checked before anything is created. Default: Blender's own Add > Metaball, one
    BALL of radius 2 at the origin - never an empty metaball, which renders nothing."""
    raw = params.get("elements")
    if raw is None:
        raw = [{"type": "BALL"}]
    if not isinstance(raw, list) or not raw or len(raw) > 1000:
        raise MifOpError("elements is a list of 1 to 1000 {type, co, radius, stiffness, size, rotation, negative}; a "
                         "metaball with no element renders nothing. %s" % tail)
    types = [i.identifier for i in bpy.types.MetaElement.bl_rna.properties["type"].enum_items]
    out = []
    for i, el in enumerate(raw):
        if not isinstance(el, dict):
            raise MifOpError("elements[%d] is an object {type, co, radius, ...}, got %r. %s" % (i, el, tail))
        extra = set(el) - _META_ELEMENT_KEYS
        if extra:
            raise MifOpError("elements[%d] takes %s; unknown %s. %s" % (i, sorted(_META_ELEMENT_KEYS), sorted(extra),
                                                                         tail))
        etype = str(el.get("type") or "BALL").upper()
        if etype not in types:
            raise MifOpError("elements[%d].type is one of %s (this build's), got %r. %s" % (i, ", ".join(types), etype,
                                                                                           tail))
        row = {"type": etype}
        co = el.get("co", [0.0, 0.0, 0.0])
        if not isinstance(co, (list, tuple)) or len(co) != 3:
            raise MifOpError("elements[%d].co is [x, y, z], got %r. %s" % (i, co, tail))
        row["co"] = finite_floats(list(co), "elements[%d].co" % i)
        for key, lo, hi, dflt in (("radius", 0.0, 1e6, 2.0), ("stiffness", 0.0, 10.0, 2.0)):
            v = finite_float(el.get(key, dflt), "elements[%d].%s" % (i, key))
            if not (lo < v <= hi if key == "radius" else lo <= v <= hi):
                raise MifOpError("elements[%d].%s is %s, got %g. %s"
                                 % (i, key, "above 0" if key == "radius" else "0 to 10", v, tail))
            row[key] = v
        if el.get("size") is not None:
            size = el["size"]
            if not isinstance(size, (list, tuple)) or len(size) != 3:
                raise MifOpError("elements[%d].size is [x, y, z] (CAPSULE, PLANE, ELLIPSOID and CUBE use it), got %r. "
                                 "%s" % (i, size, tail))
            size = finite_floats(list(size), "elements[%d].size" % i)
            if any(s < 0 for s in size):
                raise MifOpError("elements[%d].size is 0 or more on each axis. %s" % (i, tail))
            if etype == "BALL":
                raise MifOpError("elements[%d] is a BALL, which has no size - its radius is its size. %s" % (i, tail))
            row["size"] = size
        if el.get("rotation") is not None:
            r = el["rotation"]
            if not isinstance(r, (list, tuple)) or len(r) not in (3, 4):
                raise MifOpError("elements[%d].rotation is a quaternion [w, x, y, z] or Euler radians [x, y, z], got "
                                 "%r. %s" % (i, r, tail))
            r = finite_floats(list(r), "elements[%d].rotation" % i)
            from mathutils import Euler, Quaternion                     # noqa: PLC0415
            q = Euler(r).to_quaternion() if len(r) == 3 else Quaternion(r)
            if q.magnitude < 1e-9:
                raise MifOpError("elements[%d].rotation is a zero quaternion. %s" % (i, tail))
            row["rotation"] = list(q.normalized())
        row["negative"] = take_bool(el, "negative", default=False)
        out.append(row)
    return out


def _meta_family(obj):
    """Metaballs of one FAMILY (the same name before the .NNN suffix) are polygonized together, by the member with
    the lowest number (Blender's basis); the others draw nothing of their own."""
    base, _, num = obj.name.rpartition(".")
    if not (base and num.isdigit()):
        base = obj.name
    members = []
    for o in bpy.context.scene.objects:
        if o.type != "META":
            continue
        b, _, n = o.name.rpartition(".")
        if b and n.isdigit():
            if b == base:
                members.append((int(n), o.name))
        elif o.name == base:
            members.append((-1, o.name))
    members.sort()
    return {"family": base, "members": [n for _, n in members], "basis": members[0][1] if members else obj.name}


def op_create_metaball(params):
    """A META object - a metaball with its elements, never an empty one.

    params:
      name (str)                 Blender appends .001 on a collision; the response says so - and a metaball's name
                                 decides its FAMILY: Name.001 joins Name's surface and draws nothing of its own
      location / rotation / scale / collection
      elements [{type (BALL default, CAPSULE, PLANE, ELLIPSOID, CUBE - this build's list), co [x, y, z] (the object's
                space, default origin), radius (default 2, Blender's), stiffness (0-10, default 2), size [x, y, z] (not
                on a BALL), rotation (quaternion [w, x, y, z] or Euler radians [x, y, z]), negative (bool: subtracts)}]
                                 default one BALL, as Add > Metaball makes
      resolution (m)             viewport polygonization size, 0.005-10000
      renderResolution (m)       render polygonization size, 0.005-10000
      threshold (0-5)            the influence the surface is drawn at
      updateMethod               this build's Metaball.update_method value (UPDATE_ALWAYS, HALFRES, FAST, NEVER)
    Reply: the elements read back, the family and its basis, and the faces and vertices it evaluates to.
    """
    reject_unknown(params, _META_KEYS, "create_metaball", _META_NOTES)
    tail = "NOTHING was created."
    try:
        placed = _place_values(params)
        scale = _vec3(params, "scale", (1.0, 1.0, 1.0)) if "scale" in params else None
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
    elements = _meta_elements(params, tail)
    settings = {}
    for key, attr, lo, hi in (("resolution", "resolution", 0.005, 10000.0),
                              ("renderResolution", "render_resolution", 0.005, 10000.0),
                              ("threshold", "threshold", 0.0, 5.0)):
        if params.get(key) is not None:
            v = finite_float(params[key], key)
            if not lo <= v <= hi:
                raise MifOpError("%s is %g to %g, got %g. %s" % (key, lo, hi, v, tail))
            settings[attr] = v
    if params.get("updateMethod") is not None:
        methods = [i.identifier for i in bpy.types.MetaBall.bl_rna.properties["update_method"].enum_items]
        um = str(params["updateMethod"]).upper()
        if um not in methods:
            raise MifOpError("updateMethod is one of %s (this build's), got %r. %s" % (", ".join(methods), um, tail))
        settings["update_method"] = um
    coll = _resolve_collection(params)
    if getattr(bpy.data, "metaballs", None) is None:
        raise MifOpError("this Blender (%s) has no bpy.data.metaballs. %s" % (bpy.app.version_string, tail))
    wanted = str(take(params, "name", default="Mball", kind=str))
    mb = bpy.data.metaballs.new(wanted)
    for attr, v in settings.items():
        setattr(mb, attr, v)
    for row in elements:
        el = mb.elements.new(type=row["type"])
        el.co = row["co"]
        el.radius = row["radius"]
        el.stiffness = row["stiffness"]
        if "size" in row:
            el.size_x, el.size_y, el.size_z = row["size"]
        if "rotation" in row:
            el.rotation = row["rotation"]
        el.use_negative = row["negative"]
    obj = bpy.data.objects.new(mb.name, mb)
    _link_new(obj, coll)
    _place(obj, placed)
    if scale is not None:
        obj.scale = scale
    out = _created(obj, coll, "META", "create_metaball", wanted)
    fam = _meta_family(obj)
    ev = _evaluated_counts(bpy.data.objects[fam["basis"]]) if fam["basis"] in bpy.data.objects else None
    out.update({
        "dataName": mb.name, "scale": rnd(list(obj.scale)),
        "resolution": round(mb.resolution, 6), "renderResolution": round(mb.render_resolution, 6),
        "threshold": round(mb.threshold, 6), "updateMethod": mb.update_method,
        "elements": [{"type": e.type, "co": rnd(list(e.co)), "radius": round(e.radius, 6),
                      "stiffness": round(e.stiffness, 6), "size": rnd([e.size_x, e.size_y, e.size_z]),
                      "rotation": rnd(list(e.rotation)), "negative": e.use_negative} for e in mb.elements],
        "family": fam["family"], "familyMembers": fam["members"], "basis": fam["basis"],
        "isBasis": fam["basis"] == obj.name, "evaluated": ev,
        "isEmpty": (ev["faces"] == 0) if ev is not None else None,
    })
    if fam["basis"] != obj.name:
        out["familyNote"] = ("'%s' is in the family '%s', whose basis '%s' polygonizes every member: these elements "
                             "draw as part of '%s', and 'evaluated' is that object's mesh. A name outside the family "
                             "gives a separate surface." % (obj.name, fam["family"], fam["basis"], fam["basis"]))
    if ev is None:
        out["evaluatedNote"] = "the basis is not in this view layer, so it was not evaluated here"
    return out


def op_create_lattice(params):
    """A lattice object - the thing a LATTICE modifier deforms with, which nothing could create.

    FOUND BY THE PRODUCER/CONSUMER QUESTION, the sixth time it has paid: add_modifier learned to
    point a LATTICE modifier at an object on 2026-09-04, and bpy.data.lattices appears NOWHERE else
    in this addon. So the modifier could be aimed and there was nothing to aim it at. Same shape as
    vertex groups, shape keys, armatures, bones and material images before it.

    A DEFAULT LATTICE IS 2x2x2 AND CAN ONLY DEFORM LINEARLY. Eight corner points describe an affine
    transform and nothing else, so a lattice left at the default bends nothing however it is moved -
    it reads back as a perfectly healthy lattice and the mesh stays straight. That is reported
    rather than left to be discovered, and it is why `resolution` is the first parameter.

    THE MESH HAS TO BE INSIDE IT. A lattice only influences geometry within its own volume, so a
    lattice created at the origin at default scale will not touch a mesh standing somewhere else,
    and the modifier will report itself perfectly configured. Position and scale it over the mesh.

    params:
      name (str)
      resolution [u,v,w] (list[int])   or pointsU / pointsV / pointsW individually. Default 2,2,2.
      interpolation (str)              KEY_LINEAR | KEY_CARDINAL | KEY_BSPLINE | KEY_CATMULL_ROM,
                                       validated against this build's own enum. Applied to all
                                       three axes.
      useOutside (bool)                Blender's Outside option (Lattice.use_outside): only the lattice's OUTER points
                                       are drawn and used, and the inner ones follow them
      location / rotation / scale / collection
      fitTo [names] / margin           Blender's Lattice Deform Selected (2026-10-05): the lattice is placed and
                                       scaled over the objects' joint WORLD box (margin world units added on every
                                       side, default 0) and a LATTICE modifier aimed at it goes on each - the
                                       bounds read, the create and one add_modifier per object were three steps.
                                       Not with location, rotation or scale; every object must be a type a Lattice
                                       modifier deforms (MESH, CURVE, SURFACE, FONT, LATTICE). fittedTo in the reply.
    """
    # PARSED BEFORE ANYTHING IS CREATED. _place used to do this itself, after bpy.data.*.new
    # and after the object was linked, so a malformed vector refused with the object already in
    # the caller's file. Five ops shared _place and so shared the defect.
    _placed = _place_values(params)
    reject_unknown(params, _LATTICE_KEYS, "create_lattice")
    fit = _lattice_fit(params)
    res = params.get("resolution")
    if res is not None and (not isinstance(res, (list, tuple)) or len(res) < 3):
        raise MifOpError("'resolution' must be [u, v, w]. NOTHING was created.")
    pts = []
    for i, key in enumerate(("pointsU", "pointsV", "pointsW")):
        raw = take_int(params, key, default=None)
        if raw is None:
            # finite_int, as take_int is for pointsU/V/W: int(res[i]) raised a raw ValueError on NaN, a raw
            # OverflowError on Infinity, and turned 2.7 into 2 in silence (5.0.1, 2026-09-20).
            try:
                raw = finite_int(res[i], "resolution[%d]" % i) if res is not None else 2
            except MifOpError as exc:
                raise MifOpError("%s. NOTHING was created." % str(exc).rstrip("."))
        if raw < 1 or raw > 64:
            raise MifOpError("%s must be between 1 and 64, got %d - Blender clamps outside that "
                             "range and the lattice you get would not be the one you asked for. "
                             "NOTHING was created." % (key, raw))
        pts.append(int(raw))
    # useOutside AND scale, BEFORE ANYTHING EXISTS. Both were read at the end - scale after the lattice was
    # linked into the scene - so {useOutside: "flase"} and {scale: [NaN, 1, 1]} were refused with the lattice
    # object and its data left in the file (measured on 5.0.1, 2026-09-21), the name taken for the retry.
    try:
        use_outside = (take_bool(params, "useOutside", default=False)
                       if params.get("useOutside") is not None else None)
        want_scale = _vec3(params, "scale", (1.0, 1.0, 1.0)) if "scale" in params else None
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was created." % str(exc).rstrip("."))
    # interpolation TOO, against the TYPE's enum. It was read after the lattice and its object existed: the
    # enum refusal removed both, but take(kind=str) refusing a non-string ({interpolation: 5}) left them
    # (audit_mutate_then_deny's DEBRIS rule, measured on 5.0.1, 2026-09-21).
    interp = take(params, "interpolation", default=None, kind=str)
    interp_match = None
    if interp:
        valid = [i.identifier for i in bpy.types.Lattice.bl_rna.properties["interpolation_type_u"].enum_items]
        want = str(interp).upper()
        match = [v for v in valid if v.upper() == want]
        if not match:
            raise MifOpError("interpolation '%s' is not one this Blender offers. Valid: %s. "
                             "NOTHING was created." % (interp, ", ".join(valid)))
        interp_match = match[0]

    coll = _resolve_collection(params)
    wanted = str(take(params, "name", default="Lattice", kind=str))
    data = bpy.data.lattices.new(wanted)
    # THE OBJECT BEFORE THE RESOLUTION, or the lattice's size depends on its resolution. Blender
    # resizes a lattice through the first object using it; with none it spaces the new points ONE
    # UNIT apart, so [3, 5, 3] rested at -1..1 x -2..2 x -1..1 while the default 2x2x2 rests at
    # -0.5..0.5 - and "scale it over the mesh" came out wrong by a factor of (points - 1). Measured
    # 2026-09-13 on 4.4: data-only 3x5x3 first point (-1, -2, -1), object-first (-0.5, -0.5, -0.5).
    # Found by the fivekinds showcase, whose pinch squeezed a capsule to a quarter of its size. With
    # the object in place every resolution rests at -0.5..0.5, as Blender's own UI gives.
    obj = bpy.data.objects.new(data.name, data)
    # RESOLUTION NEXT, before anything else touches the lattice: changing points_u resets the
    # point positions, so writing it after any edit would silently discard the edit.
    data.points_u, data.points_v, data.points_w = pts

    if interp_match:
        data.interpolation_type_u = interp_match
        data.interpolation_type_v = interp_match
        data.interpolation_type_w = interp_match
    if use_outside is not None:
        data.use_outside = use_outside

    _link_new(obj, coll)
    _place(obj, _placed)
    # SCALE IS HANDLED HERE RATHER THAN IN _place, which does location and rotation only.
    # It matters more for a lattice than for anything else this file makes: a lattice only
    # influences geometry inside its own volume, so scaling it to cover the mesh is not a
    # cosmetic choice, it is the difference between deforming something and deforming
    # nothing. audit_blender_dead_params caught it accepted and unread.
    if want_scale is not None:
        obj.scale = want_scale
    fitted = None
    if fit is not None:
        # LATTICE DEFORM SELECTED (2026-10-05): the lattice over the objects' world box - its rest grid is -0.5..0.5,
        # so the box's size IS the scale - and a LATTICE modifier on each, aimed at it.
        f_objs, f_loc, f_size, (f_lo, f_hi, f_margin) = fit
        obj.location = f_loc
        obj.scale = f_size
        added = {}
        for ob in f_objs:
            m = ob.modifiers.new(name="Lattice", type="LATTICE")
            m.object = obj
            added[ob.name] = m.name
        bpy.context.view_layer.update()
        fitted = {"objects": [o.name for o in f_objs], "margin": f_margin,
                  "boundsMin": [round(x, 6) for x in f_lo], "boundsMax": [round(x, 6) for x in f_hi],
                  "modifiersAdded": sorted(added), "modifierNames": added}

    # READ BACK OFF THE DATABLOCK. points_u is clamped by Blender rather than refused, so the
    # resolution the lattice HAS is not necessarily the one that was asked for.
    got = [int(data.points_u), int(data.points_v), int(data.points_w)]
    # THE REST GRID, MEASURED OFF THE POINTS. set_point_positions writes in this space, so a caller
    # needs it - and assuming it is how the fivekinds showcase squeezed a capsule to a quarter size.
    rest = [tuple(p.co) for p in data.points]
    out = _created(obj, coll, "LATTICE", "create_lattice", wanted)
    out.update({
        "restBounds": {"min": [round(min(c[a] for c in rest), 6) for a in range(3)],
                       "max": [round(max(c[a] for c in rest), 6) for a in range(3)]},
        "resolution": got,
        "resolutionRequested": pts,
        "pointCount": got[0] * got[1] * got[2],
        "interpolation": str(data.interpolation_type_u),
        "useOutside": bool(data.use_outside),
        "fittedTo": fitted,
        # THE NUMBER THAT DECIDES WHETHER IT CAN BEND ANYTHING. Two points on an axis describe a
        # straight line and nothing else.
        "canDeformNonLinearly": any(p > 2 for p in got),
        "note": ("every axis is at 2 points, so this lattice can only apply an AFFINE deform - "
                 "moving its corners scales, shears or translates and cannot bend. Raise "
                 "resolution on at least one axis for anything else. It reads back as a perfectly "
                 "healthy lattice either way.") if not any(p > 2 for p in got) else
                ("a lattice only influences geometry INSIDE its own volume. Position and scale it "
                 "over the mesh before adding the LATTICE modifier, or the modifier will report "
                 "itself correctly configured and deform nothing."),
    })
    return out


def op_create_texture(params):
    """A legacy texture datablock - what a DISPLACE modifier reads, and nothing could create one.

    THE SAME PRODUCER/CONSUMER GAP as create_lattice above, found in the same pass. add_modifier can
    point DISPLACE at a texture and bpy.data.textures appears NOWHERE else in this addon.

    THESE ARE NOT SHADER NODES. bpy.data.textures is Blender's older texture system, and it is what
    the modifier stack reads - a DISPLACE modifier cannot take a ShaderNodeTexNoise. Both exist and
    they are not interchangeable, which is the confusion this op is most likely to be met with.

    params:
      name (str)
      type (str)         validated against this build's own enum. BLEND, CLOUDS, DISTORTED_NOISE,
                         IMAGE, MAGIC, MARBLE, MUSGRAVE, NOISE, STUCCI, VORONOI, WOOD - identical
                         on 3.6.23, 4.2.17, 4.4.0 and 5.0.1, read off the enum rather than assumed.
                         Default CLOUDS.
      image (str)        an image datablock, for type IMAGE. Refused for any other type.
      file (str)         OR an image on disk, loaded as that datablock (type IMAGE is implied): what a
                         Displace modifier reads a height map from, with no material needed to load it.
      colorspace (str)   the image's colour space with `file`. Default Non-Color: a displacement map
                         is data, and sRGB decoding would bend every height.
      intensity / contrast (float)
    """
    reject_unknown(params, _TEXTURE_KEYS, "create_texture")
    name = str(take(params, "name", default="Texture", kind=str))
    path = take(params, "file", default=None, kind=str)
    kind = str(take(params, "type", default="IMAGE" if path is not None else "CLOUDS", kind=str)).upper()
    valid = [i.identifier for i in bpy.types.Texture.bl_rna.properties["type"].enum_items]
    match = [v for v in valid if v.upper() == kind]
    if not match:
        raise MifOpError("texture type '%s' is not one this Blender offers. Valid: %s. NOTHING was "
                         "created." % (kind, ", ".join(valid)))
    kind = match[0]

    image_name = take(params, "image", default=None, kind=str)
    want_cs = take(params, "colorspace", default=None, kind=str)
    if image_name is not None and path is not None:
        raise MifOpError("pass `image` (a datablock already in the file) OR `file` (an image on disk), not both. "
                         "NOTHING was created.")
    if want_cs is not None and path is None:
        raise MifOpError("colorspace goes with `file` - an image already in the file keeps its own (set it where "
                         "it was loaded). NOTHING was created.")
    if (image_name is not None or path is not None) and kind != "IMAGE":
        raise MifOpError("'%s' only applies to an IMAGE texture and this one is %s. Accepting it "
                         "here and writing nothing is how a caller believes they attached an image "
                         "they did not. NOTHING was created." % ("image" if path is None else "file", kind))
    img = None
    if image_name is not None:
        img = bpy.data.images.get(str(image_name))
        if img is None:
            have = sorted(i.name for i in bpy.data.images)[:25]
            raise MifOpError("no image datablock named '%s'. This file has: %s. Load one with "
                             "set_material_texture, or import a file that carries it. NOTHING was "
                             "created." % (image_name, ", ".join(have) if have else "(none)"))

    # BEFORE bpy.data.textures.new. They were read after it, so {intensity: NaN} was refused with the texture
    # datablock left behind holding its name (measured on 5.0.1, 2026-09-21).
    try:
        intensity = take_float(params, "intensity", default=None)
        contrast = take_float(params, "contrast", default=None)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was created." % str(exc).rstrip("."))

    loaded_now = False
    cs_note = None
    if path is not None:
        from .ops_material import load_image_file   # noqa: PLC0415 - the one loader set_material_texture uses
        img, loaded_now = load_image_file(path, tail="NOTHING was created.")
        chosen = str(want_cs) if want_cs else "Non-Color"
        try:
            img.colorspace_settings.name = chosen
        except TypeError:
            cs_valid = [i.identifier for i in img.colorspace_settings.bl_rna.properties["name"].enum_items]
            if loaded_now:
                bpy.data.images.remove(img)
            raise MifOpError("'%s' is not a colour space on this Blender. Available: %s. NOTHING was created."
                             % (chosen, ", ".join(cs_valid)))
        if not loaded_now:
            cs_note = ("'%s' was already loaded, so the existing datablock is used and its colour space is now %s - "
                       "that applies everywhere it is used." % (img.name, img.colorspace_settings.name))

    tex = bpy.data.textures.new(name, type=kind)
    if img is not None:
        tex.image = img
    if intensity is not None:
        tex.intensity = intensity
    if contrast is not None:
        tex.contrast = contrast

    return {
        "ok": True,
        "texture": tex.name,
        "requestedName": name,
        "nameWasSuffixed": tex.name != name,
        "type": str(tex.type),
        "image": tex.image.name if getattr(tex, "image", None) else None,
        "imageFile": (bpy.path.abspath(tex.image.filepath) if path is not None and tex.image.filepath else None),
        "imageSize": list(tex.image.size) if path is not None else None,
        "colorSpace": tex.image.colorspace_settings.name if path is not None else None,
        "loadedFromDisk": loaded_now,
        "colorSpaceNote": cs_note,
        "intensity": round(float(tex.intensity), 6),
        "contrast": round(float(tex.contrast), 6),
        "users": tex.users,
        "note": ("this is a bpy.data.textures datablock, which is the OLD texture system and the "
                 "one the modifier stack reads - a DISPLACE modifier cannot take a shader node. "
                 "It is not the same thing as a ShaderNodeTexNoise and the two are not "
                 "interchangeable."),
    }


# ---------------------------------------------------------------------------
# create_mesh -- a mesh object from vertex, face and loose-edge lists
# ---------------------------------------------------------------------------

_CREATE_MESH_KEYS = {"name", "vertices", "faces", "edges", "materialIndices", "materials", "uvs", "vertexGroups",
                     "smooth", "location", "rotation", "scale", "collection", "takeOverName"}


def _name_holders(wanted):
    """The local object and mesh datablocks that already hold `wanted`, as (row, block) pairs. A row says whether the
    holder is an ORPHAN - nothing uses it (an object in no scene; a mesh no object uses, counting as unused an orphan
    object of the same name that is its only user, the leftover of an object unlinked rather than deleted). A fake
    user is a user: Blender keeps that block on purpose."""
    found = []
    obj = bpy.data.objects.get(wanted)
    obj = obj if obj is not None and obj.library is None else None
    mesh = bpy.data.meshes.get(wanted)
    mesh = mesh if mesh is not None and mesh.library is None else None
    if obj is not None:
        found.append(({"kind": "object", "name": obj.name, "users": obj.users, "fakeUser": bool(obj.use_fake_user),
                       "orphan": obj.users == 0}, obj))
    if mesh is not None:
        via_orphan = 1 if obj is not None and obj.users == 0 and obj.data is mesh else 0
        found.append(({"kind": "mesh", "name": mesh.name, "users": mesh.users, "fakeUser": bool(mesh.use_fake_user),
                       "orphan": mesh.users - via_orphan == 0}, mesh))
    return found
_CREATE_MESH_NOTES = {
    "verts": "the key is 'vertices': a list of [x, y, z] in the object's local space, in meters.",
    "polygons": "the key is 'faces': a list of vertex-index lists, three or more distinct indices each.",
    "indices": "faces are 'faces' (index lists) and loose edges are 'edges' ([a, b] pairs).",
    "uv": "the key is 'uvs': {layerName: one [u, v] list per face, one [u, v] per corner, in the face's own order}.",
    "normals": "normals are Blender's, computed from the faces' winding; 'smooth' chooses smooth or flat per face.",
    "weights": "the key is 'vertexGroups': {groupName: [[vertexIndex, weight], ...] or [vertexIndex, ...]}.",
    "materialIndex": "the key is 'materialIndices', one slot index per face.",
}
# A FACE BELOW THIS AREA (square meters) IS REPORTED, NOT REFUSED - a collinear triangle is a legal mesh, and the
# Unreal hand-off refuses zero-area faces itself; a caller should hear it at the build, not at the export.
_ZERO_AREA_M2 = 1e-12


def _index_list(raw, key, n_verts, min_len, tail):
    """One face or edge as a list of distinct vertex indices, every one in range - or a refusal naming it."""
    if not isinstance(raw, (list, tuple)):
        raise MifOpError("%s must be a list of vertex indices, got %r. %s" % (key, raw, tail))
    if len(raw) < min_len:
        raise MifOpError("%s has %d vertex index(es); %s needs at least %d. %s"
                         % (key, len(raw), "a face" if min_len == 3 else "an edge", min_len, tail))
    out = []
    for j, v in enumerate(raw):
        if isinstance(v, bool):
            raise MifOpError("%s[%d] must be a vertex index, got %r. %s" % (key, j, v, tail))
        try:
            i = finite_int(v, "%s[%d]" % (key, j))
        except MifOpError as exc:
            raise MifOpError("%s. %s" % (str(exc).rstrip("."), tail))
        if not 0 <= i < n_verts:
            raise MifOpError("%s[%d] is %d, out of range: there are %d vertices, so an index runs 0-%d. %s"
                             % (key, j, i, n_verts, n_verts - 1, tail))
        out.append(i)
    if len(set(out)) != len(out):
        raise MifOpError("%s repeats a vertex (%s): every corner must be a different vertex, or the face folds onto "
                         "itself. %s" % (key, out, tail))
    return out


def _parse_create_mesh(params, tail):
    """Everything create_mesh takes, parsed and cross-checked BEFORE a datablock exists."""
    raw_verts = take(params, "vertices", required=True)
    if not isinstance(raw_verts, (list, tuple)) or not raw_verts:
        raise MifOpError("'vertices' must be a non-empty list of [x, y, z]. %s" % tail)
    verts = []
    for i, v in enumerate(raw_verts):
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            raise MifOpError("vertices[%d] must be [x, y, z], got %r. %s" % (i, v, tail))
        try:
            verts.append(finite_floats(v, "vertices[%d]" % i))
        except MifOpError as exc:
            raise MifOpError("%s %s" % (exc, tail))
    n = len(verts)

    raw_faces = take(params, "faces", default=[])
    raw_edges = take(params, "edges", default=[])
    for key, raw in (("faces", raw_faces), ("edges", raw_edges)):
        if not isinstance(raw, (list, tuple)):
            raise MifOpError("'%s' must be a list, got %s. %s" % (key, type(raw).__name__, tail))
    faces = [_index_list(f, "faces[%d]" % i, n, 3, tail) for i, f in enumerate(raw_faces)]
    seen = {}
    for i, f in enumerate(faces):
        k = frozenset(f)
        if k in seen:
            raise MifOpError("faces[%d] uses the same vertices as faces[%d] (%s): Blender's mesh validation deletes a "
                             "duplicate face, which would shift every per-face list after it. %s"
                             % (i, seen[k], sorted(k), tail))
        seen[k] = i
    edges = []
    seen_e = {}
    for i, e in enumerate(raw_edges):
        pair = _index_list(e, "edges[%d]" % i, n, 2, tail)
        if len(pair) != 2:
            raise MifOpError("edges[%d] has %d indices; an edge is exactly [a, b]. %s" % (i, len(pair), tail))
        k = frozenset(pair)
        if k in seen_e:
            raise MifOpError("edges[%d] repeats edges[%d] %s. %s" % (i, seen_e[k], sorted(k), tail))
        seen_e[k] = i
        edges.append(pair)

    def per_face(key, raw):
        if not isinstance(raw, (list, tuple)) or len(raw) != len(faces):
            raise MifOpError("'%s' must be a list with one entry per face (%d), got %s. %s"
                             % (key, len(faces), len(raw) if isinstance(raw, (list, tuple)) else repr(raw), tail))
        return list(raw)

    mat_idx = None
    raw_mi = take(params, "materialIndices")
    if raw_mi is not None:
        mat_idx = []
        for i, m in enumerate(per_face("materialIndices", raw_mi)):
            try:
                v = finite_int(m, "materialIndices[%d]" % i) if not isinstance(m, bool) else None
            except MifOpError as exc:
                raise MifOpError("%s. %s" % (str(exc).rstrip("."), tail))
            if v is None or v < 0 or v > 32766:
                raise MifOpError("materialIndices[%d] must be a slot index 0-32766, got %r. %s" % (i, m, tail))
            mat_idx.append(v)
    materials = None
    raw_mats = take(params, "materials")
    if raw_mats is not None:
        if not isinstance(raw_mats, (list, tuple)) or not all(isinstance(m, str) and m for m in raw_mats):
            raise MifOpError("'materials' must be a list of material names, one per slot. %s" % tail)
        known = {mat.name for mat in bpy.data.materials}
        missing = [m for m in raw_mats if m not in known]
        if missing:
            raise MifOpError("no material named %s. Make one with create_material first. %s"
                             % (", ".join(repr(m) for m in missing), tail))
        materials = list(raw_mats)
        if mat_idx and max(mat_idx) >= len(materials):
            raise MifOpError("materialIndices reaches slot %d and 'materials' fills %d slot(s) (0-%d). %s"
                             % (max(mat_idx), len(materials), len(materials) - 1, tail))

    uvs = {}
    raw_uvs = take(params, "uvs")
    if raw_uvs is not None:
        if not isinstance(raw_uvs, dict) or not raw_uvs:
            raise MifOpError("'uvs' must be {layerName: per-face corner lists}. %s" % tail)
        if len(raw_uvs) > 8:
            raise MifOpError("'uvs' names %d layers; Blender holds 8 UV maps on a mesh. %s" % (len(raw_uvs), tail))
        for layer, rows in raw_uvs.items():
            if not isinstance(layer, str) or not layer.strip():
                raise MifOpError("a UV layer name must be a non-empty string, got %r. %s" % (layer, tail))
            rows = per_face("uvs['%s']" % layer, rows)
            flat = []
            for fi, row in enumerate(rows):
                if not isinstance(row, (list, tuple)) or len(row) != len(faces[fi]):
                    raise MifOpError("uvs['%s'][%d] must hold one [u, v] per corner of faces[%d] (%d), got %r. %s"
                                     % (layer, fi, fi, len(faces[fi]), row, tail))
                for ci, uv in enumerate(row):
                    if not isinstance(uv, (list, tuple)) or len(uv) != 2:
                        raise MifOpError("uvs['%s'][%d][%d] must be [u, v], got %r. %s" % (layer, fi, ci, uv, tail))
                    try:
                        flat.extend(finite_floats(uv, "uvs['%s'][%d][%d]" % (layer, fi, ci)))
                    except MifOpError as exc:
                        raise MifOpError("%s %s" % (exc, tail))
            uvs[layer] = flat

    groups = {}
    raw_groups = take(params, "vertexGroups")
    if raw_groups is not None:
        if not isinstance(raw_groups, dict):
            raise MifOpError("'vertexGroups' must be {groupName: [[vertexIndex, weight], ...] or [vertexIndex, ...]}. "
                             "%s" % tail)
        for gname, rows in raw_groups.items():
            if not isinstance(gname, str) or not gname.strip():
                raise MifOpError("a vertex group name must be a non-empty string, got %r. %s" % (gname, tail))
            if not isinstance(rows, (list, tuple)):
                raise MifOpError("vertexGroups['%s'] must be a list. %s" % (gname, tail))
            entries = {}
            for j, row in enumerate(rows):
                key = "vertexGroups['%s'][%d]" % (gname, j)
                if isinstance(row, (list, tuple)):
                    if len(row) != 2:
                        raise MifOpError("%s must be [vertexIndex, weight] or a bare index, got %r. %s"
                                         % (key, row, tail))
                    vi_raw, w_raw = row
                else:
                    vi_raw, w_raw = row, 1.0
                try:
                    if isinstance(vi_raw, bool):
                        raise MifOpError("'%s' must be a vertex index, got %r" % (key, vi_raw))
                    vi = finite_int(vi_raw, key)
                    w = finite_float(w_raw, key + " weight")
                except MifOpError as exc:
                    raise MifOpError("%s. %s" % (str(exc).rstrip("."), tail))
                if not 0 <= vi < n:
                    raise MifOpError("%s names vertex %d; there are %d vertices (0-%d). %s" % (key, vi, n, n - 1, tail))
                if not 0.0 <= w <= 1.0:
                    raise MifOpError("%s weight is %r; a vertex weight runs 0-1. %s" % (key, w, tail))
                if vi in entries:
                    raise MifOpError("%s names vertex %d a second time in group '%s'. %s" % (key, vi, gname, tail))
                entries[vi] = w
            groups[gname] = entries

    smooth = take(params, "smooth")
    if smooth is not None and not isinstance(smooth, (list, tuple)):
        smooth = [take_bool(params, "smooth")] * len(faces)
    elif smooth is not None:
        smooth = per_face("smooth", smooth)
        for i, s in enumerate(smooth):
            if not isinstance(s, bool):
                raise MifOpError("smooth[%d] must be true or false, got %r. %s" % (i, s, tail))
    return verts, faces, edges, mat_idx, materials, uvs, groups, smooth


def op_create_mesh(params):
    """A mesh object from raw data: vertices, faces, loose edges, and optionally per-face material slots, UV layers,
    vertex groups and smooth flags.

    WHY IT EXISTS. Building a short film (2026-10-02), every garment, car and furniture builder that had its
    geometry as lists - a pattern panel, a car body section, a table - left the typed ops for raw Python to call
    mesh.from_pydata. That is the one Blender call every procedural builder starts from, and this is it typed.

    EVERYTHING IS CHECKED BEFORE A DATABLOCK EXISTS, so a refusal leaves nothing behind: a face index out of range, a
    face with under three vertices or a repeated corner, a duplicate face (Blender's validation would delete it and
    shift every per-face list after it), an edge that is not [a, b], NaN or Infinity anywhere, a per-face list of the
    wrong length, a UV row whose corner count is not its face's, a vertex weight outside 0-1, a material that does not
    exist. After the build Blender's own mesh.validate runs; if it had to change anything the object is removed and
    the call refused.

    params:
      vertices [[x,y,z], ...]            required; local space, meters
      faces [[i,j,k,...], ...]           three or more distinct indices each, wound counter-clockwise seen from the
                                         side the normal should face
      edges [[a,b], ...]                 loose edges (an edge already on a face is reported in edgesOnFaces)
      materialIndices [int, ...]         one slot per face; without 'materials' the slots are made empty, for
                                         set_material_slots or assign_material to fill
      materials [name, ...]              existing materials, one per slot, in order
      uvs {layer: [[[u,v], ...], ...]}   one row per face, one [u,v] per corner in the face's order; the first layer
                                         is the active one
      vertexGroups {group: [[index, weight], ...] | [index, ...]}   a bare index means weight 1
      smooth (bool | [bool, ...])        all faces, or one per face; default flat
      name (str)                         default "Mesh"; Blender appends .001 on a collision and the reply says so -
                                         nameWasSuffixed for the object, meshNameWasSuffixed for its mesh, and
                                         nameHeldBy names each object or mesh already holding the name, its users and
                                         whether it is an orphan (2026-10-05: a rebuild after a delete that left the
                                         mesh behind came back '.001' with nothing saying why)
      takeOverName (bool)                default false; true removes the ORPHAN object and/or mesh holding the name
                                         (no users, no fake user) so both get it exactly, listed in nameTakenOverFrom;
                                         a holder in use is refused by name and nothing is created or removed
      location / rotation / scale / collection

    The reply reads every count back off the built mesh, and reports zeroAreaFaces (area under 1e-12 m2) - legal, but
    the Unreal hand-off refuses them.
    """
    tail = "NOTHING was created."
    _placed = _place_values(params)
    reject_unknown(params, _CREATE_MESH_KEYS, "create_mesh", _CREATE_MESH_NOTES)
    try:
        want_scale = _vec3(params, "scale", (1.0, 1.0, 1.0)) if "scale" in params else None
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
    verts, faces, edges, mat_idx, materials, uvs, groups, smooth = _parse_create_mesh(params, tail)
    coll = _resolve_collection(params)
    wanted = str(take(params, "name", default="Mesh", kind=str))
    if not wanted.strip():
        raise MifOpError("'name' must not be empty. %s" % tail)
    take_over = params.get("takeOverName")
    if take_over is not None and not isinstance(take_over, bool):
        raise MifOpError("'takeOverName' must be true or false, got %r. %s" % (take_over, tail))
    holders = _name_holders(wanted)
    if take_over:
        live = [row for row, _ in holders if not row["orphan"]]
        if live:
            raise MifOpError("takeOverName takes a name only from an orphan, and '%s' is held by %s - it is in use, so "
                             "it is not removed. Pick another name, or delete it first. %s" % (
                                 wanted, "; ".join("%s '%s' (%d user%s%s)" % (
                                     r["kind"], r["name"], r["users"], "" if r["users"] == 1 else "s",
                                     ", a fake user" if r["fakeUser"] else "") for r in live), tail))

    mesh = bpy.data.meshes.new(wanted)
    mesh.from_pydata(verts, edges, faces)
    mesh.update(calc_edges=True)
    changed = mesh.validate(verbose=False)
    if changed or len(mesh.vertices) != len(verts) or len(mesh.polygons) != len(faces):
        got = (len(mesh.vertices), len(mesh.polygons))
        bpy.data.meshes.remove(mesh)
        raise MifOpError("Blender's mesh validation had to change the data (vertices %d -> %d, faces %d -> %d), so the "
                         "per-face lists would no longer line up. %s" % (len(verts), got[0], len(faces), got[1], tail))
    if mat_idx is not None:
        mesh.polygons.foreach_set("material_index", mat_idx)
    if smooth is not None:
        mesh.polygons.foreach_set("use_smooth", [bool(s) for s in smooth])
    for layer, flat in uvs.items():
        uv = mesh.uv_layers.new(name=layer)
        if uv is None:
            bpy.data.meshes.remove(mesh)
            raise MifOpError("Blender refused UV layer '%s'. %s" % (layer, tail))
        uv.data.foreach_set("uv", flat)
    if uvs:
        mesh.uv_layers.active = mesh.uv_layers[0]
    if materials is not None:
        for m in materials:
            mesh.materials.append(bpy.data.materials[m])
    elif mat_idx:
        for _ in range(max(mat_idx) + 1):
            mesh.materials.append(None)
    mesh.update()

    # TAKING THE NAME OVER happens last, once nothing below can refuse: the orphans go (the object first - it may be
    # the orphan mesh's only user), then the new mesh takes the freed name.
    taken = []
    if take_over:
        for row, block in sorted(holders, key=lambda h: h[0]["kind"] != "object"):
            (bpy.data.objects if row["kind"] == "object" else bpy.data.meshes).remove(block)
            taken.append({"kind": row["kind"], "name": row["name"], "users": row["users"]})
        if taken:
            assign_id_name(mesh, wanted, bpy.data.meshes)

    # THE OBJECT ASKS FOR THE NAME ITSELF, not for the mesh's: an orphan MESH holding the name (its object deleted,
    # the mesh not purged) suffixed the mesh, and the object used to inherit that suffix with nothing holding its own.
    obj = bpy.data.objects.new(wanted, mesh)
    _link_new(obj, coll)
    _place(obj, _placed)
    if want_scale is not None:
        obj.scale = want_scale
    for gname, entries in groups.items():
        vg = obj.vertex_groups.new(name=gname)
        by_weight = {}
        for vi, w in entries.items():
            by_weight.setdefault(w, []).append(vi)
        for w, idx in by_weight.items():
            vg.add(idx, w, "REPLACE")

    out = _created(obj, coll, "MESH", "create_mesh", wanted)
    face_edges = {frozenset(k) for p in mesh.polygons for k in p.edge_keys}
    on_faces = [i for i, e in enumerate(edges) if frozenset(e) in face_edges]
    zero = [p.index for p in mesh.polygons if p.area < _ZERO_AREA_M2]
    counts = {}
    for v in mesh.vertices:
        for g in v.groups:
            counts[g.group] = counts.get(g.group, 0) + 1
    out.update({
        "meshName": mesh.name,
        "meshNameWasSuffixed": mesh.name != wanted,
        "nameHeldBy": [row for row, _ in holders] if holders and not taken else None,
        "nameTakenOverFrom": taken or None,
        "vertexCount": len(mesh.vertices), "edgeCount": len(mesh.edges), "faceCount": len(mesh.polygons),
        "looseEdgeCount": sum(1 for e in mesh.edges if e.is_loose),
        "edgesOnFaces": on_faces[:50] or None,
        "edgesOnFacesNote": ("%d of the 'edges' given are also face edges, so they are not loose" % len(on_faces))
                            if on_faces else None,
        "uvLayers": [u.name for u in mesh.uv_layers],
        "activeUvLayer": mesh.uv_layers.active.name if mesh.uv_layers.active else None,
        "materialSlots": [m.name if m else None for m in mesh.materials],
        "smoothFaceCount": sum(1 for p in mesh.polygons if p.use_smooth),
        "vertexGroups": [{"name": g.name, "vertexCount": counts.get(g.index, 0)} for g in obj.vertex_groups],
        "zeroAreaFaces": zero[:50] or None,
        "zeroAreaFaceCount": len(zero),
        "zeroAreaNote": ("%d face(s) have an area under %g m2 - legal here, but the Unreal hand-off refuses "
                         "zero-area faces" % (len(zero), _ZERO_AREA_M2)) if zero else None,
        "bboxLocal": {"min": rnd([min(v[a] for v in verts) for a in range(3)]),
                      "max": rnd([max(v[a] for v in verts) for a in range(3)])},
    })
    return out


# ---------------------------------------------------------------------------
# duplicate_object -- a copy under a new name, full or linked
# ---------------------------------------------------------------------------

_DUPLICATE_KEYS = ("objects", "object", "names", "suffix", "linked", "collection", "offset", "rotation", "scale",
                   "includeChildren", "hide")


def _data_collection(data):
    """The bpy.data collection a datablock lives in (meshes, curves, lights ...), or None when it cannot be told."""
    kind = getattr(data, "id_type", None)
    if not kind:
        return None
    guess = {"CURVES": "hair_curves", "LIGHT_PROBE": "lightprobes", "GREASEPENCIL": "grease_pencils",
             "GREASEPENCIL_V3": "grease_pencils_v3"}.get(kind, kind.lower() + "s")
    coll = getattr(bpy.data, guess, None)
    return coll if coll is not None and data.name in coll and coll.get(data.name) is data else None


def copy_object(src, linked, name=None):
    """A new object copied from `src`, linked to NO collection yet: its own copy of the data unless `linked`, the
    modifiers, constraints, vertex groups and material slots of the object, and its parent. Returns (copy, name asked
    for). The data takes the object's name when it is a full copy, through assign_id_name (3.6 renames the wrong block
    on a clash)."""
    new = src.copy()
    if not linked and src.data is not None:
        new.data = src.data.copy()
    wanted = name if name is not None else src.name
    assign_id_name(new, wanted, bpy.data.objects)
    if not linked and new.data is not None:
        holder = _data_collection(new.data)
        if holder is not None:
            assign_id_name(new.data, new.name, holder)
    return new, wanted


def copy_row(src, new, wanted, linked):
    return {
        "source": src.name,
        "name": new.name,
        "requestedName": wanted,
        "nameWasSuffixed": new.name != wanted,
        "data": new.data.name if new.data is not None else None,
        "sharesData": new.data is not None and new.data is src.data,
        "collections": sorted(c.name for c in new.users_collection),
        "parent": new.parent.name if new.parent is not None else None,
        "location": rnd(list(new.matrix_world.to_translation())),
    }


def op_duplicate_object(params):
    """Duplicate objects under new names: a full copy (its own mesh) or a linked one (the same mesh, as Alt+D).

    THE GAP, 2026-10-04. The most basic object operation had no typed call - importing a file twice is refused, and
    build_lod_chain's copies are not a general duplicate. A shell made from a copy of the donut, a light made from the
    sun and edited, six sprinkle variants from one, a simulated shirt kept as a hidden backup, a high-poly copy for a
    bake: every one went through raw Python.

    objects (a list) or object (one). names (one per object) or suffix (added to each source name); with neither,
    each copy asks for its source's name and Blender appends .001 - the reply echoes the name every copy actually got,
    with nameWasSuffixed. linked (default false) shares the data: an edit to one mesh then shows on both, which is what
    a linked duplicate is for. Materials are shared either way (Blender's own default), and so is any action.
    collection puts every copy there; by default each copy goes into its source's collections. offset [x,y,z] (world
    units) moves each top-level copy, rotation [x,y,z] (radians, as transform_object) turns it about its own origin and
    scale multiplies its scale. includeChildren copies each object's children too, parented to the copies. hide
    'original' or 'copy' hides that one in the view layer and the render (a kept backup). Every name and target is
    checked before anything is made.
    """
    reject_unknown(params, _DUPLICATE_KEYS, "duplicate_object")
    tail = "NOTHING was created."
    raw = take(params, "objects", "object")
    if raw is None:
        raise MifOpError("pass objects (a list of names) or object (one name). %s" % tail)
    names_in = [raw] if isinstance(raw, str) else raw
    if not isinstance(names_in, (list, tuple)) or not names_in or not all(isinstance(n, str) for n in names_in):
        raise MifOpError("objects must be a non-empty list of object names. %s" % tail)
    if len(set(names_in)) != len(names_in):
        raise MifOpError("objects names one object twice. %s" % tail)
    sources = [get_object(n) for n in names_in]
    new_names = params.get("names")
    if isinstance(new_names, str):
        new_names = [new_names]
    suffix = take(params, "suffix", kind=str)
    if new_names is not None and suffix is not None:
        raise MifOpError("pass names or suffix, not both - each says what the copies are called. %s" % tail)
    if new_names is not None:
        if not isinstance(new_names, (list, tuple)) or len(new_names) != len(sources) \
                or not all(isinstance(n, str) and n for n in new_names):
            raise MifOpError("names must give one non-empty name per object (%d objects, got %r). %s"
                             % (len(sources), new_names, tail))
        if len(set(new_names)) != len(new_names):
            raise MifOpError("names gives two copies the same name. %s" % tail)
    linked = take_bool(params, "linked", default=False)
    include_children = take_bool(params, "includeChildren", default=False)
    hide = take(params, "hide", kind=str)
    if hide is not None and hide.lower() not in ("original", "copy"):
        raise MifOpError("hide must be 'original' or 'copy' (got %r). %s" % (hide, tail))
    target = None
    coll_name = take(params, "collection", kind=str)
    if coll_name is not None:
        target = bpy.data.collections.get(coll_name)
        if target is None:
            scene = bpy.context.scene
            if coll_name == scene.collection.name or coll_name == scene.name:
                target = scene.collection
            else:
                known = sorted(c.name for c in bpy.data.collections)[:30]
                raise MifOpError("no collection named '%s'. Present: %s. %s"
                                 % (coll_name, ", ".join(known) or "<none>", tail))
    try:
        offset = _vec3(params, "offset", (0, 0, 0)) if params.get("offset") is not None else None
        rotation = _vec3(params, "rotation", (0, 0, 0)) if params.get("rotation") is not None else None
        scale = _vec3(params, "scale", (1, 1, 1)) if params.get("scale") is not None else None
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
    if scale is not None and any(s == 0.0 for s in scale):
        raise MifOpError("scale %s has a 0, which flattens the copy. %s" % (rnd(scale), tail))
    if include_children:
        # a source that is a descendant of another source is copied with it, once
        def descends(o, of):
            p = o.parent
            while p is not None:
                if p in of:
                    return True
                p = p.parent
            return False
        chosen = set(sources)
        doubled = [o.name for o in sources if descends(o, chosen)]
        if doubled:
            raise MifOpError("%s %s already copied as the child of another object in objects (includeChildren). "
                             "Leave %s out. %s" % (", ".join(doubled), "is" if len(doubled) == 1 else "are",
                                                   "it" if len(doubled) == 1 else "them", tail))

    rows, made = [], []
    for i, src in enumerate(sources):
        wanted = new_names[i] if new_names is not None else (src.name + suffix if suffix is not None else None)
        new, asked = copy_object(src, linked, wanted)
        made.append(new)
        homes = [target] if target is not None else (list(src.users_collection) or [bpy.context.scene.collection])
        for home in homes:
            home.objects.link(new)
        family = [(src, new)]
        if include_children:
            stack = [(src, new)]
            while stack:
                parent_src, parent_new = stack.pop()
                for child in parent_src.children:
                    c_new, c_asked = copy_object(child, linked, None)
                    c_new.parent = parent_new
                    c_new.matrix_parent_inverse = child.matrix_parent_inverse.copy()
                    for home in ([target] if target is not None else
                                 (list(child.users_collection) or [bpy.context.scene.collection])):
                        home.objects.link(c_new)
                    made.append(c_new)
                    family.append((child, c_new))
                    stack.append((child, c_new))
        if scale is not None:
            new.scale = [a * b for a, b in zip(new.scale, scale)]
        if rotation is not None:
            write_rotation(new, rotation, True)
        bpy.context.view_layer.update()
        if offset is not None:
            from mathutils import Vector   # local, as op_transform_objects imports it
            mw = new.matrix_world.copy()
            mw.translation = mw.translation + Vector(offset)
            new.matrix_world = mw
        if hide is not None:
            hidden = new if hide.lower() == "copy" else src
            hidden.hide_render = True
            try:
                hidden.hide_set(True)
            except RuntimeError:
                hidden.hide_viewport = True
        rows.append((family, asked))
    bpy.context.view_layer.update()

    copies = []
    for family, asked in rows:
        src, new = family[0]
        row = copy_row(src, new, asked, linked)
        if len(family) > 1:
            row["children"] = [copy_row(s, n, s.name, linked) for s, n in family[1:]]
        copies.append(row)
    view_objects = bpy.context.view_layer.objects
    outside = [o.name for o in made if o.name not in view_objects]
    out = {
        "copies": copies,
        "count": len(made),
        "linked": linked,
        "collection": target.name if target is not None else None,
        "materialsShared": True,
    }
    if hide is not None:
        hidden = [r["name"] if hide.lower() == "copy" else r["source"] for r in copies]
        out["hidden"] = {"which": hide.lower(), "objects": hidden,
                         "hiddenInViewLayer": [bpy.data.objects[n].hide_get() for n in hidden],
                         "hideRender": [bpy.data.objects[n].hide_render for n in hidden]}
    if outside:
        out["notInViewLayer"] = outside
        out["viewLayerNote"] = ("%d cop%s sit in a collection this view layer does not show (excluded, or not under "
                                "the scene), so they exist and do not draw or render here."
                                % (len(outside), "y" if len(outside) == 1 else "ies"))
    return out


OPS = {
    "create_primitive": op_create_primitive,
    "create_lattice": op_create_lattice,
    "create_curves": op_create_curves,
    "create_point_cloud": op_create_point_cloud,
    "create_volume": op_create_volume,
    "create_nurbs_surface": op_create_nurbs_surface,
    "create_metaball": op_create_metaball,
    "create_texture": op_create_texture,
    "transform_object": op_transform_object,
    "transform_objects": op_transform_objects,
    "set_parent": op_set_parent,
    "boolean_op": op_boolean_op,
    "join_objects": op_join_objects,
    "separate_mesh": op_separate_mesh,
    "convert_object": op_convert_object,
    "create_empty": op_create_empty,
    "create_force_field": op_create_force_field,
    "create_curve": op_create_curve,
    "set_curve": op_set_curve,
    "create_text": op_create_text,
    "set_text": op_set_text,
    "create_armature": op_create_armature,
    "create_mesh": op_create_mesh,
    "duplicate_object": op_duplicate_object,
}
