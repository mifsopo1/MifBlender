"""Rigid bodies, cloth, collision and softbody - simulation, and the bake that makes it real.

=============================================================================
RIGID BODIES NEED A WORLD, AND blender's OWN OPERATOR IS THE ONLY THING THAT MAKES ONE
=============================================================================
An object cannot simply be given a rigid_body: the setting lives on the object but the SIMULATION
lives in a scene-level RigidBodyWorld with its own collection, and without it nothing falls.
bpy.ops.rigidbody.object_add() creates that world on demand; assigning obj.rigid_body directly is
not possible at all. So this op goes through the operator, with the context override that makes it
work when nothing is selected - which is the normal state for a bridge call.

=============================================================================
A SIMULATION THAT HAS NOT BEEN BAKED DOES NOT EXIST AT A GIVEN FRAME
=============================================================================
Physics is evaluated by stepping FORWARD from the start frame. Jumping straight to frame 200 shows
the object at its rest position, not where it would have fallen to, and a render of that frame is
simply wrong. Nothing about the scene says so. So bake_physics is here as its own op, and the
docstrings say plainly that setting up a sim without baking it leaves you with a scene that looks
right in the viewport only if you happened to scrub through it.
"""
import math

import os
import re
import shutil
import time

import bpy

from .ops_common import (remove_objects, MifOpError, exclude_rows, finite_floats, get_object, limit_rows, reject_unknown, rnd,
                         tag_fluid_domain, take, take_bool, take_float, take_int, FLUID_FRAME,
                         fluid_frame_numbers)

_RB_KEYS = {"object", "name", "type", "mass", "friction", "bounciness", "restitution",
            "collisionShape", "kinematic", "margin", "linearDamping", "angularDamping",
            "objects", "collection", "copyFrom", "update", "density", "material", "useDeactivation",
            "startDeactivated", "deactivateLinearVelocity", "deactivateAngularVelocity", "enabled"}
_CLOTH_KEYS = {"object", "name", "quality", "mass", "stiffness", "damping", "gravity",
               "usePressure", "pressure", "collisionQuality", "selfCollision",
               "pinGroup", "pinStiffness", "preset", "bending", "bendingDamping", "bendingModel",
               "bendingGroup", "bendingMax", "pressureGroup", "pressureFactor"}
_COLL_KEYS = {"object", "name", "damping", "friction", "thickness", "remove"}
_SOFTBODY_KEYS = {"object", "name", "mass", "friction", "speed", "useGoal", "goalDefault",
                  "goalSpring", "pull", "push", "bend", "useEdges", "selfCollision", "gravity",
                  "goalVertexGroup", "goalMin", "goalMax", "goalFriction"}
_RBC_KEYS = {"name", "type", "object1", "object2", "location", "rotation", "disableCollisions",
             "breakingThreshold", "enabled", "objects", "active", "pattern", "pivot", "collection",
             "solverIterations", "useBreaking", "update", "names"}

# THE AXIS EACH JOINT TYPE WORKS ALONG, in the constraint object's own frame. MEASURED 2026-09-16 on 3.6.23, 4.2.17,
# 4.4.0, 5.0.1 and 5.2.1: a passive anchor and a plank sticking out along +X, the joint at the anchor's face, gravity -Z,
# stepped 60 frames, the joint in three orientations. HINGE swung only when the joint was turned 90 deg about X (local Z
# laid along world Y) - so its axis is local Z, vertical by default, and a default hinge is a door. SLIDER and PISTON
# dropped only when turned 90 deg about Y (local X pointing down): local X. POINT swung identically in all three. A
# MOTOR with gravity off turned about local X (world X unrotated and turned about X, world Z when turned about Y).
# Second seat, report 87 defect 5: nothing took a rotation and nothing said which way the axis pointed.
_RBC_AXIS = {"HINGE": "Z", "SLIDER": "X", "PISTON": "X", "MOTOR": "X"}

# THE SAME ROLE-SPLIT AS FLUID, for the same reason: a CANVAS receives paint and a BRUSH applies
# it, they carry different fields, and a setting sent to the wrong one must be refused naming the
# role that owns it rather than dropped.
_DP_COMMON = {"object", "name", "type"}
_DP_BY_TYPE = {
    "CANVAS": {"surfaceType", "surfaceFormat", "frameStart", "frameEnd", "useDissolve",
               "dissolveSpeed"},
    "BRUSH": {"paintColor", "paintAlpha", "paintWetness", "paintSource", "paintDistance",
              "useAbsoluteAlpha"},
}
# WRITTEN OUT, NOT COMPUTED - parity_check reads accept-lists STATICALLY and a computed set is
# invisible to it, which cost a false contract break on add_fluid an hour ago. The assertion below
# keeps the literal and the roles in step.
_DP_KEYS = {"object", "name", "type",
            "surfaceType", "surfaceFormat", "frameStart", "frameEnd", "useDissolve",
            "dissolveSpeed",
            "paintColor", "paintAlpha", "paintWetness", "paintSource", "paintDistance",
            "useAbsoluteAlpha"}
assert _DP_KEYS == set(_DP_COMMON).union(*_DP_BY_TYPE.values()), (
    "_DP_KEYS is out of step with _DP_BY_TYPE: %s"
    % sorted(_DP_KEYS ^ set(_DP_COMMON).union(*_DP_BY_TYPE.values())))


# GROUPED BY WHICH fluid_type OWNS THEM, so a setting sent to the wrong role is REFUSED naming the
# role it belongs to rather than silently ignored. Passing resolutionMax with type:FLOW is a caller
# who believes they set a resolution; dropping it quietly is the failure this file refuses.
_FLUID_COMMON = {"object", "name", "type"}
_FLUID_BY_TYPE = {
    "DOMAIN": {"domainType", "resolutionMax", "cacheStart", "cacheEnd", "useAdaptiveDomain", "useMesh",
               "timeScale", "cacheType", "cacheDirectory", "cacheResumable"},
    "FLOW": {"flowType", "flowBehavior", "useInflow", "surfaceDistance", "subframes"},
    "EFFECTOR": {"effectorType", "useEffector", "surfaceDistance"},
}
# WRITTEN OUT, NOT COMPUTED, and the assertion below is why both forms exist. parity_check reads
# every reject_unknown accept-list STATICALLY - it is the tool that guarantees the MCP server and
# this addon agree - and it cannot evaluate a for-loop. Building this by iterating _FLUID_BY_TYPE
# left it seeing only {name, object, type}, so it reported bl_add_fluid sending twelve params the
# endpoint "refuses". The contract was fine; the checker could not see it, which for a static
# checker is the same thing.
_FLUID_KEYS = {"object", "name", "type",
               "domainType", "resolutionMax", "cacheStart", "cacheEnd", "useAdaptiveDomain", "useMesh",
               "timeScale", "cacheType", "cacheDirectory", "cacheResumable",
               "flowType", "flowBehavior", "useInflow", "surfaceDistance", "subframes",
               "effectorType", "useEffector"}

# AND THE TWO CANNOT DRIFT. A literal the checker can read plus a computed union it must equal - if
# a role gains a setting and this list does not, the addon refuses to import rather than accepting a
# key no role owns or advertising one no role accepts.
assert _FLUID_KEYS == set(_FLUID_COMMON).union(*_FLUID_BY_TYPE.values()), (
    "_FLUID_KEYS is out of step with _FLUID_BY_TYPE: %s"
    % sorted(_FLUID_KEYS ^ set(_FLUID_COMMON).union(*_FLUID_BY_TYPE.values())))
# NO "type". It sat on this list and was read nowhere: bake_physics goes through
# bpy.ops.ptcache.bake_all, which bakes EVERY cache in the scene and takes no filter, so
# there was nothing for a type to select. A caller could send type:"CLOTH", pass the guard,
# and have every cache baked anyway - a parameter that appears to narrow the work and does
# not. Per-type baking is a different implementation and is filed in the spec.
_BAKE_KEYS = {"start", "end", "clear", "rebake", "object", "modifier", "diskCache", "toFrame", "free",
               # one fluid domain stage by stage (2026-10-04)
               "stages", "resume", "freeStages",
               # one geometry-nodes Bake node (1.3) - read by _bake_node
               "bakeNode", "bakeMode"}


def _ctx_override(obj):
    """A context the physics operators will accept from a headless bridge call.

    bpy.ops.rigidbody.* reads the ACTIVE object and the current view layer, neither of which a
    socket call has set up. temp_override is the supported way to supply them; without it the
    operator raises a poll failure that says nothing about the real problem.
    """
    win = bpy.context.window
    return bpy.context.temp_override(
        window=win,
        scene=bpy.context.scene,
        view_layer=bpy.context.view_layer,
        object=obj,
        active_object=obj,
        selected_objects=[obj],
        selected_editable_objects=[obj],
    )


def _rigid_body_state(rb):
    """A rigid body's settings, read off the body - ONE reader for add_rigid_body and physics_info.

    TWO COPIES DRIFTED, which is why this exists (2026-09-16, reported by the second seat). physics_info
    read ten fields; add_rigid_body's own response read six of the same body and never linearDamping,
    angularDamping, collisionMargin or marginEnabled - three of which it had just WRITTEN. A caller who
    set them learned nothing from the op that set them and had to ask a second op.
    """
    return {
        "type": rb.type,
        "mass": round(float(rb.mass), 6),
        "friction": round(float(rb.friction), 6),
        "restitution": round(float(rb.restitution), 6),
        "collisionShape": rb.collision_shape,
        "kinematic": bool(rb.kinematic),
        # THE MARGIN IS THE ONE NUMBER HERE WITH A GATE IN FRONT OF IT, and until 2026-09-15 only the
        # number was reported. rb.collision_margin always holds a value - Blender's default is 0.04 - but
        # it applies only while rb.use_margin is on, and add_rigid_body turns that on ONLY when the
        # caller passes `margin`. It is never turned off.
        #
        # MEASURED on a headless 5.0.1: add_rigid_body with no margin at all, and physics_info answered
        # collisionMargin 0.04 - a real number, for a margin NOT IN EFFECT, with nothing in the response
        # to say so. Mass, friction and the damping pair have no such gate - their defaults are always
        # in effect, so reporting them bare is honest. This one needed its flag.
        "collisionMargin": round(float(rb.collision_margin), 6),
        "marginEnabled": bool(getattr(rb, "use_margin", False)),
        "linearDamping": round(float(rb.linear_damping), 6),
        "angularDamping": round(float(rb.angular_damping), 6),
        "enabled": bool(getattr(rb, "enabled", True)),
        # DEACTIVATION, read since 2026-10-04 when add_rigid_body learned to write it: a wall kept inert until struck
        # is these four, and a body whose startDeactivated is on but useDeactivation off never sleeps at all.
        # getattr, as `enabled` above: physics_info's offline checks (test_blender_refusals B115) hand it a stand-in.
        "useDeactivation": bool(getattr(rb, "use_deactivation", False)),
        "startDeactivated": bool(getattr(rb, "use_start_deactivated", False)),
        "deactivateLinearVelocity": round(float(getattr(rb, "deactivate_linear_velocity", 0.0)), 6),
        "deactivateAngularVelocity": round(float(getattr(rb, "deactivate_angular_velocity", 0.0)), 6),
    }


def _rb_targets(params, op_name):
    """(objects, skipped, bulk) for add_rigid_body: `object`, `objects` [names] or `collection`, exactly one.

    A NON-MESH IS SKIPPED AND NAMED, not refused, when it came in a list or a collection: Blender's own Copy from Active
    deselects non-meshes and carries on, while rigidbody.objects_add with one non-mesh selected refuses the WHOLE call
    ("Can't add Rigid Body to non mesh object", measured 2026-10-04 on 3.6.23, 4.4.0 and 5.2.1). A single `object` that is
    not a mesh is refused as it always was. A list that leaves nothing to work on is refused: an empty result is not a
    success."""
    given = [k for k in ("object", "name", "objects", "collection") if params.get(k) is not None]
    given = [k for k in given if not (k == "name" and "object" in given)]    # name is object's alias
    if len(given) != 1:
        raise MifOpError("%s takes exactly one of object, objects [names] or collection - got %s. NOTHING was changed."
                         % (op_name, ", ".join(given) or "none"))
    which = given[0]
    if which in ("object", "name"):
        return [get_object(take(params, "object", "name", required=True), want_mesh=True)], [], False
    if which == "objects":
        names = params.get("objects")
        if not isinstance(names, (list, tuple)) or not names or not all(isinstance(n, str) for n in names):
            raise MifOpError("objects must be a non-empty list of object names, got %r. NOTHING was changed." % (names,))
        missing = [n for n in names if bpy.data.objects.get(n) is None]
        if missing:
            raise MifOpError("no object named %s. NOTHING was changed." % ", ".join("'%s'" % n for n in missing[:12]))
        seen, pool = set(), []
        for n in names:
            if n not in seen:
                seen.add(n)
                pool.append(bpy.data.objects[n])
    else:
        cname = take(params, "collection", kind=str)
        coll = bpy.data.collections.get(cname)
        if coll is None:
            raise MifOpError("no collection named '%s'. Present: %s. NOTHING was changed."
                             % (cname, ", ".join(sorted(c.name for c in bpy.data.collections)[:20]) or "<none>"))
        pool = sorted(coll.all_objects, key=lambda o: o.name)
    objs = [o for o in pool if o.type == "MESH"]
    skipped = [{"object": o.name, "why": "a %s, not a mesh - a rigid body needs a mesh" % o.type} for o in pool
               if o.type != "MESH"]
    if not objs:
        raise MifOpError("none of the %d object(s) given is a mesh (%s), so there is nothing to make a rigid body. "
                         "NOTHING was changed." % (len(pool), ", ".join(s["object"] for s in skipped[:12]) or "none"))
    return objs, skipped, True


def _closed_volume(obj, depsgraph):
    """(volume m3 in world scale, None) of the EVALUATED mesh, or (None, why) when it is not closed.

    A VOLUME IS DEFINED ONLY FOR A CLOSED MESH: an edge with one face (a hole) or three (a fin) leaves the inside
    undefined, and bmesh.calc_volume would still return a number for it. So every edge must carry exactly two faces."""
    import bmesh
    bm = bmesh.new()
    try:
        bm.from_object(obj, depsgraph)
        bm.transform(obj.matrix_world)
        if not bm.faces:
            return None, "it has no faces"
        open_edges = sum(1 for e in bm.edges if len(e.link_faces) != 2)
        if open_edges:
            return None, "%d of its %d edges do not join exactly two faces, so it is not closed" % (open_edges,
                                                                                                  len(bm.edges))
        return abs(float(bm.calc_volume(signed=True))), None
    finally:
        bm.free()


def _mass_presets(obj):
    """The material presets rigidbody.mass_calculate offers on this build, read from the operator itself.

    ITS ENUM IS DYNAMIC: get_rna_type() lists only DEFAULT on 3.6.23, 4.4.0 and 5.2.1, while the operator takes 46 named
    presets and Custom. Blender lists the live set when a value it does not know is passed, before poll or exec run -
    measured on all three builds, nothing is changed by the refused call - so that list is read rather than typed."""
    try:
        with bpy.context.temp_override(scene=bpy.context.scene, view_layer=bpy.context.view_layer, object=obj,
                                       active_object=obj, selected_objects=[], selected_editable_objects=[]):
            bpy.ops.rigidbody.mass_calculate(material="\x01mif-probe")
    except (TypeError, ValueError, RuntimeError) as exc:
        text = str(exc)
        if "not found in (" in text:
            inner = text[text.index("not found in (") + len("not found in ("):text.rindex(")")]
            return [p.strip().strip("'\"") for p in inner.split("',")]
    return []


def _preset_density(obj, preset):
    """kg/m3 of one of Blender's mass presets, measured on `obj` (already a rigid body) and restored.

    BLENDER DOES NOT EXPOSE THE TABLE, and mass_calculate does not store its density where Python can read it
    (operator_properties_last read 1.0 / Air after every run, measured on 3.6.23 and 5.2.1). Its volume rule is the
    collision shape's: BOX is the object's dimensions multiplied (a 2 x 1 x 1 cube read 4640 kg for Concrete as BOX,
    CONVEX_HULL and MESH, and 9718 kg as SPHERE). So the shape is set to BOX for one run, the density is mass / that
    volume, and the shape and mass are put back."""
    rb = obj.rigid_body
    shape, mass = rb.collision_shape, rb.mass
    dims = obj.dimensions
    box = float(dims[0]) * float(dims[1]) * float(dims[2])
    try:
        rb.collision_shape = "BOX"
        with bpy.context.temp_override(scene=bpy.context.scene, view_layer=bpy.context.view_layer, object=obj,
                                       active_object=obj, selected_objects=[obj], selected_editable_objects=[obj]):
            bpy.ops.rigidbody.mass_calculate(material=preset)
        measured = float(rb.mass)
    finally:
        rb.collision_shape = shape
        rb.mass = mass
    if box <= 0.0:
        raise MifOpError("'%s' has a zero dimension, so the density of preset '%s' cannot be read off it." % (obj.name,
                                                                                                         preset))
    return measured / box


def _check_preset(obj, material):
    """The preset's name as Blender spells it, or a refusal listing this build's presets. `obj` is a rigid body."""
    presets = _mass_presets(obj)
    match = {p.lower(): p for p in presets}.get(str(material).strip().lower())
    if match is None or match == "Custom":
        raise MifOpError("material '%s' is not one of this Blender's (%s) mass presets: %s. For your own value pass "
                         "density (kg/m3). NOTHING was changed."
                         % (material, bpy.app.version_string,
                            ", ".join(p for p in presets if p != "Custom") or "(none could be read)"))
    return match


def _take_back_bodies(objs, remove_world):
    """Remove the rigid bodies this call just made - and the world, when the call made that too."""
    with bpy.context.temp_override(window=bpy.context.window, scene=bpy.context.scene,
                                   view_layer=bpy.context.view_layer, object=objs[0], active_object=objs[0],
                                   selected_objects=objs, selected_editable_objects=objs):
        bpy.ops.rigidbody.objects_remove()
        if remove_world and bpy.context.scene.rigidbody_world is not None:
            bpy.ops.rigidbody.world_remove()


# BLENDER'S OWN COPY FROM ACTIVE (bl_operators/rigidbody.py CopyRigidbodySettings._attrs, the same 18 on 3.6, 4.4 and
# 5.2). Copied in this order, each only where this build has it.
_RB_COPY_ATTRS = ("type", "kinematic", "mass", "collision_shape", "use_margin", "collision_margin", "friction",
                  "restitution", "use_deactivation", "use_start_deactivated", "deactivate_linear_velocity",
                  "deactivate_angular_velocity", "linear_damping", "angular_damping", "collision_collections",
                  "mesh_source", "use_deform", "enabled")


def op_add_rigid_body(params):
    """Make objects ACTIVE (falls) or PASSIVE (is fallen onto) rigid bodies, or update bodies that exist.

    params:
      object (str) | objects [names] | collection (str)     exactly one. Non-meshes in a list are skipped and named.
      type            ACTIVE | PASSIVE, default ACTIVE (copyFrom's type with copyFrom; unchanged with update)
      update (bool)   write the settings of bodies that ALREADY exist. Without it an existing body is refused:
                      Blender's object_add re-types an existing body to the call's type (measured), so a settings
                      call without type turned a PASSIVE floor ACTIVE.
      copyFrom (str)  a rigid body whose 18 settings (Blender's Copy from Active) every target takes first
      mass (float)              kg, ACTIVE only
      density (float) | material (str)   mass = density x the evaluated mesh's closed volume in world scale;
                      material is one of Blender's mass presets, read from the operator on this build
      friction / bounciness     0..1  (restitution is an alias for bounciness)
      collisionShape            CONVEX_HULL | MESH | BOX | SPHERE | CAPSULE | CYLINDER | CONE
      kinematic (bool)          animated rather than simulated - keyframes drive it
      margin, linearDamping, angularDamping
      useDeactivation, startDeactivated, deactivateLinearVelocity, deactivateAngularVelocity
      enabled (bool)            Blender's Dynamic flag - ACTIVE only, refused on a PASSIVE body
    """
    reject_unknown(params, _RB_KEYS, "add_rigid_body", {
        "dynamic": "spelt enabled - Blender's Dynamic checkbox is RigidBodyObject.enabled, ACTIVE only.",
        "deactivated": "spelt startDeactivated (use_start_deactivated); useDeactivation turns deactivation on.",
        "volume": "the volume is measured, not given: pass density (kg/m3) or material and the mass is density x the "
                  "evaluated mesh's closed volume.",
    })
    objs, skipped, bulk = _rb_targets(params, "add_rigid_body")
    update = take_bool(params, "update", default=False)
    kind_raw = take(params, "type", default=None, kind=str)
    kind = str(kind_raw).upper() if kind_raw is not None else None
    if kind is not None and kind not in ("ACTIVE", "PASSIVE"):
        raise MifOpError("rigid body type must be ACTIVE or PASSIVE, got '%s'. NOTHING was changed." % kind)

    # EVERY SETTING READ AND CHECKED HERE, BEFORE object_add. They were read after the body existed, and the
    # refusals said so ("The body WAS created.") - or, for a bad number from take_float, did not say it at all:
    # {mass: NaN} was refused with the object already a rigid body, and a scene that had none gained a
    # RigidBodyWorld as well (measured on 5.0.1, 2026-09-21). Every one of these can be judged without the body -
    # the shape enum is on the TYPE's bl_rna - so a refusal now changes nothing.
    try:
        mass = take_float(params, "mass", default=None)
        density = take_float(params, "density", default=None)
        fr = take_float(params, "friction", default=None)
        bo = take_float(params, "bounciness", "restitution", default=None)
        kinematic = take_bool(params, "kinematic", default=False) if "kinematic" in params else None
        mg = take_float(params, "margin", default=None)
        ld = take_float(params, "linearDamping", default=None)
        ad = take_float(params, "angularDamping", default=None)
        use_deact = take_bool(params, "useDeactivation", default=True) if "useDeactivation" in params else None
        start_deact = take_bool(params, "startDeactivated", default=False) if "startDeactivated" in params else None
        dlin = take_float(params, "deactivateLinearVelocity", default=None)
        dang = take_float(params, "deactivateAngularVelocity", default=None)
        enabled = take_bool(params, "enabled", default=True) if "enabled" in params else None
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was changed." % str(exc).rstrip("."))
    material = take(params, "material", default=None, kind=str)
    if sum(v is not None for v in (mass, density, material)) > 1:
        raise MifOpError("mass, density and material are three answers to one question - the body's mass. Pass one. "
                         "NOTHING was changed.")
    if density is not None and density <= 0.0:
        raise MifOpError("density must be above 0 kg/m3, got %g. NOTHING was changed." % density)
    for key, val in (("deactivateLinearVelocity", dlin), ("deactivateAngularVelocity", dang)):
        if val is not None and val < 0.0:
            raise MifOpError("%s must be 0 or more, got %g. NOTHING was changed." % (key, val))
    shape = take(params, "collisionShape", default=None, kind=str)
    if shape:
        valid_shapes = {i.identifier for i in
                        bpy.types.RigidBodyObject.bl_rna.properties["collision_shape"].enum_items}
        if str(shape).upper() not in valid_shapes:
            raise MifOpError("unknown collisionShape '%s'. Valid: %s. NOTHING was changed."
                             % (shape, ", ".join(sorted(valid_shapes))))

    source = None
    src_name = take(params, "copyFrom", default=None, kind=str)
    if src_name is not None:
        source = get_object(src_name)
        if getattr(source, "rigid_body", None) is None:
            raise MifOpError("copyFrom '%s' is not a rigid body, so it has no settings to copy. NOTHING was changed."
                             % source.name)
        objs = [o for o in objs if o.name != source.name] or objs

    # NEW OR EXISTING, judged against update before anything is written.
    existing = [o.name for o in objs if o.rigid_body is not None]
    fresh = [o.name for o in objs if o.rigid_body is None]
    if update and fresh:
        raise MifOpError("update writes the settings of bodies that exist, and %s %s not a rigid body yet. Leave update "
                         "out to make them. NOTHING was changed." % (", ".join(fresh[:12]),
                                                                     "is" if len(fresh) == 1 else "are"))
    if not update and existing:
        raise MifOpError(
            "%s %s already a rigid body. Pass update: true to change its settings - adding again RE-TYPES an existing "
            "body to this call's type (ACTIVE when none is given; measured on 3.6.23, 4.4.0 and 5.2.1, a PASSIVE floor "
            "became ACTIVE), so it is not done implicitly. NOTHING was changed."
            % (", ".join(existing[:12]), "is" if len(existing) == 1 else "are"))

    def _final_type(o):
        if kind is not None:
            return kind
        if update:
            return o.rigid_body.type
        if source is not None:
            return source.rigid_body.type
        return "ACTIVE"

    passive = [o.name for o in objs if _final_type(o) == "PASSIVE"]
    if passive and (mass is not None or density is not None or material is not None):
        raise MifOpError("a mass applies to an ACTIVE rigid body; %s %s PASSIVE and is never moved by the simulation. "
                         "NOTHING was changed." % (", ".join(passive[:12]), "is" if len(passive) == 1 else "are"))
    if passive and enabled is not None:
        raise MifOpError("enabled is Blender's Dynamic flag and belongs to an ACTIVE body (the panel shows it for ACTIVE "
                         "only); %s %s PASSIVE. NOTHING was changed."
                         % (", ".join(passive[:12]), "is" if len(passive) == 1 else "are"))

    preset = None
    probe = next((o for o in objs if o.rigid_body is not None), None)
    if material is not None and probe is not None:
        preset = _check_preset(probe, material)

    volumes = {}
    if density is not None or material is not None:
        deps = bpy.context.evaluated_depsgraph_get()
        bad = []
        for o in objs:
            vol, why = _closed_volume(o, deps)
            if vol is None:
                bad.append("'%s' (%s)" % (o.name, why))
            else:
                volumes[o.name] = vol
        if bad:
            raise MifOpError("a mass from a density needs a closed mesh, whose volume is defined: %s. Close the mesh or "
                             "pass mass in kg. NOTHING was changed." % "; ".join(bad[:8]))

    # WHETHER THE WORLD WAS ALREADY BAKED, read before the body joins it - see bakedWorldNote below.
    rbw_before = bpy.context.scene.rigidbody_world
    pc_before = getattr(rbw_before, "point_cache", None) if rbw_before is not None else None
    baked_before = (((pc_before.info or "").strip() or "baked") if pc_before is not None and pc_before.is_baked
                    else None)

    to_add = [o for o in objs if o.rigid_body is None]
    if to_add:
        first_type = _final_type(to_add[0])
        groups = {}
        for o in to_add:
            groups.setdefault(_final_type(o), []).append(o)
        for t, group in sorted(groups.items(), key=lambda kv: kv[0] != first_type):
            try:
                # ONE CALL PER TYPE over the whole group: rigidbody.objects_add walks selected_objects, so a 300-piece
                # slab is one operator call, not 300.
                with bpy.context.temp_override(window=bpy.context.window, scene=bpy.context.scene,
                                               view_layer=bpy.context.view_layer, object=group[0],
                                               active_object=group[0], selected_objects=group,
                                               selected_editable_objects=group):
                    if len(group) == 1:
                        bpy.ops.rigidbody.object_add(type=t)
                    else:
                        bpy.ops.rigidbody.objects_add(type=t)
            except RuntimeError as exc:
                raise MifOpError("Blender refused to add a rigid body to %s: %s. This is usually a mesh with no faces, "
                                 "or a scene whose rigid body world could not be created. %s"
                                 % (", ".join(o.name for o in group[:8]), exc,
                                    "NOTHING was changed." if t == first_type else
                                    "The %s bodies before them WERE added." % first_type))
        still = [o.name for o in to_add if o.rigid_body is None]
        if still:
            raise MifOpError("the operator reported success and %s still %s no rigid_body - so the scene has no usable "
                             "RigidBodyWorld." % (", ".join(still[:8]), "has" if len(still) == 1 else "have"))
        if material is not None and preset is None:
            # THE PRESET NAME IS JUDGED HERE ONLY WHEN NO TARGET WAS A BODY BEFORE: mass_calculate's poll wants an active
            # rigid body before it reads its arguments (a mesh that is not one read "Context missing active object",
            # measured on 3.6.23, 4.4.0 and 5.2.1). A refusal takes back the bodies, and the world, it just made.
            try:
                preset = _check_preset(to_add[0], material)
            except MifOpError:
                _take_back_bodies(to_add, rbw_before is None)
                raise

    # WRITES ONLY - every value was read and checked above, before any body existed.
    density_used = density
    if preset is not None:
        density_used = _preset_density(objs[0], preset)
    rows = []
    for o in objs:
        rb = o.rigid_body
        if source is not None:
            src = source.rigid_body
            for attr in _RB_COPY_ATTRS:
                if hasattr(src, attr) and hasattr(rb, attr):
                    val = getattr(src, attr)
                    setattr(rb, attr, list(val) if attr == "collision_collections" else val)
        if kind is not None:
            rb.type = kind
        if mass is not None:
            rb.mass = mass
        if density_used is not None:
            rb.mass = density_used * volumes[o.name]
        if fr is not None:
            rb.friction = fr
        if bo is not None:
            rb.restitution = bo
        if shape:
            rb.collision_shape = str(shape).upper()
        if kinematic is not None:
            rb.kinematic = kinematic
        if mg is not None:
            rb.use_margin = True
            rb.collision_margin = mg
        if ld is not None:
            rb.linear_damping = ld
        if ad is not None:
            rb.angular_damping = ad
        if use_deact is not None:
            rb.use_deactivation = use_deact
        if start_deact is not None:
            # START DEACTIVATED IS INERT WITHOUT DEACTIVATION: the panel greys it while use_deactivation is off. Asked
            # for, so deactivation is turned on with it rather than left as a flag that does nothing.
            rb.use_start_deactivated = start_deact
            if start_deact and not rb.use_deactivation:
                rb.use_deactivation = True
        if dlin is not None:
            rb.deactivate_linear_velocity = dlin
        if dang is not None:
            rb.deactivate_angular_velocity = dang
        if enabled is not None:
            rb.enabled = enabled
        row = {"object": o.name}
        row.update(_rigid_body_state(rb))
        if density_used is not None:
            row["volume"] = round(volumes[o.name], 6)
        rows.append(row)

    scene = bpy.context.scene
    rbw = scene.rigidbody_world
    world_count = len(rbw.collection.objects) if rbw and rbw.collection else 0
    if bulk:
        out = {"objects": rows, "count": len(rows), "skipped": skipped or None, "updated": bool(update),
               "worldObjectCount": world_count}
    else:
        out = dict(rows[0])
        out.update({"updated": bool(update), "worldObjectCount": world_count})
    if source is not None:
        out["copiedFrom"] = source.name
    if density_used is not None:
        out.update({"density": round(density_used, 6), "material": preset,
                    "totalMass": round(sum(r["mass"] for r in rows), 6),
                    "massRule": ("density x the evaluated mesh's closed volume in world scale. Blender's own Calculate "
                                 "Mass uses the collision shape's volume instead: the same 2 x 1 x 1 box read 4640 kg "
                                 "of Concrete as BOX, CONVEX_HULL or MESH and 9718 kg as SPHERE (measured).")})
    out["bakeNote"] = ("a rigid body is stepped forward from the start frame. Jumping to a late frame shows it at REST, "
                       "not where it would have fallen - call bake_physics before rendering that frame.")
    # HOW OFTEN THE WORLD STEPS, AND WHAT THAT DOES TO BOUNCINESS. Reported by the second seat
    # (2026-09-16): restitution 0.15 / 0.6 / 0.95 at 60 fps gave identical curves and no bounce at all,
    # with every field echoed correctly. MEASURED on a clean scene outside the addon, 4.4.0 and 5.2.1
    # identical - a sphere dropped 3 m onto a box, restitution 0.95 on both, baked - and it is the
    # SIMULATION STEP RATE, fps x substeps_per_frame, not the fps and not the collision margin (0.005
    # changed nothing):
    #     steps/s   240 -> rebound 1.41-1.57 m    300 -> 0.91    360 -> 0.43    420 -> 0.13    480+ -> 0.00
    # 30 fps x 10 and 60 fps x 5 agree, as do 60 x 8, 30 x 16 and 24 x 20. Blender's default is 10
    # substeps, so a 60 fps scene steps 600 times a second and restitution is inert in it.
    if rbw is not None and getattr(rbw, "substeps_per_frame", None):
        fps = float(scene.render.fps) / float(scene.render.fps_base or 1.0)
        steps = fps * rbw.substeps_per_frame
        out["stepsPerSecond"] = round(steps, 3)
        if any(float(o.rigid_body.restitution) > 0.0 for o in objs) and steps >= 360.0:
            out["restitutionNote"] = (
                "this world steps %g times a second (%g fps x %d substeps), and bounciness is %s at that "
                "rate - measured, restitution 0.95 on a 3 m drop rebounds 1.4 m at 240 steps/s, 0.43 m at "
                "360 and not at all from 480 up. Lower set_physics_world substeps (e.g. to %d) if the "
                "bounce is meant to be seen."
                % (steps, fps, rbw.substeps_per_frame, "much reduced" if steps < 480.0 else "INERT",
                   max(1, int(240.0 // fps))))
    # A BODY ADDED TO A BAKED WORLD IS NOT IN ITS FRAMES, and is not freed here: bake_physics keeps an existing bake
    # on purpose, since it can be minutes of simulation. MEASURED 2026-09-16, identical on 3.6.23, 4.2.17, 4.4.0,
    # 5.0.1 and 5.2.1: bodies A and B baked over 1-40, then a cube C added at z 3 - C stood at z 3.0 on frames 1, 20
    # and 40 while A and B played their bake, and a plain bake_physics kept that bake (C still at 3.0). Freeing
    # the bake straight after adding C made the next bake drop it (z 0.37 by frame 20).
    if baked_before:
        names = ", ".join(o.name for o in objs[:8]) + (" ..." if len(objs) > 8 else "")
        out["bakedWorldNote"] = (
            "this scene's rigid body world was ALREADY BAKED (%s) when %s %s, and those frames do not include it - "
            "measured, a body added after a bake does not move at all, and bake_physics keeps an existing bake. "
            "Call bake_physics {rebake: true} before rendering."
            % (baked_before, names, "changed" if update else "joined it"))
    # BOUNCINESS NEEDS BOTH SURFACES. MEASURED 2026-09-16, identical on all five builds: a 0.25 sphere dropped 3 m
    # onto a floor, 24 fps x 10 substeps, baked - sphere 0.99 on a floor at 0 rebounds 0.00 m, floor 0.99 under a
    # sphere at 0 rebounds 0.00 m, both at 0.99 rebounds 2.29 m. Only the bodies already in the world can be named.
    if bo is not None and bo > 0.0 and rbw is not None and getattr(rbw, "collection", None) is not None:
        mine = {o.name for o in objs}
        flat = sorted(o.name for o in rbw.collection.objects
                      if o.name not in mine and getattr(o, "rigid_body", None) is not None
                      and float(o.rigid_body.restitution) == 0.0)
        if flat:
            out["bouncinessNote"] = (
                "bounciness acts only where BOTH surfaces have it - measured, a sphere at 0.99 dropped 3 m rebounds "
                "0.00 m off a floor at 0 and 2.29 m off a floor at 0.99. Bounciness is 0 on %d bod%s in this world: "
                "%s. Give whatever these bodies land on bounciness too."
                % (len(flat), "y" if len(flat) == 1 else "ies", ", ".join(flat[:12]) + (" ..." if len(flat) > 12
                                                                                    else "")))
    return out


# FABRIC PRESETS (spec "Extend bl_add_cloth with fabric presets, bending stiffness and pressure by region", 2026-10-06).
# Blender ships presets/cloth Cotton, Denim, Leather, Rubber and Silk on every supported build; each is a script of
# `bpy.context.cloth.settings.<attr> = <value>` lines that runs only inside the Properties panel, so they are read
# here from the RUNNING build's own files and set field by field - the 5.0+ files add 14 lines (internal springs, and
# use_pressure False with uniform_pressure_force 76.0) that 4.x's do not. FLEECE and CANVAS are MifBridge's own
# starting points, not Blender presets and not measured fabrics: fleece a knit between Cotton and Denim with the
# bending that keeps a hood up, canvas a firm woven near Denim with Leather-like bending. Their reply says so.
_OWN_CLOTH_PRESETS = {
    "FLEECE": {"quality": 7, "mass": 0.45, "tension_stiffness": 20, "compression_stiffness": 20,
               "shear_stiffness": 15, "bending_stiffness": 5, "tension_damping": 10, "compression_damping": 10,
               "shear_damping": 10, "air_damping": 1},
    "CANVAS": {"quality": 12, "mass": 0.6, "tension_stiffness": 60, "compression_stiffness": 60,
               "shear_stiffness": 60, "bending_stiffness": 40, "tension_damping": 25, "compression_damping": 25,
               "shear_damping": 25, "air_damping": 1},
}
_PRESET_LINE = re.compile(r"^bpy\.context\.cloth\.settings\.(\w+)\s*=\s*(.+?)\s*$")


def _build_cloth_presets():
    """{NAME: path} of the running build's presets/cloth files."""
    found = {}
    try:
        dirs = bpy.utils.preset_paths("cloth")
    except Exception:   # noqa: BLE001 - a build without the presets folder lists none
        dirs = []
    for d in dirs:
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for fn in names:
            if fn.endswith(".py"):
                found.setdefault(fn[:-3].upper(), os.path.join(d, fn))
    return found


def _read_cloth_preset(name):
    """(values {attr: value}, source) for a preset name, or a refusal naming this build's list."""
    import ast   # noqa: PLC0415
    files = _build_cloth_presets()
    key = str(name).strip().upper()
    if key in _OWN_CLOTH_PRESETS:
        return dict(_OWN_CLOTH_PRESETS[key]), "MifBridge (a starting point, not a Blender preset)"
    if key not in files:
        raise MifOpError("preset %r is not a cloth preset of this Blender (%s): it ships %s; MifBridge adds %s. NOTHING "
                         "was added." % (name, bpy.app.version_string, ", ".join(sorted(files)) or "none",
                                         ", ".join(sorted(_OWN_CLOTH_PRESETS))))
    values = {}
    with open(files[key], encoding="utf-8") as fh:
        for line in fh:
            m = _PRESET_LINE.match(line.strip())
            if m:
                try:
                    values[m.group(1)] = ast.literal_eval(m.group(2))
                except (ValueError, SyntaxError):
                    continue
    if not values:
        raise MifOpError("preset file %s sets no cloth setting this op can read. NOTHING was added." % files[key])
    return values, os.path.basename(files[key])


def _cloth_group(obj, params, key):
    name = take(params, key, default=None, kind=str)
    if name is not None and name not in obj.vertex_groups:
        have = sorted(g.name for g in obj.vertex_groups)
        raise MifOpError("%s '%s' is not a vertex group on '%s' (it has: %s). Make it first with set_vertex_weights. "
                         "NOTHING was added." % (key, name, obj.name, ", ".join(have) if have else "none"))
    return name


_PRESSURE_LAYERS_NOTE = (
    "pressure is one value per object (pressureGroup only scales it 0-1 by vertex): a lining joined at the rims into "
    "one closed shell (add_inner_layer join RIM) with positive pressure pushes its two walls apart - what an outer +50 "
    "against an inner -10 does in a pattern tool. Opposite signs on two regions of one object have no Blender setting, "
    "and Blender cloth has no collision layer, so layered pieces are built and draped in order.")


def op_add_cloth(params):
    """Add a cloth simulation. The object needs enough subdivisions to drape at all.

    preset COTTON | DENIM | LEATHER | RUBBER | SILK reads the running build's presets/cloth file and sets it field by
    field (FLEECE and CANVAS are MifBridge's own starting points); explicit keys override it and the reply lists the
    values in effect. bending, bendingDamping, bendingModel (ANGULAR | LINEAR), bendingGroup and bendingMax set the
    bending springs - what decides whether a hood stands; pressureGroup and pressureFactor scale pressure by region.
    """
    reject_unknown(params, _CLOTH_KEYS, "add_cloth", {
        # THE RESPONSE'S NAME SENT BACK IN (second seat, mailbox id 30). The reply says gravityWeight because the value
        # weights the scene's gravity, and physics_info reports the scene's own gravity VECTOR as `gravity`.
        "gravityWeight": "spelt gravity on the way in - a weight on the scene's gravity, which the response reports as "
                         "gravityWeight to keep it apart from physics_info's gravity vector",
    })
    obj = get_object(take(params, "object", "name", required=True), want_mesh=True)
    if len(obj.data.vertices) < 16:
        raise MifOpError("'%s' has %d vertices, which cannot drape - cloth deforms the mesh it is "
                         "given and a quad has nothing to bend. Subdivide it first. NOTHING was "
                         "added." % (obj.name, len(obj.data.vertices)))
    # THE GUARD ITS TWO NEIGHBOURS ALREADY HAD, and cloth was the one without it. add_soft_body
    # (:278) and add_fluid both refuse a second modifier by name; add_cloth went straight to
    # modifiers.new, which RETURNS None rather than raising when Blender declines - so a second
    # add_cloth on one object crashed on `mod.settings` and the caller got a raw
    # "AttributeError: 'NoneType' object has no attribute 'settings'".
    #
    # Measured 2026-09-10, and the first reading was WRONG: audit_blender_inert_params reported it
    # against selfCollision, so it looked like a flag defect. On a FRESH grid selfCollision true and
    # false both succeed - the flag is fine. What crashes is the second call on the same object,
    # which the probe hit because its fixture had already put cloth on MifGrid.
    existing = [m for m in obj.modifiers if m.type == "CLOTH"]
    if existing:
        raise MifOpError("'%s' already has a cloth modifier ('%s'). Blender allows a second one and "
                         "the result is not a stronger simulation, it is two solvers fighting over "
                         "the same vertices. NOTHING was added." % (obj.name, existing[0].name))

    # THE PIN GROUP, checked BEFORE the modifier exists so a bad name adds nothing. A cloth with no pin
    # has nothing holding it and falls off the frame - the second seat's wind scene did exactly that on
    # 2026-09-16, with a 46-vertex Pin group built and no parameter to hand it to. add_soft_body had
    # goalVertexGroup for the same job; cloth's is ClothSettings.vertex_group_mass ("Pin Group" in the
    # UI), present on every build this addon supports.
    pin_group = take(params, "pinGroup", default=None, kind=str)
    if pin_group is not None and pin_group not in obj.vertex_groups:
        have = sorted(g.name for g in obj.vertex_groups)
        raise MifOpError("pinGroup '%s' is not a vertex group on '%s' (it has: %s). Make it first with "
                         "set_vertex_weights. NOTHING was added."
                         % (pin_group, obj.name, ", ".join(have) if have else "none"))
    pin_stiffness = take_float(params, "pinStiffness", default=None)
    if pin_stiffness is not None and pin_stiffness < 0.0:
        raise MifOpError("pinStiffness must be >= 0, got %g. NOTHING was added." % pin_stiffness)
    bending_group = _cloth_group(obj, params, "bendingGroup")
    pressure_group = _cloth_group(obj, params, "pressureGroup")
    preset_name = take(params, "preset", default=None, kind=str)
    preset_vals, preset_src = _read_cloth_preset(preset_name) if preset_name is not None else (None, None)
    bending_model = take(params, "bendingModel", default=None, kind=str)
    if bending_model is not None:
        bending_model = bending_model.upper()
        if bending_model not in ("ANGULAR", "LINEAR"):
            raise MifOpError("bendingModel must be ANGULAR or LINEAR, got %r. NOTHING was added." % bending_model)
    # THE REST OF THE SETTINGS, READ HERE - before the modifier exists, as pinGroup and pinStiffness above already
    # were. They were read after modifiers.new, so {mass: NaN} was refused with the Cloth modifier left on the
    # object (measured on 5.0.1, 2026-09-21), and the retry the refusal invites hits "already has a cloth
    # modifier".
    try:
        # quality AND collisionQuality ARE INTS, read with take_int: take_float then int() floored {quality: 5.5} to 5
        # under ok:true (the 2026-09-21 pack-rule review). A fraction is refused, as take_int refuses it everywhere.
        q = take_int(params, "quality", default=None)
        m = take_float(params, "mass", default=None)
        sf = take_float(params, "stiffness", default=None)
        dp = take_float(params, "damping", default=None)
        pressure_flag_given = "usePressure" in params
        use_pressure = take_bool(params, "usePressure", default=False) if pressure_flag_given else None
        pr = take_float(params, "pressure", default=None)
        gv = take_float(params, "gravity", default=None)
        cq = take_int(params, "collisionQuality", default=None)
        self_col = take_bool(params, "selfCollision", default=False) if "selfCollision" in params else None
        bend = {k: take_float(params, k, default=None) for k in ("bending", "bendingDamping", "bendingMax",
                                                                 "pressureFactor")}
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was added." % str(exc).rstrip("."))
    for k, v in bend.items():
        if v is not None and not (math.isfinite(v) and v >= 0):
            raise MifOpError("%s must be a finite number >= 0, got %r. NOTHING was added." % (k, v))

    mod = obj.modifiers.new(name="Cloth", type="CLOTH")
    # BELT AND BRACES, because the guard above covers the case that was measured and not the class.
    # bpy's modifiers.new returns None on any refusal - a locked library object, a mesh in an
    # unexpected state - and every one of those would land on the same unguarded dereference.
    if mod is None:
        raise MifOpError("Blender declined to add a cloth modifier to '%s' and gave no reason - "
                         "modifiers.new returned None. NOTHING was added." % obj.name)
    st = mod.settings
    preset_set = {}
    if preset_vals:
        have = st.bl_rna.properties.keys()
        for attr, val in preset_vals.items():
            if attr in have and not st.bl_rna.properties[attr].is_readonly:
                setattr(st, attr, val)
                preset_set[attr] = val
    for key, attr in (("bending", "bending_stiffness"), ("bendingDamping", "bending_damping"),
                      ("bendingMax", "bending_stiffness_max"), ("pressureFactor", "pressure_factor")):
        if bend[key] is not None:
            setattr(st, attr, bend[key])
    if bending_model is not None:
        st.bending_model = bending_model
    if bending_group is not None:
        st.vertex_group_bending = bending_group
    if pressure_group is not None:
        st.vertex_group_pressure = pressure_group
    if q is not None:
        st.quality = int(q)
    if m is not None:
        st.mass = m
    if sf is not None:
        for attr in ("tension_stiffness", "compression_stiffness", "shear_stiffness"):
            if hasattr(st, attr):
                setattr(st, attr, sf)
    if dp is not None:
        for attr in ("tension_damping", "compression_damping", "shear_damping"):
            if hasattr(st, attr):
                setattr(st, attr, dp)
    # AN EXPLICIT usePressure WINS. `pressure` alone still turns the feature on - a force with nothing
    # to apply it would be a strange thing to ask for - but beside an explicit usePressure it no longer
    # overrides it. It used to: the flag was written, then `pressure` set use_pressure True over it, so
    # {pressure:0.0, usePressure:false} came back usePressure TRUE (the second seat, 2026-09-16; pinned
    # as "the precedence wart" in test_blender_physics_params' advisory 5). The value is still stored,
    # and a note says it is not in effect.
    if pressure_flag_given:
        st.use_pressure = use_pressure
    if pr is not None:
        st.uniform_pressure_force = pr
        if not pressure_flag_given:
            st.use_pressure = True
    if pin_group is not None:
        st.vertex_group_mass = pin_group
    if pin_stiffness is not None:
        st.pin_stiffness = pin_stiffness
    if gv is not None:
        # Cloth gravity is a WEIGHT on the scene's gravity, not an acceleration - 1.0 is normal,
        # 0.0 makes the cloth float. It was in the accept list and unread until param_reach said so.
        st.effector_weights.gravity = gv
    if cq is not None:
        mod.collision_settings.collision_quality = int(cq)
    if self_col is not None:
        mod.collision_settings.use_self_collision = self_col
    extra = {"bending": round(float(st.bending_stiffness), 6), "bendingDamping": round(float(st.bending_damping), 6),
             "bendingModel": str(st.bending_model), "bendingGroup": st.vertex_group_bending or None,
             "bendingMax": round(float(st.bending_stiffness_max), 6), "pressureGroup": st.vertex_group_pressure or None,
             "pressureFactor": round(float(st.pressure_factor), 6), "shrinkMin": round(float(st.shrink_min), 6)}
    if preset_vals is not None:
        now = {}
        for attr in preset_set:
            v = getattr(st, attr)
            now[attr] = round(float(v), 6) if isinstance(v, float) else v
        given = {"quality": ("quality",), "mass": ("mass",), "usePressure": ("use_pressure",),
                 "pressure": ("uniform_pressure_force", "use_pressure"), "bending": ("bending_stiffness",),
                 "bendingDamping": ("bending_damping",), "bendingMax": ("bending_stiffness_max",),
                 "pressureFactor": ("pressure_factor",), "bendingModel": ("bending_model",),
                 "stiffness": ("tension_stiffness", "compression_stiffness", "shear_stiffness"),
                 "damping": ("tension_damping", "compression_damping", "shear_damping")}
        over = {a for k, attrs in given.items() if params.get(k) is not None for a in attrs if a in preset_set}

        # a value Blender stored differently from the file - 5.x's internal_spring_max_diversion = 45 is an angle in
        # radians, clamped to 45 degrees, as the panel's own run of the file does
        differ = sorted(a for a, want in preset_set.items() if a not in over and now[a] != (
            round(float(want), 6) if isinstance(now[a], float) else want))
        extra["preset"] = {"name": preset_name.upper(), "source": preset_src,
                           "values": {a: now[a] for a in sorted(now)}, "overridden": sorted(over),
                           "storedDifferently": differ}
        if preset_set.get("use_pressure") is False and not st.use_pressure:
            extra["presetPressureNote"] = (
                "the %s preset on Blender %s turns pressure OFF and leaves uniform_pressure_force at %g; pass "
                "usePressure or pressure to turn it on." % (preset_name.upper(), bpy.app.version_string,
                                                            float(st.uniform_pressure_force)))
    if st.use_pressure or pressure_group is not None or bend["pressureFactor"] is not None:
        extra["pressureLayersNote"] = _PRESSURE_LAYERS_NOTE
    return {"object": obj.name, "modifier": mod.name, "quality": int(st.quality), **extra,
            "mass": round(float(st.mass), 6), "vertices": len(obj.data.vertices),
            "gravityWeight": round(float(st.effector_weights.gravity), 6),
            "selfCollision": bool(mod.collision_settings.use_self_collision),
            # usePressure WAS WRITTEN AND NEVER REPORTED, beside selfCollision which always was -
            # so the inert-parameter probe read it as doing nothing. Read back from the settings,
            # which also makes the override visible: a `pressure` turns pressure ON even beside an
            # explicit usePressure:false, and this field is where a caller sees that it did.
            "usePressure": bool(st.use_pressure),
            "pressure": round(float(st.uniform_pressure_force), 6),
            "pinGroup": st.vertex_group_mass or None,
            "pinStiffness": round(float(st.pin_stiffness), 6),
            # STIFFNESS AND DAMPING WERE WRITTEN AND READ BACK BY NOTHING - not here, not physics_info (second seat,
            # mailbox ids 57 and 61). One value goes to three springs, so all three are read back: a build that lacks
            # one shows it missing rather than a single number standing in for three.
            "stiffness": {k: round(float(getattr(st, a)), 6) for k, a in (("tension", "tension_stiffness"),
                          ("compression", "compression_stiffness"), ("shear", "shear_stiffness")) if hasattr(st, a)},
            "damping": {k: round(float(getattr(st, a)), 6) for k, a in (("tension", "tension_damping"),
                        ("compression", "compression_damping"), ("shear", "shear_damping")) if hasattr(st, a)},
            **({"pressureNote": "pressure %g is stored but NOT in effect - usePressure was given as "
                                "false, and an explicit flag wins." % pr}
               if pr is not None and pressure_flag_given and not st.use_pressure else {}),
            "bakeNote": "cloth is stepped forward like any sim - bake_physics before a late frame."}


def op_add_soft_body(params):
    """Add a soft body simulation - the deformable one, between cloth and a rigid body.

    THIS MODULE HAS CLAIMED SOFTBODY SINCE IT WAS WRITTEN and nothing could make one. Its first line
    says "Rigid bodies, cloth, collision and softbody", `_SIM_MODIFIERS` lists SOFT_BODY so
    bake_physics and physics_info both handle it, and add_collision's own docstring says it collides
    with softbody. Every consumer, no creator. Found 2026-09-07 by surveying the addon against
    Blender's feature surface rather than against what a round trip happens to need.

    SETTINGS LIVE ON THE OBJECT, NOT THE MODIFIER, which is the trap that separates this from
    add_cloth twenty lines up. Cloth is configured through `modifier.settings`; a soft body is
    configured through `object.soft_body`, and the modifier merely marks its presence. Writing to
    the modifier and reading back from the object is how a caller convinces themselves a value
    landed when it did not.

    GOAL IS ON BY DEFAULT AND THAT IS WHY NOTHING MOVES. Blender ships `use_goal=True` with
    `goal_default=0.7`, which pins every vertex toward its ORIGINAL position - so a soft body added
    with defaults sags slightly and then stops, and reads as a simulation that does not work. That
    default is NOT overridden here: silently inverting what the engine chose is the failure this
    codebase refuses everywhere else. It is REPORTED instead, in `useGoal` and in a note, so the
    caller learns it from the response rather than from twenty minutes of scrubbing a timeline.

    Verified on 3.6.23, 4.2.17 LTS, 4.4.0 and 5.0.1: every attribute this writes exists on all four,
    so there is nothing to version-gate.
    """
    reject_unknown(params, _SOFTBODY_KEYS, "add_soft_body", {
        "gravityWeight": "spelt gravity on the way in - a weight on the scene's gravity, which the response reports as "
                         "gravityWeight to keep it apart from physics_info's gravity vector",
    })
    obj = get_object(take(params, "object", "name", required=True), want_mesh=True)
    # THE SAME FLOOR AS CLOTH, and for the same reason: a soft body DEFORMS the mesh it is given,
    # and a mesh with a handful of vertices has nothing to deform into. Refused rather than added,
    # so the caller is not left watching a cube that will never wobble.
    if len(obj.data.vertices) < 16:
        raise MifOpError("'%s' has %d vertices, which cannot deform - a soft body moves the mesh it "
                         "is given and a quad has nothing to bend. Subdivide it first. NOTHING was "
                         "added." % (obj.name, len(obj.data.vertices)))
    existing = [m for m in obj.modifiers if m.type == "SOFT_BODY"]
    if existing:
        raise MifOpError("'%s' already has a soft body modifier ('%s'). Blender allows a second one "
                         "and the result is not a stronger simulation, it is two solvers fighting "
                         "over the same vertices. NOTHING was added." % (obj.name, existing[0].name))

    # A GOAL VERTEX GROUP IS WHAT PINS PART OF A MESH. Without one every vertex gets the same goal, so the body is one
    # mass on one spring and sways whole - measured 2026-09-13, five puddings kept their deformed box exactly while
    # swinging up to 1.86 m. With one, a vertex's goal is its weight TIMES goalDefault, mapped into goalMin-goalMax:
    # a weight-1 base holds and a weight-0 top moves. RESOLVED BEFORE THE MODIFIER IS ADDED, so a group that does not
    # exist, one useGoal:false would ignore, or a range that runs backwards refuses with nothing added.
    #
    # THE WEIGHT MULTIPLIES goalDefault, IT DOES NOT REPLACE IT, and the line above used to say otherwise. Measured
    # 2026-09-13 on 4.4.0 while writing a soft-body showcase: three sheets - no group with goalDefault 0.5, a group
    # weighted 0.5 everywhere with goalDefault 1.0, and a group weighted 1.0 with goalDefault 0.5 - baked to bit-
    # identical bounding boxes at every frame. So goalDefault:0 beside a group is a body with NO pin anywhere, which
    # is exactly what the old wording invited ("0 so the unpinned ones are free, weight 1 where I want the pin") and
    # it silently does nothing: a top row weighted 1.0 fell to the floor with the rest.
    goal_group = take(params, "goalVertexGroup", default=None, kind=str)
    goal_min = take_float(params, "goalMin", default=None)
    goal_max = take_float(params, "goalMax", default=None)
    goal_friction = take_float(params, "goalFriction", default=None)
    # useGoal IS PARSED HERE, with the rest, not where it is used. It was read only inside the goal-group branch
    # and again after the modifier was added, so the same malformed value was refused in one call and not in
    # another (audit_blender_shortcircuit_take, 2026-09-13) - and with no group given, refused only after the
    # soft body was already on the object.
    use_goal = take_bool(params, "useGoal", default=True) if "useGoal" in params else None
    goal_group_nonzero = None
    if goal_group is not None:
        vg = obj.vertex_groups.get(goal_group)
        if vg is None:
            raise MifOpError("'%s' has no vertex group '%s' to use as the goal. It has: %s. Make one with "
                             "set_vertex_weights. NOTHING was added."
                             % (obj.name, goal_group, ", ".join(g.name for g in obj.vertex_groups) or "<none>"))
        if use_goal is False:
            raise MifOpError("goalVertexGroup pins vertices THROUGH the goal, and useGoal:false turns the goal off - "
                             "the group would be stored and ignored. NOTHING was added.")
        goal_group_nonzero = sum(1 for v in obj.data.vertices
                                 for g in v.groups if g.group == vg.index and g.weight > 0.0)
    if goal_min is not None and goal_max is not None and goal_min > goal_max:
        raise MifOpError("goalMin %g is above goalMax %g - a goal group's weights map INTO that range, so it runs low "
                         "to high. NOTHING was added." % (goal_min, goal_max))
    # goalDefault:0 BESIDE A GROUP CANNOT PIN ANYTHING, and it is the natural thing to write. The weight multiplies
    # goalDefault (measured, see above), so zero times any weight is zero: the group is stored, reported, counted -
    # and inert. Refused in the same shape as useGoal:false above, rather than added and left to be discovered by
    # watching a pinned mesh fall.
    goal_default = take_float(params, "goalDefault", default=None)
    if goal_group is not None and goal_default == 0.0:
        raise MifOpError("goalDefault:0 with a goal group pins nothing: a vertex's goal is its weight TIMES "
                         "goalDefault, so every weight multiplies to zero and '%s' would be stored and ignored. "
                         "Leave goalDefault out (Blender's 0.7) or set it to the strength the weight-1 vertices "
                         "should hold at. NOTHING was added." % goal_group)

    # THE REST OF THE SETTINGS, READ HERE - before the modifier exists, as the goal settings above already
    # were. They were read after modifiers.new, so {mass: NaN} was refused with the Softbody modifier left on
    # the object (measured on 5.0.1, 2026-09-21), and the retry the refusal invites hits "already has a soft
    # body modifier".
    try:
        m = take_float(params, "mass", default=None)
        fr = take_float(params, "friction", default=None)
        sp = take_float(params, "speed", default=None)
        gs = take_float(params, "goalSpring", default=None)
        pl = take_float(params, "pull", default=None)
        pu = take_float(params, "push", default=None)
        bd = take_float(params, "bend", default=None)
        use_edges = take_bool(params, "useEdges", default=True) if "useEdges" in params else None
        self_col = take_bool(params, "selfCollision", default=False) if "selfCollision" in params else None
        gv = take_float(params, "gravity", default=None)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was added." % str(exc).rstrip("."))

    mod = obj.modifiers.new(name="Softbody", type="SOFT_BODY")
    sb = obj.soft_body

    if m is not None:
        sb.mass = m
    if fr is not None:
        sb.friction = fr
    if sp is not None:
        sb.speed = sp
    if use_goal is not None:
        sb.use_goal = use_goal
    if goal_default is not None:                 # read and validated above, before the modifier existed
        sb.goal_default = goal_default
    if gs is not None:
        sb.goal_spring = gs
    if goal_group is not None:
        sb.vertex_group_goal = goal_group
    if goal_min is not None:
        sb.goal_min = goal_min
    if goal_max is not None:
        sb.goal_max = goal_max
    if goal_friction is not None:
        sb.goal_friction = goal_friction
    if pl is not None:
        sb.pull = pl
    if pu is not None:
        sb.push = pu
    if bd is not None:
        sb.bend = bd
    if use_edges is not None:
        sb.use_edges = use_edges
    if self_col is not None:
        sb.use_self_collision = self_col
    if gv is not None:
        # A WEIGHT ON THE SCENE'S GRAVITY, not an acceleration - the same shape as cloth's, where
        # this exact parameter was accepted and unread until param_reach said so.
        sb.effector_weights.gravity = gv

    out = {"object": obj.name, "modifier": mod.name,
           "mass": round(float(sb.mass), 6), "friction": round(float(sb.friction), 6),
           "speed": round(float(sb.speed), 6), "pull": round(float(sb.pull), 6),
           "push": round(float(sb.push), 6), "bend": round(float(sb.bend), 6),
           "useGoal": bool(sb.use_goal), "goalDefault": round(float(sb.goal_default), 6),
           # goalSpring was written and never reported - the one setting here a caller could not check.
           "goalSpring": round(float(sb.goal_spring), 6),
           "goalVertexGroup": sb.vertex_group_goal or None, "goalMin": round(float(sb.goal_min), 6),
           "goalMax": round(float(sb.goal_max), 6), "goalFriction": round(float(sb.goal_friction), 6),
           "useEdges": bool(sb.use_edges), "selfCollision": bool(sb.use_self_collision),
           "gravityWeight": round(float(sb.effector_weights.gravity), 6),
           "vertices": len(obj.data.vertices),
           "bakeNote": "a soft body is stepped forward like any sim - bake_physics before a late "
                       "frame, or scrub from frame 1."}
    if goal_group is not None:
        out["goalVertexGroupNonZero"] = goal_group_nonzero
    if sb.use_goal and sb.vertex_group_goal:
        out["goalNote"] = ("GOAL IS WEIGHTED BY '%s': each vertex's goal is its weight there TIMES goalDefault (%.2f), "
                           "mapped into goalMin-goalMax (%.2f-%.2f), so a weight-1 vertex holds at %.2f and weight-0 "
                           "ones - and any not in the group - move freely. The weight scales goalDefault, it does not "
                           "replace it (measured on 4.4.0), which is why goalDefault:0 beside a group is refused. "
                           "%d vertex(es) carry a non-zero weight."
                           % (sb.vertex_group_goal, float(sb.goal_default), float(sb.goal_min), float(sb.goal_max),
                              float(sb.goal_min) + float(sb.goal_default) * (float(sb.goal_max) - float(sb.goal_min)),
                              goal_group_nonzero or 0))
    elif sb.use_goal:
        # SAID ON THE RESPONSE THAT CREATED IT, not left for physics_info to reveal later. This is
        # the single most likely reason a caller reports "the soft body does nothing".
        out["goalNote"] = ("GOAL IS ON (Blender's default, goalDefault %.2f), which pins every "
                           "vertex toward its ORIGINAL position - the body will barely move. Pass "
                           "useGoal:false for a free-falling soft body, or lower goalDefault to "
                           "loosen the pin. Not overridden here, because the default is Blender's "
                           "and silently inverting it would be worse."
                           % float(sb.goal_default))
    return out


def op_add_collision(params):
    """Make an object collide with cloth, softbody and particles - NOT with rigid bodies.

    Rigid bodies collide through the rigid body world, not this modifier, and giving a floor a
    Collision modifier and expecting a cube to bounce off it is a common and silent mistake. Say so
    rather than let it look configured.
    """
    reject_unknown(params, _COLL_KEYS, "add_collision")
    obj = get_object(take(params, "object", "name", required=True), want_mesh=True)
    if take_bool(params, "remove", default=False):
        existing = [m for m in obj.modifiers if m.type == "COLLISION"]
        for m in existing:
            obj.modifiers.remove(m)
        return {"object": obj.name, "removed": len(existing),
                "hasCollision": any(m.type == "COLLISION" for m in obj.modifiers)}
    # READ BEFORE THE MODIFIER IS ADDED. They were read after modifiers.new, so {damping: NaN} was refused
    # with a Collision modifier left on the object (measured on 5.0.1, 2026-09-21).
    try:
        d = take_float(params, "damping", default=None)
        f = take_float(params, "friction", default=None)
        th = take_float(params, "thickness", default=None)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was changed." % str(exc).rstrip("."))
    mod = next((m for m in obj.modifiers if m.type == "COLLISION"), None)
    if mod is None:
        mod = obj.modifiers.new(name="Collision", type="COLLISION")
    st = obj.collision
    if d is not None:
        st.damping = d
    if f is not None:
        st.cloth_friction = f
    if th is not None:
        st.thickness_outer = th
    out = {"object": obj.name, "modifier": mod.name,
           "damping": round(float(st.damping), 6),
           "thicknessOuter": round(float(st.thickness_outer), 6),
           "scopeNote": ("this collides with cloth, softbody and particles. RIGID BODIES do NOT "
                         "use it - they collide through the rigid body world, so a floor for a "
                         "falling crate needs add_rigid_body {type:PASSIVE} instead.")}
    # A RIGID BODY THAT MUST PUSH CLOTH NEEDS BOTH, and its shape decides (rule d, measured 2026-10-05 on 3.6.23, 4.4.0
    # and 5.2.1): a ball dropped through a pinned cloth sheet moved the sheet only as a rigid body with this modifier AND
    # collision shape SPHERE; CONVEX_HULL or MESH with this modifier left the sheet as if no ball were there, and the
    # cloth never held the ball up in any variant.
    rbody = getattr(obj, "rigid_body", None)
    if rbody is not None:
        out["rigidBodyNote"] = (
            "'%s' is a rigid body (%s), and a rigid body that must push cloth needs this modifier AND collision shape "
            "SPHERE - measured, CONVEX_HULL and MESH with a Collision modifier moved a cloth sheet not at all. The "
            "coupling is one-way: cloth never holds a rigid body up." % (obj.name, rbody.collision_shape))
    return out


def op_add_dynamic_paint(params):
    """Make an object a dynamic paint CANVAS or BRUSH - wetmaps, footprints in snow, weathering.

    THE SEVENTH AND LAST of the modifier types this addon could add and not configure. Like FLUID it
    sits in `_SIM_MODIFIERS`, so bake_physics and physics_info have always handled one; like FLUID
    its modifier carries exactly ONE writable property, `ui_type`, and everything real lives on
    canvas_settings or brush_settings.

    ui_type ALONE DOES NOTHING, which is the trap here. Setting `modifier.ui_type = "CANVAS"` leaves
    canvas_settings as None - measured, on 3.6.23, 4.4.0 and 5.0.1 alike - so a caller who set it
    and stopped would have a modifier that looks configured in the outliner and has no settings at
    all. The settings are created by `dpaint.type_toggle`, an operator, which this runs under a
    temp_override the way create_force_field runs forcefield_toggle.

    A CANVAS ARRIVES WITH ONE SURFACE, named "Surface", created by that operator. This configures
    that first surface rather than adding more: a canvas with several surfaces is a real thing and a
    second op's job, and quietly adding one here would make the count depend on how many times the
    caller had called this.

    surface_type IS A DYNAMIC ENUM. Its members depend on the surface FORMAT - on a fresh VERTEX
    surface this Blender offers only PAINT - so it is validated against `enum_items` at the moment of
    writing rather than against a list here. A refusal therefore names what is available NOW, which
    is the only correct answer to "why can I not set WAVE".

    Nothing is baked. bake_physics is the separate, expensive step, as for every other sim here.
    """
    reject_unknown(params, _DP_KEYS, "add_dynamic_paint", {
        "bake": "nothing is baked here - dynamic paint is stepped forward like any sim. Call "
                "bake_physics when the setup is right.",
        "surfaces": "a canvas arrives with exactly ONE surface and this configures that one. "
                    "Several surfaces on a canvas is a real setup and is a separate op's job; "
                    "adding one here would make the count depend on how often you called this.",
        "surface": "see 'surfaces'. The response reports the surface's real name.",
        "uiType": "setting ui_type alone does NOTHING - it leaves canvas_settings as None, so the "
                  "modifier looks configured and has no settings at all. Pass type instead, which "
                  "runs the operator that actually creates them.",
    })
    kind = str(take(params, "type", required=True, kind=str)).upper()
    if kind not in _DP_BY_TYPE:
        raise MifOpError("dynamic paint type must be CANVAS or BRUSH, got '%s'. NOTHING was "
                         "changed." % kind)
    mine = _DP_BY_TYPE[kind]
    for key in params:
        if key in _DP_COMMON or key in mine:
            continue
        owner = next((k for k, v in _DP_BY_TYPE.items() if key in v), None)
        raise MifOpError("'%s' belongs to a dynamic paint %s, not a %s - a canvas receives paint "
                         "and a brush applies it, and dropping it silently would leave you "
                         "believing it was set. NOTHING was changed." % (key, owner or "?", kind))

    obj = get_object(take(params, "object", "name", required=True), want_mesh=True)
    if any(m.type == "DYNAMIC_PAINT" for m in obj.modifiers):
        raise MifOpError("'%s' already has a dynamic paint modifier. NOTHING was changed."
                         % obj.name)

    stype = take(params, "surfaceType", default=None, kind=str)
    sformat = take(params, "surfaceFormat", default=None, kind=str)
    # THE SURFACE'S FRAMES AND DISSOLVE SPEED ARE INTS, read with take_int - take_float then int() floored a fraction
    # under ok:true (the 2026-09-21 pack-rule review).
    fstart = take_int(params, "frameStart", default=None)
    fend = take_int(params, "frameEnd", default=None)
    dissolve = take_bool(params, "useDissolve", default=None) if "useDissolve" in params else None
    dspeed = take_int(params, "dissolveSpeed", default=None)
    pcolor = take(params, "paintColor", default=None)
    palpha = take_float(params, "paintAlpha", default=None)
    pwet = take_float(params, "paintWetness", default=None)
    psource = take(params, "paintSource", default=None, kind=str)
    pdist = take_float(params, "paintDistance", default=None)
    pabs = take_bool(params, "useAbsoluteAlpha", default=None) if "useAbsoluteAlpha" in params else None
    if pcolor is not None:
        pcolor = finite_floats(pcolor, "paintColor")
        # EXACTLY THREE: a brush's paint_color is RGB and its alpha is paint_alpha. A fourth number was accepted and
        # dropped without a word (found 2026-10-25).
        if len(pcolor) != 3:
            raise MifOpError("paintColor is [r, g, b] - got %d number(s). A brush's alpha is its own setting: pass it "
                             "as paintAlpha. NOTHING was changed." % len(pcolor))

    mod = obj.modifiers.new(name=str(take(params, "name", default="Dynamic Paint", kind=str)),
                            type="DYNAMIC_PAINT")
    # THE TYPE GOES ON BEFORE THE TOGGLE, OR A BRUSH PAINTS NOTHING. ui_type is the modifier's own type field, and the
    # toggle is what rebuilds the scene's relations. MEASURED 2026-09-16, identical on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and
    # 5.2.1, a DISPLACE canvas and a sphere brush dragged across it over frames 1-30: ui_type set BEFORE type_toggle -
    # 0.40 BU of relief; set AFTER it - ui_type reads BRUSH, brush_settings exist, 0.0 relief; never set - 0.0. This op
    # set it after (found chasing the second seat's flat DISPLACE canvas, mailbox id 52).
    # NOT EVERY BRUSH IT MADE WAS INERT, which this comment first claimed. The second seat painted 0.50 relief with the
    # old op (mailbox id 107). The add-on at 5053b042, driven through this op on all five builds: {type: BRUSH} alone
    # painted 0.0; the same call with any brush setting painted 0.40-0.50 - even paintSource VOLUME, which is already
    # the default, so it is the WRITE to brush_settings that revived it, not the value. The inert brushes were the bare
    # ones. With the type set first, the bare call and all three others paint (scratchpad measure_dpaint_old_new.py).
    try:
        mod.ui_type = kind
    except Exception:                               # noqa: BLE001 - measured to take on all five builds
        pass
    try:
        with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
            bpy.ops.dpaint.type_toggle(type=kind)
    except Exception as exc:                       # noqa: BLE001
        obj.modifiers.remove(mod)
        raise MifOpError("could not make '%s' a dynamic paint %s (%s). NOTHING was left behind."
                         % (obj.name, kind, exc))

    settings = mod.canvas_settings if kind == "CANVAS" else mod.brush_settings
    if settings is None:
        obj.modifiers.remove(mod)
        raise MifOpError("dpaint.type_toggle reported no error and left %s_settings None, so there "
                         "is nothing to configure. NOTHING was left behind." % kind.lower())

    def _enum(holder, attr, value, label):
        # ASSIGNED, AND BLENDER ANSWERS - not checked against bl_rna's enum_items first. surface_type
        # is a DYNAMIC enum whose members depend on the surface's format, and enum_items read off
        # bl_rna is evaluated with NO instance: it listed PAINT alone, so DISPLACE, WEIGHT and WAVE -
        # which rna_dynamicpaint.cc (v4.4.0, rna_DynamicPaint_surface_type_itemf) offers on a VERTEX
        # surface, WAVE on every format - were refused on every call. Found 2026-09-11 by the showcase
        # probe asking for a WAVE canvas. The TypeError Blender raises for a bad member lists the
        # members THIS instance offers, so it is the right refusal for static and dynamic enums alike.
        got = str(value).upper()
        try:
            setattr(holder, attr, got)
        except (TypeError, ValueError) as exc:
            obj.modifiers.remove(mod)
            raise MifOpError("%s %r was refused by Blender on this surface (%s). The choices depend on "
                             "the surface format, and that list is Blender's own for THIS surface. "
                             "NOTHING was created." % (label, value, exc))
        if getattr(holder, attr) != got:
            obj.modifiers.remove(mod)
            raise MifOpError("%s %r was accepted and read back as %r. NOTHING was created."
                             % (label, value, getattr(holder, attr)))

    # paintType IS MEASURED FROM THE SETTINGS THAT EXIST, not read off ui_type. type_toggle(type='BRUSH') on its own
    # creates brush_settings and leaves ui_type at CANVAS, so every brush this made once reported "paintType":
    # "CANVAS" - found 2026-09-11 by the showcase probe. ui_type is now set before the toggle (above), and it is not
    # only the Properties tab: it decides whether a canvas sees the brush at all.
    made =[t for t, s in (("CANVAS", mod.canvas_settings), ("BRUSH", mod.brush_settings)) if s is not None]
    out = {"object": obj.name, "modifier": mod.name, "paintType": made[0] if len(made) == 1 else made}

    if kind == "CANVAS":
        surf = settings.canvas_surfaces.active
        if surf is None:
            obj.modifiers.remove(mod)
            raise MifOpError("the canvas was created with no active surface, which should not "
                             "happen on any supported Blender. NOTHING was left behind.")
        # FORMAT BEFORE TYPE, because surface_type's valid members depend on the format - setting
        # the type first and the format second can leave a type the new format does not offer.
        if sformat is not None:
            _enum(surf, "surface_format", sformat, "surfaceFormat")
        if stype is not None:
            _enum(surf, "surface_type", stype, "surfaceType")
        if fstart is not None:
            surf.frame_start = int(fstart)
        if fend is not None:
            surf.frame_end = int(fend)
        if dissolve is not None:
            surf.use_dissolve = dissolve
        if dspeed is not None:
            surf.dissolve_speed = int(dspeed)
        out.update({"surface": surf.name, "surfaceType": surf.surface_type,
                    "surfaceFormat": surf.surface_format,
                    "frameStart": int(surf.frame_start), "frameEnd": int(surf.frame_end),
                    "useDissolve": bool(surf.use_dissolve),
                    "dissolveSpeed": int(surf.dissolve_speed),
                    "surfaceCount": len(settings.canvas_surfaces),
                    "surfaceNote": "a canvas arrives with ONE surface and this configures that one. "
                                   "Several surfaces on a canvas is a real setup and is not built "
                                   "here, so the count above does not depend on how often you "
                                   "called this."})
    else:
        if psource is not None:
            _enum(settings, "paint_source", psource, "paintSource")
        if pcolor is not None:
            settings.paint_color = pcolor[:3]
        if palpha is not None:
            settings.paint_alpha = palpha
        if pwet is not None:
            settings.paint_wetness = pwet
        if pdist is not None:
            settings.paint_distance = pdist
        if pabs is not None:
            settings.use_absolute_alpha = pabs
        out.update({"paintSource": settings.paint_source,
                    "paintColor": [round(float(v), 6) for v in settings.paint_color],
                    "paintAlpha": round(float(settings.paint_alpha), 6),
                    "paintWetness": round(float(settings.paint_wetness), 6),
                    "paintDistance": round(float(settings.paint_distance), 6),
                    "useAbsoluteAlpha": bool(settings.use_absolute_alpha)})

    out["bakeNote"] = ("nothing is baked - dynamic paint is stepped forward like any sim, so call "
                       "bake_physics when the setup is right.")
    return out


def _material_volume_linked(obj):
    """True when a material in one of obj's slots has something on its rendering output's Volume socket."""
    for slot in getattr(obj, "material_slots", ()):
        mat = slot.material
        if mat is None or not mat.use_nodes or mat.node_tree is None:
            continue
        outs = [n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL"]
        active = [n for n in outs if getattr(n, "is_active_output", False)] or outs
        sock = active[0].inputs.get("Volume") if active else None
        if sock is not None and sock.is_linked:
            return True
    return False


def op_add_fluid(params):
    """Make an object a fluid DOMAIN, FLOW or EFFECTOR - smoke, fire and liquid.

    FLUID HAS BEEN IN `_SIM_MODIFIERS` SINCE THAT LIST WAS WRITTEN, so bake_physics and
    physics_info have always handled one, and nothing could create one. The same consumer-with-no-
    creator shape as soft body, and found the same way - by measuring which modifier types
    add_modifier can ADD but not TUNE. FLUID's modifier carries exactly ONE writable property,
    `fluid_type`; every setting that matters lives on domain_settings, flow_settings or
    effector_settings, none of which add_modifier's curated writer can reach.

    THIS DOES NOT BAKE, and that is deliberate rather than an omission. A fluid bake is minutes to
    hours and blocks Blender's main thread for all of it - the addon's own render docstring says a
    long main-thread job reads as a hung bridge. This builds the setup and returns; bake_physics is
    the separate, expensive step, as it is for cloth and rigid bodies.

    TWO DEFAULTS THAT MAKE A CORRECT SETUP LOOK BROKEN, reported rather than overridden:

      resolution_max is 32. That is Blender's default and it is very low - smoke at 32 is visibly
      blocky. It is not changed here, because silently choosing somebody's simulation resolution is
      a performance decision this op has no business making, but the response says what it is.

      flow_behavior defaults to GEOMETRY, NOT INFLOW. GEOMETRY turns the mesh into fluid once;
      INFLOW is the continuous emitter people mean by "add smoke". So a FLOW object built with
      defaults produces a single puff rather than a stream, which reads as a simulation that barely
      works. Reported in a note on the response that creates it.

    A SETTING SENT TO THE WRONG ROLE IS REFUSED, naming the role that owns it. The three fluid_types
    are different objects with different fields, and quietly dropping resolutionMax because the
    caller said FLOW would leave them believing they had set a resolution.

    Verified on Blender 3.6.23, 4.2.17 LTS, 4.4.0 and 5.0.1.
    """
    # WHY NOT THAT KEY, for the guesses this op's docstring already answers - the reason lived in
    # a docstring no caller holding a refusal can see. See reject_unknown for the argument.
    reject_unknown(params, _FLUID_KEYS, "add_fluid", {
        "bake": "nothing is baked here on purpose. A fluid bake is minutes to hours and blocks "
                "Blender's main thread for all of it, which reads as a hung bridge - call "
                "bake_physics deliberately once the setup is right.",
        "resolution": "spelt resolutionMax, and it belongs to the DOMAIN only. It defaults to 32, "
                      "which is Blender's default and is low enough that smoke looks blocky.",
        "smoke": "there is no smoke type. Smoke, fire and liquid are all the same simulation - "
                 "pick the role with type (DOMAIN, FLOW or EFFECTOR) and the substance with "
                 "flowType on the FLOW.",
        "fire": "see 'smoke' - flowType on the FLOW object chooses it.",
    })
    kind = str(take(params, "type", required=True, kind=str)).upper()
    if kind not in _FLUID_BY_TYPE:
        raise MifOpError("fluid type must be one of %s, got '%s'. NOTHING was changed."
                         % (", ".join(sorted(_FLUID_BY_TYPE)), kind))

    # WRONG-ROLE KEYS FIRST, before the object is even resolved: this is a caller mistake about what
    # they are building, and it should be answered before anything else can fail for another reason.
    mine = _FLUID_BY_TYPE[kind]
    for key in params:
        if key in _FLUID_COMMON or key in mine:
            continue
        owner = next((k for k, v in _FLUID_BY_TYPE.items() if key in v), None)
        raise MifOpError("'%s' belongs to a fluid %s, not a %s - the three roles carry different "
                         "settings and dropping it silently would leave you believing it was set. "
                         "NOTHING was changed." % (key, owner or "(unknown role)", kind))

    obj = get_object(take(params, "object", "name", required=True), want_mesh=True)
    if any(m.type == "FLUID" for m in obj.modifiers):
        raise MifOpError("'%s' already has a fluid modifier. One object is one role - a domain that "
                         "is also a flow is not a thing Mantaflow solves. NOTHING was changed."
                         % obj.name)

    # EVERY VALUE PARSED BEFORE THE MODIFIER EXISTS, the ordering create_force_field had to learn.
    # resolutionMax, the cache frames and subframes ARE INTS, read with take_int: take_float then int() floored
    # {resolutionMax: 32.5} to 32 under ok:true (the 2026-09-21 pack-rule review).
    res = take_int(params, "resolutionMax", default=None)
    cstart = take_int(params, "cacheStart", default=None)
    cend = take_int(params, "cacheEnd", default=None)
    adaptive = take_bool(params, "useAdaptiveDomain", default=None) if "useAdaptiveDomain" in params else None
    use_mesh = take_bool(params, "useMesh", default=None) if "useMesh" in params else None
    sdist = take_float(params, "surfaceDistance", default=None)
    subf = take_int(params, "subframes", default=None)
    inflow = take_bool(params, "useInflow", default=None) if "useInflow" in params else None
    use_eff = take_bool(params, "useEffector", default=None) if "useEffector" in params else None
    dtype = take(params, "domainType", default=None, kind=str)
    # A SURFACE IS A LIQUID'S. On a gas domain use_mesh does nothing, so useMesh:true answered ok and made no mesh.
    if use_mesh and str(dtype or "GAS").upper() != "LIQUID":
        raise MifOpError("useMesh:true needs domainType LIQUID: a %s domain has no surface to mesh, and it would answer ok "
                         "with no mesh. NOTHING was changed." % str(dtype or "GAS").upper())
    ftype = take(params, "flowType", default=None, kind=str)
    fbehav = take(params, "flowBehavior", default=None, kind=str)
    etype = take(params, "effectorType", default=None, kind=str)
    # THE DOMAIN'S TIME SCALE, CACHE TYPE, FOLDER AND RESUMABLE FLAG (2026-10-04), so a staged bake can be set up in
    # the call that makes the domain. time_scale is a look choice tied to scene scale and 24 fps (1.0 for a faucet at
    # real scale, 0.5 for a 3 m smoke box or a liquid jet, 0.3 for a liquid guided by smoke) - a named parameter, never a
    # changed default. cache_resumable must be on BEFORE the data bake: writing it afterwards frees every stage
    # (measured on 3.6.23, 4.4.0 and 5.2.1), and without it Blender's panel withholds the noise, mesh and particle bakes.
    tscale = take_float(params, "timeScale", default=None)
    if tscale is not None and tscale <= 0.0:
        raise MifOpError("timeScale must be above 0, got %g. NOTHING was changed." % tscale)
    ctype = take(params, "cacheType", default=None, kind=str)
    if ctype is not None:
        valid = [i.identifier for i in bpy.types.FluidDomainSettings.bl_rna.properties["cache_type"].enum_items]
        if str(ctype).upper() not in valid:
            raise MifOpError("cacheType must be one of %s, got %r. NOTHING was changed." % (", ".join(valid), ctype))
        ctype = str(ctype).upper()
    cdir = take(params, "cacheDirectory", default=None, kind=str)
    if cdir is not None and not str(cdir).strip():
        raise MifOpError("cacheDirectory is empty - Blender would put the cache in its session temp folder, which is "
                         "deleted when it closes. NOTHING was changed.")
    cres = take_bool(params, "cacheResumable", default=False) if "cacheResumable" in params else None

    mod = obj.modifiers.new(name=str(take(params, "name", default="Fluid", kind=str)), type="FLUID")
    mod.fluid_type = kind

    def _enum(settings, attr, value, label):
        allowed = [i.identifier for i in settings.bl_rna.properties[attr].enum_items]
        got = str(value).upper()
        if got not in allowed:
            obj.modifiers.remove(mod)
            raise MifOpError("%s must be one of %s, got %r. NOTHING was created."
                             % (label, ", ".join(allowed), value))
        setattr(settings, attr, got)

    out = {"object": obj.name, "modifier": mod.name, "fluidType": mod.fluid_type}

    if kind == "DOMAIN":
        st = mod.domain_settings
        if dtype is not None:
            _enum(st, "domain_type", dtype, "domainType")
        if res is not None:
            st.resolution_max = int(res)
        if cstart is not None:
            st.cache_frame_start = int(cstart)
        if cend is not None:
            st.cache_frame_end = int(cend)
        if ctype is not None:
            st.cache_type = ctype
        if cres is not None:
            st.cache_resumable = cres
        if cdir is not None:
            st.cache_directory = str(cdir)
        if tscale is not None:
            st.time_scale = tscale
        tag_fluid_domain(st)
        if adaptive is not None:
            st.use_adaptive_domain = adaptive
        # A LIQUID WITH NO SURFACE RENDERS NOTHING. MEASURED 2026-09-16 (scratchpad measure_liquid_defaults.py,
        # measure_fluid_usemesh.py): a new LIQUID domain's use_mesh defaults to False on 3.6.23 and True on 4.2.17 and
        # 5.2.1; the second seat's splash on 3.6 simulated (data frames 1-40 on disk) and evaluated to the bare 8-vertex
        # cube, and with use_mesh on it wrote mesh frames 1-40 and evaluated to 19,932 vertices at frame 30 (bake 0.76 s ->
        # 1.98 s). So a LIQUID domain gets a surface on every build unless useMesh:false says otherwise, and the reply
        # says when this op turned it on.
        mesh_turned_on = False
        if st.domain_type == "LIQUID" and hasattr(st, "use_mesh"):
            if use_mesh is not None:
                st.use_mesh = use_mesh
            elif not st.use_mesh:
                st.use_mesh = True
                mesh_turned_on = True
        out.update({"domainType": st.domain_type, "resolutionMax": int(st.resolution_max),
                    "cacheStart": int(st.cache_frame_start), "cacheEnd": int(st.cache_frame_end),
                    "useAdaptiveDomain": bool(st.use_adaptive_domain),
                    "cacheType": st.cache_type,
                    "cacheResumable": bool(st.cache_resumable),
                    "cacheDirectory": bpy.path.abspath(st.cache_directory) if st.cache_directory else None,
                    "timeScale": round(float(st.time_scale), 6),
                    "useMesh": bool(st.use_mesh) if st.domain_type == "LIQUID" and hasattr(st, "use_mesh") else None,
                    "useMeshNote": ("this Blender (%s) starts a LIQUID domain with mesh generation OFF, which simulates "
                                    "the liquid and renders no surface - it has been turned ON, as 4.2 and later start "
                                    "it. Pass useMesh:false to keep it off." % bpy.app.version_string)
                    if mesh_turned_on else None})
        # A GAS DOMAIN NEEDS A VOLUME SHADER TO APPEAR. MEASURED 2026-09-17 on all five builds (docs/22 B80): with no
        # material it renders as a SOLID BOX in EEVEE and as NOTHING in Cycles, and a Principled Volume with no surface
        # shader renders the simulated smoke in both - which is what set_material_volume writes.
        if st.domain_type == "GAS" and not _material_volume_linked(obj):
            # allowResize ONLY WHEN IT IS NEEDED: set_material_slots refuses a slot-count change without it, and a
            # new mesh has no slots - so the call this note recommends would itself have been refused.
            out["volumeNote"] = (
                "a GAS domain renders as a SOLID BOX in EEVEE and as NOTHING in Cycles until one of its materials has "
                "a volume shader - measured on every build, and '%s' has none. For smoke or fire: create_material, "
                "set_material_volume{material, surface:'disconnect'}, then set_material_slots{object, "
                "slots:[material]%s}. Its default densityAttribute 'density' reads this simulation."
                % (obj.name, "" if len(obj.material_slots) == 1 else ", allowResize:true"))
        if int(st.resolution_max) <= 32:
            out["resolutionNote"] = (
                "resolutionMax is %d, which is Blender's default and is LOW - smoke at this "
                "resolution is visibly blocky. Not changed for you, because picking somebody's "
                "simulation resolution is a performance decision this op should not make."
                % int(st.resolution_max))
    elif kind == "FLOW":
        st = mod.flow_settings
        if ftype is not None:
            _enum(st, "flow_type", ftype, "flowType")
        if fbehav is not None:
            _enum(st, "flow_behavior", fbehav, "flowBehavior")
        if inflow is not None:
            st.use_inflow = inflow
        if sdist is not None:
            st.surface_distance = sdist
        if subf is not None:
            st.subframes = int(subf)
        out.update({"flowType": st.flow_type, "flowBehavior": st.flow_behavior,
                    "useInflow": bool(st.use_inflow),
                    "surfaceDistance": round(float(st.surface_distance), 6),
                    "subframes": int(st.subframes)})
        if st.flow_behavior == "GEOMETRY":
            out["behaviorNote"] = (
                "flowBehavior is GEOMETRY, which is Blender's default and turns the MESH into fluid "
                "ONCE - a single puff. The continuous emitter people mean by 'add smoke' is INFLOW. "
                "Not overridden, because GEOMETRY is a real and useful behaviour and silently "
                "choosing the other would be a guess about what you are building.")
    else:
        st = mod.effector_settings
        if etype is not None:
            _enum(st, "effector_type", etype, "effectorType")
        if use_eff is not None:
            st.use_effector = use_eff
        if sdist is not None:
            st.surface_distance = sdist
        out.update({"effectorType": st.effector_type, "useEffector": bool(st.use_effector),
                    "surfaceDistance": round(float(st.surface_distance), 6)})

    out["bakeNote"] = ("NOTHING IS BAKED. A fluid bake takes minutes to hours and blocks Blender's "
                       "main thread for all of it - call bake_physics deliberately when the setup "
                       "is right, and expect it to be slow.")
    return out


def _joint_row(obj):
    """One joint read back off its constraint - the reply row for a joint made or edited."""
    rbc = obj.rigid_body_constraint
    return {"object": obj.name, "constraintType": rbc.type,
            "object1": rbc.object1.name if rbc.object1 else None, "object2": rbc.object2.name if rbc.object2 else None,
            # THE EMPTY'S OWN LOCATION: matrix_world of an empty made in this call is not evaluated yet and read 0, 0, 0.
            "location": [round(float(v), 6) for v in obj.location],
            "enabled": bool(rbc.enabled), "disableCollisions": bool(rbc.disable_collisions),
            "useBreaking": bool(rbc.use_breaking), "breakingThreshold": round(float(rbc.breaking_threshold), 6),
            "overrideSolverIterations": bool(rbc.use_override_solver_iterations),
            "solverIterations": int(rbc.solver_iterations)}


def _joint_settings(params, tail):
    """The per-joint settings a call writes, parsed and checked -> dict of present ones only."""
    out = {}
    try:
        if "disableCollisions" in params:
            out["disable_collisions"] = take_bool(params, "disableCollisions", default=False)
        if "enabled" in params:
            out["enabled"] = take_bool(params, "enabled", default=True)
        if "useBreaking" in params:
            out["use_breaking"] = take_bool(params, "useBreaking", default=False)
        brk = take_float(params, "breakingThreshold", default=None)
        its = take_int(params, "solverIterations", default=None)
    except MifOpError as exc:
        raise MifOpError("%s %s" % (str(exc).rstrip("."), tail))
    if brk is not None:
        if brk < 0.0:
            raise MifOpError("breakingThreshold must be 0 or more, got %g. %s" % (brk, tail))
        out["breaking_threshold"] = brk
        # THE FLAG AND THE NUMBER TOGETHER: breaking_threshold is ignored unless use_breaking is on, so the number
        # alone reads back as asked and never breaks anything. An explicit useBreaking beats it.
        out.setdefault("use_breaking", True)
    if its is not None:
        if its < 0:
            raise MifOpError("solverIterations must be 1 or more (0 turns the override off), got %d. %s" % (its, tail))
        out["use_override_solver_iterations"] = its > 0
        if its > 0:
            out["solver_iterations"] = its
    return out


def _joint_pairs_existing(rbw):
    """{frozenset(name1, name2)} of every pair a joint in this world already holds."""
    held = set()
    if rbw is not None and rbw.constraints is not None:
        for o in rbw.constraints.objects:
            c = o.rigid_body_constraint
            if c is not None and c.object1 is not None and c.object2 is not None:
                held.add(frozenset((c.object1.name, c.object2.name)))
    return held


def _update_joints(params, sc, rbw):
    """add_rigid_body_constraint {update: true}: write settings on joints that exist, named or by collection."""
    tail = "NOTHING was changed."
    allowed = {"update", "names", "collection", "breakingThreshold", "useBreaking", "enabled", "solverIterations",
               "disableCollisions"}
    extra = sorted(k for k in params if k not in allowed)
    if extra:
        raise MifOpError("update edits joints that exist and writes breakingThreshold, useBreaking, enabled, "
                         "solverIterations and disableCollisions; %s %s for making joints. %s"
                         % (", ".join(extra), "is" if len(extra) == 1 else "are", tail))
    names = params.get("names")
    cname = take(params, "collection", default=None, kind=str)
    if (names is None) == (cname is None):
        raise MifOpError("update takes names [joint names] or collection, exactly one. %s" % tail)
    if names is not None:
        if not isinstance(names, (list, tuple)) or not names or not all(isinstance(n, str) for n in names):
            raise MifOpError("names must be a non-empty list of joint object names, got %r. %s" % (names, tail))
        missing = [n for n in names if bpy.data.objects.get(n) is None]
        if missing:
            raise MifOpError("no object named %s. %s" % (", ".join("'%s'" % n for n in missing[:12]), tail))
        joints = [bpy.data.objects[n] for n in dict.fromkeys(names)]
    else:
        coll = bpy.data.collections.get(cname)
        if coll is None:
            raise MifOpError("no collection named '%s'. %s" % (cname, tail))
        joints = sorted(coll.all_objects, key=lambda o: o.name)
    notj = [o.name for o in joints if getattr(o, "rigid_body_constraint", None) is None]
    if names is not None and notj:
        raise MifOpError("%s %s not a rigid body joint. %s" % (", ".join(notj[:12]), "is" if len(notj) == 1 else "are",
                                                              tail))
    joints = [o for o in joints if getattr(o, "rigid_body_constraint", None) is not None]
    if not joints:
        raise MifOpError("collection '%s' holds no rigid body joint. %s" % (cname, tail))
    settings = _joint_settings(params, tail)
    if not settings:
        raise MifOpError("update was given nothing to write - pass breakingThreshold, useBreaking, enabled, "
                         "solverIterations or disableCollisions. %s" % tail)
    before = {o.name: _joint_row(o) for o in joints}
    for o in joints:
        for attr, val in settings.items():
            setattr(o.rigid_body_constraint, attr, val)
    rows = []
    for o in joints:
        row = _joint_row(o)
        row["before"] = {k: before[o.name][k] for k in ("enabled", "disableCollisions", "useBreaking",
                                                        "breakingThreshold", "overrideSolverIterations",
                                                        "solverIterations")}
        rows.append(row)
    return {"updated": True, "joints": rows, "count": len(rows),
            "skipped": [{"object": n, "why": "not a rigid body joint"} for n in notj] or None,
            "worldJointCount": len(rbw.constraints.objects) if rbw.constraints is not None else 0}


def _joint_collection(sc, cname, tail):
    """(collection, created) the joints' empties go into; a missing one is made under the scene's collection."""
    if cname is None:
        return sc.collection, False
    coll = bpy.data.collections.get(cname)
    if coll is None:
        return None, True
    if coll != sc.collection and coll not in sc.collection.children_recursive:
        raise MifOpError("collection '%s' is not in this scene, so joints put there would not be evaluated. Link it "
                         "into the scene or name another. %s" % (cname, tail))
    return coll, False


def op_add_rigid_body_constraint(params):
    """Join rigid bodies - a hinge, a slider, a motor, or a whole fractured slab - and edit joints that exist.

    WHY IT WAS A GAP. `add_rigid_body` has existed for a while and `set_physics_world` tunes the
    solver, so bodies could fall, bounce and be baked - and nothing could JOIN two of them. A door
    needs a hinge, a chain needs points, a ragdoll needs limits, a piston needs a slider. Found
    2026-09-07 by surveying the addon against Blender's feature surface. Note that `add_constraint`
    is a different system entirely: those are OBJECT constraints (Track To, Copy Location) evaluated
    during the depsgraph, not physics joints solved by the rigid body world.

    A CONSTRAINT IS COLLECTION MEMBERSHIP, not a modifier and not a datablock you can allocate.
    `object.rigid_body_constraint` is None until the object is linked into
    `scene.rigidbody_world.constraints`, at which point Blender creates the settings itself. That is
    why this needs no operator at all - unlike create_force_field, which does. Verified on 3.6.23,
    4.4.0 and 5.0.1: linking creates the pointer on all three and all three offer the same 8 types.

    THE JOINT IS AN EMPTY AND ITS LOCATION IS THE PIVOT. A hinge swings about the constraint
    object's origin, so `location` is not decoration - put it on the hinge line or the door swings
    about the wrong axis. Nothing here can guess where that is.

    ITS ROTATION IS THE AXIS. A HINGE turns about the joint's local Z, and SLIDER, PISTON and MOTOR work along its
    local X (measured, see _RBC_AXIS) - so an unrotated hinge has a vertical axis and a plank on it swings like a door
    and never falls. `rotation` turns the joint; axisLocal / axisWorld say where the axis ended up.

    MANY AT ONCE (2026-10-04): objects [names] joins them the way Blender's Connect does (bl_operators/rigidbody.py,
    the same on 3.6, 4.4 and 5.2) - pattern SELECTED_TO_ACTIVE joins each to `active`, CHAIN_DISTANCE walks
    nearest-first from `active` with no maximum distance; pivot CENTER (between the two), ACTIVE or SELECTED. Written
    natively, with world positions, so the empties can go into `collection` instead of the scene's. A pair a joint
    already holds is skipped and named. update: true writes settings on joints that exist (names or collection).

    params:
      name (str)                  default "Joint" (a bulk call's joints are this name, suffixed by Blender)
      type (str)                  FIXED | POINT | HINGE | SLIDER | PISTON | GENERIC |
                                  GENERIC_SPRING | MOTOR - refused if this Blender lacks it
      object1, object2 (str)      one joint; both must already be rigid bodies
      objects [names], active, pattern, pivot, collection    many joints (see above)
      location (3-list)           the pivot (one joint only)
      rotation (3-list)           the joint's Euler XYZ rotation in RADIANS, as transform_object takes it -
                                  [1.5708, 0, 0] lays a hinge's axis along world -Y
      disableCollisions (bool)    let the two joined bodies interpenetrate, default Blender's
      breakingThreshold (float)   sets use_breaking AND the number together; useBreaking (bool) sets the flag alone
      solverIterations (int)      the joint's own solver iterations (turns the override on; 0 turns it off)
      enabled (bool)
      update (bool), names [..]   edit existing joints instead of making them
    """
    reject_unknown(params, _RBC_KEYS, "add_rigid_body_constraint", {
        "selected": "spelt objects - the bodies to join - with active for the one they join to.",
        "connectionPattern": "spelt pattern: SELECTED_TO_ACTIVE or CHAIN_DISTANCE.",
        "pivotType": "spelt pivot: CENTER, ACTIVE or SELECTED.",
    })
    sc = bpy.context.scene
    rbw = getattr(sc, "rigidbody_world", None)
    if take_bool(params, "update", default=False):
        if rbw is None:
            raise MifOpError("this scene has no rigid body world, so it holds no joints to update. NOTHING was changed.")
        return _update_joints(params, sc, rbw)
    if "names" in params:
        raise MifOpError("names picks joints to edit and needs update: true; to make joints pass object1/object2 or "
                         "objects. NOTHING was created.")
    bulk = params.get("objects") is not None
    if bulk and (params.get("object1") is not None or params.get("object2") is not None):
        raise MifOpError("pass object1/object2 for one joint OR objects for many, not both. NOTHING was created.")
    if not bulk:
        for key in ("active", "pattern", "pivot"):
            if key in params:
                raise MifOpError("%s belongs to a many-joint call with objects [names]. NOTHING was created." % key)
    if bulk and "location" in params:
        raise MifOpError("location places one joint; a many-joint call places each by pivot (CENTER, ACTIVE or "
                         "SELECTED). NOTHING was created.")

    # EVERY PARSE AND EVERY LOOKUP BEFORE THE FIRST WRITE. create_force_field shipped with its
    # floats read after the object was made, and blender_version_matrix reported 48 rows of
    # "refusals that left something behind" on the first run. Same author, same afternoon.
    wanted_type = str(take(params, "type", default="FIXED", kind=str)).upper()
    offered = [i.identifier for i in
               bpy.types.RigidBodyConstraint.bl_rna.properties["type"].enum_items]
    if wanted_type not in offered:
        raise MifOpError("constraint type '%s' is not one this Blender (%s) offers. Valid: %s. "
                         "NOTHING was created."
                         % (wanted_type, bpy.app.version_string, ", ".join(offered)))

    pairs, skipped_pairs = [], []
    pivot = "CENTER"
    pattern = None
    if bulk:
        names = params.get("objects")
        if not isinstance(names, (list, tuple)) or len(names) < 2 or not all(isinstance(n, str) for n in names):
            raise MifOpError("objects must list at least two rigid body names, got %r. NOTHING was created." % (names,))
        missing = [n for n in names if bpy.data.objects.get(n) is None]
        if missing:
            raise MifOpError("no object named %s. NOTHING was created." % ", ".join("'%s'" % n for n in missing[:12]))
        objs = [bpy.data.objects[n] for n in dict.fromkeys(names)]
        not_rb = [o.name for o in objs if o.rigid_body is None]
        if not_rb:
            raise MifOpError("%s %s not a rigid body, so there is nothing for a joint to hold. Call add_rigid_body "
                             "first (it takes objects too). NOTHING was created."
                             % (", ".join(not_rb[:12]), "is" if len(not_rb) == 1 else "are"))
        act_name = take(params, "active", default=objs[0].name, kind=str)
        act = bpy.data.objects.get(act_name)
        if act is None or act.rigid_body is None:
            raise MifOpError("active '%s' must be a rigid body%s. NOTHING was created."
                             % (act_name, "" if act is None else " and is not one"))
        pattern = str(take(params, "pattern", default="SELECTED_TO_ACTIVE", kind=str)).upper()
        pivot = str(take(params, "pivot", default="CENTER", kind=str)).upper()
        connect = bpy.ops.rigidbody.connect.get_rna_type().properties
        for value, key, prop in ((pattern, "pattern", "connection_pattern"), (pivot, "pivot", "pivot_type")):
            valid = [i.identifier for i in connect[prop].enum_items]
            if value not in valid:
                raise MifOpError("%s must be one of %s (Blender's own Connect on this build), got '%s'. NOTHING was "
                                 "created." % (key, ", ".join(valid), value))
        others = [o for o in objs if o != act]
        if not others:
            raise MifOpError("objects holds only the active body '%s', so there is nothing to join it to. NOTHING was "
                             "created." % act.name)
        if pattern == "CHAIN_DISTANCE":
            order, last, pool = [act], act, list(others)
            while pool:
                pool.sort(key=lambda o: (last.matrix_world.translation - o.matrix_world.translation).length)
                last = pool.pop(0)
                order.append(last)
            raw = [(order[i - 1], order[i]) for i in range(1, len(order))]
        else:
            raw = [(act, o) for o in others]
        held = _joint_pairs_existing(rbw)
        for a, b in raw:
            if frozenset((a.name, b.name)) in held:
                skipped_pairs.append({"object1": a.name, "object2": b.name, "why": "a joint already holds this pair"})
            else:
                pairs.append((a, b))
        if not pairs:
            raise MifOpError("every pair (%d) is already held by a joint, so nothing would be made. NOTHING was created."
                             % len(skipped_pairs))
    else:
        a = get_object(take(params, "object1", required=True))
        b = get_object(take(params, "object2", required=True))
        if a is b:
            raise MifOpError("object1 and object2 are both '%s'. A rigid body joined to itself is not a "
                             "constraint Blender can solve. NOTHING was created." % a.name)
        for ob in (a, b):
            if ob.rigid_body is None:
                raise MifOpError("'%s' is not a rigid body, so there is nothing for a constraint to "
                                 "hold. Call add_rigid_body on it first. NOTHING was created." % ob.name)
        pairs = [(a, b)]

    loc_raw = take(params, "location", default=None)
    loc = None
    if loc_raw is not None:
        # CONVERTED HERE, not at the assignment forty lines down. A malformed vector is exactly what
        # made five ops in ops_create refuse with an object already in the caller's file.
        try:
            # finite_floats. I wrote this bare an hour before the audit learned to see it,
            # which is how location:[NaN,0,0] would have put a nan pivot on a hinge.
            loc = finite_floats(loc_raw, "location")
        except (TypeError, ValueError):
            raise MifOpError("location must be three numbers, got %r. NOTHING was created." % (loc_raw,))
        if len(loc) != 3:
            raise MifOpError("location needs 3 numbers, got %d. NOTHING was created." % len(loc))
    rot_raw = take(params, "rotation", default=None)
    rot = None
    if rot_raw is not None:
        if not isinstance(rot_raw, (list, tuple)) or len(rot_raw) != 3:
            raise MifOpError("rotation must be three numbers [x, y, z] in radians, got %r. NOTHING was created."
                             % (rot_raw,))
        try:
            rot = finite_floats(rot_raw, "rotation")
        except MifOpError as exc:
            raise MifOpError("%s NOTHING was created." % exc)
    settings = _joint_settings(params, "NOTHING was created.")
    wanted = str(take(params, "name", default="Joint", kind=str))
    cname = take(params, "collection", default=None, kind=str)
    dest, make_coll = _joint_collection(sc, cname, "NOTHING was created.")

    if rbw is None:
        # Both objects ARE rigid bodies, so a world existed when they were added. If it is gone the
        # scene has been edited underneath us and guessing at a replacement would silently give the
        # caller a second world their bodies are not in.
        raise MifOpError("this scene has no rigid body world, though %s are rigid bodies - it has been removed since "
                         "they were added. Re-add them with add_rigid_body. NOTHING was created."
                         % ", ".join("'%s'" % n for n in sorted({o.name for p in pairs for o in p})[:6]))
    if rbw.constraints is None:
        # CREATED, not refused: Blender leaves this None until the first constraint exists, so an
        # otherwise healthy scene reaches here on the very first call. This is the same collection
        # the Blender UI makes when you add a constraint by hand.
        rbw.constraints = bpy.data.collections.new("RigidBodyConstraints")
    if make_coll:
        dest = bpy.data.collections.new(cname)
        sc.collection.children.link(dest)

    made = []
    for a, b in pairs:
        obj = bpy.data.objects.new(wanted, None)
        dest.objects.link(obj)
        obj.empty_display_type = "ARROWS"
        if loc is not None:
            obj.location = loc
        elif bulk:
            pa, pb = a.matrix_world.translation, b.matrix_world.translation
            obj.location = pa if pivot == "ACTIVE" else pb if pivot == "SELECTED" else (pa + pb) / 2.0
        if rot is not None:
            obj.rotation_euler = rot
        rbw.constraints.objects.link(obj)
        rbc = obj.rigid_body_constraint
        if rbc is None:
            remove_objects([obj])
            raise MifOpError("linking into the rigid body world's constraint collection did not create the constraint "
                             "settings, which should not happen on any supported Blender. %s"
                             % ("NOTHING was left behind." if not made else
                                "The %d joint(s) before it WERE made: %s." % (len(made), ", ".join(made[:8]))))
        rbc.type = wanted_type
        rbc.object1 = a
        rbc.object2 = b
        for attr, val in settings.items():
            setattr(rbc, attr, val)
        made.append(obj.name)

    if bulk:
        return {"joints": [_joint_row(bpy.data.objects[n]) for n in made], "count": len(made),
                "pattern": pattern, "pivot": pivot, "constraintType": wanted_type,
                "collection": dest.name, "collectionCreated": make_coll,
                "pairsSkipped": skipped_pairs or None, "worldJointCount": len(rbw.constraints.objects),
                "bakeNote": "rigid bodies are stepped forward - bake_physics before a late frame."}

    obj = bpy.data.objects[made[0]]
    rbc = obj.rigid_body_constraint
    # READ OFF THE JOINT, not the request: the object is unparented, so its basis rotation is its world rotation.
    axis_local = _RBC_AXIS.get(rbc.type)
    axis_world = None
    if axis_local is not None:
        col = {"X": 0, "Z": 2}[axis_local]                  # a basis matrix's columns are its local axes in world
        vec = [float(obj.matrix_basis[r][col]) for r in range(3)]
        length = math.sqrt(sum(v * v for v in vec)) or 1.0
        axis_world = [round(v / length, 6) + 0.0 for v in vec]
    return {"object": obj.name, "requestedName": wanted, "nameWasSuffixed": obj.name != wanted,
            "constraintType": rbc.type, "object1": rbc.object1.name, "object2": rbc.object2.name,
            "location": [round(float(v), 6) for v in obj.location],
            "rotationEulerRad": [round(float(v), 6) for v in obj.rotation_euler],
            # None for FIXED and POINT, which have no axis, and for GENERIC / GENERIC_SPRING, whose limits are per axis
            # of the joint's frame - rotationEulerRad is that frame.
            "axisLocal": axis_local,
            "axisWorld": axis_world,
            "disableCollisions": bool(rbc.disable_collisions),
            "enabled": bool(rbc.enabled), "useBreaking": bool(rbc.use_breaking),
            "breakingThreshold": round(float(rbc.breaking_threshold), 6),
            "overrideSolverIterations": bool(rbc.use_override_solver_iterations),
            "solverIterations": int(rbc.solver_iterations),
            "collection": dest.name, "collectionCreated": make_coll,
            "worldJointCount": len(rbw.constraints.objects),
            "pivotNote": "the constraint object's ORIGIN is the pivot - a HINGE swings about it, so "
                         "put this object on the hinge line rather than at the world origin.",
            "bakeNote": "rigid bodies are stepped forward - bake_physics before a late frame."}


def _scene_cache_ranges(sc):
    """(label, holder, kind) for every simulation range bake_all bakes in this scene; each holder
    carries frame_start and frame_end. The rigid body world's cache; each modifier's own cache (cloth,
    soft body); each particle system's cache, which hangs off the SYSTEM rather than its modifier; and
    each dynamic-paint canvas SURFACE, whose own frame range is what Blender copies into its cache."""
    out = []
    rbw = sc.rigidbody_world
    if rbw and rbw.point_cache:
        out.append(("rigidbody", rbw.point_cache, "rigidbody"))
    for ob in sc.objects:
        for m in ob.modifiers:
            pc = getattr(m, "point_cache", None)
            if pc is not None:
                out.append(("%s:%s" % (ob.name, m.type.lower()), pc, "modifier"))
            canvas = getattr(m, "canvas_settings", None) if m.type == "DYNAMIC_PAINT" else None
            if canvas is not None:
                for surf in canvas.canvas_surfaces:
                    out.append(("%s:dynamic_paint:%s" % (ob.name, surf.name), surf, "dynamic_paint"))
        for ps in getattr(ob, "particle_systems", ()):
            if getattr(ps, "point_cache", None) is not None:
                out.append(("%s:particles:%s" % (ob.name, ps.name), ps.point_cache, "particles"))
    return out


def _write_range(holder, start, end, first="frame_start", last="frame_end"):
    """END FIRST, then start, then end again: a start past the old end would otherwise be clamped."""
    if end is not None:
        setattr(holder, last, int(end))
    if start is not None:
        setattr(holder, first, int(start))
    if end is not None and getattr(holder, last) != int(end):
        setattr(holder, last, int(end))


# A MANTAFLOW DOMAIN'S RANGE IS NOT A POINT CACHE'S, so _scene_cache_ranges - which walks point caches - never saw one,
# and bake_physics' start/end reached every simulation in the scene but fluid. Measured 2026-09-21 on 4.4.0 building a
# fire: a GAS domain made with cacheEnd 120, then bake_physics {start: 1, end: 220}, baked frames 1-120 and reported
# frames [1, 120] with isBaked true - the range asked for was neither written nor said to be missing, and a clip longer
# than the domain's own range would have frozen at its end. The range lives on domain_settings.cache_frame_start/end.
def _fluid_domains(sc):
    """(label, domain_settings) for every Mantaflow DOMAIN in the scene."""
    return [("%s:fluid" % ob.name, m.domain_settings) for ob in sc.objects for m in ob.modifiers
            if m.type == "FLUID" and getattr(m, "fluid_type", None) == "DOMAIN" and m.domain_settings is not None]


def _free_fluid_domain(ds):
    """Free a Mantaflow domain's baked frames NOW - data, noise, mesh and particles, and guides for REPLAY.

    NOT fluid.free_all. Its exec only STARTS a window-manager job (physics_fluid.cc fluid_free_exec -> WM_jobs_start,
    read at v4.4.0), which a headless Blender never runs before the next call - a rebake re-used the very frames it was
    meant to free - and which a GUI Blender runs later, deleting files a new bake may be writing, with the interface
    locked meanwhile. Writing time_scale onto itself runs Blender's own update for that property,
    rna_Fluid_datacache_reset (rna_fluid.c 3.6.0 and rna_fluid.cc 4.4.0 / 5.2.0): BKE_fluid_cache_free, synchronously,
    then a depsgraph tag - the same free the UI does when a setting is touched."""
    ds.time_scale = ds.time_scale


def _fluid_folder_frames(ds, sub):
    """The frame numbers in one subfolder of a domain's cache ("data", "noise", "mesh"...), or None without it."""
    root = bpy.path.abspath(ds.cache_directory) if ds.cache_directory else ""
    path = os.path.join(root, sub) if root else ""
    if not path or not os.path.isdir(path):
        return None
    return _fluid_frame_numbers(os.listdir(path))


def _fluid_frames_on_disk(ds):
    """How many simulated frames of `data` a domain's cache folder holds inside its own range, or None without one."""
    frames = _fluid_folder_frames(ds, "data")
    if frames is None:
        return None
    lo, hi = int(ds.cache_frame_start), int(ds.cache_frame_end)
    return len([f for f in frames if lo <= f <= hi])


def _fluid_modular_ops(ds):
    """The fluid.* bake operators a MODULAR domain needs, in order: data, then each part this domain produces."""
    ops = ["bake_data"]
    if ds.domain_type == "GAS" and getattr(ds, "use_noise", False):
        ops.append("bake_noise")
    if ds.domain_type == "LIQUID" and getattr(ds, "use_mesh", False):
        ops.append("bake_mesh")
    if ds.domain_type == "LIQUID" and any(getattr(ds, a, False) for a in ("use_spray_particles", "use_foam_particles",
                                                                         "use_bubble_particles")):
        ops.append("bake_particles")
    return ops


def _bake_fluid_domains(sc, rebake):
    """Bake every Mantaflow domain over its OWN range, the way Blender's fluid panel does for its cache type -> rows.

    WHY, 2026-09-21, building a fire: ptcache.bake_all is a point-cache baker. It steps the scene from the
    lowest to the highest point-cache range it finds (pointcache.cc BKE_ptcache_bake), and a domain's own point cache is
    not in RNA at all and starts at 1-250 - so a GAS domain asked for 720 frames stopped at 250, a start before frame 1
    was never stepped, and a MODULAR or ALL domain got nothing (measured 2026-09-16: 0.06 s, no files). A LIQUID hid it
    because its FLIP particle system carries the scene's range. So each domain is finished here:

      REPLAY    if the point-cache pass left frames of its range missing, the range is stepped in order from its start -
                frames already on disk are read back, the rest simulate and are written, which is REPLAY's own behaviour
      ALL       fluid.bake_all: data, noise, mesh, particles and guides in one pass over the domain's range
      MODULAR   fluid.bake_data, then bake_noise / bake_mesh / bake_particles where the domain has them

    The fluid operators' exec runs the whole bake in this thread (physics_fluid.cc fluid_bake_exec) - there is no job to
    wait on. A domain that already holds its range is left alone unless rebake.

    REPLAY DOMAINS ARE STEPPED TOGETHER: a frame change evaluates every domain in the scene, so one pass over the union
    of their ranges simulates them all - reported against each, with the others it shared that pass with. Stepping them
    one after another credited the whole 475 s to the first and read the other two as untouched (measured 2026-09-21,
    three domains, -119..180)."""
    rows = []
    domains = [(ob, m.domain_settings) for ob in sc.objects for m in ob.modifiers
               if m.type == "FLUID" and getattr(m, "fluid_type", None) == "DOMAIN" and m.domain_settings is not None]
    # ONE SNAPSHOT, BEFORE ANYTHING RUNS. A bake steps the whole scene, so every other REPLAY domain simulates along with
    # it: judged one at a time, a REPLAY domain reached after an ALL domain's fluid.bake_all found the frames that bake
    # had just written and was reported "kept" (tools/test_blender_fire_reach.py F122, 2026-09-21). "Kept" means
    # complete before this call began - after any rebake free.
    todo = []
    for ob, ds in domains:
        lo, hi = int(ds.cache_frame_start), int(ds.cache_frame_end)
        if (_fluid_frames_on_disk(ds) or 0) >= hi - lo + 1:
            rows.append({"object": ob.name, "cacheType": ds.cache_type, "range": [lo, hi], "how": None,
                         "kept": True, "seconds": 0.0})
        else:
            todo.append((ob, ds, lo, hi))

    # REPLAY FIRST, stepped together over the union of their ranges, each from its own start. Stepped after another
    # domain's bake, a pre-rolled REPLAY domain would already hold frames that bake wrote from frame 1 with no pre-roll
    # behind them, and would read those back instead of simulating them. ALL and MODULAR domains only READ their cache
    # while frames are stepped, so stepping cannot touch them.
    replay = [t for t in todo if t[1].cache_type == "REPLAY"]
    if replay:
        lo, hi = min(t[2] for t in replay), max(t[3] for t in replay)
        for _ob, ds, _lo, _hi in replay:
            tag_fluid_domain(ds)
        t0, back = time.time(), sc.frame_current
        for frame in range(lo, hi + 1):
            sc.frame_set(frame)
        sc.frame_set(back)
        seconds = round(time.time() - t0, 1)
        names = [t[0].name for t in replay]
        for ob, ds, own_lo, own_hi in replay:
            rows.append({"object": ob.name, "cacheType": ds.cache_type, "range": [own_lo, own_hi],
                         "how": "stepped %d-%d" % (lo, hi), "steppedWith": [n for n in names if n != ob.name] or None,
                         "seconds": seconds})

    for ob, ds, lo, hi in (t for t in todo if t[1].cache_type != "REPLAY"):
        ops = ["bake_all"] if ds.cache_type == "ALL" else _fluid_modular_ops(ds)
        t0 = time.time()
        retried = []
        try:
            for name in ops:
                _res, again = run_fluid_op(name, sc, ob)
                if again:
                    retried.append(name)
        except RuntimeError as exc:
            raise MifOpError("the fluid bake of '%s' (%s) failed: %s. The domains before it are baked; this one holds "
                             "%d of its %d frames."
                             % (ob.name, ds.cache_type, exc, _fluid_frames_on_disk(ds) or 0, hi - lo + 1))
        rows.append({"object": ob.name, "cacheType": ds.cache_type, "range": [lo, hi],
                     "how": "fluid." + " + fluid.".join(ops), "seconds": round(time.time() - t0, 1)})
        if retried:
            rows[-1]["retriedAfterPendingError"] = retried
    return rows


def _stored_frames(pc):
    """How many frames a cache actually holds, off its own report ("600 frames in memory (41.6 MiB)."),
    or None when it does not say. The range properties are not this: see _ranges_held."""
    text = (getattr(pc, "info", "") or "").strip()
    head = text.split(" ", 1)[0]
    return int(head) if head.isdigit() and " frames" in text else None


def _frames_by_cache(sc):
    """{label: frames stored} for every point cache in the scene, baked or not, off each cache's own report."""
    out = {}
    for label, holder, kind in _scene_cache_ranges(sc):
        pc = holder.point_cache if kind == "dynamic_paint" else holder
        if pc is not None:
            out[label] = _stored_frames(pc) or 0
    return out


def _ranges_held(sc, start, end):
    """(held, missed) over every cache, as (label, asked, holds) - read back after the bake.

    A WRITTEN RANGE IS NOT A HELD ONE, AND A RANGE PROPERTY IS NOT A STORED ONE. A particle system
    bakes over the SCENE's frame range whatever its cache is told. Measured 2026-09-12 in a 1-300 scene:
    asked 1-200 it stored 300 frames; asked 1-600 it stored 300, and a second call left its cache
    READING 1-600 over the same 300 stored - while every call listed it in rangeAppliedTo. The particles
    and vortex showcase finals baked 1-250 that way and froze at the emitter from frame 251 of 600.
    Rigid-body and cloth caches did hold. So a cache holds only if its range matches AND, once baked,
    the frames it stores cover that range.

    EXCEPT WHERE BLENDER RIGHTLY STOPS SHORT. A particle cache ends when its last particle can die -
    emitting to frame 100 with lifetime 50 stored exactly 1-151 - and nothing is alive after that, so it
    is judged against that end (_particle_needs). An unbaked cache - a rigid body world with no bodies
    stores 0 frames - is judged by its range alone, having had nothing to bake."""
    needs = _particle_needs(sc)
    held, missed = [], []
    for label, holder, kind in _scene_cache_ranges(sc):
        pc = holder.point_cache if kind == "dynamic_paint" else holder
        asked = [int(start) if start is not None else holder.frame_start,
                 int(end) if end is not None else holder.frame_end]
        holds = [holder.frame_start, holder.frame_end]
        stored = _stored_frames(pc) if pc is not None and pc.is_baked else None
        if kind == "particles":
            # FROM ITS FIRST BIRTH: a system emitting late stores nothing before it does, and that is
            # complete. Measured 2026-09-12 - cannons firing at 33 and 65 of 180 stored 148 and 116 frames.
            first_birth, last_death = needs.get(label, (asked[0], asked[1]))
            first, last = max(asked[0], first_birth), min(asked[1], last_death)
            ok = holds[1] >= last and (stored is None or stored >= last - first + 1)
        else:
            ok = asked == holds and (stored is None or stored >= asked[1] - asked[0] + 1)
        # STORED IS CARRIED OUT, because without it a rangeNotHeld entry can read as a
        # contradiction. Measured 2026-09-15 on 4.2.17: bake a cloth cache 1-60, then bake 1-120
        # WITHOUT clearing, and the entry said `asked [1, 120], holds [1, 120]` - identical - under
        # the heading "not held". The range really was written; what was not is the FRAMES, which
        # stayed at 60 because bake_all does not re-bake an already-baked cache. framesStored is the
        # only number that distinguishes the two, and it was the one number the entry did not carry.
        (held if ok else missed).append((label, asked, holds, stored))
    # AND EVERY FLUID DOMAIN, judged by what its cache folder holds - see _fluid_domains.
    for label, ds in _fluid_domains(sc):
        asked = [int(start) if start is not None else int(ds.cache_frame_start),
                 int(end) if end is not None else int(ds.cache_frame_end)]
        holds = [int(ds.cache_frame_start), int(ds.cache_frame_end)]
        stored = _fluid_frames_on_disk(ds)
        ok = asked == holds and stored is not None and stored >= asked[1] - asked[0] + 1
        (held if ok else missed).append((label, asked, holds, stored))
    return held, missed


def _particle_needs(sc):
    """label -> (first, last): the frames a particle system's cache has to span - its first birth to its
    last birth plus the lifetime, plus one. Blender starts and ends the cache there itself (emit 1-100
    with lifetime 50 stored 1-151; a burst at frame 33 of 180 stored 148 frames, 33-180)."""
    return {"%s:particles:%s" % (ob.name, ps.name):
            (int(ps.settings.frame_start), int(ps.settings.frame_end + ps.settings.lifetime) + 1)
            for ob in sc.objects for ps in getattr(ob, "particle_systems", ())}


# The cache-file frame reading lives in ops_common (fluid_frame_numbers), which the frame-change guard shares.
_FLUID_FRAME = FLUID_FRAME
_fluid_frame_numbers = fluid_frame_numbers


def _fluid_domain_row(ob, mod):
    """One `caches` row for a Mantaflow domain: its cache type, folder and the frames on disk per subfolder."""
    ds = mod.domain_settings
    root = bpy.path.abspath(ds.cache_directory) if ds.cache_directory else ""
    on_disk = {}
    if root and os.path.isdir(root):
        for sub in sorted(os.listdir(root)):
            path = os.path.join(root, sub)
            if not os.path.isdir(path):
                continue
            frames = _fluid_frame_numbers(os.listdir(path))
            on_disk[sub] = [min(frames), max(frames), len(frames)] if frames else [None, None, 0]
    start, end = int(ds.cache_frame_start), int(ds.cache_frame_end)
    # IN RANGE, not on disk: a folder can hold frames from an earlier, longer bake that this range does not read.
    stored = _fluid_frames_on_disk(ds) or 0
    covered = stored >= end - start + 1
    return {"kind": "fluid", "object": ob.name, "domainType": ds.domain_type, "cacheType": ds.cache_type,
            "cacheDirectory": root or None, "frames": [start, end], "framesOnDisk": on_disk or None,
            "framesStored": stored, "isBaked": covered,
            "displayParticleSystems": [ps.name for ps in getattr(ob, "particle_systems", ())
                                       if getattr(ps.settings, "type", "EMITTER") not in ("EMITTER", "HAIR")] or None}


def _group_inputs(group):
    """A node group's exposed inputs: name, identifier (the modifier key) and socket type."""
    if hasattr(group, "interface"):
        return [{"name": it.name, "identifier": it.identifier, "socketType": getattr(it, "socket_type", None)}
                for it in group.interface.items_tree
                if getattr(it, "item_type", "SOCKET") == "SOCKET" and it.in_out == "INPUT"]
    return [{"name": s.name, "identifier": s.identifier, "socketType": s.bl_socket_idname} for s in group.inputs]


def _has_sim_zone(group, seen=None):
    """True when a node group holds a Simulation Output, directly or in a nested group."""
    if group is None:
        return False
    seen = seen if seen is not None else set()
    if group.name in seen:
        return False
    seen.add(group.name)
    for n in group.nodes:
        if n.bl_idname == "GeometryNodeSimulationOutput":
            return True
        if getattr(n, "node_tree", None) is not None and _has_sim_zone(n.node_tree, seen):
            return True
    return False


def _object_caches(obj):
    """(label, kind, holder) for one object's caches: point caches ('pc': the cache) and geometry-node simulations
    ('nodes': the modifier). The label is the modifier, particle system or canvas surface name."""
    out = []
    for m in obj.modifiers:
        if m.type == "NODES" and _has_sim_zone(m.node_group):
            out.append((m.name, "nodes", m))
        elif m.type == "OCEAN":
            out.append((m.name, "ocean", m))
        elif getattr(m, "point_cache", None) is not None:
            out.append((m.name, "pc", m.point_cache))
        elif m.type == "DYNAMIC_PAINT" and getattr(m, "canvas_settings", None) is not None:
            for surf in m.canvas_settings.canvas_surfaces:
                out.append((surf.name, "pc", surf.point_cache))
    for ps in getattr(obj, "particle_systems", ()):
        if getattr(getattr(ps, "settings", None), "type", "EMITTER") in ("EMITTER", "HAIR"):
            out.append((ps.name, "pc", ps.point_cache))
    return out


def _node_bake_files(mod):
    """How many files a node simulation's disk bake holds under its directory, or None when it has no directory."""
    raw = getattr(mod, "bake_directory", "") or ""
    if not raw:
        return None
    root = bpy.path.abspath(raw)
    if not os.path.isdir(root):
        return 0
    return sum(len(files) for _d, _s, files in os.walk(root))


# AN OCEAN'S BAKE IS FILES (2026-10-05). object.ocean_bake writes disp_####.exr per frame into the modifier's filepath,
# foam_####.exr with foam and normal_####.exr with normals, from a JOB THREAD: measured on 4.2.23, 4.4.0, 5.0.1 and
# 5.2.2 under the bpy module it returns FINISHED at once and the files land over the next seconds - and the job's end
# step, which sets is_cached, runs on a window-manager timer a headless session never ticks, so is_cached stays false
# there and the modifier keeps simulating live. So the bake waits for the files it expects; free=True frees the cache
# and leaves the files (Blender's own behaviour, measured).
def _ocean_maps(mod):
    """The file prefixes this Ocean's bake writes per frame."""
    return ["disp"] + (["foam"] if mod.use_foam else []) + (["normal"] if mod.use_normals else []) + \
        (["spray"] if getattr(mod, "use_spray", False) else [])


def _ocean_files(mod):
    """The .exr maps in this Ocean's bake folder, by name."""
    root = bpy.path.abspath(mod.filepath) if mod.filepath else ""
    if not root or not os.path.isdir(root):
        return []
    return sorted(f for f in os.listdir(root) if f.endswith(".exr") and f.split("_")[0] in
                  ("disp", "foam", "normal", "spray"))


def _bake_ocean(obj, mod, free):
    root = bpy.path.abspath(mod.filepath)
    before = set(_ocean_files(mod))
    with bpy.context.temp_override(object=obj, active_object=obj):
        res = bpy.ops.object.ocean_bake(modifier=mod.name, free=bool(free))
    row = {"cache": mod.name, "kind": "ocean", "operator": sorted(res)[0] if res else None,
           "bakeDirectory": root, "frames": [int(mod.frame_start), int(mod.frame_end)]}
    if free:
        row.update({"freed": True, "isCached": bool(mod.is_cached), "filesOnDisk": len(_ocean_files(mod)),
                    "freeNote": "freeing an Ocean bake frees its cache; Blender leaves the map files on disk."})
        return row
    want = {"%s_%04d.exr" % (p, f) for p in _ocean_maps(mod) for f in range(int(mod.frame_start),
                                                                          int(mod.frame_end) + 1)}
    deadline = time.time() + 60.0 + 2.0 * len(want)
    sizes, stable = None, 0
    while time.time() < deadline:
        have = set(_ocean_files(mod))
        if want <= have:
            now = tuple(os.path.getsize(os.path.join(root, f)) for f in sorted(want))
            stable = stable + 1 if now == sizes else 0
            sizes = now
            if stable >= 2:
                break
        time.sleep(0.1)
    have = set(_ocean_files(mod))
    missing = sorted(want - have)
    row.update({"filesWritten": len((have - before) | (want & have)), "filesExpected": len(want),
                "maps": _ocean_maps(mod), "missing": missing[:20] or None, "isCached": bool(mod.is_cached)})
    if not mod.is_cached:
        row["cacheNote"] = ("the maps are on disk and is_cached is false: the bake runs as a job whose end step - which "
                            "sets is_cached and points the modifier at the files - runs on a window-manager timer, "
                            "and a headless session never ticks it (measured on 4.2-5.2). In an interactive "
                            "Blender it flips when the job ends; here the modifier keeps simulating live.")
    return row


def _bake_one_object(params, sc):
    """bake_physics for ONE object - see op_bake_physics's docstring. Every refusal comes before the first write."""
    for key in ("clear", "rebake", "start", "end"):
        if params.get(key) is not None:
            raise MifOpError("%s acts on every cache in the scene; with object, use free (or toFrame) on that object's "
                             "caches. NOTHING was changed." % key)
    obj = get_object(take(params, "object", kind=str))
    mod_name = take(params, "modifier", default=None, kind=str)
    dom = next((m for m in obj.modifiers if m.type == "FLUID" and getattr(m, "fluid_type", None) == "DOMAIN"), None)
    if dom is not None:
        ctype = getattr(dom.domain_settings, "cache_type", "MODULAR")
        if ctype != "MODULAR":
            # AN ALL OR REPLAY CACHE HAS NO STAGES: this used to point it at stages, which refuse it (settled
            # 2026-10-18, test_blender_settle_routes BS701) - the scene-wide bake is its route (fluid.bake_all).
            raise MifOpError("'%s' is a fluid domain with a %s cache, which bakes in one pass (fluid.bake_all for ALL) "
                             "- leave object out to bake every cache in the scene; stages exist for a MODULAR cache "
                             "only. NOTHING was changed." % (obj.name, ctype))
        raise MifOpError("'%s' is a fluid domain, which bakes by stage: pass stages (GUIDES, DATA, NOISE, MESH, "
                         "PARTICLES), resume or freeStages with it - or leave object out to bake every cache in the "
                         "scene. NOTHING was changed." % obj.name)
    caches = _object_caches(obj)
    if mod_name is not None:
        caches = [c for c in caches if c[0] == mod_name]
    if not caches:
        raise MifOpError("'%s' has no %s to bake (a cloth, soft body, particle system, dynamic-paint surface, an "
                         "Ocean modifier or a geometry-nodes modifier with a simulation zone). NOTHING was changed."
                         % (obj.name, "cache named %r" % mod_name if mod_name else "simulation cache"))
    free = take_bool(params, "free", default=False)
    to_frame = take_int(params, "toFrame", default=None)
    disk = take_bool(params, "diskCache", default=None) if params.get("diskCache") is not None else None
    if free and (to_frame is not None or disk is not None):
        raise MifOpError("free empties the caches; toFrame and diskCache fill them. Pass one. NOTHING was changed.")
    if to_frame is not None and disk is not None:
        raise MifOpError("toFrame steps the simulation without baking, and diskCache says where a bake is stored. "
                         "NOTHING was changed.")
    oceans = [c for c in caches if c[1] == "ocean"]
    if oceans and (to_frame is not None or disk is not None):
        raise MifOpError("%s is an Ocean modifier: its bake is always map files in its filepath, and it has no frames "
                         "to step - toFrame and diskCache do not apply. NOTHING was changed." % oceans[0][0])
    if oceans and not free:
        empty = [label for label, _k, m in oceans if not (m.filepath or "").strip()]
        if empty:
            raise MifOpError("Ocean modifier %s has no filepath, which is where its maps are written - set_modifier "
                             "{settings: {filepath: '//ocean/'}} first. NOTHING was changed." % ", ".join(empty))
    if to_frame is not None and not sc.frame_start <= to_frame <= 1048574:
        raise MifOpError("toFrame %d is before the scene's start frame %d. NOTHING was changed."
                         % (to_frame, sc.frame_start))
    ver = bpy.app.version
    saved = bool(bpy.data.filepath)
    nodes = [c for c in caches if c[1] == "nodes"]
    if not free and to_frame is None:
        if disk and not saved:
            raise MifOpError("a disk cache is written beside the .blend, and this file has never been saved. Save it "
                             "(save_file) first. NOTHING was changed.")
        if nodes and ver < (4, 2, 0):
            raise MifOpError("a geometry-nodes simulation bake does not finish headless on Blender %s - its operator "
                             "runs modal and never returns under -b (measured, ledger B267). Use toFrame to step it, or "
                             "Blender 4.2 or later. NOTHING was changed." % bpy.app.version_string)
        if nodes and ver < (4, 4, 0) and not saved:
            raise MifOpError("Blender %s bakes a geometry-nodes simulation only in a saved file (it writes files "
                             "beside it). Save it first. NOTHING was changed." % bpy.app.version_string)
        if nodes and disk is False and not hasattr(nodes[0][2], "bake_target"):
            raise MifOpError("diskCache false asks for a bake packed into the .blend, which Blender %s cannot do - "
                             "its node bakes always go to files. NOTHING was changed." % bpy.app.version_string)
    if to_frame is not None:
        baked = [label for label, kind, holder in caches if kind == "pc" and holder.is_baked]
        if baked:
            raise MifOpError("%s %s baked, so stepping to frame %d computes nothing - free first. NOTHING was changed."
                             % (", ".join(baked), "is" if len(baked) == 1 else "are", to_frame))
    rows = []
    sel = {"selected_objects": [obj], "selected_editable_objects": [obj], "active_object": obj, "object": obj}
    for label, kind, holder in caches:
        if kind == "ocean":
            rows.append(_bake_ocean(obj, holder, free))
            continue
        row = {"cache": label, "kind": "geometry_nodes_simulation" if kind == "nodes" else "point_cache"}
        if kind == "pc":
            if free:
                with bpy.context.temp_override(scene=sc, point_cache=holder, object=obj, active_object=obj):
                    bpy.ops.ptcache.free_bake()
                # FREE_BAKE LEAVES A DISK CACHE'S FRAMES (250 of 250 kept, measured 2026-10-04 on 5.2.1) and the next
                # run replays them - the trap bake_physics clear already handles: rename the cache, step to its start.
                if (_stored_frames(holder) or 0) > 0:
                    holder.name = holder.name
                    sc.frame_set(int(holder.frame_start))
            elif to_frame is not None:
                for f in range(int(holder.frame_start), min(to_frame, int(holder.frame_end)) + 1):
                    sc.frame_set(f)
            else:
                if disk is not None:
                    holder.use_disk_cache = disk
                with bpy.context.temp_override(scene=sc, point_cache=holder, object=obj, active_object=obj):
                    bpy.ops.ptcache.bake(bake=True)
            row.update({"isBaked": bool(holder.is_baked), "framesStored": _stored_frames(holder),
                        "diskCache": bool(holder.use_disk_cache), "frames": [int(holder.frame_start),
                                                                              int(holder.frame_end)]})
        else:
            if free:
                with bpy.context.temp_override(**sel):
                    res = bpy.ops.object.simulation_nodes_cache_delete(selected=True)
            elif to_frame is not None:
                for f in range(sc.frame_start, to_frame + 1):
                    sc.frame_set(f)
                res = {"STEPPED"}
            else:
                if disk is not None and hasattr(holder, "bake_target"):
                    holder.bake_target = "DISK" if disk else "PACKED"
                # AN EMPTY bake_directory WRITES NOTHING: a DISK bake answered FINISHED and played back with no file
                # anywhere beside the .blend (5.2.1, 2026-10-04); with //simbake/ set it wrote 500. So one is set.
                if (disk or not hasattr(holder, "bake_target")) and not (getattr(holder, "bake_directory", "") or ""):
                    holder.bake_directory = "//%s_bake/" % bpy.path.clean_name("%s_%s" % (obj.name, holder.name))
                with bpy.context.temp_override(**sel):
                    res = bpy.ops.object.simulation_nodes_cache_bake(selected=True)
            row.update({"operator": sorted(res)[0] if res else None,
                        "bakeTarget": getattr(holder, "bake_target", "DISK"),
                        "group": holder.node_group.name if holder.node_group else None,
                        "bakeDirectory": getattr(holder, "bake_directory", None) or None,
                        "filesOnDisk": _node_bake_files(holder)})
        rows.append(row)
    out = {"object": obj.name, "action": "free" if free else ("toFrame" if to_frame is not None else "bake"),
           "caches": rows, "frame": sc.frame_current}
    if to_frame is not None:
        out["toFrame"] = to_frame
    if nodes:
        out["nodesNote"] = ("Blender exposes no baked flag for a geometry-nodes simulation: operator is its own answer "
                            "(FINISHED), filesOnDisk counts a DISK bake's files, and a PACKED bake (the default from "
                            "4.4) is inside the .blend - a bake plays back when frames are jumped out of order, which "
                            "a live simulation cannot do.")
    return out


# A MANTAFLOW DOMAIN BAKED STAGE BY STAGE (2026-10-04). Blender's fluid panel bakes a MODULAR cache in stages - guides,
# data, then noise (gas) or mesh and particles (liquid) - and frees each on its own. Stage order, the operator each
# stage runs, the subfolder its frames land in, and its pause-frame property.
_FLUID_STAGES = ("GUIDES", "DATA", "NOISE", "MESH", "PARTICLES")
_STAGE_OP = {"GUIDES": "bake_guides", "DATA": "bake_data", "NOISE": "bake_noise", "MESH": "bake_mesh",
             "PARTICLES": "bake_particles"}
_STAGE_DIR = {"GUIDES": "guiding", "DATA": "data", "NOISE": "noise", "MESH": "mesh", "PARTICLES": "particles"}
_STAGE_FLAG = {"GUIDES": "guide", "DATA": "data", "NOISE": "noise", "MESH": "mesh", "PARTICLES": "particles"}
# THE PER-STAGE FREE: a setting written onto itself, whose RNA update frees exactly that stage's cache synchronously -
# rna_Fluid_noisecache_reset, _meshcache_reset, _particlescache_reset, _datacache_reset and _guidingcache_reset call
# BKE_fluid_cache_free with that stage's flag (rna_fluid.cc, read at v4.4.0). NOT fluid.free_*: their exec starts a
# window-manager job, and headless on 3.6.23 and 4.4.0 it answered FINISHED in 0.0 s and freed nothing (5.2.1 freed).
# MEASURED 2026-10-04, identical on 3.6.23, 4.4.0 and 5.2.1 (MODULAR, resumable, 24 divisions, 5-6 frames):
#   mesh_particle_radius   mesh/ 6 -> 0 files, data and particles kept          (pause_mesh 6 -> 0)
#   sndparticle_potential_min_wavecrest   particles/ 6 -> 0, data kept          (pause_particles 6 -> 0)
#   noise_strength         noise/ 5 -> 0, data kept                            (pause_noise 5 -> 0)
#   time_scale             data, config, noise, mesh and particles all to 0 - data's free takes what is built on it
#   guide_parent           guiding/ and every other stage to 0 - Blender's own free_guides frees the same set
# A freed stage bakes again: mesh 0 -> 6 files after its free, noise 0 -> 5.
_STAGE_FREE_BY = {"DATA": "time_scale", "NOISE": "noise_strength", "MESH": "mesh_particle_radius",
                  "PARTICLES": "sndparticle_potential_min_wavecrest", "GUIDES": "guide_parent"}
# WHAT EACH FREE TAKES WITH IT, as Blender's own free operators define it (physics_fluid.cc fluid_free_startjob).
_STAGE_FREES_ALSO = {"DATA": ("NOISE", "MESH", "PARTICLES"), "GUIDES": ("DATA", "NOISE", "MESH", "PARTICLES"),
                     "NOISE": (), "MESH": (), "PARTICLES": ()}


def _stage_files(ds):
    """{subfolder: file count} of a domain's cache folder."""
    root = bpy.path.abspath(ds.cache_directory) if ds.cache_directory else ""
    out = {}
    if root and os.path.isdir(root):
        for sub in sorted(os.listdir(root)):
            path = os.path.join(root, sub)
            if os.path.isdir(path):
                out[sub] = len(os.listdir(path))
    return out


def _stage_state(ds, stage):
    """(baked, pause frame) of one stage, off the domain."""
    flag = _STAGE_FLAG[stage]
    return bool(getattr(ds, "has_cache_baked_" + flag)), int(getattr(ds, "cache_frame_pause_" + flag))


def _stage_applies(ds, stage):
    """None when this domain has the stage, else why it does not."""
    if stage == "GUIDES":
        if not ds.use_guide:
            return "use_guide is off"
        if ds.guide_source != "EFFECTOR":
            return ("guide_source is %s - a DOMAIN-source guide follows its parent domain's data and has nothing of "
                    "its own to bake" % ds.guide_source)
    elif stage == "NOISE":
        if ds.domain_type != "GAS" or not ds.use_noise:
            return "noise is a GAS domain's stage with use_noise on (this is %s, use_noise %s)" % (ds.domain_type,
                                                                                                ds.use_noise)
    elif stage == "MESH":
        if ds.domain_type != "LIQUID" or not ds.use_mesh:
            return "mesh is a LIQUID domain's stage with use_mesh on (this is %s, use_mesh %s)" % (ds.domain_type,
                                                                                                ds.use_mesh)
    elif stage == "PARTICLES":
        if ds.domain_type != "LIQUID" or not any(getattr(ds, a, False) for a in (
                "use_spray_particles", "use_foam_particles", "use_bubble_particles")):
            return ("particles is a LIQUID domain's stage with spray, foam or bubbles on (use_spray_particles, "
                    "use_foam_particles, use_bubble_particles)")
    return None


def _unbaked_guide(ds):
    """Why the data bake must wait for guides, or None - Blender's own panel rule (check_domain_has_unbaked_guide)."""
    if not ds.use_guide or ds.has_cache_baked_guide:
        return None
    if ds.guide_source == "EFFECTOR":
        return "this domain's guides come from an EFFECTOR and are not baked"
    if ds.guide_source == "DOMAIN" and ds.guide_parent is None:
        return "this domain follows a DOMAIN guide and no guide_parent is set"
    return None


def _stage_list(raw, key):
    if not isinstance(raw, (list, tuple)) or not raw or not all(isinstance(s, str) for s in raw):
        raise MifOpError("%s must be a non-empty list of stages from %s, got %r. NOTHING was baked."
                         % (key, ", ".join(_FLUID_STAGES), raw))
    out = [str(s).upper() for s in raw]
    bad = [s for s in out if s not in _FLUID_STAGES]
    if bad:
        raise MifOpError("%s takes %s; %s %s not a stage. NOTHING was baked."
                         % (key, ", ".join(_FLUID_STAGES), ", ".join(bad), "is" if len(bad) == 1 else "are"))
    return [s for s in _FLUID_STAGES if s in out]


def run_fluid_op(name, sc, ob):
    """bpy.ops.fluid.<name>() on one domain under a context override. Returns (result, the error retried after or None).

    A PENDING MANTAFLOW ERROR, NOT THIS BAKE'S (settled 2026-10-18, test_blender_settle_routes BS701): on 4.4.0, in 2 to 3
    of 6 fresh sessions with a guided gas domain, the first fluid bake raised "ViewLayer.update() returned a result with an
    exception set" (cause: the fluid script's "'NoneType' object has no attribute 'getDataPointer'") from bpy.ops' own
    pre-call view-layer update, before the operator ran. Raising it clears it, and the same call made again bakes - so it
    is made once more, and the caller says so. Any other SystemError, or a second one, is raised as a RuntimeError."""
    retried = None
    for attempt in (0, 1):
        try:
            with bpy.context.temp_override(scene=sc, view_layer=bpy.context.view_layer, object=ob, active_object=ob):
                return getattr(bpy.ops.fluid, name)(), retried
        except SystemError as exc:
            if attempt or "exception set" not in str(exc):
                raise RuntimeError(str(exc)) from exc
            retried = str(exc)
    return None, retried


def _bake_fluid_stages(params, sc, start, end):
    """bake_physics with stages / freeStages / resume: one Mantaflow domain, stage by stage. See _FLUID_STAGES."""
    for key in ("clear", "rebake"):
        if key in params:
            raise MifOpError("%s works on every cache in the scene; stages, freeStages and resume work on one fluid "
                             "domain - freeStages is the per-stage clear. NOTHING was baked." % key)
    domains = [(ob, m) for ob in sc.objects for m in ob.modifiers
               if m.type == "FLUID" and getattr(m, "fluid_type", None) == "DOMAIN" and m.domain_settings is not None]
    name = take(params, "object", default=None, kind=str)
    if name is not None:
        pick = [(ob, m) for ob, m in domains if ob.name == name]
        if not pick:
            get_object(name)
            raise MifOpError("'%s' is not a fluid DOMAIN. Domains in this scene: %s. NOTHING was baked."
                             % (name, ", ".join(ob.name for ob, _m in domains) or "none"))
    elif len(domains) == 1:
        pick = domains
    else:
        raise MifOpError("a stage bake works on one fluid domain and this scene has %d (%s) - pass object. NOTHING "
                         "was baked." % (len(domains), ", ".join(ob.name for ob, _m in domains) or "none"))
    ob, mod = pick[0]
    ds = mod.domain_settings
    if ds.cache_type != "MODULAR":
        raise MifOpError("'%s' caches as %s; stages exist for a MODULAR cache only (ALL bakes everything in one pass, "
                         "REPLAY steps frames) - set_modifier {cache_type: 'MODULAR'} first, or bake without stages. "
                         "NOTHING was baked." % (ob.name, ds.cache_type))
    stages = _stage_list(params["stages"], "stages") if params.get("stages") is not None else []
    frees = _stage_list(params["freeStages"], "freeStages") if params.get("freeStages") is not None else []
    resume = take_bool(params, "resume", default=False)
    if resume and not stages:
        stages = [s for s in _FLUID_STAGES if _stage_applies(ds, s) is None
                  and _stage_state(ds, s)[0] and _stage_state(ds, s)[1] < int(end if end is not None
                                                                             else ds.cache_frame_end)]
        if not stages:
            raise MifOpError("resume found no stage of '%s' to continue: none is baked with its pause frame before the "
                             "end (%d). NOTHING was baked." % (ob.name, int(end if end is not None
                                                                           else ds.cache_frame_end)))
    if not stages and not frees:
        raise MifOpError("pass stages, freeStages or resume. NOTHING was baked.")

    for s in set(stages) | set(frees):
        why = _stage_applies(ds, s)
        if why is not None:
            raise MifOpError("'%s' has no %s stage: %s. NOTHING was baked." % (ob.name, s, why))
    cache_end = int(end if end is not None else ds.cache_frame_end)
    # WHAT EACH STAGE WILL FIND, after the frees this call makes and the stages before it.
    freed_now = set(frees)
    for s in frees:
        freed_now.update(_STAGE_FREES_ALSO[s])
    plan_baked = {s: (_stage_state(ds, s)[0] and s not in freed_now) for s in _FLUID_STAGES}
    for s in stages:
        baked, pause = _stage_state(ds, s)
        if s in freed_now:
            baked, pause = False, 0
        if resume:
            if not ds.cache_resumable:
                raise MifOpError("resume needs a resumable cache, and '%s' has cache_resumable off - a paused bake of it "
                                 "cannot continue. NOTHING was baked." % ob.name)
            if not baked:
                raise MifOpError("resume continues a baked stage, and %s of '%s' is not baked - leave resume out to "
                                 "bake it. NOTHING was baked." % (s, ob.name))
            if pause >= cache_end:
                raise MifOpError("%s of '%s' is complete (its pause frame %d is the end, %d), so there is nothing to "
                                 "resume. Raise end to extend it, or free it to bake again. NOTHING was baked."
                                 % (s, ob.name, pause, cache_end))
        elif baked:
            # A BAKED STAGE IS NOT BAKED AGAIN FROM THE START by its operator: the bake begins at the pause frame
            # (physics_fluid.cc fluid_bake_sequence), so on a finished stage it redoes the last frame only.
            raise MifOpError("%s of '%s' is already baked (pause frame %d of %d). Pass freeStages [%s] to bake it again "
                             "from the start, or resume true to continue it. NOTHING was baked."
                             % (s, ob.name, pause, cache_end, s))
        if s == "DATA":
            why = _unbaked_guide(ds) if "GUIDES" not in stages and not plan_baked["GUIDES"] else None
            if why is None and ds.use_guide and "GUIDES" in freed_now and "GUIDES" not in stages:
                why = "this call frees the guides and does not bake them again"
            if why is not None:
                raise MifOpError("DATA of '%s' cannot bake yet: %s. Blender's panel withholds Bake Data until the guides "
                                 "are baked (its operator runs regardless - measured, it bakes data that ignores the "
                                 "guide). Bake GUIDES first, in this call or before. NOTHING was baked." % (ob.name, why))
        if s in ("NOISE", "MESH", "PARTICLES"):
            if not ds.cache_resumable:
                raise MifOpError("%s of '%s' needs a resumable cache: Blender's panel withholds the noise, mesh and "
                                 "particle bakes when cache_resumable is off (its operator writes frames anyway - "
                                 "measured - which the panel does not stand behind). Turn cache_resumable on BEFORE the "
                                 "data bake: writing it frees every stage (measured). NOTHING was baked."
                                 % (s, ob.name))
            if not plan_baked["DATA"] and "DATA" not in stages:
                raise MifOpError("%s of '%s' is built on the data stage, which is not baked%s. Add DATA to stages. "
                                 "NOTHING was baked." % (s, ob.name, " after this call's frees" if "DATA" in freed_now
                                                         else ""))
    if ds.use_adaptive_domain and ds.use_guide and ds.guide_source == "EFFECTOR" and (
            "GUIDES" in stages or "DATA" in stages):
        raise MifOpError("'%s' has use_adaptive_domain on with EFFECTOR guides - Blender removes the Adaptive Domain "
                         "panel for effector guides ('require a fixed domain size'). Turn it off with set_modifier "
                         "{use_adaptive_domain: false}. NOTHING was baked." % ob.name)

    # ---- writes. The range first (written only when it differs - see bake_physics), then frees, then bakes.
    range_written = False
    if ((start is not None and int(ds.cache_frame_start) != int(start))
            or (end is not None and int(ds.cache_frame_end) != int(end))):
        _write_range(ds, start, end, "cache_frame_start", "cache_frame_end")
        tag_fluid_domain(ds)
        range_written = True
    freed = []
    for s in frees:
        before = _stage_files(ds)
        prop = _STAGE_FREE_BY[s]
        setattr(ds, prop, getattr(ds, prop))
        tag_fluid_domain(ds)
        freed.append({"stage": s, "by": prop, "alsoFrees": list(_STAGE_FREES_ALSO[s]) or None,
                      "filesBefore": before, "filesAfter": _stage_files(ds)})
    rows = []
    for s in stages:
        _baked, pause_before = _stage_state(ds, s)
        t0 = time.time()
        try:
            result, retried = run_fluid_op(_STAGE_OP[s], sc, ob)
        except RuntimeError as exc:
            raise MifOpError("fluid.%s on '%s' failed: %s. %s" % (_STAGE_OP[s], ob.name, exc,
                                                                   "The stages before it ARE baked: %s."
                                                                   % ", ".join(r["stage"] for r in rows)
                                                                   if rows else ("NOTHING was baked%s."
                                                                                 % (" (the frees above WERE made)"
                                                                                    if freed else ""))))
        baked, pause = _stage_state(ds, s)
        folder = _fluid_folder_frames(ds, _STAGE_DIR[s]) or set()
        kinds = None
        if s == "PARTICLES":
            # RULE (d): each kind adds particle work to this stage, and which ones is the caller's choice.
            kinds = [k for k, a in (("spray", "use_spray_particles"), ("foam", "use_foam_particles"),
                                    ("bubbles", "use_bubble_particles")) if getattr(ds, a, False)]
        rows.append({"stage": s, "operator": "fluid." + _STAGE_OP[s], "result": sorted(result), "particleKinds": kinds,
                     "seconds": round(time.time() - t0, 2), "resumedFrom": pause_before if resume else None,
                     "baked": baked, "pauseFrame": pause,
                     "framesOnDisk": [min(folder), max(folder), len(folder)] if folder else [None, None, 0]})
        if retried:
            rows[-1]["retriedAfter"] = retried
    root = bpy.path.abspath(ds.cache_directory) if ds.cache_directory else ""
    tmp = os.path.normcase(os.path.abspath(bpy.app.tempdir)) if bpy.app.tempdir else ""
    out = {"baked": bool(rows), "object": ob.name, "stages": rows or None, "freed": freed or None,
           "rangeWritten": range_written, "range": [int(ds.cache_frame_start), int(ds.cache_frame_end)],
           "cacheResumable": bool(ds.cache_resumable), "filesOnDisk": _stage_files(ds), "cacheDirectory": root or None,
           # RULE (c): the time scale is a look tied to scene scale and fps, so it is reported, never chosen here.
           "timeScale": round(float(ds.time_scale), 6), "resolutionMax": int(ds.resolution_max)}
    if any(r["stage"] == "DATA" for r in rows) and not ds.use_guide:
        # RULE (b), the unguided half.
        out["resolutionNote"] = ("an unguided domain can be iterated at a low resolution (64) and raised for the final "
                                 "bake only: free DATA, write resolution_max and bake again.")
    if any(r["stage"] == "GUIDES" for r in rows):
        out["guideNote"] = ("the guides are baked, so resolution_max and time_scale are fixed from here: writing either "
                            "frees the data and KEEPS the guides (measured on 3.6.23, 4.4.0 and 5.2.1), which leaves "
                            "guides made at the old setting. Free GUIDES before changing them.")
    if ds.use_adaptive_domain and ds.use_guide and ds.guide_source == "DOMAIN":
        out["adaptiveNote"] = ("use_adaptive_domain is on with a DOMAIN guide. Blender hides the Adaptive Domain panel "
                               "only for effector guides, and whether it suits a domain guide is unmeasured - a guided "
                               "domain is usually baked with it off.")
    if root and tmp and os.path.normcase(os.path.abspath(root)).startswith(tmp):
        out["cacheNote"] = ("this domain's cache is in Blender's session temp folder (%s), which is deleted when Blender "
                            "closes - set cache_directory (set_modifier) to a folder of its own, one per domain, before "
                            "a bake meant to survive." % root)
    return out


def _bake_node(params, sc):
    """bake_physics bakeNode (1.3 gap audit, 2026-10-19): bake (or free) ONE geometry-nodes Bake node.

    MEASURED 2026-10-19: object.simulation_nodes_cache_bake - this op's one-object node bake - wrote no file for a Bake
    node on 4.4.0 and 5.2.1 (0 files, FINISHED), while object.geometry_node_bake_single wrote it (1-2 files). 4.2.17 needs
    the .blend saved first ("File must be saved before baking"); 3.6 has no Bake node. So the single-node operator is the
    route, by the node's name or its bake id."""
    tail = "NOTHING was changed."
    if "bakes" not in bpy.types.NodesModifier.bl_rna.properties:
        raise MifOpError("the geometry-nodes Bake node arrived in Blender 4.2; this is %s. %s"
                         % (bpy.app.version_string, tail))
    obj = get_object(take(params, "object", required=True, kind=str))
    mods = [m for m in obj.modifiers if m.type == "NODES"]
    want_mod = take(params, "modifier", default=None, kind=str)
    if want_mod is not None:
        mods = [m for m in mods if m.name == want_mod]
        if not mods:
            raise MifOpError("'%s' has no geometry-nodes modifier '%s'. %s" % (obj.name, want_mod, tail))
    key = params.get("bakeNode")
    hits = []
    for m in mods:
        for b in m.bakes:
            node = getattr(b, "node", None)
            if (isinstance(key, str) and node is not None and node.name == key) or (
                    isinstance(key, int) and not isinstance(key, bool) and b.bake_id == key):
                hits.append((m, b))
    if len(hits) != 1:
        have = ["%s: %s (id %d)" % (m.name, getattr(b.node, "name", "?"), b.bake_id) for m in mods for b in m.bakes]
        raise MifOpError("bakeNode %r matched %d Bake nodes on '%s' - give the node's name or bake id (and modifier "
                         "when two modifiers share one). It has: %s. %s"
                         % (key, len(hits), obj.name, "; ".join(have) or "no Bake node", tail))
    mod, bake = hits[0]
    mode = params.get("bakeMode")
    if mode is not None:
        mode = str(mode).upper()
        if mode not in ("STILL", "ANIMATION"):
            raise MifOpError("bakeMode is STILL or ANIMATION, got %r. %s" % (params.get("bakeMode"), tail))
    free = take_bool(params, "free", default=False)
    disk = take_bool(params, "diskCache") if params.get("diskCache") is not None else None
    for k in ("toFrame", "start", "end", "clear", "rebake"):
        if params.get(k) is not None:
            raise MifOpError("%s is the whole-cache bake's; bakeNode bakes one node (bakeMode STILL or ANIMATION). %s"
                             % (k, tail))
    if mode is not None:
        bake.bake_mode = mode
    if disk is not None and hasattr(bake, "bake_target"):
        bake.bake_target = "DISK" if disk else "PACKED"
    target = getattr(bake, "bake_target", "DISK")
    if target == "INHERIT":
        target = getattr(mod, "bake_target", "DISK")
    if target == "DISK" and not (getattr(mod, "bake_directory", "") or "") and not getattr(bake, "use_custom_path", False):
        mod.bake_directory = "//%s_bake/" % bpy.path.clean_name("%s_%s" % (obj.name, mod.name))
    sel = {"selected_objects": [obj], "selected_editable_objects": [obj], "active_object": obj, "object": obj}
    op = "geometry_node_bake_delete_single" if free else "geometry_node_bake_single"
    try:
        with bpy.context.temp_override(**sel):
            res = getattr(bpy.ops.object, op)(session_uid=obj.session_uid, modifier_name=mod.name, bake_id=bake.bake_id)
    except RuntimeError as exc:
        raise MifOpError("Blender refused the %s: %s%s" % ("free" if free else "bake", str(exc).strip(),
                                                           " - save_file first" if "saved" in str(exc) else ""))
    return {"object": obj.name, "modifier": mod.name, "bakeNode": getattr(bake.node, "name", None),
            "bakeId": bake.bake_id, "action": "free" if free else "bake", "operator": sorted(res)[0] if res else None,
            "bakeMode": bake.bake_mode, "bakeTarget": getattr(bake, "bake_target", "DISK"),
            "bakeDirectory": getattr(mod, "bake_directory", None) or None, "filesOnDisk": _node_bake_files(mod)}


def op_bake_physics(params):
    """Bake the point caches so a given frame shows the simulated state, not the rest state.

    BLOCKS for the length of the bake, like any main-thread work here.

    ONE OBJECT (2026-10-04): object (and modifier: one modifier, particle system or canvas surface by name) bakes only
    that object's caches - a cloth, soft body, particle system, dynamic-paint surface, or a geometry-nodes modifier
    whose group holds a simulation zone. diskCache true writes the bake beside the saved .blend (a node bake's
    bake_target DISK; false packs a node bake into the file, 4.4 and later); toFrame N steps the caches to frame N
    without baking (Blender's Calculate to Frame); free empties them. A node-simulation bake runs headless from 4.2
    and needs a saved file before 4.4; on 3.6 its operator never returns under -b and is refused (ledger B267).

    ONE BAKE NODE (1.3, 4.2+): object + bakeNode (the Bake node's name or bake id; modifier when two share one) bakes that
    node alone (object.geometry_node_bake_single) - bakeMode STILL | ANIMATION, diskCache, free. The whole-cache node
    bake above does NOT bake Bake nodes (measured on 4.4.0 and 5.2.1: no file).
    """
    reject_unknown(params, _BAKE_KEYS, "bake_physics")
    sc = bpy.context.scene
    if params.get("bakeNode") is not None:
        return _bake_node(params, sc)
    if params.get("bakeMode") is not None:
        raise MifOpError("bakeMode goes with bakeNode (one geometry-nodes Bake node). NOTHING was changed.")
    # TWO FORMS NAME AN OBJECT, and the stage keys decide which: stages / resume / freeStages bake one fluid domain stage
    # by stage (object picks the domain); object alone bakes one object's point caches and node simulations.
    stage_form = any(params.get(k) is not None for k in ("stages", "resume", "freeStages"))
    if stage_form:
        mixed = [k for k in ("modifier", "diskCache", "toFrame", "free") if params.get(k) is not None]
        if mixed:
            raise MifOpError("%s belong(s) to the one-object bake, and stages, resume and freeStages to a fluid domain's "
                             "stage bake - freeStages is that bake's free. Pass one form. NOTHING was changed."
                             % ", ".join(mixed))
    elif params.get("object") is not None:
        return _bake_one_object(params, sc)
    for key in ("modifier", "diskCache", "toFrame", "free"):
        if params.get(key) is not None:
            raise MifOpError("%s goes with object - name the object whose caches it acts on. NOTHING was changed."
                             % key)
    # take_int: every cache range is whole frames, and take_float then int() baked {start: 1.5} from frame 1 under ok:true
    # - the same floor the 2026-09-21 pack-rule review named in add_particles, add_cloth, add_fluid and add_dynamic_paint.
    start = take_int(params, "start", default=None)
    end = take_int(params, "end", default=None)
    # PARSED AND REFUSED ABOVE THE FIRST WRITE. clear/rebake were read below the cache-range writes, so
    # {start, end, clear, rebake} rewrote every cache's range and THEN answered "NOTHING was changed"
    # (audit_mutate_then_deny, 2026-09-16 - the rebake key was added the same morning without moving it).
    clear = take_bool(params, "clear", default=False)
    rebake = take_bool(params, "rebake", default=False)
    if clear and rebake:
        raise MifOpError("clear and rebake are two different requests - clear frees every cache and bakes "
                         "nothing, rebake frees and then bakes. Pass one. NOTHING was changed.")
    # ONE FLUID DOMAIN, STAGE BY STAGE - see _FLUID_STAGES. Nothing below runs for it: no point cache is baked.
    if stage_form:
        return _bake_fluid_stages(params, sc, start, end)
    # start/end GO TO EVERY CACHE bake_all will bake, not only the rigid body world's. They used to
    # reach that one alone, so a cloth, soft body, particle system or dynamic-paint surface kept
    # Blender's default range of 1-250 and stopped simulating at frame 250 whatever the caller asked
    # for - invisible in any clip of 250 frames or fewer, which is every suite fixture and every
    # showcase preview. Found 2026-09-11 reading this op before the first 600-frame showcase renders.
    asked_range = start is not None or end is not None
    if asked_range:
        caches_now = _scene_cache_ranges(sc)
        lo = int(start) if start is not None else sc.frame_start
        hi = int(end) if end is not None else sc.frame_end
        # REFUSED BEFORE ANYTHING IS BAKED: a particle system bakes over the scene's range, so a request
        # that runs past it would bake the particles short under a report that looked complete. Their
        # cache range is not written either - it changes a number and not the bake (_ranges_held).
        particles = [label for label, _h, kind in caches_now if kind == "particles"]
        if particles and (lo < sc.frame_start or hi > sc.frame_end):
            raise MifOpError("a particle system bakes over the SCENE's frame range (%d-%d), not the range "
                             "asked (%d-%d) - its cache can read the asked range and still store only the "
                             "scene's frames. Extend the scene first: set_frame_range {start: %d, end: "
                             "%d}, then bake. Particle systems: %s. NOTHING was baked."
                             % (sc.frame_start, sc.frame_end, lo, hi, min(lo, sc.frame_start),
                                max(hi, sc.frame_end), ", ".join(particles)))
        for _label, holder, kind in caches_now:
            if kind != "particles":
                _write_range(holder, start, end)
        # ONLY WHEN IT DIFFERS: a domain's cache range is not a number beside the bake, it is the bake's own extent,
        # and a caller asking again for the range it already holds must not have that write reach the cache.
        #
        # A write is followed by a tag, or the copy that simulates keeps the old range (ops_common.tag_fluid_domain).
        for _label, ds in _fluid_domains(sc):
            if ((start is not None and int(ds.cache_frame_start) != int(start))
                    or (end is not None and int(ds.cache_frame_end) != int(end))):
                _write_range(ds, start, end, "cache_frame_start", "cache_frame_end")
                tag_fluid_domain(ds)

    if clear:
        held_before = _frames_by_cache(sc)
        try:
            with bpy.context.temp_override(scene=sc, view_layer=bpy.context.view_layer):
                bpy.ops.ptcache.free_bake_all()
            # AND EVERY FLUID DOMAIN'S OWN CACHE - free_bake_all frees point caches, and a Mantaflow domain's frames
            # are files in its cache folder (see _free_fluid_domain for why not fluid.free_all).
            for _label, ds in _fluid_domains(sc):
                _free_fluid_domain(ds)
        except RuntimeError as exc:
            raise MifOpError("could not clear the caches: %s" % exc)
        # AND THE FRAMES A CACHE HOLDS WITHOUT A BAKE. free_bake_all frees BAKES only: a cloth stepped frame by frame
        # into a disk cache kept all 20 frames (measured 2026-10-04 on 5.2.1, free_bake and a frame_end write alike),
        # and the next run replayed them (the garment sim, 2026-10-03). Rewriting the cache's own name marks it
        # outdated, and an outdated cache is emptied when the scene is at its start frame - 20 frames to 0, measured.
        # THE SCENE STAYS AT THAT START FRAME: going back to frame 10 after the clear wrote 2 new frames into the cache
        # it had just emptied (measured, test GD132), so the frame is reported rather than restored.
        stale = []
        for label, holder, kind in _scene_cache_ranges(sc):
            pc = holder.point_cache if kind == "dynamic_paint" else holder
            if pc is not None and not pc.is_baked and (_stored_frames(pc) or 0) > 0:
                pc.name = pc.name
                stale.append((label, pc))
        frame_before = sc.frame_current
        if stale:
            for start in sorted({int(pc.frame_start) for _label, pc in stale}, reverse=True):
                sc.frame_set(start)
        held_after = _frames_by_cache(sc)
        kept = {label: n for label, n in held_after.items() if n}
        out = {"cleared": not kept, "baked": False,
               "framesBefore": held_before, "framesAfter": held_after,
               "unbakedFramesFreed": [label for label, _pc in stale],
               "frame": sc.frame_current, "frameBefore": frame_before}
        if sc.frame_current != frame_before:
            out["frameNote"] = ("the scene was moved to frame %d, where the unbaked caches empty, and left there: "
                                "returning to frame %d would write new frames into them straight away."
                                % (sc.frame_current, frame_before))
        if kept:
            out["warning"] = ("these caches still hold frames after the clear: %s. Read them with physics_info."
                              % ", ".join("%s (%d)" % kv for kv in sorted(kept.items())))
        return out

    # BAKE_ALL KEEPS AN EXISTING BAKE, and anything changed since it was made is not in the result.
    # MEASURED 2026-09-16 on 4.4.0, a clean scene with no addon: a ball falling past a FORCE field moved
    # 36.8 m sideways when the field existed before the bake; bake, THEN add the field, bake_all again, and
    # it moved 0.0 - the old frames were kept whole; free first, and it moved 36.8 again (WIND: 39.0 / 0.0 /
    # 39.0). The second seat's five-field scene came back "byte-identical to a no-field control" this way,
    # and read as rigid bodies ignoring fields - which they do not. So which caches were already baked is
    # recorded BEFORE bake_all, reported as keptExistingBake, and `rebake` frees them first. Not freed by
    # default: an existing bake can be minutes of fluid or cloth a caller means to keep.
    already = []
    for label, holder, kind in _scene_cache_ranges(sc):
        pc = holder.point_cache if kind == "dynamic_paint" else holder
        if pc is not None and getattr(pc, "is_baked", False):
            already.append(label)

    # AND FREEING IS NOT ENOUGH, which is the second half of the same measurement. Once bake_all has run
    # over an existing bake AFTER a field was added, the field stays out of every later bake - free and
    # bake again and the ball still moves 0.0 m. Bisected on 3.6.23, 4.4.0 and 5.2.1, FORCE and WIND alike:
    # frame changes are harmless, the kept bake_all is what does it, and of six recoveries tried (toggling
    # the world, touching the cache range, view_layer.update, depsgraph.update, a frame_set to the start,
    # writing a field setting) ONLY writing a setting on the field itself brings it back - 36.8 m again.
    # So every field in the scene has a setting written to itself before any bake that starts from
    # nothing - a rebake, or a first bake after a clear - and a bake over an existing one is left alone.
    fields_refreshed = []
    if rebake or not already:
        for ob in sc.objects:
            fld = getattr(ob, "field", None)
            if fld is not None and fld.type != "NONE":
                fld.strength = fld.strength
                fields_refreshed.append(ob.name)

    if rebake and already:
        try:
            with bpy.context.temp_override(scene=sc, view_layer=bpy.context.view_layer):
                bpy.ops.ptcache.free_bake_all()
        except RuntimeError as exc:
            raise MifOpError("rebake could not free the existing bake first: %s. NOTHING was baked." % exc)

    # THE RIGID BODY WORLD HAS THE SAME TRAP. MEASURED 2026-09-16, identical on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and
    # 5.2.1: bodies baked, a cube C added, then a plain bake_all over the kept bake - and from then on C stayed at its
    # start through bake_physics {rebake: true} (free_bake_all, bake_all). Of five recoveries written before that
    # rebake, frame_set to the start and rewriting the cache's frame_end left it frozen; writing a setting on C, on
    # every body, or on the WORLD (substeps_per_frame onto itself) dropped it to z 0.37 by frame 20 as a fresh scene
    # does. The world is the one write that reaches every body, so it is made before any bake starting from nothing -
    # and AFTER the free, so a refused rebake has written nothing: free, write, bake measured the same 0.37 on all
    # five builds, and free then bake with no write left C at 3.0.
    # A REBAKE FREES EVERY FLUID DOMAIN, REPLAY INCLUDED, whether or not a point cache was baked. The frames on disk are
    # what a domain's "complete" is judged by, and they outlive the settings that made them: a showcase rebuilt with a
    # new time_scale and resolution pointed its new domains at its first build's folders (cache_directory's update frees
    # nothing - rna_Fluid_update), judged the 360 frames there complete, and replayed the old fire (measured 2026-09-21).
    if rebake:
        for _label, ds in _fluid_domains(sc):
            _free_fluid_domain(ds)

    rigid_world_refreshed = False
    if rebake or not already:
        rbw_now = sc.rigidbody_world
        if rbw_now is not None and getattr(rbw_now, "substeps_per_frame", None) is not None:
            rbw_now.substeps_per_frame = rbw_now.substeps_per_frame
            rigid_world_refreshed = True

    # THE FLUID DOMAINS FIRST, each over its own range. The point-cache pass below steps from frame 1: run first, it
    # wrote a REPLAY domain's frames from 1 with no pre-roll behind them, and the domain's own pass then read those
    # frames back instead of simulating them - a fire that restarts at frame 1. Finished first, a domain holds its
    # whole range and the point-cache pass only reads it.
    fluid_baked = _bake_fluid_domains(sc, rebake)
    try:
        with bpy.context.temp_override(scene=sc, view_layer=bpy.context.view_layer):
            bpy.ops.ptcache.bake_all(bake=True)
    except RuntimeError as exc:
        raise MifOpError("bake failed: %s. Nothing in the scene may have a point cache - a rigid "
                         "body world, cloth or softbody is needed before there is anything to "
                         "bake." % exc)

    # A DYNAMIC-PAINT SURFACE IS NOT BAKED BY bake_all, AND ITS ZERO-FRAME "BAKE" FREEZES IT. Measured
    # 2026-09-12 in a throwaway Blender with no addon: a WAVE canvas simulates when its frames are stepped
    # in order, but bake_all left its cache '0 frames on disk' marked baked, and the canvas then showed one
    # frozen state at every frame. So a zero-frame bake is freed; a real one stays.
    #
    # AND LEFT LIVE IT RENDERS FLAT - this used to say a live surface animates in render_animation, and it does not
    # render at all. MEASURED 2026-09-16, identical on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1 (the second seat said so
    # in mailbox id 52, and was right): a DISPLACE canvas with a brush dragged across frames 1-30, stepped in order -
    # 0.40 BU of relief in the session - rendered at frame 30 like a grid with no paint at all (0.0-0.0002/255 mean
    # difference, where a real 0.4 BU bump is 1.0); ptcache.bake with the cache in memory stored "0 frames in memory";
    # with the file saved and the cache on disk it stored "30 frames on disk" and rendered the relief (2.5-2.7/255).
    # A render re-evaluates the canvas, and only a disk cache gives it frames. So with the file saved each surface is
    # baked to its own disk cache here; unsaved, it is named, with why.
    dp_live, dp_baked, dp_failed = [], [], []
    saved = bool(bpy.data.filepath)
    for label, holder, kind in _scene_cache_ranges(sc):
        pc = holder.point_cache if kind == "dynamic_paint" else None
        if pc is None:
            continue
        if pc.is_baked and (_stored_frames(pc) or 0) > 0:
            # A real bake - kept, or made just now by bake_all, which does store a disk cache once the file is saved.
            dp_baked.append(label)
            continue
        if pc.is_baked:
            try:
                with bpy.context.temp_override(scene=sc, point_cache=pc):
                    bpy.ops.ptcache.free_bake()
            except RuntimeError as exc:
                raise MifOpError("%s baked ZERO frames, which freezes a dynamic-paint canvas at one state, and "
                                 "freeing that bake failed: %s" % (label, exc))
        if not saved:
            dp_live.append(label)
            continue
        owner = holder.id_data
        pc.use_disk_cache = True
        try:
            with bpy.context.temp_override(scene=sc, point_cache=pc, object=owner, active_object=owner):
                bpy.ops.ptcache.bake(bake=True)
        except RuntimeError as exc:
            dp_failed.append({"surface": label, "error": str(exc)[:200]})
        if pc.is_baked and (_stored_frames(pc) or 0) > 0:
            dp_baked.append(label)
        else:
            dp_live.append(label)

    held, missed = _ranges_held(sc, start, end) if asked_range else ([], [])

    # READ BACK which caches actually hold frames. bake_all reports FINISHED with nothing to do.
    caches = []
    rbw = sc.rigidbody_world
    if rbw and rbw.point_cache:
        caches.append({"kind": "rigidbody", "isBaked": bool(rbw.point_cache.is_baked),
                       "frames": [rbw.point_cache.frame_start, rbw.point_cache.frame_end],
                       "framesStored": _stored_frames(rbw.point_cache)})
    for ob in bpy.data.objects:
        for m in ob.modifiers:
            pc = getattr(getattr(m, "point_cache", None), "is_baked", None)
            if pc is not None:
                caches.append({"kind": m.type.lower(), "object": ob.name, "isBaked": bool(pc),
                               "frames": [m.point_cache.frame_start, m.point_cache.frame_end],
                               "framesStored": _stored_frames(m.point_cache)})
        # PARTICLE CACHES ARE NOT ON THE MODIFIER, which is why they were missing from this report
        # entirely. A ParticleSystem modifier has no `point_cache`; the cache hangs off the SYSTEM
        # at ob.particle_systems[i].point_cache. The loop above therefore skipped every particle
        # sim in the scene and the answer looked like "only the rigid body was baked".
        #
        # Found 2026-09-05: five EMITTER/NEWTON systems were created, bake_physics returned
        # baked:true with cacheCount 1 and kind "rigidbody", and I concluded the particles had not
        # been baked. ptcache.bake_all does bake them - what was broken was this readback, so the
        # op under-reported its own work and there was no way to tell that from the outside.
        for ps in getattr(ob, "particle_systems", ()):
            pc = getattr(ps, "point_cache", None)
            # A FLUID DOMAIN'S OWN PARTICLE SYSTEMS ARE NOT POINT CACHES. A LIQUID domain carries a "Liquid" system of
            # type FLIP (render HALO) - measured on 3.6.23, 4.2.17 and 5.2.1 - which Mantaflow fills from its own cache;
            # its point cache is never baked, so it read isBaked:false in every liquid scene and held allBaked false
            # whatever the simulation did. It is reported on the fluid row below instead.
            if getattr(getattr(ps, "settings", None), "type", "EMITTER") not in ("EMITTER", "HAIR"):
                continue
            if pc is not None:
                caches.append({"kind": "particles", "object": ob.name, "system": ps.name,
                               "isBaked": bool(pc.is_baked),
                               "frames": [pc.frame_start, pc.frame_end],
                               # WHAT IT STORES, which a particle cache's range can misstate.
                               "framesStored": _stored_frames(pc)})
        # DYNAMIC-PAINT SURFACES, for the reason particles were added above: their caches hang off the
        # canvas SURFACE, not the modifier, so the loop over modifiers never reported one.
        for m in ob.modifiers:
            canvas = getattr(m, "canvas_settings", None) if m.type == "DYNAMIC_PAINT" else None
            for surf in (canvas.canvas_surfaces if canvas is not None else ()):
                spc = getattr(surf, "point_cache", None)
                caches.append({"kind": "dynamic_paint", "object": ob.name, "surface": surf.name,
                               "isBaked": bool(spc.is_baked) if spc is not None else None,
                               "frames": [surf.frame_start, surf.frame_end],
                               "framesStored": _stored_frames(spc) if spc is not None else None})
    # KEPT: point caches baked before this call, and fluid domains whose folder already held their range.
    kept_all = already + ["%s:fluid" % r["object"] for r in fluid_baked if r.get("kept")]
    fluid_rows = [_fluid_domain_row(ob, m) for ob in bpy.data.objects for m in ob.modifiers
                  if m.type == "FLUID" and getattr(m, "fluid_type", None) == "DOMAIN"]
    caches.extend(fluid_rows)
    fluid_empty = [r["object"] for r in fluid_rows if not r["framesStored"]]
    return {"baked": True, "caches": caches, "cacheCount": len(caches),
            # `baked` HAS ALWAYS MEANT "bake_all ran", and it read true beside a dynamic paint surface left live, a
            # cache holding the wrong range, or a scene with nothing to bake. allBaked is the claim a caller wants
            # before render_animation: at least one cache, every cache baked, and no dynamic paint surface left live.
            "allBaked": bool(caches) and all(c.get("isBaked") is True for c in caches) and not dp_live and not dp_failed,
            # WHICH caches HOLD the range asked for, read back after the bake - not which ones it was
            # written to, which is what this listed when a particle cache was baking 1-250 under it.
            "rangeAppliedTo": [h[0] for h in held] if asked_range else None,
            "rangeNotHeld": ([{"cache": m[0], "asked": m[1], "holds": m[2], "framesStored": m[3],
                                "why": ("the cache READS this range and stores only %s frame(s) - "
                                        "bake_all does not re-bake one that is already baked. Clear "
                                        "it first: bake_physics {clear: true}." % m[3])
                                       if m[1] == m[2] else
                                       "the cache holds a different range from the one asked for"}
                               for m in missed] if asked_range else None),
            "warning": ("%d cache(s) hold a range other than the one asked for and simulate only over "
                        "what they hold (a particle system always bakes the scene's frame range): %s"
                        % (len(missed), ", ".join(m[0] for m in missed))) if missed else None,
            # DYNAMIC-PAINT SURFACES - see the measurement above the bake. Baked: on their own disk caches. Live: they
            # simulate as frames are stepped in THIS session, and a render - still or animation - shows them flat.
            # HOW EACH MANTAFLOW DOMAIN WAS FINISHED - see _bake_fluid_domains. how is null for one that already held
            # its range.
            "fluidBaked": fluid_baked or None,
            "fluidNote": (("no simulated frames are stored for %s, so it renders at rest - its cache folder holds none "
                           "of its range after the bake. Check its cacheDirectory is writable, and that a flow object "
                           "emits into it." % ", ".join(fluid_empty)) if fluid_empty else None),
            "dynamicPaintBaked": dp_baked or None,
            "dynamicPaintLive": dp_live or None,
            "dynamicPaintBakeFailed": dp_failed or None,
            "dynamicPaintNote": (
                ("%d dynamic paint surface(s) are NOT baked - %s - and a surface that is not baked renders FLAT, however "
                 "its frames are stepped (measured on all five builds: 0.4 BU of relief in the session, none in the "
                 "render). %s"
                 % (len(dp_live), ", ".join(dp_live),
                    "A dynamic paint surface bakes only to a disk cache kept beside a saved .blend: save_file "
                    "(repointSession), then bake_physics again." if not saved else
                    "The file is saved and baking to disk still stored no frames - see dynamicPaintBakeFailed."))
                if dp_live else None),
            # WHAT THIS CALL DID NOT RE-SIMULATE - see the measurement above the bake.
            # A FLUID DOMAIN THAT ALREADY HELD ITS RANGE IS KEPT TOO, and listed with the point caches - its frames can
            # come from other settings (see _bake_fluid_domains).
            "keptExistingBake": (kept_all or None) if not rebake else None,
            "freedBeforeBake": (already or None) if rebake else None,
            # The force fields whose effector link was re-established before this bake - see the second
            # measurement above the bake. Empty when there are none, None when the bake kept an old one.
            "fieldsRefreshed": fields_refreshed if (rebake or not already) else None,
            # Whether the rigid body world had a setting rewritten before this bake - see the measurement above it.
            "rigidBodyWorldRefreshed": rigid_world_refreshed if (rebake or not already) else None,
            "staleBakeNote": ("%d cache(s) were ALREADY baked and bake_all keeps an existing bake as it is: %s. "
                              "Anything changed since that bake - a force field added, a setting edited, an "
                              "object moved - is NOT in these frames. Measured: a field added after a bake "
                              "moved a rigid body 0.0 m on a re-bake and 36.8 m after freeing. Pass "
                              "rebake: true to free and bake again." % (len(kept_all), ", ".join(kept_all)))
                             if kept_all and not rebake else None,
            "emptyNote": (None if caches else
                          "bake_all reported success and NOTHING has a point cache - there was no "
                          "simulation to bake. This is a no-op reported as a success by Blender.")}


_COLLECT_KEYS = {"object", "folder", "prefix", "destination", "start", "end", "baseName", "digits", "firstNumber",
                 "sampleRanges"}
# "name_0012.vdb", "name_-0003.vdb" (Mantaflow's pre-roll), "name.0012.vdb": the stem, a minus sign right before the
# digits, and the digits - the same "digits after the last non-digit" Unreal's importer reads, with the sign kept.
_VDB_NUMBERED = re.compile(r"^(?P<stem>.*?)(?P<sign>-?)(?P<digits>\d+)\.vdb$", re.IGNORECASE)


def _vdb_sequences(folder):
    """{prefix: {frame: file name}} for the numbered .vdb files in a folder, and [(name, frame, prefix)] for files
    two of which claim one frame. A minus sign right before the digits is a NEGATIVE frame (Mantaflow's pre-roll)
    only where the same prefix also has unsigned files; where every file has it, it is a separator ("fire-0001")."""
    raw = {}
    for name in sorted(os.listdir(folder)):
        m = _VDB_NUMBERED.match(name)
        if m:
            raw.setdefault(m.group("stem"), []).append((name, m.group("sign"), int(m.group("digits"))))
    out, clashes = {}, []
    for stem, rows in raw.items():
        signs = {sign for _n, sign, _d in rows}
        separator = signs == {"-"}
        prefix = stem + "-" if separator else stem
        frames = out.setdefault(prefix, {})
        for name, sign, digits in rows:
            frame = -digits if (sign and not separator) else digits
            if frame in frames:
                clashes.append((name, frame, prefix))
            frames[frame] = name
    return out, clashes


def _vdb_grids_blender(path):
    """The grids of one .vdb as Blender's own reader sees them: [{name, dataType, channels, voxelSize}], or raises.
    A Volume datablock is made, read and removed - nothing is left in the file."""
    vol = bpy.data.volumes.new("mif_vdb_probe")
    try:
        vol.filepath = path
        vol.grids.load()
        if getattr(vol.grids, "error_message", ""):
            raise MifOpError("Blender could not read %s: %s" % (path, vol.grids.error_message))
        rows = []
        for g in vol.grids:
            row = {"name": g.name, "dataType": g.data_type, "channels": int(g.channels)}
            try:
                m = g.matrix_object
                row["voxelSize"] = [round(m.col[i].xyz.length, 6) for i in range(3)]
            except Exception:  # noqa: BLE001 - a grid without a transform is still a grid worth listing
                row["voxelSize"] = None
            rows.append(row)
        return rows
    finally:
        bpy.data.volumes.remove(vol)


def _vdb_value_ranges(paths):
    """Per file, per grid: min, max and active voxel count, through Blender's bundled openvdb module (5.x ships it;
    older builds may not). ({file: [...]}, None) or (None, why)."""
    mod = None
    for modname in ("openvdb", "pyopenvdb"):
        try:
            mod = __import__(modname)
            break
        except Exception:  # noqa: BLE001 - absent module is an answer, not a failure
            continue
    if mod is None:
        return None, ("this Blender has no openvdb Python module, so value ranges cannot be read here; the grids "
                      "are still listed")
    out = {}
    for path in paths:
        grids, _meta = mod.readAll(path)
        rows = []
        for g in grids:
            row = {"name": g.name, "activeVoxels": int(g.activeVoxelCount())}
            try:
                lo, hi = g.evalMinMax()
                if isinstance(lo, (tuple, list)):
                    # A VECTOR GRID'S evalMinMax IS BY LENGTH, not per component: a velocity grid came back with
                    # min z 24.1 above max z 12.5 (measured on 5.2.1). Only the largest magnitude is meaningful.
                    row["maxMagnitude"] = round(math.sqrt(sum(float(c) * float(c) for c in hi)), 6)
                else:
                    row["min"], row["max"] = lo, hi
                    row["constant"] = lo == hi
            except Exception as exc:  # noqa: BLE001 - reported per grid
                row["error"] = str(exc)[:160]
            rows.append(row)
        out[os.path.basename(path)] = rows
    return out, None


def op_collect_vdb_frames(params):
    """Copy the frames of a baked OpenVDB sequence into a clean folder, renumbered from 0 - COPIED, never moved.

    WHY IT EXISTS. Unreal's OpenVDB importer reads a frame number as the digits after the last non-digit, takes every
    .vdb whose name starts with the chosen file's base name, and stops at the first gap. A Mantaflow cache breaks all
    three: a domain whose cache starts before frame 1 writes its pre-roll as fluid_data_-0001.vdb, which the importer
    reads as frame 1 beside fluid_data_0001.vdb. Measured 2026-09-22 on 5.8.2: a cache of frames -9..40 (50 files)
    imported as 41 frames and nothing said nine were dropped. So the frames to keep are copied to a folder of their
    own, named <baseName>_0000.vdb upward, which every importer reads the same way.

    WHY HERE AND NOT (ONLY) IN UNREAL. The frames come from Blender, and the domain knows which of them are pre-roll -
    its cache range against the scene's - so the caller names an object instead of working out Mantaflow's file
    layout and sign convention. Unreal's import_vdb_as_volume_texture does the other half: it audits a folder with
    the importer's own rule and refuses one it would read wrongly, whatever tool wrote it. One copies, one checks;
    neither does the other's job.

    NOTHING IS DELETED OR OVERWRITTEN: a destination that already holds a .vdb the new sequence would absorb is
    refused rather than cleared.
    """
    reject_unknown(params, _COLLECT_KEYS, "collect_vdb_frames", {
        "move": "frames are only ever COPIED - the cache stays as it was, so a re-bake or a second export still has it.",
        "overwrite": "not offered: a destination holding .vdb files the new sequence would absorb is refused, never "
                     "cleared. Pick an empty folder.",
        "name": "spelt baseName - it names the files, and through them the Unreal asset (fire -> fire_0000.vdb -> fire).",
        "frames": "spelt start and end - the SOURCE frame numbers to keep, inclusive.",
    })
    obj_name = take(params, "object", default=None, kind=str)
    folder = take(params, "folder", default=None, kind=str)
    if bool(obj_name) == bool(folder):
        raise MifOpError("pass exactly one of object (a fluid DOMAIN, whose cache folder is found for you) or folder "
                         "(any folder of .vdb files). NOTHING was copied.")
    dest = take(params, "destination", required=True, kind=str)
    start = take_int(params, "start", default=None)
    end = take_int(params, "end", default=None)
    base = take(params, "baseName", default=None, kind=str)
    digits = take_int(params, "digits", default=4)
    first_number = take_int(params, "firstNumber", default=0)
    sample = take_bool(params, "sampleRanges", default=True)
    want_prefix = take(params, "prefix", default=None, kind=str)
    if digits < 1 or digits > 9:
        raise MifOpError("digits must be 1..9, got %d. NOTHING was copied." % digits)
    if first_number < 0:
        raise MifOpError("firstNumber must not be negative - a minus sign is exactly what Unreal's importer drops. "
                         "NOTHING was copied.")

    source = {}
    if obj_name:
        obj = get_object(obj_name)
        mod = next((m for m in obj.modifiers if m.type == "FLUID" and getattr(m, "fluid_type", None) == "DOMAIN"), None)
        if mod is None:
            raise MifOpError("'%s' is not a fluid DOMAIN - the frames live in a domain's cache folder. NOTHING was "
                             "copied." % obj.name)
        ds = mod.domain_settings
        if getattr(ds, "cache_data_format", "OPENVDB") != "OPENVDB":
            raise MifOpError("'%s' caches its data as %s, not OpenVDB - set cache_data_format to OPENVDB (bl_set_modifier) "
                             "and bake again. NOTHING was copied." % (obj.name, ds.cache_data_format))
        root = bpy.path.abspath(ds.cache_directory) if ds.cache_directory else ""
        src_folder = os.path.join(root, "data") if root else ""
        if not src_folder or not os.path.isdir(src_folder):
            raise MifOpError("'%s' has no data folder at %r - bake it first (bake_physics). NOTHING was copied."
                             % (obj.name, src_folder))
        sc = bpy.context.scene
        default_lo = max(int(sc.frame_start), int(ds.cache_frame_start))
        default_hi = int(ds.cache_frame_end)
        want_prefix = want_prefix or "fluid_data_"
        source.update({"kind": "domain", "object": obj.name, "cacheRange": [int(ds.cache_frame_start), default_hi],
                       "sceneStart": int(sc.frame_start)})
        default_base = re.sub(r"[^A-Za-z0-9]+", "_", obj.name).strip("_") or "volume"
    else:
        src_folder = bpy.path.abspath(folder)
        if not os.path.isdir(src_folder):
            raise MifOpError("no folder at %r. NOTHING was copied." % src_folder)
        default_lo, default_hi = 0, None
        source["kind"] = "folder"
        default_base = None

    sequences, clashes = _vdb_sequences(src_folder)
    if not sequences:
        raise MifOpError("%s holds no numbered .vdb files. NOTHING was copied." % src_folder)
    if want_prefix is None:
        if len(sequences) > 1:
            raise MifOpError("%s holds %d sequences (%s) - pass prefix to choose one. NOTHING was copied."
                             % (src_folder, len(sequences), ", ".join(sorted(sequences))))
        want_prefix = next(iter(sequences))
    if want_prefix not in sequences:
        raise MifOpError("no sequence with prefix %r in %s - it has %s. NOTHING was copied."
                         % (want_prefix, src_folder, ", ".join(sorted(sequences))))
    frames = sequences[want_prefix]
    mine_clashes = [c for c in clashes if c[2] == want_prefix]
    if mine_clashes:
        raise MifOpError("%d files in %s claim a frame another file claims (%s) - the source itself is ambiguous. "
                         "NOTHING was copied." % (len(mine_clashes), src_folder, mine_clashes[0][0]))
    on_disk = sorted(frames)
    lo = start if start is not None else default_lo
    hi = end if end is not None else (default_hi if default_hi is not None else on_disk[-1])
    if hi < lo:
        raise MifOpError("end %d is before start %d. NOTHING was copied." % (hi, lo))
    keep = [f for f in on_disk if lo <= f <= hi]
    if not keep:
        raise MifOpError("no frame of %s lies in %d..%d - on disk: %d..%d. NOTHING was copied."
                         % (want_prefix, lo, hi, on_disk[0], on_disk[-1]))
    missing = [f for f in range(keep[0], keep[-1] + 1) if f not in frames]
    if missing:
        raise MifOpError("frames %s are missing between %d and %d. Unreal's importer stops at the first gap, and "
                         "renumbering past one would hide it - bake the missing frames or narrow start/end. NOTHING "
                         "was copied." % (missing[:8], keep[0], keep[-1]))
    if keep[0] > lo or keep[-1] < hi:
        source["rangeNote"] = ("asked %d..%d, and the files on disk cover %d..%d of it - those are what was copied"
                               % (lo, hi, keep[0], keep[-1]))

    base = base if base is not None else (default_base or re.sub(r"[_.\-]+$", "", want_prefix) or "volume")
    if not base or base[-1].isdigit():
        raise MifOpError("baseName %r must not be empty or end in a digit - Unreal reads a trailing number as part of "
                         "the frame number. NOTHING was copied." % base)
    if not re.match(r"^[A-Za-z0-9_\-.]+$", base):
        raise MifOpError("baseName %r may hold only letters, digits, '_', '-' and '.'. NOTHING was copied." % base)
    stem = base if base[-1] in "_-." else base + "_"
    last_number = first_number + len(keep) - 1
    if len(str(last_number)) > digits:
        raise MifOpError("%d frames from %d need %d digits, not %d. NOTHING was copied."
                         % (len(keep), first_number, len(str(last_number)), digits))

    dest_abs = os.path.abspath(bpy.path.abspath(dest))
    if os.path.normcase(dest_abs) == os.path.normcase(os.path.abspath(src_folder)):
        raise MifOpError("destination is the source folder itself - the point is a folder holding nothing but the "
                         "kept frames. NOTHING was copied.")
    if os.path.isdir(dest_abs):
        # Unreal admits every .vdb whose name STARTS WITH the base name, case-insensitively, so one left behind -
        # an earlier export, or fire_smoke_0001.vdb beside fire_0000.vdb - would join this sequence.
        clash = [n for n in os.listdir(dest_abs) if n.lower().endswith(".vdb") and n.lower().startswith(stem.lower())]
        if clash:
            raise MifOpError("%s already holds %d .vdb file(s) starting with %r (%s) that Unreal would read as part of "
                             "this sequence - choose an empty folder or another baseName. NOTHING was copied."
                             % (dest_abs, len(clash), stem, clash[0]))
    else:
        os.makedirs(dest_abs)

    copied, total = [], 0
    for i, f in enumerate(keep):
        src = os.path.join(src_folder, frames[f])
        dst = os.path.join(dest_abs, "%s%0*d.vdb" % (stem, digits, first_number + i))
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            raise MifOpError("copying %s failed after %d of %d frames: %s. The %d already copied are in %s; the source "
                             "is untouched." % (frames[f], len(copied), len(keep), exc, len(copied), dest_abs))
        size = os.path.getsize(dst)
        if size != os.path.getsize(src):
            raise MifOpError("%s copied short (%d of %d bytes). The source is untouched."
                             % (dst, size, os.path.getsize(src)))
        total += size
        copied.append(dst)

    grids = _vdb_grids_blender(copied[0])
    ranges, ranges_note = (None, "sampleRanges was false")
    if sample:
        probe = [copied[0], copied[len(copied) // 2], copied[-1]]
        ranges, ranges_note = _vdb_value_ranges(list(dict.fromkeys(probe)))
    names = {g["name"] for g in grids}
    hint = None
    if "density" in names:
        hint = {"A": {"format": "float16", "r": "density", "g": "temperature" if "temperature" in names else "",
                      "b": "", "a": "flame" if "flame" in names else ""}}
    skipped_before = [f for f in on_disk if f < keep[0]]
    skipped_after = [f for f in on_disk if f > keep[-1]]
    return {"source": dict(source, folder=src_folder, prefix=want_prefix,
                           framesOnDisk=[on_disk[0], on_disk[-1], len(on_disk)]),
            "destination": dest_abs, "baseName": stem, "copied": len(copied),
            "sourceFrames": [keep[0], keep[-1]], "frames": [first_number, last_number],
            "firstFile": os.path.basename(copied[0]), "lastFile": os.path.basename(copied[-1]), "bytes": total,
            "skipped": {"before": [skipped_before[0], skipped_before[-1], len(skipped_before)] if skipped_before else None,
                        "negativeBefore": len([f for f in skipped_before if f < 0]),
                        "after": [skipped_after[0], skipped_after[-1], len(skipped_after)] if skipped_after else None},
            "grids": grids, "valueRanges": ranges, "valueRangesNote": ranges_note,
            "ueImport": {"file": copied[0].replace("\\", "/"), "assetName": re.sub(r"[_.\-]+$", "", stem),
                         "channelMap": hint},
            "note": "copied, never moved - the source frames are untouched"}


_INFO_KEYS = {"object", "name", "limit", "nameExcludes"}

# The modifier types that carry a simulation. Named explicitly rather than detected by the presence
# of a point_cache, because DYNAMIC_PAINT has one and is not a physics sim in the sense meant here,
# and a list is readable where a duck-type is not.
_SIM_MODIFIERS = ("CLOTH", "SOFT_BODY", "COLLISION", "FLUID", "DYNAMIC_PAINT")

# THE READ HALF FOR THE SIMULATION FAMILIES, added 2026-09-15. physics_info has always reported a sim
# modifier's NAME, TYPE and VISIBILITY and never one of its settings - so add_cloth, add_soft_body,
# add_collision and add_rigid_body_constraint could each be configured and then not asked. Their own
# responses do read the stored values back off the datablock, honestly; what was missing is a SECOND
# way in, the same gap set_light_influence had and for the same cost: an agent that reconnects, or
# inherits a file somebody else built, had to remember what it set.
#
# RIGID BODY WAS THE ONE FAMILY WITH A READ HALF, which is why that one is not in this table - it is
# reported above from obj.rigid_body, margin gate and all.
#
# WHERE THE SETTINGS LIVE IS NOT UNIFORM, and that is the whole reason this is a table rather than a
# loop over mod.settings. Cloth splits across mod.settings and mod.collision_settings; soft body is
# not on the modifier at all but on obj.soft_body; the constraint is on obj.rigid_body_constraint and
# has no modifier. Each row is (jsonKey, attribute, kind), and the holder is resolved per family by
# _sim_holders below.
_SIM_READBACK = {
    "CLOTH": (
        ("settings", (("quality", "quality", int),
                      ("mass", "mass", float),
                      ("usePressure", "use_pressure", bool),
                      ("uniformPressureForce", "uniform_pressure_force", float),
                      # add_cloth's pinGroup / pinStiffness, readable from a second op like the rest.
                      ("pinGroup", "vertex_group_mass", str),
                      ("pinStiffness", "pin_stiffness", float),
                      # add_cloth's stiffness and damping, which nothing read back until 2026-09-16 (mailbox ids 57, 61).
                      ("tensionStiffness", "tension_stiffness", float),
                      ("compressionStiffness", "compression_stiffness", float),
                      ("shearStiffness", "shear_stiffness", float),
                      ("tensionDamping", "tension_damping", float),
                      ("compressionDamping", "compression_damping", float),
                      ("shearDamping", "shear_damping", float),
                      # add_cloth's bending and pressure-by-region keys (2026-10-06): bending decides whether a hood
                      # stands, and the regional scale on pressure is the only per-region pressure Blender has.
                      ("bending", "bending_stiffness", float),
                      ("bendingDamping", "bending_damping", float),
                      ("bendingModel", "bending_model", str),
                      ("bendingGroup", "vertex_group_bending", str),
                      ("bendingMax", "bending_stiffness_max", float),
                      ("airDamping", "air_damping", float),
                      ("shrinkMin", "shrink_min", float),
                      ("pressureGroup", "vertex_group_pressure", str),
                      ("pressureFactor", "pressure_factor", float))),
        ("settings.effector_weights", (("gravityWeight", "gravity", float),)),
        ("collision_settings", (("collisionQuality", "collision_quality", int),
                                ("selfCollision", "use_self_collision", bool))),
    ),
    "COLLISION": (
        ("settings", (("damping", "damping", float),
                      ("clothFriction", "cloth_friction", float),
                      ("thicknessOuter", "thickness_outer", float))),
    ),
    # FLUID SPLITS BY ROLE, and all three holders are listed because only one of them exists at a
    # time: a DOMAIN modifier has domain_settings and None for the other two, a FLOW has
    # flow_settings. _sim_holders skips a holder that is None, so listing all three reads whichever
    # role this modifier actually is without asking fluid_type first - and a fluid_type of NONE
    # yields nothing at all rather than an empty block that would read as "a domain with nothing
    # set". surfaceDistance appears under both FLOW and EFFECTOR because Blender keeps a separate
    # one on each, and only the live holder contributes.
    "FLUID": (
        ("@modifier", (("fluidType", "fluid_type", str),)),
        # useMesh ADDED 2026-09-21 (tools/audit_value_without_gate.py): mesh generation decides whether a LIQUID
        # domain renders a surface at all, and a new domain starts with it OFF on 3.6. add_fluid turns it on and
        # says so in its own reply, but nothing READ it afterwards - an agent that reconnects, or inherits a file,
        # could not tell a liquid that renders from one that simulates and shows nothing.
        ("domain_settings", (("domainType", "domain_type", str),
                             ("resolutionMax", "resolution_max", int),
                             ("cacheFrameStart", "cache_frame_start", int),
                             ("cacheFrameEnd", "cache_frame_end", int),
                             ("useAdaptiveDomain", "use_adaptive_domain", bool),
                             ("useMesh", "use_mesh", bool))),
        ("flow_settings", (("flowType", "flow_type", str),
                           ("flowBehavior", "flow_behavior", str),
                           ("useInflow", "use_inflow", bool),
                           ("surfaceDistance", "surface_distance", float),
                           ("subframes", "subframes", int))),
        ("effector_settings", (("effectorType", "effector_type", str),
                               ("useEffector", "use_effector", bool),
                               ("surfaceDistance", "surface_distance", float))),
    ),
    # DYNAMIC PAINT IS THE DEEPEST OF THE FAMILIES. The modifier carries exactly one writable
    # property, ui_type, and everything real is on brush_settings or on a SURFACE under
    # canvas_settings - and the surfaces are a collection, so the canvas state is three hops down.
    # A modifier is a brush or a canvas, never both, so the absent side simply yields nothing.
    #
    # ui_type IS REPORTED AND IS NOT THE ANSWER TO "which is this". op_add_dynamic_paint says so
    # where it measures the same thing: "paintType IS MEASURED FROM THE SETTINGS THAT EXIST, not
    # read off ui_type. ui_type is only the tab the Properties panel shows" - type_toggle can create
    # brush_settings and leave ui_type alone. So paintType is derived from which struct exists, the
    # same way that op derives it, and ui_type is reported beside it as the separate fact it is.
    "DYNAMIC_PAINT": (
        ("@modifier", (("uiType", "ui_type", str),)),
        ("brush_settings", (("paintSource", "paint_source", str),
                            ("paintColor", "paint_color", list),
                            ("paintAlpha", "paint_alpha", float),
                            ("paintWetness", "paint_wetness", float),
                            ("paintDistance", "paint_distance", float),
                            ("useAbsoluteAlpha", "use_absolute_alpha", bool))),
        ("canvas_settings.canvas_surfaces.active", (("surfaceName", "name", str),
                                                    ("surfaceType", "surface_type", str),
                                                    ("surfaceFormat", "surface_format", str),
                                                    ("frameStart", "frame_start", int),
                                                    ("frameEnd", "frame_end", int),
                                                    ("useDissolve", "use_dissolve", bool),
                                                    ("dissolveSpeed", "dissolve_speed", int))),
    ),
    "SOFT_BODY": (
        # obj.soft_body, NOT mod.settings - a SOFT_BODY modifier carries no settings of its own.
        ("@object.soft_body", (("useGoal", "use_goal", bool),
                               ("goalDefault", "goal_default", float),
                               ("goalSpring", "goal_spring", float),
                               ("goalFriction", "goal_friction", float),
                               ("goalMin", "goal_min", float),
                               ("goalMax", "goal_max", float),
                               ("vertexGroupGoal", "vertex_group_goal", str),
                               ("useEdges", "use_edges", bool),
                               ("selfCollision", "use_self_collision", bool))),
    ),
}


def _sim_holders(obj, mod):
    """(label, holder, rows) for each settings struct a sim modifier keeps its state in.

    A holder that does not exist is SKIPPED rather than reported empty - a fluid modifier with
    fluid_type NONE genuinely has no domain_settings, and an empty dict there would read as "a
    domain with nothing set" instead of "not a domain".
    """
    for attr, rows in _SIM_READBACK.get(str(mod.type), ()):
        if attr == "@modifier":
            # Some state lives on the modifier itself rather than in a settings struct -
            # mod.fluid_type is the one that decides which of the others even exists.
            holder, label = mod, "modifier"
        else:
            # A DOTTED PATH, walked one hop at a time and abandoned the moment a hop is None.
            # Dynamic paint needs three: canvas_settings.canvas_surfaces.active, because a canvas
            # keeps its state on a SURFACE and the surfaces are a collection. Walking rather than
            # special-casing it means the next family that hides state one level deeper costs a
            # table row instead of another branch here.
            root = obj if attr.startswith("@object.") else mod
            path = attr[len("@object."):] if attr.startswith("@object.") else attr
            holder, label = root, path
            for hop in path.split("."):
                holder = getattr(holder, hop, None)
                if holder is None:
                    break
        if holder is not None:
            yield label, holder, rows


def _finite_or_none(number):
    """A float safe for a response, or None if it is NaN or Infinity. Takes an ALREADY-CONVERTED
    float, so the conversion stays at the call site where the getattr is visible - see _read_rows.
    """
    return round(number, 6) if math.isfinite(number) else None


def _read_rows(holder, rows):
    """({jsonKey: value}, [keys whose value was not finite]) for the rows this build has.

    Absent properties are omitted, which is different from present-but-not-finite: the first means
    this Blender does not have the property, the second means it does and is holding something JSON
    cannot carry. The caller reports the second so they stay distinguishable.
    """
    # EVERY CONVERSION HAPPENS HERE, at the getattr, rather than inside a helper taking the value as
    # a parameter. audit_unguarded_numbers decides read-back from caller-input by whether the
    # converted expression reaches a bpy attribute, and its own comment notes it cannot follow a
    # value split across two functions - so a helper doing the float() would be a permanent finding
    # in a GATED audit, for code that is a read-back. Keeping the conversion at the call site keeps
    # the audit's rule meaningful instead of papering over its one blind spot.
    #
    # AND THE FINITE CHECK IS MORE THAN THE AUDIT ASKS FOR, deliberately. It reasons that a
    # read-back is "finite and in range by construction", which is true of values THIS ADDON wrote -
    # every write path goes through take_float/finite_floats, which is what that audit enforces. It
    # is not true of a .blend authored somewhere else, or hand-edited in the GUI. json.dumps writes
    # NaN as a bare `NaN`, which is not valid JSON, so one pathological property would make the
    # WHOLE response unparseable to a strict client rather than spoiling one field.
    have = holder.bl_rna.properties.keys()
    out, nonfinite = {}, []
    for key, attr, kind in rows:
        if attr not in have:
            continue
        value = getattr(holder, attr)
        if kind is bool:
            out[key] = bool(value)
        elif kind is int:
            out[key] = int(getattr(holder, attr))
        elif kind is str:
            out[key] = str(value)
        elif kind is list:
            # A COLOUR OR A VECTOR. paint_color is an RNA float array, so the kind is declared
            # rather than sniffed, and the rounding matches every other vector in this file.
            out[key] = [_finite_or_none(float(v)) for v in getattr(holder, attr)]
            if any(v is None for v in out[key]):
                nonfinite.append(key)
        else:
            out[key] = _finite_or_none(float(getattr(holder, attr)))
            if out[key] is None:
                nonfinite.append(key)
    return out, nonfinite


def _cache_row(pc, scene):
    """A point cache described, including whether its range still COVERS the scene's.

    THE STALE-BAKE TRAP. A cache baked before the frame range was extended is baked, valid, and
    short - is_baked stays true, and the frames past its end silently fall back to the rest state.
    Nothing announces it, so the coverage comparison is made here rather than left to a caller who
    would have to know to make it.
    """
    if pc is None:
        return None
    row = {
        "isBaked": bool(pc.is_baked),
        "frameStart": int(pc.frame_start),
        "frameEnd": int(pc.frame_end),
        # WRITTEN BY set_physics_world's cacheStep, and read by nothing here until 2026-09-16 (second seat, id 103).
        "frameStep": int(getattr(pc, "frame_step", 1)),
    }
    row["coversSceneRange"] = (pc.frame_start <= scene.frame_start
                               and pc.frame_end >= scene.frame_end)
    return row


# WHICH COLLIDERS A SIMULATION SEES. FOUND 2026-10-04: a cloth drape fell straight through a body whose collider had
# hide_viewport on, and physics_info said nothing. MEASURED 2026-10-05 on 4.2.23, 4.4.0, 5.0.1 and 5.2.2 - a 6x6 cloth
# dropped 40 frames onto a collision floor: it rests on the floor when the floor is visible, hidden with the eye
# (hide_set), hidden from the render or its layer collection hidden; it falls through when the floor has hide_viewport,
# its collection is hidden in viewports or excluded from the view layer, or the Collision modifier's show_viewport is
# off. So visible_get() is NOT the test (an eye-hidden collider works); the depsgraph is: the object among the view
# layer's evaluated ids, with the modifier on in the viewport. A cloth or soft body whose collision collection is set
# collides only with the objects in it.
def _collider_rows(sc):
    """(rows, blockers) for every object in the scene carrying a COLLISION modifier."""
    if not hasattr(bpy.context, "evaluated_depsgraph_get"):
        # ONLY A STUBBED bpy LACKS IT: test_blender_refusals runs these ops offline on a SimpleNamespace context, with
        # no depsgraph to ask. Every real Blender has it, so a real call never takes this branch.
        return [], []
    vl = bpy.context.view_layer
    dg = bpy.context.evaluated_depsgraph_get()
    evaluated = {i.name_full for i in dg.ids if isinstance(i, bpy.types.Object)}
    in_layer = {o.name_full for o in vl.objects}
    limited = []          # (sim object, its collision collection's objects)
    for o in sc.objects:
        for m in o.modifiers:
            cs = getattr(m, "collision_settings", None)
            coll = getattr(cs, "collection", None) if cs is not None else None
            if m.type in ("CLOTH", "SOFT_BODY") and coll is not None:
                limited.append((o.name, coll.name, {x.name_full for x in coll.all_objects}))
            elif m.type == "SOFT_BODY" and getattr(getattr(o, "soft_body", None), "collision_collection", None):
                coll = o.soft_body.collision_collection
                limited.append((o.name, coll.name, {x.name_full for x in coll.all_objects}))
    rows, blockers = [], []
    for o in sc.objects:
        mods = [m for m in o.modifiers if m.type == "COLLISION"]
        if not mods:
            continue
        m = mods[0]
        seen = o.name_full in evaluated and bool(m.show_viewport)
        why = None
        if not seen:
            if o.name_full not in in_layer:
                why = "its collection is excluded from view layer '%s'" % vl.name
            elif o.hide_viewport:
                why = "hide_viewport is on (Disable in Viewports) - the depsgraph leaves the object out"
            elif o.name_full not in evaluated:
                hidden = [c.name for c in o.users_collection if c.hide_viewport]
                why = ("its collection %s is hidden in viewports (Disable in Viewports)" % ", ".join(hidden)
                       if hidden else "the view layer's depsgraph does not evaluate it")
            else:
                why = "the Collision modifier's show_viewport is off - the modifier is not evaluated"
        row = {"object": o.name, "modifier": m.name, "seenBySimulation": seen, "visibleGet": bool(o.visible_get()),
               "hideViewport": bool(o.hide_viewport), "hiddenInViewLayer": bool(o.hide_get()),
               "hideRender": bool(o.hide_render), "showViewport": bool(m.show_viewport)}
        if why:
            row["why"] = why
            blockers.append("collider '%s' cannot be seen by any simulation: %s. A cloth or soft body falls straight "
                            "through it. Turn it back on (set_object_visibility hideViewport false, "
                            "set_collection_visibility, or the modifier's show_viewport)." % (o.name, why))
        ignored = [name for name, _cn, members in limited if o.name_full not in members]
        if ignored:
            row["ignoredBy"] = ignored
            row["ignoredNote"] = ("%s collide%s only with %s" % (
                ", ".join(ignored), "s" if len(ignored) == 1 else "",
                ", ".join("collection '%s'" % cn for name, cn, _m in limited if name in ignored)))
        rows.append(row)
    return rows, blockers


def op_physics_info(params):
    """What the physics setup IS - the read half of a family that could only write.

    add_rigid_body, add_cloth, add_collision and bake_physics all set, and until 2026-09-03 nothing
    anywhere reported what they had set. scene_info carries no physics at all. So a caller could not
    ask what mass a rigid body has, which collision shape it uses, whether an object is kinematic,
    or - the one that decides whether a render is right - whether the cache is baked.

    A rigid body is NOT a modifier: it lives on obj.rigid_body, so list_modifiers cannot see it and
    there was no route to it by any op.

    THE INERT STATE THIS EXISTS TO CATCH. An object can carry a fully configured obj.rigid_body -
    mass, friction, shape, all reading back perfectly - and still not simulate at all, because the
    simulation is driven by the scene's RigidBodyWorld and only acts on objects in ITS COLLECTION.
    Remove the object from that collection and every field on it stays correct while it hangs in
    the air. That is reported as `inSimulation` per object, not inferred from the settings.

    params:
      object / name (str)   report one object only. Default every object with physics.
    """
    reject_unknown(params, _INFO_KEYS, "physics_info")
    sc = bpy.context.scene
    only = take(params, "object", "name", kind=str)
    if only:
        objects = [get_object(only)]
    else:
        objects = list(bpy.data.objects)

    rbw = getattr(sc, "rigidbody_world", None)
    rbw_names = set()
    if rbw is not None and getattr(rbw, "collection", None) is not None:
        rbw_names = {o.name for o in rbw.collection.objects}

    world = {
        "exists": rbw is not None,
        "collection": (rbw.collection.name
                       if rbw is not None and getattr(rbw, "collection", None) else None),
        "objectCount": len(rbw_names),
        # THE JOINTS, counted since add_rigid_body_constraint makes many in one call (2026-10-04).
        "constraintCount": (len(rbw.constraints.objects)
                            if rbw is not None and getattr(rbw, "constraints", None) is not None else 0),
        "enabled": bool(getattr(rbw, "enabled", False)) if rbw is not None else None,
        "substeps": getattr(rbw, "substeps_per_frame", None) if rbw is not None else None,
        "solverIterations": getattr(rbw, "solver_iterations", None) if rbw is not None else None,
        "pointCache": _cache_row(getattr(rbw, "point_cache", None), sc) if rbw is not None else None,
        # GRAVITY AND ITS GATE, on the SCENE rather than the world - which is why they sit here
        # beside the world's own fields rather than inside it, and why `exists: false` does not make
        # them None: a scene has gravity whether or not a rigid body world has been created.
        #
        # set_physics_world's own docstring has said it since it was written - "A GRAVITY VECTOR IS
        # INERT WHILE use_gravity IS OFF. Blender stores the vector either way" - and until
        # 2026-09-15 use_gravity was readable only out of that op's own `applied`, which means only
        # by making a write. So a caller could read a gravity of -9.81 from a scene that has gravity
        # switched off entirely. Sixth of these found on one day; the first five are in
        # audit_value_without_gate.py's docstring.
        "gravity": [round(float(v), 6) for v in sc.gravity],
        "useGravity": bool(sc.use_gravity),
    }

    rows, blockers = [], []
    for obj in objects:
        rb = getattr(obj, "rigid_body", None)
        mods = [m for m in getattr(obj, "modifiers", []) if m.type in _SIM_MODIFIERS]
        # A CONSTRAINT OBJECT HAS NEITHER, and this filter skipped it entirely until 2026-09-15 - so
        # a rigid body constraint was not merely under-reported, it was INVISIBLE to the one op whose
        # job is describing the physics setup. add_rigid_body_constraint makes a bare empty
        # (objects.new(name, None), ARROWS display) and links it into the world's constraint
        # collection; it carries no rigid_body and no modifier, which is exactly the shape this
        # filter dropped.
        rbc = getattr(obj, "rigid_body_constraint", None)
        node_sims = [m for m in getattr(obj, "modifiers", []) if m.type == "NODES" and _has_sim_zone(m.node_group)]
        if rb is None and rbc is None and not mods and not node_sims:
            continue
        row = {"object": obj.name, "type": obj.type}
        if node_sims:
            # A GEOMETRY-NODES SIMULATION (2026-10-04) - the 5.2 node cloth among them - was invisible here: it is a
            # NODES modifier whose group holds a simulation zone, not one of the point-cache modifier types.
            row["nodeSimulations"] = [{"modifier": m.name, "group": m.node_group.name,
                                       "showRender": bool(m.show_render),
                                       "bakeTarget": getattr(m, "bake_target", None),
                                       "inputs": _group_inputs(m.node_group)} for m in node_sims]
        if rbc is not None:
            # THE FLAG BESIDE THE NUMBER, third time tonight after collisionMargin and
            # cutoffDistance. breaking_threshold is ignored unless use_breaking is on - the op's own
            # comment says so where it writes them together - so reporting the number alone is a
            # threshold that may be holding nothing back.
            row["rigidBodyConstraint"] = {
                "constraintType": rbc.type,
                "object1": rbc.object1.name if rbc.object1 else None,
                "object2": rbc.object2.name if rbc.object2 else None,
                "enabled": bool(rbc.enabled),
                "disableCollisions": bool(rbc.disable_collisions),
                "useBreaking": bool(rbc.use_breaking),
                "breakingThreshold": round(float(rbc.breaking_threshold), 6),
                "overrideSolverIterations": bool(rbc.use_override_solver_iterations),
                "solverIterations": int(rbc.solver_iterations),
            }
        if rb is not None:
            # The same reader add_rigid_body answers with - see _rigid_body_state for why there is one,
            # and for the margin's gate.
            row["rigidBody"] = _rigid_body_state(rb)
            # THE MEASUREMENT THAT SETTINGS CANNOT GIVE. Membership of the RigidBodyWorld
            # collection is what decides whether any of the above does anything.
            row["inSimulation"] = obj.name in rbw_names
            if not row["inSimulation"]:
                blockers.append("'%s' has a fully configured rigid body and is NOT in the "
                                "RigidBodyWorld collection%s, so it will not simulate at all - "
                                "every setting on it reads back correctly and it hangs in the air."
                                % (obj.name,
                                   " (there is no RigidBodyWorld)" if rbw is None else ""))
            elif rb.type == "ACTIVE" and rb.kinematic:
                blockers.append("'%s' is an ACTIVE rigid body with kinematic ON, which means it is "
                                "driven by its animation rather than by the sim - the usual "
                                "accident when a keyframed object refuses to fall." % obj.name)
        if mods:
            # `settings` IS THE READ HALF, added 2026-09-15. These rows carried a modifier's name,
            # type and visibility and never one of its SETTINGS, so add_cloth, add_soft_body and
            # add_collision could each be configured and then not asked - their own responses read
            # the values back honestly, but there was no second way in. Merged from every holder the
            # family keeps state in, because cloth splits across settings and collision_settings and
            # soft body is not on the modifier at all. Omitted entirely when the family has no table
            # yet, rather than reported as an empty dict that would read as "nothing is set".
            row["simModifiers"] = []
            for m in mods:
                entry = {"name": m.name, "type": m.type, "showRender": bool(m.show_render),
                         "showViewport": bool(m.show_viewport),
                         "pointCache": _cache_row(getattr(m, "point_cache", None), sc)}
                settings, not_finite = {}, []
                for _label, holder, rows_spec in _sim_holders(obj, m):
                    block, bad = _read_rows(holder, rows_spec)
                    settings.update(block)
                    not_finite.extend(bad)
                # MEASURED FROM WHICH STRUCT EXISTS, not read off ui_type, because those two can
                # disagree: op_add_dynamic_paint records that type_toggle can create brush_settings
                # and leave ui_type showing the other tab. Derived here the same way that op derives
                # it, so the two cannot drift, and reported BESIDE uiType rather than instead of it -
                # the tab a panel shows is a real fact, it is just not this one.
                if str(m.type) == "DYNAMIC_PAINT":
                    made = [t for t, s in (("CANVAS", getattr(m, "canvas_settings", None)),
                                           ("BRUSH", getattr(m, "brush_settings", None)))
                            if s is not None]
                    settings["paintType"] = "+".join(made) if made else None
                if settings:
                    entry["settings"] = settings
                # NAMED, so null-because-not-finite stays distinguishable from absent-because-this-
                # build-has-no-such-property. Only when non-empty: a list that is almost always []
                # on every modifier row is noise in the common case.
                if not_finite:
                    entry["settingsNotFinite"] = sorted(set(not_finite))
                row["simModifiers"].append(entry)
            for m in mods:
                if not m.show_render:
                    blockers.append("'%s' on '%s' is disabled in the RENDER, so it is visible in "
                                    "the viewport and absent from the picture."
                                    % (m.name, obj.name))
        rows.append(row)

    # CACHE STATE ACROSS EVERYTHING, which is the question that decides whether a frame is right.
    caches = []
    if world["pointCache"] is not None:
        caches.append(dict(world["pointCache"], kind="rigidbody"))
    for row in rows:
        for m in row.get("simModifiers", []):
            if m["pointCache"] is not None:
                caches.append(dict(m["pointCache"], kind=m["type"].lower(), object=row["object"]))
    unbaked = [c for c in caches if not c["isBaked"]]
    short = [c for c in caches if c["isBaked"] and not c["coversSceneRange"]]
    # EVERY COLLIDER, seen or not (_collider_rows) - scene-wide like the caches, whatever `object` asked for.
    colliders, collider_blockers = _collider_rows(sc)
    blockers.extend(collider_blockers)
    if unbaked:
        blockers.append("%d point cache(s) are NOT baked. Physics is evaluated by stepping forward "
                        "from the start frame, so jumping straight to a late frame shows the REST "
                        "state and a render of it is simply wrong. Run bake_physics."
                        % len(unbaked))
    if short:
        blockers.append("%d cache(s) are baked but their range does NOT cover the scene's %d-%d - "
                        "a bake made before the range was extended stays valid and short, and the "
                        "frames past its end silently fall back to the rest state."
                        % (len(short), sc.frame_start, sc.frame_end))

    # ONLY THE OBJECT ROWS GROW. objectsWithPhysics, the caches and the blockers describe the whole scene and are
    # computed above from the full set - a caller asking for two rows must still be told that nine caches are
    # unbaked, because that is the answer this endpoint exists to give.
    rows = exclude_rows(rows, params, name_of=lambda r: r.get("object", ""))
    matched = len(rows)
    rows, obj_extra = limit_rows(rows, params, "object(s) with physics")
    return dict({
        "ok": True,
        "scene": sc.name,
        "sceneFrameRange": [sc.frame_start, sc.frame_end],
        "rigidBodyWorld": world,
        "objectsWithPhysics": matched,
        "objects": rows,
        "caches": caches,
        "cacheCount": len(caches),
        "colliders": colliders,
        "blockers": blockers,
        "readyToRender": not blockers,
    }, **{("objects" + k[0].upper() + k[1:]): v for k, v in obj_extra.items()})


_PHYSWORLD_KEYS = {"gravity", "useGravity", "substeps", "solverIterations", "timeScale",
                   "splitImpulse", "enabled", "cacheStart", "cacheEnd", "cacheStep"}

# The keys that need a RIGID BODY WORLD to exist, as opposed to living on the scene itself.
_NEEDS_WORLD = {"substeps", "solverIterations", "timeScale", "splitImpulse", "enabled",
                "cacheStart", "cacheEnd", "cacheStep"}


def op_set_physics_world(params):
    """The scene-level simulation settings - gravity, substeps, solver iterations, time scale.

    THE ASYMMETRY THIS CLOSES. physics_info has always reported substeps and solverIterations and
    nothing could write either, so the addon could add rigid bodies, bake them, and report exactly
    how the solver was configured while being unable to change any of it. Those two are the knobs
    that fix a simulation which jitters or tunnels through a floor, and scene gravity drives every
    rigid body in the file - none of it was reachable.

    A GRAVITY VECTOR IS INERT WHILE use_gravity IS OFF. Blender stores the vector either way, so
    setting it alone changes three numbers and not the simulation, and every field reads back
    exactly as written. Passing gravity turns the toggle on unless useGravity says otherwise - the
    same shape as cutoffDistance on a light and a focus object on a camera, and the response says
    when it happened.

    CHANGING A SOLVER SETTING DOES NOT RE-SIMULATE. An existing bake stays on disk and stays marked
    valid, so the next frame you look at was computed with the OLD substeps - a stale cache is
    exactly as convincing as a fresh one. is_baked and is_outdated are reported afterwards, and a
    bake that is now stale is named rather than left to be discovered as "the change did nothing".

    params:
      gravity [x,y,z] (list)      metres per second squared. Blender's default is [0,0,-9.81].
      useGravity (bool)
      substeps (int)              substeps_per_frame - raise this when bodies tunnel or jitter
      solverIterations (int)      constraint solver iterations per substep
      timeScale (float)           simulation speed multiplier; 0 freezes the sim
      splitImpulse (bool)         use_split_impulse - reduces bounce on stacked bodies
      enabled (bool)              the whole rigid body world on or off
      cacheStart / cacheEnd / cacheStep (int)   the point cache's frame range
    """
    reject_unknown(params, _PHYSWORLD_KEYS, "set_physics_world")
    sc = bpy.context.scene
    asked = [k for k in _PHYSWORLD_KEYS if params.get(k) is not None]
    if not asked:
        raise MifOpError("nothing to do - pass at least one of %s. NOTHING was changed."
                         % ", ".join(sorted(_PHYSWORLD_KEYS)))

    rbw = getattr(sc, "rigidbody_world", None)
    needs = [k for k in asked if k in _NEEDS_WORLD]
    if needs and rbw is None:
        # REFUSED RATHER THAN CREATED. A rigid body world is scene-wide state, and conjuring one
        # from a settings call would be a side effect nobody asked for - the same reason
        # set_compositing refuses to switch use_nodes on from a read.
        raise MifOpError(
            "this scene has no rigid body world, so %s cannot be set. One appears when the first "
            "rigid body is added - call add_rigid_body on any object first. Creating a scene-wide "
            "simulation from a settings call would be a side effect nobody asked for. NOTHING was "
            "changed." % ", ".join(sorted(needs)))

    # PARSED IN FULL BEFORE ANYTHING IS WRITTEN. Both halves of this were defects in this op as
    # first written on 2026-09-04, found by turning the day's own lenses on the day's own code:
    #
    #   {"gravity": ["a","b","c"]}           the isinstance check only proved it was a list of
    #                                        three, so float("a") raised a RAW ValueError out of the
    #                                        op - the shape blender_version_matrix's suspect-refusal
    #                                        detector exists to catch, escaping a handler's contract.
    #   {"gravity": [...], "substeps": "x"}  gravity was written and THEN take_int refused.
    #                                        Measured: -9.81 became -1.5 and the call failed. A
    #                                        half-applied settings call, which set_light_influence
    #                                        and set_material_settings both avoid by validating up
    #                                        front - this one did not, on the same day.
    applied = {}
    pending_scene, pending_world, pending_cache = {}, {}, {}

    grav = params.get("gravity")
    if grav is not None:
        if not isinstance(grav, (list, tuple)) or len(grav) < 3:
            raise MifOpError("'gravity' must be [x, y, z] in m/s^2, got %r. NOTHING was changed."
                             % (grav,))
        # finite_floats: a bare float() here stored [0, 0, nan] as the scene's gravity under ok:true (5.0.1,
        # 2026-09-20), which every simulation then integrates.
        try:
            pending_scene["gravity"] = finite_floats(grav[:3], "gravity")
        except MifOpError as exc:
            raise MifOpError("%s 'gravity' is three finite numbers in m/s^2. NOTHING was changed." % exc)
    if params.get("useGravity") is not None:
        pending_scene["useGravity"] = take_bool(params, "useGravity", default=True)

    if rbw is not None:
        for key, attr, cast in (("substeps", "substeps_per_frame", int),
                                ("solverIterations", "solver_iterations", int),
                                ("timeScale", "time_scale", float),
                                ("splitImpulse", "use_split_impulse", bool),
                                ("enabled", "enabled", bool)):
            if params.get(key) is None:
                continue
            pending_world[attr] = (take_bool(params, key, default=True) if cast is bool
                                   else (take_int(params, key) if cast is int
                                         else take_float(params, key)))
            applied[key] = pending_world[attr]
        if getattr(rbw, "point_cache", None) is not None:
            for key, attr in (("cacheStart", "frame_start"), ("cacheEnd", "frame_end"),
                              ("cacheStep", "frame_step")):
                if params.get(key) is None:
                    continue
                pending_cache[attr] = take_int(params, key)
                applied[key] = pending_cache[attr]

    # COMMIT. Nothing below can refuse - and unlike the same sentence in ops_lightcam before today,
    # that is true here because every parse above happens before the first write.
    if "gravity" in pending_scene:
        sc.gravity = pending_scene["gravity"]
        applied["gravity"] = pending_scene["gravity"]
    if "useGravity" in pending_scene:
        sc.use_gravity = pending_scene["useGravity"]
        applied["useGravity"] = sc.use_gravity

    # THE TOGGLE THAT MAKES THE VECTOR MEAN ANYTHING.
    auto_gravity = False
    if "gravity" in applied and "useGravity" not in applied and not sc.use_gravity:
        sc.use_gravity = True
        auto_gravity = True

    if rbw is not None:
        for attr, value in pending_world.items():
            setattr(rbw, attr, value)
        pc = getattr(rbw, "point_cache", None)
        if pc is not None:
            for attr, value in pending_cache.items():
                setattr(pc, attr, value)

    # READ BACK. substeps and solver_iterations are CLAMPED by Blender rather than refused, so
    # echoing the request would report a value the solver does not have.
    after = {"gravity": rnd(list(sc.gravity)), "useGravity": bool(sc.use_gravity)}
    clamped = {}
    if rbw is not None:
        after.update({
            "substeps": int(rbw.substeps_per_frame),
            "solverIterations": int(rbw.solver_iterations),
            "timeScale": round(float(rbw.time_scale), 6),
            "splitImpulse": bool(rbw.use_split_impulse),
            "enabled": bool(rbw.enabled),
        })
        # THE CACHE KEYS TOO. cacheStep was written and read back by nothing (second seat, mailbox id 103), and it does
        # not take: MEASURED 2026-09-16, identical on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1, a rigid body world's
        # point_cache.frame_step written 3, 50 or 0 reads back 1 (hard range 1-20).
        # Reported beside cacheFrameRange, where the cache's other facts already sit, not in `applied`.
        rb_cache = getattr(rbw, "point_cache", None)
        cache_now = ({"cacheStart": int(rb_cache.frame_start), "cacheEnd": int(rb_cache.frame_end),
                      "cacheStep": int(rb_cache.frame_step)} if rb_cache is not None else {})
        for key in ("substeps", "solverIterations", "timeScale"):
            if key in applied and after.get(key) != applied[key]:
                clamped[key] = {"requested": applied[key], "stored": after[key]}
        for key, stored in cache_now.items():
            if key in applied and stored != applied[key]:
                clamped[key] = {"requested": applied[key], "stored": stored}

    pc = getattr(rbw, "point_cache", None) if rbw is not None else None
    was_baked = bool(getattr(pc, "is_baked", False)) if pc is not None else False
    solver_touched = [k for k in applied
                      if k in ("substeps", "solverIterations", "timeScale", "splitImpulse",
                               "gravity", "useGravity")]
    return {
        "ok": True,
        "scene": sc.name,
        "hasRigidBodyWorld": rbw is not None,
        "applied": after,
        "clamped": clamped or None,
        "gravityWasEnabledAutomatically": auto_gravity,
        # THE CACHE STATE, because a solver change does not re-simulate and the old bake stays
        # marked valid. is_outdated is Blender's own answer to "does this need re-baking".
        "cacheIsBaked": was_baked,
        "cacheIsOutdated": bool(getattr(pc, "is_outdated", False)) if pc is not None else None,
        "cacheFrameRange": ([int(pc.frame_start), int(pc.frame_end)]
                            if pc is not None else None),
        "cacheFrameStep": int(pc.frame_step) if pc is not None else None,
        "cacheStepNote":("a rigid body world's cache steps every frame: Blender kept frame_step at %d and did not take "
                          "%d (measured on all five builds - 3, 50 and 0 all read back 1)."
                          % (clamped["cacheStep"]["stored"], clamped["cacheStep"]["requested"])
                          if "cacheStep" in clamped else None),
        "note": ("use_gravity was OFF, so the gravity vector would have been stored and ignored - "
                 "it has been turned ON. Pass useGravity:false to store it without applying it."
                 if auto_gravity else
                 ("%s changed and this scene already has a BAKED cache. Changing a solver setting "
                  "does not re-simulate - the existing bake stays on disk and stays convincing, so "
                  "every frame you look at was computed with the OLD settings. Re-run bake_physics "
                  "with clear:true and then bake again."
                  % ", ".join(sorted(solver_touched))) if (was_baked and solver_touched) else None),
    }

_STEP_KEYS = {"object", "name", "start", "end", "maxSpeed", "maxMemoryGB", "maxSeconds", "fastest"}


def _private_memory_bytes():
    """(bytes, source) for this Blender process: Windows private bytes (what the 61.7 GB sim grew), Linux VmRSS, or the
    peak resident size elsewhere - the source says which, because they are different numbers."""
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                                               "PrivateUsage")]
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(pmc)
        k32 = ctypes.windll.kernel32
        # ARGTYPES ARE DECLARED: GetCurrentProcess returns the pseudo-handle -1, which as an untyped argument overflowed
        # ctypes' default int ("OverflowError: int too long to convert", measured on 5.2.1).
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        if k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return int(pmc.PrivateUsage), "privateBytes"
    except Exception:                                                         # noqa: BLE001
        pass                     # not Windows, or the call failed: the next source below
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024, "residentBytes"
    except OSError:
        pass
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), "peakResident"
    except (ImportError, AttributeError):
        return None, "unavailable"


def _world_positions(obj, deps):
    """The evaluated mesh's vertex positions in world space, metres: a numpy (N, 3) array."""
    import numpy as np
    ev = obj.evaluated_get(deps)
    me = ev.to_mesh()
    try:
        co = np.empty(len(me.vertices) * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
    finally:
        ev.to_mesh_clear()
    co = co.reshape(-1, 3)
    mw = np.array(obj.matrix_world, dtype=np.float64)
    world = co @ mw[:3, :3].T + mw[:3, 3]
    return world * float(bpy.context.scene.unit_settings.scale_length or 1.0)


# LOOSE VERTICES IN A CLOTH. FOUND 2026-10-04: 21 vertices left inside deleted face windows - no faces - fell freely and
# drove maxSpeed past the guard, which read as the cloth running away. A vertex no face uses has no area for the
# solver to hold, so it falls on gravity alone (measured here: one drops below the sheet it came with in ten frames).
_LOOSE_LISTED = 50


def _loose_vertices(obj):
    """{count, vertices [{index, world}], truncated} for the base mesh's vertices no face uses, or None."""
    me = obj.data
    used = set()
    for poly in me.polygons:
        used.update(poly.vertices)
    loose = [v.index for v in me.vertices if v.index not in used]
    if not loose:
        return None
    mw = obj.matrix_world
    rows = [{"index": i, "world": [round(float(c), 5) for c in (mw @ me.vertices[i].co)]}
            for i in loose[:_LOOSE_LISTED]]
    return {"count": len(loose), "vertices": rows, "truncated": len(loose) > _LOOSE_LISTED}


def op_step_simulation(params):
    """Step a simulation frame by frame and stop BEFORE the next frame once it runs away.

    WHY IT EXISTS. On 2026-10-03 a garment drape in an agent's background Blender grew to 61.7 GB and the machine's
    pagefile ran out under a Movie Render Queue render, which died. Nothing typed could step a cloth a few frames
    and look: bake_physics bakes the whole range in one call, and the agent wrote its own stepper (sim_step.py, run 14
    times) to read per-vertex speed and memory between frames. That stepper caught the cause - a cuff exploding at 3, 6,
    then 9 m/s over frames 32-34, from zero-length edges - while memory stayed flat at 4.3 GB. A speed limit stops an
    explosion before it builds the pair lists that eat memory; the memory limit is the backstop.

    EACH FRAME: scene.frame_set, then the measured object's evaluated vertices in world space (metres, through the
    scene's unit scale) give maxSpeed and p99Speed (m/s, against the previous frame at the scene's fps), its box, the
    process's memory and the time taken. It stops when the end is reached, or after a frame whose max speed passed
    maxSpeed, whose memory passed maxMemoryGB, or once maxSeconds have gone by - and `stoppedBecause` says which, with
    the fastest vertices at the frame it stopped on.

    STEPPING IS IN ORDER. A cloth or soft body advances one frame at a time from its cache; a jump reads what the last
    step left. So `start` defaults to the frame after the current one, and a start that is neither that nor the cache's
    first frame is reported (`jumpNote`). Memory and time limits are on by default (12 GB, the ceiling sims here run
    under, and 60 s, since the call holds the socket); the speed limit is the caller's to choose.
    """
    reject_unknown(params, _STEP_KEYS, "step_simulation")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    sc = bpy.context.scene
    cur = int(sc.frame_current)
    start = take_int(params, "start", default=cur + 1)
    end = take_int(params, "end", required=True)
    max_speed = take_float(params, "maxSpeed", default=None)
    max_mem = take_float(params, "maxMemoryGB", default=12.0)
    max_s = take_float(params, "maxSeconds", default=60.0)
    # NUMPY AFTER THE GUARDS: every Blender bundles it, but the offline gate imports the ops with bpy stubbed in a
    # Python that may have none, and an import first made an unknown key or a missing object raise ModuleNotFoundError
    # instead of being refused (test_blender_refusals B110 and B111, 2026-10-05).
    import numpy as np
    n_fast = take_int(params, "fastest", default=10)
    if end < start:
        raise MifOpError("end (%d) is before start (%d). NOTHING was stepped." % (end, start))
    for key, val in (("maxSpeed", max_speed), ("maxMemoryGB", max_mem), ("maxSeconds", max_s)):
        if val is not None and not val > 0:
            raise MifOpError("%s must be above 0, got %r. NOTHING was stepped." % (key, val))
    if not 0 <= n_fast <= 1000:
        raise MifOpError("fastest must be 0-1000, got %d. NOTHING was stepped." % n_fast)
    if max_s > 600:
        raise MifOpError("maxSeconds is at most 600 - the call holds Blender's one socket for its whole run. Step a long "
                         "simulation in chunks: each call continues from the frame the last one stopped on. NOTHING was "
                         "stepped.")
    fps = float(sc.render.fps) / float(sc.render.fps_base or 1.0)
    sims = [m for m in obj.modifiers if getattr(m, "point_cache", None) is not None]
    cache_starts = sorted({int(m.point_cache.frame_start) for m in sims})

    out = {"object": obj.name, "start": start, "end": end, "frameBefore": cur, "fps": fps,
           "limits": {"maxSpeed": max_speed, "maxMemoryGB": max_mem, "maxSeconds": max_s},
           "simulationModifiers": [m.name for m in sims]}
    if not sims:
        out["note"] = ("'%s' has no cloth, soft body or other cached simulation modifier of its own - it is measured as "
                       "the scene moves it (a collider, an animated or driven mesh)." % obj.name)
    # NAMED BEFORE THE FIRST STEP, for a cloth or soft body: the vertices no face uses (_loose_vertices).
    if any(m.type in ("CLOTH", "SOFT_BODY") for m in obj.modifiers):
        loose = _loose_vertices(obj)
        if loose:
            out["looseVertices"] = loose
            out["looseNote"] = ("%d vertex(es) of '%s' belong to no face - left behind when faces were deleted, or on a "
                                "loose edge. The solver has no area to hold them, so they fall on gravity alone, and "
                                "their speed reads as the cloth running away. Delete them (delete_mesh_elements) or "
                                "fill the holes before simulating." % (loose["count"], obj.name))
    if start != cur + 1 and start not in cache_starts and sims:
        out["jumpNote"] = ("start %d is neither the frame after the current one (%d) nor a cache's first frame (%s): a "
                           "simulation steps one frame at a time, so the first frame may read what the last step left."
                           % (start, cur, ", ".join(str(s) for s in cache_starts)))
    # A COLLIDER THE SIMULATION CANNOT SEE is said before the first step (physics_info's blockers, _collider_rows): a
    # cloth falls straight through a body whose collider has hide_viewport on, and the speeds alone read as a drop.
    _colliders, collider_blockers = _collider_rows(sc)
    if collider_blockers:
        out["colliderBlockers"] = collider_blockers
    mem0, mem_source = _private_memory_bytes()
    out["memorySource"] = mem_source
    out["memoryAtStartGB"] = round(mem0 / 1e9, 3) if mem0 is not None else None

    deps = bpy.context.evaluated_depsgraph_get()
    prev = _world_positions(obj, deps) if start == cur + 1 else None
    rows, stopped, fast_rows, last = [], "end", [], None
    t0 = time.time()
    for frame in range(start, end + 1):
        if time.time() - t0 > max_s:
            stopped = "maxSeconds"
            break
        f0 = time.time()
        sc.frame_set(frame)
        deps = bpy.context.evaluated_depsgraph_get()
        pos = _world_positions(obj, deps)
        row = {"frame": frame, "seconds": round(time.time() - f0, 4),
               "bboxMin": [round(float(v), 5) for v in pos.min(axis=0)] if len(pos) else None,
               "bboxMax": [round(float(v), 5) for v in pos.max(axis=0)] if len(pos) else None}
        speed = None
        if prev is not None and len(prev) == len(pos) and len(pos):
            speed = np.linalg.norm(pos - prev, axis=1) * fps
            row["maxSpeed"] = round(float(speed.max()), 5)
            row["p99Speed"] = round(float(np.percentile(speed, 99)), 5)
        elif prev is not None and len(prev) != len(pos):
            row["topologyChanged"] = [len(prev), len(pos)]
        mem, _src = _private_memory_bytes()
        row["memoryGB"] = round(mem / 1e9, 3) if mem is not None else None
        rows.append(row)
        last = frame
        if speed is not None and n_fast:
            idx = np.argsort(speed)[::-1][:n_fast]
            fast_rows = [{"index": int(i), "speed": round(float(speed[i]), 5),
                          "world": [round(float(c), 5) for c in pos[i]]} for i in idx]
        prev = pos
        if max_speed is not None and speed is not None and float(speed.max()) > max_speed:
            stopped = "maxSpeed"
            break
        if max_mem is not None and mem is not None and mem / 1e9 > max_mem:
            stopped = "maxMemoryGB"
            break
    out.update({
        "stoppedBecause": stopped,
        "lastFrame": last,
        "framesStepped": len(rows),
        "frame": int(sc.frame_current),
        "elapsedSeconds": round(time.time() - t0, 3),
        "frames": rows if len(rows) <= 500 else rows[-500:],
        "framesTruncated": len(rows) > 500,
        "fastest": fast_rows,
        "peakSpeed": max((r["maxSpeed"] for r in rows if "maxSpeed" in r), default=None),
        "peakMemoryGB": max((r["memoryGB"] for r in rows if r.get("memoryGB") is not None), default=None),
    })
    if prev is None or (rows and "maxSpeed" not in rows[0]):
        out["speedNote"] = ("the first frame has no speed: speed is measured against the frame before, and the call "
                            "did not start from the frame after the current one.")
    if stopped != "end":
        out["warning"] = ("stopped at frame %s by %s - the scene is left there, and the frames after it were not "
                          "stepped." % (last, stopped))
    return out


# ---------------------------------------------------------------------------
# fracture_mesh
# ---------------------------------------------------------------------------

_FRAC_KEYS = {"object", "objects", "source", "sourceLimit", "points", "noise", "seed", "cellScale", "margin",
              "recursion", "recursionSourceLimit", "recursionClamp", "recursionChance", "materialIndex",
              "collection", "hideOriginal", "rigidBody"}
_FRAC_SOURCES = ("RANDOM", "VERTS", "PARTICLES", "POINTS")
# add_rigid_body's settings - its targets (object, objects, collection, copyFrom, update) are the cells themselves
_FRAC_RB_KEYS = _RB_KEYS - {"object", "name", "objects", "collection", "copyFrom", "update"}
_FRAC_RB_SHAPES = ("CONVEX_HULL", "MESH", "BOX", "SPHERE", "CAPSULE", "CYLINDER", "CONE")


def _frac_world_bmesh(obj):
    """The object's EVALUATED mesh (modifiers applied) as a bmesh in world space."""
    import bmesh   # noqa: PLC0415
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        bm = bmesh.new()
        bm.from_mesh(me)
    finally:
        ev.to_mesh_clear()
    bm.transform(obj.matrix_world)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def _frac_inside(bm):
    """inside(point) for a closed bmesh: the nearest surface point's face normal points away from it."""
    from mathutils.bvhtree import BVHTree   # noqa: PLC0415
    tree = BVHTree.FromBMesh(bm)

    def inside(p):
        loc, nrm, _i, _d = tree.find_nearest(p)
        return loc is not None and (p - loc).dot(nrm) < 0.0
    return inside


def _frac_random_inside(bm, count, rng):
    """Up to `count` uniform random points inside a closed bmesh (rejection inside its box)."""
    from mathutils import Vector   # noqa: PLC0415
    cos = [v.co for v in bm.verts]
    lo = Vector([min(c[i] for c in cos) for i in range(3)])
    hi = Vector([max(c[i] for c in cos) for i in range(3)])
    inside = _frac_inside(bm)
    out = []
    for _ in range(count * 200):
        p = Vector([rng.uniform(lo[i], hi[i]) for i in range(3)])
        if inside(p):
            out.append(p)
            if len(out) >= count:
                break
    return out


def _frac_voronoi(base, seeds, margin, mat_index):
    """One closed bmesh per seed: the base cut by the bisector plane to every other seed, keeping the seed's side,
    each cut capped. A plane farther from the seed than the cell's farthest vertex cannot cut it, and seeds are
    visited nearest first, so the walk stops there. Empty and degenerate cells are dropped. [(bm, volume)]"""
    import bmesh   # noqa: PLC0415
    out = []
    n = len(seeds)
    for i, p in enumerate(seeds):
        bm = base.copy()
        order = sorted(range(n), key=lambda j: (seeds[j] - p).length_squared)
        for j in order:
            if j == i:
                continue
            d = seeds[j] - p
            dist = d.length
            if dist < 1e-9:
                continue
            reach = max(((v.co - p).length for v in bm.verts), default=0.0)
            half = dist * 0.5 - margin * 0.5
            if half > reach:
                break
            nrm = d / dist
            co = p + nrm * half
            if max((v.co - co).dot(nrm) for v in bm.verts) <= 1e-7:
                continue
            bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=co, plane_no=nrm,
                                   clear_outer=True, dist=1e-6)
            if len(bm.verts) < 4:
                break
            rim = [e for e in bm.edges if e.is_boundary]
            if rim:
                for f in bmesh.ops.holes_fill(bm, edges=rim, sides=0)["faces"]:
                    f.material_index = mat_index
                    f.smooth = False
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")
        if len(bm.faces) < 4:
            bm.free()
            continue
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        vol = bm.calc_volume(signed=True)
        if vol <= 1e-12:
            bm.free()
            continue
        out.append((bm, vol))
    return out


def _frac_seeds(obj, bm, source, limit, points, noise, rng):
    """World-space seed points for one object, before noise is applied; refuses a source that gives none."""
    from mathutils import Vector   # noqa: PLC0415
    if source == "RANDOM":
        pts = _frac_random_inside(bm, limit, rng)
    elif source == "VERTS":
        pts = [v.co.copy() for v in bm.verts]
    elif source == "PARTICLES":
        dg = bpy.context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(dg)
        pts = [Vector(pa.location) for ps in ev.particle_systems for pa in ps.particles]
        if not pts:
            raise MifOpError("source PARTICLES: '%s' has no particles (%d particle system(s)). Add one with "
                             "add_particles, or use RANDOM, VERTS or POINTS. NOTHING was created."
                             % (obj.name, len(obj.particle_systems)))
    else:
        pts = [Vector(p) for p in points]
    if len(pts) > limit:
        pts = rng.sample(pts, limit)
    if noise > 0.0:
        cos = [v.co for v in bm.verts]
        size = [max(c[i] for c in cos) - min(c[i] for c in cos) for i in range(3)]
        pts = [p + Vector([rng.uniform(-1.0, 1.0) * noise * size[i] * 0.5 for i in range(3)]) for p in pts]
    seen, uniq = set(), []
    for p in pts:
        key = tuple(round(c, 6) for c in p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _frac_params(params):
    """Every parameter of fracture_mesh, read and checked - nothing is written by it."""
    names = params.get("objects")
    if params.get("object") is not None and names is not None:
        raise MifOpError("pass object OR objects, not both. NOTHING was created.")
    if names is None:
        names = [take(params, "object", required=True, kind=str)]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise MifOpError("objects must be a non-empty list of object names, got %r. NOTHING was created." % (names,))
    objs = []
    for n in dict.fromkeys(names):
        obj = bpy.data.objects.get(n)
        if obj is None:
            raise MifOpError("no object named '%s'. NOTHING was created." % n)
        if obj.type != "MESH":
            raise MifOpError("'%s' is a %s - only a mesh has a volume to fracture. NOTHING was created."
                             % (n, obj.type))
        objs.append(obj)
    source = str(take(params, "source", default="RANDOM", kind=str)).upper()
    if source not in _FRAC_SOURCES:
        raise MifOpError("source must be one of %s, got %r. NOTHING was created."
                         % (", ".join(_FRAC_SOURCES), params.get("source")))
    raw = params.get("sourceLimit", 100)
    if isinstance(raw, dict):
        unknown = [k for k in raw if k not in {o.name for o in objs}]
        if unknown:
            raise MifOpError("sourceLimit names %s, which is not among the objects fractured (%s). NOTHING was "
                             "created." % (", ".join(unknown), ", ".join(o.name for o in objs)))
        limits = {o.name: raw.get(o.name, 100) for o in objs}
    else:
        limits = {o.name: raw for o in objs}
    for name, lim in limits.items():
        if isinstance(lim, bool) or not isinstance(lim, int) or not 2 <= lim <= 5000:
            raise MifOpError("sourceLimit for '%s' must be a whole number 2-5000 (Blender's own fracture caps it "
                             "at 5000), got %r. NOTHING was created." % (name, lim))
    points = params.get("points")
    if source == "POINTS":
        if not isinstance(points, list) or len(points) < 2:
            raise MifOpError("source POINTS needs points: at least two [x, y, z] in world space, got %r. NOTHING "
                             "was created." % (points,))
        if not all(isinstance(p, list) and len(p) == 3 for p in points):
            raise MifOpError("every point is [x, y, z], got %r. NOTHING was created." % (points[:4],))
        points = [finite_floats(p, "points[%d]" % i) for i, p in enumerate(points)]
    elif points is not None:
        raise MifOpError("points is read only with source POINTS (source is %s). NOTHING was created." % source)
    noise = take_float(params, "noise", default=0.0)
    if not 0.0 <= noise <= 1.0:
        raise MifOpError("noise must be 0-1 (a fraction of the object's size), got %g. NOTHING was created." % noise)
    scale = params.get("cellScale", [1.0, 1.0, 1.0])
    scale = finite_floats(scale, "cellScale") if isinstance(scale, list) else None
    if not scale or len(scale) != 3 or min(scale) <= 0.0:
        raise MifOpError("cellScale must be [x, y, z], each above 0, got %r. NOTHING was created."
                         % (params.get("cellScale"),))
    margin = take_float(params, "margin", default=0.0)
    if margin < 0.0:
        raise MifOpError("margin is the gap between neighbouring cells and cannot be negative, got %g. NOTHING "
                         "was created." % margin)
    rec = take_int(params, "recursion", default=0)
    rec_lim = take_int(params, "recursionSourceLimit", default=8)
    clamp = take_int(params, "recursionClamp", default=250)
    chance = take_float(params, "recursionChance", default=0.25)
    if not 0 <= rec <= 8 or not 2 <= rec_lim <= 5000 or clamp < 0 or not 0.0 < chance <= 1.0:
        raise MifOpError("recursion must be 0-8, recursionSourceLimit 2-5000, recursionClamp 0 or more (0 = no "
                         "clamp) and recursionChance above 0 and at most 1 - got %r, %r, %r, %r. NOTHING was created."
                         % (rec, rec_lim, clamp, chance))
    mat = take_int(params, "materialIndex", default=0)
    for obj in objs:
        top = max(len(obj.data.materials) - 1, 0)
        if not 0 <= mat <= top:
            raise MifOpError("materialIndex %d is not a material slot of '%s' (it has %d). NOTHING was created."
                             % (mat, obj.name, len(obj.data.materials)))
    rb = params.get("rigidBody")
    if rb is not None:
        if not isinstance(rb, dict):
            raise MifOpError("rigidBody is a dict of add_rigid_body settings ({type, collisionShape, mass, ...}), got "
                             "%r. NOTHING was created." % (rb,))
        bad = sorted(set(rb) - _FRAC_RB_KEYS)
        if bad:
            raise MifOpError("rigidBody: unknown key(s) %s. Accepted: %s. NOTHING was created."
                             % (", ".join(bad), ", ".join(sorted(_FRAC_RB_KEYS))))
        if str(rb.get("type", "ACTIVE")).upper() not in ("ACTIVE", "PASSIVE"):
            raise MifOpError("rigidBody.type is ACTIVE or PASSIVE, got %r. NOTHING was created." % (rb.get("type"),))
        if str(rb.get("collisionShape", "CONVEX_HULL")).upper() not in _FRAC_RB_SHAPES:
            raise MifOpError("rigidBody.collisionShape is one of %s, got %r. NOTHING was created."
                             % (", ".join(_FRAC_RB_SHAPES), rb.get("collisionShape")))
    return {"objs": objs, "source": source, "limits": limits, "points": points, "noise": noise, "scale": scale,
            "margin": margin, "rec": rec, "recLimit": rec_lim, "clamp": clamp, "chance": chance, "mat": mat,
            "rb": rb, "seed": take_int(params, "seed", default=0),
            "hide": take_bool(params, "hideOriginal", default=False)}


def _frac_object(obj, cfg, rng):
    """(cells [(bm, volume)], seeds, source volume, recursed) for one object - all in memory, nothing written."""
    from mathutils import Vector   # noqa: PLC0415
    bm = _frac_world_bmesh(obj)
    open_edges = sum(1 for e in bm.edges if e.is_boundary)
    if open_edges or not bm.faces:
        bm.free()
        raise MifOpError("'%s' is not closed (%d open edge(s)) - a cell of an open surface has no inside. Close it "
                         "first (fill_holes, or a Solidify modifier). NOTHING was created." % (obj.name, open_edges))
    src_vol = bm.calc_volume(signed=True)
    seeds = _frac_seeds(obj, bm, cfg["source"], cfg["limits"][obj.name], cfg["points"], cfg["noise"], rng)
    if len(seeds) < 2:
        bm.free()
        raise MifOpError("'%s' got %d seed point(s) from source %s - a fracture needs at least two. NOTHING was "
                         "created." % (obj.name, len(seeds), cfg["source"]))
    sx, sy, sz = cfg["scale"]
    centre = sum((v.co for v in bm.verts), Vector()) / len(bm.verts)

    def squash(co, k):
        return Vector((centre.x + (co.x - centre.x) * sx ** k, centre.y + (co.y - centre.y) * sy ** k,
                       centre.z + (co.z - centre.z) * sz ** k))
    stretched = (sx, sy, sz) != (1.0, 1.0, 1.0)
    if stretched:
        for v in bm.verts:
            v.co = squash(v.co, -1)
        seeds = [squash(p, -1) for p in seeds]
    cells = _frac_voronoi(bm, seeds, cfg["margin"], cfg["mat"])
    bm.free()
    recursed = 0
    for _level in range(cfg["rec"]):
        if cfg["clamp"] and len(cells) >= cfg["clamp"]:
            break
        cells.sort(key=lambda c: -c[1])
        pick = max(1, int(round(cfg["chance"] * len(cells))))
        nxt = []
        for k, (cb, vol) in enumerate(cells):
            if k < pick and not (cfg["clamp"] and len(cells) - k + len(nxt) >= cfg["clamp"]):
                sub_seeds = _frac_random_inside(cb, cfg["recLimit"], rng)
                subs = _frac_voronoi(cb, sub_seeds, cfg["margin"], cfg["mat"]) if len(sub_seeds) >= 2 else []
                if len(subs) >= 2:
                    cb.free()
                    nxt.extend(subs)
                    recursed += 1
                    continue
                for sb, _v in subs:
                    sb.free()
            nxt.append((cb, vol))
        cells = nxt
    if stretched:
        import bmesh   # noqa: PLC0415
        out = []
        for cb, _vol in cells:
            for v in cb.verts:
                v.co = squash(v.co, 1)
            bmesh.ops.recalc_face_normals(cb, faces=cb.faces[:])
            out.append((cb, cb.calc_volume(signed=True)))
        cells = out
    if len(cells) < 2:
        for cb, _v in cells:
            cb.free()
        raise MifOpError("'%s' gave %d cell(s) from %d seed(s) - every other seed's cell fell outside the mesh. "
                         "NOTHING was created." % (obj.name, len(cells), len(seeds)))
    return cells, len(seeds), src_vol, recursed


def op_fracture_mesh(params):
    """Fracture closed meshes into Voronoi cells - native bmesh, no add-on needed.

    THE GAP. Blender's Cell Fracture is an optional extension from 4.2 (bundled up to 4.1) and is not in the PyPI bpy
    builds at all, so nothing here could break a wall or a brick into pieces. This cuts each object by the bisector
    plane between every pair of seed points (a Voronoi cell per seed), caps each cut, and makes one object per cell.

    params:
      object (str) | objects [names]  closed meshes; the EVALUATED mesh (modifiers applied) in world space
      source           RANDOM (default: uniform points inside) | VERTS (own vertices) | PARTICLES (own particles) |
                       POINTS (points [[x,y,z]...] in world space)
      sourceLimit      2-5000 seeds per object (default 100), or {objectName: n} - a cell count for that object's size
      noise (0-1)      seed jitter, a fraction of the object's size;  seed (int) for the random draws
      cellScale [x,y,z]  stretch the cells along each axis (Voronoi under that scaled distance)
      margin           gap between neighbouring cells, in world units
      recursion (0-8), recursionSourceLimit (8), recursionClamp (250, 0 = none), recursionChance (0.25): each level
                       re-fractures that fraction of the cells, largest first, until the clamp is reached
      materialIndex    the material slot the cut faces take (the source's slots are copied onto every cell)
      collection       where the cells go (default '<first object>_cells'); made under the scene when missing
      hideOriginal     hide the source in the viewport and the render
      rigidBody {...}  add_rigid_body's settings, applied to every cell (type, collisionShape, mass, density, ...)

    Reply: per object the cell count, the smallest and largest cell, the source's volume against the cells' sum, and
    the cells that are not manifold.
    """
    reject_unknown(params, _FRAC_KEYS, "fracture_mesh", {
        "removeOriginal": "not offered: the source is kept so a fracture can be undone by deleting the collection - "
                          "hideOriginal hides it; delete_object removes it.",
        "sourceObjects": "child objects are fractured by naming them in objects.",
    })
    cfg = _frac_params(params)
    import random   # noqa: PLC0415
    rng = random.Random(cfg["seed"])
    sc = bpy.context.scene
    cname = take(params, "collection", default=cfg["objs"][0].name + "_cells", kind=str)
    coll = bpy.data.collections.get(cname)
    if coll is not None and coll != sc.collection and coll not in sc.collection.children_recursive:
        raise MifOpError("collection '%s' exists but is not in this scene, so cells put there would not be seen. "
                         "Link it into the scene or name another. NOTHING was created." % cname)
    work = []
    try:
        for obj in cfg["objs"]:
            work.append((obj,) + _frac_object(obj, cfg, rng))
    except MifOpError:
        for row in work:
            for cb, _v in row[1]:
                cb.free()
        raise

    # ---- writes start here
    from mathutils import Vector   # noqa: PLC0415
    made_coll = coll is None
    if made_coll:
        coll = bpy.data.collections.new(cname)
        sc.collection.children.link(coll)
    rows, created = [], []
    for obj, cells, n_seeds, src_vol, recursed in work:
        cell_rows = []
        for k, (cb, vol) in enumerate(cells):
            manifold = all(e.is_manifold for e in cb.edges)
            centre = sum((v.co for v in cb.verts), Vector()) / len(cb.verts)
            for v in cb.verts:
                v.co -= centre
            me = bpy.data.meshes.new("%s_cell_%03d" % (obj.name, k))
            cb.to_mesh(me)
            n_faces = len(cb.faces)
            cb.free()
            for m in obj.data.materials:
                me.materials.append(m)
            cell = bpy.data.objects.new(me.name, me)
            cell.location = centre
            coll.objects.link(cell)
            created.append(cell.name)
            cell_rows.append({"name": cell.name, "volume": float("%.6g" % vol), "faces": n_faces, "manifold": manifold,
                              "location": rnd(centre, 5)})
        small = min(cell_rows, key=lambda r: r["volume"])
        large = max(cell_rows, key=lambda r: r["volume"])
        total = sum(r["volume"] for r in cell_rows)
        rows.append({"object": obj.name, "seeds": n_seeds, "cellCount": len(cell_rows), "recursedCells": recursed,
                     "smallestCell": {"name": small["name"], "volume": small["volume"]},
                     "largestCell": {"name": large["name"], "volume": large["volume"]},
                     "sourceVolume": float("%.7g" % src_vol), "cellsVolume": float("%.7g" % total),
                     "nonManifoldCells": [r["name"] for r in cell_rows if not r["manifold"]],
                     "cells": cell_rows[:500]})       # cellCount above 500 says the list was cut
    rb_out = None
    if cfg["rb"] is not None:
        try:
            rb_out = op_add_rigid_body(dict(cfg["rb"], objects=list(created)))
        except MifOpError as exc:
            for name in created:
                ob = bpy.data.objects.get(name)
                if ob is not None:
                    me = ob.data
                    remove_objects([ob])
                    if me is not None and me.users == 0:
                        bpy.data.meshes.remove(me)
            if made_coll:
                bpy.data.collections.remove(coll)
            raise MifOpError("rigidBody was refused by add_rigid_body: %s - the %d cell(s) made were removed again, "
                             "so NOTHING was left behind." % (exc, len(created)))
    hidden = []
    if cfg["hide"]:
        for obj in cfg["objs"]:
            try:
                obj.hide_set(True)
            except RuntimeError:            # not in this view layer: the global viewport flag instead
                obj.hide_viewport = True
            obj.hide_render = True
            hidden.append(obj.name)
    return {"objects": rows, "cellCount": len(created), "created": created[:2000],
            "collection": coll.name, "collectionCreated": made_coll, "requestedName": cname,
            "nameWasSuffixed": coll.name != cname, "hiddenOriginals": hidden,
            "rigidBodies": None if rb_out is None else {"count": len(created),
                                                        "type": str(cfg["rb"].get("type", "ACTIVE")).upper()}}


OPS = {
    "step_simulation": op_step_simulation,
    "add_rigid_body": op_add_rigid_body,
    "add_cloth": op_add_cloth,
    "add_soft_body": op_add_soft_body,
    "add_rigid_body_constraint": op_add_rigid_body_constraint,
    "add_fluid": op_add_fluid,
    "add_dynamic_paint": op_add_dynamic_paint,
    "add_collision": op_add_collision,
    "bake_physics": op_bake_physics,
    "collect_vdb_frames": op_collect_vdb_frames,
    "physics_info": op_physics_info,
    "fracture_mesh": op_fracture_mesh,
    "set_physics_world": op_set_physics_world,
}
