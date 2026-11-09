"""Lights and cameras: the two highest-value things this addon could not do without run_python.

WHY THESE TWO FIRST. A survey on 2026-09-01 found EIGHT capability families with no typed op at all
- lights, cameras, keyframes, particles, physics, rendering, world, and authoring geometry-node
trees. They are not worth the same. A scene with no light renders black and a scene with no camera
cannot be rendered at all, so these two gate everything else; particles do not gate anything.

WHY A TYPED OP RATHER THAN "just use run_python". run_python reaches all of this today, and it is
the wrong answer for three specific reasons rather than on principle:

  * IT IS AN ARBITRARY-CODE-EXECUTION SWITCH. The addon preference is literally labelled that, and
    a user who turns it off loses every light and camera call with it. A typed op keeps working.
  * IT IS ABSENT WHEN THE ADDON IS IMPORTED RATHER THAN INSTALLED. _prefs() returns None off
    sys.path, so run_python refuses - which is why run_blender_suites.py carries explicit code to
    force the hatch open before its suites run.
  * IT REPORTS NOTHING. reject_unknown catches a misspelled key at the door; exec fails somewhere
    obscure at runtime. And a typed op can read its postcondition back, which is the discipline the
    rest of this bridge is built on.

WHAT IS DELIBERATELY NOT HERE. No keyframing, no render trigger, no world/HDRI. Those are separate
entries and each is its own engine surface; doing them badly alongside these two would be worse
than leaving them recorded.
"""
import math
import os
import time

import bpy
import mathutils

from .ops_common import (remove_objects, INFLUENCE_BOOL, INFLUENCE_INT, INFLUENCE_MAP, MifOpError, PANO_FIELDS, basis_space,
                         camera_readback, check_axis_dict, exclude_rows, finite_floats,
                         get_object, light_readback, limit_rows, object_info, pano_holder,
                         refuse_unsupported_shadow, reject_unknown, rnd, rotation_euler_values, rotation_mode_note,
                         selection_restore, selection_snapshot, shadow_attr, take, take_bool, take_float, take_int,
                         write_orientation, write_rna_property, write_rotation)


# Blender's own enum, read off the RNA rather than remembered - the same discipline ops_create uses
# for primitive sizing kwargs, and for the same reason: a hardcoded list goes stale silently.
_LIGHT_KEYS = {
    "name", "type", "lightType", "kind", "location", "rotation",
    "energy", "power", "color", "radius", "size", "sizeY", "shape",
    "spotAngle", "spotBlend", "angle", "shadow", "diffuseFactor", "specularFactor",
    "temperature", "useTemperature",
}

_CAMERA_KEYS = {
    "name", "location", "rotation", "lens", "focalLength", "sensorWidth",
    "clipStart", "clipEnd", "type", "orthoScale", "dofDistance", "fStop",
    "lookAt", "makeActive", "shiftX", "shiftY",
    # BOTH WERE READ AND NOT WRITEABLE. object_info reports dofFocusObject and angle; nothing could
    # set either, so a camera could be DESCRIBED in more detail than it could be built.
    "focusObject", "fieldOfView",
}


def _vec3(params, key, default, verb="created"):
    """Parse a vector, or refuse. CALL THIS BEFORE ANYTHING EXISTS.

    IT REFUSES, so where it is called decides whether the refusal is true. Until 2026-09-04 all four
    ops here called it AFTER bpy.data.*.new and objects.link, so a malformed `location` left a light
    or a camera sitting in the caller's scene and then said "NOTHING was created". The two setters
    were worse: they called it below a comment reading "COMMIT. Nothing below can refuse", one line
    after the type had been applied, so set_light({type:"SPOT", location:"bad"}) retyped the light
    and then denied doing anything.

    That sentence is what every refusal in MifBridge is held to and callers are told to trust, so a
    false one is worse than an ordinary bug. Every call site now parses up front, which is the shape
    op_set_viewport_view adopted for exactly this reason.

    `verb` exists because "NOTHING was created" is the wrong noun for a setter - it is true and
    irrelevant, and it hides that something WAS changed.
    """
    v = params.get(key)
    if v is None:
        return tuple(default)
    if isinstance(v, dict):
        # AT LEAST ONE OF x/y/z, AND NOTHING ELSE. A dict was read with .get(..., default) for
        # each axis, so {"mif":"typo"} returned the DEFAULT vector and the call reported
        # success - a misspelled key silently placed the object at the origin, or left it where
        # it was, and every field in the response agreed. Partial dicts stay legal ({"z": 2} is
        # a useful thing to write); a dict that names none of them is a typo, not a request.
        check_axis_dict(v, key, ("x", "y", "z"))
        return tuple(finite_floats([v.get("x", default[0]), v.get("y", default[1]),
                                    v.get("z", default[2])], key))
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return tuple(finite_floats(v, key))
    raise MifOpError("'%s' must be {x,y,z} or a 3-list, got %r. NOTHING was %s."
                     % (key, v, verb))


def _valid_light_types():
    return {i.identifier for i in bpy.types.Light.bl_rna.properties["type"].enum_items}


# THE PER-TYPE KEY MAP, SHARED. create_light validated these against the type being created and
# set_light has to validate them against the type the light will BE after the call - the same rule
# asked twice, so it lives once. A second copy is how allowEditConst got past one guard and not the
# other in this repo on 2026-09-03.
_MISPLACED_LIGHT_KEYS = (
    (("spotAngle", "spotBlend"), ("SPOT",), "spotAngle/spotBlend"),
    (("size", "sizeY", "shape"), ("AREA",), "size/sizeY/shape"),
    (("angle",), ("SUN",), "angle"),
    (("radius",), ("POINT", "SPOT"), "radius (the soft-shadow size)"),
)


def _refuse_misplaced_light_keys(params, kind, verb):
    """Raise if a per-type key was given for a light that is not that type.

    Called BEFORE anything is created or written, in both create_light and set_light, because a
    typo must not leave a stray object or a half-applied change. `verb` completes the sentence so
    the message says NOTHING was created or NOTHING was changed as appropriate - the two are
    different claims and this codebase holds both to being literally true.
    """
    for keys, wants, label in _MISPLACED_LIGHT_KEYS:
        present = [k for k in keys if k in params]
        if present and kind not in wants:
            raise MifOpError("%s only applies to a %s light and this one is %s (%s given). "
                             "NOTHING was %s."
                             % (label, " or ".join(wants), kind, ", ".join(present), verb))


def _look_at_euler(frm, to):
    """Euler that points a Blender camera's -Z at `to`, with +Y up.

    A camera looks down its LOCAL -Z, not +X and not +Z, which is the single thing that makes
    hand-written camera aiming wrong on the first try. Derived here rather than left to the caller.
    """
    dx, dy, dz = (to[0] - frm[0]), (to[1] - frm[1]), (to[2] - frm[2])
    horiz = math.hypot(dx, dy)
    if horiz == 0.0 and dz == 0.0:
        raise MifOpError("lookAt is the same point as the camera location, so there is no "
                         "direction to face. NOTHING was created.")
    # DERIVED, not guessed, because the first version was off by pi and aimed the camera 166
    # degrees away from its target while returning a perfectly plausible euler. For XYZ order the
    # world matrix is Rz(rz) . Rx(rx), so:
    #     forward = Rz(rz) . Rx(rx) . (0,0,-1) = (-sin rz . sin rx,  cos rz . sin rx,  -cos rx)
    # Matching that against d/|d| gives rx = atan2(horiz, -dz) - which was right - and
    #     -sin rz = dx/horiz,  cos rz = dy/horiz   ->   rz = atan2(-dx, dy)
    # atan2(dy, dx) + pi/2 is NOT that; it is that plus pi, which points exactly backwards.
    return (math.atan2(horiz, -dz), 0.0, math.atan2(-dx, dy))


def _apply_light_temperature(data, params, checked):
    """Colour temperature in Kelvin, shared by create_light and set_light.

    A gaffer says "3200 K tungsten" or "6500 K daylight", not "(1.0, 0.72, 0.42)". These ops could
    take an RGB colour and nothing else, so every caller wanting a real light had to convert Kelvin
    to linear RGB by hand - a blackbody curve, not a lookup anyone should be retyping, and getting
    it wrong is invisible until the render looks off.

    ONE IMPLEMENTATION, CALLED TWICE, AND THAT IS THE POINT. op_create_light and op_set_light each
    carry their own copy of the energy/colour/spot/size/shape/angle assignments - two parallel
    systems for one job. Adding temperature to only the first is exactly what happened on the way
    here: create_light took 3200 K and set_light accepted the same key, returned ok:true, and did
    nothing with it, INCLUDING not refusing a temperature of "warm". Extracting the whole property
    block is the real fix and is not this change; extracting THIS one at least means the next
    version of it cannot drift between the two.

    NOT ON EVERY BLENDER. bpy.types.Light.use_temperature and .temperature are recent - present on
    5.0.1, confirmed by asking the running Blender rather than by reading a changelog - and this
    addon supports 3.6 upward. Where they are absent the request is REFUSED BY NAME rather than
    silently ignored: a caller who asked for 3200 K and got a white light with ok:true has been told
    something false.

    COLOUR AND TEMPERATURE ARE NOT ALTERNATIVES in Blender. With use_temperature on, the blackbody
    result is multiplied by the light's colour, so colour becomes a TINT. Both are accepted together.
    """
    temp, want_temp = checked
    if temp is None and want_temp is None:
        return
    if temp is not None:
        data.temperature = temp
        data.use_temperature = True if want_temp is None else bool(want_temp)
    else:
        data.use_temperature = bool(want_temp)


def _check_light_temperature(params):
    """Everything about temperature that can REFUSE, and nothing that writes.

    IT USED TO REFUSE FROM THE MIDDLE OF THE WRITES, which is the same defect this project has been
    fixing in the UE handlers all week and I put it here myself. _apply_light_props writes energy,
    colour, spot angle and blend, size, shape, angle and shadow BEFORE calling the temperature code,
    so a caller on a Blender without bpy.types.Light.use_temperature - anything before 4.x - who
    sent `energy` and `temperature` together got the energy applied and then a refusal ending
    "NOTHING was changed". Seven properties had changed.

    Splitting the refusals out means _apply_light_props can ask all of them first and write nothing
    until they all pass. audit_mutate_then_deny.py found it; it reported seven pairs across
    _apply_light_props and op_set_light, which is how many writes sat above the one refusal.

    Returns (temp, want_temp), both None when the caller asked for neither.
    """
    temp = take_float(params, "temperature", default=None)
    want_temp = take_bool(params, "useTemperature", default=None) if "useTemperature" in params else None
    if temp is None and want_temp is None:
        return None, None
    # ASKED OF THE RNA, NOT OF AN INSTANCE AND NOT WITH hasattr. Measured on three installs:
    # hasattr(bpy.types.Light, "use_temperature") is False even on 5.0 where the property exists, so
    # that spelling can never pass - a first attempt at this hoist used it and made create_light
    # refuse on every build. Membership in bl_rna.properties agrees with the instance on 3.6, 4.4
    # and 5.0, and needs no light to exist, which is what lets op_create_light run this BEFORE
    # bpy.data.lights.new() instead of after. _enum_ids introspects RNA the same way.
    if "use_temperature" not in bpy.types.Light.bl_rna.properties.keys():
        raise MifOpError(
            "this Blender (%s) has no light colour temperature - bpy.types.Light gained "
            "use_temperature and temperature in a later version. Set `color` as linear RGB "
            "instead. NOTHING was changed." % bpy.app.version_string)
    if temp is not None:
        # Blender's own UI soft range is 800..12000 K. Outside it the blackbody curve still
        # evaluates, but zero or a negative is not a temperature at all and renders black.
        if temp <= 0.0:
            raise MifOpError(
                "temperature is in KELVIN and must be positive - %.4g is not a colour, it renders "
                "black. Tungsten is about 3200, daylight about 6500. NOTHING was changed." % temp)
    return temp, want_temp


def _plan_light_props(params):
    """Every value _apply_light_props will write, parsed and validated. Writes NOTHING.

    Split out so a refusal can happen before anything exists: op_create_light calls it BEFORE
    bpy.data.lights.new(), which is what stops a malformed energy leaving a light in the scene. The
    matrix counted 71 such refusals across four builds the day this op's payload stopped being
    {"type": "POINT"}.

    ONE list, and _apply_light_props calls it rather than repeating it. This file already records
    what two copies cost: create_light and set_light "used to carry SEPARATE copies of this", and the
    first time somebody added a key, set_light went on accepting it and silently doing nothing.

    Safe to call twice on the same params because take() READS a key rather than popping it.
    """
    colour = params.get("color")
    if colour is not None and not isinstance(colour, (list, tuple)):
        # BEFORE THE SLICE, because colour[:3] on a dict raises "unhashable type: 'slice'" and that
        # is a raw TypeError one step ahead of the guard meant to name the problem. finite_floats
        # can refuse a bad ELEMENT; it never gets the chance if indexing the container throws first.
        raise MifOpError(
            "'color' must be a list of 3 numbers like [0.9, 0.85, 1.0], got %s. NOTHING was created."
            % type(colour).__name__)
    if colour is not None:
        # finite_floats, NOT a bare float(). The line this replaced was tuple(float(c) for c in
        # col[:3]) inside the apply path; moving it out of the try that wrapped it turned a
        # malformed colour from a refusal into a raw ValueError - nine of them across the builds -
        # because float("x") raises and float("nan") quietly succeeds. finite_floats refuses by
        # naming the element that is wrong.
        colour = tuple(finite_floats(colour[:3], "color"))
    shape = take(params, "shape", default=None, kind=str)
    return {
        "energy": take_float(params, "energy", "power", default=None),
        "color": colour,
        "spotAngle": take_float(params, "spotAngle", default=None),
        "spotBlend": take_float(params, "spotBlend", default=None),
        "size": take_float(params, "size", default=None),
        "sizeY": take_float(params, "sizeY", default=None),
        "shape": str(shape).upper() if shape else None,
        "angle": take_float(params, "angle", default=None),
        "hasRadius": "radius" in params,
        "radius": take_float(params, "radius", default=0.1) if "radius" in params else None,
        "hasShadow": "shadow" in params,
        "shadow": take_bool(params, "shadow", default=True) if "shadow" in params else None,
        "diffuseFactor": take_float(params, "diffuseFactor", default=None),
        "specularFactor": take_float(params, "specularFactor", default=None),
    }


def _apply_light_props(data, plan, params, checked):
    """Every light property both ops accept, applied once.

    op_create_light and op_set_light used to carry SEPARATE copies of this, and the cost showed up
    the first time anyone added a key: colour temperature went into create_light on 2026-09-06 and
    set_light went on accepting `temperature`, returning ok:true and doing nothing with it -
    including not refusing a temperature of "warm". Nothing failed. The response said ok.

    WHAT DELIBERATELY STAYS OUT, because the two ops genuinely differ there and merging it would be
    the actual bug this refactor is supposed to prevent:

      colour VALIDATION   set_light checks it ABOVE its commit point and says "NOTHING was changed";
                          create_light checks at the point of use. Only the ASSIGNMENT is shared.
      the type re-fetch   assigning data.type swaps the ID's RNA subclass, so set_light must rebind
                          `data = obj.data` or `data.size` raises on a light that IS an area by
                          then. create_light never retypes.
      location/rotation   create places the object; set moves it, and only if asked.

    create_light also used to wrap the per-type keys in an `_only(...)` guard. That was already
    vestigial - its own comment said the type mismatch had been refused earlier and it only asked
    "is there work to do" - so dropping it changes nothing and is what makes the two bodies the
    same code rather than merely similar.

    Guarded by test_light_op_parity.py, which sets every shared key through BOTH ops and requires
    the datablocks to come out identical. It was scored against a deliberate break before being
    believed.
    """
    # EVERY REFUSAL FIRST, THEN EVERY WRITE, and `checked` is how a caller proves it already did.
    #
    # The temperature check is the only thing in this function that can raise, and it used to run
    # AFTER every write below - so a caller on a Blender without bpy.types.Light.use_temperature who
    # sent energy and temperature together got the energy applied and then "NOTHING was changed".
    #
    # Hoisting it here fixed the behaviour and audit_mutate_then_deny still reported it, correctly:
    # the audit is static, it sees a call that can raise below a write in op_set_light, and it
    # cannot know an identical check ran higher. Threading the result through means the apply path
    # CANNOT raise rather than merely never reaching the raise - which is also what stops a future
    # reorder of the caller quietly reintroducing it.
    #
    # `checked` is REQUIRED rather than defaulted, and that is the difference that finally satisfied
    # the audit. With a default this function still CONTAINED a call to the checker, so the
    # call-graph walk kept finding a raise below a write in op_set_light - and was right to, because
    # a static reader cannot know the guard never fires. Requiring it removes the path rather than
    # arguing about it, and a caller that forgets gets a TypeError at import-test time.

    # `plan` IS PASSED IN, NOT DERIVED HERE, and that is the same distinction `checked` above
    # records. Deriving it inside left a parse - and therefore a possible refusal - below the writes
    # in op_set_light, and audit_mutate_then_deny kept reporting all three, correctly: it is static
    # and cannot know an identical parse ran higher. Taking the plan as an argument means this
    # function contains no parse at all, which a static reader can verify rather than take on trust.
    #
    # What it fixes in behaviour: this used to read one key and write it, then read the next - so a
    # malformed specularFactor left energy, colour, radius and shadow already applied, and in
    # op_create_light left the light itself behind.

    if plan["energy"] is not None:
        data.energy = plan["energy"]
    if plan["color"] is not None:
        data.color = plan["color"]
    if plan["spotAngle"] is not None:
        data.spot_size = plan["spotAngle"]
    if plan["spotBlend"] is not None:
        data.spot_blend = plan["spotBlend"]
    if plan["size"] is not None:
        data.size = plan["size"]
    if plan["sizeY"] is not None:
        data.size_y = plan["sizeY"]
    if plan["shape"]:
        data.shape = plan["shape"]
    if plan["angle"] is not None:
        data.angle = plan["angle"]
    if plan["hasRadius"]:
        data.shadow_soft_size = plan["radius"]
    if plan["hasShadow"]:
        # The Blender-has-no-such-attribute case is refused up front by both callers, so this lands.
        setattr(data, shadow_attr(data), plan["shadow"])
    _apply_light_temperature(data, params, checked)
    if plan["diffuseFactor"] is not None:
        data.diffuse_factor = plan["diffuseFactor"]
    if plan["specularFactor"] is not None:
        data.specular_factor = plan["specularFactor"]


_PROBE_KEYS = {"name", "type", "location", "rotation", "influenceDistance", "influenceType",
               "falloff", "clipStart", "clipEnd", "collection"}

# BLENDER RENAMED ALL THREE PROBE TYPES AT 4.2, and each group here is one probe with two names.
# 3.6 calls them CUBE / PLANAR / GRID; 4.2 and later call them SPHERE / PLANE / VOLUME. A caller
# should not have to branch on bpy.app.version to place a reflection probe, which is the same
# argument PRINCIPLED_ALIASES makes about the socket names Blender renamed at 4.0.
_PROBE_SYNONYMS = (
    ("SPHERE", "CUBE", "CUBEMAP"),
    ("PLANE", "PLANAR"),
    ("VOLUME", "GRID"),
)


def _probe_new_enum():
    """The type names `lightprobes.new()` ACCEPTS - which are not always the ones it reports.

    THE PROPERTY ENUM AND THE CREATION FUNCTION DISAGREE ON 3.6. `LightProbe.bl_rna` says the types
    are CUBEMAP, PLANAR, GRID; `bpy.data.lightprobes.new()` accepts CUBE, PLANAR, GRID - and passing
    CUBE produces a probe whose `.type` reads back as CUBEMAP. Measured, on 3.6.23 against 4.2.17,
    4.4.0 and 5.0.1 where the two agree.

    So "read the enum off the running Blender" is not enough on its own: it has to be read off the
    THING BEING CALLED. Reading the property enum and passing that to new() is a TypeError on 3.6,
    which is exactly what the first attempt at this got.
    """
    fn = bpy.data.lightprobes.bl_rna.functions["new"]
    return [i.identifier for i in fn.parameters["type"].enum_items]


def op_create_light_probe(params):
    """Create a light probe - the EEVEE objects that give a scene reflections and bounced light.

    THE ADDON COULD MAKE EVERY OTHER LIGHTING OBJECT AND NOT THIS ONE. create_light makes lamps,
    set_light_* tunes them, and a probe is what makes a lamp's light appear in a reflection or fill
    a room indirectly. Of Blender's 17 object types this addon creates eight, and LIGHT_PROBE was the
    one missing from an area the spec header names outright.

    THE TYPE NAMES CHANGED AT 4.2 AND ARE ALIASED HERE, so one script places a probe on 3.6 and on
    5.0 without branching: SPHERE and CUBE and CUBEMAP mean the same probe, as do PLANE/PLANAR and
    VOLUME/GRID. The name actually passed is whichever this build's `new()` accepts, and the response
    reports BOTH what was asked for and what Blender called it - on 3.6 those differ, because CUBE
    goes in and CUBEMAP comes back.

    WHAT IS DELIBERATELY NOT HERE. `intensity` and `grid_resolution_*` exist only on 3.6, and
    `data_display_size` only on 4.2+. Exposing a parameter that works on one build and refuses on
    another is worse than not having it, and the five settings below are present on 3.6.23, 4.2.17
    LTS, 4.4.0 and 5.0.1 alike - measured, not assumed.

    A PROBE IS NOT A LIGHT. It emits nothing; it captures the surroundings so that EEVEE can show
    them in reflections and indirect light. Placing one where nothing reflective sits is a common
    way to conclude it "does not work".

    params:
      name (str)                 Blender appends .001 on a collision; the response reports the real
                                 name and whether it was suffixed
      type (str)                 SPHERE|CUBE|CUBEMAP, PLANE|PLANAR, VOLUME|GRID - default SPHERE
      location / rotation        {x,y,z}; both are read back (rotation in radians)
      collection (str)           link it here instead of the scene's root collection, as create_empty does
      influenceDistance (float)  how far the probe's effect reaches
      influenceType (str)        ELIPSOID or BOX, validated against this build
      falloff (float)            0..1, how the influence fades at the edge
      clipStart / clipEnd        the near and far the probe captures between
    """
    reject_unknown(params, _PROBE_KEYS, "create_light_probe", {
        "intensity": "exists only on Blender 3.6 and is deliberately not exposed. A parameter that "
                     "works on one supported build and refuses on another is worse than not having "
                     "it - the five settings this op does take are on 3.6, 4.2, 4.4 and 5.0 alike.",
        "gridResolution": "see 'intensity' - the grid_resolution_* fields are 3.6-only.",
        "resolution": "see 'intensity'.",
        "dataDisplaySize": "the opposite version problem - 4.2 and later only, so it is left out "
                           "for the same reason.",
        "energy": "a probe EMITS nothing. It captures its surroundings so EEVEE can show them in "
                  "reflections and indirect light; create_light is what makes something bright.",
        "color": "see 'energy' - a probe has no colour of its own.",
    })
    accepted = _probe_new_enum()
    wanted_type = str(take(params, "type", default="SPHERE", kind=str)).upper()

    resolved = None
    if wanted_type in accepted:
        resolved = wanted_type
    else:
        for group in _PROBE_SYNONYMS:
            if wanted_type in group:
                resolved = next((g for g in group if g in accepted), None)
                break
    if resolved is None:
        raise MifOpError(
            "unknown light probe type '%s' for this Blender (%s). It accepts: %s - and the other "
            "spellings this op understands are %s, because Blender renamed all three at 4.2. "
            "NOTHING was created."
            % (wanted_type, bpy.app.version_string, ", ".join(accepted),
               "; ".join("/".join(g) for g in _PROBE_SYNONYMS)))

    # PARSED BEFORE ANYTHING EXISTS, the ordering create_light's own comment records having to learn.
    want_loc = _vec3(params, "location", (0.0, 0.0, 0.0))
    want_rot = _vec3(params, "rotation", (0.0, 0.0, 0.0))
    dist = take_float(params, "influenceDistance", default=None)
    fall = take_float(params, "falloff", default=None)
    cstart = take_float(params, "clipStart", default=None)
    cend = take_float(params, "clipEnd", default=None)
    inf_type = take(params, "influenceType", default=None, kind=str)
    # A COLLECTION, like create_empty: the probe always went to the scene root (found 2026-10-25). Resolved before
    # anything exists, by the one resolver every create op uses.
    from .ops_create import _resolve_collection   # noqa: PLC0415 - the op modules import ops_common, not each other, at load
    coll = _resolve_collection(params)

    wanted = str(take(params, "name", default="LightProbe", kind=str))
    data = bpy.data.lightprobes.new(name=wanted, type=resolved)
    obj = bpy.data.objects.new(data.name, data)
    coll.objects.link(obj)
    obj.location = want_loc
    obj.rotation_euler = want_rot

    # EACH SETTING CHECKED AGAINST THIS PROBE, because the three types are DIFFERENT CLASSES from
    # 4.2 onward and they do not carry the same fields. Measured on 3.6.23, 4.2.17, 4.4.0 and 5.0.1:
    #
    #   3.6            one class, LightProbe - every setting on every type
    #   4.2 and later  LightProbeSphere has all five; LightProbePlane and LightProbeVolume have
    #                  influence_distance and clip_start and NOTHING ELSE
    #
    # The first version of this op read data.influence_type unconditionally and raised
    # AttributeError on a PLANE - found by testing the other two types rather than only the SPHERE
    # I had probed. A setting this type does not have is REFUSED naming the types that do, which is
    # what set_light_linking does about a feature a BUILD lacks; silently dropping it would leave a
    # caller believing they had set a falloff.
    def _apply(attr, value, label):
        if value is None:
            return
        if not hasattr(data, attr):
            # THE NAME IS READ BEFORE THE DATABLOCK DIES. Building this message after the remove
            # raised "ReferenceError: StructRNA of type LightProbePlane has been removed" - so the
            # refusal was correct and its explanation was a stack trace. Exactly the trap
            # remove_node documents about nodes, met again eight hours later on a different
            # datablock: after bpy.data.*.remove(), the Python reference is dead.
            kind_name, build = data.type, bpy.app.version_string
            remove_objects([obj])
            bpy.data.lightprobes.remove(data)
            raise MifOpError(
                "'%s' is not a setting a %s probe has on this Blender (%s) - from 4.2 each probe "
                "type is its own class, and only the SPHERE/CUBE one carries falloff, clipEnd and "
                "influenceType. NOTHING was created." % (label, kind_name, build))
        if attr == "influence_type":
            allowed = [i.identifier for i in data.bl_rna.properties["influence_type"].enum_items]
            got = str(value).upper()
            if got not in allowed:
                # Same reason as above: read what the message needs before the datablock is gone.
                shown = ", ".join(allowed)
                remove_objects([obj])
                bpy.data.lightprobes.remove(data)
                raise MifOpError("influenceType must be one of %s on this Blender, got %r. NOTHING "
                                 "was created." % (shown, value))
            value = got
        setattr(data, attr, value)

    _apply("influence_type", inf_type, "influenceType")
    _apply("influence_distance", dist, "influenceDistance")
    _apply("falloff", fall, "falloff")
    _apply("clip_start", cstart, "clipStart")
    _apply("clip_end", cend, "clipEnd")

    bpy.context.view_layer.update()
    out = {
        "name": obj.name, "dataName": data.name, "requestedName": wanted,
        "nameWasSuffixed": obj.name != wanted,
        "objectType": obj.type,
        # BOTH, because on 3.6 they differ: CUBE is passed in and CUBEMAP is what Blender calls it.
        "requestedType": wanted_type, "passedType": resolved, "type": data.type,
        "location": [round(float(v), 6) for v in obj.location],
        "rotation": [round(float(v), 6) for v in obj.rotation_euler],
        "collection": coll.name,
        "note": "a probe EMITS nothing - it captures its surroundings so EEVEE can show them in "
                "reflections and indirect light. One placed where nothing reflective sits is the "
                "usual reason a probe looks like it does nothing.",
    }
    # READ BACK ONLY WHAT THIS TYPE HAS, for the same reason: reporting a falloff on a PLANE probe
    # would be inventing a field, and reporting it as null would read as "it is zero".
    for attr, label in (("influence_type", "influenceType"),
                        ("influence_distance", "influenceDistance"),
                        ("falloff", "falloff"), ("clip_start", "clipStart"),
                        ("clip_end", "clipEnd")):
        if hasattr(data, attr):
            v = getattr(data, attr)
            out[label] = v if isinstance(v, str) else round(float(v), 6)
    out["settingsOnThisType"] = sorted(k for k in ("influenceType", "influenceDistance", "falloff",
                                                   "clipStart", "clipEnd") if k in out)
    return out


def op_create_light(params):
    """Create a light. Returns what the light ACTUALLY is, not what was asked for.

    params:
      name (str)             requested name; Blender appends .001 on collision and the response
                             reports the name it really got
      type (str)             POINT | SUN | SPOT | AREA - validated against this Blender's own enum
      location / rotation    {x,y,z}
      energy (float)         watts for POINT/SPOT/AREA, irradiance for SUN
      color [r,g,b]
      radius (float)         soft-shadow size for POINT/SPOT
      size / sizeY / shape   AREA only
      spotAngle / spotBlend  SPOT only, spotAngle in RADIANS
      angle (float)          SUN only, angular diameter in radians
    """
    reject_unknown(params, _LIGHT_KEYS, "create_light")
    kind = str(take(params, "type", "lightType", "kind", default="POINT", kind=str)).upper()
    valid = _valid_light_types()
    if kind not in valid:
        raise MifOpError("unknown light type '%s' for this Blender. Valid: %s. NOTHING was created."
                         % (kind, ", ".join(sorted(valid))))

    # TYPE-SPECIFIC KEYS ARE CHECKED BEFORE THE LIGHT EXISTS. The first version created it and then
    # refused, which is honest - the error said "The light WAS created" - and still leaves a stray
    # object in the scene for a caller who did nothing but make a typo. test_blender_anim's cleanup
    # check found it by refusing to ignore an A_Bad nobody meant to keep. Everywhere else in this
    # addon a refusal means NOTHING was created, and this now matches.
    # Shared with set_light, which asks the same question about the type the light will BE.
    _refuse_misplaced_light_keys(params, kind, "created")
    refuse_unsupported_shadow(params, "created")

    # PARSED BEFORE ANYTHING EXISTS. These used to be read after the light was created AND linked,
    # so a malformed vector left a light in the scene and said "NOTHING was created".
    want_loc = _vec3(params, "location", (0.0, 0.0, 0.0))
    want_rot = _vec3(params, "rotation", (0.0, 0.0, 0.0))
    # AND EVERY OTHER PROPERTY, in the same place and for the same reason. The hoist above covered
    # two keys and left seven parsing inside _apply_light_props, which runs after the light is
    # created AND linked - so a bad energy or radius left a light behind under "NOTHING was
    # created". The result is discarded here; _apply_light_props parses again, which costs nothing
    # because take() does not pop, and one parse list beats two.
    light_plan = _plan_light_props(params)
    # The temperature refusal too, now that it no longer needs a datablock to ask.
    checked_temperature = _check_light_temperature(params)

    snap = selection_snapshot()
    try:
        wanted = str(take(params, "name", default="Light", kind=str))
        data = bpy.data.lights.new(name=wanted,
                                   type=kind)
        obj = bpy.data.objects.new(data.name, data)
        bpy.context.scene.collection.objects.link(obj)

        # Checked BEFORE the object is placed, for the same reason op_set_light does it: the only
        # thing in the property block that can refuse is the temperature, and refusing after the
        # light has been positioned would leave a placed light behind a "NOTHING was changed".
        obj.location = want_loc
        obj.rotation_euler = want_rot

        _apply_light_props(data, light_plan, params, checked_temperature)

        bpy.context.view_layer.update()
        # READ BACK off the datablock, so this reports what the light IS.
        out = {
            "name": obj.name,
            "dataName": data.name,
            "type": data.type,
            "location": rnd(list(obj.matrix_world.to_translation())),
            "rotationEuler": rnd(list(obj.matrix_world.to_euler())),
            "energy": round(float(data.energy), 6),
            "color": rnd(list(data.color)),
        }
        if data.type in ("POINT", "SPOT"):
            out["shadowSoftSize"] = round(float(data.shadow_soft_size), 6)
        if data.type == "SPOT":
            out["spotSize"] = round(float(data.spot_size), 6)
            out["spotBlend"] = round(float(data.spot_blend), 6)
        if data.type == "AREA":
            out["size"] = round(float(data.size), 6)
            out["shape"] = data.shape
        if data.type == "SUN":
            out["angle"] = round(float(data.angle), 6)
        # A NOTE IS NOT A FIELD. This said the name "may differ from what was asked for" and
        # left the caller nothing to test - so a retry after a timeout got Light.001 with prose
        # about it. requestedName and nameWasSuffixed are the answerable form.
        out["inertUnderThisEngine"], out["engineNote"] = _light_factor_engine_note(params)
        out["requestedName"] = wanted
        out["nameWasSuffixed"] = obj.name != wanted
        out["nameNote"] = ("Blender renames on collision rather than failing, so `name` is what the "
                           "object ACTUALLY got - check nameWasSuffixed rather than assuming.")
        return out
    finally:
        selection_restore(snap)


_SET_LIGHT_KEYS = set(_LIGHT_KEYS) | {"object", "light", "energyScale", "stops"}


def _energy_factor(params):
    """The multiplier energyScale or stops asks for, or None. Lights are balanced by stops - a rim made from the fill
    and divided by 3, a key doubled once it moved off axis - and the value had to be read back first (2026-10-05)."""
    scale, stops = params.get("energyScale"), params.get("stops")
    if scale is None and stops is None:
        return None
    if params.get("energy") is not None or params.get("power") is not None:
        raise MifOpError("energy sets the value and energyScale / stops multiply the one the light has - pass one of "
                         "them. NOTHING was changed.")
    if scale is not None and stops is not None:
        raise MifOpError("pass one of energyScale (a multiplier) or stops (2 to the power stops), not both. NOTHING "
                         "was changed.")
    if scale is not None:
        f = take_float(params, "energyScale")
        if not f > 0.0:
            raise MifOpError("energyScale multiplies the energy and must be above 0, got %g. NOTHING was changed." % f)
        return f
    try:
        n = take_float(params, "stops")
    except MifOpError:
        raise MifOpError("stops is a number of stops (+1 doubles, -1 halves), got %r. NOTHING was changed." % (stops,))
    if abs(n) > 64:
        raise MifOpError("stops %g multiplies the energy by 2 ** %g - outside +-64. NOTHING was changed." % (n, n))
    return 2.0 ** n


# DIFFUSE AND SPECULAR FACTORS ARE EEVEE'S. MEASURED 2026-09-16 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1: one POINT
# light at 600 W over a sphere, 128 px, each factor 1.0 against 0.0 (scratchpad measure_diffuse_factor.py and
# measure_specular_displacement.py). EEVEE: diffuse_factor on a matte sphere 21.1-23.5/255 mean, specular_factor on a
# glossy one 0.084-0.087 mean and 147-149 max - a highlight is small, which is why the second seat's 5.2.2 reading
# (0.245, mailbox id 46) looked like nothing. Cycles: 0.0, max 0.0, for both factors on both spheres, every build.
_EEVEE_ONLY_LIGHT_FACTORS = ("diffuseFactor", "specularFactor")


def _light_factor_engine_note(params):
    """(inert keys, note) for the EEVEE-only light factors this call sent while the scene renders with Cycles."""
    engine = bpy.context.scene.render.engine
    sent = [k for k in _EEVEE_ONLY_LIGHT_FACTORS if params.get(k) is not None]
    if not sent or "CYCLES" not in engine:
        return None, None
    return sent, ("%s written, and Cycles does not read them - measured to change no pixel under Cycles on every build, "
                  "where EEVEE's render moves. They are stored and apply under EEVEE." % ", ".join(sent))


def op_set_light(params):
    """Change a light that already exists, and report what it IS afterwards.

    WHY THIS EXISTS. Until 2026-09-03 a light could be created and never touched again: there was
    no way to change its energy, colour, cone, size or shadow, and no way to READ any of it back -
    object_info returns early for a light and reports only the transform. So the addon could build
    a lighting rig and then not adjust it, which is most of what lighting work actually is. The
    only route was run_python, i.e. the arbitrary-code switch a user may well have turned off.

    params:
      object / light / name (str)  which light. Required.
      type (str)                   RETYPE the light - POINT | SUN | SPOT | AREA. Per-type keys in
                                   the same call are validated against the NEW type.
      energy / power (float)       watts for POINT/SPOT/AREA, irradiance for SUN
      energyScale (float) | stops  multiply the energy the light has (stops: 2 ** stops); not with energy
      color [r,g,b]
      radius (float)               soft-shadow size, POINT/SPOT only
      size / sizeY / shape         AREA only
      spotAngle / spotBlend        SPOT only, radians
      angle (float)                SUN only, angular diameter in radians
      shadow (bool)
      diffuseFactor / specularFactor (float)
      location / rotation {x,y,z}  moves the light OBJECT; rotation is radians, written into the property the
                                   light's rotation_mode reads - as given in an Euler mode, converted from an XYZ
                                   Euler for QUATERNION or AXIS_ANGLE (every glTF import). rotationWrittenTo names
                                   the property; before/after read the world rotation.

    Every refusal fires before any write. A caller who names one bad key gets a light in exactly
    the state it was in before the call.
    """
    reject_unknown(params, _SET_LIGHT_KEYS, "set_light")
    want = take(params, "object", "light", "name", default=None, kind=str)
    if not want:
        raise MifOpError("'object' is required - the name of the light to change "
                         "(list them with list_lights). NOTHING was changed.")
    obj = bpy.data.objects.get(want)
    if obj is None:
        known = [o.name for o in bpy.data.objects if o.type == "LIGHT"][:25]
        raise MifOpError("no object named '%s'. Lights present: %s. NOTHING was changed."
                         % (want, ", ".join(known) if known else "<none>"))
    if obj.type != "LIGHT":
        raise MifOpError("'%s' is a %s, not a LIGHT. NOTHING was changed." % (want, obj.type))
    data = obj.data

    # THE TYPE THE LIGHT WILL BE, which is what the per-type keys must be judged against - not the
    # type it is now. Retyping to SPOT and setting spotAngle in one call is legitimate and has to
    # work; setting spotAngle while retyping to POINT is a caller bug and has to be refused.
    new_type = take(params, "type", "lightType", "kind", default=None, kind=str)
    if new_type is not None:
        new_type = str(new_type).upper()
        valid = _valid_light_types()
        if new_type not in valid:
            raise MifOpError("unknown light type '%s' for this Blender. Valid: %s. "
                             "NOTHING was changed." % (new_type, ", ".join(sorted(valid))))
    effective = new_type or data.type
    _refuse_misplaced_light_keys(params, effective, "changed")
    refuse_unsupported_shadow(params, "changed")

    # THE LAST REFUSAL THAT WAS NOT UP HERE. _apply_light_props calls _check_light_temperature,
    # which raises on a Blender without bpy.types.Light.use_temperature - and it used to run after
    # this function had already written data.type, obj.location and obj.rotation_euler. A caller on
    # 3.6 asking to move a light AND set 3200 K got the move, the retype, and then a refusal saying
    # NOTHING was changed. Asked here instead, above every write in this function.
    #
    # Checking the PRE-retype datablock is equivalent: use_temperature is a property of the Light
    # base type, present or absent by Blender VERSION, not by light type. Retyping cannot make it
    # appear or vanish.
    checked_temperature = _check_light_temperature(params)

    col = params.get("color")
    if col is not None and (not isinstance(col, (list, tuple)) or len(col) < 3):
        raise MifOpError("'color' must be [r,g,b] in 0..1, got %r. NOTHING was changed." % (col,))

    before = light_readback(obj, data)

    # PARSED ABOVE THE COMMIT, because _vec3 CAN refuse and this comment used to be false: a
    # malformed location was read one line after data.type had been applied, so the op retyped the
    # light and then answered "NOTHING was created".
    set_loc = _vec3(params, "location", tuple(obj.location), "changed") \
        if "location" in params else None
    # THE ROTATION THE OBJECT'S MODE USES, as the default for an axis a partial {x,y,z} leaves out - rotation_euler is
    # what a QUATERNION light (every glTF import) does not read. See ops_common.write_rotation.
    set_rot = _vec3(params, "rotation", rotation_euler_values(obj), "changed") \
        if "rotation" in params else None
    # AND THE PROPERTY BLOCK, for exactly the reason the comment above gives. _apply_light_props
    # parses as well as writes, so every take_float in it can refuse - and since 2026-09-08 it also
    # refuses a `color` that is not a sequence by name. Calling the plan here means all of that
    # happens before data.type, obj.location and obj.rotation_euler are touched.
    #
    # audit_mutate_then_deny found this the moment the colour check landed: three writes above one
    # refusal that promises "NOTHING was created". Writes nothing itself, and take() reads rather
    # than pops, so _apply_light_props parsing again below is free and there stays ONE parse list.
    light_plan = _plan_light_props(params)
    factor = _energy_factor(params)

    # COMMIT. Nothing below can refuse.
    if new_type is not None:
        data.type = new_type
        # RE-FETCH, BECAUSE THE PYTHON REFERENCE IS NOW STALE. Assigning `type` changes the ID's
        # RNA subclass - a PointLight becomes an AreaLight - but the `data` object bound above is
        # still the OLD wrapper, so `data.size` a few lines down raises
        #     AttributeError: 'PointLight' object has no attribute 'size'
        # even though the light IS an AREA by then.
        #
        # The order in this function was always correct and that is what made it confusing: type is
        # applied first, exactly as intended, and the per-type keys were still refused. Retyping and
        # setting a per-type key in one call is the case the docstring above explicitly promises
        # ("Retyping to SPOT and setting spotAngle in one call is legitimate and has to work"), and
        # it was the one case that could not work.
        #
        # Found 2026-09-05 converting bunker room lamps from POINT to AREA. The caller's workaround
        # is two calls, which is why it looked like a caller problem for a while.
        data = obj.data
    if set_loc is not None:
        obj.location = set_loc
    rot_written = None
    if set_rot is not None:
        # INTO THE PROPERTY THE MODE READS. This was `obj.rotation_euler = set_rot` whatever the mode, so a QUATERNION
        # light - every glTF import - answered ok with changedFields [] and did not turn (5.2.1, 2026-09-28).
        rot_written = write_rotation(obj, set_rot)
    _apply_light_props(data, light_plan, params, checked_temperature)
    if factor is not None:
        data.energy = float(data.energy) * factor

    bpy.context.view_layer.update()
    after = light_readback(obj, data)
    # WHAT ACTUALLY MOVED, not what was asked for. A caller who sets energy to the value it already
    # had should be able to see that nothing changed, and a retype that silently drops a per-type
    # property - Blender does discard spot_size when you leave SPOT - shows up here rather than
    # being discovered later in a render.
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    inert, inert_note = _light_factor_engine_note(params)
    mode_note = rotation_mode_note(obj, rot_written) if rot_written else None
    return {
        "light": obj.name,
        "before": before,
        "after": after,
        "changedFields": changed,
        "changedAnything": bool(changed),
        "inertUnderThisEngine": inert,
        "engineNote": inert_note,
        "rotationMode": obj.rotation_mode,
        "rotationWrittenTo": rot_written,
        "rotationModeNote": mode_note,
        "energyFactor": round(factor, 6) if factor is not None else None,
    }


def op_list_lights(params):
    """Every light in the file, with its full state. The read path that did not exist.

    params:
      nameContains (str)  optional substring filter, case-insensitive
      type (str)          optional POINT | SUN | SPOT | AREA filter

    Reports lights in bpy.data.objects, so a light datablock with no object is not listed - it
    cannot illuminate anything until something links it, and clear_scene purges those.
    """
    reject_unknown(params, {"nameContains", "type", "limit", "nameExcludes"}, "list_lights")
    sub = take(params, "nameContains", default=None, kind=str)
    want = take(params, "type", default=None, kind=str)
    if want:
        want = str(want).upper()
        valid = _valid_light_types()
        if want not in valid:
            raise MifOpError("unknown light type '%s'. Valid: %s."
                             % (want, ", ".join(sorted(valid))))
    rows = []
    for obj in bpy.data.objects:
        if obj.type != "LIGHT":
            continue
        if sub and sub.lower() not in obj.name.lower():
            continue
        if want and obj.data.type != want:
            continue
        row = light_readback(obj, obj.data)
        # VISIBILITY IS PART OF "is this lighting anything". A light hidden in the render still
        # reads as a perfectly configured light in every other field, which is exactly the sort of
        # thing somebody debugging a black render needs told.
        row["hideViewport"] = bool(obj.hide_viewport)
        row["hideRender"] = bool(obj.hide_render)
        rows.append(row)
    rows.sort(key=lambda r: r["name"])
    rows = exclude_rows(rows, params)
    matched = len(rows)
    # A SCENE-WIDE ANSWER MUST STAY SCENE-WIDE. Computed before the limit, because "does this scene light
    # anything at all" cannot become False just because the caller asked for two rows - which is exactly what
    # reading it off the truncated list would do.
    any_rendering = any(not r["hideRender"] for r in rows)
    rows, extra = limit_rows(rows, params, "light(s)")
    return dict({
        "count": matched,
        "lights": rows,
        "sceneHasAnyLight": any_rendering,
    }, **extra)


def _camera_focus_and_fov_plan(params, camera_type, verb):
    """Validate fieldOfView and focusObject BEFORE anything is created or written.

    SPLIT OUT OF THE WRITER ON 2026-09-04, because the writer was called below the commit in both
    ops and every one of its four refusals said "NOTHING was created"/"NOTHING was changed" after the
    camera existed and ten properties had been set. create_camera({lens: 50, fieldOfView: 90}) built
    the camera, linked it, wrote location, rotation, type, lens, sensor, clipping and shift, and then
    answered "NOTHING was created" - with the camera sitting in the file. set_camera was worse: the
    line above its call reads "# COMMIT. Nothing below can refuse."

    Both ops had their _vec3 parses hoisted above the commit that same morning for exactly this
    reason. This helper was ADDED after that, below the commit, and reintroduced the shape the hoist
    had just removed - which is the argument for auditing the property rather than remembering it.
    Found by tools/audit_mutate_then_deny.py; see its docstring.

    camera_type is passed in rather than read off the datablock so this can run before one exists:
    create_camera knows it as `effective_type`, set_camera as the requested type or the current one.

    SPLIT RATHER THAN REORDERED, the same choice as ops_create's _place_values. Moving the call up
    would work today and be undone by the next person who needs `data`; a function that cannot write
    cannot be called too early.
    """
    fov = take_float(params, "fieldOfView", default=None)
    if fov is not None:
        if take_float(params, "lens", "focalLength", default=None) is not None:
            raise MifOpError(
                "pass lens/focalLength OR fieldOfView, not both - they are the SAME property in "
                "two units, and setting angle to 90 degrees moves lens from 50mm to 18mm. "
                "NOTHING was %s." % verb)
        if camera_type != "PERSP":
            raise MifOpError("fieldOfView applies to a PERSP camera and this one is %s - an "
                             "orthographic camera has no field of view. NOTHING was %s."
                             % (camera_type, verb))
        if not (0.0 < fov < 180.0):
            raise MifOpError("fieldOfView is in DEGREES and must be between 0 and 180, got %g. "
                             "NOTHING was %s." % (fov, verb))

    focus_name = take(params, "focusObject", default=None, kind=str)
    target = None
    if focus_name is not None:
        target = bpy.data.objects.get(str(focus_name))
        if target is None:
            have = sorted(o.name for o in bpy.data.objects)[:25]
            raise MifOpError("no object named '%s' to focus on. This scene has: %s. NOTHING was %s."
                             % (focus_name, ", ".join(have) if have else "(none)", verb))
    return fov, target


def _write_camera_focus_and_fov(data, planned):
    """focusObject and fieldOfView - both REPORTED by object_info and, until now, writable by nothing.

    THE TWO ASYMMETRIES THIS CLOSES. ops_common reports dofFocusObject and the camera's angle; the
    write path had focus_distance and lens and neither of the other two. So a camera could be
    described in more detail than it could be built.

    A FOCUS OBJECT OVERRIDES THE FOCUS DISTANCE, silently. Blender uses the object's distance and
    ignores focus_distance entirely once focus_object is set, and BOTH fields keep reading back
    exactly as written - so a caller who sets a distance and an object gets one of them, with
    nothing to say which. That is reported rather than refused, because setting both is a reasonable
    way to express "focus here for now, track this later".

    ANGLE AND LENS ARE ONE PROPERTY IN TWO UNITS. Setting angle to 90 degrees moves lens from 50mm
    to 18mm - measured on all four builds - so passing both is contradictory and is refused rather
    than resolved by declaration order. That refusal, and every other one this pair makes, lives in
    _camera_focus_and_fov_plan above. NOTHING HERE CAN REFUSE, which is the point of the split.
    """
    fov, target = planned
    if fov is not None:
        data.angle = math.radians(fov)

    focus_set = False
    if target is not None:
        data.dof.focus_object = target
        # A FOCUS OBJECT WITHOUT use_dof IS STORED AND IGNORED, the same shape as a cutoff distance
        # with its toggle off - the field reads back perfectly and the render does not change.
        data.dof.use_dof = True
        focus_set = True
    return fov, focus_set


def op_create_camera(params):
    """Create a camera, optionally aimed at a point.

    params:
      name (str)
      location / rotation    {x,y,z}; rotation is RADIANS, like every other op here
      lookAt {x,y,z}         aim at a point INSTEAD of giving rotation - refused together with
                             rotation, because two answers to the same question is a caller bug
      lens / focalLength     mm, PERSP only
      sensorWidth (float)    mm
      type                   PERSP | ORTHO | PANO
      orthoScale (float)     ORTHO only
      clipStart / clipEnd
      fStop / dofDistance    enables depth of field when either is given
      shiftX / shiftY
      makeActive (bool)      also set scene.camera, default True
    """
    reject_unknown(params, _CAMERA_KEYS, "create_camera")
    if "lookAt" in params and "rotation" in params:
        raise MifOpError("pass lookAt OR rotation, not both - they are two answers to the same "
                         "question and there is no sensible way to combine them. NOTHING was "
                         "created.")

    # VALIDATED BEFORE THE CAMERA EXISTS, the same way create_light was fixed. Three refusals used
    # to fire after bpy.data.cameras.new() and the scene link. They were honest - each said "The
    # camera WAS created" - and still left a stray camera in somebody's scene for nothing but a
    # typo, which is what create_light's own comment objects to. Everywhere else in this addon a
    # refusal means nothing was created, and this now matches.
    #
    # The effective type has to be decided up front for the same reason: lens is refused on a
    # non-PERSP camera and orthoScale on a non-ORTHO one, and both questions are about the type the
    # camera is being CREATED as, which is knowable from params without creating anything.
    ctype = take(params, "type", default=None, kind=str)
    if ctype:
        valid = {i.identifier for i in bpy.types.Camera.bl_rna.properties["type"].enum_items}
        ctype = str(ctype).upper()
        if ctype not in valid:
            raise MifOpError("unknown camera type '%s'. Valid: %s. NOTHING was created."
                             % (ctype, ", ".join(sorted(valid))))
    effective_type = ctype or "PERSP"      # Blender's own default for a new camera
    if take_float(params, "lens", "focalLength", default=None) is not None \
            and effective_type != "PERSP":
        raise MifOpError("lens/focalLength applies to a PERSP camera and this one is %s - use "
                         "orthoScale for ORTHO. NOTHING was created." % effective_type)
    if take_float(params, "orthoScale", default=None) is not None and effective_type != "ORTHO":
        raise MifOpError("orthoScale applies to an ORTHO camera and this one is %s. "
                         "NOTHING was created." % effective_type)

    # PARSED BEFORE ANYTHING EXISTS, same reason as create_light above.
    loc = _vec3(params, "location", (0.0, 0.0, 0.0))
    want_rot = (_look_at_euler(loc, _vec3(params, "lookAt", (0.0, 0.0, 0.0)))
                if "lookAt" in params else _vec3(params, "rotation", (0.0, 0.0, 0.0)))
    cam_plan = _camera_focus_and_fov_plan(params, effective_type, "created")

    # THE REST OF THEM, PARSED HERE FOR THE SAME REASON. take_float and take_bool refuse a
    # non-finite or wrongly-typed value by raising, and every one of these used to raise AFTER
    # bpy.data.cameras.new() and the scene link - so a mistyped clipStart cost the caller a stray
    # camera in their scene. The block above already fixed three refusals on exactly this argument
    # and left these eight; the matrix found them the day create_camera's payload stopped being
    # empty. Parse now, apply after the object exists.
    p_lens = take_float(params, "lens", "focalLength", default=None)
    p_sensor_w = take_float(params, "sensorWidth", default=None)
    p_ortho = take_float(params, "orthoScale", default=None)
    p_clip_start = take_float(params, "clipStart", default=None)
    p_clip_end = take_float(params, "clipEnd", default=None)
    p_shift_x = take_float(params, "shiftX", default=None)
    p_shift_y = take_float(params, "shiftY", default=None)
    p_fstop = take_float(params, "fStop", default=None)
    p_dof_dist = take_float(params, "dofDistance", default=None)
    p_make_active = take_bool(params, "makeActive", default=True)

    snap = selection_snapshot()
    try:
        wanted = str(take(params, "name", default="Camera", kind=str))
        data = bpy.data.cameras.new(name=wanted)
        obj = bpy.data.objects.new(data.name, data)
        bpy.context.scene.collection.objects.link(obj)

        obj.location = loc
        obj.rotation_euler = want_rot

        # Type and the two type-gated properties were validated above, before anything existed.
        if ctype:
            data.type = ctype

        # APPLYING values parsed before anything existed - see the block above the try.
        if p_lens is not None:
            data.lens = p_lens
        if p_sensor_w is not None:
            data.sensor_width = p_sensor_w
        if p_ortho is not None:
            data.ortho_scale = p_ortho
        if p_clip_start is not None:
            data.clip_start = p_clip_start
        if p_clip_end is not None:
            data.clip_end = p_clip_end
        if p_shift_x is not None:
            data.shift_x = p_shift_x
        if p_shift_y is not None:
            data.shift_y = p_shift_y

        # DOF is only switched on when something was actually asked for, so a plain camera is not
        # silently given a defocus the caller never requested.
        if p_fstop is not None or p_dof_dist is not None:
            data.dof.use_dof = True
            if p_fstop is not None:
                data.dof.aperture_fstop = p_fstop
            if p_dof_dist is not None:
                data.dof.focus_distance = p_dof_dist
        _fov, _focus_set = _write_camera_focus_and_fov(data, cam_plan)

        was = bpy.context.scene.camera.name if bpy.context.scene.camera else None
        if p_make_active:
            bpy.context.scene.camera = obj

        bpy.context.view_layer.update()
        out = {
            "name": obj.name,
            # Blender renames on collision rather than failing, so a caller retrying a
            # timed-out create gets "Camera.001" while believing it holds "Camera". Same field
            # pair as _created's, and for the same reason.
            "requestedName": wanted,
            "nameWasSuffixed": obj.name != wanted,
            "dataName": data.name,
            "type": data.type,
            "location": rnd(list(obj.matrix_world.to_translation())),
            "rotationEuler": rnd(list(obj.matrix_world.to_euler())),
            "lens": round(float(data.lens), 6),
            "sensorWidth": round(float(data.sensor_width), 6),
            "clipStart": round(float(data.clip_start), 6),
            "clipEnd": round(float(data.clip_end), 6),
            "dofEnabled": bool(data.dof.use_dof),
            "dofFocusObject": (data.dof.focus_object.name
                               if getattr(data.dof, "focus_object", None) else None),
            # THE OVERRIDE, SAID OUT LOUD. Blender uses the focus OBJECT's distance and ignores
            # focus_distance entirely once one is set, and both fields keep reading back exactly as
            # written - so without this a caller who set both has no way to know which one won.
            "focusDistanceIgnored": bool(getattr(data.dof, "focus_object", None)),
            "fieldOfViewDeg": round(math.degrees(data.angle), 4) if data.type == "PERSP" else None,
            "isSceneCamera": bpy.context.scene.camera is obj,
            "previousSceneCamera": was,
        }
        if data.dof.use_dof:
            out["fStop"] = round(float(data.dof.aperture_fstop), 6)
            out["focusDistance"] = round(float(data.dof.focus_distance), 6)
        if data.type == "ORTHO":
            out["orthoScale"] = round(float(data.ortho_scale), 6)
        if "lookAt" in params:
            out["aimNote"] = ("rotation was DERIVED from lookAt - a Blender camera faces its local "
                              "-Z, which is the thing hand-written aiming gets wrong first.")
        return out
    finally:
        selection_restore(snap)


_SET_CAMERA_KEYS = set(_CAMERA_KEYS) | {"object", "camera"}


def op_set_camera(params):
    """Change a camera that already exists, including which one the scene renders through.

    Same absence set_light closed for lights: a camera could be created and never adjusted, and
    the scene camera could only be chosen at BIRTH via makeActive - so switching between two
    existing cameras, which is ordinary shot work, was impossible without run_python.

    params:
      object / camera / name (str)  which camera. Required.
      type                          PERSP | ORTHO | PANO
      lens / focalLength (float)    mm, PERSP only
      sensorWidth / sensorHeight    mm
      sensorFit                     AUTO | HORIZONTAL | VERTICAL - without which sensorWidth is a
                                    half-answer and no real lens can be matched
      orthoScale                    ORTHO only
      clipStart / clipEnd
      shiftX / shiftY
      fStop / dofDistance           enables depth of field when either is given
      lookAt {x,y,z}                aim at a WORLD point; refused together with rotation. Aims a camera in any
                                    rotation_mode, and one parented under a turned rig
      location / rotation {x,y,z}   rotation is RADIANS, written into the property the camera's rotation_mode
                                    reads - as given in an Euler mode, converted from an XYZ Euler for QUATERNION or
                                    AXIS_ANGLE (every glTF import). rotationWrittenTo names the property
      makeActive (bool)             make this the scene camera

    before/after carry the camera's world location and rotationEuler, so a rotation that did not land shows in
    changedFields. Every refusal fires before any write.
    """
    reject_unknown(params, _SET_CAMERA_KEYS | {"sensorHeight", "sensorFit"}, "set_camera")
    want = take(params, "object", "camera", "name", default=None, kind=str)
    if not want:
        raise MifOpError("'object' is required - the name of the camera to change (list them with "
                         "list_cameras). NOTHING was changed.")
    obj = bpy.data.objects.get(want)
    if obj is None:
        known = [o.name for o in bpy.data.objects if o.type == "CAMERA"][:25]
        raise MifOpError("no object named '%s'. Cameras present: %s. NOTHING was changed."
                         % (want, ", ".join(known) if known else "<none>"))
    if obj.type != "CAMERA":
        raise MifOpError("'%s' is a %s, not a CAMERA. NOTHING was changed." % (want, obj.type))
    cam = obj.data

    if "lookAt" in params and "rotation" in params:
        raise MifOpError("pass lookAt OR rotation, not both - they are two answers to the same "
                         "question. NOTHING was changed.")

    # THE TYPE THE CAMERA WILL BE, judged before anything is written - the same rule set_light
    # follows, because lens is PERSP-only and orthoScale is ORTHO-only and both questions are about
    # the type AFTER this call.
    ctype = take(params, "type", default=None, kind=str)
    if ctype:
        valid = {i.identifier for i in bpy.types.Camera.bl_rna.properties["type"].enum_items}
        ctype = str(ctype).upper()
        if ctype not in valid:
            raise MifOpError("unknown camera type '%s'. Valid: %s. NOTHING was changed."
                             % (ctype, ", ".join(sorted(valid))))
    effective = ctype or cam.type
    if take_float(params, "lens", "focalLength", default=None) is not None and effective != "PERSP":
        raise MifOpError("lens/focalLength applies to a PERSP camera and this one is %s - use "
                         "orthoScale for ORTHO. NOTHING was changed." % effective)
    if take_float(params, "orthoScale", default=None) is not None and effective != "ORTHO":
        raise MifOpError("orthoScale applies to an ORTHO camera and this one is %s. "
                         "NOTHING was changed." % effective)
    fit = take(params, "sensorFit", default=None, kind=str)
    if fit:
        fit = str(fit).upper()
        fits = {i.identifier for i in bpy.types.Camera.bl_rna.properties["sensor_fit"].enum_items}
        if fit not in fits:
            raise MifOpError("unknown sensorFit '%s'. Valid: %s. NOTHING was changed."
                             % (fit, ", ".join(sorted(fits))))

    # WHERE THE CAMERA POINTS, beside what it is - read off matrix_world, as list_cameras and light_readback read it.
    # camera_readback carries no transform, so a rotation or lookAt that did not land changed nothing this op reported:
    # a QUATERNION camera (every glTF import) answered changedFields [] to a lookAt it ignored (5.2.1, 2026-09-28).
    # The depsgraph brought current first: matrix_world is only as fresh as the last evaluation, and basis_space below
    # reads it too.
    bpy.context.view_layer.update()
    before = camera_readback(obj, cam)
    before.update(location=rnd(list(obj.matrix_world.to_translation())),
                  rotationEuler=rnd(list(obj.matrix_world.to_euler())))

    # PARSED ABOVE THE COMMIT, same as set_light. lookAt is resolved against the location this
    # call will END with, not the one the object had, so the two orders cannot disagree.
    set_loc = _vec3(params, "location", tuple(obj.location), "changed") \
        if "location" in params else None
    final_loc = set_loc if set_loc is not None else tuple(obj.location)
    # A CHILD'S location and rotation are in its PARENT'S space, and lookAt is a world point: the look-at maths gets the
    # world point the camera will stand at, and write_orientation takes the orientation out of the parent's space.
    # Taken before any write - see ops_common.basis_space. Until 2026-09-28 a camera under a turned rig answered ok
    # aimed as far off as the rig was turned.
    space = basis_space(obj)
    look_from = tuple(space @ mathutils.Vector(final_loc)) if space is not None else final_loc
    set_rot = look_rot = None
    if "lookAt" in params:
        look_rot = _look_at_euler(look_from, _vec3(params, "lookAt", (0.0, 0.0, 0.0), "changed"))
    elif "rotation" in params:
        # The rotation the mode uses as the default for a missing axis - see set_light and ops_common.write_rotation.
        set_rot = _vec3(params, "rotation", rotation_euler_values(obj), "changed")

    # Validated here, written after the commit. The line below used to be false: the focus/fov
    # writer sat under it and could refuse with "NOTHING was changed" after eleven writes.
    cam_plan = _camera_focus_and_fov_plan(params, ctype or cam.type, "changed")

    # THE REST OF THE VALUES, READ HERE (2026-09-21). They were read inside the commit below, so the line under this
    # block was false too: {type: ORTHO, location, sensorWidth: "banana"} changed the type and moved the camera, then
    # refused - and clipEnd NaN, fStop NaN and makeActive "flase" the same.
    float_vals = {key: take_float(params, key, default=None)
                  for key in ("lens", "focalLength", "sensorWidth", "sensorHeight", "orthoScale", "clipStart",
                              "clipEnd", "shiftX", "shiftY")}
    fstop = take_float(params, "fStop", default=None)
    dofd = take_float(params, "dofDistance", default=None)
    make_active = take_bool(params, "makeActive", default=False)

    # COMMIT. Nothing below can refuse.
    if ctype:
        cam.type = ctype
    if set_loc is not None:
        obj.location = set_loc
    # INTO THE PROPERTY THE MODE READS, both ways in - see set_light. lookAt is an ORIENTATION, so it is converted for
    # every mode but XYZ, another Euler order included (the look-at angles are XYZ ones); `rotation` is angles, taken as
    # the object's own Euler in an Euler mode and converted from XYZ otherwise.
    rot_written = None
    if look_rot is not None:
        rot_written = write_orientation(obj, look_rot, space)
    elif set_rot is not None:
        rot_written = write_rotation(obj, set_rot)
    for key, attr in (("lens", "lens"), ("focalLength", "lens"),
                      ("sensorWidth", "sensor_width"), ("sensorHeight", "sensor_height"),
                      ("orthoScale", "ortho_scale"), ("clipStart", "clip_start"),
                      ("clipEnd", "clip_end"), ("shiftX", "shift_x"), ("shiftY", "shift_y")):
        v = float_vals[key]
        if v is not None and hasattr(cam, attr):
            setattr(cam, attr, v)
    if fit:
        cam.sensor_fit = fit
    if (fstop is not None or dofd is not None) and getattr(cam, "dof", None) is not None:
        cam.dof.use_dof = True
        if fstop is not None:
            cam.dof.aperture_fstop = fstop
        if dofd is not None:
            cam.dof.focus_distance = dofd
    # THE SAME WRITER create_camera USES. Two ops asking the same question needs one answer;
    # a second copy is how allowEditConst got past one guard and not the other in this very
    # file, which its own comment above _LIGHT_TYPE_KEYS records.
    _write_camera_focus_and_fov(cam, cam_plan)
    if make_active:
        bpy.context.scene.camera = obj

    bpy.context.view_layer.update()
    after = camera_readback(obj, cam)
    after.update(location=rnd(list(obj.matrix_world.to_translation())),
                 rotationEuler=rnd(list(obj.matrix_world.to_euler())))
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    mode_note = rotation_mode_note(obj, rot_written) if rot_written else None
    return {
        "camera": obj.name,
        "before": before,
        "after": after,
        "changedFields": changed,
        "changedAnything": bool(changed),
        "rotationMode": obj.rotation_mode,
        "rotationWrittenTo": rot_written,
        "rotationModeNote": mode_note,
    }


def op_list_cameras(params):
    """Every camera in the file, and which one the scene actually renders through.

    sceneCamera was unobtainable anywhere in this addon: scene_info omits it,
    set_render_settings reports a bare boolean, and render_still names it only by blocking for a
    whole render. It is the first thing anybody needs to know.

    params:
      nameContains (str)  optional substring filter, case-insensitive
    """
    reject_unknown(params, ("nameContains", "limit", "nameExcludes"), "list_cameras")
    sub = take(params, "nameContains", default=None, kind=str)
    rows = []
    for obj in bpy.data.objects:
        if obj.type != "CAMERA" or obj.data is None:
            continue
        if sub and sub.lower() not in obj.name.lower():
            continue
        row = camera_readback(obj, obj.data)
        row["name"] = obj.name
        row["location"] = rnd(list(obj.matrix_world.to_translation()))
        row["rotationEuler"] = rnd(list(obj.matrix_world.to_euler()))
        row["hideViewport"] = bool(obj.hide_viewport)
        row["hideRender"] = bool(obj.hide_render)
        rows.append(row)
    rows.sort(key=lambda r: r["name"])
    sc = bpy.context.scene
    rows = exclude_rows(rows, params)
    matched = len(rows)
    rows, extra = limit_rows(rows, params, "camera(s)")
    return dict({
        "count": matched,
        "cameras": rows,
        "sceneCamera": sc.camera.name if sc.camera else None,
        "hasSceneCamera": sc.camera is not None,
    }, **extra)


# ---------------------------------------------------------------------------------------------------------------------
# set_camera_display (1.3, 2026-10-18): what the camera VIEW shows - passepartout, composition guides, safe areas and
# background images. None of it changes a render; it is what a person finishing a shot in the GUI looks through (a
# reference plate behind a matchmove camera, thirds over a layout). Read back whole on every call, so a call with only
# `object` is the read.
# ---------------------------------------------------------------------------------------------------------------------
_CAM_DISPLAY_FLAGS = (("passepartout", "show_passepartout"), ("safeAreas", "show_safe_areas"),
                      ("safeCenter", "show_safe_center"), ("showLimits", "show_limits"), ("showMist", "show_mist"),
                      ("showName", "show_name"), ("showSensor", "show_sensor"),
                      ("showBackgroundImages", "show_background_images"))
_CAM_SAFE_AREAS = (("title", "title"), ("action", "action"), ("titleCenter", "title_center"),
                   ("actionCenter", "action_center"))
_CAM_BG_KEYS = {"image", "imageName", "movie", "movieName", "alpha", "depth", "frameMethod", "offset", "rotation",
                "scale", "flipX", "flipY", "show"}
# Spelled out (not derived from _CAM_DISPLAY_FLAGS) so parity_check can read it statically.
_CAM_DISPLAY_KEYS = {"object", "camera", "name", "passepartoutAlpha", "displaySize", "guides", "safeAreaMargins",
                     "backgroundImages", "clearBackgroundImages", "passepartout", "safeAreas", "safeCenter",
                     "showLimits", "showMist", "showName", "showSensor", "showBackgroundImages"}


def _camera_guides():
    """The composition guides THIS Blender has, by their RNA names - show_composition_<guide> - read off the type rather
    than listed, so a guide a version adds or drops is neither invented nor missed."""
    pre = "show_composition_"
    return sorted(p.identifier[len(pre):] for p in bpy.types.Camera.bl_rna.properties if p.identifier.startswith(pre))


def _enum_ids(rna_type, prop):
    return {i.identifier for i in rna_type.bl_rna.properties[prop].enum_items}


def camera_display_readback(cam):
    """Everything set_camera_display writes, read off the datablock (and the scene, for the safe-area margins)."""
    sa = bpy.context.scene.safe_areas
    out = {k: bool(getattr(cam, a)) for k, a in _CAM_DISPLAY_FLAGS if hasattr(cam, a)}
    out["passepartoutAlpha"] = round(float(cam.passepartout_alpha), 6)
    out["displaySize"] = round(float(cam.display_size), 6)
    out["guides"] = {g: bool(getattr(cam, "show_composition_" + g)) for g in _camera_guides()}
    out["safeAreaMargins"] = {k: rnd(getattr(sa, a)) for k, a in _CAM_SAFE_AREAS if hasattr(sa, a)}
    rows = []
    for i, bg in enumerate(cam.background_images):
        clip = getattr(bg, "clip", None)
        rows.append({
            "index": i,
            "source": bg.source,
            "image": bg.image.name if bg.image else None,
            "imageFile": bg.image.filepath if bg.image else None,
            "movie": clip.name if clip else None,
            "movieFile": clip.filepath if clip else None,
            "alpha": round(float(bg.alpha), 6),
            "depth": bg.display_depth,
            "frameMethod": bg.frame_method,
            "offset": rnd(bg.offset),
            "rotation": round(float(bg.rotation), 6),
            "scale": round(float(bg.scale), 6),
            "flipX": bool(bg.use_flip_x),
            "flipY": bool(bg.use_flip_y),
            "show": bool(bg.show_background_image),
        })
    out["backgroundImages"] = rows
    return out


def _vec2_unit(value, key):
    """A two-number list in 0..1 - a safe-area margin."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise MifOpError("'%s' must be a list of two numbers [x, y] between 0 and 1, got %r. NOTHING was changed."
                         % (key, value))
    xy = finite_floats(value, key)
    if any(v < 0.0 or v > 1.0 for v in xy):
        raise MifOpError("'%s' must be between 0 and 1 (a fraction of the frame), got %r. NOTHING was changed."
                         % (key, value))
    return xy


def _plan_background_image(i, spec):
    """One backgroundImages entry, validated in full before anything is loaded or written."""
    where = "backgroundImages[%d]" % i
    if not isinstance(spec, dict):
        raise MifOpError("%s must be an object like {image: <file>} - got %r. NOTHING was changed." % (where, spec))
    unknown = sorted(set(spec) - _CAM_BG_KEYS)
    if unknown:
        raise MifOpError("%s has unknown key(s) %s. Accepted: %s. NOTHING was changed."
                         % (where, ", ".join(unknown), ", ".join(sorted(_CAM_BG_KEYS))))
    sources = [k for k in ("image", "imageName", "movie", "movieName") if spec.get(k) is not None]
    if len(sources) != 1:
        raise MifOpError("%s needs exactly one of image (a file), imageName (an image already in the file), movie (a "
                         "video file) or movieName (a movie clip already in the file) - got %s. NOTHING was changed."
                         % (where, ", ".join(sources) or "none"))
    plan = {"kind": sources[0], "ref": spec[sources[0]]}
    if not isinstance(plan["ref"], str) or not plan["ref"].strip():
        raise MifOpError("%s.%s must be a non-empty string. NOTHING was changed." % (where, sources[0]))
    if plan["kind"] in ("image", "movie"):
        path = bpy.path.abspath(plan["ref"])
        if not os.path.isfile(path):
            raise MifOpError("%s.%s: no file at %s. NOTHING was changed." % (where, plan["kind"], path))
        plan["path"] = path
    elif plan["kind"] == "imageName" and bpy.data.images.get(plan["ref"]) is None:
        raise MifOpError("%s.imageName: no image named '%s' in the file. NOTHING was changed." % (where, plan["ref"]))
    elif plan["kind"] == "movieName" and bpy.data.movieclips.get(plan["ref"]) is None:
        raise MifOpError("%s.movieName: no movie clip named '%s' in the file. NOTHING was changed."
                         % (where, plan["ref"]))
    bg_type = bpy.types.CameraBackgroundImage
    for key, prop in (("depth", "display_depth"), ("frameMethod", "frame_method")):
        if spec.get(key) is not None:
            v = str(spec[key]).upper()
            valid = _enum_ids(bg_type, prop)
            if v not in valid:
                raise MifOpError("%s.%s '%s' is not one of %s. NOTHING was changed."
                                 % (where, key, spec[key], ", ".join(sorted(valid))))
            plan[key] = v
    for key in ("alpha", "rotation", "scale"):
        v = take_float(spec, key, default=None)
        if v is not None:
            if key == "alpha" and not 0.0 <= v <= 1.0:
                raise MifOpError("%s.alpha must be between 0 and 1, got %r. NOTHING was changed." % (where, v))
            if key == "scale" and v <= 0.0:
                raise MifOpError("%s.scale must be above 0, got %r. NOTHING was changed." % (where, v))
            plan[key] = v
    if spec.get("offset") is not None:
        if not isinstance(spec["offset"], (list, tuple)) or len(spec["offset"]) != 2:
            raise MifOpError("%s.offset must be [x, y], got %r. NOTHING was changed." % (where, spec["offset"]))
        plan["offset"] = finite_floats(spec["offset"], where + ".offset")
    for key in ("flipX", "flipY", "show"):
        if spec.get(key) is not None:
            plan[key] = take_bool(spec, key)
    return plan


def op_set_camera_display(params):
    """What a camera's VIEW shows: passepartout, composition guides, safe areas and background images.

    None of this changes a render - it is what a person looking through the camera in the viewport sees: a reference
    plate behind a matchmove camera, a thirds grid over a layout, title-safe margins over a broadcast frame. Every
    value is read back in `display`, and a call with only `object` is the read.

    params:
      object / camera / name (str)   the camera. Required.
      passepartout (bool), passepartoutAlpha (0..1)   darken outside the frame, and how much
      guides {<guide>: bool}         composition guides by this Blender's names (thirds, center, golden, ... - the
                                     refusal lists them); the guides not named keep their state
      safeAreas, safeCenter (bool)   draw the safe areas / the center-cut safe areas
      safeAreaMargins {title, action, titleCenter, actionCenter: [x, y] in 0..1}   the margins - a SCENE setting,
                                     shared by every camera in the scene
      showLimits, showMist, showName, showSensor (bool), displaySize (> 0)   the camera's other viewport overlays
      showBackgroundImages (bool)    turn the camera's background images on or off as a set
      backgroundImages [{image | imageName | movie | movieName, alpha, depth BACK|FRONT, frameMethod STRETCH|FIT|CROP,
                        offset [x, y], rotation (radians), scale, flipX, flipY, show}]   APPENDED to the camera's list;
                                     passing any turns showBackgroundImages on unless it is given
      clearBackgroundImages (bool)   empty the list first (the images stay in the file)

    Every refusal fires before any write; an image or movie file is loaded only after every check has passed.
    """
    reject_unknown(params, _CAM_DISPLAY_KEYS, "set_camera_display")
    want = take(params, "object", "camera", "name", default=None, kind=str)
    if not want:
        raise MifOpError("'object' is required - the name of the camera (list them with list_cameras). "
                         "NOTHING was changed.")
    obj = bpy.data.objects.get(want)
    if obj is None:
        known = [o.name for o in bpy.data.objects if o.type == "CAMERA"][:25]
        raise MifOpError("no object named '%s'. Cameras present: %s. NOTHING was changed."
                         % (want, ", ".join(known) if known else "<none>"))
    if obj.type != "CAMERA":
        raise MifOpError("'%s' is a %s, not a CAMERA. NOTHING was changed." % (want, obj.type))
    cam = obj.data

    # EVERYTHING READ AND CHECKED HERE; nothing below the COMMIT line refuses.
    flags = {}
    for key, attr in _CAM_DISPLAY_FLAGS:
        if params.get(key) is not None:
            flags[attr] = take_bool(params, key)
    alpha = take_float(params, "passepartoutAlpha", default=None)
    if alpha is not None and not 0.0 <= alpha <= 1.0:
        raise MifOpError("passepartoutAlpha must be between 0 and 1, got %r. NOTHING was changed." % alpha)
    size = take_float(params, "displaySize", default=None)
    if size is not None and size <= 0.0:
        raise MifOpError("displaySize must be above 0, got %r. NOTHING was changed." % size)
    guides = take(params, "guides", default=None, kind=dict) or {}
    have = _camera_guides()
    bad = sorted(g for g in guides if g not in have)
    if bad:
        raise MifOpError("unknown guide(s) %s. This Blender's guides: %s. NOTHING was changed."
                         % (", ".join(bad), ", ".join(have)))
    guide_vals = {g: take_bool(guides, g) for g in guides}
    margins = take(params, "safeAreaMargins", default=None, kind=dict) or {}
    names = {k: a for k, a in _CAM_SAFE_AREAS if hasattr(bpy.context.scene.safe_areas, a)}
    bad = sorted(k for k in margins if k not in names)
    if bad:
        raise MifOpError("unknown safeAreaMargins key(s) %s. Accepted: %s. NOTHING was changed."
                         % (", ".join(bad), ", ".join(sorted(names))))
    margin_vals = {names[k]: _vec2_unit(v, "safeAreaMargins." + k) for k, v in margins.items()}
    specs = take(params, "backgroundImages", default=None, kind=list) or []
    plans = [_plan_background_image(i, s) for i, s in enumerate(specs)]
    clear = take_bool(params, "clearBackgroundImages", default=False)

    # LOADS FIRST, and undone if one fails, so a refused load leaves no new datablock behind.
    before = camera_display_readback(cam)
    loaded = []
    try:
        for plan in plans:
            if plan["kind"] == "image":
                had = {i.as_pointer() for i in bpy.data.images}
                plan["data"] = bpy.data.images.load(plan["path"], check_existing=True)
                if plan["data"].as_pointer() not in had:
                    loaded.append(("images", plan["data"]))
            elif plan["kind"] == "movie":
                had = {c.as_pointer() for c in bpy.data.movieclips}
                plan["data"] = bpy.data.movieclips.load(plan["path"], check_existing=True)
                if plan["data"].as_pointer() not in had:
                    loaded.append(("movieclips", plan["data"]))
            elif plan["kind"] == "imageName":
                plan["data"] = bpy.data.images[plan["ref"]]
            else:
                plan["data"] = bpy.data.movieclips[plan["ref"]]
    except (RuntimeError, OSError) as e:
        for coll, datablock in loaded:
            getattr(bpy.data, coll).remove(datablock)
        raise MifOpError("Blender could not load %s: %s. NOTHING was changed." % (plan.get("path"), e))

    # COMMIT.
    for attr, v in flags.items():
        setattr(cam, attr, v)
    if alpha is not None:
        cam.passepartout_alpha = alpha
    if size is not None:
        cam.display_size = size
    for g, v in guide_vals.items():
        setattr(cam, "show_composition_" + g, v)
    for attr, xy in margin_vals.items():
        setattr(bpy.context.scene.safe_areas, attr, xy)
    if clear:
        cam.background_images.clear()
    for plan in plans:
        bg = cam.background_images.new()
        if plan["kind"] in ("image", "imageName"):
            bg.source = "IMAGE"
            bg.image = plan["data"]
        else:
            bg.source = "MOVIE_CLIP"
            bg.clip = plan["data"]
        for key, attr in (("alpha", "alpha"), ("depth", "display_depth"), ("frameMethod", "frame_method"),
                          ("offset", "offset"), ("rotation", "rotation"), ("scale", "scale"), ("flipX", "use_flip_x"),
                          ("flipY", "use_flip_y"), ("show", "show_background_image")):
            if key in plan:
                setattr(bg, attr, plan[key])
    if plans and "show_background_images" not in flags:
        cam.show_background_images = True

    after = camera_display_readback(cam)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    out = {"camera": obj.name, "display": after, "changedFields": changed, "changedAnything": bool(changed),
           "loaded": [d.name for _c, d in loaded],
           "note": "viewport display only - none of these settings changes a render. safeAreaMargins belong to the "
                   "scene, so every camera in it shares them."}
    if before["backgroundImages"] != after["backgroundImages"]:
        out["backgroundImagesBefore"] = len(before["backgroundImages"])
    return out


def op_aim_object(params):
    """Point any object at a point or another object, and MEASURE that it now points there.

    Nothing could aim anything after creation. create_camera takes lookAt at birth and that was
    the only caller of the aiming maths; a light could not be aimed at all, which makes a spot
    light almost unusable through this bridge.

    params:
      object (str, required)   what to aim
      target (str)             an object to aim AT - its world-space origin
      lookAt {x,y,z}           a point to aim at. Exactly one of target/lookAt.

    Blender aims down LOCAL -Z for lights and cameras alike, which is the convention _look_at_euler
    derives - and got wrong by exactly pi in its first version, aiming 166 degrees off while
    returning a plausible euler. So the postcondition is the ANGLE between the object's world -Z
    and the direction to the target, not the euler that was just written. Reading back what you
    wrote is a proxy that cannot fail.

    Any rotation_mode and any parent: the orientation goes into the property the object's mode reads
    (rotationWrittenTo), converted out of its parent's space for a child. rotationEuler in the answer is
    the WORLD rotation.
    """
    reject_unknown(params, ("object", "name", "target", "lookAt"), "aim_object")
    want = take(params, "object", "name", default=None, kind=str)
    if not want:
        raise MifOpError("'object' is required - what to aim. NOTHING was changed.")
    obj = bpy.data.objects.get(want)
    if obj is None:
        raise MifOpError("no object named '%s'. NOTHING was changed." % want)

    target_name = take(params, "target", default=None, kind=str)
    has_point = "lookAt" in params
    if bool(target_name) == bool(has_point):
        raise MifOpError("pass exactly one of target (an object) or lookAt (a point) - %s. "
                         "NOTHING was changed."
                         % ("both were given" if target_name else "neither was given"))
    # THE DEPSGRAPH CURRENT before any matrix_world is read - the target's, this object's and its parent's.
    bpy.context.view_layer.update()
    if target_name:
        tgt = bpy.data.objects.get(target_name)
        if tgt is None:
            raise MifOpError("no target object named '%s'. NOTHING was changed." % target_name)
        if tgt is obj:
            raise MifOpError("'%s' cannot be aimed at itself. NOTHING was changed." % want)
        point = tuple(tgt.matrix_world.to_translation())
    else:
        point = _vec3(params, "lookAt", (0.0, 0.0, 0.0))

    frm = tuple(obj.matrix_world.to_translation())
    # THE ORIENTATION, INTO THE PROPERTY THE MODE READS (ops_common.write_orientation). This wrote rotation_euler whatever
    # the mode, so a QUATERNION object - every glTF import - never turned, and the measurement below refused the op's
    # own write: 9.46 deg off, measured on 5.2.1 with a glTF camera (2026-09-28, RD118). An object in another Euler order
    # took the XYZ angles as its own and aimed somewhere else. AND OUT OF THE PARENT'S SPACE: the look-at angles are world
    # ones, and a child of a rig turned 90 deg was refused 84.26 deg off the same day (RD120, ops_common.basis_space).
    space = basis_space(obj)
    rot_written = write_orientation(obj, _look_at_euler(frm, point), space)
    bpy.context.view_layer.update()

    # THE MEASUREMENT. World -Z after the write, against the direction to the target.
    fwd = (obj.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))).normalized()
    to = mathutils.Vector((point[0] - frm[0], point[1] - frm[1], point[2] - frm[2]))
    if to.length == 0.0:
        raise MifOpError("the target is at the object's own origin, so there is no direction to "
                         "face. The rotation may already have been written - re-read it.")
    err = fwd.angle(to.normalized())
    if err > 1e-3:
        raise MifOpError("aimed '%s' but its -Z is %.4f rad (%.2f deg) off the target. The "
                         "rotation WAS written; do not trust it." % (want, err, math.degrees(err)))
    mode_note = rotation_mode_note(obj, rot_written)
    return {
        "object": obj.name,
        "target": target_name,
        "aimedAt": rnd(list(point)),
        # OFF THE WORLD MATRIX, as create_camera and list_cameras report it. It was obj.rotation_euler, which a
        # QUATERNION object does not read - the field reported a rotation the object did not have.
        "rotationEuler": rnd(list(obj.matrix_world.to_euler())),
        "rotationMode": obj.rotation_mode,
        "rotationWrittenTo": rot_written,
        "rotationModeNote": mode_note,
        "aimErrorRadians": round(float(err), 9),
        "aimErrorDegrees": round(math.degrees(err), 7),
        "measuredNote": ("The error is the angle between the object's world-space local -Z and the "
                         "direction to the target, measured after the write - not the euler that "
                         "was written, which cannot disagree with itself."),
    }


# The panorama field table and where each Blender keeps it live in ops_common since 2026-09-13, because
# camera_readback reads them too. One definition; these names keep every use below unchanged.
_PANO_FIELDS = PANO_FIELDS


# A LITERAL, for the same reason _VISIBILITY_KEYS is one in ops_scene: parity_check resolves
# accepted-key sets statically and is fail-closed, refusing a set comprehension rather than
# skipping a check it cannot read. The duplication of _PANO_FIELDS' keys is therefore forced, so it
# is GUARDED at import - a key present in _PANO_FIELDS and missing here would be refused at the
# door while every other part of the op supports it.
_PANO_KEYS = {
    "object", "camera", "name",
    "panoramaType", "fisheyeFov", "fisheyeLens",
    "latitudeMin", "latitudeMax", "longitudeMin", "longitudeMax",
}

_pano_missing = {k for k, _a in _PANO_FIELDS} - _PANO_KEYS
if _pano_missing:
    raise RuntimeError(
        "MifBlender ops_lightcam: _PANO_FIELDS names %s but _PANO_KEYS does not, so those keys "
        "would be refused by reject_unknown while the rest of the op supports them. Add them to "
        "the literal - it is duplicated on purpose so parity_check can read it."
        % ", ".join(sorted(_pano_missing)))


_pano_holder = pano_holder


def op_set_camera_panorama(params):
    """Configure a panoramic camera - the settings create_camera could accept a type for and
    never reach.

    A DECLARED-AND-UNREACHABLE, closed. create_camera validates PANO against Blender's own enum and
    accepts it, and then nothing in this addon could set a single panorama property - so a PANO
    camera could be created and was unusable, which is worse than not offering the type at all.

    params:
      object (str, required)          must already be a PANO camera
      panoramaType (str)              EQUIRECTANGULAR | FISHEYE_EQUIDISTANT | FISHEYE_EQUISOLID |
                                      MIRRORBALL | ... validated against this Blender's own enum
      fisheyeFov / fisheyeLens (float)
      latitudeMin / latitudeMax (float)     radians, EQUIRECTANGULAR framing
      longitudeMin / longitudeMax (float)

    Panoramic rendering is a CYCLES feature. On EEVEE the settings are stored and ignored, which is
    reported rather than left to be discovered in a render that comes back rectilinear.
    """
    reject_unknown(params, _PANO_KEYS, "set_camera_panorama")
    want = take(params, "object", "camera", "name", default=None, kind=str)
    if not want:
        raise MifOpError("'object' is required - which camera. NOTHING was changed.")
    obj = bpy.data.objects.get(want)
    if obj is None:
        raise MifOpError("no object named '%s'. NOTHING was changed." % want)
    if obj.type != "CAMERA":
        raise MifOpError("'%s' is a %s, not a CAMERA. NOTHING was changed." % (want, obj.type))
    cam = obj.data
    if cam.type != "PANO":
        raise MifOpError("'%s' is a %s camera, not PANO - these settings would be stored and never "
                         "used. Set its type to PANO first with set_camera. NOTHING was changed."
                         % (want, cam.type))

    holder = _pano_holder(cam)
    if holder is None:
        raise MifOpError("this Blender exposes no panorama settings on a camera (tried the camera "
                         "data and camera.cycles), so every key here would be silently ignored. "
                         "NOTHING was changed.")

    ptype = take(params, "panoramaType", default=None, kind=str)
    if ptype:
        ptype = str(ptype).upper()
        try:
            valid = {i.identifier for i in
                     holder.bl_rna.properties["panorama_type"].enum_items}
        except (KeyError, AttributeError):
            valid = set()
        if valid and ptype not in valid:
            raise MifOpError("unknown panoramaType '%s'. Valid: %s. NOTHING was changed."
                             % (ptype, ", ".join(sorted(valid))))

    # Every requested field is resolved BEFORE any is written, so a build missing one refuses the
    # whole call rather than applying half of them.
    plan = []
    for key, attr in _PANO_FIELDS:
        if key not in params:
            continue
        if not hasattr(holder, attr):
            raise MifOpError("this Blender has no '%s' on its panorama settings, so '%s' would be "
                             "silently ignored. NOTHING was changed." % (attr, key))
        value = ptype if key == "panoramaType" else take_float(params, key, default=None)
        plan.append((attr, value))
    if not plan:
        raise MifOpError("nothing to set - pass at least one of %s. NOTHING was changed."
                         % ", ".join(k for k, _a in _PANO_FIELDS))

    for attr, value in plan:
        setattr(holder, attr, value)

    out = {"camera": obj.name, "type": cam.type,
           "storedOn": "camera" if holder is cam else "camera.cycles"}
    for key, attr in _PANO_FIELDS:
        if hasattr(holder, attr):
            v = getattr(holder, attr)
            out[key] = v if isinstance(v, str) else round(float(v), 6)
    # PANORAMIC RENDERING IS A CYCLES FEATURE. On any other engine these are stored and ignored,
    # and a caller finds out when the render comes back rectilinear. Said here instead.
    engine = bpy.context.scene.render.engine
    out["renderEngine"] = engine
    out["engineHonoursPanorama"] = "CYCLES" in engine
    return out


def op_set_light_ies(params):
    """Give a light a real-world IES profile, or clear it.

    An IES file is a MEASURED photometric distribution from a fixture manufacturer - the shape of
    the light a real luminaire throws. It is how archviz and product lighting stop looking like
    computer graphics, and no amount of energy/radius/spot-angle fiddling substitutes for one.

    A LIGHT'S DISTRIBUTION IS NOT A PROPERTY, it is a NODE TREE, which is why this needs an op of
    its own rather than a key on set_light. Blender does it with an IES Texture node feeding the
    strength of an Emission shader - three nodes and two links that nothing here could author,
    because the addon had never touched a light's node tree at all.

    params:
      object (str, required)
      filepath (str)     an .ies file on disk. Loaded as EXTERNAL, so the .blend references it.
      text (str)         the IES data inline, as an INTERNAL text datablock instead
      strength (float)   the emission strength the profile scales
      clear (bool)       remove the IES Texture node(s) and their link to the Emission strength, keep the rest of
                         the tree, and switch use_nodes off where the build honours it (5.2 keeps it on)

    filepath and text are refused together - two answers to the same question - and a filepath that
    does not exist is refused BEFORE the tree is built, because a half-built node tree on a light
    that then renders black is worse than no change.
    """
    reject_unknown(params, {"object", "light", "name", "filepath", "text", "strength", "clear"},
                   "set_light_ies")
    want = take(params, "object", "light", "name", default=None, kind=str)
    if not want:
        raise MifOpError("'object' is required - which light. NOTHING was changed.")
    obj = bpy.data.objects.get(want)
    if obj is None:
        raise MifOpError("no object named '%s'. NOTHING was changed." % want)
    if obj.type != "LIGHT":
        raise MifOpError("'%s' is a %s, not a LIGHT. NOTHING was changed." % (want, obj.type))
    data = obj.data

    if take_bool(params, "clear", default=False):
        # CLEAR REMOVES THE PROFILE ITSELF (2026-10-04). It used to write use_nodes = False and stop. Measured: on 3.6.23,
        # 4.4.0 and 5.0.1 that sticks but leaves the IES Texture wired to the Emission strength in the tree; on 5.2.1 a
        # light always uses its tree and use_nodes reads True straight after False is written - so the profile stayed
        # in every render while the reply said cleared. The TEX_IES nodes go (their links with them), the rest of the
        # tree stays, and use_nodes is written off only where it holds.
        had = bool(getattr(data, "use_nodes", False))
        nt = data.node_tree
        removed = 0
        if nt is not None:
            for n in [n for n in nt.nodes if n.type == "TEX_IES"]:
                nt.nodes.remove(n)
                removed += 1
        use_nodes_held = None
        if hasattr(data, "use_nodes"):
            data.use_nodes = False
            use_nodes_held = not data.use_nodes
        nt = data.node_tree
        linked = bool(nt is not None and any(l.from_node.type == "TEX_IES" for l in nt.links))
        left = sum(1 for n in nt.nodes if n.type == "TEX_IES") if nt is not None else 0
        return {"light": obj.name, "cleared": removed > 0 or not linked, "hadNodeTree": had,
                "iesNodesRemoved": removed, "iesNodesLeft": left, "linkedToEmission": linked,
                "useNodes": bool(getattr(data, "use_nodes", False)),
                # 5.2 IGNORES use_nodes = False for a light (always True after the write); the tree is what renders.
                "useNodesSwitchHeld": use_nodes_held}

    path = take(params, "filepath", default=None, kind=str)
    text = take(params, "text", default=None, kind=str)
    if path and text:
        raise MifOpError("pass filepath OR text, not both - they are two answers to the same "
                         "question. NOTHING was changed.")
    if not path and not text:
        raise MifOpError("pass filepath (an .ies file) or text (the IES data inline), or "
                         "clear:true. NOTHING was changed.")
    if path:
        path = os.path.abspath(os.path.expanduser(str(path)))
        if not os.path.isfile(path):
            raise MifOpError("no IES file at '%s'. Refused before building the node tree, because "
                             "a half-built tree on a light that then renders black is worse than "
                             "no change. NOTHING was changed." % path)

    strength = take_float(params, "strength", default=None)

    # THE TREE. Blender's light nodes are Emission -> Light Output, and an IES Texture drives the
    # emission STRENGTH rather than its colour.
    nodes_before = data.use_nodes
    data.use_nodes = True
    nt = data.node_tree
    if nt is None:
        # PUT BACK before refusing. Turning use_nodes on is itself a change to the light - it swaps
        # what the renderer reads - so leaving it on while saying "NOTHING was changed" is false in
        # exactly the way this whole sentence is meant to rule out.
        data.use_nodes = nodes_before
        raise MifOpError("this Blender gave the light no node tree even with use_nodes set, so "
                         "there is nowhere to put an IES profile. NOTHING was changed.")

    out_node = next((n for n in nt.nodes if n.type == "OUTPUT_LIGHT"), None)
    emit = next((n for n in nt.nodes if n.type == "EMISSION"), None)
    if emit is None:
        emit = nt.nodes.new("ShaderNodeEmission")
        emit.location = (-200, 0)
    if out_node is None:
        out_node = nt.nodes.new("ShaderNodeOutputLight")
        out_node.location = (0, 0)
    # `==` not `is` - bpy re-wraps NodeLink node references, so identity fails for the same node.
    if not any(l.to_node == out_node and l.from_node == emit for l in nt.links):
        nt.links.new(emit.outputs[0], out_node.inputs[0])

    ies = next((n for n in nt.nodes if n.type == "TEX_IES"), None)
    if ies is None:
        try:
            ies = nt.nodes.new("ShaderNodeTexIES")
        except RuntimeError as exc:
            raise MifOpError("this Blender has no IES texture node (%s), so a photometric profile "
                             "cannot be applied. NOTHING usable was built." % exc)
        ies.location = (-450, 0)

    if path:
        ies.mode = "EXTERNAL"
        ies.filepath = path
    else:
        ies.mode = "INTERNAL"
        blk = bpy.data.texts.new("%s_IES" % obj.name)
        blk.from_string(str(text))
        ies.ies = blk
    if strength is not None:
        ies.inputs["Strength"].default_value = strength

    linked = any(l.from_node == ies and l.to_node == emit for l in nt.links)
    if not linked:
        nt.links.new(ies.outputs[0], emit.inputs["Strength"])
        linked = any(l.from_node == ies and l.to_node == emit for l in nt.links)

    # AN IES PROFILE IS A CYCLES FEATURE on EVERY build (re-measured 2026-09-16 on 3.6.23-5.2.1: a hand-written ring
    # profile on a point light facing a wall changed 195-205/255 in Cycles and 0.000 in EEVEE). First MEASURED on 4.4.0: a
    # narrow-beam profile took a ring 1.8 units off-axis from 173 to 51.5 under Cycles and changed
    # NOTHING under EEVEE Next - identical pixels with the profile and without. Workbench renders no
    # light node trees at all. The tree is still built, because a scene switched to Cycles later uses
    # it; the caller is told now rather than finding out from a render, the way set_camera_panorama
    # reports engineHonoursPanorama.
    engine = bpy.context.scene.render.engine
    return {
        "renderEngine": engine,
        "engineHonoursIES": "CYCLES" in engine,
        "light": obj.name,
        # THE TREE'S ADDRESS, so what was just built can be read back with list_group_nodes - by the light
        # DATABLOCK's name, which is not always the object's.
        "tree": "light:%s" % data.name,
        "useNodes": bool(data.use_nodes),
        "iesNode": ies.name,
        "mode": ies.mode,
        "filepath": getattr(ies, "filepath", "") or None,
        "internalText": ies.ies.name if getattr(ies, "ies", None) else None,
        "strength": round(float(ies.inputs["Strength"].default_value), 6),
        # THE POSTCONDITION IS THE LINK. An IES node sitting unconnected in the tree changes
        # nothing at all and looks entirely correct in the node list - the same right-looking,
        # inert shape as an invalid constraint or a dead driver.
        "linkedToEmission": linked,
        "nodeCount": len(nt.nodes),
    }


def _emissive(obj):
    """True when one of the object's materials emits: an Emission shader, or a Principled BSDF with emission strength
    above 0 and a colour that is not black."""
    for slot in getattr(obj, "material_slots", []):
        mat = slot.material
        if mat is None or mat.node_tree is None:
            continue
        for n in mat.node_tree.nodes:
            if n.type == "EMISSION":
                return True
            if n.type == "BSDF_PRINCIPLED":
                s = n.inputs.get("Emission Strength")
                c = n.inputs.get("Emission Color") or n.inputs.get("Emission")
                if s is not None and c is not None and float(s.default_value) > 0 and \
                        any(float(v) > 0 for v in list(c.default_value)[:3]):
                    return True
    return False


def _link_member_plan(raw, key):
    """[(kind, datablock, state)] for {name: INCLUDE | EXCLUDE}, every name resolved before any write."""
    if not isinstance(raw, dict) or not raw:
        raise MifOpError("%s is {object or collection name: INCLUDE | EXCLUDE}, got %r. NOTHING was changed."
                         % (key, raw))
    plan = []
    for name, state in raw.items():
        st = str(state).upper()
        if st not in ("INCLUDE", "EXCLUDE"):
            raise MifOpError("%s[%r] is INCLUDE or EXCLUDE, got %r. NOTHING was changed." % (key, name, state))
        ob = bpy.data.objects.get(str(name))
        co = bpy.data.collections.get(str(name))
        if ob is None and co is None:
            raise MifOpError("%s names %r, which is neither an object nor a collection. NOTHING was changed."
                             % (key, name))
        plan.append(("object", ob, st) if ob is not None else ("collection", co, st))
    return plan


def _apply_link_members(coll, plan):
    """Write each member's link_state, linking it into the collection first when it is not in it -> names added.
    collection_objects[i] and collection_children[i] follow collection.objects and .children (CollectionObject
    carries no object of its own on 4.2-5.2, measured)."""
    added = []
    for kind, data, state in plan:
        if kind == "object":
            if coll.objects.get(data.name) is None:
                coll.objects.link(data)
                added.append(data.name)
            i = list(coll.objects).index(data)
            coll.collection_objects[i].light_linking.link_state = state
        else:
            if coll.children.get(data.name) is None:
                coll.children.link(data)
                added.append(data.name)
            i = list(coll.children).index(data)
            coll.collection_children[i].light_linking.link_state = state
    return added


def _link_members(coll):
    """Each object and child collection of a linking collection with its INCLUDE / EXCLUDE state, read back."""
    if coll is None:
        return None
    rows = [{"name": o.name, "kind": "object", "state": coll.collection_objects[i].light_linking.link_state}
            for i, o in enumerate(coll.objects)]
    rows += [{"name": c.name, "kind": "collection", "state": coll.collection_children[i].light_linking.link_state}
             for i, c in enumerate(coll.children)]
    return rows[:100]


def op_set_light_linking(params):
    """Control WHICH objects a light affects - light linking and shadow blocking.

    "This key light hits the product and not the backdrop" is a routine request in product and
    archviz work and is impossible to fake: moving the light changes the look, flagging it off with
    geometry changes the reflections, and turning it down changes everything. Light linking is the
    only correct answer and nothing here could reach it.

    params:
      object (str, required)          the LIGHT
      receiverCollection (str)        only objects in this collection are lit by it. Created if it
                                      does not exist.
      blockerCollection (str)         only objects in this collection cast its shadows
      clearReceivers / clearBlockers (bool)

    BLENDER 4.0+ (Object.light_linking is in the 4.0 Python API and not in 3.6's; this said 4.2 until
    2026-09-28). Older builds have no light_linking at all, and this refuses by name rather than
    accepting the keys and doing nothing - which is what a hasattr-skip would have done, and is the
    mistake create_light's `shadow` key made until it was fixed today.
    """
    reject_unknown(params, {"object", "light", "name", "receiverCollection", "blockerCollection",
                            "clearReceivers", "clearBlockers", "receiverMembers", "blockerMembers"},
                   "set_light_linking")
    want = take(params, "object", "light", "name", default=None, kind=str)
    if not want:
        raise MifOpError("'object' is required - which light. NOTHING was changed.")
    obj = bpy.data.objects.get(want)
    if obj is None:
        raise MifOpError("no object named '%s'. NOTHING was changed." % want)

    # AN EMISSIVE MESH IS AN EMITTER TOO (2026-10-05): Object.light_linking is on every object, and Cycles links the
    # light of an emission material as it does a lamp's - measured on 4.2.23, 4.4.0, 5.0.1 and 5.2.2, a glowing plane
    # whose receiver collection held one of two planes lit that one (0.71) and left the other black (0.002).
    if obj.type != "LIGHT":
        if obj.type not in ("MESH", "CURVE", "SURFACE", "FONT", "META") or not _emissive(obj):
            raise MifOpError("'%s' is a %s with no emission material - light linking applies to a LIGHT or to an object "
                             "whose material emits (an Emission shader, or a Principled BSDF with emission strength "
                             "above 0). NOTHING was changed." % (want, obj.type))
    linking = getattr(obj, "light_linking", None)
    if linking is None:
        raise MifOpError("this Blender has no light_linking on an object - it arrived in 4.0, and "
                         "this build is %s. Every key here would be silently ignored, so the call "
                         "is refused instead. NOTHING was changed."
                         % ".".join(str(v) for v in bpy.app.version))

    recv = take(params, "receiverCollection", default=None, kind=str)
    block = take(params, "blockerCollection", default=None, kind=str)
    clear_r = take_bool(params, "clearReceivers", default=False)
    clear_b = take_bool(params, "clearBlockers", default=False)
    if recv and clear_r:
        raise MifOpError("pass receiverCollection OR clearReceivers, not both. NOTHING was "
                         "changed.")
    if block and clear_b:
        raise MifOpError("pass blockerCollection OR clearBlockers, not both. NOTHING was changed.")
    plans = {}
    for key, given, clear, current in (("receiverMembers", recv, clear_r, linking.receiver_collection),
                                       ("blockerMembers", block, clear_b, linking.blocker_collection)):
        if params.get(key) is None:
            continue
        if clear:
            raise MifOpError("%s sets states in the collection the clear removes. NOTHING was changed." % key)
        if not given and current is None:
            raise MifOpError("%s sets each member's state in the %s collection, and this one has none - pass %s "
                             "too. NOTHING was changed." % (key, key[:-7], key[:-7] + "Collection"))
        plans[key] = _link_member_plan(params[key], key)
    if not any((recv, block, clear_r, clear_b)) and not plans:
        raise MifOpError("nothing to do - pass receiverCollection, blockerCollection, "
                         "clearReceivers, clearBlockers, receiverMembers or blockerMembers. NOTHING was changed.")

    def _collection(name):
        c = bpy.data.collections.get(str(name))
        if c is None:
            c = bpy.data.collections.new(str(name))
        return c

    if clear_r:
        linking.receiver_collection = None
    elif recv:
        linking.receiver_collection = _collection(recv)
    if clear_b:
        linking.blocker_collection = None
    elif block:
        linking.blocker_collection = _collection(block)
    added = {}
    for key, plan in plans.items():
        coll = linking.receiver_collection if key == "receiverMembers" else linking.blocker_collection
        added[key] = _apply_link_members(coll, plan)

    rc = getattr(linking, "receiver_collection", None)
    bc = getattr(linking, "blocker_collection", None)
    return {
        "light": obj.name,
        "emitterType": obj.type,
        "receiverMembers": _link_members(rc),
        "blockerMembers": _link_members(bc),
        "membersAdded": {k: v for k, v in added.items() if v} or None,
        # WHICH ENGINE READS IT, from each build's own UI rather than a render here: Cycles from 4.0, EEVEE's light and
        # shadow linking panels from 4.4 (properties_object.py 4.4); 4.2 draws them for Cycles only.
        "honoredBy": ["CYCLES"] + (["EEVEE"] if bpy.app.version >= (4, 4, 0) else []),
        "engineNote": ("measured in Cycles on 4.2.23-5.2.2; EEVEE's linking (4.4 and later) was not rendered here"
                       if bpy.app.version >= (4, 4, 0) else "EEVEE on this build ignores light linking"),
        "receiverCollection": rc.name if rc else None,
        "receiverCount": len(rc.objects) if rc else 0,
        "blockerCollection": bc.name if bc else None,
        "blockerCount": len(bc.objects) if bc else 0,
        # AN EMPTY RECEIVER COLLECTION LIGHTS NOTHING AT ALL. That is a legitimate state to be
        # mid-setup in and a catastrophic one to render from, and it looks identical to a correct
        # link from every other field - so it is named rather than left to be discovered.
        "litsNothing": bool(rc) and len(rc.objects) == 0,
        "blenderVersion": ".".join(str(v) for v in bpy.app.version),
    }


_SHADOW_KEYS = {
    "object", "light", "name", "enabled", "softSize", "bufferClipStart", "color",
    "filterRadius", "jitter", "jitterOverblur", "maxResolution",
    "contactShadow", "contactDistance", "contactBias", "contactThickness",
    "cyclesCastShadow", "cyclesMaxBounces", "cyclesMIS", "isPortal", "isCausticsLight",
}

# param -> (where it lives, the real property name, which builds have it)
#
# EVERY ROW WAS READ OFF bl_rna ON ALL FOUR INSTALLS, not from release notes, and the drift here is
# the reason this op exists at all rather than a set_light parameter:
#
#   light.cycles.cast_shadow          3.6 ONLY - removed at 4.2
#   contact shadows (4 properties)    3.6 and 4.2 - EEVEE Next dropped them at 4.4
#   shadow_buffer_samples / _size     3.6 only
#   shadow_buffer_bias, shadow_color  3.6 and 4.2
#   jitter / filter radius / max res  4.2 and later - they did not exist on 3.6
#
# So a caller asking for a contact shadow on 4.4 is asking for something that build cannot do, and
# the house rule is that this is REFUSED with what happened to it rather than accepted and dropped.
_SHADOW_MAP = {
    "enabled":          ("light", "use_shadow", "every build"),
    "softSize":         ("light", "shadow_soft_size", "every build"),
    "bufferClipStart":  ("light", "shadow_buffer_clip_start", "every build"),
    "color":            ("light", "shadow_color", "3.6 and 4.2 only"),
    "filterRadius":     ("light", "shadow_filter_radius", "4.2 and later"),
    "jitter":           ("light", "use_shadow_jitter", "4.2 and later"),
    "jitterOverblur":   ("light", "shadow_jitter_overblur", "4.2 and later"),
    "maxResolution":    ("light", "shadow_maximum_resolution", "4.2 and later"),
    "contactShadow":    ("light", "use_contact_shadow", "3.6 and 4.2 only - EEVEE Next dropped it"),
    "contactDistance":  ("light", "contact_shadow_distance", "3.6 and 4.2 only"),
    "contactBias":      ("light", "contact_shadow_bias", "3.6 and 4.2 only"),
    "contactThickness": ("light", "contact_shadow_thickness", "3.6 and 4.2 only"),
    "cyclesCastShadow": ("cycles", "cast_shadow", "3.6 ONLY - removed at 4.2"),
    "cyclesMaxBounces": ("cycles", "max_bounces", "every build"),
    "cyclesMIS":        ("cycles", "use_multiple_importance_sampling", "every build"),
    "isPortal":         ("cycles", "is_portal", "every build"),
    "isCausticsLight":  ("cycles", "is_caustics_light", "every build"),
}

_BOOL_SHADOW = {"enabled", "jitter", "contactShadow", "cyclesCastShadow", "cyclesMIS",
                "isPortal", "isCausticsLight"}
_INT_SHADOW = {"maxResolution", "cyclesMaxBounces"}


def _shadow_holder(data, where):
    return data if where == "light" else getattr(data, "cycles", None)


# WHAT CHANGES NO PIXEL, MEASURED 2026-09-16 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1. A 1 BU cube on a floor, one
# light 3 BU up, 128 px, each key at two extremes and the pair compared per pixel (scratchpad
# measure_area_shadow_keys.py and measure_shadow_keys_by_type_engine.py):
#
#   softSize 0.05 | 1.5      POINT 5.2-6.8/255 and SPOT 4.7-28 in both engines; AREA and SUN 0.0 in EEVEE and Cycles. An
#                            AREA light's softness is its size, a SUN's its angle. The second seat found the AREA half on
#                            5.2.2 (mailbox id 43); create_light and set_light already refuse `radius`, the same
#                            property, on those two types.
#   filterRadius 0.1 | 10    EEVEE 0.27-0.50 mean, up to 124 at the shadow edge, on POINT and AREA alike - so NOT inert
#                            on an AREA light, which the same report suspected; Cycles 0.0 on both.
#   jitter off | on          EEVEE 0.22-0.33, up to 78; Cycles 0.0.
#   bufferClipStart .05|2.5  EEVEE on 3.6.23 2.40, up to 193; EEVEE Next (4.2.17 and later) 0.0; Cycles 0.0 on every build.
#   color black | red        0.0 in EEVEE and in Cycles on 3.6.23 and 4.2.17, the two builds that have it.
#
# jitterOverblur and maxResolution read 0.0 as well, but overblur was measured with jitter OFF, which gates it, and one
# scene is too little to call a resolution cap inert - so neither is claimed.
#
# REPORTED, NOT REFUSED. This op's contract is to write what the build has and say which engine reads it; a landed
# showcase (ownlight) sends softSize to an AREA light, and refusing now would break it for no picture.
def _shadow_key_measured_inert(key, light_type, engine):
    """Why this key changes nothing for this light and engine, or None - only for what was measured."""
    cycles, eevee = "CYCLES" in engine, "EEVEE" in engine
    if key == "softSize" and light_type in ("AREA", "SUN"):
        return ("shadow_soft_size does nothing on a%s %s light - its softness is %s"
                % ("n" if light_type == "AREA" else "", light_type,
                   "its size (set_light size / sizeY)" if light_type == "AREA" else "its angle (set_light angle)"))
    if key in ("filterRadius", "jitter") and cycles:
        return "an EEVEE shadow setting; Cycles does not read it"
    if key == "bufferClipStart" and (cycles or (eevee and bpy.app.version >= (4, 2, 0))):
        return "a shadow-map setting that %s does not read" % ("Cycles" if cycles else "EEVEE Next (4.2 and later)")
    if key == "color" and (cycles or eevee):
        return "changed no pixel in EEVEE or Cycles on 3.6 and 4.2, the builds that have it"
    return None


def op_set_light_shadow(params):
    """The engine-specific shadow settings, refusing what this build cannot do.

    THE GENERAL HALF IS set_object_visibility - hide_render, per-ray visibility, holdout and shadow
    catcher live on the OBJECT and apply to every type. This is the half that lives on the LIGHT and
    differs by build and by engine, which is exactly why it was left until it could be measured
    rather than remembered.

    WHAT MOVES, read off bl_rna on 3.6.23, 4.2.17, 4.4.0 and 5.0.1:

      light.cycles.cast_shadow        3.6 ONLY. Removed at 4.2 - a Cycles shadow toggle that simply
                                      stopped existing.
      contact shadows, 4 properties   3.6 and 4.2. EEVEE Next dropped them at 4.4, so a caller
                                      asking for one on 4.4 is asking for a feature the renderer no
                                      longer has.
      shadow_buffer_samples / _size   3.6 only.
      shadow_color, shadow_buffer_bias 3.6 and 4.2.
      jitter, filter radius, max res  4.2 and later. They did not exist on 3.6.

    ANYTHING THIS BUILD LACKS IS REFUSED, NAMING WHICH BUILDS HAVE IT. Accepting a key and writing
    it only `if hasattr` is the shape refuse_unsupported_shadow was written to stop: the caller asks
    for shadows off, gets shadows on, and is told nothing.

    THE ENGINE IS REPORTED because half of these do nothing under the other renderer - contact
    shadows are EEVEE's, cycles.* are Cycles'. They are still WRITTEN when present, since a scene
    is often set up under one engine and rendered with the other, but the response says which of
    them the active engine will actually read.

    params:
      object / light / name (str)   which light. Required.
      enabled (bool)                use_shadow - the master toggle
      softSize (float)              shadow_soft_size, the radius that decides how soft the edge is
      bufferClipStart (float)
      color [r,g,b]                 shadow_color - 3.6 and 4.2 only
      filterRadius (float) / jitter (bool) / jitterOverblur (float) / maxResolution (int)
                                    4.2 and later
      contactShadow (bool) / contactDistance / contactBias / contactThickness (float)
                                    3.6 and 4.2 only
      cyclesCastShadow (bool)       3.6 only
      cyclesMaxBounces (int) / cyclesMIS (bool) / isPortal (bool) / isCausticsLight (bool)
    """
    reject_unknown(params, _SHADOW_KEYS, "set_light_shadow")
    want = take(params, "object", "light", "name", required=True, kind=str)
    obj = get_object(want)
    if obj.type != "LIGHT":
        raise MifOpError("'%s' is a %s, not a LIGHT. NOTHING was changed." % (obj.name, obj.type))
    data = obj.data

    asked = [k for k in _SHADOW_MAP if params.get(k) is not None]
    if not asked:
        raise MifOpError("nothing to do - pass at least one of %s. NOTHING was changed."
                         % ", ".join(sorted(_SHADOW_MAP)))

    # SHAPE BEFORE AVAILABILITY. A malformed colour is wrong on every Blender, so reporting it
    # should not depend on interrogating this one's bl_rna - the same ordering rule ray_cast,
    # face_info and create_collision_hull each needed, and the sixth time an offline check has
    # found it by trying the guard.
    raw_colour = params.get("color")
    if raw_colour is not None and (not isinstance(raw_colour, (list, tuple))
                                   or len(raw_colour) < 3):
        raise MifOpError("'color' must be [r,g,b], got %r. NOTHING was changed." % (raw_colour,))

    # EVERY KEY CHECKED AGAINST THIS BUILD BEFORE ANY OF THEM IS WRITTEN, so a request that is
    # half-supported does not leave half of it applied.
    missing = []
    for key in asked:
        where, attr, availability = _SHADOW_MAP[key]
        holder = _shadow_holder(data, where)
        if holder is None or attr not in holder.bl_rna.properties:
            missing.append((key, attr, availability))
    if missing:
        raise MifOpError(
            "this Blender (%s) has no %s. Available on %s. Accepting the key and writing it only "
            "where it exists is how a caller asks for shadows off, gets shadows on, and is told "
            "nothing - so it is refused. NOTHING was changed."
            % (bpy.app.version_string,
               "; ".join("%s (light.%s%s)" % (k, "cycles." if _SHADOW_MAP[k][0] == "cycles" else "",
                                              a) for k, a, _ in missing),
               "; ".join(v for _, _, v in missing)))

    # EVERY VALUE PARSED BEFORE THE FIRST WRITE. They were parsed inside the write loop, so a bad later value - a
    # fractional resolution after enabled false - was refused with the earlier keys already written (found 2026-10-25).
    applied = {}
    try:
        for key in asked:
            raw = params[key]
            if key in _BOOL_SHADOW:
                value = take_bool(params, key, default=True)
            elif key in _INT_SHADOW:
                value = take_int(params, key)
            elif key == "color":
                # finite_floats: a bare float() accepts NaN, Blender STORES it, and every later read
                # of the light comes back nan while the response agrees the call worked.
                value = finite_floats(raw[:3], "color")
            else:
                value = take_float(params, key)
            applied[key] = value
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was changed." % str(exc).rstrip("."))
    for key, value in applied.items():
        where, attr, _availability = _SHADOW_MAP[key]
        setattr(_shadow_holder(data, where), attr, value)

    # READ BACK FROM THE DATABLOCK, per key. Several of these are clamped - a negative softSize or
    # an out-of-range resolution is silently corrected rather than refused - so echoing the request
    # would report a value the light does not have.
    after, wrong = {}, {}
    for key in asked:
        where, attr, _ = _SHADOW_MAP[key]
        stored = getattr(_shadow_holder(data, where), attr)
        stored = list(stored)[:3] if hasattr(stored, "__len__") else stored
        after[key] = rnd(stored) if isinstance(stored, list) else (
            round(float(stored), 6) if isinstance(stored, float) else stored)
        want_v = applied[key]
        if isinstance(want_v, bool) and bool(stored) != want_v:
            wrong[key] = (want_v, bool(stored))
        elif isinstance(want_v, (int, float)) and not isinstance(want_v, bool) \
                and abs(float(stored) - float(want_v)) > 1e-4:
            wrong[key] = (want_v, stored)

    engine = bpy.context.scene.render.engine
    is_cycles = "CYCLES" in engine
    inert = [k for k in asked
             if (_SHADOW_MAP[k][0] == "cycles" and not is_cycles)
             or (k.startswith("contact") and is_cycles)]
    # AND WHAT WAS MEASURED TO CHANGE NOTHING - see _shadow_key_measured_inert. Kept apart from `inert` above, whose note
    # promises the value applies if the engine changes: softSize on an AREA light, or color, would not.
    measured = {k: _shadow_key_measured_inert(k, data.type, engine) for k in asked}
    measured = {k: why for k, why in measured.items() if why and k not in inert}
    return {
        "ok": True,
        "light": obj.name,
        "engine": engine,
        "applied": after,
        # CLAMPED, NOT FAILED. Reported rather than raised: Blender correcting an out-of-range value
        # is legitimate, and a caller comparing what they sent against what stuck needs to see it.
        "clamped": {k: {"requested": v[0], "stored": v[1]} for k, v in wrong.items()} or None,
        # WRITTEN BUT NOT READ BY THIS ENGINE. Still written, because a scene is often set up under
        # one renderer and rendered with another - but saying so is the difference between a
        # setting that will take effect and one that is sitting there.
        "inertUnderThisEngine": inert or None,
        "note": ("%s written and this scene renders with %s, which does not read them. They are "
                 "stored and will apply if the engine changes."
                 % (", ".join(inert), engine)) if inert else None,
        "measuredNoEffect": measured or None,
        "measuredNoEffectNote": ("written, and measured to change no pixel for this %s light under %s: %s."
                                 % (data.type, engine, "; ".join("%s - %s" % kv for kv in sorted(measured.items()))))
                                if measured else None,
        "blenderVersion": bpy.app.version_string,
    }


_INFLUENCE_KEYS = {
    "object", "light", "name", "cutoffDistance", "useCustomDistance", "volumeFactor",
    "transmissionFactor", "spread", "softFalloff", "square", "showCone",
    "cascadeCount", "cascadeMaxDistance", "cascadeExponent", "cascadeFade",
}

# param -> (real property, which TYPES have it, which BUILDS have it)
#
# TWO AXES, AND THE MESSAGE HAS TO SAY WHICH ONE FAILED. set_light_shadow's map carries build
# availability only, so a property missing because of the light TYPE would be reported as "this
# Blender has no X" - true in the letter and wrong in the part that matters, because the caller
# then goes looking for the wrong fix. Every row below was read off bl_rna on 3.6.23, 4.2.17, 4.4.0
# and 5.0.1, for all four light types.
#
#   transmission_factor   4.2 and later on EVERY type - absent on 3.6
#   use_soft_falloff      4.2 and later, and only POINT and SPOT
#   spread                AREA only, every build
#   use_square/show_cone  SPOT only, every build
#   shadow_cascade_*      SUN only, every build
#   the rest              every type, every build
# MOVED TO ops_common 2026-09-15, and aliased rather than copied. light_readback needs the same map
# to REPORT these properties, and two copies of a table that must agree is the drift this module
# already avoided once for the panorama fields. One map, one truth: a property added to it is
# written by set_light_influence and reported by list_lights without either being touched.
_INFLUENCE_MAP = INFLUENCE_MAP
_INFLUENCE_BOOL = INFLUENCE_BOOL
_INFLUENCE_INT = INFLUENCE_INT


# WHAT set_light_influence WRITES AND NO RENDER READS. Reported by the second seat on 5.2.2 (mailbox id 157) and MEASURED
# 2026-09-16 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1, EEVEE and Cycles, 96x64, mean luminance difference x255
# (scratchpad measure_engine_inert_light_world.py, measure_cascade_exp_fade.py):
#   square      a 60 degree SPOT over a floor (the lit floor reads 46.7-49.3), use_square off vs on: 0.000 everywhere
#   showCone    the same spot, show_cone off vs on: 0.000 everywhere - it is viewport display
#   cascade*    a SUN 20 degrees up over 12 posts out to 200 BU (the frame reads 95.8-110.6): count 1 / max distance 6 vs
#               4 / 400, exponent 0 vs 1 and fade 0 vs 1 - 4.385, 0.966 and 0.218 under 3.6's EEVEE Legacy, 0.000 under
#               EEVEE from 4.2 and under Cycles on every build
def _influence_key_measured_inert(key, engine):
    """Why `key` was measured to change no pixel under `engine` on this build, or None."""
    if key == "square":
        return "the square spot shape changed no pixel in EEVEE or Cycles on any build measured (3.6-5.2)"
    if key == "showCone":
        return "the cone is viewport display only - it changed no pixel in EEVEE or Cycles on any build"
    if key.startswith("cascade"):
        if "CYCLES" in engine:
            return "Cycles traces sun shadows and uses no cascaded shadow map - measured no change on every build"
        if "EEVEE" in engine and bpy.app.version >= (4, 2, 0):
            return ("EEVEE from 4.2 uses virtual shadow maps and ignores the cascade settings - measured no change; "
                    "they act only under 3.6's EEVEE Legacy")
    return None


def op_set_light_influence(params):
    """How far a light reaches and what it reaches INTO - the half set_light never covered.

    THE HOLE THIS CLOSES. set_light writes energy, colour, cone and size; set_light_shadow writes
    the shadow settings. Nothing wrote a light's INFLUENCE RADIUS, its volumetric contribution, an
    area light's spread, or a sun's shadow cascades - and those are most of what separates a lighting
    rig that renders well from one that merely exists. cutoff_distance in particular is the
    performance knob: it is the distance past which a light is not evaluated at all, the direct
    equivalent of an attenuation radius in a game engine, and there was no way to set it.

    A CUTOFF DOES NOTHING UNTIL use_custom_distance IS ON. Blender stores cutoff_distance whether or
    not the toggle is set, so writing it alone changes a number and not the render - and every field
    reads back exactly as asked. Passing cutoffDistance turns the toggle on unless useCustomDistance
    says otherwise, and the response says which happened.

    TWO AXES OF AVAILABILITY, and the refusal says which one failed. A property can be missing
    because this BUILD does not have it (transmission_factor is 4.2 and later) or because this light
    TYPE does not have it (spread is AREA only, cascades are SUN only). set_light_shadow's map
    carries build availability alone, so a type mismatch there reads as "this Blender has no X" -
    true in the letter and wrong in the part that decides what the caller does next.

    params:
      object / light / name (str)   which light. Required.
      cutoffDistance (float)        the distance past which the light is not evaluated
      useCustomDistance (bool)      whether the cutoff applies at all
      volumeFactor (float)          contribution to volumetrics
      transmissionFactor (float)    contribution through transmissive surfaces - 4.2 and later
      spread (float)                AREA only - the angle the light emits over
      softFalloff (bool)            POINT and SPOT, 4.2 and later
      square (bool) / showCone (bool)   SPOT only
      cascadeCount (int) / cascadeMaxDistance (float) / cascadeExponent (float) /
      cascadeFade (float)           SUN only - the cascaded shadow map settings
    """
    reject_unknown(params, _INFLUENCE_KEYS, "set_light_influence")
    want = take(params, "object", "light", "name", required=True, kind=str)
    obj = get_object(want)
    if obj.type != "LIGHT":
        raise MifOpError("'%s' is a %s, not a LIGHT. NOTHING was changed." % (obj.name, obj.type))
    data = obj.data
    kind = str(data.type)

    asked = [k for k in _INFLUENCE_MAP if params.get(k) is not None]
    if not asked:
        raise MifOpError("nothing to do - pass at least one of %s. NOTHING was changed."
                         % ", ".join(sorted(_INFLUENCE_MAP)))

    # EVERY KEY CHECKED BEFORE ANY IS WRITTEN, so a half-supported request does not leave half of
    # itself applied - and the two reasons are reported separately.
    wrong_type, wrong_build = [], []
    for key in asked:
        attr, types, availability = _INFLUENCE_MAP[key]
        if types is not None and kind not in types:
            wrong_type.append((key, attr, types))
        elif attr not in data.bl_rna.properties:
            wrong_build.append((key, attr, availability))
    if wrong_type:
        raise MifOpError(
            "'%s' is of type %s, and %s. That is the light TYPE, not this Blender - retyping the "
            "light with set_light would make it available. NOTHING was changed."
            % (obj.name, kind,
               "; ".join("%s (light.%s) exists only on %s" % (k, a, "/".join(ts))
                         for k, a, ts in wrong_type)))
    if wrong_build:
        raise MifOpError(
            "this Blender (%s) has no %s. Available on %s. Accepting the key and writing it only "
            "where it exists is how a caller asks for something, does not get it, and is told "
            "nothing - so it is refused. NOTHING was changed."
            % (bpy.app.version_string,
               "; ".join("%s (light.%s)" % (k, a) for k, a, _ in wrong_build),
               "; ".join(v for _, _, v in wrong_build)))

    # AND EVERY VALUE PARSED BEFORE THE FIRST WRITE, as the docstring promised: they were parsed inside the write loop,
    # so a fractional cascadeCount was refused after the keys before it were written (found 2026-10-25).
    applied = {}
    try:
        for key in asked:
            if key in _INFLUENCE_BOOL:
                applied[key] = take_bool(params, key, default=True)
            elif key in _INFLUENCE_INT:
                applied[key] = take_int(params, key)
            else:
                applied[key] = take_float(params, key)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was changed." % str(exc).rstrip("."))
    for key, value in applied.items():
        setattr(data, _INFLUENCE_MAP[key][0], value)

    # THE TOGGLE THAT MAKES THE DISTANCE MEAN ANYTHING. Blender stores cutoff_distance whether or
    # not use_custom_distance is on, so a caller who sets only the distance gets a stored number and
    # an unchanged render, with every field reading back exactly as requested.
    auto_enabled = False
    if "cutoffDistance" in applied and "useCustomDistance" not in applied \
            and not data.use_custom_distance:
        data.use_custom_distance = True
        auto_enabled = True

    # READ BACK PER KEY. Several of these clamp - a negative spread, a cascade count outside its
    # range - and echoing the request would report a value the light does not have.
    after, clamped = {}, {}
    for key in asked:
        stored = getattr(data, _INFLUENCE_MAP[key][0])
        after[key] = round(float(stored), 6) if isinstance(stored, float) else stored
        wanted = applied[key]
        if isinstance(wanted, (int, float)) and not isinstance(wanted, bool) \
                and abs(float(stored) - float(wanted)) > 1e-4:
            clamped[key] = {"requested": wanted, "stored": after[key]}

    engine = bpy.context.scene.render.engine
    measured = {k: _influence_key_measured_inert(k, engine) for k in asked}
    measured = {k: why for k, why in measured.items() if why}
    return {
        "ok": True,
        "light": obj.name,
        "lightType": kind,
        "engine": engine,
        "applied": after,
        "clamped": clamped or None,
        # WRITTEN AND STORED, and measured to change nothing in a render - see _influence_key_measured_inert.
        "measuredNoEffect": measured or None,
        "measuredNoEffectNote": ("written, and measured to change no pixel for this %s light under %s: %s."
                                 % (kind, engine, "; ".join("%s - %s" % kv for kv in sorted(measured.items()))))
                                if measured else None,
        "useCustomDistance": bool(data.use_custom_distance),
        # SAID OUT LOUD, because turning a toggle on that the caller did not mention is a real
        # change to how the light renders and should not be discovered later.
        "cutoffWasEnabledAutomatically": auto_enabled,
        "note": ("use_custom_distance was OFF, so the cutoff distance would have been stored and "
                 "ignored - it has been turned ON. Pass useCustomDistance:false to store the "
                 "distance without applying it.") if auto_enabled else None,
        "blenderVersion": bpy.app.version_string,
    }

_SET_PROBE_KEYS = {"object", "name", "settings"}

# KEPT OUT, AND WHY. The grid resolution (resolution_x/y/z from 4.2, grid_resolution_x/y/z on 3.6) is out of
# create_light_probe by a standing decision - the names differ by build - and the same reason holds here; whether this
# call should take them is an open decision, not a forgotten key. `type` is the probe's class from 4.2 (read-only
# there), so a different kind of probe is a new probe.
_PROBE_DENY = frozenset({"resolution_x", "resolution_y", "resolution_z", "grid_resolution_x", "grid_resolution_y",
                         "grid_resolution_z", "type", "name"})


def op_set_light_probe(params):
    """Write a light probe's own settings by their RNA identifiers - its bake settings above all - read back each.

    WHY (2026-10-04). create_light_probe sets influence, falloff and clip only, and no writer reached a probe's bake
    settings: a volume probe whose Surfel Resolution (surfel_density) default of 20 was far too low, Surface Offset
    (surface_bias) raised from 0 against banding, Search Distance (capture_distance) kept small, normal_bias lowered
    and view_bias raised against bleed under objects, and the viewport data display (use_data_display,
    data_display_size) on to see the points. Read off the probe's own bl_rna, as set_modifier reads a modifier's:
    enums are checked against their items, numbers must be finite, and an absent name is refused with the names this
    build's probe has - the 3.6 probe is one class and the 4.2+ probes three (sphere, plane, volume), with different
    fields.

    ALL OR NOTHING: every value is snapshotted before the first write and put back on any refusal.

    params:
      object (str, required)     a LIGHT_PROBE object (alias name)
      settings {identifier: value}
    """
    reject_unknown(params, _SET_PROBE_KEYS, "set_light_probe", {
        "resolution": "the grid resolution (resolution_x/y/z, grid_resolution_x/y/z on 3.6) is kept out by a standing "
                      "decision, as in create_light_probe.",
        "bake": "this writes the settings only; baking the probe is not done here.",
    })
    obj = get_object(take(params, "object", "name", required=True))
    if obj.type != "LIGHT_PROBE":
        raise MifOpError("'%s' is a %s, not a light probe - create_light_probe makes one. NOTHING was changed."
                         % (obj.name, obj.type))
    settings = params.get("settings")
    if not isinstance(settings, dict) or not settings:
        raise MifOpError("settings must be a non-empty object of {identifier: value}, e.g. {\"surfel_density\": 60}. "
                         "NOTHING was changed.")
    data = obj.data
    kind = data.bl_rna.identifier
    tail = "NOTHING was changed."
    denied = sorted(k for k in settings if k in _PROBE_DENY)
    if denied:
        raise MifOpError("%s %s not written by this call: the grid resolution is kept out by a standing decision (its "
                         "names differ by build: resolution_x/y/z from 4.2, grid_resolution_x/y/z on 3.6), and type "
                         "and name make a different probe. %s" % (", ".join(denied), "is" if len(denied) == 1 else "are",
                                                                  tail))
    before = {}
    for key in settings:
        prop = data.bl_rna.properties.get(key)
        if prop is not None and not prop.is_readonly:
            val = getattr(data, key)
            before[key] = list(val) if hasattr(val, "__len__") and not isinstance(val, str) else val
    after = {}
    try:
        for key, value in settings.items():
            after[key] = write_rna_property(data, key, value, tail, "light probe '%s' (%s)" % (obj.name, kind), kind,
                                            _PROBE_DENY)
    except MifOpError:
        for key, val in before.items():
            try:
                setattr(data, key, val)
            except (TypeError, ValueError, AttributeError):
                pass
        raise
    return {"object": obj.name, "probe": data.name, "probeClass": kind, "probeType": data.type,
            "written": after, "before": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in before.items()}}


_BAKE_PROBE_KEYS = {"object", "name", "all", "type"}


def _legacy_light_cache():
    """True on the EEVEE whose light cache is one scene-wide bake (3.6): SceneEEVEE carries gi_cache_info there."""
    return "gi_cache_info" in bpy.types.SceneEEVEE.bl_rna.properties


def _probe_kind(data):
    """VOLUME, SPHERE or PLANE for a probe on any build (3.6 reports GRID / CUBEMAP / PLANAR)."""
    for group in _PROBE_SYNONYMS:
        if data.type in group:
            return group[0]
    return data.type


def _probe_points(data):
    """The sample count of a volume probe - resolution_x*y*z from 4.2, grid_resolution_* on 3.6. Read only."""
    for pre in ("resolution_", "grid_resolution_"):
        if hasattr(data, pre + "x"):
            return int(getattr(data, pre + "x")) * int(getattr(data, pre + "y")) * int(getattr(data, pre + "z"))
    return None


def _world_box(obj):
    pts = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    return ([min(p[i] for p in pts) for i in range(3)], [max(p[i] for p in pts) for i in range(3)])


def _volume_box(obj):
    """A volume probe's world bounds: its unit cube (-1..1) through the object's matrix."""
    pts = [obj.matrix_world @ mathutils.Vector((x, y, z)) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    return ([min(p[i] for p in pts) for i in range(3)], [max(p[i] for p in pts) for i in range(3)])


def op_bake_light_probe(params):
    """Bake EEVEE's light probe cache - a volume probe's indirect light - and report the points, the time and the state.

    WHY (2026-10-06). create_light_probe places a probe and set_light_probe writes its settings, and nothing baked it,
    so a volume probe changed no pixel. Measured with a closed 6 m room, a spot light pointing down and a camera
    looking at the ceiling that only bounce light reaches (128 px, mean luminance):

        3.6.23  scene.light_cache_bake               0.0016  -> 0.3203
        4.2.17  object.lightprobe_cache_bake ALL      0.00158 -> 0.14218 (lightprobe_cache_free: back to 0.00158)
        5.2.1   object.lightprobe_cache_bake ALL      0.00033 -> 0.15151

    THE OPERATOR DIFFERS BY LIGHTING GENERATION. 3.6 bakes one light cache for the whole scene (subset ALL takes the
    volume grids AND the reflection cubemaps, CUBEMAPS the cubemaps alone); 4.2 and later bake volume probes only -
    sphere and plane probes update live there and have nothing to bake - one probe (ACTIVE) or every visible one (ALL).
    So on 3.6 an `object` is refused (pass all) and on 4.2+ a SPHERE or PLANE bake is refused.

    SURFACES OUTSIDE A VOLUME GET NOTHING FROM IT on 4.2+. A probe scaled 2.9 inside the 3.0 room baked real data
    (L0 up to 3.98, read out of the saved .blend) and the ceiling, 0.1 outside it, did not change by one digit; scaled
    3.3 the same bake lit it. 3.6 extends the edge samples and lit it both ways. outsideVolumes names the visible
    meshes whose bounds are not wholly inside a baked volume.

    THE STATE. 3.6 reads it as scene.eevee.gi_cache_info ("No light cache in this scene" -> "0 Ref. Cubemaps, 64 Irr.
    Samples (8.5 MiB in memory)"). From 4.2 the cache lives on the object in DNA with no RNA, and
    lightprobe_cache_free polls True with nothing baked, so cacheInfo is null there and the reply says what was
    baked rather than reading a state Blender does not expose.

    The grid resolution stays out of every writer by the standing decision (create_light_probe); points reads it.

    params:
      object (str)       one volume probe (4.2+); alias name
      all (bool)         every visible probe of `type`; exactly one of object / all
      type (str)         VOLUME (default; GRID) | SPHERE (CUBE, CUBEMAP; 3.6 only) | PLANE (refused: nothing to bake)
    """
    reject_unknown(params, _BAKE_PROBE_KEYS, "bake_light_probe", {
        "resolution": "the grid resolution is kept out by a standing decision (create_light_probe); the reply's points "
                      "reads it.",
        "subset": "pass object for one probe or all for every visible one.",
    })
    tail = "NOTHING was baked."
    obj_name = take(params, "object", "name", default=None, kind=str)
    do_all = take_bool(params, "all", default=False)
    if bool(obj_name) == bool(do_all):
        raise MifOpError("pass exactly one of object (a probe name) or all:true. %s" % tail)
    wanted = str(take(params, "type", default="VOLUME", kind=str)).upper()
    kind = next((g[0] for g in _PROBE_SYNONYMS if wanted in g), None)
    if kind is None:
        raise MifOpError("type must be VOLUME (GRID), SPHERE (CUBE, CUBEMAP) or PLANE (PLANAR), got %r. %s"
                         % (wanted, tail))
    # THE OBJECT IS RESOLVED FIRST: a missing name is the commonest mistake and must read as get_object's refusal on
    # every build, before anything asks which lighting generation this is.
    obj = get_object(obj_name) if obj_name else None
    legacy = _legacy_light_cache()
    build = bpy.app.version_string
    if kind == "PLANE":
        raise MifOpError("plane probes are not baked on any build - EEVEE renders their reflection live. %s" % tail)
    if kind == "SPHERE" and not legacy:
        raise MifOpError("from 4.2 EEVEE updates sphere probes live and bakes volume probes only (this is %s). %s"
                         % (build, tail))
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    if obj_name:
        if legacy:
            raise MifOpError("Blender %s bakes one light cache for the whole scene, not one probe - pass all:true "
                             "(type VOLUME bakes the grids and the cubemaps, SPHERE the cubemaps alone). %s"
                             % (build, tail))
        if obj.type != "LIGHT_PROBE" or _probe_kind(obj.data) != "VOLUME":
            raise MifOpError("'%s' is %s, not a VOLUME light probe - only volume probes bake on %s. %s"
                             % (obj.name, obj.type if obj.type != "LIGHT_PROBE" else "a %s probe" % obj.data.type,
                                build, tail))
        if not obj.visible_get(view_layer=view_layer):
            raise MifOpError("'%s' is hidden in the view layer, and the bake skips hidden probes. %s"
                             % (obj.name, tail))
        probes = [obj]
    else:
        probes = [o for o in view_layer.objects if o.type == "LIGHT_PROBE" and _probe_kind(o.data) == kind
                  and o.visible_get(view_layer=view_layer)]
        if not probes:
            raise MifOpError("no visible %s light probe in the view layer - create_light_probe makes one. %s"
                             % (kind, tail))

    warnings = []
    if not any(o.type == "LIGHT" and o.visible_get(view_layer=view_layer) for o in view_layer.objects):
        warnings.append("no visible light in the scene: a bake sees only the world and emissive surfaces, and with "
                        "neither it is black.")
    if "EEVEE" not in scene.render.engine:
        warnings.append("the scene renders with %s; only EEVEE reads this cache." % scene.render.engine)

    info_before = scene.eevee.gi_cache_info if legacy else None
    t0 = time.perf_counter()
    if legacy:
        operator = "scene.light_cache_bake"
        res = bpy.ops.scene.light_cache_bake(subset="ALL" if kind == "VOLUME" else "CUBEMAPS")
    else:
        operator = "object.lightprobe_cache_bake"
        if obj_name:
            snap = selection_snapshot()
            try:
                for o in view_layer.objects:
                    o.select_set(o == probes[0])
                view_layer.objects.active = probes[0]
                res = bpy.ops.object.lightprobe_cache_bake(subset="ACTIVE")
            finally:
                selection_restore(snap)
        else:
            res = bpy.ops.object.lightprobe_cache_bake(subset="ALL")
    seconds = time.perf_counter() - t0
    if "FINISHED" not in res:
        raise MifOpError("%s returned %s on %s. The cache may be partly written." % (operator, sorted(res), build))

    rows, outside = [], []
    boxes = []
    for p in probes:
        lo, hi = _volume_box(p) if kind == "VOLUME" else _world_box(p)
        boxes.append((lo, hi))
        rows.append({"name": p.name, "type": p.data.type, "points": _probe_points(p.data) if kind == "VOLUME" else None,
                     "boundsMin": rnd(lo), "boundsMax": rnd(hi)})
    if kind == "VOLUME":
        for o in view_layer.objects:
            if o.type != "MESH" or not o.visible_get(view_layer=view_layer) or o.hide_render:
                continue
            lo, hi = _world_box(o)
            if not any(all(blo[i] <= lo[i] + 1e-4 and hi[i] <= bhi[i] + 1e-4 for i in range(3)) for blo, bhi in boxes):
                outside.append(o.name)
    out = {"operator": operator, "subset": ("ALL" if kind == "VOLUME" else "CUBEMAPS") if legacy else
           ("ACTIVE" if obj_name else "ALL"), "type": kind, "probes": rows, "probeCount": len(rows),
           "points": sum(r["points"] or 0 for r in rows) if kind == "VOLUME" else None,
           "bakeSeconds": round(seconds, 3), "engine": scene.render.engine,
           "cacheInfoBefore": info_before, "cacheInfoAfter": scene.eevee.gi_cache_info if legacy else None,
           "outsideVolumes": outside[:25], "outsideVolumeCount": len(outside), "warnings": warnings,
           "blenderVersion": build}
    if legacy:
        out["note"] = ("3.6 bakes the scene's one light cache: every volume grid and reflection cubemap"
                       if kind == "VOLUME" else "3.6 baked the reflection cubemaps of the scene's light cache")
    else:
        out["note"] = ("from 4.2 Blender exposes no baked state to Python (the cache is DNA on the probe object), so "
                       "cacheInfo is null; the probes listed were baked. Surfaces outside a volume get nothing from it.")
    return out


OPS = {
    "set_light_linking": op_set_light_linking,
    "set_light_shadow": op_set_light_shadow,
    "set_light_influence": op_set_light_influence,
    "set_light_ies": op_set_light_ies,
    "set_camera_panorama": op_set_camera_panorama,
    "create_light": op_create_light,
    "create_light_probe": op_create_light_probe,
    "set_light_probe": op_set_light_probe,
    "bake_light_probe": op_bake_light_probe,
    "set_light": op_set_light,
    "list_lights": op_list_lights,
    "create_camera": op_create_camera,
    "set_camera": op_set_camera,
    "set_camera_display": op_set_camera_display,
    "list_cameras": op_list_cameras,
    "aim_object": op_aim_object,
}
