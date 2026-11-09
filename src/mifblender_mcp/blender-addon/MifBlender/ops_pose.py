"""MifBlender ops: armature display and pose options, keying a pose, copying a pose (mirrored or not), and pose assets.

THE GAP (spec, "Blender: armature options, weights, poses and assets" and "Blender: poses and keys", 2026-10-04): a pose
could be written one bone and one call at a time (set_bone_pose) and keyed one path at a time (set_keyframe), and
nothing could key a whole pose as it stands, paste it flipped onto the other side, switch the rig to its rest position,
or store a pose as an asset and apply it again.

THE MIRROR RULE, shared by every op here that flips (and by set_bone_pose mirror, set_keyframes' flipped copy and
apply_pose_asset flipped). For a rig whose two sides are mirror images across the armature's X plane - a partner's rest
is S R S with S = diag(-1, 1, 1), which is what X-axis mirror editing makes (roll negated) - the partner's pose is
S B S. In channels that is Blender's own paste-flipped rule: location x negated; an euler's Y and Z negated (any order);
a quaternion's y and z negated; an axis-angle's y and z negated; scale kept. A bone with no side is flipped in place.
Whether the rig really is a mirror image is MEASURED, not assumed: every flip reads both evaluated matrices back and
reports the mirror error (world units and degrees) - a rig whose sides differ shows up there as a large number.

THE SIDE OF A NAME is a parameter, never assumed (rule (m) of the closing research pass): FLIP_NAME is Blender's own
bpy.utils.flip_name, MEASURED IDENTICAL on 3.6.23, 4.4.0 and 5.2.1 for 27 names (arm.L/arm.R, arm_l, arm_L, arm-l,
'arm l', L_arm, left_arm, Left_arm, armLeft, arm.left, DEF-upper_arm.L, hand.L.001 -> hand.R.001; spine stays spine); a
pair [a, b] swaps that suffix only; a table {name: partner} maps both ways.

BONE SELECTION MOVED IN 5.x: Bone.select on 3.6 and 4.4, PoseBone.select on 5.2 where Bone.select is gone (measured:
"'Bone' object has no attribute 'select'"). _select_bones writes whichever this build has, and puts it back.
"""
from __future__ import annotations

import math
import os
import re
import uuid

import bpy

from .ops_common import (MifOpError, simulation_guard, get_object, reject_unknown, require_editable, rnd,
                         take, take_bool, take_float)


# --------------------------------------------------------------------------------------------- shared helpers

def _armature(params, verb="changed"):
    obj = get_object(take(params, "object", "name", required=True, kind=str))
    if obj.type != "ARMATURE":
        raise MifOpError("'%s' is a %s, not an ARMATURE. NOTHING was %s." % (obj.name, obj.type, verb))
    if obj.pose is None:
        raise MifOpError("'%s' has no pose data. NOTHING was %s." % (obj.name, verb))
    return obj


def _select_attr():
    """The bone selection flag on this build: PoseBone.select (5.x) or Bone.select (3.6, 4.x)."""
    return "pose" if "select" in bpy.types.PoseBone.bl_rna.properties else "bone"


def select_bones(obj, names):
    """Select exactly `names` on obj's pose bones; returns what to hand restore_bone_selection."""
    where = _select_attr()
    before = {}
    for pb in obj.pose.bones:
        holder = pb if where == "pose" else pb.bone
        before[pb.name] = bool(holder.select)
        holder.select = pb.name in names
    return before


def restore_bone_selection(obj, before):
    where = _select_attr()
    for pb in obj.pose.bones:
        if pb.name in before:
            (pb if where == "pose" else pb.bone).select = before[pb.name]


def pick_bones(obj, params, verb, default_all=False):
    """bones [names] | namePrefix | all true -> pose bones, in armature order. Exactly one of the three (or none when
    default_all). A named bone that does not exist is refused before anything is written."""
    names = params.get("bones")
    prefix = take(params, "namePrefix", default=None, kind=str)
    every = take_bool(params, "all", default=False)
    given = sum(1 for x in (names is not None, prefix is not None, every) if x)
    if given > 1:
        raise MifOpError("pass one of bones, namePrefix or all:true. NOTHING was %s." % verb)
    if given == 0:
        if default_all:
            return list(obj.pose.bones)
        raise MifOpError("pass bones [names], namePrefix or all:true. NOTHING was %s." % verb)
    if names is not None:
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
            raise MifOpError("bones must be a non-empty list of bone names. NOTHING was %s." % verb)
        missing = [n for n in names if obj.pose.bones.get(n) is None]
        if missing:
            raise MifOpError("'%s' has no bone(s) %s. NOTHING was %s." % (obj.name, ", ".join(missing[:20]), verb))
        wanted = set(names)
        return [pb for pb in obj.pose.bones if pb.name in wanted]
    if prefix is not None:
        found = [pb for pb in obj.pose.bones if pb.name.startswith(prefix)]
        if not found:
            raise MifOpError("no bone of '%s' starts with %r. NOTHING was %s." % (obj.name, prefix, verb))
        return found
    return list(obj.pose.bones)


_NAME_RULE_HELP = ("nameRule is 'FLIP_NAME' (Blender's own bpy.utils.flip_name: .L/.R, _l/_r, -l, L_, left/right ...), "
                   "a suffix pair like ['.L', '.R'] or ['_l', '_r'], or a table {name: partner}")


def parse_name_rule(raw, key="nameRule", verb="changed"):
    """A side rule, validated: ('FLIP_NAME', None) | ('PAIR', (a, b)) | ('TABLE', {a: b, b: a})."""
    if isinstance(raw, str) and raw.upper() == "FLIP_NAME":
        return ("FLIP_NAME", None)
    if isinstance(raw, (list, tuple)):
        if len(raw) != 2 or any(not isinstance(s, str) or not s for s in raw) or raw[0] == raw[1]:
            raise MifOpError("%s as a pair is two different suffixes, like ['.L', '.R']; got %r. NOTHING was %s."
                             % (key, raw, verb))
        return ("PAIR", (raw[0], raw[1]))
    if isinstance(raw, dict) and raw:
        table = {}
        for a, b in raw.items():
            if not isinstance(b, str) or not b or a == b:
                raise MifOpError("%s as a table maps a bone name to a different bone name; got %r: %r. NOTHING was %s."
                                 % (key, a, b, verb))
            for x, y in ((a, b), (b, a)):
                if table.get(x, y) != y:
                    raise MifOpError("%s maps '%s' to two different partners. NOTHING was %s." % (key, x, verb))
                table[x] = y
        return ("TABLE", table)
    raise MifOpError("%s; got %r. NOTHING was %s." % (_NAME_RULE_HELP, raw, verb))


def partner_name(name, rule):
    """The other side's name under `rule`, or None for a name with no side."""
    kind, data = rule
    if kind == "FLIP_NAME":
        other = bpy.utils.flip_name(name)
        return other if other != name else None
    if kind == "PAIR":
        a, b = data
        if name.endswith(a):
            return name[: -len(a)] + b
        if name.endswith(b):
            return name[: -len(b)] + a
        return None
    return data.get(name)


def rotation_path(pb):
    return {"QUATERNION": "rotation_quaternion", "AXIS_ANGLE": "rotation_axis_angle"}.get(pb.rotation_mode,
                                                                                            "rotation_euler")


def read_channels(pb):
    """A pose bone's own channels: (rotation_mode, location, rotation in its mode, scale)."""
    return (pb.rotation_mode, list(pb.location), list(getattr(pb, rotation_path(pb))), list(pb.scale))


def flip_channels(ch):
    """The mirror image of a bone's channels across X - Blender's paste-flipped rule (see the module docstring)."""
    mode, loc, rot, scl = ch
    loc = [-loc[0], loc[1], loc[2]]
    if mode in ("QUATERNION", "AXIS_ANGLE"):
        rot = [rot[0], rot[1], -rot[2], -rot[3]]
    else:
        rot = [rot[0], -rot[1], -rot[2]]
    return (mode, loc, rot, list(scl))


def write_channels(pb, ch):
    """Write channels onto a pose bone. Same rotation mode: the values as they are. Another mode: through the basis
    matrix, which Blender decomposes into the bone's own mode. Returns True when the matrix route was taken."""
    mode, loc, rot, scl = ch
    if mode == pb.rotation_mode:
        pb.location = loc
        setattr(pb, rotation_path(pb), rot)
        pb.scale = scl
        return False
    pb.matrix_basis = channels_matrix(ch)
    return True


def channels_matrix(ch):
    from mathutils import Euler, Matrix, Quaternion, Vector  # noqa: PLC0415 - offline suites stub mathutils
    mode, loc, rot, scl = ch
    if mode == "QUATERNION":
        r = Quaternion(rot).normalized().to_matrix()
    elif mode == "AXIS_ANGLE":
        r = Quaternion(Vector(rot[1:]).normalized() if Vector(rot[1:]).length else Vector((0, 1, 0)), rot[0]).to_matrix()
    else:
        r = Euler(rot, mode).to_matrix()
    return Matrix.Translation(loc) @ r.to_4x4() @ Matrix.Diagonal((scl[0], scl[1], scl[2], 1.0))


def evaluated_pose(obj):
    """{bone: armature-space matrix} through the depsgraph, after an update."""
    bpy.context.view_layer.update()
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return {pb.name: pb.matrix.copy() for pb in ev.pose.bones}


def matrix_error(obj, expected, actual):
    """(world distance, degrees) between two armature-space matrices of obj."""
    w = obj.matrix_world
    a, b = w @ expected, w @ actual
    dist = (a.to_translation() - b.to_translation()).length
    qa, qb = a.to_3x3().normalized().to_quaternion(), b.to_3x3().normalized().to_quaternion()
    ang = math.degrees(qa.rotation_difference(qb).angle)
    return dist, min(ang, 360.0 - ang)


def mirrored(m):
    """S m S, the X-mirror of an armature-space matrix."""
    from mathutils import Matrix  # noqa: PLC0415
    s = Matrix.Diagonal((-1.0, 1.0, 1.0, 1.0))
    return s @ m @ s


def mirror_report(obj, pairs, source_mats, flip=True):
    """For [(source bone, written bone)], the error between each written bone's evaluated matrix and the mirror (or the
    copy) of its source's - rows for the worst ones, and the maxima."""
    now = evaluated_pose(obj)
    rows = []
    for src, dst in pairs:
        want = mirrored(source_mats[src]) if flip else source_mats[src]
        d, a = matrix_error(obj, want, now[dst])
        rows.append({"from": src, "to": dst, "errorUnits": round(d, 6), "errorDegrees": round(a, 4)})
    rows.sort(key=lambda r: (-r["errorDegrees"], -r["errorUnits"]))
    return {"maxErrorUnits": max((r["errorUnits"] for r in rows), default=0.0),
            "maxErrorDegrees": max((r["errorDegrees"] for r in rows), default=0.0),
            "worst": rows[:10]}


def _interp_ids():
    return [i.identifier for i in bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items]


def key_bone_channels(pb, frame, paths, interp=None):
    """keyframe_insert each path of a pose bone at frame (the same call on 3.6 to 5.2), in the bone's group; then the
    keys at that frame are counted off the curves and given `interp`. Returns the key count."""
    for path in paths:
        pb.keyframe_insert(path, frame=frame, group=pb.name)
    from .ops_anim import _fcurves  # noqa: PLC0415 - reads 5.x slotted actions too
    obj = pb.id_data
    full = {pb.path_from_id(p) if not p.startswith("[") else 'pose.bones["%s"]%s' % (pb.name, p) for p in paths}
    n = 0
    for fc in _fcurves(obj):
        if fc.data_path not in full:
            continue
        for kp in fc.keyframe_points:
            if abs(kp.co[0] - frame) < 1e-4:
                if interp:
                    kp.interpolation = interp
                n += 1
    return n


# --------------------------------------------------------------------------------------------- armature options

# (param, owner, property, what it is) - owner "data" is the Armature, "pose" the object's Pose, "object" the Object.
_ARM_OPTIONS = (
    ("poseMode", "data", "pose_position", "Pose Position: POSE, or REST to show the rest pose while keeping the pose"),
    ("displayType", "data", "display_type", "Display As"),
    ("showNames", "data", "show_names", "bone names in the viewport"),
    ("showAxes", "data", "show_axes", "bone axes in the viewport"),
    ("showBoneShapes", "data", "show_bone_custom_shapes", "custom bone shapes"),
    ("showBoneColors", "data", ("show_bone_colors", "show_group_colors"), "bone colors (Group Colors on 3.6)"),
    ("xMirrorEdit", "data", "use_mirror_x", "edit-mode X-Axis Mirror"),
    ("xMirrorPose", "pose", "use_mirror_x", "pose-mode X-Axis Mirror (interactive transforms only)"),
    ("autoIK", "pose", "use_auto_ik", "Auto IK (interactive posing only)"),
    ("showInFront", "object", "show_in_front", "In Front: bones drawn through the mesh"),
)


_ARMPROPS_KEYS = {"object", "name", "poseMode", "displayType", "showNames", "showAxes", "showBoneShapes",
                  "showBoneColors", "xMirrorEdit", "xMirrorPose", "autoIK", "showInFront"}


def _arm_holder(obj, owner):
    return {"data": obj.data, "pose": obj.pose, "object": obj}[owner]


def _arm_prop(obj, owner, prop):
    holder = _arm_holder(obj, owner)
    for name in (prop if isinstance(prop, tuple) else (prop,)):
        if name in holder.bl_rna.properties:
            return holder, name
    return holder, None


def _arm_read(obj):
    out = {}
    for key, owner, prop, _what in _ARM_OPTIONS:
        holder, name = _arm_prop(obj, owner, prop)
        out[key] = getattr(holder, name) if name else None
    return out


def op_set_armature_properties(params):
    """Display and pose options of an existing armature, each read back - see the MCP wrapper for the parameters.

    The options live on three owners: the Armature data (pose_position, display_type, show_names, show_axes,
    show_bone_custom_shapes, show_bone_colors - show_group_colors on 3.6 - and the edit-mode use_mirror_x), the
    object's Pose (use_auto_ik and the pose-mode use_mirror_x) and the Object (show_in_front). Enum values are read
    from this build's RNA. The reply carries `before`, so a REST hold can be undone with its poseMode.
    """
    reject_unknown(params, _ARMPROPS_KEYS, "set_armature_properties")
    keys = [k for k, _o, _p, _w in _ARM_OPTIONS]
    obj = _armature(params)
    require_editable(obj, "change")
    plan = []
    for key, owner, prop, what in _ARM_OPTIONS:
        if params.get(key) is None:
            continue
        holder, name = _arm_prop(obj, owner, prop)
        if name is None:
            raise MifOpError("%s (%s) does not exist on this Blender (%s). NOTHING was changed."
                             % (key, what, bpy.app.version_string))
        rp = holder.bl_rna.properties[name]
        value = params[key]
        if rp.type == "ENUM":
            ids = [i.identifier for i in rp.enum_items]
            if not isinstance(value, str) or value.upper() not in ids:
                raise MifOpError("%s must be one of %s on this Blender, got %r. NOTHING was changed."
                                 % (key, ", ".join(ids), value))
            value = value.upper()
        elif not isinstance(value, bool):
            raise MifOpError("%s is true or false, got %r. NOTHING was changed." % (key, value))
        plan.append((key, holder, name, value))
    if not plan:
        raise MifOpError("nothing to set - pass at least one of %s. NOTHING was changed." % ", ".join(keys))
    before = _arm_read(obj)
    for _key, holder, name, value in plan:
        setattr(holder, name, value)
    bpy.context.view_layer.update()
    after = _arm_read(obj)
    off = [k for k, _h, _n, v in plan if after[k] != v]
    if off:
        raise MifOpError("written and read back different: %s." % ", ".join("%s=%r" % (k, after[k]) for k in off))
    out = {"object": obj.name, "before": before, "after": after,
           "changed": sorted(k for k, _h, _n, _v in plan if before[k] != after[k])}
    if any(k in ("xMirrorPose", "autoIK") for k, _h, _n, _v in plan):
        out["interactiveNote"] = ("xMirrorPose and autoIK steer interactive transforms in the viewport only - a typed "
                                  "write (set_bone_pose, copy_pose) is not mirrored or solved by them; use set_bone_pose "
                                  "mirror or copy_pose flip for that.")
    if after.get("poseMode") == "REST":
        out["restNote"] = ("poseMode REST shows and renders the rest pose; the pose and its keys are kept, and poseMode "
                           "POSE brings them back.")
    return out


# --------------------------------------------------------------------------------------------- key a pose

_KEYPOSE_KEYS = {"object", "name", "frame", "bones", "namePrefix", "all", "channels", "customProperties",
                 "interpolation", "reset"}
_CHANNELS = ("location", "rotation", "scale")


def _animatable_custom(pb):
    """Custom property keys of a pose bone that hold a number (int, float or bool) - what a key can store."""
    out = []
    for k in pb.keys():
        if k.startswith("_") or k in ("cycles",):
            continue
        try:
            v = pb[k]
        except (KeyError, TypeError):
            continue
        if isinstance(v, (int, float)) and not isinstance(v, str):
            out.append(k)
    return sorted(out)


def op_key_pose(params):
    """Key a set of pose bones at a frame as they stand - see the MCP wrapper for the parameters."""
    reject_unknown(params, _KEYPOSE_KEYS, "key_pose")
    obj = _armature(params, "keyed")
    require_editable(obj, "key")
    frame = take_float(params, "frame", required=True)
    bones = pick_bones(obj, params, "keyed")
    raw_ch = params.get("channels", list(_CHANNELS))
    if not isinstance(raw_ch, list) or not raw_ch or any(c not in _CHANNELS for c in raw_ch):
        raise MifOpError("channels is a non-empty list of %s, got %r. NOTHING was keyed." % (", ".join(_CHANNELS), raw_ch))
    channels = [c for c in _CHANNELS if c in raw_ch]
    custom = params.get("customProperties", False)
    if not isinstance(custom, (bool, list)) or (isinstance(custom, list)
                                                and any(not isinstance(c, str) or not c for c in custom)):
        raise MifOpError("customProperties is true (every numeric custom property of each bone) or a list of names, "
                         "got %r. NOTHING was keyed." % (custom,))
    interp = take(params, "interpolation", default=None, kind=str)
    if interp is not None:
        ids = _interp_ids()
        if interp.upper() not in ids:
            raise MifOpError("interpolation must be one of %s, got %r. NOTHING was keyed." % (", ".join(ids), interp))
        interp = interp.upper()
    reset = take_bool(params, "reset", default=False)
    plan = []
    missing_props = []
    for pb in bones:
        props = []
        if custom is True:
            props = _animatable_custom(pb)
        elif isinstance(custom, list):
            have = set(_animatable_custom(pb))
            props = [c for c in custom if c in have]
            missing_props.extend("%s[%s]" % (pb.name, c) for c in custom if c not in have)
        plan.append((pb, props))
    if isinstance(custom, list) and missing_props and len(missing_props) == len(custom) * len(bones):
        raise MifOpError("no bone named here holds a numeric custom property %s. NOTHING was keyed." % custom)

    if reset:
        for pb, _props in plan:
            pb.location = (0.0, 0.0, 0.0)
            pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            pb.rotation_euler = (0.0, 0.0, 0.0)
            pb.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
            pb.scale = (1.0, 1.0, 1.0)
    rows, total = [], 0
    for pb, props in plan:
        paths = []
        if "location" in channels:
            paths.append("location")
        if "rotation" in channels:
            paths.append(rotation_path(pb))
        if "scale" in channels:
            paths.append("scale")
        paths.extend('["%s"]' % p for p in props)
        n = key_bone_channels(pb, frame, paths, interp)
        total += n
        rows.append({"bone": pb.name, "rotationMode": pb.rotation_mode, "keys": n,
                     "customProperties": props or None})
    # A CONSTRAINED BONE'S KEYS HOLD ITS OWN CHANNELS, not where the constraint puts it: the evaluated pose is taken back
    # into the bone's local space and compared with what was keyed.
    bpy.context.view_layer.update()
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    driven = []
    for pb, _props in plan:
        if not any(not c.mute and c.influence > 0.0 for c in pb.constraints):
            continue
        epb = ev.pose.bones[pb.name]
        local = ev.convert_space(pose_bone=epb, matrix=epb.matrix, from_space="POSE", to_space="LOCAL")
        d = (local.to_translation() - pb.matrix_basis.to_translation()).length
        q1 = local.to_3x3().normalized().to_quaternion()
        q2 = pb.matrix_basis.to_3x3().normalized().to_quaternion()
        ang = math.degrees(q1.rotation_difference(q2).angle)
        if d > 1e-5 or ang > 1e-3:
            driven.append({"bone": pb.name, "constraints": [c.type for c in pb.constraints if not c.mute],
                           "differsUnits": round(d, 6), "differsDegrees": round(min(ang, 360 - ang), 4)})
    out = {"object": obj.name, "frame": frame, "bonesKeyed": len(rows), "keysWritten": total,
           "channels": channels, "reset": reset, "interpolation": interp, "bones": rows[:200]}
    if missing_props:
        out["customPropertiesMissing"] = missing_props[:50]
    if driven:
        out["constraintDriven"] = driven[:50]
        out["constraintNote"] = ("these bones are moved by a constraint: the keys hold their own channels, and the "
                                 "evaluated pose differs by the amounts given. Bake with bake_to_keyframes to key "
                                 "where they really are.")
    eulers = [r["bone"] for r in rows if r["rotationMode"] not in ("QUATERNION", "AXIS_ANGLE")]
    if "rotation" in channels and eulers and len(eulers) < len(rows):
        out["rotationModeNote"] = ("rotation is keyed on each bone's own mode: rotation_euler on %d bone(s), "
                                   "rotation_quaternion or rotation_axis_angle on the rest." % len(eulers))
    return out


# --------------------------------------------------------------------------------------------- copy a pose

_COPYPOSE_KEYS = {"object", "name", "fromFrame", "toFrame", "bones", "namePrefix", "all", "flip", "nameRule", "key",
                  "interpolation", "allowSimulation"}


def op_copy_pose(params):
    """Copy the pose at one frame to another, optionally flipped onto the other side - see the MCP wrapper."""
    reject_unknown(params, _COPYPOSE_KEYS, "copy_pose")
    obj = _armature(params)
    require_editable(obj, "pose")
    src_frame = take_float(params, "fromFrame", required=True)
    dst_frame = take_float(params, "toFrame", required=True)
    flip = take_bool(params, "flip", default=False)
    key = take_bool(params, "key", default=True)
    rule = None
    if flip:
        if params.get("nameRule") is None:
            raise MifOpError("flip needs nameRule - %s. The side of a name is never assumed. NOTHING was changed."
                             % _NAME_RULE_HELP)
        rule = parse_name_rule(params.get("nameRule"))
    elif params.get("nameRule") is not None:
        raise MifOpError("nameRule pairs the sides for flip:true. NOTHING was changed.")
    interp = take(params, "interpolation", default=None, kind=str)
    if interp is not None:
        if not key:
            raise MifOpError("interpolation is for the keys key:true writes. NOTHING was changed.")
        ids = _interp_ids()
        if interp.upper() not in ids:
            raise MifOpError("interpolation must be one of %s, got %r. NOTHING was changed." % (", ".join(ids), interp))
        interp = interp.upper()
    if src_frame == dst_frame and not flip:
        raise MifOpError("fromFrame and toFrame are both %g and flip is off - that copies the pose onto itself. NOTHING "
                         "was changed." % src_frame)
    bones = pick_bones(obj, params, "changed", default_all=True)
    pairs, no_partner = [], []
    for pb in bones:
        if not flip:
            pairs.append((pb.name, pb.name))
            continue
        other = partner_name(pb.name, rule)
        if other is None:
            pairs.append((pb.name, pb.name))           # no side: flipped in place
        elif obj.pose.bones.get(other) is None:
            no_partner.append(pb.name)
        else:
            pairs.append((pb.name, other))
    if not pairs:
        raise MifOpError("no bone named has a partner on '%s' under that nameRule (%s). NOTHING was changed."
                         % (obj.name, ", ".join(no_partner[:10])))
    dsts = [d for _s, d in pairs]
    if len(set(dsts)) != len(dsts):
        dup = sorted({d for d in dsts if dsts.count(d) > 1})
        raise MifOpError("two bones would be written onto the same bone (%s) under that nameRule. NOTHING was changed."
                         % ", ".join(dup[:10]))

    sc = bpy.context.scene
    frame_before = sc.frame_current
    # BOTH FRAMES ARE VISITED, and a frame change steps every live simulation (ops_common.simulation_guard).
    sim_reply = simulation_guard(params, [math.ceil(src_frame), math.ceil(dst_frame)], "copy_pose")
    sc.frame_set(int(math.floor(src_frame)), subframe=src_frame - math.floor(src_frame))
    source = {s: read_channels(obj.pose.bones[s]) for s, _d in pairs}
    source_mats = evaluated_pose(obj)
    constrained = sorted({s for s, _d in pairs
                          if any(not c.mute and c.influence > 0.0 for c in obj.pose.bones[s].constraints)})
    sc.frame_set(int(math.floor(dst_frame)), subframe=dst_frame - math.floor(dst_frame))
    converted = []
    for s, d in pairs:
        ch = flip_channels(source[s]) if flip else source[s]
        if write_channels(obj.pose.bones[d], ch):
            converted.append(d)
    keys = 0
    if key:
        for _s, d in pairs:
            pb = obj.pose.bones[d]
            keys += key_bone_channels(pb, dst_frame, ["location", rotation_path(pb), "scale"], interp)
    check = mirror_report(obj, pairs, source_mats, flip=flip)
    out = {"object": obj.name, "fromFrame": src_frame, "toFrame": dst_frame, "flip": flip,
           "bonesWritten": len(pairs), "written": [{"from": s, "to": d} for s, d in pairs][:200],
           "withoutPartner": no_partner[:100], "keyed": key, "keysWritten": keys,
           "frameBefore": frame_before, "frameNow": sc.frame_current, "check": check}
    out.update(sim_reply)
    if converted:
        out["rotationModeConverted"] = converted[:50]
    if constrained:
        out["constrainedBones"] = constrained[:50]
        out["constrainedNote"] = ("these bones carry a live constraint: their own channels were copied (Blender's Copy "
                                  "Pose rule) - copying the evaluated pose into the channels would apply the "
                                  "constraint twice - so check.worst shows the constraint's part where it differs.")
    if not key:
        out["keyNote"] = ("key:false leaves the pasted pose unkeyed - a channel the action animates takes the "
                          "action's value back at the next frame change.")
    if flip and (check["maxErrorUnits"] > 1e-3 or check["maxErrorDegrees"] > 0.5):
        out["mirrorNote"] = ("the written bones do not evaluate to the mirror of their sources (worst %.4g units, %.4g "
                             "degrees): the two sides of this rig are not mirror images in rest, or a constraint moves "
                             "them - see check.worst." % (check["maxErrorUnits"], check["maxErrorDegrees"]))
    return out


def mirror_bone_pose(obj, pb, spec):
    """set_bone_pose mirror: write pb's partner with pb's channels flipped, then measure. `spec` is validated by
    check_mirror_spec first. Returns the reply block."""
    rule, verify = spec
    other = partner_name(pb.name, rule)
    partner = obj.pose.bones.get(other) if other else None
    write_channels(partner, flip_channels(read_channels(pb)))
    out = {"partner": partner.name, "partnerWritten": {
        "location": rnd(list(partner.location)), "rotationMode": partner.rotation_mode,
        "rotation": rnd(list(getattr(partner, rotation_path(partner)))), "scale": rnd(list(partner.scale))}}
    if verify:
        now = evaluated_pose(obj)
        d, a = matrix_error(obj, mirrored(now[pb.name]), now[partner.name])
        out["mirrorErrorUnits"] = round(d, 6)
        out["mirrorErrorDegrees"] = round(a, 4)
        if d > 1e-3 or a > 0.5:
            out["mirrorNote"] = ("'%s' does not evaluate to the mirror of '%s' (%.4g units, %.4g degrees): the rig's two "
                                 "sides are not mirror images in rest, or a constraint moves one." % (
                                     partner.name, pb.name, d, a))
    return out


def check_mirror_spec(obj, pb, raw, mode_after):
    """Validate set_bone_pose's mirror block before anything is written: (rule, verify)."""
    if not isinstance(raw, dict) or set(raw) - {"pair", "axis", "verify"} or "pair" not in raw:
        raise MifOpError("mirror is {pair: 'FLIP_NAME' | [a, b] | {name: partner}, axis 'X', verify true}; pair is "
                         "required - the side of a name is never assumed. Got %r. NOTHING was changed." % (raw,))
    if str(raw.get("axis", "X")).upper() != "X":
        raise MifOpError("mirror.axis is X - an armature's left and right. NOTHING was changed.")
    rule = parse_name_rule(raw["pair"], "mirror.pair")
    verify = raw.get("verify", True)
    if not isinstance(verify, bool):
        raise MifOpError("mirror.verify is true or false. NOTHING was changed.")
    other = partner_name(pb.name, rule)
    if other is None:
        raise MifOpError("'%s' has no side under that pair, so it has no partner to mirror onto. NOTHING was changed."
                         % pb.name)
    partner = obj.pose.bones.get(other)
    if partner is None:
        raise MifOpError("'%s' would mirror onto '%s', and '%s' has no such bone. NOTHING was changed."
                         % (pb.name, other, obj.name))
    if partner.rotation_mode != mode_after:
        raise MifOpError("'%s' rotates by %s and its partner '%s' by %s - a mirrored write needs one mode on both; set "
                         "the partner's rotationMode first. NOTHING was changed."
                         % (pb.name, mode_after, partner.name, partner.rotation_mode))
    return rule, verify


# --------------------------------------------------------------------------------------------- pose assets
#
# MEASURED 2026-10-04 on 3.6.23, 4.4.0 and 5.2.1 (scratchpad probes p2_ops, p3_assets):
#   * poselib.apply_pose_asset cannot run headless on any of them: 3.6 and 4.4 "poll() failed, context is incorrect",
#     5.2 "Error: No asset in context" - it reads the asset from an asset browser's context. So apply is NATIVE here.
#   * poselib.create_pose_asset runs headless with a selection, but differently per build: a Python operator on 3.6
#     (pose_name, activate_new_action True by default - it puts the new action on the rig), native on 4.4 and 5.2 where
#     it fails without asset_library_reference ("Unexpected library type. Failed to create pose asset") and needs
#     PoseBone.select on 5.2. Create is native too, so one call behaves one way on every build: a new action is put on the
#     rig, each bone's channels keyed at frame 1 with keyframe_insert (5.x makes the slot), and the rig's own action and
#     slot put back - the same curves the operator writes (location, rotation in the bone's mode, scale, in its group).
#   * ID.asset_mark / asset_clear / asset_generate_preview run headless on all three (an action's preview is 128 x 128);
#     a preview's pixels can be set from an image; bpy.data.libraries.write puts an action in a .blend that
#     libraries.load(assets_only=True) lists again. asset_mark sets the fake user and asset_clear takes it off.
#   * A catalog is a line "UUID:path:simple name" in blender_assets.cats.txt at the library's root (beside the .blend
#     for the current file); AssetMetaData.catalog_id takes the UUID.

_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_CATS = "blender_assets.cats.txt"


def _library(raw, verb):
    """'CURRENT_FILE' -> (None, None); an asset library's NAME from Preferences -> (name, its directory)."""
    if raw is None or (isinstance(raw, str) and raw.upper() == "CURRENT_FILE"):
        return None, None
    libs = {lib.name: lib.path for lib in bpy.context.preferences.filepaths.asset_libraries}
    if not isinstance(raw, str) or raw not in libs:
        raise MifOpError("library is CURRENT_FILE or the name of an asset library in Preferences > File Paths: %s; got "
                         "%r. NOTHING was %s." % (", ".join(sorted(libs)) or "(none)", raw, verb))
    path = bpy.path.abspath(libs[raw])
    if not os.path.isdir(path):
        raise MifOpError("asset library '%s' points at %s, which is not a folder. NOTHING was %s." % (raw, path, verb))
    return raw, path


def _read_catalogs(folder):
    """{path: uuid} from a library's catalog file (missing file -> {})."""
    out = {}
    p = os.path.join(folder, _CATS)
    if not os.path.isfile(p):
        return out
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("VERSION"):
                continue
            parts = line.split(":")
            if len(parts) >= 2 and _UUID.match(parts[0]):
                out[parts[1]] = parts[0]
    return out


def _plan_catalog(raw, folder, verb):
    """A catalog, validated before any write: (uuid, path or None, folder to append to or None)."""
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip("/ "):
        raise MifOpError("catalog is a catalog path like 'Poses/Hands' or a catalog UUID, got %r. NOTHING was %s."
                         % (raw, verb))
    if _UUID.match(raw):
        known = {v: k for k, v in _read_catalogs(folder).items()} if folder else {}
        return (raw.lower(), known.get(raw.lower()), None)
    path = "/".join(p for p in raw.strip().split("/") if p)
    if folder is None:
        raise MifOpError("a catalog PATH lives in %s beside the saved .blend, and this file has not been saved - save it "
                         "first or pass the catalog's UUID. NOTHING was %s." % (_CATS, verb))
    known = _read_catalogs(folder)
    if path in known:
        return (known[path], path, None)
    return (str(uuid.uuid4()), path, folder)


def _write_catalog(plan):
    """Append a new catalog line (making the file with its VERSION line when absent). Returns True when one was made."""
    cid, path, folder = plan
    if folder is None:
        return False
    p = os.path.join(folder, _CATS)
    fresh = not os.path.isfile(p)
    with open(p, "a", encoding="utf-8") as fh:
        if fresh:
            fh.write("# This is an Asset Catalog Definition file for Blender.\n#\n# Empty lines and lines starting with "
                     "`#` will be ignored.\n# The first non-ignored line should be the version indicator.\n# Other lines "
                     "are of the format \"UUID:catalog/path/for/assets:simple catalog name\"\n\nVERSION 1\n\n")
        fh.write("%s:%s:%s\n" % (cid, path, path.replace("/", "-")))
    return True


def _current_file_folder():
    return os.path.dirname(bpy.data.filepath) if bpy.data.filepath else None


def _check_thumbnail(raw, verb):
    if raw is None:
        return "GENERATE"
    if isinstance(raw, str) and raw.upper() in ("GENERATE", "NONE"):
        return raw.upper()
    if not isinstance(raw, str) or not os.path.isfile(bpy.path.abspath(raw)):
        raise MifOpError("thumbnail is GENERATE (Blender's own preview, the default), NONE or the path of an image "
                         "file that exists; got %r. NOTHING was %s." % (raw, verb))
    return bpy.path.abspath(raw)


def _apply_thumbnail(idb, how):
    """Give a datablock its preview; returns [w, h] or None."""
    if how == "NONE":
        return None
    if how == "GENERATE":
        idb.asset_generate_preview()
    else:
        img = bpy.data.images.load(how, check_existing=False)
        try:
            w, h = img.size
            if not w or not h:
                raise MifOpError("the thumbnail image %s has no pixels." % how)
            prev = idb.preview_ensure()
            prev.image_size = (w, h)
            prev.image_pixels_float = list(img.pixels)
        finally:
            bpy.data.images.remove(img)
    p = getattr(idb, "preview", None)
    return list(p.image_size) if p is not None else None


# WHAT A POSE ASSET HOLDS, as Blender's own Create Pose Asset stores it: location, rotation in the bone's mode, scale, the
# ten bendy-bone pose properties, and any other property of those bones the rig's action keys on the current frame
# (3.6 scripts/addons/pose_library/pose_creation.py:60-70 _bbone_props, :102-109 _store_bone_pose_parameters, :111-144
# _store_animated_parameters; 4.4 and 5.x's native operator writes the same paths - settled 2026-10-18 on 3.6-5.2,
# test_blender_settle_routes BS608). The bendy-bone channels were missing here, so applying such a pose left a bone's
# curve-in / ease / roll as the rig had it.
_BBONE_POSE_PROPS = ("bbone_curveinx", "bbone_curveoutx", "bbone_curveinz", "bbone_curveoutz", "bbone_rollin",
                     "bbone_rollout", "bbone_scalein", "bbone_scaleout", "bbone_easein", "bbone_easeout")


def _key_into_new_action(obj, act, bones):
    """Key each bone's channels at frame 1 into `act`, through the rig's own animation data, then put the rig's action
    (and 5.x slot) back. Returns the number of curves made."""
    ad = obj.animation_data_create()
    old, old_slot = ad.action, getattr(ad, "action_slot", None)
    names = {pb.name for pb in bones}
    stored = {"location", "rotation_quaternion", "rotation_euler", "rotation_axis_angle", "scale"} | set(_BBONE_POSE_PROPS)
    frame = bpy.context.scene.frame_current
    animated = []
    if old is not None:
        from .ops_anim import _fcurves   # noqa: PLC0415
        for fc in _fcurves(obj):
            bone, prop = _bone_of(fc.data_path)
            if bone in names and prop not in stored and any(abs(k.co[0] - frame) < 1e-4 for k in fc.keyframe_points):
                animated.append((fc.data_path, fc.array_index, bone))
    try:
        ad.action = act
        for pb in bones:
            for path in ("location", rotation_path(pb), "scale") + _BBONE_POSE_PROPS:
                if hasattr(pb, path):
                    pb.keyframe_insert(path, frame=1.0, group=pb.name)
        for path, index, bone in animated:
            try:
                obj.keyframe_insert(path, index=index, frame=1.0, group=bone)
            except (TypeError, RuntimeError):
                pass   # a once-animated property that no longer resolves - the operator skips it too (:138-140)
    finally:
        ad.action = old
        if old is not None and old_slot is not None:
            try:
                ad.action_slot = old_slot
            except (TypeError, ValueError, AttributeError, RuntimeError):
                pass
    from .ops_anim import _action_curves   # noqa: PLC0415
    return len(_action_curves(act))


_CREATE_ASSET_KEYS = {"object", "name", "asset", "bones", "namePrefix", "all", "library", "catalog", "description",
                      "thumbnail"}


def op_create_pose_asset(params):
    """Store an armature's current pose as a pose asset - see the MCP wrapper and tool help."""
    reject_unknown(params, _CREATE_ASSET_KEYS, "create_pose_asset")
    obj = _armature(params, "made")
    asset = take(params, "asset", required=True, kind=str).strip()
    if not asset or len(asset) > 63:
        raise MifOpError("asset is the pose's name, 1-63 characters (Blender truncates longer). NOTHING was made.")
    bones = pick_bones(obj, params, "made", default_all=True)
    lib_name, lib_path = _library(params.get("library"), "made")
    desc = take(params, "description", default=None, kind=str)
    thumb = _check_thumbnail(params.get("thumbnail"), "made")
    if lib_name is None:
        if bpy.data.actions.get(asset) is not None:
            raise MifOpError("an action named '%s' already exists in this file - pick another name. NOTHING was made."
                             % asset)
        cat = _plan_catalog(params.get("catalog"), _current_file_folder(), "made")
        target = None
    else:
        target = os.path.join(lib_path, bpy.path.clean_name(asset) + ".blend")
        if os.path.exists(target):
            raise MifOpError("%s already exists in library '%s' - it is not overwritten. NOTHING was made."
                             % (target, lib_name))
        cat = _plan_catalog(params.get("catalog"), lib_path, "made")
    act = bpy.data.actions.new(name=asset)
    try:
        curves = _key_into_new_action(obj, act, bones)
        act.asset_mark()
        act.use_fake_user = True
        if desc:
            act.asset_data.description = desc
        made_cat = False
        if cat is not None:
            made_cat = _write_catalog(cat)
            act.asset_data.catalog_id = cat[0]
        preview = _apply_thumbnail(act, thumb)
    except BaseException:
        bpy.data.actions.remove(act)
        raise
    out = {"object": obj.name, "asset": act.name, "library": lib_name or "CURRENT_FILE", "bones": [pb.name for pb in
                                                                                                   bones][:200],
           "bonesHeld": len(bones), "curves": curves, "frame": 1.0, "isAsset": act.asset_data is not None,
           "catalogId": cat[0] if cat else None, "catalogPath": cat[1] if cat else None, "catalogCreated": made_cat,
           "description": desc, "preview": preview,
           "rigActionKept": obj.animation_data.action.name if obj.animation_data and obj.animation_data.action else None}
    if target is not None:
        bpy.data.libraries.write(target, {act}, fake_user=True)
        bpy.data.actions.remove(act)
        with bpy.data.libraries.load(target, assets_only=True) as (src, _dst):
            listed = list(src.actions)
        if asset not in listed:
            raise MifOpError("wrote %s and it does not list the pose asset '%s' - do not rely on it." % (target, asset))
        out["file"] = target
        out["libraryNote"] = ("the pose is in the library's file %s, not in this one - apply_pose_asset with library "
                              "'%s' reads it from there." % (target, lib_name))
    return out


def _bone_of(data_path):
    head = 'pose.bones["'
    if not data_path.startswith(head):
        return None, None
    end = data_path.find('"]', len(head))
    return data_path[len(head):end], data_path[end + 2:].lstrip(".")


def _find_in_library(lib_path, asset, verb):
    """The .blend in the library folder whose assets include the action `asset` - read only, nothing appended."""
    for root, _dirs, files in os.walk(lib_path):
        for f in sorted(files):
            if not f.endswith(".blend"):
                continue
            p = os.path.join(root, f)
            try:
                with bpy.data.libraries.load(p, assets_only=True) as (src, _dst):
                    found = asset in src.actions
            except (OSError, RuntimeError):
                continue
            if found:
                return p
    raise MifOpError("no pose asset '%s' in any .blend of %s. NOTHING was %s." % (asset, lib_path, verb))


def _append_action(path, asset):
    """Append the action `asset` from `path`; returns the local copy."""
    with bpy.data.libraries.load(path, assets_only=True) as (_src, dst):
        dst.actions = [asset]
    return dst.actions[0]


_APPLY_ASSET_KEYS = {"object", "name", "asset", "library", "blend", "flipped", "nameRule", "bones"}


def op_apply_pose_asset(params):
    """Apply a pose asset to an armature at a blend, optionally flipped - see the MCP wrapper and tool help."""
    reject_unknown(params, _APPLY_ASSET_KEYS, "apply_pose_asset")
    obj = _armature(params)
    # mathutils AFTER the first refusals: the offline refusal sweep (test_blender_refusals B110/B111) has no real one.
    from mathutils import Euler, Quaternion, Vector  # noqa: PLC0415
    require_editable(obj, "pose")
    asset = take(params, "asset", required=True, kind=str)
    blend = take_float(params, "blend", default=1.0)
    if not 0.0 <= blend <= 1.0:
        raise MifOpError("blend is 0 (nothing) to 1 (the full pose), got %g. NOTHING was changed." % blend)
    flipped = take_bool(params, "flipped", default=False)
    rule = None
    if flipped:
        if params.get("nameRule") is None:
            raise MifOpError("flipped needs nameRule ('FLIP_NAME' - Blender's own flipped apply uses flip_name - a suffix "
                             "pair or a table); the side of a name is never assumed. NOTHING was changed.")
        rule = parse_name_rule(params.get("nameRule"))
    elif params.get("nameRule") is not None:
        raise MifOpError("nameRule pairs the sides for flipped:true. NOTHING was changed.")
    only = params.get("bones")
    if only is not None and (not isinstance(only, list) or not only or any(not isinstance(b, str) for b in only)):
        raise MifOpError("bones is a non-empty list of the rig's bone names to write. NOTHING was changed.")
    if only is not None:
        absent = [b for b in only if obj.pose.bones.get(b) is None]
        if absent:
            raise MifOpError("'%s' has no bone(s) %s. NOTHING was changed." % (obj.name, ", ".join(absent[:20])))
    lib_name, lib_path = _library(params.get("library"), "changed")
    appended = None
    if lib_name is None:
        act = bpy.data.actions.get(asset)
        if act is None:
            raise MifOpError("no action named '%s' in this file - pass library to read one from an asset library. "
                             "NOTHING was changed." % asset)
    else:
        source = _find_in_library(lib_path, asset, "changed")
        act = appended = _append_action(source, asset)
    try:
        from .ops_anim import _action_curves   # noqa: PLC0415
        frame = act.frame_range[0]
        per_bone, not_applied = {}, []
        for fc in _action_curves(act):
            bone, prop = _bone_of(fc.data_path)
            if bone is None or prop not in ("location", "rotation_quaternion", "rotation_euler", "rotation_axis_angle",
                                            "scale"):
                not_applied.append("%s[%d]" % (fc.data_path, fc.array_index))
                continue
            per_bone.setdefault(bone, {}).setdefault(prop, {})[fc.array_index] = fc.evaluate(frame)
        if not per_bone:
            if appended is not None:
                bpy.data.actions.remove(appended)   # the appended copy goes before the refusal: NOTHING stays true
                appended = None
            raise MifOpError("'%s' holds no bone channel - it is not a pose. NOTHING was changed." % asset)
        plan, missing = [], []
        for bone, props in per_bone.items():
            dst = partner_name(bone, rule) if flipped else None
            dst = dst or bone
            if obj.pose.bones.get(dst) is None:
                missing.append(dst)
                continue
            if only is not None and dst not in only:
                continue
            plan.append((bone, dst, props))
        if not plan:
            if appended is not None:
                bpy.data.actions.remove(appended)
                appended = None
            raise MifOpError("no bone of '%s' matches the asset's bones%s (asset bones: %s). NOTHING was changed."
                             % (obj.name, " under that nameRule" if flipped else "", ", ".join(sorted(per_bone)[:10])))
        before = {d: obj.pose.bones[d].matrix_basis.copy() for _b, d, _p in plan}
        for _src, dst, props in plan:
            pb = obj.pose.bones[dst]
            mode = pb.rotation_mode
            loc = list(pb.location)
            scl = list(pb.scale)
            for i, v in props.get("location", {}).items():
                loc[i] = -v if (flipped and i == 0) else v
            for i, v in props.get("scale", {}).items():
                scl[i] = v
            rot_q = None
            for path in ("rotation_quaternion", "rotation_axis_angle", "rotation_euler"):
                if path not in props:
                    continue
                vals = props[path]
                if path == "rotation_quaternion":
                    full = [vals.get(i, d) for i, d in enumerate((1.0, 0.0, 0.0, 0.0))]
                    if flipped:
                        full = [full[0], full[1], -full[2], -full[3]]
                    rot_q = Quaternion(full).normalized()
                elif path == "rotation_axis_angle":
                    full = [vals.get(i, d) for i, d in enumerate((0.0, 0.0, 1.0, 0.0))]
                    if flipped:
                        full = [full[0], full[1], -full[2], -full[3]]
                    axis = Vector(full[1:])
                    rot_q = Quaternion(axis.normalized() if axis.length else Vector((0, 1, 0)), full[0])
                else:
                    full = [vals.get(i, 0.0) for i in range(3)]
                    if flipped:
                        full = [full[0], -full[1], -full[2]]
                    order = mode if mode not in ("QUATERNION", "AXIS_ANGLE") else "XYZ"
                    rot_q = Euler(full, order).to_quaternion()
                break
            cur_loc, cur_scl = list(pb.location), list(pb.scale)
            pb.location = [c + (t - c) * blend for c, t in zip(cur_loc, loc)]
            pb.scale = [c + (t - c) * blend for c, t in zip(cur_scl, scl)]
            if rot_q is not None:
                cur_q = pb.matrix_basis.to_3x3().normalized().to_quaternion()
                if cur_q.dot(rot_q) < 0:
                    rot_q.negate()
                q = cur_q.slerp(rot_q, blend)
                if mode == "QUATERNION":
                    if Quaternion(pb.rotation_quaternion).dot(q) < 0:
                        q.negate()
                    pb.rotation_quaternion = q
                elif mode == "AXIS_ANGLE":
                    axis, ang = q.to_axis_angle()
                    pb.rotation_axis_angle = (ang, axis[0], axis[1], axis[2])
                else:
                    pb.rotation_euler = q.to_euler(mode, pb.rotation_euler)
        bpy.context.view_layer.update()
        rows = []
        for _src, dst, _p in plan:
            a, b = before[dst], obj.pose.bones[dst].matrix_basis
            d = (b.to_translation() - a.to_translation()).length
            ang = math.degrees(a.to_3x3().normalized().to_quaternion().rotation_difference(
                b.to_3x3().normalized().to_quaternion()).angle)
            rows.append((dst, round(d, 6), round(min(ang, 360 - ang), 4)))
        moved = [r for r in rows if r[1] > 1e-6 or r[2] > 1e-4]
        out = {"object": obj.name, "asset": act.name if appended is None else asset, "library": lib_name or "CURRENT_FILE",
               "isAsset": act.asset_data is not None, "blend": blend, "flipped": flipped, "frame": frame,
               "bonesWritten": [d for _s, d, _p in plan][:200], "bonesMoved": [r[0] for r in moved][:200],
               "largestChangeDegrees": max((r[2] for r in rows), default=0.0),
               "largestChangeUnits": max((r[1] for r in rows), default=0.0)}
        if flipped:
            out["writtenFrom"] = [{"from": s, "to": d} for s, d, _p in plan][:200]
        if missing:
            out["bonesMissing"] = sorted(set(missing))[:100]
        if not_applied:
            out["notApplied"] = not_applied[:50]
        if not out["isAsset"]:
            out["assetNote"] = "'%s' is an ordinary action, not marked as an asset - applied all the same." % act.name
        out["keyNote"] = "the pose is written, not keyed - key_pose keys it."
        # A CHANNEL THE RIG'S ACTION ANIMATES TAKES THE WRITE BACK at the next frame change (ops_anim.animated_channels).
        from .ops_anim import animated_channels   # noqa: PLC0415
        paths = {pb.path_from_id(p) for _s, d, _pr in plan for pb in [obj.pose.bones[d]]
                 for p in ("location", rotation_path(pb), "scale")}
        animated = animated_channels(obj, paths)
        if animated:
            out["animatedChannels"] = sorted(animated)[:50]
            out["animatedNote"] = ("%d channel(s) written here are animated by the rig's action, which puts its own value "
                                   "back at the next frame change - key them (key_pose) to keep the pose."
                                   % len(animated))
        return out
    finally:
        if appended is not None:
            bpy.data.actions.remove(appended)


# The datablock kinds the asset browser lists (3.6 to 5.2), and where each lives in bpy.data.
_ASSET_KINDS = {"ACTION": "actions", "COLLECTION": "collections", "MATERIAL": "materials", "NODE_TREE": "node_groups",
                "OBJECT": "objects", "WORLD": "worlds"}
_MARK_KEYS = {"datablock", "kind", "catalog", "description", "tags", "clear", "preview"}


def op_mark_asset(params):
    """Mark (or clear) a datablock as an asset of the current file - see the MCP wrapper and tool help."""
    reject_unknown(params, _MARK_KEYS, "mark_asset")
    name = take(params, "datablock", required=True, kind=str)
    kind = str(take(params, "kind", required=True, kind=str)).upper()
    if kind not in _ASSET_KINDS:
        raise MifOpError("kind is one of %s (what the asset browser lists), got %r. NOTHING was changed."
                         % (", ".join(sorted(_ASSET_KINDS)), params.get("kind")))
    idb = getattr(bpy.data, _ASSET_KINDS[kind]).get(name)
    if idb is None:
        raise MifOpError("no %s named '%s' in this file. NOTHING was changed." % (kind.lower(), name))
    if getattr(idb, "library", None) is not None:
        raise MifOpError("'%s' is linked from %s - mark it in that file. NOTHING was changed."
                         % (name, idb.library.filepath))
    clear = take_bool(params, "clear", default=False)
    desc = take(params, "description", default=None, kind=str)
    tags = params.get("tags")
    if tags is not None and (not isinstance(tags, list) or any(not isinstance(t, str) or not t for t in tags)):
        raise MifOpError("tags is a list of words. NOTHING was changed.")
    preview = take_bool(params, "preview", default=False)
    if clear and (desc is not None or tags is not None or params.get("catalog") is not None or preview):
        raise MifOpError("clear:true takes the asset mark off; description, tags, catalog and preview describe a mark. "
                         "NOTHING was changed.")
    if clear:
        if idb.asset_data is None:
            raise MifOpError("'%s' is not an asset, so there is nothing to clear. NOTHING was changed." % name)
        idb.asset_clear()
        out = {"datablock": idb.name, "kind": kind, "isAsset": idb.asset_data is not None,
               "fakeUser": bool(idb.use_fake_user), "users": idb.users}
        if not idb.use_fake_user and idb.users == 0:
            out["warning"] = ("asset_clear took the fake user off and nothing else uses '%s' - the next save drops it."
                              % idb.name)
        return out
    cat = _plan_catalog(params.get("catalog"), _current_file_folder(), "changed")
    was = idb.asset_data is not None
    if not was:
        idb.asset_mark()
    if desc is not None:
        idb.asset_data.description = desc
    if tags:
        have = {t.name for t in idb.asset_data.tags}
        for t in tags:
            if t not in have:
                idb.asset_data.tags.new(t)
    made_cat = False
    if cat is not None:
        made_cat = _write_catalog(cat)
        idb.asset_data.catalog_id = cat[0]
    size = _apply_thumbnail(idb, "GENERATE") if preview else None
    ad = idb.asset_data
    return {"datablock": idb.name, "kind": kind, "isAsset": ad is not None, "wasAsset": was,
            "fakeUser": bool(idb.use_fake_user), "catalogId": ad.catalog_id if ad.catalog_id.strip("0-") else None,
            "catalogPath": cat[1] if cat else None, "catalogCreated": made_cat,
            "description": ad.description or None, "tags": sorted(t.name for t in ad.tags), "preview": size,
            "note": "an asset of the CURRENT file shows in the asset browser's Current File library once saved; for "
                    "another file to use it, save this one inside an asset library's folder."}


OPS = {
    "set_armature_properties": op_set_armature_properties,
    "key_pose": op_key_pose,
    "copy_pose": op_copy_pose,
    "create_pose_asset": op_create_pose_asset,
    "apply_pose_asset": op_apply_pose_asset,
    "mark_asset": op_mark_asset,
}
