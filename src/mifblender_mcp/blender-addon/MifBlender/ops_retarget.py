"""MifBlender ops: motion moved between rigs and onto a root - retarget_action and extract_root_motion (1.3 gap audit,
2026-10-19).

THE GAP. Mocap and pack clips reach a character only through per-bone Copy Rotation constraints baked by hand, and the
rest-pose difference - an A-pose clip on a T-pose rig - which is why hand retargets look broken, was handled by no op.
A clip whose travel is on the hips needs it on a root bone for root motion in an engine, or none for in-place locomotion;
Unreal's extract_root_motion fixes a clip already in the engine, and a rig with no root bone cannot be fixed there at all.

THE MATH, both ops: everything is done with armature-space rest and pose matrices, so no constraint, depsgraph pass or
operator is needed per frame. A bone's pose in armature space is P(child) = P(parent) @ R(parent)^-1 @ R(child) @ B(child),
R its rest (bone.matrix_local) and B its basis (matrix_basis); so the basis that puts a bone at a wanted armature-space
matrix D is B = (P(parent) @ R(parent)^-1 @ R(child))^-1 @ D. Bones are written parents first.
"""
from __future__ import annotations

import math
import re

import bpy

from .ops_common import MifOpError, get_object, remove_objects, reject_unknown, require_editable, take, take_bool, take_float, take_int


def _armature(name, tail, what="armature"):
    obj = get_object(name)
    if obj.type != "ARMATURE":
        raise MifOpError("'%s' is a %s, not an armature (%s). %s" % (obj.name, obj.type, what, tail))
    if obj.mode != "OBJECT":
        raise MifOpError("'%s' is in %s mode - leave it first. %s" % (obj.name, obj.mode, tail))
    return obj


def _ordered(arm):
    return sorted(arm.data.bones, key=lambda b: len(b.parent_recursive))


def _assign(obj, act):
    ad = obj.animation_data or obj.animation_data_create()
    ad.action = act
    if act is not None and "action_slot" in ad.bl_rna.properties and ad.action_slot is None:
        slot = next((s for s in act.slots if s.target_id_type == "OBJECT"), None)
        if slot is not None:
            ad.action_slot = slot


def _write_rotation(pb, quat, frame, group):
    if pb.rotation_mode == "QUATERNION":
        pb.rotation_quaternion = quat
        pb.keyframe_insert("rotation_quaternion", frame=frame, group=group)
    elif pb.rotation_mode == "AXIS_ANGLE":
        axis, ang = quat.to_axis_angle()
        pb.rotation_axis_angle = (ang, axis[0], axis[1], axis[2])
        pb.keyframe_insert("rotation_axis_angle", frame=frame, group=group)
    else:
        pb.rotation_euler = quat.to_euler(pb.rotation_mode, pb.rotation_euler)
        pb.keyframe_insert("rotation_euler", frame=frame, group=group)


_PREFIXES = re.compile(r"^(mixamorig\d*[:_]|def[-_]|org[-_]|mch[-_]|bip\d*[ _]?|b_|bn_|cc_base_|j_)", re.I)
_SIDES = [(re.compile(r"(^|[._ -])(l|left)($|[._ -])", re.I), "L"), (re.compile(r"(^|[._ -])(r|right)($|[._ -])", re.I), "R"),
          (re.compile(r"^left", re.I), "L"), (re.compile(r"^right", re.I), "R"),
          (re.compile(r"left$", re.I), "L"), (re.compile(r"right$", re.I), "R")]
_ALIASES = {"pelvis": "hips", "hip": "hips", "spine0": "spine", "spine1": "spine", "spine01": "spine",
            "upperarm": "arm", "lowerarm": "forearm", "upleg": "thigh", "leg": "calf", "shin": "calf", "thigh": "thigh",
            "calf": "calf", "lowerleg": "calf", "upperleg": "thigh", "clavicle": "shoulder", "toebase": "toe",
            "ball": "toe", "toes": "toe", "neck01": "neck", "head": "head", "hand": "hand", "foot": "foot"}


def _norm_name(n):
    s = _PREFIXES.sub("", n)
    side = ""
    for rx, tag in _SIDES:
        if rx.search(s):
            side = tag
            s = rx.sub(" ", s)
            break
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    s = _ALIASES.get(s, s)
    return side + ":" + s


def _auto_map(src, tgt):
    by = {}
    for b in src.data.bones:
        by.setdefault(_norm_name(b.name), b.name)
    out = {}
    for b in tgt.data.bones:
        k = _norm_name(b.name)
        if k in by:
            out[b.name] = by[k]
    return out


# --------------------------------------------------------------------------------------------- retarget_action

_RT_KEYS = ("source", "target", "action", "map", "frameRange", "newAction", "rootBone", "sourceRoot", "rootScale",
             "assign", "sampleFrames", "mode", "fbx", "fbxOptions", "keepSource", "translation", "translationTree",
             "skeleton", "reference", "referenceSpace")
_RT_UNREAL_ONLY = ("fbx", "fbxOptions", "keepSource", "translation", "translationTree", "skeleton", "reference",
                   "referenceSpace")
_RT_WORLD_ONLY = ("rootBone", "sourceRoot", "rootScale")


def op_retarget_action(params):
    """Move an action from a source armature onto a target with other bone names and another rest pose, baked to a new
    action on the target - with a per-bone error report.

    params:
      source, target (str)     the two armatures (object mode)
      action (str)             the source's action (default its assigned one)
      map (dict | "AUTO")      {targetBone: sourceBone}; "AUTO" (default) matches by name - prefixes such as mixamorig:,
                               DEF-, ORG- dropped, sides (.L, _l, Left) kept apart, common synonyms (pelvis = hips,
                               upperarm = arm, shin = calf). Unmapped bones of both are listed.
      frameRange ([s, e])      default the action's frame range
      newAction (str)          default "<action>_<target>"
      rootBone (str)           a target bone that also takes the source's TRANSLATION (default: the mapped bone nearest
                               the top of the target's hierarchy, usually the hips); sourceRoot its source bone (default
                               its mapping)
      rootScale (float)        the translation scale (default: target root rest height / source root rest height)
      assign (bool)            leave the new action on the target (default true; false puts the old one back)
      sampleFrames (int)       frames the error is measured at (default 8)
      mode (str)               WORLD (default, below) | UNREAL (1.4): the clip played the way Unreal plays it on a mesh of
                               the same skeleton - see _retarget_unreal
    REST-POSE COMPENSATION: each mapped bone takes the source bone's WORLD rotation change from ITS rest, applied to the
    target bone's own rest - so an A-pose clip plays on a T-pose rig. Unmapped target bones keep their rest. Reply per
    mapped bone: the largest world-rotation difference between what the target shows and what the source asked, in
    degrees, over the sampled frames.
    """
    reject_unknown(params, _RT_KEYS, "retarget_action")
    tail = "NOTHING was changed."
    mode = str(params.get("mode") or "WORLD").upper()
    if mode not in ("WORLD", "UNREAL"):
        raise MifOpError("mode is WORLD (default: rest-pose difference compensated) or UNREAL (the clip's local "
                         "rotations played as Unreal plays them), got %r. %s" % (params.get("mode"), tail))
    if mode == "UNREAL":
        return _retarget_unreal(params, tail)
    stray = [k for k in _RT_UNREAL_ONLY if params.get(k) is not None]
    if stray:
        raise MifOpError("%s belong%s to mode UNREAL. %s" % (", ".join(stray), "s" if len(stray) == 1 else "", tail))
    from mathutils import Matrix, Quaternion           # noqa: PLC0415,F401
    src = _armature(take(params, "source", required=True, kind=str), tail, "source")
    tgt = _armature(take(params, "target", required=True, kind=str), tail, "target")
    if src is tgt:
        raise MifOpError("source and target are one armature. %s" % tail)
    require_editable(tgt, "animate")
    act_name = take(params, "action", default=None, kind=str)
    if act_name is not None:
        act = bpy.data.actions.get(act_name)
        if act is None:
            raise MifOpError("no action named '%s'. %s" % (act_name, tail))
    else:
        act = src.animation_data.action if src.animation_data else None
        if act is None:
            raise MifOpError("'%s' has no action assigned - name one with action. %s" % (src.name, tail))
    raw_map = params.get("map", "AUTO")
    if isinstance(raw_map, str) and raw_map.upper() == "AUTO":
        bone_map = _auto_map(src, tgt)
        how = "AUTO"
    elif isinstance(raw_map, dict) and raw_map:
        bad = [(t, s) for t, s in raw_map.items() if tgt.data.bones.get(t) is None or not isinstance(s, str)
               or src.data.bones.get(s) is None]
        if bad:
            raise MifOpError("map pairs name bones the armatures do not have (target: source): %s. %s" % (bad[:10], tail))
        bone_map = dict(raw_map)
        how = "explicit"
    else:
        raise MifOpError("map is {targetBone: sourceBone} or 'AUTO'. %s" % tail)
    if not bone_map:
        raise MifOpError("no bone of '%s' matched one of '%s' by name - pass map {targetBone: sourceBone}. %s"
                         % (tgt.name, src.name, tail))
    fr = params.get("frameRange")
    if fr is None:
        start, end = (int(round(v)) for v in act.frame_range)
    else:
        if not isinstance(fr, list) or len(fr) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in fr) \
                or fr[1] < fr[0]:
            raise MifOpError("frameRange is [start, end] whole frames, got %r. %s" % (fr, tail))
        start, end = fr
    order = [b for b in _ordered(tgt)]
    mapped = [b.name for b in order if b.name in bone_map]
    root_t = take(params, "rootBone", default=None, kind=str) or mapped[0]
    if tgt.data.bones.get(root_t) is None:
        raise MifOpError("target has no bone '%s'. %s" % (root_t, tail))
    root_s = take(params, "sourceRoot", default=None, kind=str) or bone_map.get(root_t)
    if root_s is None or src.data.bones.get(root_s) is None:
        raise MifOpError("rootBone '%s' needs sourceRoot (it is not mapped). %s" % (root_t, tail))
    scale = take_float(params, "rootScale", default=None)
    sw, tw = src.matrix_world, tgt.matrix_world
    s_rest_h = (sw @ src.data.bones[root_s].head_local)
    t_rest_h = (tw @ tgt.data.bones[root_t].head_local)
    if scale is None:
        scale = (t_rest_h.z / s_rest_h.z) if abs(s_rest_h.z) > 1e-6 and t_rest_h.z > 1e-6 else 1.0
    new_name = take(params, "newAction", default="%s_%s" % (act.name, tgt.name), kind=str)
    samples = max(2, take_int(params, "sampleFrames", default=8))
    scene = bpy.context.scene
    s_ad, t_ad = src.animation_data, tgt.animation_data
    saved = (s_ad.action if s_ad else None, t_ad.action if t_ad else None, scene.frame_current,
             {pb.name: pb.matrix_basis.copy() for pb in tgt.pose.bones})
    srest = {b.name: (sw @ b.matrix_local).to_quaternion() for b in src.data.bones}
    trest_arm = {b.name: b.matrix_local.copy() for b in tgt.data.bones}
    tw_q = tw.to_quaternion()
    assign = take_bool(params, "assign", default=True)      # read BEFORE the new action exists
    new = bpy.data.actions.new(new_name)
    new.use_fake_user = False
    wanted = {}
    try:
        _assign(src, act)
        _assign(tgt, new)
        for pb in tgt.pose.bones:
            pb.matrix_basis = Matrix.Identity(4)
        prev_q = {}
        for f in range(start, end + 1):
            scene.frame_set(f)
            pose = {}
            want_f = {}
            for b in order:
                pb = tgt.pose.bones[b.name]
                rest_rel = (trest_arm[b.parent.name].inverted_safe() @ trest_arm[b.name]) if b.parent else trest_arm[b.name]
                parent_pose = pose[b.parent.name] if b.parent else Matrix.Identity(4)
                base = parent_pose @ rest_rel
                if b.name in bone_map:
                    sname = bone_map[b.name]
                    s_world = (sw @ src.pose.bones[sname].matrix).to_quaternion()
                    delta = s_world @ srest[sname].inverted()
                    t_world_rest = (tw @ trest_arm[b.name]).to_quaternion()
                    t_world = delta @ t_world_rest
                    want_f[b.name] = t_world
                    q_arm = tw_q.inverted() @ t_world
                    basis_q = base.to_quaternion().inverted() @ q_arm
                    if b.name in prev_q:
                        basis_q.make_compatible(prev_q[b.name])
                    prev_q[b.name] = basis_q.copy()
                    loc = pb.location.copy() * 0
                    if b.name == root_t:
                        s_head = sw @ src.pose.bones[root_s].head
                        d_world = (s_head - s_rest_h) * scale
                        d_arm = tw.to_3x3().inverted_safe() @ d_world
                        loc = base.to_3x3().inverted_safe() @ d_arm
                        pb.location = loc
                        pb.keyframe_insert("location", frame=f, group=b.name)
                    _write_rotation(pb, basis_q, f, b.name)
                    basis = Matrix.Translation(loc) @ basis_q.to_matrix().to_4x4()
                else:
                    basis = Matrix.Identity(4)
                pose[b.name] = base @ basis
            wanted[f] = want_f
        # THE MEASUREMENT: the target evaluated from its new keys, against the world rotation asked of each bone
        steps = sorted({int(round(start + (end - start) * i / (samples - 1))) for i in range(samples)})
        err = {n: 0.0 for n in mapped}
        for f in steps:
            scene.frame_set(f)
            for n in mapped:
                got = (tw @ tgt.pose.bones[n].matrix).to_quaternion()
                d = got.rotation_difference(wanted[f][n]).angle
                err[n] = max(err[n], math.degrees(min(d, 2 * math.pi - d)))
    except Exception:
        _assign(tgt, saved[1])
        bpy.data.actions.remove(new)
        for n, m in saved[3].items():
            tgt.pose.bones[n].matrix_basis = m
        raise
    finally:
        _assign(src, saved[0])
        scene.frame_set(saved[2])
    if not assign:
        _assign(tgt, saved[1])
        new.use_fake_user = True
    rows = [{"target": n, "source": bone_map[n], "maxErrorDeg": round(err[n], 4)} for n in mapped]
    worst = max((r["maxErrorDeg"] for r in rows), default=0.0)
    out = {"source": src.name, "target": tgt.name, "action": act.name, "newAction": new.name, "map": how,
           "frameStart": start, "frameEnd": end, "bonesMapped": len(rows), "rootBone": root_t, "sourceRoot": root_s,
           "rootScale": round(scale, 6), "bones": rows, "maxErrorDeg": round(worst, 4),
           "unmappedTarget": sorted(b.name for b in tgt.data.bones if b.name not in bone_map)[:200],
           "unmappedSource": sorted(set(b.name for b in src.data.bones) - set(bone_map.values()))[:200],
           "assigned": assign, "framesMeasured": steps}
    if worst > 1.0:
        out["warning"] = ("a bone's result differs from the source's rotation by up to %.2f degrees - a constraint or "
                          "driver on the target, or a bone that does not inherit rotation" % worst)
    return out


# --------------------------------------------------------------------------------------------- retarget_action UNREAL
# FOUND 2026-11-19: Unreal plays a clip's ABSOLUTE local (parent-relative) bone rotations on every mesh of
# the skeleton, with each bone's translation chosen by its retargeting mode - Animation (the clip's), Skeleton (the
# mesh's own) or AnimationScaled (the clip's, times the mesh's reference length over the SKELETON's). Blender keeps an
# imported FBX clip relative to the clip file's own bind pose, so the same clip attached to a differently posed armature
# plays rest + offset: .blend files built that way were 43-75 m off Unreal's poses. Baking each frame's local rotations
# with the target's translations brought every bone to 2-4 cm of Unreal's, every clip, every mesh.
#
# THE MATH: a bone's local transform is L = P(parent)^-1 @ P(bone) in armature space (the armature's own matrix for a
# root). The target bone's basis that plays a wanted local L is B = Rrel^-1 @ L, Rrel = R(parent)^-1 @ R(bone) its rest
# relative to its parent - so no depsgraph pass is needed per target bone. Both armatures must hold their bones in the
# same frames (two FBX imports of one skeleton with the same bone-axis options): a local rotation means nothing across
# different bone frames, which is why fbxOptions is the target's own import options.

_TR_MODES = {"ANIMATION": "ANIMATION", "SKELETON": "SKELETON", "ANIMATIONSCALED": "ANIMATION_SCALED"}


def _tr_mode(raw, key, tail):
    m = _TR_MODES.get(re.sub(r"[^A-Z]", "", str(raw).upper())) if isinstance(raw, str) else None
    if m is None:
        raise MifOpError("%s: a translation mode is Animation, Skeleton or AnimationScaled, got %r. %s" % (key, raw, tail))
    return m


def _import_fbx_source(path, options, tail):
    """Import a clip FBX; (armature, new objects, new actions). Everything imported is removed again on a refusal."""
    import os                                          # noqa: PLC0415
    if not isinstance(path, str) or not os.path.isfile(path):
        raise MifOpError("fbx: no file at %r. %s" % (path, tail))
    opts = dict(options or {})
    if not isinstance(options or {}, dict):
        raise MifOpError("fbxOptions is a dict of the FBX importer's options. %s" % tail)
    known = {p.identifier for p in bpy.ops.import_scene.fbx.get_rna_type().properties} - {"rna_type", "filepath"}
    bad = sorted(k for k in opts if k not in known)
    if bad:
        raise MifOpError("fbxOptions %s %s not an option of this Blender's FBX importer (it takes: %s). %s"
                         % (", ".join(bad), "is" if len(bad) == 1 else "are", ", ".join(sorted(known)), tail))
    objs0, acts0 = set(bpy.data.objects), set(bpy.data.actions)
    try:
        bpy.ops.import_scene.fbx(filepath=os.path.abspath(path), **opts)
    except Exception as exc:                           # noqa: BLE001 - the importer's own message
        _remove_imported([o for o in bpy.data.objects if o not in objs0], [a for a in bpy.data.actions if a not in acts0])
        raise MifOpError("the FBX importer refused %s: %s. %s" % (path, str(exc).strip()[:300], tail))
    new_objs = [o for o in bpy.data.objects if o not in objs0]
    new_acts = [a for a in bpy.data.actions if a not in acts0]
    arms = [o for o in new_objs if o.type == "ARMATURE"]
    if len(arms) != 1:
        _remove_imported(new_objs, new_acts)
        raise MifOpError("fbx %s held %d armatures - a clip file holds one. %s" % (path, len(arms), tail))
    return arms[0], new_objs, new_acts


def _remove_imported(objs, acts):
    datas = [o.data for o in objs if o.data is not None]
    remove_objects(objs)
    homes = {bpy.types.Mesh: bpy.data.meshes, bpy.types.Armature: bpy.data.armatures, bpy.types.Camera: bpy.data.cameras,
             bpy.types.Light: bpy.data.lights}
    for d in datas:
        coll = next((c for t, c in homes.items() if isinstance(d, t)), None)
        if coll is not None and d.users == 0:
            coll.remove(d)
    for a in acts:
        if a.users == 0 or (a.use_fake_user and a.users == 1):
            bpy.data.actions.remove(a)


def _local(arm_pose, b):
    """A pose bone's local (parent-relative) matrix from armature-space pose matrices."""
    m = arm_pose[b.name]
    return arm_pose[b.parent.name].inverted_safe() @ m if b.parent else m.copy()


def _parse_reference(raw, space, tgt, start, end, tail):
    if raw is None:
        return None
    frames = raw.get("frames") if isinstance(raw, dict) else None
    if not isinstance(frames, dict) or not frames:
        raise MifOpError("reference is {frames: {frame: {bone: [x, y, z]}}} - the bone heads of a pose set to compare "
                         "against. %s" % tail)
    out = {}
    for fk, bones in frames.items():
        try:
            f = int(fk)
        except (TypeError, ValueError):
            raise MifOpError("reference frame %r is not a whole frame. %s" % (fk, tail))
        if f < start or f > end:
            raise MifOpError("reference frame %d is outside the baked range %d-%d. %s" % (f, start, end, tail))
        if not isinstance(bones, dict) or not bones:
            raise MifOpError("reference frame %d names no bones. %s" % (f, tail))
        row = {}
        for bn, p in bones.items():
            if tgt.data.bones.get(bn) is None:
                raise MifOpError("reference frame %d: the target has no bone '%s'. %s" % (f, bn, tail))
            if not isinstance(p, (list, tuple)) or len(p) != 3 or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in p):
                raise MifOpError("reference frame %d bone '%s' is [x, y, z], got %r. %s" % (f, bn, p, tail))
            row[bn] = tuple(float(v) for v in p)
        out[f] = row
    if space not in ("ARMATURE", "WORLD"):
        raise MifOpError("referenceSpace is ARMATURE (default: the target armature's own space) or WORLD. %s" % tail)
    return out


def _retarget_unreal(params, tail):
    from mathutils import Matrix, Vector               # noqa: PLC0415
    tgt = _armature(take(params, "target", required=True, kind=str), tail, "target")
    require_editable(tgt, "animate")
    src_name = take(params, "source", default=None, kind=str)
    fbx = params.get("fbx")
    if (src_name is None) == (fbx is None):
        raise MifOpError("mode UNREAL takes the clip as source (an armature in the scene with the action) OR fbx (a clip "
                         "file to import) - exactly one. %s" % tail)
    stray = [k for k in _RT_WORLD_ONLY if params.get(k) is not None]
    if stray:
        raise MifOpError("%s belong%s to mode WORLD: in UNREAL each bone's translation comes from translation / "
                         "translationTree. %s" % (", ".join(stray), "s" if len(stray) == 1 else "", tail))
    if fbx is None and (params.get("fbxOptions") is not None or params.get("keepSource") is not None):
        raise MifOpError("fbxOptions and keepSource go with fbx. %s" % tail)
    # EVERYTHING ABOUT THE TARGET IS CHECKED BEFORE AN IMPORT, so a refusal leaves the scene as it was.
    bones_t = tgt.data.bones
    trans = {}
    for key in ("translation", "translationTree"):
        raw = params.get(key)
        if raw is None:
            trans[key] = {}
            continue
        if not isinstance(raw, dict):
            raise MifOpError("%s is {bone: Animation | Skeleton | AnimationScaled}. %s" % (key, tail))
        missing = sorted(b for b in raw if bones_t.get(b) is None)
        if missing:
            raise MifOpError("%s names bones the target does not have: %s. %s" % (key, ", ".join(missing[:10]), tail))
        trans[key] = {b: _tr_mode(m, "%s.%s" % (key, b), tail) for b, m in raw.items()}
    mode_of = {}
    for b in bones_t:
        m = trans["translation"].get(b.name)
        if m is None:
            m = next((trans["translationTree"][a.name] for a in [b] + list(b.parent_recursive)
                      if a.name in trans["translationTree"]), "ANIMATION")
        mode_of[b.name] = m
    rest_t = {b.name: b.matrix_local.copy() for b in bones_t}
    rel_t = {b.name: (rest_t[b.parent.name].inverted_safe() @ rest_t[b.name]) if b.parent else rest_t[b.name]
             for b in bones_t}
    scaled = sorted(n for n, m in mode_of.items() if m == "ANIMATION_SCALED")
    ratio = {}
    if scaled:
        skel_name = take(params, "skeleton", default=None, kind=str)
        if skel_name is None:
            raise MifOpError("AnimationScaled on %s needs skeleton: the armature whose rest pose is the Unreal Skeleton's "
                             "reference (the mesh the skeleton was made from) - Unreal scales by the mesh's length over "
                             "the SKELETON's, not the clip file's (ledger U776). %s" % (", ".join(scaled[:5]), tail))
        skel = _armature(skel_name, tail, "skeleton")
        for n in scaled:
            sb = skel.data.bones.get(n)
            if sb is None:
                raise MifOpError("skeleton '%s' has no bone '%s'. %s" % (skel.name, n, tail))
            srel = (sb.parent.matrix_local.inverted_safe() @ sb.matrix_local) if sb.parent else sb.matrix_local
            ls = srel.translation.length
            if ls < 1e-9:
                raise MifOpError("'%s' sits on its parent in skeleton '%s' (length 0), so AnimationScaled has nothing to "
                                 "scale by. %s" % (n, skel.name, tail))
            ratio[n] = rel_t[n].translation.length / ls
    elif params.get("skeleton") is not None:
        raise MifOpError("skeleton is only read for AnimationScaled bones, and none is set. %s" % tail)
    ref_space = str(params.get("referenceSpace") or "ARMATURE").upper()
    samples = max(2, take_int(params, "sampleFrames", default=8))
    assign = take_bool(params, "assign", default=True)
    keep_source = take_bool(params, "keepSource", default=False)
    imported = ([], [])
    if fbx is not None:
        src, objs, acts = _import_fbx_source(fbx, params.get("fbxOptions"), tail)
        imported = (objs, acts)
    else:
        src = _armature(src_name, tail, "source")
        if src is tgt:
            raise MifOpError("source and target are one armature. %s" % tail)
    try:
        act_name = take(params, "action", default=None, kind=str)
        if act_name is not None:
            act = bpy.data.actions.get(act_name)
            if act is None:
                raise MifOpError("no action named '%s'%s. %s" % (
                    act_name, " (the file brought %s)" % [a.name for a in imported[1]] if imported[1] else "", tail))
        else:
            act = src.animation_data.action if src.animation_data else None
            if act is None:
                raise MifOpError("'%s' has no action assigned - name one with action. %s" % (src.name, tail))
        raw_map = params.get("map", "SAME")
        if isinstance(raw_map, str) and raw_map.upper() == "SAME":
            bone_map = {b.name: b.name for b in bones_t if src.data.bones.get(b.name) is not None}
            how = "SAME"
        elif isinstance(raw_map, str) and raw_map.upper() == "AUTO":
            bone_map, how = _auto_map(src, tgt), "AUTO"
        elif isinstance(raw_map, dict) and raw_map:
            bad = [(t, s) for t, s in raw_map.items() if bones_t.get(t) is None or not isinstance(s, str)
                   or src.data.bones.get(s) is None]
            if bad:
                raise MifOpError("map pairs name bones the armatures do not have (target: source): %s. %s"
                                 % (bad[:10], tail))
            bone_map, how = dict(raw_map), "explicit"
        else:
            raise MifOpError("map is 'SAME' (default in UNREAL: one skeleton, one set of names), 'AUTO' or "
                             "{targetBone: sourceBone}. %s" % tail)
        if not bone_map:
            raise MifOpError("no bone of '%s' has a namesake in '%s' - they are not one skeleton; pass map. %s"
                             % (tgt.name, src.name, tail))
        fr = params.get("frameRange")
        if fr is None:
            start, end = (int(round(v)) for v in act.frame_range)
        else:
            if not isinstance(fr, list) or len(fr) != 2 or any(isinstance(v, bool) or not isinstance(v, int)
                                                                for v in fr) or fr[1] < fr[0]:
                raise MifOpError("frameRange is [start, end] whole frames, got %r. %s" % (fr, tail))
            start, end = fr
        reference = _parse_reference(params.get("reference"), ref_space, tgt, start, end, tail)
        new_name = take(params, "newAction", default="%s_%s" % (act.name, tgt.name), kind=str)
        return _bake_unreal(src, tgt, act, bone_map, how, mode_of, ratio, rel_t, start, end, new_name, samples,
                            assign, reference, ref_space, Matrix, Vector, fbx, keep_source, imported)
    finally:
        if fbx is not None and not keep_source:
            live_objs, live_acts = set(bpy.data.objects.keys()), set(bpy.data.actions.keys())
            _remove_imported([o for o in imported[0] if o.name in live_objs],
                             [a for a in imported[1] if a.name in live_acts])


def _bake_unreal(src, tgt, act, bone_map, how, mode_of, ratio, rel_t, start, end, new_name, samples, assign,
                 reference, ref_space, Matrix, Vector, fbx, keep_source, imported):
    scene = bpy.context.scene
    s_ad, t_ad = src.animation_data, tgt.animation_data
    saved = (s_ad.action if s_ad else None, t_ad.action if t_ad else None, scene.frame_current,
             {pb.name: pb.matrix_basis.copy() for pb in tgt.pose.bones})
    order = [b for b in _ordered(tgt) if b.name in bone_map]
    new = bpy.data.actions.new(new_name)
    wanted = {}
    prev_q = {}
    try:
        _assign(src, act)
        _assign(tgt, new)
        for pb in tgt.pose.bones:
            pb.matrix_basis = Matrix.Identity(4)
        for f in range(start, end + 1):
            scene.frame_set(f)
            spose = {pb.name: pb.matrix.copy() for pb in src.pose.bones}
            want_f = {}
            for b in order:
                sb = src.data.bones[bone_map[b.name]]
                loc, rot, sca = _local(spose, sb).decompose()
                m = mode_of[b.name]
                t = rel_t[b.name].translation if m == "SKELETON" else (
                    loc * ratio[b.name] if m == "ANIMATION_SCALED" else loc)
                want = Matrix.Translation(t) @ rot.to_matrix().to_4x4() @ Matrix.Diagonal(sca.to_4d())
                want_f[b.name] = (rot.copy(), Vector(t))
                bl, bq, bs = (rel_t[b.name].inverted_safe() @ want).decompose()
                if b.name in prev_q:
                    bq.make_compatible(prev_q[b.name])
                prev_q[b.name] = bq.copy()
                pb = tgt.pose.bones[b.name]
                pb.location = bl
                pb.keyframe_insert("location", frame=f, group=b.name)
                _write_rotation(pb, bq, f, b.name)
                pb.scale = bs
                pb.keyframe_insert("scale", frame=f, group=b.name)
            wanted[f] = want_f
        # THE MEASUREMENT, off the target evaluated from its new keys: each bone's local rotation and translation
        # against what the clip asked for that frame.
        steps = sorted({int(round(start + (end - start) * i / (samples - 1))) for i in range(samples)})
        err = {b.name: [0.0, 0.0] for b in order}
        for f in steps:
            scene.frame_set(f)
            tpose = {pb.name: pb.matrix.copy() for pb in tgt.pose.bones}
            for b in order:
                tl, tq, _ts = _local(tpose, b).decompose()
                wq, wt = wanted[f][b.name]
                d = tq.rotation_difference(wq).angle
                err[b.name][0] = max(err[b.name][0], math.degrees(min(d, 2 * math.pi - d)))
                err[b.name][1] = max(err[b.name][1], (tl - wt).length)
        ref_out = None
        if reference:
            per, worst = [], None
            mw = tgt.matrix_world
            for f in sorted(reference):
                scene.frame_set(f)
                top = None
                for bn, p in reference[f].items():
                    head = tgt.pose.bones[bn].head
                    got = (mw @ head) if ref_space == "WORLD" else head
                    dist = (got - Vector(p)).length
                    if top is None or dist > top[1]:
                        top = (bn, dist)
                    if worst is None or dist > worst["difference"]:
                        worst = {"frame": f, "bone": bn, "difference": round(dist, 6), "got": [round(v, 6) for v in got],
                                 "reference": list(p)}
                per.append({"frame": f, "bones": len(reference[f]), "maxDifference": round(top[1], 6),
                            "worstBone": top[0]})
            ref_out = {"space": ref_space, "frames": per, "maxDifference": worst["difference"], "worst": worst}
    except Exception:
        _assign(tgt, saved[1])
        bpy.data.actions.remove(new)
        for n, mtx in saved[3].items():
            tgt.pose.bones[n].matrix_basis = mtx
        raise
    finally:
        if src.name in bpy.data.objects:
            _assign(src, saved[0])
        scene.frame_set(saved[2])
    if not assign:
        _assign(tgt, saved[1])
        new.use_fake_user = True
    rows = [{"target": b.name, "source": bone_map[b.name], "translationMode": mode_of[b.name],
             "maxErrorDeg": round(err[b.name][0], 4), "maxTranslationError": round(err[b.name][1], 6)}
            for b in order]
    for r in rows:
        if r["translationMode"] == "ANIMATION_SCALED":
            r["scale"] = round(ratio[r["target"]], 6)
    worst = max((r["maxErrorDeg"] for r in rows), default=0.0)
    out = {"mode": "UNREAL", "source": src.name if fbx is None else None, "fbx": fbx, "target": tgt.name,
           "action": act.name if fbx is None or keep_source else None, "sourceAction": act.name,
           "newAction": new.name, "map": how, "frameStart": start, "frameEnd": end, "bonesBaked": len(rows),
           "bones": rows, "maxErrorDeg": round(worst, 4),
           "maxTranslationError": round(max((r["maxTranslationError"] for r in rows), default=0.0), 6),
           "translationModes": {m: sum(1 for r in rows if r["translationMode"] == m)
                                for m in ("ANIMATION", "SKELETON", "ANIMATION_SCALED")},
           "unmappedTarget": sorted(b.name for b in tgt.data.bones if b.name not in bone_map)[:200],
           "assigned": assign, "framesMeasured": steps, "reference": ref_out}
    if fbx is not None:
        out["sourceKept"] = keep_source
        if keep_source:
            out["sourceArmature"] = src.name
            out["imported"] = [o.name for o in imported[0]]
    if worst > 1.0:
        out["warning"] = ("a bone's local rotation differs from the clip's by up to %.2f degrees - a constraint or driver "
                          "on the target, or a bone that does not inherit rotation" % worst)
    return out


# --------------------------------------------------------------------------------------------- extract_root_motion

_RM_KEYS = ("armature", "action", "hips", "root", "mode", "axes", "yaw", "feet", "frameRange", "createRoot")


def _add_root(arm, name):
    """A root bone at the armature's origin, every parentless bone parented to it (not connected)."""
    vl = bpy.context.view_layer
    prev = vl.objects.active
    vl.objects.active = arm
    try:
        bpy.ops.object.mode_set(mode="EDIT")
        ebs = arm.data.edit_bones
        tops = [eb for eb in ebs if eb.parent is None]
        size = max((eb.length for eb in ebs), default=0.1)
        r = ebs.new(name)
        r.head = (0.0, 0.0, 0.0)
        r.tail = (0.0, size, 0.0)
        r.use_deform = False
        for eb in tops:
            eb.parent = r
            eb.use_connect = False
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
        vl.objects.active = prev


def op_extract_root_motion(params):
    """Move an armature clip's horizontal travel (and optionally its yaw) from the hips onto a root bone, keyed per frame -
    the root made when the rig has none - or remove it (IN_PLACE). The pair of Unreal's extract_root_motion, at the source.

    params:
      armature (str)        the armature (object mode)
      action (str)          default its assigned action; the action is edited in place
      hips (str)            the bone that carries the travel (default a bone named hips / pelvis)
      root (str)            the root bone (default "root"); createRoot (default true) adds it at the origin when missing,
                            parenting every parentless bone to it
      mode (str)            ROOT_MOTION (default: the travel moves to the root, the feet stay where they were) | IN_PLACE
                            (the travel is removed)
      axes (str)            "XY" (default: horizontal) | "XYZ"
      yaw (bool)            also move the hips' turn about Z onto the root (ROOT_MOTION only; default false)
      feet (list)           bones whose world positions are compared before and after (default foot / ankle / toe bones)
      frameRange ([s, e])   default the action's range
    Reply: frames keyed, the travel moved (start to end), and per foot the largest world position change - 0 for
    ROOT_MOTION (the motion is the same, only where it is carried changed), the travel for IN_PLACE.
    """
    reject_unknown(params, _RM_KEYS, "extract_root_motion")
    tail = "NOTHING was changed."
    from mathutils import Matrix, Vector               # noqa: PLC0415
    arm = _armature(take(params, "armature", required=True, kind=str), tail)
    require_editable(arm, "animate")
    act_name = take(params, "action", default=None, kind=str)
    act = bpy.data.actions.get(act_name) if act_name else (arm.animation_data.action if arm.animation_data else None)
    if act is None:
        raise MifOpError("no action %s - assign one or name it. %s" % (repr(act_name) if act_name else "on the armature",
                                                                     tail))
    bones = arm.data.bones
    hips = take(params, "hips", default=None, kind=str) or next(
        (b.name for b in bones if re.sub(r"[^a-z]", "", _PREFIXES.sub("", b.name).lower()) in ("hips", "pelvis", "hip")),
        None)
    if hips is None or bones.get(hips) is None:
        raise MifOpError("name the bone that carries the travel with hips (none is called hips or pelvis). %s" % tail)
    root = take(params, "root", default="root", kind=str)
    mode = str(take(params, "mode", default="ROOT_MOTION", kind=str)).upper()
    if mode not in ("ROOT_MOTION", "IN_PLACE"):
        raise MifOpError("mode is ROOT_MOTION or IN_PLACE, got %r. %s" % (params.get("mode"), tail))
    axes = str(take(params, "axes", default="XY", kind=str)).upper()
    if axes not in ("XY", "XYZ"):
        raise MifOpError("axes is XY or XYZ, got %r. %s" % (params.get("axes"), tail))
    yaw = take_bool(params, "yaw", default=False)
    if yaw and mode != "ROOT_MOTION":
        raise MifOpError("yaw moves the turn onto the root (ROOT_MOTION). %s" % tail)
    feet = params.get("feet")
    if feet is None:
        feet = [b.name for b in bones if re.search(r"foot|ankle|toe|ball", b.name, re.I)][:6]
    elif not isinstance(feet, list) or any(bones.get(f) is None for f in feet):
        raise MifOpError("feet is a list of bones of '%s'. %s" % (arm.name, tail))
    created = False
    if bones.get(root) is None:
        if not take_bool(params, "createRoot", default=True):
            raise MifOpError("'%s' has no bone '%s' and createRoot is false. %s" % (arm.name, root, tail))
        if mode == "IN_PLACE":
            raise MifOpError("IN_PLACE needs no root; '%s' has none - leave root out or name a bone it has. %s"
                             % (arm.name, tail))
    if bones.get(root) is not None and hips != root and bones.get(root) not in bones[hips].parent_recursive \
            and mode == "ROOT_MOTION":
        raise MifOpError("'%s' is not above '%s' in the hierarchy, so moving it would not move the hips. %s"
                         % (root, hips, tail))
    fr = params.get("frameRange")
    # CHECKED AS retarget_action CHECKS IT. This unpacked the caller's value as it came, so a string, a mapping or a
    # one-item list raised ValueError / TypeError out of the op (found 2026-10-28 by the version matrix's
    # corrupted-value pass) - after the guards above, before any write.
    if fr is None:
        start, end = (int(round(v)) for v in act.frame_range)
    else:
        if not isinstance(fr, list) or len(fr) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in fr) \
                or fr[1] < fr[0]:
            raise MifOpError("frameRange is [start, end] whole frames, got %r. %s" % (fr, tail))
        start, end = fr
    scene = bpy.context.scene
    saved_act = arm.animation_data.action if arm.animation_data else None
    saved_frame = scene.frame_current
    _assign(arm, act)
    mw = arm.matrix_world
    # 1. what the clip does now: the hips' armature-space pose and the feet's world positions, per frame
    before, feet_before = {}, {}
    for f in range(start, end + 1):
        scene.frame_set(f)
        before[f] = {b.name: arm.pose.bones[b.name].matrix.copy() for b in bones}
        feet_before[f] = {n: (mw @ arm.pose.bones[n].head) for n in feet}
    old_parent = {b.name: (b.parent.name if b.parent else None) for b in bones}
    if bones.get(root) is None:
        _add_root(arm, root)
        created = True
        bones = arm.data.bones
    rest = {b.name: b.matrix_local.copy() for b in arm.data.bones}
    h0 = before[start][hips].translation.copy()
    chain = list(reversed(bones[hips].parent_recursive))     # top ancestor first, down to the hips' parent
    rows_keyed = 0
    try:
        for f in range(start, end + 1):
            d_h = before[f][hips].copy()
            travel = d_h.translation - h0
            if axes == "XY":
                travel.z = 0.0
            root_m = Matrix.Translation(travel) if mode == "ROOT_MOTION" else Matrix.Identity(4)
            if yaw:
                fwd0 = before[start][hips].to_3x3() @ Vector((0, 1, 0))
                fwd = d_h.to_3x3() @ Vector((0, 1, 0))
                ang = math.atan2(fwd.y, fwd.x) - math.atan2(fwd0.y, fwd0.x)
                root_m = root_m @ Matrix.Rotation(ang, 4, "Z")
            want_h = d_h if mode == "ROOT_MOTION" else Matrix.Translation(-travel) @ d_h
            pose = {}
            if mode == "ROOT_MOTION":
                pb = arm.pose.bones[root]
                # the root's desired armature matrix is its rest moved by the travel (and yaw) about the origin
                d_root = root_m @ rest[root]
                rb = arm.data.bones[root]
                base = ((pose_parent_base(rest, rb, before[f])) if rb.parent else rest[root])
                basis = base.inverted_safe() @ d_root
                pb.matrix_basis = basis
                pb.keyframe_insert("location", frame=f, group=root)
                _write_rotation(pb, basis.to_quaternion(), f, root)
                pose[root] = d_root
            # every bone between the root and the hips keeps its own basis at this frame (read off the pose before the
            # change, against its parent then); its pose follows from its parent now
            for b in chain:
                if b.name in pose:
                    continue
                op = old_parent.get(b.name)
                old_base = ((before[f][op] @ rest[op].inverted_safe() @ rest[b.name]) if op else rest[b.name])
                basis_f = old_base.inverted_safe() @ before[f][b.name]
                parent_pose = pose.get(b.parent.name, before[f].get(b.parent.name)) if b.parent else Matrix.Identity(4)
                rest_rel = (rest[b.parent.name].inverted_safe() @ rest[b.name]) if b.parent else rest[b.name]
                pose[b.name] = parent_pose @ rest_rel @ basis_f
            hb = arm.data.bones[hips]
            parent_pose = ((pose[hb.parent.name] if hb.parent.name in pose else before[f][hb.parent.name])
                           if hb.parent else Matrix.Identity(4))
            rest_rel = (rest[hb.parent.name].inverted_safe() @ rest[hips]) if hb.parent else rest[hips]
            basis = (parent_pose @ rest_rel).inverted_safe() @ want_h
            hpb = arm.pose.bones[hips]
            hpb.matrix_basis = basis
            hpb.keyframe_insert("location", frame=f, group=hips)
            _write_rotation(hpb, basis.to_quaternion(), f, hips)
            rows_keyed += 1
        # 2. what it does after: the feet's world positions again
        moved = {n: 0.0 for n in feet}
        for f in range(start, end + 1):
            scene.frame_set(f)
            for n in feet:
                moved[n] = max(moved[n], ((mw @ arm.pose.bones[n].head) - feet_before[f][n]).length)
    finally:
        _assign(arm, saved_act if saved_act is not act else act)
        scene.frame_set(saved_frame)
    total = (before[end][hips].translation - h0)
    if axes == "XY":
        total.z = 0.0
    out = {"armature": arm.name, "action": act.name, "mode": mode, "hips": hips, "root": root if mode == "ROOT_MOTION"
           else None, "rootCreated": created, "axes": axes, "yaw": yaw, "framesKeyed": rows_keyed,
           "frameStart": start, "frameEnd": end, "travel": [round(v, 6) for v in mw.to_3x3() @ total],
           "feet": [{"bone": n, "maxMoved": round(moved[n], 6)} for n in feet]}
    if mode == "ROOT_MOTION" and any(v > 1e-3 for v in moved.values()):
        out["warning"] = "a foot moved in world space - the root carries the travel, so it should not have (%s)" % moved
    if created:
        out["note"] = ("a root bone was added at the origin and every top bone parented to it; a skinned mesh keeps its "
                       "weights (the root deforms nothing)")
    return out


def pose_parent_base(rest, bone, pose_now):
    """P(parent) @ R(parent)^-1 @ R(bone) for a bone whose parent keeps the pose it had (pose_now)."""
    return pose_now[bone.parent.name] @ rest[bone.parent.name].inverted_safe() @ rest[bone.name]


OPS = {
    "retarget_action": op_retarget_action,
    "extract_root_motion": op_extract_root_motion,
}
