"""MifBlender ops: editing motion that is already keyed - motion paths, a pose change spread over frames, smoothing and
thinning keys, matching one thing's world transform to another's, an action's slots, an existing NLA strip edited.

THE GAP (spec, "Motion paths for objects and bones", "Spread a pose change over a frame range with a falloff", "Smooth,
decimate and blend F-curve keys over a range", "Match an object's or a bone's world transform to another's at frames"):
keys could be written, listed, moved and eased, but nothing could show where a thing travels between them, carry an
edit across neighbouring frames, clean a dense curve, or snap one thing onto another over a range. And ("Action slots on
4.4 and later") nothing could give a second user of an action its own slot, split or merge actions by slot, or edit an
NLA strip once placed.

MOTION PATHS, MEASURED 2026-10-05 on 4.2.23, 4.4.0, 5.0.1 and 5.2.2 (headless): object.paths_calculate runs under a
context override; pose.paths_calculate needs the armature in POSE mode and reads the bone SELECTION, not a
selected_pose_bones override (with only the override it finishes and makes no path). A bone path keeps the bake
location it was made with: recalculating a HEADS path as TAILS leaves use_bone_head true and the points on the heads,
so a calculate here clears the chosen paths first. A path's points run one per frame from its frame_start to
frame_end - 1 (a manual range 1-20 gives 20 points, frame_start 1, frame_end 21); frame_step is display only.

POSE PROPAGATE, MEASURED 2026-10-05 on 4.2.23 and 5.2.2: pose.propagate copies the KEY at the current frame to the keys
its mode reaches (NEXT_KEY: the next one; BEFORE_FRAME end_frame 25: every key up to 25) - an unkeyed pose change at the
current frame propagates nothing (FINISHED, every key as it was). So spread_pose's mode route keys the bones' pose at
the frame first, and reads the pose BEFORE moving to that frame, since the frame change re-evaluates the action over
the unkeyed channels.
"""
from __future__ import annotations

import math

import bpy

from .ops_anim import _anim_holders, _bone_of_path, _fcurves
from .ops_common import (MifOpError, finite_float, finite_floats, finite_int, get_object, reject_unknown, require_editable, rnd,
                         simulation_guard, take, take_bool, take_float)
from .ops_pose import (_armature, key_bone_channels, pick_bones, restore_bone_selection, rotation_path,
                       select_bones)

_TAIL = "NOTHING was changed."


def _enum(params, key, allowed, default):
    v = params.get(key, default)
    if v is None:
        return None
    if not isinstance(v, str) or v.upper() not in allowed:
        raise MifOpError("%s must be %s, got %r. %s" % (key, " | ".join(allowed), v, _TAIL))
    return v.upper()


# --------------------------------------------------------------------------------------------- motion paths

_MP_KEYS = ("object", "name", "bones", "action", "type", "range", "frameStart", "frameEnd", "frameBefore", "frameAfter",
            "frameStep", "bakeLocation", "display", "maxPoints", "allowSimulation")
_MP_DISPLAY = {"lineThickness": "line_thickness", "lines": "lines", "color": "color", "colorPost": "color_post",
               "useCustomColor": "use_custom_color"}
def _select_attr():
    return "pose" if "select" in bpy.types.PoseBone.bl_rna.properties else "bone"


def _path_row(path, max_points, bone=None):
    pts = list(path.points)
    rows, gaps, prev = [], [], None
    for i, p in enumerate(pts):
        co = tuple(p.co)
        gap = None if prev is None else math.dist(co, prev)
        if gap is not None:
            gaps.append(gap)
        if i < max_points:
            rows.append({"frame": path.frame_start + i, "co": rnd(list(co)), "gap": None if gap is None else
                         round(gap, 6)})
        prev = co
    out = {"frameStart": path.frame_start, "frameEnd": path.frame_end - 1, "pointCount": len(pts), "points": rows,
           "pointsTruncated": len(pts) > max_points, "length": round(sum(gaps), 6),
           "maxGap": round(max(gaps), 6) if gaps else None, "minGap": round(min(gaps), 6) if gaps else None,
           "bakeLocation": "HEADS" if bone is not None and path.use_bone_head else ("TAILS" if bone else None),
           "lineThickness": path.line_thickness, "lines": bool(path.lines),
           "useCustomColor": bool(path.use_custom_color), "color": rnd(list(path.color))}
    if hasattr(path, "color_post"):
        out["colorPost"] = rnd(list(path.color_post))
    if bone is not None:
        out = dict({"bone": bone}, **out)
    return out


def _display(params):
    raw = params.get("display")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise MifOpError("display is an object {lineThickness, lines, color, colorPost, useCustomColor}, got %r. %s"
                         % (raw, _TAIL))
    unknown = sorted(set(raw) - set(_MP_DISPLAY))
    if unknown:
        raise MifOpError("display has no %s - it takes %s. %s" % (", ".join(unknown), ", ".join(sorted(_MP_DISPLAY)),
                                                                  _TAIL))
    out = {}
    for k, v in raw.items():
        attr = _MP_DISPLAY[k]
        if attr == "color_post" and "color_post" not in bpy.types.MotionPath.bl_rna.properties:
            raise MifOpError("display.colorPost (the path's colour after the current frame) is not on Blender %s - it "
                             "came in 4.2. %s" % (bpy.app.version_string, _TAIL))
        if attr == "line_thickness":
            n = finite_int(v, "display.lineThickness")
            if not 1 <= n <= 6:
                raise MifOpError("display.lineThickness is 1-6, got %r. %s" % (v, _TAIL))
            out[attr] = n
        elif attr in ("lines", "use_custom_color"):
            if not isinstance(v, bool):
                raise MifOpError("display.%s is true or false, got %r. %s" % (k, v, _TAIL))
            out[attr] = v
        else:
            if not isinstance(v, (list, tuple)) or len(v) != 3:
                raise MifOpError("display.%s is [r, g, b] (0-1), got %r. %s" % (k, v, _TAIL))
            c = finite_floats(list(v), "display." + k)
            if any(not 0 <= x <= 1 for x in c):
                raise MifOpError("display.%s components are 0-1, got %r. %s" % (k, v, _TAIL))
            out[attr] = c
    return out


# WHERE A RANGE ENDS DEPENDS ON THE BUILD. 4.2.17, 4.4.0, 5.0.1 and 5.2.2 make a manual range 1-20 into 20 points (the
# path's frame_end 21); 3.6.23 makes 19 (frame_end 20) - its range stops before frame_end, and a SCENE range of 5-15
# gave 10 points (measured 2026-10-10). Learned from the first path a build makes ({version: True when it ends early}),
# and on such a build the settings are widened by one frame for the calculation and put back after it, so a range is
# inclusive on every build as the reply says.
_MP_END_EXCLUSIVE = {}


def _mp_inclusive_end(avs, sc):
    """The last frame the path should hold for the visualisation's range, or None for a range this cannot read (the
    KEYS ranges, CURRENT_FRAME)."""
    if avs.type != "RANGE":
        return None
    if avs.range == "MANUAL":
        return avs.frame_end
    if avs.range == "SCENE":
        return sc.frame_end
    return None


def _mp_run(ns, avs, action, ptype, bones, sc, adjust):
    """paths_calculate / paths_update once; with adjust, through a range one frame longer (see _MP_END_EXCLUSIVE)."""
    saved = None
    if adjust and avs.type == "RANGE" and avs.range in ("MANUAL", "SCENE"):
        saved = (avs.range, avs.frame_start, avs.frame_end)
        lo, hi = (avs.frame_start, avs.frame_end) if avs.range == "MANUAL" else (sc.frame_start, sc.frame_end)
        avs.range = "MANUAL"
        avs.frame_end = hi + 1
        avs.frame_start = lo
    try:
        if action == "CALCULATE":
            kw = {"display_type": ptype, "range": avs.range}
            if bones:
                kw["bake_location"] = avs.bake_location
            r = ns.paths_calculate(**kw)
            if "FINISHED" not in r:
                raise MifOpError("Blender's paths_calculate returned %s. %s" % (sorted(r), _TAIL))
        else:
            ns.paths_update()
    finally:
        if saved is not None:
            avs.range, avs.frame_start, avs.frame_end = saved


def op_motion_paths(params):
    """Calculate, update or clear the motion path of an object or of pose bones, and return its points.

    THE GAP: nothing computed or read a motion path, so arcs and spacing - the two things a path is drawn to judge -
    had no answer short of the caller sampling world matrices itself.

    object; bones [names] for bone paths (else the object's own). action CALCULATE (default) | UPDATE | CLEAR. type
    RANGE (default) | CURRENT_FRAME; range SCENE (default) | KEYS_ALL | KEYS_SELECTED | MANUAL with frameStart and
    frameEnd (both, inclusive); frameBefore / frameAfter for CURRENT_FRAME; frameStep (display); bakeLocation HEADS |
    TAILS for bones; display {lineThickness 1-6, lines, color, colorPost (4.2+), useCustomColor}; showFrameNumbers and
    the other show_* flags inside display are not taken - they are viewport drawing. A calculate clears the chosen
    paths first (see the module docstring: a bone path keeps its old bake location otherwise). The mode, the active
    object, the bone selection and the frame are put back. Reply: paths [{bone, frameStart, frameEnd, pointCount,
    points [{frame, co (world), gap}], length, maxGap, minGap, bakeLocation, display}] and the settings read back.
    """
    reject_unknown(params, _MP_KEYS, "motion_paths")
    obj = get_object(take(params, "object", "name", required=True, kind=str))
    action = _enum(params, "action", ("CALCULATE", "UPDATE", "CLEAR"), "CALCULATE")
    bones = params.get("bones")
    if bones is not None:
        if obj.type != "ARMATURE":
            raise MifOpError("bones are for an armature; '%s' is a %s - leave bones out for its own path. %s"
                             % (obj.name, obj.type, _TAIL))
        if not isinstance(bones, list) or not bones or any(not isinstance(b, str) for b in bones):
            raise MifOpError("bones is a non-empty list of bone names. %s" % _TAIL)
        missing = [b for b in bones if obj.pose.bones.get(b) is None]
        if missing:
            raise MifOpError("'%s' has no bone %s (it has %s). %s" % (obj.name, ", ".join(missing),
                                                                      ", ".join(b.name for b in obj.pose.bones[:20]),
                                                                      _TAIL))
    ptype = _enum(params, "type", ("RANGE", "CURRENT_FRAME"), "RANGE")
    prange = _enum(params, "range", ("SCENE", "KEYS_ALL", "KEYS_SELECTED", "MANUAL"), "SCENE")
    bake = _enum(params, "bakeLocation", ("HEADS", "TAILS"), None)
    if bake is not None and bones is None:
        raise MifOpError("bakeLocation is for bone paths (bones); an object's path follows its origin. %s" % _TAIL)
    fs, fe = params.get("frameStart"), params.get("frameEnd")
    if prange == "MANUAL":
        if fs is None or fe is None:
            raise MifOpError("range MANUAL needs frameStart and frameEnd. %s" % _TAIL)
        fs, fe = finite_int(fs, "frameStart"), finite_int(fe, "frameEnd")
        if fe <= fs:
            raise MifOpError("frameEnd (%d) must be after frameStart (%d). %s" % (fe, fs, _TAIL))
    elif fs is not None or fe is not None:
        raise MifOpError("frameStart / frameEnd belong to range MANUAL; range %s takes its frames from %s. %s"
                         % (prange, {"SCENE": "the scene", "KEYS_ALL": "every key", "KEYS_SELECTED":
                                     "the selected keys"}[prange], _TAIL))
    before = finite_int(params["frameBefore"], "frameBefore") if params.get("frameBefore") is not None else None
    after = finite_int(params["frameAfter"], "frameAfter") if params.get("frameAfter") is not None else None
    if (before is not None or after is not None) and ptype != "CURRENT_FRAME":
        raise MifOpError("frameBefore / frameAfter belong to type CURRENT_FRAME. %s" % _TAIL)
    step = finite_int(params["frameStep"], "frameStep") if params.get("frameStep") is not None else None
    if step is not None and step < 1:
        raise MifOpError("frameStep is 1 or more, got %d. %s" % (step, _TAIL))
    disp = _display(params)
    max_points = finite_int(params.get("maxPoints", 1000), "maxPoints")
    if not 1 <= max_points <= 10000:
        raise MifOpError("maxPoints is 1-10000, got %d. %s" % (max_points, _TAIL))
    pbs = [obj.pose.bones[b] for b in bones] if bones else []
    if action == "UPDATE":
        have = [pb.name for pb in pbs if pb.motion_path] if bones else ([obj.name] if obj.motion_path else [])
        if not have:
            raise MifOpError("there is no motion path on %s to update - calculate one first. %s"
                             % ("bones " + ", ".join(bones) if bones else "'%s'" % obj.name, _TAIL))
    sc = bpy.context.scene
    start_frame = sc.frame_current
    avs = (obj.pose if bones else obj).animation_visualization.motion_path
    sim = {}
    if action != "CLEAR":
        if prange == "MANUAL":
            span = (fs, fe)
        elif action == "UPDATE" and "range" not in params and avs.range == "MANUAL":
            span = (avs.frame_start, avs.frame_end)
        else:
            span = (sc.frame_start, sc.frame_end)
        sim = simulation_guard(params, list(range(span[0], span[1] + 1)), "motion_paths",
                               tail="NOTHING was calculated.")
    vl = bpy.context.view_layer
    prev_active = vl.objects.active
    prev_mode = obj.mode
    sel_attr = _select_attr()
    bone_sel = {pb.name: bool((pb if sel_attr == "pose" else pb.bone).select) for pb in obj.pose.bones} \
        if bones else None
    obj_sel = obj.select_get()
    cleared = []
    try:
        if action != "CLEAR":
            # an UPDATE recomputes the path as it was set up: only what this call names is changed, never a default
            if action == "CALCULATE" or "type" in params:
                avs.type = ptype
            if action == "CALCULATE" or "range" in params:
                avs.range = prange
            if prange == "MANUAL":
                avs.frame_start, avs.frame_end = fs, fe
            if before is not None:
                avs.frame_before = before
            if after is not None:
                avs.frame_after = after
            if step is not None:
                avs.frame_step = step
            if bake is not None:
                avs.bake_location = bake
        vl.objects.active = obj
        obj.select_set(True)
        if bones:
            for pb in obj.pose.bones:
                (pb if sel_attr == "pose" else pb.bone).select = pb.name in bones
            if obj.mode != "POSE":
                with bpy.context.temp_override(active_object=obj, object=obj):
                    bpy.ops.object.mode_set(mode="POSE")
            ns = bpy.ops.pose
        else:
            ns = bpy.ops.object
        override = dict(object=obj, active_object=obj, selected_objects=[obj], selected_editable_objects=[obj])
        with bpy.context.temp_override(**override):
            if action in ("CLEAR", "CALCULATE"):
                had = [pb.name for pb in pbs if pb.motion_path] if bones else ([obj.name] if obj.motion_path else [])
                if had:
                    ns.paths_clear(only_selected=True)
                cleared = had
            if action != "CLEAR":
                excl = _MP_END_EXCLUSIVE.get(tuple(bpy.app.version))
                _mp_run(ns, avs, action, ptype, bones, sc, adjust=bool(excl))
                if excl is None:
                    # LEARNED ONCE PER BUILD from the path Blender made (see _mp_run): one point short of the range's
                    # last frame means this build ends a range before its frame_end - then the path is made again.
                    path0 = pbs[0].motion_path if bones else obj.motion_path
                    end = _mp_inclusive_end(avs, sc)
                    excl = path0 is not None and end is not None and path0.frame_end == end
                    if end is not None and path0 is not None:
                        _MP_END_EXCLUSIVE[tuple(bpy.app.version)] = excl
                    if excl:
                        if action == "CALCULATE":
                            ns.paths_clear(only_selected=True)
                        _mp_run(ns, avs, action, ptype, bones, sc, adjust=True)
    finally:
        if bones:
            if obj.mode != prev_mode:
                with bpy.context.temp_override(active_object=obj, object=obj):
                    bpy.ops.object.mode_set(mode=prev_mode)
            for pb in obj.pose.bones:
                (pb if sel_attr == "pose" else pb.bone).select = bone_sel.get(pb.name, False)
        obj.select_set(obj_sel)
        vl.objects.active = prev_active
        if sc.frame_current != start_frame:
            sc.frame_set(start_frame)
    paths = []
    targets = [(pb.motion_path, pb.name) for pb in pbs] if bones else [(obj.motion_path, None)]
    for path, bone in targets:
        if path is None:
            continue
        for attr, v in disp.items():
            setattr(path, attr, v)
        paths.append(_path_row(path, max_points, bone))
    out = {"object": obj.name, "action": action, "paths": paths,
           "settings": {"type": avs.type, "range": avs.range, "frameStart": avs.frame_start, "frameEnd": avs.frame_end,
                        "frameBefore": avs.frame_before, "frameAfter": avs.frame_after, "frameStep": avs.frame_step,
                        "bakeLocation": avs.bake_location if bones else None},
           "frameRestored": sc.frame_current == start_frame, "modeRestored": obj.mode == prev_mode}
    if action == "CLEAR":
        out["cleared"] = cleared
        if not cleared:
            out["note"] = "there was no motion path to clear."
    elif not paths:
        raise MifOpError("Blender finished paths_calculate and made no path on %s - nothing to report."
                         % ("bones " + ", ".join(bones) if bones else "'%s'" % obj.name))
    elif bones and len(paths) != len(bones):
        out["missing"] = [b for b in bones if obj.pose.bones[b].motion_path is None]
    out.update(sim)
    return out


# --------------------------------------------------------------------------------------------- spread a pose change

_SP_KEYS = ("object", "name", "bones", "namePrefix", "all", "frame", "before", "after", "falloff", "influence", "keys",
            "channels", "mode", "endFrame", "slider", "factor")
_SP_SLIDERS = ("BREAKDOWN", "PUSH", "RELAX")


def _neighbour_keys(fc, frame):
    """(frame, value) of the curve's nearest key before and after `frame` - None where there is none."""
    prev = nxt = None
    for k in fc.keyframe_points:
        t = k.co[0]
        if t < frame - 1e-6 and (prev is None or t > prev[0]):
            prev = (t, k.co[1])
        elif t > frame + 1e-6 and (nxt is None or t < nxt[0]):
            nxt = (t, k.co[1])
    return prev, nxt


def _spread_by_slider(obj, rows, frame, slider, factor):
    """spread_pose slider: Blender's pose sliders (pose.breakdown / push / relax, 5.2 bl_ui/space_view3d.py:4297-4299),
    which are modal and need a region, as their math. Per channel with a key before and after the frame: BREAKDOWN keys
    prev + (next - prev) x factor; PUSH moves the current value away from the straight line between the two keys by
    factor (current + (current - line) x factor), RELAX toward it (current + (line - current) x factor). A quaternion is
    normalized after its four channels are set."""
    plan, skipped = [], []
    for pb, attr, i, fc in rows:
        if fc is None:
            skipped.append('pose.bones["%s"].%s[%d] (no curve)' % (pb.name, attr, i))
            continue
        prev, nxt = _neighbour_keys(fc, frame)
        if prev is None or nxt is None:
            skipped.append('pose.bones["%s"].%s[%d] (no key %s frame %d)'
                           % (pb.name, attr, i, "before" if prev is None else "after", frame))
            continue
        cur = getattr(pb, attr)[i]
        u = (frame - prev[0]) / (nxt[0] - prev[0])
        line = prev[1] + (nxt[1] - prev[1]) * u
        if slider == "BREAKDOWN":
            val = prev[1] + (nxt[1] - prev[1]) * factor
        elif slider == "PUSH":
            val = cur + (cur - line) * factor
        else:
            val = cur + (line - cur) * factor
        plan.append([fc, pb, attr, i, cur, val, prev, nxt])
    if not plan:
        raise MifOpError("no chosen channel has a key before AND after frame %d to slide between: %s. %s"
                         % (frame, "; ".join(skipped[:8]) or "none chosen", _TAIL))
    quats = {}
    for row in plan:
        if row[2] == "rotation_quaternion":
            quats.setdefault(row[1].name, {})[row[3]] = row
    for name, comps in quats.items():
        pb = comps[next(iter(comps))][1]
        q = [comps[k][5] if k in comps else pb.rotation_quaternion[k] for k in range(4)]
        n = math.sqrt(sum(x * x for x in q)) or 1.0
        for k, row in comps.items():
            row[5] = q[k] / n
    curves = []
    for fc, pb, attr, i, cur, val, prev, nxt in plan:
        fc.keyframe_points.insert(frame, val, options={"FAST"})
        fc.update()
        vec = list(getattr(pb, attr))
        vec[i] = val
        setattr(pb, attr, vec)
        curves.append({"dataPath": fc.data_path, "index": i, "before": round(cur, 6), "after": round(fc.evaluate(frame), 6),
                       "between": [[round(prev[0], 3), round(prev[1], 6)], [round(nxt[0], 3), round(nxt[1], 6)]]})
    obj.update_tag()
    return {"object": obj.name, "frame": frame, "route": "slider", "slider": slider, "factor": factor, "curves": curves,
            "skipped": skipped or None}
_SP_CHANNELS = ("location", "rotation", "scale")
_SP_MODES = ("NEXT_KEY", "LAST_KEY", "BEFORE_FRAME", "BEFORE_END")
_REST = {"location": (0.0, 0.0, 0.0), "rotation_euler": (0.0, 0.0, 0.0), "rotation_quaternion": (1.0, 0.0, 0.0, 0.0),
         "rotation_axis_angle": (0.0, 0.0, 1.0, 0.0), "scale": (1.0, 1.0, 1.0)}


def _falloff(kind, u):
    """Weight at u = distance / span (0 at the edited frame, 1 at the edge of the range)."""
    u = min(max(u, 0.0), 1.0)
    if kind == "NONE":
        return 1.0
    if kind == "LINEAR":
        return 1.0 - u
    if kind == "SMOOTH":
        return 1.0 - (3.0 * u * u - 2.0 * u * u * u)
    if kind == "EASE_IN":
        return (1.0 - u) ** 2
    return 1.0 - u * u                                        # EASE_OUT


def _bone_curves(obj, pbs, channels):
    """[(pose bone, channel attr, index, fcurve or None)] for the chosen bones and channels, in the bone's rotation
    mode."""
    by_path = {}
    for fc in _fcurves(obj):
        by_path[(fc.data_path, fc.array_index)] = fc
    rows = []
    for pb in pbs:
        for ch in channels:
            attr = rotation_path(pb) if ch == "rotation" else ch
            n = len(getattr(pb, attr))
            for i in range(n):
                rows.append((pb, attr, i, by_path.get(('pose.bones["%s"].%s' % (pb.name, attr), i))))
    return rows


def op_spread_pose(params):
    """Spread the change made to bones' pose at a frame over the frames around it, with a falloff - or run Blender's
    own pose propagate.

    THE GAP: set_bone_pose and set_keyframe write one frame, so an edit at a strike could not be carried back into the
    anticipation that leads to it, or a correction faded out over the frames after.

    object (an armature); bones [names] | namePrefix | all; frame F. THE CHANGE is each chosen channel's current value
    (pose it with set_bone_pose first) minus its curve at F. Falloff route: before and after (frames), falloff SMOOTH
    (default) | LINEAR | EASE_IN | EASE_OUT | NONE, influence 0-1 (default 1), keys ALL_FRAMES (default: a key on every
    frame of [F - before, F + after]) | EXISTING (only the keys already in that range, and F); each frame gets its
    old value plus the change x falloff(distance / span) x influence. channels location | rotation | scale (default
    all three). A channel with no curve is not spread and is named. Mode route: mode NEXT_KEY | LAST_KEY |
    BEFORE_FRAME (endFrame) | BEFORE_END - the pose is keyed at F and Blender's pose.propagate copies it on (see the
    module docstring). Slider route (second 1.3 audit, 2026-10-25): slider BREAKDOWN | PUSH | RELAX with factor (0-1,
    default 0.5) - the pose sliders' in-between, exaggeration and relax, keyed at F (see _spread_by_slider). Reply: curves [{dataPath, index, change, keysWritten, frames, largestChange}], unkeyed, and for
    the mode route each curve's keys before and after.
    """
    reject_unknown(params, _SP_KEYS, "spread_pose")
    obj = _armature(params)
    require_editable(obj, "spread a pose on")
    pbs = pick_bones(obj, params, "changed")
    # THE POSE IS THE CURRENT FRAME'S: a bone's channels hold the pose at the scene's frame (posed there by hand or by
    # set_bone_pose), so a change measured against the curve at any other frame would be a difference of two frames.
    here = bpy.context.scene.frame_current
    frame = finite_int(params.get("frame", here), "frame")
    if frame != here:
        raise MifOpError("frame %d is not the scene's frame (%d): the bones' channels hold the pose at %d, so pose "
                         "them at %d (set_frame_range current, then set_bone_pose) and spread from there. %s"
                         % (frame, here, here, frame, _TAIL))
    mode = _enum(params, "mode", _SP_MODES, None)
    chans = params.get("channels", list(_SP_CHANNELS))
    if (not isinstance(chans, list) or not chans or any(c not in _SP_CHANNELS for c in chans)
            or len(set(chans)) != len(chans)):
        raise MifOpError("channels is a list of %s, got %r. %s" % (", ".join(_SP_CHANNELS), chans, _TAIL))
    rows = _bone_curves(obj, pbs, chans)
    slider = _enum(params, "slider", _SP_SLIDERS, None)
    if slider is not None:
        clash = [k for k in ("before", "after", "falloff", "influence", "keys", "mode", "endFrame")
                 if params.get(k) is not None]
        if clash:
            raise MifOpError("slider keys one in-between at frame and takes no %s. %s" % (", ".join(clash), _TAIL))
        factor = finite_floats([params.get("factor", 0.5)], "factor")[0]
        if not 0.0 <= factor <= 1.0:
            raise MifOpError("factor is 0-1, got %g. %s" % (factor, _TAIL))
        return _spread_by_slider(obj, rows, frame, slider, factor)
    if params.get("factor") is not None:
        raise MifOpError("factor belongs to slider. %s" % _TAIL)
    if mode is not None:
        clash = [k for k in ("before", "after", "falloff", "influence", "keys") if params.get(k) is not None]
        if clash:
            raise MifOpError("mode runs Blender's pose propagate, which has no %s - pass mode, or the falloff "
                             "route's before / after / falloff / influence / keys. %s" % (", ".join(clash), _TAIL))
        end_frame = None
        if mode == "BEFORE_FRAME":
            if params.get("endFrame") is None:
                raise MifOpError("mode BEFORE_FRAME needs endFrame - the frame the pose is copied up to. %s" % _TAIL)
            end_frame = finite_int(params["endFrame"], "endFrame")
            if end_frame <= frame:
                raise MifOpError("endFrame (%d) must be after frame (%d). %s" % (end_frame, frame, _TAIL))
        elif params.get("endFrame") is not None:
            raise MifOpError("endFrame belongs to mode BEFORE_FRAME. %s" % _TAIL)
        return _spread_by_propagate(obj, pbs, rows, frame, mode, end_frame, chans)
    if params.get("endFrame") is not None:
        raise MifOpError("endFrame belongs to mode BEFORE_FRAME. %s" % _TAIL)
    before = finite_int(params.get("before", 0), "before")
    after = finite_int(params.get("after", 0), "after")
    if before < 0 or after < 0 or before + after == 0:
        raise MifOpError("before and after are frames, 0 or more, and at least one above 0 - got %d and %d. %s"
                         % (before, after, _TAIL))
    falloff = _enum(params, "falloff", ("SMOOTH", "LINEAR", "EASE_IN", "EASE_OUT", "NONE"), "SMOOTH")
    influence = finite_floats([params.get("influence", 1.0)], "influence")[0]
    if not 0.0 < influence <= 1.0:
        raise MifOpError("influence is above 0 and at most 1, got %g. %s" % (influence, _TAIL))
    keys = _enum(params, "keys", ("ALL_FRAMES", "EXISTING"), "ALL_FRAMES")
    lo, hi = frame - before, frame + after
    plan, unkeyed = [], []
    for pb, attr, i, fc in rows:
        cur = getattr(pb, attr)[i]
        if fc is None:
            rest = _REST.get(attr, (0.0,) * 4)[i]
            if abs(cur - rest) > 1e-9:
                unkeyed.append('pose.bones["%s"].%s[%d]' % (pb.name, attr, i))
            continue
        change = cur - fc.evaluate(frame)
        if abs(change) <= 1e-9:
            continue
        if keys == "ALL_FRAMES":
            frames = list(range(lo, hi + 1))
        else:
            frames = sorted({int(round(k.co[0])) for k in fc.keyframe_points if lo <= k.co[0] <= hi} | {frame})
        writes = []
        for t in frames:
            d = t - frame
            span = before if d < 0 else after
            w = 1.0 if d == 0 else (_falloff(falloff, abs(d) / float(span)) if span else 0.0)
            old = fc.evaluate(t)
            writes.append((t, old, old + change * w * influence))
        plan.append((fc, pb, attr, i, change, writes))
    if not plan:
        raise MifOpError("nothing to spread: every chosen channel of %s already equals its curve at frame %d%s - pose "
                         "the bones (set_bone_pose) at that frame first. %s"
                         % (", ".join(pb.name for pb in pbs[:10]), frame,
                            "" if not unkeyed else " (and %s ha%s no curve)" % (", ".join(unkeyed[:6]),
                                                                                  "s" if len(unkeyed) == 1 else "ve"),
                            _TAIL))
    curves = []
    for fc, pb, attr, i, change, writes in plan:
        for t, _old, new in writes:
            # FAST, not REPLACE: REPLACE only overwrites keys that exist and adds none (an ALL_FRAMES spread wrote
            # nothing between keys with it, measured on 5.2.2); a key already on the frame is replaced either way
            fc.keyframe_points.insert(t, new, options={"FAST"})
        fc.update()
        curves.append({"dataPath": fc.data_path, "index": i, "change": round(change, 6), "keysWritten": len(writes),
                       "frames": [writes[0][0], writes[-1][0]],
                       "largestChange": round(max(abs(n - o) for _, o, n in writes), 6)})
    obj.update_tag()
    out = {"object": obj.name, "frame": frame, "route": "falloff", "falloff": falloff, "influence": influence,
           "keys": keys, "range": [lo, hi], "curves": curves, "unkeyed": unkeyed or None}
    if unkeyed:
        out["unkeyedNote"] = ("these channels differ from rest and have no curve, so there was nothing to spread their "
                              "change over - key them (set_keyframe) and spread again.")
    return out


def _spread_by_propagate(obj, pbs, rows, frame, mode, end_frame, chans):
    sc = bpy.context.scene
    start = sc.frame_current
    saved = {pb.name: [list(getattr(pb, rotation_path(pb) if c == "rotation" else c)) for c in chans] for pb in pbs}
    before = {(fc.data_path, fc.array_index): [(round(k.co[0], 3), round(k.co[1], 6)) for k in fc.keyframe_points]
              for _, _, _, fc in rows if fc is not None}
    vl = bpy.context.view_layer
    prev_active, prev_mode, obj_sel = vl.objects.active, obj.mode, obj.select_get()
    sel = None
    try:
        sc.frame_set(frame)
        for pb in pbs:
            for c, vals in zip(chans, saved[pb.name], strict=True):
                setattr(pb, rotation_path(pb) if c == "rotation" else c, vals)
            key_bone_channels(pb, frame, [rotation_path(pb) if c == "rotation" else c for c in chans])
        vl.objects.active = obj
        obj.select_set(True)
        sel = select_bones(obj, {pb.name for pb in pbs})
        if obj.mode != "POSE":
            with bpy.context.temp_override(active_object=obj, object=obj):
                bpy.ops.object.mode_set(mode="POSE")
        kw = {"mode": mode}
        if end_frame is not None:
            kw["end_frame"] = end_frame
        with bpy.context.temp_override(object=obj, active_object=obj, selected_pose_bones=list(pbs)):
            r = bpy.ops.pose.propagate(**kw)
        if "FINISHED" not in r:
            raise MifOpError("Blender's pose.propagate returned %s." % sorted(r))
    finally:
        if obj.mode != prev_mode:
            with bpy.context.temp_override(active_object=obj, object=obj):
                bpy.ops.object.mode_set(mode=prev_mode)
        if sel is not None:
            restore_bone_selection(obj, sel)
        obj.select_set(obj_sel)
        vl.objects.active = prev_active
        sc.frame_set(start)
    curves = []
    for fc in _fcurves(obj):
        key = (fc.data_path, fc.array_index)
        if not any(fc.data_path == 'pose.bones["%s"].%s' % (pb.name, rotation_path(pb) if c == "rotation" else c)
                   for pb in pbs for c in chans):
            continue
        now = [(round(k.co[0], 3), round(k.co[1], 6)) for k in fc.keyframe_points]
        if now != before.get(key):
            curves.append({"dataPath": fc.data_path, "index": fc.array_index, "keysBefore": before.get(key),
                           "keysAfter": now})
    out = {"object": obj.name, "frame": frame, "route": "propagate", "mode": mode, "endFrame": end_frame,
           "curves": curves, "frameRestored": sc.frame_current == start}
    if not curves:
        out["note"] = ("Blender's propagate changed no key: nothing past frame %d for %s to reach, or the pose already "
                       "matched them." % (frame, mode))
    return out


# --------------------------------------------------------------------------------------------- smooth / thin / blend keys

_FK_KEYS = ("object", "name", "dataPaths", "bones", "all", "frameStart", "frameEnd", "filter", "settings")
_FK_FILTERS = {"SMOOTH": (), "GAUSSIAN": ("sigma", "width"), "BUTTERWORTH": ("cutoff", "order"),
               "DECIMATE": ("ratio", "error"), "BLEND_TO_NEIGHBOR": ("factor",),
               # THE GRAPH EDITOR SLIDER FAMILY (second 1.3 audit, 2026-10-25; 5.2 bl_ui/space_graph.py:414-422, most of
               # it absent from 3.6's UI) - as native math on the keys between the range's two neighbour keys
               "EASE": ("factor",), "BLEND_OFFSET": ("factor",), "BLEND_TO_EASE": ("factor",),
               "BLEND_TO_DEFAULT": ("factor",), "MATCH_SLOPE": ("factor",), "SCALE_AVERAGE": ("factor",),
               "SCALE_FROM_NEIGHBOR": ("factor",), "PUSH_PULL": ("factor",), "SHEAR": ("factor", "direction"),
               "TIME_OFFSET": ("frames",)}
# filter -> (factor low, high, default); the neighbour keys a filter needs: L the key before the range, R after it
_FK_SLIDERS = {"EASE": (-1.0, 1.0, 0.5, "LR"), "BLEND_OFFSET": (-1.0, 1.0, 0.5, "side"),
               "BLEND_TO_EASE": (-1.0, 1.0, 0.5, "LR"), "BLEND_TO_DEFAULT": (0.0, 1.0, 0.5, ""),
               "MATCH_SLOPE": (-1.0, 1.0, 0.5, "slope"), "SCALE_AVERAGE": (0.0, 2.0, 1.0, ""),
               "SCALE_FROM_NEIGHBOR": (-1.0, 1.0, 0.5, "side"), "PUSH_PULL": (0.0, 2.0, 1.0, "LR"),
               "SHEAR": (-1.0, 1.0, 0.5, "LR"), "TIME_OFFSET": (None, None, None, "")}


def _ease_shape(u, factor):
    """0..1 -> 0..1: factor > 0 eases in (slow start), factor < 0 eases out; the exponent is 1 + 4 x |factor|."""
    p = 1.0 + 4.0 * abs(factor)
    return u ** p if factor >= 0 else 1.0 - (1.0 - u) ** p


def _channel_default(fc):
    """The value a channel rests at: 1 for scale and a quaternion's w, else 0."""
    tail = fc.data_path.rsplit(".", 1)[-1]
    if tail == "scale" or (tail == "rotation_quaternion" and fc.array_index == 0):
        return 1.0
    if tail == "rotation_axis_angle" and fc.array_index == 2:
        return 1.0
    return 0.0


def _fk_curves(obj, params):
    """The curves dataPaths ['path' or 'path[i]'] | bones [names] | all true name, on the object, its data and its
    shape keys (5.0's slotted curves through _fcurves)."""
    paths = params.get("dataPaths")
    bones = params.get("bones")
    every = params.get("all")
    picks = [k for k, v in (("dataPaths", paths), ("bones", bones), ("all", every)) if v is not None]
    if len(picks) != 1:
        raise MifOpError("pick the curves with exactly one of dataPaths, bones or all:true - got %s. %s"
                         % (picks or "none", _TAIL))
    curves = [fc for _label, holder in _anim_holders(obj) for fc in _fcurves(holder)]
    if every is not None:
        if every is not True:
            raise MifOpError("all takes true. %s" % _TAIL)
        out = curves
    elif bones is not None:
        if not isinstance(bones, list) or not bones or any(not isinstance(b, str) for b in bones):
            raise MifOpError("bones is a non-empty list of bone names. %s" % _TAIL)
        have = {_bone_of_path(fc.data_path) for fc in curves}
        missing = [b for b in bones if b not in have]
        if missing:
            raise MifOpError("no curve of '%s' animates bone %s. %s" % (obj.name, ", ".join(missing), _TAIL))
        out = [fc for fc in curves if _bone_of_path(fc.data_path) in set(bones)]
    else:
        if not isinstance(paths, list) or not paths or any(not isinstance(p, str) for p in paths):
            raise MifOpError("dataPaths is a non-empty list of data paths ('location', 'location[0]'). %s" % _TAIL)
        out = []
        for p in paths:
            base, idx = p, None
            if p.endswith("]") and "[" in p and p[p.rindex("[") + 1:-1].isdigit():
                base, idx = p[:p.rindex("[")], int(p[p.rindex("[") + 1:-1])
            hit = [fc for fc in curves if fc.data_path == base and (idx is None or fc.array_index == idx)]
            if not hit:
                raise MifOpError("'%s' has no curve on %r (its curves: %s). %s"
                                 % (obj.name, p, ", ".join(sorted({"%s[%d]" % (fc.data_path, fc.array_index)
                                                                   for fc in curves})[:12]) or "none", _TAIL))
            out.extend(fc for fc in hit if fc not in out)
    if not out:
        raise MifOpError("'%s' has no animation curve to filter. %s" % (obj.name, _TAIL))
    return out


def _fk_settings(kind, raw, fps):
    raw = raw or {}
    if not isinstance(raw, dict):
        raise MifOpError("settings is an object, got %r. %s" % (raw, _TAIL))
    unknown = sorted(set(raw) - set(_FK_FILTERS[kind]))
    if unknown:
        raise MifOpError("%s takes settings %s - not %s. %s" % (kind, ", ".join(_FK_FILTERS[kind]) or "none",
                                                                ", ".join(unknown), _TAIL))

    def num(key, default, lo, hi, lo_open=False):
        v = raw.get(key, default)
        v = finite_floats([v], "settings." + key)[0]
        if (v <= lo if lo_open else v < lo) or v > hi:
            raise MifOpError("settings.%s is %s%g to %g, got %g. %s" % (key, "above " if lo_open else "", lo, hi, v,
                                                                       _TAIL))
        return v
    s = {}
    if kind == "GAUSSIAN":
        s["sigma"] = num("sigma", 2.0, 0.0, 1000.0, lo_open=True)
        s["width"] = num("width", 3.0 * s["sigma"], 0.0, 10000.0, lo_open=True)
    elif kind == "BUTTERWORTH":
        s["cutoff"] = num("cutoff", 3.0, 0.0, 10000.0, lo_open=True)
        if s["cutoff"] >= fps / 2.0:
            raise MifOpError("settings.cutoff %g Hz is not under half the scene's %g fps (%g Hz) - nothing could pass "
                             "a lower one. %s" % (s["cutoff"], fps, fps / 2.0, _TAIL))
        order = raw.get("order", 4)
        if order not in (2, 4, 6, 8):
            raise MifOpError("settings.order is 2, 4, 6 or 8, got %r. %s" % (order, _TAIL))
        s["order"] = order
    elif kind == "DECIMATE":
        if ("ratio" in raw) == ("error" in raw):
            raise MifOpError("DECIMATE takes settings.ratio (the share of keys to keep) or settings.error (the largest "
                             "change allowed), one of them. %s" % _TAIL)
        if "ratio" in raw:
            s["ratio"] = num("ratio", None, 0.0, 1.0, lo_open=True)
        else:
            s["error"] = num("error", None, 0.0, 1e9, lo_open=True)
    elif kind == "BLEND_TO_NEIGHBOR":
        s["factor"] = num("factor", 0.5, -1.0, 1.0)
        if s["factor"] == 0:
            raise MifOpError("settings.factor 0 blends nothing: -1 to 1, negative toward the key before the range, "
                             "positive toward the key after it. %s" % _TAIL)
    elif kind == "TIME_OFFSET":
        if "frames" not in raw:
            raise MifOpError("TIME_OFFSET needs settings.frames: how far the values shift in time. %s" % _TAIL)
        s["frames"] = num("frames", None, -100000.0, 100000.0)
        if s["frames"] == 0:
            raise MifOpError("settings.frames 0 shifts nothing. %s" % _TAIL)
    elif kind in _FK_SLIDERS:
        lo, hi, dflt, _need = _FK_SLIDERS[kind]
        s["factor"] = num("factor", dflt, lo, hi)
        if kind == "SHEAR":
            d = str(raw.get("direction", "FROM_LEFT")).upper()
            if d not in ("FROM_LEFT", "FROM_RIGHT"):
                raise MifOpError("settings.direction is FROM_LEFT or FROM_RIGHT, got %r. %s" % (raw.get("direction"),
                                                                                                 _TAIL))
            s["direction"] = d
    return s


def _slider_values(kind, s, fc, kps, run, frames, values):
    """New values for the keys of `run` under one of the Graph Editor sliders (native math, every build)."""
    first, last = run[0], run[-1]
    L = (kps[first - 1].co[0], kps[first - 1].co[1]) if first > 0 else None
    R = (kps[last + 1].co[0], kps[last + 1].co[1]) if last + 1 < len(kps) else None
    f = s.get("factor", 0.0)
    need = _FK_SLIDERS[kind][3]
    label = "%s[%d]" % (fc.data_path, fc.array_index)

    def want(side):
        if side == "L" and L is None or side == "R" and R is None:
            raise MifOpError("%s has no key %s the range, which %s reads. %s"
                             % (label, "before" if side == "L" else "after", kind, _TAIL))
    if need == "LR":
        want("L")
        want("R")
    elif need in ("side", "slope"):
        want("L" if f < 0 else "R")
    if kind == "TIME_OFFSET":
        return [fc.evaluate(t - s["frames"]) for t in frames]
    if kind == "SCALE_AVERAGE":
        avg = sum(values) / len(values)
        return [avg + (v - avg) * f for v in values]
    if kind == "BLEND_TO_DEFAULT":
        d = _channel_default(fc)
        return [v + (d - v) * f for v in values]
    if kind == "BLEND_OFFSET":
        delta = (L[1] - values[0]) if f < 0 else (R[1] - values[-1])
        return [v + delta * abs(f) for v in values]
    if kind == "SCALE_FROM_NEIGHBOR":
        anchor = L[1] if f < 0 else R[1]
        return [anchor + (v - anchor) * (1.0 + abs(f)) for v in values]
    if kind == "MATCH_SLOPE":
        if f < 0:
            p = kps[first - 2] if first > 1 else None
            if p is None:
                raise MifOpError("%s needs two keys before the range to read a slope from. %s" % (label, _TAIL))
            slope = (L[1] - p.co[1]) / ((L[0] - p.co[0]) or 1.0)
            line = [L[1] + slope * (t - L[0]) for t in frames]
        else:
            p = kps[last + 2] if last + 2 < len(kps) else None
            if p is None:
                raise MifOpError("%s needs two keys after the range to read a slope from. %s" % (label, _TAIL))
            slope = (p.co[1] - R[1]) / ((p.co[0] - R[0]) or 1.0)
            line = [R[1] + slope * (t - R[0]) for t in frames]
        return [v + (ln - v) * abs(f) for v, ln in zip(values, line, strict=True)]
    span = (R[0] - L[0]) or 1.0
    us = [(t - L[0]) / span for t in frames]
    if kind == "EASE":
        return [L[1] + (R[1] - L[1]) * _ease_shape(u, f) for u in us]
    if kind == "BLEND_TO_EASE":
        return [v + (L[1] + (R[1] - L[1]) * _ease_shape(u, 0.5 if f >= 0 else -0.5) - v) * abs(f)
                for v, u in zip(values, us, strict=True)]
    if kind == "PUSH_PULL":
        return [L[1] + (R[1] - L[1]) * u + (v - (L[1] + (R[1] - L[1]) * u)) * f for v, u in zip(values, us, strict=True)]
    # SHEAR: a ramp added across the segment, rising from the left key (FROM_LEFT) or the right one
    ramp = us if s["direction"] == "FROM_LEFT" else [1.0 - u for u in us]
    return [v + f * (R[1] - L[1]) * r for v, r in zip(values, ramp, strict=True)]


def _butter_sections(order, fc_hz, fps):
    """RBJ low-pass biquads with the Butterworth Q of each pole pair: (b, a) per section."""
    out = []
    w = 2.0 * math.pi * fc_hz / fps
    cw, sw = math.cos(w), math.sin(w)
    for k in range(1, order // 2 + 1):
        q = 1.0 / (2.0 * math.cos((2 * k - 1) * math.pi / (2 * order)))
        alpha = sw / (2.0 * q)
        a0 = 1.0 + alpha
        out.append((((1 - cw) / 2 / a0, (1 - cw) / a0, (1 - cw) / 2 / a0), (1.0, -2.0 * cw / a0, (1 - alpha) / a0)))
    return out


def _biquad_run(x, b, a):
    y, z1, z2 = [], 0.0, 0.0
    # start in the steady state of the first sample, as scipy's lfilter_zi does, so a flat start is not a step
    gain = (b[0] + b[1] + b[2]) / (1.0 + a[1] + a[2])
    z1 = x[0] * gain - b[0] * x[0] if x else 0.0
    z2 = x[0] * (b[2] - a[2] * gain) if x else 0.0
    for v in x:
        o = b[0] * v + z1
        z1 = b[1] * v - a[1] * o + z2
        z2 = b[2] * v - a[2] * o
        y.append(o)
    return y


def _filtfilt(x, sections, pad):
    """Zero-phase: forward and backward through every section, the ends padded by odd reflection (scipy's filtfilt
    default), so the ends are not pulled toward zero."""
    n = len(x)
    pad = min(pad, n - 1)
    ext = [2 * x[0] - x[i] for i in range(pad, 0, -1)] + list(x) + [2 * x[-1] - x[n - 2 - i] for i in range(pad)]
    for b, a in sections:
        ext = _biquad_run(ext, b, a)
        ext = _biquad_run(ext[::-1], b, a)[::-1]
    return ext[pad:pad + n]


def _decimate_pick(frames, values, keep=None, error=None):
    """Indices to keep: the ends, then by greedy refinement the key the piecewise-linear line through the kept keys
    misses most - until `keep` keys, or until no key is missed by more than `error`."""
    chosen = {0, len(frames) - 1}

    def worst():
        ks = sorted(chosen)
        best, at = -1.0, None
        for a, b in zip(ks, ks[1:]):
            for i in range(a + 1, b):
                u = (frames[i] - frames[a]) / float(frames[b] - frames[a])
                d = abs(values[a] + u * (values[b] - values[a]) - values[i])
                if d > best:
                    best, at = d, i
        return best, at
    while True:
        if keep is not None and len(chosen) >= keep:
            break
        d, i = worst()
        if i is None or (error is not None and d <= error):
            break
        chosen.add(i)
    return sorted(chosen)


def _shift_key(kp, value):
    d = value - kp.co[1]
    kp.co[1] = value
    kp.handle_left[1] += d
    kp.handle_right[1] += d


def op_filter_keys(params):
    """Smooth, Gaussian-smooth, Butterworth-filter, decimate or blend an F-curve's keys over a frame range.

    THE GAP: edit_fcurve sets interpolation, easing and handles only; nothing smoothed, thinned or blended keys, and
    Blender's Graph Editor operators for it (graph.smooth, gaussian_smooth, butterworth_smooth from 4.2, decimate,
    blend_to_neighbor) poll for a Graph Editor this bridge has no window for. This implements them on the curves
    directly, so the reply can say what each curve became.

    object; dataPaths ['location', 'location[0]', 'pose.bones["X"].rotation_euler'] | bones [names] | all; frameStart /
    frameEnd (default each curve's whole range) - the keys inside are filtered; filter and settings:
      SMOOTH             each key but the first and last in the range becomes (before + 2 x itself + after) / 4
      GAUSSIAN           {sigma (frames, default 2), width (frames, default 3 sigma)}: a Gaussian-weighted mean of the
                         keys within width frames
      BUTTERWORTH        {cutoff (Hz, under half the scene fps; default 3), order 2|4|6|8 (default 4)}: the curve
                         sampled every frame, low-passed forward and backward (zero phase, odd-reflected ends), and the
                         keys set to it - native on every build (Blender's own is 4.2+)
      DECIMATE           {ratio (share of keys kept) | error (largest change allowed, in the curve's units)}: the
                         ends kept, then keys added back where the curve would be missed most; with error, keys are
                         added until no frame of the range moves by more than error (measured on the rebuilt curve)
      BLEND_TO_NEIGHBOR  {factor -1..1}: each key moves |factor| of the way to the key just before the range
                         (negative) or just after it (positive)
    THE GRAPH EDITOR SLIDERS (second 1.3 audit, 2026-10-25), native math on the keys of the range with L the key before
    it and R the key after, u a key's place between them (0-1) - so every build has them, 3.6 too:
      EASE               {factor -1..1}: the keys set on an ease from L to R, u^(1 + 4|factor|) (in) or its mirror (out)
      BLEND_TO_EASE      {factor -1..1}: each key moves |factor| of the way to that ease (in for +, out for -)
      BLEND_OFFSET       {factor -1..1}: the run shifted so its first key meets L (-) or its last meets R (+), by |factor|
      BLEND_TO_DEFAULT   {factor 0..1}: toward the channel's rest value (1 for scale and a quaternion's w, else 0)
      MATCH_SLOPE        {factor -1..1}: toward the line continuing the slope of the two keys before (-) or after (+)
      SCALE_AVERAGE      {factor 0..2}: scaled about the run's mean value
      SCALE_FROM_NEIGHBOR {factor -1..1}: scaled by 1 + |factor| about L (-) or R (+)
      PUSH_PULL          {factor 0..2}: the run's distance from the straight line L-R multiplied (over 1 pushes)
      SHEAR              {factor -1..1, direction FROM_LEFT | FROM_RIGHT}: a ramp of factor x (R - L) added across it
      TIME_OFFSET        {frames}: each key takes the curve's value from `frames` earlier
    A key's handles move with it. Reply: curves [{dataPath, index, keysBefore, keysAfter, largestChange (the largest
    change of the evaluated curve at any whole frame of the range)}].
    """
    reject_unknown(params, _FK_KEYS, "filter_keys")
    obj = get_object(take(params, "object", "name", required=True, kind=str))
    require_editable(obj, "filter the keys of")
    kind = _enum(params, "filter", tuple(_FK_FILTERS), None)
    if kind is None:
        raise MifOpError("filter is required: %s. %s" % (" | ".join(_FK_FILTERS), _TAIL))
    sc = bpy.context.scene
    fps = sc.render.fps / float(sc.render.fps_base or 1.0)
    settings = _fk_settings(kind, params.get("settings"), fps)
    curves = _fk_curves(obj, params)
    lo = finite_floats([params["frameStart"]], "frameStart")[0] if params.get("frameStart") is not None else None
    hi = finite_floats([params["frameEnd"]], "frameEnd")[0] if params.get("frameEnd") is not None else None
    if lo is not None and hi is not None and hi <= lo:
        raise MifOpError("frameEnd (%g) must be after frameStart (%g). %s" % (hi, lo, _TAIL))
    plans = []
    for fc in curves:
        kps = list(fc.keyframe_points)
        run = [i for i, k in enumerate(kps)
               if (lo is None or k.co[0] >= lo - 1e-6) and (hi is None or k.co[0] <= hi + 1e-6)]
        label = "%s[%d]" % (fc.data_path, fc.array_index)
        need = 1 if kind == "BLEND_TO_NEIGHBOR" or kind in _FK_SLIDERS else 3
        if len(run) < need:
            raise MifOpError("%s has %d key(s) in the range - %s needs %d or more. %s"
                             % (label, len(run), kind, need, _TAIL))
        if kind == "BLEND_TO_NEIGHBOR":
            nb = run[-1] + 1 if settings["factor"] > 0 else run[0] - 1
            if not 0 <= nb < len(kps):
                raise MifOpError("%s has no key %s the range to blend toward (factor %g). %s"
                                 % (label, "after" if settings["factor"] > 0 else "before", settings["factor"],
                                    _TAIL))
        plans.append((fc, run))
    out_rows = []
    for fc, run in plans:
        kps = fc.keyframe_points
        frames = [kps[i].co[0] for i in run]
        values = [kps[i].co[1] for i in run]
        f0, f1 = int(math.floor(frames[0])), int(math.ceil(frames[-1]))
        old = {t: fc.evaluate(t) for t in range(f0, f1 + 1)}
        before = len(kps)
        if kind == "SMOOTH":
            new = [values[0]] + [(values[i - 1] + 2 * values[i] + values[i + 1]) / 4.0
                                 for i in range(1, len(values) - 1)] + [values[-1]]
            for i, v in zip(run, new, strict=True):
                _shift_key(kps[i], v)
        elif kind == "GAUSSIAN":
            sg, wd = settings["sigma"], settings["width"]
            new = []
            for t in frames:
                ws = [(math.exp(-((u - t) ** 2) / (2 * sg * sg)), v) for u, v in zip(frames, values, strict=True)
                      if abs(u - t) <= wd]
                new.append(sum(w * v for w, v in ws) / sum(w for w, _ in ws))
            for i, v in zip(run, new, strict=True):
                _shift_key(kps[i], v)
        elif kind == "BUTTERWORTH":
            series = [old[t] for t in range(f0, f1 + 1)]
            filt = _filtfilt(series, _butter_sections(settings["order"], settings["cutoff"], fps),
                             pad=3 * settings["order"] * int(math.ceil(fps / settings["cutoff"])))
            for i, t in zip(run, frames, strict=True):
                j = t - f0
                a = int(math.floor(j))
                b = min(a + 1, len(filt) - 1)
                _shift_key(kps[i], filt[a] + (j - a) * (filt[b] - filt[a]))
        elif kind == "BLEND_TO_NEIGHBOR":
            f = settings["factor"]
            target = kps[run[-1] + 1].co[1] if f > 0 else kps[run[0] - 1].co[1]
            for i in run:
                _shift_key(kps[i], kps[i].co[1] + abs(f) * (target - kps[i].co[1]))
        elif kind in _FK_SLIDERS:
            new = _slider_values(kind, settings, fc, kps, run, frames, values)
            for i, v in zip(run, new, strict=True):
                _shift_key(kps[i], v)
        else:
            keep = None
            if "ratio" in settings:
                keep = max(2, int(math.ceil(len(run) * settings["ratio"])))
            chosen = _decimate_pick(frames, values, keep=keep, error=settings.get("error"))
            drop = [run[i] for i in range(len(run)) if i not in set(chosen)]
            originals = {run[i]: (frames[i], values[i]) for i in range(len(run))}
            for i in sorted(drop, reverse=True):
                kps.remove(kps[i])
            fc.update()
            if "error" in settings:
                # MEASURED, NOT ASSUMED: the pick reads straight lines, the curve is a Bezier; put keys back where the
                # rebuilt curve still misses by more than error, until it does not
                while True:
                    dev = {t: abs(fc.evaluate(t) - old[t]) for t in old}
                    t_bad = max(dev, key=dev.get)
                    if dev[t_bad] <= settings["error"] or not drop:
                        break
                    back = min(drop, key=lambda i: abs(originals[i][0] - t_bad))
                    drop.remove(back)
                    kps.insert(originals[back][0], originals[back][1], options={"FAST"})
                    fc.update()
        fc.update()
        change = max((abs(fc.evaluate(t) - v) for t, v in old.items()), default=0.0)
        out_rows.append({"dataPath": fc.data_path, "index": fc.array_index, "keysBefore": before,
                         "keysAfter": len(fc.keyframe_points), "keysInRange": len(run), "frames": [frames[0], frames[-1]],
                         "largestChange": round(change, 6)})
    obj.update_tag()
    return {"object": obj.name, "filter": kind, "settings": settings, "curves": out_rows, "fps": fps}


# --------------------------------------------------------------------------------------------- action slots

_SLOT_OPS = ("LIST", "NEW", "RENAME", "ASSIGN", "SEPARATE", "MERGE", "MOVE_TO_NEW", "REPLACE")
_SLOT_KEYS = ("action", "op", "slot", "name", "idType", "object", "target", "newSlot", "fromActions", "slots", "newAction")


def _has_slots():
    return "slots" in bpy.types.Action.bl_rna.properties


def _no_slots(what):
    return MifOpError("%s needs slotted actions, which Blender 4.4 introduced - this is %s, where an action has no "
                      "slots. %s" % (what, bpy.app.version_string, _TAIL))


def _get_action(name, key="action"):
    if not isinstance(name, str) or not name:
        raise MifOpError("'%s' is required - an action name (list_actions). %s" % (key, _TAIL))
    act = bpy.data.actions.get(name)
    if act is None:
        raise MifOpError("no action named '%s' (%s). Present: %s. %s"
                         % (name, key, ", ".join(a.name for a in list(bpy.data.actions)[:20]) or "<none>", _TAIL))
    return act


def _find_slot(act, ref):
    """A slot of act by identifier ('OBBall') or display name ('Ball'), or a refusal listing them."""
    for s in act.slots:
        if ref in (s.identifier, s.name_display):
            return s
    raise MifOpError("action '%s' has no slot '%s'. Its slots: %s. %s"
                     % (act.name, ref, ", ".join(s.identifier for s in act.slots) or "<none>", _TAIL))


def _slot_curves(act, slot):
    out = []
    for layer in act.layers:
        for strip in layer.strips:
            cb = strip.channelbag(slot) if hasattr(strip, "channelbag") else None
            if cb is not None:
                out.extend(cb.fcurves)
    return out


def slot_row(act, slot, curves=True):
    row = {"identifier": slot.identifier, "name": slot.name_display, "idType": slot.target_id_type,
           "users": sorted(u.name for u in slot.users())}
    if curves:
        row["curves"] = len(_slot_curves(act, slot))
    return row


def _bindings(act):
    """Every place `act` plays: [(id, strip or None)] - an ID's active action, and every NLA strip (meta strips too)
    that holds it. Read from bpy.data.user_map, so a material, a world or a Key is found as well as an object."""
    ids = list(bpy.data.user_map(subset=[act]).get(act, ()))
    for idb in list(ids):
        tree = getattr(idb, "node_tree", None)
        if tree is not None and tree not in ids:
            ids.append(tree)
    out = []

    def walk(idb, strips):
        for s in strips:
            if s.action == act:
                out.append((idb, s))
            walk(idb, getattr(s, "strips", ()))

    for idb in ids:
        ad = getattr(idb, "animation_data", None)
        if ad is None:
            continue
        if ad.action == act:
            out.append((idb, None))
        for tr in ad.nla_tracks:
            walk(idb, tr.strips)
    return out


def _bound_slot(idb, strip):
    holder = strip if strip is not None else idb.animation_data
    s = getattr(holder, "action_slot", None)
    return s.identifier if s is not None else None


def _bind(idb, strip, act, slot):
    holder = strip if strip is not None else idb.animation_data
    holder.action = act
    if slot is not None:
        holder.action_slot = slot


def _binding_label(idb, strip):
    return idb.name if strip is None else "%s (strip %s)" % (idb.name, strip.name)


def _copy_curve(src, cb):
    """src into channelbag cb, keys, handles and modifiers. 5.x copies natively (new_from_fcurve); 4.4 by RNA."""
    if hasattr(cb.fcurves, "new_from_fcurve"):
        return cb.fcurves.new_from_fcurve(src)
    dst = cb.fcurves.new(src.data_path, index=src.array_index)
    kps = list(src.keyframe_points)
    dst.keyframe_points.add(len(kps))
    for a, b in zip(kps, dst.keyframe_points):
        for prop in ("interpolation", "easing", "handle_left_type", "handle_right_type", "type", "back", "amplitude",
                     "period", "co", "handle_left", "handle_right"):
            setattr(b, prop, getattr(a, prop))
    for prop in ("extrapolation", "mute", "hide", "lock", "color_mode", "color", "auto_smoothing"):
        setattr(dst, prop, getattr(src, prop))
    for m in src.modifiers:
        n = dst.modifiers.new(m.type)
        for p in m.bl_rna.properties:
            if not p.is_readonly and p.type not in ("POINTER", "COLLECTION") and p.identifier != "rna_type":
                try:
                    setattr(n, p.identifier, getattr(m, p.identifier))
                except (AttributeError, TypeError, ValueError, RuntimeError):
                    pass
    dst.update()
    return dst


def _ensure_bag(act, slot):
    if not len(act.layers):
        act.layers.new("Layer")
    layer = act.layers[0]
    if not len(layer.strips):
        layer.strips.new(type="KEYFRAME")
    strip = layer.strips[0]
    return strip.channelbag(slot) or strip.channelbags.new(slot)


def _split_copy(act, keep, name):
    """A copy of act holding only the slots whose identifiers are in `keep`; the users of those slots rebound to it."""
    new = act.copy()
    new.name = name
    for s in [s for s in new.slots if s.identifier not in keep]:
        new.slots.remove(s)
    rebound = []
    for idb, strip in _bindings(act):
        ident = _bound_slot(idb, strip)
        if ident in keep:
            _bind(idb, strip, new, new.slots[ident])
            rebound.append(_binding_label(idb, strip))
    return new, rebound


def op_edit_action_slots(params):
    """List, make, rename, assign, merge, separate and move an action's slots (4.4+), or replace an action everywhere.

    THE GAP (spec, "Action slots on 4.4 and later"): assign_action binds an existing object slot or the lone one, so a
    second object handed the same action played and keyed into the first object's curves; nothing could give it its own
    slot, list them, or split or merge actions by slot.

    action; op:
      LIST         slots [{identifier, name, idType, users, curves}]
      NEW          name, idType (default OBJECT): a slot no one plays yet
      RENAME       slot (identifier or name), name: the identifier follows ('OBBall' -> 'OBOrbiter')
      ASSIGN       object, target object | data | shape_keys, and slot or newSlot true (a slot made for that holder,
                   named name or after it): the holder plays the action through that slot. Reply sharedWith.
      SEPARATE     one action per slot, named <action>_<slot name>, every user (active or NLA strip) rebound
      MOVE_TO_NEW  slots [..], newAction: those slots and their curves moved to a new action, their users rebound
      MERGE        fromActions [names]: their slots and curves copied into this action and their users rebound to it
      REPLACE      newAction: every user of this action plays newAction instead, through the slot with the same
                   identifier (every build; 4.2 has no slots)
    Users are read from Blender's user map: active actions and NLA strips on objects, data, Keys and node trees.
    A slot (each kind) is measured to count NLA strips that play it among its users (4.4.0 and 5.2.2).
    """
    reject_unknown(params, _SLOT_KEYS, "edit_action_slots")
    op = _enum(params, "op", _SLOT_OPS, None)
    if op is None:
        raise MifOpError("op is required: %s. %s" % (" | ".join(_SLOT_OPS), _TAIL))
    act = _get_action(take(params, "action", kind=str))
    slotted = _has_slots()
    if op == "REPLACE":
        return _slots_replace(act, params, slotted)
    if not slotted:
        raise _no_slots("edit_action_slots %s" % op)
    before = [slot_row(act, s) for s in act.slots]
    reply = {"action": act.name, "op": op, "slotsBefore": before}
    if op == "LIST":
        return {"action": act.name, "slots": before}
    if op == "NEW":
        reply.update(_slots_new(act, params))
    elif op == "RENAME":
        ref, name = take(params, "slot", kind=str), take(params, "name", kind=str)
        if not ref or not name:
            raise MifOpError("RENAME takes slot (identifier or name) and name (the new display name). %s" % _TAIL)
        slot = _find_slot(act, ref)
        was = slot.identifier
        slot.name_display = name
        reply.update({"slot": slot.identifier, "was": was})
    elif op == "ASSIGN":
        reply.update(_slots_assign(act, params))
    elif op == "SEPARATE":
        if len(act.slots) < 2:
            raise MifOpError("action '%s' has %s slot - SEPARATE needs two or more. %s"
                             % (act.name, "one" if len(act.slots) == 1 else "no", _TAIL))
        made, rebound = [], []
        for s in list(act.slots):
            new, rb = _split_copy(act, {s.identifier}, "%s_%s" % (act.name, s.name_display))
            made.append(new.name)
            rebound.extend(rb)
        reply.update({"actions": made, "rebound": rebound})
    elif op == "MOVE_TO_NEW":
        reply.update(_slots_move(act, params))
    else:
        reply.update(_slots_merge(act, params))
    reply["slots"] = [slot_row(act, s) for s in act.slots]
    return reply


def _slots_new(act, params):
    name = take(params, "name", kind=str)
    if not name:
        raise MifOpError("NEW takes name - the slot's display name. %s" % _TAIL)
    kinds = bpy.types.ActionSlot.bl_rna.properties["target_id_type"].enum_items.keys()
    kind = take(params, "idType", default="OBJECT", kind=str).upper()
    if kind not in kinds or kind == "UNSPECIFIED":
        raise MifOpError("idType must be one of %s, got %r. %s"
                         % (", ".join(k for k in kinds if k != "UNSPECIFIED"), kind, _TAIL))
    slot = act.slots.new(id_type=kind, name=name)
    return {"slot": slot.identifier}


def _slots_assign(act, params):
    from .ops_anim import _shape_key_block
    obj = get_object(take(params, "object", kind=str))
    require_editable(obj, "animate")
    target = (take(params, "target", default="object", kind=str) or "object").lower()
    holders = {"object": obj, "data": obj.data, "shape_keys": _shape_key_block(obj)}
    if target not in holders:
        raise MifOpError("target must be object | data | shape_keys, got %r. %s" % (target, _TAIL))
    holder = holders[target]
    if holder is None:
        raise MifOpError("'%s' has no %s to animate. %s" % (obj.name, target.replace("_", " "), _TAIL))
    ref = take(params, "slot", kind=str)
    made = take_bool(params, "newSlot", default=False)
    if bool(ref) == bool(made):
        raise MifOpError("ASSIGN takes one of slot (an existing one) or newSlot true. %s" % _TAIL)
    id_type = holder.id_type
    slot = None
    if ref:
        slot = _find_slot(act, ref)
        if slot.target_id_type not in (id_type, "UNSPECIFIED"):
            raise MifOpError("slot %s is for %s data, and %s's %s is %s - Blender plays a slot only on its own kind. %s"
                             % (slot.identifier, slot.target_id_type, obj.name, target, id_type, _TAIL))
    name = take(params, "name", kind=str) or holder.name
    if made:
        slot = act.slots.new(id_type=id_type, name=name)
    if holder.animation_data is None:
        holder.animation_data_create()
    ad = holder.animation_data
    ad.action = act
    ad.action_slot = slot
    if ad.action_slot is None or ad.action_slot.identifier != slot.identifier:
        raise MifOpError("bound %s to %s's %s but it did not take. Do not trust this state."
                         % (slot.identifier, obj.name, target))
    return {"object": obj.name, "target": target, "slot": slot.identifier, "madeSlot": made,
            "sharedWith": sorted(u.name for u in slot.users() if u != holder)}


def _slots_move(act, params):
    refs = params.get("slots")
    if not isinstance(refs, list) or not refs or not all(isinstance(r, str) for r in refs):
        raise MifOpError("MOVE_TO_NEW takes slots - a list of slot identifiers or names. %s" % _TAIL)
    name = take(params, "newAction", kind=str)
    if not name:
        raise MifOpError("MOVE_TO_NEW takes newAction - the new action's name. %s" % _TAIL)
    keep = {_find_slot(act, r).identifier for r in refs}
    new, rebound = _split_copy(act, keep, name)
    for s in [s for s in act.slots if s.identifier in keep]:
        act.slots.remove(s)
    return {"newAction": new.name, "moved": sorted(keep), "rebound": rebound,
            "newSlots": [slot_row(new, s) for s in new.slots]}


def _slots_merge(act, params):
    names = params.get("fromActions")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise MifOpError("MERGE takes fromActions - a list of action names. %s" % _TAIL)
    sources = []
    for n in names:
        src = _get_action(n, "fromActions")
        if src == act:
            raise MifOpError("MERGE of '%s' into itself. %s" % (act.name, _TAIL))
        if src not in sources:
            sources.append(src)
    for src in sources:
        for s in src.slots:
            if s.target_id_type == "UNSPECIFIED":
                raise MifOpError("slot %s of '%s' has no data kind yet (UNSPECIFIED), so a copy could not be made - "
                                 "assign it first. %s" % (s.identifier, src.name, _TAIL))
    mapping, rebound = [], []
    for src in sources:
        binds = _bindings(src)
        for s in src.slots:
            t = act.slots.new(id_type=s.target_id_type, name=s.name_display)
            curves = _slot_curves(src, s)
            if curves:
                bag = _ensure_bag(act, t)
                for fc in curves:
                    _copy_curve(fc, bag)
            mapping.append({"from": src.name, "slot": s.identifier, "to": t.identifier, "curves": len(curves)})
            for idb, strip in binds:
                if _bound_slot(idb, strip) == s.identifier:
                    _bind(idb, strip, act, t)
                    rebound.append(_binding_label(idb, strip))
    return {"merged": mapping, "rebound": rebound}


def _slots_replace(act, params, slotted):
    new = _get_action(take(params, "newAction", kind=str), "newAction")
    if new == act:
        raise MifOpError("REPLACE of '%s' by itself. %s" % (act.name, _TAIL))
    binds = _bindings(act)
    if not binds:
        raise MifOpError("nothing plays '%s' - no active action and no NLA strip holds it. %s" % (act.name, _TAIL))
    plan = []
    for idb, strip in binds:
        slot = None
        if slotted and len(act.slots):
            ident = _bound_slot(idb, strip)
            slot = next((s for s in new.slots if s.identifier == ident), None)
            if slot is None:
                raise MifOpError("%s plays slot %s, and '%s' has no slot %s (its slots: %s). %s"
                                 % (_binding_label(idb, strip), ident, new.name, ident,
                                    ", ".join(s.identifier for s in new.slots) or "<none>", _TAIL))
        plan.append((idb, strip, slot))
    out = []
    for idb, strip, slot in plan:
        _bind(idb, strip, new, slot)
        out.append({"user": _binding_label(idb, strip), "slot": slot.identifier if slot is not None else None})
    return {"action": act.name, "op": "REPLACE", "newAction": new.name, "replaced": out,
            "oldActionUsers": int(act.users), "oldSurvivesSave": bool(act.users) or bool(act.use_fake_user)}


# --------------------------------------------------------------------------------------------- NLA strip edits

_STRIP_KEYS = ("object", "name", "track", "strip", "rename", "frameStart", "frameEnd", "actionStart", "actionEnd",
               "repeat", "scale", "blendIn", "blendOut", "autoBlend", "mute", "reverse", "syncLength", "slot")
_STRIP_NUM = {"frameStart": "frame_start_ui", "frameEnd": "frame_end_ui", "actionStart": "action_frame_start",
              "actionEnd": "action_frame_end", "scale": "scale", "repeat": "repeat", "blendIn": "blend_in",
              "blendOut": "blend_out"}
_STRIP_BOOL = {"autoBlend": "use_auto_blend", "mute": "mute", "reverse": "use_reverse"}


def _strip_row(s):
    slot = getattr(s, "action_slot", None)
    return {"name": s.name, "action": s.action.name if s.action else None,
            "slot": slot.identifier if slot is not None else None,
            "frameStart": round(s.frame_start, 4), "frameEnd": round(s.frame_end, 4),
            "actionStart": round(s.action_frame_start, 4), "actionEnd": round(s.action_frame_end, 4),
            "repeat": round(s.repeat, 4), "scale": round(s.scale, 4), "blendIn": round(s.blend_in, 4),
            "blendOut": round(s.blend_out, 4), "autoBlend": bool(s.use_auto_blend), "mute": bool(s.mute),
            "reverse": bool(s.use_reverse), "syncLength": bool(s.use_sync_length)}


def _strip_keyed_range(s, slot):
    curves = _slot_curves(s.action, slot) if slot is not None else None
    if curves is None:
        from .ops_anim import _action_curves
        curves = _action_curves(s.action)
    ranges = [fc.range() for fc in curves if len(fc.keyframe_points)]
    if not ranges:
        return None
    return min(r[0] for r in ranges), max(r[1] for r in ranges)


def op_set_nla_strip(params):
    """Edit an existing NLA strip: its frames, action range, repeat, scale, blends, mute, reverse, name and slot.

    THE GAP (spec, "Action slots on 4.4 and later"): add_nla_strip places a strip and set_nla_track edits tracks, and
    nothing could change a strip once it existed - nor say which slot it plays. A strip made by NlaStrips.new on an
    object whose slot is not named after it plays no slot and moves nothing (measured on 4.4.0 and 5.2.2).

    object, track, strip (name; default the track's only strip); rename; frameStart moves the strip keeping its length;
    frameEnd sets its length (not with repeat or scale); actionStart / actionEnd the part of the action it plays;
    repeat, scale, blendIn, blendOut, autoBlend, mute, reverse; slot (4.4+, identifier or name); syncLength true sets
    the action range to the keys of the strip's slot now and keeps it synced, false stops syncing. Reply before, after.
    """
    reject_unknown(params, _STRIP_KEYS, "set_nla_strip")
    obj = get_object(take(params, "object", "name", required=True, kind=str))
    require_editable(obj, "edit the NLA of")
    ad = obj.animation_data
    tname = take(params, "track", required=True, kind=str)
    track = ad.nla_tracks.get(tname) if ad is not None else None
    if track is None:
        raise MifOpError("'%s' has no NLA track '%s'. Its tracks: %s. %s"
                         % (obj.name, tname, ", ".join(t.name for t in ad.nla_tracks) if ad else "<none>", _TAIL))
    sname = take(params, "strip", kind=str)
    if sname:
        strip = track.strips.get(sname)
        if strip is None:
            raise MifOpError("track '%s' has no strip '%s'. Its strips: %s. %s"
                             % (track.name, sname, ", ".join(s.name for s in track.strips) or "<none>", _TAIL))
    elif len(track.strips) == 1:
        strip = track.strips[0]
    else:
        raise MifOpError("track '%s' has %d strips - name one with strip: %s. %s"
                         % (track.name, len(track.strips), ", ".join(s.name for s in track.strips), _TAIL))
    nums = {}
    for key in _STRIP_NUM:
        v = take_float(params, key)
        if v is not None:
            nums[key] = v
    bools = {k: take_bool(params, k) for k in _STRIP_BOOL if params.get(k) is not None}
    sync = take_bool(params, "syncLength") if params.get("syncLength") is not None else None
    rename = take(params, "rename", kind=str)
    ref = take(params, "slot", kind=str)
    if not (nums or bools or rename or ref or sync is not None):
        raise MifOpError("nothing to do - pass a frame, range, repeat, scale, blend, flag, rename, slot or syncLength. "
                         "%s" % _TAIL)
    if "frameEnd" in nums and ("repeat" in nums or "scale" in nums):
        raise MifOpError("frameEnd sets the strip's length, and so do repeat and scale - pass one of them. %s" % _TAIL)
    for key in ("repeat", "scale"):
        if key in nums and nums[key] <= 0:
            raise MifOpError("%s must be above 0, got %g. %s" % (key, nums[key], _TAIL))
    for key in ("blendIn", "blendOut"):
        if key in nums and nums[key] < 0:
            raise MifOpError("%s must be 0 or more, got %g. %s" % (key, nums[key], _TAIL))
    a0 = nums.get("actionStart", strip.action_frame_start)
    a1 = nums.get("actionEnd", strip.action_frame_end)
    if ("actionStart" in nums or "actionEnd" in nums) and a1 <= a0:
        raise MifOpError("actionEnd (%g) must be after actionStart (%g). %s" % (a1, a0, _TAIL))
    if "frameEnd" in nums and nums["frameEnd"] <= nums.get("frameStart", strip.frame_start):
        raise MifOpError("frameEnd (%g) must be after the strip's start. %s" % (nums["frameEnd"], _TAIL))
    slot = None
    if ref:
        if not _has_slots():
            raise _no_slots("slot")
        if strip.action is None:
            raise MifOpError("strip '%s' holds no action, so it has no slot to play. %s" % (strip.name, _TAIL))
        slot = _find_slot(strip.action, ref)
        if slot.target_id_type not in ("OBJECT", "UNSPECIFIED"):
            raise MifOpError("slot %s is for %s data and this strip is on an object. %s"
                             % (slot.identifier, slot.target_id_type, _TAIL))
    # A STRIP THAT PLAYED NO SLOT HAS NO KEYED RANGE: NlaStrips.new on an action with no fitting slot reads 0-1 and the
    # strip ends a frame after it starts (measured on 5.2.2). Given its first slot, it takes that slot's keys.
    first_slot = slot is not None and getattr(strip, "action_slot", None) is None
    span = None
    if sync or (first_slot and sync is None):
        if strip.action is None:
            raise MifOpError("strip '%s' holds no action to sync to. %s" % (strip.name, _TAIL))
        span = _strip_keyed_range(strip, slot or getattr(strip, "action_slot", None))
        if span is not None and span[1] <= span[0]:
            span = None
        if span is None and sync:
            raise MifOpError("the strip's %s has no keyed range to sync to. %s"
                             % ("slot" if _has_slots() else "action", _TAIL))
    before = _strip_row(strip)
    if slot is not None:
        strip.action_slot = slot
    if span is not None:
        # end first: a 0-1 range cannot take start 1 before its end moves; Blender moves frame_end with the range
        strip.action_frame_end = span[1]
        strip.action_frame_start = span[0]
        strip.action_frame_end = span[1]
    if sync is not None:
        strip.use_sync_length = sync
    for key in ("actionStart", "actionEnd", "scale", "repeat", "frameStart", "frameEnd", "blendIn", "blendOut"):
        if key in nums:
            setattr(strip, _STRIP_NUM[key], nums[key])
    for key, v in bools.items():
        setattr(strip, _STRIP_BOOL[key], v)
    if rename:
        strip.name = rename
    bpy.context.view_layer.update()
    return {"object": obj.name, "track": track.name, "before": before, "after": _strip_row(strip),
            "rangeFromSlot": bool(first_slot and span is not None and not sync)}


# --------------------------------------------------------------------------------------------- match world transform

_MW_KEYS = ("object", "name", "bone", "target", "targetBone", "frames", "frameStart", "frameEnd", "key", "channels",
            "relativeTo", "relativeToBone", "allowSimulation")
_MW_CHANNELS = ("LOC", "ROT", "SCALE")


def _mw_subject(obj_name, bone_name, key):
    obj = get_object(obj_name)
    if not bone_name:
        return obj, None
    if obj.type != "ARMATURE":
        raise MifOpError("%s names a bone, and '%s' is a %s, not an armature. %s" % (key, obj.name, obj.type, _TAIL))
    pb = obj.pose.bones.get(bone_name)
    if pb is None:
        raise MifOpError("armature '%s' has no bone '%s'. Its bones: %s. %s"
                         % (obj.name, bone_name, ", ".join(b.name for b in list(obj.pose.bones)[:30]), _TAIL))
    return obj, pb


def _mw_world(obj, pb):
    """The evaluated world matrix of an object, or of a pose bone through its armature's."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    if pb is None:
        return ev.matrix_world.copy()
    return ev.matrix_world @ ev.pose.bones[pb.name].matrix


def _mw_frames(params):
    raw = params.get("frames")
    lo, hi = params.get("frameStart"), params.get("frameEnd")
    if raw is not None and (lo is not None or hi is not None):
        raise MifOpError("pass frames or frameStart / frameEnd, not both. %s" % _TAIL)
    if raw is not None:
        if not isinstance(raw, list) or not raw:
            raise MifOpError("frames must be a non-empty list of whole frames. %s" % _TAIL)
        return sorted({finite_int(f, "frames") for f in raw})
    if lo is None and hi is None:
        return [bpy.context.scene.frame_current]
    if lo is None or hi is None:
        raise MifOpError("a range takes both frameStart and frameEnd. %s" % _TAIL)
    lo, hi = finite_int(lo, "frameStart"), finite_int(hi, "frameEnd")
    if hi < lo:
        raise MifOpError("frameEnd (%d) is before frameStart (%d). %s" % (hi, lo, _TAIL))
    if hi - lo > 10000:
        raise MifOpError("a range of %d frames - at most 10001. %s" % (hi - lo + 1, _TAIL))
    return list(range(lo, hi + 1))


def _mw_write(holder, basis, channels, frame, key):
    """Write basis's loc / rotation (in the holder's own mode, continuous with the last euler) / scale; key them."""
    loc, quat, scl = basis.decompose()
    paths = []
    if "LOC" in channels:
        holder.location = loc
        paths.append("location")
    if "ROT" in channels:
        mode = holder.rotation_mode
        if mode == "QUATERNION":
            if quat.dot(holder.rotation_quaternion) < 0:
                quat.negate()
            holder.rotation_quaternion = quat
            paths.append("rotation_quaternion")
        elif mode == "AXIS_ANGLE":
            axis, angle = quat.to_axis_angle()
            holder.rotation_axis_angle = (angle, axis[0], axis[1], axis[2])
            paths.append("rotation_axis_angle")
        else:
            holder.rotation_euler = quat.to_euler(mode, holder.rotation_euler)
            paths.append("rotation_euler")
    if "SCALE" in channels:
        holder.scale = scl
        paths.append("scale")
    if key:
        for p in paths:
            if isinstance(holder, bpy.types.PoseBone):
                holder.keyframe_insert(p, frame=frame, group=holder.name)
            else:
                holder.keyframe_insert(p, frame=frame)
    return paths


def op_match_world_transform(params):
    """Match an object's or a pose bone's world transform to another's at frames, keyed, and measure the error after.

    THE GAP (spec, "Match an object's or a bone's world transform to another's at frames"): evaluate_at_frame reads a
    world matrix and set_keyframe keys channels, so only an unparented object could be matched, one round trip a frame;
    a parented object or a pose bone needs its parent's matrix and the bone's rest, which no call computed. Blender's
    Copy Global Transform needs a clipboard and a window. This writes the matrix through Blender's own setters
    (Object.matrix_world, PoseBone.matrix), which apply the parent, the parent inverse and the rest.

    object (+ bone) is moved; target (+ targetBone) is matched; frames [..] | frameStart + frameEnd (default the current
    frame); key (default true; false matches one frame only); channels [LOC, ROT, SCALE] (default all) - the rotation
    is written in the holder's own rotation mode, eulers kept continuous. relativeTo (+ relativeToBone): carry the
    match on what that moves - each frame takes relativeTo(frame) x relativeTo(first)^-1 x target(first); with no target
    the moved thing rides relativeTo from where it stands at the first frame. Reply per frame: positionError (m),
    rotationErrorDeg, scaleError, read through the depsgraph after every write.
    """
    reject_unknown(params, _MW_KEYS, "match_world_transform")
    obj, pb = _mw_subject(take(params, "object", "name", required=True, kind=str), take(params, "bone", kind=str),
                          "bone")
    require_editable(obj, "match the transform of")
    tname = take(params, "target", kind=str)
    rname = take(params, "relativeTo", kind=str)
    if params.get("targetBone") and not tname:
        raise MifOpError("targetBone names a bone of target, and no target was given. %s" % _TAIL)
    if params.get("relativeToBone") and not rname:
        raise MifOpError("relativeToBone names a bone of relativeTo, and no relativeTo was given. %s" % _TAIL)
    if not tname and not rname:
        raise MifOpError("pass target (what to match), relativeTo (what to ride), or both. %s" % _TAIL)
    tgt = _mw_subject(tname, take(params, "targetBone", kind=str), "targetBone") if tname else None
    rel = _mw_subject(rname, take(params, "relativeToBone", kind=str), "relativeToBone") if rname else None
    me = (obj, pb.name if pb else None)
    for label, other in (("target", tgt), ("relativeTo", rel)):
        if other is not None and (other[0], other[1].name if other[1] else None) == me:
            raise MifOpError("%s is the thing being moved - it would chase itself. %s" % (label, _TAIL))
    chans = params.get("channels", list(_MW_CHANNELS))
    if not isinstance(chans, list) or not chans or any(str(c).upper() not in _MW_CHANNELS for c in chans):
        raise MifOpError("channels must be a list of %s, got %r. %s" % (", ".join(_MW_CHANNELS), chans, _TAIL))
    chans = {str(c).upper() for c in chans}
    key = take_bool(params, "key", default=True)
    frames = _mw_frames(params)
    if not key and len(frames) > 1:
        raise MifOpError("key false matches one frame and keeps no keys; %d frames would leave only the last. %s"
                         % (len(frames), _TAIL))
    sc = bpy.context.scene
    start = sc.frame_current
    sim = simulation_guard(params, frames + [start], "match_world_transform")
    holder = pb if pb is not None else obj
    first = None
    rows = []
    try:
        for f in frames:
            sc.frame_set(f)
            want = _mw_world(*tgt) if tgt else None
            if rel is not None:
                rmat = _mw_world(*rel)
                if first is None:
                    first = rmat.inverted_safe() @ (want if want is not None else _mw_world(obj, pb))
                want = rmat @ first
            if pb is not None:
                pb.matrix = obj.matrix_world.inverted_safe() @ want
                bpy.context.view_layer.update()
                basis = pb.matrix_basis.copy()
            else:
                saved = obj.matrix_basis.copy()
                obj.matrix_world = want
                basis = obj.matrix_basis.copy()
                obj.matrix_basis = saved
            _mw_write(holder, basis, chans, f, key)
            rows.append([f, want])
        out = []
        for f, want in rows:
            if key:
                sc.frame_set(f)
            else:
                bpy.context.view_layer.update()
            got = _mw_world(obj, pb)
            gl, gq, gs = got.decompose()
            wl, wq, ws = want.decompose()
            ang = gq.rotation_difference(wq).angle
            out.append({"frame": f, "positionError": round((gl - wl).length, 7),
                        "rotationErrorDeg": round(math.degrees(min(ang, 2 * math.pi - ang)), 6),
                        "scaleError": round((gs - ws).length, 7)})
    finally:
        if sc.frame_current != start:
            sc.frame_set(start)
    reply = {"object": obj.name, "bone": pb.name if pb else None, "target": tname,
             "targetBone": tgt[1].name if tgt and tgt[1] else None, "relativeTo": rname,
             "channels": [c for c in _MW_CHANNELS if c in chans], "keyed": key, "frameCount": len(frames),
             "frames": out[:200], "framesTruncated": len(out) > 200,
             "maxPositionError": max(r["positionError"] for r in out),
             "maxRotationErrorDeg": max(r["rotationErrorDeg"] for r in out),
             "maxScaleError": max(r["scaleError"] for r in out)}
    if chans != set(_MW_CHANNELS):
        reply["errorNote"] = "only %s were written, so the other channels' error is the distance left, not a defect." \
            % ", ".join(sorted(chans))
    reply.update(sim)
    return reply


# ---------------------------------------------------------------------------------------------------------- pose_seams
# FOUND 2026-11-19: store creature clips keyed on every frame with linear tangents still jumped the whole
# pose in one frame (18 of 31 clips, an upper arm 162 degrees) and flipped a hand's twist 164 degrees in one frame. A
# person scrubbing saw the pop; no clipping measure did. What worked: per frame, each bone's parent-relative rotation
# against the frame before, and its twist about bone Y against the frame before; at a seam each jumping bone continues
# from the frame before and the offset fades out over 6 frames (twice for seams close together). That brought the
# largest per-frame turn to 44 degrees and the largest twist jump to 21.
#
# THE FADE is done on the bone's basis (its own channels): parent-relative = Rrel @ B with Rrel fixed, so holding the
# parent-relative rotation at the frame before is holding B there. With O = B(s-1) @ B(s)^-1 at a seam frame s, every
# frame f of s .. s+N-1 becomes slerp(1, O, 1 - (f - s) / N) @ B(f): frame s repeats s-1's rotation and the offset is
# gone by s+N. Seams are taken in frame order on the values already faded, so a second seam inside a window composes.

_PS_KEYS = ("object", "name", "action", "actions", "frameRange", "bones", "namePrefix", "all", "turnDeg", "twistDeg",
            "fadeBonesAboveDeg", "fix", "fadeFrames", "maxPasses", "maxRows")


def _ps_scan(obj, pbs, start, end, turn_lim, twist_lim):
    """Per frame and bone: parent-relative quaternion, the turn and twist jump against the frame before; and each
    bone's basis rotation, for the fade."""
    scene = bpy.context.scene
    rel, basis = {}, {}
    for f in range(start, end + 1):
        scene.frame_set(f)
        mats = {pb.name: pb.matrix.copy() for pb in obj.pose.bones}
        row, brow = {}, {}
        for pb in pbs:
            m = mats[pb.name]
            r = (mats[pb.parent.name].inverted_safe() @ m) if pb.parent else m
            row[pb.name] = r.to_quaternion()
            brow[pb.name] = pb.matrix_basis.to_quaternion()
        rel[f], basis[f] = row, brow
    jumps = {}
    worst_turn, worst_twist = (0.0, None, None), (0.0, None, None)
    for f in range(start + 1, end + 1):
        row = {}
        for pb in pbs:
            a, b = rel[f - 1][pb.name], rel[f][pb.name]
            d = a.rotation_difference(b).angle
            turn = math.degrees(min(d, 2 * math.pi - d))
            tw = lambda q: math.degrees(2.0 * math.atan2(q.y, q.w))  # noqa: E731
            twist = (tw(b) - tw(a) + 180.0) % 360.0 - 180.0
            row[pb.name] = (turn, twist)
            if turn > worst_turn[0]:
                worst_turn = (turn, f, pb.name)
            if abs(twist) > abs(worst_twist[0]):
                worst_twist = (twist, f, pb.name)
        jumps[f] = row
    seams, flips = [], []
    for f in range(start + 1, end + 1):
        over = sorted(((n, t, w) for n, (t, w) in jumps[f].items() if t > turn_lim), key=lambda x: -x[1])
        if over:
            seams.append({"frame": f, "bones": [{"bone": n, "turnDeg": round(t, 3), "twistJumpDeg": round(w, 3)}
                                                for n, t, w in over]})
        for n, (t, w) in jumps[f].items():
            if abs(w) > twist_lim:
                flips.append({"frame": f, "bone": n, "twistJumpDeg": round(w, 3), "turnDeg": round(t, 3)})
    stats = {"maxTurnDeg": round(worst_turn[0], 3), "maxTurnAt": {"frame": worst_turn[1], "bone": worst_turn[2]},
             "maxTwistJumpDeg": round(abs(worst_twist[0]), 3),
             "maxTwistJumpAt": {"frame": worst_twist[1], "bone": worst_twist[2]}}
    return {"jumps": jumps, "basis": basis, "seams": seams, "flips": flips, "stats": stats}


def _ps_write(pb, quat, frame, prev):
    """Write a basis rotation into the bone's own channels at frame and key it; returns what the next frame continues."""
    from mathutils import Quaternion   # noqa: PLC0415,F401
    mode = pb.rotation_mode
    if mode == "QUATERNION":
        q = quat.copy()
        if prev is not None:
            q.make_compatible(prev)
        pb.rotation_quaternion = q
        out = q
    elif mode == "AXIS_ANGLE":
        axis, ang = quat.to_axis_angle()
        pb.rotation_axis_angle = (ang, axis[0], axis[1], axis[2])
        out = None
    else:
        e = quat.to_euler(mode, prev) if prev is not None else quat.to_euler(mode)
        pb.rotation_euler = e
        out = e
    pb.keyframe_insert(rotation_path(pb), frame=frame, group=pb.name)
    return out


def _ps_fix(obj, pbs, scan, start, end, fade_above, n_fade):
    """Fade every seam's jumping bones; returns the (frame, bone) seams faded."""
    from mathutils import Quaternion   # noqa: PLC0415
    basis = {f: dict(row) for f, row in scan["basis"].items()}
    done, touched = [], {}
    by_name = {pb.name: pb for pb in pbs}
    seam_set = {s["frame"] for s in scan["seams"]}
    seam_frames = sorted(seam_set | {fl["frame"] for fl in scan["flips"]})
    for s in seam_frames:
        row = scan["jumps"][s]
        flipped = {fl["bone"] for fl in scan["flips"] if fl["frame"] == s}
        # a whole-pose seam takes every bone turning past fade_above with it; a flip alone only the flipped bone
        names = [n for n, (t, _w) in row.items() if (s in seam_set and t > fade_above) or n in flipped]
        for n in names:
            o = basis[s - 1][n] @ basis[s][n].inverted()
            if o.w < 0:
                o.negate()
            for f in range(s, min(end, s + n_fade - 1) + 1):
                w = 1.0 - (f - s) / float(n_fade)
                basis[f][n] = Quaternion().slerp(o, w) @ basis[f][n]
                touched.setdefault(n, set()).add(f)
            done.append({"frame": s, "bone": n, "offsetDeg": round(math.degrees(o.angle), 3)})
    scene = bpy.context.scene
    for n, frames in touched.items():
        pb = by_name[n]
        prev = None
        first = min(frames)
        if first - 1 >= start:
            scene.frame_set(first - 1)
            prev = (pb.rotation_quaternion.copy() if pb.rotation_mode == "QUATERNION" else
                    None if pb.rotation_mode == "AXIS_ANGLE" else pb.rotation_euler.copy())
        for f in sorted(frames):
            prev = _ps_write(pb, basis[f][n], f, prev)
    return done


def op_pose_seams(params):
    """Scan an armature's actions for one-frame pose seams and twist flips, and with fix fade them out.

    params:
      object (str)              the armature
      action (str) | actions [] the actions to scan (default the assigned one); each is assigned in turn and the one
                                assigned before is put back
      frameRange [s, e]         default each action's own range
      bones [..] | namePrefix | all   the bones scanned (default all)
      turnDeg (float)           a seam: a bone whose parent-relative rotation turns more than this in one frame
                                (default 60; fast authored motion measured up to 44)
      twistDeg (float)          a flip: a bone whose twist about its own Y relative to its parent jumps more than this
                                in one frame (default 90) - judged frame to frame, never by the absolute twist, which
                                goes unstable near a 180-degree fold
      fix (bool)                fade them (default false: read only)
      fadeBonesAboveDeg (float) at a seam frame, every bone turning more than this there is faded with the one that
                                crossed turnDeg, so the rest of a whole-pose jump goes too (default 15)
      fadeFrames (int)          the frames the offset fades over (default 6)
      maxPasses (int)           scan-and-fade passes while seams remain (default 3)
      maxRows (int)             seams and flips listed per action (default 200)
    Rotation only: a location jump is not looked at. A fixed frame is keyed on the bone's rotation channel.
    Reply per action: frames, seams [{frame, bones [{bone, turnDeg, twistJumpDeg}]}], flips [{frame, bone,
    twistJumpDeg, turnDeg}], before {maxTurnDeg, maxTwistJumpDeg and where}, and with fix: faded [{frame, bone,
    offsetDeg}], passes, after (the same measures on the faded action) and seamsLeft / flipsLeft.
    """
    reject_unknown(params, _PS_KEYS, "pose_seams")
    obj = _armature(params, "changed")
    fix = take_bool(params, "fix", default=False)
    if fix:
        require_editable(obj, "animate")
    if params.get("action") is not None and params.get("actions") is not None:
        raise MifOpError("pass action or actions, not both. %s" % _TAIL)
    names = params.get("actions")
    if names is None:
        one = take(params, "action", default=None, kind=str)
        if one is not None:
            names = [one]
        else:
            cur = obj.animation_data.action if obj.animation_data else None
            if cur is None:
                raise MifOpError("'%s' has no action assigned - name one with action or actions. %s" % (obj.name, _TAIL))
            names = [cur.name]
    if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
        raise MifOpError("actions is a non-empty list of action names. %s" % _TAIL)
    missing = [n for n in names if bpy.data.actions.get(n) is None]
    if missing:
        raise MifOpError("no action named %s. %s" % (", ".join(missing[:10]), _TAIL))
    pbs = pick_bones(obj, params, "changed", default_all=True)
    if not pbs:
        raise MifOpError("'%s' has no bones to scan. %s" % (obj.name, _TAIL))
    turn_lim = take_float(params, "turnDeg", default=60.0)
    twist_lim = take_float(params, "twistDeg", default=90.0)
    fade_above = take_float(params, "fadeBonesAboveDeg", default=15.0)
    for k, v in (("turnDeg", turn_lim), ("twistDeg", twist_lim), ("fadeBonesAboveDeg", fade_above)):
        if not 0.0 < v <= 180.0:
            raise MifOpError("%s is a turn in degrees, over 0 and at most 180, got %r. %s" % (k, v, _TAIL))
    n_fade = finite_int(params.get("fadeFrames", 6), "fadeFrames")
    passes = finite_int(params.get("maxPasses", 3), "maxPasses")
    max_rows = finite_int(params.get("maxRows", 200), "maxRows")
    if n_fade < 1 or passes < 1 or max_rows < 1:
        raise MifOpError("fadeFrames, maxPasses and maxRows are whole numbers of at least 1. %s" % _TAIL)
    fr = params.get("frameRange")
    if fr is not None and (not isinstance(fr, list) or len(fr) != 2
                           or any(isinstance(v, bool) or not isinstance(v, int) for v in fr) or fr[1] <= fr[0]):
        raise MifOpError("frameRange is [start, end] whole frames, end after start, got %r. %s" % (fr, _TAIL))
    scene = bpy.context.scene
    ad = obj.animation_data or obj.animation_data_create()
    saved = (ad.action, scene.frame_current)
    rows = []
    try:
        for name in names:
            act = bpy.data.actions[name]
            start, end = fr if fr is not None else (int(round(v)) for v in act.frame_range)
            if end <= start:
                raise MifOpError("action '%s' spans frame %d only - there is no frame before another to compare. %s"
                                 % (name, start, _TAIL if not rows else "Actions before it were already done."))
            _assign_action(obj, act)
            scan = _ps_scan(obj, pbs, start, end, turn_lim, twist_lim)
            row = {"action": name, "frameStart": start, "frameEnd": end, "frames": end - start + 1,
                   "bonesScanned": len(pbs), "seams": scan["seams"][:max_rows], "seamCount": len(scan["seams"]),
                   "flips": scan["flips"][:max_rows], "flipCount": len(scan["flips"]), "before": scan["stats"]}
            if fix:
                faded, n = [], 0
                while (scan["seams"] or scan["flips"]) and n < passes:
                    faded += _ps_fix(obj, pbs, scan, start, end, fade_above, n_fade)
                    n += 1
                    scan = _ps_scan(obj, pbs, start, end, turn_lim, twist_lim)
                row.update(faded=faded[:max_rows * 4], fadedCount=len(faded), passes=n, after=scan["stats"],
                           seamsLeft=scan["seams"][:max_rows], flipsLeft=scan["flips"][:max_rows])
            rows.append(row)
    finally:
        _assign_action(obj, saved[0])
        scene.frame_set(saved[1])
    out = {"object": obj.name, "turnDeg": turn_lim, "twistDeg": twist_lim, "fixed": fix, "actions": rows,
           "seamCount": sum(r["seamCount"] for r in rows), "flipCount": sum(r["flipCount"] for r in rows),
           "maxTurnDeg": max(r["before"]["maxTurnDeg"] for r in rows),
           "maxTwistJumpDeg": max(r["before"]["maxTwistJumpDeg"] for r in rows)}
    if fix:
        out["after"] = {"maxTurnDeg": max(r["after"]["maxTurnDeg"] for r in rows),
                        "maxTwistJumpDeg": max(r["after"]["maxTwistJumpDeg"] for r in rows),
                        "seamsLeft": sum(len(r["seamsLeft"]) for r in rows),
                        "flipsLeft": sum(len(r["flipsLeft"]) for r in rows)}
    return out



# ----------------------------------------------------------------------------------------------------- self_clipping
# FOUND 2026-11-19: a review flagged claws through a creature's chest; measured, 16-31 of 31 clips per body went past
# 25 cm, up to 11.5 m. Physics capsules are too coarse a proxy (they flagged Idle), and a nearest-face or ray-parity test
# flips at random on open, overlapping surfaces such as scale plates. What worked: a signed-distance grid of the BODY
# region at the rest pose, inside decided by the generalized winding number (which stays well defined on an open or
# self-overlapping surface), and each posed LIMB vertex mapped into the rest frame of its nearest body bone and sampled
# there; the depth less the vertex's own rest depth is the clipping.
#
# THE WINDING NUMBER: w(p) = sum over triangles of the signed solid angle / 4 pi (van Oosterom-Strackee), 1 inside a
# closed outward surface, 0 outside, 0.5 a half-space at an open edge. Exact near each cluster of triangles; far from a
# cluster (over twice its radius) its dipole, sum(area vector) . (c - p) / |c - p|^3 - the first-order fast winding
# number, which keeps a 40^3 grid over tens of thousands of triangles to seconds.

_SC_KEYS = ("object", "armature", "limbBones", "bodyBones", "action", "actions", "frameRange", "frameStep",
            "tolerance", "gridResolution", "maxRows")


def _sc_wn_exact(p, tri):
    import numpy as np   # noqa: PLC0415
    a = tri[None, :, 0, :] - p[:, None, :]
    b = tri[None, :, 1, :] - p[:, None, :]
    c = tri[None, :, 2, :] - p[:, None, :]
    la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
    det = np.einsum("ijk,ijk->ij", a, np.cross(b, c))
    den = (la * lb * lc + np.einsum("ijk,ijk->ij", a, b) * lc + np.einsum("ijk,ijk->ij", b, c) * la
           + np.einsum("ijk,ijk->ij", c, a) * lb)
    return (2.0 * np.arctan2(det, den)).sum(axis=1) / (4.0 * math.pi)


def _sc_winding(points, tris, bins=8):
    """Generalized winding number of `tris` (T, 3, 3) at `points` (P, 3)."""
    import numpy as np   # noqa: PLC0415
    cent = tris.mean(axis=1)
    lo, hi = cent.min(axis=0), cent.max(axis=0)
    cell = np.maximum((hi - lo) / bins, 1e-9)
    key = np.clip(((cent - lo) / cell).astype(int), 0, bins - 1)
    key = key[:, 0] * bins * bins + key[:, 1] * bins + key[:, 2]
    area_vec = 0.5 * np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    out = np.zeros(len(points))
    for k in np.unique(key):
        idx = np.nonzero(key == k)[0]
        t = tris[idx]
        c = cent[idx].mean(axis=0)
        r = np.linalg.norm(t.reshape(-1, 3) - c, axis=1).max()
        d = points - c
        dist = np.linalg.norm(d, axis=1)
        near = dist < 2.0 * r + 1e-9
        far = ~near
        if far.any():
            nvec = area_vec[idx].sum(axis=0)
            out[far] += (-(d[far] @ nvec)) / (4.0 * math.pi * dist[far] ** 3)
        ni = np.nonzero(near)[0]
        chunk = max(16, 1500000 // max(1, len(t)))      # bounds the (points x triangles) arrays to ~1.5M pairs
        for s in range(0, len(ni), chunk):
            sl = ni[s:s + chunk]
            out[sl] += _sc_wn_exact(points[sl], t)
    return out


def _sc_sample(grid, lo, cell, pts):
    """Trilinear sample of grid (nx, ny, nz) at pts (N, 3); outside the grid -> -inf (outside the body)."""
    import numpy as np   # noqa: PLC0415
    n = np.array(grid.shape)
    u = (pts - lo) / cell
    inside = np.all((u >= 0) & (u <= n - 1), axis=1)
    u = np.clip(u, 0, n - 1 - 1e-6)
    i0 = np.floor(u).astype(int)
    f = u - i0
    out = np.zeros(len(pts))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                     * (f[:, 2] if dz else 1 - f[:, 2]))
                out += w * grid[np.minimum(i0[:, 0] + dx, n[0] - 1), np.minimum(i0[:, 1] + dy, n[1] - 1),
                                np.minimum(i0[:, 2] + dz, n[2] - 1)]
    out[~inside] = -np.inf
    return out


def _sc_dominant(obj, n_verts, deform):
    """Each vertex's bone with the largest weight (None where it has none)."""
    names = {g.index: g.name for g in obj.vertex_groups if g.name in deform}
    dom = [None] * n_verts
    for v in obj.data.vertices:
        best, bw = None, 0.0
        for ge in v.groups:
            nm = names.get(ge.group)
            if nm is not None and ge.weight > bw:
                best, bw = nm, ge.weight
        dom[v.index] = best
    return dom


def op_self_clipping(params):
    """Measure how far a skinned mesh's limbs go into its own body, per clip and frame.

    params:
      object (str)          the skinned mesh (its armature modifier's armature, or armature)
      limbBones [..]        the bones whose vertices are tested - each with every bone below it (an upper arm takes the
                            forearm, hand and claws): hands, claws, forearms, a tail
      bodyBones [..]        the body region (torso, head, legs) - exactly these bones; default every deforming bone not
                            under limbBones
      action | actions      the clips (default the armature's assigned one); each is assigned in turn and put back
      frameRange [s, e]     default each clip's own; frameStep (default 1)
      tolerance (float)     clipping past this is counted (default 0.5% of the body's rest diagonal)
      gridResolution (int)  signed-distance cells along the body's longest side (default 40, 12-128)
      maxRows (int)         frames listed per clip (default 200)
    THE MEASURE: a signed-distance grid of the body region's triangles at the rest pose (inside by the generalized
    winding number, so open or overlapping plates do not flip it); per frame each limb vertex, posed, is taken into the
    rest frame of the body bone whose posed segment is nearest and sampled there; clipping = its depth less its own rest
    depth (each floored at 0). Assumes the mesh's rest shape is bound at the armature's rest pose.
    Reply per clip: deepest {clipping, vertex, frame, limbBone, bodyBone, worldPosition}, framesPastTolerance (and their
    count), perFrameMax over the scanned frames' worst, and clear: for the deepest vertex, the swing at each joint of
    its limb chain (up to three, the vertex's own bone first) that would lift it clear, estimated from the depth
    gradient - {bone, swingDeg, axisWorld}. Refused: no limb or body vertices, a mesh whose modifiers change its
    vertex count, a clip with no frames.
    """
    import numpy as np   # noqa: PLC0415
    from mathutils import Vector   # noqa: PLC0415
    from mathutils.bvhtree import BVHTree   # noqa: PLC0415
    reject_unknown(params, _SC_KEYS, "self_clipping")
    tail = "Nothing was measured."
    obj = get_object(take(params, "object", required=True, kind=str))
    if obj.type != "MESH":
        raise MifOpError("'%s' is a %s - self_clipping measures a skinned MESH. %s" % (obj.name, obj.type, tail))
    arm_name = take(params, "armature", default=None, kind=str)
    if arm_name is None:
        mod = next((m for m in obj.modifiers if m.type == "ARMATURE" and m.object is not None), None)
        if mod is None:
            raise MifOpError("'%s' has no armature modifier with an armature - name one with armature. %s"
                             % (obj.name, tail))
        arm = mod.object
    else:
        arm = get_object(arm_name)
    if arm.type != "ARMATURE":
        raise MifOpError("'%s' is not an armature. %s" % (arm.name, tail))
    bones = arm.data.bones
    limbs = params.get("limbBones")
    if not isinstance(limbs, list) or not limbs or any(not isinstance(b, str) for b in limbs):
        raise MifOpError("limbBones is a non-empty list of bone names (hands, forearms, a tail). %s" % tail)
    missing = [b for b in limbs if bones.get(b) is None]
    if missing:
        raise MifOpError("'%s' has no bone %s. %s" % (arm.name, ", ".join(missing[:10]), tail))
    limb_set = set()
    for b in limbs:
        limb_set.add(b)
        limb_set.update(c.name for c in bones[b].children_recursive)
    deform = {b.name for b in bones if b.use_deform}
    body_raw = params.get("bodyBones")
    if body_raw is None:
        body_set = deform - limb_set
    else:
        if not isinstance(body_raw, list) or not body_raw or any(not isinstance(b, str) for b in body_raw):
            raise MifOpError("bodyBones is a non-empty list of bone names. %s" % tail)
        missing = [b for b in body_raw if bones.get(b) is None]
        if missing:
            raise MifOpError("'%s' has no bone %s. %s" % (arm.name, ", ".join(missing[:10]), tail))
        both = sorted(set(body_raw) & limb_set)
        if both:
            raise MifOpError("%s %s both body and limb (under limbBones). %s"
                             % (", ".join(both[:10]), "is" if len(both) == 1 else "are", tail))
        body_set = set(body_raw)
    names = params.get("actions")
    if names is None:
        one = take(params, "action", default=None, kind=str)
        cur = arm.animation_data.action if arm.animation_data else None
        names = [one] if one else ([cur.name] if cur else None)
        if names is None:
            raise MifOpError("'%s' has no action assigned - name the clips with action or actions. %s" % (arm.name, tail))
    if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
        raise MifOpError("actions is a non-empty list of action names. %s" % tail)
    missing = [n for n in names if bpy.data.actions.get(n) is None]
    if missing:
        raise MifOpError("no action named %s. %s" % (", ".join(missing[:10]), tail))
    step = finite_int(params.get("frameStep", 1), "frameStep")
    res = finite_int(params.get("gridResolution", 40), "gridResolution")
    max_rows = finite_int(params.get("maxRows", 200), "maxRows")
    if step < 1 or not 12 <= res <= 128 or max_rows < 1:
        raise MifOpError("frameStep >= 1, gridResolution 12-128, maxRows >= 1. %s" % tail)
    fr = params.get("frameRange")
    if fr is not None and (not isinstance(fr, list) or len(fr) != 2
                           or any(isinstance(v, bool) or not isinstance(v, int) for v in fr) or fr[1] < fr[0]):
        raise MifOpError("frameRange is [start, end] whole frames, got %r. %s" % (fr, tail))

    # THE REST BODY, in armature space
    me = obj.data
    nv = len(me.vertices)
    to_arm = arm.matrix_world.inverted() @ obj.matrix_world
    rest = np.empty(nv * 3)
    me.vertices.foreach_get("co", rest)
    rest = rest.reshape(-1, 3)
    m3 = np.array(to_arm.to_3x3())
    tr = np.array(to_arm.translation)
    rest = rest @ m3.T + tr
    dom = _sc_dominant(obj, nv, deform)
    is_body = np.array([d in body_set for d in dom])
    limb_idx = np.array([i for i, d in enumerate(dom) if d in limb_set], dtype=int)
    if not len(limb_idx):
        raise MifOpError("no vertex of '%s' is weighted mostly to %s or the bones under them. %s"
                         % (obj.name, ", ".join(limbs[:6]), tail))
    me.calc_loop_triangles()
    tri_v = np.empty(len(me.loop_triangles) * 3, dtype=int)
    me.loop_triangles.foreach_get("vertices", tri_v)
    tri_v = tri_v.reshape(-1, 3)
    body_tris = tri_v[is_body[tri_v].sum(axis=1) >= 2]
    if not len(body_tris):
        raise MifOpError("no face of '%s' belongs to the body region (%d bones) - nothing to clip into. %s"
                         % (obj.name, len(body_set), tail))
    tris = rest[body_tris]
    blo, bhi = tris.reshape(-1, 3).min(axis=0), tris.reshape(-1, 3).max(axis=0)
    diag = float(np.linalg.norm(bhi - blo))
    tol = params.get("tolerance")
    tol = 0.005 * diag if tol is None else finite_float(tol, "tolerance")
    if tol < 0:
        raise MifOpError("tolerance is a distance of 0 or more. %s" % tail)
    cell = float((bhi - blo).max()) / (res - 1)
    pad = 2 * cell
    lo = blo - pad
    dims = np.ceil((bhi + pad - lo) / cell).astype(int) + 1
    gx, gy, gz = (lo[i] + cell * np.arange(dims[i]) for i in range(3))
    gpts = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
    wn = _sc_winding(gpts, tris)
    bvh = BVHTree.FromPolygons([Vector(p) for p in rest], [tuple(int(i) for i in t) for t in body_tris])
    dist = np.array([(bvh.find_nearest(Vector(p))[3] or 0.0) for p in gpts])
    grid = np.where(wn > 0.5, dist, -dist).reshape(tuple(dims))   # depth: positive inside
    rest_depth = np.maximum(_sc_sample(grid, lo, cell, rest[limb_idx]), 0.0)

    body_bones = sorted(body_set & set(b.name for b in bones))
    rest_inv = {b: np.array(bones[b].matrix_local) for b in body_bones}
    limb_dom = [dom[i] for i in limb_idx]
    scene = bpy.context.scene
    ad = arm.animation_data or arm.animation_data_create()
    saved = (ad.action, scene.frame_current)
    clips = []
    try:
        for name in names:
            act = bpy.data.actions[name]
            start, end = fr if fr is not None else (int(round(v)) for v in act.frame_range)
            frames = list(range(start, end + 1, step))
            if not frames:
                raise MifOpError("clip '%s' has no frames to measure. %s" % (name, tail))
            _assign_action(arm, act)
            per = []
            deepest = None
            for f in frames:
                scene.frame_set(f)
                dg = bpy.context.evaluated_depsgraph_get()
                ev = obj.evaluated_get(dg)
                em = ev.to_mesh()
                try:
                    if len(em.vertices) != nv:
                        raise MifOpError("'%s' evaluates to %d vertices against %d in its mesh - a modifier other "
                                         "than the armature changes the count; turn it off for the measure. %s"
                                         % (obj.name, len(em.vertices), nv, tail))
                    co = np.empty(nv * 3)
                    em.vertices.foreach_get("co", co)
                finally:
                    ev.to_mesh_clear()
                posed = co.reshape(-1, 3)[limb_idx] @ m3.T + tr
                # the nearest posed body bone segment
                heads = np.array([list(arm.pose.bones[b].head) for b in body_bones])
                tails = np.array([list(arm.pose.bones[b].tail) for b in body_bones])
                seg = tails - heads
                t = np.clip(np.einsum("nbk,bk->nb", posed[:, None, :] - heads[None], seg)
                            / np.maximum((seg * seg).sum(axis=1), 1e-12), 0, 1)
                near = np.linalg.norm(posed[:, None, :] - (heads[None] + t[..., None] * seg[None]), axis=2).argmin(1)
                mapped = np.empty_like(posed)
                for bi, b in enumerate(body_bones):
                    sel = near == bi
                    if not sel.any():
                        continue
                    pm = np.array(arm.pose.bones[b].matrix)
                    m = rest_inv[b] @ np.linalg.inv(pm)
                    mapped[sel] = posed[sel] @ m[:3, :3].T + m[:3, 3]
                depth = np.maximum(_sc_sample(grid, lo, cell, mapped), 0.0)
                clip = np.maximum(depth - rest_depth, 0.0)
                k = int(clip.argmax())
                per.append((f, float(clip[k])))
                if deepest is None or clip[k] > deepest["clipping"]:
                    vi = int(limb_idx[k])
                    wpos = arm.matrix_world @ Vector(posed[k])
                    deepest = {"clipping": round(float(clip[k]), 5), "vertex": vi, "frame": f,
                               "limbBone": limb_dom[k], "bodyBone": body_bones[int(near[k])],
                               "worldPosition": [round(v, 5) for v in wpos],
                               "_mapped": mapped[k].copy(), "_posed": posed[k].copy(), "_body": body_bones[int(near[k])]}
            if deepest is not None and deepest["clipping"] > 0:
                deepest["clear"] = _sc_clear(arm, deepest, grid, lo, cell, limb_set, step=cell)
            if deepest is not None:
                for k in ("_mapped", "_posed", "_body"):
                    deepest.pop(k, None)
            past = [f for f, c in per if c > tol]
            clips.append({"action": name, "frameStart": frames[0], "frameEnd": frames[-1], "framesMeasured": len(frames),
                          "deepest": deepest, "framesPastTolerance": past[:max_rows], "framesPastCount": len(past),
                          "perFrameMax": [{"frame": f, "clipping": round(c, 5)} for f, c in per][:max_rows]})
    finally:
        _assign_action(arm, saved[0])
        scene.frame_set(saved[1])
    return {"object": obj.name, "armature": arm.name, "tolerance": round(tol, 6), "bodyDiagonal": round(diag, 5),
            "gridCell": round(cell, 6), "gridDims": [int(d) for d in dims], "bodyTriangles": int(len(body_tris)),
            "limbVertices": int(len(limb_idx)), "bodyBones": len(body_bones), "limbBones": sorted(limb_set)[:200],
            "clips": clips, "clipsPastTolerance": sum(1 for c in clips if c["framesPastCount"]),
            "worst": max((dict(c["deepest"], action=c["action"]) for c in clips if c["deepest"]),
                         key=lambda d: d["clipping"], default=None)}


def _sc_clear(arm, deepest, grid, lo, cell, limb_set, step):
    """The swing at each joint of the deepest vertex's limb chain that would lift it clear, from the depth gradient:
    the outward direction in the body bone's rest frame taken to the pose, and the turn about each joint's head that
    moves the vertex that far along it (first order)."""
    import numpy as np   # noqa: PLC0415
    from mathutils import Vector   # noqa: PLC0415
    q = deepest["_mapped"]
    g = np.zeros(3)
    for i in range(3):
        e = np.zeros(3)
        e[i] = step
        a = _sc_sample(grid, lo, cell, np.array([q + e, q - e]))
        a = np.where(np.isfinite(a), a, -1e9)
        g[i] = (a[0] - a[1]) / (2 * step)
    if np.linalg.norm(g) < 1e-12:
        return []
    out_rest = Vector(-g / np.linalg.norm(g))         # depth falls fastest this way: out of the body
    b = deepest["_body"]
    pm = arm.pose.bones[b].matrix
    rm = arm.data.bones[b].matrix_local
    out_dir = ((pm @ rm.inverted()).to_3x3() @ out_rest).normalized()
    v = Vector(deepest["_posed"])
    need = deepest["clipping"]
    rows = []
    pb = arm.pose.bones.get(deepest["limbBone"])
    while pb is not None and pb.name in limb_set and len(rows) < 3:
        r = v - pb.head
        axis = r.cross(out_dir)
        if axis.length > 1e-9 and r.length > 1e-9:
            perp = axis.length / out_dir.length     # |r| sin(angle between r and the way out)
            ang = math.degrees(math.asin(min(1.0, need / perp))) if perp > 1e-9 else None
            aw = (arm.matrix_world.to_3x3() @ axis).normalized()
            rows.append({"bone": pb.name, "swingDeg": None if ang is None else round(ang, 3),
                         "axisWorld": [round(x, 5) for x in aw],
                         "note": None if need <= perp else "the vertex is nearer this joint than it is deep - this "
                                                           "joint alone cannot clear it"})
        pb = pb.parent
    return rows


def _assign_action(obj, act):
    ad = obj.animation_data or obj.animation_data_create()
    ad.action = act
    if act is not None and "action_slot" in ad.bl_rna.properties and ad.action_slot is None:
        slot = next((s for s in act.slots if s.target_id_type == "OBJECT"), None)
        if slot is not None:
            ad.action_slot = slot


OPS = {
    "motion_paths": op_motion_paths,
    "spread_pose": op_spread_pose,
    "filter_keys": op_filter_keys,
    "edit_action_slots": op_edit_action_slots,
    "set_nla_strip": op_set_nla_strip,
    "match_world_transform": op_match_world_transform,
    "pose_seams": op_pose_seams,
    "self_clipping": op_self_clipping,
}
