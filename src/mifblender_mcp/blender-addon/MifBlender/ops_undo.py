"""Undo and redo for the bridge's own calls - one Blender undo step per call that can change the scene.

WHY IT EXISTS, 2026-09-16. A Discord user's agent reported MifBlender had no undo, and the answer was "there should be". Every
op here writes through bpy.data or calls operators from Python, and NEITHER leaves an undo step. MEASURED 2026-09-16,
plain bpy, --background, identical on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1:
  - a bpy.ops call from Python pushes NO step of its own: base push, primitive_cube_add(), ed.undo() -> "poll() failed"
    (nothing to undo), the cube still there. Only primitive_cube_add('EXEC_DEFAULT', True) pushes one.
  - one ed.undo_push after any number of writes is ONE step: two operators then one push, one undo removed both.
  - a write made AFTER the last push is LOST by the next undo - and redo does not bring it back.
  - with no base step at all, the first undo has nothing to go back to and fails.
  - a push when nothing changed still makes a step, so a read op must not push: it would spend an undo on nothing.
  - undo_push costs 0.4 ms on 10 objects, ~2 ms on 300, 12-50 ms on 1,500 (5.2.1).
  - a Python reference taken before an undo raises ReferenceError after it - no crash.
  - Blender keeps 32 steps by default (preferences.edit.undo_steps); past that the oldest go.
  - undo after freeing a rigid body bake brings the bake back (isBaked, its frames) marked "cache is outdated".

So server.py pushes a BASE step before the first call that can change anything, and a step AFTER every such call -
success or refusal, because a refusal that wrote something before refusing is otherwise a write the next undo loses.
READ_ONLY_OPS never push. undo / redo move through those steps and report what the scene gained and lost.

THE JOURNAL IS THE BRIDGE'S OWN. Blender offers Python no way to read its undo stack, so the step names here are the
ones this server pushed. In an INTERACTIVE Blender, edits made in the UI are steps too, and undo walks back through
them in whatever order they happened; the objects added and removed are always measured, so the response says what
really changed even when the journal's names do not line up with it.
"""
import os

import bpy

from .ops_common import MifOpError, reject_unknown, take_int

# OPS THAT CANNOT CHANGE THE SCENE, so no undo step follows them. Classified 2026-09-16 with audit_mutate_then_deny's
# own write predicates over every op whose name reads as a query: 42 of 44 showed no write at all, and the other two
# put back what they touch - evaluate_at_frame restores the frame in a finally, compare_to_reference removes the
# images it loads. Everything NOT listed here pushes a step: a redundant step costs one slot of 32, a missing one
# loses a write on the next undo.
READ_ONLY_OPS = frozenset({
    "audit_placement", "call_log", "closest_point_on_mesh", "compare_to_reference", "compositor_info",
    "describe_material", "describe_op", "evaluate_at_frame", "face_info", "file_info", "gen_status", "image_stats",
    "list_actions", "list_animation_data",
    "list_bones", "list_cameras", "list_collections", "list_constraints", "list_custom_properties", "list_group_nodes",
    "artifact_inventory", "list_images", "list_keyframes", "list_lights", "list_markers", "list_materials", "list_modifiers",
    "list_node_groups", "list_objects", "list_particles", "list_previews", "list_scenes", "list_shape_keys",
    "list_vertex_groups",
    "list_view_layers", "list_worlds", "mesh_quality", "mesh_stats", "object_info", "objects_overlap", "physics_info",
    "ping", "ray_cast", "render_info", "render_status", "scene_info", "uv_info", "vertex_info", "world_info",
})
# The undo ops themselves move through the stack; pushing after them would erase the redo they just made.
UNDO_OPS = frozenset({"undo", "redo", "undo_history"})
# A FILE LOAD REPLACES THE STACK. After it the next write pushes a fresh base.
RESETS_STACK = frozenset({"open_file"})
# WHAT UNDO CANNOT TAKE BACK: files written to disk and renders. Named in the response when one is in the range undone.
EXTERNAL_EFFECTS = frozenset({"export_mesh", "export_scene", "save_file", "render_still", "render_animation",
                              "bake_texture", "gen_image", "gen_texture", "gen_mesh", "gen_asset", "mixdown_audio",
                              "extract_audio"})

_STATE = {"based": False, "journal": [], "cursor": 0, "skipped": None, "writes": 0, "dropped": 0}


def bridge_writes():
    """How many calls that can change the scene this bridge has run since the file was loaded or saved to itself.

    THE DIRTY FLAG BLENDER DOES NOT KEEP FOR US. Reported against the addon: file_info said
    isDirty false after verified writes, and open_file trusts that flag before discarding a session. MEASURED
    2026-09-16 from a saved file, plain bpy, --background: on 5.2.1 bpy.data.is_dirty reads false after moving
    vertices, a BMesh bisect, a new object, a bpy.ops cube, a transform, a delete and a modifier - every one - and
    true only once ed.undo_push runs; on 3.6-5.0 it already reads true the moment the file opens. So the undo step
    after each write sets it where it works, and this count covers the rest: recording off, a push skipped outside
    OBJECT mode, a build whose flag does not move."""
    return _STATE["writes"]


def _stack_available():
    """(on, why not) for Blender's undo stack itself: MIF_BLENDER_UNDO=0 - the showcase renderer's choice - or Blender's
    global undo preference off. undo and redo need only this."""
    if os.environ.get("MIF_BLENDER_UNDO", "").strip() in ("0", "false", "False", "off"):
        return False, "MIF_BLENDER_UNDO=0 is set in this Blender's environment, so no undo steps are recorded"
    edit = getattr(bpy.context.preferences, "edit", None)
    if edit is not None and not getattr(edit, "use_global_undo", True):
        return False, "Blender's Global Undo preference is off (Edit > Preferences > System), so there is no undo stack"
    return True, None


def _fluid_domains(limit=3):
    """The first few objects carrying a Mantaflow DOMAIN - the fluid modifier that holds the simulation."""
    names = []
    for ob in bpy.data.objects:
        if any(m.type == "FLUID" and getattr(m, "fluid_type", None) == "DOMAIN" for m in getattr(ob, "modifiers", ())):
            names.append(ob.name)
            if len(names) >= limit:
                break
    return names


# AN UNDO STEP KILLS A MANTAFLOW SIMULATION when it is pushed after a flow object changes. Reported by the second seat
# (mailbox ids 149/151): a liquid splash rendered as empty domain cubes once this bridge pushed a step per call.
# MEASURED 2026-09-16 in plain bpy, no add-on (scratchpad measure_push_trigger_recovery.py, measure_push_trigger_files.py,
# measure_keyframe_push_mantaflow2.py): seat 2's scene - a 2 BU LIQUID domain, resolution 32, REPLAY cache 1-40, a 0.45
# sphere GEOMETRY inflow keyed at 1 and 3 and out of the world at 4 - with an ed.undo_push after each setup step and
# after each of the three keys, stepped to frame 30, evaluates to the bare 8-vertex cube on 4.2.17, 4.4.0, 5.0.1 and
# 5.2.1 and steps in 0.45-0.55 s; without the pushes after the keys, 24,994-25,236 vertices of liquid in 1.6-1.8 s. The
# flow moved three times with a push after each does the same; keys on an unrelated object, keys on the domain, and a
# single key with a single push do not. On 3.6.23 the setup grows no surface to count (docs/22 B69), and stepping
# halved the same way (0.59 s to 0.29 s). A depsgraph update before each push did not prevent it; rewriting a fluid
# setting afterwards recovered it on 4.2+ but not 3.6 and would restart any simulation stepped one call at a time. Through
# the bridge (bisect_undo_mantaflow.py) the steps after set_keyframe were the ones that mattered. So while the file holds
# a fluid domain no step is pushed, and undo_history says why.
def recording():
    """(on, why not): _stack_available, and no Mantaflow domain in the file (see above)."""
    on, why = _stack_available()
    if not on:
        return on, why
    domains = _fluid_domains()
    if domains:
        return False, ("this file holds a Mantaflow fluid domain (%s), and an undo step pushed after a flow object "
                       "changes stops the domain simulating - measured on 4.2-5.2, frame 30 the empty 8-vertex cube "
                       "instead of 24,994 vertices of liquid - so no undo steps are recorded while one exists. Steps "
                       "recorded before it are still there for undo." % ", ".join(domains))
    return True, None


def _window_override():
    """A window to run ed.undo / ed.undo_push in, when this Blender has one - None in --background."""
    wm = bpy.context.window_manager
    if bpy.context.window is None and wm is not None and wm.windows:
        win = wm.windows[0]
        return {"window": win, "screen": win.screen}
    return None


def _run(op, **kw):
    over = _window_override()
    if over and hasattr(bpy.context, "temp_override"):
        with bpy.context.temp_override(**over):
            return op(**kw)
    return op(**kw)


def _push(message):
    mode = getattr(bpy.context, "mode", "OBJECT")
    if mode != "OBJECT":
        # AN EDIT-MODE STEP IS A DIFFERENT STACK. A global step pushed from edit mode is an edit-mode step, which a later
        # global undo does not walk the same way - so none is pushed, and undo_history says why.
        _STATE["skipped"] = "%s was not recorded: Blender was in %s mode, not OBJECT" % (message, mode)
        return False
    _run(bpy.ops.ed.undo_push, message=message)
    del _STATE["journal"][_STATE["cursor"]:]            # a new step ends the redo history, as it does in Blender
    _STATE["journal"].append(message)
    # BLENDER KEEPS preferences.edit.undo_steps STEPS BACK AND DROPS THE OLDEST, so this journal does too. Measured
    # 2026-09-21 in plain bpy on 3.6.23 and 5.0.1: after a base step and 40 pushes with undo_steps 32, exactly 32
    # ed.undo calls succeed and the 33rd fails its poll; after 25 with undo_steps 10, exactly 10. Unbounded, the journal
    # said undoAvailable 124 after test_blender_reject_unknown's refusals while Blender could take back 31, and undo
    # stopped on "poll() failed" instead of saying why.
    limit = _undo_limit()
    extra = len(_STATE["journal"]) - (limit + 1)
    if limit > 0 and extra > 0:
        del _STATE["journal"][:extra]
        _STATE["dropped"] += extra
    _STATE["cursor"] = len(_STATE["journal"])
    return True


def _undo_limit():
    """preferences.edit.undo_steps - how many steps back Blender keeps - or 0 when it cannot be read."""
    edit = getattr(bpy.context.preferences, "edit", None)
    try:
        return int(getattr(edit, "undo_steps", 0)) if edit is not None else 0
    except (TypeError, ValueError):
        return 0


def _extension_read(endpoint):
    """An op a separate add-on registered with undo "read" (extensions.py) pushes no step, as READ_ONLY_OPS do."""
    from . import extensions
    return extensions.undo_class(endpoint) == "read"


def before_write(endpoint):
    """Called by server.py before a call that can change the scene: the base step, once per stack."""
    if endpoint in READ_ONLY_OPS or endpoint in UNDO_OPS or _STATE["based"] or _extension_read(endpoint):
        return
    if not recording()[0]:
        return
    try:
        if _push("MifBlender: before the first bridge call"):
            _STATE["based"] = True
    except Exception as exc:                                                  # noqa: BLE001 - never breaks the op
        _STATE["skipped"] = "the base undo step could not be pushed: %s" % exc


def after_write(endpoint, label=None, result=None):
    """Called by server.py after a call that can change the scene - whether it succeeded or refused. `result` is the
    op's response when it succeeded, None when it refused or raised.

    Returns (pushed, why not): whether an undo step now holds this call, and the reason when none does - what a batch's
    atomicNote says, since whether undo can take a batch back depends on it."""
    if endpoint in UNDO_OPS or endpoint in READ_ONLY_OPS or _extension_read(endpoint):
        return False, "%s changes nothing an undo step would hold" % endpoint
    if endpoint in RESETS_STACK:
        if result is not None:
            reset()
        return False, "a file load starts Blender's undo stack again"
    # A SAVE CHANGES NOTHING IN THE SCENE, so it is neither a write to count nor a step to push - and it clears the count
    # only when it saved the SESSION'S OWN file. save_file copies by default: the session keeps pointing at its old path,
    # whose contents did not change, so the unsaved work is still unsaved there.
    if endpoint == "save_file":
        if (isinstance(result, dict) and result.get("filepath")
                and os.path.normcase(str(result.get("sessionFilepath") or ""))
                == os.path.normcase(str(result["filepath"]))):
            _STATE["writes"] = 0
        return False, "a save changes nothing in the scene"
    _STATE["writes"] += 1
    on, why = recording()
    if not on:
        return False, why
    try:
        if _push("MifBlender: %s" % (label or endpoint)):
            return True, None
        return False, _STATE["skipped"]
    except Exception as exc:                                                  # noqa: BLE001 - never breaks the op
        _STATE["skipped"] = "the undo step for %s could not be pushed: %s" % (endpoint, exc)
        return False, _STATE["skipped"]


def reset(*_args):
    """A new file: Blender's stack starts again, so ours does, and nothing in memory is unsaved. Also load_post."""
    _STATE.update(based=False, journal=[], cursor=0, skipped=None, writes=0, dropped=0)


def _snapshot():
    counts = {}
    for attr in ("objects", "meshes", "materials", "collections", "cameras", "lights", "images", "node_groups",
                 "actions", "curves", "armatures", "worlds", "scenes", "speakers", "sounds"):
        coll = getattr(bpy.data, attr, None)
        if coll is not None:
            counts[attr] = len(coll)
    return {"objects": {o.name for o in bpy.data.objects}, "counts": counts,
            "frame": bpy.context.scene.frame_current if bpy.context.scene else None}


def _diff(before, after):
    changed = {k: [before["counts"][k], after["counts"].get(k)] for k in before["counts"]
               if before["counts"][k] != after["counts"].get(k)}
    return {"objectsAdded": sorted(after["objects"] - before["objects"]),
            "objectsRemoved": sorted(before["objects"] - after["objects"]),
            "datablockCountsChanged": changed or None}


def _refuse_unless_ready(verb):
    on, why = _stack_available()
    if not on:
        raise MifOpError("cannot %s: %s. NOTHING was changed." % (verb, why))
    mode = getattr(bpy.context, "mode", "OBJECT")
    if mode != "OBJECT":
        raise MifOpError("cannot %s in %s mode - the bridge's steps are object-mode steps, and an edit-mode undo walks "
                         "a different stack. Leave edit mode first. NOTHING was changed." % (verb, mode))


def _move(params, backward, op):
    """Walk the stack: backward for undo, forward for redo. `op` is the caller's name, for its messages."""
    steps = take_int(params, "steps", default=1)
    if steps < 1 or steps > 64:
        raise MifOpError("steps must be 1 to 64, got %d. NOTHING was changed." % steps)
    _refuse_unless_ready(op)
    j = _STATE["journal"]
    before = _snapshot()
    moved, stopped = [], None
    for _ in range(steps):
        if backward and _STATE["cursor"] <= 1:
            if _STATE["dropped"]:
                stopped = ("at the oldest step Blender still keeps - preferences.edit.undo_steps is %d, and the %d "
                           "bridge step(s) before it are gone" % (_undo_limit(), _STATE["dropped"]))
            else:
                stopped = ("at the first step this bridge recorded - nothing older is the bridge's to take back" if j
                           else "this bridge has recorded no step since Blender started or the file was opened")
            break
        if not backward and _STATE["cursor"] >= len(j):
            stopped = "nothing to go forward to - no step has been taken back, or a newer call replaced that history"
            break
        try:
            _run(bpy.ops.ed.undo if backward else bpy.ops.ed.redo)
        except RuntimeError as exc:
            stopped = "Blender refused to go %s further: %s" % ("back" if backward else "forward", str(exc)[:160])
            break
        if backward:
            _STATE["cursor"] -= 1
            moved.append(j[_STATE["cursor"]])
        else:
            moved.append(j[_STATE["cursor"]])
            _STATE["cursor"] += 1
    after = _snapshot()
    names = [m.replace("MifBlender: ", "", 1) for m in moved]
    out = {("undone" if backward else "redone"): names, "steps": len(moved), "askedSteps": steps,
           "stoppedBecause": stopped, "undoAvailable": max(0, _STATE["cursor"] - 1),
           "redoAvailable": len(j) - _STATE["cursor"], "frame": after["frame"]}
    out.update(_diff(before, after))
    external = sorted({n.split(" ")[0] for n in names} & EXTERNAL_EFFECTS)
    if external:
        out["notReverted"] = ("%s wrote outside the scene - files on disk and renders stay as they are; only the scene "
                              "moved." % ", ".join(external))
    if moved and not (out["objectsAdded"] or out["objectsRemoved"] or out["datablockCountsChanged"]):
        out["note"] = ("no object or datablock was added or removed - the step changed properties, which were restored "
                       "but this summary does not list them. Read the objects you care about to confirm.")
    return out


def op_undo(params):
    """Undo the bridge's last call(s) that changed the scene.

    params:
      steps (int)   how many, default 1, at most 64. Blender keeps preferences.edit.undo_steps (32 by default).

    Stops at the first step this bridge recorded, and says why whenever it stops short. Reports the steps undone by
    name, what the scene gained and lost (objectsAdded / objectsRemoved / datablockCountsChanged), and how many steps
    are left each way.
    """
    reject_unknown(params, {"steps"}, "undo")
    return _move(params, True, "undo")


def op_redo(params):
    """Redo what undo took back. A call that changes the scene after an undo ends the redo history, as in Blender.

    params:
      steps (int)   how many, default 1, at most 64.
    """
    reject_unknown(params, {"steps"}, "redo")
    return _move(params, False, "redo")


def op_undo_history(params):
    """The steps this bridge has recorded since Blender started or the file was opened, and where undo stands."""
    reject_unknown(params, (), "undo_history")
    on, why = recording()
    edit = getattr(bpy.context.preferences, "edit", None)
    j, cur = _STATE["journal"], _STATE["cursor"]
    return {"recording": on, "whyNotRecording": why,
            "undoStepsLimit": int(getattr(edit, "undo_steps", 0)) if edit is not None else None,
            "steps": [{"index": i, "step": m.replace("MifBlender: ", "", 1), "applied": i < cur}
                      for i, m in enumerate(j)],
            "undoAvailable": max(0, cur - 1), "redoAvailable": len(j) - cur,
            # How many of the bridge's older steps Blender has let go of - it keeps undo_steps back.
            "olderStepsDropped": _STATE["dropped"],
            "lastSkipped": _STATE["skipped"],
            "note": ("the bridge's own steps. In an interactive Blender, edits made in the UI are undo steps too and are "
                     "not listed here - undo walks back through them in the order they happened.")}


OPS = {
    "undo": op_undo,
    "redo": op_redo,
    "undo_history": op_undo_history,
}


# A FILE LOADED ANY WAY AT ALL - open_file, the File menu, a script - replaces Blender's stack, so the journal starts
# again with it. Persistent, or the first load would unregister it; replaced by name, so reloading this module does not
# stack a second copy. GUARDED, because test_blender_refusals imports every ops module under a stub bpy with no
# app.handlers - an unguarded registration at import took the whole op table down with it there.
def _mif_undo_on_load(*_args):
    reset()


def _install_load_handler():
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    if handlers is None or not hasattr(handlers, "load_post"):
        return False
    for h in [h for h in handlers.load_post if getattr(h, "__name__", "") == "_mif_undo_on_load"]:
        handlers.load_post.remove(h)
    persistent = getattr(handlers, "persistent", None)
    handlers.load_post.append(persistent(_mif_undo_on_load) if persistent else _mif_undo_on_load)
    return True


_install_load_handler()
