"""MifBlender socket server -- the Blender backend of MifBridge.

THREADING CONTRACT (read this before changing anything in this file)
====================================================================
bpy.data / the DNA-RNA layer is NOT thread-safe, and bpy.ops additionally reads
the global context, pushes undo steps and tags the depsgraph. A write from a
non-main thread while the main thread is mid depsgraph-eval or mid-draw is a
data race on raw C pointers: the failure is a hard process crash or silent heap
corruption with NO Python traceback. You cannot try/except your way out of it.

So the split is absolute:

  socket thread   ->  accept(), recv(), json.loads(), enqueue, sendall()
                      ...and NOTHING else. It must never touch bpy.
  main thread     ->  every single bpy call, run out of a queue by ONE timer.

The marshalling primitive is bpy.app.timers.register, as required. Note WHERE
it is called from: the drain timer is registered ONCE, from the addon's
register() on the main thread (see __init__.py). The socket thread only ever
does _JOBS.put(job).

That differs from the usual pattern (which calls bpy.app.timers.register from
the socket thread, once per command). I could NOT verify from Blender source or
docs that bpy.app.timers.register is safe to call off the main thread, and this
whole design would rest on it. A queue plus one main-thread-registered timer
needs no such assumption. Do not "simplify" it back.

BACKGROUND MODE
===============
Under `blender -b` there is no event loop, so timers NEVER fire. Three places
branch on bpy.app.background, and all three are required or a headless run
hangs forever with no output:
  1. start()          - run the accept loop on the main thread, no server thread
  2. _server_loop()   - handle each client serially, no client thread
  3. _handle_client() - execute the job inline instead of queueing it

Consequently the addon does NOT auto-start in background mode: register() would
never return. Headless callers use MifBlender.serve_forever() explicitly.

Adapted from blender-mcp (github.com/MCPBlender/blender-mcp),
(c) 2025 Siddharth Ahuja, MIT licence.
"""

from __future__ import annotations

import atexit
import hmac
import json
import math
import os
import queue
import socket
import sys
import threading
import time
import traceback

import bpy

from . import artifacts, framing, journal, registry
from .ops_common import MifOpError, finite_float, jsonable, take_bool

# The listener this process registered (port, pid), so an exit without stop() still removes its entry (registry.py).
_REGISTERED = [None]


def _unregister_at_exit():
    if _REGISTERED[0] is not None:
        registry.unregister(*_REGISTERED[0])
        _REGISTERED[0] = None


atexit.register(_unregister_at_exit)

# Loopback ONLY. There is deliberately no bind-address preference: the socket
# accepts commands that move files and (optionally) run arbitrary Python, and a
# 0.0.0.0 checkbox is a foot-gun no matter how it is labelled. If you genuinely
# need remote access, tunnel it.
HOST = "127.0.0.1"
DEFAULT_PORT = 8792  # UE plugin is 8791; third-party blender-mcp is 9876. No clash.

PROTOCOL_VERSION = 1

# The MCP server abandons a work request after MIF_BLENDER_TIMEOUT, default 180s
# (tools/mcp-server/server.py). This end must give up FIRST or the ladder is
# inverted: the MCP times out, calls _bl_close(), and Blender carries on mutating
# the scene for another several minutes on behalf of a caller that has already
# been told the op failed. 150s leaves 30s of headroom for the socket round trip.
# If you raise MIF_BLENDER_TIMEOUT, raise this too, and keep it below.
DEFAULT_JOB_TIMEOUT = 150.0

_JOBS: "queue.Queue[_Job]" = queue.Queue()
_TIMER_REGISTERED = False


# ---------------------------------------------------------------------------
# Job: one request, executed on the main thread, awaited by the socket thread
# ---------------------------------------------------------------------------

class _Job:
    __slots__ = ("request", "response", "done", "cancelled")

    def __init__(self, request):
        self.request = request
        self.response = None
        self.done = threading.Event()
        self.cancelled = False


def _drain_timer():
    """Registered once, on the main thread, by the addon's register().

    MUST NOT RAISE. A timer callback that raises is unregistered by Blender, and
    the server would then accept requests forever and answer none of them.
    """
    try:
        try:
            job = _JOBS.get_nowait()
        except queue.Empty:
            return 0.05  # idle poll -- 20 Hz is imperceptible and costs nothing

        if not job.cancelled:
            try:
                job.response = _execute(job.request)
            except BaseException as exc:  # noqa: BLE001 - a timer may not raise
                traceback.print_exc()
                job.response = {"ok": False, "error": "internal: %s" % exc}
        job.done.set()

        # Come straight back if more work is waiting.
        return 0.0 if not _JOBS.empty() else 0.05
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        return 0.25


def ensure_timer():
    """Idempotent. Call from register() on the main thread."""
    global _TIMER_REGISTERED
    if _TIMER_REGISTERED and bpy.app.timers.is_registered(_drain_timer):
        return
    if not bpy.app.timers.is_registered(_drain_timer):
        bpy.app.timers.register(_drain_timer, first_interval=0.0, persistent=True)
    _TIMER_REGISTERED = True


def remove_timer():
    global _TIMER_REGISTERED
    if bpy.app.timers.is_registered(_drain_timer):
        bpy.app.timers.unregister(_drain_timer)
    _TIMER_REGISTERED = False


# ---------------------------------------------------------------------------
# Dispatch  (runs on the MAIN thread only, from _drain_timer or inline in -b)
# ---------------------------------------------------------------------------

def _closest_ops(name, table, limit=5):
    """The op names a caller most likely meant: shared words first, weighted by how rare each word is, then spelling.

    ADDED 2026-09-16 for the second seat's first message (mailbox id 2): it guessed set_object_transform for
    transform_object from the read side's vocabulary and got eight refusals, each ending in a 190-name alphabetical list.
    difflib alone does not find that one - the words are the same and the order is not. Counting shared words is not
    enough either: set_object_visibility shares two words with it as well, and "set" and "object" are in dozens of op
    names while "transform" is in three. So each shared word scores log(ops / ops containing it), the rare word decides,
    and difflib's close spellings fill what is left.
    """
    import difflib
    import math
    ops = list(table)
    df = {}
    for op in ops:
        for w in set(op.split("_")):
            df[w] = df.get(w, 0) + 1
    words = set(str(name).lower().split("_")) - {""}
    scored = []
    for op in ops:
        shared = words & set(op.split("_"))
        if shared:
            scored.append((sum(math.log(float(len(ops)) / df[w]) for w in shared), len(shared), op))
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    picked = [op for score, n, op in scored if n >= 2 or score >= math.log(len(ops) / 3.0)]
    for op in difflib.get_close_matches(str(name).lower(), ops, n=limit, cutoff=0.6):
        if op not in picked:
            picked.append(op)
    return picked[:limit]


def _op_table():
    """Every op the dispatcher answers: the built-ins, then the ops separate add-ons registered (extensions.py). A
    built-in always wins - registration already refuses a built-in's name, and setdefault keeps it so twice."""
    table = _builtin_op_table()
    from . import extensions
    for name, fn in extensions.ops().items():
        table.setdefault(name, fn)
    return table


def _builtin_op_table():
    # Imported here rather than at module scope so a syntax error in an ops
    # module surfaces as a per-request error instead of a dead addon.
    from . import (ops_scene, ops_mesh, ops_gen, ops_rig, ops_create, ops_material,
                   ops_lightcam, ops_anim, ops_render, ops_world,
                   ops_physics, ops_particles, ops_nodes, ops_viewport, ops_file, ops_constraint,
                   ops_collection, ops_viewlayer, ops_io, ops_query, ops_describe, ops_audio, ops_tree)
    from . import ops_nature

    table = {}
    table.update(ops_scene.OPS)
    table.update(ops_mesh.OPS)
    table.update(ops_gen.OPS)
    table.update(ops_rig.OPS)
    table.update(ops_create.OPS)
    table.update(ops_material.OPS)
    table.update(ops_lightcam.OPS)
    table.update(ops_anim.OPS)
    table.update(ops_render.OPS)
    table.update(ops_world.OPS)
    table.update(ops_physics.OPS)
    table.update(ops_particles.OPS)
    table.update(ops_nodes.OPS)
    table.update(ops_viewport.OPS)
    table.update(ops_file.OPS)
    table.update(ops_constraint.OPS)
    table.update(ops_collection.OPS)
    table.update(ops_viewlayer.OPS)
    table.update(ops_io.OPS)
    table.update(ops_query.OPS)
    table.update(ops_describe.OPS)
    from . import ops_artifacts
    table.update(ops_artifacts.OPS)
    table.update(ops_audio.OPS)
    table.update(ops_tree.OPS)
    table.update(ops_nature.OPS)
    from . import ops_kit
    table.update(ops_kit.OPS)
    from . import ops_undo
    table.update(ops_undo.OPS)
    from . import ops_topology
    table.update(ops_topology.OPS)
    from . import ops_image
    table.update(ops_image.OPS)
    from . import ops_uv
    table.update(ops_uv.OPS)
    from . import ops_pose
    table.update(ops_pose.OPS)
    from . import ops_rigify
    table.update(ops_rigify.OPS)
    from . import ops_motion
    table.update(ops_motion.OPS)
    from . import ops_garment
    table.update(ops_garment.OPS)
    from . import ops_body
    table.update(ops_body.OPS)
    from . import ops_video
    table.update(ops_video.OPS)
    from . import ops_gpencil
    table.update(ops_gpencil.OPS)
    from . import ops_preview
    table.update(ops_preview.OPS)
    from . import ops_datablock
    table.update(ops_datablock.OPS)
    from . import ops_meshdata
    table.update(ops_meshdata.OPS)
    from . import ops_handoff
    table.update(ops_handoff.OPS)
    from . import ops_retarget
    table.update(ops_retarget.OPS)
    from . import ops_lookdev
    table.update(ops_lookdev.OPS)
    from . import ops_sculpt
    table.update(ops_sculpt.OPS)
    from . import ops_gamebake
    table.update(ops_gamebake.OPS)
    from . import ops_detailbake
    table.update(ops_detailbake.OPS)
    return table


def op_names():
    try:
        return sorted(_op_table().keys())
    except Exception as exc:  # noqa: BLE001
        return ["<op table failed to load: %s>" % exc]


# HOW LONG A BATCH MAY RUN BEFORE IT STOPS ON PURPOSE.
#
# NOT A CAP ON OP COUNT, and that is the whole design decision. Measured 2026-09-05 on 5.0 headless
# from the elapsedMs every response already carries: list_objects 0.05ms, set_keyframe 0.15ms,
# apply_modifier 10.9ms, add_modifier SUBSURF 205ms, render_still 610ms on a first render. Four
# orders of magnitude. So 1000 ordinary ops is about a second while 250 renders is 150 seconds, and
# any number of OPS is the wrong axis to limit.
#
# WHY THERE IS A BUDGET AT ALL, given the ceiling barely binds. In BACKGROUND mode there is no
# timeout of any kind - _handle_client runs the job inline, which is the path every suite and every
# showcase stage uses - so a long batch there is simply slow. In the GUI the drain timer waits on
# job_timeout, and a batch that outruns it leaves the caller with a timeout and no idea which ops
# landed. The budget converts that into a partial result that NAMES where it stopped.
#
# The default is deliberately below the addon's own job_timeout so the batch stops itself before the
# transport does. A caller who wants the headless behaviour asks for it with budgetSeconds.
_BATCH_DEFAULT_BUDGET = 120.0
_BATCH_MAX_OPS = 500          # a runaway-loop backstop, not a design limit - see above


def _atomic_note(pushed, why_not):
    """A batch's atomicNote: not a transaction, and whether undo can take it back - which depends on the step.

    IT SAID "Nothing was rolled back and nothing can be" until 2026-09-28 (the 2026-09-21 pack-rule review), and that
    "Blender pushes an undo step per operator" - both false since the batch became ONE undo step (ops_undo). Measured on
    5.2.1: a batch of create_primitive then a failing delete_object, and undo {steps: 1} removed the cube it made."""
    head = ("a batch is NOT a transaction: when an op fails, the ops in `applied` stay applied - nothing was "
            "rolled back for you.")
    if pushed:
        return head + (" The whole batch is ONE undo step, so undo {steps: 1} takes all of it back, the applied ops "
                       "included.")
    return head + (" No undo step was recorded for it (%s), so undo cannot take it back."
                   % (why_not or "undo is not recording"))


def _run_batch(params, table, started, committing=False):
    """Run several ops as ONE job, and report exactly what happened to each.

    THE CONTRACT IS NOT ATOMIC AND SAYS SO. A batch that fails at op 7 of 10 leaves six applied. (This
    said Blender pushes an undo step per bpy.ops call; measured 2026-09-16 on all five builds, a bpy.ops
    call from Python pushes none. A batch is ONE undo step now - see ops_undo - so undo takes back the
    whole of it, the six applied ops included.) There is no transaction to roll back - the UE side wraps a transacted endpoint and
    PM-007 records that even that cancel does not really roll back. So the honest contract, and the
    one this repo already uses everywhere else: say what was applied, name what was not, and never
    claim atomicity.

    STOPS AT THE FIRST FAILURE by default. A batch is usually a build sequence - create, then
    modify, then assign - where op 5 failing makes op 6 meaningless, and running on would turn one
    diagnosable error into six confusing ones. continueOnError:true runs the rest for the cases
    where the ops are independent, and the response reports which mode ran.

    THE ENVELOPE IS VALIDATED FOR ITS OWN KEYS ONLY. Each op's params are validated by the handler
    it dispatches to, which already refuses unknown keys by name - checking them here as well would
    be a second guard to keep in step with 172 ops, and the UE batch settled this the same way.

    AN EMPTY ops[] IS A HARD FAILURE rather than ok:true with zero results. The response IS the
    audit trail, so "nothing to do" is a caller error worth naming.
    """
    ops = params.get("ops")
    if ops is None:
        raise MifOpError("batch needs 'ops': a list of {op, params} entries. NOTHING was run.")
    if not isinstance(ops, list):
        raise MifOpError("'ops' must be a LIST of {op, params} entries, got %s. NOTHING was run."
                         % type(ops).__name__)
    if not ops:
        raise MifOpError("'ops' is empty. A batch's response IS its audit trail, so nothing-to-do "
                         "is a caller error rather than a success with no rows. NOTHING was run.")
    if len(ops) > _BATCH_MAX_OPS:
        raise MifOpError("%d ops is past the %d-entry backstop. That limit is a runaway guard, not "
                         "a performance one - if you genuinely need more, send two batches. NOTHING "
                         "was run." % (len(ops), _BATCH_MAX_OPS))

    unknown = [k for k in params if k not in
               ("ops", "continueOnError", "budgetSeconds", "stopOnBudget")]
    if unknown:
        raise MifOpError("batch: unknown param(s) %s. Accepted: ops, continueOnError, "
                         "budgetSeconds, stopOnBudget. NOTHING was run."
                         % ", ".join(sorted(unknown)))

    # EVERY ENTRY CHECKED BEFORE ANY OF THEM RUNS. A batch whose eighth entry names an op that does
    # not exist should not first perform seven mutations - the same parse-before-write ordering
    # create_force_field had to learn, one level up.
    prepared = []
    for i, entry in enumerate(ops):
        if not isinstance(entry, dict):
            raise MifOpError("ops[%d] must be an object like {\"op\": \"create_light\", "
                             "\"params\": {...}}, got %s. NOTHING was run."
                             % (i, type(entry).__name__))
        name = entry.get("op") or entry.get("endpoint")
        if not name:
            raise MifOpError("ops[%d] has no 'op'. NOTHING was run." % i)
        if name == "batch":
            raise MifOpError("ops[%d] is 'batch' - a batch inside a batch would share one job and "
                             "one budget with its parent, so the nesting buys nothing and the "
                             "reporting stops being readable. NOTHING was run." % i)
        sub_params = entry.get("params") or {}
        if not isinstance(sub_params, dict):
            raise MifOpError("ops[%d] ('%s') has params of type %s, expected an object. NOTHING "
                             "was run." % (i, name, type(sub_params).__name__))
        fn = table.get(name)
        if fn is None:
            near = _closest_ops(name, table)
            raise MifOpError("ops[%d] names unknown op '%s'%s. NOTHING was run - the whole batch is "
                             "checked before any of it runs, so a typo in the last entry does not "
                             "leave the first eight applied." % (i, name, (" (closest: %s)" % ", ".join(near))
                                                                  if near else ""))
        prepared.append((name, sub_params, fn))

    # take_bool and finite_float, not bool() and float(). Measured on 5.0.1, 2026-09-21: bool("false") is True,
    # so {continueOnError: "false"} RAN ON past a failed op and made what came after it, and
    # {stopOnBudget: "false"} stopped on the budget anyway; and float(NaN) is a number, so budgetSeconds NaN
    # switched the budget off in silence (every comparison with NaN is False) while the response echoed it.
    try:
        cont = take_bool(params, "continueOnError", default=False)
        stop_on_budget = take_bool(params, "stopOnBudget", default=True)
        budget = finite_float(params.get("budgetSeconds", _BATCH_DEFAULT_BUDGET), "budgetSeconds")
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was run." % exc)

    # THE CHANGE CAP (cap.py; docs/15 section 6): counted whole, after every entry has been checked and before the first
    # op runs. Over the cap it raises a preview to commit with its previewId - unless this IS that commit.
    if not committing:
        from . import cap
        cap.gate("batch", params)

    results = []
    applied, failed, skipped = [], [], []
    stopped_for = None
    # EACH OP IN A BATCH IS JOURNALED AS ITS OWN CALL, carrying the batch's seq and its index - a Blender that dies at
    # op 137 of 200 should name op 137, not only "batch". The batch's own start is the last one written.
    parent_seq = journal.current_seq()
    for i, (name, sub_params, fn) in enumerate(prepared):
        # THE BUDGET IS CHECKED BEFORE THE NEXT OP, NOT DURING ONE. Nothing here can interrupt a
        # running handler - the addon is single-threaded by design and this must not change that -
        # so the guarantee is "it will not START one it cannot afford", which is the only one
        # available and is enough: the caller learns where it stopped and can send the rest.
        elapsed = time.perf_counter() - started
        if stop_on_budget and budget > 0 and elapsed >= budget and i > 0:
            stopped_for = ("budget", i, name, round(elapsed, 3))
            skipped = [n for n, _p, _f in prepared[i:]]
            break
        t0 = time.perf_counter()
        row = {"index": i, "op": name}
        sub_seq = journal.call_start(name, sub_params, batch=parent_seq, index=i)
        try:
            res = fn(sub_params)
            row["ok"] = True
            if isinstance(res, dict):
                row["result"] = res
            elif res is not None:
                row["result"] = {"result": res}
            applied.append(name)
        except MifOpError as exc:
            row["ok"] = False
            row["error"] = str(exc)
            failed.append(name)
        except Exception as exc:                                   # noqa: BLE001
            traceback.print_exc()
            row["ok"] = False
            row["error"] = "%s: %s" % (type(exc).__name__, exc)
            row["traceback"] = traceback.format_exc()
            failed.append(name)
        row["elapsedMs"] = round((time.perf_counter() - t0) * 1000.0, 2)
        journal.call_end(sub_seq, name, row["ok"], row.get("error"), row["elapsedMs"])
        artifacts.flush(name, row["ok"])
        results.append(row)
        # THE VIEW LAYER IS UPDATED BETWEEN TWO OPS OF A BATCH, as it is between two calls. An object removed by one op
        # stays in view_layer.objects - as None - until an update (measured 5.0.1, 2026-10-18), and the next op of the
        # batch iterated it: delete_object then export_scene failed with "'NoneType' object has no attribute 'type'".
        # The add-on's own removals update it (ops_common.remove_objects); this covers a run_python step that removes.
        try:
            bpy.context.view_layer.update()
        except Exception:                                          # noqa: BLE001
            pass
        if not row["ok"] and not cont:
            stopped_for = ("error", i, name, None)
            skipped = [n for n, _p, _f in prepared[i + 1:]]
            break

    out = {
        "requested": len(prepared),
        "ran": len(results),
        "applied": applied,
        "failed": failed,
        "skipped": skipped,
        "results": results,
        "continueOnError": cont,
        "budgetSeconds": budget,
        # NOT ATOMIC, AND THE FIELD SAYS SO RATHER THAN THE DOCUMENTATION ALONE. A caller reading
        # only the response must not be able to believe a failed batch changed nothing. atomicNote is
        # written by the dispatcher once the batch's undo step is pushed or not - see _atomic_note.
        "atomic": False,
    }
    if stopped_for:
        kind, idx, name, elapsed = stopped_for
        if kind == "budget":
            out["stoppedBecause"] = (
                "the %.1fs budget was reached after %d op(s) (%.3fs elapsed), so ops[%d] ('%s') was "
                "NOT started. The ops before it ARE applied. Send the rest as a second batch, or "
                "raise budgetSeconds - in background mode there is no transport timeout at all, so "
                "a bigger budget there costs only wall clock."
                % (budget, len(results), elapsed, idx, name))
        else:
            out["stoppedBecause"] = (
                "ops[%d] ('%s') failed and continueOnError is false, so the %d op(s) after it were "
                "not started. A batch is usually a build sequence where a later op depends on an "
                "earlier one, and running on would turn one diagnosable error into several."
                % (idx, name, len(skipped)))
    out["ok"] = not failed
    return out


# WHICH RESPONSE KEYS readBack:false SUPPRESSES, per op. GENERATED by
# tools/gen_readback_map.py from the addon's own source - re-derive rather than edit by hand.
# tools/test_blender_refusals.py B135 re-derives both tables and fails if either drifts: they did,
# from the 15 ops of 2026-09-15 to a derivation of 28 by 2026-09-23, with nothing comparing them.
# Regenerated 2026-09-23 - 19 ops once the new reads went to READBACK_REFUSED, and the keys that must
# survive (a warning, a fingerprint no read returns) to the generator's KEEP. Its docstring says why a
# new row is read before it is pasted.
#
# PER OP AND NOT A SET OF KEY NAMES, because the same name means different things: `note` here
# on link_objects is a serialised list of names, while `note` on boolean_op is the "changed
# nothing" advisory that must SURVIVE suppression. A flat key set would delete the sentence.
# 2026-10-04: the topology ops (ops_topology) - "after" is mesh_counts read after the write; no refusal in them reads it,
# each refuses before its bmesh is written (read before pasting, as gen_readback_map.py asks).
# 2026-10-05: subdivide_mesh (ops_topology) - "after" as the other topology ops; every refusal fires before the write.
# 2026-10-06: the clip and library ops (ops_video, ops_file) - each key is the state read back, and every refusal
# fires before the first write. sculpt_stroke (ops_topology) the same: its multires and dyntopo refusals fire before the first write.
# 2026-10-06: add_quads_on_surface (ops_topology) - "after" is mesh_counts read after the write; every refusal fires
# before the bmesh is written.
# 2026-10-04: add_rigid_body_constraint's many-joint "joints" - rows read off the joints made; every refusal fires before
# the first joint exists, and none reads a row. set_world's "sun" is the world's sun read back inside "after" (its sun
# refusals all fire before the first write).
# 2026-10-06: assign_node_group - "exposedInputs" is the attached group's inputs read back off the modifier after the
# write; a refusal after a library append takes the append back first.
# 2026-10-08: draw_grease_pencil_strokes (ops_gpencil) - "layerFrames" is the layer's frames read after the write; every
# refusal fires before the first stroke is drawn.
# 2026-10-07: attach_hood, attach_sleeve and displace_by_rule (ops_garment) - "counts" is mesh_counts read after the
# write; every refusal fires before the bmesh is written.
READBACK_SUPPRESSES = {
    "add_group_interface":     ("interface",),
    "add_inner_layer":         ("after",),
    "add_quads_on_surface":    ("after",),
    "add_rigid_body_constraint": ("joints",),
    "add_strip":               ("soundStrip",),
    "assign_action":           ("info", "previousActionNow"),
    "assign_node_group":       ("exposedInputs",),
    "attach_hood":             ("counts",),
    "attach_sleeve":           ("counts",),
    "bevel_edges":             ("objectAfter", "offSeamVerts"),
    "build_rib":               ("after",),
    "create_action":           ("info",),
    "create_node_group":       ("interface",),
    "create_primitive":        ("object",),
    "cut_eyelets":             ("after",),
    "displace_by_rule":        ("counts",),
    "dissolve_mesh_elements": ("after",),
    "duplicate_collection":    ("inScene",),
    "edit_strip":              ("right", "strip"),
    "export_mesh":             ("exported",),
    "extrude_mesh":           ("after",),
    "extrude_skirt":           ("objectAfter", "offSeamVerts"),
    "fill_mesh_loop":         ("after",),
    "gather_loop":             ("after",),
    "import_mesh":             ("imported",),
    "inset_faces":            ("after",),
    "join_boundary_loops":    ("after",),
    "join_objects":            ("verts",),
    "knife_cut":              ("after",),
    "library_override":       ("hierarchy",),
    "link_objects":            ("inScene", "note"),
    "list_clip":              ("camera", "clip", "settings", "stabilization", "stabilizationResidual"),
    "list_libraries":         ("libraries",),
    "load_clip":              ("clip",),
    "loop_cut":               ("after",),
    "merge_vertices":         ("after",),
    "offset_edges":           ("after",),
    "quilt_surface":           ("after",),
    "remove_modifier":         ("meshUnchanged",),
    "sculpt_stroke":           ("after",),
    "rip_edges":               ("after",),
    "separate_mesh":           ("verts",),
    "set_clip":               ("stabilizationResidual",),
    "set_collection_exporter": ("exporter",),
    "set_custom_property":     ("libraryOverridable", "properties"),
    "set_freestyle":           ("lineSets",),
    "set_keying_set":          ("keyingSet", "keyingSets"),
    "set_library":            ("after",),
    "set_light_linking":       ("blockerMembers", "receiverMembers"),
    "set_mesh_attribute":      ("attribute",),
    "set_modifier":            ("readBack",),
    "set_nla_strip":           ("after",),
    "set_render_settings":     ("after", "eeveeRaytracing", "video"),
    "set_vertex_color":        ("activeColor",),
    "set_world":               ("after", "sun"),
    "subdivide_mesh":          ("after",),
    "transfer_by_uv":          ("counts",),
    "transform_object":        ("object",),
    "trim_to_outline":         ("after",),
    "wrap_to_target":          ("counts",),
    "draw_grease_pencil_strokes": ("layerFrames",),
}

# REFUSED, not honored: each exists to return its payload - the reads, and open_file's report of what
# it loaded - so suppressing it would take the answer, or the note qualifying it, while still
# reporting ok:true.
READBACK_REFUSED = (
    "object_info", "file_info", "list_group_nodes", "open_file", "mesh_stats", "ray_cast",
    "closest_point_on_mesh", "face_info", "vertex_info", "select_faces", "objects_overlap",
    "render_info",
)

# A WRITE OP'S READ, on which readBack:false suppresses nothing: these answer a read with the key their write echoes, and
# emptying it would be a wrong answer rather than a smaller one. Generated with the tables above. The first names the
# actions that read; the second is for an op with no action, naming the params that select rather than write, so a call
# naming nothing else asks for no change.
READBACK_READ_ACTIONS = {
    "set_keying_set": ("LIST",),
    "set_mesh_attribute": ("READ",),
}
READBACK_READ_WHEN_ONLY = {
    "set_freestyle": ("viewLayer",),
}


def _execute(request):
    """Run one request - recorded in the call journal BEFORE it runs, and closed after with ok or its error.

    THE JOURNAL'S ONE DOOR. Both callers - the drain timer and the headless inline path - come through here, so nothing
    reaches an op unrecorded: an unknown endpoint, a malformed readBack and a batch are journaled exactly like a call
    that works, and a refusal carries its error. An exception that escapes (SystemExit from run_python's code) still
    closes its record on the way out. See journal.py.
    """
    started = time.perf_counter()
    endpoint = params = agent = None
    if isinstance(request, dict):
        endpoint = request.get("endpoint") or request.get("op") or request.get("type")
        params = request.get("params")
        agent = journal.clean_agent(request.get("agent"))
    seq = journal.call_start(endpoint, params, agent=agent)
    response, escaped = None, None
    from . import ops_common
    deadline = request.get("deadline") if isinstance(request, dict) else None
    ops_common.CALL_DEADLINE[0] = (float(deadline) if isinstance(deadline, (int, float))
                                   and not isinstance(deadline, bool) and math.isfinite(deadline) else None)
    try:
        response = _late_refusal(endpoint) or _dispatch(request, started)
    except BaseException as exc:                                  # noqa: BLE001 - recorded, then re-raised untouched
        escaped = exc
        raise
    finally:
        ops_common.CALL_DEADLINE[0] = None
        ok = isinstance(response, dict) and response.get("ok") is True
        err = None
        if escaped is not None:
            err = "%s escaped the op: %s" % (type(escaped).__name__, escaped)
        elif not ok and isinstance(response, dict):
            err = response.get("error") or response.get("stoppedBecause")
        from . import ops_preview as preview
        journal.call_end(seq, endpoint, ok, err, (time.perf_counter() - started) * 1000.0, bulk=preview.take_journal())
        artifacts.flush(endpoint, ok)
    return response


def _late_refusal(endpoint):
    """A request whose caller has stopped waiting is refused, never run (ops_common.CALL_DEADLINE).

    THE CALLER WAS ALREADY TOLD IT FAILED. FOUND 2026-11-07: a background Blender serves one connection at a time, and a
    bake_texture_set start sat in the accept backlog behind another client's open socket past its caller's 600 s
    timeout; it ran 23 minutes later and baked a session the caller had changed in the meantime - roughness with black
    holes. Run late, a write acts on state its caller no longer expects, and a read answers nobody. Journaled as a
    refusal like any other.
    """
    from . import ops_common
    late = ops_common.caller_gave_up()
    if late is None:
        return None
    return {"ok": False, "endpoint": endpoint, "callerGaveUp": True, "nothingWritten": True,
            "secondsLate": round(late, 1),
            "error": "'%s' was not run: its caller stopped waiting %.1f s ago (the request reached Blender's main "
                     "thread after the caller's deadline - it queued behind other work, in a background Blender "
                     "behind another client's open connection). That caller was already told it timed out, so "
                     "running it now would act on whatever the session holds by then. NOTHING was done; send it "
                     "again." % (endpoint, late)}


def _dispatch(request, started):
    endpoint = request.get("endpoint") or request.get("op") or request.get("type")
    params = request.get("params") or {}
    if not isinstance(params, dict):
        return {"ok": False, "error": "params must be a JSON object, got %s"
                                      % type(params).__name__}
    if not endpoint:
        return {"ok": False, "error": "request has no 'endpoint' (aliases: op, type). "
                                      "Call 'ping' for the op list."}

    # readBack IS TAKEN OFF BEFORE DISPATCH. Ops reject unknown params individually, so a flag
    # left in `params` would be refused by every op that does not declare it - which is all of
    # them. It is a transport-level request about the RESPONSE, not an argument to any op.
    read_back = True
    if "readBack" in params:
        read_back = params.pop("readBack")
        if not isinstance(read_back, bool):
            return {"ok": False, "endpoint": endpoint,
                    "error": "readBack must be true or false, got %s. It suppresses the "
                             "SERIALISED state in the response and never turns verification "
                             "off - the write is still read back and still refused if the read "
                             "disagrees." % type(read_back).__name__}
        if read_back is False and endpoint in READBACK_REFUSED:
            return {"ok": False, "endpoint": endpoint,
                    "error": "readBack:false is refused on '%s': its payload IS what you "
                             "asked for, so suppressing it would answer nothing while still "
                             "reporting ok:true. Use it on the WRITE ops, whose read-back is "
                             "an echo of state you can read again when you need it."
                             % endpoint}
    # READ BEFORE THE OP RUNS, because an op may take its params off as it reads them. An empty value asks for no change.
    selecting = READBACK_READ_WHEN_ONLY.get(endpoint)
    asks_nothing = selecting is not None and not any(v not in (None, [], {}) for k, v in params.items()
                                                     if k not in selecting)

    table = _op_table()
    # ONE UNDO STEP PER CALL THAT CAN CHANGE THE SCENE - see ops_undo. A base step before the first such call, one after
    # each, on refusal too: a refusal that wrote before it refused is otherwise a write the next undo silently loses.
    from . import ops_undo
    # BATCH IS HANDLED HERE RATHER THAN IN AN ops_ MODULE, because it needs the op TABLE and an
    # ops module importing this one would be circular. It is also what keeps a batch ONE job on the
    # single-threaded queue, which server.py's own header insists must not be "simplified" away.
    from . import cap
    if endpoint == "batch":
        # THE BATCH PREVIEW (ops_preview.batch_spec; docs/15 section 5, 2026-10-09): dryRun:true plans every op with its
        # own plan and writes nothing; the previewId it answers commits them all, rolled back together on a mismatch. A
        # previewId the change cap made (a target fingerprint) still commits through cap.take_commit below.
        from . import ops_preview as preview
        try:
            bulk = preview.mode(params)
            if bulk == "preview":
                return jsonable(dict(preview.answer_preview("batch", params, preview.batch_spec(table)),
                                     endpoint="batch", elapsedMs=round((time.perf_counter() - started) * 1000.0, 2)))
        except MifOpError as exc:
            return jsonable(dict(getattr(exc, "payload", None) or {}, ok=False, endpoint="batch", error=str(exc),
                                 nothingWritten=True, elapsedMs=round((time.perf_counter() - started) * 1000.0, 2)))
        if bulk == "commit" and preview.is_bulk_batch(params.get("previewId")):
            ops_undo.before_write("batch")
            try:
                result = preview.answer_commit("batch", params, preview.batch_runner(table), preview.batch_spec(table))
            except MifOpError as exc:
                return jsonable(dict(ok=False, endpoint="batch", error=str(exc), nothingWritten=True,
                                     elapsedMs=round((time.perf_counter() - started) * 1000.0, 2)))
            if not (result.get("nothingWritten") or result.get("rollbackComplete")):
                ops_undo.after_write("batch", "batch of %d (previewed)" % len(result.get("result", {}).get("results")
                                                                            or []), result=result)
            return jsonable(dict(result, endpoint="batch", elapsedMs=round((time.perf_counter() - started) * 1000.0, 2)))
        ops_undo.before_write("batch")
        try:
            # {previewId} alone commits the batch a capped call previewed: the stored ops, exactly (cap.take_commit).
            committing = "previewId" in params
            if committing:
                params = cap.take_commit("batch", params)
            result = _run_batch(params, table, started, committing=committing)
        except MifOpError as exc:
            ops_undo.after_write("batch")
            return jsonable(dict(getattr(exc, "payload", None) or {}, ok=False, endpoint="batch", error=str(exc),
                                 elapsedMs=round((time.perf_counter() - started) * 1000.0, 2)))
        pushed, why_not = ops_undo.after_write("batch", "batch of %d" % len(params.get("ops") or []), result=result)
        result["atomicNote"] = _atomic_note(pushed, why_not)
        result["endpoint"] = "batch"
        result["elapsedMs"] = round((time.perf_counter() - started) * 1000.0, 2)
        return jsonable(result)

    fn = table.get(endpoint)
    if fn is None:
        near = _closest_ops(endpoint, table)
        return {"ok": False,
                "endpoint": endpoint,
                "closest": near or None,
                "error": "unknown endpoint '%s'.%s Known: %s"
                         % (endpoint, (" Closest: %s." % ", ".join(near)) if near else "", ", ".join(sorted(table)))}

    # THE BULK PREVIEW (ops_preview.py; docs/15 section 5): an op with a plan takes dryRun:true as a preview - which writes
    # nothing, so it pushes no undo step - and {previewId} alone as its commit. The keys are read strictly first.
    from . import ops_preview as preview
    plan_spec = preview.spec(endpoint)
    bulk = None
    if plan_spec is not None:
        try:
            bulk = preview.mode(params)
            if bulk == "preview":
                return jsonable(dict(preview.answer_preview(endpoint, params, plan_spec), endpoint=endpoint,
                                     elapsedMs=round((time.perf_counter() - started) * 1000.0, 2)))
        except MifOpError as exc:
            return jsonable(dict(getattr(exc, "payload", None) or {}, ok=False, endpoint=endpoint, error=str(exc),
                                 nothingWritten=True, elapsedMs=round((time.perf_counter() - started) * 1000.0, 2)))

    ops_undo.before_write(endpoint)
    try:
        if bulk == "commit":
            result = preview.answer_commit(endpoint, params, fn, plan_spec)
            if result.get("nothingWritten") or result.get("rollbackComplete"):
                # Nothing of it stands: an undo step now would be a step on nothing, which spends one of 32.
                return jsonable(dict(result, elapsedMs=round((time.perf_counter() - started) * 1000.0, 2)))
            ops_undo.after_write(endpoint, result=result)
            return jsonable(dict(result, elapsedMs=round((time.perf_counter() - started) * 1000.0, 2)))
        # THE CHANGE CAP (cap.py; docs/15 section 6): an op that would change more than the file's cap of things writes
        # nothing - a preview to commit with {previewId}, or a refusal saying how to go on.
        if endpoint in cap.PREVIEWABLE and "previewId" in params:
            params = cap.take_commit(endpoint, params)
        else:
            cap.gate(endpoint, params)
        result = fn(params)
    except MifOpError as exc:
        ops_undo.after_write(endpoint)
        # A deliberate, actionable refusal from an op. Not a bug -- do not
        # print a traceback for these, the message IS the diagnosis. A capped call's preview rides in its payload.
        return jsonable(dict(getattr(exc, "payload", None) or {}, ok=False, endpoint=endpoint, error=str(exc),
                             elapsedMs=round((time.perf_counter() - started) * 1000.0, 2)))
    except Exception as exc:  # noqa: BLE001
        ops_undo.after_write(endpoint)
        traceback.print_exc()
        return {"ok": False, "endpoint": endpoint,
                "error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": traceback.format_exc(),
                "elapsedMs": round((time.perf_counter() - started) * 1000.0, 2)}
    ops_undo.after_write(endpoint, result=result if isinstance(result, dict) else {})

    out = {"ok": True, "endpoint": endpoint}
    if isinstance(result, dict):
        out.update(result)
    elif result is not None:
        out["result"] = result
    # SUPPRESSED HERE, AFTER THE OP HAS RUN AND VERIFIED. The op did its read-back and would
    # have raised if the read disagreed; all that is dropped is the serialisation of state the
    # caller said it does not want shipped. `readBackSuppressed` names what went, so a response
    # is never quietly smaller than its documentation.
    if read_back is False and not asks_nothing and out.get("action") not in READBACK_READ_ACTIONS.get(endpoint, ()):
        dropped = [k for k in READBACK_SUPPRESSES.get(endpoint, ()) if k in out]
        for k in dropped:
            out.pop(k, None)
        if dropped:
            out["readBackSuppressed"] = dropped
    out["elapsedMs"] = round((time.perf_counter() - started) * 1000.0, 2)
    # COERCED AT THE DOOR, not on one branch of it. jsonable's own docstring says a response that
    # cannot be serialised is "a silent hang from the caller's point of view" - and it was applied
    # only to the `result` key, which is the RARE path. An op returning a dict, which is nearly all
    # of them, went to json.dumps untouched.
    #
    # What that cost: a NaN anywhere in a response reached the wire as bare `NaN`, which is not
    # valid JSON. Python's json.loads accepts it, so the Python client never noticed; a strict
    # parser rejects the whole frame. jsonable already turns a non-finite float into a string and
    # was simply never asked to.
    return jsonable(out)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class MifBlenderServer:
    def __init__(self, port=DEFAULT_PORT, token="dev", job_timeout=DEFAULT_JOB_TIMEOUT,
                 verbose=True):
        self.host = HOST
        self.port = int(port)
        self.token = token or ""
        self.job_timeout = float(job_timeout)
        self.verbose = bool(verbose)
        self.running = False
        self.socket = None
        self.server_thread = None
        self._clients = 0

    # -- logging -----------------------------------------------------------
    def _log(self, *parts):
        if self.verbose:
            print("[MifBlender]", *parts)

    # -- lifecycle ---------------------------------------------------------
    @staticmethod
    def _set_exclusive(sock):
        """Make a second bind of the same address FAIL, on either platform.

        SO_REUSEADDR does not mean the same thing on Windows as it does on
        POSIX. On Windows it lets a SECOND socket bind an address another socket
        is already listening on -- VERIFIED on this box, Python 3.11/win32: two
        sockets both bound 127.0.0.1 and both listen()ed, no error either time.
        Two Blender windows with the addon auto-starting would therefore BOTH log
        "listening", both show green in the N-panel, and the MCP would reach a
        nondeterministic one. "My edits went to the other Blender" is a whole day.

        SO_EXCLUSIVEADDRUSE is the Windows-only opposite: the second bind fails
        with WSAEADDRINUSE (10048), which the OSError arm in start() already
        handles -- it just never used to fire.

          MEASURED on this box, same session: with SO_EXCLUSIVEADDRUSE the second
          bind raises OSError errno 10048; a plain no-option socket is refused
          too. And the TIME_WAIT worry does NOT bite: closing an accepted
          connection server-side and immediately rebinding the listening port
          with SO_EXCLUSIVEADDRUSE succeeded, so stop-then-start still works.

        On POSIX, SO_REUSEADDR already refuses a second bind of a LISTENING
        address (it only relaxes TIME_WAIT), so the existing behaviour is kept.
        """
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def start(self):
        if self.running:
            self._log("already running on %s:%d" % (self.host, self.port))
            return True

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._set_exclusive(sock)
            sock.bind((self.host, self.port))
            sock.listen(4)
        except OSError as exc:
            self._log("FAILED to bind %s:%d -- %s" % (self.host, self.port, exc))
            self._log("       another MifBlender (or something else) already owns that "
                      "port -- most likely a SECOND Blender window with this addon "
                      "enabled. Close it, or change the port in Preferences > Add-ons > "
                      "MifBlender. The N-panel of whichever Blender DID get the port "
                      "shows its pid; compare it against bl_status's `pid`.")
            try:
                sock.close()
            except Exception:
                pass
            self.socket = None
            self.running = False
            return False

        self.socket = sock
        self.running = True
        # The call journal's session header - this pid, this port, this Blender - before the first call can arrive.
        journal.open_session(self.port)
        # WHERE THIS LISTENER IS, for the MCP server to find (registry.py, 2026-10-12): a MifBlender on its own port preference
        # was unreachable to a server that only knew 8792. Written before a background accept loop blocks; removed in stop()
        # and, for a Blender that quits without stopping, at exit.
        self._register()
        if not self.token:
            self._log("WARNING: token is EMPTY -- any loopback process may drive "
                      "this Blender. Set one in the addon preferences.")
        self._log("listening on %s:%d (protocol v%d, pid %d)"
                  % (self.host, self.port, PROTOCOL_VERSION, os.getpid()))

        if bpy.app.background:
            # No event loop in -b, so the drain timer will never fire. Run
            # everything on this (the main) thread. THIS BLOCKS.
            self._log("background mode: blocking accept loop on the main thread")
            self._server_loop()
        else:
            self.server_thread = threading.Thread(
                target=self._server_loop, name="MifBlenderAccept", daemon=True)
            self.server_thread.start()
        return True

    def _register(self):
        try:
            pkg = sys.modules.get(__package__ or "")
            version = getattr(pkg, "bl_info", {}).get("version") if pkg else None
            registry.register(self.port, os.getpid(),
                              blenderVersion=getattr(bpy.app, "version_string", None),
                              addonVersion=".".join(str(v) for v in version) if version else None,
                              blendFile=bpy.data.filepath or "",
                              background=bool(bpy.app.background),
                              protocol=PROTOCOL_VERSION)
            _REGISTERED[0] = (self.port, os.getpid())
        except Exception:
            pass   # discovery must never cost the listener

    def stop(self):
        self.running = False
        if _REGISTERED[0] is not None:
            registry.unregister(*_REGISTERED[0])
            _REGISTERED[0] = None
        if self.socket is not None:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
        if self.server_thread is not None:
            try:
                if self.server_thread.is_alive():
                    # The listening socket has a 1 s timeout, so the loop
                    # notices self.running went False within ~1 s.
                    self.server_thread.join(timeout=2.0)
            except Exception:
                pass
            self.server_thread = None
        self._log("stopped")

    # -- accept loop -------------------------------------------------------
    def _server_loop(self):
        # 1 s timeout on the LISTENING socket is what makes stop() work: without
        # it accept() blocks forever and the thread can never notice `running`.
        try:
            self.socket.settimeout(1.0)
        except Exception:
            return

        while self.running:
            try:
                try:
                    client, addr = self.socket.accept()
                except socket.timeout:
                    continue
                except OSError:
                    # socket closed by stop()
                    break

                if addr[0] != "127.0.0.1":
                    # Cannot normally happen (we bind loopback) -- belt only.
                    try:
                        framing.send_json_message(
                            client, {"ok": False, "error": "non-loopback caller rejected"})
                        client.close()
                    except Exception:
                        pass
                    continue

                self._clients += 1
                self._log("client connected from %s:%d" % addr)
                if bpy.app.background:
                    self._handle_client(client)
                else:
                    threading.Thread(target=self._handle_client, args=(client,),
                                     name="MifBlenderClient", daemon=True).start()
            except Exception as exc:  # noqa: BLE001
                if not self.running:
                    break
                self._log("accept loop error: %s" % exc)
                time.sleep(0.5)
        self._log("accept loop exited")

    # -- per-client loop ---------------------------------------------------
    def _handle_client(self, client):
        """SOCKET THREAD. Zero bpy calls below this line -- see the module docstring.

        (bpy.app.background is read, but that is an immutable process-level flag,
        not scene data.)
        """
        background = bpy.app.background
        try:
            client.settimeout(None)  # block; the accept socket owns the stoppability
            while self.running:
                try:
                    request = framing.receive_framed_json(client)
                except json.JSONDecodeError as exc:
                    # Must precede the ValueError arm below -- JSONDecodeError IS a
                    # ValueError, and a mis-encoded body deserves a real answer
                    # rather than a silent hang-up.
                    self._reply(client, {"ok": False, "error": "malformed JSON body: %s" % exc})
                    break
                except (ConnectionError, OSError, ValueError) as exc:
                    self._log("client gone / bad frame: %s" % exc)
                    break

                if not isinstance(request, dict):
                    self._reply(client, {"ok": False,
                                         "error": "frame must be a JSON object"})
                    continue

                if not self._auth_ok(request):
                    self._reply(client, {"ok": False,
                                         "error": "bad or missing token (send it as the "
                                                  "'token' field; it is set in the addon "
                                                  "preferences)"})
                    break

                if background:
                    # Main thread already -- run it here, timers are dead in -b.
                    response = _execute(request)
                else:
                    response = self._run_on_main_thread(request)

                if not self._reply(client, response):
                    break
        except Exception as exc:  # noqa: BLE001
            self._log("client handler error: %s" % exc)
            traceback.print_exc()
        finally:
            try:
                client.close()
            except Exception:
                pass
            self._clients = max(0, self._clients - 1)

    def _run_on_main_thread(self, request):
        """Hand the request to the main thread and WAIT for it.

        Waiting (rather than firing and forgetting) is deliberate: the reply must
        not be sent until the bpy work has actually completed. A fire-and-forget
        'DONE' would tell the caller a file is on disk before Blender has written
        it -- which for the mesh round-trip means Unreal reimports a stale file.
        """
        # A caller that knows this one job is long (a 4k bake, a long render) sends its own
        # jobTimeout, in seconds, next to the endpoint: the session's job_timeout stays the
        # guard for every other call, and the caller's own transport must wait longer still.
        wait = self.job_timeout
        asked = request.get("jobTimeout")
        if isinstance(asked, (int, float)) and not isinstance(asked, bool) and asked > 0:
            wait = min(float(asked), 86400.0)
        job = _Job(request)
        _JOBS.put(job)
        if job.done.wait(timeout=wait):
            return job.response
        job.cancelled = True  # skipped if it has not started yet
        return {"ok": False,
                "error": "timed out after %.0fs waiting for Blender's main thread. "
                         "Blender is busy (modal operator, long bake, or a dialog is "
                         "open). The job was cancelled if it had not started; if it "
                         "had, it may still complete -- check the scene before "
                         "retrying. A job known to be long can send 'jobTimeout' "
                         "(seconds) beside its endpoint." % wait}

    def _auth_ok(self, request):
        if not self.token:
            return True  # explicitly unauthenticated; warned about at start()
        supplied = request.get("token")
        if not isinstance(supplied, str):
            return False
        return hmac.compare_digest(supplied, self.token)

    def _reply(self, client, payload):
        try:
            framing.send_json_message(client, payload)
            return True
        except (TypeError, ValueError) as exc:
            # Payload could not be framed. Never let that look like success - and say WHICH of the
            # two reasons it was, because they point at completely different fixes.
            #
            # TOO LARGE IS NOT A BUG IN THE OP. The old message said "op produced a
            # non-serialisable response ... the op must return JSON-safe values" for both cases, so
            # a response that serialised perfectly and was merely bigger than the frame sent the
            # caller looking for a serialisation defect in an op that has none. Measured on 5.0.1:
            # list_objects costs about 169 bytes per object, so the 64 MiB cap arrives near 400,000
            # objects - far off, reachable, and worth naming correctly when it happens.
            too_big = "frame too large" in str(exc)
            if too_big:
                message = ("the response is larger than the %d-byte frame limit (%s). The op and "
                           "its values are fine - there is simply too much of it. Narrow the "
                           "request: ask for one object, one collection or one frame range rather "
                           "than the whole scene."
                           % (framing.MAX_MESSAGE_BYTES, exc))
            else:
                message = ("op produced a non-serialisable response (%s). This is a MifBlender bug "
                           "-- the op must return JSON-safe values." % exc)
            try:
                framing.send_json_message(
                    client,
                    {"ok": False,
                     "endpoint": payload.get("endpoint") if isinstance(payload, dict) else None,
                     "responseTooLarge": too_big,
                     "error": message})
                return True
            except Exception:
                return False
        except Exception as exc:  # noqa: BLE001
            self._log("failed to send reply: %s" % exc)
            return False

    # -- introspection -----------------------------------------------------
    def status(self):
        return {
            "running": self.running,
            "host": self.host,
            "port": self.port,
            # The pid is here so a duplicate Blender is VISIBLE. The N-panel
            # prints it and op_ping returns it, so "which Blender did that edit
            # land in" is one comparison instead of a guess.
            "pid": os.getpid(),
            "clients": self._clients,
            "jobTimeout": self.job_timeout,
            "background": bpy.app.background,
            "authRequired": bool(self.token),
            "protocolVersion": PROTOCOL_VERSION,
        }
