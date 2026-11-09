"""THE CHANGE CAP, Blender's half (docs/15_SAFETY_GATE_AND_JOURNAL.md section 6).

Decided 2026-10-01: any single call that would change MORE than N distinct things must not write. It answers the list
of what it would change instead, and only a commit carrying that answer's previewId applies it - so the user always sees
the list before a mass change, and a person decides. The Unreal plugin does the same through its bulk preview handshake.

WHAT IS COUNTED: distinct data-blocks a call changes, creates or removes - objects, materials, collections, meshes -
never operations. Ten modifiers on one object is one target; a bone is part of its armature; a batch that edits one
object forty times is one target. A batch is counted whole, before its first op runs.

THE CAP: the open .blend's own value (a custom property on its scenes, recorded by project_mode's changeCap - the
smallest when scenes disagree), else MIF_CHANGE_CAP in Blender's environment, else 25. 0 switches it off. A value that
is not a whole number from 0 up is reported and the cap stays on: a typo must not switch it off.

OVER THE CAP:
  - batch answers a PREVIEW: ok:false, capped:true, the targets, a previewId. batch called with {previewId} alone
    commits exactly the stored request, once, within 10 minutes - refused, writing nothing, when a target named then
    no longer exists (or a name it would create is taken now).
  - an op with a bulk preview plan (ops_preview.specs(): transform_objects, purge_orphans, fit_mesh and every other op
    with a PREVIEW entry) answers its own preview - every change with its value before - and commits through that
    handshake: planned again, refused on any difference, read back, rolled back on a mismatch.
  - the other many-target ops (COUNTERS below without a preview) REFUSE, naming the count and the two ways on: split
    the call, or have the user raise the file's cap. separate_mesh is counted from what the separate WILL make (the
    mesh's loose pieces or face material indices, separate_parts), since 2026-10-11.
Under the cap every op runs exactly as before.
"""
import itertools
import os
import time

import bpy

from .ops_common import MifOpError

CAP_KEY = "mifbridge_change_cap"
DEFAULT_CAP = 25
PREVIEW_TTL = 600.0
_MAX_PREVIEWS = 64
_SHOWN = 200                       # targets listed inline in an answer


class MifCapped(MifOpError):
    """A refusal that carries fields: the dispatcher answers ok:false with the message AND these (server._dispatch)."""

    def __init__(self, message, payload):
        MifOpError.__init__(self, message)
        self.payload = payload


# ---------------------------------------------------------------------------------------------------------- the setting
def _whole(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def change_cap():
    """{cap, on, source, error?} - the cap in force for the open file."""
    out = {"cap": DEFAULT_CAP, "source": "default"}
    errors = []
    env = os.environ.get("MIF_CHANGE_CAP", "").strip()
    if env:
        if env.isdigit() and len(env) <= 9:
            out.update(cap=int(env), source="env")
        else:
            errors.append("MIF_CHANGE_CAP is %r, not a whole number from 0 (off) up - the cap stays at %d"
                          % (env, out["cap"]))
    try:
        values = [s.get(CAP_KEY) for s in bpy.data.scenes if s.get(CAP_KEY) is not None]
    except Exception as exc:                                       # noqa: BLE001 - restricted context: report, keep the cap
        values = []
        errors.append("the file's changeCap could not be read (%s)" % exc)
    good = [v for v in values if _whole(v)]
    if len(good) != len(values):
        errors.append("a scene's %s is not a whole number from 0 (off) up - it is ignored" % CAP_KEY)
    if good:
        out.update(cap=min(good), source="file")
    out["on"] = out["cap"] > 0
    if errors:
        out["error"] = "; ".join(errors)
    return out


def record(cap):
    """project_mode's changeCap: on every scene of the open file, as the generative mode is."""
    if not _whole(cap):
        raise MifOpError("project_mode: changeCap must be a whole number from 0 (off) up, not %r. Nothing was changed."
                         % (cap,))
    scenes = list(bpy.data.scenes)
    if not scenes:
        raise MifOpError("project_mode: this file has no scene to record the change cap on. Nothing was changed.")
    for scene in scenes:
        scene[CAP_KEY] = cap


# ------------------------------------------------------------------------------------------------------------ counting
_READ_PREFIXES = ("list_", "describe_")
_READ_OPS = frozenset((
    "artifact_inventory", "ping", "scene_info", "object_info", "file_info", "mesh_stats", "inspect_node_output", "image_stats", "ray_cast", "closest_point_on_mesh", "face_info",
    "vertex_info", "objects_overlap", "audit_placement", "points_inside", "compare_to_reference", "evaluate_at_frame", "mesh_quality",
    "undo_history", "call_log", "render_status", "render_info", "uv_info", "physics_info", "compositor_info",
    "world_info", "select_edges", "select_faces", "project_mode", "gen_status", "save_file", "export_mesh",
    "export_scene", "export_actions", "export_material_textures", "export_camera", "export_groom", "heightmap_from_mesh", "frame_viewport", "set_viewport_shading", "set_viewport_view", "undo", "redo", "open_file"))
_CREATE_PREFIXES = ("create_", "build_", "gen_", "import_")
# A material NAMED by an op that assigns it is not changed; set_material_* edits the material it names.
_TARGET_KEYS = ("object", "objects", "target", "source", "sources", "armature", "camera", "light", "collection", "mesh")


def _names(value):
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str) and v]
    return []


def _type_filter(params):
    t = params.get("type", params.get("objectType"))
    return str(t).upper() if t is not None else None


def _count_clear_scene(params):
    want = _type_filter(params)
    return ["object:" + o.name for o in bpy.data.objects if not want or o.type == want]


def _count_delete_object(params):
    names = params.get("objects")
    if names is None:
        names = params.get("object", params.get("name"))
    return ["object:" + n for n in _names(names)]


def _count_purge_orphans(params):
    only = params.get("only")
    if isinstance(only, list):
        return ["orphan:" + str(n) for n in only]
    from .ops_file import _orphans
    by_type, _total = _orphans()
    return ["%s:%s" % (kind, n) for kind, names in by_type.items() for n in names]


def _count_delete_collection(params):
    name = params.get("collection", params.get("name"))
    coll = bpy.data.collections.get(name) if isinstance(name, str) else None
    keys = ["collection:%s" % name]
    if coll is not None and params.get("deleteObjects") is True:
        keys += ["object:" + o.name for o in coll.all_objects]
        keys += ["collection:" + c.name for c in coll.children_recursive]
    return keys


def _count_join_objects(params):
    # THE JOIN WRITES ONE OBJECT, the target; a source counts only as a removal, when the join consumes it (2026-11-07,
    # as the Unreal plugin's merge_static_mesh_actors counts since R81: the mesh and its actor, plus each piece only when
    # keepInputs false destroys it). keepSources joins copies and leaves every source, so it is the target alone.
    keys = ["object:" + n for n in _names(params.get("target"))]
    if params.get("keepSources") is not True:
        keys += ["object:" + n for n in _names(params.get("objects")) + _names(params.get("sources"))]
    return keys


def _count_import_mesh(params):
    keys = ["new:import_mesh"]
    if params.get("clearScene") is True or params.get("clear_scene") is True:
        from .ops_mesh import clear_targets   # the op's own list, so the cap counts what the clear deletes
        scope = str(params.get("clearScope") or "scene").strip().lower()
        doomed, _kept = clear_targets(bpy.context.scene, scope if scope in ("scene", "file") else "scene")
        keys += ["object:" + o.name for o in doomed]
    return keys


def _count_append_from_file(params):
    if params.get("listOnly") is True:
        return []
    keys = []
    for kind in ("objects", "collections", "materials", "nodeGroups", "actions", "worlds", "fonts"):
        keys += ["%s:%s" % (kind, n) for n in _names(params.get(kind))]
    return keys


def _count_transform_objects(params):
    return [] if params.get("dryRun") is True else ["object:" + n for n in _names(params.get("objects"))]


def _count_fit_mesh(params):
    # The fitted object and every carried one move; the target is only read.
    if params.get("dryRun") is True:
        return []
    return ["object:" + n for n in _names(params.get("object")) + _names(params.get("carry"))]


def separate_parts(mesh, mode):
    """How many objects bpy.ops.mesh.separate leaves a mesh as - the source plus the new ones - read before it runs.

    MEASURED 2026-10-11 on 3.6, 4.4 and 5.0 against the operator itself, not taken from its docs:
      loose     every connected piece of verts and edges, a lone vertex included (3 cubes and 2 lone verts are 5).
      material  one per DISTINCT face material_index in use, the raw index even past the slot count (indices 5 and 6 on
                a one-slot mesh made 2); loose verts and edges stay with the source, and a mesh with no faces makes 0.
    Returns 1 when the mesh has nothing to split on (the op writes nothing)."""
    if mode == "material":
        polys = len(mesh.polygons)
        if not polys:
            return 1
        idx = [0] * polys
        mesh.polygons.foreach_get("material_index", idx)
        return max(1, len(set(idx)))
    nv = len(mesh.vertices)
    if nv < 2:
        return 1
    flat = [0] * (2 * len(mesh.edges))
    mesh.edges.foreach_get("vertices", flat)
    parent = list(range(nv))
    pieces = nv
    for i in range(0, len(flat), 2):
        a, b = flat[i], flat[i + 1]
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a != b:
            parent[a] = b
            pieces -= 1
    return pieces


def _count_separate_mesh(params):
    # WHAT THE ENGINE WILL MAKE, PREDICTED (spec "A cap for the calls whose output is not known before they run"): the
    # source, which loses geometry, and every new object. Mode selection makes one object and leaves the source whole
    # with duplicate true.
    name = params.get("object", params.get("name"))
    obj = bpy.data.objects.get(name) if isinstance(name, str) else None
    if obj is None or obj.type != "MESH":
        return []                                       # the op refuses it, naming the object
    mode = str(params.get("mode") or "loose").lower()
    if mode == "selection":
        return ([] if params.get("duplicate") is True else ["object:" + obj.name]) + ["new:separate_mesh.part"]
    if mode not in ("loose", "material"):
        return []
    parts = separate_parts(obj.data, mode)
    if parts <= 1:
        return []
    return ["object:" + obj.name] + ["new:%s.part%d" % (obj.name, i) for i in range(1, parts)]


# op -> what it changes, read off its request and the open file before it runs. Keys are "<kind>:<name>", so one object
# named by two ops of a batch counts once.
COUNTERS = {
    "clear_scene": _count_clear_scene,
    "delete_object": _count_delete_object,
    "purge_orphans": _count_purge_orphans,
    "delete_collection": _count_delete_collection,
    "join_objects": _count_join_objects,
    "import_mesh": _count_import_mesh,
    "append_from_file": _count_append_from_file,
    "transform_objects": _count_transform_objects,
    "fit_mesh": _count_fit_mesh,
    "separate_mesh": _count_separate_mesh,
}
# batch's preview is the cap's own (a fingerprint of its targets); every op with a plan in ops_preview.py previews through
# the bulk preview handshake instead (gate below).
PREVIEWABLE = ("batch",)


def _generic(op, params, index):
    """A batch op no counter knows: the objects it names, and one new data-block for an op that makes one."""
    if op in _READ_OPS or op.startswith(_READ_PREFIXES) or params.get("dryRun") is True:
        return []
    if op in ("link_objects", "unlink_objects"):
        # Membership is the collection's: as Unreal counts add_to_collection, one target however many objects it lists.
        return ["collection:" + n for n in _names(params.get("collection"))] or ["new:ops[%d].%s" % (index, op)]
    if op == "create_collection":
        return ["collection:" + n for n in _names(params.get("name"))] or ["new:ops[%d].%s" % (index, op)]
    keys = []
    for k in _TARGET_KEYS:
        kind = "collection" if k == "collection" else "object"
        keys += ["%s:%s" % (kind, n) for n in _names(params.get(k))]
    if op.startswith("set_material_") and op != "set_material_slots":
        keys += ["material:" + n for n in _names(params.get("material"))]
    if op.startswith(_CREATE_PREFIXES) or op in ("add_sound",):
        made = params.get("name") or params.get("objectName")
        kind = "collection" if op == "create_collection" else "material" if op == "create_material" else "object"
        keys.append("%s:%s" % (kind, made) if isinstance(made, str) and made else "new:ops[%d].%s" % (index, op))
    return keys


def _extension_targets(op, params):
    """An op a separate add-on registered with capKeys (extensions.py): the data-blocks those keys name. None when it is
    not one, or declares no capKeys - then it is counted as any op with no counter is."""
    from . import extensions
    return extensions.cap_targets(op, params)


def targets_of(op, params, index=None):
    """The distinct targets one op would change - "<kind>:<name>" keys."""
    try:
        if op in COUNTERS:
            return COUNTERS[op](params)
        ext = _extension_targets(op, params)
        if ext is not None:
            return ext
        if index is not None:
            return _generic(op, params, index)
    except Exception:                                              # noqa: BLE001 - a request the op itself will refuse
        return []
    return []


def batch_targets(ops):
    keys, per_op = [], []
    for i, entry in enumerate(ops):
        if not isinstance(entry, dict):
            continue
        name = entry.get("op") or entry.get("endpoint")
        sub = entry.get("params") or {}
        if not isinstance(name, str) or not isinstance(sub, dict):
            continue
        mine = targets_of(name, sub, index=i)
        per_op.append({"index": i, "op": name, "targets": len(set(mine))})
        keys += mine
    seen = []
    for k in keys:
        if k not in seen:
            seen.append(k)
    return seen, per_op


# ------------------------------------------------------------------------------------------------------- previews
_PREVIEWS = {}
_SERIAL = itertools.count(1)


def _fingerprint(op, params, keys):
    """What a commit must still find: each named data-block's existence and type, and for transform_objects each
    object's world matrix. A name the call would create is only checked for not having been taken since."""
    out = []
    for k in keys:
        kind, _sep, name = k.partition(":")
        holder = {"object": bpy.data.objects, "material": bpy.data.materials, "collection": bpy.data.collections,
                  "objects": bpy.data.objects, "materials": bpy.data.materials,
                  "collections": bpy.data.collections}.get(kind)
        found = holder.get(name) if holder is not None else None
        row = [k, found is not None, getattr(found, "type", None)]
        if op == "transform_objects" and found is not None:
            row.append([round(v, 5) for col in found.matrix_world for v in col])
        out.append(row)
    return out


def _prune():
    """Drop previews an hour past expiry, then the oldest while the store is full. Returns the ids evicted for room."""
    now = time.monotonic()
    for pid in [p for p, r in _PREVIEWS.items() if r["expires"] + 3600.0 < now]:
        del _PREVIEWS[pid]
    evicted = []
    while len(_PREVIEWS) >= _MAX_PREVIEWS:
        oldest = min(_PREVIEWS, key=lambda p: _PREVIEWS[p]["created"])
        evicted.append(oldest)
        del _PREVIEWS[oldest]
    return evicted


def new_id():
    """bpv-<Blender pid>-<n>: the pid names the Blender a preview lives in, so a commit sent to another one says so."""
    return "bpv-%d-%d" % (os.getpid(), next(_SERIAL))


def _whose(cap):
    return {"file": "this file's cap", "env": "the cap MIF_CHANGE_CAP sets"}.get(cap["source"], "the default cap")


def gate(op, params):
    """Before an op runs: None to run it, or MifCapped raised - a preview or a refusal. Never writes."""
    if op not in COUNTERS and op != "batch" and _extension_targets(op, params) is None:
        return None
    cap = change_cap()
    if not cap["on"]:
        return None
    if op == "batch":
        keys, per_op = batch_targets(params.get("ops") or [])
    else:
        keys, per_op = list(dict.fromkeys(targets_of(op, params))), None
    if len(keys) <= cap["cap"]:
        return None
    capinfo = {"cap": cap["cap"], "source": cap["source"], "targets": len(keys)}
    if cap.get("error"):
        capinfo["error"] = cap["error"]
    payload = {"capped": True, "nothingWritten": True, "changeCap": capinfo,
               "targets": [k for k in keys[:_SHOWN]], "targetsTotal": len(keys)}
    if per_op is not None:
        payload["perOp"] = per_op
    from . import ops_preview as preview
    plan_spec = preview.spec(op)
    if plan_spec is not None:
        # AN OP WITH A PLAN (ops_preview.py) ANSWERS THE BULK PREVIEW ITSELF - every change with its value before - and its
        # previewId commits through the same handshake as a dryRun:true preview does.
        planned = preview.preview_payload(op, params, plan_spec)
        planned.update(payload)
        pid = planned.get("previewId")
        if pid is None:
            return None                                # nothing to change after all: the op runs, and writes nothing
        raise MifCapped(
            "this would change %d items, over %s of %d, so nothing was written - review the list (changes), then "
            "commit with {\"previewId\": \"%s\"} on %s, and nothing else. To do it in smaller steps, send at most %d "
            "items per call." % (len(keys), _whose(cap), cap["cap"], pid, op, cap["cap"]), planned)
    if op in PREVIEWABLE:
        _prune()
        pid = new_id()
        now = time.monotonic()
        _PREVIEWS[pid] = {"op": op, "params": dict(params), "keys": keys, "fingerprint": _fingerprint(op, params, keys),
                          "created": now, "expires": now + PREVIEW_TTL, "spent": False}
        payload.update(previewId=pid, expiresInSeconds=int(PREVIEW_TTL), dryRun=True)
        raise MifCapped(
            "this would change %d items, over %s of %d, so nothing was written - review the list (targets), then "
            "commit with {\"previewId\": \"%s\"} on %s, and nothing else. To do it in smaller steps, send at most %d "
            "items per call." % (len(keys), _whose(cap), cap["cap"], pid, op, cap["cap"]), payload)
    if op == "import_mesh":
        raise MifCapped(_import_mesh_refusal(keys, cap, payload, params), payload)
    if op == "separate_mesh":
        # ONE SEPARATE CANNOT BE SPLIT into smaller calls, so "split it" is not the way on here.
        payload["newObjects"] = len(keys) - 1
        raise MifCapped(
            "separate_mesh mode %s would leave '%s' as %d objects (%d new), over %s of %d, so nothing was written. Take "
            "the pieces off a few at a time with mode selection (faces), or ask the user to raise the file's cap "
            "(mif_project_mode change_cap)." % (str(params.get("mode") or "loose").lower(),
                                                params.get("object", params.get("name")), len(keys), len(keys) - 1,
                                                _whose(cap), cap["cap"]), payload)
    if op == "join_objects":
        # A CONSUMING JOIN REMOVES EVERY SOURCE, and each removal counts. keepSources is the one-write way on.
        payload["removedSources"] = len(keys) - 1
        raise MifCapped(
            "join_objects consumes its %d source(s): with the target that is %d items, over %s of %d, so nothing was "
            "written. keepSources true joins copies and leaves the sources, which counts as the one object it writes; "
            "or join in calls of at most %d sources, or ask the user to raise the file's cap (mif_project_mode "
            "change_cap)." % (len(keys) - 1, len(keys), _whose(cap), cap["cap"], cap["cap"] - 1), payload)
    raise MifCapped(
        "this would change %d items, over %s of %d, and %s has no preview, so nothing was written. Split it into calls "
        "of at most %d items, or ask the user to raise the file's cap (mif_project_mode change_cap)."
        % (len(keys), _whose(cap), cap["cap"], op, cap["cap"]), payload)


def _import_mesh_refusal(keys, cap, payload, params=None):
    """import_mesh's own refusal: an import cannot be split, and its count is clearScene's.

    THE COUNT IS RIGHT AND THE OLD ADVICE WAS NOT (a short film, 2026-10-02: an import into an EMPTY scene was refused
    over objects in other scenes). clearScene deletes every object in the FILE, every scene's (ops_mesh.op_import_mesh,
    measured 2026-09-21), so those objects really are what the call would change, and the cap counting them is the cap
    doing its job. What was wrong was the answer: "split it into calls" means nothing for one import, and it did not say
    the objects were in other scenes. It now says where they are and the way on."""
    here = {o.name for o in bpy.context.scene.objects}
    doomed = [k[len("object:"):] for k in keys if k.startswith("object:")]
    elsewhere = [n for n in doomed if n not in here]
    payload["clearSceneDeletes"] = len(doomed)
    payload["inTargetScene"] = len(doomed) - len(elsewhere)
    payload["inOtherScenes"] = len(elsewhere)
    payload["otherScenesObjects"] = elsewhere[:_SHOWN] or None
    if str((params or {}).get("clearScope") or "scene").strip().lower() != "file":
        return ("import_mesh with clearScene deletes the %d object(s) of scene '%s' before it imports - %d items with "
                "the import, over %s of %d, so nothing was deleted or imported. Import with clearScene false (MCP "
                "clear_scene=False), which changes one item; delete what should go first with delete_object or "
                "clear_scene; or ask the user to raise the file's cap (mif_project_mode change_cap)."
                % (len(doomed), bpy.context.scene.name, len(keys), _whose(cap), cap["cap"]))
    return ("import_mesh with clearScope 'file' deletes every object in the FILE, not only scene '%s', before it imports: %d "
            "object(s), %d of them in other scenes - %d items with the import, over %s of %d, so nothing was deleted or "
            "imported. clearScope 'scene' (the default) clears only this scene; clearScene false (MCP clear_scene=False) "
            "changes one item; delete what should "
            "go first with delete_object or clear_scene; or ask the user to raise the file's cap (mif_project_mode "
            "change_cap)." % (bpy.context.scene.name, len(doomed), len(elsewhere), len(keys), _whose(cap), cap["cap"]))


def claim(op, params):
    """{previewId} alone on a previewable op: (pid, its record), the preview spent from here on, whatever follows. Raises
    when the id cannot commit here, writing nothing and leaving the preview unspent. Shared by the change cap's batch
    commit and the bulk preview's (ops_preview.py)."""
    pid = params.get("previewId")
    extra = sorted(k for k in params if k != "previewId")
    if not isinstance(pid, str) or not pid:
        raise MifOpError("previewId must be the non-empty string a preview answered with. NOTHING was changed.")
    if extra:
        raise MifOpError("a commit takes the previewId and nothing else, and this one also sent %s - the request was "
                         "fixed when it was previewed. NOTHING was changed, and the preview is still usable."
                         % ", ".join(extra))
    rec = _PREVIEWS.get(pid)
    if rec is None:
        parts = pid.split("-")
        if len(parts) >= 3 and parts[1].isdigit() and int(parts[1]) != os.getpid():
            raise MifOpError("previewId %s was made by the Blender with pid %s, and this one is pid %d. Previews live in "
                             "the memory of the Blender that made them - commit it there, or preview again here. "
                             "NOTHING was changed." % (pid, parts[1], os.getpid()))
        raise MifOpError("previewId %s is unknown to this Blender (pid %d): previews live in memory and a restart loses "
                         "them. Call again to preview again. NOTHING was changed." % (pid, os.getpid()))
    if rec["op"] != op:
        raise MifOpError("previewId %s is a preview of %s and commits only on %s. NOTHING was changed."
                         % (pid, rec["op"], rec["op"]))
    if rec["spent"]:
        raise MifOpError("previewId %s was already committed (outcome: %s) - a preview commits once. NOTHING was changed."
                         % (pid, rec.get("outcome") or "committed"))
    if time.monotonic() > rec["expires"]:
        raise MifOpError("previewId %s expired (it lasts %d s). Call again to preview again. NOTHING was changed."
                         % (pid, int(rec.get("ttl", PREVIEW_TTL))))
    rec["spent"] = True
    return pid, rec


def take_commit(op, params):
    """{previewId} alone on a batch the cap previewed: the stored request to run, the preview spent. Raises when it
    cannot commit, writing nothing."""
    pid, rec = claim(op, params)
    now = _fingerprint(op, rec["params"], rec["keys"])
    if now != rec["fingerprint"]:
        drift = [{"target": a[0], "previewed": a[1:], "now": b[1:]} for a, b in zip(rec["fingerprint"], now) if a != b]
        raise MifCapped("the file changed since preview %s: %d target(s) are not as previewed (drift), so NOTHING was "
                        "changed and the preview is spent. Call again to preview again." % (pid, len(drift)),
                        {"drift": drift[:_SHOWN], "nothingWritten": True, "previewId": pid})
    return dict(rec["params"])
