"""MifBlender ops: introspection, scene housekeeping, and the run_python hatch."""

from __future__ import annotations

import fnmatch
import io
import math
import os
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr

import bpy

from .ops_common import (remove_objects, MifOpError, UU_PER_BU, VIS_FLAGS, assign_id_name, children_by_parent,
                         exclude_rows, finite_float,
                         finite_floats, get_object, jsonable,
                         limit_rows, material_slot_names, mesh_counts, object_info, reject_unknown,
                         require_editable, take,
                         take_bool, take_float, take_int, vis_target, visibility_readback)


def _prefs():
    addon = bpy.context.preferences.addons.get(__package__)
    return addon.preferences if addon else None


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------

def op_ping(params):
    """Is this add-on answering, and which one: versions, protocol, pid, open .blend, generative switches, change cap,
    op list, source hash and install check; echo comes back as sent."""
    reject_unknown(params, {"echo"}, "ping")
    from . import server as _server  # local import: avoids an import cycle
    # THE VERSION IS bl_info's. This op kept its own ADDON_VERSION tuple, and both copies sat at 0.1.0 from 0.3.0 to
    # 1.0.0. make_release's check_addon_version holds bl_info to the uplugin; reading it here keeps ping on the same
    # number.
    from . import bl_info as _bl_info
    from . import ops_gen as _gen
    from . import cap as _cap

    prefs = _prefs()
    out = {
        "pong": True,
        "addon": "MifBlender",
        "addonVersion": list(_bl_info["version"]),
        "protocolVersion": _server.PROTOCOL_VERSION,
        "blenderVersion": list(bpy.app.version),
        "blenderVersionString": bpy.app.version_string,
        "background": bpy.app.background,
        "pid": os.getpid(),
        "python": sys.version.split()[0],
        "blendFile": bpy.data.filepath or None,
        "sceneName": bpy.context.scene.name if bpy.context.scene else None,
        "objectCount": len(bpy.data.objects),
        "unitsPerBlenderUnit": UU_PER_BU,
        "runPythonAllowed": bool(getattr(prefs, "allow_run_python", False)),
        # the add-on's two generative switches (ops_gen.py): the preference, the open file's project mode, and whether
        # the gen_* ops may run under both; the MCP server's bl_status adds its own switch
        "allowGenerative": _gen.preference_allows(),
        "generativeMode": _gen.file_mode(),
        "generativeAllowed": _gen.generative_allowed(),
        # the change cap in force for this file (cap.py): how many things one call may change before it previews
        "changeCap": _cap.change_cap(),
        "ops": _server.op_names(),
    }
    # the ops separate add-ons registered (extensions.py): the api version, and each provider's ops - which are in
    # "ops" above too, dispatched like any built-in
    from . import extensions as _ext
    out["extensionApi"] = _ext.summary()

    # WHICH COPY IS ANSWERING. addonVersion above is hand-written and does not move when a module
    # changes, and "ops" only differs when an op is ADDED - so neither notices the addon being
    # loaded from a stale scripts/addons copy while you edit a checkout somewhere else. See the
    # block above PACKAGE_DIR in __init__.py for what that cost.
    from . import PACKAGE_DIR, SOURCE_HASH_AT_IMPORT, source_hash  # local: avoids an import cycle
    live = source_hash()
    out["addonPath"] = PACKAGE_DIR
    out["sourceHash"] = live
    out["sourceHashAtImport"] = SOURCE_HASH_AT_IMPORT
    # None on either side means the sources could not be read, and "unknown" is not "unchanged" -
    # so this is None too rather than a confident False.
    out["sourceChangedSinceStart"] = (
        None if (live is None or SOURCE_HASH_AT_IMPORT is None)
        else live != SOURCE_HASH_AT_IMPORT)
    # IS THIS INSTALL WHOLE, AND THE ONLY ONE BLENDER CAN SEE? An upgrade left 43 backup copies beside
    # the add-on, each loaded as another, and a hybrid tree whose Add-ons entry still showed ticked. See install_check.
    from . import install_check
    try:
        out["install"] = install_check.check(PACKAGE_DIR)
    except Exception as exc:                                               # noqa: BLE001 - ping must still answer
        out["install"] = {"ok": None, "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
    echo = params.get("echo")
    if echo is not None:
        out["echo"] = jsonable(echo)
    return out


# ---------------------------------------------------------------------------
# call_log
# ---------------------------------------------------------------------------

# How many calls disk:true returns when no limit is given. Memory holds 64 at most and needs no default; the part of the
# file that is read back can hold a couple of thousand.
_CALL_LOG_DISK_DEFAULT = 50


def op_call_log(params):
    """The calls this Blender ran through the bridge, newest first - the call journal read back (see journal.py).

    FROM MEMORY by default: this Blender's last 64 calls, whatever the disk preference says. disk:true reads the journal
    FILE instead - every Blender that wrote it, and the sessions before a restart - and lists `unfinished`: calls whose
    start has no end, which is a call another Blender is still inside or the one it died in. The calls this op is itself
    part of are left out of both: they have no end yet only because they are the one reading.
    """
    reject_unknown(params, {"limit", "disk"}, "call_log")
    from_disk = take_bool(params, "disk", default=False)
    from . import journal
    out = journal.status()
    if from_disk:
        calls, sessions, info = journal.read_disk()
        out.update(info)
        unfinished = [c for c in calls if not c["ended"]]
        bounded = dict(params)
        if bounded.get("limit") is None:
            bounded["limit"] = _CALL_LOG_DISK_DEFAULT
        calls, extra = limit_rows(calls, bounded, "calls", excludes=False)
        out.update(source="disk", calls=calls, count=len(calls), unfinished=unfinished[:20],
                   unfinishedCount=len(unfinished), sessions=sessions[:20], sessionCount=len(sessions))
        out["note"] = ("read from the last %d bytes of the file, newest first, from every Blender that wrote it. "
                       "unfinished: a start with no end - that Blender was still inside the call when the file was "
                       "read, or it died in it (sessionDied). A session with died:true stopped without a shutdown "
                       "record and a later Blender came up with its pid - it was killed or crashed; closed:false "
                       "without died is still running, or died with nothing after it yet. restarted:true is a "
                       "server restart inside one Blender." % (out.get("readBytes") or 0))
    else:
        rows = []
        for row in journal.recent(None):
            if row.get("ok") is None:
                continue                      # still running: this call_log, or a batch it is part of
            row["t"] = journal.iso(row["t"])
            row.pop("summary", None)          # the UI's one-line form of args, which are here in full
            rows.append(row)
        calls, extra = limit_rows(rows, params, "calls", excludes=False)
        out.update(source="memory", calls=calls, count=len(calls))
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# list_objects
# ---------------------------------------------------------------------------

def op_list_objects(params):
    """Every object in the file, filtered by type, a name pattern (a glob when it holds * ? or [, else a substring) and
    near {point, radius}; detail adds object_info's full record per object."""
    reject_unknown(params, {"type", "objectType", "pattern", "detail", "limit", "nameExcludes", "near"}, "list_objects")
    near = _near_filter(params.get("near"))
    want_type = take(params, "type", "objectType")
    pattern = take(params, "pattern")
    detail = take_bool(params, "detail", default=False)

    if want_type is not None:
        want_type = str(want_type).upper()

    # `name in view_layer.objects` IS A LINEAR SCAN, so doing it per object made this loop
    # quadratic. MEASURED on 4.4.0 headless: the membership test alone took 0.031s over 2000
    # objects, 0.19s over 5000, 0.74s over 10000 and 3.06s over 20000 - four times the work each
    # time the count doubles - and that 3.06s was essentially all of list_objects' 3.2s at 20000.
    # The same call with detail:true, which produces THREE TIMES the bytes, took 0.45s, because it
    # goes through object_info and never touches this line. A slower op finishing seven times
    # faster than a faster one is what gave it away.
    #
    # The set is built once, costs one pass, and answers identically - checked against the
    # collection form on every object at 20000, not sampled.
    #
    # NOT GUARDED FOR A NULL view_layer, deliberately, because the line it replaces was not either.
    # `if view_layer else set()` would turn the AttributeError this used to raise into every object
    # reporting inViewLayer:false - a plausible answer that is wrong, which is worse than a crash.
    # op_scene_info below DOES guard, because its original did.
    in_view_layer = set(bpy.context.view_layer.objects.keys())

    # A GLOB IS A GLOB. `pattern` was only ever a case-insensitive substring, so pattern:"UCX_*" matched nothing
    # and answered count 0 - a plausible empty answer while a UCX_ object existed (hand check, 2026-09-13). A
    # pattern holding * ? or [ is now matched as a case-insensitive glob; anything else stays a substring, so
    # every existing caller gets exactly the answer it got before. filteredBy says which one was used.
    pat = str(pattern).lower() if pattern else ""
    glob_mode = any(ch in pat for ch in "*?[")

    def matches(name):
        low = name.lower()
        return fnmatch.fnmatchcase(low, pat) if glob_mode else pat in low

    rows = []
    kids = children_by_parent() if detail else None         # one pass, not one scan per object - see object_info
    for obj in bpy.data.objects:
        if want_type and obj.type != want_type:
            continue
        if pattern and not matches(obj.name):
            continue
        dist = _near_distance(obj, near) if near else None
        if near and dist > near["radius"]:
            continue
        if detail:
            rows.append(object_info(obj, children=kids))
            if near:
                rows[-1]["distance"] = round(dist, 6)
            continue
        row = {"name": obj.name, "type": obj.type,
               "inViewLayer": obj.name in in_view_layer}
        if near:
            row["distance"] = round(dist, 6)
        if obj.type == "MESH":
            # BLENDER'S OWN BOX HERE, NOT THE MEASURED ONE object_info gives (and detail:true, which calls it). The two
            # differ in two measured cases: a stack that evaluates to no vertices reads as the base mesh's box from 4.2
            # on, and a Subdivision Surface last in the stack reads as its cage from 5.0 on (ops_common
            # evaluated_mesh_bounds). Measuring here would read every modified mesh's evaluated vertices in one listing,
            # and on 5.x that read is what makes Blender subdivide a last-in-stack Subsurf on the CPU - a cost Blender
            # defers to the GPU and a listing has no business forcing for every object in the file.
            row["dimensionsBU"] = [round(float(v), 6) for v in obj.dimensions]
            row.update(mesh_counts(obj))
        # EVERY TYPE THAT HOLDS SLOTS, not MESH alone - the helper object_info and set_material_slots
        # use, so a text object's material reads back here too.
        slots = material_slot_names(obj)
        if slots is not None:
            row["materialSlots"] = slots
        rows.append(row)

    rows = exclude_rows(rows, params)
    if near:
        rows.sort(key=lambda r: (r["distance"], r["name"]))         # nearest first
    matched = len(rows)
    rows, extra = limit_rows(rows, params, "object(s)")
    return dict({"count": matched, "objects": rows,
                 "filteredBy": {"type": want_type, "pattern": pattern,
                                "patternMode": ("glob" if glob_mode else "substring") if pattern else None,
                                "nameExcludes": params.get("nameExcludes"),
                                "near": dict(near, point=list(near["point"])) if near else None}}, **extra)


def _near_filter(raw):
    """list_objects' near {point [x,y,z], radius, by ORIGIN|BOUNDS}, checked; None when not given."""
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) - {"point", "radius", "by"} or "point" not in raw or "radius" not in raw:
        raise MifOpError("near is {point: [x, y, z], radius: r, by: ORIGIN | BOUNDS}, got %r." % (raw,))
    point = finite_floats(raw["point"], "near.point") if isinstance(raw["point"], list) else None
    if not point or len(point) != 3:
        raise MifOpError("near.point is [x, y, z] in world space, got %r." % (raw["point"],))
    radius = finite_float(raw["radius"], "near.radius")
    if radius < 0:
        raise MifOpError("near.radius cannot be negative, got %g." % radius)
    by = str(raw.get("by", "ORIGIN")).upper()
    if by not in ("ORIGIN", "BOUNDS"):
        raise MifOpError("near.by is ORIGIN (the object's world origin) or BOUNDS (the nearest point of its "
                         "world-space bounding box), got %r." % (raw.get("by"),))
    return {"point": tuple(point), "radius": radius, "by": by}


def _near_distance(obj, near):
    """World distance from near's point to the object's origin, or to its world-axis-aligned bounding box."""
    p = near["point"]
    mw = obj.matrix_world
    if near["by"] == "ORIGIN" or obj.type in ("EMPTY", "LIGHT", "CAMERA", "SPEAKER", "LIGHT_PROBE"):
        t = mw.translation
        return math.sqrt(sum((t[i] - p[i]) ** 2 for i in range(3)))
    from mathutils import Vector   # noqa: PLC0415
    corners = [mw @ Vector(c) for c in obj.bound_box]
    gap = [max(min(c[i] for c in corners) - p[i], 0.0, p[i] - max(c[i] for c in corners)) for i in range(3)]
    return math.sqrt(sum(g * g for g in gap))


# ---------------------------------------------------------------------------
# scene_info
# ---------------------------------------------------------------------------

def op_scene_info(params):
    """What is actually in this Blender right now, and is it safe to export from.

    Read-only. Beyond the obvious census it reports scene.unit_settings, because
    scale_length is a silent multiplier on every FBX this pipeline writes.

      MEASURED on Blender 4.4.0 headless, factory startup, with the addon's own
      FBX_EXPORT_ARGS: the same 10 BU cube exported at scale_length 1.0 reimports
      at 10.0 BU, and exported at scale_length 0.01 reimports at 0.1 BU (object
      scale 0.01). UnitScaleFactor stays 1.0 in the file BOTH times, so the
      header does not give the mismatch away -- only the magnitudes do.

    A round trip run in a scene with a non-default scale_length therefore comes
    back 100x wrong while every ok:true in the chain stays true, which is why it
    is a warning here rather than a footnote in a README.
    """
    reject_unknown(params, {"detail", "limit", "nameExcludes"}, "scene_info")
    detail = take_bool(params, "detail", default=False)

    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    units = scene.unit_settings if scene else None
    active = view_layer.objects.active if view_layer else None

    by_type = {}
    for obj in bpy.data.objects:
        by_type[obj.type] = by_type.get(obj.type, 0) + 1

    # Same quadratic membership test as op_list_objects - see the note there. This one cost
    # 2.85s of scene_info's own 2.89s at 20000 objects.
    vl_names = set(view_layer.objects.keys()) if view_layer else set()
    if detail:
        # inViewLayer IS ADDED HERE TOO, and used not to be. detail:true built its rows from
        # object_info(), which knows about an object and nothing about a view layer, so asking for
        # MORE detail silently dropped the one field that says whether the object is in the layer
        # at all - the field a caller is most likely to be filtering on. Found 2026-09-14 by
        # test_blender_unsent_ops comparing the two forms key by key; a caller who added detail:true
        # to an existing call would have lost it with nothing to say so.
        kids = children_by_parent()
        objects = [dict(object_info(o, children=kids), inViewLayer=o.name in vl_names) for o in bpy.data.objects]
    else:
        objects = [{"name": o.name, "type": o.type,
                    "inViewLayer": o.name in vl_names}
                   for o in bpy.data.objects]

    warnings = []
    scale_length = float(units.scale_length) if units else 1.0
    if units is not None and abs(scale_length - 1.0) > 1e-9:
        warnings.append(
            "scene.unit_settings.scale_length is %g, not 1.0. MEASURED on 4.4.0: this "
            "scales every FBX this addon writes by that factor while UnitScaleFactor in "
            "the file stays 1.0 -- a 1000 uu tile would land in Unreal at %g uu and "
            "nothing in the export would report an error. Set it back to 1.0 before "
            "exporting, or expect the round trip's fidelity gate to abort."
            % (scale_length, 1000.0 * scale_length))
    if units is not None and units.system not in ("METRIC", "NONE"):
        warnings.append(
            "scene.unit_settings.system is %r. The centimetre reasoning this pipeline "
            "rests on (1 BU = %g uu) was verified under METRIC only -- an IMPERIAL scene "
            "is UNVERIFIED here." % (units.system, UU_PER_BU))

    # THE NAME IS SCENE-SCOPED AND THE COUNT IS FILE-SCOPED, and until set_current_scene existed
    # nothing could tell. Found 2026-09-20 the first time the active scene was switched through the
    # bridge: with the empty scene 'Shot_B' current, this op answered sceneName:'Shot_B' beside
    # objectCount:3 - the three objects of the OTHER scene, because objectCount, objectsByType,
    # meshCount and `objects` all walk bpy.data.objects, which is the whole FILE.
    #
    # THE NUMBERS ARE NOT CHANGED. A file census is a legitimate thing for this op to report and
    # callers read these fields today; silently re-scoping them would break every one of them and
    # would be the harder failure to spot. What was missing is anything SAYING so, which is why
    # this is a warning and two extra counts rather than an edit to objectCount.
    scene_objects = len(scene.objects) if scene else 0
    if scene is not None and len(bpy.data.scenes) > 1 and scene_objects != len(bpy.data.objects):
        warnings.append(
            "objectCount, objectsByType, meshCount and `objects` are FILE-wide (bpy.data.objects: "
            "%d), not this scene's. This file has %d scenes and '%s' holds %d object(s). The "
            "scene-scoped answers are sceneObjectCount, viewLayerObjectCount and the per-row "
            "inViewLayer flag."
            % (len(bpy.data.objects), len(bpy.data.scenes), scene.name, scene_objects))

    # THE OBJECT LIST IS THE ONLY PART OF THIS ANSWER THAT GROWS, and it is the largest single response in the
    # add-on: detail:true measures about 506 bytes an object, so a thousand-object scene is roughly 126 000 tokens
    # in one reply (docs/25_TOKEN_COST.md, 2026-09-14). objectCount stays the TRUE total whatever the limit, because
    # every other field here describes the whole scene and a caller reading a scene summary must not be handed a
    # partial count as if it were the scene's size.
    objects = exclude_rows(objects, params)
    objects, obj_extra = limit_rows(objects, params, "object(s)")
    return dict({
        "sceneName": scene.name if scene else None,
        "blendFile": bpy.data.filepath or None,
        "background": bpy.app.background,
        "objectCount": len(bpy.data.objects),
        "objectsByType": by_type,
        "objects": objects,
        "meshCount": by_type.get("MESH", 0),
        "activeObject": active.name if active else None,
        "selectedObjects": [o.name for o in bpy.context.selected_objects],
        # THE TWO SCENE-SCOPED COUNTS, and they are not the same number. sceneObjectCount is what is
        # LINKED into the scene; viewLayerObjectCount is what the active view layer can see, which
        # is smaller when a collection is excluded or hidden. objectCount above is the whole file.
        "sceneObjectCount": scene_objects,
        "viewLayerObjectCount": len(view_layer.objects) if view_layer else 0,
        "sceneCount": len(bpy.data.scenes),
        "collections": [c.name for c in bpy.data.collections],
        "unitSettings": {
            "system": units.system if units else None,
            "systemRotation": units.system_rotation if units else None,
            "scaleLength": scale_length,
            "lengthUnit": units.length_unit if units else None,
        },
        "unrealUnitsPerBlenderUnit": UU_PER_BU,
        "frameCurrent": scene.frame_current if scene else None,
        "warnings": warnings,
    }, **{("objects" + k[0].upper() + k[1:]): v for k, v in obj_extra.items()})


# ---------------------------------------------------------------------------
# object_info
# ---------------------------------------------------------------------------

def op_object_info(params):
    """One object's full record by name: type, transform (basis and world location, rotation mode, scale), visibility,
    parent and children, material slots, and per-type detail such as a mesh's counts or a light's settings."""
    reject_unknown(params, {"object", "name"}, "object_info")
    name = take(params, "object", "name", required=True, kind=str)
    return {"object": object_info(get_object(name))}


# ---------------------------------------------------------------------------
# clear_scene / delete_object
# ---------------------------------------------------------------------------

def _stale_rigid_bakes(objs):
    """READ ONLY: (scene, point cache, [body names], what it held) for every scene whose rigid body world is BAKED and
    has one of these objects as a body. _delete asks this before it deletes, and frees the bakes after.

    A RIGID BODY BAKE IS STORED BY POSITION IN THE WORLD'S COLLECTION, NOT BY OBJECT, so deleting a body from a baked
    world does not just leave stale frames - it hands them to the wrong objects. MEASURED 2026-09-16, identical on
    3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1, the addon imported from the repo, positions read after the bake:
      - bodies [Floor, A, B] baked; delete A; B at frame 10 stood at [-2.0, 0.0, 3.3061] - A's own frame-10 position,
        not B's [2.0, 0.0, 0.8061] - and the cache still read isBaked with its 40 frames.
      - clear_scene between two fixtures: the scene keeps its rigid body world and that bake, bake_physics keeps an
        existing bake, and every body of the next fixture replayed the first fixture's. The second seat's
        collisionShape, friction and bounciness rows came back byte-identical this way (mailbox id 57) and read as
        settings that never reach Bullet - they do, in a fresh scene or once the bake is freed: 4 of 4 shapes, 5 of 5
        on a ramp and 3 of 3 frictions distinct.
    Freed AFTER the delete - freeing before it measured the same - the next plain bake simulates the bodies that are
    really there. Only the RIGID BODY cache: cloth, soft body, particle and paint caches belong to modifiers and
    systems and leave with their own objects. A world none of these objects is a body in keeps its bake."""
    doomed = {o.name for o in objs}
    stale = []
    for sc in bpy.data.scenes:
        rbw = getattr(sc, "rigidbody_world", None)
        pc = getattr(rbw, "point_cache", None) if rbw is not None else None
        coll = getattr(rbw, "collection", None) if rbw is not None else None
        if pc is None or coll is None or not pc.is_baked:
            continue
        bodies = sorted(o.name for o in coll.objects if o.name in doomed)
        if bodies:
            stale.append((sc, pc, bodies, (pc.info or "").strip()))
    return stale


def _free_rigid_bakes(stale):
    """Free what _stale_rigid_bakes found, once the objects are gone. Returns (freed, notFreed) rows.

    NEVER A REFUSAL, because it runs after the delete: an op that has deleted and then raises cannot keep the promise
    a refusal makes. A bake that will not free is reported beside the delete that did happen."""
    freed, not_freed = [], []
    for sc, pc, bodies, was in stale:
        row = {"scene": sc.name, "bodies": bodies, "bakeWas": was}
        try:
            with bpy.context.temp_override(scene=sc, point_cache=pc):
                bpy.ops.ptcache.free_bake()
        except RuntimeError as exc:
            not_freed.append(dict(row, error=str(exc)))
            continue
        if pc.is_baked:
            not_freed.append(dict(row, error="free_bake reported success and the cache still reads baked"))
            continue
        freed.append(row)
    return freed, not_freed


def _joints_holding(objs):
    """READ ONLY: rigid body joints NOT among objs whose object1 or object2 is one of them, in every scene's world."""
    doomed = {o.name for o in objs}
    found = {}
    for sc in bpy.data.scenes:
        rbw = getattr(sc, "rigidbody_world", None)
        cons = getattr(rbw, "constraints", None) if rbw is not None else None
        if cons is None:
            continue
        for joint in cons.objects:
            rbc = getattr(joint, "rigid_body_constraint", None)
            if joint.name in doomed or rbc is None:
                continue
            if any(body is not None and body.name in doomed for body in (rbc.object1, rbc.object2)):
                found[joint.name] = joint
    return [found[n] for n in sorted(found)]


def _detach_rigid_bodies(objs):
    """Take the doomed joints, THEN the doomed bodies, out of their rigid body worlds through Blender's own operators, so
    bpy.data.objects.remove finds no Bullet object left to free.

    objects.remove FREES RIGID BODY DATA WITHOUT TAKING IT OUT OF BULLET'S WORLD. A joint with disable_collisions links
    both ways - each body holds a reference to the joint, and the joint to each body - and whichever is freed second
    reads the first. Both orders crashed, on different builds (2026-09-16, through the add-on, rigs baked, read, bake
    cleared): 5.2.1 died deleting a joint whose bodies had gone (btRigidBody::removeConstraintRef <- RB_constraint_delete);
    4.2.17, with joints deleted first, died deleting a body (btRigidBody::removeConstraintRef <- RB_body_delete).
    rigidbody.constraint_remove takes the joint out of the world first - which is what drops the references, while both
    bodies still exist - and rigidbody.object_remove then does the same for a body no joint refers to any more."""
    doomed = {o.name for o in objs}
    for sc in bpy.data.scenes:
        rbw = getattr(sc, "rigidbody_world", None)
        if rbw is None:
            continue
        for coll, attr, op_name in ((getattr(rbw, "constraints", None), "rigid_body_constraint", "constraint_remove"),
                                    (getattr(rbw, "collection", None), "rigid_body", "object_remove")):
            members = [o for o in coll.objects if o.name in doomed] if coll is not None else []
            for ob in members:
                if getattr(ob, attr, None) is not None:
                    _rigidbody_op_on(sc, ob, getattr(bpy.ops.rigidbody, op_name))


def _rigidbody_op_on(sc, ob, op):
    """Run a rigid body operator on ob as its view layer's ACTIVE object, and put the active object back.

    AN OVERRIDE IS NOT ENOUGH BEFORE 5.2. MEASURED 2026-09-16, headless, a fresh hinge rig per method:
    temp_override(object=joint, active_object=joint) - with bpy.context.object reading the joint - still made
    rigidbody.constraint_remove raise "Object has no Rigid Body Constraint to remove" on 3.6.23, 4.2.17, 4.4.0 and
    5.0.1 (5.2.1 took it); the view layer's active object worked on all five, and object_remove took either."""
    vl = next((v for v in sc.view_layers if v.objects.get(ob.name) is not None), None)
    if vl is None:
        with bpy.context.temp_override(scene=sc, object=ob, active_object=ob):
            op()
        return
    before = vl.objects.active
    vl.objects.active = ob
    try:
        with bpy.context.temp_override(scene=sc, view_layer=vl):
            op()
    finally:
        vl.objects.active = before


_PURGE_COLLS = ("meshes", "materials", "images", "armatures", "lights", "cameras", "node_groups", "particles", "actions",
                "curves", "lattices", "textures", "lightprobes", "hair_curves", "pointclouds", "volumes",
                "speakers", "sounds", "metaballs")


# OBJECT DATA: what an object holds as its .data (and its particle systems' settings) - a datablock that, with no object,
# is nothing but a NAME the next object of that name cannot have. delete_object's purge takes these (scope
# "object_data"); materials, images, node groups, textures, actions and sounds are work a rebuild may reuse.
_OBJECT_DATA_COLLS = ("meshes", "armatures", "lights", "cameras", "particles", "curves", "lattices", "lightprobes",
                      "hair_curves", "pointclouds", "volumes", "speakers", "metaballs")


def _zero_user_names():
    """{(collection name, datablock name)} of every datablock in _PURGE_COLLS with no user. READ ONLY."""
    found = set()
    for coll_name in _PURGE_COLLS:
        for datablock in getattr(bpy.data, coll_name, None) or ():
            if datablock.users == 0:
                found.add((coll_name, datablock.name))
    return found


def _delete(objs, purge, scope="file"):
    """(names, purged, freed, notFreed, jointsAdded) - freed/notFreed are rigid body bakes (see _stale_rigid_bakes),
    jointsAdded the joints that went with their bodies. `purged` is (count, names), and (count, names, kept names)
    with scope "object_data".

    SCOPE "object_data" (delete_object, 2026-10-05): the purge removes object data with no user (_OBJECT_DATA_COLLS:
    meshes, curves, lights, cameras, armatures, particle settings ...) and nothing else. A car mesh deleted with purge to
    rebuild it took its paint materials with the file-wide sweep (a mesh's materials have no user once the mesh goes),
    and the rebuild's create_mesh refused "no material named ...": the names were reported and the work was lost
    anyway. Freeing object data is what frees the NAME a rebuild needs - an earlier delete without purge leaves its
    mesh holding the object's name too, so the sweep stays file-wide for object data (test_blender_camera_params B934:
    a camera deleted without purge, then its .001 deleted with purge, and the next create_camera gets the bare name).
    What the delete leaves without a user and is not object data is kept and named (kept), for purge_orphans to take
    on request. clear_scene, import_mesh's clear and delete_collection keep "file", below.

    THE PURGE IS FILE-WIDE AND IT IS NOT WHAT THE NAME SUGGESTS. It walks every datablock in the
    collections below and removes ANY with zero users - not only the ones this delete orphaned.
    Measured 2026-09-20 on 5.0.1: `create_material "MifDraft"` (an orphan, which is what a material
    is between being made and being assigned), then `delete_object{object:"MifUnrelated",
    purge:true}` on an object with no connection to it - and MifDraft is gone, orphansPurged 2. The
    control without purge leaves it. Sharper still, because clear_scene defaults purge TRUE:
    `clear_scene{type:"LIGHT"}` in a scene with no lights answered removedCount 0 and
    orphansPurged 2, destroying a material and a node group the caller had just created, in a call
    that by its own report deleted nothing.

    THE BEHAVIOUR IS DELIBERATE AND IS NOT BEING CHANGED HERE. The comments below record what the
    file-wide sweep fixes - Blender reserves a datablock's NAME until the datablock goes, so a
    scene cleared and rebuilt got "MifT_CURVES.002" and every later call addressing it by name
    reached the wrong one. Narrowing the sweep would bring that back. What was missing is that
    nothing SAID so and nothing said WHAT it took: a count of 2 does not tell a caller which two
    things they just lost. So the names are collected and reported, and a purge that removed
    something while removing NO object says so in as many words.

    A JOINT GOES WITH ITS BODIES, AND EVERYTHING LEAVES THE RIGID BODY WORLD BEFORE IT IS REMOVED - see
    _detach_rigid_bodies for the two crashes that decided it. Deleting a body takes every joint that holds it: a joint
    left holding a freed body is the same hazard waiting for its own delete, and dead in Blender anyway."""
    joints = _joints_holding(objs)
    joint_names = [j.name for j in joints]              # read now: a removed object has no name to read
    objs = sorted(list(objs) + joints,
                  key=lambda o: 0 if getattr(o, "rigid_body_constraint", None) is not None else 1)
    stale = _stale_rigid_bakes(objs)                    # before the detach, which takes bodies out of the collection
    names = [o.name for o in objs]
    zero_before = _zero_user_names() if purge and scope == "object_data" else None
    _detach_rigid_bodies(objs)
    remove_objects(objs)
    freed, not_freed = _free_rigid_bakes(stale)
    purged = 0
    purged_names = []
    if purge:
        # orphan meshes/materials left behind by the delete
        # LIGHTS, CAMERAS AND NODE GROUPS ADDED 2026-09-01, and their absence was not theoretical.
        # This list predates the ops that create them, so clearing a scene left orphaned light and
        # camera DATA behind holding their names - and the next create_light called "Fluoro_01" got
        # "Fluoro_01.001", because bpy.data.objects.new() takes the datablock's name and the
        # datablock could not have the old one. A later keyframe call then failed with "no object
        # named 'Fluoro_01'" on a scene that had just been cleared and rebuilt. The response had
        # reported the real name all along; nothing was reading it.
        #
        # AND THE TYPES ADDED SINCE, 2026-09-12 - the same failure one generation later. create_text and
        # create_curve make bpy.data.curves, create_lattice lattices, create_texture textures, the probe
        # lightprobes, and create_curves / create_point_cloud / create_volume hair_curves, pointclouds
        # and volumes. None was purged, so a scene cleared and rebuilt handed the second "MifT_CURVES"
        # the name "MifT_CURVES.002" - found by running test_blender_creation twice in one Blender.
        # getattr because a build may predate a collection. NOT texts: a text datablock with no users
        # is often somebody's script in the Text Editor, and purging it would be destroying work.
        # METABALLS, 2026-10-26, with create_metaball: a metaball's data keeps its name like a light's (a create_nurbs_surface
        # surface is a bpy.data.curves datablock, already listed).
        # SPEAKERS AND SOUNDS, 2026-09-23, with add_sound: a speaker's data is object data like a light's and keeps
        # its name the same way, and a sound held only by that speaker loses its last user once the speaker data
        # goes - so speakers come before sounds, and the passes below catch the chain either way.
        # BY bpy.data's OWN COLLECTION NAME, which is what each purged name is prefixed with below: "materials:Rust" is
        # the shape purge_orphans {only} takes and file_info's orphans are keyed by. The class name was used until
        # 2026-09-28 (the 2026-09-21 pack-rule review) - "Material:Rust" - which `only` refused as no such datablock.
        colls = [(n, getattr(bpy.data, n, None)) for n in _PURGE_COLLS]
        colls = [(n, c) for n, c in colls if c is not None]
        if zero_before is not None:
            for coll_name, coll in colls:
                for datablock in list(coll) if coll_name in _OBJECT_DATA_COLLS else ():
                    if datablock.users == 0:
                        purged_names.append("%s:%s" % (coll_name, datablock.name))
                        coll.remove(datablock)
            kept = sorted("%s:%s" % key for key in _zero_user_names() - zero_before)
            return names, (len(purged_names), purged_names, kept), freed, not_freed, joint_names
        # UNTIL A PASS REMOVES NOTHING, not once through. A datablock loses its last user only when what used
        # it goes, and one pass in a fixed order misses every chain that runs against the order: materials
        # are swept BEFORE curves, so a material held only by a text or a curve still had that user when its
        # turn came, the curve went after, and the material stayed behind with no users at all. Measured
        # 2026-09-16 on 5.2.1 rebuilding the pcbuild hero scene in one session: the rebuild's create_material
        # "RGB" came back "RGB.001", and every call after it addressed by name reached the stale material.
        # Bounded, because a pass that removes something always shrinks a finite set - eight is headroom.
        for _pass in range(8):
            removed_now = 0
            for coll_name, coll in colls:
                for datablock in list(coll):
                    if datablock.users == 0:
                        # READ THE NAME BEFORE THE REMOVE. A freed datablock has no name to read,
                        # which is the same trap _joints_holding records one function up.
                        purged_names.append("%s:%s" % (coll_name, datablock.name))
                        coll.remove(datablock)
                        removed_now += 1
            purged += removed_now
            if not removed_now:
                break
    return names, (purged, purged_names), freed, not_freed, joint_names


def _purge_note(purge, purged, purged_names, removed_count, op):
    """What the purge actually took, in the caller's words rather than as a number.

    TWO SENTENCES AND THE SECOND IS THE ONE THAT MATTERS. A purge that freed things while removing
    NO object cannot have been freeing what this call orphaned, and that is the case where a caller
    loses work without a hint: clear_scene{type:'LIGHT'} in a scene with no lights answered
    removedCount 0 and orphansPurged 2, taking a material and a node group made moments earlier.
    """
    if not purge or not purged:
        return None
    what = ", ".join(purged_names[:8]) + (" ..." if len(purged_names) > 8 else "")
    base = ("%s purged %d orphan datablock(s), and the sweep is FILE-WIDE: it removes ANY datablock "
            "with zero users, not only the ones this call orphaned. Taken: %s." % (op, purged, what))
    if removed_count == 0:
        base += (" THIS CALL REMOVED NO OBJECT, so none of that was orphaned by it - a datablock "
                 "made and not yet used (a material before it is assigned, a node group before it "
                 "is instanced) has zero users and is swept. Pass purge:false to keep them.")
    return base


def _object_data_purge_note(purged_names, kept):
    """delete_object's word on its purge (scope "object_data"): what it took, and what it left without a user and kept."""
    if not purged_names and not kept:
        return None
    note = ("delete_object's purge took object data with no user - meshes, curves, lights, cameras, armatures, particle "
            "settings, in the whole file, which frees their names - and nothing else%s."
            % ((": " + ", ".join(purged_names[:8]) + (" ..." if len(purged_names) > 8 else "")) if purged_names else ""))
    if kept:
        note += (" KEPT, though the delete left them with no user: %s. A rebuild can use them again; Blender drops them "
                 "when the file is saved and reopened, purge_orphans {only: [these names]} takes them now, and "
                 "clear_scene sweeps every orphan in the file." % (", ".join(kept[:8]) + (" ..." if len(kept) > 8 else "")))
    return note


def _joints_note(joints):
    """The response's word on joints deleted because their bodies were, or None when there were none."""
    if not joints:
        return None
    return ("the rigid body joint(s) %s were deleted too: a joint left holding a deleted body is dead in Blender, and "
            "removing joints and their bodies without taking them out of the physics world first crashed Blender - "
            "5.2.1 deleting the joint after its bodies, 4.2.17 deleting a body after its joint. They are in `removed`."
            % ", ".join("'%s'" % j for j in joints))


def _freed_note(freed, not_freed):
    """The response's word on the rigid body bakes a delete freed - or could not free - or None when there were none."""
    parts = []
    if freed:
        parts.append("the rigid body bake of %s was FREED: a rigid body cache is stored by position in the world's "
                     "collection, so after deleting %s it would have replayed old frames onto the wrong bodies "
                     "(measured: the survivor of a deleted pair stood exactly where the deleted body had). "
                     "bake_physics bakes the bodies that are left."
                     % (", ".join("'%s'" % f["scene"] for f in freed),
                        ", ".join(b for f in freed for b in f["bodies"])))
    if not_freed:
        parts.append("WARNING: the objects were deleted and the rigid body bake of %s could NOT be freed (%s) - it "
                     "will replay old frames onto the bodies left in that world until bake_physics {clear: true} or "
                     "{rebake: true}."
                     % (", ".join("'%s'" % f["scene"] for f in not_freed),
                        "; ".join(f["error"] for f in not_freed)))
    return " ".join(parts) or None


# COLOUR MANAGEMENT SURVIVES clear_scene - it is scene settings, not objects. The second seat measured it (mailbox id 86:
# gamma 11.613 against 36.328 in the next build), and it became a real trap in id 109: a scene that picked an HDR display
# left it for the next, where the house stage() asked for AgX, which that display refuses, so the next finished clip
# would have rendered in PQ. Reported rather than reset, like the empty collections: a caller rebuilding a scene may
# mean to keep its grade.
#
# FACTORY VALUES, measured 2026-09-16 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1 (read_factory_settings, with and
# without the startup file): display sRGB, look None, exposure 0.0, gamma 1.0, curve mapping off on every build; view
# transform Filmic on 3.6.23 and AgX on 4.2.17 and later. 4.0 and 4.1 were not measured, so there the view transform
# is not compared.
def _colour_management_kept(sc):
    """(values, note) - the scene's colour management, and a note naming whatever differs from the factory values."""
    vs = getattr(sc, "view_settings", None)
    ds = getattr(sc, "display_settings", None)
    if vs is None or ds is None:
        return None, None
    kept = {"displayDevice": ds.display_device, "viewTransform": vs.view_transform, "look": vs.look,
            "exposure": round(float(vs.exposure), 6), "gamma": round(float(vs.gamma), 6),
            "useCurveMapping": bool(vs.use_curve_mapping)}
    factory = {"displayDevice": "sRGB", "look": "None", "exposure": 0.0, "gamma": 1.0, "useCurveMapping": False}
    if bpy.app.version < (4, 0, 0):
        factory["viewTransform"] = "Filmic"
    elif bpy.app.version >= (4, 2, 0):
        factory["viewTransform"] = "AgX"
    changed = sorted(k for k, v in factory.items() if kept[k] != v)
    note = ("clear_scene removes objects, not scene settings, and this scene's colour management is not at Blender's "
            "factory values: %s. The next build renders through it unless set_color_management resets it."
            % ", ".join("%s %r (factory %r)" % (k, kept[k], factory[k]) for k in changed)) if changed else None
    return kept, note


def op_clear_scene(params):
    """Delete every object in the file, or every one of a type; purgeOrphans defaults TRUE and is file-wide. Collections
    stay and are named when left empty; a baked rigid body world losing a body has its bake freed."""
    reject_unknown(params, {"type", "objectType", "purgeOrphans", "purge"}, "clear_scene")
    want_type = take(params, "type", "objectType")
    purge = take_bool(params, "purgeOrphans", "purge", default=True)
    if want_type is not None:
        want_type = str(want_type).upper()

    targets = [o for o in bpy.data.objects if not want_type or o.type == want_type]
    removed, (purged, purged_names), freed, not_freed, joints = _delete(targets, purge)
    # THE COLLECTIONS STAY, AND SAYING SO IS THE FIX. clear_scene deletes objects - by contract - and a collection is not
    # one, so the second seat's probe (mailbox id 75) looped fixtures through clear_scene and every case after the first
    # found create_collection("Props") refused because the empty Props from the case before was still there: the new box
    # was never put in the collection being hidden, and a working flag measured as dead. Deleting collections here
    # would break every caller that rebuilds into the ones it has, so the response names the ones left empty instead.
    empty = sorted(c.name for c in bpy.data.collections if len(c.all_objects) == 0)
    cm_kept, cm_note = _colour_management_kept(bpy.context.scene)
    return {"removed": removed, "removedCount": len(removed),
            "colorManagementKept": cm_kept, "colorManagementNote": cm_note,
            "orphansPurged": purged, "remaining": len(bpy.data.objects),
            # WHICH ONES, not just how many. purge defaults TRUE here, and the sweep is FILE-WIDE:
            # a material or node group made and not yet used is an orphan, and this takes it even
            # when the type filter removed no object at all.
            "orphansPurgedNames": purged_names[:50] or None,
            "orphanPurgeNote": _purge_note(purge, purged, purged_names, len(removed), "clear_scene"),
            "freedRigidBodyBake": freed or None, "rigidBodyBakeNotFreed": not_freed or None,
            "rigidBodyBakeNote": _freed_note(freed, not_freed),
            "jointsRemovedWithBodies": joints or None, "jointsNote": _joints_note(joints),
            "emptyCollectionsLeft": empty[:50] or None,
            "emptyCollectionCount": len(empty),
            "collectionsNote": ("clear_scene deletes objects, not collections: %d collection(s) are left empty (%s), and "
                                "create_collection refuses a name that is still taken - delete_collection removes one."
                                % (len(empty), ", ".join(empty[:10]) + (" ..." if len(empty) > 10 else "")))
                               if empty else None}


def op_delete_object(params):
    """Delete one object or a list of them by name; purgeOrphans (default false) then removes only the object data they
    leave without a user. A baked rigid body world losing a body has its bake freed."""
    reject_unknown(params, {"object", "name", "objects", "purgeOrphans", "purge"},
                   "delete_object")
    purge = take_bool(params, "purgeOrphans", "purge", default=False)
    names = take(params, "objects")
    if names is None:
        names = [take(params, "object", "name", required=True, kind=str)]
    if not isinstance(names, list):
        raise MifOpError("'objects' must be a list of object names")

    targets = [get_object(n) for n in names]
    removed, purge_result, freed, not_freed, joints = _delete(targets, purge, scope="object_data")
    purged, purged_names, kept = purge_result if purge else (0, [], [])
    return {"removed": removed, "removedCount": len(removed), "orphansPurged": purged,
            "orphansPurgedNames": purged_names[:50] or None,
            "orphansKept": len(kept), "orphansKeptNames": kept[:50] or None,
            "orphanPurgeNote": _object_data_purge_note(purged_names, kept) if purge else None,
            "freedRigidBodyBake": freed or None, "rigidBodyBakeNotFreed": not_freed or None,
            "rigidBodyBakeNote": _freed_note(freed, not_freed),
            "jointsRemovedWithBodies": joints or None, "jointsNote": _joints_note(joints)}


# ---------------------------------------------------------------------------
# run_python  -- the escape hatch
# ---------------------------------------------------------------------------

RUN_PYTHON_HELP = (
    "run_python is disabled. Enable it in Edit > Preferences > Add-ons > MifBlender "
    "> 'Allow run_python'. It executes arbitrary code inside Blender with your user's "
    "privileges -- only enable it on a machine you control."
)


def op_run_python(params):
    """Execute arbitrary Python on the main thread.

    Everything MifBlender does not have a first-class op for goes through here.
    It runs on the main thread like every other op, so bpy is safe to touch.

    Contract: whatever the code assigns to a module-level name `result` comes
    back in the response, coerced to JSON-safe values. stdout and stderr are
    captured. An exception is returned as ok:false with the traceback -- it does
    NOT kill the connection. Names persist from call to call, as in a console,
    until a .blend is loaded (_run_python_namespace); `result` starts each call
    as None.

    THE PERSISTENCE IS NEVER SILENT (a short film, 2026-10-02: a variable an earlier call left behind silently changed
    a later run). namesFromEarlierCalls lists every name this code READ before assigning it that an earlier call had
    left in the shared namespace, with a warning. `fresh: true` runs the code in a clean namespace of its own - the
    pre-imported modules only - and leaves the shared one exactly as it was; nothing the fresh call defines carries
    over. The default stays shared, decided after benchmark round six, where 4 of 9 failed Blender calls used a
    name an earlier call had defined (docs/TOKEN_OPTIMIZATION.md 2.10).
    """
    reject_unknown(params, {"code", "script", "file", "filepath", "returnLocals", "fresh"},
                   "run_python", {"reset": "fresh:true runs one call in a clean namespace; the shared one is cleared "
                                           "when a .blend loads.",
                                  "persistent": "names persist by default; fresh:true opts one call out."})
    prefs = _prefs()
    if not getattr(prefs, "allow_run_python", False):
        raise MifOpError(RUN_PYTHON_HELP)

    code = take(params, "code", "script")
    path = take(params, "file", "filepath")
    if code is None and path is None:
        raise MifOpError("run_python needs 'code' (a string) or 'file' (a path to a .py)")
    if code is not None and path is not None:
        raise MifOpError("pass 'code' or 'file', not both")
    if path is not None:
        if not os.path.isfile(path):
            raise MifOpError("no such file: %s" % path)
        with open(path, "r", encoding="utf-8") as handle:
            code = handle.read()
    if not isinstance(code, str):
        raise MifOpError("'code' must be a string")
    teardown = _host_teardown(code)
    if teardown:
        raise MifOpError(
            "run_python refused, and NOTHING was run: the code calls %s, which stops MifBlender - its server exits, a "
            "background Blender quits, and the next call has nothing to answer it (measured on 5.2.1: "
            "read_factory_settings and read_factory_userpref disable every add-on, MifBlender with it). To start from "
            "an empty scene, call bl_clear_scene (every object, orphans purged), or run "
            "bpy.ops.wm.read_homefile(use_empty=True), which keeps the add-ons (measured: the server stays up). To "
            "open a file, bl_open_file." % teardown)

    fresh = take_bool(params, "fresh", default=False)
    namespace = _new_run_python_namespace() if fresh else _run_python_namespace()
    namespace["result"] = None          # a result is this call's alone; every other name carries over
    carried = [] if fresh else _names_from_earlier_calls(code, namespace)

    out_buf, err_buf = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            exec(compile(code, "<mifblender:run_python>", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
            "stdout": out_buf.getvalue(),
            "stderr": err_buf.getvalue(),
            "namespace": "fresh" if fresh else "shared",
            "namesFromEarlierCalls": carried or None,
            "warning": _carried_warning(carried),
        }

    response = {
        "stdout": out_buf.getvalue(),
        "stderr": err_buf.getvalue(),
        "result": jsonable(namespace.get("result")),
        "namespace": "fresh" if fresh else "shared",
        "namesFromEarlierCalls": carried or None,
        "warning": _carried_warning(carried),
    }
    if take_bool(params, "returnLocals", default=False):
        response["names"] = sorted(k for k in namespace if not k.startswith("__") and k not in _RUN_PYTHON_GIVEN)
    return response


# CALLS THAT STOP THE ADD-ON RUNNING THEM. benchmark round ten (X2, 2026-10-01): an agent began with
# bpy.ops.wm.read_factory_settings(use_empty=True); the factory reset disabled every add-on, unregister() stopped the
# server, the background Blender quit, and the run was recorded crashed. Measured on 5.2.1 through run_python:
# read_factory_settings and read_factory_userpref both print "[MifBlender] stopped" and the port closes;
# read_homefile(use_empty=True) leaves the server answering. quit_blender ends the process; addon_disable or
# addon_remove of MifBlender runs the same unregister(). Read from the parsed code, so a comment or a string that
# merely mentions one is not refused, and getattr(bpy.ops.wm, "read_factory_settings") is.
_TEARDOWN_CALLS = ("read_factory_settings", "read_factory_userpref", "quit_blender")
_ADDON_OFF_CALLS = ("addon_disable", "addon_remove")


def _host_teardown(code):
    """The call in `code` that would stop MifBlender, as 'bpy.ops.wm.<name>' - or None. Code that does not parse is
    left for exec to report."""
    import ast
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    # the add-on's own module name, whether it is installed as 'MifBlender' or as an extension 'bl_ext.<repo>.<name>'
    own = (__package__ or "MifBlender").split(".")[-1].lower()
    names_addon = False
    hits = []
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.split(".")[-1].lower() == own:
                names_addon = True
            name = node.value
        if name in _TEARDOWN_CALLS:
            hits.append("bpy.ops.wm.%s" % name)
        elif name in _ADDON_OFF_CALLS:
            hits.append("bpy.ops.preferences.%s" % name)
    for hit in hits:
        if hit.split(".")[-1] in _TEARDOWN_CALLS or names_addon:
            return hit
    return None


# ONE NAMESPACE FOR EVERY run_python CALL UNTIL A .blend LOADS, decided after benchmark round six
# (docs/TOKEN_OPTIMIZATION.md 2.10). Each call got a fresh one, and agents write Python the way a console keeps it:
# 4 of the round's 9 failed Blender calls used a name an earlier call had imported or defined (Vector three times on
# the walk cycle, where raw Python - whose namespace persists - failed none). A loaded file clears it, since its
# objects are not the ones an old name would point at; `result` is reset on every call. mathutils' Vector, Matrix,
# Euler, Quaternion and Color are given alongside bpy, bmesh, math and mathutils.
_RUN_PYTHON_NS = None
_RUN_PYTHON_GIVEN = ("bpy", "bmesh", "mathutils", "math", "Vector", "Matrix", "Euler", "Quaternion", "Color")


def _new_run_python_namespace():
    """A clean namespace: the pre-imported modules and nothing an earlier call made."""
    namespace = {"__name__": "mifblender_run_python", "bpy": bpy, "math": math, "result": None}
    try:
        import bmesh  # noqa: WPS433 - convenience for callers
        namespace["bmesh"] = bmesh
    except Exception:
        pass
    try:
        import mathutils  # noqa: WPS433
        namespace["mathutils"] = mathutils
        for name in ("Vector", "Matrix", "Euler", "Quaternion", "Color"):
            namespace[name] = getattr(mathutils, name)
    except Exception:
        pass
    return namespace


def _run_python_namespace():
    global _RUN_PYTHON_NS
    if _RUN_PYTHON_NS is None:
        _RUN_PYTHON_NS = _new_run_python_namespace()
    return _RUN_PYTHON_NS


def _names_from_earlier_calls(code, namespace):
    """The names `code` reads before it binds them that an earlier call left in `namespace` - what a persistent
    namespace changes about this run. Read from the parsed code: a name is counted when its first read is on or
    before the line of its first binding (assignment, import, def, class) - by LINE, because `x = x + 1` reads x
    before it binds it while its target sits left of its value - and never when it is a function's own parameter.
    Code that does not parse is left for exec to report."""
    import ast
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []
    earlier = {k for k in namespace if not k.startswith("__") and k not in _RUN_PYTHON_GIVEN and k != "result"}
    if not earlier:
        return []
    first_load, first_bind, params = {}, {}, set()

    def note(table, name, node):
        pos = getattr(node, "lineno", 0)
        if name not in table or pos < table[name]:
            table[name] = pos

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            note(first_load if isinstance(node.ctx, ast.Load) else first_bind, node.id, node)
        elif isinstance(node, ast.arg):
            params.add(node.arg)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            note(first_load, node.target.id, node)      # x += 1 reads x first; its Name says only Store
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                note(first_bind, (alias.asname or alias.name).split(".")[0], node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            note(first_bind, node.name, node)
    return sorted(n for n, pos in first_load.items()
                  if n in earlier and n not in params and (n not in first_bind or pos <= first_bind[n]))


def _carried_warning(carried):
    if not carried:
        return None
    return ("this code read %d name(s) an earlier run_python call left in the shared namespace (%s) - their values "
            "came from that call, not this one. Pass fresh:true to run without them (over MCP, bl_run_python takes it: mif_call('bl_run_python', {'code': ..., 'fresh': true}); run_python does not); the "
            "shared namespace is cleared when a .blend loads."
            % (len(carried), ", ".join(carried[:20]) + (" ..." if len(carried) > 20 else "")))


def _reset_run_python_namespace(*_args):
    global _RUN_PYTHON_NS
    _RUN_PYTHON_NS = None


# ops_undo._install_load_handler's shape, and for its reasons: installed at import, because a harness that imports the
# add-on without enabling it (the suites' serve_forever) never runs register(); persistent, or the first load would
# unregister it; replaced by name, so reloading this module does not stack a second copy; and GUARDED, because
# test_blender_refusals imports every ops module under a stub bpy with no app.handlers - a decorator here took the whole
# op table down with it there.
def _install_run_python_reset():
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    if handlers is None or not hasattr(handlers, "load_post"):
        return False
    for h in [h for h in handlers.load_post if getattr(h, "__name__", "") == "_reset_run_python_namespace"]:
        handlers.load_post.remove(h)
    persistent = getattr(handlers, "persistent", None)
    handlers.load_post.append(persistent(_reset_run_python_namespace) if persistent else _reset_run_python_namespace)
    return True


_install_run_python_reset()


# RAY VISIBILITY MOVED OFF THE CYCLES ADDON onto the object itself in 3.0, so both spellings are
# tried, newest first - the same discipline the light shadow flag needed. A hardcoded single name
# is a silent no-op on half the Blenders this addon supports.
# MOVED TO ops_common 2026-09-15 and aliased rather than copied, because object_info reports these
# flags now - set_object_visibility wrote eight of them and this table was read from three lines,
# all inside that op, so a caller who did not make the write could not discover it. Two copies of a
# table that must agree is the drift the guard below exists to catch; one table cannot drift at all.
# The guard still runs, and still compares against the LITERAL _VISIBILITY_KEYS, which stays here.
_VIS_FLAGS = VIS_FLAGS


# A LITERAL, because parity_check resolves accepted-key sets statically and is fail-closed - it
# reads a set literal and refuses both a function-local build and a set comprehension rather than
# SKIP a check it cannot read. That is the right behaviour and it costs this duplication of the
# names in _VIS_FLAGS.
#
# So the duplication is GUARDED rather than trusted. The check below runs at import: it can only
# fail if somebody edits one list and not the other, which is exactly the moment you want to be
# stopped - and a key present in _VIS_FLAGS but missing here would be silently REFUSED at the door
# while looking supported everywhere else, which is the worst of both.
_VISIBILITY_KEYS = {
    "object", "name",
    "hideViewport", "hideRender", "hiddenInViewLayer",
    "visibleCamera", "visibleDiffuse", "visibleGlossy", "visibleTransmission",
    "visibleVolumeScatter", "visibleShadow",
    "holdout", "shadowCatcher",
    # DISPLAY (2026-10-04): how the object is drawn and whether it can be clicked - read by _display_plan
    "displayAs", "showWire", "showInFront", "showBounds", "boundsType", "hideSelect",
}

# (key, Object attribute, kind) - kind "enum" is checked against this build's own enum items.
_DISPLAY_FLAGS = (
    ("displayAs", "display_type", "enum"),
    ("showWire", "show_wire", bool),
    ("showInFront", "show_in_front", bool),
    ("showBounds", "show_bounds", bool),
    ("boundsType", "display_bounds_type", "enum"),
    ("hideSelect", "hide_select", bool),
)


def _display_readback(obj):
    """The display settings this op writes, read off the object."""
    out = {}
    for key, attr, _kind in _DISPLAY_FLAGS:
        if hasattr(obj, attr):
            v = getattr(obj, attr)
            out[key] = v if isinstance(v, str) else bool(v)
    return out


def _display_plan(obj, params):
    """[(attribute, value)] for the display keys given - every one checked before anything is written."""
    plan = []
    for key, attr, kind in _DISPLAY_FLAGS:
        if key not in params:
            continue
        if not hasattr(obj, attr):
            raise MifOpError("this Blender has no Object.%s, so %s would have been ignored. NOTHING was changed."
                             % (attr, key))
        if kind == "enum":
            items = [e.identifier for e in obj.bl_rna.properties[attr].enum_items]
            val = str(take(params, key, kind=str)).upper()
            if val not in items:
                raise MifOpError("%s must be one of this Blender's %s, got %r. NOTHING was changed."
                                 % (key, ", ".join(items), params.get(key)))
        else:
            val = take_bool(params, key)
        plan.append((attr, val))
    return plan

# WHY NOT indirectOnly - the key this op took until 2026-09-28, which wrote the SHADOW CATCHER (see ops_common.VIS_FLAGS).
_VISIBILITY_NOTES = {
    "indirectOnly": "Indirect Only is not an object setting - Blender keeps it per COLLECTION per view layer "
                    "(LayerCollection.indirect_only), and set_collection_visibility {indirectOnly} writes it. This op "
                    "took indirectOnly until 2026-09-28 and wrote the object's SHADOW CATCHER under that name; that "
                    "flag is spelt shadowCatcher.",
    "isShadowCatcher": "spelt shadowCatcher",
    "indirect_only": "see indirectOnly: a collection's flag, set_collection_visibility",
}

_missing = {k for k, _a, _c in _VIS_FLAGS} - _VISIBILITY_KEYS
if _missing:
    raise RuntimeError(
        "MifBlender ops_scene: _VIS_FLAGS names %s but _VISIBILITY_KEYS does not, so those keys "
        "would be refused by reject_unknown while every other part of the op supports them. Add "
        "them to the literal - it is duplicated on purpose so parity_check can read it."
        % ", ".join(sorted(_missing)))


# Both moved to ops_common beside VIS_FLAGS, for the reason given there. Aliased so the call sites
# in this file read as they did.
_vis_target = vis_target
_visibility_readback = visibility_readback


def op_set_object_visibility(params):
    """Control what an object is visible TO - the viewport, the render, and each ray type.

    WHY THIS IS THE MOST-WANTED TOGGLE. Stopping a softbox appearing as a white rectangle in every
    reflection is visibleGlossy=false, and it is the single most common adjustment in product and
    archviz lighting. Nothing in this addon could reach it. The general form also answers the other
    frequent question - "why is my object missing from the render" - because hide_render and the
    per-ray flags are exactly where that answer lives.

    Deliberately for EVERY object type rather than lights alone: ray visibility is an OBJECT
    property, and a mesh acting as a reflector or a holdout needs it as much as a lamp does.

    params:
      object (str, required)
      hideViewport / hideRender (bool)
      visibleCamera / visibleDiffuse / visibleGlossy / visibleTransmission /
      visibleVolumeScatter / visibleShadow (bool)
      hiddenInViewLayer (bool)       the outliner's eye for the ACTIVE view layer (Object.hide_set) - not a datablock
                                     flag, so it is per view layer and an object outside the layer is refused
      holdout (bool)                 Object.is_holdout
      shadowCatcher (bool)           Object.is_shadow_catcher, Cycles' Shadow Catcher. Indirect Only is a
                                     COLLECTION flag - set_collection_visibility {indirectOnly} - and is refused here
      displayAs                      Object.display_type: BOUNDS | WIRE | SOLID | TEXTURED (this build's enum)
      showWire / showInFront / showBounds (bool)   the wireframe overlay, drawn in front, the bounds box
      boundsType                     Object.display_bounds_type: BOX | SPHERE | CYLINDER | CONE | CAPSULE
      hideSelect (bool)              Object.hide_select - the object cannot be clicked in the viewport
                                     (a Boolean cutter shown as WIRE, a backdrop made unselectable)

    A flag this Blender does not expose is REFUSED by name rather than silently ignored - the
    mistake create_light's `shadow` made until it was fixed the same day.
    """
    reject_unknown(params, _VISIBILITY_KEYS, "set_object_visibility", _VISIBILITY_NOTES)
    want = take(params, "object", "name", default=None, kind=str)
    if not want:
        raise MifOpError("'object' is required. NOTHING was changed.")
    obj = bpy.data.objects.get(want)
    if obj is None:
        raise MifOpError("no object named '%s'. NOTHING was changed." % want)
    # hide_viewport AND hide_render LIVE ON THE OBJECT DATABLOCK, not on the local scene's base, so
    # on linked data they are as unsaveable as a transform. Measured on 5.0.1: this hid a linked
    # cube and reported before/after as fact, and the hide is gone on the next reload.
    require_editable(obj, "hide or show")

    # RESOLVE EVERY REQUESTED FLAG BEFORE WRITING ANY, so a flag this build lacks refuses the whole
    # call rather than leaving half of them applied.
    # THE VIEW LAYER'S HIDE is a method pair (hide_set / hide_get), not a flag the table below can set - parsed here,
    # with the other values, before anything is written.
    layer_hide = take_bool(params, "hiddenInViewLayer", default=None) if "hiddenInViewLayer" in params else None
    if layer_hide is not None and obj.name not in bpy.context.view_layer.objects:
        raise MifOpError("'%s' is not in the active view layer '%s', which is the layer hiddenInViewLayer hides it in. "
                         "NOTHING was changed." % (obj.name, bpy.context.view_layer.name))
    plan = []
    for key, attrs, cyc in _VIS_FLAGS:
        if key not in params:
            continue
        tgt = _vis_target(obj, attrs, cyc)
        if tgt is None:
            raise MifOpError("this Blender exposes no '%s' on an object (tried %s%s), so it would "
                             "have been silently ignored. NOTHING was changed."
                             % (key, ", ".join(attrs),
                                " and cycles.%s" % cyc if cyc else ""))
        plan.append((key, tgt, take_bool(params, key, default=True)))
    display = _display_plan(obj, params)
    if not plan and layer_hide is None and not display:
        raise MifOpError("no visibility flag was given. Pass at least one of %s. NOTHING was "
                         "changed." % ", ".join(sorted([k for k, _a, _c in _VIS_FLAGS] + ["hiddenInViewLayer"]
                                                       + [k for k, _a, _c in _DISPLAY_FLAGS])))

    before = dict(_visibility_readback(obj), **_display_readback(obj))
    for _key, (holder, attr), value in plan:
        setattr(holder, attr, value)
    for attr, value in display:
        setattr(obj, attr, value)
    if layer_hide is not None:
        obj.hide_set(layer_hide)
    after = dict(_visibility_readback(obj), **_display_readback(obj))
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    return {
        "object": obj.name,
        "before": before,
        "after": after,
        "changedFields": changed,
        "changedAnything": bool(changed),
        # THE QUESTION PEOPLE ACTUALLY ASK. Two flags decide whether this object appears at all,
        # and they are easy to set and then forget about.
        "appearsInRender": not after.get("hideRender", False)
                           and after.get("visibleCamera", True),
    }


_CUSTOM_SKIP = {"_RNA_UI", "cycles", "cycles_visibility"}


def _custom_holder(obj, bone_name, verb, data=False):
    """Object, its data block (data true - an armature's DATA, data["name"] in a driver path) or a pose bone. Custom
    properties live on all three and mean the same thing on each."""
    if data:
        if bone_name:
            raise MifOpError("pass bone OR data:true - a property lives on one of them. NOTHING was %s." % verb)
        if getattr(obj, "data", None) is None:
            raise MifOpError("'%s' is an %s with no data block. NOTHING was %s." % (obj.name, obj.type, verb))
        return obj.data, "data '%s'" % obj.data.name
    if not bone_name:
        return obj, "object"
    if obj.type != "ARMATURE" or obj.pose is None:
        raise MifOpError("'bone' was given but '%s' is not a posed ARMATURE. NOTHING was %s."
                         % (obj.name, verb))
    pb = obj.pose.bones.get(str(bone_name))
    if pb is None:
        raise MifOpError("no bone named '%s' on '%s'. NOTHING was %s."
                         % (bone_name, obj.name, verb))
    return pb, "bone '%s'" % pb.name


def _custom_rows(holder):
    rows = []
    for key in holder.keys():
        if key in _CUSTOM_SKIP:
            continue
        try:
            raw = holder[key]
        except (KeyError, TypeError):
            continue
        try:
            value = list(raw) if hasattr(raw, "__len__") and not isinstance(raw, str) else raw
        except TypeError:
            value = str(raw)
        row = {"key": key, "value": value, "type": type(raw).__name__}
        # UI metadata is a separate store from the value, and a rig slider is useless without it.
        try:
            ui = holder.id_properties_ui(key).as_dict()
            row["min"] = ui.get("min")
            row["max"] = ui.get("max")
            row["description"] = ui.get("description") or None
        except (AttributeError, TypeError, KeyError):
            row["min"] = row["max"] = row["description"] = None
        rows.append(row)
    return sorted(rows, key=lambda r: r["key"])


def op_list_custom_properties(params):
    """Custom properties on an object or pose bone, with their UI range.

    params:
      object (str, required)
      bone (str)   read the pose bone's instead
      data (bool)  read the object's DATA block's instead (an armature's Armature - data["name"] in a driver path)
      limit (int)  how many rows to return. Unlimited by default; `count` is always the true
                   total and `truncated` says when rows were dropped.

    Custom properties are how a rig exposes controls, and glTF writes them into the engine as
    `extras` - so they are metadata that travels, not just annotations. The UI min/max is a
    separate store from the value and is reported alongside, because a slider without a range is
    not a control.

    IT GROWS WITH THE OBJECT, WHICH IS WHY IT NEEDED A BOUND. Measured 2026-09-20 on 5.0.1: 200
    string properties on one cube answered 26,454 bytes - about 132 bytes each, and each row is five
    fields because the UI range is reported with the value. Metadata that travels is metadata
    somebody generates, and a per-face or per-part id scheme reaches hundreds on a single object
    without anything looking unusual. Same optional, unlimited-by-default shape as the other 22
    listings, so no existing caller sees a change.
    """
    reject_unknown(params, {"object", "name", "bone", "limit", "data"}, "list_custom_properties")
    obj = get_object(take(params, "object", "name", required=True))
    holder, label = _custom_holder(obj, take(params, "bone", default=None, kind=str), "read",
                                   take_bool(params, "data", default=False))
    rows = _custom_rows(holder)
    total = len(rows)
    rows, extra = limit_rows(rows, params, "custom propert(ies)", excludes=False)
    return dict({
        "object": obj.name,
        "owner": label,
        "count": total,
        "properties": rows,
        # Named rather than silently filtered: cycles settings live in the same namespace and are
        # not user metadata, so a count that included them would be wrong in a confusing way.
        "skippedInternalKeys": sorted(k for k in holder.keys() if k in _CUSTOM_SKIP),
    }, **extra)


def op_set_custom_property(params):
    """Set a custom property, and its UI range, on an object or pose bone.

    params:
      object (str, required)
      key (str, required)
      value                    number, bool, string or list
      bone (str)
      data (bool)              the object's DATA block instead (an armature's Armature, where a rig keeps a limb switch
                               or its rig_id; a driver reads it as data["name"]). Refused with bone
      min / max (float)        UI range - a rig slider without one is not a control
      description (str)
      delete (bool)            remove the property instead
      libraryOverridable (bool) mark the property overridable (property_overridable_library_set), so an animator can
                               key it on a library override of this object or bone - set in the SOURCE file, read
                               back with is_property_overridable_library. With no value it marks an existing key

    THE TYPE IS REPORTED BACK because Blender coerces silently: an int written where a float is
    expected stays an int, and a driver or an exporter reading it later gets a different type than
    the caller thinks they stored.
    """
    reject_unknown(params, {"object", "name", "key", "value", "bone", "min", "max",
                            "description", "delete", "data", "libraryOverridable"}, "set_custom_property")
    obj = get_object(take(params, "object", "name", required=True))
    holder, label = _custom_holder(obj, take(params, "bone", default=None, kind=str), "changed",
                                   take_bool(params, "data", default=False))
    key = take(params, "key", default=None, kind=str)
    if not key:
        raise MifOpError("'key' is required. NOTHING was changed.")
    key = str(key)
    if "\0" in key:
        # Blender's ID property API raises ValueError ("embedded null character") on such a key - refuse it first.
        raise MifOpError("'key' contains a null character, which Blender cannot store in a property name. "
                         "NOTHING was changed.")
    if key in _CUSTOM_SKIP:
        raise MifOpError("'%s' is an internal key, not user metadata - writing it would collide "
                         "with Blender's own storage. NOTHING was changed." % key)

    overridable = None if params.get("libraryOverridable") is None else take_bool(params, "libraryOverridable")
    if overridable is not None:
        if take_bool(params, "delete", default=False):
            raise MifOpError("libraryOverridable with delete - a deleted property has no flag. NOTHING was changed.")
        if not hasattr(holder, "property_overridable_library_set"):
            raise MifOpError("%s cannot carry an overridable flag. NOTHING was changed." % label)
        if "value" not in params:
            if key not in holder.keys():
                raise MifOpError("no custom property '%s' on %s to mark - pass value to create it. NOTHING was "
                                 "changed." % (key, label))
            if any(params.get(k) is not None for k in ("min", "max", "description")):
                raise MifOpError("min, max and description need value. NOTHING was changed.")
            path = '["%s"]' % key.replace("\\", "\\\\").replace('"', '\\"')
            holder.property_overridable_library_set(path, overridable)
            return {"object": obj.name, "owner": label, "key": key,
                    "libraryOverridable": bool(holder.is_property_overridable_library(path))}

    if take_bool(params, "delete", default=False):
        if key not in holder.keys():
            raise MifOpError("no custom property '%s' on %s to delete. NOTHING was changed."
                             % (key, label))
        del holder[key]
        holder.id_data.update_tag()      # see the write below: a subscript write reaches no evaluated copy on its own
        if key in holder.keys():
            raise MifOpError("deleted '%s' but it is still present. Do not trust this state." % key)
        return {"object": obj.name, "owner": label, "key": key, "deleted": True,
                "properties": _custom_rows(holder)}

    if "value" not in params:
        raise MifOpError("'value' is required unless delete:true. NOTHING was changed.")
    value = params.get("value")
    if isinstance(value, (list, tuple)):
        value = list(value)

    # THE RANGE IS CHECKED BEFORE THE VALUE IS WRITTEN, and typed to it. An INT property's UI range takes ints only:
    # MEASURED 2026-09-16 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1, id_properties_ui(key).update(min=0.0, max=9.0) on
    # an int property raises "TypeError: 'float' object cannot be interpreted as an integer" and min=0, max=9 works.
    # take_float made every bound a float, so {value: 3, min: 0, max: 9} wrote the value and then failed the range
    # (reported by the second seat, mailbox id 67). A whole-number bound is now handed over as an int; a fractional
    # one on an int property is refused, with nothing written.
    lo = take_float(params, "min", default=None)
    hi = take_float(params, "max", default=None)
    desc = take(params, "description", default=None, kind=str)
    if lo is not None and hi is not None and lo > hi:
        raise MifOpError("min %g is above max %g. NOTHING was changed." % (lo, hi))
    if isinstance(value, int) and not isinstance(value, bool):
        fractional = [b for b in (lo, hi) if b is not None and b != int(b)]
        if fractional:
            raise MifOpError("'%s' is an INTEGER property (value %d), and its slider range takes whole numbers - got %s. "
                             "Pass whole-number bounds, or a float value for a float property. NOTHING was changed."
                             % (key, value, ", ".join("%g" % b for b in fractional)))
        lo = int(lo) if lo is not None else None
        hi = int(hi) if hi is not None else None
    holder[key] = value
    # TAGGED, because a subscript write is not an RNA write and tells the depsgraph nothing. MEASURED 2026-09-16 on
    # 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1: once an object has an evaluated copy, obj["lift"] = 1.45 never reaches
    # it - a driver on ANOTHER object reading ["lift"] resolved nothing, and Blender flags such a driver invalid for
    # good (tagging afterwards did not revive it); a later change to 2.2 left a working driver at 1.45. Written then
    # update_tag(): valid, 1.45, and 2.2 after the change. Reported by the second seat (mailbox id 67, report 87 defect
    # 3); plain bpy did not reproduce it because a fresh script writes the property before anything is evaluated.
    # A pose bone's id_data is its armature object, so a bone's property is tagged the same way.
    holder.id_data.update_tag()
    ui_set = False
    if lo is not None or hi is not None or desc:
        try:
            ui = holder.id_properties_ui(key)
            kwargs = {}
            if lo is not None:
                kwargs["min"] = lo
            if hi is not None:
                kwargs["max"] = hi
            if desc:
                kwargs["description"] = str(desc)
            ui.update(**kwargs)
            ui_set = True
        except (AttributeError, TypeError, KeyError) as exc:
            raise MifOpError("the value was written but this Blender would not take the UI range "
                             "for '%s' (%s). The property EXISTS and has no slider bounds."
                             % (key, exc))

    path = '["%s"]' % key.replace("\\", "\\\\").replace('"', '\\"')
    if overridable is not None:
        holder.property_overridable_library_set(path, overridable)
    rows = [r for r in _custom_rows(holder) if r["key"] == key]
    return {
        "object": obj.name,
        "owner": label,
        "key": key,
        "property": rows[0] if rows else None,
        "libraryOverridable": (bool(holder.is_property_overridable_library(path))
                               if hasattr(holder, "is_property_overridable_library") else None),
        "uiRangeSet": ui_set,
        # BLENDER COERCES SILENTLY. An int written where a float was meant stays an int, and an
        # exporter or a driver reading it later gets a type the caller did not think they stored.
        "storedType": type(holder[key]).__name__,
        "requestedType": type(value).__name__,
        "typeChanged": type(holder[key]).__name__ != type(value).__name__,
    }


_SCENE_COPY_TYPES = ("NEW", "EMPTY", "LINK_COPY", "FULL_COPY")


def _copy_scene(params, want):
    tail = "NOTHING was changed."
    src_name = take(params, "copyFrom", default=None, kind=str)
    src = bpy.context.scene if src_name is None else bpy.data.scenes.get(src_name)
    if src is None:
        raise MifOpError("no scene named '%s'. Scenes: %s. %s"
                         % (src_name, ", ".join(s.name for s in bpy.data.scenes), tail))
    kind = str(take(params, "type", default="FULL_COPY", kind=str)).upper()
    if kind not in _SCENE_COPY_TYPES:
        raise MifOpError("type must be one of %s, got %r. %s" % (", ".join(_SCENE_COPY_TYPES), params.get("type"), tail))
    win = bpy.context.window or next((w for w in bpy.context.window_manager.windows), None)
    if win is None:
        raise MifOpError("this Blender has no window, and Blender's New Scene copies through one. %s" % tail)
    kinds = ("objects", "meshes", "materials", "collections", "cameras", "lights", "actions", "worlds")
    counts = {k: len(getattr(bpy.data, k)) for k in kinds}
    scenes_before = set(bpy.data.scenes)
    prev = win.scene
    try:
        win.scene = src
        with bpy.context.temp_override(window=win, scene=src):
            bpy.ops.scene.new(type=kind)
    except RuntimeError as exc:
        raise MifOpError("Blender refused New Scene (%s): %s. %s" % (kind, str(exc).strip(), tail))
    finally:
        made = [s for s in bpy.data.scenes if s not in scenes_before]
        win.scene = prev
    if len(made) != 1:
        raise MifOpError("Blender's New Scene made %d scenes. Check the scene list." % len(made))
    scene = made[0]
    assign_id_name(scene, want, bpy.data.scenes)
    return {"name": scene.name, "requestedName": want, "renamed": scene.name != want, "type": kind,
            "copyFrom": src.name, "objectCount": len(scene.objects), "collectionCount": len(scene.collection.children_recursive),
            "made": {k: len(getattr(bpy.data, k)) - counts[k] for k in kinds},
            "objectsShared": sum(1 for o in scene.objects if src in o.users_scene),
            "activeScene": bpy.context.window.scene.name if bpy.context.window else prev.name,
            "sceneCountAfter": len(bpy.data.scenes)}


def op_create_scene(params):
    """Create a scene. The only entity type a caller could NAME and not create.

    FOUND BY audit_blender_consumer_no_creator, which asks what the addon consumes and cannot
    produce - the same question that found collections, empties, armatures and vertex groups, and
    the fifth time it has paid. render_animation takes `scene` BY NAME and resolves it through
    bpy.data.scenes.get(), so a caller could render a scene somebody else had made and never make
    one, and there was no way to set up a second render configuration - a different resolution, a
    different frame range, a different output path - without editing the only scene there was.

    bpy.data.scenes.new(), NOT bpy.ops.scene.new(). The operator needs a window context and does
    nothing useful under `blender -b`, which is the path every suite and every showcase stage takes.
    The data-level call works identically in both, and this addon's threading contract runs jobs
    inline in background mode.

    THE NEW SCENE IS EMPTY AND THAT IS NOT A LIMITATION BEING HIDDEN. bpy.data.scenes.new() makes a
    scene with its own master collection holding nothing, its own render settings, its own frame
    range. It does not copy the current scene, and there is no supported data-level way to do a full
    copy without the operator and its context. The response says so rather than leaving a caller to
    discover that their objects are missing.

    NOT ACTIVATED EITHER, AND THE REASON GIVEN HERE USED TO BE FALSE. It read "Switching
    bpy.context.window.scene needs a window, so it cannot work in background mode". Measured
    2026-09-20 on 3.6.23, 4.2.17 LTS, 4.4.0, 5.0.1 and 5.2.1 LTS, `blender -b --factory-startup`
    has one window manager and ONE WINDOW, bpy.context.window is not None, and the assignment works
    on all five (tools/probe_blender_background_window.py, ledger B153). What really needs a UI context is
    bpy.ops.scene.*, which is a different thing and is covered two paragraphs up.

    So creation still does not activate, but now as a CHOICE rather than a limitation: building a
    scene should not move the caller out of the one they are working in, and everything that takes a
    scene - render_animation among them - takes it by NAME. The caller who does want to move is told
    what to call, which is set_current_scene.

    params:
      name (str)   required. The scene name. Blender appends .001 on a collision rather than
                   failing, so the name you get back may not be the name you asked for.
      copyFrom (str), type (str)   (1.3) a scene made from another, as Blender's New Scene menu: type NEW (an empty
                   scene, default settings), EMPTY (the source's settings, no objects), LINK_COPY (the source's
                   collections and objects shared), FULL_COPY (independent copies of objects and their data). Through
                   scene.new on the source in this Blender's window; the window's scene is put back, so nothing is
                   activated. Reply: objects, collections and the datablocks the copy made.
    """
    reject_unknown(params, {"name", "scene", "sceneName", "copyFrom", "type"}, "create_scene")
    want = take(params, "name", "scene", "sceneName", required=True, kind=str)
    want = str(want).strip()
    if not want:
        raise MifOpError("'name' is empty. A scene needs a name - it is how every other op that "
                         "takes a scene resolves it. NOTHING was changed.")
    if params.get("copyFrom") is not None or params.get("type") is not None:
        return _copy_scene(params, want)

    before = [s.name for s in bpy.data.scenes]
    scene = bpy.data.scenes.new(want)
    if scene is None:
        # Documented as always returning the new scene; checked rather than assumed, because a
        # None here would otherwise surface as an AttributeError from the read-back below.
        raise MifOpError("bpy.data.scenes.new(%r) returned None. NOTHING was changed." % want)

    # THE NAME BLENDER ACTUALLY GAVE IT. A duplicate name gets a .001 suffix and the incumbent is
    # left alone, so a caller who then asks for their scene by the string they sent would get the
    # OLD one. Reported rather than refused - a second scene called "Render" is a legitimate thing
    # to want - but reported loudly, because silence here is how the wrong scene gets rendered.
    renamed = scene.name != want
    return {
        "name": scene.name,
        "requestedName": want,
        "renamed": renamed,
        "renamedNote": ("Blender already had a scene called '%s', so the new one is '%s'. Use the "
                        "name in `name`, not the one you sent." % (want, scene.name))
                       if renamed else None,
        "objectCount": len(scene.objects),
        "emptyNote": "A new scene is EMPTY - it does not copy the current one. Link objects into "
                     "it, or build it up, before rendering.",
        "activeScene": bpy.context.scene.name if bpy.context.scene else None,
        # THE REASON THIS FIELD USED TO GIVE WAS FALSE. It read "that needs a window and cannot work
        # under blender -b" until 2026-09-20, when `blender -b --factory-startup` was measured on
        # 3.6.23, 4.2.17 LTS, 4.4.0, 5.0.1 and 5.2.1 LTS and found to have one window in every
        # one, with bpy.context.window not None and window.scene assignable. create_scene still does not
        # activate what it makes - that is a choice, so a caller building a scene is not moved out
        # of the one they are working in - but the caller is now told what to call, not that it is
        # impossible. See tools/probe_blender_background_window.py.
        "activeNote": "The new scene is NOT activated - the one you were in stays current. Switch "
                      "with set_current_scene, or leave it and pass the name to ops that take a "
                      "scene.",
        "sceneCountBefore": len(before),
        "sceneCountAfter": len(bpy.data.scenes),
        "scenes": [s.name for s in bpy.data.scenes],
    }


_DELETE_SCENE_KEYS = {"name", "scene", "sceneName"}


def op_delete_scene(params):
    """Remove a scene. The counterpart create_scene never had.

    EVERY OTHER CREATABLE THING HAS A REMOVER - delete_object, delete_collection,
    delete_view_layer, remove_constraint, remove_driver, remove_modifier, remove_node - and a scene
    did not. So a scene made through the bridge lived for the rest of the Blender session and was
    written into the .blend if the file was saved, with no way back except run_python, which is off
    by default and is not an answer. It bit a suite immediately: test_blender_listings left
    MifLsScene behind, the next run added MifLsScene.001, and a row about nameExcludes failed on the
    accumulation rather than on the thing it was testing.

    THREE REFUSALS, and each is a way to make a file Blender cannot use:

      * the scene does not exist - named separately from the others so a typo does not read as a
        policy refusal;
      * it is the LAST scene. bpy.data.scenes.remove() will happily take it and leave a file with
        none, which Blender's own UI cannot produce and does not expect;
      * it is the ACTIVE scene. Removing it leaves bpy.context.scene dangling. Refused rather than
        half-done, and the message names the fix, which is set_current_scene.
        THIS REFUSAL USED TO GIVE A FALSE REASON - "re-pointing it needs bpy.context.window, which
        does not exist under `blender -b`". Measured 2026-09-20 on 3.6.23, 4.2.17 LTS, 4.4.0 and
        5.0.1: `blender -b --factory-startup` has one window manager and one window on every one of
        them, bpy.context.window is not None, and window.scene is assignable. That false premise is
        why there was no switch op for six days; it is now set_current_scene, and the refusal is
        kept on its own merits - deleting the scene you are standing in should be deliberate.

    THE OBJECTS DO NOT GO WITH IT, and that is the thing a caller is most likely to get wrong.
    Removing a scene unlinks its master collection; the objects themselves stay in bpy.data.objects
    with zero users and are purged on the next save-and-reload, not now. So the count this returns
    is what was UNLINKED, not what was deleted, and it says so in those words.

    params:
      name / scene / sceneName (str)   required. The scene to remove.
    """
    reject_unknown(params, _DELETE_SCENE_KEYS, "delete_scene")
    want = take(params, "name", "scene", "sceneName", required=True, kind=str)
    want = str(want).strip()
    scene = bpy.data.scenes.get(want)
    if scene is None:
        raise MifOpError(
            "no scene called %r. Present: %s. NOTHING was changed."
            % (want, ", ".join(sorted(s.name for s in bpy.data.scenes)) or "(none)"))
    if len(bpy.data.scenes) <= 1:
        raise MifOpError(
            "'%s' is the only scene in this file, and Blender cannot open a file with none. "
            "Make another scene first with create_scene. NOTHING was changed." % want)
    active = bpy.context.scene
    if active is not None and active.name == scene.name:
        # THE REASON THIS REFUSAL GAVE WAS FALSE, and it was the load-bearing half of the sentence.
        # It said re-pointing "needs bpy.context.window, which does not exist under 'blender -b'";
        # measured 2026-09-20, `blender -b --factory-startup` has one window on 3.6.23, 4.2.17 LTS,
        # 4.4.0, 5.0.1 and 5.2.1 LTS and the assignment works on all five. The REFUSAL still stands
        # - deleting the scene you are standing in is worth a deliberate step, and doing the switch
        # silently would move the caller somewhere they did not ask to be - but it now names the op
        # that fixes it instead of telling them it cannot be done.
        raise MifOpError(
            "'%s' is the ACTIVE scene, and removing it would leave bpy.context.scene dangling. "
            "Switch first with set_current_scene - %s %s available. NOTHING was changed."
            % (want,
               ", ".join(sorted(s.name for s in bpy.data.scenes if s.name != want)),
               "is" if len(bpy.data.scenes) == 2 else "are"))

    # COUNTED BEFORE THE REMOVE, because scene.objects is a view onto the scene and reading it
    # afterwards would either be empty or raise on a freed datablock.
    unlinked = [o.name for o in scene.objects]
    before = sorted(s.name for s in bpy.data.scenes)
    bpy.data.scenes.remove(scene)

    # READ BACK. The op's whole claim is that the scene is gone, and a remove that silently did
    # nothing would otherwise be reported as success.
    if bpy.data.scenes.get(want) is not None:
        raise MifOpError("removed scene '%s' and it is still in bpy.data.scenes." % want)

    return {
        "ok": True,
        "deleted": want,
        "sceneCountBefore": len(before),
        "sceneCountAfter": len(bpy.data.scenes),
        "scenes": sorted(s.name for s in bpy.data.scenes),
        "activeScene": bpy.context.scene.name if bpy.context.scene else None,
        # UNLINKED, not deleted, and the field is named for what happened rather than for what a
        # reader might assume. These objects still exist with no users; they go when the file is
        # saved and reloaded, or when something else removes them.
        "objectsUnlinked": len(unlinked),
        "objectNames": sorted(unlinked)[:50],
        "objectNote": "Removing a scene does NOT remove the objects it held - they stay in "
                      "bpy.data.objects with no users and are purged on the next save and reload. "
                      "Use delete_object on any you want gone now.",
    }


_SET_CURRENT_SCENE_KEYS = {"name", "scene", "sceneName", "allWindows"}


def op_set_current_scene(params):
    """Switch the ACTIVE scene - the op three other ops already told callers to use.

    IT WAS NAMED BEFORE IT EXISTED. `list_scenes` answered a caller who passed `name` with "this
    LISTS scenes; set_current_scene is what switches to one", and there was no such op anywhere in
    the repo - that string was its only occurrence. A caller following the advice got "unknown
    endpoint".

    AND THE REASON GIVEN FOR NOT HAVING IT WAS MEASURABLY FALSE. create_scene's activeNote said the
    new scene "is NOT activated - that needs a window and cannot work under blender -b", and
    delete_scene refused the active scene because "re-pointing it needs bpy.context.window, which
    does not exist under 'blender -b'". Measured 2026-09-20 with `blender -b --factory-startup` on
    3.6.23, 4.2.17 LTS, 4.4.0, 5.0.1 and 5.2.1 LTS: there is one window manager and ONE WINDOW in
    every one of them, bpy.context.window is not None, `bpy.context.window.scene = other` raises
    nothing, and bpy.context.scene follows it immediately. The premise was wrong on every build this addon
    supports, and it had propagated into two response fields, one refusal, the MCP wrapper's
    docstring and the spec. tools/probe_blender_background_window.py re-runs the measurement.

    WHAT SWITCHING ACTUALLY MOVES, and it is more than the name. bpy.context.scene is what almost
    every op in this addon reads - render settings, frame range, the active view layer, the world,
    the object collections `list_objects` walks. So this is not cosmetic: it re-points the whole
    read surface, which is why it reports the frame range, the world and the object count of both
    scenes rather than only the new name.

    ONE WINDOW OR ALL OF THEM. A GUI Blender can have several windows open on different scenes on
    purpose, so the default moves only the context window and REPORTS the others rather than
    silently leaving them - `windows` carries a row per window. allWindows:true moves every one.
    Under `blender -b` there is exactly one window and the two are the same thing.

    Idempotent: asking for the scene that is already current is not a refusal, it answers
    changed:false. A refusal there would make "make sure we are in Shot_B" a two-call dance.

    params:
      name / scene / sceneName (str)   required. The scene to switch to.
      allWindows (bool)                default false. Move every window, not just the context one.
    """
    reject_unknown(params, _SET_CURRENT_SCENE_KEYS, "set_current_scene",
                   notes={
                       "create": "this SWITCHES to a scene that exists; create_scene makes one, "
                                 "and does not activate what it makes",
                       "activate": "see `name` - this op IS the activation",
                       "window": "which window is not selectable; pass allWindows to move them "
                                 "all, or the context window is used",
                   })
    want = str(take(params, "name", "scene", "sceneName", required=True, kind=str)).strip()
    every = take_bool(params, "allWindows", default=False)

    scene = bpy.data.scenes.get(want)
    if scene is None:
        raise MifOpError(
            "no scene called %r. Present: %s. list_scenes reports them with a `current` flag. "
            "NOTHING was changed."
            % (want, ", ".join(sorted(s.name for s in bpy.data.scenes)) or "(none)"))

    before = bpy.context.scene.name if bpy.context.scene else None
    window = getattr(bpy.context, "window", None)
    targets = [w for m in bpy.data.window_managers for w in m.windows] if every else (
        [window] if window is not None else [])
    if not targets:
        # NOT REACHABLE ON ANY BUILD MEASURED - `blender -b` has a window on 3.6 through 5.0 - but a
        # scene is only switchable through a window, so an embedding with none gets the reason
        # rather than an AttributeError. Named as a measurement so a future reader can re-check it.
        raise MifOpError(
            "this Blender (%s, background=%s) has no window to re-point, and the active scene is a "
            "property of a window. %d window manager(s), %d window(s). NOTHING was changed."
            % (bpy.app.version_string, bpy.app.background, len(bpy.data.window_managers),
               sum(len(m.windows) for m in bpy.data.window_managers)))

    moved = []
    for i, w in enumerate(targets):
        if w.scene == scene:
            continue
        try:
            w.scene = scene
        except Exception as exc:                                              # noqa: BLE001
            raise MifOpError("Blender refused to put window %d on scene '%s': %s. %s"
                             % (i, want, str(exc).strip(),
                                "Earlier windows were moved." if moved else "NOTHING was changed."))
        moved.append(i)

    # READ BACK through bpy.context, which is what every other op in this addon reads. The write
    # goes to window.scene and the claim is about context.scene, and those are two different
    # attributes - a switch that moved the window and left the context behind would otherwise be
    # reported as success.
    now = bpy.context.scene.name if bpy.context.scene else None
    if now != want:
        raise MifOpError(
            "set window.scene to '%s' and bpy.context.scene is still '%s'. The active scene did "
            "NOT move; read list_scenes before continuing." % (want, now))

    view_layer = bpy.context.view_layer.name if getattr(bpy.context, "view_layer", None) else None
    return {
        "scene": now,
        "requested": want,
        "previous": before,
        "changed": now != before,
        "windowsMoved": len(moved),
        "allWindows": every,
        # A ROW PER WINDOW, because the default moves ONE and a GUI session can have more. A caller
        # who sees a window still on the old scene here knows it without having to ask again.
        #
        # `==`, NOT `is`. bpy.context.window hands back a FRESH wrapper on every access, so the
        # identity test read False for the very window it had just moved - measured 2026-09-20 on
        # 5.0.1, where the only window in a `-b` session reported current:false. bpy_struct
        # compares by the datablock it wraps, so `==` answers the question that was meant.
        "windows": [{"index": i, "scene": w.scene.name if w.scene else None,
                     "current": window is not None and w == window}
                    for i, w in enumerate([w2 for m in bpy.data.window_managers
                                           for w2 in m.windows])][:32],
        "windowCount": sum(len(m.windows) for m in bpy.data.window_managers),
        # WHAT MOVED WITH IT. The name is the least of it: these are the per-scene settings every
        # other op reads off bpy.context.scene, and they are what makes a wrong active scene render
        # the wrong frames at the wrong resolution.
        "objectCount": len(scene.objects),
        "frameStart": scene.frame_start,
        "frameEnd": scene.frame_end,
        "frameCurrent": scene.frame_current,
        "engine": scene.render.engine,
        "world": scene.world.name if scene.world else None,
        "camera": scene.camera.name if scene.camera else None,
        "viewLayer": view_layer,
        "activeNote": ("bpy.context.scene is now '%s'. Every op that does not take a scene by name "
                       "reads THIS one - render settings, frame range, active view layer, world "
                       "and the objects list_objects walks." % now),
    }


_LIST_SCENES_KEYS = ("limit", "nameExcludes")


def op_list_scenes(params):
    """Every scene in the file, by name, and which one is current.

    THE THIRD COLLECTION OF THIS SHAPE, found 2026-09-08 by audit_blender_unlistable the day it was
    written. create_scene makes one and several ops take a scene NAME, but scene_info reports the
    CURRENT scene and never touches bpy.data.scenes - so a caller working in a multi-scene file had
    no way to learn what the other scenes were called. Unlike images and node_groups there was not
    even a refusal to read them out of.

    `current` is reported per row rather than as a separate field, because the question "what scenes
    are there" and "which am I in" are asked together and answering them apart invites a caller to
    pair them wrongly.
    """
    reject_unknown(params, _LIST_SCENES_KEYS, "list_scenes",
                   notes={
                       "name": "this LISTS scenes; set_current_scene is what switches to one",
                       "scene": "see `name` - this is the enumerator, not a selector",
                       "current": "each row carries its own `current` flag; there is nothing to pass",
                   })
    cur = getattr(bpy.context, "scene", None)
    cur_name = cur.name if cur else None
    rows = []
    for sc in bpy.data.scenes:
        rows.append({
            "name": sc.name,
            "current": sc.name == cur_name,
            "objectCount": len(sc.objects),
            "frameStart": int(sc.frame_start),
            "frameEnd": int(sc.frame_end),
            # THE PREVIEW RANGE OVERRIDES THE ONE ABOVE, and until 2026-09-15 it was reported
            # nowhere. set_frame_range writes frame_preview_start/end and turns use_preview_range
            # ON as a side effect of being given either - so a scene can report frameStart 1 and
            # frameEnd 250 while every render, playback and bake actually runs 100..200, with
            # nothing in any response saying so. That is worse than the usual write-only case,
            # because the value a caller DOES get back is then wrong about the scene rather than
            # merely incomplete.
            #
            # usePreviewRange is the gate, reported beside the pair it decides - seventh of these
            # found in a day, and the one audit_value_without_gate could not see until it learned to
            # pair a flag with values in nested scopes that share its holder.
            "usePreviewRange": bool(sc.use_preview_range),
            "previewFrameStart": int(sc.frame_preview_start),
            "previewFrameEnd": int(sc.frame_preview_end),
            "users": int(getattr(sc, "users", 0)),
            "fakeUser": bool(getattr(sc, "use_fake_user", False)),
        })
    rows.sort(key=lambda r: r["name"])
    rows = exclude_rows(rows, params)
    matched = len(rows)
    rows, extra = limit_rows(rows, params, "scene(s)")
    return dict({"scenes": rows, "count": matched, "current": cur_name}, **extra)


# ---------------------------------------------------------------------------
# call_operator, list_recent_operators, enable_addon - behind Allow run_python (2026-10-05)
# ---------------------------------------------------------------------------
# DECIDED 2026-10-04 (FEATURE_PARITY_SPEC "Run a registered Blender operator by id ..."): a generic operator call and
# an add-on install both run arbitrary code, so they sit behind the run_python preference.

# REFUSED BY ID: what quits or replaces the session (the server dies with it), what writes the user's files or
# preferences (save_file is the typed, owner-checked save), and what reloads add-ons - MifBlender among them.
_OPERATOR_DENY = {
    "wm.quit_blender": "it quits Blender, and the bridge with it",
    "wm.read_homefile": "it replaces the session - clear_scene, or run_python read_homefile(use_empty=True)",
    "wm.read_factory_settings": "it disables every add-on, MifBlender with it",
    "wm.read_factory_userpref": "it disables every add-on, MifBlender with it",
    "wm.read_userpref": "it reloads the preferences over the session's",
    "wm.open_mainfile": "it replaces the session - open_file is the typed call",
    "wm.revert_mainfile": "it replaces the session",
    "wm.recover_last_session": "it replaces the session",
    "wm.recover_auto_save": "it replaces the session",
    "wm.save_mainfile": "it writes the user's .blend - save_file is the typed, checked save",
    "wm.save_as_mainfile": "it writes a .blend - save_file is the typed, checked save",
    "wm.save_homefile": "it overwrites the user's startup file",
    "wm.save_userpref": "it writes the user's preferences",
    "script.reload": "it reloads every add-on, MifBlender with it",
    "preferences.addon_install": "enable_addon {file} is the call that installs",
    "preferences.addon_enable": "enable_addon is the call that enables",
    "preferences.addon_disable": "enable_addon {enable: false} is the call that disables",
    "preferences.addon_remove": "enable_addon {uninstall: true} is the call that removes",
    "wm.window_close": "it closes a window of the session",
}
# THE TYPED OP, named in the reply when one does the same job with checks this generic call cannot make.
_OPERATOR_TYPED = {
    "mesh.primitive_": "create_primitive", "object.modifier_add": "add_modifier",
    "object.modifier_apply": "apply_modifier", "object.modifier_remove": "remove_modifier",
    "object.delete": "delete_object", "object.duplicate": "duplicate_object", "object.join": "join_objects",
    "mesh.separate": "separate_mesh", "mesh.subdivide": "subdivide_mesh", "object.shade_smooth": "set_shading",
    "object.shade_flat": "set_shading", "object.transform_apply": "apply_transform", "object.origin_set": "set_origin",
    "uv.smart_project": "uv_unwrap", "uv.unwrap": "uv_unwrap", "mesh.remove_doubles": "clean_mesh",
    "object.convert": "convert_object", "render.render": "render_still", "export_scene.fbx": "export_mesh",
    "import_scene.fbx": "import_mesh", "anim.keyframe_insert": "set_keyframe", "object.parent_set": "set_parent",
    "object.material_slot_add": "set_material_slots", "mesh.normals_make_consistent": "clean_mesh",
    "mesh.flip_normals": "clean_mesh", "mesh.bevel": "bevel_edges", "mesh.extrude_region": "extrude_mesh",
}
_BRIDGE_OPERATOR_CALLS = []


def _require_run_python_gate(op):
    if not getattr(_prefs(), "allow_run_python", False):
        raise MifOpError("%s is refused: it runs code with the user's privileges, so it is gated by the same Allow "
                         "run_python preference as run_python, and that is off. %s NOTHING was done." % (op, RUN_PYTHON_HELP))


def _operator_by_id(op_id):
    """The bpy.ops callable for 'module.name', or a refusal naming the near ids."""
    if not isinstance(op_id, str) or op_id.count(".") != 1:
        raise MifOpError("id is 'module.name', as Blender's Python tooltips show it (mesh.primitive_monkey_add); got %r. "
                         "NOTHING was done." % (op_id,))
    mod_name, fn_name = op_id.split(".")
    mod = getattr(bpy.ops, mod_name, None)
    names = [n for n in dir(mod) if not n.startswith("_")] if mod is not None else []
    if fn_name not in names:
        import difflib   # noqa: PLC0415 - local, only a refusal needs it
        pool = names if names else [m for m in dir(bpy.ops) if not m.startswith("_")]
        near = difflib.get_close_matches(fn_name if names else mod_name, pool, n=5, cutoff=0.6)
        raise MifOpError("no operator %r on Blender %s.%s NOTHING was done."
                         % (op_id, bpy.app.version_string,
                            (" Near: %s." % ", ".join(("%s.%s" % (mod_name, n)) if names else n for n in near))
                            if near else ""))
    return getattr(mod, fn_name)


def _operator_props(rna):
    rows = []
    for prop in rna.properties:
        if prop.identifier == "rna_type":
            continue
        row = {"id": prop.identifier, "type": prop.type, "name": prop.name, "description": prop.description or None}
        if prop.type == "ENUM":
            row["items"] = [e.identifier for e in prop.enum_items]
            row["default"] = sorted(prop.default_flag) if prop.is_enum_flag else prop.default
            row["multiple"] = bool(prop.is_enum_flag)
        elif prop.type in ("INT", "FLOAT", "BOOLEAN"):
            n = getattr(prop, "array_length", 0)
            row["default"] = list(prop.default_array) if n else prop.default
            if n:
                row["length"] = n
            if prop.type in ("INT", "FLOAT"):
                row["min"], row["max"] = prop.hard_min, prop.hard_max
        elif prop.type == "STRING":
            row["default"] = prop.default
        rows.append(row)
    return rows


def _check_operator_props(op_id, rna, props):
    """Every property checked against the operator's RNA before it runs: the name, the enum item, the range - Blender
    clamps a number outside its hard range in silence (size -5 on primitive_monkey_add made a monkey of no size)."""
    if props is None:
        return {}
    if not isinstance(props, dict):
        raise MifOpError("properties is {name: value}; got %r. NOTHING was done." % (props,))
    known = {p.identifier: p for p in rna.properties if p.identifier != "rna_type"}
    for key, val in props.items():
        prop = known.get(key)
        if prop is None:
            raise MifOpError("%s has no property %r. It has: %s. NOTHING was done." % (op_id, key, ", ".join(known)))
        if prop.type == "ENUM":
            items = [e.identifier for e in prop.enum_items]
            vals = val if (prop.is_enum_flag and isinstance(val, (list, tuple, set))) else [val]
            bad = [v for v in vals if v not in items]
            # A DYNAMIC enum (items made by a callback) reads as Blender's one-item placeholder [DEFAULT] from the
            # operator's RNA type, enum_items and enum_items_static alike: rigidbody.mass_calculate's material refused
            # "takes DEFAULT; Iron is not one" (5.0.1, 2026-10-15). Blender checks such a value itself when the operator
            # is called and raises TypeError, which the call below reports with nothing done.
            if bad and items and items != ["DEFAULT"]:
                raise MifOpError("%s.%s takes %s; %s is not one. NOTHING was done."
                                 % (op_id, key, ", ".join(items[:40]), ", ".join(map(str, bad))))
        elif prop.type in ("INT", "FLOAT"):
            nums = val if isinstance(val, (list, tuple)) else [val]
            for v in nums:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
                    raise MifOpError("%s.%s takes a number, got %r. NOTHING was done." % (op_id, key, val))
                if v < prop.hard_min or v > prop.hard_max:
                    raise MifOpError("%s.%s %r is outside its range (minimum %g, maximum %g) - Blender would clamp it "
                                     "in silence. NOTHING was done." % (op_id, key, v, prop.hard_min, prop.hard_max))
    return dict(props)


def op_call_operator(params):
    """Run a registered Blender operator by id - gated by Allow run_python.

    id 'module.name' (mesh.primitive_monkey_add); properties {name: value}, each checked against the operator's own RNA
    first - an unknown name, an enum item it lacks, or a number outside its hard range (which Blender clamps in silence)
    is refused; context {object (made active and selected), selected [names], mode (OBJECT, EDIT ... switched to before
    and back after)}; describe true lists the operator's properties on this build (defaults, enum items, ranges) and runs
    nothing. undo true runs it as Blender's UI does - an undo step pushed and the operator added to Blender's own
    operator log (WindowManager.operators, which list_recent_operators reads); false (default) is bpy.ops's own default,
    which adds neither. MEASURED 2026-10-09 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1 headless: bpy.ops.X() leaves the
    log empty, bpy.ops.X('EXEC_DEFAULT', True) puts X in it - background mode included. The call is refused when the
    operator's poll fails in that context, and by id for what quits, replaces or saves the session or reloads add-ons
    (_OPERATOR_DENY). Reply: result, objectsAdded / objectsRemoved, the active object, logged (undo true: whether the
    operator is now the log's last entry), and typedOp when a typed op does the same job with checks this call cannot
    make. A RuntimeError's text (Blender's report) is the refusal.
    """
    reject_unknown(params, ("id", "properties", "context", "describe", "undo"), "call_operator")
    _require_run_python_gate("call_operator")
    op_id = take(params, "id", required=True, kind=str)
    if op_id in _OPERATOR_DENY:
        raise MifOpError("%s is denied: %s. NOTHING was done." % (op_id, _OPERATOR_DENY[op_id]))
    fn = _operator_by_id(op_id)
    rna = fn.get_rna_type()
    typed = next((v for k, v in _OPERATOR_TYPED.items() if op_id == k or (k.endswith("_") and op_id.startswith(k))),
                 None)
    undo = take_bool(params, "undo", default=False)
    if take_bool(params, "describe", default=False):
        if params.get("properties") is not None or params.get("context") is not None or undo:
            raise MifOpError("describe lists the operator and runs nothing; properties, context and undo are for a "
                             "call. NOTHING was done.")
        return {"id": op_id, "name": rna.name, "description": rna.description or None,
                "properties": _operator_props(rna), "typedOp": typed}
    props = _check_operator_props(op_id, rna, params.get("properties"))
    ctx = params.get("context") or {}
    if not isinstance(ctx, dict) or set(ctx) - {"object", "selected", "mode"}:
        raise MifOpError("context is {object, selected, mode}; got %r. NOTHING was done." % (ctx,))
    obj = get_object(ctx["object"]) if ctx.get("object") is not None else None
    sel = [get_object(n) for n in ctx.get("selected") or []]
    if obj is not None and obj not in sel:
        sel.append(obj)
    mode = str(ctx["mode"]).upper() if ctx.get("mode") is not None else None
    if mode is not None and obj is None:
        raise MifOpError("context.mode needs context.object - a mode is the object's. NOTHING was done.")
    override = {}
    if obj is not None:
        override.update(object=obj, active_object=obj)
    if sel:
        override.update(selected_objects=sel, selected_editable_objects=sel)
    before = set(bpy.data.objects.keys())
    vl = bpy.context.view_layer
    active_before = vl.objects.active
    sel_before = [o for o in bpy.data.objects if o.select_get()]
    try:
        if obj is not None:
            vl.objects.active = obj
            for o in bpy.data.objects:
                o.select_set(o in sel)
        if mode is not None and mode != "OBJECT":
            with bpy.context.temp_override(**override):
                bpy.ops.object.mode_set(mode=mode)
        with bpy.context.temp_override(**override):
            if not fn.poll():
                raise MifOpError("%s's poll failed in this context (%s) - Blender would not run it here. Give it what it "
                                 "acts on with context {object, selected, mode}. NOTHING was done."
                                 % (op_id, ", ".join("%s %s" % (k, getattr(v, "name", v)) for k, v in
                                                     (("object", obj), ("mode", mode))) or "no object"))
            log_before = len(bpy.context.window_manager.operators)
            try:
                # undo is bpy.ops's second positional argument; True keeps the call at undo depth 0, which is what
                # puts a finished operator into WindowManager.operators, as a UI-run one is.
                result = fn("EXEC_DEFAULT", True, **props) if undo else fn(**props)
            except RuntimeError as exc:
                raise MifOpError("%s raised: %s" % (op_id, str(exc).strip()))
            except TypeError as exc:
                raise MifOpError("%s refused its properties: %s. NOTHING was done." % (op_id, exc))
    finally:
        if mode is not None and mode != "OBJECT" and bpy.context.mode != "OBJECT":
            try:
                with bpy.context.temp_override(**override):
                    bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:  # noqa: BLE001 - the mode put back is best effort, the result stands
                pass
        if obj is not None:
            try:
                for o in bpy.data.objects:
                    o.select_set(o in sel_before)
                vl.objects.active = active_before if active_before is None or active_before.name in bpy.data.objects \
                    else None
            except ReferenceError:
                pass
    after = set(bpy.data.objects.keys())
    res = sorted(result)
    _BRIDGE_OPERATOR_CALLS.append({"id": op_id, "result": res, "properties": jsonable(props)})
    del _BRIDGE_OPERATOR_CALLS[:-100]
    if "CANCELLED" in res:
        raise MifOpError("%s returned CANCELLED - it ran and, by Blender's own account, changed nothing." % op_id)
    act = vl.objects.active
    logged = None
    if undo:
        # The log keeps its last 32 entries, so its length is no measure; its last entry is.
        log = list(bpy.context.window_manager.operators)
        logged = bool(log) and _logged_id(log[-1]) == op_id and (len(log) != log_before or len(log) >= 32)
    return {"id": op_id, "result": res, "properties": jsonable(props), "objectsAdded": sorted(after - before),
            "objectsRemoved": sorted(before - after), "activeObject": act.name if act is not None else None,
            "undo": undo, "logged": logged, "typedOp": typed,
            "typedOpNote": ("%s does this job with the bridge's own checks and read-back - prefer it." % typed)
            if typed else None}


def _logged_id(o):
    """'module.name' of an entry in WindowManager.operators - its bl_idname is MODULE_OT_name."""
    mod, sep, name = o.bl_idname.partition("_OT_")
    return ("%s.%s" % (mod.lower(), name)) if sep else o.bl_idname.lower()


def op_list_recent_operators(params):
    """WindowManager.operators - Blender's log of the last operators run - and the calls made through call_operator.
    Gated by Allow run_python. Rows are {id, name, properties}, oldest first; Blender keeps the last 32.
    WHAT ENTERS THE LOG, MEASURED 2026-10-09 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1 headless: an operator that
    finishes at undo depth 0 - one run from the UI, or from Python with bpy.ops's undo argument true (call_operator
    {undo: true}). bpy.ops's default (undo false) adds nothing, which is why a monkey and a cube made through
    call_operator and run_python left a 5.2.1 GUI session's log empty on 2026-10-06, and why the log read empty headless
    on 2026-10-05: background mode keeps it like a GUI does. bridgeCalls is the record of every call_operator call."""
    reject_unknown(params, ("limit",), "list_recent_operators")
    _require_run_python_gate("list_recent_operators")
    limit = take_int(params, "limit", default=20)
    if not 1 <= limit <= 200:
        raise MifOpError("limit must be 1-200, got %d." % limit)
    rows = []
    for o in list(bpy.context.window_manager.operators)[-limit:]:
        props = {}
        try:
            for prop in o.properties.bl_rna.properties:
                if prop.identifier != "rna_type":
                    props[prop.identifier] = jsonable(getattr(o.properties, prop.identifier, None))
        except Exception:  # noqa: BLE001 - a property that will not read is left out, the row stays
            pass
        rows.append({"id": _logged_id(o), "name": o.name, "properties": props})
    out = {"operators": rows, "count": len(rows), "bridgeCalls": _BRIDGE_OPERATOR_CALLS[-limit:],
           "background": bool(bpy.app.background)}
    if not rows:
        out["note"] = ("the operator log is empty. It holds operators that finished at undo depth 0 - run from Blender's "
                       "UI, or through call_operator with undo true; bpy.ops's default (undo false: call_operator's "
                       "default, run_python, scripts) adds nothing. bridgeCalls are the calls made through "
                       "call_operator in this session.")
    return out


def _extension_manifest(src):
    """The blender_manifest.toml of an extension package .zip - at the zip's root or in its one top folder - parsed, or
    None when the zip has none (a legacy add-on zip)."""
    import zipfile   # noqa: PLC0415
    with zipfile.ZipFile(src) as zf:
        names = [n for n in zf.namelist() if n.rstrip("/").split("/")[-1] == "blender_manifest.toml"
                 and n.count("/") <= 1]
        if not names:
            return None
        raw = zf.read(sorted(names, key=len)[0])
    # BEFORE THE PARSE: 3.6.23 ships Python 3.10, which has no tomllib, so a package there answered a bare
    # "ModuleNotFoundError: No module named 'tomllib'" instead of this refusal (measured 2026-10-07).
    if bpy.app.version < (4, 2, 0):
        raise MifOpError("%r is an extension package (blender_manifest.toml), which needs Blender 4.2 or later; this is "
                         "%s. NOTHING was done." % (src, bpy.app.version_string))
    import tomllib   # noqa: PLC0415 - Python 3.11+, which every 4.2+ build ships
    try:
        return tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise MifOpError("the package's blender_manifest.toml does not parse: %s. NOTHING was done." % exc)


def _extension_repo(name):
    """A LOCAL extensions repository by module name (user_default by default): (index, repo). Refuses a remote one, a
    disabled one, the system one, and an unknown name with the list."""
    repos = list(bpy.context.preferences.extensions.repos)
    local = [r.module for r in repos if not r.use_remote_url and r.enabled and r.module != "system"]
    for i, r in enumerate(repos):
        if r.module == name:
            if r.module not in local:
                raise MifOpError("extensions repository %r is %s - install into a local user repository (%s). NOTHING "
                                 "was done." % (name, "remote" if r.use_remote_url else ("disabled" if not r.enabled
                                                                                        else "Blender's own"),
                                                ", ".join(local) or "none"))
            return i, r
    raise MifOpError("no extensions repository %r - local ones here: %s. NOTHING was done."
                     % (name, ", ".join(local) or "none"))


def op_enable_addon(params):
    """Enable, disable, install or remove an add-on - gated by Allow run_python.

    module (an installed add-on's module name; an unknown one is refused with the near names) or file (a .py or .zip,
    installed into the user's scripts/addons by Blender's own preferences.addon_install, then enabled); enable (default
    true) - false disables; persistent (default false) also records it in the preferences, which keep it only if they
    are saved; uninstall true disables a module that lives in the user's add-ons folder and deletes its file or folder.
    MifBlender itself is never disabled here. MEASURED 2026-10-05 on 4.2.23-5.2.2, headless: io_curve_svg is bundled and
    enabled, and addon_utils.disable / enable take its import operator away and back; preferences.addon_install puts a
    .py into the user's scripts/addons and addon_utils.enable registers its classes.

    EXTENSION PACKAGES (Blender 4.2+, added 2026-10-06): a .zip with a blender_manifest.toml (at its root or in its one
    top folder) is installed by Blender's own extensions.package_install_files into the local repository `repo`
    (default user_default; a remote, disabled or the system repository is refused) and enabled as bl_ext.<repo>.<id>;
    uninstall on a bl_ext.<repo>.<id> module goes through extensions.package_uninstall. A manifest whose type is not
    add-on is refused. MEASURED 2026-10-06 on 4.2.23, 4.4.0, 5.0.1 and 5.2.2 headless: both operators run in background
    mode, the package lands in extensions/user_default/<id>, addon_utils sees bl_ext.user_default.<id> with the
    manifest's version, and the uninstall removes the folder and the class.

    Reply: module, enabledBefore / enabledAfter, name, version (bl_info), installedTo, classesAdded / classesRemoved
    (bpy.types names), uninstalled, and extension {id, version, repo, type} for a package.
    """
    reject_unknown(params, ("module", "file", "enable", "persistent", "uninstall", "repo"), "enable_addon")
    _require_run_python_gate("enable_addon")
    import addon_utils   # noqa: PLC0415 - Blender's own add-on API, needed only here
    module = take(params, "module", default=None, kind=str)
    path = take(params, "file", default=None, kind=str)
    if (module is None) == (path is None):
        raise MifOpError("pass module (an add-on's name) OR file (a .py or .zip to install). NOTHING was done.")
    enable = take_bool(params, "enable", default=True)
    persistent = take_bool(params, "persistent", default=False)
    uninstall = take_bool(params, "uninstall", default=False)
    repo_name = take(params, "repo", default=None, kind=str)
    if uninstall and (path is not None or params.get("enable") is not None or repo_name is not None):
        raise MifOpError("uninstall takes a module alone. NOTHING was done.")
    user_dir = bpy.utils.user_resource("SCRIPTS", path="addons")
    installed_to = None
    manifest = None
    ext_repo = None
    if path is not None:
        src = os.path.abspath(bpy.path.abspath(path))
        ext = os.path.splitext(src)[1].lower()
        if ext not in (".py", ".zip"):
            raise MifOpError("file must be an add-on's .py or .zip, got %r. NOTHING was done." % src)
        if not os.path.isfile(src):
            raise MifOpError("no file at %r. NOTHING was done." % src)
        if ext == ".zip":
            manifest = _extension_manifest(src)
        if repo_name is not None and manifest is None:
            raise MifOpError("repo is the extensions repository a 4.2+ extension package goes into; %r is not one (no "
                             "blender_manifest.toml). NOTHING was done." % src)
        if manifest is not None:
            if bpy.app.version < (4, 2, 0) or not hasattr(bpy.ops, "extensions"):
                raise MifOpError("%r is an extension package (blender_manifest.toml), which needs Blender 4.2 or later; "
                                 "this is %s. NOTHING was done." % (src, bpy.app.version_string))
            ext_id = manifest.get("id")
            if not isinstance(ext_id, str) or not ext_id:
                raise MifOpError("the package's blender_manifest.toml has no id. NOTHING was done.")
            if manifest.get("type", "add-on") != "add-on":
                raise MifOpError("the package is a %r extension, not an add-on - enable_addon installs add-ons. NOTHING "
                                 "was done." % manifest.get("type"))
            ext_repo = _extension_repo(repo_name or "user_default")
            module = "bl_ext.%s.%s" % (ext_repo[1].module, ext_id)
        elif ext == ".py":
            module = os.path.splitext(os.path.basename(src))[0]
        else:
            import zipfile   # noqa: PLC0415
            with zipfile.ZipFile(src) as zf:
                tops = sorted({n.split("/")[0] for n in zf.namelist() if n.strip("/")})
            if len(tops) != 1:
                raise MifOpError("the .zip holds %s at its top level - an add-on zip holds one package folder. NOTHING "
                                 "was done." % (tops[:5],))
            module = os.path.splitext(tops[0])[0]
    if module == __package__ and not (enable and not uninstall):
        raise MifOpError("%s is the bridge answering this call - disabling or removing it here would cut the call off. "
                         "NOTHING was done." % __package__)
    if repo_name is not None and path is None:
        raise MifOpError("repo goes with file (an extension package to install). NOTHING was done.")
    if path is not None and manifest is not None:
        result = bpy.ops.extensions.package_install_files(filepath=src, repo=ext_repo[1].module,
                                                          enable_on_install=False, overwrite=True)
        if "FINISHED" not in result:
            raise MifOpError("extensions.package_install_files returned %s for %r." % (sorted(result), src))
        addon_utils.modules_refresh()
        installed_to = os.path.join(ext_repo[1].directory, manifest["id"])
    elif path is not None:
        result = bpy.ops.preferences.addon_install(filepath=src, overwrite=True)
        if "FINISHED" not in result:
            raise MifOpError("preferences.addon_install returned %s for %r." % (sorted(result), src))
        addon_utils.modules_refresh()
        installed_to = user_dir
    mods = {m.__name__: m for m in addon_utils.modules()}
    if module not in mods:
        import difflib   # noqa: PLC0415
        near = difflib.get_close_matches(module, list(mods), n=5, cutoff=0.6)
        raise MifOpError("no add-on module %r on this Blender%s. NOTHING was done."
                         % (module, (" - near: %s" % ", ".join(near)) if near else ""))
    info = addon_utils.module_bl_info(mods[module])
    before_enabled = bool(addon_utils.check(module)[1])
    types_before = set(dir(bpy.types))
    removed_path = None
    if uninstall and module.startswith("bl_ext."):
        parts = module.split(".")
        if len(parts) != 3:
            raise MifOpError("%r is not bl_ext.<repo>.<id>. NOTHING was done." % module)
        idx, repo = _extension_repo(parts[1])
        if before_enabled:
            addon_utils.disable(module, default_set=True)
        result = bpy.ops.extensions.package_uninstall(repo_directory=repo.directory, repo_index=idx, pkg_id=parts[2])
        if "FINISHED" not in result:
            raise MifOpError("extensions.package_uninstall returned %s for %r." % (sorted(result), module))
        removed_path = os.path.join(repo.directory, parts[2])
        addon_utils.modules_refresh()
    elif uninstall:
        mfile = getattr(mods[module], "__file__", "") or ""
        target = os.path.dirname(mfile) if os.path.basename(mfile) == "__init__.py" else mfile
        if not target or os.path.dirname(os.path.abspath(target)) != os.path.abspath(user_dir):
            raise MifOpError("%r is not in the user's add-ons folder (%s) - uninstall removes only what was installed "
                             "there. NOTHING was done." % (module, user_dir))
        addon_utils.disable(module, default_set=persistent)
        import shutil   # noqa: PLC0415
        if os.path.isdir(target):
            shutil.rmtree(target)
        else:
            os.remove(target)
        removed_path = target
        addon_utils.modules_refresh()
    elif enable:
        # default_set=True ALWAYS (2026-10-15, the settle pass): an add-on whose register() reads its own
        # preferences.addons entry fails without one - enable_addon('rigify') answered "reported no error and it is
        # disabled" on 5.0.1 and left the generator HALF-REGISTERED (the trap ops_rigify.py measured on 3.6, 4.4 and
        # 5.2). The entry is in memory; when the call is not persistent, preferences.is_dirty is put back as found so
        # an auto-save of preferences does not record it.
        prefs = bpy.context.preferences
        was_dirty = prefs.is_dirty
        addon_utils.enable(module, default_set=True, persistent=persistent)
        if not persistent and not was_dirty:
            prefs.is_dirty = False
    else:
        addon_utils.disable(module, default_set=persistent)
    types_after = set(dir(bpy.types))
    after_enabled = bool(addon_utils.check(module)[1]) if not uninstall else False
    if not uninstall and after_enabled != enable:
        raise MifOpError("addon_utils %s %r reported no error and it is %s - its register() may have failed; Blender "
                         "prints the traceback." % ("enable" if enable else "disable", module,
                                                    "enabled" if after_enabled else "disabled"))
    version = info.get("version")
    return {"module": module, "enabledBefore": before_enabled, "enabledAfter": after_enabled,
            "name": info.get("name") or None, "version": list(version) if version else None,
            "installedTo": installed_to, "uninstalled": bool(removed_path), "removedPath": removed_path,
            "persistent": persistent, "classesAdded": sorted(types_after - types_before)[:50],
            "classesRemoved": sorted(types_before - types_after)[:50],
            "extension": {"id": manifest.get("id"), "version": manifest.get("version"), "repo": ext_repo[1].module,
                          "type": manifest.get("type", "add-on")} if manifest is not None else None,
            "note": ("installed into %s - it stays there for this user's Blender until uninstall removes it."
                     % ("the %s extensions repository" % ext_repo[1].module if manifest is not None
                        else "the user's add-ons folder")) if installed_to else None}


# ---------------------------------------------------------------------------
# set_scene_units
# ---------------------------------------------------------------------------

_UNIT_KEYS = ("system", "scaleLength", "lengthUnit", "massUnit", "timeUnit", "temperatureUnit", "rotationUnit",
              "separate")
_UNIT_PROPS = {"system": "system", "scaleLength": "scale_length", "lengthUnit": "length_unit", "massUnit": "mass_unit",
               "timeUnit": "time_unit", "temperatureUnit": "temperature_unit", "rotationUnit": "system_rotation",
               "separate": "use_separate"}


def _units_read(units):
    return {k: (round(float(getattr(units, p)), 9) if k == "scaleLength" else getattr(units, p))
            for k, p in _UNIT_PROPS.items() if hasattr(units, p)}


def op_set_scene_units(params):
    """Set the active scene's unit system, unit scale and display units - read back from the scene.

    THE GAP (2026-10-02): a street scene and a character file were set to metric at scale 1.0 (and one to centimetres
    for display) in raw Python; scene_info reports these fields and warns when scaleLength is not 1, and nothing wrote
    them. system NONE, METRIC or IMPERIAL; scaleLength (above 0; 1.0 is the scale the Unreal hand-off assumes - 0.01 is
    the centimetre convention, and FBX export multiplies by it); lengthUnit, massUnit, timeUnit, temperatureUnit (each
    checked against this build's list for the system - lengthUnit CENTIMETERS needs METRIC, INCHES IMPERIAL); rotationUnit
    DEGREES or RADIANS; separate (show compound units). The system is written first, because Blender resets the unit
    choices when it changes. Every value is checked before any is written, and the reply carries before and after.
    """
    tail = "NOTHING was changed."
    reject_unknown(params, _UNIT_KEYS, "set_scene_units")
    units = bpy.context.scene.unit_settings
    asked = {k: params[k] for k in _UNIT_KEYS if params.get(k) is not None}
    if not asked:
        raise MifOpError("nothing to set - pass one of %s. %s" % (", ".join(_UNIT_KEYS), tail))
    want = {}
    if "scaleLength" in asked:
        s = take_float(params, "scaleLength")
        if not s > 0:
            raise MifOpError("scaleLength must be above 0, got %g. %s" % (s, tail))
        want["scaleLength"] = s
    if "separate" in asked:
        want["separate"] = take_bool(params, "separate")
    system = str(asked.get("system", units.system)).upper()
    enum_keys = [k for k in ("system", "lengthUnit", "massUnit", "timeUnit", "temperatureUnit", "rotationUnit")
                 if k in asked]
    for k in enum_keys:
        if not isinstance(asked[k], str):
            raise MifOpError("'%s' must be a name, got %r. %s" % (k, asked[k], tail))
        if k in ("temperatureUnit", "timeUnit") and not hasattr(units, _UNIT_PROPS[k]):
            raise MifOpError("%s is not on this Blender's unit settings. %s" % (k, tail))
    if "system" in asked:
        items = [i.identifier for i in units.bl_rna.properties["system"].enum_items]
        if system not in items:
            raise MifOpError("system must be one of %s, got %r. %s" % (items, asked["system"], tail))
        want["system"] = system
    if "rotationUnit" in asked:
        r = asked["rotationUnit"].upper()
        if r not in ("DEGREES", "RADIANS"):
            raise MifOpError("rotationUnit must be DEGREES or RADIANS, got %r. %s" % (asked["rotationUnit"], tail))
        want["rotationUnit"] = r
    before = _units_read(units)
    # The unit enums' items depend on the system and are generated (bl_rna lists only DEFAULT), so each is checked by
    # writing it: the system first (Blender resets the unit choices when it changes), then the rest, and a value Blender
    # refuses puts every setting back as it was before the refusal is raised.
    try:
        if "system" in want:
            units.system = want["system"]
        for k in ("lengthUnit", "massUnit", "timeUnit", "temperatureUnit"):
            if k in asked:
                u = asked[k].upper()
                try:
                    setattr(units, _UNIT_PROPS[k], u)
                except TypeError as exc:
                    raise MifOpError("%s %r is not one Blender takes with system %s: %s. %s"
                                     % (k, asked[k], units.system, str(exc).split(":", 1)[-1].strip(), tail))
        for k, v in want.items():
            if k != "system":
                setattr(units, _UNIT_PROPS[k], v)
    except MifOpError:
        units.system = before["system"]
        for k, v in before.items():
            if k != "system":
                setattr(units, _UNIT_PROPS[k], v)
        raise
    after = _units_read(units)
    out = {"scene": bpy.context.scene.name, "before": before, "after": after,
           "changed": sorted(k for k in after if before.get(k) != after.get(k))}
    if abs(after["scaleLength"] - 1.0) > 1e-9:
        out["note"] = ("scaleLength is %g, not 1.0: FBX export and every length shown in the UI are scaled by it, and the "
                       "Unreal hand-off assumes 1.0." % after["scaleLength"])
    return out


OPS = {
    "list_scenes": op_list_scenes,
    "set_scene_units": op_set_scene_units,
    "list_custom_properties": op_list_custom_properties,
    "set_custom_property": op_set_custom_property,
    "ping": op_ping,
    "call_log": op_call_log,
    "scene_info": op_scene_info,
    "list_objects": op_list_objects,
    "object_info": op_object_info,
    "clear_scene": op_clear_scene,
    "delete_object": op_delete_object,
    "run_python": op_run_python,
    "set_object_visibility": op_set_object_visibility,
    "create_scene": op_create_scene,
    "delete_scene": op_delete_scene,
    "set_current_scene": op_set_current_scene,
    "call_operator": op_call_operator,
    "list_recent_operators": op_list_recent_operators,
    "enable_addon": op_enable_addon,
}
