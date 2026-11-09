"""Saving, opening and describing the .blend itself - the thing that made every other op temporary.

WHY THIS EXISTS, and why a 2026-09-03 coverage review put it first ahead of every feature. Nothing
in this addon called wm.save_mainfile. Not once. So every light, camera, world, keyframe, physics
setup, node group and material it authored lived only inside a running Blender and DIED WITH THE
PROCESS. The only artefacts MifBlender could produce were a mesh FBX, a baked texture and exactly
one rendered frame.

For a UE round trip that does not bite, because the FBX is the deliverable and the .blend is
scratch. For everyone else - film, archviz, motion graphics, print, anyone using Blender as
Blender - the deliverable IS the .blend, and the tool could not make one. That is the difference
between a mesh-conditioning pipeline and a Blender automation tool, and it is one module wide.

SAVE-A-COPY IS THE DEFAULT, deliberately. wm.save_as_mainfile normally REPOINTS the session at the
new path, so a subsequent save silently writes somewhere the caller never named, and an agent that
saved a scratch copy has quietly adopted it as the working file. `copy=True` writes the file and
leaves bpy.data.filepath alone. A caller who wants the session repointed asks for it by name.

THE POSTCONDITION THAT MATTERS IS NOT "the file exists". It is WHAT THE SAVE DESTROYED. Blender
purges datablocks with no users and no fake user when it writes, so an action that has been unlinked
from its object, an image with nothing pointing at it, a node group nobody instanced - all of them
are gone the moment you save, permanently and without a word. That is data loss caused BY the
successful operation, and it is invisible unless somebody counts first. So this counts first, and
reports what went.
"""
import os

import bpy

from . import ops_undo
from .ops_common import (remove_objects, MifOpError, check_output_path, reject_unknown, take, take_bool, take_int)

# The ID collections a save can purge. Not every bpy.data collection - only the ones that carry
# authored work somebody would miss, which is the same judgement clear_scene's purge list makes.
# FONTS ADDED 2026-09-08, and it closes two gaps rather than one, because this tuple drives BOTH
# _counts() and _orphans(). A caller could not see how many typefaces a file held, and - the half
# that matters - an orphaned font was not reported among the datablocks the next save destroys.
# create_text gained a `font` parameter the day before, so a session can now load faces that nothing
# was counting and nothing warned about losing.
# SOUNDS AND SPEAKERS ADDED 2026-09-23 with add_sound, which creates both: a speaker's data outlives its deleted object
# and a sound outlives every strip and speaker that played it, and a save drops both - so the census counts them, the
# orphan report names them, and purge_orphans `only` can take one ("sounds:tone.wav").
_PURGEABLE = ("actions", "images", "materials", "meshes", "node_groups", "textures",
              "armatures", "cameras", "lights", "worlds", "curves", "collections", "objects",
              "fonts", "speakers", "sounds")


def _orphans():
    """Datablocks a save is about to delete: no users, no fake user.

    Reported BEFORE the write, because afterwards they do not exist to be counted. This is the
    difference between "the save worked" and "the save worked and took your unlinked walk cycle
    with it" - and Blender says nothing about the second.
    """
    out = {}
    total = 0
    for coll_name in _PURGEABLE:
        coll = getattr(bpy.data, coll_name, None)
        if coll is None:
            continue
        names = []
        for db in coll:
            try:
                if db.users == 0 and not db.use_fake_user:
                    names.append(db.name)
            except AttributeError:
                continue          # not every ID type carries both; skip rather than guess
        if names:
            out[coll_name] = sorted(names)
            total += len(names)
    return out, total


def _counts():
    """How much of each thing this file holds. The cheap census a caller needs to resume work."""
    out = {}
    for coll_name in _PURGEABLE:
        coll = getattr(bpy.data, coll_name, None)
        if coll is not None:
            out[coll_name] = len(coll)

    # SCENE-OWNED STRUCTURE, counted here and deliberately NOT added to _PURGEABLE.
    #
    # view_layers is not a bpy.data collection at all - every scene owns its own - so putting the
    # name in _PURGEABLE would have counted precisely nothing, because both this function and
    # _orphans() getattr it off bpy.data and skip a miss. It is also not purgeable, which is what
    # that tuple means, so it does not belong there on meaning either.
    #
    # WHY IT IS COUNTED AT ALL, 2026-09-08: audit_blender_demanded_kind_has_creator asks, for every
    # datablock kind an op DEMANDS, whether anything in the addon creates one - and it measures a
    # creator by watching a file_info count move. Four ops demand a view layer (compositor_info,
    # create_view_layer, delete_view_layer, set_view_layer) and the audit could not judge any of
    # them, reporting the kind as UNMEASURABLE and saying in its own output that this "is not a
    # pass". It was the only kind left unjudged. Counting it here makes the question answerable.
    #
    # scenes comes with it because a view-layer total means little without knowing how many scenes
    # hold them - two scenes with one layer each and one scene with two are not the same file.
    scenes = getattr(bpy.data, "scenes", None)
    if scenes is not None:
        out["scenes"] = len(scenes)
        out["view_layers"] = sum(len(sc.view_layers) for sc in scenes)
        # SEQUENCER STRIPS, every scene's, metas' contents included - they are scene-owned like view layers, and
        # add_sound makes them. strips_all from 4.4, sequences_all before (by capability).
        out["sequencer_strips"] = sum(
            len(ed.strips_all if hasattr(ed, "strips_all") else ed.sequences_all)
            for ed in (getattr(sc, "sequence_editor", None) for sc in scenes) if ed is not None)
    return out


def op_file_info(params):
    """What file this is, whether it has unsaved work, and what it holds.

    No parameters. Cheap enough to call before anything else, and it answers the two questions that
    decide whether a session can be resumed at all: is there a file on disk behind this, and is
    there work in memory that is not in it.
    """
    # An empty TUPLE, not set(). parity_check resolves the accepted-key set statically and is
    # fail-closed by design - it reads a set/list/tuple literal and refuses a set() CALL, because
    # a call could be anything. reject_unknown only does `k not in accepted` and sorted(accepted),
    # both of which a tuple satisfies. Taking no parameters is still a contract worth declaring.
    reject_unknown(params, (), "file_info")
    orphans, orphan_total = _orphans()
    path = bpy.data.filepath or ""
    return {
        "filepath": path,
        "isSaved": bool(path),
        # BLENDER'S FLAG OR THE BRIDGE'S COUNT - either says unsaved work exists. Reported: isDirty read false
        # after verified writes, because on 5.2 no data-API or Python-operator write sets bpy.data.is_dirty (measured -
        # see ops_undo.bridge_writes). Both halves are reported so a caller can see which one spoke.
        "isDirty": bool(bpy.data.is_dirty) or ops_undo.bridge_writes() > 0,
        "isDirtyNative": bool(bpy.data.is_dirty),
        "bridgeWritesSinceSave": ops_undo.bridge_writes(),
        "fileExists": bool(path) and os.path.isfile(path),
        "fileBytes": os.path.getsize(path) if path and os.path.isfile(path) else -1,
        "blenderVersion": ".".join(str(v) for v in bpy.app.version),
        "counts": _counts(),
        "orphanCount": orphan_total,
        "orphans": orphans,
        "orphanNote": ("Datablocks with no users and no fake user. A SAVE DELETES THESE - "
                       "permanently and silently. Give one a fake user to keep it."),
    }


_APPEND_KEYS = {"filepath", "path", "objects", "collections", "materials", "nodeGroups",
                "actions", "worlds", "fonts", "link", "activeCollection", "listOnly", "instanceEmpty"}

# WHAT CAN BE PULLED IN, and the bpy.data collection each name maps to. Deliberately NOT every
# collection Blender has: these six are the ones an asset file is built out of, and each one is
# something this addon can go on to USE. Adding brushes or workspaces would be a capability that
# imports something nothing else here can touch.
_APPEND_KINDS = (
    ("objects", "objects"),
    ("collections", "collections"),
    ("materials", "materials"),
    ("nodeGroups", "node_groups"),
    ("actions", "actions"),
    ("worlds", "worlds"),
    # A TYPEFACE IS THE ONE THING YOU CANNOT RE-DERIVE from a file you did not build: `font` on
    # create_text takes a PATH, so a caller opening somebody else's .blend can see text set in a
    # face they have no way to name. listOnly:true now answers that without importing anything.
    ("fonts", "fonts"),
)


def op_append_from_file(params):
    """Bring datablocks IN from another .blend - the half of the file ops that did not exist.

    THE ADDON COULD SAVE A .blend AND NEVER READ ONE BACK. save_file writes, open_file REPLACES the
    session, and there was nothing in between - so a scene built by this addon could be saved and
    then never reused. Every asset-library workflow is this op: build a bunker once, pull it into
    the next scene. Found by asking which bpy.data collections the addon never touches;
    `libraries` was one of two that named a real capability rather than a UI or paint feature.

    APPEND COPIES, LINK REFERENCES, and the difference is not cosmetic - it decides whether you can
    edit what you just brought in:

      link=false (default)  the datablocks are COPIED into this file. Editable, and they survive
                            the source file being moved or deleted.
      link=true             they stay in the source file and this one points at it. NOT editable -
                            a linked object refuses transform and mesh edits, which reads as the
                            addon being broken if you did not know you linked. The source file must
                            keep existing at that path forever.

    Reported either way as `linked`, with the consequence spelled out in the response rather than
    left in this docstring where a caller cannot see it.

    listOnly:true READS THE FILE AND IMPORTS NOTHING, which is how you find out what is in it
    without committing. The same call, the same path handling, no mutation - so a caller can look
    before choosing names.

    NAMES ARE MATCHED EXACTLY AND A MISS IS REFUSED, naming what the file does contain. Blender's
    own append silently imports nothing for a name that is not there, which is the failure mode this
    repo refuses everywhere: a call that reports success and does nothing.

    Objects are linked into the scene's active collection so they are VISIBLE. bpy.data.libraries
    .load puts them in bpy.data and nothing else - an appended object that is in no collection is
    in no scene, invisible in the viewport and absent from the render, which is the same inert shape
    _link_new was written for on the creation side.

    params:
      filepath (str, required)     the .blend to read
      objects / collections / materials / nodeGroups / actions / worlds / fonts
                                   fonts added 2026-09-08 - a typeface is the one thing you
                                   cannot re-derive from a file you did not build, because
                                   create_text's `font` takes a PATH
                                   lists of names to bring in. At least one is required unless
                                   listOnly is set.
      link (bool | "PACKED")       default false - copy rather than reference. "PACKED" (5.0+) links and
                                   stores the linked data in this file: still not editable, but it
                                   survives the source file moving
      activeCollection (str)       where objects land; defaults to the scene collection
      listOnly (bool)              report what the file holds and import NOTHING
      instanceEmpty (bool)         collections land as an EMPTY instancing each one (instance_type COLLECTION) -
                                   Blender's own File > Link default - instead of as a child collection. Added
                                   2026-10-06 with the library overrides: an instanced collection is what
                                   library_override MAKE replaces with the override
    """
    reject_unknown(params, _APPEND_KEYS, "append_from_file", {
        "append": "spelt link, inverted - link:false (the default) APPENDS a copy, link:true "
                  "references the source file.",
        "meshes": "a mesh datablock alone has no object to hold it, so it would be invisible. Name "
                  "the OBJECT instead and its mesh comes with it.",
        "images": "an image comes in attached to whatever material references it - name the "
                  "material.",
        "relative": "paths are resolved absolutely here. A relative library path is a property of "
                    "the SAVED file and this session may not have been saved.",
        "overwrite": "nothing is overwritten. A name that already exists gets Blender's .001 "
                     "suffix, and the response reports every imported name so you can see which.",
    })
    raw = take(params, "filepath", "path", default=None, kind=str)
    if not raw:
        raise MifOpError("'filepath' is required - the .blend to read from. NOTHING was imported.")
    path = os.path.abspath(os.path.expanduser(str(raw)))
    if not os.path.isfile(path):
        raise MifOpError("no file at '%s'. NOTHING was imported." % path)
    if not path.lower().endswith(".blend"):
        raise MifOpError("'%s' is not a .blend - this reads Blender files, not meshes. import_mesh "
                         "is the op for OBJ, FBX and glTF. NOTHING was imported." % path)
    if bpy.data.is_saved and os.path.normcase(path) == os.path.normcase(bpy.data.filepath):
        raise MifOpError("that is THIS file (%s). Appending a file into itself duplicates every "
                         "name you asked for and is never what was meant. NOTHING was imported."
                         % path)

    # LINK PACKED (2026-10-06, 5.0+): linked data stored inside this file - not editable until unpacked, and alive
    # when the source file moves. link "PACKED"; refused by name before 5.0, where libraries.load has no pack.
    packed = isinstance(params.get("link"), str) and params["link"].upper() == "PACKED"
    if packed and "is_linked_packed" not in bpy.types.ID.bl_rna.properties:
        raise MifOpError("link PACKED needs Blender 5.0 or later (this is %s, whose loader cannot pack linked data). "
                         "NOTHING was imported." % bpy.app.version_string)
    link = True if packed else take_bool(params, "link", default=False)
    list_only = take_bool(params, "listOnly", default=False)
    instance_empty = take_bool(params, "instanceEmpty", default=False)
    if instance_empty and (list_only or not params.get("collections")):
        raise MifOpError("instanceEmpty places collections as instancing empties - it needs collections, and does "
                         "nothing with listOnly. NOTHING was imported.")

    wanted = {}
    for key, attr in _APPEND_KINDS:
        names = params.get(key)
        if names is None:
            continue
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise MifOpError("'%s' must be a list of names. NOTHING was imported." % key)
        wanted[attr] = list(names)
    if not wanted and not list_only:
        raise MifOpError("name at least one thing to import - %s - or pass listOnly:true to see "
                         "what the file holds. NOTHING was imported."
                         % ", ".join(k for k, _ in _APPEND_KINDS))

    # WHAT THE FILE HOLDS, read first in every case. The refusal below needs it, and listOnly is
    # this same read with the import skipped - one path, so the two cannot disagree.
    available = {}
    try:
        with bpy.data.libraries.load(path, link=link) as (src, _dst):
            for _key, attr in _APPEND_KINDS:
                available[attr] = sorted(getattr(src, attr, []) or [])
    except Exception as exc:                                       # noqa: BLE001
        raise MifOpError("could not read '%s' as a Blender file (%s). NOTHING was imported."
                         % (path, exc))

    if list_only:
        return {"filepath": path, "listOnly": True, "linked": None,
                "available": available,
                "availableCounts": {k: len(v) for k, v in available.items()},
                "note": "nothing was imported. Name what you want in objects/collections/"
                        "materials/nodeGroups/actions/worlds and call again."}

    # EVERY MISSING NAME REFUSED BEFORE ANYTHING IS IMPORTED. Blender's own append imports nothing
    # for a name that is not in the file and reports no error at all, so a caller with one typo in
    # a list of ten gets nine objects and a success.
    missing = {}
    for attr, names in wanted.items():
        gone = [n for n in names if n not in available.get(attr, [])]
        if gone:
            missing[attr] = gone
    if missing:
        detail = "; ".join(
            "%s: %s (the file has %s)"
            % (a, ", ".join(v), ", ".join(available.get(a, [])[:8]) or "none")
            for a, v in sorted(missing.items()))
        raise MifOpError("'%s' does not contain %s. Blender imports NOTHING for a name that is not "
                         "there and reports no error, so this refuses instead. NOTHING was "
                         "imported." % (path, detail))

    # THE TYPE OF activeCollection IS CHECKED HERE, BEFORE THE IMPORT, even though the NAME cannot
    # be - the name has to wait because the import is what produces the datablocks to link, and the
    # missing-name refusal below says so in as many words. A wrong TYPE needs no such wait.
    #
    # WHY IT MOVED, found by the second seat 2026-09-15 (B1350). take(..., kind=str) raised
    # "'activeCollection' must be str, got bool" from BELOW the load, so the datablocks were already
    # in bpy.data - Camera.002 and friends - and unlike the missing-name refusal three lines down,
    # that message said nothing about it. A caller who read ok:false and moved on had silently grown
    # their file, and the suffixed name they would need to clean it up was the one thing the
    # response did not carry. Validating first makes the wrong-type case cost NOTHING rather than
    # reporting a cost, which is the better of the two fixes available.
    # take() is a pure reader - it does not consume - so calling it twice is free and the second
    # call below simply reads a value already proven to be a string.
    #
    # AND THE NAME, SINCE 2026-09-21. The reasoning above - "the name has to wait because the import
    # is what produces the datablocks to link" - holds for the datablocks, not for the TARGET: the
    # collection objects land in is looked up in bpy.data, and a misspelt one was refused only after
    # the load, with every datablock left in bpy.data and in no scene (the refusal said so). A target
    # that could only come from this import would be one of the collections it appends - and linking
    # that collection into itself is refused by Blender - so the target must exist before the call.
    coll_asked = take(params, "activeCollection", default=None, kind=str)
    if coll_asked is not None and bpy.data.collections.get(coll_asked) is None:
        have = sorted(c.name for c in bpy.data.collections)[:20]
        raise MifOpError("activeCollection: no collection named '%s' in this file - it names where the appended "
                         "objects land, and must exist before the call (create_collection makes one). This file "
                         "has: %s. NOTHING was imported." % (coll_asked, ", ".join(have) if have else "(none)"))

    before = {attr: {d.name for d in getattr(bpy.data, attr)} for _k, attr in _APPEND_KINDS}
    # SNAPSHOT THE ASKED-FOR NAMES BEFORE THE LOAD, because the loader MUTATES the lists it is
    # handed. Assigning `wanted["objects"]` straight to `dst.objects` gives Blender that list
    # object, and it comes back EMPTIED - so the `renamed` calculation below compared the imported
    # names against nothing and reported every single one as renamed. Measured: an append into an
    # empty scene returned imported ["MifSrcCube"] and renamed ["MifSrcCube"], which cannot both be
    # true. A copy per attribute is the fix, and it is why the assignment below is list(names).
    asked = sorted(n for names in wanted.values() for n in names)
    with bpy.data.libraries.load(path, link=link, **({"pack": True} if packed else {})) as (src, dst):
        for attr, names in wanted.items():
            setattr(dst, attr, list(names))

    imported = {}
    for _key, attr in _APPEND_KINDS:
        if attr not in wanted:
            continue
        now = {d.name for d in getattr(bpy.data, attr)}
        imported[attr] = sorted(now - before[attr])

    # LINKED INTO A COLLECTION, or the objects are in bpy.data and in no scene - invisible in the
    # viewport, absent from the render, and present in the file. The same inert shape _link_new
    # exists for on the creation side.
    # Type already proven above, before the import; this one only reads it.
    coll_name = take(params, "activeCollection", default=None, kind=str)
    if coll_name is not None:
        target = bpy.data.collections.get(coll_name)
        if target is None:
            raise MifOpError("no collection named '%s'. The datablocks WERE imported; they are in "
                             "bpy.data and not in a scene." % coll_name)
    else:
        target = bpy.context.scene.collection
    placed = []
    for name in imported.get("objects", []):
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.name not in target.objects:
            target.objects.link(obj)
            placed.append(obj.name)
    instancers = []
    for name in imported.get("collections", []):
        child = next((c for c in bpy.data.collections if c.name == name and (c.library is not None) == bool(link)),
                     None) or bpy.data.collections.get(name)
        if child is None:
            continue
        if instance_empty:
            emp = bpy.data.objects.new(child.name, None)
            emp.instance_type = "COLLECTION"
            emp.instance_collection = child
            target.objects.link(emp)
            instancers.append({"empty": emp.name, "collection": child.name})
            placed.append(emp.name)
        elif child.name not in {c.name for c in target.children}:
            target.children.link(child)
            placed.append(child.name)

    out = {
        "filepath": path,
        "linked": bool(link),
        "packed": [n for attr, names in imported.items() for n in names
                   if getattr(getattr(bpy.data, attr).get(n), "is_linked_packed", False)] if packed else None,
        "imported": imported,
        "importedCounts": {k: len(v) for k, v in imported.items()},
        "placedInCollection": target.name,
        "placed": placed,
        "renamed": sorted(n for names in imported.values() for n in names if n not in asked),
    }
    if instance_empty:
        out["instancers"] = instancers
    out["linkNote"] = (
        "LINKED AND PACKED: the linked datablocks are stored inside this file, so moving or deleting %s no longer "
        "breaks it. They stay NOT editable until unpacked or made local." % path if packed else
        "LINKED, not copied: these datablocks live in %s and this file only points at them. They "
        "are NOT editable - a transform or a mesh edit on a linked object is refused by Blender, "
        "which reads as this addon being broken if you did not know. Moving or deleting that file "
        "breaks this one." % path if link else
        "APPENDED as copies: editable, and they no longer depend on %s existing." % path)
    if out["renamed"]:
        out["renameNote"] = ("Blender appended a numeric suffix because those names were already "
                             "taken here. The names above are what you actually have.")
    return out


def op_save_file(params):
    """Write the .blend to disk. Reports what the save destroyed, not just that it worked.

    params:
      filepath (str, required)   where to write. .blend is appended if absent.
      overwrite (bool)           default False. An existing file is REFUSED without this.
      repointSession (bool)      default False. When false the session keeps its current filepath
                                 and this is a save-a-copy; when true, subsequent saves go here.
      compress (bool)            default False, Blender's own default.

    REFUSES BEFORE WRITING in every case, so a rejected call leaves the disk untouched.

    externalFiles (1.3) counts the images, sounds, fonts, clips, caches, libraries and volumes the saved file reads from
    disk, naming the missing ones and the ones at absolute paths; externalFilesWarning says so when there are any (the
    file opens elsewhere with them missing) and names manage_external_data.
    """
    reject_unknown(params, {"filepath", "path", "overwrite", "repointSession", "compress"},
                   "save_file")
    raw_path = take(params, "filepath", "path", default=None, kind=str)
    if not raw_path:
        raise MifOpError("'filepath' is required - where to write the .blend. NOTHING was saved.")
    path = os.path.abspath(os.path.expanduser(str(raw_path)))
    # SAME CHECK THE FOUR OTHER WRITERS USE. This one already refused with a sentence rather than a
    # raw exception, because save_as_mainfile fails immediately and there is no expensive work to
    # waste - but a caller gets a clearer answer from the shared message than from Blender's, and an
    # op that writes a file and skips the shared check is exactly what audit_output_paths is for.
    check_output_path(raw_path, path, "saved")
    if not path.lower().endswith(".blend"):
        path += ".blend"

    overwrite = take_bool(params, "overwrite", default=False)
    existed = os.path.isfile(path)
    if existed and not overwrite:
        raise MifOpError("'%s' already exists and overwrite was not asked for. NOTHING was saved."
                         % path)

    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as exc:
            raise MifOpError("cannot create the directory %s: %s. NOTHING was saved."
                             % (parent, exc))

    # COUNTED BEFORE THE WRITE, because afterwards they are gone and uncountable. See _orphans.
    orphans, orphan_total = _orphans()
    dirty_before = bool(bpy.data.is_dirty)
    was_filepath = bpy.data.filepath or ""
    repoint = take_bool(params, "repointSession", default=False)

    try:
        bpy.ops.wm.save_as_mainfile(filepath=path,
                                    copy=not repoint,
                                    compress=take_bool(params, "compress", default=False))
    except (RuntimeError, OSError) as exc:
        raise MifOpError("Blender refused to save to '%s': %s. Nothing was written unless the "
                         "error says otherwise - check the path and permissions." % (path, exc))

    # MEASURED OFF DISK. save_as_mainfile raises on most failures but a zero-byte or missing file
    # is still the question worth asking, and the answer is a stat rather than the absence of an
    # exception.
    on_disk = os.path.isfile(path)
    size = os.path.getsize(path) if on_disk else -1
    if not on_disk or size <= 0:
        raise MifOpError("save reported no error but '%s' is %s. Treat the file as not written."
                         % (path, "empty" if on_disk else "not on disk"))
    # COMPRESSION, READ OFF THE FILE'S OWN HEADER. An uncompressed .blend starts with the bytes
    # BLENDER; a compressed one (zstd since 3.0) does not. compress went straight through to Blender
    # and nothing reported it, so a caller had only fileBytes to go on - which moves with everything
    # else in the file - and audit_blender_inert_params listed the flag as changing nothing.
    with open(path, "rb") as fh:
        compressed_on_disk = not fh.read(7).startswith(b"BLENDER")

    # EXTERNAL FILES THE SAVED FILE DEPENDS ON (1.3 gap audit, 2026-10-19): a .blend handed to another machine opens with
    # pink textures when its images are missing, or point at absolute paths of this one. manage_external_data's report,
    # taken after the save; a warning names them, and that op packs or relativizes them.
    from .ops_datablock import _external_rows   # noqa: PLC0415 - the one external-file reader
    ext_rows = _external_rows(["IMAGE", "SOUND", "FONT", "MOVIECLIP", "CACHE_FILE", "LIBRARY", "VOLUME"])
    missing = [r["name"] for r in ext_rows if not r["exists"] and not r["packed"]]
    absolute = [r["name"] for r in ext_rows if not r["relative"] and not r["packed"]]
    external = {"files": len(ext_rows), "packed": sum(r["packed"] for r in ext_rows), "missing": missing[:50],
                "absolute": absolute[:50]}
    ext_warning = None
    parts = []
    if missing:
        parts.append("%d external file(s) are missing: %s" % (len(missing), ", ".join(missing[:10])))
    if absolute:
        parts.append("%d point at absolute paths: %s" % (len(absolute), ", ".join(absolute[:10])))
    if parts:
        ext_warning = ("%s - on another machine this file opens with them missing. manage_external_data PACK_ALL "
                       "packs them, MAKE_RELATIVE makes them relative to the .blend, FIND_MISSING looks for the missing "
                       "ones." % "; ".join(parts))
    return {
        "externalFiles": external,
        "externalFilesWarning": ext_warning,
        "filepath": path,
        "fileBytes": size,
        "compressedOnDisk": compressed_on_disk,
        "overwrote": existed,
        "repointedSession": repoint,
        "sessionFilepath": bpy.data.filepath or "",
        "sessionFilepathUnchanged": (bpy.data.filepath or "") == was_filepath,
        "dirtyBefore": dirty_before,
        "dirtyAfter": bool(bpy.data.is_dirty),
        # THE PART BLENDER NEVER TELLS YOU. These no longer exist in the saved file.
        "purgedOrphans": orphan_total,
        "purgedOrphansByType": orphans,
        "purgeNote": ("A save DELETES datablocks with no users and no fake user. The counts above "
                      "are what this save removed - an unlinked action, an unused image, a node "
                      "group nobody instanced. Give one a fake user first to keep it."),
        "copyNote": ("copy=True unless repointSession was asked for, so bpy.data.filepath is "
                     "unchanged and the next save still goes where it did before. This is a "
                     "save-a-copy, not a Save As."),
    }


def op_open_file(params):
    """Open a .blend, discarding whatever is in memory.

    params:
      filepath (str, required)
      discardUnsaved (bool)   required to be true when the current session is DIRTY. Without it an
                              open over unsaved work is refused, because there is no undo for it.

    DESTRUCTIVE AND UNRECOVERABLE. Everything in the current session is gone - this is the one op
    here that can lose work that was never on disk, which is why the dirty check is not optional.
    """
    reject_unknown(params, {"filepath", "path", "discardUnsaved"}, "open_file")
    path = take(params, "filepath", "path", default=None, kind=str)
    if not path:
        raise MifOpError("'filepath' is required. NOTHING was opened.")
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        raise MifOpError("no file at '%s'. NOTHING was opened." % path)

    # READ AND TYPE-CHECK BEFORE THE is_dirty TEST, NOT INSIDE IT. This was
    #
    #     if bpy.data.is_dirty and not take_bool(params, "discardUnsaved", default=False):
    #
    # and `and` short-circuits, so on a CLEAN session take_bool never ran and the value was never
    # looked at. `discardUnsaved: {"mif": "not-a-value"}` came back ok. Found by the version matrix
    # on 2026-09-08, which reported a dict and a list both accepted where a value belongs - the op
    # had the right guard and simply skipped it half the time.
    #
    # The consequence is worse than a missed refusal: the SAME call was accepted on a clean session
    # and refused on a dirty one, so a caller with a malformed flag learns nothing until the day it
    # matters, which is exactly the day this parameter exists to protect.
    discard = take_bool(params, "discardUnsaved", default=False)
    # EITHER DIRTY SOURCE REFUSES. Blender's flag alone let a load discard bridge-made work on 5.2.
    writes = ops_undo.bridge_writes()
    if (bpy.data.is_dirty or writes > 0) and not discard:
        raise MifOpError(
            "this session has UNSAVED changes (%s) and opening '%s' would discard them permanently - "
            "there is no undo across a file load. Save first with save_file, or pass "
            "discardUnsaved:true if the work is genuinely disposable. NOTHING was opened."
            % ("; ".join(x for x in ("Blender marks the file modified" if bpy.data.is_dirty else "",
                                     "%d bridge call(s) that can change the scene since it was loaded or saved"
                                     % writes if writes else "") if x), path))

    try:
        bpy.ops.wm.open_mainfile(filepath=path)
    except (RuntimeError, OSError) as exc:
        raise MifOpError("Blender could not open '%s': %s. The previous session may or may not "
                         "still be loaded - call file_info to find out." % (path, exc))

    # READ BACK. open_mainfile replaces bpy.data wholesale, so the check is that we are now looking
    # at the file we asked for rather than at whatever survived a partial load.
    now = bpy.data.filepath or ""
    if os.path.normcase(now) != os.path.normcase(path):
        raise MifOpError("asked to open '%s' but the session reports '%s' afterwards. Do not trust "
                         "this state - call file_info." % (path, now))
    # NO STATIC WARNING FIELD HERE. A first version returned a discardedNote saying the previous
    # session was gone, and audit_blender_consequence_fields flagged it: a constant string in a
    # response is something no suite can check and no caller can act on differently, and the only
    # honest way to test it would be to have a suite actually destroy its own session mid-run.
    # The warning belongs where it is read BEFORE the call - the tool help and this docstring - not
    # in the reply that arrives after the work is already irreversible.
    return {
        "filepath": now,
        "opened": True,
        "isDirty": bool(bpy.data.is_dirty),
        "counts": _counts(),
    }


_PURGE_KEYS = ("confirm", "recursive", "only", "clearFakeUser", "dryRun", "previewId", "previewTtlSeconds")

# WHAT TAKES A DATABLOCK'S LAST USER AWAY, per type - so a refusal over a datablock still in use can say
# how to unlink it first. Blender's lifecycle is unlink-then-purge (see op_purge_orphans).
_UNLINK_HINTS = {
    "materials": "set_material_slots on the objects holding it",
    "node_groups": "remove_node on the Group node holding it, or remove_modifier on the Geometry Nodes "
                   "modifier running it",
    "worlds": "set_world or create_world {useAsSceneWorld:true} to give the scene another world",
    "actions": "assign_action {clear:true} on the object it animates",
    "images": "remove_node on the Image Texture node using it",
    "textures": "remove_modifier on the modifier using it",
    "meshes": "delete_object on the objects using it",
    "curves": "delete_object on the objects using it",
    "objects": "delete_object",
    "collections": "delete_collection",
}


def _purge_named(entries, recursive, clear_fake=False):
    """purge_orphans {only: [...]}: remove exactly the NAMED orphans, all or nothing.

    A FAKE USER IS A REQUEST TO KEEP, so a named datablock carrying one is refused - unless the call also
    says clearFakeUser:true, which clears it on the NAMED datablocks only and then purges them. Without that
    an action made by create_action (fake user on by default, so it survives a save) could never be removed
    by anything. The flag is refused without `only`: a file-wide fake-user clear is not a thing to ask for
    by accident.

    WHY A SCOPE, and why still a purge. The docstring below records the decision that per-type deleters
    are the wrong shape - a datablock with users cannot go without breaking what uses it - and that
    stands: only a datablock with NO users and no fake user is taken, and one still in use is refused
    naming what uses it and how to unlink it. What was missing is the scope. The file-wide purge takes
    every orphan, including a material between being created and being assigned (B158), so tidying ONE
    thing meant risking all the others; and an orphan keeps its NAME reserved until it is freed (B6), so a
    fixture made and abandoned turns the next run's name into "X.001". audit_blender_creator_no_remover
    found materials, node groups and textures removable only as a side effect and worlds, fonts and
    actions not at all (B159).

    AN ENTRY is "type:name" in the shape file_info and orphansPurgedNames already print
    ("materials:Rust"), or a bare name when exactly one purgeable datablock carries it.
    """
    targets = _purge_named_targets(entries, recursive, clear_fake)
    orphans_before, _ = _orphans()
    rows = []
    for coll_name, db in targets:
        name = db.name
        if getattr(db, "use_fake_user", False):
            db.use_fake_user = False          # validated above: clearFakeUser was asked for
        getattr(bpy.data, coll_name).remove(db)
        rows.append((coll_name, name))
    left = [(c, n) for c, n in rows if getattr(bpy.data, c).get(n) is not None]
    if left:
        raise MifOpError("Blender reported no error but %s are still present. Do not trust this state."
                         % ", ".join("%s:%s" % r for r in left))
    after, after_total = _orphans()
    purged = {}
    for c, n in rows:
        purged.setdefault(c, []).append(n)
    untouched = sum(len(v) for v in orphans_before.values()) - len(rows)
    return {"purged": len(rows),
            "purgedByType": {k: len(v) for k, v in purged.items()},
            "purgedNames": {k: sorted(v) for k, v in purged.items()},
            "only": True,
            "recursive": False,
            "orphansRemaining": after_total,
            # READ OFF THE FILE AFTERWARDS: every name is gone from its collection, so it is FREE again -
            # a create under the same name now gets that name, not "<name>.001".
            "namesFreed": sorted("%s:%s" % r for r in rows),
            "note": ("only the named datablocks were removed; the file's %d other orphan(s) were left where "
                     "they were." % max(untouched, 0))}


def _purge_named_targets(entries, recursive, clear_fake):
    """`only`'s every check, writing nothing: [(collection name, datablock)] - or the whole call refused. The purge and
    its bulk preview plan both run this."""
    if recursive:
        raise MifOpError("recursive and only do not go together - a recursive pass takes datablocks the "
                         "first one orphaned, which are ones you did not name. Name them in `only` too. "
                         "NOTHING was purged.")
    if not isinstance(entries, (list, tuple)) or not entries:
        raise MifOpError("only is a non-empty list of 'type:name' entries (or bare names), e.g. "
                         "['materials:Rust', 'node_groups:Leaves'] - got %r. NOTHING was purged." % (entries,))
    targets, problems, seen = [], [], set()
    for raw in entries:
        if not isinstance(raw, str) or not raw:
            problems.append("%r is not a name" % (raw,))
            continue
        kind, _, rest = raw.partition(":")
        if rest and kind in _PURGEABLE:
            kinds, name = [kind], rest
        else:
            kinds, name = list(_PURGEABLE), raw
        found = []
        for coll_name in kinds:
            coll = getattr(bpy.data, coll_name, None)
            db = coll.get(name) if coll is not None else None
            if db is not None:
                found.append((coll_name, db))
        if not found:
            problems.append("no purgeable datablock named '%s'%s" % (
                name, (" among %s" % kinds[0]) if len(kinds) == 1 else ""))
            continue
        if len(found) > 1:
            problems.append("'%s' names a datablock in %s - say which, as 'type:%s'" % (
                name, " and ".join(c for c, _ in found), name))
            continue
        coll_name, db = found[0]
        fake = bool(getattr(db, "use_fake_user", False))
        if fake and not clear_fake:
            problems.append("%s:%s has a FAKE USER - that is somebody asking for it to be kept; pass "
                            "clearFakeUser:true to clear it on the named datablocks and purge them" % (coll_name, name))
            continue
        # A fake user counts as one of db.users, so the in-use test below discounts it when it is being cleared.
        if db.users - (1 if fake else 0) > 0:
            try:
                users = sorted("%s '%s'" % (type(u).__name__, u.name)
                               for u in bpy.data.user_map(subset={db}).get(db, ()))
            except Exception:                                       # noqa: BLE001 - the hint is extra, the refusal is not
                users = []
            problems.append("%s:%s still has %d user(s)%s - unlink it first (%s), then purge it" % (
                coll_name, name, db.users - (1 if fake else 0), (": " + ", ".join(users[:8])) if users else "",
                _UNLINK_HINTS.get(coll_name, "take it off whatever uses it")))
            continue
        if (coll_name, db.name) in seen:
            continue                              # named twice - once is enough
        seen.add((coll_name, db.name))
        targets.append((coll_name, db))
    if problems:
        raise MifOpError("cannot purge: %s. The whole call is refused, so NOTHING was purged."
                         % "; ".join(problems))
    return targets


def op_purge_orphans(params):
    """Delete datablocks with no users and no fake user - NOW, without saving.

    THE ACT HALF OF A REPORT THAT ALREADY EXISTED. file_info has reported orphans since it was
    written, with a purgeNote saying "a save DELETES datablocks with no users and no fake user" -
    and nothing could act on that without either saving the file or clearing the scene. That is the
    detect-but-cannot-fix shape the addon has closed twice before by its own account: uv_unwrap for
    UVs, apply_transform for pivots, normalize_weights for weights, each named in their docstrings
    as the write half of a read that could already see the problem.

    WHY THERE IS NO delete_material OR delete_image, which is the question that led here. Blender's
    datablock lifecycle is unlink-then-purge, not delete: a material with users cannot be removed
    without breaking what uses it, and one without users costs nothing until the file is written.
    So per-type deleters would be the wrong shape, and THIS is the right one - the same operation
    the Outliner's "Purge" performs.

    CONFIRM IS REQUIRED, matching destroy_collection. This destroys authored work that happens to be
    unlinked - an unlinked material somebody spent an hour on looks exactly like scratch from here,
    and file_info is the way to look before leaping. The response reports what went, by type, taken
    as a before-and-after difference rather than from the return of the purge call, so the numbers
    describe what actually happened to this file.

    RECURSIVE IS OFF BY DEFAULT, and deliberately. A recursive purge repeats until nothing more is
    unused, which reaches datablocks that only became orphaned because the first pass removed their
    last user - correct for a deep clean, and more than a caller asking to tidy up expects.

    `only` SCOPES IT (2026-09-20): a list of "type:name" entries takes exactly those orphans and leaves
    every other one - see _purge_named. Still unlink-then-purge: a named datablock with users is refused.

    dryRun:true IS THE BULK PREVIEW (docs/15 section 5, 2026-10-09): every data-block the purge would take, with its users
    and fake user before, and a previewId - no confirm needed to look. {previewId} alone commits it and stands in for
    confirm:true: it is planned again, refused on any difference, and takes EXACTLY the previewed data-blocks (as
    `only`), never one the preview did not list. A recursive purge cannot be previewed - its later passes take what the
    first one orphans. Nothing brings a purged data-block back, so a commit never rolls back: a mismatch is reported per
    data-block (partial).
    """
    reject_unknown(params, _PURGE_KEYS, "purge_orphans",
                   notes={
                       "all": "without `only` a purge takes every orphan; with `only` it takes exactly the "
                              "named ones. Pass confirm:true either way",
                       "force": "spelt confirm, and it is required",
                       "types": "a purge is not per-type; scope it by NAME with `only`: ['materials:Rust'] - "
                                "file_info's orphans field lists the candidates by type",
                       "names": "spelt only: a list of 'type:name' entries, or bare names",
                       "name": "spelt only, and it is a list: ['materials:Rust']",
                   })
    only = params.get("only")
    if take_bool(params, "dryRun", default=False):
        # Reached inside a batch: a call of its own previews through the handshake (server._dispatch, ops_preview.py).
        plan = _plan_purge_orphans(params)
        return {"dryRun": True, "changed": False, "wouldPurge": [c["target"] for c in plan["changes"]],
                "wouldPurgeCount": len(plan["changes"]), "note": "DRY RUN - nothing was purged."}
    if not take_bool(params, "confirm", default=False):
        if only is not None:
            raise MifOpError(
                "purge_orphans needs confirm:true - even scoped by `only`, it permanently removes the "
                "datablocks named, and a freed datablock has no undo through this bridge. NOTHING was purged.")
        raise MifOpError(
            "purge_orphans needs confirm:true - it DELETES every datablock with no users and no "
            "fake user, which includes authored work that merely happens to be unlinked. "
            "file_info reports exactly what would go, by type, and changes nothing.")

    recursive = take_bool(params, "recursive", default=False)
    if only is not None:
        return _purge_named(only, recursive, take_bool(params, "clearFakeUser", default=False))
    if take_bool(params, "clearFakeUser", default=False):
        raise MifOpError("clearFakeUser needs `only` - it clears the fake user on the datablocks NAMED there. A "
                         "file-wide fake-user clear is not something to ask for by accident. NOTHING was purged.")
    before, before_total = _orphans()
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=recursive)
    after, after_total = _orphans()

    # MEASURED, NOT ASSUMED. orphans_purge returns a count on some builds and None on others, and a
    # recursive pass removes things that were not orphans when it started. Diffing the report
    # file_info already computes describes what happened to THIS file on THIS build.
    removed = {}
    for kind, names in before.items():
        gone = sorted(set(names) - set(after.get(kind, [])))
        if gone:
            removed[kind] = gone
    return {"purged": before_total - after_total,
            "purgedByType": {k: len(v) for k, v in removed.items()},
            "purgedNames": removed,
            "recursive": recursive,
            "orphansRemaining": after_total}


# ---------------------------------------------------------------------------------------------------------------------
# THE BULK PREVIEW of purge_orphans (ops_preview.py; docs/15 section 5). One change per data-block, "type:name", deleted; the
# commit purges exactly those through `only`, so a data-block that became an orphan after the preview is not taken.
# ---------------------------------------------------------------------------------------------------------------------
_PURGE_NO_ROLLBACK = "a purged data-block is freed, and no inverse brings it back"


def _purge_state(db):
    return {"users": int(db.users), "fakeUser": bool(getattr(db, "use_fake_user", False))}


def _plan_purge_orphans(request):
    reject_unknown(request, _PURGE_KEYS, "purge_orphans")
    recursive = take_bool(request, "recursive", default=False)
    clear_fake = take_bool(request, "clearFakeUser", default=False)
    only = request.get("only")
    if only is not None:
        targets = _purge_named_targets(only, recursive, clear_fake)
    else:
        if recursive:
            raise MifOpError("a recursive purge cannot be previewed: its later passes take data-blocks the first pass "
                             "orphans, which no plan can list before they are orphaned. Preview without recursive, "
                             "commit, and preview again for what that left. NOTHING was purged.")
        if clear_fake:
            raise MifOpError("clearFakeUser needs `only` - it clears the fake user on the datablocks NAMED there. "
                             "NOTHING was purged.")
        by_type, _total = _orphans()
        targets = [(kind, getattr(bpy.data, kind)[name]) for kind, names in by_type.items() for name in names]
    changes = [{"kind": "datablock", "op": "delete", "target": "%s:%s" % (kind, db.name), "before": _purge_state(db),
                "after": None} for kind, db in targets]
    commit = {"confirm": True, "only": [c["target"] for c in changes]}
    if clear_fake:
        commit["clearFakeUser"] = True
    return {"changes": changes, "unchanged": [], "context": [], "commit": commit,
            "response": {"wouldPurge": commit["only"], "wouldPurgeCount": len(changes), "recursive": False}}


def _read_purge_orphans(change, _kept):
    kind, _sep, name = change["target"].partition(":")
    db = getattr(bpy.data, kind).get(name)
    return None if db is None else _purge_state(db)


# ---------------------------------------------------------------------------------------------------------------------
# ASSET LIBRARIES - a folder registered in the preferences so its assets show in the asset browser.
# ---------------------------------------------------------------------------------------------------------------------

_ASSET_LIB_KEYS = ("name", "action", "path", "importMethod", "relativePath", "enabled", "newName", "save")
_ASSET_LIB_ACTIONS = ("ADD", "EDIT", "REMOVE")


def _asset_libs():
    return bpy.context.preferences.filepaths.asset_libraries


def _import_methods():
    """The import methods this build takes now: LINK, APPEND and APPEND_REUSE before 5.0; from 5.0 LINK, APPEND and
    PACK - or APPEND_REUSE in place of PACK while the experimental "No Data-Block Packing" is on.

    MEASURED 2026-10-06 on 5.0.1 and 5.2.2: the RNA enum lists all four, but an assignment accepts only the three its
    item function offers, and APPEND_REUSE written with packing on raised a TypeError from the setter after the library
    had been added. So the list a caller sees, and the one every request is checked against, is the settable one.
    """
    prop = bpy.types.UserAssetLibrary.bl_rna.properties["import_method"]
    items = [e.identifier for e in prop.enum_items]
    if "PACK" in items and "APPEND_REUSE" in items:
        no_pack = bool(getattr(bpy.context.preferences.experimental, "no_data_block_packing", False))
        items.remove("PACK" if no_pack else "APPEND_REUSE")
    return items


def _userpref_path():
    return os.path.join(bpy.utils.user_resource("CONFIG"), "userpref.blend")


def _blend_count(path, cap=2000):
    """How many .blend files sit under a library folder - the files the asset browser reads. Capped: a library on a
    huge share is counted up to the cap and said so, not walked to the end."""
    n = 0
    if not os.path.isdir(path):
        return 0, False
    for _root, _dirs, files in os.walk(path):
        n += sum(1 for f in files if f.lower().endswith(".blend"))
        if n >= cap:
            return n, True
    return n, False


def _asset_lib_row(lib):
    path = bpy.path.abspath(lib.path) if lib.path else ""
    blends, capped = _blend_count(path)
    row = {"name": lib.name, "path": lib.path, "pathExists": bool(path) and os.path.isdir(path),
           "importMethod": lib.import_method, "relativePath": bool(lib.use_relative_path),
           "blendFiles": blends}
    if capped:
        row["blendFilesCapped"] = True
    # 5.2 adds an enabled switch and remote libraries; read where the build has them, never created here.
    if hasattr(lib, "enabled"):
        row["enabled"] = bool(lib.enabled)
    if hasattr(lib, "use_remote_url"):
        row["remote"] = bool(lib.use_remote_url)
        if lib.use_remote_url:
            row["remoteUrl"] = lib.remote_url
    return row


def _asset_lib_state():
    prefs = bpy.context.preferences
    libs = _asset_libs()
    out = {"libraries": [_asset_lib_row(lib) for lib in libs],
           "importMethods": _import_methods(),
           "userprefPath": _userpref_path(),
           "userprefExists": os.path.isfile(_userpref_path()),
           "prefsDirty": bool(getattr(prefs, "is_dirty", False)),
           "autoSaveOnQuit": bool(getattr(prefs, "use_preferences_save", False))}
    online = getattr(prefs, "asset_libraries", None)
    if online is not None and hasattr(online, "use_online_essentials"):
        out["useOnlineEssentials"] = bool(online.use_online_essentials)
    return out


def op_list_asset_libraries(params):
    """Every asset library registered in the preferences, read-only.

    THE GAP: no op read preferences.filepaths.asset_libraries, so a caller could not tell which folders the asset
    browser shows or how their assets come in (link, append, append-reuse, pack).

    Each row: name, path, pathExists, importMethod, relativePath, blendFiles (the .blend files under the folder, capped
    at 2000), and on 5.2 enabled and remote (remoteUrl) - remote libraries are listed, never created. Also the build's
    importMethods, the userpref file and whether it exists, prefsDirty, autoSaveOnQuit (Blender saves dirty preferences
    when it quits if Auto-Save Preferences is on) and on 5.2 useOnlineEssentials.
    """
    reject_unknown(params, (), "list_asset_libraries")
    return _asset_lib_state()


def _new_asset_lib(name, path):
    libs = _asset_libs()
    if hasattr(libs, "new"):
        return libs.new(name=name, directory=path)
    # 3.6: no AssetLibraryCollection.new - the preferences operator adds one at the end, named from the folder.
    count = len(libs)
    bpy.ops.preferences.asset_library_add(directory=path)
    if len(libs) != count + 1:
        raise MifOpError("preferences.asset_library_add added no library for '%s'. NOTHING was added." % path)
    lib = libs[len(libs) - 1]
    lib.name = name
    return lib


def _remove_asset_lib(lib):
    libs = _asset_libs()
    if hasattr(libs, "remove"):
        libs.remove(lib)
        return
    bpy.ops.preferences.asset_library_remove(index=list(libs).index(lib))


def op_set_asset_library(params):
    """Register a folder as an asset library, change one, or remove one - the asset browser's Preferences > File Paths.

    THE GAP: no op touched preferences.filepaths.asset_libraries. append_from_file needs the caller to know the file,
    and a pose library or a bundled group read from a library could not be pointed at a folder the caller has.

    params:
      name (str, required)      the library's name - unique: a name already used is refused on ADD (Blender would add
                                "Name.001" in silence)
      action (str)              ADD (default), EDIT or REMOVE
      path (str)                the folder (must exist); required for ADD
      importMethod (str)        LINK, APPEND or APPEND_REUSE on every build, PACK from 5.0 - the build's own enum, an
                                identifier it lacks is refused naming the list
      relativePath (bool)       store relative paths to the library's files when its assets come in
      enabled (bool)            5.2 and later only (refused by name before)
      newName (str)             EDIT: rename
      save (bool)               default false: also write the preferences to disk (wm.save_userpref). They change
                                user configuration, so they are saved only when asked, and the reply says whether.

    Every refusal comes before any write. The reply carries the library's row (as list_asset_libraries), every library,
    saved, the userpref file, and autoSaveOnQuit - with it on, Blender writes dirty preferences when it quits anyway.
    """
    reject_unknown(params, _ASSET_LIB_KEYS, "set_asset_library", {
        "directory": "spelt path - the folder to register.",
        "remote": "remote libraries are listed by list_asset_libraries and not created here.",
        "importmethod": "spelt importMethod.",
    })
    name = take(params, "name", kind=str)
    if name is None or not name.strip():
        raise MifOpError("'name' is required - the library's name. NOTHING was changed.")
    action = str(take(params, "action", default="ADD", kind=str)).upper()
    if action not in _ASSET_LIB_ACTIONS:
        raise MifOpError("action must be one of %s, got %r. NOTHING was changed."
                         % (", ".join(_ASSET_LIB_ACTIONS), params.get("action")))
    libs = _asset_libs()
    lib = libs.get(name) if hasattr(libs, "get") else next((x for x in libs if x.name == name), None)
    have = ", ".join(x.name for x in libs) or "(none)"
    if action == "ADD" and lib is not None:
        raise MifOpError("an asset library named '%s' already exists (at %s) - use action EDIT to change it, or pick "
                         "another name. NOTHING was added." % (name, lib.path))
    if action != "ADD" and lib is None:
        raise MifOpError("no asset library named '%s'. The preferences have: %s. NOTHING was changed." % (name, have))

    raw_path = take(params, "path", kind=str)
    path = None
    if raw_path is not None:
        if action == "REMOVE":
            raise MifOpError("path does not apply to REMOVE. NOTHING was removed.")
        path = os.path.abspath(os.path.expanduser(raw_path))
        if not os.path.isdir(path):
            raise MifOpError("path: '%s' is not a folder - an asset library is a folder of .blend files and must "
                             "exist. NOTHING was changed." % path)
    if action == "ADD" and path is None:
        raise MifOpError("path is required for ADD - the folder to register. NOTHING was added.")

    method = take(params, "importMethod", kind=str)
    methods = _import_methods()
    if method is not None:
        method = method.upper()
        if method not in methods:
            static = [e.identifier for e in bpy.types.UserAssetLibrary.bl_rna.properties["import_method"].enum_items]
            extra = ""
            if method == "PACK" and method not in static:
                extra = (" PACK (the linked data stored in the file) arrives in Blender 5.0; this is %s."
                         % bpy.app.version_string)
            elif method in static:
                extra = (" On 5.0+ PACK and APPEND_REUSE exclude each other: APPEND_REUSE is offered only while the "
                         "experimental preference No Data-Block Packing (experimental.no_data_block_packing) is on.")
            raise MifOpError("importMethod %r is not one of this build's: %s.%s NOTHING was changed."
                             % (params.get("importMethod"), ", ".join(methods), extra))
    relative = take(params, "relativePath")
    relative = None if relative is None else take_bool(params, "relativePath")
    enabled = take(params, "enabled")
    enabled = None if enabled is None else take_bool(params, "enabled")
    if enabled is not None and "enabled" not in bpy.types.UserAssetLibrary.bl_rna.properties:
        raise MifOpError("enabled needs Blender 5.2 or later (UserAssetLibrary.enabled); this is %s. NOTHING was "
                         "changed." % bpy.app.version_string)
    new_name = take(params, "newName", kind=str)
    if new_name is not None:
        if action != "EDIT":
            raise MifOpError("newName renames a library and applies to EDIT only. NOTHING was changed.")
        if not new_name.strip():
            raise MifOpError("newName is blank. NOTHING was changed.")
        if new_name != name and any(x.name == new_name for x in libs):
            raise MifOpError("an asset library named '%s' already exists. NOTHING was changed." % new_name)
    if action == "REMOVE" and any(v is not None for v in (method, relative, enabled)):
        raise MifOpError("importMethod, relativePath and enabled do not apply to REMOVE. NOTHING was removed.")
    save = take_bool(params, "save", default=False)
    if save:
        pref = _userpref_path()
        check_output_path(pref, pref, "saved")      # before the library is touched

    before = _asset_lib_row(lib) if lib is not None else None
    if action == "REMOVE":
        _remove_asset_lib(lib)
        row = None
    else:
        if action == "ADD":
            lib = _new_asset_lib(name, path)
        else:
            if path is not None:
                lib.path = path
            if new_name is not None:
                lib.name = new_name
        if method is not None:
            lib.import_method = method
        if relative is not None:
            lib.use_relative_path = relative
        if enabled is not None:
            lib.enabled = enabled
        row = _asset_lib_row(lib)

    saved = False
    if save:
        try:
            saved = "FINISHED" in bpy.ops.wm.save_userpref()
        except RuntimeError as exc:
            raise MifOpError("the library was %s, but Blender refused to save the preferences: %s"
                             % ({"ADD": "added", "EDIT": "changed", "REMOVE": "removed"}[action], exc))
    out = {"action": action, "library": row, "before": before, "saved": saved}
    out.update(_asset_lib_state())
    out["saveNote"] = ("written to %s." % out["userprefPath"] if saved else
                       "the preferences were NOT saved (save false): the change lasts this session"
                       + (" - unless Blender quits with Auto-Save Preferences on, which writes them." if
                          out["autoSaveOnQuit"] else "."))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# LINKED LIBRARIES AND LIBRARY OVERRIDES
# ---------------------------------------------------------------------------------------------------------------------

# The ID collections read for linked data and overrides - what a character or a set is built of.
_LIB_KINDS = (("COLLECTION", "collections"), ("OBJECT", "objects"), ("MESH", "meshes"), ("MATERIAL", "materials"),
              ("ARMATURE", "armatures"), ("ACTION", "actions"), ("CAMERA", "cameras"), ("LIGHT", "lights"),
              ("CURVE", "curves"), ("NODETREE", "node_groups"), ("IMAGE", "images"), ("WORLD", "worlds"))
_LIB_ATTR = dict(_LIB_KINDS)


_LIB_TYPES = (("COLLECTION", "Collection"), ("OBJECT", "Object"), ("MESH", "Mesh"), ("MATERIAL", "Material"),
              ("ARMATURE", "Armature"), ("ACTION", "Action"), ("CAMERA", "Camera"), ("LIGHT", "Light"),
              ("CURVE", "Curve"), ("NODETREE", "NodeTree"), ("IMAGE", "Image"), ("WORLD", "World"))


def _id_kind(idb):
    for kind, tname in _LIB_TYPES:
        if isinstance(idb, getattr(bpy.types, tname)):
            return kind
    return type(idb).__name__.upper()


def _id_state(idb):
    """LINKED (another file's, not editable), SYSTEM_OVERRIDE (an override Blender made to hold the hierarchy - not
    meant for edits), EDITABLE_OVERRIDE, or LOCAL."""
    if idb.library is not None:
        return "LINKED"
    ovr = idb.override_library
    if ovr is None:
        return "LOCAL"
    return "SYSTEM_OVERRIDE" if ovr.is_system_override else "EDITABLE_OVERRIDE"


def _override_row(idb):
    ovr = idb.override_library
    row = {"name": idb.name, "kind": _id_kind(idb), "state": _id_state(idb)}
    if ovr is not None:
        ref = ovr.reference
        row["reference"] = ref.name if ref is not None else None
        row["library"] = (ref.library.name if ref is not None and ref.library is not None else None)
        root = ovr.hierarchy_root
        row["hierarchyRoot"] = root.name if root is not None else None
        row["overridden"] = sorted({p.rna_path for p in ovr.properties})
    return row


def _library_row(lib):
    absp = bpy.path.abspath(lib.filepath)
    linked = {}
    for kind, attr in _LIB_KINDS:
        names = sorted(i.name for i in getattr(bpy.data, attr) if i.library == lib)
        if names:
            linked[kind] = names
    overrides = 0
    for _kind, attr in _LIB_KINDS:
        for i in getattr(bpy.data, attr):
            ovr = i.override_library
            if i.library is None and ovr is not None and ovr.reference is not None and ovr.reference.library == lib:
                overrides += 1
    row = {"name": lib.name, "filepath": lib.filepath, "absPath": os.path.normpath(absp),
           "isMissing": bool(getattr(lib, "is_missing", False)), "fileExists": os.path.isfile(absp),
           "parent": lib.parent.name if lib.parent is not None else None,
           "linked": linked, "linkedCount": sum(len(v) for v in linked.values()), "overrideCount": overrides}
    if hasattr(lib, "needs_liboverride_resync"):
        row["needsResync"] = bool(lib.needs_liboverride_resync)
    return row


def _all_overrides():
    return [i for _k, attr in _LIB_KINDS for i in getattr(bpy.data, attr)
            if i.library is None and i.override_library is not None]


def op_list_libraries(params):
    """Every linked library and every library override in this file - read-only.

    THE GAP: nothing read bpy.data.libraries, and nothing said which datablocks were linked, which were overrides and
    which of those Blender made only to hold the hierarchy (system overrides) - so a caller met "make a library
    override" in a refusal with no way to see the state it described.

    params:
      library (str)   only this library (its name, as listed) and the overrides of its data
      limit (int)     cap the override rows (default 500)

    Reply: libraries [{name, filepath, absPath, isMissing, fileExists, parent, linked {KIND: [names]}, linkedCount,
    overrideCount, needsResync}], overrides [{name, kind, state SYSTEM_OVERRIDE | EDITABLE_OVERRIDE, reference,
    library, hierarchyRoot, overridden (the RNA paths the override stores)}], overrideCount.
    """
    reject_unknown(params, ("library", "limit"), "list_libraries")
    want = take(params, "library", kind=str)
    limit = take_int(params, "limit", default=500)
    if limit is not None and limit < 1:
        raise MifOpError("limit must be 1 or more.")
    libs = list(bpy.data.libraries)
    if want is not None:
        libs = [lb for lb in libs if lb.name == want]
        if not libs:
            raise MifOpError("no library named '%s'. This file links: %s."
                             % (want, ", ".join(lb.name for lb in bpy.data.libraries) or "(none)"))
    rows = [_override_row(i) for i in _all_overrides()
            if want is None or (i.override_library.reference is not None
                                and i.override_library.reference.library is not None
                                and i.override_library.reference.library.name == want)]
    out = {"libraries": [_library_row(lb) for lb in libs], "overrideCount": len(rows), "overrides": rows[:limit]}
    if len(rows) > limit:
        out["overridesTruncated"] = True
    return out


def _get_library(name):
    lib = bpy.data.libraries.get(name) if isinstance(name, str) else None
    if lib is None:
        raise MifOpError("no library named %r. This file links: %s. NOTHING was changed."
                         % (name, ", ".join(lb.name for lb in bpy.data.libraries) or "(none)"))
    return lib


def op_set_library(params):
    """Relocate a linked library to another .blend, or reload it from disk - without reverting the whole file.

    THE GAP: a source .blend moved or renamed left its library missing, and a source edited elsewhere could only be
    picked up by reopening this file. wm.lib_relocate and wm.lib_reload are file-browser operators; this is the data
    route: Library.filepath and Library.reload().

    params:
      library (str, required)  the library's name (list_libraries)
      filepath (str)           relocate: the .blend it now lives at (must exist); reloaded at once
      reload (bool)            reload from its current path (true when filepath is given)

    MEASURED 2026-10-06 on 4.2.23 and 5.2.2: Library.reload() of a path that does not exist raises and leaves the
    library as it was; the reload of a source that gained an object in an overridden collection resyncs the override
    on its own (the new object arrives as an override). Reply: before / after (the library row), the overrides before
    and after, and added (override names the reload brought).
    """
    reject_unknown(params, ("library", "filepath", "path", "reload"), "set_library", {
        "relocate": "pass filepath - the .blend the library now lives at.",
    })
    lib = _get_library(take(params, "library", kind=str))
    raw = take(params, "filepath", "path", kind=str)
    reload = take_bool(params, "reload", default=raw is not None)
    if raw is None and not reload:
        raise MifOpError("nothing to do - pass filepath to relocate, or reload:true. NOTHING was changed.")
    if raw is not None and not reload:
        raise MifOpError("a relocation is always reloaded - reload:false with filepath cannot be honoured. NOTHING "
                         "was changed.")
    path = None
    if raw is not None:
        path = os.path.abspath(os.path.expanduser(raw))
        if not os.path.isfile(path) or not path.lower().endswith(".blend"):
            raise MifOpError("filepath: no .blend at '%s'. NOTHING was changed." % path)
        if bpy.data.is_saved and os.path.normcase(path) == os.path.normcase(bpy.data.filepath):
            raise MifOpError("that is THIS file - a library cannot point at the file that links it. NOTHING was "
                             "changed.")
    elif not os.path.isfile(bpy.path.abspath(lib.filepath)):
        raise MifOpError("'%s' is missing at %s - a reload would fail. Pass filepath to relocate it. NOTHING was "
                         "changed." % (lib.name, bpy.path.abspath(lib.filepath)))
    name = lib.name
    before = _library_row(lib)
    ov_before = {(r["kind"], r["name"]) for r in (_override_row(i) for i in _all_overrides())}
    old = lib.filepath
    if path is not None:
        lib.filepath = path
    try:
        lib.reload()
    except RuntimeError as exc:
        lib = bpy.data.libraries.get(name)
        if lib is not None and path is not None:
            lib.filepath = old
        raise MifOpError("Blender refused to reload '%s' (%s); its path was put back. NOTHING was changed."
                         % (name, exc))
    lib = bpy.data.libraries.get(name) or next((lb for lb in bpy.data.libraries
                                                if os.path.normcase(bpy.path.abspath(lb.filepath))
                                                == os.path.normcase(path or bpy.path.abspath(old))), None)
    ov_after = [_override_row(i) for i in _all_overrides()]
    added = sorted("%s %s" % k for k in {(r["kind"], r["name"]) for r in ov_after} - ov_before)
    return {"library": lib.name if lib is not None else name, "relocated": path is not None, "reloaded": True,
            "before": before, "after": _library_row(lib) if lib is not None else None,
            "overrideCountBefore": len(ov_before), "overrideCountAfter": len(ov_after), "added": added}


_OVR_ACTIONS = ("MAKE", "EDITABLE", "SYSTEM", "RESYNC", "RESYNC_ENFORCE", "RESET", "CLEAR", "MAKE_LOCAL")
_OVR_SCOPES = ("SELECTED", "CONTENT", "SELECTED_AND_CONTENT")
_OVR_KEYS = ("target", "kind", "action", "scope", "editable", "fullyEditable", "data", "hierarchy")


def _find_ids(name, kind, want):
    """IDs named `name` (COLLECTION or OBJECT, or either when kind is None) that pass `want`."""
    kinds = [kind] if kind else ["COLLECTION", "OBJECT"]
    return [i for k in kinds for i in getattr(bpy.data, _LIB_ATTR[k]) if i.name == name and want(i)]


def _content(idb):
    """What a collection or object holds: a collection's objects and child collections (all levels), an object's
    children (all levels)."""
    if isinstance(idb, bpy.types.Collection):
        return list(idb.all_objects) + list(idb.children_recursive)
    return list(idb.children_recursive)


def _hierarchy(root):
    return [root] + [i for i in _all_overrides() if i != root and i.override_library.hierarchy_root == root]


def _pick(names, members, what):
    """Members of an override hierarchy by name - every name must be one."""
    by = {}
    for m in members:
        by.setdefault(m.name, m)
    miss = [n for n in names if n not in by]
    if miss:
        raise MifOpError("%s: %s not in the override hierarchy, which holds %s. NOTHING was changed."
                         % (what, ", ".join(miss), ", ".join(sorted(by))))
    return [by[n] for n in names]


def _scope_set(sel, scope, members):
    member_set = set(members)
    out = []
    if scope in ("SELECTED", "SELECTED_AND_CONTENT"):
        out += sel
    if scope in ("CONTENT", "SELECTED_AND_CONTENT"):
        for s in sel:
            out += [c for c in _content(s) if c in member_set]
    seen, uniq = set(), []
    for i in out:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


def _names_list(params, key):
    val = params.get(key)
    if val is None:
        return None
    if not isinstance(val, list) or not all(isinstance(n, str) and n for n in val):
        raise MifOpError("%s must be a list of names. NOTHING was changed." % key)
    return list(val)


def _scene_instancers(linked_coll):
    sc = bpy.context.scene
    return [o for o in sc.objects if o.library is None and o.instance_type == "COLLECTION"
            and o.instance_collection == linked_coll]


def _rows(ids):
    return [_override_row(i) if i.override_library is not None else
            {"name": i.name, "kind": _id_kind(i), "state": _id_state(i)} for i in ids]


def op_library_override(params):
    """Make, edit, resync, reset, clear or localize a library override - the step require_editable names.

    THE GAP: a write to linked data is refused with "make a library override for it", and no op could. The outliner's
    liboverride operators need an Outliner area, so this picks the IDs itself and calls the ID methods.

    params:
      target (str, required)  MAKE: a LINKED collection or object; every other action: an override (its hierarchy's
                              root is used for RESYNC, RESYNC_ENFORCE and CLEAR's hierarchy)
      kind (str)              COLLECTION or OBJECT when a collection and an object share the name
      action (str)            MAKE (default) - override_hierarchy_create; EDITABLE / SYSTEM - mark members editable or
                              back to system overrides; RESYNC / RESYNC_ENFORCE (4.2+, whole hierarchy); RESET -
                              drop the stored edits; CLEAR - the overrides become linked data again; MAKE_LOCAL (4.2+)
      scope (str)             SELECTED (default), CONTENT or SELECTED_AND_CONTENT - the outliner's selection sets:
                              the editable names, what they hold (a collection's objects and collections, an object's
                              children), or both
      editable (list)         the names scope applies to (default the target) - MAKE and EDITABLE / SYSTEM
      fullyEditable (bool)    MAKE: every override editable (do_fully_editable)
      data (list)             MAKE / EDITABLE: objects whose data block (a mesh, so its shape keys can be edited) is
                              overridden too
      hierarchy (bool)        RESET, CLEAR, MAKE_LOCAL: the whole hierarchy (default true) or the target only

    MAKE also does what the outliner does around the call: a scene empty instancing the linked collection is removed
    and the linked collection is unlinked from the scene's collections, so the override is not drawn on top of the
    linked copy. Every refusal comes before any write. Reply: the hierarchy's rows (name, kind, state, reference,
    overridden), what changed, and removedInstancers / unlinkedFrom for MAKE.
    """
    reject_unknown(params, _OVR_KEYS, "library_override", {
        "object": "spelt target - a linked collection or object, or an override.",
        "selectionSet": "spelt scope: SELECTED, CONTENT or SELECTED_AND_CONTENT.",
    })
    name = take(params, "target", kind=str)
    if not name:
        raise MifOpError("'target' is required. NOTHING was changed.")
    kind = take(params, "kind", kind=str)
    if kind is not None:
        kind = kind.upper()
        if kind not in ("COLLECTION", "OBJECT"):
            raise MifOpError("kind must be COLLECTION or OBJECT, got %r. NOTHING was changed." % params.get("kind"))
    action = str(take(params, "action", default="MAKE", kind=str)).upper()
    if action not in _OVR_ACTIONS:
        raise MifOpError("action must be one of %s, got %r. NOTHING was changed."
                         % (", ".join(_OVR_ACTIONS), params.get("action")))
    scope = take(params, "scope", kind=str)
    if scope is not None:
        scope = scope.upper()
        if scope not in _OVR_SCOPES:
            raise MifOpError("scope must be one of %s, got %r. NOTHING was changed."
                             % (", ".join(_OVR_SCOPES), params.get("scope")))
    editable = _names_list(params, "editable")
    data_names = _names_list(params, "data")
    fully = take_bool(params, "fullyEditable", default=False)
    hier = take(params, "hierarchy")
    hier = True if hier is None else take_bool(params, "hierarchy")
    if action not in ("MAKE", "EDITABLE", "SYSTEM") and (scope or editable is not None):
        raise MifOpError("scope and editable apply to MAKE, EDITABLE and SYSTEM, not %s. NOTHING was changed." % action)
    if action not in ("MAKE", "EDITABLE") and data_names is not None:
        raise MifOpError("data applies to MAKE and EDITABLE, not %s. NOTHING was changed." % action)
    if action != "MAKE" and fully:
        raise MifOpError("fullyEditable applies to MAKE. NOTHING was changed.")
    if fully and (scope or editable is not None):
        raise MifOpError("fullyEditable makes every override editable - pass it, or scope/editable, not both. "
                         "NOTHING was changed.")
    if action in ("MAKE", "EDITABLE", "SYSTEM", "RESYNC", "RESYNC_ENFORCE") and params.get("hierarchy") is not None:
        raise MifOpError("hierarchy applies to RESET, CLEAR and MAKE_LOCAL. NOTHING was changed.")
    scope = scope or "SELECTED"

    resync_fn = bpy.types.IDOverrideLibrary.bl_rna.functions.get("resync")
    if action in ("RESYNC", "RESYNC_ENFORCE") and (
            resync_fn is None or "do_whole_hierarchy" not in [p.identifier for p in resync_fn.parameters]):
        raise MifOpError("%s needs Blender 4.2 or later (IDOverrideLibrary.resync with do_whole_hierarchy); this is "
                         "%s, where resync is only the outliner's troubleshoot operator. NOTHING was changed."
                         % (action, bpy.app.version_string))
    if action == "MAKE_LOCAL" and "clear_liboverride" not in [
            p.identifier for p in bpy.types.ID.bl_rna.functions["make_local"].parameters]:
        raise MifOpError("MAKE_LOCAL needs Blender 4.2 or later (ID.make_local clear_liboverride); this is %s. "
                         "NOTHING was changed." % bpy.app.version_string)

    if action == "MAKE":
        hits = _find_ids(name, kind, lambda i: i.library is not None)
        if not hits:
            ovs = _find_ids(name, kind, lambda i: i.override_library is not None)
            raise MifOpError("'%s' is %s - MAKE takes a LINKED collection or object (list_libraries shows them). "
                             "NOTHING was changed." % (name, "already an override" if ovs else "not linked data"))
        if len(hits) > 1:
            raise MifOpError("'%s' names %d linked datablocks (%s) - pass kind. NOTHING was changed."
                             % (name, len(hits), ", ".join("%s from %s" % (_id_kind(h), h.library.name) for h in hits)))
        ref = hits[0]
        members = [ref] + _content(ref)
        sel = _pick(editable, members, "editable") if editable is not None else [ref]
        data_objs = _pick(data_names, members, "data") if data_names is not None else []
        for o in data_objs:
            if not isinstance(o, bpy.types.Object) or o.data is None:
                raise MifOpError("data: '%s' is not an object with a data block. NOTHING was changed." % o.name)
        return _make_override(ref, sel, scope, fully, data_objs)

    hits = _find_ids(name, kind, lambda i: i.library is None and i.override_library is not None)
    if not hits:
        linked = _find_ids(name, kind, lambda i: i.library is not None)
        raise MifOpError("'%s' is not a library override%s. NOTHING was changed."
                         % (name, " - it is linked; action MAKE makes one" if linked else ""))
    if len(hits) > 1:
        raise MifOpError("'%s' names an override collection and an override object - pass kind. NOTHING was changed."
                         % name)
    target = hits[0]
    root = target.override_library.hierarchy_root or target
    members = _hierarchy(root)
    if action in ("EDITABLE", "SYSTEM"):
        sel = _pick(editable, members, "editable") if editable is not None else [target]
        data_objs = _pick(data_names, members, "data") if data_names is not None else []
        for o in data_objs:
            if not isinstance(o, bpy.types.Object) or o.data is None:
                raise MifOpError("data: '%s' is not an object with a data block. NOTHING was changed." % o.name)
            if o.data.library is None and o.data.override_library is None:
                raise MifOpError("data: '%s' already has local data. NOTHING was changed." % o.name)
        changed = []
        for i in _scope_set(sel, scope, members):
            want_sys = action == "SYSTEM"
            if i.override_library.is_system_override != want_sys:
                i.override_library.is_system_override = want_sys
                changed.append(i.name)
        made = _override_data(data_objs)
        return {"action": action, "scope": scope, "changed": changed, "dataOverrides": made,
                "hierarchy": _rows(_hierarchy(root))}
    return _override_maintain(action, target, root, hier)


def _override_data(objs):
    made = []
    for o in objs:
        d = o.data
        if d.library is None:
            continue
        new = d.override_create(remap_local_usages=True)
        if new is None or o.data != new:
            raise MifOpError("Blender made no override of '%s' (the data of '%s'); the object overrides above were "
                             "made." % (d.name, o.name))
        made.append(new.name)
    return made


def _make_override(ref, sel, scope, fully, data_objs):
    sc = bpy.context.scene
    instancers = _scene_instancers(ref) if isinstance(ref, bpy.types.Collection) else []
    parents = [c for c in [sc.collection] + list(sc.collection.children_recursive)
               if c.library is None and ((isinstance(ref, bpy.types.Collection) and ref in c.children.values())
                                         or (isinstance(ref, bpy.types.Object) and ref in c.objects.values()))]
    sel_names = [(type(s), s.name) for s in sel]
    member_keys = {(type(m), m.name) for m in [ref] + _content(ref)}
    ref_name, ref_lib = ref.name, ref.library
    kw = {"do_fully_editable": fully}
    if instancers:
        kw["reference"] = instancers[0]
    root = ref.override_hierarchy_create(sc, bpy.context.view_layer, **kw)
    if root is None:
        raise MifOpError("Blender made no override of '%s'. NOTHING was changed." % ref_name)
    ref = next(i for i in getattr(bpy.data, _LIB_ATTR[_id_kind(root)])
               if i.name == ref_name and i.library == ref_lib)
    removed = []
    for e in instancers:
        removed.append(e.name)
        remove_objects([e])
    unlinked = []
    for c in parents:
        if isinstance(ref, bpy.types.Collection) and ref in c.children.values():
            c.children.unlink(ref)
            unlinked.append(c.name)
        elif isinstance(ref, bpy.types.Object) and ref in c.objects.values():
            c.objects.unlink(ref)
            unlinked.append(c.name)
    if isinstance(root, bpy.types.Collection) and root not in sc.collection.children_recursive:
        sc.collection.children.link(root)
    elif isinstance(root, bpy.types.Object) and root not in list(sc.objects):
        sc.collection.objects.link(root)
    members = _hierarchy(root)
    by_key = {(type(m), m.name): m for m in members}
    editable = []
    if not fully:
        sel_ids = [by_key[k] for k in sel_names if k in by_key]
        for i in members:
            if i.override_library.is_system_override is False and i not in sel_ids:
                i.override_library.is_system_override = True
        for i in _scope_set(sel_ids, scope, members):
            i.override_library.is_system_override = False
            editable.append(i.name)
    data_made = _override_data([by_key[(type(o), o.name)] for o in data_objs if (type(o), o.name) in by_key])
    return {"action": "MAKE", "root": root.name, "scope": None if fully else scope, "fullyEditable": fully,
            "editable": editable if not fully else [m.name for m in members], "dataOverrides": data_made,
            "removedInstancers": removed, "unlinkedFrom": unlinked,
            "hierarchy": _rows(_hierarchy(root)), "memberCount": len(member_keys)}


def _override_maintain(action, target, root, hier):
    kind, tname = _id_kind(target), target.name
    rname, rkind = root.name, _id_kind(root)
    before = _rows(_hierarchy(root))
    if action in ("RESYNC", "RESYNC_ENFORCE"):
        ok = root.override_library.resync(scene=bpy.context.scene, view_layer=bpy.context.view_layer,
                                          do_hierarchy_enforce=action == "RESYNC_ENFORCE", do_whole_hierarchy=True)
        if not ok:
            raise MifOpError("Blender's resync of '%s' did not succeed - is its library missing? (list_libraries)"
                             % rname)
    elif action == "RESET":
        target.override_library.reset(do_hierarchy=hier)
    elif action == "CLEAR":
        (root if hier else target).override_library.destroy(do_hierarchy=hier)
    else:  # MAKE_LOCAL
        for i in (_hierarchy(root) if hier else [target]):
            i.make_local(clear_liboverride=True)
    after_root = next((i for i in getattr(bpy.data, _LIB_ATTR[rkind]) if i.name == rname and i.library is None), None)
    after = _rows(_hierarchy(after_root)) if after_root is not None and after_root.override_library else []
    b = {(r["kind"], r["name"]): r for r in before}
    a = {(r["kind"], r["name"]): r for r in after}
    out = {"action": action, "target": tname, "kind": kind, "root": rname, "hierarchyApplied": hier,
           "before": before, "hierarchy": after,
           "added": sorted("%s %s" % k for k in set(a) - set(b)),
           "removed": sorted("%s %s" % k for k in set(b) - set(a)),
           "reset": sorted("%s %s" % k for k in set(a) & set(b) if b[k].get("overridden") and not a[k].get("overridden"))}
    if action in ("CLEAR", "MAKE_LOCAL"):
        out["targetStates"] = [r["state"] for r in _rows([i for i in getattr(bpy.data, _LIB_ATTR[kind])
                                                          if i.name == tname])]
    return out


PREVIEW = {"purge_orphans": {"plan": _plan_purge_orphans, "read": _read_purge_orphans, "capture": None,
                             "restore": None, "noRollback": _PURGE_NO_ROLLBACK}}

OPS = {
    "purge_orphans": op_purge_orphans,
    "file_info": op_file_info,
    "save_file": op_save_file,
    "append_from_file": op_append_from_file,
    "open_file": op_open_file,
    "list_asset_libraries": op_list_asset_libraries,
    "set_asset_library": op_set_asset_library,
    "list_libraries": op_list_libraries,
    "set_library": op_set_library,
    "library_override": op_library_override,
}
