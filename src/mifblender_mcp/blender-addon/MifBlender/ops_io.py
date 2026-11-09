"""Scene interchange both ways - glTF, OBJ, USD, Alembic, STL, PLY. What a Blender user ships.

WHY THIS IS NOT AN EXTENSION OF export_mesh. Measured 2026-09-03: import_mesh accepts .fbx, .gltf
and .glb, and export_mesh writes .fbx AND NOTHING ELSE. So glTF could come IN and not go OUT, and
USD - the interchange format of every film and Omniverse pipeline - was absent in both directions,
as were OBJ, Alembic, STL and PLY. That is the round-trip bias the standing rule names: FBX is what
the Unreal path needs, and it had become the whole of what the addon could write.

export_mesh is not the place to fix it. It is FBX-specific to its bones - FBX_EXPORT_ARGS, the
axis and bone-axis overrides, and an object_types filter whose semantics are the FBX exporter's -
and it says so itself, refusing CAMERA and LIGHT with "exporting a whole scene including lights and
cameras is a different job than export_mesh". This is that job. export_mesh keeps the UE round trip
and its skeletal-export handling; this writes interchange.

=============================================================================
THE OPERATOR NAMES MOVED, AND NOT ALL AT ONCE
=============================================================================
Blender rewrote its OBJ, STL and PLY exporters in C++ and moved them from export_scene.* /
export_mesh.* to wm.*_export, at DIFFERENT versions - measured 2026-09-28: wm.obj_export and
wm.ply_export are already on 3.6.23, wm.stl_export first on 4.2.17 (this said OBJ and PLY at 4.0). This
addon supports 3.6 through 5.2, so on any given build some of these live at the new name and some at the
old. Each format therefore carries a LIST of candidates tried in order, the same shape as the
fcurve accessor in ops_anim and the group interface in ops_nodes - and _op_exists asks the operator
REGISTRY, because the attribute lookup it used answered yes for all of them.

AND THE SELECTION KEYWORD IS DIFFERENT FOR EVERY ONE OF THEM: use_selection, export_selected_objects,
selected_objects_only, selected. Guessing one and passing it to the wrong operator raises, or worse
is silently ignored on an operator that takes **kwargs loosely - so it is spelled per format rather
than assumed.

=============================================================================
ok:true IS NOT A FILE
=============================================================================
An exporter returns {'FINISHED'} whether or not anything appeared, and a leftover file from an
earlier run passes any existence check. So the mtime is taken BEFORE the call and the file must be
newer than that as well as non-empty - the same rule render_animation applies per frame.
"""
import os
import time

import bpy
from mathutils import Vector

from .ops_common import (MifOpError, check_output_path, check_writable_dir, file_before, finite_float, get_object,
                         reject_unknown, replaced_fields, take, take_bool, take_int)

_EXPORT_KEYS = {"file", "filepath", "path", "objects", "object", "name", "selectedOnly",
                "applyModifiers", "frameStart", "frameEnd", "overwrite", "replaceExisting",
                "animation", "includeLights", "includeCameras", "exporterOptions", "bvh"}

# THIS OP'S OWN SWITCH for an exporter option, where it has one - what notCarried names as `option` (decided
# 2026-09-28: the glTF exporter's defaults stay, lights and cameras off, because in a hand-off file they clutter an
# Unreal import; a caller who wants them asks). The exporter's own name stays in the row as exporterOption.
_OP_OPTION = {"export_lights": "includeLights", "export_cameras": "includeCameras"}
_MCP_OPTION = {"includeLights": "include_lights", "includeCameras": "include_cameras"}

# WHAT EVERY OTHER FORMAT DOES WITH A LIGHT OR A CAMERA, for the refusal of includeLights / includeCameras outside
# glTF - the same table as _carriage, in a sentence (measured on all five builds, docs/22 B255-B258).
_INCLUDE_ELSEWHERE = {
    ("USD", "includeLights"): "the USD exporter writes lights without being asked (a spot light only from 4.2)",
    ("USD", "includeCameras"): "the USD exporter writes cameras without being asked",
    ("Alembic", "includeLights"): "Alembic has no form for a light - an empty transform stands in its place",
    ("Alembic", "includeCameras"): "the Alembic exporter writes cameras without being asked",
    ("mesh", "includeLights"): "OBJ, STL and PLY hold mesh geometry only",
    ("mesh", "includeCameras"): "OBJ, STL and PLY hold mesh geometry only",
    ("GP", "includeLights"): "an SVG or PDF holds Grease Pencil strokes only",
    ("GP", "includeCameras"): "an SVG or PDF holds Grease Pencil strokes only",
}

# extension -> (candidate operators in order, selection kwarg, extra kwargs, human label)
#
# EVERY ENTRY WAS WRITTEN FROM THE OPERATOR'S OWN SIGNATURE rather than from memory, because the
# selection keyword differs for all of them and a wrong one is either a raise or a silent no-op.
_EXPORTERS = {
    ".glb":  (("export_scene.gltf",), "use_selection", {"export_format": "GLB"}, "glTF binary"),
    ".gltf": (("export_scene.gltf",), "use_selection", {"export_format": "GLTF_SEPARATE"},
              "glTF + separate resources"),
    ".obj":  (("wm.obj_export", "export_scene.obj"),
              {"wm.obj_export": "export_selected_objects", "export_scene.obj": "use_selection"},
              {}, "Wavefront OBJ"),
    ".usd":  (("wm.usd_export",), "selected_objects_only", {}, "USD"),
    ".usda": (("wm.usd_export",), "selected_objects_only", {}, "USD ASCII"),
    ".usdc": (("wm.usd_export",), "selected_objects_only", {}, "USD binary"),
    ".usdz": (("wm.usd_export",), "selected_objects_only", {}, "USDZ"),
    ".abc":  (("wm.alembic_export",), "selected", {}, "Alembic"),
    ".stl":  (("wm.stl_export", "export_mesh.stl"),
              {"wm.stl_export": "export_selected_objects", "export_mesh.stl": "use_selection"},
              {}, "STL"),
    ".ply":  (("wm.ply_export", "export_mesh.ply"),
              {"wm.ply_export": "export_selected_objects", "export_mesh.ply": "use_selection"},
              {}, "PLY"),
    # GREASE PENCIL STROKES AS SVG AND PDF (2026-10-11), the 2D vector hand-off. Measured on 3.6.23, 4.2.17, 5.0.1 and
    # 5.2.1: wm.gpencil_export_* before 4.3, wm.grease_pencil_export_* after; selection is the ENUM
    # selected_object_type ACTIVE | SELECTED | VISIBLE, not a boolean (export_scene sets it); no scene camera is needed.
    # 4.4.0's exporter CRASHES Blender headless (_GP_EXPORT_CRASHES).
    ".svg":  (("wm.grease_pencil_export_svg", "wm.gpencil_export_svg"), "selected_object_type", {},
              "Grease Pencil SVG"),
    ".pdf":  (("wm.grease_pencil_export_pdf", "wm.gpencil_export_pdf"), "selected_object_type", {},
              "Grease Pencil PDF"),
}
_GP_VECTOR = (".svg", ".pdf")
# EXCEPTION_ACCESS_VIOLATION in io::grease_pencil::get_visible_strokes (from prepare_render_params) on every headless SVG
# and PDF export Blender 4.4.0 was given - with and without a depsgraph update or a frame_set first (2026-10-11). 4.3 and
# 4.5 ship the same exporter generation and are refused with it, unmeasured; 3.6, 4.2, 5.0 and 5.2 export.
_GP_EXPORT_CRASHES = ((4, 3, 0), (5, 0, 0))
# THE 5.x PDF EXPORTER FAILS A WHOLE SESSION AT A TIME, AND NOT BY VERSION (2026-10-11, headless): it reports FINISHED
# and writes no file, with libharu errors on the console ("Unable to export PDF"). In one Blender process every export
# fails or every one works: 5.0.1 wrote 25 of 25 in the morning and 0 of 25 (five fresh processes) four hours later,
# while 5.2.1 went the other way (21 of 26 missing, then 4 of 5 processes writing every file); render size, camera and
# selection mode made no difference. The legacy exporter (3.6, 4.2) wrote 25 of 25. So the export is attempted and a
# missing file is refused with this explanation (_GP_PDF_FLAKY_NOTE) - refusing 5.x outright would block the sessions
# where it works. A first reading of the 5.2 numbers as a 5.2 defect refused PDF on 5.2 for a few hours; that was wrong.
_GP_PDF_FLAKY_NOTE = (" Blender's Grease Pencil PDF exporter from 5.0 reports success and writes nothing in some "
                      "Blender sessions - every PDF in that session alike, with libharu errors on Blender's console - and "
                      "works in others (measured on 5.0.1 and 5.2.1, 2026-10-11; render size, camera and selection made "
                      "no difference). SVG export is not affected; a fresh Blender session may write the PDF.")

# Formats that carry TIME. Passing a frame range to one that does not is refused rather than
# dropped, because "I exported an animation" and "I exported frame 1" look identical afterwards.
_ANIMATED = {".abc", ".usd", ".usda", ".usdc", ".usdz", ".glb", ".gltf"}

_FRAME_KWARG = {".abc": ("start", "end"), ".usd": ("start", "end"), ".usda": ("start", "end"),
                ".usdc": ("start", "end"), ".usdz": ("start", "end")}

# =============================================================================
# WHAT THE FILE DOES NOT CARRY, BY OBJECT TYPE
# =============================================================================
# A glTF of a light and a camera came out 132 bytes on 2026-09-28 - a valid file with NOTHING of either in it,
# because the glTF exporter writes lights and cameras only with its export_lights / export_cameras options, which are
# off by default - and export_scene answered a bare ok. The answer now names every object in scope the file does not
# carry. The defaults stay (decided the same day: in a hand-off file lights and cameras clutter an Unreal import), and
# includeLights / includeCameras set the two options for a caller who wants them.
#
# MEASURED, NOT RECALLED - tools/probe_export_types.py re-runs it on every installed build, docs/22 B255-B258 hold the
# result - on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1 (2026-09-28): one object of every type Blender
# makes, each exported ALONE through the operator and selection keyword resolve_exporter picks, exporter defaults
# otherwise, and read back by something other than the exporter - the .gltf JSON's nodes, OBJ's o/v/f lines, STL's
# triangle count, PLY's header, the .usda prims under the object, the Alembic file imported again. Then every option
# of that exporter that is OFF by default and names an object kind was turned on alone, to find the ones that bring a
# type in. "Carried" means the type's content: a light probe glTF writes as a bare node, or a light Alembic writes as
# an empty transform, is not carried. With each type's own data - what a geometry-nodes modifier makes of an object is
# not read here.
_MESHLIKE = ("MESH", "CURVE", "FONT", "META", "SURFACE")
_FAMILY = {".glb": "glTF", ".gltf": "glTF", ".obj": "mesh", ".stl": "mesh", ".ply": "mesh",
           ".usd": "USD", ".usda": "USD", ".usdc": "USD", ".usdz": "USD", ".abc": "Alembic", ".svg": "GP", ".pdf": "GP"}


def _carriage(family, obj, version, included=()):
    """How a `family` file treats obj on this build: "carried", "unverified", or (exporter option or None, why).

    "unverified" is a type this was not measured with for that format - a point cloud before 5.0, which no operator
    here could make on 3.6-4.4; a volume in USD, whose exporter writes file-backed grids the measurement had none of;
    any type a later Blender adds. It is reported as unverified, never guessed into either list.

    `included` holds the types the caller asked in with includeLights / includeCameras ("LIGHT", "CAMERA"); only a
    glTF reads it, because only glTF's exporter has the switch (the others are refused before an export runs).
    """
    t = obj.type
    light = getattr(obj.data, "type", None) if t == "LIGHT" else None
    if family == "GP":                                      # SVG, PDF: the Grease Pencil exporters
        if t in ("GPENCIL", "GREASEPENCIL"):
            return "carried"
        return (None, "SVG and PDF export write Grease Pencil strokes only")
    if t == "POINTCLOUD" and version < (5, 0, 0):
        return "unverified"
    if family == "mesh":                                    # OBJ, STL, PLY
        if t in _MESHLIKE:
            return "carried"
        if t in ("CURVES", "POINTCLOUD", "VOLUME", "GPENCIL", "GREASEPENCIL", "ARMATURE", "LATTICE", "EMPTY",
                 "LIGHT", "LIGHT_PROBE", "CAMERA", "SPEAKER"):
            return (None, "the format holds mesh geometry only")
        return "unverified"
    if family == "glTF":
        if t in ("MESH", "CURVE", "FONT", "SURFACE", "ARMATURE", "EMPTY"):
            return "carried"
        if t == "LIGHT":
            if light == "AREA":
                return (None, "glTF's lights (KHR_lights_punctual) are point, spot and sun - an area light is left "
                              "out with includeLights (the exporter's export_lights) on too")
            if "LIGHT" in included:
                return "carried"
            return ("export_lights", "the glTF exporter writes lights only with its export_lights option, off by "
                                     "default - includeLights sets it")
        if t == "CAMERA":
            if "CAMERA" in included:
                return "carried"
            return ("export_cameras", "the glTF exporter writes cameras only with its export_cameras option, off by "
                                      "default - includeCameras sets it")
        if t == "POINTCLOUD":
            return (("use_mesh_vertices", "5.2's glTF exporter writes the points as a mesh with its use_mesh_vertices "
                                          "option") if version >= (5, 2, 0)
                    else (None, "this glTF exporter writes no point cloud (measured on 5.0.1)"))
        if t in ("META", "CURVES", "VOLUME", "GPENCIL", "GREASEPENCIL", "LATTICE", "LIGHT_PROBE", "SPEAKER"):
            return (None, "the glTF exporter has no form for this type - at most a bare node stands in its place")
        return "unverified"
    if family == "USD":
        if t in ("MESH", "CURVE", "META", "CURVES", "EMPTY", "CAMERA", "POINTCLOUD"):
            return "carried"
        if t == "LIGHT":
            if light == "SPOT" and version < (4, 2, 0):
                return (None, "this USD exporter writes no spot light (measured on 3.6.23; 4.2.17 writes one)")
            return "carried"
        if t == "ARMATURE":
            return "carried" if version >= (4, 2, 0) else (None, "this USD exporter writes no armature (measured on "
                                                                 "3.6.23; 4.2.17 writes a Skeleton)")
        if t == "FONT":
            return "carried" if version >= (5, 0, 0) else (None, "this USD exporter writes no text object (measured "
                                                                 "on 3.6.23-4.4.0; 5.0.1 writes it as a mesh)")
        if t == "SURFACE":
            return "carried" if version >= (5, 2, 0) else (None, "this USD exporter writes no NURBS surface (measured "
                                                                 "on 3.6.23-5.0.1; 5.2.1 writes it as a mesh)")
        if t in ("GPENCIL", "GREASEPENCIL", "LATTICE", "LIGHT_PROBE", "SPEAKER"):
            return (None, "the USD exporter has no form for this type")
        return "unverified"
    if family == "Alembic":
        if t in ("MESH", "CURVE", "META", "EMPTY", "CAMERA"):
            return "carried"
        if t == "CURVES":
            return "carried" if version >= (4, 2, 0) else (None, "this Alembic exporter writes no hair curves "
                                                                 "(measured on 3.6.23; 4.2.17 writes them)")
        if t == "SURFACE":
            return ("curves_as_mesh", "the Alembic exporter writes a NURBS surface only with its curves_as_mesh "
                                      "option, off by default - an empty transform stands in its place")
        if t in ("FONT", "POINTCLOUD", "VOLUME", "GPENCIL", "GREASEPENCIL", "ARMATURE", "LATTICE", "LIGHT",
                 "LIGHT_PROBE", "SPEAKER"):
            return (None, "the Alembic exporter has no form for this type - an empty transform stands in its place")
        return "unverified"
    return "unverified"


def _not_carried(ext, label, in_scope, included=()):
    """(notCarried rows, notVerified rows, note or None) for the objects this export was asked to write.

    One row per type and option, every object named: what a caller needs to see a 132-byte glTF for what it is.
    `option` is THIS op's parameter that would include them (includeLights / includeCameras), None where it has none;
    exporterOption is the exporter's own name for it, where the exporter has one - use_mesh_vertices and curves_as_mesh
    have no parameter here, and the note says so."""
    family = _FAMILY.get(ext)
    version = tuple(bpy.app.version)
    dropped, unverified = {}, {}
    for obj in in_scope:
        verdict = _carriage(family, obj, version, included) if family else "unverified"
        if verdict == "carried":
            continue
        if verdict == "unverified":
            unverified.setdefault(obj.type, []).append(obj.name)
            continue
        option, why = verdict
        row = dropped.setdefault((obj.type, option, why), [])
        row.append(obj.name)
    rows = [{"type": t, "count": len(names), "objects": sorted(names)[:50], "option": _OP_OPTION.get(option),
             "exporterOption": option, "why": why}
            for (t, option, why), names in sorted(dropped.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))]
    unv = [{"type": t, "count": len(names), "objects": sorted(names)[:50]} for t, names in sorted(unverified.items())]
    missing = sum(r["count"] for r in rows)
    if not missing:
        return rows, unv, None
    parts = []
    for r in rows:
        them = "it" if r["count"] == 1 else "them"
        if r["option"]:
            hint = " (%s would include %s)" % (r["option"], them)
        elif r["exporterOption"]:
            hint = " (its exporter's %s option would include %s - export_scene does not set it)" % (r["exporterOption"],
                                                                                                   them)
        else:
            hint = ""
        parts.append("%d %s%s" % (r["count"], r["type"], hint))
    head = ("the file carries NONE of the %d object(s) in scope - a valid %s file with nothing of them in it"
            % (len(in_scope), label) if missing == len(in_scope)
            else "the file does not carry %d of the %d object(s) in scope" % (missing, len(in_scope)))
    options = sorted({r["option"] for r in rows if r["option"]})
    note = ("%s: %s. %snotCarried names each object and why."
            % (head, "; ".join(parts),
               "Through MCP bl_export_scene takes %s. " % " and ".join(_MCP_OPTION[o] for o in options) if options
               else ""))
    return rows, unv, note


def _op_exists(dotted):
    """Is bpy.ops.<a>.<b> actually REGISTERED on this build?

    An exporter can be a DISABLED ADD-ON, in which case the operator simply is not there. Asking
    first turns "AttributeError: 'BPyOps' object has no attribute" into a sentence naming the
    format and the add-on, which is the difference between a message and a traceback.

    NOT hasattr, which this was until 2026-09-28: bpy.ops makes an operator wrapper for ANY name it is
    asked for, so hasattr said yes to every candidate on every build and the first one in each list
    always won. On 3.6.23, which has no wm.stl_export (it arrived at 4.2; wm.stl_import is already
    there), export_scene .stl called it anyway and failed "could not be found" instead of falling back to
    export_mesh.stl, and a disabled exporter raised that AttributeError rather than the sentence above.
    get_rna_type() is the registry lookup: it raises for an operator that is not there (docs/22 B259).
    test_blender_refusals B121 now stubs bpy.ops the way Blender builds it, lazily.
    """
    mod, _, name = dotted.partition(".")
    group = getattr(bpy.ops, mod, None)
    if group is None:
        return False
    try:
        getattr(group, name).get_rna_type()
        return True
    except Exception:                       # noqa: BLE001
        return False


def resolve_exporter(ext):
    """(dotted operator, selection kwarg, extra kwargs, label) for a file extension.

    PURE, and separated for that reason: everything else here needs a real Blender, while the
    version-drift logic - which name exists on this build, which selection keyword goes with it - is
    a table lookup and is where the mistakes live.
    """
    entry = _EXPORTERS.get(ext)
    if entry is None:
        raise MifOpError("no exporter for '%s'. This op writes: %s. FBX is written by export_mesh, "
                         "which also handles skeletal export for the Unreal round trip. NOTHING "
                         "was written." % (ext, ", ".join(sorted(_EXPORTERS))))
    candidates, sel, extra, label = entry
    for dotted in candidates:
        if _op_exists(dotted):
            sel_kwarg = sel[dotted] if isinstance(sel, dict) else sel
            return dotted, sel_kwarg, dict(extra), label
    raise MifOpError("this Blender (%s) has no %s exporter - tried %s. It is usually an add-on and "
                     "may be disabled in Preferences > Add-ons. NOTHING was written."
                     % (bpy.app.version_string, label, " and ".join(candidates)))


# =============================================================================
# BVH - MOTION CAPTURE IN AND OUT (1.3 gap audit, 2026-10-19)
# =============================================================================
# BVH is the motion-capture interchange (capture libraries, markerless and phone capture), and the usual input of
# retarget_action. Route: the bundled io_anim_bvh add-on's import_anim.bvh / export_anim.bvh, enabled for the session when
# off (measured present on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1). Options arrive as a `bvh` dict, each checked here.
_BVH_IMPORT = {"rotateMode": ("rotate_mode", str), "scale": ("global_scale", float), "frameStart": ("frame_start", int),
               "useFpsScale": ("use_fps_scale", bool), "updateSceneFps": ("update_scene_fps", bool),
               "updateSceneDuration": ("update_scene_duration", bool), "cyclic": ("use_cyclic", bool)}
_BVH_EXPORT = {"rotateMode": ("rotate_mode", str), "scale": ("global_scale", float),
               "rootTransformOnly": ("root_transform_only", bool)}


def _bvh_enable():
    if _op_exists("import_anim.bvh"):
        return None
    import addon_utils   # noqa: PLC0415
    try:
        addon_utils.enable("io_anim_bvh", default_set=False, persistent=False)
    except Exception:                                           # noqa: BLE001
        pass
    if not _op_exists("import_anim.bvh"):
        raise MifOpError("this Blender (%s) has no BVH importer/exporter (io_anim_bvh could not be enabled). NOTHING was "
                         "changed." % bpy.app.version_string)
    return "io_anim_bvh"


def _bvh_options(raw, table, dotted, tail):
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise MifOpError("bvh is an object of %s. %s" % (", ".join(sorted(table)), tail))
    bad = sorted(set(raw) - set(table))
    if bad:
        raise MifOpError("bvh has unknown key(s) %s - accepted: %s. %s" % (bad, ", ".join(sorted(table)), tail))
    props = _rna_props(dotted)
    out = {}
    for k, v in raw.items():
        prop, kind = table[k]
        if kind is bool and not isinstance(v, bool) or kind is str and not isinstance(v, str) or kind in (int, float) and (
                isinstance(v, bool) or not isinstance(v, (int, float))):
            raise MifOpError("bvh.%s must be a %s, got %r. %s" % (k, kind.__name__, v, tail))
        if kind is str:
            items = [i.identifier for i in props[prop].enum_items]
            v = v.upper()
            if v not in items:
                raise MifOpError("bvh.%s must be one of %s, got %r. %s" % (k, ", ".join(items), raw[k], tail))
        out[prop] = kind(v)
    return out


def _rna_props(dotted):
    mod, name = dotted.split(".")
    return {p.identifier: p for p in getattr(getattr(bpy.ops, mod), name).get_rna_type().properties}


def _bvh_file_info(path):
    joints, frames, ftime = 0, None, None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith(("ROOT ", "JOINT ")):
                joints += 1
            elif s.startswith("Frames:"):
                frames = int(s.split(":")[1])
            elif s.startswith("Frame Time:"):
                ftime = float(s.split(":")[1])
                break
    return {"joints": joints, "frames": frames, "frameTime": ftime,
            "fps": round(1.0 / ftime, 4) if ftime else None}


def _import_bvh(params, path):
    tail = "NOTHING was imported."
    extra = sorted(k for k in params if k not in ("file", "filepath", "path", "collection", "bvh"))
    if extra:
        raise MifOpError("%s are other formats' options; a BVH takes bvh {%s}. %s"
                         % (extra, ", ".join(sorted(_BVH_IMPORT)), tail))
    enabled = _bvh_enable()
    opts = _bvh_options(params.get("bvh"), _BVH_IMPORT, "import_anim.bvh", tail)
    coll_name = take(params, "collection", default=None, kind=str)
    coll = None
    if coll_name is not None:
        coll = bpy.data.collections.get(coll_name)
        if coll is None:
            raise MifOpError("no collection named '%s'. %s" % (coll_name, tail))
    if not os.path.isfile(path):
        raise MifOpError("no such file: %s. %s" % (path, tail))
    before = set(bpy.data.objects)
    acts_before = set(bpy.data.actions)
    try:
        bpy.ops.import_anim.bvh(filepath=path, **opts)
    except RuntimeError as exc:
        raise MifOpError("Blender's BVH importer refused %s: %s" % (path, str(exc).strip()))
    new = [o for o in bpy.data.objects if o not in before]
    arms = [o for o in new if o.type == "ARMATURE"]
    if not arms:
        raise MifOpError("the BVH importer reported done and made no armature from %s." % path)
    if coll is not None:
        for o in new:
            for c in list(o.users_collection):
                c.objects.unlink(o)
            coll.objects.link(o)
    arm = arms[0]
    act = arm.animation_data.action if arm.animation_data else None
    sc = bpy.context.scene
    out = {"file": path, "format": "BVH", "armature": arm.name, "bones": len(arm.data.bones),
           "action": act.name if act else None,
           "frameRange": [round(v, 3) for v in act.frame_range] if act else None,
           "sceneFps": round(sc.render.fps / sc.render.fps_base, 4), "fileInfo": _bvh_file_info(path),
           "options": opts, "actionsCreated": [a.name for a in bpy.data.actions if a not in acts_before],
           "imported": [o.name for o in new], "collection": coll.name if coll else None}
    if enabled:
        out["addonEnabledForSession"] = enabled
    return out


def _export_bvh(params, path, raw):
    tail = "NOTHING was written."
    extra = sorted(k for k in params if k not in ("file", "filepath", "path", "object", "objects", "frameStart",
                                                  "frameEnd", "overwrite", "replaceExisting", "bvh"))
    if extra:
        raise MifOpError("%s are other formats' options; a BVH takes object (the armature), frameStart, frameEnd and "
                         "bvh {%s}. %s" % (extra, ", ".join(sorted(_BVH_EXPORT)), tail))
    enabled = _bvh_enable()
    names = params.get("objects", params.get("object"))
    names = [names] if isinstance(names, str) else names
    if not isinstance(names, list) or len(names) != 1:
        raise MifOpError("a BVH holds one armature - name it with object. %s" % tail)
    arm = get_object(names[0])
    if arm.type != "ARMATURE":
        raise MifOpError("'%s' is a %s; a BVH is written from an armature. %s" % (arm.name, arm.type, tail))
    overwrite = take_bool(params, "overwrite", "replaceExisting", default=True)
    before = file_before(path)
    if before is not None and not overwrite:
        raise MifOpError("%s already exists and overwrite:false. %s" % (path, tail))
    sc = bpy.context.scene
    start = take_int(params, "frameStart", default=sc.frame_start)
    end = take_int(params, "frameEnd", default=sc.frame_end)
    if end < start:
        raise MifOpError("frameEnd %d is before frameStart %d. %s" % (end, start, tail))
    opts = _bvh_options(params.get("bvh"), _BVH_EXPORT, "export_anim.bvh", tail)
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        raise MifOpError("the folder '%s' does not exist. %s" % (folder, tail))
    vl = bpy.context.view_layer
    prev = vl.objects.active
    try:
        vl.objects.active = arm
        bpy.ops.export_anim.bvh(filepath=path, frame_start=start, frame_end=end, **opts)
    except RuntimeError as exc:
        raise MifOpError("Blender's BVH exporter refused: %s. %s" % (str(exc).strip(), tail))
    finally:
        vl.objects.active = prev
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        raise MifOpError("the BVH exporter reported done and wrote nothing at %s." % path)
    info = _bvh_file_info(path)
    out = {"file": path, "format": "BVH", "armature": arm.name, "bones": len(arm.data.bones), "frameStart": start,
           "frameEnd": end, "bytes": os.path.getsize(path), "fileInfo": info, "options": opts}
    out.update(replaced_fields(before))
    if info["joints"] != len(arm.data.bones):
        out["warning"] = "the file holds %d joints and the armature %d bones" % (info["joints"], len(arm.data.bones))
    if enabled:
        out["addonEnabledForSession"] = enabled
    return out


def op_export_scene(params):
    """Write glTF, OBJ, USD, Alembic, STL or PLY - everything except the FBX export_mesh owns.

    And Grease Pencil strokes as .svg or .pdf (2026-10-11): the objects named or selected, or every visible one;
    refused when nothing Grease Pencil is in scope, and on Blender 4.3-4.5, whose exporter crashed 4.4.0 headless. A 5.x
    PDF that the exporter does not write (it fails whole sessions at a time) is refused after the call, saying so. An
    SVG's stroke elements are counted in the written file (svgStrokes).

    params:
      file / filepath / path (str)   required. The FORMAT COMES FROM THE EXTENSION.
      objects / object (list|str)    export only these. Default the whole scene.
      selectedOnly (bool)            export the current selection instead of naming objects
      applyModifiers (bool)          where the format's exporter supports it
      frameStart / frameEnd (int)    for the formats that carry time. Refused on those that do not.
      animation (bool)               glTF only - whether to write animation at all. REFUSED on every
                                     other format rather than ignored; use frameStart/frameEnd there
      includeLights (bool)           glTF only - write point, spot and sun lights (the exporter's
                                     export_lights). Default false, the exporter's own default
      includeCameras (bool)          glTF only - write cameras (export_cameras). Default false
      overwrite (bool)               default true: an existing file is REPLACED, and the answer says so
      exporterOptions (dict)         {name: value} of the chosen exporter operator's own options, checked
                                     against its RNA on this build (2026-10-06)
      bvh (dict)                     a .bvh (1.3): object names the armature; frameStart / frameEnd (default the
                                     scene's); bvh {rotateMode, scale, rootTransformOnly}. The written file's joints
                                     and frames are read back.

    AN EXISTING FILE IS REPLACED, NEVER SILENTLY (decided 2026-09-28). overwrite defaults true - 1.0.0 did it,
    Blender's exporters do it, and an agent re-exporting to one path would pay a refusal every time - and the
    answer carries overwrote, with replacedBytes and replacedModified (UTC) measured off the old file before the
    write. overwrite:false refuses an existing path by name before anything runs. The named file is the one
    measured: a .gltf's .bin and an .obj's .mtl go with it.

    LIGHTS AND CAMERAS STAY OUT OF A glTF UNLESS ASKED (decided 2026-09-28): the exporter's defaults are kept,
    because a light or camera in a hand-off file clutters an Unreal import. includeLights / includeCameras put
    them in. Every other format decides for itself - USD writes both, Alembic writes cameras and has no form for a
    light, OBJ/STL/PLY hold mesh geometry only - so there the two are REFUSED rather than accepted and ignored,
    as `animation` is.

    WHAT THE FILE DOES NOT CARRY IS IN THE ANSWER. Every other exporter option stays at its default, and each
    format leaves some object types out: glTF writes lights and cameras only when asked, OBJ, STL and PLY hold
    mesh geometry only, Alembic writes no light. notCarried names every object in scope (objectsInScope) the file
    does not carry - by type, with the parameter of this op that would include it (option) and the exporter's own
    name for it (exporterOption) where there is one - notCarriedCount counts them, and notCarriedNote says so, and
    says NONE when the file is a valid one with nothing asked for in it. notVerified names objects of a type this was
    not measured with for the format. Measured per type on all five builds (_carriage).
    """
    reject_unknown(params, _EXPORT_KEYS, "export_scene")
    raw = take(params, "file", "filepath", "path", required=True, kind=str)
    path = bpy.path.abspath(str(raw))
    # Checked before the extension is read, so an unusable path is refused as one rather than as a
    # missing format. The glTF exporter otherwise fails deep inside itself and the traceback ends up
    # pasted into this op's message.
    check_output_path(raw, path, "written")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bvh":
        return _export_bvh(params, os.path.abspath(path), raw)
    if params.get("bvh") is not None:
        raise MifOpError("bvh holds a .bvh file's options, and this file is %s. NOTHING was written." % (ext or "unnamed"))
    if not ext:
        raise MifOpError("'%s' has no file extension, and the format is taken from it. Give one of: "
                         "%s. NOTHING was written." % (raw, ", ".join(sorted(_EXPORTERS))))
    dotted, sel_kwarg, kwargs, label = resolve_exporter(ext)
    if ext in _GP_VECTOR and _GP_EXPORT_CRASHES[0] <= tuple(bpy.app.version) < _GP_EXPORT_CRASHES[1]:
        raise MifOpError("%s export is refused on Blender %s: this generation's Grease Pencil exporter crashed Blender "
                         "4.4.0 on every headless SVG and PDF export (an access violation in its visible-stroke pass), "
                         "and 4.3 and 4.5 ship the same exporter. Blender 3.6, 4.2 and 5.0+ export it. NOTHING was "
                         "written." % (label, bpy.app.version_string))

    # Read BEFORE the exists() test - see audit_blender_shortcircuit_take. Inside the `and`, a
    # malformed overwrite was only ever checked when the file happened to be there.
    overwrite = take_bool(params, "overwrite", "replaceExisting", default=True)
    if os.path.exists(path) and not overwrite:
        raise MifOpError("%s already exists and overwrite:false. NOTHING was written." % path)

    start = take_int(params, "frameStart", default=None)
    end = take_int(params, "frameEnd", default=None)
    if (start is not None or end is not None) and ext not in _ANIMATED:
        raise MifOpError("%s carries no animation, so a frame range would have been accepted and "
                         "silently ignored - 'I exported an animation' and 'I exported one frame' "
                         "look identical afterwards. Drop frameStart/frameEnd, or write %s instead. "
                         "NOTHING was written."
                         % (label, " or ".join(sorted(_ANIMATED))))
    if start is not None and end is not None and end < start:
        raise MifOpError("frameEnd %d is before frameStart %d. NOTHING was written." % (end, start))

    # `animation` IS A glTF OPTION AND IS REFUSED EVERYWHERE ELSE - the frame-range rule just above,
    # applied to the key that escaped it. It used to be read only inside the glTF branch further
    # down, so OBJ, STL, PLY, USD and Alembic accepted it and ignored it: measured on 3.6.23 through
    # 5.2.1, OBJ off and on wrote the same bytes while glTF went 1936 -> 2700. On Alembic and USD it
    # was worse than harmless - animation:false still wrote whatever the frame range produced. Found
    # by audit_blender_inert_params. HERE, above the directory probe, because it needs nothing but
    # params and a refused call should not have touched the disk first.
    if params.get("animation") is not None and ext not in (".glb", ".gltf"):
        raise MifOpError("animation is a glTF option, and %s would have accepted it and ignored "
                         "it. %s NOTHING was written."
                         % (label, "Set the range with frameStart/frameEnd instead."
                            if ext in _ANIMATED else "This format carries no animation at all."))
    # includeLights / includeCameras ARE glTF OPTIONS TOO, refused elsewhere for the same reason and at the same
    # point: every other format decides for itself what it does with a light or a camera, so either value would be
    # accepted and change nothing. The sentence says what that format does instead.
    for key in ("includeLights", "includeCameras"):
        if params.get(key) is not None and ext not in (".glb", ".gltf"):
            raise MifOpError("%s is a glTF option - the glTF exporter writes %s only when asked - and %s: %s, so it "
                             "would have changed nothing. Drop %s. NOTHING was written."
                             % (key, "lights" if key == "includeLights" else "cameras", label,
                                _INCLUDE_ELSEWHERE[(_FAMILY[ext], key)], key))

    # THE OUTPUT DIRECTORY IS CHECKED BY WRITING TO IT. os.access reports the DACL on Windows and
    # not the effective permission, so it answers yes for directories a write then fails on.
    # check_writable_dir, shared with render_animation - see it for why the probe's name is no longer fixed.
    outdir = os.path.dirname(path) or "."
    try:
        check_writable_dir(outdir)
    except Exception as exc:                # noqa: BLE001
        raise MifOpError("the output directory '%s' cannot be written (%s). NOTHING was written."
                         % (outdir, exc))

    names = take(params, "objects", "object", "name")
    if isinstance(names, str):
        names = [names]
    selected_only = take_bool(params, "selectedOnly", default=False)
    if names is not None and selected_only:
        raise MifOpError("pass objects OR selectedOnly, not both - one names a set and the other "
                         "uses whatever happens to be selected. NOTHING was written.")
    if names is not None and not isinstance(names, (list, tuple)):
        raise MifOpError("'objects' must be a list of names. NOTHING was written.")

    targets = []
    if names:
        # RESOLVED BEFORE ANYTHING IS SELECTED, so a typo cannot leave the scene's selection
        # rearranged as a side effect of a failed export.
        targets = [get_object(n) for n in names]
    # WHAT THE EXPORTER IS HANDED, taken before the selection is touched: the named objects, the selection, or every
    # object in the view layer. The answer's notCarried is judged against exactly this set.
    if names:
        in_scope = list(targets)
    elif selected_only:
        in_scope = [o for o in bpy.context.view_layer.objects if o.select_get()]
    else:
        in_scope = list(bpy.context.view_layer.objects)

    kwargs["filepath"] = path
    if ext in _GP_VECTOR:
        # AN ENUM, NOT A SWITCH: the named objects or the selection go as SELECTED, the whole scene as VISIBLE. Nothing
        # Grease Pencil in scope is refused here - the exporter writes a valid, empty SVG for it (measured: 359-380
        # bytes with no stroke element) and reports FINISHED.
        kwargs[sel_kwarg] = "SELECTED" if (names or selected_only) else "VISIBLE"
        if not any(o.type in ("GPENCIL", "GREASEPENCIL") for o in in_scope):
            raise MifOpError("no Grease Pencil object is in scope (%d object(s): %s) - %s export writes Grease Pencil "
                             "strokes only, and would write an empty file. NOTHING was written."
                             % (len(in_scope), ", ".join(sorted(o.name for o in in_scope)[:8]) or "none", label))
    elif names or selected_only:
        kwargs[sel_kwarg] = True
    # WHAT RANGE ACTUALLY REACHED THE FILE, and by which route. Reported rather than derived from
    # _FRAME_KWARG at the bottom: that lookup returns ("", "") for glTF, so `frameRange` came back
    # [null, null] on a glTF export whose range HAD been applied through the scene. A field that
    # reports what happened saying nothing happened is the shape this project keeps finding worse
    # versions of - here the caller's values were honoured and the response denied it. Found by the
    # second seat 2026-09-15 (advisory 4 of their batch 4).
    applied_range, range_route = None, None
    if start is not None or end is not None:
        sk, ek = _FRAME_KWARG.get(ext, (None, None))
        if sk:
            kwargs[sk] = int(start if start is not None else bpy.context.scene.frame_start)
            kwargs[ek] = int(end if end is not None else bpy.context.scene.frame_end)
            applied_range = [kwargs[sk], kwargs[ek]]
            range_route = "exporter arguments (%s, %s)" % (sk, ek)
        else:
            # glTF takes its range from the SCENE rather than from arguments, so the scene is set
            # and restored instead of a kwarg being invented that the operator would reject. The
            # values are computed here and written below, inside the try that restores them.
            applied_range = [int(start if start is not None else bpy.context.scene.frame_start),
                             int(end if end is not None else bpy.context.scene.frame_end)]
            range_route = "the scene's frame range, set for the call and restored after it"
    if ext in (".glb", ".gltf") and params.get("animation") is not None:
        kwargs["export_animations"] = take_bool(params, "animation", default=True)
    included = set()
    if ext in (".glb", ".gltf"):
        for key, exporter_key, kind in (("includeLights", "export_lights", "LIGHT"),
                                        ("includeCameras", "export_cameras", "CAMERA")):
            if params.get(key) is not None:
                kwargs[exporter_key] = take_bool(params, key, default=False)
            if kwargs.get(exporter_key):
                included.add(kind)
    if params.get("applyModifiers") is not None:
        # SPELLED PER FORMAT, and refused where the exporter has no such option rather than dropped.
        # An applyModifiers that is accepted and ignored is the worst outcome here: the file is
        # written from unevaluated geometry and looks like a successful export of the wrong thing.
        if ext in (".glb", ".gltf"):
            kwargs["export_apply"] = take_bool(params, "applyModifiers", default=True)
        elif ext == ".obj" and dotted == "wm.obj_export":
            kwargs["apply_modifiers"] = take_bool(params, "applyModifiers", default=True)
        else:
            raise MifOpError("applyModifiers is not something the %s exporter on this build takes. "
                             "NOTHING was written." % label)

    # THE EXPORTER'S OWN OPTIONS (2026-10-06): exporterOptions {name: value}, each checked against this exporter
    # operator's RNA on this build (and an enum value against its items) - OBJ apply_transform off for local space,
    # USD merge parent transforms, STL modifier evaluation mode. What this op already sets (the path, the selection,
    # a frame range or an option above) is refused here, so one value cannot be given twice.
    opts = params.get("exporterOptions")
    if opts is not None:
        if not isinstance(opts, dict) or not opts:
            raise MifOpError("exporterOptions is {name: value} of the %s exporter's own options. NOTHING was written."
                             % label)
        mod_name, _, op_name = dotted.partition(".")
        rna = getattr(getattr(bpy.ops, mod_name), op_name).get_rna_type().properties
        offered = sorted(p.identifier for p in rna if p.identifier != "rna_type" and not p.is_readonly)
        for key, value in opts.items():
            if key not in offered or key in ("filepath", sel_kwarg):
                raise MifOpError("%s has no option %r this can set. Its options: %s. NOTHING was written."
                                 % (dotted, key, ", ".join(o for o in offered if o not in ("filepath", sel_kwarg))))
            if key in kwargs:
                raise MifOpError("%s is already set by this call's own parameters (%r). NOTHING was written."
                                 % (key, kwargs[key]))
            prop = rna[key]
            if prop.type == "ENUM":
                items = {i.identifier for i in prop.enum_items}
                vals = value if getattr(prop, "is_enum_flag", False) and isinstance(value, list) else [value]
                if any(v not in items for v in vals):
                    raise MifOpError("%s %r is not one of %s. NOTHING was written." % (key, value,
                                                                                       ", ".join(sorted(items))))
                value = set(vals) if getattr(prop, "is_enum_flag", False) else value
            kwargs[key] = value

    from .ops_common import selection_restore, selection_snapshot
    # WHAT THIS EXPORT REPLACES, measured now - after every refusal, right before the write - because afterwards the
    # old file is gone. See file_before.
    replaced = file_before(path)
    snap = selection_snapshot()
    scene = bpy.context.scene
    frames_before = (scene.frame_start, scene.frame_end)
    started = time.time()
    try:
        if names:
            for obj in bpy.context.view_layer.objects:
                obj.select_set(False)
            for obj in targets:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = targets[0]
        if ext in (".glb", ".gltf") and applied_range is not None:
            scene.frame_start, scene.frame_end = applied_range
        mod, _, opname = dotted.partition(".")
        try:
            getattr(getattr(bpy.ops, mod), opname)(**kwargs)
        except TypeError as exc:
            raise MifOpError("the %s exporter on this build does not accept one of %s (%s). This "
                             "is version drift in the operator's own signature. NOTHING was "
                             "written." % (label, sorted(kwargs), exc))
        except RuntimeError as exc:
            raise MifOpError("%s export failed: %s" % (label, exc))
    finally:
        scene.frame_start, scene.frame_end = frames_before
        selection_restore(snap)

    # ok:true IS NOT A FILE. Existence alone passes on a leftover from an earlier run, so the mtime
    # has to have moved as well - the same rule render_animation applies per frame.
    if not os.path.isfile(path):
        raise MifOpError("the %s exporter reported success and no file is at %s.%s"
                         % (label, path, _GP_PDF_FLAKY_NOTE if ext == ".pdf" and tuple(bpy.app.version) >= (5, 0, 0)
                            else ""))
    size = os.path.getsize(path)
    mtime = os.path.getmtime(path)
    if mtime < started - 1.0:
        raise MifOpError("the %s exporter reported success but %s was last written %.1fs before "
                         "this call started - it is a LEFTOVER from an earlier run, not this "
                         "export." % (label, path, started - mtime))
    if size == 0:
        raise MifOpError("the %s exporter reported success and wrote a ZERO-BYTE file at %s."
                         % (label, path))

    # NOR IS ok:true EVERYTHING THAT WAS THERE. See _carriage: a glTF of a light and a camera is a valid 132-byte file
    # with neither in it, and this used to answer that with a bare ok.
    not_carried, not_verified, carried_note = _not_carried(ext, label, in_scope, included)
    missing = sum(r["count"] for r in not_carried)
    gltf = ext in (".glb", ".gltf")
    # AN SVG'S STROKES ARE COUNTED IN THE FILE: one <polyline> per stroke from the legacy exporter (3.6, 4.2), one <path>
    # from 5.0+. Zero means the exporter wrote the empty document it writes for nothing visible.
    svg_strokes = None
    if ext == ".svg":
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        svg_strokes = text.count("<polyline") + text.count("<path")
        if svg_strokes == 0:
            raise MifOpError("the %s exporter wrote %s (%d bytes) with no stroke element in it - the Grease Pencil "
                             "objects in scope show nothing at the scene's frame %d."
                             % (label, path, size, bpy.context.scene.frame_current))
    return dict({
        "ok": True,
        "file": path,
        "format": label,
        "operator": dotted,
        "bytes": size,
        # glTF only, like the two parameters: whether lights and cameras went in. None on every other format.
        "includeLights": bool(kwargs.get("export_lights")) if gltf else None,
        "includeCameras": bool(kwargs.get("export_cameras")) if gltf else None,
        # SVG only: the stroke elements read out of the written file.
        "svgStrokes": svg_strokes,
        "objectsNamed": sorted(o.name for o in targets) if targets else None,
        "wholeScene": not (names or selected_only),
        "objectsInScope": len(in_scope),
        "notCarried": not_carried,
        "notCarriedCount": missing,
        "notVerified": not_verified,
        "notCarriedNote": carried_note,
        # None, not [null, null], when no range was asked for: an absent range and a range that
        # failed to apply must not read the same. See applied_range above.
        "frameRange": applied_range,
        "frameRangeRoute": range_route,
        "note": ("FBX is written by export_mesh, which also handles the skeletal case - an armature "
                 "must be in the SELECTION there or the mesh is written frozen in its current pose."),
    }, **replaced_fields(replaced))


# NO "asBackground". It was declared and never read - THIRD dead key this session, after
# render_animation.scene and list_view_layers.scene, all three caught by param_reach rather
# than by reading. The reflex is to add a key that sounds plausible for the operator; an
# accepted parameter that is silently ignored is worse than an absent one.
_IMPORT_KEYS = {"file", "filepath", "path", "collection", "importMaterials", "texturesMode", "texturesDir",
                "importAllMaterials", "usdPreview", "materialBlend", "materialNameCollision", "imagePlane", "svg",
                "bvh"}
# USD IMPORT OPTIONS (2026-10-04): key -> (operator property, kind). Each is checked against THIS build's operator before
# the import runs - a name the build lacks is refused, not dropped.
_USD_IMPORT_OPTIONS = {
    "importMaterials": ("import_materials", bool),
    "texturesMode": ("import_textures_mode", "enum"),
    "texturesDir": ("import_textures_dir", str),
    "importAllMaterials": ("import_all_materials", bool),
    "usdPreview": ("import_usd_preview", bool),
    "materialBlend": ("set_material_blend", bool),
    "materialNameCollision": ("mtl_name_collision_mode", "enum"),
}

# extension -> (candidate operators in order, human label)
#
# glTF AND FBX ARE DELIBERATELY ABSENT: import_mesh already reads both, and it knows things this
# would not - that useCustomNormals is an FBX-importer option with no glTF equivalent, and that
# passing an axis conversion to glTF applies it twice because the spec already fixes +Y up. Two ops
# reading the same format is how they drift apart. Every format has exactly one home here, and the
# refusal names it.
_IMPORTERS = {
    ".obj":  (("wm.obj_import", "import_scene.obj"), "Wavefront OBJ"),
    ".usd":  (("wm.usd_import",), "USD"),
    ".usda": (("wm.usd_import",), "USD ASCII"),
    ".usdc": (("wm.usd_import",), "USD binary"),
    ".usdz": (("wm.usd_import",), "USDZ"),
    ".abc":  (("wm.alembic_import",), "Alembic"),
    ".stl":  (("wm.stl_import", "import_mesh.stl"), "STL"),
    ".ply":  (("wm.ply_import", "import_mesh.ply"), "PLY"),
    # AN IMAGE AS A TEXTURED PLANE AND AN SVG AS CURVES (2026-10-05): image.import_as_mesh_planes is core from 4.2
    # (measured on 4.2.23, 4.4.0, 5.0.1 and 5.2.2: a 200 x 100 PNG arrives as a 2 x 1 plane); import_image.to_plane is
    # the bundled add-on before it. import_curve.svg is the bundled io_curve_svg, enabled for the session if it is off.
    ".png":  (("image.import_as_mesh_planes", "import_image.to_plane"), "image as plane"),
    ".jpg":  (("image.import_as_mesh_planes", "import_image.to_plane"), "image as plane"),
    ".jpeg": (("image.import_as_mesh_planes", "import_image.to_plane"), "image as plane"),
    ".exr":  (("image.import_as_mesh_planes", "import_image.to_plane"), "image as plane"),
    ".tif":  (("image.import_as_mesh_planes", "import_image.to_plane"), "image as plane"),
    ".tiff": (("image.import_as_mesh_planes", "import_image.to_plane"), "image as plane"),
    ".svg":  (("import_curve.svg",), "SVG"),
}
_PLANE_KEYS = {"shader": "shader", "height": "height", "sizeMode": "size_mode", "alignAxis": "align_axis"}


def _plane_options(params, dotted):
    """{operator property: value} for imagePlane, each checked against this build's importer."""
    raw = params.get("imagePlane")
    if raw is None:
        return {}
    if not dotted.endswith(("import_as_mesh_planes", "to_plane")):
        raise MifOpError("imagePlane is for an image file (.png, .jpg, .exr, .tif); this one is read by %s. NOTHING "
                         "was imported." % dotted)
    if not isinstance(raw, dict) or not raw or set(raw) - set(_PLANE_KEYS):
        raise MifOpError("imagePlane is {%s}, got %r. NOTHING was imported." % (", ".join(_PLANE_KEYS), raw))
    mod, _, name = dotted.partition(".")
    props = getattr(getattr(bpy.ops, mod), name).get_rna_type().properties
    out = {}
    for key, attr in _PLANE_KEYS.items():
        if key not in raw:
            continue
        prop = props.get(attr)
        if prop is None:
            raise MifOpError("this Blender's %s has no %s, so imagePlane.%s would be ignored. NOTHING was imported."
                             % (dotted, attr, key))
        if prop.type == "ENUM":
            val = str(raw[key]).upper()
            items = [e.identifier for e in prop.enum_items]
            if val not in items:
                raise MifOpError("imagePlane.%s must be one of %s, got %r. NOTHING was imported."
                                 % (key, ", ".join(items), raw[key]))
            out[attr] = val
        else:
            v = finite_float(raw[key], "imagePlane." + key)
            if not v > 0:
                raise MifOpError("imagePlane.%s must be above 0, got %g. NOTHING was imported." % (key, v))
            out[attr] = v
    if "height" in out and "size_mode" not in out:
        out["size_mode"] = "ABSOLUTE"
    return out


_SVG_AS = ("CURVES", "GREASE_PENCIL")
# AN SVG AS GREASE PENCIL STROKES (2026-10-11): wm.gpencil_import_svg before 4.3, wm.grease_pencil_import_svg after -
# measured on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1, a closed path arrives as one Grease Pencil object with a layer and
# a stroke on frame 1. No add-on to enable, unlike the curve importer.
_SVG_GP_IMPORTERS = ("wm.grease_pencil_import_svg", "wm.gpencil_import_svg")


def _svg_as(params, ext):
    """'CURVES' or 'GREASE_PENCIL' for an .svg (svg.as, default CURVES), None for any other file. Read before anything
    is enabled or imported."""
    raw = params.get("svg")
    if ext != ".svg" or not isinstance(raw, dict) or raw.get("as") is None:
        return "CURVES" if ext == ".svg" else None
    want = str(raw["as"]).upper()
    if want not in _SVG_AS:
        raise MifOpError("svg.as is %s, got %r. NOTHING was imported." % (" or ".join(_SVG_AS), raw["as"]))
    return want


def _svg_scale(params, dotted):
    raw = params.get("svg")
    if raw is None:
        return None
    if dotted not in ("import_curve.svg",) + _SVG_GP_IMPORTERS:
        raise MifOpError("svg is for an .svg file; this one is read by %s. NOTHING was imported." % dotted)
    if not isinstance(raw, dict) or not raw or set(raw) - {"scale", "as"}:
        raise MifOpError("svg is {scale, as}, got %r. NOTHING was imported." % (raw,))
    if raw.get("scale") is None:
        return None
    s = finite_float(raw["scale"], "svg.scale")
    if not s > 0:
        raise MifOpError("svg.scale must be above 0, got %g. NOTHING was imported." % s)
    return s


def _world_bounds(objs):
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    if not pts:
        return None
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    return {"min": [round(v, 5) for v in lo], "max": [round(v, 5) for v in hi],
            "size": [round(hi[i] - lo[i], 5) for i in range(3)]}

_ELSEWHERE = {".fbx": "import_mesh", ".gltf": "import_mesh", ".glb": "import_mesh"}


def resolve_importer(ext):
    """(dotted operator, label) for a file extension. Pure, and separated for the same reason.

    The importers moved with the exporters and at the same versions - OBJ and PLY to wm.*_import at
    4.0, STL at 4.2 - so this carries the same candidate lists. Getting it wrong on the IMPORT side
    is quieter, because a missing importer looks like an unsupported file rather than a missing
    add-on.
    """
    home = _ELSEWHERE.get(ext)
    if home:
        raise MifOpError("%s is read by %s, not here - it knows things this op does not, such as "
                         "useCustomNormals being an FBX option with no glTF equivalent, and that "
                         "an axis conversion applied to glTF is applied twice. NOTHING was "
                         "imported." % (ext, home))
    entry = _IMPORTERS.get(ext)
    if entry is None:
        raise MifOpError("no importer for '%s'. This op reads: %s. FBX and glTF are read by "
                         "import_mesh. NOTHING was imported."
                         % (ext, ", ".join(sorted(_IMPORTERS))))
    candidates, label = entry
    for dotted in candidates:
        if _op_exists(dotted):
            return dotted, label
    raise MifOpError("this Blender (%s) has no %s importer - tried %s. It is usually an add-on and "
                     "may be disabled in Preferences > Add-ons. A missing importer looks like an "
                     "unsupported file, which is why it is named here. NOTHING was imported."
                     % (bpy.app.version_string, label, " and ".join(candidates)))


def _usd_options(params, dotted):
    """{operator property: value} for the USD import options given, checked against this build. Refused on another
    format, and for a property this build's importer lacks."""
    given = [k for k in _USD_IMPORT_OPTIONS if params.get(k) is not None]
    if not given:
        return {}
    if dotted != "wm.usd_import":
        raise MifOpError("%s %s for USD; this file is read by %s. NOTHING was imported."
                         % (", ".join(given), "is" if len(given) == 1 else "are", dotted))
    props = {p.identifier: p for p in bpy.ops.wm.usd_import.get_rna_type().properties}
    out = {}
    for key in given:
        attr, kind = _USD_IMPORT_OPTIONS[key]
        prop = props.get(attr)
        if prop is None:
            raise MifOpError("this Blender's USD importer (%s) has no %s, so %s would be ignored. NOTHING was imported."
                             % (bpy.app.version_string, attr, key))
        if kind is bool:
            out[attr] = take_bool(params, key)
        elif kind == "enum":
            val = str(params[key]).upper()
            items = [e.identifier for e in prop.enum_items]
            if val not in items:
                raise MifOpError("%s must be one of this build's %s, got %r. NOTHING was imported."
                                 % (key, ", ".join(items), params[key]))
            out[attr] = val
        else:
            out[attr] = str(params[key])
    return out


def op_import_scene(params):
    """Read OBJ, USD, Alembic, STL or PLY - the other half of export_scene.

    WHY IT EXISTS AND WHY NOW. export_scene landed first and immediately created the asymmetry it
    was written to remove, one direction over: the addon could WRITE six formats and read three.
    Leaving that is the same shape this session kept finding elsewhere - a family that can only go
    one way - so it is closed rather than filed.

    THE POSTCONDITION IS WHAT ARRIVED, taken by set difference, because every import operator
    returns {'FINISHED'} and none of them returns the objects it made. A file that parses and holds
    nothing importable - an animation-only USD, a camera-only export, an Alembic with no geometry -
    reports success and adds nothing, and that is refused rather than returned as ok:true. Copied
    from import_mesh, which learned it first.

    params:
      file / filepath / path (str)   required. The FORMAT COMES FROM THE EXTENSION.
      collection (str)               link what arrives into this collection instead of the scene
                                     root. Resolved BEFORE the import, so a bad name cannot leave
                                     objects already in the scene.
      BVH (1.3): a .bvh makes an armature with its action; bvh {rotateMode, scale, frameStart, useFpsScale,
      updateSceneFps, updateSceneDuration, cyclic} (io_anim_bvh, enabled for the session when off).
      USD only (2026-10-04): importMaterials, texturesMode IMPORT_NONE | IMPORT_PACK | IMPORT_COPY, texturesDir,
      importAllMaterials, usdPreview, materialBlend, materialNameCollision MAKE_UNIQUE | REFERENCE_EXISTING - each
      checked against this build's usd_import; the materials and images that arrived are counted.
    """
    reject_unknown(params, _IMPORT_KEYS, "import_scene")
    raw = take(params, "file", "filepath", "path", required=True, kind=str)
    path = os.path.abspath(bpy.path.abspath(str(raw)))
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bvh":
        return _import_bvh(params, path)
    if params.get("bvh") is not None:
        raise MifOpError("bvh holds a .bvh file's options, and this file is %s. NOTHING was imported." % (ext or "unnamed"))
    if not ext:
        raise MifOpError("'%s' has no file extension, and the format is taken from it. Give one "
                         "of: %s. NOTHING was imported." % (raw, ", ".join(sorted(_IMPORTERS))))
    addon_enabled = None
    svg_as = _svg_as(params, ext)
    # THE BUNDLED IMPORTERS, for this session only (not saved in the preferences): io_curve_svg for an SVG, and before
    # 4.2 - where image.import_as_mesh_planes became core - io_import_images_as_planes for an image. The image one was
    # named as tried and never enabled, so a factory-startup 3.6.23 refused every image (found 2026-10-10).
    bundled = None
    if ext == ".svg" and svg_as == "CURVES" and not _op_exists("import_curve.svg"):
        bundled = "io_curve_svg"
    elif (_IMPORTERS.get(ext, ((),))[0][:1] == ("image.import_as_mesh_planes",)
          and not any(_op_exists(d) for d in _IMPORTERS[ext][0])):
        bundled = "io_import_images_as_planes"
    if bundled is not None:
        import addon_utils   # noqa: PLC0415
        try:
            addon_utils.enable(bundled, default_set=False, persistent=False)
            addon_enabled = bundled
        except Exception:                                       # noqa: BLE001 - resolve_importer names the miss
            pass
    dotted, label = resolve_importer(ext)
    if svg_as == "GREASE_PENCIL":
        dotted = next((d for d in _SVG_GP_IMPORTERS if _op_exists(d)), None)
        if dotted is None:
            raise MifOpError("this Blender (%s) has no Grease Pencil SVG importer - tried %s. NOTHING was imported."
                             % (bpy.app.version_string, " and ".join(_SVG_GP_IMPORTERS)))
        label = "SVG as Grease Pencil"

    if not os.path.isfile(path):
        raise MifOpError("no such file: %s. NOTHING was imported." % path)
    size = os.path.getsize(path)
    if size <= 0:
        raise MifOpError("%s is empty (0 bytes) - there is nothing to import. NOTHING was "
                         "imported." % path)

    # RESOLVED FIRST, for the reason ops_create learned the hard way the same day: a collection
    # looked up after the work is done leaves the work behind when the lookup fails.
    coll_name = take(params, "collection", kind=str)
    coll = None
    if coll_name:
        coll = bpy.data.collections.get(coll_name)
        if coll is None:
            known = sorted(c.name for c in bpy.data.collections)[:25]
            raise MifOpError("no collection named '%s'. Present: %s. Make one with "
                             "create_collection. NOTHING was imported."
                             % (coll_name, ", ".join(known) if known else "<none>"))

    usd_kw = _usd_options(params, dotted)
    plane_kw = _plane_options(params, dotted)
    svg_scale = _svg_scale(params, dotted)
    before = set(bpy.data.objects)
    mats_before, imgs_before = set(bpy.data.materials), set(bpy.data.images)
    mod, _, opname = dotted.partition(".")
    try:
        if label == "image as plane":
            getattr(getattr(bpy.ops, mod), opname)(files=[{"name": os.path.basename(path)}],
                                                   directory=os.path.dirname(path), **plane_kw)
        elif label == "SVG as Grease Pencil":
            getattr(getattr(bpy.ops, mod), opname)(filepath=path, directory=os.path.dirname(path),
                                                   files=[{"name": os.path.basename(path)}])
        else:
            getattr(getattr(bpy.ops, mod), opname)(filepath=path, **usd_kw)
    except RuntimeError as exc:
        raise MifOpError("%s import failed: %s" % (label, exc))
    except TypeError as exc:
        raise MifOpError("the %s importer on this build does not take a plain filepath (%s). This "
                         "is version drift in the operator's own signature." % (label, exc))
    bpy.context.view_layer.update()
    created = [o for o in bpy.data.objects if o not in before]

    if not created:
        raise MifOpError("the %s importer reported success and produced NO objects from %s (%d "
                         "bytes). The file parsed and held nothing importable - an animation-only "
                         "or camera-only export will do exactly this. NOTHING is in the scene from "
                         "it." % (label, path, size))

    moved = []
    if coll is not None:
        # The importer links into the scene's active collection; move rather than add, or the
        # objects end up in two places and a later unlink from one looks like it did nothing.
        for obj in created:
            for c in list(bpy.data.collections) + [bpy.context.scene.collection]:
                if obj.name in c.objects and c is not coll:
                    c.objects.unlink(obj)
            if obj.name not in coll.objects:
                coll.objects.link(obj)
            moved.append(obj.name)
        missing = [o.name for o in created if o.name not in coll.objects]
        if missing:
            raise MifOpError("imported %d object(s) and %d are not in '%s' afterwards: %s"
                             % (len(created), len(missing), coll.name, ", ".join(missing[:8])))

    if svg_scale is not None:
        for obj in created:
            obj.scale = obj.scale * svg_scale
            obj.location = obj.location * svg_scale
        bpy.context.view_layer.update()
    extra = {}
    if label == "image as plane":
        plane = next((o for o in created if o.type == "MESH"), None)
        dims = [round(v, 5) for v in plane.dimensions] if plane is not None else None
        flat = sorted(dims, reverse=True)[:2] if dims else None
        extra["plane"] = {"object": plane.name if plane else None, "dimensions": dims,
                          "aspect": round(flat[0] / flat[1], 5) if flat and flat[1] else None,
                          "options": plane_kw or None,
                          "material": plane.active_material.name if plane and plane.active_material else None,
                          "addonEnabled": addon_enabled}
    if label == "SVG":
        curves = [o for o in created if o.type == "CURVE"]
        extra["svg"] = {"curveObjects": len(curves), "splineCount": sum(len(o.data.splines) for o in curves),
                        "bounds": _world_bounds(curves), "scale": svg_scale, "addonEnabled": addon_enabled,
                        "as": "CURVES"}
    if label == "SVG as Grease Pencil":
        gps = [o for o in created if o.type in ("GPENCIL", "GREASEPENCIL")]
        strokes = 0
        for o in gps:
            for layer in o.data.layers:
                for fr in layer.frames:
                    strokes += len(fr.drawing.strokes) if o.type == "GREASEPENCIL" else len(fr.strokes)
        extra["svg"] = {"greasePencilObjects": len(gps), "generation": gps[0].type if gps else None,
                        "layers": sum(len(o.data.layers) for o in gps), "strokes": strokes,
                        "bounds": _world_bounds(gps), "scale": svg_scale, "as": "GREASE_PENCIL"}
    by_type = {}
    for obj in created:
        by_type[obj.type] = by_type.get(obj.type, 0) + 1
    return {
        **extra,
        "ok": True,
        "file": path,
        "format": label,
        "operator": dotted,
        "bytes": size,
        "created": sorted(o.name for o in created),
        "createdCount": len(created),
        "createdByType": by_type,
        "collection": coll.name if coll is not None else None,
        "movedIntoCollection": sorted(moved) if moved else None,
        "usdOptions": usd_kw or None,
        "materialsCreated": sorted(m.name for m in bpy.data.materials if m not in mats_before),
        "imagesCreated": sorted(i.name for i in bpy.data.images if i not in imgs_before),
        "note": ("no MESH arrived - %s. That is a legitimate import and rarely what somebody "
                 "expects from one." % ", ".join("%d %s" % (v, k) for k, v in sorted(by_type.items())))
        if "MESH" not in by_type else None,
    }


OPS = {
    "export_scene": op_export_scene,
    "import_scene": op_import_scene,
}
