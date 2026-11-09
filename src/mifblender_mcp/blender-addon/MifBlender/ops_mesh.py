"""MifBlender ops: FBX import/export and bmesh edge bevelling.

=============================================================================
FBX AXIS + SCALE -- THE ONE THING THAT MUST NOT BE GUESSED
=============================================================================
Unreal writes, and Unreal reads back, this axis system:

    Up    = Z, sign +1
    Front = -eParityOdd, i.e. axis Y, sign -1
    Coord = X, sign +1   (right-handed)
    System unit = centimetres

  VERIFIED in engine source, UE 5.3.2:
    Engine/Source/Editor/UnrealEd/Private/Fbx/FbxMainExport.cpp:268-276
        FbxAxisSystem UnrealZUp(eZAxis, -eParityOdd, eRightHanded);
        Scene->GetGlobalSettings().SetAxisSystem(UnrealZUp);
        Scene->GetGlobalSettings().SetSystemUnit(FbxSystemUnit::cm);
    Engine/Source/Editor/UnrealEd/Private/Fbx/FbxMainImport.cpp:1500-1515
        builds the IDENTICAL axis system and only calls ConvertScene()
        `if (SourceSetup != UnrealImportAxis)`. Match it and nothing rotates.
    FbxMainImport.cpp:1540-1542 -- unit conversion only runs if the file's
        system unit is not already cm.

Blender's operator DEFAULTS DO NOT MATCH. Defaults are axis_up='Y',
axis_forward='-Z' (the Maya Y-up convention). Exporting with the defaults writes
UpAxis=1 and UE rotates the mesh on import.

  VERIFIED empirically on Blender 4.4.0 by exporting the same object twice and
  parsing GlobalSettings straight out of the binary FBX:

    axis_up='Z', axis_forward='Y'  ->  UpAxis 2 (+1)  FrontAxis 1 (-1)  CoordAxis 0 (+1)
    operator defaults              ->  UpAxis 1 (+1)  FrontAxis 2 (+1)  CoordAxis 0 (+1)

  The first row is bit-for-bit UE's system. ('Z','Y') is also the row Blender's
  own table marks `# Blender system!`
  (4.4/scripts/addons_core/io_scene_fbx/fbx_utils.py:126), so this is the
  identity mapping in both directions -- not a correction, an absence of one.

SCALE. With apply_unit_scale=True and apply_scale_options='FBX_SCALE_NONE'
(both defaults, both pinned here anyway) Blender bakes x100 into the geometry
and writes UnitScaleFactor=1.0, i.e. centimetre magnitudes in a centimetre
file. VERIFIED by reading UnitScaleFactor=1.0 out of the exported binary while
a 10.0 Blender-unit object measured 1000 in the file. So:

        1 Blender unit == 100 Unreal units.   A 1000 uu road is 10.0 BU.

IMPORT takes NO axis arguments on purpose. use_manual_orientation defaults to
False, which makes the importer read FrontAxis/UpAxis/CoordAxis out of the file
and reverse-map them (io_scene_fbx/import_fbx.py:3136-3145). Passing axis args
would only be able to make it wrong.

  VERIFIED 2026-08-27, three of the four legs, on Blender 4.4.0 AND 5.0.1:

      UE export_asset  ->  PH_HumanGizmoSitLowPoly.fbx, 164,880 bytes
      import_mesh      ->  802 vertices
      extrude_skirt    ->  995 vertices  (boundaryOnly, depth 2.0)
      export_mesh      ->  91,420 bytes on 4.4 / 91,484 on 5.0

  Both engines produce the SAME vertex counts, so the geometry path is not
  version-sensitive between 4.4 and 5.0.

  STILL NOT VERIFIED: the last leg, FBX back INTO Unreal. import_asset persists
  to disk, so the safety gate refuses it in scratch mode - correctly, and it is
  not something to work around. Until someone runs that leg in full mode, the
  axis/scale claims above rest on the byte-level header read plus three of four
  legs, not on a mesh having made the whole loop and come back the right way up.

  NOT VERIFIED: mesh_smooth_type. 'FACE' writes smoothing groups AND normals,
  which is strictly more information than the 'OFF' default -- but which of the
  two Unreal actually consumes depends on the static-mesh import options
  ("Normal Import Method"), and I did not confirm that. If normals come back
  wrong, this is the first knob to turn.
=============================================================================
"""

from __future__ import annotations

import array
import hashlib
import heapq
import json
import math
import mmap
import os
import random
import re

import bmesh
import bpy
from mathutils import Matrix, Vector

from .ops_common import (remove_objects, MifOpError, UU_PER_BU, simulation_guard, assign_id_name, axis_index, bone_segment, check_output_path,
                         children_by_parent,
                         edit_mode_stale, file_before, finite_float, finite_floats,
                         get_object, mesh_counts, object_info, reject_unknown, replaced_fields, require_editable,
                         rnd, rotation_is_identity, select_only, selection_restore, selection_snapshot,
                         shared_data_note, take, take_bool, take_float, take_int,
                         material_slot_names, instances_note, evaluated_mesh_bounds, drawn_instances,
                         _FLT_MIN, _UV_OPEN_RADIUS, _f32, _open_collapsed_uvs, drop_doubled_and_fin_faces, open_collapsed_uvs_of,
                         triangulate_ngons)

# Pinned, not defaulted. Anything that changes the geometry, the axes or the
# units of the written file lives in this dict so there is exactly one place to
# audit. axis_up / axis_forward are the two that are NOT operator defaults.
FBX_EXPORT_ARGS = {
    "axis_up": "Z",                          # default 'Y'  -- MUST override
    "axis_forward": "Y",                     # default '-Z' -- MUST override
    "apply_unit_scale": True,
    "apply_scale_options": "FBX_SCALE_NONE",
    "global_scale": 1.0,
    "use_space_transform": True,
    "bake_space_transform": False,           # flagged experimental upstream; leave OFF
    "object_types": {"MESH"},                # default drags in EMPTY/CAMERA/LIGHT/ARMATURE
    # add_leaf_bones DELIBERATELY DISAGREES WITH BLENDER, whose default is True. A leaf bone is a
    # synthetic tip Blender appends to every chain end so the FBX records bone LENGTH; Unreal has no
    # concept of it and imports each one as a real extra bone named <parent>_end. On a hand that is
    # five junk bones, and they then appear in every retarget chain and every anim asset built from
    # that skeleton. Off unless the caller asks.
    "add_leaf_bones": False,
    "use_armature_deform_only": False,
    "primary_bone_axis": "Y",
    "secondary_bone_axis": "X",
    "use_mesh_modifiers": True,
    "mesh_smooth_type": "FACE",              # see the NOT VERIFIED note above
    # TANGENT SPACE IS WRITTEN, against the exporter's own default (2026-10-05). A mesh with zero-area UV faces came into
    # Unreal 5.8.2 warning "nearly zero tangents / bi-normals (tolerance 1E-4)"; with tangents in the file the warning
    # went, with the engine's recompute on or off. export_mesh's tangentSpace says, per mesh, when the exporter wrote
    # none anyway (no UV map; a face of more than 4 corners without use_triangles).
    "use_tspace": True,
    "use_triangles": False,
    "colors_type": "SRGB",
    "use_custom_props": False,
    "bake_anim": False,                      # a static mesh has no animation to write
    "path_mode": "AUTO",
}

FBX_IMPORT_ARGS = {
    "use_custom_normals": True,
    "use_anim": False,
    "use_image_search": False,               # do not crawl the filesystem for textures
    # NO axis args, NO global_scale. The importer reads them from the file.
}

# glTF ADDED 2026-08-31, and the reason it is safe is the reason OBJ still is not.
#
# The old refusal said "FBX is the only format whose axis and unit round-trip is verified". That was
# a claim about EVIDENCE, so it was answered with evidence rather than argued with: a 1 x 2 x 3 box -
# asymmetric on every axis, so any swap shows as a permutation - exported to GLB and imported back on
# Blender 5.0.1 came back 1 x 2 x 3 exactly, name intact.
#
# It round-trips because the glTF SPEC fixes the convention: +Y up, right-handed, metres, and
# Blender's importer applies that conversion itself. There is nothing to guess. FBX by contrast
# carries its axis and unit metadata IN THE FILE, which is why FBX_IMPORT_ARGS passes no axis
# arguments and lets the importer read them. OBJ remains refused because it fixes nothing and
# declares nothing: UE's exporter swaps Y/Z, de-indexes, and writes no normals, and the file cannot
# tell you it did.
#
# WHAT GLTF DOES CHANGE, and callers are told rather than left to discover it: it has no concept of
# a shared vertex with split normals, so vertices are de-indexed per corner. The 8-vertex cube in
# that probe came back with 24. Nothing is lost - the shape, dimensions and normals are identical -
# but a caller comparing vertexCount across a round trip will see it, and a surprise like that reads
# as corruption.
# PER-VERB, and that split is not cosmetic - it is a bug this file already had for two commits.
#
# _check_format is called by BOTH import_mesh and export_mesh. Widening one shared _SUPPORTED tuple
# to add glTF IMPORT silently widened export too, and export_mesh does not dispatch on extension -
# it always calls export_scene.fbx. So `export_mesh {file: "x.glb"}` answered ok:true and wrote a
# file beginning "Kaydara FBX Binary". Silent, plausible, and exactly the class this bridge exists
# to refuse: a caller gets a .glb that no glTF loader will open and nothing said a word.
#
# Caught by asking what else touches _check_format, not by a tool. Two verbs sharing a capability
# list is fine only while their capabilities are the same, and the moment they diverge the shared
# list is a liability rather than a convenience.
_IMPORT_FORMATS = (".fbx", ".gltf", ".glb")
_EXPORT_FORMATS = (".fbx",)
_GLTF = (".gltf", ".glb")


def _check_format(path, verb, allowed=None):
    ext = os.path.splitext(path)[1].lower()
    allowed = allowed or _IMPORT_FORMATS
    if ext in _GLTF and allowed is _EXPORT_FORMATS:
        # audit-ok: export_mesh, import_mesh - this branch is gated on `allowed is _EXPORT_FORMATS`,
        # so export_mesh is the ONLY caller that can reach it and naming it is correct. import_mesh
        # is named on purpose too: the useful half of the sentence is which verb DOES take glTF.
        raise MifOpError(
            "export_mesh writes FBX only (got '%s'). import_mesh DOES take glTF and GLB, so the "
            "asymmetry is deliberate rather than an oversight: the FBX export path carries the "
            "armature and object_types handling that keeps a rigged mesh from being written frozen "
            "in whatever pose it happened to be in, and none of that transfers to the glTF "
            "exporter unexamined. Export .fbx, or use run_python and own the result." % ext)
    if ext not in allowed:
        # THE LIST MUST BE THIS CALLER'S LIST. This branch used to recite "FBX and glTF/GLB"
        # whatever it was passed, which is the import set - so `export_mesh {file:"x.obj"}` was
        # told glTF and GLB were available, and the caller who believed it got the branch ABOVE
        # saying "export_mesh writes FBX only". One helper, two refusals, contradicting each other,
        # and the verb name on both was right. That is what made it survive: a wrong refusal gets
        # reported, a right refusal followed by a wrong sentence gets acted on.
        #
        # Derived from `allowed` rather than restated, so widening either tuple cannot desynchronise
        # the message again - which is the same failure the comment at the top of this block already
        # records once, when a shared _SUPPORTED let export_mesh write FBX bytes into a .glb.
        takes_gltf = any(g in allowed for g in _GLTF)
        # audit-ok: import_mesh - the verb this refusal is FOR arrives as `verb` and is formatted
        # in, never hard-coded. import_mesh appears only in the not-takes_gltf arm, which is
        # reachable only by export_mesh, and it is there to say where glTF support does live.
        raise MifOpError(
            "%s: supported formats are %s (got '%s'). %s OBJ in particular does not, which is why "
            "it stays refused: UE's OBJ exporter swaps Y/Z, de-indexes to 3 verts per triangle and "
            "writes no normals, and the file cannot tell you so. Use run_python if you need "
            "another format and accept that the orientation is on you."
            % (verb,
               "FBX and glTF/GLB" if takes_gltf else "FBX",
               ext or "<no extension>",
               "Those two round-trip axis and unit verifiably - glTF because its spec FIXES the "
               "convention (+Y up, metres) and FBX because it carries its own metadata."
               if takes_gltf else
               "FBX round-trips axis and unit verifiably because it carries its own metadata. "
               "import_mesh additionally takes glTF and GLB; this verb does not, and the reason "
               "is on the glTF refusal above."))
    return ext


def _resolve_out_path(path):
    path = bpy.path.abspath(str(path))
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as exc:
            raise MifOpError("cannot create output directory %s: %s" % (parent, exc))
    return path


# ---------------------------------------------------------------------------
# import_mesh
# ---------------------------------------------------------------------------

def clear_targets(scene, scope):
    """(objects a clear deletes, names it keeps) for import_mesh's clearScene - shared with cap.py's count, so the cap
    counts exactly what the op deletes. "file": every object in the file. "scene": the scene's objects that no other
    scene uses; one linked into another scene too is kept and named."""
    if scope == "file":
        return list(bpy.data.objects), []
    doomed, kept = [], []
    for o in scene.objects:
        if all(s == scene for s in o.users_scene):
            doomed.append(o)
        else:
            kept.append(o.name)
    return doomed, kept


def op_import_mesh(params):
    """Import an .fbx, .gltf or .glb file and report the objects it made; clearScene (default false) first deletes the
    scene's objects (clearScope scene) or the file's (file), only after every parameter has been checked."""
    reject_unknown(params, {"file", "filepath", "path", "clearScene", "clear_scene", "clearScope",
                            "useCustomNormals", "importAnimation", "rename"}, "import_mesh")
    raw = take(params, "file", "filepath", "path", required=True, kind=str)
    path = os.path.abspath(bpy.path.abspath(raw))
    _check_format(path, "import_mesh")
    if not os.path.isfile(path):
        raise MifOpError("no such file: %s" % path)
    size = os.path.getsize(path)
    if size <= 0:
        raise MifOpError("file %s is empty (0 bytes) -- nothing to import" % path)

    # PARSED HERE, DONE LAST. The clear used to run on this line, before every check below it - so
    # {clearScene: true, importAnimation: "flase"} deleted every object in the file and THEN refused the
    # typo, and the glTF useCustomNormals / importAnimation:false refusals came after it the same way
    # (measured on 5.0.1, 2026-09-21: six objects before the refused call, none after). Now nothing is
    # deleted until every parameter has been read and the import is the next line.
    clear = take_bool(params, "clearScene", "clear_scene", default=False)
    # WHAT clearScene CLEARS (decided 2026-10-04): "scene" (the default) deletes the objects that belong to the
    # scene being imported into and to no other; "file" is the old file-wide clear, every scene's objects. File-wide
    # was the only behaviour, so an import into an empty scene of a multi-scene file deleted the other scenes' work -
    # a short film's import was refused by the change cap for exactly that on 2026-10-02.
    scope = str(take(params, "clearScope", default="scene", kind=str)).strip().lower()
    if scope not in ("scene", "file"):
        raise MifOpError("clearScope must be 'scene' (the objects of the scene being imported into) or 'file' (every "
                         "scene's objects), got %r. NOTHING was changed." % params.get("clearScope"))
    rename = take(params, "rename", default=None, kind=str)

    ext = os.path.splitext(path)[1].lower()
    is_gltf = ext in _GLTF

    # useCustomNormals is an FBX-IMPORTER ARGUMENT and glTF's importer has no equivalent - normals
    # come from the file or are generated by the spec's rules. Refused rather than ignored: silently
    # accepting a parameter that cannot do anything is the class this bridge refuses on principle,
    # and a caller who set it would otherwise believe it took effect.
    if is_gltf and "useCustomNormals" in params:
        raise MifOpError(
            "useCustomNormals is an FBX importer option and has no glTF equivalent - glTF "
            "normals come from the file, or are generated per the spec when absent. Drop the "
            "parameter for a %s import rather than assuming it did nothing." % ext)

    # THE SAME VERB BEHAVED OPPOSITELY BY EXTENSION, and said nothing either way. FBX_IMPORT_ARGS
    # pinned use_anim False with no parameter reaching it, so every FBX animation was dropped; the
    # glTF importer has NO animation option at all - measured on 3.6.23, 4.2.17, 4.4.0 and 5.0.1,
    # where the only animation properties on the operators are fbx's use_anim and anim_offset and
    # gltf has none - so it always imports them. An animated cube written to both and imported back
    # on 4.4 came in with no action from FBX and with one from glTF.
    #
    # importAnimation reaches use_anim on FBX. On glTF it can only be refused when false, for
    # useCustomNormals' reason one paragraph up: accepting a parameter that cannot do anything is
    # the class this bridge refuses on principle.
    want_anim = params.get("importAnimation")
    # Read BEFORE the format test - see audit_blender_shortcircuit_take. A malformed
    # importAnimation was refused on a glTF and accepted on an FBX, for the same call.
    import_animation = take_bool(params, "importAnimation", default=True)
    if is_gltf and want_anim is not None and not import_animation:
        raise MifOpError(
            "importAnimation:false cannot be honoured for %s - Blender's glTF importer has no "
            "animation option on any supported build (3.6, 4.2, 4.4, 5.0) and always imports "
            "them. Import, then delete the action if it is unwanted. Refused rather than "
            "accepted and ignored." % ext)

    # The import operators return {'FINISHED'}, never the objects they made, so
    # capture by set difference. (bpy.ops.import_scene.fbx has no other handle, and
    # import_scene.gltf has none either.)
    args = dict(FBX_IMPORT_ARGS)
    if not is_gltf:
        if "useCustomNormals" in params:
            args["use_custom_normals"] = take_bool(params, "useCustomNormals", default=True)
        # DEFAULT STAYS FALSE so nothing silently changes for callers who have been importing static
        # meshes; what changed is that the response now SAYS an animation was left behind.
        args["use_anim"] = take_bool(params, "importAnimation", default=False)

    # THE CLEAR, NOW THAT NOTHING ABOVE CAN REFUSE. FILE-WIDE, as it always was - bpy.data.objects is every
    # scene's objects, not the active scene's - and the response now says what went, and which of it was not
    # in the scene being imported into, because until now a caller could not tell.
    #
    # THROUGH ops_scene._delete, THE RIGID-BODY-SAFE PATH clear_scene and delete_object use (2026-09-28, the 2026-09-21
    # pack-rule review). This removed each object with a bare bpy.data.objects.remove, which frees a rigid body or joint
    # without taking it out of Bullet's world - the use after free that crashed 5.2.1 and 4.2.17 through delete_object
    # (docs/22 B51) - and kept a baked world's cache after every body in it was gone: measured on 5.2.1 through this
    # op, five baked hinge rigs cleared and the world's point cache still read isBaked true. purge=False: the clear has
    # never purged, and does not start now.
    cleared, cleared_elsewhere, shared_kept = [], [], []
    freed, not_freed, bake_note = [], [], None
    if clear:
        from .ops_scene import _delete, _freed_note    # runtime: the op modules import ops_common, not each other
        scene = bpy.context.scene
        here = {o.name for o in scene.objects}
        doomed, shared_kept = clear_targets(scene, scope)
        cleared, _purged, freed, not_freed, _joints = _delete(doomed, False)
        cleared_elsewhere = [n for n in cleared if n not in here]
        bake_note = _freed_note(freed, not_freed)

    before = set(bpy.data.objects)
    # ACTIONS TOO, because whether an animation arrived is the thing that was going unreported and
    # it cannot be read off the request - use_anim True on a file with no animation creates nothing.
    actions_before = set(bpy.data.actions)
    if is_gltf:
        # NO axis or scale arguments, deliberately, and for a different reason than FBX's: the glTF
        # spec fixes +Y up / metres and Blender's importer converts from it. Passing a conversion
        # here would apply it twice.
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        bpy.ops.import_scene.fbx(filepath=path, **args)
    bpy.context.view_layer.update()
    created = [o for o in bpy.data.objects if o not in before]

    if not created:
        raise MifOpError(
            "import produced NO objects from %s (%d bytes). The file parsed but held "
            "nothing importable -- check it is a mesh %s and not, say, an animation-only "
            "or camera-only export.%s" % (path, size, "glTF" if is_gltf else "FBX",
                                          (" clearScene had ALREADY deleted %d object(s) before the import ran: %s."
                                           % (len(cleared), ", ".join(cleared[:20]))) if cleared else ""))

    if rename:
        if len(created) != 1:
            # AFTER THE IMPORT, so the objects are in the file and the sentence says so rather than reading as a no-op.
            raise MifOpError("'rename' needs exactly one imported object, got %d: %s. They ARE imported - "
                             "the import ran before the count could be known%s."
                             % (len(created), ", ".join(o.name for o in created),
                                (", and clearScene deleted %d object(s) first" % len(cleared)) if cleared else ""))
        # Naming an imported object over one already in the scene renames THAT object on 3.6
        # when the import's own name sorts higher - ledger B12. An import is exactly where it
        # would bite: the incoming name comes from a file, so the sort is not the caller's to
        # predict.
        assign_id_name(created[0], str(rename), bpy.data.objects)

    new_actions = [a.name for a in bpy.data.actions if a not in actions_before]
    animated = [o.name for o in created
                if getattr(o, "animation_data", None) and o.animation_data.action]

    warnings = []
    if not is_gltf and not args.get("use_anim") and not new_actions:
        warnings.append(
            "animation was NOT imported - import_mesh passes use_anim:false to the FBX importer "
            "by default, so any animation in this file was dropped. Pass importAnimation:true to "
            "keep it. glTF imports animation unconditionally, so the same call behaves differently "
            "by extension; that is now said out loud rather than left to be discovered.")
    kids = children_by_parent()          # one pass for every object_info below - see ops_common.children_by_parent
    non_identity = [o.name for o in created if not object_info(o, children=kids)["isIdentityTransform"]]
    if non_identity:
        warnings.append(
            "imported object(s) %s do NOT have an identity transform. Do not call "
            "transform_apply to 'fix' it -- that bakes the round trip into the mesh and "
            "shears anything that tiles. Leave the transform alone."
            % ", ".join(non_identity))
    if is_gltf:
        # Not a defect and not optional to mention. glTF has no shared-vertex-with-split-normals
        # concept, so a cube's 8 vertices come back as 24 - measured, not assumed. Shape, dimensions
        # and normals are identical; a caller comparing vertexCount across a round trip is not
        # looking at corruption.
        warnings.append(
            "glTF de-indexes vertices per corner, so vertexCount is higher than the source "
            "mesh had (a cube's 8 becomes 24). Geometry, dimensions and normals are unchanged - "
            "compare dimensions rather than vertex counts across a glTF round trip.")
    if len(created) > 1:
        warnings.append(
            "%d objects were imported. If you expected one mesh, the source export "
            "probably included LODs or collision -- re-export with levelOfDetail:false "
            "and collision:false." % len(created))

    return {
        "file": path,
        "fileSizeBytes": size,
        "importedCount": len(created),
        "imported": [object_info(o, children=kids) for o in created],
        # MEASURED, NOT REQUESTED. use_anim:true on a file with no animation creates
        # nothing, so asking what was requested answers a different question.
        "animationImported": bool(new_actions),
        "actionsCreated": new_actions,
        "objectsWithAction": animated,
        # WHAT clearScene DELETED, which the response never said. It is FILE-WIDE, so an object in another scene
        # goes too; clearedFromOtherScenes names those, measured 2026-09-21 on 5.0.1 (a second scene's object was
        # deleted by an import into the first).
        "clearScene": clear,
        "clearScope": scope if clear else None,
        # An object linked into this scene AND another is not deleted by a scene clear - deleting it would take it
        # out of the other scene too. It is still here, in this scene, and named.
        "sharedKept": shared_kept[:50] or None,
        "clearedCount": len(cleared),
        "clearedObjects": cleared[:50] or None,
        "clearedFromOtherScenes": cleared_elsewhere[:50] or None,
        # THE RIGID BODY BAKES THE CLEAR LEFT STALE, freed - or not - and said, as clear_scene says them.
        "freedRigidBodyBake": freed or None, "rigidBodyBakeNotFreed": not_freed or None,
        "rigidBodyBakeNote": bake_note,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# export_mesh
# ---------------------------------------------------------------------------

_EXPORT_OVERRIDES = {
    "meshSmoothType": ("mesh_smooth_type", str),
    "useTriangles": ("use_triangles", bool),
    "useMeshModifiers": ("use_mesh_modifiers", bool),
    "useTspace": ("use_tspace", bool),
    # SKELETAL EXPORT, added 2026-08-30. See op_export_mesh's own note for why the defaults are
    # what they are - two of them deliberately disagree with Blender's.
    "addLeafBones": ("add_leaf_bones", bool),
    "armatureDeformOnly": ("use_armature_deform_only", bool),
    "primaryBoneAxis": ("primary_bone_axis", str),
    "secondaryBoneAxis": ("secondary_bone_axis", str),
    "bakeAnim": ("bake_anim", bool),
    # ANIMATION OPTIONS (2026-10-04): a rig's clips written beside or apart from its mesh
    "bakeAnimAllActions": ("bake_anim_use_all_actions", bool),
    "bakeAnimNla": ("bake_anim_use_nla_strips", bool),
    "bakeAnimSimplify": ("bake_anim_simplify_factor", float),
    "bakeAnimForceKeying": ("bake_anim_force_startend_keying", bool),
}


# ---------------------------------------------------------------------------
# export_mesh splitShapeCorners: a keyed mesh's points split at its hard corners
# ---------------------------------------------------------------------------
# WHY (1.3, measured 2026-10-19). A shape key's normals never reach Unreal from an FBX, whatever the file says: every
# route reads a shape's POSITIONS only - the 5.3 importer (FbxSkeletalMeshImport.cpp :1510, and LODUtilities.cpp 5.3
# :3209 "we can't get imported morphtarget normal from FBX"), 5.8's Interchange FBX SDK parser (FbxMesh.cpp 5.8 :518 "For
# the shape we just need the vertex positions") and its ufbx parser (UfbxMesh.cpp 5.8 :373 "#ufbx_todo: normals").
# Unreal COMPUTES the morph's normals itself, and per POINT: LODUtilities.cpp 5.8 :3258-3263 (5.3 the same) maps each
# point to the LAST face corner that uses it (a TMap Add per corner), so every render vertex at a point gets that one
# corner's normal change. On a hard corner - a flat-shaded cube's, where three faces meet with three normals - all three
# get the change of whichever face is listed last; Blender lists a cube's top and bottom last, so a key that turns the
# cube about Z arrived with no normal change at all (tools/test_lane13_x2b.py MD101, 5.3.2 and 5.8.3; the file's own
# shape normals are zero too - export_fbx_bin.py writes np.zeros - but nothing reads them). Splitting the points at hard
# corners gives each face corner a point of its own, so Unreal's per-point normal is that face's. Unreal splits render
# vertices at hard corners anyway, so its vertex count does not grow; the FBX's point count does. The file's corner
# normals are compared before and after the split, and a mesh whose normals would change is exported unsplit and named.

def _hard_edges(mesh, normals):
    """Indices of the edges between two faces whose corner normals differ at either end - the corners the exporter
    writes with two normals."""
    edge_faces = {}
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            edge_faces.setdefault(mesh.loops[li].edge_index, []).append((poly.index, li))
    corner_of = {}
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            corner_of[(poly.index, mesh.loops[li].vertex_index)] = li
    hard = []
    for e, uses in edge_faces.items():
        if len(uses) != 2:
            continue
        fa, fb = uses[0][0], uses[1][0]
        for v in mesh.edges[e].vertices:
            na, nb = normals[corner_of[(fa, v)]], normals[corner_of[(fb, v)]]
            if na.dot(nb) < 0.99999:
                hard.append(e)
                break
    return hard


def _split_shape_corners(obj):
    """A copy of obj's mesh with its points split at hard corners, shape keys carried -> (copy, report), or
    (None, report) when nothing needs splitting or the split would change a written normal."""
    from .ops_kit import _corner_normals        # noqa: PLC0415 - the one 3.6/4.1 corner-normal reader
    mesh = obj.data
    before = _corner_normals(mesh)
    hard = _hard_edges(mesh, before)
    if not hard:
        return None, None
    copy = mesh.copy()
    bm = bmesh.new()
    try:
        bm.from_mesh(copy)
        bm.edges.ensure_lookup_table()
        bmesh.ops.split_edges(bm, edges=[bm.edges[i] for i in hard])
        bm.to_mesh(copy)
    finally:
        bm.free()
    if mesh.has_custom_normals:
        if hasattr(copy, "use_auto_smooth"):
            copy.use_auto_smooth = True
        copy.normals_split_custom_set(before)
    after = _corner_normals(copy)
    worst = min((a.dot(b) for a, b in zip(before, after)), default=1.0) if len(after) == len(before) else -1.0
    report = {"pointsBefore": len(mesh.vertices), "pointsAfter": len(copy.vertices), "hardEdges": len(hard)}
    keys_ok = copy.shape_keys is not None and len(copy.shape_keys.key_blocks) == len(mesh.shape_keys.key_blocks)
    if worst < 0.9999 or not keys_ok:
        bpy.data.meshes.remove(copy)
        report["notSplit"] = ("splitting would have changed the file's corner normals" if keys_ok
                              else "the split copy lost its shape keys")
        return None, report
    return copy, report


# ---------------------------------------------------------------------------
# export_mesh localSpace: the file written round one object's origin
# ---------------------------------------------------------------------------
# WHY IT EXISTS. Unreal's FBX import bakes each node's world transform into the vertices, so a tree standing at
# (5, 3, 0.5) turned 30 degrees arrived as an asset whose pivot was 5.9 m from its trunk - measured 2026-09-27 on
# 5.8.2 (scratchpad handoff_axes.py). Asking Unreal to skip that instead (transformVertexToAbsolute false) drops the
# 100x unit conversion with it: ComputeTotalMatrix returns identity (FbxMainImport.cpp 5.8:2060-2098) and
# ImportUniformScale rides the matrix it skipped (:2734-2742), so the same tree came in 3 cm tall. The fix belongs
# here: write the file as if the object stood at the origin, unrotated - its scale kept, see _local_space_move - and
# put it back in the same call.

def _save_transform(obj):
    """Every property a matrix_world assignment rewrites, by VALUE. Decomposing the matrix again would normalize the
    angles - a 370 degree turn comes back as 10 - and the numbers the user typed are theirs, not just their effect."""
    return (obj.rotation_mode, obj.location.copy(), obj.rotation_euler.copy(), obj.rotation_quaternion.copy(),
            tuple(obj.rotation_axis_angle), obj.scale.copy())


def _restore_transform(obj, saved):
    # rotation_mode FIRST: assigning it converts the rotation values into the new mode (rna_Object_rotation_mode_set),
    # which the three assignments after it then overwrite with the saved ones.
    obj.rotation_mode = saved[0]
    obj.location, obj.rotation_euler, obj.rotation_quaternion, obj.rotation_axis_angle, obj.scale = saved[1:]


def _matrix_close(got, want):
    tol = 1e-5 * max(1.0, max(abs(want[i][j]) for i in range(4) for j in range(4)))
    return all(abs(got[i][j] - want[i][j]) <= tol for i in range(4) for j in range(4))


def _shape_follows_placement(obj):
    """Can moving this object change the mesh the exporter writes? Only through what evaluates on top of its data: a
    modifier (a Boolean against an object left behind, a geometry node reading its own position) or a shape key a
    driver moves."""
    return obj.type == "MESH" and (obj.data.shape_keys is not None or any(m.show_viewport for m in obj.modifiers))


def _written_box(obj, depsgraph, with_modifiers):
    """{name, min, max, triangles} of the mesh the exporter writes for `obj`, in BU in the FILE's frame - its
    matrix_world while localSpace has the export moved: the evaluated mesh when modifiers are exported, else the base
    mesh. min and max are None for a mesh with no vertices. What a receiving engine's bounds and triangle count are
    checked against (mif_send_to_unreal), so it is read off the vertices, never off the cached bound_box.

    materialsUsed names the material in each slot at least one written polygon uses, in slot order (None for an empty
    slot that faces use), beside materialSlotCount, every slot the object carries. AN UNUSED SLOT DOES NOT ARRIVE: a
    dumpster with four materials and faces on two came into 5.8.2 as two slots, and the hand-off held it to all four and
    refused a valid mesh (2026-10-02). The index is clamped to the last slot, as Blender draws one past the end."""
    source = obj.evaluated_get(depsgraph) if with_modifiers else obj
    mesh = source.to_mesh() if with_modifiers else obj.data
    try:
        mesh.calc_loop_triangles()
        row = {"name": obj.name, "min": None, "max": None, "triangles": len(mesh.loop_triangles)}
        slots = [s.material.name if s.material else None for s in obj.material_slots]
        row["materialSlotCount"] = len(slots)
        if slots and len(mesh.polygons):
            idx = array.array("i", bytes(4 * len(mesh.polygons)))
            mesh.polygons.foreach_get("material_index", idx)
            used = {min(max(i, 0), len(slots) - 1) for i in idx}
            row["materialsUsed"] = [slots[i] for i in sorted(used)]
        else:
            row["materialsUsed"] = []
        count = len(mesh.vertices)
        if not count:
            return row
        flat = array.array("f", bytes(12 * count))
        mesh.vertices.foreach_get("co", flat)
        axes = (flat[0::3], flat[1::3], flat[2::3])
        m = obj.matrix_world
        if all(abs(m[i][j] - (1.0 if i == j else 0.0)) <= 1e-9 for i in range(4) for j in range(4)):
            lo = [min(a) for a in axes]
            hi = [max(a) for a in axes]
        else:
            lo, hi = [], []
            for r in range(3):
                a, b, c, d = m[r]
                vals = [a * x + b * y + c * z for x, y, z in zip(*axes)]
                lo.append(min(vals) + d)
                hi.append(max(vals) + d)
        row["min"], row["max"] = rnd(lo), rnd(hi)
        return row
    finally:
        if with_modifiers:
            source.to_mesh_clear()


def _local_space_move(ref, export_set, with_modifiers):
    """Move the export's ROOTS - the objects no exported object carries - so `ref` stands at the origin, unrotated, ITS
    SCALE KEPT, and every other exported object keeps its place relative to it (a UCX_ hull sharing the tree's transform
    lands on the tree). Returns (saved transforms, world matrices before, meshes checked, what was written, the
    placement matrix); on any failure puts every object back and raises, so nothing is written.

    THE SCALE STAYS IN THE FILE, and the first version took it out. Location and rotation say where a thing stands;
    scale says how big it is, and an asset should be the size it looks. It mattered most for a mesh that came FROM
    Unreal: import_mesh keeps the FBX's centimeters as the object's 0.01 scale, so writing that object unscaled would
    have made its asset 100 times too big. So the file's frame is the object's location and rotation undone, read off
    its world matrix by decompose(); a transform that is more than location, rotation and scale - a parent's
    non-uniform scale under a rotation shears it - cannot be written this way, and the move check refuses it."""
    ref_world = ref.matrix_world.copy()
    if abs(ref_world.determinant()) < 1e-12:
        raise MifOpError("localSpace writes the file round '%s', and its world matrix cannot be inverted (a zero scale "
                         "on an axis, its own or a parent's), so there is no origin to write it round. NOTHING was "
                         "written." % ref.name)
    loc, rot, _scale = ref_world.decompose()
    place = Matrix.Translation(loc) @ rot.to_matrix().to_4x4()
    inv = place.inverted()
    members = {o.name for o in export_set}
    roots = []
    for obj in export_set:
        par = obj.parent
        while par is not None and par.name not in members:
            par = par.parent
        if par is None:
            roots.append(obj)
    before = {o.name: o.matrix_world.copy() for o in export_set}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    shapes = [(o, _mesh_shape(o.evaluated_get(depsgraph).data)) for o in export_set
              if with_modifiers and _shape_follows_placement(o)]
    saved = [(o, _save_transform(o)) for o in roots]
    try:
        for obj in roots:
            obj.matrix_world = inv @ before[obj.name]
        bpy.context.view_layer.update()
        stuck = [o.name for o in export_set if not _matrix_close(o.matrix_world, inv @ before[o.name])]
        if stuck:
            raise MifOpError(
                "localSpace could not move %s to where the file needs %s: something else sets that transform - an "
                "animation, a driver, a constraint - or a parent's non-uniform scale shears it, which an object's own "
                "location, rotation and scale cannot hold. NOTHING was written, and every object is back where it "
                "was." % (", ".join("'%s'" % n for n in stuck), "them" if len(stuck) > 1 else "it"))
        depsgraph = bpy.context.evaluated_depsgraph_get()
        moved_shape = [o.name for o, digest in shapes if _mesh_shape(o.evaluated_get(depsgraph).data) != digest]
        if moved_shape:
            raise MifOpError(
                "localSpace moved %s to the origin and %s evaluated mesh CHANGED: a modifier or a geometry node reads "
                "where the object stands (a Boolean against an object that stays behind, a position-driven effect), "
                "so the file would not hold the mesh you see. NOTHING was written, and every object is back where it "
                "was. Export without localSpace, or apply that modifier first."
                % (", ".join("'%s'" % n for n in moved_shape), "their" if len(moved_shape) > 1 else "its"))
        written = [_written_box(o, depsgraph, with_modifiers) for o in export_set if o.type == "MESH"]
    except BaseException:
        _local_space_restore(saved)
        raise
    return saved, before, [o.name for o, _digest in shapes], written, place


def _local_space_restore(saved):
    for obj, values in saved:
        _restore_transform(obj, values)
    bpy.context.view_layer.update()


def _written_height(targets, args):
    """The tallest exported mesh's height as the FBX will carry it - so a 4x scale error is a number before the import.

    WHERE THE EXPORTER PUTS THE SCALE, measured 2026-10-04 by reading the written files back with io_scene_fbx's own
    parser on 3.6.23, 4.4.0 and 5.2.1 (identical): a 1 x 1 x 2 BU box (a unit cube scaled 2 in Z), apply_unit_scale on,
    global_scale 0.25:
        FBX_SCALE_NONE    Lcl Scaling z 50,  UnitScaleFactor 1     -> 50 file units, 50 cm
        FBX_SCALE_UNITS   Lcl Scaling z 0.5, UnitScaleFactor 100   -> 0.5 file units, 50 cm
        FBX_SCALE_CUSTOM  Lcl Scaling z 200, UnitScaleFactor 0.25  -> 200 file units, 50 cm
        FBX_SCALE_ALL     Lcl Scaling z 2,   UnitScaleFactor 25    -> 2 file units, 50 cm
    The vertices stay unscaled in every case; the scale rides on the node. So the centimetres agree across the four
    and the raw numbers do not: an importer that ignores UnitScaleFactor reads fileUnits as centimetres."""
    unit = 100.0 * float(bpy.context.scene.unit_settings.scale_length) if args.get("apply_unit_scale") else 1.0
    gs = finite_float(args.get("global_scale", 1.0), "globalScale")
    opt = args.get("apply_scale_options", "FBX_SCALE_NONE")
    raw_factor, usf = {"FBX_SCALE_NONE": (unit * gs, 1.0), "FBX_SCALE_UNITS": (gs, unit),
                       "FBX_SCALE_CUSTOM": (unit, gs), "FBX_SCALE_ALL": (1.0, unit * gs)}.get(opt, (None, None))
    best = None
    for obj in targets:
        if obj.type != "MESH" or obj.data is None:
            continue
        bounds = evaluated_mesh_bounds(obj)
        if bounds is None:
            if not obj.data.vertices:
                continue
            co = [v.co for v in obj.data.vertices]
            lo = [min(c[i] for c in co) for i in range(3)]
            hi = [max(c[i] for c in co) for i in range(3)]
        else:
            lo, hi = bounds[0], bounds[1]
            if bounds[2] == 0:
                continue
        corners = [obj.matrix_world @ Vector((x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                   for z in (lo[2], hi[2])]
        h = max(c.z for c in corners) - min(c.z for c in corners)
        if best is None or h > best[1]:
            best = (obj.name, h)
    if best is None or raw_factor is None:
        return None
    return {"object": best[0], "heightBU": round(best[1], 6), "applyScaleOptions": opt, "globalScale": gs,
            "fileUnits": round(best[1] * raw_factor, 6), "unitScaleFactor": round(usf, 6),
            "centimetres": round(best[1] * raw_factor * usf, 6)}


def _material_images(objs):
    """Every image an image-texture node in these objects' materials reads (nested groups included), once each."""
    seen, out, trees = set(), [], []
    for o in objs:
        for slot in getattr(o, "material_slots", []):
            if slot.material is not None and slot.material.node_tree is not None:
                trees.append(slot.material.node_tree)
    done = set()
    while trees:
        t = trees.pop()
        if t.name in done:
            continue
        done.add(t.name)
        for n in t.nodes:
            if n.type == "TEX_IMAGE" and n.image is not None and n.image.name not in seen:
                seen.add(n.image.name)
                out.append(n.image)
            elif n.type == "GROUP" and getattr(n, "node_tree", None) is not None:
                trees.append(n.node_tree)
    return out


def _tangent_report(targets, args, path):
    """export_mesh's tangentSpace and zeroAreaUvFaces: what the file holds, and why a mesh got no tangents.

    inFile IS READ OFF THE FILE - a Tangents data node - not predicted, and not the "LayerElementTangent" string: on a
    mesh it skipped, the exporter still writes the Layer's reference to a tangent element and no tangents (5.2.2, a
    pentagon). The per-mesh reasons are the exporter's
    own two rules (io_scene_fbx export_fbx_bin.py): tangents are written per UV map, so a mesh with none gets none,
    and a mesh holding a face of more than 4 corners gets none at all - only a report on the operator, which nothing
    here saw - unless use_triangles splits it first. Zero-area UV faces are counted on UV channel 0, the one Unreal
    builds tangents from, on the mesh as written (with its modifiers when use_mesh_modifiers is on)."""
    requested = bool(args.get("use_tspace"))
    depsgraph = bpy.context.evaluated_depsgraph_get() if args.get("use_mesh_modifiers") else None
    skipped, zero = [], []
    for obj in targets:
        bm = bmesh.new()
        try:
            if depsgraph is not None:
                bm.from_object(obj, depsgraph)
            else:
                bm.from_mesh(obj.data)
            layers = bm.loops.layers.uv
            uv = layers[0] if len(layers) else None
            big = 0 if args.get("use_triangles") else sum(1 for f in bm.faces if len(f.verts) > 4)
            if requested and uv is None:
                skipped.append({"object": obj.name, "why": "no UV map - the exporter writes tangents per UV map, and "
                                                           "Unreal builds them from UV channel 0 (bl_uv_unwrap)"})
            elif requested and big:
                skipped.append({"object": obj.name,
                                "why": "%d face(s) with more than 4 corners and useTriangles off - Blender's FBX "
                                       "exporter computes tangents on triangles and quads only and skips the whole "
                                       "mesh; pass useTriangles true" % big})
            if uv is not None:
                faces = [f.index for f in bm.faces if _uv_area(f, uv) <= NEAR_ZERO_UV_AREA]
                if faces:
                    zero.append({"object": obj.name, "uvLayer": uv.name, "count": len(faces), "faces": faces[:50]})
        finally:
            bm.free()
    try:
        with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            in_file = mm.find(b"\x08Tangents") >= 0
    except (OSError, ValueError):
        in_file = None
    if not requested:
        note = ("useTspace false: no tangents were written, so Unreal computes its own - and a mesh with zero-area UV "
                "faces then imports with 'nearly zero tangents' warnings.")
    elif skipped:
        note = ("%d of %d mesh(es) got no tangents - skipped says why. Unreal computes those itself."
                % (len(skipped), len(targets)))
    elif in_file is not True:
        note = "tangents were asked for and the file holds none - read the exporter's console output."
    else:
        note = None
    out = {"tangentSpace": {"requested": requested, "inFile": in_file, "skipped": skipped or None, "note": note},
           "zeroAreaUvFaces": zero or None}
    if zero:
        out["zeroAreaUvNote"] = (
            "%d face(s) on UV channel 0 have a UV area of %g or less (zero area in UV space) - legal here, but Unreal "
            "gives their corners nearly zero tangents and warns on import (tolerance 1E-4) when it computes tangents "
            "itself; with tangents in the file it did not (5.8.2). Re-unwrap them (bl_uv_unwrap); bl_uv_info names "
            "every one." % (sum(z["count"] for z in zero), NEAR_ZERO_UV_AREA))
    return out


def op_export_mesh(params):
    """Export one object, a list of objects, or every mesh in the view layer to an .fbx file (glTF export is refused)
    with the bridge's pinned scale and axis settings; an existing file is replaced unless overwrite is false."""
    reject_unknown(params, set(_EXPORT_OVERRIDES) | {
        "object", "name", "objects", "file", "filepath", "path",
        "overwrite", "replaceExisting", "objectTypes", "localSpace", "animationsOnly", "globalScale",
        "applyScaleOptions", "pathMode", "embedTextures", "facingBody", "splitShapeCorners", "sockets"}, "export_mesh")
    # THE SCALE PINS CAN BE CHANGED, NEVER BY DEFAULT (2026-10-04): a caller who sends neither gets the pinned
    # FBX_SCALE_NONE and 1.0 exactly as before. Read and checked here, before anything is written.
    gscale = take_float(params, "globalScale", default=None)
    if gscale is not None and not 0.001 <= gscale <= 1000.0:
        raise MifOpError("globalScale must be 0.001-1000 (the exporter's own range), got %g. NOTHING was written."
                         % gscale)
    # THE ANIMATIONS ALONE (2026-10-04): the second file of a rig's two-file export - armature and clips, no mesh. The
    # one case where a file without a mesh is wanted, so it is a named switch rather than objectTypes [ARMATURE].
    anims_only = take_bool(params, "animationsOnly", default=False)
    if anims_only and params.get("objectTypes") is not None:
        raise MifOpError("animationsOnly writes the armature alone; objectTypes would contradict it. NOTHING was "
                         "written.")
    if anims_only and params.get("bakeAnim") is False:
        raise MifOpError("animationsOnly writes the clips; bakeAnim false would leave the file empty. NOTHING was "
                         "written.")
    anim_keys = [k for k in ("bakeAnimAllActions", "bakeAnimNla", "bakeAnimSimplify", "bakeAnimForceKeying")
                 if params.get(k) is not None]
    if anim_keys and not anims_only and params.get("bakeAnim") is not True:
        raise MifOpError("%s shape the baked animation, and bakeAnim is off - pass bakeAnim true or animationsOnly. "
                         "NOTHING was written." % ", ".join(anim_keys))

    raw = take(params, "file", "filepath", "path", required=True, kind=str)
    path = _resolve_out_path(raw)
    # BEFORE THE EXPORT RUNS. A control character in the path survives every guard here and then
    # collapses inside the exporter, which comes back as a bare RuntimeError carrying a Python
    # traceback - measured on 5.0.1 with an FBX export. Every other refusal in this addon is a
    # sentence.
    check_output_path(raw, path, "exported")
    _check_format(path, "export_mesh", _EXPORT_FORMATS)
    # WHERE THE TEXTURES GO (2026-10-05): path_mode was pinned AUTO, so a textured scan's images could not travel with
    # its file. Measured on 4.2.23, 4.4.0, 5.0.1 and 5.2.2: COPY with embed_textures writes each image's bytes inside the
    # .fbx; COPY alone copies them into an <name>.fbm folder beside it; embed_textures with any other mode writes no
    # image at all - Blender ignores it - so that is refused rather than passed on.
    path_mode = take(params, "pathMode", default=None, kind=str)
    embed = take_bool(params, "embedTextures", default=False)
    if path_mode is not None:
        modes = [i.identifier for i in bpy.ops.export_scene.fbx.get_rna_type().properties["path_mode"].enum_items]
        if str(path_mode).upper() not in modes:
            raise MifOpError("pathMode must be one of %s (this build's exporter), got %r. NOTHING was written."
                             % (", ".join(modes), path_mode))
        path_mode = str(path_mode).upper()
    if embed and path_mode != "COPY":
        raise MifOpError("embedTextures needs pathMode COPY - with any other mode Blender writes no image into the file "
                         "(measured), so the textures would not travel. NOTHING was written.")
    # the exporter's own list, read after the path is known good (audit_output_paths)
    scale_opt = take(params, "applyScaleOptions", default=None, kind=str)
    if scale_opt is not None:
        opts = [i.identifier for i in
                bpy.ops.export_scene.fbx.get_rna_type().properties["apply_scale_options"].enum_items]
        if str(scale_opt).upper() not in opts:
            raise MifOpError("applyScaleOptions must be one of %s (this build's exporter), got %r. NOTHING was written."
                             % (", ".join(opts), scale_opt))
        scale_opt = str(scale_opt).upper()

    # Read BEFORE the exists() test - see audit_blender_shortcircuit_take.
    # REPLACED BY DEFAULT, NEVER SILENTLY (decided 2026-09-28, the rule every export follows - see
    # ops_common.file_before): the answer says overwrote, and what it replaced.
    overwrite = take_bool(params, "overwrite", "replaceExisting", default=True)
    if os.path.exists(path) and not overwrite:
        raise MifOpError("%s already exists and overwrite:false. Pass overwrite:true or "
                         "a different file. NOTHING was written." % path)

    names = take(params, "objects")
    if names is not None and (params.get("object") is not None or params.get("name") is not None):
        # BOTH WERE TAKEN AND ONE DROPPED: objects won and object was ignored without a word (2026-10-04, when the tool
        # began sending objects).
        raise MifOpError("pass object OR objects, not both - one file is written from one list. NOTHING was written.")
    if names is None:
        single = take(params, "object", "name")
        names = [single] if single else None
    if names is not None and not isinstance(names, list):
        raise MifOpError("'objects' must be a list of object names")
    local = take_bool(params, "localSpace", default=False)
    if local and not names:
        raise MifOpError("localSpace writes the file round the FIRST NAMED object's origin, and none was named - every "
                         "mesh in the view layer has no first one. Pass 'object', or 'objects' with that object "
                         "first. NOTHING was written.")

    if names:
        targets = [get_object(n, want_mesh=True) for n in names]
    else:
        targets = [o for o in bpy.context.view_layer.objects if o.type == "MESH"]
        if not targets:
            raise MifOpError("no mesh objects in the view layer to export. Pass 'object' "
                             "explicitly, or import something first.")
    # FACES THAT POINT INTO THE BODY (2026-10-05): a rebuilt cuff and hem band that faced inward shipped and showed as
    # holes under a one-sided material. facingBody {body, maxInwardShare (default 0.1), bandStep, reach} measures each
    # mesh exported as mesh_quality facingAgainst does and refuses before the write when a larger share faces inward.
    facing_reply = None
    if params.get("facingBody") is not None:
        from .ops_garment import facing_report, parse_facing    # noqa: PLC0415 - the garment measurements live there
        spec = params["facingBody"]
        spec = dict(spec) if isinstance(spec, dict) else {"body": spec}
        max_share = finite_float(spec.pop("maxInwardShare", 0.1), "facingBody.maxInwardShare")
        if not 0.0 <= max_share <= 1.0:
            raise MifOpError("facingBody.maxInwardShare is a share from 0 to 1, got %r. NOTHING was written." % max_share)
        facing_reply = []
        for t in targets:
            body, step, reach = parse_facing(spec, t, tail="NOTHING was written.")
            rep_ = facing_report(t, body, step, reach)
            facing_reply.append({"object": t.name, "body": body.name, "inward": rep_["inward"],
                                 "facesMeasured": rep_["facesMeasured"], "inwardShare": rep_["inwardShare"]})
            if rep_["inwardShare"] > max_share:
                raise MifOpError("%d of %d faces of '%s' near '%s' point into it (share %.4f, over maxInwardShare %g) - "
                                 "under a one-sided material they show as holes. Flip them (clean_mesh flipNormals with "
                                 "mesh_quality facingAgainst's inwardFaces), or raise maxInwardShare for folded-under "
                                 "edges that face the body by design. NOTHING was written."
                                 % (rep_["inward"], rep_["facesMeasured"], t.name, body.name, rep_["inwardShare"],
                                    max_share))

    args = dict(FBX_EXPORT_ARGS)
    for key, (arg, kind) in _EXPORT_OVERRIDES.items():
        if key in params:
            value = params[key]
            if kind is bool:
                value = take_bool(params, key)
            elif kind is float:
                value = take_float(params, key)
                if not 0.0 <= value <= 100.0:
                    raise MifOpError("%s must be 0-100 (0 keeps every key), got %r. NOTHING was written."
                                     % (key, value))
            args[arg] = value
    if anims_only:
        args["bake_anim"] = True
        args["object_types"] = {"ARMATURE"}
    if gscale is not None:
        args["global_scale"] = gscale
    if scale_opt is not None:
        args["apply_scale_options"] = scale_opt
    if path_mode is not None:
        args["path_mode"] = path_mode
    if embed:
        args["embed_textures"] = True

    # ------------------------- SKELETAL EXPORT -------------------------
    # WHAT WAS WRONG BEFORE THIS, and it was worse than "the skeleton is missing". object_types was
    # pinned to {"MESH"}, and io_scene_fbx only backs up and restores the REST POSE when ARMATURE is
    # in that set (export_fbx_bin.py). With it absent, an Armature modifier is evaluated like any
    # other modifier, so the mesh was written DEFORMED INTO WHATEVER POSE THE RIG HAPPENED TO BE IN.
    # A character posed mid-animation exported as a static mesh frozen in that pose, with no
    # skeleton, and the file was perfectly valid - nothing failed, nothing warned. The Unreal side
    # needs no work at all: UFbxFactory auto-detects skeletal versus static from the file.
    types = take(params, "objectTypes")
    if types is not None:
        if not isinstance(types, (list, tuple)) or not types:
            raise MifOpError("objectTypes must be a non-empty array, e.g. [\"MESH\", \"ARMATURE\"]. "
                             "NOTHING was written.")
        # WHAT THE EXPORTER ACCEPTS IS NOT WHAT THIS OP CAN REACH. object_types is a FILTER applied
        # to the selection, and the selection this op builds is meshes (want_mesh=True, or every
        # MESH in the view layer) plus the armatures their modifiers point at - nothing else is ever
        # selected. So EMPTY, CAMERA, LIGHT and OTHER were accepted, forwarded, and could not change
        # the output by a single byte. That is a key the op declares and never really reads, which
        # this addon refuses rather than tolerates.
        #
        # The comment a few lines down already makes this exact point for ARMATURE - "must be in the
        # SELECTION, not merely allowed by object_types" - and the other four were never revisited
        # in that light.
        reachable = {"MESH", "ARMATURE"}
        valid = {"EMPTY", "CAMERA", "LIGHT", "ARMATURE", "MESH", "OTHER"}
        types = [str(t).upper() for t in types]
        bad = [t for t in types if t not in valid]
        if bad:
            raise MifOpError("unknown objectTypes %s. Accepted: %s. NOTHING was written."
                             % (", ".join("'%s'" % b for b in bad), ", ".join(sorted(valid))))
        unreachable = [t for t in types if t not in reachable]
        if unreachable:
            raise MifOpError(
                "objectTypes %s cannot affect this export. object_types filters the SELECTION, and "
                "export_mesh selects meshes and the armatures their modifiers point at - never a "
                "camera, light or empty - so naming them would have changed nothing while looking "
                "like it did. Use %s. Exporting a whole scene including lights and cameras is a "
                "different job than export_mesh. NOTHING was written."
                % (", ".join("'%s'" % u for u in unreachable), " or ".join(sorted(reachable))))
        if "MESH" not in types:
            raise MifOpError("objectTypes must include MESH - this is export_mesh, and a file with "
                             "no mesh in it is not something any caller here wants. NOTHING was "
                             "written.")
        args["object_types"] = set(types)

    # THE ARMATURE MUST BE IN THE SELECTION, not merely allowed by object_types. The exporter
    # gathers its context objects from the selection when use_selection is on, so naming ARMATURE in
    # object_types without selecting the armature produces exactly the same silently-posed static
    # mesh as before - the filter would have let it through and nothing offered it. Found by reading
    # export_fbx_bin.py rather than by exporting and wondering.
    export_set = list(targets)
    armatures = []
    unexported_deformers = []
    # A MESH ON A BONE IS NOT A SKIN: the engine's importer can drop a mesh parented to a bone or held by a Child Of
    # constraint without a word (a reported case, not measured here), so each is named before the write.
    bone_parented = sorted(o.name for o in targets if getattr(o, "parent_type", "") == "BONE"
                           or any(c.type == "CHILD_OF" for c in getattr(o, "constraints", [])))
    for obj in targets:
        for mod in getattr(obj, "modifiers", []):
            if mod.type == "ARMATURE" and getattr(mod, "object", None) is not None:
                arm = mod.object
                if "ARMATURE" in args["object_types"]:
                    if arm not in export_set:
                        export_set.append(arm)
                    if arm.name not in armatures:
                        armatures.append(arm.name)
                elif arm.name not in unexported_deformers:
                    unexported_deformers.append(arm.name)
    # SOCKETS (second 1.3 audit, 2026-10-25): sockets [names] writes those Empties into the file as well, each named
    # SOCKET_<name> and parented to an exported mesh, so Unreal's FBX importer makes them static-mesh sockets (it reads a
    # null node with that prefix - UE 5.8 Editor/UnrealEd/Private/Fbx/FbxStaticMeshImport.cpp:2524). objectTypes cannot
    # reach an Empty (above), so they are a list of their own, and EMPTY joins the written types only with them.
    socket_names = take(params, "sockets")
    socket_objs = []
    if socket_names is not None:
        if not isinstance(socket_names, list) or not all(isinstance(n, str) for n in socket_names):
            raise MifOpError("sockets is a list of Empty names, e.g. [\"SOCKET_Muzzle\"]. NOTHING was written.")
        target_names = {t.name for t in targets}
        for n in socket_names:
            e = bpy.data.objects.get(n)
            if e is None or e.type != "EMPTY" or not n.startswith("SOCKET_") \
                    or e.parent is None or e.parent.name not in target_names:
                raise MifOpError("sockets: '%s' must be an Empty named SOCKET_<name> whose parent is an exported mesh "
                                 "(%s) - Unreal reads a socket only from that prefix, and one parented elsewhere has no "
                                 "mesh to belong to. NOTHING was written." % (n, ", ".join(sorted(target_names))))
            if e not in socket_objs:
                socket_objs.append(e)
        if socket_objs:
            export_set.extend(socket_objs)
            args["object_types"] = set(args["object_types"]) | {"EMPTY"}
    if anims_only and not armatures:
        raise MifOpError("animationsOnly writes a rig's clips, and no object exported here is skinned to an armature "
                         "(an Armature modifier with an object). NOTHING was written.")

    # SHAPE KEYS ARE DROPPED by a mesh-modifier export. Measured 2026-10-04 on 5.2.1 through this op and import_mesh: a
    # cube with keys Basis/Up came back with both; with a Displace modifier on it (useMeshModifiers, the default), with
    # NONE; and with only an Armature modifier whose rig was not in the file, with none as well - an unexported rig's
    # modifier is applied like any other. The exporter says nothing, so every such mesh is named before the write.
    keys_dropped = {}
    if args.get("use_mesh_modifiers"):
        for obj in targets:
            if obj.type != "MESH" or not obj.data.shape_keys or len(obj.data.shape_keys.key_blocks) < 2:
                continue
            applied = [m.name for m in obj.modifiers if m.show_viewport
                       and not (m.type == "ARMATURE" and getattr(m, "object", None) in export_set)]
            if applied:
                keys_dropped[obj.name] = applied

    # A KEYED MESH'S POINTS SPLIT AT ITS HARD CORNERS (1.3, splitShapeCorners, default on) - why at _split_shape_corners.
    # Read here, after every refusal above; the copies are made inside the try below and always removed.
    split_corners = take_bool(params, "splitShapeCorners", default=True)
    keyed = [o for o in targets if o.type == "MESH" and o.data.shape_keys is not None
             and len(o.data.shape_keys.key_blocks) >= 2 and o.name not in keys_dropped
             and "MESH" in args["object_types"]]

    # THE HEIGHT THE FILE WILL CARRY, of the tallest mesh, read before localSpace can move anything.
    written_height = _written_height(targets, args)

    # WHAT THIS EXPORT REPLACES, measured after every refusal and right before the write.
    replaced = file_before(path)
    fbm = os.path.splitext(path)[0] + ".fbm"
    fbm_before = set(os.listdir(fbm)) if os.path.isdir(fbm) else set()
    snapshot = selection_snapshot()
    ref_world = targets[0].matrix_world.copy()
    moved = None
    swapped, split_report = [], {}
    try:
        if split_corners:
            for obj in keyed:
                copy, rep_ = _split_shape_corners(obj)
                if rep_ is not None:
                    split_report[obj.name] = rep_
                if copy is not None:
                    swapped.append((obj, obj.data, copy))
                    obj.data = copy
        select_only(export_set)
        if local:
            moved = _local_space_move(targets[0], export_set, bool(args.get("use_mesh_modifiers")))
        bpy.ops.export_scene.fbx(filepath=path, use_selection=True, **args)
    finally:
        if moved:
            _local_space_restore(moved[0])
        for obj, original, copy in swapped:
            obj.data = original
            bpy.data.meshes.remove(copy)
        selection_restore(snapshot)

    # ------------------------- VERIFY AFTER WRITE -------------------------
    # The operator reports {'FINISHED'} on paths that produce nothing useful, so
    # re-stat the file rather than trusting the return. An op that answered
    # ok:true over a missing or 0-byte FBX is the exact bug this guards.
    exists = os.path.isfile(path)
    size = os.path.getsize(path) if exists else -1
    if not exists or size <= 0:
        raise MifOpError(
            "export wrote no usable file: %s (exists=%s, bytes=%d). Check Blender's "
            "console for the exporter's own error, and that the path is writable."
            % (path, exists, size))

    bones_written = 0
    for name in armatures:
        arm = bpy.data.objects.get(name)
        if arm is not None and getattr(arm, "data", None) is not None:
            bones_written += len(arm.data.bones)

    kids = children_by_parent()          # one pass for every object_info below - see ops_common.children_by_parent
    out = {
        "file": path,
        "fileExists": True,
        "fileSizeBytes": size,
        **replaced_fields(replaced),
        "exportedCount": len(targets),
        "exported": [object_info(o, children=kids) for o in targets],
        "objectTypesWritten": sorted(args["object_types"]),
        "armaturesExported": armatures,
        "bonesWritten": bones_written,
        "leafBonesAdded": bool(args.get("add_leaf_bones")),
        "isSkeletal": bool(armatures),
        "axis": {"up": "Z", "front": "-Y", "handedness": "right", "unit": "cm",
                 "unrealUnitsPerBlenderUnit": UU_PER_BU},
        "fbxArgs": {k: (sorted(v) if isinstance(v, set) else v)
                    for k, v in args.items()},
        "writtenHeight": written_height,
    }
    if socket_names is not None:
        out["socketsWritten"] = sorted(e.name for e in socket_objs)
    if embed:
        # FOUND IN THE FILE, image by image: each image file the exported materials read is looked for in the .fbx by
        # its own first 256 bytes, so the count is what the file carries rather than what was asked for.
        with open(path, "rb") as fh:
            blob = fh.read()
        found, missing = [], []
        for img in _material_images(targets):
            src = bpy.path.abspath(img.filepath) if img.filepath else ""
            if not src or not os.path.isfile(src):
                missing.append(img.name)
                continue
            with open(src, "rb") as fh:
                head = fh.read(256)
            (found if head and head in blob else missing).append(img.name)
        out["embeddedImages"] = len(found)
        out["embeddedImageNames"] = found[:50]
        out["imagesNotEmbedded"] = missing[:50] or None
    elif args.get("path_mode") == "COPY":
        now = set(os.listdir(fbm)) if os.path.isdir(fbm) else set()
        out["texturesCopied"] = sorted(os.path.join(fbm, f) for f in now - fbm_before)[:50] or sorted(
            os.path.join(fbm, f) for f in now)[:50]
    if (written_height and abs(finite_float(args.get("global_scale", 1.0), "globalScale") - 1.0) > 1e-9
            and args.get("apply_scale_options") == "FBX_SCALE_NONE" and armatures):
        out["scaledArmatureWarning"] = (
            "globalScale %g with FBX_SCALE_NONE writes the scale into each node's transform (Lcl Scaling), not into the "
            "vertices - measured on 3.6.23, 4.4.0 and 5.2.1 - and the exporter itself warns that some importers do not "
            "support scaled armatures. Read the imported root bone's scale after the first import, or scale the data in "
            "Blender (an armature scale applied before export) or at import (uniformScale), not both."
            % float(args["global_scale"]))
    # THE PINS STAY THE DEFAULT, AND LEAVING THEM IS SAID (2026-10-04: "add opt-in parameters that warn in the
    # reply when they change the verified round trip"). Scale 1.0 with FBX_SCALE_NONE and apply_unit_scale is the
    # measured 1 Blender unit = 100 Unreal units; any other pair writes a file the round-trip checks never measured.
    gs_used = finite_float(args.get("global_scale", 1.0), "globalScale")
    opt_used = args.get("apply_scale_options", "FBX_SCALE_NONE")
    if abs(gs_used - 1.0) > 1e-9 or opt_used != "FBX_SCALE_NONE":
        out["roundTripWarning"] = (
            "globalScale %g with %s leaves the verified round trip (1.0 with FBX_SCALE_NONE: 1 Blender unit = 100 Unreal "
            "units, measured) - this file carries writtenHeight, not the scene's own size. Import it without a second "
            "scale, and expect mif_mesh_roundtrip's size checks to disagree." % (gs_used, opt_used))

    # THE OTHER DIRECTION OF A CHECK THIS FILE ALREADY MAKES. import_mesh REFUSES a non-identity
    # transform on the way in; export was silent on the way out, which is the costly half. A
    # non-identity object transform means the pivot has moved and the loc/rot/scale is NOT
    # baked into the mesh data, so the receiving engine applies it again on top - a mesh that
    # is double-scaled or offset from its own origin, discovered in Unreal rather than here.
    # isIdentityTransform was ALREADY in the response, per object, inside `exported`; nothing
    # raised it to where somebody reads it. A note rather than a refusal, because exporting a
    # transformed object is legitimate when you mean it - apply_transform is the fix when you
    # do not.
    if moved:
        # MEASURED, not assumed: the transforms are read back after the restore, against the matrices from before
        # the move. A mismatch cannot stop a file already written, so it is said instead.
        off = [o.name for o in export_set if not _matrix_close(o.matrix_world, moved[1][o.name])]
        out["localSpace"] = {
            "relativeTo": targets[0].name,
            # WHERE IT STANDS IN THE SCENE, which the file no longer says: rows of the world matrix, in BU.
            "sceneWorldMatrix": [rnd(row) for row in ref_world],
            # ...and the part of it the file left out - its location and rotation, no scale. Placing the imported
            # asset with this puts it back where it stood (mif_send_to_unreal's place does).
            "placementMatrix": [rnd(row) for row in moved[4]],
            "movedForExport": [o.name for o, _saved in moved[0]],
            "meshesCheckedUnchanged": moved[2],
            # WHAT THE FILE HOLDS, per mesh, in its own frame: the box (BU) and triangles of the vertices written.
            "written": moved[3],
            "restored": not off,
        }
        if off:
            out["localSpaceRestoreWarning"] = (
                "%s did not come back to the transform it had before the export. The file is right; the scene is "
                "not. Check those objects' location, rotation and scale." % ", ".join("'%s'" % n for n in off))
    # With localSpace the file carries the first object at identity and the rest RELATIVE to it, on purpose, so the
    # warning below - about a transform the file carries - does not apply.
    shifted = [] if moved else [o["name"] for o in out["exported"] if not o.get("isIdentityTransform", True)]
    if shifted:
        out["nonIdentityTransformWarning"] = (
            "%s do NOT have an identity transform, so their loc/rot/scale is not baked into "
            "the mesh data and the importing engine will apply it AGAIN on top of what is in "
            "the file. Call apply_transform first if the mesh should arrive where it looks "
            "like it is, or pass localSpace:true to write the file round the first object's "
            "origin. The file was written either way."
            % ", ".join("'%s'" % n for n in shifted))

    # THE WARNING THAT MATTERS. Silent pose-baking is what this whole block exists to stop, so a
    # mesh whose deforming armature was left out of the file is named, not left for the caller to
    # discover on import.
    if "MESH" in args["object_types"]:
        out.update(_tangent_report(targets, args, path))
    if anims_only:
        out["animationsOnly"] = True
    if bone_parented:
        out["boneParented"] = bone_parented
        out["boneParentedWarning"] = (
            "%s %s parented to a bone or held by a Child Of constraint, not skinned - the engine's FBX importer can drop "
            "such a mesh without a word. Skin it to the armature (set_parent ARMATURE_AUTO, or rigid weights with "
            "set_vertex_weights exclusive) to keep it." % (", ".join(bone_parented), "is" if len(bone_parented) == 1
                                                         else "are"))
    if unexported_deformers:
        out["deformerNotExportedWarning"] = (
            "%s deform%s a mesh in this export and %s NOT written to the file, because objectTypes "
            "is %s. The exporter only preserves the REST POSE when ARMATURE is included, so the "
            "mesh has been written deformed into its CURRENT POSE, as a static mesh. If that was "
            "not intended, re-export with objectTypes:[\"MESH\",\"ARMATURE\"]."
            % (", ".join("'%s'" % a for a in unexported_deformers),
               "s" if len(unexported_deformers) == 1 else "",
               "was" if len(unexported_deformers) == 1 else "were",
               sorted(args["object_types"])))
    if keys_dropped:
        out["shapeKeysDropped"] = sorted(keys_dropped)
        out["shapeKeysDroppedWarning"] = (
            "%s have shape keys and modifiers the export applied (%s), and the FBX exporter writes NO shape keys for a "
            "mesh whose modifiers it applies - measured, every key was gone on re-import. Re-export with "
            "useMeshModifiers:false, apply the modifiers first (apply_modifier asShapeKey keeps a deforming result as "
            "a key), or include the rig with objectTypes [\"MESH\", \"ARMATURE\"] when an Armature is the only one."
            % (", ".join("'%s'" % n for n in sorted(keys_dropped)),
               "; ".join("%s: %s" % (n, ", ".join(m)) for n, m in sorted(keys_dropped.items()))))
    if armatures and args.get("add_leaf_bones"):
        out["leafBoneWarning"] = (
            "addLeafBones is on. Blender appends a synthetic tip bone to every chain end so the FBX "
            "can record bone length; Unreal has no concept of one and imports each as a REAL extra "
            "bone named <parent>_end, which then appears in every retarget chain and anim asset "
            "built on this skeleton. It is off by default here for that reason.")
    if facing_reply is not None:
        out["facingBody"] = facing_reply
    if split_report:
        out["shapeCornersSplit"] = split_report
        out["shapeCornersNote"] = (
            "a keyed mesh's points were split at its hard corners in the file (pointsBefore -> pointsAfter; the scene is "
            "unchanged). Unreal reads no shape normals from an FBX and computes a morph target's normal change per point, "
            "from one face at each - so without the split a hard-cornered key arrives with the wrong normals or none. "
            "Unreal splits render vertices at hard corners anyway; splitShapeCorners false writes the points unsplit.")
    elif keyed and not split_corners:
        out["shapeCornersNote"] = (
            "splitShapeCorners false: Unreal computes a morph target's normal change per point, from one face at each, so "
            "a hard-cornered key arrives with that face's change at every corner of the point (or none).")
    return out


# ---------------------------------------------------------------------------
# bevel_edges
# ---------------------------------------------------------------------------

_BEVEL_KEYS = {
    "object", "name",
    # selection
    "minAngleDeg", "maxAngleDeg", "axis", "side", "tolerance",
    "boundaryOnly", "edgeIndices", "allEdges",
    # geometry
    "offset", "offsetUU", "offsetType", "segments", "profile", "clampOverlap",
    "loopSlide", "hardenNormals", "miterOuter", "miterInner", "spread",
    # the hard-surface finish (1.4)
    "finish",
    # safety
    "preserveAxes", "preserveTolerance", "assertAxes", "assertTolerance",
    "seamTolerance", "seamBand", "dryRun", "previewId", "previewTtlSeconds"}
_BEVEL_FLAT_DEG = 0.5
_BEVEL_SELECTOR_KEYS = ("minAngleDeg", "maxAngleDeg", "axis", "boundaryOnly", "edgeIndices", "allEdges")
_FINISH_KEYS = ("minAngleDeg", "triangleBudget")
# The finish's sharp edges when no selector is given: faces meeting 30 degrees or more off flat.
_FINISH_MIN_ANGLE_DEG = 30.0


def _finish_spec(params):
    """bevel_edges finish: None when not asked, else {minAngleDeg, triangleBudget}. Refused before anything is touched:
    an unknown key, hardenNormals false beside it, a minAngleDeg beside the caller's own selector."""
    raw = params.get("finish")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        if not take_bool({"finish": raw}, "finish"):
            return None
        raw = {}
    unknown = sorted(str(k) for k in raw if k not in _FINISH_KEYS)
    if unknown:
        raise MifOpError("finish takes %s; %s %s not among them. NOTHING was changed."
                         % (", ".join(_FINISH_KEYS), ", ".join(unknown), "is" if len(unknown) == 1 else "are"))
    # Read before the test, so a malformed value refuses whatever else is in the call; absent is not false.
    if not take_bool(params, "hardenNormals", default=True):
        raise MifOpError("finish hardens the bevel's normals - that is half of what it is - so hardenNormals false "
                         "beside it is a contradiction. NOTHING was changed.")
    angle = raw.get("minAngleDeg")
    own = [k for k in _BEVEL_SELECTOR_KEYS if params.get(k) not in (None, False)]
    if angle is not None:
        if own:
            raise MifOpError("finish.minAngleDeg picks the sharp edges when no selector is given; %s already select%s "
                             "them. Pass one. NOTHING was changed." % (", ".join(own), "" if len(own) > 1 else "s"))
        angle = finite_float(angle, "finish.minAngleDeg")
        if not 0.0 < angle < 180.0:
            raise MifOpError("finish.minAngleDeg is the angle between two faces in DEGREES, above 0 and under 180, "
                             "got %r. NOTHING was changed." % raw.get("minAngleDeg"))
    budget = raw.get("triangleBudget")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < 1):
        raise MifOpError("finish.triangleBudget is a whole number of triangles, 1 or more, got %r. NOTHING was "
                         "changed." % (budget,))
    return {"minAngleDeg": None if own else (angle if angle is not None else _FINISH_MIN_ANGLE_DEG),
            "triangleBudget": budget, "ownSelector": own}


def _flat_normal_spread(obj, new_faces):
    """How far each face's shading normals lean off its own normal, on the object as it EVALUATES (the Weighted Normal
    modifier included): the largest over the FLAT faces - faces the bevel did not make whose every other neighbor
    across an edge lies within a degree of their plane (a cube's sides, a cylinder's caps; a cylinder's sides curve
    and should lean) - and the mean over the faces the bevel made (they should blend, so their edges catch light).
    Degrees."""
    from .ops_kit import _corner_normals     # noqa: PLC0415 - the one 3.6/4.1 corner-normal reader
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        corners = _corner_normals(me)
        by_edge = {}
        for p in me.polygons:
            for k in p.edge_keys:
                by_edge.setdefault(k, []).append(p.index)
        polys = me.polygons
        flat_max, flat_bent, flat_n, bevel_sum, bevel_n = 0.0, 0, 0, 0.0, 0
        for p in polys:
            fn = p.normal
            worst = 0.0
            for li in p.loop_indices:
                c = Vector(corners[li])
                if c.length > 1e-12:
                    worst = max(worst, math.degrees(fn.angle(c.normalized(), 0.0)))
            if p.index in new_faces:
                bevel_sum += worst
                bevel_n += 1
                continue
            flat = all(q == p.index or q in new_faces or math.degrees(fn.angle(polys[q].normal, 0.0)) < 1.0
                       for k in p.edge_keys for q in by_edge.get(k, ()))
            if flat:
                flat_n += 1
                flat_max = max(flat_max, worst)
                flat_bent += worst > 1.0
        return {"flatFacesMaxLeanDeg": round(flat_max, 3) if flat_n else None, "flatFacesLeaningOver1Deg": flat_bent,
                "flatFaces": flat_n,
                "bevelFacesMeanLeanDeg": round(bevel_sum / bevel_n, 3) if bevel_n else None,
                "bevelFaces": bevel_n, "facesEvaluated": len(polys)}
    finally:
        ev.to_mesh_clear()


def _bevel_finish(obj, spec, new_faces, tris_before, tris_after):
    """After the bevel is written: every face smooth and a Weighted Normal modifier through set_shading (on a build
    before 4.1 auto smooth goes on at 180 degrees, which is what lets custom normals take effect there), then the
    shading measured and the triangles held to the budget."""
    from .ops_query import op_set_shading    # noqa: PLC0415 - set_shading owns the Weighted Normal modifier
    shade = {"object": obj.name, "smooth": True, "weightedNormals": True, "keepSharp": True}
    if hasattr(obj.data, "use_auto_smooth"):
        shade["autoSmoothAngle"] = 180.0
    shading = op_set_shading(shade)
    wn = obj.modifiers.get(((shading.get("weightedNormals") or {}).get("modifier")) or "")
    influence_was = None
    if wn is not None and hasattr(wn, "use_face_influence"):
        influence_was = bool(wn.use_face_influence)
        wn.use_face_influence = True
    given = spec["triangleBudget"]
    budget_from = {"from": "given", "triangleBudget": given} if given is not None else triangle_budget_for_size(obj)
    budget = (budget_from or {}).get("triangleBudget")
    added = tris_after - tris_before
    return {
        "hardenNormals": True,
        "weightedNormals": dict(shading.get("weightedNormals") or {},
                                faceInfluence=bool(getattr(wn, "use_face_influence", False)),
                                faceInfluenceWas=influence_was),
        "faceStrength": "AFFECTED - the faces the bevel touched weigh less than the flat faces beside them",
        "weightedNormalsMoved": shading.get("weightedNormalsMoved"),
        "autoSmoothVia": shading.get("autoSmoothVia"),
        "smoothFaces": shading.get("smoothFaces"),
        "normals": _flat_normal_spread(obj, new_faces),
        "triangles": {"before": tris_before, "after": tris_after, "added": added, "budget": budget,
                      "budgetFrom": budget_from,
                      "addedShareOfBudget": round(added / budget, 4) if budget else None,
                      "withinBudget": (tris_after <= budget) if budget else None},
        "note": "flatFacesMaxLeanDeg near 0 means the faces the bevel did not make shade flat; bevelFacesMeanLeanDeg "
                "above 0 means the bevel's faces blend between their neighbors, which is what catches the light",
    }


# SELECT BY TRAIT (2026-10-06): mesh_quality lists only the first 50 non-manifold or loose edges, and nothing listed
# interior ones. NON_MANIFOLD is bmesh's own is_contiguous turned round: not exactly two faces wound consistently - a
# boundary, a wire edge, a crowded edge, or two faces whose normals flip across it, as Blender's Select Non-Manifold.
EDGE_TRAITS = ("NON_MANIFOLD", "INTERIOR", "LOOSE", "BOUNDARY", "DEGENERATE")
_DEGENERATE_EDGE_LENGTH = 1e-6     # world units: a zero-length edge, below float noise at any sane scale
_EDGE_TRAIT_TESTS = {
    "NON_MANIFOLD": lambda e, mw: not e.is_contiguous,
    "INTERIOR": lambda e, mw: len(e.link_faces) == 2,
    "LOOSE": lambda e, mw: not e.link_faces,
    "BOUNDARY": lambda e, mw: len(e.link_faces) == 1,
    "DEGENERATE": lambda e, mw: (mw @ e.verts[0].co - mw @ e.verts[1].co).length < _DEGENERATE_EDGE_LENGTH,
}


def _select_edges(bm, obj, params, op_name="this op"):
    """Build the edge set. Every supplied criterion is ANDed.

    Two selectors matter for real work:
      * by angle   -- minAngleDeg/maxAngleDeg, the angle between the two faces
                      an edge joins (0 = coplanar, 90 = a box corner). This is
                      the "chamfer everything sharper than X" selector.
      * by extreme -- axis + side, i.e. edges lying in the object's min or max
                      plane along an axis. This is the "min/max Z" selector.
    """
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    # A LOOP OR A RING FROM A SEED EDGE (2026-10-04): the ring round a cup could not be named without computing it
    # client side. seedEdge with loop:true or ring:true, alone - ops_topology walks them (edge_loop, edge_ring).
    seed = params.get("seedEdge")
    want_loop = take_bool(params, "loop", default=False)
    want_ring = take_bool(params, "ring", default=False)
    if seed is not None or want_loop or want_ring or params.get("stopAt") is not None:
        if seed is None or want_loop == want_ring:
            raise MifOpError("seedEdge goes with exactly one of loop:true or ring:true (got seedEdge %r, loop %s, ring "
                             "%s)." % (seed, want_loop, want_ring))
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < len(bm.edges):
            raise MifOpError("seedEdge %r is not an edge index of '%s' (it has %d)." % (seed, obj.name, len(bm.edges)))
        others = [k for k in ("edgeIndices", "minAngleDeg", "maxAngleDeg", "axis", "boundaryOnly", "allEdges", "trait")
                  if params.get(k) not in (None, False)]
        if others:
            raise MifOpError("seedEdge selects on its own; drop %s." % ", ".join(others))
        # STOPS (2026-10-06): stopAt ends the walk at a seam, a sharp edge or a material change, and a boundary loop at
        # a corner or an n-gon - the stops 5.1's loop select offers, walked here on every build.
        stop_at = params.get("stopAt")
        if stop_at is None:
            stop_at = []
        if not isinstance(stop_at, list) or any(not isinstance(x, str) or x not in _STOP_AT for x in stop_at):
            raise MifOpError("stopAt must be a list of %s (got %r)." % (", ".join(sorted(_STOP_AT)), stop_at))
        corner = sorted(set(stop_at) & {"INNER_CORNER", "OUTER_CORNER", "NGON"})
        if corner and (want_ring or len(bm.edges[seed].link_faces) != 1):
            raise MifOpError("stopAt %s apply to a loop along a boundary; seedEdge %d %s." % (
                ", ".join(corner), seed, "walks a ring" if want_ring else "is not a boundary edge"))
        from .ops_topology import edge_loop, edge_ring   # noqa: PLC0415 - the walkers loop_cut uses
        if want_loop:
            found, closed = edge_loop(bm.edges[seed], frozenset(stop_at))
        else:
            pairs, closed = edge_ring(bm.edges[seed], frozenset(stop_at))
            found = [next(e for e in a.link_edges if e.other_vert(a) is b) for a, b in pairs]
        return found, {"seedEdge": seed, "loop" if want_loop else "ring": len(found), "closed": closed,
                       "stopAt": sorted(set(stop_at))}

    explicit = params.get("edgeIndices")
    trait = take(params, "trait", kind=str)
    if trait is not None:
        trait = trait.upper()
        if trait not in EDGE_TRAITS:
            raise MifOpError("trait must be one of %s, got %r. NOTHING was changed." % (", ".join(EDGE_TRAITS), trait))
        if explicit is not None:
            raise MifOpError("edgeIndices names the edges itself; trait selects them - pass one. NOTHING was changed.")
    if explicit is not None:
        if not isinstance(explicit, list):
            raise MifOpError("'edgeIndices' must be a list of integers")
        count = len(bm.edges)
        bad = [i for i in explicit if not isinstance(i, int) or i < 0 or i >= count]
        if bad:
            raise MifOpError("edgeIndices out of range for '%s' (%d edges): %s"
                             % (obj.name, count, bad[:10]))
        return [bm.edges[i] for i in explicit], {"edgeIndices": len(explicit)}

    min_angle = take_float(params, "minAngleDeg")
    max_angle = take_float(params, "maxAngleDeg")
    axis_name = take(params, "axis")
    boundary_only = take_bool(params, "boundaryOnly", default=False)
    all_edges = take_bool(params, "allEdges", default=False)

    if not any((min_angle is not None, max_angle is not None, axis_name,
                boundary_only, all_edges, trait)):
        # NAMES THE OP THE CALLER ACTUALLY INVOKED. This helper is shared by THREE ops -
        # select_edges, bevel_edges and extrude_skirt - and the message hardcoded "bevel_edges"
        # for all of them. Calling extrude_skirt without a selector sent you to investigate an
        # op you had never called, which is a wrong answer wearing the clothes of a helpful one.
        # Found by driving a real UE-exported FBX through the round trip on 4.4 and 5.0.
        raise MifOpError(
            "%s needs a selector, and refuses to guess. Use minAngleDeg / "
            "maxAngleDeg (angle between the two faces an edge joins), or axis + side "
            "(e.g. axis:'Z', side:'max' for the top edges), or boundaryOnly:true, or trait, or "
            "edgeIndices:[...], or allEdges:true to really mean every edge."
            % op_name)

    criteria = {}
    candidates = list(bm.edges)

    if boundary_only:
        candidates = [e for e in candidates if len(e.link_faces) == 1]
        criteria["boundaryOnly"] = len(candidates)

    if trait:
        test = _EDGE_TRAIT_TESTS[trait]
        mw = obj.matrix_world
        candidates = [e for e in candidates if test(e, mw)]
        criteria["trait"] = {"trait": trait, "matched": len(candidates)}

    if min_angle is not None or max_angle is not None:
        lo = math.radians(min_angle) if min_angle is not None else -1.0
        hi = math.radians(max_angle) if max_angle is not None else math.pi + 1.0
        kept = []
        for edge in candidates:
            angle = edge.calc_face_angle(None)  # None for wire/boundary/non-manifold
            if angle is None:
                continue
            if lo <= angle <= hi:
                kept.append(edge)
        candidates = kept
        criteria["angle"] = {"minDeg": min_angle, "maxDeg": max_angle,
                             "matched": len(candidates)}

    if axis_name:
        idx = axis_index(axis_name)
        side = str(take(params, "side", default="both")).lower()
        if side not in ("min", "max", "both"):
            raise MifOpError("'side' must be 'min', 'max' or 'both' (got %r)" % side)
        tol = take_float(params, "tolerance", default=1e-4)
        coords = [v.co[idx] for v in bm.verts]
        lo_val, hi_val = min(coords), max(coords)

        def at_extreme(edge):
            a, b = edge.verts[0].co[idx], edge.verts[1].co[idx]
            if side in ("min", "both") and abs(a - lo_val) <= tol and abs(b - lo_val) <= tol:
                return True
            if side in ("max", "both") and abs(a - hi_val) <= tol and abs(b - hi_val) <= tol:
                return True
            return False

        candidates = [e for e in candidates if at_extreme(e)]
        criteria["extreme"] = {"axis": axis_name.upper(), "side": side,
                               "toleranceBU": tol,
                               "minBU": round(lo_val, 6), "maxBU": round(hi_val, 6),
                               "matched": len(candidates)}

    if all_edges and not criteria:
        criteria["allEdges"] = len(candidates)

    return candidates, criteria


def _axis_list(params, key):
    """Accept ["X","Z"] or the forgiving "X" / "xz" spellings."""
    raw = take(params, key, default=[]) or []
    if isinstance(raw, str):
        raw = list(raw) if len(raw) > 1 else [raw]
    if not isinstance(raw, list):
        raise MifOpError("'%s' must be a list of axis names, e.g. [\"X\"]" % key)
    return [axis_index(a, key) for a in raw]


def _axis_sizes(bm):
    if not bm.verts:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    lo = [min(v.co[i] for v in bm.verts) for i in range(3)]
    hi = [max(v.co[i] for v in bm.verts) for i in range(3)]
    return tuple(lo), tuple(hi), tuple(hi[i] - lo[i] for i in range(3))


# ---------------------------------------------------------------------------
# SEAM PLANARITY -- the measurement a bounding-box extent cannot make
# ---------------------------------------------------------------------------
#
# WHY THE EXTENT CHECK IS NOT ENOUGH. sizeBeforeBU/sizeAfterBU and assertAxes are
# built on _axis_sizes, which is a bounding box. A bounding box is pinned by its
# EXTREMES, so as long as ONE vertex survives on each seam plane the reported
# size is bit-identical no matter what the rest of the end cap did.
#
#   MEASURED on Blender 4.4.0 headless. A 1000 x 300 x 50 uu tile (10 x 3 x 0.5
#   BU), bevel of the Y-extreme edge loops, offset 15 uu, guards OFF:
#
#     verts 8 -> 32     sizeDeltaUU [0.0, 0.0, 0.0]      <- extent says CLEAN
#     verts inside the near-but-off band on X:  0 -> 24   <- 24 verts 15 uu in
#
#   assertAxes:["X"] passes that. The tile ships, and every instance along the
#   spline shears at the join. With preserveAxes:["X"] the same run snaps 24
#   verts back and the band count returns to 0 -- so the guard WORKS, it was
#   just never measuring the thing that fails.
#
# WHAT IS MEASURED, per axis, and reported unconditionally:
#
#   bandVerts        verts lying strictly INSIDE (lo, lo+band) or (hi-band, hi):
#                    off the seam plane, but close enough that the preserve snap
#                    would have caught them. An increase IS the shear. This is
#                    the primary detector.
#   movedOffSeam     vertices that were ON the plane before and, having SURVIVED
#                    the op, are now further than `band` from it. Identity, by
#                    BMVert reference. Catches a survivor that drifted clean past
#                    the band, which the band cannot see.
#   seamVertsRemoved tracked seam verts the op destroyed (BMVert.is_valid False).
#                    Reported, never failed on: MEASURED, the bevel above removes
#                    all 8 originals and rebuilds 8 in the same places, which is
#                    correct behaviour. It is here so `movedOffSeam: 0` is never
#                    read as "all the originals are fine" when they are gone.
#   onSeamBefore/After  population of the plane, for reading the other three.
#
# BAND WIDTH. The band must be as wide as the lateral distance the op can drag a
# seam vertex, which is the same number preserveTolerance defaults to: a bevel
# pulls the seam in by its offset, so the band is the offset; a skirt moves
# nothing in X or Y at all, so it is an epsilon. Both are overridable with
# `seamBand`. A band NARROWER than the snap tolerance is the blind spot again --
# it would pass verts the snap itself considers on-seam.


# THE TAG, AND THE HOLE IT CLOSES (2026-09-16, reported by the second seat). Both triggers
# above could be walked past by a LARGE edit. A cube, the eight edges bounding its two X faces,
# assertAxes X,Y, seamBand 0.02: offsets 0.02-0.14 were refused and 0.18 / 0.22 / 0.30 PASSED with
# the bounding box unchanged - the unsafe direction. bmesh.ops.bevel DESTROYS the seam verts and
# builds replacements (seamVertsRemoved says so), so identity has nothing left to follow, and the
# replacements landed further off the plane than the band, so the band did not count them either.
#
# A CUSTOM-DATA TAG FOLLOWS WHAT IDENTITY CANNOT. Each seam vert gets -1 / +1 (lo / hi plane) in a
# temporary int layer per axis before the edit; bmesh carries vertex data onto the verts an op
# builds from a tagged one. MEASURED on 3.6.23, 4.4.0 and 5.2.1 with bmesh.ops.bevel: the seat's
# case goes 8 tagged verts -> 16, and exactly 8 sit off the plane at a distance EQUAL TO THE OFFSET
# for every offset 0.02-0.30; the four edges PARALLEL to X - a chamfer that moves seam verts within
# the plane and never off it - go 8 -> 16 with 0 off. So `leftSeam` answers "began on the seam, or
# was built from something that did, and is now off it by ANY distance", with no band in it.
#
# THE LAYERS NEVER REACH THE MESH: _seam_verdict removes them, and every caller runs the verdict
# before bm.to_mesh. A dry run returns before the verdict and frees the bmesh without writing it.
_SEAM_TAG_LAYER = "_mif_seam_tag_%s"


def _seam_snapshot(bm, lo, hi, axes, on_tol, band):
    """Record which verts sit on each seam plane - BY IDENTITY, and by a tag that
    survives an op rebuilding them - plus the band count.

    Called before the edit. `axes` is a list of axis indices; pass all three so
    the report exists whether or not the axis is guarded.
    """
    # EVERY LAYER BEFORE ANY ROW. Adding (or removing) a VERTEX custom-data layer invalidates every
    # BMVert wrapper Python already holds - MEASURED on 3.6.23, 4.4.0 and 5.2.1: a vert taken before
    # layers.int.new() reads is_valid False after it, while BMEdge and BMFace wrappers survive. The
    # first version made one layer per axis inside the loop, so the X rows were dead by the time Y's
    # layer existed and every X and Y seam vert read as DESTROYED (C108 caught it on all five builds).
    # The same rule binds callers: a BMVert list taken before this call is dead after it.
    names = {}
    for idx in axes:
        names[idx] = _SEAM_TAG_LAYER % "xyz"[idx]
        if bm.verts.layers.int.get(names[idx]) is None:
            bm.verts.layers.int.new(names[idx])
    watch = {}
    for idx in axes:
        name = names[idx]
        layer = bm.verts.layers.int.get(name)
        rows = []
        for vert in bm.verts:
            coord = vert.co[idx]
            if abs(coord - lo[idx]) <= on_tol:
                rows.append((vert, lo[idx]))
                vert[layer] = -1
            elif abs(coord - hi[idx]) <= on_tol:
                rows.append((vert, hi[idx]))
                vert[layer] = 1
            else:
                vert[layer] = 0
        watch[idx] = {"rows": rows, "band": _band_count(bm, lo, hi, idx, band), "layer": name}
    return watch


def _band_count(bm, lo, hi, idx, band):
    """Verts strictly inside the near-but-off band on either seam plane."""
    return sum(1 for vert in bm.verts
               if lo[idx] < vert.co[idx] < lo[idx] + band
               or hi[idx] - band < vert.co[idx] < hi[idx])


def _seam_verdict(watch, bm, lo, hi, on_tol, band):
    """Re-measure after the edit. Returns {axisLetter: {...}} -- pure measurement,
    no policy: which axes are allowed to fail is _seam_violations' business."""
    report = {}
    for idx, before in watch.items():
        moved = removed = 0
        worst = 0.0
        for vert, plane in before["rows"]:
            if not vert.is_valid:
                removed += 1
                continue
            drift = abs(vert.co[idx] - plane)
            if drift > band:
                moved += 1
                worst = max(worst, drift)
        on_after = sum(1 for vert in bm.verts
                       if abs(vert.co[idx] - lo[idx]) <= on_tol
                       or abs(vert.co[idx] - hi[idx]) <= on_tol)
        # THE TAGGED SET, measured without a band: see _SEAM_TAG_LAYER.
        left = 0
        worst_left = 0.0
        layer = bm.verts.layers.int.get(before["layer"]) if before.get("layer") else None
        if layer is not None:
            for vert in bm.verts:
                side = vert[layer]
                if side:
                    drift = abs(vert.co[idx] - (lo[idx] if side < 0 else hi[idx]))
                    if drift > on_tol:
                        left += 1
                        worst_left = max(worst_left, drift)
        report["XYZ"[idx]] = {
            "onSeamBefore": len(before["rows"]),
            "onSeamAfter": on_after,
            "bandVertsBefore": before["band"],
            "bandVertsAfter": _band_count(bm, lo, hi, idx, band),
            "movedOffSeam": moved,
            "seamVertsRemoved": removed,
            "maxSeamDriftBU": round(worst, 8),
            "maxSeamDriftUU": round(worst * UU_PER_BU, 6),
            "leftSeam": left,
            "maxLeftSeamDriftBU": round(worst_left, 8),
            "maxLeftSeamDriftUU": round(worst_left * UU_PER_BU, 6),
        }
    # REMOVED HERE, so no caller can write them into the mesh: every caller runs this before to_mesh.
    for before in watch.values():
        name = before.get("layer")
        layer = bm.verts.layers.int.get(name) if name else None
        if layer is not None:
            bm.verts.layers.int.remove(layer)
    return report


def _seam_violations(report, guarded_axes):
    """The subset of `report` that is a hard failure on a GUARDED axis.

    Two triggers, both measured rather than inferred:
      bandVertsAfter > bandVertsBefore -- verts now sit off the plane but inside
        the snap band. This is the 24-of-32 case above.
      movedOffSeam > 0 -- a vert that was on the plane survived and left it by
        more than the band, which is too far for the band to see.
    """
    bad = []
    for letter in guarded_axes:
        row = report.get(letter)
        if row is None:
            continue
        if row["bandVertsAfter"] > row["bandVertsBefore"]:
            bad.append("%s: %d vert(s) now sit off the seam plane but inside the snap "
                       "band (was %d)" % (letter, row["bandVertsAfter"],
                                          row["bandVertsBefore"]))
        # leftSeam CONTAINS movedOffSeam (a survivor past the band is also off the plane), so the
        # identity line is only written when the tag found nothing - which means the layer did not
        # carry, and the older measurement is then the only one there is.
        if row.get("leftSeam"):
            bad.append("%s: %d vert(s) that began on the seam plane - or were built from one - now "
                       "sit off it, up to %.4f uu, however far past the band"
                       % (letter, row["leftSeam"], row["maxLeftSeamDriftUU"]))
        elif row["movedOffSeam"]:
            bad.append("%s: %d vert(s) that were ON the seam plane moved off it (max "
                       "%.4f uu)" % (letter, row["movedOffSeam"], row["maxSeamDriftUU"]))
    return bad


def _off_seam_verts(report):
    """The single flat number per axis, reported the way sizeDeltaBU is."""
    return {letter: row["bandVertsAfter"] for letter, row in report.items()}


def _seam_warnings(report, guarded, criteria, band):
    """Planarity warnings for axes that are NOT guarded. Deliberately quiet.

    Two suppressions, because a warning nobody reads is worse than no warning:

      * the SELECTOR's own axis. `axis:'Y'` means "bevel the Y extremes", so
        taking verts off the Y seam plane is the request, not a surprise.
      * every unguarded axis, once ANY axis is guarded. Passing assertAxes:['X']
        is the caller stating which axis tiles; warning about Y and Z after that
        is guessing on their behalf. MEASURED: without this, the ordinary
        preserve_x bevel of a closed tile emits two warnings that are true,
        irrelevant and identical every single time.

    Nothing is hidden by either: seamPlanarity and offSeamVerts carry the raw
    per-axis numbers on every response, guarded or not.
    """
    if guarded:
        return []
    sel_axis = (criteria.get("extreme") or {}).get("axis")
    out = []
    for letter in ("X", "Y", "Z"):
        row = report.get(letter)
        if row is None or letter == sel_axis:
            continue
        if row["bandVertsAfter"] > row["bandVertsBefore"]:
            out.append(
                "seam planarity on %s degraded: %d -> %d vert(s) now sit off the %s "
                "min/max plane but within %.4f uu of it, while the SIZE along %s is "
                "UNCHANGED. No extent check can ever see this, and a tile like it shears "
                "at every join. No axis was guarded on this call -- add %s to assertAxes "
                "(preserveAxes follows it automatically and snaps such verts back)."
                % (letter, row["bandVertsBefore"], row["bandVertsAfter"], letter,
                   band * UU_PER_BU, letter, letter))
    return out


def _guard_axes(params, default_band):
    """assertAxes / preserveAxes / the seam tolerances, resolved TOGETHER.

    preserveAxes DEFAULTS TO assertAxes when it is not supplied. Asserting an
    axis without preserving it is not a configuration, it is a guaranteed
    failure: the assert measures exactly the drift the preserve exists to
    remove. Passing preserveAxes:[] explicitly still means "none" -- the default
    only applies when the key is absent.
    """
    assert_axes = _axis_list(params, "assertAxes")
    if "preserveAxes" in params and params["preserveAxes"] is not None:
        preserve = _axis_list(params, "preserveAxes")
    else:
        preserve = list(assert_axes)
    snap_tol = take_float(params, "preserveTolerance", default=default_band)
    on_tol = take_float(params, "seamTolerance", default=1e-4)
    band = take_float(params, "seamBand", default=max(snap_tol, on_tol))
    return assert_axes, preserve, snap_tol, on_tol, band


# ---------------------------------------------------------------------------
# select_edges  -- the selector, resolved and REPORTED, nothing written
# ---------------------------------------------------------------------------

_SELECT_KEYS = {
    "object", "name",
    # selection -- byte-identical grammar to bevel_edges / extrude_skirt
    "minAngleDeg", "maxAngleDeg", "axis", "side", "tolerance",
    "boundaryOnly", "edgeIndices", "allEdges",
    "maxReported",
    # a loop or a ring walked from one edge (2026-10-04), and where it stops (2026-10-06)
    "seedEdge", "loop", "ring", "stopAt",
    # a trait, the full list (2026-10-06)
    "trait",
}
_STOP_AT = {"SEAM", "SHARP", "MATERIAL", "INNER_CORNER", "OUTER_CORNER", "NGON"}


def op_select_edges(params):
    """Resolve an edge selector against a mesh and report what it matches.

    READ-ONLY: the bmesh is never written back. This runs the SAME _select_edges
    the two editing ops run -- not a reimplementation -- so what it reports is
    exactly what bevel_edges / extrude_skirt would act on. Run it first: a
    selector that matches nothing is the commonest reason an edit "did nothing".
    """
    reject_unknown(params, _SELECT_KEYS, "select_edges")
    obj = get_object(take(params, "object", "name", required=True, kind=str),
                     want_mesh=True)
    limit = take_int(params, "maxReported", default=512)
    if limit < 0:
        raise MifOpError("'maxReported' must be >= 0 (it caps how many edge indices "
                         "come back, 0 for all of them; the count is always exact)")

    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        edges, criteria = _select_edges(bm, obj, params, "select_edges")
        lo0, hi0, size0 = _axis_sizes(bm)

        boundary = sum(1 for e in edges if len(e.link_faces) == 1)
        wire = sum(1 for e in edges if len(e.link_faces) == 0)
        interior = len(edges) - boundary - wire
        indices = [e.index for e in edges]
        total = len(bm.edges)

        warnings = []
        if not edges:
            warnings.append(
                "the selector matched 0 of %d edges on '%s'. bevel_edges and "
                "extrude_skirt both REFUSE on an empty selection rather than "
                "no-op, so fix the selector here before calling either."
                % (total, obj.name))
        if interior or wire:
            warnings.append(
                "%d interior and %d wire edge(s) are in this selection. "
                "extrude_skirt refuses those (extruding an interior edge splits "
                "the mesh instead of adding a skirt) -- add boundaryOnly:true."
                % (interior, wire))

        # THE FACES OF A RING (2026-10-05): the row of faces between a ring's edges - each quad the walk crossed - which
        # is what a separate or a delete by faces takes (an iris row split from an eyeball). The walk returns its edges
        # in order, so each face is the one two neighbors share; a closed ring also has one between its ends.
        ring_faces = None
        if criteria.get("ring") is not None:
            ring_faces = []
            pairs = list(zip(edges, edges[1:]))
            if criteria.get("closed") and len(edges) > 2:
                pairs.append((edges[-1], edges[0]))
            for a, b in pairs:
                ring_faces.extend(sorted(f.index for f in set(a.link_faces) & set(b.link_faces)))
            ring_faces = list(dict.fromkeys(ring_faces))
        cap = limit or None
        # THE VERTICES OF A TRAIT (2026-10-06): the stray non-manifold vertices of a scan are what a merge takes. For
        # NON_MANIFOLD they also take the PINCHED vertices - where two fans of faces touch at one point, bmesh's
        # is_manifold false, which no edge shows (MEASURED: is_manifold is true on an open mesh's rim, so it alone
        # would miss every boundary vertex Blender's Select Non-Manifold takes).
        trait_verts = None
        if criteria.get("trait"):
            on_edges = {v.index for e in edges for v in e.verts}
            pinched = None
            if criteria["trait"]["trait"] == "NON_MANIFOLD":
                pinched = sorted(v.index for v in bm.verts if not v.is_manifold and v.link_faces)
                on_edges |= set(pinched)
            on_edges = sorted(on_edges)
            trait_verts = {"count": len(on_edges), "indices": on_edges[:cap]}
            if pinched is not None:
                trait_verts["pinched"] = {"count": len(pinched), "indices": pinched[:cap]}
        return {
            "object": obj.name,
            "count": len(edges),
            "ringFaces": ring_faces,
            "totalEdges": total,
            "boundaryEdges": boundary,
            "interiorEdges": interior,
            "wireEdges": wire,
            "edgeIndices": indices[:cap],
            "edgeIndicesTruncated": len(indices[:cap]) < len(indices),
            "edgeIndicesOmitted": len(indices) - len(indices[:cap]),
            "vertices": trait_verts,
            "criteria": criteria,
            "boundsLocalMinBU": rnd(lo0),
            "boundsLocalMaxBU": rnd(hi0),
            "boundsLocalSizeBU": rnd(size0),
            "boundsLocalSizeUU": rnd([v * UU_PER_BU for v in size0], 4),
            "warnings": warnings,
        }
    finally:
        bm.free()


# ---------------------------------------------------------------------------
# transform_selection -- move existing mesh components without helper objects
# ---------------------------------------------------------------------------

_TRANSFORM_SELECTION_KEYS = {
    "object", "name", "componentType", "indices", "boxMin", "boxMax",
    "operation", "vector", "pivot", "space", "dryRun", "maxReported", "shapeKey",
    "angle", "axis", "falloff", "mask", "distance", "factor", "previewId", "previewTtlSeconds",
}

# PROPORTIONAL FALLOFF, Blender's curves for proportional editing: t = 1 - distance / radius inside the radius, 0
# outside; the result scales a vertex's move. RANDOM is t times a seeded random number.
_FALLOFF_CURVES = {
    "SMOOTH": lambda t: 3.0 * t * t - 2.0 * t * t * t,
    "SPHERE": lambda t: math.sqrt(max(0.0, 2.0 * t - t * t)),
    "ROOT": lambda t: math.sqrt(t),
    "INVERSE_SQUARE": lambda t: t * (2.0 - t),
    "SHARP": lambda t: t * t,
    "LINEAR": lambda t: t,
    "CONSTANT": lambda t: 1.0,
    "RANDOM": None,
}
_FALLOFF_KEYS = {"type", "radius", "connectedOnly", "projectAxis", "seed"}


def _required_vec3(params, key):
    value = params.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise MifOpError("'%s' must be [x,y,z]" % key)
    return Vector(finite_floats(value, key))


def _d3(rows, v):
    """3x3 rows @ v, in Python floats (doubles)."""
    return [rows[r][0] * v[0] + rows[r][1] * v[1] + rows[r][2] * v[2] for r in range(3)]


def _dinv3(m):
    """The inverse of a 3x3 (rows of Python floats) by cofactors, in doubles - mathutils inverts in 32 bits."""
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-300:
        raise MifOpError("the object's transform has zero scale on an axis, so a WORLD move cannot be taken back into "
                         "its local space. NOTHING was changed.")
    return [[(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
            [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
            [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det]]


def _drotation(degrees, axis):
    """A rotation of `degrees` about the unit `axis` (Rodrigues), in doubles. mathutils' Matrix.Rotation is 32-bit: its
    cos 90 degrees is -4.4e-8, not 6e-17, which puts a turned vertex 44 nanometres off the axis it should sit on."""
    t = math.radians(degrees)
    c, s, k = math.cos(t), math.sin(t), 1.0 - math.cos(t)
    x, y, z = (float(axis[0]), float(axis[1]), float(axis[2]))
    return [[c + x * x * k, x * y * k - z * s, x * z * k + y * s],
            [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
            [z * x * k - y * s, z * y * k + x * s, c + z * z * k]]


def _falloff_distances(bm, seeds, place, radius, connected):
    """{vertex: distance to the nearest of `seeds`} for every other vertex closer than `radius`.

    `place` maps a vertex to the point distances are measured between (the requested space, with a projected axis
    zeroed). connected walks the edges (Dijkstra over edge lengths), so only vertices joined to the selection count;
    otherwise the straight distance to the nearest seed, found with a KD tree."""
    seed_set = set(seeds)
    if connected:
        best = {v: 0.0 for v in seeds}
        heap = [(0.0, v.index, v) for v in seeds]
        heapq.heapify(heap)
        while heap:
            d, _i, v = heapq.heappop(heap)
            if d > best.get(v, math.inf):
                continue
            here = place(v)
            for e in v.link_edges:
                o = e.other_vert(v)
                nd = d + (place(o) - here).length
                if nd < radius and nd < best.get(o, math.inf):
                    best[o] = nd
                    heapq.heappush(heap, (nd, o.index, o))
        return {v: d for v, d in best.items() if v not in seed_set}
    # DEFERRED: the offline suites load this module against a stub mathutils (see _crossing_face_pairs).
    from mathutils.kdtree import KDTree   # noqa: PLC0415
    tree = KDTree(len(seeds))
    for i, v in enumerate(seeds):
        tree.insert(place(v), i)
    tree.balance()
    out = {}
    for v in bm.verts:
        if v in seed_set:
            continue
        _co, _i, d = tree.find(place(v))
        if d is not None and d < radius:
            out[v] = d
    return out


def op_transform_selection(params, _plan=None):
    """Translate, scale or rotate selected vertices, edges or faces of an existing mesh in place.

    Components are selected either by explicit indices, or (for vertices) by an inclusive local or
    world-space box. boxMin, boxMax, vector and an explicit pivot are all interpreted in the requested
    space. Edge and face selections transform the union of their vertices. Everything is resolved and
    validated before bmesh is written back, and dryRun returns the exact affected vertices and
    predicted bounds without changing the mesh.

    WITH SHAPE KEYS, WITHOUT shapeKey, this edits the base mesh AND the reference key, and every key
    relative to it moves by the same offset - Blender's edit-mode behaviour. MEASURED 2026-09-17 on
    3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1: a 2 BU cube whose Squash key lifts the top face 1 BU, scaled
    0.5 in Z, came back mesh and Basis z -0.5..0.5 and Squash -0.5..1.5 - the lift kept, not scaled. So
    the second seat's "blending Squash changed nothing" was a key that was a copy of Basis, not an edit
    that missed. shapeKeysNote says this whenever keys exist.

    `shapeKey` EDITS ONE KEY INSTEAD, relative to where that key's vertices are: its positions stand in
    for the mesh's, so box selection, the default pivot and the reported bounds all measure the shape
    being edited, and only that key's selected vertices are written. The reference key is refused - omit
    shapeKey to edit it. A key at value 0, or muted, changes nothing you can see; shapeKeyNote says so.

    ROTATE (2026-10-04) turns the selection by `angle` degrees about `axis` (X, Y, Z or [x,y,z], default Z) through
    `pivot` (default the selection's center), in `space` - a handle face turned step by step, a sleeve end lined up with
    the arm. It takes no vector.

    `falloff` IS PROPORTIONAL EDITING: {type SMOOTH (default) | SPHERE | ROOT | INVERSE_SQUARE | SHARP | LINEAR |
    CONSTANT | RANDOM, radius (required, in `space` units), connectedOnly (distance along edges, and only vertices joined
    to the selection), projectAxis (X, Y or Z: distance measured with that axis left out, so a column of vertices moves
    together), seed (RANDOM)}. Every vertex within radius of the nearest selected one moves too, by the same transform
    scaled by Blender's falloff curve - an icing edge roughened, a region moved inside a shape key that blends into its
    neighbors. `mask` (a vertex group name, or {group, invert}) scales each vertex's move by its weight in the group
    (0 outside it; invert uses 1 - weight), selected and falloff vertices alike. A mask that leaves every vertex at 0 is
    refused. falloffVertexCount and affectedVertexCount say how many moved beyond the selection and in all.

    WORLD SPACE FAR FROM THE ORIGIN IS EXACT (2026-10-04). Every move is computed in doubles as an
    offset - never by carrying a vertex out to its world position and back in 32 bits, which put a vertex at y 240 moved
    by -0.1 at -0.600006 instead of -0.6 - so the same edit at y 240 and at the origin writes the same raw coordinates.

    dryRun:true IS ALSO THE BULK PREVIEW (docs/15 section 5): every affected vertex, its coordinate before and after (the
    shape key's when shapeKey is given), and a previewId that {previewId} alone commits - planned again, refused on any
    difference, read back, and on a mismatch every vertex of the mesh and of each shape key put back as it was.
    """
    reject_unknown(params, _TRANSFORM_SELECTION_KEYS, "transform_selection")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    require_editable(obj, "transform selected components on")
    key_name = take(params, "shapeKey", kind=str)
    key_blocks = obj.data.shape_keys.key_blocks if obj.data.shape_keys is not None else None
    key_block = None
    if key_name is not None:
        if not key_blocks:
            raise MifOpError("'%s' has no shape keys, so there is no '%s' to transform. add_shape_key makes one. "
                             "NOTHING was changed." % (obj.name, key_name))
        key_block = key_blocks.get(key_name)
        if key_block is None:
            raise MifOpError("'%s' has no shape key '%s'. It has: %s. NOTHING was changed."
                             % (obj.name, key_name, ", ".join(kb.name for kb in key_blocks)))
        if key_block == obj.data.shape_keys.reference_key:
            raise MifOpError("'%s' is the reference key of '%s'. Omit shapeKey to edit it: that moves the base mesh and "
                             "'%s' together and carries every relative key along by the same offset, as Blender's edit "
                             "mode does. NOTHING was changed." % (key_block.name, obj.name, key_block.name))

    component = str(take(params, "componentType", default="VERTEX")).upper()
    if component not in ("VERTEX", "EDGE", "FACE"):
        raise MifOpError("'componentType' must be VERTEX, EDGE or FACE (got %r)" % component)
    operation = str(take(params, "operation", required=True, kind=str)).upper()
    if operation not in ("TRANSLATE", "SCALE", "ROTATE", "SHRINK_FATTEN", "TO_SPHERE"):
        raise MifOpError("'operation' must be TRANSLATE, SCALE, ROTATE, SHRINK_FATTEN or TO_SPHERE (got %r)"
                         % operation)
    space = str(take(params, "space", default="LOCAL")).upper()
    if space not in ("LOCAL", "WORLD"):
        raise MifOpError("'space' must be LOCAL or WORLD (got %r)" % space)
    # SHRINK_FATTEN AND TO_SPHERE (2026-10-05): a thumb pad given volume along its normals after a loop cut, a loop
    # made round. Each takes its own amount and nothing else: distance (along each vertex normal, in `space` units),
    # factor (0 to 1, how far toward the sphere).
    shrink_d = sphere_f = None
    if operation in ("SHRINK_FATTEN", "TO_SPHERE"):
        extra = [k for k in ("vector", "angle", "axis") if params.get(k) is not None]
        if extra:
            raise MifOpError("%s takes %s, not %s. NOTHING was changed."
                             % (operation, "distance" if operation == "SHRINK_FATTEN" else "factor", ", ".join(extra)))
        if operation == "SHRINK_FATTEN":
            if params.get("factor") is not None:
                raise MifOpError("factor belongs to TO_SPHERE; SHRINK_FATTEN takes distance. NOTHING was changed.")
            if params.get("distance") is None:
                raise MifOpError("SHRINK_FATTEN needs distance: how far along each vertex normal (negative shrinks). "
                                 "NOTHING was changed.")
            shrink_d = take_float(params, "distance")
            if operation == "SHRINK_FATTEN" and params.get("pivot") is not None:
                raise MifOpError("'pivot' applies to SCALE, ROTATE and TO_SPHERE; SHRINK_FATTEN moves along normals. "
                                 "NOTHING was changed.")
        else:
            if params.get("distance") is not None:
                raise MifOpError("distance belongs to SHRINK_FATTEN; TO_SPHERE takes factor. NOTHING was changed.")
            if params.get("factor") is None:
                raise MifOpError("TO_SPHERE needs factor, 0 (unchanged) to 1 (every vertex on the sphere). NOTHING was "
                                 "changed.")
            sphere_f = take_float(params, "factor")
            if not 0.0 <= sphere_f <= 1.0:
                raise MifOpError("TO_SPHERE factor must be between 0 and 1, got %g. NOTHING was changed." % sphere_f)
    elif params.get("distance") is not None or params.get("factor") is not None:
        raise MifOpError("distance belongs to SHRINK_FATTEN and factor to TO_SPHERE, not %s. NOTHING was changed."
                         % operation)
    angle, turn_axis = None, None
    if operation in ("SHRINK_FATTEN", "TO_SPHERE"):
        vector = Vector((0.0, 0.0, 0.0))
    elif operation == "ROTATE":
        if params.get("vector") is not None:
            raise MifOpError("ROTATE takes angle (degrees) and axis, not vector. NOTHING was changed.")
        angle = take_float(params, "angle", required=True)
        axis_raw = params.get("axis", "Z")
        if isinstance(axis_raw, str) and axis_raw.strip().upper() in ("X", "Y", "Z"):
            turn_axis = Vector([1.0 if axis_raw.strip().upper() == c else 0.0 for c in "XYZ"])
        elif isinstance(axis_raw, (list, tuple)) and len(axis_raw) == 3:
            turn_axis = Vector(finite_floats(axis_raw, "axis"))
            if turn_axis.length < 1e-9:
                raise MifOpError("axis [0,0,0] has no direction. NOTHING was changed.")
        else:
            raise MifOpError("axis must be X, Y, Z or [x,y,z] (got %r). NOTHING was changed." % (axis_raw,))
        turn_axis = turn_axis.normalized()
        vector = Vector((0.0, 0.0, 0.0))
    else:
        if params.get("angle") is not None or params.get("axis") is not None:
            raise MifOpError("angle and axis belong to ROTATE; %s takes vector. NOTHING was changed." % operation)
        vector = _required_vec3(params, "vector")
    if operation == "SCALE" and any(v == 0.0 for v in vector):
        raise MifOpError("SCALE vector components must be non-zero; zero collapses geometry")
    if operation == "TRANSLATE" and params.get("pivot") is not None:
        raise MifOpError("'pivot' applies to SCALE and ROTATE; it would be ignored by TRANSLATE")
    falloff = params.get("falloff")
    curve = fall_type = radius = project_idx = None
    connected, seed = False, 0
    if falloff is not None:
        if not isinstance(falloff, dict) or set(falloff) - _FALLOFF_KEYS:
            raise MifOpError("falloff must be {%s} (got %r). NOTHING was changed."
                             % (", ".join(sorted(_FALLOFF_KEYS)), falloff))
        fall_type = str(falloff.get("type", "SMOOTH")).upper()
        if fall_type not in _FALLOFF_CURVES:
            raise MifOpError("falloff.type must be one of %s (got %r). NOTHING was changed."
                             % (", ".join(_FALLOFF_CURVES), falloff.get("type")))
        radius = take_float(falloff, "radius", required=True)
        if radius <= 0.0:
            raise MifOpError("falloff.radius must be > 0 (got %g). NOTHING was changed." % radius)
        connected = take_bool(falloff, "connectedOnly", default=False)
        if falloff.get("projectAxis") is not None:
            project_idx = axis_index(falloff.get("projectAxis"), "falloff.projectAxis")
        seed = take_int(falloff, "seed", default=0)
        curve = _FALLOFF_CURVES[fall_type]
    mask = params.get("mask")
    mask_group, mask_invert = None, False
    if mask is not None:
        if isinstance(mask, str):
            mask_group = mask
        elif isinstance(mask, dict) and isinstance(mask.get("group"), str) and not set(mask) - {"group", "invert"}:
            mask_group = mask["group"]
            mask_invert = take_bool(mask, "invert", default=False)
        else:
            raise MifOpError("mask must be a vertex group name or {group, invert} (got %r). NOTHING was changed."
                             % (mask,))
        if obj.vertex_groups.get(mask_group) is None:
            raise MifOpError("'%s' has no vertex group '%s'. It has: %s. NOTHING was changed."
                             % (obj.name, mask_group, ", ".join(g.name for g in obj.vertex_groups) or "none"))
    dry_run = take_bool(params, "dryRun", default=False)
    limit = take_int(params, "maxReported", default=512)
    if limit < 0:
        raise MifOpError("'maxReported' must be >= 0")

    explicit = params.get("indices")
    has_box = params.get("boxMin") is not None or params.get("boxMax") is not None
    if (explicit is None and not has_box) or (explicit is not None and has_box):
        raise MifOpError("pass exactly one selector: 'indices', or both 'boxMin' and 'boxMax'")
    if has_box and component != "VERTEX":
        raise MifOpError("box selection applies to VERTEX only; use explicit EDGE/FACE indices")
    if has_box and (params.get("boxMin") is None or params.get("boxMax") is None):
        raise MifOpError("box selection needs both 'boxMin' and 'boxMax'")
    if explicit is not None and not isinstance(explicit, list):
        raise MifOpError("'indices' must be a list of integers")

    mesh = obj.data
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        if key_block is not None:
            # THE KEY'S POSITIONS STAND IN FOR THE MESH'S - box selection, the pivot and the bounds below all measure
            # the shape being edited. Nothing is written back through this bmesh on this path; see the write below.
            for vert in bm.verts:
                vert.co = key_block.data[vert.index].co
        tables = {"VERTEX": bm.verts, "EDGE": bm.edges, "FACE": bm.faces}
        table = tables[component]

        if explicit is not None:
            wrong_type = list(dict.fromkeys(repr(i) for i in explicit if not isinstance(i, int)))
            if wrong_type:
                raise MifOpError("indices for %s selection must be integers; got: %s"
                                 % (component, wrong_type[:10]))
            bad = list(dict.fromkeys(i for i in explicit if i < 0 or i >= len(table)))
            if bad:
                raise MifOpError("indices out of range for %s selection on '%s' (%d components): %s"
                                 % (component, obj.name, len(table), bad[:10]))
            components = [table[i] for i in dict.fromkeys(explicit)]
            criteria = {"componentType": component, "indices": len(components)}
        else:
            box_min = _required_vec3(params, "boxMin")
            box_max = _required_vec3(params, "boxMax")
            if any(box_min[i] > box_max[i] for i in range(3)):
                raise MifOpError("every boxMin component must be <= boxMax")
            matrix = obj.matrix_world if space == "WORLD" else None
            components = []
            for vert in bm.verts:
                co = matrix @ vert.co if matrix is not None else vert.co
                if all(box_min[i] <= co[i] <= box_max[i] for i in range(3)):
                    components.append(vert)
            criteria = {"componentType": component, "boxMin": rnd(box_min),
                        "boxMax": rnd(box_max), "space": space}

        verts = set()
        for item in components:
            if component == "VERTEX":
                verts.add(item)
            else:
                verts.update(item.verts)
        verts = sorted(verts, key=lambda v: v.index)
        if not verts:
            raise MifOpError("the selector matched 0 vertices on '%s'; NOTHING was changed"
                             % obj.name)

        before_all = _axis_sizes(bm)
        before_selected_local = [v.co.copy() for v in verts]
        # EVERY MOVE IN DOUBLES, AND NEVER THROUGH A WORLD COORDINATE (2026-10-04). Mesh coordinates and
        # mathutils are 32-bit: carrying a vertex 240 m out into world space and back cost about 1.5e-5 m, so a WORLD
        # move of a vertex at y 240 by -0.1 landed at -0.600006 local instead of -0.6 (docs/02_GOTCHAS.md). Now the
        # move is worked out as an OFFSET - the vector, or (scale or turn - 1) x the vertex's offset from the pivot, that
        # offset formed as W3 x (local - local centre) or W3 x local + (translation - pivot), so the big translation
        # cancels before it touches a coordinate - taken back through a double-precision inverse of the 3x3, and added
        # to the local coordinate, which is rounded to 32 bits once.
        world = obj.matrix_world
        w3 = [[float(world[r][c]) for c in range(3)] for r in range(3)]
        wt = [float(world[r][3]) for r in range(3)]
        is_world = space == "WORLD"
        inv3 = _dinv3(w3) if is_world else None
        lin = w3 if is_world else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        def local(v):
            return [float(v.co[0]), float(v.co[1]), float(v.co[2])]

        sel_local = [local(v) for v in verts]
        centre = [sum(p[i] for p in sel_local) / len(sel_local) for i in range(3)]
        given_pivot = None
        if operation in ("TRANSLATE", "SHRINK_FATTEN"):
            pivot = None
        elif params.get("pivot") is None:
            # reported in `space`: the centre itself only here, where the size of the number does no harm
            pivot = [a + b for a, b in zip(_d3(w3, centre), wt)] if is_world else centre
        else:
            given_pivot = list(_required_vec3(params, "pivot"))   # finite floats: _required_vec3 checked them
            pivot = given_pivot
        rot = _drotation(angle, turn_axis) if operation == "ROTATE" else None

        def offset(p):
            """p's offset from the pivot in `space` units, the big translation cancelled first."""
            if given_pivot is None:
                return _d3(lin, [p[i] - centre[i] for i in range(3)])
            if is_world:
                q = _d3(w3, p)
                return [q[i] + (wt[i] - given_pivot[i]) for i in range(3)]
            return [p[i] - given_pivot[i] for i in range(3)]

        # TO_SPHERE's radius: the mean distance of the selected vertices from the pivot, in `space` - Blender's own
        # (transform_mode_tosphere: each vertex goes to centre + dir x (its distance x (1 - f) + radius x f)).
        sphere_r = None
        if operation == "TO_SPHERE":
            sphere_r = sum(math.sqrt(sum(c * c for c in offset(p))) for p in sel_local) / len(sel_local)
        if operation == "SHRINK_FATTEN":
            bm.normal_update()
        inv_t = [[inv3[c][r] for c in range(3)] for r in range(3)] if is_world else None

        def move(p, vert=None):
            """The full move of local point p (vertex vert), in LOCAL units."""
            if operation == "TRANSLATE":
                d = [float(vector[0]), float(vector[1]), float(vector[2])]
            elif operation == "SHRINK_FATTEN":
                # along the vertex normal: in WORLD space the normal goes through the inverse transpose, so a scaled
                # object is fattened by `distance` world units along the surface's world normal
                n = [float(vert.normal[0]), float(vert.normal[1]), float(vert.normal[2])]
                if is_world:
                    n = _d3(inv_t, n)
                ln = math.sqrt(sum(c * c for c in n))
                d = [c / ln * shrink_d for c in n] if ln > 0.0 else [0.0, 0.0, 0.0]
            elif operation == "TO_SPHERE":
                q = offset(p)
                ln = math.sqrt(sum(c * c for c in q))
                d = [c * sphere_f * (sphere_r / ln - 1.0) for c in q] if ln > 0.0 else [0.0, 0.0, 0.0]
            else:
                q = offset(p)
                if operation == "SCALE":
                    d = [q[i] * (float(vector[i]) - 1.0) for i in range(3)]
                else:
                    r = _d3(rot, q)
                    d = [r[i] - q[i] for i in range(3)]
            return _d3(inv3, d) if is_world else d

        # THE WEIGHT EACH VERTEX'S MOVE IS SCALED BY: 1 on the selection, the falloff curve beside it, times the mask.
        weights = {v: 1.0 for v in verts}
        if falloff is not None:
            places = {}

            def place(v):
                """Where distances are measured from: `space`'s rotation and scale applied, its translation left off -
                distances do not depend on it, and leaving it off keeps the numbers small."""
                p = places.get(v)
                if p is None:
                    p = Vector(_d3(lin, local(v)))
                    if project_idx is not None:
                        p[project_idx] = 0.0
                    places[v] = p
                return p

            rng = random.Random(seed)
            for v, d in sorted(_falloff_distances(bm, verts, place, radius, connected).items(),
                               key=lambda kv: kv[0].index):
                t = 1.0 - d / radius
                w = rng.random() * t if curve is None else curve(t)
                if w > 0.0:
                    weights[v] = w
        falloff_count = len(weights) - len(verts)
        masked_out = 0
        selected_set = set(verts)
        if mask_group is not None:
            gi = obj.vertex_groups[mask_group].index
            layer = bm.verts.layers.deform.active
            for v in list(weights):
                w = layer is not None and v[layer].get(gi, 0.0) or 0.0
                w = 1.0 - w if mask_invert else w
                if w <= 0.0:
                    masked_out += 1 if v in selected_set else 0
                    del weights[v]
                else:
                    weights[v] *= w
            if not weights:
                raise MifOpError("the mask '%s'%s weights every vertex of this selection%s 0, so nothing would move. "
                                 "NOTHING was changed." % (mask_group, " (inverted)" if mask_invert else "",
                                                           " and its falloff" if falloff is not None else ""))
        affected = sorted(weights, key=lambda v: v.index)
        moved_before = [v.co.copy() for v in affected]
        for vert in affected:
            p = local(vert)
            d = move(p, vert)
            w = weights[vert]
            vert.co = Vector([p[i] + d[i] * w for i in range(3)])
        after_all = _axis_sizes(bm)
        after_selected_local = [v.co.copy() for v in verts]
        moves = any((v.co - b).length > 1e-12 for v, b in zip(affected, moved_before))
        if _plan is not None:
            # THE BULK PREVIEW'S PLAN (_plan_transform_selection): every affected vertex, before and after, from this
            # same computation - so the preview cannot predict a move the op would not make.
            _plan.update(object=obj.name, shapeKey=key_block.name if key_block is not None else None,
                         verts=[(v.index, tuple(b), tuple(v.co)) for v, b in zip(affected, moved_before)])

        result = {
            "object": obj.name,
            "componentType": component,
            "componentCount": len(components),
            "vertexCount": len(verts),
            "vertexIndices": [v.index for v in verts[:limit]],
            "vertexIndicesTruncated": len(verts) > limit,
            "criteria": criteria,
            "operation": operation,
            "space": space,
            "vector": rnd(vector) if operation in ("TRANSLATE", "SCALE") else None,
            "angle": angle,
            "distance": shrink_d,
            "factor": sphere_f,
            "sphereRadius": round(sphere_r, 6) if sphere_r is not None else None,
            "axis": rnd(turn_axis) if turn_axis is not None else None,
            "pivot": rnd(pivot) if pivot is not None else None,
            "falloff": ({"type": fall_type, "radius": radius, "connectedOnly": connected,
                         "projectAxis": "XYZ"[project_idx] if project_idx is not None else None, "seed": seed}
                        if falloff is not None else None),
            "falloffVertexCount": falloff_count,
            "mask": {"group": mask_group, "invert": mask_invert} if mask_group is not None else None,
            "maskedOutSelected": masked_out,
            "affectedVertexCount": len(affected),
            "affectedVertexIndices": [v.index for v in affected[:limit]],
            "boundsBeforeLocalBU": {"min": rnd(before_all[0]), "max": rnd(before_all[1]),
                                    "size": rnd(before_all[2])},
            "boundsAfterLocalBU": {"min": rnd(after_all[0]), "max": rnd(after_all[1]),
                                   "size": rnd(after_all[2])},
            "selectedBeforeLocalBU": [rnd(co) for co in before_selected_local[:limit]],
            "selectedAfterLocalBU": [rnd(co) for co in after_selected_local[:limit]],
            "coordinatesTruncated": len(verts) > limit,
            "dryRun": dry_run,
            # CHANGED MEANS THE MESH CHANGED, so a dry run answers false. It answered the prediction: the second seat
            # read changed:true off a dryRun whose mesh measured 1.5 x 1.5 x 1.5 before and after (mailbox id 49), and
            # every other dryRun op here says "nothing was modified". The prediction is wouldChange, on both paths.
            "changed": (not dry_run) and moves,
            "wouldChange": moves,
        }
        if dry_run:
            result["note"] = "nothing was modified. Drop dryRun to apply."
        result["shapeKey"] = key_block.name if key_block is not None else None
        if key_block is None and key_blocks:
            result["shapeKeysNote"] = (
                "'%s' has %d shape key(s), and this edited the base mesh and the reference key '%s': every key relative "
                "to it moved by the same offset, so each keeps its own difference from the base - measured on every "
                "build. To reshape one key relative to its own vertices, pass shapeKey."
                % (obj.name, len(key_blocks), obj.data.shape_keys.reference_key.name))
        if key_block is not None and (key_block.mute or key_block.value == 0.0):
            result["shapeKeyNote"] = (
                "'%s' is %s, so this edit changes nothing you can see until it %s - set_shape_key."
                % (key_block.name, "muted" if key_block.mute else "at value 0",
                   "is unmuted" if key_block.mute else "is raised"))
        result.update(shared_data_note(obj))
        if not dry_run and key_block is not None:
            for vert in affected:
                key_block.data[vert.index].co = vert.co
            # TAGGED THE WAY set_point_positions TAGS ITS KEY WRITES - a write into key data does not reach an evaluated
            # copy on its own (see the subscript-write note in ops_common).
            mesh.update_tag()
            obj.data.shape_keys.update_tag()
            bpy.context.view_layer.update()
        elif not dry_run:
            bm.to_mesh(mesh)
            mesh.update()
            # THE CACHE REFRESH ops_common.local_bounds SAYS EVERY EDITING OP MAKES. Without it
            # obj.dimensions / obj.bound_box kept the PRE-edit size: the second seat read object_info's
            # dimensionsBU [2,2,2] beside boundsLocalSizeBU [2.8,2.8,0.8] after scaling a sphere's verts
            # (2026-09-16), and test_blender_dimensions_fresh D100 measured it red before this line.
            bpy.context.view_layer.update()
        return result
    finally:
        bm.free()


def _bevel_key(bm):
    """Every vertex position to 6 places, sorted: two bevels that produced the same mesh have the same key."""
    return sorted((round(v.co.x, 6), round(v.co.y, 6), round(v.co.z, 6)) for v in bm.verts)


def _bevel_trial(src, geom_ref, offset, kwargs):
    """The key of a bevel run on a COPY of src - the mesh being edited is never touched."""
    trial = src.copy()
    try:
        trial.verts.ensure_lookup_table()
        trial.edges.ensure_lookup_table()
        geom = [trial.edges[i] if kind == "E" else trial.verts[i] for kind, i in geom_ref]
        bmesh.ops.bevel(trial, geom=geom, offset=offset, **kwargs)
        return _bevel_key(trial)
    finally:
        trial.free()


def _open_flat_spokes(bm, geom, offset, kwargs):
    """Take the edges across a flat face out of a clamped bevel's way, and say how many went.

    A FLAT FACE SPLIT INTO TRIANGLES HOLDS A BEVEL BACK WHERE THE SAME FACE WHOLE DOES NOT. A cube and a tube unioned: 0.3
    round the junction applied 0.3 with the cut's n-gons, and 0.0096 once they were split, the thin triangles clamping it
    (measured on 5.2.1, 2026-11-01). So when the clamp would bite and a flat edge touches the selection, those edges go first
    and the n-gons are split again after the bevel. Only then, because a grid across a flat face can be there on purpose
    (build_kit_piece's, for a vertex-color mask), and never across a material, a seam, a sharp edge or a UV break."""
    selected = {g for g in geom if isinstance(g, bmesh.types.BMEdge)}
    touching = {v for g in geom for v in (g.verts if isinstance(g, bmesh.types.BMEdge) else (g,))}
    uv = bm.loops.layers.uv.active
    flat = math.radians(_BEVEL_FLAT_DEG)

    def whole(e):
        a, b = e.link_faces
        if a.material_index != b.material_index or a.smooth != b.smooth or e.seam or not e.smooth:
            return False
        if uv is None:
            return True
        for v in e.verts:
            at = {f: lp[uv].uv for f in (a, b) for lp in f.loops if lp.vert is v}
            if (at[a] - at[b]).length > 1e-6:
                return False
        return True
    spokes = [e for e in bm.edges if e not in selected and len(e.link_faces) == 2 and (e.verts[0] in touching or e.verts[1] in touching)
              and e.calc_face_angle(math.pi) <= flat and whole(e)]
    if not spokes:
        return 0
    bm.verts.index_update()
    bm.edges.index_update()
    ref = [("E", g.index) if isinstance(g, bmesh.types.BMEdge) else ("V", g.index) for g in geom]
    try:
        if _bevel_trial(bm, ref, offset, dict(kwargs, clamp_overlap=False)) == _bevel_trial(bm, ref, offset, kwargs):
            return 0
    except (ValueError, TypeError, RuntimeError):
        return 0            # the bevel itself refuses these words, and says so where it runs
    before = len(bm.edges)
    bmesh.ops.dissolve_edges(bm, edges=spokes, use_verts=False, use_face_split=False)
    return before - len(bm.edges)


def _bevel_applied_offset(src, geom_ref, offset, kwargs, result_key):
    """(offset applied, clamped?) for a bevel run with clamp_overlap on - (None, None) if a trial fails.

    offsetBU WAS THE OFFSET ASKED FOR, and clamp_overlap - on by default - caps a bevel where the geometry has no room.
    MEASURED 2026-09-16 in plain bmesh, identical on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1: a unit cube's four top
    edges bevelled at 0.6 and 0.9 give exactly the geometry 0.5 gives, and below 0.5 clamped and unclamped results are
    identical; a 32-sided cylinder's rim is unclamped to 0.9. So: the same bevel unclamped on a copy - equal means the
    clamp did nothing; different means it did, and bisection on copies finds the smallest offset giving this result,
    which is the offset applied (0.5 on the cube, found exactly). Second seat, mailbox id 49."""
    try:
        if _bevel_trial(src, geom_ref, offset, dict(kwargs, clamp_overlap=False)) == result_key:
            return offset, False
        lo, hi = 0.0, offset
        for _ in range(20):
            mid = (lo + hi) / 2.0
            if _bevel_trial(src, geom_ref, mid, kwargs) == result_key:
                hi = mid
            else:
                lo = mid
        return hi, True
    except (ValueError, TypeError, RuntimeError):
        return None, None


# A BEVEL CAN PASS THROUGH ITSELF WHILE EVERY GUARD IN op_bevel_edges HOLDS. MEASURED 2026-09-17 through server._execute
# (docs/22 B81): a 2 BU cube with a radius-0.5 tube booleaned onto it, bevel_edges on the tube's rim or on a blind hole's
# floor ring at offsets from 0.5 BU - the tube's radius - up, answered ok with 8 to 590 pairs of faces crossing on 3.6.23
# and 5.2.1. clampOverlap stops it on 3.6.23 and 4.2.17 (0.7 asked on the rim, 0.498 applied, nothing crossing) and does
# NOT from 4.4.0 (0.689 applied, 386 pairs on 4.4.0, 5.0.1 and 5.2.1). Offsets up to 0.35 crossed nothing on either ring
# on all five builds, and the concave fillet rendered clean. The count
# is the mesh's BVH overlap with itself between faces sharing no vertex: 0 for one cube, 6 for two overlapping cubes in
# one mesh.
_CROSSING_FACE_CAP = 250_000


def _crossing_face_pairs(bm):
    """How many pairs of faces pass through each other and share no vertex - or None above _CROSSING_FACE_CAP faces.

    MEASURED COST on a subdivided cube, 3.6.23 and 5.2.1: 0.02 s at 10k faces, 0.2 s at 119k and 0.5 s at 242k with the
    shared-vertex test over every face. It runs here only over the faces the tree reported, which on a clean mesh is none.
    """
    if len(bm.faces) > _CROSSING_FACE_CAP:
        return None
    # DEFERRED: the offline suites load this module against a stub mathutils (see the UV overlap check below).
    from mathutils.bvhtree import BVHTree   # noqa: PLC0415
    bm.verts.index_update()
    bm.faces.ensure_lookup_table()
    tree = BVHTree.FromBMesh(bm, epsilon=0.0)
    pairs = set()
    corners = {}
    for a, b in tree.overlap(tree):
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in pairs:
            continue
        va = corners.get(a)
        if va is None:
            va = corners[a] = {v.index for v in bm.faces[a].verts}
        vb = corners.get(b)
        if vb is None:
            vb = corners[b] = {v.index for v in bm.faces[b].verts}
        if not va & vb:
            pairs.add(key)
    return len(pairs)


def _mesh_shape(mesh):
    """A fingerprint of every vertex position, the sha256 ops_rig._evaluated_counts reports as `shape`. Profile and offset
    move vertices and change no count, so this is how two bevels that differ only in those are told apart."""
    buf = array.array("f", [0.0]) * (3 * len(mesh.vertices))
    mesh.vertices.foreach_get("co", buf)
    return hashlib.sha256(buf.tobytes()).hexdigest()[:16]


def op_bevel_edges(params):
    """Chamfer or round a selected set of edges, with a tiling guard.

    segments=1 gives a flat chamfer; segments>1 gives a rounded fillet.

    THE TILING GUARD. A bevel moves the endpoints of every edge it touches, so
    bevelling the long side edges of a tiling mesh also pulls its end-cap verts
    inward and the tile no longer butts up against the next one. Three knobs:

      preserveAxes: ["X"]  -- after the bevel, snap any vert that ended up near
                              the ORIGINAL min/max along X back onto it.
                              DEFAULTS TO assertAxes when it is not supplied.
      assertAxes:   ["X"]  -- after that, fail and throw the whole edit away if
                              either (a) the SIZE along X moved by more than
                              assertTolerance, or (b) the seam PLANARITY on X
                              broke. Nothing is written.
      seamBand             -- how wide the near-but-off band is. Defaults to the
                              preserve snap tolerance, which for a bevel is the
                              offset: that is exactly how far a bevel can drag a
                              seam vertex.

    (b) is not a refinement of (a), it is the check that (a) cannot make. The
    size along X is a bounding-box EXTENT and one surviving corner vertex pins
    it -- MEASURED, a guards-off bevel on a 1000 uu tile put 24 verts 15 uu
    inside the X seam and still reported sizeDeltaUU [0,0,0]. See the SEAM
    PLANARITY block above for the numbers.

    assertAxes defaults to empty, i.e. off. The response ALWAYS carries
    sizeBeforeBU / sizeAfterBU / sizeDeltaBU AND offSeamVerts / seamPlanarity for
    all three axes, so neither kind of drift can pass unnoticed even with the
    guards off -- off means "do not fail", never "do not look".

    THE RESULT CAN PASS THROUGH ITSELF WHILE ALL OF THAT HOLDS - see _crossing_face_pairs.
    selfIntersectingFacePairs counts faces that cross and share no vertex, with the count
    before the bevel whenever there are any, and selfIntersectionNote says when the bevel
    added them. It is reported, not refused: the asserts above are the caller's to ask for.
    shapeBefore / shapeAfter fingerprint every vertex position, because profile and offset
    move vertices and change no count - the second seat told profile 0.15 / 0.50 / 0.85 apart
    only by a shape hash.

    THE HARD-SURFACE FINISH (1.4): finish true or {minAngleDeg (30), triangleBudget}. With no selector of its own it
    bevels every edge whose faces meet minAngleDeg or more off flat; hardened normals, face strength on the faces the
    bevel touched, every face smooth and a Weighted Normal modifier with face influence (set_shading's) follow, so flat
    faces shade flat and the bevel's faces catch the light with no support loops. The answer's finish block measures
    it on the evaluated mesh - flatFacesMaxLeanDeg (0.000 on a beveled cube on 3.6 and 5.2; 1.64 with the modifier
    alone, 70 with hardened normals alone) and bevelFacesMeanLeanDeg - and holds the triangles to the gameReady budget
    (triangleBudget, else the size class's, triangle_budget_for_size): before, after, added, addedShareOfBudget.
    """
    reject_unknown(params, _BEVEL_KEYS, "bevel_edges", {
        # THE RESPONSE'S FIELD NAMES, sent back as parameters - the second seat's eight refusals in a row (mailbox id 62).
        "offsetBU": "offsetBU is what the RESPONSE reports. The parameter is `offset` (Blender units) or `offsetUU` "
                    "(Unreal units), with `offsetType` for how it is measured.",
        "offsetAppliedBU": "offsetAppliedBU is measured and reported - what the bevel applied after clampOverlap. "
                           "Ask with `offset`.",
    })
    obj = get_object(take(params, "object", "name", required=True, kind=str),
                     want_mesh=True)

    offset = take_float(params, "offset")
    offset_uu = take_float(params, "offsetUU")
    if (offset is None) == (offset_uu is None):
        raise MifOpError("pass exactly one of 'offset' (Blender units) or 'offsetUU' "
                         "(Unreal units; 1 BU = %g uu)" % UU_PER_BU)
    if offset is None:
        offset = offset_uu / UU_PER_BU
    if offset <= 0.0:
        raise MifOpError("offset must be > 0")

    segments = take_int(params, "segments", default=1)
    if segments < 1:
        raise MifOpError("'segments' must be >= 1 (1 = flat chamfer, >1 = rounded)")

    dry_run = take_bool(params, "dryRun", default=False)
    # THE HARD-SURFACE FINISH (1.4): a bevel of the sharp edges with hardened normals, every face smooth and a Weighted
    # Normal modifier - flat faces shade flat, the edges catch light, no support loops. Read before anything is touched.
    finish = _finish_spec(params)
    select_params = params
    if finish is not None and finish["minAngleDeg"] is not None:
        select_params = dict(params, minAngleDeg=finish["minAngleDeg"])

    mesh = obj.data
    warnings = []
    if mesh.has_custom_normals:
        warnings.append(
            "'%s' carries custom split normals. New bevel geometry has no custom normal "
            "of its own, so shading across the new faces may look wrong until normals "
            "are re-authored." % obj.name)

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        edges, criteria = _select_edges(bm, obj, select_params, "bevel_edges")
        if criteria.get("allEdges") is not None or ("extreme" in criteria and "angle" not in criteria):
            # EVERY EDGE THAT HAS AN EDGE TO ROUND, for a selector that does not look at angles. One across a flat face, as a
            # fan cap or a split n-gon has, has nothing there: a strip on the flat, 4 n-gons and 12 near-zero tangents on a cube
            # with a hole, and a rim's clamp cut to 0.005 by its cap's own spokes (measured on 5.2.1, 2026-11-01).
            flat = math.radians(_BEVEL_FLAT_DEG)
            rounded = [e for e in edges if len(e.link_faces) != 2 or e.calc_face_angle(0.0) > flat]
            criteria["flatEdgesSkipped"] = len(edges) - len(rounded)
            edges = rounded
        if finish is not None and finish["minAngleDeg"] is not None:
            criteria["finish"] = "no selector was given, so the finish took every edge whose faces meet %g degrees or " \
                                 "more off flat" % finish["minAngleDeg"]
        lo0, hi0, size0 = _axis_sizes(bm)
        verts0, faces0 = len(bm.verts), len(bm.faces)
        tris0 = sum(len(f.verts) - 2 for f in bm.faces)

        if not edges:
            raise MifOpError(
                "the selector matched 0 edges on '%s'. Mesh local bounds are min %s "
                "max %s (Blender units); selection breakdown: %s"
                % (obj.name, rnd(lo0), rnd(hi0), criteria))

        # affect='EDGES' is a SILENT NO-OP on a pure boundary edge (exactly one linked face) -
        # VERIFIED empirically 2026-08-27/28 on both a minimal test plane and this addon's own
        # barrel fixture: bmesh.ops.bevel(geom=boundary_edges, affect='EDGES', ...) leaves
        # vert/face counts bit-for-bit unchanged regardless of offset or segment count, while
        # ok:true and every reported "before" field were still correct - a confidently-wrong
        # answer for exactly the tool's own headline use case (_MIF_DEFAULT_EDGE_SELECTOR's
        # comment: "the two long edges of a road/sidewalk tile", which boundaryOnly selects as
        # PURE boundary edges by construction). affect='VERTICES' with the same edges' vertices
        # as geom genuinely bevels them - also verified empirically, same fixture.
        #
        # affect='VERTICES' is NOT a universal replacement, though: on already-working interior
        # edges it produces MORE geometry than affect='EDGES' for the identical selection (measured:
        # +368 verts vs +141 for the same 21-edge sharp-angle selection on the barrel) - a vertex
        # bevel treats every touched vertex as its own corner rather than following the edge loop's
        # dihedral, which is a real behaviour change, not just a bug fix, for the case that already
        # worked. So this is NOT swapped in wholesale - only routed to boundary edges, which is the
        # ONLY case that was actually broken.
        #
        # boundaryOnly and the angle selector (min/maxAngleDeg) are each PURE by construction -
        # boundaryOnly keeps only len(link_faces)==1, and calc_face_angle(None) returns None (and
        # is skipped) for exactly those same edges - so in normal use this partition is never mixed.
        # A mixed selection is only reachable via axis+side, edgeIndices or allEdges, and mixing the
        # two Blender bevel algorithms in one call has no single correct answer, so it is refused
        # rather than guessed at.
        boundary_edges = [e for e in edges if len(e.link_faces) == 1]
        other_edges = [e for e in edges if len(e.link_faces) != 1]
        if boundary_edges and other_edges:
            raise MifOpError(
                "the selection mixes %d boundary edge(s) (one linked face) with %d non-boundary "
                "edge(s) - these need different Blender bevel algorithms (affect='VERTICES' vs "
                "'EDGES') and there is no single correct way to bevel both in one call. Make two "
                "calls: one with a selector that matches only boundary edges (boundaryOnly:true), "
                "one that matches only the rest (minAngleDeg/maxAngleDeg, or edgeIndices filtered "
                "by hand)." % (len(boundary_edges), len(other_edges)))
        if boundary_edges:
            bevel_affect = "VERTICES"
            bevel_geom = list({v for e in boundary_edges for v in e.verts})
        else:
            bevel_affect = "EDGES"
            bevel_geom = edges

        # Resolved BEFORE the edit: preserveAxes defaults to assertAxes, and a
        # bad axis name should refuse without having touched the mesh.
        assert_axes, preserve, snap_tol, seam_on_tol, seam_band = _guard_axes(
            params, offset * 1.01 + 1e-4)
        guarded = ["XYZ"[i] for i in assert_axes]
        # All three axes are WATCHED whatever is guarded -- guarding decides what
        # fails, never what is looked at.
        watch = _seam_snapshot(bm, lo0, hi0, (0, 1, 2), seam_on_tol, seam_band)
        if bevel_affect == "VERTICES":
            # RE-TAKEN AFTER THE SNAPSHOT. The snapshot adds vertex layers, which invalidates every
            # BMVert wrapper taken before it (see _seam_snapshot), and this list was built above out of
            # exactly those - so a boundary bevel handed bmesh.ops.bevel a list of dead verts. The
            # edges it came from are still valid; the verts are simply asked for again.
            bevel_geom = list({v for e in boundary_edges for v in e.verts})

        if dry_run:
            return {
                "dryRun": True,
                "object": obj.name,
                "selectedEdges": len(edges),
                "edgeIndices": [e.index for e in edges][:512],
                "edgeIndicesTruncated": len(edges) > 512,
                "criteria": criteria,
                "sizeBeforeBU": rnd(size0),
                "boundsLocalMinBU": rnd(lo0),
                "boundsLocalMaxBU": rnd(hi0),
                "preserveAxes": ["XYZ"[i] for i in preserve],
                "assertAxes": guarded,
                "seamToleranceBU": seam_on_tol,
                "seamBandBU": seam_band,
                "onSeamVertsBefore": {"XYZ"[i]: len(watch[i]["rows"]) for i in watch},
                "warnings": warnings,
            }

        # THE THREE ENUMS BLENDER VALIDATES ITSELF, converted from a raw ValueError into a refusal.
        # offsetType, miterOuter and miterInner go straight into bmesh, which raises
        # "miter_inner: 'x' not found in ([...])" for anything it does not know - a raw Python
        # exception where this addon's contract is MifOpError. The matrix counted 36 of them across
        # four builds the day this op's payload stopped sending 4 of its 28 keys.
        #
        # BLENDER'S OWN MESSAGE IS THE VALID LIST, and that is why it is quoted rather than replaced.
        # bmesh's enums are not introspectable like an RNA type, and they do NOT match the bpy
        # operator's: bmesh takes PATCH for miter_inner and bpy.ops.mesh.bevel does not, so a
        # hand-written list here would refuse values that work.
        #
        # Nothing has been written to the mesh at this point - bevel operates on the bmesh working
        # copy and to_mesh is further down - so this refuses without leaving the object changed.
        # ONE SET OF ARGUMENTS, for the bevel and for the trials that measure what it applied (below).
        bevel_kwargs = dict(
            offset_type=str(take(params, "offsetType", default="OFFSET")).upper(),
            segments=segments,
            profile=take_float(params, "profile", default=0.5),
            # affect is 'EDGES' for a normal (non-boundary) selection and 'VERTICES' for a pure
            # boundary one - see the note above where bevel_affect/bevel_geom are computed; a
            # boundary edge has only one linked face, and affect='EDGES' is a silent no-op on
            # those regardless of offset or segments. material=0 would drag every new face into
            # the first material slot instead of inheriting, which -1 avoids either way.
            affect=bevel_affect,
            material=-1,
            clamp_overlap=take_bool(params, "clampOverlap", default=True),
            loop_slide=take_bool(params, "loopSlide", default=True),
            harden_normals=True if finish is not None else take_bool(params, "hardenNormals", default=False),
            miter_outer=str(take(params, "miterOuter", default="SHARP")).upper(),
            miter_inner=str(take(params, "miterInner", default="SHARP")).upper(),
            spread=take_float(params, "spread", default=0.0),
        )
        if finish is not None:
            # The bevel marks the faces it touched as weaker than the flat faces beside them, and the Weighted Normal
            # modifier (face influence on, _bevel_finish) then lets a flat face's own normal win - measured on 3.6.23
            # and 5.2.1 on a 1 m cube with a 2 cm bevel: the flat faces' shading normals leaned 1.64 degrees with the
            # modifier alone and 0.000 with face strength; hardened normals alone leaned them 70 degrees.
            bevel_kwargs["face_strength_mode"] = "AFFECTED"
        flat_opened = _open_flat_spokes(bm, bevel_geom, offset, bevel_kwargs) if bevel_kwargs["clamp_overlap"] else 0
        bm.verts.index_update()
        bm.edges.index_update()
        geom_ref = [("E", g.index) if isinstance(g, bmesh.types.BMEdge) else ("V", g.index) for g in bevel_geom]
        probe = bm.copy() if bevel_kwargs["clamp_overlap"] else None
        try:
            try:
                bevel_out = bmesh.ops.bevel(bm, geom=bevel_geom, offset=offset, **bevel_kwargs)
            except (ValueError, TypeError) as exc:
                raise MifOpError(
                    "bevel refused a value: %s. offsetType, miterOuter and miterInner are passed to "
                    "bmesh, and the list in that message is this Blender's own - it differs between "
                    "builds and between bmesh and bpy.ops.mesh.bevel, so trust it rather than the "
                    "documentation. NOTHING was changed." % exc)
            applied, clamped = offset, None
            if probe is not None:
                applied, clamped = _bevel_applied_offset(probe, geom_ref, offset, bevel_kwargs, _bevel_key(bm))
        finally:
            if probe is not None:
                probe.free()

        # ---- preserve: snap drifted extremes back onto the original planes ----
        # The snap tolerance must exceed the offset, or the very verts the bevel
        # pulled inward are the ones it fails to catch. seam_band is the same
        # number by default, so "in the band" == "the snap should have got it".
        snapped = 0
        for idx in preserve:
            for vert in bm.verts:
                if abs(vert.co[idx] - lo0[idx]) <= snap_tol:
                    if vert.co[idx] != lo0[idx]:
                        vert.co[idx] = lo0[idx]
                        snapped += 1
                elif abs(vert.co[idx] - hi0[idx]) <= snap_tol:
                    if vert.co[idx] != hi0[idx]:
                        vert.co[idx] = hi0[idx]
                        snapped += 1

        lo1, hi1, size1 = _axis_sizes(bm)
        delta = [size1[i] - size0[i] for i in range(3)]
        seam = _seam_verdict(watch, bm, lo0, hi0, seam_on_tol, seam_band)

        # ---- assert 1 of 2: EXTENT. Discard rather than write a sheared tile ----
        assert_tol = take_float(params, "assertTolerance", default=1e-5)
        violated = [("XYZ"[i], size0[i], size1[i]) for i in assert_axes
                    if abs(delta[i]) > assert_tol]
        if violated:
            raise MifOpError(
                "bevel changed a size that was asserted constant: %s (tolerance %g BU). "
                "The mesh was NOT modified. Anything tiling along that axis would shear. "
                "Either add the axis to preserveAxes, reduce the offset, or pick a "
                "selector that does not touch the seam."
                % ("; ".join("%s %.6f -> %.6f" % v for v in violated), assert_tol))

        # ---- assert 2 of 2: PLANARITY. The one the extent cannot see ----
        seam_bad = _seam_violations(seam, guarded)
        if seam_bad:
            raise MifOpError(
                "the bevel broke seam PLANARITY on an asserted axis: %s (band %g BU / "
                "%.4f uu). The mesh was NOT modified. The bounding box is UNCHANGED and "
                "would have passed the size assert -- one surviving corner vertex pins "
                "the extent while the rest of the end cap slides inward, which is exactly "
                "the tile that shears at every spline join. Full measurement: %s. Fix it "
                "by adding the axis to preserveAxes (it snaps such verts back), reducing "
                "the offset, or picking a selector that does not reach the seam."
                % ("; ".join(seam_bad), seam_band, seam_band * UU_PER_BU, seam))

        for i in range(3):
            if abs(delta[i]) > 1e-5 and i not in assert_axes:
                warnings.append(
                    "size along %s changed %.6f -> %.6f BU (%.4f uu). If the mesh tiles "
                    "along %s this will shear the seam -- use preserveAxes/assertAxes."
                    % ("XYZ"[i], size0[i], size1[i], delta[i] * UU_PER_BU, "XYZ"[i]))
        warnings.extend(_seam_warnings(seam, guarded, criteria, seam_band))

        # REFUSE RATHER THAN PRETEND, same discipline as decimate_mesh: a non-empty selection
        # that added no geometry at all is not success, whatever bmesh.ops.bevel itself returned.
        # This is a safety net behind the boundary/interior split above, not a replacement for it -
        # it exists for whatever combination of offset/segments/topology neither experiment covered.
        if len(bm.verts) == verts0 and len(bm.faces) == faces0:
            raise MifOpError(
                "bevel selected %d edge(s) on '%s' but added no geometry at all - vert and face "
                "counts are unchanged. The mesh was NOT modified. This can happen when "
                "clamp_overlap reduces every selected edge's effective offset to zero against "
                "tightly packed neighbouring geometry; try a smaller offset, or check the "
                "selection with select_edges first." % (len(edges), obj.name))

        # ---- crossings, on the final geometry - see _crossing_face_pairs ----
        crossings = _crossing_face_pairs(bm)
        crossings_before = None
        if crossings:
            before_bm = bmesh.new()
            try:
                before_bm.from_mesh(mesh)          # the mesh data is still the pre-bevel mesh until to_mesh below
                crossings_before = _crossing_face_pairs(before_bm)
            finally:
                before_bm.free()
        shape_before = _mesh_shape(mesh)

        # THE N-GONS THE BEVEL ITSELF MADE, where more than three bevelled edges meet: gameReady fails them, and a mesh's own
        # n-gons are left for whoever is still modeling it. UVs its new strips collapsed are opened.
        made_ngons = [f for f in bevel_out.get("faces", []) if f.is_valid and len(f.verts) > 4]
        if flat_opened:
            triangulate_ngons(bm)
        elif made_ngons:
            bmesh.ops.triangulate(bm, faces=made_ngons, quad_method="BEAUTY", ngon_method="BEAUTY")
        uv_opened = _open_collapsed_uvs(bm, mesh.uv_layers[0].name) if len(mesh.uv_layers) else 0
        bm.faces.index_update()
        new_faces = {f.index for f in bevel_out.get("faces", []) if f.is_valid}
        tris1 = sum(len(f.verts) - 2 for f in bm.faces)
        bm.normal_update()
        bm.to_mesh(mesh)
        mesh.update()
        # obj.bound_box and obj.dimensions are CACHES and neither mesh.update()
        # nor obj.update_tag() refreshes them -- MEASURED on 4.4.0, see
        # ops_common.local_bounds. This is the call that does. local_bounds does
        # not depend on it (it reads the verts), but every other consumer of
        # obj.dimensions does, including list_objects and the viewport.
        bpy.context.view_layer.update()
    finally:
        bm.free()
    finish_block = _bevel_finish(obj, finish, new_faces, tris0, tris1) if finish is not None else None

    return {
        "object": obj.name,
        "selectedEdges": len(edges),
        "criteria": criteria,
        "finish": finish_block,
        "offsetBU": round(offset, 8),                  # what was ASKED for
        "offsetUU": round(offset * UU_PER_BU, 6),
        # WHAT THE BEVEL APPLIED - see _bevel_applied_offset. clampedByOverlap is None when clampOverlap is off.
        "offsetAppliedBU": round(applied, 6) if applied is not None else None,
        "clampedByOverlap": clamped,
        "clampNote": ("clampOverlap held the bevel to about %.4g BU of the %g BU asked for: the geometry around the "
                      "selected edges has no room for more, and Blender clamps every selected edge to the tightest. "
                      "The mesh is bevelled at the smaller offset - pass clampOverlap:false to let it overlap itself, "
                      "or select fewer edges." % (applied, offset)) if clamped else None,
        "segments": segments,
        "uvFacesOpened": uv_opened,
        "flatEdgesOpened": flat_opened,
        # WHAT THE GEOMETRY DID, which no count shows - see _crossing_face_pairs and _mesh_shape.
        "shapeBefore": shape_before,
        "shapeAfter": _mesh_shape(mesh),
        "selfIntersectingFacePairs": crossings,
        "selfIntersectingFacePairsBefore": crossings_before,
        "selfIntersectionNote": (
            "the bevel left %d pair(s) of faces passing through each other (%d before it): the offset is larger than the "
            "geometry around the selected edges has room for, so new faces cross older ones and render interpenetrating. "
            "The mesh WAS modified. clampOverlap does not prevent it from Blender 4.4 - measured, a 0.7 BU bevel on a "
            "radius-0.5 tube's rim applied 0.689 and crossed 386 pairs, where 3.6 and 4.2 clamped it to 0.498. Use an offset "
            "below the smallest feature the edges touch, "
            "or bevel fewer edges." % (crossings, crossings_before or 0))
        if crossings and crossings > (crossings_before or 0) else None,
        "selfIntersectionCheckNote": ("not checked: %d faces is more than the %d this check reads"
                                      % (len(mesh.polygons), _CROSSING_FACE_CAP)) if crossings is None else None,
        "vertsBefore": verts0,
        "vertsAfter": len(mesh.vertices),
        "facesBefore": faces0,
        "facesAfter": len(mesh.polygons),
        "sizeBeforeBU": rnd(size0),
        "sizeAfterBU": rnd(size1),
        "sizeDeltaBU": rnd(delta, 8),
        "sizeDeltaUU": rnd([d * UU_PER_BU for d in delta], 6),
        "boundsLocalMinBU": rnd(lo1),
        "boundsLocalMaxBU": rnd(hi1),
        "preserveAxes": ["XYZ"[i] for i in preserve],
        "assertAxes": guarded,
        "preserveSnappedVerts": snapped,
        "preserveToleranceBU": snap_tol,
        "seamToleranceBU": seam_on_tol,
        "seamBandBU": seam_band,
        # Reported unconditionally, for all three axes, exactly the way
        # sizeDeltaBU is -- an unguarded axis is still MEASURED.
        "offSeamVerts": _off_seam_verts(seam),
        "seamPlanarity": seam,
        "objectAfter": object_info(obj),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# extrude_skirt
# ---------------------------------------------------------------------------

_SKIRT_KEYS = {
    "object", "name",
    # selection -- byte-identical grammar to bevel_edges / select_edges
    "minAngleDeg", "maxAngleDeg", "axis", "side", "tolerance",
    "boundaryOnly", "edgeIndices", "allEdges",
    # geometry
    "depth", "depthUU", "direction", "flipNormals",
    # safety
    "preserveAxes", "preserveTolerance", "assertAxes", "assertTolerance",
    "seamTolerance", "seamBand", "allowNonBoundary", "dryRun", "previewId", "previewTtlSeconds"}

# Sign applied to the extrude vector's Z. X and Y are ZERO in both cases and
# there is deliberately no knob to make them anything else -- see the docstring.
_SKIRT_DIRECTIONS = {"down": -1.0, "up": 1.0}


def op_extrude_skirt(params):
    """Extrude a boundary edge loop straight along Z, forming a skirt.

    WHY THIS EXISTS. A road/sidewalk tile with a flat cut edge hovers wherever
    the terrain falls away under it. A skirt hides that. A bevel would too, but
    a bevel moves the endpoints of every edge it touches -- including the verts
    on the end-cap seam -- and a tile whose seam has moved shears when it is
    instanced along a spline. This op moves NOTHING in X or Y:

        new geometry = a duplicate of the selected boundary loop,
                       translated by (0, 0, +/-depth).

    That is by construction, not by assertion. The extrude duplicates each
    selected edge's verts in place (bmesh.ops.extrude_edge_only) and the only
    subsequent edit is bmesh.ops.translate with a vec whose X and Y are literal
    zeros. VERIFIED on Blender 4.4.0 headless against a 10 x 3 BU grid tile:
    8 boundary edges extruded and translated -0.15 in Z gave dX 0.0, dY 0.0,
    dZ 0.15, minX/maxX unmoved, and zero verts near-but-off the X seam planes.

    NORMALS. use_normal_flip defaults False, and on that same test the new side
    faces on the +Y boundary came back with normal (0, +1, 0) -- outward, which
    is what a skirt wants when the tile's top faces +Z. Pass flipNormals:true if
    your source winding is the other way round.

    UVs. The new faces get whatever bmesh copies from the source loop, which is
    not a meaningful skirt unwrap. Expect stretched texturing until UVs are
    authored -- nothing here does that for you, and it is not a bug to report.

    SAFETY. Same guard vocabulary as bevel_edges (preserveAxes / assertAxes /
    preserveTolerance / assertTolerance / seamTolerance / seamBand) and the same
    two-part assert: EXTENT plus seam PLANARITY, because an extent check alone
    cannot see a seam that has partially slid (one surviving corner vert pins the
    bounding box -- MEASURED, see the SEAM PLANARITY block above). preserveAxes
    defaults to assertAxes. assertAxes must NOT include Z: the whole point is
    that Z grows by up to `depth`.
    """
    reject_unknown(params, _SKIRT_KEYS, "extrude_skirt")
    obj = get_object(take(params, "object", "name", required=True, kind=str),
                     want_mesh=True)

    depth = take_float(params, "depth")
    depth_uu = take_float(params, "depthUU")
    if (depth is None) == (depth_uu is None):
        raise MifOpError("pass exactly one of 'depth' (Blender units) or 'depthUU' "
                         "(Unreal units; 1 BU = %g uu)" % UU_PER_BU)
    if depth is None:
        depth = depth_uu / UU_PER_BU
    if depth <= 0.0:
        raise MifOpError("depth must be > 0. Use direction:'up' rather than a negative "
                         "depth, so the response reports what actually happened.")

    direction = str(take(params, "direction", default="down")).lower()
    if direction not in _SKIRT_DIRECTIONS:
        raise MifOpError("'direction' must be 'down' or 'up' (got %r). There is no "
                         "sideways option: moving the loop in X or Y is exactly the "
                         "seam-shearing edit this op exists to avoid." % direction)
    flip_normals = take_bool(params, "flipNormals", default=False)
    allow_non_boundary = take_bool(params, "allowNonBoundary", default=False)
    dry_run = take_bool(params, "dryRun", default=False)

    mesh = obj.data
    warnings = []
    if mesh.has_custom_normals:
        warnings.append(
            "'%s' carries custom split normals. The new skirt faces have no custom "
            "normal of their own, so shading down the skirt may look wrong until "
            "normals are re-authored." % obj.name)

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        edges, criteria = _select_edges(bm, obj, params, "extrude_skirt")
        lo0, hi0, size0 = _axis_sizes(bm)
        verts0, faces0 = len(bm.verts), len(bm.faces)

        if not edges:
            raise MifOpError(
                "the selector matched 0 edges on '%s'. Mesh local bounds are min %s "
                "max %s (Blender units); selection breakdown: %s. Run select_edges to "
                "iterate on the selector without touching the mesh."
                % (obj.name, rnd(lo0), rnd(hi0), criteria))

        non_boundary = [e for e in edges if len(e.link_faces) != 1]
        if non_boundary and not allow_non_boundary:
            raise MifOpError(
                "%d of %d selected edges on '%s' are not boundary edges (they have 0 or "
                "2+ linked faces). Extruding an interior edge does not add a skirt -- it "
                "duplicates the loop and SPLITS the mesh along it, and the seam is then "
                "invisible from outside. Add boundaryOnly:true to the selector, or pass "
                "allowNonBoundary:true if you genuinely mean to split."
                % (len(non_boundary), len(edges), obj.name))

        # Nothing here moves laterally at all, so the default snap band is an
        # epsilon rather than bevel's offset-wide one: a wide band would drag
        # genuine interior verts onto the seam and call that a fix.
        assert_axes, preserve, snap_tol, seam_on_tol, seam_band = _guard_axes(
            params, 1e-6)
        if 2 in assert_axes:
            raise MifOpError(
                "assertAxes includes Z, which this op is guaranteed to violate: a skirt "
                "extends the mesh along Z by up to depth (%.6f BU). Assert X and/or Y."
                % depth)
        guarded = ["XYZ"[i] for i in assert_axes]
        watch = _seam_snapshot(bm, lo0, hi0, (0, 1, 2), seam_on_tol, seam_band)

        if dry_run:
            return {
                "dryRun": True,
                "object": obj.name,
                "selectedEdges": len(edges),
                "boundaryEdges": len(edges) - len(non_boundary),
                "nonBoundaryEdges": len(non_boundary),
                "edgeIndices": [e.index for e in edges][:512],
                "edgeIndicesTruncated": len(edges) > 512,
                "criteria": criteria,
                "depthBU": round(depth, 8),
                "depthUU": round(depth * UU_PER_BU, 6),
                "direction": direction,
                "sizeBeforeBU": rnd(size0),
                "boundsLocalMinBU": rnd(lo0),
                "boundsLocalMaxBU": rnd(hi0),
                "preserveAxes": ["XYZ"[i] for i in preserve],
                "assertAxes": guarded,
                "seamToleranceBU": seam_on_tol,
                "seamBandBU": seam_band,
                "onSeamVertsBefore": {"XYZ"[i]: len(watch[i]["rows"]) for i in watch},
                "warnings": warnings,
            }

        # ---- the extrude: duplicate the loop IN PLACE, then move it in Z only ----
        # bmesh.ops.extrude_edge_only(bmesh, edges=[], use_normal_flip=False,
        #   use_select_history=False) -> dict(geom=[])   [signature read off
        #   bmesh.ops.extrude_edge_only.__doc__ on Blender 4.4.0]
        result = bmesh.ops.extrude_edge_only(bm, edges=edges,
                                             use_normal_flip=flip_normals)
        geom = result.get("geom") or []
        new_verts = [g for g in geom if isinstance(g, bmesh.types.BMVert)]
        new_faces = [g for g in geom if isinstance(g, bmesh.types.BMFace)]
        if not new_verts:
            raise MifOpError(
                "extrude_edge_only returned no new vertices for %d selected edges on "
                "'%s'. The mesh was NOT modified. This is not a selector problem -- the "
                "extrude itself produced nothing, which usually means the edges were "
                "already consumed by a previous op on the same bmesh."
                % (len(edges), obj.name))

        # X and Y are literal zeros. This is the whole safety argument of the op.
        bmesh.ops.translate(bm, verts=new_verts,
                            vec=(0.0, 0.0, _SKIRT_DIRECTIONS[direction] * depth))

        # ---- preserve: a belt, and it should never have anything to do ----
        snapped = 0
        for idx in preserve:
            for vert in bm.verts:
                if abs(vert.co[idx] - lo0[idx]) <= snap_tol:
                    if vert.co[idx] != lo0[idx]:
                        vert.co[idx] = lo0[idx]
                        snapped += 1
                elif abs(vert.co[idx] - hi0[idx]) <= snap_tol:
                    if vert.co[idx] != hi0[idx]:
                        vert.co[idx] = hi0[idx]
                        snapped += 1

        lo1, hi1, size1 = _axis_sizes(bm)
        delta = [size1[i] - size0[i] for i in range(3)]
        seam = _seam_verdict(watch, bm, lo0, hi0, seam_on_tol, seam_band)

        # ---- assert 1 of 2: EXTENT ----
        assert_tol = take_float(params, "assertTolerance", default=1e-5)
        violated = [("XYZ"[i], size0[i], size1[i]) for i in assert_axes
                    if abs(delta[i]) > assert_tol]
        if violated:
            raise MifOpError(
                "extrude_skirt changed a size that was asserted constant: %s (tolerance "
                "%g BU). The mesh was NOT modified. A skirt cannot do this by itself, so "
                "the selector picked up edges that are not on the boundary loop you "
                "meant -- run select_edges and check boundaryEdges/interiorEdges."
                % ("; ".join("%s %.6f -> %.6f" % v for v in violated), assert_tol))

        # ---- assert 2 of 2: PLANARITY, the failure an extent cannot see ----
        seam_bad = _seam_violations(seam, guarded)
        if seam_bad:
            raise MifOpError(
                "seam PLANARITY broke on an asserted axis: %s (band %g BU / %.4f uu). The "
                "mesh was NOT modified. The bounding box is UNCHANGED, so the size assert "
                "passed -- one corner vert left on the plane keeps the reported extent "
                "identical while the rest of the seam has slid inward. A skirt moves "
                "nothing in X or Y, so this means the selection was not the boundary loop "
                "you meant: run select_edges and check boundaryEdges/interiorEdges. Full "
                "measurement: %s"
                % ("; ".join(seam_bad), seam_band, seam_band * UU_PER_BU, seam))

        # Z grows by up to `depth` and that is the point; anything outside
        # [0, depth] is not a skirt and gets said out loud.
        if delta[2] < -1e-5 or delta[2] > depth + 1e-5:
            warnings.append(
                "Z size changed %.6f -> %.6f BU (delta %.6f), which is outside the "
                "[0, depth=%.6f] a skirt can produce. Inspect before exporting."
                % (size0[2], size1[2], delta[2], depth))
        for i in (0, 1):
            if abs(delta[i]) > 1e-5 and i not in assert_axes:
                warnings.append(
                    "size along %s changed %.6f -> %.6f BU (%.4f uu). A skirt moves "
                    "nothing in X or Y, so this means the selection was not a clean "
                    "boundary loop. If the mesh tiles along %s this will shear the seam."
                    % ("XYZ"[i], size0[i], size1[i], delta[i] * UU_PER_BU, "XYZ"[i]))
        warnings.extend(_seam_warnings(seam, guarded, criteria, seam_band))

        new_face_count = len(new_faces)
        bm.normal_update()
        bm.to_mesh(mesh)
        mesh.update()
        # See the identical call in op_bevel_edges: obj.bound_box / obj.dimensions
        # are stale until a view-layer update, MEASURED on 4.4.0.
        bpy.context.view_layer.update()
    finally:
        bm.free()

    return {
        "object": obj.name,
        "selectedEdges": len(edges),
        "criteria": criteria,
        "depthBU": round(depth, 8),
        "depthUU": round(depth * UU_PER_BU, 6),
        "direction": direction,
        "flipNormals": flip_normals,
        "skirtVertsAdded": len(new_verts),
        "skirtFacesAdded": new_face_count,
        "vertsBefore": verts0,
        "vertsAfter": len(mesh.vertices),
        "facesBefore": faces0,
        "facesAfter": len(mesh.polygons),
        "sizeBeforeBU": rnd(size0),
        "sizeAfterBU": rnd(size1),
        "sizeDeltaBU": rnd(delta, 8),
        "sizeDeltaUU": rnd([d * UU_PER_BU for d in delta], 6),
        "boundsLocalMinBU": rnd(lo1),
        "boundsLocalMaxBU": rnd(hi1),
        "preserveAxes": ["XYZ"[i] for i in preserve],
        "assertAxes": guarded,
        "preserveSnappedVerts": snapped,
        "preserveToleranceBU": snap_tol,
        "seamToleranceBU": seam_on_tol,
        "seamBandBU": seam_band,
        # Reported unconditionally, for all three axes, the way sizeDeltaBU is.
        "offSeamVerts": _off_seam_verts(seam),
        "seamPlanarity": seam,
        "objectAfter": object_info(obj),
        "warnings": warnings,
    }



_MATERIAL_SLOT_KEYS = {"object", "slots", "allowResize", "links", "dryRun", "previewId", "previewTtlSeconds"}


def op_set_material_slots(params):
    """Set an object's material slot names, in ORDER.

    THE GAP THIS CLOSES WAS ALREADY DETECTED AND UNFIXABLE. mif_mesh_roundtrip
    compares the material-slot SEQUENCE before and after a Blender edit and warns
    when it differs, with the note "the mesh is valid, the assignment may not be -
    a human decides". It warns because there was nothing it could call. Slot
    ORDER is what decides which Unreal material lands on which face, so a
    reordered slot list renders the wrong material on a mesh that is otherwise
    perfect - and the round trip could see that and not act on it.

    WHY NAMES AND NOT MATERIALS. A slot holds a bpy.types.Material. The round trip
    cares about the SEQUENCE OF NAMES, because that is what lines up against
    Unreal's FStaticMaterial array on reimport. So this takes names, reuses an
    existing material of that name if one is in the file, and creates an empty one
    only when it must - the material's CONTENT is irrelevant here and inventing a
    shader would be inventing data.

    THE COUNT IS NOT CHANGED BY DEFAULT. Adding or removing a slot silently
    re-indexes every polygon's material_index and can leave faces pointing past the
    end of the list. So a list whose length differs from the current slot count is
    REFUSED unless allowResize is passed, and the refusal says what the counts are.

    params:
      object       (str, required)  a mesh - or a curve, text, surface, metaball, hair curves,
                                    point cloud or volume, which hold slots the same way
      slots        (list[str|null], required)  the slot names, in order
      allowResize  (bool, default False)  permit changing the slot COUNT
      links        (list[str|null])  per slot, OBJECT or DATA (MaterialSlot.link): where the slot's material is
                                    stored - on the object, or on the mesh every user shares. Set before the
                                    material, so the material lands there. OBJECT is how an override of a linked
                                    mesh object keeps its own material across a reload; a DATA slot on linked data
                                    that is not overridden is refused (the write would not survive a reload)
      dryRun       (bool)  the BULK PREVIEW (docs/15 section 5): every slot, link and material that would change, with
                                    its value before, and a previewId that {previewId} alone commits - planned again,
                                    refused on any difference, read back, and put back (slots, links, polygon material
                                    indices, the materials it made) on a mismatch
    """
    reject_unknown(params, _MATERIAL_SLOT_KEYS, "set_material_slots")
    dry_run = take_bool(params, "dryRun", False)
    obj = get_object(take(params, "object"))
    # NOT MESH-ONLY. A text object was refused as "a FONT, not a MESH", so a title could only ever
    # render in the default grey. A mesh still gets get_object's OBJECT-mode check, which is about
    # bmesh and does not apply to the other types.
    if obj.type == "MESH":
        get_object(obj.name, want_mesh=True)
    elif material_slot_names(obj) is None:
        raise MifOpError("object '%s' is a %s, which holds no material slots - a mesh, curve, text, "
                         "surface, metaball, hair curves, point cloud or volume does. NOTHING was "
                         "changed." % (obj.name, obj.type))
    require_editable(obj, "reslot")
    slots = params.get("slots")
    if not isinstance(slots, list) or not slots:
        raise MifOpError("'slots' is required: a list of material names in ORDER, "
                         "e.g. [\"M_Road\", \"M_Kerb\"]. null means an empty slot. "
                         "Read the current order from object_info.materialSlots.")
    for s in slots:
        if s is not None and not isinstance(s, str):
            raise MifOpError("every entry in 'slots' must be a string or null; got %r" % (s,))

    before = material_slot_names(obj)
    allow_resize = take_bool(params, "allowResize", False)
    links = params.get("links")
    if links is not None:
        if (not isinstance(links, list) or len(links) != len(slots)
                or any(v is not None and v not in ("OBJECT", "DATA") for v in links)):
            raise MifOpError("'links' must be a list as long as slots (%d), each OBJECT, DATA or null; got %r. "
                             "NOTHING was changed." % (len(slots), links))
    data_linked = (getattr(obj.data, "library", None) is not None
                   and getattr(obj.data, "override_library", None) is None)
    if data_linked:
        for i, name in enumerate(slots):
            link = (links[i] if links is not None and links[i] is not None else
                    (obj.material_slots[i].link if i < len(obj.material_slots) else "DATA"))
            current = obj.material_slots[i].material if i < len(obj.material_slots) else None
            if link == "DATA" and (current.name if current is not None else None) != name:
                raise MifOpError(
                    "slot %d of '%s' stores its material on the data '%s', which is LINKED from %s - the change "
                    "would not survive a reload. Pass links with OBJECT for that slot (the object keeps its own "
                    "material), or override the data (library_override data). NOTHING was changed."
                    % (i, obj.name, obj.data.name, obj.data.library.filepath))
        if allow_resize and len(slots) != len(before):
            raise MifOpError("'%s' uses linked data '%s', whose slot count this file cannot change. NOTHING was "
                             "changed." % (obj.name, obj.data.name))

    if len(slots) != len(before) and not allow_resize:
        # REFUSED rather than resized. Changing the count re-indexes polygons, and a
        # face whose material_index now points past the end renders as the LAST slot
        # with no error anywhere - exactly the silent-wrong-result this project keeps
        # finding. Pass allowResize when that is genuinely what you want.
        raise MifOpError(
            "'%s' has %d material slot(s) and %d were given. Changing the COUNT "
            "re-indexes every polygon's material_index, and a face left pointing past "
            "the end renders as the last slot with no error. Pass allowResize:true if "
            "that is intended. Current order: %s. NOTHING was changed."
            % (obj.name, len(before), len(slots), before))

    if dry_run:
        # EVERY CHECK ABOVE HAS RUN; nothing below has. What the write would leave, predicted from the same inputs.
        links_now = [ms.link for ms in obj.material_slots]
        # A slot the resize adds takes the preference "Link Materials To" (OBDATA, the factory value, or OBJECT).
        new_link = "OBJECT" if getattr(bpy.context.preferences.edit, "material_link", "OBDATA") == "OBJECT" else "DATA"
        links_after = [(links[i] if links is not None and links[i] is not None
                        else (links_now[i] if i < len(links_now) else new_link)) for i in range(len(slots))]
        would_create = list(dict.fromkeys(n for n in slots if n is not None and bpy.data.materials.get(n) is None))
        return {"object": obj.name, "dryRun": True, "changed": False, "before": before, "materialSlots": list(slots),
                "slotCount": len(slots), "links": links_after, "linksBefore": links_now,
                "wouldCreateMaterials": would_create, "wouldChange": (list(slots) != before or links_after != links_now
                                                                      or bool(would_create)),
                "note": "DRY RUN - nothing was written."}

    # Resize first, so the assignment loop below always has a slot to write into.
    if allow_resize:
        while len(obj.material_slots) < len(slots):
            obj.data.materials.append(None)
        while len(obj.material_slots) > len(slots):
            obj.data.materials.pop()

    if links is not None:
        for i, link in enumerate(links):
            if link is not None:
                obj.material_slots[i].link = link
    created = []
    for i, name in enumerate(slots):
        if name is None:
            obj.material_slots[i].material = None
            continue
        mat = bpy.data.materials.get(name)
        if mat is None:
            # Created EMPTY and reported. The name is what the round trip lines up
            # against Unreal's slot array; the shader is Unreal's business, and
            # inventing one here would be inventing data nobody asked for.
            mat = bpy.data.materials.new(name=name)
            created.append(mat.name)
            if mat.name != name:
                # Blender uniquifies a clashing name silently (M_Road -> M_Road.001).
                # A slot named M_Road.001 will not line up with Unreal's M_Road, so
                # this is reported rather than left to be discovered at reimport.
                raise MifOpError(
                    "asked for material '%s' but Blender created '%s' - a name clash "
                    "was silently uniquified, and that name will NOT line up against "
                    "Unreal's slot array on reimport. NOTHING further was changed."
                    % (name, mat.name))
        obj.material_slots[i].material = mat

    after = material_slot_names(obj)

    # Every polygon's material_index must still be in range. Checked rather than
    # assumed, because an out-of-range index is exactly the failure allowResize
    # invites and it is invisible until something renders. None for a type with no
    # polygons: a 0 there would report a check that never ran.
    out_of_range = None
    if obj.type == "MESH":
        out_of_range = (sum(1 for poly in obj.data.polygons if poly.material_index >= len(after))
                        if after else len(obj.data.polygons))

    result = {
        "object": obj.name,
        "before": before,
        "materialSlots": after,
        "slotCount": len(after),
        "createdMaterials": created,
        "polygonsOutOfRange": out_of_range,
        "links": [ms.link for ms in obj.material_slots],
    }
    if out_of_range:
        result["warning"] = (
            "%d polygon(s) have a material_index past the end of the slot list. They "
            "will render as the last slot. This is a consequence of resizing; fix it "
            "by restoring the slot count or reassigning those faces."
            % out_of_range)
    return result


_UV_KEYS = {
    "object", "name",
    "method", "angleLimitDeg", "islandMargin", "uvLayer", "replace",
    "correctAspect", "dryRun",
    # seams -> unwrap -> pack -> transform, in that order and no other. Packing before unwrapping
    # packs the old layout; seams marked afterwards do nothing until the next unwrap.
    "markSeams", "clearSeams", "uvPack", "packMargin", "uvTransform", "keepBounds",
    # method BOX only: the world size of one UV tile, given directly or as a density on a texture size.
    "metersPerTile", "texelDensity", "textureSize", "boxSpace",
    # THE OTHER UNWRAP METHODS (2026-10-06): fillHoles for ANGLE, CONFORMAL and MINIMUM_STRETCH; activeFace for
    # FOLLOW_ACTIVE_QUADS
    "fillHoles", "activeFace", "previewId", "previewTtlSeconds",
    # SEAMS BY RULE AND A PACK TO A FILL TARGET (2026-11-07)
    "seamRules", "packFill"}

# bpy.ops.uv.unwrap's method for each of this op's names. MINIMUM_STRETCH is in the operator's enum on 4.4, 5.0 and
# 5.2 and not on 4.2 (read off its rna_type, 2026-10-06); 3.6 and 4.2 have only uv.minimize_stretch, which relaxes a
# layout rather than unwrapping one - uv_relax runs it.
_UNWRAP_METHODS = {"ANGLE": "ANGLE_BASED", "ANGLE_BASED": "ANGLE_BASED", "CONFORMAL": "CONFORMAL",
                   "MINIMUM_STRETCH": "MINIMUM_STRETCH"}


# Blender's hard maximum UV layers per mesh. MEASURED on 3.6.23, 4.2.17, 4.4.0 and 5.0.1 -
# all four accept 8 and return None from new() on the ninth, silently.
_UV_LAYER_CAP = 8


def _uv_transform_values(transform):
    """(sx, sy, ox, oy) from a caller's uvTransform, or None when it was not given. Refuses before any write.

    MOVED OUT OF THE WRITE. It was parsed after the unwrap and every refusal had to end "The unwrap above
    already ran." - true, and the defect: the refused call had already replaced the layout.
    """
    if transform is None:
        return None
    if not isinstance(transform, dict):
        raise MifOpError("uvTransform must be an object like {scale: [1,1], offset: [0,0]}. NOTHING was "
                         "changed.")
    unknown = set(transform) - {"scale", "offset"}
    if unknown:
        raise MifOpError("uvTransform does not take %s - it takes scale [u,v] and offset "
                         "[u,v]. Rotation is not offered because rotating a packed layout "
                         "moves islands out of 0-1 with nothing to put them back. NOTHING was changed."
                         % ", ".join(sorted(unknown)))

    def _pair(key, default):
        v = transform.get(key, default)
        # THE isinstance CHECK DOES NOT EXCLUDE NaN, because a NaN IS a float. A NaN scale or
        # offset lands on the UV layer and every coordinate it touches becomes nan - and the
        # unwrap that follows reports success over a layer that is now unusable.
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return tuple(finite_floats([v, v], "uvTransform.%s" % key))
        if not isinstance(v, (list, tuple)) or len(v) != 2 \
                or any(not isinstance(x, (int, float)) or isinstance(x, bool) for x in v):
            raise MifOpError("uvTransform.%s must be a number or [u,v]; got %r. NOTHING was changed." % (key, v))
        return tuple(finite_floats(v, "uvTransform.%s" % key))

    try:
        sx, sy = _pair("scale", 1.0)
        ox, oy = _pair("offset", 0.0)
    except MifOpError as exc:
        text = str(exc)
        raise MifOpError(text if "NOTHING" in text else "%s NOTHING was changed." % text.rstrip("."))
    return sx, sy, ox, oy


def _pinned_op_kwargs(op, wanted):
    """The keyword arguments an operator really takes, out of `wanted`, for THIS Blender.

    OPERATOR PROPERTIES PERSIST BETWEEN bpy.ops CALLS IN ONE SESSION. A property the caller does not pass is not reset
    to its default: it keeps whatever the last call - a script, another op, the user in the UV editor - left it at, so
    a layout would depend on history nothing in the request shows (the same lesson as fill_holes below). Every
    property that shapes the result is pinned - but only those this build HAS, and only enum values it lists:
    pack_islands on 3.6 takes rotate and margin and nothing else, and an unknown keyword raises.
    (Measured 2026-09-27, Blender 5.2: SMART + uvPack gave layouts using 0.03% to 22% of the tile. That turned out to
    be multi-object editing - see the selection note in op_uv_unwrap - not remembered properties; pinning changed
    nothing there, and is kept because the persistence is real.)"""
    try:
        props = op.get_rna_type().properties
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for key, value in wanted.items():
        if key not in props.keys():
            continue
        prop = props[key]
        if prop.type == "ENUM":
            if value not in {item.identifier for item in prop.enum_items}:
                continue
        out[key] = value
    return out


# pack_islands' options, every one that shapes the layout (see _pinned_op_kwargs) - uv_unwrap's pack and uv_pack share it
_PACK_PINNED = {"rotate": True, "rotate_method": "ANY", "scale": True, "merge_overlap": False, "margin_method": "SCALED",
                "pin": False, "shape_method": "CONCAVE", "udim_source": "CLOSEST_UDIM"}


def _uv_layout_area(mesh, layer_name):
    """Total UV area of a layer's faces (shoelace per polygon, so concave n-gons count once) - how much of the 0-1
    tile a layout uses. 1.0 is the whole tile; overlapping islands would push it past that."""
    layer = mesh.uv_layers.get(layer_name)
    if layer is None or not len(layer.data):
        return 0.0
    n = len(layer.data)
    flat = [0.0] * (n * 2)
    layer.data.foreach_get("uv", flat)
    total = 0.0
    for poly in mesh.polygons:
        start, count = poly.loop_start, poly.loop_total
        acc = 0.0
        for k in range(count):
            a = start + k
            b = start + (k + 1) % count
            acc += flat[2 * a] * flat[2 * b + 1] - flat[2 * b] * flat[2 * a + 1]
        total += abs(acc) * 0.5
    return total


_BOX_KEYS = ("metersPerTile", "texelDensity", "textureSize", "boxSpace")


def _box_tile_size(params, method):
    """method BOX's tile: {metersPerTile, texelDensity, textureSize, space, from}, or None for any other method.

    Every refusal happens here, before anything is written: a BOX key beside another method (it would be ignored),
    metersPerTile and texelDensity together (two answers to one question), textureSize without texelDensity, and any
    value that is not a finite positive number."""
    given = [k for k in _BOX_KEYS if params.get(k) is not None]
    if method != "BOX":
        if given:
            raise MifOpError("%s %s read only with method BOX, and method %s would ignore %s. NOTHING was changed."
                             % (", ".join(given), "is" if len(given) == 1 else "are", method,
                                "it" if len(given) == 1 else "them"))
        return None
    mpt = take_float(params, "metersPerTile")
    density = take_float(params, "texelDensity")
    tex = take_float(params, "textureSize")
    space = (take(params, "boxSpace", default="OBJECT", kind=str) or "OBJECT").upper()
    if space not in ("OBJECT", "WORLD"):
        raise MifOpError("boxSpace must be OBJECT (the object's own axes with its rotation and scale, about its origin) "
                         "or WORLD (world coordinates, location included); got %r. NOTHING was changed." % space)
    if mpt is not None and density is not None:
        raise MifOpError("pass metersPerTile OR texelDensity, not both - each sets the tile's size, and two answers to "
                         "one question would leave one of them unused. NOTHING was changed.")
    if tex is not None and density is None:
        raise MifOpError("textureSize is read only with texelDensity (pixels per meter on a texture that many pixels "
                         "wide). With metersPerTile the tile's size is already given. NOTHING was changed.")
    for key, value in (("metersPerTile", mpt), ("texelDensity", density), ("textureSize", tex)):
        if value is not None and not (math.isfinite(value) and value > 0.0):
            raise MifOpError("%s must be a finite number above 0; got %r. NOTHING was changed." % (key, value))
    tex = tex if tex is not None else 1024.0
    if density is not None:
        mpt, source = tex / density, "texelDensity"
    elif mpt is not None:
        source = "metersPerTile"
    else:
        mpt, source = 1.0, "default"
    return {"metersPerTile": round(mpt, 6), "texelDensity": round(tex / mpt, 4), "textureSize": tex,
            "space": space, "from": source}


def _box_project(obj, mesh, layer_name, tile):
    """Write a box projection in meters into one UV layer, then MEASURE it: the density each face really got.

    Each face takes the two axes across its dominant normal axis (Z first on a tie, then X, then Y), in the space the
    tile names, divided by the tile's size. The first axis is signed by the normal so no face reads mirrored: looking
    at +X, u runs along +Y; at -X along -Y; at +Y along -X; at -Y along +X; at +Z along +X; at -Z along -X. v is Z on
    the side faces and Y on the top and bottom ones."""
    m3 = obj.matrix_world.to_3x3()
    offset = obj.matrix_world.translation.copy() if tile["space"] == "WORLD" else Vector((0.0, 0.0, 0.0))
    normal_m = m3.inverted_safe().transposed()
    size = tile["metersPerTile"]

    n_verts = len(mesh.vertices)
    flat = array.array("f", [0.0]) * (3 * n_verts)
    mesh.vertices.foreach_get("co", flat)
    pts = [m3 @ Vector((flat[3 * i], flat[3 * i + 1], flat[3 * i + 2])) + offset for i in range(n_verts)]
    n_loops = len(mesh.loops)
    loop_vert = array.array("i", [0]) * n_loops
    mesh.loops.foreach_get("vertex_index", loop_vert)

    uv = array.array("f", [0.0]) * (2 * n_loops)
    by_axis = {"x": 0, "y": 0, "z": 0}
    world_areas = []
    for poly in mesh.polygons:
        n = normal_m @ poly.normal
        ax, ay, az = abs(n.x), abs(n.y), abs(n.z)
        if az >= ax - 1e-6 and az >= ay - 1e-6:
            axis, sign = "z", (1.0 if n.z >= 0.0 else -1.0)
        elif ax >= ay - 1e-6:
            axis, sign = "x", (1.0 if n.x >= 0.0 else -1.0)
        else:
            axis, sign = "y", (1.0 if n.y >= 0.0 else -1.0)
        by_axis[axis] += 1
        start, count = poly.loop_start, poly.loop_total
        acc = Vector((0.0, 0.0, 0.0))
        for k in range(count):
            p = pts[loop_vert[start + k]]
            if axis == "x":
                u, v = p.y * sign, p.z
            elif axis == "y":
                u, v = -p.x * sign, p.z
            else:
                u, v = p.x * sign, p.y
            uv[2 * (start + k)] = u / size
            uv[2 * (start + k) + 1] = v / size
            acc += p.cross(pts[loop_vert[start + (k + 1) % count]])
        # The face's own area in the projected space (Newell), so the ratio below compares like with like.
        world_areas.append(acc.length * 0.5)
    layer = mesh.uv_layers[layer_name]
    layer.data.foreach_set("uv", uv)
    mesh.update()

    # MEASURED OFF THE LAYER, not off the array that was written: foreach_set reports nothing, so the density is read
    # back per face - 1.0 x the asked density on a face square to its axis, less on a slanted one.
    back = array.array("f", [0.0]) * (2 * n_loops)
    layer.data.foreach_get("uv", back)
    ratios = []
    mirrored = 0
    for poly, world_area in zip(mesh.polygons, world_areas):
        if world_area <= _DEGENERATE_FACE_AREA:
            continue
        start, count = poly.loop_start, poly.loop_total
        area2 = 0.0
        for k in range(count):
            a_, b_ = start + k, start + (k + 1) % count
            area2 += back[2 * a_] * back[2 * b_ + 1] - back[2 * b_] * back[2 * a_ + 1]
        # A face's loops run counter-clockwise seen from outside, so a UV face that runs clockwise reads mirrored.
        if area2 < 0.0:
            mirrored += 1
        ratios.append(math.sqrt(abs(area2) * 0.5 / world_area) * size)

    ratios.sort()
    tex = tile["textureSize"]
    out = dict(tile)
    out["facesByAxis"] = by_axis
    out["facesMirrored"] = mirrored
    if ratios:
        out["measuredTexelDensityMin"] = round(ratios[0] * tex / size, 3)
        out["measuredTexelDensityMedian"] = round(ratios[len(ratios) // 2] * tex / size, 3)
        out["measuredTexelDensityMax"] = round(ratios[-1] * tex / size, 3)
        out["facesSlanted"] = sum(1 for r in ratios if r < 0.999)
        out["densityNote"] = ("pixels per meter at textureSize, read back from the layer per face. A face square to its "
                              "axis gets exactly texelDensity; a slanted one (facesSlanted) gets less along its slope - "
                              "about 0.84 of it at 45 degrees - a box projection's own limit.")
    return out


def _follow_quads_unwrap(obj, mesh, active_face):
    """Follow Active Quads over every selected face from `active_face`, its UVs first set to a rectangle of its world
    edge lengths, so the grid starts square and in proportion. In edit mode, with everything selected."""
    bm = bmesh.from_edit_mesh(mesh)
    bm.faces.ensure_lookup_table()
    uvl = bm.loops.layers.uv.active
    m3 = obj.matrix_world.to_3x3()
    act = bm.faces[active_face]
    bm.faces.active = act
    ls = act.loops
    a = 0.5 * ((m3 @ (ls[1].vert.co - ls[0].vert.co)).length + (m3 @ (ls[2].vert.co - ls[3].vert.co)).length)
    b = 0.5 * ((m3 @ (ls[2].vert.co - ls[1].vert.co)).length + (m3 @ (ls[3].vert.co - ls[0].vert.co)).length)
    for lp, c in zip(ls, ((0.0, 0.0), (a, 0.0), (a, b), (0.0, b))):
        lp[uvl].uv = c
    bmesh.update_edit_mesh(mesh)
    bpy.ops.uv.select_all(action="SELECT")
    res = bpy.ops.uv.follow_active_quads(mode="LENGTH_AVERAGE")
    if "FINISHED" not in res:
        raise MifOpError("Follow Active Quads did not finish (%s)." % sorted(res))
    # INTO THE 0-1 TILE, proportions kept: the grid comes out in metres off the start face, and every other method
    # leaves a layout in the tile
    bm = bmesh.from_edit_mesh(mesh)
    uvl = bm.loops.layers.uv.active
    pts = [lp[uvl] for f in bm.faces for lp in f.loops]
    u0, v0 = min(p.uv.x for p in pts), min(p.uv.y for p in pts)
    span = max(max(p.uv.x for p in pts) - u0, max(p.uv.y for p in pts) - v0)
    if span > 1e-12:
        for p in pts:
            p.uv = ((p.uv.x - u0) / span, (p.uv.y - v0) / span)
        bmesh.update_edit_mesh(mesh)
def _refit_islands(mesh, layer_name, old_uv):
    """uv_unwrap keepBounds: each island of the new layout scaled uniformly and centred into the bounds its faces' UVs
    had before (old_uv, the layer's flat uv array). Returns {islands, refitted, rows (first 50: before / after bounds)}."""
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        uv = bm.loops.layers.uv[layer_name]
        rows = []
        for faces in _uv_islands(bm, uv)[0]:
            loops = [lp for f in faces for lp in f.loops]
            ou = [old_uv[2 * lp.index] for lp in loops]
            ov = [old_uv[2 * lp.index + 1] for lp in loops]
            nu = [lp[uv].uv.x for lp in loops]
            nv = [lp[uv].uv.y for lp in loops]
            ow, oh = max(ou) - min(ou), max(ov) - min(ov)
            nw, nh = max(nu) - min(nu), max(nv) - min(nv)
            if nw <= 1e-12 and nh <= 1e-12:
                continue
            k = min(ow / nw if nw > 1e-12 else math.inf, oh / nh if nh > 1e-12 else math.inf)
            ocx, ocy = (max(ou) + min(ou)) / 2, (max(ov) + min(ov)) / 2
            ncx, ncy = (max(nu) + min(nu)) / 2, (max(nv) + min(nv)) / 2
            for lp in loops:
                lp[uv].uv = (ocx + (lp[uv].uv.x - ncx) * k, ocy + (lp[uv].uv.y - ncy) * k)
            rows.append({"faces": len(faces), "before": [round(min(ou), 6), round(min(ov), 6), round(max(ou), 6),
                                                         round(max(ov), 6)],
                         "after": [round(ocx - nw * k / 2, 6), round(ocy - nh * k / 2, 6), round(ocx + nw * k / 2, 6),
                                   round(ocy + nh * k / 2, 6)]})
        bm.to_mesh(mesh)
        mesh.update()
    finally:
        bm.free()
    return {"refitted": len(rows), "islands": rows[:50]}


# ------------------------------------------------------------------------------------- uv_unwrap's seams by rule
# SEAMS BY RULE (2026-11-07). The automatic unwrap cut a rule-built test character's face into many islands and filled
# 38% of the texture. These rules mark the seams a character artist would: the head off the neck and up the back of the
# skull, and every piece of a garment cut open along its inside until it lies flat - then an ANGLE unwrap and a pack to a
# fill target. World space throughout; the object's origin is the working origin.
_SEAM_RULES = ("head", "garment")
_HEAD_RULE_KEYS = {"rule", "bone", "armature", "back"}
_GARMENT_RULE_KEYS = {"rule", "midPlane", "prefer", "preferWeight", "splitAtMidPlane"}
_MAX_SEAM_RULES = 8


def _rule_vec(raw, key, rule_i):
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise MifOpError("seamRules[%d].%s must be [x, y, z] (world), got %r. NOTHING was changed." % (rule_i, key, raw))
    v = Vector(finite_floats(list(raw), "seamRules[%d].%s" % (rule_i, key)))
    if v.length < 1e-9:
        raise MifOpError("seamRules[%d].%s must not be zero. NOTHING was changed." % (rule_i, key))
    return v


def _parse_seam_rules(raw, obj):
    """seamRules checked whole before any edge is touched: [(kind, spec)]. Resolves the head rule's armature and bone."""
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)) or not raw or len(raw) > _MAX_SEAM_RULES:
        raise MifOpError("seamRules is a list of 1 to %d rules, each {rule: %s, ...}; got %r. NOTHING was changed."
                         % (_MAX_SEAM_RULES, " | ".join(_SEAM_RULES), raw))
    out = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict) or r.get("rule") not in _SEAM_RULES:
            raise MifOpError("seamRules[%d] must be {rule: %s, ...}; got %r. NOTHING was changed."
                             % (i, " | ".join(_SEAM_RULES), r))
        kind = r["rule"]
        allowed = _HEAD_RULE_KEYS if kind == "head" else _GARMENT_RULE_KEYS
        extra = sorted(set(r) - allowed)
        if extra:
            raise MifOpError("seamRules[%d] (%s) does not take %s - it takes %s. NOTHING was changed."
                             % (i, kind, ", ".join(extra), ", ".join(sorted(allowed - {"rule"}))))
        if kind == "head":
            arm_name = r.get("armature")
            if arm_name is not None:
                arm = bpy.data.objects.get(arm_name) if isinstance(arm_name, str) else None
                if arm is None or arm.type != "ARMATURE":
                    raise MifOpError("seamRules[%d].armature %r is not an armature object. NOTHING was changed."
                                     % (i, arm_name))
            else:
                mod = next((m for m in obj.modifiers if m.type == "ARMATURE" and m.object is not None), None)
                arm = mod.object if mod is not None else (obj.parent if obj.parent is not None
                                                          and obj.parent.type == "ARMATURE" else None)
                if arm is None:
                    raise MifOpError("seamRules[%d] head needs the armature: '%s' has no Armature modifier with an object "
                                     "and no armature parent - pass armature. NOTHING was changed." % (i, obj.name))
            bname = r.get("bone")
            if bname is not None and (not isinstance(bname, str) or bname not in arm.data.bones):
                raise MifOpError("seamRules[%d].bone %r is not a bone of '%s'. NOTHING was changed." % (i, bname, arm.name))
            if bname is None:
                found = [b.name for b in arm.data.bones if b.name.lower() == "head"] or \
                        [b.name for b in arm.data.bones if "head" in b.name.lower()]
                if len(found) != 1:
                    raise MifOpError("seamRules[%d] head: '%s' has %s - name it with bone. NOTHING was changed."
                                     % (i, arm.name, "no bone called head" if not found else
                                        "%d bones with head in the name (%s)" % (len(found), ", ".join(found[:6]))))
                bname = found[0]
            back = _rule_vec(r["back"], "back", i) if r.get("back") is not None else Vector((0.0, 1.0, 0.0))
            out.append(("head", {"armature": arm, "bone": bname, "back": back.normalized()}))
        else:
            mp = r.get("midPlane")
            if mp is None:
                plane = (Vector((0.0, 0.0, 0.0)), Vector((1.0, 0.0, 0.0)))
            else:
                if not isinstance(mp, dict) or set(mp) - {"point", "normal"} or "normal" not in mp:
                    raise MifOpError("seamRules[%d].midPlane is {point: [x,y,z], normal: [x,y,z]}; got %r. NOTHING was "
                                     "changed." % (i, mp))
                plane = (_rule_vec(mp["point"], "midPlane.point", i) if mp.get("point") is not None
                         else Vector((0.0, 0.0, 0.0)), _rule_vec(mp["normal"], "midPlane.normal", i).normalized())
            pref = r.get("prefer", "inner")
            if isinstance(pref, str):
                if pref not in ("inner", "back"):
                    raise MifOpError("seamRules[%d].prefer is inner, back or a world direction [x,y,z]; got %r. NOTHING "
                                     "was changed." % (i, pref))
                pref_v = Vector((0.0, 1.0, 0.0)) if pref == "back" else None
            else:
                pref_v = _rule_vec(pref, "prefer", i).normalized()
            w = r.get("preferWeight", 4.0)
            if isinstance(w, bool) or not isinstance(w, (int, float)) or not 0.0 <= float(w) <= 100.0:
                raise MifOpError("seamRules[%d].preferWeight must be 0 to 100, got %r. NOTHING was changed." % (i, w))
            split = r.get("splitAtMidPlane", False)
            if not isinstance(split, bool):
                raise MifOpError("seamRules[%d].splitAtMidPlane is true or false, got %r. NOTHING was changed." % (i, split))
            out.append(("garment", {"plane": plane, "prefer": pref if isinstance(pref, str) else "direction",
                                    "preferVector": pref_v, "weight": float(w), "split": split}))
    return out


def _cut_dijkstra(sources, goal_of, allowed, pos, cost):
    """The cheapest edge path from any vertex in `sources` to the first vertex goal_of answers for, over the edges
    `allowed` admits. (path edges, path verts, goal key) or None."""
    import heapq as _hq
    dist, prev, heap, tie = {}, {}, [], 0
    for v in sources:
        dist[v] = 0.0
        heap.append((0.0, tie, v))
        tie += 1
    _hq.heapify(heap)
    while heap:
        d, _t, v = _hq.heappop(heap)
        if d > dist.get(v, float("inf")):
            continue
        key = goal_of(v)
        if key is not None and v not in sources:
            edges, verts = [], [v]
            while v in prev:
                u, e = prev[v]
                edges.append(e)
                verts.append(u)
                v = u
            return edges, verts, key
        for e in v.link_edges:
            if not allowed(e):
                continue
            w = e.other_vert(v)
            nd = d + cost(e)
            if nd < dist.get(w, float("inf")):
                dist[w] = nd
                prev[w] = (v, e)
                tie += 1
                _hq.heappush(heap, (nd, tie, w))
    return None


def _seam_islands(bm):
    """Faces grouped by connection across edges that are neither seams nor borders: [[BMFace]]."""
    seen, islands = set(), []
    for f0 in bm.faces:
        if f0.index in seen:
            continue
        seen.add(f0.index)
        stack, isl = [f0], []
        while stack:
            f = stack.pop()
            isl.append(f)
            for e in f.edges:
                if e.seam or len(e.link_faces) != 2:
                    continue
                for g in e.link_faces:
                    if g.index not in seen:
                        seen.add(g.index)
                        stack.append(g)
        islands.append(isl)
    return islands


def _rule_head(bm, obj, pos, wn, spec):
    """The head rule: a seam round the head where its faces meet the rest - the skin weights decide, so it runs under
    the chin and round the nape - then one up the back of the skull to the crown. Returns (edges, detail)."""
    arm, bname, back = spec["armature"], spec["bone"], spec["back"]
    bone = arm.data.bones[bname]
    amw = arm.matrix_world
    h0, h1 = amw @ bone.head_local, amw @ bone.tail_local
    axis = (h1 - h0).normalized()
    side = back.cross(axis)
    if side.length < 1e-6:
        raise MifOpError("seamRules head: back %s runs along bone '%s' - pass a back across it. NOTHING was changed."
                         % (rnd(list(back)), bname))
    side.normalize()
    head_set = {bname} | {c.name for c in bone.children_recursive}
    deform = {b.name for b in arm.data.bones if b.use_deform}
    gi = {g.index: g.name for g in obj.vertex_groups if g.name in deform}
    me = obj.data
    by = "weights"
    if any(g in head_set for g in gi.values()):
        vw = [{} for _ in me.vertices]
        for v in me.vertices:
            for g in v.groups:
                n = gi.get(g.group)
                if n is not None and g.weight > 0.0:
                    lab = "head" if n in head_set else n
                    vw[v.index][lab] = vw[v.index].get(lab, 0.0) + g.weight
        region = set()
        for f in bm.faces:
            tot = {}
            for v in f.verts:
                for lab, w in vw[v.index].items():
                    tot[lab] = tot.get(lab, 0.0) + w
            if tot and max(tot, key=lambda k: (tot[k], k == "head")) == "head":
                region.add(f)
    else:
        # NO HEAD WEIGHTS: the faces above the bone's head across its axis, within its length three times over
        by = "bonePlane"
        reach = 3.0 * (h1 - h0).length
        region = set()
        for f in bm.faces:
            c = sum((pos[v] for v in f.verts), Vector()) / len(f.verts)
            if (c - h0).dot(axis) >= 0.0 and (c - h0).length <= reach:
                region.add(f)
    if not region:
        raise MifOpError("seamRules head: no face of '%s' belongs to bone '%s' (%s). NOTHING was changed."
                         % (obj.name, bname, "by weight" if by == "weights" else "above its head"))
    ring = [e for e in bm.edges if len(e.link_faces) == 2 and ((e.link_faces[0] in region) != (e.link_faces[1] in region))]
    rim = ring or [e for e in bm.edges if len(e.link_faces) == 1 and e.link_faces[0] in region]
    rim_verts = list({v for e in rim for v in e.verts})
    region_verts = list({v for f in region for v in f.verts})
    if not rim_verts:
        raise MifOpError("seamRules head: the head of '%s' has no edge to the rest of the mesh and no open edge, so "
                         "there is nowhere for the back seam to start. NOTHING was changed." % obj.name)
    start = max(rim_verts, key=lambda v: (pos[v] - h0).dot(back) - 2.0 * abs((pos[v] - h0).dot(side)))
    crown = max(region_verts, key=lambda v: (pos[v] - h0).dot(axis) - 0.5 * abs((pos[v] - h0).dot(side)))
    in_region = {e for f in region for e in f.edges}
    mean_len = sum((pos[e.verts[0]] - pos[e.verts[1]]).length for e in in_region) / max(1, len(in_region))

    def cost(e):
        a, b = pos[e.verts[0]], pos[e.verts[1]]
        d = 0.5 * (abs((a - h0).dot(side)) + abs((b - h0).dot(side)))
        # the back of the skull: an edge whose faces look forward costs more
        n = sum((wn(f) for f in e.link_faces), Vector()).normalized()
        return (a - b).length * (1.0 + 10.0 * d / max(mean_len, 1e-9) + 4.0 * max(0.0, -n.dot(back)))
    got = _cut_dijkstra({start}, lambda v: 1 if v is crown else None, lambda e: e in in_region, pos, cost)
    back_seam = got[0] if got else []
    return ring + back_seam, {"rule": "head", "bone": bname, "armature": arm.name, "regionFaces": len(region),
                              "regionBy": by, "ringEdges": len(ring), "backSeamEdges": len(back_seam),
                              "crown": rnd(list(pos[crown])), "nape": rnd(list(pos[start]))}


def _rule_garment(bm, pos, wn, spec):
    """The garment rule: every piece with two or more open loops (cuffs, hems, a waist, a neck) cut along its inside
    from loop to loop until it is one disc, which lies flat. Prim's tree over the loops from the longest: each cut the
    cheapest edge path from what is joined so far to the nearest loop not yet joined, an edge costing more the more its
    faces look away from the preferred side - toward the mid-plane (inner: the inside of a sleeve or leg), the back, or
    a world direction. Returns (edges, detail)."""
    p_pt, p_no = spec["plane"]
    pref_v, weight = spec["preferVector"], spec["weight"]
    cuts, pieces, paths = [], [], []
    split_edges = []
    if spec["split"]:
        # THE CENTRE LINES FIRST: every edge between a face whose centre is on one side of the mid-plane and a face on
        # the other - the front and back rise of a pair of trousers, a shirt's front and back centre - so the halves
        # come apart whatever the mesh's own centre column. Two half pieces pack tighter than one wide one.
        def side(f):
            c = sum((pos[v] for v in f.verts), Vector()) / len(f.verts)
            return (c - p_pt).dot(p_no) >= 0.0
        split_edges = [e for e in bm.edges if len(e.link_faces) == 2 and not e.seam
                       and side(e.link_faces[0]) != side(e.link_faces[1])]
        for e in split_edges:
            e.seam = True
    for isl in _seam_islands(bm):
        fset = set(isl)
        border = [e for f in isl for e in f.edges
                  if e.seam or len(e.link_faces) != 2 or not all(g in fset for g in e.link_faces)]
        parent = {}

        def find(v):
            r = v
            while parent.get(r, r) is not r:
                r = parent[r]
            while parent.get(v, v) is not v:
                parent[v], v = r, parent[v]
            return r
        for e in set(border):
            a, b = find(e.verts[0]), find(e.verts[1])
            if a is not b:
                parent[a] = b
        loops = {}
        for e in set(border):
            loops.setdefault(find(e.verts[0]), set()).update(e.verts)
        loops = sorted(loops.values(), key=lambda vs: -len(vs))
        if len(loops) == 1:
            continue
        inner = {e for f in isl for e in f.edges
                 if not e.seam and len(e.link_faces) == 2 and all(g in fset for g in e.link_faces)}
        mean_len = sum((pos[e.verts[0]] - pos[e.verts[1]]).length for e in inner) / max(1, len(inner))

        def cost(e):
            a, b = pos[e.verts[0]], pos[e.verts[1]]
            n = sum((wn(f) for f in e.link_faces), Vector())
            n = n.normalized() if n.length > 1e-12 else n
            if pref_v is None:
                mid = (a + b) * 0.5
                dd = (mid - p_pt).dot(p_no)
                want = -p_no if dd > 0 else p_no
            else:
                want = pref_v
            pen = 0.5 * (1.0 - n.dot(want))
            return (a - b).length * (1.0 + weight * pen) + 1e-9 * mean_len
        if not loops:
            # A CLOSED PIECE (an ear shell, a button) has no edge to open from, and an unwrap of a closed surface comes
            # apart face by face. One slit between its two farthest points, on the preferred side, opens it to a disc.
            verts = list({v for f in isl for v in f.verts})
            cen = sum((pos[v] for v in verts), Vector()) / len(verts)
            a = max(verts, key=lambda v: (pos[v] - cen).length)
            b = max(verts, key=lambda v: (pos[v] - pos[a]).length)
            got = _cut_dijkstra({a}, lambda v: 1 if v is b else None, lambda e: e in inner, pos, cost)
            if got:
                cuts.extend(got[0])
                paths.append(got[1])
            pieces.append({"faces": len(isl), "openLoops": 0, "cuts": 1 if got else 0, "closed": True})
            continue
        joined = set(loops[0])
        rest = [set(lp) for lp in loops[1:]]
        owner = {}
        for k, lp in enumerate(rest):
            for v in lp:
                owner[v] = k
        made = 0
        while rest:
            alive = {k for k, lp in enumerate(rest) if lp}
            got = _cut_dijkstra(joined, lambda v: owner.get(v) if owner.get(v) in alive else None,
                                lambda e: e in inner, pos, cost)
            if got is None:
                break
            edges, verts, k = got
            cuts.extend(edges)
            paths.append(verts)
            made += 1
            joined.update(verts)
            joined.update(rest[k])
            rest[k] = set()
            if not any(rest):
                break
        pieces.append({"faces": len(isl), "openLoops": len(loops), "cuts": made})
    # WHERE EACH CUT RUNS, world points along it (at most 64 a cut, 32 cuts), so the seam can be checked without a render
    rows = [{"edges": len(pv) - 1, "points": [rnd(list(pos[v])) for v in pv[:64]], "truncated": len(pv) > 64}
            for pv in paths[:32]]
    return cuts + split_edges, {"rule": "garment", "prefer": spec["prefer"], "pieces": pieces, "cutEdges": len(cuts),
                                "cutPaths": rows, "splitEdges": len(split_edges) if spec["split"] else None}


def _apply_seam_rules(bm, obj, rules):
    """Mark every rule's seams on the bmesh, in order - a later rule sees the seams an earlier one marked. Returns
    (edges marked new, per-rule detail). A rule that marks nothing is reported, not refused: a garment already cut to
    discs has nothing left to cut."""
    mw = obj.matrix_world
    bm.verts.ensure_lookup_table()
    bm.faces.index_update()
    pos = {v: mw @ v.co for v in bm.verts}
    nmat = mw.inverted_safe().transposed().to_3x3()

    def wn(f):
        return (nmat @ f.normal).normalized()
    marked, rows = 0, []
    for kind, spec in rules:
        before = sum(1 for e in bm.edges if e.seam)
        edges, detail = (_rule_head(bm, obj, pos, wn, spec) if kind == "head" else _rule_garment(bm, pos, wn, spec))
        for e in set(edges):
            e.seam = True
        new = sum(1 for e in bm.edges if e.seam) - before
        detail["marked"] = new
        rows.append(detail)
        marked += new
    return marked, rows


def _layout_report(mesh, layer_name, m3, texture=1024):
    """How a UV layer uses the tile: fill (its UV area, 1.0 the whole tile), islands, and texel density - pixels per
    metre at `texture` pixels, overall and the spread between islands (max over min of each island's density)."""
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        uvl = bm.loops.layers.uv.get(layer_name)
        if uvl is None:
            return None
        islands, _fi = _uv_islands(bm, uvl)
        rows = _island_rows(islands, uvl, m3)
        uva = sum(r["_uva"] for r in rows)
        a3 = sum(r["_a3"] for r in rows)
        dens = sorted(math.sqrt(r["_uva"] / r["_a3"]) * texture for r in rows if r["_a3"] > 1e-12 and r["_uva"] > 1e-12)
        return {"fill": round(uva, 4), "islands": len(islands),
                "texelDensity": round(math.sqrt(uva / a3) * texture, 2) if a3 > 1e-12 else None,
                "texelDensityTexture": texture,
                "densitySpread": round(dens[-1] / dens[0], 3) if dens else None,
                "worldArea": round(a3, 6)}
    finally:
        bm.free()


def op_uv_unwrap(params):
    """Generate a UV layer. The addon has REPORTED uvLayers in three places from the
    beginning -- object_info, gen_status, and a quality warning that says in as many
    words "no UVs - texturing and lightmaps will both fail until it is unwrapped" --
    and had nothing that could unwrap. Detects the problem, cannot act on it: the same
    inverse gap set_material_slots and create_water_zone were built to close.

    THREE METHODS, and they are for different jobs:

      SMART      (default) bpy.ops.uv.smart_project. Cuts its own seams by angle. The
                 one to use on a prop that has none, which is most imported geometry.
      LIGHTMAP   bpy.ops.uv.lightmap_pack. Non-overlapping islands in 0-1, which is
                 what Unreal wants from a LIGHTMAP channel specifically. Usually
                 belongs on a SECOND layer - see uvLayer below.
      CONFORMAL, MINIMUM_STRETCH   bpy.ops.uv.unwrap's other solvers on the same seams (2026-10-06);
                 MINIMUM_STRETCH is offered on 4.4 and later and refused by name on 4.2. fillHoles
                 (default true) for these and ANGLE.
      FOLLOW_ACTIVE_QUADS  a quad mesh laid out as a grid from activeFace (default the first quad),
                 set to a rectangle of its world edge lengths first, then scaled into the 0-1 tile with
                 its proportions kept; non-quads are not laid out.
      ANGLE      bpy.ops.uv.unwrap, angle-based. Respects SEAMS you have already
                 marked; with no seams on the mesh it flattens the whole thing as one
                 island and the result is unusable. Warned about rather than refused,
                 because a mesh WITH seams is exactly when you want this.
      BOX        a box projection in METERS: each face takes the two axes across its
                 dominant normal axis, divided by `metersPerTile` (default 1.0; or
                 `texelDensity` pixels per meter on a `textureSize`, default 1024). Two
                 pieces of one kit, of any size, get the same texel density, and pieces
                 laid side by side on a grid that is a multiple of the tile continue the
                 same texture. `boxSpace` OBJECT (default) projects the object's axes
                 with its rotation and scale, about its origin; WORLD adds its location,
                 so the texture is continuous across pieces wherever they stand - and
                 moves under a piece that is moved later. Written straight to the layer,
                 no operator and no edit mode; uvPack is refused with it, because a pack
                 scales the islands to the tile and undoes the world scale. The answer's
                 `box` block measures the density on every face.

    THESE OPERATORS NEED EDIT MODE, and that is worth recording because the sibling
    case goes the other way: bpy.ops.mesh.bevel cannot run under `blender -b` at all -
    it needs a real VIEW_3D area - which is why bevel_edges goes through bmesh.ops
    instead. The UV operators only need mode_set plus a selection, both of which work
    headless. VERIFIED on 4.4.0 and 5.0.1 before this was written, not assumed from the
    bevel case.

    uvLayer NAMES THE TARGET. Unreal reads lightmaps from a second UV channel, so
    generating one usually means adding a layer rather than overwriting the first.
    Passing a name that already exists is refused unless replace:true, because
    silently overwriting an artist's UVs is not a thing to do by default.

    The response reports uvLayersBefore and uvLayersAfter by NAME, so "did it land on
    the channel I meant" is answerable without a second call.
    """
    reject_unknown(params, _UV_KEYS, "uv_unwrap")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)

    # kind=str, because .upper() on a dict is an AttributeError and this handler's contract is
    # MifOpError. Found 2026-09-08 by the version matrix's bad-value sweep, which sends a sentinel
    # at every parameter a payload SENDS - `method` had just been added to uv_unwrap's payload, so
    # it was fuzzed for the first time and crashed instead of refusing.
    method = (take(params, "method", kind=str) or "SMART").upper()
    if method == "ANGLE_BASED":
        method = "ANGLE"
    if method not in ("SMART", "LIGHTMAP", "ANGLE", "BOX", "CONFORMAL", "MINIMUM_STRETCH", "FOLLOW_ACTIVE_QUADS"):
        raise MifOpError(
            "unknown method %r. Use SMART (cuts its own seams by angle - the one for a prop "
            "with none), LIGHTMAP (non-overlapping islands in 0-1, what Unreal wants from a "
            "lightmap channel), ANGLE (respects seams you have already marked), CONFORMAL or "
            "MINIMUM_STRETCH (the same seams, Blender's other solvers), FOLLOW_ACTIVE_QUADS (a quad "
            "mesh laid out as a grid from activeFace), or BOX (a box projection in meters, one texel "
            "density across pieces of any size)." % method)
    # fillHoles AND activeFace (2026-10-06): fill_holes was pinned True for ANGLE (see the unwrap call); it is a
    # parameter now, default true as before, and refused on a method that has no such option.
    fill_holes = take_bool(params, "fillHoles", default=True)
    if params.get("fillHoles") is not None and method not in ("ANGLE", "CONFORMAL", "MINIMUM_STRETCH"):
        raise MifOpError("fillHoles is an option of the ANGLE, CONFORMAL and MINIMUM_STRETCH unwraps, not %s. NOTHING "
                         "was changed." % method)
    active_face = take_int(params, "activeFace", default=None)
    if active_face is not None and method != "FOLLOW_ACTIVE_QUADS":
        raise MifOpError("activeFace is the start face of a FOLLOW_ACTIVE_QUADS unwrap, not %s. NOTHING was changed."
                         % method)
    if method in _UNWRAP_METHODS:
        _offered = [e.identifier for e in bpy.ops.uv.unwrap.get_rna_type().properties["method"].enum_items]
        if _UNWRAP_METHODS[method] not in _offered:
            raise MifOpError("this Blender (%s) has no %s unwrap - its uv.unwrap offers %s. 4.4 and later have it; here "
                             "uv_relax (Blender's Minimize Stretch on a layout already unwrapped) is the nearest. NOTHING "
                             "was changed." % (bpy.app.version_string, method, ", ".join(_offered)))
    # BOX's OWN KEYS, READ AND CHECKED HERE, before anything is written - see the packMargin note below for what reading
    # a value after the unwrap cost this op once. Each is refused beside another method rather than ignored.
    box_tile = _box_tile_size(params, method)

    angle_deg = take_float(params, "angleLimitDeg")
    margin = take_float(params, "islandMargin")
    # kind=str for the same reason as `method` above: this name is handed to
    # mesh.uv_layers.new(name=...), and bpy raises a TypeError for a non-string rather than anything
    # this op can turn into a sentence. Refusing here keeps the contract - every failure a caller can
    # cause is a MifOpError naming the parameter.
    layer_name = take(params, "uvLayer", kind=str)
    replace = take_bool(params, "replace", default=False)
    correct_aspect = take_bool(params, "correctAspect", default=True)
    dry_run = take_bool(params, "dryRun", default=False)

    if angle_deg is not None and method != "SMART":
        raise MifOpError("'angleLimitDeg' applies to SMART only; %s does not take one." % method)
    if angle_deg is None:
        angle_deg = 66.0
    if not (0.0 < angle_deg < 90.0):
        raise MifOpError("'angleLimitDeg' must be between 0 and 90; got %r. 66 is Blender's own "
                         "default and a sane starting point." % angle_deg)
    if margin is None:
        margin = 0.02
    if not (0.0 <= margin < 1.0):
        raise MifOpError("'islandMargin' must be in [0, 1); got %r" % margin)
    # READ UP HERE, BESIDE THE MARGIN IT MIRRORS, and not at the pack call below. take_float
    # refuses a NaN and a non-number, but down there it is inside the try/except that wraps every
    # failure as "uv_unwrap failed" - and by then THE UNWRAP HAS ALREADY WRITTEN TO THE MESH. A
    # caller sending a bad packMargin got a refusal over a layout that had already been replaced.
    # Measured 2026-09-08: packMargin=nan refused with the mesh unwrapped and the old layout gone.
    #
    # NO RANGE CHECK, and that is measured too rather than assumed - see the block at the
    # pack_islands call for what Blender does with -5.0 and 1e6.
    pack_margin = take_float(params, "packMargin")
    # uvPack AND uvTransform, FOR THE SAME REASON - the rest of the lesson the paragraph above records. uvPack
    # was read with take_bool INSIDE the unwrap's try, after the unwrap had written: {uvPack: "flase"} came
    # back "uv_unwrap failed ... 'uvPack' must be a boolean" over a layout already replaced. uvTransform was
    # validated after the unwrap too, its refusals ending "The unwrap above already ran." (both measured on
    # 5.0.1, 2026-09-21). Read here, a bad value changes nothing.
    uv_pack = take_bool(params, "uvPack", default=False)
    uv_xform = _uv_transform_values(take(params, "uvTransform", default=None))
    # A FILL TARGET IS A PACK (2026-11-07): packFill 0-1 packs, then packs again with smaller margins until the layout
    # uses that much of the tile or the margins run out, keeping the fullest - and says which.
    pack_fill = take_float(params, "packFill", default=None)
    if pack_fill is not None:
        if not 0.0 < pack_fill <= 1.0:
            raise MifOpError("packFill is the share of the 0-1 tile the packed layout should use, above 0 and at most "
                             "1 (0.7 is tight for a character); got %r. NOTHING was changed." % pack_fill)
        if params.get("uvPack") is False:
            raise MifOpError("packFill packs - it cannot go with uvPack false. NOTHING was changed.")
        uv_pack = True
    seam_rules = _parse_seam_rules(params.get("seamRules"), obj)
    if seam_rules and method in ("SMART", "LIGHTMAP", "BOX", "FOLLOW_ACTIVE_QUADS"):
        raise MifOpError("seamRules mark seams for an unwrap that follows them, and %s %s - pass method ANGLE (or "
                         "CONFORMAL / MINIMUM_STRETCH). NOTHING was changed."
                         % (method, "cuts its own" if method in ("SMART", "LIGHTMAP") else "ignores seams"))
    # UNWRAP IN PLACE (2026-10-06, 5.2's, here on every build): keepBounds fits each new island, uniformly and
    # centred, into the UV bounds its faces had on the layer before - a re-unwrap that leaves the layout where it was.
    keep_bounds = take_bool(params, "keepBounds", default=False)
    if keep_bounds and (uv_pack or uv_xform is not None or method == "BOX"):
        raise MifOpError("keepBounds puts each island back where its faces were; uvPack%s, uvTransform and method BOX "
                         "place it elsewhere - pass one. NOTHING was changed."
                         % (" (packFill packs)" if pack_fill is not None else ""))
    if method == "BOX" and uv_pack:
        raise MifOpError(
            "uvPack is refused with method BOX: a pack scales and moves the islands to fill the 0-1 tile, which undoes "
            "the meters-per-tile scale BOX exists for. A lightmap channel wants its own layer - pass method LIGHTMAP "
            "with a second uvLayer. NOTHING was changed.")

    mesh = obj.data
    before = [uv.name for uv in mesh.uv_layers]
    faq_quads = None
    if method == "FOLLOW_ACTIVE_QUADS":
        faq_quads = sum(1 for p in mesh.polygons if len(p.vertices) == 4)
        if not faq_quads:
            raise MifOpError("'%s' has no quads, and FOLLOW_ACTIVE_QUADS lays quads out from one. NOTHING was changed."
                             % obj.name)
        if active_face is None:
            active_face = next(p.index for p in mesh.polygons if len(p.vertices) == 4)
        elif not 0 <= active_face < len(mesh.polygons) or len(mesh.polygons[active_face].vertices) != 4:
            raise MifOpError("activeFace %d is not a quad of '%s' (it has %d faces). NOTHING was changed."
                             % (active_face, obj.name, len(mesh.polygons)))

    if layer_name and layer_name in before and not replace:
        raise MifOpError(
            "'%s' already has a UV layer named %r. Pass replace:true to overwrite it, or a "
            "different uvLayer name to add a channel beside it. Overwriting somebody's UVs is "
            "not something to do by default. Existing layers: %s"
            % (obj.name, layer_name, before))
    # THE CAP WAS ALREADY GUARDED HERE and this check is unchanged in substance - only the literal
    # 8 has moved into _UV_LAYER_CAP, which is measured rather than remembered (3.6.23, 4.2.17,
    # 4.4.0 and 5.0.1 all accept 8). Two spellings of the same number in one file is how one of
    # them goes stale.
    if len(before) >= _UV_LAYER_CAP and (not layer_name or layer_name not in before):
        raise MifOpError(
            "'%s' already has %d UV layers and Blender's limit is %d, so there is no room for "
            "another. Pass uvLayer with one of the existing names plus replace:true. Layers: %s"
            % (obj.name, len(before), _UV_LAYER_CAP, before))

    # ---- SEAMS FIRST, and before the ANGLE check below, which is the whole point ----------------
    # That check refuses an ANGLE unwrap on a mesh with no seams, so the endpoint has always offered
    # a method its callers could not use: nothing in this addon could set edge.use_seam. Marking
    # here, in the same call, is what makes ANGLE reachable.
    #
    # The angle criterion is edge.calc_face_angle() - the DIHEDRAL angle between the two faces an
    # edge joins - which is exactly right for seams and is why _select_edges is reused here even
    # though the same grammar was refuted for FACE selection, where a dihedral angle means nothing.
    seams = None
    mark = take(params, "markSeams", default=None)
    clear = take_bool(params, "clearSeams", default=False)
    rule_rows = None
    if mark is not None or clear or seam_rules:
        import bmesh as _bmesh
        bm = _bmesh.new()
        try:
            bm.from_mesh(mesh)
            bm.edges.ensure_lookup_table()
            cleared = 0
            if clear:
                for e in bm.edges:
                    if e.seam:
                        e.seam = False
                        cleared += 1
            marked = 0
            criteria = None
            if mark is not None:
                spec = mark if isinstance(mark, dict) else {}
                if mark is True:
                    # Bare true is ambiguous - "every edge" and "the sharp ones" are different
                    # meshes, and guessing would produce one of them silently.
                    raise MifOpError(
                        "markSeams:true is ambiguous - say which edges. Pass an object using the "
                        "same selectors bevel_edges takes: {minAngleDeg: 40} for everything "
                        "sharper than 40 degrees, {boundaryOnly: true}, {edgeIndices: [...]}, or "
                        "{allEdges: true}. NOTHING was changed.")
                edges, criteria = _select_edges(bm, obj, spec, "uv_unwrap.markSeams")
                for e in edges:
                    if not e.seam:
                        e.seam = True
                        marked += 1
                if not edges:
                    raise MifOpError(
                        "markSeams matched NO edges on '%s', so it would mark nothing and the "
                        "unwrap below would behave as though seams had never been asked for. "
                        "Criteria: %s. NOTHING was changed." % (obj.name, criteria))
            if seam_rules:
                rule_marked, rule_rows = _apply_seam_rules(bm, obj, seam_rules)
                marked += rule_marked
            # A DRY RUN WRITES NO SEAM (2026-10-09). It did: this to_mesh ran before the dry-run return below, so
            # {dryRun: true, markSeams: ...} marked the seams on the mesh and answered "nothing was modified" - found when
            # the bulk preview's suite snapshotted the file around a uv_unwrap preview. The count is the bmesh's instead.
            planned_seams = sum(1 for e in bm.edges if e.seam)
            if not dry_run:
                bm.to_mesh(mesh)
                mesh.update()
        finally:
            bm.free()
        # Read back off the MESH, not the bmesh that wrote it - except on a dry run, which wrote nothing to read.
        seams = {
            "marked": marked,
            "cleared": cleared,
            "seamEdgesNow": planned_seams if dry_run else sum(1 for e in mesh.edges if e.use_seam),
            "criteria": criteria,
            "rules": rule_rows,
        }

    warnings = []
    has_seams = seams["seamEdgesNow"] > 0 if seams is not None else any(e.use_seam for e in mesh.edges)
    if method in ("ANGLE", "CONFORMAL", "MINIMUM_STRETCH") and not has_seams:
        warnings.append(
            "%s unwrap on '%s', which has NO seams marked. Without seams the whole mesh "
            "flattens as a single island and the result is unusable for texturing. Mark seams "
            "first, or use SMART, which cuts its own." % (method, obj.name))
    if not mesh.polygons:
        raise MifOpError("'%s' has no faces, so there is nothing to unwrap." % obj.name)

    if dry_run:
        return {
            "dryRun": True,
            "object": obj.name,
            "method": method,
            "uvLayersBefore": before,
            "targetLayer": layer_name,
            "angleLimitDeg": angle_deg,
            "islandMargin": margin,
            "box": box_tile,
            "seams": seams,
            "warnings": warnings,
            "note": "nothing was modified. Drop dryRun to apply.",
        }

    # Defaulted here because it is only set on the LIGHTMAP branch, and a response field that
    # exists on some paths and raises NameError on others is its own bug.
    lightmap_div = None
    box_report = None

    # The layer has to exist BEFORE the unwrap, and be the ACTIVE one, or the operator writes
    # into whichever channel happened to be active - which is how a lightmap lands on top of
    # the base UVs and nobody notices until the bake.
    # A BACKSTOP, NOT THE CAP CHECK - that one is above and pre-dates this, and it is what actually
    # fires on a full mesh. This guards the DIFFERENT failure: uv_layers.new() signals "no room" by
    # RETURNING None rather than raising, measured on all four builds, so any path that reaches a
    # new() the count check did not cover would take .name off None and produce an AttributeError
    # from the middle of an op that has already entered edit mode.
    #
    # Said plainly because it would be easy to write this up as a fix for something that was
    # already handled: the cap was covered, the None return was not.
    def _new_layer(name):
        if len(mesh.uv_layers) >= _UV_LAYER_CAP:
            raise MifOpError(
                "'%s' already has %d UV layers, which is Blender's maximum - uv_layers.new() "
                "returns None past it rather than failing, so there is no room for '%s'. Remove "
                "one, or pass uvLayer with an existing name plus replace:true. Layers: %s. NOTHING "
                "was changed." % (obj.name, len(mesh.uv_layers), name, ", ".join(before)))
        made = mesh.uv_layers.new(name=name)
        if made is None:
            raise MifOpError("uv_layers.new() returned None for '%s' on '%s' with %d layer(s) "
                             "present, which is how Blender reports 'no room' - it does not raise. "
                             "NOTHING was changed." % (name, obj.name, len(mesh.uv_layers)))
        return made

    created = None
    if layer_name:
        if layer_name in before:
            target = mesh.uv_layers[layer_name]
        else:
            target = _new_layer(layer_name)
            created = target.name
    elif not before:
        target = _new_layer("UVMap")
        created = target.name
    else:
        target = mesh.uv_layers.active or mesh.uv_layers[0]
    if keep_bounds and created:
        mesh.uv_layers.remove(target)
        raise MifOpError("keepBounds refits islands into the bounds they had, and %r is a new layer with none. "
                         "NOTHING was changed." % created)
    mesh.uv_layers.active = target
    active_name = target.name
    old_uv = None
    if keep_bounds:
        old_uv = array.array("f", [0.0]) * (2 * len(target.data))
        target.data.foreach_get("uv", old_uv)
    packed = False          # set where the pack ACTUALLY runs, not from the request
    fill_report = None
    area_before_pack = None  # the layout's UV area just before the pack, measured

    # A FINGERPRINT OF EVERY OTHER LAYER, TAKEN BEFORE THE UNWRAP.
    #
    # THE DEFAULT FAILURE OF A SECOND UV CHANNEL IS WRITING INTO THE FIRST ONE. uv_layers.new()
    # does NOT make the new layer active - measured on 3.6.23, 4.2.17, 4.4.0 and 5.0.1, where
    # active_index stayed 0 after creating a layer - and every UV operator writes to the ACTIVE
    # layer. Miss the `uv_layers.active = target` line above and lightmap_pack silently repacks the
    # base colour UVs while the layer you asked for stays empty. Nothing raises, the response looks
    # correct, and it is found at bake time.
    #
    # This op sets active correctly. What it could not do until now is PROVE it: activeLayer and
    # createdLayer both report what was INTENDED, and neither can disagree with itself. So the
    # other layers are fingerprinted before and after, and the answer is a measurement.
    #
    # foreach_get, not a Python loop - it is a C-level bulk copy, so this costs almost nothing on a
    # mesh where a per-loop comprehension would be the expensive part of the whole op.
    def _uv_fingerprint(layer):
        n = len(layer.data)
        if not n:
            return (0, b"")
        buf = array.array("f", [0.0]) * (2 * n)
        layer.data.foreach_get("uv", buf)
        return (n, hashlib.sha256(buf.tobytes()).digest())

    others_before = {uv.name: _uv_fingerprint(uv) for uv in mesh.uv_layers
                     if uv.name != active_name}

    prev_active = bpy.context.view_layer.objects.active
    prev_mode = obj.mode
    # ONLY THIS OBJECT MAY BE SELECTED, and the rest of the selection is put back afterwards. Edit mode takes in EVERY
    # selected mesh (multi-object editing), and smart_project, lightmap_pack, unwrap and pack_islands then lay all of
    # them out in ONE shared tile: the target's islands come back crammed into a corner of it, and every other selected
    # object's UVs are rewritten without being named in the call. Found 2026-09-27 on Blender 5.2 (a gun model's
    # parts): a barrel unwrapped this way used 0.03% of its tile, in a strip, because the frame, the slide and four more
    # parts were still selected from an earlier op; the same call with the barrel alone used 53%, and 63% packed.
    sel_snapshot = selection_snapshot()
    try:
        if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        select_only([obj])
        bpy.context.view_layer.objects.active = obj
        if method != "BOX":
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
        if method == "BOX":
            # IN OBJECT MODE, ON PURPOSE: a write to mesh.uv_layers in edit mode is overwritten by the edit-mode bmesh
            # on the way out. No operator is involved, so nothing remembered from an earlier call can change it.
            box_report = _box_project(obj, mesh, active_name, box_tile)
        elif method == "SMART":
            # Pinned for the reason _pinned_op_kwargs gives: a remembered area_weight or scale_to_bounds from an
            # earlier call would change this layout with nothing in the request saying so.
            bpy.ops.uv.smart_project(angle_limit=math.radians(angle_deg),
                                     island_margin=margin,
                                     correct_aspect=correct_aspect,
                                     **_pinned_op_kwargs(bpy.ops.uv.smart_project,
                                                         {"area_weight": 0.0, "scale_to_bounds": False,
                                                          "margin_method": "SCALED", "rotate_method": "AXIS_ALIGNED_Y"}))
        elif method == "LIGHTMAP":
            # PREF_MARGIN_DIV IS CAPPED AT 1.0, read off its own rna_type on 3.6.23, 4.2.17, 4.4.0
            # and 5.0.1 - hard_min 0.001, hard_max 1.0 on every one of them.
            #
            # This used to pass `margin * 100.0`. islandMargin is validated to [0, 1), so anything
            # from 0.01 upward multiplied past the cap and was clamped by RNA to exactly 1.0 -
            # INCLUDING this op's own default of 0.02. The entire declared range above 0.01 was one
            # value, and the response echoed the requested margin back as though it had applied.
            # A parameter that reports itself applied and does nothing is worse than one that is
            # missing, because nobody goes looking.
            #
            # Passed through directly now, floored at the operator's own minimum. What was actually
            # handed over is reported, so a caller can see it rather than infer it.
            lightmap_div = max(0.001, min(1.0, margin))
            bpy.ops.uv.lightmap_pack(PREF_MARGIN_DIV=lightmap_div)
        elif method == "FOLLOW_ACTIVE_QUADS":
            _follow_quads_unwrap(obj, mesh, active_face)
        else:
            # fill_holes IS PASSED EVERY TIME, and that is not tidiness. Its DEFAULT FLIPPED at 4.4:
            # True on 3.6.23 and 4.2.17, False on 4.4.0 and 5.0.1 - read off the operator's own
            # rna_type on each build rather than from release notes. Left unpassed, the identical
            # call fills holes on the LTS builds and does not on the newer ones, so the same op on
            # the same mesh produces two different layouts depending on which Blender is running,
            # with nothing anywhere saying so.
            #
            # `method` was already pinned to ANGLE_BASED, which is why THAT flip - ANGLE_BASED to
            # CONFORMAL, same version - never bit. This is the same lesson, one argument over.
            #
            # It was pinned to True, the behaviour this op had on the builds it was written and tested
            # against; since 2026-10-06 it is the fillHoles parameter, default True, so the default
            # layout is the same on every build and a caller can still turn it off.
            bpy.ops.uv.unwrap(method=_UNWRAP_METHODS[method], margin=margin, fill_holes=fill_holes,
                              correct_aspect=correct_aspect)

        # PACK AFTER THE UNWRAP, never before - packing first packs the OLD layout, which looks
        # like it worked and is simply the previous islands rearranged.
        if uv_pack:
            # packMargin IS NOT RANGE-CHECKED, and that is a measurement rather than an oversight.
            # islandMargin is validated to [0, 1) because bpy.ops.uv.unwrap does not clamp it;
            # pack_islands does. Probed on Blender 4.4 (2026-09-08), cube, SMART, bounds read back
            # off the layer and cross-checked against mesh_quality:
            #
            #   packMargin     packed bounds                    uvAreaTotal  outside 0-1
            #   0.01           u 0.0020..0.9980 v 0.0020..0.6647   0.6510     no
            #   0.0            u 0.0000..1.0000 v 0.0000..0.6667   0.6667     no
            #   -5.0           u 0.0000..1.0000 v 0.0000..0.6667   0.6667     no   <- clamped to 0
            #   1e6            u 0.0909..0.5758 v 0.0909..0.9091   0.1378     no   <- clamped high
            #
            # Every value stayed inside the 0-1 tile. Nothing flew off and nothing came back broken
            # while reporting success, so a bound here would invent a refusal Blender does not have.
            # A huge margin still COSTS you - 1e6 uses a fifth of the UV area of a sane one - which
            # is why packedBounds is reported below instead: say what happened, do not forbid it.
            bpy.ops.uv.select_all(action="SELECT")
            bpy.ops.object.mode_set(mode="OBJECT")
            area_before_pack = _uv_layout_area(mesh, active_name)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.select_all(action="SELECT")
            # EVERY PROPERTY PINNED (see _pinned_op_kwargs): margin alone left rotate, scale and the island shape to
            # whatever the session last used, and a pack that neither scales nor rotates barely moves anything.
            first_margin = margin if pack_margin is None else pack_margin
            if pack_fill is not None:
                # ONE TEXEL DENSITY FIRST: every island scaled to its share of the surface, so the tight pack below
                # cannot buy fill by blowing one island up
                bpy.ops.uv.average_islands_scale()
            bpy.ops.uv.pack_islands(margin=first_margin, **_pinned_op_kwargs(bpy.ops.uv.pack_islands, _PACK_PINNED))
            packed = True
            if pack_fill is not None:
                fill_attempts = []
                best = None
                for k, m_try in enumerate((first_margin, first_margin / 2.0, first_margin / 4.0, first_margin / 8.0)):
                    if k:
                        bpy.ops.uv.select_all(action="SELECT")
                        bpy.ops.uv.pack_islands(margin=m_try,
                                                **_pinned_op_kwargs(bpy.ops.uv.pack_islands, _PACK_PINNED))
                    bpy.ops.object.mode_set(mode="OBJECT")
                    got = _uv_layout_area(mesh, active_name)
                    fill_attempts.append({"margin": round(m_try, 6), "fill": round(got, 4)})
                    if best is None or got > best[0]:
                        lay = mesh.uv_layers[active_name]
                        buf = array.array("f", [0.0]) * (2 * len(lay.data))
                        lay.data.foreach_get("uv", buf)
                        best = (got, m_try, buf)
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.mesh.select_all(action="SELECT")
                    if got >= pack_fill:
                        break
                bpy.ops.object.mode_set(mode="OBJECT")
                # THE FULLEST LAYOUT IS THE ONE KEPT, which need not be the last tried
                mesh.uv_layers[active_name].data.foreach_set("uv", best[2])
                mesh.update()
                fill_report = {"target": pack_fill, "achieved": round(best[0], 4), "met": best[0] >= pack_fill,
                               "margin": round(best[1], 6), "attempts": fill_attempts}
                if not fill_report["met"]:
                    warnings.append(
                        "the pack reached %.0f%% of the tile, short of packFill %.0f%%, at margins down to %g. More "
                        "islands than the layout needs, or long thin ones, cap the fill: seamRules or fewer seams "
                        "help more than a smaller margin." % (best[0] * 100.0, pack_fill * 100.0, fill_attempts[-1]["margin"]))
    except Exception as exc:
        # A layer created for an unwrap that then threw is debris that would change the next
        # export without appearing in any response.
        if created and created in [uv.name for uv in mesh.uv_layers]:
            mesh.uv_layers.remove(mesh.uv_layers[created])
        raise MifOpError("uv_unwrap failed on '%s' (%s): %s: %s"
                         % (obj.name, method, type(exc).__name__, exc))
    finally:
        try:
            if obj.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            if prev_mode and prev_mode != "OBJECT":
                pass    # deliberately left in OBJECT: headless has no mode to return to
            selection_restore(sel_snapshot)
            bpy.context.view_layer.objects.active = prev_active
        except Exception:
            pass

    # ---- WHERE THE PACK ACTUALLY LANDED --------------------------------------------------------
    # `packed` used to be `bool(take_bool(params, "uvPack"))` - the REQUEST, echoed back. It could
    # not disagree with itself, which is the same fault this op already calls out for
    # lightmapMarginDiv two fields down. The point of packing is the resulting layout, and the
    # layout was the one thing the response never mentioned.
    #
    # Measured, not asserted: packMargin=1e6 packs into a fifth of the UV area of a sane margin
    # (see the table at the pack call). That is legal, Blender clamps rather than breaking, and it
    # is invisible unless somebody reports the bounds. So report them.
    kept = _refit_islands(mesh, active_name, old_uv) if old_uv is not None else None
    packed_bounds = None
    if packed:
        _layer = mesh.uv_layers.get(active_name) or mesh.uv_layers.active
        if _layer is not None and len(_layer.data):
            _n = len(_layer.data)
            _flat = [0.0] * (_n * 2)
            _layer.data.foreach_get("uv", _flat)
            _us, _vs = _flat[0::2], _flat[1::2]
            packed_bounds = {
                "min": [min(_us), min(_vs)],
                "max": [max(_us), max(_vs)],
                "loops": _n,
                # A packed layout that leaves 0-1 is the failure this readback exists to catch.
                # Blender clamped every margin probed, so this should be false - if it is ever
                # true, the margin bound this op deliberately does not have is suddenly needed.
                "outside01": bool(min(_us) < -1e-6 or min(_vs) < -1e-6
                                  or max(_us) > 1.0 + 1e-6 or max(_vs) > 1.0 + 1e-6),
            }
            # HOW MUCH OF THE TILE THE LAYOUT USES, before and after the pack: the proof that the pack did
            # something. A pack that moves nothing answers packed:true with the two numbers equal.
            packed_bounds["uvAreaBeforePack"] = (round(area_before_pack, 4) if area_before_pack is not None else None)
            packed_bounds["uvArea"] = round(_uv_layout_area(mesh, active_name), 4)
            if area_before_pack is not None and packed_bounds["uvArea"] < area_before_pack * 1.02 \
                    and packed_bounds["uvArea"] < 0.5:
                warnings.append(
                    "the pack left the layout using %.0f%% of the tile (%.0f%% before it): islands this small "
                    "waste most of a baked texture." % (packed_bounds["uvArea"] * 100.0, area_before_pack * 100.0))
            if packed_bounds["outside01"]:
                warnings.append(
                    "the packed layout extends outside 0-1 (u %.4f..%.4f, v %.4f..%.4f). "
                    "pack_islands is supposed to keep it inside the tile; a lightmap channel "
                    "like this will bake wrong in Unreal."
                    % (min(_us), max(_us), min(_vs), max(_vs)))

    # ---- TRANSFORM LAST, on the layer that was just written ------------------------------------
    # Plain RNA on the UV loops rather than an operator: no mode change, no selection, and the
    # postcondition is the bounds read back off the layer afterwards.
    uv_transform = None
    if uv_xform is not None:
        sx, sy, ox, oy = uv_xform          # parsed at the top, before the unwrap - see _uv_transform_values
        layer = mesh.uv_layers.get(active_name) or mesh.uv_layers.active
        if layer is None:
            raise MifOpError("no UV layer to transform after the unwrap, which should be "
                             "impossible - reported rather than crashing on None.")
        n = len(layer.data)
        flat = [0.0] * (n * 2)
        layer.data.foreach_get("uv", flat)
        before_u = flat[0::2]
        before_v = flat[1::2]
        for i in range(n):
            flat[i * 2] = flat[i * 2] * sx + ox
            flat[i * 2 + 1] = flat[i * 2 + 1] * sy + oy
        layer.data.foreach_set("uv", flat)
        mesh.update()

        check = [0.0] * (n * 2)
        layer.data.foreach_get("uv", check)
        us, vs = check[0::2], check[1::2]
        # Read back off the layer - foreach_set reports nothing, and a transform that did not take
        # would leave a layout that looks unwrapped and is in the wrong place.
        uv_transform = {
            "scale": [sx, sy],
            "offset": [ox, oy],
            "loops": n,
            "boundsBefore": {"min": [min(before_u), min(before_v)],
                             "max": [max(before_u), max(before_v)]} if n else None,
            "boundsAfter": {"min": [min(us), min(vs)], "max": [max(us), max(vs)]} if n else None,
        }
        if n and (min(us) < -1e-6 or min(vs) < -1e-6 or max(us) > 1.0 + 1e-6
                  or max(vs) > 1.0 + 1e-6):
            # Said, not refused: leaving 0-1 is legal and sometimes deliberate (tiling), but it is
            # NOT what Unreal wants from a lightmap channel and nothing else would mention it.
            warnings.append(
                "after uvTransform the layout extends outside 0-1 (u %.4f..%.4f, v %.4f..%.4f). "
                "That is legal and fine for a tiling texture, but a lightmap channel must stay "
                "inside 0-1 or Unreal will pack it wrong."
                % (min(us), max(us), min(vs), max(vs)))

    after = [uv.name for uv in mesh.uv_layers]
    _clobbered = sorted(n for n, fp in others_before.items()
                        if n in mesh.uv_layers and _uv_fingerprint(mesh.uv_layers[n]) != fp)
    _active_render = next((uv.name for uv in mesh.uv_layers if uv.active_render), None)
    return {
        # SHARED MESH DATA - see ops_common.shared_data_note. This edit lands on every
        # object sharing the datablock, and nothing here said so.
        **shared_data_note(obj),
        "object": obj.name,
        "method": method,
        "uvLayersBefore": before,
        "uvLayersAfter": after,
        "activeLayer": active_name,
        "createdLayer": created,
        # MEASURED, NOT ASSERTED. See the fingerprint above: this is the difference between
        # "the op meant to write to the layer you named" and "no other layer moved".
        "otherLayersUnchanged": not _clobbered,
        "layersClobbered": _clobbered,
        # WHICH LAYER THE RENDERER READS FOR TEXTURES, which is not the same as the active one and
        # is what a caller adding a lightmap channel needs to see stayed put. Unreal reads the base
        # colour from active_render and the lightmap from a second channel; if a lightmap pass has
        # moved active_render, the material is now sampling the lightmap layout.
        "activeRenderLayer": _active_render,
        "angleLimitDeg": angle_deg if method == "SMART" else None,
        "fillHoles": fill_holes if method in _UNWRAP_METHODS else None,
        "activeFace": active_face if method == "FOLLOW_ACTIVE_QUADS" else None,
        "facesNotQuads": (len(mesh.polygons) - faq_quads) if faq_quads is not None else None,
        "islandMargin": margin,
        # WHAT THE OPERATOR ACTUALLY GOT, on the one path where it is not the margin
        # itself. None on every other method. Reported because this value was silently
        # pinned at its 1.0 cap for every islandMargin from 0.01 up, the default among
        # them, while this response echoed the request back.
        "lightmapMarginDiv": lightmap_div,
        # method BOX only: the tile size used and the density MEASURED on the layer afterwards, per face.
        "box": box_report,
        "faces": len(mesh.polygons),
        "seams": seams,
        # MEASURED AT THE PACK CALL, not read back off the request. See packedBounds.
        "packed": packed,
        "packMargin": pack_margin if packed else None,
        # packFill's answer: the target, the fill reached, the margin kept and every pack tried
        "packFill": fill_report,
        # HOW THE LAYOUT USES THE TILE, read off the layer: fill, islands, and texel density (pixels per metre at
        # 1024, overall, and the spread between islands)
        "layout": _layout_report(mesh, active_name, obj.matrix_world.to_3x3()),
        "packedBounds": packed_bounds,
        "uvTransform": uv_transform,
        "keepBounds": kept,
        "warnings": warnings,
        "note": ("the unwrap wrote into %r. Unreal reads lightmaps from a SECOND UV channel, so "
                 "pass uvLayer to put a LIGHTMAP pass somewhere other than the base UVs."
                 % active_name),
    }


_DECIMATE_KEYS = {
    "object", "name",
    "ratio", "targetTris", "targetTriangles",
    "mode", "angleLimit", "iterations", "keepNgons",
    "dryRun", "previewId", "previewTtlSeconds"}


def op_decimate_mesh(params):
    """Reduce triangle count. The edit a game pipeline wants most, and the one
    analyze_skeletal_split's triangle counts had nowhere to send their answer.

    THREE MODES, because Blender's decimate modifier is three different algorithms
    wearing one name, and they fail in different ways:

      COLLAPSE   (default) ratio-driven edge collapse. The general-purpose one.
                 Give it `ratio` (0-1) or `targetTris` and it solves for the ratio.
      UNSUBDIV   reverses subdivision. Only sensible on quad grids that WERE
                 subdivided; on arbitrary geometry it mangles.
      DISSOLVE   planar merge by `angleLimit`. Removes only geometry that was flat
                 anyway, so it is the lossless-looking one -- and it may remove
                 NOTHING on an already-tight mesh, which is not a failure.

    WHAT IT REPORTS IS WHAT HAPPENED, NOT WHAT WAS ASKED. A collapse decimate
    almost never lands exactly on the requested ratio -- it solves for a face
    budget and cannot split a triangle to hit a target -- so the response carries
    trisBefore, trisAfter and the ratioAchieved that actually resulted, beside the
    ratioRequested. Echoing the request back would be the silent-wrong-number this
    project keeps finding.

    IT REFUSES TO PRETEND. If the modifier removed nothing at all, that is said in
    words rather than returned as a cheerful ok with two identical counts - and so is
    a mesh left with NO faces (emptied: true), which a DISSOLVE near 180 degrees does
    to any closed shape. angleLimit outside 0-180 degrees and iterations outside
    0-32767 are refused rather than clamped by Blender in silence, and so is a
    parameter the chosen mode does not read.

    UVs AND CUSTOM NORMALS. Decimation rewrites topology, so a UV layer is
    stretched across the survivors and custom split normals do not survive a
    collapse. Both are WARNED about when present rather than silently damaged --
    the caller may not know the mesh had them.

    OUTPUT THAT PASSES mesh_quality gameReady (2026-11-01): a DISSOLVE's merged n-gons are split into triangles, the
    surface unchanged (keepNgons true leaves them, for a mesh still being modeled); a fin a COLLAPSE folds, a triangle
    flapping off an edge that now has three faces, is removed (finFacesRemoved); faces whose UVs collapse are opened.
    """
    reject_unknown(params, _DECIMATE_KEYS, "decimate_mesh")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)

    mode = (take(params, "mode") or "COLLAPSE").upper()
    if mode not in ("COLLAPSE", "UNSUBDIV", "DISSOLVE"):
        raise MifOpError(
            "unknown mode %r. Use COLLAPSE (ratio-driven, the general one), UNSUBDIV "
            "(reverses subdivision; only sensible on quad grids that WERE subdivided), "
            "or DISSOLVE (planar merge by angleLimit, removes only what was already "
            "flat)." % mode)

    ratio = take_float(params, "ratio")
    target = take_int(params, "targetTris", "targetTriangles")
    angle_limit = take_float(params, "angleLimit")
    iterations = take_int(params, "iterations")
    dry_run = take_bool(params, "dryRun", default=False)
    keep_ngons = take_bool(params, "keepNgons", default=False)
    if keep_ngons and mode != "DISSOLVE":
        raise MifOpError("keepNgons is for mode DISSOLVE: %s leaves no n-gon of its own. NOTHING was changed." % mode)

    before = mesh_counts(obj)
    tris0 = before.get("tris", -1)

    # A PARAMETER THE MODE DOES NOT READ IS REFUSED, NOT ECHOED. Measured on 5.0.1, 2026-09-21: COLLAPSE {ratio: 0.5,
    # angleLimit: 30}, DISSOLVE {angleLimit: 10, iterations: 3} and UNSUBDIV {iterations: 1, angleLimit: 10} each ran,
    # ignored the extra value, and reported it back under angleLimit or iterations as though it had been used. ratio and
    # targetTris outside COLLAPSE were already refused this way.
    unused = [key for key, value, reader in (("angleLimit", angle_limit, "DISSOLVE"),
                                             ("iterations", iterations, "UNSUBDIV"))
              if value is not None and mode != reader]
    if unused:
        raise MifOpError(
            "%s %s nothing in %s - angleLimit drives DISSOLVE only, iterations drives UNSUBDIV only, and COLLAPSE "
            "takes ratio or targetTris. It is refused rather than ignored. NOTHING was changed."
            % (" and ".join("'%s'" % k for k in unused), "does" if len(unused) == 1 else "do", mode))

    if mode == "COLLAPSE":
        if (ratio is None) == (target is None):
            raise MifOpError(
                "COLLAPSE needs exactly one of 'ratio' (0-1, the fraction of faces to KEEP) "
                "or 'targetTris' (an absolute triangle count to aim for). Passing both is "
                "ambiguous and passing neither has no meaning. This mesh has %d triangles "
                "now." % tris0)
        if target is not None:
            if target < 1:
                raise MifOpError("'targetTris' must be >= 1; got %d" % target)
            if tris0 <= 0:
                raise MifOpError(
                    "cannot solve a ratio for targetTris: this mesh reports %d triangles, so "
                    "there is nothing to divide by. Pass 'ratio' directly." % tris0)
            if target >= tris0:
                raise MifOpError(
                    "targetTris %d is not BELOW the current %d, so there is nothing to "
                    "decimate. Decimation only removes; it cannot add detail."
                    % (target, tris0))
            ratio = float(target) / float(tris0)
        if not (0.0 < ratio <= 1.0):
            raise MifOpError(
                "'ratio' must be in (0, 1]; got %r. It is the fraction of faces to KEEP, so "
                "0.25 means a quarter of them." % ratio)
    else:
        if ratio is not None or target is not None:
            raise MifOpError(
                "'ratio' and 'targetTris' apply to COLLAPSE only. %s is driven by %s."
                % (mode, "angleLimit" if mode == "DISSOLVE" else "iterations"))
        if mode == "DISSOLVE" and angle_limit is None:
            angle_limit = 5.0
        if mode == "UNSUBDIV" and iterations is None:
            iterations = 1
        # REFUSED OUTSIDE THE MODIFIER'S OWN RANGE, because Blender clamps without a word. Measured on 5.0.1 against
        # a 960-triangle UV sphere, 2026-09-21: DISSOLVE angleLimit 400 (and 180) left 2 vertices, 1 edge and no
        # faces with ok:true and no note, echoing 400; angleLimit -5 removed nothing and the note blamed coplanarity;
        # UNSUBDIV iterations -3 removed nothing and 40000 ran as 32767, down to one triangle, each echoed as asked.
        # The RNA ranges are angle_limit [0, pi] and iterations [0, 32767], the same on 3.6.23 and 5.2.1.
        if mode == "DISSOLVE" and not (0.0 <= angle_limit <= 180.0):
            raise MifOpError(
                "'angleLimit' is an angle in DEGREES from 0 to 180 - the Decimate modifier's own range - and got %r. "
                "Blender clamps a value outside it without saying so: 400 ran as 180 and dissolved a 960-triangle "
                "sphere to 2 vertices and no faces, and -5 removed nothing. NOTHING was changed." % angle_limit)
        if mode == "UNSUBDIV" and not (0 <= iterations <= 32767):
            raise MifOpError(
                "'iterations' must be 0 to 32767 - the Decimate modifier's own range - and got %d. Blender clamps a "
                "value outside it without saying so: -3 removed nothing and 40000 ran as 32767. NOTHING was changed."
                % iterations)

    warnings = []
    mesh = obj.data
    if mode == "COLLAPSE" and mesh.has_custom_normals:
        warnings.append(
            "'%s' carries custom split normals. A collapse decimate rewrites topology and they "
            "will NOT survive it -- re-author normals afterwards, or use DISSOLVE, which only "
            "removes geometry that was already planar." % obj.name)
    if getattr(mesh, "uv_layers", None) and len(mesh.uv_layers) > 0:
        warnings.append(
            "'%s' has %d UV layer(s). Decimation stretches UVs across the surviving vertices; it "
            "does not re-unwrap. Check the result before baking anything against it."
            % (obj.name, len(mesh.uv_layers)))

    if dry_run:
        return {
            "dryRun": True,
            "object": obj.name,
            "mode": mode,
            "trisBefore": tris0,
            "countsBefore": before,
            "ratioRequested": round(ratio, 6) if ratio is not None else None,
            "targetTris": target,
            "angleLimit": angle_limit,
            "iterations": iterations,
            "warnings": warnings,
            "note": "nothing was modified. Drop dryRun to apply.",
        }

    # THROUGH THE MODIFIER, not a hand-rolled bmesh collapse. Blender's decimate is a
    # quadric-error solver and anything written here would be a worse one wearing the same name.
    mod = obj.modifiers.new(name="MifDecimate", type="DECIMATE")
    # THE NAME AS A PYTHON STRING, taken now. modifier_apply frees the modifier, and the except
    # below covers that call - so reading mod.name in the handler, or handing `mod` to remove(),
    # would be a read of released RNA memory. boolean_op did exactly that and raised
    # UnicodeDecodeError from inside a plain `.name` on Blender 5.0.1 while passing on 3.6/4.2/4.4.
    mod_name = str(mod.name)
    try:
        mod.decimate_type = mode
        # READ BACK BEFORE THE APPLY FREES THE MODIFIER: the reply reports what Blender stored, not the request.
        if mode == "COLLAPSE":
            mod.ratio = ratio
        elif mode == "UNSUBDIV":
            mod.iterations = iterations
            iterations = int(mod.iterations)
        else:
            mod.angle_limit = math.radians(angle_limit)
            angle_limit = round(math.degrees(mod.angle_limit), 4)

        prev = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = obj
        try:
            bpy.ops.object.modifier_apply(modifier=mod_name)
        finally:
            bpy.context.view_layer.objects.active = prev
    except Exception as exc:
        # Leave nothing behind on the failure path. A stranded modifier would change the next
        # export without appearing in any response -- an invisible edit is the worst kind.
        stale = obj.modifiers.get(mod_name)
        if stale is not None:
            obj.modifiers.remove(stale)
        raise MifOpError("decimate failed on '%s': %s: %s" % (obj.name, type(exc).__name__, exc))

    # A monkey at any ratio from 0.1 to 0.5 grows one fin (5.2.1, 2026-11-01), which gameReady fails and nothing later mends.
    ngons_split = fins = 0
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        if mode == "DISSOLVE" and not keep_ngons:
            ngons_split = triangulate_ngons(bm)[0]
        if mode == "COLLAPSE":
            fins = drop_doubled_and_fin_faces(bm)
        if ngons_split or fins:
            bm.to_mesh(obj.data)
            obj.data.update()
    finally:
        bm.free()
    uv_opened = open_collapsed_uvs_of(obj)
    after = mesh_counts(obj)
    tris1 = after.get("tris", -1)
    achieved = (float(tris1) / float(tris0)) if tris0 > 0 and tris1 >= 0 else None

    result = {
        "object": obj.name,
        "mode": mode,
        "trisBefore": tris0,
        "trisAfter": tris1,
        "trisRemoved": (tris0 - tris1) if (tris0 >= 0 and tris1 >= 0) else None,
        # round(), not rnd(). rnd maps over a SEQUENCE - it exists for vectors - and passing it a
        # scalar raised "float object is not iterable" on every successful path while every
        # REFUSAL path returned cleanly, so the guards all looked right and the op never ran.
        "ratioRequested": round(ratio, 6) if ratio is not None else None,
        "ratioAchieved": round(achieved, 6) if achieved is not None else None,
        "ngonsSplit": ngons_split,
        "finFacesRemoved": fins,
        "uvFacesOpened": uv_opened,
        "targetTris": target,
        "angleLimit": angle_limit,
        "iterations": iterations,
        "countsBefore": before,
        "countsAfter": after,
        "warnings": warnings,
    }

    # SAID IN WORDS, not left to be spotted in two identical numbers. An op reporting ok while
    # changing nothing is the exact failure shape this codebase keeps finding.
    #
    # AND AN EMPTIED MESH IS SAID IN WORDS TOO, as boolean_op says it. A DISSOLVE at 180 degrees merged every face of a
    # closed sphere and answered ok:true with trisAfter 0 and no note (5.0.1, 2026-09-21) - a result nobody asks for,
    # reported like any other.
    if tris0 > 0 and after.get("faces") == 0:
        result["emptied"] = True
        result["note"] = (
            "the mesh is now EMPTY - every face was removed, and %d vert%s and %d edge%s are left. %s bl_undo takes "
            "it back." % (after.get("verts"), "ex" if after.get("verts") == 1 else "ices",
                          after.get("edges"), "" if after.get("edges") == 1 else "s",
                          "A DISSOLVE at %s degrees merges every face of a closed shape into nothing; a few degrees is "
                          "the usual range." % angle_limit if mode == "DISSOLVE" else "Almost certainly not what was "
                          "wanted."))
    elif tris0 >= 0 and tris1 == tris0:
        result["nothingRemoved"] = True
        result["note"] = "the mesh has the SAME %d triangles it started with -- nothing was removed. %s" % (
            tris0, {"DISSOLVE": "No faces were coplanar within angleLimit=%s degrees; raise it to merge more."
                                % angle_limit,
                    "UNSUBDIV": "UNSUBDIV found nothing it could un-subdivide at iterations=%s." % iterations,
                    }.get(mode, "The ratio was too close to 1 to drop a single face."))
    elif ratio is not None and achieved is not None and abs(achieved - ratio) > 0.05:
        result["note"] = (
            "landed at ratio %.3f against the %.3f requested. A collapse decimate solves for a "
            "face budget and cannot split a triangle to hit a target exactly; this is the real "
            "figure, not a rounding of the request." % (achieved, ratio))
    return result


def _skin_world(meshes):
    """Every mesh's evaluated vertices in world space (armature pose and modifiers applied)."""
    dg = bpy.context.evaluated_depsgraph_get()
    out = {}
    for m in meshes:
        ev = m.evaluated_get(dg)
        me = ev.to_mesh()
        mw = ev.matrix_world.copy()
        out[m.name] = [mw @ v.co for v in me.vertices]
        ev.to_mesh_clear()
    return out


def _apply_armature_transform(obj, do_loc, do_rot, do_scale, with_meshes):
    """apply_transform on an ARMATURE: the armature and its skinned meshes together, the skin left where it was."""
    in_layer = set(bpy.context.view_layer.objects)
    meshes = [] if not with_meshes else sorted(
        (o for o in bpy.data.objects if o.type == "MESH" and o in in_layer
         and (o.parent is obj or any(md.type == "ARMATURE" and md.object is obj for md in o.modifiers))),
        key=lambda o: o.name)
    shared = [o.name for o in [obj] + meshes if o.data.users > 1]
    if shared:
        raise MifOpError("%s share their data with other objects, and applying a transform would move those too. Make "
                         "the data single-user first. NOTHING was changed." % ", ".join(shared))
    keyed = []
    ad = obj.animation_data
    act = ad.action if ad is not None else None
    if act is not None and do_scale and any(abs(s - 1.0) > 1e-9 for s in obj.scale):
        from .ops_anim import _fcurves
        keyed = sorted({fc.data_path.split('"')[1] for fc in _fcurves(obj)
                        if fc.data_path.startswith("pose.bones[") and fc.data_path.endswith("].location")})
    before_skin = _skin_world(meshes)
    heads = {b.name: obj.matrix_world @ b.head_local for b in obj.data.bones}
    before = object_info(obj)
    snap = selection_snapshot()
    try:
        select_only([obj] + meshes)
        bpy.ops.object.transform_apply(location=do_loc, rotation=do_rot, scale=do_scale)
    finally:
        selection_restore(snap)
    bpy.context.view_layer.update()
    after_skin = _skin_world(meshes)
    moved = {n: max((a - b).length for a, b in zip(after_skin[n], before_skin[n])) if before_skin[n] else 0.0
             for n in before_skin}
    head_move = max(((obj.matrix_world @ b.head_local) - heads[b.name]).length for b in obj.data.bones) \
        if obj.data.bones else 0.0
    after = object_info(obj)
    out = {"object": obj.name, "type": "ARMATURE",
           "applied": {"location": do_loc, "rotation": do_rot, "scale": do_scale},
           "meshes": [{"name": m.name, "isIdentityTransform": object_info(m)["isIdentityTransform"],
                       "maxSkinMove": round(moved[m.name], 7)} for m in meshes],
           "maxSkinMove": round(max(moved.values()), 7) if moved else None,
           "maxBoneHeadMove": round(head_move, 7),
           "before": before, "after": after, "isIdentityTransform": after["isIdentityTransform"]}
    if keyed:
        out["animationNote"] = ("the scale was applied to the bones' rest, and the action '%s' keys location on %s - "
                                "those keys are in the old scale's units, so the animation now moves those bones by a different distance."
                                % (act.name, ", ".join(keyed[:10])))
    if not meshes and with_meshes:
        out["note"] = "no mesh is parented to '%s' or deformed by it; the armature alone was applied." % obj.name
    return out


def op_apply_transform(params):
    """Bake an object's loc/rot/scale into its MESH DATA, restoring the identity transform.

    THE ONE THAT MAKES THE FIDELITY GATE SATISFIABLE. object_info reports
    isIdentityTransform, and mif_mesh_roundtrip asserts it before and after every
    edit, because a non-identity object transform means the pivot moved -- an FBX
    written from an object scaled 0.01 at the OBJECT level imports into Unreal at a
    size nobody asked for, and the mesh's own vertices still disagree with what the
    viewport showed. Until now the addon could DETECT that state and had no way out
    of it: nothing could apply a transform. Detecting a problem you cannot fix is
    the same half-a-subsystem shape uv_unwrap closed for UVs.

    WHICH CHANNELS, and why they are separate. Name none of `location`, `rotation`
    and `scale` and all three apply; name ANY of them and only the named ones do -
    the rest default to false. They are independent on purpose: baking rotation into
    a mesh destined for a rig is usually wrong (the armature expects the object
    rotation), while baking SCALE is almost always right, because non-uniform object
    scale is what silently breaks normals on import.

    THE DEFAULT CHANGED 2026-09-12. Each channel used to default to
    true on its own, so {scale: true} READ as "apply the scale" and applied location
    and rotation too - it moved the origins of the dominoes showcase's 110 rigid
    bodies to the world origin, and nothing in the response contradicted the reading
    until the simulation did. `applied` names what really ran.

    NEGATIVE SCALE IS REPORTED, NOT SILENTLY FIXED. A mirrored object carries a
    negative scale component, and applying it inverts the winding order, so the mesh
    renders inside-out. Blender does not warn. This does, and recalculates the
    normals when `fixNormals` is left on -- which is the correct repair, and it is
    named in the response rather than done invisibly.

    MULTI-USER MESH DATA IS REFUSED. Applying a transform to a mesh shared by two
    objects would move BOTH, one of them silently. Blender's own operator raises;
    this refuses first with the sharing count in the message.
    """
    reject_unknown(params, ("object", "name", "location", "rotation", "scale", "fixNormals", "withMeshes"),
                   "apply_transform")
    # AN ARMATURE IS TAKEN TOO (2026-10-06): applying a rig's scale or rotation with its skinned meshes was raw Python
    # (select the armature and its meshes, transform_apply). withMeshes (default true) brings every mesh parented to it
    # or deformed by its Armature modifier; the reply measures the evaluated skin and the bone heads in world space
    # before and after (maxSkinMove, maxBoneHeadMove), so "the character did not move" is read, not assumed.
    obj_name = take(params, "object", "name", required=True)
    obj = get_object(obj_name)
    if obj.type != "ARMATURE":
        obj = get_object(obj_name, want_mesh=True)
    elif params.get("fixNormals") is not None:
        raise MifOpError("fixNormals is for a mesh; an armature's meshes keep their normals. NOTHING was changed.")
    elif obj.mode != "OBJECT":
        raise MifOpError("'%s' is in %s mode; apply_transform needs object mode. NOTHING was changed."
                         % (obj.name, obj.mode))
    if obj.type == "MESH" and params.get("withMeshes") is not None:
        raise MifOpError("withMeshes is for an armature; '%s' is a mesh. NOTHING was changed." % obj.name)

    # Asked BEFORE the takes consume the keys: naming any channel turns the unnamed ones off.
    named = any(k in params for k in ("location", "rotation", "scale"))
    do_loc = take_bool(params, "location", default=not named)
    do_rot = take_bool(params, "rotation", default=not named)
    do_scale = take_bool(params, "scale", default=not named)
    # READ HERE, with the channel flags - before transform_apply. It was read after the apply, so
    # {fixNormals: "flase"} was refused with the transform ALREADY baked into the mesh (measured on 5.0.1,
    # 2026-09-21). "Read BEFORE the scale tests" below was the half of the lesson it had learnt.
    fix_normals = take_bool(params, "fixNormals", default=True)
    if not (do_loc or do_rot or do_scale):
        raise MifOpError(
            "apply_transform was asked to apply nothing - location, rotation and scale are all "
            "false. Leave them unset to apply all three, or set at least one. NOTHING was changed.")
    if obj.type == "ARMATURE":
        return _apply_armature_transform(obj, do_loc, do_rot, do_scale, take_bool(params, "withMeshes", default=True))

    if obj.data.users > 1:
        raise MifOpError(
            "'%s' shares its mesh data with %d other object(s), and applying a transform would "
            "move every one of them - one of which you did not ask about. Make the data "
            "single-user in Blender first. NOTHING was changed." % (obj.name, obj.data.users - 1))

    before = object_info(obj)
    had_negative_scale = any(v < 0.0 for v in obj.scale)

    snap = selection_snapshot()
    try:
        select_only([obj])
        bpy.ops.object.transform_apply(location=do_loc, rotation=do_rot, scale=do_scale)
    finally:
        selection_restore(snap)

    fixed_normals = False
    # fix_normals was read with the channel flags at the top - before the scale tests (see
    # audit_blender_shortcircuit_take: it only ever ran on a negatively-scaled object, so on every other mesh
    # the flag went unchecked) and before the apply.
    if had_negative_scale and do_scale and fix_normals:
        me = obj.data
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        me.update()
        fixed_normals = True

    after = object_info(obj)
    out = {
        "object": obj.name,
        "applied": {"location": do_loc, "rotation": do_rot, "scale": do_scale},
        "before": before,
        "after": after,
        "isIdentityTransform": after["isIdentityTransform"],
        "hadNegativeScale": had_negative_scale,
        "normalsRecalculated": fixed_normals,
    }
    if had_negative_scale:
        out["negativeScaleNote"] = (
            "this object had a NEGATIVE scale component - it was mirrored. Applying that inverts "
            "the winding order, so the mesh would render inside-out in Unreal. Face normals were "
            "recalculated to repair it." if fixed_normals else
            "this object had a NEGATIVE scale component - it was mirrored. Applying that inverts "
            "the winding order, so the mesh will render inside-out in Unreal. fixNormals was off, "
            "so nothing was corrected - call clean_mesh{recalcNormals:true} if that was not "
            "intended.")
    if not after["isIdentityTransform"]:
        out["note"] = (
            "the transform is still not identity, because only some channels were applied. "
            "isIdentityTransform is what mif_mesh_roundtrip gates on, so apply all three before "
            "exporting for Unreal.")
    return out


_ORIGIN_CORNERS = ("-x-y", "+x-y", "-x+y", "+x+y")
_ORIGIN_SIDES = ("-x", "+x", "-y", "+y")


def world_vertex_bounds(obj):
    """(min, max) of an object's vertices in WORLD space, as two 3-lists, or (None, None) with no vertices.

    Read off the vertices with foreach_get, never off obj.bound_box: that box is cached and stays at the old extent
    after an edit until the view layer updates (mesh_stats measured it on every build)."""
    me = obj.data
    n = len(me.vertices)
    if not n:
        return None, None
    flat = array.array("f", [0.0]) * (3 * n)
    me.vertices.foreach_get("co", flat)
    mw = obj.matrix_world
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for i in range(n):
        p = mw @ Vector((flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]))
        for k in range(3):
            if p[k] < lo[k]:
                lo[k] = p[k]
            if p[k] > hi[k]:
                hi[k] = p[k]
    return lo, hi


def _origin_corner_side(vmin, vmax, corner=None, side=None):
    """The world point for a bottom corner ('-x-y'...) or the bottom middle of a side ('-x'...), always at minimum Z."""
    mid = [(vmin[k] + vmax[k]) * 0.5 for k in range(3)]
    if corner is not None:
        x = vmin[0] if corner[0] == "-" else vmax[0]
        y = vmin[1] if corner[2] == "-" else vmax[1]
        return (x, y, vmin[2])
    axis = 0 if side[1] == "x" else 1
    point = [mid[0], mid[1], vmin[2]]
    point[axis] = vmin[axis] if side[0] == "-" else vmax[axis]
    return tuple(point)


def pivot_in_bounds(obj):
    """Where the object's origin sits inside its world vertex bounds: per axis, 0 at the minimum and 1 at the maximum
    (below 0 or above 1 is outside). An axis with no extent answers None - a flat plane has no Z fraction."""
    vmin, vmax = world_vertex_bounds(obj)
    if vmin is None:
        return None
    origin = obj.matrix_world.translation
    out = []
    for k in range(3):
        span = vmax[k] - vmin[k]
        out.append(round((origin[k] - vmin[k]) / span, 4) if span > 1e-9 else None)
    return out


def op_set_origin(params):
    """Move an object's ORIGIN without moving its geometry in the world.

    The pivot is what Unreal rotates and places the mesh around, and Blender puts
    it wherever the object happened to be created. A prop whose origin sits in the
    middle of its bounding box cannot be placed on a floor by setting Z; a door
    whose origin is not on its hinge edge cannot be rotated open. Neither is fixable
    on the Unreal side -- the origin is baked into the FBX -- so it has to be right
    before export, and nothing here could set it.

    MODES:
      geometry (default) the median point of the mesh, Blender's ORIGIN_GEOMETRY.
      bounds             the centre of the bounding box. Different from geometry on
                         any mesh with uneven vertex density, which is most of them.
      bottom             the bounding-box centre in X/Y, its MINIMUM in Z. The one a
                         placeable prop almost always wants, because it puts the
                         pivot on the floor.
      cursor             the 3D cursor's current position.
      world              the world origin, (0,0,0).
      point              an explicit `location` in Blender units.
      bottomCorner       a bottom corner of the bounds: `corner` names which, '-x-y' (the
                         default: minimum X, Y and Z), '+x-y', '-x+y' or '+x+y'. The pivot a
                         floor module tiles by - whole-module moves from one corner.
      bottomSide         the bottom middle of one side face: `side` is '-x', '+x', '-y' or
                         '+y' (required). A wall module's pivot on its face, so it snaps flush
                         against a line on the floor.

    THE CORNER AND SIDE MODES READ THE BOUNDS OFF THE VERTICES, in world axes, not off
    obj.bound_box - that box is cached and reads stale after an edit until the view layer
    updates (mesh_stats says so). pivotInBounds reports where the origin landed as a fraction
    of each world axis of the bounds, 0 at the minimum and 1 at the maximum.

    THE GEOMETRY DOES NOT MOVE. That is the whole point, and it is asserted: the
    world-space bounds are measured before and after and reported together, so a
    caller can see that only the pivot moved.
    """
    reject_unknown(params, ("object", "name", "mode", "location", "corner", "side"), "set_origin")
    obj = get_object(take(params, "object", "name", required=True), want_mesh=True)
    # kind=str, because .lower() on whatever arrived raised AttributeError out of the op for
    # a dict - take() with kind refuses instead, which is the handler's contract.
    asked = (take(params, "mode", default=None, kind=str) or "").lower()
    # A LOCATION MEANS MODE 'point'. location is read only in mode 'point', and with the mode
    # left out this op fell back to 'geometry' and ignored the location it was handed: the
    # pivot stayed in the middle and the call answered ok. The bounce showcase's balls came to
    # rest half through the floor that way (2026-09-13). Now a location with no mode is
    # 'point', and the response says it was inferred; a location beside any other mode is
    # refused rather than dropped.
    has_location = params.get("location") is not None
    inferred = not asked and has_location
    mode = asked or ("point" if has_location else "geometry")
    # The two camel-case modes are matched without case, like the rest: 'bottomcorner' is what the
    # .lower() above makes of what a caller sends.
    mode = {"bottomcorner": "bottomCorner", "bottomside": "bottomSide"}.get(mode, mode)

    valid = ("geometry", "bounds", "bottom", "cursor", "world", "point", "bottomCorner", "bottomSide")
    if mode not in valid:
        raise MifOpError("unknown mode '%s' for set_origin. Accepted: %s. NOTHING was changed."
                         % (mode, ", ".join(valid)))
    if has_location and mode != "point":
        raise MifOpError("location is read only in mode 'point', and mode '%s' places the origin by "
                         "its own rule - the location would be ignored. Drop location, or pass mode "
                         "'point'. NOTHING was changed." % mode)
    # corner AND side, REFUSED BESIDE A MODE THAT WOULD IGNORE THEM - the lesson location taught above.
    corner = take(params, "corner", default=None, kind=str)
    side = take(params, "side", default=None, kind=str)
    if corner is not None and mode != "bottomCorner":
        raise MifOpError("corner is read only in mode 'bottomCorner', and mode '%s' would ignore it. "
                         "NOTHING was changed." % mode)
    if side is not None and mode != "bottomSide":
        raise MifOpError("side is read only in mode 'bottomSide', and mode '%s' would ignore it. "
                         "NOTHING was changed." % mode)
    if mode == "bottomCorner":
        corner = (corner or "-x-y").lower()
        if corner not in _ORIGIN_CORNERS:
            raise MifOpError("corner must be one of %s (the X side, then the Y side, of the bounds' bottom); "
                             "got %r. NOTHING was changed." % (", ".join(_ORIGIN_CORNERS), corner))
    if mode == "bottomSide":
        if side is None:
            raise MifOpError("mode 'bottomSide' needs side: one of %s - the face of the bounds whose bottom "
                             "middle becomes the pivot. NOTHING was changed." % ", ".join(_ORIGIN_SIDES))
        side = side.lower()
        if side not in _ORIGIN_SIDES:
            raise MifOpError("side must be one of %s; got %r. NOTHING was changed."
                             % (", ".join(_ORIGIN_SIDES), side))
    # THE POINT IS PARSED HERE, BEFORE ANYTHING IS TOUCHED, and every coordinate must be finite. It was read
    # inside the try below with a bare float(): [NaN, 0, 0] moved the object's origin to nan on every axis
    # under ok:true, and ["abc", 0, 0] was a raw ValueError (measured on 5.0.1, 2026-09-20, found by
    # audit_unguarded_numbers' provenance pass).
    point_target = None
    if mode == "point":
        loc = take(params, "location", required=True)
        if not isinstance(loc, (list, tuple)) or len(loc) != 3:
            raise MifOpError("mode 'point' needs location as [x, y, z] in Blender units. NOTHING was changed.")
        try:
            point_target = Vector(finite_floats(loc, "location"))
        except MifOpError as exc:
            raise MifOpError("%s NOTHING was changed." % exc)

    def world_bounds():
        pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        return (
            [min(p[i] for p in pts) for i in range(3)],
            [max(p[i] for p in pts) for i in range(3)],
        )

    wmin_before, wmax_before = world_bounds()
    before_origin = list(obj.matrix_world.translation)

    snap = selection_snapshot()
    saved_cursor = list(bpy.context.scene.cursor.location)
    try:
        select_only([obj])
        if mode == "geometry":
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
        elif mode == "bounds":
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        else:
            if mode == "bottom":
                target = Vector(((wmin_before[0] + wmax_before[0]) * 0.5,
                                 (wmin_before[1] + wmax_before[1]) * 0.5,
                                 wmin_before[2]))
            elif mode in ("bottomCorner", "bottomSide"):
                vmin, vmax = world_vertex_bounds(obj)
                if vmin is None:
                    raise MifOpError("'%s' has no vertices, so it has no bounds to put a corner or a side "
                                     "on. NOTHING was changed." % obj.name)
                target = Vector(_origin_corner_side(vmin, vmax, corner if mode == "bottomCorner" else None,
                                                    side if mode == "bottomSide" else None))
            elif mode == "world":
                target = Vector((0.0, 0.0, 0.0))
            elif mode == "point":
                target = point_target
            else:  # cursor
                target = Vector(saved_cursor)
            bpy.context.scene.cursor.location = target
            bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    finally:
        bpy.context.scene.cursor.location = saved_cursor
        selection_restore(snap)

    wmin_after, wmax_after = world_bounds()
    moved = max(abs(wmin_after[i] - wmin_before[i]) for i in range(3))
    moved = max(moved, max(abs(wmax_after[i] - wmax_before[i]) for i in range(3)))

    out = {
        "object": obj.name,
        "mode": mode,
        "modeInferred": inferred,
        "originBeforeBU": rnd(before_origin),
        "originAfterBU": rnd(list(obj.matrix_world.translation)),
        "worldBoundsMinBU": rnd(wmin_after),
        "worldBoundsMaxBU": rnd(wmax_after),
        "geometryMovedBU": round(moved, 6),
        "geometryStayedPut": moved < 1e-4,
    }
    if mode == "bottomCorner":
        out["corner"] = corner
    elif mode == "bottomSide":
        out["side"] = side
    if mode in ("bottomCorner", "bottomSide"):
        # WHERE IT LANDED, MEASURED against the vertex bounds after the move - 0 at each axis' minimum,
        # 1 at its maximum - so a caller reads the corner or face it asked for instead of trusting it.
        out["pivotInBounds"] = pivot_in_bounds(obj)
    if not out["geometryStayedPut"]:
        out["note"] = (
            "the world-space bounds moved by %.6f BU. Setting an origin should move the PIVOT and "
            "leave the geometry where it is, so this is reported rather than assumed harmless."
            % moved)
    return out


_QUADRIFY_KEYS = {"maxAngleDeg": 40.0, "shapeAngleDeg": 40.0, "respectSeams": False, "respectSharp": False,
                  "respectMaterial": False, "respectUV": False, "topologyInfluence": None}


def _quadrify_params(raw, do_tri):
    """clean_mesh's quadrify, checked before anything is written: None when not asked."""
    if raw is None or raw is False:
        return None
    if raw is True:
        raw = {}
    if not isinstance(raw, dict) or set(raw) - set(_QUADRIFY_KEYS):
        raise MifOpError("quadrify is true or {%s}; got %r. NOTHING was changed." % (", ".join(_QUADRIFY_KEYS), raw))
    if do_tri:
        raise MifOpError("quadrify joins triangles into quads and triangulate splits them again. Pass one. NOTHING was "
                         "changed.")
    out = {}
    # TOPOLOGY INFLUENCE (2026-10-06): join_triangles takes topology_influence (0-2) from 4.4 - it prefers the pairings
    # that leave a quad grid; the BMesh op has no such slot on 4.2, so it is refused there by name.
    ti = raw.get("topologyInfluence")
    if ti is not None:
        ti = take_float(raw, "topologyInfluence")
        if not 0.0 <= ti <= 2.0:
            raise MifOpError("quadrify.topologyInfluence must be 0-2, got %r. NOTHING was changed." % ti)
        if bpy.app.version < (4, 4, 0):
            raise MifOpError("quadrify.topologyInfluence needs Blender 4.4 or later (bmesh join_triangles has no "
                             "topology_influence on %s). NOTHING was changed." % bpy.app.version_string)
    out["topologyInfluence"] = ti
    for key, default in _QUADRIFY_KEYS.items():
        if key == "topologyInfluence":
            continue
        if isinstance(default, bool):
            out[key] = take_bool(raw, key, default=default)
        else:
            out[key] = take_float(raw, key, default=default)
            if not 0.0 <= out[key] <= 180.0:
                raise MifOpError("quadrify.%s must be 0-180 degrees, got %r. NOTHING was changed." % (key, out[key]))
    return out


def _relax_params(raw):
    """clean_mesh relaxCrumples -> (creaseDeg, growRings, iterations, factor) or None, checked before anything is written."""
    if raw is None:
        return None
    if raw is True:
        raw = {}
    if not isinstance(raw, dict) or set(raw) - {"creaseDeg", "growRings", "iterations", "factor"}:
        raise MifOpError("relaxCrumples is true or {creaseDeg (55), growRings (3), iterations (25), factor (0.6)}, got "
                         "%r. NOTHING was changed." % (raw,))
    crease = finite_float(raw.get("creaseDeg", 55.0), "relaxCrumples.creaseDeg")
    grow, iters = raw.get("growRings", 3), raw.get("iterations", 25)
    factor = finite_float(raw.get("factor", 0.6), "relaxCrumples.factor")
    if not 1.0 <= crease < 180.0:
        raise MifOpError("relaxCrumples.creaseDeg must be 1-179 degrees, got %r. NOTHING was changed." % crease)
    for key, val, hi in (("growRings", grow, 50), ("iterations", iters, 500)):
        if isinstance(val, bool) or not isinstance(val, int) or not (0 if key == "growRings" else 1) <= val <= hi:
            raise MifOpError("relaxCrumples.%s must be a whole number up to %d, got %r. NOTHING was changed."
                             % (key, hi, val))
    if not 0.0 < factor <= 1.0:
        raise MifOpError("relaxCrumples.factor must be above 0 and at most 1, got %r. NOTHING was changed." % factor)
    return crease, grow, iters, factor


def _crease_edges(bm, crease_deg):
    lim = math.radians(crease_deg)
    out = []
    for e in bm.edges:
        if len(e.link_faces) == 2:
            a, b = e.link_faces
            if a.normal.length > 0.5 and b.normal.length > 0.5 and a.normal.angle(b.normal, 0.0) > lim:
                out.append(e)
    return out


def _relax_crumples(bm, crease_deg, grow, iters, factor):
    """Laplacian-relax only the crumples: the vertices of every edge whose two faces turn more than crease_deg apart,
    grown `grow` edge rings; each pass moves a vertex `factor` of the way to its neighbours' mean - a boundary vertex
    only toward its neighbours along the boundary, so a rim keeps its line."""
    bm.normal_update()
    before = _crease_edges(bm, crease_deg)
    seed = {v for e in before for v in e.verts}
    region = set(seed)
    frontier = set(seed)
    for _ in range(grow):
        frontier = {e.other_vert(v) for v in frontier for e in v.link_edges} - region
        region |= frontier
    order = sorted(region, key=lambda v: v.index)
    for _ in range(iters if region else 0):
        new = {}
        for v in order:
            if v.is_boundary:
                nb = [e.other_vert(v) for e in v.link_edges if e.is_boundary]
            else:
                nb = [e.other_vert(v) for e in v.link_edges]
            if not nb:
                continue
            avg = sum((x.co for x in nb), Vector()) / len(nb)
            new[v] = v.co * (1.0 - factor) + avg * factor
        for v, co in new.items():
            v.co = co
    bm.normal_update()
    after = _crease_edges(bm, crease_deg)
    return {"creaseDeg": crease_deg, "growRings": grow, "iterations": iters, "factor": factor,
            "creaseEdgesBefore": len(before), "creaseEdgesAfter": len(after), "seedVertices": len(seed),
            "relaxedVertices": len(region), "seedVertexIndices": sorted(v.index for v in seed)[:50]}


def _open_uv_layer(obj, raw):
    """clean_mesh's openCollapsedUvs, checked before anything changes: a layer's name, or None when not asked."""
    if raw is None or raw is False:
        return None
    layers = obj.data.uv_layers
    if raw is True:
        if layers.active is None:
            raise MifOpError("'%s' has no UV layer, so openCollapsedUvs has nothing to open. NOTHING was changed." % obj.name)
        return layers.active.name
    if not isinstance(raw, str):
        raise MifOpError("openCollapsedUvs is a UV layer's name or true, got %r. NOTHING was changed." % (raw,))
    if layers.get(raw) is None:
        raise MifOpError("no UV layer named '%s' on '%s'. It has %s. NOTHING was changed."
                         % (raw, obj.name, ", ".join(layer.name for layer in layers) or "none"))
    return raw


def _winding_scan(bm):
    """(edges whose two faces walk them the same way, closed parts wound inside out as a whole), over an indexed bmesh."""
    flipped_edges = []
    for e in bm.edges:
        if len(e.link_faces) == 2:
            starts = [lp.vert for f in e.link_faces for lp in f.loops if lp.edge is e]
            if len(starts) == 2 and starts[0] is starts[1]:
                flipped_edges.append(e.index)
    flipped_set = set(flipped_edges)
    seen, inside_out = set(), []
    for f0 in bm.faces:
        if f0.index in seen:
            continue
        part, stack = [], [f0]
        seen.add(f0.index)
        while stack:
            f = stack.pop()
            part.append(f)
            for e in f.edges:
                for g in e.link_faces:
                    if g.index not in seen:
                        seen.add(g.index)
                        stack.append(g)
        edges_of = {e for f in part for e in f.edges}
        if any(len(e.link_faces) != 2 for e in edges_of) or any(e.index in flipped_set for e in edges_of):
            continue
        vol = 0.0
        for f in part:
            a = f.verts[0].co
            for k in range(1, len(f.verts) - 1):
                vol += a.dot(f.verts[k].co.cross(f.verts[k + 1].co))
        if vol < 0:
            inside_out.append({"faces": len(part), "firstFace": part[0].index, "signedVolume": round(vol / 6.0, 9)})
    return flipped_edges, inside_out


def _nested_inward_parts(mesh):
    """How many closed parts facing inward sit inside another part's bounds: a cavity, or an inner part wound wrong."""
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.edges.index_update()
        bm.faces.index_update()
        bm.faces.ensure_lookup_table()
        inward = {row["firstFace"] for row in _winding_scan(bm)[1]}
        if not inward:
            return 0
        seen, boxes = set(), []
        for f0 in bm.faces:
            if f0.index in seen:
                continue
            seen.add(f0.index)
            stack, verts = [f0], set()
            while stack:
                f = stack.pop()
                verts.update(f.verts)
                for e in f.edges:
                    for g in e.link_faces:
                        if g.index not in seen:
                            seen.add(g.index)
                            stack.append(g)
            cos = [v.co for v in verts]
            boxes.append((f0.index, [min(c[i] for c in cos) for i in range(3)], [max(c[i] for c in cos) for i in range(3)]))
        count = 0
        for first, low, high in boxes:
            if first in inward and any(other != first and all(lo[i] <= low[i] and high[i] <= hi[i] for i in range(3))
                                       for other, lo, hi in boxes):
                count += 1
        return count
    finally:
        bm.free()


def _winding_after_recalc(mesh):
    """What recalcNormals would leave, on a throwaway copy: (flipped edges, inside-out parts, faces on the flipped edges now).
    A fix is sent as given, and on 3 of 34 shipped meshes that failed this check recalculating did not clear it, and on one
    it took 4 to 31."""
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.edges.index_update()
        bm.faces.index_update()
        bm.edges.ensure_lookup_table()
        was, _inside = _winding_scan(bm)
        faces = sorted({f.index for i in was for f in bm.edges[i].link_faces})
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        bm.edges.index_update()
        bm.faces.index_update()
        flipped, inside = _winding_scan(bm)
        return len(flipped), len(inside), faces
    finally:
        bm.free()


def op_clean_mesh(params):
    """The cleanup pass an imported or edited mesh needs before it goes back to Unreal.

    FIVE INDEPENDENT STEPS, each off or on by name, run in the only order that is
    correct: merge first (so loose/degenerate detection sees the merged topology),
    then delete loose, then dissolve degenerates, then triangulate, then recalc
    normals last (because every earlier step can change what a face's normal should
    be).

      mergeDistance   weld vertices closer than this. The single most useful one on
                      an imported mesh: duplicate verts along a seam are invisible
                      in the viewport and split the mesh's smoothing in Unreal.
      removeLoose     delete verts and edges belonging to no face. They export, they
                      cost nothing visible, and they make Unreal's bounds wrong.
      dissolveDegenerate  collapse zero-area faces and zero-length edges. true works at 1e-6; a number
                      (mesh units, up to 0.01) is the distance - a face no wider than it goes too,
                      which is how mesh_quality gameReady's sliverFaces are fixed (1.4).
      triangulate     Unreal triangulates on import anyway; doing it here means the
                      triangulation you SEE is the one you ship, rather than one the
                      importer picked.
      recalcNormals   make normals consistently outward.

    OPEN COLLAPSED UVS (1.4, 2026-11-01): openCollapsedUvs, a UV layer's name or true for the active one, gives each face
    with a triangle of no UV area a ring of corners _UV_OPEN_RADIUS across, about where they were: Unreal then computes
    tangents there and nothing else in the layer moves, so a palette or atlas keeps its look. It is what mesh_quality
    gameReady's nearZeroTangents names.

    IT REPORTS WHAT EACH STEP DID, not that it ran. Counts before and after per
    step, so "cleaned" is a number rather than a claim -- a mesh that was already
    clean returns zeroes and says so, instead of a cheerful ok that reads as work
    performed.

    KEEPING THEM (1.4, 2026-11-01): keepCustomNormals, with recalcNormals, flipNormals or splitNonManifold and no step that
    changes faces, puts each corner's normal back after the step. splitNonManifold gives each face on an edge of three or
    more faces its own copy of that edge; nothing moves.

    CUSTOM SPLIT NORMALS. recalcNormals discards them, so it is refused with a named
    reason when the mesh has them and `force` is not set, rather than quietly
    throwing away data a rigger put there on purpose.

    mergeDistance IS IN THE MESH'S OWN UNITS unless space is WORLD. Measured 2026-10-04 on 5.2.1: a
    grid with 0.1-unit spacing welded 441 -> 170 verts at 0.11 under a parent scaled 0.01, 1 and 100
    alike - so under a scaled parent (an FBX rig at 0.01 or 100) the same number is a different world
    distance, and the garment work collapsed 9,400 faces to 850 that way (2026-10-03). space WORLD
    divides by the object's world scale (refused when that scale is not uniform), and every merge
    reports both distances.

    THE EXPORT FINISH (2026-10-04): minFaceArea deletes faces smaller than that area (mesh units squared) - slivers
    a boolean or a remesh leaves - with the edges and vertices only they used; removeUnusedSlots drops every material
    slot no face uses. Mesh.materials.pop() already shifts the faces' slot indices down (a raw script that shifted them
    again put a plant's leaves on its pot), so each face's material is compared by NAME before and after the drop.

    fillHolesUpTo N (2026-10-04) fills every CLOSED boundary loop of at most N vertices with a face - the holes a
    voxel-built sheet leaves - after the loose and degenerate steps and before triangulate. bmesh.ops.holes_fill copies
    the surrounding loops' data, so the new face takes its UVs from its neighbours. An open sheet's own rim is a
    boundary loop too, so N is what keeps a cuff or a hem open: the loops left are listed by size (holesLeft). SINCE
    2026-10-05 each hole is filled on its own, two holes touching at a vertex are split apart, and every hole not filled
    is named with its reason (_fill_holes) - one holes_fill over every hole added no face on a garment with five.

    FLIP (2026-10-05): flipNormals true reverses every face, a list of face indices those faces - LAST, after
    recalcNormals, so recalcNormals plus flipNormals true points a closed mesh inward (bmesh.ops.reverse_faces). Cloth
    pressure acts along the normal, so a blazer whose shoulders sagged was fixed by flipping them all; recalcNormals
    could never reverse a direction on purpose. A face list names faces as they are now, so it is refused with a step
    that renumbers them.

    QUADRIFY (2026-10-04) joins triangle pairs back into quads - bmesh.ops.join_triangles, Blender's Tris to Quads:
    quadrify {maxAngleDeg (40: the largest angle between the two triangles' normals), shapeAngleDeg (40), respectSeams,
    respectSharp, respectMaterial, respectUV} - a respected edge is never removed, so a marked sewing line stays. It runs
    after the hole fill and is refused with triangulate. Triangles and quads before and after come back.

    FINS AND RAGGED LOOPS (2026-10-04), before the hole fill: despikeDistance (WORLD units) pulls a vertex within
    nearBoundaryRings (default 4) edge rings of a boundary back to its neighbours' average when it stands farther than
    that from it (a rim vertex is averaged with its two neighbours ALONG THE RIM - by boundary edge: the recorded rule
    took every boundary VERTEX next to it, and on a short tube, whose every vertex is on a rim, that pulled all 16
    vertices of a clean tube toward each other), up to 6 passes - the fins a drape leaves on
    a hood rim (despike.py moved 90 at 6 mm). A corner of a rim also stands off its neighbours' average, so the distance
    must be above that (a square grid's corner: 0.71 of its edge length). smoothLoopNear [x,y,z] straightens the
    boundary loop nearest that world point along itself, smoothLoopIterations times (default 10): each loop vertex goes
    halfway to its two loop neighbours' midpoint, then the loop is scaled about its centroid back to its mean distance
    from it - plain halving shrinks a circle 15% a pass on 8 sides (0.5 + 0.5 cos 45), so a cuff would close; a round
    loop is left where it is and only the raggedness goes.

    CRUMPLES (2026-10-05): relaxCrumples {creaseDeg (55), growRings (3), iterations (25), factor (0.6)} - or true for
    those - finds every edge whose two faces turn more than creaseDeg apart (a drape's sub-centimetre crumples at a hood
    border or a neck junction), grows those edges' vertices growRings edge rings, and Laplacian-relaxes that region alone
    `iterations` times (each vertex factor of the way to its neighbours' mean; a boundary vertex only along its loop), so
    the folds elsewhere keep their shape - a whole-mesh smooth flattens every fold. Runs after smoothLoopNear and before
    the hole fill; steps.relaxedCrumples gives seedVertices, relaxedVertices and creaseEdgesBefore / After.
    """
    reject_unknown(params, ("object", "name", "mergeDistance", "removeLoose",
                            "dissolveDegenerate", "triangulate", "recalcNormals", "force", "space",
                            "minFaceArea", "removeUnusedSlots", "fillHolesUpTo", "despikeDistance",
                            "nearBoundaryRings", "smoothLoopNear", "smoothLoopIterations", "quadrify", "flipNormals",
                            "relaxCrumples", "openCollapsedUvs", "keepCustomNormals", "splitNonManifold"),
                   "clean_mesh")
    obj = get_object(take(params, "object", "name", required=True), want_mesh=True)
    # A LINKED MESH CANNOT BE CLEANED. Measured on 5.0.1: this ran, changed nothing, and answered
    # ok with a full before/after - the quiet half of the failure the transform had loudly.
    require_editable(obj, "clean")

    merge_distance = take_float(params, "mergeDistance", default=0.0)
    do_loose = take_bool(params, "removeLoose", default=False)
    # dissolveDegenerate true dissolves at 1e-6; a number (1.4, 2026-10-30) is the distance in mesh units, for the
    # sliver faces mesh_quality gameReady names: dissolve_degenerate takes a face no wider than the distance.
    raw_degen = params.get("dissolveDegenerate")
    if isinstance(raw_degen, (int, float)) and not isinstance(raw_degen, bool):
        degen_dist = finite_float(raw_degen, "dissolveDegenerate")
        if not 0.0 < degen_dist <= 0.01:
            raise MifOpError("dissolveDegenerate as a number is the distance in mesh units, above 0 and at most 0.01 "
                             "(a face no wider and an edge no longer than it is dissolved), got %r. NOTHING was "
                             "changed." % raw_degen)
        do_degenerate = True
    else:
        do_degenerate = take_bool(params, "dissolveDegenerate", default=False)
        degen_dist = 1e-6
    do_tri = take_bool(params, "triangulate", default=False)
    do_normals = take_bool(params, "recalcNormals", default=False)
    keep_custom = take_bool(params, "keepCustomNormals", default=False)
    do_split_nm = take_bool(params, "splitNonManifold", default=False)
    min_area = take_float(params, "minFaceArea", default=0.0)
    do_slots = take_bool(params, "removeUnusedSlots", default=False)
    fill_upto = take_int(params, "fillHolesUpTo", default=None)
    despike = take_float(params, "despikeDistance", default=None)
    rings = take_int(params, "nearBoundaryRings", default=4)
    smooth_near = _point_param(params, "smoothLoopNear")
    smooth_iters = take_int(params, "smoothLoopIterations", default=10)
    if despike is not None and not despike > 0.0:
        raise MifOpError("despikeDistance must be above 0 (world units), got %r. NOTHING was changed." % despike)
    if params.get("nearBoundaryRings") is not None and despike is None:
        raise MifOpError("nearBoundaryRings goes with despikeDistance. NOTHING was changed.")
    if not 0 <= rings <= 100:
        raise MifOpError("nearBoundaryRings must be 0-100, got %d. NOTHING was changed." % rings)
    if params.get("smoothLoopIterations") is not None and smooth_near is None:
        raise MifOpError("smoothLoopIterations goes with smoothLoopNear. NOTHING was changed.")
    if not 1 <= smooth_iters <= 200:
        raise MifOpError("smoothLoopIterations must be 1-200, got %d. NOTHING was changed." % smooth_iters)
    if smooth_near is not None:
        _users = {}
        for _p in obj.data.polygons:
            for _k in _p.edge_keys:
                _users[_k] = _users.get(_k, 0) + 1
        if not any(n == 1 for n in _users.values()):
            raise MifOpError("'%s' has no open boundary, so smoothLoopNear has no loop to smooth. NOTHING was changed."
                             % obj.name)
    if despike is not None:
        _ws = [abs(s) for s in obj.matrix_world.to_scale()]
        if max(_ws) - min(_ws) > 1e-6 * max(max(_ws), 1e-12) or max(_ws) <= 0.0:
            raise MifOpError("'%s' has a non-uniform or zero world scale, so one despikeDistance is not one mesh "
                             "distance. Apply the scale first. NOTHING was changed." % obj.name)
    if fill_upto is not None and fill_upto < 3:
        raise MifOpError("fillHolesUpTo is a hole's vertex count and must be at least 3, got %d. NOTHING was changed."
                         % fill_upto)
    if min_area < 0.0:
        raise MifOpError("minFaceArea must be 0 or more, got %r. NOTHING was changed." % min_area)
    if do_slots and any(sl.link == "OBJECT" for sl in obj.material_slots):
        raise MifOpError("'%s' has object-linked material slots, which belong to the object and not the mesh - "
                         "removeUnusedSlots only drops mesh slots. NOTHING was changed." % obj.name)

    quad = _quadrify_params(params.get("quadrify"), do_tri)
    relax = _relax_params(params.get("relaxCrumples"))
    open_uv = _open_uv_layer(obj, params.get("openCollapsedUvs"))
    flip = params.get("flipNormals")
    if flip is not None:
        if isinstance(flip, bool):
            flip = True if flip else None
        elif isinstance(flip, (list, tuple)):
            if not flip:
                raise MifOpError("flipNormals as a list is a non-empty list of face indices. NOTHING was changed.")
            n_faces = len(obj.data.polygons)
            bad = [i for i in flip if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n_faces]
            if bad:
                raise MifOpError("flipNormals %s are not face indices of '%s' (it has %d). NOTHING was changed."
                                 % (bad[:10], obj.name, n_faces))
            renumber = [k for k, on in (("mergeDistance", merge_distance > 0.0), ("removeLoose", do_loose),
                                        ("dissolveDegenerate", do_degenerate), ("minFaceArea", min_area > 0.0),
                                        ("fillHolesUpTo", fill_upto is not None), ("quadrify", quad is not None),
                                        ("triangulate", do_tri)) if on]
            if renumber:
                raise MifOpError("flipNormals as a face list names faces as they are now, and %s renumber the faces "
                                 "before the flip runs - flip in a call of its own, or pass true. NOTHING was changed."
                                 % ", ".join(renumber))
            flip = list(dict.fromkeys(flip))
        else:
            raise MifOpError("flipNormals is true or a list of face indices (true flips every face), got %r. NOTHING "
                             "was changed." % (flip,))
    if not (merge_distance > 0.0 or do_loose or do_degenerate or do_tri or do_normals or min_area > 0.0 or do_slots
            or fill_upto is not None or despike is not None or smooth_near is not None or quad is not None
            or flip is not None or relax is not None or open_uv is not None or do_split_nm):
        raise MifOpError(
            "clean_mesh was asked to do nothing - every step is off. Set at least one of "
            "mergeDistance, removeLoose, dissolveDegenerate, minFaceArea, despikeDistance, smoothLoopNear, "
            "fillHolesUpTo, quadrify, triangulate, recalcNormals, flipNormals, removeUnusedSlots, relaxCrumples, "
            "openCollapsedUvs, splitNonManifold. NOTHING was changed.")
    # Only the steps that leave every face and its corners' vertices in place: the normals are put back corner by corner.
    if keep_custom and (not (do_normals or do_split_nm or flip is not None) or merge_distance > 0.0 or do_loose or do_degenerate or do_tri
                        or min_area > 0.0 or fill_upto is not None or despike is not None or smooth_near is not None
                        or quad is not None or relax is not None):
        raise MifOpError("keepCustomNormals goes with recalcNormals, flipNormals or splitNonManifold alone: the other steps add or remove "
                         "faces, so a corner's normal has nowhere certain to go. Send them in a call of their own. NOTHING "
                         "was changed.")

    space_given = params.get("space") is not None
    space = str(take(params, "space", default="LOCAL", kind=str)).upper()
    if space not in ("LOCAL", "WORLD"):
        raise MifOpError("'space' must be LOCAL or WORLD (got %r). NOTHING was changed." % space)
    if space_given and not merge_distance > 0.0:
        raise MifOpError("'space' says which units mergeDistance is in, and no mergeDistance was given. "
                         "NOTHING was changed.")
    world_scale = [abs(s) for s in obj.matrix_world.to_scale()]
    uniform = max(world_scale) - min(world_scale) <= 1e-6 * max(max(world_scale), 1e-12)
    local_distance = merge_distance
    if merge_distance > 0.0 and space == "WORLD":
        if not uniform:
            raise MifOpError(
                "'%s' has a non-uniform world scale (%s), so one world distance is a different mesh distance "
                "along each axis and cannot be welded with one number. Pass space LOCAL with a mesh-unit "
                "distance, or apply the scale first. NOTHING was changed."
                % (obj.name, ", ".join("%.6g" % s for s in world_scale)))
        if world_scale[0] <= 0.0:
            raise MifOpError("'%s' has a zero world scale, so no world distance maps onto its mesh. NOTHING "
                             "was changed." % obj.name)
        local_distance = merge_distance / world_scale[0]

    me = obj.data
    has_custom_normals = bool(getattr(me, "has_custom_normals", False))
    # Read BEFORE the custom-normals test - see audit_blender_shortcircuit_take. `force` guards a
    # destructive path, and it was validated only on meshes that already had custom normals.
    force = take_bool(params, "force", default=False)
    if do_normals and has_custom_normals and not force and not keep_custom:
        raise MifOpError(
            "'%s' has CUSTOM SPLIT NORMALS, and recalcNormals would discard them - they are "
            "usually authored deliberately (hard-surface shading, foliage cards). Pass keepCustomNormals:true "
            "to turn the faces and keep them, force:true to recalculate anyway, or leave recalcNormals off. "
            "NOTHING was changed." % obj.name)
    # by face, then by the corner's place in it: recalc can reverse a face, and a split gives a corner a new vertex.
    kept_normals = None
    if keep_custom and has_custom_normals:
        from .ops_kit import _corner_normals    # noqa: PLC0415 - the one 3.6/4.1 corner-normal reader
        corners = _corner_normals(me)
        kept_normals = [[(tuple(corners[li]), me.loops[li].vertex_index) for li in p.loop_indices] for p in me.polygons]

    before = mesh_counts(obj)
    steps = {}

    bm = bmesh.new()
    bm.from_mesh(me)

    if merge_distance > 0.0:
        v0 = len(bm.verts)
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=local_distance)
        merged = {"vertsRemoved": v0 - len(bm.verts), "distance": merge_distance, "space": space,
                  "localDistance": local_distance, "worldScale": [round(s, 9) for s in world_scale]}
        if uniform:
            merged["worldDistance"] = local_distance * world_scale[0]
        else:
            merged["worldDistanceByAxis"] = [local_distance * s for s in world_scale]
        if space == "LOCAL" and not (uniform and abs(world_scale[0] - 1.0) <= 1e-6):
            merged["spaceNote"] = (
                "the distance was applied in the MESH's own units, and '%s' has a world scale of %s (its parents "
                "included), so it welded at %s in the world. Pass space WORLD to give the distance in world units."
                % (obj.name, ", ".join("%.6g" % s for s in world_scale),
                   ("%.6g" % merged["worldDistance"]) if uniform
                   else "/".join("%.6g" % d for d in merged["worldDistanceByAxis"])))
        steps["merged"] = merged

    if do_loose:
        loose_v = [v for v in bm.verts if not v.link_faces]
        loose_e = [e for e in bm.edges if not e.link_faces]
        n_e = len(loose_e)
        if loose_e:
            bmesh.ops.delete(bm, geom=loose_e, context="EDGES")
        loose_v = [v for v in bm.verts if not v.link_faces]
        n_v = len(loose_v)
        if loose_v:
            bmesh.ops.delete(bm, geom=loose_v, context="VERTS")
        steps["removedLoose"] = {"verts": n_v, "edges": n_e}

    if do_degenerate:
        f0, e0 = len(bm.faces), len(bm.edges)
        bmesh.ops.dissolve_degenerate(bm, dist=degen_dist, edges=bm.edges[:])
        steps["dissolvedDegenerate"] = {"facesRemoved": f0 - len(bm.faces),
                                        "edgesRemoved": e0 - len(bm.edges), "distance": degen_dist}

    if min_area > 0.0:
        small = [f for f in bm.faces if f.calc_area() < min_area]
        if small:
            bmesh.ops.delete(bm, geom=small, context="FACES")
        steps["removedSmallFaces"] = {"faces": len(small), "minFaceArea": min_area}

    if despike is not None:
        thr = despike / abs(obj.matrix_world.to_scale()[0])
        bm.verts.ensure_lookup_table()
        zone = {v for v in bm.verts if v.is_boundary}
        for _ in range(rings):
            zone |= {e.other_vert(v) for v in list(zone) for e in v.link_edges}
        order = sorted(zone, key=lambda v: v.index)
        moved, per_pass = set(), []
        for _ in range(6):
            n = 0
            for v in order:
                rim = [e.other_vert(v) for e in v.link_edges if len(e.link_faces) == 1]
                nb = rim or [e.other_vert(v) for e in v.link_edges]
                if not nb:
                    continue
                avg = sum((x.co for x in nb), Vector()) / len(nb)
                if (v.co - avg).length > thr:
                    v.co = avg
                    moved.add(v.index)
                    n += 1
            per_pass.append(n)
            if not n:
                break
        steps["despiked"] = {"verticesMoved": len(moved), "perPass": per_pass, "zoneVertices": len(zone),
                             "despikeDistance": despike, "nearBoundaryRings": rings}

    if smooth_near is not None:
        sloops = _boundary_loops(bm)
    if smooth_near is not None and not sloops:
        steps["smoothedLoop"] = {"loops": 0, "note": "the steps before it left no open boundary to smooth"}
    elif smooth_near is not None:
        sloop, sdist = _loop_nearest(sloops, smooth_near, obj.matrix_world)
        lverts = sloop[1]
        lset = set(lverts)

        def _mean_r(pts):
            c = sum(pts, Vector()) / len(pts)
            return c, sum((p - c).length for p in pts) / len(pts)
        for _ in range(smooth_iters):
            _c0, r0 = _mean_r([v.co for v in lverts])
            new = {}
            for v in lverts:
                nb = [e.other_vert(v) for e in v.link_edges if e.is_boundary and e.other_vert(v) in lset]
                if len(nb) == 2:
                    new[v] = v.co * 0.5 + (nb[0].co + nb[1].co) * 0.25
            for v, p in new.items():
                v.co = p
            c1, r1 = _mean_r([v.co for v in lverts])
            if r1 > 1e-12:
                for v in lverts:
                    v.co = c1 + (v.co - c1) * (r0 / r1)
        steps["smoothedLoop"] = {"loopVertices": len(lverts), "iterations": smooth_iters,
                                 "distance": round(sdist, 6), "loops": len(sloops)}

    if relax is not None:
        steps["relaxedCrumples"] = _relax_crumples(bm, *relax)

    if fill_upto is not None:
        steps["filledHoles"] = _fill_holes(bm, fill_upto)

    if quad is not None:
        def _tq():
            return (sum(1 for f in bm.faces if len(f.verts) == 3), sum(1 for f in bm.faces if len(f.verts) == 4))
        t0, q0 = _tq()
        extra = {}
        if quad["topologyInfluence"] is not None:
            bm.normal_update()
            extra["topology_influence"] = quad["topologyInfluence"]
        bmesh.ops.join_triangles(bm, faces=[f for f in bm.faces if len(f.verts) == 3],
                                 angle_face_threshold=math.radians(quad["maxAngleDeg"]),
                                 angle_shape_threshold=math.radians(quad["shapeAngleDeg"]),
                                 cmp_seam=quad["respectSeams"], cmp_sharp=quad["respectSharp"],
                                 cmp_materials=quad["respectMaterial"], cmp_uvs=quad["respectUV"],
                                 cmp_vcols=False, **extra)
        t1, q1 = _tq()
        steps["quadrified"] = dict(quad, trianglesBefore=t0, quadsBefore=q0, trianglesAfter=t1, quadsAfter=q1,
                                   trianglesLeft=t1)

    if do_tri:
        n_before = len([f for f in bm.faces if len(f.verts) > 3])
        if n_before:
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
        steps["triangulated"] = {"nonTriFacesConverted": n_before}

    if do_normals:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        steps["recalcNormals"] = {"faces": len(bm.faces),
                                  "discardedCustomSplitNormals": has_custom_normals and not keep_custom}

    if flip is not None:
        bm.faces.ensure_lookup_table()
        to_flip = bm.faces[:] if flip is True else [bm.faces[i] for i in flip]
        bmesh.ops.reverse_faces(bm, faces=to_flip, flip_multires=True)
        steps["flippedNormals"] = {"faces": len(to_flip), "all": flip is True,
                                   "indices": None if flip is True else flip[:50]}

    if do_split_nm:
        # The edge's faces stop sharing it and nothing moves. A weld cannot do this: those faces already share one vertex
        # pair, so there is nothing to merge.
        fins = [e for e in bm.edges if len(e.link_faces) > 2]
        if fins:
            bmesh.ops.split_edges(bm, edges=fins)
        steps["splitNonManifold"] = {"edges": len(fins)}

    if open_uv is not None:
        steps["openedCollapsedUvs"] = {"layer": open_uv, "faces": _open_collapsed_uvs(bm, open_uv),
                                       "radius": _UV_OPEN_RADIUS}

    bm.to_mesh(me)
    bm.free()
    me.update()
    if kept_normals is not None:
        # A reversed face keeps its corners in the opposite order, so a reversal is told by the face's first two vertices
        # against what they were; a split keeps the order and gives new vertex numbers, so it is matched by place.
        rows = []
        for p, was in zip(me.polygons, kept_normals):
            now = [me.loops[li].vertex_index for li in p.loop_indices]
            by_vertex = {v: n for n, v in was}
            if all(v in by_vertex for v in now):
                rows.extend(by_vertex[v] for v in now)
            else:
                rows.extend(n for n, _v in was)
        if hasattr(me, "use_auto_smooth"):
            me.use_auto_smooth = True
        me.normals_split_custom_set(rows)
        me.update()
        steps["keptCustomNormals"] = {"corners": len(rows)}
    if do_slots:
        names_before = [me.materials[p.material_index].name if p.material_index < len(me.materials)
                        and me.materials[p.material_index] else None for p in me.polygons]
        used = {p.material_index for p in me.polygons}
        dropped = []
        for i in range(len(me.materials) - 1, -1, -1):
            if i not in used:
                mat = me.materials[i]
                dropped.append(mat.name if mat else None)
                me.materials.pop(index=i)
        names_after = [me.materials[p.material_index].name if p.material_index < len(me.materials)
                       and me.materials[p.material_index] else None for p in me.polygons]
        if names_after != names_before:
            raise MifOpError("removing the unused slots changed which material %d face(s) use - the mesh is now "
                             "wrong; undo it." % sum(1 for a, b in zip(names_before, names_after) if a != b))
        steps["removedUnusedSlots"] = {"slots": list(reversed(dropped)),
                                       "slotsAfter": [m.name if m else None for m in me.materials]}
    # The cache refresh local_bounds relies on every editing op making - measured stale without it
    # (test_blender_dimensions_fresh D102: a collapsing merge left dimensionsBU at the old size).
    bpy.context.view_layer.update()

    after = mesh_counts(obj)
    removed_total = (before["verts"] - after["verts"]) + (before["faces"] - after["faces"])
    flipped = bool((steps.get("flippedNormals") or {}).get("faces"))
    flipped = flipped or bool((steps.get("relaxedCrumples") or {}).get("relaxedVertices"))
    flipped = flipped or bool((steps.get("openedCollapsedUvs") or {}).get("faces"))
    out = {
        "object": obj.name,
        "before": before,
        "after": after,
        "steps": steps,
        # A dropped slot changes no count, so it is counted here or a slot-only clean reads as "already clean".
        "changedAnything": before != after or bool((steps.get("removedUnusedSlots") or {}).get("slots")) or flipped,
    }
    unfilled = [h for h in ((steps.get("filledHoles") or {}).get("skipped") or []) if h.get("fillable")]
    if unfilled:
        # AN OK THAT FOUND HOLES AND DID NOT FILL THEM IS NOT A CLEAN MESH (2026-10-05): a garment answered "already
        # clean" with five holes listed and none filled.
        out["note"] = ("%d hole(s) at or under fillHolesUpTo %d were found and not filled - %s. See steps.filledHoles."
                       "skipped for their vertices."
                       % (len(unfilled), fill_upto,
                          "; ".join("%d vertices: %s" % (h["vertexCount"], h["reason"]) for h in unfilled[:5])))
    elif not out["changedAnything"] and (steps.get("filledHoles") or {}).get("skipped"):
        out["note"] = ("nothing changed: every open boundary loop is larger than fillHolesUpTo %d (%s vertices) - an "
                       "open sheet's own rim is one. A larger fillHolesUpTo fills them."
                       % (fill_upto, ", ".join(str(n) for n in steps["filledHoles"]["holesLeft"][:10])))
    elif not out["changedAnything"]:
        out["note"] = (
            "every requested step ran and the mesh was already clean by those measures - the "
            "counts are identical. Said in words rather than returned as an ok that reads as work "
            "performed.")
    if out["changedAnything"]:
        out["netElementsRemoved"] = removed_total
    # SHARED MESH DATA. See ops_common.shared_data_note - this edit lands on every object
    # that shares the datablock, which is what a linked duplicate is for and which nothing
    # in this response said.
    out.update(shared_data_note(obj))
    return out


_DELETE_ELEMENT_KINDS = {"VERTS": "VERTS", "EDGES": "EDGES", "FACES": "FACES", "ONLY_FACES": "FACES_ONLY"}


def _parse_occluded(raw):
    """occludedBy -> (meshes, reach, margin)."""
    if not isinstance(raw, dict) or not raw.get("meshes") or set(raw) - {"meshes", "reach", "margin"}:
        raise MifOpError("occludedBy is {meshes: [names], reach (m, default 0.1), margin (face rings kept, default 1)}, "
                         "got %r. NOTHING was deleted." % (raw,))
    names = raw["meshes"]
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, (list, tuple)) or any(not isinstance(n, str) for n in names):
        raise MifOpError("occludedBy.meshes is a list of object names, got %r. NOTHING was deleted." % (names,))
    meshes = [get_object(n, want_mesh=True) for n in names]
    reach = finite_floats([raw.get("reach", 0.1)], "occludedBy.reach")[0]
    margin = raw.get("margin", 1)
    if not reach > 0:
        raise MifOpError("occludedBy.reach must be above 0, got %r. NOTHING was deleted." % reach)
    if isinstance(margin, bool) or not isinstance(margin, int) or not 0 <= margin <= 50:
        raise MifOpError("occludedBy.margin is 0-50 face rings, got %r. NOTHING was deleted." % (margin,))
    return meshes, reach, margin


def _occluded_faces(obj, bm, meshes, reach, margin):
    """(faces to delete, report) - a face is covered when rays along its normal, from its centre and from each of its
    vertices, all meet one of `meshes` within reach; `margin` rings of covered faces next to an uncovered one are kept."""
    from mathutils.bvhtree import BVHTree
    trees = []
    for m in meshes:
        wb = bmesh.new()
        ev = m.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = ev.to_mesh()
        try:
            wb.from_mesh(me)
        finally:
            ev.to_mesh_clear()
        wb.transform(m.matrix_world)
        trees.append(BVHTree.FromBMesh(wb))
        wb.free()
    mw = obj.matrix_world
    m3 = mw.to_3x3().inverted().transposed()
    covered = set()
    for f in bm.faces:
        n = (m3 @ f.normal).normalized()
        if n.length < 0.5:
            continue
        pts = [mw @ f.calc_center_median()] + [mw @ v.co for v in f.verts]
        if all(any(t.ray_cast(p + n * 1e-5, n, reach)[0] is not None for t in trees) for p in pts):
            covered.add(f.index)
    kept_ring = set()
    edge = {f.index for f in bm.faces if f.index in covered
            and any(nb.index not in covered for e in f.edges for nb in e.link_faces)}
    frontier = set(edge)
    for _ in range(margin):
        kept_ring |= frontier
        nxt = set()
        for fi in frontier:
            for e in bm.faces[fi].edges:
                for nb in e.link_faces:
                    if nb.index in covered and nb.index not in kept_ring:
                        nxt.add(nb.index)
        frontier = nxt
    gone = sorted(covered - kept_ring)
    return [bm.faces[i] for i in gone], {"facesCovered": len(covered), "facesKeptByMargin": len(covered & kept_ring),
                                         "occluders": [m.name for m in meshes], "reach": reach, "margin": margin}


def _parse_disagree(raw):
    if not isinstance(raw, dict) or "reference" not in raw or set(raw) - {"reference", "maxDistance", "minDot"}:
        raise MifOpError("disagreeWith is {reference: name, maxDistance (m, default 0.01), minDot (default 0.3)}, got "
                         "%r. NOTHING was deleted." % (raw,))
    ref = get_object(str(raw["reference"]), want_mesh=True)
    max_d = finite_floats([raw.get("maxDistance", 0.01)], "disagreeWith.maxDistance")[0]
    min_dot = finite_floats([raw.get("minDot", 0.3)], "disagreeWith.minDot")[0]
    if not max_d > 0 or not -1.0 <= min_dot <= 1.0:
        raise MifOpError("disagreeWith.maxDistance must be above 0 and minDot from -1 to 1, got %r and %r. NOTHING was "
                         "deleted." % (max_d, min_dot))
    return ref, max_d, min_dot


def _disagreeing_faces(obj, bm, ref, max_d, min_dot):
    """(faces to delete, report): a face whose centre is farther than max_d from the reference surface, or whose normal's
    dot with the reference's normal at the nearest point is under min_dot."""
    from mathutils.bvhtree import BVHTree
    wb = bmesh.new()
    ev = ref.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    try:
        wb.from_mesh(me)
    finally:
        ev.to_mesh_clear()
    wb.transform(ref.matrix_world)
    wb.normal_update()
    tree = BVHTree.FromBMesh(wb)
    wb.free()
    mw = obj.matrix_world
    m3 = mw.to_3x3().inverted().transposed()
    far = turned = 0
    out = []
    for f in bm.faces:
        c = mw @ f.calc_center_median()
        loc, rn, _i, d = tree.find_nearest(c, max_d)
        if loc is None:
            far += 1
            out.append(f)
            continue
        if (m3 @ f.normal).normalized().dot(rn) < min_dot:
            turned += 1
            out.append(f)
    return out, {"reference": ref.name, "maxDistance": max_d, "minDot": min_dot, "facesTooFar": far,
                 "facesTurnedAway": turned, "facesKept": len(bm.faces) - len(out)}


def op_delete_mesh_elements(params):
    """Delete vertices, edges or faces of a mesh: by index, by vertex group weight, or every loose island under N faces.

    THE GAP, 2026-10-04. No op deleted part of a mesh. Garment work did it in raw Python again and again - the inner
    shell a solidify left, the body faces hidden under clothing, the stray one-face islands a voxel remesh leaves
    behind (keep_largest, hood_inner_delete, outfit_shell_delete) - and select_faces could FIND the faces it could not
    remove. So the indices come from select_faces (or vertex_info), and this removes them.

    ONE SELECTOR PER CALL, refused otherwise: indices (of the kind named by `elements`), vertexGroup with minWeight
    (vertices at or above it; their faces go with them unless elements is ONLY_FACES), or looseIslandsUnder (every
    face-connected island with fewer faces than N - the cleanup after a remesh).

    OCCLUDED (2026-10-05): occludedBy {meshes, reach (default 0.1 m), margin (default 1)} deletes the faces hidden
    under other meshes - a body under its outfit, the robust fix for skin poking through. A face is covered when a ray
    along its normal from its centre and from each of its vertices meets one of the meshes within reach (the object as
    its base mesh under its transform, the meshes as they evaluate: compare them in the same pose); `margin` rings of
    covered faces beside an uncovered one are kept, so the cut sits under the cloth, not at its edge.

    DISAGREEING (2026-10-05): disagreeWith {reference, maxDistance (default 0.01 m), minDot (default 0.3)} deletes every
    face whose centre is farther than maxDistance from the reference surface, or whose normal's dot with the
    reference's normal at the nearest point is under minDot - so the outer sheet of a voxel-remeshed, thickened garment
    is kept by agreement with the garment it came from, and the inner wall and the rim go.

    elements: VERTS (default for vertexGroup), EDGES, FACES (default for indices and islands; a face's now-unused
    edges and vertices go too) or ONLY_FACES (the faces alone; their edges and vertices stay, as Blender's own
    Delete > Only Faces). Counts before and after are reported, and a selector that matches nothing is refused rather
    than answered with an ok that changed nothing.
    """
    reject_unknown(params, ("object", "name", "elements", "indices", "vertexGroup", "minWeight", "looseIslandsUnder",
                            "occludedBy", "disagreeWith"), "delete_mesh_elements")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    require_editable(obj, "delete elements of")   # get_object(want_mesh=True) has refused a mesh in edit mode already
    indices = params.get("indices")
    group = take(params, "vertexGroup", default=None, kind=str)
    islands_under = take_int(params, "looseIslandsUnder", default=None)
    occ_raw, dis_raw = params.get("occludedBy"), params.get("disagreeWith")
    chosen = [n for n, v in (("indices", indices), ("vertexGroup", group), ("looseIslandsUnder", islands_under),
                             ("occludedBy", occ_raw), ("disagreeWith", dis_raw)) if v is not None]
    if len(chosen) != 1:
        raise MifOpError("pass exactly one of indices, vertexGroup, looseIslandsUnder, occludedBy or disagreeWith (got "
                         "%s). Find indices with select_faces or vertex_info. NOTHING was deleted."
                         % (", ".join(chosen) or "none"))
    occ = _parse_occluded(occ_raw) if occ_raw is not None else None
    dis = _parse_disagree(dis_raw) if dis_raw is not None else None
    if occ is not None and obj in occ[0] or dis is not None and dis[0] is obj:
        raise MifOpError("the object is among the meshes it is measured against. NOTHING was deleted.")
    if params.get("minWeight") is not None and group is None:
        raise MifOpError("minWeight goes with vertexGroup. NOTHING was deleted.")
    default_kind = "VERTS" if group is not None else "FACES"
    kind = str(take(params, "elements", default=default_kind, kind=str)).upper()
    if kind not in _DELETE_ELEMENT_KINDS:
        raise MifOpError("elements must be one of %s (got %r). NOTHING was deleted."
                         % (", ".join(sorted(_DELETE_ELEMENT_KINDS)), kind))
    if (occ is not None or dis is not None) and kind not in ("FACES", "ONLY_FACES"):
        raise MifOpError("%s deletes FACES; elements must be FACES or ONLY_FACES. NOTHING was deleted." % chosen[0])
    if islands_under is not None:
        if islands_under < 2:
            raise MifOpError("looseIslandsUnder must be at least 2 - an island has at least one face. NOTHING was "
                             "deleted.")
        if kind not in ("FACES", "ONLY_FACES"):
            raise MifOpError("looseIslandsUnder removes whole islands of FACES; elements must be FACES or ONLY_FACES. "
                             "NOTHING was deleted.")
    min_w = take_float(params, "minWeight", default=0.5)
    if group is not None and group not in obj.vertex_groups:
        raise MifOpError("'%s' has no vertex group named '%s' (it has %d). NOTHING was deleted."
                         % (obj.name, group, len(obj.vertex_groups)))

    me = obj.data
    before = mesh_counts(obj)
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        island_sizes = None
        report = None
        if occ is not None:
            bm.normal_update()
            geom, report = _occluded_faces(obj, bm, *occ)
        elif dis is not None:
            bm.normal_update()
            geom, report = _disagreeing_faces(obj, bm, *dis)
        elif indices is not None:
            if not isinstance(indices, (list, tuple)) or not indices:
                raise MifOpError("indices must be a non-empty list of whole numbers. NOTHING was deleted.")
            pool = {"VERTS": bm.verts, "EDGES": bm.edges}.get(kind, bm.faces)
            bad = [i for i in indices if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(pool)]
            if bad:
                raise MifOpError("indices %s are not %s indices of '%s' (it has %d). NOTHING was deleted."
                                 % (bad[:10], kind.lower().replace("only_", ""), obj.name, len(pool)))
            geom = [pool[i] for i in sorted(set(indices))]
        elif group is not None:
            gi = obj.vertex_groups[group].index
            hit = {v.index for v in me.vertices for g in v.groups if g.group == gi and g.weight >= min_w}
            if kind == "VERTS":
                geom = [bm.verts[i] for i in sorted(hit)]
            elif kind == "EDGES":
                geom = [e for e in bm.edges if all(v.index in hit for v in e.verts)]
            else:
                geom = [f for f in bm.faces if all(v.index in hit for v in f.verts)]
        else:
            seen, islands = set(), []
            for f in bm.faces:
                if f.index in seen:
                    continue
                stack, comp = [f], []
                seen.add(f.index)
                while stack:
                    cur = stack.pop()
                    comp.append(cur)
                    for e in cur.edges:
                        for nb in e.link_faces:
                            if nb.index not in seen:
                                seen.add(nb.index)
                                stack.append(nb)
                islands.append(comp)
            island_sizes = sorted((len(c) for c in islands), reverse=True)
            geom = [f for c in islands if len(c) < islands_under for f in c]
        if not geom:
            raise MifOpError("the selector matched no %s on '%s', so there is nothing to delete%s. NOTHING was deleted."
                             % (kind.lower().replace("only_", ""), obj.name,
                                (" (islands by face count: %s)" % island_sizes[:20]) if island_sizes else ""))
        matched = len(geom)
        bmesh.ops.delete(bm, geom=geom, context=_DELETE_ELEMENT_KINDS[kind])
        bm.to_mesh(me)
    finally:
        bm.free()
    me.update()
    bpy.context.view_layer.update()
    after = mesh_counts(obj)
    out = {"object": obj.name, "elements": kind, "selector": chosen[0], "matched": matched,
           "before": before, "after": after}
    if report is not None:
        out[chosen[0]] = report
    if island_sizes is not None:
        out["islandsBefore"] = len(island_sizes)
        out["islandFaceCounts"] = island_sizes[:50]
        out["islandsRemoved"] = sum(1 for n in island_sizes if n < islands_under)
    out.update(shared_data_note(obj))
    return out


# uv_info's overlap search stops at these and says so. An atlas of stacked foliage cards has tens of thousands of islands
# whose boxes all overlap: holding every pair passed 8 GB, and filtering each island's triangles per pair ran for minutes.
_UV_BOX_PAIR_LIMIT = 200000
_UV_BOX_TEST_LIMIT = 20000000
_UV_TRI_FILTER_LIMIT = 20000000


def _uv_box_pairs(boxes):
    """Island pairs (i, j), i < j, whose boxes share area, in that order, and whether the search stopped at a limit. A
    sweep along U, so islands apart in U are never compared."""
    active, pairs, tests = [], [], 0
    for k in sorted(range(len(boxes)), key=lambda n: boxes[n][0][0]):
        kmin, kmax = boxes[k]
        alive = []
        for a in active:
            amin, amax = boxes[a]
            if amax[0] <= kmin[0]:
                continue
            alive.append(a)
            tests += 1
            if amin[0] < kmax[0] and amin[1] < kmax[1] and kmin[1] < amax[1]:
                pairs.append((a, k) if a < k else (k, a))
        if len(pairs) >= _UV_BOX_PAIR_LIMIT or tests >= _UV_BOX_TEST_LIMIT:
            return sorted(pairs), True
        alive.append(k)
        active = alive
    return sorted(pairs), False


def _uv_tri_shrunk(t, k=1e-4):
    """The triangle pulled 0.01% toward its centroid, so two that only share an edge or a corner no longer meet."""
    c = (t[0] + t[1] + t[2]) / 3.0
    return tuple(c + (p - c) * (1.0 - k) for p in t)


def _uv_islands(bm, uv_layer):
    """Faces grouped into UV islands - ([[BMFace]], {face index: island number}).

    By shared UV coordinate, walked over face adjacency: a neighbour across an edge joins when the two faces share a UV
    corner. Quantised to 1e-5 so two loops that are the same corner but differ in the last float bit are one point.
    uv_info counts islands this way and uv_pack equalises them this way, so the two always agree."""
    face_island = {}
    islands = []
    for face in bm.faces:
        if face.index in face_island:
            continue
        stack = [face]
        island = []
        face_island[face.index] = len(islands)
        while stack:
            f = stack.pop()
            island.append(f)
            keys = {(round(lp[uv_layer].uv.x, 5), round(lp[uv_layer].uv.y, 5)) for lp in f.loops}
            for edge in f.edges:
                for nf in edge.link_faces:
                    if nf.index in face_island:
                        continue
                    nkeys = {(round(lp[uv_layer].uv.x, 5), round(lp[uv_layer].uv.y, 5)) for lp in nf.loops}
                    if keys & nkeys:
                        face_island[nf.index] = len(islands)
                        stack.append(nf)
        islands.append(island)
    return islands, face_island


def _face_uv_area(face, uv_layer):
    """A face's UV area, unsigned (shoelace): a folded face counts its whole area. Blender's average_islands_scale
    uses a signed area, which a fold reduces - it left the left sleeve at 3x the density of the rest (uv_equalize.py)."""
    uvs = [lp[uv_layer].uv for lp in face.loops]
    acc = 0.0
    for i in range(len(uvs)):
        a, b = uvs[i], uvs[(i + 1) % len(uvs)]
        acc += a.x * b.y - b.x * a.y
    return abs(acc) * 0.5


def _face_area_through(face, m3):
    """A face's area after the 3x3 part of the world matrix (Newell's vector area, right for any planar polygon). The
    translation is left out on purpose: area does not depend on it, and float32 cross products of coordinates 200 m
    from the origin lose the area in cancellation."""
    pts = [m3 @ v.co for v in face.verts]
    acc = Vector((0.0, 0.0, 0.0))
    for i in range(len(pts)):
        acc += pts[i].cross(pts[(i + 1) % len(pts)])
    return acc.length * 0.5


def _island_rows(islands, uv_layer, m3):
    """One row per island: faces, UV area, area in the world, and their ratio (UV area per square world unit)."""
    rows = []
    for i, isl in enumerate(islands):
        uva = sum(_face_uv_area(f, uv_layer) for f in isl)
        a3 = sum(_face_area_through(f, m3) for f in isl)
        us = [lp[uv_layer].uv.x for f in isl for lp in f.loops]
        vs = [lp[uv_layer].uv.y for f in isl for lp in f.loops]
        rows.append({"index": i, "faces": len(isl), "faceIndices": sorted(f.index for f in isl)[:50],
                     "uvArea": round(uva, 6), "area3d": round(a3, 6),
                     "ratio": round(uva / a3, 6) if a3 > 1e-12 and uva > 1e-12 else None,
                     "uvMin": [round(min(us), 6), round(min(vs), 6)], "uvMax": [round(max(us), 6), round(max(vs), 6)],
                     "_uva": uva, "_a3": a3})
    return rows


def _ratio_stats(rows):
    """(median ratio, max/min spread) over the islands with area in UV and in 3D; (None, None) when there are none."""
    ratios = sorted(r["_uva"] / r["_a3"] for r in rows if r["_a3"] > 1e-12 and r["_uva"] > 1e-12)
    if not ratios:
        return None, None
    n = len(ratios)
    med = ratios[n // 2] if n % 2 else 0.5 * (ratios[n // 2 - 1] + ratios[n // 2])
    return med, ratios[-1] / ratios[0]


def uv_stretch_mode(raw):
    """uv_info's stretch: ANGLE | AREA, or {mode}. None when not asked."""
    if raw is None or raw is False:
        return None
    mode = raw.get("mode") if isinstance(raw, dict) and set(raw) == {"mode"} else raw
    if not isinstance(mode, str) or mode.upper() not in ("ANGLE", "AREA"):
        raise MifOpError("stretch is ANGLE or AREA (or {mode: ...}), got %r." % (raw,))
    return mode.upper()


def _uv_corner_angle(prev_uv, uv, next_uv):
    ax, ay = prev_uv.x - uv.x, prev_uv.y - uv.y
    bx, by = next_uv.x - uv.x, next_uv.y - uv.y
    la, lb = math.hypot(ax, ay), math.hypot(bx, by)
    if la < 1e-12 or lb < 1e-12:
        return 0.0
    return math.acos(max(-1.0, min(1.0, (ax * bx + ay * by) / (la * lb))))


def uv_stretch(islands, uv_layer, m3, mode, limit=64):
    """Per-island distortion of a UV layout against the mesh, the number the UV editor's Display Stretch paints.

    AREA: each face's share of its island's UV area over its share of the island's world area, r; the face's stretch is
    1 - min(r, 1/r) - 0 when the face is the size it should be, towards 1 as it is squashed or blown up.
    ANGLE: each face's mean |UV corner angle - world corner angle|, in degrees - 0 when the layout keeps every angle.
    Per island the area-weighted mean and the worst face; for the layer the same over every face."""
    rows = []
    all_w = all_s = 0.0
    worst_all = 0.0
    for k, isl in enumerate(islands):
        uvas = [_face_uv_area(f, uv_layer) for f in isl]
        a3s = [_face_area_through(f, m3) for f in isl]
        tu, t3 = sum(uvas), sum(a3s)
        acc_w = acc_s = worst = 0.0
        worst_face = None
        for f, uva, a3 in zip(isl, uvas, a3s):
            if mode == "AREA":
                if tu <= 1e-15 or t3 <= 1e-15 or a3 <= 1e-15:
                    continue
                r = (uva / tu) / (a3 / t3)
                st = 1.0 - min(r, 1.0 / r) if r > 1e-12 else 1.0
            else:
                ls = list(f.loops)
                n = len(ls)
                diffs = [abs(_uv_corner_angle(ls[j - 1][uv_layer].uv, ls[j][uv_layer].uv, ls[(j + 1) % n][uv_layer].uv)
                             - ls[j].calc_angle()) for j in range(n)]
                st = math.degrees(sum(diffs) / n)
            acc_w += a3
            acc_s += st * a3
            if st >= worst:
                worst, worst_face = st, f.index
        if acc_w > 0:
            all_w += acc_w
            all_s += acc_s
            worst_all = max(worst_all, worst)
            rows.append({"island": k, "mean": round(acc_s / acc_w, 6), "worst": round(worst, 6),
                         "worstFace": worst_face})
    rows.sort(key=lambda r: -r["mean"])
    return {"mode": mode, "unit": "degrees" if mode == "ANGLE" else "1 - min(r, 1/r)",
            "mean": round(all_s / all_w, 6) if all_w else None, "worst": round(worst_all, 6) if all_w else None,
            "islands": rows[:limit], "islandsReported": min(len(rows), limit)}


def op_uv_info(params):
    """Read a mesh's UVs per layer - the verification half of the unwrap this addon can already do.

    uv_unwrap can create a layer; object_info reports only that layers EXIST. So
    "did the unwrap actually produce something Unreal can bake a lightmap into"
    had no answer short of opening Blender and looking. This answers it.

    WHAT UNREAL ACTUALLY REQUIRES, which is why these particular numbers:
      inside 0-1     a lightmap UV must lie within the unit square. Unreal's
                     lightmass packs charts into it; anything outside is clamped
                     and bakes as garbage. facesOutside01 is therefore the first
                     thing to check on a lightmap channel and means nothing on a
                     texture channel, where tiling outside 0-1 is the point.
      non-overlapping  two islands sharing UV space share lightmap texels, so one
                     surface's light bleeds onto another. Overlap is legal and
                     normal on a texture channel and fatal on a lightmap one.
      the channel INDEX is what Unreal's Lightmap Coordinate Index points at, and
                     it is positional - the order layers appear here is the order
                     FBX writes them.

    So `lightmapReady` is reported PER LAYER with a named reason, and it is a
    judgement about that layer used AS a lightmap - a texture layer failing it is
    not a defect and the reason says which test it failed.

    OVERLAP IS MEASURED IN BMESH, NOT VIA bpy.ops.uv.select_overlap, which needs a
    UV editor area and cannot run under `blender -b` at all - the same constraint
    that pushed bevel_edges onto bmesh.ops. Islands are found by walking shared UV
    coordinates. Pairs whose BOUNDING BOXES overlap are only CANDIDATES, and each is
    then checked TRIANGLE BY TRIANGLE with mathutils.geometry.intersect_tri_tri_2d -
    which is plain mathutils and runs headless - as sharing UV area, only touching (a
    shared edge or corner: no area, but light bleeds across it), or apart.

    THIS USED TO STOP AT THE BOXES, and called the addon's own LIGHTMAP unwrap broken.
    Measured 2026-09-13 on the gameready toadstool: 48 island pairs overlapping by box
    at every margin tried, and per triangle 0 of them overlapping or touching - ring
    and fan islands have boxes that enclose their neighbours. A false "your lightmap
    is broken" sends someone re-unwrapping a fine mesh, which this docstring always
    said was the worse outcome. The box count is still reported beside the answer, and
    a work budget keeps a huge pair from stalling the call - a pair past it is counted
    by its box, and the response says how many were. overlapCheckCapped says the search for
    pairs itself stopped at its limit (an atlas of stacked cards): the counts are then lower
    bounds and lightmapReady is false, since not every island was cleared.

    NEAR-ZERO UV AREA AND UNREAL'S ZERO TANGENTS (a short film, 2026-10-02: SM_Pothole_B passed this op and arrived
    in Unreal with "nearly zero tangents"). Each row carries nearZeroUvAreaFaces against the stated
    NEAR_ZERO_UV_AREA and nearZeroTangents from uv_degeneracy - MikkTSpace on a triangulated scratch copy plus
    mikktspace's zero-UV-area rule, since Blender's calc_tangents never answers zero. unrealTangentWarning reads UV
    channel 0, the one Unreal builds tangents from.

    STRETCH (2026-10-06): stretch AREA | ANGLE adds the UV editor's Display Stretch as numbers per island (uv_stretch),
    the measure uv_move_vertices and uv_relax report before and after.

    A PURE READ. Nothing here writes the mesh, so it cannot damage a mesh it is diagnosing; the scratch copy the
    tangents are computed on is removed before the reply.
    """
    reject_unknown(params, ("object", "name", "layer", "maxReportedIslands", "islands", "stretch"), "uv_info")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    # STRETCH (2026-10-06): the number a relax is judged by - the UV editor's Display Stretch, per island
    stretch_mode = uv_stretch_mode(params.get("stretch"))
    want_layer = take(params, "layer", default=None)
    max_islands = take_int(params, "maxReportedIslands", default=64)
    # PER-ISLAND ROWS (2026-10-04): which island steals texture space was a recorded script ten times (uv_islands.py).
    want_islands = take_bool(params, "islands", default=False)
    if max_islands < 1:
        raise MifOpError("maxReportedIslands must be at least 1, got %d." % max_islands)

    me = obj.data
    if not me.uv_layers:
        return {
            "object": obj.name,
            "layerCount": 0,
            "layers": [],
            "note": ("this mesh has NO UV layers at all. Texturing and lightmap baking both fail "
                     "until it is unwrapped - call uv_unwrap."),
        }

    names = [l.name for l in me.uv_layers]
    if want_layer is not None and want_layer not in names:
        raise MifOpError("no UV layer named '%s' on '%s'. It has %d: %s."
                         % (want_layer, obj.name, len(names), ", ".join(names)))

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    loop_tris = bm.calc_loop_triangles()

    layers_out = []
    for index, name in enumerate(names):
        if want_layer is not None and name != want_layer:
            continue
        uv_layer = bm.loops.layers.uv.get(name)
        if uv_layer is None:
            continue

        umin = [float("inf"), float("inf")]
        umax = [float("-inf"), float("-inf")]
        outside = 0
        zero_area = 0
        densities = []

        islands, face_island = _uv_islands(bm, uv_layer)

        island_boxes = []
        for island in islands:
            imin = [float("inf"), float("inf")]
            imax = [float("-inf"), float("-inf")]
            for f in island:
                for l in f.loops:
                    uv = l[uv_layer].uv
                    for a, v in enumerate((uv.x, uv.y)):
                        imin[a] = min(imin[a], v)
                        imax[a] = max(imax[a], v)
                        umin[a] = min(umin[a], v)
                        umax[a] = max(umax[a], v)
            island_boxes.append((imin, imax))

        for face in bm.faces:
            uvs = [l[uv_layer].uv for l in face.loops]
            if any(uv.x < -1e-6 or uv.x > 1.0 + 1e-6 or uv.y < -1e-6 or uv.y > 1.0 + 1e-6
                   for uv in uvs):
                outside += 1
            # UV-space area by the shoelace formula, against 3D area, for texel density.
            a2 = 0.0
            for i in range(len(uvs)):
                x1, y1 = uvs[i].x, uvs[i].y
                x2, y2 = uvs[(i + 1) % len(uvs)].x, uvs[(i + 1) % len(uvs)].y
                a2 += x1 * y2 - x2 * y1
            uv_area = abs(a2) * 0.5
            if uv_area <= 1e-12:
                zero_area += 1
                continue
            face_area = face.calc_area()
            if face_area > 1e-12:
                densities.append((uv_area / face_area) ** 0.5)

        # BOX PAIRS ARE CANDIDATES; THE TRIANGLES DECIDE. See the docstring for the 48 pairs this used to report.
        box_pairs, capped = _uv_box_pairs(island_boxes)
        overlaps = touching = unconfirmed = 0
        if box_pairs:
            # DEFERRED, not at module level: the offline suites import this module against a stub `mathutils` that is a
            # plain module, and a module-level `from mathutils.geometry import ...` crashed test_blender_refusals.
            from mathutils.geometry import intersect_tri_tri_2d   # noqa: PLC0415
            island_tris = [[] for _ in islands]
            for lt in loop_tris:
                t = tuple(Vector((l[uv_layer].uv.x, l[uv_layer].uv.y)) for l in lt)
                island_tris[face_island[lt[0].face.index]].append(
                    (t, min(p.x for p in t), max(p.x for p in t), min(p.y for p in t), max(p.y for p in t)))
            budget = 400000          # triangle-pair tests per layer; a pair past it is counted by its box
            filter_left = _UV_TRI_FILTER_LIMIT
            for i, j in box_pairs:
                (amin, amax), (bmin, bmax) = island_boxes[i], island_boxes[j]
                filter_left -= len(island_tris[i]) + len(island_tris[j])
                if filter_left < 0:
                    unconfirmed += 1
                    continue
                # Inclusive, so a triangle that only touches the other island's box stays a candidate.
                cand_a = [e[0] for e in island_tris[i]
                          if e[1] <= bmax[0] and e[2] >= bmin[0] and e[3] <= bmax[1] and e[4] >= bmin[1]]
                cand_b = [e[0] for e in island_tris[j]
                          if e[1] <= amax[0] and e[2] >= amin[0] and e[3] <= amax[1] and e[4] >= amin[1]]
                cost = len(cand_a) * len(cand_b)
                if cost > budget:
                    unconfirmed += 1
                    continue
                budget -= cost
                touch = share = False
                for ta in cand_a:
                    for tb in cand_b:
                        if intersect_tri_tri_2d(*ta, *tb):
                            touch = True
                            if intersect_tri_tri_2d(*_uv_tri_shrunk(ta), *_uv_tri_shrunk(tb)):
                                share = True
                                break
                    if share:
                        break
                if share:
                    overlaps += 1
                elif touch:
                    touching += 1
            overlaps += unconfirmed

        densities.sort()
        face_total = len(bm.faces)
        deg = uv_degeneracy(me, name)
        # FLIPPED ISLANDS (2026-10-06; 5.2's uv.select_by_winding, native here): a face whose UV corners wind clockwise
        # has a negative signed UV area - mirrored in UV space, so its tangents point the other way. An island is
        # flipped when its summed signed area is negative.
        island_area = [0.0] * len(islands)
        flipped_faces = 0
        for f in bm.faces:
            co = [lp[uv_layer].uv for lp in f.loops]
            a = 0.5 * sum(p.x * q.y - q.x * p.y for p, q in zip(co, co[1:] + co[:1]))
            if a < 0:
                flipped_faces += 1
            island_area[face_island[f.index]] += a
        flipped_islands = [k for k, a in enumerate(island_area) if a < 0]
        row = {
            "name": name,
            "index": index,
            "islandCount": len(islands),
            "flippedIslands": len(flipped_islands),
            "flippedIslandIndices": flipped_islands[:20],
            "flippedFaces": flipped_faces,
            "boundsMin": rnd(umin) if face_total else None,
            "boundsMax": rnd(umax) if face_total else None,
            "facesTotal": face_total,
            "facesOutside01": outside,
            "zeroAreaFaces": zero_area,
            "zeroAreaThreshold": 1e-12,
            "nearZeroUvAreaFaces": deg.get("nearZeroUvAreaFaces"),
            "nearZeroUvAreaFaceIndices": deg.get("nearZeroUvAreaFaceIndices") or None,
            "nearZeroUvAreaThreshold": NEAR_ZERO_UV_AREA,
            "nearZeroTangents": deg.get("tangents"),
            "overlappingIslandPairs": overlaps,
            "touchingIslandPairs": touching,
            "boxOverlapPairs": len(box_pairs),
            "pairsNotConfirmed": unconfirmed,
            "overlapCheckCapped": capped,
            "texelDensityMin": round(densities[0], 6) if densities else None,
            "texelDensityMax": round(densities[-1], 6) if densities else None,
            "texelDensityMedian": round(densities[len(densities) // 2], 6) if densities else None,
        }
        if want_islands:
            irows = _island_rows(islands, uv_layer, obj.matrix_world.to_3x3())
            med, spread = _ratio_stats(irows)
            irows.sort(key=lambda r: -r["_uva"])
            row["islands"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in irows[:max_islands]]
            row["islandsReported"] = len(row["islands"])
            row["islandRatioMedian"] = round(med, 6) if med is not None else None
            row["islandRatioSpread"] = round(spread, 4) if spread is not None else None
            row["islandsNote"] = ("biggest UV area first. ratio is UV area per square world unit (the object's "
                                  "scale and rotation applied); islandRatioSpread is the largest ratio over the "
                                  "smallest - 1.0 when every island has the same texel density. uv_pack "
                                  "equalizeByArea makes it so.")
        if stretch_mode is not None:
            row["stretch"] = uv_stretch(islands, uv_layer, obj.matrix_world.to_3x3(), stretch_mode, max_islands)
        if len(islands) > max_islands:
            row["islandsTruncatedNote"] = (
                "%d islands, more than maxReportedIslands (%d). The COUNTS above are exact - only "
                "per-island detail is truncated%s."
                % (len(islands), max_islands, " (islands lists the biggest by UV area)" if want_islands
                   else ", and none is reported unless islands:true"))

        reasons = []
        if outside:
            reasons.append("%d of %d faces lie outside 0-1" % (outside, face_total))
        if overlaps:
            reasons.append("%d island pair(s) share UV area" % overlaps)
        if touching:
            reasons.append("%d island pair(s) touch with no gap, so light bleeds across the seam" % touching)
        if capped:
            reasons.append("the overlap search stopped at its work limit, so it did not clear every island")
        if zero_area:
            reasons.append("%d face(s) have zero UV area" % zero_area)
        row["lightmapReady"] = not reasons
        if reasons:
            row["lightmapReadyReason"] = (
                "not usable as a LIGHTMAP channel: " + "; ".join(reasons) +
                ". None of this is a defect on a TEXTURE channel - tiling outside 0-1 and "
                "overlapping islands are both normal and often deliberate there.")
        if box_pairs:
            row["overlapNote"] = (
                "%d island pair(s) overlap by bounding box, and each was checked triangle by triangle "
                "(mathutils intersect_tri_tri_2d; triangles shrunk 0.01%% so a shared edge is not counted as area): "
                "%d share area, %d only touch, %d are apart.%s"
                % (len(box_pairs), overlaps - unconfirmed, touching, len(box_pairs) - overlaps - touching,
                   ((" %d pair(s) were past the work budget and are counted as overlapping by their box."
                     % unconfirmed) if unconfirmed else "")
                   + (" The search stopped at its work limit (overlapCheckCapped), so every count here is a lower "
                      "bound." if capped else "")))
        layers_out.append(row)

    bm.free()
    uv0 = next((r for r in layers_out if r["index"] == 0), None)
    zero_t = ((uv0 or {}).get("nearZeroTangents") or {}).get("nearZeroVertexInstances")
    return {
        "object": obj.name,
        "layerCount": len(names),
        "activeLayer": me.uv_layers.active.name if me.uv_layers.active else None,
        "layers": layers_out,
        # UNREAL BUILDS TANGENTS FROM UV CHANNEL 0, so only that layer's count predicts its import warning.
        "unrealTangentWarning": ("%d vertex instance(s) on UV channel 0 ('%s') will get nearly zero tangents in Unreal "
                                 "(tolerance 1E-4), which it warns about on import and shades wrongly - faces whose "
                                 "UVs collapse to a line or a point; that layer's nearZeroTangents names them."
                                 % (zero_t, uv0["name"])) if zero_t else None,
        "indexNote": ("index is positional and is what FBX writes and what Unreal's Lightmap "
                      "Coordinate Index points at."),
    }


_UCX_KEYS = {"object", "name", "index", "worldSpace", "maxVertices", "collection", "prefix",
             "shape", "perPart", "maxHulls", "vertexGroup", "vertexIndices", "selected"}

# Unreal's collision-mesh naming. UCX_<RenderMeshName>_## alongside the render mesh in one FBX, and
# the importer attaches it as a convex collision primitive.
#
# STATED AS THE CONVENTION, NOT AS SOMETHING THIS ADDON VERIFIED. Everything else in this op was
# measured on four real Blenders; whether a given Unreal build parses a given name is a UE-side
# behaviour that cannot be checked from here, and this repo's rule is to say so rather than to imply
# a test that never ran.
_UCX_PREFIX = "UCX_"

# UE's convex collision limit. Above this the importer rejects or simplifies the hull depending on
# version, so it is reported rather than silently exceeded.
_UE_HULL_VERT_LIMIT = 255


def _hull_audit(mesh):
    """The seven measurements that separate a real hull from one that merely looks like one.

    A BROKEN HULL HAS THE RIGHT VERTEX COUNT. Measured on Suzanne, 4.4.0 and 5.0.1: the recipe every
    tutorial gives - bm.from_mesh(source), convex_hull, delete geom_interior - returns 66 vertices,
    and so does the correct one. Identical count, and the broken one carries 51 NON-MANIFOLD edges,
    4 boundary edges, Euler 16 instead of 2, 22 convexity violations and a volume inflated from
    3.5321 to 3.8298. Nothing raises. Anything that checks "did I get a hull with a sensible number
    of verts" passes on the wreck.

    So the postcondition is these seven, re-read from the finished MESH DATABLOCK rather than from
    the bmesh that built it - a bmesh reports what was constructed, the datablock reports what
    survived being written.
    """
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        boundary = sum(1 for e in bm.edges if len(e.link_faces) == 1)
        nonmanifold = sum(1 for e in bm.edges if len(e.link_faces) > 2)
        loose = sum(1 for v in bm.verts if not v.link_edges)
        euler = len(bm.verts) - len(bm.edges) + len(bm.faces)
        volume = bm.calc_volume(signed=True)
        # CONVEXITY, MEASURED RATHER THAN ASSUMED: every vertex must lie behind every face plane.
        # O(V*F), which is 8448 dot products for a 66-vertex hull - cheap enough to run every time,
        # and the only check that catches a hull that is closed and manifold and still concave.
        span = 1.0
        if bm.verts:
            span = max((max(v.co[i] for v in bm.verts) - min(v.co[i] for v in bm.verts))
                       for i in range(3)) or 1.0
        violations = _convexity_violations(bm, span)
        return {
            "vertices": len(bm.verts), "edges": len(bm.edges), "faces": len(bm.faces),
            "boundaryEdges": boundary, "nonManifoldEdges": nonmanifold, "looseVertices": loose,
            "eulerCharacteristic": euler, "volume": round(float(volume), 6),
            "convexityViolations": violations,
        }
    finally:
        bm.free()


# HOW FAR, AS A SHARE OF THE HULL'S LARGEST EXTENT, A VERTEX MAY MOVE AND THE HULL STILL COUNT AS CONVEX - see
# _convexity_violations for the measurement it rests on.
_CONVEX_SLACK = 1e-4


def _convexity_violations(bm, span):
    """How many (face, vertex) pairs put the vertex OUTSIDE the face's plane by more than that face can be trusted to.

    THE TOLERANCE IS PER FACE, because a sliver's plane cannot be trusted at a distance. A dense hull from
    bmesh.ops.convex_hull carries long thin triangles where nearly coplanar points meet; such a face's normal tilts
    with any movement of its vertices across its own narrow width, and at the far side of the hull the tilt reads as a
    point standing outside. Measured on Blender 5.2.1, bl_build_rock rocks, exact hull: seed 7 at 180x120x90 cm (607
    hull vertices) gave 108 violations up to 3.4 mm out, seed 3 at 200x150x120 cm (1,855) gave 2,581 up to 47 cm out
    of a 2.5 mm^2 sliver - identical in double precision, every face outward, Euler 2, closed - so the op deleted
    every exact hull of a dense rock (benchmark round ten, 6 of 6 calls). A fixed distance cannot tell that from a
    fold.

    So the test is "convex once every vertex may move by eps": a vertex at distance r from a face of altitude h may sit
    eps * (1 + 2r/h) outside its plane. eps is _CONVEX_SLACK of the span: on those two rocks the largest movement any
    flagged pair needed was 1.8e-5 and 3.3e-5 of the span (convex_hull's own coplanar slack, 66 um on a 2 m rock), so
    the slack is three times the worst measured. A real fold - a corner pushed in, the tutorial recipe's interior
    geometry - needs a movement of a sizeable share of the hull and fails as before (test_blender_collision_parts
    CP111); the tutorial wreck also fails the Euler and manifold checks. Planes are computed in doubles from the
    face's vertices, not read from the single-precision normal."""
    eps = max(_CONVEX_SLACK * span, 1e-7)
    planes = _face_planes(([tuple(float(c) for c in v.co) for v in f.verts], f.normal) for f in bm.faces)
    violations = 0
    for v in bm.verts:
        p = tuple(float(c) for c in v.co)
        violations += sum(1 for pl in planes if _past_plane(p, pl, eps) is not None)
    return violations


def _face_planes(faces):
    """(unit normal, a point on it, centre, altitude) per face, in doubles - from (vertex coordinates, the face's own
    normal for the side) pairs. The best-conditioned corner gives the plane: an n-gon's first three may be nearly
    collinear. A face with no area gives no plane."""
    planes = []
    for fv, fn in faces:
        a = fv[0]
        best, normal = 0.0, None
        for k in range(1, len(fv) - 1):
            u = [fv[k][i] - a[i] for i in range(3)]
            w = [fv[k + 1][i] - a[i] for i in range(3)]
            n = (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0])
            ln = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
            if ln > best:
                best, normal = ln, n
        if normal is None or best <= 0.0:
            continue
        sign = 1.0 if (normal[0] * fn[0] + normal[1] * fn[1] + normal[2] * fn[2]) >= 0.0 else -1.0
        normal = tuple(sign * c / best for c in normal)
        longest = max(math.dist(fv[k], fv[(k + 1) % len(fv)]) for k in range(len(fv)))
        # the altitude: twice the (best corner's) area over the longest edge
        planes.append((normal, a, tuple(sum(q[i] for q in fv) / len(fv) for i in range(3)),
                       best / longest if longest > 0.0 else 0.0))
    return planes


def _past_plane(p, plane, eps):
    """How far p stands outside the plane, when that is more than the face can resolve (eps * (1 + 2r/h), see
    _convexity_violations); None when it is inside, on it, or within that."""
    normal, a, centre, height = plane
    d = normal[0] * (p[0] - a[0]) + normal[1] * (p[1] - a[1]) + normal[2] * (p[2] - a[2])
    if d <= eps:
        return None
    if height > 0.0 and d <= eps * (1.0 + 2.0 * math.dist(p, centre) / height):
        return None
    return d


def _build_hull(points, merge_dist):
    """A bmesh holding the convex hull of a POINT SET. Caller frees it.

    POINTS ONLY - never bm.from_mesh(source). That is the whole finding and it is not a nicety:
    feeding convex_hull a bmesh that already carries the source's edges and faces produces a hull
    with the source's topology tangled through it, and the result is broken on ANY concave mesh
    while looking entirely plausible. See _hull_audit.

    THE THREE RESULT KEYS OVERLAP AND NONE IS RELIABLY POPULATED. geom_interior and geom_unused
    were the same four vertices on a convex test case and 441-and-empty on Suzanne, so all three are
    collected and DEDUPED BY id() - deleting the same vertex twice is a crash, and trusting one key
    alone leaves interior geometry behind.

    context='VERTS' is the only delete context that removes anything: measured, EDGES, FACES,
    EDGES_FACES, FACES_ONLY and FACES_KEEP_BOUNDARY all leave the mesh untouched with no exception
    and no warning. Five of six spellings silently do nothing.
    """
    bm = bmesh.new()
    for co in points:
        bm.verts.new(co)
    bm.verts.ensure_lookup_table()
    if merge_dist > 0.0:
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=merge_dist)
        bm.verts.ensure_lookup_table()
    result = bmesh.ops.convex_hull(bm, input=bm.verts)
    seen, doomed = set(), []
    for key in ("geom_interior", "geom_unused", "geom_holes"):
        for element in result.get(key, []):
            if id(element) not in seen:
                seen.add(id(element))
                doomed.append(element)
    if doomed:
        bmesh.ops.delete(bm, geom=doomed, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def _hull_subset(src, params, per_part):
    """create_collision_hull's vertexGroup | vertexIndices | selected: the vertex indices to hull, or None for all."""
    by = [k for k in ("vertexGroup", "vertexIndices", "selected") if params.get(k) not in (None, False)]
    if not by:
        return None
    if len(by) > 1 or per_part:
        raise MifOpError("pass one of vertexGroup, vertexIndices or selected:true, and not with perPart (got %s). "
                         "NOTHING was created." % ", ".join(by + (["perPart"] if per_part else [])))
    mesh = src.data
    if by[0] == "vertexGroup":
        group = src.vertex_groups.get(take(params, "vertexGroup", kind=str))
        if group is None:
            raise MifOpError("'%s' has no vertex group %r (it has %s). NOTHING was created." % (
                src.name, params["vertexGroup"], ", ".join(g.name for g in src.vertex_groups) or "none"))
        picked = [v.index for v in mesh.vertices
                  if any(g.group == group.index and g.weight > 0.0 for g in v.groups)]
    elif by[0] == "vertexIndices":
        raw = params["vertexIndices"]
        if (not isinstance(raw, list) or any(isinstance(i, bool) or not isinstance(i, int)
                                             or not 0 <= i < len(mesh.vertices) for i in raw)):
            raise MifOpError("vertexIndices is a list of vertex indices of '%s' (0 to %d). NOTHING was created."
                             % (src.name, len(mesh.vertices) - 1))
        picked = sorted(set(raw))
    else:
        take_bool(params, "selected")
        picked = [v.index for v in mesh.vertices if v.select]
    if len(picked) < 4:
        raise MifOpError("the %s picks %d vertice(s) of '%s'; a hull needs at least 4. NOTHING was created."
                         % (by[0], len(picked), src.name))
    return picked


def op_create_collision_hull(params):
    """A convex hull collision mesh - or a box, or one per loose part - named for Unreal's UCX_ convention.

    WHY THIS IS NOT add_collision. That op makes an object a RIGID BODY COLLIDER - a physics
    setting, evaluated by Blender's own simulation. This builds a separate MESH that travels in the
    FBX beside the render mesh and becomes collision in the engine. Same word, unrelated jobs, and
    the addon had the physics one and not this one.

    THE TUTORIAL RECIPE IS WRONG AND LOOKS RIGHT. bm.from_mesh(source) then convex_hull then delete
    geom_interior gives, on Suzanne, a mesh with the SAME 66 vertices as the correct hull and also
    51 non-manifold edges, 4 boundary edges, Euler 16 instead of 2, 22 convexity violations and a
    volume inflated by 8%. Nothing raises. Measured on 4.4.0 and 5.0.1. This builds from POINTS ONLY
    and then AUDITS the result, refusing rather than returning a wreck.

    DEGENERATE INPUT PRODUCES SILENT GARBAGE, never an exception - collinear points give an empty
    mesh, coplanar points give a zero-volume sheet - which is why the audit is a refusal and not a
    warning.

    NAME COLLISIONS ARE A UCX KILLER. bpy.data.objects.new on a taken name returns 'UCX_Rock_00.001'
    and that dot-suffix survives FBX export, so the engine parses the base name as 'Rock_00.001' and
    attaches the hull to nothing. Refused rather than uniquified.

    params:
      object / name (str)      the render mesh to hull. Required.
      index (int)              the ## suffix. Default 0, giving UCX_<object>_00.
      name (str)               override the whole object name, bypassing the convention
      prefix (str)             default 'UCX_'
      worldSpace (bool)        hull the world-space points and leave the object at the origin,
                               instead of hulling local points and copying the source transform
      maxVertices (int)        simplify to at most this many, to a hull that still ENCLOSES every point:
                               the exact hull's own face planes, added where it sticks out most
                               (_enclosing_hull). Decimating a hull breaks convexity; merging points and
                               re-hulling, which this did until 2026-10-01, shrank it inside the mesh.
      collection (str)         where to link it. Default the source's own collection.
      shape (str)              'convex' (default) or 'box' - the box of the points' bounds in the
                               hull's own axes (the source's local axes, or world with worldSpace):
                               8 vertices, 6 faces, the cheapest primitive a piece can collide with.
      perPart (bool)           one hull per LOOSE PART (vertices joined by edges), numbered from
                               `index` up - UCX_<object>_00, _01, ... A wall with an opening built
                               from separate pieces keeps its opening; one hull round it would close
                               it. A part too flat or too small to enclose a volume (a card, a
                               single face) gets no hull and is listed in skippedParts.
      maxHulls (int)           perPart only: at most this many hulls. Past it, the parts are merged
                               two at a time - always the pair whose combined box grows least - and
                               each group gets one hull round all its parts (partsPerHull says which).
      vertexGroup | vertexIndices | selected   (2026-10-06) one of them: a hull or box round a SUBSET -
                               the vertices weighted above 0 in a group, listed by index, or selected in
                               the mesh (a head-only box from a vertex cluster). Not with perPart.

    EVERY ANSWER CARRIES hulls[] (one row per hull made: object, vertices, audit, withinEngineLimit)
    and primitiveCount, so a caller reads one shape whatever it asked for. Without perPart the rest
    of the answer is what it always was.
    """
    reject_unknown(params, _UCX_KEYS, "create_collision_hull")
    # EVERY CHEAP ARGUMENT CHECKED BEFORE THE MESH IS TOUCHED. Reading the source's geometry to
    # report a negative index is work done on the way to refusing, and it also makes the guards
    # untestable without a real mesh - which is how three of these went unexercised until an offline
    # check tried them. Same ordering rule as ray_cast and face_info.
    src_name = take(params, "object", required=True, kind=str)
    world = take_bool(params, "worldSpace", default=False)
    index = take_int(params, "index", default=0)
    if index < 0:
        raise MifOpError("index cannot be negative, got %d. NOTHING was created." % index)
    prefix = take(params, "prefix", default=_UCX_PREFIX, kind=str)
    override = take(params, "name", kind=str)
    max_verts_early = take_int(params, "maxVertices", default=None)
    if max_verts_early is not None and max_verts_early < 4:
        raise MifOpError("maxVertices must be at least 4 to enclose a volume, got %d. NOTHING was "
                         "created." % max_verts_early)
    coll_early = take(params, "collection", kind=str)
    if coll_early and bpy.data.collections.get(coll_early) is None:
        known = sorted(c.name for c in bpy.data.collections)[:25]
        raise MifOpError("no collection named '%s'. Present: %s. NOTHING was created."
                         % (coll_early, ", ".join(known) if known else "<none>"))
    shape = (take(params, "shape", default="convex", kind=str) or "convex").lower()
    if shape not in ("convex", "box"):
        raise MifOpError("shape must be 'convex' (the convex hull of the points) or 'box' (the box of their "
                         "bounds, 8 vertices); got %r. NOTHING was created." % shape)
    if shape == "box" and max_verts_early is not None:
        raise MifOpError("maxVertices simplifies a CONVEX hull; a box always has 8 vertices, so it would be ignored. "
                         "Drop maxVertices, or pass shape 'convex'. NOTHING was created.")
    per_part = take_bool(params, "perPart", default=False)
    max_hulls = take_int(params, "maxHulls", default=None)
    if max_hulls is not None and not per_part:
        raise MifOpError("maxHulls is read only with perPart:true - without it there is one hull, so a limit would "
                         "be ignored. NOTHING was created.")
    if max_hulls is not None and max_hulls < 1:
        raise MifOpError("maxHulls must be at least 1, got %d. NOTHING was created." % max_hulls)
    if per_part and override:
        raise MifOpError("name names ONE hull, and perPart makes one per part - they would all want the same name. "
                         "Drop name (the hulls are numbered %s<object>_NN from index), or drop perPart. NOTHING "
                         "was created." % prefix)

    src = get_object(src_name, want_mesh=True)
    mesh = src.data
    if len(mesh.vertices) < 4:
        raise MifOpError("'%s' has %d vertice(s). A convex hull needs at least 4 non-coplanar "
                         "points to enclose any volume at all - fewer produces an empty or flat "
                         "result with no error. NOTHING was created."
                         % (src.name, len(mesh.vertices)))

    max_verts = max_verts_early
    coll = bpy.data.collections.get(coll_early) if coll_early else None
    subset = _hull_subset(src, params, per_part)

    matrix = src.matrix_world
    points = [(matrix @ v.co) if world else v.co.copy() for v in mesh.vertices]
    if subset is not None:
        points = [points[i] for i in subset]
    span = max((max(p[i] for p in points) - min(p[i] for p in points)) for i in range(3)) or 1.0

    # ---- WHICH POINTS EACH HULL IS MADE FROM ---------------------------------------------------------------------
    skipped = []
    merged_groups = None
    part_count = None
    if per_part:
        parts = _loose_parts(mesh)
        part_count = len(parts)
        groups = [[k] for k in range(len(parts))]
        if max_hulls is not None and len(groups) > max_hulls:
            if len(parts) > _HULL_MERGE_PART_CAP:
                raise MifOpError(
                    "'%s' has %d loose parts, more than the %d this op merges down to a hull limit (every pair is "
                    "weighed). Join the parts that belong together first, or drop maxHulls. NOTHING was created."
                    % (src.name, len(parts), _HULL_MERGE_PART_CAP))
            groups = _merge_parts_by_box(parts, points, max_hulls)
            merged_groups = groups
        plans = []
        for g in groups:
            pts = [points[i] for k in g for i in parts[k]]
            reason = _hull_degenerate(pts, shape, span)
            if reason:
                skipped.append({"parts": g, "vertices": len(pts), "reason": reason})
            else:
                plans.append((g, pts))
        if not plans:
            raise MifOpError("none of the %d loose part(s) of '%s' encloses a volume, so no hull was made: %s. "
                             "NOTHING was created." % (len(parts), src.name,
                                                       "; ".join("parts %s: %s" % (s["parts"], s["reason"])
                                                                 for s in skipped[:8])))
        names = ["%s%s_%02d" % (prefix, src.name, index + k) for k in range(len(plans))]
    else:
        plans = [(None, points)]
        names = [override or ("%s%s_%02d" % (prefix, src.name, index))]

    # REFUSED, NOT UNIQUIFIED - EVERY NAME, BEFORE ANY IS MADE. See the docstring: Blender's .001 suffix silently
    # breaks the naming the whole convention rests on, and it survives the export.
    existing = set(bpy.data.objects.keys())
    taken = [n for n in names if n in existing]
    if taken:
        raise MifOpError("an object named '%s' already exists. Blender would silently rename this "
                         "to '%s.001', and that suffix survives FBX export - the engine then parses "
                         "the base name wrongly and attaches the hull to nothing. Pass a different "
                         "index, or delete the existing one. NOTHING was created."
                         % (taken[0], taken[0]))

    target_coll = coll or (src.users_collection[0] if src.users_collection else bpy.context.scene.collection)
    made = []   # (object, mesh, simplifiedTo, parts)
    rows = []
    try:
        for name, (g, pts) in zip(names, plans):
            hull_mesh, simplified, outside = _hull_mesh(name, pts, shape, max_verts, span)
            hull_obj = bpy.data.objects.new(name, hull_mesh)
            target_coll.objects.link(hull_obj)
            if not world:
                hull_obj.matrix_world = src.matrix_world.copy()
            made.append((hull_obj, hull_mesh, simplified, g))
            audit = _hull_audit(hull_mesh)
            # THE REFUSAL IS THE POINT. A hull that fails any of these is worse than no hull: it imports,
            # it looks like collision, and things fall through it.
            failures = _hull_failures(audit)
            if failures:
                raise MifOpError("the hull built from '%s'%s failed its audit and was REMOVED rather than "
                                 "returned: %s. A broken hull imports, looks like collision, and things "
                                 "fall through it. Measurements: %s"
                                 % (src.name, (" (parts %s)" % g) if g is not None else "",
                                    "; ".join(failures), audit))
            rows.append({"object": hull_obj.name, "parts": g, "vertices": audit["vertices"],
                         "simplifiedTo": simplified, "audit": audit, "sourceOutsideM": round(outside, 6),
                         "withinEngineLimit": audit["vertices"] <= _UE_HULL_VERT_LIMIT})
    except Exception:
        # ALL OR NOTHING: a call that made three hulls and failed on the fourth leaves none behind, so a refusal
        # still means NOTHING was created.
        for hull_obj, hull_mesh, _s, _g in made:
            remove_objects([hull_obj])
            bpy.data.meshes.remove(hull_mesh)
        raise

    first = rows[0]
    audit = first["audit"]
    over = [r["object"] for r in rows if not r["withinEngineLimit"]]
    out = {
        "ok": True,
        "object": first["object"],
        "source": src.name,
        "shape": shape,
        "space": "WORLD" if world else "LOCAL (source transform copied)",
        "collection": next((c.name for c in bpy.data.collections if first["object"] in c.objects),
                           bpy.context.scene.collection.name),
        "simplifiedTo": first["simplifiedTo"],
        "audit": audit,
        "sourceVertices": len(mesh.vertices),
        "subsetVertices": len(subset) if subset is not None else None,
        "withinEngineLimit": not over,
        "note": (None if not over else
                 ("%d vertices is above the %d a convex collision hull is usually limited to - pass "
                  "maxVertices to reduce it." % (audit["vertices"], _UE_HULL_VERT_LIMIT)) if not per_part else
                 ("%s above the %d vertices a convex collision hull is usually limited to - pass maxVertices to "
                  "reduce it." % (", ".join(over), _UE_HULL_VERT_LIMIT))),
        "hulls": rows,
        "hullCount": len(rows),
        # ONE HULL IS ONE PRIMITIVE to the engine's importer: a UCX_ mesh becomes one convex element.
        "primitiveCount": len(rows),
        "namingNote": ("UCX_<RenderMesh>_## is the engine-side convention this name follows. That "
                       "half is NOT verified here - everything else in this response was measured "
                       "on this Blender, but whether a given engine build parses a given name "
                       "cannot be checked from inside Blender."),
    }
    if per_part:
        out["partCount"] = part_count
        out["maxHulls"] = max_hulls
        out["partsPerHull"] = [r["parts"] for r in rows]
        out["partsMerged"] = merged_groups is not None
        out["skippedParts"] = skipped
        if skipped:
            out["skippedNote"] = ("%d group(s) of parts got no hull because they enclose no volume (%s). A flat "
                                  "card or a single face has nothing to collide with as a convex element."
                                  % (len(skipped), "; ".join(s["reason"] for s in skipped[:4])))
    return out


# THE MERGE WEIGHS EVERY PAIR (a heap over them, rebuilt around each merge), so its cost grows with the square of the
# parts; past this many it is refused with the count rather than left to run for minutes.
_HULL_MERGE_PART_CAP = 1024


def _loose_parts(mesh):
    """Vertex-index lists, one per loose part (vertices joined by edges), in order of each part's lowest index."""
    n = len(mesh.vertices)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    flat = array.array("i", [0]) * (2 * len(mesh.edges))
    mesh.edges.foreach_get("vertices", flat)
    for k in range(len(mesh.edges)):
        ra, rb = find(flat[2 * k]), find(flat[2 * k + 1])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [groups[r] for r in sorted(groups)]


def _box_of(pts):
    return ([min(p[i] for p in pts) for i in range(3)], [max(p[i] for p in pts) for i in range(3)])


def _box_volume(lo, hi):
    return max(hi[0] - lo[0], 0.0) * max(hi[1] - lo[1], 0.0) * max(hi[2] - lo[2], 0.0)


def _merge_parts_by_box(parts, points, limit):
    """Group part indices into at most `limit` groups: repeatedly join the pair whose combined box grows least.

    Growth is the union box's volume less the two boxes' own, so two pieces that touch or overlap join before two
    that stand apart. Deterministic: ties go to the lower part indices."""
    import heapq
    boxes = {k: _box_of([points[i] for i in part]) for k, part in enumerate(parts)}
    members = {k: [k] for k in boxes}

    def cost(a, b):
        (la, ha), (lb, hb) = boxes[a], boxes[b]
        lo = [min(la[i], lb[i]) for i in range(3)]
        hi = [max(ha[i], hb[i]) for i in range(3)]
        return _box_volume(lo, hi) - _box_volume(la, ha) - _box_volume(lb, hb)

    heap = [(cost(a, b), a, b) for a in boxes for b in boxes if a < b]
    heapq.heapify(heap)
    next_id = len(parts)
    while len(members) > limit and heap:
        _c, a, b = heapq.heappop(heap)
        if a not in members or b not in members:
            continue
        (la, ha), (lb, hb) = boxes.pop(a), boxes.pop(b)
        merged = members.pop(a) + members.pop(b)
        boxes[next_id] = ([min(la[i], lb[i]) for i in range(3)], [max(ha[i], hb[i]) for i in range(3)])
        members[next_id] = sorted(merged)
        for other in members:
            if other != next_id:
                heapq.heappush(heap, (cost(min(other, next_id), max(other, next_id)), min(other, next_id),
                                      max(other, next_id)))
        next_id += 1
    return sorted(members.values())


def _hull_degenerate(pts, shape, span):
    """Why these points cannot enclose a volume, or None. Checked before an object is made, per part."""
    if len(pts) < 4:
        return "%d vertice(s), fewer than the 4 a volume needs" % len(pts)
    lo, hi = _box_of(pts)
    thin = [("xyz"[i]) for i in range(3) if hi[i] - lo[i] <= max(span * 1e-6, 1e-7)]
    if thin:
        return "flat along %s (no thickness to collide with)" % "".join(thin)
    if shape == "convex":
        bm = _build_hull(pts, merge_dist=span * 1e-6)
        try:
            if len(bm.verts) < 4 or abs(bm.calc_volume(signed=True)) <= 1e-9:
                return "the points are coplanar or collinear, so the hull encloses nothing"
        finally:
            bm.free()
    return None


def _hull_mesh(name, pts, shape, max_verts, span):
    """(mesh datablock, simplifiedTo) for one hull. A box is its 8 corners and 6 quads; a convex hull comes from the
    points only, simplified to an enclosing hull of support planes when maxVertices asks. Returns (mesh,
    simplifiedTo, sourceOutsideM)."""
    if shape == "box":
        lo, hi = _box_of(pts)
        bm = bmesh.new()
        try:
            v = [bm.verts.new((x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
            # Corner k is (x, y, z) with x = k>>2, y = (k>>1)&1, z = k&1; each quad listed counter-clockwise from
            # outside, then recalc_face_normals makes that true whatever the listing got wrong.
            for quad in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
                bm.faces.new([v[i] for i in quad])
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            hull_mesh = bpy.data.meshes.new(name)
            bm.to_mesh(hull_mesh)
        finally:
            bm.free()
        return hull_mesh, None, 0.0
    bm = _build_hull(pts, merge_dist=span * 1e-6)
    simplified = None
    try:
        # SIMPLIFIED TO A HULL THAT STILL ENCLOSES, never by decimating and no longer by merging. A decimated hull is
        # no longer convex - measured: ratio 0.1 gave 24 convexity violations at maxdist 0.017. Merging the points and
        # re-hulling stays convex, but every merged point lies INSIDE the hull it came from, so the hull shrank: on a
        # 180x120x90 cm build_rock at 48 vertices it topped out at 0.811 m over a rock reaching 0.846, 754 of the
        # rock's 2,500 vertices more than 2 cm outside it, and on seed 11 at 120x90x70 the merged hull failed its own
        # convexity audit, so build_rock delivered no hull at all (benchmark round ten, 5.2.1). _enclosing_hull cuts
        # the hull's own support planes instead - see there.
        keep = [vv.co.copy() for vv in bm.verts]
        if max_verts is not None and len(bm.verts) > max_verts:
            faces = [(f.normal.copy(), f.calc_area()) for f in bm.faces]
            bm.free()
            bm = _enclosing_hull(keep, faces, max_verts, span)
            simplified = len(bm.verts)
        hull_mesh = bpy.data.meshes.new(name)
        bm.to_mesh(hull_mesh)
    finally:
        bm.free()
    # MEASURED, NOT ASSUMED: the exact hull's vertices are the points that matter (every other point lies inside them)
    return hull_mesh, simplified, _source_outside(keep, hull_mesh)


# HOW FAR OUT AN ENCLOSING HULL'S PLANES SIT past the hull they simplify, and how close two of its corners may come
# before they are one, both as shares of the points' largest extent. The merge can move a corner inward by at most the
# merge distance, so the margin is twice it: every point of the source stays inside (sourceOutsideM measures it).
_ENCLOSE_MARGIN = 2e-4
_ENCLOSE_MERGE = 1e-4
# Two of the hull's faces closer in direction than this are one candidate plane (the larger kept): near-parallel planes
# meet in slivers whose normals the audit cannot trust.
_ENCLOSE_SAME_DIR = math.cos(math.radians(2.0))


def _support_planes(dirs, hull_pts, span):
    """(normal, offset) per direction: the plane touching the hull points on that side, pushed out by the margin."""
    out = []
    for n in dirs:
        n = Vector(n).normalized()
        out.append((n, max(n.dot(p) for p in hull_pts) + span * _ENCLOSE_MARGIN))
    return out


def _det3(q):
    return (q[0][0] * (q[1][1] * q[2][2] - q[1][2] * q[2][1]) - q[0][1] * (q[1][0] * q[2][2] - q[1][2] * q[2][0])
            + q[0][2] * (q[1][0] * q[2][1] - q[1][1] * q[2][0]))


def _plane_corners(planes, center, merge):
    """The corners of the intersection of half-spaces n.x <= d, each holding `center` strictly inside, or None.

    BY DUALITY: each plane maps to the point n / (d - n.center), and each face of those points' convex hull is a
    corner where its planes meet - solved from three of them, in doubles. Planes that touch nothing are interior dual
    points and simply drop out. Corners closer than `merge` are one."""
    dual = bmesh.new()
    try:
        for n, d in planes:
            dual.verts.new(n / (d - n.dot(center)))
        dual.verts.ensure_lookup_table()
        hull = bmesh.ops.convex_hull(dual, input=dual.verts)
        dual.verts.index_update()
        corners = []
        for f in (g for g in hull.get("geom", []) if isinstance(g, bmesh.types.BMFace)):
            rows = [planes[v.index] for v in f.verts[:3]]
            # mathutils is single precision; Cramer's rule in Python floats is double
            a = [[float(c) for c in r[0]] for r in rows]
            b = [float(r[1]) for r in rows]
            dd = _det3(a)
            if abs(dd) < 1e-12:
                return None
            x = []
            for col in range(3):
                q = [row[:] for row in a]
                for r in range(3):
                    q[r][col] = b[r]
                x.append(_det3(q) / dd)
            p = Vector(x)
            if not any((p - c).length <= merge for c in corners):
                corners.append(p)
        return corners
    finally:
        dual.free()


def _enclosing_hull(hull_pts, hull_faces, max_verts, span):
    """A convex hull of at most max_verts vertices that holds every one of hull_pts (the exact hull's vertices) - a
    bmesh, which the caller frees.

    SUPPORT PLANES, ADDED WHERE THE HULL STICKS OUT MOST. It starts from the six axis planes touching the points (four
    tetrahedron planes when the budget is under 8) and repeatedly takes the corner farthest outside the exact hull and
    cuts it with the exact hull's own face plane that cuts deepest there, kept only while the corners stay within the
    budget. Every plane touches the points from outside, so the hull encloses them by construction, and it tightens
    toward the exact hull as planes are added. The corners are meshed through _build_hull like any hull, and audited by
    the caller; a hull that would not pass is rebuilt without its last plane."""
    center = Vector((0.0, 0.0, 0.0))
    for p in hull_pts:
        center += p
    center /= len(hull_pts)
    if max_verts >= 8:
        start = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    else:
        start = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    chosen = _support_planes(start, hull_pts, span)
    # the exact hull's faces, largest first, one per direction
    cands = []
    for n, area in sorted(hull_faces, key=lambda r: -r[1]):
        if n.length < 0.5 or any(n.dot(c) > _ENCLOSE_SAME_DIR for c in cands):
            continue
        cands.append(n.normalized())
    exact = [(n, max(n.dot(p) for p in hull_pts)) for n in cands]
    cand_planes = _support_planes(cands, hull_pts, span)
    merge = span * _ENCLOSE_MERGE
    corners = _plane_corners(chosen, center, merge)
    banned = set()
    history = []
    for _ in range(4 * max_verts + 16):
        if corners is None:
            break
        # how far each corner lies outside the exact hull, and the candidate plane that cuts it deepest
        rows = []
        for c in corners:
            k, out = max(((k, n.dot(c) - d) for k, (n, d) in enumerate(exact)), key=lambda kv: kv[1])
            rows.append((out, k))
        rows.sort(reverse=True)
        if not rows or rows[0][0] <= span * _ENCLOSE_MARGIN * 2.0:
            break
        added = False
        tries = 0
        for out, k in rows:
            if out <= span * _ENCLOSE_MARGIN * 2.0 or tries >= 8:
                break
            if k in banned:
                continue
            tries += 1
            trial = _plane_corners(chosen + [cand_planes[k]], center, merge)
            if trial is not None and len(trial) <= max_verts:
                history.append((list(chosen), corners))
                chosen.append(cand_planes[k])
                corners = trial
                added = True
                break
            banned.add(k)
        if not added:
            break
    while True:
        bm = _build_hull(corners or [], merge_dist=merge)
        bm.normal_update()
        # the audit's own test (_convexity_violations), so a hull kept here is one the caller's audit passes
        sound = len(bm.verts) >= 4 and _convexity_violations(bm, span) == 0
        if sound or not history:
            return bm
        bm.free()
        chosen, corners = history.pop()


def _source_outside(obj_points, hull_mesh):
    """The farthest any of the points lies outside the hull's face planes, in the hull's own space - 0 when all are
    inside or on it. A point counts as on a face when it is within what that face can resolve (_past_plane, the
    audit's own allowance): a sliver of a dense exact hull otherwise put a rock's own vertices 47 cm outside the hull
    built from them (seed 3, 1,855 hull vertices, Blender 5.2.1)."""
    verts = [tuple(float(c) for c in v.co) for v in hull_mesh.vertices]
    if not verts:
        return 0.0
    span = max((max(q[i] for q in verts) - min(q[i] for q in verts)) for i in range(3)) or 1.0
    eps = max(_CONVEX_SLACK * span, 1e-7)
    planes = _face_planes(([verts[i] for i in poly.vertices], poly.normal) for poly in hull_mesh.polygons)
    worst = 0.0
    for p in obj_points:
        p = tuple(float(c) for c in p)
        for pl in planes:
            d = _past_plane(p, pl, eps)
            if d is not None and d > worst:
                worst = d
    return worst


def _hull_failures(audit):
    """The audit's failures as sentences; empty when the hull is sound."""
    failures = []
    if audit["boundaryEdges"]:
        failures.append("%d boundary edge(s) - the hull is not closed" % audit["boundaryEdges"])
    if audit["nonManifoldEdges"]:
        failures.append("%d non-manifold edge(s)" % audit["nonManifoldEdges"])
    if audit["looseVertices"]:
        failures.append("%d loose vertice(s)" % audit["looseVertices"])
    if audit["eulerCharacteristic"] != 2:
        failures.append("Euler characteristic %d, not 2 - leftover interior geometry"
                        % audit["eulerCharacteristic"])
    if audit["volume"] <= 1e-9:
        failures.append("volume %g - the points are collinear or coplanar, so this encloses nothing"
                        % audit["volume"])
    if audit["convexityViolations"]:
        failures.append("%d convexity violation(s) - vertices lie OUTSIDE the face planes"
                        % audit["convexityViolations"])
    return failures

_RENAME_KEYS = {"object", "name", "to", "newName", "renameData", "allowCollision"}
_VCOL_KEYS = {"object", "name", "color", "domain", "dataType", "faces", "makeActive",
              "makeRender", "vertexValues", "vertexIndices", "falloff", "baseColor", "blend"}
_VCOL_BLENDS = ("REPLACE", "OVER", "MAX")
# set_vertex_color's falloff takes transform_selection's curves (_FALLOFF_CURVES) by the same key, `type`; RANDOM is
# left out - a mask wants the same value on every call. Its own key set, beside transform_selection's _FALLOFF_KEYS.
_VCOL_FALLOFF_TYPES = tuple(k for k, f in _FALLOFF_CURVES.items() if f is not None)
_VCOL_FALLOFF_KEYS = {"points", "radius", "inner", "type"}
_MAX_FALLOFF_POINTS = 4096


def _vertex_value_spec(params, tail):
    """set_vertex_color's per-vertex form, read before anything is written: (kind, payload) or None.

    kind "values": (indices or None, values) - one value 0-1 per vertex, or per listed vertex.
    kind "falloff": (points [Vector], radius, inner, shape) - a value from the distance to the nearest point."""
    values = params.get("vertexValues")
    indices = params.get("vertexIndices")
    fall = params.get("falloff")
    if values is None and fall is None:
        if indices is not None:
            raise MifOpError("vertexIndices goes with vertexValues (a value for each listed vertex). %s" % tail)
        return None
    if values is not None and fall is not None:
        raise MifOpError("pass vertexValues OR falloff - each gives every vertex its value. %s" % tail)
    if values is not None:
        if not isinstance(values, (list, tuple)) or not values:
            raise MifOpError("vertexValues must be a non-empty list of numbers 0-1, or of [index, value] pairs, got %r. %s"
                             % (type(values).__name__, tail))
        if isinstance(values[0], (list, tuple)):
            # THE PAIRS FORM, [[index, value], ...] - create_mesh's vertexGroups spelling, and the same as vertexIndices
            # beside vertexValues
            if indices is not None:
                raise MifOpError("vertexValues as [index, value] pairs already names its vertices - drop vertexIndices. %s"
                                 % tail)
            if any(not isinstance(pr, (list, tuple)) or len(pr) != 2 or isinstance(pr[0], bool)
                   or not isinstance(pr[0], int) for pr in values):
                raise MifOpError("vertexValues pairs must each be [vertex index, value 0-1]. %s" % tail)
            indices = [pr[0] for pr in values]
            values = [pr[1] for pr in values]
        vals = finite_floats(list(values), "vertexValues")
        bad = [i for i, v in enumerate(vals) if not 0.0 <= v <= 1.0]
        if bad:
            raise MifOpError("vertexValues[%d] is %g - each value is 0 (baseColor) to 1 (color). %s"
                             % (bad[0], vals[bad[0]], tail))
        idx = None
        if indices is not None:
            if not isinstance(indices, (list, tuple)) or any(isinstance(i, bool) or not isinstance(i, int)
                                                             for i in indices):
                raise MifOpError("vertexIndices must be a list of whole vertex indices. %s" % tail)
            if len(indices) != len(vals):
                raise MifOpError("vertexIndices has %d entries and vertexValues %d - one value per listed vertex. %s"
                                 % (len(indices), len(vals), tail))
            if len(set(indices)) != len(indices):
                raise MifOpError("vertexIndices lists a vertex more than once. %s" % tail)
            idx = list(indices)
        return "values", (idx, vals)
    if not isinstance(fall, dict) or set(fall) - _VCOL_FALLOFF_KEYS:
        raise MifOpError("falloff is {points: [[x,y,z], ...], radius, inner, type}; got %r. %s" % (fall, tail))
    pts = fall.get("points")
    if not isinstance(pts, (list, tuple)) or not pts or len(pts) > _MAX_FALLOFF_POINTS or any(
            not isinstance(p, (list, tuple)) or len(p) != 3 for p in pts):
        raise MifOpError("falloff.points must be 1 to %d world points [x,y,z]. %s" % (_MAX_FALLOFF_POINTS, tail))
    pts = [Vector(finite_floats(list(p), "falloff.points")) for p in pts]
    radius = take_float(fall, "radius", required=True)
    inner = take_float(fall, "inner", default=0.0)
    if not radius > 0.0 or not 0.0 <= inner < radius:
        raise MifOpError("falloff needs radius above 0 and inner from 0 up to below radius (world units), got radius %r "
                         "and inner %r. %s" % (radius, inner, tail))
    shape = str(fall.get("type", "SMOOTH")).upper()
    if shape not in _VCOL_FALLOFF_TYPES:
        raise MifOpError("falloff.type must be one of %s, got %r. %s" % (", ".join(_VCOL_FALLOFF_TYPES), fall.get("type"),
                                                                          tail))
    return "falloff", (pts, radius, inner, shape)


def _falloff_value(d, radius, inner, shape):
    """1 within inner, 0 at radius and beyond, between them by transform_selection's curve (t runs 1 at inner to 0 at
    radius)."""
    if d <= inner:
        return 1.0
    if d >= radius:
        return 0.0
    return _FALLOFF_CURVES[shape](1.0 - (d - inner) / (radius - inner))
_STATS_KEYS = {"object", "name", "evaluated", "attributeStats", "frame", "components", "attributeValues",
               "allowSimulation", "instances"}
_INST_KEYS = {"start", "limit", "matrix"}


def _parse_instances(params):
    """None, or {start, limit, matrix} from instances: true | {start, limit, matrix}. Checked before anything is read."""
    raw = params.get("instances")
    if raw is None or raw is False:
        return None
    if raw is True:
        raw = {}
    if not isinstance(raw, dict):
        raise MifOpError("instances is true or {start, limit, matrix}, got %r. NOTHING was read." % (raw,))
    bad = sorted(k for k in raw if k not in _INST_KEYS)
    if bad:
        raise MifOpError("instances takes %s; %s is not one. NOTHING was read."
                         % (", ".join(sorted(_INST_KEYS)), ", ".join(bad)))
    limit = raw.get("limit", 1000)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10000:
        raise MifOpError("instances.limit must be a whole number 1-10000, got %r. NOTHING was read." % (limit,))
    start = raw.get("start", 0)
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise MifOpError("instances.start must be a whole number 0 or more, got %r. NOTHING was read." % (start,))
    matrix = raw.get("matrix", False)
    if not isinstance(matrix, bool):
        raise MifOpError("instances.matrix is true or false, got %r. NOTHING was read." % (matrix,))
    return {"start": start, "limit": limit, "matrix": matrix}


def _instance_source(inst, parent_name):
    """(key, row) naming what one depsgraph instance draws.

    MEASURED ON 5.2.1 (2026-10-25). A collection instance, a particle instance or vertex / face instancing draws a REAL
    object: inst.object.original is that object. A Geometry Nodes instance draws a temporary object named after the
    INSTANCER, whose data is an evaluated copy with no pointer back to bpy.data (data.original is itself, is_evaluated is
    false): an Object Info instance's data keeps the source mesh's NAME ('Cube.001' off a Pebble whose mesh is
    'Cube.001'), and geometry the tree made itself (a Cone node) is named 'Mesh' and is in no bpy.data list. So a Geometry
    Nodes source is found by its data name among bpy.data's meshes and the objects that use it, and is GEOMETRY when no
    such mesh exists."""
    o = inst.object
    d = o.data
    otype = o.type
    if o.original.name != parent_name:
        src = o.original
        return (("OBJECT", src.name)), {"kind": "OBJECT", "source": src.name, "type": otype,
                                        "data": src.data.name if src.data is not None else None}
    dname = d.name if d is not None else None
    coll = {"MESH": bpy.data.meshes, "CURVES": getattr(bpy.data, "hair_curves", None),
            "POINTCLOUD": getattr(bpy.data, "pointclouds", None)}.get(otype)
    real = coll.get(dname) if (coll is not None and dname) else None
    if real is not None:
        users = sorted(ob.name for ob in bpy.data.objects if ob.data is real)
        if users:
            return (("OBJECT", users[0])), {"kind": "OBJECT", "source": users[0], "type": otype, "data": dname,
                                            **({"sharedBy": users} if len(users) > 1 else {})}
    nverts = len(d.vertices) if otype == "MESH" and d is not None else None
    return (("GEOMETRY", otype, dname, nverts)), {"kind": "GEOMETRY", "source": None, "type": otype, "data": dname,
                                                  **({"vertices": nverts} if nverts is not None else {})}


def _instance_rows(obj, spec):
    """The `instances` answer: every evaluated instance `obj` draws - its source and world transform - and a count per
    source, paged. depsgraph.object_instances is the route the bundled FBX and glTF exporters take."""
    deps = bpy.context.evaluated_depsgraph_get()
    name = obj.name
    groups = {}
    order = []
    rows = []
    total = 0
    first, limit = spec["start"], spec["limit"]
    for inst in deps.object_instances:
        if not inst.is_instance or inst.parent is None or inst.parent.original.name != name:
            continue
        key, src = _instance_source(inst, name)
        g = groups.get(key)
        if g is None:
            g = groups[key] = dict(src, count=0, index=len(order))
            order.append(key)
        g["count"] += 1
        if first <= total < first + limit:
            mw = inst.matrix_world
            loc, rot, scl = mw.decompose()
            e = rot.to_euler("XYZ")
            row = {"i": total, "source": g["index"],
                   "location": [round(v, 6) for v in loc],
                   "rotation": [round(v, 6) for v in e],
                   "scale": [round(v, 6) for v in scl]}
            if spec["matrix"]:
                row["matrix"] = [[round(v, 6) for v in r] for r in mw]
            rows.append(row)
        total += 1
    sources = [dict(groups[k]) for k in order]
    for s in sources:
        s.pop("index", None)
    out = {"count": total, "sources": sources, "rows": rows, "start": first, "limit": limit,
           "truncated": total > first + limit, "readBy": "depsgraph.object_instances",
           "rowsNote": ("each row's source is an index into sources; location in meters, rotation XYZ Euler radians (as transform_object),"
                        "scale - world space. kind GEOMETRY is geometry the node tree made itself (no object holds "
                        "it). A Geometry Nodes point cloud or curves output is drawn as an instance on 3.6-4.4 and "
                        "is listed with its type.")}
    if first and first >= total and total:
        raise MifOpError("instances.start %d is past the last of %d instance(s). NOTHING was read." % (first, total))
    return out

# THE EVALUATED GEOMETRY'S OTHER COMPONENTS (2026-10-04). A Geometry Nodes object can output a mesh, a point cloud,
# curves and instances at once, and the evaluated mesh is only the first. Object.evaluated_geometry() (a GeometrySet)
# reads them all from Blender 5.0 - measured on 5.2.1: .mesh, .pointcloud, .curves, .volume, .grease_pencil and
# .instances_pointcloud(), whose POINT attributes are the instance domain's (an index stored on INSTANCE read 0-6 back
# off 7 instances). It is absent on 3.6.23 and 4.4.0, where only the evaluated mesh and the depsgraph's instance count
# can be read, so those builds answer the mesh and the instance count and refuse the rest by name.
_GEO_COMPONENTS = ("MESH", "POINTCLOUD", "CURVES", "INSTANCES")
_AV_KEYS = {"component", "domain", "name", "limit", "start"}


def _parse_geometry_request(params):
    """(components bool, attributeValues dict or None) - checked before anything is read or the frame moves."""
    comps = take_bool(params, "components", default=False) if "components" in params else False
    av = params.get("attributeValues")
    if av is None:
        return comps, None
    if not isinstance(av, dict) or not av.get("name"):
        raise MifOpError("attributeValues is {component, domain, name, limit} with a name, got %r. NOTHING was read."
                         % (av,))
    bad = sorted(k for k in av if k not in _AV_KEYS)
    if bad:
        raise MifOpError("attributeValues takes %s; %s is not one. NOTHING was read."
                         % (", ".join(sorted(_AV_KEYS)), ", ".join(bad)))
    comp = str(av.get("component", "MESH")).upper()
    if comp not in _GEO_COMPONENTS:
        raise MifOpError("attributeValues.component must be one of %s, got %r. NOTHING was read."
                         % (", ".join(_GEO_COMPONENTS), av.get("component")))
    limit = av.get("limit", 100)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10000:
        raise MifOpError("attributeValues.limit must be a whole number 1-10000, got %r. NOTHING was read." % (limit,))
    # START PAGES PAST THE LIMIT: a mesh of more than 10,000 corners had values no call could reach (2026-10-05).
    start = av.get("start", 0)
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise MifOpError("attributeValues.start must be a whole number 0 or more, got %r. NOTHING was read." % (start,))
    if comp != "MESH" and not hasattr(bpy.types.Object, "evaluated_geometry"):
        raise MifOpError("this Blender (%s) reads an object's evaluated MESH and its instance count only - "
                         "Object.evaluated_geometry, which reads point clouds, curves and the instance domain, arrives "
                         "in 5.0. NOTHING was read." % bpy.app.version_string)
    return comps, {"component": comp, "domain": str(av["domain"]).upper() if av.get("domain") else None,
                   "name": str(av["name"]), "limit": limit, "start": start}


def _attr_value(item):
    for key in ("value", "vector", "color"):
        if hasattr(item, key):
            v = getattr(item, key)
            if isinstance(v, (int, bool, str)):
                return v
            if isinstance(v, float):
                return round(v, 6)
            try:
                return [round(float(x), 6) for x in v]
            except TypeError:
                return [[round(float(x), 6) for x in row] for row in v]
    return None


def _geometry_read(obj, comps, av):
    """The `components` and `attributeValues` answer for one object, read off its evaluated geometry now."""
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    out = {}
    holders = {}
    if hasattr(ev, "evaluated_geometry"):
        gs = ev.evaluated_geometry()
        inst = gs.instances_pointcloud()
        holders = {"MESH": gs.mesh, "POINTCLOUD": gs.pointcloud, "CURVES": gs.curves, "INSTANCES": inst}
        if comps:
            m, pc, cv = gs.mesh, gs.pointcloud, gs.curves
            out["components"] = {
                "mesh": {"vertices": len(m.vertices), "edges": len(m.edges), "faces": len(m.polygons)} if m else None,
                "pointCloud": {"points": len(pc.points)} if pc else None,
                "curves": {"curves": len(cv.curves), "points": len(cv.points)} if cv else None,
                "instances": {"count": len(inst.points)} if inst else None,
                "volume": gs.volume is not None, "greasePencil": getattr(gs, "grease_pencil", None) is not None,
                "attributes": {k.lower(): sorted("%s (%s %s)" % (a.name, a.domain, a.data_type) for a in h.attributes
                                                 if not a.name.startswith("."))
                               for k, h in holders.items() if h is not None},
                "readBy": "Object.evaluated_geometry"}
    else:
        mesh = ev.data if obj.type == "MESH" else None
        holders = {"MESH": mesh}
        if comps:
            out["components"] = {
                "mesh": {"vertices": len(mesh.vertices), "edges": len(mesh.edges), "faces": len(mesh.polygons)}
                if mesh is not None else None,
                # NOT THE INSTANCE COUNT: the depsgraph also draws a point cloud and curves as instances - 7 instances
                # beside one curve and one point cloud read 9 on 3.6.23 and 4.4.0 (measured).
                "drawnInstances": drawn_instances(obj, deps),
                "instances": None, "pointCloud": None, "curves": None,
                "readBy": "the evaluated mesh and the depsgraph's instances",
                "componentsNote": ("this Blender (%s) cannot read an object's point-cloud and curve components or "
                                   "instance attributes - Object.evaluated_geometry arrives in 5.0. drawnInstances "
                                   "counts every instance the depsgraph draws, the point cloud and curves among them. "
                                   "Realize the instances, or convert to a mesh, to read them here."
                                   % bpy.app.version_string)}
    if av is not None:
        holder = holders.get(av["component"])
        if holder is None:
            raise MifOpError("'%s' has no %s component in its evaluated geometry. NOTHING was read."
                             % (obj.name, av["component"].lower()))
        attr = holder.attributes.get(av["name"])
        if attr is None:
            raise MifOpError("the %s component of '%s' has no attribute '%s'. It has: %s. NOTHING was read."
                             % (av["component"].lower(), obj.name, av["name"],
                                ", ".join(sorted(a.name for a in holder.attributes)) or "none"))
        domain = attr.domain
        want = av["domain"]
        # THE INSTANCE DOMAIN IS THE INSTANCES POINT CLOUD'S POINT DOMAIN on 5.x - the same attribute either way.
        if want and not (want == domain or (av["component"] == "INSTANCES" and want == "INSTANCE" and domain == "POINT")):
            raise MifOpError("'%s' is on the %s domain of the %s component, not %s. NOTHING was read."
                             % (av["name"], domain, av["component"].lower(), want))
        n = len(attr.data)
        first = av["start"]
        if first and first >= n:
            raise MifOpError("attributeValues.start %d is past the last of '%s''s %d value(s). NOTHING was read."
                             % (first, attr.name, n))
        out["attributeValues"] = {"component": av["component"], "name": attr.name,
                                  "domain": "INSTANCE" if av["component"] == "INSTANCES" else domain,
                                  "dataType": attr.data_type, "count": n,
                                  "start": first,
                                  "values": [_attr_value(attr.data[i]) for i in range(first, min(n, first + av["limit"]))],
                                  "truncated": n > first + av["limit"]}
    return out


# Blender's ID name limit on 3.6 through 4.4. It rose at 5.0 - a 200-character name truncates to 63
# on the older three and is stored whole on 5.0.1 - which means two names differing only past
# character 63 collide on one build and stay distinct on another. Measured, not remembered.
_ID_NAME_LIMIT = 63


def op_rename_object(params):
    """Rename an object, refusing a collision instead of letting Blender resolve it.

    THE COLLISION RESOLVES IN OPPOSITE DIRECTIONS ACROSS THE 3.6 / 4.x LINE, and that is not a
    curiosity - it is silent corruption of an object the caller never mentioned.

    Measured on all four builds. With an existing object called 'Alpha', setting another object's
    name to 'Alpha' gives:

        3.6.23              whichever of the two current names sorts first keeps the name: 'Zulu' renamed
                            to 'Alpha' TAKES it and the INCUMBENT becomes 'Alpha.001', while 'Aardvark'
                            renamed to 'Mango' becomes 'Mango.001' (both orders measured 2026-10-26; this
                            said the renamer always took it, and the refusal below had it right)
        4.2 / 4.4 / 5.0     the incumbent keeps 'Alpha' and the RENAMER becomes 'Alpha.001'

    So the same script renames a different object depending on the Blender running it, and on 3.6 it
    quietly renames something nobody asked it to touch - which then breaks every STRING reference to
    that object elsewhere in the file. Refusing makes the behaviour identical everywhere.

    THE DATA NAME DOES NOT FOLLOW. After obj.name = 'Renamed', obj.data.name is still whatever it
    was - objects and meshes live in separate namespaces. A mesh still named after the old object
    survives into the FBX and confuses anybody reading it later, so renameData defaults ON.

    POINTERS SURVIVE A RENAME AND STRINGS DO NOT. Modifier .object, constraint .target and driver
    variable targets are pointers and are unaffected - verified live. What breaks is anything
    holding the NAME: a vertex group named after a bone, a shape key driver expression, an exporter
    convention like UCX_<name>_00. Those are reported so the caller can see what may need following.

    params:
      object (str)                which object. Required.
      to / newName / name (str)   the new name. Required.
      renameData (bool)           rename the object's DATA to match. Default true.
      allowCollision (bool)       let Blender resolve a clash its own way, and say which way:
                                  resolvedTo is the name this object got (4.2 and later),
                                  otherObjectRenamed the one the incumbent got (3.6). Default false.
    """
    reject_unknown(params, _RENAME_KEYS, "rename_object")
    src = take(params, "object", required=True, kind=str)
    wanted = take(params, "to", "newName", "name", required=True, kind=str)
    wanted = str(wanted)
    if not wanted.strip():
        raise MifOpError("the new name is empty. NOTHING was changed.")
    allow = take_bool(params, "allowCollision", default=False)
    rename_data = take_bool(params, "renameData", default=True)

    obj = get_object(src)
    require_editable(obj, "rename")
    if obj.name == wanted:
        return {"ok": True, "object": obj.name, "renamedFrom": obj.name, "changed": False,
                "dataName": getattr(obj.data, "name", None),
                "note": "the object is already called that - nothing to do."}

    clash = bpy.data.objects.get(wanted)
    if clash is not None and clash is not obj and not allow:
        raise MifOpError(
            "an object named '%s' already exists, and letting Blender resolve that does DIFFERENT "
            "things on different builds: 4.2 and later always rename THIS object to '%s.001', while "
            "3.6 renames whichever of the two sorts first by name - so with this object currently "
            "called '%s' it would %s, which you cannot tell from the response afterwards. Pick "
            "another name, rename the existing one first, or pass allowCollision:true to accept a "
            "per-version outcome. NOTHING was changed."
            % (wanted, wanted, obj.name,
               ("take the name and silently rename the existing '%s' to '%s.001' - corrupting an "
                "object you did not ask to touch" % (wanted, wanted))
               if obj.name.lower() > wanted.lower() else
               ("also become '%s.001' here" % wanted)))

    if len(wanted) > _ID_NAME_LIMIT and bpy.app.version < (5, 0, 0):
        raise MifOpError(
            "'%s' is %d characters and this Blender (%s) truncates object names at %d, so the name "
            "you get back would not be the one you asked for - and two names differing only past "
            "that point would collide here and not on 5.0. Shorten it. NOTHING was changed."
            % (wanted, len(wanted), bpy.app.version_string, _ID_NAME_LIMIT))

    was = obj.name
    other_before = clash.name if clash is not None else None
    obj.name = wanted
    # THE POSTCONDITION, and it is the whole point on a name assignment: Blender resolves silently,
    # so the only way to know what happened is to read it back.
    #
    # EXCEPT WHERE THE CALLER ASKED FOR BLENDER'S RESOLUTION. With allowCollision on 4.2 and later
    # the renamed object becomes '<name>.001' - the very outcome the refusal above offers the flag
    # for - and this raised "resolved rather than refused" about it, AFTER the rename had happened.
    # So on every current build the flag could only fail, and the failed call had still renamed the
    # object. Measured on 4.4.0: MifRA -> MifRB came back refused, with MifRB.001 in the scene.
    resolved = obj.name != wanted
    if resolved and not (allow and clash is not None):
        # PUT BACK BEFORE THE REFUSAL: this raised after obj.name was assigned, so a refused call left the object
        # renamed (found 2026-10-25). Nothing can predict every resolution beforehand - a name a linked library holds,
        # a length limit - so the rename is undone instead, the incumbent's name too if it moved.
        stored = obj.name
        obj.name = was
        if clash is not None and clash.name != other_before:
            clash.name = other_before
        raise MifOpError("asked for '%s' and Blender stored '%s' - the name was resolved rather "
                         "than refused. The object is back to '%s'. NOTHING was changed."
                         % (wanted, stored, obj.name))
    data_name = None
    if rename_data and getattr(obj, "data", None) is not None:
        try:
            # THE NAME THE OBJECT ENDED UP WITH - the one asked for, unless Blender resolved it.
            obj.data.name = obj.name
            data_name = obj.data.name
        except (AttributeError, TypeError):
            data_name = getattr(obj.data, "name", None)

    stolen = (clash is not None and clash.name != other_before)
    return {
        "ok": True,
        "object": obj.name,
        "renamedFrom": was,
        "changed": True,
        # ONLY WITH allowCollision: the name Blender gave this object instead of the one asked for.
        "resolvedTo": obj.name if resolved else None,
        "dataName": data_name if rename_data else getattr(obj.data, "name", None),
        "dataRenamed": bool(rename_data and data_name == obj.name),
        # ONLY REACHABLE WITH allowCollision. Reported because on 3.6 this is how an untouched
        # object ends up renamed, and a caller who opted in should still be told it happened.
        "otherObjectRenamed": ({"was": other_before, "now": clash.name} if stolen else None),
        "note": ("pointers survive a rename - modifier targets, constraint targets and driver "
                 "variables all still point at this object. What breaks is anything holding the "
                 "old NAME as a string, such as an exporter convention like UCX_%s_00." % was),
    }


def _color_attr_name(mesh, which):
    """The ACTIVE or the RENDER colour attribute's name, or None - one reader for set_vertex_color's reply and
    mesh_stats. render_color_index is an index into color_attributes; -1 or past the end means none."""
    ca = getattr(mesh, "color_attributes", None)
    if ca is None:
        return None
    try:
        if which == "active":
            return getattr(getattr(ca, "active_color", None), "name", None)
        idx = ca.render_color_index
        return ca[idx].name if 0 <= idx < len(ca) else None
    except (AttributeError, TypeError, IndexError):
        return None


def op_set_vertex_color(params):
    """Write a colour attribute - what games use for masks, wear and blend weights.

    mesh.vertex_colors is the legacy path; color_attributes is the one that exists on every build
    this addon supports, so there is no branch here. BYTE_COLOR on the CORNER domain is the default
    because it is what Blender's own legacy call produced and what survives an FBX round trip.

    THE NAME CAN SILENTLY FAIL. color_attributes.new() with an over-long name TRUNCATES to 63
    characters on 3.6 and RETURNS None on 4.2, 4.4 and 5.0 - measured, with no exception on any of
    them. The next line taking .name off None is an AttributeError from the middle of an op, so the
    length is checked first and the None is guarded anyway.

    params:
      object (str)          required, must be a MESH
      name (str)            attribute name. Default 'Col'.
      color [r,g,b] or [r,g,b,a]   linear float. Required unless the attribute already exists.
      domain (str)          CORNER (per face-corner, the default) or POINT (per vertex)
      dataType (str)        BYTE_COLOR (default) or FLOAT_COLOR
      faces (list[int])     colour only these faces. CORNER domain only.
      makeActive (bool)     make it the active colour attribute. Default true.
      makeRender (bool)     make it the one the renderer uses. Default true.

    A VALUE PER VERTEX (2026-11-07): one colour on a face list painted a skin blush in steps (eight levels, one per
    call). vertexValues [0-1, ...] gives every vertex its own value - one per vertex in index order, one per vertex in
    vertexIndices, or [[index, value], ...] pairs - and falloff {points: [[x,y,z], ...] world, radius, inner (default 0), type SMOOTH (default) or
    any of transform_selection's falloff curves but RANDOM} makes it from the distance to the nearest point: 1 within inner, 0 at radius. Each
    vertex then takes baseColor (default black) mixed toward color by its value; on CORNER every corner takes its
    vertex's value, so a mask is smooth across faces (faces still limits which corners are written). blend: REPLACE
    (default) writes that mix; OVER mixes from the value already stored toward color instead of baseColor; MAX keeps,
    per channel, the larger of the stored value and the mix. color is required with either. The reply's values gives
    their count, min, max, mean and how many are above 0.
    """
    reject_unknown(params, _VCOL_KEYS, "set_vertex_color")
    src = take(params, "object", required=True, kind=str)
    name = str(take(params, "name", default="Col", kind=str))
    if len(name) > _ID_NAME_LIMIT:
        raise MifOpError("the attribute name is %d characters and Blender's limit is %d. Past it, "
                         "3.6 silently TRUNCATES and 4.2+ silently returns None - neither raises, "
                         "so a longer name fails differently on different builds. NOTHING was "
                         "created." % (len(name), _ID_NAME_LIMIT))
    domain = str(take(params, "domain", default="CORNER", kind=str)).upper()
    if domain not in ("CORNER", "POINT"):
        raise MifOpError("domain must be CORNER (per face-corner) or POINT (per vertex), got '%s'. "
                         "NOTHING was created." % domain)
    dtype = str(take(params, "dataType", default="BYTE_COLOR", kind=str)).upper()
    if dtype not in ("BYTE_COLOR", "FLOAT_COLOR"):
        raise MifOpError("dataType must be BYTE_COLOR or FLOAT_COLOR, got '%s'. NOTHING was "
                         "created." % dtype)
    faces = params.get("faces")
    if faces is not None:
        if domain != "CORNER":
            raise MifOpError("'faces' selects face corners, so it needs domain CORNER - got %s. "
                             "NOTHING was created." % domain)
        if not isinstance(faces, (list, tuple)):
            raise MifOpError("'faces' must be a list of face indices, got %s. NOTHING was created."
                             % type(faces).__name__)
    raw = params.get("color")
    if raw is not None:
        if not isinstance(raw, (list, tuple)) or len(raw) not in (3, 4):
            raise MifOpError("'color' must be [r,g,b] or [r,g,b,a], got %r. NOTHING was created."
                             % (raw,))
        # finite_floats before the tuple: a NaN vertex colour is stored, exported, and only shows
        # up as a black or invisible patch in whatever consumes the mesh.
        colour = tuple(finite_floats(raw, "color")) + ((1.0,) if len(raw) == 3 else ())
    else:
        colour = None
    per_vertex = _vertex_value_spec(params, "NOTHING was created.")
    blend = str(take(params, "blend", default="REPLACE", kind=str)).upper()
    base_raw = params.get("baseColor")
    if per_vertex is None and (params.get("blend") is not None or base_raw is not None):
        raise MifOpError("baseColor and blend go with vertexValues or falloff - one colour on faces has nothing to "
                         "mix. NOTHING was created.")
    if blend not in _VCOL_BLENDS:
        raise MifOpError("blend must be one of %s, got %r. NOTHING was created." % (", ".join(_VCOL_BLENDS), blend))
    base = (0.0, 0.0, 0.0, 1.0)
    if base_raw is not None:
        if not isinstance(base_raw, (list, tuple)) or len(base_raw) not in (3, 4):
            raise MifOpError("'baseColor' must be [r,g,b] or [r,g,b,a], got %r. NOTHING was created." % (base_raw,))
        base = tuple(finite_floats(base_raw, "baseColor")) + ((1.0,) if len(base_raw) == 3 else ())
    if per_vertex is not None and colour is None:
        raise MifOpError("vertexValues and falloff mix toward 'color' - pass it (the colour at value 1). NOTHING was "
                         "created.")

    obj = get_object(src, want_mesh=True)
    mesh = obj.data
    if not mesh.polygons:
        raise MifOpError("'%s' has no faces, so there are no corners to colour." % obj.name)
    vvals = None
    if per_vertex is not None:
        kind, spec = per_vertex
        nv = len(mesh.vertices)
        if kind == "values":
            idx, vals = spec
            if idx is None:
                if len(vals) != nv:
                    raise MifOpError("vertexValues has %d values and '%s' has %d vertices - one per vertex, or pass "
                                     "vertexIndices. NOTHING was created." % (len(vals), obj.name, nv))
                vvals = list(vals)
            else:
                bad = [i for i in idx if not 0 <= i < nv]
                if bad:
                    raise MifOpError("vertex index %d is out of range - '%s' has %d vertices. NOTHING was created."
                                     % (bad[0], obj.name, nv))
                vvals = [None] * nv
                for i, v in zip(idx, vals):
                    vvals[i] = v
        else:
            pts, radius, inner, shape = spec
            co = array.array("f", [0.0]) * (3 * nv)
            mesh.vertices.foreach_get("co", co)
            mw = obj.matrix_world
            vvals = []
            for i in range(nv):
                w = mw @ Vector((co[3 * i], co[3 * i + 1], co[3 * i + 2]))
                vvals.append(_falloff_value(min((w - p).length for p in pts), radius, inner, shape))

    # THE FACE RANGE AND BOTH FLAGS, CHECKED BEFORE THE ATTRIBUTE EXISTS. The range was checked after
    # color_attributes.new ("The attribute WAS created." - true, and the defect), and makeActive / makeRender were
    # read after every write, so {makeActive: "flase"} was refused with the colours already written (measured on
    # 5.0.1, 2026-09-21).
    if colour is not None and faces is not None:
        highest = len(mesh.polygons) - 1
        bad = sorted(i for i in faces if not isinstance(i, int) or i < 0 or i > highest)
        if bad:
            raise MifOpError("face index %s is out of range - '%s' has %d face(s). NOTHING was created."
                             % (bad[0], obj.name, len(mesh.polygons)))
    try:
        make_active = take_bool(params, "makeActive", default=True)
        make_render = take_bool(params, "makeRender", default=True)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was changed." % str(exc).rstrip("."))

    existed = name in mesh.color_attributes
    if not existed:
        if colour is None:
            raise MifOpError("'%s' has no colour attribute named '%s' and no 'color' was given, so "
                             "there is nothing to create it with. NOTHING was created."
                             % (obj.name, name))
        attr = mesh.color_attributes.new(name=name, type=dtype, domain=domain)
        if attr is None:
            raise MifOpError("color_attributes.new() returned None for '%s' - Blender signals "
                             "failure that way rather than raising. NOTHING was created." % name)
    else:
        attr = mesh.color_attributes[name]
        if attr.domain != domain or attr.data_type != dtype:
            raise MifOpError("'%s' already exists as %s/%s and was asked for as %s/%s. Changing "
                             "either would rewrite every value; delete it first if that is what "
                             "you want. NOTHING was changed."
                             % (name, attr.domain, attr.data_type, domain, dtype))

    written = 0
    sample_index = 0
    value_stats = None
    if vvals is not None:
        # EVERY ELEMENT IS READ, MIXED AND WRITTEN IN ONE PASS (foreach_get / foreach_set): an element without a value
        # (a vertex vertexIndices left out, a corner of a face `faces` left out) keeps what it holds.
        n_el = len(attr.data)
        buf = array.array("f", [0.0]) * (4 * n_el)
        attr.data.foreach_get("color", buf)
        if attr.domain == "POINT":
            el_vert = list(range(n_el))
        else:
            el_vert = array.array("i", [0]) * len(mesh.loops)
            mesh.loops.foreach_get("vertex_index", el_vert)
        allowed = None
        if faces is not None:
            allowed = set()
            for fi in faces:
                allowed.update(mesh.polygons[fi].loop_indices)
        first = None
        used = []
        for e in range(n_el):
            if allowed is not None and e not in allowed:
                continue
            t = vvals[el_vert[e]]
            if t is None:
                continue
            for c in range(4):
                old = buf[4 * e + c] if blend != "REPLACE" else base[c]
                mix = (old if blend == "OVER" else base[c]) + (colour[c] - (old if blend == "OVER" else base[c])) * t
                buf[4 * e + c] = max(old, mix) if blend == "MAX" else mix
            written += 1
            if first is None:
                first = e
        attr.data.foreach_set("color", buf)
        sample_index = first or 0
        used = [v for v in vvals if v is not None]
        value_stats = {"count": len(used), "min": round(min(used), 6) if used else None,
                       "max": round(max(used), 6) if used else None,
                       "mean": round(sum(used) / len(used), 6) if used else None,
                       "aboveZero": sum(1 for v in used if v > 0.0),
                       "from": per_vertex[0], "blend": blend, "baseColor": rnd(list(base))}
    elif colour is not None:
        if faces is not None:
            for fi in faces:
                for li in mesh.polygons[fi].loop_indices:
                    attr.data[li].color = colour
                    written += 1
            # THE READ-BACK READS AN ELEMENT THIS CALL WROTE: element 0 can be a corner of a face that was not asked
            # for, so storedColor, quantised and the note described the wrong value (found 2026-10-25).
            if faces:
                sample_index = mesh.polygons[faces[0]].loop_start
        else:
            for entry in attr.data:
                entry.color = colour
            written = len(attr.data)

    if make_active:
        mesh.color_attributes.active_color = attr
    if make_render:
        try:
            mesh.color_attributes.render_color_index = mesh.color_attributes.find(name)
        except (AttributeError, TypeError):
            pass
    mesh.update()

    # READ BACK FROM THE MESH, not from the value that was written. A colour written to a
    # BYTE_COLOR attribute is quantised to 8 bits per channel, so what comes back is not what went
    # in, and reporting the request would hide that entirely.
    sample = list(attr.data[sample_index].color) if len(attr.data) else None

    # THE RENDER SLOT, which nothing reported. activeColor answered makeActive; makeRender had no
    # field at all, and its write above swallows AttributeError/TypeError - so "it did nothing" was
    # possible and invisible. Found by audit_blender_inert_params and measured on 3.6.23, 4.2.17,
    # 4.4.0, 5.0.1 and 5.2.1: the flag works. On a mesh with no colour attributes the FIRST one
    # becomes active and render whatever the flags say, which is why the probe read it as inert.
    render_name = _color_attr_name(mesh, "render")
    return {
        "ok": True,
        "object": obj.name,
        "attribute": attr.name,
        "created": not existed,
        "domain": attr.domain,
        "dataType": attr.data_type,
        "elements": len(attr.data),
        "elementsWritten": written,
        "requestedColor": rnd(list(colour)) if colour else None,
        "values": value_stats,
        "storedColor": rnd(sample) if sample else None,
        "storedColorElement": sample_index if sample else None,
        "quantised": bool(colour and sample and vvals is None and
                          any(abs(a - b) > 1e-4 for a, b in zip(colour, sample))),
        "activeColor": _color_attr_name(mesh, "active"),
        "renderColor": render_name,
        "note": ("BYTE_COLOR stores 8 bits per channel, so storedColor differs from what was asked "
                 "for. That is the format doing its job, not a failure - use FLOAT_COLOR if the "
                 "exact value matters more than the file size.")
        if (colour and sample and vvals is None and any(abs(a - b) > 1e-4 for a, b in zip(colour, sample)))
        else None,
    }


_ATTRIBUTE_ROWS_CAP = 64

# HOW ONE ELEMENT OF EACH ATTRIBUTE TYPE IS READ: the foreach_get field, how many numbers it holds, and
# whether they are integers. Read off Blender's own attribute classes (FloatAttributeValue.value,
# FloatVectorAttributeValue.vector, Float2AttributeValue.vector, ByteColorAttributeValue.color - the LINEAR
# value, which is what a shader reads), not guessed. A type not listed is reported as unmeasured rather
# than read wrongly.
_ATTR_LAYOUT = {
    "FLOAT": ("value", 1, False), "INT": ("value", 1, True), "INT8": ("value", 1, True),
    "BOOLEAN": ("value", 1, True), "FLOAT_VECTOR": ("vector", 3, False), "FLOAT2": ("vector", 2, False),
    "INT32_2D": ("value", 2, True), "FLOAT_COLOR": ("color", 4, False), "BYTE_COLOR": ("color", 4, False),
    "QUATERNION": ("value", 4, False),
}


def _attribute_stats(mesh, names):
    """{name: {domain, dataType, count, min, max, mean}} for the named attributes - their VALUES, not just names.

    WHY: A NAME THAT EXISTS CAN STILL HOLD NOTHING USEFUL. Wiring an Ocean's foam through a ColorRamp with
    stops at 0.35 and 0.8 rendered solid black on 5.0.1 (2026-09-20) while the foam attribute was present and
    named - the values under that particular sea never reached 0.35, and nothing could say so short of a
    render and a guess. Min, max and mean per component answer "where do the ramp stops belong" and "is there
    any foam at all" in one call. Colours are read LINEAR, as a shader sees them; a boolean's mean is the
    fraction that is true.

    An unknown name is REFUSED naming what exists: a stat block for a typo would be empty, and an empty
    answer here reads exactly like an attribute that holds zeros.
    """
    present = {a.name: a for a in mesh.attributes if not a.name.startswith(".")}
    if names is True:
        names = sorted(present)
    missing = [n for n in names if n not in present]
    if missing:
        raise MifOpError("no attribute named %s on this mesh. It has: %s. NOTHING was read."
                         % (", ".join("'%s'" % n for n in missing), ", ".join(sorted(present)) or "(none)"))
    out = {}
    for name in names:
        attr = present[name]
        n = len(attr.data)
        row = {"domain": attr.domain, "dataType": attr.data_type, "count": n}
        layout = _ATTR_LAYOUT.get(attr.data_type)
        if layout is None:
            row["note"] = "a %s attribute has no numeric range to report" % attr.data_type
            out[name] = row
            continue
        field, width, is_int = layout
        if n == 0:
            row["note"] = "the attribute holds no elements"
            out[name] = row
            continue
        buf = array.array("i" if is_int else "f", [0]) * (n * width)
        try:
            attr.data.foreach_get(field, buf)
        except (TypeError, RuntimeError, AttributeError):
            # A BOOLEAN reads into a bool-sized buffer on some builds; a plain list takes anything.
            vals = [0] * (n * width)
            attr.data.foreach_get(field, vals)
            buf = vals
        comps = [buf[i::width] for i in range(width)]
        mins = [min(c) for c in comps]
        maxs = [max(c) for c in comps]
        means = [sum(c) / float(n) for c in comps]

        def _one(v):
            return round(float(v), 6)
        if width == 1:
            row.update({"min": _one(mins[0]), "max": _one(maxs[0]), "mean": _one(means[0])})
        else:
            row.update({"min": [_one(v) for v in mins], "max": [_one(v) for v in maxs],
                        "mean": [_one(v) for v in means]})
        out[name] = row
    return out

# OBJECT TYPES WHOSE GEOMETRY EXISTS ONLY ONCE EVALUATED - they hold splines, characters or surface
# patches, and a mesh appears when Blender evaluates bevel, extrude, resolution and the modifier stack.
_EVALUATED_ONLY_TYPES = ("CURVE", "FONT", "SURFACE")


def _box(points):
    """{min, max, size, center} of a point list, or None for no points."""
    if not points:
        return None
    lo = [min(p[i] for p in points) for i in range(3)]
    hi = [max(p[i] for p in points) for i in range(3)]
    return {"min": rnd(lo), "max": rnd(hi),
            "size": rnd([hi[i] - lo[i] for i in range(3)]),
            "center": rnd([(hi[i] + lo[i]) / 2.0 for i in range(3)])}


def _stats_of_evaluated_only(obj, want_eval, stat_names=None):
    """mesh_stats for a CURVE, TEXT or SURFACE object: what it EVALUATES to, since it has no mesh of its own.

    THE READ HALF create_curve NEVER HAD. mesh_stats refused these outright ("is a CURVE, not a MESH"), so
    nothing reported what a curve BUILDS: create_curve's bevelResolution and resolution change the geometry
    and were written to curve data no op read back - the second seat recorded both as "writable, unreadable"
    (incoming/second-seat/SECOND_SEAT.md, adv 1) after finding mesh_stats closed and audit_placement blind to
    a scene of curves. And a curve carrying a Geometry Nodes group - a stem with leaves instanced along it,
    measured on 5.0.1, 2026-09-20, drawing 40 instances - could not be measured at all.

    The evaluated mesh is what renders and what an FBX export writes, so it is the honest answer. `base` is
    null rather than zero: a curve's splines are not vertices, and 0 would claim they were counted.
    """
    data = obj.data
    out = {
        "ok": True,
        "object": obj.name,
        "objectType": obj.type,
        "base": None,
        "baseNote": ("a %s holds %s, not vertices - its mesh exists only once evaluated (bevel, extrude, "
                     "resolution and the modifier stack), so `evaluated` is the whole answer."
                     % (obj.type, "characters" if obj.type == "FONT" else "splines")),
        "splineCount": len(getattr(data, "splines", ()) or ()),
        "materialSlots": [s.material.name if s.material else None for s in obj.material_slots],
        "modifiers": [m.name for m in obj.modifiers],
    }
    if not want_eval:
        if stat_names:
            raise MifOpError("attributeStats reads the evaluated mesh, and a %s has no other - it cannot go with "
                             "evaluated:false. NOTHING was read." % obj.type)
        out["evaluated"] = None
        out["evaluatedNote"] = "evaluated:false - and a %s has no base mesh, so nothing was counted." % obj.type
        return out
    depsgraph = bpy.context.evaluated_depsgraph_get()
    source = obj.evaluated_get(depsgraph)
    try:
        emesh = source.to_mesh()
    except RuntimeError as exc:
        out["evaluated"] = None
        out["evaluatedNote"] = "Blender could not make a mesh of '%s': %s" % (obj.name, exc)
        return out
    try:
        if emesh is None:
            out["evaluated"] = None
            out["evaluatedNote"] = "'%s' evaluates to no mesh at all." % obj.name
            return out
        emesh.calc_loop_triangles()
        epts = [source.matrix_world @ v.co for v in emesh.vertices]
        eval_attrs, eval_attrs_cut = _attribute_rows(emesh)
        out["evaluated"] = {
            "vertices": len(emesh.vertices), "edges": len(emesh.edges),
            "faces": len(emesh.polygons), "triangles": len(emesh.loop_triangles),
            "bboxWorld": _box(epts),
            "attributes": eval_attrs,
            **({"attributesTruncated": eval_attrs_cut} if eval_attrs_cut else {}),
        }
        if stat_names:
            out["attributeStats"] = _attribute_stats(emesh, stat_names)
            out["attributeStatsFrom"] = "evaluated"
        out["instancesNote"] = instances_note(obj, "the evaluated counts and box", depsgraph)
    finally:
        source.to_mesh_clear()
    return out


def _attribute_rows(mesh):
    """Every NAMED attribute a mesh carries - [{name, domain, dataType}] - and how many were left out.

    WHY: A MATERIAL READS AN ATTRIBUTE BY NAME, and the attributes that matter most exist only on the
    EVALUATED mesh. An Ocean modifier with foam on writes its foam as a BYTE_COLOR corner attribute - named
    "Col" until foam_layer_name is set, "foam" after (measured on 5.0.1, 2026-09-20) - and a Store Named
    Attribute node writes whatever it was given. mesh_stats reported the BASE mesh's colorAttributes and
    nothing for the evaluated one, so an Attribute node wired to "foam" could not be checked against
    anything: the name was invisible through every typed op.

    Names starting with "." are Blender's own internal layers (.edge_verts, .corner_vert, .select_vert) and
    are left out - they are not something a material or a caller reads. `position` stays: it is a real,
    readable attribute. Capped, because a scanned or generated mesh can carry a great many.
    """
    rows = [{"name": a.name, "domain": a.domain, "dataType": a.data_type}
            for a in mesh.attributes if not a.name.startswith(".")]
    return rows[:_ATTRIBUTE_ROWS_CAP], max(0, len(rows) - _ATTRIBUTE_ROWS_CAP)


# ---------------------------------------------------------------------------
# mesh_stats at a frame, and the simulation caches its evaluated read depends on
# ---------------------------------------------------------------------------

# THE MODIFIERS WHOSE EVALUATED MESH IS A POINT CACHE'S FRAME. Measured on 5.2.1 (2026-10-03): an UNBAKED cloth jumped
# from frame 1 to 20 and then to 70 reads the SAME evaluated box at both - a simulation is stepped forward one frame at a
# time and a jump reads what the last step left - which is a short film's "same bbox at frames 20 and 70". And a
# BAKED cloth whose settings changed after the bake is is_outdated and still plays the OLD bake.
_SIM_CACHE_TYPES = ("CLOTH", "SOFT_BODY")


def _sim_cache_rows(obj, frame):
    """One row per simulation cache the object's evaluated state comes from at `frame`: its cloth and soft body
    modifiers' point caches, and the rigid body world's when the object is a rigid body. `readable` is whether the
    evaluated state at that frame is the simulation's: at or before the cache's first frame (the rest state), or inside
    a bake that is not outdated."""
    rows = []
    # THE FLAGS ARE SET WHEN THE DEPSGRAPH EVALUATES A CHANGE, so pending updates are evaluated first (at the scene's own
    # frame; nothing moves). AND ONLY THEN PAST THE CACHE'S FIRST FRAME - measured on 5.2.1: a cloth quality change made
    # with the scene at frame 40 marks the bake is_outdated; the same change at frame 1 marks nothing, and the old bake
    # still plays. So isOutdated is Blender's flag as it stands, and a bake changed at its first frame is not caught.
    bpy.context.evaluated_depsgraph_get()
    caches = [(m.name, m.type, m.point_cache) for m in obj.modifiers
              if m.type in _SIM_CACHE_TYPES and m.show_viewport and getattr(m, "point_cache", None) is not None]
    rbw = getattr(bpy.context.scene, "rigidbody_world", None)
    if getattr(obj, "rigid_body", None) is not None and rbw is not None and getattr(rbw, "point_cache", None):
        caches.append(("RigidBodyWorld", "RIGID_BODY", rbw.point_cache))
    for name, kind, pc in caches:
        baked, outdated = bool(pc.is_baked), bool(getattr(pc, "is_outdated", False))
        start, end = int(pc.frame_start), int(pc.frame_end)
        if frame <= start:
            why = None
        elif not baked:
            why = ("not baked (%s) - a simulation is stepped forward one frame at a time, so a jump to frame %d reads "
                   "what the last step left, not frame %d" % (pc.info or "nothing cached", frame, frame))
        elif outdated:
            why = ("baked but OUTDATED - a setting changed after the bake, and the cache still plays the old result")
        elif frame > end:
            why = "baked for frames %d-%d, and frame %d is past its end" % (start, end, frame)
        else:
            why = None
        rows.append({"source": name, "type": kind, "isBaked": baked, "isOutdated": outdated,
                     "cacheFrameStart": start, "cacheFrameEnd": end, "readable": why is None, "why": why})
    return rows


def op_mesh_stats(params):
    """Counts and a world bounding box - computed from vertices, never from bound_box - at the scene's frame or at
    `frame`.

    AT A FRAME (a short film, 2026-10-02: a baked cloth read the same box at frames 20 and 70). `frame` moves the
    scene there with frame_set, measures, and puts the frame back (frameRestored is read, not assumed). When the
    object's evaluated state comes from a simulation cache - a cloth or soft body modifier, or the rigid body world -
    the reply's `simulation` rows say whether that cache can answer for the frame, and a `frame` it cannot answer for
    is REFUSED before the frame moves: an unbaked cache (a jump reads the last step, not the frame), a bake made before
    a setting changed (outdated: it plays the old result - Blender's own flag, which it sets only for a change made past
    the cache's first frame), or a frame past the bake's end. bake_physics first. A read
    at the scene's own frame is not refused - a caller who stepped there has a real state - but carries
    simulationWarning when the cache cannot vouch for it.

    params:
      object (str)          required: a MESH, or a CURVE / FONT (text) / SURFACE object
      evaluated (bool)      include the modifier-result counts and box. Default true.
      attributeStats (list) attribute names to read min/max/mean of (true: every named attribute) - see _mesh_stats
      frame (int)           measure at this frame, the scene's frame put back after; implies the evaluated read
      components (bool)     counts per evaluated component - mesh, point cloud, curves, instances (see _GEO_COMPONENTS)
      attributeValues       {component, domain, name, limit, start}: one attribute's values off any component (5.0+
                            beyond MESH); start (default 0) pages past limit, which stops at 10,000
      instances             true or {start, limit (default 1000, at most 10,000), matrix}: every evaluated instance the
                            object draws (Geometry Nodes, particles, a collection instance's Empty) - its source and
                            world location / rotation / scale (matrix: the 4x4 too), and a count per source. An Empty
                            answers its instances only. See _instance_rows.
    """
    reject_unknown(params, _STATS_KEYS, "mesh_stats")
    frame = take_int(params, "frame")
    comps, av = _parse_geometry_request(params)
    inst = _parse_instances(params)
    rest = {k: v for k, v in params.items() if k not in ("frame", "components", "attributeValues", "allowSimulation",
                                                          "instances")}
    sc = bpy.context.scene
    # AN INSTANCER THAT IS NOT A MESH - a collection instance's Empty - has instances and nothing else to count.
    if inst is not None:
        probe = get_object(take(params, "object", "name", required=True, kind=str))
        if probe.type not in ("MESH",) + tuple(_EVALUATED_ONLY_TYPES):
            if frame is not None or comps or av is not None or params.get("attributeStats"):
                raise MifOpError("'%s' is a %s: only instances can be read off it. NOTHING was read."
                                 % (probe.name, probe.type))
            return {"ok": True, "object": probe.name, "type": probe.type, "instances": _instance_rows(probe, inst)}
    if frame is None:
        simulation_guard(params, [], "mesh_stats")       # reads allowSimulation's type; no frame moves
        out = _mesh_stats(rest)
        probe = bpy.data.objects.get(out.get("object") or "")
        if comps or av is not None:
            out.update(_geometry_read(probe, comps, av))
        if inst is not None:
            out["instances"] = _instance_rows(probe, inst)
        rows = _sim_cache_rows(probe, sc.frame_current) if probe is not None else []
        if rows:
            out["simulation"] = rows
            stale = [r for r in rows if not r["readable"]]
            if stale:
                out["simulationWarning"] = (
                    "the evaluated mesh at frame %d may not be the simulation's state: %s. bake_physics, then read "
                    "with frame." % (sc.frame_current, "; ".join("%s (%s): %s" % (r["source"], r["type"], r["why"])
                                                               for r in stale)))
        return out
    if not take_bool(params, "evaluated", default=True):
        raise MifOpError("frame moves the scene to read the EVALUATED mesh, and evaluated:false turns that read off. "
                         "Drop one. NOTHING was read.")
    obj = get_object(take(params, "object", "name", required=True, kind=str))
    rows = _sim_cache_rows(obj, frame)
    stale = [r for r in rows if not r["readable"]]
    if stale:
        raise MifOpError("mesh_stats cannot read '%s' at frame %d - %s. Run bake_physics (again after any change to "
                         "the simulation's settings) and read again. NOTHING was read, and the scene frame did not "
                         "move." % (obj.name, frame, "; ".join("%s (%s) is %s" % (r["source"], r["type"], r["why"])
                                                               for r in stale)))
    # THE OTHER OBJECTS' SIMULATIONS STEP TOO: the check above is this object's own cache; a frame change evaluates
    # every live simulation in the view layer (ops_common.simulation_guard).
    sim_reply = simulation_guard(params, [frame], "mesh_stats frame", tail="NOTHING was read, and the frame did not "
                                 "move.")
    started_on = sc.frame_current
    try:
        sc.frame_set(frame)
        out = _mesh_stats(rest)
        if comps or av is not None:
            out.update(_geometry_read(obj, comps, av))
        if inst is not None:
            out["instances"] = _instance_rows(obj, inst)
    finally:
        sc.frame_set(started_on)
    out["frame"] = frame
    out["frameRestored"] = sc.frame_current == started_on
    out["startedOnFrame"] = started_on
    if rows:
        out["simulation"] = rows
    out.update(sim_reply)
    return out


def _mesh_stats(params):
    """Counts and a world bounding box - computed from vertices, never from bound_box.

    obj.bound_box IS CACHED AND STALE. Measured on all four builds: move a vertex to z=50 and
    bound_box[6] still reads the old extent. mesh.update() does NOT refresh it - only
    bpy.context.view_layer.update() does. So a box read straight off bound_box after any edit is
    quietly wrong, which is the worst kind of wrong for a number people position things against.

    This computes the box from the vertices directly, so there is nothing to be stale.

    THE EVALUATED COUNTS ARE THE ONES THAT MATTER FOR EXPORT. A mesh with a Subsurf modifier has a
    base count and a rendered count that differ by an order of magnitude, and both are reported
    rather than one being chosen for the caller.

    A CURVE, TEXT OR SURFACE IS MEASURED TOO, by its evaluated mesh, since 2026-09-20 - see
    _stats_of_evaluated_only.

    params:
      object (str)          required: a MESH, or a CURVE / FONT (text) / SURFACE object
      evaluated (bool)      include the modifier-result counts and box. Default true.
      attributeStats (list) attribute NAMES to read the values of - min, max and mean per component (true: every
                            named attribute, false: none),
                            off the evaluated mesh when there is one, else the base mesh
                            (attributeStatsFrom says which). See _attribute_stats.
    """
    reject_unknown(params, _STATS_KEYS, "mesh_stats")
    obj_name = take(params, "object", "name", required=True, kind=str)
    want_eval = take_bool(params, "evaluated", default=True)
    stat_names = params.get("attributeStats")
    # TRUE READS EVERY NAMED ATTRIBUTE, false none. The name reads as a switch, and benchmark round ten's agents passed
    # true 12 times (A5, A8) before resending ['Col']; the attribute list it would read is the one `attributes` names.
    if stat_names is False:
        stat_names = None
    if stat_names is not None and stat_names is not True:
        if (not isinstance(stat_names, (list, tuple)) or not stat_names
                or not all(isinstance(n, str) and n for n in stat_names)):
            raise MifOpError("attributeStats is a non-empty list of attribute names, e.g. ['Col'], or true for every "
                             "named attribute - got %r. NOTHING was read." % (stat_names,))
        stat_names = list(dict.fromkeys(stat_names))
    probe = get_object(obj_name)
    if probe.type in _EVALUATED_ONLY_TYPES:
        return _stats_of_evaluated_only(probe, want_eval, stat_names)
    obj = get_object(obj_name, want_mesh=True)
    mesh = obj.data

    mw = obj.matrix_world
    local_pts = [v.co for v in mesh.vertices]
    world_pts = [mw @ v.co for v in mesh.vertices]
    mesh.calc_loop_triangles()
    base_attrs, base_attrs_cut = _attribute_rows(mesh)
    out = {
        "ok": True,
        "object": obj.name,
        "base": {
            "vertices": len(mesh.vertices),
            "edges": len(mesh.edges),
            "faces": len(mesh.polygons),
            "triangles": len(mesh.loop_triangles),
            "loops": len(mesh.loops),
        },
        "bboxLocal": _box(local_pts),
        "bboxWorld": _box(world_pts),
        "uvLayers": [uv.name for uv in mesh.uv_layers],
        "colorAttributes": [a.name for a in mesh.color_attributes],
        # WHICH ONE IS ACTIVE AND WHICH ONE RENDERS - what set_vertex_color's makeActive and makeRender write, and until
        # 2026-09-21 only its own reply said (audit_write_only_state). They can differ: an agent painting a mask into a
        # second layer leaves the base colours rendering, which is right, and was invisible here.
        "activeColorAttribute": _color_attr_name(mesh, "active"),
        "renderColorAttribute": _color_attr_name(mesh, "render"),
        # THE BASE MESH'S attributes. The EVALUATED mesh's are under evaluated.attributes, and they are the
        # ones a modifier or a node group added - see _attribute_rows.
        "attributes": base_attrs,
        **({"attributesTruncated": base_attrs_cut} if base_attrs_cut else {}),
        "materialSlots": [s.material.name if s.material else None for s in obj.material_slots],
        "modifiers": [m.name for m in obj.modifiers],
        "bboxNote": ("computed from vertices, NOT from obj.bound_box - that is cached and does not "
                     "refresh after a vertex edit until view_layer.update() runs, so it reports the "
                     "previous extent with nothing to say so."),
    }
    # SHAPE KEYS DEFORM WITHOUT A MODIFIER. A cube keyed 1 m up at value 1 read its rest box here, under "no modifiers,
    # so the evaluated mesh is the base mesh" (2026-10-04, measured on 5.2.1: vertex_info evaluated put the vertex at
    # z 0.5, this said -0.5). Any mesh with keys beyond its basis is evaluated like a modifier stack.
    keyed = bool(mesh.shape_keys and len(mesh.shape_keys.key_blocks) > 1)
    if want_eval and (obj.modifiers or keyed):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        source = obj.evaluated_get(depsgraph)
        # evaluated_get SILENTLY RETURNS THE UNEVALUATED OBJECT when the object is not in the
        # active view layer's depsgraph, so the identity is checked rather than assumed.
        if source.data is mesh:
            out["evaluated"] = None
            out["evaluatedNote"] = ("evaluated_get returned the BASE mesh - the object is not in "
                                    "the active view layer's depsgraph, so no modifier result "
                                    "exists to measure. This is silent in Blender.")
        else:
            emesh = source.to_mesh()
            try:
                emesh.calc_loop_triangles()
                epts = [source.matrix_world @ v.co for v in emesh.vertices]
                eval_attrs, eval_attrs_cut = _attribute_rows(emesh)
                base_names = {a["name"] for a in base_attrs}
                out["evaluated"] = {
                    "vertices": len(emesh.vertices), "edges": len(emesh.edges),
                    "faces": len(emesh.polygons), "triangles": len(emesh.loop_triangles),
                    "bboxWorld": _box(epts),
                    "attributes": eval_attrs,
                    **({"attributesTruncated": eval_attrs_cut} if eval_attrs_cut else {}),
                    # WHAT THE STACK ADDED, which is the question a material wiring an Attribute node is asking.
                    "attributesAddedByModifiers": sorted(a["name"] for a in eval_attrs
                                                         if a["name"] not in base_names),
                }
                if stat_names:
                    out["attributeStats"] = _attribute_stats(emesh, stat_names)
                    out["attributeStatsFrom"] = "evaluated"
                out["instancesNote"] = instances_note(obj, "the evaluated counts and box", depsgraph)
            finally:
                source.to_mesh_clear()
    elif want_eval:
        out["evaluated"] = None
        out["evaluatedNote"] = "no modifiers and no shape keys, so the evaluated mesh is the base mesh."
    if stat_names and "attributeStats" not in out:
        out["attributeStats"] = _attribute_stats(mesh, stat_names)
        out["attributeStatsFrom"] = "base"
    return out


_SETUV_KEYS = {"object", "name", "layer", "active", "rename", "remove"}


def _uv_layer_map(mesh):
    """Layer name -> index. The INDEX is what an engine reads, so it is the thing worth watching."""
    return dict((layer.name, i) for i, layer in enumerate(mesh.uv_layers))


def op_set_uv_layer(params):
    """Choose the ACTIVE UV layer, rename one, or remove one - none of which was possible.

    THE ACTIVE LAYER IS THE ONE EVERYTHING WRITES TO. uv_info reports activeLayer and the only way
    to change it was uv_unwrap, which sets it as a side effect of unwrapping - so selecting a layer
    meant re-unwrapping it and destroying the UVs it held. That is the file's own trap seen from the
    other side: uv_unwrap carries a comment about active_index staying 0 after creating a layer,
    because every UV operator and every bake writes to the ACTIVE layer, and missing that line makes
    a lightmap pass silently repack the base colour UVs.

    REMOVING A LAYER SHIFTS EVERY LATER LAYER'S INDEX. Measured on 3.6.23 and 5.0.1: with UVMap, A,
    B, C, removing A moves B from 2 to 1 and C from 3 to 2. Unreal's lightmap coordinate index is an
    INDEX, not a name, so removing an earlier layer silently repoints what the engine reads at a
    mesh it has already imported. The layers that moved are named in the response.

    A MESH IS LIMITED TO 8 UV LAYERS, which is why removing junk ones matters at all.

    params:
      object / name (str)   required, must be a MESH
      layer (str)           which layer. Required.
      active (bool)         make it the active layer
      rename (str)          a new name; Blender suffixes a clash rather than refusing
      remove (bool)         delete it - excludes active and rename
    """
    reject_unknown(params, _SETUV_KEYS, "set_uv_layer")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    mesh = obj.data
    want = take(params, "layer", required=True, kind=str)

    if not mesh.uv_layers:
        raise MifOpError("'%s' has NO UV layers at all, so there is none to change - uv_unwrap "
                         "makes one. NOTHING was changed." % obj.name)
    layer = mesh.uv_layers.get(str(want))
    if layer is None:
        raise MifOpError("'%s' has no UV layer named '%s'. It has: %s. NOTHING was changed."
                         % (obj.name, want, ", ".join(l.name for l in mesh.uv_layers)))

    remove = take_bool(params, "remove", default=False)
    rename = take(params, "rename", default=None, kind=str)
    make_active = params.get("active")
    # READ HERE, BEFORE THE RENAME (2026-09-21): it was read after the layer had been renamed, so {rename, active:
    # "flase"} renamed the layer and then refused.
    set_active = take_bool(params, "active", default=True) if make_active is not None else False
    if remove and (rename is not None or make_active is not None):
        raise MifOpError("remove deletes the layer, so making it active or renaming it in the same "
                         "call is contradictory. NOTHING was changed.")
    if not remove and rename is None and make_active is None:
        raise MifOpError("nothing to do - pass active, rename or remove. NOTHING was changed.")
    # ACTIVE FALSE ALONE IS NOTHING TO DO, and answered ok having changed nothing (found 2026-10-25): a mesh always has
    # one active UV layer, so a layer stops being active only when another one is made active.
    if not remove and rename is None and not set_active:
        raise MifOpError("active false on its own changes nothing - a mesh always has an active UV layer. Make "
                         "another layer active with set_uv_layer {layer: <that one>, active: true}. NOTHING was "
                         "changed.")

    before_map = _uv_layer_map(mesh)
    before_active = mesh.uv_layers.active.name if mesh.uv_layers.active else None

    removed, moved, renamed_to = None, {}, None
    # TRACKED AS AN EVENT, not inferred by comparing NAMES before and after. Renaming the
    # active layer changes that name, so a name comparison called it a switch and the note
    # told a caller the active layer had moved when only its label had.
    switched = False
    if remove:
        if len(mesh.uv_layers) == 1:
            raise MifOpError(
                "'%s' is the only UV layer on '%s'. Removing it leaves the mesh unwrappable - "
                "texturing and lightmap baking both fail until it is unwrapped again. Refused "
                "rather than done quietly. NOTHING was changed." % (want, obj.name))
        removed = layer.name
        mesh.uv_layers.remove(layer)
        after_map = _uv_layer_map(mesh)
        # WHICH LAYERS MOVED, by name, because an engine reads the INDEX.
        moved = dict((n, {"was": before_map[n], "now": after_map[n]})
                     for n in after_map if before_map.get(n) != after_map[n])
    else:
        if rename is not None:
            asked = str(rename)
            if len(asked) > _ID_NAME_LIMIT:
                raise MifOpError("the name is %d characters and Blender truncates at %d, so the "
                                 "layer you get would not be the one you named. NOTHING was "
                                 "changed." % (len(asked), _ID_NAME_LIMIT))
            layer.name = asked
            renamed_to = layer.name
        # Read BEFORE the make_active test - see audit_blender_shortcircuit_take.
        if make_active is not None and set_active:
            mesh.uv_layers.active = layer
            switched = True

    after_map = _uv_layer_map(mesh)
    active_now = mesh.uv_layers.active.name if mesh.uv_layers.active else None
    return {
        "ok": True,
        "object": obj.name,
        "layers": [{"name": n, "index": i} for n, i in sorted(after_map.items(),
                                                              key=lambda kv: kv[1])],
        "layerCount": len(after_map),
        "activeLayer": active_now,
        "activeLayerIndex": after_map.get(active_now) if active_now else None,
        "activeChanged": active_now != before_active,
        "removed": removed,
        "renamedTo": renamed_to,
        "nameWasSuffixed": bool(renamed_to and rename is not None and renamed_to != str(rename)),
        # THE FIELD THAT MATTERS ON A REMOVE. Named layers whose INDEX moved, because an engine
        # importing this mesh reads a lightmap coordinate INDEX and will now read a different layer.
        "indicesShifted": moved or None,
        "note": (("removing '%s' moved %s. Unreal's lightmap coordinate index is an INDEX, not a "
                  "name, so anything already importing this mesh now reads a different layer."
                  % (removed, ", ".join("%s %d->%d" % (n, v["was"], v["now"])
                                        for n, v in sorted(moved.items()))))
                 if moved else
                 ("every UV operator and every bake writes to the ACTIVE layer, which is now '%s'."
                  % active_now if switched else None)),
    }

# ---------------------------------------------------------------------------
# mesh_quality
# ---------------------------------------------------------------------------

# Faces smaller than this in square metres are treated as degenerate. Not zero: a float-exact zero
# almost never occurs, and a face of 1e-12 m2 is a defect that exports and renders as nothing.
_DEGENERATE_FACE_AREA = 1e-9
# UV coordinates are compared against 0-1 with a tolerance, because an unwrap that lands a vertex at
# 1.0000001 is not a defect and reporting it as one is how a check gets ignored.
_UV_BOUND_EPS = 1e-4


# ---------------------------------------------------------------------------
# near-zero UV area and near-zero tangents - what Unreal warns about on import
# ---------------------------------------------------------------------------

# A FACE BELOW THIS UV AREA (in UV units squared) IS NEAR-ZERO: 1e-9 is about 1/240 of one texel of a 2048 texture
# (a texel is 2.4e-7). Stated in every reply, because "near" means nothing without the number.
NEAR_ZERO_UV_AREA = 1e-9
# Unreal's own tolerance: StaticMeshBuild.cpp HasBadNTB tests each vertex's TangentX and TangentY with
# IsNearlyZero(UE_KINDA_SMALL_NUMBER) - every component within 1e-4 - and warns "has some nearly zero tangents ...
# (Tolerance of 1E-4)" (UE 5.8 source).
UE_TANGENT_TOLERANCE = 1e-4


def uv_degeneracy(me, layer_name):
    """Near-zero UV-area faces and the vertex instances Unreal will give nearly zero tangents, on one UV layer.

    THE TANGENTS ARE MIKKTSPACE, computed by Blender's own mesh.calc_tangents on a triangulated scratch copy (Unreal
    triangulates before it computes them, and calc_tangents refuses an ngon). Read with Unreal's tolerance: a loop
    whose tangent or bitangent has every component within 1e-4 is near-zero.

    BUT BLENDER'S MIKKTSPACE NEVER ANSWERS ZERO, and that is measured, not assumed: on 5.2.1 a face whose UVs collapse
    to a line or a point gets the placeholder tangent (1, 0, 0) and bitangent (0, +-1, 0) from calc_tangents, where
    the reference MikkTSpace that Unreal links leaves a group with no UV area at zero - which is the warning seen
    on SM_Pothole_B. So the Unreal answer is mikktspace's own rule, applied here: a vertex instance (one position,
    normal and UV - mikktspace welds on all three) gets a zero tangent when every triangle around it has a UV signed
    area that is zero in float32 (mikktspace's NotZero is |a| > FLT_MIN). blenderNearZero is the calc_tangents read,
    kept beside it so a build whose MikkTSpace does answer zero is still caught.

    A scratch mesh is made and removed; the object's own mesh is not touched.
    """
    out = {"layer": layer_name, "nearZeroUvAreaThreshold": NEAR_ZERO_UV_AREA}
    bm = bmesh.new()
    tmp = None
    try:
        bm.from_mesh(me)
        uv = bm.loops.layers.uv.get(layer_name)
        if uv is None:
            out["notMeasured"] = "no UV layer named '%s'" % layer_name
            return out
        near = []
        for f in bm.faces:
            if _uv_area(f, uv) <= NEAR_ZERO_UV_AREA:
                near.append(f.index)
        out["nearZeroUvAreaFaces"] = len(near)
        out["nearZeroUvAreaFaceIndices"] = near[:50]

        orig = bm.faces.layers.int.new("mif_orig_face")
        for f in bm.faces:
            f[orig] = f.index
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
        tmp = bpy.data.meshes.new("_mif_tangent_probe")
        bm.to_mesh(tmp)
        try:
            tmp.calc_tangents(uvmap=layer_name)
        except RuntimeError as exc:
            out["tangents"] = {"notMeasured": "calc_tangents refused: %s" % exc}
            return out
        n_loops = len(tmp.loops)
        tan = array.array("f", [0.0]) * (n_loops * 3)
        bit = array.array("f", [0.0]) * (n_loops * 3)
        nor = array.array("f", [0.0]) * (n_loops * 3)
        uvs = array.array("f", [0.0]) * (n_loops * 2)
        lv = array.array("i", [0]) * n_loops
        tmp.loops.foreach_get("tangent", tan)
        tmp.loops.foreach_get("bitangent", bit)
        tmp.loops.foreach_get("normal", nor)
        tmp.loops.foreach_get("vertex_index", lv)
        tmp.uv_layers[layer_name].data.foreach_get("uv", uvs)
        pos = array.array("f", [0.0]) * (len(tmp.vertices) * 3)
        tmp.vertices.foreach_get("co", pos)
        src = array.array("i", [0]) * len(tmp.polygons)
        tmp.attributes["mif_orig_face"].data.foreach_get("value", src)

        tol = UE_TANGENT_TOLERANCE
        blender_zero = set()
        for i in range(n_loops):
            t, b = tan[3 * i:3 * i + 3], bit[3 * i:3 * i + 3]
            if (all(abs(c) <= tol or c != c for c in t)) or (all(abs(c) <= tol or c != c for c in b)):
                blender_zero.add(i)

        # mikktspace's rule: a welded vertex instance takes its tangent only from the triangles around it with UV area.
        has_area = {}
        loop_key = [None] * n_loops
        loop_face = [0] * n_loops
        for p in tmp.polygons:
            ls = list(p.loop_indices)
            (u0, v0), (u1, v1), (u2, v2) = [(uvs[2 * k], uvs[2 * k + 1]) for k in ls]
            t21x, t21y, t31x, t31y = _f32(u1 - u0), _f32(v1 - v0), _f32(u2 - u0), _f32(v2 - v0)
            area2 = _f32(_f32(t21x * t31y) - _f32(t21y * t31x))
            alive = abs(area2) > _FLT_MIN
            for k in ls:
                vi = lv[k]
                key = (tuple(pos[3 * vi:3 * vi + 3]), tuple(nor[3 * k:3 * k + 3]), (uvs[2 * k], uvs[2 * k + 1]))
                loop_key[k] = key
                loop_face[k] = src[p.index]
                has_area[key] = has_area.get(key, False) or alive
        zero_keys = {k for k, alive in has_area.items() if not alive}
        bad_loops = [i for i in range(n_loops) if loop_key[i] in zero_keys or i in blender_zero]
        out["tangents"] = {
            "method": "MikkTSpace via mesh.calc_tangents on a triangulated copy, and mikktspace's zero-UV-area rule",
            "tolerance": tol,
            "nearZeroVertexInstances": len(zero_keys | {loop_key[i] for i in blender_zero}),
            "collapsedUvInstances": len(zero_keys),
            "vertices": sorted({lv[i] for i in bad_loops})[:50],
            "faces": sorted({loop_face[i] for i in bad_loops})[:50],
            "blenderNearZero": len(blender_zero),
            "note": ("Unreal computes tangents from UV channel 0 (index 0) and warns 'has some nearly zero tangents ... "
                     "(Tolerance of 1E-4)' for these; a face whose UVs collapse to a line or a point is the usual "
                     "cause. Blender's own calc_tangents substitutes a placeholder axis there instead of zero, so "
                     "blenderNearZero alone would read clean.") if bad_loops else None,
        }
        return out
    finally:
        bm.free()
        if tmp is not None:
            bpy.data.meshes.remove(tmp)


def _uv_area(face, uv_layer):
    """Signed UV area of one face, by the shoelace formula over its loops."""
    total = 0.0
    loops = face.loops
    n = len(loops)
    for i in range(n):
        x1, y1 = loops[i][uv_layer].uv
        x2, y2 = loops[(i + 1) % n][uv_layer].uv
        total += (x1 * y2) - (x2 * y1)
    return abs(total) * 0.5


# ---------------------------------------------------------------------------
# mesh_quality conventions - how a piece is usually built for a game engine, each as a finding with its number
# ---------------------------------------------------------------------------

# Size classes by the longest side, in centimeters.
_CONV_SMALL_CM = 100.0
_CONV_LARGE_CM = 300.0
# The usual range of triangles for a mesh of each size class, and the point past which a count is rare - with the same
# per square meter of its bounding box's surface (small pieces carry far more detail per meter). The top is the 90th
# percentile of the non-Nanite meshes measured, since it is also the default gameReady budget: nearer the middle, it
# failed ordinary pieces on their count alone (a 90 cm cart of 1384 triangles).
_CONV_TRIS = {"small": (14, 2642, 6465), "medium": (94, 7780, 23858), "large": (266, 36014, 180430)}
_CONV_TRIS_PER_M2 = {"small": (36.97, 3471.0), "medium": (19.28, 656.0), "large": (3.23, 542.0)}
# Usual thickness of a wall and of a floor piece, centimeters (the median), for the thickness finding's context.
_CONV_TYPICAL_THICKNESS_CM = {"wall": 12.1, "floor": 6.8}
# A side is on the module when it is within this of a whole multiple, centimeters.
_CONV_MODULE_TOL_CM = 0.1
# Material names a modeling tool gives a slot by default - a slot still named this was never named by anybody.
_CONV_DEFAULT_MATERIAL = re.compile(
    r"^(material(\.\d+)?|mat(\.\d+)?|lambert\d*|blinn\d*|phong\d*|initialshadinggroup|default|defaultmaterial|"
    r"default[-_ ]material|standard(surface)?\d*|\d+ - default|worldgridmaterial|none|no name|untitled)$", re.I)
_CONV_LIGHTMAP_NAME = re.compile(r"light\s*_?map|^lm$|_lm$|^uv_?lm", re.I)
_CONV_HULL_PREFIXES = ("UCX", "UBX", "USP", "UCP")
_CONV_NAME = re.compile(r"^SM_[A-Za-z0-9]+(_[A-Za-z0-9]+)*$")
_CONV_NAME_NUMBERED = re.compile(r"_\d{2}[a-z]?$")


def _conv_axis(origin, lo, hi):
    """(at an edge, at the middle, inside, outside) for one axis of the pivot, with a 1% or 5 mm tolerance."""
    span = hi - lo
    tol = max(0.005, 0.01 * span)
    if span <= 1e-9:
        return {"edge": abs(origin - lo) <= tol, "center": abs(origin - lo) <= tol, "outside": abs(origin - lo) > tol}
    return {"edge": abs(origin - lo) <= tol or abs(origin - hi) <= tol,
            "center": abs(origin - (lo + hi) * 0.5) <= tol,
            "outside": origin < lo - tol or origin > hi + tol}


def _conv_pivot_classes(origin, lo, hi):
    """Z: bottom, center, top, below, above or inside; XY: corner, side, centered, inside or outside."""
    tol = max(0.005, 0.01 * (hi[2] - lo[2]))
    z = origin[2]
    if abs(z - lo[2]) <= tol:
        zc = "bottom"
    elif abs(z - hi[2]) <= tol:
        zc = "top"
    elif abs(z - (lo[2] + hi[2]) * 0.5) <= tol:
        zc = "center"
    elif z < lo[2]:
        zc = "below"
    elif z > hi[2]:
        zc = "above"
    else:
        zc = "inside"
    ax, ay = _conv_axis(origin[0], lo[0], hi[0]), _conv_axis(origin[1], lo[1], hi[1])
    if ax["outside"] or ay["outside"]:
        xyc = "outside"
    elif ax["edge"] and ay["edge"]:
        xyc = "corner"
    elif ax["center"] and ay["center"]:
        xyc = "centered"
    elif ax["edge"] or ay["edge"]:
        xyc = "side"
    else:
        xyc = "inside"
    return zc, xyc


def _conv_kind_of(size):
    """floor, wall or prop from the bounds' size in centimeters - the rule the conventions' kind finding states, and
    the one recipe_game_ready picks a pivot and a hull shape by."""
    length, across = max(size[0], size[1]), min(size[0], size[1])
    if size[0] >= 100.0 and size[1] >= 100.0 and size[2] < 0.1 * min(size[0], size[1]):
        return "floor"
    if length >= 100.0 and size[2] >= 100.0 and across < 0.1 * length:
        return "wall"
    return "prop"


def _conv_size_class(longest_cm):
    """small, medium or large by the longest side in centimeters - the conventions' size class."""
    return "small" if longest_cm < _CONV_SMALL_CM else ("medium" if longest_cm < _CONV_LARGE_CM else "large")


def triangle_budget_for_size(obj):
    """The triangle budget a mesh gets when none is given (1.4, gameReady and bevel_edges finish): the top of the usual
    triangle range for its size class - the same _CONV_TRIS row the conventions' "triangles for the size" finding
    holds the count to, so the default and that finding can never disagree. None for a mesh with no vertices."""
    lo, hi = world_vertex_bounds(obj)
    if lo is None:
        return None
    longest = max((hi[k] - lo[k]) * UU_PER_BU for k in range(3))
    size_class = _conv_size_class(longest)
    usual_lo, usual_hi, rare = _CONV_TRIS[size_class]
    return {"triangleBudget": usual_hi, "from": "sizeClass", "sizeClass": size_class,
            "longestSideCm": round(longest, 2), "usualRange": [usual_lo, usual_hi], "rareAbove": rare,
            "rule": "the top of the usual triangle range for the size class (small under 100 cm, medium 100-300 cm, "
                    "large 300 cm and over) - mesh_quality conventions' triangles for the size"}


def _conventions(obj, kind_arg, module_cm):
    """The conventions block of mesh_quality: every finding carries what was MEASURED, the convention it is held to,
    and `follows` (true, false, or null where the convention does not decide). Nothing here is a defect on its own -
    each is a note with its number, for the piece's author to weigh."""
    lo, hi = world_vertex_bounds(obj)
    if lo is None:
        raise MifOpError("'%s' has no vertices, so it has no size, pivot or shape to hold to a convention. NOTHING "
                         "was measured." % obj.name)
    cm = UU_PER_BU
    size = [(hi[k] - lo[k]) * cm for k in range(3)]
    longest = max(size)
    size_class = _conv_size_class(longest)
    length, across = max(size[0], size[1]), min(size[0], size[1])
    notes = []
    out = {}

    out["sizeClass"] = {
        "measured": {"sizeCm": [round(s, 2) for s in size], "longestSideCm": round(longest, 2)},
        "value": size_class,
        "rule": "by the longest side: small under 100 cm, medium 100-300 cm, large 300 cm and over",
    }

    # ---- the kind of piece ------------------------------------------------------------------------------------
    inferred = _conv_kind_of(size)
    kind = kind_arg or inferred
    out["kind"] = {
        "value": kind,
        "inferred": inferred,
        "given": kind_arg,
        "measured": {"lengthCm": round(length, 2), "acrossCm": round(across, 2), "heightCm": round(size[2], 2)},
        "rule": ("floor: both sides at least 100 cm and under a tenth as thick; wall: at least 100 cm long and tall "
                 "and under a tenth as thick as it is long; anything else is a prop"),
    }
    if kind_arg and kind_arg != inferred:
        notes.append("kind was given as %s; by its bounds it reads as %s" % (kind_arg, inferred))

    # ---- the pivot --------------------------------------------------------------------------------------------
    origin = list(obj.matrix_world.translation)
    zc, xyc = _conv_pivot_classes(origin, lo, hi)
    if kind == "floor":
        convention, follows = "bottom, on a corner - a floor module tiles by whole moves from one corner", \
            zc == "bottom" and xyc == "corner"
    elif kind == "wall":
        convention, follows = ("bottom, on a corner or at the middle of a face - a wall module snaps by its face",
                               zc == "bottom" and xyc in ("corner", "side"))
    elif size_class == "small":
        convention, follows = ("bottom center, or the center - a small prop is placed on a surface or held",
                               xyc == "centered" and zc in ("bottom", "center"))
    else:
        convention, follows = "bottom center - the pivot a placed prop stands on", xyc == "centered" and zc == "bottom"
    out["pivot"] = {
        "measured": {"originBU": rnd(origin), "pivotInBounds": pivot_in_bounds(obj), "z": zc, "xy": xyc},
        "convention": convention,
        "follows": follows,
    }
    if not follows:
        notes.append("the pivot sits %s in Z and %s in XY; a %s usually has it %s (set_origin %s)"
                     % (zc, xyc, kind, convention.split(" - ")[0],
                        {"floor": "mode bottomCorner", "wall": "mode bottomSide or bottomCorner"}.get(kind, "mode bottom")))

    # ---- UV layers: UV0 for the texture, the last one for the lightmap -----------------------------------------
    me = obj.data
    names = [uv.name for uv in me.uv_layers]
    lm_named = [i for i, n in enumerate(names) if _CONV_LIGHTMAP_NAME.search(n)]
    lm_index = lm_named[-1] if lm_named else (len(names) - 1 if len(names) >= 2 else None)
    uv = {"measured": {"count": len(names), "names": names,
                       "lightmapLayer": names[lm_index] if lm_index is not None else None,
                       "lightmapIndex": lm_index,
                       "lightmapFoundBy": ("name" if lm_named else ("position (the last of two or more)"
                                                                     if lm_index is not None else None))},
          "convention": "two layers - UV0 for the texture, the LAST for the lightmap, which Unreal's lightmap "
                        "coordinate index points at by position"}
    if not names:
        uv["follows"] = False
        notes.append("no UV layer at all - neither a texture nor a lightmap can be laid out")
    elif lm_index is None:
        uv["follows"] = None
        uv["note"] = ("one layer: Unreal's importer can generate the lightmap from UV0 into the next channel "
                      "(Generate Lightmap UVs), which is the common way. A lightmap made here goes in a second layer "
                      "(uv_unwrap method LIGHTMAP).")
    else:
        row = (op_uv_info({"object": obj.name, "layer": names[lm_index]}).get("layers") or [{}])[0]
        is_last = lm_index == len(names) - 1
        uv["measured"]["lightmapIsLast"] = is_last
        uv["measured"]["lightmapReady"] = row.get("lightmapReady")
        if row.get("lightmapReadyReason"):
            uv["measured"]["lightmapReadyReason"] = row.get("lightmapReadyReason")
        uv["follows"] = bool(is_last and row.get("lightmapReady"))
        if not is_last:
            notes.append("the lightmap layer '%s' is channel %d of %d, not the last" % (names[lm_index], lm_index,
                                                                                       len(names)))
        if not row.get("lightmapReady"):
            notes.append("the lightmap layer '%s' is not lightmap-ready: %s"
                         % (names[lm_index], row.get("lightmapReadyReason") or "see uv_info"))
    out["uvLayers"] = uv

    # ---- sides on the module ----------------------------------------------------------------------------------
    if module_cm is not None:
        if kind == "wall":
            sides = {"length": length, "height": size[2]}
        elif kind == "floor":
            sides = {"x": size[0], "y": size[1]}
        else:
            sides = {"x": size[0], "y": size[1], "z": size[2]}
        rows = {}
        on_all = True
        for label, value in sides.items():
            steps = round(value / module_cm)
            off = abs(value - steps * module_cm)
            on = steps >= 1 and off <= _CONV_MODULE_TOL_CM
            on_all = on_all and on
            rows[label] = {"cm": round(value, 3), "modules": round(value / module_cm, 4), "offModuleCm": round(off, 3),
                           "onModule": on}
            if not on:
                notes.append("the %s is %.2f cm, %s" % (label, value, ("shorter than one %.0f cm module (a filler)"
                                                                       % module_cm) if steps < 1 else
                                                         ("%.2f cm off a whole %.0f cm module" % (off, module_cm))))
        out["module"] = {"measured": rows, "moduleCm": module_cm,
                         "convention": "the long sides of a modular piece are whole multiples of the kit's module, "
                                       "so pieces snap edge to edge",
                         "follows": on_all}

    # ---- triangles for the size ---------------------------------------------------------------------------------
    tris = sum(len(p.vertices) - 2 for p in me.polygons)
    area_m2 = 2.0 * ((size[0] * size[1]) + (size[1] * size[2]) + (size[0] * size[2])) / (cm * cm)
    per_m2 = tris / area_m2 if area_m2 > 1e-9 else None
    lo_t, hi_t, p95 = _CONV_TRIS[size_class]
    lo_d, hi_d = _CONV_TRIS_PER_M2[size_class]
    band = ("below the usual range" if tris < lo_t else "within the usual range" if tris <= hi_t
            else "above the usual range" if tris <= p95 else "far above the usual range (rare at this size)")
    out["triangles"] = {
        "measured": {"triangles": tris, "boxSurfaceM2": round(area_m2, 4),
                     "perM2": round(per_m2, 2) if per_m2 is not None else None},
        "usualRange": [lo_t, hi_t], "rareAbove": p95, "usualPerM2": [lo_d, hi_d],
        "position": band,
        "follows": lo_t <= tris <= hi_t,
        "note": ("a count past the usual range is ordinary on a Nanite mesh, which spends detail on large pieces; "
                 "below it, a piece can read as low-poly at close range"),
    }
    if not (lo_t <= tris <= hi_t):
        notes.append("%d triangles is %s for a %s piece (%d-%d)" % (tris, band, size_class, lo_t, hi_t))

    # ---- material slots ---------------------------------------------------------------------------------------
    slots = [s.material.name if s.material else None for s in obj.material_slots]
    defaults = [n for n in slots if n and _CONV_DEFAULT_MATERIAL.match(n)]
    usual_slots = 1 if size_class != "large" else 4
    out["materialSlots"] = {
        "measured": {"count": len(slots), "names": slots, "defaultNamed": defaults},
        "convention": ("one slot; two or more is unusual below the large size class, four or more appears only on "
                       "large assembled pieces. A slot still named a modeling tool's default was never named"),
        "follows": (not defaults) and 1 <= len(slots) <= usual_slots,
    }
    if defaults:
        notes.append("material slot(s) still named a modeling tool's default: %s" % ", ".join(defaults))
    if len(slots) > usual_slots:
        notes.append("%d material slots on a %s piece; one is usual" % (len(slots), size_class))
    if not slots:
        notes.append("no material slot - the piece imports with the engine's default material")

    # ---- thickness of a wall or floor --------------------------------------------------------------------------
    if kind in ("wall", "floor"):
        thick = across if kind == "wall" else size[2]
        bm = bmesh.new()
        try:
            bm.from_mesh(me)
            boundary = sum(1 for e in bm.edges if len(e.link_faces) == 1)
        finally:
            bm.free()
        zero = thick < 0.01
        out["thickness"] = {
            "measured": {"thicknessCm": round(thick, 3), "boundaryEdges": boundary},
            "typicalCm": _CONV_TYPICAL_THICKNESS_CM[kind],
            "convention": "a thin slab with real thickness; a single plane is seen from one side only and needs a "
                          "two-sided material",
            "follows": not zero,
        }
        if zero:
            notes.append("the %s has no thickness - a single plane (%d open edges), seen from one side only"
                         % (kind, boundary))

    # ---- transform ----------------------------------------------------------------------------------------------
    scale = [round(v, 6) for v in obj.scale]
    rot_ok = rotation_is_identity(obj)
    out["transform"] = {
        "measured": {"scale": scale, "rotationApplied": rot_ok},
        "convention": "scale and rotation applied (1, 1, 1), so the exported size is the mesh's own",
        "follows": scale == [1.0, 1.0, 1.0] and rot_ok,
    }
    if not out["transform"]["follows"]:
        notes.append("the transform is not applied (scale %s, rotation %s) - apply_transform before export"
                     % (scale, "applied" if rot_ok else "not applied"))

    # ---- collision hulls ------------------------------------------------------------------------------------------
    hull_re = re.compile(r"^(%s)_%s_\d+$" % ("|".join(_CONV_HULL_PREFIXES), re.escape(obj.name)))
    hulls = []
    for o in bpy.data.objects:
        if o.type == "MESH" and hull_re.match(o.name):
            hulls.append({"object": o.name, "vertices": len(o.data.vertices),
                          "withinEngineLimit": len(o.data.vertices) <= _UE_HULL_VERT_LIMIT})
    hulls.sort(key=lambda h: h["object"])
    over = [h["object"] for h in hulls if not h["withinEngineLimit"]]
    out["collision"] = {
        "measured": {"hulls": hulls, "count": len(hulls)},
        "convention": ("a few simple hulls named UCX_<mesh>_NN travel with the mesh; with none, the engine makes "
                       "collision on import or the piece has none - common on small props"),
        "follows": (not over) if hulls else None,
    }
    if over:
        notes.append("hull(s) over %d vertices: %s" % (_UE_HULL_VERT_LIMIT, ", ".join(over)))

    # ---- the name -------------------------------------------------------------------------------------------------
    out["name"] = {
        "measured": {"name": obj.name, "prefixSM": obj.name.startswith("SM_"),
                     "numbered": bool(_CONV_NAME_NUMBERED.search(obj.name))},
        "convention": "SM_ then words, then a two-digit number and a variant letter - SM_Wall_Plain_01a",
        "follows": bool(_CONV_NAME.match(obj.name)) and bool(_CONV_NAME_NUMBERED.search(obj.name)),
    }
    if not out["name"]["follows"]:
        notes.append("the name '%s' is not SM_<Words>_NNa" % obj.name)

    out["notes"] = notes
    out["followCount"] = sum(1 for v in out.values() if isinstance(v, dict) and v.get("follows") is True)
    out["differCount"] = sum(1 for v in out.values() if isinstance(v, dict) and v.get("follows") is False)
    out["reachNote"] = ("each finding is a note with its number, not a verdict: a convention is how pieces are "
                        "usually built for a game engine, and a piece can have a reason to differ")
    return out


def _closed_edge_loops(bm):
    """Every CLOSED edge loop of the mesh as a list of vertices: from an edge, step across each vertex of valence 4 to
    the edge that shares no face with the one arrived on; a loop that reaches a pole or a boundary is open and dropped."""
    bm.edges.ensure_lookup_table()
    seen, loops = set(), []
    for e0 in bm.edges:
        if e0.index in seen or len(e0.link_faces) != 2:
            continue
        chain, closed = [e0], False
        for side in (0, 1):
            e, v = e0, e0.verts[side]
            while True:
                if v.is_boundary or len(v.link_edges) != 4:
                    break
                faces = {f.index for f in e.link_faces}
                nxt = [x for x in v.link_edges if x.index != e.index and not ({f.index for f in x.link_faces} & faces)]
                if len(nxt) != 1:
                    break
                e = nxt[0]
                if e.index == e0.index:
                    closed = True
                    break
                chain.append(e)
                v = e.other_vert(v)
            if closed:
                break
        seen.update(x.index for x in chain)
        if closed:
            verts, order = {}, []
            for x in chain:
                for vv in x.verts:
                    if vv.index not in verts:
                        verts[vv.index] = vv
                        order.append(vv)
            loops.append(order)
    return loops


def _poles_report(bm):
    """Interior vertices whose valence is not 4: 3 and 5-or-more, with indices (200 listed each)."""
    by_val = {}
    for v in bm.verts:
        if v.is_boundary or not v.link_faces:
            continue
        n = len(v.link_edges)
        if n != 4:
            by_val.setdefault(n, []).append(v.index)
    three = sorted(by_val.get(3, []))
    five = sorted(i for n, ids in by_val.items() if n >= 5 for i in ids)
    return {"valence3": {"count": len(three), "indices": three[:200]},
            "valence5plus": {"count": len(five), "indices": five[:200]},
            "byValence": {str(n): len(ids) for n, ids in sorted(by_val.items())},
            "note": "interior vertices only - a boundary vertex's valence is not a pole's"}


def _joint_spec(spec, obj):
    """mesh_quality jointRings checked before anything is measured: (bone name, head, unit axis, range) or None."""
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) - {"bone", "armature", "range"} or not spec.get("bone"):
        raise MifOpError("jointRings is {bone, armature, range}; got %r. NOTHING was measured." % (spec,))
    rng = take_float(spec, "range", default=0.1)
    if not rng > 0.0:
        raise MifOpError("jointRings.range must be above 0 (world units). NOTHING was measured.")
    _arm, name, head, tail = bone_segment(obj, str(spec["bone"]), spec.get("armature"), "measured")
    return name, head, (tail - head).normalized(), rng


def _joint_rings(bm, mw, joint):
    """Closed edge loops that go round a bone's axis within `range` of its head (world units)."""
    name, head, axis, rng = joint
    rings = []
    for verts in _closed_edge_loops(bm):
        pts = [mw @ v.co for v in verts]
        c = sum(pts, Vector()) / len(pts)
        along = (c - head).dot(axis)
        if abs(along) > rng:
            continue
        # THE WINDING ABOUT THE AXIS: a ring round the limb turns 2 pi about it, a loop beside it turns 0.
        flat = [(p - head) - axis * (p - head).dot(axis) for p in pts]
        ref = flat[0].normalized() if flat[0].length > 1e-12 else None
        if ref is None:
            continue
        side = axis.cross(ref)
        ang = [math.atan2(f.dot(side), f.dot(ref)) for f in flat]
        turn = 0.0
        for a, b in zip(ang, ang[1:] + ang[:1]):
            d = b - a
            turn += (d + math.pi) % (2.0 * math.pi) - math.pi
        if abs(abs(turn) - 2.0 * math.pi) < 0.5:
            rings.append({"vertices": len(verts), "along": round(along, 6)})
    rings.sort(key=lambda r: r["along"])
    return {"bone": name, "range": rng, "count": len(rings), "rings": rings[:50]}


def _loop_bone_axis(obj, bone, armature, want_loops):
    """(bone name, head, unit axis) in world space from the bone's REST position - where a base mesh is - or None.
    Refuses before anything is measured."""
    if bone is None:
        if armature is not None:
            raise MifOpError("armature goes with bone. NOTHING was measured.")
        return None
    if not want_loops:
        raise MifOpError("bone measures each boundary loop's span along it, so it needs boundaryLoops:true. NOTHING was "
                         "measured.")
    _arm, name, head, tail = bone_segment(obj, bone, armature, "measured")
    return name, head, (tail - head).normalized()


def _boundary_loop_rows(bm, mw, bone_axis, limit=50):
    """The open boundary loops as data, in world units - see mesh_quality's boundaryLoops. Positions are taken with the
    object's origin as the working origin (_shifted) and the origin added back, so lengths keep their millimetres."""
    shift = mw.translation.copy()
    work = _shifted(mw, shift)
    loops = _boundary_loops(bm)
    rows = []
    for i, (edges, verts) in enumerate(loops[:limit]):
        pos = {v: work @ v.co for v in verts}
        deg = {}
        for e in edges:
            for v in e.verts:
                deg[v] = deg.get(v, 0) + 1
        c = sum(pos.values(), Vector()) / len(verts)
        row = {"index": i, "edges": len(edges), "vertices": len(verts),
               "closed": all(n == 2 for n in deg.values()),
               "length": round(sum((pos[e.verts[0]] - pos[e.verts[1]]).length for e in edges), 6),
               "centroid": [round(c[k] + shift[k], 6) for k in range(3)],
               "boxMin": [round(min(p[k] for p in pos.values()) + shift[k], 6) for k in range(3)],
               "boxMax": [round(max(p[k] for p in pos.values()) + shift[k], 6) for k in range(3)]}
        if bone_axis is not None:
            name, head, axis = bone_axis
            ts = [(p - (head - shift)).dot(axis) for p in pos.values()]
            row["along"] = {"bone": name, "from": round(min(ts), 6), "to": round(max(ts), 6),
                            "span": round(max(ts) - min(ts), 6)}
        rows.append(row)
    return {"loopCount": len(loops), "loops": rows, "listed": len(rows),
            "note": ("each connected run of boundary edges is one loop; closed is false where a bow-tie vertex (more "
                     "than two boundary edges) joins two. along is measured on the bone's REST axis, from its head."
                     if loops else "no open boundary - every edge has two faces.")}


# MODULE-LEVEL AND A PLAIN LITERAL: parity_check.py resolves an accepted-key expression statically.
_THICK_METHODS = ("BOX", "MEAN", "LOCAL", "ALL")


def parse_thickness(spec, min_cm):
    """mesh_quality's thickness ask: True, or {method, samples, seed}. (method, samples, seed) or None when not asked.
    Read before anything is measured, so a bad value changes nothing."""
    if spec is None:
        if min_cm is not None:
            spec = True
        else:
            return None
    if spec is False:
        if min_cm is not None:
            raise MifOpError("thicknessMinCm is read against the thickness block, and thickness:false turns it off. "
                             "NOTHING was measured.")
        return None
    if spec is True:
        spec = {}
    if not isinstance(spec, dict):
        raise MifOpError("thickness is true or {method, samples, seed}, not %r. NOTHING was measured." % (spec,))
    reject_unknown(spec, ("method", "samples", "seed"), "mesh_quality thickness")
    method = str(spec.get("method", "ALL")).upper()
    if method not in _THICK_METHODS:
        raise MifOpError("thickness.method is BOX, MEAN, LOCAL or ALL, not %r. NOTHING was measured." % spec.get("method"))
    samples = spec.get("samples", 2000)
    if isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= 100000:
        raise MifOpError("thickness.samples is a whole number from 1 to 100000, not %r. NOTHING was measured." % (samples,))
    seed = spec.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MifOpError("thickness.seed is a whole number, not %r. NOTHING was measured." % (seed,))
    if min_cm is not None and not (math.isfinite(min_cm) and min_cm > 0.0):
        raise MifOpError("thicknessMinCm is a thickness in CENTIMETERS above 0, not %r. NOTHING was measured." % min_cm)
    return method, samples, seed


def thickness_report(obj, method, samples, seed, min_cm):
    """The piece's thickness three ways, in centimeters, from the mesh AS AUTHORED in world space (spec "Measure the
    thinnest local thickness of a piece against a threshold", 2026-10-11). They disagree on a hollow or bevelled piece,
    so each says what it is:
      boxCm     the thin side of the world bounds - exact for a slab, blind to a hollow
      meanCm    twice the volume over the surface area of a CLOSED mesh (a slab: V = a x t, A ~ 2a) - None when open
      local     rays from sampled surface points, inward along the reversed face normal, to the first far-side hit:
                localMinCm, localP5Cm (the 5th percentile), thinnestAt [x, y, z] in world units, hitRate - a low rate
                means an open or non-manifold piece, whose rays find no far side
    The tool never picks the threshold: `follows` is held against thicknessMinCm when one is given, else null - the
    figure depends on the renderer's tracing mode and settings, which the caller knows and this does not."""
    from bisect import bisect_left              # noqa: PLC0415
    from itertools import accumulate            # noqa: PLC0415
    from mathutils.bvhtree import BVHTree       # noqa: PLC0415 - Blender's own module, only here
    cm = UU_PER_BU
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.transform(obj.matrix_world)
        bm.normal_update()
        faces = [f for f in bm.faces if f.calc_area() > 0.0]
        if not faces:
            raise MifOpError("'%s' has no face with area, so it has no thickness to measure. NOTHING was measured."
                             % obj.name)
        out = {"method": method}
        lo = [min(v.co[k] for v in bm.verts) for k in range(3)]
        hi = [max(v.co[k] for v in bm.verts) for k in range(3)]
        if method in ("BOX", "ALL"):
            out["boxCm"] = round(min(hi[k] - lo[k] for k in range(3)) * cm, 3)
        boundary = sum(1 for e in bm.edges if len(e.link_faces) != 2)
        if method in ("MEAN", "ALL"):
            area = sum(f.calc_area() for f in faces)
            if boundary:
                out["meanCm"] = None
                out["meanNote"] = ("%d edge(s) are open or shared by more than two faces, so the mesh encloses no "
                                   "volume to divide" % boundary)
            else:
                out["meanCm"] = round(2.0 * abs(bm.calc_volume(signed=False)) / area * cm, 3)
        if method in ("LOCAL", "ALL"):
            tree = BVHTree.FromBMesh(bm)
            rng = random.Random(seed)
            weights = list(accumulate([f.calc_area() for f in faces]))
            total = weights[-1]
            eps = 1e-5 * max(1e-6, max(hi[k] - lo[k] for k in range(3)))
            hits = []
            for _i in range(samples):
                r = rng.random() * total
                f = faces[bisect_left(weights, r)]
                # a uniform point on the face: a random triangle of its fan by area, then barycentric
                tris = [(f.verts[0].co, f.verts[j].co, f.verts[j + 1].co) for j in range(1, len(f.verts) - 1)]
                ta = [((b - a).cross(c - a)).length for a, b, c in tris]
                pick = rng.random() * sum(ta)
                k = 0
                while k < len(ta) - 1 and pick > ta[k]:
                    pick -= ta[k]
                    k += 1
                a, b, c = tris[k]
                u, v = rng.random(), rng.random()
                if u + v > 1.0:
                    u, v = 1.0 - u, 1.0 - v
                p = a + (b - a) * u + (c - a) * v
                n = f.normal
                loc, _nrm, _idx, dist = tree.ray_cast(p - n * eps, -n)
                if loc is not None and dist is not None and dist > eps:
                    hits.append((dist + eps, p))
            out["samples"] = samples
            out["hitRate"] = round(len(hits) / float(samples), 4)
            if hits:
                hits.sort(key=lambda h: h[0])
                out["localMinCm"] = round(hits[0][0] * cm, 3)
                out["localP5Cm"] = round(hits[min(len(hits) - 1, int(0.05 * len(hits)))][0] * cm, 3)
                out["thinnestAt"] = [round(x, 5) for x in hits[0][1]]
            else:
                out["localMinCm"] = out["localP5Cm"] = out["thinnestAt"] = None
                out["localNote"] = "no ray found a far side - the piece is open (a single sheet) or has no inside"
    finally:
        bm.free()
    out["thicknessMinCm"] = min_cm
    if min_cm is None:
        out["follows"] = None
    else:
        # the thinnest measure asked for decides; the local 5th percentile over the minimum, so one stray ray does not
        figures = [out.get(k) for k in ("localP5Cm", "meanCm", "boxCm") if out.get(k) is not None]
        out["follows"] = (figures[0] >= min_cm) if figures else None
        out["judgedBy"] = next((k for k in ("localP5Cm", "meanCm", "boxCm") if out.get(k) is not None), None)
    return out


MESH_QUALITY_PARAMS = (
    "object", "name", "uvLayer", "texelDensityFor", "conventions", "kind", "module", "boundaryLoops", "bone",
    "armature", "poles", "jointRings", "facingAgainst", "flippedAgainst", "collection", "thickness", "thicknessMinCm",
    "gameReady",
)
_KIT_ONE_MESH_KEYS = ("object", "name", "boundaryLoops", "bone", "armature", "poles", "jointRings", "facingAgainst",
                      "flippedAgainst", "thickness", "thicknessMinCm", "gameReady")
_GAME_READY_KEYS = ("triangleBudget", "allowNgons", "uvOneTile", "allowFloating", "allowWaste")


def _game_ready_spec(raw):
    """mesh_quality's gameReady, checked before anything is measured: None when not asked, else {triangleBudget (an int
    above 0, or None: the top of the usual range for the mesh's size class is used - triangle_budget_for_size), allowNgons
    (default false), uvOneTile (default false: UVs past 0-1 are right on a tiling texture, and only a bake or an atlas
    must hold them inside), allowFloating (default false: parts meant to stand apart are named in notJudged)}."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        if not take_bool({"gameReady": raw}, "gameReady"):
            return None
        raw = {}
    unknown = sorted(str(k) for k in raw if k not in _GAME_READY_KEYS)
    if unknown:
        raise MifOpError("gameReady takes %s; %s %s not among them. NOTHING was measured."
                         % (", ".join(_GAME_READY_KEYS), ", ".join(unknown), "is" if len(unknown) == 1 else "are"))
    budget = raw.get("triangleBudget")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < 1):
        raise MifOpError("gameReady.triangleBudget is a whole number of triangles, 1 or more, got %r. NOTHING was "
                         "measured." % (budget,))
    return {"triangleBudget": budget, "allowNgons": take_bool(raw, "allowNgons", default=False),
            "allowWaste": take_bool(raw, "allowWaste", default=False),
            "uvOneTile": take_bool(raw, "uvOneTile", default=False),
            "allowFloating": take_bool(raw, "allowFloating", default=False)}


# STRAY PIECES AND SLIVERS (1.4, 2026-10-30), measured in world space on the mesh as authored.
# A part is FLOATING when its surface is under this share of the largest part's and it touches no other part: no vertex
# within _GR_TOUCH_REL of the object's longest side of another part's surface, and no face crossing one. A merlon set on
# a wall (1% of the tower's area, measured) touches it and is not floating; a chip a boolean left in the air is.
# 10%, not the 1% this began at: a tool cart's two side braces hung 100 mm from everything at 2% of the frame and passed.
# A part above it is a piece in its own right, such as a second rock in one mesh.
# AND IT HAS TO STAND AWAY FROM THE REST: at least _GR_FLOAT_GAP_REL of the longest side. On 936 shipped static meshes
# (2026-11-01) the rule without a gap failed 86, leaves a few millimeters off a branch and fittings a hair off a wall, and
# all but a few foliage cards stood under 2%; the part that does hang in the air, a tool cart's braces, stands 10% away.
# Nearer ones are counted in notJudged, since nothing measured tells a fitting from a chip that close.
# A GROUP FLOATS TOO (2026-11-02). Each part of a caster touches the next, so no part of it floats, and four whole casters
# hung 12 to 15 mm off a 0.95 m cart and passed: 1.3 to 1.6% of the longest side, 9 to 15% of their own size. So parts joined
# by touching or crossing are a group, and a small group is judged against the body when one group holds over half the
# surface (_GR_BODY_SHARE): a mesh with no body has nothing its pieces should hang from (a bush: largest group 10%). It floats
# at _GR_GROUP_GAP_REL of the longest side and _GR_GROUP_GAP_OWN of its own longest side, or at _GR_GROUP_GAP_FAR whatever
# its size. Its own size is what keeps a long pane in its frame (2.3% away, 3% of its length) and a small part a hair off
# a big wall out. And what it stands nearest has to be the body: blades of a grass tuft, whose crossing blades do make a
# body, stood 1.5 cm from the next blades, 9% of their size, which is a caster's number, and it was only this that told them
# apart. On 500 shipped meshes of more than one part, no mesh that passed fails for it and one that failed already gains the
# reason, a roof tile 4 cm clear of its deck. Of the groups apart, the 90th percentile stands 0.05 to 0.3% of the longest
# side away, and 1.2% in the worst class of props.
_GR_FLOAT_AREA_RATIO = 0.1
_GR_TOUCH_REL = 1e-4
_GR_FLOAT_GAP_REL = 0.02
_GR_BODY_SHARE = 0.5
_GR_GROUP_GAP_REL = 0.01
_GR_GROUP_GAP_OWN = 0.05
_GR_GROUP_GAP_FAR = 0.03
# A BOARD HUNG SHORT (2026-11-02). 66 of 72 wooden builds with boards touching nothing passed, gaps up to 15 mm inside those rules. A solid
# stands apart when the gap is over _GR_BOARD_GAP_REL of the longest side and _GR_BOARD_GAP_THICK of its own thickness, its least side, itself
# at least _GR_BOARD_THICK_REL of the longest side so a sheet or a leaf is not judged by it. Real attachments stand closer for their thickness.
# The gap here is the exact one, edges to edges as well as vertices to faces: two long edges can cross near where no vertex is near anything.
_GR_BOARD_GAP_REL = 0.004
_GR_BOARD_GAP_THICK = 0.2
_GR_BOARD_THICK_REL = 0.005
_GR_BOARD_EDGE_PAIRS = 200000
# A face is a SLIVER when it is narrower than this share of the object's longest side and at least this many times
# longer than it is wide (width = twice the area over the longest edge). The castle tower's eight were 2.5 m and 0.7 m
# long and under 4 micrometers wide; a 3 m pole's long triangles (16 mm wide, 190 to 1) and a 2 cm ring's (1.5 mm,
# 1.4 to 1) are not slivers.
# AND UNDER _GR_SLIVER_MAX_WIDTH_M, whatever the mesh's size: 1/10000 of a 10 m mesh is 1 mm, and a 1 mm face is a bevel
# strip, not a needle. On the shipped set, three of four meshes the share alone failed had no face under 10 micrometers,
# and the distance dissolving their widest takes bevels with it (over 20% of the triangles on some).
_GR_SLIVER_REL_WIDTH = 1e-4
_GR_SLIVER_ASPECT = 100.0
_GR_SLIVER_MAX_WIDTH_M = 1e-5
# A mesh of this many parts is a cloud of cards or clutter, and its parts are not examined: the shipped meshes still
# failing past the gap rule were foliage cards in meshes of 2,379 to 15,234 parts (each part is also a BVH overlap test).
_GR_SCATTER_PARTS = 1000


def _group_stands_apart(gap, longest, size):
    return gap >= _GR_GROUP_GAP_FAR * longest or (gap >= _GR_GROUP_GAP_REL * longest and gap >= _GR_GROUP_GAP_OWN * size)


def _board_stands_apart(gap, longest, thick):
    return gap >= _GR_BOARD_GAP_REL * longest and gap >= _GR_BOARD_GAP_THICK * thick


def _segments_apart(p1, q1, p2, q2):
    """The least distance between two segments."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1.dot(d1), d2.dot(d2), d2.dot(r)
    if a < 1e-18 and e < 1e-18:
        return r.length
    if a < 1e-18:
        s, t = 0.0, max(0.0, min(1.0, f / e))
    else:
        c = d1.dot(r)
        if e < 1e-18:
            s, t = max(0.0, min(1.0, -c / a)), 0.0
        else:
            b = d1.dot(d2)
            denom = a * e - b * b
            s = max(0.0, min(1.0, (b * f - c * e) / denom)) if denom > 1e-18 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t, s = 1.0, max(0.0, min(1.0, (b - c) / a))
    return (r + d1 * s - d2 * t).length


def _board_gap_exact(bm, faces, others_of, lo, hi, gap):
    """The gap from these faces to the rest, with edges against edges added to the vertex searches it already has. None when too many edges
    lie in reach to measure: the caller then leaves the group uncalled rather than fail it on a number it did not finish."""
    own = {e for f in faces for e in f.edges}
    near = []
    for e in bm.edges:
        if e in own or not others_of(e):
            continue
        a, b = e.verts[0].co, e.verts[1].co
        if all(min(a[i], b[i]) <= hi[i] + gap and max(a[i], b[i]) >= lo[i] - gap for i in range(3)):
            near.append((a, b))
    if len(own) * len(near) > _GR_BOARD_EDGE_PAIRS:
        return None
    best = gap
    for e in own:
        a, b = e.verts[0].co, e.verts[1].co
        for c, d in near:
            best = min(best, _segments_apart(a, b, c, d))
    return best


def _game_ready_shape(obj):
    """Parts and groups of touching parts that float free of the rest, and sliver faces, in world space. {floating:
    [{faces, parts, areaRatio, gapM, sizeM}], floatingFaces: [indices], nearby: what touches nothing but stands too near
    to judge, noBody: groups apart in a mesh no group dominates, slivers: [indices], sliverWidthMax, longestSideM, parts,
    groups, notExamined}."""
    from mathutils.bvhtree import BVHTree   # noqa: PLC0415 - deferred, as elsewhere here (stub mathutils offline)
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.transform(obj.matrix_world)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        if not bm.verts:
            return None
        lo = [min(v.co[k] for v in bm.verts) for k in range(3)]
        hi = [max(v.co[k] for v in bm.verts) for k in range(3)]
        longest = max(hi[k] - lo[k] for k in range(3))
        # ---- slivers ----
        slivers, sliver_w = [], 0.0
        width_limit = min(_GR_SLIVER_REL_WIDTH * longest, _GR_SLIVER_MAX_WIDTH_M)
        for f in bm.faces:
            area = f.calc_area()
            if area < _DEGENERATE_FACE_AREA:
                continue            # degenerateFaces has its own reason
            edge = max(e.calc_length() for e in f.edges)
            width = 2.0 * area / edge if edge > 0.0 else 0.0
            if width < width_limit and edge >= _GR_SLIVER_ASPECT * width:
                slivers.append(f.index)
                sliver_w = max(sliver_w, width)
        # ---- parts, by faces sharing an edge ----
        part_of = [-1] * len(bm.faces)
        parts = []
        for f0 in bm.faces:
            if part_of[f0.index] >= 0:
                continue
            k = len(parts)
            part_of[f0.index] = k
            stack, members = [f0], []
            while stack:
                f = stack.pop()
                members.append(f)
                for e in f.edges:
                    for g in e.link_faces:
                        if part_of[g.index] < 0:
                            part_of[g.index] = k
                            stack.append(g)
            parts.append(members)
        floating, floating_faces, nearby, not_examined, no_body, group_count = [], [], [], 0, 0, len(parts)
        if len(parts) >= _GR_SCATTER_PARTS:
            areas = [sum(f.calc_area() for f in p) for p in parts]
            cut = _GR_FLOAT_AREA_RATIO * max(areas)
            not_examined = sum(1 for a in areas if a < cut)
        elif len(parts) > 1:
            areas = [sum(f.calc_area() for f in p) for p in parts]
            biggest = max(areas) or 1.0
            tree = BVHTree.FromBMesh(bm)
            touch = _GR_TOUCH_REL * longest
            link = list(range(len(parts)))

            def root(a):
                while link[a] != a:
                    link[a] = link[link[a]]
                    a = link[a]
                return a

            # Touching is looked for from every part but the one with the most faces: it is found from the other side.
            heaviest = max(range(len(parts)), key=lambda k: len(parts[k]))
            part_verts = {}
            for k, members in enumerate(parts):
                if k == heaviest:
                    continue
                verts = part_verts[k] = list({v for f in members for v in f.verts})
                for v in verts:
                    for h in tree.find_nearest_range(v.co, touch):
                        if h[2] is not None and part_of[h[2]] != k:
                            link[root(k)] = root(part_of[h[2]])
                index = {v: i for i, v in enumerate(verts)}
                sub = BVHTree.FromPolygons([v.co for v in verts], [[index[v] for v in f.verts] for f in members])
                for _a, b in sub.overlap(tree):
                    if part_of[b] != k:
                        link[root(k)] = root(part_of[b])
            groups = {}
            for k in range(len(parts)):
                groups.setdefault(root(k), []).append(k)
            group_count = len(groups)
            group_area = {g: sum(areas[k] for k in members) for g, members in groups.items()}
            main = max(group_area, key=group_area.get)
            whole = sum(group_area.values()) or 1.0
            has_body = group_area[main] > _GR_BODY_SHARE * whole
            for g, members in groups.items():
                if g == main:
                    continue
                alone = len(members) == 1 and areas[members[0]] < _GR_FLOAT_AREA_RATIO * biggest
                small = group_area[g] < _GR_FLOAT_AREA_RATIO * group_area[main]
                if not alone and not (small and has_body):
                    no_body += 1 if small else 0
                    continue
                verts = [v for k in members for v in part_verts.get(k) or {v for f in parts[k] for v in f.verts}]
                lo_g = [min(v.co[i] for v in verts) for i in range(3)]
                hi_g = [max(v.co[i] for v in verts) for i in range(3)]
                size = max(hi_g[i] - lo_g[i] for i in range(3))
                # The gap: the nearest other group, from this group's vertices (a bounded search, doubled).
                gap = nearest = None
                reach = max(touch, 1e-6)
                while gap is None and reach < 4.0 * longest + 1.0:
                    reach *= 2.0
                    for v in verts:
                        for h in tree.find_nearest_range(v.co, reach):
                            if h[2] is not None and root(part_of[h[2]]) != g and (gap is None or h[3] < gap):
                                gap, nearest = h[3], root(part_of[h[2]])
                # And from the other side, before it can fail: the group's vertices can all stand farther off than a point of
                # the body stands from one of its faces (a block hung under a vertex read 2.7 times its gap).
                if gap is not None and gap >= min(_GR_GROUP_GAP_REL, _GR_FLOAT_GAP_REL) * longest:
                    index = {v: i for i, v in enumerate(verts)}
                    sub = BVHTree.FromPolygons([v.co for v in verts], [[index[v] for v in f.verts] for k in members for f in parts[k]])
                    near = gap
                    for v in bm.verts:
                        if all(lo_g[i] - near <= v.co[i] <= hi_g[i] + near for i in range(3)) and v not in index:
                            hit = sub.find_nearest(v.co, near)
                            if hit[3] is not None and hit[3] < near:
                                near = hit[3]
                                nearest = root(part_of[v.link_faces[0].index]) if v.link_faces else nearest
                    gap = near
                own = [f.index for k in members for f in parts[k]]
                row = {"faces": len(own), "parts": len(members), "areaRatio": round(group_area[g] / group_area[main], 6),
                       "gapM": round(gap, 6) if gap is not None else None, "sizeM": round(size, 6), "firstFace": own[0]}
                as_part = alone and (gap is None or gap >= _GR_FLOAT_GAP_REL * longest)
                as_group = False
                if small and has_body and nearest == main and gap is not None:
                    thick = min(hi_g[i] - lo_g[i] for i in range(3))
                    as_group = _group_stands_apart(gap, longest, size)
                    board = thick >= _GR_BOARD_THICK_REL * longest
                    if as_group or (board and _board_stands_apart(gap, longest, thick)):
                        exact = _board_gap_exact(bm, [f for k in members for f in parts[k]],
                                                 lambda e: any(root(part_of[x.index]) != g for x in e.link_faces), lo_g, hi_g, gap)
                        if exact is not None:
                            row["gapM"], row["thicknessM"] = round(exact, 6), round(thick, 6)
                            as_group = _group_stands_apart(exact, longest, size) or (board and _board_stands_apart(exact, longest, thick))
                if as_part or as_group:
                    floating.append(row)
                    floating_faces.extend(own)
                else:
                    nearby.append(row)
        return {"floating": floating, "floatingFaces": sorted(floating_faces), "nearby": nearby, "slivers": slivers,
                "sliverWidthMax": sliver_w, "longestSideM": longest, "parts": len(parts), "notExamined": not_examined,
                "noBody": no_body, "groups": group_count,
                "sliverWidthLimitM": width_limit, "touchM": _GR_TOUCH_REL * longest}
    finally:
        bm.free()


# WASTED FACES (2026-11-01): triangles under the budget can still draw nothing. FLAT: what a Limited Dissolve at a hair's angle
# removes, keeping materials, seams, sharp edges and UV islands. HIDDEN: faces no ray from just off them reaches open space
# along, sampled. Held to shares and counts where they fail no good mesh, on 380 shipped meshes: of the 192 passing, the only
# four at or past the flat rule were painted, their vertices carrying colors, so a mesh whose colors vary is not judged on it,
# and none is past the hidden rule. 21 others fail it, all of them on another reason too. So shipped meshes rarely need it; a
# flat grid with no colors and a sphere sealed in a box, which tools make, are what it is for.
_GR_WASTE_FLAT_SHARE, _GR_WASTE_FLAT_TRIS = 0.9, 1000
_GR_WASTE_HIDDEN_SHARE, _GR_WASTE_HIDDEN_TRIS = 0.5, 2000
_GR_WASTE_MAX_TRIS = 400_000
_GR_WASTE_FLAT_DEG = 0.1
_GR_WASTE_SAMPLES = 3000
_GR_WASTE_RAYS = 96


def _colors_vary(mesh):
    """Does any color attribute hold more than one value, so its vertices carry paint however flat the mesh?"""
    for attr in getattr(mesh, "color_attributes", ()):
        data = attr.data
        first = None
        for i in range(0, len(data), max(1, len(data) // 2000)):
            val = tuple(round(c, 3) for c in data[i].color)
            if first is None:
                first = val
            elif val != first:
                return True
    return False


def _game_ready_waste(obj):
    """{flatTris, flatShare, hiddenTris, hiddenShare, colorsVary}, or None past _GR_WASTE_MAX_TRIS triangles."""
    from mathutils.bvhtree import BVHTree   # noqa: PLC0415 - deferred, as _game_ready_shape's
    mesh = obj.data
    tris = mesh_counts(obj)["tris"]
    if tris > _GR_WASTE_MAX_TRIS or not tris:
        return None
    known = [i.identifier for i in bpy.ops.mesh.dissolve_limited.get_rna_type().properties["delimit"].enum_items]
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(_GR_WASTE_FLAT_DEG), use_dissolve_boundaries=False,
                                 verts=bm.verts[:], edges=bm.edges[:],
                                 delimit={d for d in ("MATERIAL", "SEAM", "SHARP", "UV") if d in known})
        flat = tris - sum(len(f.verts) - 2 for f in bm.faces)
    finally:
        bm.free()
    golden = math.pi * (3.0 - math.sqrt(5.0))
    dirs = []
    for i in range(_GR_WASTE_RAYS):
        z = 1.0 - 2.0 * (i + 0.5) / _GR_WASTE_RAYS
        r = math.sqrt(max(0.0, 1.0 - z * z))
        dirs.append(Vector((r * math.cos(golden * i), r * math.sin(golden * i), z)))
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.normal_update()
        tree = BVHTree.FromBMesh(bm)
        span = max(max(v.co[k] for v in bm.verts) - min(v.co[k] for v in bm.verts) for k in range(3)) or 1.0
        faces = bm.faces[:]
        hid = seen = 0
        for f in faces[::max(1, len(faces) // _GR_WASTE_SAMPLES)]:
            n = len(f.verts) - 2
            seen += n
            p = f.calc_center_median() + f.normal * (1e-4 * span)
            if not any(d.dot(f.normal) > 0.02 and tree.ray_cast(p, d)[0] is None for d in dirs):
                hid += n
    finally:
        bm.free()
    share = hid / max(seen, 1)
    return {"flatTris": flat, "flatShare": round(flat / tris, 4), "hiddenTris": int(round(share * tris)),
            "hiddenShare": round(share, 4), "hiddenSampled": seen, "colorsVary": _colors_vary(mesh)}


def _sliver_fix_distance(width):
    """dissolve_degenerate's distance for slivers up to `width` wide: it dissolves a face no wider than the distance
    (measured on 3.6 and 5.2: a 1 m triangle 1e-4 wide went at 1e-4 and stayed at 1e-5), so the next power of ten above
    the widest, never under clean_mesh's own 1e-6."""
    d = 1e-6
    while d <= width * 1.0001:
        d *= 10.0
    return float("%.0e" % d)


def _game_ready_verdict(obj, checks, spec):
    """Pass or fail for a game engine, read off the checks mesh_quality has just taken - plus the triangle count, and
    since 1.4's second pass the parts that float free and the sliver faces (_game_ready_shape). Each reason carries its
    number, its limit and the call that fixes it (args ready to send)."""
    name = obj.name
    tris = mesh_counts(obj)["tris"]
    faces = checks["faces"]
    reasons = []

    def fail(check, measured, limit, why, *fixes):
        reasons.append({"check": check, "measured": measured, "limit": limit, "why": why,
                        "fix": [{"call": c, "args": dict(a, object=name)} for c, a in fixes]})

    budget = spec["triangleBudget"]
    budget_from = {"from": "given"} if budget is not None else triangle_budget_for_size(obj)
    if budget is None and budget_from is not None:
        budget = budget_from["triangleBudget"]
    if budget is not None and tris > budget:
        given = budget_from.get("from") == "given"
        cut = (("dissolve_mesh_elements", {"mode": "LIMITED"}), ("decimate_mesh", {"targetTris": budget}))
        fail("triangles", tris, budget, "%d triangles, %d over the budget of %d%s. A Limited Dissolve takes only flat "
             "detail, a decimate takes the rest - check again after either"
             % (tris, tris - budget, budget,
                "" if given else
                " (no triangleBudget was given: the top of the usual range for a %s piece, %d cm on its longest side - "
                "a Nanite mesh past it is ordinary; give triangleBudget to judge it by another number. No fix is named "
                "against a number nobody asked for: take one of `choices`)"
                % (budget_from["sizeClass"], budget_from["longestSideCm"])),
             *(cut if given else ()))
        if not given:
            # On shipped meshes over it, the decimate left a median 31% of the triangles and took over 1% of the surface
            # from more than a quarter: the number is a convention, not this mesh's budget.
            reasons[-1]["choices"] = [
                {"when": "this mesh has a budget of its own: ask again with gameReady triangleBudget", "call": "mesh_quality",
                 "args": {"object": name, "gameReady": {"triangleBudget": "your number"}}},
                {"when": "flat detail can go", "call": "dissolve_mesh_elements", "args": {"object": name, "mode": "LIMITED"}},
                {"when": "it should come down to the usual range, shape and all", "call": "decimate_mesh",
                 "args": {"object": name, "targetTris": budget}}]
    if not spec["allowNgons"] and faces["ngons"]:
        fail("ngons", faces["ngons"], 0, "%d face(s) with more than four corners - an engine splits them on import, "
             "however it likes" % faces["ngons"], ("clean_mesh", {"triangulate": True}))
    # A fix is sent as given, and clean_mesh refuses to lose an imported mesh's normals, so the fix says to keep them.
    keep_args = {"keepCustomNormals": True} if getattr(obj.data, "has_custom_normals", False) else {}
    nm = checks["nonManifoldEdges"]["count"]
    if nm:
        fail("nonManifoldEdges", nm, 0, "%d edge(s) on three or more faces - they break bakes, collision and "
             "lightmaps. splitNonManifold gives each face its own copy of the edge and moves nothing; a weld cannot help, "
             "since those faces already share the edge" % nm,
             ("clean_mesh", dict({"splitNonManifold": True}, **keep_args)))
    lv, le = checks["looseVerts"]["count"], checks["looseEdges"]["count"]
    if lv or le:
        fail("looseGeometry", lv + le, 0, "%d loose vertices and %d loose edges - on no face, they draw nothing and "
             "still widen the bounds" % (lv, le), ("clean_mesh", {"removeLoose": True}))
    if "uv" not in checks:
        fail("uvMissing", 0, 1, "no UV layer - a texture or a lightmap has nothing to map by",
             ("uv_unwrap", {"method": "SMART"}))
    elif spec["uvOneTile"] and checks["uv"]["loopsOutside01"]:
        out = checks["uv"]["loopsOutside01"]
        fail("uvOutOfBounds", out, 0, "%d UV corner(s) outside the 0-1 tile of layer '%s'"
             % (out, checks["uv"]["layer"]),
             ("uv_unwrap", {"method": "SMART", "uvLayer": checks["uv"]["layer"], "replace": True}))
    if not checks["transform"]["scaleApplied"]:
        fail("scaleNotApplied", checks["transform"]["scale"], [1.0, 1.0, 1.0], "the object's scale is %s, so the mesh "
             "arrives at another size or with its normals bent" % (checks["transform"]["scale"],),
             ("apply_transform", {"scale": True}))
    flipped, inside = checks["windingFlippedEdges"]["count"], checks["insideOutParts"]["count"]
    if flipped or inside:
        left_edges, left_parts, after_faces = _winding_after_recalc(obj.data)
        unsettled = left_edges + left_parts
        # A PART FACING INWARD INSIDE ANOTHER IS USUALLY A CAVITY, A SOLIDIFY'S INNER SHELL (measured on 5.2.1, 2026-11-02): a
        # recalculation passes it and leaves a second surface facing out inside the solid, where a later cut finds no
        # cavity. It can also be an inner part wound wrong, so the call is the caller's. The check still fails either way.
        nested = _nested_inward_parts(obj.data) if inside else 0
        fail("inconsistentNormals", flipped + inside, 0, "%d edge(s) between faces wound opposite ways and %d closed "
             "part(s) facing inward - a one-sided material draws them see-through" % (flipped, inside),
             *(() if unsettled or nested else (("clean_mesh", dict({"recalcNormals": True}, **keep_args)),)))
        if nested and not unsettled:
            reasons[-1]["why"] += (". %d of them sit inside another part, which is what a cavity looks like, so no fix is "
                                   "named: take `choices`" % nested)
            reasons[-1]["choices"] = [
                {"when": "it is a cavity to open", "call": "boolean_op",
                 "args": {"target": name, "cutter": "an object through the wall", "operation": "difference"}},
                {"when": "it is a cavity nobody needs", "call": "delete_mesh_elements",
                 "args": {"object": name, "elements": "FACES", "indices": "the inner part's faces"}},
                {"when": "it is a separate inner part that is wound wrong", "call": "clean_mesh",
                 "args": dict({"object": name, "recalcNormals": True}, **keep_args)}]
        if unsettled:
            # NO FIX, ON PURPOSE: recalculating was tried on a copy and does not clear this mesh.
            reasons[-1]["why"] += (". Recalculating would leave %d of them, so no fix is named: look at windingFaces, "
                                   "then take `choices`" % unsettled)
            reasons[-1]["windingFaces"] = after_faces
            reasons[-1]["choices"] = [
                {"when": "you can see which faces point the wrong way", "call": "clean_mesh",
                 "args": dict({"object": name, "flipNormals": "the faces you chose"}, **keep_args)}]
    deg = checks["degenerateFaces"]["count"]
    if deg:
        fail("degenerateFaces", deg, 0, "%d face(s) of nearly no area - they shade as artifacts" % deg,
             ("clean_mesh", {"dissolveDegenerate": True}))
    shape = _game_ready_shape(obj) or {}
    floating = shape.get("floating") or []
    if floating and not spec["allowFloating"]:
        gaps = [p["gapM"] for p in floating if p["gapM"] is not None]
        fail("floatingParts", len(floating), 0, "%d part(s) of %d face(s) in all float free of the rest of the mesh, "
             "each under %g%% of the main body's surface and touching nothing outside itself; a part counts as floating "
             "from %g%% of the mesh's longest side, and parts touching each other, nearest the body, from %g%% when that is "
             "also %g%% of their own size, or from %g%% when that is %g%% of their thickness (the nearest is %s m away) - boolean debris, "
             "a piece left behind, or a wheel, brace or board hung "
             "short of what should hold it; they draw where nothing holds them and widen the bounds. Only you can tell "
             "which, so no fix is named: look at them (measured.floatingFaceIndices), then take one of `choices`"
             % (len(floating), len(shape["floatingFaces"]), 100 * _GR_FLOAT_AREA_RATIO, 100 * _GR_FLOAT_GAP_REL,
                100 * _GR_GROUP_GAP_REL, 100 * _GR_GROUP_GAP_OWN, 100 * _GR_BOARD_GAP_REL, 100 * _GR_BOARD_GAP_THICK,
                ("%.4g" % min(gaps)) if gaps else "farther than the mesh is wide"))
        # NO FIX, ON PURPOSE: a delete sent as given takes a wanted part, and "send each fix until it passes" sends it.
        reasons[-1]["choices"] = [
            {"when": "it belongs and should touch what holds it: split it off, move it, join it back",
             "call": "separate_mesh", "args": {"object": name, "mode": "selection", "faces": shape["floatingFaces"]}},
            {"when": "it is a leftover nobody wanted",
             "call": "delete_mesh_elements",
             "args": {"object": name, "indices": shape["floatingFaces"], "elements": "FACES"}},
            {"when": "it is meant to stand apart: ask again with gameReady allowFloating true", "call": "mesh_quality",
             "args": {"object": name, "gameReady": {"allowFloating": True}}}]
    slivers = shape.get("slivers") or []
    if slivers:
        dist = _sliver_fix_distance(shape["sliverWidthMax"])
        fail("sliverFaces", len(slivers), 0, "%d face(s) under %.3g m wide (1/%d of the mesh's %.3g m, %g m at most) and "
             "%d or more times longer than wide - the needles a cut or a triangulated n-gon leaves; the widest is %.3g "
             "m. They shade as streaks and give Unreal near-zero tangents. clean_mesh dissolveDegenerate at %g dissolves "
             "faces up to that width" % (len(slivers), shape["sliverWidthLimitM"], round(1 / _GR_SLIVER_REL_WIDTH),
                                         shape["longestSideM"], _GR_SLIVER_MAX_WIDTH_M, _GR_SLIVER_ASPECT,
                                         shape["sliverWidthMax"], dist),
             ("clean_mesh", {"dissolveDegenerate": dist}))
    waste = _game_ready_waste(obj)
    wasted = []
    if waste is not None:
        if waste["flatShare"] >= _GR_WASTE_FLAT_SHARE and waste["flatTris"] >= _GR_WASTE_FLAT_TRIS and not waste["colorsVary"]:
            wasted.append("%d triangles (%d%%) lie flat on one another and a Limited Dissolve takes them without changing the "
                          "silhouette or the shading" % (waste["flatTris"], round(100 * waste["flatShare"])))
        if waste["hiddenShare"] >= _GR_WASTE_HIDDEN_SHARE and waste["hiddenTris"] >= _GR_WASTE_HIDDEN_TRIS:
            wasted.append("about %d triangles (%d%%) face nothing a ray can reach from outside: sealed inside the mesh, or "
                          "pressed against it" % (waste["hiddenTris"], round(100 * waste["hiddenShare"])))
    if wasted and not spec["allowWaste"]:
        fail("wastedFaces", waste["flatTris"] + waste["hiddenTris"], 0, "; ".join(wasted) + ". Only you can tell if they have "
             "a use, such as a grid for a mask you have not painted yet or a room behind a door, so no fix is named: take one "
             "of `choices`")
        reasons[-1]["choices"] = [
            {"when": "flat detail can go", "call": "dissolve_mesh_elements", "args": {"object": name, "mode": "LIMITED", "angle": 0.1}},
            {"when": "they have a use: ask again with gameReady allowWaste true", "call": "mesh_quality",
             "args": {"object": name, "gameReady": {"allowWaste": True}}}]
    tangents = checks.get("tangents") or {}
    zero_t = tangents.get("nearZeroVertexInstances") or 0
    t_faces = tangents.get("faces") or []
    on_slivers = bool(t_faces) and set(t_faces) <= set(slivers)
    # A UV FAULT OR A SLIVER, OR NOT THIS REASON. On 52 of 191 shipped meshes this failed, some near-zero tangents outlived
    # opening every collapsed face: a few on faces with UV area, read only from calc_tangents. No UV edit reaches them, so a
    # reason there could never clear.
    tangent_rest = 0 if on_slivers else zero_t - (tangents.get("collapsedUvInstances") or 0)
    zero_t -= tangent_rest
    if zero_t:
        fixes = [("clean_mesh", {"dissolveDegenerate": _sliver_fix_distance(shape["sliverWidthMax"])})] \
            if on_slivers else []
        if not on_slivers:
            # NOT a fresh unwrap: that replaces the whole layer, and on the shipped meshes this reason fails it changed the
            # UVs of every one (median UV area down to a quarter). Opening the collapsed faces moves nothing else.
            fixes.append(("clean_mesh", {"openCollapsedUvs": tangents.get("layer")}))
        fail("nearZeroTangents", zero_t, 0, "%d vertex instance(s) get nearly zero tangents in Unreal (tolerance 1E-4) "
             "from UV channel 0 '%s', on face(s) %s - Unreal warns on import and normal maps shade wrong there. %s"
             % (zero_t, tangents.get("layer"), t_faces[:12],
                "Every one is on a sliver face, so dissolving the slivers clears them" if on_slivers else
                "Their UVs collapse to a line or a point; opening those faces by a fraction of a texel gives them area "
                "and keeps every other UV (a fresh unwrap would replace the layer, so only for UVs nothing is painted "
                "on yet)"), *fixes)
    not_judged = []
    if waste is None:
        not_judged.append("wastedFaces: not examined past %d triangles" % _GR_WASTE_MAX_TRIS)
    elif wasted and spec["allowWaste"]:
        not_judged.append("wastedFaces: allowWaste is true")
    elif waste["colorsVary"] and waste["flatShare"] >= _GR_WASTE_FLAT_SHARE and waste["flatTris"] >= _GR_WASTE_FLAT_TRIS:
        not_judged.append("wastedFaces flat (%d): its vertices carry colors that vary, so a flat face's vertices have a use"
                          % waste["flatTris"])
    if tangent_rest:
        not_judged.append("nearZeroTangents on faces with UV area (%d): calc_tangents reads them near zero from the shape or "
                          "the shading normals, and no UV edit reaches them" % tangent_rest)
    if budget is None:
        not_judged.append("triangles (%d): no triangleBudget was given and the mesh has no size" % tris)
    if floating and spec["allowFloating"]:
        not_judged.append("floatingParts (%d): allowFloating is true" % len(floating))
    if shape.get("nearby"):
        not_judged.append("floatingParts nearby (%d): touching nothing, but too near the rest to call (the closest of "
                          "these stand under %g%% of the longest side away) - a leaf on its branch or a fitting on its "
                          "wall, and a chip that close reads the same; look if you cut this mesh"
                          % (len(shape["nearby"]), 100 * _GR_FLOAT_GAP_REL))
    if shape.get("noBody"):
        not_judged.append("floatingParts: %d group(s) of touching parts stand apart, but no group holds over %g%% of the "
                          "surface, so there is no body they should hang from (a bush, a set of pieces)"
                          % (shape["noBody"], 100 * _GR_BODY_SHARE))
    if shape.get("notExamined"):
        not_judged.append("floatingParts: %d small part(s) were not examined: at %d parts or more a mesh is a cloud of "
                          "cards or clutter (this one has %d)"
                          % (shape["notExamined"], _GR_SCATTER_PARTS, shape["parts"]))
    if spec["allowNgons"] and faces["ngons"]:
        not_judged.append("ngons (%d): allowNgons is true" % faces["ngons"])
    if not spec["uvOneTile"] and (checks.get("uv") or {}).get("loopsOutside01"):
        not_judged.append("uvLoopsOutside01 (%d): UVs past the 0-1 tile are right on a tiling texture; pass uvOneTile "
                          "true for a bake or an atlas, where they sample the wrong place"
                          % checks["uv"]["loopsOutside01"])
    if checks["boundaryEdges"]["count"]:
        not_judged.append("boundaryEdges (%d): an open edge is right on a sheet, a flag or a leaf"
                          % checks["boundaryEdges"]["count"])
    mods = [m.name for m in obj.modifiers]
    # A GIVEN BUDGET IS A CEILING, AND A PASS UNDER ONE SAID NOTHING OF WHAT THE SIZE USUALLY TAKES. A brief said "under 6,000" for a 95 cm cart and
    # 11 of the 20 carts built to it, up to 5,048 triangles, passed their own number and stood over the 2,642 most pieces that size stay under.
    # Three runs read both numbers and went on. The verdict does not change: the number given is the caller's, and this is said once beside it.
    if budget_from is not None and budget_from.get("from") == "given":
        usual = triangle_budget_for_size(obj)
        if usual is not None and tris > usual["triangleBudget"] and tris <= budget:
            not_judged.append("triangles against the size (%d): within the %d given, which is a ceiling and not a target, and %d over the %d "
                              "most %s pieces stay under; spend what changes the outline and cut what does not"
                              % (tris, budget, tris - usual["triangleBudget"], usual["triangleBudget"], usual["sizeClass"]))
    if mods:
        not_judged.append("%d modifier(s) not applied (%s) - the base mesh was judged; an export that applies them "
                          "ships different geometry" % (len(mods), ", ".join(mods[:8])))
    return {"pass": not reasons, "verdict": "pass" if not reasons else "fail",
            "failCount": len(reasons), "reasons": reasons,
            "measured": {"triangles": tris, "faces": faces["total"], "ngons": faces["ngons"], "nonManifoldEdges": nm,
                         "looseVerts": lv, "looseEdges": le, "uvLayer": (checks.get("uv") or {}).get("layer"),
                         "uvLoopsOutside01": (checks.get("uv") or {}).get("loopsOutside01"),
                         "scale": checks["transform"]["scale"], "windingFlippedEdges": flipped,
                         "insideOutParts": inside, "degenerateFaces": deg,
                         "parts": shape.get("parts"), "partGroups": shape.get("groups"), "floatingParts": floating[:20],
                         "floatingFaceIndices": (shape.get("floatingFaces") or [])[:200],
                         "floatingNearby": (shape.get("nearby") or [])[:20],
                         "sliverFaces": len(slivers), "sliverFaceIndices": slivers[:50],
                         "sliverWidthMaxM": shape.get("sliverWidthMax"),
                         "wastedFaces": waste,
                         "sliverLimits": {"widthUnderM": shape.get("sliverWidthLimitM"),
                                          "lengthOverWidthAtLeast": _GR_SLIVER_ASPECT} if shape else None,
                         "floatingLimits": {"areaUnderShareOfLargestPart": _GR_FLOAT_AREA_RATIO,
                                            "gapAtLeastShareOfLongestSide": _GR_FLOAT_GAP_REL,
                                            "groupBodyOverShareOfSurface": _GR_BODY_SHARE,
                                            "groupGapAtLeastShareOfLongestSide": _GR_GROUP_GAP_REL,
                                            "groupGapAtLeastShareOfOwnSize": _GR_GROUP_GAP_OWN,
                                            "groupGapAtLeastWhateverItsSize": _GR_GROUP_GAP_FAR,
                                            "boardGapAtLeastShareOfLongestSide": _GR_BOARD_GAP_REL,
                                            "boardGapAtLeastShareOfThickness": _GR_BOARD_GAP_THICK,
                                            "boardThicknessAtLeastShareOfLongestSide": _GR_BOARD_THICK_REL,
                                            "touchingWithinM": shape.get("touchM")} if shape else None,
                         "nearZeroTangentVertexInstances": zero_t + tangent_rest if tangents else None},
            "triangleBudget": budget, "triangleBudgetFrom": budget_from,
            "allowNgons": spec["allowNgons"], "uvOneTile": spec["uvOneTile"], "allowFloating": spec["allowFloating"],
            "notJudged": not_judged,
            "reachNote": "pass means none of these measured defects is present; it says nothing about how the model "
                         "looks"}
_KIT_HULL_PREFIXES = ("UCX_", "UBX_", "USP_", "UCP_")
# ONE KIT, ONE TEXEL DENSITY: pieces whose median pixels per meter differ by more than this ratio read as one blurry
# beside one sharp when they meet at a seam.
_KIT_DENSITY_RATIO = 1.25


def _kit_conventions(params):
    """mesh_quality over a collection: every piece's conventions and concerns, and what holds across the kit. Each
    piece goes through mesh_quality itself, so a piece reads exactly as it would alone."""
    clash = [k for k in _KIT_ONE_MESH_KEYS if params.get(k) is not None]
    if clash:
        raise MifOpError("collection measures a whole kit, and %s %s one mesh's - pass collection OR object, and the "
                         "one-mesh reports (boundaryLoops, bone, armature, poles, jointRings, facingAgainst, "
                         "flippedAgainst) only with object. NOTHING was measured."
                         % (", ".join(clash), "is" if len(clash) == 1 else "are"))
    conv = take_bool(params, "conventions", default=True)
    if conv is False:
        raise MifOpError("collection measures a kit's conventions, and conventions:false turns them off - drop it. "
                         "NOTHING was measured.")
    name = take(params, "collection", required=True, kind=str)
    coll = bpy.data.collections.get(name)
    if coll is None:
        raise MifOpError("no collection named '%s'. Collections: %s. NOTHING was measured."
                         % (name, ", ".join(sorted(c.name for c in bpy.data.collections)[:30]) or "(none)"))
    seen, objs = set(), []
    for c in [coll] + list(coll.children_recursive):
        for o in c.objects:
            if o.name not in seen:
                seen.add(o.name)
                objs.append(o)
    hulls = [o.name for o in objs if o.type == "MESH" and o.name.startswith(_KIT_HULL_PREFIXES)]
    hull_set = set(hulls)
    skipped = [{"object": o.name, "type": o.type} for o in objs if o.type != "MESH"]
    pieces = [o for o in objs if o.type == "MESH" and o.name not in hull_set]
    if not pieces:
        raise MifOpError("collection '%s' (with its child collections) holds no mesh piece - %d object(s), %d of them "
                         "collision hulls - so there is no kit to measure, and an empty report would read as a clean "
                         "one. NOTHING was measured." % (name, len(objs), len(hulls)))
    sub = {k: params[k] for k in ("kind", "module", "texelDensityFor", "uvLayer") if params.get(k) is not None}
    rows, by_finding, medians, unmeasured = [], {}, [], []
    for o in sorted(pieces, key=lambda x: x.name):
        try:
            q = op_mesh_quality(dict(sub, object=o.name, conventions=True))
        except MifOpError as exc:
            # ONE PIECE THAT CANNOT BE MEASURED (no vertices, no such UV layer) is listed, not the kit refused
            unmeasured.append({"object": o.name, "error": str(exc).replace(" NOTHING was measured.", "")})
            continue
        conv = q.get("conventions") or {}
        differs = sorted(k for k, v in conv.items() if isinstance(v, dict) and v.get("follows") is False)
        for k in differs:
            by_finding.setdefault(k, []).append(o.name)
        td = (q.get("checks") or {}).get("texelDensity") or {}
        med = td.get("pixelsPerMetreMedian")
        if med:
            medians.append((med, o.name))
        rows.append({"object": o.name, "kind": (conv.get("kind") or {}).get("value"),
                     "sizeCm": ((conv.get("sizeClass") or {}).get("measured") or {}).get("sizeCm"),
                     "followCount": conv.get("followCount"), "differCount": conv.get("differCount"),
                     "differs": differs, "notes": conv.get("notes") or [], "concerns": q.get("concerns") or [],
                     "texelDensityMedian": med, "notMeasured": sorted(q.get("notMeasured") or {})})
    if not rows:
        raise MifOpError("collection '%s' has %d mesh piece(s) and none could be measured: %s. NOTHING was measured."
                         % (name, len(unmeasured), "; ".join("%s: %s" % (u["object"], u["error"]) for u in unmeasured[:5])))
    lower = {}
    for o in pieces:
        lower.setdefault(o.name.lower(), []).append(o.name)
    case_clash = sorted(sorted(v) for v in lower.values() if len(v) > 1)
    density = None
    if medians:
        medians.sort()
        lo, hi = medians[0], medians[-1]
        ratio = hi[0] / lo[0] if lo[0] > 0 else None
        density = {"pixelsPerMetreMin": lo[0], "lowest": lo[1], "pixelsPerMetreMax": hi[0], "highest": hi[1],
                   "ratio": round(ratio, 3) if ratio else None,
                   "follows": bool(ratio is not None and ratio <= _KIT_DENSITY_RATIO),
                   "rule": "one kit, one texel density: every piece's median within %g x of the others'"
                           % _KIT_DENSITY_RATIO,
                   "piecesMeasured": len(medians)}
    notes = []
    if density and not density["follows"]:
        notes.append("texel density runs %.1f to %.1f px/m (%s to %s), %.2f x - pieces of one kit meet at seams, and "
                     "the lower one reads blurry beside the other (uv_unwrap BOX with one metersPerTile evens it)"
                     % (density["pixelsPerMetreMin"], density["pixelsPerMetreMax"], density["lowest"],
                        density["highest"], density["ratio"]))
    for group in case_clash:
        notes.append("%s differ only by case - Unreal's asset names do not, so they would land on one asset"
                     % " and ".join(group))
    for k in sorted(by_finding):
        notes.append("%s differs on %d of %d piece(s): %s" % (k, len(by_finding[k]), len(rows),
                                                               ", ".join(by_finding[k][:12])))
    return {
        "collection": coll.name, "pieceCount": len(rows), "hulls": sorted(hulls), "skipped": skipped,
        "pieces": rows, "unmeasured": unmeasured,
        "kit": {"piecesFollowingAll": sum(1 for r in rows if not r["differs"]),
                "differByFinding": {k: v for k, v in sorted(by_finding.items())},
                "namesClashingWithoutCase": case_clash, "texelDensity": density,
                "notes": notes},
        "reachNote": ("each piece reads as mesh_quality conventions reads it alone; kit adds what only shows across "
                      "pieces. A convention is a note with its number, not a verdict - and nothing here says how the "
                      "kit looks."),
    }


def op_mesh_quality(params):
    """Measure the objectively checkable things that get an asset rejected.

    NOT A VERDICT ON THE ART. Everything here is a number a machine can defend: topology, UV bounds,
    texel density spread, applied transforms, loose and degenerate geometry. Whether the thing is
    beautiful is not in scope and is not claimed.

    params:
      object (str, required)   the mesh to measure
      uvLayer (str)            which UV layer to judge; default is the active one
      texelDensityFor (float | [w, h])  texture size in pixels used for the density figure, default 1024; a non-square
                               [w, h] gives sqrt(uv area x w x h / world area) per face
      conventions (bool)       add a `conventions` block: how the piece holds to the way game pieces are
                               usually built - size class, kind, pivot, UV layers and the lightmap's place,
                               sides on a module, triangles for the size, material slots and their names,
                               thickness, applied transform, collision hulls and the name. Each finding carries
                               what was MEASURED and `follows` (true, false, or null where the convention does
                               not decide); none of them is added to concerns. Default false. Implied by kind or
                               module.
      kind (str)               auto (default: read off the bounds), wall, floor or prop - which pivot convention
                               the piece is held to
      module (float)           the kit's module in CENTIMETERS; the long sides are checked against whole
                               multiples of it
      boundaryLoops (bool)     add `boundaryLoops`: each open boundary loop as data - edges, vertices, closed, length,
                               centroid and box in WORLD units, biggest first (50 listed, loopCount exact). Which loop
                               is a torn cuff took a recorded script (loops_report.py, 2026-10-04).
      bone (str)               with boundaryLoops: each loop's span along this bone's rest axis (from, to, span, in
                               world units from the bone's head) - a cuff ring spans about 0, a cuff torn open along
                               the forearm 14 cm
      armature (str)           the bone's armature; default the mesh's Armature modifier's object, else its parent
      poles (bool)             add `poles`: interior vertices of valence 3 and of 5 or more, with indices (2026-10-04)
      jointRings {bone, armature, range}   add `jointRings`: the closed edge loops that go round the bone's axis within
                               range (0.1 world units) of its head - a clean bend wants closed rings at the neck,
                               shoulders and leg roots
      facingAgainst (str | {body, bandStep, reach})   add `facingAgainst`: the faces whose normal points toward the
                               nearest point of the body (within reach, default 0.2 m), counted by height band
                               (bandStep, default 0.1 m) - a rebuilt cuff and hem band that faced inward showed as holes
                               under a one-sided material (2026-10-04: 25.4% inward before the fix, 6.0% after, the rest
                               folded-under edges that face the body by design)
      collection (str)         IN PLACE OF object: the conventions over a whole kit - every mesh in the collection and
                               its child collections (collision hulls named UCX_, UBX_, USP_ or UCP_ are counted as
                               hulls, not pieces), each piece's conventions and concerns, then what holds or breaks
                               across the kit: one texel density (the pieces' medians within 25%), names that are
                               equal without case (one asset in Unreal), and every finding that differs with the
                               pieces it differs on. Takes conventions, kind, module, texelDensityFor and uvLayer;
                               the one-mesh reports are refused with it. An empty collection is refused.
      gameReady {triangleBudget, allowNgons, uvOneTile, allowFloating, allowWaste}   add `gameReady`: pass or fail for a game engine (1.4), each reason a
                               measured number with its limit and the call that fixes it - triangles over the budget,
                               n-gons (unless allowNgons), non-manifold and loose geometry, a missing UV layer, UVs
                               outside 0-1 (only with uvOneTile, for a bake or an atlas: tiling needs them outside),
                               unapplied scale, inconsistent normals and degenerate faces. Read off this
                               op's own checks. A mesh with no faces, or one in edit mode, is refused, not passed.
                               SINCE 1.4's second pass: with no triangleBudget the budget is the top of the usual
                               range for the size class (conventions' triangles for the size; triangleBudgetFrom says
                               so), and three more reasons - floatingParts (parts under 10% of the largest part's
                               surface, touching nothing and 2% or more of the longest side away, or a group of touching
                               parts under 10% of a body holding over half the surface and standing nearest it, 1% of the
                               longest side and 5% of its own size away, or 3% whatever its size, or a solid 0.4% away and
                               20% of its thickness, the gap measured exactly; no fix is named, only
                               `choices`, since a delete sent blind takes wanted parts), sliverFaces (narrower than 1/10000 of the mesh and 100 times
                               longer than wide) and nearZeroTangents (checks.tangents, what Unreal warns about)
      flippedAgainst (str)     add `flippedAgainst`: the faces whose normal turned over against the same face of a
                               reference copy (the mesh before a reshape, same faces vertex for vertex) - a fold
    """
    reject_unknown(params, MESH_QUALITY_PARAMS, "mesh_quality")
    if params.get("collection") is not None:
        return _kit_conventions(params)
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    want_loops = take_bool(params, "boundaryLoops", default=False)
    want_poles = take_bool(params, "poles", default=False)
    joint = _joint_spec(params.get("jointRings"), obj)
    from .ops_garment import facing_report, flipped_report, parse_facing, parse_flipped  # noqa: PLC0415 - there
    facing = parse_facing(params["facingAgainst"], obj) if params.get("facingAgainst") is not None else None
    flip_ref = parse_flipped(params["flippedAgainst"], obj) if params.get("flippedAgainst") is not None else None
    loop_bone = _loop_bone_axis(obj, take(params, "bone", default=None, kind=str),
                                take(params, "armature", default=None, kind=str), want_loops)
    # THE CONVENTION KEYS, READ AND CHECKED BEFORE ANYTHING IS MEASURED.
    # take_bool has no "absent" answer, so absence is read first: absent, true and false are three different asks.
    conv_asked = take_bool(params, "conventions") if params.get("conventions") is not None else None
    conv_kind = (take(params, "kind", default=None, kind=str) or "").lower() or None
    conv_module = take_float(params, "module")
    if conv_asked is False and (conv_kind or conv_module is not None):
        raise MifOpError("kind and module are read only for the conventions block, and conventions:false turns it "
                         "off - they would be ignored. Drop them, or drop conventions:false. NOTHING was measured.")
    if conv_kind is not None and conv_kind not in ("auto", "wall", "floor", "prop"):
        raise MifOpError("kind must be auto, wall, floor or prop; got %r. NOTHING was measured." % conv_kind)
    if conv_module is not None and not (math.isfinite(conv_module) and conv_module > 0.0):
        raise MifOpError("module is the kit's module in CENTIMETERS and must be a finite number above 0; got %r. "
                         "NOTHING was measured." % conv_module)
    want_conventions = bool(conv_asked) or conv_kind is not None or conv_module is not None
    # GAME READY (1.4), read and refused before anything is measured: a verdict on a stale or empty mesh would be a pass
    # about nothing.
    game_ready = _game_ready_spec(params.get("gameReady"))
    if game_ready is not None and getattr(obj, "mode", "OBJECT") != "OBJECT":
        raise MifOpError("'%s' is in %s mode, so its mesh data is the state before the edits now open - gameReady will "
                         "not judge that. Leave edit mode and call again. NOTHING was measured." % (obj.name, obj.mode))
    if game_ready is not None and not len(obj.data.polygons):
        raise MifOpError("'%s' has no faces (%d vertices, %d edges), so gameReady has nothing to judge - that is not a "
                         "pass. NOTHING was measured." % (obj.name, len(obj.data.vertices), len(obj.data.edges)))
    # THICKNESS (2026-10-11), read before anything is measured: thickness true or {method, samples, seed}, and
    # thicknessMinCm, the caller's threshold - the tool never picks the number (thickness_report).
    thick_min = take_float(params, "thicknessMinCm")
    thick_ask = parse_thickness(params.get("thickness"), thick_min)

    # A read off mesh.polygons is STALE in edit mode - the live data is in a BMesh nobody has
    # written back. Reporting numbers from it would describe the mesh as it was before the user
    # started editing, confidently.
    stale = edit_mode_stale(obj)

    raw_tex = params.get("texelDensityFor")
    if isinstance(raw_tex, (list, tuple)):
        if len(raw_tex) != 2 or any(isinstance(v, bool) for v in raw_tex):
            raise MifOpError("texelDensityFor takes a size or [w, h] in PIXELS - got %r. NOTHING was measured."
                             % (raw_tex,))
        tex_wh = finite_floats(list(raw_tex), "texelDensityFor")
        tex_report = [tex_wh[0], tex_wh[1]]
    else:
        tex_px = finite_float(take_float(params, "texelDensityFor", default=1024.0), "texelDensityFor")
        tex_wh = [tex_px, tex_px]
        tex_report = tex_px
    if min(tex_wh) <= 0:
        raise MifOpError("texelDensityFor is a texture size in PIXELS and must be positive - got "
                         "%r. NOTHING was measured." % (raw_tex,))
    tex_px = (tex_wh[0] * tex_wh[1]) ** 0.5

    mesh = obj.data
    skipped = {}
    checks = {}
    concerns = []
    # THE CONVENTIONS FIRST, because they can refuse ("has no vertices") and the tangent check below makes and removes
    # a scratch mesh: a refusal after it would promise "NOTHING was measured" past a write. They read only the object.
    conventions = (_conventions(obj, None if conv_kind in (None, "auto") else conv_kind, conv_module)
                   if want_conventions else None)
    # FACING TOO, for the same reason: it can refuse (nothing within reach), so it runs before the scratch mesh.
    facing_block = facing_report(obj, *facing) if facing is not None else None

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()

        # ---- topology -------------------------------------------------------------------
        tris = quads = ngons = 0
        degenerate = []
        world = obj.matrix_world
        world_area = 0.0
        for f in bm.faces:
            n = len(f.verts)
            if n == 3:
                tris += 1
            elif n == 4:
                quads += 1
            else:
                ngons += 1
            # Area in WORLD space, because a metre is what texel density is per - an object scaled
            # 100x has the same local area and a hundredth of the density.
            a = f.calc_area() * (world.to_scale().x * world.to_scale().y)
            world_area += a
            if a < _DEGENERATE_FACE_AREA:
                degenerate.append(f.index)

        checks["faces"] = {"total": len(bm.faces), "tris": tris, "quads": quads, "ngons": ngons}
        checks["degenerateFaces"] = {"count": len(degenerate), "indices": degenerate[:50]}

        # An edge on 0 faces is loose; on 1 it is a boundary (legitimate on an open mesh); on 3+ it
        # is non-manifold and will confuse every baker and engine downstream. Counted separately
        # because only the last is unambiguously wrong.
        loose_edges = [e.index for e in bm.edges if len(e.link_faces) == 0]
        boundary = [e.index for e in bm.edges if len(e.link_faces) == 1]
        nonmanifold = [e.index for e in bm.edges if len(e.link_faces) > 2]
        loose_verts = [v.index for v in bm.verts if len(v.link_edges) == 0]
        checks["looseVerts"] = {"count": len(loose_verts), "indices": loose_verts[:50]}
        checks["looseEdges"] = {"count": len(loose_edges), "indices": loose_edges[:50]}
        checks["boundaryEdges"] = {"count": len(boundary),
                                   "note": "legitimate on an open mesh; a closed one should have 0"}
        boundary_loops = _boundary_loop_rows(bm, world, loop_bone) if want_loops else None
        checks["nonManifoldEdges"] = {"count": len(nonmanifold), "indices": nonmanifold[:50]}

        # ---- winding (2026-10-06) -------------------------------------------------------------
        # A band of faces wound the other way is invisible to every count above: a lathe that turned each face away
        # from its profile's centroid left 20 % of a tapered can's outer wall facing in, closed and manifold, and
        # Unreal drew that band see-through. Two faces sharing an edge must walk it in opposite directions; where they
        # walk it the same way, one of them is flipped. And a closed part whose faces all agree can still be wound
        # inside out as a whole - its signed volume is then negative.
        flipped_edges, inside_out = _winding_scan(bm)
        checks["windingFlippedEdges"] = {
            "count": len(flipped_edges), "indices": flipped_edges[:50],
            "note": "edges whose two faces walk them the same way - one face of the pair points the other way"}
        checks["insideOutParts"] = {
            "count": len(inside_out), "parts": inside_out[:20],
            "note": "closed parts whose faces agree with each other and whose signed volume is negative - the whole "
                    "part faces in"}

        # ---- UVs ------------------------------------------------------------------------
        want_layer = take(params, "uvLayer", default=None, kind=str)
        uv_layer = None
        if not bm.loops.layers.uv:
            skipped["uv"] = ("this mesh has NO UV layer, so UV bounds and texel density were not "
                             "measured - that is an absence of data, not a pass")
        else:
            if want_layer:
                uv_layer = bm.loops.layers.uv.get(want_layer)
                if uv_layer is None:
                    raise MifOpError(
                        "no UV layer named '%s' on '%s'. Layers: %s. NOTHING was measured."
                        % (want_layer, obj.name, ", ".join(bm.loops.layers.uv.keys()) or "(none)"))
            else:
                uv_layer = bm.loops.layers.uv.verify()

        if uv_layer is not None:
            outside = 0
            uv_total = 0.0
            densities = []
            for f in bm.faces:
                ua = _uv_area(f, uv_layer)
                uv_total += ua
                for loop in f.loops:
                    u, v = loop[uv_layer].uv
                    if (u < -_UV_BOUND_EPS or u > 1.0 + _UV_BOUND_EPS
                            or v < -_UV_BOUND_EPS or v > 1.0 + _UV_BOUND_EPS):
                        outside += 1
                wa = f.calc_area() * (world.to_scale().x * world.to_scale().y)
                if wa > _DEGENERATE_FACE_AREA and ua > 0.0:
                    # px per metre: the texture edge in pixels times the UV edge length, over the
                    # world edge length. Using areas, that is sqrt(uv_area)/sqrt(world_area)*px.
                    densities.append((ua ** 0.5) / (wa ** 0.5) * tex_px)

            checks["uv"] = {
                "layer": uv_layer.name if hasattr(uv_layer, "name") else str(want_layer or "active"),
                "loopsOutside01": outside,
                "uvAreaTotal": round(uv_total, 6),
                "note": ("uvAreaTotal well above 1.0 means islands overlap or spill outside the "
                         "tile; it does NOT prove overlap on its own, and this op does not do "
                         "island intersection."),
            }
            judged = checks["uv"]["layer"]
            deg = uv_degeneracy(mesh, judged)
            checks["uv"].update({"nearZeroUvAreaFaces": deg.get("nearZeroUvAreaFaces"),
                                 "nearZeroUvAreaFaceIndices": deg.get("nearZeroUvAreaFaceIndices") or None,
                                 "nearZeroUvAreaThreshold": NEAR_ZERO_UV_AREA})
            # TANGENTS COME FROM UV CHANNEL 0 IN UNREAL, whichever layer is judged above.
            uv0 = mesh.uv_layers[0].name
            checks["tangents"] = dict(deg["tangents"] if uv0 == judged else uv_degeneracy(mesh, uv0).get("tangents")
                                      or {}, layer=uv0)
            if densities:
                densities.sort()
                lo, hi = densities[0], densities[-1]
                mid = densities[len(densities) // 2]
                checks["texelDensity"] = {
                    "pixelsPerMetreMin": round(lo, 2),
                    "pixelsPerMetreMedian": round(mid, 2),
                    "pixelsPerMetreMax": round(hi, 2),
                    "spreadRatio": round(hi / lo, 2) if lo > 0 else None,
                    "forTextureSize": tex_report,
                    "note": ("spreadRatio is max/min across faces. A uniform unwrap sits near 1; a "
                             "large number means some faces get far more texture than others, "
                             "which is what makes one part of a model look blurry beside another."),
                }
            else:
                skipped["texelDensity"] = ("no face had both a non-zero world area and a non-zero "
                                           "UV area, so density is undefined rather than 0")
    finally:
        bm.free()

    # ---- transforms -----------------------------------------------------------------------
    sc = tuple(round(v, 6) for v in obj.scale)
    rot = tuple(round(v, 6) for v in obj.rotation_euler)
    checks["transform"] = {
        "scale": list(sc),
        "scaleApplied": sc == (1.0, 1.0, 1.0),
        "rotationEuler": list(rot),
        # THE ROTATION THE OBJECT'S MODE USES, not rotation_euler whatever the mode (2026-09-28, the same blindness the
        # 2026-09-21 pack-rule review found in object_info): a glTF object is QUATERNION, ignores rotation_euler, and
        # read rotationApplied true while turned 90 degrees.
        "rotationMode": obj.rotation_mode,
        "rotationApplied": rotation_is_identity(obj),
        "note": ("unapplied scale is the most common export surprise: the mesh looks right in "
                 "Blender and arrives in the engine at a different size or with broken normals."),
    }

    slots = [s.material.name if s.material else None for s in obj.material_slots]
    checks["materials"] = {"slots": len(slots), "empty": sum(1 for s in slots if s is None),
                           "names": [s for s in slots if s]}

    # ---- the only judgements made, and each one is defensible -------------------------------
    if ngons:
        concerns.append("%d ngon(s) - many stores and engines require tris or quads only" % ngons)
    if nonmanifold:
        concerns.append("%d non-manifold edge(s) - these break bakes, booleans and collision"
                        % len(nonmanifold))
    if checks["windingFlippedEdges"]["count"]:
        concerns.append("%d edge(s) between faces wound opposite ways - a band of faces points the other way, and a "
                        "one-sided material draws it see-through" % checks["windingFlippedEdges"]["count"])
    if checks["insideOutParts"]["count"]:
        concerns.append("%d closed part(s) wound inside out (negative signed volume) - every face points in"
                        % checks["insideOutParts"]["count"])
    if loose_verts or loose_edges:
        concerns.append("%d loose vert(s) and %d loose edge(s) - geometry that renders as nothing "
                        "and still costs file size" % (len(loose_verts), len(loose_edges)))
    if degenerate:
        concerns.append("%d zero-area face(s) - they export, shade unpredictably and show as "
                        "artefacts" % len(degenerate))
    if not checks["transform"]["scaleApplied"]:
        concerns.append("scale is not applied (%s) - apply it before export" % (list(sc),))
    if checks["materials"]["empty"]:
        concerns.append("%d empty material slot(s)" % checks["materials"]["empty"])
    if "uv" in checks and checks["uv"]["loopsOutside01"]:
        concerns.append("%d UV loop(s) outside the 0-1 tile" % checks["uv"]["loopsOutside01"])
    zero_t = (checks.get("tangents") or {}).get("nearZeroVertexInstances")
    if zero_t:
        concerns.append("%d vertex instance(s) will get nearly zero tangents in Unreal (tolerance 1E-4) from UV channel 0 "
                        "'%s' - Unreal warns on import and shades them wrongly; faces whose UVs collapse to a line or a "
                        "point, listed in checks.tangents.faces" % (zero_t, checks["tangents"]["layer"]))

    out = {
        "object": obj.name,
        "checks": checks,
        "concerns": concerns,
        "concernCount": len(concerns),
        # REACH, NOT GREEN. An empty concerns list next to a skipped check is not a clean bill of
        # health, and saying so here is what stops it being read as one.
        "notMeasured": skipped,
        "reachNote": ("concerns lists only defects with an objective threshold. It says NOTHING "
                      "about whether the asset looks good, and anything in notMeasured was not "
                      "judged at all - an absence of data rather than a pass."),
    }
    if game_ready is not None:
        out["gameReady"] = _game_ready_verdict(obj, checks, game_ready)
    if conventions is not None:
        out["conventions"] = conventions
    if boundary_loops is not None:
        out["boundaryLoops"] = boundary_loops
    if want_poles or joint is not None:
        pbm = bmesh.new()
        pbm.from_mesh(obj.data)
        try:
            if want_poles:
                out["poles"] = _poles_report(pbm)
            if joint is not None:
                out["jointRings"] = _joint_rings(pbm, obj.matrix_world, joint)
        finally:
            pbm.free()
    if facing_block is not None:
        out["facingAgainst"] = facing_block
    if flip_ref is not None:
        out["flippedAgainst"] = flipped_report(obj, flip_ref)
    if thick_ask is not None:
        out["thickness"] = thickness_report(obj, thick_ask[0], thick_ask[1], thick_ask[2], thick_min)
    out.update(stale)
    out.update(shared_data_note(obj))
    return out

# ---------------------------------------------------------------------------
# recipe_game_ready - the boring pipeline, banked
# ---------------------------------------------------------------------------

_RECIPE_KINDS = ("auto", "wall", "floor", "prop")
# The pivot each kind is placed by (FINDINGS: a floor module tiles from a corner, a wall snaps by the bottom of a face,
# a prop stands on its bottom center). "kind" picks by the piece's kind; the others name one outright.
_RECIPE_PIVOTS = ("kind", "bottom", "bottomCorner", "bottomSide")
_RECIPE_COLLISION = ("auto", "convex", "box", "none")
_RECIPE_LIGHTMAP_LAYER = "Lightmap"


def _ue_mesh_name(name):
    """SM_<name> with every run of characters Unreal will not keep in an asset name made one underscore; a name
    already starting SM_ is kept as it is."""
    if name.startswith("SM_"):
        return name
    base = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return ("SM_" + base) if base else None


def _hulls_named_for(name):
    """The UCX_/UBX_/USP_/UCP_<name>_NN meshes in the file, sorted - the hulls the engine attaches to `name`."""
    hull_re = re.compile(r"^(%s)_%s_(\d+)$" % ("|".join(_CONV_HULL_PREFIXES), re.escape(name)))
    return sorted(o.name for o in bpy.data.objects if o.type == "MESH" and hull_re.match(o.name))


def op_recipe_game_ready(params):
    """Apply transforms, ensure UVs, and MEASURE the result - or, asked for, the whole pass a piece needs before it
    leaves for Unreal.

    Composes ops that already exist rather than reimplementing them, so every guard they carry -
    shared mesh data, linked libraries, edit mode - still applies. It does nothing a caller could not
    do by hand; it does it in the right order and then checks.

    params:
      object (str, required)
      applyTransform (bool)  bake loc/rot/scale into the mesh data. Default true.
      unwrap (bool)          unwrap when there is no UV layer. Default true.
      forceUnwrap (bool)     unwrap even if a layer already exists. Default FALSE - re-unwrapping a
                             mesh somebody already laid out by hand is destructive and silent.
      uvMethod (str)         SMART, ANGLE or BOX (1 m per tile, OBJECT space), default SMART.

    THE FULL PASS - each step off unless asked for, so a call without these keys does what it always did:
      full (bool)            every step below at its default: pivot 'kind', lightmap, collision 'auto' one hull
                             per part, ueName 'auto'. Any key given beside it wins.
      kind (str)             auto (default: read off the bounds after the transform is baked), wall, floor or
                             prop - what the pivot, the hull shape and the conventions check go by.
      pivot (str)            'kind' (floor: bottomCorner -x-y; wall: bottomSide on the face across its
                             thickness, -y for a wall running along X; prop: bottom center), or bottom,
                             bottomCorner, bottomSide outright. Read off the vertices, never the stale bound_box.
      lightmap (bool)        a lightmap layer made LAST with uv_unwrap LIGHTMAP, named 'Lightmap', and measured by
                             uv_info lightmapReady. A lightmap layer that is already last and ready is kept; one
                             that is last and not ready is laid out again. UV0 stays the active layer.
      collision (str)        auto (box for a wall or floor, convex for a prop), convex, box or none, through
                             create_collision_hull - after the rename, so the hulls carry the final name.
                             Hulls already named for the object are kept, and the step says so.
      perPart (bool)         one hull per loose part (default true under full, false otherwise)
      maxHulls (int)         perPart only: at most this many hulls
      ueName (str)           'auto' puts SM_ in front of the name (every run of other characters one
                             underscore), or the exact name to give it. Hulls already named for the old
                             name are renamed with it.
      module (float)         the kit's module in CENTIMETERS, for the conventions check

    With any of the full-pass keys the measure step asks mesh_quality for its conventions block, and
    the answer's conventionNotes lists each difference in a sentence. A step that fails raises, naming
    every step that already ran - nothing is undone.
    """
    reject_unknown(params, {"object", "name", "applyTransform", "unwrap", "forceUnwrap",
                            "uvMethod", "full", "kind", "pivot", "lightmap", "collision", "perPart", "maxHulls",
                            "ueName", "module"}, "recipe_game_ready")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)

    do_apply = take_bool(params, "applyTransform", default=True)
    do_unwrap = take_bool(params, "unwrap", default=True)
    force_unwrap = take_bool(params, "forceUnwrap", default=False)
    uv_method = str(take(params, "uvMethod", default="SMART", kind=str)).upper()

    # ---- the full pass's keys, every one checked BEFORE anything is changed ----------------------------------------
    full = take_bool(params, "full", default=False)
    kind_arg = (take(params, "kind", kind=str) or "").lower() or None
    if kind_arg is not None and kind_arg not in _RECIPE_KINDS:
        raise MifOpError("kind must be one of %s; got %r. NOTHING was changed." % (", ".join(_RECIPE_KINDS), kind_arg))
    pivot = take(params, "pivot", kind=str)
    if pivot is not None:
        pivot = {p.lower(): p for p in _RECIPE_PIVOTS}.get(pivot.lower(), pivot)
        if pivot not in _RECIPE_PIVOTS:
            raise MifOpError("pivot must be one of %s ('kind' picks by the piece's kind); got %r. NOTHING was "
                             "changed." % (", ".join(_RECIPE_PIVOTS), pivot))
    elif full:
        pivot = "kind"
    want_lightmap = take_bool(params, "lightmap", default=full)
    collision = take(params, "collision", kind=str)
    if collision is not None:
        collision = collision.lower()
        if collision not in _RECIPE_COLLISION:
            raise MifOpError("collision must be one of %s; got %r. NOTHING was changed."
                             % (", ".join(_RECIPE_COLLISION), collision))
    elif full:
        collision = "auto"
    if collision == "none":
        collision = None
    per_part = take_bool(params, "perPart", default=full and collision is not None)
    max_hulls = take_int(params, "maxHulls", default=None)
    if collision is None and (params.get("perPart") is not None or max_hulls is not None):
        raise MifOpError("perPart and maxHulls shape the collision step, and no collision was asked for (pass "
                         "collision, or full). NOTHING was changed.")
    if max_hulls is not None and not per_part:
        raise MifOpError("maxHulls is read only with perPart:true. NOTHING was changed.")
    if max_hulls is not None and max_hulls < 1:
        raise MifOpError("maxHulls must be at least 1, got %d. NOTHING was changed." % max_hulls)
    ue_name = take(params, "ueName", kind=str)
    if ue_name is None and full:
        ue_name = "auto"
    module_cm = take_float(params, "module")
    if module_cm is not None and not module_cm > 0:
        raise MifOpError("module is the kit's module in centimeters and must be above 0, got %r. NOTHING was "
                         "changed." % module_cm)
    full_pass = (full or want_lightmap or any(v is not None for v in (kind_arg, pivot, collision, ue_name,
                                                                  module_cm)))

    me = obj.data
    uv_names = [uv.name for uv in me.uv_layers]
    lm_named = [i for i, n in enumerate(uv_names) if _CONV_LIGHTMAP_NAME.search(n)]
    if want_lightmap:
        if not uv_names and not do_unwrap:
            raise MifOpError("a lightmap layer goes AFTER UV0, and '%s' has no UV layer and unwrap is false - the "
                             "lightmap would become UV0, the layer the texture reads. Let the recipe unwrap, or add "
                             "UV0 first. NOTHING was changed." % obj.name)
        if lm_named and (lm_named[-1] != len(uv_names) - 1 or len(uv_names) == 1):
            raise MifOpError("'%s' has a lightmap layer '%s' at channel %d of %d, and Unreal's lightmap index points "
                             "at a channel by position: the lightmap goes LAST, after UV0. Blender cannot move a UV "
                             "layer; remove it (set_uv_layer remove) and let the recipe make one. NOTHING was "
                             "changed." % (obj.name, uv_names[lm_named[-1]], lm_named[-1], len(uv_names)))
        if not lm_named and len(uv_names) >= _UV_LAYER_CAP:
            raise MifOpError("'%s' already has %d UV layers, Blender's maximum, so there is no room for a lightmap "
                             "layer. NOTHING was changed." % (obj.name, len(uv_names)))
    new_name = None
    if ue_name is not None:
        new_name = _ue_mesh_name(obj.name) if ue_name.lower() == "auto" else ue_name.strip()
        if not new_name:
            raise MifOpError("ueName %r leaves no name to give '%s'. NOTHING was changed." % (ue_name, obj.name))
        bad = sorted({c for c in new_name if not (c.isalnum() or c == "_")})
        if bad:
            raise MifOpError("ueName %r holds %r - an asset name keeps to letters, digits and underscores, and "
                             "Unreal refuses several of those characters in an object name. NOTHING was changed."
                             % (new_name, bad))
        if new_name != obj.name:
            clash = bpy.data.objects.get(new_name)
            if clash is not None:
                raise MifOpError("an object named '%s' already exists, so '%s' cannot take the name. Rename or "
                                 "delete that one first. NOTHING was changed." % (new_name, obj.name))
            taken = [h for h in _hulls_named_for(obj.name)
                     if bpy.data.objects.get(new_name.join(h.rsplit(obj.name, 1))) is not None]
            if taken:
                raise MifOpError("the hull(s) %s travel with '%s' and are renamed with it, and a hull of that new "
                                 "name already exists. NOTHING was changed." % (taken, obj.name))
            if len(new_name) > _ID_NAME_LIMIT and bpy.app.version < (5, 0, 0):
                raise MifOpError("'%s' is %d characters and this Blender truncates names at %d. NOTHING was "
                                 "changed." % (new_name, len(new_name), _ID_NAME_LIMIT))

    steps = []
    left_behind = []
    recipe_concerns = []
    kept_uv = None
    passed = {}

    def _record(name, changed, detail=""):
        steps.append({"step": name, "changed": bool(changed), "detail": detail})

    def _stopped(step, exc):
        """The error for a step that failed, naming what the steps before it already did - none of it is undone."""
        done = [s for s in steps if s.get("changed")]
        for s in done:
            if s["step"] == "applyTransform":
                left_behind.append(
                    "applyTransform ALREADY RAN and its result is baked into the mesh data - the "
                    "object's scale and rotation are now identity and that is NOT undone by this "
                    "failure. The mesh is in a valid state, just further along than you asked for.")
            else:
                left_behind.append("%s ALREADY RAN (%s) and is not undone." % (s["step"], s["detail"]))
        return MifOpError("recipe stopped at %s: %s%s"
                          % (step, exc, (" WHAT IS LEFT BEHIND: " + " ".join(left_behind)) if left_behind else ""))

    # ---- 1. transforms ------------------------------------------------------------------
    # WHY FIRST. Unwrapping before the scale is baked lays UVs out against the unscaled mesh, so a
    # non-uniformly scaled object gets a texel density that is wrong the moment the scale is applied.
    # Order is the entire value of a recipe; doing the same three ops in the wrong one is why this
    # keeps being got wrong by hand.
    if do_apply:
        before_scale = tuple(round(v, 6) for v in obj.scale)
        # rotation_is_identity READS THE ROTATION THE MODE USES - a QUATERNION object (every glTF import) turned 90
        # degrees has rotation_euler (0, 0, 0), and this skipped its bake as "already identity" until 2026-09-28. The
        # rotation named in the step is the one the mode uses, for the same reason.
        before_rot = tuple(round(v, 6) for v in ({"QUATERNION": obj.rotation_quaternion,
                                                  "AXIS_ANGLE": obj.rotation_axis_angle}.get(obj.rotation_mode,
                                                                                             obj.rotation_euler)))
        if before_scale == (1.0, 1.0, 1.0) and rotation_is_identity(obj):
            _record("applyTransform", False, "already identity - nothing to bake")
        else:
            try:
                op_apply_transform({"object": obj.name, "location": False,
                                    "rotation": True, "scale": True})
                _record("applyTransform", True,
                        "baked scale %s and rotation %s (%s) into the mesh data"
                        % (list(before_scale), list(before_rot), obj.rotation_mode))
            except MifOpError as exc:
                # NOTHING HAS BEEN CHANGED YET at this point, so the recipe can stop cleanly and
                # say so. That is worth stating rather than leaving the caller to infer it.
                raise MifOpError(
                    "recipe stopped at applyTransform and NOTHING was changed: %s" % exc)
    else:
        _record("applyTransform", False, "skipped - applyTransform was false")

    # ---- 1b. the kind, read off the baked mesh - a turned wall is a wall only once its turn is in its bounds ----
    kind = None
    if full_pass:
        lo, hi = world_vertex_bounds(obj)
        if lo is None:
            raise _stopped("kind", "'%s' has no vertices, so it has no bounds to read a kind or place a pivot by"
                           % obj.name)
        size = [(hi[k] - lo[k]) * UU_PER_BU for k in range(3)]
        inferred = _conv_kind_of(size)
        kind = kind_arg if kind_arg not in (None, "auto") else inferred
        passed["kind"] = {"value": kind, "inferred": inferred, "sizeCm": [round(s, 2) for s in size]}

    # ---- 2. the pivot, by the kind -------------------------------------------------------------------------------
    # BEFORE THE HULLS: a hull copies the object's transform when it is made, so a pivot moved afterwards would leave
    # every hull's origin behind.
    if pivot:
        mode = pivot if pivot != "kind" else {"floor": "bottomCorner", "wall": "bottomSide"}.get(kind, "bottom")
        args = {"object": obj.name}
        if mode == "bottom":
            # set_origin's 'bottom' reads obj.bound_box, which reads stale after the bake above; the bottom center of
            # the VERTEX bounds is placed as a point instead.
            args.update(mode="point", location=[(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, lo[2]])
        elif mode == "bottomSide":
            # the face across the wall's thickness: a wall running along X has its faces at -y and +y
            args.update(mode="bottomSide", side="-y" if size[0] >= size[1] else "-x")
        else:
            args.update(mode="bottomCorner", corner="-x-y")
        try:
            got = op_set_origin(args)
        except MifOpError as exc:
            raise _stopped("pivot", exc)
        passed["pivot"] = {"mode": mode, "side": args.get("side"), "corner": args.get("corner"),
                           "originBU": got.get("originAfterBU"), "pivotInBounds": pivot_in_bounds(obj),
                           "geometryStayedPut": got.get("geometryStayedPut")}
        _record("pivot", got.get("originBeforeBU") != got.get("originAfterBU"),
                "%s%s for a %s" % (mode, (" " + (args.get("side") or args.get("corner") or ""))
                                   if mode != "bottom" else " center", kind))

    # ---- 3. UVs -------------------------------------------------------------------------
    had_uvs = len(obj.data.uv_layers) > 0
    if do_unwrap and (force_unwrap or not had_uvs):
        unwrap_args = {"object": obj.name, "method": uv_method}
        if want_lightmap and had_uvs:
            # INTO UV0 BY NAME: the active layer can be the lightmap (a layer made active by hand, or an older pass),
            # and an unwrap into the active layer would lay the texture's UVs over it.
            unwrap_args.update(uvLayer=obj.data.uv_layers[0].name, replace=True)
        try:
            op_uv_unwrap(unwrap_args)
            _record("uvUnwrap", True,
                    "unwrapped with %s (%s)" % (uv_method,
                                                "replaced an existing layout" if had_uvs
                                                else "there was no UV layer"))
        except MifOpError as exc:
            # HALF-APPLIED, AND IT SAYS SO. The transform above is already baked into the mesh data
            # and no exception undoes it - FTransaction-style rollback does not exist here. The UE
            # recipes' contract is not "leaves nothing" but "TELLS YOU what it left", and this is
            # that contract on the Blender side.
            raise _stopped("uvUnwrap", exc)
    elif had_uvs and not force_unwrap:
        _record("uvUnwrap", False,
                "skipped - a UV layer already exists and forceUnwrap is false. Re-unwrapping a "
                "layout somebody made by hand is destructive and silent, so it is opt-in.")
        # THE LAYER IT KEEPS IS MEASURED, NOT TRUSTED. Keeping a hand-made layout is right; keeping one nobody made is
        # not, and the two look the same from here. A joined mesh's primitives each bring a layout filling the whole
        # tile, so their islands overlap - and on 2026-09-13 this recipe kept exactly that, raised no concern, and a
        # bake through it wrote a toadstool's stem over its cap. uv_info, in this module, measured 3 overlapping
        # island pairs on that layer all along; asking it is the whole fix. By bounding box, so conservative - it
        # can also count islands whose boxes touch - and the concern says so.
        me = obj.data
        kept = me.uv_layers.active.name if me.uv_layers.active else me.uv_layers[0].name
        row = (op_uv_info({"object": obj.name, "layer": kept}).get("layers") or [{}])[0]
        pairs = int(row.get("overlappingIslandPairs") or 0)
        kept_uv = {"layer": kept, "islandCount": row.get("islandCount"), "overlappingIslandPairs": pairs}
        if pairs:
            recipe_concerns.append(
                "the UV layer kept, '%s', has %d island pair(s) overlapping by bounding box - a bake through it "
                "writes one face's texels over another's. Pass forceUnwrap:true to lay it out again, or read it "
                "with uv_info first if somebody made that layout on purpose." % (kept, pairs))
    else:
        _record("uvUnwrap", False, "skipped - unwrap was false")

    # ---- 4. the lightmap layer, LAST ------------------------------------------------------------------------------
    # Unreal's lightmap coordinate index points at a channel by POSITION, and the usual layout is UV0 for the texture
    # and the last channel for the lightmap. Made after UV0 so it cannot become UV0.
    if want_lightmap:
        me = obj.data
        names = [uv.name for uv in me.uv_layers]
        lm = [i for i, n in enumerate(names) if _CONV_LIGHTMAP_NAME.search(n)]
        layer = names[lm[-1]] if lm else _RECIPE_LIGHTMAP_LAYER
        ready_before = None
        if lm:
            ready_before = (op_uv_info({"object": obj.name, "layer": layer}).get("layers") or [{}])[0]
        if ready_before and ready_before.get("lightmapReady"):
            _record("lightmap", False, "kept '%s', already the last layer and lightmap-ready" % layer)
        else:
            try:
                op_uv_unwrap({"object": obj.name, "method": "LIGHTMAP", "uvLayer": layer, "replace": bool(lm)})
            except MifOpError as exc:
                raise _stopped("lightmap", exc)
            _record("lightmap", True, ("laid out '%s' again - it was not lightmap-ready: %s"
                                       % (layer, ready_before.get("lightmapReadyReason"))) if lm
                    else "made '%s' as channel %d with uv_unwrap LIGHTMAP" % (layer, len(me.uv_layers) - 1))
        # UV0 STAYS THE ACTIVE LAYER - the one a texture, a bake and mesh_quality's UV checks read by default.
        me.uv_layers.active_index = 0
        row = (op_uv_info({"object": obj.name, "layer": layer}).get("layers") or [{}])[0]
        names = [uv.name for uv in me.uv_layers]
        passed["lightmap"] = {"layer": layer, "channel": names.index(layer), "layers": names,
                              "lightmapReady": row.get("lightmapReady"),
                              "lightmapReadyReason": row.get("lightmapReadyReason")}
        if not row.get("lightmapReady"):
            recipe_concerns.append("the lightmap layer '%s' is not lightmap-ready after the pass: %s"
                                   % (layer, row.get("lightmapReadyReason") or "see uv_info"))

    # ---- 5. the Unreal name, before the hulls that carry it ------------------------------------------------------
    if new_name is not None:
        was = obj.name
        if new_name == was:
            _record("ueName", False, "already named '%s'" % was)
        else:
            old_hulls = _hulls_named_for(was)
            try:
                op_rename_object({"object": was, "to": new_name})
                for h in old_hulls:
                    op_rename_object({"object": h, "to": new_name.join(h.rsplit(was, 1))})
            except MifOpError as exc:
                raise _stopped("ueName", exc)
            _record("ueName", True, "renamed '%s' to '%s'%s" % (was, new_name, (" with its hulls %s" % old_hulls)
                                                                 if old_hulls else ""))
            passed["renamedFrom"] = was

    # ---- 6. collision -----------------------------------------------------------------------------------------------
    if collision:
        shape = collision if collision != "auto" else ("box" if kind in ("wall", "floor") else "convex")
        existing = _hulls_named_for(obj.name)
        if existing:
            _record("collision", False, "kept the hulls already named for it: %s. Delete them to make new ones."
                    % ", ".join(existing))
            passed["collision"] = {"kept": existing, "hulls": existing, "primitiveCount": len(existing)}
        else:
            args = {"object": obj.name, "shape": shape, "perPart": per_part}
            if max_hulls is not None:
                args["maxHulls"] = max_hulls
            try:
                made = op_create_collision_hull(args)
            except MifOpError as exc:
                raise _stopped("collision", exc)
            rows = made.get("hulls") or []
            _record("collision", True, "%d %s hull(s)%s" % (len(rows), shape, " one per part" if per_part else ""))
            passed["collision"] = {"shape": shape, "perPart": per_part, "hulls": [r.get("object") for r in rows],
                                   "primitiveCount": made.get("primitiveCount"),
                                   "withinEngineLimit": made.get("withinEngineLimit"),
                                   "skippedParts": made.get("skippedParts")}

    # ---- 7. measure ---------------------------------------------------------------------
    # THE POINT OF THE RECIPE. Reporting the steps it took would be exactly the "ok:true is not
    # proof" failure in a new hat: the steps ran, and whether the result is shippable is a different
    # question that only a measurement answers.
    measure = {"object": obj.name}
    if full_pass:
        measure.update(conventions=True, kind=kind)
        if module_cm is not None:
            measure["module"] = module_cm
    quality = op_mesh_quality(measure)

    out = {
        "object": obj.name,
        "steps": steps,
        # NO leftBehind FIELD HERE, and its absence is deliberate. A failure RAISES, so this dict is
        # only ever built on the success path where nothing was left half-applied - the field could
        # never be anything but empty, and a field that is always empty is decoration. What was left
        # behind is named in the ERROR TEXT instead, which is where the UE recipes put it too and
        # where a caller who hit the failure will actually read it. audit_blender_consequence_fields
        # caught this by asking who reads the field; the answer was nobody, correctly.
        "quality": quality,
        # THE RECIPE'S OWN CONCERNS BESIDE quality's, not inside it: quality is mesh_quality's measurement, unedited,
        # and the kept-layer check is this recipe's. concernCount counts both.
        "keptUvLayer": kept_uv,
        "recipeConcerns": recipe_concerns,
        "concernCount": (quality.get("concernCount") or 0) + len(recipe_concerns),
        "recipeNote": ("the steps above say what was DONE; quality says whether the result is "
                       "shippable, and they are different questions. Read quality.notMeasured "
                       "before treating concernCount:0 as a clean bill of health."),
    }
    if full_pass:
        conv = quality.get("conventions") or {}
        # THE CONVENTIONS' FINDINGS, lifted to the top: each difference in a sentence with its number. Notes, not
        # concerns - a piece can have a reason to differ - so they do not count in concernCount.
        out["pass"] = passed
        out["conventionNotes"] = conv.get("notes") or []
        out["conventionsFollowed"] = conv.get("followCount")
        out["conventionsDiffer"] = conv.get("differCount")
    return out

# ---------------------------------------------------------------------------
# Garment surface repair (a short film's raw-Python steps, 2026-10-04)
# ---------------------------------------------------------------------------

def _boundary_loops(bm):
    """The boundary edges (one face each) grouped into connected loops: [(edges, verts)], most edges first.

    BY CONNECTED COMPONENT, NOT BY WALKING ONE EDGE TO THE NEXT. A walker with no visited check loops forever at a
    bow-tie vertex (four boundary edges meeting): the police uniform's loop walker did, and its Blender was killed at
    12.57 GB (PROGRESS_garment, 2026-10-04). A component cannot loop."""
    edges = [e for e in bm.edges if len(e.link_faces) == 1]
    parent = {}

    def find(v):
        root = v
        while parent.get(root, root) is not root:
            root = parent[root]
        while parent.get(v, v) is not v:
            parent[v], v = root, parent[v]
        return root

    for e in edges:
        a, b = find(e.verts[0]), find(e.verts[1])
        if a is not b:
            parent[a] = b
    groups = {}
    for e in edges:
        groups.setdefault(find(e.verts[0]), []).append(e)
    loops = []
    for es in groups.values():
        verts = list(dict.fromkeys(v for e in es for v in e.verts))
        loops.append((es, verts))
    loops.sort(key=lambda t: (-len(t[0]), min(v.index for v in t[1])))
    return loops


def _hole_sector(v, face):
    """The faces round vertex v reachable from `face` across edges at v that join two faces - one fan sector."""
    seen, stack = {face}, [face]
    while stack:
        f = stack.pop()
        for e in f.edges:
            if v in e.verts and len(e.link_faces) == 2:
                for g in e.link_faces:
                    if g not in seen:
                        seen.add(g)
                        stack.append(g)
    return seen


def _hole_cycles(edges):
    """Split one boundary component into its holes: ([[verts in hole order], ...], None) or (None, reason).

    A HOLE RUNS AGAINST ITS FACES. Each boundary edge has one face, and the hole goes along it the other way, so at a
    vertex where two holes touch (four boundary edges, two fan sectors of faces) the hole that came in through one
    sector's edge leaves through the OTHER sector's edge - in a lone hole both edges bound the one sector. Measured on
    a sheet with two quads gone that share a corner: two 4-vertex holes, where the component read as one 7-vertex loop
    that was not closed. Three sectors at a vertex would need the angles to pair them, and are named instead."""
    out_of = {}
    into = {}
    for e in edges:
        lp = e.link_loops[0]
        a, b = lp.vert, lp.link_loop_next.vert          # the face runs a -> b, the hole b -> a
        out_of.setdefault(b, []).append((e, a))
        into[a] = into.get(a, 0) + 1
    for v, outs in out_of.items():
        if into.get(v, 0) != len(outs):
            return None, ("the faces round vertex %d run opposite ways (flipped normals), so the hole has no one "
                          "direction - recalcNormals first" % v.index)
    if any(v not in out_of for v in into):
        return None, "the faces round it run opposite ways (flipped normals) - recalcNormals first"
    for v, outs in out_of.items():
        if len(outs) > 2:
            return None, "three or more holes meet at vertex %d" % v.index
    used, cycles = set(), []
    for start_v in list(out_of):
        for start in out_of[start_v]:
            if start[0] in used:
                continue
            cyc, v, nxt = [], start_v, start
            while nxt[0] not in used:
                e, w = nxt
                used.add(e)
                cyc.append(v)
                v = w
                outs = out_of[v]
                if len(outs) == 2:
                    sector = _hole_sector(v, e.link_faces[0])
                    pick = [c for c in outs if c[0].link_faces[0] not in sector]
                    if len(pick) != 1:
                        return None, "the faces round vertex %d do not separate the holes that meet there" % v.index
                    nxt = pick[0]
                else:
                    nxt = outs[0]
            cycles.append(cyc)
    return cycles, None


def _fill_holes(bm, upto):
    """clean_mesh fillHolesUpTo: every hole of at most `upto` vertices filled ON ITS OWN, or named with the reason.

    MEASURED 2026-10-05 on 5.2.2: bmesh.ops.holes_fill adds no face for a hole whose vertex also holds a wire edge
    across it, and none at all when the edges given include a loop longer than `sides` - and this step used to hand it
    every hole in one call, so a garment answered holes 5, facesAdded 0. So each hole gets its own holes_fill, then -
    when that adds nothing - a face through its vertices in hole order (the winding its neighbours imply), each corner
    copied from a neighbouring face's corner at that vertex so UVs and colours follow. A component where holes touch
    at a vertex is split first (_hole_cycles)."""
    bm.verts.index_update()
    filled, skipped, counts = [], [], []
    f0 = len(bm.faces)
    for edges, verts in _boundary_loops(bm):
        if len(verts) > upto and all(sum(1 for e in v.link_edges if e in edges) == 2 for v in verts):
            skipped.append({"vertexCount": len(verts), "vertices": sorted(v.index for v in verts)[:20],
                            "reason": "larger than fillHolesUpTo %d" % upto, "fillable": False})
            continue
        cycles, why = _hole_cycles(edges)
        if why:
            skipped.append({"vertexCount": len(verts), "vertices": sorted(v.index for v in verts)[:20],
                            "reason": why, "fillable": True})
            continue
        for cyc in cycles:
            if len(cyc) > upto:
                skipped.append({"vertexCount": len(cyc), "vertices": sorted(v.index for v in cyc)[:20],
                                "reason": "larger than fillHolesUpTo %d" % upto, "fillable": False})
                continue
            counts.append(len(cyc))
            cyc_edges = [next(e for e in a.link_edges if e.other_vert(a) is b and len(e.link_faces) == 1)
                         for a, b in zip(cyc, cyc[1:] + cyc[:1])]
            n0 = len(bm.faces)
            if len(cycles) == 1:
                bmesh.ops.holes_fill(bm, edges=cyc_edges, sides=upto)
            if len(bm.faces) > n0:
                filled.append({"vertexCount": len(cyc), "by": "holes_fill"})
                continue
            try:
                face = bm.faces.new(cyc)
            except ValueError as exc:
                skipped.append({"vertexCount": len(cyc), "vertices": sorted(v.index for v in cyc)[:20],
                                "reason": ("a face over these vertices already exists (a fold where the surface "
                                           "doubles back)" if "exists" in str(exc) else str(exc)), "fillable": True})
                counts.pop()
                continue
            border = [f for e in cyc_edges for f in e.link_faces if f is not face]
            face.material_index = border[0].material_index
            face.smooth = border[0].smooth
            for lp in face.loops:
                src = sorted((x for x in lp.vert.link_loops if x.face is not face),
                             key=lambda x: 0 if x.face in border else 1)
                if src:
                    lp.copy_from(src[0])
            filled.append({"vertexCount": len(cyc), "by": "face"})
    bm.normal_update()
    left = [h["vertexCount"] for h in skipped]
    return {"holes": len(counts), "holeVertexCounts": sorted(counts), "facesAdded": len(bm.faces) - f0, "upTo": upto,
            "filled": filled[:50], "skipped": skipped[:50], "holesLeft": sorted(left, reverse=True)[:50]}


def _loop_nearest(loops, point, mw):
    """The loop holding the boundary vertex nearest a WORLD point, and that distance."""
    best, best_d = None, None
    for loop in loops:
        for v in loop[1]:
            d = ((mw @ v.co) - point).length
            if best_d is None or d < best_d:
                best, best_d = loop, d
    return best, best_d


def _world_matrix_or_refuse(obj, op_name):
    """obj.matrix_world, refused when it cannot be inverted or mirrors: these ops work in world space and write back
    through the inverse, and a mirrored transform turns every face's winding - and so its normal - inside out."""
    mw = obj.matrix_world
    det = mw.to_3x3().determinant()
    if not math.isfinite(det) or abs(det) < 1e-12:
        raise MifOpError("'%s' has a transform that cannot be inverted (scale 0 on an axis), so %s cannot write its "
                         "result back. NOTHING was changed." % (obj.name, op_name))
    if det < 0:
        raise MifOpError("'%s' is mirrored (a negative scale), which turns its normals inside out in world space - "
                         "%s would push the wrong way. Apply the transform first (bl_apply_transform). NOTHING was "
                         "changed." % (obj.name, op_name))
    return mw


def _no_shape_keys(obj, op_name):
    if getattr(obj.data, "shape_keys", None) is not None:
        raise MifOpError("'%s' has shape keys, and %s writes the base mesh only - the keys would pull the moved "
                         "vertices back. Apply or remove them first. NOTHING was changed." % (obj.name, op_name))


def _shifted(mw, shift):
    """mw with `shift` taken off its translation: world space moved so the edited object's origin is at 0.

    FLOAT32 FAR FROM THE ORIGIN. Mesh coordinates are 32-bit floats, so carrying a mesh 240 m out into world space and
    back moved its vertices by 2 micrometres (measured 2026-10-04, 5.2.1: a fold's ring at -0.489395 for
    -0.489393) - on a 5 mm fold. Rotation and scale are applied; the big translation never touches a coordinate."""
    m = mw.copy()
    m.translation = mw.translation - shift
    return m


def _world_bmesh(obj, evaluated, shift):
    """A bmesh of obj in world space moved by -shift (see _shifted) - evaluated: its modifiers and pose applied - with
    normals updated. Caller frees."""
    bm = bmesh.new()
    if evaluated:
        ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = ev.to_mesh()
        try:
            bm.from_mesh(me)
        finally:
            ev.to_mesh_clear()
    else:
        bm.from_mesh(obj.data)
    bm.transform(_shifted(obj.matrix_world, shift))
    bm.normal_update()
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    return bm


def _box_param(params, key):
    """A world box [[minX,minY,minZ],[maxX,maxY,maxZ]] or None, corners in either order."""
    raw = params.get(key)
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2 or any(
            not isinstance(c, (list, tuple)) or len(c) != 3 for c in raw):
        raise MifOpError("%s must be [[minX,minY,minZ],[maxX,maxY,maxZ]] in world units, got %r. NOTHING was changed."
                         % (key, raw))
    a, b = finite_floats(list(raw[0]), key), finite_floats(list(raw[1]), key)
    return [min(a[i], b[i]) for i in range(3)], [max(a[i], b[i]) for i in range(3)]


def _in_box(p, box):
    return box is None or all(box[0][i] <= p[i] <= box[1][i] for i in range(3))


def _point_param(params, key):
    raw = params.get(key)
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise MifOpError("%s must be a world point [x,y,z], got %r. NOTHING was changed." % (key, raw))
    return Vector(finite_floats(list(raw), key))


_PUSH_KEYS = ("object", "collider", "offset", "side", "reach", "iterations", "smooth", "coverDistance",
              "skipFacingCollider", "skipSlots", "insideBox", "dryRun", "previewId", "previewTtlSeconds")


def _push_measure(bm, col_tree, col_pts, offset, reach, skip_slots, skip_facing, box):
    """Both sides of the gap, measured. GARMENT SIDE: each garment vertex (in the box, not on a skipped slot) against
    the nearest collider point within reach - s is its height above the collider along the collider's normal.
    COLLIDER SIDE: each covered collider point against the nearest garment face within reach - depth is how far the
    point stands OUTSIDE that face along the face's normal. A body bump between garment vertices shows only on the
    collider side (the acromion poked up between vertices 1.25 cm apart and a vertex test never saw it)."""
    from mathutils.bvhtree import BVHTree
    g_tested = g_inside = g_short = 0
    g_max_pen = g_max_short = 0.0
    for v in bm.verts:
        if box is not None and not _in_box(v.co, box):
            continue
        if skip_slots and v.link_faces and all(f.material_index in skip_slots for f in v.link_faces):
            continue
        loc, n, _i, _d = col_tree.find_nearest(v.co, reach)
        if loc is None:
            continue
        g_tested += 1
        s = (v.co - loc).dot(n)
        if s < 0:
            g_inside += 1
            g_max_pen = max(g_max_pen, -s)
        if s < offset - 1e-9:
            g_short += 1
            g_max_short = max(g_max_short, offset - s)
    tree = BVHTree.FromBMesh(bm)
    c_tested = c_through = c_short = facing = 0
    c_max_pen = c_max_short = 0.0
    for p, pn in col_pts:
        loc, _n, fi, _d = tree.find_nearest(p, reach)
        if loc is None:
            continue
        f = bm.faces[fi]
        if f.material_index in skip_slots:
            continue
        if skip_facing and f.normal.dot(pn) < 0.2:
            facing += 1
            continue
        c_tested += 1
        depth = (p - loc).dot(f.normal)
        if depth > 1e-9:
            c_through += 1
            c_max_pen = max(c_max_pen, depth)
        if depth + offset > 1e-9:
            c_short += 1
            c_max_short = max(c_max_short, depth + offset)
    return {
        "garmentSide": {"verticesTested": g_tested, "verticesInside": g_inside, "maxPenetration": round(g_max_pen, 6),
                        "verticesUnderOffset": g_short, "maxShortfall": round(g_max_short, 6)},
        "colliderSide": {"pointsTested": c_tested, "pointsThrough": c_through, "maxPenetration": round(c_max_pen, 6),
                         "pointsUnderOffset": c_short, "maxShortfall": round(c_max_short, 6),
                         "skippedFacingCollider": facing},
    }


def op_push_out_of(params):
    """Push one mesh out of another to a clearance: a garment off the body, or off the layer under it.

    THE GAP (row 9 of the spec's Why-Python table): after a drape or a fit the garment work pushed one mesh out of
    another in raw Python six times (pushout.py), then again from the body's side (pushout_body.py), because a cloth
    collider keeps VERTICES out and a body bump between them still pokes through.

    TWO SIDES (`side`), the measurement the same for both:
      garment   (default) each vertex of `object` within `reach` of `collider` whose height above it, along the
                collider's normal, is under `offset` moves out along that normal to `offset`; the moves are then spread
                to neighbours `smooth` times so no dent shows (a neighbour takes the average when it is bigger than its
                own).
      collider  each COVERED point of the collider (a ray from 3 cm inside it, along its normal, meets the garment
                within coverDistance - so a hand past a cuff or a face in a hood opening is left alone: the first run
                chased the hands 174 mm) against the nearest garment face within reach: when the point stands less than
                `offset` under that face, the face's vertices move out along their normals by the shortfall. Repeated
                `iterations` times, then spread `smooth` rings. skipFacingCollider (default true) leaves a garment face
                that faces the body alone - a lining, a folded-under lip - as the recorded run had to.

    BOTH SIDES ARE MEASURED BEFORE AND AFTER (before/after): garmentSide (vertices inside the collider, the worst depth)
    and colliderSide (collider points through the garment - the bump between vertices - and the worst depth).
    remainingPenetration is after.colliderSide.maxPenetration: what is still through after the push.

    WORLD UNITS, the collider as it is now (modifiers and pose applied), the object's base mesh written. A garment
    deformed by its own modifiers at this frame is computed on its base mesh against the posed collider - put the rig
    at rest first; deformedNote says when the two differ. A mirrored or uninvertible transform and shape keys are
    refused. dryRun measures and plans, and writes nothing. A call that tests nothing (no garment vertex within reach,
    no covered collider point) is refused: it would answer clean having looked at nothing.
    """
    from mathutils.bvhtree import BVHTree
    reject_unknown(params, _PUSH_KEYS, "push_out_of")
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    col = get_object(take(params, "collider", required=True, kind=str), want_mesh=True)
    if obj is col:
        raise MifOpError("object and collider are the same mesh. NOTHING was changed.")
    require_editable(obj, "push")
    side = str(take(params, "side", default="garment", kind=str)).lower()
    if side not in ("garment", "collider"):
        raise MifOpError("side must be 'garment' (each garment vertex against the collider) or 'collider' (each "
                         "covered collider point against the garment), got %r. NOTHING was changed." % side)
    offset = take_float(params, "offset", default=0.005)
    reach = take_float(params, "reach", default=0.03 if side == "garment" else 0.04)
    cover = take_float(params, "coverDistance", default=0.08)
    iters = take_int(params, "iterations", default=3)
    smooth = take_int(params, "smooth", default=3)
    skip_facing = take_bool(params, "skipFacingCollider", default=True)
    dry = take_bool(params, "dryRun", default=False)
    box = _box_param(params, "insideBox")
    raw_slots = params.get("skipSlots")
    if raw_slots is not None and (not isinstance(raw_slots, (list, tuple)) or any(
            isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in raw_slots)):
        raise MifOpError("skipSlots must be a list of material slot indices, got %r. NOTHING was changed." % (raw_slots,))
    skip_slots = set(raw_slots or ())
    for key, val, lo in (("offset", offset, 0.0), ("reach", reach, 1e-9), ("coverDistance", cover, 1e-9)):
        if not (math.isfinite(val) and val >= lo):
            raise MifOpError("%s must be a finite number %s, got %r. NOTHING was changed."
                             % (key, "of 0 or more" if lo == 0.0 else "above 0", val))
    if not 1 <= iters <= 50 or not 0 <= smooth <= 50:
        raise MifOpError("iterations must be 1-50 and smooth 0-50, got %d and %d. NOTHING was changed." % (iters, smooth))
    if side == "garment" and params.get("iterations") is not None:
        raise MifOpError("iterations is for side 'collider', which pushes faces and measures again; side 'garment' "
                         "moves each vertex once to the offset. NOTHING was changed.")
    _no_shape_keys(obj, "push_out_of")
    mw = _world_matrix_or_refuse(obj, "push_out_of")
    _world_matrix_or_refuse(col, "push_out_of")
    if not len(obj.data.polygons) or not len(col.data.polygons):
        raise MifOpError("'%s' has no faces. NOTHING was changed." % (obj.name if not len(obj.data.polygons) else col.name))

    shift = mw.translation.copy()
    if box is not None:
        box = ([box[0][i] - shift[i] for i in range(3)], [box[1][i] - shift[i] for i in range(3)])
    cbm = _world_bmesh(col, True, shift)
    bm = _world_bmesh(obj, False, shift)
    try:
        col_tree = BVHTree.FromBMesh(cbm)
        g_tree = BVHTree.FromBMesh(bm)
        raw_pts = [(v.co.copy(), v.normal.copy()) for v in cbm.verts if _in_box(v.co, box)]
        col_pts = [(p, n) for p, n in raw_pts if g_tree.ray_cast(p - n * 0.03, n, 0.03 + cover)[0] is not None]
        before = _push_measure(bm, col_tree, col_pts, offset, reach, skip_slots, skip_facing, box)
        tested = before["garmentSide" if side == "garment" else "colliderSide"]
        if not (tested.get("verticesTested") or tested.get("pointsTested")):
            raise MifOpError(
                "nothing was tested: %s. A push that looked at nothing would answer clean. Raise reach%s, or check the "
                "two objects are where you think. NOTHING was changed."
                % ("no vertex of '%s' is within reach (%g) of '%s'" % (obj.name, reach, col.name) if side == "garment"
                   else "%d of %d collider points are covered by '%s' (a ray along the collider's normal meets it within "
                        "coverDistance %g), and none of them is within reach (%g) of a garment face it may push%s"
                        % (len(col_pts), len(raw_pts), obj.name, cover, reach,
                           " - %d met only faces that face the collider" % tested.get("skippedFacingCollider")
                           if tested.get("skippedFacingCollider") else ""),
                   "" if side == "garment" else " or coverDistance"))
        total = {}
        per_iter = []
        if side == "garment":
            disp = {}
            for v in bm.verts:
                if box is not None and not _in_box(v.co, box):
                    continue
                if skip_slots and v.link_faces and all(f.material_index in skip_slots for f in v.link_faces):
                    continue
                loc, n, _i, _d = col_tree.find_nearest(v.co, reach)
                if loc is None:
                    continue
                s = (v.co - loc).dot(n)
                if s < offset - 1e-9:
                    disp[v.index] = n * (offset - s)
            field = dict(disp)
            for _ in range(smooth):
                new = {}
                for v in bm.verts:
                    vals = [field.get(e.other_vert(v).index, Vector()) for e in v.link_edges]
                    vals.append(field.get(v.index, Vector()))
                    avg = sum(vals, Vector()) / len(vals)
                    cur = field.get(v.index, Vector())
                    new[v.index] = cur if cur.length >= avg.length else avg
                field = {i: d for i, d in new.items() if d.length > 1e-9}
            pushed, moves = len(disp), field
            max_push = max((d.length for d in disp.values()), default=0.0)
            if not dry:
                for i, d in moves.items():
                    bm.verts[i].co += d
            spread = len(moves) - len(disp)
        else:
            for _ in range(iters):
                bm.normal_update()
                tree = BVHTree.FromBMesh(bm)
                push = {}
                for p, pn in col_pts:
                    loc, _n, fi, _d = tree.find_nearest(p, reach)
                    if loc is None:
                        continue
                    f = bm.faces[fi]
                    if f.material_index in skip_slots or (skip_facing and f.normal.dot(pn) < 0.2):
                        continue
                    need = (p - loc).dot(f.normal) + offset
                    if need > 1e-9:
                        for v in f.verts:
                            if need > push.get(v.index, 0.0):
                                push[v.index] = need
                per_iter.append(len(push))
                if not push:
                    break
                for i, amt in push.items():
                    v = bm.verts[i]
                    v.co += v.normal * amt
                    total[i] = total.get(i, 0.0) + amt
                if dry:
                    break
            field = dict(total)
            for _ in range(smooth if not dry else 0):
                new = {}
                for v in bm.verts:
                    if v.index in field:
                        continue
                    nb = [field[e.other_vert(v).index] for e in v.link_edges if e.other_vert(v).index in field]
                    if nb:
                        new[v.index] = 0.5 * sum(nb) / len(v.link_edges)
                for i, amt in new.items():
                    bm.verts[i].co += bm.verts[i].normal * amt
                field.update(new)
            pushed, max_push, spread = len(total), max(total.values(), default=0.0), len(field) - len(total)
        bm.normal_update()
        after = None if dry else _push_measure(bm, col_tree, col_pts, offset, reach, skip_slots, skip_facing, box)
        if not dry and (pushed or spread):
            bm.transform(_shifted(mw, shift).inverted())
            bm.to_mesh(obj.data)
            obj.data.update()
            bpy.context.view_layer.update()
    finally:
        bm.free()
        cbm.free()
    out = {"object": obj.name, "collider": col.name, "side": side, "offset": offset, "reach": reach,
           "verticesPushed": pushed, "verticesSpread": spread, "maxPush": round(max_push, 6), "dryRun": dry,
           "before": before, "after": after,
           "remainingPenetration": (after or {}).get("colliderSide", {}).get("maxPenetration"),
           "space": "WORLD units; the collider as evaluated now, the object's base mesh"}
    if side == "collider":
        out.update({"iterations": per_iter, "colliderPoints": len(raw_pts), "colliderPointsCovered": len(col_pts),
                    "coverDistance": cover})
    if dry:
        out["note"] = ("dryRun: the first pass was planned (verticesPushed, maxPush) and nothing was written; the "
                       "spread and any further passes are not run.")
    elif not pushed:
        out["note"] = ("every tested point already clears the offset - nothing needed pushing. before says what was "
                       "tested.")
    out.update(shared_data_note(obj))
    return out


_FOLD_KEYS = ("object", "thickness", "depth", "loopNear", "straighten", "group")


def op_fold_boundary_under(params):
    """Fold a single-sheet mesh's open rim under itself: edge-only thickness for a hem, a cuff or a face opening.

    THE GAP (row 9): a draped garment is one sheet, so its hem, cuffs and openings read as paper-thin in a render.
    Solidify thickens the whole sheet and doubles the triangles; the garment work folded the rim under in raw Python
    ten times (rim_lip.py). Two rings are added per boundary loop: ring 1 is the boundary pushed in along minus the
    surface normal by `thickness` (the visible edge wall), ring 2 is ring 1 carried back under the fabric by `depth`
    along the surface, toward the faces the rim belongs to. WORLD units.

    loopNear [x,y,z] folds only the loop with the boundary vertex nearest that world point (a cuff, not the hem);
    without it every loop is folded. straighten N smooths the rim along itself N times first (each vertex halfway to
    its two boundary neighbours' midpoint) - the drape left the rims ragged. A BOW-TIE VERTEX (more than two boundary
    edges) is left out with its edges and counted (skippedBowtieEdges): extruding there made non-manifold edges.
    group names a vertex group that receives the old boundary - the folded edge - at weight 1.0, for stitching or a pin.
    Mirrored or uninvertible transforms and shape keys are refused, as is a mesh with no open boundary.
    """
    reject_unknown(params, _FOLD_KEYS, "fold_boundary_under")
    obj = get_object(take(params, "object", required=True, kind=str), want_mesh=True)
    require_editable(obj, "fold the rim of")
    thick = take_float(params, "thickness", default=0.005)
    depth = take_float(params, "depth", default=0.015)
    straighten = take_int(params, "straighten", default=0)
    near = _point_param(params, "loopNear")
    group = take(params, "group", default=None, kind=str)
    for key, val in (("thickness", thick), ("depth", depth)):
        if not (math.isfinite(val) and val > 0):
            raise MifOpError("%s must be a finite number above 0 (world units), got %r. NOTHING was changed."
                             % (key, val))
    if not 0 <= straighten <= 100:
        raise MifOpError("straighten must be 0-100, got %d. NOTHING was changed." % straighten)
    if group is not None and not group.strip():
        raise MifOpError("group must be a non-empty name. NOTHING was changed.")
    _no_shape_keys(obj, "fold_boundary_under")
    mw = _world_matrix_or_refuse(obj, "fold_boundary_under")
    me = obj.data
    before = mesh_counts(obj)
    shift = mw.translation.copy()
    work = _shifted(mw, shift)
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bm.transform(work)
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        loops = _boundary_loops(bm)
        if not loops:
            raise MifOpError("'%s' has no open boundary - every edge has two faces, so there is no rim to fold. "
                             "NOTHING was changed." % obj.name)
        nearest_d = None
        if near is not None:
            loop, nearest_d = _loop_nearest(loops, near - shift, Matrix.Identity(4))
            chosen = [loop]
        else:
            chosen = loops
        edges = [e for loop in chosen for e in loop[0]]
        deg = {}
        for e in edges:
            for v in e.verts:
                deg[v] = deg.get(v, 0) + 1
        fold = [e for e in edges if deg[e.verts[0]] == 2 and deg[e.verts[1]] == 2]
        if not fold:
            raise MifOpError("every boundary edge chosen meets a bow-tie vertex (more than two boundary edges), so "
                             "nothing can be folded cleanly. NOTHING was changed.")
        bv = list(dict.fromkeys(v for e in fold for v in e.verts))
        for _ in range(straighten):
            new = {}
            for v in bv:
                nb = [e.other_vert(v) for e in v.link_edges if len(e.link_faces) == 1]
                if len(nb) == 2:
                    new[v] = v.co * 0.5 + (nb[0].co + nb[1].co) * 0.25
            for v, p in new.items():
                v.co = p
        bm.normal_update()
        nrm = {v: v.normal.copy() for v in bv}
        inward = {}
        for v in bv:
            t = sum(((f.calc_center_median() - v.co) for f in v.link_faces), Vector())
            t = t - nrm[v] * t.dot(nrm[v])
            inward[v] = t.normalized() if t.length > 1e-12 else Vector()
        fold_ix = sorted(v.index for v in bv)
        r1 = bmesh.ops.extrude_edge_only(bm, edges=fold)
        v1 = [g for g in r1["geom"] if isinstance(g, bmesh.types.BMVert)]
        v1_set = set(v1)
        bv_set = set(bv)
        src = {}
        for v in v1:
            for e in v.link_edges:
                o = e.other_vert(v)
                if o in bv_set:
                    src[v] = o
                    break
        for v, o in src.items():
            v.co = o.co - nrm[o] * thick
        e1 = [g for g in r1["geom"] if isinstance(g, bmesh.types.BMEdge)
              and g.verts[0] in v1_set and g.verts[1] in v1_set]
        r2 = bmesh.ops.extrude_edge_only(bm, edges=e1)
        for v in (g for g in r2["geom"] if isinstance(g, bmesh.types.BMVert)):
            for e in v.link_edges:
                o = e.other_vert(v)
                if o in src:
                    v.co = o.co + inward[src[o]] * depth
                    break
        no_inward = sum(1 for v in bv if inward[v].length == 0.0)
        bm.transform(work.inverted())
        bm.normal_update()
        boundary_after = sum(1 for e in bm.edges if len(e.link_faces) == 1)
        bm.to_mesh(me)
    finally:
        bm.free()
    me.update()
    group_count = None
    if group is not None:
        vg = obj.vertex_groups.get(group) or obj.vertex_groups.new(name=group)
        vg.add(fold_ix, 1.0, "REPLACE")
        group_count = len(fold_ix)
    bpy.context.view_layer.update()
    after = mesh_counts(obj)
    out = {"object": obj.name, "loops": len(loops), "loopsFolded": len(chosen), "boundaryEdgesFolded": len(fold),
           "skippedBowtieEdges": len(edges) - len(fold), "thickness": thick, "depth": depth,
           "straighten": straighten, "before": before, "after": after, "boundaryEdgesAfter": boundary_after,
           "group": group, "groupVertices": group_count, "space": "WORLD units"}
    if nearest_d is not None:
        out["loopNearDistance"] = round(nearest_d, 6)
    if no_inward:
        out["noInwardNote"] = ("%d rim vertices had no surface direction to fold toward (their faces' centres lie "
                               "along the normal), so ring 2 sits on ring 1 there." % no_inward)
    out.update(shared_data_note(obj))
    return out


_UV_PACK_KEYS = ("object", "name", "uvLayer", "margin", "rotate", "equalizeByArea", "textureSize", "mergeOverlap",
                 "targetTexelDensity", "unit", "region")


def _run_pack_islands(obj, name, margin, pinned):
    """Blender's pack_islands on one object's layer `name`, every island selected, with `pinned` (see _PACK_PINNED) cut
    to what this build takes. The active UV layer, the mode and the selection are put back; the UVs are not - the
    caller keeps a copy when a failure must undo the pack. uv_pack and uv_set_texel_density both pack through here."""
    me = obj.data
    prev_uv_active = me.uv_layers.active.name if me.uv_layers.active else None
    sel = selection_snapshot()
    try:
        me.uv_layers.active = me.uv_layers[name]
        if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        select_only([obj])
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.select_all(action="SELECT")
        bpy.ops.uv.pack_islands(margin=margin, **_pinned_op_kwargs(bpy.ops.uv.pack_islands, pinned))
        bpy.ops.object.mode_set(mode="OBJECT")
    finally:
        try:
            if obj.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            selection_restore(sel)
            if prev_uv_active is not None and prev_uv_active in me.uv_layers:
                me.uv_layers.active = me.uv_layers[prev_uv_active]
        except Exception:  # noqa: BLE001
            pass


def _scale_layout_islands(me, layer_name, k):
    """Every island of `layer_name` scaled by k about its own bounding-box centre - densities move together."""
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        uv = bm.loops.layers.uv[layer_name]
        for faces in _uv_islands(bm, uv)[0]:
            loops = [lp for f in faces for lp in f.loops]
            us = [lp[uv].uv.x for lp in loops]
            vs = [lp[uv].uv.y for lp in loops]
            cx, cy = 0.5 * (min(us) + max(us)), 0.5 * (min(vs) + max(vs))
            for lp in loops:
                lp[uv].uv = (cx + (lp[uv].uv.x - cx) * k, cy + (lp[uv].uv.y - cy) * k)
        bm.to_mesh(me)
    finally:
        bm.free()
    me.update()


def _measure_islands(me, layer_name, m3):
    """_island_rows for one UV layer of a mesh, off a scratch bmesh that is freed."""
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        uv = bm.loops.layers.uv[layer_name]
        rows = _island_rows(_uv_islands(bm, uv)[0], uv, m3)
    finally:
        bm.free()
    return rows


def op_uv_pack(params):
    """Pack a mesh's existing UV islands into the 0-1 tile WITHOUT re-unwrapping - optionally equalising them by area.

    THE GAP (row 7 of the spec's Why-Python table): uv_unwrap always unwraps before it packs, so a layout made by hand,
    or one an ANGLE unwrap made from careful seams, could not be repacked without being replaced. The garment work
    packed in raw Python (sn3_pack.py) and equalised island density in raw Python (uv_equalize.py).

    equalizeByArea scales every island about its own UV centre so that its UV area per square world unit is the median
    island's - one texel density across the garment - BEFORE the pack. Areas are UNSIGNED per face: Blender's
    average_islands_scale uses signed areas, which a folded face reduces, and it left a sleeve at 3x the density of the
    rest. An island with no UV or no world area is left at its size and counted. The pack is Blender's pack_islands
    with every option pinned (uv_unwrap's), scaling all islands together, so the equal densities stay equal.

    The answer measures the islands before and after (ratioSpreadBefore/After: largest UV-per-world-area over the
    smallest, 1.0 when every island has one density), the layout's UV area and its bounds. If the pack throws, the
    layer is put back as it was - equalised islands included - and the call fails saying so. The active UV layer and
    the selection are restored.

    textureSize (n | [w, h]) packs for that texture: on a non-square one V is stretched by w / h first so texels stay
    square, and rotation is refused - pack_islands in a session without a UV editor packs as if the tile were square,
    and a rotating pack of a V-stretched island measured texels 2.65 times longer one way than the other (3.6.23 and
    5.2.1, 2026-10-04). The reply's texelShape is the longer texel side over the shorter, per triangle (median, max): 1
    is square. targetTexelDensity (+ unit PX_PER_CM default | PX_PER_M, needs textureSize) scales the layout to that
    density instead of the largest that fits, and packs with scale off: islandsOutside01 counts what no longer fits.
    mergeOverlap keeps islands that already overlap stacked (repeated details on one tile). region [uMin, vMin, uMax,
    vMax] packs into that part of the tile instead (proportions kept; insideRegion says every UV landed in it).
    """
    reject_unknown(params, _UV_PACK_KEYS, "uv_pack")
    from .ops_uv import _target_px_per_m, _texture_size, density_stats, texel_shape   # noqa: PLC0415
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    require_editable(obj, "pack the UVs of")
    layer_name = take(params, "uvLayer", default=None, kind=str)
    margin = take_float(params, "margin", default=0.02)
    equalize = take_bool(params, "equalizeByArea", default=False)
    if not 0.0 <= margin < 1.0:
        raise MifOpError("margin must be in [0, 1), got %r. NOTHING was changed." % margin)
    tex = _texture_size(params["textureSize"]) if params.get("textureSize") is not None else None
    # A REGION OF THE TILE (2026-10-06, 5.0's custom region, packed here on every build): region [uMin, vMin, uMax,
    # vMax]. A non-square region is packed the way a non-square texture is - V stretched by its width over its height,
    # packed, then placed - so the islands keep their proportions; rotation is refused there for the same reason.
    region = None
    if params.get("region") is not None:
        raw = params["region"]
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise MifOpError("region is [uMin, vMin, uMax, vMax]. NOTHING was changed.")
        region = finite_floats(list(raw), "region")
        if not (region[2] > region[0] and region[3] > region[1]):
            raise MifOpError("region %s: uMax and vMax must be above uMin and vMin. NOTHING was changed." % region)
        if tex is not None or params.get("targetTexelDensity") is not None:
            raise MifOpError("region packs into part of the tile; textureSize and targetTexelDensity size the whole "
                             "layout - pass one or the other. NOTHING was changed.")
        rw, rh = region[2] - region[0], region[3] - region[1]
        if abs(rw - rh) > 1e-9:
            tex = (rw, rh)
    square = tex is None or tex[0] == tex[1]
    rotate_asked = take_bool(params, "rotate") if params.get("rotate") is not None else None
    if rotate_asked and not square:
        raise MifOpError("rotate is refused on a non-square textureSize %s: the pack treats the tile as square, so a "
                         "rotated island loses its square texels (measured 2.65 to 1). Drop rotate, or pack for a "
                         "square size. NOTHING was changed." % [int(tex[0]), int(tex[1])])
    rotate = square if rotate_asked is None else rotate_asked
    merge = take_bool(params, "mergeOverlap") if params.get("mergeOverlap") is not None else None
    if merge is not None and "merge_overlap" not in bpy.ops.uv.pack_islands.get_rna_type().properties.keys():
        raise MifOpError("mergeOverlap: this Blender's pack_islands has no merge_overlap. NOTHING was changed.")
    target = None
    if params.get("targetTexelDensity") is not None:
        if tex is None:
            raise MifOpError("targetTexelDensity needs textureSize - a density is pixels of a texture per metre. "
                             "NOTHING was changed.")
        target = _target_px_per_m({"target": params["targetTexelDensity"], "unit": params.get("unit")})
    elif params.get("unit") is not None:
        raise MifOpError("unit is targetTexelDensity's; pass that too, or drop unit. NOTHING was changed.")
    me = obj.data
    if not me.uv_layers:
        raise MifOpError("'%s' has no UV layer to pack - unwrap it first (uv_unwrap). NOTHING was changed." % obj.name)
    names = [uv.name for uv in me.uv_layers]
    if layer_name is not None and layer_name not in names:
        raise MifOpError("'%s' has no UV layer named %r (it has %s). NOTHING was changed."
                         % (obj.name, layer_name, ", ".join(names)))
    if not len(me.polygons):
        raise MifOpError("'%s' has no faces. NOTHING was changed." % obj.name)
    name = layer_name or (me.uv_layers.active or me.uv_layers[0]).name
    m3 = obj.matrix_world.to_3x3()
    rows_before = _measure_islands(me, name, m3)
    med, spread_before = _ratio_stats(rows_before)
    layer = me.uv_layers[name]
    saved = array.array("f", [0.0]) * (2 * len(layer.data))
    layer.data.foreach_get("uv", saved)
    scaled = skipped = 0
    scale_to = None
    try:
        if equalize and med is not None:
            bm = bmesh.new()
            try:
                bm.from_mesh(me)
                uv = bm.loops.layers.uv[name]
                for faces in _uv_islands(bm, uv)[0]:
                    uva = sum(_face_uv_area(f, uv) for f in faces)
                    a3 = sum(_face_area_through(f, m3) for f in faces)
                    if a3 <= 1e-12 or uva <= 1e-12:
                        skipped += 1
                        continue
                    k = (med / (uva / a3)) ** 0.5
                    loops = [lp for f in faces for lp in f.loops]
                    cx = sum(lp[uv].uv.x for lp in loops) / len(loops)
                    cy = sum(lp[uv].uv.y for lp in loops) / len(loops)
                    for lp in loops:
                        lp[uv].uv = (cx + (lp[uv].uv.x - cx) * k, cy + (lp[uv].uv.y - cy) * k)
                    scaled += 1
                bm.to_mesh(me)
            finally:
                bm.free()
        if not square:
            # SQUARE TEXELS ON A w x h TILE: a UV step du covers du*w pixels and dv covers dv*h, so V is stretched by
            # w / h and one world unit spans as many pixels across as up.
            flat_uv = array.array("f", [0.0]) * (2 * len(layer.data))
            layer.data.foreach_get("uv", flat_uv)
            for i in range(1, len(flat_uv), 2):
                flat_uv[i] *= tex[0] / tex[1]
            layer.data.foreach_set("uv", flat_uv)
            me.update()
        if target is not None:
            rows_t = _measure_islands(me, name, m3)
            uva, a3 = sum(r["_uva"] for r in rows_t), sum(r["_a3"] for r in rows_t)
            if uva <= 1e-12 or a3 <= 1e-12:
                raise MifOpError("no island has area in both UV and the world, so no density can be set")
            scale_to = target[0] / (uva * tex[0] * tex[1] / a3) ** 0.5
            _scale_layout_islands(me, name, scale_to)
        pinned = dict(_PACK_PINNED, rotate=rotate, scale=target is None)
        if merge is not None:
            pinned["merge_overlap"] = merge
        if not rotate:
            pinned.pop("rotate_method")
        _run_pack_islands(obj, name, margin, pinned)
        if region is not None:
            placed = array.array("f", [0.0]) * (2 * len(layer.data))
            me.uv_layers[name].data.foreach_get("uv", placed)
            side = region[2] - region[0] if tex is None else tex[0]
            for i in range(0, len(placed), 2):
                placed[i] = region[0] + placed[i] * side
                placed[i + 1] = region[1] + placed[i + 1] * (side if tex is None else side * tex[1] / tex[0])
            me.uv_layers[name].data.foreach_set("uv", placed)
            me.update()
    except Exception as exc:
        try:
            if obj.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:  # noqa: BLE001
            pass
        me.uv_layers[name].data.foreach_set("uv", saved)
        me.update()
        raise MifOpError("uv_pack failed on '%s': %s: %s. The layer was put back as it was." % (obj.name,
                         type(exc).__name__, exc))
    rows_after = _measure_islands(me, name, m3)
    _m, spread_after = _ratio_stats(rows_after)
    flat = [0.0] * (2 * len(me.uv_layers[name].data))
    me.uv_layers[name].data.foreach_get("uv", flat)
    us, vs = flat[0::2], flat[1::2]
    out = {"object": obj.name, "layer": name, "islands": len(rows_after), "margin": margin, "rotate": rotate,
           "equalizeByArea": equalize, "islandsScaled": scaled, "islandsWithoutArea": skipped,
           "ratioMedianBefore": round(med, 6) if med is not None else None,
           "ratioSpreadBefore": round(spread_before, 4) if spread_before is not None else None,
           "ratioSpreadAfter": round(spread_after, 4) if spread_after is not None else None,
           "uvAreaBefore": round(sum(r["_uva"] for r in rows_before), 6),
           "uvAreaAfter": round(sum(r["_uva"] for r in rows_after), 6),
           "bounds": {"min": [round(min(us), 6), round(min(vs), 6)], "max": [round(max(us), 6), round(max(vs), 6)]},
           "outside01": bool(min(us) < -1e-6 or min(vs) < -1e-6 or max(us) > 1 + 1e-6 or max(vs) > 1 + 1e-6),
           "activeLayer": me.uv_layers.active.name if me.uv_layers.active else None}
    if equalize and med is None:
        out["note"] = "no island has area in both UV and 3D, so nothing was equalised; the pack ran."
    if merge is not None:
        out["mergeOverlap"] = merge
    if region is not None:
        out["region"] = [round(x, 6) for x in region]
        out["insideRegion"] = bool(min(us) >= region[0] - 1e-6 and min(vs) >= region[1] - 1e-6
                                   and max(us) <= region[2] + 1e-6 and max(vs) <= region[3] + 1e-6)
    elif tex is not None:
        unit = target[1] if target is not None else "PX_PER_CM"
        out.update({"textureSize": [int(tex[0]), int(tex[1])], "vStretch": round(tex[0] / tex[1], 6),
                    "texelShape": texel_shape(obj, name, tex[0], tex[1]), "densityUnit": unit,
                    "densityAfter": density_stats([(r["_uva"] * tex[0] * tex[1] / r["_a3"]) ** 0.5
                                                   if r["_a3"] > 1e-12 and r["_uva"] > 1e-12 else None
                                                   for r in rows_after], unit)})
    if target is not None:
        out.update({"targetTexelDensity": round(target[2], 6), "targetPxPerMetre": round(target[0], 4),
                    "layoutScale": round(scale_to, 6)})
        if out["outside01"]:
            out["fitNote"] = ("the layout at this density does not fit one tile - a larger textureSize or a lower "
                              "target holds it (uv_set_texel_density's ladder names the smallest that does)")
    out.update(shared_data_note(obj))
    return out


_SEAM_KEYS = ("object", "name", "clear", "selector", "byMaterialBoundary", "byVertexGroupBoundary", "minWeight",
              "boundaryLoopNear", "path", "dryRun", "byBones", "armature", "mark", "value",
              "previewId", "previewTtlSeconds", "fromIslands")
_EDGE_MARKS = ("SEAM", "SHARP", "CREASE", "BEVEL_WEIGHT")
# WHERE CREASE AND BEVEL WEIGHT LIVE, measured 2026-10-04 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1: 3.6 has bmesh edge
# layers `crease` and `bevel_weight`; from 4.0 they are float EDGE attributes named crease_edge and bevel_weight_edge,
# reached in bmesh as edges.layers.float. A crease of 1 on every edge held a subdivided 2 m cube's corners at x 1.0 on
# all five. Sharp is BMEdge.smooth (False = sharp) on all five.
_MARK_ATTRS = {"CREASE": ("crease", "crease_edge"), "BEVEL_WEIGHT": ("bevel_weight", "bevel_weight_edge")}


def _edge_mark(bm, mark, create):
    """(get, put) for one edge mark on `bm`. A crease or bevel weight layer is made only when `create`; without one
    every edge reads 0 and put is None."""
    if mark == "SEAM":
        return (lambda e: e.seam), (lambda e, v: setattr(e, "seam", bool(v)))
    if mark == "SHARP":
        return (lambda e: not e.smooth), (lambda e, v: setattr(e, "smooth", not v))
    legacy_name, attr_name = _MARK_ATTRS[mark]
    legacy = getattr(bm.edges.layers, legacy_name, None)
    if legacy is not None:
        layer = legacy.verify() if create else legacy.active
    else:
        layer = bm.edges.layers.float.get(attr_name)
        if layer is None and create:
            layer = bm.edges.layers.float.new(attr_name)
    if layer is None:
        return (lambda e: 0.0), None
    return (lambda e: e[layer]), (lambda e, v: e.__setitem__(layer, float(v)))


def _mesh_mark_count(me, mark):
    """Edges carrying `mark`, read off the MESH (not the bmesh that wrote it)."""
    if mark == "SEAM":
        return sum(1 for e in me.edges if e.use_seam)
    if mark == "SHARP":
        return sum(1 for e in me.edges if e.use_edge_sharp)
    legacy_name, attr_name = _MARK_ATTRS[mark]
    attr = me.attributes.get(attr_name)
    if attr is not None:
        return sum(1 for d in attr.data if d.value > 0.0)
    return sum(1 for e in me.edges if getattr(e, legacy_name, 0.0) > 0.0)


def _bone_panels(obj, spec, armature):
    """byBones parsed and checked: (armature object, {bone: label} for the bones weighed, or a refusal).

    spec true: every deform bone of the armature is its own label. spec {bones: [...]}: only those bones are weighed.
    spec {panels: {name: [bones]}}: each listed bone is weighed under its panel's name; with panels and no bones, only the
    listed bones count. The armature is `armature`, else the mesh's Armature modifier's object, else its armature parent."""
    if armature is not None:
        arm = bpy.data.objects.get(armature)
        if arm is None or arm.type != "ARMATURE":
            raise MifOpError("%r is not an armature object. NOTHING was changed." % armature)
    else:
        mod = next((m for m in obj.modifiers if m.type == "ARMATURE" and m.object is not None), None)
        arm = mod.object if mod is not None else (obj.parent if obj.parent is not None
                                                  and obj.parent.type == "ARMATURE" else None)
        if arm is None:
            raise MifOpError("byBones needs the armature: '%s' has no Armature modifier with an object and no armature "
                             "parent - pass armature. NOTHING was changed." % obj.name)
    deform = {b.name for b in arm.data.bones if b.use_deform}
    if spec is True:
        return arm, {b: b for b in deform}
    if not isinstance(spec, dict) or not spec or set(spec) - {"bones", "panels"}:
        raise MifOpError("byBones is true, {bones: [...]} or {panels: {name: [bones]}}; got %r. NOTHING was changed."
                         % (spec,))
    labels = {}
    panels = spec.get("panels")
    if panels is not None:
        if not isinstance(panels, dict) or not panels or any(
                not isinstance(v, (list, tuple)) or not v for v in panels.values()):
            raise MifOpError("byBones.panels must be {name: [bone, ...]}, each list non-empty. NOTHING was changed.")
        for name, bones in panels.items():
            for b in bones:
                if b in labels:
                    raise MifOpError("bone %r is in two panels. NOTHING was changed." % b)
                labels[b] = name
    bones = spec.get("bones")
    if bones is not None:
        if not isinstance(bones, (list, tuple)) or not bones or any(not isinstance(b, str) for b in bones):
            raise MifOpError("byBones.bones must be a non-empty list of bone names. NOTHING was changed.")
        if panels is not None and set(labels) - set(bones):
            raise MifOpError("a panel names bones that byBones.bones leaves out: %s. NOTHING was changed."
                             % ", ".join(sorted(set(labels) - set(bones))))
        labels = {b: labels.get(b, b) for b in bones}
    unknown = sorted(b for b in labels if b not in deform)
    if unknown:
        raise MifOpError("armature '%s' has no deform bone %s. NOTHING was changed." % (arm.name, ", ".join(unknown)))
    return arm, labels
_PATH_KEYS = ("from", "to", "nearPlane", "planeWeight")


def _seam_path(bm, pos, start, goal, plane, weight):
    """The cheapest edge path from start to goal: Dijkstra over edge lengths, each scaled by 1 + weight x (the edge's
    mean distance from the plane / the mesh's mean edge length) when a plane is given - so the path hugs the plane
    and stays short. Returns (edges, verts) or None when goal cannot be reached."""
    import heapq
    mean_len = sum((pos[e.verts[0]] - pos[e.verts[1]]).length for e in bm.edges) / max(1, len(bm.edges)) or 1.0

    def cost(e):
        a, b = pos[e.verts[0]], pos[e.verts[1]]
        length = (a - b).length
        if plane is None:
            return length
        d = 0.5 * (abs((a - plane[0]).dot(plane[1])) + abs((b - plane[0]).dot(plane[1])))
        return length * (1.0 + weight * d / mean_len)

    dist = {start: 0.0}
    prev = {}
    heap = [(0.0, 0, start)]
    tie = 1
    while heap:
        dd, _t, v = heapq.heappop(heap)
        if v is goal:
            break
        if dd > dist.get(v, float("inf")):
            continue
        for e in v.link_edges:
            w = e.other_vert(v)
            nd = dd + cost(e)
            if nd < dist.get(w, float("inf")):
                dist[w] = nd
                prev[w] = (v, e)
                heapq.heappush(heap, (nd, tie, w))
                tie += 1
    if goal not in prev:
        return None
    edges, verts = [], [goal]
    v = goal
    while v is not start:
        u, e = prev[v]
        edges.append(e)
        verts.append(u)
        v = u
    return list(reversed(edges)), list(reversed(verts))


def op_mark_seams(params):
    """Mark (or clear) UV seams, sharp edges, crease or bevel weight by rule, without unwrapping: material and
    vertex-group boundaries, a boundary loop, the cheapest edge path between two points near a plane, or an edge
    selector.

    `mark` (2026-10-04) says which: SEAM (default), SHARP, CREASE or BEVEL_WEIGHT, with `value` - true (default) or
    false for SEAM and SHARP, 0 to 1 (default 1) for CREASE and BEVEL_WEIGHT; false or 0 clears it on the matched edges.
    A V-neck creased at 1, a fold held by a crease and released with 0, a seam loop's sharp mark moved to its
    neighbor, a solidify's rim weighted for a bevel - the rules below find the edges for every mark alike.
    previousValues counts the matched edges' values before the write; markedBefore and markedAfter count the edges
    carrying the mark, read off the mesh. Crease and bevel weight are written where this build keeps them (3.6: the
    edge's own layer; 4.0 on: the crease_edge and bevel_weight_edge attributes).

    THE GAP (row 7 of the spec's Why-Python table, and "Seams by rule"): the garment work marked seams in hand-written
    Python - plane cuts and material boundaries (reseam.py x6), panels from bone regions with an underarm seam as a
    shortest edge path (seam_regions.py x5), every edge of a boundary loop, a seam down the back of each tube
    (tube_seam.py). uv_unwrap could mark seams only inside an unwrap, by angle or index.

    RULES, any number per call, their edges joined (each rule's own count is reported, and a rule that matches no edge
    is refused - an unwrap after it would behave as though it had never been asked):
      byBones               panels from skin weights: each face takes the deform bone (or panel) with the most summed
                            weight over its vertices, and a seam goes on every edge between faces that differ. true: every
                            deform bone of the armature; {bones: [...]}: only those are weighed (the rest of the mesh is
                            one unlabelled region, so the seam outlines them); {panels: {name: [bones]}}: bones grouped
                            as vertex_groups_from_bones groups them. armature defaults to the Armature modifier's object,
                            then the parent; bones.facesByLabel says how many faces each label took
      byMaterialBoundary    edges between faces on different MATERIALS (slots holding the same material are one)
      byVertexGroupBoundary edges between faces in a vertex group and faces out of it; a face is in when every one of its
                            vertices weighs at least minWeight (default 0.5)
      boundaryLoopNear      every edge of the open boundary loop with the vertex nearest this world point
      path                  {from, to: world points; nearPlane: {point, normal} (optional); planeWeight (default 10)} -
                            the cheapest edge path between the vertices nearest from and to, each edge's length scaled
                            up by its distance from the plane, so a back-centre seam follows the spine plane
      selector              select_edges' grammar (minAngleDeg, boundaryOnly, edgeIndices, axis + side, allEdges)
      fromIslands           true (the active UV layer) or a UV layer's name: every edge on an island border of that
                            layer - an edge between two faces whose UVs differ at either end (uv.seams_from_islands,
                            5.2 bl_ui/space_image.py:473). An imported mesh arrives with UVs and no seams; marked, a
                            later unwrap or relax keeps the layout. The mesh's own open boundary is not marked.
    clear:true removes the mark from every edge first (alone: clear only). dryRun reports and writes nothing. World
    units, the object's origin as the working origin. Seams are read back off the mesh (seamsAfter, for SEAM).
    """
    reject_unknown(params, _SEAM_KEYS, "mark_seams")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    require_editable(obj, "mark seams on")
    clear = take_bool(params, "clear", default=False)
    dry = take_bool(params, "dryRun", default=False)
    by_mat = take_bool(params, "byMaterialBoundary", default=False)
    group = take(params, "byVertexGroupBoundary", default=None, kind=str)
    min_w = take_float(params, "minWeight", default=0.5)
    loop_near = _point_param(params, "boundaryLoopNear")
    selector = params.get("selector")
    path = params.get("path")
    mark = str(take(params, "mark", default="SEAM", kind=str)).upper()
    if mark not in _EDGE_MARKS:
        raise MifOpError("mark must be one of %s (got %r). NOTHING was changed." % (", ".join(_EDGE_MARKS), mark))
    if mark in ("SEAM", "SHARP"):
        value = take_bool(params, "value", default=True)
    else:
        value = take_float(params, "value", default=1.0)
        if not 0.0 <= value <= 1.0:
            raise MifOpError("value for %s must be 0 to 1 (got %r); 0 clears it. NOTHING was changed." % (mark, value))
    if params.get("minWeight") is not None and group is None:
        raise MifOpError("minWeight goes with byVertexGroupBoundary. NOTHING was changed.")
    if not 0.0 < min_w <= 1.0:
        raise MifOpError("minWeight must be above 0 and at most 1, got %r. NOTHING was changed." % min_w)
    if group is not None and group not in obj.vertex_groups:
        raise MifOpError("'%s' has no vertex group named %r (it has %s). NOTHING was changed."
                         % (obj.name, group, ", ".join(g.name for g in obj.vertex_groups) or "none"))
    if selector is not None and not isinstance(selector, dict):
        raise MifOpError("selector is select_edges' grammar as an object, e.g. {minAngleDeg: 40}; got %r. NOTHING was "
                         "changed." % (selector,))
    p_from = p_to = plane = None
    weight = 10.0
    if path is not None:
        if not isinstance(path, dict):
            raise MifOpError("path must be {from: [x,y,z], to: [x,y,z], nearPlane: {point, normal}, planeWeight}; got "
                             "%r. NOTHING was changed." % (path,))
        unknown = sorted(set(path) - set(_PATH_KEYS))
        if unknown:
            raise MifOpError("path does not take %s - it takes %s. NOTHING was changed."
                             % (", ".join(unknown), ", ".join(_PATH_KEYS)))
        p_from, p_to = _point_param(path, "from"), _point_param(path, "to")
        if p_from is None or p_to is None:
            raise MifOpError("path needs both from and to (world points). NOTHING was changed.")
        weight = take_float(path, "planeWeight", default=10.0)
        if weight < 0:
            raise MifOpError("path.planeWeight must be 0 or more, got %r. NOTHING was changed." % weight)
        near = path.get("nearPlane")
        if near is not None:
            if not isinstance(near, dict) or set(near) - {"point", "normal"}:
                raise MifOpError("path.nearPlane must be {point: [x,y,z], normal: [x,y,z]}; got %r. NOTHING was "
                                 "changed." % (near,))
            pp, pn = _point_param(near, "point"), _point_param(near, "normal")
            if pp is None or pn is None or pn.length < 1e-12:
                raise MifOpError("path.nearPlane needs a point and a non-zero normal. NOTHING was changed.")
            plane = (pp, pn.normalized())
    islands = params.get("fromIslands")
    if islands is False:
        islands = None
    if islands is not None and islands is not True and (not isinstance(islands, str) or not islands):
        raise MifOpError("fromIslands is true (the active UV layer) or a UV layer's name, got %r. NOTHING was changed."
                         % (islands,))
    if isinstance(islands, str) and obj.data.uv_layers.get(islands) is None:
        raise MifOpError("'%s' has no UV layer %r (it has %s). NOTHING was changed."
                         % (obj.name, islands, ", ".join(u.name for u in obj.data.uv_layers) or "none"))
    if islands is True and obj.data.uv_layers.active is None:
        raise MifOpError("'%s' has no UV layer, so it has no islands. NOTHING was changed." % obj.name)
    by_bones = params.get("byBones")
    if by_bones is False:
        by_bones = None
    if params.get("armature") is not None and by_bones is None:
        raise MifOpError("armature goes with byBones. NOTHING was changed.")
    bone_arm = bone_labels = None
    if by_bones is not None:
        bone_arm, bone_labels = _bone_panels(obj, by_bones, take(params, "armature", kind=str))
    asked = [k for k, v in (("byBones", by_bones is not None),
                            ("byMaterialBoundary", by_mat), ("byVertexGroupBoundary", group is not None),
                            ("boundaryLoopNear", loop_near is not None), ("path", path is not None),
                            ("selector", selector is not None), ("fromIslands", islands is not None)) if v]
    if not asked and not clear:
        raise MifOpError("say which %s: byBones, byMaterialBoundary, byVertexGroupBoundary, boundaryLoopNear, path, "
                         "fromIslands or selector - or clear:true to remove the %s mark from them all. NOTHING was changed."
                         % ("seams" if mark == "SEAM" else "edges", mark.lower()))
    mw = obj.matrix_world
    shift = mw.translation.copy()
    work = _shifted(mw, shift)
    me = obj.data
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        pos = {v: work @ v.co for v in bm.verts}
        get, put = _edge_mark(bm, mark, create=not dry)
        before = sum(1 for e in bm.edges if get(e))
        found, details = {}, {}
        if by_mat:
            mats = [s.material.name if s.material else None for s in obj.material_slots]

            def mat_of(f):
                return mats[f.material_index] if 0 <= f.material_index < len(mats) else None
            found["byMaterialBoundary"] = [e for e in bm.edges if len(e.link_faces) == 2
                                           and mat_of(e.link_faces[0]) != mat_of(e.link_faces[1])]
        if bone_labels is not None:
            # EACH FACE TAKES THE LABEL WITH THE MOST SUMMED WEIGHT over its vertices (ties: the name first in order), a
            # face with none of the weighed bones takes none; a seam is every edge between two faces that differ.
            gl = {g.index: bone_labels[g.name] for g in obj.vertex_groups if g.name in bone_labels}
            group_names = {g.name for g in obj.vertex_groups}        # once, not a bpy lookup per bone
            vw = [{} for _ in me.vertices]
            for v in me.vertices:
                for g in v.groups:
                    lab = gl.get(g.group)
                    if lab is not None and g.weight > 0.0:
                        vw[v.index][lab] = vw[v.index].get(lab, 0.0) + g.weight
            face_label = {}
            for f in bm.faces:
                tot = {}
                for v in f.verts:
                    for lab, w in vw[v.index].items():
                        tot[lab] = tot.get(lab, 0.0) + w
                face_label[f] = min(tot, key=lambda k: (-tot[k], k)) if tot else None
            found["byBones"] = [e for e in bm.edges if len(e.link_faces) == 2
                                and face_label[e.link_faces[0]] != face_label[e.link_faces[1]]]
            counts_by = {}
            for lab in face_label.values():
                counts_by[str(lab)] = counts_by.get(str(lab), 0) + 1
            details["bones"] = {"armature": bone_arm.name, "facesByLabel": counts_by,
                                "groupsWeighed": sorted(n for n in bone_labels if n in group_names)}
        if group is not None:
            gi = obj.vertex_groups[group].index
            heavy = {v.index for v in me.vertices for g in v.groups if g.group == gi and g.weight >= min_w}

            def inside(f):
                return all(v.index in heavy for v in f.verts)
            found["byVertexGroupBoundary"] = [e for e in bm.edges if len(e.link_faces) == 2
                                              and inside(e.link_faces[0]) != inside(e.link_faces[1])]
        if loop_near is not None:
            loops = _boundary_loops(bm)
            if not loops:
                raise MifOpError("'%s' has no open boundary, so boundaryLoopNear has no loop. NOTHING was changed."
                                 % obj.name)
            loop, d = _loop_nearest(loops, loop_near - shift, work)
            found["boundaryLoopNear"] = list(loop[0])
            details["boundaryLoop"] = {"edges": len(loop[0]), "loops": len(loops), "distance": round(d, 6)}
        if path is not None:
            a = min(bm.verts, key=lambda v: (pos[v] - (p_from - shift)).length)
            b = min(bm.verts, key=lambda v: (pos[v] - (p_to - shift)).length)
            if a is b:
                raise MifOpError("path's from and to are nearest the same vertex (%d), so there is no path. NOTHING was "
                                 "changed." % a.index)
            got = _seam_path(bm, pos, a, b, None if plane is None else (plane[0] - shift, plane[1]), weight)
            if got is None:
                raise MifOpError("no edge path joins vertex %d to vertex %d - they are on separate pieces. NOTHING was "
                                 "changed." % (a.index, b.index))
            pe, pv = got
            found["path"] = pe
            details["path"] = {
                "vertices": [v.index for v in pv], "edges": len(pe),
                "length": round(sum((pos[e.verts[0]] - pos[e.verts[1]]).length for e in pe), 6),
                "maxPlaneDistance": round(max(abs((pos[v] - (plane[0] - shift)).dot(plane[1])) for v in pv), 6)
                if plane is not None else None}
        if islands is not None:
            lname = islands if isinstance(islands, str) else me.uv_layers.active.name
            uvl = bm.loops.layers.uv.get(lname)

            def face_uvs(f, e):
                lp = next(lo for lo in f.loops if lo.edge == e)
                return {lp.vert: lp[uvl].uv.copy(), lp.link_loop_next.vert: lp.link_loop_next[uvl].uv.copy()}
            border = []
            for e in bm.edges:
                if len(e.link_faces) != 2:
                    continue
                a, b = face_uvs(e.link_faces[0], e), face_uvs(e.link_faces[1], e)
                if any((a[v] - b[v]).length > 1e-5 for v in e.verts):
                    border.append(e)
            found["fromIslands"] = border
            details["fromIslands"] = {"uvLayer": lname, "edges": len(border)}
        if selector is not None:
            edges, criteria = _select_edges(bm, obj, dict(selector), "mark_seams.selector")
            found["selector"] = list(edges)
            details["selectorCriteria"] = criteria
        empty = [k for k, v in found.items() if not v]
        if empty:
            raise MifOpError("%s matched no edge on '%s' - whatever reads the %s mark next would behave as though none "
                             "had been asked for. NOTHING was changed." % (" and ".join(empty), obj.name, mark.lower()))
        union = list(dict.fromkeys(e for es in found.values() for e in es))
        previous = {}
        for e in union:
            was = get(e)
            k = str(bool(was)).lower() if mark in ("SEAM", "SHARP") else str(round(float(was), 4))
            previous[k] = previous.get(k, 0) + 1
        cleared = 0
        off = False if mark in ("SEAM", "SHARP") else 0.0
        if clear:
            for e in bm.edges:
                if get(e):
                    cleared += 1
                    if not dry:
                        put(e, off)

        def differs(e):
            return (bool(get(e)) != bool(value)) if mark in ("SEAM", "SHARP") else abs(get(e) - value) > 1e-6
        marked = sum(1 for e in union if differs(e) or (clear and dry and bool(value)))
        if not dry:
            for e in union:
                put(e, value)
            bm.to_mesh(me)
    finally:
        bm.free()
    if not dry:
        me.update()
    after = _mesh_mark_count(me, mark)
    out = {"object": obj.name, "mark": mark, "value": value, "rules": {k: len(v) for k, v in found.items()},
           "edgesMatched": len(union), "marked": marked, "cleared": cleared, "previousValues": previous,
           "markedBefore": before, "markedAfter": after, "dryRun": dry}
    if mark == "SEAM":
        out["seamsBefore"], out["seamsAfter"] = before, after
    out.update(details)
    if dry:
        out["note"] = "dryRun: nothing was written; markedAfter is the mesh as it is."
    out.update(shared_data_note(obj))
    return out


_SKIN_KEYS = ("object", "name", "vertices", "all", "group", "radius", "scale", "root", "loose", "read", "limit")


def _skin_piece(v):
    """The vertices joined to v by edges (its connected piece) - where one root lives."""
    seen, stack = {v}, [v]
    while stack:
        cur = stack.pop()
        for e in cur.link_edges:
            o = e.other_vert(cur)
            if o not in seen:
                seen.add(o)
                stack.append(o)
    return seen


def op_set_skin_vertices(params):
    """Write a Skin modifier's per-vertex data: the radius across x and y, the root and the loose flag.

    THE GAP: add_modifier makes a SKIN modifier, but how thick each vertex is lives on the MESH (mesh.skin_vertices),
    not on the modifier, so a hand grown from one vertex - fattened, extruded into a forearm, the root at the wrist,
    the palm vertex loose so five digits branch from it, the digits slimmed - had no call.

    vertices (indices) | all true | group (a vertex group: the vertices with weight above 0) pick the vertices.
    radius (a number, or [x, y]) sets the radius; scale multiplies the radius each has. root true makes each vertex
    the root of its piece and clears the piece's other root, as Blender's Mark Root does (measured 2026-10-05 on 3.6.23
    and 5.2.1); two roots asked in one piece are refused. loose true lets the vertex's branches leave it freely.
    Measured on 3.6.23, 4.4.0 and 5.2.1: the layer holds radius (x, y), use_root and use_loose; adding a Skin modifier
    makes it with 0.25 and the first vertex as root; radius (0.5, 0.25) on a 2 m chain evaluated 1.0 x 0.5 x 2.0. With
    no layer one is made (radius 0.25, no root - the modifier picks its own). Reply: the radii written (index, x, y; 50
    listed), roots, looseCount, rootsCleared, skinLayerCreated and hasSkinModifier.

    read:true (1.3) lists the layer instead - each vertex's radius, root and loose (vertices narrows it, limit default 500)
    - and writes nothing; mesh_stats attributeValues cannot reach it (the skin layer is not a generic attribute).
    """
    reject_unknown(params, _SKIN_KEYS, "set_skin_vertices")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    if take_bool(params, "read", default=False):
        # THE READ HALF (1.3 gap audit, 2026-10-19): the skin layer is not a generic attribute - mesh_stats
        # attributeValues does not list it (measured on 3.6.23 and 5.2.1; face sets it does read, as .sculpt_face_set)
        # - so a later session could not see what this op wrote. read:true lists it and writes nothing.
        stray = [k for k in ("radius", "scale", "root", "loose", "all", "group") if params.get(k) is not None]
        if stray:
            raise MifOpError("read lists the skin layer and takes vertices and limit only (got %s). NOTHING was "
                             "changed." % stray)
        me = obj.data
        if not len(me.skin_vertices):
            return {"object": obj.name, "read": True, "skinLayer": False, "vertices": [],
                    "hasSkinModifier": any(m.type == "SKIN" for m in obj.modifiers)}
        data = me.skin_vertices[0].data
        want = params.get("vertices")
        idx = list(range(len(data))) if want is None else [i for i in want if isinstance(i, int) and 0 <= i < len(data)]
        limit = take_int(params, "limit", default=500)
        rows = [{"index": i, "radius": [round(data[i].radius[0], 6), round(data[i].radius[1], 6)],
                 "root": bool(data[i].use_root), "loose": bool(data[i].use_loose)} for i in idx[:limit]]
        return {"object": obj.name, "read": True, "skinLayer": True, "count": len(idx), "vertices": rows,
                "truncated": len(idx) > limit, "roots": [i for i in range(len(data)) if data[i].use_root][:100],
                "hasSkinModifier": any(m.type == "SKIN" for m in obj.modifiers)}
    if params.get("limit") is not None:
        raise MifOpError("limit bounds read:true's listing. NOTHING was changed.")
    require_editable(obj, "set the skin radii of")
    me = obj.data
    picks = [k for k in ("vertices", "all", "group") if params.get(k) is not None]
    if len(picks) != 1:
        raise MifOpError("pick the vertices with exactly one of vertices (indices), all (true) or group (a vertex group) "
                         "- got %s. NOTHING was changed." % (picks or "none"))
    radius = params.get("radius")
    if radius is not None:
        if isinstance(radius, (list, tuple)) and len(radius) == 2:
            rx, ry = finite_floats(list(radius), "radius")
        elif isinstance(radius, (int, float)) and not isinstance(radius, bool):
            rx = ry = finite_float(radius, "radius")
        else:
            raise MifOpError("radius is a number or [x, y], got %r. NOTHING was changed." % (radius,))
        if rx <= 0 or ry <= 0:
            raise MifOpError("radius must be above 0, got %r. NOTHING was changed." % (radius,))
    scale = take_float(params, "scale", default=None)
    if scale is not None and not scale > 0:
        raise MifOpError("scale multiplies the radius and must be above 0, got %g. NOTHING was changed." % scale)
    if radius is not None and scale is not None:
        raise MifOpError("pass radius (set it) OR scale (multiply it), not both. NOTHING was changed.")
    root = take_bool(params, "root") if params.get("root") is not None else None
    loose = take_bool(params, "loose") if params.get("loose") is not None else None
    if radius is None and scale is None and root is None and loose is None:
        raise MifOpError("nothing to write - pass radius, scale, root or loose. NOTHING was changed.")
    n = len(me.vertices)
    if params.get("all") is not None:
        if take_bool(params, "all") is not True:
            raise MifOpError("all takes true; pick vertices or a group instead. NOTHING was changed.")
        idx = list(range(n))
    elif params.get("group") is not None:
        gname = take(params, "group", kind=str)
        vg = obj.vertex_groups.get(gname)
        if vg is None:
            raise MifOpError("'%s' has no vertex group %r (it has %s). NOTHING was changed."
                             % (obj.name, gname, ", ".join(g.name for g in obj.vertex_groups) or "none"))
        idx = [v.index for v in me.vertices if any(g.group == vg.index and g.weight > 0.0 for g in v.groups)]
        if not idx:
            raise MifOpError("vertex group %r holds no vertex with weight above 0. NOTHING was changed." % gname)
    else:
        raw = params["vertices"]
        if not isinstance(raw, (list, tuple)) or not raw:
            raise MifOpError("vertices is a non-empty list of vertex indices. NOTHING was changed.")
        bad = [i for i in raw if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n]
        if bad:
            raise MifOpError("vertices %s are not vertex indices of '%s' (it has %d). NOTHING was changed."
                             % (bad[:10], obj.name, n))
        idx = list(dict.fromkeys(raw))
    created = len(me.skin_vertices) == 0
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        lay = bm.verts.layers.skin.verify()
        verts = [bm.verts[i] for i in idx]
        cleared = []
        if root:
            pieces = {}
            for v in verts:
                piece = frozenset(_skin_piece(v))
                if piece in pieces:
                    raise MifOpError("vertices %d and %d are in one connected piece, which has one root. NOTHING was "
                                     "changed." % (pieces[piece].index, v.index))
                pieces[piece] = v
            for piece, v in pieces.items():
                for o in piece:
                    if o is not v and o[lay].use_root:
                        o[lay].use_root = False
                        cleared.append(o.index)
        for v in verts:
            sv = v[lay]
            if radius is not None:
                sv.radius = (rx, ry)
            elif scale is not None:
                sv.radius = (sv.radius[0] * scale, sv.radius[1] * scale)
            if root is not None:
                sv.use_root = root
            if loose is not None:
                sv.use_loose = loose
        bm.to_mesh(me)
    finally:
        bm.free()
    me.update()
    data = me.skin_vertices[0].data
    written = [[i, round(data[i].radius[0], 6), round(data[i].radius[1], 6)] for i in idx]
    out = {"object": obj.name, "vertices": len(idx), "radii": written[:50], "radiiTruncated": len(written) > 50,
           "roots": [i for i in range(n) if data[i].use_root][:50],
           "looseCount": sum(1 for i in range(n) if data[i].use_loose),
           "rootsCleared": sorted(cleared) or None, "skinLayerCreated": created,
           "hasSkinModifier": any(m.type == "SKIN" for m in obj.modifiers)}
    if not out["hasSkinModifier"]:
        out["note"] = "'%s' has no Skin modifier - these values shape nothing until one is added (add_modifier SKIN)." \
                      % obj.name
    out.update(shared_data_note(obj))
    return out


_RMA_KINDS = ("CUSTOM_NORMALS", "SHARP_EDGE", "SHARP_FACE", "COLOR")


def _custom_normals_clear(obj):
    """Clear a mesh's custom normals through Blender's own operator - the one route that works on every build.

    MEASURED 2026-10-05: custom normals are an attribute named custom_normal from 4.4 (removing it clears them) and a
    layer that is no attribute at all on 4.2; mesh.customdata_custom_splitnormals_clear under a temp_override cleared
    them headless on 4.2.23, 4.4.0, 5.0.1 and 5.2.2."""
    with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj],
                                   selected_editable_objects=[obj]):
        bpy.ops.mesh.customdata_custom_splitnormals_clear()


def op_remove_mesh_attributes(params):
    """Take attributes off a mesh: by name, or by kind - CUSTOM_NORMALS, SHARP_EDGE, SHARP_FACE, COLOR.

    THE GAP, 2026-10-04: after a round trip through an exchange format a custom-normals attribute and a sharp-edge
    attribute had to come off before the mesh behaved - an extrude misbehaved until the custom normals were cleared - and
    set_vertex_color and the kit helpers only ever add attributes.

    names: attributes by name (mesh_stats lists them); a UV map named here goes through uv_layers. kinds: CUSTOM_NORMALS
    (Blender's Clear Custom Split Normals - see _custom_normals_clear), SHARP_EDGE (the sharp_edge attribute: no edge is
    marked sharp after), SHARP_FACE (sharp_face: EVERY FACE SHADES SMOOTH after, since a face with no sharp_face value
    is smooth), COLOR (every colour attribute). Every name is checked before anything is removed: a required attribute
    (position), an internal one (a name starting with '.', Blender's own bookkeeping) or one the mesh lacks is refused.
    A kind with nothing on the mesh is listed under notPresent; a call that would remove nothing at all is refused.
    Reply: removed [{name, domain, dataType}], notPresent, remaining (the attributes left, internal ones not listed),
    customNormalsBefore / After.
    """
    reject_unknown(params, ("object", "name", "names", "kinds"), "remove_mesh_attributes")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    require_editable(obj, "remove attributes of")
    me = obj.data
    names = params.get("names")
    kinds = params.get("kinds")
    if names is None and kinds is None:
        raise MifOpError("pass names or kinds (attribute names; %s), or both. NOTHING was changed."
                         % ", ".join(_RMA_KINDS))
    for key, val in (("names", names), ("kinds", kinds)):
        if val is not None and (not isinstance(val, (list, tuple)) or not val
                                or not all(isinstance(x, str) and x for x in val)):
            raise MifOpError("%s is a non-empty list of strings, got %r. NOTHING was changed." % (key, val))
    kinds = [str(k).upper() for k in kinds or []]
    bad = [k for k in kinds if k not in _RMA_KINDS]
    if bad:
        raise MifOpError("kinds %s are not among %s. NOTHING was changed." % (bad, ", ".join(_RMA_KINDS)))
    uv_names = {lay.name for lay in me.uv_layers}
    for n in names or []:
        a = me.attributes.get(n)
        if a is None and n not in uv_names:
            listed = [x.name for x in me.attributes if not x.name.startswith(".")]
            raise MifOpError("'%s' has no attribute %r (it has %s). NOTHING was changed."
                             % (obj.name, n, ", ".join(listed) or "none"))
        if n.startswith(".") or getattr(a, "is_internal", False):
            raise MifOpError("%r is internal - Blender's own bookkeeping (selection, topology), not data to remove. "
                             "NOTHING was changed." % n)
        if getattr(a, "is_required", False):
            raise MifOpError("%r is required: a mesh cannot exist without it. NOTHING was changed." % n)
    custom_before = bool(getattr(me, "has_custom_normals", False))
    plan, not_present = [], []                       # plan: (kind label, attribute name or None)
    for n in names or []:
        plan.append(("NAME", n))
    for k in kinds:
        if k == "CUSTOM_NORMALS":
            (plan if custom_before else not_present).append(("CUSTOM_NORMALS", None) if custom_before else k)
        elif k in ("SHARP_EDGE", "SHARP_FACE"):
            an = k.lower()
            (plan.append((k, an)) if me.attributes.get(an) is not None else not_present.append(k))
        else:
            cols = [c.name for c in getattr(me, "color_attributes", [])]
            if cols:
                plan.extend((k, c) for c in cols)
            else:
                not_present.append(k)
    plan = list(dict.fromkeys(plan))
    if not plan:
        raise MifOpError("nothing to remove: '%s' has none of %s. NOTHING was changed."
                         % (obj.name, ", ".join(not_present)))
    removed, done = [], set()
    for kind, an in plan:
        if kind == "CUSTOM_NORMALS":
            _custom_normals_clear(obj)
            removed.append({"name": "custom normals", "kind": "CUSTOM_NORMALS"})
            done.add("custom_normal")
            continue
        if an in done:
            continue
        done.add(an)
        a = me.attributes.get(an)
        if a is None and an in {lay.name for lay in me.uv_layers}:
            me.uv_layers.remove(me.uv_layers[an])
            removed.append({"name": an, "domain": "CORNER", "dataType": "FLOAT2", "kind": "UV"})
            continue
        if a is None:
            continue
        row = {"name": an, "domain": a.domain, "dataType": a.data_type, "kind": kind}
        if an == "custom_normal":
            _custom_normals_clear(obj)
        else:
            me.attributes.remove(a)
        removed.append(row)
    me.update()
    bpy.context.view_layer.update()
    out = {"object": obj.name, "removed": removed, "notPresent": not_present or None,
           "remaining": [{"name": a.name, "domain": a.domain, "dataType": a.data_type} for a in me.attributes
                         if not a.name.startswith(".")],
           "customNormalsBefore": custom_before, "customNormalsAfter": bool(getattr(me, "has_custom_normals", False))}
    notes = []
    if any(r.get("name") == "sharp_face" for r in removed):
        notes.append("sharp_face is gone, so every face now shades smooth - set_shading flat brings flat shading back")
    if any(r.get("name") == "sharp_edge" for r in removed):
        notes.append("sharp_edge is gone, so no edge is marked sharp")
    if notes:
        out["note"] = "; ".join(notes) + "."
    out.update(shared_data_note(obj))
    return out


# ---------------------------------------------------------------------------
# export_vertex_positions
# ---------------------------------------------------------------------------

_VERTEX_POSITION_KEYS = {"object", "name", "file", "filepath", "path", "root", "overwrite", "shape_key", "shapeKey"}


def op_export_vertex_positions(params):
    """A mesh's vertex positions written to a .json file in Unreal's mesh frame - centimeters, Y negated - in the mesh's
    own vertex order: the file set_skeletal_mesh_vertices reads (its `file`), {"positions": [[x, y, z], ...]}.

    WHY (2026-10-09). set_skeletal_mesh_vertices takes positions in the order export_asset's FBX writes them, which a
    Blender import of that FBX keeps - so a skeletal mesh edited in Blender could go back only as a positions list
    built by hand from the Blender mesh, axis and unit included, through the caller.

    THE FRAME, MEASURED 2026-10-09 (5.8.2 export_asset, Blender 5.0.1 import_mesh): a skeletal mesh's FBX comes in as an
    EMPTY named after the asset (scale 0.01, the importer's centimeters) holding the ARMATURE (scale 100) holding the
    MESH (scale 0.01); the mesh's vertex `co` is the file's own centimeters, export index for index, Unreal's Y negated
    - export vertex 0 at Unreal (-20, -20, 120) is Blender vertex 0 at (-20, 20, 120). So a position is taken through
    the object's matrix into `root`'s frame (default: the object's topmost parent, the import's empty) and its Y
    negated: moving or turning the whole import in Blender changes nothing, moving the mesh against its armature does.

    BASE MESH, NOT THE MODIFIER RESULT: the armature modifier poses the mesh, and the positions wanted are the bind
    pose's. Shape keys: the base mesh is the Basis; the reply says when other keys exist. Edit mode is refused - the
    live edits are in the edit BMesh, not the mesh this reads.

    params:
      object / name (str)     required, a MESH in OBJECT mode
      file (str)              required, a .json path - written {"positions": [...]} (replaced by default, and said)
      root (str)              the object whose frame is the file's: an ancestor of `object`, or `object` itself.
                              Default the topmost parent
      overwrite (bool)        default true; false refuses an existing file
      shape_key (str)         a shape key's name: that key's coordinates, in the same frame and vertex order, instead of the
                              base mesh's - so a sculpted key goes to Unreal as a morph target in one call (2026-10-11).
                              A name the mesh lacks is refused with the ones it has
    """
    reject_unknown(params, _VERTEX_POSITION_KEYS, "export_vertex_positions")
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    raw = take(params, "file", "filepath", "path", required=True, kind=str)
    if not str(raw).lower().endswith(".json"):
        raise MifOpError("'file' must be a .json path - set_skeletal_mesh_vertices reads its positions from JSON. Got "
                         "%r. NOTHING was written." % raw)
    overwrite = take_bool(params, "overwrite", default=True)
    root_name = take(params, "root", default=None, kind=str)
    key_name = take(params, "shape_key", "shapeKey", default=None, kind=str)
    key_block = None
    if key_name is not None:
        kb = getattr(getattr(obj.data, "shape_keys", None), "key_blocks", None)
        key_block = kb.get(key_name) if kb is not None else None
        if key_block is None:
            raise MifOpError("'%s' has no shape key %r (it has %s). NOTHING was written."
                             % (obj.name, key_name, ", ".join(k.name for k in kb) if kb else "none"))
    chain, cur = [], obj
    while cur is not None:
        chain.append(cur)
        cur = cur.parent
    if root_name is None:
        root = chain[-1]
    else:
        root = next((o for o in chain if o.name == root_name), None)
        if root is None:
            raise MifOpError("root '%s' is not '%s' or one of its parents (%s). NOTHING was written."
                             % (root_name, obj.name, " > ".join(o.name for o in chain)))
    if abs(root.matrix_world.determinant()) < 1e-12:
        raise MifOpError("root '%s' has a zero scale, so no position can be taken into its frame. NOTHING was written."
                         % root.name)
    path = _resolve_out_path(raw)
    check_output_path(raw, path, "written")
    if os.path.exists(path) and not overwrite:
        raise MifOpError("%s already exists and overwrite:false. Pass overwrite:true or a different file. NOTHING was "
                         "written." % path)

    to_root = root.matrix_world.inverted() @ obj.matrix_world
    mesh = obj.data
    positions = []
    lo, hi = [math.inf] * 3, [-math.inf] * 3
    for v in mesh.vertices:
        p = to_root @ (key_block.data[v.index].co if key_block is not None else v.co)
        row = [round(float(p.x), 5), round(float(-p.y), 5), round(float(p.z), 5)]
        if not all(math.isfinite(c) for c in row):
            raise MifOpError("vertex %d of '%s' is not a finite position (%s). NOTHING was written."
                             % (v.index, obj.name, row))
        for i in range(3):
            lo[i], hi[i] = min(lo[i], row[i]), max(hi[i], row[i])
        positions.append(row)
    if not positions:
        raise MifOpError("'%s' has no vertices. NOTHING was written." % obj.name)
    out = {"object": obj.name, "file": path, "vertexCount": len(positions), "root": root.name,
           "rootIsObject": root is obj, "boundsCm": {"min": rnd(lo), "max": rnd(hi)},
           "frame": "Unreal mesh space: centimeters, Y negated, in %s's frame" % root.name}
    warnings = []
    keys = getattr(mesh, "shape_keys", None)
    if key_block is not None:
        out["shapeKey"] = key_block.name
        out["shapeKeys"] = [k.name for k in keys.key_blocks]
    elif keys is not None and len(keys.key_blocks) > 1:
        out["shapeKeys"] = [k.name for k in keys.key_blocks]
        warnings.append("'%s' has %d shape keys: these are the base mesh's positions (%s); edits made on another key "
                        "are not in them." % (obj.name, len(keys.key_blocks), keys.key_blocks[0].name))
    shaping = [m.name for m in obj.modifiers if m.type != "ARMATURE" and m.show_viewport]
    if shaping:
        warnings.append("modifiers %s are not applied - these are the base mesh's positions." % ", ".join(shaping))
    ident = Matrix.Identity(4)
    if any(abs(to_root[r][c] - ident[r][c]) > 1e-6 for r in range(4) for c in range(4)):
        out["objectToRoot"] = [rnd(list(r)) for r in to_root]
        warnings.append("'%s' is moved, turned or scaled against '%s', and the positions carry it." % (obj.name,
                                                                                                      root.name))
    # THE WRITE LAST, so every refusal and every reply field above comes before a file exists.
    before = file_before(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"positions": positions, "object": obj.name, "root": root.name,
                   "frame": "unreal mesh space, cm (Y negated)"}, fh)
    out.update(replaced_fields(before))
    if warnings:
        out["warnings"] = warnings
    return out


# ---------------------------------------------------------------------------------------------------------------------
# THE BULK PREVIEWS of set_material_slots and transform_selection (ops_preview.py; docs/15 section 5).
# ---------------------------------------------------------------------------------------------------------------------
def _plan_set_material_slots(request):
    resp = op_set_material_slots(dict(request, dryRun=True))     # every refusal, and writes nothing
    obj = bpy.data.objects[resp["object"]]
    before, after = resp["before"], resp["materialSlots"]
    links_before, links_after = resp["linksBefore"], resp["links"]
    changes = [{"kind": "material", "op": "create", "target": "materials:%s" % n, "before": None, "after": "exists"}
               for n in resp["wouldCreateMaterials"]]
    unchanged = []
    for i in range(max(len(before), len(after))):
        has_b, has_a = i < len(before), i < len(after)
        row = {"kind": "slot", "target": obj.name, "field": "material_slots[%d]" % i,
               "before": {"material": before[i], "link": links_before[i]} if has_b else None,
               "after": {"material": after[i], "link": links_after[i]} if has_a else None}
        row["op"] = "modify" if has_b and has_a else "create" if has_a else "delete"
        (unchanged if row["before"] == row["after"] else changes).append(row)
    commit = {k: v for k, v in request.items() if k != "dryRun"}
    return {"changes": changes, "unchanged": unchanged, "commit": commit, "context": [], "response": resp}


def _read_set_material_slots(change, _kept):
    if change["kind"] == "material":
        return "exists" if bpy.data.materials.get(change["target"].partition(":")[2]) is not None else None
    obj = bpy.data.objects.get(change["target"])
    i = int(change["field"][len("material_slots["):-1])
    if obj is None or i >= len(obj.material_slots):
        return None
    slot = obj.material_slots[i]
    return {"material": slot.material.name if slot.material else None, "link": slot.link}


def _capture_set_material_slots(plan):
    obj = bpy.data.objects[plan["response"]["object"]]
    kept = {"object": obj.name, "names": material_slot_names(obj), "links": [ms.link for ms in obj.material_slots],
            "materials": set(m.name for m in bpy.data.materials), "faces": None}
    if obj.type == "MESH":
        faces = [0] * len(obj.data.polygons)
        obj.data.polygons.foreach_get("material_index", faces)
        kept["faces"] = faces
    return kept


def _restore_set_material_slots(kept, _plan):
    obj = bpy.data.objects.get(kept["object"])
    if obj is None:
        raise MifOpError("object '%s' is gone, so its slots cannot be put back" % kept["object"])
    while len(obj.material_slots) < len(kept["names"]):
        obj.data.materials.append(None)
    while len(obj.material_slots) > len(kept["names"]):
        obj.data.materials.pop()
    for i, (name, link) in enumerate(zip(kept["names"], kept["links"])):
        obj.material_slots[i].link = link
        obj.material_slots[i].material = bpy.data.materials.get(name) if name is not None else None
    if kept["faces"] is not None and len(kept["faces"]) == len(obj.data.polygons):
        obj.data.polygons.foreach_set("material_index", kept["faces"])
        obj.data.update()
    for mat in [m for m in bpy.data.materials if m.name not in kept["materials"] and m.users == 0]:
        bpy.data.materials.remove(mat)


def _plan_transform_selection(request):
    sink = {}
    resp = op_transform_selection(dict(request, dryRun=True), _plan=sink)
    key = sink.get("shapeKey")
    field = "shape_keys['%s'].data[%%d].co" % key if key else "vertices[%d].co"
    changes, unchanged = [], []
    for index, b, a in sink.get("verts") or []:
        row = {"kind": "vertex", "op": "modify", "target": sink["object"], "field": field % index,
               "before": rnd(b), "after": rnd(a), "_index": index, "_key": key}
        (unchanged if row["before"] == row["after"] else changes).append(row)
    commit = {k: v for k, v in request.items() if k != "dryRun"}
    return {"changes": changes, "unchanged": unchanged, "commit": commit,
            "context": ["shapeKey: %s" % key] if key else [], "response": resp}


def _read_transform_selection(change, _kept):
    obj = bpy.data.objects.get(change["target"])
    if obj is None or obj.type != "MESH":
        return None
    if change["_key"]:
        keys = obj.data.shape_keys
        block = keys.key_blocks.get(change["_key"]) if keys is not None else None
        data = block.data if block is not None else None
    else:
        data = obj.data.vertices
    return rnd(data[change["_index"]].co) if data is not None and change["_index"] < len(data) else None


def _capture_transform_selection(plan):
    mesh = bpy.data.objects[plan["response"]["object"]].data
    flat = [0.0] * (len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", flat)
    keys = {}
    if mesh.shape_keys is not None:
        for block in mesh.shape_keys.key_blocks:
            data = [0.0] * (len(block.data) * 3)
            block.data.foreach_get("co", data)
            keys[block.name] = data
    return {"object": plan["response"]["object"], "co": flat, "keys": keys}


def _restore_transform_selection(kept, _plan):
    obj = bpy.data.objects.get(kept["object"])
    if obj is None or obj.type != "MESH" or len(obj.data.vertices) * 3 != len(kept["co"]):
        raise MifOpError("'%s' is gone or its vertex count changed, so its vertices cannot be put back" % kept["object"])
    mesh = obj.data
    mesh.vertices.foreach_set("co", kept["co"])
    for name, data in kept["keys"].items():
        block = mesh.shape_keys.key_blocks.get(name) if mesh.shape_keys is not None else None
        if block is not None and len(block.data) * 3 == len(data):
            block.data.foreach_set("co", data)
    mesh.update()
    if mesh.shape_keys is not None:
        mesh.shape_keys.update_tag()
    bpy.context.view_layer.update()


PREVIEW = {
    "set_material_slots": {"plan": _plan_set_material_slots, "read": _read_set_material_slots,
                           "capture": _capture_set_material_slots, "restore": _restore_set_material_slots},
    "transform_selection": {"plan": _plan_transform_selection, "read": _read_transform_selection,
                            "capture": _capture_transform_selection, "restore": _restore_transform_selection},
}
# THE OPS THAT PLAN ON A SCRATCH COPY (ops_preview.mesh_spec, 2026-10-09): each computes its result in the pass that writes
# it - a bevel, a skirt, the seams, a push - or only through a Blender operator (the unwrap, the decimate), so the plan runs
# the op itself on a copy of the object and reads the copy.
from . import ops_preview as _preview   # noqa: E402 - after the ops it names

PREVIEW.update({
    "bevel_edges": _preview.mesh_spec(op_bevel_edges),
    "extrude_skirt": _preview.mesh_spec(op_extrude_skirt),
    "uv_unwrap": _preview.mesh_spec(op_uv_unwrap),
    "decimate_mesh": _preview.mesh_spec(op_decimate_mesh),
    "push_out_of": _preview.mesh_spec(op_push_out_of, keys=("object",)),
    "mark_seams": _preview.mesh_spec(op_mark_seams),
})

# ------------------------------------------------------------------------------------------------- knife_project
# KNIFE PROJECT WITHOUT A VIEW (second 1.3 audit, 2026-10-26). mesh.knife_project projects through the 3D View's region,
# and in a background session the borrowed region has no size: measured on 3.6.23, 4.2.17 and 5.2.1 it answers FINISHED
# and cuts nothing (1 face before, 1 after). So the cut is made here: the cutter's outline is swept along the direction
# into a closed prism, the target is split by an EXACT boolean into the part outside it and the part inside, and the two
# are welded back along the cut - the target's faces divided along the projected outline, through the whole mesh
# (cut_through), nothing removed.
_KP_KEYS = {"object", "name", "cutter", "direction", "cutThrough", "cleanWeld", "weldDistance"}

# FRONT FACES ONLY AND A CLEAN WELD (1.3 wave 3, 2026-10-26). cut_through false is Blender's own knife_project option:
# only the faces seen first along the direction are cut. Here a face is "seen first" when it is not edge-on to the
# direction and a ray from its center back toward the cutter meets nothing else of the mesh - decided per face by its
# center, so a face half hidden is cut whole or not at all. And the seam: the DIFFERENCE and INTERSECT halves of a
# curved or solid cut meet with a few outline vertices that do not coincide (a vertex of one half lying on an edge of
# the other - a T-junction - or two a hair apart); the clean weld merges the near ones and splits the edge under each
# T-junction vertex, only on open edges, so the mesh's own closed surface is never touched.


def _kp_front_faces(obj, direction):
    """{polygon index} of the faces seen first along direction (world), and how many faces were looked at."""
    from mathutils.bvhtree import BVHTree                         # noqa: PLC0415
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    bm.normal_update()
    bm.faces.ensure_lookup_table()
    bvh = BVHTree.FromBMesh(bm)
    d = Vector(direction).normalized()
    size = max((Vector(a) - Vector(b)).length for a in obj.bound_box for b in obj.bound_box) * \
        max(obj.matrix_world.to_scale())
    eps = max(1e-6, 1e-5 * size)
    front = set()
    for f in bm.faces:
        if abs(f.normal.dot(d)) < 1e-4:                                  # edge-on: not seen
            continue
        c = f.calc_center_median()
        hit = bvh.ray_cast(c - d * eps, -d)
        if hit[2] is None or hit[2] == f.index:
            front.add(f.index)
    total = len(bm.faces)
    bm.free()
    return front, total


def _kp_clean_weld(bm, dist):
    """Close the seam on OPEN edges only: merge open-edge vertices closer than dist, and split an open edge under each
    open-edge vertex that lies on it (within dist) and weld the two. Returns (vertices merged, T-junctions closed)."""
    from mathutils.geometry import intersect_point_line           # noqa: PLC0415
    merged = closed = 0
    for _ in range(8):
        open_verts = list({v for e in bm.edges if not e.is_manifold for v in e.verts})
        if not open_verts:
            break
        n0 = len(bm.verts)
        bmesh.ops.remove_doubles(bm, verts=open_verts, dist=dist)
        merged += n0 - len(bm.verts)
        open_edges = [e for e in bm.edges if not e.is_manifold]
        open_verts = list({v for e in open_edges for v in e.verts})
        if len(open_edges) * len(open_verts) > 4000000:                  # a mesh that is mostly border: stop here
            break
        best = {}
        for e in open_edges:
            a, b = e.verts[0].co, e.verts[1].co
            for v in open_verts:
                if v in e.verts:
                    continue
                pt, fac = intersect_point_line(v.co, a, b)
                if not 0.0 < fac < 1.0:
                    continue
                gap = (pt - v.co).length
                if gap < dist and (e not in best or gap < best[e][0]):
                    best[e] = (gap, v, fac)
        if not best:
            break
        targetmap = {}
        used = set()
        for e, (_gap, v, fac) in best.items():
            if v in used or not e.is_valid:
                continue
            _ne, nv = bmesh.utils.edge_split(e, e.verts[0], fac)
            targetmap[nv] = v
            used.add(v)
        if not targetmap:
            break
        bmesh.ops.weld_verts(bm, targetmap=targetmap)
        closed += len(targetmap)
    return merged, closed


def _kp_prism(cutter, direction, reach, tail):
    """A closed prism mesh in world space: the cutter's evaluated outline (its faces, or its closed edge loops filled)
    swept along direction from -reach to +reach."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = cutter.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.transform(cutter.matrix_world)
    finally:
        ev.to_mesh_clear()
    if not bm.faces:
        bmesh.ops.contextual_create(bm, geom=list(bm.edges))
    if not bm.faces:
        bm.free()
        raise MifOpError("the cutter '%s' has no face and no closed outline to fill - a knife cuts along a closed "
                         "shape. %s" % (cutter.name, tail))
    d = Vector(direction).normalized()
    bmesh.ops.translate(bm, verts=list(bm.verts), vec=-d * reach)
    res = bmesh.ops.extrude_face_region(bm, geom=list(bm.faces))
    moved = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=moved, vec=d * (2.0 * reach))
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    prism = bpy.data.meshes.new("_mif_kp_prism")
    bm.to_mesh(prism)
    bm.free()
    return prism


def op_knife_project(params):
    """Cut a cutter object's outline into a mesh along a direction, through the whole mesh - knife project without a view.

    params:
      object (str, required)   the mesh cut (OBJECT mode)
      cutter (str, required)   a mesh, curve or text object whose shape is the outline (its faces, or its closed loops)
      direction [x, y, z]      world direction the outline is projected along (default [0, 0, -1], straight down)
      cutThrough (bool)        default true: every face the outline crosses, front and back. false: only the faces seen
                               first along the direction (Blender's cut_through off), each decided by its center
      cleanWeld (bool)         default true: merge the outline vertices the two boolean halves leave a hair apart and
                               close their T-junctions, on open edges only (the weld reply counts both)
      weldDistance (m)         the clean weld's distance in the object's space, default 1e-4 of its bounding diagonal
    The target keeps every face: the ones the outline crosses are divided along it. Reply: faces and vertices before and
    after, and the vertices that lie on the cut.
    """
    reject_unknown(params, _KP_KEYS, "knife_project")
    tail = "NOTHING was changed."
    obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    cutter = get_object(take(params, "cutter", required=True, kind=str))
    if cutter == obj:
        raise MifOpError("the cutter is the object being cut. %s" % tail)
    if cutter.type not in ("MESH", "CURVE", "FONT"):
        raise MifOpError("the cutter '%s' is a %s; a mesh, curve or text outline cuts. %s" % (cutter.name, cutter.type, tail))
    if obj.mode != "OBJECT":
        raise MifOpError("'%s' is in %s mode; leave it first. %s" % (obj.name, obj.mode, tail))
    require_editable(obj, "cut")
    direction = finite_floats(list(params.get("direction") or [0.0, 0.0, -1.0])[:3], "direction")
    if Vector(direction).length < 1e-9:
        raise MifOpError("direction has no length. %s" % tail)
    cut_through = take_bool(params, "cutThrough", default=True) if params.get("cutThrough") is not None else True
    clean_weld = take_bool(params, "cleanWeld", default=True) if params.get("cleanWeld") is not None else True
    diag = max((Vector(a) - Vector(b)).length for a in obj.bound_box for b in obj.bound_box)
    weld_dist = max(1e-5, 1e-4 * diag)
    if params.get("weldDistance") is not None:
        if not clean_weld:
            raise MifOpError("weldDistance is the clean weld's; it goes with cleanWeld true. %s" % tail)
        weld_dist = finite_float(params["weldDistance"], "weldDistance")
        if not 0.0 < weld_dist <= 0.1 * max(diag, 1e-6):
            raise MifOpError("weldDistance is above 0 and at most a tenth of the object's bounding diagonal (%g), got "
                             "%g - a wider weld would merge the mesh's own detail. %s" % (0.1 * diag, weld_dist, tail))
    front, looked = (None, None) if cut_through else _kp_front_faces(obj, direction)
    if front is not None and not front:
        return {"object": obj.name, "cutter": cutter.name, "cutThrough": False, "frontFaces": 0, "facesLookedAt": looked,
                "cut": False, "note": "no face of the mesh is seen along that direction (every face is edge-on) - "
                                      "nothing was divided"}
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box] + [cutter.matrix_world @ Vector(c)
                                                                       for c in cutter.bound_box]
    reach = max((a - b).length for a in corners for b in corners) * 2.0 + 1.0
    prism_me = _kp_prism(cutter, direction, reach, tail)
    prism = bpy.data.objects.new("_mif_kp_prism", prism_me)
    bpy.context.scene.collection.objects.link(prism)
    made = [prism]
    f0, v0 = len(obj.data.polygons), len(obj.data.vertices)
    _b0 = bmesh.new()
    _b0.from_mesh(obj.data)
    open0 = len([e for e in _b0.edges if not e.is_manifold])     # the plane's own border counts here
    _b0.free()
    src_me = obj.data
    rest = None
    if front is not None:
        # ONLY THE FRONT FACES GO THROUGH THE BOOLEANS; the rest is put back beside them and welded along the border
        src_me = bpy.data.meshes.new("_mif_kp_front")
        fb = bmesh.new()
        fb.from_mesh(obj.data)
        fb.faces.ensure_lookup_table()
        bmesh.ops.delete(fb, geom=[f for f in fb.faces if f.index not in front], context="FACES")
        fb.to_mesh(src_me)
        fb.free()
        rest = bmesh.new()
        rest.from_mesh(obj.data)
        rest.faces.ensure_lookup_table()
        bmesh.ops.delete(rest, geom=[f for f in rest.faces if f.index in front], context="FACES")
        rest_me = bpy.data.meshes.new("_mif_kp_rest")
        rest.to_mesh(rest_me)
        rest.free()
        rest = rest_me
    weld_report = None
    try:
        parts = []
        for op in ("DIFFERENCE", "INTERSECT"):
            tmp = bpy.data.objects.new("_mif_kp_part", src_me)
            tmp.matrix_world = obj.matrix_world
            bpy.context.scene.collection.objects.link(tmp)
            made.append(tmp)
            mod = tmp.modifiers.new("kp", "BOOLEAN")
            mod.operation = op
            mod.object = prism
            if hasattr(mod, "solver"):
                mod.solver = "EXACT"
            dg = bpy.context.evaluated_depsgraph_get()
            ev = tmp.evaluated_get(dg)
            parts.append(bpy.data.meshes.new_from_object(ev))
        bm = bmesh.new()
        for pm in parts + ([rest] if rest is not None else []):
            bm.from_mesh(pm)
            bpy.data.meshes.remove(pm)
        rest = None
        weld = bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-5)
        # THE PRISM'S WALLS ARE NOT THE TARGET'S: inside a solid the INTERSECT part carries the prism's sides (measured: a
        # cube cut through gained area and lost its volume), so every face off the original surface is dropped
        from mathutils.bvhtree import BVHTree                     # noqa: PLC0415
        src = bmesh.new()
        src.from_mesh(obj.data)
        src_bvh = BVHTree.FromBMesh(src)
        src.free()
        off = [f for f in bm.faces if (src_bvh.find_nearest(f.calc_center_median())[3] or 0.0) > 1e-4]
        if off:
            bmesh.ops.delete(bm, geom=off, context="FACES")
            loose = [v for v in bm.verts if not v.link_faces]
            if loose:
                bmesh.ops.delete(bm, geom=loose, context="VERTS")
        raw_open = max(0, len([e for e in bm.edges if not e.is_manifold]) - open0)
        if clean_weld:
            merged, closed = _kp_clean_weld(bm, weld_dist)
            weld_report = {"distance": weld_dist, "verticesMerged": merged, "tJunctionsClosed": closed,
                           "openEdgesBefore": raw_open}
        non_manifold = max(0, len([e for e in bm.edges if not e.is_manifold]) - open0)   # new open or bad edges
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()
    finally:
        remove_objects(made)
        bpy.data.meshes.remove(prism_me)
        if src_me is not obj.data and src_me.name in bpy.data.meshes and src_me.users == 0:
            bpy.data.meshes.remove(src_me)
        if rest is not None:
            bpy.data.meshes.remove(rest)
    f1, v1 = len(obj.data.polygons), len(obj.data.vertices)
    del weld
    scope = {"cutThrough": cut_through}
    if front is not None:
        scope.update(frontFaces=len(front), facesLookedAt=looked)
    if f1 <= f0:
        return dict(scope, object=obj.name, cutter=cutter.name, facesBefore=f0, facesAfter=f1, verticesBefore=v0,
                    verticesAfter=v1, cut=False, weld=weld_report,
                    note="the outline did not cross the %s along that direction - nothing was divided"
                         % ("mesh" if cut_through else "faces seen first"))
    return dict(scope, object=obj.name, cutter=cutter.name, direction=rnd(direction), facesBefore=f0, facesAfter=f1,
                verticesBefore=v0, verticesAfter=v1, cut=True, nonManifoldEdges=non_manifold, weld=weld_report,
                note=("%d open edge(s) are left on the outline after the %s (nonManifoldEdges) - a wider weldDistance, "
                      "merge_by_distance or a fill closes them" % (non_manifold, "clean weld" if clean_weld
                                                                    else "weld at 1e-5"))
                if non_manifold else None)


OPS = {
    "knife_project": op_knife_project,
    "export_vertex_positions": op_export_vertex_positions,
    "delete_mesh_elements": op_delete_mesh_elements,
    "push_out_of": op_push_out_of,
    "fold_boundary_under": op_fold_boundary_under,
    "import_mesh": op_import_mesh,
    "decimate_mesh": op_decimate_mesh,
    "uv_unwrap": op_uv_unwrap,
    "create_collision_hull": op_create_collision_hull,
    "rename_object": op_rename_object,
    "set_vertex_color": op_set_vertex_color,
    "mesh_stats": op_mesh_stats,
    "mesh_quality": op_mesh_quality,
    "recipe_game_ready": op_recipe_game_ready,
    "export_mesh": op_export_mesh,
    "select_edges": op_select_edges,
    "transform_selection": op_transform_selection,
    "bevel_edges": op_bevel_edges,
    "extrude_skirt": op_extrude_skirt,
    "set_material_slots": op_set_material_slots,
    "apply_transform": op_apply_transform,
    "set_origin": op_set_origin,
    "clean_mesh": op_clean_mesh,
    "uv_info": op_uv_info,
    "uv_pack": op_uv_pack,
    "set_skin_vertices": op_set_skin_vertices,
    "remove_mesh_attributes": op_remove_mesh_attributes,
    "mark_seams": op_mark_seams,
    "set_uv_layer": op_set_uv_layer,
}
