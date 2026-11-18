"""MifBlender ops: the hand-off to Unreal beyond meshes - animation clips one FBX per action, a material's textures with
engine names, an animated camera, and hair as an Alembic groom (1.3 gap audit, 2026-10-19).

THE GAP. mif_send_to_unreal handled static and skinned meshes and nothing else: export_mesh's bakeAnimAllActions wrote every
clip into one file (a re-export of one clip re-imported all of them, named Armature|Action), a mesh reached Unreal with
material SLOTS and no textures, export_mesh refuses a camera by design, and hair had no route at all. Every FBX here is
written with export_mesh's axes and scale (ops_mesh.FBX_EXPORT_ARGS: Z up, -Y forward, 1 Blender unit = 100 Unreal units),
so a clip lands on the skeleton a skinned send made.
"""
from __future__ import annotations

import array
import json
import os
import re
import struct
import time
import zlib
import uuid

import bpy
from mathutils import Vector

from .ops_common import (MifOpError, check_output_path, check_writable_dir, file_before, get_object, reject_unknown,
                         remove_objects, replaced_fields, select_only, selection_restore, selection_snapshot, take, take_bool,
                         take_float, take_int, finite_floats, finite_int)

_BONE_PATH = re.compile(r'^pose\.bones\["((?:[^"\\]|\\.)*)"\]')


def _out_dir(params, tail):
    raw = take(params, "folder", "directory", required=True, kind=str)
    folder = os.path.abspath(bpy.path.abspath(raw))
    try:
        check_writable_dir(folder)
    except (OSError, ValueError) as exc:   # ValueError: a NUL in the path
        raise MifOpError("the folder '%s' cannot be written (%s). %s" % (folder, exc, tail))
    return folder


def _safe(name):
    return re.sub(r'[\\/:*?"<>|\s.]+', "_", str(name)).strip("_") or "unnamed"


def _fbx_args():
    from .ops_mesh import FBX_EXPORT_ARGS                       # noqa: PLC0415 - the one set of axes and scale
    return dict(FBX_EXPORT_ARGS)


# --------------------------------------------------------------------------------------------- export_actions

_EA_KEYS = ("armature", "actions", "folder", "directory", "pattern", "deformOnly", "overwrite", "simplify",
            "includeMesh")


def _action_bones(act):
    from .ops_anim import _action_curves                        # noqa: PLC0415 - slotted actions on 4.4+
    bones = set()
    other = 0
    for fc in _action_curves(act):
        m = _BONE_PATH.match(fc.data_path)
        if m:
            bones.add(m.group(1).replace('\\"', '"'))
        else:
            other += 1
    return bones, other


def _assign(arm, act):
    ad = arm.animation_data or arm.animation_data_create()
    ad.action = act
    if "action_slot" in ad.bl_rna.properties and act is not None and ad.action_slot is None:
        slot = next((s for s in act.slots if s.target_id_type == "OBJECT"), None)
        if slot is not None:
            ad.action_slot = slot


def op_export_actions(params):
    """Export each chosen action of an armature to its OWN FBX - the clips half of a character hand-off.

    params:
      armature (str)      the armature object
      actions (list)      the actions (default every action whose curves animate this armature's bones)
      folder (str)        where the files go
      pattern (str)       file name: {rig} and {action} are replaced (default A_{rig}_{action})
      deformOnly (bool)   only deform bones (default FALSE since 2026-11-15, as export_mesh writes the skeleton: true
                          dropped the root bone the mesh FBX carries, so the clip skeleton stopped matching the mesh;
                          true still works and the reply names the bones it left out)
      simplify (float)    the exporter's key simplify factor (default 0: every frame kept)
      includeMesh (bool)  write the skinned meshes too (default false: the armature and its keys)
      overwrite (bool)    default true; false refuses an existing file
    Each clip's length is the action's frame range - its MANUAL range when set (set_action frameRange), else the range of
    its keys; the scene's range is set to it for the export and put back. An action that keys bones the armature does
    not have is refused, naming them. Reply per file: frames, fps, bones keyed, bytes.
    """
    reject_unknown(params, _EA_KEYS, "export_actions", notes={"file": "one file per action goes to folder, named by pattern"})
    tail = "NOTHING was written."
    arm = get_object(take(params, "armature", required=True, kind=str))
    if arm.type != "ARMATURE":
        raise MifOpError("'%s' is a %s, not an armature. %s" % (arm.name, arm.type, tail))
    folder = _out_dir(params, tail)
    names = params.get("actions")
    bone_names = {b.name for b in arm.data.bones}
    if names is None:
        acts = [a for a in bpy.data.actions if _action_bones(a)[0] & bone_names]
        if not acts:
            raise MifOpError("no action in this file keys a bone of '%s'. %s" % (arm.name, tail))
    else:
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            raise MifOpError("actions is a non-empty list of action names. %s" % tail)
        absent = [n for n in names if bpy.data.actions.get(n) is None]
        if absent:
            raise MifOpError("no action(s) %s. %s" % (absent, tail))
        acts = [bpy.data.actions[n] for n in dict.fromkeys(names)]
    pattern = take(params, "pattern", default="A_{rig}_{action}", kind=str)
    if "{action}" not in pattern:
        raise MifOpError("pattern must contain {action}, or every clip would write one file. %s" % tail)
    overwrite = take_bool(params, "overwrite", default=True)
    simplify = take_float(params, "simplify", default=0.0)
    if not 0.0 <= simplify <= 100.0:
        raise MifOpError("simplify is 0-100, got %g. %s" % (simplify, tail))
    plan = []
    for act in acts:
        bones, _other = _action_bones(act)
        missing = sorted(bones - bone_names)
        if missing:
            raise MifOpError("action '%s' keys bone(s) '%s' does not have: %s - a retarget or a rename comes first. %s"
                             % (act.name, arm.name, ", ".join(missing[:20]), tail))
        if not bones:
            raise MifOpError("action '%s' keys no bone. %s" % (act.name, tail))
        path = os.path.join(folder, _safe(pattern.replace("{rig}", arm.name).replace("{action}", act.name)) + ".fbx")
        check_output_path(path, path, "written")
        before = file_before(path)
        if before is not None and not overwrite:
            raise MifOpError("'%s' exists and overwrite is false. %s" % (path, tail))
        plan.append((act, bones, path, before))
    dup = {p for _a, _b, p, _f in plan if sum(1 for x in plan if x[2] == p) > 1}
    if dup:
        raise MifOpError("pattern gives two actions one file name: %s. %s" % (sorted(dup), tail))
    meshes = []
    if take_bool(params, "includeMesh", default=False):
        meshes = [o for o in bpy.data.objects if o.type == "MESH"
                  and any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers)]
    args = _fbx_args()
    args.update({"object_types": {"ARMATURE", "MESH"} if meshes else {"ARMATURE"}, "bake_anim": True,
                 "bake_anim_use_all_actions": False, "bake_anim_use_nla_strips": False,
                 "bake_anim_simplify_factor": simplify, "bake_anim_force_startend_keying": True,
                 "use_armature_deform_only": take_bool(params, "deformOnly", default=False), "add_leaf_bones": False})
    # THE BONES deformOnly LEAVES OUT, named: export_mesh writes every bone by default (FOUND 2026-11-11: the clips lost
    # the root the mesh FBX carries and no longer matched its skeleton on import)
    dropped = sorted(b.name for b in arm.data.bones if not b.use_deform) if args["use_armature_deform_only"] else []
    scene = bpy.context.scene
    ad = arm.animation_data
    saved = (ad.action if ad else None, getattr(ad, "action_slot", None) if ad else None,
             scene.frame_start, scene.frame_end, scene.frame_current)
    nla_mute = None
    if ad is not None and ad.use_nla and len(ad.nla_tracks):
        nla_mute = ad.use_nla
    snap = selection_snapshot()
    rows = []
    # THE TAKE IS NAMED AFTER THE SCENE (FOUND 2026-11-16): with one action and no NLA export, Blender's FBX exporter
    # names its single AnimStack scene.name (io_scene_fbx export_fbx_bin.fbx_animations_do: "get_blenderID_name(ref_id)
    # if ref_id else scene.name"), so every clip reached the engine as "Scene". The scene takes the action's name for
    # each file and gets its own back; the take is read back from the file.
    scene_name = scene.name
    try:
        if nla_mute is not None:
            ad.use_nla = False             # an NLA track would layer under the clip
        select_only([arm] + meshes)
        fps = scene.render.fps / scene.render.fps_base
        for act, bones, path, before in plan:
            _assign(arm, act)
            fr = act.frame_range
            start, end = int(round(fr[0])), int(round(fr[1]))
            if end <= start:
                end = start + 1
            scene.frame_start, scene.frame_end = start, end
            scene.name = act.name
            bpy.ops.export_scene.fbx(filepath=path, use_selection=True, **args)
            if not os.path.isfile(path) or os.path.getsize(path) <= 0:
                raise MifOpError("the FBX exporter reported done and wrote nothing at %s." % path)
            takes = [t["name"] for t in _read_fbx(path)["animations"]]
            row = {"action": act.name, "file": path, "bytes": os.path.getsize(path), "frameStart": start,
                   "frameEnd": end, "frames": end - start + 1, "fps": round(fps, 6),
                   "seconds": round((end - start) / fps, 6), "bonesKeyed": len(bones),
                   "manualFrameRange": bool(getattr(act, "use_frame_range", False)), "takes": takes}
            if takes != [act.name]:
                # another scene already holds the action's name, so Blender suffixed this one
                row["takeNameWarning"] = ("the file's take is %s, not '%s' - another scene in this file is named '%s', "
                                          "so the take could not take the action's name; rename that scene."
                                          % (takes, act.name, act.name))
            row.update(replaced_fields(before))
            rows.append(row)
    finally:
        scene.name = scene_name
        a = arm.animation_data
        if a is not None:
            a.action = saved[0]
            if saved[1] is not None and "action_slot" in a.bl_rna.properties:
                try:
                    a.action_slot = saved[1]
                except (TypeError, RuntimeError):
                    pass
            if nla_mute is not None:
                a.use_nla = nla_mute
        scene.frame_start, scene.frame_end = saved[2], saved[3]
        scene.frame_set(saved[4])
        selection_restore(snap)
    return {"armature": arm.name, "folder": folder, "files": rows, "exported": len(rows),
            "deformOnly": args["use_armature_deform_only"], "meshesIncluded": [m.name for m in meshes],
            "bonesWritten": len(arm.data.bones) - len(dropped),
            "nonDeformBonesDropped": dropped or None,
            "skeletonMismatchWarning": (
                "deformOnly true left out %d non-deform bone(s) (%s) that export_mesh writes by default, so these clips' "
                "skeleton will not match a mesh exported with the default - pass deformOnly false, or export the mesh "
                "with armatureDeformOnly true as well." % (len(dropped), ", ".join(dropped[:10])))
            if dropped else None,
            "axis": {"up": "Z", "front": "-Y", "unit": "cm"},
            "note": "each file holds one take, named after its action (read back into takes); Unreal imports it onto an "
                    "existing Skeleton (import_asset skeleton)"}


# --------------------------------------------------------------------------------------------- export_material_textures

# A map's URP Lit property and the keyword that turns it on (Universal Render Pipeline/Lit).
_URP_MAPS = {"BC": ("_BaseMap", None), "N": ("_BumpMap", "_NORMALMAP"), "MS": ("_MetallicGlossMap", "_METALLICSPECGLOSSMAP"),
             "E": ("_EmissionMap", "_EMISSION"), "R": (None, None), "M": (None, None)}
_EMT_KEYS = ("objects", "object", "materials", "folder", "directory", "pattern", "normalConvention", "packOrm",
             "format", "overwrite", "profile")
_INPUTS = {   # map -> the Principled input names that feed it, newest first
    "BC": ("Base Color",), "N": ("Normal",), "R": ("Roughness",), "M": ("Metallic",),
    "E": ("Emission Color", "Emission"), "A": ("Alpha",),
}


def _principled(mat):
    if not mat.use_nodes or mat.node_tree is None:
        return None
    nt = mat.node_tree
    out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None) or next(
        (n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if out is not None and out.inputs["Surface"].is_linked:
        n = out.inputs["Surface"].links[0].from_node
        if n.type == "BSDF_PRINCIPLED":
            return n
    return next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)


def _unity_meta(path, texture_type, srgb):
    """Write the import settings beside a texture so Unity reads it right with no script of ours in the project: a partial
    .meta is honored and filled in on import (measured on 6000.0.31f1 and 6000.5.0f1), where the defaults for any PNG are
    Default and sRGB. An existing .meta is kept: its GUID is what the project's references point at. The new GUID comes
    from the file's name, so the same export gives the same one."""
    meta = path + ".meta"
    if os.path.exists(meta):
        return "kept"
    guid = uuid.uuid5(uuid.NAMESPACE_URL, os.path.basename(path)).hex
    with open(meta, "w", newline="\n", encoding="utf-8") as fh:
        fh.write("fileFormatVersion: 2\nguid: %s\nTextureImporter:\n  serializedVersion: 13\n  mipmaps:\n"
                 "    sRGBTexture: %d\n  textureType: %d\n" % (guid, 1 if srgb else 0, texture_type))
    return "written"


def _socket(node, names):
    return next((node.inputs[n] for n in names if n in node.inputs), None)


def _trace(sock, depth=0):
    """What feeds an input: ("image", image, channel) | ("value", v) | ("procedural", node type) | ("normalmap", ...)."""
    if sock is None:
        return ("none",)
    if not sock.is_linked:
        v = getattr(sock, "default_value", None)
        try:
            v = tuple(float(x) for x in v)
        except TypeError:
            v = float(v) if v is not None else None
        return ("value", v)
    link = sock.links[0]
    n, out = link.from_node, link.from_socket
    if depth > 8:
        return ("procedural", n.type)
    if n.type == "TEX_IMAGE":
        if n.image is None:
            return ("procedural", "TEX_IMAGE with no image")
        return ("image", n.image, "A" if out.name == "Alpha" else "RGB")
    if n.type in ("SEPARATE_COLOR", "SEPRGB", "SEPARATE_RGB"):
        src = _trace(n.inputs[0], depth + 1)
        if src[0] == "image" and src[2] == "RGB":
            ch = {"Red": "R", "Green": "G", "Blue": "B", "R": "R", "G": "G", "B": "B"}.get(out.name)
            if ch:
                return ("image", src[1], ch)
        return ("procedural", n.type)
    if n.type == "NORMAL_MAP":
        src = _trace(n.inputs["Color"], depth + 1)
        return ("normalmap", src, n.space, float(n.inputs["Strength"].default_value))
    if n.type == "REROUTE":
        return _trace(n.inputs[0], depth + 1)
    return ("procedural", n.type)


def _surface_shader(mat):
    """The node wired to the active output's Surface, or None."""
    nt = mat.node_tree if mat.use_nodes else None
    if nt is None:
        return None
    out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None) or next(
        (n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if out is None or not out.inputs["Surface"].is_linked:
        return None
    return out.inputs["Surface"].links[0].from_node


_SIMPLE_SHADERS = {"BSDF_GLASS": ("Color", "Roughness"), "BSDF_DIFFUSE": ("Color", "Roughness"), "BSDF_GLOSSY": ("Color", "Roughness"),
                   "BSDF_ANISOTROPIC": ("Color", "Roughness"), "EMISSION": ("Color", None), "BSDF_TRANSPARENT": ("Color", None)}


def _simple_shader_row(mat, row):
    """A material with no Principled BSDF, as the same constants: its viewport values with no node tree, or a lone Glass, Emission,
    Diffuse, Glossy or Transparent shader. Anything else (a Mix of shaders with no Principled in it) is one row of notCarried."""
    if not mat.use_nodes or mat.node_tree is None:
        row["constants"].update(baseColor=[float(v) for v in mat.diffuse_color[:3]], roughness=float(mat.roughness),
                                metallic=float(mat.metallic))
        if mat.diffuse_color[3] < 1.0:
            row["constants"]["opacity"] = float(mat.diffuse_color[3])
        return
    n = _surface_shader(mat)
    kinds = _SIMPLE_SHADERS
    if n is None or n.type not in kinds:
        row["notCarried"].append({"input": "*", "why": "no Principled BSDF in the material's tree (its surface is %s)"
                                  % (n.type if n is not None else "not connected")})
        return
    col_name, rough_name = kinds[n.type]
    col = _trace(n.inputs.get(col_name))
    rough = _trace(n.inputs.get(rough_name)) if rough_name else ("none",)
    for label, v in (("Color", col), ("Roughness", rough)):
        if v[0] not in ("value", "none"):
            row["notCarried"].append({"input": label, "why": "driven by a node chain on a %s shader (bake it)" % n.type})
    rgb = list(col[1])[:3] if col[0] == "value" else [1.0, 1.0, 1.0]
    if rough[0] == "value":
        row["constants"]["roughness"] = rough[1]
    if n.type == "EMISSION":
        strength = _trace(n.inputs.get("Strength"))
        k = strength[1] if strength[0] == "value" else 1.0
        row["constants"].update(baseColor=[0.0, 0.0, 0.0], emissive=[c * k for c in rgb])
    elif n.type == "BSDF_TRANSPARENT":
        row["constants"].update(baseColor=rgb, opacity=0.0)
    else:
        row["constants"]["baseColor"] = rgb
        if n.type == "BSDF_GLASS":
            ior = _trace(n.inputs.get("IOR"))
            row["constants"].update(transmission=1.0, ior=ior[1] if ior[0] == "value" else 1.5)
        elif n.type in ("BSDF_GLOSSY", "BSDF_ANISOTROPIC"):
            row["constants"]["metallic"] = 1.0


def _blend_mode(mat, row):
    """Unreal's blend mode for the row: a constant alpha or any transmission draws through, Translucent; an alpha map cuts, Masked,
    unless the material is set to blend (blend_method, or surface_render_method from 4.2)."""
    if any(m["map"] == "A" for m in row["maps"]):
        blended = getattr(mat, "surface_render_method", None) == "BLENDED" or getattr(mat, "blend_method", None) == "BLEND"
        return "Translucent" if blended else "Masked"
    c = row["constants"]
    return "Translucent" if c.get("opacity", 1.0) < 1.0 or c.get("transmission", 0.0) > 0.0 else "Opaque"


def op_export_material_textures(params):
    """Save every image a material's Principled inputs read, with engine names and conventions - the textures half of a
    send to Unreal (a mesh used to arrive with material slots and no textures).

    params:
      objects / object (list)   the objects whose materials are exported, or
      materials (list)          materials by name
      folder (str)              where the textures go
      pattern (str)             {material} and {suffix} replaced (default T_{material}_{suffix})
      normalConvention (str)    DIRECTX (default, Unreal's: green flipped from Blender's OpenGL) | OPENGL
      packOrm (bool)            pack occlusion (1), roughness, metallic into one _ORM map (default true); false writes _R
                                and _M separately
      format (str)              PNG (default) | TARGA
      overwrite (bool)          default true
      profile (str)             unreal (what no profile means) or unity. Under unity, for URP Lit: normalConvention
                                defaults to OPENGL, and in place of ORM one _MS map holds metallic in R, G and B and
                                smoothness (1 - roughness) in A, since a dielectric's ORM read as a mirror there (measured).
                                packOrm is refused. Each map names its URP property, and a .meta beside it sets the
                                texture type and sRGB, an existing .meta being kept. Occlusion is not carried: the
                                Principled BSDF has no input for it.
    Maps: BC (base color, with the Alpha input's image alpha when it reads one), N, ORM (or R, M), E (emission color).
    A channel read straight from an image (or an image's R/G/B through Separate Color, or a Normal Map node's image) is
    carried; anything else - a ramp, a math node, a procedural texture - is listed in notCarried (bl_bake_texture or
    bl_bake_texture_set bakes it). An unlinked input is reported as a constant for the material instance: baseColor, roughness,
    metallic, specular (when not the default 0.5), emissive (color x strength, linear, over 1 where the glow is), opacity,
    transmission and ior. A separate alpha map is written as A (R, G and B hold it). blendMode says how the row draws in Unreal.
    With no Principled BSDF a lone Glass, Emission, Diffuse, Glossy or Transparent shader, or a material with no node tree, gives
    the same constants.
    """
    reject_unknown(params, _EMT_KEYS, "export_material_textures")
    tail = "NOTHING was written."
    from .ops_mesh import export_profile                        # noqa: PLC0415 - the one list of profiles
    profile = export_profile(params, tail)
    unity = profile == "unity"
    if unity and params.get("packOrm") is not None:
        raise MifOpError("packOrm is Unreal's layout; profile unity writes URP's _MS map (metallic in R, smoothness in A) "
                         "instead. Drop packOrm. %s" % tail)
    from .ops_image import op_channel_pack_image                # noqa: PLC0415 - the one packer
    mats = []
    if params.get("materials") is not None:
        raw = params["materials"]
        if not isinstance(raw, list) or not raw or not all(isinstance(n, str) for n in raw):
            raise MifOpError("materials is a list of material names. %s" % tail)
        for n in dict.fromkeys(raw):
            m = bpy.data.materials.get(n)
            if m is None:
                raise MifOpError("no material named '%s'. %s" % (n, tail))
            mats.append(m)
    else:
        raw = params.get("objects", params.get("object"))
        names = [raw] if isinstance(raw, str) else raw
        if not isinstance(names, list) or not names:
            raise MifOpError("give objects (whose materials are exported) or materials. %s" % tail)
        for n in names:
            o = get_object(n)
            for s in o.material_slots:
                if s.material is not None and s.material not in mats:
                    mats.append(s.material)
        if not mats:
            raise MifOpError("%s use no material. %s" % (names, tail))
    folder = _out_dir(params, tail)
    pattern = take(params, "pattern", default="T_{material}_{suffix}", kind=str)
    conv = str(take(params, "normalConvention", default="OPENGL" if unity else "DIRECTX", kind=str)).upper()
    if conv not in ("DIRECTX", "OPENGL"):
        raise MifOpError("normalConvention is DIRECTX or OPENGL, got %r. %s" % (conv, tail))
    fmt = str(take(params, "format", default="PNG", kind=str)).upper()
    if fmt not in ("PNG", "TARGA"):
        raise MifOpError("format is PNG or TARGA, got %r. %s" % (fmt, tail))
    ext = ".png" if fmt == "PNG" else ".tga"
    pack_orm = take_bool(params, "packOrm", default=True)
    overwrite = take_bool(params, "overwrite", default=True)
    made_images = []
    rows = []
    try:
        for mat in mats:
            row = {"material": mat.name, "maps": [], "constants": {}, "notCarried": []}
            p = _principled(mat)
            top = _surface_shader(mat)
            # a lone shader wired to the output is the material, whatever unconnected Principled the tree still holds
            if p is None or (top is not None and top.type in _SIMPLE_SHADERS):
                _simple_shader_row(mat, row)
                row["blendMode"] = _blend_mode(mat, row)
                rows.append(row)
                continue
            src = {k: _trace(_socket(p, names)) for k, names in _INPUTS.items()}

            def path_of(suffix, mat=mat):
                return os.path.join(folder, _safe(pattern.replace("{material}", mat.name).replace("{suffix}", suffix))
                                    + ext)

            def write(suffix, chans, srgb, extra=None, mat=mat, row=row):
                fp = path_of(suffix)
                req = {"channels": chans, "name": "MifTex_%s_%s" % (_safe(mat.name), suffix), "filepath": fp,
                       "format": fmt, "overwrite": overwrite, "fakeUser": False}
                got = op_channel_pack_image(req)
                made_images.append(got["image"])
                r = {"map": suffix, "file": got["file"], "bytes": got.get("bytes"), "size": got["size"],
                     "srgb": srgb, "channels": {c: {k: v for k, v in got["channels"][c].items() if k != "written"}
                                                for c in got["channels"]}}
                r.update(extra or {})
                if unity:
                    prop = _URP_MAPS[suffix]
                    normal = suffix == "N"
                    r["unity"] = {"textureType": "NormalMap" if normal else "Default", "sRGB": srgb,
                                  "property": prop[0], "keyword": prop[1],
                                  "meta": _unity_meta(got["file"], 1 if normal else 0, srgb)}
                    if normal and conv == "DIRECTX":
                        r["unity"]["flipGreenChannel"] = True
                row["maps"].append(r)

            for key, label in (("R", "roughness"), ("M", "metallic")):
                if src[key][0] == "value":
                    row["constants"][label] = src[key][1]
            # BC (+ alpha)
            bc = src["BC"]
            if bc[0] == "image" and bc[2] == "RGB":
                chans = {"R": {"image": bc[1].name, "channel": "R"}, "G": {"image": bc[1].name, "channel": "G"},
                         "B": {"image": bc[1].name, "channel": "B"}}
                al = src["A"]
                if al[0] == "image":
                    chans["A"] = {"image": al[1].name, "channel": "A" if al[2] == "A" else al[2]}
                write("BC", chans, True, {"source": bc[1].name})
            elif bc[0] == "value":
                row["constants"]["baseColor"] = list(bc[1])[:3]
            elif bc[0] != "none":
                row["notCarried"].append({"input": "Base Color", "why": "driven by %s" % (bc[1],)})
            # N
            nm = src["N"]
            if nm[0] == "normalmap":
                inner = nm[1]
                if inner[0] == "image" and inner[2] == "RGB" and nm[2] == "TANGENT":
                    im = inner[1].name
                    write("N", {"R": {"image": im, "channel": "R"},
                                "G": {"image": im, "channel": "G", "invert": conv == "DIRECTX"},
                                "B": {"image": im, "channel": "B"}}, False,
                          {"source": im, "convention": conv, "greenFlipped": conv == "DIRECTX",
                           "strength": nm[3]})
                    if abs(nm[3] - 1.0) > 1e-6:
                        row["notCarried"].append({"input": "Normal", "why": "Normal Map strength %g is not in the "
                                                  "texture - set it on the material in %s"
                                                  % (nm[3], "Unity (_BumpScale)" if unity else "Unreal")})
                else:
                    row["notCarried"].append({"input": "Normal", "why": "a %s-space normal map, or not an image" % nm[2]})
            elif nm[0] not in ("none", "value"):
                row["notCarried"].append({"input": "Normal", "why": "driven by %s (a bump or a node chain)" % (nm[1],)})
            # ORM / R / M
            rough, metal = src["R"], src["M"]
            if unity and (rough[0] == "image" or metal[0] == "image"):
                # URP LIT READS R AND A: metallic = map.r, smoothness = map.a * _Smoothness (LitInput.hlsl), so _Smoothness
                # goes to 1 or every value halves. Linear: imported sRGB the same map rendered 0.55 against 0.74.
                def ms_chan(s, dflt, invert):
                    if s[0] == "image":
                        return {"image": s[1].name, "channel": "R" if s[2] == "RGB" else s[2], "invert": invert}
                    v = float(s[1]) if s[0] == "value" and s[1] is not None else dflt
                    return 1.0 - v if invert else v
                m_ch = ms_chan(metal, 0.0, False)
                write("MS", {"R": m_ch, "G": m_ch, "B": m_ch, "A": ms_chan(rough, 0.5, True)}, False,
                      {"layout": "R, G, B metallic; A smoothness (1 - roughness)"})
                row["urp"] = {"_Smoothness": 1.0, "smoothnessSource": "Metallic Alpha"}
            elif unity:
                row["urp"] = {"_Metallic": float(metal[1]) if metal[0] == "value" and metal[1] is not None else 0.0,
                              "_Smoothness": round(1.0 - float(rough[1]), 6) if rough[0] == "value" and rough[1] is not None
                              else 0.5}
            elif pack_orm and (rough[0] == "image" or metal[0] == "image"):
                def chan(s, dflt):
                    if s[0] == "image":
                        return {"image": s[1].name, "channel": "R" if s[2] == "RGB" else s[2]}
                    return float(s[1]) if s[0] == "value" and s[1] is not None else dflt
                write("ORM", {"R": 1.0, "G": chan(rough, 0.5), "B": chan(metal, 0.0)}, False,
                      {"layout": "R occlusion (1), G roughness, B metallic"})
            else:
                for s, suffix in ((rough, "R"), (metal, "M")):
                    if s[0] == "image":
                        c = "R" if s[2] == "RGB" else s[2]
                        write(suffix, {"R": {"image": s[1].name, "channel": c}, "G": {"image": s[1].name, "channel": c},
                                       "B": {"image": s[1].name, "channel": c}}, False)
            for s, label in ((rough, "Roughness"), (metal, "Metallic")):
                if s[0] == "procedural":
                    row["notCarried"].append({"input": label, "why": "driven by %s" % s[1]})
            em = src["E"]
            strength = _trace(_socket(p, ("Emission Strength",)))
            if strength[0] not in ("value", "none"):
                row["notCarried"].append({"input": "Emission Strength", "why": "driven by a node chain (bake EMIT)"})
            k = strength[1] if strength[0] == "value" and strength[1] is not None else 1.0
            if em[0] == "image" and em[2] == "RGB":
                im = em[1].name
                write("E", {c: {"image": im, "channel": c} for c in "RGB"}, True, {"source": im})
                if abs(k - 1.0) > 1e-6:
                    row["notCarried"].append({"input": "Emission Strength", "why": "strength %g is not in the texture - multiply "
                                              "it in the material in Unreal" % k})
            elif em[0] == "value" and em[1] is not None and k > 0.0 and max(em[1][:3]) > 0.0:
                row["constants"]["emissive"] = [float(c) * k for c in em[1][:3]]
            elif em[0] == "procedural":
                row["notCarried"].append({"input": "Emission Color", "why": "driven by %s" % em[1]})
            # WHAT DRAWS THROUGH. The alpha value and the map are read whatever the base color is; transmission and IOR are constants
            # the send to Unreal turns into a translucent material with refraction (Blender 3.6 names the socket Transmission).
            al = src["A"]
            if al[0] == "value" and al[1] is not None and al[1] < 1.0:
                row["constants"]["opacity"] = al[1]
            elif al[0] == "image":
                c = "A" if al[2] == "A" else ("R" if al[2] == "RGB" else al[2])
                write("A", {x: {"image": al[1].name, "channel": c} for x in "RGB"}, False, {"source": al[1].name})
            elif al[0] == "procedural":
                row["notCarried"].append({"input": "Alpha", "why": "driven by %s" % al[1]})
            tr = _trace(_socket(p, ("Transmission Weight", "Transmission")))
            if tr[0] == "value" and tr[1] is not None and tr[1] > 0.0:
                ior = _trace(_socket(p, ("IOR",)))
                row["constants"].update(transmission=tr[1], ior=ior[1] if ior[0] == "value" else 1.5)
            elif tr[0] not in ("value", "none"):
                row["notCarried"].append({"input": "Transmission", "why": "driven by a node chain (bake it, or set a value)"})
            spec = _trace(_socket(p, ("Specular IOR Level", "Specular")))
            if spec[0] == "value" and spec[1] is not None and abs(spec[1] - 0.5) > 1e-6:
                row["constants"]["specular"] = spec[1]
            if top is None or top.name != p.name:
                row["notCarried"].append({"input": "*", "why": "the surface is a %s, not the Principled BSDF read here - what it mixes "
                                          "in is not carried" % (top.type if top is not None else "disconnected node")})
            row["blendMode"] = _blend_mode(mat, row)
            rows.append(row)
    finally:
        for n in made_images:
            im = bpy.data.images.get(n)
            if im is not None:
                bpy.data.images.remove(im)
    carried = sum(len(r["maps"]) for r in rows)
    out = {"folder": folder, "materials": rows, "texturesWritten": carried, "normalConvention": conv,
           "packOrm": pack_orm and not unity}
    if profile is not None:
        out["profile"] = profile
    if not carried:
        out["warning"] = "no texture was carried - every input is a constant or a node chain (see notCarried)"
    return out


# --------------------------------------------------------------------------------------------- export_camera

_EC_KEYS = ("camera", "file", "frameRange", "overwrite", "samples")


def op_export_camera(params):
    """Export an animated camera to FBX in Unreal's axes, with its lens animation, and read back what was written.

    params:
      camera (str)          a camera object
      file (str)            the .fbx path
      frameRange ([s, e])   default the scene's range
      samples (int)         lens values sampled into the reply (default 5)
      overwrite (bool)      default true
    The FBX exporter writes the camera's transform and focal length (and focus distance) per frame; the f-stop has no FBX
    channel, so focal length, focus distance and f-stop are also sampled into the reply (lensFrames) - the values an
    importer keys on a CineCamera (Unreal's import_sequence_fbx reads the file onto a camera binding).
    """
    reject_unknown(params, _EC_KEYS, "export_camera")
    tail = "NOTHING was written."
    cam = get_object(take(params, "camera", required=True, kind=str))
    if cam.type != "CAMERA":
        raise MifOpError("'%s' is a %s, not a camera. %s" % (cam.name, cam.type, tail))
    raw = take(params, "file", required=True, kind=str)
    path = os.path.abspath(bpy.path.abspath(raw))
    check_output_path(raw, path, "written")
    if not path.lower().endswith(".fbx"):
        raise MifOpError("file must be a .fbx path, got %r. %s" % (raw, tail))
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        raise MifOpError("the folder '%s' does not exist. %s" % (folder, tail))
    before = file_before(path)
    overwrite = take_bool(params, "overwrite", default=True)
    if before is not None and not overwrite:
        raise MifOpError("'%s' exists and overwrite is false. %s" % (path, tail))
    scene = bpy.context.scene
    fr = params.get("frameRange")
    if fr is None:
        start, end = scene.frame_start, scene.frame_end
    else:
        if (not isinstance(fr, list) or len(fr) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in fr)
                or fr[1] < fr[0]):
            raise MifOpError("frameRange is [start, end] whole frames, end >= start, got %r. %s" % (fr, tail))
        start, end = fr
    n = max(2, take_int(params, "samples", default=5))
    args = _fbx_args()
    args.update({"object_types": {"CAMERA"}, "bake_anim": True, "bake_anim_use_all_actions": False,
                 "bake_anim_use_nla_strips": False, "bake_anim_simplify_factor": 0.0,
                 "bake_anim_force_startend_keying": True})
    saved = (scene.frame_start, scene.frame_end, scene.frame_current)
    snap = selection_snapshot()
    frames = []
    try:
        steps = sorted({int(round(start + (end - start) * i / (n - 1))) for i in range(n)})
        for f in steps:
            scene.frame_set(f)
            d = cam.data
            frames.append({"frame": f, "focalLength": round(d.lens, 5), "sensorWidth": round(d.sensor_width, 5),
                           "focusDistance": round(d.dof.focus_distance, 5) if d.dof.use_dof else None,
                           "fStop": round(d.dof.aperture_fstop, 5) if d.dof.use_dof else None,
                           "location": [round(v, 5) for v in cam.matrix_world.translation],
                           "forward": [round(v, 6) for v in (cam.matrix_world.to_3x3().col[2] * -1.0).normalized()]})
        scene.frame_start, scene.frame_end = start, end
        select_only([cam])
        bpy.ops.export_scene.fbx(filepath=path, use_selection=True, **args)
    finally:
        scene.frame_start, scene.frame_end = saved[0], saved[1]
        scene.frame_set(saved[2])
        selection_restore(snap)
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        raise MifOpError("the FBX exporter reported done and wrote nothing at %s." % path)
    anim = cam.animation_data is not None and cam.animation_data.action is not None
    lens_anim = cam.data.animation_data is not None and cam.data.animation_data.action is not None
    out = {"camera": cam.name, "file": path, "bytes": os.path.getsize(path), "frameStart": start, "frameEnd": end,
           "frames": end - start + 1, "fps": round(scene.render.fps / scene.render.fps_base, 6),
           "transformAnimated": anim, "lensAnimated": lens_anim, "lensFrames": frames,
           "axis": {"up": "Z", "front": "-Y", "unit": "cm"},
           "fStopNote": "FBX has no f-stop channel - key CurrentAperture from lensFrames on the Unreal side"}
    out.update(replaced_fields(before))
    return out


# --------------------------------------------------------------------------------------------- export_groom

_EG_KEYS = ("objects", "object", "file", "frameRange", "scale", "overwrite")


def _groom_curves_arrays(data):
    """(positions Nx3, per-curve point counts, radii N) of a Curves datablock, by its own attributes."""
    import numpy as np                                                # noqa: PLC0415
    pos = np.empty(len(data.points) * 3, dtype=np.float64)
    data.attributes["position"].data.foreach_get("vector", pos)
    off = np.empty(len(data.curves) + 1, dtype=np.int64)
    data.curve_offset_data.foreach_get("value", off)
    radii = np.full(len(data.points), 0.005, dtype=np.float64)        # Blender's own default where a Curves object has none
    if data.attributes.get("radius") is not None:
        data.attributes["radius"].data.foreach_get("value", radii)
    return pos.reshape(-1, 3), np.diff(off), radii


def _groom_shape_error(name, pos, counts, scale):
    """Why Unreal's groom builder cannot take this hair, or None. These kill the EDITOR, in the engine's own voxelizer, which
    runs on every import before the asset exists (GroomBuilder.cpp 5.8 :3031, :3059): a group flat along an axis makes a grid
    with no cells on it and the next write is out of bounds (measured 2026-11-01 on 5.8.3, every point at Y 0: "Array index out
    of bounds: 30 into an array of size 1"). By the same code, read and not run, a curve of one point underflows its segment
    count and a zero-length segment divides by its zero steps."""
    import numpy as np                                                # noqa: PLC0415
    if int(counts.min()) < 2:
        return "'%s' has a curve of fewer than 2 points, which crashes Unreal's groom builder" % name
    ext = (pos.max(axis=0) - pos.min(axis=0)) * scale
    flat = [a for a, e in zip("XYZ", ext) if e < 1e-3]
    if flat:
        return ("'%s' is flat along %s (its box is %s cm), which crashes Unreal's groom builder: it voxelizes each group, and a "
                "flat one has no cells on that axis" % (name, "/".join(flat), [round(float(e), 4) for e in ext]))
    within = np.ones(len(pos) - 1, dtype=bool)                        # False at the step from one curve to the next
    within[np.cumsum(counts)[:-1] - 1] = False
    seg = np.linalg.norm(pos[1:] - pos[:-1], axis=1)[within] * scale
    if seg.size and float(seg.min()) <= 1e-6:
        return "'%s' has a zero-length curve segment, which crashes Unreal's groom builder" % name
    return None


def _groom_particle_curves(o, depsgraph):
    """A mesh's hair particles, children included, as world-space polylines: (positions Nx3, counts, radii N)."""
    import numpy as np                                                # noqa: PLC0415
    ev = o.evaluated_get(depsgraph)
    pts, counts, radii = [], [], []
    for ps in ev.particle_systems:
        st = ps.settings
        if st.type != "HAIR":
            continue
        steps = 2 ** st.display_step
        root, tip = st.root_radius * st.radius_scale, st.tip_radius * st.radius_scale
        for i in range(len(ps.particles) + len(ps.child_particles)):
            for k in range(steps + 1):
                pts.append(tuple(ps.co_hair(ev, particle_no=i, step=k)))
                radii.append(root + (tip - root) * k / steps)
            counts.append(steps + 1)
    return (np.array(pts, dtype=np.float64).reshape(-1, 3), np.array(counts, dtype=np.int64),
            np.array(radii, dtype=np.float64))


def _groom_temp_curve(name, pos, counts, radii):
    """A POLY curve object holding those polylines, for the Alembic writer: linear, no basis, no knots, on every build. Blender's
    default Catmull-Rom hair is written with a variable order and too few knots for it: Unreal's translator ensures
    (AlembicHairTranslator.cpp 5.8 :869, "Computed number of knots would exceed the number available"), an error in the log on an
    import that then goes on. A copy of the Curves object itself took Blender 4.2.17 down, so the arrays are written instead."""
    import numpy as np                                                # noqa: PLC0415
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    start = 0
    for c in counts.tolist():
        sp = cu.splines.new("POLY")
        sp.points.add(c - 1)
        co = np.ones((c, 4), dtype=np.float64)
        co[:, :3] = pos[start:start + c]
        sp.points.foreach_set("co", co.ravel())
        sp.points.foreach_set("radius", radii[start:start + c])
        start += c
    tmp = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(tmp)
    return tmp


def op_export_groom(params):
    """Export hair to an Alembic the Unreal groom importer reads: Curves objects (each one a groom GROUP) or a mesh's hair
    particle systems, positions and widths, in centimeters.

    params:
      objects / object      Curves objects (4.2+ hair), or meshes with hair particle systems; each object is one group
      file (str)            the .abc path
      frameRange ([s, e])   default [current, current] - a static groom
      scale (float)         default 100 (1 Blender unit = 100 Unreal units, as every send here)
      overwrite (bool)      default true
    MEASURED 2026-10-19 on 4.2.17, 4.4.0, 5.0.1 and 5.2.1: Blender's Alembic exporter writes a Curves object's positions
    and its radius as width, and DROPS every other curve attribute (groom_group_id, groom_root_uv, groom_guide, groom_width
    written and read back: none survived). So groups are carried as separate objects, guides are not carried (Unreal
    makes its own), and root UVs come from the importer. The file is Y-up; a scripted Unreal import needs the conversion
    named, rotation X 90 and scale Y -1, or the hair lies on its side (mif_send_to_unreal names it).
    Written through a temporary POLY curve, the scene untouched: Curves as they evaluate, particle hair as polylines read from
    the particle system, children included (the writer's own hair export also writes the mesh, and a file holding a mesh is
    one Unreal's groom importer refuses, as "Unknown extension"). A frame range of more than one frame writes Curves objects as
    they stand, and is refused for particle hair. A flat group, a curve of one point and a zero-length segment are refused: each
    crashes the editor in Unreal's groom builder.
    """
    reject_unknown(params, _EG_KEYS, "export_groom")
    tail = "NOTHING was written."
    raw = params.get("objects", params.get("object"))
    names = [raw] if isinstance(raw, str) else raw
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise MifOpError("objects is a Curves object name or a list of them. %s" % tail)
    objs = [get_object(n) for n in dict.fromkeys(names)]
    groups, sources = [], []
    import numpy as np                                                # noqa: PLC0415
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for o in objs:
        if o.type == "CURVES":
            pos, counts, radii = _groom_curves_arrays(o.evaluated_get(depsgraph).data)
            m = np.array([list(r) for r in o.matrix_world], dtype=np.float64)
            pos = pos @ m[:3, :3].T + m[:3, 3]
            kind = "curves"
        elif o.type == "MESH" and any(ps.settings.type == "HAIR" for ps in o.particle_systems):
            pos, counts, radii = _groom_particle_curves(o, depsgraph)
            kind = "particle hair"
        else:
            raise MifOpError("'%s' is a %s with no hair - give Curves objects or meshes with hair particles. %s"
                             % (o.name, o.type, tail))
        sources.append({"object": o, "pos": pos, "counts": counts, "radii": radii})
        groups.append({"object": o.name, "kind": kind, "curves": int(counts.size), "points": int(pos.shape[0])})
    empty = [g["object"] for g in groups if not g["curves"]]
    if empty:
        raise MifOpError("%s hold no curves - an empty groom. %s" % (empty, tail))
    fp = take(params, "file", required=True, kind=str)
    path = os.path.abspath(bpy.path.abspath(fp))
    check_output_path(fp, path, "written")
    if not path.lower().endswith(".abc"):
        raise MifOpError("file must be a .abc path, got %r. %s" % (fp, tail))
    before = file_before(path)
    overwrite = take_bool(params, "overwrite", default=True)
    if before is not None and not overwrite:
        raise MifOpError("'%s' exists and overwrite is false. %s" % (path, tail))
    scene = bpy.context.scene
    fr = params.get("frameRange") or [scene.frame_current, scene.frame_current]
    if not isinstance(fr, list) or len(fr) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in fr):
        raise MifOpError("frameRange is [start, end] whole frames, got %r. %s" % (fr, tail))
    scale = take_float(params, "scale", default=100.0)
    if not scale > 0:
        raise MifOpError("scale must be above 0, got %g. %s" % (scale, tail))
    animated = fr[0] != fr[1]
    if animated and any(src["object"].type != "CURVES" for src in sources):
        raise MifOpError("a frame range of more than one frame is for Curves objects: particle hair goes through a polyline "
                         "copy of one frame, so it would not move. %s" % tail)
    for src in sources:
        if src["counts"].size:
            why = _groom_shape_error(src["object"].name, src["pos"], src["counts"], scale)
            if why:
                raise MifOpError("%s. %s" % (why, tail))
    for src, g in zip(sources, groups):
        lo, hi = src["pos"].min(axis=0) * scale, src["pos"].max(axis=0) * scale
        # The box is the caller's scale times Blender's positions: a scale near the float limit overflows it to inf,
        # which JSON cannot carry and Unreal cannot size a groom by. Refused here, before the temporary curves exist.
        try:
            lo, hi = finite_floats(lo, "boxMinCM (positions x scale)"), finite_floats(hi, "boxMaxCM (positions x scale)")
        except MifOpError as exc:
            raise MifOpError("%s Lower scale. %s" % (exc, tail))
        g["boxMinCM"], g["boxMaxCM"] = [round(v, 4) for v in lo], [round(v, 4) for v in hi]
    snap = selection_snapshot()
    temps = []
    try:
        if animated:
            targets = objs
        else:
            for src in sources:
                name = "%s_groom" % src["object"].name
                temps.append(_groom_temp_curve(name, src["pos"], src["counts"], src["radii"]))
            targets = temps
        select_only(targets)
        kw = dict(filepath=path, selected=True, start=fr[0], end=fr[1], global_scale=scale, export_hair=False,
                  export_particles=False, uvs=False, normals=False, vcolors=False, face_sets=False)
        bpy.ops.wm.alembic_export(**kw)
    finally:
        selection_restore(snap)
        curves = [tmp.data for tmp in temps]
        remove_objects(temps)
        for data in curves:
            bpy.data.curves.remove(data)
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        raise MifOpError("the Alembic exporter reported done and wrote nothing at %s." % path)
    out = {"file": path, "bytes": os.path.getsize(path), "groups": groups, "scale": scale, "frameRange": fr,
           "curves": sum(g["curves"] for g in groups),
           "carried": ["positions", "width (from radius)"],
           "notCarried": ["groom_guide (Unreal generates guides)", "groom_root_uv (the importer computes it)",
                          "other curve attributes"],
           "unrealImport": "name the conversion in the groom import: rotation X 90, scale Y -1, for a Y-up Alembic"}
    out.update(replaced_fields(before))
    return out


# --------------------------------------------------------------------------------------------- heightmaps
# UNREAL'S LANDSCAPE STORES A HEIGHT AS uint16 with 32768 at the actor's own Z and one unit = scale.z / 128 cm
# (LANDSCAPE_ZSCALE 1/128 and the mid value 32768: UE 5.3 and 5.8 Runtime/Landscape/Public/LandscapeDataAccess.h:13, :27).
# A vertex (i, j) of the export sits at the actor's location + (i x scale.x, j x scale.y) cm. In Blender, at the FBX
# hand-off's convention (Blender (x, y, z) m = Unreal (100x, -100y, 100z)), that is origin + (i x cell, -j x cell).
_HM_FROM_KEYS = ("file", "width", "height", "cellSize", "cellSizeY", "heightScale", "midValue", "origin", "name",
                 "collection")
_HM_TO_KEYS = ("object", "file", "width", "height", "cellSize", "cellSizeY", "heightScale", "midValue", "origin",
               "missValue", "overwrite", "samples")


def _read_heights(path, width, height, tail):
    """[uint16] row-major from a .r16 / .raw (little-endian, no header; width and height required) or a 16-bit PNG."""
    import struct                                                     # noqa: PLC0415
    ext = os.path.splitext(path)[1].lower()
    if ext in (".r16", ".raw"):
        if not width or not height:
            raise MifOpError("a .r16 has no header: pass width and height. %s" % tail)
        data = open(path, "rb").read()
        if len(data) != width * height * 2:
            raise MifOpError("'%s' is %d bytes; %d x %d 16-bit samples are %d. Refused rather than sheared. %s"
                             % (path, len(data), width, height, width * height * 2, tail))
        return list(struct.unpack("<%dH" % (width * height), data)), width, height
    if ext != ".png":
        raise MifOpError("a heightmap is a .r16 / .raw or a 16-bit PNG, got '%s'. %s" % (ext, tail))
    img = bpy.data.images.load(path, check_existing=False)
    try:
        img.colorspace_settings.name = "Non-Color"
        w, h = int(img.size[0]), int(img.size[1])
        if (width and width != w) or (height and height != h):
            raise MifOpError("the PNG is %dx%d, not the %sx%s asked for. %s" % (w, h, width, height, tail))
        # A 16-BIT PNG LOADS AS A FLOAT BUFFER and an 8-bit one as bytes (measured: depth reads 32 for a 16-bit grey PNG,
        # so it says nothing about the file) - an 8-bit heightmap has 256 steps and is refused rather than stretched.
        if not img.is_float:
            raise MifOpError("'%s' is an 8-bit image, not a 16-bit greyscale heightmap. %s" % (path, tail))
        px = [0.0] * (w * h * img.channels)
        img.pixels.foreach_get(px)
        ch = img.channels
        # Blender stores rows bottom-up; the file's first row is its top row (y = 0 in Unreal's export)
        out = [0] * (w * h)
        for j in range(h):
            src = (h - 1 - j) * w
            for i in range(w):
                out[j * w + i] = int(round(px[(src + i) * ch] * 65535.0))
        return out, w, h
    finally:
        bpy.data.images.remove(img)


def _hm_scales(params, tail):
    cx = take_float(params, "cellSize", required=True)
    cy = take_float(params, "cellSizeY", default=cx)
    hs = take_float(params, "heightScale", required=True)
    mid = take_float(params, "midValue", default=32768.0)
    if not cx > 0 or not cy > 0 or not hs > 0:
        raise MifOpError("cellSize, cellSizeY and heightScale are above 0 (metres per cell, metres per height unit). %s"
                         % tail)
    org = params.get("origin", [0.0, 0.0, 0.0])
    if not isinstance(org, (list, tuple)) or len(org) != 3:
        raise MifOpError("origin is [x, y, z] metres, Blender world. %s" % tail)
    from .ops_common import finite_floats                            # noqa: PLC0415
    return cx, cy, hs, mid, Vector(finite_floats(list(org), "origin"))


def op_mesh_from_heightmap(params):
    """A grid mesh displaced by a 16-bit heightmap at a stated scale - an Unreal landscape as a Blender surface.

    WHY (second 1.3 audit, 2026-10-25): no op read a heightmap, and the scale (32768 = the landscape's Z, one unit =
    scale.z / 128 cm) was left to the caller. mif_send_landscape_to_blender passes Unreal's numbers in here.

    params:
      file (str)          a .r16 / .raw (little-endian uint16, row-major; width and height required) or a 16-bit PNG
      width, height       samples per row and rows
      cellSize (m)        the spacing along X (Unreal's scale.x / 100); cellSizeY along -Y (default cellSize)
      heightScale (m)     metres per height unit (Unreal's scale.z / 128 / 100)
      midValue            the sample value at origin's height (default 32768)
      origin [x, y, z]    Blender world metres of sample (0, 0) at midValue
      name, collection
    Vertex (i, j) sits at origin + (i x cellSize, -j x cellSizeY, (v - midValue) x heightScale): Unreal's +Y is
    Blender's -Y. Reply: the object, counts, the height range, the world box.
    """
    reject_unknown(params, _HM_FROM_KEYS, "mesh_from_heightmap")
    tail = "NOTHING was created."
    path = os.path.abspath(bpy.path.abspath(take(params, "file", required=True, kind=str)))
    if not os.path.isfile(path):
        raise MifOpError("no file '%s'. %s" % (path, tail))
    w = take_int(params, "width", default=0)
    h = take_int(params, "height", default=0)
    cx, cy, hs, mid, org = _hm_scales(params, tail)
    vals, w, h = _read_heights(path, w, h, tail)
    if w < 2 or h < 2:
        raise MifOpError("a heightmap needs at least 2 x 2 samples. %s" % tail)
    if w * h > 16_777_216:
        raise MifOpError("%d x %d samples is over 16.7 million vertices. %s" % (w, h, tail))
    name = take(params, "name", default="Terrain", kind=str)
    # RESOLVED BEFORE ANYTHING IS CREATED: a collection that does not exist refused after objects.new left the mesh
    # and the object behind in bpy.data, in no scene, under a refusal saying NOTHING was created.
    from .ops_create import _link_new, _resolve_collection             # noqa: PLC0415
    coll = _resolve_collection(params)
    me = bpy.data.meshes.new(name)
    verts = [(org.x + i * cx, org.y - j * cy, org.z + (vals[j * w + i] - mid) * hs) for j in range(h) for i in range(w)]
    faces = [(j * w + i, j * w + i + 1, (j + 1) * w + i + 1, (j + 1) * w + i) for j in range(h - 1) for i in range(w - 1)]
    me.from_pydata(verts, [], faces)
    me.update()
    # FACING UP: the winding above runs clockwise seen from +Z once Y is flipped, so the normals are turned
    if me.polygons and me.polygons[0].normal.z < 0:
        me.flip_normals()
    obj = bpy.data.objects.new(me.name, me)
    _link_new(obj, coll)
    zs = [v[2] for v in verts]
    return {"object": obj.name, "samples": [w, h], "vertices": len(verts), "faces": len(faces),
            "requestedName": name, "nameWasSuffixed": obj.name != name,
            "heightRange": {"min": round(min(zs), 6), "max": round(max(zs), 6)},
            "rawRange": {"min": min(vals), "max": max(vals)},
            "bboxWorld": {"min": [round(org.x, 6), round(org.y - (h - 1) * cy, 6), round(min(zs), 6)],
                          "max": [round(org.x + (w - 1) * cx, 6), round(org.y, 6), round(max(zs), 6)]},
            "mapping": "vertex (i, j) at origin + (i x %g, -j x %g, (v - %g) x %g) m" % (cx, cy, mid, hs)}


def op_heightmap_from_mesh(params):
    """Ray-cast a terrain mesh straight down on a grid and write the heights as a 16-bit heightmap - the reverse of
    mesh_from_heightmap, at the same mapping, for import_landscape_heightmap.

    params:
      object (str)        the terrain (its evaluated mesh: modifiers count)
      file (str)          the .r16 / .raw to write (little-endian uint16, row-major, no header)
      width, height       the grid - the landscape's vertex count per side
      cellSize, cellSizeY, heightScale, midValue, origin - as mesh_from_heightmap: sample (i, j) is read at
                          origin + (i x cellSize, -j x cellSizeY) and v = midValue + (z - origin.z) / heightScale
      missValue           what a grid point the terrain does not cover gets (default midValue); misses are counted
      samples [[i, j]]    grid points to report the value and the hit height of (default the corners and the centre)
      overwrite (bool)    default true
    A height beyond 0-65535 is CLAMPED and counted - the landscape's range at that scale (scale.z x 256 m each way).
    Reply: hits, misses, clamped, the value range, the sample rows.
    """
    import struct                                                     # noqa: PLC0415
    reject_unknown(params, _HM_TO_KEYS, "heightmap_from_mesh")
    tail = "NOTHING was written."
    obj = get_object(take(params, "object", required=True, kind=str))
    path = os.path.abspath(bpy.path.abspath(take(params, "file", required=True, kind=str)))
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".r16", ".raw"):
        raise MifOpError("file is a .r16 / .raw - little-endian uint16, row-major, no header: the landscape's own raw "
                         "format, written byte-exact (a PNG goes through Blender's image saver and its colour handling). %s"
                         % tail)
    w = take_int(params, "width", required=True)
    h = take_int(params, "height", required=True)
    if w < 2 or h < 2 or w * h > 16_777_216:
        raise MifOpError("width and height are 2 or more, at most 16.7 million samples. %s" % tail)
    cx, cy, hs, mid, org = _hm_scales(params, tail)
    miss_v = take_float(params, "missValue", default=mid)
    if not 0 <= miss_v <= 65535:
        raise MifOpError("missValue is 0-65535. %s" % tail)
    check_output_path(path, path, "written")
    overwrite = take_bool(params, "overwrite", default=True)
    if file_before(path) is not None and not overwrite:
        raise MifOpError("'%s' exists and overwrite is false. %s" % (path, tail))
    from mathutils.bvhtree import BVHTree                              # noqa: PLC0415
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    try:
        me = ev.to_mesh()
    except RuntimeError:
        me = None
    if me is None or not len(me.polygons):
        raise MifOpError("'%s' has no faces to cast onto. %s" % (obj.name, tail))
    try:
        mw = obj.matrix_world
        tree = BVHTree.FromPolygons([mw @ v.co for v in me.vertices], [tuple(p.vertices) for p in me.polygons])
        top = max((mw @ v.co).z for v in me.vertices) + 1.0
    finally:
        ev.to_mesh_clear()
    down = Vector((0.0, 0.0, -1.0))
    vals = [0] * (w * h)
    hits = misses = clamped = 0
    heights = {}
    # A RAY DOWN AN EXACT EDGE OR CORNER CAN MISS (measured: the corners and the far column of a terrain the grid matches
    # vertex for vertex missed on 3.6, 4.4 and 5.2), so a miss is tried again a hair inside each way before it counts.
    # (1e-4 of a cell: 1e-6 is under float32's step at a few tens of metres and rounded back onto the edge)
    eps = 1e-4 * min(cx, cy)
    nudges = ((0.0, 0.0), (eps, -eps), (-eps, -eps), (eps, eps), (-eps, eps))
    for j in range(h):
        for i in range(w):
            for dx, dy in nudges:
                loc, _n, _f, _d = tree.ray_cast(Vector((org.x + i * cx + dx, org.y - j * cy + dy, top)), down)
                if loc is not None:
                    break
            if loc is None:
                v = miss_v
                misses += 1
            else:
                v = mid + (loc.z - org.z) / hs
                hits += 1
                heights[(i, j)] = loc.z
            iv = int(round(v))
            if iv < 0 or iv > 65535:
                clamped += 1
                iv = min(65535, max(0, iv))
            vals[j * w + i] = iv
    if not hits:
        raise MifOpError("no grid point hit '%s' - the grid (origin %s, %d x %d at %g m) does not lie over it. %s"
                         % (obj.name, list(org), w, h, cx, tail))
    # PARSED BEFORE THE WRITE: a malformed sample was a raw TypeError, a NaN a raw ValueError and 2.7 read as 2, each
    # after the heightmap was already on disk
    pts = params.get("samples") or [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [w // 2, h // 2]]
    if not isinstance(pts, list) or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in pts):
        raise MifOpError("samples is a list of [i, j] pairs. %s" % tail)
    want = [(finite_int(p[0], "samples"), finite_int(p[1], "samples")) for p in pts[:200]]
    with open(path, "wb") as fh:
        fh.write(struct.pack("<%dH" % (w * h), *vals))
    rows = []
    for i, j in want:
        if 0 <= i < w and 0 <= j < h:
            rows.append({"i": i, "j": j, "value": vals[j * w + i],
                         "hitZ": round(heights[(i, j)], 6) if (i, j) in heights else None})
    return {"object": obj.name, "file": path, "samples": [w, h], "hits": hits, "misses": misses, "clamped": clamped,
            "valueRange": {"min": min(vals), "max": max(vals)}, "rows": rows, "bytes": os.path.getsize(path),
            "mapping": "sample (i, j) at origin + (i x %g, -j x %g); v = %g + (z - %g) / %g" % (cx, cy, mid, org.z, hs)}


# --------------------------------------------------------------------------------------------- material from maps
# THE REVERSE OF export_material_textures (second 1.3 audit, 2026-10-25): Unreal's textures, each sampled for one material
# input and often PACKED (ORM: R occlusion, G roughness, B metallic) and DirectX-green normals, as one Blender material.
# Each map is written as its own file - a packed channel split out to grayscale, a DirectX normal's green inverted - so
# export_material_textures can read it straight back, and import_pbr_texture_set wires the folder (its colour spaces:
# sRGB for base colour and emission only).
_MFM_KEYS = ("material", "folder", "maps", "overwrite")
_MFM_ROLES = {"baseColor": "basecolor", "normal": "normal", "roughness": "roughness", "metallic": "metallic",
              "ao": "ao", "alpha": "opacity", "emissive": "emission"}
_MFM_COLOR = ("baseColor", "normal", "emissive")


def op_material_from_maps(params):
    """Build a Blender material's Principled inputs from texture files, each map given its role and the channel it is
    read from - a packed ORM split back to three maps, a DirectX normal map's green flipped.

    params:
      material (str)      an existing material (the one an FBX import made for the slot)
      folder (str)        where the per-role files are written: <folder>/<material>/<material>_<role word>.png (each
                          material's own names since 2026-11-16 - every material's maps were basecolor.png, normal.png,
                          and an engine import names a texture asset after its file, so they collided)
      maps [{file, role, channel?, directX?}]
                          role: baseColor | normal | roughness | metallic | ao | alpha | emissive; channel: RGB (the
                          default for baseColor, normal, emissive) | R | G | B | A (a scalar role's default is R);
                          directX (normal only): the file is DirectX-convention (green down) and is flipped to OpenGL
      overwrite (bool)    default true
    Reply: per role the file written with its channel means, and import_pbr_texture_set's wiring.
    """
    import shutil                                                   # noqa: PLC0415
    import numpy as np                                              # noqa: PLC0415 - Blender bundles it
    from .ops_material import op_import_pbr_texture_set             # noqa: PLC0415
    reject_unknown(params, _MFM_KEYS, "material_from_maps")
    tail = "NOTHING was changed."
    mat_name = take(params, "material", required=True, kind=str)
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        raise MifOpError("no material named '%s'. %s" % (mat_name, tail))
    maps = take(params, "maps", required=True)
    if not isinstance(maps, list) or not maps or not all(isinstance(m, dict) for m in maps):
        raise MifOpError("maps is a non-empty list of {file, role, channel?, directX?}. %s" % tail)
    overwrite = take_bool(params, "overwrite", default=True)
    plan, seen = [], set()
    for m in maps:
        extra = set(m) - {"file", "role", "channel", "directX"}
        if extra:
            raise MifOpError("a map takes file, role, channel and directX; unknown %s. %s" % (sorted(extra), tail))
        role = m.get("role")
        # A STRING FIRST: `role not in _MFM_ROLES` hashes the value, and a mapping or a list raised TypeError
        # (unhashable) out of the op - found 2026-10-28 by the version matrix's corrupted-value pass.
        if not isinstance(role, str) or role not in _MFM_ROLES:
            raise MifOpError("role %r is not one of %s. %s" % (role, ", ".join(_MFM_ROLES), tail))
        if role in seen:
            raise MifOpError("role '%s' is given twice - one map per input. %s" % (role, tail))
        seen.add(role)
        ch = str(m.get("channel") or ("RGB" if role in _MFM_COLOR else "R")).upper()
        if ch not in ("RGB", "R", "G", "B", "A") or (role in _MFM_COLOR) != (ch == "RGB"):
            raise MifOpError("%s reads channel %r; a colour role reads RGB, a scalar one R, G, B or A. %s"
                             % (role, m.get("channel"), tail))
        if m.get("directX") is not None and not isinstance(m.get("directX"), bool):
            raise MifOpError("directX is true or false, got %r. %s" % (m.get("directX"), tail))
        if m.get("directX") and role != "normal":
            raise MifOpError("directX belongs to the normal map, not %s. %s" % (role, tail))
        src = os.path.abspath(bpy.path.abspath(str(m.get("file") or "")))
        if not os.path.isfile(src):
            raise MifOpError("%s: no file '%s'. %s" % (role, src, tail))
        plan.append((role, ch, m.get("directX") is True, src))
    folder = os.path.join(_out_dir(params, tail), _safe(mat.name))
    prefix = _safe(mat.name) + "_"

    def final(role):
        return os.path.join(folder, prefix + _MFM_ROLES[role] + ".png")
    # WIRED FROM ROLE WORDS, NAMED AFTER THE MATERIAL: import_pbr_texture_set finds a map by the words in its file name,
    # and a material's own name can hold one ("Metal_Panel" reads as metallic), so the maps are written under their role
    # words in a staging folder, wired from there, then moved to <material>_<role> and the images pointed at them
    staging = os.path.join(folder, "_mif_wire")
    # EVERY FILE THIS WRITES, CHECKED BEFORE THE FOLDER IS TOUCHED
    for role, _ch, _dx, _src in plan:
        check_output_path(final(role), final(role), "changed")
    os.makedirs(folder, exist_ok=True)
    words = {prefix + w for w in _MFM_ROLES.values()} | set(_MFM_ROLES.values())
    stale = [f for f in os.listdir(folder) if os.path.splitext(f)[0] in words]
    if stale and not overwrite:
        raise MifOpError("'%s' already holds %s and overwrite is false. %s" % (folder, stale, tail))
    for f in stale:
        os.remove(os.path.join(folder, f))
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    written = []
    for role, ch, dx, src in plan:
        out = os.path.join(staging, _MFM_ROLES[role] + ".png")
        if ch == "RGB" and not dx and src.lower().endswith(".png"):
            shutil.copyfile(src, out)
            written.append({"role": role, "file": out, "from": src, "channel": ch, "how": "copied"})
            continue
        img = bpy.data.images.load(src, check_existing=False)
        try:
            img.colorspace_settings.name = "Non-Color"
            w, h, n = int(img.size[0]), int(img.size[1]), int(img.channels)
            px = np.empty(w * h * n, dtype=np.float32)
            img.pixels.foreach_get(px)
            px = px.reshape(h, w, n)
        finally:
            bpy.data.images.remove(img)
        rgba = np.ones((h, w, 4), dtype=np.float32)
        if ch == "RGB":
            rgba[..., :3] = px[..., :3] if n >= 3 else px[..., :1]
            if dx:
                rgba[..., 1] = 1.0 - rgba[..., 1]
            how = "green flipped (DirectX to OpenGL)" if dx else "copied"
        else:
            k = "RGBA".index(ch)
            if k >= n:
                # A channel count is known only once the file is loaded, so earlier roles may be written and the old
                # maps removed by now: the refusal says which, rather than NOTHING was changed
                done = ("already written: %s; removed first: %s" % ([r["file"] for r in written], stale)
                        if written or stale else tail)
                raise MifOpError("%s reads channel %s of '%s', which has %d channel(s). %s" % (role, ch, src, n, done))
            rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = px[..., k]
            how = "channel %s split out" % ch
        new = bpy.data.images.new(prefix + _MFM_ROLES[role], w, h, alpha=False)
        try:
            new.colorspace_settings.name = "Non-Color"
            new.pixels.foreach_set(rgba.ravel())
            new.filepath_raw = out
            new.file_format = "PNG"
            new.save()
        finally:
            bpy.data.images.remove(new)
        written.append({"role": role, "file": out, "from": src, "channel": ch, "how": how,
                        "mean": [round(float(rgba[..., c].mean()), 4) for c in range(3)]})
    try:
        wired = op_import_pbr_texture_set({"material": mat.name, "folder": staging})
        moved = {}
        for row in written:
            dst = final(row["role"])
            os.replace(row["file"], dst)
            moved[os.path.normcase(os.path.abspath(row["file"]))] = dst
            row["file"] = dst
        for img in bpy.data.images:
            fp = os.path.normcase(os.path.abspath(bpy.path.abspath(img.filepath))) if img.filepath else None
            if fp in moved:
                img.filepath = moved[fp]
                img.name = os.path.splitext(os.path.basename(moved[fp]))[0]
                img.reload()
        for w in (wired.get("wired") or {}).values():
            if isinstance(w, dict):
                for k in ("file", "path", "filepath"):
                    src_ = os.path.normcase(os.path.abspath(str(w.get(k)))) if w.get(k) else None
                    if src_ in moved:
                        w[k] = moved[src_]
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {"material": mat.name, "folder": folder, "maps": written, "wired": wired.get("wired"),
            "kinds": wired.get("kinds"), "skipped": wired.get("skipped"), "notes": wired.get("notes")}


# ============================================================================================================ uefn_check
# A UEFN-READY CHECK AND EXPORT (2026-11-16). UEFN loads no third-party editor plugin, so a piece made here reaches it
# as a file. This judges meshes against the limits Epic PUBLISHES for UEFN before anything is written, writes the file
# through the exporters that already exist (export_mesh for .fbx, export_scene for .glb/.gltf), and reads that file back
# to say what UEFN's importer will find in it. It makes no LODs of its own: generateLods calls build_lod_chain.
#
# EVERY LIMIT IS EPIC'S OWN SENTENCE, read 2026-11-16 from Epic's
# pages on UEFN publishing and import limits. What is not on a page - the bytes each texture and mesh costs - is THIS OP'S ESTIMATE, named as one in the
# answer; UEFN's own figure comes from cooked data (memory page: "UEFN calculates your island's publishing memory from
# data produced at cook time"), which only UEFN can produce.
_UEFN_SOURCES = {
    "memory": ("Epic, Memory Management in UEFN",
               "dev.epicgames.com/documentation/en-us/fortnite/memory-management-in-unreal-editor-for-fortnite"),
    "simplify": ("Epic, Simplify Static Meshes in Fortnite",
                 "dev.epicgames.com/documentation/en-us/fortnite/simplify-static-meshes-in-fortnite"),
    "ready": ("Epic, Creating Fortnite-Ready Assets in UEFN",
              "dev.epicgames.com/documentation/fortnite/fortniteready-assets-best-practices-in-fortnite"),
    "modeling": ("Epic, Modeling Tips in UEFN",
                 "dev.epicgames.com/documentation/en-us/fortnite/modeling-tips-in-unreal-editor-for-fortnite"),
    "import": ("Epic, Importing Assets in UEFN",
               "dev.epicgames.com/documentation/en-us/fortnite/importing-assets-in-unreal-editor-for-fortnite"),
    "fab": ("Epic, Upload UEFN Source to Fab",
            "dev.epicgames.com/documentation/fab/upload-unreal-editor-for-fortnite-source-to-fab"),
}
# memory: "You cannot publish your island if it exceeds the 300 MiB publishing memory budget."
UEFN_PUBLISH_BUDGET_MIB = 300
# simplify: "there's a rule that warns you if the mesh has an LOD0 and is over 30,000 vertices."
UEFN_LOD0_VERTEX_WARN = 30000
# simplify: "The static mesh validation reports any static mesh that has 500 vertices or more with only a single LOD"
UEFN_SINGLE_LOD_VERTICES = 500
# ready: "All texture maps should have a size/resolution of 2k or less."
UEFN_TEXTURE_MAX = 2048
# ready: "Try to limit your mesh to 10 UCX collision meshes or less."
UEFN_MAX_UCX = 10
# fab: "Keep every asset file path to 140 characters or less, including the name of the top-level folder."
FAB_PATH_MAX = 140
# fab: "Use only English alphanumeric characters and underscores in folder and file names."
_FAB_NAME = re.compile(r"^[A-Za-z0-9_]+$")
# ready: "Assets for real-time experiences should have three LODs (LOD0, LOD1 and LOD2)" and "Each LOD level should be
# half the polygons of the previous level." - so generateLods' default is two LODs at a half and a quarter.
UEFN_LOD_RATIOS = (0.5, 0.25)
# import: FBX opens the FBX Import Options and glTF/GLB the Interchange options. OBJ and USD are on the page too; this
# op writes the two PROGRAM-chosen families, FBX through export_mesh's measured axes and scale, glTF through the spec's.
_UEFN_FORMATS = (".fbx", ".glb", ".gltf")
_HULL_PREFIXES = ("UCX_", "UBX_", "USP_", "UCP_")
_LOD_SUFFIX = re.compile(r"_LOD(\d+)$")
_UC_KEYS = ("objects", "object", "file", "overwrite", "generateLods", "assetFolder", "strict", "embedTextures",
            "includeHulls", "animations")
_UC_NOTES = {
    "actions": "animations true (the default) writes every action that keys a rigged mesh's bones, one file each "
               "through export_actions; false names them in export.notWritten.",
    "format": "the format comes from the file's extension: .fbx, .glb or .gltf.",
    "lods": "generateLods true makes the missing LOD chains with build_lod_chain; a list of fractions sets them.",
    "lodRatios": "generateLods takes the ratios itself: generateLods [0.5, 0.25].",
    "force": "nothing forces an export past a failed check - fix what it names, or call export_mesh / export_scene "
             "directly, which judge nothing for UEFN.",
}

# THE ESTIMATE'S RATES (not from an Epic page; each is the size of the format UE picks by default for that kind of
# source). A float source imports as TC_HDR (5.8 Editor/UnrealEd/Private/Factories/EditorFactories.cpp:3463), RGBA16F:
# 8 bytes a pixel. A normal map is BC5, a source with an alpha channel BC3, anything else BC1: 1, 1 and 0.5 bytes a
# pixel. A full mip chain adds a third; a texture that is not a power of two is counted without one, which is what
# Epic's "power-of-2 ... makes it possible to do mipmapping" implies.
_BYTES_PER_PIXEL = {"hdr": 8.0, "normal": 1.0, "alpha": 1.0, "color": 0.5}
# A static mesh vertex: position 12 bytes, packed normal and tangent 8, 4 per UV channel (half precision), 4 for a
# color; an index 2 bytes under 65,536 vertices, else 4.
_VERTEX_BYTES = 20
_UV_BYTES = 4
_COLOR_BYTES = 4
# A SKINNED VERTEX'S BONE WEIGHTS (2026-11-16), read off UE 5.8's source, NOT measured on an imported mesh: the skin
# weight buffer holds, per vertex, MaxBoneInfluences (bone index, weight) pairs (Engine/Public/Rendering/
# SkinWeightVertexBuffer.h GetConstantInfluencesVertexStride), the influence count rounded up to 4 or 8 - 12 needs
# unlimited bone influences (Private/Rendering/SkinWeightVertexBuffer.cpp SetMaxBoneInfluences); an index is 1 byte, 2
# past 256 bones (Private/SkeletalMeshLODRenderData.cpp: GetNumUnifiedBones() > 256), and a weight 1 byte unless the
# mesh asks for high-precision weights. The influences are counted in Blender: the bones each vertex is weighted to.
_INFLUENCE_STEPS = (4, 8, 12)
_SKIN_INDEX_WIDE_BONES = 256


def _lod_group(obj):
    grp = obj.parent
    if grp is not None and grp.type == "EMPTY" and grp.get("fbx_type") == "LodGroup":
        return grp
    return None


def _lod_index(obj):
    m = _LOD_SUFFIX.search(obj.name)
    return int(m.group(1)) if m else 0


def _is_hull(obj):
    return obj.name.startswith(_HULL_PREFIXES)


def _hulls_of(name):
    return sorted((o for o in bpy.data.objects if o.type == "MESH" and any(
        o.name == p + name or o.name.startswith(p + name + "_") for p in _HULL_PREFIXES)), key=lambda o: o.name)


def _skinned(obj):
    return _rig_of(obj) is not None


def _rig_of(obj):
    from .ops_kit import _skin_armature                         # noqa: PLC0415 - build_lod_chain's own rule
    return _skin_armature(obj)


def _skinned_lods(obj):
    """A skinned mesh's LOD chain as build_lod_chain's skinned path leaves it - <name>_LOD1, <name>_LOD2, ... beside it,
    deformed by the same armature (no LodGroup: a skeletal mesh's LODs import one file each). LOD0 first."""
    rig = _rig_of(obj)
    out = [obj]
    if rig is None:
        return out
    for k in range(1, 8):
        lod = bpy.data.objects.get("%s_LOD%d" % (obj.name, k))
        if lod is None or lod.type != "MESH" or _rig_of(lod) is not rig:
            break
        out.append(lod)
    return out


def _skinned_lod0(obj):
    """The LOD0 a skinned <name>_LOD<k> belongs to, or None when obj is not one."""
    m = _LOD_SUFFIX.search(obj.name)
    if not m or int(m.group(1)) == 0:
        return None
    base = bpy.data.objects.get(obj.name[:m.start()])
    if base is None or base.type != "MESH" or _rig_of(base) is None or obj not in _skinned_lods(base):
        return None
    return base


def _skin_bytes(obj, rig):
    """(bytes, influences the buffer holds, bones, most bones one vertex is weighted to) for a skinned mesh's bone
    weights - _INFLUENCE_STEPS' rule on the bones Blender weights each vertex to (only groups that name a bone count)."""
    bones = {b.name for b in rig.data.bones}
    bone_group = {g.index for g in obj.vertex_groups if g.name in bones}
    most = 0
    for v in obj.data.vertices:
        n = sum(1 for g in v.groups if g.group in bone_group and g.weight > 0.0)
        if n > most:
            most = n
    steps = next((s for s in _INFLUENCE_STEPS if most <= s), _INFLUENCE_STEPS[-1])
    index = 2 if len(bones) > _SKIN_INDEX_WIDE_BONES else 1
    return steps * (index + 1), steps, len(bones), most


def _engine_vertices(obj, dg):
    """What a mesh costs once an engine builds it: Blender's vertex count, the ESTIMATED engine vertex count - one per
    distinct (vertex, corner normal, every UV, color) - and triangles, UV channels and color, of the evaluated mesh.

    modeling: "Get your final vertex count from UE rather than the modeling software you're using. Hard edges, multiple
    smoothing groups, and extra UVs can increase the asset's final vertex count in-engine." - so the limits are judged
    on the split count, and Blender's own is reported beside it."""
    import numpy as np                                          # noqa: PLC0415 - bundled with Blender, only needed here
    ev = obj.evaluated_get(dg)
    mesh = ev.to_mesh()
    try:
        mesh.calc_loop_triangles()
        tris = len(mesh.loop_triangles)
        nl = len(mesh.loops)
        nv = len(mesh.vertices)
        uvs = len(mesh.uv_layers)
        colors = len(getattr(mesh, "color_attributes", ()) or ())
        if nl == 0:
            return {"vertices": nv, "engineVertices": 0, "triangles": tris, "uvChannels": uvs, "hasColor": bool(colors)}
        vi = np.empty(nl, dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", vi)
        nrm = np.empty(nl * 3, dtype=np.float32)
        # BY VERSION, NOT hasattr: 3.6.23 has a corner_normals attribute too, and foreach_get("vector") on it raises
        # "internal error setting the array" (measured 2026-11-16)
        if tuple(bpy.app.version) >= (4, 1, 0):
            mesh.corner_normals.foreach_get("vector", nrm)
        else:
            mesh.calc_normals_split()
            mesh.loops.foreach_get("normal", nrm)
        cols = [vi.astype(np.float64)[:, None], np.round(nrm.reshape(-1, 3).astype(np.float64), 4) + 0.0]
        for layer in mesh.uv_layers:
            a = np.empty(nl * 2, dtype=np.float32)
            layer.data.foreach_get("uv", a)
            cols.append(np.round(a.reshape(-1, 2).astype(np.float64), 6) + 0.0)
        if colors:
            ca = mesh.color_attributes[max(0, getattr(mesh.color_attributes, "render_color_index", 0))]
            n = len(ca.data)
            c = np.empty(n * 4, dtype=np.float32)
            ca.data.foreach_get("color", c)
            c = np.round(c.reshape(-1, 4).astype(np.float64), 3) + 0.0
            cols.append(c[vi] if ca.domain == "POINT" else c)
        engine = int(np.unique(np.hstack(cols), axis=0).shape[0])
        return {"vertices": nv, "engineVertices": engine, "triangles": tris, "uvChannels": uvs,
                "hasColor": bool(colors)}
    finally:
        ev.to_mesh_clear()


def _mesh_bytes(row):
    v = row["engineVertices"]
    per = _VERTEX_BYTES + _UV_BYTES * row["uvChannels"] + (_COLOR_BYTES if row["hasColor"] else 0)
    return v * per + row["triangles"] * 3 * (2 if v < 65536 else 4)


def _image_kinds(objs):
    """{image name: (image, kinds)} for every image an image-texture node in these objects' materials reads, nested
    groups included; kind 'normal' when the node feeds a Normal Map node, else 'color'."""
    out, trees, done = {}, [], set()
    for o in objs:
        for slot in getattr(o, "material_slots", []):
            if slot.material is not None and slot.material.node_tree is not None:
                trees.append(slot.material.node_tree)
    while trees:
        t = trees.pop()
        # BY POINTER, NOT NAME (FOUND 2026-11-16): every material's own tree is named "Shader Nodetree", so a
        # name check read the first material's tree and skipped every other one's images
        if t.as_pointer() in done:
            continue
        done.add(t.as_pointer())
        # A NORMAL MAP BEHIND A FLIP (FOUND 2026-11-16): set_material_texture's DIRECTX route on a build whose Normal Map
        # node has no convention puts Separate Color, 1 - G and Combine Color between the image and the node, and the
        # image was counted as a color texture. Every image upstream of a Normal Map node's Color input is a normal map.
        normal_fed, todo, seen = set(), [lk.from_node for lk in t.links if lk.to_node.type == "NORMAL_MAP"
                                         and lk.to_socket.name == "Color"], set()
        while todo:
            nd = todo.pop()
            if nd.name in seen or len(seen) > 64:
                continue
            seen.add(nd.name)
            normal_fed.add(nd.name)
            todo.extend(lk.from_node for lk in t.links if lk.to_node == nd)
        for n in t.nodes:
            if n.type == "TEX_IMAGE" and n.image is not None:
                img, kinds = out.setdefault(n.image.name, (n.image, set()))
                kinds.add("normal" if n.name in normal_fed else "color")
            elif n.type == "GROUP" and getattr(n, "node_tree", None) is not None:
                trees.append(n.node_tree)
    return out


_LOOK_INPUTS = (("BC", "Base Color"), ("N", "Normal"), ("R", "Roughness"), ("M", "Metallic"), ("E", "Emission Color"),
                ("A", "Alpha"))


def _look_lost(mat):
    """What of a material's look no exported file can carry - the inputs driven by attributes or procedural nodes
    rather than an image or a value - as [{input, why}]. export_material_textures' own reading (_trace, and
    _simple_shader_row for a surface with no Principled BSDF), so the two ops never disagree on what is carried."""
    lost = []
    p = _principled(mat)
    top = _surface_shader(mat)
    if p is None or (top is not None and top.type in _SIMPLE_SHADERS):
        row = {"constants": {}, "notCarried": lost}
        _simple_shader_row(mat, row)
        return lost
    for key, label in _LOOK_INPUTS:
        src = _trace(_socket(p, _INPUTS[key]))
        if src[0] == "procedural":
            lost.append({"input": label, "why": "driven by %s" % src[1]})
        elif src[0] == "normalmap" and src[1][0] == "procedural":
            lost.append({"input": label, "why": "the Normal Map node reads %s, not an image wired straight to it"
                         % src[1][1]})
    if top is None or top.name != p.name:
        lost.append({"input": "*", "why": "the surface is a %s, not the Principled BSDF" % (
            top.type if top is not None else "disconnected node")})
    return lost


def _wired_images(objs):
    """The images whose texture node feeds another node in these objects' materials - what a material USES, as opposed
    to an image node left unconnected in its tree."""
    out, trees, done = set(), [], set()
    for o in objs:
        for slot in getattr(o, "material_slots", []):
            if slot.material is not None and slot.material.node_tree is not None:
                trees.append(slot.material.node_tree)
    while trees:
        t = trees.pop()
        # BY POINTER, NOT NAME (FOUND 2026-11-16): every material's own tree is named "Shader Nodetree", so a
        # name check read the first material's tree and skipped every other one's images
        if t.as_pointer() in done:
            continue
        done.add(t.as_pointer())
        for n in t.nodes:
            if n.type == "TEX_IMAGE" and n.image is not None and any(o.is_linked for o in n.outputs):
                out.add(n.image.name)
            elif n.type == "GROUP" and getattr(n, "node_tree", None) is not None:
                trees.append(n.node_tree)
    return out


def _pow2(n):
    return n > 0 and n & (n - 1) == 0


def _texture_row(img, kinds):
    w, h = (int(img.size[0]), int(img.size[1]))
    pow2 = _pow2(w) and _pow2(h)
    if img.is_float:
        kind = "hdr"
    elif "normal" in kinds:
        kind = "normal"
    elif int(getattr(img, "depth", 0) or 0) in (32, 64, 128):
        kind = "alpha"
    else:
        kind = "color"
    mips = 4.0 / 3.0 if pow2 else 1.0
    src = bpy.path.abspath(img.filepath) if img.filepath else ""
    return {"image": img.name, "width": w, "height": h, "powerOfTwo": pow2, "kind": kind,
            "source": img.source, "file": src or None, "fileExists": bool(src) and os.path.isfile(src),
            "packed": img.packed_file is not None,
            "estimatedBytes": int(w * h * _BYTES_PER_PIXEL[kind] * mips)}


def _check(checks, cid, severity, src, quote, message, **extra):
    row = {"check": cid, "severity": severity, "message": message, "source": _UEFN_SOURCES[src][0],
           "url": _UEFN_SOURCES[src][1], "quote": quote}
    row.update({k: v for k, v in extra.items() if v is not None})
    checks.append(row)


def _uefn_judge(targets, asset_folder, out_name, ext=None):
    """The check itself: meshes, textures, names, the memory estimate. Writes nothing. ext is the file to be written:
    a glTF carries LOD0 alone, so a chain is not counted for it, and an FBX embeds only images that are files."""
    dg = bpy.context.evaluated_depsgraph_get()
    checks, meshes, mem_meshes = [], [], 0
    gltf = ext in (".glb", ".gltf")
    for obj in targets:
        grp = _lod_group(obj)
        rig = _rig_of(obj)
        levels = [obj]
        if grp is not None and not gltf:
            levels = sorted((k for k in grp.children if k.type == "MESH"), key=_lod_index)
        elif rig is not None and not gltf:
            levels = _skinned_lods(obj)
        rows = []
        for lvl in levels:
            r = _engine_vertices(lvl, dg)
            r["object"] = lvl.name
            r["lod"] = _lod_index(lvl) if len(levels) > 1 else 0
            r["estimatedBytes"] = _mesh_bytes(r)
            if rig is not None:
                # THE BONE WEIGHTS of a skinned mesh (FOUND 2026-11-16: left out, about 8 bytes a vertex)
                per, steps, nbones, most = _skin_bytes(lvl, rig)
                r["skinWeights"] = {"bytesPerVertex": per, "influences": steps, "mostBonesOnAVertex": most,
                                    "bones": nbones, "estimatedBytes": per * r["engineVertices"]}
                r["estimatedBytes"] += per * r["engineVertices"]
            mem_meshes += r["estimatedBytes"]
            rows.append(r)
        lod0 = rows[0]
        hulls = _hulls_of(obj.name)
        ucx = [h.name for h in hulls if h.name.startswith("UCX_")]
        skinned = rig is not None
        sc = obj.matrix_world.to_scale()
        meshes.append({"object": obj.name, "skinned": skinned, "rig": rig.name if rig else None,
                       "lodGroup": grp.name if grp else None,
                       "sockets": sorted(c.name for c in obj.children if c.type == "EMPTY"
                                         and c.name.startswith("SOCKET_")) or None,
                       "lods": rows, "collisionHulls": [h.name for h in hulls],
                       "worldScale": [round(s, 6) for s in sc]})
        # THE VALIDATOR'S TWO VERTEX RULES, judged on the engine estimate - on a skinned mesh too (FOUND 2026-11-16: a
        # 910,000-vertex skinned body passed with no word). Epic publishes the numbers for the static mesh validator and
        # none for skeletal meshes, so a skinned mesh is held to the same ones and the message says so.
        kind = "skinned mesh" if skinned else "mesh"
        held = (" (a skeletal mesh, held to the static mesh validator's numbers - Epic publishes none of its own for "
                "skeletal meshes)" if skinned else "")
        if lod0["engineVertices"] > UEFN_LOD0_VERTEX_WARN:
            _check(checks, "lod0Vertices", "warn", "simplify",
                   "there's a rule that warns you if the mesh has an LOD0 and is over 30,000 vertices.",
                   "%s '%s' LOD0 has about %d engine vertices (%d in Blender), over %d - UEFN's validator warns%s. "
                   "Reduce it (decimate_mesh) or rely on Nanite." % (kind, obj.name, lod0["engineVertices"],
                                                                     lod0["vertices"], UEFN_LOD0_VERTEX_WARN, held),
                   object=obj.name, value=lod0["engineVertices"], limit=UEFN_LOD0_VERTEX_WARN, skinned=skinned)
        if len(rows) < 2 and lod0["engineVertices"] >= UEFN_SINGLE_LOD_VERTICES:
            _check(checks, "singleLod", "warn", "simplify",
                   "The static mesh validation reports any static mesh that has 500 vertices or more with only a "
                   "single LOD", "%s '%s' has about %d engine vertices and one LOD - UEFN's validator reports it%s. "
                   "generateLods true makes a chain (build_lod_chain), FBX only%s." % (
                       kind, obj.name, lod0["engineVertices"], held,
                       ", one file per LOD for a skinned mesh" if skinned else ""),
                   object=obj.name, value=lod0["engineVertices"], limit=UEFN_SINGLE_LOD_VERTICES, skinned=skinned)
        if lod0["uvChannels"] == 0:
            _check(checks, "uvSet", "warn", "modeling", "Meshes need at least one UV set.",
                   "'%s' has no UV map (bl_uv_unwrap)." % obj.name, object=obj.name)
        if len(ucx) > UEFN_MAX_UCX:
            _check(checks, "ucxCount", "advice", "ready", "Try to limit your mesh to 10 UCX collision meshes or less.",
                   "'%s' has %d UCX_ hulls, over %d." % (obj.name, len(ucx), UEFN_MAX_UCX), object=obj.name,
                   value=len(ucx), limit=UEFN_MAX_UCX)
        if max(sc) - min(sc) > 1e-4 * max(1.0, max(abs(s) for s in sc)):
            _check(checks, "uniformScale", "advice", "ready", "Make sure you apply the scale uniformly.",
                   "'%s' is scaled %s - not uniform. apply_transform bakes it into the mesh." % (
                       obj.name, [round(s, 4) for s in sc]), object=obj.name)

    # TEXTURES - every image the exported materials read, LODs and hulls included
    every = []
    for m in meshes:
        every.extend(bpy.data.objects[r["object"]] for r in m["lods"])
    images = _image_kinds(every)
    textures, mem_tex = [], 0
    for name in sorted(images):
        img, kinds = images[name]
        row = _texture_row(img, kinds)
        textures.append(row)
        mem_tex += row["estimatedBytes"]
        if row["width"] == 0 or row["height"] == 0:
            _check(checks, "textureMissing", "warn", "import",
                   "Materials and textures are imported by default with FBX",
                   "image '%s' has no pixels (its file %s is missing) - nothing of it reaches the file."
                   % (name, row["file"] or "(none)"), image=name)
            continue
        if ext == ".fbx" and not row["fileExists"]:
            _check(checks, "textureNotAFile", "warn", "import",
                   "Materials and textures are imported by default with FBX",
                   "image '%s' is %s with no file on disk - the FBX can neither embed nor point at it, so UEFN's FBX "
                   "import finds no texture. save_image it to a file first (a glTF writes it from its pixels)."
                   % (name, "packed" if row["packed"] else row["source"].lower()), image=name)
        if not row["powerOfTwo"]:
            _check(checks, "powerOfTwo", "warn", "ready",
                   "Use power-of-2 scaling for texture maps (64, 128, 256, 512, 1024, 2048). This makes it possible "
                   "to do mipmapping.",
                   "'%s' is %d x %d, not a power of two - it gets no mipmaps and cannot stream, so it always costs "
                   "its full size (memory page: textures that cannot stream \"always use the maximum amount of "
                   "memory\"; a texture set to No Mipmaps fails validation). resize_image to a power of two."
                   % (name, row["width"], row["height"]), image=name, value=[row["width"], row["height"]])
        if max(row["width"], row["height"]) > UEFN_TEXTURE_MAX:
            _check(checks, "textureSize", "advice", "ready",
                   "All texture maps should have a size/resolution of 2k or less.",
                   "'%s' is %d x %d, over %d." % (name, row["width"], row["height"], UEFN_TEXTURE_MAX), image=name,
                   value=max(row["width"], row["height"]), limit=UEFN_TEXTURE_MAX)

    # TWO TEXTURES, ONE ASSET NAME (FOUND 2026-11-16 on 58 assets: every material's maps were baseColor.png, normal.png
    # ...): UEFN names a texture asset after its file, so two images whose files share a name land on one asset name
    by_name = {}
    for t in textures:
        key = os.path.splitext(os.path.basename(t["file"]))[0] if t["file"] else t["image"]
        by_name.setdefault(key.lower(), []).append(t["image"])
    for key, imgs in sorted(by_name.items()):
        if len(imgs) > 1:
            _check(checks, "textureNameClash", "warn", "import", "Materials and textures are imported by default with FBX",
                   "images %s all become a texture named '%s' - one asset name in UEFN, so one replaces or renames the "
                   "other. Give each file its own name (material_from_maps names them <material>_<role>)."
                   % (", ".join(imgs), key), images=imgs, name=key)

    # A LOOK NO FILE CARRIES (FOUND 2026-11-16: a creature painted by mesh attributes driving a procedural shader
    # reported 0 MiB of textures and nothing else): an exported file carries image textures and values, never a node
    # chain, so an input driven by attributes or procedural nodes reaches UEFN as nothing.
    looks = []
    for mat in sorted({s.material for o in every for s in o.material_slots if s.material is not None},
                      key=lambda m: m.name):
        lost = _look_lost(mat)
        if lost:
            looks.append({"material": mat.name, "notCarried": lost})
            _check(checks, "materialLook", "warn", "import", "Materials and textures are imported by default with FBX",
                   "material '%s': %s - an exported file carries image textures and values, not a node chain, so this "
                   "look does not reach UEFN. Bake it to images (bake_texture_set, or bake_texture per map) and wire "
                   "them in (set_material_texture) before export." % (
                       mat.name, "; ".join("%s %s" % (x["input"], x["why"]) for x in lost[:6])),
                   material=mat.name, inputs=[x["input"] for x in lost])

    # NAMES - what becomes an asset (each mesh and LOD, each material, each texture's file name) and the file itself
    named = []
    for m in meshes:
        named.extend(("mesh", r["object"]) for r in m["lods"])
        named.extend(("collision", h) for h in m["collisionHulls"])
    mats = sorted({s.material.name for o in every for s in o.material_slots if s.material is not None})
    named.extend(("material", n) for n in mats)
    named.extend(("texture", os.path.splitext(os.path.basename(t["file"]))[0] if t["file"] else t["image"])
                 for t in textures)
    if out_name:
        named.append(("file", out_name))
    folder = (asset_folder or "").strip("/\\")
    for kind, n in named:
        if not _FAB_NAME.match(n):
            _check(checks, "assetName", "warn", "fab",
                   "Use only English alphanumeric characters and underscores in folder and file names.",
                   "%s name '%s' has characters outside A-Z, a-z, 0-9 and underscore - a UEFN Source package for Fab "
                   "refuses them (rename_object / set_material_properties name)." % (kind, n), name=n, kind=kind)
        length = len(n) + len(".uasset") + (len(folder) + 1 if folder else 0)
        if length > FAB_PATH_MAX:
            _check(checks, "pathLength", "warn", "fab",
                   "Keep every asset file path to 140 characters or less, including the name of the top-level "
                   "folder.", "%s '%s' makes a %d-character asset path%s, over %d." % (
                       kind, n, length, " under '%s'" % folder if folder else "", FAB_PATH_MAX),
                   name=n, kind=kind, value=length, limit=FAB_PATH_MAX)

    total = mem_meshes + mem_tex
    mib = total / 1048576.0
    if mib > UEFN_PUBLISH_BUDGET_MIB:
        _check(checks, "memoryBudget", "fail", "memory",
               "You cannot publish your island if it exceeds the 300 MiB publishing memory budget.",
               "these objects alone are estimated at %.1f MiB, over the island's %d MiB publishing budget."
               % (mib, UEFN_PUBLISH_BUDGET_MIB), value=round(mib, 2), limit=UEFN_PUBLISH_BUDGET_MIB)
    memory = {"estimatedMiB": round(mib, 3), "meshesMiB": round(mem_meshes / 1048576.0, 3),
              "texturesMiB": round(mem_tex / 1048576.0, 3), "budgetMiB": UEFN_PUBLISH_BUDGET_MIB,
              "shareOfBudget": round(mib / UEFN_PUBLISH_BUDGET_MIB, 4),
              "note": "an ESTIMATE by this op, not UEFN's figure: UEFN calculates publishing memory from cooked data, "
                      "for the whole island (devices, landscape and every other asset count too). Textures at the "
                      "format UE picks by default - BC1 0.5 B/px, BC3 or BC5 1 B/px, a float source RGBA16F 8 B/px - "
                      "with a full mip chain (+1/3) on a power-of-two texture; mesh vertices at 20 bytes plus 4 per UV "
                      "channel and 4 for a color, indices 2 or 4 bytes; a skinned mesh's bone weights at its "
                      "influences (rounded up to 4 or 8) times a 1-byte weight and a 1-byte bone index (2 past 256 "
                      "bones) a vertex - UE's skin weight buffer layout read from its source, not measured on an "
                      "import; every LOD counted.",
              "skinWeightsMiB": round(sum(r.get("skinWeights", {}).get("estimatedBytes", 0) for m in meshes
                                          for r in m["lods"]) / 1048576.0, 3)}
    return meshes, textures, checks, memory, looks


# --------------------------------------------------------------------------------------------------- read-back
def _image_dims(blob):
    """(width, height) of PNG or JPEG bytes, or None."""
    b = bytes(blob[:65536]) if not isinstance(blob, bytes) else blob
    if b[:8] == b"\x89PNG\r\n\x1a\n" and len(b) >= 24:
        return tuple(int(x) for x in struct.unpack(">II", b[16:24]))
    if b[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(b):
            if b[i] != 0xFF:
                i += 1
                continue
            marker = b[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", b[i + 5:i + 9])
                return (int(w), int(h))
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            (seg,) = struct.unpack(">H", b[i + 2:i + 4])
            i += 2 + seg
    return None


def _fbx_props(buf, pos, n):
    props = []
    for _ in range(n):
        t = chr(buf[pos])
        pos += 1
        if t in "YCIFDL":
            fmt = {"Y": "<h", "C": "<?", "I": "<i", "F": "<f", "D": "<d", "L": "<q"}[t]
            props.append(struct.unpack_from(fmt, buf, pos)[0])
            pos += struct.calcsize(fmt)
        elif t in "fdlib":
            length, enc, clen = struct.unpack_from("<III", buf, pos)
            pos += 12
            data = bytes(buf[pos:pos + clen])
            pos += clen
            props.append(("array", t, length, zlib.decompress(data) if enc else data))
        elif t in "SR":
            (length,) = struct.unpack_from("<I", buf, pos)
            pos += 4
            raw = bytes(buf[pos:pos + length])
            pos += length
            props.append(raw.decode("utf-8", "replace") if t == "S" else raw)
        else:
            raise ValueError("unknown FBX property type %r at byte %d" % (t, pos - 1))
    return props, pos


def _fbx_node(buf, pos, wide):
    if wide:
        end, nprops, _plen = struct.unpack_from("<QQQ", buf, pos)
        pos += 24
    else:
        end, nprops, _plen = struct.unpack_from("<III", buf, pos)
        pos += 12
    nlen = buf[pos]
    pos += 1
    if end == 0:
        return None, pos
    name = bytes(buf[pos:pos + nlen]).decode("ascii", "replace")
    pos += nlen
    props, pos = _fbx_props(buf, pos, nprops)
    kids = []
    while pos < end:
        kid, npos = _fbx_node(buf, pos, wide)
        if kid is None:
            pos = npos
            break
        kids.append(kid)
        pos = kid["end"]
    return {"name": name, "props": props, "children": kids, "end": end}, end


def _fbx_name(s):
    return s.split("\x00")[0] if isinstance(s, str) else s


def _read_fbx(path):
    """What a binary FBX holds, read off its bytes: units and axes, every mesh (control points, polygons, triangles,
    UV sets, bounds in centimeters), LOD groups and their children in the order an importer takes them, materials and
    textures (embedded or referenced, and an embedded image's size)."""
    with open(path, "rb") as fh:
        buf = memoryview(fh.read())
    if bytes(buf[:20]) != b"Kaydara FBX Binary  ":
        raise MifOpError("%s is not a binary FBX - the read-back reads binary FBX only." % path)
    (version,) = struct.unpack_from("<I", buf, 23)
    wide = version >= 7500
    pos, top = 27, {}
    while pos < len(buf):
        node, pos = _fbx_node(buf, pos, wide)
        if node is None:
            break
        top.setdefault(node["name"], node)
    settings = {}
    for p70 in (top.get("GlobalSettings") or {"children": []})["children"]:
        if p70["name"] == "Properties70":
            for p in p70["children"]:
                if p["props"] and p["props"][0] in ("UpAxis", "UpAxisSign", "FrontAxis", "FrontAxisSign",
                                                     "CoordAxis", "CoordAxisSign", "UnitScaleFactor"):
                    settings[p["props"][0]] = p["props"][-1]
    unit_cm = float(settings.get("UnitScaleFactor", 1.0))
    models, geoms, mats, videos, attrs, stacks = {}, {}, {}, {}, {}, []
    for n in (top.get("Objects") or {"children": []})["children"]:
        oid = n["props"][0] if n["props"] else None
        if n["name"] == "Model":
            # THE NODE'S OWN SCALE COUNTS: Blender's FBX exporter writes the unit conversion into the root node's Lcl
            # Scaling and leaves the vertices in Blender units (measured 2026-11-16 on 5.2.1: a 2 m cube's vertices
            # span 2.0 with UnitScaleFactor 1, under a node scaled 100).
            scale = [1.0, 1.0, 1.0]
            for c in n["children"]:
                if c["name"] == "Properties70":
                    for p in c["children"]:
                        if p["props"] and p["props"][0] == "Lcl Scaling" and len(p["props"]) >= 7:
                            scale = [float(v) for v in p["props"][4:7]]
            models[oid] = {"name": _fbx_name(n["props"][1]), "type": n["props"][2] if len(n["props"]) > 2 else None,
                           "scale": scale}
        elif n["name"] == "Geometry" and len(n["props"]) > 2 and n["props"][2] == "Mesh":
            g = {"controlPoints": 0, "polygons": 0, "triangles": 0, "uvSets": 0, "boundsCm": None}
            for c in n["children"]:
                if c["name"] == "Vertices" and c["props"] and isinstance(c["props"][0], tuple):
                    arr = array.array("d")
                    arr.frombytes(c["props"][0][3])
                    g["controlPoints"] = len(arr) // 3
                    if arr:
                        xs, ys, zs = arr[0::3], arr[1::3], arr[2::3]
                        g["boundsCm"] = [(max(v) - min(v)) * unit_cm for v in (xs, ys, zs)]
                elif c["name"] == "PolygonVertexIndex" and c["props"] and isinstance(c["props"][0], tuple):
                    arr = array.array("i")
                    arr.frombytes(c["props"][0][3])
                    sides, polys, tris = 0, 0, 0
                    for v in arr:
                        sides += 1
                        if v < 0:
                            polys += 1
                            tris += sides - 2
                            sides = 0
                    g["polygons"], g["triangles"] = polys, tris
                elif c["name"] == "LayerElementUV":
                    g["uvSets"] += 1
            geoms[oid] = g
        elif n["name"] == "Material":
            mats[oid] = _fbx_name(n["props"][1])
        elif n["name"] == "Video":
            v = {"name": _fbx_name(n["props"][1]), "file": None, "embeddedBytes": 0, "width": None, "height": None}
            for c in n["children"]:
                if c["name"] == "RelativeFilename" and c["props"]:
                    v["file"] = c["props"][0]
                elif c["name"] == "Content" and c["props"] and isinstance(c["props"][0], bytes) and c["props"][0]:
                    v["embeddedBytes"] = len(c["props"][0])
                    dims = _image_dims(c["props"][0])
                    if dims:
                        v["width"], v["height"] = dims
            videos[oid] = v
        elif n["name"] == "NodeAttribute":
            attrs[oid] = [p for p in n["props"] if isinstance(p, str)]
        elif n["name"] == "AnimationStack":
            # a take: its name and its length (FBX time is 46,186,158,000 ticks a second)
            t = {"name": _fbx_name(n["props"][1]), "seconds": None}
            for c in n["children"]:
                if c["name"] == "Properties70":
                    span = {p["props"][0]: p["props"][-1] for p in c["children"]
                            if p["props"] and p["props"][0] in ("LocalStart", "LocalStop")}
                    if "LocalStop" in span:
                        t["seconds"] = round((span["LocalStop"] - span.get("LocalStart", 0)) / 46186158000.0, 6)
            stacks.append(t)
    links = [(c["props"][1], c["props"][2]) for c in (top.get("Connections") or {"children": []})["children"]
             if c["name"] == "C" and c["props"] and c["props"][0] == "OO"]
    lod_attr = {a for a, strings in attrs.items() if "LodGroup" in strings}
    out_meshes = []
    for gid, g in geoms.items():
        owner = next((dst for src, dst in links if src == gid and dst in models), None)
        row = dict(g, name=models[owner]["name"] if owner in models else None)
        parent = next((dst for src, dst in links if src == owner and dst in models), None)
        row["parent"] = models[parent]["name"] if parent in models else None
        # boundsCm: the geometry's extent along its own axes times every node scale up the chain and the file's unit
        chain, node, seen = [1.0, 1.0, 1.0], owner, set()
        while node in models and node not in seen:
            seen.add(node)
            chain = [a * abs(b) for a, b in zip(chain, models[node]["scale"])]
            node = next((dst for src, dst in links if src == node and dst in models), None)
        row["nodeScale"] = [round(s, 6) for s in chain]
        if row["boundsCm"]:
            row["boundsCm"] = [round(b * s, 4) for b, s in zip(row["boundsCm"], chain)]
        row["materials"] = [mats[src] for src, dst in links if dst == owner and src in mats]
        out_meshes.append(row)
    groups = []
    for gm in {dst for src, dst in links if src in lod_attr and dst in models}:
        groups.append({"group": models[gm]["name"],
                       "lods": [models[src]["name"] for src, dst in links if dst == gm and src in models]})
    return {"format": "FBX", "version": version, "settings": settings, "centimetersPerFileUnit": unit_cm,
            "meshes": sorted(out_meshes, key=lambda r: r["name"] or ""), "lodGroups": groups,
            "materials": sorted(set(mats.values())), "textures": sorted(videos.values(), key=lambda v: v["name"]),
            "animations": stacks,
            "sockets": sorted({m["name"] for m in models.values() if (m["name"] or "").startswith("SOCKET_")})}


def _read_gltf(path):
    """What a .glb or .gltf holds: every mesh's vertices and triangles per its accessors (glTF stores one vertex per
    distinct corner, so this IS the count an importer starts from), UV sets, bounds in centimeters, materials and
    images with their size read from their bytes."""
    with open(path, "rb") as fh:
        raw = fh.read()
    bins = {}
    if raw[:4] == b"glTF":
        (jlen,) = struct.unpack_from("<I", raw, 12)
        doc = json.loads(raw[20:20 + jlen].decode("utf-8"))
        rest = 20 + jlen
        if rest + 8 <= len(raw):
            (blen,) = struct.unpack_from("<I", raw, rest)
            bins[0] = raw[rest + 8:rest + 8 + blen]
    else:
        doc = json.loads(raw.decode("utf-8"))
        for i, b in enumerate(doc.get("buffers", [])):
            uri = b.get("uri")
            if uri and not uri.startswith("data:"):
                fp = os.path.join(os.path.dirname(path), uri)
                if os.path.isfile(fp):
                    with open(fp, "rb") as fh:
                        bins[i] = fh.read()
    acc = doc.get("accessors", [])
    meshes = []
    for node in doc.get("nodes", []):
        if node.get("mesh") is None:
            continue
        m = doc["meshes"][node["mesh"]]
        verts = tris = 0
        uvs = 0
        lo, hi = None, None
        mats = []
        for prim in m.get("primitives", []):
            a = prim.get("attributes", {})
            if "POSITION" in a:
                pa = acc[a["POSITION"]]
                verts += int(pa.get("count", 0))
                if pa.get("min") and pa.get("max"):
                    lo = pa["min"] if lo is None else [min(x, y) for x, y in zip(lo, pa["min"])]
                    hi = pa["max"] if hi is None else [max(x, y) for x, y in zip(hi, pa["max"])]
            tris += int(acc[prim["indices"]]["count"]) // 3 if prim.get("indices") is not None else \
                int(acc[a["POSITION"]]["count"]) // 3 if "POSITION" in a else 0
            uvs = max(uvs, sum(1 for k in a if k.startswith("TEXCOORD_")))
            if prim.get("material") is not None:
                mats.append(doc["materials"][prim["material"]].get("name"))
        meshes.append({"name": node.get("name"), "vertices": verts, "triangles": tris, "uvSets": uvs,
                       "boundsCm": [round((b - a) * 100.0, 4) for a, b in zip(lo, hi)] if lo else None,
                       "scale": node.get("scale"), "materials": mats})
    images = []
    for img in doc.get("images", []):
        row = {"name": img.get("name"), "mimeType": img.get("mimeType"), "file": img.get("uri"), "width": None,
               "height": None}
        blob = None
        if img.get("bufferView") is not None:
            bv = doc["bufferViews"][img["bufferView"]]
            data = bins.get(bv.get("buffer", 0))
            if data is not None:
                off = bv.get("byteOffset", 0)
                blob = data[off:off + bv["byteLength"]]
        elif img.get("uri") and not img["uri"].startswith("data:"):
            fp = os.path.join(os.path.dirname(path), img["uri"])
            if os.path.isfile(fp):
                with open(fp, "rb") as fh:
                    blob = fh.read(65536)
        dims = _image_dims(blob) if blob else None
        if dims:
            row["width"], row["height"] = dims
        images.append(row)
    return {"format": "glTF", "generator": (doc.get("asset") or {}).get("generator"),
            "meshes": sorted(meshes, key=lambda r: r["name"] or ""), "lodGroups": [],
            "materials": sorted({m.get("name") for m in doc.get("materials", []) if m.get("name")}),
            "textures": images, "extensionsUsed": doc.get("extensionsUsed") or [],
            "animations": [a.get("name") for a in doc.get("animations", [])]}


def _judge_file(rb):
    """The limits applied again to what the FILE holds - names, texture sizes, vertex counts."""
    found = []
    for m in rb["meshes"]:
        n = m.get("name") or ""
        if not _FAB_NAME.match(n):
            found.append("mesh '%s' has characters outside A-Z, a-z, 0-9 and underscore" % n)
        verts = m.get("vertices", m.get("controlPoints", 0))
        if rb["format"] == "glTF" and verts > UEFN_LOD0_VERTEX_WARN:
            found.append("mesh '%s' holds %d vertices, over %d" % (n, verts, UEFN_LOD0_VERTEX_WARN))
        if not m.get("uvSets") and not n.startswith(_HULL_PREFIXES):    # a collision hull is never drawn
            found.append("mesh '%s' has no UV set" % n)
    for n in rb["materials"]:
        if not _FAB_NAME.match(n or ""):
            found.append("material '%s' has characters outside A-Z, a-z, 0-9 and underscore" % n)
    for t in rb["textures"]:
        if t.get("width") and t.get("height"):
            if not (_pow2(t["width"]) and _pow2(t["height"])):
                found.append("texture '%s' is %d x %d, not a power of two" % (t["name"], t["width"], t["height"]))
            if max(t["width"], t["height"]) > UEFN_TEXTURE_MAX:
                found.append("texture '%s' is over %d" % (t["name"], UEFN_TEXTURE_MAX))
    return found



class _NormalSwap:
    """The exported materials' OpenGL normal maps swapped for green-flipped copies while an FBX is written, and put back.
    rows: what was flipped."""

    def __init__(self):
        self.rows, self._nodes, self._made, self._renamed, self._dir = [], [], [], [], None

    def restore(self):
        for node, img in self._nodes:
            try:
                node.image = img
            except ReferenceError:
                pass
        for img in self._made:
            try:
                bpy.data.images.remove(img)
            except ReferenceError:
                pass
        for img, name in self._renamed:
            try:
                img.name = name
            except ReferenceError:
                pass
        if self._dir:
            import shutil                                           # noqa: PLC0415
            shutil.rmtree(self._dir, ignore_errors=True)


def _directx_normals(textures):
    """THE NORMAL MAP UNREAL READS (FOUND 2026-11-16): Blender reads a tangent normal map OpenGL-style (green up), and
    material_from_maps flips a DirectX file to OpenGL for it - but Unreal reads DirectX (green down), and the FBX
    embeds the file the material holds, so the relief arrived inverted. For the write, every image wired straight into
    a Normal Map node read as OpenGL (no convention property, or OPENGL) is replaced by a copy with its green flipped,
    under the same image name and file name; a node set to DIRECTX is left alone, as its file already is."""
    import tempfile                                                 # noqa: PLC0415
    import numpy as np                                              # noqa: PLC0415 - Blender bundles it
    swap = _NormalSwap()
    want = {t["image"] for t in textures if t["kind"] == "normal" and t["width"] and t["height"]}
    if not want:
        return swap
    by_name = {i.name: i for i in bpy.data.images}
    want = {by_name[n].as_pointer() for n in want if n in by_name}
    copies = {}
    try:
        for mat in bpy.data.materials:
            t = mat.node_tree if mat.use_nodes else None
            if t is None:
                continue
            for lk in list(t.links):
                n, to = lk.from_node, lk.to_node
                if n.type != "TEX_IMAGE" or n.image is None or n.image.as_pointer() not in want                         or to.type != "NORMAL_MAP" or lk.to_socket.name != "Color":
                    continue
                if str(getattr(to, "convention", "OPENGL")).upper() == "DIRECTX":
                    continue
                src = n.image
                if src.as_pointer() not in copies:
                    w, h, ch = int(src.size[0]), int(src.size[1]), int(src.channels)
                    px = np.empty(w * h * ch, dtype=np.float32)
                    src.pixels.foreach_get(px)
                    px = px.reshape(-1, ch)
                    if ch >= 2:
                        px[:, 1] = 1.0 - px[:, 1]
                    if swap._dir is None:
                        swap._dir = tempfile.mkdtemp(prefix="mif_uefn_dx_")
                    base = os.path.basename(bpy.path.abspath(src.filepath)) if src.filepath else src.name + ".png"
                    path = os.path.join(swap._dir, os.path.splitext(base)[0] + ".png")
                    name = src.name
                    swap._renamed.append((src, name))
                    src.name = name + "_mifOpenGL"
                    new = bpy.data.images.new(name, w, h, alpha=ch == 4)
                    swap._made.append(new)
                    new.colorspace_settings.name = "Non-Color"
                    rgba = np.ones((w * h, 4), dtype=np.float32)
                    rgba[:, :min(ch, 4)] = px[:, :4]
                    if ch < 3:
                        rgba[:, 2] = px[:, 0]
                    new.pixels.foreach_set(rgba.ravel())
                    new.filepath_raw = path
                    new.file_format = "PNG"
                    new.save()
                    copies[src.as_pointer()] = new
                    swap.rows.append({"image": name, "inBlender": "OPENGL", "inFile": "DIRECTX",
                                      "how": "green flipped for the write; the scene keeps its OpenGL map"})
                swap._nodes.append((n, src))
                n.image = copies[src.as_pointer()]
    except Exception:
        swap.restore()
        raise
    return swap

def op_uefn_check(params):
    """Judge meshes against UEFN's published limits and, with file, write a .fbx, .glb or .gltf UEFN imports and read
    it back. Every limit cites Epic's page; the memory figure is this op's estimate. Writes nothing without file; refuses
    the export when a check fails (or warns, with strict)."""
    reject_unknown(params, _UC_KEYS, "uefn_check", _UC_NOTES)
    started = time.perf_counter()
    raw = take(params, "file", kind=str)
    ext = os.path.splitext(raw)[1].lower() if raw else None
    if raw is not None and ext not in _UEFN_FORMATS:
        raise MifOpError("file must end in %s - the formats this writes for UEFN (FBX through export_mesh, glTF through "
                         "export_scene). Got %r. NOTHING was written." % (", ".join(_UEFN_FORMATS), raw))
    overwrite = take_bool(params, "overwrite", default=True)
    strict = take_bool(params, "strict", default=False)
    embed = take_bool(params, "embedTextures", default=True)
    with_hulls = take_bool(params, "includeHulls", default=True)
    # A RIGGED MESH'S CLIPS (FOUND 2026-11-16): the mesh went out through export_mesh with bakeAnim off, so its actions
    # stayed behind and the reply did not say so. animations true writes each action keying its bones, one file each
    # through export_actions (an FBX); false names them in export.notWritten.
    with_anims = take_bool(params, "animations", default=True)
    # generateLods: true (Epic's three LODs, each half the last) or the ratios themselves - one key, so the MCP tool
    # stays inside its pack's listed-character budget (test_mcp_packs P110)
    ratios = params.get("generateLods")
    if isinstance(ratios, list):
        if not ratios or any(isinstance(r, bool) or not isinstance(r, (int, float)) for r in ratios):
            raise MifOpError("generateLods takes true or a list of fractions, e.g. [0.5, 0.25], got %r. NOTHING was "
                             "changed." % (ratios,))
        ratios = finite_floats(ratios, "generateLods")
        if any(not 0.0 < r < 1.0 for r in ratios) or any(b >= a for a, b in zip(ratios, ratios[1:])) \
                or len(ratios) > 7:
            raise MifOpError("generateLods ratios must fall, each above 0 and below 1, at most 7 (build_lod_chain's "
                             "rules), got %s. NOTHING was changed." % ratios)
        gen = True
    else:
        gen = take_bool(params, "generateLods", default=False)
        ratios = list(UEFN_LOD_RATIOS)
    if gen and ext in (".glb", ".gltf"):
        raise MifOpError("generateLods writes FBX LOD groups, and a glTF file carries no LOD chain an importer reads as "
                         "LODs - UEFN builds them itself after import. Export .fbx, or drop generateLods. NOTHING was "
                         "changed.")
    folder = take(params, "assetFolder", kind=str)

    names = take(params, "objects")
    single = take(params, "object")
    if names is not None and single is not None:
        raise MifOpError("pass object OR objects, not both. NOTHING was changed.")
    if names is None and single is not None:
        names = [single]
    if names is not None and (not isinstance(names, list) or not names):
        raise MifOpError("'objects' is a non-empty list of mesh names. NOTHING was changed.")
    targets = []
    if names is not None:
        for n in names:
            o = get_object(n, want_mesh=True)
            if _lod_group(o) is not None and _lod_index(o) > 0:
                raise MifOpError("'%s' is LOD%d of '%s' - name its LOD0, which brings the whole chain. NOTHING was "
                                 "changed." % (o.name, _lod_index(o), _lod_group(o).name))
            base = _skinned_lod0(o)
            if base is not None:
                raise MifOpError("'%s' is LOD%d of the skinned '%s' - name its LOD0, which brings the whole chain. "
                                 "NOTHING was changed." % (o.name, _lod_index(o), base.name))
            if o not in targets:
                targets.append(o)
    else:
        for o in bpy.context.view_layer.objects:
            if o.type != "MESH" or not o.select_get() or _is_hull(o):
                continue
            grp = _lod_group(o)
            if grp is not None and _lod_index(o) > 0:
                o = next((k for k in grp.children if k.type == "MESH" and _lod_index(k) == 0), None)
                if o is None:
                    continue
            o = _skinned_lod0(o) or o
            if o not in targets:
                targets.append(o)
        if not targets:
            raise MifOpError("nothing to judge: no mesh is selected (collision hulls do not count). Select the meshes or "
                             "pass objects. NOTHING was changed.")

    # LODS FIRST, when asked: every object that needs a chain is checked against build_lod_chain's own refusals
    # before ANY chain is made, so a refusal leaves the scene as it was. A skinned mesh takes build_lod_chain's skinned
    # path (FOUND 2026-11-16: it was skipped): <name>_LOD1... beside it on the same armature, written one file each.
    lods_made = []
    if gen:
        dg = bpy.context.evaluated_depsgraph_get()
        need = [o for o in targets if _lod_group(o) is None and len(_skinned_lods(o)) == 1
                and _engine_vertices(o, dg)["engineVertices"] >= UEFN_SINGLE_LOD_VERTICES]
        blocked = []
        present = {ob.name for ob in bpy.data.objects}
        for o in need:
            rig = _rig_of(o)
            if o.parent is not None and o.parent is not rig:
                blocked.append("'%s' is parented to '%s'" % (o.name, o.parent.name))
            elif [c for c in o.children if not (c.type == "EMPTY" and c.name.startswith("SOCKET_"))]:
                blocked.append("'%s' has children (%s)" % (o.name, ", ".join(
                    c.name for c in o.children if not (c.type == "EMPTY" and c.name.startswith("SOCKET_")))))
            group = [] if rig is not None else ["%s_LODGroup" % o.name]
            taken = [n for n in group + ["%s_LOD%d" % (o.name, i + 1) for i in range(len(ratios))] if n in present]
            if taken:
                blocked.append("'%s': %s already exist" % (o.name, ", ".join(taken)))
        if blocked:
            raise MifOpError("generateLods cannot chain every mesh that needs it: %s (build_lod_chain's rules). NOTHING "
                             "was changed." % "; ".join(blocked))
        from .ops_kit import op_build_lod_chain                   # noqa: PLC0415 - the one LOD maker
        for o in need:
            r = op_build_lod_chain({"object": o.name, "ratios": ratios})
            lods_made.append({"object": o.name, "group": r.get("group"), "skinned": r.get("skinned") is True,
                              "lods": [{"object": x["object"], "triangles": x["triangles"]} for x in r["lods"]]})

    out_name = os.path.splitext(os.path.basename(raw))[0] if raw else None
    meshes, textures, checks, memory, looks = _uefn_judge(targets, folder, out_name, ext)
    if ext == ".fbx" and not embed and any(t["kind"] == "normal" for t in textures):
        _check(checks, "normalConvention", "warn", "import", "Materials and textures are imported by default with FBX",
               "embedTextures is off, so the FBX points at the normal map files the materials read - OpenGL (green up), "
               "as Blender reads them - and Unreal reads DirectX (green down). Embed them (embedTextures true, "
               "flipped for the file) or flip the green channel on import.",
               images=[t["image"] for t in textures if t["kind"] == "normal"])
    # THE CLIPS EACH RIG CARRIES: every action that keys one of its bones (export_actions' own default set)
    rigs = []
    for m in meshes:
        if m["rig"] and m["rig"] not in [r["rig"] for r in rigs]:
            arm = bpy.data.objects[m["rig"]]
            bones = {b.name for b in arm.data.bones}
            rigs.append({"rig": arm.name, "actions": sorted(a.name for a in bpy.data.actions
                                                            if _action_bones(a)[0] & bones)})
    counts = {s: sum(1 for c in checks if c["severity"] == s) for s in ("fail", "warn", "advice")}
    out = {"ok": True, "objects": [o.name for o in targets], "passed": counts["fail"] == 0,
           "counts": counts, "checks": checks, "meshes": meshes, "textures": textures, "memory": memory,
           "lodsGenerated": lods_made or None, "rigs": rigs or None, "materialsNotCarried": looks or None,
           "notChecked": [
               "the 2 GB cook memory limit for one static mesh (simplify page) - UEFN estimates it at cook time",
               "Minimum LOD for Quality (Low = 2, Mid = 2) - a UEFN static mesh setting, not in the file",
               "runtime memory per platform and the island's other assets - only UEFN's memory calculation sees them",
               "MipGenSettings - a UEFN texture setting; a power-of-two texture can have mips, set in UEFN"]}

    if raw is None:
        out["note"] = "check only - pass file (.fbx, .glb, .gltf) to export."
        out["elapsedSeconds"] = round(time.perf_counter() - started, 2)
        return out
    stop = [c for c in checks if c["severity"] == "fail" or (strict and c["severity"] == "warn")]
    if stop:
        raise MifOpError("%d check(s) stop the export%s: %s. NOTHING was written."
                         % (len(stop), " (strict)" if strict else "",
                            "; ".join("%s: %s" % (c["check"], c["message"]) for c in stop[:8])))

    written_files = []
    not_written = {}
    if ext == ".fbx":
        objs = [o.name for o in targets]
        hulls = []
        if with_hulls:
            for o in targets:
                hulls.extend(h.name for h in _hulls_of(o.name) if h.name not in objs and h.name not in hulls)
        sub = {"objects": objs + hulls, "file": raw, "overwrite": overwrite}
        if any(_lod_group(o) is not None for o in targets):
            sub["lodGroups"] = True
        # SOCKETS GO WITH THEIR MESH (FOUND 2026-11-16: only meshes and hulls were written, so a SOCKET_ Empty under a
        # mesh was lost): export_mesh sockets writes each, and Unreal's FBX importer makes it a static mesh socket
        sockets = [s_ for m in meshes if not m["skinned"] for s_ in (m["sockets"] or [])]
        if sockets:
            sub["sockets"] = sockets
        if embed and textures:
            sub["pathMode"] = "COPY"
            sub["embedTextures"] = True
        from .ops_mesh import op_export_mesh                      # noqa: PLC0415 - the one FBX writer
        swap = _directx_normals(textures) if embed else None
        try:
            written = op_export_mesh(sub)
            written_files.append(written["file"])
            # A SKINNED CHAIN'S LODS GO ONE FILE EACH, as build_lod_chain writes them and a skeletal mesh imports them
            # (the Skeletal Mesh editor's LOD Import takes one file per LOD): <file>_<mesh>_LOD<k>.fbx, each with the rig
            stem = os.path.splitext(written["file"])[0]
            lod_files = []
            for m in meshes:
                if not m["skinned"] or m["lodGroup"]:
                    continue
                for r in m["lods"][1:]:
                    lsub = {"objects": [r["object"]], "file": "%s_%s_LOD%d.fbx" % (stem, m["object"], r["lod"]),
                            "overwrite": overwrite}
                    if "pathMode" in sub:
                        lsub.update(pathMode="COPY", embedTextures=True)
                    got = op_export_mesh(lsub)
                    written_files.append(got["file"])
                    lrb = _read_fbx(got["file"])
                    lod_files.append({"object": m["object"], "lod": r["lod"], "mesh": r["object"], "file": got["file"],
                                      "bytes": got.get("fileSizeBytes"),
                                      "meshesInFile": [x["name"] for x in lrb["meshes"]]})
        finally:
            if swap is not None:
                swap.restore()
        out["export"] = {"through": "export_mesh", "params": sub, "file": written.get("file"),
                         "bytes": written.get("fileSizeBytes"), "overwrote": written.get("overwrote"),
                         "axis": written.get("axis"), "embeddedImages": written.get("embeddedImages"),
                         "imagesNotEmbedded": written.get("imagesNotEmbedded"),
                         "lodGroupsWritten": written.get("lodGroupsWritten"),
                         "socketsWritten": written.get("socketsWritten"),
                         "warnings": {k: v for k, v in written.items() if k.endswith("Warning")} or None}
        rb = _read_fbx(written["file"])
        flipped = {r_["image"] for r_ in (swap.rows if swap is not None else [])}
        if swap is not None and swap.rows:
            out["export"]["normalMapsFlipped"] = swap.rows
        for t in textures:
            if t["kind"] == "normal":
                t["normalConvention"] = {"inFile": "DIRECTX" if embed else "the file on disk, as the material reads it",
                                         "flippedForTheFile": t["image"] in flipped}
        if lod_files:
            out["export"]["skinnedLodFiles"] = lod_files
        if rigs:
            if with_anims:
                anims = []
                for rg in rigs:
                    if not rg["actions"]:
                        continue
                    got = op_export_actions({"armature": rg["rig"], "actions": rg["actions"],
                                             "folder": os.path.dirname(written["file"]),
                                             "pattern": os.path.basename(stem) + "_{action}", "overwrite": overwrite})
                    for f in got["files"]:
                        written_files.append(f["file"])
                        anims.append({"rig": rg["rig"], "action": f["action"], "file": f["file"], "takes": f.get("takes"),
                                      "frames": f["frames"], "seconds": f["seconds"], "bytes": f["bytes"]})
                out["animations"] = anims or None
            elif any(rg["actions"] for rg in rigs):
                not_written["animations"] = [a for rg in rigs for a in rg["actions"]]
    else:
        # a skinned mesh's armature goes with it, or the glTF holds the mesh posed and static; its actions ride along
        names_out = [o.name for o in targets] + [rg["rig"] for rg in rigs]
        sub = {"objects": names_out, "file": raw, "overwrite": overwrite, "applyModifiers": True}
        if rigs and not with_anims:
            sub["animation"] = False
        from .ops_io import op_export_scene                       # noqa: PLC0415 - the one glTF writer
        written = op_export_scene(sub)
        written_files.append(written["file"])
        out["export"] = {"through": "export_scene", "params": sub, "file": written.get("file"),
                         "bytes": written.get("bytes"), "overwrote": written.get("overwrote")}
        rb = _read_gltf(written["file"])
        left = [o.name for o in targets if _lod_group(o) is not None or len(_skinned_lods(o)) > 1]
        hull_names = [h.name for o in targets for h in _hulls_of(o.name)]
        socks = [s_ for m in meshes for s_ in (m["sockets"] or [])]
        if socks:
            not_written["sockets"] = socks
        for t in textures:
            if t["kind"] == "normal":
                t["normalConvention"] = {"inFile": "OPENGL - glTF's own convention, which an importer converts",
                                         "flippedForTheFile": False}
        if left:
            not_written["lodChains"] = left
        if hull_names:
            not_written["collisionHulls"] = hull_names
        if left or hull_names:
            not_written["why"] = ("a glTF holds LOD0 only - no LOD chain an importer reads as LODs - and a UCX_ hull is "
                                  "an FBX importer convention; export .fbx to carry them.")
        if rigs:
            in_file = set(rb.get("animations") or [])
            out["animations"] = [{"rig": rg["rig"], "action": a, "file": written["file"]}
                                 for rg in rigs for a in rg["actions"] if a in in_file] or None
            missing = [a for rg in rigs for a in rg["actions"] if a not in in_file]
            if missing:
                not_written["animations"] = missing
    if not_written:
        out["export"]["notWritten"] = not_written
    rb["foundAgain"] = _judge_file(rb)
    # every LOD the main file should hold: a LodGroup's whole chain on FBX, LOD0 alone on glTF and for a skinned chain
    want = set()
    for m in meshes:
        rows = m["lods"] if ext == ".fbx" and not (m["skinned"] and not m["lodGroup"]) else m["lods"][:1]
        want.update(r["object"] for r in rows)
    have = {m["name"] for m in rb["meshes"]}
    rb["missingFromFile"] = sorted(want - have) or None
    # AN IMAGE THE MATERIALS USE AND THE FILE LEFT OUT (FOUND 2026-11-16: a normal map behind a flip was dropped by the
    # FBX exporter and strict passed): read off the file, never the scene. A texture already named missing or not a
    # file is not counted again.
    flagged = {c.get("image") for c in checks if c["check"] in ("textureMissing", "textureNotAFile")}
    held = set()
    for t in rb["textures"]:
        if ext == ".fbx" and embed and not t.get("embeddedBytes"):
            continue
        for key in (t.get("name"), os.path.splitext(os.path.basename(t.get("file") or ""))[0]):
            if key:
                held.add(key)
    wired = _wired_images([bpy.data.objects[r["object"]] for m in meshes for r in m["lods"]])
    dropped = [t["image"] for t in textures if t["image"] in wired and t["image"] not in flagged and t["width"] and t["height"]
               and t["image"] not in held and os.path.splitext(os.path.basename(t["file"] or ""))[0] not in held]
    if dropped:
        _check(checks, "imageNotInFile", "warn", "import", "Materials and textures are imported by default with FBX",
               "the written file %s %s - the exporter left %s out (an image reaches a file only wired straight to a "
               "Principled input or a Normal Map node), so UEFN imports the material without %s."
               % ("does not embed" if ext == ".fbx" and embed else "does not hold", ", ".join(dropped),
                  "it" if len(dropped) == 1 else "them", "it" if len(dropped) == 1 else "them"),
               images=dropped)
        out["counts"] = {s: sum(1 for c in checks if c["severity"] == s) for s in ("fail", "warn", "advice")}
        if strict:
            for f in written_files:
                try:
                    os.remove(f)
                except OSError:
                    pass
            raise MifOpError("strict: the written file left out %s, which the materials use - the file%s written by "
                             "this call (%s) %s removed%s. Wire each image straight to its input (or bake the chain to "
                             "an image) and export again."
                             % (", ".join(dropped), "s" if len(written_files) > 1 else "",
                                ", ".join(os.path.basename(f) for f in written_files),
                                "were" if len(written_files) > 1 else "was",
                                "; a file it replaced is gone too" if out["export"].get("overwrote") else ""))
    out["counts"] = {s_: sum(1 for c in checks if c["severity"] == s_) for s_ in ("fail", "warn", "advice")}
    out["readBack"] = rb
    out["elapsedSeconds"] = round(time.perf_counter() - started, 2)
    return out


OPS = {
    "material_from_maps": op_material_from_maps,
    "mesh_from_heightmap": op_mesh_from_heightmap,
    "heightmap_from_mesh": op_heightmap_from_mesh,
    "export_actions": op_export_actions,
    "export_material_textures": op_export_material_textures,
    "export_camera": op_export_camera,
    "export_groom": op_export_groom,
    "uefn_check": op_uefn_check,
}
