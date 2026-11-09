"""MifBlender ops: the hand-off to Unreal beyond meshes - animation clips one FBX per action, a material's textures with
engine names, an animated camera, and hair as an Alembic groom (1.3 gap audit, 2026-10-19).

THE GAP. mif_send_to_unreal handled static and skinned meshes and nothing else: export_mesh's bakeAnimAllActions wrote every
clip into one file (a re-export of one clip re-imported all of them, named Armature|Action), a mesh reached Unreal with
material SLOTS and no textures, export_mesh refuses a camera by design, and hair had no route at all. Every FBX here is
written with export_mesh's axes and scale (ops_mesh.FBX_EXPORT_ARGS: Z up, -Y forward, 1 Blender unit = 100 Unreal units),
so a clip lands on the skeleton a skinned send made.
"""
from __future__ import annotations

import os
import re

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
      deformOnly (bool)   only deform bones (default true - the bones a skinned send writes for Unreal)
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
                 "use_armature_deform_only": take_bool(params, "deformOnly", default=True), "add_leaf_bones": False})
    scene = bpy.context.scene
    ad = arm.animation_data
    saved = (ad.action if ad else None, getattr(ad, "action_slot", None) if ad else None,
             scene.frame_start, scene.frame_end, scene.frame_current)
    nla_mute = None
    if ad is not None and ad.use_nla and len(ad.nla_tracks):
        nla_mute = ad.use_nla
    snap = selection_snapshot()
    rows = []
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
            bpy.ops.export_scene.fbx(filepath=path, use_selection=True, **args)
            if not os.path.isfile(path) or os.path.getsize(path) <= 0:
                raise MifOpError("the FBX exporter reported done and wrote nothing at %s." % path)
            row = {"action": act.name, "file": path, "bytes": os.path.getsize(path), "frameStart": start,
                   "frameEnd": end, "frames": end - start + 1, "fps": round(fps, 6),
                   "seconds": round((end - start) / fps, 6), "bonesKeyed": len(bones),
                   "manualFrameRange": bool(getattr(act, "use_frame_range", False))}
            row.update(replaced_fields(before))
            rows.append(row)
    finally:
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
            "axis": {"up": "Z", "front": "-Y", "unit": "cm"},
            "note": "each file holds one take; Unreal imports it onto an existing Skeleton (import_asset skeleton)"}


# --------------------------------------------------------------------------------------------- export_material_textures

_EMT_KEYS = ("objects", "object", "materials", "folder", "directory", "pattern", "normalConvention", "packOrm",
             "format", "overwrite")
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
    conv = str(take(params, "normalConvention", default="DIRECTX", kind=str)).upper()
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
                                                  "texture - set it on the material in Unreal" % nm[3]})
                else:
                    row["notCarried"].append({"input": "Normal", "why": "a %s-space normal map, or not an image" % nm[2]})
            elif nm[0] not in ("none", "value"):
                row["notCarried"].append({"input": "Normal", "why": "driven by %s (a bump or a node chain)" % (nm[1],)})
            # ORM / R / M
            rough, metal = src["R"], src["M"]
            if pack_orm and (rough[0] == "image" or metal[0] == "image"):
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
           "packOrm": pack_orm}
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
      folder (str)        where the per-role files are written: <folder>/<material>/<role word>.png
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
    # EVERY FILE THIS WRITES, CHECKED BEFORE THE FOLDER IS TOUCHED - the names are fixed role words, so this guards the
    # folder half, which is the caller's
    for role, _ch, _dx, _src in plan:
        out = os.path.join(folder, _MFM_ROLES[role] + ".png")
        check_output_path(out, out, "changed")
    os.makedirs(folder, exist_ok=True)
    stale = [f for f in os.listdir(folder) if os.path.splitext(f)[0] in _MFM_ROLES.values()]
    if stale and not overwrite:
        raise MifOpError("'%s' already holds %s and overwrite is false. %s" % (folder, stale, tail))
    for f in stale:
        os.remove(os.path.join(folder, f))
    written = []
    for role, ch, dx, src in plan:
        out = os.path.join(folder, _MFM_ROLES[role] + ".png")
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
        new = bpy.data.images.new(_MFM_ROLES[role], w, h, alpha=False)
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
    wired = op_import_pbr_texture_set({"material": mat.name, "folder": folder})
    return {"material": mat.name, "folder": folder, "maps": written, "wired": wired.get("wired"),
            "kinds": wired.get("kinds"), "skipped": wired.get("skipped"), "notes": wired.get("notes")}


OPS = {
    "material_from_maps": op_material_from_maps,
    "mesh_from_heightmap": op_mesh_from_heightmap,
    "heightmap_from_mesh": op_heightmap_from_mesh,
    "export_actions": op_export_actions,
    "export_material_textures": op_export_material_textures,
    "export_camera": op_export_camera,
    "export_groom": op_export_groom,
}
