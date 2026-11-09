"""MifBlender ops: Rigify - bone collections, metarigs and samples, rig types, generation, and the generated rig's snap.

THE GAP (spec, "Blender: Rigify metarigs, rig types, generation, bone collections and snapping", 2026-10-04): Blender's
bundled rig generator is how a humanoid or an animal gets a full control rig, and no typed call reached it - nor the bone
collections (layers on 3.6) a rig is organised by. bl_create_armature and bl_add_bones built bones from lists only.

MEASURED BEFORE BUILT, on 3.6.23 (rigify 0.6.7), 4.4.0 and 5.2.1 (0.6.10), headless (--background --factory-startup):
  * addon_utils.enable("rigify", default_set=False) HALF-REGISTERS the generator on all three: its register() reads
    preferences.addons["rigify"] and raises, after PoseBone.rigify_type exists but with RigifyParameters holding 3
    properties instead of 90 (3.6) or 98 (4.4, 5.2), and a later generate throws "'RigifyParameters' object has no
    attribute 'make_custom_pivot'" with a 159-bone partial rig left behind. So the generator is enabled with
    default_set=True - an in-memory preferences entry - and preferences.is_dirty is put back when the op found it
    clean, so an auto-save of preferences does not record it. Nothing here saves preferences.
  * the metarig add operators (object.armature_<name>_metarig_add, eight on every build) run headless in object mode
    and place the armature at the 3D cursor; armature.metarig_sample_add needs EDIT_ARMATURE and places the sample at
    the armature's origin; pose.rigify_generate runs headless: the human metarig gives 706 bones in 3.4-4.1 s and
    regenerates in 7-9 s, with its UI script run and its snap operators registered in the same session.
  * the generator reports a failure with report(ERROR) and returns FINISHED, so bpy.ops raises AFTER the half-built rig
    exists; 4.4 and 5.2 refuse a metarig with no non-reserved collection holding a UI row ('No bone collections have
    UI buttons assigned - please check the Bone Collections UI sub-panel') while 3.6 generates it.
  * a face bone left with no children throws "'bpy_prop_collection[key]: key "ORG-nose" not found'" with a 68-bone
    partial rig left (all three builds).
  * a limb whose first two bones are nearly straight is rigged about the bone's ROLL axis instead of its bend: the
    generator's own rule (utils/bones.py compute_chain_x_axis, identical 3.6-5.2) falls back when |first.y x
    (last.tail - first.head)| < first.length / 100. Measured on an arm of two 0.3 m bones: bent 0.55 degrees the IK
    elbow did not bend at all when the hand was pulled in (0.001 m), bent 0.6 degrees it bent the intended way
    (0.163 m), the same on 3.6, 4.4 and 5.2; at 0 degrees generation throws 'Vector.angle(other): zero length vectors
    have no valid angle'. A leg's FK axis flips sign below the same threshold. The checks here use that exact rule.

THE BUILD DECIDES THE COLLECTION FORM: 3.6 has 32 boolean layers (armature.layers, bone.layers) with the generator's UI
rows in armature.rigify_layers; 4.x and 5.x have named, nested armature.collections. A call in the wrong form is refused
with the right one.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re

import bpy
from mathutils import Vector

from .ops_common import (remove_objects, MifOpError, simulation_guard, finite_floats, get_object, reject_unknown, require_editable, rnd,
                         selection_restore, selection_snapshot, take, take_bool, take_float, take_int,
                         write_rna_property)

_V4 = bpy.app.version >= (4, 0, 0)
# THE GENERATOR'S RESERVED COLLECTIONS (utils/layers.py SPECIAL_COLLECTIONS, 4.2-5.2) and 3.6's reserved layers
# (utils/layers.py ROOT_LAYER 28, DEF_LAYER 29, MCH_LAYER 30, ORG_LAYER 31).
_SPECIAL = ("Root", "DEF", "ORG", "MCH")
_LAYER_ROLE = {28: "Root", 29: "DEF", 30: "MCH", 31: "ORG"}
_NAME_LIMIT = 63
_TAIL = "NOTHING was changed."


# --------------------------------------------------------------------------------------------- the generator itself

def _rigify_on():
    return "rigify" in bpy.context.preferences.addons and hasattr(bpy.types.PoseBone, "rigify_type")


def _rigify_module():
    import addon_utils   # noqa: PLC0415 - Blender's own; absent where the add-on is imported without Blender (offline suites)
    for mod in addon_utils.modules():
        if mod.__name__ == "rigify":
            return mod
    return None


class _PrefsGuard:
    """Put preferences.is_dirty back when this op found it clean. Enabling the generator (and some of its own steps)
    marks the preferences dirty, and with Auto-Save Preferences on - the factory default - quitting would then write
    the generator into the user's saved preferences. Only the op's own dirtying is undone."""

    def __enter__(self):
        # getattr: a context with no preferences (the add-on imported without Blender, in the offline refusal suite)
        # has nothing to guard, and the op's own refusal must still be the one that answers
        prefs = getattr(bpy.context, "preferences", None)
        self.clean = prefs is not None and not getattr(prefs, "is_dirty", True)
        return self

    def __exit__(self, *exc):
        if self.clean and bpy.context.preferences.is_dirty:
            try:
                bpy.context.preferences.is_dirty = False
            except Exception:  # noqa: BLE001 - a read-only preferences block is not this op's failure
                pass
        return False


def ensure_rigify(params, tail=_TAIL):
    """Enable the generator for this session if it is off. Returns True when this call enabled it."""
    enable = take_bool(params, "enableRigify", default=True)
    if _rigify_on():
        return False
    if hasattr(bpy.types.PoseBone, "rigify_type"):
        raise MifOpError("the Rigify generator is HALF-REGISTERED in this session: its types exist but its preferences "
                         "entry does not, which is what addon_utils.enable('rigify', default_set=False) leaves (its "
                         "register() fails reading preferences.addons['rigify']; RigifyParameters then lacks the rig "
                         "types' parameters and generation throws). Disable and re-enable it in Preferences > Add-ons, "
                         "or restart Blender. %s" % tail)
    if not enable:
        raise MifOpError("the Rigify generator is off and enableRigify is false. Pass enableRigify true to enable it for "
                         "this session (the preferences are not saved). %s" % tail)
    if _rigify_module() is None:
        raise MifOpError("this Blender (%s) has no bundled 'rigify' add-on to enable. %s"
                         % (bpy.app.version_string, tail))
    prefs = bpy.context.preferences
    was_dirty = prefs.is_dirty
    errors = []
    import addon_utils   # noqa: PLC0415 - see _rigify_module
    addon_utils.enable("rigify", default_set=True, handle_error=lambda exc: errors.append(str(exc)))
    if not was_dirty:
        prefs.is_dirty = False
    if errors or not _rigify_on():
        raise MifOpError("enabling the Rigify generator failed: %s. %s" % ("; ".join(errors) or "it did not register",
                                                                          tail))
    return True


def _rig_types():
    from rigify import rig_lists  # noqa: PLC0415 - importable only once the generator is enabled
    return rig_lists.rigs


def _rig_type_names():
    return sorted(_rig_types())


def _metarig_names():
    """The bundled metarigs, read from the generator's own folder - usable BEFORE it is enabled, so a bad name is
    refused without enabling anything. The add operator is object.armature_<file stem>_metarig_add."""
    mod = _rigify_module()
    if mod is None:
        return []
    base = os.path.join(os.path.dirname(mod.__file__), "metarigs")
    names = []
    for root, _dirs, files in os.walk(base):
        for f in files:
            if f.endswith(".py") and not f.startswith(("_", ".")):
                names.append(f[:-3])
    return sorted(set(names))


def _ptype_kind(prop):
    out = {"name": prop.identifier, "kind": prop.type}
    if prop.type == "ENUM":
        out["items"] = [i.identifier for i in prop.enum_items]
        out["default"] = prop.default
    elif prop.type in ("BOOLEAN", "INT", "FLOAT"):
        n = getattr(prop, "array_length", 0)
        out["default"] = (list(prop.default_array) if n > 1 else
                          (round(prop.default, 6) if prop.type == "FLOAT" else prop.default))
        if n > 1:
            out["arrayLength"] = n
    elif prop.type == "STRING":
        out["default"] = prop.default
    elif prop.type == "COLLECTION":
        out["kind"] = "BONE_COLLECTIONS"
    return out


_DEFINES = {}


def type_parameters(rig_type):
    """The parameters a rig type defines: its add_parameters run against a recorder, the way the generator's own
    RigifyParameterValidator does at registration (rigify/__init__.py, 3.6 and 4.2-5.2), then each name described from
    the REGISTERED property on RigifyParameters. Every type's parameters share one namespace, so this is the only way to
    know which names belong to a type."""
    if rig_type in _DEFINES:
        return _DEFINES[rig_type]
    import rigify  # noqa: PLC0415
    entry = _rig_types().get(rig_type)
    if entry is None:
        return []
    module = entry["module"]
    cls = getattr(module, "Rig", None)
    rig_def = cls if hasattr(cls, "add_parameters") else module
    names = []

    class _Recorder:
        def __getattr__(self, name):
            return getattr(rigify.RigifyParameters, name)

        def __setattr__(self, name, value):
            if name not in names:
                names.append(name)

    if hasattr(rig_def, "add_parameters"):
        rig_def.add_parameters(_Recorder())
    props = bpy.types.PoseBone.bl_rna.properties["rigify_parameters"].fixed_type.properties
    out = [_ptype_kind(props[n]) for n in names if n in props]
    _DEFINES[rig_type] = out
    return out


# --------------------------------------------------------------------------------------------- shared readers

def _armature(params, keys=("object",), verb="changed"):
    obj = get_object(take(params, *keys, required=True, kind=str))
    if obj.type != "ARMATURE":
        raise MifOpError("'%s' is a %s, not an ARMATURE. %s" % (obj.name, obj.type, _TAIL))
    require_editable(obj, verb)
    return obj


def _object_mode(obj, op_name):
    mode = bpy.context.object.mode if bpy.context.object else "OBJECT"
    if mode != "OBJECT":
        raise MifOpError("Blender is in %s mode. %s switches modes itself and needs OBJECT mode to start from, or "
                         "whatever is being edited would be dropped. %s" % (mode, op_name, _TAIL))


def _activate(obj):
    vl = bpy.context.view_layer
    for o in vl.objects:
        if o.select_get():
            o.select_set(False)
    obj.hide_set(False)
    obj.select_set(True)
    vl.objects.active = obj


def is_generated(obj):
    return obj is not None and obj.type == "ARMATURE" and obj.data.get("rig_id") is not None


def _membership(data):
    """{bone name: [collection names]} in one pass over the collections (4.x and later)."""
    out = {}
    for c in data.collections_all:
        for b in c.bones:
            out.setdefault(b.name, []).append(c.name)
    return out


def _collections_of(bone, data, membership=None):
    if _V4:
        return list((membership if membership is not None else _membership(data)).get(bone.name, []))
    return [i for i, v in enumerate(bone.layers) if v]


def collection_rows(data):
    """[{name, parent, boneCount, visible, solo, uiRow...}] on 4.x+; the used layers on 3.6."""
    if _V4:
        rows = []
        for c in data.collections_all:
            row = {"name": c.name, "parent": c.parent.name if c.parent else None, "boneCount": len(c.bones),
                   "visible": bool(c.is_visible), "solo": bool(getattr(c, "is_solo", False))}
            if hasattr(c, "rigify_ui_row"):
                row["uiRow"] = int(c.rigify_ui_row)
                row["uiTitle"] = c.rigify_ui_title
                if hasattr(c, "rigify_sel_set"):
                    row["selectionSet"] = bool(c.rigify_sel_set)
                if hasattr(c, "rigify_color_set_name"):
                    row["colorSet"] = c.rigify_color_set_name or None
            rows.append(row)
        return rows
    counts = [0] * 32
    for b in data.bones:
        for i, v in enumerate(b.layers):
            if v:
                counts[i] += 1
    rl = getattr(data, "rigify_layers", None)
    rows = []
    for i in range(32):
        row = {"layer": i, "boneCount": counts[i], "visible": bool(data.layers[i])}
        if i in _LAYER_ROLE:
            row["reservedFor"] = _LAYER_ROLE[i]
        if rl is not None and i < len(rl):
            row.update(uiTitle=rl[i].name, uiRow=int(rl[i].row), selectionSet=bool(rl[i].selset),
                       colorSet=int(rl[i].group))
        if counts[i] or row.get("uiTitle") or (row["visible"] and i < 28):
            rows.append(row)
    return rows


def rig_type_map(obj):
    if obj.pose is None or not hasattr(bpy.types.PoseBone, "rigify_type"):
        return {}
    return {pb.name: pb.rigify_type for pb in obj.pose.bones if pb.rigify_type}


def rigify_bone_fields(obj, rows):
    """list_bones' per-bone additions: the bone's collections (layer indices on 3.6) and its rig type."""
    data = obj.data
    types = rig_type_map(obj) if hasattr(bpy.types.PoseBone, "rigify_type") else None
    member = _membership(data) if _V4 else None
    for row in rows:
        bone = data.bones.get(row.get("name"))
        if bone is None:
            continue
        row["collections" if _V4 else "layers"] = _collections_of(bone, data, member)
        row["rigifyType"] = (types.get(bone.name, "") if types is not None else None)
    return rows


def _chain(data, name):
    """The bone and its connected children while each has exactly one - the generator's connected_children_names."""
    bone = data.bones[name]
    out = [bone]
    while True:
        conn = [c for c in bone.children if c.use_connect]
        if len(conn) != 1:
            return out
        bone = conn[0]
        out.append(bone)


# The rig types that orient their chain about the bend of its bones, and which bones and which parameter decide it -
# rigs/limbs/arm.py, leg.py, paw.py (front_paw, rear_paw and super_limb build on these), super_finger.py,
# simple_tentacle.py, identical 3.6-5.2.
_BEND_RULE = {
    "limbs.arm": ("rotation_axis", 2), "limbs.leg": ("rotation_axis", 2), "limbs.paw": ("rotation_axis", 2),
    "limbs.front_paw": ("rotation_axis", 2), "limbs.rear_paw": ("rotation_axis", 2),
    "limbs.super_limb": ("rotation_axis", 2),
    "limbs.super_finger": ("primary_rotation_axis", None), "limbs.simple_tentacle": ("roll_alignment", None),
}


def _disjoint_chains(obj, types):
    """A chain rig told to connect to its parent rig (rigify_parameters.connect_chain - a head or a tail on a spine) whose
    first bone sits neither at the parent chain's last tail nor at its first head. The generator's own rule
    (rigs/chain_rigs.py ConnectingChainRig.initialize, 3.6 :233-240, 4.2-5.2 :236-243; is_same_position and
    is_connected_position in utils/bones.py, a 1e-5 distance) - it raises "Cannot connect chain - bone position is
    disjoint." only AFTER the rig object exists, leaving a partial rig (settled 2026-10-18 on 3.6-5.2: a human metarig with
    spine.004 disconnected and its head moved 5 cm left a 159-bone rig behind)."""
    data = obj.data
    chains = {t: [b.name for b in _chain(data, t)] for t in types}
    out = []
    for name in types:
        pb = obj.pose.bones.get(name)
        if pb is None or not getattr(pb.rigify_parameters, "connect_chain", False):
            continue
        parent_orgs = None
        anc = data.bones[name].parent
        while anc is not None and parent_orgs is None:
            parent_orgs = next((ch for t, ch in chains.items() if t != name and anc.name in ch), None)
            anc = anc.parent
        if not parent_orgs:
            continue
        head = data.bones[name].head_local
        direct = (data.bones[parent_orgs[-1]].tail_local - head).length < 1e-5
        reverse = (data.bones[parent_orgs[0]].head_local - head).length < 1e-5
        if not (direct or reverse):
            gap = (data.bones[parent_orgs[-1]].tail_local - head).length
            out.append({"check": "disjointChain", "bone": name,
                        "text": "bone '%s' (%s) connects its chain to the parent rig's chain %s, but its head is %.4f m "
                                "from '%s''s tail (the generator allows 1e-5). Generation throws \"Bone '%s': Cannot "
                                "connect chain - bone position is disjoint.\" and leaves a partial rig. Move its head "
                                "onto that tail (connect it), or turn its rigify parameter connect_chain off with "
                                "set_rigify_type" % (name, types[name], parent_orgs, gap, parent_orgs[-1], name)})
    return out


def _bend(data, first, rule_bones):
    chain = _chain(data, first)
    if rule_bones is not None:
        chain = chain[:rule_bones]
    if len(chain) < 2:
        return None
    a, last = chain[0], chain[-1]
    y1 = (a.tail_local - a.head_local)
    if y1.length == 0:
        return None
    y1 = y1.normalized()
    rot = y1.cross(last.tail_local - a.head_local)
    y2 = (chain[1].tail_local - chain[1].head_local)
    bend = math.degrees(y1.angle(y2)) if y2.length > 0 else 0.0
    return {"bone": a.name, "chain": [b.name for b in chain], "bendDeg": round(bend, 4),
            "planeRatio": round(rot.length / a.length, 6) if a.length else 0.0,
            "fallsBackToRoll": rot.length < a.length / 100.0}


def generate_checks(obj, allow_straight=False):
    """What stops this armature generating, in the order generation would meet it. Returns (problems, bends)."""
    problems = []
    data = obj.data
    if is_generated(obj):
        problems.append({"check": "generatedRig", "text": "'%s' is itself a generated rig (it holds rig_id %s); generate "
                                                          "from its metarig" % (obj.name, data.get("rig_id"))})
    loc, rot, scale = obj.matrix_world.decompose()
    if any(abs(s - 1.0) > 1e-5 for s in scale) or rot.angle > 1e-5:
        problems.append({"check": "transform",
                         "text": "'%s' has world scale %s and rotation %.4f degrees; the generator copies bone positions "
                                 "without the object's scale, so the rig comes out the wrong size with no message. Bake "
                                 "it into the bones first: a new metarig takes height or scale on create_armature (baked, "
                                 "object left at scale 1), and an existing one takes apply_transform {object, scale: true, "
                                 "rotation: true} (it applies an armature with the meshes it deforms)"
                                 % (obj.name, rnd(scale, 4), math.degrees(rot.angle))})
    types = rig_type_map(obj) if hasattr(bpy.types.PoseBone, "rigify_type") else {}
    if not types:
        problems.append({"check": "noRigType", "text": "no bone of '%s' has a rig type; set one with set_rigify_type"
                                                       % obj.name})
    if _V4:
        rows = [c for c in data.collections_all if getattr(c, "rigify_ui_row", 0) > 0 and c.name not in _SPECIAL]
        if not rows:
            problems.append({"check": "uiRow",
                             "text": "no bone collection other than Root, DEF, ORG and MCH has a UI row above 0 - the "
                                     "generator refuses with 'No bone collections have UI buttons assigned - please "
                                     "check the Bone Collections UI sub-panel'. set_bone_collections SET_UI uiRow 1 on "
                                     "one"})
    for name, typ in types.items():
        if typ == "faces.super_face" and not data.bones[name].children:
            problems.append({"check": "leftoverFace",
                             "text": "bone '%s' is typed faces.super_face and has no child bones - its face was removed "
                                     "and the bone left behind. Generation throws \"'bpy_prop_collection[key]: key "
                                     "\"ORG-nose\" not found'\" and leaves a partial rig. Remove the bone or clear its "
                                     "type" % name})
    problems.extend(_disjoint_chains(obj, types))
    bends = []
    for name, typ in types.items():
        rule = _BEND_RULE.get(typ)
        if rule is None:
            continue
        pb = obj.pose.bones[name]
        axis = getattr(pb.rigify_parameters, rule[0], "automatic")
        if axis != "automatic":
            continue
        b = _bend(data, name, rule[1])
        if b is None:
            continue
        b["rigType"] = typ
        bends.append(b)
        if b["fallsBackToRoll"] and not allow_straight:
            problems.append({"check": "straightChain",
                             "text": "%s chain from '%s' is bent %.4f degrees at its first joint (plane ratio %.6f, "
                                     "below the generator's 0.01): it is oriented by the bone's ROLL instead of its "
                                     "bend, so an IK joint does not bend (measured: 0.55 degrees on two 0.3 m bones) "
                                     "and FK turns about the roll axis; at 0 generation throws. Bend the middle joint "
                                     "slightly the way it should fold, or pass allowStraightChains true when the roll "
                                     "is set on purpose" % (typ, name, b["bendDeg"], b["planeRatio"])})
    return problems, bends


def _move_edit_bones(ebs, names, offset, factor):
    """Scale edit bones by factor about the armature origin, then move them by offset - every end computed BEFORE any
    is written. A connected child's head IS its parent's tail, so writing bone by bone (EditBone.transform included)
    moves a shared joint twice: measured on 5.2.1, the human metarig scaled 3x that way came out 17.3 m tall, not 5.94.
    The display sizes (envelope, B-Bone) scale with it."""
    plan = [(ebs[n], ebs[n].head * factor + offset, ebs[n].tail * factor + offset, ebs[n].roll) for n in names]
    for eb, head, tail, roll in plan:
        eb.head = head
        eb.tail = tail
        eb.roll = roll
        eb.head_radius *= factor
        eb.tail_radius *= factor
        eb.envelope_distance *= factor
        eb.bbone_x *= factor
        eb.bbone_z *= factor


def _height(obj):
    zs = [v for b in obj.data.bones for v in (b.head_local.z, b.tail_local.z)]
    return (max(zs) - min(zs)) if zs else 0.0


def _ready(obj):
    problems, bends = generate_checks(obj)
    return {"generateReady": not problems, "notReadyBecause": [p["text"] for p in problems] or None,
            "limbBends": bends or None}


# --------------------------------------------------------------------------------------------- bone collections

_COLL_ACTIONS = ("LIST", "CREATE", "RENAME", "REMOVE", "MOVE", "ASSIGN", "UNASSIGN", "SET_VISIBLE", "SET_SOLO",
                 "SET_PARENT", "SET_UI")
_COLL_KEYS = ("object", "action", "collection", "bones", "namePrefix", "parent", "newName", "index", "visible",
              "solo", "uiRow", "uiTitle", "selectionSet", "colorSet", "enableRigify")
_COLL_USES = {  # the keys each action reads beyond object, action and collection
    "LIST": (), "CREATE": ("parent", "bones", "namePrefix", "uiRow", "uiTitle", "selectionSet", "colorSet",
                           "enableRigify"),
    "RENAME": ("newName",), "REMOVE": (), "MOVE": ("index",), "ASSIGN": ("bones", "namePrefix"),
    "UNASSIGN": ("bones", "namePrefix"), "SET_VISIBLE": ("visible",), "SET_SOLO": ("solo",), "SET_PARENT": ("parent",),
    "SET_UI": ("uiRow", "uiTitle", "selectionSet", "colorSet", "enableRigify"),
}
_UI_KEYS = ("uiRow", "uiTitle", "selectionSet", "colorSet")
_V36_REFUSED = {
    "CREATE": "Blender %s has 32 fixed layers, not named collections - there is nothing to create; ASSIGN bones to a "
              "layer 0-31 and give it a button with SET_UI uiTitle",
    "RENAME": "3.6 layers have no names; the generator's button label is SET_UI uiTitle",
    "REMOVE": "3.6 layers are fixed; UNASSIGN the bones instead",
    "MOVE": "3.6 layers are fixed in place",
    "SET_SOLO": "3.6 layers have no solo; SET_VISIBLE the other layers off",
    "SET_PARENT": "3.6 layers do not nest",
}


def _pick_bones(params, data, required):
    names = params.get("bones")
    prefix = params.get("namePrefix")
    if names is not None and prefix is not None:
        raise MifOpError("pass bones OR namePrefix. %s" % _TAIL)
    if names is None and prefix is None:
        if required:
            raise MifOpError("this action needs bones [names] or namePrefix. %s" % _TAIL)
        return []
    if prefix is not None:
        if not isinstance(prefix, str) or not prefix:
            raise MifOpError("namePrefix must be non-empty text. %s" % _TAIL)
        picked = [b for b in data.bones if b.name.startswith(prefix)]
        if not picked:
            raise MifOpError("no bone of '%s' starts with '%s'. %s" % (data.name, prefix, _TAIL))
        return picked
    if not isinstance(names, (list, tuple)) or not names or not all(isinstance(n, str) for n in names):
        raise MifOpError("bones must be a non-empty list of bone names. %s" % _TAIL)
    have = set(data.bones.keys())
    missing = [n for n in names if n not in have]
    if missing:
        raise MifOpError("no bone(s) %s on '%s' (it has %d; list_bones names them). %s"
                         % (missing[:10], data.name, len(data.bones), _TAIL))
    return [data.bones[n] for n in dict.fromkeys(names)]


def _ui_values(params, data):
    """uiRow, uiTitle, selectionSet, colorSet parsed and checked - before anything is written."""
    out = {}
    if params.get("uiRow") is not None:
        row = take_int(params, "uiRow")
        lo = 0 if _V4 else 1
        if not lo <= row <= (2147483647 if _V4 else 32):
            raise MifOpError("uiRow must be %s, got %d (%s). %s"
                             % ("0 or more (0 means no button)" if _V4 else "1-32", row,
                                "the generator's own range on this build", _TAIL))
        out["uiRow"] = row
    if params.get("uiTitle") is not None:
        out["uiTitle"] = take(params, "uiTitle", kind=str)
    if params.get("selectionSet") is not None:
        out["selectionSet"] = take_bool(params, "selectionSet")
    if params.get("colorSet") is not None:
        cs = params.get("colorSet")
        sets = [c.name for c in getattr(data, "rigify_colors", [])]
        if _V4:
            if not isinstance(cs, str):
                raise MifOpError("colorSet takes a color set NAME (or '' for none); this armature has %s. %s"
                                 % (sets or "none", _TAIL))
            if cs and cs not in sets:
                raise MifOpError("no color set '%s' on '%s'. It has: %s. %s" % (cs, data.name, sets or "none", _TAIL))
        else:
            if isinstance(cs, bool) or not isinstance(cs, int) or not 0 <= cs <= 32:
                raise MifOpError("colorSet on 3.6 is the bone group number 0-32 (rigify_layers group), got %r. %s"
                                 % (cs, _TAIL))
        out["colorSet"] = cs
    return out


def _write_ui(target, ui, data):
    if _V4:
        if "uiRow" in ui:
            target.rigify_ui_row = ui["uiRow"]
        if "uiTitle" in ui:
            target.rigify_ui_title = ui["uiTitle"]
        if "selectionSet" in ui:
            target.rigify_sel_set = ui["selectionSet"]
        if "colorSet" in ui:
            target.rigify_color_set_name = ui["colorSet"]
        return
    rl = data.rigify_layers
    while len(rl) <= target:
        rl.add()
    item = rl[target]
    if "uiRow" in ui:
        item.row = ui["uiRow"]
    if "uiTitle" in ui:
        item.name = ui["uiTitle"]
    if "selectionSet" in ui:
        item.selset = ui["selectionSet"]
    if "colorSet" in ui:
        item.group = ui["colorSet"]


def op_set_bone_collections(params):
    """List, create, rename, remove, move, assign, show, hide, solo and nest an armature's bone collections, and give
    them the generator's UI rows (3.6: its 32 layers).

    action LIST (default) | CREATE | RENAME (newName) | REMOVE | MOVE (index among its siblings) | ASSIGN | UNASSIGN
    (bones [names] or namePrefix) | SET_VISIBLE (visible, default true) | SET_SOLO (solo, default true) | SET_PARENT
    (parent, '' for the top level) | SET_UI (uiRow, uiTitle, selectionSet, colorSet - the Rigify button; CREATE takes
    them too). collection is a NAME on 4.x and later and a layer index 0-31 on 3.6, where layers have no names, no
    nesting and no solo - those actions are refused there with what does exist. A key the action does not read is
    refused. The UI keys need the Rigify generator, which is enabled for the session when off (enableRigify false
    refuses instead). A bone may be in several collections; ASSIGN reports the ones already members.

    Reply: collections [{name, parent, boneCount, visible, solo, uiRow, uiTitle, selectionSet, colorSet}] (3.6: the
    used layers), and on a GENERATED rig generated true with the warning that a regenerate erases the edit - the
    generator removes the target rig's collections before rebuilding, so the edit belongs on the metarig.
    """
    reject_unknown(params, _COLL_KEYS, "set_bone_collections")
    with _PrefsGuard():
        obj = _armature(params)
        data = obj.data
        action = str(take(params, "action", default="LIST", kind=str)).upper()
        if action not in _COLL_ACTIONS:
            raise MifOpError("action must be one of %s, got %r. %s" % (", ".join(_COLL_ACTIONS), action, _TAIL))
        extra = [k for k in params if k not in ("object", "action", "collection") + _COLL_USES[action]
                 and params[k] is not None]
        if extra:
            raise MifOpError("%s does not read %s - it takes %s. %s"
                             % (action, ", ".join(sorted(extra)),
                                ", ".join(("collection",) + _COLL_USES[action]) if action != "LIST" else "nothing more",
                                _TAIL))
        if obj.mode == "EDIT":
            raise MifOpError("'%s' is in edit mode, where bone membership is not written back until it is left. Leave "
                             "edit mode first. %s" % (obj.name, _TAIL))
        if not _V4 and action in _V36_REFUSED:
            raise MifOpError("%s: %s. %s" % (action, _V36_REFUSED[action] % bpy.app.version_string
                                             if "%s" in _V36_REFUSED[action] else _V36_REFUSED[action], _TAIL))
        enabled_now = False
        if any(params.get(k) is not None for k in _UI_KEYS):
            enabled_now = ensure_rigify(params)
        raw = params.get("collection")
        target = None
        if action != "LIST" or raw is not None:
            if raw is None:
                raise MifOpError("%s needs collection (%s). %s"
                                 % (action, "a name" if _V4 else "a layer index 0-31", _TAIL))
            if _V4:
                if isinstance(raw, bool) or not isinstance(raw, str) or not raw:
                    raise MifOpError("collection is a NAME on Blender %s (bone collections); a layer index is the "
                                     "3.6 form. Got %r. %s" % (bpy.app.version_string, raw, _TAIL))
                if len(raw) > _NAME_LIMIT:
                    raise MifOpError("collection name is %d characters; Blender keeps %d. %s"
                                     % (len(raw), _NAME_LIMIT, _TAIL))
                target = data.collections_all.get(raw)
                if action == "CREATE" and target is not None:
                    raise MifOpError("'%s' already has a collection '%s'. %s" % (obj.name, raw, _TAIL))
                if action != "CREATE" and target is None:
                    raise MifOpError("no collection '%s' on '%s'. It has: %s. %s"
                                     % (raw, obj.name, [c.name for c in data.collections_all] or "none", _TAIL))
            else:
                if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw <= 31:
                    raise MifOpError("collection on Blender %s is a LAYER INDEX 0-31 (3.6 has no named bone "
                                     "collections); got %r. %s" % (bpy.app.version_string, raw, _TAIL))
                target = raw
        ui = _ui_values(params, data)
        if action == "SET_UI" and not ui:
            raise MifOpError("SET_UI needs at least one of uiRow, uiTitle, selectionSet, colorSet. %s" % _TAIL)
        if not _V4 and ui and target >= 28:
            raise MifOpError("layer %d is reserved by the generator for %s and gets no button. %s"
                             % (target, _LAYER_ROLE[target], _TAIL))
        bones = _pick_bones(params, data, action in ("ASSIGN", "UNASSIGN")) \
            if action in ("CREATE", "ASSIGN", "UNASSIGN") else []
        parent = None
        if action in ("CREATE", "SET_PARENT") and params.get("parent"):
            parent = data.collections_all.get(take(params, "parent", kind=str))
            if parent is None:
                raise MifOpError("no collection '%s' on '%s' to nest under. %s" % (params["parent"], obj.name, _TAIL))
            if action == "SET_PARENT":
                walk = parent
                while walk is not None:
                    if walk == target:
                        raise MifOpError("'%s' is '%s' or inside it, so nesting would make a loop. %s"
                                         % (parent.name, target.name, _TAIL))
                    walk = walk.parent
        new_name = None
        if action == "RENAME":
            new_name = take(params, "newName", required=True, kind=str)
            if not new_name or len(new_name) > _NAME_LIMIT or new_name in data.collections_all:
                raise MifOpError("newName must be 1-%d characters and not already a collection on '%s'. %s"
                                 % (_NAME_LIMIT, obj.name, _TAIL))
            if target.name in _SPECIAL:
                raise MifOpError("'%s' is a name the generator reserves; renaming it breaks what it finds by name. %s"
                                 % (target.name, _TAIL))
        index = None
        if action == "MOVE":
            index = take_int(params, "index", required=True)
            siblings = list(target.parent.children if target.parent else data.collections)
            if not 0 <= index < len(siblings):
                raise MifOpError("index must be 0-%d (its place among the %d collections beside it). %s"
                                 % (len(siblings) - 1, len(siblings), _TAIL))
        visible = take_bool(params, "visible", default=True) if action == "SET_VISIBLE" else None
        solo = take_bool(params, "solo", default=True) if action == "SET_SOLO" else None
        if action == "UNASSIGN" and not _V4:
            stranded = [b.name for b in bones if [i for i, v in enumerate(b.layers) if v] == [target]]
            if stranded:
                raise MifOpError("bone(s) %s are on layer %d only, and a 3.6 bone must stay on at least one layer. "
                                 "ASSIGN them to another first. %s" % (stranded[:10], target, _TAIL))
        if action == "SET_VISIBLE" and not _V4 and not visible:
            if [i for i, v in enumerate(data.layers) if v] == [target]:
                raise MifOpError("layer %d is the only visible layer, and 3.6 keeps at least one visible. %s"
                                 % (target, _TAIL))

        out = {"object": obj.name, "action": action}
        already = []
        # ---- writes
        if action == "CREATE":
            target = data.collections.new(raw, parent=parent) if parent is not None else data.collections.new(raw)
            out["created"] = target.name
            for b in bones:
                target.assign(b)
            if ui:
                _write_ui(target, ui, data)
        elif action == "RENAME":
            out["renamedFrom"] = target.name
            target.name = new_name
        elif action == "REMOVE":
            out["removed"] = target.name
            data.collections.remove(target)
            target = None
        elif action == "MOVE":
            target.child_number = index
        elif action in ("ASSIGN", "UNASSIGN"):
            members = {b.name for b in target.bones} if _V4 else None
            for b in bones:
                if _V4:
                    member = b.name in members
                    if action == "ASSIGN":
                        if member:
                            already.append(b.name)
                        else:
                            target.assign(b)
                    elif member:
                        target.unassign(b)
                    else:
                        already.append(b.name)
                else:
                    member = bool(b.layers[target])
                    if (action == "ASSIGN") == member:
                        already.append(b.name)
                    else:
                        b.layers[target] = action == "ASSIGN"
            out["alreadyMembers" if action == "ASSIGN" else "notMembers"] = already
            out["bones"] = len(bones)
        elif action == "SET_VISIBLE":
            if _V4:
                target.is_visible = visible
            else:
                data.layers[target] = visible
        elif action == "SET_SOLO":
            target.is_solo = solo
        elif action == "SET_PARENT":
            target.parent = parent
        elif action == "SET_UI":
            _write_ui(target, ui, data)
        out["collections" if _V4 else "layers"] = collection_rows(data)
        if target is not None and _V4:
            out["collection"] = next((r for r in out["collections"] if r["name"] == target.name), None)
        elif target is not None:
            out["collection"] = next((r for r in out["layers"] if r["layer"] == target),
                                     {"layer": target, "boneCount": 0})
        if is_generated(obj):
            out["generated"] = True
            out["warning"] = ("'%s' is a GENERATED rig: regenerate erases this - the generator removes every bone "
                              "collection of the target rig before rebuilding it from the metarig. Make the edit on the "
                              "metarig." % obj.name)
        out["rigifyEnabledNow"] = enabled_now
        return out


# --------------------------------------------------------------------------------------------- metarig and sample

_METARIG_ONLY = {"bones": "a metarig brings its own bones; add more afterwards with add_bones",
                 "rotation": "a metarig is left at rotation 0 so generation never meets an unapplied rotation",
                 "displayType": "a metarig keeps the generator's own display",
                 "showInFront": "a metarig keeps the generator's own display"}


def create_metarig(params):
    """create_armature's metarig form: a bundled Rigify metarig at a location, sized by height (meters) or scale,
    the size BAKED into the bones and the object left at scale 1 and rotation 0."""
    with _PrefsGuard():
        bad = [k for k in _METARIG_ONLY if params.get(k) is not None]
        if bad:
            raise MifOpError("%s: %s. NOTHING was created." % (bad[0], _METARIG_ONLY[bad[0]]))
        kind = take(params, "metarig", required=True, kind=str)
        known = _metarig_names()
        if not known:
            raise MifOpError("this Blender (%s) has no bundled Rigify generator, so there are no metarigs. NOTHING was "
                             "created." % bpy.app.version_string)
        if kind not in known:
            raise MifOpError("metarig '%s' is not one this Blender ships. It has: %s. NOTHING was created."
                             % (kind, ", ".join(known)))
        height = take_float(params, "height", default=None)
        scale = take_float(params, "scale", default=None)
        if height is not None and scale is not None:
            raise MifOpError("pass height OR scale - both size the metarig. NOTHING was created.")
        for key, v in (("height", height), ("scale", scale)):
            if v is not None and v <= 0:
                raise MifOpError("%s must be above 0, got %r. NOTHING was created." % (key, v))
        loc = params.get("location")
        if loc is not None:
            if not isinstance(loc, (list, tuple)) or len(loc) != 3:
                raise MifOpError("location must be [x,y,z], got %r. NOTHING was created." % (loc,))
            loc = Vector(finite_floats(list(loc), "location"))
        else:
            loc = Vector((0.0, 0.0, 0.0))
        cname = take(params, "collection", kind=str)
        coll = bpy.context.scene.collection
        if cname:
            coll = bpy.data.collections.get(cname)
            if coll is None:
                raise MifOpError("no collection named '%s'. Make one with create_collection. NOTHING was created."
                                 % cname)
        wanted = take(params, "name", kind=str)
        if wanted is not None and (not wanted or len(wanted) > _NAME_LIMIT):
            raise MifOpError("name must be 1-%d characters. NOTHING was created." % _NAME_LIMIT)
        _object_mode(None, "create_armature")
        enabled_now = ensure_rigify(params, "NOTHING was created.")
        op = getattr(bpy.ops.object, "armature_%s_metarig_add" % kind, None)
        if op is None:
            raise MifOpError("the generator registered no object.armature_%s_metarig_add. No object was made, and the "
                             "generator stays enabled for this session." % kind)
        snap = selection_snapshot()
        before = set(bpy.data.objects)
        obj = None
        try:
            op()
            made = [o for o in bpy.data.objects if o not in before]
            obj = next((o for o in made if o.type == "ARMATURE"), None)
            if obj is None:
                raise MifOpError("object.armature_%s_metarig_add made no armature, and the generator stays "
                                 "enabled for this session." % kind)
            natural = _height(obj)
            factor = scale if scale is not None else (height / natural if height is not None and natural > 0 else 1.0)
            if abs(factor - 1.0) > 1e-9:
                _activate(obj)
                bpy.ops.object.mode_set(mode="EDIT")
                _move_edit_bones(obj.data.edit_bones, [eb.name for eb in obj.data.edit_bones], Vector(), factor)
                bpy.ops.object.mode_set(mode="OBJECT")
            obj.location = loc
            obj.rotation_mode = "XYZ"
            obj.rotation_euler = (0.0, 0.0, 0.0)
            obj.scale = (1.0, 1.0, 1.0)
            if wanted:
                obj.name = wanted
                obj.data.name = wanted
            if cname:
                for c in list(obj.users_collection):
                    c.objects.unlink(obj)
                coll.objects.link(obj)
        except Exception:
            if obj is not None and obj.name in bpy.data.objects:
                if bpy.context.object is obj and obj.mode != "OBJECT":
                    bpy.ops.object.mode_set(mode="OBJECT")
                data = obj.data
                remove_objects([obj])
                if data.users == 0:
                    bpy.data.armatures.remove(data)
            raise
        finally:
            selection_restore(snap)
        out = {"ok": True, "name": obj.name, "object": obj.name, "type": "ARMATURE", "metarig": kind,
               "requestedName": wanted, "nameWasSuffixed": bool(wanted and obj.name != wanted),
               "collection": obj.users_collection[0].name if obj.users_collection else None,
               "location": rnd(obj.location), "rotation": rnd(obj.rotation_euler), "scale": rnd(obj.scale),
               "boneCount": len(obj.data.bones), "heightM": round(_height(obj), 6),
               "naturalHeightM": round(natural, 6), "scaleBaked": round(factor, 6),
               "rigTypes": rig_type_map(obj), "collections" if _V4 else "layers": collection_rows(obj.data),
               "rigifyEnabledNow": enabled_now, "modeAfter": bpy.context.object.mode if bpy.context.object else "OBJECT"}
        out.update(_ready(obj))
        return out


_SAMPLE_KEYS = ("rigType", "origin", "scale")


def add_rig_sample(params):
    """add_bones' sample form: one rig type's sample bones, through the generator's armature.metarig_sample_add in
    edit mode, moved to origin and scaled about it, with object mode restored."""
    with _PrefsGuard():
        if params.get("bones") is not None:
            raise MifOpError("pass bones OR sample - a sample makes its own bones. %s" % _TAIL)
        obj = _armature(params, ("object", "name", "armature"))
        sample = params.get("sample")
        if not isinstance(sample, dict):
            raise MifOpError("sample must be {rigType, origin [x,y,z], scale}, got %r. %s" % (sample, _TAIL))
        bad = sorted(k for k in sample if k not in _SAMPLE_KEYS)
        if bad:
            raise MifOpError("sample has %s, which it does not take - it takes %s. %s"
                             % (", ".join(bad), ", ".join(_SAMPLE_KEYS), _TAIL))
        rig_type = take(sample, "rigType", required=True, kind=str)
        origin = sample.get("origin")
        if origin is not None:
            if not isinstance(origin, (list, tuple)) or len(origin) != 3:
                raise MifOpError("sample.origin must be [x,y,z] in the armature's space, got %r. %s" % (origin, _TAIL))
            origin = Vector(finite_floats(list(origin), "sample.origin"))
        else:
            origin = Vector((0.0, 0.0, 0.0))
        s = take_float(sample, "scale", default=1.0)
        if s <= 0:
            raise MifOpError("sample.scale must be above 0, got %r. %s" % (s, _TAIL))
        _object_mode(obj, "add_bones")
        enabled_now = ensure_rigify(params)
        rigs = _rig_types()
        if rig_type not in rigs:
            raise MifOpError("'%s' is not a rig type of this Blender's generator. It has: %s. %s"
                             % (rig_type, ", ".join(sorted(rigs)), _TAIL))
        if not hasattr(rigs[rig_type]["module"], "create_sample"):
            has = sorted(k for k, v in rigs.items() if hasattr(v["module"], "create_sample"))
            raise MifOpError("rig type '%s' has no sample. Types with one: %s. %s" % (rig_type, ", ".join(has), _TAIL))
        before = {b.name for b in obj.data.bones}
        snap = selection_snapshot()
        made = []
        try:
            _activate(obj)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.armature.metarig_sample_add(metarig_type=rig_type)
            ebs = obj.data.edit_bones
            made = [eb.name for eb in ebs if eb.name not in before]
            _move_edit_bones(ebs, made, origin, s)
        finally:
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                pass
            selection_restore(snap)
        mode_after = bpy.context.object.mode if bpy.context.object else "OBJECT"
        data = obj.data
        have = set(data.bones.keys())
        landed = [n for n in made if n in have]
        types = rig_type_map(obj)
        member = _membership(data) if _V4 else None
        out = {"ok": True, "object": obj.name, "sample": rig_type, "added": landed,
               "suffixed": [n for n in landed if re.search(r"\.\d{3}$", n) and n.rsplit(".", 1)[0] in before] or None,
               "boneCountBefore": len(before), "boneCountAfter": len(data.bones),
               "bones": [{"name": n, "head": rnd(data.bones[n].head_local), "tail": rnd(data.bones[n].tail_local),
                          "parent": data.bones[n].parent.name if data.bones[n].parent else None,
                          "rigType": types.get(n, ""), "collections" if _V4 else "layers":
                              _collections_of(data.bones[n], data, member)} for n in landed],
               "rigTypes": types, "modeAfter": mode_after, "rigifyEnabledNow": enabled_now,
               "note": ("the sample's bones are in no bone collection; put them in one with a UI row "
                        "(set_bone_collections CREATE ... uiRow 1) before generating, and parent the sample's first "
                        "bone with set_bone_properties" if _V4 else
                        "the sample's bones are on layer 0; parent its first bone with set_bone_properties")}
        out.update(_ready(obj))
        return out


# --------------------------------------------------------------------------------------------- rig types and parameters

_TYPE_KEYS = ("object", "bones", "chainFrom", "type", "parameters", "copyTo", "enableRigify")
_COPY_KEYS = ("bones", "chainChildren", "mirror")


def _param_snapshot(pb, names):
    snap = {"__type": pb.rigify_type}
    p = pb.rigify_parameters
    for n in names:
        val = getattr(p, n, None)
        if hasattr(val, "__len__") and not isinstance(val, str):
            if hasattr(val, "add"):
                snap[n] = ("COLL", [(r.uid, r.get("name")) for r in val])
            else:
                snap[n] = list(val)
        else:
            snap[n] = val
    return snap


def _param_restore(pb, snap):
    pb.rigify_type = snap["__type"]
    p = pb.rigify_parameters
    for n, v in snap.items():
        if n == "__type":
            continue
        try:
            if isinstance(v, tuple) and v and v[0] == "COLL":
                coll = getattr(p, n)
                coll.clear()
                for uid, nm in v[1]:
                    r = coll.add()
                    r.uid = uid
                    if nm is not None:
                        r["name"] = nm
            else:
                setattr(p, n, v)
        except Exception:  # noqa: BLE001 - restoring best effort; the refusal says what failed
            pass


def _param_read(pb, defines):
    p = pb.rigify_parameters
    out = {}
    for d in defines:
        v = getattr(p, d["name"], None)
        if d["kind"] == "BONE_COLLECTIONS":
            out[d["name"]] = [r.name for r in v] if v is not None else []
        elif hasattr(v, "__len__") and not isinstance(v, str):
            out[d["name"]] = [round(x, 6) if isinstance(x, float) else x for x in v]
        else:
            out[d["name"]] = round(v, 6) if isinstance(v, float) else v
    return out


def _write_param(pb, d, value, obj):
    p = pb.rigify_parameters
    name = d["name"]
    if d["kind"] == "BONE_COLLECTIONS":
        if not _V4:
            raise MifOpError("'%s' is a 4.x collection reference and this is 3.6. %s" % (name, _TAIL))
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
            raise MifOpError("'%s' takes a list of bone collection NAMES, got %r. %s" % (name, value, _TAIL))
        found = []
        for v in value:
            c = obj.data.collections_all.get(v)
            if c is None:
                raise MifOpError("'%s' names collection '%s', which '%s' does not have. It has: %s. %s"
                                 % (name, v, obj.name, [c.name for c in obj.data.collections_all], _TAIL))
            found.append(c)
        coll = getattr(p, name)
        coll.clear()
        for c in found:
            coll.add().set_collection(c)
        return [r.name for r in coll]
    return write_rna_property(p, name, value, _TAIL, "bone '%s' rig parameters" % pb.name, "rig parameter",
                              frozenset())


def op_set_rigify_type(params):
    """Write a metarig bone's Rigify rig type and its parameters - to bones, to a chain, and copied to other bones or
    the mirrored side.

    Targets: bones [names] or chainFrom (the bone and its connected children while each has one - the generator's own
    chain). type: a rig type of THIS build's generator ('' clears it); omitted keeps each bone's type, so parameters
    alone can be written. parameters {name: value}: only names the type defines - every type's parameters share one
    namespace, so a name of another type would be accepted silently, and is refused here naming the type's own. A
    bone-collection reference parameter (4.x, e.g. fk_coll_refs) takes collection names. copyTo then copies the type and
    every parameter it defines from the (first) target to: bones [names], chainChildren '<bone>' (every direct child of
    that bone - the first bone of each chain hanging off it), or mirror true (each target's other-side name, by
    Blender's flip_name: .L/.R, _l/_r, Left/Right). Every value is validated and a failed write rolls the targets back.

    Reply: per bone its type and every parameter of that type read back; the type's parameter definitions (name, kind,
    default, items); copiedTo.
    """
    reject_unknown(params, _TYPE_KEYS, "set_rigify_type")
    with _PrefsGuard():
        obj = _armature(params)
        data = obj.data
        names = params.get("bones")
        chain_from = params.get("chainFrom")
        if (names is None) == (chain_from is None):
            raise MifOpError("pass bones [names] OR chainFrom '<bone>'. %s" % _TAIL)
        if names is not None:
            if not isinstance(names, (list, tuple)) or not names or not all(isinstance(n, str) for n in names):
                raise MifOpError("bones must be a non-empty list of bone names. %s" % _TAIL)
            have = set(data.bones.keys())
            missing = [n for n in names if n not in have]
            if missing:
                raise MifOpError("no bone(s) %s on '%s'. %s" % (missing[:10], obj.name, _TAIL))
            targets = list(dict.fromkeys(names))
        else:
            if not isinstance(chain_from, str) or chain_from not in data.bones:
                raise MifOpError("chainFrom must name a bone of '%s', got %r. %s" % (obj.name, chain_from, _TAIL))
            targets = [b.name for b in _chain(data, chain_from)]
        typ = params.get("type")
        if typ is not None and not isinstance(typ, str):
            raise MifOpError("type must be text (a rig type, or '' to clear), got %r. %s" % (typ, _TAIL))
        values = params.get("parameters")
        if values is not None and (not isinstance(values, dict) or not values):
            raise MifOpError("parameters must be a non-empty {name: value}. %s" % _TAIL)
        copy = params.get("copyTo")
        if copy is not None:
            if not isinstance(copy, dict) or len(copy) != 1 or next(iter(copy)) not in _COPY_KEYS:
                raise MifOpError("copyTo must be ONE of {bones: [names]}, {chainChildren: '<bone>'}, {mirror: true}, "
                                 "got %r. %s" % (copy, _TAIL))
        if typ is None and values is None and copy is None:
            raise MifOpError("nothing to write: pass type, parameters or copyTo. %s" % _TAIL)
        if obj.mode == "EDIT":
            raise MifOpError("'%s' is in edit mode; pose bones (which hold the rig type) are written outside it. Leave "
                             "edit mode first. %s" % (obj.name, _TAIL))
        if is_generated(obj):
            raise MifOpError("'%s' is a GENERATED rig; rig types belong on its metarig, and a regenerate rebuilds this "
                             "one from there. %s" % (obj.name, _TAIL))
        enabled_now = ensure_rigify(params)
        rigs = _rig_types()
        if typ and typ not in rigs:
            raise MifOpError("'%s' is not a rig type of this Blender's generator (%s). It has: %s. %s"
                             % (typ, bpy.app.version_string, ", ".join(sorted(rigs)), _TAIL))
        pbs = obj.pose.bones
        eff = {n: (typ if typ is not None else pbs[n].rigify_type) for n in targets}
        if values is not None:
            for n in targets:
                t = eff[n]
                if not t:
                    raise MifOpError("bone '%s' has no rig type, so it has no parameters; pass type. %s" % (n, _TAIL))
                defs = {d["name"]: d for d in type_parameters(t)}
                unknown = [k for k in values if k not in defs]
                if unknown:
                    raise MifOpError("rig type '%s' does not define %s (a name another type defines is accepted by "
                                     "Blender and does nothing for this one). It defines: %s. %s"
                                     % (t, ", ".join(unknown), ", ".join(sorted(defs)) or "(nothing)", _TAIL))
        copy_targets = []
        have = set(data.bones.keys())
        if copy is not None:
            key, val = next(iter(copy.items()))
            if key == "bones":
                if not isinstance(val, (list, tuple)) or not val or not all(isinstance(v, str) for v in val):
                    raise MifOpError("copyTo.bones must be a non-empty list of bone names. %s" % _TAIL)
                missing = [v for v in val if v not in have]
                if missing:
                    raise MifOpError("copyTo.bones names %s, not on '%s'. %s" % (missing[:10], obj.name, _TAIL))
                copy_targets = [(targets[0], v) for v in dict.fromkeys(val) if v != targets[0]]
            elif key == "chainChildren":
                if not isinstance(val, str) or val not in data.bones:
                    raise MifOpError("copyTo.chainChildren must name a bone of '%s', got %r. %s"
                                     % (obj.name, val, _TAIL))
                copy_targets = [(targets[0], c.name) for c in data.bones[val].children if c.name != targets[0]]
                if not copy_targets:
                    raise MifOpError("'%s' has no other child bone to copy to. %s" % (val, _TAIL))
            else:
                if take_bool(copy, "mirror") is not True:
                    raise MifOpError("copyTo.mirror must be true. %s" % _TAIL)
                for n in targets:
                    other = bpy.utils.flip_name(n)
                    if other == n:
                        raise MifOpError("'%s' has no side in its name (.L/.R, _l/_r, Left/Right), so it has no mirror. "
                                         "%s" % (n, _TAIL))
                    if other not in have:
                        raise MifOpError("the mirror of '%s' is '%s', which '%s' does not have. %s"
                                         % (n, other, obj.name, _TAIL))
                    copy_targets.append((n, other))
        touched = list(dict.fromkeys(targets + [d for _s, d in copy_targets]))
        all_defs = set()
        for n in touched:
            for t in {eff.get(n), pbs[n].rigify_type, typ}:
                if t:
                    all_defs.update(d["name"] for d in type_parameters(t))
        snaps = {n: _param_snapshot(pbs[n], sorted(all_defs)) for n in touched}
        try:
            for n in targets:
                pb = pbs[n]
                if typ is not None:
                    pb.rigify_type = typ
                if values:
                    defs = {d["name"]: d for d in type_parameters(pb.rigify_type)}
                    for k, v in values.items():
                        _write_param(pb, defs[k], v, obj)
            copied = []
            for src, dst in copy_targets:
                a, b = pbs[src], pbs[dst]
                b.rigify_type = a.rigify_type
                for d in type_parameters(a.rigify_type):
                    v = getattr(a.rigify_parameters, d["name"])
                    if d["kind"] == "BONE_COLLECTIONS":
                        refs = [r.name for r in v]
                        if key == "mirror":
                            refs = [bpy.utils.flip_name(r) if bpy.utils.flip_name(r) in data.collections_all else r
                                    for r in refs]
                        _write_param(b, d, refs, obj)
                    else:
                        setattr(b.rigify_parameters, d["name"], v if not hasattr(v, "__len__") or isinstance(v, str)
                                else list(v))
                copied.append(dst)
        except Exception as exc:
            for n, snap in snaps.items():
                _param_restore(pbs[n], snap)
            if isinstance(exc, MifOpError):
                text = str(exc)
                raise MifOpError(text if text.endswith(_TAIL) else "%s %s" % (text, _TAIL))
            raise
        rows = []
        defs_out = {}
        for n in touched:
            t = pbs[n].rigify_type
            row = {"bone": n, "type": t}
            if t:
                defs = type_parameters(t)
                defs_out[t] = defs
                row["parameters"] = _param_read(pbs[n], defs)
            rows.append(row)
        return {"object": obj.name, "bones": rows, "typeParameters": defs_out, "copiedTo": copied or None,
                "rigifyEnabledNow": enabled_now}


# --------------------------------------------------------------------------------------------- generation

_GEN_KEYS = ("metarig", "name", "targetRig", "collection", "dryRun", "backup", "allowStraightChains",
             "enableRigify", "previewId", "previewTtlSeconds")
_RIG_ERR = re.compile(r"RIGIFY ERROR: Bone '(?P<bone>[^']+)': (?P<text>.+)", re.S)


def _layer_collection(root, coll):
    if root.collection == coll:
        return root
    for child in root.children:
        found = _layer_collection(child, coll)
        if found is not None:
            return found
    return None


def _rig_summary(rig):
    data = rig.data
    out = {"rig": rig.name, "rigId": data.get("rig_id"), "boneCount": len(data.bones)}
    if _V4:
        by = {}
        for c in data.collections_all:
            by[c.name] = len(c.bones)
        special = {k: by.get(k, 0) for k in _SPECIAL}
        member = {b.name for c in data.collections_all if c.name not in _SPECIAL for b in c.bones}
        special["controls"] = len(member)
        out["bonesByCollection"] = special
    else:
        counts = {role: sum(1 for b in data.bones if b.layers[i]) for i, role in _LAYER_ROLE.items()}
        counts["controls"] = sum(1 for b in data.bones if any(b.layers[i] for i in range(28)))
        out["bonesByLayer"] = counts
    deform = [b.name for b in data.bones if b.use_deform]
    out["deformBoneCount"] = len(deform)
    out["deformBones"] = deform
    out["ikFkSwitches"] = [{"bone": pb.name, "property": "IK_FK", "value": round(float(pb["IK_FK"]), 6)}
                           for pb in rig.pose.bones if "IK_FK" in pb.keys()]
    return out


def ui_text(rig):
    """The generated rig's UI script: the text whose rig_id line names this rig's id."""
    rid = rig.data.get("rig_id")
    if not rid:
        return None
    for meta in bpy.data.objects:
        if meta.type == "ARMATURE" and getattr(meta.data, "rigify_target_rig", None) == rig:
            t = getattr(meta.data, "rigify_rig_ui", None)
            if t is not None:
                return t
    want = 'rig_id = "%s"' % rid
    for t in bpy.data.texts:
        if any(ln.body.strip() == want for ln in t.lines[:80]):
            return t
    return None


def op_generate_rigify_rig(params):
    """Generate (or regenerate) a Rigify rig from a metarig, refusing first what the generator does not check or only
    reports after a half-built rig exists.

    metarig: the metarig armature. Refused before running (dryRun true runs only these): the armature is itself a
    generated rig; its world scale is not 1 or its rotation not 0 (the rig would come out the wrong size with no
    message); no bone has a rig type; on 4.x and later no collection but Root, DEF, ORG and MCH has a UI row above 0 (the
    generator's own refusal); a faces.super_face bone with no children (throws "key 'ORG-nose' not found" and leaves a
    partial rig); a chain told to connect to its parent rig (connect_chain) whose first bone is disjoint from it (throws
    "Cannot connect chain - bone position is disjoint." and leaves a partial rig); and a limb, finger or tentacle chain so straight the generator falls back to the bone's roll axis
    (|first.y x (last.tail - first.head)| < first.length / 100 - measured: an arm bent 0.55 degrees gets an IK elbow that
    does not bend; allowStraightChains true when the roll is meant). name: the new rig's name (data.rigify_rig_basename;
    the generator's default turns metarig into rig, META into RIG, else RIG-<name>). targetRig: an existing rig to
    regenerate into. collection: where a new rig goes. backup (default true when regenerating): a hidden copy of the
    old rig, because a regenerate removes every bone and collection of the target first and a bone added only to the rig
    is gone.

    Reply: rig, rigId, boneCount, bones by collection (Root, DEF, ORG, MCH, controls; layers on 3.6), the deform bone
    names, the IK/FK switch properties, uiScript, uiRegistered, autoRunAllowed; on a regenerate erased [bone names] and
    backup; on a generator failure ok false with the parsed {bone, text}, partial true and what was left behind.

    dryRun is also the BULK PREVIEW (docs/15 section 5) for a NEW rig: the name the rig will get, and a previewId that
    {previewId} alone commits - planned again, refused on any difference, read back, and everything the generation made
    removed on a mismatch. A regenerate cannot be previewed (see _plan_generate_rigify_rig).
    """
    reject_unknown(params, _GEN_KEYS, "generate_rigify_rig")
    with _PrefsGuard():
        obj = _armature(params, ("metarig",))
        dry = take_bool(params, "dryRun", default=False)
        allow = take_bool(params, "allowStraightChains", default=False)
        name = take(params, "name", kind=str)
        if name is not None and (not name or len(name) > _NAME_LIMIT):
            raise MifOpError("name must be 1-%d characters. %s" % (_NAME_LIMIT, _TAIL))
        target = None
        if params.get("targetRig") is not None:
            target = get_object(take(params, "targetRig", kind=str))
            if target.type != "ARMATURE" or target == obj:
                raise MifOpError("targetRig must be another ARMATURE (the rig to regenerate into), got '%s'. %s"
                                 % (target.name, _TAIL))
            require_editable(target)
        coll = None
        if params.get("collection") is not None:
            coll = bpy.data.collections.get(take(params, "collection", kind=str))
            if coll is None:
                raise MifOpError("no collection named '%s'. %s" % (params["collection"], _TAIL))
            if _layer_collection(bpy.context.view_layer.layer_collection, coll) is None:
                raise MifOpError("collection '%s' is not in the active view layer, so the rig could not be put there. "
                                 "%s" % (coll.name, _TAIL))
        _object_mode(obj, "generate_rigify_rig")
        enabled_now = ensure_rigify(params)
        existing = target if target is not None else getattr(obj.data, "rigify_target_rig", None)
        if name is not None and existing is not None:
            raise MifOpError("name names a NEW rig, and '%s' regenerates into '%s'. Rename that rig with rename_object "
                             "instead. %s" % (obj.name, existing.name, _TAIL))
        backup = take_bool(params, "backup", default=existing is not None)
        problems, bends = generate_checks(obj, allow)
        report = {"metarig": obj.name, "checks": problems, "limbBends": bends or None,
                  "regenerates": existing.name if existing is not None else None, "rigifyEnabledNow": enabled_now}
        if dry:
            report.update(dryRun=True, generateReady=not problems)
            return report
        if problems:
            raise MifOpError("'%s' would not generate cleanly: %s. %s"
                             % (obj.name, " | ".join(p["text"] for p in problems), _TAIL))
        before_bones = [b.name for b in existing.data.bones] if existing is not None else []
        before_colls = ([c.name for c in existing.data.collections_all] if existing is not None and _V4 else [])
        snap = selection_snapshot()
        vl = bpy.context.view_layer
        active_lc = vl.active_layer_collection
        old_target = obj.data.rigify_target_rig
        backup_obj = None
        if existing is not None and backup:
            backup_obj = existing.copy()
            backup_obj.data = existing.data.copy()
            backup_obj.name = existing.name + "_backup"
            for c in existing.users_collection:
                c.objects.link(backup_obj)
            backup_obj.hide_set(True)
            backup_obj.hide_render = True
        error = None
        try:
            if target is not None:
                obj.data.rigify_target_rig = target
            if name is not None:
                obj.data.rigify_rig_basename = name
            if coll is not None and existing is None:
                vl.active_layer_collection = _layer_collection(vl.layer_collection, coll)
            _activate(obj)
            bpy.ops.pose.rigify_generate()
        except RuntimeError as exc:
            error = str(exc)
        finally:
            try:
                if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
                    bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                pass
            if coll is not None:
                vl.active_layer_collection = active_lc
            selection_restore(snap)
        rig = obj.data.rigify_target_rig
        if error is not None:
            # REPORTED, NOT RAISED BY THE GENERATOR: it calls report(ERROR) and returns FINISHED, so bpy.ops raises
            # only here - after the rig object exists, half-built. What it left is said, not cleaned up behind the
            # caller's back: on a regenerate the old rig's bones are already gone, and the backup is the way back.
            text = re.sub(r"^Error:\s*", "", error.strip()).strip()
            m = _RIG_ERR.search(text)
            fail = MifOpError("the generator failed on '%s': %s. It reports this AFTER building part of the rig - see "
                              "leftBehind%s." % (obj.name, text, " and backup" if backup_obj is not None else ""))
            fail.payload = {
                "metarig": obj.name,
                "failure": ({"bone": m.group("bone"), "text": m.group("text").strip()} if m
                            else {"bone": None, "text": text}),
                "partial": rig is not None,
                "leftBehind": ({"rig": rig.name, "boneCount": len(rig.data.bones),
                                "wasTarget": rig is old_target} if rig is not None else None),
                "backup": backup_obj.name if backup_obj is not None else None,
                "rigifyEnabledNow": enabled_now}
            raise fail
        if rig is None:
            raise MifOpError("the generator finished and '%s' points at no rig." % obj.name)
        out = {"ok": True, "metarig": obj.name}
        out.update(_rig_summary(rig))
        if existing is not None:
            now = {b.name for b in rig.data.bones}
            out["erased"] = [n for n in before_bones if n not in now]
            if _V4:
                cnow = {c.name for c in rig.data.collections_all}
                out["erasedCollections"] = [n for n in before_colls if n not in cnow]
            out["backup"] = backup_obj.name if backup_obj is not None else None
        txt = ui_text(rig)
        rid = rig.data.get("rig_id")
        out["uiScript"] = txt.name if txt is not None else None
        out["uiRegistered"] = bool(rid and hasattr(bpy.types, "VIEW3D_PT_rig_ui_" + rid))
        out["autoRunAllowed"] = bool(bpy.context.preferences.filepaths.use_scripts_auto_execute)
        out["regenerated"] = existing is not None
        out["rigifyEnabledNow"] = enabled_now
        return out


# --------------------------------------------------------------------------------------------- snapping

_SNAP_KEYS = ("object", "limb", "direction", "frame", "key", "switch", "bake", "allowSimulation")
_LABEL = re.compile(r"text='(?P<label>[^']*)'")
_OPER = re.compile(r"props = \w+\.operator\('(?P<op>[\w.]+)'")
_SLIDER = re.compile(r"\.prop\(pose_bones\['(?P<bone>[^']+)'\], '\[\"IK_FK\"\]', text='IK-FK \((?P<limb>[^)]+)\)'")


def snap_buttons(text):
    """Every operator button with properties in the rig's UI text, and every IK-FK slider."""
    lines = [ln.body.strip() for ln in text.lines]
    buttons, sliders = [], {}
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = _OPER.search(ln)
        if m:
            lab = _LABEL.search(ln)
            props = {}
            j = i + 1
            while j < len(lines) and lines[j].startswith("props."):
                k, _, v = lines[j][len("props."):].partition(" = ")
                try:
                    props[k] = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    props[k] = v
                j += 1
            buttons.append({"operator": m.group("op"), "label": lab.group("label") if lab else "",
                            "properties": props})
            i = j
            continue
        s = _SLIDER.search(ln)
        if s:
            sliders[s.group("limb")] = s.group("bone")
        i += 1
    return buttons, sliders


def _switch_direction(rig, bone):
    """Which value of IK_FK means IK, read from the drivers it feeds: the influence of the constraint copying the IK
    chain, evaluated at 0 and 1 through the driver curve (its modifiers included)."""
    ad = rig.animation_data
    if ad is None:
        return None
    path = 'pose.bones["%s"]["IK_FK"]' % bone
    pose_names = set(rig.pose.bones.keys())
    for fc in ad.drivers:
        d = fc.driver
        if not any(t.data_path == path for v in d.variables for t in v.targets):
            continue
        m = re.match(r'pose\.bones\["([^"]+)"\]\.constraints\["([^"]+)"\]\.influence', fc.data_path)
        if not m or m.group(1) not in pose_names:
            continue
        con = rig.pose.bones[m.group(1)].constraints.get(m.group(2))
        sub = getattr(con, "subtarget", "") or ""
        at0, at1 = fc.evaluate(0.0), fc.evaluate(1.0)
        copies_ik = "_ik" in sub.lower() or "ik_" in sub.lower()
        mods = [{"type": md.type, "mode": getattr(md, "mode", None),
                 "coefficients": [round(c, 6) for c in getattr(md, "coefficients", [])]} for md in fc.modifiers]
        ik_value = (0.0 if at0 > at1 else 1.0) if copies_ik else (1.0 if at0 > at1 else 0.0)
        return {"drivenBy": fc.data_path, "copies": sub, "influenceAt0": round(at0, 6), "influenceAt1": round(at1, 6),
                "driverType": d.type, "curveModifiers": mods, "ikAt": ik_value, "fkAt": 1.0 - ik_value}
    return None


def _chain_gap(rig, a, b):
    pbs = rig.pose.bones
    gaps = [(pbs[x].matrix.translation - pbs[y].matrix.translation).length for x, y in zip(a, b)
            if x in pbs and y in pbs]
    return round(max(gaps), 6) if gaps else None


def op_rigify_snap(params):
    """Snap a generated Rigify rig's limb IK to FK or FK to IK with the rig's own snap operators, or list them and which
    way each limb's IK/FK switch runs.

    object: the generated rig. limb: the limb as its UI names it (hand.L, foot.R) or its settings bone
    (upper_arm_parent.L). direction LIST (default; limb optional) | IK_TO_FK (the IK controls moved onto the FK pose) |
    FK_TO_IK (the FK controls moved onto the IK result). The operator and its bone lists come from the rig's UI text,
    exactly as its panel button calls them. frame: go to this frame first. key true: key what the snap moved (the
    operator keys only with auto-keying, which is turned on for the call and restored). switch true: also set the limb's
    IK_FK switch to the side snapped to (and key it with key). bake {start, end}: the operator's bake form over that
    range instead - it re-snaps every keyed frame of the source chain, and needs an action. The switch direction is read
    from the driver curve, never assumed: 0 is IK on a generated rig (the IK copy's influence is 1 - value).

    Reply: LIST: limbs [{limb, switch {bone, value, ikAt, fkAt, mode}, ikToFk, fkToIk, operators}] and buttons; a snap:
    the operator run, the chain gap (largest head distance between the two chains) before and after, and the switch.
    """
    reject_unknown(params, _SNAP_KEYS, "rigify_snap")
    with _PrefsGuard():
        rig = _armature(params)
        if not is_generated(rig):
            raise MifOpError("'%s' is not a generated Rigify rig (it has no rig_id). %s" % (rig.name, _TAIL))
        direction = str(take(params, "direction", default="LIST", kind=str)).upper()
        if direction not in ("LIST", "IK_TO_FK", "FK_TO_IK"):
            raise MifOpError("direction must be LIST, IK_TO_FK or FK_TO_IK, got %r. %s" % (direction, _TAIL))
        if direction == "LIST":
            extra = [k for k in ("frame", "key", "switch", "bake") if params.get(k) is not None]
            if extra:
                raise MifOpError("LIST reads nothing but object and limb; %s are for a snap. %s"
                                 % (", ".join(extra), _TAIL))
        rid = rig.data["rig_id"]
        text = ui_text(rig)
        if text is None:
            raise MifOpError("no UI script for rig id %s is in this file, so its snap buttons cannot be read. Regenerate "
                             "the rig. %s" % (rid, _TAIL))
        buttons, sliders = snap_buttons(text)
        registered = hasattr(bpy.types, "VIEW3D_PT_rig_ui_" + rid)
        limbs = []
        pose_names = set(rig.pose.bones.keys())
        names = list(dict.fromkeys(list(sliders) + [re.sub(r"^.*\((.+)\)$", r"\1", b["label"])
                                                    for b in buttons if re.search(r"(IK->FK|FK->IK) \(", b["label"])]))
        for limb in names:
            ik2fk = next((b for b in buttons if b["label"] == "IK->FK (%s)" % limb), None)
            fk2ik = next((b for b in buttons if b["label"] == "FK->IK (%s)" % limb), None)
            bone = sliders.get(limb) or (ik2fk or {}).get("properties", {}).get("prop_bone")
            sw = None
            if bone and bone in pose_names and "IK_FK" in rig.pose.bones[bone].keys():
                v = float(rig.pose.bones[bone]["IK_FK"])
                sw = {"bone": bone, "property": "IK_FK", "value": round(v, 6)}
                dirn = _switch_direction(rig, bone)
                if dirn:
                    sw.update(dirn)
                    sw["mode"] = ("IK" if abs(v - dirn["ikAt"]) < 1e-6 else
                                  "FK" if abs(v - dirn["fkAt"]) < 1e-6 else "blend")
            limbs.append({"limb": limb, "switch": sw, "ikToFk": ik2fk, "fkToIk": fk2ik})
        want = params.get("limb")
        if want is not None:
            if not isinstance(want, str):
                raise MifOpError("limb must be text, got %r. %s" % (want, _TAIL))
            picked = [x for x in limbs if want in (x["limb"], (x["switch"] or {}).get("bone"))]
            if not picked:
                raise MifOpError("no limb '%s' on '%s'. Its limbs: %s. %s"
                                 % (want, rig.name, ["%s (%s)" % (x["limb"], (x["switch"] or {}).get("bone"))
                                                     for x in limbs] or "none", _TAIL))
            limbs = picked
        if direction == "LIST":
            return {"object": rig.name, "rigId": rid, "uiScript": text.name, "uiRegistered": registered,
                    "autoRunAllowed": bool(bpy.context.preferences.filepaths.use_scripts_auto_execute),
                    "limbs": limbs, "buttonCount": len(buttons)}
        if want is None:
            raise MifOpError("a snap needs limb (one of %s). %s" % ([x["limb"] for x in limbs], _TAIL))
        limb = limbs[0]
        button = limb["ikToFk"] if direction == "IK_TO_FK" else limb["fkToIk"]
        if button is None:
            raise MifOpError("limb '%s' has no %s button in its UI. %s"
                             % (limb["limb"], "IK->FK" if direction == "IK_TO_FK" else "FK->IK", _TAIL))
        if not registered:
            raise MifOpError("the rig's UI script was never run in this session (a re-opened file with Auto Run Python "
                             "Scripts %s), so its snap operators do not exist. Run the text '%s' or allow auto-run and "
                             "re-open. %s" % ("off" if not bpy.context.preferences.filepaths.use_scripts_auto_execute
                                              else "on but not yet run", text.name, _TAIL))
        frame = take_int(params, "frame", default=None)
        key = take_bool(params, "key", default=False)
        switch = take_bool(params, "switch", default=False)
        bake = params.get("bake")
        if bake is not None:
            if not isinstance(bake, dict) or set(bake) != {"start", "end"}:
                raise MifOpError("bake must be {start, end} (frames), got %r. %s" % (bake, _TAIL))
            b0, b1 = take_int(bake, "start", required=True), take_int(bake, "end", required=True)
            if b1 < b0:
                raise MifOpError("bake end %d is before start %d. %s" % (b1, b0, _TAIL))
            if rig.animation_data is None or rig.animation_data.action is None:
                raise MifOpError("the bake form re-snaps keyed frames and '%s' has no action. %s" % (rig.name, _TAIL))
            if frame is not None or switch:
                raise MifOpError("bake runs over its range; frame and switch are for a single snap. %s" % _TAIL)
        sw = limb["switch"]
        if switch and (sw is None or "ikAt" not in sw):
            raise MifOpError("limb '%s' has no IK_FK switch whose direction could be read, so switch cannot be set. %s"
                             % (limb["limb"], _TAIL))
        # A FRAME TO SNAP AT, or a bake's range, steps every live simulation (ops_common.simulation_guard).
        visits = [b0, b1] if bake is not None else ([frame] if frame is not None else [])
        sim_reply = simulation_guard(params, visits, "rigify_snap", tail=_TAIL)
        opname = button["operator"]
        if bake is not None:
            opname = opname.replace("_" + rid, "_bake_" + rid)
        opid = opname.split(".", 1)[1]
        op = getattr(bpy.ops.pose, opid, None)
        if op is None:
            raise MifOpError("operator %s is not registered. %s" % (opname, _TAIL))
        props = dict(button["properties"])
        if direction == "IK_TO_FK":
            pair = (json.loads(props.get("fk_bones", "[]")), json.loads(props.get("ik_bones", "[]")))
        else:
            pair = (json.loads(props.get("output_bones", "[]")), json.loads(props.get("input_bones", "[]")))
        scene = bpy.context.scene
        ts = scene.tool_settings
        snap = selection_snapshot()
        auto = ts.use_keyframe_insert_auto
        # ONLY INSERT AVAILABLE IS ON BY DEFAULT IN 5.2 (preferences.edit.use_keyframe_insert_available, factory
        # settings): the snap operators key through auto-keying with that flag (rigify utils/animation.py
        # get_autokey_flags, 5.2 :110-121), so on a rig with no curves yet they keyed nothing while this answered
        # keyed true (settled 2026-10-18, test_blender_settle_routes BS604b). It is turned off for the call and put
        # back with the preferences' dirty flag, as auto-keying is.
        prefs = bpy.context.preferences
        avail = prefs.edit.use_keyframe_insert_available
        dirty = prefs.is_dirty
        from .ops_anim import _fcurves   # noqa: PLC0415
        keys_before = sum(len(fc.keyframe_points) for fc in _fcurves(rig))
        wm = bpy.context.window_manager
        saved_range = None
        try:
            if frame is not None:
                scene.frame_set(frame)
            bpy.context.view_layer.update()
            gap_before = _chain_gap(rig, *pair)
            _activate(rig)
            if key:
                ts.use_keyframe_insert_auto = True
                prefs.edit.use_keyframe_insert_available = False
            if bake is not None:
                saved_range = (wm.rigify_transfer_use_frame_range, wm.rigify_transfer_start_frame,
                               wm.rigify_transfer_end_frame)
                wm.rigify_transfer_use_frame_range = True
                wm.rigify_transfer_start_frame = b0
                wm.rigify_transfer_end_frame = b1
            op(**props)
            bpy.context.view_layer.update()
            if switch:
                target_value = sw["fkAt"] if direction == "FK_TO_IK" else sw["ikAt"]
                pb = rig.pose.bones[sw["bone"]]
                pb["IK_FK"] = target_value
                if key:
                    pb.keyframe_insert('["IK_FK"]', frame=scene.frame_current)
                bpy.context.view_layer.update()
            gap_after = _chain_gap(rig, *pair)
        except RuntimeError as exc:
            raise MifOpError("%s refused: %s" % (opname, str(exc).strip()))
        finally:
            ts.use_keyframe_insert_auto = auto
            if prefs.edit.use_keyframe_insert_available != avail:
                prefs.edit.use_keyframe_insert_available = avail
                prefs.is_dirty = dirty
            if saved_range is not None:
                (wm.rigify_transfer_use_frame_range, wm.rigify_transfer_start_frame,
                 wm.rigify_transfer_end_frame) = saved_range
            selection_restore(snap)
        swb = sw["bone"] if sw else None
        return dict({"object": rig.name, "limb": limb["limb"], "direction": direction, "operator": opname,
                "frame": scene.frame_current, "keyed": key, "baked": bake,
                "keysAdded": sum(len(fc.keyframe_points) for fc in _fcurves(rig)) - keys_before,
                "chainGapBefore": gap_before, "chainGapAfter": gap_after,
                "chains": {"moved": pair[0], "onto": pair[1]} if direction == "FK_TO_IK"
                else {"moved": json.loads(props.get("ctrl_bones", "[]")), "onto": pair[0]},
                "switch": ({"bone": swb, "value": round(float(rig.pose.bones[swb]["IK_FK"]), 6),
                            "ikAt": sw.get("ikAt"), "fkAt": sw.get("fkAt")} if swb else None)}, **sim_reply)


# ---------------------------------------------------------------------------------------------------------------------
# THE BULK PREVIEW of generate_rigify_rig (ops_preview.py; docs/15 section 5, 2026-10-09). What a NEW rig will be called is
# the generator's own rule (rigify/generate.py: rigify_rig_basename, else metarig -> rig, META -> RIG, else RIG-<name>),
# made unique as Blender makes any object name; the rig's bones are the generator's and are not known before it runs, so
# the change is the metarig's target rig, read back by name. The inverse removes every data-block the generation made -
# the rig, its armature, the widget objects and their collection, the UI script - and puts the metarig's target back. A
# REGENERATE cannot be previewed: it rewrites the target's bones in place and only its backup brings them back.
# ---------------------------------------------------------------------------------------------------------------------
def _new_rig_name(meta, name):
    base = name or getattr(meta.data, "rigify_rig_basename", "") or (
        meta.name.replace("metarig", "rig") if "metarig" in meta.name
        else meta.name.replace("META", "RIG") if "META" in meta.name else "RIG-" + meta.name)
    if bpy.data.objects.get(base) is None:
        return base
    n = 1
    while bpy.data.objects.get("%s.%03d" % (base, n)) is not None:
        n += 1
    return "%s.%03d" % (base, n)


def _plan_generate_rigify_rig(request):
    resp = op_generate_rigify_rig(dict(request, dryRun=True))          # every check, and writes nothing
    # A REFUSED PREVIEW STILL CARRIES THE DRY RUN'S DIAGNOSIS (server._dispatch answers an error's payload): checks[] with
    # each failing check by name, limbBends and generateReady false. Refused with the text alone, a dryRun on a metarig
    # that would not generate lost the very report it exists for (found 2026-10-10, test_blender_rigify RG130).
    if resp.get("checks"):
        exc = MifOpError("'%s' would not generate cleanly: %s. Nothing was written."
                         % (resp["metarig"], " | ".join(p["text"] for p in resp["checks"])))
        exc.payload = resp
        raise exc
    if resp.get("regenerates"):
        exc = MifOpError("a regenerate cannot be previewed: it rewrites '%s' bone by bone in place, which bones it makes "
                         "is the generator's own, and only its backup brings the old ones back. Generate without dryRun "
                         "(backup is on by default for a regenerate). Nothing was written." % resp["regenerates"])
        exc.payload = resp
        raise exc
    meta = bpy.data.objects[resp["metarig"]]
    rig = _new_rig_name(meta, request.get("name"))
    change = {"kind": "rig", "op": "create", "target": meta.name, "field": "rigify_target_rig", "label": rig,
              "before": None, "after": rig}
    typed = sum(1 for pb in meta.pose.bones if getattr(pb, "rigify_type", ""))
    return {"changes": [change], "unchanged": [], "commit": dict(request), "response": resp,
            "context": ["metarig %s: %d bones, %d with a rig type" % (meta.name, len(meta.data.bones), typed)]}


def _read_generate_rigify_rig(change, _kept):
    meta = bpy.data.objects.get(change["target"])
    rig = getattr(meta.data, "rigify_target_rig", None) if meta is not None and meta.type == "ARMATURE" else None
    return rig.name if rig is not None else None


def _capture_generate_rigify_rig(plan):
    from .ops_preview import _ids   # noqa: PLC0415
    meta = bpy.data.objects[plan["response"]["metarig"]]
    return {"ids": _ids(), "metarig": meta.name, "basename": getattr(meta.data, "rigify_rig_basename", "")}


def _restore_generate_rigify_rig(kept, _plan):
    from .ops_preview import _ID_KINDS   # noqa: PLC0415
    made = [i for k in _ID_KINDS if hasattr(bpy.data, k) for i in getattr(bpy.data, k)
            if i.as_pointer() not in kept["ids"].get(k, set())]
    meta = bpy.data.objects.get(kept["metarig"])
    if meta is not None:
        meta.data.rigify_target_rig = None
        meta.data.rigify_rig_basename = kept["basename"]
    if made:
        bpy.data.batch_remove(made)
    bpy.context.view_layer.update()


PREVIEW = {"generate_rigify_rig": {"plan": _plan_generate_rigify_rig, "read": _read_generate_rigify_rig,
                                   "capture": _capture_generate_rigify_rig, "restore": _restore_generate_rigify_rig}}

OPS = {
    "set_bone_collections": op_set_bone_collections,
    "set_rigify_type": op_set_rigify_type,
    "generate_rigify_rig": op_generate_rigify_rig,
    "rigify_snap": op_rigify_snap,
}
