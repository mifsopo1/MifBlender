"""Keyframes, the frame range, and reading animation back.

WHY THIS IS THE THIRD GAP CLOSED, after lights and cameras. Of the eight families with no typed op,
this is the one that gates a whole CLASS of request rather than a property: without it nothing
moves, so a fan cannot spin, a light cannot flicker, a camera cannot travel and an object cannot
fall. Every one of those was in the benchmark.

=============================================================================
THE THING THAT MAKES KEYFRAMING BY HAND WRONG: WHAT DO YOU KEY, THE OBJECT OR THE DATA?
=============================================================================
An object's transform lives on the OBJECT (obj.location). A light's brightness lives on its DATA
(obj.data.energy), and so does a camera's focal length. keyframe_insert() has to be called on the
datablock that OWNS the property, and calling it on the wrong one raises a bare RuntimeError naming
a data path rather than the mistake.

So `target` is explicit here - "object" or "data" - and defaults to the right one for the channel
being written rather than making every caller know. A caller keying `energy` gets the light data
without asking; a caller keying `location` gets the object.

=============================================================================
INSERTING A KEYFRAME MEANS WRITING THE VALUE FIRST
=============================================================================
keyframe_insert reads the CURRENT value of the property and stores it at the given frame. It does
not take a value. So every op here sets the property, then keys it - and a caller who expects to
pass a value and have the object left alone afterwards is wrong about what a keyframe is: the
object really is left holding the last value written. That is stated rather than hidden, because
"my object moved" is otherwise a confusing side effect.
"""
import math
import re

import bpy

from .ops_common import (MifOpError, simulation_guard, assign_id_name, check_axis_dict, exclude_rows, finite_float, finite_floats, finite_int, 
                         get_object, limit_rows, reject_unknown, require_editable, rnd, select_only, 
                         selection_restore, selection_snapshot, take, take_bool, take_float, take_int)

# AN OBJECT'S OWN TRANSFORM CHANNELS, which keys, drivers and NLA write and a parent, a constraint or a rigid body never
# does - evaluate_at_frame names the difference when one is asked for. Deltas are in the basis matrix too.
_TRANSFORM_CHANNELS = {"location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale",
                       "delta_location", "delta_rotation_euler", "delta_rotation_quaternion", "delta_scale",
                       "matrix_basis"}
_KEY_KEYS = {
    "object", "name", "frame", "location", "rotation", "scale",
    "dataPath", "path", "value", "index", "target", "interpolation",
    # a world's, a material's or a node group's tree (target world | material | node_group) - 2026-10-04
    "datablock",
    # a geometry-nodes modifier input by modifier and socket name or identifier - 2026-10-06
    "modifierInput",
}
_RANGE_KEYS = {"start", "end", "fps", "current", "frameStart", "frameEnd",
               "fpsBase", "frameStep", "previewStart", "previewEnd", "usePreviewRange", "syncMode", "allowSimulation",
               "fitToStrips"}
_LIST_KEYS = {"object", "name", "target"}
# A SEPARATE SET, and the reason is a defect this made within the minute: _LIST_KEYS is shared by
# list_keyframes and list_animation_data, so adding the three bounds to it handed all three to
# list_animation_data as well - accepted, passed the guard, and read by nothing. audit_blender_dead_params
# named all three immediately, which is the audit doing exactly its job. list_animation_data returns a
# summary per holder (three at most) and has nothing to bound; only the op whose answer grows gets the keys.
_KEYFRAME_LIST_KEYS = _LIST_KEYS | {"limit", "nameExcludes", "maxKeyframes", "datablock"}

# Channels this op knows where to put without being told. Anything else needs an explicit target.
_OBJECT_CHANNELS = {"location", "rotation_euler", "scale", "rotation_quaternion",
                    "hide_viewport", "hide_render"}
_DATA_HINTS = {"energy", "color", "spot_size", "spot_blend", "lens", "focus_distance",
               "shadow_soft_size", "size", "angle", "ortho_scale", "default_value"}



def _fcurves(holder):
    """Every fcurve on `holder`, on ANY Blender this addon supports.

    BLENDER 5.0 MOVED THEM AND THE OLD PATH IS GONE, not deprecated. Up to 4.4 an action owned its
    curves directly as action.fcurves. In 5.0's slotted actions they live at
        action.layers[i].strips[j].channelbag(slot).fcurves
    and `Action` has no `fcurves` attribute at all, so the old access raises AttributeError rather
    than returning empty - which is how this was found: the first run of set_keyframe died with
    "'Action' object has no attribute 'fcurves'" on 5.0.1.

    Both paths are tried, newest first, because the suites run 3.6, 4.2, 4.4 and 5.0 and a helper
    that only knows one of them makes three of those versions red.
    """
    ad = getattr(holder, "animation_data", None)
    if not ad or not ad.action:
        return []
    act = ad.action
    slot = getattr(ad, "action_slot", None)
    # 4.4's Action.fcurves IS THE FIRST SLOT'S CURVES ONLY (measured 2026-10-06 on 4.4.0: a second object's slot keyed
    # on location[2] listed the first object's location[0]), so a layered action is read through its slot there too.
    if hasattr(act, "fcurves") and not (slot is not None and len(getattr(act, "layers", ()))):
        return list(act.fcurves)
    out = []
    for layer in getattr(act, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            cb = None
            if hasattr(strip, "channelbag") and slot is not None:
                try:
                    cb = strip.channelbag(slot)
                except (RuntimeError, TypeError):
                    cb = None
            if cb is None:
                for maybe in getattr(strip, "channelbags", ()):
                    out.extend(getattr(maybe, "fcurves", ()))
                continue
            out.extend(getattr(cb, "fcurves", ()))
    return out


def _vec3(params, key):
    v = params.get(key)
    if isinstance(v, dict):
        # AT LEAST ONE OF x/y/z, AND NOTHING ELSE - see ops_create for why. Fifth copy of this
        # parser in the addon and the defect was in every one: a dict read with .get(axis,
        # default) turns {"mif":"typo"} into the DEFAULT vector and reports success.
        check_axis_dict(v, key, ("x", "y", "z"))
        return tuple(finite_floats([v.get("x", 0.0), v.get("y", 0.0), v.get("z", 0.0)], key))
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return tuple(finite_floats(v, key))
    raise MifOpError("'%s' must be {x,y,z} or a 3-list, got %r. NOTHING was keyed." % (key, v))


def _shape_key_block(obj):
    """The Key datablock holding obj's shape keys, or None. It is its OWN ID - not the mesh - so a key's value is
    keyed and listed there, and a path from the mesh into it spans two IDs, which Blender refuses to key."""
    data = getattr(obj, "data", None)
    return getattr(data, "shape_keys", None) if data is not None else None


def _anim_holders(obj):
    """(label, datablock) for every place an object's action curves can live: the object, its data, and the data's
    shape-key Key block - the same three list_keyframes reads. A node tree stand-in (_TreeSubject) has exactly one.

    THE KEY WAS MISSING FROM EVERY WRITER BUT ONE until 2026-09-28 (the 2026-09-21 pack-rule review): set_keyframe
    keys a shape key on the Key and list_keyframes lists it there, while move_keyframes, edit_fcurve, delete_keyframe
    and add_fcurve_modifier walked the object and its data only - so a curve set_keyframe had just made answered
    "no fcurve" to all four (measured on 5.2.1)."""
    if isinstance(obj, _TreeSubject):
        return [(obj.kind, obj.tree)]
    out = [("object", obj)]
    if obj.data is not None:
        out.append(("data", obj.data))
    key = _shape_key_block(obj)
    if key is not None:
        out.append(("shape_keys", key))
    return out


# ---- a world's, a material's or a node group's node tree (2026-10-04) --------------------------------------------
# A NODE SOCKET IS ANIMATED ON ITS TREE'S OWN ANIMATION DATA, not the object's: an HDRI's Mapping rotation is keyed on the
# world's node tree, a material's roughness on the material's. MEASURED 2026-10-04 on 3.6.23, 4.4.0 and 5.2.1: the tree is
# an ID, keyframe_insert('nodes["Mapping"].inputs[2].default_value', index=2) on it makes an action of its own and the
# world's rotation reads 1.5708 at frame 40 between keys 0 and 3.1416 at 0 and 80 - but a material on no object and a
# node group no modifier uses keep their last written value on a frame change: nothing evaluates them, so the key plays
# only where the tree is in use, and the reply says when it is not.
_TREE_KINDS = ("world", "material", "node_group")


class _TreeSubject:
    """Stands where an object stands in the keyframe ops: `name` for their messages, `data` None, and _anim_holders
    answers its one holder - the node tree."""

    def __init__(self, kind, idblock, tree):
        self.kind, self.idblock, self.tree = kind, idblock, tree
        self.name = idblock.name
        self.data = None


def _tree_subject(params, op_name, tail="NOTHING was changed."):
    """A _TreeSubject when `target` names a world, a material or a node group (by `datablock`), else None."""
    raw = params.get("target")
    kind = raw.strip().lower() if isinstance(raw, str) else None
    if kind not in _TREE_KINDS:
        if params.get("datablock") is not None:
            raise MifOpError("datablock names a world, a material or a node group, and goes with target world, material "
                             "or node_group. %s" % tail)
        return None
    if params.get("object") is not None or params.get("name") is not None:
        raise MifOpError("target %s keys the %s's node tree, named by datablock - an object is a different holder. Pass "
                         "one. %s" % (kind, kind.replace("_", " "), tail))
    name = take(params, "datablock", required=True, kind=str)
    coll = {"world": bpy.data.worlds, "material": bpy.data.materials, "node_group": bpy.data.node_groups}[kind]
    idb = coll.get(name)
    if idb is None:
        have = sorted(d.name for d in coll)[:25]
        raise MifOpError("no %s named '%s'. Present: %s. %s" % (kind.replace("_", " "), name, ", ".join(have) or "(none)",
                                                               tail))
    tree = idb if kind == "node_group" else getattr(idb, "node_tree", None)
    if tree is None:
        raise MifOpError("%s '%s' has no node tree (use_nodes is off), so it has no sockets to key. %s"
                         % (kind, idb.name, tail))
    return _TreeSubject(kind, idb, tree)


def _anim_subject(params, op_name, tail="NOTHING was changed.", tree_target_only=True):
    """The object the keyframe ops act on, or a node tree's stand-in when `target` names one. With tree_target_only
    (edit_fcurve, move_keyframes, delete_keyframe), any other `target` is refused - those find an object's curves on all
    three of its holders without one."""
    sub = _tree_subject(params, op_name, tail)
    if sub is not None:
        return sub
    if params.get("target") is not None and tree_target_only:
        raise MifOpError("target here takes world, material or node_group (with datablock); an object's own curves are "
                         "found on the object, its data and its shape keys without one. %s" % tail)
    return get_object(take(params, "object", "name", required=True))


def _tree_in_use(sub):
    """None when something evaluates this tree, else a sentence saying the key will not play."""
    idb = sub.idblock
    if sub.kind == "world":
        if any(sc.world == idb for sc in bpy.data.scenes):
            return None
        return "no scene uses world '%s', so its key does not play until one does" % idb.name
    if sub.kind == "material":
        if any(slot.material == idb for ob in bpy.data.objects for slot in ob.material_slots):
            return None
        return ("no object uses material '%s', so its key does not play until one does (measured: an unused "
                "material keeps its last written value on a frame change)" % idb.name)
    if idb.users > (1 if idb.use_fake_user else 0):
        return None
    return ("nothing uses node group '%s', so its key does not play until a modifier or a node does (measured: an "
            "unused group keeps its last written value on a frame change)" % idb.name)


def _key_tree_value(params, sub, frame, interp):
    """set_keyframe on a node tree: dataPath relative to the tree ('nodes[\"Mapping\"].inputs[2].default_value')."""
    for key in ("location", "rotation", "scale"):
        if key in params:
            raise MifOpError("%s keys an object's transform; a node tree takes dataPath + value. NOTHING was keyed." % key)
    path = take(params, "dataPath", "path", default=None, kind=str)
    if not path:
        raise MifOpError("target %s needs dataPath - the path on the node tree, e.g. 'nodes[\"Mapping\"].inputs[2]."
                         "default_value'. NOTHING was keyed." % sub.kind)
    if "value" not in params:
        raise MifOpError("dataPath needs a `value` - keyframe_insert stores the property's CURRENT value. NOTHING was "
                         "keyed.")
    value = params.get("value")
    index = take_int(params, "index", default=None)
    tree = sub.tree
    head, leaf = (path.rsplit(".", 1) if "." in path else ("", path))
    try:
        holder = tree.path_resolve(head) if head else tree
    except (ValueError, AttributeError, TypeError) as exc:
        names = ", ".join(n.name for n in tree.nodes)[:400]
        raise MifOpError("could not resolve '%s' on %s '%s' (%s). Its nodes: %s. NOTHING was keyed."
                         % (head, sub.kind, sub.name, exc, names or "(none)"))
    prop = _rna_property(holder, leaf)
    if prop is None:
        raise MifOpError("'%s' has no property '%s'. NOTHING was keyed." % (head or sub.name, leaf))
    if not prop.is_animatable:
        raise MifOpError("'%s' can be set but not animated. NOTHING was keyed." % path)
    if index is not None:
        length = getattr(prop, "array_length", 0)
        if not 0 <= index < max(length, 1) or length == 0:
            raise MifOpError("index %d is not an element of '%s' (it has %d). NOTHING was keyed." % (index, path, length))
        want = _rna_value(prop, value, path)
    elif isinstance(value, (list, tuple)):
        want = tuple(_rna_value(prop, v, path) for v in value)
    else:
        want = _rna_value(prop, value, path)
    try:
        if index is not None:
            getattr(holder, leaf)[int(index)] = want
        else:
            setattr(holder, leaf, want)
    except (AttributeError, TypeError, ValueError) as exc:
        raise MifOpError("could not write %r to '%s' on %s '%s': %s. NOTHING was keyed."
                         % (value, path, sub.kind, sub.name, exc))
    # THE PATH BLENDER STORES, not the one asked for. Measured 2026-10-04 on 5.2.1: a socket named in the path
    # (inputs["Roughness"]) is keyed on a curve whose data_path is the socket's INDEX (inputs[2]), so a match on the
    # caller's spelling counted 0 keys at the frame. path_from_id gives the stored form, and both are reported.
    stored = holder.path_from_id(leaf) if holder is not tree else leaf
    kwargs = {"data_path": stored, "frame": frame}
    if index is not None:
        kwargs["index"] = int(index)
    try:
        tree.keyframe_insert(**kwargs)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise MifOpError("Blender would not key '%s' on %s '%s': %s. The value WAS written; no keyframe exists."
                         % (path, sub.kind, sub.name, exc))
    keyed_here = _apply_interpolation(tree, stored, frame, interp)
    curves = [{"dataPath": fc.data_path, "index": fc.array_index, "keyframes": len(fc.keyframe_points)}
              for fc in _fcurves(tree)]
    return {
        "target": sub.kind,
        "datablock": sub.name,
        "tree": tree.name,
        "frame": frame,
        "written": [{"target": sub.kind, "dataPath": path, "storedPath": stored, "index": index,
                     "keysAtThisFrame": keyed_here}],
        "keyframesTotal": sum(c["keyframes"] for c in curves),
        "curves": curves,
        "action": tree.animation_data.action.name if tree.animation_data and tree.animation_data.action else None,
        "playsNote": _tree_in_use(sub),
    }


def _resolve_target(obj, data_path, explicit):
    """The datablock that OWNS data_path, and why - so a refusal can say which one it tried."""
    if explicit:
        want = str(explicit).lower()
        if want in ("object", "obj"):
            return obj, "object (explicit)"
        if want in ("data", "datablock"):
            if obj.data is None:
                raise MifOpError("target 'data' was asked for but '%s' has no data datablock. "
                                 "NOTHING was keyed." % obj.name)
            return obj.data, "data (explicit)"
        if want in ("shape_keys", "shapekeys", "keys"):
            if _shape_key_block(obj) is None:
                raise MifOpError("target 'shape_keys' was asked for but '%s' has no shape keys - "
                                 "add_shape_key makes them. NOTHING was keyed." % obj.name)
            return _shape_key_block(obj), "shape keys (explicit)"
        raise MifOpError("unknown target '%s' - use 'object', 'data' or 'shape_keys'. NOTHING was keyed."
                         % explicit)
    root = data_path.split(".")[0].split("[")[0]
    # key_blocks LIVES ON THE MESH'S KEY DATABLOCK, not on the mesh: routed there as the channels below are.
    if root == "key_blocks" and _shape_key_block(obj) is not None:
        return _shape_key_block(obj), "shape keys (key_blocks lives on the Key datablock)"
    # rigid_body, collision, cloth and friends hang off the OBJECT, and a dotted path rooted at one
    # of them is an object path however deep it goes.
    if root in _OBJECT_CHANNELS:
        return obj, "object (a transform channel lives on the object)"
    # A HINT NAMES A DATABLOCK THAT HAS THE PROPERTY, so it only routes when this object's data does.
    # It routed on `obj.data is not None` alone until 2026-09-16, which sent `color` - here for a
    # LIGHT's colour - to the Mesh of every mesh object, and keying Object.color (what an Object Info
    # node reads: the ordinary way to animate one shared material differently per object) was refused
    # with "color lives on the Mesh datablock". A Mesh has no color; the object the caller named does.
    # The attribute fallback below already knew that, and the hint returned before it could say so.
    if root in _DATA_HINTS and obj.data is not None and hasattr(obj.data, root):
        return obj.data, "data (%s lives on the %s datablock)" % (root, type(obj.data).__name__)
    # Fall back to whichever one actually HAS the attribute - resolved, not guessed.
    if hasattr(obj, root):
        return obj, "object (resolved by attribute)"
    if obj.data is not None and hasattr(obj.data, root):
        return obj.data, "data (resolved by attribute)"
    raise MifOpError(
        "neither '%s' nor its data has '%s', so there is nothing to key. Pass target explicitly if "
        "this is a nested path. NOTHING was keyed." % (obj.name, root))


def _apply_interpolation(owner, data_path, frame, mode):
    """Set interpolation on the points just written. LINEAR and CONSTANT are the two that matter:
    a flickering light on BEZIER eases between states and stops reading as a flicker at all."""
    n = 0
    for fc in _fcurves(owner):
        if fc.data_path != data_path:
            continue
        for kp in fc.keyframe_points:
            if abs(kp.co[0] - frame) < 1e-6:
                kp.interpolation = mode
                n += 1
    return n


def _rna_property(holder, leaf):
    """The RNA definition of holder.leaf, or None for a path bl_rna does not list."""
    if not hasattr(holder, "bl_rna"):
        return None
    return holder.bl_rna.properties.get(leaf)


def _rna_value(prop, value, path):
    """One value as the type the property stores, or a refusal naming the mismatch.

    THE PROPERTY'S TYPE DECIDES, NOT THE JSON'S. Every number used to go through finite_float, so an
    INT property - an array's count, a subdivision level, a particle count - answered "expected an int
    type, not float" and could never be keyed, and an ENUM such as a remesh mode was refused as "not a
    number" before Blender saw it. Float properties worked, which is why no suite noticed. Found
    2026-09-11 by the showcase renders. prop None (a path bl_rna does not list) keeps the old handling.
    """
    kind = getattr(prop, "type", None)
    if kind == "INT":
        num = None if isinstance(value, bool) else finite_float(value, "value")
        if num is None or not num.is_integer():
            raise MifOpError("'%s' is an INT property; pass a whole number, not %r - a fraction is "
                             "refused rather than truncated. NOTHING was keyed." % (path, value))
        return int(num)
    if kind == "BOOLEAN":
        if isinstance(value, bool) or (isinstance(value, (int, float)) and value in (0, 1)):
            return bool(value)
        raise MifOpError("'%s' is a BOOLEAN property; pass true or false, not %r. NOTHING was keyed."
                         % (path, value))
    if kind == "ENUM":
        # ASSIGNED EXACTLY, AND BLENDER ANSWERS. enum_items read off bl_rna lists a dynamic enum's
        # members without the instance and can be wrong (see add_dynamic_paint); the TypeError a bad
        # identifier raises names the members THIS property offers.
        if not isinstance(value, str):
            raise MifOpError("'%s' is an ENUM property; pass one of its identifiers as a string, not "
                             "%r. NOTHING was keyed." % (path, value))
        return value
    if isinstance(value, bool):
        return value
    return finite_float(value, "value")


# EULER KEYS ON SOMETHING THAT DOES NOT ROTATE BY EULER DO NOTHING. MEASURED 2026-09-16 on 3.6.23, 4.2.17, 4.4.0, 5.0.1
# and 5.2.1 (scratchpad probe_euler_keys_quaternion.py, probe_euler_keys_axis_angle.py): rotation_euler z keyed 0 -> 1.5
# over frames 1-10, read at frame 10 - an object in QUATERNION or AXIS_ANGLE mode turns 0.0 degrees, one in XYZ 85.944,
# and delta_rotation_euler is ignored the same way (ZXY with both keyed: 126.051; QUATERNION, AXIS_ANGLE: 0.0). A pose
# bone starts in QUATERNION mode, so a bone keyed through pose.bones["X"].rotation_euler never moves - set_bone_pose
# refuses Euler on such a bone and this op keyed it without a word (second seat, mailbox id 122). Through this op on
# every build: the quaternion object 0.0, the quaternion bone 0.0.
_EULER_PATHS = ("rotation_euler", "delta_rotation_euler")


def _euler_key_ignored(holder, leaf, label):
    """A sentence when keys on this euler channel will not move `holder`, else None."""
    mode = getattr(holder, "rotation_mode", None)
    if leaf not in _EULER_PATHS or mode not in ("QUATERNION", "AXIS_ANGLE"):
        return None
    other = "rotation_quaternion" if mode == "QUATERNION" else "rotation_axis_angle"
    if leaf.startswith("delta_"):
        other = "delta_" + other if mode == "QUATERNION" else "delta_rotation_euler with an Euler rotation_mode"
    route = (" (set_bone_pose rotationMode, which keeps the bone's current pose)"
             if getattr(getattr(bpy, "types", None), "PoseBone", None) is not None
             and isinstance(holder, bpy.types.PoseBone) else "")
    return ("%s rotates by %s, so these %s keys change nothing - measured on every build. Key %s instead, or give it an "
            "Euler rotation_mode first%s." % (label, mode, leaf, other, route))


def _modifier_input_path(obj, spec):
    """modifiers["M"]["<identifier>"] for {modifier, socket} - a NODES modifier's group input by its interface name or
    identifier. The rest of set_keyframe re-roots it on 5.2, where the input lives in mod.properties.inputs.

    WHY (spec, "Key a simulation zone's Skip input through a group input"): a socket inside a tree is keyed on the tree,
    and a zone's Skip is driven per object through a group input linked to it, keyed on the modifier. MEASURED
    2026-10-06 on 4.2.23 and 5.2.2: a Bool group input linked to a simulation zone's Skip, keyed false at 1, true at
    10 and false at 21 (CONSTANT), stepped a frame at a time - a zone adding 1 in Z a frame reads 10 over frames 10-20
    and resumes at 21. The identifier ('Socket_2') was the caller's to find; the name is what the interface shows."""
    if not isinstance(spec, dict) or set(spec) - {"modifier", "socket"} or not spec.get("socket"):
        raise MifOpError("modifierInput is {modifier, socket} - socket a group input's name or identifier, modifier "
                         "the NODES modifier's name (default the object's only one), got %r. NOTHING was keyed." % (spec,))
    nodes = [m for m in obj.modifiers if m.type == "NODES"]
    want = spec.get("modifier")
    mod = obj.modifiers.get(want) if want else (nodes[0] if len(nodes) == 1 else None)
    if mod is None or mod.type != "NODES":
        raise MifOpError("%s - modifierInput needs a geometry-nodes modifier. '%s' has: %s. NOTHING was keyed."
                         % ("no NODES modifier named '%s'" % want if want else "%d NODES modifiers, name one" % len(nodes),
                            obj.name, ", ".join("%s (%s)" % (m.name, m.type) for m in obj.modifiers) or "<none>"))
    group = mod.node_group
    # tree.interface from 4.0; 3.6 keeps a group's inputs in tree.inputs (its sockets name their type bl_socket_idname),
    # and reading interface there raised AttributeError out of set_keyframe (found 2026-10-10 on 3.6.23).
    if group is not None and hasattr(group, "interface"):
        items = [it for it in group.interface.items_tree
                 if getattr(it, "item_type", "") == "SOCKET" and it.in_out == "INPUT"]
    else:
        items = list(group.inputs) if group is not None else []
    sock = spec["socket"]
    hit = [it for it in items if it.identifier == sock] or [it for it in items if it.name == sock]
    if len(hit) != 1:
        raise MifOpError("modifier '%s' has %s input '%s'. Its inputs: %s. NOTHING was keyed."
                         % (mod.name, "no" if not hit else "%d inputs named" % len(hit), sock,
                            ", ".join("%s (%s)" % (it.name, it.identifier) for it in items) or "<none>"))
    item = hit[0]
    item_type = getattr(item, "socket_type", None) or getattr(item, "bl_socket_idname", "")
    if item_type in ("NodeSocketGeometry", "NodeSocketObject", "NodeSocketCollection", "NodeSocketMaterial",
                            "NodeSocketImage", "NodeSocketTexture"):
        raise MifOpError("input '%s' is a %s, which no key can hold. NOTHING was keyed." % (item.name, item_type))
    return 'modifiers["%s"]["%s"]' % (mod.name, item.identifier), "%s.%s is %s" % (mod.name, item.name,
                                                                                  item.identifier)


def op_set_keyframe(params):
    """Write a value and key it at a frame. Returns the keyframe count read back off the fcurves.

    params:
      object (str, required)
      frame (int, required)
      location / rotation / scale   {x,y,z} - keyed on the OBJECT
      dataPath (alias path) + value - any other property; `value` may be a number, bool, enum
                                    identifier or list, converted to the property's own type (an
                                    INT refuses 2.5 rather than truncating it)
      index (int)                   which array element to key, default all
      target                        object | data | shape_keys - only needed for a path this cannot place.
                                    A path from the mesh into its shape keys (shape_keys.key_blocks["X"].value)
                                    is re-rooted onto the Key datablock, where Blender keeps shape-key animation
      interpolation                 CONSTANT | LINEAR | BEZIER (default BEZIER, Blender's own)
      modifierInput {modifier, socket}  in place of dataPath: a geometry-nodes modifier's group input by its name
                                    or identifier (modifier defaults to the object's only NODES one)
    """
    reject_unknown(params, _KEY_KEYS, "set_keyframe")
    sub = _tree_subject(params, "set_keyframe", "NOTHING was keyed.")
    if sub is None:
        obj = get_object(take(params, "object", "name", required=True))
        require_editable(obj, "key")
    frame = take_float(params, "frame", required=True)
    interp = str(take(params, "interpolation", default="BEZIER", kind=str)).upper()
    if interp not in ("CONSTANT", "LINEAR", "BEZIER"):
        raise MifOpError("interpolation must be CONSTANT, LINEAR or BEZIER, got '%s'. NOTHING was "
                         "keyed." % interp)
    if sub is not None:
        return _key_tree_value(params, sub, frame, interp)

    transforms = {"location": "location", "rotation": "rotation_euler", "scale": "scale"}
    asked = [k for k in transforms if k in params]
    path = take(params, "dataPath", "path", default=None, kind=str)
    mod_input = params.get("modifierInput")
    socket_note = None
    if mod_input is not None:
        if path or asked:
            raise MifOpError("modifierInput names the property itself - drop dataPath and the transform channels. "
                             "NOTHING was keyed.")
        path, socket_note = _modifier_input_path(obj, mod_input)
    if not asked and not path:
        raise MifOpError("nothing to key - pass location, rotation, scale, or dataPath+value. "
                         "NOTHING was keyed.")
    if asked and path:
        raise MifOpError("pass transform channels OR dataPath, not both - they key different "
                         "things and combining them hides which one failed. NOTHING was keyed.")

    written = []
    if asked:
        for key in asked:
            dp = transforms[key]
            setattr(obj, dp, _vec3(params, key))
            obj.keyframe_insert(data_path=dp, frame=frame)
            # THE SAME COUNT THE dataPath BRANCH KEEPS. It was fixed there and not here,
            # which left the branch this op's docstring LEADS with - location, rotation,
            # scale, the ordinary way anybody keys anything - reporting no per-call
            # evidence at all. keyframesTotal below cannot stand in for it: it sums every
            # fcurve on the object and its data, so any PRIOR key keeps it non-zero and it
            # cannot fail. A read/write asymmetry inside one function.
            keyed_here = _apply_interpolation(obj, dp, frame, interp)
            written.append({"target": "object", "dataPath": dp,
                            "keysAtThisFrame": keyed_here})
            ignored = _euler_key_ignored(obj, dp, "'%s'" % obj.name)
            if ignored:
                written[-1]["rotationModeNote"] = ignored
    else:
        owner, why = _resolve_target(obj, path, take(params, "target", default=None, kind=str))
        if "value" not in params:
            raise MifOpError("dataPath needs a `value` - keyframe_insert stores the property's "
                             "CURRENT value, so there is nothing to record without one. NOTHING "
                             "was keyed.")
        value = params.get("value")
        index = take_int(params, "index", default=None)  # take_int, NOT int(): a bad value must REFUSE, not raise ValueError out of the op.
        # A DOTTED PATH NEEDS WALKING. keyframe_insert takes "rigid_body.kinematic" happily, but
        # setattr does not - it would look for an attribute literally named that and fail with
        # "'Object' object has no attribute 'kinematic'", which names the leaf and hides that the
        # problem was the walk. Found keying a rigid body's kinematic flag, which is the ordinary
        # way to hold an object still and then hand it to the simulation.
        holder, leaf = owner, path
        # A PATH WHOSE LAST STEP IS A SUBSCRIPT, added 2026-09-13. The split below is on a DOT, so
        # modifiers["GeometryNodes"]["Socket_2"] - a geometry-node group's exposed value, which has no
        # dot in it at all - was never split: holder stayed the object and the write became
        # setattr(obj, 'modifiers["GeometryNodes"]["Socket_2"]', v), which is the "'Object' object has
        # no attribute" the caller saw. The comment below already names "modifiers by name, node
        # inputs" among the bracketed paths it set out to fix; it fixed the ones with a dot after the
        # bracket and not the ones that END in one.
        #
        # THE LEAF IS A KEY, NOT AN ATTRIBUTE, which is why this is its own branch rather than a
        # cleverer split: the holder is an IDPropertyGroup and the write is holder[key] = value, and
        # it has no RNA property to check is_animatable against.
        item_key = None
        if path.endswith("]") and "[" in path:
            cut = path.rfind("[")
            if cut > 0 and "." not in path[cut:]:
                head_sub, raw = path[:cut], path[cut + 1:-1].strip()
                if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
                    try:
                        holder = owner.path_resolve(head_sub)
                    except (ValueError, AttributeError, TypeError) as exc:
                        raise MifOpError("could not resolve '%s' on '%s' (%s). Check the path against "
                                         "Blender's own - right-click a field and Copy Full Data Path, "
                                         "then drop the object prefix. NOTHING was keyed."
                                         % (head_sub, obj.name, exc))
                    item_key, leaf = raw[1:-1], raw[1:-1]
                    # BLENDER 5.2 MOVED A GEOMETRY-NODE SOCKET OUT FROM UNDER THE MODIFIER, and the
                    # old path does not degrade - it raises. Up to 5.0 the input was a flat
                    # id-property (`mod["Socket_2"]`); on 5.2 the NodesModifier supports none at all
                    # and the value lives in mod.properties.inputs[identifier]["value"].
                    #
                    # ops_nodes already carries that measurement across all five builds and uses it
                    # in assign_node_group; this path never got the same treatment, so keying a
                    # socket answered "could not write 0.0 to modifiers[...][...]" on 5.2 while
                    # SETTING one worked. The helper is imported rather than re-implemented.
                    #
                    # THE PATH IS REWRITTEN TOO, not just the holder, because keyframe_insert takes
                    # the full path and resolves it itself. Measured on 5.2.1: the bracketed form is
                    # the one it accepts - `...inputs["Socket_0"]["value"]` keys, while
                    # `...inputs["Socket_0"].value` and the bare `["Socket_0"]` are both "property
                    # not found".
                    from .ops_nodes import _modifier_socket_group, _MOD_SOCKET_VALUE_KEY
                    group = _modifier_socket_group(holder, item_key)
                    if group is not None:
                        holder = group
                        item_key = leaf = _MOD_SOCKET_VALUE_KEY
                        path = '%s.properties.inputs["%s"]["%s"]' % (head_sub, raw[1:-1],
                                                                     _MOD_SOCKET_VALUE_KEY)
        if item_key is None and "." in path:
            head, leaf = path.rsplit(".", 1)
            # RESOLVED BY BLENDER, NOT BY A HAND-ROLLED WALK. The previous version did
            # `getattr(holder, part.split("[")[0])`, which STRIPS a subscript and never puts it
            # back: for pose.bones["hand"].location it walked to the bones COLLECTION and then
            # tried to set `location` on it. Every bracketed path was therefore unwritable, which
            # is most of the interesting ones - bones, modifiers by name, shape keys, node inputs,
            # constraints, particle settings. keyframe_insert below never had the problem because
            # it takes the FULL path and Blender resolves it, so the op could key a path it could
            # not write, and the caller got a keyframe holding the OLD value.
            #
            # path_resolve does the same resolution Blender does everywhere else, subscripts
            # included. It raises ValueError for a path that does not exist, which is a better
            # error than AttributeError naming only the leaf.
            try:
                holder = owner.path_resolve(head)
            except (ValueError, AttributeError, TypeError) as exc:
                raise MifOpError("could not resolve '%s' on '%s' (%s). Check the path against "
                                 "Blender's own - right-click a field and Copy Full Data Path, "
                                 "then drop the object prefix. NOTHING was keyed."
                                 % (head, obj.name, exc))
            if holder is None:
                raise MifOpError("'%s' is None on '%s', so '%s' cannot be written. A rigid "
                                 "body path needs add_rigid_body to have run first. NOTHING "
                                 "was keyed." % (head, obj.name, path))
            # A PATH THAT RUNS INTO ANOTHER DATABLOCK. Blender keys a property only on the ID that owns it
            # ("keyframe_insert() path spans ID blocks") and said so only AFTER the value had been written - a
            # refusal that changed the scene. shape_keys.key_blocks["X"].value from a mesh is the ordinary case,
            # and every shape key lives on the mesh's Key datablock, so that one is re-rooted there with the path
            # Blender itself gives from that ID. Anything else is refused BEFORE the write. Found 2026-09-13 by the
            # morph showcase: a shape key could be made and set, and not animated.
            hid = getattr(holder, "id_data", None)
            if hid is not None and hid != owner:
                if hid == _shape_key_block(obj):
                    owner, why = hid, "shape keys (the path runs into the mesh's Key datablock)"
                    path = holder.path_from_id(leaf)
                else:
                    raise MifOpError("'%s' runs from %s '%s' into %s '%s', and Blender keys a property only "
                                     "on the datablock that owns it - key that datablock's own path. NOTHING "
                                     "was keyed." % (path, type(owner).__name__, owner.name,
                                                     type(hid).__name__, hid.name))
        prop = None if item_key is not None else _rna_property(holder, leaf)
        # REFUSED BEFORE THE WRITE, not after it. keyframe_insert raises for a property Blender will
        # not animate, and by then the value has been written - a refusal that changed the scene.
        # Enums made this matter: they are reachable now, and plenty of them are not animatable.
        if prop is not None and not prop.is_animatable:
            raise MifOpError("'%s' can be set but not animated - Blender does not allow keying "
                             "%s.%s. NOTHING was keyed." % (path, holder.bl_rna.identifier, leaf))
        # CONVERTED BEFORE ANYTHING IS WRITTEN, so a refusal leaves the property as it was.
        if index is not None:
            want = _rna_value(prop, value, path)
        elif isinstance(value, (list, tuple)):
            want = tuple(_rna_value(prop, v, path) for v in value)
            if prop is not None and prop.type == "ENUM" and prop.is_enum_flag:
                want = set(want)
        else:
            want = _rna_value(prop, value, path)
        try:
            if item_key is not None:
                # AN ITEM, NOT AN ATTRIBUTE. A modifier's exposed socket lives in the modifier's
                # IDProperty mapping, so this is holder[key] = value; setattr would miss it entirely.
                if index is not None:
                    holder[item_key][int(index)] = want
                else:
                    holder[item_key] = want
            elif index is not None:
                getattr(holder, leaf)[int(index)] = want
            else:
                setattr(holder, leaf, want)
        except (AttributeError, TypeError, ValueError) as exc:
            raise MifOpError("could not write %r to '%s' on the %s datablock: %s. NOTHING was "
                             "keyed." % (value, path, why, exc))
        kwargs = {"data_path": path, "frame": frame}
        if index is not None:
            kwargs["index"] = int(index)
        try:
            owner.keyframe_insert(**kwargs)
        except (ValueError, TypeError, RuntimeError) as exc:
            # A SENTENCE, NOT A RAW EXCEPTION. The cross-datablock case is caught above, before the write;
            # this is the net for anything Blender refuses that nothing here predicted - and it says plainly
            # that the value is already written.
            raise MifOpError("Blender would not key '%s' on the %s datablock: %s. The value WAS written; no "
                             "keyframe exists." % (path, why, exc))
        # THE COUNT IS KEPT. _apply_interpolation returns how many keys it actually found AT THIS
        # FRAME on this path, and it was being discarded - so the only per-call measurement that
        # this keyframe exists went in the bin, leaving keyframesTotal below as the only evidence.
        # That total sums every fcurve on the object and its data, so any PRIOR key keeps it
        # non-zero and it cannot fail. Same shape as op_transfer_weights counting group membership.
        keyed_here = _apply_interpolation(owner, path, frame, interp)
        written.append({"target": why, "dataPath": path, "keysAtThisFrame": keyed_here})
        ignored = _euler_key_ignored(holder, leaf, "'%s'" % path.rsplit(".", 1)[0] if "." in path else "'%s'" % obj.name)
        if ignored:
            written[-1]["rotationModeNote"] = ignored

    # READ BACK off the fcurves. keyframe_insert returns a bool, and a True that produced no curve
    # is exactly the silent success this bridge exists to refuse.
    total = 0
    curves = []
    for holder in (obj, obj.data if obj.data is not None else None, _shape_key_block(obj)):
        if holder is None:
            continue
        for fc in _fcurves(holder):
            total += len(fc.keyframe_points)
            curves.append({"dataPath": fc.data_path, "index": fc.array_index,
                           "keyframes": len(fc.keyframe_points)})
    return {
        "object": obj.name,
        "frame": frame,
        "written": written,
        "interpolation": interp,
        "fcurves": curves,
        "keyframesTotal": total,
        # WHAT THIS CALL DID, as opposed to what the object now holds. keyframesTotal is a scene
        # measurement and cannot fall to zero once anything has ever been keyed, so on its own it
        # can never report a failure. This is the per-call number, summed from the paths written
        # above, and it IS zero when a call keyed nothing.
        "keysWrittenAtThisFrame": sum(w.get("keysAtThisFrame") or 0 for w in written),
        # ONE PLACE TO LOOK: the per-path notes above, gathered, or None. See _euler_key_ignored.
        "rotationModeNote": " ".join(w["rotationModeNote"] for w in written if w.get("rotationModeNote")) or None,
        "valueNote": ("keyframe_insert stores the property's CURRENT value, so the value passed was "
                      "WRITTEN to the object before being keyed - the object is left holding it."),
        "modifierInput": socket_note,
    }


def _fit_to_strips(sc, params, fit):
    """fitToStrips SCENE | PREVIEW: the scene (or preview) range set to the earliest first frame and the latest last
    frame over every strip in this scene's sequencer, metas' contents included - what sequencer.set_range_to_strips
    (preview false / true) sets, without the Sequencer area that operator polls for. Returns params with the pair
    filled in, so the range goes through the same checks and writes as one given by hand."""
    from .ops_video import strips_range
    which = str(fit).upper() if isinstance(fit, str) else ("SCENE" if fit is True else None)
    if which not in ("SCENE", "PREVIEW"):
        raise MifOpError("fitToStrips is SCENE or PREVIEW (true means SCENE), got %r. NOTHING was changed." % (fit,))
    keys = ("start", "end", "frameStart", "frameEnd") if which == "SCENE" else ("previewStart", "previewEnd")
    clash = [k for k in keys if params.get(k) is not None]
    if clash:
        raise MifOpError("fitToStrips %s sets the %s range from the strips - %s would fight it. NOTHING was changed."
                         % (which, which.lower(), ", ".join(clash)))
    rng = strips_range(sc)
    if rng is None:
        raise MifOpError("scene '%s' has no strips in its sequencer, so there is no range to fit to. NOTHING was "
                         "changed." % sc.name)
    out = {k: v for k, v in params.items() if k != "fitToStrips"}
    if which == "SCENE":
        out["start"], out["end"] = rng
    else:
        out["previewStart"], out["previewEnd"] = rng
    return out


def op_set_frame_range(params):
    """Scene frame range, fps and current frame. Reports what the scene ACTUALLY holds. fitToStrips SCENE | PREVIEW
    sets the scene or preview range to the sequencer's strips (see _fit_to_strips)."""
    reject_unknown(params, _RANGE_KEYS, "set_frame_range")
    sc = bpy.context.scene
    fit = params.get("fitToStrips")
    if fit is not None:
        params = _fit_to_strips(sc, params, fit)
    before = {"start": sc.frame_start, "end": sc.frame_end, "fps": sc.render.fps,
              "current": sc.frame_current}
    start = take_float(params, "start", "frameStart", default=None)
    end = take_float(params, "end", "frameEnd", default=None)

    # CHECKED BEFORE ANYTHING IS WRITTEN, and the first version checked afterwards and did not
    # work. Blender CLAMPS these against each other as they are assigned: setting frame_end to 10
    # while frame_start is 90 drags frame_start down to 10 as a side effect, so by the time the
    # resulting state was inspected it read 10..10 - consistent, accepted, and renders a single
    # frame. The bad input had already been swallowed by the clamp.
    #
    # So the REQUESTED range is validated against itself, falling back to the current value for
    # whichever end was not supplied. Found by test_blender_anim A102 on its first run.
    want_start = finite_int(start, "start") if start is not None else before["start"]
    want_end = finite_int(end, "end") if end is not None else before["end"]
    if want_end < want_start:
        raise MifOpError("end (%d) is before start (%d), which Blender does not reject - it CLAMPS "
                         "one to the other as you assign them, leaving a range that renders a "
                         "single frame or nothing. NOTHING was changed; the range is still %d..%d."
                         % (want_end, want_start, before["start"], before["end"]))

    # EVERYTHING ELSE IS READ AND CHECKED HERE TOO, before the range is written. fps, fpsBase, frameStep, the
    # preview pair, usePreviewRange and current were each read below the range write, so a bad one was refused
    # with the range already moved: {start: 10, end: 20, frameStep: 0} left the scene at 10..20 under ok:false,
    # and so did fps 24.5, fpsBase -1, current Infinity and usePreviewRange "flase" (measured on 5.0.1,
    # 2026-09-21). And the preview pair went through a bare int(): previewStart 2.5 was stored as 2.
    try:
        fps = take_float(params, "fps", default=None)
        want_fps = finite_int(fps, "fps") if fps is not None else None
        fps_base = take_float(params, "fpsBase", default=None)
        step = take_float(params, "frameStep", default=None)
        want_step = finite_int(step, "frameStep") if step is not None else None
        pv_start = take_float(params, "previewStart", default=None)
        pv_end = take_float(params, "previewEnd", default=None)
        want_pv_start = finite_int(pv_start, "previewStart") if pv_start is not None else None
        want_pv_end = finite_int(pv_end, "previewEnd") if pv_end is not None else None
        use_preview = (take_bool(params, "usePreviewRange", default=True)
                       if "usePreviewRange" in params else None)
        cur = take_float(params, "current", default=None)
        want_cur = finite_int(cur, "current") if cur is not None else None
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was changed." % str(exc).rstrip("."))
    if fps_base is not None and fps_base <= 0:
        raise MifOpError("fpsBase must be positive, got %g. NOTHING was changed." % fps_base)
    if want_step is not None and want_step < 1:
        raise MifOpError("frameStep must be at least 1, got %g. NOTHING was changed." % step)
    if want_pv_start is not None or want_pv_end is not None:
        ws = want_pv_start if want_pv_start is not None else sc.frame_preview_start
        we = want_pv_end if want_pv_end is not None else sc.frame_preview_end
        if we < ws:
            raise MifOpError("previewEnd (%d) is before previewStart (%d). NOTHING was changed."
                             % (we, ws))
    # PLAYBACK SYNC (2026-10-04): scene.sync_mode, NONE | FRAME_DROP | AUDIO_SYNC on 3.6.23, 4.4.0 and 5.2.1 - read from
    # the build's RNA, not typed here.
    sync = params.get("syncMode")
    if sync is not None:
        ids = [i.identifier for i in bpy.types.Scene.bl_rna.properties["sync_mode"].enum_items]
        if not isinstance(sync, str) or sync.upper() not in ids:
            raise MifOpError("syncMode must be one of %s on this Blender, got %r. NOTHING was changed."
                             % (", ".join(ids), sync))
        sync = sync.upper()
    sync_before = sc.sync_mode
    # A NEW CURRENT FRAME STEPS EVERY LIVE SIMULATION (ops_common.simulation_guard): refused here, before the range is
    # written, unless allowSimulation says the step is meant.
    sim_reply = simulation_guard(params, [want_cur] if want_cur is not None else [], "set_frame_range current")

    # Widen first, then narrow, so the clamp never fights a legitimate move: assigning end before
    # start when both are increasing would otherwise drag start along with it.
    if want_end >= before["end"]:
        sc.frame_end = want_end
        sc.frame_start = want_start
    else:
        sc.frame_start = want_start
        sc.frame_end = want_end
    # WHAT WAS ASKED FOR, kept so the response can say whether it landed. See the `clamped` block
    # at the bottom: the previewStart defect was invisible for exactly as long as it was because
    # every field in the response read back consistently with ITSELF, and none of them read back
    # against the request.
    asked_frames = {"start": want_start, "end": want_end}
    if want_fps is not None:
        sc.render.fps = want_fps
    # FPS_BASE IS THE OTHER HALF OF THE FRAME RATE and nothing here could set it. Blender stores
    # 29.97 as fps 30 with fps_base 1.001, and 23.976 as 24 with 1.001 - so every broadcast rate is
    # unreachable through `fps` alone, and a caller asking for 30 on an NTSC scene silently got
    # 29.97 with no way to tell or to change it.
    if fps_base is not None:
        sc.render.fps_base = fps_base
    if want_step is not None:
        sc.frame_step = want_step

    # PREVIEW RANGE, which overrides the scene range for PLAYBACK (and viewport renders - RNA's own words) when it is on,
    # and NOT for a render. This comment said a scene with one enabled "renders the preview range" until 2026-09-28, and
    # rendersFrames below reported the preview pair to match (docs/22 B26's review note). MEASURED 2026-09-28 on 3.6.23,
    # 4.2.17 and 5.2.1, plain bpy, scene 1..6 with preview 3..4 on: render.render(animation=True), `blender -b file -a` and
    # render_animation's own `-s 1 -e 6 -a` all wrote frames 1-6 (tools/probe_preview_range_render.py; docs/22 B253).
    if want_pv_start is not None or want_pv_end is not None:
        # ws / we and their order were settled above, before anything was written.
        # ENABLE THE FLAG BEFORE WRITING THE PAIR. This is NOT the same bug as the main range's, and
        # ordering alone does not fix it. Blender's two preview setters each RE-SEED THE OTHER END
        # FROM THE SCENE RANGE while SCER_PRV_RANGE is off:
        #
        #     frame_preview_start = v   ->  if preview off: pefra = efra;  psfra = min(v, pefra)
        #     frame_preview_end   = v   ->  if preview off: psfra = sfra;  pefra = max(v, psfra)
        #
        # So with the flag still off, whichever end is written SECOND wipes the one written first.
        # `sc.frame_preview_start, sc.frame_preview_end = ws, we` asked for 100..200 on a 1..250
        # scene and stored (1, 200) - the end's setter had reset the start to sfra. `ok` was true,
        # nothing warned, and the preview pair came back (1, 200), a range the caller never asked
        # for, which then PLAYS instead of frame_start..frame_end.
        #
        # Measured 2026-09-15 on 4.2.17 and 5.0.1: reordering the two assignments changed nothing,
        # because no order survives a setter that re-seeds its partner. Setting the flag first does.
        # Found by the second seat, held by B1153. Their diagnosis named Python's left-to-right
        # tuple assignment and the main range's clamp; the value really was dropped, but the
        # mechanism is this one, and the widen-first branch they pointed at is not the fix.
        sc.use_preview_range = True
        # Widen first ANYWAY, for the case the flag being on does not cover: with SCER_PRV_RANGE
        # set, `psfra = min(ws, pefra)` still clamps a start that is above the stored end.
        if we >= sc.frame_preview_end:
            sc.frame_preview_end = we
            sc.frame_preview_start = ws
        else:
            sc.frame_preview_start = ws
            sc.frame_preview_end = we
        asked_frames["previewStart"] = ws
        asked_frames["previewEnd"] = we
    if use_preview is not None:
        sc.use_preview_range = use_preview
    if sync is not None:
        sc.sync_mode = sync

    if want_cur is not None:
        sc.frame_set(want_cur)

    # THE TRUE RATE IS fps / fps_base, and durationSeconds divided by fps alone until 2026-09-03 -
    # so every NTSC rate was reported 0.1% short. Small, and exactly the kind of small that makes a
    # 60-minute programme land a frame and a half out against audio.
    base = float(sc.render.fps_base) or 1.0
    effective_fps = float(sc.render.fps) / base
    span = (sc.frame_end - sc.frame_start + 1)

    # DID THE FOUR FRAME NUMBERS ACTUALLY LAND? Blender clamps each of these against its partner as
    # you assign it, and a clamp is not an error - it returns cleanly and every other field in this
    # response then agrees with the clamped value, so the response is self-consistent and wrong.
    # That is what hid the previewStart write-order defect. Ordering the writes fixes the case we
    # know about; reading the request back makes the whole CLASS loud instead of silent, including
    # any clamp this addon has not predicted.
    #
    # `clamped` is the name ops_material already uses for "what did not land" and suites read it,
    # so it keeps that name here. None when everything landed, which is the normal case.
    stored_frames = {"start": sc.frame_start, "end": sc.frame_end,
                     "previewStart": sc.frame_preview_start, "previewEnd": sc.frame_preview_end}
    clamped = {k: {"requested": v, "stored": stored_frames[k],
                   "why": "Blender clamped this against the other end of its pair as it was "
                          "assigned. The value asked for is NOT what the scene holds."}
               for k, v in asked_frames.items() if stored_frames[k] != v}
    sync_reply = {}
    if sync is not None:
        sync_reply = {"syncMode": sc.sync_mode, "syncModeBefore": sync_before,
                      "syncNote": ("syncMode changes interactive playback only (FRAME_DROP skips frames to keep real "
                                   "time, AUDIO_SYNC follows the sound) - renders and bakes evaluate every frame.")}
    return dict({"before": before,
            "clamped": clamped or None,
            "after": {"start": sc.frame_start, "end": sc.frame_end,
                      "fps": sc.render.fps, "current": sc.frame_current},
            "fpsBase": round(base, 6),
            "effectiveFps": round(effective_fps, 6),
            "frameStep": sc.frame_step,
            "usePreviewRange": bool(sc.use_preview_range),
            "previewStart": sc.frame_preview_start,
            "previewEnd": sc.frame_preview_end,
            "durationSeconds": round(span / effective_fps, 6),
            # WHAT A RENDER USES: frame_start..frame_end, whether or not a preview range is on - see the PREVIEW RANGE
            # comment above for the measurement. It reported the preview pair while one was on until 2026-09-28; the
            # pair is in previewStart / previewEnd, and usePreviewRange says whether playback uses it.
            "rendersFrames": [sc.frame_start, sc.frame_end]}, **sync_reply, **sim_reply)


# WHICH HOLDERS `target` SELECTS, and a refusal for a word this addon does not know.
#
# Both list ops used to test membership inline and silently select nothing for an unrecognised
# value, so target:"objekt" came back as curveCount 0 with ok:true - indistinguishable from an
# object that genuinely has no animation. take_bool's own comment states the rule this breaks: "a
# false is a decision; a typo is not".
_ANIM_TARGETS = {
    # "all" IS THE DEFAULT since 2026-09-13. Shape-key animation lives on the Key datablock, a third holder, and
    # a reader that looked at the object and its data only reported an animated shape key as no animation.
    "all": ("object", "data", "shape_keys"),
    "shape_keys": ("shape_keys",),
    "keys": ("shape_keys",),
    "both": ("object", "data"),
    "object": ("object",),
    "obj": ("object",),
    "data": ("data",),
    "datablock": ("data",),
}


def _anim_target(params):
    """The holder labels `target` names, or a refusal listing what it accepts."""
    want = str(take(params, "target", default="all", kind=str)).lower()
    if want not in _ANIM_TARGETS:
        raise MifOpError(
            "'target' must be one of %s (got %r). It selects WHICH datablock to read - the object "
            "or its mesh/curve data - and an unrecognised word used to select neither, which "
            "reported no animation rather than refusing. NOTHING was read."
            % (", ".join(sorted(_ANIM_TARGETS)), want))
    return _ANIM_TARGETS[want]


def op_list_keyframes(params):
    """Every fcurve on an object and/or its data, with the frames actually stored.

    The read half. Without it `set_keyframe` can only be trusted on its own word, and this repo's
    standing rule is that a write is not verified by the writer.
    """
    reject_unknown(params, _KEYFRAME_LIST_KEYS, "list_keyframes")
    sub = _tree_subject(params, "list_keyframes", "")
    obj = sub if sub is not None else get_object(take(params, "object", "name", required=True))
    wants = {} if sub is not None else _anim_target(params)
    out = []
    holders = list(_anim_holders(sub)) if sub is not None else []
    if "object" in wants:
        holders.append(("object", obj))
    if "data" in wants and obj.data is not None:
        holders.append(("data", obj.data))
    if "shape_keys" in wants and _shape_key_block(obj) is not None:
        holders.append(("shape_keys", _shape_key_block(obj)))
    # maxKeyframes IS THE ONE THAT MATTERS. Measured 2026-09-14 at 15.2 bytes a keyframe: a ten-second
    # three-channel bake is ~2,700 tokens and a 60-channel rig over 3,000 frames is ~686,000 - in one reply.
    # A `limit` on the curves alone would not touch that, because the explosion is INSIDE a row.
    max_keys = params.get("maxKeyframes")
    if max_keys is not None:
        try:
            # finite_int: int() raised a raw OverflowError for Infinity, past this except, and took 2.5 as 2
            # (5.0.1, 2026-09-20; audit_unguarded_numbers' provenance pass).
            max_keys = finite_int(max_keys, "maxKeyframes")
        except MifOpError:
            raise MifOpError("'maxKeyframes' must be a whole number of keyframes, got %r." % (max_keys,))
        if max_keys < 0:
            raise MifOpError("'maxKeyframes' cannot be negative, got %d." % max_keys)
    for label, holder in holders:
        for fc in _fcurves(holder):
            points = fc.keyframe_points
            kept = points if max_keys is None else points[:max_keys]
            row = {
                "target": label,
                "dataPath": fc.data_path,
                "index": fc.array_index,
                # THE COUNT IS OF THE WHOLE CURVE, not of what came back. A caller trimming the answer still
                # has to be told how long the curve really is, or a 3,000-key bake reads as a 10-key one.
                "keyframeCount": len(points),
                "keyframes": [rnd([kp.co[0], kp.co[1]]) for kp in kept],
                # INTERPOLATION IS READ OFF THE WHOLE CURVE TOO. It is a summary of the curve, not of the
                # sample - "these keys are all LINEAR" must not become true because the caller asked for two.
                "interpolation": sorted({kp.interpolation for kp in points}),
            }
            if max_keys is not None and len(kept) < len(points):
                row["keyframesTruncated"] = True
                row["keyframesReturned"] = len(kept)
            out.append(row)
    # THE EXCLUSION IS ON THE dataPath, which is what names a channel - "pose.bones[\"DEF_spine\"].location"
    # or "key_blocks[...].value". A rig's deform channels are most of an action and almost never the ones
    # being asked about, the same argument as nameExcludes on list_vertex_groups.
    out = exclude_rows(out, params, name_of=lambda r: r.get("dataPath", ""))
    matched_curves = len(out)
    # keyframeTotal counts every key on every curve that MATCHED, never on the rows returned and never on the
    # trimmed samples inside them - it is the size of the answer the caller did not ask for in full.
    total_keys = sum(r.get("keyframeCount", len(r.get("keyframes") or [])) for r in out)
    out, extra = limit_rows(out, params, "curve(s)")
    return dict({"object": obj.name, "curves": out, "curveCount": matched_curves,
            "keyframeTotal": total_keys,
            # SAYS WHAT IT DID NOT LOOK AT. This walks animation_data.action only, so an object
            # animated entirely by drivers or by NLA strips reports curveCount 0 here and is not
            # un-animated. Anyone reading a zero needs to know which question was asked.
            "scopeNote": ("Action fcurves only. Drivers and NLA strips are separate collections "
                          "and are NOT counted here - use list_animation_data for all three.")},
        **extra)


def _anim_summary(label, holder):
    """Every route by which `holder` can be animated, not just the one list_keyframes reads.

    THE REASON THIS EXISTS. list_keyframes walks animation_data.action, which is one of THREE
    places animation lives. Drivers are on animation_data.drivers and NLA strips on
    animation_data.nla_tracks, both separate collections. So an object driven entirely by drivers -
    a rig control, a procedural offset, anything wired rather than keyed - came back with
    curveCount 0 from an op whose entire purpose is verification. That is not a missing answer, it
    is a WRONG one, and a caller checking whether their driver landed was told it had not.
    """
    ad = getattr(holder, "animation_data", None)
    if ad is None:
        return {"target": label, "hasAnimationData": False, "animatedBy": []}

    action = getattr(ad, "action", None)
    action_curves = len(_fcurves(holder)) if action is not None else 0

    drivers = []
    for dr in (getattr(ad, "drivers", None) or []):
        try:
            var_targets = []
            for var in (dr.driver.variables or []):
                for tgt in (var.targets or []):
                    if tgt.id is not None:
                        var_targets.append(tgt.id.name)
            drivers.append({
                "dataPath": dr.data_path,
                "index": dr.array_index,
                "expression": getattr(dr.driver, "expression", ""),
                "type": getattr(dr.driver, "type", None),
                # A driver whose variable points at a deleted object is the silent failure mode -
                # it evaluates to zero and reports nothing.
                "isValid": bool(getattr(dr.driver, "is_valid", True)),
                "variableTargets": sorted(set(var_targets)),
            })
            drivers[-1].update(_driver_curve(dr))
        except (AttributeError, TypeError):
            continue

    tracks = []
    for tr in (getattr(ad, "nla_tracks", None) or []):
        tracks.append({
            "name": tr.name,
            "mute": bool(tr.mute),
            # SOLO SILENCES EVERY OTHER TRACK while each of them still reports mute:false.
            # Measured on all four builds: two tracks moving a cube to -5 and +5 evaluate to
            # -5 with both live and +5 with the first soloed, and the second track says
            # nothing about it. Reporting mute alone made this op actively misleading -
            # every track unmuted, one playing, no field that could explain it.
            "isSolo": bool(tr.is_solo),
            "lock": bool(tr.lock),
            # BLEND AND INFLUENCE WERE NOT HERE, so nothing could reveal that a strip's influence was not in effect
            # (second seat, mailbox id 79). effectiveInfluence is what plays: 1.0 unless animated influence is on.
            "strips": [{"name": s.name,
                        "action": s.action.name if s.action else None,
                        "frameStart": round(float(s.frame_start), 4),
                        "frameEnd": round(float(s.frame_end), 4),
                        "blendType": s.blend_type,
                        "influence": round(float(s.influence), 6),
                        "animatedInfluence": bool(s.use_animated_influence),
                        "effectiveInfluence": (round(float(s.influence), 6) if s.use_animated_influence
                                               else 1.0)} for s in (tr.strips or [])],
        })

    by = []
    if action_curves:
        by.append("action")
    if drivers:
        by.append("drivers")
    if tracks:
        by.append("nla")
    return {
        "target": label,
        "hasAnimationData": True,
        "actionName": action.name if action is not None else None,
        # An action with a fake user survives a save; one without is DELETED on save, which is
        # data loss nobody is warned about anywhere else.
        "actionHasFakeUser": bool(getattr(action, "use_fake_user", False)) if action else False,
        "actionCurveCount": action_curves,
        "driverCount": len(drivers),
        "drivers": drivers,
        "nlaTrackCount": len(tracks),
        "nlaTracks": tracks,
        "animatedBy": by,
    }


def _variable_range(drv):
    """(lo, hi, where) - the range the driver's input runs over: the UI range of the one custom property its one
    variable reads, else (0, 1) and says so. A Rigify IK/FK switch is such a property, 0 to 1."""
    if len(drv.variables) == 1 and drv.variables[0].type == "SINGLE_PROP":
        tgt = drv.variables[0].targets[0]
        path = tgt.data_path or ""
        if tgt.id is not None and path.endswith('"]') and '["' in path:
            head, key = path[:path.rfind('["')], path[path.rfind('["') + 2:-2]
            try:
                owner = tgt.id.path_resolve(head) if head else tgt.id
                ui = owner.id_properties_ui(key).as_dict()
                lo, hi = ui.get("min"), ui.get("max")
                if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and hi > lo and hi - lo < 1e6:
                    return float(lo), float(hi), "the UI range of %s" % path
            except (ValueError, AttributeError, TypeError, KeyError):
                pass
    return 0.0, 1.0, "0 and 1 (no single custom property with a UI range feeds this driver)"


def _driver_curve(fc):
    """A driver's F-curve: its modifiers (a Rigify switch's GENERATOR [1, -1] flips which end means IK), its keys, and
    the value it gives at the two ends of its input's range - evaluated, so modifiers and keys both count (2026-10-04,
    split from the Rigify snap item)."""
    mods = []
    for md in fc.modifiers:
        row = {"type": md.type, "mute": bool(getattr(md, "mute", False))}
        for attr, key in (("mode", "mode"), ("poly_order", "polyOrder"), ("use_additive", "additive"),
                          ("function_type", "functionType")):
            if hasattr(md, attr):
                row[key] = getattr(md, attr)
        if hasattr(md, "coefficients"):
            row["coefficients"] = [round(float(c), 6) for c in md.coefficients]
        mods.append(row)
    lo, hi, where = _variable_range(fc.driver)
    return {"curveModifiers": mods, "curveKeyframes": len(fc.keyframe_points),
            "curveAt": [{"input": lo, "value": round(fc.evaluate(lo), 6)},
                        {"input": hi, "value": round(fc.evaluate(hi), 6)}],
            "inputRangeFrom": where}


def op_list_animation_data(params):
    """Every route by which an object is animated - action, drivers AND NLA.

    list_keyframes answers "which keyframes are on the action", which is a narrower question than
    it looks, and returns curveCount 0 for an object animated entirely by drivers. This answers
    "is this animated at all, and by what".

    params:
      object (str, required)
      target (str)   object | data | shape_keys | both (object + data) | all (default - all three)
    """
    reject_unknown(params, _LIST_KEYS, "list_animation_data")
    obj = get_object(take(params, "object", "name", required=True))
    wants = _anim_target(params)
    rows = []
    if "object" in wants:
        rows.append(_anim_summary("object", obj))
    if "data" in wants and obj.data is not None:
        rows.append(_anim_summary("data", obj.data))
    if "shape_keys" in wants and _shape_key_block(obj) is not None:
        rows.append(_anim_summary("shape_keys", _shape_key_block(obj)))
    routes = sorted({r for row in rows for r in row.get("animatedBy", [])})
    return {
        "object": obj.name,
        "sources": rows,
        "animatedBy": routes,
        "isAnimated": bool(routes),
        "invalidDrivers": sum(1 for row in rows for d in row.get("drivers", [])
                              if not d.get("isValid", True)),
    }


def op_delete_keyframe(params):
    """Remove a keyframe, and prove it went. The correction path set_keyframe never had.

    params:
      object (str, required)
      dataPath (alias path, required)  e.g. "location", "hide_render", 'pose.bones["x"].location'
      frame (int)                      the frame to clear. Omitted removes EVERY key on the path.
      index (int)                      which array element; omitted means all of them

    Counted before and after on the matching curves, because keyframe_delete returns a bool that is
    False both for "there was nothing there" and for "it refused", and those are different answers.
    """
    reject_unknown(params, {"object", "name", "dataPath", "path", "frame", "index", "target", "datablock"},
                   "delete_keyframe")
    obj = _anim_subject(params, "delete_keyframe", "NOTHING was deleted.")
    path = take(params, "dataPath", "path", default=None, kind=str)
    if not path:
        raise MifOpError("'dataPath' is required - which channel to clear. NOTHING was deleted.")
    frame = take_float(params, "frame", default=None)  # take_float, NOT float(): same contract - see the note on index above.
    index = take_int(params, "index", default=None)  # take_int, NOT int(): a bad value must REFUSE, not raise ValueError out of the op.

    # _curves_for, so a shape key's curve on the mesh's Key is found here too - see _anim_holders.
    curves = _curves_for(obj, path, index)
    if not curves:
        raise MifOpError("no fcurve on '%s' for dataPath '%s'%s. Nothing to delete - list them "
                         "with list_keyframes. NOTHING was deleted."
                         % (obj.name, path,
                            "" if index is None else " at index %s" % index))
    before = sum(len(fc.keyframe_points) for fc in curves)

    removed = 0
    for fc in curves:
        # Reverse order: removing shifts the collection under an ascending walk, which silently
        # skips every second key.
        for kp in reversed(list(fc.keyframe_points)):
            if frame is None or abs(kp.co[0] - float(frame)) < 1e-6:
                fc.keyframe_points.remove(kp)
                removed += 1
        fc.update()

    after = sum(len(fc.keyframe_points) for fc in _curves_for(obj, path, index))
    if removed == 0:
        raise MifOpError("'%s' on '%s' has %d keyframe(s) but none at frame %s. NOTHING was "
                         "deleted." % (path, obj.name, before, frame))
    return {
        "object": obj.name,
        "dataPath": path,
        "frame": frame,
        "index": index,
        "keyframesBefore": before,
        "keyframesAfter": after,
        "removed": removed,
        # MEASURED, not the operator's word. before-after is the count that actually left the
        # curve; `removed` is what this op thinks it did, and they must agree.
        "countsAgree": (before - after) == removed,
    }


def op_evaluate_at_frame(params):
    """What an object ACTUALLY is at given frames, through the depsgraph.

    THE VERIFICATION SUBSTRATE, and the reason it comes before the rest of the animation work.
    Every read in this addon reads the RAW property off the datablock, and that is not the value
    the scene evaluates to whenever a constraint, driver, NLA stack, parent or simulation cache is
    involved. A constraint does not touch obj.matrix_world at all: reading the base object reports
    every constraint as having done nothing, and reading back the value you just wrote is a proxy
    that cannot fail. Anything built on top of drivers or constraints cannot be proven without
    this.

    params:
      object (str, required)
      frames (list[int], required)  the frames to sample. A single int is accepted.
      dataPaths (list[str])         extra properties to read at each frame, evaluated. The world
                                    matrix is always reported.

    THE SCENE FRAME IS RESTORED and the restoration is ASSERTED, not assumed - this op moves
    frame_current to sample, and leaving somebody's scene on frame 47 because a read had a side
    effect is exactly the kind of quiet damage this codebase refuses.
    """
    reject_unknown(params, {"object", "name", "frames", "frame", "dataPaths", "paths", "allowSimulation", "bones",
                            "point"}, "evaluate_at_frame")
    obj = get_object(take(params, "object", "name", required=True))
    # BONES IN WORLD SPACE (2026-10-05): a pose bone's head or tail through the evaluated armature's world matrix -
    # what a motion path draws - so a hand's arc and spacing need no matrix arithmetic of the caller's own.
    bone_names = params.get("bones")
    point = str(params.get("point", "HEAD")).upper()
    if bone_names is not None:
        if obj.type != "ARMATURE":
            raise MifOpError("bones are read off an armature; '%s' is a %s. NOTHING was read." % (obj.name, obj.type))
        if not isinstance(bone_names, list) or not bone_names or any(not isinstance(b, str) for b in bone_names):
            raise MifOpError("bones is a non-empty list of bone names. NOTHING was read.")
        missing = [b for b in bone_names if obj.pose.bones.get(b) is None]
        if missing:
            raise MifOpError("'%s' has no bone %s. NOTHING was read." % (obj.name, ", ".join(missing)))
        if point not in ("HEAD", "TAIL"):
            raise MifOpError("point is HEAD or TAIL, got %r. NOTHING was read." % (params.get("point"),))
    elif params.get("point") is not None:
        raise MifOpError("point belongs with bones (the HEAD or TAIL of each). NOTHING was read.")

    raw_frames = params.get("frames", params.get("frame"))
    if raw_frames is None:
        raise MifOpError("'frames' is required - which frames to sample. NOTHING was read.")
    if isinstance(raw_frames, (int, float)):
        raw_frames = [raw_frames]
    if not isinstance(raw_frames, (list, tuple)) or not raw_frames:
        raise MifOpError("'frames' must be a non-empty list of frame numbers, got %r. NOTHING was "
                         "read." % (raw_frames,))
    try:
        # finite_int PER ELEMENT. A bare int() has no 32-bit bound, so frames:[2**40] reached
        # Blender as a raw ValueError from inside the assignment instead of a refusal here.
        frames = [finite_int(f, "frames") for f in raw_frames]
    # MifOpError IS CAUGHT TOO, and that is not tidiness. finite_int raises MifOpError rather than
    # ValueError, so without this its own wording ("must be an integer") replaced the wording this
    # op has always promised ("must be a number") - and B108 in test_blender_refusals asserts the
    # promised phrase. The guard is the change worth having; the caller-facing sentence is not, so
    # the sentence is preserved and finite_int's detail is appended to it.
    except (TypeError, ValueError, MifOpError) as exc:
        raise MifOpError("every entry in 'frames' must be a number: %s. NOTHING was read." % exc)

    paths = take(params, "dataPaths", "paths", default=None)
    if paths is not None and not isinstance(paths, (list, tuple)):
        raise MifOpError("'dataPaths' must be a list of property paths. NOTHING was read.")
    paths = [str(p) for p in (paths or [])]

    sc = bpy.context.scene
    started_on = sc.frame_current
    sim_reply = simulation_guard(params, frames, "evaluate_at_frame",
                                 tail="NOTHING was read, and the frame did not move.")
    samples = []
    # WHETHER THE WORLD MATRIX EVER LEFT THE OBJECT'S OWN CHANNELS, for channelsNote below.
    channels_differ = False
    try:
        for f in frames:
            sc.frame_set(f)
            # RE-FETCHED EVERY FRAME. The evaluated object is a temporary owned by the depsgraph
            # and is invalidated by the frame change; holding one across frames reads stale data
            # that looks perfectly plausible.
            dg = bpy.context.evaluated_depsgraph_get()
            ev = obj.evaluated_get(dg)
            mw = ev.matrix_world
            mb = ev.matrix_basis
            if any(abs(mw[i][j] - mb[i][j]) > 1e-5 for i in range(4) for j in range(4)):
                channels_differ = True
            row = {
                "frame": f,
                "location": rnd(list(mw.to_translation())),
                "rotationEuler": rnd(list(mw.to_euler())),
                "scale": rnd(list(mw.to_scale())),
            }
            if bone_names:
                row["bones"] = {b: rnd(list(mw @ getattr(ev.pose.bones[b], point.lower()))) for b in bone_names}
            if paths:
                values = {}
                for p in paths:
                    # OBJECT FIRST, THEN ITS DATABLOCK - the same two places set_keyframe writes.
                    #
                    # This resolved against the OBJECT only, which made the obvious question
                    # unanswerable: set_keyframe puts a light's `energy` on the light DATA (and
                    # list_keyframes reports target "data" to say so), and asking this op for
                    # "energy" came back
                    #     Object.path_resolve("energy") could not be resolved
                    # So the op named for evaluating a keyframed value could not evaluate the
                    # commonest keyframed value in the addon, and a caller had to already know that
                    # one op means data.energy while its sibling means object.energy.
                    #
                    # _resolve_target above owns that object-or-data decision for set_keyframe;
                    # this deliberately does not re-derive it, it just tries both on the EVALUATED
                    # copies and reports which one answered.
                    v, came_from, exc = None, None, None
                    for holder, tag in ((ev, "object"), (getattr(ev, "data", None), "data")):
                        if holder is None:
                            continue
                        try:
                            v = holder.path_resolve(p)
                            came_from = tag
                            break
                        except (ValueError, AttributeError, TypeError) as e:
                            exc = e
                    if came_from is None:
                        values[p] = {"error": "%s - tried the object and its %s datablock. "
                                              "set_keyframe writes light energy and camera lens to "
                                              "the DATA; list_keyframes reports which target a "
                                              "curve uses."
                                              % (str(exc)[:90],
                                                 type(getattr(ev, "data", None)).__name__)}
                        continue
                    # A STRING IS NOT A VECTOR. list('BLOCKS') is six letters and rnd raised
                    # ValueError on float('B') - a traceback for every enum path, found 2026-09-12
                    # the moment set_keyframe could key a remesh mode. A flag enum reads as a set.
                    if isinstance(v, str):
                        out = v
                    elif isinstance(v, (set, frozenset)):
                        out = sorted(v)
                    else:
                        try:
                            out = rnd(list(v))
                        except TypeError:
                            out = round(float(v), 6) if isinstance(v, (int, float)) else str(v)
                    # WHICH DATABLOCK ANSWERED. A bare number leaves the caller unable to tell
                    # object.foo from data.foo when both could exist, which is exactly the
                    # ambiguity that made this op unusable in the first place.
                    values[p] = out if came_from == "object" else {"value": out, "from": came_from}
                row["values"] = values
            samples.append(row)
    finally:
        sc.frame_set(started_on)

    restored = sc.frame_current
    if restored != started_on:
        raise MifOpError("sampled the frames but the scene was left on frame %d instead of %d. Do "
                         "not trust the scene state - set the frame yourself."
                         % (restored, started_on))

    # DID ANYTHING ACTUALLY MOVE. A caller verifying a constraint, a driver or a bake needs to know
    # the samples DIFFER, and a list of identical matrices is the normal failure - it is what a
    # dead driver, a muted NLA track or a bake that lost its motion all look like.
    locs = [tuple(s["location"]) for s in samples]
    rots = [tuple(s["rotationEuler"]) for s in samples]
    out = {
        "object": obj.name,
        "frames": frames,
        "samples": samples,
        "frameRestored": restored == started_on,
        "startedOnFrame": started_on,
        "movedAcrossFrames": len(set(locs)) > 1 or len(set(rots)) > 1,
        # WHICH HALF OF THE ROW EACH MOVER REACHES - the old sentence said constraints, parents and simulation caches
        # were "all applied", and a caller read values.location on a baked rigid body as the simulated position.
        "evaluatedNote": ("Read through evaluated_get(depsgraph). The top-level location, rotationEuler and scale are "
                          "the WORLD matrix, with keys, drivers, NLA, parents, constraints and simulation caches "
                          "applied. `values` are properties: keys, drivers and NLA reach them, but a parent, a "
                          "constraint or a rigid body moves only the world matrix, never the object's own "
                          "location/rotation/scale channels."),
    }
    # A TRANSFORM CHANNEL ASKED FOR WHILE SOMETHING ELSE MOVES THE OBJECT, said where the answer is. Reported by the
    # second seat (mailbox id 60): dataPaths ['location'] on a baked rigid body read 3.4 at every frame beside a
    # top-level location falling 3.4 -> 1.73 -> 0.48. MEASURED 2026-09-16, identical on 3.6.23, 4.2.17, 4.4.0, 5.0.1
    # and 5.2.1, on the evaluated object at frames 1/20/40: a keyed and a driven object's `location` channel follows
    # its world position; a child of a keyed parent, a COPY_LOCATION target and a baked rigid body keep the channel
    # at rest while matrix_world moves (the body's world z 4.0 -> 0.94 -> -0.40, its channel 4.0 throughout). That
    # is Blender's channel, correctly read, so it is not replaced - the difference is named instead.
    if bone_names:
        # SPACING: the world distance each bone's point travelled since the previous sample - what the gaps between a
        # motion path's dots show. Read through the depsgraph frame by frame (boneMethod), never from curves alone.
        out["bonePoint"] = point
        out["boneMethod"] = "depsgraph"
        out["boneSpacing"] = {b: [None] + [round(math.dist(samples[i - 1]["bones"][b], samples[i]["bones"][b]), 6)
                                           for i in range(1, len(samples))] for b in bone_names}
    asked_channels = sorted(p for p in paths if p.split("[")[0] in _TRANSFORM_CHANNELS)
    if asked_channels:
        out["transformChannelsDiffer"] = channels_differ
        if channels_differ:
            movers = []
            if obj.parent is not None:
                movers.append("parent '%s'" % obj.parent.name)
            on = [c for c in obj.constraints if getattr(c, "enabled", not getattr(c, "mute", False))]
            if on:
                movers.append("constraint%s %s" % ("s" if len(on) > 1 else "",
                                                   ", ".join("%s (%s)" % (c.name, c.type) for c in on)))
            rbw = sc.rigidbody_world
            if (getattr(obj, "rigid_body", None) is not None and rbw is not None
                    and getattr(rbw, "collection", None) is not None and obj.name in rbw.collection.objects):
                movers.append("a %s rigid body in the scene's rigid body world" % obj.rigid_body.type)
            out["channelsNote"] = (
                "%s %s the object's OWN channel%s, and something else moves it: the world matrix differs from those "
                "channels on at least one sampled frame. Moved by: %s. Where the object IS is the top-level "
                "location / rotationEuler / scale of each sample, not values."
                % (", ".join("values.%s" % p for p in asked_channels), "is" if len(asked_channels) == 1 else "are",
                   "" if len(asked_channels) == 1 else "s",
                   "; ".join(movers) if movers else "nothing this op can name - no parent, no enabled constraint "
                                                    "and no rigid body"))
    out.update(sim_reply)
    return out


def _curves_for(obj, path, index):
    """Fcurves on the object, its data or its shape keys matching a data path, and optionally one array index.

    A SHAPE KEY'S CURVE IS ON THE MESH'S Key DATABLOCK, as key_blocks["X"].value - the path list_keyframes reports. The
    path from the mesh, shape_keys.key_blocks["X"].value, which set_keyframe also takes, is matched there too."""
    found = []
    for label, holder in _anim_holders(obj):
        want = path[len("shape_keys."):] if label == "shape_keys" and path.startswith("shape_keys.") else path
        for fc in _fcurves(holder):
            if fc.data_path != want:
                continue
            if index is not None and fc.array_index != int(index):
                continue
            found.append(fc)
    return found


def _driver_curves(obj, path, index):
    """The driver F-curves for a driven property of obj - on the object, its data or its shape keys' Key, where
    add_driver puts them (_driver_owner re-roots data.* and shape-key paths); optionally one array index. The curve
    tools read an action's curves only, so a driver's response curve could be written and never reshaped."""
    found = []
    try:
        owner, rel, _where = _driver_owner(obj, path, "")
        wants = [(owner, rel)]
    except MifOpError:
        wants = []
    for _label, holder in _anim_holders(obj):
        wants.append((holder, path))
    for holder, want in wants:
        ad = getattr(holder, "animation_data", None)
        for fc in (getattr(ad, "drivers", None) or []) if ad is not None else []:
            if fc.data_path == want and (index is None or fc.array_index == int(index)) and fc not in found:
                found.append(fc)
    return found


def _enum_ids(rna_type, prop):
    """Blender's own enum for a property, read off the RNA rather than remembered."""
    try:
        return {i.identifier for i in rna_type.bl_rna.properties[prop].enum_items}
    except (KeyError, AttributeError):
        return set()


def op_edit_fcurve(params):
    """Change how a curve moves BETWEEN its keys - interpolation, easing, handles, extrapolation.

    set_keyframe can set an interpolation at INSERT time and reaches three of Blender's thirteen.
    Nothing could change one afterwards, and nothing could touch easing at all - which is most of
    the craft in motion graphics, where the difference between BACK/ease-out and LINEAR is the
    entire look. Retiming or re-feeling an existing animation meant deleting and re-keying it.

    params:
      object (str, required)
      dataPath (alias path, required)
      index (int)              one array element; omitted means every curve on the path
      frame (int)              one keyframe; omitted means every key on the curve
      interpolation (str)      CONSTANT | LINEAR | BEZIER | SINE | QUAD | CUBIC | QUART | QUINT |
                               EXPO | CIRC | BACK | BOUNCE | ELASTIC - validated against this
                               Blender's own enum
      easing (str)             AUTO | EASE_IN | EASE_OUT | EASE_IN_OUT - the DIRECTION an easing
                               interpolation runs, and meaningless without one of the above
      handleType (str)         FREE | ALIGNED | VECTOR | AUTO | AUTO_CLAMPED, both handles
      extrapolation (str)      CONSTANT | LINEAR, on the curve rather than its keys
      amplitude (float)        ELASTIC only, per key: how far each swing reaches. 0 (Blender's
                               default) swings as far as the move itself; larger boosts every swing
      period (float)           ELASTIC only, per key, in FRAMES between swings. 0 (the default)
                               means 0.3 of the segment - on a long segment the spring is over
                               almost at once, and a longer period is what makes it read
      back (float)             BACK only, per key: how far the move overshoots (Blender's 1.70158)
      mute / lock / hide (bool) the curve's own flags (FCurve.mute, .lock, .hide): a muted curve stops animating its
                               channel - only NLA tracks had a mute before. curveFlags gives each curve's previous and
                               new flags
      handleLeft / handleRight [frame, value]  one key's handle position (needs frame, and one curve). AUTO and
                               AUTO_CLAMPED recompute a handle, so the key's handle type is set to FREE first (ALIGNED when
                               only one side is given, Blender then keeps the other side in line) and handleTypeNote says
                               so. A left handle after its key, or a right one before it, is refused

    AMPLITUDE, PERIOD AND BACK WERE UNREACHABLE until 2026-09-13. The easing showcase's ELASTIC ball
    went from the floor to the top almost at once and then jittered, and nothing here could slow the
    spring down. They are written per key and READ BACK into keyParamsAfter; a key whose
    interpolation never reads one is named in warnings instead of being silently accepted.

    Every value is validated before anything is written.
    """
    reject_unknown(params, {"object", "name", "dataPath", "path", "index", "frame",
                            "interpolation", "easing", "handleType", "extrapolation",
                            "amplitude", "period", "back", "mute", "lock", "hide", "handleLeft", "handleRight",
                            "target", "datablock", "driver", "value"},
                   "edit_fcurve")
    obj = _anim_subject(params, "edit_fcurve")
    path = take(params, "dataPath", "path", default=None, kind=str)
    if not path:
        raise MifOpError("'dataPath' is required. NOTHING was changed.")
    index = take_int(params, "index", default=None)  # take_int, NOT int(): a bad value must REFUSE, not raise ValueError out of the op.
    frame = take_float(params, "frame", default=None)  # take_float, NOT float(): same contract - see the note on index above.
    # A DRIVER'S RESPONSE CURVE (2026-10-06): driver true finds the curve among the drivers of the driven property;
    # frame then names a key by its INPUT value (a driver curve's x is the variable, not a frame).
    driver = take_bool(params, "driver", default=False)
    key_value = take_float(params, "value", default=None)

    wants = {}
    for key, prop, rna in (("interpolation", "interpolation", bpy.types.Keyframe),
                           ("easing", "easing", bpy.types.Keyframe),
                           ("handleType", "handle_left_type", bpy.types.Keyframe),
                           ("extrapolation", "extrapolation", bpy.types.FCurve)):
        raw = take(params, key, default=None, kind=str)
        if raw is None:
            continue
        val = str(raw).upper()
        valid = _enum_ids(rna, prop)
        if valid and val not in valid:
            raise MifOpError("unknown %s '%s'. Valid: %s. NOTHING was changed."
                             % (key, val, ", ".join(sorted(valid))))
        wants[key] = val
    for key in ("amplitude", "period", "back"):
        val = take_float(params, key, default=None)
        if val is None:
            continue
        # NEGATIVE IS REFUSED: no negative swing, spacing or overshoot means anything, and the value
        # read back afterwards would not be the one the caller sent.
        if val < 0:
            raise MifOpError("'%s' must be 0 or more, got %s. NOTHING was changed." % (key, val))
        wants[key] = val
    flags = {}
    for key in ("mute", "lock", "hide"):
        if params.get(key) is not None:
            if not isinstance(params[key], bool):
                raise MifOpError("'%s' is true or false, got %r. NOTHING was changed." % (key, params[key]))
            flags[key] = params[key]
    handles = {}
    for key in ("handleLeft", "handleRight"):
        raw = params.get(key)
        if raw is None:
            continue
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise MifOpError("'%s' is [frame, value], got %r. NOTHING was changed." % (key, raw))
        handles[key] = finite_floats(list(raw), key)
    if handles:
        if frame is None:
            raise MifOpError("handleLeft / handleRight move ONE key's handle - pass frame. NOTHING was changed.")
        if wants.get("handleType") in ("AUTO", "AUTO_CLAMPED", "VECTOR"):
            raise MifOpError("handleType %s places the handles itself, so it cannot take handle coordinates - pass "
                             "FREE or ALIGNED, or leave handleType out. NOTHING was changed." % wants["handleType"])
    if key_value is not None and frame is None:
        raise MifOpError("value moves ONE key - pass frame (on a driver curve, the key's input value). NOTHING was "
                         "changed.")
    if not wants and not flags and not handles and key_value is None:
        raise MifOpError("nothing to change - pass at least one of interpolation, easing, handleType, "
                         "extrapolation, amplitude, period, back, mute, lock, hide, handleLeft, handleRight or value. "
                         "NOTHING was changed.")

    if driver and isinstance(obj, _TreeSubject):
        raise MifOpError("driver true reads an object's drivers, not a node tree's. NOTHING was changed.")
    curves = _driver_curves(obj, path, index) if driver else _curves_for(obj, path, index)
    if not curves:
        raise MifOpError("no %s on '%s' for dataPath '%s'%s - list them with %s. NOTHING was changed."
                         % ("driver" if driver else "fcurve", obj.name, path,
                            "" if index is None else " at index %s" % index,
                            "list_animation_data" if driver else "list_keyframes"))
    if key_value is not None and len(curves) != 1:
        raise MifOpError("value moves one key on ONE curve and '%s' matches %d - pass index. NOTHING was changed."
                         % (path, len(curves)))

    before = sorted({kp.interpolation for fc in curves for kp in fc.keyframe_points})

    # COUNTED BEFORE ANYTHING IS WRITTEN. This refusal used to live below the loop, and the loop
    # writes fc.extrapolation on EVERY curve before it ever looks at a keyframe - so
    # edit_fcurve({extrapolation: "LINEAR", frame: 999}) changed the extrapolation of every curve on
    # the object and then answered "NOTHING was changed". The frame filter only gates the per-KEY
    # writes, which is what made the two look consistent.
    if frame is not None and not any(abs(kp.co[0] - float(frame)) <= 1e-6
                                     for fc in curves for kp in fc.keyframe_points):
        raise MifOpError("'%s' has no keyframe at frame %s, so nothing was edited. NOTHING was "
                         "changed." % (path, frame))
    if handles:
        if len(curves) != 1:
            raise MifOpError("handleLeft / handleRight name one key on ONE curve and '%s' matches %d - pass index. "
                             "NOTHING was changed." % (path, len(curves)))
        key_at = [kp for kp in curves[0].keyframe_points if abs(kp.co[0] - float(frame)) <= 1e-6][0]
        if "handleLeft" in handles and handles["handleLeft"][0] > key_at.co[0] + 1e-9:
            raise MifOpError("handleLeft's frame %g is after its key at %g - the curve would loop back on itself. "
                             "NOTHING was changed." % (handles["handleLeft"][0], key_at.co[0]))
        if "handleRight" in handles and handles["handleRight"][0] < key_at.co[0] - 1e-9:
            raise MifOpError("handleRight's frame %g is before its key at %g - the curve would loop back on itself. "
                             "NOTHING was changed." % (handles["handleRight"][0], key_at.co[0]))
    flags_before = [{k: bool(getattr(fc, k)) for k in ("mute", "lock", "hide")} for fc in curves]

    touched = 0
    uses = {"amplitude": "ELASTIC", "period": "ELASTIC", "back": "BACK"}
    shaped = {}     # amplitude / period / back AS READ BACK from each key after the write
    inert = {}      # the interpolations of keys that were given one and never read it
    handle_note = None
    for fc in curves:
        if "extrapolation" in wants:
            fc.extrapolation = wants["extrapolation"]
        for key, value in flags.items():
            setattr(fc, key, value)
        for kp in fc.keyframe_points:
            if frame is not None and abs(kp.co[0] - float(frame)) > 1e-6:
                continue
            if "interpolation" in wants:
                kp.interpolation = wants["interpolation"]
            if "easing" in wants:
                kp.easing = wants["easing"]
            if "handleType" in wants:
                kp.handle_left_type = wants["handleType"]
                kp.handle_right_type = wants["handleType"]
            for key, needs in uses.items():
                if key in wants:
                    setattr(kp, key, wants[key])
                    shaped.setdefault(key, set()).add(round(float(getattr(kp, key)), 6))
                    if kp.interpolation != needs:
                        inert.setdefault(key, set()).add(kp.interpolation)
            if handles:
                # A HANDLE BLENDER PLACES ITSELF WOULD BE PUT BACK on the next update, so the type goes first: FREE for
                # both sides, ALIGNED for one (the other side stays in line with it).
                want_type = wants.get("handleType") or ("FREE" if len(handles) == 2 else "ALIGNED")
                was = (kp.handle_left_type, kp.handle_right_type)
                if was != (want_type, want_type):
                    kp.handle_left_type = want_type
                    kp.handle_right_type = want_type
                    handle_note = ("the key's handle type was %s / %s and is now %s, so the handle stays where it was "
                                   "put (AUTO and AUTO_CLAMPED recompute it)" % (was[0], was[1], want_type))
                if "handleLeft" in handles:
                    kp.handle_left = handles["handleLeft"]
                if "handleRight" in handles:
                    kp.handle_right = handles["handleRight"]
            if key_value is not None:
                # the handles move with the key, so its shape either side is kept
                dv = key_value - kp.co[1]
                kp.co[1] = key_value
                kp.handle_left[1] += dv
                kp.handle_right[1] += dv
            touched += 1
        fc.update()

    after = sorted({kp.interpolation for fc in curves for kp in fc.keyframe_points})
    result = {
        "object": obj.name,
        "dataPath": path,
        "curves": len(curves),
        "keyframesTouched": touched,
        "applied": wants,
        "interpolationBefore": before,
        "interpolationAfter": after,
        "extrapolation": [fc.extrapolation for fc in curves],
    }
    if driver:
        result["driver"] = True
    if key_value is not None:
        result["keyAfter"] = [rnd(list(kp.co)) for kp in curves[0].keyframe_points
                              if abs(kp.co[0] - float(frame)) <= 1e-6]
    if shaped:
        result["keyParamsAfter"] = {k: sorted(v) for k, v in sorted(shaped.items())}
    if inert:
        result["warnings"] = ["'%s' only shapes %s keys, and the key(s) it was written to are %s - it "
                              "changes nothing until their interpolation is %s"
                              % (k, uses[k], ", ".join(sorted(v)), uses[k]) for k, v in sorted(inert.items())]
    if flags:
        result["curveFlags"] = [{"index": fc.array_index, "before": b,
                                 "after": {k: bool(getattr(fc, k)) for k in ("mute", "lock", "hide")}}
                                for fc, b in zip(curves, flags_before)]
        if flags.get("mute"):
            result["muteNote"] = "a muted curve no longer animates its channel; the channel keeps its last value."
    if handles:
        kp = [k for k in curves[0].keyframe_points if abs(k.co[0] - float(frame)) <= 1e-6][0]
        result["handlesAfter"] = {"left": rnd(list(kp.handle_left)), "right": rnd(list(kp.handle_right)),
                                  "leftType": kp.handle_left_type, "rightType": kp.handle_right_type}
        if handle_note:
            result["handleTypeNote"] = handle_note
    return result


def op_add_fcurve_modifier(params):
    """Put a modifier on a curve - most usefully CYCLES, which is how an animation LOOPS.

    There was no way to loop anything. Every turntable, idle, cycling fan and blinking light had
    to be keyed out to its full length by hand, and a two-key rotation could not be made to repeat
    at all. A CYCLES modifier is the one-call answer and Blender has had it forever.

    params:
      object (str, required)
      dataPath (alias path, required)
      index (int)          one array element; omitted means every curve on the path
      type (str)           CYCLES | NOISE | GENERATOR | LIMITS | STEPPED | ENVELOPE | FNGENERATOR, and SMOOTH
                           from 5.2 - validated against this Blender's own enum. Default CYCLES.
      modeBefore/modeAfter (str)  CYCLES only: NONE | REPEAT | REPEAT_OFFSET | MIRROR
      strength (float)     NOISE only
      scale (float)        NOISE only
      settings (dict)      any writable property of THIS type's modifier, by Blender's name - LIMITS use_max_y /
                           max_y, STEPPED frame_step, GENERATOR poly_order / coefficients, FNGENERATOR function_type /
                           amplitude, ENVELOPE reference_value / default_min / default_max, SMOOTH sigma /
                           filter_width. Checked against this build's RNA before anything is added.
      controlPoints (list) ENVELOPE only: [{frame, min, max}, ...]
      stackIndex (int)     where the new modifier sits in the stack (SMOOTH belongs first); default last
      remove (str)         ALL | FIRST | a type: removes instead, over the same curves, and takes no type

    A curve with fewer than two keyframes is REFUSED for CYCLES: there is no cycle to repeat, and
    Blender adds the modifier anyway and does nothing with it, which looks like success.

    WHAT IT DID TO THE CURVE IS MEASURED, not assumed (second seat, mailbox id 61: SMOOTH, LIMITS and ENVELOPE could be
    added and not configured, and countsAgree came back true for a modifier that changed nothing). MEASURED 2026-09-16
    on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1, a curve keyed 0 -> 10 over frames 1-21: with defaults LIMITS and ENVELOPE
    leave every sampled value exactly as it was - LIMITS has no bound enabled, and even an envelope point at the default
    range is the identity - while LIMITS with use_max_y and max_y 5 clamps at once, and fcurve.evaluate() sees a
    modifier the moment it is added. So each curve is sampled across its keyed range and one range either side, before
    and after, and effect / noEffect say what changed.
    """
    reject_unknown(params, {"object", "name", "dataPath", "path", "index", "type",
                            "modeBefore", "modeAfter", "strength", "scale", "settings", "controlPoints",
                            "bones", "all", "action", "driver", "remove", "stackIndex"},
                   "add_fcurve_modifier")
    obj = get_object(take(params, "object", "name", required=True))
    path = take(params, "dataPath", "path", default=None, kind=str)
    driver = take_bool(params, "driver", default=False)
    if driver and (params.get("bones") is not None or params.get("all") is not None):
        raise MifOpError("driver true names one driven property by dataPath, not bones or all. NOTHING was added.")
    # MANY CURVES IN ONE CALL (2026-10-04): bones [names] | all true, from the object's action or a named one - a Cycles
    # modifier on every curve of a 30-bone walk was dozens of calls.
    many = _many_curves_selection(obj, params, path, "added")
    if not path and many is None:
        raise MifOpError("pass dataPath, bones [names] or all:true. NOTHING was added.")
    index = take_int(params, "index", default=None)  # take_int, NOT int(): a bad value must REFUSE, not raise ValueError out of the op.
    if params.get("remove") is not None:
        return _fmod_remove(obj, params, path, index, driver, many)
    at = take_int(params, "stackIndex", default=None)
    if at is not None and at < 0:
        raise MifOpError("stackIndex is a position in the modifier stack, 0 or more. NOTHING was added.")

    kind = str(take(params, "type", default="CYCLES", kind=str)).upper()
    valid = _enum_ids(bpy.types.FModifier, "type")
    if valid and kind not in valid:
        raise MifOpError("unknown fcurve modifier type '%s'. Valid: %s. NOTHING was added."
                         % (kind, ", ".join(sorted(valid))))

    modes = _enum_ids(bpy.types.FModifierCycles, "mode_before")
    named = []                                   # the named parameters, as (RNA property, value)
    for key, prop in (("modeBefore", "mode_before"), ("modeAfter", "mode_after")):
        raw = take(params, key, default=None, kind=str)
        if raw is None:
            continue
        if kind != "CYCLES":
            raise MifOpError("%s applies to a CYCLES modifier and this one is %s. NOTHING was "
                             "added." % (key, kind))
        if modes and str(raw).upper() not in modes:
            raise MifOpError("unknown %s '%s'. Valid: %s. NOTHING was added."
                             % (key, raw, ", ".join(sorted(modes))))
        named.append((prop, str(raw).upper()))
    for key in ("strength", "scale"):
        value = take_float(params, key, default=None)
        if value is None:
            continue
        # REFUSED, NOT IGNORED: strength on a LIMITS modifier used to be read by nothing and reported by nothing.
        if kind != "NOISE":
            raise MifOpError("%s applies to a NOISE modifier and this one is %s. NOTHING was added." % (key, kind))
        named.append((key, value))

    settings = _fmod_settings(kind, params.get("settings"))
    twice = sorted({k for k, _v in named} & {k for k, _v in settings})
    if twice:
        raise MifOpError("%s given both as a parameter and in settings - give it once. NOTHING was added."
                         % ", ".join(twice))
    points = _fmod_control_points(kind, params.get("controlPoints"))

    skipped = {}
    if many is not None:
        curves = [fc for fc in many if index is None or fc.array_index == int(index)]
        if kind == "CYCLES":
            skipped["fewerThanTwoKeys"] = [_curve_label(fc) for fc in curves if len(fc.keyframe_points) < 2]
        skipped["alreadyHasOne"] = [_curve_label(fc) for fc in curves if any(m.type == kind for m in fc.modifiers)]
        drop = set(skipped["alreadyHasOne"]) | set(skipped.get("fewerThanTwoKeys", ()))
        curves = [fc for fc in curves if _curve_label(fc) not in drop]
        if not curves:
            raise MifOpError("no curve left to give a %s modifier: %s. NOTHING was added."
                             % (kind, "; ".join("%s: %s" % (k, ", ".join(v[:10])) for k, v in skipped.items() if v)
                                or "the selection matched no curve"))
    else:
        curves = _driver_curves(obj, path, index) if driver else _curves_for(obj, path, index)
    if not curves:
        raise MifOpError("no %s on '%s' for dataPath '%s' - %s. NOTHING was added."
                         % ("driver" if driver else "fcurve", obj.name, path,
                            "add one with add_driver" if driver else "key it first with set_keyframe"))
    if kind == "CYCLES":
        thin = [fc for fc in curves if len(fc.keyframe_points) < 2]
        if thin:
            raise MifOpError(
                "a CYCLES modifier needs at least TWO keyframes to have a cycle to repeat, and %d "
                "of the matching curve(s) have fewer. Blender would add the modifier and do "
                "nothing with it, which looks like success. NOTHING was added." % len(thin))

    before_counts = [len(fc.modifiers) for fc in curves]
    sample_frames = [_fmod_sample_frames(fc) for fc in curves]
    values_before = [[fc.evaluate(f) for f in frames] for fc, frames in zip(curves, sample_frames)]
    made = []
    try:
        for fc in curves:
            mod = fc.modifiers.new(type=kind)
            made.append((fc, mod))
            _fmod_apply(mod, named + settings, points)
    except (TypeError, ValueError, AttributeError, RuntimeError) as exc:
        # Everything checkable was checked above; what is left is Blender's own word on a combination (a
        # coefficients list the wrong length for poly_order). Half a request is not left on the curves.
        for fc, mod in made:
            fc.modifiers.remove(mod)
            fc.update()
        raise MifOpError("Blender refused the %s modifier's settings: %s. The %d modifier(s) this call had added "
                         "were removed again - NOTHING was added." % (kind, str(exc)[:200], len(made)))
    if at is not None:
        made = _fmod_move_all(made, at)
    added = []
    for fc, mod in made:
        added.append({"index": fc.array_index,
                      "stackIndex": list(fc.modifiers).index(mod),
                      "type": mod.type,
                      "modeBefore": getattr(mod, "mode_before", None),
                      "modeAfter": getattr(mod, "mode_after", None),
                      # READ BACK OFF THE MODIFIER, not echoed from the request.
                      "settings": {k: _fmod_read(mod, k) for k, _v in named + settings} or None,
                      "controlPoints": ([{"frame": round(float(p.frame), 4), "min": round(float(p.min), 6),
                                          "max": round(float(p.max), 6)} for p in mod.control_points]
                                        if kind == "ENVELOPE" else None)})
        fc.update()

    values_after = [[fc.evaluate(f) for f in frames] for fc, frames in zip(curves, sample_frames)]
    changed = [sum(1 for a, b in zip(va, vb) if abs(a - b) > 1e-6) for va, vb in zip(values_after, values_before)]
    max_change = max((abs(a - b) for va, vb in zip(values_after, values_before) for a, b in zip(va, vb)), default=0.0)
    no_effect = not any(changed)
    after_counts = [len(fc.modifiers) for fc in (curves if many is not None or driver else _curves_for(obj, path, index))]
    extra = {}
    if many is not None:
        added = added[:50]
        extra = {"curvesChanged": [_curve_label(fc) for fc in curves][:200],
                 "skipped": {k: v[:100] for k, v in skipped.items() if v} or None}
    return dict({
        "object": obj.name,
        "dataPath": path,
        "type": kind,
        "curves": len(curves),
        "modifiersAdded": added,
        "modifierCountBefore": before_counts,
        "modifierCountAfter": after_counts,
        # COUNTED OFF THE CURVES. fcurve.modifiers.new returns an object whether or not it stuck,
        # and one modifier per curve is the postcondition - not "the call returned something".
        "countsAgree": all(a == b + 1 for a, b in zip(after_counts, before_counts)),
        # WHAT THE MODIFIER DID TO THE CURVE, sampled before and after - see the docstring.
        "effect": {"sampledFrames": sum(len(f) for f in sample_frames), "framesChanged": sum(changed),
                   "maxChange": round(float(max_change), 6)},
        "noEffect": no_effect,
        "noEffectNote": ("the %s modifier changed no sampled value of the curve (its keyed range and one range either "
                         "side). %s" % (kind, "It was added muted (settings.mute)." if any(m.mute for _f, m in made)
                                        else _FMOD_NO_EFFECT_HINTS.get(kind, "Check its settings.")))
                        if no_effect else None,
        "loopNote": ("A CYCLES modifier repeats the keyed range outside it. Sample with "
                     "evaluate_at_frame beyond the last key to prove the loop is live - the "
                     "modifier existing is not the same as it having an effect."),
    }, **extra)


def _fmod_remove(obj, params, path, index, driver, many):
    """add_fcurve_modifier remove ALL | FIRST | a type, over the same curves an add would take (2026-10-06: nothing
    removed a modifier, 5.2 deletes in bulk)."""
    extra = sorted(k for k in ("type", "modeBefore", "modeAfter", "strength", "scale", "settings", "controlPoints",
                               "stackIndex") if params.get(k) is not None)
    if extra:
        raise MifOpError("remove takes no %s. NOTHING was removed." % ", ".join(extra))
    which = str(take(params, "remove", kind=str)).upper()
    kinds = _enum_ids(bpy.types.FModifier, "type")
    if which not in ("ALL", "FIRST") and which not in kinds:
        raise MifOpError("remove is ALL, FIRST or a type (%s), got %r. NOTHING was removed."
                         % (", ".join(sorted(kinds)), which))
    if many is not None:
        curves = [fc for fc in many if index is None or fc.array_index == index]
    elif not path:
        raise MifOpError("pass dataPath, bones [names] or all:true. NOTHING was removed.")
    else:
        curves = _driver_curves(obj, path, index) if driver else _curves_for(obj, path, index)
    plan = [(fc, [m for i, m in enumerate(fc.modifiers) if which == "ALL" or (which == "FIRST" and i == 0)
                  or m.type == which]) for fc in curves]
    if not any(gone for _fc, gone in plan):
        raise MifOpError("no %s modifier on the %d matching curve(s). NOTHING was removed."
                         % ("" if which in ("ALL", "FIRST") else which, len(curves)))
    rows = []
    for fc, gone in plan:
        before = len(fc.modifiers)
        for m in gone:
            fc.modifiers.remove(m)
        fc.update()
        rows.append({"curve": _curve_label(fc), "before": before, "after": len(fc.modifiers),
                     "left": [m.type for m in fc.modifiers]})
    return {"object": obj.name, "dataPath": path, "remove": which, "curves": rows[:200],
            "removed": sum(r["before"] - r["after"] for r in rows)}


def _fmod_snap(m, mine=False):
    """One modifier as (type, writable settings, envelope points, whether it is the one being placed)."""
    props = {p.identifier: getattr(m, p.identifier) for p in m.bl_rna.properties
             if not p.is_readonly and p.type not in ("POINTER", "COLLECTION") and p.identifier != "type"}
    props = {k: (tuple(v) if hasattr(v, "__len__") and not isinstance(v, str) else v) for k, v in props.items()}
    pts = [(c.frame, c.min, c.max) for c in m.control_points] if m.type == "ENVELOPE" else []
    return (m.type, props, pts, mine)


def _fmod_rebuild(fc, snaps):
    """fc's modifiers replaced by `snaps` in that order; returns the placed one (or None). Blender refuses a CYCLES that
    is not first before 5.2 (measured: 4.2.23 to 5.0.1 log "can only be first in stack" and return None)."""
    for m in list(fc.modifiers):
        fc.modifiers.remove(m)
    placed = None
    for kind, props, pts, mine in snaps:
        m = fc.modifiers.new(type=kind)
        if m is None:
            continue
        for _twice in (0, 1):     # frame_start/end and poly_order before coefficients clamp one another
            for k, v in sorted(props.items(), key=lambda kv: kv[0] == "coefficients"):
                try:
                    setattr(m, k, v)
                except (TypeError, ValueError, AttributeError):
                    pass
        for f, lo, hi in pts:
            c = m.control_points.add(f)
            c.min, c.max = lo, hi
        placed = m if mine else placed
    fc.update()
    return placed


def _fmod_move_all(made, at):
    """Each new modifier rebuilt into place `at` (no API moves one; SMOOTH belongs first), every setting and envelope
    point carried over. All or nothing: a stack Blender will not hold puts every curve back as it was before this call
    and refuses."""
    plans = []
    for fc, mod in made:
        mods = list(fc.modifiers)
        before = [_fmod_snap(m) for m in mods if m != mod]
        order = [_fmod_snap(m, m == mod) for m in mods if m != mod]
        order.insert(min(at, len(order)), _fmod_snap(mod, True))
        plans.append((fc, mod, before, order))
    out = []
    for i, (fc, mod, before, order) in enumerate(plans):
        placed = _fmod_rebuild(fc, order)
        if placed is None or [m.type for m in fc.modifiers] != [o[0] for o in order]:
            got = [m.type for m in fc.modifiers]
            for fc2, mod2, before2, _o in plans[:i + 1]:
                _fmod_rebuild(fc2, before2)
            for fc2, mod2, _b, _o in plans[i + 1:]:
                fc2.modifiers.remove(mod2)
                fc2.update()
            raise MifOpError("Blender would not hold the stack %s on %s (it kept %s) - before 5.2 a CYCLES must be "
                             "first. Every curve was put back as it was. NOTHING was added."
                             % ([o[0] for o in order], _curve_label(fc), got))
        out.append((fc, placed))
    return out

def _curve_label(fc):
    return "%s[%d]" % (fc.data_path, fc.array_index)


def _bone_of_path(data_path):
    """'arm.L' for 'pose.bones["arm.L"].location', else None."""
    head = 'pose.bones["'
    if not data_path.startswith(head):
        return None
    end = data_path.find('"]', len(head))
    return data_path[len(head):end] if end > 0 else None


def _many_curves_selection(obj, params, path, verb):
    """bones [names] | all true over the object's action (or `action`, a name): the curves, or None when neither was
    given. Refused with dataPath, or with both."""
    bones = params.get("bones")
    every = take_bool(params, "all", default=False)
    act_name = take(params, "action", default=None, kind=str)
    if bones is None and not every:
        if act_name is not None:
            raise MifOpError("action names where bones or all:true read their curves. NOTHING was %s." % verb)
        return None
    if path:
        raise MifOpError("pass dataPath OR bones / all:true, not both. NOTHING was %s." % verb)
    if bones is not None and every:
        raise MifOpError("pass bones OR all:true. NOTHING was %s." % verb)
    if bones is not None and (not isinstance(bones, list) or not bones
                              or any(not isinstance(b, str) or not b for b in bones)):
        raise MifOpError("bones is a non-empty list of bone names. NOTHING was %s." % verb)
    if act_name is not None:
        act = bpy.data.actions.get(act_name)
        if act is None:
            raise MifOpError("no action named '%s'. NOTHING was %s." % (act_name, verb))
        curves = _action_curves(act)
    else:
        curves = [fc for _label, holder in _anim_holders(obj) for fc in _fcurves(holder)]
    if bones is not None:
        have = {_bone_of_path(fc.data_path) for fc in curves}
        missing = [b for b in bones if b not in have]
        if missing:
            raise MifOpError("no curve animates bone(s) %s. NOTHING was %s." % (", ".join(missing[:20]), verb))
        wanted = set(bones)
        curves = [fc for fc in curves if _bone_of_path(fc.data_path) in wanted]
    if not curves:
        raise MifOpError("'%s' has no animation curve to act on. NOTHING was %s."
                         % (act_name or obj.name, verb))
    return curves


# WHY A MODIFIER CAN BE ADDED AND CHANGE NOTHING, per type - each a measured default (see op_add_fcurve_modifier).
_FMOD_NO_EFFECT_HINTS = {
    "LIMITS": "LIMITS clamps only what is enabled: pass settings use_min_y / use_max_y (or _x) with min_y / max_y.",
    "ENVELOPE": ("ENVELOPE reshapes only through control points whose min/max differ from default_min/default_max: "
                 "pass controlPoints [{frame, min, max}]."),
    "CYCLES": "CYCLES repeats outside the keyed range - check modeBefore/modeAfter are not NONE.",
    "NOISE": "NOISE with strength 0 adds nothing.",
    "STEPPED": "STEPPED holds values for frame_step frames - a step of 1 or less holds nothing.",
    "SMOOTH": "SMOOTH averages over filter_width frames - a straight segment stays straight.",
}


def _fmod_struct(kind):
    """The RNA struct a modifier of this type is, on THIS build - None for a type this build does not have."""
    name = {"GENERATOR": "FModifierGenerator", "FNGENERATOR": "FModifierFunctionGenerator",
            "ENVELOPE": "FModifierEnvelope", "CYCLES": "FModifierCycles", "NOISE": "FModifierNoise",
            "LIMITS": "FModifierLimits", "STEPPED": "FModifierStepped", "SMOOTH": "FModifierSmooth"}.get(kind)
    return getattr(bpy.types, name, None) if name else None


def _fmod_settings(kind, raw):
    """settings validated against this build's RNA for the type, BEFORE any modifier exists: [(name, value), ...] with
    poly_order ahead of coefficients, whose length it decides."""
    if raw is None:
        return []
    if not isinstance(raw, dict) or not raw:
        raise MifOpError("'settings' must be a non-empty object of {property: value}. NOTHING was added.")
    struct = _fmod_struct(kind)
    if struct is None:
        raise MifOpError("this Blender has no settings table for a %s modifier. NOTHING was added." % kind)
    props = {p.identifier: p for p in struct.bl_rna.properties
             if not p.is_readonly and p.identifier != "rna_type" and p.type in ("ENUM", "BOOLEAN", "INT", "FLOAT")}
    out = []
    for key, value in raw.items():
        p = props.get(str(key))
        if p is None:
            raise MifOpError("'%s' is not a writable property of a %s modifier on this Blender (%s). Writable: %s. "
                             "NOTHING was added." % (key, kind, bpy.app.version_string, ", ".join(sorted(props))))
        label = "settings.%s" % key
        if p.type == "ENUM":
            ids = {i.identifier for i in p.enum_items}
            if getattr(p, "is_enum_flag", False) or not isinstance(value, str) or value.upper() not in ids:
                raise MifOpError("%s must be one of %s, got %r. NOTHING was added." % (label, ", ".join(sorted(ids)), value))
            value = value.upper()
        elif p.type == "BOOLEAN":
            if not isinstance(value, bool):
                raise MifOpError("%s must be true or false, got %r. NOTHING was added." % (label, value))
        else:
            many = bool(getattr(p, "is_array", False))
            if many != isinstance(value, (list, tuple)) or (many and not value):
                raise MifOpError("%s is %s, got %r. NOTHING was added."
                                 % (label, "a list of numbers" if many else "one number", value))
            convert = finite_int if p.type == "INT" else finite_float
            try:
                if any(isinstance(v, bool) for v in (value if many else [value])):
                    raise MifOpError("'%s' must be a number, got %r" % (label, value))
                value = [convert(v, label) for v in value] if many else convert(value, label)
            except MifOpError as exc:
                raise MifOpError("%s. NOTHING was added." % str(exc).rstrip(". "))
            # REFUSED, NOT CLAMPED: Blender clamps an out-of-range write to the property's hard range without a word -
            # cycles_before = 10**6 read back 32767 on all five builds.
            for v in (value if many else [value]):
                if not (p.hard_min <= v <= p.hard_max):
                    raise MifOpError("%s must be between %s and %s on this Blender, got %r. NOTHING was added."
                                     % (label, _fmod_bound(p.hard_min), _fmod_bound(p.hard_max), v))
        out.append((str(key), value))
    return sorted(out, key=lambda kv: 0 if kv[0] == "poly_order" else 1)


def _fmod_bound(v):
    return "%g" % v if abs(v) < 1e30 else ("unbounded" if v > 0 else "-unbounded")


def _fmod_control_points(kind, raw):
    """controlPoints for an ENVELOPE, validated before anything is added: [(frame, min, max), ...]."""
    if raw is None:
        return []
    if kind != "ENVELOPE":
        raise MifOpError("controlPoints belong to an ENVELOPE modifier and this one is %s. NOTHING was added." % kind)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise MifOpError("'controlPoints' must be a non-empty list of {frame, min, max}. NOTHING was added.")
    out = []
    for i, pt in enumerate(raw):
        if not isinstance(pt, dict) or set(pt) != {"frame", "min", "max"}:
            raise MifOpError("controlPoints[%d] must be exactly {frame, min, max}, got %r. NOTHING was added." % (i, pt))
        try:
            frame, lo, hi = (finite_float(pt[k], "controlPoints[%d].%s" % (i, k)) for k in ("frame", "min", "max"))
        except MifOpError as exc:
            raise MifOpError("%s. NOTHING was added." % str(exc).rstrip(". "))
        if any(abs(frame - f) < 1e-6 for f, _lo, _hi in out):
            # MEASURED on all five builds: a second point at one frame raises "Already a control point at frame 11"
            # - after the modifier exists.
            raise MifOpError("controlPoints[%d] repeats frame %g - one point per frame. NOTHING was added." % (i, frame))
        out.append((frame, lo, hi))
    return out


def _fmod_apply(mod, settings, points):
    for key, value in settings:
        setattr(mod, key, value)
    for frame, lo, hi in points:
        cp = mod.control_points.add(frame)
        cp.min, cp.max = lo, hi


def _fmod_read(mod, key):
    v = getattr(mod, key, None)
    if hasattr(v, "__len__") and not isinstance(v, str):
        return [round(float(x), 6) for x in v]
    return round(float(v), 6) if isinstance(v, float) else v


def _fmod_sample_frames(fc):
    """Frames to sample a curve at: across its keyed range and one range either side, 25 of them."""
    keys = [k.co[0] for k in fc.keyframe_points]
    lo, hi = (min(keys), max(keys)) if keys else (1.0, 2.0)
    span = max(hi - lo, 1.0)
    start, end = lo - span, hi + span
    return [start + (end - start) * i / 24.0 for i in range(25)]


def _action_curves(act):
    """Fcurves belonging to an ACTION, on any supported Blender.

    _fcurves() above takes a HOLDER and goes through its animation_data, which is the right shape
    for "what is animating this object" and the wrong one for "what is in this action" - an action
    with no user has no holder to reach it through. Same 5.0 slotted-layout problem, one level up:
    Action.fcurves is gone in 5.0 and the curves live under layers/strips/channelbags.
    """
    # every slot's curves: 4.4's Action.fcurves holds the first slot's only (see _fcurves)
    if hasattr(act, "fcurves") and not len(getattr(act, "layers", ())):
        return list(act.fcurves)
    out = []
    for layer in getattr(act, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for cb in getattr(strip, "channelbags", ()):
                out.extend(getattr(cb, "fcurves", ()))
    return out


def animated_channels(obj, paths):
    """{"path[index]": [what animates it]} for the given RNA data paths of obj: the active action's curves and those of
    every NLA strip that plays - unmuted curves, strips and tracks, the solo tracks alone when any track is solo, no
    NLA when use_nla is off. Drivers are not included.

    WHY. A write to an animated channel reads back and does not last. MEASURED 2026-09-16, identical on 3.6.23,
    4.2.17, 4.4.0, 5.0.1 and 5.2.1, a cube with x and z keyed 0 -> 2 / 0 -> 3 over frames 1-20, at frame 10:
    location = (7, 5, 9) then view_layer.update() read (7, 5, 9); after a frame change x and z were the animation's
    again (0.9211, 1.3817) and y kept 5.0. The same through an NLA strip; on a muted track, and with no animation, the
    write stayed. transform_object reads back right after its update, so the response agreed with a value the next
    frame threw away (second seat, mailbox id 93)."""
    ad = getattr(obj, "animation_data", None)
    if ad is None:
        return {}
    found = {}

    def take_curves(curves, source):
        for fc in curves:
            if fc.data_path in paths and not getattr(fc, "mute", False):
                found.setdefault("%s[%d]" % (fc.data_path, fc.array_index), [])
                if source not in found["%s[%d]" % (fc.data_path, fc.array_index)]:
                    found["%s[%d]" % (fc.data_path, fc.array_index)].append(source)

    if ad.action is not None:
        take_curves(_fcurves(obj), "action '%s'" % ad.action.name)
    if getattr(ad, "use_nla", True):
        tracks = list(ad.nla_tracks)
        solo = [t for t in tracks if getattr(t, "is_solo", False)]
        for track in (solo or tracks):
            if track.mute:
                continue
            for strip in track.strips:
                if not strip.mute and strip.action is not None:
                    take_curves(_action_curves(strip.action), "NLA track '%s'" % track.name)
    return found


def _action_row(act):
    curves = _action_curves(act)
    try:
        frame_range = [round(float(v), 4) for v in act.frame_range]
    except (AttributeError, TypeError):
        frame_range = None
    row = {
        "name": act.name,
        "users": int(act.users),
        "useFakeUser": bool(act.use_fake_user),
        "curveCount": len(curves),
        "keyframeTotal": sum(len(fc.keyframe_points) for fc in curves),
        "frameRange": frame_range,
        # THE FIELD THAT MATTERS. An action with no users and no fake user is DELETED the next time
        # the file is saved - silently, and by the save succeeding. That is the same purge
        # save_file reports as purgedOrphans, seen from the other side.
        "survivesSave": bool(act.users) or bool(act.use_fake_user),
        # THE CLIP LENGTH AN EXPORT WRITES (2026-10-19): the FBX all-actions export and glTF take a clip's length from
        # act.frame_range, which is the manual range when useFrameRange is on - set_action writes all three.
        "useFrameRange": bool(getattr(act, "use_frame_range", False)),
        "manualFrameRange": ([round(float(act.frame_start), 4), round(float(act.frame_end), 4)]
                             if getattr(act, "use_frame_range", False) else None),
        "cyclic": bool(getattr(act, "use_cyclic", False)),
    }
    if "slots" in act.bl_rna.properties:
        # 4.4+: who plays which slot - users() counts NLA strips that play it as well as active actions
        row["slots"] = [{"identifier": s.identifier, "name": s.name_display, "idType": s.target_id_type,
                         "users": sorted(u.name for u in s.users())} for s in act.slots]
    return row


def op_list_actions(params):
    """Every action in the file, who uses it, and whether it will survive a save.

    params:
      nameContains (str)  optional substring filter, case-insensitive

    An action is also the CLIP NAME glTF and FBX write into an engine, so "whatever Blender
    auto-named it" becomes a name somebody has to live with downstream.
    """
    reject_unknown(params, ("nameContains", "limit", "nameExcludes"), "list_actions")
    sub = take(params, "nameContains", default=None, kind=str)
    rows = []
    for act in bpy.data.actions:
        if sub and sub.lower() not in act.name.lower():
            continue
        rows.append(_action_row(act))
    rows.sort(key=lambda r: r["name"])

    # WHO IS USING WHAT, built by walking objects rather than asked of the action - an action knows
    # its user COUNT and not their names, and "which object is this clip on" is the question.
    by_object = {}
    for obj in bpy.data.objects:
        # THE SHAPE KEYS TOO (2026-09-28): an action animating a mesh's shape keys is used by its Key datablock, and this
        # read the object and its data only - so such an action listed users 1 and usedBy [].
        for label, holder in _anim_holders(obj):
            ad = getattr(holder, "animation_data", None) if holder is not None else None
            if ad is not None and ad.action is not None:
                by_object.setdefault(ad.action.name, []).append("%s (%s)" % (obj.name, label))
    for row in rows:
        row["usedBy"] = sorted(by_object.get(row["name"], []))

    doomed = [r["name"] for r in rows if not r["survivesSave"]]
    rows = exclude_rows(rows, params)
    matched = len(rows)
    rows, extra = limit_rows(rows, params, "action(s)")
    return dict({
        "count": matched,
        "actions": rows,
        "willBeDeletedOnSave": doomed,
        "willBeDeletedOnSaveCount": len(doomed),
    }, **extra)


def op_create_action(params):
    """Create a named action, optionally assigning it. Naming is the point.

    An object gets whatever Blender auto-named its action - "Action.003" - and that string is what
    glTF and FBX write into the engine as the clip name. Nothing here could set it, so every clip
    exported through this bridge arrived downstream named after nothing.

    params:
      name (str, required)
      object (str)        assign it to this object as well
      fakeUser (bool)     default TRUE. An action with no users and no fake user is deleted on the
                          next save, and a freshly created unassigned one has no users by
                          definition - so the safe default is the one that does not lose work.
    """
    reject_unknown(params, {"name", "object", "fakeUser"}, "create_action")
    name = take(params, "name", default=None, kind=str)
    if not name:
        raise MifOpError("'name' is required - actions are the clip name an engine sees, so an "
                         "auto-generated one is the thing this op exists to avoid. NOTHING was "
                         "created.")
    obj_name = take(params, "object", default=None, kind=str)
    obj = None
    if obj_name:
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            raise MifOpError("no object named '%s' to assign to. NOTHING was created." % obj_name)

    # DEFAULTS TRUE on purpose: a new unassigned action has zero users, so the default that loses
    # the work is the one that does nothing. READ BEFORE THE ACTION EXISTS: it was read on the line after
    # bpy.data.actions.new, so {fakeUser: "flase"} was refused with the action left behind holding the
    # name (measured on 5.0.1, 2026-09-21).
    try:
        fake = take_bool(params, "fakeUser", default=True)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was created." % str(exc).rstrip("."))
    act = bpy.data.actions.new(name=str(name))
    act.use_fake_user = fake

    assigned = False
    if obj is not None:
        if obj.animation_data is None:
            obj.animation_data_create()
        obj.animation_data.action = act
        assigned = obj.animation_data.action is act

    return {
        "action": act.name,
        "requestedName": str(name),
        # Blender uniquifies silently, and a caller who then looks up the name they asked for finds
        # a DIFFERENT action - or none.
        #
        # `nameWasTaken` WAS EMITTED HERE AND IS GONE as of 2026-09-07 - see create_collection for
        # the reasoning. Both spellings shipped together from 2026-09-04; this drops the private one
        # at the release boundary.
        # THE CANONICAL SPELLING, ALONGSIDE. Ten other ops call this nameWasSuffixed, and a caller
        # testing that name silently missed this op. Both are emitted so nothing that reads the old
        # one breaks; retiring it is a release-boundary decision, not a tidy-up.
        "nameWasSuffixed": act.name != str(name),
        "assignedTo": obj.name if assigned else None,
        "assigned": assigned,
        "info": _action_row(act),
    }


def op_assign_action(params):
    """Put an existing action on an object, or clear it. The switch nothing could make.

    An object held one action forever: set_keyframe creates one on first use and nothing could
    swap it, so a second clip on the same rig was impossible.

    params:
      object (str, required)
      action (str)       the action to assign. Omit with clear:true to unlink.
      slot (str)         Blender 4.4+: which of the action's slots the object plays, by identifier
                         ('OBBall') or display name ('Ball'). Needed only when the action has
                         several object slots and none is named for this object.
      clear (bool)       unlink the current action instead of assigning one.
      restPose (bool)    an armature's pose bones go back to rest (location 0, rotation identity, scale 1) before the
                         action is assigned. WHY (2026-10-25): a channel the new action does not key KEEPS whatever the
                         previous action left there - measured on 5.2.1 with two Unreal clips imported from FBX, the
                         second (which left the root at its reference pose) played with the first one's root turn,
                         165 cm off at the head. Unreal plays an unkeyed bone at its reference pose; this matches it.

    CLEARING IS THE DANGEROUS ONE and it says so: an unlinked action with no fake user drops to
    zero users and is deleted on the next save. The response reports whether the action that was
    unlinked will survive.
    """
    reject_unknown(params, {"object", "name", "action", "slot", "clear", "restPose"}, "assign_action")
    obj = get_object(take(params, "object", "name", required=True))
    require_editable(obj, "animate")
    clear = take_bool(params, "clear", default=False)
    rest_pose = take_bool(params, "restPose", default=False)
    if rest_pose and (clear or obj.type != "ARMATURE"):
        raise MifOpError("restPose puts an armature's pose bones back to rest before an action is assigned; '%s' is %s. "
                         "NOTHING was changed." % (obj.name, "being cleared" if clear else "a " + obj.type))
    want = take(params, "action", default=None, kind=str)
    if clear and want:
        raise MifOpError("pass an action to assign OR clear:true, not both. NOTHING was changed.")
    if not clear and not want:
        raise MifOpError("'action' is required unless clear:true. NOTHING was changed.")

    # THE ACTION IS RESOLVED BEFORE ANY DATABLOCK IS MADE. Until 2026-09-15 animation_data_create()
    # ran first - everything below it needs animation_data to exist, which is true - and the action
    # lookup came afterwards. So a typo left a fresh object carrying an EMPTY animation_data block
    # while the refusal said "NOTHING was changed":
    #
    #     assign_action{object: <fresh cube>, action: "<no such action>"}
    #       -> ok:false "no action named ..."   and hasAnimationData went false -> TRUE
    #
    # Measured by the second seat on 4.2.0, 4.4.0, 5.0.1 and 5.2.1 LTS. An empty block is saved with
    # the file, shows in the dope sheet and the NLA, and is exactly what list_animation_data reports
    # when a caller asks whether an object is animated.
    #
    # The CREATE still happens before the assignment - that half of the original reasoning was
    # right. Only the lookup moved above it.
    act = None
    if not clear:
        act = bpy.data.actions.get(str(want))
        if act is None:
            known = [a.name for a in bpy.data.actions][:20]
            raise MifOpError("no action named '%s'. Present: %s. NOTHING was changed."
                             % (want, ", ".join(known) if known else "<none>"))

    if obj.animation_data is None:
        if clear:
            raise MifOpError("'%s' has no animation data, so there is no action to clear. NOTHING "
                             "was changed." % obj.name)
        obj.animation_data_create()

    previous = obj.animation_data.action
    prev_row = _action_row(previous) if previous is not None else None

    if clear:
        obj.animation_data.action = None
        after = obj.animation_data.action
        if after is not None:
            raise MifOpError("asked to clear the action on '%s' but it still holds '%s'. Do not "
                             "trust this state." % (obj.name, after.name))
        # RE-READ the previous action AFTER unlinking: its user count has changed and that is the
        # number that decides whether it survives the next save.
        return {
            "object": obj.name,
            "cleared": True,
            "previousAction": prev_row["name"] if prev_row else None,
            "previousActionNow": _action_row(previous) if previous is not None else None,
            "previousSurvivesSave": (bool(previous.users) or bool(previous.use_fake_user))
                                    if previous is not None else None,
        }

    # `act` was resolved above, before animation_data_create() - see the note there.

    # A SLOTTED ACTION PLAYS ONLY THROUGH A SLOT. From Blender 4.4 an action keeps its curves per
    # slot and an object plays the slot it is bound to - so an action keyed on one object and
    # handed to a second bound NO slot there, and the second object sat still while list_actions
    # counted it as a user and this op answered ok. Measured on 4.4.0, 5.0.1 and 5.2.1: after
    # animation_data.action = act the second object's action_slot is None and it does not move;
    # binding it to the first object's slot makes both play, and the slot lists both as users.
    # Found 2026-09-13 by the bounce showcase's second ball. RESOLVED BEFORE anything is assigned,
    # so a refusal here changes nothing. 3.6 and 4.2 have no slots and take the old path.
    slotted = hasattr(obj.animation_data, "action_slot")
    want_slot = take(params, "slot", default=None, kind=str)
    if want_slot and not slotted:
        raise MifOpError("'slot' names an action slot, and Blender %s has none - its actions are "
                         "not slotted. Drop it. NOTHING was changed." % bpy.app.version_string)
    slot = None
    if slotted:
        fits = [s for s in getattr(act, "slots", [])
                if getattr(s, "target_id_type", "OBJECT") in ("OBJECT", "UNSPECIFIED")]
        if want_slot:
            slot = next((s for s in fits if want_slot in (s.identifier, s.name_display)), None)
            if slot is None:
                raise MifOpError("action '%s' has no object slot named '%s'. Its object slots: %s. "
                                 "NOTHING was changed." % (act.name, want_slot,
                                                           ", ".join(s.identifier for s in fits) or "<none>"))
        else:
            own = [s for s in fits if s.name_display == obj.name]
            if own:
                slot = own[0]
            elif len(fits) == 1:
                slot = fits[0]
            elif len(fits) > 1:
                raise MifOpError("action '%s' has %d object slots (%s) and none is named for '%s', so "
                                 "which one it plays is a choice - pass slot. NOTHING was changed."
                                 % (act.name, len(fits), ", ".join(s.identifier for s in fits),
                                    obj.name))

    reset = 0
    if rest_pose:
        for pb in obj.pose.bones:
            pb.location = (0.0, 0.0, 0.0)
            pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            pb.rotation_euler = (0.0, 0.0, 0.0)
            pb.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
            pb.scale = (1.0, 1.0, 1.0)
            reset += 1
    obj.animation_data.action = act
    if obj.animation_data.action is not act:
        raise MifOpError("assigned '%s' to '%s' but it did not take. Do not trust this state."
                         % (want, obj.name))
    bound = None
    if slotted:
        now = obj.animation_data.action_slot
        if slot is not None and (now is None or now.identifier != slot.identifier):
            obj.animation_data.action_slot = slot
        bound = obj.animation_data.action_slot
        if len(getattr(act, "slots", [])) and bound is None:
            raise MifOpError("assigned '%s' to '%s' but no slot is bound, so it animates nothing. Do "
                             "not trust this state." % (act.name, obj.name))
    return {
        "object": obj.name,
        "action": act.name,
        # None on a build without slots, and on an action with no slots yet (nothing keyed).
        "slot": bound.identifier if bound is not None else None,
        "slotUsers": sorted(u.name for u in bound.users()) if bound is not None else None,
        # ANOTHER USER OF THE SAME SLOT PLAYS AND KEYS THE SAME CURVES - edit_action_slots ASSIGN newSlot gives one its own
        "sharedWith": sorted(u.name for u in bound.users() if u != obj) if bound is not None else None,
        "previousAction": prev_row["name"] if prev_row else None,
        "previousSurvivesSave": (bool(previous.users) or bool(previous.use_fake_user))
                                if previous is not None else None,
        "info": _action_row(act),
        "poseBonesReset": reset if rest_pose else None,
    }


def _sample_world(obj, frames):
    """Evaluated world matrices at frames, as (loc, quat) pairs. The measurement bake_to_keyframes
    is judged by - see its docstring for why a key COUNT is not one."""
    sc = bpy.context.scene
    out = []
    for f in frames:
        sc.frame_set(int(f))
        dg = bpy.context.evaluated_depsgraph_get()
        mw = obj.evaluated_get(dg).matrix_world
        out.append((mw.to_translation().copy(), mw.to_quaternion().copy()))
    return out


def _sample_bones(obj, frames):
    """Evaluated WORLD matrices of every pose bone at frames, one {name: (loc, quat)} per frame - the pose half
    of bake_to_keyframes' measurement. An armature's own world matrix never moves with its pose, so for an
    armature _sample_world alone measures nothing."""
    sc = bpy.context.scene
    out = []
    for f in frames:
        sc.frame_set(int(f))
        dg = bpy.context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(dg)
        row = {}
        for pb in ev.pose.bones:
            m = ev.matrix_world @ pb.matrix
            row[pb.name] = (m.to_translation().copy(), m.to_quaternion().copy())
        out.append(row)
    return out


def op_bake_to_keyframes(params):
    """Bake evaluated motion into real keyframes, and PROVE the motion survived.

    WHY THIS MATTERS MORE THAN IT SOUNDS. bake_physics bakes POINT CACHES, and no exporter reads
    them - not FBX, not glTF, not Alembic through this addon. So a rigid-body simulation authored
    through this bridge could be rendered here and handed to NOTHING. Constraints and drivers have
    the same problem one step removed: they evaluate correctly in Blender and export as a static
    object, because an exporter writes keyframes and a constraint is not one.

    params:
      object (str, required)
      frameStart / frameEnd (int)   defaults to the scene range
      step (int)                    default 1
      visualKeying (bool)           default TRUE. Off, the bake records the object's OWN transform
                                    and throws away everything a constraint or simulation was
                                    contributing - which is the normal way this goes wrong.
      clearConstraints (bool)       default False. Removes the constraints after baking; without
                                    this they keep evaluating ON TOP of the new keys.
      clearParents (bool)           default False.
      removeRigidBody (bool)        default False. A baked object whose rigid body is still active
                                    is still driven by the sim, and the keys are ignored.

    AN ARMATURE BAKES ITS POSE TOO. Its motion is in the pose, and baking the object alone keyed an IK arm
    standing still and reported motionPreserved - the check read the object's world matrix, which a pose never
    moves (found 2026-09-13 building the robot-arm showcase). So an ARMATURE bakes OBJECT and POSE, every bone;
    each bone is measured the way the object is; and clearConstraints removes bone constraints too, because
    Blender's own bake clears them when it bakes POSE. Every bone, not a chosen few: the operator's bone filter
    (only_selected) takes its armatures from context.selected_pose_bones (bl_operators/anim.py,
    BakeAction.execute), and this op does not enter pose mode.

    THE POSTCONDITION IS THE MOTION, NOT THE KEY COUNT. Producing the right NUMBER of keyframes
    while losing the motion is the normal failure when visual keying is off, and a key count cannot
    see it. So the evaluated world matrix is sampled across the range BEFORE the bake, again after,
    and the maximum position and rotation error between them is reported. A bake that kept the keys
    and lost the movement shows up as a large error rather than as a success.
    """
    reject_unknown(params, {"object", "name", "frameStart", "frameEnd", "step", "visualKeying",
                            "clearConstraints", "clearParents", "removeRigidBody", "allowSimulation"},
                   "bake_to_keyframes")
    obj = get_object(take(params, "object", "name", required=True))
    sc = bpy.context.scene
    f0 = finite_int(take_float(params, "frameStart", default=sc.frame_start), "frameStart")
    f1 = finite_int(take_float(params, "frameEnd", default=sc.frame_end), "frameEnd")
    if f1 < f0:
        raise MifOpError("frameEnd (%d) is before frameStart (%d). NOTHING was baked." % (f1, f0))
    step = finite_int(take_float(params, "step", default=1), "step")
    if step < 1:
        raise MifOpError("step must be at least 1, got %d. NOTHING was baked." % step)
    visual = take_bool(params, "visualKeying", default=True)
    # EVERY FLAG READ HERE, before the bake. clearConstraints and clearParents were read as the operator's own
    # arguments (so they ran first, which was fine), but removeRigidBody was read AFTER the bake: {removeRigidBody:
    # "flase"} was refused with the object already baked to keyframes (measured on 5.0.1, 2026-09-21).
    clear_constraints = take_bool(params, "clearConstraints", default=False)
    clear_parents = take_bool(params, "clearParents", default=False)
    remove_rigid = take_bool(params, "removeRigidBody", default=False)
    # THE BAKE PLAYS frameStart..frameEnd, so every live simulation steps through it - a rigid body world nobody baked
    # among them. That can be the point (keying a simulation), and allowSimulation says so; bake_physics first keys a
    # bake instead of stepping one.
    sim_reply = simulation_guard(params, [f0, f1], "bake_to_keyframes", tail="NOTHING was baked.")

    # THE PROBE FRAMES, spread across the range rather than taken from one end - a bake that loses
    # the motion often still matches at the first frame, where the object has not moved yet.
    span = f1 - f0
    probes = sorted({f0 + int(round(span * t / 6.0)) for t in range(7)}) if span else [f0]
    started_on = sc.frame_current
    is_armature = obj.type == "ARMATURE" and obj.pose is not None
    try:
        before = _sample_world(obj, probes)
        bones_before = _sample_bones(obj, probes) if is_armature else []
    finally:
        sc.frame_set(started_on)

    had_constraints = len(obj.constraints)
    had_bone_constraints = sum(len(pb.constraints) for pb in obj.pose.bones) if is_armature else 0
    had_parent = obj.parent.name if obj.parent else None
    had_rigid = getattr(obj, "rigid_body", None) is not None

    snap = selection_snapshot()
    try:
        # A LIST, not a bare object. select_only iterates its argument, so passing the
        # object itself raises "TypeError: 'Object' object is not iterable" - and this was
        # the ONLY one of nine call sites getting it wrong, so bake_to_keyframes had NEVER
        # WORKED on any Blender since it was written. Found 2026-09-03 by
        # blender_version_matrix, and only after a payload fix let the op past its own
        # guards - it had been refusing for a bad argument and never reaching this line.
        select_only([obj])
        try:
            # only_selected OFF for an armature: on, BakeAction takes its objects from
            # context.selected_pose_bones rather than from the selected objects - see the docstring.
            bpy.ops.nla.bake(frame_start=f0, frame_end=f1, step=step,
                             only_selected=not is_armature, visual_keying=visual,
                             clear_constraints=clear_constraints,
                             clear_parents=clear_parents,
                             bake_types={"OBJECT", "POSE"} if is_armature else {"OBJECT"})
        except RuntimeError as exc:
            raise MifOpError("Blender's bake refused: %s. NOTHING was baked." % exc)
    finally:
        selection_restore(snap)

    rb_error = None
    if remove_rigid and had_rigid:
        # THROUGH THE OPERATOR, under the override add_rigid_body uses to create one. This used to
        # unlink the object from the rigid body world's collection instead, on the claim that the
        # data path "does the same thing". MEASURED 2026-09-12 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and
        # 5.2.1, it does not: the object leaves the world's collection and KEEPS obj.rigid_body,
        # through a view-layer update and a frame change, on every build - so removeRigidBody had
        # never removed a rigid body. object_remove clears both on all five. A refusal still does
        # not raise - the keys are baked, which is this op's job - but it is no longer silent:
        # rigidBodyAfter and removeRigidBodyWarning below report it, with Blender's own words.
        from .ops_physics import _ctx_override
        try:
            with _ctx_override(obj):
                bpy.ops.rigidbody.object_remove()
        except RuntimeError as exc:
            rb_error = str(exc)
    # WHAT THE CLEAR FLAGS LEFT, measured rather than assumed. hadConstraints and hadRigidBody said
    # what was there BEFORE and nothing said what was left, so clearConstraints, clearParents and
    # removeRigidBody could not be seen from the response at all - and the unlink above swallows its
    # own failure. Found 2026-09-12 by audit_blender_inert_params, once a precondition put a
    # constraint and a rigid body on MifBake: both flags still read as changing nothing.
    constraints_after = len(obj.constraints)
    parent_after = obj.parent.name if obj.parent else None
    rigid_after = getattr(obj, "rigid_body", None) is not None

    curves = _fcurves(obj)
    keys = sum(len(fc.keyframe_points) for fc in curves)
    if keys == 0:
        raise MifOpError("the bake produced NO keyframes on '%s'. Nothing was captured - check the "
                         "frame range and that the object actually moves." % obj.name)

    try:
        after = _sample_world(obj, probes)
        bones_after = _sample_bones(obj, probes) if is_armature else []
    finally:
        sc.frame_set(started_on)

    max_pos = 0.0
    max_rot = 0.0
    for (lb, qb), (la, qa) in zip(before, after):
        max_pos = max(max_pos, (la - lb).length)
        try:
            max_rot = max(max_rot, qb.rotation_difference(qa).angle)
        except (AttributeError, ValueError):
            pass
    bone_pos = 0.0
    bone_rot = 0.0
    worst_bone = None
    for rb, ra in zip(bones_before, bones_after):
        for name, (lb, qb) in rb.items():
            if name not in ra:
                continue
            la, qa = ra[name]
            if (la - lb).length > bone_pos:
                bone_pos, worst_bone = (la - lb).length, name
            try:
                bone_rot = max(bone_rot, qb.rotation_difference(qa).angle)
            except (AttributeError, ValueError):
                pass

    out = {
        "object": obj.name,
        "frameStart": f0,
        "frameEnd": f1,
        "step": step,
        "visualKeying": visual,
        "curveCount": len(curves),
        "keyframeTotal": keys,
        "probeFrames": probes,
        # THE MEASUREMENT. Small means the baked keys reproduce what the scene was doing; large
        # means the keys exist and the motion is gone, which is what visual_keying=False does and
        # what a key count cannot see.
        "maxPositionError": round(float(max_pos), 6),
        "maxRotationErrorRadians": round(float(max_rot), 6),
        "motionPreserved": max_pos < 1e-3 and max_rot < 1e-3 and bone_pos < 1e-3 and bone_rot < 1e-3,
        "bakeTypes": ["OBJECT", "POSE"] if is_armature else ["OBJECT"],
        "hadConstraints": had_constraints,
        "hadRigidBody": had_rigid,
        "hadParent": had_parent,
        "constraintsAfter": constraints_after,
        "parentAfter": parent_after,
        "rigidBodyAfter": rigid_after,
        "exportNote": ("This is why the op exists: bake_physics writes POINT CACHES and no "
                       "exporter reads them, and a constraint or driver exports as a static "
                       "object because an exporter writes keyframes and a constraint is not one. "
                       "These are real keyframes and will travel."),
        "stillDrivenNote": ("A source left in place keeps evaluating ON TOP of the new keys - "
                            "constraints unless clearConstraints, and an active rigid body unless "
                            "removeRigidBody. hadConstraints/hadRigidBody say what was there, and "
                            "constraintsAfter/rigidBodyAfter what is left."),
    }
    if is_armature:
        # THE POSE HALF, measured like the object half: every bone's evaluated world matrix at the same probe
        # frames, before and after. A bone constraint left in place still drives the pose ON TOP of the keys.
        out.update({
            "bonesBaked": len(obj.pose.bones),
            "maxBonePositionError": round(float(bone_pos), 6),
            "maxBoneRotationErrorRadians": round(float(bone_rot), 6),
            "worstBone": worst_bone,
            "hadBoneConstraints": had_bone_constraints,
            "boneConstraintsAfter": sum(len(pb.constraints) for pb in obj.pose.bones),
        })
    if remove_rigid and had_rigid and rigid_after:
        out["removeRigidBodyWarning"] = (("removeRigidBody was asked for and '%s' still has its rigid "
                                          "body, so the simulation still drives it and the new keys "
                                          "are ignored." % obj.name)
                                         + (" Blender refused the removal: %s" % rb_error
                                            if rb_error else ""))
    out.update(sim_reply)
    return out


def _marker_row(m):
    return {
        "name": m.name,
        "frame": int(m.frame),
        # THE BOUND CAMERA is the whole reason markers matter beyond being labels: binding one
        # makes the scene CUT to that camera at that frame, which is how a multi-camera edit is
        # actually done in Blender and is invisible from anywhere else in this addon.
        "camera": m.camera.name if getattr(m, "camera", None) else None,
    }


def op_list_markers(params):
    """Every timeline marker, and which camera each one cuts to.

    A marker with a bound camera makes the scene switch to it at that frame, so a render can use
    several cameras without anything else in the file recording that - list_cameras reports which
    camera is active NOW, and this reports which one each part of the timeline uses.

    params:
      limit (int)                  return at most this many markers. Unlimited by default.
      nameExcludes (str|list)      drop markers whose name contains any of these

    BOUNDED BECAUSE timeline_markers IS SCENE-WIDE. Unlike list_particles or list_constraints,
    which are bounded by the one object the caller named, a marker list grows with the edit - a
    long sequence carries hundreds, and nothing in the request limits it.
    """
    reject_unknown(params, ("limit", "nameExcludes"), "list_markers")
    sc = bpy.context.scene
    rows = sorted((_marker_row(m) for m in sc.timeline_markers), key=lambda r: r["frame"])
    rows = exclude_rows(rows, params)
    # THE CAMERA SUMMARY IS COUNTED ON WHAT MATCHED, NEVER ON WHAT WAS RETURNED. A scene that cuts
    # between cameras must not stop cutting because the caller asked for the first ten markers -
    # the same rule keyframeTotal follows in list_keyframes, and the reason `count` is `matched`.
    bound = [r for r in rows if r["camera"]]
    matched = len(rows)
    rows, extra = limit_rows(rows, params, "marker(s)")
    return dict({
        "count": matched,
        "markers": rows,
        "boundCameraCount": len(bound),
        # A scene with ANY bound marker cuts between cameras, and scene.camera is then only the
        # camera for frames before the first binding. Worth saying outright.
        "sceneCutsBetweenCameras": len(bound) > 1,
        "sceneCamera": sc.camera.name if sc.camera else None,
    }, **extra)


def op_set_marker(params):
    """Create, move, rename, camera-bind or delete a timeline marker.

    params:
      name (str, required)   the marker to act on, matched by name
      frame (int)            create it here, or MOVE it here if it already exists
      camera (str)           bind a camera - the scene cuts to it at this marker's frame
      unbindCamera (bool)    clear the binding
      rename (str)           new name
      delete (bool)          remove the marker

    Markers are matched BY NAME, and Blender allows duplicates - two markers can share one. The
    response reports how many matched, because acting on the first of several silently is how a
    caller ends up moving the wrong one.
    """
    reject_unknown(params, {"name", "frame", "camera", "unbindCamera", "rename", "delete"},
                   "set_marker")
    sc = bpy.context.scene
    name = take(params, "name", default=None, kind=str)
    if not name:
        raise MifOpError("'name' is required - which marker. NOTHING was changed.")
    matches = [m for m in sc.timeline_markers if m.name == str(name)]

    if take_bool(params, "delete", default=False):
        if not matches:
            raise MifOpError("no marker named '%s' to delete. NOTHING was changed." % name)
        for m in list(matches):
            sc.timeline_markers.remove(m)
        remaining = [m for m in sc.timeline_markers if m.name == str(name)]
        if remaining:
            raise MifOpError("removed %d marker(s) named '%s' but %d remain. Do not trust this "
                             "state." % (len(matches), name, len(remaining)))
        return {"name": str(name), "deleted": len(matches),
                "markers": len(sc.timeline_markers)}

    frame = take_float(params, "frame", default=None)
    cam_name = take(params, "camera", default=None, kind=str)
    cam = None
    if cam_name:
        cam = bpy.data.objects.get(str(cam_name))
        if cam is None:
            raise MifOpError("no object named '%s' to bind. NOTHING was changed." % cam_name)
        if cam.type != "CAMERA":
            raise MifOpError("'%s' is a %s, not a CAMERA - only a camera can be bound to a "
                             "marker. NOTHING was changed." % (cam_name, cam.type))
    # Read BEFORE the cam_name test - see audit_blender_shortcircuit_take. The second read below
    # sits in an elif and is equally skippable, so neither one validated this on its own.
    unbind_camera = take_bool(params, "unbindCamera", default=False)
    if cam_name and unbind_camera:
        raise MifOpError("pass a camera to bind OR unbindCamera, not both. NOTHING was changed.")
    # rename AND THE FRAME AS A WHOLE NUMBER, read before anything moves. rename was read inside the write loop,
    # after the marker had been created or moved, so {frame: 50, rename: 5} left a new marker behind under the
    # refusal; and a marker's frame is an integer, so int(12.5) put it on 12 in silence (measured on 5.0.1,
    # 2026-09-21).
    try:
        new_name = take(params, "rename", default=None, kind=str)
        want_frame = finite_int(frame, "frame") if frame is not None else None
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was changed." % str(exc).rstrip("."))

    created = False
    if not matches:
        if want_frame is None:
            raise MifOpError("no marker named '%s' exists, so 'frame' is required to create one. "
                             "NOTHING was changed." % name)
        m = sc.timeline_markers.new(str(name), frame=want_frame)
        created = True
        matches = [m]
    else:
        if want_frame is not None:
            for m in matches:
                m.frame = want_frame

    for m in matches:
        if cam is not None:
            m.camera = cam
        elif unbind_camera:
            m.camera = None
        if new_name:
            m.name = str(new_name)

    final_name = matches[0].name
    rows = [_marker_row(m) for m in sc.timeline_markers if m.name == final_name]
    return {
        "name": final_name,
        "created": created,
        # NAMED because Blender permits duplicates and this op acts on ALL of them: a caller who
        # thinks they moved one marker and moved three should be told, not left to discover it.
        "matched": len(matches),
        "markers": rows,
        "totalMarkers": len(sc.timeline_markers),
    }


def _driver_target(obj, path, index):
    """The existing driver fcurve for a path, or None."""
    ad = getattr(obj, "animation_data", None)
    if ad is None:
        return None
    for dr in (getattr(ad, "drivers", None) or []):
        if dr.data_path == path and (index is None or dr.array_index == int(index)):
            return dr
    return None


def _driver_owner(obj, path, tail):
    """(the datablock a driver on `path` belongs to, the path from it, where) - resolved, never guessed.

    A DRIVER LIVES ON THE ID THAT OWNS THE PROPERTY, and Blender refuses one whose path spans two IDs: add_driver
    passed data.shape_keys.key_blocks["X"].value - the path Blender's own UI copies for a shape key - straight to
    obj.driver_add, and it came back RAW, "driver_add(): path spans ID blocks". Measured 2026-09-13 on 4.4, after the
    morph showcase found the same boundary in set_keyframe. So a path that lands on the object's own data or on its
    mesh's Key is re-rooted there with the path Blender gives from that ID; the Key-relative forms (key_blocks[...],
    shape_keys.key_blocks[...]) resolve there directly; any other datablock is refused, named.
    """
    key = _shape_key_block(obj)
    bases = []
    if key is not None and path.startswith("key_blocks"):
        bases.append(key)
    if obj.data is not None and path.startswith("shape_keys."):
        bases.append(obj.data)
    bases.append(obj)
    err = None
    for base in bases:
        try:
            base.path_resolve(path)
        except (ValueError, AttributeError, TypeError) as exc:
            err = exc
            continue
        holder, leaf = base, path
        if "." in path:
            head, leaf = path.rsplit(".", 1)
            holder = base.path_resolve(head)
        hid = getattr(holder, "id_data", None)
        if hid is None or hid == obj:
            return obj, path, "object"
        if hid == key or (obj.data is not None and hid == obj.data):
            try:
                rel = holder.path_from_id(leaf)
            except (ValueError, TypeError) as exc:
                raise MifOpError("'%s' lands on %s '%s' and its path from there could not be read (%s). %s"
                                 % (path, type(hid).__name__, hid.name, exc, tail))
            return hid, rel, ("its shape keys (%s '%s')" if hid == key else "its data (%s '%s')") % (
                type(hid).__name__, hid.name)
        raise MifOpError("'%s' runs from '%s' into %s '%s', and Blender keeps a driver on the datablock that owns "
                         "the property - drive that datablock's own path. %s"
                         % (path, obj.name, type(hid).__name__, hid.name, tail))
    raise MifOpError("'%s' does not resolve on '%s' (%s), so a driver on it would be created invalid and "
                     "evaluate to zero forever. %s" % (path, obj.name, err, tail))


_TRANSFORM_TYPES = ("LOC_X", "LOC_Y", "LOC_Z", "ROT_X", "ROT_Y", "ROT_Z", "ROT_W", "SCALE_X", "SCALE_Y", "SCALE_Z",
                    "SCALE_AVG")
_TRANSFORM_SPACES = ("WORLD_SPACE", "TRANSFORM_SPACE", "LOCAL_SPACE")
_VAR_KEYS = {"name", "object", "dataPath", "type", "bone", "transformType", "rotationMode", "space", "object2", "bone2",
             "space2"}


def _driver_var_spec(v, i):
    """A driver variable's type and targets, every name resolved and checked before the driver exists."""
    extra = set(v) - _VAR_KEYS
    if extra:
        raise MifOpError("variables[%d] has unknown key(s) %s (it takes %s). NOTHING was added."
                         % (i, ", ".join(sorted(extra)), ", ".join(sorted(_VAR_KEYS))))
    vtype = str(v.get("type") or "SINGLE_PROP").upper()
    if vtype not in ("SINGLE_PROP", "TRANSFORMS", "LOC_DIFF", "ROTATION_DIFF"):
        raise MifOpError("variables[%d].type must be SINGLE_PROP, TRANSFORMS, LOC_DIFF or ROTATION_DIFF, got %r. NOTHING "
                         "was added." % (i, v.get("type")))
    if vtype == "SINGLE_PROP":
        for k in ("bone", "transformType", "rotationMode", "space", "object2", "bone2", "space2"):
            if v.get(k) is not None:
                raise MifOpError("variables[%d].%s is for TRANSFORMS, LOC_DIFF or ROTATION_DIFF. NOTHING was added."
                                 % (i, k))
        return {"type": vtype}
    if v.get("dataPath") is not None:
        raise MifOpError("variables[%d] is %s, which reads a transform, not a dataPath. NOTHING was added." % (i, vtype))

    def target(okey, bkey, skey):
        name = v.get(okey)
        ob = bpy.data.objects.get(str(name)) if name else None
        if ob is None:
            raise MifOpError("variables[%d].%s must name an object (got %r) - a %s variable reading nothing evaluates "
                             "to zero silently. NOTHING was added." % (i, okey, name, vtype))
        bone = v.get(bkey)
        if bone is not None and (ob.type != "ARMATURE" or ob.data.bones.get(str(bone)) is None):
            raise MifOpError("variables[%d].%s %r is not a bone of '%s'. NOTHING was added." % (i, bkey, bone, ob.name))
        space = str(v.get(skey) or "WORLD_SPACE").upper()
        if space not in _TRANSFORM_SPACES:
            raise MifOpError("variables[%d].%s must be one of %s, got %r. NOTHING was added."
                             % (i, skey, ", ".join(_TRANSFORM_SPACES), v.get(skey)))
        return ob, str(bone) if bone is not None else None, space
    out = {"type": vtype, "targets": [target("object", "bone", "space")]}
    if vtype == "TRANSFORMS":
        tt = str(v.get("transformType") or "LOC_X").upper()
        if tt not in _TRANSFORM_TYPES:
            raise MifOpError("variables[%d].transformType must be one of %s, got %r. NOTHING was added."
                             % (i, ", ".join(_TRANSFORM_TYPES), v.get("transformType")))
        rm = str(v.get("rotationMode") or "AUTO").upper()
        modes = [e.identifier for e in bpy.types.DriverTarget.bl_rna.properties["rotation_mode"].enum_items]
        if rm not in modes:
            raise MifOpError("variables[%d].rotationMode must be one of this Blender's %s, got %r. NOTHING was added."
                             % (i, ", ".join(modes), v.get("rotationMode")))
        out.update(transformType=tt, rotationMode=rm)
        for k in ("object2", "bone2", "space2"):
            if v.get(k) is not None:
                raise MifOpError("variables[%d].%s is for LOC_DIFF and ROTATION_DIFF. NOTHING was added." % (i, k))
    else:
        for k in ("transformType", "rotationMode"):
            if v.get(k) is not None:
                raise MifOpError("variables[%d].%s is for TRANSFORMS. NOTHING was added." % (i, k))
        out["targets"].append(target("object2", "bone2", "space2"))
    return out


_HANDLES = ("AUTO_CLAMPED", "VECTOR", "FREE", "ALIGNED", "AUTO")


def _driver_curve_spec(params):
    """(points [(x, y, handle)], extrapolation) for curve or mapRange, or (None, None). Checked before anything."""
    curve, mr = params.get("curve"), params.get("mapRange")
    if curve is not None and mr is not None:
        raise MifOpError("pass curve OR mapRange - both write the driver's response curve. NOTHING was added.")
    if mr is not None:
        if not isinstance(mr, dict) or set(mr) - {"inMin", "inMax", "outMin", "outMax", "clamp"}:
            raise MifOpError("mapRange is {inMin, inMax, outMin, outMax, clamp}; got %r. NOTHING was added." % (mr,))
        try:
            a, b = take_float(mr, "inMin", default=0.0), take_float(mr, "inMax", default=1.0)
            c, d = take_float(mr, "outMin", default=0.0), take_float(mr, "outMax", default=1.0)
        except MifOpError as exc:
            raise MifOpError("%s NOTHING was added." % str(exc).rstrip("."))
        if a == b:
            raise MifOpError("mapRange inMin and inMax are both %g - no range to map. NOTHING was added." % a)
        clamp = take_bool(mr, "clamp", default=True)
        return [(a, c, "VECTOR"), (b, d, "VECTOR")], "CONSTANT" if clamp else "LINEAR"
    if curve is None:
        return None, None
    if not isinstance(curve, dict) or set(curve) - {"points", "extrapolation"}:
        raise MifOpError("curve is {points [{x, y, handle}], extrapolation}; got %r. NOTHING was added." % (curve,))
    pts = curve.get("points")
    if not isinstance(pts, (list, tuple)) or len(pts) < 2:
        raise MifOpError("curve.points needs at least two {x, y}. NOTHING was added.")
    out = []
    for j, p in enumerate(pts):
        if not isinstance(p, dict) or set(p) - {"x", "y", "handle"}:
            raise MifOpError("curve.points[%d] is {x, y, handle}; got %r. NOTHING was added." % (j, p))
        try:
            x, y = take_float(p, "x"), take_float(p, "y")
        except MifOpError as exc:
            raise MifOpError("%s NOTHING was added." % str(exc).rstrip("."))
        h = str(p.get("handle") or "AUTO_CLAMPED").upper()
        if h not in _HANDLES:
            raise MifOpError("curve.points[%d].handle must be one of %s, got %r. NOTHING was added."
                             % (j, ", ".join(_HANDLES), p.get("handle")))
        out.append((x, y, h))
    xs = [x for x, _y, _h in out]
    if len(set(xs)) != len(xs):
        raise MifOpError("curve.points share an x - one input cannot give two values. NOTHING was added.")
    ex = str(curve.get("extrapolation") or "CONSTANT").upper()
    if ex not in ("CONSTANT", "LINEAR"):
        raise MifOpError("curve.extrapolation must be CONSTANT or LINEAR, got %r. NOTHING was added." % ex)
    return sorted(out), ex


def _three(pts):
    """The first, middle and last input of a response curve."""
    xs = [x for x, _y, _h in pts]
    return [xs[0], (xs[0] + xs[-1]) / 2.0, xs[-1]]


def op_add_driver(params):
    """Wire a property to an expression, and PROVE the driver actually evaluates.

    DRIVERS ARE THE ONE ANIMATION FEATURE THAT FAILS COMPLETELY SILENTLY. A driver with a broken
    expression, or a variable pointing at an object that no longer exists, stays in place and
    evaluates to ZERO. Nothing errors. Nothing warns. The property simply sits at 0 while every
    field a caller can read - the expression, the variable, the data path - looks perfectly
    correct. Blender shows it purple-then-red in the UI and reports it nowhere else.

    params:
      object (str, required)
      dataPath (alias path, required)   the property to drive
      index (int)                       array element; omitted drives element 0 for a vector
      expression (str)                  default "var" - the driver's SCRIPTED expression
      variables (list)                  [{name, object, dataPath}] - each becomes a driver
                                        variable reading a property off another object
                                        (type SINGLE_PROP, the default). type TRANSFORMS {object, bone,
                                        transformType LOC_X..SCALE_AVG, rotationMode, space WORLD_SPACE |
                                        TRANSFORM_SPACE | LOCAL_SPACE} reads the EVALUATED channel - the one a
                                        corrective reads, since SINGLE_PROP reads 0 on a bone IK or a
                                        constraint turns; LOC_DIFF and ROTATION_DIFF take {object, bone,
                                        object2, bone2} (2026-10-04)
      driverType                        SCRIPTED (default) | SUM | AVERAGE | MIN | MAX - only SCRIPTED takes
                                        an expression
      curve {points [{x, y, handle}], extrapolation}   the driver's response curve: keys mapping the
                                        driver's value (x) to the property (y); handle AUTO_CLAMPED (default)
                                        | VECTOR | FREE | ALIGNED | AUTO; extrapolation CONSTANT (default) | LINEAR
      mapRange {inMin, inMax, outMin, outMax, clamp}   the common curve: a straight line from (inMin,
                                        outMin) to (inMax, outMax), held flat beyond unless clamp is false - a
                                        corrective that reaches exactly 1 at the extreme pose

    THE POSTCONDITION IS EVALUATION, not existence. is_valid is checked, and the driven property
    is read back through the depsgraph, because a driver that exists and evaluates to nothing is
    the normal failure and looks identical to a working one from the data.
    """
    reject_unknown(params, {"object", "name", "dataPath", "path", "index", "expression",
                            "variables", "driverType", "curve", "mapRange"}, "add_driver")
    obj = get_object(take(params, "object", "name", required=True))
    path = take(params, "dataPath", "path", default=None, kind=str)
    if not path:
        raise MifOpError("'dataPath' is required - the property to drive. NOTHING was added.")
    index = take_int(params, "index", default=None)  # take_int, NOT int(): a bad value must REFUSE, not raise ValueError out of the op.

    # THE PROPERTY MUST EXIST BEFORE IT CAN BE DRIVEN. Blender will happily create a driver on a
    # path that resolves to nothing and leave it permanently invalid, which is the silent failure
    # this op is arranged to prevent rather than to produce.
    owner, dpath, where = _driver_owner(obj, path, "NOTHING was added.")

    if _driver_target(owner, dpath, index) is not None:
        raise MifOpError("'%s' already has a driver on '%s'%s. Remove it first - Blender allows a "
                         "second and the result is not what anybody means. NOTHING was added."
                         % (obj.name, path, "" if index is None else "[%s]" % index))

    variables = params.get("variables") or []
    if not isinstance(variables, (list, tuple)):
        raise MifOpError("'variables' must be a list of {name, object, dataPath}. NOTHING was "
                         "added.")
    resolved = []
    for i, v in enumerate(variables):
        if not isinstance(v, dict):
            raise MifOpError("variables[%d] must be an object, got %r. NOTHING was added."
                             % (i, v))
        vname = str(v.get("name") or "var")
        vobj_name = v.get("object")
        vobj = bpy.data.objects.get(str(vobj_name)) if vobj_name else None
        if vobj_name and vobj is None:
            raise MifOpError("variables[%d] targets no object named '%s'. A driver variable "
                             "pointing at nothing evaluates to zero SILENTLY, so this is refused "
                             "rather than created. NOTHING was added." % (i, vobj_name))
        resolved.append((vname, vobj, str(v.get("dataPath") or "location"), _driver_var_spec(v, i)))

    before = None
    try:
        before = owner.path_resolve(dpath)
        before = list(before) if hasattr(before, "__len__") else float(before)
    except (TypeError, ValueError):
        before = None

    # THE EXPRESSION, READ BEFORE driver_add. It was read after the driver existed, so {expression: 5} was refused
    # with the driver left on the property (measured on 5.0.1, 2026-09-21) - and the retry the refusal invites hits
    # "already has a driver".
    try:
        expression = str(take(params, "expression", default="var", kind=str))
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was added." % str(exc).rstrip("."))
    dtype = str(take(params, "driverType", default="SCRIPTED", kind=str)).upper()
    if dtype not in ("SCRIPTED", "SUM", "AVERAGE", "MIN", "MAX"):
        raise MifOpError("driverType must be SCRIPTED, SUM, AVERAGE, MIN or MAX, got %r. NOTHING was added." % dtype)
    if dtype != "SCRIPTED" and params.get("expression") is not None:
        raise MifOpError("an expression is read only by a SCRIPTED driver; %s combines its variables itself. NOTHING "
                         "was added." % dtype)
    curve_pts, extrap = _driver_curve_spec(params)
    # POSITIONAL, NOT KEYWORDS. driver_add() takes NO keyword arguments - not on 5.0, and not on
    # 3.6, 4.2 or 4.4 either. This op called it as driver_add(data_path=..., index=...) and had
    # therefore NEVER WORKED ON ANY BLENDER, on any build this addon supports.
    #
    # It shipped and stayed green because the TypeError was caught right here and re-raised as a
    # MifOpError reading "Blender refused to add a driver" - so an API break that had never once
    # succeeded wore the clothes of a legitimate refusal, and every check that asks only "did it
    # refuse politely" agreed with it.
    #
    # Found 2026-09-03 by blender_version_matrix, and only because the reach report made it
    # conspicuous that this op never reached its bpy calls on any version. Verified against all four
    # installs: keywords raise TypeError everywhere, positional returns an FCurve everywhere.
    try:
        fc = owner.driver_add(dpath) if index is None else owner.driver_add(dpath, int(index))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise MifOpError("Blender refused to add a driver on '%s' (%s): %s. NOTHING was added."
                         % (path, where, exc))
    # driver_add returns a LIST for a vector property when no index was given.
    if isinstance(fc, list):
        fc = fc[0]

    drv = fc.driver
    drv.type = dtype
    for vname, vobj, vpath, vspec in resolved:
        var = drv.variables.new()
        var.name = vname
        var.type = vspec["type"]
        if vspec["type"] == "SINGLE_PROP":
            if vobj is not None:
                var.targets[0].id = vobj
                var.targets[0].data_path = vpath
        else:
            for t, (tob, bone, space) in zip(var.targets, vspec["targets"]):
                t.id = tob
                if bone:
                    t.bone_target = bone
                if space:
                    t.transform_space = space
            if vspec["type"] == "TRANSFORMS":
                var.targets[0].transform_type = vspec["transformType"]
                var.targets[0].rotation_mode = vspec["rotationMode"]
    if dtype == "SCRIPTED":
        drv.expression = expression
    if curve_pts is not None:
        # THE GENERATOR GOES FIRST: a new driver curve carries a Generator modifier that maps value to value, and
        # keyframes beside it are overridden by it, so a response curve needs it removed.
        for m in list(fc.modifiers):
            fc.modifiers.remove(m)
        # AND THE KEYS IT ALREADY HAS: a new driver curve on 5.2.1 carried a point at (1, 1), which a map from 0-1.047
        # kept as a third key and bent the curve (0.7136 at mid-range where the line gives 0.5).
        while len(fc.keyframe_points):
            fc.keyframe_points.remove(fc.keyframe_points[0])
        for x, y, handle in curve_pts:
            kp = fc.keyframe_points.insert(x, y)
            kp.handle_left_type = kp.handle_right_type = handle
        fc.extrapolation = extrap
        fc.update()

    # EACH TARGET IS TAGGED before the first evaluation. A custom property written without a tag - from Blender's own
    # Python console, say - is not in the target's evaluated copy, the driver resolves nothing, and Blender marks it
    # invalid for good. Measured on all five builds: that order gave isValid false and z 0.0; tagging the target in the
    # same call gave isValid true and 1.45. set_custom_property tags its own writes; this covers every other writer.
    for _vname, vobj, _vpath, vspec in resolved:
        for tob in [vobj] + [t[0] for t in vspec.get("targets", [])]:
            if tob is not None:
                tob.update_tag()
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = owner.evaluated_get(dg)
    try:
        after = ev.path_resolve(dpath)
        after = list(after) if hasattr(after, "__len__") else float(after)
    except (ValueError, AttributeError, TypeError):
        after = None

    valid = bool(getattr(drv, "is_valid", True))
    return {
        "object": obj.name,
        "dataPath": dpath,
        # WHERE THE DRIVER LIVES when that is not the object - a shape key's on the Key, a light's energy on the light
        # - and the path as it was asked for, so a caller can find it again either way.
        "drivenOn": where,
        "requestedPath": path if (dpath != path or where != "object") else None,
        "index": index,
        "expression": drv.expression,
        "driverType": drv.type,
        "variables": [{"name": v.name, "type": v.type,
                       "target": (v.targets[0].id.name if v.targets and v.targets[0].id
                                  else None),
                       "targetPath": (v.targets[0].data_path if v.targets else None),
                       "bone": (v.targets[0].bone_target or None) if v.type != "SINGLE_PROP" else None,
                       "transformType": v.targets[0].transform_type if v.type == "TRANSFORMS" else None}
                      for v in drv.variables],
        # THE RESPONSE CURVE AT THREE INPUTS, read off the curve itself (FCurve.evaluate), when one was written.
        "curve": ({"points": [[round(k.co[0], 6), round(k.co[1], 6)] for k in fc.keyframe_points],
                   "extrapolation": fc.extrapolation,
                   "evaluatedAt": {str(x): round(fc.evaluate(x), 6) for x in _three(curve_pts)}}
                  if curve_pts is not None else None),
        # THE MEASUREMENT. is_valid is Blender's own verdict, and the evaluated value is the proof.
        # A driver that exists and evaluates to nothing is the NORMAL failure and is identical to a
        # working one from every other field.
        "isValid": valid,
        "valueBefore": before,
        "valueAfterEvaluated": after,
        # NO STATIC WARNING FIELD. The silent-failure explanation belongs in the tool help, where
        # it is read BEFORE the call - a constant string in the response is something no suite can
        # check and no caller can act on differently, which is what
        # audit_blender_consequence_fields objects to and what open_file's discardedNote was
        # removed for earlier today. isValid and evaluates are the measurements; they carry it.
        "evaluates": valid and after is not None,
    }


def op_remove_driver(params):
    """Remove a driver, and report what the property fell back to.

    params:
      object (str, required)
      dataPath (alias path, required)
      index (int)

    A property with its driver removed returns to whatever it was last set to, which is not
    necessarily what it was showing while driven - so the value is read back rather than assumed.
    """
    reject_unknown(params, {"object", "name", "dataPath", "path", "index"}, "remove_driver")
    obj = get_object(take(params, "object", "name", required=True))
    path = take(params, "dataPath", "path", default=None, kind=str)
    if not path:
        raise MifOpError("'dataPath' is required. NOTHING was removed.")
    index = take_int(params, "index", default=None)  # take_int, NOT int(): a bad value must REFUSE, not raise ValueError out of the op.
    owner, dpath, where = _driver_owner(obj, path, "NOTHING was removed.")
    if _driver_target(owner, dpath, index) is None:
        raise MifOpError("no driver on '%s' for '%s'%s - list them with list_animation_data. "
                         "NOTHING was removed."
                         % (obj.name, path, "" if index is None else "[%s]" % index))

    # POSITIONAL for the same reason as driver_add above - driver_remove() takes no keyword
    # arguments either, so the removal half had never worked on any build either.
    try:
        if index is None:
            owner.driver_remove(dpath)
        else:
            owner.driver_remove(dpath, int(index))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise MifOpError("Blender refused to remove the driver on '%s' (%s): %s." % (path, where, exc))

    still = _driver_target(owner, dpath, index)
    if still is not None:
        raise MifOpError("asked to remove the driver on '%s' but one is still there. Do not trust "
                         "this state." % path)
    bpy.context.view_layer.update()
    try:
        now = owner.path_resolve(dpath)
        now = list(now) if hasattr(now, "__len__") else float(now)
    except (ValueError, AttributeError, TypeError):
        now = None
    return {
        "object": obj.name,
        "dataPath": path,
        "drivenOn": where,
        "index": index,
        "removed": True,
        "valueNow": now,
        "fallbackNote": ("The property has returned to its own stored value, which is not "
                         "necessarily what it was showing while driven."),
    }


def op_add_nla_strip(params):
    """Put an action on an NLA track as a strip - how several clips live on one object.

    An object holds ONE active action at a time. The NLA is how a walk, an idle and a wave coexist
    on the same rig and how they blend, and it is what glTF reads to export multiple clips. None of
    it was reachable.

    params:
      object (str, required)
      action (str, required)    the action to place
      track (str)               track name; a new track is created if it does not exist
      start (int)               first frame, default the action's own start
      stripName (str)           name the strip, default the action's name. NOT `name`.
      name (str)                an ALIAS FOR `object`, not for the strip.

    THE SAME CONTRACT ERROR add_constraint had, found by sweeping for it. This documented
    `name` as the strip name while line 1666 reads `stripName`, and `name` is an alias for
    `object` - so the documented spelling is refused as an alias conflict, and alone it
    names the wrong thing. Measured against a live Blender on 2026-09-04: `name` fails,
    `stripName` works.
      blendType (str)           REPLACE | ADD | SUBTRACT | MULTIPLY
      influence (float)         0..1
      pushDownActive (bool)     default FALSE. See below - this is the trap.

    THE ACTIVE ACTION SHADOWS THE NLA. Blender evaluates animation_data.action ON TOP of the whole
    NLA stack, so an object with an active action set plays THAT and the strips appear to do
    nothing. This op detects it and says so rather than leaving a stack that looks correct and is
    inert; pushDownActive moves the active action onto its own track first, which is what the UI's
    Push Down button does.
    """
    reject_unknown(params, {"object", "name", "action", "track", "start", "stripName",
                            "blendType", "influence", "pushDownActive"}, "add_nla_strip")
    obj = get_object(take(params, "object", "name", required=True))
    act_name = take(params, "action", default=None, kind=str)
    if not act_name:
        raise MifOpError("'action' is required - which action to place. NOTHING was added.")
    act = bpy.data.actions.get(str(act_name))
    if act is None:
        known = [a.name for a in bpy.data.actions][:20]
        raise MifOpError("no action named '%s'. Present: %s. NOTHING was added."
                         % (act_name, ", ".join(known) if known else "<none>"))

    blend = take(params, "blendType", default=None, kind=str)
    if blend:
        blend = str(blend).upper()
        valid = _enum_ids(bpy.types.NlaStrip, "blend_type")
        if valid and blend not in valid:
            raise MifOpError("unknown blendType '%s'. Valid: %s. NOTHING was added."
                             % (blend, ", ".join(sorted(valid))))
    influence = take_float(params, "influence", default=None)
    if influence is not None and not (0.0 <= influence <= 1.0):
        raise MifOpError("influence must be between 0 and 1, got %g. NOTHING was added."
                         % influence)

    # Read BEFORE the action test - see audit_blender_shortcircuit_take.
    push_down = take_bool(params, "pushDownActive", default=False)
    # track, start and stripName TOO, before the push-down: they were read after it, so {pushDownActive: true,
    # start: NaN} was refused with the active action already moved onto a new track and cleared - raised
    # straight out of take_float, past _undo_and_refuse (measured on 5.0.1, 2026-09-21). NlaStrips.new takes an
    # integer frame, so a fractional start is refused rather than floored.
    try:
        track_name = take(params, "track", default=None, kind=str)
        start_in = take_float(params, "start", default=None)
        want_start = finite_int(start_in, "start") if start_in is not None else None
        strip_name_in = take(params, "stripName", default=None, kind=str)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was added." % str(exc).rstrip("."))

    # ANIMATION DATA IS MADE ONLY NOW, WITH EVERY VALUE READ - and taken off again by a refusal below when this call made
    # it. It was created above the reads until 2026-09-28 (the 2026-09-21 pack-rule review), so {start: 1.5} on an object
    # with none was refused "NOTHING was added" and left the object carrying empty animation data (measured on 5.2.1:
    # list_animation_data hasAnimationData false before, true after).
    made_animation_data = obj.animation_data is None
    if made_animation_data:
        obj.animation_data_create()
    ad = obj.animation_data

    # EVERY TRACK THIS OP MAKES, so a refusal below can put the stack back exactly as it found it.
    # Both refusals here promise "NOTHING was added" and neither used to keep it: a failed push left
    # a named empty track behind, and a strip that would not fit left BOTH that track and a second
    # one - plus the active action already cleared, which is the damaging half. A caller retrying
    # after the refusal found its animation gone and two tracks it never asked for.
    made_tracks = []
    restore_action = None

    def _undo_and_refuse(message):
        for made in reversed(made_tracks):
            ad.nla_tracks.remove(made)
        del made_tracks[:]
        if restore_action is not None:
            ad.action = restore_action
        if made_animation_data:
            obj.animation_data_clear()
        raise MifOpError(message)

    pushed = None
    if ad.action is not None and push_down:
        pushed = ad.action.name
        restore_action = ad.action
        tr = ad.nla_tracks.new()
        made_tracks.append(tr)
        tr.name = "%s_pushed" % pushed
        try:
            tr.strips.new(pushed, int(ad.action.frame_range[0]), ad.action)
        except RuntimeError as exc:
            _undo_and_refuse("could not push the active action '%s' down onto a track: %s. "
                             "NOTHING was added." % (pushed, exc))
        ad.action = None

    track = None
    if track_name:
        track = ad.nla_tracks.get(str(track_name))
    if track is None:
        track = ad.nla_tracks.new()
        made_tracks.append(track)
        if track_name:
            track.name = str(track_name)

    start = want_start if want_start is not None else int(act.frame_range[0])
    strip_name = strip_name_in or act.name
    try:
        strip = track.strips.new(str(strip_name), start, act)
    except RuntimeError as exc:
        _undo_and_refuse("Blender refused to place '%s' at frame %d on track '%s': %s. A strip "
                         "cannot overlap another on the same track. NOTHING was added."
                         % (act.name, start, track.name, exc))
    # THE NAME IS WRITTEN AFTER THE STRIP EXISTS, because NlaStrips.new() does not reliably use the
    # one it is handed. Found by the second seat 2026-09-15 (B1416, their Defect F): on 4.2.0 and
    # 4.4.0 they measured three spellings - including one sharing no characters with the action's
    # name - and every strip came back named after the ACTION, with .001 on collision.
    #
    # MEASURED HERE AND IT IS VERSION-DEPENDENT, which is worth more than the fix. On 3.6.23 the
    # name is ignored exactly as they describe; on 4.2.17 LTS and 5.2.1 LTS the strip already comes
    # back with the requested name. So B1416 was red on 3.6 and green on the other two, and their
    # 4.2.0 reading and my 4.2.17 reading disagree inside one minor version.
    #
    # Assigning it unconditionally is correct on every build: NlaStrip.name is writable, this is a
    # no-op where Blender already honoured it, and it removes the version question from the caller's
    # problem entirely. Blender may suffix on collision, which is why the response reports
    # strip.name rather than what was asked for.
    strip.name = str(strip_name)
    if blend:
        strip.blend_type = blend
    # INFLUENCE IS IGNORED UNLESS use_animated_influence IS ON, and this wrote the value and then turned the flag OFF.
    # Reported by the second seat (mailbox id 79): influence 0.0 / 0.5 / 1.0 all played the full action, while the
    # response echoed the number. MEASURED 2026-09-16, identical on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1 - a cube
    # keyed x 0 -> 9 over 1-40, pushed into a strip, evaluated at frame 40:
    #   influence 0.0 or 0.5, flag off                 x 9.0 - full strength, the value unused
    #   flag on, THEN influence 0.5                    x 0.0 - turning the flag on KEYS influence at its current
    #                                                  value (an F-curve appears), and that key overrides the write
    #   influence 0.5, THEN flag on                    x 4.5 - the key captures 0.5, and the strip plays at half
    # So the value goes first and the flag second, and the response says an F-curve now holds it.
    if influence is not None:
        strip.influence = influence
        strip.use_animated_influence = True
        bpy.context.view_layer.update()

    # THE TRAP, REPORTED. animation_data.action is evaluated ON TOP of the entire NLA stack, so a
    # remaining active action makes every strip below it inert while the stack reads as correct.
    shadowed = ad.action is not None
    # AND THE OTHER ONE: THE SAME ACTION ON TWO TRACKS. pushDownActive with the active action being the action placed
    # puts it on "<action>_pushed" AND on the track asked for - both play it, so muting one changes nothing you can see
    # (second seat, mailbox id 93: three cubes moved identically with every mute reported applied).
    on_tracks = sorted({t.name for t in ad.nla_tracks for s in t.strips if s.action is not None and s.action == act})
    return {
        "object": obj.name,
        "track": track.name,
        "strip": {
            "name": strip.name,
            "action": strip.action.name if strip.action else None,
            "frameStart": round(float(strip.frame_start), 4),
            "frameEnd": round(float(strip.frame_end), 4),
            "blendType": strip.blend_type,
            "influence": round(float(strip.influence), 6),
            # WHAT PLAYS, beside what is stored: without animated influence Blender plays a strip at full strength
            # whatever `influence` holds - a new strip stores 0.0 and plays at 1.0.
            "animatedInfluence": bool(strip.use_animated_influence),
            "effectiveInfluence": (round(float(strip.influence), 6) if strip.use_animated_influence else 1.0),
            "influenceKeyed": any(fc.data_path == "influence" for fc in strip.fcurves),
        },
        # A PARTIAL INFLUENCE WITH NOTHING BENEATH BLENDS TOWARD 0, NOT TOWARD THE OBJECT'S OWN VALUE. Second seat, mailbox
        # id 129: a hop keyed 0.6 -> 2.6 at influence 0.25 rested at 0.15, sunk into the floor. MEASURED 2026-09-16 on
        # 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1 through this op: location z keyed 0.6 at frame 1 and 2.6 at 20, the object's
        # static z then set to 5.0 - a REPLACE strip at 0.25 played 0.15 and 0.65 (toward 0, not toward 5.0), ADD at 0.25
        # the same, REPLACE at 1.0 played 0.6 and 2.6. Said only when it applies: a partial effective influence on the
        # BOTTOM track, where nothing lies beneath the strip to blend against.
        "influenceNote": (
            "effective influence %g on the bottom track: with no strip beneath it, every channel this action animates blends "
            "toward 0, not toward the object's own value - a z keyed 0.6 -> 2.6 plays 0.15 -> 0.65 at 0.25 (measured on every "
            "build). Key the motion as an offset from 0, or put a full-influence strip beneath it." % round(float(strip.influence), 6))
            if strip.use_animated_influence and float(strip.influence) < 1.0 and len(ad.nla_tracks) and ad.nla_tracks[0] == track
            else None,
        "trackCount": len(ad.nla_tracks),
        "stripCount": sum(len(t.strips) for t in ad.nla_tracks),
        "pushedDownActive": pushed,
        "activeActionShadowsNla": shadowed,
        # No static explanation field. activeActionShadowsNla is the measurement and carries the
        # answer; the reason it matters belongs in the tool help, read BEFORE the call. Two such
        # notes were removed elsewhere today for the same reason - a constant string is something
        # no suite can check and no caller can act on differently.
        "activeAction": ad.action.name if ad.action else None,
        "actionOnTracks": on_tracks,
        "duplicatePlaybackNote": (
            "'%s' is now on %d tracks (%s)%s, and each plays it - muting one leaves the other playing the same motion."
            % (act.name, len(on_tracks), ", ".join(on_tracks),
               " - pushDownActive pushed the active action, which was this one" if pushed == act.name else "")
            if len(on_tracks) > 1 else None),
    }


def op_move_keyframes(params):
    """Retime keyframes - shift them, or scale the timing about a pivot.

    Retiming is a core animation operation and nothing here could do it. delete_keyframe and
    set_keyframe together can only rebuild an animation from scratch, which loses every handle,
    interpolation and easing on the way - so "make this 20% slower" or "push everything after
    frame 50 back ten frames" meant re-authoring it.

    params:
      object (str, required)
      dataPath (alias path)     limit to one channel; omitted moves EVERY curve on the object
      index (int)               limit to one array element
      offset (float)            frames to shift by. Negative moves earlier.
      scale (float)             multiply the timing about `pivot`. 2.0 is half speed.
      pivot (float)             the frame that stays put under scale, default the first key
      frameStart / frameEnd     only touch keys inside this range

    offset and scale are refused TOGETHER: applying both leaves the order ambiguous, and a caller
    who wants each can ask twice and see each result.

    THE HANDLES MOVE WITH THE KEYS. A bezier handle is stored in absolute frame coordinates, so
    moving only co leaves the handles behind and silently reshapes every curve it touches.
    """
    reject_unknown(params, {"object", "name", "dataPath", "path", "index", "offset", "scale",
                            "pivot", "frameStart", "frameEnd", "target", "datablock"}, "move_keyframes")
    obj = _anim_subject(params, "move_keyframes", "NOTHING was moved.")
    path = take(params, "dataPath", "path", default=None, kind=str)
    index = take_int(params, "index", default=None)  # take_int, NOT int(): a bad value must REFUSE, not raise ValueError out of the op.

    offset = take_float(params, "offset", default=None)
    scale = take_float(params, "scale", default=None)
    if offset is not None and scale is not None:
        raise MifOpError("pass offset OR scale, not both - applying both leaves the order "
                         "ambiguous. Call twice if you want each. NOTHING was moved.")
    if offset is None and scale is None:
        raise MifOpError("pass offset (frames to shift) or scale (a timing multiplier). NOTHING "
                         "was moved.")
    if scale is not None and scale <= 0:
        raise MifOpError("scale must be positive, got %g - a zero or negative multiplier collapses "
                         "or reverses the timing and is never what is meant. NOTHING was moved."
                         % scale)

    lo = take_float(params, "frameStart", default=None)
    hi = take_float(params, "frameEnd", default=None)
    if lo is not None and hi is not None and hi < lo:
        raise MifOpError("frameEnd (%g) is before frameStart (%g). NOTHING was moved." % (hi, lo))

    curves = _curves_for(obj, path, index) if path else [
        fc for _label, holder in _anim_holders(obj)
        for fc in _fcurves(holder)
        if index is None or fc.array_index == int(index)]
    if not curves:
        raise MifOpError("no fcurve on '%s'%s - nothing to retime. List them with list_keyframes. "
                         "NOTHING was moved."
                         % (obj.name, " for dataPath '%s'" % path if path else ""))

    selected = []
    for fc in curves:
        for kp in fc.keyframe_points:
            f = kp.co[0]
            if lo is not None and f < lo:
                continue
            if hi is not None and f > hi:
                continue
            selected.append((fc, kp))
    if not selected:
        raise MifOpError("no keyframe on '%s' falls inside %s. NOTHING was moved."
                         % (obj.name,
                            "frames %s..%s" % (lo if lo is not None else "-inf",
                                               hi if hi is not None else "+inf")))

    pivot = take_float(params, "pivot", default=None)
    if pivot is None:
        pivot = min(kp.co[0] for _fc, kp in selected)

    before = sorted({round(float(kp.co[0]), 4) for _fc, kp in selected})

    def new_frame(f):
        return f + offset if offset is not None else pivot + (f - pivot) * scale

    # MOVED IN THE DIRECTION OF TRAVEL. Blender keeps keyframe_points sorted by frame, and shifting
    # a key past its neighbour while iterating forwards makes the walk skip or revisit keys. Moving
    # later-first when going forwards, and earlier-first when going backwards, means a key never
    # crosses one that has not moved yet.
    forwards = (offset is not None and offset > 0) or (scale is not None and scale > 1.0)
    ordered = sorted(selected, key=lambda pair: pair[1].co[0], reverse=forwards)
    for fc, kp in ordered:
        f0 = kp.co[0]
        f1 = new_frame(f0)
        delta = f1 - f0
        kp.co[0] = f1
        # HANDLES ARE ABSOLUTE FRAME COORDINATES. Moving only co leaves them behind and reshapes
        # the curve silently - the interpolation still reads BEZIER and the motion is different.
        try:
            kp.handle_left[0] += delta
            kp.handle_right[0] += delta
        except (AttributeError, TypeError):
            pass
    for fc in curves:
        fc.update()

    after = sorted({round(float(kp.co[0]), 4) for _fc, kp in selected})
    return {
        "object": obj.name,
        "dataPath": path,
        "curves": len(curves),
        "keyframesMoved": len(selected),
        "offset": offset,
        "scale": scale,
        "pivot": round(float(pivot), 4),
        # THE TOTALS. Both lists stop at 40, and keyframesMoved is NOT their count: before and
        # after are sorted sets of DISTINCT frame numbers, so several keyframes sharing a frame
        # across curves collapse to one entry and len(selected) is always >= these. framesChanged
        # below is computed on the FULL lists and stays correct either way - it is these two
        # visible lists that could be read as complete when they are not.
        "framesBeforeTotal": len(before),
        "framesAfterTotal": len(after),
        "framesBefore": before[:40],
        "framesAfter": after[:40],
        # MEASURED. A retime that produced the same frame list did nothing, whatever the parameters
        # said - which is what an offset of 0 or a scale of 1 looks like, and what a range that
        # matched nothing would look like if it were not refused above.
        "framesChanged": before != after,
        "handlesMoved": True,
    }



_SETACTION_KEYS = {"action", "name", "fakeUser", "rename", "frameRange", "useFrameRange", "cyclic"}


def op_set_action(params):
    """Protect an action from being destroyed by the next save, or rename it.

    THE SHARPEST ASYMMETRY IN THIS FILE. _action_row computes survivesSave and its own comment says
    what it means: "An action with no users and no fake user is DELETED the next time the file is
    saved - silently, and by the save succeeding." list_actions reports it for every action, and
    assign_action reports previousSurvivesSave for the action it just displaced - which is precisely
    the moment one drops to zero users.

    So the addon told you your work was about to be destroyed, in two places, and there was nothing
    you could do about it: use_fake_user was writable only by create_action, at birth. An action
    imported with a mesh, or displaced by assign_action, could not be protected at all.

    A FAKE USER IS THE WHOLE MECHANISM. Blender purges datablocks with no users on save; a fake user
    is a deliberate reference that says "keep this even though nothing points at it". It is the same
    purge save_file reports as purgedOrphans, seen from the side that can prevent it.

    params:
      action / name (str)   which action. Required.
      fakeUser (bool)       keep it through a save even with no users
      rename (str)          a new name. Blender suffixes a clash rather than refusing, so the name
                            you get is reported and may not be the one you asked for.
      frameRange ([s, e])   the action's MANUAL frame range (turns useFrameRange on). The FBX all-actions export and
                            glTF take each clip's length from it (export_fbx_bin.py 3.6 :2434, 5.2 :2595), so a walk
                            cycle with a stray key at frame 300 exports 300 frames long unless its range is set
      useFrameRange (bool)  use the manual range (false: the range of the keys again)
      cyclic (bool)         the action loops (use_cyclic) over its manual range
    """
    reject_unknown(params, _SETACTION_KEYS, "set_action")
    want = take(params, "action", "name", required=True, kind=str)
    act = bpy.data.actions.get(str(want))
    if act is None:
        have = sorted(a.name for a in bpy.data.actions)[:25]
        raise MifOpError("no action named '%s'. This file has: %s. NOTHING was changed."
                         % (want, ", ".join(have) if have else "(none)"))

    if all(params.get(k) is None for k in ("fakeUser", "rename", "frameRange", "useFrameRange", "cyclic")):
        raise MifOpError("nothing to do - pass fakeUser, rename, frameRange, useFrameRange or cyclic. NOTHING was "
                         "changed.")
    fr = params.get("frameRange")
    if fr is not None:
        if (not isinstance(fr, (list, tuple)) or len(fr) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in fr)):
            raise MifOpError("frameRange is [start, end], got %r. NOTHING was changed." % (fr,))
        fr = finite_floats(fr, "frameRange")
        if not fr[1] > fr[0]:
            raise MifOpError("frameRange end %g must be after start %g. NOTHING was changed." % (fr[1], fr[0]))
    use_range = take_bool(params, "useFrameRange") if params.get("useFrameRange") is not None else None
    if use_range is False and fr is not None:
        raise MifOpError("frameRange sets the manual range and useFrameRange false turns it off - pass one. NOTHING was "
                         "changed.")
    cyclic = take_bool(params, "cyclic") if params.get("cyclic") is not None else None
    if cyclic and not (fr is not None or use_range or getattr(act, "use_frame_range", False)):
        raise MifOpError("cyclic loops the MANUAL frame range, and this action has none - pass frameRange too. NOTHING "
                         "was changed.")

    # EVERY VALUE IS READ AND CHECKED BEFORE THE FIRST WRITE (2026-09-21). fakeUser used to be written first, so a
    # non-string rename was refused by take() with the fake user already changed and nothing said about it, and a
    # 64-character one was refused with "The fake-user change, if any, stands" - a refusal the caller could not retry.
    before = _action_row(act)
    want_fake = take_bool(params, "fakeUser", default=True) if params.get("fakeUser") is not None else None
    new_name = take(params, "rename", default=None, kind=str)
    if new_name is not None and len(str(new_name)) > 63:
        raise MifOpError("rename is %d characters and Blender truncates at 63, so the action you get would not be "
                         "the one you named. NOTHING was changed." % len(str(new_name)))
    if want_fake is not None:
        act.use_fake_user = want_fake
    if fr is not None:
        act.use_frame_range = True
        # end, start, end: either setter may clamp against the other's old value
        act.frame_end = fr[1]
        act.frame_start = fr[0]
        act.frame_end = fr[1]
    elif use_range is not None:
        act.use_frame_range = use_range
    if cyclic is not None:
        act.use_cyclic = cyclic

    renamed_to = None
    if new_name is not None:
        asked = str(new_name)
        # An Action is an ID, so naming it over one that exists can rename the OTHER action
        # on 3.6 - see ops_common.assign_id_name and ledger B12. renamed_to below reads the
        # name back either way, so the caller's report is unchanged; what changes is that no
        # action the caller never mentioned is touched.
        assign_id_name(act, asked, bpy.data.actions)
        renamed_to = act.name

    after = _action_row(act)
    return {
        "ok": True,
        "action": act.name,
        "before": before,
        "after": after,
        "renamedTo": renamed_to,
        # BLENDER SUFFIXES A CLASHING NAME rather than refusing, so anything looking this up by the
        # name it asked for would find the wrong action - or nothing.
        "nameWasSuffixed": bool(renamed_to and renamed_to != str(new_name)),
        "changedFields": sorted(k for k in set(before) | set(after)
                                if before.get(k) != after.get(k)),
        "note": ("this action has no users and no fake user, so the next save DELETES it - silently, "
                 "and by the save succeeding. Pass fakeUser:true to keep it."
                 if not after["survivesSave"] else
                 ("it had no users and would have been purged by the next save; the fake user now "
                  "keeps it." if not before["survivesSave"] else None)),
    }


_NLATRACK_KEYS = {"object", "name", "track", "mute", "solo", "lock", "rename", "remove"}


def op_set_nla_track(params):
    """Mute, solo, lock or rename an NLA track - none of which was possible.

    list_keyframes has always reported a track's mute and nothing could write it, so a track could
    be built and never silenced. That is the small half.

    THE LARGE HALF IS SOLO, AND THE READ SIDE WAS MISLEADING ABOUT IT. Setting is_solo on one track
    silences every OTHER track's contribution while each of them still reports mute:False and
    is_solo:False. Measured on 3.6.23, 4.2.17, 4.4.0 and 5.0.1: two tracks moving a cube to z=-5 and
    z=+5, evaluated at frame 10, give -5 with both live and +5 with the first soloed - the second
    track's contribution is gone and nothing about the second track says so.

    So a caller reading every track, seeing every one unmuted, and asking why only one plays had no
    field that could answer. list_keyframes now reports isSolo per track and names the soloing track
    when one exists.

    SOLO IS EXCLUSIVE, and Blender enforces that itself: setting is_solo on a second track clears it
    on the first. The response reports which track ended up soloed rather than assuming it is this
    one.

    params:
      object / name (str)   required
      track (str)           which NLA track. Required.
      mute (bool)           silence this track
      solo (bool)           silence every OTHER track
      lock (bool)           protect it from editing
      rename (str)          a new name
      remove (bool)         delete the named track and its strips (2026-10-06: a rig's leftover tracks were
                            removed before an export in raw Python); track "*" deletes every track on the
                            object. The active action is kept. Takes no other setting.
    """
    reject_unknown(params, _NLATRACK_KEYS, "set_nla_track")
    obj = get_object(take(params, "object", "name", required=True, kind=str))
    ad = getattr(obj, "animation_data", None)
    tracks = list(getattr(ad, "nla_tracks", None) or []) if ad is not None else []
    remove = take_bool(params, "remove", default=False)
    all_tracks = remove and params.get("track") == "*"
    if remove:
        others = [k for k in ("mute", "solo", "lock", "rename") if params.get(k) is not None]
        if others:
            raise MifOpError("remove deletes the track, so %s has nothing to act on - send them in another call. "
                             "NOTHING was changed." % ", ".join(others))
        if params.get("track") is None:
            raise MifOpError("remove needs track: a track's name, or \"*\" for every track. NOTHING was changed.")
        if not tracks:
            raise MifOpError("'%s' has no NLA tracks to remove - a removal of nothing is not a clean result. NOTHING "
                             "was changed." % obj.name)
        if all_tracks:
            doomed = tracks
        else:
            name = str(params["track"])
            doomed = [tr for tr in tracks if tr.name == name]
            if not doomed:
                raise MifOpError("'%s' has no NLA track named '%s'. It has: %s. NOTHING was changed."
                                 % (obj.name, name, ", ".join(tr.name for tr in tracks)))
        rows = [{"track": tr.name, "strips": [st.name for st in tr.strips]} for tr in doomed]
        for tr in doomed:
            ad.nla_tracks.remove(tr)
        left = [tr.name for tr in ad.nla_tracks]
        return {"ok": True, "object": obj.name, "removed": rows, "tracksLeft": left, "trackCount": len(left),
                "activeAction": ad.action.name if ad.action is not None else None,
                "note": None if ad.action is not None or left else
                ("'%s' now has no NLA track and no active action, so nothing animates it." % obj.name)}
    if not tracks:
        raise MifOpError("'%s' has no NLA tracks. add_nla_strip makes one - it takes a `track` name "
                         "and creates it if absent. NOTHING was changed." % obj.name)

    want = take(params, "track", required=True, kind=str)
    track = None
    for tr in tracks:
        if tr.name == str(want):
            track = tr
            break
    if track is None:
        raise MifOpError("'%s' has no NLA track named '%s'. It has: %s. NOTHING was changed."
                         % (obj.name, want, ", ".join(tr.name for tr in tracks)))

    asked = [k for k in ("mute", "solo", "lock", "rename") if params.get(k) is not None]
    if not asked:
        raise MifOpError("nothing to do - pass mute, solo, lock or rename. NOTHING was changed.")

    before = {"mute": bool(track.mute), "isSolo": bool(track.is_solo), "lock": bool(track.lock),
              "name": track.name}
    # READ ALL FOUR BEFORE WRITING ANY (2026-09-21): mute was written, then lock read - so {mute: true, lock: "flase"}
    # muted the track and refused.
    want = {k: take_bool(params, k, default=True) for k in ("mute", "lock", "solo") if params.get(k) is not None}
    new_name = take(params, "rename", default=None, kind=str)
    if "mute" in want:
        track.mute = want["mute"]
    if "lock" in want:
        track.lock = want["lock"]
    if "solo" in want:
        track.is_solo = want["solo"]
    renamed_to = None
    if new_name is not None:
        track.name = str(new_name)
        renamed_to = track.name

    # READ BACK ACROSS EVERY TRACK, not just this one. Solo is exclusive and Blender clears it on
    # the others itself, so "which track is soloed" is a property of the stack rather than of the
    # track that was written.
    soloed = [tr.name for tr in tracks if tr.is_solo]
    after = {"mute": bool(track.mute), "isSolo": bool(track.is_solo), "lock": bool(track.lock),
             "name": track.name}
    silenced = [tr.name for tr in tracks
                if not tr.is_solo and not tr.mute and soloed and tr.name not in soloed]
    return {
        "ok": True,
        "object": obj.name,
        "track": track.name,
        "before": before,
        "after": after,
        "renamedTo": renamed_to,
        "changedFields": sorted(k for k in before if before[k] != after[k]),
        "soloedTrack": soloed[0] if soloed else None,
        # THE FIELD THE READ SIDE COULD NOT ANSWER. These tracks are NOT muted and contribute
        # NOTHING, because another track is soloed - and every one of them reports mute:False.
        "silencedBySolo": silenced or None,
        "trackCount": len(tracks),
        "note": ("'%s' is soloed, so %s contribute NOTHING while still reporting mute:false. That "
                 "is what solo does and it is invisible on the tracks it silences."
                 % (soloed[0], ", ".join(silenced))) if silenced else
                ("this track is muted - its strips still exist and do nothing."
                 if after["mute"] else None),
    }

# =============================================================================================================
# COPYING KEYS AND CLEANING CURVES (2026-10-04, spec "Blender: poses and keys")
# =============================================================================================================
# Blender's graph.copy / graph.paste(flipped) and graph.clean / graph.euler_filter all poll for a Graph Editor area, so
# both ops are native: they read and write keyframe_points directly, on the curves _fcurves finds (5.x slotted actions
# included). A key is copied WITH its interpolation, easing, handle types and handles - the handles move and scale with
# it, as move_keyframes moves them in time.

_KEY_ATTRS = ("handle_left_type", "handle_right_type", "interpolation", "easing", "amplitude", "back", "period", "type")
_PATH_INDEX = re.compile(r"^(.*)\[(\d+)\]$")


def _key_snap(kp):
    snap = {"co": (kp.co[0], kp.co[1]), "hl": (kp.handle_left[0], kp.handle_left[1]),
            "hr": (kp.handle_right[0], kp.handle_right[1])}
    for a in _KEY_ATTRS:
        if hasattr(kp, a):
            snap[a] = getattr(kp, a)
    return snap


def _key_put(fc, snap, shift=0.0, value=lambda v: v):
    """Insert a snapped key moved by `shift` frames with its value (and handles' values) through `value`."""
    kp = fc.keyframe_points.insert(snap["co"][0] + shift, value(snap["co"][1]), options={"FAST"})
    for a in _KEY_ATTRS:
        if a in snap:
            try:
                setattr(kp, a, snap[a])
            except (TypeError, ValueError, AttributeError):
                pass
    kp.handle_left = (snap["hl"][0] + shift, value(snap["hl"][1]))
    kp.handle_right = (snap["hr"][0] + shift, value(snap["hr"][1]))
    return kp


def _split_index(entry):
    """'location[1]' -> ('location', 1); 'location' -> ('location', None). A custom property's ["x"] is not an index."""
    m = _PATH_INDEX.match(entry)
    return (m.group(1), int(m.group(2))) if m else (entry, None)


def _holder_for(obj, label):
    return {"object": obj, "data": getattr(obj, "data", None), "shape_keys": _shape_key_block(obj)}.get(label)


def _curves_labelled(obj):
    return [(label, holder, fc) for label, holder in _anim_holders(obj) for fc in _fcurves(holder)]


def _pick_source_curves(obj, params, verb):
    """dataPaths [..] | bones [..] | all true -> [(label, holder, fcurve)], with `index` narrowing."""
    paths = params.get("dataPaths")
    bones = params.get("bones")
    every = take_bool(params, "all", default=False)
    if sum(1 for x in (paths is not None, bones is not None, every) if x) != 1:
        raise MifOpError("pass one of dataPaths, bones or all:true. NOTHING was %s." % verb)
    index = take_int(params, "index", default=None)
    found = _curves_labelled(obj)
    if paths is not None:
        if not isinstance(paths, list) or not paths or any(not isinstance(p, str) or not p for p in paths):
            raise MifOpError("dataPaths is a non-empty list of data paths ('location', 'location[1]', "
                             "'pose.bones[\"arm.L\"].rotation_quaternion'). NOTHING was %s." % verb)
        out = []
        for entry in paths:
            p, i = _split_index(entry)
            hit = [t for t in found if t[2].data_path == p and (i is None or t[2].array_index == i)]
            if not hit:
                raise MifOpError("no curve on '%s' for '%s' - list them with list_keyframes. NOTHING was %s."
                                 % (obj.name, entry, verb))
            out.extend(t for t in hit if t not in out)
    elif bones is not None:
        if not isinstance(bones, list) or not bones or any(not isinstance(b, str) or not b for b in bones):
            raise MifOpError("bones is a non-empty list of bone names. NOTHING was %s." % verb)
        have = {_bone_of_path(t[2].data_path) for t in found}
        missing = [b for b in bones if b not in have]
        if missing:
            raise MifOpError("no curve animates bone(s) %s on '%s'. NOTHING was %s."
                             % (", ".join(missing[:20]), obj.name, verb))
        out = [t for t in found if _bone_of_path(t[2].data_path) in set(bones)]
    else:
        out = list(found)
    if index is not None:
        out = [t for t in out if t[2].array_index == index]
    if not out:
        raise MifOpError("'%s' has no curve matching that selection. NOTHING was %s." % (obj.name, verb))
    return out


def _ensure_curve(holder, data_path, index, group):
    """The curve for (data_path, index) on holder's action, made when missing (an action too). (fcurve, created)."""
    for fc in _fcurves(holder):
        if fc.data_path == data_path and fc.array_index == index:
            return fc, False
    ad = holder.animation_data_create()
    act = ad.action
    fresh = act is None
    if fresh:
        act = bpy.data.actions.new(name="%sAction" % holder.name)
    # Up to 4.4 the action owns its curves (fcurves.new). 5.x has no action.fcurves: Blender's own ensure puts the curve
    # in the slot the holder uses (MEASURED: on 4.4.0 that ensure takes no group_name - "takes at most 3 arguments").
    # 4.4 with a slot bound: Action.fcurves.new writes into the FIRST slot, so Blender's ensure (no group_name there)
    if not fresh and hasattr(act, "fcurves") and getattr(ad, "action_slot", None) is not None and len(act.slots) > 1:
        return act.fcurve_ensure_for_datablock(holder, data_path, index=index), True
    if hasattr(act, "fcurves"):
        fc = act.fcurves.new(data_path, index=index, action_group=group or "")
        if fresh:
            # ASSIGNED AFTER THE CURVE, so 4.4 gives the holder the slot fcurves.new made: assigned first, the action had
            # no slot to give, and the curve existed and animated nothing (measured on 4.4.0: the target read 0 at every
            # frame while the curve evaluated right).
            ad.action = act
            if getattr(ad, "action_slot", 0) is None and len(getattr(act, "slots", ())):
                ad.action_slot = act.slots[0]
        return fc, True
    if fresh:
        ad.action = act
    return act.fcurve_ensure_for_datablock(holder, data_path, index=index, group_name=group or ""), True


def _remove_curve(holder, fc):
    ad = getattr(holder, "animation_data", None)
    act = ad.action if ad else None
    if act is None:
        return False
    if hasattr(act, "fcurves"):
        act.fcurves.remove(fc)
        return True
    for layer in getattr(act, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for cb in getattr(strip, "channelbags", ()):
                if any(c == fc for c in cb.fcurves):
                    cb.fcurves.remove(fc)
                    return True
    return False


def _leaf_prop(holder, data_path):
    """(owner struct, RNA property) for a data path on holder, or (None, None) for a custom property / no such path."""
    if data_path.endswith('"]'):
        return None, None
    head, _, leaf = data_path.rpartition(".")
    try:
        owner = holder.path_resolve(head) if head else holder
    except (ValueError, AttributeError, TypeError):
        return None, None
    return owner, owner.bl_rna.properties.get(leaf) if hasattr(owner, "bl_rna") else None


# Blender's paste-flipped rule per channel: which array elements change sign (see ops_pose's module docstring).
_FLIP_NEGATE = {"location": (0,), "rotation_euler": (1, 2), "rotation_quaternion": (2, 3),
                "rotation_axis_angle": (2, 3)}


_COPYKEYS_KEYS = {"object", "name", "dataPaths", "bones", "all", "index", "fromRange", "to", "toDataPath", "toObject",
                  "repeat", "period", "merge", "valueScale", "valueOffset", "pivot", "flip", "nameRule"}


def op_copy_keyframes(params):
    """Copy keys between channels and frames - offset, repeat, value scale about a pivot, flipped paste. See the MCP
    wrapper and tool help for the parameters."""
    reject_unknown(params, _COPYKEYS_KEYS, "copy_keyframes")
    obj = get_object(take(params, "object", "name", required=True, kind=str))
    sources = _pick_source_curves(obj, params, "copied")
    dst_obj = get_object(take(params, "toObject", default=obj.name, kind=str))
    require_editable(dst_obj, "key")
    rng = params.get("fromRange") or {}
    if not isinstance(rng, dict) or set(rng) - {"frameStart", "frameEnd"}:
        raise MifOpError("fromRange is {frameStart, frameEnd}, got %r. NOTHING was copied." % (params.get("fromRange"),))
    lo = finite_float(rng["frameStart"], "fromRange.frameStart") if rng.get("frameStart") is not None else None
    hi = finite_float(rng["frameEnd"], "fromRange.frameEnd") if rng.get("frameEnd") is not None else None
    if lo is not None and hi is not None and hi < lo:
        raise MifOpError("fromRange.frameEnd (%g) is before fromRange.frameStart (%g). NOTHING was copied." % (hi, lo))
    to = params.get("to")
    if not isinstance(to, dict) or len(to) != 1 or not set(to) <= {"frame", "offset"}:
        raise MifOpError("to is {frame: n} (where the first copied key lands) or {offset: n} (frames to shift), got "
                         "%r. NOTHING was copied." % (to,))
    repeat = take_int(params, "repeat", default=1)
    if not 1 <= repeat <= 100:
        raise MifOpError("repeat must be 1-100, got %d. NOTHING was copied." % repeat)
    merge_ids = [i.identifier for i in
                 bpy.ops.graph.paste.get_rna_type().properties["merge"].enum_items]
    merge = str(take(params, "merge", default="MIX", kind=str)).upper()
    if merge not in merge_ids:
        raise MifOpError("merge must be one of %s (Blender's paste merge), got %r. NOTHING was copied."
                         % (", ".join(merge_ids), params.get("merge")))
    scale = take_float(params, "valueScale", default=1.0)
    offset = take_float(params, "valueOffset", default=0.0)
    pivot = params.get("pivot", "FIRST_KEY")
    if isinstance(pivot, dict) and set(pivot) == {"value"}:
        pivot = finite_float(pivot["value"], "pivot.value")
    elif isinstance(pivot, (int, float)) and not isinstance(pivot, bool):
        pivot = finite_float(pivot, "pivot")
    elif not (isinstance(pivot, str) and pivot.upper() == "FIRST_KEY"):
        raise MifOpError("pivot is FIRST_KEY (each curve's first copied key) or {value: v}, got %r. NOTHING was "
                         "copied." % (pivot,))
    flip = take_bool(params, "flip", default=False)
    to_path = take(params, "toDataPath", default=None, kind=str)
    rule = None
    if flip:
        if params.get("nameRule") is None:
            raise MifOpError("flip needs nameRule ('FLIP_NAME', a suffix pair or a table) - the side of a name is never "
                             "assumed. NOTHING was copied.")
        from .ops_pose import parse_name_rule, partner_name   # noqa: PLC0415
        rule = parse_name_rule(params.get("nameRule"), verb="copied")
        if to_path:
            raise MifOpError("flip writes each bone's curves onto its partner's; toDataPath names one destination - "
                             "pass one or the other. NOTHING was copied.")
    elif params.get("nameRule") is not None:
        raise MifOpError("nameRule pairs the sides for flip:true. NOTHING was copied.")

    # THE KEYS, SNAPPED BEFORE ANYTHING IS WRITTEN - a copy within one curve must read the curve as it was.
    plan = []
    for label, holder, fc in sources:
        keys = [_key_snap(kp) for kp in fc.keyframe_points
                if (lo is None or kp.co[0] >= lo - 1e-6) and (hi is None or kp.co[0] <= hi + 1e-6)]
        if keys:
            plan.append([label, holder, fc, keys])
    if not plan:
        raise MifOpError("no key of the chosen curves lies in %s..%s. NOTHING was copied."
                         % ("-inf" if lo is None else "%g" % lo, "+inf" if hi is None else "%g" % hi))
    first = min(k["co"][0] for p in plan for k in p[3])
    last = max(k["co"][0] for p in plan for k in p[3])
    period = take_float(params, "period", default=None)
    if period is None:
        period = last - first
    if repeat > 1 and not period > 0:
        raise MifOpError("repeat needs a period above 0 - the copied keys span %g frames; pass period. NOTHING was "
                         "copied." % (last - first))
    shift0 = (finite_float(to["frame"], "to.frame") - first) if "frame" in to else finite_float(to["offset"], "to.offset")

    # DESTINATIONS, resolved and type-checked before the first write.
    if to_path:
        dpath, didx = _split_index(to_path)
        if len({(p[2].data_path) for p in plan}) != 1 or (didx is not None and len(plan) != 1):
            raise MifOpError("toDataPath names ONE destination, and the selection holds %d curve(s) on %d path(s) - "
                             "narrow it with dataPaths or index. NOTHING was copied."
                             % (len(plan), len({p[2].data_path for p in plan})))
    not_flipped, rows = [], []
    for p in plan:
        label, holder, fc, _keys = p
        dpath_i, didx_i = fc.data_path, fc.array_index
        negate = ()
        if to_path:
            dpath_i = dpath
            didx_i = didx if didx is not None else fc.array_index
        elif flip:
            bone = _bone_of_path(fc.data_path)
            if bone is None:
                not_flipped.append(_curve_label(fc))
            else:
                other = partner_name(bone, rule)
                if other is not None:
                    dpath_i = fc.data_path.replace('pose.bones["%s"]' % bone, 'pose.bones["%s"]' % other, 1)
                negate = _FLIP_NEGATE.get(fc.data_path.rpartition(".")[2], ())
        dholder = _holder_for(dst_obj, label)
        if dholder is None:
            raise MifOpError("'%s' has no %s to hold '%s'. NOTHING was copied." % (dst_obj.name, label, dpath_i))
        s_owner, s_prop = _leaf_prop(holder, fc.data_path)
        d_owner, d_prop = _leaf_prop(dholder, dpath_i)
        if not dpath_i.endswith('"]'):
            if d_prop is None:
                raise MifOpError("'%s' has no property '%s' to copy keys onto. NOTHING was copied."
                                 % (dst_obj.name, dpath_i))
            if s_prop is not None and (s_prop.identifier != d_prop.identifier and (
                    s_prop.type != d_prop.type or s_prop.identifier.startswith("rotation_")
                    or d_prop.identifier.startswith("rotation_"))):
                raise MifOpError("'%s' (%s) and '%s' (%s) hold different kinds of value - keys of one mean nothing on "
                                 "the other. NOTHING was copied." % (fc.data_path, s_prop.identifier, dpath_i,
                                                                      d_prop.identifier))
            length = getattr(d_prop, "array_length", 0) or 1
            if didx_i >= length:
                raise MifOpError("'%s' has %d element(s); index %d does not exist. NOTHING was copied."
                                 % (dpath_i, length, didx_i))
        rows.append((p, dholder, dpath_i, didx_i, negate))
    dsts = [(id(r[1]), r[2], r[3]) for r in rows]
    if len(set(dsts)) != len(dsts):
        raise MifOpError("two source curves would be written onto the same destination curve. NOTHING was copied.")

    # THE TRANSFORM of a value: flip's sign, then scale about the pivot, then the offset - handles go through it too.
    def value_fn(first_value, neg):
        sign = -1.0 if neg else 1.0
        piv = sign * first_value if isinstance(pivot, str) else pivot
        return lambda v: piv + (sign * v - piv) * scale + offset

    # SOURCE SAMPLES for the check, taken before the write.
    checks = []
    for (p, _dh, _dp, _di, negate) in rows:
        fc, keys = p[2], p[3]
        fn = value_fn(keys[0]["co"][1], p[2].array_index in negate)
        frames = sorted({k["co"][0] for k in keys})
        mids = [(a + b) / 2.0 for a, b in zip(frames, frames[1:])]
        samples = [(f, fn(fc.evaluate(f))) for f in frames + mids]
        checks.append(samples)

    sc_frames = []
    created, written = [], []
    for (p, dholder, dpath_i, didx_i, negate), samples in zip(rows, checks):
        label, _holder, fc, keys = p
        group = _bone_of_path(dpath_i) or (fc.group.name if fc.group else "")
        dfc, made = _ensure_curve(dholder, dpath_i, didx_i, group)
        if made:
            created.append(_curve_label(dfc))
        fn = value_fn(keys[0]["co"][1], fc.array_index in negate)
        spans = [(shift0 + r * period + first, shift0 + r * period + last) for r in range(repeat)]
        if merge == "OVER_ALL":
            for kp in list(dfc.keyframe_points)[::-1]:
                dfc.keyframe_points.remove(kp, fast=True)
        elif merge in ("OVER_RANGE", "OVER_RANGE_ALL"):
            if merge == "OVER_RANGE":
                a = min(k["co"][0] for k in keys) + shift0
                b = max(k["co"][0] for k in keys) + shift0 + (repeat - 1) * period
            else:
                a, b = spans[0][0], spans[-1][1]
            for kp in list(dfc.keyframe_points)[::-1]:
                if a - 1e-6 <= kp.co[0] <= b + 1e-6:
                    dfc.keyframe_points.remove(kp, fast=True)
        n = 0
        for r in range(repeat):
            for k in keys:
                _key_put(dfc, k, shift0 + r * period, fn)
                n += 1
        dfc.update()
        sc_frames.extend(s for sp in spans for s in sp)
        written.append((dfc, samples, n, fc))

    diff, sampled = 0.0, 0
    out_rows = []
    for dfc, samples, n, fc in written:
        worst = 0.0
        for r in range(repeat):
            for f, want in samples:
                got = dfc.evaluate(f + shift0 + r * period)
                worst = max(worst, abs(got - want))
                sampled += 1
        diff = max(diff, worst)
        out_rows.append({"from": _curve_label(fc), "to": _curve_label(dfc), "keys": n, "maxDifference": round(worst, 6)})
    out = {"object": obj.name, "toObject": dst_obj.name, "curvesWritten": len(out_rows),
           "keysWritten": sum(r["keys"] for r in out_rows), "curves": out_rows[:100],
           "framesCovered": [round(min(sc_frames), 4), round(max(sc_frames), 4)], "repeat": repeat,
           "period": round(period, 4), "merge": merge, "valueScale": scale, "valueOffset": offset,
           "pivot": pivot if not isinstance(pivot, str) else "FIRST_KEY", "flip": flip,
           "createdCurves": created[:100] or None,
           "check": {"sampled": sampled, "maxDifference": round(diff, 6)}}
    if not_flipped:
        out["notFlipped"] = not_flipped[:50]
        out["notFlippedNote"] = "these curves are not a bone's, so they were copied without a side swap or sign change."
    if diff > 1e-4:
        out["checkNote"] = ("a destination curve does not evaluate to the copied keys everywhere (max %.6g): with merge "
                            "MIX the destination's own keys inside the pasted range shape it too - see curves[]."
                            % diff)
    return out


_CLEAN_KEYS = {"object", "name", "dataPaths", "bones", "all", "index", "threshold", "channels", "eulerFilter",
               "frameStart", "frameEnd"}


def _euler_unwrap(fc, lo, hi):
    """Blender's discontinuity filter on one euler curve: each key shifted by whole turns so it is within half a turn of
    the key before it. Returns (keys shifted, largest step before, after) in degrees."""
    pts = [kp for kp in fc.keyframe_points
           if (lo is None or kp.co[0] >= lo - 1e-6) and (hi is None or kp.co[0] <= hi + 1e-6)]
    vals = [kp.co[1] for kp in pts]
    step_before = max((abs(b - a) for a, b in zip(vals, vals[1:])), default=0.0)
    shifted, prev = 0, None
    for kp in pts:
        if prev is not None:
            turns = round((kp.co[1] - prev) / (2.0 * math.pi))
            if turns:
                d = -turns * 2.0 * math.pi
                kp.co[1] += d
                kp.handle_left[1] += d
                kp.handle_right[1] += d
                shifted += 1
        prev = kp.co[1]
    fc.update()
    vals = [kp.co[1] for kp in pts]
    step_after = max((abs(b - a) for a, b in zip(vals, vals[1:])), default=0.0)
    return shifted, math.degrees(step_before), math.degrees(step_after)


def _clean_curve(fc, threshold, lo, hi):
    """Drop every key whose removal moves the evaluated curve by less than threshold at every integer frame of the
    range (against the curve as it was, so errors do not add up). Returns (keys before, after, largest deviation)."""
    keys = sorted(fc.keyframe_points, key=lambda k: k.co[0])
    before = len(keys)
    if before < 2:
        return before, before, 0.0
    a = keys[0].co[0] if lo is None else max(lo, keys[0].co[0])
    b = keys[-1].co[0] if hi is None else min(hi, keys[-1].co[0])
    frames = list(range(int(math.ceil(a - 1e-6)), int(math.floor(b + 1e-6)) + 1))
    orig = {f: fc.evaluate(f) for f in frames}
    worst = 0.0
    i = 0
    while True:
        pts = sorted(fc.keyframe_points, key=lambda k: k.co[0])
        cand = [k for k in pts if (lo is None or k.co[0] >= lo - 1e-6) and (hi is None or k.co[0] <= hi + 1e-6)]
        if i >= len(cand) or len(pts) <= 1:
            break
        kp = cand[i]
        pos = [round(k.co[0], 6) for k in pts].index(round(kp.co[0], 6))
        if pos == 0:
            # THE CURVE'S FIRST KEY STAYS, as in Blender's graph.clean (settled 2026-10-18 on 3.6-5.2 under a Graph Editor
            # override, test_blender_settle_routes BS703): a hold of equal keys at the start lost its first key here -
            # no evaluated change, but the action's range began at the next key - while graph.clean kept it.
            i += 1
            continue
        wa = pts[max(0, pos - 2)].co[0]
        wb = pts[min(len(pts) - 1, pos + 2)].co[0]
        snap = _key_snap(kp)
        fc.keyframe_points.remove(kp)
        fc.update()
        dev = max((abs(fc.evaluate(f) - orig[f]) for f in frames if wa - 1e-6 <= f <= wb + 1e-6), default=0.0)
        if dev > threshold:
            _key_put(fc, snap)
            fc.update()
            i += 1
        else:
            worst = max(worst, dev)
    after = len(fc.keyframe_points)
    worst = max((abs(fc.evaluate(f) - orig[f]) for f in frames), default=0.0)
    return before, after, worst


def op_clean_keyframes(params):
    """Remove redundant keys, and/or make euler curves continuous - see the MCP wrapper and tool help."""
    reject_unknown(params, _CLEAN_KEYS, "clean_keyframes")
    obj = get_object(take(params, "object", "name", required=True, kind=str))
    require_editable(obj, "clean")
    sources = _pick_source_curves(obj, params, "changed")
    default = bpy.ops.graph.clean.get_rna_type().properties["threshold"].default
    threshold = take_float(params, "threshold", default=default)
    if not threshold >= 0:
        raise MifOpError("threshold must be 0 or more, got %r. NOTHING was changed." % threshold)
    channels = take_bool(params, "channels", default=False)
    ef = params.get("eulerFilter", False)
    if isinstance(ef, str) and ef.upper() == "ONLY":
        ef = "ONLY"
    elif not isinstance(ef, bool):
        raise MifOpError("eulerFilter is true, false or 'ONLY' (the filter without the clean), got %r. NOTHING was "
                         "changed." % (ef,))
    if ef == "ONLY" and (channels or params.get("threshold") is not None):
        raise MifOpError("eulerFilter 'ONLY' runs the filter alone; threshold and channels belong to the clean. NOTHING "
                         "was changed.")
    lo = take_float(params, "frameStart", default=None)
    hi = take_float(params, "frameEnd", default=None)
    if lo is not None and hi is not None and hi < lo:
        raise MifOpError("frameEnd (%g) is before frameStart (%g). NOTHING was changed." % (hi, lo))
    rows, removed_curves = [], []
    for label, holder, fc in sources:
        row = {"curve": _curve_label(fc), "target": label}
        if ef and fc.data_path.rpartition(".")[2] in ("rotation_euler", "delta_rotation_euler"):
            shifted, sb, sa = _euler_unwrap(fc, lo, hi)
            row.update({"keysShifted": shifted, "maxStepBeforeDegrees": round(sb, 4),
                        "maxStepAfterDegrees": round(sa, 4)})
        if ef != "ONLY":
            kb, ka, dev = _clean_curve(fc, threshold, lo, hi)
            row.update({"keysBefore": kb, "keysAfter": ka, "maxDeviation": round(dev, 6)})
            if channels and ka == 1:
                _owner, prop = _leaf_prop(holder, fc.data_path)
                if prop is not None:
                    dflt = (prop.default_array[fc.array_index] if getattr(prop, "is_array", False)
                            else getattr(prop, "default", None))
                    if isinstance(dflt, (int, float)) and abs(fc.keyframe_points[0].co[1] - dflt) <= threshold:
                        if _remove_curve(holder, fc):
                            removed_curves.append(row["curve"])
                            row["curveRemoved"] = True
        rows.append(row)
    out = {"object": obj.name, "curves": rows[:200], "curveCount": len(rows), "threshold": threshold,
           "eulerFilter": ef, "channels": channels}
    if ef != "ONLY":
        out["keysRemoved"] = sum(r.get("keysBefore", 0) - r.get("keysAfter", 0) for r in rows)
        out["maxDeviation"] = max((r.get("maxDeviation", 0.0) for r in rows), default=0.0)
    if ef:
        out["keysShifted"] = sum(r.get("keysShifted", 0) for r in rows)
        out["eulerNote"] = ("the filter shifts keys by whole turns so each is within 180 degrees of the one before - "
                            "it makes a curve continuous, it does not wrap it into 0-360.")
    if removed_curves:
        out["removedCurves"] = removed_curves[:100]
    return out


def _slot_curve_bags(act, slot):
    """[(owner, fcurves)] holding `slot`'s curves on `act`: the channelbags of a layered action (4.4+), or the
    action itself on a build without slots. owner.fcurves.remove() removes one."""
    bags = []
    for layer in getattr(act, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            if hasattr(strip, "channelbag") and slot is not None:
                try:
                    cb = strip.channelbag(slot)
                except (RuntimeError, TypeError):
                    cb = None
                if cb is not None:
                    bags.append((cb, list(cb.fcurves)))
    if not bags and hasattr(act, "fcurves"):
        bags.append((act, list(act.fcurves)))
    return bags


def op_copy_animation(params):
    """Copy one object's action onto other objects - an independent copy each, or the same action shared.

    THE GAP. assign_action shares ONE action across objects, so editing one object's keys edits them all; nothing
    made a copy that could then be changed alone (a fractured wall's cells each keyed from one template, a clip
    reused on a second rig and then retimed).

    params:
      source (str)      the object whose current action is copied
      targets [names]   the objects that receive it (the source itself is refused)
      copy (bool)       true (default): each target gets its own copy, named '<action>_<target>'; false: the source's
                        action itself, shared, bound to the source's slot on 4.4+ (assign_action's behaviour)
      keys [dataPath]   copy only the curves whose data path starts with one of these (copy true only)
      frameOffset       shift every copied key by this many frames (copy true only)

    Each target's previous action is reported with whether it survives a save.
    """
    reject_unknown(params, {"source", "targets", "copy", "keys", "frameOffset"}, "copy_animation", {
        "target": "spelt targets, a list - one target is a one-element list.",
        "link": "copy false shares the source's action (Blender's Link Animation Data).",
    })
    src = get_object(take(params, "source", required=True, kind=str))
    names = params.get("targets")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise MifOpError("targets must be a non-empty list of object names, got %r. NOTHING was changed." % (names,))
    names = list(dict.fromkeys(names))
    if src.name in names:
        raise MifOpError("'%s' is the source - it already plays this action. NOTHING was changed." % src.name)
    missing = [n for n in names if bpy.data.objects.get(n) is None]
    if missing:
        raise MifOpError("no object named %s. NOTHING was changed." % ", ".join("'%s'" % n for n in missing[:12]))
    targets = [bpy.data.objects[n] for n in names]
    for t in targets:
        require_editable(t, "animate")
    copy = take_bool(params, "copy", default=True)
    keys = params.get("keys")
    offset = take_float(params, "frameOffset", default=0.0)
    if not copy and (keys is not None or offset):
        raise MifOpError("keys and frameOffset change the copy, and copy false shares the source's action itself - "
                         "pass copy true, or drop them. NOTHING was changed.")
    if keys is not None and (not isinstance(keys, list) or not keys or not all(isinstance(k, str) and k for k in keys)):
        raise MifOpError("keys must be a non-empty list of data paths (or their starts), got %r. NOTHING was changed."
                         % (keys,))
    ad = src.animation_data
    act = ad.action if ad is not None else None
    if act is None:
        raise MifOpError("'%s' has no action to copy. NOTHING was changed." % src.name)
    slot = getattr(ad, "action_slot", None)
    curves = _fcurves(src)
    kept = [fc for fc in curves if keys is None or any(fc.data_path.startswith(k) for k in keys)]
    if not kept:
        raise MifOpError("none of '%s''s %d curve(s) (%s) starts with %s, so a copy would hold nothing. NOTHING was "
                         "changed." % (act.name, len(curves), ", ".join(sorted({fc.data_path for fc in curves}))[:300],
                                       ", ".join(keys or [])))

    rows, created = [], []
    for t in targets:
        prev = t.animation_data.action if t.animation_data is not None else None
        if copy:
            new = act.copy()
            new.name = "%s_%s" % (act.name, t.name)
            new_slot = None
            if slot is not None:
                new_slot = next((s for s in new.slots if s.identifier == slot.identifier), None)
            for owner, fcs in _slot_curve_bags(new, new_slot):
                for fc in fcs:
                    if keys is not None and not any(fc.data_path.startswith(k) for k in keys):
                        owner.fcurves.remove(fc)
                        continue
                    if offset:
                        for kp in fc.keyframe_points:
                            kp.co.x += offset
                            kp.handle_left.x += offset
                            kp.handle_right.x += offset
                        fc.update()
            created.append(new.name)
            use, use_slot = new, new_slot
        else:
            use, use_slot = act, slot
        if t.animation_data is None:
            t.animation_data_create()
        t.animation_data.action = use
        if use_slot is not None and hasattr(t.animation_data, "action_slot"):
            t.animation_data.action_slot = use_slot
        bound = getattr(t.animation_data, "action_slot", None)
        rows.append({"object": t.name, "action": use.name,
                     "slot": bound.identifier if bound is not None else None,
                     "curves": len(_fcurves(t)), "keyframes": sum(len(fc.keyframe_points) for fc in _fcurves(t)),
                     "previousAction": prev.name if prev is not None else None,
                     "previousSurvivesSave": (bool(prev.users) or bool(prev.use_fake_user)) if prev is not None
                     else None})
    return {"source": src.name, "action": act.name, "copy": copy, "keys": keys, "frameOffset": offset,
            "curvesCopied": len(kept), "targets": rows, "created": created}


OPS = {
    "copy_keyframes": op_copy_keyframes,
    "clean_keyframes": op_clean_keyframes,
    "move_keyframes": op_move_keyframes,
    "add_nla_strip": op_add_nla_strip,
    "set_nla_track": op_set_nla_track,
    "add_driver": op_add_driver,
    "remove_driver": op_remove_driver,
    "list_markers": op_list_markers,
    "set_marker": op_set_marker,
    "bake_to_keyframes": op_bake_to_keyframes,
    "set_keyframe": op_set_keyframe,
    "set_frame_range": op_set_frame_range,
    "list_keyframes": op_list_keyframes,
    "list_animation_data": op_list_animation_data,
    "delete_keyframe": op_delete_keyframe,
    "evaluate_at_frame": op_evaluate_at_frame,
    "edit_fcurve": op_edit_fcurve,
    "add_fcurve_modifier": op_add_fcurve_modifier,
    "list_actions": op_list_actions,
    "create_action": op_create_action,
    "set_action": op_set_action,
    "assign_action": op_assign_action,
    "copy_animation": op_copy_animation,
}
