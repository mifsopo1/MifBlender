"""Authoring geometry-node TREES: create a group, add nodes, link them, expose inputs.

WHAT WAS ALREADY POSSIBLE AND WHAT WAS NOT. add_modifier can already attach a NODES modifier - it
validates against bpy.types.Modifier.bl_rna and takes any type Blender knows - so "no geometry
nodes" was always too strong a claim. What could not be done was building the TREE: creating a
group, putting nodes in it, wiring them together, and exposing a value as a modifier input. That is
the half a procedural grime or cable system is made of, and it is the half this file adds.

=============================================================================
THE GROUP INTERFACE MOVED IN 4.0 AND THE OLD API IS GONE
=============================================================================
Up to 3.6 a group's sockets were tree.inputs / tree.outputs, with .new(type, name). From 4.0 they
live in tree.interface, created with .new_socket(name, in_out=, socket_type=), and the old
collections do not exist. This addon supports 3.6 through 5.0, so every interface call goes through
one helper that tries the new API and falls back - the same shape as the fcurve accessor in
ops_anim, and for the same reason: a helper that knows one version makes three of them red.

=============================================================================
A NODE TREE THAT IS NOT LINKED TO ITS OUTPUT DOES NOTHING, SILENTLY
=============================================================================
An unconnected Group Output is not an error in Blender, and it does NOT pass the geometry
through - the modifier evaluates to EMPTY and the object disappears from the viewport and from
every export. Measured on 3.6.23, 4.2.17, 4.4.0 and 5.0.1: a default cube through a NODES
modifier with an unlinked Group Output evaluates to 0 vertices on all four. This file said
"passes through unchanged" in four places, which sends somebody to the opposite end of the
pipeline - looking for a tree that ran and did nothing, rather than one that ate the mesh. So
list_group_nodes reports whether the output is reachable, and assign_node_group says so on the
way out.
"""
import os

import bpy

from .ops_common import (MifOpError, check_axis_dict, exclude_rows, finite_float, finite_floats, finite_int, 
                         get_object, limit_rows, reject_unknown, rnd, take, take_bool, take_float,
                         write_rna_property)


_CREATE_KEYS = {"name", "type", "withGroupIO", "tool"}
_ADDNODE_KEYS = {"group", "tree", "type", "nodeType", "name", "location", "inputs", "label",
                 "operation", "dataType", "domain", "mode", "nodeGroup",
                 "properties", "parent", "zone"}
_LINK_KEYS = {"group", "tree", "fromNode", "fromSocket", "toNode", "toSocket", "remove"}
_LIST_KEYS = {"group", "tree", "values"}
_SETNODE_KEYS = {"group", "tree", "node", "properties", "inputs", "label", "location",
                 # A NODE'S ITEM COLLECTIONS (2026-10-04): zone state, repeat, capture, bake, menu and index-switch items
                 # and a File Output node's slots - add, rename, remove; and the File Output's folder, either build
                 "items", "basePath",
                 # every unlinked socket hidden, or shown again (2026-10-04)
                 "hideUnusedSockets",
                 # an RGB Curves, Float Curve or Vector Curves node's control points (2026-10-06)
                 "curve"}

# ZONES (2026-10-04): (input bl_idname, output bl_idname, the output's item collection, first build). Ledger B265/B266.
_ZONES = {
    "SIMULATION": ("GeometryNodeSimulationInput", "GeometryNodeSimulationOutput", "state_items", (3, 6, 0)),
    "REPEAT": ("GeometryNodeRepeatInput", "GeometryNodeRepeatOutput", "repeat_items", (4, 2, 0)),
    "FOR_EACH": ("GeometryNodeForeachGeometryElementInput", "GeometryNodeForeachGeometryElementOutput", "main_items",
                 (4, 4, 0)),
    # A CLOSURE (2026-10-11, measured on 5.0.1 and 5.2.1): NodeClosureInput / NodeClosureOutput pair like the others;
    # the output holds two item collections - input_items (the closure's parameters, what zone.items makes) and
    # output_items (its results, set afterwards with set_node_property items {collection: "output_items"}).
    "CLOSURE": ("NodeClosureInput", "NodeClosureOutput", "input_items", (5, 0, 0)),
}

_RMNODE_KEYS = {"group", "tree", "node", "allowStructural",
                # the node's through-connection kept (2026-10-06)
                "dissolve"}

# NODES WHOSE REMOVAL BREAKS THE TREE RATHER THAN EDITING IT. A material with no Material Output
# renders black; a group with no Group Output evaluates to nothing and the object VANISHES, which
# this module already records measuring as 0 vertices on all four builds. Removable, but only when
# the caller says the word.
_STRUCTURAL_NODES = ("NodeGroupInput", "NodeGroupOutput", "ShaderNodeOutputMaterial",
                     "ShaderNodeOutputWorld", "ShaderNodeOutputLight", "CompositorNodeComposite",
                     "CompositorNodeViewer")
_IFACE_KEYS = {"group", "tree", "name", "socketType", "inOut", "default", "min", "max", "rename", "move", "remove",
               "confirm", "defaultInput"}
_ASSIGN_KEYS = {"object", "name", "group", "tree", "modifierName", "inputs", "reuse", "outputAttributes",
                # a group found by name in an asset library - the bundled one by default - and appended (2026-10-06)
                "library"}


def _tree(name):
    # THE ONE CHOKEPOINT ALL FIVE AUTHORING OPS GO THROUGH, which is why the scene compositor is
    # reached here rather than by a parallel set of compositor add/link/list ops. scene.node_tree is
    # not in bpy.data.node_groups - it belongs to the scene - so before 2026-09-03 nothing in this
    # module could address it and the entire compositing subsystem was outside the typed path.
    owned = _owned_tree(name)
    if owned is not None:
        return owned
    t = bpy.data.node_groups.get(name)
    if t is None:
        raise MifOpError("no node group named '%s'. create_node_group makes one; the groups in "
                         "this file are: %s" % (name, ", ".join(sorted(bpy.data.node_groups.keys()))
                                                or "(none)"))
    return t


def _iface_new(tree, name, socket_type, in_out):
    """Create a group socket on any supported Blender.

    4.0+ : tree.interface.new_socket(name, in_out="INPUT"|"OUTPUT", socket_type="NodeSocketFloat")
    3.6  : tree.inputs.new("NodeSocketFloat", name)
    The old collections do not merely still work in 4.0+, they are GONE, so this is a real branch
    rather than a compatibility nicety.
    """
    if hasattr(tree, "interface"):
        return tree.interface.new_socket(name=name, in_out=in_out, socket_type=socket_type)
    coll = tree.inputs if in_out == "INPUT" else tree.outputs
    return coll.new(socket_type, name)


def _iface_items(tree):
    """The group's sockets, each with the IDENTIFIER a caller needs to address it.

    THE NAME IS NOT THE KEY, and reporting only the name left the caller unable to build a modifier
    path from anything this bridge told them. A geometry-nodes modifier is addressed by socket
    IDENTIFIER - `modifiers["GeometryNodes"]["Socket_2"]` - and the identifier is generated, not
    chosen: measured 2026-09-14, the same "Lift" socket is `Input_0` on 3.6.23 and `Socket_2` on 4.4
    and later, because the older builds used tree.inputs and the newer ones use tree.interface.

    So a caller who hardcoded the 4.x spelling wrote a key that did not exist. On 3.6 that does not
    raise: the modifier is an id-property bag, so `mod["Socket_2"] = v` CREATES a new unused property
    which then keys and animates perfectly while the real socket never moves - the same
    two-wrong-names-succeed trap _MOD_SOCKET_VALUE_KEY records for the value key. test_blender_anim's
    A113 had exactly that shape: fcurve present, two keys, value flat at 0.0 on every frame.

    `identifier` is therefore reported on every row from 2026-09-14.
    """
    if hasattr(tree, "interface"):
        out = []
        for it in tree.interface.items_tree:
            if getattr(it, "item_type", "SOCKET") != "SOCKET":
                continue
            out.append({"name": it.name, "inOut": it.in_out,
                        "identifier": getattr(it, "identifier", None),
                        "socketType": getattr(it, "socket_type", None)})
        return out
    return ([{"name": s.name, "inOut": "INPUT", "identifier": getattr(s, "identifier", None),
              "socketType": s.bl_socket_idname}
             for s in tree.inputs] +
            [{"name": s.name, "inOut": "OUTPUT", "identifier": getattr(s, "identifier", None),
              "socketType": s.bl_socket_idname}
             for s in tree.outputs])


def op_create_node_group(params):
    """Create a geometry node group, with Group Input/Output nodes on a Geometry socket pair - NOT linked to each other.

    Until something is linked into Group Output the group outputs no geometry, so an object it is assigned to evaluates
    to nothing (assign_node_group's effectNote says so, and object_info's evaluatedVerts reads 0)."""
    reject_unknown(params, _CREATE_KEYS, "create_node_group")
    name = str(take(params, "name", default="MifNodes", kind=str))
    if name in (SCENE_COMPOSITOR, SCENE_WORLD) or name.startswith(RESERVED_PREFIXES):
        raise MifOpError("'%s' is reserved: names like %s, %s, '%s<name>', '%s<name>' and '%s<name>' "
                         "address trees that are OWNED by a scene, material, world or light rather than "
                         "living in bpy.data.node_groups, and a group by such a name would make which one "
                         "you meant depend on what happened to exist. Pick another name. NOTHING "
                         "was created."
                         % (name, SCENE_COMPOSITOR, SCENE_WORLD, MATERIAL_PREFIX, WORLD_PREFIX, LIGHT_PREFIX))
    kind = str(take(params, "type", default="GeometryNodeTree", kind=str))
    if kind not in ("GeometryNodeTree", "ShaderNodeTree", "CompositorNodeTree"):
        raise MifOpError("type must be GeometryNodeTree, ShaderNodeTree or CompositorNodeTree, "
                         "got '%s'. NOTHING was created." % kind)
    # READ BEFORE THE GROUP EXISTS. It was read on the line after bpy.data.node_groups.new, so
    # {withGroupIO: "flase"} was refused with the empty group left behind holding the name (measured on
    # 5.0.1, 2026-09-21).
    try:
        with_io = take_bool(params, "withGroupIO", default=True)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was created." % str(exc).rstrip("."))
    tool = _tool_spec(params.get("tool"), kind, "NOTHING was created.")
    tree = bpy.data.node_groups.new(name=name, type=kind)
    if tool is not None:
        _mark_tool(tree, *tool)

    made = []
    if with_io and kind == "GeometryNodeTree":
        # A geometry group with no Geometry in/out cannot be used as a modifier at all - the
        # modifier needs a geometry socket to feed and one to read back.
        _iface_new(tree, "Geometry", "NodeSocketGeometry", "INPUT")
        _iface_new(tree, "Geometry", "NodeSocketGeometry", "OUTPUT")
        gin = tree.nodes.new("NodeGroupInput")
        gout = tree.nodes.new("NodeGroupOutput")
        gin.location = (-400, 0)
        gout.location = (400, 0)
        made = [gin.name, gout.name]
    return {"group": tree.name, "type": kind, "nodes": made,
            # Blender renames on collision rather than failing, so a caller retrying a timed-out
            # create gets "Foo.001" while believing it holds "Foo" - and assign_node_group by the
            # name it asked for then finds the WRONG group, or none.
            "requestedName": name,
            "nameWasSuffixed": tree.name != name,
            "interface": _iface_items(tree),
            "nodeCount": len(tree.nodes),
            "tool": ({"modes": tool[0], "types": tool[1],
                      "operator": getattr(tree, "node_tool_idname", None) or None} if tool is not None else None),
            "apiNote": ("group sockets live in tree.interface from Blender 4.0 and in "
                        "tree.inputs/outputs before it; the old collections are GONE in 4.0+, not "
                        "deprecated.")}


# WHICH bpy.data COLLECTION A POINTER SOCKET DRAWS FROM.
#
# THE HOLE THIS CLOSES. The socket writer coerced every non-list value with float(), so the only
# thing a socket could hold was a number. Every datablock socket - Object, Collection, Material,
# Image, Texture - was unreachable, and so was String, because float("Cube") and float("hello")
# fail the same way. That is not a corner: "scatter this on that", "instance this collection",
# "use this material", "read this image" are what geometry nodes are FOR, and all of them need a
# name to become a pointer. A caller got "cannot write 'Cube' to socket 'Object'" and no route.
#
# THE NAME IS RESOLVED HERE rather than left to Blender, because assigning a string to a pointer
# socket raises a TypeError naming the type and not the mistake. Refusing with the datablocks that
# ARE present turns "that failed" into "you meant one of these".
_POINTER_SOCKETS = {
    "NodeSocketObject": ("objects", "object"),
    "NodeSocketCollection": ("collections", "collection"),
    "NodeSocketMaterial": ("materials", "material"),
    "NodeSocketImage": ("images", "image"),
    "NodeSocketTexture": ("textures", "texture"),
    # A SOUND (2026-10-11): 5.2's Sample Sound Frequencies takes a bpy.data sound on its Sound socket, and a name sent
    # there was refused as "must be a number" - the float() coercion this table exists to bypass.
    "NodeSocketSound": ("sounds", "sound"),
}


def _modifier_socket_group(mod, identifier):
    """The 5.2+ home for one geometry-nodes modifier input, or None on older builds.

    BLENDER 5.2 MOVED THESE AND THE OLD PATH DOES NOT DEGRADE - IT RAISES. Up to and including
    5.0, a modifier input was a flat id-property on the modifier: `mod["Socket_2"] = 7.0`. On 5.2
    the NodesModifier has no id-property support at all ("bpy_struct[key] = val: id properties not
    supported for this type") and the values live in mod.properties.inputs[identifier], an
    IDPropertyGroup keyed by socket identifier.

    Measured 2026-09-08 on all five installed builds, which is what this gate is chosen from:

        build        mod.properties.inputs    mod[identifier] = v
        3.6.23              absent                   works
        4.2.17 LTS          absent                   works
        4.4.0               absent                   works
        5.0.1               absent                   works
        5.2.1 LTS           PRESENT                  raises

    The two are exact inverses, so the CAPABILITY is a sound gate and no version number is needed -
    which is this addon's idiom everywhere else (there is one numeric bpy.app.version comparison in
    the whole codebase, at ops_mesh.py:3160).
    """
    props = getattr(mod, "properties", None)
    inputs = getattr(props, "inputs", None)
    if inputs is None:
        return None
    try:
        return inputs[identifier]
    except (KeyError, TypeError):
        return None


# THE KEY INSIDE THE GROUP IS 'value', AND GETTING THAT WRONG IS SILENT. Measured on 5.2.1: a
# group holds {'value': ..., 'type': 1} plus 'attribute_name' for attribute-capable sockets, and
# assigning to 'default_value' or 'Value' SUCCEEDS - it simply creates a new unused key that reads
# back the value you just wrote while Blender goes on using 'value'. A readback cannot catch that,
# so the name is recorded here rather than rediscovered by someone trusting a round-trip.
_MOD_SOCKET_VALUE_KEY = "value"


def _set_modifier_socket(mod, identifier, value):
    """Write one modifier input on whichever storage this build uses."""
    grp = _modifier_socket_group(mod, identifier)
    if grp is None:
        mod[identifier] = value
    else:
        grp[_MOD_SOCKET_VALUE_KEY] = value
    # TAGGED: neither write is an RNA write. MEASURED 2026-09-16 on all five builds, a group translating by a 'Lift'
    # input: re-setting Lift 1.0 -> 2.5 left the evaluated mesh at z 1.0 until the object was tagged - so
    # assign_node_group's reuse path (no other write in the call) changed the stored input and not the geometry.
    mod.id_data.update_tag()


def _get_modifier_socket(mod, identifier):
    """Read one modifier input back, same split. Returns None when it did not hold."""
    grp = _modifier_socket_group(mod, identifier)
    if grp is None:
        return mod.get(identifier)
    return grp.get(_MOD_SOCKET_VALUE_KEY)


def _socket_value(kind, sock_name, val, where, tail="NOTHING was changed."):
    """Coerce a JSON value for a socket of type `kind`. Raises MifOpError naming the options.

    THE TAIL IS THE CALLER'S TO TELL. This refused with "NOTHING was changed." for everybody, and
    two of its three callers had already built something by the time they asked: add_group_node has
    a node in the tree, assign_node_group may have just made a modifier. Both then answered a bad
    socket value with a sentence promising the file was untouched. Same shape as check_axis_dict's
    tail, and the same reason - a shared refusal cannot know what its caller has already done, so it
    has to be told rather than guess.

    TAKES THE TYPE STRING, NOT A SOCKET, because the two callers hold different things: the node
    writer has a real socket with .bl_idname, and assign_node_group has only the group's INTERFACE,
    where the type is it.socket_type on 4.x and s.bl_socket_idname on 3.6. Passing the resolved
    string keeps one implementation instead of two that drift.
    """
    if kind in _POINTER_SOCKETS:
        attr, noun = _POINTER_SOCKETS[kind]
        # None CLEARS a pointer socket, which is a legitimate thing to want and is not the same as
        # leaving it alone - an unset Object socket and one pointed at the wrong object look
        # identical in a response that only echoes what was sent.
        if val is None:
            return None
        if not isinstance(val, str):
            raise MifOpError("socket '%s' on %s holds a %s, so it needs the NAME of one as a "
                             "string - got %r. %s"
                             % (sock_name, where, noun, val, tail))
        coll = getattr(bpy.data, attr)
        found = coll.get(val)
        if found is None:
            have = sorted(d.name for d in coll)[:25]
            raise MifOpError("no %s named '%s'. This file has: %s. %s"
                             % (noun, val, ", ".join(have) if have else "(none)", tail))
        return found
    # A VECTOR, COLOUR OR ROTATION DEFAULT IS A SEQUENCE. add_group_node writes those element by
    # element into a live socket; the group INTERFACE takes the whole thing at once.
    if isinstance(val, (list, tuple)):
        # A bare float() here let ["mif", "not-a-value"] out as a raw ValueError - "could not
        # convert string to float: 'mif'" - straight past this addon's refusal contract. Found by
        # giving the matrix's corrupted-payload pass a LIST sentinel alongside its string and dict.
        return finite_floats(val, sock_name)
    # A MENU SOCKET IS AN ENUM and a String socket is text; float() destroyed both.
    if kind in ("NodeSocketString", "NodeSocketMenu"):
        return str(val)
    if kind == "NodeSocketBool":
        # bool() ACCEPTS ANYTHING - bool({"mif":"bad"}) is True - so a garbage default became a
        # silent True. Same shape as _write_node_property's boolean branch, in code I wrote the
        # same day, found by the matrix reporting that a DICT had been accepted.
        if not isinstance(val, bool) and val not in (0, 1):
            raise MifOpError("a %s default is true or false, got %r. %s"
                             % (kind, val, tail))
        return bool(val)
    # EVERY INT SUBTYPE, not only the plain one. A Curve Spiral's Resolution is a NodeSocketIntUnsigned
    # (Percentage and Factor are the others), and == "NodeSocketInt" sent it to finite_float: Blender
    # refused 64.0 with "expected an int type, not float", so a whole class of integer inputs could
    # not be set at all. Found 2026-09-13 by the fivekinds showcase. The Interface guard keeps the
    # prefix off Blender 3.x's NodeSocketInterface* class names.
    if kind.startswith("NodeSocketInt") and not kind.startswith("NodeSocketInterface"):
        return finite_int(val, sock_name)
    # A NaN socket default is accepted by Blender and renders as black or as nothing, with the
    # socket reading back nan and the response agreeing the value was written.
    return finite_float(val, sock_name)


# WHICH NODE TYPES HOLD ANOTHER TREE. A Group node is the only way to compose procedural systems,
# and add_group_node could create one and never point it at anything: node_tree stayed None, so the
# node had no inputs, no outputs and no effect, and the op reported success with "inputs": [].
_GROUP_NODE_TYPES = ("GeometryNodeGroup", "ShaderNodeGroup", "CompositorNodeGroup",
                     "TextureNodeGroup")


# PROPERTIES THAT ARE NOT A NODE'S BUSINESS TO SET THROUGH THIS PARAMETER. name and label have
# their own arguments and setting them twice is ambiguous; location likewise; node_tree has
# nodeGroup, which carries the recursion read-back. The rest are Blender's own bookkeeping.
_PROPERTY_DENY = {"name", "label", "location", "node_tree", "parent", "select", "width",
                  "height", "bl_idname", "bl_label", "type", "internal_links"}


# ===========================================================================
# A NODE-TYPE VOCABULARY, AND THE VERSION GATE BEHIND IT
# ===========================================================================
# WHY A VOCABULARY. `type` was the raw bl_idname and nothing else, so authoring a procedural
# material meant knowing that a noise texture is "ShaderNodeTexNoise", a colour ramp is
# "ShaderNodeValToRGB" (not ...ColorRamp, which is what it is called everywhere a human reads it)
# and texture coordinates are "ShaderNodeTexCoord". Those are learnable, but ValToRGB in
# particular is the kind of name a caller guesses wrong every time, and a wrong guess is a
# refusal rather than a graph.
#
# THE RAW bl_idname STILL WORKS. This is a lookup in front of tree.nodes.new(), not a
# replacement for it: an unmatched string is passed through untouched, so every existing caller
# and every node type not listed here behaves exactly as before.
#
# Keys are matched with the punctuation and case removed, so "colorRamp", "color_ramp" and
# "ColorRamp" are one key.
SHADER_NODE_ALIASES = {
    # -- procedural textures, which is what this table was written for -----------------------
    "noise": "ShaderNodeTexNoise",
    "voronoi": "ShaderNodeTexVoronoi",
    "wave": "ShaderNodeTexWave",
    "gradient": "ShaderNodeTexGradient",
    "magic": "ShaderNodeTexMagic",
    "brick": "ShaderNodeTexBrick",
    "checker": "ShaderNodeTexChecker",
    "musgrave": "ShaderNodeTexMusgrave",
    "whitenoise": "ShaderNodeTexWhiteNoise",
    "gabor": "ShaderNodeTexGabor",
    "image": "ShaderNodeTexImage",
    "environment": "ShaderNodeTexEnvironment",
    "sky": "ShaderNodeTexSky",
    # -- ramps, curves and conversion --------------------------------------------------------
    "colorramp": "ShaderNodeValToRGB",
    "rgbcurve": "ShaderNodeRGBCurve",
    "rgbtobw": "ShaderNodeRGBToBW",
    "huesaturation": "ShaderNodeHueSaturation",
    "brightcontrast": "ShaderNodeBrightContrast",
    "gamma": "ShaderNodeGamma",
    "invert": "ShaderNodeInvert",
    # -- relief ------------------------------------------------------------------------------
    "bump": "ShaderNodeBump",
    "normalmap": "ShaderNodeNormalMap",
    "displacement": "ShaderNodeDisplacement",
    # -- coordinates -------------------------------------------------------------------------
    "mapping": "ShaderNodeMapping",
    "texcoord": "ShaderNodeTexCoord",
    "texturecoordinate": "ShaderNodeTexCoord",
    "uvmap": "ShaderNodeUVMap",
    "attribute": "ShaderNodeAttribute",
    "objectinfo": "ShaderNodeObjectInfo",
    # -- maths and mixing, because a real material blends several patterns --------------------
    "math": "ShaderNodeMath",
    "vectormath": "ShaderNodeVectorMath",
    "maprange": "ShaderNodeMapRange",
    "clamp": "ShaderNodeClamp",
    "mix": "ShaderNodeMix",
    "mixrgb": "ShaderNodeMixRGB",
    "mixshader": "ShaderNodeMixShader",
    "addshader": "ShaderNodeAddShader",
    "value": "ShaderNodeValue",
    "rgb": "ShaderNodeRGB",
    "fresnel": "ShaderNodeFresnel",
    "layerweight": "ShaderNodeLayerWeight",
    # -- shaders and output ------------------------------------------------------------------
    "principled": "ShaderNodeBsdfPrincipled",
    "emission": "ShaderNodeEmission",
    "transparent": "ShaderNodeBsdfTransparent",
    "materialoutput": "ShaderNodeOutputMaterial",
}

# WHAT TO SAY WHEN A NODE TYPE IS NOT ON THIS BUILD, for the two cases where "it does not exist"
# is a true but useless answer.
#
# MUSGRAVE IS THE ONE THIS TABLE EXISTS FOR. Measured rather than taken from release notes: it is
# in the shader add menu of the 3.6 install on this machine (scripts/startup/nodeitems_builtins.py)
# and in NONE of the 4.2, 4.4 or 5.0 ones - it was folded into the Noise texture, whose fractal
# behaviour is now a property. This repo had already recorded the same boundary from the other
# side when create_texture was written: the legacy bpy.data.textures datablock still offers
# MUSGRAVE on all four builds while the shader NODE does not, so "Musgrave exists" is true of one
# system and false of the other.
#
# THE ADVICE NAMES NO PROPERTY. Which knob replaced it is drawn off the LIVE RNA below rather
# than written here, because the shader nodes draw their buttons in C and nothing in the shipped
# Python says what those properties are called - a name written from memory here would be the
# exact "pattern-matched from a plausible name" mistake this addon keeps finding.
_NODE_TYPE_REPLACEMENTS = {
    "ShaderNodeTexMusgrave": (
        "ShaderNodeTexNoise",
        "the Musgrave texture was FOLDED INTO the Noise texture - its fractal types became a "
        "property of ShaderNodeTexNoise rather than a node of their own. Use type 'noise' and set "
        "the fractal knob through `properties`"),
    "ShaderNodeMixRGB": (
        "ShaderNodeMix",
        "Mix RGB is the LEGACY colour-mix node, superseded by the typed Mix node. Use type 'mix' "
        "and set its data type through `properties`"),
}


def _rna_props(idname):
    """The writable property names of a REGISTERED node class, without creating a node.

    Read off bpy.types.<idname>.bl_rna, so a refusal can name what this build actually offers
    instead of a property name remembered from a changelog. Creating a node to ask would mutate
    the tree inside a refusal path, which is the one thing every op here is arranged to avoid.
    """
    cls = getattr(bpy.types, idname, None)
    rna = getattr(cls, "bl_rna", None)
    if rna is None:
        return []
    return sorted(p.identifier for p in rna.properties
                  if not p.is_readonly and p.identifier not in _PROPERTY_DENY
                  and not p.identifier.startswith("bl_"))


def _canon_type_key(text):
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def resolve_node_type(requested):
    """A friendly node name or a raw bl_idname -> the bl_idname to create, or a refusal.

    REFUSED BY NAME WITH THE VERSION, never silently skipped and never left to
    tree.nodes.new()'s own message, which says only that Blender does not know the type. That is
    true and unhelpful for a node that USED to exist: the caller wants to know it was removed,
    when, and what does the job now.
    """
    requested = str(requested)
    idname = SHADER_NODE_ALIASES.get(_canon_type_key(requested), requested)
    if hasattr(bpy.types, idname):
        return idname
    # NOT REGISTERED ON THIS BUILD. Everything below is message quality - the answer is already no.
    swap = _NODE_TYPE_REPLACEMENTS.get(idname)
    if swap is not None:
        replacement, why = swap
        props = _rna_props(replacement)
        raise MifOpError(
            "Blender %s has no node type '%s'%s - %s. That node's writable properties on THIS "
            "build, read off its own RNA rather than remembered: %s. NOTHING was added."
            % (bpy.app.version_string, idname,
               (" (asked for as '%s')" % requested) if requested != idname else "",
               why, ", ".join(props) or "(none)"))
    known = sorted(SHADER_NODE_ALIASES)
    raise MifOpError(
        "Blender %s has no node type '%s'%s - it is not registered on this build, so it is either "
        "gone or not yet added. The friendly names this op knows are: %s. Any other raw bl_idname "
        "is passed through to Blender unchanged. NOTHING was added."
        % (bpy.app.version_string, idname,
           (" (asked for as '%s')" % requested) if requested != idname else "",
           ", ".join(known)))


def usable_sockets(collection, name):
    """(enabled matches, disabled matches) for `name` in a socket collection.

    THE SAME NAME CAN APPEAR SEVERAL TIMES ON ONE NODE, and this is the trap that makes the Mix
    node almost unusable from a script. ShaderNodeMix carries a Factor, an A and a B for EVERY
    data type it supports - float, vector, colour, rotation - all named identically, and Blender
    switches between them with the `enabled` flag rather than by adding and removing sockets.
    node.inputs["A"] returns the FIRST match, which is the float one, so setting a colour on a
    Mix node whose data_type is RGBA writes to a socket nothing reads: accepted, reported as
    applied, and completely inert.

    Blender's own bundled node_wrangler filters every socket list it builds by `.enabled` for
    this reason (utils/nodes.py: `not socket.hide and socket.enabled and socket.type != 'CUSTOM'`),
    which is the corroboration that this is the discriminator rather than a guess.
    """
    matches = [s for s in collection if s.name == name]
    return ([s for s in matches if getattr(s, "enabled", True)],
            [s for s in matches if not getattr(s, "enabled", True)])


def socket_label(sock):
    """'name' or 'name (inactive)' - so a listing never hides that half of it is not read."""
    return sock.name if getattr(sock, "enabled", True) else "%s (inactive)" % sock.name


# ===========================================================================
# READING A GRAPH BACK
# ===========================================================================
# THE ASYMMETRY THIS CLOSES. Every node-authoring op here could WRITE a socket default and a node
# property, and nothing could read either one back: list_group_nodes reported socket NAMES,
# describe_material reported the Principled BSDF and image textures. So a caller could set a Noise
# texture's scale to 12 and had no way, through this bridge, to find out whether it landed - which
# for a procedural material is the whole of the work. That is the read/write asymmetry this repo
# already treats as a defect class (blendMethod was reported by two ops and writable by none; this
# is the same shape pointing the other way).
_NODE_BASE_PROPS = None


def _base_node_props():
    """Property names every node has - Blender's own bookkeeping rather than node settings.

    Computed from bpy.types.Node once, not listed by hand, so it cannot drift: mute, hide, color,
    use_custom_color, show_options and the rest live on the base class, and reporting ten of them
    beside a Noise texture's two real knobs is how a read-back becomes unreadable.
    """
    global _NODE_BASE_PROPS
    if _NODE_BASE_PROPS is None:
        base = getattr(bpy.types, "Node", None)
        rna = getattr(base, "bl_rna", None)
        _NODE_BASE_PROPS = ({p.identifier for p in rna.properties} if rna is not None else set())
    return _NODE_BASE_PROPS


def _socket_default(sock):
    """A socket's default_value as JSON, or None when it has no value at all.

    A Shader or Geometry socket carries no value - it carries a connection - so `hasattr` is the
    test rather than a type list that would need a new entry every release.
    """
    if not hasattr(sock, "default_value"):
        return None
    val = sock.default_value
    try:
        if hasattr(val, "__len__") and not isinstance(val, str):
            return rnd(list(val))
        if isinstance(val, float):
            return round(float(val), 6)
        if isinstance(val, (bool, int, str)):
            return val
    except (TypeError, ValueError):
        return None
    # A POINTER socket - an Object, a Collection, an Image. The NAME, because the datablock itself
    # is not JSON and a repr of it tells a reader nothing they can use in the next call.
    return getattr(val, "name", None)


def node_settings(node):
    """The node's OWN properties and their current values - what `properties` can write.

    Deliberately the same set _write_node_property accepts, so the read half and the write half of
    this addon describe the same node. A property that can be written and not read is how a caller
    ends up unable to check its own work; one that can be read and not written is a capability
    hiding behind a report.
    """
    out = {}
    base = _base_node_props()
    for prop in node.bl_rna.properties:
        key = prop.identifier
        if (prop.is_readonly or key in _PROPERTY_DENY or key.startswith("bl_")
                or key in base):
            continue
        try:
            val = getattr(node, key)
        except AttributeError:
            continue
        if prop.type == "POINTER":
            out[key] = getattr(val, "name", None)
        elif prop.type in ("ENUM", "STRING"):
            out[key] = val
        elif prop.type == "BOOLEAN":
            out[key] = (list(val) if hasattr(val, "__len__") else bool(val))
        elif prop.type == "INT":
            out[key] = (list(val) if hasattr(val, "__len__") else int(val))
        elif prop.type == "FLOAT":
            out[key] = (rnd(list(val)) if hasattr(val, "__len__") else round(float(val), 6))
        # COLLECTION and anything else is skipped here on purpose: the one collection that matters
        # is a colour ramp's stops, and those get a field of their own below rather than a generic
        # dump that would report a CurveMapping as an empty object.
    return out


def color_ramp_stops(node):
    """[{position, color}] for a node that carries a ColorRamp, else None.

    ANY node with a `color_ramp` - ShaderNodeValToRGB is the one people mean, and the compositor
    and the legacy texture nodes have their own. Asked by attribute rather than by type list so it
    keeps working for the ones not thought of here.
    """
    ramp = getattr(node, "color_ramp", None)
    if ramp is None:
        return None
    return {
        "interpolation": ramp.interpolation,
        "colorMode": ramp.color_mode,
        "hueInterpolation": getattr(ramp, "hue_interpolation", None),
        # POSITION ORDER IS BLENDER'S, NOT THE CALLER'S. The elements collection is kept sorted by
        # position, so index 0 is always the leftmost stop whatever order it was written in.
        "stops": [{"position": round(float(e.position), 6), "color": rnd(list(e.color))}
                  for e in ramp.elements],
        "stopCount": len(ramp.elements),
    }


# A NODE'S CURVE MAPPING (2026-10-06): the component names of each curve-bearing node, in the order of its
# mapping.curves. RGB Curves keeps the per-channel curves first and the combined C curve last (read off the node on
# 4.2.23 and 5.2.2: four curves; the C curve is index 3). A node not named here is addressed by index.
_CURVEMAP_COMPONENTS = {
    "ShaderNodeRGBCurve": ("R", "G", "B", "C"), "CompositorNodeCurveRGB": ("R", "G", "B", "C"),
    "ShaderNodeVectorCurve": ("X", "Y", "Z"), "CompositorNodeCurveVec": ("X", "Y", "Z"),
    "CompositorNodeHueCorrect": ("H", "S", "V"),
}
_CURVEMAP_HANDLES = ("AUTO", "AUTO_CLAMPED", "VECTOR")


def curve_mapping_points(node):
    """{component -> [[x, y, handle], ...]} plus the clip range for a node with a curve mapping, else None.

    THE GENERIC SETTINGS DUMP SKIPS IT: node_settings drops a COLLECTION/POINTER, so an RGB Curves or Float Curve
    read back as nothing and a shaped curve was indistinguishable from a straight one."""
    mapping = getattr(node, "mapping", None)
    if mapping is None or not hasattr(mapping, "curves"):
        return None
    names = _CURVEMAP_COMPONENTS.get(node.bl_idname)
    curves = {}
    for i, cu in enumerate(mapping.curves):
        key = names[i] if names and i < len(names) else ("VALUE" if len(mapping.curves) == 1 else str(i))
        curves[key] = [[round(float(p.location[0]), 6), round(float(p.location[1]), 6), p.handle_type]
                       for p in cu.points]
    return {"curves": curves, "useClip": bool(mapping.use_clip),
            "clip": [round(float(v), 6) for v in (mapping.clip_min_x, mapping.clip_min_y, mapping.clip_max_x,
                                                  mapping.clip_max_y)]}


def _curvemap_plan(node, spec, tail):
    """[(curve index, component name, [(x, y, handle)])] from set_node_property's `curve`, every value checked
    against the node's own mapping - nothing written."""
    mapping = getattr(node, "mapping", None)
    if mapping is None or not hasattr(mapping, "curves"):
        raise MifOpError("curve shapes a node's curve mapping (RGB Curves, Float Curve, Vector Curves), and '%s' is a "
                         "%s with none. %s" % (node.name, node.bl_idname, tail))
    rows = spec if isinstance(spec, list) else [spec]
    if not rows or not all(isinstance(r, dict) for r in rows):
        raise MifOpError("curve is {component, points [[x, y], ...], handles} or a list of them, got %r. %s"
                         % (spec, tail))
    names = _CURVEMAP_COMPONENTS.get(node.bl_idname)
    count = len(mapping.curves)
    offered = list(names[:count]) if names else (["VALUE"] if count == 1 else [str(i) for i in range(count)])
    plan, seen = [], set()
    for k, row in enumerate(rows):
        bad = sorted(set(row) - {"component", "points", "handles"})
        if bad:
            raise MifOpError("curve[%d] has unknown key(s) %s - accepted: component, points, handles. %s"
                             % (k, bad, tail))
        comp = row.get("component")
        if comp is None and count == 1:
            comp = offered[0]
        comp = str(comp).upper() if comp is not None else None
        if comp not in offered:
            raise MifOpError("curve[%d] component %r: '%s' (%s) has %s. %s"
                             % (k, row.get("component"), node.name, node.bl_idname, ", ".join(offered), tail))
        if comp in seen:
            raise MifOpError("curve[%d] names component %s twice. %s" % (k, comp, tail))
        seen.add(comp)
        pts = row.get("points")
        if not isinstance(pts, list) or len(pts) < 2 or not all(isinstance(p, (list, tuple)) and len(p) == 2
                                                                  for p in pts):
            raise MifOpError("curve[%d] points is a list of two or more [x, y], got %r. %s" % (k, pts, tail))
        xy = [finite_floats(p, "curve[%d].points[%d]" % (k, j)) for j, p in enumerate(pts)]
        handles = row.get("handles", "AUTO")
        hl = handles if isinstance(handles, list) else [handles] * len(xy)
        if len(hl) != len(xy) or not all(isinstance(h, str) and h.upper() in _CURVEMAP_HANDLES for h in hl):
            raise MifOpError("curve[%d] handles is one of %s, or a list of them one per point, got %r. %s"
                             % (k, ", ".join(_CURVEMAP_HANDLES), handles, tail))
        xs = sorted(x for x, _y in xy)
        if any(b - a < 1e-6 for a, b in zip(xs, xs[1:])):
            raise MifOpError("curve[%d] has two points at the same x - a curve maps each x to one y. %s" % (k, tail))
        if mapping.use_clip:
            lo_x, lo_y, hi_x, hi_y = mapping.clip_min_x, mapping.clip_min_y, mapping.clip_max_x, mapping.clip_max_y
            out = [p for p in xy if not (lo_x - 1e-6 <= p[0] <= hi_x + 1e-6 and lo_y - 1e-6 <= p[1] <= hi_y + 1e-6)]
            if out:
                raise MifOpError("curve[%d] point(s) %s lie outside the node's clip range x %g..%g, y %g..%g - Blender "
                                 "would clamp them. %s" % (k, out, lo_x, hi_x, lo_y, hi_y, tail))
        pts3 = sorted(((x, y, h.upper()) for (x, y), h in zip(xy, hl)), key=lambda t: t[0])
        plan.append((offered.index(comp), comp, pts3))
    return plan


def _curvemap_snapshot(node):
    return [[(tuple(p.location), p.handle_type) for p in cu.points] for cu in node.mapping.curves]


def _curvemap_write(node, rows):
    """Write [(curve index, [(x, y, handle)])] onto the node's mapping: the point count matched first (a curve keeps
    at least two points), then each point set in x order, then the mapping updated."""
    mapping = node.mapping
    for idx, pts in rows:
        cu = mapping.curves[idx]
        while len(cu.points) > len(pts):
            cu.points.remove(cu.points[-1])
        while len(cu.points) < len(pts):
            cu.points.new(0.5, 0.5)
        for p, (x, y, h) in zip(cu.points, pts):
            p.location = (x, y)
            p.handle_type = h
    mapping.update()
    # THE TREE IS TAGGED, or nothing downstream sees the edit: a Float Curve in a Nodes modifier evaluated the old
    # curve after mapping.update() alone, and the new one after update_tag (measured 4.2.23 and 5.2.2).
    node.id_data.update_tag()


def node_readback(node, values=True):
    """One node as JSON: its shape always, its settings and socket values when asked.

    A LINKED INPUT REPORTS ITS SOURCE AND NOT ITS DEFAULT, and that is not tidiness. Blender
    evaluates the incoming link and never reads default_value on a connected socket, so printing
    the stale default beside a live connection invites exactly the mistake set_material_properties
    refuses outright: believing a number that nothing renders.
    """
    out = {"name": node.name, "type": node.bl_idname, "label": node.label,
           "location": [round(node.location[0], 2), round(node.location[1], 2)]}
    if not values:
        out["inputs"] = [socket_label(s) for s in node.inputs]
        out["outputs"] = [socket_label(s) for s in node.outputs]
        return out

    ins = []
    for sock in node.inputs:
        # identifier BESIDE name: a Math node's three inputs are all "Value", and only the identifier
        # (Value / Value_001 / Value_002) says which operand a value or a link belongs to.
        entry = {"name": sock.name, "identifier": sock.identifier, "type": sock.bl_idname,
                 "active": bool(getattr(sock, "enabled", True)),
                 "linked": bool(sock.is_linked)}
        if sock.is_linked:
            entry["from"] = ["%s.%s" % (l.from_node.name, l.from_socket.name) for l in sock.links]
        else:
            entry["value"] = _socket_default(sock)
        ins.append(entry)
    out["inputs"] = ins
    out["outputs"] = [{"name": s.name, "active": bool(getattr(s, "enabled", True)),
                       "linkedTo": ["%s.%s" % (l.to_node.name, l.to_socket.name)
                                    for l in s.links]}
                      for s in node.outputs]
    out["settings"] = node_settings(node)
    ramp = color_ramp_stops(node)
    if ramp is not None:
        out["colorRamp"] = ramp
    curves = curve_mapping_points(node)
    if curves is not None:
        out["curveMapping"] = curves
    # A MUTED NODE PASSES ITS INPUT STRAIGHT THROUGH and is invisible in every other field. It is
    # reported only when true so it reads as a warning rather than as noise on every node.
    if getattr(node, "mute", False):
        out["muted"] = True
        out["mutedNote"] = ("this node is MUTED - Blender passes the input through and the node "
                            "does nothing, while every setting on it still reads back correctly.")
    return out


def _write_node_property(node, key, value, tail="NOTHING was changed."):
    """Write one RNA property on a node, chosen by the property's OWN declared type.

    THE TAIL IS THE CALLER'S TO TELL, for the reason _socket_value's docstring gives above: a
    shared refusal cannot know what its caller has already built. add_group_node and
    add_procedural_texture both remove the node they made before the refusal leaves (add_group_node
    since 2026-09-16 - it used to leave it, and said "The node WAS added."), and set_node_property
    changes an existing node, so each passes its own sentence - and a refusal that lies about the state
    it left is worse than one that says nothing about it.

    THE FOUR-KNOB PROBLEM. This op could set exactly `operation`, `dataType`, `domain` and `mode`,
    each hardcoded and each uppercased as a string. Blender ships hundreds of node types and most
    of what makes a node do a particular thing is a property rather than a socket - a Math node's
    clamp, a Noise texture's dimensions, a Map Range's interpolation, a Mix node's blend type. All
    of it was unreachable, and a curated list of four was never going to catch up.

    DISPATCHED ON prop.type, so an enum is validated against ITS OWN items rather than uppercased
    and hoped for. Uppercasing worked for the four that happened to be SCREAMING_CASE enums and
    would have quietly mangled a string property.
    """
    # THE WORDING IS THIS WRAPPER'S, THE DISPATCH IS SHARED. The body moved to ops_common.write_rna_property
    # on 2026-09-20 so set_modifier could use it; every sentence a node refusal printed before is printed
    # exactly as it was, because `who` and `kind` below are the two strings it was built from.
    return write_rna_property(
        node, key, value, tail, "node '%s' (%s)" % (node.name, node.bl_idname), node.bl_idname, _PROPERTY_DENY,
        # node_tree IS a real pointer, denied here on purpose - nodeGroup sets it and checks the group is the
        # right kind of tree for the node - so the refusal says where to go.
        extra=(" A group node's tree is set with nodeGroup, which checks the kind of tree."
               if key == "node_tree" else ""))


def _item_types(coll):
    """The socket types this item collection's new() takes on this build, or None when it takes no type."""
    fn = coll.bl_rna.functions.get("new")
    prm = fn.parameters.get("socket_type") if fn is not None else None
    return [e.identifier for e in prm.enum_items] if prm is not None else None


def _items_owner(node):
    """The node that holds a zone's items: the output, reached from a paired input."""
    paired = getattr(node, "paired_output", None)
    return paired if paired is not None else node


def _item_rows(node, cname):
    """Each item of a collection with the identifiers of its sockets on the node and on its zone partner."""
    owner = _items_owner(node)
    partners = [owner] + [n for n in owner.id_data.nodes if getattr(n, "paired_output", None) is not None
                          and n.paired_output.name == owner.name]
    rows = []
    for it in getattr(owner, cname):
        name = getattr(it, "name", None)
        socks = {}
        for n in partners:
            socks[n.name] = {"input": next((s.identifier for s in n.inputs if s.name == name), None),
                             "output": next((s.identifier for s in n.outputs if s.name == name), None)}
        rows.append({"name": name, "type": getattr(it, "socket_type", getattr(it, "data_type", None)),
                     "sockets": socks})
    return rows


def _add_items(coll, items, tail):
    """Validate every {type, name} against the build's own enum, then make them. Raises before the first .new()."""
    types = _item_types(coll)
    plan = []
    for it in items:
        if not isinstance(it, dict) or set(it) - {"type", "name"} or not isinstance(it.get("name"), str):
            raise MifOpError("an item is {type, name}; got %r. %s" % (it, tail))
        t = it.get("type")
        if types is None:
            if t is not None:
                raise MifOpError("this collection takes names only, no type. %s" % tail)
        else:
            t = str(t or "").upper()
            if t not in types:
                raise MifOpError("item type %r is not one of this Blender's %s. %s" % (it.get("type"), ", ".join(types),
                                                                                     tail))
        plan.append((t, it["name"]))
    for t, name in plan:
        if types is None:
            coll.new(name)
        else:
            coll.new(t, name)
    return len(plan)


def _add_zone(tree, params):
    """add_group_node zone - see op_add_group_node. Both nodes go again on any refusal."""
    spec = params.get("zone")
    if not isinstance(spec, dict) or set(spec) - {"kind", "items", "link"}:
        raise MifOpError("zone is {kind SIMULATION | REPEAT | FOR_EACH | CLOSURE, items [{name, type}], link}; got %r. NOTHING "
                         "was added." % (spec,))
    if params.get("type") is not None or params.get("nodeType") is not None:
        raise MifOpError("zone makes its own two nodes; drop type. NOTHING was added.")
    for key in ("inputs", "properties", "operation", "dataType", "domain", "mode", "nodeGroup", "parent"):
        if params.get(key) is not None:
            raise MifOpError("%s sets one node; a zone is two. Set it afterwards with set_node_property. NOTHING was "
                             "added." % key)
    kind = str(spec.get("kind") or "").upper()
    if kind not in _ZONES:
        raise MifOpError("zone.kind must be SIMULATION, REPEAT, FOR_EACH or CLOSURE, got %r. NOTHING was added."
                         % spec.get("kind"))
    if tree.bl_idname != "GeometryNodeTree":
        raise MifOpError("a zone lives in a geometry node tree; '%s' is a %s. NOTHING was added."
                         % (tree.name, tree.bl_idname))
    in_id, out_id, cname, first = _ZONES[kind]
    if bpy.app.version < first:
        raise MifOpError("the %s zone arrived in Blender %d.%d; this is %s. NOTHING was added."
                         % (kind, first[0], first[1], bpy.app.version_string))
    items = spec.get("items") or []
    if not isinstance(items, (list, tuple)):
        raise MifOpError("zone.items is a list of {name, type}. NOTHING was added.")
    link = take_bool(spec, "link", default=True)
    base = take(params, "name", default=None, kind=str)
    loc = params.get("location")
    x, y = (finite_floats(list(loc)[:2], "location") if isinstance(loc, (list, tuple)) and len(loc) >= 2
            else (0.0, 0.0))
    made = []
    try:
        zin = tree.nodes.new(in_id)
        made.append(zin)
        zout = tree.nodes.new(out_id)
        made.append(zout)
        if base:
            zin.name, zout.name = base + " Input", base + " Output"
        zin.location, zout.location = (x - 150.0, y), (x + 150.0, y)
        if not zin.pair_with_output(zout):
            raise MifOpError("Blender would not pair the two %s nodes. NOTHING was added." % kind)
        added = _add_items(getattr(zout, cname), items, "NOTHING was added.")
        linked = None
        if link:
            src = next((s for s in zin.outputs if s.type == "GEOMETRY"), None)
            dst = next((s for s in zout.inputs if s.type == "GEOMETRY"), None)
            if src is not None and dst is not None:
                tree.links.new(src, dst)
                linked = "%s.%s -> %s.%s" % (zin.name, src.identifier, zout.name, dst.identifier)
    except Exception:
        for n in made:
            try:
                tree.nodes.remove(n)
            except (RuntimeError, ReferenceError):
                pass
        raise
    return {"group": tree.name, "zone": kind, "inputNode": zin.name, "outputNode": zout.name,
            "paired": getattr(zin, "paired_output", None) is not None and zin.paired_output.name == zout.name,
            "itemCollection": cname, "itemsAdded": added, "items": _item_rows(zout, cname), "link": linked,
            "inputSockets": [s.identifier for s in zin.inputs], "outputSockets": [s.identifier for s in zout.outputs]}


# ===========================================================================
# A NODE'S ITEM COLLECTIONS - set_node_property items{} (2026-10-04)
# ===========================================================================
# MEASURED 2026-10-04 on 3.6.23, 4.4.0 and 5.2.1 (scratchpad probe of every node COLLECTION with a new()):
#   * zone and geometry items take new(socket_type, name): a Simulation Output's state_items (3.6 on), a Repeat Output's
#     repeat_items (4.2 on), Capture Attribute's capture_items (4.2 on; 3.6 has a data_type instead), Bake's bake_items;
#     a Menu Switch's enum_items take new(name), an Index Switch's index_switch_items new() with no name at all;
#   * a File Output node holds file_slots and layer_slots on 3.6.23 and 4.4.0 - new(name), the file slot's name is its
#     `path`, its own format used only with use_node_format off - and ONE file_output_items collection on 5.2.1 -
#     new(socket_type, name), its format used with override_node_format on; the folder is base_path before 5.0 and
#     directory (with file_name beside it) on 5.2.1.
_NOT_ITEMS = ("inputs", "outputs", "internal_links", "panel_states", "rna_type")


def _item_collections(node):
    """{collection name: the collection} of every item collection a node (or its zone's output) has."""
    owner = _items_owner(node)
    out = {}
    for p in owner.bl_rna.properties:
        if p.type != "COLLECTION" or p.identifier in _NOT_ITEMS:
            continue
        srna = getattr(p, "srna", None)
        if srna is not None and "new" in [f.identifier for f in srna.functions]:
            out[p.identifier] = getattr(owner, p.identifier)
    return owner, out


def _drop_item(node, coll, item):
    """Remove one item. A File Output slot collection's remove() takes the node's input SOCKET (3.6.23, 4.4.0), which
    sits at the slot's own index; every other collection takes the item."""
    fn = coll.bl_rna.functions["remove"]
    prm = next(iter(fn.parameters), None)
    if prm is not None and getattr(prm, "fixed_type", None) is not None and prm.fixed_type.identifier == "NodeSocket":
        idx = next(k for k, it in enumerate(coll) if it == item)
        coll.remove(node.inputs[idx])
    else:
        coll.remove(item)


def _new_params(coll):
    fn = coll.bl_rna.functions["new"]
    return [p.identifier for p in fn.parameters if p.identifier not in ("item", "socket")]


def _file_output_rows(node, cname):
    rows = []
    coll = getattr(node, cname)
    for it in coll:
        nm = getattr(it, "name", None) if "name" in it.bl_rna.properties else getattr(it, "path", None)
        own = (not it.use_node_format) if hasattr(it, "use_node_format") else bool(getattr(it, "override_node_format",
                                                                                           False))
        fmt = getattr(it, "format", None)
        rows.append({"name": nm, "type": getattr(it, "socket_type", None),
                     "socket": next((s.identifier for s in node.inputs if s.name == nm), None),
                     "ownFormat": own if fmt is not None else None,
                     "format": ({"file_format": fmt.file_format, "color_depth": fmt.color_depth,
                                 "exr_codec": getattr(fmt, "exr_codec", None)} if fmt is not None and own else None)})
    return rows


def _items_rows(node, owner, cname):
    if node.bl_idname == "CompositorNodeOutputFile":
        return _file_output_rows(node, cname)
    return _item_rows(owner, cname)


def _plan_items(node, spec, tail):
    """items{} checked against the node's own collection, before anything is written -> a plan dict."""
    if not isinstance(spec, dict) or not spec or set(spec) - {"collection", "add", "rename", "remove"}:
        raise MifOpError("items is {collection, add [{name, type, format}], rename {old: new}, remove [names]}; got "
                         "%r. %s" % (spec, tail))
    owner, colls = _item_collections(node)
    if not colls:
        raise MifOpError("node '%s' (%s) has no item collection to edit. %s" % (node.name, node.bl_idname, tail))
    cname = spec.get("collection")
    why = "named"
    if cname is None:
        if len(colls) == 1:
            cname, why = next(iter(colls)), "its only item collection"
        elif set(colls) == {"file_slots", "layer_slots"}:
            multi = node.format.file_format == "OPEN_EXR_MULTILAYER"
            cname = "layer_slots" if multi else "file_slots"
            why = "the node's format is %s" % node.format.file_format
        else:
            raise MifOpError("node '%s' has several item collections (%s) - pass collection. %s"
                             % (node.name, ", ".join(sorted(colls)), tail))
    if cname not in colls:
        raise MifOpError("node '%s' has no item collection %r. It has: %s. %s"
                         % (node.name, cname, ", ".join(sorted(colls)), tail))
    coll = colls[cname]
    accepts = _new_params(coll)
    types = _item_types(coll)
    # THE ELEMENT'S OWN TYPE, off the collection property: a File Output slot collection's new() returns the input
    # SOCKET it made, not the slot (3.6.23, 4.4.0), so new()'s return type does not describe the item
    elem_props = owner.bl_rna.properties[cname].fixed_type.properties
    name_attr = "name" if "name" in elem_props else "path"
    present = [getattr(it, name_attr, None) for it in coll]
    adds = spec.get("add") or []
    if not isinstance(adds, (list, tuple)):
        raise MifOpError("items.add is a list of {name, type, format}. %s" % tail)
    plan_add = []
    names_now = list(present)
    for it in adds:
        if not isinstance(it, dict) or set(it) - {"name", "type", "format"}:
            raise MifOpError("an added item is {name, type, format}; got %r. %s" % (it, tail))
        name = it.get("name")
        if "name" in accepts:
            if not isinstance(name, str) or not name.strip():
                raise MifOpError("an item of %s needs a name. %s" % (cname, tail))
            if name in names_now:
                raise MifOpError("%s already has an item named '%s' - Blender would add '%s.001' beside it. %s"
                                 % (cname, name, name, tail))
            names_now.append(name)
        elif name is not None:
            raise MifOpError("%s items take no name on this build (new() has none). %s" % (cname, tail))
        t = it.get("type")
        if types is not None:
            if t is None:
                raise MifOpError("an item of %s needs a type, one of this Blender's %s. %s"
                                 % (cname, ", ".join(types), tail))
            t = str(t).upper()
            if t not in types:
                raise MifOpError("item type %r is not one of this Blender's %s. %s" % (it.get("type"),
                                                                                    ", ".join(types), tail))
        elif t is not None:
            raise MifOpError("%s items take no type on this build. %s" % (cname, tail))
        fmt = it.get("format")
        if fmt is not None:
            if elem_props is None or "format" not in elem_props:
                raise MifOpError("%s items have no format of their own. %s" % (cname, tail))
            if not isinstance(fmt, dict) or not fmt:
                raise MifOpError("format is {identifier: value} of the image settings, got %r. %s" % (fmt, tail))
            known = {q.identifier for q in bpy.types.ImageFormatSettings.bl_rna.properties if not q.is_readonly}
            bad = sorted(k for k in fmt if k not in known)
            if bad:
                raise MifOpError("format has no setting %s. It has: %s. %s" % (", ".join(bad), ", ".join(sorted(known)),
                                                                            tail))
        plan_add.append((name, t, fmt))
    rename = spec.get("rename") or {}
    if not isinstance(rename, dict):
        raise MifOpError("items.rename is {old: new}. %s" % tail)
    for old, new in rename.items():
        if old not in present:
            raise MifOpError("%s has no item '%s' to rename. It has: %s. %s" % (cname, old, ", ".join(map(str, present)),
                                                                              tail))
        if not isinstance(new, str) or not new.strip() or new in names_now:
            raise MifOpError("'%s' cannot become %r: a name must be new and not empty. %s" % (old, new, tail))
        names_now.append(new)
    remove = spec.get("remove") or []
    if not isinstance(remove, (list, tuple)):
        raise MifOpError("items.remove is a list of names. %s" % tail)
    for nm in remove:
        if nm not in present:
            raise MifOpError("%s has no item '%s' to remove. It has: %s. %s" % (cname, nm, ", ".join(map(str, present)),
                                                                             tail))
        if nm in rename:
            raise MifOpError("'%s' is both renamed and removed in one call. %s" % (nm, tail))
    if not (plan_add or rename or remove):
        raise MifOpError("items names nothing to add, rename or remove. %s" % tail)
    return {"owner": owner, "cname": cname, "why": why, "coll": coll, "accepts": accepts, "nameAttr": name_attr,
            "add": plan_add, "rename": dict(rename), "remove": list(remove)}


def _apply_items(node, plan, undo, tail):
    """Adds and renames now (undoable: the undo list takes a closure per step); removals come later."""
    coll, attr = plan["coll"], plan["nameAttr"]
    made = []
    for name, t, fmt in plan["add"]:
        args = []
        if "socket_type" in plan["accepts"]:
            args.append(t)
        if "name" in plan["accepts"]:
            args.append(name)
        coll.new(*args)
        item = coll[len(coll) - 1]           # the slot itself, whatever new() returned (see _plan_items)
        made.append(item)
        undo.append(lambda c=coll, i=item, n=node: _drop_item(n, c, i))
        if fmt:
            if hasattr(item, "use_node_format"):
                item.use_node_format = False
            elif hasattr(item, "override_node_format"):
                item.override_node_format = True
            for k, v in fmt.items():
                write_rna_property(item.format, k, v, tail, "the item's format", "ImageFormatSettings", frozenset())
    for old, new in plan["rename"].items():
        item = next(i for i in coll if getattr(i, attr) == old)
        setattr(item, attr, new)
        undo.append(lambda i=item, a=attr, o=old: setattr(i, a, o))
    return len(made)


def _remove_items(node, plan):
    coll, attr = plan["coll"], plan["nameAttr"]
    for nm in plan["remove"]:
        item = next(i for i in coll if getattr(i, attr) == nm)
        _drop_item(node, coll, item)
    return len(plan["remove"])


def op_add_group_node(params):
    """Add a node to a group and optionally set its input defaults.

    `inputs` is {socket: value} and is applied AFTER the node exists, so a key that does not match is
    refused with the sockets the node really has - a value written to a socket that is not there would
    otherwise vanish without a word. A key is a socket NAME, else its unique IDENTIFIER ("Value_001"
    for a Math node's second operand, which a name cannot reach), else an index; the response's
    inputIdentifiers lists them. ANY refusal removes the node again - NOTHING is left behind.
    """
    reject_unknown(params, _ADDNODE_KEYS, "add_group_node")
    tree = _tree(take(params, "group", "tree", required=True, kind=str))
    if params.get("zone") is not None:
        return _add_zone(tree, params)
    asked = str(take(params, "type", "nodeType", required=True, kind=str))
    # RESOLVED AND VERSION-CHECKED BEFORE ANYTHING IS CREATED. A friendly name becomes a bl_idname,
    # a raw bl_idname passes through, and a type this build does not register is refused naming the
    # version - see resolve_node_type.
    ntype = resolve_node_type(asked)
    try:
        node = tree.nodes.new(ntype)
    except RuntimeError as exc:
        raise MifOpError("Blender does not know a node type '%s' in a %s: %s. NOTHING was added."
                         % (ntype, tree.bl_idname, exc))
    # AN UNDEFINED NODE IS NOT AN ERROR IN BLENDER, and it is the shape a removed node type takes
    # in a build that still knows the name for file compatibility: nodes.new() returns something,
    # it sits in the tree, it has no sockets and it evaluates to nothing. Reading the idname back
    # is the only way to tell that apart from a working node, and an inert node reported as added
    # is the silent-wrong-result this addon exists to refuse.
    if node.bl_idname != ntype:
        made = node.bl_idname
        tree.nodes.remove(node)
        raise MifOpError(
            "Blender %s accepted '%s' and produced a '%s' instead - an UNDEFINED node, which is "
            "what a type that has been removed but is still recognised by name becomes. It has no "
            "sockets and evaluates to nothing. It was removed again. NOTHING was added."
            % (bpy.app.version_string, ntype, made))

    # EVERY REFUSAL FROM HERE ON TAKES THE NODE BACK OUT (2026-09-16). Until now five of them said "The
    # node WAS added as 'X'" and left X in the tree - a bad input name, an inactive socket, a value the
    # socket would not take, nodeGroup on a node that is not a Group, an enum knob the node lacks - and
    # test_blender_node_bake_params' B1375 pinned two of those as deliberate: "the type resolved, the node
    # exists, and only the extra parameter could not be honoured". The second seat, which wrote that row,
    # reported the same shape as a defect on a bad input name: it finished a probing session with Math.001,
    # Math.002 and Math.003 littering its group, one per refusal. A refused call that leaves work behind is
    # the thing every other op in this addon promises not to do ("NOTHING was added"), and the non-dict
    # `inputs` case already moved to removal on 2026-09-15. So the rest follow: the body runs in
    # _configure_added_node, and anything it raises removes the node before the refusal leaves.
    try:
        return _configure_added_node(tree, node, ntype, asked, params)
    except Exception:
        try:
            tree.nodes.remove(node)
        except (RuntimeError, ReferenceError):
            pass      # already removed by the refusal itself (the nodeGroup paths do that)
        raise


_NODE_GONE = "The node this call made was removed again - NOTHING was added."


def _input_socket(node, key, tail=_NODE_GONE):
    """(socket, note) for an `inputs` key: by NAME, then by Blender's unique socket IDENTIFIER, then by
    index - or (None, None) when nothing matches. Raises MifOpError for a name that matches only
    inactive sockets, ending with the CALLER's tail: add_group_node has removed its node by then,
    set_node_property has changed nothing, and one sentence cannot be true of both.

    WHY THE IDENTIFIER. A Math node has THREE inputs all named "Value", and a name reaches only the
    first live one: {"Value": 4.0} set operand one and nothing could set operand two - Value_001,
    Value.001 and "Value 2" were all refused (the second seat, 2026-09-16, building nodemath). Blender
    gives every socket a per-node unique `identifier` ("Value", "Value_001", "Value_002"), which is
    what this falls back to; a digit is an index, the same fallback link_group_nodes already had.
    A NAME that matches more than one live socket still writes the first, as before, and the note says
    which identifiers the others answer to - so the first operand keeps working for every existing
    caller and the second is one lookup away.
    """
    live, inactive = usable_sockets(node.inputs, key)
    if live:
        note = None
        if len(live) > 1:
            note = ("'%s' names %d live inputs on '%s'; the first (identifier '%s') was written. The "
                    "others answer to their identifiers: %s."
                    % (key, len(live), node.name, live[0].identifier,
                       ", ".join(s.identifier for s in live[1:])))
        return live[0], note
    if inactive:
        raise MifOpError(
            "node '%s' (%s) has an input called '%s', but it is INACTIVE at this node's current "
            "settings - Blender carries one socket per data type or per operation under the same name "
            "and only reads the active one, so a value written here would be stored and never used. Set "
            "the property that selects it (data_type, operation, dimensions...) through `properties` IN "
            "THIS SAME CALL, which is applied before inputs. %s"
            % (node.name, node.bl_idname, key, tail))
    by_id = [s for s in node.inputs if s.identifier == key]
    if by_id:
        if not getattr(by_id[0], "enabled", True):
            raise MifOpError("input '%s' of '%s' is INACTIVE at this node's current settings, so a value "
                             "written to it would never be read. %s" % (key, node.name, tail))
        return by_id[0], None
    if key.isdigit() and 0 <= int(key) < len(node.inputs):
        sock = node.inputs[int(key)]
        if not getattr(sock, "enabled", True):
            raise MifOpError("input %s ('%s') of '%s' is INACTIVE at this node's current settings, so a "
                             "value written to it would never be read. %s"
                             % (key, sock.name, node.name, tail))
        return sock, None
    return None, None


def _coerce_input(node, sock, key, val, tail):
    """What `inputs[key] = val` would store in `sock`, validated and converted. WRITES NOTHING.

    ONE CONVERSION FOR BOTH WRITERS. add_group_node set a list element by element with a bare
    float(), and that path had three defects the scalar path beside it did not - measured on 5.0.1,
    2026-09-20, through the op:
      * {"Color": [NaN, 0, 0, 1]} came back ok:true, and the socket read back nan - finite_floats
        guarded every other vector this addon parses, and this one loop bypassed it;
      * a list LONGER than the socket - [1, 0, 0, 1, 7] into an RGBA - died as a raw
        "IndexError: bpy_prop_array[index] = value: index out of range" with a traceback;
      * a list SHORTER than the socket - {"Scale": [3.0]} into a Mapping node - wrote x alone,
        left [3.0, 1.0, 1.0], and echoed [3.0] back as applied.
    So a list is converted whole: finite numbers only, and exactly as many as the socket holds.
    The one short form accepted is [r, g, b] into a COLOUR socket, which keeps the alpha it has -
    what the old element-wise write did, what ops_material._coerce_socket already accepts, and the
    only shortening every caller in this repo actually sends.

    Built BEFORE anything is written, so set_node_property can check every input in a call first.
    """
    try:
        if not hasattr(sock, "default_value"):
            # A Shader, Geometry or Matrix input carries a CONNECTION, not a value - there is nothing
            # to write, and the old path found that out as an AttributeError on assignment.
            raise MifOpError("input '%s' of '%s' is a %s socket, which carries a LINK and holds no value. "
                             "Wire something into it with link_group_nodes."
                             % (key, node.name, sock.bl_idname))
        if isinstance(val, (list, tuple)):
            cur = getattr(sock, "default_value", None)
            if cur is None or isinstance(cur, str) or not hasattr(cur, "__len__"):
                raise MifOpError("input '%s' of '%s' (%s) holds a single value, and got a list %r."
                                 % (key, node.name, sock.bl_idname, list(val)))
            nums = list(_socket_value(sock.bl_idname, key, list(val), "'%s'" % node.name, tail))
            width = len(cur)
            if len(nums) == 3 and width == 4 and sock.bl_idname.startswith("NodeSocketColor"):
                nums.append(float(cur[3]))
            if len(nums) != width:
                raise MifOpError(
                    "input '%s' of '%s' (%s) has %d components, and got %d: %r. A shorter list is not "
                    "padded - the old writer kept the socket's other components and reported the list as "
                    "applied, which is how [3] meant as a uniform scale became [3, 1, 1]. Give all %d%s."
                    % (key, node.name, sock.bl_idname, width, len(nums), list(val), width,
                       " (or [r, g, b], which keeps the alpha)" if width == 4
                       and sock.bl_idname.startswith("NodeSocketColor") else ""))
            return tuple(nums)
        if isinstance(val, bool):
            return val
        return _socket_value(sock.bl_idname, key, val, "'%s'" % node.name, tail)
    except MifOpError as exc:
        # THE CALLER'S TAIL ON EVERY REFUSAL. finite_float and finite_floats word their own messages
        # and end without it, so a NaN refused here said nothing about the node the call had made or
        # the one it had left alone. Appended once, only where it is missing.
        text = str(exc)
        raise MifOpError(text if (not tail or text.endswith(tail)) else "%s %s" % (text, tail))


def _put_in_frame(tree, node, parent_name, tail):
    """Put `node` inside the Frame named `parent_name` - add_group_node's `parent` and group_nodes' FRAME, one rule."""
    frame = tree.nodes.get(parent_name)
    if frame is None or frame.bl_idname != "NodeFrame":
        frames = sorted(n.name for n in tree.nodes if n.bl_idname == "NodeFrame")
        raise MifOpError("parent must be a Frame node of '%s'; %r is %s. Frames here: %s. %s"
                         % (tree.name, parent_name, "not a node" if frame is None else "a " + frame.bl_idname,
                            ", ".join(frames) or "(none)", tail))
    node.parent = frame
    if node.parent is None or node.parent.name != frame.name:
        raise MifOpError("Blender did not keep '%s' inside frame '%s'. %s" % (node.name, frame.name, tail))


def _configure_added_node(tree, node, ntype, asked, params):
    """Everything add_group_node does to a node after it exists. Any exception here is the caller's
    cue to remove the node again - see op_add_group_node."""
    nm = take(params, "name", default=None, kind=str)
    if nm:
        node.name = nm
        node.label = nm
    lbl = take(params, "label", default=None, kind=str)
    if lbl:
        node.label = lbl
    loc = params.get("location")
    if isinstance(loc, dict):
        # TWO AXES HERE, not three - a node location is 2D. Same defect as the vector parsers:
        # {"mif":"typo"} put the node at the origin and reported success.
        check_axis_dict(loc, "location", ("x", "y"), _NODE_GONE)
        xy = [loc.get("x", 0.0), loc.get("y", 0.0)]
    elif isinstance(loc, (list, tuple)) and len(loc) >= 2:
        xy = list(loc)[:2]
    else:
        xy = None
    if xy is not None:
        # finite_floats: a bare float() put the node at x = nan under ok:true for [NaN, 0] and {"x": NaN},
        # raised a raw ValueError for ["abc", 0], and let Infinity through for Blender to clamp to 1e6 in
        # silence (5.0.1, 2026-09-21). A refusal here removes the node, as everything this function raises does.
        try:
            node.location = tuple(finite_floats(xy, "location"))
        except MifOpError as exc:
            raise MifOpError("%s %s" % (exc, _NODE_GONE))

    # INSIDE A FRAME (2026-10-04): the frame node can be made and labelled, and nothing put a node in it. Layout only.
    parent_name = take(params, "parent", default=None, kind=str)
    if parent_name is not None:
        _put_in_frame(tree, node, parent_name, _NODE_GONE)

    # Enum-ish knobs that live on the node rather than on a socket. Written only when present, and
    # refused by name when the node has no such property.
    for key in ("operation", "dataType", "domain", "mode"):
        if key not in params:
            continue
        attr = {"operation": "operation", "dataType": "data_type",
                "domain": "domain", "mode": "mode"}[key]
        if not hasattr(node, attr):
            raise MifOpError("node type '%s' has no '%s'. %s" % (ntype, key, _NODE_GONE))
        setattr(node, attr, str(params[key]).upper())

    # THE CONTAINED TREE, and it has to be set before `inputs` is read: a Group node has NO sockets
    # at all until it points at something, so an inputs dict would refuse every key as unknown.
    want_group = take(params, "nodeGroup", default=None, kind=str)
    if want_group is not None:
        if ntype not in _GROUP_NODE_TYPES:
            raise MifOpError(
                "'nodeGroup' only applies to a Group node (%s), and this is a %s. Accepting it here "
                "and writing nothing is how a caller believes they nested a group they did not. %s"
                % (", ".join(_GROUP_NODE_TYPES), ntype, _NODE_GONE))
        inner = bpy.data.node_groups.get(str(want_group))
        if inner is None:
            have = sorted(g.name for g in bpy.data.node_groups)[:25]
            tree.nodes.remove(node)
            raise MifOpError("no node group named '%s'. This file has: %s. NOTHING was added."
                             % (want_group, ", ".join(have) if have else "(none)"))
        if inner.bl_idname != tree.bl_idname:
            tree.nodes.remove(node)
            raise MifOpError(
                "'%s' is a %s and '%s' is a %s - a group node can only contain a tree of its own "
                "kind. NOTHING was added." % (inner.name, inner.bl_idname, tree.name, tree.bl_idname))
        node.node_tree = inner
        # READ BACK, because Blender rejects a RECURSIVE assignment SILENTLY. Measured on 3.6.23,
        # 4.4.0 and 5.0.1: pointing a group node at its own tree, directly or through a second
        # group, leaves node_tree at None and raises nothing at all. An inert Group node is
        # indistinguishable from a working one in the outliner, so this is refused rather than
        # returned as a success with an empty socket list.
        if node.node_tree is not inner:
            tree.nodes.remove(node)
            raise MifOpError(
                "Blender would not put '%s' inside '%s' and reported no error - node_tree stayed "
                "empty. That happens when it would be RECURSIVE: a group cannot contain itself, "
                "directly or through another group that contains it. The node was removed rather "
                "than left holding nothing. NOTHING was added." % (inner.name, tree.name))

    # ANY OTHER NODE PROPERTY, by its real RNA name. The four keys above are kept because they are
    # already in use and are friendlier spellings; this is the general form behind them.
    props_applied = {}
    given_props = params.get("properties")
    if given_props is not None:
        if not isinstance(given_props, dict):
            raise MifOpError("'properties' must be a {name: value} object, got %r. %s"
                             % (given_props, _NODE_GONE))
        for key, value in given_props.items():
            props_applied[key] = _write_node_property(node, str(key), value, _NODE_GONE)

    applied, written, notes = {}, {}, []
    given = params.get("inputs")
    if given is not None:
        if not isinstance(given, dict):
            # THE NODE GOES, AND THE REFUSAL SAYS SO - the shape add_procedural_texture in this
            # same file already uses ("Every node this call made was removed again - NOTHING was
            # changed"). Until 2026-09-15 a refused `inputs` left the node in the tree and
            # list_group_nodes went 0 -> 1 after a call that reported failure. Reported by the
            # second seat as defect O's second instance.
            #
            # REMOVED rather than kept, unlike assign_node_group above: there the modifier is the
            # thing the caller asked for and only the socket values failed, so it stays. Here the
            # node has no value of its own without the inputs that were being set on it.
            tree.nodes.remove(node)
            raise MifOpError("'inputs' must be a {socketName: value} object, got %r. The node this "
                             "call made was removed again - NOTHING was added." % (given,))
        have = ["%s [%s]" % (socket_label(s), s.identifier) for s in node.inputs]
        for sock_name, val in given.items():
            # ENABLED FIRST, and a name that ONLY matches a disabled socket is refused. See
            # usable_sockets: a Mix node carries four A sockets and four B sockets, one pair per
            # data type, all identically named, and the inactive ones are written and never read.
            # This runs AFTER `properties`, deliberately - setting data_type or operation in the
            # same call is what decides which socket is the live one, so the order is what makes
            # {"properties": {"data_type": "RGBA"}, "inputs": {"A": [1,0,0,1]}} land correctly.
            # Then the unique identifier, then an index - see _input_socket.
            sock, note = _input_socket(node, str(sock_name))
            if sock is None:
                raise MifOpError("node '%s' has no input socket '%s' - not as a name, a socket "
                                 "identifier or an index. It has (name [identifier]): %s. %s"
                                 % (node.name, sock_name, ", ".join(have), _NODE_GONE))
            if note:
                notes.append(note)
            # ONE CONVERSION, shared with set_node_property - see _coerce_input for the three defects
            # the element-by-element write here had (NaN stored, a long list a raw IndexError, a short
            # one silently partial).
            value = _coerce_input(node, sock, str(sock_name), val, _NODE_GONE)
            try:
                sock.default_value = value
            except (AttributeError, TypeError, ValueError) as exc:
                raise MifOpError("cannot write %r to socket '%s' of '%s': %s. %s"
                                 % (val, sock_name, node.name, exc, _NODE_GONE))
            applied[sock_name] = val
            written[sock_name] = sock.identifier

    return {"group": tree.name, "node": node.name, "type": ntype,
            # WHAT WAS ASKED FOR, when it was a friendly name rather than a bl_idname. A caller
            # that sent "colorRamp" gets back the ShaderNodeValToRGB it actually has, which is the
            # name every other op in this addon wants for it.
            "requestedType": asked if asked != ntype else None,
            # WHAT THIS NODE CONTAINS, if anything. A Group node with None here has no sockets and
            # does nothing, which is the state this parameter exists to make reachable and visible.
            "containsGroup": (node.node_tree.name
                              if getattr(node, "node_tree", None) is not None else None),
            "location": [round(node.location[0], 3), round(node.location[1], 3)],
            "parent": node.parent.name if node.parent is not None else None,
            # INACTIVE SOCKETS ARE MARKED RATHER THAN HIDDEN. A bare name list of a Mix node reads
            # as "Factor, A, B, Factor, A, B, ..." with no way to tell which three are live, and
            # picking the wrong one writes a value nothing reads.
            "inputs": [socket_label(s) for s in node.inputs],
            "outputs": [socket_label(s) for s in node.outputs],
            "activeInputs": [s.name for s in node.inputs if getattr(s, "enabled", True)],
            # THE UNIQUE NAMES, index-aligned with `inputs`: the only way to tell three "Value"s apart,
            # and the spelling `inputs` accepts for the ones a name cannot reach.
            "inputIdentifiers": [s.identifier for s in node.inputs],
            "inputsApplied": applied,
            # WHICH SOCKET EACH KEY LANDED ON, by identifier - "Value" and "Value_001" are different
            # operands, and an echo of the key alone could not say which one a name reached.
            "inputsWrittenTo": written,
            **({"notes": notes} if notes else {}),
            # READ BACK OFF THE NODE, not echoed - an enum matched case-insensitively is reported
            # as the identifier Blender stored, and a clamped number as the value it kept.
            "propertiesApplied": props_applied,
            "nodeCount": len(tree.nodes)}


def _find_socket(node, name, collection):
    # THE ENABLED ONE WINS AMONG IDENTICAL NAMES. A Mix node has an A and a B per data type, all
    # spelled the same, and only one pair is live; taking exact[0] wired the float pair whatever
    # the node was set to. See usable_sockets for the measurement behind this.
    live, inactive = usable_sockets(collection, name)
    if live:
        return live[0]
    if inactive:
        # RETURNED, NOT REFUSED, and the difference from add_group_node's `inputs` is deliberate.
        # A default value written to an inactive socket is read by nothing and can never become
        # live without being rewritten, so it is a silent no-op. A LINK is different: it stays in
        # the tree and starts carrying signal the moment the node's data type changes. So it is
        # allowed and REPORTED - see toSocketInactive on the way out.
        return inactive[0]
    # THE UNIQUE IDENTIFIER before the index, the same order add_group_node's inputs resolve in
    # (_input_socket): a name reaches only the first of a Math node's three "Value" sockets, and
    # "Value_001" is the spelling Blender itself gives the second.
    by_id = [s for s in collection if s.identifier == name]
    if by_id:
        return by_id[0]
    if name.isdigit():
        idx = int(name)
        if 0 <= idx < len(collection):
            return collection[idx]
    return None


def _principled_fallback(node, name):
    """A Principled INPUT named with ANOTHER version's spelling, resolved to this one.

    Blender renamed these sockets at 4.0 - "Emission" became "Emission Color", "Specular" became
    "Specular IOR Level" - and link_group_nodes matches socket names literally, so a graph that
    wires an emission mask is a different call on 3.6 than on 4.4 and a caller has to branch on
    the version to build one tree. ops_material already owns that alias table and resolves every
    Principled WRITE through it; this is the same table reaching the LINK path so the two halves
    of the material surface agree about what a socket is called.

    THE IMPORT IS DEFERRED ON PURPOSE. ops_material imports this module at module level, for the
    tree resolver; importing it back at module level here would be a cycle. Inside the function
    both modules are long since loaded, and the direction of the module-level dependency stays
    one way, which is what keeps it readable.
    """
    # CALLED ONLY FROM THE toSocket PATH, rather than deciding here whether `collection` is the
    # node's inputs: bpy hands back a NEW proxy object on every attribute access, so
    # `collection is node.inputs` is False for the very collection it was read from. That exact
    # identity mistake already cost this file a false "linked: false" - see op_link_group_nodes.
    if node.type != "BSDF_PRINCIPLED":
        return None
    from .ops_material import principled_socket   # noqa: PLC0415 - see the docstring
    return principled_socket(node, name)


def _link_row(link):
    return {"fromNode": link.from_node.name, "fromSocket": link.from_socket.name,
            "fromIdentifier": link.from_socket.identifier,
            "toNode": link.to_node.name, "toSocket": link.to_socket.name,
            "toIdentifier": link.to_socket.identifier}


def _unlink(tree, params):
    """link_group_nodes {remove:true}: take links OUT, and name every one that went.

    THE COUNTERPART NOTHING HAD. A link could be made and never unmade except by deleting a node -
    remove_node's severedLinks - so taking a texture off a Roughness input to go back to a constant
    meant destroying the texture node and everything else it fed. Scoped by the INPUT end, the end a
    link belongs to (an input takes one link unless it is multi-input; an output feeds many): toNode,
    plus toSocket or fromNode or both, narrowed further by fromSocket. A scope that matches nothing is
    a refusal, never ok:true with nothing removed.
    """
    tn_name = str(take(params, "toNode", required=True, kind=str))
    tn = tree.nodes.get(tn_name)
    if tn is None:
        raise MifOpError("no node named '%s' in '%s'. It has: %s. NOTHING was removed."
                         % (tn_name, tree.name, ", ".join(n.name for n in tree.nodes)))
    ts_name = str(take(params, "toSocket", default="", kind=str))
    fn_name = str(take(params, "fromNode", default="", kind=str))
    fs_name = str(take(params, "fromSocket", default="", kind=str))
    if not ts_name and not fn_name:
        raise MifOpError("remove:true needs toSocket, fromNode or both - removing every link into '%s' "
                         "is too wide to be what one call meant. NOTHING was removed." % tn.name)
    ts = None
    if ts_name:
        ts = _find_socket(tn, ts_name, tn.inputs) or _principled_fallback(tn, ts_name)
        if ts is None:
            raise MifOpError("node '%s' has no input socket '%s'. It has: %s. NOTHING was removed."
                             % (tn.name, ts_name, ", ".join(socket_label(s) for s in tn.inputs)))
    fn = fs = None
    if fn_name:
        fn = tree.nodes.get(fn_name)
        if fn is None:
            raise MifOpError("no node named '%s' in '%s'. It has: %s. NOTHING was removed."
                             % (fn_name, tree.name, ", ".join(n.name for n in tree.nodes)))
        if fs_name:
            fs = _find_socket(fn, fs_name, fn.outputs)
            if fs is None:
                raise MifOpError("node '%s' has no output socket '%s'. It has: %s. NOTHING was removed."
                                 % (fn.name, fs_name, ", ".join(socket_label(s) for s in fn.outputs)))
    elif fs_name:
        raise MifOpError("fromSocket needs fromNode - a socket name alone does not say which node's. "
                         "NOTHING was removed.")

    # MATCHED BY NAME AND IDENTIFIER, never by identity: bpy hands back a fresh proxy on every access,
    # so `link.to_node is tn` is False for the same node (see op_remove_node).
    doomed = [l for l in tree.links
              if l.to_node.name == tn.name
              and (ts is None or l.to_socket.identifier == ts.identifier)
              and (fn is None or l.from_node.name == fn.name)
              and (fs is None or l.from_socket.identifier == fs.identifier)]
    if not doomed:
        into = [_link_row(l) for l in tree.links if l.to_node.name == tn.name]
        raise MifOpError("no link matches that scope, so nothing is removed and this is not a success. "
                         "Links into '%s': %s. NOTHING was removed."
                         % (tn.name, "; ".join("%s.%s -> %s" % (r["fromNode"], r["fromSocket"], r["toSocket"])
                                                for r in into) or "none"))
    rows = [_link_row(l) for l in doomed]
    before = len(tree.links)
    for l in doomed:
        tree.links.remove(l)
    return {"group": tree.name, "removed": True, "removedLinks": rows, "removedCount": len(rows),
            "linkCountBefore": before, "linkCountAfter": len(tree.links),
            "note": ("each removed link is listed in full, so link_group_nodes can put any of them back. An "
                     "input left with no link reads its own value again - list_group_nodes{values:true} "
                     "shows what that is.")}


def op_link_group_nodes(params):
    """Wire one node's output into another's input, and read the link back - or, with remove:true, take
    links out (see _unlink).

    A NEW LINK INTO AN INPUT THAT ALREADY HAS ONE REPLACES IT, AND BLENDER SAYS NOTHING. Measured on
    5.0.1, 2026-09-20: V1.Value -> Roughness, then V2.Value -> Roughness, left one link (V2's), with
    linkCountBefore and linkCountAfter both 2 and nothing else in the response - the same silent severing
    remove_node was built to report. The displaced link is now captured before links.new and returned as
    replacedLinks, which is enough to put it back. A MULTI-INPUT socket (Join Geometry) keeps both, and
    re-making a link that is already there replaces nothing.
    """
    reject_unknown(params, _LINK_KEYS, "link_group_nodes", {
        "unlink": "spelt remove:true - it takes toNode plus toSocket and/or fromNode, and lists every "
                  "link it removed.",
        "delete": "see 'unlink'.",
        "replace": "a link into an input that already has one REPLACES it - that is Blender's rule for "
                   "a single input, and the displaced link comes back in replacedLinks. There is "
                   "nothing to switch on.",
    })
    tree = _tree(take(params, "group", "tree", required=True, kind=str))
    if take_bool(params, "remove", default=False):
        return _unlink(tree, params)
    fn_name = str(take(params, "fromNode", required=True, kind=str))
    tn_name = str(take(params, "toNode", required=True, kind=str))
    fn = tree.nodes.get(fn_name)
    tn = tree.nodes.get(tn_name)
    if fn is None or tn is None:
        raise MifOpError("no node named '%s' in group '%s'. It has: %s. NOTHING was linked."
                         % (fn_name if fn is None else tn_name, tree.name,
                            ", ".join(n.name for n in tree.nodes)))
    # AN UNPAIRED ZONE NODE CARRIES NOTHING (2026-10-04): two zone halves made one by one stay apart until
    # input.pair_with_output(output) - add_group_node zone does it.
    for n in (fn, tn):
        if hasattr(n, "paired_output") and n.paired_output is None:
            raise MifOpError("'%s' is a zone input that is not paired with an output, so nothing passes through it. "
                             "Make the zone with add_group_node zone {kind}, which pairs the two. NOTHING was linked."
                             % n.name)
    fs_name = str(take(params, "fromSocket", default="", kind=str))
    ts_name = str(take(params, "toSocket", default="", kind=str))
    fs = _find_socket(fn, fs_name, fn.outputs) if fs_name else (fn.outputs[0] if fn.outputs else None)
    ts = _find_socket(tn, ts_name, tn.inputs) if ts_name else (tn.inputs[0] if tn.inputs else None)
    # THE VERSION-RENAME FALLBACK, tried only after the literal name has failed so it can never
    # shadow a socket that really is called that on this build.
    ts_alias = None
    if ts is None and ts_name:
        ts = _principled_fallback(tn, ts_name)
        if ts is not None:
            ts_alias = ts.name
    if fs is None:
        raise MifOpError("node '%s' has no output socket '%s'. It has: %s. NOTHING was linked."
                         % (fn.name, fs_name, ", ".join(socket_label(s) for s in fn.outputs)))
    if ts is None:
        raise MifOpError("node '%s' has no input socket '%s'. It has: %s. NOTHING was linked."
                         % (tn.name, ts_name, ", ".join(socket_label(s) for s in tn.inputs)))

    before = len(tree.links)
    fn_name, tn_name_final = fn.name, tn.name
    fs_final, ts_final = fs.name, ts.name
    fs_id, ts_id = fs.identifier, ts.identifier
    ts_inactive = not getattr(ts, "enabled", True)
    # WHAT THIS LINK WILL DISPLACE, captured before links.new removes it - see the docstring.
    displaced = [] if getattr(ts, "is_multi_input", False) else [
        _link_row(l) for l in tree.links
        if l.to_node.name == tn_name_final and l.to_socket.identifier == ts_id
        and not (l.from_node.name == fn_name and l.from_socket.identifier == fs_id)]
    tree.links.new(fs, ts)
    # READ BACK BY NAME, NOT BY IDENTITY. links.new returns a link object even when Blender
    # immediately drops it as invalid (mismatched socket types), so the link has to be looked for
    # rather than assumed - but `l.from_node is fn` does NOT work: bpy RNA references are proxy
    # objects recreated on access, so `is` is False even for the same underlying node. The first
    # version used identity and reported linked:false for three links that had demonstrably been
    # made - linkCount went 0 to 3 and outputReachable was true in the same breath. A false
    # NEGATIVE, which is the worse direction: it sends a caller to debug working wiring.
    # BY SOCKET IDENTIFIER, not name: a Math node's three inputs are all "Value", so a name match
    # could find a link into Value while the one asked for, into Value_001, was never made.
    made = [l for l in tree.links
            if l.from_node.name == fn_name and l.to_node.name == tn_name_final
            and l.from_socket.identifier == fs_id and l.to_socket.identifier == ts_id]
    still = {(r["fromNode"], r["fromIdentifier"]) for r in
             (_link_row(l) for l in tree.links
              if l.to_node.name == tn_name_final and l.to_socket.identifier == ts_id)}
    replaced = [r for r in displaced if (r["fromNode"], r["fromIdentifier"]) not in still]
    return {"group": tree.name,
            "from": "%s.%s" % (fn_name, fs_final),
            "to": "%s.%s" % (tn_name_final, ts_final),
            "linked": bool(made),
            # WHICH SPELLING THIS BUILD USED, when the name asked for was another version's. Null
            # when the socket really is called what the caller called it.
            "toSocketResolvedFrom": (ts_name if ts_alias else None),
            "linkCountBefore": before, "linkCountAfter": len(tree.links),
            "replacedLinks": replaced,
            "replacedNote": (
                "'%s' on '%s' takes ONE link, so this one REPLACED %s - Blender removes the old link "
                "without a word. It is listed here in full so link_group_nodes can restore it."
                % (ts_final, tn_name_final, ", ".join("%s.%s" % (r["fromNode"], r["fromSocket"])
                                                     for r in replaced))) if replaced else None,
            # A LINK INTO AN INACTIVE SOCKET IS REAL AND CARRIES NOTHING. It is stored, it survives
            # a save, and it starts working the moment the node's data type or operation selects
            # that socket - so it is not an error, and it is not something to find out about later.
            "toSocketInactive": ts_inactive or None,
            "inactiveNote": (
                "'%s' on '%s' is not the socket this node is currently reading - Blender keeps one "
                "per data type or operation under the same name. The link exists and carries "
                "nothing until the node's setting selects it."
                % (ts_final, tn_name_final)) if ts_inactive else None,
            "validNote": (None if made else
                          "links.new returned but no link is present - Blender rejected it, which "
                          "usually means the socket TYPES are incompatible. Nothing is wired.")}


def _iface_sockets(tree, in_out):
    """[(item, name)] for the group's sockets on one side, in the order the modifier panel shows them."""
    if hasattr(tree, "interface"):
        return [(it, it.name) for it in tree.interface.items_tree
                if getattr(it, "item_type", "SOCKET") == "SOCKET" and it.in_out == in_out]
    coll = tree.inputs if in_out == "INPUT" else tree.outputs
    return [(s, s.name) for s in coll]


def _iface_remove(tree, item, in_out):
    """Remove one group socket where this build keeps it: tree.interface from 4.0, tree.inputs / tree.outputs on 3.6."""
    if hasattr(tree, "interface"):
        tree.interface.remove(item)
    else:
        (tree.inputs if in_out == "INPUT" else tree.outputs).remove(item)


def _iface_links(tree, item, in_out):
    """How many links leave (INPUT) or reach (OUTPUT) this group socket inside the tree."""
    ident = getattr(item, "identifier", None)
    side = "NodeGroupInput" if in_out == "INPUT" else "NodeGroupOutput"
    n = 0
    for link in tree.links:
        sock = link.from_socket if in_out == "INPUT" else link.to_socket
        if sock.node.bl_idname == side and getattr(sock, "identifier", None) == ident:
            n += 1
    return n


def _edit_group_interface(tree, params):
    """add_group_interface rename / move / remove - every name resolved and every refusal made before the first write."""
    in_out = str(take(params, "inOut", default="INPUT", kind=str)).upper()
    if in_out not in ("INPUT", "OUTPUT"):
        raise MifOpError("inOut must be INPUT or OUTPUT, got '%s'. NOTHING was changed." % in_out)
    for key in ("name", "socketType", "default", "min", "max"):
        if params.get(key) is not None:
            raise MifOpError("%s makes a new socket; rename, move and remove edit the existing ones. Pass one or the "
                             "other. NOTHING was changed." % key)
    rename, move, remove = params.get("rename"), params.get("move"), params.get("remove")
    confirm = take_bool(params, "confirm", default=False)
    socks = _iface_sockets(tree, in_out)
    names = [n for _i, n in socks]

    def one(name, what):
        hits = [i for i, n in socks if n == name]
        if not hits:
            raise MifOpError("'%s' has no %s socket named %r to %s (it has %s). NOTHING was changed."
                             % (tree.name, in_out.lower(), name, what, ", ".join(names) or "none"))
        if len(hits) > 1:
            raise MifOpError("'%s' has %d %s sockets named %r, so a name cannot say which to %s. NOTHING was changed."
                             % (tree.name, len(hits), in_out.lower(), name, what))
        return hits[0]
    if rename is not None and (not isinstance(rename, dict) or not rename
                               or any(not isinstance(v, str) or not v for v in rename.values())):
        raise MifOpError("rename is {old: new}, every new name a non-empty string. NOTHING was changed.")
    if move is not None and (not isinstance(move, dict) or not move
                             or any(isinstance(v, bool) or not isinstance(v, int) for v in move.values())):
        raise MifOpError("move is {name: index}, the index a whole number. NOTHING was changed.")
    if remove is not None and (not isinstance(remove, (list, tuple)) or not remove
                               or any(not isinstance(v, str) for v in remove)):
        raise MifOpError("remove is a list of socket names. NOTHING was changed.")
    plan_rename = [(one(old, "rename"), old, new) for old, new in (rename or {}).items()]
    plan_move = [(one(nm, "move"), nm, idx) for nm, idx in (move or {}).items()]
    plan_remove = [(one(nm, "remove"), nm) for nm in (remove or [])]
    for _i, nm, idx in plan_move:
        if not 0 <= idx < len(socks):
            raise MifOpError("move %r to %d: the %s sockets are 0-%d. NOTHING was changed."
                             % (nm, idx, in_out.lower(), len(socks) - 1))
    linked = [(nm, _iface_links(tree, it, in_out)) for it, nm in plan_remove]
    linked = [(nm, n) for nm, n in linked if n]
    if linked and not confirm:
        raise MifOpError("removing %s would cut %s inside '%s'. Pass confirm:true to remove them anyway. NOTHING was "
                         "changed." % (", ".join(nm for nm, _n in linked),
                                       ", ".join("%d link(s) from %s" % (n, nm) for nm, n in linked), tree.name))
    before = _iface_items(tree)
    for it, _old, new in plan_rename:
        it.name = new
    for it, _nm, idx in plan_move:
        cur = [i for i, _n in _iface_sockets(tree, in_out)]
        if hasattr(tree, "interface"):
            target = cur[idx]
            pos = list(tree.interface.items_tree).index(target)
            here = list(tree.interface.items_tree).index(it)
            tree.interface.move(it, pos + 1 if pos > here else pos)
        else:
            coll = tree.inputs if in_out == "INPUT" else tree.outputs
            coll.move(cur.index(it), idx)
    for it, _nm in plan_remove:
        _iface_remove(tree, it, in_out)
    after = _iface_items(tree)
    order = [n for _i, n in _iface_sockets(tree, in_out)]
    for _it, nm, idx in plan_move:
        new_nm = dict((o, n) for _i, o, n in plan_rename).get(nm, nm)
        if order.index(new_nm) != idx:
            raise MifOpError("'%s' did not land at %d (the %s sockets read %s); the edits before it were made."
                             % (new_nm, idx, in_out.lower(), order))
    by_id_before = {r["identifier"]: r for r in before}
    rows = [{"identifier": r["identifier"], "nameBefore": by_id_before.get(r["identifier"], {}).get("name"),
             "name": r["name"], "inOut": r["inOut"]} for r in after]
    return {"group": tree.name, "inOut": in_out, "renamed": len(plan_rename), "moved": len(plan_move),
            "removed": [nm for _i, nm in plan_remove], "linksCut": dict(linked), "order": order,
            "sockets": rows, "interfaceBefore": before, "interface": after}


def op_add_group_interface(params):
    """Expose a value as a group input or output - what turns a tree into a modifier with sliders.

    EDIT THE EXISTING ONES (2026-10-04): rename {old: new}, move {name: index} and remove [names] on the inOut side
    (INPUT default). The socket keeps its identifier through a rename and a move - so a modifier's value and every
    link stay - which is why rebuilding a socket is not a substitute. A remove that would cut a link inside the tree
    is refused unless confirm is true. Each socket's identifier comes back with its name before and after.
    """
    reject_unknown(params, _IFACE_KEYS, "add_group_interface")
    tree = _tree(take(params, "group", "tree", required=True, kind=str))
    if any(params.get(k) is not None for k in ("rename", "move", "remove")):
        if params.get("defaultInput") is not None:
            raise MifOpError("defaultInput goes with a new socket, not rename, move or remove. NOTHING was changed.")
        return _edit_group_interface(tree, params)
    if params.get("confirm") is not None:
        raise MifOpError("confirm goes with remove. NOTHING was added.")
    name = str(take(params, "name", required=True, kind=str))
    stype = str(take(params, "socketType", default="NodeSocketFloat", kind=str))
    in_out = str(take(params, "inOut", default="INPUT", kind=str)).upper()
    if in_out not in ("INPUT", "OUTPUT"):
        raise MifOpError("inOut must be INPUT or OUTPUT, got '%s'. NOTHING was added." % in_out)
    try:
        sock = _iface_new(tree, name, stype, in_out)
    except (RuntimeError, TypeError) as exc:
        raise MifOpError("could not create a '%s' socket named '%s': %s. NOTHING was added."
                         % (stype, name, exc))
    # DEFAULT INPUT (2026-10-06): what an unlinked input reads - INDEX, POSITION, and from 5.2 the scene frame or the
    # self object. The choices depend on the socket type and the build, so Blender's own list is the check.
    default_input = take(params, "defaultInput", kind=str)
    if default_input is not None:
        # THE NEW SOCKET IS TAKEN BACK THE WAY THIS BUILD STORES IT - tree.interface from 4.0, tree.inputs/outputs on
        # 3.6, where the bare tree.interface.remove raised AttributeError out of the op (found 2026-10-10).
        if in_out != "INPUT":
            _iface_remove(tree, sock, in_out)
            raise MifOpError("defaultInput is for a group INPUT socket. NOTHING was added.")
        if not hasattr(sock, "default_input"):
            _iface_remove(tree, sock, in_out)
            raise MifOpError("this Blender (%s) has no default input on a group socket (its interface sockets carry no "
                             "default_input), so defaultInput %r cannot be set. NOTHING was added."
                             % (bpy.app.version_string, default_input))
        try:
            sock.default_input = default_input.upper()
        except TypeError as exc:
            _iface_remove(tree, sock, in_out)
            raise MifOpError("a '%s' input cannot take defaultInput %r on Blender %s: %s. NOTHING was added."
                             % (stype, default_input, bpy.app.version_string, str(exc).split(": ", 1)[-1]))
    # TYPE-AWARE, AND GUARDED. This used to run every one of these through take_float and then
    # setattr the result unprotected, which had two consequences measured on 4.4.0 and 5.0.1:
    #
    #   default:true  on a NodeSocketBool -> float(True) is 1.0, and writing 1.0 to a boolean RNA
    #   default:5     on a NodeSocketInt  -> float(5) is 5.0, likewise
    #
    # both raised a bare TypeError straight out of the op, ESCAPING its own MifOpError contract -
    # a raw exception where every other refusal here is a sentence. And a Vector, String or Object
    # default was refused with "must be a number", so a group interface could only ever hold a
    # FLOAT default. Reusing _socket_value gives all of them the same resolution the node sockets
    # got, including a datablock default resolved from its name.
    for key, attr in (("default", "default_value"), ("min", "min_value"), ("max", "max_value")):
        if params.get(key) is None:
            continue
        if not hasattr(sock, attr):
            raise MifOpError(
                "a '%s' socket has no %s on Blender %s, so '%s' cannot be applied - only numeric "
                "sockets carry a range. Accepting it and writing nothing is how a caller believes "
                "they set something they did not. The socket WAS created."
                % (stype, attr, bpy.app.version_string, key))
        value = (_socket_value(stype, name, params[key], "group '%s'" % tree.name)
                 if key == "default" else take_float(params, key))
        try:
            setattr(sock, attr, value)
        except (TypeError, ValueError) as exc:
            raise MifOpError("could not set %s on a '%s' socket to %r: %s. The socket WAS created."
                             % (key, stype, params[key], exc))
    return {"group": tree.name, "socket": name, "socketType": stype, "inOut": in_out,
            "defaultInput": getattr(sock, "default_input", None) if default_input is not None else None,
            # MEASURED 5.2.2: a SCENE_FRAME input fed 1 and 10 at frames 1 and 10 through a group node, and 0 as the
            # group of a Nodes modifier, whose input carries its own value.
            "defaultInputNote": ("read where the group is used as a node inside another tree; a Nodes modifier's own "
                                 "input keeps its value instead") if default_input is not None else None,
            "interface": _iface_items(tree)}


def op_set_node_property(params):
    """Change the settings of a node that already exists - the half that was write-once.

    A NODE'S SETTINGS COULD BE WRITTEN ONCE AND NEVER AGAIN. `_write_node_property` has exactly two
    callers, `add_group_node` and `add_procedural_texture`, and both are CREATION ops. So a Noise
    Texture's scale, a Math node's operation, a Mapping node's rotation were settable at the moment
    the node was made and fixed for ever after. `list_group_nodes{values:true}` could READ every one
    of them back, which makes this a plain read/write asymmetry rather than a missing feature.

    IT IS ALSO WHAT MADE THE SHOWCASE'S MATERIAL STAGE ONE-WAY. b3b_materials says so in its own
    docstring: "no typed op REMOVES a node ... so this stage can add a palette and it cannot take
    one off, and it cannot re-tune one in place. Re-running with different numbers means rebuilding
    from b1_shell". Removal is still missing and is a separate question - a node with links into it
    cannot simply vanish. RE-TUNING is not, from here.

    REUSES `_write_node_property` RATHER THAN REIMPLEMENTING IT. That function already resolves
    enums against the node's own items, coerces bools, uses finite_int/finite_float so a NaN cannot
    be stored, and handles vector properties. A second copy of that coercion would be the
    parallel-systems shape this project keeps collapsing, and it would drift in the direction that
    matters: the creation path would keep the NaN guard and this one would quietly lose it.

    ADDRESSED THE SAME WAY EVERY OTHER NODE OP IS, through `_tree`: a node group by name, or
    "scene:compositor", "scene:world", "material:<name>", "world:<name>", "light:<name>". So this reaches the
    material graphs the procedural texture ops build, which is the case it exists for.

    AND ITS INPUT SOCKETS WERE STILL WRITE-ONCE, until 2026-09-20. This op wrote PROPERTIES only and
    refused `inputs`, pointing at the creating op - so a Noise Texture's Scale, a Bump's Strength and
    Distance, a Mix Shader's Factor could be set when the node was made and never again. The ledger
    recorded what that cost (docs/22 SD45): a plaster variant "cannot re-tune" PeelBump's Strength and
    Distance, because "set_node_property cannot write - it writes PROPERTIES only", and the relief was
    left as it was for want of a call. `inputs` resolves and converts exactly as add_group_node's does
    (_input_socket, _coerce_input - one implementation, not two), is applied AFTER `properties` because
    a data_type or operation decides which of several same-named sockets is live, and refuses a LINKED
    input, whose value Blender never reads.

    ALL OR NOTHING, which it was not. Properties were written one at a time, so {"operation": "MULTIPLY",
    "no_such_prop": 1} set MULTIPLY, then refused the second key with "The node was NOT changed." -
    measured on 5.0.1, 2026-09-20, the Math node read MULTIPLY afterwards. Everything this call can
    touch is snapshotted first and put back on any refusal, so the sentence is now true.

    params:
      group / tree (str, required)   the tree, by any of the addresses above
      node (str, required)           the node's name within that tree
      properties (dict)              property -> value, validated against that node's own type
      inputs (dict)                  input socket -> value, by name, identifier ("Value_001") or index;
                                     read BACK into inputsApplied
      label (str)                    rename the node's LABEL, which is display only
      location (2-list)              move it in the editor - cosmetic, and the reason it is here is
                                     that a graph nobody can lay out is a graph nobody reads
      curve ({component, points, handles} or a list of them)   (2026-10-06) THE GAP: the settings dump skipped
                                     a CurveMapping, so an RGB Curves or Float Curve node could not be shaped or
                                     read. component C|R|G|B (RGB Curves; C is the combined curve), X|Y|Z (Vector
                                     Curves), omitted on a one-curve node; points [[x, y], ...], two or more, inside
                                     the clip range; handles AUTO | AUTO_CLAMPED | VECTOR, one or one per point. The
                                     reply's curve is read back off the mapping, and curveEvaluated samples each
                                     written curve through Blender's own CurveMapping.evaluate.
    """
    reject_unknown(params, _SETNODE_KEYS, "set_node_property", {
        "sockets": "spelt `inputs`: {socket: value}, by the socket's name, its identifier "
                   "(Value_001 for a Math node's second operand) or its index.",
        "value": "name the property and give its value - this op takes one property at a time so "
                 "the refusal can name the exact one that was wrong.",
    })
    tree = _tree(take(params, "group", "tree", required=True, kind=str))
    node_name = str(take(params, "node", required=True, kind=str))
    node = tree.nodes.get(node_name)
    if node is None:
        known = [n.name for n in tree.nodes]
        raise MifOpError("no node named '%s' in that tree. Present: %s. NOTHING was changed."
                         % (node_name, ", ".join(known[:20]) if known else "<none>"))

    props = take(params, "properties", default=None)
    inputs = take(params, "inputs", default=None)
    label = take(params, "label", default=None, kind=str)
    loc = take(params, "location", default=None)
    items = params.get("items")
    base_path = take(params, "basePath", default=None, kind=str)
    hide_unused = take_bool(params, "hideUnusedSockets", default=False) if "hideUnusedSockets" in params else None
    curve_spec = params.get("curve")
    if (not props and not inputs and label is None and loc is None and items is None and base_path is None
            and hide_unused is None and curve_spec is None):
        # A CALL THAT CHANGES NOTHING MUST NOT REPORT SUCCESS, which is the same refusal
        # set_constraint makes and for the same reason: ok:true with no effect is indistinguishable
        # from ok:true with one.
        raise MifOpError("nothing to change: pass properties, inputs, items, basePath, curve, label or location. "
                         "NOTHING was changed.")
    base_attr = None
    if base_path is not None:
        # THE FILE OUTPUT'S FOLDER: base_path before 5.0, directory (file_name is its own property) on 5.2.1
        base_attr = next((a for a in ("base_path", "directory") if a in node.bl_rna.properties), None)
        if base_attr is None:
            raise MifOpError("basePath is a File Output node's folder, and '%s' is a %s. NOTHING was changed."
                             % (node.name, node.bl_idname))
    if props is not None and not isinstance(props, dict):
        raise MifOpError("properties must be an object of name -> value, got %s. NOTHING was "
                         "changed." % type(props).__name__)
    if inputs is not None and not isinstance(inputs, dict):
        raise MifOpError("inputs must be an object of socket -> value, got %s. NOTHING was "
                         "changed." % type(inputs).__name__)
    # PARSED BEFORE THE FIRST WRITE, so a bad location cannot refuse after the properties landed.
    loc_xy = None
    if loc is not None:
        if not isinstance(loc, (list, tuple)) or len(loc) < 2:
            raise MifOpError("location is [x, y] - a node sits on a 2D canvas - got %r. NOTHING was "
                             "changed." % (loc,))
        loc_xy = finite_floats(loc, "location")[:2]

    unchanged = "The node was NOT changed."
    # THE CURVE IS CHECKED WHOLE BEFORE ANYTHING IS WRITTEN, and its points are snapshotted for the rollback below.
    curve_plan = _curvemap_plan(node, curve_spec, unchanged) if curve_spec is not None else None
    curve_snap = _curvemap_snapshot(node) if curve_plan is not None else None
    prop_snap = _snapshot_node_props(node, [str(k) for k in (props or {})] + ([base_attr] if base_attr else []))
    sock_snap = []
    undo = []
    plan = None
    applied, inputs_applied, written, notes = {}, {}, {}, []
    try:
        for key, value in (props or {}).items():
            # _write_node_property refuses an unknown or read-only name itself, naming what IS settable
            # on this node type - so the validation lives in one place for both the creation path and
            # this one.
            applied[str(key)] = _write_node_property(node, str(key), value, tail=unchanged)
        # ITEMS AFTER THE PROPERTIES (a File Output's format decides which slot collection is meant) and BEFORE the
        # inputs (an input can be a socket an item made); removals wait until every write that can refuse has landed.
        if items is not None:
            plan = _plan_items(node, items, unchanged)
            _apply_items(node, plan, undo, unchanged)
        if base_attr is not None:
            setattr(node, base_attr, base_path)

        # EVERY INPUT RESOLVED AND CONVERTED BEFORE ANY IS WRITTEN - after the properties, which decide
        # which of a Mix node's same-named sockets is the live one.
        staged = []
        for key, value in (inputs or {}).items():
            key = str(key)
            sock, note = _input_socket(node, key, tail=unchanged)
            if sock is None:
                raise MifOpError("node '%s' has no input socket '%s' - not as a name, a socket identifier "
                                 "or an index. It has (name [identifier]): %s. %s"
                                 % (node.name, key, ", ".join("%s [%s]" % (socket_label(s), s.identifier)
                                                             for s in node.inputs), unchanged))
            if sock.is_linked:
                src = ["%s.%s" % (l.from_node.name, l.from_socket.name) for l in sock.links]
                raise MifOpError(
                    "input '%s' of '%s' is LINKED from %s, and a connected input ignores its own value - "
                    "Blender evaluates the link. Writing it would change nothing that renders while reading "
                    "back as the new value. Remove the link first (link_group_nodes with remove:true), or "
                    "change the node driving it. %s"
                    % (key, node.name, ", ".join(src) or "another node", unchanged))
            if note:
                notes.append(note)
            staged.append((key, sock, _coerce_input(node, sock, key, value, unchanged)))

        for key, sock, value in staged:
            sock_snap.append((sock, _copy_value(sock.default_value)))
            try:
                sock.default_value = value
            except (AttributeError, TypeError, ValueError) as exc:
                raise MifOpError("cannot write %r to input '%s' of '%s': %s. %s"
                                 % (inputs[key], key, node.name, exc, unchanged))
            # READ BACK, not echoed: a factor socket clamps, and a colour keeps the alpha it had.
            inputs_applied[key] = _socket_default(sock)
            written[key] = sock.identifier
        if curve_plan is not None:
            _curvemap_write(node, [(idx, pts) for idx, _comp, pts in curve_plan])
        removed = _remove_items(node, plan) if plan is not None else 0
    except Exception:
        if curve_snap is not None:
            try:
                _curvemap_write(node, [(i, [(lx, ly, h) for (lx, ly), h in cu]) for i, cu in enumerate(curve_snap)])
            except Exception:                                       # noqa: BLE001 - see _restore_node_props
                pass
        for sock, old in reversed(sock_snap):
            try:
                sock.default_value = old
            except Exception:                                       # noqa: BLE001 - see _restore_node_props
                pass
        for step in reversed(undo):
            try:
                step()
            except Exception:                                       # noqa: BLE001 - see _restore_node_props
                pass
        _restore_node_props(node, prop_snap)
        raise

    items_out = None
    if plan is not None:
        items_out = {"collection": plan["cname"], "chosenBecause": plan["why"], "added": len(plan["add"]),
                     "renamed": plan["rename"] or None, "removed": removed,
                     "items": _items_rows(node, plan["owner"], plan["cname"]),
                     "itemsNode": plan["owner"].name}
    if label is not None:
        node.label = label
    if loc_xy is not None:
        node.location = loc_xy
    hidden = None
    if hide_unused is not None:
        # BLENDER'S TOGGLE HIDDEN SOCKETS (node.hide_socket_toggle), done natively: every enabled socket with no link
        # hidden, or every socket shown again with false. Layout only - a hidden socket keeps its value.
        socks = [s for s in list(node.inputs) + list(node.outputs) if getattr(s, "enabled", True)]
        for s in socks:
            s.hide = bool(hide_unused and not s.is_linked)
        hidden = sorted("%s.%s" % ("in" if s.is_output is False else "out", s.identifier) for s in socks if s.hide)

    curve_out = None
    if curve_plan is not None:
        # READ BACK, and EVALUATED: the points off the mapping, and each written curve sampled at eleven x across its
        # clip range by Blender's own CurveMapping.evaluate - the shape that renders, not the request echoed.
        mapping = node.mapping
        if hasattr(mapping, "initialize"):
            mapping.initialize()
        lo, hi = float(mapping.clip_min_x), float(mapping.clip_max_x)
        xs = [lo + (hi - lo) * i / 10.0 for i in range(11)]
        curve_out = {"curve": curve_mapping_points(node),
                     "curveEvaluated": {comp: [[round(x, 6), round(float(mapping.evaluate(mapping.curves[idx], x)), 6)]
                                               for x in xs] for idx, comp, _pts in curve_plan}}
    return {"tree": getattr(tree, "name", None), "node": node.name, "nodeType": node.bl_idname,
            **(curve_out or {}),
            **({"socketsHidden": hidden} if hidden is not None else {}),
            "applied": applied, "appliedCount": len(applied),
            "inputsApplied": inputs_applied, "inputsWrittenTo": written,
            **({"notes": notes} if notes else {}),
            **({"items": items_out} if items_out is not None else {}),
            **({"basePath": getattr(node, base_attr), "basePathProperty": base_attr} if base_attr else {}),
            "label": node.label,
            "location": [round(float(v), 6) for v in node.location],
            "note": "read the whole graph back with list_group_nodes{values:true} - this reports "
                    "only what it wrote."}


# ---------------------------------------------------------------------------
# group_nodes: wrap existing nodes in a labelled frame, or into a node group (2026-10-04)
# ---------------------------------------------------------------------------
_GROUP_NODES_KEYS = {"group", "tree", "nodes", "mode", "label", "name"}
_GROUP_NODE_FOR_TREE = {"ShaderNodeTree": "ShaderNodeGroup", "GeometryNodeTree": "GeometryNodeGroup",
                        "CompositorNodeTree": "CompositorNodeGroup"}
# THE INTERFACE SOCKET A BOUNDARY LINK BECOMES, by the inner socket's type. A type not here is refused, not guessed.
_IFACE_TYPE = {"VALUE": "NodeSocketFloat", "INT": "NodeSocketInt", "BOOLEAN": "NodeSocketBool",
               "VECTOR": "NodeSocketVector", "RGBA": "NodeSocketColor", "STRING": "NodeSocketString",
               "SHADER": "NodeSocketShader", "GEOMETRY": "NodeSocketGeometry", "OBJECT": "NodeSocketObject",
               "COLLECTION": "NodeSocketCollection", "TEXTURE": "NodeSocketTexture",
               "MATERIAL": "NodeSocketMaterial", "IMAGE": "NodeSocketImage", "ROTATION": "NodeSocketRotation",
               "MATRIX": "NodeSocketMatrix", "MENU": "NodeSocketMenu"}
# NOT COPIED by the generic pass: the node's place in its tree, or state Blender owns.
_COPY_SKIP = {"name", "location", "parent", "select", "dimensions", "width_hidden", "height", "show_options",
              "show_preview", "show_texture", "internal_links", "inputs", "outputs", "type"}


def _copy_ramp(src, dst):
    a, b = src.color_ramp, dst.color_ramp
    b.interpolation, b.color_mode = a.interpolation, a.color_mode
    if hasattr(a, "hue_interpolation"):
        b.hue_interpolation = a.hue_interpolation
    while len(b.elements) < len(a.elements):
        b.elements.new(0.5)
    while len(b.elements) > len(a.elements):
        b.elements.remove(b.elements[-1])
    for ea, eb in zip(a.elements, b.elements):
        eb.position, eb.color = ea.position, tuple(ea.color)


def _copy_curves(src, dst):
    ma, mb = src.mapping, dst.mapping
    for ca, cb in zip(ma.curves, mb.curves):
        while len(cb.points) < len(ca.points):
            cb.points.new(0.5, 0.5)
        while len(cb.points) > len(ca.points):
            cb.points.remove(cb.points[-1])
        for pa, pb in zip(ca.points, cb.points):
            pb.location = tuple(pa.location)
            pb.handle_type = pa.handle_type
    mb.update()


def _copy_node(src, dst_tree, offset):
    """A copy of `src` in `dst_tree`: its own settings, colour ramp, curves and every input's value."""
    dst = dst_tree.nodes.new(src.bl_idname)
    for prop in src.bl_rna.properties:
        key = prop.identifier
        if prop.is_readonly or key in _COPY_SKIP or key.startswith("bl_"):
            continue
        try:
            val = getattr(src, key)
            setattr(dst, key, val if prop.type == "POINTER" or isinstance(val, (str, int, float, bool))
                    else tuple(val) if hasattr(val, "__len__") else val)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass
    if getattr(src, "color_ramp", None) is not None:
        _copy_ramp(src, dst)
    if getattr(src, "mapping", None) is not None and hasattr(src.mapping, "curves"):
        _copy_curves(src, dst)
    dst.location = (src.location[0] - offset[0], src.location[1] - offset[1])
    for sa, sb in zip(src.inputs, dst.inputs):
        if hasattr(sa, "default_value") and hasattr(sb, "default_value"):
            try:
                sb.default_value = _copy_value(sa.default_value)
            except (AttributeError, TypeError, ValueError):
                pass
    return dst


def _settings_view(node):
    """What must survive the copy, as comparable data."""
    row = node_readback(node, values=True)
    return {"settings": row.get("settings"), "colorRamp": row.get("colorRamp"),
            "inputs": {i.get("identifier"): i.get("value") for i in row.get("inputs") or [] if "value" in i}}


def _has_items_or_zone(node):
    """True for a node this call cannot copy: a zone's input or output, or a node holding item-list entries."""
    if getattr(node, "paired_output", None) is not None:
        return True
    owner, colls = _item_collections(node)
    return owner.name != node.name or any(len(c) for c in colls.values())


def _socket_by_identifier(coll, ident):
    return next((s for s in coll if s.identifier == ident), None)


def op_group_nodes(params):
    """Wrap existing nodes in a labelled FRAME, or move them into a new node GROUP with the links that crossed the
    boundary exposed as the group's sockets.

    FRAME puts each node in a new Frame through the same parent rule add_group_node's `parent` uses (a node's location
    then reads relative to the frame). GROUP is what Blender's Make Group (node.group_make) does, done natively because
    that operator needs a node editor: each node is re-made inside a new group with its settings, colour ramp, curves
    and input values copied - and COMPARED, node by node, before anything is removed - the links between the moved
    nodes rebuilt inside, one group input per outside source and one group output per inside source, and the group
    node wired where the nodes were. A node whose copy does not read back the same is refused, and the call takes back
    everything it made.

    params:
      group / tree (str, required)   the tree, by any address _tree takes (a node group, material:<name>, ...)
      nodes [names] (required)
      mode                           FRAME | GROUP
      label (str)                    the frame's label, or the group node's
      name (str)                     the frame node's name, or the new node group's name
    """
    reject_unknown(params, _GROUP_NODES_KEYS, "group_nodes", {"as": "spelt mode: FRAME or GROUP."})
    tree = _tree(take(params, "group", "tree", required=True, kind=str))
    mode = str(take(params, "mode", required=True, kind=str)).upper()
    if mode not in ("FRAME", "GROUP"):
        raise MifOpError("mode must be FRAME or GROUP, got %r. NOTHING was changed." % params.get("mode"))
    names = params.get("nodes")
    if not isinstance(names, (list, tuple)) or not names or not all(isinstance(n, str) for n in names):
        raise MifOpError("nodes must be a non-empty list of node names, got %r. NOTHING was changed." % (names,))
    names = list(dict.fromkeys(names))
    missing = [n for n in names if tree.nodes.get(n) is None]
    if missing:
        raise MifOpError("'%s' has no node named %s. Present: %s. NOTHING was changed."
                         % (tree.name, ", ".join("'%s'" % n for n in missing),
                            ", ".join(n.name for n in tree.nodes)[:600]))
    nodes = [tree.nodes[n] for n in names]
    label = take(params, "label", default=None, kind=str)
    new_name = take(params, "name", default=None, kind=str)
    structural = [n.name for n in nodes if n.bl_idname in _STRUCTURAL_NODES or n.bl_idname in ("NodeFrame",)]
    if structural:
        raise MifOpError("%s cannot be grouped: a tree's input, output and frame nodes stay where they are. NOTHING "
                         "was changed." % ", ".join(structural))
    if mode == "FRAME":
        return _frame_nodes(tree, nodes, label, new_name)
    return _group_into_tree(tree, nodes, label, new_name)


def _frame_nodes(tree, nodes, label, name):
    frame = tree.nodes.new("NodeFrame")
    if name:
        frame.name = name
    if label is not None:
        frame.label = label
    try:
        for node in nodes:
            _put_in_frame(tree, node, frame.name, "The frame was removed again - NOTHING was changed.")
    except MifOpError:
        for node in nodes:
            if node.parent is not None and node.parent.name == frame.name:
                node.parent = None
        tree.nodes.remove(frame)
        raise
    return {"tree": tree.name, "mode": "FRAME", "frame": frame.name, "label": frame.label,
            "requestedName": name, "nameWasSuffixed": bool(name) and frame.name != name,
            "nodes": [{"name": n.name, "parent": n.parent.name if n.parent else None,
                       "location": [round(n.location[0], 2), round(n.location[1], 2)]} for n in nodes]}


def _group_into_tree(tree, nodes, label, name):
    gtype = _GROUP_NODE_FOR_TREE.get(tree.bl_idname)
    if gtype is None:
        raise MifOpError("a %s cannot hold a node group. NOTHING was changed." % tree.bl_idname)
    zoned = [n.name for n in nodes if _has_items_or_zone(n)]
    if zoned:
        raise MifOpError("%s carr%s item lists or belong to a zone, which this call does not copy - group them in "
                         "Blender, or leave them out. NOTHING was changed."
                         % (", ".join(zoned), "ies" if len(zoned) == 1 else "y"))
    inside = {n.name for n in nodes}
    incoming = [lk for lk in tree.links if lk.to_node.name in inside and lk.from_node.name not in inside]
    outgoing = [lk for lk in tree.links if lk.from_node.name in inside and lk.to_node.name not in inside]
    internal = [lk for lk in tree.links if lk.from_node.name in inside and lk.to_node.name in inside]
    for lk, end in [(lk, lk.to_socket) for lk in incoming] + [(lk, lk.from_socket) for lk in outgoing]:
        if end.type not in _IFACE_TYPE:
            raise MifOpError("the link %s.%s -> %s.%s crosses the boundary on a %s socket, which this call cannot "
                             "expose. NOTHING was changed." % (lk.from_node.name, lk.from_socket.name, lk.to_node.name,
                                                                lk.to_socket.name, end.type))
    # PLAIN DATA FIRST: the link objects are not touched after the first write.
    inc = [(lk.from_node.name, lk.from_socket.identifier, lk.to_node.name, lk.to_socket.identifier,
            lk.to_socket.name, lk.to_socket.type) for lk in incoming]
    out = [(lk.from_node.name, lk.from_socket.identifier, lk.from_socket.name, lk.from_socket.type,
            lk.to_node.name, lk.to_socket.identifier) for lk in outgoing]
    intl = [(lk.from_node.name, lk.from_socket.identifier, lk.to_node.name, lk.to_socket.identifier) for lk in internal]
    cx = sum(n.location[0] for n in nodes) / len(nodes)
    cy = sum(n.location[1] for n in nodes) / len(nodes)
    gtree = bpy.data.node_groups.new(name or "NodeGroup", tree.bl_idname)
    gnode = None
    try:
        copies = {}
        for n in nodes:
            copies[n.name] = _copy_node(n, gtree, (cx, cy))
        wrong = []
        for n in nodes:
            a, b = _settings_view(n), _settings_view(copies[n.name])
            # A LINKED INPUT HAS NO VALUE TO COMPARE - Blender reads the link - so only the original's unlinked
            # inputs are held against the copy's.
            b["inputs"] = {k: v for k, v in b["inputs"].items() if k in a["inputs"]}
            if a != b:
                keys = sorted(k for k in a if a[k] != b[k])
                wrong.append("%s (%s)" % (n.name, ", ".join(keys)))
        if wrong:
            raise MifOpError("the copy of %s does not read back like the node it copies, so moving it would change "
                             "the tree. NOTHING was changed." % "; ".join(wrong))
        gin = gtree.nodes.new("NodeGroupInput")
        gout = gtree.nodes.new("NodeGroupOutput")
        xs = [c.location[0] for c in copies.values()]
        gin.location = (min(xs) - 300, 0)
        gout.location = (max(xs) + 300, 0)
        for fn, fs, tn, ts in intl:
            gtree.links.new(_socket_by_identifier(copies[fn].outputs, fs), _socket_by_identifier(copies[tn].inputs, ts))
        in_ids, out_ids, iface = {}, {}, []
        for fn, fs, tn, ts, sname, stype in inc:
            key = (fn, fs)
            if key not in in_ids:
                item = _iface_new(gtree, sname, _IFACE_TYPE[stype], "INPUT")
                in_ids[key] = item.identifier
                iface.append({"name": item.name, "inOut": "INPUT", "identifier": item.identifier,
                              "from": "%s.%s" % (fn, fs)})
        for fn, fs, sname, stype, tn, ts in out:
            key = (fn, fs)
            if key not in out_ids:
                item = _iface_new(gtree, sname, _IFACE_TYPE[stype], "OUTPUT")
                out_ids[key] = item.identifier
                iface.append({"name": item.name, "inOut": "OUTPUT", "identifier": item.identifier,
                              "from": "%s.%s" % (fn, fs)})
        for fn, fs, tn, ts, _n, _t in inc:
            gtree.links.new(_socket_by_identifier(gin.outputs, in_ids[(fn, fs)]),
                            _socket_by_identifier(copies[tn].inputs, ts))
        for fn, fs, _n, _t, tn, ts in out:
            gtree.links.new(_socket_by_identifier(copies[fn].outputs, fs),
                            _socket_by_identifier(gout.inputs, out_ids[(fn, fs)]))
        gnode = tree.nodes.new(gtype)
        gnode.node_tree = gtree
        gnode.location = (cx, cy)
        if label is not None:
            gnode.label = label
        for fn, fs, tn, ts, _n, _t in inc:
            tree.links.new(_socket_by_identifier(tree.nodes[fn].outputs, fs),
                           _socket_by_identifier(gnode.inputs, in_ids[(fn, fs)]))
        for fn, fs, _n, _t, tn, ts in out:
            tree.links.new(_socket_by_identifier(gnode.outputs, out_ids[(fn, fs)]),
                           _socket_by_identifier(tree.nodes[tn].inputs, ts))
    except Exception:
        if gnode is not None:
            tree.nodes.remove(gnode)
        bpy.data.node_groups.remove(gtree)
        raise
    moved = [{"node": n.name, "becomes": copies[n.name].name} for n in nodes]
    for n in nodes:
        tree.nodes.remove(n)
    return {"tree": tree.name, "mode": "GROUP", "nodeGroup": gtree.name, "groupNode": gnode.name,
            "label": gnode.label, "requestedName": name, "nameWasSuffixed": bool(name) and gtree.name != name,
            "moved": moved, "interface": iface, "linksInside": len(intl), "linksIn": len(inc), "linksOut": len(out),
            "verified": "every copy read back the same settings, colour ramp and input values as its node"}


def _copy_value(val):
    """A detached copy of an RNA value - an array read off a node is a LIVE view, and restoring from a
    view of the thing being restored restores nothing."""
    if isinstance(val, (set, frozenset)):
        return set(val)
    if hasattr(val, "__len__") and not isinstance(val, str) and not hasattr(val, "bl_rna"):
        return tuple(val)
    return val


def _snapshot_node_props(node, keys):
    """{key: current value} for every key that names a writable property - what set_node_property puts
    back if a later key refuses. A key that names nothing is skipped: _write_node_property refuses it
    before writing, so there is nothing of it to restore."""
    snap = {}
    for key in keys:
        rna = node.bl_rna.properties.get(key)
        if rna is None or rna.is_readonly:
            continue
        try:
            snap[key] = _copy_value(getattr(node, key))
        except AttributeError:
            continue
    return snap


def _restore_node_props(node, snap):
    """Put every snapshotted property back. NEVER RAISES: a rollback that throws on its way out replaces a
    clear refusal with an obscure one and leaves the node half-written anyway (ops_rig._restore_settings,
    same reason). Enum-type keys first, because a data_type or operation decides what the others mean."""
    order = sorted(snap, key=lambda k: 0 if node.bl_rna.properties[k].type == "ENUM" else 1)
    for key in order:
        try:
            setattr(node, key, snap[key])
        except Exception:                                           # noqa: BLE001
            pass


def op_remove_node(params):
    """Delete a node, and say exactly which links died with it.

    THE COUNTERPART THAT NEVER EXISTED. add_group_node and link_group_nodes had no opposite, and
    b3b_materials records what that cost in its own docstring: "no typed op REMOVES a node ... so
    this stage can add a palette and it cannot take one off, and it cannot re-tune one in place.
    Re-running with different numbers means rebuilding from b1_shell". set_node_property answered
    the re-tuning half on 2026-09-07; this answers the other.

    REMOVAL SEVERS LINKS IN SILENCE, which is the whole reason this reports rather than just
    deletes. Measured on 4.4: a tree of three links, removing one middle node leaves one - both
    links touching it are gone and Blender says nothing. A caller who deleted a colour ramp and did
    not know its two links went with it would be looking at an unlinked Base Color and no
    explanation. So every link touching the node is recorded BEFORE the delete and returned, which
    is enough to rewire.

    LINKS ARE MATCHED BY NAME, NOT BY IDENTITY, and that is a real Blender trap rather than style.
    `link.from_node is node` is FALSE even when they are the same node: bpy hands back a fresh RNA
    wrapper on each access, so identity comparison finds nothing and the report would come back
    empty on a node that had links. Found by writing it the obvious way first and getting
    "links touching: []" beside a link count that dropped by two.

    AND THE PYTHON REFERENCE DIES WITH THE NODE. After `nodes.remove(n)`, touching `n.name` raises
    ReferenceError - so the name and everything else reported is captured first.

    params:
      group / tree (str, required)  the tree, as everywhere: a group name, "scene:compositor",
                                    "scene:world", "material:<name>", "world:<name>", "light:<name>"
      node (str, required)          which node, by name
      allowStructural (bool)        permit removing a Group Output, Material Output and the like.
                                    Default false: a material with no output renders black, and a
                                    group with no Group Output evaluates to EMPTY - this module
                                    measured that as an object vanishing on all four builds.
      dissolve (bool)               (2026-10-06) KEEP THE THROUGH-CONNECTION. THE GAP: an ambient-occlusion
                                    node taken out of a roughness chain left the chain cut, to be rewired
                                    by hand from severedLinks. With dissolve the node's first linked output
                                    is matched to its first linked input of the same socket type, and that
                                    input's source is linked to everything the output fed - Blender's
                                    Dissolve, done here because node.internal_links is empty on 4.2 until
                                    the tree is redrawn (measured 4.2.23). Refused, nothing removed: no
                                    linked input or no linked output, or no linked input of the output's
                                    type (the types are named). keptLinks is read back off the tree.
    """
    reject_unknown(params, _RMNODE_KEYS, "remove_node", {
        # allowStructural ON THE WIRE. The MCP wrapper's Python argument is allow_structural and it
        # sends this spelling, so a snake_case answer here would be right for one caller and
        # rejected for the other - and this message only ever reaches the one on the wire.
        "force": "spelt allowStructural on the wire (the MCP wrapper's argument is "
                 "allow_structural), and it is narrower than force. Material Output, Group Output, "
                 "Composite and friends are refused because removing them renders black or "
                 "evaluates a group to EMPTY, which makes the object vanish.",
        "relink": "spelt dissolve: true - the node's first linked output is bridged to the source of its first "
                  "linked input of the same type. Every other link that dies is REPORTED as "
                  "fromNode/fromSocket/toNode/toSocket, enough to rebuild with link_group_nodes.",
    })
    tree = _tree(take(params, "group", "tree", required=True, kind=str))
    node_name = str(take(params, "node", required=True, kind=str))
    allow_structural = take_bool(params, "allowStructural", default=False)
    dissolve = take_bool(params, "dissolve", default=False)

    node = tree.nodes.get(node_name)
    if node is None:
        known = [n.name for n in tree.nodes]
        raise MifOpError("no node named '%s' in that tree. Present: %s. NOTHING was removed."
                         % (node_name, ", ".join(known[:20]) if known else "<none>"))

    bl_idname = node.bl_idname
    if bl_idname in _STRUCTURAL_NODES and not allow_structural:
        raise MifOpError(
            "'%s' is a %s, and removing it breaks the tree rather than editing it - a material with "
            "no output renders black, and a group with no Group Output evaluates to EMPTY, which "
            "makes the object vanish. Pass allowStructural:true if that is genuinely what you want. "
            "NOTHING was removed." % (node_name, bl_idname))

    bridge = _dissolve_plan(node) if dissolve else None

    # CAPTURED BEFORE THE DELETE, and matched BY NAME - see the docstring for why identity does not
    # work here. Deleting first and reporting after would report nothing at all: the links are gone
    # and the node reference raises.
    severed = []
    for link in tree.links:
        if link.from_node.name == node_name or link.to_node.name == node_name:
            severed.append({"fromNode": link.from_node.name, "fromSocket": link.from_socket.name,
                            "toNode": link.to_node.name, "toSocket": link.to_socket.name})
    label = node.label
    links_before = len(tree.links)

    tree.nodes.remove(node)
    kept = _dissolve_relink(tree, bridge) if bridge is not None else None

    return {"tree": getattr(tree, "name", None), "removed": node_name, "nodeType": bl_idname,
            **({"dissolved": True, "keptLinks": kept, "bridgedType": bridge["type"]} if bridge is not None else {}),
            "label": label, "wasStructural": bl_idname in _STRUCTURAL_NODES,
            "severedLinks": severed, "severedCount": len(severed),
            "linksBefore": links_before, "linksAfter": len(tree.links),
            "nodesRemaining": len(tree.nodes),
            "note": "the severed links are listed so they can be rebuilt with link_group_nodes - "
                    "Blender removes them silently and nothing else reports which they were."}


def _dissolve_plan(node):
    """remove_node dissolve: {source (node, socket identifier), targets [(node, socket identifier)], type}, or a
    refusal - nothing changed. The first linked output, and the first linked input of the same socket type."""
    tail = "NOTHING was removed."
    ins = [s for s in node.inputs if s.is_linked and getattr(s, "enabled", True)]
    outs = [s for s in node.outputs if s.is_linked and getattr(s, "enabled", True)]
    if not ins or not outs:
        raise MifOpError("dissolve keeps the link through '%s', and it has %s - there is nothing to bridge. Remove it "
                         "without dissolve. %s" % (node.name, "no linked input" if not ins else "no linked output",
                                                   tail))
    out_sock = outs[0]
    in_sock = next((s for s in ins if s.type == out_sock.type), None)
    if in_sock is None:
        raise MifOpError("dissolve bridges an input to an output of the same type, and '%s''s linked output %s is %s "
                         "while its linked inputs are %s. Remove it without dissolve and link what you mean with "
                         "link_group_nodes. %s" % (node.name, out_sock.name, out_sock.type,
                                                   ", ".join("%s (%s)" % (s.name, s.type) for s in ins), tail))
    src = in_sock.links[0]
    return {"source": (src.from_node.name, src.from_socket.identifier),
            "targets": [(lk.to_node.name, lk.to_socket.identifier) for lk in out_sock.links],
            "type": out_sock.type, "through": [in_sock.name, out_sock.name]}


def _dissolve_relink(tree, bridge):
    """Make the bridge's links after the node is gone, and read them back off the tree."""
    fnode = tree.nodes[bridge["source"][0]]
    fsock = next(s for s in fnode.outputs if s.identifier == bridge["source"][1])
    for tname, tident in bridge["targets"]:
        tnode = tree.nodes[tname]
        tree.links.new(fsock, next(s for s in tnode.inputs if s.identifier == tident))
    want = {(fnode.name, fsock.identifier, t, i) for t, i in bridge["targets"]}
    return [{"fromNode": lk.from_node.name, "fromSocket": lk.from_socket.name, "toNode": lk.to_node.name,
             "toSocket": lk.to_socket.name, "valid": bool(lk.is_valid)} for lk in tree.links
            if (lk.from_node.name, lk.from_socket.identifier, lk.to_node.name, lk.to_socket.identifier) in want]


_LIST_GROUPS_KEYS = ("type", "limit", "nameExcludes")


def op_list_node_groups(params):
    """Every node group in the file, by name - the enumerator five ops needed and none provided.

    FOUND 2026-09-08 while giving audit_blender_read_purity a fixture. It reported list_group_nodes
    as "NOT PROBED AT ALL - takes group/tree, needs a node group to exist", and there was no way to
    discover one: add_group_interface, add_group_node, assign_node_group, link_group_nodes and
    list_group_nodes all take a group NAME, create_node_group makes one, and nothing listed them. A
    caller had to remember what it had created, or read the names out of a REFUSAL - _tree()'s
    not-found message prints them, which is discovery by deliberate error.

    WHY NEITHER ASYMMETRY AUDIT CAUGHT IT, which is the part worth keeping. node_groups has a
    creator (create_node_group) and is reached by a caller-supplied key, so
    audit_blender_consumer_no_creator counts it as covered and
    audit_blender_demanded_kind_has_creator finds its creator. Both are right. The missing piece is
    a third shape neither asks about: CREATED, CONSUMED, AND NOT LISTABLE.

    `type` filters by tree type the way the rest of this module spells it (GeometryNodeTree,
    ShaderNodeTree, CompositorNodeTree). Omit it for every group.
    """
    reject_unknown(params, _LIST_GROUPS_KEYS, "list_node_groups",
                   notes={
                       "name": "this LISTS groups; list_group_nodes reads the nodes inside one by name",
                       "group": "see `name` - list_group_nodes takes a group, this returns them all",
                       "tree": "see `name` - this is the enumerator, not the reader",
                   })
    want = take(params, "type", default=None, kind=str)
    # A TYPE THIS BUILD DOES NOT KNOW IS REFUSED: compared exactly, a misspelled type answered an empty list - an empty
    # result that should fail (found 2026-10-25). Known = every registered node tree class, add-on trees included, and
    # whatever the file's groups already are.
    if want:
        known = {t for t in ("GeometryNodeTree", "ShaderNodeTree", "CompositorNodeTree", "TextureNodeTree")
                 if hasattr(bpy.types, t)}
        known |= {getattr(c, "bl_idname", "") for c in bpy.types.NodeTree.__subclasses__()} - {""}
        known |= {g.bl_idname for g in bpy.data.node_groups}
        if want not in known:
            raise MifOpError("type %r is not a node tree type this Blender knows. Known: %s (case matters)."
                             % (want, ", ".join(sorted(known))))
    rows = []
    for g in bpy.data.node_groups:
        if want and g.bl_idname != want:
            continue
        rows.append({
            "name": g.name,
            "type": g.bl_idname,
            # USERS AND FAKE USER, because a group with neither is one the next save DELETES - the
            # same fact file_info reports for orphans, said here where somebody choosing a group
            # will see it.
            "users": int(getattr(g, "users", 0)),
            "fakeUser": bool(getattr(g, "use_fake_user", False)),
            "nodes": len(g.nodes),
        })
    rows.sort(key=lambda r: r["name"])
    rows = exclude_rows(rows, params)
    matched = len(rows)
    rows, extra = limit_rows(rows, params, "node group(s)")
    return dict({"groups": rows, "count": matched,
            "filteredBy": want or None,
            "totalInFile": len(bpy.data.node_groups)}, **extra)


def op_list_group_nodes(params):
    """Every node and link in a group, plus whether the Group Output is actually reachable.

    THE REACHABILITY LINE IS THE POINT. An unlinked Group Output is not an error - the modifier
    evaluates to EMPTY and the object vanishes, measured as 0 vertices on 3.6, 4.2, 4.4 and 5.0.
    Nothing else in Blender tells you, and nothing about the scene says which end to look at.

    `values` ADDS THE SETTINGS, which is what turns this from a shape listing into a read-back.
    Without it the response says a Noise Texture exists and nothing about its scale, detail or
    dimensions - so an agent that has just authored a procedural material can see the graph it
    drew and none of the numbers it wrote, and has no way to check its own work. Off by default
    because the graphs this addon builds for geometry nodes are wide and the shape is usually the
    question.

    params:
      limit (int)                  return at most this many NODES. Unlimited by default.
      nameExcludes (str|list)      drop nodes whose name contains any of these
      maxLinks (int)               return at most this many LINKS. Unlimited by default.

    TWO BOUNDS BECAUSE A GRAPH HAS TWO LISTS THAT GROW INDEPENDENTLY, following maxKeyframes in
    list_keyframes rather than inventing a shape beside it. A dense tree is link-heavy and a
    `values` read is node-heavy, so one number cannot bound both: capping nodes alone still
    returns every link, and a caller who wanted the shape gets the part they did not ask for.
    """
    reject_unknown(params, _LIST_KEYS | {"limit", "nameExcludes", "maxLinks"}, "list_group_nodes")
    tree = _tree(take(params, "group", "tree", required=True, kind=str))
    values = take_bool(params, "values", default=False)
    nodes = [node_readback(n, values=values) for n in tree.nodes]
    links = [{"from": "%s.%s" % (l.from_node.name, l.from_socket.name),
              "to": "%s.%s" % (l.to_node.name, l.to_socket.name),
              # "Math.Value" is either operand; the identifier is the one that says which.
              "fromIdentifier": l.from_socket.identifier,
              "toIdentifier": l.to_socket.identifier,
              # WHETHER A LINK CARRIES A FIELD (2026-10-04): each end's display_shape, Blender's own - CIRCLE a single
              # value, DIAMOND a field or a value, DIAMOND_DOT a field holding one value; 5.x adds LINE (geometry),
              # VOLUME_GRID and LIST. Read on 3.6.23, 4.4.0 and 5.2.1 (an Index output is DIAMOND on all three).
              "fromShape": l.from_socket.display_shape, "toShape": l.to_socket.display_shape,
              "valid": bool(l.is_valid)} for l in tree.links]

    # THE TERMINAL DEPENDS ON THE TREE TYPE. This looked only for NodeGroupOutput, which a
    # compositor tree does not have by design - so pointed at one it would have reported "nothing
    # is connected to the Group Output" for a perfectly wired compositor. A wrong answer from a
    # field whose entire purpose is telling you the tree is inert.
    terminals, note = _terminals(tree)
    outs = [n for n in tree.nodes if n.bl_idname in terminals]
    reachable = False
    if outs:
        seen, stack = set(), [outs[0]]
        while stack:
            cur = stack.pop()
            if cur.name in seen:
                continue
            seen.add(cur.name)
            for sock in cur.inputs:
                for link in sock.links:
                    if link.from_node.bl_idname == "NodeGroupInput":
                        reachable = True
                    stack.append(link.from_node)
        reachable = reachable or any(l.to_node.bl_idname in terminals for l in tree.links)
    # THE BOUNDS APPLY TO THE ROWS ONLY. outputReachable, outputNodes and _terminals all walk
    # tree.nodes and tree.links directly, so the reachability answer is the whole tree's whatever
    # the caller asked to see - a sampled graph must not be able to report itself inert.
    nodes = exclude_rows(nodes, params)
    matched_nodes = len(nodes)
    total_links = len(links)
    max_links = params.get("maxLinks")
    links_truncated = False
    if max_links is not None:
        try:
            # finite_int: int() raised a raw OverflowError for Infinity, past this except (5.0.1, 2026-09-20).
            max_links = finite_int(max_links, "maxLinks")
        except MifOpError:
            raise MifOpError("'maxLinks' must be a whole number of links, got %r." % (max_links,))
        if max_links < 0:
            raise MifOpError("'maxLinks' cannot be negative, got %d." % max_links)
        links_truncated = max_links < total_links
        links = links[:max_links]
    nodes, extra = limit_rows(nodes, params, "node(s)")
    out = {"group": tree.name, "nodes": nodes, "links": links,
           # COUNTS ARE THE MATCHED TOTALS, never the returned lengths - a caller deciding whether
           # to page needs the true size, and linkCount is the size of a list they may have capped.
           "nodeCount": matched_nodes, "linkCount": total_links,
           "interface": _iface_items(tree),
           "outputReachable": reachable,
           "treeType": getattr(tree, "bl_idname", None),
           "outputNodes": [n.name for n in outs],
           "reachabilityNote": None if reachable else note}
    if links_truncated:
        out["linksTruncated"] = True
        out["linksReturned"] = len(links)
    return dict(out, **extra)


def _output_attribute_plan(tree, raw):
    """{output name: (identifier, attribute name)} for outputAttributes, checked before the modifier exists. Only a
    field output (not Geometry) can be written out as a named attribute."""
    if raw is None:
        return {}
    if not isinstance(raw, dict) or not raw or any(not isinstance(v, str) or not v for v in raw.values()):
        raise MifOpError("outputAttributes is {output socket name: attribute name}. NOTHING was assigned.")
    outs = {}
    if hasattr(tree, "interface"):
        for it in tree.interface.items_tree:
            if getattr(it, "item_type", "SOCKET") == "SOCKET" and it.in_out == "OUTPUT":
                outs[it.name] = (it.identifier, getattr(it, "socket_type", ""))
    else:
        for s in tree.outputs:
            outs[s.name] = (s.identifier, getattr(s, "bl_socket_idname", ""))
    plan = {}
    for name, attr in raw.items():
        if name not in outs:
            raise MifOpError("'%s' has no output %r (it has %s). NOTHING was assigned."
                             % (tree.name, name, ", ".join(outs) or "none"))
        ident, kind = outs[name]
        if "Geometry" in kind:
            raise MifOpError("%r is the Geometry output - only a field output (a float, vector, colour...) is written "
                             "out as an attribute. NOTHING was assigned." % name)
        plan[name] = (ident, attr)
    return plan


def _set_output_attribute(mod, ident, attr):
    """Name the attribute a field output writes: mod.properties.outputs[id]['attribute_name'] on 5.2, a flat
    '<id>_attribute_name' id-property before it (the same split as _modifier_socket_group)."""
    outs = getattr(getattr(mod, "properties", None), "outputs", None)
    if outs is not None:
        outs[ident]["attribute_name"] = attr
    else:
        mod[ident + "_attribute_name"] = attr


def _get_output_attribute(mod, ident):
    outs = getattr(getattr(mod, "properties", None), "outputs", None)
    if outs is not None:
        return outs[ident].get("attribute_name") if ident in outs.keys() else None
    return mod.get(ident + "_attribute_name")


def _attribute_switch(mod, ident):
    """Where an input's attribute mode lives: ('rna', the 5.2 input struct) or ('flat', None) for the
    '<id>_use_attribute' id-properties of 3.6-5.0 - or (None, None) for an input that cannot read an attribute (not a
    field: Geometry, an Object, a single value). Measured 2026-10-05: 5.2.2's mod.properties.inputs.<id> has type
    VALUE | ATTRIBUTE and attribute_name only on a field input; 4.2.23 to 5.0.1 have the two flat keys."""
    inputs = getattr(getattr(mod, "properties", None), "inputs", None)
    if inputs is not None:
        rna = getattr(inputs, ident, None)
        return ("rna", rna) if rna is not None and hasattr(rna, "attribute_name") and hasattr(rna, "type") \
            else (None, None)
    return ("flat", None) if ident + "_use_attribute" in mod.keys() else (None, None)


def _set_modifier_attribute(mod, ident, label, spec):
    """Feed input `ident` from a named attribute -> the name read back. Refuses a spec that is not {attribute: name}
    and an input that cannot read an attribute."""
    name = spec.get("attribute")
    if set(spec) != {"attribute"} or not isinstance(name, str) or not name:
        raise MifOpError("a field input fed by an attribute is {attribute: name}, got %r." % (spec,))
    where, rna = _attribute_switch(mod, ident)
    if where is None:
        raise MifOpError("input '%s' is not a field input (Geometry, a datablock or a single value), so it cannot "
                         "read an attribute." % label)
    if where == "rna":
        rna.type = "ATTRIBUTE"
        rna.attribute_name = name
        got = rna.attribute_name if rna.type == "ATTRIBUTE" else None
    else:
        # THE FLAG'S OWN TYPE: an int id-property on 4.2.23, a Boolean one on 4.4.0 and 5.0.1, which refuses an int
        old = mod[ident + "_use_attribute"]
        mod[ident + "_use_attribute"] = True if isinstance(old, bool) else 1
        mod[ident + "_attribute_name"] = name
        got = mod[ident + "_attribute_name"] if mod[ident + "_use_attribute"] else None
    mod.id_data.update_tag()
    return got


def _use_value_mode(mod, ident):
    """A plain value given to an input that reads an attribute puts it back on its value, or the value is ignored."""
    where, rna = _attribute_switch(mod, ident)
    if where == "rna" and rna.type == "ATTRIBUTE":
        rna.type = "VALUE"
    elif where == "flat" and mod[ident + "_use_attribute"]:
        old = mod[ident + "_use_attribute"]
        mod[ident + "_use_attribute"] = False if isinstance(old, bool) else 0


# A NODE GROUP FROM AN ASSET LIBRARY (2026-10-06). append_from_file appends a named group from a .blend the caller
# names, and nothing found the file: the groups Blender ships (Scatter on Surface, the hair groups) sit in the build's
# own datafiles/assets folder, whose path differs per install and whose layout moved between builds. MEASURED on the
# five installs: 3.6.23 geometry_nodes/procedural_hair_node_assets.blend only; 4.2.17 and 4.4.0 add
# geometry_nodes/smooth_by_angle.blend; 5.0.1 nodes/{geometry_nodes_essentials, procedural_hair_node_assets,
# shading_nodes_essentials, compositing_nodes_essentials}.blend; 5.2.1 adds geometry_nodes_dynamics_assets and
# principal_components. One name can be in several files (Separate Spherical is in the geometry, shading and
# compositing files on 5.0), so the tree type is checked after the load and a wrong one taken out again.
_ASSET_SOURCE_KEY = "mif_asset_source"
_LIBRARY_BLEND_CAP = 2000


def _library_blends(library, tail):
    """(label, [.blend files]) for DEFAULT (the build's bundled assets), a registered asset library's name, or a path
    to a folder or a .blend."""
    if not isinstance(library, str) or not library.strip():
        raise MifOpError("library is DEFAULT (the assets this Blender ships), the name of an asset library in the "
                         "preferences, or a path to a folder or a .blend - got %r. %s" % (library, tail))
    if library.strip().upper() in ("DEFAULT", "ESSENTIALS", "BUNDLED"):
        root = bpy.utils.system_resource("DATAFILES", path="assets")
        label = "DEFAULT (%s)" % root
    else:
        from .ops_file import _asset_libs
        lib = next((x for x in _asset_libs() if x.name == library), None)
        root = bpy.path.abspath(lib.path) if lib is not None else os.path.abspath(os.path.expanduser(library))
        label = library if lib is None else "%s (%s)" % (library, root)
        if lib is None and not os.path.exists(root):
            raise MifOpError("library %r is not DEFAULT, not an asset library in the preferences (they are: %s) and "
                             "not a folder or file that exists. %s"
                             % (library, ", ".join(x.name for x in _asset_libs()) or "none", tail))
    if not root or not os.path.exists(root):
        raise MifOpError("library %s points at '%s', which does not exist. %s" % (label, root, tail))
    if os.path.isfile(root):
        if not root.lower().endswith(".blend"):
            raise MifOpError("library '%s' is a file but not a .blend. %s" % (root, tail))
        return label, [root]
    files = []
    for base, dirs, names in os.walk(root):
        dirs.sort()
        files.extend(os.path.join(base, n) for n in sorted(names) if n.lower().endswith(".blend"))
        if len(files) >= _LIBRARY_BLEND_CAP:
            break
    return label, files


def _library_group(name, library, tail):
    """Find the asset-marked GeometryNodeTree `name` in a library, append it (or reuse the copy an earlier call
    appended from the same file) -> (tree, report). A name the library does not hold is refused with what it does."""
    label, files = _library_blends(library, tail)
    held, hits = set(), []
    for path in files:
        try:
            with bpy.data.libraries.load(path, assets_only=True) as (src, _dst):
                names = list(src.node_groups)
        except Exception:                                          # noqa: BLE001 - an unreadable file holds nothing
            continue
        held.update(names)
        if name in names:
            hits.append(path)
    if not hits:
        listed = sorted(held)
        raise MifOpError("library %s holds no asset node group named %r. It holds %d: %s%s. %s"
                         % (label, name, len(listed), ", ".join(listed[:80]),
                            " ..." if len(listed) > 80 else "", tail))
    wrong = []
    for path in hits:
        source = "%s|%s" % (os.path.normcase(os.path.abspath(path)), name)
        for g in bpy.data.node_groups:
            if g.get(_ASSET_SOURCE_KEY) == source and g.bl_idname == "GeometryNodeTree":
                return g, {"library": label, "file": path, "name": name, "appended": False, "reused": g.name,
                           "alsoAppended": []}
        before = set(bpy.data.node_groups)
        with bpy.data.libraries.load(path, link=False) as (_src, dst):
            dst.node_groups = [name]
        tree = dst.node_groups[0] if dst.node_groups else None
        added = [g for g in bpy.data.node_groups if g not in before]
        if tree is None or tree.bl_idname != "GeometryNodeTree":
            wrong.append("%s (a %s)" % (os.path.basename(path), getattr(tree, "bl_idname", "nothing")))
            for g in added:
                bpy.data.node_groups.remove(g)
            continue
        tree[_ASSET_SOURCE_KEY] = source
        return tree, {"library": label, "file": path, "name": name, "appended": True, "reused": None,
                      "alsoAppended": sorted(g.name for g in added if g != tree)}
    raise MifOpError("library %s holds %r only as another kind of node tree - %s - and only a geometry node group "
                     "drives a Nodes modifier. NOTHING was appended. %s" % (label, name, "; ".join(wrong), tail))


def _menu_items(holder, key):
    """[(name, int value)] of a Menu input stored as an int id-property, from its UI data - or None."""
    try:
        items = holder.id_properties_ui(key).as_dict().get("items")
    except (TypeError, KeyError, AttributeError):
        return None
    return [(row[0], row[4] if len(row) > 4 else i) for i, row in enumerate(items or [])] or None


def _set_menu_input(mod, ident, label, val):
    """Write a Menu input by its item NAME -> the name read back. MEASURED 2026-10-06 on 5.0.1: the modifier stores a
    menu input as an INT id-property ('Instance Type' Object=1, Collection=0), so a name was refused with "Cannot assign
    a 'str' value to the existing Int IDProperty"; the names and their ints are in the property's UI data. On 5.2.1 the
    group holds the int too and has no UI data, and mod.properties.inputs.<identifier>.value is an RNA enum of the
    names - written there."""
    rna = getattr(getattr(getattr(mod, "properties", None), "inputs", None), ident, None)
    if rna is not None and "value" in rna.bl_rna.properties and rna.bl_rna.properties["value"].type == "ENUM":
        names = [e.identifier for e in rna.bl_rna.properties["value"].enum_items]
        try:
            rna.value = val
        except TypeError:
            raise MifOpError("input '%s' is a menu of %s - got %r." % (label, ", ".join(names) or "(none)", val))
        mod.id_data.update_tag()
        return rna.value
    grp = _modifier_socket_group(mod, ident)
    holder, key = (mod, ident) if grp is None else (grp, _MOD_SOCKET_VALUE_KEY)
    current = holder.get(key)
    if not isinstance(current, int) or isinstance(current, bool):
        _set_modifier_socket(mod, ident, str(val))
        return _get_modifier_socket(mod, ident)
    items = _menu_items(holder, key)
    if not items:
        raise MifOpError("input '%s' is a menu stored as a number and this build lists no items for it." % label)
    hit = next((v for n, v in items if n == val), None)
    if hit is None:
        raise MifOpError("input '%s' is a menu of %s - got %r." % (label, ", ".join(n for n, _v in items), val))
    _set_modifier_socket(mod, ident, hit)
    back = holder.get(key)
    return next((n for n, v in items if v == back), back)


def _input_keys(tree):
    """{key: (identifier, socket type, name)} for a group's inputs - each by IDENTIFIER, and by NAME where the name is
    the only input of that name. A bundled group can repeat one (Scatter on Surface has two 'Randomize Scale' inputs,
    a vector and a float), and a name-keyed map kept whichever came last; such a name is mapped to None, refused."""
    rows = []
    if hasattr(tree, "interface"):
        for it in tree.interface.items_tree:
            if getattr(it, "item_type", "SOCKET") == "SOCKET" and it.in_out == "INPUT":
                rows.append((it.name, it.identifier, getattr(it, "socket_type", "")))
    else:
        for s in tree.inputs:
            rows.append((s.name, s.identifier, getattr(s, "bl_socket_idname", "")))
    keys = {ident: (ident, kind, name) for name, ident, kind in rows}
    by_name = {}
    for name, ident, kind in rows:
        by_name.setdefault(name, []).append((ident, kind, name))
    for name, entries in by_name.items():
        if name in keys and keys[name] is not None and keys[name][0] == name:
            continue                                   # an identifier spelled like a name keeps the identifier
        keys[name] = entries[0] if len(entries) == 1 else None
    return keys


def op_assign_node_group(params):
    """Attach a node group to an object as a Nodes modifier, and set its exposed inputs.

    A GROUP FROM AN ASSET LIBRARY (2026-10-06): library DEFAULT (the assets this Blender ships), the name of an asset
    library in the preferences, or a folder or .blend path - the asset-marked geometry group named `group` is found in
    it and appended (an earlier call's copy from the same file is reused), then attached. A name the library does not
    hold is refused with the list of what it holds; the reply's `asset` says which file, and `exposedInputs` names
    every input with its identifier. Inputs are keyed by name or by identifier; a name two inputs share is refused.

    AN INPUT CAN READ A NAMED ATTRIBUTE (2026-10-05): inputs {socket: {attribute: name}} switches a field input to
    attribute mode and names the attribute - a vertex group read as a density mask. A plain value switches it back.

    IT WAS MESH-ONLY UNTIL 2026-09-07, and that was never a real constraint - it was `want_mesh=True`
    on the object lookup, the same helper every mesh-editing op uses. Geometry nodes drive four
    object types, and refusing the other three meant this addon could not run a node group on a
    curve at all.

    MEASURED ON 3.6.23, 4.4.0 AND 5.0.1, IDENTICALLY, with a group whose output MATCHES the object
    type - which is the whole trick, because an evaluated object can only hold geometry of its own
    kind and a mesh-outputting group on a CURVES object correctly produces nothing:

      MESH, CURVES, POINTCLOUD, VOLUME   geometry lands - 9, 33, 9 and 1 element respectively
      CURVE (legacy), FONT               the modifier attaches and this probe saw nothing arrive
      everything else                    modifiers.new() returns None; there is no modifier at all

    THE REFUSAL IS ON THE None, NOT ON A LIST OF TYPE NAMES. bpy hands back None for an object that
    cannot carry a modifier - CAMERA, LIGHT, EMPTY, ARMATURE, LATTICE, META, SURFACE - WITHOUT
    raising, which is exactly how the first version of that measurement fooled me: it called new(),
    never looked at the result, and reported that every object type accepted a NODES modifier.

    CURVE AND FONT ARE ALLOWED WITH A WARNING RATHER THAN REFUSED. The probe saw nothing land on
    them, but it counted elements on the evaluated datablock and a legacy curve's evaluated result
    may not be readable that way. Refusing on that evidence would block something that may well
    work; saying so lets a caller check.
    """
    reject_unknown(params, _ASSIGN_KEYS, "assign_node_group")
    obj = get_object(take(params, "object", "name", required=True))
    if obj.mode != "OBJECT":
        raise MifOpError("object '%s' is in %s mode. Leave edit mode before assigning a node "
                         "group. NOTHING was assigned." % (obj.name, obj.mode))
    group_name = take(params, "group", "tree", required=True, kind=str)
    library = params.get("library")
    asset = None
    if library is not None:
        # An object that cannot carry a modifier is refused BEFORE the library append, so a refused call leaves no
        # appended group behind. The types are the ones measured below (modifiers.new returns None for them).
        if obj.type in ("CAMERA", "LIGHT", "EMPTY", "ARMATURE", "LATTICE", "META", "SURFACE"):
            raise MifOpError("'%s' is a %s, and Blender will not put a modifier on one. Geometry nodes drive MESH, "
                             "CURVES, POINTCLOUD and VOLUME objects; CURVE and FONT take the modifier too. NOTHING was "
                             "appended or assigned." % (obj.name, obj.type))
        # Only ever a GeometryNodeTree: _library_group removes and refuses any other kind it appended.
        tree, asset = _library_group(group_name, library, "NOTHING was assigned.")
    else:
        tree = _tree(group_name)
        if tree.bl_idname != "GeometryNodeTree":
            raise MifOpError("'%s' is a %s, and only a GeometryNodeTree can drive a Nodes modifier. "
                             "NOTHING was assigned." % (tree.name, tree.bl_idname))
    out_attrs = _output_attribute_plan(tree, params.get("outputAttributes"))
    # REUSE THE MODIFIER THAT ALREADY HOLDS THIS GROUP, rather than stacking a second one.
    #
    # Measured on 4.4: three calls with the same object and group produced GeometryNodes,
    # GeometryNodes.001 and GeometryNodes.002, all three evaluating in sequence. An
    # assign-look-adjust loop - which is how anybody actually uses this - silently tripled
    # the stack, and until list_modifiers learned to report a NODES modifier's group there
    # was no way to even SEE it had happened.
    #
    # Reuse is the default because reassigning the same group to the same object means
    # "update it" essentially every time. reuse:false forces a second modifier for the rare
    # case where two instances of one group is deliberate, and `reused` is reported either
    # way so nothing about it is silent.
    reuse = take_bool(params, "reuse", default=True)
    mod_name = take(params, "modifierName", default=None, kind=str)
    mod = None
    if reuse:
        for m in obj.modifiers:
            if (m.type == "NODES" and getattr(m, "node_group", None) is tree
                    and (mod_name is None or m.name == str(mod_name))):
                mod = m
                break
    reused = mod is not None
    # DECIDED HERE, where the two cases are already separate, rather than at the call below. The
    # sentence a socket refusal ends with depends on whether this op just made a modifier, and
    # writing that as a conditional at the call site hides the correlation from anything reading the
    # code - including the audit, which cannot see that the "NOTHING was changed" branch is the one
    # where nothing was created.
    socket_tail = "NOTHING was changed."
    if mod is None:
        mod = obj.modifiers.new(name=str(mod_name or "GeometryNodes"), type="NODES")
        # NONE IS THE REFUSAL, and it is not an exception. bpy returns None for an object that
        # cannot carry a modifier rather than raising, so the very next line - mod.node_group -
        # would have been an AttributeError with no explanation of why. Measured across 3.6, 4.4
        # and 5.0: CAMERA, LIGHT, EMPTY, ARMATURE, LATTICE, META and SURFACE all return None.
        if mod is None:
            if asset and asset.get("appended"):
                bpy.data.node_groups.remove(tree)    # this call's append is taken back, so NOTHING stays true
            raise MifOpError(
                "'%s' is a %s, and Blender will not put a modifier on one - it returned no "
                "modifier at all rather than an error. Geometry nodes drive MESH, CURVES, "
                "POINTCLOUD and VOLUME objects (measured on 3.6, 4.4 and 5.0); CURVE and FONT take "
                "the modifier too. NOTHING was assigned." % (obj.name, obj.type))
        mod.node_group = tree
        socket_tail = "The '%s' modifier WAS added." % mod.name

    applied, refused = {}, {}
    given = params.get("inputs")
    if given:
        if not isinstance(given, dict):
            # socket_tail IS THE SENTENCE THIS FUNCTION ALREADY BUILT, twelve lines above, for
            # exactly this situation - and until 2026-09-15 the refusals below it did not carry it.
            # A caller got "'inputs' must be a {socketName: value} object" and a GeometryNodes
            # modifier on their object, with nothing connecting the two. Reported by the second
            # seat as defect O; B1511 asserts the honest PAIR - either the refused call left no
            # modifier, or it said that it added one - rather than demanding a particular route.
            #
            # THE MODIFIER STAYS. It is the thing the caller asked for and only the socket values
            # failed, so removing it would throw away work that succeeded. add_group_node below
            # takes the other route for the opposite reason.
            raise MifOpError("'inputs' must be a {socketName: value} object, got %r. %s"
                             % (given, socket_tail))
        # The modifier addresses exposed inputs by IDENTIFIER (Socket_2), not by name, which is the
        # single most confusing thing about driving geometry nodes from script. Resolve the name to
        # its identifier rather than making a caller know that.
        # THE TYPE IS CARRIED BESIDE THE IDENTIFIER. Without it an Object input took the string
        # "Cube" straight into mod[key], which Blender stores as a string and then ignores - the
        # modifier reads an empty socket and the scatter produces nothing, with no error anywhere.
        # Where the type lives moved: it.socket_type on 4.x, s.bl_socket_idname on 3.6.
        # BY NAME OR BY IDENTIFIER (2026-10-06), and a name two inputs share is refused rather than resolved to
        # whichever came last - see _input_keys.
        ident = _input_keys(tree)
        for k, v in given.items():
            if k in ident and ident[k] is None:
                refused[k] = "two or more inputs are named %r - key it by identifier: %s" % (k, ", ".join(
                    e[0] for e in ident.values() if e is not None and e[2] == k))
                continue
            entry = ident.get(k)
            if entry is None:
                refused[k] = "not an exposed input; this group exposes: %s" % (
                    ", ".join(sorted({e[2] for e in ident.values() if e is not None})) or "(none)")
                continue
            key, kind, _label = entry
            if isinstance(v, dict):
                # A FIELD INPUT FED BY A NAMED ATTRIBUTE (2026-10-05): {attribute: name}.
                try:
                    applied[k] = {"attribute": _set_modifier_attribute(mod, key, k, v)}
                except MifOpError as exc:
                    refused[k] = str(exc)
                continue
            if kind == "NodeSocketMenu" and isinstance(v, str):
                try:
                    applied[k] = _set_menu_input(mod, key, k, v)
                except MifOpError as exc:
                    refused[k] = str(exc)
                continue
            try:
                coerced = _socket_value(kind, k, v, "modifier '%s'" % mod.name, socket_tail)
                _use_value_mode(mod, key)
                _set_modifier_socket(mod, key, coerced)
                # READ BACK, because a pointer socket accepts a wrong-typed value silently on some
                # paths and simply does not hold it. Reporting the NAME rather than the datablock
                # keeps the response JSON-safe.
                got = _get_modifier_socket(mod, key)
                applied[k] = getattr(got, "name", got)
            except MifOpError as exc:
                refused[k] = str(exc)
            except (TypeError, KeyError) as exc:
                refused[k] = str(exc)

    written_attrs = {}
    for name, (ident, attr) in out_attrs.items():
        _set_output_attribute(mod, ident, attr)
        written_attrs[name] = _get_output_attribute(mod, ident)
    if out_attrs:
        mod.id_data.update_tag()
    outs = any(l.to_node.bl_idname == "NodeGroupOutput" for l in tree.links)
    return {"object": obj.name, "modifier": mod.name, "group": tree.name,
            "outputAttributes": written_attrs or None,
            # WHERE A LIBRARY GROUP CAME FROM, and every input it exposes - a caller who did not build the group has
            # no other way to learn the names (or the identifiers, where two inputs share a name).
            "asset": asset,
            "exposedInputs": ([{k: r[k] for k in ("name", "identifier", "socketType")}
                               for r in _iface_items(tree) if r.get("inOut") == "INPUT"]
                              if asset is not None else None),
            # WHICH IT WAS. False means a new modifier was added to the stack; a caller in
            # an adjust loop seeing False repeatedly is stacking duplicates.
            "reused": reused,
            "nodesModifierCount": len([m for m in obj.modifiers if m.type == "NODES"]),
            "inputsApplied": applied, "inputsRefused": refused,
            "outputConnected": outs,
            # THE OBJECT TYPE, because a node group behaves differently on each and the caller
            # cannot see which one they attached to from the modifier name.
            "objectType": obj.type,
            # AND THE THING THAT CATCHES PEOPLE ONCE THIS IS NO LONGER MESH-ONLY: an evaluated
            # object can only hold geometry of ITS OWN kind. A Mesh Grid on a CURVES object
            # evaluates fine and lands NOTHING, which reads as a broken group rather than a
            # mismatched output. Measured on 3.6, 4.4 and 5.0.
            "geometryTypeNote": (
                None if obj.type == "MESH" else
                "this is a %s object, so the group's OUTPUT has to be %s geometry - a mesh-"
                "outputting group evaluates without error and lands nothing on it."
                % (obj.type,
                   {"CURVES": "curve", "POINTCLOUD": "point", "VOLUME": "volume"}.get(
                       obj.type, "matching"))),
            "effectNote": (None if outs else
                           "the group's output is not connected, so this modifier evaluates to "
                           "EMPTY - the object will DISAPPEAR from the viewport and from every "
                           "export, not pass through unchanged. Valid in Blender, and it reports "
                           "no error. Link something to the Group Output.")}


_COMPOSITING_KEYS = {"enabled", "useCompositing", "useSequencer", "withDefaultNodes"}
_COMPINFO_KEYS = {"viewLayer", "limit", "nameExcludes"}

# THE RESERVED TREE NAME. The scene's compositor is scene.node_tree, which is NOT in
# bpy.data.node_groups - it is a tree owned by the scene - so the five node-authoring ops here could
# not reach it at all. Rather than grow a parallel set of add/link/list ops for compositing, _tree
# resolves this one reserved string to it, and every one of them works on the compositor unchanged.
#
# create_node_group REFUSES to create a group by this name, which is what keeps the reservation from
# ever being ambiguous. A precedence rule - "a real group of that name wins" - would have been the
# other option, and it makes the reachable set depend on what somebody happened to call something.
SCENE_COMPOSITOR = "scene:compositor"
SCENE_WORLD = "scene:world"
MATERIAL_PREFIX = "material:"
WORLD_PREFIX = "world:"
# A LIGHT'S TREE, since 2026-09-13. set_light_ies builds one - an IES Texture into the Emission's strength - and
# nothing could read it back: list_lights has no IES field and the tree had no address, so the only proof a profile
# was there was a Cycles render. Addressed by the light DATABLOCK's name, as materials and worlds are by theirs.
LIGHT_PREFIX = "light:"

# EVERY TREE THAT IS NOT IN bpy.data.node_groups. A node group is a datablock in its own right; a
# material's tree, a world's tree and the compositor are OWNED by the thing they shade, so none of
# them could be addressed by the five authoring ops here at all.
#
# WHAT THAT COST, measured rather than guessed: describe_material reads a material's node tree in
# full - every node, every link, every image texture - and set_material_properties could write only
# the Principled BSDF's own socket values. So the addon could DESCRIBE a shader graph in detail and
# not add a single node to one. No mix shaders, no procedural noise, no bump or normal map wired,
# no UV mapping node, no emission blend. The read half was thorough and the write half was one node
# deep.
#
# The fix is a resolver rather than a second set of add/link/list ops, for the same reason the
# compositor got one: add_group_node is already tree-agnostic - it calls tree.nodes.new(ntype) and
# names tree.bl_idname in its own error - so the only thing missing was a way to hand it the tree.
RESERVED_PREFIXES = (MATERIAL_PREFIX, WORLD_PREFIX, LIGHT_PREFIX)

# THE TERMINAL NODE IS NOT THE SAME IN EVERY TREE TYPE, and getting this wrong produces a WRONG
# answer rather than a missing one. list_group_nodes looked only for NodeGroupOutput, which a
# compositor tree does not have by design - so it would have reported "nothing is connected to the
# Group Output" for a perfectly wired compositor.
_TERMINALS = {
    "GeometryNodeTree": ("NodeGroupOutput",),
    "ShaderNodeTree": ("NodeGroupOutput", "ShaderNodeOutputMaterial", "ShaderNodeOutputWorld",
                       "ShaderNodeOutputLight"),
    "CompositorNodeTree": ("CompositorNodeComposite", "NodeGroupOutput"),
}


def _terminals(tree):
    """The bl_idnames that count as this tree's OUTPUT, and a phrase describing the pass-through."""
    kind = getattr(tree, "bl_idname", "GeometryNodeTree")
    if kind == "CompositorNodeTree":
        return _TERMINALS[kind], ("nothing is connected to a Composite node, so this compositor "
                                  "writes NOTHING to the render result. Blender treats that as "
                                  "valid and reports no error.")
    if kind == "ShaderNodeTree":
        return _TERMINALS[kind], ("nothing is connected to an output node, so this shader tree "
                                  "contributes nothing.")
    return _TERMINALS.get(kind, _TERMINALS["GeometryNodeTree"]), (
        "nothing is connected to the Group Output, so this tree evaluates to EMPTY - anything "
        "using it DISAPPEARS rather than passing through. Blender treats that as valid and "
        "reports no error.")


def _owned_tree(name):
    """Resolve one of the reserved targets to the tree it names, or None if `name` is not one.

    Refuses rather than enabling. Turning use_nodes on for a material or a world as a side effect of
    ADDRESSING it would mean a call that says "put a node here" silently converted a flat-colour
    material into a node-based one - a different thing to render, decided by a lookup.
    """
    if name == SCENE_COMPOSITOR:
        return _scene_tree()
    if name == SCENE_WORLD:
        world = bpy.context.scene.world
        if world is None:
            raise MifOpError("scene '%s' has no world, so there is no world shader tree to "
                             "address. Make one with set_world. NOTHING was changed."
                             % bpy.context.scene.name)
        return _shader_tree(world, "world", world.name)
    for prefix, store, label in ((MATERIAL_PREFIX, "materials", "material"),
                                 (WORLD_PREFIX, "worlds", "world"),
                                 (LIGHT_PREFIX, "lights", "light")):
        if not name.startswith(prefix):
            continue
        key = name[len(prefix):]
        if not key:
            raise MifOpError("'%s' names no %s - use '%s<name>'. NOTHING was changed."
                             % (name, label, prefix))
        holder = getattr(bpy.data, store).get(key)
        if holder is None:
            known = sorted(x.name for x in getattr(bpy.data, store))[:25]
            raise MifOpError("no %s named '%s'. Present: %s. NOTHING was changed."
                             % (label, key, ", ".join(known) if known else "<none>"))
        return _shader_tree(holder, label, key)
    return None


def _shader_tree(holder, label, key):
    """A material's, world's or light's own node tree, refusing when use_nodes is off.

    A datablock with use_nodes FALSE ignores its tree completely and renders its flat colour, so
    adding nodes to it would be authoring something nothing looks at - every node reading back
    perfectly and changing no pixel. Named rather than silently enabled.
    """
    if not holder.use_nodes or holder.node_tree is None:
        raise MifOpError("%s '%s' has use_nodes OFF, so its node tree is ignored entirely and its "
                         "flat colour is what renders - adding nodes would author something "
                         "nothing looks at. Turn it on first. NOTHING was changed."
                         % (label, key))
    return holder.node_tree


# THE COMPOSITOR MOVED IN 5.0, AND THE OLD ATTRIBUTE IS GONE.
#
# Established empirically on 2026-09-03 by running all four installed Blenders headless, after a
# LIVE call to compositor_info came back "AttributeError: 'Scene' object has no attribute
# 'node_tree'". Everything shipped for the compositor earlier that day was dead on the newest
# Blender this addon claims to support, and every static gate was green.
#
#   3.6 / 4.2 / 4.4   scene.node_tree, an EMBEDDED tree, gated by scene.use_nodes (default FALSE)
#   5.0               scene.node_tree is ABSENT. scene.compositing_node_group holds a real node
#                     group from bpy.data.node_groups, and scene.use_nodes defaults TRUE while the
#                     group is still None - so "use_nodes is on" no longer implies a tree exists.
#
# AND THE OUTPUT NODE CHANGED WITH IT: CompositorNodeComposite does not exist on 5.0 at all. The
# compositor is a genuine node group there, so its terminal is NodeGroupOutput. _TERMINALS already
# lists both, which is why reachability survived the move and nothing else did.
COMPOSITOR_ATTR_NEW = "compositing_node_group"      # 5.0+
COMPOSITOR_ATTR_OLD = "node_tree"                   # <= 4.4


def compositor_era(scene=None):
    """('new'|'old', attribute name) for this build - which compositor API is present."""
    sc = scene or bpy.context.scene
    if hasattr(sc, COMPOSITOR_ATTR_NEW):
        return "new", COMPOSITOR_ATTR_NEW
    return "old", COMPOSITOR_ATTR_OLD


def compositor_tree(scene=None):
    """The scene's compositing tree on either API, or None. Never raises, never mutates."""
    sc = scene or bpy.context.scene
    era, attr = compositor_era(sc)
    if era == "old" and not getattr(sc, "use_nodes", False):
        return None
    return getattr(sc, attr, None)


def _scene_tree():
    """The scene's compositing tree, or a refusal that says how to make one.

    NOT a mutator. Creating the tree here would mean a read op quietly switched compositing on for
    the whole scene, which changes what every subsequent render does - the same objection that keeps
    world_info off _background_node.
    """
    sc = bpy.context.scene
    tree = compositor_tree(sc)
    if tree is None:
        era, _ = compositor_era(sc)
        why = ("scene.use_nodes is off" if era == "old" else
               "scene.compositing_node_group is unset - on Blender 5.0 use_nodes defaults to TRUE "
               "and means nothing on its own, so a scene can look compositing-enabled with no tree "
               "at all")
        raise MifOpError("scene '%s' has no compositing tree - %s. Make one with set_compositing, "
                         "which also wires the default Render Layers -> output pair. NOTHING was "
                         "changed." % (sc.name, why))
    return tree


def _group_output_image(tree, out_node):
    """The Group Output's 'Image' input on a 5.0+ compositor node group, declaring the interface socket first
    when the group has none. An existing OUTPUT socket named Image is reused, so calling this twice never
    stacks a second Image output."""
    existing = [item for item in tree.interface.items_tree
                if getattr(item, "item_type", "") == "SOCKET" and getattr(item, "in_out", "") == "OUTPUT"
                and item.name == "Image"]
    if not existing:
        tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    return out_node.inputs["Image"]


def op_set_compositing(params):
    """Turn the scene compositor on or off - and the SECOND switch that also has to be on.

    TWO INDEPENDENT FLAGS DECIDE WHETHER COMPOSITING HAPPENS, and having one on and the other off is
    the classic silent failure:

      scene.use_nodes              whether a compositing tree EXISTS and is edited
      scene.render.use_compositing whether the render PIPELINE runs it

    With use_nodes on and use_compositing off, the whole tree sits there reading perfectly, the
    backdrop in the compositor updates, and the rendered file is completely unprocessed. Nothing
    reports it. So both are set here and both are read back.

    params:
      enabled (bool)            scene.use_nodes. Default true.
      useCompositing (bool)     scene.render.use_compositing. Defaults to follow `enabled`, since
                                turning the tree on and leaving the pipeline off is never what
                                somebody meant.
      useSequencer (bool)       scene.render.use_sequencer - the VSE runs AFTER the compositor and
                                overrides it if a strip exists.
      withDefaultNodes (bool)   wire Render Layers -> Composite if the tree is empty. Default true
                                when enabling, because an empty compositor writes nothing at all.
    """
    reject_unknown(params, _COMPOSITING_KEYS, "set_compositing")
    sc = bpy.context.scene
    enabled = take_bool(params, "enabled", default=True)
    # EVERY FLAG READ HERE, BEFORE THE FIRST WRITE. useCompositing, useSequencer and withDefaultNodes were read
    # between writes: {useCompositing: "flase"} was refused with scene.use_nodes already switched, and
    # {withDefaultNodes: "flase"} on 5.0+ with the compositor node group already created and assigned (measured
    # on 5.0.1, 2026-09-21).
    try:
        use_comp = take_bool(params, "useCompositing") if "useCompositing" in params else None
        use_seq = (take_bool(params, "useSequencer", default=True)
                   if params.get("useSequencer") is not None else None)
        with_default_nodes = take_bool(params, "withDefaultNodes", default=True)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was changed." % str(exc).rstrip("."))
    before = {"useNodes": bool(sc.use_nodes),
              "useCompositing": bool(sc.render.use_compositing),
              "useSequencer": bool(sc.render.use_sequencer)}

    era, attr = compositor_era(sc)
    # REFUSED BEFORE THE WRITE. On 5.0+ scene.use_nodes is fixed on (measured below), so enabled false is honored only
    # through render.use_compositing off - and useCompositing true asks for the opposite. This used to write both flags
    # and then fail the use_nodes read-back, leaving the scene changed under a refusal (found 2026-10-25).
    if not enabled and era == "new" and use_comp is True:
        raise MifOpError("enabled false with useCompositing true cannot hold on Blender %s: scene.use_nodes stays on from "
                         "5.0, so the compositor is turned off only by render.use_compositing off. Pass enabled false "
                         "alone, or enabled true with useCompositing true. NOTHING was changed."
                         % ".".join(str(v) for v in bpy.app.version[:2]))
    # use_nodes STILL EXISTS on 5.0 and is still worth writing - it is what the UI reflects - but on
    # that build it does not create or destroy the tree, so the group is handled separately below.
    sc.use_nodes = enabled
    # TYPE-CHECKED, and note the line two below already did it right: `useSequencer` goes through
    # take_bool behind a presence check, and this one did not. Two spellings of the same idea, two
    # lines apart, and only one of them refuses a list or a dict.
    #
    # Read raw, bool(["mif"]) is True and bool({...}) is True - so a caller sending the wrong SHAPE
    # turned compositing ON and got a success report. This op WRITES, which is what makes it worth
    # more than a tidy-up.
    #
    # Found 2026-09-08 by tools/audit_blender_raw_bool.py on its first run, after the same shape was
    # found by hand in select_faces and twice in set_shading. The version matrix's sentinel fuzz
    # could not have caught this one: it only probes parameters a payload SENDS, and useCompositing
    # is not in any payload.
    sc.render.use_compositing = use_comp if use_comp is not None else enabled
    if use_seq is not None:
        sc.render.use_sequencer = use_seq

    added = []
    tree = getattr(sc, attr, None)
    if enabled and era == "new" and tree is None:
        # ON 5.0 THE TREE IS A REAL NODE GROUP and has to be created and assigned; there is no
        # embedded tree for use_nodes to bring into being. Verified headless: assigning a
        # bpy.data.node_groups.new(name, "CompositorNodeTree") is the whole of it.
        tree = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
        setattr(sc, attr, tree)
    # withDefaultNodes was read at the top - before the enabled/tree tests (see audit_blender_shortcircuit_take:
    # disabling compositing in the same call meant it was never looked at) and before any write.
    if enabled and tree is not None and with_default_nodes:
        if not len(tree.nodes):
            rl = tree.nodes.new("CompositorNodeRLayers")
            # THE OUTPUT NODE IS NOT THE SAME NODE. CompositorNodeComposite is UNDEFINED on 5.0 -
            # `ng.nodes.new("CompositorNodeComposite")` raises "Node type undefined" - because the
            # compositor is a node group there and terminates in NodeGroupOutput. Verified on all
            # four installs rather than assumed.
            out_type = "NodeGroupOutput" if era == "new" else "CompositorNodeComposite"
            comp = tree.nodes.new(out_type)
            rl.location = (-300, 0)
            comp.location = (300, 0)
            # A GROUP OUTPUT HAS NO SOCKETS UNTIL ITS INTERFACE DECLARES ONE. This used to link to inputs[0] -
            # the empty EXTEND socket a user drags onto in the UI, which creates an interface item there and
            # nowhere else. From Python the link is made, marked INVALID, and nothing is added: measured on
            # 5.2.1 on 2026-09-16, the tree listed Group Output inputs [""], the link read valid:false, a still
            # rendered with compositing on was byte-identical to one with it off, and compositor_info reported
            # no blocker. Every compositor effect wired after it had no output to reach - a scene's lens
            # distortion refused with "Group Output has no input socket 'Image'". So the Image output is
            # declared first (reusing one if the interface already has it), linked to, and checked.
            socket = comp.inputs["Image"] if era == "old" else _group_output_image(tree, comp)
            link = tree.links.new(rl.outputs["Image"], socket)
            if not link.is_valid:
                tree.links.remove(link)
                raise MifOpError("the compositor's Render Layers could not be linked to its %s - Blender "
                                 "marked the link invalid. The two nodes WERE added; wire them with "
                                 "link_group_nodes on tree '%s'." % (comp.name, SCENE_COMPOSITOR))
            added = [rl.name, comp.name]

    after = {"useNodes": bool(sc.use_nodes),
             "useCompositing": bool(sc.render.use_compositing),
             "useSequencer": bool(sc.render.use_sequencer)}
    # VERIFIED INDIVIDUALLY. use_nodes is on the scene and use_compositing is on scene.render -
    # two datablocks - and the whole point of this op is that having one without the other is the
    # failure, so a check that only looked at one would miss exactly what it exists for.
    # ON 5.0 AND LATER scene.use_nodes IS FIXED ON, and render.use_compositing is the switch that matters. MEASURED
    # 2026-09-16: writing use_nodes False reads back True on 5.0.1 and 5.2.1 (it takes False on 3.6.23, 4.2.17 and
    # 4.4.0), while render.use_compositing False bypasses the tree on every build - an Invert between Render Layers and
    # the output rendered 188.6 with compositing on and 123.4 with it off, the uncomposited baseline, on 5.0.1 and 5.2.1
    # (scratchpad measure_compositing_disable.py). This refused enabled=False there with "set scene.use_nodes to False
    # but it reads back as True" - AFTER writing use_compositing False, so the refusal left the compositor off (second
    # seat, mailbox id 135). Disabling is now a success where the pipeline switch took and use_nodes is fixed, and says so.
    use_nodes_fixed = (not enabled and era == "new" and after["useNodes"] and not after["useCompositing"])
    if after["useNodes"] != enabled and not use_nodes_fixed:
        raise MifOpError("set scene.use_nodes to %s but it reads back as %s. render.use_compositing WAS written (%s)."
                         % (enabled, after["useNodes"], after["useCompositing"]))
    return {
        "useNodesFixed": use_nodes_fixed or None,
        "useNodesNote": ("scene.use_nodes stays on in Blender 5.0 and later; the compositor is off because "
                         "render.use_compositing is off, which is the switch the render reads.") if use_nodes_fixed else None,
        "ok": True,
        "scene": sc.name,
        "before": before,
        "nodesAdded": added,
        "treeName": tree.name if tree is not None else None,
        "compositorApi": ("scene.compositing_node_group (Blender 5.0+)" if era == "new"
                          else "scene.node_tree (Blender <= 4.4)"),
        "addressAs": SCENE_COMPOSITOR,
        "note": ("address this tree from add_group_node, link_group_nodes and list_group_nodes by "
                 "passing tree:'%s' - it is the scene's own tree and is not in bpy.data.node_groups."
                 % SCENE_COMPOSITOR),
        **after,
    }


def op_compositor_info(params):
    """What the compositor IS, and every way it can be on and doing nothing.

    THE WHOLE SUBSYSTEM WAS UNREACHABLE before 2026-09-03. create_node_group could make a
    CompositorNodeTree, but that is a node GROUP in bpy.data.node_groups - the scene's compositor is
    scene.node_tree, a different tree that nothing here could address. So glare, colour grading,
    denoising, cryptomatte, lens distortion, file output and every other post-process was outside
    the typed path entirely.

    FOUR WAYS TO BE ON AND INERT, each reported as a distinct blocker because the fix differs:

      use_nodes off                 no tree at all; the render is unprocessed
      use_compositing off           the tree exists and is edited and the render PIPELINE skips it.
                                    This is the classic one - the backdrop updates while the file
                                    on disk is untouched.
      no Composite node linked      the tree runs and writes nothing to the render result. A Viewer
                                    node is NOT a substitute: it feeds the backdrop only, which is
                                    why "it looks right in the compositor" and the file is wrong.
      no Render Layers feeding it   the compositor is not looking at the render at all.

    params:
      viewLayer (str)   which view layer's enabled passes to report. Default the active one.
    """
    reject_unknown(params, _COMPINFO_KEYS, "compositor_info")
    sc = bpy.context.scene
    vl_name = take(params, "viewLayer", kind=str)
    if vl_name:
        vl = sc.view_layers.get(vl_name)
        if vl is None:
            known = sorted(v.name for v in sc.view_layers)
            raise MifOpError("no view layer named '%s' in scene '%s'. Present: %s."
                             % (vl_name, sc.name, ", ".join(known)))
    else:
        vl = bpy.context.view_layer

    out = {
        "ok": True,
        "scene": sc.name,
        "useNodes": bool(sc.use_nodes),
        "useCompositing": bool(sc.render.use_compositing),
        "useSequencer": bool(sc.render.use_sequencer),
        "viewLayer": vl.name,
        "addressAs": SCENE_COMPOSITOR,
    }
    blockers = []

    era, _attr = compositor_era(sc)
    out["compositorApi"] = ("scene.compositing_node_group (Blender 5.0+)" if era == "new"
                            else "scene.node_tree (Blender <= 4.4)")
    tree = compositor_tree(sc)
    if tree is None:
        blockers.append(
            "scene.use_nodes is OFF, so there is no compositing tree and the render is written "
            "unprocessed. Turn it on with set_compositing." if era == "old" else
            "there is no compositing node group assigned, so the render is written unprocessed. On "
            "Blender 5.0 use_nodes defaults to TRUE and means nothing by itself - the tree is a "
            "real node group in scene.compositing_node_group and it is unset. Make one with "
            "set_compositing.")
        out["nodeCount"] = 0
    else:
        out["treeName"] = tree.name
        out["nodeCount"] = len(tree.nodes)
        by_type = {}
        for n in tree.nodes:
            by_type[n.bl_idname] = by_type.get(n.bl_idname, 0) + 1
        out["nodesByType"] = by_type
        # THE NODE ROWS ARE THE ONLY PART THAT GROWS, and they grow with ONE tree rather than with the file -
        # the mildest of the growing classes, but a heavy comp graph is still hundreds of rows. nodeCount above
        # stays the tree's true size, and every derived answer below (terminals, composites, what is fed) is
        # computed from tree.nodes, never from the limited rows, because those are the findings the op exists for.
        node_rows = [{"name": n.name, "type": n.bl_idname, "label": n.label,
                      "muted": bool(getattr(n, "mute", False))} for n in tree.nodes]
        node_rows = exclude_rows(node_rows, params)
        node_rows, node_extra = limit_rows(node_rows, params, "compositor node(s)")
        out["nodes"] = node_rows
        out.update({("nodes" + k[0].upper() + k[1:]): v for k, v in node_extra.items()})
        out["linkCount"] = len(tree.links)

        # THE OUTPUT NODE IS ERA-DEPENDENT, and hard-coding CompositorNodeComposite made this
        # report a CORRECTLY WIRED 5.0 compositor as broken - a wrong answer, not a missing one,
        # from the field whose whole job is telling you the tree is inert. Caught by running the op
        # headless on 5.0 after the accessor was fixed, not by reading. Same fix _terminals already
        # applies to list_group_nodes, reused here rather than restated.
        terminals, _term_note = _terminals(tree)
        composites = [n for n in tree.nodes if n.bl_idname in terminals]
        viewers = [n for n in tree.nodes if n.bl_idname == "CompositorNodeViewer"]
        rlayers = [n for n in tree.nodes if n.bl_idname == "CompositorNodeRLayers"]
        # ONLY VALID LINKS FEED ANYTHING. An invalid link - the shape set_compositing itself left on 5.x
        # until 2026-09-16, into a Group Output's empty extend socket - is listed in tree.links and carries
        # nothing. Counting it reported that tree as connected with no blocker while every render came out
        # unprocessed. Invalid links are named as their own blocker below.
        live = [l for l in tree.links if l.is_valid and getattr(l.to_socket, "identifier", "") != "__extend__"]
        dead = [l for l in tree.links if l not in live]
        fed = {l.to_node.name for l in live}
        out["compositeNodes"] = [n.name for n in composites]
        out["compositeConnected"] = any(n.name in fed for n in composites)
        out["renderLayersNodes"] = [n.name for n in rlayers]
        out["renderLayersFeeding"] = any(l.from_node.bl_idname == "CompositorNodeRLayers" for l in live)
        out["invalidLinks"] = ["%s.%s -> %s.%s" % (l.from_node.name, l.from_socket.name, l.to_node.name,
                                                   l.to_socket.name or l.to_socket.identifier) for l in dead]
        if dead:
            blockers.append("%d link(s) are INVALID and carry nothing: %s. On Blender 5.x a link into the Group "
                            "Output's empty extend socket is this shape - declare the output first with "
                            "add_group_interface {tree:'%s', inOut:'OUTPUT', name:'Image', socketType:"
                            "'NodeSocketColor'} and link to Image."
                            % (len(dead), "; ".join(out["invalidLinks"][:4]), SCENE_COMPOSITOR))
        out["mutedNodes"] = [n.name for n in tree.nodes if getattr(n, "mute", False)]

        out["outputNodeTypes"] = sorted(terminals)
        if not composites:
            blockers.append("there is no %s in the tree, so the compositor writes "
                            "NOTHING to the render result%%s."
                            % ("Group Output node" if era == "new" else "Composite node")
                            % (" - the Viewer node feeds the backdrop only, which is why it looks "
                               "right in the compositor and the saved file is wrong" if viewers
                               else ""))
        elif not out["compositeConnected"]:
            blockers.append(("the %s exists but nothing is linked into it, so the compositor "
                             "writes nothing to the render result%%s."
                             % ("Group Output node" if era == "new" else "Composite node"))
                            % (" - a Viewer node is connected, and that feeds the backdrop only"
                               if any(n.name in fed for n in viewers) else ""))
        if not rlayers:
            blockers.append("there is no Render Layers node, so the compositor is not looking at "
                            "the render at all.")
        elif not out["renderLayersFeeding"]:
            blockers.append("the Render Layers node is not linked to anything, so the render is "
                            "not entering the compositor.")
        if out["mutedNodes"]:
            blockers.append("%d node(s) are MUTED and pass their input straight through: %s."
                            % (len(out["mutedNodes"]), ", ".join(out["mutedNodes"][:8])))

    # GUARDED ON THE TREE, NOT ON use_nodes, because on 5.0 use_nodes is TRUE out of the box and
    # would make this fire on every fresh scene that has no compositor at all.
    if tree is not None and not sc.render.use_compositing:
        blockers.append("scene.render.use_compositing is OFF, so the render pipeline SKIPS the "
                        "compositor entirely. The tree still exists, the backdrop still updates, "
                        "and the file on disk is completely unprocessed - the two switches are "
                        "independent and this is the one that is usually missed.")
    if sc.render.use_sequencer:
        # NOT A BLOCKER BY ITSELF. The VSE only overrides the compositor when strips exist, and a
        # scene with an empty sequencer is the normal case - calling that a blocker would train
        # people to ignore the list.
        seq = getattr(sc, "sequence_editor", None)
        # strips_all FROM 4.4, sequences_all BEFORE, by capability. Read as sequences_all alone this counted ZERO on
        # 5.0 and 5.2, where that name is gone, and dropped the blocker below with a strip right there - found
        # 2026-09-23 when the version matrix gained a sound strip fixture and 3.6-4.4 said 1 while 5.0-5.2 said 0.
        strips = (len(seq.strips_all if hasattr(seq, "strips_all") else seq.sequences_all)
                  if seq is not None else 0)
        out["sequencerStrips"] = strips
        if strips:
            blockers.append("use_sequencer is on and the VSE holds %d strip(s). The sequencer runs "
                            "AFTER the compositor and its output is what gets written, so the "
                            "compositor's result can be replaced wholesale." % strips)

    passes = {}
    for attr in dir(vl):
        if attr.startswith("use_pass_"):
            try:
                passes[attr[len("use_pass_"):]] = bool(getattr(vl, attr))
            except (AttributeError, TypeError):     # noqa: PERF203
                continue
    out["enabledPasses"] = sorted(k for k, v in passes.items() if v)
    out["availablePasses"] = sorted(passes)

    out["blockers"] = blockers
    out["compositorAffectsRender"] = not blockers
    return out


# ===========================================================================
# build_pass_composite - the light passes summed back into the image (2026-10-06)
# ===========================================================================
# MEASURED 2026-10-06 on 4.2.23, 4.4.0, 5.0.1 and 5.2.2 (scratchpad comp2.py / comp3.py): the Render Layers sockets are
# DiffDir, GlossInd, Emit, Env, AO ... before 5.0 and "Diffuse Direct", "Glossy Indirect", "Emission", "Environment",
# "Ambient Occlusion" from it; the colour mix is CompositorNodeMixRGB (Fac, Image, Image) before 5.0 and ShaderNodeMix
# (data_type RGBA: Factor_Float, A_Color, B_Color -> Result_Color) from it; the ramp CompositorNodeValToRGB before and
# ShaderNodeValToRGB from 5.0; Glare's type and quality are properties to 4.4 and menu sockets from 5.0; a custom group
# is CompositorNodeGroup on all four. A 5.x compositor needs a GPU context to execute (5.2.2 aborts headless on
# libEGL), so the image is compared with the Combined pass on 4.2 and 4.4 only.
_BPC_KEYS = {"viewLayer", "passes", "group", "denoise", "extras", "output", "filePath", "fileFormat"}
_BPC_LIGHT = {
    # pass: ((Render Layers socket before 5.0, from 5.0), view-layer switch) for direct, indirect, color
    "DIFFUSE": ((("DiffDir", "Diffuse Direct"), "diffuse_direct"), (("DiffInd", "Diffuse Indirect"), "diffuse_indirect"),
                (("DiffCol", "Diffuse Color"), "diffuse_color")),
    "GLOSSY": ((("GlossDir", "Glossy Direct"), "glossy_direct"), (("GlossInd", "Glossy Indirect"), "glossy_indirect"),
               (("GlossCol", "Glossy Color"), "glossy_color")),
    "TRANSMISSION": ((("TransDir", "Transmission Direct"), "transmission_direct"),
                     (("TransInd", "Transmission Indirect"), "transmission_indirect"),
                     (("TransCol", "Transmission Color"), "transmission_color")),
    "VOLUME": ((("VolumeDir", "Volume Direct"), "cycles.volume_direct"),
               (("VolumeInd", "Volume Indirect"), "cycles.volume_indirect"), None),
}
_BPC_SINGLE = {"EMISSION": (("Emit", "Emission"), "emit"), "ENVIRONMENT": (("Env", "Environment"), "environment"),
               "AO": (("AO", "Ambient Occlusion"), "ambient_occlusion"), "MIST": (("Mist", "Mist"), "mist")}
_BPC_ORDER = ("DIFFUSE", "GLOSSY", "TRANSMISSION", "VOLUME", "EMISSION", "ENVIRONMENT")
_BPC_MIXER = "MifPassMixer"


def _bpc_on(vl, switch):
    holder, _, name = switch.rpartition(".")
    owner = getattr(vl, holder) if holder else vl
    return bool(getattr(owner, "use_pass_" + name, False))


def _bpc_socket(rl, names):
    return next((o for o in rl.outputs if o.name in names and o.enabled), None)


class _Builder:
    """Nodes and links made in one tree, for the reply and for taking them out again on a failure."""

    def __init__(self, tree, era):
        self.tree, self.era, self.nodes, self.links = tree, era, [], []
        self.x = 0

    def node(self, kind, label=None):
        n = self.tree.nodes.new(kind)
        if label:
            n.name = n.label = label        # Blender adds .001 to a name already taken; the label stays as given
        n.location = (self.x, -120 * (len(self.nodes) % 12))
        self.x += 40
        self.nodes.append(n)
        return n

    def link(self, out, inp):
        lk = self.tree.links.new(out, inp)
        self.links.append("%s.%s -> %s.%s" % (out.node.name, out.name, inp.node.name, inp.name or inp.identifier))
        return lk

    def mix(self, blend, a, b, fac=None, label=None):
        """A colour mix: (node output). a / b are sockets or colours; fac a socket or a number (default 1)."""
        if self.era == "new":
            n = self.node("ShaderNodeMix", label)
            n.data_type = "RGBA"
            n.blend_type = blend
            ins = {i.identifier: i for i in n.inputs}          # by identifier: the three pairs share their names
            fi, ai, bi = ins["Factor_Float"], ins["A_Color"], ins["B_Color"]
            out = next(o for o in n.outputs if o.identifier == "Result_Color")
        else:
            n = self.node("CompositorNodeMixRGB", label)
            n.blend_type = blend
            fi, ai, bi, out = n.inputs[0], n.inputs[1], n.inputs[2], n.outputs[0]
        for sock, val in ((fi, 1.0 if fac is None else fac), (ai, a), (bi, b)):
            if hasattr(val, "is_output"):
                self.link(val, sock)
            else:
                sock.default_value = val
        return out

    def ramp(self, source, stops, interpolation=None, label=None):
        n = self.node("ShaderNodeValToRGB" if self.era == "new" else "CompositorNodeValToRGB", label)
        els = n.color_ramp.elements
        els[0].position, els[1].position = stops
        if interpolation:
            n.color_ramp.interpolation = interpolation
        self.link(source, n.inputs[0])
        return n.outputs[0]

    def undo(self):
        for n in reversed(self.nodes):
            try:
                self.tree.nodes.remove(n)
            except (RuntimeError, ReferenceError):
                pass


def _bpc_mixer(era):
    """The reusable pass mixer: (direct + indirect) * color, inputs direct and indirect default black, color white, the
    values hidden on the inputs - an unconnected input adds nothing and multiplies by white. Reused when it exists."""
    g = bpy.data.node_groups.get(_BPC_MIXER)
    if g is not None and g.bl_idname == "CompositorNodeTree":
        return g, False
    g = bpy.data.node_groups.new(_BPC_MIXER, "CompositorNodeTree")
    b = _Builder(g, era)
    try:
        # _iface_new: tree.interface from 4.0, tree.inputs/outputs on 3.6. The bare interface call raised on 3.6.23 AFTER
        # the group was made, so the socketless group stayed and every later build "reused" it and failed on its
        # missing inputs (found 2026-10-10); the sockets are now inside the try that removes a half-made group.
        for nm, val in (("direct", (0.0, 0.0, 0.0, 1.0)), ("indirect", (0.0, 0.0, 0.0, 1.0)),
                        ("color", (1.0, 1.0, 1.0, 1.0))):
            sock = _iface_new(g, nm, "NodeSocketColor", "INPUT")
            sock.default_value = val
            sock.hide_value = True
        _iface_new(g, "Image", "NodeSocketColor", "OUTPUT")
        gi, go = b.node("NodeGroupInput"), b.node("NodeGroupOutput")
        total = b.mix("ADD", gi.outputs["direct"], gi.outputs["indirect"])
        b.link(b.mix("MULTIPLY", total, gi.outputs["color"]), go.inputs["Image"])
    except Exception:
        bpy.data.node_groups.remove(g)
        raise
    return g, True


def _bpc_put_back(tree, removed_links, made_socket):
    """build_pass_composite's rollback beyond the nodes: the links it removed from the output, and the Image interface
    socket it declared. A link whose ends are gone is skipped - an end the undo removed was a node this call made."""
    for from_name, from_id, to_name, to_id in removed_links:
        src, dst = tree.nodes.get(from_name), tree.nodes.get(to_name)
        if src is None or dst is None:
            continue
        out = next((s for s in src.outputs if s.identifier == from_id), None)
        inp = next((s for s in dst.inputs if s.identifier == to_id), None)
        if out is not None and inp is not None:
            tree.links.new(out, inp)
    if made_socket is not None:
        try:
            tree.interface.remove(made_socket)
        except (ReferenceError, RuntimeError):
            pass


def op_build_pass_composite(params):
    """Rebuild the image from its light passes in the scene compositor - 20 or more links per pass by hand before this.

    THE FORMULA: for diffuse, glossy and transmission, (Direct + Indirect) x Color; volume Direct + Indirect (no Color
    pass); the results added in that order, then Emission, then Environment - the order matters or the chain breaks.
    Through a reusable "MifPassMixer" group (direct, indirect, color; black, black, white, values hidden) unless group is
    false. passes (DIFFUSE, GLOSSY, TRANSMISSION, VOLUME, EMISSION, ENVIRONMENT) default to every one this view layer
    outputs in full; one it does not output is refused naming set_view_layer's switch. denoise NONE | FINAL (one Denoise
    at the end) | PER_PASS (one after each light pass), Albedo and Normal wired when the layer stores denoising data.
    extras {ao {ramp [black, white], cryptomatteObject} - the AO pass through a ramp multiplied over the result, limited
    to one object's Cryptomatte matte; emissionGlare - Fog Glow on Emission; mist {ramp [start, end], color} - the Mist
    pass through a B-spline ramp adding a fog colour}. output COMPOSITE (default; the Group Output's Image from 5.0) |
    VIEWER | FILE_OUTPUT (filePath, fileFormat). Every node and link made comes back; on a failure they are removed.
    """
    reject_unknown(params, _BPC_KEYS, "build_pass_composite")
    tail = "NOTHING was built."
    sc = bpy.context.scene
    vl_name = take(params, "viewLayer", default=None, kind=str)
    vl = sc.view_layers.get(vl_name) if vl_name else bpy.context.view_layer
    if vl is None:
        raise MifOpError("scene '%s' has no view layer '%s' (it has %s). %s"
                         % (sc.name, vl_name, ", ".join(v.name for v in sc.view_layers), tail))
    use_group = take_bool(params, "group", default=True)
    denoise = str(take(params, "denoise", default="NONE", kind=str)).upper()
    if denoise not in ("NONE", "FINAL", "PER_PASS"):
        raise MifOpError("denoise is NONE, FINAL or PER_PASS, got %r. %s" % (params.get("denoise"), tail))
    output = str(take(params, "output", default="COMPOSITE", kind=str)).upper()
    if output not in ("COMPOSITE", "VIEWER", "FILE_OUTPUT"):
        raise MifOpError("output is COMPOSITE, VIEWER or FILE_OUTPUT, got %r. %s" % (params.get("output"), tail))
    file_path = take(params, "filePath", default=None, kind=str)
    file_format = take(params, "fileFormat", default=None, kind=str)
    if output != "FILE_OUTPUT" and (file_path is not None or file_format is not None):
        raise MifOpError("filePath and fileFormat go with output FILE_OUTPUT. %s" % tail)
    if output == "FILE_OUTPUT" and not file_path:
        raise MifOpError("output FILE_OUTPUT needs filePath - the folder (and name) the node writes to. %s" % tail)
    if file_format is not None:
        formats = [i.identifier for i in bpy.types.ImageFormatSettings.bl_rna.properties["file_format"].enum_items]
        if file_format.upper() not in formats:
            raise MifOpError("fileFormat %r is not one of %s. %s" % (file_format, ", ".join(formats), tail))
        file_format = file_format.upper()

    tree = _scene_tree()

    def missing(switches):
        return ["cycles {use_pass_%s}" % sw.split(".")[1] if "." in sw else sw for sw in switches
                if sw and not _bpc_on(vl, sw)]
    raw = params.get("passes")
    if raw is None:
        passes = [pk for pk in _BPC_ORDER if not missing(
            [c[1] for c in _BPC_LIGHT[pk] if c] if pk in _BPC_LIGHT else [_BPC_SINGLE[pk][1]])]
        if not passes:
            raise MifOpError("view layer '%s' outputs no light pass in full - set_view_layer enablePasses "
                             "diffuse_direct, diffuse_indirect, diffuse_color (and the glossy, transmission, emit, "
                             "environment ones). %s" % (vl.name, tail))
    else:
        if not isinstance(raw, (list, tuple)) or not raw or any(not isinstance(x, str) for x in raw):
            raise MifOpError("passes is a list of %s. %s" % (", ".join(_BPC_ORDER), tail))
        want = [x.upper() for x in raw]
        bad = [x for x in want if x not in _BPC_ORDER]
        if bad:
            raise MifOpError("passes has %s; it takes %s. %s" % (", ".join(bad), ", ".join(_BPC_ORDER), tail))
        off = {pk: missing([c[1] for c in _BPC_LIGHT[pk] if c] if pk in _BPC_LIGHT else [_BPC_SINGLE[pk][1]])
               for pk in want}
        off = {k: v for k, v in off.items() if v}
        if off:
            raise MifOpError("view layer '%s' does not output %s - set_view_layer enablePasses %s first. %s"
                             % (vl.name, ", ".join(sorted(off)), ", ".join(x for v in off.values() for x in v), tail))
        passes = [pk for pk in _BPC_ORDER if pk in want]
    extras = params.get("extras") or {}
    if not isinstance(extras, dict) or set(extras) - {"ao", "emissionGlare", "mist"}:
        raise MifOpError("extras is {ao {ramp, cryptomatteObject}, emissionGlare, mist {ramp, color}}; got %r. %s"
                         % (extras, tail))
    ao = extras.get("ao")
    crypto = None
    if ao is not None:
        if not isinstance(ao, dict) or set(ao) - {"ramp", "cryptomatteObject"}:
            raise MifOpError("extras.ao is {ramp [black, white], cryptomatteObject}. %s" % tail)
        if not _bpc_on(vl, "ambient_occlusion"):
            raise MifOpError("extras.ao reads the AO pass - set_view_layer enablePasses ambient_occlusion. %s" % tail)
        if ao.get("cryptomatteObject") is not None:
            crypto = get_object(take(ao, "cryptomatteObject", kind=str))
            if not _bpc_on(vl, "cryptomatte_object"):
                raise MifOpError("cryptomatteObject reads the object Cryptomatte - set_view_layer enablePasses "
                                 "cryptomatte_object. %s" % tail)
    glare = take_bool(extras, "emissionGlare", default=False)
    if glare and "EMISSION" not in passes:
        raise MifOpError("emissionGlare acts on the Emission pass, which this build is not summing. %s" % tail)
    mist = extras.get("mist")
    if mist is not None:
        if not isinstance(mist, dict) or set(mist) - {"ramp", "color"}:
            raise MifOpError("extras.mist is {ramp [start, end], color [r, g, b]}. %s" % tail)
        if not _bpc_on(vl, "mist"):
            raise MifOpError("extras.mist reads the Mist pass - set_view_layer enablePasses mist. %s" % tail)

    def stops(spec, key, default):
        val = spec.get(key, default) if isinstance(spec, dict) else default
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            raise MifOpError("%s is [first stop, second stop] between 0 and 1. %s" % (key, tail))
        a, b = finite_floats(list(val), key)
        if not 0.0 <= a < b <= 1.0:
            raise MifOpError("%s %s must rise inside 0-1. %s" % (key, [a, b], tail))
        return a, b
    ao_stops = stops(ao, "ramp", [0.0, 1.0]) if ao is not None else None
    mist_stops = stops(mist, "ramp", [0.0, 1.0]) if mist is not None else None
    fog = None
    if mist is not None:
        c = mist.get("color", [1.0, 1.0, 1.0])
        if not isinstance(c, (list, tuple)) or len(c) not in (3, 4):
            raise MifOpError("extras.mist.color is [r, g, b]. %s" % tail)
        fog = tuple(finite_floats(list(c)[:3], "mist.color")) + (1.0,)
    era, _attr = compositor_era(sc)
    den_data = bool(getattr(vl.cycles, "denoising_store_passes", False)) if hasattr(vl, "cycles") else False

    b = _Builder(tree, era)
    made_group = False
    # WHAT THE ROLLBACK PUTS BACK BESIDES THE NODES: the links a COMPOSITE output removed from the Composite node or
    # Group Output, and the Image interface socket a 5.0+ Group Output was given. b.undo() removes only the nodes it
    # added, and the refusal says NOTHING was built (found 2026-10-25).
    removed_links, made_socket = [], None
    try:
        rl = next((n for n in tree.nodes if n.bl_idname == "CompositorNodeRLayers" and n.layer == vl.name), None)
        if rl is None:
            rl = b.node("CompositorNodeRLayers", vl.name)
            rl.layer = vl.name
        mixer = None
        if use_group:
            mixer, made_group = _bpc_mixer(era)

        def sock(names):
            out = _bpc_socket(rl, names)
            if out is None:
                raise MifOpError("the Render Layers node has no enabled output %s" % " / ".join(names))
            return out

        def denoised(src, label):
            d = b.node("CompositorNodeDenoise", label)
            b.link(src, d.inputs["Image"])
            if den_data:
                b.link(sock(("Denoising Albedo",)), d.inputs["Albedo"])
                b.link(sock(("Denoising Normal",)), d.inputs["Normal"])
            return d.outputs["Image"]

        terms = []
        for pk in passes:
            if pk in _BPC_LIGHT:
                comps = _BPC_LIGHT[pk]
                if mixer is not None:
                    gn = b.node("CompositorNodeGroup", pk.title())
                    gn.node_tree = mixer
                    for (names, _sw), key in zip([c for c in comps if c], ("direct", "indirect", "color")):
                        b.link(sock(names), gn.inputs[key])
                    out = gn.outputs["Image"]
                else:
                    out = b.mix("ADD", sock(comps[0][0]), sock(comps[1][0]), label=pk.title() + " light")
                    if comps[2]:
                        out = b.mix("MULTIPLY", out, sock(comps[2][0]), label=pk.title())
                if denoise == "PER_PASS":
                    out = denoised(out, "Denoise " + pk.title())
            else:
                out = sock(_BPC_SINGLE[pk][0])
                if pk == "EMISSION" and glare:
                    g = b.node("CompositorNodeGlare", "Emission glare")
                    if "glare_type" in g.bl_rna.properties:
                        g.glare_type, g.quality = "FOG_GLOW", "HIGH"
                    else:
                        g.inputs["Type"].default_value, g.inputs["Quality"].default_value = "Fog Glow", "High"
                    b.link(out, g.inputs["Image"])
                    out = g.outputs["Image"]
            terms.append(out)
        result = terms[0]
        for t in terms[1:]:
            result = b.mix("ADD", result, t, label="Sum")
        if ao is not None:
            shade = b.ramp(sock(_BPC_SINGLE["AO"][0]), ao_stops, label="AO ramp")
            fac = 1.0
            if crypto is not None:
                cm = b.node("CompositorNodeCryptomatteV2", "AO matte")
                cm.source = "RENDER"
                if "scene" in cm.bl_rna.properties:
                    cm.scene = sc
                cm.layer_name = vl.name + ".CryptoObject"     # the layer's object Cryptomatte, by its enum id
                cm.matte_id = crypto.name
                b.link(result, cm.inputs["Image"])
                fac = cm.outputs["Matte"]
            result = b.mix("MULTIPLY", result, shade, fac=fac, label="AO")
        if mist is not None:
            haze = b.ramp(sock(_BPC_SINGLE["MIST"][0]), mist_stops, "B_SPLINE", label="Mist ramp")
            result = b.mix("ADD", result, fog, fac=haze, label="Mist")
        if denoise == "FINAL":
            result = denoised(result, "Denoise")
        if output == "COMPOSITE":
            if era == "new":
                target_node = next((n for n in tree.nodes if n.bl_idname == "NodeGroupOutput"), None) \
                    or b.node("NodeGroupOutput")
                had = {i.identifier for i in tree.interface.items_tree if getattr(i, "item_type", "") == "SOCKET"}
                target = _group_output_image(tree, target_node)
                made_socket = next((i for i in tree.interface.items_tree if getattr(i, "item_type", "") == "SOCKET"
                                    and i.identifier not in had), None)
            else:
                target_node = next((n for n in tree.nodes if n.bl_idname == "CompositorNodeComposite"), None) \
                    or b.node("CompositorNodeComposite")
                target = target_node.inputs["Image"]
            replaced = [lk.from_node.name for lk in target.links]
            for lk in list(target.links):
                removed_links.append((lk.from_node.name, lk.from_socket.identifier, target_node.name,
                                      target.identifier))
                tree.links.remove(lk)
        elif output == "VIEWER":
            target_node, replaced = b.node("CompositorNodeViewer"), []
            target = target_node.inputs["Image"]
        else:
            target_node, replaced = b.node("CompositorNodeOutputFile", "Pass composite"), []
            if "file_output_items" in target_node.bl_rna.properties:
                target_node.directory = os.path.dirname(file_path) or "//"
                target_node.file_name = os.path.basename(file_path) or "composite"
                if not len(target_node.file_output_items):
                    target_node.file_output_items.new("RGBA", "Image")
            else:
                target_node.base_path = file_path
            if file_format:
                # 5.0+ files a File Output under media type MULTI_LAYER_IMAGE, whose only format is a multilayer EXR;
                # a single-image format needs media type IMAGE first (measured 5.0.1, 5.2.2)
                if "media_type" in target_node.format.bl_rna.properties and file_format != "OPEN_EXR_MULTILAYER":
                    target_node.format.media_type = "IMAGE"
                target_node.format.file_format = file_format
            target = next(s for s in target_node.inputs if s.enabled)
        lk = b.link(result, target)
        if not lk.is_valid:
            raise MifOpError("Blender marked the link into %s invalid" % target_node.name)
    except MifOpError as exc:
        b.undo()
        _bpc_put_back(tree, removed_links, made_socket)
        if made_group:
            bpy.data.node_groups.remove(bpy.data.node_groups[_BPC_MIXER])
        raise MifOpError("%s. The nodes this call had added were removed - %s" % (str(exc).rstrip("."), tail))
    except Exception as exc:  # noqa: BLE001 - anything Blender raises mid-build takes the half-built chain out again
        b.undo()
        _bpc_put_back(tree, removed_links, made_socket)
        if made_group:
            bpy.data.node_groups.remove(bpy.data.node_groups[_BPC_MIXER])
        raise MifOpError("building the pass composite failed: %s: %s. The nodes this call had added were removed - %s"
                         % (type(exc).__name__, exc, tail))
    return {"scene": sc.name, "viewLayer": vl.name, "tree": tree.name, "addressAs": SCENE_COMPOSITOR,
            "compositorApi": "scene.compositing_node_group" if era == "new" else "scene.node_tree",
            "passes": passes, "group": mixer.name if mixer is not None else None, "groupCreated": made_group,
            "denoise": denoise, "denoiseAlbedoNormal": den_data if denoise != "NONE" else None,
            "denoiseNote": (None if denoise == "NONE" or den_data else
                            "the layer stores no denoising data, so only Image is wired - set_view_layer cycles "
                            "{denoising_store_passes: true} gives the Denoise nodes Albedo and Normal"),
            "extras": sorted(k for k in ("ao", "mist") if extras.get(k) is not None) + (["emissionGlare"] if glare
                                                                                      else []),
            "output": output, "outputNode": target_node.name, "replacedLinkFrom": replaced,
            "nodesMade": [n.name for n in b.nodes], "linksMade": b.links, "linkCount": len(b.links),
            "note": ("the sum equals the Combined pass up to noise; compare the two before trusting a grade. A 5.x "
                     "compositor runs on the GPU - headless without one it does not execute.")}

# ===========================================================================
# RUN A NODE TOOL (2026-10-06)
# ===========================================================================
# A node group marked as a TOOL (4.0+) runs on an object's geometry once, the way a menu operator does, instead of
# living on the stack as a modifier. Nothing ran one. MEASURED on the five installs:
#
#   3.6.23                  no node tools (GeometryNodeTree has no is_tool)
#   4.2.17, 4.4.0, 5.0.1    geometry.execute_node_group(name=...). Its inputs are not operator keywords ("keyword
#                           Socket_2 unrecognized"); run without them it reads the group INTERFACE defaults, so inputs
#                           go in as temporary interface defaults and are put back after. In EDIT mode it gathers the
#                           objects through CTX_wm_view3d and, with no 3D view, finds none and returns FINISHED having
#                           changed nothing; under temp_override(window, VIEW_3D area, region) it runs and respects
#                           the edit-mode selection.
#   5.2.1                   no execute_node_group: each tool registers its own operator from node_tool_idname, and the
#                           registration runs only from the window manager's notifier loop (wm_event_do_notifiers), which a
#                           headless Blender never runs - node_tool_idname set, asset_mark, save and reopen all left
#                           bpy.ops.geometry without it. So 5.1+ runs the tool natively: a copy of the group as a Nodes
#                           modifier, first in the stack, applied. A Tool Selection node in a modifier evaluates to
#                           nothing useful ("Node must be run as tool"), so in the copy it reads a temporary point
#                           attribute written from the edit-mode vertex selection. Other tool-only nodes (Set Selection,
#                           Active Element, Face Set, 3D Cursor, Mouse Position, Viewport Transform) have no such stand-in
#                           and are refused on that route.
_TOOL_KEYS = {"object", "tool", "group", "inputs", "mode", "library"}
_TOOL_SELECTION = "GeometryNodeToolSelection"
_TOOL_SELECTION_ATTR = "mif_tool_selection"
_TOOL_TYPE_FLAGS = {"MESH": ("is_type_mesh",), "CURVES": ("is_type_curve",),
                    "POINTCLOUD": ("is_type_pointcloud", "is_type_point_cloud")}


def _tool_only_nodes(tree, seen=None, depth=0):
    """[(group name, node name, bl_idname, nested)] for every tool-only node, nested groups included."""
    seen = seen if seen is not None else set()
    if tree is None or tree.name in seen:
        return []
    seen.add(tree.name)
    out = []
    for n in tree.nodes:
        if n.bl_idname.startswith("GeometryNodeTool"):
            out.append((tree.name, n.name, n.bl_idname, depth > 0))
        sub = getattr(n, "node_tree", None)
        if sub is not None:
            out.extend(_tool_only_nodes(sub, seen, depth + 1))
    return out


def _tool_counts(obj):
    """The evaluated element counts: vertices, edges, faces for a mesh; points (and curves) for curves and points."""
    try:
        if obj.mode == "EDIT":
            obj.update_from_editmode()
        ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        if obj.type == "MESH":
            me = ev.to_mesh()
            out = {"vertices": len(me.vertices), "edges": len(me.edges), "faces": len(me.polygons), "points": None}
            ev.to_mesh_clear()
            return out
        data = ev.data
        return {"vertices": None, "edges": None, "faces": None, "points": len(data.points),
                "curves": len(data.curves) if hasattr(data, "curves") else None}
    except Exception:                                              # noqa: BLE001 - a report must not break the op
        return {}


def _view3d_override(obj):
    """A context with a 3D view in it (the edit-mode gather needs one): an existing VIEW_3D area, else None."""
    wm = bpy.context.window_manager
    for win in (wm.windows if wm is not None else []):
        for area in win.screen.areas if win.screen else []:
            if area.type == "VIEW_3D":
                region = next((r for r in area.regions if r.type == "WINDOW"), None)
                return {"window": win, "screen": win.screen, "area": area, "region": region}
    return None


def _set_mode(obj, mode):
    """obj into OBJECT or EDIT mode, and only obj.

    THE VIEW LAYER'S ACTIVE OBJECT IS WHAT mode_set SWITCHES, whatever the override says: with another object active
    (the startup Cube) the override's active_object did not reach it on 5.2.1, that object went into edit mode instead,
    and the next modifier_apply was refused "not allowed from Edit mode" (measured 2026-10-07). So obj is made active
    for the switch; op_run_node_tool puts the scene's own active object back at the end."""
    if obj.mode == mode:
        return
    bpy.context.view_layer.objects.active = obj
    with bpy.context.temp_override(active_object=obj, object=obj, selected_objects=[obj],
                                   selected_editable_objects=[obj]):
        bpy.ops.object.mode_set(mode=mode)


def _plan_tool_inputs(tree, given, tail):
    """{identifier: (name, kind, value)} for the inputs given, every one checked before anything runs."""
    if given is None:
        return {}
    if not isinstance(given, dict):
        raise MifOpError("inputs is {input name or identifier: value}, got %r. %s" % (given, tail))
    keys, plan = _input_keys(tree), {}
    for k, v in given.items():
        if k in keys and keys[k] is None:
            raise MifOpError("two or more inputs are named %r - key it by identifier: %s. %s" % (k, ", ".join(
                e[0] for e in keys.values() if e is not None and e[2] == k), tail))
        entry = keys.get(k)
        if entry is None:
            raise MifOpError("%r is not an input of '%s'; it has: %s. %s" % (k, tree.name, ", ".join(
                sorted({e[2] for e in keys.values() if e is not None and "Geometry" not in e[1]})) or "(none)", tail))
        ident, kind, name = entry
        if "Geometry" in kind:
            raise MifOpError("'%s' is the tool's geometry input - the object's own geometry goes there. %s"
                             % (name, tail))
        if isinstance(v, dict):
            raise MifOpError("'%s': a tool input takes a value, not an attribute. %s" % (name, tail))
        value = v if kind == "NodeSocketMenu" else _socket_value(kind, name, v, "tool '%s'" % tree.name, tail)
        plan[ident] = (name, kind, value)
    return plan


def _iface_input(tree, ident):
    if hasattr(tree, "interface"):
        return next((it for it in tree.interface.items_tree if getattr(it, "item_type", "SOCKET") == "SOCKET"
                     and it.in_out == "INPUT" and it.identifier == ident), None)
    return next((s for s in tree.inputs if s.identifier == ident), None)


def _run_by_operator(obj, tree, mode, plan, tail):
    """4.x-5.0: geometry.execute_node_group with the inputs as temporary interface defaults."""
    others = [o.name for o in bpy.context.view_layer.objects if o.mode == "EDIT" and o != obj]
    if mode == "EDIT" and others:
        raise MifOpError("in EDIT mode a tool runs on every object being edited, and %s %s in edit mode too. Leave "
                         "edit mode on them first. %s" % (", ".join(others), "is" if len(others) == 1 else "are", tail))
    over = _view3d_override(obj)
    if mode == "EDIT" and over is None:
        raise MifOpError("this Blender has no 3D view, and execute_node_group finds no object to edit without one "
                         "(it returns FINISHED and changes nothing). %s" % tail)
    saved, applied = [], {}
    try:
        for ident, (name, kind, value) in plan.items():
            item = _iface_input(tree, ident)
            saved.append((item, item.default_value if not hasattr(item.default_value, "__len__")
                          or isinstance(item.default_value, str) else tuple(item.default_value)))
            try:
                item.default_value = value
            except (TypeError, ValueError) as exc:
                raise MifOpError("input '%s' would not take %r: %s. %s" % (name, value, exc, tail))
            got = item.default_value
            applied[name] = (got if isinstance(got, (str, int, float, bool)) or got is None
                             else getattr(got, "name", None) or [round(float(x), 6) for x in got])
        _set_mode(obj, mode)
        ctx = dict(over or {}, active_object=obj, object=obj, selected_objects=[obj], selected_editable_objects=[obj])
        with bpy.context.temp_override(**ctx):
            ret = bpy.ops.geometry.execute_node_group(name=tree.name)
    finally:
        for item, old in saved:
            try:
                item.default_value = old
            except (TypeError, ValueError):
                pass
    if "FINISHED" not in ret:
        raise MifOpError("geometry.execute_node_group returned %s for '%s'. %s" % (sorted(ret), tree.name, tail))
    return applied


def _run_as_modifier(obj, tree, mode, plan, tail):
    """5.1+: the tool's group applied as a Nodes modifier, its Tool Selection reading the edit-mode selection."""
    reads_selection = any(t[2] == _TOOL_SELECTION for t in _tool_only_nodes(tree))
    if reads_selection and obj.type != "MESH":
        raise MifOpError("'%s' reads the edit-mode selection, and on this build (no execute_node_group) the selection "
                         "is carried into the tool only for a MESH; this is a %s. %s" % (tree.name, obj.type, tail))
    if obj.data.users > 1:
        raise MifOpError("'%s' shares its data with %d other object(s), and running a tool rewrites it for all of "
                         "them. %s" % (obj.name, obj.data.users - 1, tail))
    _set_mode(obj, "OBJECT")
    tmp = tree.copy()
    attr = None
    mod = None
    try:
        for n in [n for n in tmp.nodes if n.bl_idname == _TOOL_SELECTION]:
            for out in n.outputs:
                for link in list(out.links):
                    na = tmp.nodes.new("GeometryNodeInputNamedAttribute")
                    na.data_type = "BOOLEAN" if out.type == "BOOLEAN" else "FLOAT"
                    na.inputs["Name"].default_value = _TOOL_SELECTION_ATTR
                    tmp.links.new(na.outputs["Attribute"], link.to_socket)
            tmp.nodes.remove(n)
        if reads_selection:
            attr = obj.data.attributes.new(_TOOL_SELECTION_ATTR, "BOOLEAN", "POINT")
            sel = [False] * len(obj.data.vertices)
            obj.data.vertices.foreach_get("select", sel)
            attr.data.foreach_set("value", sel)
        mod = obj.modifiers.new(name="MifNodeTool", type="NODES")
        if mod is None:
            raise MifOpError("'%s' is a %s, which takes no Nodes modifier. %s" % (obj.name, obj.type, tail))
        mod.node_group = tmp
        applied = {}
        for ident, (name, kind, value) in plan.items():
            if kind == "NodeSocketMenu":
                applied[name] = _set_menu_input(mod, ident, name, value)
            else:
                _set_modifier_socket(mod, ident, value)
                got = _get_modifier_socket(mod, ident)
                applied[name] = getattr(got, "name", got)
                if hasattr(got, "__len__") and not isinstance(got, str):
                    applied[name] = [round(float(x), 6) for x in got]
        while obj.modifiers[0] != mod:
            with bpy.context.temp_override(object=obj, active_object=obj):
                bpy.ops.object.modifier_move_up(modifier=mod.name)
        with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj],
                                       selected_editable_objects=[obj]):
            bpy.ops.object.modifier_apply(modifier=mod.name)
        mod = None
    except MifOpError:
        raise
    except Exception as exc:                                       # noqa: BLE001
        raise MifOpError("running '%s' as a Nodes modifier failed: %s. %s" % (tree.name, exc, tail))
    finally:
        if mod is not None and mod.name in obj.modifiers:
            obj.modifiers.remove(mod)
        gone = obj.data.attributes.get(_TOOL_SELECTION_ATTR) if obj.type == "MESH" else None
        if gone is not None:
            obj.data.attributes.remove(gone)
        bpy.data.node_groups.remove(tmp)
    if mode == "EDIT":
        _set_mode(obj, "EDIT")
    return applied


def op_run_node_tool(params):
    """Run a geometry-nodes node TOOL on an object once, with its inputs - the menu operator, not a modifier.

    params:
      object (str, required)    a MESH, CURVES or POINTCLOUD object
      tool (str, required)      the node group's name (alias group); with library, the asset's name, appended first
      library (str)             DEFAULT, an asset library's name or a folder or .blend path (as assign_node_group)
      inputs (dict)             {input name or identifier: value}; a menu takes its item's name. Every one is checked
                                before anything runs.
      mode (str)                OBJECT or EDIT; default the object's own mode. EDIT runs on the edit-mode selection.

    The group must be marked as a tool (create_node_group tool) for that mode and object type - Blender's own rule.
    A tool that reads the selection (a Tool Selection node) is refused in OBJECT mode, where it would act on
    everything. Route: geometry.execute_node_group on 4.x-5.0 (inputs as temporary interface defaults, put back after);
    on 5.1+, where each tool's operator registers only in a GUI session's event loop, the group is applied as a Nodes
    modifier with Tool Selection reading the edit-mode vertex selection, and other tool-only nodes are refused there.
    3.6 has no node tools. Reply: route, inputsApplied, evaluated counts before and after.
    """
    reject_unknown(params, _TOOL_KEYS, "run_node_tool")
    tail = "NOTHING was run."
    # THE OBJECT FIRST: a missing one is the commonest mistake, and it is named before anything reads bpy.types
    # (test_blender_refusals B111 sweeps every op with a missing object).
    obj = get_object(take(params, "object", required=True, kind=str))
    if "is_tool" not in bpy.types.GeometryNodeTree.bl_rna.properties:
        raise MifOpError("node tools arrived in Blender 4.0; this is %s. %s" % (bpy.app.version_string, tail))
    if obj.type not in _TOOL_TYPE_FLAGS:
        raise MifOpError("'%s' is a %s; a node tool runs on a MESH, CURVES or POINTCLOUD object. %s"
                         % (obj.name, obj.type, tail))
    if obj.mode not in ("OBJECT", "EDIT"):
        raise MifOpError("'%s' is in %s mode; a tool runs here in OBJECT or EDIT mode. %s" % (obj.name, obj.mode, tail))
    mode = str(take(params, "mode", default=obj.mode, kind=str)).upper()
    if mode not in ("OBJECT", "EDIT"):
        raise MifOpError("mode is OBJECT or EDIT, got %r. %s" % (params.get("mode"), tail))
    name = take(params, "tool", "group", required=True, kind=str)
    library = params.get("library")
    asset = None
    if library is not None:
        tree, asset = _library_group(name, library, tail)
    else:
        tree = _tree(name)
    if tree.bl_idname != "GeometryNodeTree" or not tree.is_tool:
        tools = sorted(g.name for g in bpy.data.node_groups if g.bl_idname == "GeometryNodeTree" and g.is_tool)
        raise MifOpError("'%s' is not marked as a node tool (create_node_group tool marks one). Tool groups in this "
                         "file: %s. %s" % (tree.name, ", ".join(tools) or "(none)", tail))
    if not getattr(tree, "is_mode_" + mode.lower()):
        raise MifOpError("'%s' is not a tool for %s mode (its modes: %s). %s" % (tree.name, mode, ", ".join(
            m.upper() for m in ("object", "edit", "sculpt") if getattr(tree, "is_mode_" + m, False)) or "none", tail))
    flag = next((f for f in _TOOL_TYPE_FLAGS[obj.type] if hasattr(tree, f)), None)
    if flag and not getattr(tree, flag):
        raise MifOpError("'%s' is not a tool for %s objects. %s" % (tree.name, obj.type, tail))
    tool_nodes = _tool_only_nodes(tree)
    if mode == "OBJECT" and any(t[2] == _TOOL_SELECTION for t in tool_nodes):
        raise MifOpError("'%s' reads the selection (a Tool Selection node), and in OBJECT mode Blender hands a tool "
                         "every element as selected - run it with mode EDIT on the selection it is for. %s"
                         % (tree.name, tail))
    plan = _plan_tool_inputs(tree, params.get("inputs"), tail)
    # NOT hasattr: bpy.ops makes a wrapper for any name, so hasattr is true on 5.2 too, where the operator is gone and
    # the call fails "could not be found" (measured 2026-10-07). get_rna_type() is the registry lookup (ops_io._op_exists).
    try:
        bpy.ops.geometry.execute_node_group.get_rna_type()
        native = False
    except Exception:                                              # noqa: BLE001 - not registered on this build
        native = True
    if native:
        blocked = [t for t in tool_nodes if t[2] != _TOOL_SELECTION or t[3]]
        if blocked:
            raise MifOpError("this build (%s) has no execute_node_group, and a tool's own operator registers only in a "
                             "GUI session's event loop, so the tool runs here as a Nodes modifier - which cannot stand "
                             "in for %s. %s" % (bpy.app.version_string, ", ".join(
                                 "%s ('%s'%s)" % (t[2], t[1], " in nested group '%s'" % t[0] if t[3] else "")
                                 for t in blocked), tail))
    before = _tool_counts(obj)
    start_mode = obj.mode
    vl = bpy.context.view_layer
    prev_active = vl.objects.active
    try:
        if native:
            applied = _run_as_modifier(obj, tree, mode, plan, tail)
        else:
            applied = _run_by_operator(obj, tree, mode, plan, tail)
    finally:
        if obj.mode != start_mode:
            _set_mode(obj, start_mode)
        if obj.mode == "OBJECT" and vl.objects.active != prev_active:
            vl.objects.active = prev_active
    after = _tool_counts(obj)
    return {"object": obj.name, "tool": tree.name, "mode": mode, "asset": asset,
            "route": "nodes modifier (applied)" if native else "geometry.execute_node_group",
            "readsSelection": any(t[2] == _TOOL_SELECTION for t in tool_nodes),
            "inputsApplied": applied, "countsBefore": before, "countsAfter": after,
            "countsChanged": before != after}


_TOOL_MODES = ("OBJECT", "EDIT", "SCULPT")
_TOOL_TYPES = ("MESH", "CURVES", "POINTCLOUD")


def _tool_spec(raw, kind, tail):
    """create_node_group tool: true, or {modes: [...], types: [...]} -> (modes, types) or None. Checked before the
    group exists."""
    if raw is None or raw is False:
        return None
    if "is_tool" not in bpy.types.GeometryNodeTree.bl_rna.properties:
        raise MifOpError("node tools arrived in Blender 4.0; this is %s. %s" % (bpy.app.version_string, tail))
    if kind != "GeometryNodeTree":
        raise MifOpError("only a GeometryNodeTree can be a node tool, not a %s. %s" % (kind, tail))
    if raw is True:
        return ["OBJECT", "EDIT"], ["MESH"]
    if not isinstance(raw, dict) or set(raw) - {"modes", "types"}:
        raise MifOpError("tool is true or {modes: [OBJECT|EDIT|SCULPT], types: [MESH|CURVES|POINTCLOUD]}, got %r. %s"
                         % (raw, tail))
    out = []
    for key, allowed, default in (("modes", _TOOL_MODES, ["OBJECT", "EDIT"]), ("types", _TOOL_TYPES, ["MESH"])):
        vals = raw.get(key, default)
        if (not isinstance(vals, list) or not vals or
                any(not isinstance(v, str) or v.upper() not in allowed for v in vals)):
            raise MifOpError("tool %s is a list of %s, got %r. %s" % (key, "|".join(allowed), vals, tail))
        out.append([v.upper() for v in vals])
    return out[0], out[1]


def _mark_tool(tree, modes, types):
    """Blender's own new-tool setup (bl_operators/geometry_nodes.py): is_tool, a fake user (a tool has no other user
    and would be dropped on save), the mode and type flags, and on 5.1+ the operator identifier."""
    import re
    tree.is_tool = True
    tree.use_fake_user = True
    for m in _TOOL_MODES:
        setattr(tree, "is_mode_" + m.lower(), m in modes)
    for t, flags in _TOOL_TYPE_FLAGS.items():
        for f in flags:
            if hasattr(tree, f):
                setattr(tree, f, t in types)
    if hasattr(tree, "node_tool_idname"):
        tree.node_tool_idname = "geometry." + re.sub("[^0-9a-z]+", "_", tree.name.strip().lower())


# --------------------------------------------------------------------------------------------- inspect_node_output
_INSPECT_KEYS = {"object", "modifier", "node", "socket", "attributes"}
# Each zone's two ends, by node idname - the 3.6 simulation zone to the 4.2+ repeat and for-each zones.
_ZONE_PAIRS = (("GeometryNodeSimulationInput", "GeometryNodeSimulationOutput"),
               ("GeometryNodeRepeatInput", "GeometryNodeRepeatOutput"),
               ("GeometryNodeForeachGeometryElementInput", "GeometryNodeForeachGeometryElementOutput"))


def _link_sig(tree):
    return sorted((lk.from_node.name, lk.from_socket.identifier, lk.to_node.name, lk.to_socket.identifier)
                  for lk in tree.links)


def _zone_of(tree, node):
    """The zone input node's name when `node` sits inside a zone (forward-reachable from its input AND backward-
    reachable from its output), else None. A zone's own input node is inside; its output node's outputs are not."""
    fwd, back = {}, {}
    for lk in tree.links:
        fwd.setdefault(lk.from_node.name, set()).add(lk.to_node.name)
        back.setdefault(lk.to_node.name, set()).add(lk.from_node.name)

    def reach(start, edges):
        seen, todo = set(), [start]
        while todo:
            n = todo.pop()
            for m in edges.get(n, ()):
                if m not in seen:
                    seen.add(m)
                    todo.append(m)
        return seen
    for zin in tree.nodes:
        pair = next((p for p in _ZONE_PAIRS if zin.bl_idname == p[0]), None)
        if pair is None:
            continue
        paired = getattr(zin, "paired_output", None)
        outs = [paired] if paired is not None else [n for n in tree.nodes if n.bl_idname == pair[1]]
        for zout in outs:
            inside = (reach(zin.name, fwd) | {zin.name}) & reach(zout.name, back)
            if node.name in inside:
                return zin.name
    return None


def _geometry_bounds(ev):
    """World box of the evaluated geometry: the evaluated mesh, and on 5.0+ the point cloud and curves too."""
    pts = []
    mw = ev.matrix_world
    if hasattr(ev, "evaluated_geometry"):
        gs = ev.evaluated_geometry()
        for comp in (gs.mesh, gs.pointcloud, gs.curves):
            if comp is not None and "position" in comp.attributes:
                pts += [mw @ d.vector for d in comp.attributes["position"].data]
    else:
        try:
            m = ev.to_mesh()
        except RuntimeError:
            m = None
        if m is not None:
            pts = [mw @ v.co for v in m.vertices]
            ev.to_mesh_clear()
    if not pts:
        return None
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    return {"min": rnd(lo), "max": rnd(hi), "size": rnd([hi[i] - lo[i] for i in range(3)])}


def op_inspect_node_output(params):
    """Evaluate an object's Geometry Nodes tree AT ONE NODE, then put every link back and prove it.

    WHY (second 1.3 audit, 2026-10-25). 'Which node empties the geometry' meant rewiring by hand with four calls and
    remembering to restore; list_group_nodes reports reachability and mesh_stats the FINAL output only. The UI route
    (node.link_viewer, 5.2 bl_ui/space_node.py:738) shows its geometry only in the spreadsheet, so this relinks
    through tree.links: the node's geometry output goes to the Group Output's geometry input, the object is evaluated
    and read, and the original links are restored - the tree's whole link list is compared before and after.

    params:
      object (str)      an object with a Geometry Nodes modifier
      modifier (str)    which one (default the first NODES modifier)
      node (str)        the node in that modifier's tree (top level - a node inside a nested group is not reachable)
      socket            its geometry output, by name, identifier or index (default its first geometry output)
      attributes (list) attribute names to read min / max / mean of on the mesh (as mesh_stats attributeStats)
    A node inside a simulation, repeat or for-each zone is refused: a link out of a zone to the Group Output is
    invalid. The answer: components (counts per component and their attributes, as mesh_stats components), the world
    box, and restored (true only when the link list matched afterwards).
    """
    reject_unknown(params, _INSPECT_KEYS, "inspect_node_output")
    tail = "NOTHING was changed."
    obj = get_object(take(params, "object", required=True, kind=str))
    mname = take(params, "modifier", kind=str)
    mods = [m for m in obj.modifiers if m.type == "NODES"]
    mod = next((m for m in mods if m.name == mname), None) if mname else (mods[0] if mods else None)
    if mod is None:
        raise MifOpError("'%s' has no Geometry Nodes modifier%s (it has: %s). %s"
                         % (obj.name, " named %r" % mname if mname else "",
                            ", ".join(m.name for m in mods) or "none", tail))
    tree = mod.node_group
    if tree is None:
        raise MifOpError("modifier '%s' has no node group. %s" % (mod.name, tail))
    nname = take(params, "node", required=True, kind=str)
    node = tree.nodes.get(nname)
    if node is None:
        raise MifOpError("tree '%s' has no node %r. It has: %s. %s"
                         % (tree.name, nname, ", ".join(sorted(n.name for n in tree.nodes))[:600], tail))
    geo_outs = [s for s in node.outputs if s.type == "GEOMETRY" and getattr(s, "enabled", True)]
    want = params.get("socket")
    if want is None:
        sock = geo_outs[0] if geo_outs else None
    else:
        sock = _find_socket(node, str(want), node.outputs)
    if sock is None or sock.type != "GEOMETRY":
        raise MifOpError("node '%s' has no geometry output%s. Its outputs: %s. %s"
                         % (node.name, " %r" % want if want is not None else "",
                            ", ".join("%s (%s)" % (s.name, s.type) for s in node.outputs) or "none", tail))
    zone = _zone_of(tree, node)
    if zone:
        raise MifOpError("node '%s' is inside the zone that starts at '%s': a link from inside a zone to the Group "
                         "Output is invalid. Inspect the zone's output node instead. %s" % (node.name, zone, tail))
    gout = next((n for n in tree.nodes if n.type == "GROUP_OUTPUT" and getattr(n, "is_active_output", True)), None)
    target = next((s for s in gout.inputs if s.type == "GEOMETRY"), None) if gout is not None else None
    if target is None:
        raise MifOpError("tree '%s' has no Group Output with a geometry input. %s" % (tree.name, tail))
    names = params.get("attributes")
    if names is not None and (not isinstance(names, list) or not all(isinstance(n, str) and n for n in names)):
        raise MifOpError("attributes is a list of attribute names. %s" % tail)
    before = _link_sig(tree)
    held = [(lk.from_node.name, lk.from_socket.identifier) for lk in tree.links if lk.to_socket == target]
    from .ops_mesh import _attribute_stats, _geometry_read   # noqa: PLC0415 - op modules do not import each other at load
    out = {"object": obj.name, "modifier": mod.name, "tree": tree.name, "node": node.name, "socket": sock.name,
           "socketIdentifier": sock.identifier}
    temp = None
    try:
        for lk in [lk for lk in tree.links if lk.to_socket == target]:
            tree.links.remove(lk)
        temp = tree.links.new(sock, target)
        bpy.context.view_layer.update()
        if not temp.is_valid:
            raise MifOpError("Blender marked the link from '%s' to the Group Output invalid. Nothing was read; the "
                             "links are put back." % node.name)
        deps = bpy.context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(deps)
        out.update(_geometry_read(obj, True, None))
        out["bboxWorld"] = _geometry_bounds(ev)
        if obj.type == "MESH":
            try:
                m = ev.to_mesh()
                out["mesh"] = {"vertices": len(m.vertices), "edges": len(m.edges), "faces": len(m.polygons)}
                if names:
                    out["attributeStats"] = _attribute_stats(m, list(dict.fromkeys(names)))
            finally:
                ev.to_mesh_clear()
        out["empty"] = not out.get("bboxWorld") and not (out.get("components") or {}).get("instances")
    finally:
        if temp is not None:
            try:
                tree.links.remove(temp)
            except (RuntimeError, ReferenceError):
                pass
        for fn, fid in held:
            src = tree.nodes.get(fn)
            s = next((x for x in src.outputs if x.identifier == fid), None) if src is not None else None
            if s is not None:
                tree.links.new(s, target)
        bpy.context.view_layer.update()
    after = _link_sig(tree)
    out["restored"] = after == before
    out["links"] = len(after)
    if not out["restored"]:
        out["restoreError"] = {"missing": [list(x) for x in before if x not in after][:20],
                               "extra": [list(x) for x in after if x not in before][:20]}
    return out


OPS = {
    "inspect_node_output": op_inspect_node_output,
    "set_compositing": op_set_compositing,
    "compositor_info": op_compositor_info,
    "build_pass_composite": op_build_pass_composite,
    "create_node_group": op_create_node_group,
    "add_group_node": op_add_group_node,
    "link_group_nodes": op_link_group_nodes,
    "add_group_interface": op_add_group_interface,
    "list_group_nodes": op_list_group_nodes,
    "list_node_groups": op_list_node_groups,
    "set_node_property": op_set_node_property,
    "group_nodes": op_group_nodes,
    "remove_node": op_remove_node,
    "assign_node_group": op_assign_node_group,
    "run_node_tool": op_run_node_tool,
}
