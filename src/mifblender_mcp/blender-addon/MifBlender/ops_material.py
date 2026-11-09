"""Material creation, inspection and per-face assignment.

WHAT WAS MISSING. The addon had exactly one material verb, set_material_slots, and it
assigns NAMES to slots on purpose - its own docstring says "a material's content is
Unreal's business". That was the right call for a round trip, where the material really
does live in Unreal and Blender only has to keep the slot ORDER lined up. It is the wrong
shape for building an asset IN Blender, which is what was asked for on 2026-08-30: there
was no way to create a material, set a shading value, or read back what a material holds.
There was no material READ op of any kind - object_info reports slot names and nothing
about what is in them.

=============================================================================
THE HARD PART IS NOT THE NODES, IT IS THE VERSION SPREAD
=============================================================================
This addon supports Blender 3.6 through 5.0, and Blender RENAMED Principled BSDF inputs
across that range:

    3.6              4.0+                       what it is
    "Specular"       "Specular IOR Level"       the specular level
    "Emission"       "Emission Color"           the emission colour
    "Transmission"   "Transmission Weight"      transmission
    "Sheen"          "Sheen Weight"             sheen
    "Clearcoat"      "Coat Weight"              clearcoat

A node input is looked up by string, and `node.inputs["Specular"]` on 4.x raises KeyError -
but the far worse failure is the one that does NOT raise: writing to an input that exists
under a different name on another version means the value silently lands nowhere on that
version, and the material looks subtly wrong with nothing to read. So every property here
resolves through an ALIAS LIST, and a name that resolves on NO alias is REFUSED by name
with the inputs that actually exist on this Blender. Never skipped, never defaulted.

set_material_properties reports `resolvedInputs` - the real socket name each requested
property landed on - so a caller can see which spelling this Blender used rather than
having to know.
"""
import math
import os
import time

import bpy

from .ops_common import (abs_output_path, MifOpError, assign_id_name, check_output_path, exclude_rows, finite_float, finite_floats, get_object, 
                         jsonable, limit_rows, reject_unknown, rnd, select_only, selection_restore, 
                         selection_snapshot, take, take_bool, take_float, take_int)
# IMPORTED RATHER THAN REIMPLEMENTED, and the direction of this dependency is deliberate:
# ops_material -> ops_nodes at module level, never the other way. ops_nodes owns the ONE tree
# resolver (_tree, which is what reaches "material:<name>", "scene:compositor" and the rest), the
# ONE node-type vocabulary and version gate, and the ONE typed property writer. A second copy of
# any of them living here is the parallel system this project keeps paying to remove - the
# compositor got a resolver instead of a second set of add/link/list ops for exactly this reason.
# ops_nodes reaches BACK for principled_socket, and does it with a deferred import inside the one
# function that needs it, so the module-level direction stays one way.
from .ops_nodes import (_socket_value, _tree, _write_node_property, color_ramp_stops,
                        node_readback, resolve_node_type, usable_sockets)

# Requested name -> the socket names it may be called on some supported Blender.
# Order matters only for reporting; the first that EXISTS on this build wins.
# normalStrength is DELIBERATELY ABSENT. It is not a Principled socket - normal strength lives on a
# separate Normal Map node - so it cannot be written here. It was briefly listed with an empty
# alias tuple and then filtered out of the write loop, which meant reject_unknown ACCEPTED it and
# the handler silently ignored it: the exact silent no-op this module's docstring calls the worst
# outcome. Leaving it out means a caller passing it gets a refusal naming the real properties.
PRINCIPLED_ALIASES = {
    "baseColor":    ("Base Color",),
    "metallic":     ("Metallic",),
    "roughness":    ("Roughness",),
    "specular":     ("Specular IOR Level", "Specular"),
    "ior":          ("IOR",),
    "alpha":        ("Alpha",),
    "emissive":     ("Emission Color", "Emission"),
    "emissiveStrength": ("Emission Strength",),
    "transmission": ("Transmission Weight", "Transmission"),
    "sheen":        ("Sheen Weight", "Sheen"),
    "clearcoat":    ("Coat Weight", "Clearcoat"),
    "anisotropic":  ("Anisotropic",),
    # SUBSURFACE (2026-10-06): skin, wax and glaze. 4.0 and later call the weight "Subsurface Weight" and add Subsurface
    # Scale; 3.6 has "Subsurface" and no scale, so subsurfaceScale resolves on no socket there and is refused by name.
    # Read off bl_rna on 4.2.23, 4.4.0, 5.0.1 and 5.2.2: Weight, Radius (a vector), Scale, IOR, Anisotropy.
    "subsurfaceWeight":     ("Subsurface Weight", "Subsurface"),
    "subsurfaceScale":      ("Subsurface Scale",),
    "subsurfaceRadius":     ("Subsurface Radius",),
    "subsurfaceIor":        ("Subsurface IOR",),
    "subsurfaceAnisotropy": ("Subsurface Anisotropy",),
}

# A UNION OF TWO LITERALS, not a computed tuple. parity_check.py can resolve `{...} | set(NAME)`
# where NAME is a module-level dict literal, and refuses anything it cannot read statically - which
# is right, because an unverifiable parameter list is where drift hides.
SET_MATERIAL_PROPERTY_PARAMS = {"material", "name", "rename", "viewportColor", "viewportMetallic",
                                "viewportRoughness", "subsurfaceMethod", "weathering"} | set(PRINCIPLED_ALIASES)

# THE VIEWPORT DISPLAY PANEL (2026-10-05): Material.diffuse_color, metallic and roughness - what Solid view and an asset
# preview show, not the shader. A linked asset was grey in Solid view with no typed way to colour it.
_VIEWPORT_MAT = {"viewportColor": "diffuse_color", "viewportMetallic": "metallic", "viewportRoughness": "roughness"}


COLOR_PROPS = ("baseColor", "emissive")


def _principled(mat):
    """The Principled BSDF node, or a refusal naming what the material actually has."""
    if not mat.use_nodes or mat.node_tree is None:
        raise MifOpError(
            "material '%s' has no node tree (use_nodes is off), so it has no Principled BSDF "
            "to write to. create_material makes one; a material imported without nodes has to "
            "be converted by hand." % mat.name)
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    kinds = sorted({n.type for n in mat.node_tree.nodes})
    raise MifOpError(
        "material '%s' has no Principled BSDF node - its tree holds: %s. This op writes "
        "Principled inputs; a material built on a different shader needs its nodes edited "
        "directly." % (mat.name, ", ".join(kinds) or "(nothing)"))


def _resolve_input(node, prop):
    """The real socket for a requested property on THIS Blender, or None."""
    for alias in PRINCIPLED_ALIASES.get(prop, ()):
        if alias in node.inputs:
            return node.inputs[alias]
        # BY NAME WHEN THE KEYED LOOKUP MISSES. Measured 2026-10-06 on 4.2.23 and 5.2.2: "Subsurface IOR" is a socket
        # whose name and identifier both read "Subsurface IOR", and `"Subsurface IOR" in node.inputs` is still False.
        for sock in node.inputs:
            if sock.name == alias:
                return sock
    return None


def principled_socket(node, requested):
    """A Principled input named ANY of the three ways, resolved to this build's socket, or None.

    THE THIRD WAY IS THE ONE THAT WAS MISSING. This module has always resolved its own property
    names - "emissive", "baseColor" - through PRINCIPLED_ALIASES, so every WRITE has been
    version-safe since the day it was written. Linking had no such route: link_group_nodes matches
    socket names literally, so wiring a mask into emission is toSocket:"Emission" on 3.6 and
    toSocket:"Emission Color" on 4.0 and later, and a caller building one graph for both had to
    branch on bpy.app.version. Now the table serves both halves and the same call works on every
    supported build.

    Accepted spellings, in the order they are tried:
      1. the socket name THIS build uses            "Emission Color" on 4.x
      2. this addon's property name                 "emissive"
      3. the socket name ANOTHER supported build uses, mapped through the alias table
                                                    "Emission" on 4.x -> "Emission Color"

    Returns None rather than raising, because "no such input" is a different sentence in every
    caller and only the caller knows what it has already built.
    """
    if node.type != "BSDF_PRINCIPLED":
        return None
    if requested in node.inputs:
        return node.inputs[requested]
    want = str(requested).strip().lower()
    for prop, aliases in PRINCIPLED_ALIASES.items():
        if prop.lower() == want or any(a.lower() == want for a in aliases):
            sock = _resolve_input(node, prop)
            if sock is not None:
                return sock
    return None


def _material_json(mat, deep=False, nodes=False):
    out = {
        "name": mat.name,
        "users": mat.users,
        "useNodes": bool(mat.use_nodes),
        "blendMethod": getattr(mat, "blend_method", None),
    }
    # EVERY SETTING set_material_settings WRITES, read off the same map, so the read side cannot fall behind the write
    # side again. It had: the second seat set eight settings in one ok:true call and describe_material came back with
    # blendMethod alone (mailbox ids 6 and 8, 4.4.0) - backfaceCulling, alphaThreshold, surfaceRenderMethod and the rest
    # were writable and had exactly one reader, the op that wrote them. Only what this build has is listed; the map
    # names the builds for the rest.
    out["settings"] = {key: (round(float(getattr(mat, attr)), 6) if isinstance(getattr(mat, attr), float)
                             else getattr(mat, attr))
                       for key, (attr, _builds) in _MATSET_MAP.items() if attr in mat.bl_rna.properties}
    # THE VIEWPORT DISPLAY (what Solid view and an asset preview show), the read twin of set_material_properties'
    # viewportColor, viewportMetallic and viewportRoughness (2026-10-05)
    out["viewport"] = _viewport_read(mat)
    if not mat.use_nodes or mat.node_tree is None:
        out["diffuseColor"] = rnd(list(mat.diffuse_color))
        out["note"] = ("this material has no node tree, so it has no shading inputs to report. "
                       "Its viewport diffuse_color is all there is.")
        return out
    # NAMED tree_nodes, NOT nodes, AND THAT IS THE WHOLE OF A REAL BUG. This line used to read
    # `nodes = mat.node_tree.nodes`, which SHADOWED the `nodes` parameter in this function's own
    # signature. By the time `if nodes:` ran at the bottom it was testing a bpy collection - truthy
    # for every material that has a node tree at all - so nodeGraph was emitted ALWAYS and the
    # caller's nodes=False default could never be honoured.
    #
    # MEASURED 2026-09-09 on a three-node material: the response was 6484 bytes and nodeGraph was
    # 5480 of them, 84.5%. describe_material with nodes:true came back BYTE-IDENTICAL to the
    # default, which is what sent me looking. A user had reported "~3k tokens for a single material"
    # and was right about the symptom; this was the cause. Fixing it took the default to 930 bytes.
    #
    # Nothing in this repo reads nodeGraph and nothing passes nodes:, so honouring the default
    # changes no caller - it just stops charging every caller for a field they did not ask for.
    # T4100b pins both halves and was mutation-tested against this exact shadow.
    tree_nodes = mat.node_tree.nodes
    out["nodeCount"] = len(tree_nodes)
    out["nodeTypes"] = sorted({n.type for n in tree_nodes})
    # THE READ TWIN OF set_material_properties' weathering: present only when a MifWear chain is in the tree
    wear = [n for n in tree_nodes if n.name.startswith("MifWear")]
    if wear:
        out["weathering"] = {"nodes": len(wear),
                             "masks": sorted(n.name[8:] for n in wear if n.name[8:] in ("Wear", "Grime")),
                             "regions": sorted(n.attribute_name[len(_WEAR_ATTR):] for n in wear
                                               if n.type == "ATTRIBUTE" and n.attribute_name.startswith(_WEAR_ATTR)),
                             "edgeSource": ("bevel" if any(n.type == "BEVEL" for n in wear) else
                                            "pointiness" if any(n.name == "MifWear EdgeMask" for n in wear) else None)}

    bsdf = next((n for n in tree_nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf:
        props = {}
        resolved = {}
        for prop in PRINCIPLED_ALIASES:
            sock = _resolve_input(bsdf, prop)
            if sock is None:
                continue
            resolved[prop] = sock.name
            try:
                val = sock.default_value
                props[prop] = rnd(list(val)) if hasattr(val, "__len__") else round(float(val), 6)
            except Exception:  # noqa: BLE001
                continue
        if "subsurface_method" in bsdf.bl_rna.properties:
            # NOT A SOCKET: a property of the node, the read twin of set_material_properties' subsurfaceMethod
            props["subsurfaceMethod"] = bsdf.subsurface_method
        out["principled"] = props
        # The socket NAMES this Blender used. Reported because they differ by version and a
        # caller writing them back needs to know which spelling landed.
        out["resolvedInputs"] = resolved
    else:
        out["principled"] = None
        out["note"] = "no Principled BSDF in this material's node tree"

    # IMAGE TEXTURES AND THEIR PATHS. This is the field an Unreal-side import has to resolve,
    # so it is reported whether or not `deep` was asked for.
    # AN IMAGE-LESS TEX_IMAGE NODE IS REPORTED, and that is the whole change here.
    #
    # This used to require `n.image is not None`, so a texture node with no image was skipped
    # entirely - absent from `textures` and uncounted. That is precisely the state Blender renders
    # MAGENTA: loud on screen, and this op reported textureCount 0, indistinguishable from a
    # material that has no texture nodes at all. The read side hid the one state somebody looking
    # at a bright pink material most needs to see.
    #
    # COLOUR SPACE IS REPORTED TOO, because it is the other way a texture is wrong while every
    # field looks right: a normal or roughness map decoded as sRGB comes back through the transfer
    # curve and every value the shader reads is off. set_material_texture chooses it correctly on
    # the way in; this is how you check what a material IMPORTED from elsewhere actually has.
    textures = []
    for n in tree_nodes:
        if n.type != "TEX_IMAGE":
            continue
        img = n.image
        entry = {
            "node": n.name,
            "image": img.name if img is not None else None,
            "filepath": img.filepath if img is not None else None,
            "size": list(img.size) if img is not None else None,
            "packed": bool(img.packed_file) if img is not None else None,
            "colorSpace": (img.colorspace_settings.name if img is not None else None),
            # THE TWO BROKEN STATES, named rather than left to be inferred from a null.
            "hasImage": img is not None,
            "hasPixels": (tuple(img.size) != (0, 0)) if img is not None else False,
        }
        linked = []
        for outsock in n.outputs:
            for link in outsock.links:
                linked.append("%s.%s" % (link.to_node.name, link.to_socket.name))
        entry["linkedTo"] = linked
        if img is None:
            entry["warning"] = (
                "this texture node has NO IMAGE. Blender renders it MAGENTA - obvious on screen "
                "and, until now, invisible here.")
        elif tuple(img.size) == (0, 0):
            entry["warning"] = (
                "'%s' has no pixels (0x0), so its source did not resolve. Blender renders this "
                "MAGENTA." % img.name)
        elif not linked:
            entry["warning"] = (
                "this texture node is not connected to anything, so it is loaded and unused - "
                "which reads identically to a working one in every other field.")
        textures.append(entry)
    out["textures"] = textures
    out["textureCount"] = len(textures)
    # SEPARATE COUNTS, because "how many texture nodes" and "how many will render" are different
    # questions and the second is the one that decides what the material looks like.
    out["texturesWithImage"] = len([e for e in textures if e["hasImage"]])
    broken = [e["node"] for e in textures if e.get("warning")]
    out["texturesNeedingAttention"] = broken or None

    if deep:
        out["links"] = [
            {"from": "%s.%s" % (l.from_node.name, l.from_socket.name),
             "to": "%s.%s" % (l.to_node.name, l.to_socket.name)}
            for l in mat.node_tree.links
        ]

    # THE GRAPH, WITH THE NUMBERS IN IT. Everything above this line describes a material as a
    # Principled BSDF plus a list of image files, which is exactly what this op could produce
    # before procedural nodes were reachable. It says nothing about a Noise texture's scale, a
    # colour ramp's stops or a Mapping node's rotation - so an agent that has just authored a
    # procedural material can read back the SHAPE of what it built and none of the values, and has
    # no way to check its own work without opening Blender.
    #
    # Off by default because it is large and because the Principled summary is the usual question.
    # node_readback is ops_nodes' and is shared with list_group_nodes, so a material's graph and a
    # geometry group's graph are described in the same words.
    if nodes:
        out["nodeGraph"] = [node_readback(n, values=True) for n in mat.node_tree.nodes]
        out["proceduralTextures"] = sorted(
            n.name for n in mat.node_tree.nodes
            if n.bl_idname.startswith("ShaderNodeTex") and n.bl_idname != "ShaderNodeTexImage")
        out["colorRampNodes"] = sorted(
            n.name for n in mat.node_tree.nodes if getattr(n, "color_ramp", None) is not None)
    return out


# ---------------------------------------------------------------------------
def op_create_material(params):
    """Create a material with a Principled BSDF, and report the name Blender gave it.

    THE NAME IS ECHOED FROM THE MATERIAL, NOT FROM THE REQUEST. bpy.data.materials.new() silently
    appends .001 on a collision, so a caller who asked for "Wood" can end up with "Wood.003" and
    never know. Reporting the requested name back would be wrong roughly as often as a scene has a
    name clash in it, which is often.

    `reuse` RETURNS THE EXISTING MATERIAL rather than a numbered copy - the idempotent-create shape
    a pipeline uses by default. It APPLIES the inline values too. An earlier version returned early
    on that path, silently discarding baseColor/metallic/roughness while attaching a note claiming
    "the end state you asked for is already in place" - which was false precisely when a shading
    value had been passed, on the most common call shape there is.

    REUSE ALSO REPAIRS use_nodes. bpy.data.materials.new() leaves nodes OFF on 3.6, 4.2 and 4.4 and
    ON only on 5.0, and set_material_slots creates bare materials with no node tree at all - so a
    reused material may have no Principled BSDF, contradicting this op's whole contract. It is
    switched on and the BSDF verified, rather than handing back something set_material_properties
    would then refuse.

    NOTHING IS CREATED UNTIL EVERY VALUE VALIDATES. The material used to be created first, so a
    refusal left a new datablock behind while the message said "NOTHING was changed".

    `skipLinkedInputs` IS FOR GET-OR-CREATE, and it exists because the default refusal makes an
    honest pipeline unrunnable. A LINKED Principled input is refused (see _check_unlinked) because
    writing one is a silent no-op that reads back as if it worked. That is right when the caller
    MEANS to set a value. It is wrong for the commonest shape there is - `reuse` with the same
    literal values every time, used to MEAN "give me this material", where the values are what to
    build it from if it does not exist yet.

    Measured on the bunker showcase: `S.mat("Bunker_Steel", ...)` appears in four stage files with
    identical arguments, purely to fetch the material. Once stage 3b drives Base Color through a
    colour ramp, all four refuse, so the stage order that showcase DOCUMENTS - 3b after 3, before 4
    - could not be run. The refusal was correct and the pipeline was still broken.

    With this on, a linked input is not written and is NAMED in `skippedLinkedInputs` alongside the
    node driving it. That is the difference between this and the early-return that used to live on
    the reuse path: that one discarded values silently while claiming success. Off by default, so
    the guard still catches the caller who meant it.
    """
    reject_unknown(params, ("name", "reuse", "baseColor", "metallic", "roughness",
                            "skipLinkedInputs", "copyFrom"), "create_material")
    name = take(params, "name", required=True, kind=str)
    reuse = take_bool(params, "reuse", default=False)
    # COPY A MATERIAL (2026-10-04): the node tree duplicated under a new name - one material per row of a hair atlas,
    # each picking its row by its Mapping offset. Blender names the copy; the name it gave comes back.
    copy_name = take(params, "copyFrom", default=None, kind=str)
    source = None
    if copy_name is not None:
        if reuse:
            raise MifOpError("copyFrom makes a new material and reuse returns an existing one - pass one. NOTHING was "
                             "changed.")
        source = bpy.data.materials.get(copy_name)
        if source is None:
            raise MifOpError("no material named %r to copy. NOTHING was changed." % copy_name)
    skip_linked = take_bool(params, "skipLinkedInputs", default=False)
    inline = {p: params[p] for p in ("baseColor", "metallic", "roughness")
              if p in params and params[p] is not None}

    existing = bpy.data.materials.get(name)
    mat = None
    created = False
    if existing is not None and reuse:
        mat = existing
        if not mat.use_nodes or mat.node_tree is None:
            mat.use_nodes = True
    else:
        # DEFERRED until the values are known-good: see the docstring.
        pass

    if mat is None and source is not None:
        # NAMED ONLY AFTER THE VALUES VALIDATE (below), so a refusal removes a copy that renamed nothing.
        mat = source.copy()
        created = True
    elif mat is None:
        # Validate what we can before creating anything. The socket check needs a BSDF, so the
        # remaining validation happens against the new material and is rolled back on failure.
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        created = True

    bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None) if mat.node_tree else None
    if bsdf is None and source is not None and not inline:
        pass        # a copy keeps whatever its source was built from; only inline values need a Principled BSDF
    elif bsdf is None:
        if created:
            bpy.data.materials.remove(mat)
        raise MifOpError(
            "material '%s' has no Principled BSDF node%s. NOTHING usable was produced."
            % (name, " even with use_nodes on, which should not happen on any supported Blender"
               if created else " - it was created without one, and this op cannot add it"))

    applied = {}
    skipped = {}
    try:
        # RESOLVE AND COERCE EVERYTHING FIRST, then write. A bad value must not leave a
        # half-configured material, and must not leave a new one behind at all.
        staged = []
        for prop, val in inline.items():
            sock = _resolve_input(bsdf, prop)
            if sock is None:
                raise MifOpError(
                    "this Blender's Principled BSDF has no input for '%s' (tried: %s). Available "
                    "inputs: %s. NOTHING was changed."
                    % (prop, ", ".join(PRINCIPLED_ALIASES[prop]) or "(none)",
                       ", ".join(i.name for i in bsdf.inputs)))
            # THE MISSING INPUT ABOVE IS STILL FATAL EVEN HERE, deliberately. A socket that does not
            # exist on this Blender means the caller is asking for something this version cannot do,
            # which is a different fact from "that input is driven by a node".
            if skip_linked and sock.is_linked:
                src = sock.links[0].from_node if sock.links else None
                skipped[prop] = src.name if src else "(unknown node)"
                continue
            _check_unlinked(sock, prop)
            staged.append((prop, sock, _coerce_socket(sock, prop, val)))
        for prop, sock, value in staged:
            sock.default_value = value
            applied[prop] = sock.name
    except MifOpError:
        if created:
            bpy.data.materials.remove(mat)
        raise

    if source is not None:
        # THROUGH assign_id_name: a plain `mat.name = name` on 3.6 can rename the material that already held the name
        # (ledger B12) - the copy takes the suffix on every build instead.
        assign_id_name(mat, name, bpy.data.materials)
    out = _material_json(mat)
    out["created"] = created
    if source is not None:
        out["copiedFrom"] = source.name
        out["nodeCount"] = len(mat.node_tree.nodes) if mat.node_tree else 0
        out["sourceNodeCount"] = len(source.node_tree.nodes) if source.node_tree else 0
    out["requestedName"] = name
    # NO FLAG AT ALL until now - this op reported the name asked for and the name given and left the
    # caller to compare two strings, which is the one thing a response field exists to save them.
    # Nothing to be compatible with here, so it gets the canonical spelling directly.
    out["nameWasSuffixed"] = mat.name != name
    if applied:
        out["resolvedInputs"] = applied
        out["applied"] = sorted(applied)
    # REPORTED WHENEVER ANYTHING WAS SKIPPED, and never silently. The whole objection to the old
    # early-return was that it discarded values while claiming the end state was in place; a caller
    # who passed a value that did not land has to be able to see that from the response alone.
    if skipped:
        out["skippedLinkedInputs"] = skipped
        out["skipped"] = sorted(skipped)
    if not created:
        out["note"] = (
            "a material with that name already existed and reuse was requested, so it was NOT "
            "recreated." + (" The values you passed were applied to it - see applied[]."
                            if applied else " No shading values were passed, so nothing about it "
                                            "changed."))
    elif mat.name != name:
        out["nameNote"] = ("Blender renamed this to '%s' because '%s' was already taken - new() "
                           "never overwrites and never fails, it appends a number. Pass reuse:true "
                           "to get the existing material instead." % (mat.name, name))
    return out


def _coerce_socket(sock, prop, val):
    """Validate a value for a socket and return what would be written. Writes NOTHING.

    SPLIT OUT FROM THE WRITE so every value in a call can be checked before any is applied. An
    earlier version validated inside the write loop, so a bad value on the fifth property left the
    first four already written - while the docstring promised the opposite.
    """
    if prop in COLOR_PROPS or hasattr(sock.default_value, "__len__"):
        # A LIST OR TUPLE, not merely something with __len__. A DICT has one, so {"mif":"x"} passed
        # this guard and then [float(x) for x in val] iterated its KEYS and raised ValueError out of
        # the op - found by the matrix bad-value pass once it gained a dict sentinel, because a
        # string sentinel is caught by the isinstance check one line up and never reaches here.
        if not isinstance(val, (list, tuple)):
            raise MifOpError("'%s' takes a colour as [r,g,b] or [r,g,b,a], got %r" % (prop, val))
        try:
            # finite_floats - this is the socket coercion every material write goes through, so a
            # NaN admitted here reaches any Principled input the caller names.
            vals = finite_floats(val, "value")
        except (TypeError, ValueError):
            raise MifOpError("'%s' takes NUMBERS as [r,g,b] or [r,g,b,a], got %r" % (prop, val))
        want = len(sock.default_value)
        if len(vals) == 3 and want == 4:
            vals.append(1.0)          # alpha defaults to opaque rather than to zero
        if len(vals) != want:
            raise MifOpError("'%s' expects %d components on this Blender, got %d"
                             % (prop, want, len(vals)))
        return vals
    return finite_float(val, prop)


def _check_unlinked(sock, prop):
    """A LINKED socket ignores default_value entirely, so writing one changes nothing that renders.

    Blender evaluates the incoming link and never reads default_value on a connected input. Writing
    it anyway succeeds at the Python level and shows up when read back, so the op would report the
    new value while the material rendered and exported exactly as before - a silent no-op that
    confirms itself. Refused instead, naming what is driving the socket.
    """
    if sock.is_linked:
        src = sock.links[0].from_node if sock.links else None
        raise MifOpError(
            "'%s' is LINKED to %s, and a connected input ignores its default value completely - "
            "Blender evaluates the link instead. Writing it would change nothing that renders or "
            "exports while still reading back as the new value. Disconnect the link first, or edit "
            "the node driving it. NOTHING was changed."
            % (prop, ("node '%s'" % src.name) if src else "another node"))


def _viewport_plan(params):
    """{Material attribute: value} for the viewport keys, every value checked before any write."""
    plan = {}
    col = params.get("viewportColor")
    if col is not None:
        if not isinstance(col, (list, tuple)) or len(col) not in (3, 4):
            raise MifOpError("viewportColor is [r, g, b] or [r, g, b, a], got %r. NOTHING was changed." % (col,))
        vals = finite_floats(list(col), "viewportColor")
        if any(v < 0.0 or v > 1.0 for v in vals):
            raise MifOpError("viewportColor values run 0-1, got %r. NOTHING was changed." % (col,))
        plan["diffuse_color"] = tuple(vals) + ((1.0,) if len(vals) == 3 else ())
    for key in ("viewportMetallic", "viewportRoughness"):
        if params.get(key) is not None:
            v = take_float(params, key)
            if not 0.0 <= v <= 1.0:
                raise MifOpError("%s runs 0-1, got %g. NOTHING was changed." % (key, v))
            plan[_VIEWPORT_MAT[key]] = v
    return plan


def _viewport_read(mat):
    return {"color": rnd(list(mat.diffuse_color)), "metallic": round(float(mat.metallic), 6),
            "roughness": round(float(mat.roughness), 6)}


# ---------------------------------------------------------------------------------------------------------------------
# WEATHERING (1.4, 2026-10-30): edge wear, grime and breakup spliced in front of a Principled BSDF's own inputs.
#
# THE GAP. Artists judged one-prompt props beginner-level and unrealistic, and the materials those runs made were one
# flat color and one roughness per part. Real surfaces wear where they are handled - the convex edges - collect dirt
# where nothing reaches - the cavities - and vary everywhere. The masks here are driven by the mesh, not painted:
#   edge wear   curvature from a Bevel node (1 - the beveled normal . the true normal: 0 on a flat face, rising inside
#               `width` of a sharp edge - a crisp mask even on an 8-vertex box) or Cycles' per-vertex Pointiness
#   grime       1 - Ambient Occlusion over `distance` (the object only), so creases and inside corners darken
#   breakup     two Noise textures in OBJECT space (no UVs needed, and the same on every bake), one making the wear
#               and grime patchy rather than a uniform band, one varying color value and roughness
# Each Principled input keeps what fed it - a constant becomes an RGB / Value node, a texture stays linked - and the
# chain sits between that and the input, so the same call weathers a flat material and a textured one. Every node is
# named "MifWear ..." and parented to one frame; calling again replaces the chain, false removes it and puts each input
# back as it was.
#
# Bevel and Pointiness are Cycles nodes: EEVEE shades the chain without edge wear (AO grime still shows). Baking is
# Cycles, so bake_texture_set (BASE_COLOR, ROUGHNESS, METALLIC, NORMAL) carries the whole look to any engine.
_WEAR_PREFIX = "MifWear"
_WEAR_KEYS = ("edgeWear", "grime", "roughnessVariation", "colorVariation", "scale", "seed", "regions", "objects",
              "armature")
_WEAR_EDGE_KEYS = ("amount", "width", "source", "color", "roughness", "metallic", "bump")
_WEAR_GRIME_KEYS = ("amount", "distance", "color", "roughness")


def _wear_unit(block, key, default, lo, hi, where):
    v = block.get(key, default)
    if v is None:
        return None
    try:
        v = finite_float(v, "%s.%s" % (where, key))
    except (TypeError, ValueError):
        raise MifOpError("%s.%s is a number, got %r. NOTHING was changed." % (where, key, block.get(key)))
    if not lo <= v <= hi:
        raise MifOpError("%s.%s runs %g to %g, got %g. NOTHING was changed." % (where, key, lo, hi, v))
    return v


def _wear_color(block, key, where):
    v = block.get(key)
    if v is None:
        return None
    if not isinstance(v, (list, tuple)) or len(v) not in (3, 4):
        raise MifOpError("%s.%s is [r, g, b] (linear 0-1), got %r. NOTHING was changed." % (where, key, v))
    vals = finite_floats(list(v)[:3], "%s.%s" % (where, key))
    if any(c < 0.0 or c > 1.0 for c in vals):
        raise MifOpError("%s.%s values run 0-1, got %r. NOTHING was changed." % (where, key, v))
    return vals + [1.0]


def _weathering_plan(raw):
    """Validate a weathering block completely before anything is built. False -> {"remove": True}."""
    if raw is False:
        return {"remove": True}
    if raw is True:
        raw = {}
    if not isinstance(raw, dict):
        raise MifOpError("weathering is {edgeWear, grime, roughnessVariation, colorVariation, scale, seed}, true for "
                         "the defaults, or false to remove it; got %r. NOTHING was changed." % (raw,))
    bad = sorted(set(raw) - set(_WEAR_KEYS))
    if bad:
        raise MifOpError("weathering has no %s - its keys are %s. NOTHING was changed."
                         % (", ".join(bad), ", ".join(_WEAR_KEYS)))
    plan = {"remove": False}
    for key, keys in (("edgeWear", _WEAR_EDGE_KEYS), ("grime", _WEAR_GRIME_KEYS)):
        blk = raw.get(key, True)
        if blk is False or blk is None:
            plan[key] = None
            continue
        if blk is True:
            blk = {}
        if not isinstance(blk, dict):
            raise MifOpError("weathering.%s is {%s}, true or false; got %r. NOTHING was changed."
                             % (key, ", ".join(keys), blk))
        bad = sorted(set(blk) - set(keys))
        if bad:
            raise MifOpError("weathering.%s has no %s - its keys are %s. NOTHING was changed."
                             % (key, ", ".join(bad), ", ".join(keys)))
        where = "weathering.%s" % key
        p = {"amount": _wear_unit(blk, "amount", 0.5, 0.0, 1.0, where),
             "color": _wear_color(blk, "color", where),
             "roughness": _wear_unit(blk, "roughness", None, 0.0, 1.0, where)}
        if key == "edgeWear":
            p["width"] = _wear_unit(blk, "width", 0.02, 1e-4, 10.0, where)
            p["metallic"] = _wear_unit(blk, "metallic", None, 0.0, 1.0, where)
            p["bump"] = _wear_unit(blk, "bump", 0.0, 0.0, 1.0, where)
            src = str(blk.get("source", "bevel")).lower()
            if src not in ("bevel", "pointiness"):
                raise MifOpError("weathering.edgeWear.source is bevel (default: curvature within `width` of an edge, "
                                 "crisp on any mesh) or pointiness (per vertex - needs a dense mesh); got %r. NOTHING "
                                 "was changed." % blk.get("source"))
            p["source"] = src
        else:
            p["distance"] = _wear_unit(blk, "distance", 0.2, 1e-4, 100.0, where)
            if p["roughness"] is None:
                p["roughness"] = 0.9
            if p["color"] is None:
                p["color"] = [0.035, 0.028, 0.02, 1.0]
        plan[key] = p
    plan["roughnessVariation"] = _wear_unit(raw, "roughnessVariation", 0.15, 0.0, 1.0, "weathering")
    plan["colorVariation"] = _wear_unit(raw, "colorVariation", 0.12, 0.0, 1.0, "weathering")
    plan["scale"] = _wear_unit(raw, "scale", 4.0, 1e-3, 1e4, "weathering")
    plan["seed"] = _wear_unit(raw, "seed", 0.0, -1e6, 1e6, "weathering")
    plan["regions"] = _wear_regions_plan(raw.get("regions"))
    for key in ("objects", "armature"):
        if raw.get(key) is not None and not plan["regions"]:
            raise MifOpError("weathering.%s says where the region masks go, and no regions were given. NOTHING was "
                             "changed." % key)
    objs = raw.get("objects")
    if objs is not None and (not isinstance(objs, (list, tuple)) or not objs or
                             not all(isinstance(o, str) and o for o in objs)):
        raise MifOpError("weathering.objects is a list of mesh names (default: every mesh using the material), got %r. "
                         "NOTHING was changed." % (objs,))
    plan["objects"] = list(objs) if objs else None
    arm = raw.get("armature")
    if arm is not None and not (isinstance(arm, str) and arm):
        raise MifOpError("weathering.armature is an armature's name, got %r. NOTHING was changed." % (arm,))
    plan["armature"] = arm
    if plan["edgeWear"] is None and plan["grime"] is None and not plan["roughnessVariation"] \
            and not plan["colorVariation"] and not plan["regions"]:
        raise MifOpError("weathering with edgeWear and grime off and no variation adds nothing - pass false to remove "
                         "it. NOTHING was changed.")
    return plan


def _wear_held_value(mat, sock):
    """The MifWear node holding the constant a weathered input had, or None (not weathered, or it was a link)."""
    if not sock.is_linked or not sock.links[0].from_node.name.startswith(_WEAR_PREFIX):
        return None
    const = {"Base Color": "BaseColor", "Roughness": "BaseRoughness", "Metallic": "BaseMetallic"}.get(sock.name)
    node = mat.node_tree.nodes.get("%s %s" % (_WEAR_PREFIX, const)) if const else None
    if node is None:
        raise MifOpError("'%s' is driven by the weathering chain over a texture or node of its own, so it has no "
                         "value to write - remove the weathering (weathering: false) or edit what feeds it. NOTHING "
                         "was changed." % sock.name)
    return node.outputs[0]


def _wear_nodes(tree):
    return [n for n in tree.nodes if n.name.startswith(_WEAR_PREFIX)]


def _weathering_remove(mat, bsdf):
    """Take a MifWear chain out, putting each Principled input back on what fed the chain. Returns the inputs restored."""
    tree = mat.node_tree
    nodes = {n.name: n for n in _wear_nodes(tree)}
    if not nodes:
        return []
    restored = []
    # (principled input, the chain node whose input holds the original source, that input's identifier, const node)
    entries = (("Base Color", "MifWear ColorVar", "Color", "MifWear BaseColor"),
               ("Roughness", "MifWear RoughVar", None, "MifWear BaseRoughness"),
               ("Metallic", "MifWear MetalMix", "A_Float", "MifWear BaseMetallic"),
               ("Normal", "MifWear Bump", "Normal", None))
    for input_name, holder, ident, const in entries:
        sock = bsdf.inputs.get(input_name)
        node = nodes.get(holder)
        if sock is None or node is None:
            continue
        hin = node.inputs[0] if ident is None else next(s for s in node.inputs if s.identifier == ident)
        src = hin.links[0].from_socket if hin.is_linked else None
        for lk in list(sock.links):
            tree.links.remove(lk)
        if src is not None and src.node.name == const:
            sock.default_value = (tuple(src.node.outputs[0].default_value) if input_name == "Base Color"
                                  else float(src.node.outputs[0].default_value))
        elif src is not None and not src.node.name.startswith(_WEAR_PREFIX):
            tree.links.new(src, sock)
        restored.append(input_name)
    for n in list(_wear_nodes(tree)):
        tree.nodes.remove(n)
    return restored


def _wear_new(tree, frame, bl, name, loc, **props):
    n = tree.nodes.new(bl)
    n.name = n.label = "%s %s" % (_WEAR_PREFIX, name)
    n.location = loc
    n.parent = frame
    for k, v in props.items():
        setattr(n, k, v)
    return n


def _wear_maprange(tree, frame, name, loc, src, lo, hi, out_lo=0.0, out_hi=1.0):
    mr = _wear_new(tree, frame, "ShaderNodeMapRange", name, loc, clamp=True)
    if "data_type" in mr.bl_rna.properties:
        mr.data_type = "FLOAT"
    ins = [s for s in mr.inputs if s.enabled] if hasattr(mr.inputs[0], "enabled") else list(mr.inputs)
    by = {s.name: s for s in ins}
    tree.links.new(src, by["Value"])
    by["From Min"].default_value = lo
    by["From Max"].default_value = hi
    by["To Min"].default_value = out_lo
    by["To Max"].default_value = out_hi
    return mr.outputs[0]


def _wear_math(tree, frame, name, loc, op, a, b, clamp=False):
    m = _wear_new(tree, frame, "ShaderNodeMath", name, loc, operation=op, use_clamp=clamp)
    for i, v in enumerate((a, b)):
        if hasattr(v, "is_linked") or hasattr(v, "links"):
            tree.links.new(v, m.inputs[i])
        else:
            m.inputs[i].default_value = v
    return m.outputs[0]


def _wear_mix(tree, frame, name, loc, data_type, a, b, fac):
    mx = _wear_new(tree, frame, "ShaderNodeMix", name, loc)
    mx.data_type = data_type
    kind = "Color" if data_type == "RGBA" else "Float"
    ins = {s.identifier: s for s in mx.inputs}
    for ident, v in (("A_" + kind, a), ("B_" + kind, b), ("Factor_Float", fac)):
        if hasattr(v, "links"):
            tree.links.new(v, ins[ident])
        else:
            ins[ident].default_value = v
    return next(s for s in mx.outputs if s.identifier == "Result_" + kind)


def _wear_source(tree, frame, sock, bl, name, loc):
    """What feeds a Principled input: its link, or its constant moved onto a named RGB / Value node."""
    if sock.is_linked:
        return sock.links[0].from_socket
    n = _wear_new(tree, frame, bl, name, loc)
    n.outputs[0].default_value = sock.default_value
    return n.outputs[0]


# REGIONS FOR CLOTHES (1.4, 2026-11-07): edge wear and AO grime follow the mesh's shape; a garment wears where the body
# works it - knees and seat rubbed pale, cuffs and hems dirty, grease where the hands go. Each region is a mask PER
# VERTEX written as a float attribute 'MifWear_<region>' on every mesh the material is on, from a vertex group, from
# the bones named, or by a rule off the armature's bones (role_bones reads them by name; the character faces -y of the
# armature, its left at +x). The shader reads the attribute (an Attribute node), so the same material on a mesh without
# it is untouched, and bake_texture_set bakes it with the rest.
_WEAR_REGIONS = {
    # region: (kind, radius or band width in metres, strength)
    "knees": ("wear", 0.09, 0.8),
    "seat": ("wear", 0.15, 0.7),
    "cuffs": ("grime", 0.06, 0.8),
    "hems": ("grime", 0.04, 0.6),
    "hands": ("grease", 0.2, 0.8),
    "chest": ("grease", 0.16, 0.5),
}
_WEAR_KINDS = {
    # kind: (color - None lifts the surface's own, as rubbed-pale cloth - , roughness - None adds 0.12 to the surface's)
    "wear": (None, None),
    "grease": ([0.022, 0.017, 0.012, 1.0], 0.38),
    "grime": ([0.11, 0.092, 0.065, 1.0], 0.95),
}
_WEAR_REGION_KEYS = ("region", "strength", "kind", "vertexGroup", "bones", "radius", "color", "roughness")
_WEAR_ATTR = "MifWear_"


def _wear_regions_plan(raw):
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)) or not raw or len(raw) > 12:
        raise MifOpError("weathering.regions is a list of 1-12 {region (%s), strength, kind (%s), vertexGroup | bones, "
                         "radius, color, roughness}, got %r. NOTHING was changed."
                         % (", ".join(_WEAR_REGIONS), ", ".join(_WEAR_KINDS), raw))
    out, seen = [], set()
    for k, blk in enumerate(raw):
        where = "weathering.regions[%d]" % k
        if not isinstance(blk, dict) or "region" not in blk:
            raise MifOpError("%s is {region, ...}, got %r. NOTHING was changed." % (where, blk))
        bad = sorted(set(blk) - set(_WEAR_REGION_KEYS))
        if bad:
            raise MifOpError("%s has no %s - its keys are %s. NOTHING was changed."
                             % (where, ", ".join(bad), ", ".join(_WEAR_REGION_KEYS)))
        name = blk["region"]
        if name not in _WEAR_REGIONS:
            raise MifOpError("%s.region %r is not one of %s. NOTHING was changed."
                             % (where, name, ", ".join(_WEAR_REGIONS)))
        if name in seen:
            raise MifOpError("weathering.regions names '%s' twice - one entry per region. NOTHING was changed." % name)
        seen.add(name)
        kind0, rad0, str0 = _WEAR_REGIONS[name]
        kind = blk.get("kind", kind0)
        if kind not in _WEAR_KINDS:
            raise MifOpError("%s.kind is one of %s, got %r. NOTHING was changed." % (where, ", ".join(_WEAR_KINDS), kind))
        if "vertexGroup" in blk and "bones" in blk:
            raise MifOpError("%s takes vertexGroup OR bones, not both. NOTHING was changed." % where)
        vg = blk.get("vertexGroup")
        if vg is not None and not (isinstance(vg, str) and vg):
            raise MifOpError("%s.vertexGroup is a vertex group's name, got %r. NOTHING was changed." % (where, vg))
        bones = blk.get("bones")
        if bones is not None and (not isinstance(bones, (list, tuple)) or not bones or
                                  not all(isinstance(b, str) and b for b in bones)):
            raise MifOpError("%s.bones is a list of bone names, got %r. NOTHING was changed." % (where, bones))
        kc, kr = _WEAR_KINDS[kind]
        col = _wear_color(blk, "color", where)
        rough = _wear_unit(blk, "roughness", None, 0.0, 1.0, where)
        out.append({"region": name, "kind": kind, "strength": _wear_unit(blk, "strength", str0, 0.0, 1.0, where),
                    "radius": _wear_unit(blk, "radius", rad0, 1e-3, 2.0, where), "vertexGroup": vg,
                    "bones": list(bones) if bones else None, "color": col or kc, "roughness": rough
                    if rough is not None else kr})
    return out


def _wear_falloff(d, r):
    x = min(max(d / r, 0.0), 1.0)
    return 1.0 - x * x * (3.0 - 2.0 * x)


def _wear_ramp(x, lo, hi):
    t = min(max((x - lo) / (hi - lo), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _wear_arm_of(obj, name):
    if name:
        a = get_object(name)
        if a.type != "ARMATURE":
            raise MifOpError("weathering.armature '%s' is a %s, not an ARMATURE. NOTHING was changed." % (a.name, a.type))
        return a
    a = next((m.object for m in obj.modifiers if m.type == "ARMATURE" and m.object is not None), None)
    if a is None and obj.parent is not None and obj.parent.type == "ARMATURE":
        a = obj.parent
    return a


def _wear_region_masks(mat, plan):
    """[(object, {region: [value per vertex]})] and a row per region - every mask computed, and a region that marks
    nothing on any object refused, before anything is written."""
    from mathutils import Vector   # noqa: PLC0415
    from mathutils.geometry import intersect_point_line   # noqa: PLC0415
    from mathutils.kdtree import KDTree   # noqa: PLC0415
    from .ops_garment import role_bones   # noqa: PLC0415
    tail = "NOTHING was changed."
    users = [o for o in bpy.data.objects if o.type == "MESH" and any(sl.material == mat for sl in o.material_slots)]
    if plan["objects"]:
        objs = [get_object(n, want_mesh=True) for n in plan["objects"]]
        stray = [o.name for o in objs if o not in users]
        if stray:
            raise MifOpError("weathering.objects %s do not use '%s' - a region mask is read by the material on the "
                             "mesh it is written to. %s" % (stray, mat.name, tail))
    else:
        objs = users
    if not objs:
        raise MifOpError("no mesh uses '%s', so there is nothing to write its region masks on - assign it first "
                         "(set_material_slots). %s" % (mat.name, tail))
    out, totals, skipped = [], {r["region"]: [0, 0.0, []] for r in plan["regions"]}, []
    for obj in objs:
        me = obj.data
        mw = obj.matrix_world
        m3 = mw.to_3x3().inverted().transposed()
        pts = [mw @ v.co for v in me.vertices]
        nrm = [(m3 @ v.normal).normalized() for v in me.vertices]
        arm = None
        need_arm = [r for r in plan["regions"] if not r["vertexGroup"]]
        if need_arm:
            arm = _wear_arm_of(obj, plan["armature"])
            if arm is None and not plan["objects"]:
                # BY DEFAULT A MESH WITH NO RIG IS PASSED OVER, and said so: a garment's loose pockets and buckles share
                # its material and are not skinned. Named in weathering.objects, it is refused below.
                skipped.append(obj.name)
                continue
            if arm is None:
                raise MifOpError("region %s follows the bones and '%s' has no armature (no Armature modifier, no armature "
                                 "parent) - pass weathering.armature, or a vertexGroup for the region. %s"
                                 % ("/".join(r["region"] for r in need_arm), obj.name, tail))
        roles = role_bones(arm) if arm is not None else {}
        amw = arm.matrix_world if arm is not None else None
        fwd = (amw.to_3x3() @ Vector((0.0, -1.0, 0.0))).normalized() if arm is not None else None

        def seg(bname):
            b = arm.data.bones[bname]
            return amw @ b.head_local, amw @ b.tail_local

        def role(rl, side):
            b = roles.get((rl, side))
            return b

        def need(region, *wanted):
            miss = ["%s_%s" % w for w in wanted if role(*w) is None]
            if miss:
                raise MifOpError("region '%s' reads the %s bones and '%s' has none named for %s (bones are read by "
                                 "name). Pass bones or a vertexGroup for it. %s"
                                 % (region, "/".join(sorted({w[0] for w in wanted})), arm.name, ", ".join(miss), tail))

        boundary = None
        masks = {}
        for r in plan["regions"]:
            name, rad = r["region"], r["radius"]
            if r["vertexGroup"]:
                vg = obj.vertex_groups.get(r["vertexGroup"])
                vals = [0.0] * len(pts)
                if vg is not None:
                    for v in me.vertices:
                        for g in v.groups:
                            if g.group == vg.index:
                                vals[v.index] = g.weight
            elif r["bones"]:
                miss = [b for b in r["bones"] if arm.data.bones.get(b) is None]
                if miss:
                    raise MifOpError("region '%s': '%s' has no bone %s. %s" % (name, arm.name, miss, tail))
                segs = [seg(b) for b in r["bones"]]
                vals = []
                for p in pts:
                    d = min((p - _closest_on(p, h, t, intersect_point_line)).length for h, t in segs)
                    vals.append(_wear_falloff(d, rad))
            elif name in ("knees", "seat", "hands", "chest"):
                # A JOINT'S BAND, measured ALONG the limb: the knee is inside the leg, so a distance from the joint
                # never reached the cloth a leg's radius away (2026-11-07: 0.07 at most on a pair of overalls). Each
                # spot is (point, axis, lateral limit); a band is radius along the axis, within the limit across it.
                up = (amw.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()
                if name == "knees":
                    need(name, ("thigh", "l"), ("thigh", "r"), ("calf", "l"), ("calf", "r"))
                    spots = []
                    for s in "lr":
                        th, tt = seg(role("thigh", s))
                        ch, ct = seg(role("calf", s))
                        spots.append((ch, ((tt - th).normalized() + (ct - ch).normalized()).normalized(), 0.12))
                    face = 1.0
                elif name == "seat":
                    need(name, ("thigh", "l"), ("thigh", "r"))
                    hips = [seg(role("thigh", s))[0] for s in "lr"]
                    spots = [((hips[0] + hips[1]) * 0.5 - up * 0.05, up, 0.3)]
                    face = -1.0
                elif name == "hands":
                    # WHERE THE HANDS GO with the arms down: the wrist's height on each side, the thigh beside it - the
                    # rest pose of an A- or T-posed rig holds them out where no garment is
                    need(name, ("upperarm", "l"), ("upperarm", "r"), ("lowerarm", "l"), ("lowerarm", "r"))
                    spots = []
                    for s in "lr":
                        sh = seg(role("upperarm", s))[0]
                        reach = arm.data.bones[role("upperarm", s)].length + arm.data.bones[role("lowerarm", s)].length
                        hb = role("hand", s)
                        reach += 0.5 * (arm.data.bones[hb].length if hb else 0.0)
                        spots.append((sh - up * reach + fwd * 0.04, None, None))
                    face = 0.0
                else:
                    spines = sorted((b for (rl, sd), b in roles.items() if rl == "spine"),
                                    key=lambda b: arm.data.bones[b].head_local.z)
                    if not spines:
                        need(name, ("spine", "c"))
                    h, t = seg(spines[-1])
                    spots = [((h + t) * 0.5, (t - h).normalized(), 0.3)]
                    face = 1.0

                def spot_d(p, spot):
                    q, ax, lat = spot
                    d = p - q
                    if ax is None:
                        return d.length
                    along = d.dot(ax)
                    return abs(along) if (d - ax * along).length <= lat else math.inf
                vals = []
                for p, n in zip(pts, nrm):
                    v = _wear_falloff(min(spot_d(p, q) for q in spots), rad)
                    if face:
                        v *= _wear_ramp(face * n.dot(fwd), -0.1, 0.5)
                    vals.append(v)
            else:
                # cuffs and hems: a band of `radius` along the open edges - every one for hems, those nearest a hand,
                # forearm, foot or shin for cuffs
                if boundary is None:
                    edge_faces = {}
                    for poly in me.polygons:
                        for ek in poly.edge_keys:
                            edge_faces[ek] = edge_faces.get(ek, 0) + 1
                    boundary = sorted({i for ek, c in edge_faces.items() if c == 1 for i in ek})
                rim = boundary
                if name == "cuffs":
                    need(name, ("calf", "l"), ("calf", "r"))
                    ends = {"hand", "lowerarm", "foot", "calf"}
                    all_segs = [(rl, seg(b)) for (rl, _sd), b in roles.items()]
                    rim = []
                    for i in boundary:
                        best = min(all_segs, key=lambda rs, p=pts[i]: (p - _closest_on(p, rs[1][0], rs[1][1],
                                                                                         intersect_point_line)).length)
                        if best[0] in ends:
                            rim.append(i)
                vals = [0.0] * len(pts)
                if rim:
                    kd = KDTree(len(rim))
                    for k, i in enumerate(rim):
                        kd.insert(pts[i], k)
                    kd.balance()
                    vals = [_wear_falloff(kd.find(p)[2], rad) for p in pts]
            masks[name] = vals
            tt = totals[name]
            tt[0] += sum(1 for x in vals if x >= 0.5)
            tt[1] = max(tt[1], max(vals) if vals else 0.0)
            tt[2].append({"object": obj.name, "verticesAtHalf": sum(1 for x in vals if x >= 0.5),
                          "max": round(max(vals) if vals else 0.0, 4)})
        out.append((obj, masks))
    if not out:
        raise MifOpError("no mesh using '%s' has an armature for the regions to follow (%s) - pass weathering.armature, "
                         "or a vertexGroup for each region. %s" % (mat.name, ", ".join(skipped), tail))
    empty = [n for n, (_c, mx, _rows) in totals.items() if mx < 0.25]
    if empty:
        hint = {r["region"]: ("vertex group '%s'" % r["vertexGroup"]) if r["vertexGroup"] else
                ("bones %s within %g m" % (r["bones"], r["radius"])) if r["bones"] else
                "its rule within %g m" % r["radius"] for r in plan["regions"]}
        raise MifOpError("region(s) %s mark nothing on %s - no vertex reaches a quarter of the mask (%s), and a mask of "
                         "nothing is not a weathering. Check the group "
                         "or bones, or raise radius. %s"
                         % (empty, [o.name for o in objs], "; ".join("%s: %s" % (n, hint[n]) for n in empty), tail))
    rows = [{"region": r["region"], "kind": r["kind"], "strength": r["strength"], "radius": r["radius"],
             "source": ("vertexGroup " + r["vertexGroup"]) if r["vertexGroup"] else
                       ("bones " + ", ".join(r["bones"])) if r["bones"] else "rule",
             "attribute": _WEAR_ATTR + r["region"], "objects": totals[r["region"]][2]} for r in plan["regions"]]
    return out, rows, skipped


def _closest_on(p, h, t, ipl):
    q, f = ipl(p, h, t)
    return h if f <= 0 else t if f >= 1 else q


def _wear_attrs_in_use(skip_mat):
    """Region attribute names another material's MifWear chain still reads."""
    used = set()
    for m in bpy.data.materials:
        if m is skip_mat or not m.use_nodes or m.node_tree is None:
            continue
        for n in m.node_tree.nodes:
            if n.name.startswith(_WEAR_PREFIX + " Region") and n.type == "ATTRIBUTE":
                used.add(n.attribute_name)
    return used


def _wear_clear_attrs(mat):
    """Remove the region attributes from the meshes this material is on, keeping any another weathered material reads.
    Returns the (object, attribute) pairs removed."""
    keep = _wear_attrs_in_use(mat)
    gone = []
    for o in bpy.data.objects:
        if o.type != "MESH" or not any(sl.material == mat for sl in o.material_slots):
            continue
        for a in [a.name for a in o.data.attributes if a.name.startswith(_WEAR_ATTR) and a.name not in keep]:
            o.data.attributes.remove(o.data.attributes[a])
            gone.append((o.name, a))
    return gone


def _wear_write_attrs(masks):
    for obj, regions in masks:
        me = obj.data
        for name, vals in regions.items():
            an = _WEAR_ATTR + name
            old = me.attributes.get(an)
            if old is not None:
                me.attributes.remove(old)
            at = me.attributes.new(an, "FLOAT", "POINT")
            at.data.foreach_set("value", vals)       # computed here, every one a float in 0-1
        me.update()


def _weathering_apply(mat, bsdf, plan):
    tree = mat.node_tree
    base_x, base_y = bsdf.location[0] - 1700, bsdf.location[1] + 300
    frame = tree.nodes.new("NodeFrame")
    frame.name = frame.label = "%s Frame" % _WEAR_PREFIX
    s_col, s_rough = bsdf.inputs["Base Color"], bsdf.inputs["Roughness"]
    s_metal, s_nrm = bsdf.inputs["Metallic"], bsdf.inputs["Normal"]
    base_col = _wear_source(tree, frame, s_col, "ShaderNodeRGB", "BaseColor", (base_x, base_y))
    base_r = _wear_source(tree, frame, s_rough, "ShaderNodeValue", "BaseRoughness", (base_x, base_y - 200))
    # ---- coordinates and the two noises (object space: the same pattern on every bake, no UVs needed)
    tc = _wear_new(tree, frame, "ShaderNodeTexCoord", "Coords", (base_x - 600, base_y - 600))
    mp = _wear_new(tree, frame, "ShaderNodeMapping", "Mapping", (base_x - 400, base_y - 600))
    tree.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    mp.inputs["Scale"].default_value = (plan["scale"],) * 3
    seed = plan["seed"]
    mp.inputs["Location"].default_value = (seed * 13.37, seed * 7.31, seed * 3.11)
    noises = []
    # Noise1 breaks the wear into chips (fine: an edge is a few cm wide), Noise2 is the color / roughness variation,
    # Noise3 the grime's patches
    for i, (scl, det) in enumerate(((3.0, 8.0), (4.0, 4.0), (1.7, 5.0))):
        nz = _wear_new(tree, frame, "ShaderNodeTexNoise", "Noise%d" % (i + 1), (base_x - 200, base_y - 500 - 220 * i))
        tree.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
        nz.inputs["Scale"].default_value = scl
        nz.inputs["Detail"].default_value = det
        if "Roughness" in nz.inputs:
            nz.inputs["Roughness"].default_value = 0.6
        if "Lacunarity" in nz.inputs:
            nz.inputs["Lacunarity"].default_value = 2.0
        if i == 2 and "Distortion" in nz.inputs:
            nz.inputs["Distortion"].default_value = 0.5
        # contrast-stretched 0-1: a Noise Fac sits near 0.5 +- 0.15 with detail on every build
        noises.append(_wear_maprange(tree, frame, "Noise%dC" % (i + 1), (base_x, base_y - 500 - 220 * i),
                                     nz.outputs["Fac"], 0.32, 0.68))
    n_break, n_fine, n_grime = noises
    x = base_x + 250
    wear = grime = None
    # ---- edge mask
    edge_p = plan["edgeWear"]
    if edge_p is not None:
        if edge_p["source"] == "bevel":
            bev = _wear_new(tree, frame, "ShaderNodeBevel", "Bevel", (x, base_y - 1200))
            bev.samples = 8
            bev.inputs["Radius"].default_value = edge_p["width"]
            geo = _wear_new(tree, frame, "ShaderNodeNewGeometry", "Geometry", (x, base_y - 1400))
            dot = _wear_new(tree, frame, "ShaderNodeVectorMath", "EdgeDot", (x + 200, base_y - 1250),
                            operation="DOT_PRODUCT")
            tree.links.new(bev.outputs["Normal"], dot.inputs[0])
            tree.links.new(geo.outputs["Normal"], dot.inputs[1])
            dot_out = next(s for s in dot.outputs if s.name == "Value")
            curv = _wear_math(tree, frame, "Curvature", (x + 400, base_y - 1250), "SUBTRACT", 1.0, dot_out)
            # 1 - cos: about 0.29 on a 90-degree edge, falling to 0 at `width` from it - a gradient across the band,
            # so the noise below decides how far in from the edge each chip reaches
            edge = _wear_maprange(tree, frame, "EdgeMask", (x + 600, base_y - 1250), curv, 0.0, 0.15)
        else:
            geo = _wear_new(tree, frame, "ShaderNodeNewGeometry", "Geometry", (x, base_y - 1400))
            edge = _wear_maprange(tree, frame, "EdgeMask", (x + 600, base_y - 1250), geo.outputs["Pointiness"],
                                  0.52, 0.6)
        mod = _wear_math(tree, frame, "EdgeBreak", (x + 800, base_y - 1100), "ADD", n_break, 0.1)
        t = _wear_math(tree, frame, "EdgePatchy", (x + 1000, base_y - 1100), "MULTIPLY", edge, mod)
        thr = 1.0 - edge_p["amount"]
        wear = _wear_maprange(tree, frame, "Wear", (x + 1200, base_y - 1100), t, thr, thr + 0.08)
    # ---- grime mask
    grime_p = plan["grime"]
    if grime_p is not None:
        ao = _wear_new(tree, frame, "ShaderNodeAmbientOcclusion", "AO", (x, base_y - 1700), samples=16,
                       only_local=True)
        ao.inputs["Distance"].default_value = grime_p["distance"]
        occl = _wear_math(tree, frame, "Occlusion", (x + 200, base_y - 1700), "SUBTRACT", 1.0, ao.outputs["AO"])
        occ = _wear_maprange(tree, frame, "Cavity", (x + 400, base_y - 1700), occl, 0.01, 0.12)
        mod = _wear_math(tree, frame, "GrimeBreak", (x + 800, base_y - 1600), "ADD", n_grime, 0.35)
        t = _wear_math(tree, frame, "GrimePatchy", (x + 1000, base_y - 1600), "MULTIPLY", occ, mod)
        thr = 1.0 - grime_p["amount"]
        grime = _wear_maprange(tree, frame, "Grime", (x + 1200, base_y - 1600), t, thr * 0.8, thr * 0.8 + 0.35)
    # ---- region masks (clothes): the vertex attribute times strength, broken up by the grime noise
    regions = []
    for k, rg in enumerate(plan.get("regions") or []):
        y = base_y - 2000 - 260 * k
        at = _wear_new(tree, frame, "ShaderNodeAttribute", "Region %s" % rg["region"], (x, y))
        at.attribute_name = _WEAR_ATTR + rg["region"]
        t = _wear_math(tree, frame, "Region %s Strength" % rg["region"], (x + 200, y), "MULTIPLY",
                       at.outputs["Fac"], rg["strength"])
        mod = _wear_math(tree, frame, "Region %s Break" % rg["region"], (x + 400, y + 80), "ADD", n_grime, 0.4)
        t = _wear_math(tree, frame, "Region %s Patchy" % rg["region"], (x + 600, y), "MULTIPLY", t, mod, clamp=True)
        regions.append((rg, _wear_maprange(tree, frame, "Region %s Mask" % rg["region"], (x + 800, y), t, 0.05, 0.75)))
    # ---- base color: value breakup, then the wear color on the edges, then the grime in the cavities
    cv = plan["colorVariation"]
    hsv = _wear_new(tree, frame, "ShaderNodeHueSaturation", "ColorVar", (x + 1400, base_y))
    tree.links.new(base_col, hsv.inputs["Color"])
    tree.links.new(_wear_maprange(tree, frame, "ValueVar", (x + 1200, base_y - 150), n_fine, 0.0, 1.0,
                                  1.0 - cv, 1.0 + cv), hsv.inputs["Value"])
    # SATURATION AND A LITTLE HUE VARY TOO, at a broader scale than the value: a value-only breakup measured a
    # saturation sd of 0.010 on a painted box against 0.040-0.138 on the worn wood and paint scans; +-1.5 x cv on the
    # chip noise overshot (0.174 on the metal against 0.096-0.103 for the painted metal scans) and read as clouds
    # (tools/measure_weathering.py)
    tree.links.new(_wear_maprange(tree, frame, "SatVar", (x + 1200, base_y - 300), n_grime, 0.0, 1.0,
                                  1.0 - 0.75 * cv, 1.0 + 0.75 * cv), hsv.inputs["Saturation"])
    tree.links.new(_wear_maprange(tree, frame, "HueVar", (x + 1200, base_y - 450), n_break, 0.0, 1.0,
                                  0.5 - 0.05 * cv, 0.5 + 0.05 * cv), hsv.inputs["Hue"])
    col = hsv.outputs["Color"]
    if wear is not None:
        if edge_p["color"] is not None:
            wear_col = edge_p["color"]
        else:
            # NO COLOR GIVEN: the surface's own color, lighter and grayer - paint or oxide rubbed back to the
            # material under it. Derived from the input, so a textured base wears in its own colors.
            lift = _wear_new(tree, frame, "ShaderNodeHueSaturation", "WearColor", (x + 1400, base_y + 250))
            lift.inputs["Saturation"].default_value = 0.45
            lift.inputs["Value"].default_value = 1.9
            tree.links.new(base_col, lift.inputs["Color"])
            wear_col = lift.outputs["Color"]
        col = _wear_mix(tree, frame, "WearMix", (x + 1650, base_y), "RGBA", col, wear_col, wear)
    if grime is not None:
        col = _wear_mix(tree, frame, "GrimeMix", (x + 1850, base_y), "RGBA", col, grime_p["color"], grime)
    for k, (rg, m) in enumerate(regions):
        rc = rg["color"]
        if rc is None:
            # RUBBED PALE: the cloth's own color, lighter and grayer - faded denim at a knee, not a painted patch
            lift = _wear_new(tree, frame, "ShaderNodeHueSaturation", "Region %s Color" % rg["region"],
                             (x + 1850 + 200 * k, base_y + 250))
            lift.inputs["Saturation"].default_value = 0.55
            lift.inputs["Value"].default_value = 1.6
            tree.links.new(base_col, lift.inputs["Color"])
            rc = lift.outputs["Color"]
        col = _wear_mix(tree, frame, "Region %s ColorMix" % rg["region"], (x + 2050 + 200 * k, base_y), "RGBA", col,
                        rc, m)
    tree.links.new(col, s_col)
    # ---- roughness: noise variation, then smoother where handled, rougher where dirty
    rv = plan["roughnessVariation"]
    var = _wear_maprange(tree, frame, "RoughVarRange", (x + 1200, base_y - 400), n_fine, 0.0, 1.0, -rv, rv)
    r = _wear_math(tree, frame, "RoughVar", (x + 1400, base_y - 400), "ADD", base_r, var, clamp=True)
    if wear is not None:
        if edge_p["roughness"] is not None:
            wr = edge_p["roughness"]
        else:
            wr = _wear_math(tree, frame, "WearRough", (x + 1400, base_y - 600), "MULTIPLY", base_r, 0.6, clamp=True)
        r = _wear_mix(tree, frame, "WearRoughMix", (x + 1650, base_y - 400), "FLOAT", r, wr, wear)
    if grime is not None:
        r = _wear_mix(tree, frame, "GrimeRoughMix", (x + 1850, base_y - 400), "FLOAT", r, grime_p["roughness"], grime)
    for k, (rg, m) in enumerate(regions):
        rr = rg["roughness"]
        if rr is None:
            rr = _wear_math(tree, frame, "Region %s Rough" % rg["region"], (x + 1850 + 200 * k, base_y - 600), "ADD",
                            base_r, 0.12, clamp=True)
        r = _wear_mix(tree, frame, "Region %s RoughMix" % rg["region"], (x + 2050 + 200 * k, base_y - 400), "FLOAT", r,
                      rr, m)
    tree.links.new(r, s_rough)
    spliced = ["baseColor", "roughness"]
    # ---- metallic: bare metal where the coating wore through
    if wear is not None and edge_p["metallic"] is not None:
        base_m = _wear_source(tree, frame, s_metal, "ShaderNodeValue", "BaseMetallic", (base_x, base_y - 400))
        m = _wear_mix(tree, frame, "MetalMix", (x + 1650, base_y - 700), "FLOAT", base_m, edge_p["metallic"], wear)
        tree.links.new(m, s_metal)
        spliced.append("metallic")
    # ---- normal: the chipped edge sits a little below the coat
    if wear is not None and edge_p["bump"]:
        bump = _wear_new(tree, frame, "ShaderNodeBump", "Bump", (x + 1850, base_y - 900), invert=True)
        bump.inputs["Strength"].default_value = edge_p["bump"]
        bump.inputs["Distance"].default_value = 0.002
        tree.links.new(wear, bump.inputs["Height"])
        if s_nrm.is_linked:
            tree.links.new(s_nrm.links[0].from_socket, bump.inputs["Normal"])
        tree.links.new(bump.outputs["Normal"], s_nrm)
        spliced.append("normal")
    return {"spliced": spliced, "nodes": len(_wear_nodes(tree)) + 0,
            "edgeWear": edge_p, "grime": grime_p, "roughnessVariation": rv, "colorVariation": cv,
            "scale": plan["scale"], "seed": seed, "regions": plan.get("regionRows") or [],
            "regionsSkipped": plan.get("regionsSkipped") or [],
            "preview": "Cycles renders the whole chain; EEVEE shows no edge wear (Bevel and Pointiness are Cycles "
                       "nodes) - bake_texture_set types BASE_COLOR, ROUGHNESS, METALLIC, NORMAL carries it to any "
                       "engine"}


def op_set_material_properties(params):
    """Write Principled BSDF values on an existing material, resolving names per version.

    EVERY PROPERTY IS RESOLVED THROUGH AN ALIAS LIST, and one that resolves on NO alias is
    REFUSED rather than skipped. Blender renamed these inputs between 3.6 and 4.0 -
    "Specular" became "Specular IOR Level", "Emission" became "Emission Color" - and this
    addon supports both. Writing to a name that does not exist on the running version would
    land nowhere and leave a material that looks wrong with nothing to read, which is the
    exact silent-failure shape the rest of this project refuses.

    `resolvedInputs` reports the real socket each property landed on, so a caller can see
    which spelling this Blender used.

    `rename` gives the material a new name, with or without values, and a material with no
    Principled BSDF can be renamed. NO OP RENAMED A MATERIAL: benchmark round ten's agents searched
    "rename material" 9 times in 15 runs (A3, A5, A8) to give build_rock's, build_plant's and
    build_tree's materials the slot names their task asked for, and each made a new material
    instead. A name another material holds is REFUSED, never left to Blender - its .001 suffix
    survives FBX export, and on 3.6 the incumbent is the one renamed (assign_id_name).

    SUBSURFACE (2026-10-06): subsurfaceWeight, subsurfaceScale, subsurfaceRadius [r, g, b], subsurfaceIor and
    subsurfaceAnisotropy are sockets resolved like the rest; subsurfaceMethod (BURLEY | RANDOM_WALK |
    RANDOM_WALK_SKIN, and RANDOM_WALK_LEGACY on 5.2) is the node's own enum, checked before the first write. Skin
    and wax want weight 1 and the radius reset from (1.0, 0.2, 0.1) to (1, 1, 1) so it stops reading red.

    WEATHERING (1.4): weathering {edgeWear, grime, roughnessVariation, colorVariation, scale, seed} splices wear in front
    of the Principled inputs, every mask driven by the mesh - edgeWear {amount 0.5, width 0.02 m, source bevel |
    pointiness, color, roughness, metallic, bump} on the convex edges (curvature from a Bevel node, or Pointiness),
    grime {amount 0.5, distance 0.2 m, color, roughness 0.9} where Ambient Occlusion is low, both broken up by object-space
    noise; roughnessVariation (0.15) and colorVariation (0.12) vary the whole surface. Whatever fed an input stays under
    the chain (a texture weathers in its own colors); a later baseColor / roughness / metallic writes the value under
    it. Calling again replaces it; weathering false removes it and restores each input. Cycles renders it (Bevel and
    Pointiness are Cycles nodes); bake_texture_set BASE_COLOR, ROUGHNESS, METALLIC, NORMAL carries it to Unreal.

    REGIONS FOR CLOTHES (1.4): weathering.regions [{region, strength, kind, vertexGroup | bones, radius, color,
    roughness}] - knees and seat (kind wear: the cloth rubbed pale and rougher, front and back), cuffs and hems (grime:
    a band along the open edges nearest the hands and feet, or all of them), hands (grease, darker and glossier, on the
    thighs where the hands hang) and chest (grease, the front at the top spine bone) - each a float attribute
    'MifWear_<region>' per vertex on every mesh using the material (weathering.objects narrows it), from a vertex group,
    the bones named (within radius) or a rule off the mesh's armature (weathering.armature, default its Armature
    modifier's). A region that marks nothing is refused before anything is written.
    """
    reject_unknown(params, SET_MATERIAL_PROPERTY_PARAMS, "set_material_properties")
    mat_name = take(params, "material", "name", required=True, kind=str)
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        have = sorted(m.name for m in bpy.data.materials)[:25]
        raise MifOpError("no material named '%s'. This file has: %s"
                         % (mat_name, ", ".join(have) or "(none)"))
    new_name = take(params, "rename", kind=str)
    if new_name is not None:
        new_name = new_name.strip()
        clash = bpy.data.materials.get(new_name)
        if not new_name:
            raise MifOpError("rename is empty - pass the material's new name. NOTHING was changed.")
        if clash is not None and clash is not mat:
            raise MifOpError("a material named '%s' already exists, and Blender would resolve the clash by renaming "
                             "one of the two to '%s.001' - which survives FBX export. Pick another name, or rename or "
                             "delete the existing one first. NOTHING was changed." % (new_name, new_name))
        if len(new_name) > 63 and bpy.app.version < (5, 0, 0):
            raise MifOpError("'%s' is %d characters and this Blender (%s) truncates names at 63, so the name you got "
                             "back would not be the one asked for. Shorten it. NOTHING was changed."
                             % (new_name, len(new_name), bpy.app.version_string))

    viewport = _viewport_plan(params)
    wanted = {k: params[k] for k in PRINCIPLED_ALIASES if k in params}
    # THE SUBSURFACE METHOD IS A NODE PROPERTY, not a socket (Burley, Random Walk, Random Walk Skin; 5.2 adds Random
    # Walk Legacy) - resolved against the node's own enum below, before the first write.
    method = take(params, "subsurfaceMethod", kind=str)
    # WEATHERING (1.4): validated whole here, before any write - see _weathering_apply
    wplan = _weathering_plan(params["weathering"]) if params.get("weathering") is not None else None
    if not wanted and new_name is None and not viewport and method is None and wplan is None:
        raise MifOpError(
            "no property to set. Accepted: %s, %s, subsurfaceMethod, weathering, or rename. NOTHING was changed."
            % (", ".join(sorted(PRINCIPLED_ALIASES)), ", ".join(sorted(_VIEWPORT_MAT))))
    if not wanted and new_name is None and method is None and wplan is None:
        for attr, value in viewport.items():
            setattr(mat, attr, value)
        out = _material_json(mat)
        out["applied"] = sorted(k for k in _VIEWPORT_MAT if k in params)
        out["viewport"] = _viewport_read(mat)
        return out
    if not wanted and method is None and wplan is None:
        was = mat.name
        # the clash is refused above, so this gets the plain name; assign_id_name is the one way an ID is named here
        assign_id_name(mat, new_name, bpy.data.materials)
        if mat.name != new_name:            # read back: Blender resolves a name silently
            raise MifOpError("the material came out named '%s', not '%s'." % (mat.name, new_name))
        for attr, value in viewport.items():
            setattr(mat, attr, value)
        out = _material_json(mat)
        out["renamedFrom"] = was
        out["applied"] = sorted(k for k in _VIEWPORT_MAT if k in params)
        if viewport:
            out["viewport"] = _viewport_read(mat)
        return out
    bsdf = _principled(mat)

    resolved = {}
    unresolved = []
    # RESOLVE EVERY NAME BEFORE WRITING ANY, so a bad one in the middle cannot leave the
    # material half-written - the same all-or-nothing rule the UE side's show flags use.
    for prop in wanted:
        sock = _resolve_input(bsdf, prop)
        if sock is None:
            unresolved.append(prop)
        else:
            resolved[prop] = sock
    if unresolved:
        raise MifOpError(
            "this Blender (%s) has no Principled input for: %s. Tried these spellings: %s. "
            "The inputs that DO exist are: %s. Nothing was written - the whole call is "
            "refused so a rename cannot half-apply."
            % (bpy.app.version_string, ", ".join(sorted(unresolved)),
               "; ".join("%s -> %s" % (p, "/".join(PRINCIPLED_ALIASES[p]) or "(none)")
                         for p in sorted(unresolved)),
               ", ".join(i.name for i in bsdf.inputs)))

    # EVERY VALUE COERCED AND EVERY SOCKET LINK-CHECKED BEFORE THE FIRST WRITE. The names were
    # already resolved above; doing the values in the same pass is what makes "nothing was written"
    # true on a refusal rather than merely claimed.
    staged = {}
    for prop, sock in resolved.items():
        # A WEATHERED INPUT IS WRITTEN UNDER THE WEATHERING: the chain keeps the constant it replaced on a named node
        # (MifWear BaseColor / BaseRoughness / BaseMetallic), and that node is the value this call means
        held = _wear_held_value(mat, sock)
        if held is not None:
            resolved[prop] = held
            staged[prop] = _coerce_socket(sock, prop, wanted[prop])
            continue
        _check_unlinked(sock, prop)
        staged[prop] = _coerce_socket(sock, prop, wanted[prop])
    method_id = (_enum_value_or_refuse(bsdf, "subsurface_method", method, "subsurfaceMethod")
                 if method is not None else None)
    # REGION MASKS COMPUTED BEFORE THE FIRST WRITE: a region that marks nothing, a missing armature or bone refuses here
    region_masks = None
    if wplan is not None and not wplan["remove"] and wplan["regions"]:
        region_masks, wplan["regionRows"], wplan["regionsSkipped"] = _wear_region_masks(mat, wplan)
    for prop, sock in resolved.items():
        sock.default_value = staged[prop]
    wear_out = None
    if wplan is not None:
        restored = _weathering_remove(mat, bsdf)
        cleared = _wear_clear_attrs(mat)
        if region_masks:
            _wear_write_attrs(region_masks)
        wear_out = {"removed": True, "restored": restored} if wplan["remove"] else _weathering_apply(mat, bsdf, wplan)
        if cleared and wplan["remove"]:
            wear_out["regionAttributesRemoved"] = ["%s.%s" % c for c in cleared]
        if not wplan["remove"]:
            wear_out["replacedPrevious"] = bool(restored)
    if method_id is not None:
        bsdf.subsurface_method = method_id
    for attr, value in viewport.items():
        setattr(mat, attr, value)
    was = mat.name
    if new_name is not None:
        assign_id_name(mat, new_name, bpy.data.materials)
        if mat.name != new_name:
            raise MifOpError("the values were written, and the material came out named '%s', not '%s'."
                             % (mat.name, new_name))

    out = _material_json(mat)
    if new_name is not None:
        out["renamedFrom"] = was
    out["applied"] = (sorted(wanted) + (["subsurfaceMethod"] if method_id is not None else [])
                      + sorted(k for k in _VIEWPORT_MAT if k in params))
    if method_id is not None:
        out["subsurfaceMethod"] = bsdf.subsurface_method      # read back off the node
    if viewport:
        out["viewport"] = _viewport_read(mat)
    out["resolvedInputs"] = {p: ("%s.%s" % (s.node.name, s.name) if s.node.name.startswith(_WEAR_PREFIX) else s.name)
                             for p, s in resolved.items()}
    if wear_out is not None:
        out["weathering"] = wear_out
        out["applied"].append("weathering")
    out["blenderVersion"] = bpy.app.version_string
    return out


def op_list_materials(params):
    """Every material in the file, with its user count. The read half that did not exist.

    `unused` (nothing uses it: no users, or only its fake user) matters for export: such a material is not written to
    an FBX at all, so a material that was created and never assigned silently does not arrive in Unreal. A fake user
    keeps a material in the file; it does not put it on anything, and counting it made usedOnly keep every kept material.
    """
    reject_unknown(params, ("nameContains", "usedOnly", "limit", "nameExcludes"), "list_materials")
    contains = take(params, "nameContains", kind=str)
    used_only = take_bool(params, "usedOnly", default=False)

    rows = []
    for mat in bpy.data.materials:
        if contains and contains.lower() not in mat.name.lower():
            continue
        if used_only and mat.users - int(mat.use_fake_user) == 0:
            continue
        rows.append({
            "name": mat.name,
            "users": mat.users,
            "fakeUser": bool(mat.use_fake_user),
            "useNodes": bool(mat.use_nodes),
            "hasPrincipled": bool(
                mat.use_nodes and mat.node_tree
                and any(n.type == "BSDF_PRINCIPLED" for n in mat.node_tree.nodes)),
        })
    unused = [r["name"] for r in rows if r["users"] - int(r["fakeUser"]) == 0]
    rows = exclude_rows(rows, params)
    matched = len(rows)
    rows, extra = limit_rows(rows, params, "material(s)")
    out = dict({"materials": rows, "count": matched, "unused": unused}, **extra)
    if unused:
        out["unusedNote"] = ("%d material(s) are on nothing (no users, or only a fake user). A material on nothing "
                             "is NOT written to an FBX, so one created and never assigned will not "
                             "arrive in Unreal." % len(unused))
    return out


def op_describe_material(params):
    """One material in full: Principled values, node tree shape, and texture file paths.

    The texture paths are the point for a pipeline - they are what an Unreal-side import has
    to resolve, and nothing in the addon reported them before.

    `nodes` ADDS EVERY NODE'S SETTINGS AND SOCKET VALUES, which is what makes this a read-back for
    a procedural material rather than a summary of one. Without it a graph of noise, a colour ramp
    and a bump node reports as three names and a link list - correct, and no help at all to
    somebody asking whether the scale they wrote landed. `links` gives the connections on their
    own, which is the cheaper answer when the shape is the question.
    """
    reject_unknown(params, ("material", "name", "links", "nodes"), "describe_material")
    mat_name = take(params, "material", "name", required=True, kind=str)
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        have = sorted(m.name for m in bpy.data.materials)[:25]
        raise MifOpError("no material named '%s'. This file has: %s"
                         % (mat_name, ", ".join(have) or "(none)"))
    return _material_json(mat, deep=take_bool(params, "links", default=False),
                          nodes=take_bool(params, "nodes", default=False))


_BAKE_TYPES = {
    "AO": "AO",
    "NORMAL": "NORMAL",
    "DIFFUSE": "DIFFUSE",
    "COMBINED": "COMBINED",
    "ROUGHNESS": "ROUGHNESS",
    "EMIT": "EMIT",
    "GLOSSY": "GLOSSY",
    "SHADOW": "SHADOW",
    # Pointiness (Cycles' per-vertex convexity: 0.5 flat, above convex, below concave) baked through a temporary
    # emission on the SOURCE objects - selected-to-active, or the object itself (a self-bake, 1.3).
    "CURVATURE": "EMIT",
    # 1.3 gap audit (2026-10-19): the rest of a PBR set through the same temporary emission - _EMIT_ROUTE below
    "METALLIC": "EMIT", "ALPHA": "EMIT", "MATERIAL_ID": "EMIT", "POSITION": "EMIT", "THICKNESS": "EMIT",
    # 1.4 (2026-10-30): the albedo. A Cycles DIFFUSE color bake is base color x (1 - metallic), so a worn metal's
    # bare edges bake BLACK; BASE_COLOR routes the Base Color input itself through the emission, metallic or not
    "BASE_COLOR": "EMIT",
    "HEIGHT": "EMIT",          # composed of three internal bakes (_bake_height)
    "_POSITION_RAW": "EMIT", "_NORMAL_WORLD": "EMIT",
}


def _pixel_signature(image):
    """(min, max, mean) over the raw float buffer - the cheapest honest 'did anything change'."""
    px = image.pixels[:]
    if not px:
        return (0.0, 0.0, 0.0)
    return (round(min(px), 6), round(max(px), 6), round(sum(px) / len(px), 6))


# ---------------------------------------------------------------------------------------------------------------------
# NORMAL-MAP BLOB FIX (2026-10-06, spec "A typed high-to-low bake" / "Fix normal-map blobs where bake rays hit a
# neighbouring surface"). Where cloth pinches, a selected-to-active bake's rays land on the facing surface and the map
# gets connected patches of extreme tilt; the garment work repaired them offline (fix_normal_blobs.py). The same rule:
# texels tilted past maxTiltDeg, OPENED - eroded one texel, components under minBlobPx dropped, then grown back by
# _BLOB_GROW texels - so a thin real stitch line survives, then filled from the texels round them by repeated
# neighbour averaging and renormalised.

_BLOB_GROW = 6


def _shift(a, dy, dx, fill):
    import numpy as np   # noqa: PLC0415 - Blender ships numpy; only these helpers need it
    out = np.full_like(a, fill)
    h, w = a.shape[:2]
    ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    out[ys, xs] = a[yd, xd]
    return out


_NEIGH = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]


def fix_normal_blobs(image, max_tilt_deg, min_blob_px):
    """Repair blobs of extreme tilt in a tangent-space normal map image in place. Returns the counts."""
    import numpy as np   # noqa: PLC0415
    w, h = image.size
    px = np.array(image.pixels[:], dtype=np.float64).reshape(h, w, 4)
    n = px[..., :3] * 2.0 - 1.0
    length = np.linalg.norm(n, axis=2)
    nz = np.where(length > 1e-9, n[..., 2] / np.maximum(length, 1e-9), 1.0)
    tilted = np.degrees(np.arccos(np.clip(nz, -1.0, 1.0))) > max_tilt_deg
    eroded = tilted.copy()
    for dy, dx in _NEIGH:
        eroded &= _shift(tilted, dy, dx, False)
    # components of the eroded mask, 8-connected; the small ones are dropped
    seen = np.zeros_like(eroded)
    keep = np.zeros_like(eroded)
    blobs = 0
    for y0, x0 in zip(*np.nonzero(eroded)):
        if seen[y0, x0]:
            continue
        stack, comp = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy, dx in _NEIGH:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and eroded[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        if len(comp) >= min_blob_px:
            blobs += 1
            for y, x in comp:
                keep[y, x] = True
    mask = keep.copy()
    for _ in range(_BLOB_GROW):     # grown back past the erosion
        grown = mask.copy()
        for dy, dx in _NEIGH:
            grown |= _shift(mask, dy, dx, False)
        mask = grown
    filled = int(mask.sum())
    if filled == 0:
        return {"tiltedTexels": int(tilted.sum()), "blobs": 0, "texelsFilled": 0}
    if filled == w * h:
        raise MifOpError("every texel of '%s' is in a blob, so there is nothing to fill from - this is not a normal "
                         "map with blobs. NOTHING was changed." % image.name)
    known = ~mask
    vals = np.where(known[..., None], n, 0.0)
    while not known.all():
        acc = np.zeros_like(vals)
        cnt = np.zeros((h, w))
        for dy, dx in _NEIGH:
            k = _shift(known, dy, dx, False)
            acc += np.where(k[..., None], _shift(vals, dy, dx, 0.0), 0.0)
            cnt += k
        front = (~known) & (cnt > 0)
        if not front.any():
            break
        vals[front] = acc[front] / cnt[front][:, None]
        known = known | front
    vals /= np.maximum(np.linalg.norm(vals, axis=2), 1e-9)[..., None]
    out = px.copy()
    out[..., :3] = np.where(mask[..., None], vals * 0.5 + 0.5, px[..., :3])
    image.pixels[:] = out.ravel().tolist()
    image.update()
    return {"tiltedTexels": int(tilted.sum()), "blobs": blobs, "texelsFilled": filled}


def _parse_blob_fix(raw, tail):
    if raw is None or raw is False:
        return None
    if raw is True:
        raw = {}
    if not isinstance(raw, dict) or set(raw) - {"maxTiltDeg", "minBlobPx"}:
        raise MifOpError("blobFix is true or {maxTiltDeg (70), minBlobPx (40)}, got %r. %s" % (raw, tail))
    tilt = finite_float(raw.get("maxTiltDeg", 70.0), "maxTiltDeg")
    blob = raw.get("minBlobPx", 40)
    if not 0 < tilt < 180:
        raise MifOpError("maxTiltDeg must be between 0 and 180, got %g. %s" % (tilt, tail))
    if isinstance(blob, bool) or not isinstance(blob, int) or blob < 1:
        raise MifOpError("minBlobPx must be a whole number of at least 1, got %r. %s" % (blob, tail))
    return tilt, blob


def op_fix_normal_map(params):
    """Repair normal-map blobs - texels tilted past maxTiltDeg where bake rays hit a facing surface - in an image.

    image names an image in this file, or filepath loads one; the fix is written to the image (and saved to
    outputPath, or back to filepath with overwrite true). blobFix's rule (bake_texture blobFix runs the same one):
    texels tilted past maxTiltDeg (default 70) are eroded one texel so a thin real stitch line drops out, components
    under minBlobPx (default 40) are dropped, the rest grown back by 6 texels, and that mask is filled from the texels
    round it by repeated neighbour averaging and renormalised. Reply: tiltedTexels, blobs, texelsFilled.
    """
    tail = "NOTHING was changed."
    reject_unknown(params, ("image", "filepath", "outputPath", "overwrite", "maxTiltDeg", "minBlobPx"),
                   "fix_normal_map")
    name = take(params, "image", default=None, kind=str)
    path = take(params, "filepath", default=None, kind=str)
    out_path = take(params, "outputPath", default=None, kind=str)
    overwrite = take_bool(params, "overwrite", default=False)
    rule = _parse_blob_fix({k: params[k] for k in ("maxTiltDeg", "minBlobPx") if params.get(k) is not None}, tail)
    if (name is None) == (path is None):
        raise MifOpError("pass image (a name in this file) OR filepath. %s" % tail)
    if path is not None and out_path is None and not overwrite:
        raise MifOpError("a file is fixed into outputPath, or back into filepath with overwrite true. %s" % tail)
    if overwrite and out_path is not None:
        raise MifOpError("overwrite writes back to filepath; outputPath writes elsewhere - pass one. %s" % tail)
    if out_path:
        check_output_path(out_path, bpy.path.abspath(out_path), "fixed")
    loaded = None
    if path is not None:
        full = os.path.abspath(bpy.path.abspath(path))
        if not os.path.isfile(full):
            raise MifOpError("no file at %r. %s" % (path, tail))
        # A scratch datablock, removed below; an 8-bit file's pixels read back as stored whatever its colour space.
        loaded = image = bpy.data.images.load(full, check_existing=False)
    else:
        image = bpy.data.images.get(name)
        if image is None:
            raise MifOpError("no image named %r. %s" % (name, tail))
    try:
        if not image.size[0]:
            raise MifOpError("'%s' has no pixels to fix. %s" % (image.name, tail))
        stats = fix_normal_blobs(image, *rule)
        saved = None
        if path is not None or out_path:
            target = abs_output_path(out_path or path)
            image.filepath_raw = target
            image.file_format = {".png": "PNG", ".tga": "TARGA", ".jpg": "JPEG", ".jpeg": "JPEG",
                                 ".exr": "OPEN_EXR"}.get(os.path.splitext(target)[1].lower(), "PNG")
            image.save()
            saved = target
    finally:
        if loaded is not None:
            bpy.data.images.remove(loaded)
    out = {"image": None if loaded is not None else image.name, "savedTo": saved, "maxTiltDeg": rule[0],
           "minBlobPx": rule[1], "grownBy": _BLOB_GROW}
    out.update(stats)
    if not stats["texelsFilled"]:
        out["note"] = "no blob of %d or more texels tilted past %g degrees - nothing was changed." % rule[::-1]
    return out


def _parse_bake_pieces(params, tail):
    """[(low, [sources], cage or None)] from bake_texture pieces, every object checked, nothing written."""
    raw = params.get("pieces")
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)) or not raw:
        raise MifOpError("pieces is a list of {object, source, cage} - the low part, its high part(s), optionally "
                         "a cage. %s" % tail)
    out, seen = [], {}
    for k, row in enumerate(raw):
        if not isinstance(row, dict):
            raise MifOpError("pieces[%d] must be {object, source, cage}, got %r. %s" % (k, row, tail))
        bad = sorted(set(row) - {"object", "source", "cage"})
        if bad:
            raise MifOpError("pieces[%d] has unknown key(s) %s - accepted: object, source, cage. %s" % (k, bad, tail))
        if not isinstance(row.get("object"), str):
            raise MifOpError("pieces[%d].object names the low part baked onto. %s" % (k, tail))
        low = get_object(row["object"], want_mesh=True)
        if not low.data.polygons:
            raise MifOpError("pieces[%d] '%s' has no faces. %s" % (k, low.name, tail))
        names = row.get("source")
        names = [names] if isinstance(names, str) else names
        if not isinstance(names, (list, tuple)) or not names or not all(isinstance(n, str) for n in names):
            raise MifOpError("pieces[%d].source is the high part's name or a list of them. %s" % (k, tail))
        srcs = [get_object(n, want_mesh=True) for n in dict.fromkeys(names)]
        cage = None
        if row.get("cage") is not None:
            cage = get_object(str(row["cage"]), want_mesh=True)
            if (len(cage.data.vertices), len(cage.data.polygons)) != (len(low.data.vertices), len(low.data.polygons)):
                raise MifOpError("pieces[%d] cage '%s' is not '%s''s topology (%d/%d vertices, %d/%d faces). %s"
                                 % (k, cage.name, low.name, len(cage.data.vertices), len(low.data.vertices),
                                    len(cage.data.polygons), len(low.data.polygons), tail))
        for o in [low] + srcs + ([cage] if cage is not None else []):
            if o.name in seen:
                raise MifOpError("'%s' is in pieces[%d] and pieces[%d] - each object belongs to one piece. %s"
                                 % (o.name, seen[o.name], k, tail))
            seen[o.name] = k
        out.append((low, srcs, cage))
    return out


def _curvature_material():
    mat = bpy.data.materials.new("MifCurvatureBake")
    mat.use_nodes = True
    nt = mat.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    em = nt.nodes.new("ShaderNodeEmission")
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(geo.outputs["Pointiness"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], outn.inputs["Surface"])
    return mat


# ---------------------------------------------------------------------------------------------------------------------
# THE EMISSION ROUTE FOR MAPS CYCLES HAS NO PASS FOR (1.3 gap audit, 2026-10-19). Metallic, alpha, a material ID, a
# position and a thickness map are not Cycles bake types; like CURVATURE they are baked as EMIT through a temporary
# emission material on the objects the rays read - the sources of a high-to-low bake, or the object itself (a self-bake,
# which CURVATURE now allows too: a mesh's own curvature is the common edge-wear mask). METALLIC and ALPHA copy each
# material and route the Principled input's own link (or value) into the emission, so a node chain feeding it is baked
# as the material computes it. MATERIAL_ID gives each material (idBy MATERIAL) or object (idBy OBJECT) one flat color
# from its name. POSITION is world position mapped into the baked objects' bounds (0-1 per axis, the bounds returned).
# THICKNESS is Cycles' Ambient Occlusion node looking INSIDE the mesh (thin is dark), over thicknessDistance.
_EMIT_ROUTE = {"CURVATURE", "METALLIC", "ALPHA", "MATERIAL_ID", "POSITION", "THICKNESS", "_POSITION_RAW",
               "_NORMAL_WORLD", "BASE_COLOR"}
_HEIGHT_BOUNDS = [None]          # set by _bake_height around its internal position bakes (one shared box)


def _id_color(name):
    """A stable, saturated color for a name (HSV from a hash), linear."""
    import colorsys                                             # noqa: PLC0415
    import hashlib                                              # noqa: PLC0415
    h = int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16)
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360.0, 0.75 + ((h >> 9) % 25) / 100.0, 0.7 + ((h >> 17) % 30) / 100.0)
    return [round(r, 6), round(g, 6), round(b, 6)]


def _principled_of(mat):
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return None
    return next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)


def _emit_material(kind, mat, obj, ctx, image):
    """A temporary material that emits the map `kind` for one slot (mat may be None); the bake target image node is
    added and made active when `image` is given (a self-bake writes into the material it renders)."""
    if kind in ("METALLIC", "ALPHA", "BASE_COLOR") and mat is not None and _principled_of(mat) is not None:
        tmp = mat.copy()
        tmp.name = "MifEmitBake_%s" % kind
        nt = tmp.node_tree
        p = _principled_of(tmp)
        sock = p.inputs[{"METALLIC": "Metallic", "ALPHA": "Alpha", "BASE_COLOR": "Base Color"}[kind]]
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None) or next(
            (n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
        if out is None:
            out = nt.nodes.new("ShaderNodeOutputMaterial")
        em = nt.nodes.new("ShaderNodeEmission")
        if sock.is_linked:
            nt.links.new(sock.links[0].from_socket, em.inputs["Color"])
        elif kind == "BASE_COLOR":
            em.inputs["Color"].default_value = tuple(sock.default_value)
        else:
            v = float(sock.default_value)
            em.inputs["Color"].default_value = (v, v, v, 1.0)
        for lk in list(out.inputs["Surface"].links):
            nt.links.remove(lk)
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    else:
        tmp = bpy.data.materials.new("MifEmitBake_%s" % kind)
        tmp.use_nodes = True
        nt = tmp.node_tree
        for nd in list(nt.nodes):
            nt.nodes.remove(nd)
        em = nt.nodes.new("ShaderNodeEmission")
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
        if kind in ("METALLIC", "ALPHA", "BASE_COLOR"):
            v = {"METALLIC": 0.0, "ALPHA": 1.0, "BASE_COLOR": 0.8}[kind]     # no Principled: Blender's own defaults
            em.inputs["Color"].default_value = (v, v, v, 1.0)
        elif kind == "CURVATURE":
            geo = nt.nodes.new("ShaderNodeNewGeometry")
            nt.links.new(geo.outputs["Pointiness"], em.inputs["Color"])
        elif kind == "MATERIAL_ID":
            key = obj.name if ctx["idBy"] == "OBJECT" else (mat.name if mat is not None else "(no material)")
            col = _id_color(key)
            ctx["idColors"][key] = col
            em.inputs["Color"].default_value = col + [1.0]
        elif kind in ("POSITION", "_POSITION_RAW"):
            geo = nt.nodes.new("ShaderNodeNewGeometry")
            lo, size = ctx["bounds"]
            sub = nt.nodes.new("ShaderNodeVectorMath")
            sub.operation = "SUBTRACT"
            sub.inputs[1].default_value = lo
            div = nt.nodes.new("ShaderNodeVectorMath")
            div.operation = "DIVIDE"
            div.inputs[1].default_value = [s if s > 1e-9 else 1.0 for s in size]
            nt.links.new(geo.outputs["Position"], sub.inputs[0])
            nt.links.new(sub.outputs[0], div.inputs[0])
            nt.links.new(div.outputs[0], em.inputs["Color"])
        elif kind == "_NORMAL_WORLD":
            geo = nt.nodes.new("ShaderNodeNewGeometry")
            mul = nt.nodes.new("ShaderNodeVectorMath")
            mul.operation = "MULTIPLY_ADD"
            mul.inputs[1].default_value = (0.5, 0.5, 0.5)
            mul.inputs[2].default_value = (0.5, 0.5, 0.5)
            nt.links.new(geo.outputs["Normal"], mul.inputs[0])
            nt.links.new(mul.outputs[0], em.inputs["Color"])
        elif kind == "THICKNESS":
            ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
            ao.inside = True
            ao.only_local = True
            ao.inputs["Distance"].default_value = ctx["thickness"]
            nt.links.new(ao.outputs["AO"], em.inputs["Color"])
    if image is not None:
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = image
        node.name = "MifBakeTarget"
        for other in nt.nodes:
            other.select = False
        node.select = True
        nt.nodes.active = node
    ctx["made"].append(tmp)
    return tmp


def _world_box(objs):
    from mathutils import Vector                                # noqa: PLC0415
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    return lo, [hi[i] - lo[i] for i in range(3)]


def _bake_height(params):
    """bake_texture type HEIGHT: the distance from the low poly to the high along the low's normal, per texel.

    Three internal emission bakes into float images through the same UVs and margin - the high's world position seen
    from the low (selected-to-active), the low's own world position, the low's world normal - one bounding box shared,
    then h = dot(high - low, normal). heightRange (default the largest |h| found) maps -range..+range to 0..1 (0.5 is on
    the low surface); the range and the measured min / max distance are returned. Written as a 32-bit float image
    (Non-Color); filepath saves it (EXR keeps it exact, PNG at 16 bits)."""
    import numpy as np                                          # noqa: PLC0415
    tail = "NOTHING was baked."
    if not params.get("source") or params.get("pieces") is not None or params.get("multires") is not None:
        raise MifOpError("a HEIGHT bake measures a high poly from the low - pass source (pieces and multires are other "
                         "bakes). %s" % tail)
    if str(params.get("target") or "IMAGE").upper() != "IMAGE":
        raise MifOpError("a HEIGHT bake writes an image. %s" % tail)
    rng = params.get("heightRange")
    if rng is not None and (isinstance(rng, bool) or not isinstance(rng, (int, float)) or not rng > 0):
        raise MifOpError("heightRange is a distance above 0 (scene units), got %r. %s" % (rng, tail))
    low = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
    names = params["source"] if isinstance(params["source"], list) else [params["source"]]
    srcs = [get_object(n, want_mesh=True) for n in names]
    base = {k: v for k, v in params.items() if k in ("object", "name", "width", "height", "uvLayer", "margin",
                                                     "samples", "device", "source", "extrusion", "maxRayDistance",
                                                     "cage", "include")}
    final_path = params.get("filepath")
    fmt = params.get("format")
    image_name = params.get("imageName") or ("MifBake_%s_HEIGHT" % low.name)
    lo, size = _world_box([low] + srcs)
    _HEIGHT_BOUNDS[0] = (lo, size)
    made = []
    try:
        reps = {}
        for key, extra in (("hi", {"type": "_POSITION_RAW"}),
                           ("lo", {"type": "_POSITION_RAW", "source": None, "extrusion": None, "maxRayDistance": None,
                                   "cage": None}),
                           ("n", {"type": "_NORMAL_WORLD", "source": None, "extrusion": None, "maxRayDistance": None,
                                  "cage": None})):
            p = dict(base, float=True, imageName="MifHeight_%s" % key, **extra)
            p = {k: v for k, v in p.items() if v is not None}
            r = op_bake_texture(p)
            reps[key] = r
            made.append(r["image"])
    finally:
        _HEIGHT_BOUNDS[0] = None
    try:
        def px(name):
            im = bpy.data.images[name]
            a = np.empty(im.size[0] * im.size[1] * 4, dtype=np.float32)
            im.pixels.foreach_get(a)
            # a tuple, not im.size: that array reads the image, which is removed below (reading it after crashed 5.0.1)
            return a.reshape(im.size[1], im.size[0], 4), (int(im.size[0]), int(im.size[1]))
        A, size_px = px(made[0])
        L, _s = px(made[1])
        N, _s = px(made[2])
        sz = np.array([s if s > 1e-9 else 1.0 for s in size], dtype=np.float32)
        hi_w = A[..., :3] * sz
        lo_w = L[..., :3] * sz
        nrm = N[..., :3] * 2.0 - 1.0
        nl = np.linalg.norm(nrm, axis=2, keepdims=True)
        nrm = nrm / np.maximum(nl, 1e-6)
        h = np.sum((hi_w - lo_w) * nrm, axis=2)
        baked = nl[..., 0] > 0.5            # texels the bake wrote (a unit normal); the rest are the sentinel
        hv = h[baked]
        if hv.size == 0:
            raise MifOpError("the HEIGHT bake wrote no texel - does the low have UVs the high is in front of? %s" % tail)
        r_used = float(rng) if rng is not None else float(max(np.abs(hv).max(), 1e-6))
        out_img = np.zeros(A.shape, dtype=np.float32)
        val = np.clip(0.5 + h / (2.0 * r_used), 0.0, 1.0)
        out_img[..., 0] = out_img[..., 1] = out_img[..., 2] = np.where(baked, val, 0.5)
        out_img[..., 3] = 1.0
        im = bpy.data.images.new(image_name, size_px[0], size_px[1], alpha=False, float_buffer=True)
        im.colorspace_settings.name = "Non-Color"
        im.pixels.foreach_set(out_img.ravel())
        im.update()
        saved = None
        if final_path:
            from .ops_image import op_save_image                # noqa: PLC0415
            req = {"image": im.name, "filepath": final_path}
            if fmt:
                req["format"] = "OPEN_EXR" if str(fmt).upper() == "EXR" else str(fmt).upper()
            saved = op_save_image(req)
    finally:
        for n in made:
            if bpy.data.images.get(n) is not None:
                bpy.data.images.remove(bpy.data.images[n])
    return {"object": low.name, "bakeType": "HEIGHT", "image": im.name, "width": size_px[0], "height": size_px[1],
            "colorSpace": im.colorspace_settings.name, "sources": [s.name for s in srcs],
            "heightRange": round(r_used, 6), "rangeGiven": rng is not None,
            "distanceMin": round(float(hv.min()), 6), "distanceMax": round(float(hv.max()), 6),
            "texelsBaked": int(baked.sum()), "encoding": "0.5 + distance / (2 x heightRange), clamped: 0.5 is on the "
                                                        "low surface, 1 is heightRange out along its normal",
            "savedTo": saved["file"] if saved else None, "isFloat": True,
            "note": "in memory%s" % ("" if saved else " only - pass filepath to write it")}


_MULTIRES_BAKE_KEYS = {"object", "name", "multires", "width", "height", "imageName", "filepath", "margin", "keepNode"}


def _bake_multires(obj, params):
    """bake_texture multires {type NORMAL | DISPLACEMENT | VECTOR_DISPLACEMENT (5.0+), space OBJECT | TANGENT (5.0+),
    lowResMesh, lowLevel}: Blender's multires bake (object.bake_image) of the Multires top level against lowLevel
    (default 0, the base), into a new float data image through an active image node. 2026-10-06: its switches were never
    reached - scene.render.use_bake_multires / bake_type / use_bake_lores_mesh to 4.4, render.bake.use_multires / type /
    displacement_space / use_lores_mesh from 5.0, both read off the build."""
    tail = "NOTHING was baked."
    extra = sorted(k for k in params if k not in _MULTIRES_BAKE_KEYS and params[k] is not None)
    if extra:
        raise MifOpError("a multires bake takes %s, not %s. %s" % (", ".join(sorted(_MULTIRES_BAKE_KEYS)),
                                                                 ", ".join(extra), tail))
    spec = params["multires"]
    if not isinstance(spec, dict) or any(k not in ("type", "space", "lowResMesh", "lowLevel") for k in spec):
        raise MifOpError("multires is {type, space, lowResMesh, lowLevel}, got %r. %s" % (spec, tail))
    mres = next((m for m in obj.modifiers if m.type == "MULTIRES"), None)
    if mres is None or mres.total_levels < 1:
        raise MifOpError("'%s' has no Multires modifier with a level to bake from. %s" % (obj.name, tail))
    if not obj.data.uv_layers:
        raise MifOpError("'%s' has NO UV layer - uv_unwrap creates one. %s" % (obj.name, tail))
    render = bpy.context.scene.render
    new_api = hasattr(render.bake, "use_multires")
    holder = render.bake if new_api else render
    type_prop = "type" if new_api else "bake_type"
    kinds = {i.identifier for i in holder.bl_rna.properties[type_prop].enum_items}
    kind = str(spec.get("type", "DISPLACEMENT")).upper()
    kind = "NORMALS" if kind == "NORMAL" else kind
    if kind not in kinds:
        raise MifOpError("multires type %r is not one of this build's %s (VECTOR_DISPLACEMENT is 5.0+). %s"
                         % (spec.get("type"), ", ".join(sorted(kinds)), tail))
    space = spec.get("space")
    if space is not None:
        spaces = ({i.identifier for i in holder.bl_rna.properties["displacement_space"].enum_items}
                  if new_api else set())
        if not spaces:
            raise MifOpError("multires space needs Blender 5.0 or later (this is %s). %s" % (bpy.app.version_string,
                                                                                         tail))
        if str(space).upper() not in spaces:
            raise MifOpError("multires space %r is not one of %s. %s" % (space, ", ".join(sorted(spaces)), tail))
        space = str(space).upper()
    lores = take_bool(spec, "lowResMesh", default=False)
    low = take_int(spec, "lowLevel", default=0)
    if not 0 <= low < mres.total_levels:
        raise MifOpError("lowLevel %d must be 0 to %d (below the top level %d). %s" % (low, mres.total_levels - 1,
                                                                                  mres.total_levels, tail))
    width = take_int(params, "width", default=512)
    height = take_int(params, "height", default=512)
    if not (1 <= width <= 8192 and 1 <= height <= 8192):
        raise MifOpError("width/height must be between 1 and 8192; got %dx%d. %s" % (width, height, tail))
    margin = take_int(params, "margin", default=4)
    keep_node = take_bool(params, "keepNode", default=False)
    filepath = take(params, "filepath", default=None, kind=str)
    if filepath:
        check_output_path(filepath, bpy.path.abspath(filepath), "baked")
        filepath = abs_output_path(filepath)        # Image.save() writes nothing to a relative path (abs_output_path)
    offered = {i.identifier for i in bpy.types.ColorManagedInputColorspaceSettings.bl_rna.properties["name"].enum_items}
    data_space = next((w for w in ("Non-Color", "Non-Colour", "Raw", "Generic Data") if w in offered), None)
    image_name = take(params, "imageName", default=None, kind=str) or "MifBake_%s_MULTIRES_%s" % (obj.name, kind)

    image = bpy.data.images.new(image_name, width=width, height=height, float_buffer=True, alpha=False)
    image.generated_color = (1.0, 0.0, 1.0, 1.0)
    if data_space:
        image.colorspace_settings.name = data_space
    created_material = None
    if not any(sl.material for sl in obj.material_slots):
        mat = bpy.data.materials.new("%s_BakeMat" % obj.name)
        mat.use_nodes = True
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        created_material = mat.name
    materials = [sl.material for sl in obj.material_slots if sl.material]
    names = {"multires": ("use_multires", "use_bake_multires"), "type": ("type", "bake_type"),
             "lores": ("use_lores_mesh", "use_bake_lores_mesh"), "margin": ("margin", "bake_margin"),
             "clear": ("use_clear", "use_bake_clear")}
    want = {"multires": True, "type": kind, "lores": lores, "margin": margin, "clear": True}
    prev = {k: getattr(holder, names[k][0 if new_api else 1]) for k in want}
    prev_space = render.bake.displacement_space if new_api else None
    prev_levels = mres.levels
    added = []
    snap = selection_snapshot()
    try:
        for mat in materials:
            mat.use_nodes = True
            node = mat.node_tree.nodes.new("ShaderNodeTexImage")
            node.image = image
            node.name = "MifBakeTarget"
            added.append((mat, node))
            mat.node_tree.nodes.active = node
        for k, v in want.items():
            setattr(holder, names[k][0 if new_api else 1], v)
        if space is not None:
            render.bake.displacement_space = space
        mres.levels = low
        select_only([obj])
        bpy.context.view_layer.objects.active = obj
        before = _pixel_signature(image)
        bpy.ops.object.bake_image()
        after = _pixel_signature(image)
    except Exception as exc:
        for mat, node in added:
            mat.node_tree.nodes.remove(node)
        bpy.data.images.remove(image)
        raise MifOpError("the multires bake failed on '%s': %s: %s. NOTHING was written."
                         % (obj.name, type(exc).__name__, exc))
    finally:
        for k, v in prev.items():
            setattr(holder, names[k][0 if new_api else 1], v)
        if prev_space is not None:
            render.bake.displacement_space = prev_space
        mres.levels = prev_levels
        selection_restore(snap)
    if before == after:
        for mat, node in added:
            mat.node_tree.nodes.remove(node)
        bpy.data.images.remove(image)
        raise MifOpError("the multires bake returned and the image is still its magenta fill - nothing was written. "
                         "The image was discarded.")
    px = list(image.pixels[:])
    reds = px[0::4]
    saved = None
    if filepath:
        image.filepath_raw = filepath
        image.update()
        image.file_format = "OPEN_EXR" if filepath.lower().endswith(".exr") else "PNG"
        image.save()
        saved = filepath
    if not keep_node:
        for mat, node in added:
            mat.node_tree.nodes.remove(node)
    return {"object": obj.name, "bakeType": "MULTIRES_" + kind, "image": image.name, "width": width,
            "height": height, "colorSpace": image.colorspace_settings.name, "fromLevel": mres.total_levels,
            "lowLevel": low, "lowResMesh": lores, "space": space, "route": "render.bake" if new_api else "render",
            "redMin": round(min(reds), 6), "redMax": round(max(reds), 6), "signatureBefore": list(before),
            "signatureAfter": list(after), "changed": True, "savedTo": saved, "createdMaterial": created_material,
            "targetNodeKept": bool(keep_node)}


# Unreal's collision-mesh prefixes (box, sphere, capsule, convex) - names, not geometry, which is how the FBX importer
# picks them out too.
_COLLISION_PREFIXES = ("UCX_", "UBX_", "USP_", "UCP_")
# What a bake ray can strike. Lights, cameras, empties, armatures and the rest cast no occlusion, so a one-object bake
# leaves them alone.
_GEOMETRY_TYPES = {"MESH", "CURVE", "SURFACE", "FONT", "META", "CURVES", "POINTCLOUD", "VOLUME"}


def op_bake_texture(params):
    """Bake AO / normal / diffuse and the rest into an image, and prove the image actually moved.

    THE FAILURE THIS IS ARRANGED AROUND IS A SILENT SUCCESS, measured rather than assumed. With no
    ACTIVE image-texture node in the material, bpy.ops.object.bake returns {'FINISHED'} and writes
    nothing at all - no error, no warning, an untouched image and a call that looks like it worked.
    That is the worst outcome available here, because the caller then saves a blank PNG and wires it
    into a material.

    So success is judged from the IMAGE: is_dirty plus a before/after pixel signature. A bake that
    reports FINISHED over an unchanged buffer is reported as a failure, which is what it is.

    WHAT IS LOUD ALREADY, and therefore not re-implemented: a missing UV layer. The operator raises
    "No active UV layer found in the object" itself. It is still checked up front, because a
    pre-flight refusal names the fix and costs nothing, and because the entry that asked for this
    predicted that case would be silent - it is not, and that is worth recording where the next
    person will look.

    THE SETUP IS RESTORED. Render engine, device, sample count and the whole selection state belong
    to whoever is using this Blender; a bake that leaves the scene on CYCLES with 4096 samples is a
    side effect nobody asked for.
    """
    reject_unknown(params, ("object", "name", "type", "bakeType", "width", "height", "imageName",
                            "filepath", "uvLayer", "margin", "samples", "keepNode", "device", "aoDistance",
                            "colorspace", "float", "format", "source", "extrusion", "maxRayDistance", "cage",
                            "greenChannel", "blobFix", "pieces", "explode", "multires", "include",
                            # 1.3: a color-attribute target, the ID map's key, HEIGHT's range, THICKNESS's reach
                            "target", "colorAttribute", "idBy", "heightRange", "thicknessDistance"),
                   "bake_texture")
    # PER PIECE (2026-10-06): pieces [{object, source, cage}] bakes each low part from its OWN high part into the one
    # image - each pass selects only that pair, so a ray from one part cannot land on a neighbour's high. explode
    # (scene units) also moves each pair apart along +X by its index times explode for the bake and back after, so an
    # AO bake is not darkened by a neighbour either. Parsed first: with pieces, object is the first piece's low.
    tail = "NOTHING was baked."
    pieces = _parse_bake_pieces(params, tail)
    explode = take_float(params, "explode", default=None)
    if explode is not None and not pieces:
        raise MifOpError("explode moves the pieces apart for the bake - pass pieces too. %s" % tail)
    if explode is not None and not explode > 0:
        raise MifOpError("explode must be above 0 (scene units between pieces), got %g. %s" % (explode, tail))
    if params.get("multires") is not None:
        # MULTIRES BAKES ONE OBJECT FROM ITS OWN SCULPT LEVELS - there is no high/low pair to bake per piece, so the two
        # are refused together rather than one of them silently winning.
        if pieces or explode is not None:
            raise MifOpError("multires bakes one object from its own Multires levels; pieces and explode bake each low "
                             "part from its own high part - pass one. %s" % tail)
        mr_obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)
        return _bake_multires(mr_obj, params)
    if pieces:
        for key in ("source", "cage"):
            if params.get(key) is not None:
                raise MifOpError("%s is per piece when pieces is passed - put it in each piece. %s" % (key, tail))
        first = pieces[0][0]
        named = params.get("object", params.get("name"))
        if named is not None and named != first.name:
            raise MifOpError("object %r is not the first piece's object '%s' - with pieces, object may be left out. %s"
                             % (named, first.name, tail))
        obj = first
    else:
        obj = get_object(take(params, "object", "name", required=True, kind=str), want_mesh=True)

    raw_type = take(params, "type", "bakeType") or "AO"
    bake_type = str(raw_type).strip().upper()
    if bake_type not in _BAKE_TYPES or (bake_type.startswith("_") and _HEIGHT_BOUNDS[0] is None):
        raise MifOpError("unknown bake type %r - use one of %s. NOTHING was baked."
                         % (raw_type, ", ".join(sorted(t for t in _BAKE_TYPES if not t.startswith("_")))))
    if bake_type == "HEIGHT":
        return _bake_height(params)
    for key, owner in (("heightRange", "HEIGHT"), ("thicknessDistance", "THICKNESS"), ("idBy", "MATERIAL_ID")):
        if params.get(key) is not None and bake_type != owner:
            raise MifOpError("%s is a %s bake's; this is %s. %s" % (key, owner, bake_type, tail))
    target = str(take(params, "target", default="IMAGE", kind=str)).upper()
    if target not in ("IMAGE", "COLOR_ATTRIBUTE"):
        raise MifOpError("target is IMAGE (default) or COLOR_ATTRIBUTE, got %r. %s" % (params.get("target"), tail))
    if target == "COLOR_ATTRIBUTE":
        clash = [k for k in ("filepath", "format", "float", "blobFix", "keepNode", "imageName", "pieces", "colorspace",
                             "width", "height") if params.get(k) is not None]
        if clash:
            raise MifOpError("%s shape an image, and target COLOR_ATTRIBUTE bakes into the mesh's color attribute. %s"
                             % (clash, tail))
    elif params.get("colorAttribute") is not None:
        raise MifOpError("colorAttribute names the attribute a target COLOR_ATTRIBUTE bake writes. %s" % tail)
    id_by = str(take(params, "idBy", default="MATERIAL", kind=str)).upper()
    if id_by not in ("MATERIAL", "OBJECT"):
        raise MifOpError("idBy is MATERIAL or OBJECT, got %r. %s" % (params.get("idBy"), tail))

    mesh = obj.data
    if not mesh.polygons:
        raise MifOpError("'%s' has no faces, so there is nothing to bake. NOTHING was baked."
                         % obj.name)

    # HIGH TO LOW (2026-10-06): source names the high-poly object(s) baked onto this one's UVs - Blender's
    # selected-to-active. extrusion pushes the low poly out along its normals before rays are cast back (default 0.01
    # scene units), maxRayDistance limits them, cage names an object with the low poly's vertex count to cast from
    # instead. greenChannel DIRECTX writes a normal map's green flipped (Unreal's convention), OPENGL (default) not.
    # blobFix repairs the patches where pinched cloth's rays hit the facing surface (fix_normal_blobs above).
    raw_src = params.get("source")
    sources = []
    if raw_src is not None:
        names = [raw_src] if isinstance(raw_src, str) else raw_src
        if not isinstance(names, (list, tuple)) or not names or not all(isinstance(n, str) for n in names):
            raise MifOpError("source is an object name or a list of them. %s" % tail)
        for n in dict.fromkeys(names):
            src = get_object(n, want_mesh=True)
            if src is obj:
                raise MifOpError("'%s' is the object baked onto - a source is the HIGH poly baked from. %s"
                                 % (obj.name, tail))
            sources.append(src)
    extrusion = take_float(params, "extrusion", default=None)
    max_ray = take_float(params, "maxRayDistance", default=None)
    cage_name = take(params, "cage", default=None, kind=str)
    if pieces:
        sources = [src for _low, srcs, _cage in pieces for src in srcs]
    for key, val in (("extrusion", extrusion), ("maxRayDistance", max_ray), ("cage", cage_name)):
        if val is not None and not sources:
            raise MifOpError("%s shapes a high-to-low bake's rays - pass source too. %s" % (key, tail))
    if extrusion is not None and extrusion < 0:
        raise MifOpError("extrusion must be 0 or above, got %g. %s" % (extrusion, tail))
    if max_ray is not None and max_ray < 0:
        raise MifOpError("maxRayDistance must be 0 (no limit) or above, got %g. %s" % (max_ray, tail))
    cage = None
    if cage_name is not None:
        cage = get_object(cage_name, want_mesh=True)
        if len(cage.data.vertices) != len(mesh.vertices) or len(cage.data.polygons) != len(mesh.polygons):
            raise MifOpError("cage '%s' has %d vertices and %d faces, '%s' %d and %d - a cage is the low poly's own "
                             "topology, pushed out. %s" % (cage.name, len(cage.data.vertices), len(cage.data.polygons),
                                                          obj.name, len(mesh.vertices), len(mesh.polygons), tail))
        if extrusion is not None:
            raise MifOpError("extrusion pushes the low poly out when there is no cage; with cage the cage's own shape "
                             "is where rays start - pass one. %s" % tail)
    if pieces and extrusion is not None and any(c is not None for _l, _s, c in pieces):
        raise MifOpError("extrusion pushes the low poly out when there is no cage; with cage the cage's own shape "
                         "is where rays start - pass one. %s" % tail)
    plan = pieces or [(obj, sources, cage)]
    if bake_type in _EMIT_ROUTE and pieces:
        raise MifOpError("a %s bake goes through a temporary emission per object, and pieces bakes each pair into one "
                         "image - bake it once per low, or with source. %s" % (bake_type, tail))
    thickness = take_float(params, "thicknessDistance", default=None)
    if thickness is not None and not thickness > 0:
        raise MifOpError("thicknessDistance must be above 0 (scene units), got %g. %s" % (thickness, tail))
    green = take(params, "greenChannel", default=None, kind=str)
    if green is not None:
        green = str(green).upper()
        if bake_type != "NORMAL":
            raise MifOpError("greenChannel is a normal map's convention, and this is a %s bake. %s" % (bake_type, tail))
        if green not in ("OPENGL", "DIRECTX"):
            raise MifOpError("greenChannel must be OPENGL (Blender, +Y) or DIRECTX (Unreal, -Y), got %r. %s"
                             % (green, tail))
    # WHAT ELSE THE RAYS MAY HIT (FOUND 2026-10-12): a plain one-object AO bake picked up every visible object in the scene,
    # the UCX_ hulls placed around the prop among them, and the hulls darkened the baked prop. A one-object bake now
    # hides every other geometry object from render for the bake (hide_render, put back after); lights, cameras and empties
    # stay, so a SHADOW or COMBINED bake keeps its lighting. Collision meshes (UCX_, UBX_, USP_, UCP_) are hidden in EVERY
    # bake, a high-to-low or per-piece one included, unless they are the object or a source. include [names] keeps the
    # named objects visible - the case that wants its neighbours' occlusion.
    raw_inc = params.get("include")
    include = []
    if raw_inc is not None:
        if not isinstance(raw_inc, (list, tuple)) or not all(isinstance(n, str) for n in raw_inc):
            raise MifOpError("include is a list of object names kept visible for the bake. %s" % tail)
        include = [get_object(n) for n in dict.fromkeys(raw_inc)]
    blob_rule = _parse_blob_fix(params.get("blobFix"), tail)
    if blob_rule is not None and bake_type != "NORMAL":
        raise MifOpError("blobFix repairs a normal map, and this is a %s bake. %s" % (bake_type, tail))

    # THE COLOUR SPACE IS RESOLVED HERE, before anything is created. This used to test availability
    # by assigning to a live image, so it could only run once the image, a generated material and
    # the use_nodes flips existed - and then refused, leaving all of them behind. The enum lives on
    # ColorManagedInputColorspaceSettings itself, so the question is answerable with nothing built.
    #
    # Named by the OCIO config rather than remembered: 'Non-Color' on stock Blender, but a studio
    # config may spell it differently, and silently leaving sRGB is the bug this is here to stop.
    # DIFFUSE, COMBINED, EMIT and GLOSSY are radiometric colour and stay sRGB. The split is by
    # what the channel MEANS, which is why it is a table rather than a rule about scalars.
    _NON_COLOUR_BAKES = {"NORMAL", "ROUGHNESS", "AO", "SHADOW", "CURVATURE", "METALLIC", "ALPHA", "POSITION",
                         "THICKNESS", "_POSITION_RAW", "_NORMAL_WORLD"}
    non_colour_space = None
    if bake_type in _NON_COLOUR_BAKES:
        offered = {i.identifier for i in bpy.types.ColorManagedInputColorspaceSettings
                   .bl_rna.properties["name"].enum_items}
        non_colour_space = next((w for w in ("Non-Color", "Non-Colour", "Raw", "Generic Data")
                                 if w in offered), None)
        if non_colour_space is None:
            raise MifOpError(
                "a %s bake must not be written with an sRGB transfer, and this Blender's colour "
                "config offers none of Non-Color/Non-Colour/Raw/Generic Data to say so. Writing it "
                "anyway would produce a map that is wrong in every engine while looking correct. "
                "NOTHING was baked." % bake_type)
    # AN OVERRIDE (2026-10-04): a roughness, normal or displacement map baked THROUGH AN EMISSION shader is an EMIT bake
    # and was written sRGB. colorspace DATA writes it as data; any of this config's names is taken as given.
    want_cs = take(params, "colorspace", default=None, kind=str)
    if want_cs is not None:
        offered = {i.identifier for i in bpy.types.ColorManagedInputColorspaceSettings
                   .bl_rna.properties["name"].enum_items}
        if str(want_cs).upper() == "DATA":
            non_colour_space = next((w for w in ("Non-Color", "Non-Colour", "Raw", "Generic Data") if w in offered),
                                    None)
            if non_colour_space is None:
                raise MifOpError("this colour config offers no data space (Non-Color, Raw, Generic Data). NOTHING was "
                                 "baked.")
        elif want_cs in offered:
            non_colour_space = want_cs
        else:
            raise MifOpError("colorspace %r is not DATA or one of this config's spaces: %s. NOTHING was baked."
                             % (want_cs, ", ".join(sorted(offered))))

    # PRE-FLIGHT, even though the operator raises for this itself. Its message names the object;
    # this one names the fix.
    uv_name = take(params, "uvLayer")
    for low in [p_[0] for p_ in plan]:
        if not low.data.uv_layers:
            raise MifOpError(
                "'%s' has NO UV layer, and a bake writes through UVs - there is nowhere for the result "
                "to land. uv_unwrap creates one. (The operator raises for this too rather than failing "
                "silently, so this refusal is only about naming the fix.) NOTHING was baked."
                % low.name)
    # THE PATH IS CHECKED BEFORE THE FIRST WRITE, not beside the bake it guards. Placing it next to
    # the save looked right and put it BELOW the active-UV-layer switch, so an unusable filepath
    # refused with the layer already changed. audit_mutate_then_deny found it once it learned to see
    # refusals coming from ops_common - check_output_path is imported, and until then a shared
    # helper's refusal was invisible to it.
    filepath = take(params, "filepath")
    if filepath:
        check_output_path(filepath, bpy.path.abspath(str(filepath)), "baked")
        filepath = abs_output_path(filepath)        # Image.save() writes nothing to a relative path (abs_output_path)

    # A FLOAT IMAGE AND THE FILE FORMAT, ASKED FOR (2026-10-05). The bake image was always 8-bit, so a value past 0-1 -
    # a bump height, an emission over 1 - was clipped before it reached any file, and an .exr path only chose the
    # format at save time, of an 8-bit buffer. float makes a 32-bit float image with no alpha; format names the file
    # format (default: from the extension). An EXR keeps floats, so it makes the image float unless float is false,
    # which is refused. Checked here, before anything is made.
    want_float = take_bool(params, "float", default=None) if params.get("float") is not None else None
    fmt = take(params, "format", default=None, kind=str)
    ext_formats = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".tga": "TARGA", ".exr": "OPEN_EXR"}
    if fmt is not None:
        fmt = str(fmt).strip().upper()
        fmt = "OPEN_EXR" if fmt == "EXR" else fmt
        if fmt not in set(ext_formats.values()):
            raise MifOpError("format %r is not one of %s (EXR is OPEN_EXR). NOTHING was baked."
                             % (fmt, ", ".join(sorted(set(ext_formats.values())))))
        if not filepath:
            raise MifOpError("format is the format of the file the bake is saved to - pass filepath too. NOTHING was "
                             "baked.")
        ext = os.path.splitext(str(filepath))[1].lower()
        if ext in ext_formats and ext_formats[ext] != fmt:
            raise MifOpError("format %s and the filepath's extension %s disagree, and the file would carry the wrong "
                             "one. NOTHING was baked." % (fmt, ext))
    elif filepath:
        fmt = ext_formats.get(os.path.splitext(str(filepath))[1].lower())
    if fmt == "OPEN_EXR":
        if want_float is False:
            raise MifOpError("an EXR stores floats, and float false bakes into an 8-bit image whose values were "
                             "already clipped to 0-1 - pass float true, or save a PNG. NOTHING was baked.")
        want_float = True
        # AN EXR HOLDS THE BAKED VALUES ONLY FROM A DATA IMAGE. Measured 2026-10-05: a float image in Linear Rec.709
        # saved as EXR came back 3.0 -> 1.612 on 4.2.23 and 4.4.0 (the sRGB curve applied on the way out) and 3.0 on
        # 5.0.1 and 5.2.2; as Non-Color it is 3.0 on all four. A float buffer is scene-linear either way, so the bake
        # writes the same numbers - and the space cannot be switched after the bake, because a generated image's
        # buffer is made again when it changes (its pixels went 3.0 -> 1.0, the fill). So it is set here, before.
        if non_colour_space is None and want_cs is None:
            offered = {i.identifier for i in bpy.types.ColorManagedInputColorspaceSettings
                       .bl_rna.properties["name"].enum_items}
            non_colour_space = next((w for w in ("Non-Color", "Non-Colour", "Raw", "Generic Data") if w in offered),
                                    None)
    want_float = bool(want_float)

    # THE SAME RULE AS filepath ABOVE, and these four were left behind when it moved. The comment
    # up there records the lesson - "placing it next to the save looked right and put it BELOW the
    # active-UV-layer switch, so an unusable filepath refused with the layer already changed" - and
    # width, height, samples and margin were still validated where they are READ, which is below the
    # switch. Reported by the second seat as defect P:
    #
    #     two UV layers, the FIRST active
    #     bake_texture{..., uvLayer: <the second>, samples: 99999}
    #       -> ok:false "samples must be between 1 and 4096"    and the SECOND layer was now active
    #
    # The active layer decides which channel the NEXT bake or export writes through, and a caller
    # whose call was refused has no reason to re-check it. Same reads, same defaults, same order -
    # the only difference is which state a refusal leaves behind.
    width = take_int(params, "width", default=512)
    height = take_int(params, "height", default=512)
    if width < 1 or height < 1 or width > 8192 or height > 8192:
        raise MifOpError("width/height must be between 1 and 8192; got %dx%d" % (width, height))
    samples = take_int(params, "samples", default=16)
    if samples < 1 or samples > 4096:
        raise MifOpError("samples must be between 1 and 4096; got %d" % samples)
    margin = take_int(params, "margin", default=4)
    # take_bool, AND ABOVE THE LAYER SWITCH for the reason the paragraph above gives. It was bool(take(...)) below
    # it: bool("false") is True, so {keepNode: "false"} left the bake's image node in the material and answered
    # targetNodeKept:true (measured on 5.0.1, 2026-09-21) - and a refusing parse there would have left the
    # active UV layer switched.
    try:
        keep_node = take_bool(params, "keepNode", default=False)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was baked." % exc)
    # HOW FAR AN AO BAKE LOOKS FOR OCCLUDERS. Cycles' AO bake reads the WORLD's light_settings.distance (measured on
    # 5.2.1: two planes 1 m apart baked white at 0.1 and mean 0.69 at the default 10), so an enclosed hood interior baked
    # near black and nothing typed could shorten it (2026-10-04). Set for the bake and restored after.
    try:
        ao_distance = take_float(params, "aoDistance", default=None)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was baked." % exc)
    if ao_distance is not None and bake_type != "AO":
        raise MifOpError("aoDistance is how far an AO bake looks for occluders, and this is a %s bake. NOTHING was "
                         "baked." % bake_type)
    if ao_distance is not None and not ao_distance > 0.0:
        raise MifOpError("aoDistance must be above 0 (scene units), got %g. NOTHING was baked." % ao_distance)

    if uv_name:
        for low in [p_[0] for p_ in plan]:
            if low.data.uv_layers.get(uv_name) is None:
                raise MifOpError("'%s' has no UV layer named %r. It has: %s. NOTHING was baked."
                                 % (low.name, uv_name, ", ".join(u.name for u in low.data.uv_layers)))
        for low in [p_[0] for p_ in plan]:
            low.data.uv_layers.active = low.data.uv_layers[uv_name]

    image_name = take(params, "imageName") or ("MifBake_%s_%s" % (obj.name, bake_type))

    # THE LAYER IT BAKES THROUGH IS MEASURED, and the answer is a WARNING, not a refusal. Where two islands truly
    # overlap, the later face's texels overwrite the earlier's: on 2026-09-13 a DIFFUSE bake through a joined
    # toadstool's layer wrote its stem over its cap, and nothing said so. But MIRRORED or STACKED UVs share texels on
    # purpose - a symmetric model's two halves baking into one - so refusing would break a real workflow. uv_info
    # already answers the question, by island bounding box (conservative: it also counts boxes that merely touch).
    # A pure read, done before anything is created. Deferred import, the way ops_nodes reaches back into this module:
    # the module-level direction stays one way.
    from .ops_mesh import op_uv_info   # noqa: PLC0415
    baked_layer = mesh.uv_layers.active.name if mesh.uv_layers.active else mesh.uv_layers[0].name
    uv_pairs = 0
    for low in [p_[0] for p_ in plan]:
        lay = low.data.uv_layers.active.name if low.data.uv_layers.active else low.data.uv_layers[0].name
        uv_row = (op_uv_info({"object": low.name, "layer": lay}).get("layers") or [{}])[0]
        uv_pairs += int(uv_row.get("overlappingIslandPairs") or 0)

    # A material is where the bake TARGET lives, so one is required. Created rather than refused,
    # and reported - an AO bake on an unmaterialed mesh is a perfectly reasonable request.
    created_materials = []
    for low in [p_[0] for p_ in plan]:
        if not low.data.materials or all(sl.material is None for sl in low.material_slots):
            mat = bpy.data.materials.new("%s_BakeMat" % low.name)
            mat.use_nodes = True
            if low.data.materials:
                low.data.materials[0] = mat
            else:
                low.data.materials.append(mat)
            created_materials.append(mat.name)
    created_material = created_materials[0] if created_materials else None

    materials = list(dict.fromkeys(sl.material for p_ in plan for sl in p_[0].material_slots if sl.material))
    for mat in materials:
        if not mat.use_nodes:
            mat.use_nodes = True

    # A SENTINEL FILL, because "the image did not change" is NOT the same question as "the bake
    # wrote nothing". A fresh image is black, and a legitimately black bake result - AO on a face
    # with nothing to occlude it, an EMIT pass on an unlit material - leaves the buffer identical
    # to an untouched one. is_dirty does not separate them either: it goes true merely from the
    # bake touching the image. Filling with magenta first means ANY value the bake writes differs
    # from the start state, so "unchanged" means untouched and nothing else.
    image = bpy.data.images.new(image_name, width=width, height=height,
                                **({"float_buffer": True, "alpha": False} if want_float else {}))
    image.generated_color = (1.0, 0.0, 1.0, 1.0)

    # NON-COLOUR BAKES MUST NOT CARRY AN sRGB TRANSFER, and images.new() defaults to sRGB. Nothing
    # here set colorspace_settings before 2026-09-03, so every NORMAL, ROUGHNESS, AO and SHADOW map
    # this op has ever written went to disk with a gamma curve applied to data that is not colour -
    # a surface direction, a scalar mask. Unreal, Unity, Godot and glTF all read those channels
    # linearly, so the result is wrong everywhere in the same direction: normals too shallow,
    # roughness too bright.
    #
    # It is invisible in review because the map still LOOKS like a normal map, and invisible in this
    # op's own postcondition because the magenta-sentinel check asks whether the buffer CHANGED, not
    # whether what landed in it is right. A pixel signature cannot catch a transfer curve.
    #
    # DIFFUSE, COMBINED, EMIT and GLOSSY are radiometric colour and stay sRGB. The split is by what
    # the channel MEANS, which is why it is a table rather than a rule about scalars.
    colour_space = None
    if non_colour_space is not None:
        # Resolved above, against the config's own list. Nothing here can refuse.
        image.colorspace_settings.name = non_colour_space
        colour_space = non_colour_space
    else:
        colour_space = image.colorspace_settings.name
    if image.name != image_name:
        # Blender uniquifies silently, and a caller who then looks up image_name finds the OLD one.
        note_renamed = image.name
    else:
        note_renamed = None

    scene = bpy.context.scene
    prev = {
        "engine": scene.render.engine,
        "samples": getattr(getattr(scene, "cycles", None), "samples", None),
        "device": getattr(getattr(scene, "cycles", None), "device", None),
        "active": bpy.context.view_layer.objects.active,
        "world": scene.world,
        "aoDistance": scene.world.light_settings.distance if scene.world is not None else None,
    }
    snap = selection_snapshot()
    added_nodes = []
    temp_world = None
    ao_used = None
    swapped, curv_mat = [], None
    emit_ctx = {"made": [], "idColors": {}}
    color_attr_restore, color_attrs = [], []
    moved, piece_rows = [], []
    baked = {o for low, srcs, pcage in plan for o in [low] + srcs + ([pcage] if pcage is not None else [])}
    kept = baked | set(include)
    one_object = not pieces and not sources
    hidden = []
    try:
        for o in scene.objects:
            if o in kept or o.hide_render:
                continue
            collision = o.name.upper().startswith(_COLLISION_PREFIXES)
            if collision or (one_object and o.type in _GEOMETRY_TYPES):
                o.hide_render = True
                hidden.append(o)
        if bake_type in _EMIT_ROUTE:
            # THE EMITTERS: the sources of a high-to-low bake, else the object itself - whose temporary materials then
            # carry the bake target, since the slot it renders is the one the bake writes through
            self_bake = not sources
            emitters = sources if sources else [low_ for low_, _s, _c in plan]
            box = _HEIGHT_BOUNDS[0] or _world_box(emitters)
            emit_ctx.update({"bounds": box, "idBy": id_by,
                             "thickness": thickness if thickness is not None else max(max(box[1]), 1e-3) * 0.25})
            for src in emitters:
                if not src.material_slots:
                    tmp = _emit_material(bake_type, None, src, emit_ctx, image if self_bake else None)
                    src.data.materials.append(tmp)
                    swapped.append((src, None, None))
                    continue
                for slot in src.material_slots:
                    tmp = _emit_material(bake_type, slot.material, src, emit_ctx, image if self_bake else None)
                    swapped.append((src, slot, (slot.material, slot.link)))
                    slot.link = "OBJECT"
                    slot.material = tmp
        if target == "COLOR_ATTRIBUTE":
            # A COLOR-ATTRIBUTE TARGET writes the active color attribute of each low (made when missing, CORNER byte
            # color), filled with the magenta sentinel first so an untouched one is told from a black result
            ca_name = take(params, "colorAttribute", default="Bake_%s" % bake_type, kind=str)
            for low_, _s, _c in plan:
                cas = low_.data.color_attributes
                color_attr_restore.append((low_, cas.active_color))
                ca = cas.get(ca_name) or cas.new(ca_name, "BYTE_COLOR", "CORNER")
                for d_ in ca.data:
                    d_.color = (1.0, 0.0, 1.0, 1.0)
                cas.active_color = ca
                if hasattr(cas, "render_color_index"):
                    cas.render_color_index = list(cas).index(ca)
                color_attrs.append((low_, ca_name))
        if ao_distance is not None:
            if scene.world is None:
                # No world, nothing to carry the distance: a temporary one, removed after.
                temp_world = bpy.data.worlds.new("MifBakeWorld")
                scene.world = temp_world
            scene.world.light_settings.distance = ao_distance
        if bake_type == "AO" and scene.world is not None:
            ao_used = round(float(scene.world.light_settings.distance), 6)
        scene.render.engine = "CYCLES"
        if hasattr(scene, "cycles"):
            scene.cycles.samples = samples
            # CPU by default: a headless box may have no configured GPU device at all, and a bake
            # that silently falls back is a bake nobody can reason about.
            scene.cycles.device = str(take(params, "device") or "CPU").upper()

        for mat in materials:
            node = mat.node_tree.nodes.new("ShaderNodeTexImage")
            node.image = image
            node.name = "MifBakeTarget"
            added_nodes.append((mat, node))
            for other in mat.node_tree.nodes:
                other.select = False
            node.select = True
            mat.node_tree.nodes.active = node

        if explode is not None:
            # Each object's own (local) matrix is kept and put back; one whose parent also moves follows it.
            group = {o: i for i, (low, srcs, pcage) in enumerate(plan)
                     for o in [low] + srcs + ([pcage] if pcage is not None else [])}
            for o, i in group.items():
                moved.append((o, o.matrix_basis.copy()))
                anc = o.parent
                while anc is not None and anc not in group:
                    anc = anc.parent
                if anc is None:
                    mw = o.matrix_world.copy()
                    mw.translation.x += i * explode
                    o.matrix_world = mw
            bpy.context.view_layer.update()

        before = _pixel_signature(image)
        for i, (low, srcs, pcage) in enumerate(plan):
            # SELECTION IS PART OF THE CONTRACT: bake reads the selected objects and writes into the
            # ACTIVE one. Leaving a stray selection from earlier work is one of the ways this produces
            # nothing while reporting FINISHED.
            select_only(srcs + [low])
            bpy.context.view_layer.objects.active = low
            # bpy.ops.object.bake fills every unpassed property from scene.render.bake, which the .blend
            # saves. A file last used for a high-to-low bake keeps use_selected_to_active on, and this
            # one-object bake then refused with "No valid selected objects" (2026-10-04, outfit.blend).
            # Only the first piece clears the image; the rest write their own UV islands into it.
            kwargs = {"type": _BAKE_TYPES[bake_type], "margin": margin, "use_clear": i == 0,
                      "use_selected_to_active": bool(srcs), "use_cage": pcage is not None}
            if srcs:
                kwargs["cage_extrusion"] = extrusion if extrusion is not None else 0.01
                if max_ray is not None:
                    kwargs["max_ray_distance"] = max_ray
                if pcage is not None:
                    kwargs["cage_object"] = pcage.name
            if bake_type == "NORMAL":
                kwargs["normal_space"] = "TANGENT"
                kwargs["normal_g"] = "NEG_Y" if green == "DIRECTX" else "POS_Y"
            if bake_type == "DIFFUSE":
                kwargs["pass_filter"] = {"COLOR"}
            if target == "COLOR_ATTRIBUTE":
                kwargs["target"] = "VERTEX_COLORS"
            prior = _pixel_signature(image) if pieces else None
            bpy.ops.object.bake(**kwargs)
            if pieces:
                sig = _pixel_signature(image)
                piece_rows.append({"object": low.name, "sources": [o.name for o in srcs],
                                   "cage": pcage.name if pcage is not None else None, "changed": sig != prior,
                                   "offsetX": i * explode if explode is not None else None})
        after = _pixel_signature(image)
    except Exception as exc:
        for mat, node in added_nodes:
            try:
                mat.node_tree.nodes.remove(node)
            except Exception:  # noqa: BLE001
                pass
        try:
            bpy.data.images.remove(image)
        except Exception:  # noqa: BLE001
            pass
        raise MifOpError("bake failed on '%s' (%s): %s: %s. NOTHING was written."
                         % (obj.name, bake_type, type(exc).__name__, exc))
    finally:
        for o in hidden:
            o.hide_render = False
        for o, mb in moved:
            o.matrix_basis = mb
        if moved:
            bpy.context.view_layer.update()
        for src, slot, old in reversed(swapped):
            if slot is None:
                src.data.materials.pop(index=len(src.data.materials) - 1)
            else:
                slot.material = None
                slot.link = old[1]
                if old[1] == "OBJECT":
                    slot.material = old[0]
        if curv_mat is not None:
            bpy.data.materials.remove(curv_mat)
        for tmp in emit_ctx["made"]:
            try:
                bpy.data.materials.remove(tmp)
            except ReferenceError:
                pass
        for low_, before_ in color_attr_restore:
            if before_ is not None:
                low_.data.color_attributes.active_color = before_
        if temp_world is not None:
            scene.world = prev["world"]
            bpy.data.worlds.remove(temp_world)
        elif prev["aoDistance"] is not None and scene.world is not None:
            scene.world.light_settings.distance = prev["aoDistance"]
        scene.render.engine = prev["engine"]
        if hasattr(scene, "cycles"):
            if prev["samples"] is not None:
                scene.cycles.samples = prev["samples"]
            if prev["device"] is not None:
                scene.cycles.device = prev["device"]
        selection_restore(snap)
        try:
            bpy.context.view_layer.objects.active = prev["active"]
        except Exception:  # noqa: BLE001
            pass

    if target == "COLOR_ATTRIBUTE":
        # THE POSTCONDITION is the attribute's: any corner that left the magenta sentinel
        for mat, node in added_nodes:
            try:
                mat.node_tree.nodes.remove(node)
            except Exception:  # noqa: BLE001
                pass
        bpy.data.images.remove(image)
        rows_ca = []
        for low_, ca_name in color_attrs:
            ca = low_.data.color_attributes[ca_name]
            vals = [tuple(round(c, 3) for c in d_.color) for d_ in ca.data]
            changed = sum(1 for v in vals if v != (1.0, 0.0, 1.0, 1.0))
            mean = [round(sum(v[i] for v in vals) / max(len(vals), 1), 6) for i in range(3)]
            rows_ca.append({"object": low_.name, "colorAttribute": ca_name, "domain": ca.domain, "corners": len(vals),
                            "cornersWritten": changed, "mean": mean})
        if not any(r["cornersWritten"] for r in rows_ca):
            raise MifOpError("the bake reported FINISHED and no corner of the color attribute left its magenta sentinel - "
                             "nothing was written.")
        return {"object": obj.name, "bakeType": bake_type, "target": "COLOR_ATTRIBUTE", "attributes": rows_ca,
                "samples": samples, "sources": [o.name for o in sources] or None,
                "idColors": emit_ctx["idColors"] or None}

    # THE POSTCONDITION. bake returns {'FINISHED'} with no active image-texture node and writes
    # nothing whatsoever - measured, not feared. So the image is asked whether it moved.
    # CAPTURED BEFORE THE IMAGE IS REMOVED. The first version read image.is_dirty in the message
    # AFTER bpy.data.images.remove(image), which is a use-after-free on freed RNA - the same
    # mistake boolean_op made against a modifier earlier the same night, and Blender says so:
    # "ReferenceError: StructRNA of type Image has been removed".
    was_dirty = image.is_dirty
    if not was_dirty or before == after:
        for mat, node in added_nodes:
            try:
                mat.node_tree.nodes.remove(node)
            except Exception:  # noqa: BLE001
                pass
        try:
            bpy.data.images.remove(image)
        except Exception:  # noqa: BLE001
            pass
        raise MifOpError(
            "the bake reported FINISHED and the image is still the magenta sentinel it was "
            "filled with (signature %s, dirty=%s) - so nothing was written. That is "
            "bpy.ops.object.bake's silent-success case: with no active image-texture node to bake "
            "into it returns FINISHED, touches nothing and reports no error at all. The sentinel "
            "exists so this is distinguishable from a legitimately BLACK result, which a fresh "
            "image would have matched exactly. The image was discarded rather than handed back "
            "blank." % (before, was_dirty))

    idle = [row["object"] for row in piece_rows if not row["changed"]]
    if idle:
        for mat, node in added_nodes:
            try:
                mat.node_tree.nodes.remove(node)
            except Exception:  # noqa: BLE001
                pass
        bpy.data.images.remove(image)
        raise MifOpError("the piece(s) %s left the image as it was - their bake wrote nothing (no UV island of their "
                         "own, or no ray reached their source). The image was discarded rather than handed back with "
                         "holes." % ", ".join(idle))

    blob_stats = fix_normal_blobs(image, *blob_rule) if blob_rule is not None else None

    saved = None
    saved_sig = None
    if filepath:
        image.filepath_raw = str(filepath)
        # THE FORMAT ONLY STICKS ON A BUFFER THAT EXISTS (save_image measured it on 3.6.23, 4.4.0 and 5.2.1): after a
        # generated image's colour space changes, file_format reads back PNG until the buffer is made again, and the
        # file came out a PNG named .exr. update() makes it; the format is read back before the save.
        image.update()
        image.file_format = fmt or "PNG"
        if image.file_format != (fmt or "PNG"):
            raise MifOpError("the bake succeeded and the image kept file_format %s after %s was written, so it was not "
                             "saved - save_image writes it." % (image.file_format, fmt or "PNG"))
        image.save()
        saved = str(filepath)
        if not os.path.isfile(saved):
            raise MifOpError("image.save() reported no error but '%s' is not on disk. The bake "
                             "itself succeeded." % saved)
        # WHAT LANDED ON DISK, read back: the file loaded into a scratch datablock, its signature taken, the
        # datablock removed. A float bake saved as an 8-bit format would show its clip here.
        probe = bpy.data.images.load(os.path.abspath(bpy.path.abspath(saved)), check_existing=False)
        try:
            saved_sig = list(_pixel_signature(probe))
        finally:
            bpy.data.images.remove(probe)

    if not keep_node:
        for mat, node in added_nodes:
            try:
                mat.node_tree.nodes.remove(node)
            except Exception:  # noqa: BLE001
                pass

    result = {
        "object": obj.name,
        "bakeType": bake_type,
        "image": image.name,
        # WHICH TRANSFER THE FILE CARRIES. A normal or roughness map written as sRGB is wrong in
        # every engine and looks fine to the eye, so the caller is told rather than left to assume.
        # Read back off the image, not from the table, so it reports what was actually set.
        "colorSpace": image.colorspace_settings.name,
        "colorSpaceNote": ("NORMAL/ROUGHNESS/AO/SHADOW are data, not colour, and are written with "
                           "no sRGB transfer. DIFFUSE/COMBINED/EMIT/GLOSSY are colour and keep it."),
        "width": width,
        "height": height,
        "samples": samples,
        "margin": margin,
        # The occluder distance the AO bake used (the world's, or aoDistance); null for other bakes.
        "aoDistance": ao_used,
        "uvLayer": mesh.uv_layers.active.name if mesh.uv_layers.active else None,
        "uvOverlappingIslandPairs": uv_pairs,
        "materials": [m.name for m in materials],
        "createdMaterial": created_material,
        "signatureBefore": list(before),
        "signatureAfter": list(after),
        "changed": True,
        "savedTo": saved,
        "savedFormat": fmt if saved else None,
        "savedSignature": saved_sig,
        "isFloat": bool(image.is_float),
        "targetNodeKept": keep_node,
        "sources": [o.name for o in sources] or None,
        "extrusion": (extrusion if extrusion is not None else (0.01 if cage is None else None)) if sources else None,
        "maxRayDistance": max_ray, "cage": cage.name if cage is not None else None,
        "greenChannel": (green or "OPENGL") if bake_type == "NORMAL" else None,
        "blobFix": dict(zip(("maxTiltDeg", "minBlobPx"), blob_rule), **blob_stats) if blob_rule else None,
        "pieces": piece_rows or None, "explode": explode,
        # What the rays could not hit: every other geometry object for a one-object bake, collision meshes always.
        "excluded": len(hidden), "excludedNames": [o.name for o in hidden][:40],
        "included": [o.name for o in include] or None,
    }
    if bake_type in _EMIT_ROUTE:
        result["route"] = "temporary emission on %s" % ("the sources" if sources else "the object itself (self-bake)")
        if bake_type == "MATERIAL_ID":
            result["idBy"] = id_by
            result["idColors"] = emit_ctx["idColors"]
        if bake_type == "POSITION":
            box = emit_ctx["bounds"]
            result["positionBounds"] = {"min": [round(v, 6) for v in box[0]],
                                        "size": [round(v, 6) for v in box[1]],
                                        "encoding": "(world position - min) / size, per axis"}
        if bake_type == "THICKNESS":
            result["thicknessDistance"] = round(emit_ctx["thickness"], 6)
    if len(created_materials) > 1:
        result["createdMaterials"] = created_materials
    if note_renamed:
        result["imageRenamed"] = note_renamed
        result["renameNote"] = (
            "an image named %r already existed, so Blender uniquified this one to %r. Look it up "
            "by the returned name, not the one you asked for." % (image_name, note_renamed))
    if uv_pairs:
        result["uvOverlapWarning"] = (
            "the layer baked through, '%s', has %d island pair(s) overlapping by bounding box - where they truly "
            "overlap, the later face's texels overwrite the earlier's. Deliberate for MIRRORED or STACKED UVs sharing "
            "one texture; a defect for a unique bake such as a lightmap or a game-ready atlas. uv_info reads the "
            "layer; uv_unwrap, or recipe_game_ready with forceUnwrap, lays it out again." % (baked_layer, uv_pairs))
    result["note"] = (
        "the image is IN MEMORY%s. Nothing else references it - the bake target node was %s - so "
        "it is lost on file reload unless it was saved."
        % ("" if saved else " and NOT saved; pass filepath to write it to disk",
           "kept" if keep_node else "removed"))
    return result


def op_assign_material_to_faces(params):
    """Point a range of polygons at one of the object's material SLOTS.

    set_material_slots decides WHICH materials a mesh has and in what order; this decides
    which faces use which. Multi-material meshes built in Blender need both, and only the
    first existed.

    ADDRESSED BY SLOT INDEX, not by material name, because the slot index is what a polygon
    actually stores (`polygon.material_index`) and what Unreal's FStaticMaterial array lines
    up against on import. A name would have to be resolved to an index anyway, and would be
    ambiguous the moment two slots hold the same material.

    `faces` omitted means EVERY polygon. A face index past the end is REFUSED rather than
    ignored, and `changed` reports how many polygons actually moved - a selection that
    matches nothing is otherwise indistinguishable from success.

    `fromSlot` selects every polygon CURRENTLY on that slot, which is the operation you
    want after set_material_slots reorders or resizes the slot list. Without it that costs
    a read, a client-side filter and a write of an explicit index list. It is REFUSED when
    no polygon uses that slot, unlike an empty `faces` list: asking for nothing is a
    request, but believing faces live on an empty slot is a wrong assumption about the
    mesh, and changed:0 would let it pass as success.
    """
    reject_unknown(params, ("object", "name", "slot", "slotIndex", "faces", "fromSlot"),
                   "assign_material_to_faces")
    obj = get_object(take(params, "object", "name", required=True), want_mesh=True)
    slot = take_int(params, "slot", "slotIndex", required=True)
    mesh = obj.data

    if not obj.material_slots:
        raise MifOpError(
            "'%s' has no material slots, so there is nothing to assign to. set_material_slots "
            "creates them." % obj.name)
    if slot < 0 or slot >= len(obj.material_slots):
        raise MifOpError(
            "slot %d is out of range - '%s' has %d slot(s) (0..%d). Faces store the slot INDEX, "
            "so an out-of-range one would render as the last slot with no error at all."
            % (slot, obj.name, len(obj.material_slots), len(obj.material_slots) - 1))

    total = len(mesh.polygons)
    if total == 0:
        raise MifOpError(
            "'%s' has NO polygons, so there is nothing to assign a material to. Without this check "
            "the loop below would simply not run and the op would report changed:0 with no error - "
            "the exact indistinguishable-from-success case it exists to prevent. NOTHING was "
            "changed." % obj.name)
    faces = params.get("faces")
    from_slot = params.get("fromSlot")
    if faces is not None and from_slot is not None:
        raise MifOpError(
            "pass EITHER faces OR fromSlot, not both - they are two ways of naming the same "
            "selection and combining them would silently pick one. NOTHING was changed.")

    if from_slot is not None:
        # A slot-to-slot remap: every polygon CURRENTLY on from_slot moves to slot. This is the
        # operation you want after set_material_slots reorders or resizes the list.
        if not isinstance(from_slot, int) or isinstance(from_slot, bool) \
                or from_slot < 0 or from_slot >= len(obj.material_slots):
            raise MifOpError(
                "fromSlot %r is out of range - '%s' has %d slot(s) (0..%d). NOTHING was changed."
                % (from_slot, obj.name, len(obj.material_slots), len(obj.material_slots) - 1))
        targets = [p.index for p in mesh.polygons if p.material_index == from_slot]
        if not targets:
            # DIFFERENT FROM faces:[] ON PURPOSE. An empty faces list is a caller asking for
            # nothing; an empty fromSlot is a caller who believes faces live on a slot that is
            # empty - a wrong assumption about the mesh, and changed:0 would let it pass.
            raise MifOpError(
                "no polygon on '%s' is currently using slot %d, so this would move nothing and "
                "report changed:0 as though it had worked. The slots in use are: %s. NOTHING was "
                "changed."
                % (obj.name, from_slot,
                   ", ".join("%d" % i for i in sorted({p.material_index for p in mesh.polygons}))))
    elif faces is None:
        targets = range(total)
    else:
        if not hasattr(faces, "__len__") or isinstance(faces, str):
            raise MifOpError("faces must be a list of polygon indices, or omitted for all")
        if len(faces) == 0:
            raise MifOpError(
                "faces:[] is empty - that assigns nothing and would report changed:0 as though it "
                "had worked. Omit `faces` to assign every polygon. NOTHING was changed.")
        bad = [f for f in faces if not isinstance(f, int) or f < 0 or f >= total]
        if bad:
            raise MifOpError(
                "face index/indices out of range: %s - this mesh has %d polygon(s) (0..%d). "
                "Refused rather than skipped, because a silently ignored index leaves the mesh "
                "looking assigned when it is not." % (bad[:10], total, total - 1))
        targets = faces

    changed = 0
    for i in targets:
        if mesh.polygons[i].material_index != slot:
            mesh.polygons[i].material_index = slot
            changed += 1
    mesh.update()

    counts = {}
    for poly in mesh.polygons:
        counts[poly.material_index] = counts.get(poly.material_index, 0) + 1
    return {
        "object": obj.name,
        "slot": slot,
        "slotMaterial": (obj.material_slots[slot].material.name
                         if obj.material_slots[slot].material else None),
        # len(targets), NOT the mesh size. `faces is None` is true on the fromSlot branch
        # too, where targets is only the polygons currently on that slot - so moving 12
        # faces off a slot on a 121-polygon mesh reported requested:121 changed:12 and read
        # as 109 silently skipped, when every face asked for had landed. fromSlot exists to
        # repair slot order, which is exactly when somebody is counting.
        "requested": len(targets),
        # MEASURED, not requested: assigning a face to the slot it already had is a no-op,
        # and reporting the request as the result would hide that.
        "changed": changed,
        "polygonCount": total,
        "facesPerSlot": {str(k): v for k, v in sorted(counts.items())},
    }


# "slot" IS DELIBERATELY ABSENT, for the reason spelled out above PRINCIPLED_ALIASES about
# normalStrength. It was listed here while the handler ignored it, so reject_unknown ACCEPTED
# it and nothing happened - the silent no-op this module calls the worst outcome available.
# parity_check caught it as an unreachable parameter. A caller passing it now gets a refusal
# naming what this op does take, which is the honest answer until slots are implemented.
SET_TEXTURE_KEYS = {"material", "name", "input", "file", "filepath", "image",
                    "colorspace", "colorSpace", "uvMap", "strength", "replace", "interpolation",
                    "scale", "offset", "rotation",
                    "extension",
                    # ANY SOCKET OF THE TREE (2026-10-04): an image into a Color Ramp, a Map Range, a Mix, a Math, an
                    # Invert or a Displacement node, a free node, an Invert step, a named node to reuse
                    "toNode", "toSocket", "fromSocket", "nodeName", "link", "invert", "normalConvention",
                    # the image's alpha mode (2026-10-04): STRAIGHT | PREMULTIPLIED (PREMUL) | CHANNEL_PACKED | NONE
                    "alphaMode",
                    # input "displacement" (2026-10-06): the Displacement node's Scale and Midlevel
                    "displacementScale", "midlevel"}

# WHICH COLOUR SPACE EACH SLOT NEEDS, and this is the trap the op exists around.
#
# A texture is DATA or it is COLOUR, and Blender decides how to decode it from the image's
# colorspace_settings rather than from where it is plugged in. A roughness or metallic map loaded as
# sRGB comes back through the transfer curve, so the values the artist authored are not the values
# the shader reads - and it looks plausible, just wrong, which is the failure this bridge is built
# to refuse. Only base colour and emission are colour; everything else is data.
#
# THE DEFAULT IS DERIVED, NOT GUESSED, and it is overridable, because an sRGB-encoded mask is a real
# thing somebody may have on disk.
_COLOR_INPUTS = ("baseColor", "emissive")
_DATA_NOTE = ("Non-Color, because this input reads VALUES rather than colour - decoding it through "
              "the sRGB transfer curve would change every number the shader sees.")

# normal is NOT a Principled alias and never can be: the Normal input takes a vector, and a normal
# map has to pass through a Normal Map node that converts tangent-space RGB into one. Wiring the
# image straight into Normal is the single most common way to get a material that looks lit wrong
# and reads as correct in every field. It is handled as its own case below.
_NORMAL_INPUT = "normal"

# DISPLACEMENT IS NOT A PRINCIPLED INPUT EITHER (2026-10-06). A height map for true displacement goes through a
# Displacement node (Height in, Displacement out) into the Material Output's Displacement socket - not into Normal,
# which only shades. The material's displacement method decides what Cycles does with it: BUMP (the default) shades
# only, DISPLACEMENT or BOTH moves the surface, and the reply names the method so a BUMP material is not mistaken for
# a displaced one.
_DISPLACEMENT_INPUT = "displacement"

# THE FACTOR SOCKET WAS RENAMED AND ITS IDENTIFIER WAS NOT. Measured 2026-10-04: an Invert, a Color Ramp and a
# Hue/Saturation node name it "Fac" on 3.6.23 and 4.4.0 and "Factor" on 5.2.1, identifier "Fac" on all three - so a
# recipe that says Fac is refused by name on 5.2 and one that says Factor on 3.6. Either spelling finds it.
_SOCKET_SPELLINGS = {"fac": "factor", "factor": "fac"}


def find_input_socket(node, wanted):
    """(socket, how) for `wanted` on a node: its NAME among the enabled inputs, else its IDENTIFIER, else the other
    spelling of Fac/Factor. (None, None) when nothing matches. Only ENABLED sockets count - a Mix node carries one
    A and B per data type under the same name and reads only the active one (usable_sockets)."""
    live, _inactive = usable_sockets(node.inputs, str(wanted))
    if live:
        return live[0], "name"
    for sock in node.inputs:
        if getattr(sock, "enabled", True) and sock.identifier == str(wanted):
            return sock, "identifier"
    other = _SOCKET_SPELLINGS.get(str(wanted).strip().lower())
    if other:
        for sock in node.inputs:
            if getattr(sock, "enabled", True) and other in (sock.name.lower(), sock.identifier.lower()):
                return sock, "the other spelling (%s on this build)" % sock.name
    return None, None


def load_image_file(path, tail="NOTHING was changed."):
    """Load an image from disk, refusing a missing file and a file with no pixels. Returns (image, loaded_now).

    STAT IT FIRST: images.load on a missing path still creates a datablock, and Blender renders an unresolved image
    MAGENTA - loud on screen, invisible in every response. A file that exists and is not an image loads without
    raising and reports 0x0; it is removed again if this call loaded it. Shared by set_material_texture and
    create_texture (2026-10-04), so the two cannot accept different files."""
    abspath = os.path.abspath(bpy.path.abspath(str(path)))
    if not os.path.isfile(abspath):
        raise MifOpError("no file at '%s'. Blender would still create an image datablock and render it MAGENTA, "
                         "which is obvious on screen and invisible to anything reading this response. %s"
                         % (abspath, tail))
    before = {i.name for i in bpy.data.images}
    try:
        img = bpy.data.images.load(abspath, check_existing=True)
    except RuntimeError as exc:
        raise MifOpError("Blender could not load '%s': %s. %s" % (abspath, exc, tail))
    loaded_now = img.name not in before
    if tuple(img.size) == (0, 0):
        if loaded_now:
            bpy.data.images.remove(img)
        raise MifOpError("'%s' loaded but has no pixels (size 0x0), so it is not an image Blender can read.%s %s"
                         % (path, " It was removed again." if loaded_now else "", tail))
    return img, loaded_now


def _uv_pair(params, key, third):
    """A scale/offset parameter as the 3-vector a Mapping node takes, or None if absent.

    LIFTED OUT OF set_material_texture, where it was a closure, when add_procedural_texture needed
    the same three lines. Two copies of a units-and-shape parser is how the second one ends up
    accepting a boolean or a 3-list that the first refuses, and the caller finds out from a
    texture that tiles differently through one op than the other.

    scale and offset take a scalar (applied to both U and V) or [u, v]. The third component is the
    caller's because it differs by role: 1.0 is the identity for a scale and 0.0 for a location,
    and a zero W scale on a 3D procedural texture flattens it to a slice.
    """
    raw = take(params, key, default=None)
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise MifOpError("%s takes a number or [u, v], not a boolean. NOTHING was changed." % key)
    if isinstance(raw, (int, float)):
        # Handed to finite_floats RAW rather than through float(). The isinstance above already
        # guarantees a number, and finite_floats does the coercion and the NaN/Infinity refusal
        # itself - so the float() calls were doing nothing except tripping audit_unguarded_numbers,
        # which cannot see that the value is already narrowed and is right to flag a bare float()
        # on caller input by default.
        vals = finite_floats([raw, raw], key)
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        vals = finite_floats([raw[0], raw[1]], key)
    else:
        raise MifOpError(
            "%s takes a number (applied to both U and V) or [u, v]. NOTHING was changed." % key)
    return (vals[0], vals[1], third)


def _procedural_vec(params, key, third):
    """add_procedural_texture's `scale` / `offset`: what _uv_pair takes - a number (X and Y) or [u, v] - AND [x, y, z].

    A PROCEDURAL PATTERN IS 3D, and _uv_pair (built for an image's two axes) writes the third component as the
    identity. In Object or Generated space, on a face that does not face Z, a scale that reaches only X and Y
    stretches the pattern along Z by the same factor: building the ambient cabin (2026-09-21, ledger B191) a Voronoi
    fieldstone at scale 5.5 rendered as vertical wood grain on the chimney breast until the Mapping node's Z was
    written by hand. A number keeps meaning X and Y - scenes built on that must render as they did - and [x, y, z]
    is how a caller scales all three; the reply's scaleNote says when a 3D space got only two.

    Returns (the Mapping vector or None, True when all three were given)."""
    raw = take(params, key, default=None)
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        if any(isinstance(v, bool) for v in raw):
            raise MifOpError("%s takes numbers, not booleans. NOTHING was changed." % key)
        return tuple(finite_floats([raw[0], raw[1], raw[2]], key)), True
    if raw is not None and not isinstance(raw, (bool, int, float)) and not (
            isinstance(raw, (list, tuple)) and len(raw) == 2):
        raise MifOpError("%s takes a number (applied to X and Y), [u, v], or [x, y, z]. NOTHING was changed." % key)
    return _uv_pair(params, key, third), False


def _flip_green(tree, source, normal_node, made):
    """Separate Color -> 1 - G -> Combine Color between an image and a Normal Map node: a DirectX (-Y) map read as
    OpenGL, on builds whose Normal Map node has no convention. Returns the combined colour output."""
    x, y = normal_node.location[0] - 600, normal_node.location[1]
    sep = tree.nodes.new("ShaderNodeSeparateColor")
    flip = tree.nodes.new("ShaderNodeMath")
    comb = tree.nodes.new("ShaderNodeCombineColor")
    made.extend([sep, flip, comb])
    sep.location, flip.location, comb.location = (x, y), (x + 200, y), (x + 400, y)
    flip.operation = "SUBTRACT"
    flip.inputs[0].default_value = 1.0
    tree.links.new(source, sep.inputs[0])
    tree.links.new(sep.outputs[0], comb.inputs[0])
    tree.links.new(sep.outputs[1], flip.inputs[1])
    tree.links.new(flip.outputs[0], comb.inputs[1])
    tree.links.new(sep.outputs[2], comb.inputs[2])
    return comb.outputs[0]


def op_set_material_texture(params):
    """Put an image from disk into a material - which nothing in this addon could do.

    THE HOLE THIS CLOSES, found by asking what the read side reports that the write side cannot
    produce. describe_material has always walked TEX_IMAGE nodes and reported their images, and
    bpy.data.images.load appears exactly ONCE in the whole addon - in ops_world, for an HDRI. So a
    material could be given numbers and never a map, and every job whose deliverable is a textured
    material was closed at the door.

    A MISSING FILE RENDERS MAGENTA, not black and not an error. Blender creates the image datablock
    with its source unresolved and shades it bright pink, so the mistake is loud in the viewport and
    completely silent in every response an agent can read. The path is stat'd BEFORE anything is
    created, and a load that produces no pixels is refused with the node removed again.

    params:
      material / name (str)     required
      input (str)               which Principled input to drive. Any of set_material_properties'
                                names, plus 'normal', which is not a Principled alias - see below.
      toNode + toSocket (str)   OR any node's input in the same tree: a node NAME, or a node TYPE
                                (colorRamp, mapRange, mix, math, invert, displacement, bump,
                                hueSaturation, a raw bl_idname) to create one. A socket is found by
                                name, then identifier; Fac and Factor are one socket (renamed at 5.x).
      link (bool)               false leaves the Image Texture node FREE - wired to nothing.
      nodeName (str)            name the Image Texture node; an Image Texture node already called
                                that is REUSED and only its image swapped (its links stay).
      fromSocket (str)          Color (default) or Alpha. 'alpha' as the input takes Alpha.
      invert (bool)             an Invert node between the image and the destination.
      file / filepath (str)     the image on disk. Excludes `image`.
      image (str)               an image datablock ALREADY in the file, by name. Excludes `file`.
      colorspace (str)          override. Default sRGB for baseColor/emissive, Non-Color for the
                                other Principled inputs; for toNode, sRGB when the socket takes a
                                colour and Non-Color when it takes a value or a vector; a free or
                                reused node keeps the image's own unless one is given.
      uvMap (str)               drive the texture from a named UV layer instead of the active one.
      strength (float)          normal input only - the Normal Map node's strength.
      normalConvention (str)    normal input only: OPENGL (default) or DIRECTX (green flipped, Unreal's
                                maps) - the node's own switch on 5.2+, a 1 - G step before it earlier.
      interpolation (str)       Linear | Closest | Cubic | Smart
      extension (str)           REPEAT | EXTEND | CLIP | MIRROR
      alphaMode (str)           the image's alpha: STRAIGHT | PREMULTIPLIED | CHANNEL_PACKED | NONE (this build's
                                list) - premultiplied stops a logo's outline showing; set on the image, last
      displacementScale, midlevel (float)  input 'displacement' only: the Displacement node's Scale and Midlevel.
                                The image feeds its Height and it feeds the active Material Output's
                                Displacement; Non-Color. The reply's displacementMethod says whether the
                                surface moves (DISPLACEMENT / BOTH) or is only shaded (BUMP).
      replace (bool)            replace an image already on that input. Default true.

    NORMAL IS ITS OWN CASE. The Principled Normal socket takes a VECTOR, and a tangent-space normal
    map is RGB - so this builds the Normal Map node between them. Wiring the image directly gives a
    material that is wrong in a way every field in every response still reports as correct.
    """
    reject_unknown(params, SET_TEXTURE_KEYS, "set_material_texture")
    mat_name = take(params, "material", "name", required=True, kind=str)
    which = take(params, "input", default=None, kind=str)
    which = str(which) if which is not None else None
    path = take(params, "file", "filepath", default=None, kind=str)
    image_name = take(params, "image", default=None, kind=str)
    to_node_name = take(params, "toNode", default=None, kind=str)
    to_socket_name = take(params, "toSocket", default=None, kind=str)
    from_socket = take(params, "fromSocket", default=None, kind=str)
    node_name = take(params, "nodeName", default=None, kind=str)
    link_flag = params.get("link")
    if link_flag is not None and not isinstance(link_flag, bool):
        raise MifOpError("link takes true or false, got %r. NOTHING was changed." % (link_flag,))
    invert = take_bool(params, "invert", default=False)

    if (path is None) == (image_name is None):
        raise MifOpError("pass `file` (a path on disk) OR `image` (a datablock already in this "
                         "file), not both and not neither. NOTHING was changed.")

    known = sorted(set(PRINCIPLED_ALIASES) | {_NORMAL_INPUT, _DISPLACEMENT_INPUT})
    if which is not None and which not in known:
        raise MifOpError("'%s' is not an input this op can drive. Available: %s - or any node's socket "
                         "with toNode + toSocket. NOTHING was changed." % (which, ", ".join(known)))
    if to_node_name is not None and to_socket_name is None:
        raise MifOpError("`toNode` needs `toSocket` - a node has several inputs and this op will not guess "
                         "which one. NOTHING was changed.")
    if to_socket_name is not None and to_node_name is None:
        raise MifOpError("`toSocket` needs `toNode` to say which node it is on. NOTHING was changed.")

    mat = bpy.data.materials.get(str(mat_name))
    if mat is None:
        have = sorted(m.name for m in bpy.data.materials)[:25]
        raise MifOpError("no material named '%s'. This file has: %s. NOTHING was changed."
                         % (mat_name, ", ".join(have) if have else "(none)"))
    if not mat.use_nodes or mat.node_tree is None:
        raise MifOpError("material '%s' has use_nodes OFF, so its node tree is ignored entirely and its flat "
                         "colour is what renders - an image added to it would be something nothing looks at. "
                         "create_material makes a material with nodes on. NOTHING was changed." % mat.name)
    tree = mat.node_tree

    # ---- WHERE THE IMAGE GOES: one of a Principled input, a node's socket, nowhere, or a node already there
    reuse = tree.nodes.get(str(node_name)) if node_name else None
    if reuse is not None and reuse.type != "TEX_IMAGE":
        raise MifOpError("nodeName '%s' is already a %s in '%s', not an Image Texture node, so it cannot hold the "
                         "image. Pick another name. NOTHING was changed." % (node_name, reuse.bl_idname, mat.name))
    dests = [k for k, v in (("input", which is not None), ("toNode", to_node_name is not None),
                            ("link:false", link_flag is False)) if v]
    if len(dests) > 1:
        raise MifOpError("pass ONE of %s - each says where the image goes. NOTHING was changed."
                         % " / ".join(dests))
    if not dests and reuse is None:
        raise MifOpError("say where the image goes: `input` (a Principled input), `toNode` + `toSocket` (any "
                         "node's input), `link: false` (a free node), or `nodeName` of an Image Texture node "
                         "already in the tree to swap its image. NOTHING was changed.")
    swap_only = not dests
    if invert and (swap_only or link_flag is False):
        raise MifOpError("invert puts an Invert node on the way to a destination, and this call has none. "
                         "NOTHING was changed.")
    if invert and which == _NORMAL_INPUT:
        raise MifOpError("invert flips all three channels, which turns a tangent-space normal map inside out "
                         "rather than inverting a value - it is refused on the normal input. NOTHING was changed.")
    if params.get("strength") is not None and which != _NORMAL_INPUT:
        raise MifOpError("strength is the Normal Map node's strength and only applies to input:'normal'. "
                         "NOTHING was changed.")
    disp_stray = [k for k in ("displacementScale", "midlevel") if params.get(k) is not None]
    if disp_stray and which != _DISPLACEMENT_INPUT:
        raise MifOpError("%s set(s) the Displacement node and only appl(ies) to input:'displacement'. NOTHING was "
                         "changed." % ", ".join(disp_stray))
    disp_scale = take_float(params, "displacementScale", default=None)
    disp_mid = take_float(params, "midlevel", default=None)
    if swap_only or link_flag is False:
        stray = [k for k in ("replace", "fromSocket") if params.get(k) is not None]
        if stray:
            raise MifOpError("%s describe(s) a connection, and this call makes none. NOTHING was changed."
                             % ", ".join(stray))
    if reuse is not None:
        stray = [k for k in ("uvMap", "scale", "offset", "rotation") if params.get(k) is not None]
        if stray:
            raise MifOpError("%s build(s) coordinates for a NEW Image Texture node, and nodeName '%s' reuses one "
                             "already wired. Drop them, or name a new node. NOTHING was changed."
                             % (", ".join(stray), reuse.name))

    out_name = "Alpha" if which == "alpha" else "Color"
    if from_socket is not None:
        match = [s for s in ("Color", "Alpha") if s.lower() == str(from_socket).strip().lower()]
        if not match:
            raise MifOpError("fromSocket '%s' is not an Image Texture output. It has: Color, Alpha. NOTHING was "
                             "changed." % from_socket)
        out_name = match[0]
        if which in (_NORMAL_INPUT, _DISPLACEMENT_INPUT) and out_name != "Color":
            raise MifOpError("a %s map is read from the image's Color, not its Alpha. NOTHING was changed." % which)

    # EVERY REFUSAL ABOVE THIS LINE, before a node or an image datablock exists. A half-built
    # material is worse than none: it renders, and it renders wrong.
    node = None
    target = None
    dest_node = None
    create_type = None
    if which == _DISPLACEMENT_INPUT:
        outs = [n for n in tree.nodes if n.type == "OUTPUT_MATERIAL"]
        node = next((n for n in outs if n.is_active_output), outs[0] if outs else None)
        if node is None:
            raise MifOpError("material '%s' has no Material Output node, so a displacement has nothing to feed. "
                             "NOTHING was changed." % mat.name)
        target = node.inputs.get("Displacement")
        if target is None:
            raise MifOpError("the Material Output on Blender %s has no Displacement input. NOTHING was changed."
                             % bpy.app.version_string)
    elif which is not None:
        node = _principled(mat)
        if which == _NORMAL_INPUT:
            target = node.inputs.get("Normal")
            if target is None:
                raise MifOpError("this Principled BSDF has no Normal input on Blender %s. NOTHING was "
                                 "changed." % bpy.app.version_string)
        else:
            target = _resolve_input(node, which)
            if target is None:
                raise MifOpError(
                    "'%s' has no socket on Blender %s - the Principled inputs were renamed at 4.0 and "
                    "this build offers: %s. NOTHING was changed."
                    % (which, bpy.app.version_string, ", ".join(s.name for s in node.inputs)))
    elif to_node_name is not None:
        dest_node = tree.nodes.get(str(to_node_name))
        if dest_node is None:
            # NOT A NODE OF THIS TREE: a TYPE to create, if the vocabulary knows it. Refused naming both lists.
            try:
                create_type = resolve_node_type(str(to_node_name))
            except MifOpError:
                raise MifOpError("'%s' is neither a node in material '%s' nor a node type. The tree holds: %s. A "
                                 "type is a friendly name (colorRamp, mapRange, mix, math, invert, displacement, "
                                 "bump, hueSaturation, ...) or a bl_idname. NOTHING was changed."
                                 % (to_node_name, mat.name, ", ".join(n.name for n in tree.nodes) or "(nothing)"))
            if not create_type.startswith("ShaderNode"):
                raise MifOpError("'%s' resolves to %s, which is not a shader node. NOTHING was changed."
                                 % (to_node_name, create_type))
        else:
            target, how = find_input_socket(dest_node, to_socket_name)
            if target is None:
                raise MifOpError("node '%s' (%s) has no input '%s'. Its inputs (name / identifier): %s. NOTHING "
                                 "was changed." % (dest_node.name, dest_node.bl_idname, to_socket_name,
                                                   ", ".join("%s / %s" % (s.name, s.identifier)
                                                             for s in dest_node.inputs
                                                             if getattr(s, "enabled", True)) or "(none)"))

    replace = take_bool(params, "replace", default=True)
    if target is not None and target.is_linked and not replace:
        raise MifOpError("'%s' on '%s' is already driven by %s and replace:false was given. "
                         "NOTHING was changed."
                         % (target.name, target.node.name, target.links[0].from_node.name))

    # Parsed before the image loads, so a bad value leaves no datablock behind.
    interp = take(params, "interpolation", default=None, kind=str)
    ext = take(params, "extension", default=None, kind=str)
    scale_uv = _uv_pair(params, "scale", 1.0)
    offset_uv = _uv_pair(params, "offset", 0.0)
    rot_deg = take(params, "rotation", default=None)
    if rot_deg is not None:
        rot_deg = finite_float(rot_deg, "rotation")
    if scale_uv is not None and (scale_uv[0] == 0.0 or scale_uv[1] == 0.0):
        raise MifOpError(
            "scale 0 collapses the texture to one pixel's colour across the whole surface, "
            "which looks like a flat fill rather than an error. Pass a non-zero scale. "
            "NOTHING was changed.")
    strength = take_float(params, "strength", default=None)
    # THE NORMAL MAP'S CONVENTION (2026-10-06): DIRECTX reads green flipped (Unreal's maps). The Normal Map node's own
    # switch from 5.2 (its `convention`, read on the build); before it a Separate Color, 1 - G and Combine Color in
    # front of the node. invert flips all three channels and is refused on normal for that reason.
    convention = take(params, "normalConvention", default=None, kind=str)
    if convention is not None:
        convention = convention.upper()
        if convention not in ("OPENGL", "DIRECTX"):
            raise MifOpError("normalConvention is OPENGL (+Y, Blender) or DIRECTX (-Y, Unreal), got %r. NOTHING was "
                             "changed." % params.get("normalConvention"))
        if which != _NORMAL_INPUT:
            raise MifOpError("normalConvention is a normal map's - pass input 'normal'. NOTHING was changed.")
    alpha_mode = take(params, "alphaMode", default=None, kind=str)
    if alpha_mode is not None:
        alpha_mode = str(alpha_mode).upper()
        alpha_mode = {"PREMULTIPLIED": "PREMUL"}.get(alpha_mode, alpha_mode)
        modes = [e.identifier for e in bpy.types.Image.bl_rna.properties["alpha_mode"].enum_items]
        if alpha_mode not in modes:
            raise MifOpError("alphaMode must be one of this Blender's %s, got %r. NOTHING was changed."
                             % (", ".join(modes), params.get("alphaMode")))

    if image_name is not None:
        img = bpy.data.images.get(str(image_name))
        if img is None:
            have = sorted(i.name for i in bpy.data.images)[:25]
            raise MifOpError("no image datablock named '%s'. This file has: %s. NOTHING was "
                             "changed." % (image_name, ", ".join(have) if have else "(none)"))
        if tuple(img.size) == (0, 0):
            raise MifOpError("'%s' has no pixels (size 0x0), so it is not an image Blender can read. NOTHING was "
                             "changed." % image_name)
        loaded_now = False
    else:
        img, loaded_now = load_image_file(path)

    # EVERYTHING THIS BUILDS IS UNDONE IF ANY OF IT REFUSES, and this op shipped without that
    # for a few hours on 2026-09-04 - written the same day, in the same session, as the fix for
    # exactly this shape in ops_lightcam's _vec3. Finding it in my own new code an hour later is
    # the reason the rule is a rule rather than a habit.
    #
    # The image is loaded from disk BEFORE the node exists, because a file that turns out not to
    # be an image must be caught before anything is wired to it. So the undo is structural: nodes
    # are tracked as they are made, a reused node gets its old image back, links this call removed
    # are restored, and the image is removed only if THIS call loaded it.
    _made_nodes = []
    _removed_links = []
    swapped_from = None
    cs_before = img.colorspace_settings.name
    try:
        if reuse is not None:
            tex = reuse
            swapped_from = (tex.image.name if tex.image is not None else "")
            tex.image = img
        else:
            tex = tree.nodes.new("ShaderNodeTexImage")
            _made_nodes.append(tex)
            tex.image = img
            if node_name:
                tex.name = str(node_name)
                tex.label = str(node_name)
            anchor = node if node is not None else (dest_node or _principled_or_none(tree))
            ax, ay = (anchor.location[0], anchor.location[1]) if anchor is not None else (0.0, 0.0)
            tex.location = (ax - (900 if invert else 600), ay)
        if interp:
            _enum_or_refuse(tex, "interpolation", interp, "interpolation")
        if ext:
            _enum_or_refuse(tex, "extension", ext, "extension")

        # THE DESTINATION NODE, made now that the image is known good, and its socket checked on the node
        # itself (a type created here has its sockets only once it exists).
        if create_type is not None:
            dest_node = tree.nodes.new(create_type)
            _made_nodes.append(dest_node)
            if dest_node.bl_idname != create_type:
                raise MifOpError("Blender %s accepted '%s' and produced a '%s' - an UNDEFINED node. Every node this "
                                 "call made was removed again - NOTHING was changed."
                                 % (bpy.app.version_string, create_type, dest_node.bl_idname))
            dest_node.location = (tex.location[0] + (600 if invert else 300), tex.location[1])
            target, how = find_input_socket(dest_node, to_socket_name)
            if target is None:
                raise MifOpError("a %s has no input '%s'. Its inputs (name / identifier): %s. Every node this call "
                                 "made was removed again - NOTHING was changed."
                                 % (dest_node.bl_idname, to_socket_name,
                                    ", ".join("%s / %s" % (s.name, s.identifier) for s in dest_node.inputs
                                              if getattr(s, "enabled", True)) or "(none)"))

        # COLOUR SPACE IS SET ON THE IMAGE, NOT THE NODE, so it applies everywhere that image is used -
        # which is why an override is reported rather than assumed.
        want_cs = take(params, "colorspace", "colorSpace", default=None, kind=str)
        if want_cs:
            chosen, cs_why = str(want_cs), "asked for"
        elif which is not None:
            chosen = "sRGB" if which in _COLOR_INPUTS else "Non-Color"
            cs_why = ("a colour input" if which in _COLOR_INPUTS else
                      "this input reads values, not colour")
        elif target is not None:
            # A SOCKET THAT TAKES A COLOUR gets a colour image; a value or a vector reads data. A mask into a Color
            # Ramp's Fac, a cavity map into a Displacement Height, a roughness map into a Math node: Non-Color.
            chosen = "sRGB" if target.type == "RGBA" else "Non-Color"
            cs_why = ("the socket takes a colour" if target.type == "RGBA" else
                      "the socket takes a %s, so the image is read as data" % target.type.lower())
        else:
            chosen, cs_why = None, "kept: a free or reused node does not say what the image holds"
        if chosen is not None and chosen != img.colorspace_settings.name and img.source == "GENERATED" \
                and img.is_dirty:
            # A GENERATED IMAGE IS REGENERATED BY A COLOUR-SPACE CHANGE. Measured 2026-10-04 on 3.6.23, 4.4.0 and
            # 5.2.1: pixels written as [0.2, 0.4, 0.6] read back [0, 0, 0] (its fill colour) after colorspace became
            # Non-Color. A painted canvas that was never saved would be wiped by wiring it, so that is refused.
            raise MifOpError("'%s' is a generated image with unsaved pixels, and changing its colour space from %s to %s "
                             "regenerates it from its fill colour - the painting would be lost. save_image it first "
                             "(it then reads its file), or pass colorspace '%s'. NOTHING was changed."
                             % (img.name, img.colorspace_settings.name, chosen, img.colorspace_settings.name))
        if chosen is not None:
            try:
                img.colorspace_settings.name = chosen
            except TypeError:
                valid = [i.identifier for i in
                         img.colorspace_settings.bl_rna.properties["name"].enum_items]
                raise MifOpError("'%s' is not a colour space on this Blender. Available: %s. NOTHING was "
                                 "changed." % (chosen, ", ".join(valid)))

        uv_node = None
        uv_map = take(params, "uvMap", default=None, kind=str)
        if uv_map:
            uv_node = tree.nodes.new("ShaderNodeUVMap")
            _made_nodes.append(uv_node)
            uv_node.uv_map = str(uv_map)
            uv_node.location = (tex.location[0] - 300, tex.location[1])
            tree.links.new(uv_node.outputs["UV"], tex.inputs["Vector"])

        # ---- TILING, WHICH THIS OP COULD NOT DO -------------------------------------------------
        # WHY IT MATTERS, from a real one: a floor material built through this op takes the mesh's
        # UVs at 1:1, so a 2 m texture stretched over a 30 m deck and read as carpet. A Mapping node is
        # the per-texture, non-destructive answer. scale and offset take a scalar or [u, v]; rotation is
        # DEGREES and scalar, converted to the socket's radians at the boundary.
        want_mapping = scale_uv is not None or offset_uv is not None or rot_deg is not None
        if want_mapping:
            mapping = tree.nodes.new("ShaderNodeMapping")
            _made_nodes.append(mapping)
            mapping.location = (tex.location[0] - 320, tex.location[1] - 140)
            if scale_uv is not None:
                mapping.inputs["Scale"].default_value = scale_uv
            if offset_uv is not None:
                mapping.inputs["Location"].default_value = offset_uv
            if rot_deg is not None:
                mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(rot_deg))
            # A SOURCE IS REQUIRED. An unconnected Mapping Vector reads as a constant zero and paints the
            # whole surface one colour; a Texture Coordinate node supplies the active layer explicitly.
            if uv_node is None:
                uv_node = tree.nodes.new("ShaderNodeTexCoord")
                _made_nodes.append(uv_node)
                uv_node.location = (mapping.location[0] - 240, mapping.location[1])
            tree.links.new(uv_node.outputs["UV"], mapping.inputs["Vector"])
            tree.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])

        links_made = []
        replaced = None
        normal_node = None
        disp_node = None
        invert_node = None
        convention_route = None
        if target is not None:
            # THE OLD LINK GOES FIRST. links.new on a single-input socket replaces silently, but the node
            # that fed it is left floating in the tree where describe_material will keep reporting it.
            for link in list(target.links):
                replaced = link.from_node.name
                _removed_links.append((link.from_socket, link.to_socket))
                tree.links.remove(link)
            source = tex.outputs[out_name]
            if invert:
                invert_node = tree.nodes.new("ShaderNodeInvert")
                _made_nodes.append(invert_node)
                invert_node.location = (tex.location[0] + 300, tex.location[1])
                inv_in = invert_node.inputs["Color"]
                tree.links.new(source, inv_in)
                links_made.append("%s.%s -> %s.%s" % (tex.name, source.name, invert_node.name, inv_in.name))
                source = invert_node.outputs["Color"]
            if which == _NORMAL_INPUT:
                normal_node = tree.nodes.new("ShaderNodeNormalMap")
                _made_nodes.append(normal_node)
                normal_node.location = (node.location[0] - 300, node.location[1] - 300)
                if strength is not None:
                    normal_node.inputs["Strength"].default_value = strength
                if uv_map:
                    normal_node.uv_map = str(uv_map)
                if convention == "DIRECTX" and "convention" in normal_node.bl_rna.properties:
                    normal_node.convention = "DIRECTX"
                    convention_route = "the Normal Map node's convention"
                elif convention == "DIRECTX":
                    source = _flip_green(tree, source, normal_node, _made_nodes)
                    convention_route = "Separate Color, 1 - G, Combine Color"
                tree.links.new(source, normal_node.inputs["Color"])
                links_made.append("%s.%s -> %s.Color" % (tex.name, source.name, normal_node.name))
                source = normal_node.outputs["Normal"]
            elif which == _DISPLACEMENT_INPUT:
                disp_node = tree.nodes.new("ShaderNodeDisplacement")
                _made_nodes.append(disp_node)
                disp_node.location = (node.location[0] - 250, node.location[1] - 250)
                if disp_scale is not None:
                    disp_node.inputs["Scale"].default_value = disp_scale
                if disp_mid is not None:
                    disp_node.inputs["Midlevel"].default_value = disp_mid
                tree.links.new(source, disp_node.inputs["Height"])
                links_made.append("%s.%s -> %s.Height" % (source.node.name, source.name, disp_node.name))
                source = disp_node.outputs["Displacement"]
            tree.links.new(source, target)
            links_made.append("%s.%s -> %s.%s" % (source.node.name, source.name, target.node.name, target.name))

            # READ BACK OFF THE TREE. links.new returns a link object whether or not Blender kept it, and a
            # rejected link leaves the socket exactly as unlinked as it was.
            feeder = source.node.name
            if not (target.is_linked and any(lk.from_node.name == feeder for lk in target.links)):
                raise MifOpError("the link to '%s' was not accepted by Blender and the nodes were removed "
                                 "again - nothing is wired. NOTHING was changed." % target.name)

        alpha_before = img.alpha_mode
        if alpha_mode is not None:
            # SET ON THE IMAGE, like the colour space, so every user of it sees it - written last, after the
            # last refusal, so a refused call leaves an existing image's alpha as it was.
            img.alpha_mode = alpha_mode
        return {
            "ok": True,
            "material": mat.name,
            "input": which,
            "alphaMode": img.alpha_mode,
            "alphaModeBefore": alpha_before,
            "toNode": dest_node.name if dest_node is not None else None,
            "toNodeCreated": create_type is not None,
            "toSocket": target.name if (target is not None and which is None) else None,
            "toSocketIdentifier": target.identifier if (target is not None and which is None) else None,
            "socketMatchedBy": how if (target is not None and which is None) else None,
            "fromSocket": out_name if target is not None else None,
            "imageNode": tex.name,
            "reusedNode": reuse is not None,
            "previousImage": swapped_from if reuse is not None else None,
            "image": img.name,
            "imageFile": bpy.path.abspath(img.filepath) if img.filepath else None,
            "imageSize": list(img.size),
            "hasAlphaChannel": bool(img.depth in (32, 64)),
            # MEASURED OFF THE SOCKET, not assumed from the call. This is the field that says the
            # texture will actually be read.
            "linked": target is not None,
            "free": target is None and link_flag is False,
            "nodesMade": [n.name for n in _made_nodes],
            "linksMade": links_made,
            "colorSpace": img.colorspace_settings.name,
            "colorSpaceWasChanged": cs_before != img.colorspace_settings.name,
            "colorSpaceReason": cs_why,
            "colorSpaceNote": (None if which is None or which in _COLOR_INPUTS else _DATA_NOTE),
            "invertNode": getattr(invert_node, "name", None),
            "normalMapNode": getattr(normal_node, "name", None),
            "displacementNode": getattr(disp_node, "name", None),
            "displacementScale": (round(float(disp_node.inputs["Scale"].default_value), 6)
                                  if disp_node is not None else None),
            "midlevel": (round(float(disp_node.inputs["Midlevel"].default_value), 6)
                         if disp_node is not None else None),
            "displacementMethod": (getattr(mat, "displacement_method", None) if disp_node is not None else None),
            "displacementNote": (_displacement_note(mat) if disp_node is not None else None),
            "normalConvention": convention,
            "normalConventionRoute": convention_route,
            "uvMapNode": getattr(uv_node, "name", None),
            "uvMapNote": _uv_map_note(mat, uv_map),
            "replacedLinkFrom": replaced,
            "loadedFromDisk": loaded_now,
            "note": ("wired through a Normal Map node, which converts tangent-space RGB into the vector "
                     "the Normal input takes. Linking the image straight in is the usual mistake and "
                     "reads as correct in every field.") if normal_node is not None else None,
        }
    except Exception:  # noqa: BLE001 - widened 2026-09-07; the original is re-raised untouched
        # WIDENED FROM MifOpError when add_procedural_texture was written against this file's
        # pattern and the same hole showed up in the new code. A refusal is the expected exit and
        # is what this rollback was built for, but it is not the only one: a socket a build spells
        # differently raises KeyError and a missing attribute raises AttributeError, and catching
        # only MifOpError left the nodes AND the freshly loaded image behind while the call
        # reported a failure. The original exception is re-raised unchanged.
        for _n in _made_nodes:
            try:
                mat.node_tree.nodes.remove(_n)
            except Exception:  # noqa: BLE001 - cleanup must not mask the real failure
                pass
        for _from, _to in _removed_links:
            try:
                mat.node_tree.links.new(_from, _to)
            except Exception:  # noqa: BLE001
                pass
        if reuse is not None:
            try:
                reuse.image = bpy.data.images.get(swapped_from) if swapped_from else None
            except Exception:  # noqa: BLE001
                pass
        try:
            if img.colorspace_settings.name != cs_before:
                img.colorspace_settings.name = cs_before
        except Exception:  # noqa: BLE001
            pass
        if loaded_now:
            try:
                bpy.data.images.remove(img)
            except Exception:  # noqa: BLE001
                pass
        raise


def _displacement_note(mat):
    """What the material's displacement method does with a displacement it was just given, or None when it moves."""
    method = getattr(mat, "displacement_method", None)
    if method == "BUMP":
        return ("this material's displacementMethod is BUMP, so Cycles only SHADES the height map - the surface does "
                "not move. bl_set_material_settings displacementMethod BOTH (or DISPLACEMENT) moves it, on a mesh "
                "with enough faces (a Subdivision Surface modifier) to carry it.")
    if method is None:
        return "this Blender has no per-material displacement method; it is set on the Cycles material settings."
    return None


def _enum_value_or_refuse(holder, attr, value, label):
    """The identifier this enum value resolves to, or a refusal naming what the build offers.

    SPLIT OUT FROM THE SETTER so a caller with several enums to write can resolve all of them
    before writing any. set_color_ramp is why: it writes colorMode, hueInterpolation and
    interpolation, and with validation inside the write a bad third value left the first two
    applied under a message that ends "NOTHING was changed" - the exact shape _coerce_socket was
    split out of set_material_properties to end, in this same file.
    """
    prop = holder.bl_rna.properties.get(attr)
    if prop is None:
        raise MifOpError("%s does not exist on a %s on Blender %s. NOTHING was changed."
                         % (label, holder.bl_rna.identifier, bpy.app.version_string))
    valid = [i.identifier for i in prop.enum_items]
    want = str(value).upper()
    match = [v for v in valid if v.upper() == want]
    if not match:
        raise MifOpError("'%s' is not a valid %s on Blender %s. Available: %s. NOTHING was changed."
                         % (value, label, bpy.app.version_string, ", ".join(valid)))
    return match[0]


def _enum_or_refuse(holder, attr, value, label):
    """Set an enum from the live list, or refuse naming what this build offers."""
    setattr(holder, attr, _enum_value_or_refuse(holder, attr, value, label))



_MATSET_KEYS = {"material", "name", "blendMethod", "shadowMethod", "backfaceCulling",
                "alphaThreshold", "screenRefraction", "displacementMethod", "surfaceRenderMethod",
                "transparentShadow", "raytraceRefraction"}

# param -> (real property name, which builds have it)
#
# EVERY ROW READ OFF bl_rna ON ALL FOUR INSTALLS, not from release notes, and the drift is the
# reason this is a separate op rather than more parameters on set_material_properties:
#
#   blend_method              every build, same four values
#   shadow_method             3.6 and 4.2 ONLY - EEVEE Next dropped it at 4.4
#   displacement_method       4.2 and later - absent on 3.6
#   surface_render_method     4.2 and later
#   use_transparent_shadow    4.2 and later
#   use_raytrace_refraction   4.2 and later
#
# So a caller asking for a shadow method on 4.4 is asking for something that build cannot do, and
# the house rule is that this is REFUSED naming what happened to it rather than accepted and
# dropped. Same shape as set_light_shadow, for the same reason.
_MATSET_MAP = {
    "blendMethod":          ("blend_method", "every build"),
    "shadowMethod":         ("shadow_method", "3.6 and 4.2 only - EEVEE Next dropped it at 4.4"),
    "backfaceCulling":      ("use_backface_culling", "every build"),
    "alphaThreshold":       ("alpha_threshold", "every build"),
    "screenRefraction":     ("use_screen_refraction", "every build"),
    "displacementMethod":   ("displacement_method", "4.2 and later"),
    "surfaceRenderMethod":  ("surface_render_method", "4.2 and later"),
    "transparentShadow":    ("use_transparent_shadow", "4.2 and later"),
    "raytraceRefraction":   ("use_raytrace_refraction", "4.2 and later"),
}
_MATSET_BOOL = {"backfaceCulling", "screenRefraction", "transparentShadow", "raytraceRefraction"}
_MATSET_FLOAT = {"alphaThreshold"}
# THE ONE KEY CYCLES READS - see inertUnderThisEngine in op_set_material_settings.
_MATSET_CYCLES_READS = {"displacementMethod"}


def op_set_material_settings(params):
    """The material settings that are NOT Principled inputs - transparency, shadows, culling.

    THE READ SIDE HAS ALWAYS REPORTED blend_method. describe_material and create_material both
    return blendMethod, and NOTHING anywhere could write it - a read/write asymmetry this repo
    treats as a defect class, sitting in the middle of the most-used material op. It is also the
    property that decides whether a transparent material is actually transparent: an alpha of 0.2
    on an OPAQUE material renders solid, and every field in every response still reads correctly.

    WHAT MOVES, read off bl_rna on 3.6.23, 4.2.17, 4.4.0 and 5.0.1:

      blend_method              every build ACCEPTS OPAQUE | CLIP | HASHED | BLEND, and from 4.2 on
                                only TWO of them can be stored. It became a shim over
                                surface_render_method there: OPAQUE, CLIP and HASHED all store as
                                HASHED, and only BLEND stores as BLEND. RNA still advertises all
                                four, so the refusal above still lists four. This used to read
                                "same four everywhere", which was true up to 4.0 and has been wrong
                                since. The readback now reports the difference in `clamped` instead
                                of leaving the caller to discover it in a render.
      shadow_method             3.6 and 4.2 ONLY. EEVEE Next dropped it at 4.4, so asking for one
                                on a newer build is asking for a feature the renderer no longer has.
      displacement_method       4.2 and later. Absent on 3.6.
      surface_render_method     4.2 and later.
      use_transparent_shadow    4.2 and later.
      use_raytrace_refraction   4.2 and later.

    ANYTHING THIS BUILD LACKS IS REFUSED, NAMING WHICH BUILDS HAVE IT, and every key is checked
    BEFORE any of them is written so a half-supported request does not leave half of it applied.

    THESE ARE EEVEE'S SETTINGS. Cycles decides transparency from the shader itself and ignores
    blend_method entirely, so the active engine is reported and the keys it will not read are named.

    params:
      material / name (str)      required
      blendMethod (str)          OPAQUE | CLIP | HASHED | BLEND
      shadowMethod (str)         NONE | OPAQUE | CLIP | HASHED - 3.6 and 4.2 only
      backfaceCulling (bool)
      alphaThreshold (float)     the cutoff for CLIP
      screenRefraction (bool)
      displacementMethod (str)   BUMP | DISPLACEMENT | BOTH - 4.2 and later
      surfaceRenderMethod (str)  DITHERED | BLENDED - 4.2 and later
      transparentShadow (bool)   4.2 and later
      raytraceRefraction (bool)  4.2 and later
    """
    reject_unknown(params, _MATSET_KEYS, "set_material_settings")
    want = take(params, "material", "name", required=True, kind=str)
    mat = bpy.data.materials.get(str(want))
    if mat is None:
        have = sorted(m.name for m in bpy.data.materials)[:25]
        raise MifOpError("no material named '%s'. This file has: %s. NOTHING was changed."
                         % (want, ", ".join(have) if have else "(none)"))

    asked = [k for k in _MATSET_MAP if params.get(k) is not None]
    if not asked:
        raise MifOpError("nothing to do - pass at least one of %s. NOTHING was changed."
                         % ", ".join(sorted(_MATSET_MAP)))

    # EVERY KEY CHECKED AGAINST THIS BUILD BEFORE ANY OF THEM IS WRITTEN.
    missing = [(k, _MATSET_MAP[k][0], _MATSET_MAP[k][1]) for k in asked
               if _MATSET_MAP[k][0] not in mat.bl_rna.properties]
    if missing:
        raise MifOpError(
            "this Blender (%s) has no %s. Available on %s. Accepting the key and writing it only "
            "where it exists is how a caller asks for a shadow mode, does not get one, and is told "
            "nothing - so it is refused. NOTHING was changed."
            % (bpy.app.version_string,
               "; ".join("%s (material.%s)" % (k, a) for k, a, _ in missing),
               "; ".join(v for _, _, v in missing)))

    # EVERY VALUE RESOLVED BEFORE ANY OF THEM IS WRITTEN - the second half of the promise the
    # build check above already keeps. Until 2026-09-15 the enum validation lived INSIDE the write
    # loop, so _MATSET_MAP's order decided who got written before a later key was refused:
    #
    #     blendMethod HASHED
    #     set_material_settings{blendMethod: BLEND, displacementMethod: <invalid>}
    #       -> ok:false  "... NOTHING was changed."      and the material was on BLEND
    #
    # blendMethod is first in that map and displacementMethod is judged six keys later. Measured by
    # the second seat on NINE builds - 4.2.0 through 5.2.1 LTS - identical on every one.
    #
    # TWO PASSES, NOT A REORDER. Reordering the map only moves which key gets to be the one that
    # breaks the promise. ops_physics.py records fixing this same shape in set_physics_world on
    # 2026-09-04, and its comment names THIS op as one that "avoids it by validating up front" -
    # true of the version check, and now true of the value check too.
    staged = []
    for key in asked:
        attr = _MATSET_MAP[key][0]
        raw = params[key]
        if key in _MATSET_BOOL:
            value = take_bool(params, key, default=True)
        elif key in _MATSET_FLOAT:
            value = take_float(params, key)
        else:
            # AN ENUM, VALIDATED AGAINST ITS OWN ITEMS so the refusal names what this build offers
            # rather than letting RNA raise a TypeError about types.
            valid = [i.identifier for i in mat.bl_rna.properties[attr].enum_items]
            match = [v for v in valid if v.upper() == str(raw).upper()]
            if not match:
                raise MifOpError("'%s' is not a valid %s on Blender %s. Available: %s. NOTHING was "
                                 "changed." % (raw, key, bpy.app.version_string, ", ".join(valid)))
            value = match[0]
        staged.append((key, attr, value))

    applied = {}
    for key, attr, value in staged:
        setattr(mat, attr, value)
        applied[key] = value

    # READ BACK PER KEY. alpha_threshold clamps rather than refusing, so echoing the request would
    # report a value the material does not have.
    #
    # EVERY KEY IS COMPARED, NOT ONLY THE FLOATS, and until 2026-09-15 only floats were. The guard
    # was `if isinstance(wanted, float)`, so a float that clamped was reported and an ENUM that
    # landed somewhere else was reported nowhere: `applied` showed the stored value, `clamped` was
    # null and `ok` was true.
    #
    # THAT IS NOT HYPOTHETICAL. From 4.2 on, blend_method is a lossy shim over
    # surface_render_method - measured by the second seat on 4.2.0 and 4.4.0:
    #     sent OPAQUE -> stored HASHED      sent CLIP  -> stored HASHED
    #     sent HASHED -> stored HASHED      sent BLEND -> stored BLEND
    # RNA still advertises all four items, so the validation above accepts OPAQUE, writes it, and
    # the material ends up HASHED. Four values in, two values out - and a caller asking for OPAQUE
    # was told it worked and given a dithered transparency mode. That is the exact inversion of the
    # warning in this op's own note, which says blendMethod "is what decides whether a transparent
    # material is actually transparent".
    #
    # `clamped` KEEPS ITS NAME because callers and suites already read it, and it is the field that
    # answers "what did not land". The word is wrong for an enum the build remapped, so each entry
    # now says WHY rather than leaving the reader to infer it from the field name.
    after, wrong = {}, {}
    for key in asked:
        stored = getattr(mat, _MATSET_MAP[key][0])
        after[key] = round(float(stored), 6) if isinstance(stored, float) else stored
        wanted = applied[key]
        if isinstance(wanted, float):
            if abs(float(stored) - wanted) > 1e-4:
                wrong[key] = {"requested": wanted, "stored": after[key],
                              "why": "clamped by the property's own range"}
        elif after[key] != wanted:
            wrong[key] = {"requested": wanted, "stored": after[key],
                          "why": "this build stored a DIFFERENT value - the property is a shim over "
                                 "another one and cannot represent what was asked for. The name "
                                 "'clamped' is inherited; nothing was clamped, it was remapped."}

    engine = bpy.context.scene.render.engine
    is_cycles = "CYCLES" in engine
    # CYCLES DECIDES TRANSPARENCY FROM THE SHADER and ignores blend_method entirely, so these are
    # written and inert under it. Still written, because a scene is often set up under one renderer
    # and rendered with another.
    #
    # EXCEPT displacementMethod, which Cycles renders (docs/22 B66: BUMP against DISPLACEMENT 3.87-4.56 / 255 in Cycles
    # on 4.2.17-5.2.1, and BOTH apart from DISPLACEMENT as well) - it was listed inert here with every other key until
    # 2026-09-28 (the 2026-09-21 pack-rule review), so a caller setting a true displacement for a Cycles render was told
    # the write did nothing.
    inert = sorted(k for k in asked if k not in _MATSET_CYCLES_READS) if is_cycles else []
    return {
        "ok": True,
        "material": mat.name,
        "engine": engine,
        "applied": after,
        "clamped": wrong or None,
        "inertUnderThisEngine": inert or None,
        # BOTH IS DISPLACEMENT IN EEVEE. MEASURED 2026-09-16 on 4.2.17, 4.4.0, 5.0.1 and 5.2.1 (3.6 has no such
        # property): a subdivided plane displaced by a noise texture, 128 px - EEVEE DISPLACEMENT against BOTH 0.0, max
        # 0.0, where BUMP against either is 3.4-4.0 mean; Cycles separates all three (DISPLACEMENT against BOTH
        # 5.85-5.89 on 4.2/4.4, 0.23-0.24 on 5.0/5.2). The second seat saw it first, on 5.2.2 (mailbox id 108).
        "bothNote": ("displacementMethod BOTH renders exactly as DISPLACEMENT in %s - measured pixel-identical on 4.2 "
                     "to 5.2; only Cycles adds the bump on top." % engine)
                    if applied.get("displacementMethod") == "BOTH" and "EEVEE" in engine else None,
        "note": ("%s written and this scene renders with %s, which decides transparency from the "
                 "SHADER and does not read them. They are stored and will apply if the engine "
                 "changes." % (", ".join(inert), engine)) if inert else
                ("blendMethod is what decides whether a transparent material is actually "
                 "transparent - an alpha of 0.2 on an OPAQUE material renders solid, and every "
                 "field in every response still reads correctly."
                 if "blendMethod" in applied else None),
        "blenderVersion": bpy.app.version_string,
    }


# ===========================================================================
# PROCEDURAL SHADER NODES
# ===========================================================================
# WHAT WAS MISSING, and it was the biggest hole left in this file. create_material sets base
# colour, metallic and roughness; set_material_texture loads an IMAGE off disk. So every material
# this addon could build was either a flat plastic colour or something that needed texture files
# shipped beside it. Concrete that looks like concrete, steel that looks worn, a floor with grain -
# all of that is a handful of procedural nodes in Blender and none of it was reachable.
#
# It was not reachable through the GENERIC path either, which is the part worth being precise
# about. add_group_node has addressed material trees since 2026-09-03 ("material:<name>"), so a
# caller who knew that ShaderNodeValToRGB is what everyone else calls a Colour Ramp could add one -
# and then could not put a single stop on it, because color_ramp is a POINTER property and the
# generic property writer refuses those by design. The nodes could be placed and not configured.
#
# WHY A COMPOSITE OP AND NOT MORE add_group_node CALLS. Wiring a procedural pattern into a
# Principled input by hand is five or six calls, and three of them are traps this file already
# knows about and the caller does not:
#
#   * A Mapping node whose Vector is unconnected reads as a CONSTANT ZERO and paints the whole
#     surface one colour. It is not "use the default coordinates" - that is what an unconnected
#     TEXTURE node does, and the difference between those two facts is a flat grey floor.
#   * Rotation on a Mapping node is RADIANS, and a texture turns about the UV plane's normal.
#     Degrees where radians were wanted is the mistake that put a showcase roof at 82 degrees.
#   * A pattern driving ROUGHNESS wants the scalar output and a pattern driving BASE COLOUR wants
#     the colour one, and Blender converts silently either way - so the wrong choice renders
#     plausibly and reads as correct in every field.
#
# set_material_texture is the same shape for images and exists for the same reason. This is its
# procedural twin, and it shares its parameter vocabulary (input, scale, offset, rotation, uvMap,
# replace) so the two do not have to be learned separately.

# THE TYPES THIS OP WILL BUILD, as canonical keys - lower case, punctuation removed. Deliberately
# NOT a second copy of ops_nodes' bl_idname table: the names here declare the SCOPE of this op and
# the actual resolution and version gate still go through resolve_node_type, so an idname can only
# ever be written in one place.
#
# Image textures are absent on purpose: they have set_material_texture, which handles the colour
# space per input, and folding them in here would mean two ops that load images and disagree.
PROCEDURAL_TEXTURE_TYPES = ("noise", "voronoi", "wave", "gradient", "magic", "brick", "checker",
                            "musgrave", "whiteNoise", "gabor")
_PROCEDURAL_KEYS = frozenset("".join(c for c in t.lower() if c.isalnum())
                             for t in PROCEDURAL_TEXTURE_TYPES)

# WHICH OUTPUT TO TAKE, IN ORDER OF PREFERENCE, and this is a real decision rather than a default.
# A procedural texture offers a scalar and a colour, Blender converts between them silently, and
# the two look different: Roughness driven from Color reads the colour's luminance, which is not
# the pattern the node computed. Voronoi is why this is a LIST and not the string "Fac" - it has no
# Fac at all, its scalar output is Distance, and a hardcoded "Fac" would have refused every Voronoi
# in the addon.
# "Factor" ADDED 2026-09-08, and it was found by the version matrix rather than by reading.
# Blender 5.0 renamed the Noise Texture's scalar output from Fac to Factor, so on 5.0 this
# list matched NOTHING and selection fell through to the type-based fallback below - which
# happened to pick the right socket, and reported itself as "this node has no Fac/Distance/
# Value output". Correct by luck: the fallback takes the FIRST value-typed output, and the
# whole reason this list exists is that a node offering several can have the pattern on a
# later one. A preference list that silently stops matching is worse than no list, because
# the result still looks right until the one node where it does not.
_SCALAR_OUTPUTS = ("Fac", "Factor", "Distance", "Value")
_COLOR_OUTPUTS = ("Color",)

# Blender's own limit on a colour ramp - MAXCOLORBAND. Checked before anything is written because
# the write is destructive: the existing stops are cleared down to one before the new ones go on,
# so discovering the limit half way through would leave a ramp that is neither the old one nor the
# one that was asked for.
_MAX_RAMP_STOPS = 32

RAMP_KEYS = {"material", "tree", "group", "node", "stops", "interpolation", "colorMode",
             "hueInterpolation"}

PROCEDURAL_KEYS = {"material", "name", "type", "nodeName", "input", "toNode", "toSocket",
                   "fromSocket", "settings", "properties", "space", "uvMap",
                   "scale", "offset", "rotation", "colorRamp", "relief", "strength",
                   "distance", "replace"}


def _material_or_refuse(name, tail="NOTHING was changed."):
    mat = bpy.data.materials.get(str(name))
    if mat is None:
        have = sorted(m.name for m in bpy.data.materials)[:25]
        raise MifOpError("no material named '%s'. This file has: %s. %s"
                         % (name, ", ".join(have) if have else "(none)", tail))
    return mat


def _check_ramp_stops(raw, tail="NOTHING was changed."):
    """Validate a stops list completely and return [(position, [r,g,b,a])]. Writes NOTHING.

    SEPARATE FROM THE WRITE because the write destroys what is there first - see _MAX_RAMP_STOPS.
    Every position and every colour is checked before the ramp is touched, so a bad fourth stop
    cannot leave a two-stop ramp behind.

    POSITIONS OUTSIDE 0..1 ARE REFUSED RATHER THAN CLAMPED. Blender clamps them silently, so a
    stop asked for at 1.4 lands at 1.0, the ramp reads back as 1.0, and the response agrees with
    the file while disagreeing with the request - which is indistinguishable from having asked for
    1.0 in the first place.
    """
    if not isinstance(raw, (list, tuple)) or not raw:
        raise MifOpError("stops must be a non-empty list of {position, color} objects, got %r. "
                         "A colour ramp always has at least one stop. %s" % (raw, tail))
    if len(raw) > _MAX_RAMP_STOPS:
        raise MifOpError("a Blender colour ramp holds at most %d stops and %d were given. %s"
                         % (_MAX_RAMP_STOPS, len(raw), tail))
    out = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise MifOpError("stop %d must be an object like {\"position\": 0.5, \"color\": "
                             "[0.2,0.2,0.2]}, got %r. %s" % (i, item, tail))
        unknown = [k for k in item if k not in ("position", "pos", "color", "colour")]
        if unknown:
            raise MifOpError("stop %d has unknown key(s) %s - a stop takes position and color. %s"
                             % (i, ", ".join(sorted(unknown)), tail))
        if "position" not in item and "pos" not in item:
            raise MifOpError("stop %d has no position. Every stop needs one, in 0..1. %s"
                             % (i, tail))
        pos = finite_float(item.get("position", item.get("pos")), "stops[%d].position" % i)
        if pos < 0.0 or pos > 1.0:
            raise MifOpError(
                "stops[%d].position is %g, and a colour ramp runs from 0 to 1 - Blender would "
                "CLAMP it silently, so the ramp would read back as %g and agree with itself while "
                "disagreeing with you. %s" % (i, pos, min(max(pos, 0.0), 1.0), tail))
        col = item.get("color", item.get("colour"))
        if col is None:
            raise MifOpError("stop %d has no color. %s" % (i, tail))
        if not isinstance(col, (list, tuple)):
            raise MifOpError("stops[%d].color takes [r,g,b] or [r,g,b,a], got %r. %s"
                             % (i, col, tail))
        vals = finite_floats(list(col), "stops[%d].color" % i)
        if len(vals) == 3:
            # ALPHA DEFAULTS TO OPAQUE, matching _coerce_socket above. A ramp element's colour is
            # RGBA and a missing alpha read as 0 makes every stop transparent, which on a ramp
            # driving roughness looks like nothing happened at all.
            vals = vals + [1.0]
        if len(vals) != 4:
            raise MifOpError("stops[%d].color takes 3 or 4 numbers, got %d. %s"
                             % (i, len(vals), tail))
        out.append((pos, vals))
    return out


def _write_ramp_stops(ramp, stops):
    """Put exactly `stops` on a validated ColorRamp. The caller has already checked every value.

    THE ELEMENTS COLLECTION IS KEPT SORTED BY POSITION, and that is what makes the obvious
    implementation wrong. Writing element[i].position = stops[i].position re-sorts the collection
    on every assignment, so index i stops meaning what it meant one line earlier: with a ramp at
    [0.0, 0.5] and stops at [0.7, 0.9], setting element[0] to 0.7 moves it behind the old 0.5 and
    the next write lands on the element that was just set.

    So the ramp is reduced to ONE element and the rest are appended in ascending order, each one
    coloured through the reference elements.new() returns, before the next insert can move
    anything. Ascending inserts always land at the end, so no reordering happens at all.

    One element is the floor, not zero: elements.remove() raises on the last one - a colour ramp
    cannot be empty.
    """
    ordered = sorted(stops, key=lambda s: s[0])
    while len(ramp.elements) > 1:
        ramp.elements.remove(ramp.elements[len(ramp.elements) - 1])
    first = ramp.elements[0]
    first.position = ordered[0][0]
    first.color = ordered[0][1]
    for pos, col in ordered[1:]:
        el = ramp.elements.new(pos)
        if el is None:
            raise MifOpError(
                "Blender would not add another stop at %g and returned nothing. The ramp now holds "
                "%d of the %d stops asked for - it is neither what it was nor what you asked for, "
                "and needs setting again." % (pos, len(ramp.elements), len(ordered)))
        el.color = col


# ---------------------------------------------------------------------------
# A VOLUME SHADER - smoke, fire, fog. MEASURED 2026-09-17 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1 in plain bpy (docs/22
# B80). A GAS fluid domain - resolution 32, a sphere INFLOW, frames 1-24 stepped, black world, one sun, 96 px - renders
# as a SOLID LIT BOX in EEVEE (mean 61-65 of 255) and as NOTHING in Cycles while it has no material, which is the state
# add_fluid leaves it in and how the second seat measured "frame 2 vs 40: 0.000". A Principled Volume on the output's
# Volume socket, attribute inputs at their defaults, renders the SIMULATED smoke on every build in both engines (EEVEE
# 8.1-10.1, Cycles 4.9-6.8, the top half brighter where it rises), and the same material on a domain with no flow renders
# nothing. Two ways it still does not show, both measured on every build: a surface shader left on Surface
# (create_material's Principled BSDF) renders the box in front of the smoke in both engines, and a cleared density
# attribute fills the whole box uniformly in EEVEE and renders nothing in Cycles. The node's type (PRINCIPLED_VOLUME),
# its thirteen inputs and their defaults are identical on all five builds.
_VOLUME_INPUTS = {
    "color": ("Color", "color"),
    "density": ("Density", "min0"),
    "anisotropy": ("Anisotropy", "signed1"),
    "absorptionColor": ("Absorption Color", "color"),
    "emissionStrength": ("Emission Strength", "min0"),
    "emissionColor": ("Emission Color", "color"),
    "blackbodyIntensity": ("Blackbody Intensity", "min0"),
    "blackbodyTint": ("Blackbody Tint", "color"),
    "temperature": ("Temperature", "min0"),
    "densityAttribute": ("Density Attribute", "string"),
    "colorAttribute": ("Color Attribute", "string"),
    "temperatureAttribute": ("Temperature Attribute", "string"),
}
# A LITERAL, NOT ("material", ...) + tuple(_VOLUME_INPUTS): parity_check and describe_op read accepted keys statically,
# and a computed set is one neither can read - describe_op answered acceptedKeys null for export_mesh for exactly that.
# A key added to _VOLUME_INPUTS and not here is refused, which test_blender_material_volume V101 would catch.
_VOLUME_KEYS = {"material", "name", "surface", "remove",
                "color", "density", "anisotropy", "absorptionColor", "emissionStrength", "emissionColor",
                "blackbodyIntensity", "blackbodyTint", "temperature",
                "densityAttribute", "colorAttribute", "temperatureAttribute"}


def _stage_volume_value(key, value):
    """One Principled Volume value validated and shaped for its socket. Writes NOTHING."""
    kind = _VOLUME_INPUTS[key][1]
    if kind == "string":
        if not isinstance(value, str):
            raise MifOpError("'%s' takes an attribute NAME as a string (\"\" for none), got %r. NOTHING was changed."
                             % (key, value))
        return value
    if kind == "color":
        if not isinstance(value, (list, tuple)):
            raise MifOpError("'%s' takes a colour as [r,g,b] or [r,g,b,a], got %r. NOTHING was changed."
                             % (key, value))
        rgba = finite_floats(value, key)
        if len(rgba) == 3:
            rgba.append(1.0)
        if len(rgba) != 4:
            raise MifOpError("'%s' takes 3 or 4 components, got %d. NOTHING was changed." % (key, len(rgba)))
        return rgba
    number = finite_float(value, key)
    if kind == "min0" and number < 0.0:
        raise MifOpError("'%s' cannot be negative, got %s. NOTHING was changed." % (key, number))
    if kind == "signed1" and not -1.0 <= number <= 1.0:
        raise MifOpError("'%s' runs from -1 (scatters back) to 1 (scatters forward), got %s. NOTHING was changed."
                         % (key, number))
    return number


def _active_material_output(tree):
    """(the Material Output that renders - the active one, else the first - or None, how many outputs there are)."""
    outs = [n for n in tree.nodes if n.type == "OUTPUT_MATERIAL"]
    active = [n for n in outs if getattr(n, "is_active_output", False)]
    return (active or outs or [None])[0], len(outs)


def _gas_domains_using(mat):
    """Objects holding this material in a slot that carry a fluid GAS domain - the case a volume shader exists for."""
    names = []
    for ob in bpy.data.objects:
        if not any(slot.material == mat for slot in getattr(ob, "material_slots", ())):
            continue
        for md in ob.modifiers:
            if (md.type == "FLUID" and md.fluid_type == "DOMAIN"
                    and getattr(md.domain_settings, "domain_type", None) == "GAS"):
                names.append(ob.name)
                break
    return sorted(names)


def _volume_readback(node):
    """Every Principled Volume value this op writes, read off the node - or the node driving an input that is linked."""
    out = {}
    for key, (socket_name, kind) in _VOLUME_INPUTS.items():
        sock = node.inputs.get(socket_name)
        if sock is None:
            continue
        if sock.is_linked:
            out[key] = {"linkedFrom": sock.links[0].from_node.name if sock.links else None}
        elif kind == "string":
            out[key] = sock.default_value
        elif kind == "color":
            out[key] = rnd(sock.default_value)
        else:
            out[key] = round(float(sock.default_value), 6)
    return out


def op_set_material_volume(params):
    """Give a material a VOLUME shader - smoke, fire, fog - as a Principled Volume on its output's Volume socket.

    THE HALF OF A MATERIAL NOTHING COULD WRITE. create_material, set_material_properties and add_procedural_texture all
    work on the Principled BSDF on Surface, so add_fluid could build a smoke or fire simulation and nothing could make it
    appear - the second seat measured frame 2 against frame 40 of a GAS domain at 0.000 of 255 apart. Measured on all
    five builds (the block above _VOLUME_INPUTS, docs/22 B80): a domain with no material renders as a solid box in EEVEE
    and as nothing in Cycles, and a Principled Volume renders the simulated smoke in both.

    ONE NODE, FOUND OR MADE. A Principled Volume already on the output's Volume socket is edited in place; with nothing
    there one is created and linked. ANY OTHER node on that socket - Volume Scatter, a Mix Shader, a group - is REFUSED,
    naming it: overwriting a hand-built volume graph is not what a settings call is for.

    `surface` IS EXPLICIT AND DEFAULTS TO keep. A surface shader renders in front of the volume inside the mesh -
    create_material's Principled BSDF left on Surface hid the smoke completely in EEVEE and in Cycles on every build.
    surface:'disconnect' unlinks Surface and leaves its nodes in the tree. It is not done for you, because a surface
    meant to be seen - glass around a volume - is a real material; surfaceNote says whenever a surface is still linked.

    NOTHING IS CREATED OR WRITTEN UNTIL EVERY VALUE VALIDATES, the rule create_material follows. `remove` unlinks the
    Principled Volume from the output, and deletes it when nothing else reads it.

      material (or name)       required
      density                  >= 0 (Blender's default is 1)
      color, absorptionColor, emissionColor, blackbodyTint     [r,g,b] or [r,g,b,a]
      anisotropy               -1 to 1
      emissionStrength, blackbodyIntensity, temperature        >= 0; temperature is in kelvin
      densityAttribute         the grid density is read from. Left out it is left as it is - a new Principled Volume
                               starts at Blender's 'density', which is what a smoke domain writes. "" makes the
                               density uniform, measured as a FILLED BOX in EEVEE and NOTHING in
                               Cycles on a domain
      colorAttribute, temperatureAttribute
      surface                  keep | disconnect
      remove                   true unlinks the volume shader from the output and deletes the node unless another
                               node reads it (keptBecauseReadBy); it takes no other setting
    """
    reject_unknown(params, _VOLUME_KEYS, "set_material_volume")
    mat_name = take(params, "material", "name", required=True, kind=str)
    remove = take_bool(params, "remove", default=False)
    surface = take(params, "surface", kind=str)
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        have = sorted(m.name for m in bpy.data.materials)[:25]
        raise MifOpError("no material named '%s'. This file has: %s. NOTHING was changed."
                         % (mat_name, ", ".join(have) or "(none)"))
    if not mat.use_nodes or mat.node_tree is None:
        raise MifOpError("material '%s' has no node tree (use_nodes is off), so it has no output to put a volume on. "
                         "create_material with reuse:true switches nodes on. NOTHING was changed." % mat.name)
    mode = None
    if surface is not None:
        mode = surface.strip().upper()
        if mode not in ("KEEP", "DISCONNECT"):
            raise MifOpError("surface takes 'keep' or 'disconnect', got %r. NOTHING was changed." % surface)
    staged = {key: _stage_volume_value(key, params[key])
              for key in _VOLUME_INPUTS if key in params and params[key] is not None}

    tree = mat.node_tree
    output, n_outputs = _active_material_output(tree)
    if output is None or output.inputs.get("Volume") is None:
        raise MifOpError("material '%s' has no Material Output with a Volume socket, so nothing it holds can render as "
                         "a volume. NOTHING was changed." % mat.name)
    vol_socket = output.inputs["Volume"]
    feeding = vol_socket.links[0].from_node if vol_socket.is_linked and vol_socket.links else None
    if feeding is not None and feeding.type != "PRINCIPLED_VOLUME":
        raise MifOpError("material '%s' already has '%s' (%s) on its Volume output - a volume graph this op did not "
                         "build. Edit that node directly, or disconnect it first. NOTHING was changed."
                         % (mat.name, feeding.name, feeding.bl_idname))

    if remove:
        if staged or mode is not None:
            raise MifOpError("remove:true takes no other settings (got %s). NOTHING was changed."
                             % ", ".join(sorted(list(staged) + (["surface"] if mode is not None else []))))
        if feeding is None:
            raise MifOpError("material '%s' has no volume shader on its output, so there is nothing to remove. "
                             "NOTHING was changed." % mat.name)
        before = _volume_readback(feeding)
        removed = feeding.name
        readers = sorted({link.to_node.name for sock in feeding.outputs for link in sock.links
                          if link.to_node != output})
        for link in list(vol_socket.links):
            tree.links.remove(link)
        if not readers:
            tree.nodes.remove(feeding)
        out = _material_json(mat)
        out.update({"removed": removed, "deleted": not readers, "keptBecauseReadBy": readers or None,
                    "before": before, "volumeLinked": bool(vol_socket.is_linked)})
        return out

    # A LINKED INPUT IGNORES ITS DEFAULT (see _check_unlinked), so it is refused before anything is written.
    if feeding is not None:
        for key in staged:
            _check_unlinked(feeding.inputs[_VOLUME_INPUTS[key][0]], key)
    before = _volume_readback(feeding) if feeding is not None else None

    created = feeding is None
    node = feeding
    if created:
        node = tree.nodes.new("ShaderNodeVolumePrincipled")
        node.location = (output.location.x - 300.0, output.location.y - 260.0)
        tree.links.new(node.outputs["Volume"], vol_socket)
    for key, value in staged.items():
        node.inputs[_VOLUME_INPUTS[key][0]].default_value = value

    surface_socket = output.inputs.get("Surface")
    disconnected = []
    if mode == "DISCONNECT" and surface_socket is not None:
        for link in list(surface_socket.links):
            disconnected.append(link.from_node.name)
            tree.links.remove(link)
    surface_from = ([link.from_node.name for link in surface_socket.links]
                    if surface_socket is not None and surface_socket.is_linked else [])
    gas = _gas_domains_using(mat)
    after = _volume_readback(node)

    out = _material_json(mat)
    out.update({
        "volumeNode": node.name,
        "createdVolumeNode": created,
        "applied": sorted(staged),
        "before": before,
        "volume": after,
        "volumeLinked": bool(vol_socket.is_linked),
        "surfaceLinked": bool(surface_from),
        "surfaceShaders": surface_from or None,
        "surfaceDisconnected": disconnected or None,
        "gasDomains": gas or None,
    })
    if surface_from:
        out["surfaceNote"] = (
            "Surface is still fed by %s: a surface shader renders in front of the volume inside the mesh. Measured on "
            "every build, a smoke domain that kept create_material's Principled BSDF on Surface rendered as the solid "
            "box in EEVEE and in Cycles, the smoke hidden. For smoke, fire or fog pass surface:'disconnect'; keep the "
            "surface only when it is meant to be seen, such as glass around the volume."
            % ", ".join("'%s'" % n for n in surface_from))
    if gas and after.get("densityAttribute") == "":
        out["densityAttributeNote"] = (
            "densityAttribute is empty, so density is uniform instead of read from the simulation. On a GAS domain "
            "(%s here) that measured as a FILLED BOX in EEVEE and as NOTHING in Cycles on every build; a smoke domain "
            "writes its density to 'density'." % ", ".join(gas))
    if created is False and not staged and mode is None:
        out["note"] = ("nothing was passed and '%s' already had a Principled Volume on its output - NOTHING was "
                       "changed." % mat.name)
    if n_outputs > 1:
        out["outputNote"] = ("this material has %d Material Output nodes; the volume is on the ACTIVE one, '%s', which "
                             "is the one that renders." % (n_outputs, output.name))
    return out


def op_set_color_ramp(params):
    """Set the stops of a ColorRamp node - the one thing the generic node writer cannot do.

    WHY THIS IS ITS OWN OP. add_group_node writes node properties by their RNA name, dispatched on
    the property's declared type, and REFUSES pointers - correctly, because a datablock named as a
    string is stored and ignored. A colour ramp is a pointer to a ColorRamp holding a COLLECTION of
    elements, so it falls outside that writer twice over. The result was that a ramp could be added
    to a material and never given a stop: two black-to-white defaults and no way to change them,
    which makes the node an expensive no-op. This is the missing write half.

    A RAMP IS HOW A PROCEDURAL PATTERN BECOMES A MATERIAL. Noise on its own is grey mush; noise
    through a ramp with three stops is concrete, rust or moss depending on where the stops sit.

    params:
      material (str)            address a material's tree by name, the common case
      tree / group (str)        OR any other tree - a node group, "scene:compositor",
                                "scene:world", "material:<name>", "world:<name>", "light:<name>"
      node (str)     required   the node holding the ramp
      stops (list)              [{position: 0..1, color: [r,g,b] or [r,g,b,a]}, ...]
      interpolation (str)       EASE | CARDINAL | LINEAR | B_SPLINE | CONSTANT
      colorMode (str)           RGB | HSV | HSL
      hueInterpolation (str)    NEAR | FAR | CW | CCW - written whatever the mode, and Blender reads it only while
                                colorMode is HSV or HSL

    `material` IS NOT AN ALIAS OF `tree`, which is why they are separate parameters rather than two
    spellings taken by one take() call: one is a material NAME and the other is a tree ADDRESS, and
    "Concrete" means different things to them. Passing both is refused instead of quietly
    preferring one.
    """
    reject_unknown(params, RAMP_KEYS, "set_color_ramp")
    mat_name = take(params, "material", default=None, kind=str)
    tree_name = take(params, "tree", "group", default=None, kind=str)
    if (mat_name is None) == (tree_name is None):
        raise MifOpError("pass `material` (a material name) OR `tree` (a tree address such as "
                         "'material:Concrete', 'scene:world' or a node group name), not both and "
                         "not neither. NOTHING was changed.")
    if mat_name is not None:
        _material_or_refuse(mat_name)
        tree = _tree("material:%s" % mat_name)
    else:
        tree = _tree(tree_name)

    node_name = str(take(params, "node", required=True, kind=str))
    node = tree.nodes.get(node_name)
    if node is None:
        raise MifOpError("no node named '%s' in '%s'. It holds: %s. NOTHING was changed."
                         % (node_name, tree.name,
                            ", ".join(n.name for n in tree.nodes) or "(nothing)"))
    ramp = getattr(node, "color_ramp", None)
    if ramp is None:
        # ASKED BY ATTRIBUTE, not by bl_idname, so this keeps working for the compositor's own
        # ValToRGB and for any other node that carries a ramp.
        ramps = sorted({n.name for n in tree.nodes if getattr(n, "color_ramp", None) is not None})
        raise MifOpError(
            "node '%s' is a %s and has no colour ramp on it. The nodes in '%s' that do: %s. "
            "add_group_node with type 'colorRamp' makes one. NOTHING was changed."
            % (node.name, node.bl_idname, tree.name, ", ".join(ramps) or "(none)"))

    # EVERYTHING VALIDATED BEFORE THE FIRST WRITE, and here that matters more than usual: the
    # stop write clears the ramp down to one element before rebuilding it, so a refusal in the
    # middle would leave a ramp that is neither the old one nor the new one.
    raw_stops = params.get("stops")
    stops = _check_ramp_stops(raw_stops) if raw_stops is not None else None
    interp = take(params, "interpolation", default=None, kind=str)
    mode = take(params, "colorMode", default=None, kind=str)
    hue = take(params, "hueInterpolation", default=None, kind=str)
    if stops is None and interp is None and mode is None and hue is None:
        raise MifOpError("nothing to set - pass stops, interpolation, colorMode or "
                         "hueInterpolation. NOTHING was changed.")

    # EVERY ENUM RESOLVED BEFORE THE FIRST IS WRITTEN. Three enums written in sequence means a bad
    # third one refuses with "NOTHING was changed" over two that were already applied.
    staged = []
    if mode is not None:
        staged.append(("color_mode", _enum_value_or_refuse(ramp, "color_mode", str(mode),
                                                           "colorMode")))
    if hue is not None:
        staged.append(("hue_interpolation",
                       _enum_value_or_refuse(ramp, "hue_interpolation", str(hue),
                                             "hueInterpolation")))
    if interp is not None:
        staged.append(("interpolation", _enum_value_or_refuse(ramp, "interpolation", str(interp),
                                                              "interpolation")))

    before = color_ramp_stops(node)
    for attr, value in staged:
        setattr(ramp, attr, value)
    # THE STOPS GO LAST because that write is the destructive one - see _write_ramp_stops.
    if stops is not None:
        _write_ramp_stops(ramp, stops)

    after = color_ramp_stops(node)
    out = {
        "ok": True,
        "tree": tree.name,
        "node": node.name,
        "nodeType": node.bl_idname,
        # READ BACK OFF THE RAMP, not echoed from the request. Positions come back in Blender's
        # order, which is sorted by position whatever order they were sent in - so a caller who
        # sent them out of order can see which stop is now which index.
        "colorRamp": after,
        "before": before,
        "stopCountBefore": (before or {}).get("stopCount"),
        "stopsReordered": (stops is not None
                           and [round(p, 6) for p, _ in stops]
                           != [s["position"] for s in after["stops"]]),
    }
    if out["stopsReordered"]:
        out["orderNote"] = ("Blender keeps a colour ramp's elements SORTED BY POSITION, so the "
                            "stops came back in a different order than they were sent. The ramp is "
                            "correct; the indices are Blender's.")
    if node.mute:
        out["mutedNote"] = ("this node is MUTED, so the ramp is set and Blender passes its input "
                            "straight through. Nothing about the ramp says so.")
    if not any(s.is_linked for s in node.outputs):
        out["unlinkedNote"] = ("this ramp's outputs are not connected to anything, so it is "
                               "configured and unused - which reads identically to a working one "
                               "in every other field.")
    return out


def _procedural_type_or_refuse(asked):
    """A procedural texture name or bl_idname -> the bl_idname to create, version-checked."""
    key = "".join(c for c in str(asked).lower() if c.isalnum())
    if key.startswith("shadernodetex"):
        key = key[len("shadernodetex"):]
    if key not in _PROCEDURAL_KEYS:
        raise MifOpError(
            "'%s' is not a procedural texture this op builds. Available: %s. An IMAGE texture is "
            "set_material_texture's job, because the colour space has to be chosen per input; any "
            "other node type is add_group_node with group 'material:<name>'. NOTHING was changed."
            % (asked, ", ".join(PROCEDURAL_TEXTURE_TYPES)))
    # THE VERSION GATE IS ops_nodes', not a second one here. Musgrave is the case it exists for -
    # it is in 3.6's shader menu and in none of 4.2's, 4.4's or 5.0's, having been folded into the
    # Noise texture - and the refusal names the running version and reads the replacement's real
    # properties off its own RNA rather than quoting a changelog.
    return resolve_node_type(key)


def _pick_output(node, want_scalar, explicit, tail):
    """Which of a texture node's outputs to take, and why. Refuses naming the real ones."""
    outs = [s for s in node.outputs if getattr(s, "enabled", True)]
    if explicit:
        match = [s for s in outs if s.name == explicit]
        if not match:
            raise MifOpError("node '%s' (%s) has no output '%s'. It has: %s. %s"
                             % (node.name, node.bl_idname, explicit,
                                ", ".join(s.name for s in outs), tail))
        return match[0], "asked for by name"
    order = _SCALAR_OUTPUTS if want_scalar else _COLOR_OUTPUTS
    for want in order:
        for sock in outs:
            if sock.name == want:
                return sock, ("the destination takes a single value, so the pattern's scalar "
                              "output is used rather than its colour"
                              if want_scalar else
                              "the destination takes a colour, so the pattern's colour output "
                              "is used")
    typed = [s for s in outs if (s.type == "VALUE") == want_scalar]
    if typed:
        return typed[0], "the first output of the right kind - this node has no %s output" % (
            "/".join(order))
    if outs:
        return outs[0], ("the only outputs on this node are the wrong kind for the destination, so "
                         "Blender will convert - check the result")
    raise MifOpError("node '%s' has no outputs to take. %s" % (node.name, tail))


def _principled_or_none(tree):
    """The Principled BSDF if there is one, used only to place new nodes near it. Never refuses."""
    for node in tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    for node in tree.nodes:
        if node.type == "OUTPUT_MATERIAL":
            return node
    return None


# A UV MAP NODE NAMING A LAYER THE MESH DOES NOT HAVE RENDERS A FALLBACK, not the active layer. MEASURED 2026-09-16 on
# 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1: a plane carrying UVMap and a rotated Rot layer, a checker driven through the node,
# 64 px EEVEE - an empty name renders exactly as UVMap (0.0), "Missing1" and "Missing2" render identically to each other
# (0.0) and 26.0-30.4/255 away from UVMap and from Rot alike (scratchpad measure_uvmap_missing.py). Both texture ops
# accepted any name with ok:true and no word (second seat, mailbox id 140). A material has no single mesh, so the check is
# against every mesh that uses it; with none yet, the reply says the name could not be checked.
def _uv_map_note(mat, uv_map):
    """A sentence when no mesh using `mat` has a UV layer named `uv_map`, else None."""
    if not uv_map:
        return None
    users = [ob for ob in bpy.data.objects
             if ob.type == "MESH" and ob.data is not None and any(s.material == mat for s in ob.material_slots)]
    if not users:
        return ("no mesh uses this material yet, so uvMap '%s' could not be checked. A UV Map node naming a layer the mesh "
                "lacks renders a fixed fallback, not the active layer - measured on every build." % uv_map)
    lacking = sorted(ob.name for ob in users if ob.data.uv_layers.get(str(uv_map)) is None)
    if not lacking:
        return None
    have = sorted({layer.name for ob in users for layer in ob.data.uv_layers})
    return ("uvMap '%s' is not a UV layer on %s, which use this material, so those render a fixed fallback rather than the "
            "active layer (measured on every build). Layers present: %s."
            % (uv_map, ", ".join(lacking[:10]) + (" ..." if len(lacking) > 10 else ""), ", ".join(have[:20]) or "none"))


# THE BUMP NODE'S DEFAULT DISTANCE CHANGED UNDER THIS OP. MEASURED 2026-09-16: ShaderNodeBump's Distance defaults to 1.0 on
# 3.6.23, 4.2.17 and 4.4.0 and to 0.001 on 5.0.1 and 5.2.1. A noise bump on a 1 BU sphere at strength 1.0 against strength
# 0: at the build default, EEVEE 26.9-28.6/255 and Cycles 28.9-29.4 on 3.6-4.4, but 0.097 and 0.099 on 5.0/5.2 (max 2-3);
# at an explicit 0.3, 16.9-18.0 and 15.5-16.9 on every build - so the socket means the same thing everywhere and only
# the default moved (scratchpad measure_bump_default_distance.py). This op left Distance unset, so one call rendered
# relief on 4.4 and a flat surface on 5.x; the second seat met the flat half on 5.2.2 (mailbox id 66). When no distance
# is given it now writes the long-standing 1.0, and the reply says so where the build's own default differs.
_BUMP_DISTANCE_WHEN_UNSET = 1.0


def op_add_procedural_texture(params):
    """Build a procedural pattern into a material and wire it somewhere that reads it.

    This is set_material_texture for patterns rather than files: same vocabulary, same all-or-
    nothing rollback, same insistence that the link is MEASURED off the socket afterwards rather
    than assumed from the call.

    params:
      material / name (str)  required   the material
      type (str)             required   noise | voronoi | wave | gradient | magic | brick |
                                        checker | musgrave | whiteNoise | gabor
      nodeName (str)                    name the texture node so later calls can address it
      settings (dict)                   the node's own input sockets: {"Scale": 12, "Detail": 6}
      properties (dict)                 the node's own RNA properties: {"voronoi_dimensions":"3D"}

      input (str)                       a Principled input to drive, by ANY version's spelling -
                                        baseColor / "Base Color", roughness, metallic, emissive /
                                        "Emission Color" / "Emission", alpha, ... plus 'normal'
      toNode + toSocket (str)           OR any node's input socket in the same tree, which is how
                                        two patterns get combined through a Mix or Math node
      fromSocket (str)                  which texture output to take. Chosen from the destination
                                        when omitted - see _pick_output
      replace (bool)                    replace whatever already drives the destination. Default
                                        true; false refuses rather than stacking

      space (str)                       Generated | Object | UV | Normal | Camera | Window |
                                        Reflect - the Texture Coordinate output to drive it from
      uvMap (str)                       a named UV layer instead, through a UV Map node
      scale / offset                    a number or [u, v], through a Mapping node
      rotation (float)                  DEGREES about the texture plane's normal

      colorRamp (list)                  stops to put between the pattern and the destination
      relief (str)                      bump | normalMap, only with input:'normal'
      strength (float)                  relief strength
      distance (float)                  bump distance - 1.0 when omitted, on every build (5.0 lowered
                                        the node's own default to 0.001, which renders almost no relief)

    THREE THINGS THIS DOES THAT A HAND-BUILT CHAIN GETS WRONG.

    A Mapping node is only created when there is something to map, and it is NEVER left with its
    Vector unconnected. An unconnected Mapping Vector is a constant zero and paints the surface one
    flat colour - it is not "the default coordinates", which is what an unconnected TEXTURE node
    does. Those two facts sit one node apart and produce a grey floor.

    Rotation is DEGREES here and radians on the socket, converted at the boundary rather than
    passed through, which is the same lesson the showcase learned from a roof at 82 degrees.

    'normal' goes through a BUMP node, not a Normal Map node, and that is the opposite of what
    set_material_texture does with an image - deliberately. A Normal Map node reads tangent-space
    RGB, where the blue channel is the axis pointing out of the surface; a procedural pattern is a
    HEIGHT field, and feeding a height into a Normal Map node reads its red channel as an X normal
    and produces lighting that is wrong in a way no field reports. relief:'normalMap' is available
    for a pattern that genuinely is tangent-space RGB.
    """
    reject_unknown(params, PROCEDURAL_KEYS, "add_procedural_texture")
    mat_name = take(params, "material", "name", required=True, kind=str)
    asked_type = take(params, "type", required=True, kind=str)

    # ---- EVERYTHING THAT CAN REFUSE, BEFORE ANYTHING IS CREATED --------------------------------
    ntype = _procedural_type_or_refuse(asked_type)
    mat = _material_or_refuse(mat_name)
    if not mat.use_nodes or mat.node_tree is None:
        raise MifOpError(
            "material '%s' has use_nodes OFF, so its node tree is ignored entirely and its flat "
            "colour is what renders - a pattern added to it would be something nothing looks at. "
            "create_material makes a material with nodes on. NOTHING was changed." % mat.name)
    tree = mat.node_tree

    which = take(params, "input", default=None, kind=str)
    to_node_name = take(params, "toNode", default=None, kind=str)
    to_socket_name = take(params, "toSocket", default=None, kind=str)
    if which is not None and (to_node_name is not None or to_socket_name is not None):
        raise MifOpError("pass `input` (a Principled input) OR `toNode`+`toSocket` (any node's "
                         "socket), not both. NOTHING was changed.")
    if to_node_name is not None and to_socket_name is None:
        raise MifOpError("`toNode` needs `toSocket` - a node has several inputs and this op will "
                         "not guess which one. NOTHING was changed.")
    if to_socket_name is not None and to_node_name is None:
        raise MifOpError("`toSocket` needs `toNode` to say which node it is on. NOTHING was "
                         "changed.")

    relief = take(params, "relief", default=None, kind=str)
    strength = take_float(params, "strength", default=None)
    distance = take_float(params, "distance", default=None)
    has_dest = which is not None or to_node_name is not None
    if not has_dest:
        # A LOOSE NODE IS A LEGITIMATE REQUEST - it is how a graph gets built up before it is
        # wired - but the parameters that only describe a CONNECTION are refused rather than
        # accepted and dropped. Accepting relief here would mean building a Bump node attached to
        # nothing and reporting it as applied.
        stray = [k for k in ("relief", "strength", "distance", "fromSocket", "replace")
                 if params.get(k) is not None]
        if stray:
            raise MifOpError(
                "%s only describe(s) how the pattern is CONNECTED, and no destination was given - "
                "pass `input` or `toNode`+`toSocket`, or drop %s. NOTHING was changed."
                % (", ".join(stray), "them" if len(stray) > 1 else "it"))

    bsdf = None
    target = None
    target_desc = None
    if which is not None:
        bsdf = _principled(mat)
        if str(which).strip().lower() == _NORMAL_INPUT:
            which = _NORMAL_INPUT
            target = bsdf.inputs.get("Normal")
            if target is None:
                raise MifOpError("this Principled BSDF has no Normal input on Blender %s. NOTHING "
                                 "was changed." % bpy.app.version_string)
        else:
            target = principled_socket(bsdf, str(which))
            if target is None:
                raise MifOpError(
                    "'%s' is not a Principled input on Blender %s. This build's inputs are: %s. "
                    "This op also accepts the property names %s and the spellings older or newer "
                    "Blenders use for them. NOTHING was changed."
                    % (which, bpy.app.version_string,
                       ", ".join(s.name for s in bsdf.inputs),
                       ", ".join(sorted(PRINCIPLED_ALIASES))))
        target_desc = "%s.%s" % (bsdf.name, target.name)
    elif to_node_name is not None:
        dest_node = tree.nodes.get(str(to_node_name))
        if dest_node is None:
            raise MifOpError("no node named '%s' in material '%s'. It holds: %s. NOTHING was "
                             "changed." % (to_node_name, mat.name,
                                           ", ".join(n.name for n in tree.nodes) or "(nothing)"))
        live, inactive = usable_sockets(dest_node.inputs, str(to_socket_name))
        if not live and inactive:
            raise MifOpError(
                "'%s' on node '%s' (%s) is INACTIVE at that node's current settings - Blender "
                "carries one socket per data type under the same name and reads only the active "
                "one. Set the node's data type first with add_group_node's `properties`. NOTHING "
                "was changed." % (to_socket_name, dest_node.name, dest_node.bl_idname))
        if not live:
            raise MifOpError("node '%s' (%s) has no input '%s'. It has: %s. NOTHING was changed."
                             % (dest_node.name, dest_node.bl_idname, to_socket_name,
                                ", ".join(s.name for s in dest_node.inputs) or "(none)"))
        target = live[0]
        target_desc = "%s.%s" % (dest_node.name, target.name)

    if relief is not None:
        relief = str(relief).strip().lower().replace("_", "").replace(" ", "")
        if relief not in ("bump", "normalmap"):
            raise MifOpError("relief takes 'bump' or 'normalMap', got %r. NOTHING was changed."
                             % take(params, "relief"))
        if which != _NORMAL_INPUT:
            raise MifOpError(
                "relief only applies to input:'normal' - a Bump or Normal Map node outputs a "
                "NORMAL VECTOR, and wiring one into %s would be meaningless. Drop relief, or "
                "target the normal. NOTHING was changed."
                % (target_desc or "a colour or value input"))
    elif which == _NORMAL_INPUT:
        # THE DEFAULT IS BUMP, and the docstring says why at length: a procedural pattern is a
        # height field, not tangent-space RGB.
        relief = "bump"
    if strength is not None and which != _NORMAL_INPUT:
        raise MifOpError("strength is the Bump or Normal Map node's strength and only applies to "
                         "input:'normal'. NOTHING was changed.")
    if distance is not None and relief != "bump":
        # TWO DIFFERENT MISTAKES, and one message for both would misdescribe one of them. Asking
        # for a distance on a Normal Map node is asking for a socket that node does not have;
        # asking for one on a roughness input is asking for a node that is not being built at all.
        raise MifOpError(
            "distance is the BUMP node's distance, and %s. NOTHING was changed."
            % ("this call is building a Normal Map node instead, which has no such socket"
               if relief == "normalmap" else
               "no relief node is being built here - distance only applies to input:'normal'"))

    replace = take_bool(params, "replace", default=True)
    if target is not None and target.is_linked and not replace:
        raise MifOpError("%s is already driven by '%s' and replace:false was given. NOTHING was "
                         "changed." % (target_desc, target.links[0].from_node.name))

    # Mapping and coordinate parameters, all parsed before any node exists.
    scale_uv, scale_3d = _procedural_vec(params, "scale", 1.0)
    offset_uv, _offset_3d = _procedural_vec(params, "offset", 0.0)
    rot_deg = take(params, "rotation", default=None)
    if rot_deg is not None:
        rot_deg = finite_float(rot_deg, "rotation")
    if scale_uv is not None and (scale_uv[0] == 0.0 or scale_uv[1] == 0.0 or scale_uv[2] == 0.0):
        raise MifOpError(
            "scale 0 collapses the pattern to one point's value across the whole surface, which "
            "looks like a flat fill rather than an error. Pass a non-zero scale. NOTHING was "
            "changed.")
    space = take(params, "space", default=None, kind=str)
    uv_map = take(params, "uvMap", default=None, kind=str)
    if space is not None and uv_map is not None:
        raise MifOpError("pass `space` (a Texture Coordinate output) or `uvMap` (a named UV "
                         "layer), not both - they are two answers to the same question. NOTHING "
                         "was changed.")

    ramp_stops = None
    if params.get("colorRamp") is not None:
        ramp_stops = _check_ramp_stops(params.get("colorRamp"))

    settings = params.get("settings")
    if settings is not None and not isinstance(settings, dict):
        raise MifOpError("`settings` must be a {socketName: value} object, got %r. NOTHING was "
                         "changed." % (settings,))
    props = params.get("properties")
    if props is not None and not isinstance(props, dict):
        raise MifOpError("`properties` must be a {name: value} object, got %r. NOTHING was "
                         "changed." % (props,))

    # ---- FROM HERE ON NODES EXIST, AND EVERY REFUSAL REMOVES THEM AGAIN -----------------------
    # The same structural undo as set_material_texture: made nodes are tracked as they are created
    # and torn down in one place, rather than each refusal path remembering what it has to clean.
    made = []
    node_tail = "Every node this call made was removed again - NOTHING was changed."
    anchor = bsdf if bsdf is not None else _principled_or_none(tree)
    base_x = (anchor.location[0] if anchor is not None else 0.0) - 700.0
    base_y = (anchor.location[1] if anchor is not None else 0.0) - 300.0
    try:
        tex = tree.nodes.new(ntype)
        made.append(tex)
        if tex.bl_idname != ntype:
            raise MifOpError(
                "Blender %s accepted '%s' and produced a '%s' - an UNDEFINED node, which has no "
                "sockets and evaluates to nothing. %s"
                % (bpy.app.version_string, ntype, tex.bl_idname, node_tail))
        tex.location = (base_x, base_y)
        node_name = take(params, "nodeName", default=None, kind=str)
        if node_name:
            tex.name = str(node_name)
            tex.label = str(node_name)

        # PROPERTIES BEFORE SETTINGS, for add_group_node's reason: a property such as
        # voronoi_dimensions or wave_type decides WHICH sockets are live, so writing a socket first
        # can land on one the node then stops reading.
        props_applied = {}
        for key, value in (props or {}).items():
            props_applied[key] = _write_node_property(tex, str(key), value, tail=node_tail)

        settings_applied = {}
        for sock_name, value in (settings or {}).items():
            live, inactive = usable_sockets(tex.inputs, str(sock_name))
            if not live and inactive:
                raise MifOpError(
                    "'%s' on a %s exists but is INACTIVE at this node's settings, so a value "
                    "written there is never read. Set the property that selects it through "
                    "`properties` in this same call - properties are applied first. %s"
                    % (sock_name, tex.bl_idname, node_tail))
            if not live:
                raise MifOpError("a %s has no input '%s'. It has: %s. %s"
                                 % (tex.bl_idname, sock_name,
                                    ", ".join(s.name for s in tex.inputs) or "(none)", node_tail))
            sock = live[0]
            try:
                if isinstance(value, (list, tuple)):
                    for i, v in enumerate(value):
                        sock.default_value[i] = float(v)
                elif isinstance(value, bool):
                    sock.default_value = value
                else:
                    sock.default_value = _socket_value(sock.bl_idname, sock.name, value,
                                                       "'%s'" % tex.name, node_tail)
            except MifOpError:
                raise
            except (AttributeError, TypeError, ValueError) as exc:
                raise MifOpError("cannot write %r to '%s' on a %s: %s. %s"
                                 % (value, sock_name, tex.bl_idname, exc, node_tail))
            settings_applied[sock_name] = value

        # ---- COORDINATES ----------------------------------------------------------------------
        coord_node = None
        mapping_node = None
        want_mapping = scale_uv is not None or offset_uv is not None or rot_deg is not None
        if want_mapping or space is not None or uv_map is not None:
            vector_in = tex.inputs.get("Vector")
            if vector_in is None:
                raise MifOpError(
                    "a %s has no Vector input, so it cannot be given coordinates - drop space, "
                    "uvMap, scale, offset and rotation. %s" % (tex.bl_idname, node_tail))
            if uv_map is not None:
                coord_node = tree.nodes.new("ShaderNodeUVMap")
                made.append(coord_node)
                coord_node.uv_map = str(uv_map)
                coord_out = coord_node.outputs["UV"]
            else:
                coord_node = tree.nodes.new("ShaderNodeTexCoord")
                made.append(coord_node)
                # VALIDATED AGAINST THE NODE'S REAL OUTPUTS rather than a list written here, so a
                # build that adds or drops one is described correctly without this op being
                # touched.
                want = str(space) if space is not None else "Generated"
                match = [s for s in coord_node.outputs if s.name.lower() == want.strip().lower()]
                if not match:
                    raise MifOpError(
                        "'%s' is not a coordinate space on Blender %s. The Texture Coordinate node "
                        "offers: %s. %s"
                        % (space, bpy.app.version_string,
                           ", ".join(s.name for s in coord_node.outputs), node_tail))
                coord_out = match[0]
            coord_node.location = (base_x - 620, base_y)

            if want_mapping:
                mapping_node = tree.nodes.new("ShaderNodeMapping")
                made.append(mapping_node)
                mapping_node.location = (base_x - 320, base_y)
                if scale_uv is not None:
                    mapping_node.inputs["Scale"].default_value = scale_uv
                if offset_uv is not None:
                    mapping_node.inputs["Location"].default_value = offset_uv
                if rot_deg is not None:
                    # DEGREES IN, RADIANS ON THE SOCKET, about the texture plane's normal. The
                    # other two axes are not what anybody means by rotating a texture.
                    mapping_node.inputs["Rotation"].default_value = (0.0, 0.0,
                                                                     math.radians(rot_deg))
                # THE MAPPING NODE ALWAYS GETS A SOURCE. An unconnected Mapping Vector is a
                # constant zero, not "the default coordinates" - the whole surface takes the value
                # the pattern has at the origin and renders as a flat fill.
                tree.links.new(coord_out, mapping_node.inputs["Vector"])
                tree.links.new(mapping_node.outputs["Vector"], vector_in)
            else:
                tree.links.new(coord_out, vector_in)

        # ---- WHAT COMES OUT, AND WHAT THE DESTINATION WANTS ------------------------------------
        explicit_out = take(params, "fromSocket", default=None, kind=str)
        ramp_node = None
        source_node = tex
        source_out = None
        from_note = None
        if ramp_stops is not None:
            # A RAMP READS A SINGLE VALUE, so the pattern's scalar output feeds it whatever the
            # final destination is.
            ramp_in, _why = _pick_output(tex, True, explicit_out, node_tail)
            ramp_node = tree.nodes.new("ShaderNodeValToRGB")
            made.append(ramp_node)
            ramp_node.location = (base_x + 300, base_y)
            _write_ramp_stops(ramp_node.color_ramp, ramp_stops)
            tree.links.new(ramp_in, ramp_node.inputs["Fac"])
            source_node = ramp_node
            source_out = ramp_node.outputs["Color"]
            from_note = ("the pattern's '%s' output feeds the colour ramp, and the ramp's Color is "
                         "what reaches the destination - fromSocket below names the ramp's socket, "
                         "not the texture's." % ramp_in.name)

        relief_node = None
        bump_distance_filled, bump_built_default = False, None
        if target is not None:
            if relief is not None:
                # The relief node reads a HEIGHT or a COLOUR, and the destination is the Normal
                # socket, so the choice of texture output is made against the RELIEF node's input
                # rather than against Normal.
                if relief == "bump":
                    relief_node = tree.nodes.new("ShaderNodeBump")
                    made.append(relief_node)
                    relief_in = relief_node.inputs["Height"]
                    bump_built_default = round(float(relief_node.inputs["Distance"].default_value), 6)
                    if distance is not None:
                        relief_node.inputs["Distance"].default_value = distance
                    else:
                        # THE SAME CALL, THE SAME RELIEF, ON EVERY BUILD - see _BUMP_DISTANCE_WHEN_UNSET.
                        relief_node.inputs["Distance"].default_value = _BUMP_DISTANCE_WHEN_UNSET
                        bump_distance_filled = True
                else:
                    relief_node = tree.nodes.new("ShaderNodeNormalMap")
                    made.append(relief_node)
                    relief_in = relief_node.inputs["Color"]
                    if uv_map is not None:
                        relief_node.uv_map = str(uv_map)
                relief_node.location = (base_x + 560, base_y)
                if strength is not None:
                    relief_node.inputs["Strength"].default_value = strength
                if source_out is None:
                    source_out, from_note = _pick_output(
                        tex, relief_in.type == "VALUE", explicit_out, node_tail)
                tree.links.new(source_out, relief_in)
                source_node = relief_node
                source_out = relief_node.outputs["Normal"]
            elif source_out is None:
                source_out, from_note = _pick_output(
                    tex, target.type == "VALUE", explicit_out, node_tail)

            # THE OLD LINK IS REMOVED LAST, and it is remembered so it can be put back. links.new
            # on a single-input socket replaces silently, leaving the previous driver floating in
            # the tree where describe_material keeps reporting it - and if the new link is then
            # rejected, the socket ends up driven by nothing at all. Neither is acceptable from an
            # op whose refusals say nothing was changed.
            #
            # REMEMBERED AS NAMES, NOT AS THE SOCKET OBJECT. bpy hands back a new proxy on every
            # attribute access and the one read off a link is not guaranteed to survive that link
            # being removed - the same class of mistake as comparing nodes with `is`, which cost
            # link_group_nodes a false "linked: false" on wiring that had demonstrably been made.
            # A name survives anything, and the node it names is still in the tree.
            replaced = None
            restore = None
            for link in list(target.links):
                replaced = link.from_node.name
                restore = (link.from_node.name, link.from_socket.name)
                tree.links.remove(link)
            tree.links.new(source_out, target)

            # MEASURED OFF THE SOCKET. links.new returns a link object whether or not Blender kept
            # it, and a rejected link leaves the socket exactly as unlinked as it was.
            wired = target.is_linked and any(l.from_node.name == source_node.name
                                             for l in target.links)
            if not wired:
                put_back = False
                if restore is not None:
                    src = tree.nodes.get(restore[0])
                    old = (next((s for s in src.outputs if s.name == restore[1]), None)
                           if src is not None else None)
                    if old is not None:
                        tree.links.new(old, target)
                        put_back = bool(target.is_linked)
                raise MifOpError(
                    "Blender did not accept the link from '%s.%s' into %s - which usually means "
                    "the socket types are incompatible. %s%s"
                    % (source_node.name, source_out.name, target_desc, node_tail,
                       (" The link from '%s' that was there before was put back."
                        % restore[0]) if put_back else
                       (" The link from '%s' that was there before could NOT be restored, so that "
                        "input is now driven by nothing." % restore[0]) if restore else ""))
        else:
            replaced = None

        chain = [n.name for n in (coord_node, mapping_node, tex, ramp_node, relief_node)
                 if n is not None]
        out = {
            "ok": True,
            "material": mat.name,
            "type": ntype,
            "requestedType": str(asked_type) if str(asked_type) != ntype else None,
            "node": tex.name,
            # NAMES, because Blender appends .001 on a collision and every later call has to use
            # the name it actually got.
            "nodesCreated": [n.name for n in made],
            "chain": chain,
            "mappingNode": getattr(mapping_node, "name", None),
            # THE SCALE AS WRITTEN, read back off the node, and a note when a 3D space got only X and Y (B191).
            "mappingScale": ([round(float(v), 6) for v in mapping_node.inputs["Scale"].default_value]
                             if mapping_node is not None else None),
            "scaleNote": (("scale reached X and Y only, so Z stays %g - on a face that does not face Z this pattern "
                           "stretches by the X/Y factor. Pass scale as [x, y, z] to scale all three (the pattern node's "
                           "own Scale input multiplies on top of this: settings {Scale: ...})." % scale_uv[2])
                          if scale_uv is not None and not scale_3d and uv_map is None
                          and str(space or "Generated").strip().lower() != "uv" else None),
            "coordinateNode": getattr(coord_node, "name", None),
            "colorRampNode": getattr(ramp_node, "name", None),
            "reliefNode": getattr(relief_node, "name", None),
            # WHAT THE RELIEF NODE HOLDS, read back - strength on either node, distance on a Bump node.
            "reliefStrength": (round(float(relief_node.inputs["Strength"].default_value), 6)
                               if relief_node is not None and "Strength" in relief_node.inputs else None),
            "reliefDistance": (round(float(relief_node.inputs["Distance"].default_value), 6)
                               if relief_node is not None and "Distance" in relief_node.inputs else None),
            "distanceNote": (("no distance was given, so %g was written - this Blender's Bump node defaults to %g, which "
                              "renders almost no relief (measured on 5.0 and 5.2: 0.1/255 against 27-29 at 1.0). Pass "
                              "distance to choose." % (_BUMP_DISTANCE_WHEN_UNSET, bump_built_default))
                             if bump_distance_filled and bump_built_default != _BUMP_DISTANCE_WHEN_UNSET else None),
            "target": target_desc,
            "fromSocket": source_out.name if source_out is not None else None,
            "fromSocketNote": from_note,
            "replacedLinkFrom": replaced,
            # MEASURED, not assumed - see above. False here with a target set would mean the op
            # refused, so this is true whenever there is a target at all.
            "linked": bool(target is not None and target.is_linked),
            "settingsApplied": settings_applied,
            "propertiesApplied": props_applied,
            "blenderVersion": bpy.app.version_string,
            # THE READ-BACK, so the caller can check its own work in the same round trip rather
            # than following every write with describe_material.
            "readback": node_readback(tex, values=True),
            "uvMapNote": _uv_map_note(mat, uv_map),
        }
        if ramp_node is not None:
            out["colorRamp"] = color_ramp_stops(ramp_node)
        if target is None:
            out["note"] = ("no destination was given, so this pattern is in the tree and drives "
                           "nothing - it renders exactly as it did before. Wire it with "
                           "link_group_nodes (group 'material:%s'), or pass `input` next time."
                           % mat.name)
        elif relief_node is not None:
            out["note"] = (
                "wired through a %s node. A procedural pattern is a HEIGHT field, so Bump is the "
                "right converter; a Normal Map node would read the pattern as tangent-space RGB "
                "and light the surface wrongly while every field here still read as correct."
                % relief_node.bl_idname)
        elif coord_node is None:
            out["note"] = ("no coordinates were given, so the texture uses Blender's implicit "
                           "GENERATED coordinates - object space, normalised to the bounding box, "
                           "no UVs needed. Pass space or uvMap to choose.")
        return out
    except Exception:  # noqa: BLE001 - see below; the original is re-raised untouched
        # EVERY EXCEPTION, NOT ONLY MifOpError. A refusal is the expected way out of this block and
        # is what the rollback was written for, but it is not the only way out: a socket this build
        # spells differently raises KeyError, a stub or a future Blender raises AttributeError, and
        # catching only MifOpError leaves a half-built chain of nodes in the material while the
        # call reports a failure. Debris from a bug is worse than debris from a refusal, because
        # nobody is looking for it. The original exception is re-raised unchanged, so nothing about
        # WHAT went wrong is hidden by the tidying up.
        for node in made:
            try:
                tree.nodes.remove(node)
            except Exception:  # noqa: BLE001 - cleanup must not mask the real failure
                pass
        raise


_LIST_IMAGES_KEYS = ("source", "limit", "nameExcludes")


def op_list_images(params):
    """Every image datablock in the file, by name - the second collection found unlistable.

    FOUND 2026-09-08 by re-running the question that produced list_node_groups: which bpy.data
    collections can be CREATED and CONSUMED BY KEY and not ENUMERATED? Images were the other one.

      created    bpy.data.images.new here, and .load in ops_world (HDRI), ops_render and this file
      consumed   create_texture and set_material_texture both take an image NAME
      purged     clear_scene's purge list and file_info's _PURGEABLE both carry images
      listed     NOWHERE. The only two places that enumerated image names in the whole addon were
                 inside REFUSALS - `have = sorted(i.name for i in bpy.data.images)[:25]` - so a
                 caller discovered them by deliberately asking for one that does not exist, and got
                 at most twenty-five.

    NEITHER ASYMMETRY AUDIT SEES THIS, for the same reason they miss node_groups: images HAVE a
    creator and ARE reached by a caller-supplied key, so consumer-no-creator and
    demanded-kind-has-creator both pass. The missing shape is the third one - created, consumed, and
    not listable.

    SIZE AND USERS ARE REPORTED because both answer a question the name cannot. A 4096x4096 image is
    a different proposition from a 64x64 one when you are choosing which to plug into a material,
    and an image with no users and no fake user is one the next save DELETES - the same fact
    file_info gives for orphans, said where somebody choosing an image will see it.

    `source` filters by Blender's own image source enum (FILE, GENERATED, VIEWER...). Omit it for
    every image.
    """
    reject_unknown(params, _LIST_IMAGES_KEYS, "list_images",
                   notes={
                       "name": "this LISTS images; pass a name to create_texture or set_material_texture to USE one",
                       "image": "see `name` - this is the enumerator, not a lookup",
                       "filter": "spelt source, and it filters by Blender's image source enum (FILE, GENERATED, ...)",
                       "path": "there is no path filter; every image is returned and each row carries its own filepath",
                   })
    want = take(params, "source", default=None, kind=str)
    # CHECKED AGAINST THE ENUM, case-folded: compared exactly and never checked, a misspelled source answered zero rows
    # and no refusal - an empty result that should fail (found 2026-10-25).
    if want:
        known = [i.identifier for i in bpy.types.Image.bl_rna.properties["source"].enum_items]
        if want.upper() not in known:
            raise MifOpError("source %r is not an image source. Blender's are: %s." % (want, ", ".join(known)))
        want = want.upper()
    rows = []
    for img in bpy.data.images:
        if want and img.source != want:
            continue
        try:
            size = [int(img.size[0]), int(img.size[1])]
        except Exception:                                           # noqa: BLE001 - a freed or lazy image
            size = [0, 0]
        rows.append({
            "name": img.name,
            "source": img.source,
            "sizePx": size,
            "channels": int(getattr(img, "channels", 0)),
            "filepath": img.filepath or "",
            "packed": bool(getattr(img, "packed_file", None)),
            "users": int(getattr(img, "users", 0)),
            "fakeUser": bool(getattr(img, "use_fake_user", False)),
        })
        if img.source == "TILED":
            # a UDIM image's size is its first tile's; each tile is read at its own file's size (create_udim_image)
            rows[-1]["tiles"] = [{"number": int(t.number), "size": [int(t.size[0]), int(t.size[1])]}
                                 for t in img.tiles]
    rows.sort(key=lambda r: r["name"])
    rows = exclude_rows(rows, params)
    matched = len(rows)
    rows, extra = limit_rows(rows, params, "image(s)")
    return dict({"images": rows, "count": matched,
            "filteredBy": want or None,
            "totalInFile": len(bpy.data.images)}, **extra)


def op_transfer_face_materials(params):
    """Give each face of a target mesh the material of the nearest face of a source mesh.

    THE GAP, 2026-10-03: a high poly made from the low poly (convert, subdivide, displace) came out with every face on
    slot 0, so its ID bake was one colour - and a re-meshed garment lost its panel materials the same way. The garment
    work copied them across in raw Python six times (mat_transfer.py). Weights had transfer_weights; materials had
    nothing.

    BY NEAREST FACE IN WORLD SPACE: each target face's centre finds the closest point on the source's faces (a BVH of
    the source's base mesh, through both objects' transforms), and takes that face's material. The material is
    matched by NAME to a slot of the target, and a slot is added when the target has none holding it (slotsAdded) -
    so the target's own slot order is kept. maxDistance (world units) leaves a face whose nearest source face is
    farther than that unchanged and counts it; without it every face is reassigned. A source face on an empty slot
    is counted and left alone. The distances are reported so a mismatched pair (wrong object, wrong scale) shows.
    """
    from mathutils.bvhtree import BVHTree
    reject_unknown(params, ("source", "target", "maxDistance"), "transfer_face_materials")
    src = get_object(take(params, "source", required=True, kind=str), want_mesh=True)
    dst = get_object(take(params, "target", required=True, kind=str), want_mesh=True)
    if src is dst:
        raise MifOpError("source and target are the same object. NOTHING was changed.")
    max_d = take_float(params, "maxDistance", default=None)
    if max_d is not None and not max_d > 0:
        raise MifOpError("maxDistance must be above 0, got %r. NOTHING was changed." % max_d)
    sme, dme = src.data, dst.data
    if not len(sme.polygons) or not len(dme.polygons):
        raise MifOpError("'%s' has no faces to %s. NOTHING was changed."
                         % ((src.name, "copy from") if not len(sme.polygons) else (dst.name, "assign to")))
    smw = src.matrix_world
    tree = BVHTree.FromPolygons([smw @ v.co for v in sme.vertices], [list(p.vertices) for p in sme.polygons])
    src_mats = [s.material.name if s.material else None for s in src.material_slots]
    dmw = dst.matrix_world
    plan, far, empty, dists = [], 0, 0, []
    for p in dme.polygons:
        hit = tree.find_nearest(dmw @ p.center)
        if hit[2] is None:
            far += 1
            continue
        dists.append(hit[3])
        if max_d is not None and hit[3] > max_d:
            far += 1
            continue
        si = sme.polygons[hit[2]].material_index
        name = src_mats[si] if 0 <= si < len(src_mats) else None
        if name is None:
            empty += 1
            continue
        plan.append((p.index, name))
    if not plan:
        raise MifOpError("no face of '%s' found a source face with a material%s. NOTHING was changed."
                         % (dst.name, " within maxDistance %g" % max_d if max_d is not None else ""))
    # Slots by material name - the target's own order kept, any missing material appended.
    have = {s.material.name: i for i, s in enumerate(dst.material_slots) if s.material is not None}
    added = []
    for name in sorted({n for _f, n in plan}):
        if name not in have:
            dme.materials.append(bpy.data.materials[name])
            have[name] = len(dst.material_slots) - 1
            added.append(name)
    changed, by_mat = 0, {}
    for fi, name in plan:
        idx = have[name]
        if dme.polygons[fi].material_index != idx:
            dme.polygons[fi].material_index = idx
            changed += 1
        by_mat[name] = by_mat.get(name, 0) + 1
    dme.update()
    dists.sort()
    return {"source": src.name, "target": dst.name, "faces": len(dme.polygons), "facesAssigned": len(plan),
            "facesChanged": changed, "byMaterial": by_mat, "slotsAdded": added, "fartherThanMax": far,
            "sourceSlotEmpty": empty,
            "distance": {"max": round(dists[-1], 6), "p95": round(dists[int(0.95 * (len(dists) - 1))], 6)}
            if dists else None,
            "slots": [s.material.name if s.material else None for s in dst.material_slots]}


def op_reload_image(params):
    """Re-read image datablocks from their files - after a bake or an outside tool rewrote them.

    THE GAP, 2026-10-03: a 4K normal map re-baked to the same path kept showing the OLD pixels in every material that
    used it, and the garment work reloaded it through raw Python seven times. Blender reads an image file once; writing
    the file again changes nothing in the session until Image.reload().

    REFUSED, per image, before anything reloads: a PACKED image (reload reads the packed copy, not the file on disk -
    unpack it or point it at the file), a GENERATED or file-less one, a file that is not there, and an image with
    UNSAVED edits in memory - a bake that has not been saved yet is exactly that, and reloading throws it away, so it
    needs discardUnsaved:true. images (a list) or all:true (every image with a file) reload several at once.
    """
    reject_unknown(params, ("image", "name", "images", "all", "discardUnsaved"), "reload_image")
    one = take(params, "image", "name", default=None, kind=str)
    many = params.get("images")
    every = take_bool(params, "all", default=False)
    discard = take_bool(params, "discardUnsaved", default=False)
    if sum(x is not None and x is not False for x in (one, many, every or None)) != 1:
        raise MifOpError("pass exactly one of image, images or all:true. NOTHING was reloaded.")
    if many is not None and (not isinstance(many, (list, tuple)) or not many
                             or not all(isinstance(n, str) for n in many)):
        raise MifOpError("images must be a non-empty list of image names (list_images names them). NOTHING was "
                         "reloaded.")
    if every:
        targets = [img for img in bpy.data.images if img.source in ("FILE", "SEQUENCE", "MOVIE") and img.filepath]
        if not targets:
            raise MifOpError("no image in this file is read from a file, so there is nothing to reload.")
    else:
        names = [one] if one is not None else list(many)
        have = set(bpy.data.images.keys())   # once - bpy collection membership is a walk of the collection
        missing = [n for n in names if n not in have]
        if missing:
            raise MifOpError("no image named %s (list_images names them). NOTHING was reloaded."
                             % ", ".join("'%s'" % n for n in missing))
        targets = [bpy.data.images[n] for n in names]
    problems = []
    for img in targets:
        path = bpy.path.abspath(img.filepath) if img.filepath else ""
        if getattr(img, "packed_file", None):
            problems.append("'%s' is PACKED - reload reads the packed copy, not %s" % (img.name, path or "a file"))
        elif img.source not in ("FILE", "SEQUENCE", "MOVIE") or not path:
            problems.append("'%s' is %s with no file to read" % (img.name, img.source))
        elif not os.path.isfile(path):
            problems.append("'%s' points at %s, which does not exist" % (img.name, path))
        elif img.is_dirty and not discard:
            problems.append("'%s' has UNSAVED pixels in memory (an unsaved bake reads like this) - reloading discards "
                            "them; save the image first, or pass discardUnsaved:true" % img.name)
    if problems:
        raise MifOpError("%s. NOTHING was reloaded." % "; ".join(problems))
    rows = []
    for img in targets:
        before = [int(img.size[0]), int(img.size[1])] if img.has_data else None
        path = bpy.path.abspath(img.filepath)
        img.reload()
        # A reloaded image is not read until something asks for it; size reads it, which is the read-back.
        after = [int(img.size[0]), int(img.size[1])]
        rows.append({"image": img.name, "filepath": path, "sizeBefore": before, "sizeAfter": after,
                     "fileModified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(path))),
                     "hasData": bool(img.has_data)})
    unreadable = [r["image"] for r in rows if r["sizeAfter"] == [0, 0]]
    out = {"reloaded": len(rows), "images": rows}
    if unreadable:
        out["warning"] = ("these reloaded with no pixels - the file did not decode: %s" % ", ".join(unreadable))
    return out


# ---------------------------------------------------------------------------
# import_pbr_texture_set - a folder of PBR maps as one material (2026-10-05)
# ---------------------------------------------------------------------------
_PBR_KINDS = ("baseColor", "normal", "roughness", "metallic", "displacement", "ao", "alpha", "emissive",
              "subsurface")
_PBR_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".tga", ".bmp", ".hdr", ".webp")
# NAME TOKENS, checked in this order - a file is the first kind whose word it carries. baseColor last, because "color"
# and "col" are the loosest words and an "Emission_Color" is an emission map.
_PBR_WORDS = (
    ("normal", ("normal", "normals", "nrm", "nor", "nrml", "normalgl", "normaldx", "normalmap")),
    ("roughness", ("roughness", "rough", "rgh")),
    ("metallic", ("metallic", "metalness", "metal", "mtl")),
    ("displacement", ("displacement", "disp", "displace", "height", "heightmap", "bump")),
    ("ao", ("ao", "ambientocclusion", "occlusion", "occ")),
    ("alpha", ("opacity", "alpha", "transparency")),
    ("emissive", ("emission", "emissive", "emit", "glow")),
    ("subsurface", ("subsurface", "sss", "scattering")),
    ("baseColor", ("basecolor", "albedo", "diffuse", "diff", "color", "colour", "col", "base")),
)


def _pbr_tokens(stem):
    import re   # noqa: PLC0415 - local, the one regex this file needs
    low = stem.lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", low) if t]
    # CamelCase too ("WoodBaseColor"): split before each capital of the original, then rejoin pairs like base+color.
    camel = [t.lower() for t in re.findall(r"[A-Z]?[a-z]+|[0-9]+[kK]?|[A-Z]+(?![a-z])", stem)]
    out = list(dict.fromkeys(tokens + camel))
    out += [a + b for a, b in zip(camel, camel[1:])]
    return out


def _pbr_classify(stem):
    """(kind or None, resolution token or None, udim tile or None, normal convention or None) from a file name."""
    import re   # noqa: PLC0415
    toks = _pbr_tokens(stem)
    kind = next((k for k, words in _PBR_WORDS if any(w in toks for w in words)), None)
    res = next((t.upper() for t in toks if re.fullmatch(r"[0-9]{1,2}k", t)), None)
    tile = next((int(t) for t in re.split(r"[^0-9]+", stem) if t and len(t) == 4 and 1001 <= int(t) <= 2000), None)
    conv = None
    if kind == "normal":
        if any(t in toks for t in ("gl", "opengl", "normalgl", "ogl")):
            conv = "OpenGL"
        elif any(t in toks for t in ("dx", "directx", "normaldx", "d3d")):
            conv = "DirectX"
    return kind, res, tile, conv


def _pbr_mix_multiply(tree, a_socket, b_socket, to_socket):
    """A Mix node (colour, MULTIPLY, factor 1) between two colours, into to_socket. ShaderNodeMix from 3.4; its A, B and
    Result exist once per data type under one name, so they are found by identifier."""
    mix = tree.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    ins = {s.identifier: s for s in mix.inputs}
    outs = {s.identifier: s for s in mix.outputs}
    ins["Factor_Float"].default_value = 1.0
    for link in list(to_socket.links):
        tree.links.remove(link)
    tree.links.new(a_socket, ins["A_Color"])
    tree.links.new(b_socket, ins["B_Color"])
    tree.links.new(outs["Result_Color"], to_socket)
    mix.location = (to_socket.node.location[0] - 250, to_socket.node.location[1] + 150)
    return mix


def op_import_pbr_texture_set(params):
    """A folder of PBR maps into one material in one call: each map found by its file name, wired where it belongs, in
    the colour space it needs.

    THE GAP, 2026-10-04: set_material_texture wires one map a call and could not do ambient occlusion, displacement or
    subsurface, so a downloaded set (base colour, normal, roughness, metallic, displacement, AO) was six calls and three
    raw-Python steps. Maps are found by the words in their names (_PBR_WORDS: basecolor / albedo / diffuse / color,
    normal / nrm, roughness / rough, metallic / metal, displacement / height, ao / occlusion, opacity / alpha, emission,
    subsurface / sss), CamelCase or separated. Only base colour and emission are sRGB; every other map is Non-Color
    (set_material_texture's rule). normal goes through a Normal Map node; displacement through a Displacement node into
    the Material Output's Displacement (Cycles reads it as bump unless the material's displacement method says
    otherwise - the reply says which it is); ao is MULTIPLIED into the base colour by a Mix node, or left as a free node
    when there is no base colour; subsurface drives the Principled subsurface weight.

    ONE FILE PER MAP. Several candidates for one map (a 4K and an 8K roughness; an OpenGL and a DirectX normal) choose by
    `resolution` (e.g. "4K"), then an OpenGL normal over a DirectX one (Blender reads OpenGL green), then name; every
    file not used is listed in skipped with the reason, as is an image whose name carries no map word. A name holding a
    number 1001-2000 is a UDIM tile: bpy.data.images.load reads it as ONE image (measured on 4.2.23-5.2.2 - detection is
    the file browser's), which is the default and is named in udimFiles; udim true makes it a TILED image, which finds
    the set's other tiles (1001, 1002 ...). include limits the kinds; mapping {scale, offset} drives every image through
    one shared Mapping node. Every check runs before the first node; a failure while wiring removes what this call made.
    """
    reject_unknown(params, ("material", "name", "folder", "include", "resolution", "mapping", "udim"),
                   "import_pbr_texture_set")
    mat_name = take(params, "material", "name", required=True, kind=str)
    folder = take(params, "folder", required=True, kind=str)
    resolution = take(params, "resolution", default=None, kind=str)
    udim = take_bool(params, "udim", default=False)
    include = params.get("include")
    if include is not None:
        if not isinstance(include, (list, tuple)) or not include or not all(isinstance(k, str) for k in include):
            raise MifOpError("include is a non-empty list of map kinds (%s). NOTHING was changed."
                             % ", ".join(_PBR_KINDS))
        bad = [k for k in include if k not in _PBR_KINDS]
        if bad:
            raise MifOpError("include names %s, which are not map kinds (%s). NOTHING was changed."
                             % (bad, ", ".join(_PBR_KINDS)))
    mapping = params.get("mapping")
    scale_uv = offset_uv = None
    if mapping is not None:
        if not isinstance(mapping, dict) or not mapping or set(mapping) - {"scale", "offset"}:
            raise MifOpError("mapping is {scale, offset} - each a number or [u, v]; got %r. NOTHING was changed."
                             % (mapping,))
        scale_uv = _uv_pair(mapping, "scale", 1.0)
        offset_uv = _uv_pair(mapping, "offset", 0.0)
        if scale_uv is not None and (scale_uv[0] == 0.0 or scale_uv[1] == 0.0):
            raise MifOpError("mapping scale 0 collapses every map to one pixel. NOTHING was changed.")
    mat = bpy.data.materials.get(str(mat_name))
    if mat is None:
        raise MifOpError("no material named '%s' - create_material makes one. NOTHING was changed." % mat_name)
    bsdf = _principled(mat)
    tree = mat.node_tree
    root = os.path.abspath(bpy.path.abspath(str(folder)))
    if not os.path.isdir(root):
        raise MifOpError("no folder at '%s'. NOTHING was changed." % root)

    # ---- every image in the folder, classified; UDIM tiles of one set grouped
    found, skipped = {}, []
    groups = {}
    for fname in sorted(os.listdir(root)):
        path = os.path.join(root, fname)
        stem, ext = os.path.splitext(fname)
        if not os.path.isfile(path) or ext.lower() not in _PBR_IMAGE_EXT:
            continue
        kind, res, tile, conv = _pbr_classify(stem)
        if kind is None:
            skipped.append({"file": path, "reason": "no map kind in the name (%s)" % fname})
            continue
        key = (kind, stem.replace(str(tile), "#") if tile else stem, ext.lower())
        g = groups.setdefault(key, {"kind": kind, "res": res, "conv": conv, "files": [], "tiles": []})
        g["files"].append(path)
        if tile:
            g["tiles"].append(tile)
    for g in groups.values():
        found.setdefault(g["kind"], []).append(g)
    if not found:
        raise MifOpError("no PBR map recognised in '%s' - no image there carries a map word (%s). NOTHING was changed."
                         % (root, ", ".join(w for _k, ws in _PBR_WORDS for w in ws[:2])))
    want = list(include) if include is not None else [k for k in _PBR_KINDS if k in found]
    missing = [k for k in want if k not in found]
    if missing:
        raise MifOpError("include asks for %s and no file in '%s' is one (found: %s). NOTHING was changed."
                         % (", ".join(missing), root, ", ".join(sorted(found))))
    conv_rank = {"OpenGL": 0, None: 1, "DirectX": 2}
    chosen = {}
    for kind in sorted(found):
        cands = found[kind]
        ranked = sorted(cands, key=lambda g: (0 if (resolution and g["res"] == str(resolution).upper()) else 1,
                                              conv_rank.get(g["conv"], 1), 0 if g["res"] is None else 1,
                                              os.path.basename(g["files"][0])))
        if kind not in want:
            for g in cands:
                skipped.append({"file": g["files"][0], "reason": "%s was not included" % kind})
            continue
        chosen[kind] = ranked[0]
        for g in ranked[1:]:
            why = "another %s file was chosen: %s" % (kind, os.path.basename(ranked[0]["files"][0]))
            if kind == "normal" and g["conv"] == "DirectX" and ranked[0]["conv"] == "OpenGL":
                why += " (OpenGL preferred - Blender reads a normal map's green as OpenGL does)"
            skipped.append({"file": g["files"][0], "reason": why})
    udim_files = []
    for kind, g in chosen.items():
        if g["tiles"]:
            files = sorted(g["files"])
            g["files"] = files
            udim_files.append({"kind": kind, "files": files, "tiles": sorted(g["tiles"]),
                               "loadedAs": "TILED" if udim else "one image (the first tile)"})
            if not udim:
                for f in files[1:]:
                    skipped.append({"file": f, "reason": "another tile of a UDIM set - udim true loads the set as one "
                                                         "TILED image"})

    # ---- wire, undoing everything on a failure
    nodes_before = {n.name for n in tree.nodes}
    links_before = [(lk.from_node.name, lk.from_socket.identifier, lk.to_node.name, lk.to_socket.identifier)
                    for lk in tree.links]
    images_before = {i.name for i in bpy.data.images}
    wired, notes = {}, []
    order = [k for k in ("baseColor", "normal", "roughness", "metallic", "alpha", "emissive", "subsurface",
                         "displacement", "ao") if k in chosen]
    try:
        for kind in order:
            g = chosen[kind]
            path = g["files"][0]
            call = {"material": mat.name, "file": path, "nodeName": "PBR %s" % kind}
            if kind in ("baseColor", "normal", "roughness", "metallic", "alpha", "emissive"):
                call["input"] = kind
            elif kind == "subsurface":
                sock = bsdf.inputs.get("Subsurface Weight") or bsdf.inputs.get("Subsurface")
                call.update({"toNode": bsdf.name, "toSocket": sock.name if sock is not None else "Subsurface Weight"})
            elif kind == "displacement":
                call.update({"toNode": "displacement", "toSocket": "Height"})
            else:
                call.update({"link": False, "colorspace": "Non-Color"})
            r = op_set_material_texture(call)
            img = bpy.data.images.get(r["image"])
            row = {"file": path, "image": r["image"], "node": r["imageNode"], "colorspace": r["colorSpace"],
                   "source": img.source if img is not None else None}
            if kind == "normal":
                row["convention"] = g["conv"] or "unknown"
                if g["conv"] == "DirectX":
                    notes.append("the normal map is DirectX-convention (by its name), and Blender reads OpenGL: its "
                                 "green channel reads inverted - bumps read as dents.")
            if kind == "displacement":
                disp = tree.nodes.get(r["toNode"])
                out = next((n for n in tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output),
                           next((n for n in tree.nodes if n.type == "OUTPUT_MATERIAL"), None))
                if out is None:
                    raise MifOpError("material '%s' has no Material Output to take the displacement." % mat.name)
                tree.links.new(disp.outputs["Displacement"], out.inputs["Displacement"])
                method = getattr(getattr(mat, "cycles", None), "displacement_method", None) or \
                    getattr(mat, "displacement_method", None)
                row.update({"displacementNode": disp.name, "displacementMethod": method})
                notes.append("displacement goes through a Displacement node into the Material Output; this material's "
                             "displacement method is %s, so Cycles %s." % (method, "moves the surface" if method in
                                                                           ("DISPLACEMENT", "BOTH") else
                                                                           "shades it as bump, moving no vertex"))
            if kind == "ao":
                base = wired.get("baseColor")
                if base is not None:
                    base_tex = tree.nodes.get(base["node"])
                    ao_tex = tree.nodes.get(r["imageNode"])
                    mix = _pbr_mix_multiply(tree, base_tex.outputs["Color"], ao_tex.outputs["Color"],
                                            _resolve_input(bsdf, "baseColor"))
                    row.update({"how": "MULTIPLY into baseColor", "mixNode": mix.name})
                else:
                    row["how"] = "a free node - there is no base colour to multiply it into"
            if udim and g["tiles"] and img is not None:
                img.source = "TILED"
                row.update({"source": img.source, "tiles": sorted(t.number for t in img.tiles)})
            wired[kind] = row
        if scale_uv is not None or offset_uv is not None:
            coord = tree.nodes.new("ShaderNodeTexCoord")
            mp = tree.nodes.new("ShaderNodeMapping")
            if scale_uv is not None:
                mp.inputs["Scale"].default_value = scale_uv
            if offset_uv is not None:
                mp.inputs["Location"].default_value = offset_uv
            tree.links.new(coord.outputs["UV"], mp.inputs["Vector"])
            for row in wired.values():
                tree.links.new(mp.outputs["Vector"], tree.nodes[row["node"]].inputs["Vector"])
            coord.location = (bsdf.location[0] - 1400, bsdf.location[1])
            mp.location = (bsdf.location[0] - 1150, bsdf.location[1])
    except Exception as exc:
        for n in [n for n in tree.nodes if n.name not in nodes_before]:
            tree.nodes.remove(n)
        have = {(lk.from_node.name, lk.from_socket.identifier, lk.to_node.name, lk.to_socket.identifier)
                for lk in tree.links}
        for fn, fs, tn, ts in links_before:
            if (fn, fs, tn, ts) not in have and fn in tree.nodes and tn in tree.nodes:
                a = next((x for x in tree.nodes[fn].outputs if x.identifier == fs), None)
                b = next((x for x in tree.nodes[tn].inputs if x.identifier == ts), None)
                if a is not None and b is not None:
                    tree.links.new(a, b)
        for img in [i for i in bpy.data.images if i.name not in images_before and i.users == 0]:
            bpy.data.images.remove(img)
        raise MifOpError("the set could not be wired (%s), and everything this call made was removed again. NOTHING "
                         "was changed." % str(exc).rstrip("."))
    return {"material": mat.name, "folder": root, "wired": wired, "kinds": order,
            "skipped": skipped[:100] or None, "udimFiles": udim_files or None,
            "mapping": ({"scale": list(scale_uv) if scale_uv else None, "offset": list(offset_uv) if offset_uv else None}
                        if mapping is not None else None),
            "notes": notes or None}


OPS = {
    "reload_image": op_reload_image,
    "transfer_face_materials": op_transfer_face_materials,
    "create_material": op_create_material,
    "set_material_properties": op_set_material_properties,
    "list_materials": op_list_materials,
    "list_images": op_list_images,
    "describe_material": op_describe_material,
    "bake_texture": op_bake_texture,
    "fix_normal_map": op_fix_normal_map,
    "assign_material_to_faces": op_assign_material_to_faces,
    "set_material_texture": op_set_material_texture,
    "set_material_settings": op_set_material_settings,
    "set_material_volume": op_set_material_volume,
    "add_procedural_texture": op_add_procedural_texture,
    "set_color_ramp": op_set_color_ramp,
    "import_pbr_texture_set": op_import_pbr_texture_set,
}
