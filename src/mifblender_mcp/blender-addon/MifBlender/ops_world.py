"""The world: background colour, strength, and an HDRI - the difference between lit and black.

WHY IT IS A GAP WORTH ITS OWN OP. A scene with no world contributes no ambient light at all, so an
interior lit only by its own fixtures renders as pure black outside their falloff, and the usual
first reaction is to blame the lights. Setting it is three nodes of shader graph, which is exactly
the sort of thing nobody should be writing through an exec hatch.

WHAT "STRENGTH" ACTUALLY DOES, since it is the most misread control here: it multiplies the
background's emission. A strength of 1.0 with a mid-grey colour is roughly an overcast day and will
wash out a deliberately dark interior; the abandoned-lab look wants something nearer 0.02-0.1.
Stated because the number that looks reasonable is usually two orders of magnitude too high.
"""
import os

import bpy

from .ops_common import (MifOpError, exclude_rows, finite_float, finite_floats, finite_int, limit_rows, reject_unknown, rnd, take, take_bool,
                         take_float)

_WORLD_KEYS = {
    "name", "color", "backgroundColor", "strength", "hdri", "hdriPath",
    "rotation", "mistUse", "mistStart", "mistDepth", "useAsLight", "sun", "probeResolution",
}

# THE WORLD'S SUN AND PROBE RESOLUTION (EEVEE from 4.2): on bpy.types.World on 4.4.0 and 5.2.1, absent on 3.6.23
# (measured 2026-10-04). sun_threshold defaults to 10: world light above it is extracted into a sun of its own, with a
# shadow. An HDRI meant as skylight only, beside a real sun lamp that owns the sun's height and direction, wants 0.
_SUN_KEYS = {"threshold": "sun_threshold", "angle": "sun_angle"}
_SUN_SHADOW_KEYS = {"enabled": "use_sun_shadow", "jitter": "use_sun_shadow_jitter",
                    "overblur": "sun_shadow_jitter_overblur"}


def _parse_sun(params):
    """[(attribute, value)] for set_world's sun and probeResolution, checked against this build - or a refusal."""
    sun = params.get("sun")
    res = params.get("probeResolution")
    if sun is None and res is None:
        return []
    world_rna = bpy.types.World.bl_rna.properties
    if world_rna.get("sun_threshold") is None:
        raise MifOpError("the world's sun and probe resolution are EEVEE settings from Blender 4.2, and this Blender (%s) "
                         "has neither. NOTHING was changed." % bpy.app.version_string)
    out = []
    if sun is not None:
        if not isinstance(sun, dict) or not sun:
            raise MifOpError("sun must be an object {threshold, angle, shadow {enabled, jitter, overblur}}, got %r. "
                             "NOTHING was changed." % (sun,))
        bad = sorted(k for k in sun if k not in _SUN_KEYS and k != "shadow")
        if bad:
            raise MifOpError("sun takes threshold, angle and shadow; %s is not one. NOTHING was changed."
                             % ", ".join(bad))
        for key, attr in _SUN_KEYS.items():
            if key in sun:
                val = finite_float(sun[key], "sun." + key)
                if val < 0.0:
                    raise MifOpError("sun.%s must be 0 or more, got %g. NOTHING was changed." % (key, val))
                out.append((attr, val))
        shadow = sun.get("shadow")
        if shadow is not None:
            if not isinstance(shadow, dict) or not shadow:
                raise MifOpError("sun.shadow must be an object {enabled, jitter, overblur}, got %r. NOTHING was "
                                 "changed." % (shadow,))
            bad = sorted(k for k in shadow if k not in _SUN_SHADOW_KEYS)
            if bad:
                raise MifOpError("sun.shadow takes enabled, jitter and overblur; %s is not one. NOTHING was changed."
                                 % ", ".join(bad))
            for key, attr in _SUN_SHADOW_KEYS.items():
                if key not in shadow:
                    continue
                if key == "overblur":
                    val = finite_float(shadow[key], "sun.shadow.overblur")
                    if val < 0.0:
                        raise MifOpError("sun.shadow.overblur must be 0 or more, got %g. NOTHING was changed." % val)
                elif isinstance(shadow[key], bool):
                    val = shadow[key]
                else:
                    raise MifOpError("sun.shadow.%s must be true or false, got %r. NOTHING was changed."
                                     % (key, shadow[key]))
                out.append((attr, val))
    if res is not None:
        valid = [i.identifier for i in world_rna["probe_resolution"].enum_items]
        got = str(finite_int(res, "probeResolution")) if isinstance(res, (int, float)) else str(res)
        if got not in valid:
            raise MifOpError("probeResolution must be one of %s on this Blender, got %r. NOTHING was changed."
                             % (", ".join(valid), res))
        out.append(("probe_resolution", got))
    return out


def _sun_state(world):
    """The world's sun and probe resolution, read back - None on a build without them."""
    if world is None or not hasattr(world, "sun_threshold"):
        return None
    return {"threshold": round(float(world.sun_threshold), 6), "angle": round(float(world.sun_angle), 6),
            "shadow": {"enabled": bool(world.use_sun_shadow), "jitter": bool(world.use_sun_shadow_jitter),
                       "overblur": round(float(world.sun_shadow_jitter_overblur), 6)},
            "probeResolution": world.probe_resolution}


def _ensure_world(name):
    """The scene's world, or the one `name` asks for.

    `name` USED TO BE HONOURED ONLY ON THE BRANCH THAT CREATED A WORLD. On a scene that already had
    one - which is every scene Blender opens - it was dropped, and set_world edited whatever world
    the scene happened to be using:

        set_world {"name": "BrandNewWorld", "strength": 2.0}
          -> ok: true, "world": "World"          <- a different datablock, silently

    while world_info, thirty lines down THIS FILE, documents the same key as "a specific world by
    name" and refuses one it cannot find. Two ops in one module disagreeing about their shared
    parameter, with the write half being the one that guessed.

    Found by asking a live Blender which DATABLOCK KINDS the addon demands - hand every op a value
    that cannot exist and read the refusal, which names the kind - then checking each kind has a
    creator. `world` was the only one of seven with no create op, and chasing that turned up this
    instead: not a missing creator, a parameter being ignored.

    Now: a named world that exists is selected (and becomes the scene's world, which is what naming
    it asks for); a named world that does not exist is refused in world_info's own words, unless
    there is no world at all, in which case it is created under that name - the original behaviour
    on the original branch, kept because op_set_world's ordering comments depend on it.
    """
    sc = bpy.context.scene
    if name:
        world = bpy.data.worlds.get(name)
        if world is None:
            if sc.world is not None:
                known = sorted(w.name for w in bpy.data.worlds)
                raise MifOpError(
                    "no world named '%s'. Present: %s. The scene is using '%s' - omit name to "
                    "edit that one. NOTHING was changed."
                    % (name, ", ".join(known) if known else "<none>", sc.world.name))
            world = bpy.data.worlds.new(name)
        sc.world = world
    elif sc.world is None:
        sc.world = bpy.data.worlds.new("World")
    if not sc.world.use_nodes:
        sc.world.use_nodes = True
    return sc.world


def _ensure_wired(world, bg):
    """Make sure this Background node actually reaches the world output.

    WHY THIS IS NOT JUST DONE AT CREATION TIME, which is what it used to be. The helper below wired
    the node only on the branch that CREATED one. A world that already had a Background node but no
    link from it to the output got that node handed straight back, and every value written to it was
    inert - set_world returned ok:true, the colour and strength landed on the node, and the scene's
    ambient did not change by any amount.

    world_info could already SEE it and said so in as many words - contributesLight false, with a
    blocker reading "the Background node exists but is NOT connected to the world output, so every
    value on it is inert - it accepts writes and changes no light". So the diagnosis existed and the
    op doing the writing never consulted it. Found 2026-09-05 after tuning a bunker's ambient across
    several renders with nothing happening, and only noticing because world_info was called for an
    unrelated reason.

    Wiring it is the same judgement the creation branch already makes - there is exactly one
    sensible answer for where a Background node's output goes - applied to the case where the node
    exists and the link does not.
    """
    tree = world.node_tree
    out = next((n for n in tree.nodes if n.type == "OUTPUT_WORLD"), None)
    if out is None:
        out = tree.nodes.new("ShaderNodeOutputWorld")
    surface = out.inputs.get("Surface")
    if surface is None:
        return False
    for link in tree.links:
        if link.to_socket == surface and link.from_node == bg:
            return False                      # already wired, nothing to do
    tree.links.new(bg.outputs["Background"], surface)
    return True                               # we had to connect it


def _background_node(world):
    for n in world.node_tree.nodes:
        if n.type == "BACKGROUND":
            _ensure_wired(world, n)
            return n
    # A world whose tree was rebuilt by hand may have no Background node; make one and wire it
    # rather than refusing, since there is exactly one sensible answer.
    bg = world.node_tree.nodes.new("ShaderNodeBackground")
    _ensure_wired(world, bg)
    return bg


def op_set_world(params):
    """Set the world background, and report what it holds afterwards.

    params:
      color / backgroundColor [r,g,b]   flat background colour
      strength (float)                  emission multiplier - see the header, 1.0 is bright
      hdri / hdriPath (str)             an image file to use as the environment instead
      rotation (float)                  radians, Z rotation of the HDRI
      mistUse / mistStart / mistDepth   the scene mist pass
      useAsLight (bool)                 whether the world contributes to lighting at all
    """
    reject_unknown(params, _WORLD_KEYS, "set_world")

    # CHECKED BEFORE _ensure_world, which is not a reader: on a scene with no world it CREATES one
    # and turns use_nodes on. So set_world({hdri, color}) used to build a World datablock, enable
    # its node tree, and then refuse with "NOTHING was changed" - leaving a world in a file that had
    # none. The conflict needs nothing but params, so there was never a reason for it to be below.
    hdri = take(params, "hdri", "hdriPath", default=None, kind=str)
    col = params.get("color", params.get("backgroundColor"))
    if hdri and col is not None:
        raise MifOpError("pass an hdri OR a colour, not both - the environment texture replaces "
                         "the flat colour, so one of them would silently do nothing. NOTHING was "
                         "changed.")

    # The file check moves up for the same reason: it needs a path and the filesystem, nothing from
    # the world, and it was refusing "NOTHING was changed" with a freshly built World in the file.
    hdri_path = bpy.path.abspath(str(hdri)) if hdri else None
    if hdri_path is not None and not os.path.isfile(hdri_path):
        raise MifOpError("no such HDRI file: %s. Blender would create a broken image "
                         "datablock and render black rather than failing. NOTHING was changed."
                         % hdri_path)

    # And the colour's SHAPE AND VALUES, for the third time the same argument: it is a check on the
    # argument, it needs no world, and below _ensure_world it was refusing with one already created.
    # Every channel finite - see _finite_colour for what the bare float() let through.
    col_vals = _finite_colour(col, "NOTHING was changed.") if col is not None else None

    # AND EVERYTHING ELSE, THE FOURTH TIME. rotation, strength, useAsLight and the three mist keys were read below
    # _ensure_world and below the colour / HDRI writes, so {color: [...], strength: NaN} was refused with the
    # colour already written - and on a scene with no world, with a World created (measured on 5.0.1,
    # 2026-09-21). All of them are checks on the argument, and rotation's "needs an environment texture" can be
    # answered from the world as it is and the hdri in this call.
    world_name = take(params, "name", default=None, kind=str)
    try:
        rot = take_float(params, "rotation", default=None)
        strength = take_float(params, "strength", default=None)
        asked_light = "useAsLight" in params
        want_light = take_bool(params, "useAsLight", default=True) if asked_light else None
        want_mist = take_bool(params, "mistUse", default=False) if "mistUse" in params else None
        ms = take_float(params, "mistStart", default=None)
        md = take_float(params, "mistDepth", default=None)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was changed." % str(exc).rstrip("."))
    try:
        sun_writes = _parse_sun(params)
    except MifOpError as exc:
        text = str(exc)
        raise MifOpError(text if text.endswith("NOTHING was changed.") else "%s NOTHING was changed." % text.rstrip("."))
    if rot is not None and not hdri:
        target = bpy.data.worlds.get(world_name) if world_name else bpy.context.scene.world
        has_tex = bool(target is not None and target.node_tree is not None
                       and any(n.type == "TEX_ENVIRONMENT" for n in target.node_tree.nodes))
        if not has_tex:
            raise MifOpError(
                "rotation turns an environment texture and this world has none - it would be showing %s. "
                "Load one with hdriPath in this call or a previous one, then rotate it. NOTHING was changed."
                % ("a flat colour" if col is not None else "no environment texture"))

    world = _ensure_world(world_name)
    bg = _background_node(world)

    before = {
        "color": rnd(list(bg.inputs["Color"].default_value)[:3]),
        "strength": round(float(bg.inputs["Strength"].default_value), 6),
        "environmentTexturePresent": any(n.type == "TEX_ENVIRONMENT"
                                         for n in world.node_tree.nodes),
        "environmentTextureDriving": (
            _trace_to_texture(world.node_tree, bg.inputs["Color"]).name
            if _trace_to_texture(world.node_tree, bg.inputs["Color"]) is not None else None),
        **_colour_state(world, bg),
    }

    used_hdri = None
    if hdri:
        path = hdri_path
        tex = next((n for n in world.node_tree.nodes if n.type == "TEX_ENVIRONMENT"), None)
        if tex is None:
            tex = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
            tex.location = (bg.location[0] - 320, bg.location[1])
        tex.image = bpy.data.images.load(path, check_existing=True)
        world.node_tree.links.new(tex.outputs["Color"], bg.inputs["Color"])
        used_hdri = path
    elif col is not None:
        # Unlink any environment texture first, or the flat colour is written and then overridden
        # by the texture that is still plugged in - a change that reports success and does nothing.
        for link in list(world.node_tree.links):
            # See the note in _surface_source: `is` fails on NodeLink node references.
            if link.to_node == bg and link.to_socket.name == "Color":
                world.node_tree.links.remove(link)
        bg.inputs["Color"].default_value = col_vals

    # ROTATION, AND IT USED TO LIVE INSIDE `if hdri:`. So it only ever applied in a call that ALSO
    # loaded an HDRI: sending {rotation: 1.5} on its own to a world whose HDRI was already loaded
    # returned ok:true and turned nothing, while the identical value sent beside an hdriPath worked.
    # Accepted-and-ignored, which is the shape audit_blender_dead_params exists to find. Found by
    # the second seat 2026-09-15 (B1266, their Defect G).
    #
    # IT NEEDS AN ENVIRONMENT TEXTURE TO TURN, and that is the honest refusal rather than a silent
    # no-op: a world showing a flat colour has nothing to rotate, and writing a Mapping node into it
    # would build shader nodes that affect nothing. Refused by name, saying what the world actually
    # has - the same shape set_light_influence uses for a flag belonging to another light type.
    if rot is not None:
        # checked above, before any write: an hdri in this call or one already in the tree
        tex_now = next((n for n in world.node_tree.nodes if n.type == "TEX_ENVIRONMENT"), None)
        # THE MAPPING WIRED INTO THE TEXTURE, the one world_info reads: the first Mapping anywhere could be a stray one
        # wired to nothing, and a rotation written there turns nothing (found with world_info's, 2026-10-26).
        mapping = _trace_to_texture(world.node_tree, tex_now.inputs["Vector"], types=("MAPPING",))
        if mapping is None:
            mapping = world.node_tree.nodes.new("ShaderNodeMapping")
            coord = world.node_tree.nodes.new("ShaderNodeTexCoord")
            mapping.location = (tex_now.location[0] - 220, tex_now.location[1])
            coord.location = (mapping.location[0] - 220, mapping.location[1])
            world.node_tree.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
            world.node_tree.links.new(mapping.outputs["Vector"], tex_now.inputs["Vector"])
        mapping.inputs["Rotation"].default_value[2] = rot

    if strength is not None:
        bg.inputs["Strength"].default_value = strength

    sc = bpy.context.scene
    has_vis = hasattr(sc.world, "cycles_visibility")
    if asked_light:
        if hasattr(sc.world, "cycles_visibility"):
            sc.world.cycles_visibility.diffuse = want_light
            sc.world.cycles_visibility.glossy = want_light
    if want_mist is not None:
        sc.world.mist_settings.use_mist = want_mist
    if ms is not None:
        sc.world.mist_settings.start = ms
    if md is not None:
        sc.world.mist_settings.depth = md
    sun_before = _sun_state(world)
    for attr, val in sun_writes:
        setattr(world, attr, val)

    return {
        "world": world.name,
        "before": before,
        "after": {
            "color": rnd(list(bg.inputs["Color"].default_value)[:3]),
            "strength": round(float(bg.inputs["Strength"].default_value), 6),
            "hdri": used_hdri,
            # PRESENT vs DRIVING, and they are not the same question. The flat-colour branch
            # above removes the LINK into Background.Color and leaves the TEX_ENVIRONMENT node
            # in the tree, so the old single `hasEnvironmentTexture` stayed true forever once an
            # HDRI had ever been set - reporting an environment in play when the render used the
            # flat colour. In a node tree the effect lives on the link; see world_info.
            "environmentTexturePresent": any(n.type == "TEX_ENVIRONMENT"
                                             for n in world.node_tree.nodes),
            "environmentTextureDriving": (
                _trace_to_texture(world.node_tree, bg.inputs["Color"]).name
                if _trace_to_texture(world.node_tree, bg.inputs["Color"]) is not None else None),
            **_colour_state(world, bg),
            "mist": bool(sc.world.mist_settings.use_mist),
            # THE SUN AND PROBE RESOLUTION, read back on a build that has them - None on 3.6.
            "sun": _sun_state(world),
            # useAsLight WRITES cycles_visibility and nothing reported it, so the inert-parameter
            # probe read the flag as doing nothing. None when this build has no cycles_visibility
            # to report - see useAsLightNote.
            "useAsLight": (bool(sc.world.cycles_visibility.diffuse
                                and sc.world.cycles_visibility.glossy) if has_vis else None),
        },
        "useAsLightNote": ("this Blender has no world cycles_visibility (Cycles is not loaded), so "
                           "useAsLight had nothing to set and NOTHING was changed by it.")
        if (asked_light and not has_vis) else None,
        # A CYCLES SETTING, AND THE SCENE MAY NOT BE RENDERING WITH CYCLES. Reported by the second seat on 5.2.2
        # (mailbox id 157) and MEASURED 2026-09-16 on 3.6.23-5.2.1 (scratchpad measure_engine_inert_light_world.py):
        # a white world lighting a white sphere, cycles_visibility diffuse and glossy off - 115.6-120.8/255 changed in
        # Cycles, 0.000 in EEVEE. Still written, since a scene set up under EEVEE may be rendered in Cycles.
        "inertUnderThisEngine": (["useAsLight"] if (asked_light and has_vis
                                                    and "EEVEE" in bpy.context.scene.render.engine) else None),
        "engineNote": (("useAsLight writes the world's Cycles ray visibility, which %s does not read - turning it "
                        "off changed no pixel in EEVEE on any build measured (3.6-5.2). It is stored and applies "
                        "under Cycles." % bpy.context.scene.render.engine)
                       if (asked_light and has_vis and "EEVEE" in bpy.context.scene.render.engine) else None),
        "sunBefore": sun_before if sun_writes else None,
        "sunNote": ("the sun threshold acts after the Background node, so a strength change moves the extracted sun: "
                    "retune the threshold after the strength. 0 keeps all of an HDRI's light as skylight, for a scene "
                    "whose sun is a lamp of its own.") if sun_writes else None,
        "strengthNote": ("strength multiplies the background's emission. 1.0 with a mid grey is "
                         "roughly an overcast day and will wash out a deliberately dark interior - "
                         "a dim room usually wants 0.02 to 0.1."),
    }


_INFO_KEYS = {"name"}

_CREATE_KEYS = {"name", "world", "useAsSceneWorld", "color", "strength"}

_MAX_TRACE = 32


def _output_node(world):
    """The OUTPUT_WORLD node, if there is one. Pure - creates nothing."""
    return next((n for n in world.node_tree.nodes if n.type == "OUTPUT_WORLD"), None)


def _surface_source(world):
    """The node actually feeding the world output's Surface socket, or None.

    NOT the same question as "is there a Background node", and the difference is the whole reason
    this exists. _background_node returns the first BACKGROUND node in the tree whether or not it is
    WIRED to anything, so a world whose Background node has been disconnected accepts every write,
    reads every value back correctly, and renders as if none of it happened. That is the shape
    02_GOTCHAS calls right-looking and inert, and in a node tree the effect always lives on the LINK
    rather than on the node.
    """
    out = _output_node(world)
    if out is None:
        return None
    for link in world.node_tree.links:
        # `==` NOT `is`. bpy re-wraps non-ID sub-structs, so a NodeLink's .to_node/.from_node is a
    # DIFFERENT Python object from the same node reached through tree.nodes - measured on
    # Blender 5.0: `link.to_node is out` False, `link.to_node == out` True, different id().
    # ID datablocks (objects, meshes, cameras) ARE interned and compare fine with `is`; node
    # graph pieces are not. ops_nodes.py had already found this and the knowledge had not
    # travelled.
        if link.to_node == out and link.to_socket.name == "Surface":
            return link.from_node
    return None


def _find_background(world):
    """(node, connected). A BACKGROUND node and whether it actually drives the world output.

    PURE, unlike _background_node, which CREATES a Background node and wires it when the tree has
    none. That is right for a setter and disqualifying for a reader: an info op that silently
    authors two nodes changes the file it was asked to describe, and the answer it then returns is
    about the world it just made rather than the one that was there.
    """
    src = _surface_source(world)
    if src is not None and src.type == "BACKGROUND":
        return src, True
    node = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
    return node, False


def _colour_state(world, bg):
    """What actually drives the Background's Colour - {colorLinked, colorDrivenBy, skyTextureDriving}.

    A LINKED COLOUR SOCKET RENDERS ITS LINK, NEVER ITS VALUE. world_info reported the Colour socket's
    default_value as `color` whatever drove it, and judged light from it: a world whose Background was fed by
    a Sky Texture, with the socket's own value left black, came back contributesLight:false under "the
    background colour is black and no environment texture drives it" - while a cube lit by that world alone
    rendered at a mean luminance of 0.51 (measured on 5.0.1, 2026-09-20). A Sky Texture is how most outdoor
    and ocean scenes are lit, and environmentTextureDriving only ever looked for an HDRI.
    """
    sock = bg.inputs["Color"]
    src = next((l.from_node for l in world.node_tree.links if l.to_socket == sock), None)
    sky = _trace_to_texture(world.node_tree, sock, types=("TEX_SKY",))
    return {"colorLinked": src is not None,
            "colorDrivenBy": ("%s (%s)" % (src.name, src.type)) if src is not None else None,
            "skyTextureDriving": sky.name if sky is not None else None}


def _finite_colour(col, tail):
    """[r, g, b, 1.0] from a caller's colour, every channel finite - or a refusal ending with `tail`.

    float() WAS BARE HERE, in both set_world and create_world: [NaN, 0, 0] was stored and read back nan,
    and ["abc", 0, 0] died as a raw ValueError with a traceback - after create_world had already made its
    World, which stayed in the file holding the name. Measured on 5.0.1, 2026-09-20.
    """
    if not isinstance(col, (list, tuple)) or len(col) < 3:
        raise MifOpError("colour must be [r,g,b], got %r. %s" % (col, tail))
    try:
        vals = finite_floats(list(col)[:3], "color")
    except MifOpError as exc:
        raise MifOpError("%s %s" % (exc, tail))
    return vals + [1.0]


def _trace_to_texture(tree, socket, seen=None, depth=0, types=("TEX_ENVIRONMENT",)):
    """Walk BACKWARDS from an input socket to the TEX_ENVIRONMENT actually driving it, if any.

    `types` widens the search (2026-09-20): _colour_state asks for TEX_SKY with the same walk.

    A caller can put a Mapping, a Mix or a colour ramp between the environment texture and the
    Background node, so "is the texture linked directly" is the wrong test - it answers no for a
    perfectly normal graph. Following the links answers the question that matters, which is whether
    any environment texture reaches this socket at all.

    WHAT ACTUALLY GUARANTEES TERMINATION IS THE DEPTH LIMIT, not the seen-set, and that was
    established rather than assumed: a plant removing the seen-set on 2026-09-03 changed nothing -
    the suite stayed green because a cycle simply recurses until depth exceeds _MAX_TRACE. The
    seen-set is a WORK SAVER, stopping a diamond-shaped graph from being re-walked once per path.
    Both are kept; the comment used to credit the wrong one.

    A KNOWN AND BOUNDED LIMITATION FROM COMBINING THEM: a node first reached by a long path is
    marked seen and may have been cut off by the depth limit there, so a shorter second path to the
    same node is not re-explored. That can miss a texture more than _MAX_TRACE nodes deep on one
    route and shallow on another. It needs a graph over 32 nodes deep on the first route tried,
    which no world tree this addon builds comes near, and the failure is a null - reported as "no
    environment texture driving this" - rather than a wrong name.

    THE TREE IS PASSED IN rather than reached through socket.id_data. For a node tree embedded in a
    World, what id_data resolves to is exactly the kind of API detail that cannot be checked from
    here - there is no Blender in the offline suite to ask - and guessing it would put an
    unverifiable assumption underneath every answer this module gives. The caller always has the
    tree already.
    """
    if seen is None:
        seen = set()
    if depth > _MAX_TRACE:
        return None
    for link in tree.links:
        if link.to_socket != socket:   # `is` fails on re-wrapped sockets, see _surface_source
            continue
        node = link.from_node
        if node.name in seen:
            continue
        seen.add(node.name)
        if node.type in types:
            return node
        for inp in node.inputs:
            got = _trace_to_texture(tree, inp, seen, depth + 1, types)
            if got is not None:
                return got
    return None


def op_world_info(params):
    """What the world IS - the read half of set_world, which had none, for the op that decides black.

    THE FAMILY WAS WRITE-ONLY. set_world could set a colour, a strength, an HDRI, a rotation and
    mist, and nothing anywhere could read any of it back: scene_info omits the world entirely and
    render_info reports only its NAME. So "what is my world actually set to" was unanswerable
    through the typed path, on the one datablock that decides whether an interior renders black.

    EVERY ANSWER HERE IS TAKEN FROM THE LINK, NOT THE NODE, because a shader tree's effect lives on
    its connections:

      * The Background node is reported with `backgroundConnected`, because a Background node that
        is not wired to the world output accepts every write and contributes nothing.
      * The environment texture is reported as `environmentTextureDriving`, found by walking
        backwards from the Background's Colour socket - so a Mapping or a Mix in between still
        counts, and a texture sitting unlinked in the tree correctly does not. set_world's old
        `hasEnvironmentTexture` asked only whether the NODE existed, which stays true forever after
        an HDRI is replaced by a flat colour: the link is removed and the node is left behind.
      * `useNodes` is reported because a world with use_nodes False IGNORES THE WHOLE TREE and
        renders its flat `world.color` instead. Every node in it reads perfectly and none applies.

    `contributesLight` is the diagnosis rather than the inputs to it, matching render_info.

    params:
      name (str)   a specific world by name. Default the scene's world.
    """
    reject_unknown(params, _INFO_KEYS, "world_info")
    name = take(params, "name", kind=str)
    sc = bpy.context.scene
    if name:
        world = bpy.data.worlds.get(name)
        if world is None:
            known = sorted(w.name for w in bpy.data.worlds)
            raise MifOpError("no world named '%s'. Present: %s."
                             % (name, ", ".join(known) if known else "<none>"))
    else:
        world = sc.world

    if world is None:
        # NOT AN ERROR. "There is no world" is a real and common state, it is the single most
        # common cause of a black interior, and refusing here would make the op unable to report
        # the very thing it is most useful for.
        return {
            "ok": True,
            "world": None,
            "isSceneWorld": False,
            "contributesLight": False,
            "worldsInFile": sorted(w.name for w in bpy.data.worlds),
            "blockers": ["scene '%s' has NO world datablock, so it contributes no ambient light at "
                         "all - an interior renders pure black outside its own fixtures and the "
                         "lights usually get blamed. Create one with set_world." % sc.name],
        }

    use_nodes = bool(world.use_nodes)
    out = {
        "ok": True,
        "world": world.name,
        "isSceneWorld": sc.world is not None and sc.world.name == world.name,
        "useNodes": use_nodes,
        "worldsInFile": sorted(w.name for w in bpy.data.worlds),
        # THE FLAT COLOUR, which is what actually renders when useNodes is False.
        "flatColor": rnd(list(world.color)[:3]) if hasattr(world, "color") else None,
    }

    blockers = []
    bg = None
    if not use_nodes:
        blockers.append("use_nodes is FALSE on this world, so its entire node tree is IGNORED and "
                        "the flat world.color renders instead. Every node in it reads correctly "
                        "and none of it applies.")
    else:
        bg, connected = _find_background(world)
        surface = _surface_source(world)
        out["nodeCount"] = len(world.node_tree.nodes)
        out["surfaceDrivenBy"] = surface.type if surface is not None else None
        out["backgroundConnected"] = bool(connected)
        if bg is None:
            blockers.append("there is no Background node in this world's tree at all - nothing "
                            "drives the world output, so it contributes no light.")
        else:
            colour_socket = bg.inputs["Color"]
            tex = _trace_to_texture(world.node_tree, colour_socket)
            strength = round(float(bg.inputs["Strength"].default_value), 6)
            out["color"] = rnd(list(colour_socket.default_value)[:3])
            # `color` IS THE SOCKET'S OWN VALUE, which renders only while nothing is linked into it - see
            # _colour_state. colorLinked says whether that is so; colorDrivenBy names what renders instead.
            out.update(_colour_state(world, bg))
            if out["colorLinked"]:
                out["colorNote"] = ("the Background's Colour is LINKED from %s, so `color` above is the socket's "
                                    "unused value, not what renders." % out["colorDrivenBy"])
            out["strength"] = strength
            out["environmentTexturePresent"] = any(n.type == "TEX_ENVIRONMENT"
                                                   for n in world.node_tree.nodes)
            out["environmentTextureDriving"] = tex.name if tex is not None else None
            out["hdri"] = (getattr(tex.image, "filepath", None) if tex is not None
                           and tex.image is not None else None)
            if tex is not None and tex.image is None:
                blockers.append("the environment texture '%s' drives the background but has NO "
                                "image assigned - it contributes black." % tex.name)
            # THE MAPPING WIRED INTO THE DRIVING TEXTURE, not the first Mapping node anywhere in the tree: a stray one,
            # wired to nothing, answered its rotation while the sky turned by another (found 2026-10-25).
            mapping = (_trace_to_texture(world.node_tree, tex.inputs["Vector"], types=("MAPPING",))
                       if tex is not None else None)
            out["rotation"] = (round(float(mapping.inputs["Rotation"].default_value[2]), 6)
                               if mapping is not None else None)
            out["rotationMapping"] = mapping.name if mapping is not None else None
            if not connected:
                blockers.append("the Background node exists but is NOT connected to the world "
                                "output, so every value on it is inert - it accepts writes and "
                                "changes no light.")
            if strength == 0.0:
                blockers.append("world strength is 0, so the background emits nothing regardless "
                                "of its colour.")
            elif (not out["colorLinked"]
                  and all(c <= 0.0 for c in out["color"])):
                # ONLY AN UNLINKED BLACK IS BLACK. A linked socket renders its link - a Sky Texture, an HDRI, a
                # Mix - whatever its own value says, and judging light from that value called a sky-lit world
                # dark (see _colour_state).
                blockers.append("the background colour is black and nothing is linked into it, so the "
                                "world contributes no light. strength will not help.")

    mist = getattr(world, "mist_settings", None)
    out["mist"] = {
        "use": bool(mist.use_mist) if mist else None,
        "start": round(float(mist.start), 6) if mist else None,
        "depth": round(float(mist.depth), 6) if mist else None,
    }
    # How far Cycles' AO looks for occluders - an AO bake reads it (bake_texture aoDistance sets it for one bake).
    ls = getattr(world, "light_settings", None)
    out["aoDistance"] = round(float(ls.distance), 6) if ls is not None else None
    vis = getattr(world, "cycles_visibility", None)
    out["cyclesVisibility"] = ({"diffuse": bool(vis.diffuse), "glossy": bool(vis.glossy)}
                               if vis is not None else None)
    if vis is not None and not vis.diffuse and not vis.glossy:
        blockers.append("cycles_visibility has diffuse and glossy both off (set_world's "
                        "useAsLight:false), so the world is visible in the background but lights "
                        "nothing.")

    out["blockers"] = blockers
    out["contributesLight"] = not blockers
    return out


def op_create_world(params):
    """Make a World datablock.

    WHY THIS EXISTS, and it was found twice on one day by two tools that do not share any code.
    audit_blender_consumer_no_creator reported `worlds` as a collection a caller can name and nobody
    creates; audit_blender_demanded_kind_has_creator reported the same thing from the other end, by
    asking the running addon which datablock kinds its refusals demand. world was the only one of
    seven with no creator.

    It became VISIBLE that day rather than becoming true: set_world's `name` used to be dropped on
    any scene that already had a world, so nothing indexed bpy.data.worlds by a caller-supplied key
    and the collection was not "reached". Fixing that made the gap detectable - which is the useful
    thing about a fix landing next to an audit that watches the same shape.

    set_world REMAINS a setter. It edits the world you name and refuses one that is not there, which
    is the convention every set_* op in this addon follows; making it create on a miss would have
    turned a typo into a new datablock. So the gap was a missing create_*, and this is it.

    params:
      name / world (str)       required. The datablock name.
      useAsSceneWorld (bool)   default false. A World that is not the scene's contributes NO light -
                               it is a datablock sitting in the file - so this says out loud which
                               one you get, rather than leaving a caller to wonder why the render
                               did not change.
      color [r,g,b]            optional background colour, applied through the same Background node
                               set_world uses, so the two agree about where a colour lives.
      strength (float)         optional emission multiplier. See this module's header: 1.0 is
                               roughly an overcast day and will wash out a dark interior.
    """
    reject_unknown(params, _CREATE_KEYS, "create_world")
    name = take(params, "name", "world", required=True, kind=str)

    # REFUSED, not silently reused. A caller asking to CREATE a world that is already there has
    # either lost track of it or meant set_world, and handing back the existing one would make
    # "created" a lie in the response. create_collection refuses the same way for the same reason.
    if bpy.data.worlds.get(name) is not None:
        raise MifOpError("a world named '%s' already exists - edit it with set_world{name} or pick "
                         "another name. NOTHING was created." % name)

    col = params.get("color")
    # EVERY CHANNEL FINITE, CHECKED BEFORE THE WORLD EXISTS. A string channel raised a raw ValueError AFTER
    # bpy.data.worlds.new, leaving a World holding the name - see _finite_colour.
    col_vals = _finite_colour(col, "NOTHING was created.") if col is not None else None
    strength = take_float(params, "strength", default=None)
    as_scene = take_bool(params, "useAsSceneWorld", default=False)

    world = bpy.data.worlds.new(name)
    world.use_nodes = True
    bg = _background_node(world)          # creates and WIRES it - an unwired one is inert
    if col_vals is not None:
        bg.inputs["Color"].default_value = col_vals
    if strength is not None:
        bg.inputs["Strength"].default_value = float(strength)

    sc = bpy.context.scene
    if as_scene:
        sc.world = world

    return {
        "world": world.name,
        "requestedName": name,
        # THE CANONICAL SPELLING, which audit_created_name_reported requires of every op that names
        # a datablock it created. Blender uniquifies on collision, and a caller that is not told
        # will address the wrong datablock afterwards.
        "nameWasSuffixed": world.name != name,
        "useNodes": True,
        "isSceneWorld": sc.world is world,
        "sceneWorldNote": (None if sc.world is world else
                           "this world is NOT the scene's, so it contributes no light until "
                           "useAsSceneWorld:true or set_world{name} selects it"),
        "color": rnd(list(bg.inputs["Color"].default_value)[:3]),
        "strength": round(float(bg.inputs["Strength"].default_value), 6),
    }


_LIST_WORLDS_KEYS = ("limit", "nameExcludes")


def op_list_worlds(params):
    """Every world datablock in the file, by name, and which the current scene uses.

    THE FOURTH COLLECTION OF THIS SHAPE. create_world makes one, ops take a world NAME, and the only
    enumeration of world names in the addon was inside world_info's own refusal - `known = sorted(
    w.name for w in bpy.data.worlds)` - which is discovery by deliberately asking for one that does
    not exist, exactly as images and node_groups were before 2026-09-08.

    users and fakeUser are reported for the reason list_images reports them: a world with neither is
    one the next save DELETES, and a caller choosing between worlds is precisely who needs to know
    that before choosing.
    """
    reject_unknown(params, _LIST_WORLDS_KEYS, "list_worlds",
                   notes={
                       "name": "this LISTS worlds; world_info reads ONE, and takes the name",
                       "world": "see `name` - world_info is the reader, this is the enumerator",
                       "current": "each row carries its own `current` flag; there is nothing to pass",
                   })
    scene = getattr(bpy.context, "scene", None)
    in_use = scene.world.name if (scene and scene.world) else None
    rows = []
    for w in bpy.data.worlds:
        rows.append({
            "name": w.name,
            "current": w.name == in_use,
            "usesNodes": bool(getattr(w, "use_nodes", False)),
            "users": int(getattr(w, "users", 0)),
            "fakeUser": bool(getattr(w, "use_fake_user", False)),
        })
    rows.sort(key=lambda r: r["name"])
    rows = exclude_rows(rows, params)
    matched = len(rows)
    rows, extra = limit_rows(rows, params, "world(s)")
    return dict({"worlds": rows, "count": matched, "current": in_use}, **extra)


OPS = {
    "list_worlds": op_list_worlds,
    "create_world": op_create_world,
    "set_world": op_set_world,
    "world_info": op_world_info,
}
