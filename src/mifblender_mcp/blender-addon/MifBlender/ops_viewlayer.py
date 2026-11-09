"""View layers and render passes - what the compositor is given to work with.

WHY THIS FOLLOWS THE COMPOSITOR. compositor_info reports which passes are enabled and nothing could
turn one on, which is the same read/write asymmetry that turned up world and physics - just facing
the other way. It matters more here than a missing setter usually does, because a Render Layers node
only offers sockets for passes the view layer ACTUALLY OUTPUTS: ask for a Z-depth composite with the
Z pass off and there is no socket to connect, or in some versions a socket that outputs black. The
compositing ops shipped without the one thing that decides what they can see.

=============================================================================
A VIEW LAYER WITH use OFF IS THE INERT SHAPE AGAIN
=============================================================================
`view_layer.use` decides whether the layer is rendered at all. With it off, every pass, every
collection assignment, every override on that layer reads back perfectly and no pixel of it is ever
produced. Nothing warns. So it is reported as `renders` and called out as a blocker rather than
being left as one boolean among many.

=============================================================================
THE PASS SET IS NOT FIXED, SO IT IS NOT HARD-CODED
=============================================================================
Which use_pass_* properties exist depends on the Blender version and the render engine - Cycles
offers passes EEVEE does not, cryptomatte moved, and new ones arrive. So the accepted names are
enumerated from the view layer in front of us, the same reasoning as set_color_management reading
its enums from the instance rather than from bpy.types. A remembered list goes wrong quietly, by
refusing something that works.
"""
import bpy

from .ops_common import MifOpError, limit_rows, reject_unknown, take, take_bool, take_int, write_rna_property

# NO "scene" KEY. It was declared here reflexively and never read - every op in this module works
# on bpy.context.scene, and an accepted parameter that is silently ignored is worse than an absent
# one. Second time this session: render_animation carried the same dead key and param_reach caught
# both. If cross-scene addressing is wanted it is a real piece of work, not a key.
_LIST_KEYS = {"withPasses", "limit"}
_SET_KEYS = {"name", "viewLayer", "use", "enablePasses", "disablePasses", "passes", "samples",
             # THE LAYER'S ENGINE SETTINGS AND EXTRAS (2026-10-04): view_layer.cycles / view_layer.eevee by identifier
             # (denoising_store_passes, the volume passes), the Cryptomatte depth, light groups, AOVs and the
             # material override - none of which use_pass_* reaches
             "cycles", "eevee", "cryptomatteDepth", "lightgroups", "aovs", "materialOverride"}
_CREATE_KEYS = {"name", "copyFrom", "use"}
_DELETE_KEYS = {"name", "viewLayer"}

_PREFIX = "use_pass_"


def _pass_names(vl):
    """Every pass this view layer actually offers, without the use_pass_ prefix.

    ENUMERATED, NOT LISTED. The set differs by Blender version and by render engine, so a constant
    in this file would refuse passes that exist and accept ones that do not - and the failure would
    be a refusal of something legal, which reads like a bug in Blender rather than in here.
    """
    out = {}
    for attr in dir(vl):
        if not attr.startswith(_PREFIX):
            continue
        try:
            out[attr[len(_PREFIX):]] = bool(getattr(vl, attr))
        except (AttributeError, TypeError):
            continue
    return out


def _find_layer(scene, name):
    vl = scene.view_layers.get(name)
    if vl is None:
        known = sorted(v.name for v in scene.view_layers)
        raise MifOpError("no view layer named '%s' in scene '%s'. Present: %s. NOTHING was changed."
                         % (name, scene.name, ", ".join(known)))
    return vl


def _layer_row(vl, scene, with_passes):
    passes = _pass_names(vl)
    row = {
        "name": vl.name,
        # THE ONE THAT DECIDES WHETHER ANY OF THE REST HAPPENS.
        "renders": bool(getattr(vl, "use", True)),
        "isActive": bpy.context.view_layer.name == vl.name,
        "enabledPasses": sorted(k for k, v in passes.items() if v),
        "passCount": len(passes),
        "samples": getattr(vl, "samples", None),
        # THE READ TWIN OF set_view_layer's extras (2026-10-04)
        "cryptomatteDepth": getattr(vl, "pass_cryptomatte_depth", None),
        "materialOverride": vl.material_override.name if getattr(vl, "material_override", None) else None,
        "aovs": [{"name": a.name, "type": a.type} for a in getattr(vl, "aovs", [])],
        "lightgroups": [g.name for g in getattr(vl, "lightgroups", [])],
    }
    if with_passes:
        row["passes"] = passes
    return row


def op_list_view_layers(params):
    """Every view layer, what it outputs, and whether it renders at all.

    THE BIGGEST UNBOUNDED RESPONSE IN THE ADDON, MEASURED. `withPasses` adds the full on/off map -
    about 60 entries - to EVERY row, and there was no way to ask for fewer rows. Measured 2026-09-20
    on 5.0.1: 61 view layers answered 8,960 bytes plain and 52,880 bytes with passes, roughly 144 and
    867 bytes a layer, growing with the SCENE rather than with anything the caller sent. A per-shot
    or per-character layer setup reaches dozens without being unusual. `limit` is the same optional,
    unlimited-by-default shape the other 22 listings use, and `count` stays the true total.

    params:
      withPasses (bool)   include the full on/off map per layer, not just the enabled names.
      limit (int)         how many rows to return. Unlimited by default; `count` is always the
                          true total and `truncated` says when rows were dropped.
    """
    reject_unknown(params, _LIST_KEYS, "list_view_layers")
    sc = bpy.context.scene
    with_passes = take_bool(params, "withPasses", default=False)
    rows = [_layer_row(vl, sc, with_passes) for vl in sc.view_layers]
    # COMPUTED BEFORE THE LIMIT, and from the full set. notRendering is a warning about the scene,
    # not about the page - a caller handed the first 10 of 60 layers must still learn that layer 47
    # renders nothing, because that is the blocker this op exists to surface.
    dark = [r["name"] for r in rows if not r["renders"]]
    total = len(rows)
    rows, extra = limit_rows(rows, params, "view layer(s)", excludes=False)
    return dict({
        "ok": True,
        "scene": sc.name,
        "activeViewLayer": bpy.context.view_layer.name,
        "count": total,
        "viewLayers": rows,
        "notRendering": dark,
        "note": ("%d view layer(s) have use OFF: every pass and collection assignment on them reads "
                 "back correctly and no pixel of them is ever produced." % len(dark)) if dark else None,
    }, **extra)


# ---- the layer's engine settings and extras (2026-10-04) ----------------------------------------------------------
# MEASURED 2026-10-04 on 3.6.23, 4.4.0 and 5.2.1: view_layer.cycles holds denoising_store_passes (the Denoising Data
# passes a Denoise node takes as Albedo and Normal) and the volume passes, view_layer.eevee use_pass_bloom /
# _transparent / _volume_direct (5.2 adds ambient_occlusion_distance); pass_cryptomatte_depth is 2-16, default 6;
# lightgroups.add makes "Key_001" for a name that exists, and 3.6 has NO lightgroups.remove - so a light group is added
# only when missing, and LAST, after every other write has landed; aovs.add makes "AOV" COLOR, a name that exists
# becomes "Mask_001", and aovs.remove exists on every build.
_LAYER_GROUPS = ("cycles", "eevee")


def _extras_any(extras):
    return bool(extras["groups"] or extras["crypto"] is not None or extras["lightgroups"] or extras["aovs"]
                or extras["overrideGiven"])


def _extras_wanted(vl, params):
    """Everything the extras keys ask for, checked before the first write of the call."""
    out = {"groups": [], "crypto": None, "lightgroups": [], "aovs": [], "override": None, "overrideGiven": False}
    for group in _LAYER_GROUPS:
        want = params.get(group)
        if want is None:
            continue
        holder = getattr(vl, group, None)
        if holder is None:
            raise MifOpError("this Blender (%s) has no view_layer.%s, so `%s` has nothing to write on. NOTHING was "
                             "changed." % (bpy.app.version_string, group, group))
        if not isinstance(want, dict) or not want:
            raise MifOpError("`%s` takes a non-empty object of identifier -> value, got %r. NOTHING was changed."
                             % (group, want))
        writable = sorted(p.identifier for p in holder.bl_rna.properties
                          if not p.is_readonly and not p.identifier.startswith("bl_")
                          and p.identifier not in ("rna_type", "name"))
        for key, value in want.items():
            if key not in writable:
                raise MifOpError("view_layer.%s has no writable setting %r on Blender %s. It has: %s. NOTHING was "
                                 "changed." % (group, key, bpy.app.version_string, ", ".join(writable) or "(none)"))
            out["groups"].append((group, holder, key, value))
    if params.get("cryptomatteDepth") is not None:
        prop = vl.bl_rna.properties["pass_cryptomatte_depth"]
        depth = take_int(params, "cryptomatteDepth")
        if not prop.hard_min <= depth <= prop.hard_max:
            raise MifOpError("cryptomatteDepth %d is outside %d-%d (Blender's own range). NOTHING was changed."
                             % (depth, prop.hard_min, prop.hard_max))
        out["crypto"] = depth
    raw = params.get("lightgroups")
    if raw is not None:
        if not isinstance(raw, (list, tuple)) or not raw or not all(isinstance(n, str) and n.strip() for n in raw):
            raise MifOpError("lightgroups takes a non-empty list of names, got %r. NOTHING was changed." % (raw,))
        out["lightgroups"] = list(dict.fromkeys(str(n) for n in raw))
    raw = params.get("aovs")
    if raw is not None:
        types = [i.identifier for i in bpy.types.AOV.bl_rna.properties["type"].enum_items]
        if not isinstance(raw, (list, tuple)) or not raw:
            raise MifOpError("aovs takes a list of {name, type %s}, got %r. NOTHING was changed."
                             % (" | ".join(types), raw))
        seen = set()
        for item in raw:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip() \
                    or set(item) - {"name", "type"}:
                raise MifOpError("each aov is {name, type %s}, got %r. NOTHING was changed." % (" | ".join(types), item))
            kind = str(item.get("type", "COLOR")).upper()
            if kind not in types:
                raise MifOpError("aov type %r is not one this Blender has. Valid: %s. NOTHING was changed."
                                 % (item.get("type"), ", ".join(types)))
            if item["name"] in seen:
                raise MifOpError("aov '%s' is named twice in one call. NOTHING was changed." % item["name"])
            seen.add(item["name"])
            out["aovs"].append((item["name"], kind))
    if "materialOverride" in params:
        out["overrideGiven"] = True
        name = params.get("materialOverride")
        if name is not None:
            if not isinstance(name, str):
                raise MifOpError("materialOverride takes a material name, or null to clear it; got %r. NOTHING was "
                                 "changed." % (name,))
            mat = bpy.data.materials.get(name)
            if mat is None:
                have = sorted(m.name for m in bpy.data.materials)[:25]
                raise MifOpError("no material named '%s' for materialOverride. Present: %s. NOTHING was changed."
                                 % (name, ", ".join(have) or "(none)"))
            out["override"] = mat
    return out


def _extras_snapshot(vl, extras):
    # getattr throughout: only what the call writes is touched, and a layer without one of these (a stand-in in the
    # offline refusal suite, B118) must not fail a call that never asked for it
    return {"groups": [(h, k, getattr(h, k)) for _g, h, k, _v in extras["groups"]],
            "crypto": getattr(vl, "pass_cryptomatte_depth", None),
            "override": getattr(vl, "material_override", None),
            "aovs": {a.name: a.type for a in getattr(vl, "aovs", [])},
            "lightgroups": [g.name for g in getattr(vl, "lightgroups", [])]}


def _extras_write(vl, extras):
    """Write the extras in a fixed order: the undoable ones, then the group values (checked by type as they are written,
    so they can still refuse and put the rest back), and light groups LAST - 3.6 cannot remove one."""
    applied = {}
    tail = "NOTHING was changed - every setting this call wrote was put back."
    if extras["crypto"] is not None:
        vl.pass_cryptomatte_depth = extras["crypto"]
        applied["cryptomatteDepth"] = vl.pass_cryptomatte_depth
    if extras["overrideGiven"]:
        vl.material_override = extras["override"]
        applied["materialOverride"] = vl.material_override.name if vl.material_override else None
    if extras["aovs"]:
        made, retyped = [], []
        for name, kind in extras["aovs"]:
            aov = next((a for a in vl.aovs if a.name == name), None)
            if aov is None:
                aov = vl.aovs.add()
                aov.name = name
                made.append(name)
            if aov.type != kind:
                aov.type = kind
                retyped.append(name)
        applied["aovs"] = [{"name": a.name, "type": a.type} for a in vl.aovs]
        applied["aovsMade"] = made
        applied["aovsRetyped"] = retyped
    for group, holder, key, value in extras["groups"]:
        applied.setdefault(group, {})[key] = write_rna_property(holder, key, value, tail, "view_layer.%s" % group,
                                                                "view_layer.%s" % group, frozenset())
    if extras["lightgroups"]:
        have = {g.name for g in vl.lightgroups}
        made = [n for n in extras["lightgroups"] if n not in have]
        for n in made:
            vl.lightgroups.add(name=n)
        applied["lightgroups"] = [g.name for g in vl.lightgroups]
        applied["lightgroupsMade"] = made
        applied["lightgroupsNote"] = ("a light group collects the light of the objects and worlds whose lightgroup "
                                      "setting names it into a pass of its own - a Cycles pass")
    return applied


def _extras_restore(vl, snap):
    for holder, key, value in snap["groups"]:
        try:
            setattr(holder, key, value)
        except (AttributeError, TypeError, ValueError):     # noqa: PERF203
            pass
    if snap["crypto"] is not None and vl.pass_cryptomatte_depth != snap["crypto"]:
        vl.pass_cryptomatte_depth = snap["crypto"]
    if hasattr(vl, "material_override") and vl.material_override != snap["override"]:
        vl.material_override = snap["override"]
    for aov in list(getattr(vl, "aovs", [])):
        if aov.name not in snap["aovs"]:
            vl.aovs.remove(aov)
        elif aov.type != snap["aovs"][aov.name]:
            aov.type = snap["aovs"][aov.name]
    remover = getattr(getattr(vl, "lightgroups", None), "remove", None)
    if remover is not None:
        for g in list(vl.lightgroups):
            if g.name not in snap["lightgroups"]:
                remover(g)


def op_set_view_layer(params):
    """Turn render passes on or off, and decide whether the layer renders at all.

    A Render Layers node only offers sockets for passes the layer OUTPUTS, so this is what decides
    what the compositor can see. Asking for a Z-depth composite with the Z pass off leaves nothing
    to connect.

    PASS NAMES ARE VALIDATED AGAINST THIS LAYER, not a list in this file - the set depends on the
    Blender version and the render engine, and a hard-coded one would refuse passes that exist.
    Names are given without the use_pass_ prefix: "z", "normal", "mist", "cryptomatte_object".

    params:
      name / viewLayer (str)      which layer. Default the active one.
      use (bool)                  whether this layer renders AT ALL.
      enablePasses (list[str])    passes to turn on
      disablePasses (list[str])   passes to turn off
      passes (dict)               {"z": true, "mist": false} - an alternative to the two lists
      samples (int)               per-layer sample override, 0 to follow the scene
      cycles / eevee (dict)       view_layer.cycles / view_layer.eevee by identifier - denoising_store_passes
                                  (Denoising Data: the Albedo and Normal a Denoise node takes), the volume passes
      cryptomatteDepth (int)      Cryptomatte levels, 2-16 (Blender's range)
      lightgroups (list[str])     light groups to have on the layer; existing names are left as they are
      aovs (list)                 [{name, type VALUE | COLOR}] - made when missing, retyped when different
      materialOverride (str)      a material every object on the layer renders with; null clears it
    """
    reject_unknown(params, _SET_KEYS, "set_view_layer")
    sc = bpy.context.scene
    name = take(params, "name", "viewLayer", kind=str)
    vl = _find_layer(sc, name) if name else bpy.context.view_layer

    available = _pass_names(vl)
    wanted = {}
    for key, value in (("enablePasses", True), ("disablePasses", False)):
        raw = params.get(key)
        if raw is None:
            continue
        if not isinstance(raw, (list, tuple)):
            raise MifOpError("'%s' must be a list of pass names, got %s. NOTHING was changed."
                             % (key, type(raw).__name__))
        for p in raw:
            wanted[str(p)] = value
    table = params.get("passes")
    if table is not None:
        if not isinstance(table, dict):
            raise MifOpError("'passes' must be a {name: bool} map, got %s. NOTHING was changed."
                             % type(table).__name__)
        for p, v in table.items():
            # take_bool, NOT bool() (2026-09-21): bool("false") is True, so {"mist": "false"} turned the mist pass ON.
            # take_bool reads "false" as false, as every other boolean here does, and refuses "flase" naming the pass.
            key = "passes.%s" % p
            wanted[str(p)] = take_bool({key: v}, key)

    # VALIDATED BEFORE ANYTHING IS WRITTEN, all of them, so a typo in the fourth name does not leave
    # three passes changed and the call reported as a failure.
    unknown = [p for p in wanted if p not in available]
    if unknown:
        raise MifOpError("this view layer has no pass named %s. Available on this Blender and "
                         "engine: %s. Names are given without the use_pass_ prefix. NOTHING was "
                         "changed." % (", ".join(sorted(unknown)[:6]), ", ".join(sorted(available))))

    use = params.get("use")
    samples = take_int(params, "samples", default=None)
    extras = _extras_wanted(vl, params)
    if not wanted and use is None and samples is None and not _extras_any(extras):
        raise MifOpError("nothing to do - pass enablePasses, disablePasses, passes, use, samples, cycles, eevee, "
                         "cryptomatteDepth, lightgroups, aovs or materialOverride. NOTHING was changed.")
    # READ AND CHECKED BEFORE THE FIRST WRITE (2026-09-21). `use` was read after the passes were written, and the
    # samples check came after both - "Everything else requested was applied; samples was not."
    want_use = take_bool(params, "use", default=True) if use is not None else None
    if samples is not None and not hasattr(vl, "samples"):
        raise MifOpError("this view layer has no per-layer sample override on this Blender, so 'samples' cannot be "
                         "set. NOTHING was changed.")

    before = {"renders": bool(getattr(vl, "use", True)),
              "enabledPasses": sorted(k for k, v in available.items() if v)}
    samples_before = getattr(vl, "samples", None)

    extras_before = _extras_snapshot(vl, extras)

    for p, v in wanted.items():
        setattr(vl, _PREFIX + p, v)
    if want_use is not None:
        vl.use = want_use
    if samples is not None:
        vl.samples = samples
    try:
        extras_applied = _extras_write(vl, extras)
    except Exception:
        for p in wanted:
            try:
                setattr(vl, _PREFIX + p, available[p])
            except (AttributeError, TypeError):     # noqa: PERF203
                pass
        if want_use is not None:
            vl.use = before["renders"]
        if samples is not None and samples_before is not None:
            vl.samples = samples_before
        _extras_restore(vl, extras_before)
        raise

    after = _pass_names(vl)
    # READ BACK EVERY REQUESTED PASS INDIVIDUALLY. Several use_pass_* properties are read-only under
    # a given engine - the property exists, the write is accepted, and the value does not move - so
    # a response built from what was asked for would report a pass that is still off.
    wrong = {p: (v, after.get(p)) for p, v in wanted.items() if after.get(p) != v}
    if wrong:
        # PUT BACK FIRST (2026-09-21): this refusal used to leave every other pass, `use` and `samples` as written.
        for p in wanted:
            try:
                setattr(vl, _PREFIX + p, available[p])
            except (AttributeError, TypeError):     # noqa: PERF203 - the read-only pass that caused this
                pass
        if want_use is not None:
            vl.use = before["renders"]
        if samples is not None and samples_before is not None:
            vl.samples = samples_before
        _extras_restore(vl, extras_before)
        raise MifOpError("wrote %s but the layer reads back %s afterwards - some passes are "
                         "read-only under the current render engine (%s), and the write is "
                         "accepted without taking effect. Everything this call wrote was put back."
                         % ({k: v[0] for k, v in wrong.items()},
                            {k: v[1] for k, v in wrong.items()}, sc.render.engine))

    row = _layer_row(vl, sc, False)
    blockers = []
    if not row["renders"]:
        blockers.append("view layer '%s' has use OFF, so it is not rendered at all - every pass "
                        "and collection assignment on it reads back correctly and no pixel of it "
                        "is ever produced." % vl.name)
    return {
        "ok": True,
        "scene": sc.name,
        "before": before,
        "passesChanged": sorted(wanted),
        "blockers": blockers,
        **row,
        **extras_applied,
    }


def op_create_view_layer(params):
    """Add a view layer - a second pass over the same scene with its own collections and outputs.

    params:
      name (str)        required
      copyFrom (str)    copy the enabled passes from this layer. Blender's new layers start from
                        defaults, which is rarely what somebody splitting a shot wants.
      use (bool)        whether the new layer renders. Default true.
    """
    reject_unknown(params, _CREATE_KEYS, "create_view_layer")
    sc = bpy.context.scene
    name = take(params, "name", required=True, kind=str)
    if sc.view_layers.get(name) is not None:
        raise MifOpError("scene '%s' already has a view layer named '%s'. NOTHING was created."
                         % (sc.name, name))
    src = _find_layer(sc, take(params, "copyFrom", kind=str)) if params.get("copyFrom") else None
    src_passes = _pass_names(src) if src is not None else None
    # `use` READ BEFORE THE LAYER EXISTS - it was read after view_layers.new, so {use: "flase"} was refused with the
    # new layer left in the scene (measured on 5.0.1, 2026-09-21).
    try:
        want_use = take_bool(params, "use", default=True) if params.get("use") is not None else None
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was created." % str(exc).rstrip("."))

    vl = sc.view_layers.new(name)
    if src_passes is not None:
        for p, v in src_passes.items():
            try:
                setattr(vl, _PREFIX + p, v)
            except (AttributeError, TypeError):     # noqa: PERF203
                continue                            # read-only under this engine; not fatal
    if want_use is not None:
        vl.use = want_use

    made = sc.view_layers.get(name)
    if made is None:
        raise MifOpError("created view layer '%s' and it is not in scene.view_layers afterwards."
                         % name)
    return {"ok": True, "scene": sc.name, "copiedFrom": src.name if src else None,
            **_layer_row(made, sc, False)}


def op_delete_view_layer(params):
    """Remove a view layer, refusing to remove the last one.

    A scene with no view layer cannot render at all, and Blender does not stop you from getting
    there through the API.

    params:
      name / viewLayer (str)   required
    """
    reject_unknown(params, _DELETE_KEYS, "delete_view_layer")
    sc = bpy.context.scene
    name = take(params, "name", "viewLayer", required=True, kind=str)
    vl = _find_layer(sc, name)
    if len(sc.view_layers) <= 1:
        raise MifOpError("'%s' is the only view layer in scene '%s', and a scene with none cannot "
                         "be rendered at all. NOTHING was changed." % (name, sc.name))
    sc.view_layers.remove(vl)
    if sc.view_layers.get(name) is not None:
        raise MifOpError("removed view layer '%s' and it is still in scene.view_layers." % name)
    return {"ok": True, "scene": sc.name, "deleted": name,
            "remaining": sorted(v.name for v in sc.view_layers),
            "activeViewLayer": bpy.context.view_layer.name}


OPS = {
    "list_view_layers": op_list_view_layers,
    "set_view_layer": op_set_view_layer,
    "create_view_layer": op_create_view_layer,
    "delete_view_layer": op_delete_view_layer,
}
