"""Particle systems: emitters for smoke and steam, and hair for scattered debris.

WHY BOTH TYPES ARE ONE OP. Blender's two particle types are the same datablock with a switch, and
they answer two completely different questions - EMITTER throws particles over time (steam from a
pipe, sparks), HAIR instances geometry across a surface and does not move (grass, rubble, cables on
a wall). Splitting them into two ops would duplicate every shared setting; keeping them together
means the type-specific ones have to be REFUSED on the wrong type, which is the same discipline
create_light uses for SPOT and AREA.

THE INSTANCE OBJECT IS WHERE THIS GETS USEFUL, and it is also the part with a trap: setting
render_type to OBJECT without setting instance_object leaves a system that renders NOTHING and
reports no error at all. So they are validated together here rather than independently.
"""
import bpy

from .ops_common import (MifOpError, finite_float, finite_floats, finite_int, get_object, limit_rows, reject_unknown,
                         take,
                         take_bool, take_float, take_int)

# A CORE PLUS A UNION EACH WAY, and the shape is dictated by parity_check rather than taste.
# It resolves a set literal, a module-level name bound to one, and A | B - and REFUSES a
# subtraction, which is right: an accepted-key list it cannot read statically is exactly
# where drift hides. The first version of set_particles wrote (set(_PS_KEYS) - {...}) | {...}
# and the checker failed it by name, so the core moved out instead.
#
# systemName NAMES a system being created; system ADDRESSES one that exists. Neither op
# accepts the other's, so a caller sending systemName to set_particles is refused rather
# than having it silently ignored.
_PS_CORE = {
    "object", "name", "type", "count", "seed",
    "frameStart", "frameEnd", "lifetime", "lifetimeRandom",
    "emitFrom", "distribution", "useModifierStack",
    "physicsType", "normalFactor", "randomFactor", "gravityFactor", "dampingFactor",
    "size", "sizeRandom", "renderType", "instanceObject", "instanceCollection",
    "hairLength", "childCount", "showEmitter", "rotationMode", "useRotations",
    "children", "material", "collection", "rotation", "groups", "vertexGroupDensity",
    # VELOCITY ALONG THE OBJECT'S AXES, HAIR DYNAMICS AND THE STRAND'S SHAPE (2026-10-05)
    "objectAlignFactor", "objectFactor", "hairDynamics", "strandShape",
}
_PS_KEYS = _PS_CORE | {"systemName"}
_LIST_KEYS = {"object", "name", "limit"}


def _enum(rna_type, prop):
    return {i.identifier for i in rna_type.bl_rna.properties[prop].enum_items}


def _apply_particle_settings(obj, psys, params, kind):
    """Every particle setting, written once and called from BOTH ops.

    LIFTED OUT OF op_add_particles ON 2026-09-04, unchanged in substance, because a
    particle system could be created and never adjusted. add_particles stacks a NEW
    system on every call - three calls give three systems, all emitting - so the only
    way to change a count was to add another system beside the one you meant to edit.
    Multiple systems on one object are legitimate (dust and sparks), which is why
    add_particles still stacks; what was missing was a way to reach one that exists.

    ONE WRITER, for the reason this file already knows twice over - the useModifierStack
    comment below records three keys that were accepted and read nowhere, and
    ops_lightcam records allowEditConst getting past one guard and not the other. Two
    copies of this would drift the same way.

    EVERY KEY IS READ AND CHECKED BEFORE THE FIRST WRITE, since 2026-09-16. It used to validate
    as it went: type, count, seed, the frame or hair keys were written, and only then were
    emitFrom, distribution, physicsType, rotationMode, renderType, instanceObject and
    instanceCollection checked - so a set_particles refusal ended "The system WAS MODIFIED.",
    honest and not all-or-nothing (second seat, mailbox id 13; taken in id 34 and left undone).
    Every refusal below now fires before anything is written and ends "NOTHING was created.",
    which set_particles rewrites to "NOTHING was changed." add_particles removes the system it
    made either way.

    WHAT CAN STILL REFUSE AFTER A WRITE is only the TypeError backstops on distribution and
    renderType, for a build whose type-dependent sets differ from the five measured: on 3.6.23,
    4.2.17, 4.4.0, 5.0.1 and 5.2.1 a HAIR system rejects distribution GRID and renderType HALO and
    LINE, and nothing else in emit_from, distribution, physics_type, rotation_mode or render_type
    depends on the type (scratchpad probe_particle_enum_sets.py). Those two keep "The system WAS
    created." because by then it is true.
    """
    st = psys.settings

    # ---- read every key. take() and take_float() refuse a wrong kind of value, so this is validation too.
    render_type = take(params, "renderType", default=None, kind=str)
    inst = take(params, "instanceObject", default=None, kind=str)
    coll = take(params, "instanceCollection", default=None, kind=str)
    # take_int FOR THE THREE WHOLE NUMBERS. They went through take_float and then int(), which FLOORS: {count: 100.5} made
    # 100 particles and {seed: 3.7} seed 3 under ok:true (the 2026-09-21 pack-rule review; measured on 5.2.1). A fraction
    # is refused, as take_int refuses it everywhere; a whole-valued float (100.0) still passes.
    n = take_int(params, "count", default=None)
    sd = take_int(params, "seed", default=None)
    fs = fe = lt = lr = hl = cc = None
    if kind == "EMITTER":
        fs = take_float(params, "frameStart", default=None)
        fe = take_float(params, "frameEnd", default=None)
        lt = take_float(params, "lifetime", default=None)
        lr = take_float(params, "lifetimeRandom", default=None)
    else:
        hl = take_float(params, "hairLength", default=None)
        cc = take_int(params, "childCount", default=None)
    ef = take(params, "emitFrom", default=None, kind=str)
    dist = take(params, "distribution", default=None, kind=str)
    pt = take(params, "physicsType", default=None, kind=str)
    rot = take(params, "rotationMode", default=None, kind=str)
    # WIRED 2026-09-03. These three were on the accept list and read NOWHERE: a caller could send
    # useModifierStack, useRotations or rotationMode, pass the guard, and have nothing happen. They
    # were invisible to audit_blender_dead_params because it blanked the reject_unknown CALL and not
    # the module-level constant the call names, which is where the literals live - so the audit
    # could not fail for any op with a named key set. Both are fixed together.
    use_stack = take_bool(params, "useModifierStack", default=True) if "useModifierStack" in params else None
    use_rot = take_bool(params, "useRotations", default=True) if "useRotations" in params else None
    factors = [(attr, take_float(params, key, default=None))
               for key, attr in (("normalFactor", "normal_factor"), ("randomFactor", "factor_random"),
                                 ("gravityFactor", "effector_weights.gravity"), ("dampingFactor", "damping"))]
    sz = take_float(params, "size", default=None)
    szr = take_float(params, "sizeRandom", default=None)
    show = take_bool(params, "showEmitter", default=True) if "showEmitter" in params else None
    # DENSITY BY VERTEX GROUP (2026-10-05): the group lives on the particle SYSTEM (psys.vertex_group_density), where no
    # writer reached - fur on one part of a body. "" clears it. Checked here with the rest, before any write.
    vgd = take(params, "vertexGroupDensity", default=None, kind=str)
    if vgd and obj.vertex_groups.get(vgd) is None:
        raise MifOpError("vertexGroupDensity %r is not a vertex group of '%s' (it has %s). NOTHING was created."
                         % (vgd, obj.name, ", ".join(g.name for g in obj.vertex_groups) or "none"))

    # ---- check every value against what Blender will take, still before any write.
    for value, key, prop in ((ef, "emitFrom", "emit_from"), (dist, "distribution", "distribution"),
                             (pt, "physicsType", "physics_type"), (rot, "rotationMode", "rotation_mode"),
                             (render_type, "renderType", "render_type")):
        # distribution USED TO BE THE ONE ENUM ON THIS OP THAT VALIDATED NOTHING - a typo came back as
        # Blender's own `TypeError: bpy_struct: item.attr = val: enum "X" not found in (...)`.
        if value:
            valid = _enum(bpy.types.ParticleSettings, prop)
            if str(value).upper() not in valid:
                raise MifOpError("unknown %s '%s'. Valid: %s. NOTHING was created."
                                 % (key, value, ", ".join(sorted(valid))))
    # Against the type the system will be after the call. See _HAIR_RENDER_TYPES.
    if kind == "HAIR" and render_type and str(render_type).upper() not in _HAIR_RENDER_TYPES:
        raise MifOpError("renderType '%s' is not available on a HAIR system - HAIR takes %s, and HALO and LINE are "
                         "EMITTER-only. NOTHING was created." % (str(render_type).upper(), ", ".join(_HAIR_RENDER_TYPES)))
    # GRID IS EMITTER-ONLY. First measured on 4.4.0 (ledger B17) - the same system accepts GRID as an EMITTER and
    # refuses it as HAIR with ('JIT', 'RAND'), independent of emitFrom and physicsType - and on all five builds since.
    if kind == "HAIR" and dist and str(dist).upper() == "GRID":
        raise MifOpError("distribution 'GRID' is not available on a HAIR system - GRID is EMITTER-only, so a HAIR system "
                         "takes JIT or RAND. Change the type first, or pick one of those. NOTHING was created.")
    inst_ob = None
    if inst:
        # SAYS THE SAME THING AS ITS SIBLING BELOW. get_object raises "no object named 'X'. Present: ..." and stops
        # there; the trailing sentence is what tells a caller whether a system was left behind.
        try:
            inst_ob = get_object(inst)
        except MifOpError as exc:
            raise MifOpError("%s. NOTHING was created." % str(exc).rstrip().rstrip("."))
    inst_coll = None
    if coll:
        inst_coll = bpy.data.collections.get(coll)
        if inst_coll is None:
            raise MifOpError("no collection named '%s'. NOTHING was created." % coll)
    show_targets = []
    if show is not None:
        # THE FLAG MOVED OFF ParticleSettings. It was settings.use_render_emitter in 2.7x and is
        # now show_instancer_for_render / _viewport on the OBJECT - so writing the old name raises
        # AttributeError rather than being ignored. Found the hard way: it took down a live build
        # at the dust-emitter step, because the suite never passed showEmitter and so never
        # reached this line. Newest name first, with the old one as a fallback.
        show_targets = [(holder, attr) for holder, attr in ((obj, "show_instancer_for_render"),
                                                            (obj, "show_instancer_for_viewport"),
                                                            (st, "use_render_emitter"))
                        if hasattr(holder, attr)]
        if not show_targets:
            raise MifOpError("this Blender exposes no emitter-visibility flag this op knows "
                             "(tried show_instancer_for_render/_viewport and use_render_emitter), "
                             "so showEmitter would have been silently ignored. NOTHING was created.")

    extra_writes, extra_notes = _plan_extras(obj, psys, params)

    # ---- write, in the order the op always has.
    st.type = kind
    if vgd is not None:
        psys.vertex_group_density = vgd
    if n is not None:
        st.count = int(n)
    if sd is not None:
        psys.seed = int(sd)
    if kind == "EMITTER":
        if fs is not None:
            st.frame_start = fs
        if fe is not None:
            st.frame_end = fe
        if lt is not None:
            st.lifetime = lt
        if lr is not None:
            st.lifetime_random = lr
    else:
        if hl is not None:
            st.hair_length = hl
        if cc is not None:
            st.child_type = "SIMPLE"
            _set_child_count(st, int(cc))
    if ef:
        st.emit_from = str(ef).upper()
    if dist:
        # A BACKSTOP for a build whose HAIR set differs from the five measured.
        try:
            st.distribution = str(dist).upper()
        except TypeError:
            raise MifOpError(
                "distribution '%s' is not available on a %s system on this Blender. The system WAS created."
                % (str(dist).upper(), st.type))
    if pt:
        st.physics_type = str(pt).upper()
    if use_stack is not None:
        # Emit from the EVALUATED mesh rather than the base one. With it off, a system on a
        # subdivided or displaced mesh emits from the undisplaced cage - particles that float above
        # or sink into the surface they were supposed to sit on, with every field reading correctly.
        st.use_modifier_stack = use_stack
    if use_rot is not None:
        st.use_rotations = use_rot
    if rot:
        st.rotation_mode = str(rot).upper()
        # ROTATION MODE DOES NOTHING WHILE use_rotations IS OFF, and it reads back perfectly either
        # way - so setting one without the other is a silent no-op. Turned on rather than refused,
        # because a caller naming a rotation mode has said what they want unambiguously.
        if not st.use_rotations:
            st.use_rotations = True
    for attr, v in factors:
        if v is None:
            continue
        if "." in attr:
            head, tail = attr.split(".", 1)
            setattr(getattr(st, head), tail, v)
        else:
            setattr(st, attr, v)
    if sz is not None:
        st.particle_size = sz
    if szr is not None:
        st.size_random = szr
    if render_type:
        # A BACKSTOP to the refusal above, for a build whose HAIR set differs from the five measured: Blender's
        # TypeError becomes a sentence, and add_particles' cleanup - MifOpError only - runs.
        try:
            st.render_type = str(render_type).upper()
        except TypeError:
            raise MifOpError("renderType '%s' is not available on a %s system on this Blender. The system WAS created."
                             % (str(render_type).upper(), st.type))
    if inst_ob is not None:
        st.instance_object = inst_ob
        if st.render_type != "OBJECT":
            st.render_type = "OBJECT"
    if inst_coll is not None:
        st.instance_collection = inst_coll
        st.render_type = "COLLECTION"
    for holder, attr in show_targets:
        setattr(holder, attr, show)
    for holder, attr, value in extra_writes:
        # A HOLDER THAT EXISTS ONLY ONCE AN EARLIER WRITE MADE IT - psys.cloth appears when use_hair_dynamics goes on
        setattr(holder() if callable(holder) else holder, attr, value)
    return extra_notes


# HAIR AND SCATTER SETTINGS THE WRITER LACKED (2026-10-04). Each name was read off the live RNA on 3.6.23, 4.4.0 and
# 5.2.1 before it was typed here: the 13 vertex-group slots and their invert flags live on the ParticleSystem, not the
# settings, on all three builds; child_type is NONE | SIMPLE | INTERPOLATED; the viewport child count is child_nbr on 3.6
# and child_percent from 4.x (see _child_count); use_advanced_hair defaults True on 3.6 and False on 4.4 and 5.2.
_CHILD_KEYS = {"viewport": None, "render": "rendered_child_count", "clump": "clump_factor",
               "clumpShape": "clump_shape", "length": "child_length", "radius": "child_radius"}
_ROUGH_KEYS = {"uniform": "roughness_1", "endpoint": "roughness_endpoint", "random": "roughness_2"}
_COLLECTION_KEYS = {"pickRandom": "use_collection_pick_random", "useCount": "use_collection_count"}
_ROTATION_KEYS = {"advanced": "use_advanced_hair", "randomFactor": "rotation_factor_random",
                  "phase": "phase_factor", "phaseRandom": "phase_factor_random"}
_GROUP_KEYS = {"density": "density", "velocity": "velocity", "length": "length", "clump": "clump", "kink": "kink",
               "roughness1": "roughness_1", "roughness2": "roughness_2", "roughnessEnd": "roughness_end",
               "size": "size", "tangent": "tangent", "rotation": "rotation", "field": "field", "twist": "twist"}


def _dict_param(params, key, allowed):
    raw = params.get(key)
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw:
        raise MifOpError("%s must be an object with any of %s, got %r. NOTHING was created."
                         % (key, ", ".join(sorted(allowed)), raw))
    bad = sorted(k for k in raw if k not in allowed)
    if bad:
        raise MifOpError("%s takes %s; %s %s not one. NOTHING was created."
                         % (key, ", ".join(sorted(allowed)), ", ".join(bad), "is" if len(bad) == 1 else "are"))
    return raw


def _num(value, label, lo=None, hi=None, whole=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MifOpError("%s must be a number, got %r. NOTHING was created." % (label, value))
    try:
        v = finite_int(value, label) if whole else finite_float(value, label)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was created." % str(exc).rstrip("."))
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise MifOpError("%s must be %s-%s, got %r. NOTHING was created." % (label, lo, hi, value))
    return v


def _flag(value, label):
    if not isinstance(value, bool):
        raise MifOpError("%s must be true or false, got %r. NOTHING was created." % (label, value))
    return value


def _plan_extras(obj, psys, params):
    """(writes [(holder, attr, value)], notes) for children, material, collection, rotation and groups - every value
    checked against this build's RNA before anything is written."""
    st = psys.settings
    rna = st.bl_rna.properties
    writes, notes = [], []
    ch = _dict_param(params, "children", set(_CHILD_KEYS) | {"type", "roughness"})
    if ch is not None:
        if "childCount" in params:
            raise MifOpError("childCount and children are two answers to the child count - childCount writes SIMPLE "
                             "children at one count; children sets the type and each count. Pass one. NOTHING was "
                             "created.")
        if "type" in ch:
            valid = [i.identifier for i in rna["child_type"].enum_items]
            got = str(ch["type"]).upper()
            if got not in valid:
                raise MifOpError("children.type must be one of %s, got %r. NOTHING was created."
                                 % (", ".join(valid), ch["type"]))
            writes.append((st, "child_type", got))
        for key, attr in _CHILD_KEYS.items():
            if key not in ch:
                continue
            if attr is None:
                attr = "child_percent" if "child_percent" in rna else "child_nbr"
            prop = rna[attr]
            whole = prop.type == "INT"
            writes.append((st, attr, _num(ch[key], "children." + key, prop.hard_min, prop.hard_max, whole)))
        rough = ch.get("roughness")
        if rough is not None:
            rough = _dict_param({"children.roughness": rough}, "children.roughness", set(_ROUGH_KEYS))
            for key, attr in _ROUGH_KEYS.items():
                if key in rough:
                    prop = rna[attr]
                    writes.append((st, attr, _num(rough[key], "children.roughness." + key, prop.hard_min,
                                                  prop.hard_max)))
        if "type" not in ch and st.child_type == "NONE":
            notes.append("the system has child_type NONE, so the child settings are stored and draw nothing until "
                         "children.type is SIMPLE or INTERPOLATED.")
    mat = params.get("material")
    if mat is not None:
        slots = [(i + 1, s.material.name if s.material else None) for i, s in enumerate(obj.material_slots)]
        if isinstance(mat, bool):
            raise MifOpError("material is a slot number (1 is the first) or a material name, got %r. NOTHING was "
                             "created." % mat)
        if isinstance(mat, (int, float)):
            index = _num(mat, "material", 1, max(1, len(slots)), whole=True)
            if not slots:
                raise MifOpError("'%s' has no material slots, so there is no slot %d for the particles to render "
                                 "with. set_material_slots adds them. NOTHING was created." % (obj.name, index))
        else:
            index = next((i for i, n in slots if n == str(mat)), None)
            if index is None:
                raise MifOpError("'%s' has no slot holding material '%s'. Its slots: %s. NOTHING was created."
                                 % (obj.name, mat, ", ".join("%d %s" % (i, n) for i, n in slots) or "none"))
        writes.append((st, "material", index))
    coll = _dict_param(params, "collection", set(_COLLECTION_KEYS))
    if coll is not None:
        for key, attr in _COLLECTION_KEYS.items():
            if key in coll:
                writes.append((st, attr, _flag(coll[key], "collection." + key)))
        will_be = str(params.get("renderType") or st.render_type).upper()
        if will_be != "COLLECTION":
            notes.append("the collection settings apply when renderType is COLLECTION; this system renders %s, so "
                         "they are stored and act once it instances a collection." % will_be)
    rot = _dict_param(params, "rotation", set(_ROTATION_KEYS))
    if rot is not None:
        for key, attr in _ROTATION_KEYS.items():
            if key not in rot:
                continue
            if key == "advanced":
                writes.append((st, attr, _flag(rot[key], "rotation.advanced")))
            else:
                prop = rna[attr]
                writes.append((st, attr, _num(rot[key], "rotation." + key, prop.hard_min, prop.hard_max)))
        if any(k in rot for k in ("randomFactor", "phase", "phaseRandom")):
            # THE ROTATION VALUES ACT ONLY WITH ROTATION ON, and on a HAIR system the Rotation panel is behind Advanced
            # (use_advanced_hair: True on 3.6, False on 4.4 and 5.2 by default). Asked for, so both are turned on, as
            # rotationMode turns use_rotations on; an explicit advanced false is refused instead.
            if rot.get("advanced") is False and str(params.get("type") or st.type).upper() == "HAIR":
                raise MifOpError("rotation values on a HAIR system need Advanced on (use_advanced_hair), and "
                                 "rotation.advanced is false. NOTHING was created.")
            if params.get("useRotations") is False:
                raise MifOpError("rotation values act only with useRotations on, and useRotations is false. NOTHING "
                                 "was created.")
            writes.append((st, "use_rotations", True))
            if str(params.get("type") or st.type).upper() == "HAIR" and "advanced" not in rot:
                writes.append((st, "use_advanced_hair", True))
    groups = _dict_param(params, "groups", set(_GROUP_KEYS))
    if groups is not None:
        names = [g.name for g in obj.vertex_groups]
        for key, slot in _GROUP_KEYS.items():
            if key not in groups:
                continue
            val = groups[key]
            invert = None
            if isinstance(val, dict):
                bad = sorted(k for k in val if k not in ("group", "invert"))
                if bad or "group" not in val:
                    raise MifOpError("groups.%s takes a group name or {group, invert}, got %r. NOTHING was created."
                                     % (key, val))
                invert = _flag(val["invert"], "groups.%s.invert" % key) if "invert" in val else None
                val = val["group"]
            if val is None or val == "":
                val = ""
            elif not isinstance(val, str) or val not in names:
                raise MifOpError("groups.%s names vertex group %r, which '%s' does not have. Its groups: %s. "
                                 "NOTHING was created." % (key, val, obj.name, ", ".join(names) or "none"))
            attr = "vertex_group_" + slot
            if attr not in psys.bl_rna.properties:
                raise MifOpError("this Blender's particle system has no %s. NOTHING was created." % attr)
            writes.append((psys, attr, val))
            if invert is not None:
                writes.append((psys, "invert_" + attr, invert))
    _plan_motion_and_strands(obj, psys, params, writes, notes)
    return writes, notes


# VELOCITY, HAIR DYNAMICS AND STRAND SHAPE (2026-10-05). Read off the live RNA on 4.2.23 and 5.2.2 before they were
# typed here: object_align_factor [x, y, z] and object_factor on ParticleSettings (-200 to 200); shape (-1 to 1),
# root_radius, tip_radius, radius_scale and use_close_tip there too; hair dynamics on the SYSTEM - use_hair_dynamics,
# and psys.cloth, which is None until it is on, holding settings.quality, pin_stiffness, mass, bending_stiffness and
# collision_settings.distance_min / collection; the cache range on psys.point_cache.
_DYN_SETTINGS = {"quality": "quality", "pinStiffness": "pin_stiffness", "mass": "mass",
                 "bendingStiffness": "bending_stiffness"}
_STRAND_KEYS = {"shape": "shape", "rootRadius": "root_radius", "tipRadius": "tip_radius",
                "radiusScale": "radius_scale", "closeTip": "use_close_tip"}


def _plan_motion_and_strands(obj, psys, params, writes, notes):
    st = psys.settings
    rna = st.bl_rna.properties
    oaf = params.get("objectAlignFactor")
    if oaf is not None:
        if not isinstance(oaf, (list, tuple)) or len(oaf) != 3:
            raise MifOpError("objectAlignFactor is [x, y, z] (velocity along the emitter's own axes), got %r. NOTHING "
                             "was created." % (oaf,))
        vals = [_num(v, "objectAlignFactor", -200.0, 200.0) for v in oaf]
        writes.append((st, "object_align_factor", tuple(vals)))
    if params.get("objectFactor") is not None:
        writes.append((st, "object_factor", _num(params["objectFactor"], "objectFactor", -200.0, 200.0)))
    strand = _dict_param(params, "strandShape", set(_STRAND_KEYS))
    if strand is not None:
        for key, attr in _STRAND_KEYS.items():
            if key not in strand:
                continue
            if key == "closeTip":
                writes.append((st, attr, _flag(strand[key], "strandShape.closeTip")))
            else:
                prop = rna[attr]
                hi = prop.hard_max if prop.hard_max < 1e30 else None
                writes.append((st, attr, _num(strand[key], "strandShape." + key, prop.hard_min, hi)))
    dyn = _dict_param(params, "hairDynamics", set(_DYN_SETTINGS) | {"use", "collision", "cacheRange"})
    if dyn is None:
        return
    will_be = str(params.get("type") or st.type).upper()
    if will_be != "HAIR":
        raise MifOpError("hairDynamics simulates HAIR strands, and this system is %s. NOTHING was created." % will_be)
    use = _flag(dyn["use"], "hairDynamics.use") if "use" in dyn else None
    on = use if use is not None else bool(psys.use_hair_dynamics)
    rest = [k for k in dyn if k != "use"]
    if rest and not on:
        raise MifOpError("hairDynamics %s live on the hair's cloth, which exists only with dynamics on - pass use: true. "
                         "NOTHING was created." % ", ".join(sorted(rest)))
    if use is not None:
        writes.append((psys, "use_hair_dynamics", use))
    for key, attr in _DYN_SETTINGS.items():
        if key in dyn:
            whole = key == "quality"
            writes.append((lambda: psys.cloth.settings, attr,
                           _num(dyn[key], "hairDynamics." + key, 1 if whole else 0.0, None, whole)))
    col = dyn.get("collision")
    if col is not None:
        col = _dict_param({"hairDynamics.collision": col}, "hairDynamics.collision", {"distanceMin", "collection"})
        if "distanceMin" in col:
            writes.append((lambda: psys.cloth.collision_settings, "distance_min",
                           _num(col["distanceMin"], "hairDynamics.collision.distanceMin", 0.001, 1.0)))
        if "collection" in col:
            name = col["collection"]
            c = bpy.data.collections.get(str(name)) if name else None
            if name and c is None:
                raise MifOpError("hairDynamics.collision.collection %r is not a collection. NOTHING was created." % name)
            writes.append((lambda: psys.cloth.collision_settings, "collection", c))
    rng = dyn.get("cacheRange")
    if rng is not None:
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            raise MifOpError("hairDynamics.cacheRange is [start, end], got %r. NOTHING was created." % (rng,))
        a = _num(rng[0], "hairDynamics.cacheRange start", whole=True)
        b = _num(rng[1], "hairDynamics.cacheRange end", whole=True)
        if b < a:
            raise MifOpError("hairDynamics.cacheRange ends before it starts (%d, %d). NOTHING was created." % (a, b))
        writes.append((psys.point_cache, "frame_end", b))
        writes.append((psys.point_cache, "frame_start", a))
    notes.append("hair dynamics simulate on playback; bake_physics {object, modifier: '%s'} bakes them." % psys.name)


def _live_particles(obj, psys):
    """An EMITTER's particles as the depsgraph has them at the current frame: how many are alive and their bounds.
    The reader the velocity settings lacked - nothing else in the add-on reports where particles are."""
    if psys.settings.type != "EMITTER":
        return None
    try:
        ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        pev = ev.particle_systems.get(psys.name)
        alive = [p.location for p in list(pev.particles)[:20000] if p.alive_state == "ALIVE"] if pev else []
    except (RuntimeError, AttributeError, ReferenceError):
        return None
    out = {"frame": bpy.context.scene.frame_current, "alive": len(alive)}
    if alive:
        out["boundsMin"] = [round(min(v[i] for v in alive), 5) for i in range(3)]
        out["boundsMax"] = [round(max(v[i] for v in alive), 5) for i in range(3)]
    return out


def _motion_row(psys):
    st = psys.settings
    cloth = psys.cloth
    dyn = {"use": bool(psys.use_hair_dynamics)}
    if cloth is not None:
        dyn.update({k: (int(getattr(cloth.settings, a)) if k == "quality" else round(float(getattr(cloth.settings, a)),
                                                                                   5))
                    for k, a in _DYN_SETTINGS.items()})
        cs = cloth.collision_settings
        dyn["collision"] = {"distanceMin": round(float(cs.distance_min), 5),
                            "collection": cs.collection.name if cs.collection else None}
        dyn["cacheRange"] = [int(psys.point_cache.frame_start), int(psys.point_cache.frame_end)]
    return {"objectAlignFactor": [round(float(v), 5) for v in st.object_align_factor],
            "objectFactor": round(float(st.object_factor), 5),
            "strandShape": {k: (bool(getattr(st, a)) if k == "closeTip" else round(float(getattr(st, a)), 5))
                            for k, a in _STRAND_KEYS.items()},
            "hairDynamics": dyn if st.type == "HAIR" else None}


def _extras_row(obj, psys):
    """The settings _plan_extras writes, read back."""
    st = psys.settings
    groups = {}
    for key, slot in _GROUP_KEYS.items():
        name = getattr(psys, "vertex_group_" + slot, "")
        if name:
            groups[key] = {"group": name, "invert": bool(getattr(psys, "invert_vertex_group_" + slot, False))}
    slot = int(st.material)
    mat = obj.material_slots[slot - 1].material if 0 < slot <= len(obj.material_slots) else None
    return {
        "children": {"type": st.child_type, "viewport": _child_count(st),
                     "render": int(st.rendered_child_count) if hasattr(st, "rendered_child_count") else None,
                     "clump": round(float(st.clump_factor), 5), "clumpShape": round(float(st.clump_shape), 5),
                     "length": round(float(st.child_length), 5), "radius": round(float(st.child_radius), 5),
                     "roughness": {k: round(float(getattr(st, a)), 5) for k, a in _ROUGH_KEYS.items()}},
        "material": {"slot": slot, "name": mat.name if mat else None},
        "collection": {k: bool(getattr(st, a)) for k, a in _COLLECTION_KEYS.items()},
        "rotation": {"advanced": bool(st.use_advanced_hair), "randomFactor": round(float(st.rotation_factor_random), 5),
                     "phase": round(float(st.phase_factor), 5), "phaseRandom": round(float(st.phase_factor_random), 5)},
        "groups": groups or None,
        **_motion_row(psys),
    }


# THE VIEWPORT CHILD COUNT IS SPELLED DIFFERENTLY BEFORE AND AFTER BLENDER 4, and reading the 4.x
# name unguarded broke EVERY particle op on 3.6.
#
# 3.6.23 has `child_nbr` and `rendered_child_count`; 4.x renamed the first to `child_percent`.
# _particle_row read `st.child_percent` unconditionally, and that row is built by add_particles,
# set_particles AND list_particles - so on 3.6 all three answered
# "AttributeError: 'ParticleSettings' object has no attribute 'child_percent'". Measured 2026-09-14
# by asking a live 3.6.23 which attributes it has, not by guessing the old name.
#
# WORSE THAN A REFUSAL: add_particles creates the system and THEN builds the row, so a failed call
# left the object carrying an orphaned particle system that list_particles could not even show -
# five of them accumulated during one probe. A mutate-then-fail with no cleanup.
#
# getattr BY NAME rather than a version number, because the property is what matters and a version
# check would have to be re-dated for every build. The same reason MifBridgeStyle's engine guards
# test for the symbol where they can.


def _child_count(st):
    """The viewport child count, whatever this build calls it."""
    if hasattr(st, "child_percent"):
        return int(st.child_percent)
    if hasattr(st, "child_nbr"):
        return int(st.child_nbr)
    return 0


def _set_child_count(st, n):
    """Write both the viewport and the render child count, on either spelling."""
    if hasattr(st, "child_percent"):
        st.child_percent = n
    elif hasattr(st, "child_nbr"):
        st.child_nbr = n
    if hasattr(st, "rendered_child_count"):
        st.rendered_child_count = n


# WHAT DRAWS NOTHING. Measured 2026-09-16 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1, in EEVEE, Cycles and Workbench: an
# emitter plane hidden from render, 300 particles emitted over frames 1-5 and stepped to frame 12, rendered at 128 px and
# compared pixel by pixel with the same render with the particle modifier's render switched off. HALO, NONE, OBJECT with
# no instance_object and COLLECTION with no instance_collection all differed by 0.0/255 at EVERY pixel, on every build
# and every engine. The controls drew: a 0.08 BU sphere instanced by OBJECT, and by a COLLECTION holding it, 3.7-4.7/255
# mean in EEVEE and Cycles and 8.7-9.1 in Workbench, across 2200-2450 pixels. (A first collection control read 0.0 too,
# and was the fixture's fault: collection instancing skips members hidden from render and adds each member's location.)
#
# HALO IS THE ONE THAT MATTERED, because it is Blender's default for an EMITTER and so this op's: the second seat found
# it on 5.2.2 (mailbox id 49) - add a system, bake, render, and get an empty frame with no refusal and no warning.
# add_particles still takes HALO (a caller may want the legacy point in the viewport), and says what it will render.
# A HAIR system comes up as PATH on every build, so the default is only a trap on an emitter.
#
# ONE PREDICATE for renderNote and list_particles' rendersNothing. The listing had its own copy, and it had already
# drifted twice: OBJECT only until 2026-09-14 (ledger B16), then OBJECT and COLLECTION with NONE and HALO still false.
def _blank_render_reason(st):
    """Why this system renders nothing, as a sentence - or None when it draws something."""
    rt = st.render_type
    if rt == "HALO":
        return ("renderType HALO renders NOTHING in EEVEE, Cycles or Workbench - measured pixel-identical to no "
                "particles on 3.6 to 5.2. Pass instanceObject or instanceCollection to see the particles.")
    if rt == "NONE":
        return "renderType NONE renders nothing."
    # ZERO-LENGTH HAIR, added the same day. 400 strands on a plane at hair_length 0 matched the no-particle render at
    # every pixel as PATH, as OBJECT and as COLLECTION instancing, in EEVEE, Cycles and Workbench on all five builds;
    # at 0.5 the same systems drew 4.0-11.3/255. On hair normalFactor IS the length (see
    # _refuse_keys_for_the_other_type), which is how "normalFactor 0" made bald spheres.
    if st.type == "HAIR" and abs(float(st.hair_length)) < 1e-9:
        return ("hairLength 0 renders NOTHING - on a HAIR system normalFactor is the strand length too "
                "(hairLength = 4 x normalFactor), so normalFactor 0 means no hair.")
    if rt == "OBJECT" and st.instance_object is None:
        return "renderType OBJECT with no instanceObject renders nothing - pass instanceObject."
    if rt == "COLLECTION" and st.instance_collection is None:
        return "renderType COLLECTION with no instanceCollection renders nothing - pass instanceCollection."
    return None


# THE RENDER TYPES A HAIR SYSTEM TAKES. HALO and LINE are EMITTER-only: assigning either to a HAIR system raised
# Blender's own `TypeError: bpy_struct: item.attr = val: enum "HALO" not found in ('NONE', 'PATH', 'OBJECT',
# 'COLLECTION')` straight out of both ops - and add_particles' cleanup catches MifOpError only, so it LEFT the system and
# its settings datablock on the mesh behind that error. Measured 2026-09-16, identical on 3.6.23, 4.2.17, 4.4.0, 5.0.1
# and 5.2.1: an EMITTER accepts all six static items, a HAIR system these four. Second seat, mailbox id 13.
_HAIR_RENDER_TYPES = ("NONE", "PATH", "OBJECT", "COLLECTION")

# FIELDS A SYSTEM'S ROW DERIVES rather than reads off a setting. set_particles' changedFields lists what the call
# changed, and a note that appears because renderType moved is not a second change.
_DERIVED_ROW_KEYS = ("bakeNote", "renderNote")


def _particle_row(obj, psys):
    """What a particle system IS, off the datablock. One reader for add_particles and set_particles.

    Extracted with the writer on 2026-09-04 for the same reason: two ops answering the same question
    in two places is how they drift.
    """
    st = psys.settings
    return {
        "object": obj.name,
        "system": psys.name,
        "settings": st.name,
        "type": st.type,
        "count": int(st.count),
        "vertexGroupDensity": psys.vertex_group_density or None,
        "emitFrom": st.emit_from,
        "physicsType": st.physics_type,
        "renderType": st.render_type,
        # WHY THIS SYSTEM WILL DRAW NOTHING, or None. See _blank_render_reason: an EMITTER's default, HALO, is one of
        # the four, so the shortest correct-looking use of add_particles rendered an empty frame and said nothing.
        "renderNote": _blank_render_reason(st),
        "instanceObject": st.instance_object.name if st.instance_object else None,
        # ITS SIBLING, added 2026-09-14. instanceCollection has always been writable by both
        # add_particles and set_particles and was reported by neither, so a caller could set it
        # and had no way to see whether it landed - and the suite could not check it either,
        # which is why it went unnoticed while its OBJECT twin two lines up was covered.
        "instanceCollection": st.instance_collection.name if st.instance_collection else None,
        "frameRange": [round(float(st.frame_start), 3), round(float(st.frame_end), 3)]
                      if st.type == "EMITTER" else None,
        "hairLength": round(float(st.hair_length), 4) if st.type == "HAIR" else None,
        "systemsOnObject": len(obj.particle_systems),
        "showEmitter": bool(getattr(obj, "show_instancer_for_render", True)),
        # EVERYTHING ELSE set_particles CAN WRITE, added 2026-09-13. It accepts 29 keys and this row
        # named 12 of them, so gravityFactor, dampingFactor, size, lifetime and nine more were writable
        # and unverifiable: a caller could not tell a change that landed from one that did not, and a
        # showcase scene asserting its own set_particles call had nothing to assert against. The same
        # shape as add_soft_body's goalSpring, found the same day and fixed the same way - read them
        # back off the settings datablock rather than echoing the request.
        "size": round(float(st.particle_size), 5),
        "sizeRandom": round(float(st.size_random), 5),
        "lifetime": round(float(st.lifetime), 3),
        "lifetimeRandom": round(float(st.lifetime_random), 5),
        "normalFactor": round(float(st.normal_factor), 5),
        "randomFactor": round(float(st.factor_random), 5),
        "gravityFactor": round(float(st.effector_weights.gravity), 5),
        "dampingFactor": round(float(st.damping), 5),
        "childCount": _child_count(st),
        # THE OTHER TWO THINGS childCount WRITES - _set_child_count also sets child_type SIMPLE and the RENDER count, and
        # neither was read by anything (audit_write_only_state, 2026-09-21), so the viewport count was the only proof
        # children existed. A render count different from the viewport one is a system that renders unlike it draws.
        "childType": st.child_type,
        "renderedChildCount": int(st.rendered_child_count) if hasattr(st, "rendered_child_count") else None,
        "distribution": st.distribution,
        "rotationMode": st.rotation_mode,
        "useRotations": bool(st.use_rotations),
        "useModifierStack": bool(st.use_modifier_stack),
        "seed": int(psys.seed),
        "liveParticles": _live_particles(obj, psys),
        "bakeNote": _bake_note(psys, st),
        **_extras_row(obj, psys),
    }


def _bake_note(psys, st):
    """What baking means for THIS system - by its type, not one sentence for all of them.

    IT WAS ONE SENTENCE FOR EVERY SYSTEM, and wrong twice. It said "EMITTER particles are stepped forward from
    frame_start; a late frame shows nothing until the sim has run through" - on a HAIR system too, which is not
    simulated at all. And for an emitter the second half does not happen: measured 2026-09-21 on 3.6.23, 4.2.17,
    4.4.0, 5.0.1 and 5.2.1, an unbaked emitter (50 particles, frames 1-10, lifetime 500) jumped straight to frame
    120, 240 and 300 - past the cache's end - drew all 50 every time, because Blender simulates the frames in
    between. A HAIR system drew its 50 strands at frame 1 and at 200 alike. The note sends a caller to bake, and
    the reason it gave was not true.
    """
    if st.type == "HAIR":
        if getattr(psys, "use_hair_dynamics", False):
            return ("HAIR with dynamics on is simulated, and bake_physics covers its cache.")
        return "HAIR is not simulated: the strands are the same on every frame, so there is nothing to bake."
    return ("EMITTER particles are SIMULATED from frame_start, not stored: reading or rendering a late frame "
            "simulates every frame before it first (an unbaked emitter jumped straight to frame 240 drew all its "
            "particles, measured on 3.6-5.2). bake_physics stores the result, so later reads and renders do not "
            "recompute it.")


# ONE PER-TYPE CHECK FOR BOTH OPS, moved out of op_add_particles on 2026-09-16. set_particles' docstring has always said
# its per-type key check runs against the type the system will be after the call, and it never had one: measured on
# 3.6.23 and 5.2.1 through the op in-process, set_particles took hairLength and childCount on an EMITTER and frameStart
# and lifetime on a HAIR system, answered ok with changedFields [], and wrote nothing - the writer reads each key only on
# its own type's branch. add_particles refused the same payloads by name.
#
# AND hairLength WITH normalFactor. They are ONE stored value on every build, emitter or hair: hair_length reads
# 4 x normal_factor, writing hair_length 0.5 reads normal_factor 0.125 back, and writing normal_factor 1.0 reads
# hair_length 4.0 (measured 2026-09-16 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1). The writer sets hairLength first and
# normalFactor after, so add_particles {HAIR, hairLength 0.5, normalFactor 1.0} answered hairLength 4.0 - the caller's
# hairLength dropped without a word. The second seat hit the other face of it (mailbox id 13): normalFactor 0.0 on a
# hair system, meant as "no initial push", rendered bald spheres, because on hair it is the strand. Two values that
# agree are one request and go through; two that disagree are refused, since either winner would be a guess.
#
# THE MESSAGE ENDS "NOTHING was created." - add_particles calls this before it creates anything, and set_particles calls
# it inside the block that rewrites that sentence to "NOTHING was changed.", before the writer.
def _refuse_keys_for_the_other_type(params, kind):
    """Refuse keys the system's type (after this call) does not read, and a hairLength/normalFactor pair that disagree."""
    for keys, want, label in ((("frameStart", "frameEnd", "lifetime", "lifetimeRandom"),
                               "EMITTER", "frameStart/frameEnd/lifetime"),
                              (("hairLength", "childCount"), "HAIR", "hairLength/childCount")):
        present = [k for k in keys if k in params]
        if present and kind != want:
            raise MifOpError("%s applies to a %s system and this one is %s (%s given). NOTHING was "
                             "created." % (label, want, kind, ", ".join(present)))
    if kind == "HAIR" and "hairLength" in params and "normalFactor" in params:
        hl = take_float(params, "hairLength", default=None)
        nf = take_float(params, "normalFactor", default=None)
        if hl is not None and nf is not None and abs(hl - 4.0 * nf) > 1e-6:
            raise MifOpError("hairLength %g and normalFactor %g disagree: on a HAIR system they are ONE stored value, "
                             "hairLength = 4 x normalFactor, so normalFactor %g would make the strands %g long. Send "
                             "one of the two. NOTHING was created." % (hl, nf, nf, 4.0 * nf))


def op_add_particles(params):
    """Add a particle system. Returns the settings read back off the datablock.

    params:
      object (str, required)   the emitter
      type                     EMITTER (default) | HAIR
      count (int)              number of particles
      frameStart / frameEnd    EMITTER only - the emission window
      lifetime                 EMITTER only
      hairLength / childCount  HAIR only
      emitFrom                 VERT | FACE | VOLUME
      physicsType              NO | NEWTON | KEYED | BOIDS | FLUID
      normalFactor / randomFactor / gravityFactor / dampingFactor
                               on a HAIR system normalFactor IS the strand length: hairLength = 4 x
                               normalFactor, one stored value. Send one; two that disagree are refused.
      size / sizeRandom
      renderType               NONE | HALO | LINE | PATH | OBJECT | COLLECTION. An EMITTER defaults to HALO,
                               which renders NOTHING - the response's renderNote says so. HAIR takes NONE,
                               PATH, OBJECT or COLLECTION and defaults to PATH.
      instanceObject           the object to instance - REQUIRED when renderType is OBJECT
      showEmitter (bool)
    """
    reject_unknown(params, _PS_KEYS, "add_particles")
    obj = get_object(take(params, "object", "name", required=True), want_mesh=True)
    kind = str(take(params, "type", default="EMITTER", kind=str)).upper()
    if kind not in ("EMITTER", "HAIR"):
        raise MifOpError("particle type must be EMITTER or HAIR, got '%s'. NOTHING was created."
                         % kind)

    # REFUSED BEFORE ANYTHING IS CREATED, the same way create_light does it - a typo should not
    # leave a half-built particle system on somebody's mesh.
    _refuse_keys_for_the_other_type(params, kind)

    render_type = take(params, "renderType", default=None, kind=str)
    inst = take(params, "instanceObject", default=None, kind=str)
    if render_type and str(render_type).upper() == "OBJECT" and not inst:
        raise MifOpError("renderType OBJECT needs instanceObject - without it the system renders "
                         "NOTHING and Blender reports no error at all. NOTHING was created.")
    # THE SAME GUARD FOR THE OTHER INSTANCE SOURCE, added 2026-09-14. COLLECTION without a
    # collection is the identical failure and was accepted silently: measured through the depsgraph
    # on 3.6.23 and 4.4.0, render_type COLLECTION with instance_collection None evaluates to 0
    # instances and to 20 once a collection is set - the same 0-vs-20 the OBJECT case gives. Two
    # adjacent paths, one guarded since the op was written and one not. Ledger B16.
    # READ INTO A LOCAL BEFORE THE CONDITION, not inside it - which is exactly what `inst` two lines
    # above does, and the first draft of this guard did not. Behind an `and` the take() only runs
    # when renderType happens to be COLLECTION, so its kind check and its alias-clash check would
    # fire or not depending on an unrelated parameter: the shape audit_blender_shortcircuit_take
    # exists for, and it caught this within the hour. The value is applied later by
    # _apply_particle_settings, which reads it again; take() does not consume, so that is one
    # parameter read twice rather than a value stolen from the apply pass.
    want_coll = take(params, "instanceCollection", default=None, kind=str)
    if render_type and str(render_type).upper() == "COLLECTION" and not want_coll:
        raise MifOpError("renderType COLLECTION needs instanceCollection - without it the system "
                         "renders NOTHING and Blender reports no error at all, exactly as with "
                         "renderType OBJECT. NOTHING was created.")

    mod = obj.modifiers.new(name=str(take(params, "systemName", default="ParticleSystem",
                                          kind=str)), type="PARTICLE_SYSTEM")
    psys = obj.particle_systems[-1]
    # THE SETTINGS DATABLOCK IS CAPTURED SEPARATELY. modifiers.remove() takes the system off
    # the object and leaves bpy.data.particles holding an orphaned ParticleSettings - which
    # is a different collection and a different lifetime. The leak pass caught this on the
    # SECOND run, after the first fix removed only the modifier: my hand probes would have
    # stopped at "the system is gone", because that is the part you think to look at.
    _settings = psys.settings
    # THE SYSTEM GOES IF THE SETTINGS REFUSE. The writer validates deep - enums against the
    # live RNA, per-type keys, instanceObject resolution - and every one of those refusals
    # used to fire with the system already on the mesh. Its messages say "The system WAS
    # created", which was honest but is not what a caller wants: a typo'd count left a
    # particle system behind on their object. Found by the matrix leak pass, which corrupts
    # one payload value at a time and compares bpy.data - the same three-defect shape it was
    # built for. set_particles does NOT do this: it edits a system that already existed, so
    # removing one on a refusal would destroy something the caller made earlier.
    try:
        notes = _apply_particle_settings(obj, psys, params, kind)
    except MifOpError as exc:
        try:
            obj.modifiers.remove(mod)
        except Exception:  # noqa: BLE001 - cleanup must not mask the real refusal
            pass
        try:
            if _settings is not None and _settings.users == 0:
                bpy.data.particles.remove(_settings)
        except Exception:  # noqa: BLE001
            pass
        raise MifOpError(str(exc).replace("The system WAS created.",
                                          "The system was removed again, NOTHING was "
                                          "created."))
    row = _particle_row(obj, psys)
    row["settingsNotes"] = notes or None
    return row


def op_list_particles(params):
    """Every particle system on an object, read off the datablocks - the verification half.

    params:
      object (str, required)
      limit (int)   how many systems to return. Unlimited by default; `count` is always the true
                    total and `truncated` says when rows were dropped.

    THE FATTEST ROW OF ANY LISTING HERE. Each one carries the whole of _particle_row - the same
    twelve-plus fields add_particles reports - so it is about 800 bytes a system, measured
    2026-09-20 on 5.0.1: eight systems on one cube answered 6,420 bytes. add_particles STACKS by
    design, which is right for particles (dust and sparks on one mesh is ordinary) and means the
    count is bounded by nothing. Same optional, unlimited-by-default `limit` as the other
    listings.
    """
    reject_unknown(params, _LIST_KEYS, "list_particles")
    obj = get_object(take(params, "object", "name", required=True), want_mesh=True)
    # THROUGH _particle_row, since 2026-09-13. This built its OWN thinner row - nine fields against
    # add_particles' twelve - which is exactly the drift _particle_row was extracted to stop, and it had
    # already drifted: an op whose docstring calls it "the verification half" could not report the
    # gravityFactor or the size that set_particles had just written. The per-object fields come off the
    # row because they belong to the response, not to each system in it.
    out = []
    for psys in obj.particle_systems:
        st = psys.settings
        row = _particle_row(obj, psys)
        for k in ("object", "systemsOnObject", "bakeNote"):
            row.pop(k, None)
        # BOTH INSTANCE SOURCES, not just the object one. This field exists to answer "will this
        # system draw anything", and COLLECTION with no instance_collection is the same failure
        # as OBJECT with no instance_object - which add_particles refuses by name while this
        # answered False for the collection case and reported a system that draws nothing as
        # fine. Measured through the depsgraph on 3.6.23 and 4.4.0: render_type COLLECTION with
        # instance_collection None evaluates to 0 instances and 20 once a collection is set,
        # exactly like the OBJECT case. Ledger B16.
        #
        # AND HALO AND NONE, since 2026-09-16: both render nothing on every build and engine measured, and HALO is an
        # emitter's default - so this answered false for the most common dead system there is. The predicate is
        # _blank_render_reason, the same one renderNote reads, so the two cannot disagree again.
        row["rendersNothing"] = _blank_render_reason(st) is not None
        out.append(row)
    total = len(out)
    out, extra = limit_rows(out, params, "particle system(s)", excludes=False)
    return dict({"object": obj.name, "systems": out, "count": total}, **extra)



_SET_PS_KEYS = _PS_CORE | {"system", "edit"}

# ---------------------------------------------------------------------------
# set_particles edit: comb, puff, cut, length and smooth a HAIR system's guides (2026-10-05)
# ---------------------------------------------------------------------------
# THE ROUTE, MEASURED FIRST on 3.6.23, 4.4.0 and 5.2.1 (identical): a new hair system's ORIGINAL particles list is empty
# until particle edit mode has been entered once; after one particle_edit_toggle on and off (under a context override,
# headless) it holds every guide, and HairKey.co_object_set(evaluated object, its modifier, the evaluated particle,
# co) moves a key - a tip set 0.3 along X read back 0.3 along X on the evaluated hair, is_edited turned true, and the
# edit survived another toggle. Writing hair_keys[j].co or .co_local directly does nothing (the original's co reads 0,0,0
# and the evaluated hair did not move, in and out of edit mode), and particle.brush_edit's poll fails without a 3D view.
# So the brush is done here, on the key positions, and written through co_object_set.
_EDIT_TOOLS = ("COMB", "PUFF", "CUT", "LENGTH", "SMOOTH")
_EDIT_KEYS = {"tool", "center", "radius", "strength", "direction", "falloff", "symmetry"}


def _plan_edit(raw):
    if not isinstance(raw, dict) or not raw:
        raise MifOpError("edit is {tool, center, radius, strength, direction, falloff, symmetry}, got %r. NOTHING was "
                         "changed." % (raw,))
    bad = sorted(k for k in raw if k not in _EDIT_KEYS)
    if bad:
        raise MifOpError("edit takes %s; %s is not one. NOTHING was changed." % (", ".join(sorted(_EDIT_KEYS)),
                                                                             ", ".join(bad)))
    tool = str(raw.get("tool", "")).upper()
    if tool not in _EDIT_TOOLS:
        raise MifOpError("edit.tool must be one of %s, got %r. NOTHING was changed." % (", ".join(_EDIT_TOOLS),
                                                                                    raw.get("tool")))

    def vec(key):
        v = raw.get(key)
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            raise MifOpError("edit.%s must be [x, y, z], got %r. NOTHING was changed." % (key, v))
        try:
            return finite_floats(v, "edit." + key)
        except MifOpError as exc:
            raise MifOpError("%s NOTHING was changed." % str(exc).rstrip("."))
    center = vec("center")
    radius = _num(raw.get("radius"), "edit.radius", 1e-6, 1e6)
    lo = -1.0 if tool == "LENGTH" else 0.0
    strength = _num(raw.get("strength", 1.0), "edit.strength", lo, 1.0)
    direction = vec("direction") if tool == "COMB" or "direction" in raw else None
    if tool != "COMB" and direction is not None:
        raise MifOpError("edit.direction is the COMB's stroke; %s takes none. NOTHING was changed." % tool)
    falloff = str(raw.get("falloff", "SMOOTH")).upper()
    if falloff not in ("SMOOTH", "LINEAR", "CONSTANT"):
        raise MifOpError("edit.falloff must be SMOOTH, LINEAR or CONSTANT, got %r. NOTHING was changed."
                         % raw.get("falloff"))
    sym = raw.get("symmetry")
    if sym not in (None, "X", "x"):
        raise MifOpError("edit.symmetry is 'X' (mirror the brush across the object's X) or absent, got %r. NOTHING "
                         "was changed." % (sym,))
    return {"tool": tool, "center": center, "radius": radius, "strength": strength, "direction": direction,
            "falloff": falloff, "symmetry": bool(sym)}


def _falloff(d, radius, kind):
    if d >= radius:
        return 0.0
    t = 1.0 - d / radius
    return 1.0 if kind == "CONSTANT" else t if kind == "LINEAR" else t * t * (3.0 - 2.0 * t)


def _rebuild(pts, lengths):
    """Keep the root and each segment's length; take each segment's direction from `pts`."""
    out = [pts[0].copy()]
    for j in range(1, len(pts)):
        d = pts[j] - pts[j - 1]
        d = d.normalized() if d.length > 1e-12 else (pts[j] - out[-1]).normalized()
        out.append(out[-1] + d * lengths[j - 1])
    return out


def _brush_guide(pts, plan, center, direction, normal):
    """One guide's keys (object space Vectors) after one brush pass -> new list, or None when untouched."""
    lengths = [(pts[j] - pts[j - 1]).length for j in range(1, len(pts))]
    w = [_falloff((p - center).length, plan["radius"], plan["falloff"]) * plan["strength"] for p in pts]
    if not any(abs(x) > 0 for x in w[1:]):
        return None
    tool = plan["tool"]
    if tool == "COMB":
        moved = [pts[0]] + [pts[j] + direction * w[j] for j in range(1, len(pts))]
        return _rebuild(moved, lengths)
    if tool == "PUFF":
        seg = [(pts[j] - pts[j - 1]) for j in range(1, len(pts))]
        out = [pts[0].copy()]
        for j, s in enumerate(seg, 1):
            d = s.normalized().lerp(normal, max(0.0, min(1.0, w[j]))).normalized()
            out.append(out[-1] + d * lengths[j - 1])
        return out
    if tool == "SMOOTH":
        moved = [pts[0]] + [pts[j].lerp((pts[j - 1] + pts[j + 1]) * 0.5, max(0.0, min(1.0, w[j])))
                            for j in range(1, len(pts) - 1)] + [pts[-1]]
        return _rebuild(moved, lengths)
    # CUT shortens by the weight, LENGTH grows (strength > 0) or shrinks (< 0): the guide is scaled from its root by
    # the weight at its tip, every key keeping its place along the guide.
    tip_w = max(w[1:], key=abs)
    factor = (1.0 - tip_w) if tool == "CUT" else (1.0 + tip_w)
    factor = max(0.01, factor)
    return [pts[0].copy()] + [pts[0] + (pts[j] - pts[0]) * factor for j in range(1, len(pts))]


def _edit_guides(obj, psys, params):
    """set_particles {edit}: one brush pass over the guides of a HAIR system, written through co_object_set."""
    extra = sorted(k for k in params if k not in ("object", "name", "system", "edit"))
    if extra:
        raise MifOpError("edit runs alone - a setting changed in the same call (%s) can regenerate the guides and drop "
                         "the edit. Send it in its own call. NOTHING was changed." % ", ".join(extra))
    try:
        plan = _plan_edit(params.get("edit"))
    except MifOpError as exc:
        raise MifOpError(str(exc).replace("NOTHING was created.", "NOTHING was changed."))
    if psys.settings.type != "HAIR":
        raise MifOpError("edit grooms a HAIR system's guides, and '%s' is an EMITTER. NOTHING was changed." % psys.name)
    if obj.mode != "OBJECT":
        raise MifOpError("'%s' is in %s mode; edit needs it in OBJECT mode (it enters particle edit once itself). "
                         "NOTHING was changed." % (obj.name, obj.mode))
    from mathutils import Vector
    idx = list(obj.particle_systems).index(psys)
    if not psys.is_edited or not len(psys.particles):
        prev = obj.particle_systems.active_index
        obj.particle_systems.active_index = idx
        with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj],
                                       selected_editable_objects=[obj]):
            bpy.ops.particle.particle_edit_toggle()
            bpy.ops.particle.particle_edit_toggle()
        obj.particle_systems.active_index = prev
        if obj.mode != "OBJECT" or not len(psys.particles):
            raise MifOpError("entering and leaving particle edit did not give '%s' editable guides (mode %s, %d "
                             "guides) on this Blender (%s). NOTHING was changed beyond that toggle."
                             % (obj.name, obj.mode, len(psys.particles), bpy.app.version_string))
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    pev = ev.particle_systems[idx]
    mod_name = next(m.name for m in obj.modifiers if m.type == "PARTICLE_SYSTEM" and m.particle_system == psys)
    evmod = ev.modifiers[mod_name]
    mw = obj.matrix_world
    inv = mw.inverted()
    centers = [inv @ Vector(plan["center"])]
    dirs = [inv.to_3x3() @ Vector(plan["direction"])] if plan["direction"] else [None]
    if plan["symmetry"]:
        c = Vector(centers[0])
        centers.append(Vector((-c.x, c.y, c.z)))
        dirs.append(Vector((-dirs[0].x, dirs[0].y, dirs[0].z)) if dirs[0] is not None else None)
    writes, longest = [], 0.0
    for i, (p_orig, p_ev) in enumerate(zip(psys.particles, pev.particles)):
        pts = [Vector(k.co_object(ev, evmod, p_ev)) for k in p_ev.hair_keys]
        start = [Vector(p) for p in pts]
        hit = False
        for center, direction in zip(centers, dirs):
            normal = None
            if plan["tool"] == "PUFF":
                ok, _loc, nrm, _face = obj.closest_point_on_mesh(pts[0])
                normal = Vector(nrm) if ok else Vector((0, 0, 1))
            new = _brush_guide(pts, plan, center, direction, normal)
            if new is not None:
                pts, hit = new, True
        if hit:
            moves = [((mw @ a) - (mw @ b)).length for a, b in zip(pts, start)]
            if max(moves) > 1e-9:
                writes.append((i, pts, start[-1]))
                longest = max(longest, max(moves))
    for i, pts, _tip in writes:
        p_orig, p_ev = psys.particles[i], pev.particles[i]
        for k, co in zip(p_orig.hair_keys, pts):
            k.co_object_set(ev, evmod, p_ev, co)
    obj.update_tag()
    bpy.context.view_layer.update()
    # READ BACK off the evaluated hair, not echoed: each moved guide's tip where it now is.
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    pev = ev.particle_systems[idx]
    evmod = ev.modifiers[mod_name]
    rows, worst = [], 0.0
    for i, pts, tip in writes:
        p_ev = pev.particles[i]
        now = Vector(p_ev.hair_keys[-1].co_object(ev, evmod, p_ev))
        worst = max(worst, (now - pts[-1]).length)
        if len(rows) < 8:
            rows.append({"guide": i, "tipBefore": [round(v, 5) for v in (mw @ tip)],
                         "tipAfter": [round(v, 5) for v in (mw @ now)]})
    return {"object": obj.name, "system": psys.name, "tool": plan["tool"], "guides": len(psys.particles),
            "keysPerGuide": len(psys.particles[0].hair_keys) if len(psys.particles) else 0,
            "guidesMoved": len(writes), "longestDisplacement": round(longest, 6), "symmetry": plan["symmetry"],
            "tips": rows, "readBackMatches": worst < 1e-4,
            "edited": bool(psys.is_edited),
            "editNote": ("the guides are edited now (is_edited): a later change to the count, the hair length or the "
                         "emission regenerates them and drops the edit - Blender's own rule for groomed hair.")}



def op_set_particles(params):
    """Change a particle system that already exists - which nothing could do.

    add_particles STACKS: three calls give three systems on one object, all emitting. That is right
    for particles, where dust and sparks on one mesh is ordinary - and it meant the only way to
    change a count was to add a fourth system beside the one you meant to edit. Same absence
    set_light closed for lights and set_camera for cameras.

    THE SYSTEM IS NAMED, OR THERE MUST BE EXACTLY ONE. An object with two systems and no `system`
    argument is refused rather than guessed at, because editing the wrong emitter looks like the
    edit doing nothing.

    RETYPING IS ALLOWED, like set_light's, and the per-type key check runs against the type the
    system will BE after the call rather than the one it was - the same rule asked in the same
    place, because the writer is shared.

    params:
      object / name (str)     required
      system (str)            which system, when the object has more than one
      type (str)              EMITTER | HAIR - retyping is allowed
      ...                     every key add_particles takes except systemName
    """
    reject_unknown(params, _SET_PS_KEYS, "set_particles")
    obj = get_object(take(params, "object", "name", required=True), want_mesh=True)
    if not obj.particle_systems:
        raise MifOpError("'%s' has no particle system to change. add_particles makes one. NOTHING "
                         "was changed." % obj.name)

    want = take(params, "system", default=None, kind=str)
    if want is None:
        if len(obj.particle_systems) > 1:
            raise MifOpError(
                "'%s' has %d particle systems (%s) and no 'system' was given. Editing the wrong "
                "emitter looks exactly like the edit doing nothing, so it is refused rather than "
                "guessed. NOTHING was changed."
                % (obj.name, len(obj.particle_systems),
                   ", ".join(p.name for p in obj.particle_systems)))
        psys = obj.particle_systems[0]
    else:
        psys = obj.particle_systems.get(str(want))
        if psys is None:
            raise MifOpError("'%s' has no particle system named '%s'. It has: %s. NOTHING was "
                             "changed." % (obj.name, want,
                                           ", ".join(p.name for p in obj.particle_systems)))

    if "edit" in params:
        return _edit_guides(obj, psys, params)

    # THE TYPE THE SYSTEM WILL BE AFTER THIS CALL, not the one it is - the per-type guards inside
    # the writer have to validate against the outcome, which is set_light's rule for the same reason.
    kind = str(take(params, "type", default=psys.settings.type, kind=str)).upper()
    if kind not in ("EMITTER", "HAIR"):
        raise MifOpError("particle type must be EMITTER or HAIR, got '%s'. NOTHING was changed."
                         % kind)

    before = _particle_row(obj, psys)
    try:
        _refuse_keys_for_the_other_type(params, kind)
        notes = _apply_particle_settings(obj, psys, params, kind)
    except MifOpError as exc:
        # THE WRITER'S MESSAGES SAY "created", because it was lifted out of add_particles unchanged
        # and its format strings are left alone deliberately. Adapting the two trailing sentences
        # here is one place rather than nine, and it stops a set_particles refusal claiming a system
        # was created when nothing was.
        text = str(exc).replace("The system WAS created.", "The system WAS MODIFIED.")
        text = text.replace("NOTHING was created.", "NOTHING was changed.")
        raise MifOpError(text)

    after = _particle_row(obj, psys)
    changed = sorted(k for k in set(before) | set(after)
                     if k not in _DERIVED_ROW_KEYS and before.get(k) != after.get(k))
    out = dict(after)
    out["settingsNotes"] = notes or None
    out.update({
        "before": before,
        "changedFields": changed,
        "changedAnything": bool(changed),
    })
    return out

OPS = {
    "add_particles": op_add_particles,
    "list_particles": op_list_particles,
    "set_particles": op_set_particles,
}
