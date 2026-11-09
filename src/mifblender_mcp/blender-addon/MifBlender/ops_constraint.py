"""Object and bone constraints - the procedural half of posing, and the one that fails silently.

WHY A SEPARATE MODULE. Constraints apply to OBJECTS and to POSE BONES with the same API and the
same failure modes, so splitting them between ops_scene and ops_rig would put one rule in two
places. This addon has spent a day finding out what that costs.

WHY THEY MATTER MORE THAN aim_object. aim_object points something at a target ONCE, and the moment
either end moves the aim is stale. A Track To constraint stays correct as the target moves, which
is why it - not a computed euler - is how camera and light rigs are actually built. Nothing here
could create one.

=============================================================================
THE MEASUREMENT PROBLEM, WHICH IS THE WHOLE DESIGN OF THIS FILE
=============================================================================
A CONSTRAINT DOES NOT TOUCH obj.matrix_world. It is applied by the depsgraph when the scene
evaluates, and the object's own transform is left exactly as it was. So:

  * reading obj.matrix_world after adding a constraint reports NO CHANGE, every time, for every
    constraint that works perfectly;
  * reading back the influence value you just wrote is a proxy that cannot fail;
  * and a constraint pointing at a deleted target, or one whose influence is zero, looks
    identical to a working one from anywhere except the evaluated depsgraph.

So every op here measures through evaluated_get(depsgraph). add_constraint samples the evaluated
world matrix before and after and reports how far the object actually MOVED - which is the only
evidence that the constraint does anything at all.
"""
import bpy

from .ops_common import (MifOpError, finite_float, finite_floats, finite_int, get_object,
                         limit_rows, refuse_override_data, reject_unknown, require_editable, rna_properties, rnd, take,
                         take_bool, take_float)


def _holder(obj, bone_name, verb):
    """The object or pose bone that owns the constraint stack, with a message that names which."""
    if not bone_name:
        return obj, "object"
    if obj.type != "ARMATURE":
        raise MifOpError("'bone' was given but '%s' is a %s, not an ARMATURE. NOTHING was %s."
                         % (obj.name, obj.type, verb))
    if obj.pose is None:
        raise MifOpError("'%s' has no pose data. NOTHING was %s." % (obj.name, verb))
    pb = obj.pose.bones.get(str(bone_name))
    if pb is None:
        known = [b.name for b in obj.pose.bones][:25]
        raise MifOpError("no bone named '%s' on '%s'. Present: %s. NOTHING was %s."
                         % (bone_name, obj.name, ", ".join(known) if known else "<none>", verb))
    return pb, "bone '%s'" % pb.name


def _valid_types():
    return {i.identifier for i in bpy.types.Constraint.bl_rna.properties["type"].enum_items}


def _from_library(c, owner):
    """True when constraint c on an override owner came from the library - Blender's is_override_data, and on a build
    that reads it False for a library constraint (3.6, see ops_common.refuse_override_data) the reference's own path."""
    ovr = getattr(owner, "override_library", None) if owner is not None else None
    if ovr is None:
        return False
    if bool(getattr(c, "is_override_data", False)):
        return True
    try:
        ovr.reference.path_resolve(c.path_from_id())
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _constraint_row(c, owner=None):
    tgt = getattr(c, "target", None)
    row = _constraint_fields(c, tgt)
    # A LIBRARY OVERRIDE'S CONSTRAINTS (2026-10-05): the library's are fixed - remove_constraint refuses them, as
    # Blender's own operators do - and one added locally is the override's own. Blender's flag, and the answer read
    # through the reference where a build's flag says False for a library constraint.
    if getattr(owner, "override_library", None) is not None:
        row["isOverrideData"] = bool(getattr(c, "is_override_data", False))
        row["fromLibrary"] = _from_library(c, owner)
    return row


def _constraint_fields(c, tgt):
    return {
        "name": c.name,
        "type": c.type,
        "influence": round(float(getattr(c, "influence", 1.0)), 6),
        "mute": bool(getattr(c, "mute", False)),
        "target": tgt.name if tgt is not None else None,
        "subtarget": getattr(c, "subtarget", None) or None,
        # A CONSTRAINT WITH A MISSING TARGET IS THE SILENT FAILURE. Blender leaves it in place,
        # contributes nothing, and shows a red field in the UI that no API caller can see.
        #
        # is_valid ALONE IS NOT ENOUGH, measured on 5.0.1 and it used to be all this read. Created
        # with no target, is_valid is correctly False for 19 of the 20 target-taking types. Have
        # the target DELETED afterwards and it stays TRUE - through view_layer.update(),
        # update_tag(), a depsgraph update, and on the evaluated copy. It is simply never
        # recomputed. That is the exact case this docstring describes and the exact case the flag
        # could not report, so a deleted target read as a healthy constraint.
        "isValid": bool(getattr(c, "is_valid", True)) and not _target_missing(c),
        "needsTarget": hasattr(c, "target"),
        # WHICH OF THE TWO IT WAS. isValid false because Blender says so, and isValid false because
        # the target is gone, want different fixes - re-point it, or look at why it was deleted -
        # and collapsing them into one boolean loses that.
        "targetMissing": _target_missing(c),
    }


# PIVOT is the ONE target-taking type whose is_valid is TRUE with no target, and it is right: a
# Pivot constraint with no target pivots around the object's own point. Measured across all 29
# types on 5.0.1 - 20 have a .target, 19 of those report is_valid False when created without one.
# Treating PIVOT like the other nineteen would report a correctly configured constraint as broken,
# and a false failure is worse than a false pass because it teaches the reader to ignore the field.
_TARGET_OPTIONAL = ("PIVOT",)


def _target_missing(c):
    """True when this constraint NEEDS a target and has none."""
    if not hasattr(c, "target") or c.type in _TARGET_OPTIONAL:
        return False
    return getattr(c, "target", None) is None


def _evaluated_matrix(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    return obj.evaluated_get(dg).matrix_world.copy()


# THE FIELDS EVERY CONSTRAINT HAS. add_constraint already writes target, subtarget and influence,
# and this op deliberately does NOT: two ways to set one value is the parallel-systems shape this
# project keeps collapsing, and the read side (list_constraints) already reports them.
_CONSTRAINT_GENERIC = {"name", "influence", "target", "subtarget", "mute", "enabled",
                       "show_expanded", "is_valid", "active", "error_location", "error_rotation",
                       "is_override_data"}

_SETCON_KEYS = {"object", "name", "bone", "constraint", "settings"}


def op_set_constraint(params):
    """Write the TYPE-SPECIFIC settings of an existing constraint - the half nothing could reach.

    WHY THIS IS A REAL GAP AND NOT A CONVENIENCE. add_constraint sets target, subtarget and
    influence, which are the fields EVERY constraint shares. Everything that makes a constraint do
    its particular job was unreachable. Measured on a live 4.4: 28 constraint types carry 253
    type-specific writable properties between them, and NOT ONE type has zero - so no constraint in
    this addon could ever be fully configured.

    WHAT THAT COSTS, concretely rather than in the abstract:

      LIMIT_ROTATION and LIMIT_LOCATION are INERT until their limits are set. use_limit_x and the
      min/max values are type-specific, so the addon could add a limit constraint that limits
      nothing - present in the stack, reported by list_constraints, and doing precisely nothing.
      That is the "looks configured and does nothing" failure this codebase refuses everywhere else.

      TRACK_TO tracks along Blender's default axis. For a camera or a turret that is usually the
      wrong one, and track_axis/up_axis were unreachable.

      IK takes the default chain length, which is almost never what a rigger wants; chain_count,
      pole_target and iterations were unreachable.

    GENERIC, NOT CURATED, and that is the opposite choice from add_modifier one module over. Its
    docstring explains why it curates: "Blender ships 100+ modifier types and describing every
    property of each would be effort spent on the wrong problem". Constraints are 28 types and the
    properties are read off `bl_rna` at call time, so there is no list here to go stale - this
    accepts exactly what the running Blender accepts and refuses the rest by name, with the valid
    list in the refusal.

    params:
      object (str, required)   `name` is an alias for it, as everywhere in this addon
      bone (str)               the pose bone whose stack holds the constraint
      constraint (str)         which constraint, by name. Required - an object may hold several.
      settings (dict)          property name -> value, validated against this constraint's own type
    """
    reject_unknown(params, _SETCON_KEYS, "set_constraint")
    obj = get_object(take(params, "object", "name", required=True))
    # A LINKED OBJECT IS REFUSED, as add and remove refuse it: the write was accepted and Blender cannot save it (found
    # 2026-10-25). An override passes - editing a library constraint's settings is what an override is for.
    require_editable(obj, "change a constraint on")
    holder, label = _holder(obj, take(params, "bone", default=None, kind=str), "write")
    con_name = take(params, "constraint", required=True, kind=str)
    con = holder.constraints.get(str(con_name))
    if con is None:
        known = [c.name for c in holder.constraints]
        raise MifOpError("no constraint named '%s' on %s. Present: %s. NOTHING was changed."
                         % (con_name, label, ", ".join(known) if known else "<none>"))

    settings = take(params, "settings", default=None)
    if not isinstance(settings, dict) or not settings:
        # REFUSED RATHER THAN TREATED AS A NO-OP. A call that changes nothing and returns ok:true is
        # indistinguishable from one that worked, which is the report this whole file is careful
        # about - list_constraints exists precisely because an inert constraint looks like a live one.
        raise MifOpError("settings must be a non-empty object of property -> value. Nothing to "
                         "write, so NOTHING was changed.")

    rna = con.bl_rna.properties
    writable = sorted(p.identifier for p in rna
                      if not p.is_readonly and p.identifier not in _CONSTRAINT_GENERIC)

    # EVERY KEY VALIDATED BEFORE THE FIRST WRITE. A half-applied settings dict leaves a constraint
    # configured in a way the caller never asked for and cannot see, and this file's neighbours have
    # paid for that ordering twice today.
    staged = []
    for key, value in settings.items():
        prop = rna.get(key)
        if prop is None or prop.is_readonly:
            raise MifOpError(
                "'%s' is not a writable setting on a %s constraint. Settable here: %s. Target, "
                "subtarget and influence belong to add_constraint, not this op. NOTHING was changed."
                % (key, con.type, ", ".join(writable) if writable else "<none>"))
        if key in _CONSTRAINT_GENERIC:
            raise MifOpError(
                "'%s' is shared by every constraint and is add_constraint's to set - writing it "
                "here would be a second way to set one value. NOTHING was changed." % key)
        if prop.type == "ENUM":
            allowed = [i.identifier for i in prop.enum_items]
            got = str(value).upper()
            if got not in allowed:
                raise MifOpError("'%s' on a %s constraint takes one of: %s. Got %r. NOTHING was "
                                 "changed." % (key, con.type, ", ".join(allowed), value))
            staged.append((key, got))
        elif prop.type == "POINTER":
            # A POINTER IS RESOLVED BY NAME, the way every other target in this addon is. Passing
            # None clears it deliberately, which is how a pole target is removed.
            staged.append((key, None if value is None else get_object(str(value))))
        elif prop.type == "BOOLEAN":
            # bool() TAKES ANYTHING, and that is the bug the version matrix's fuzz pass caught in
            # this op on its first run: bool(["mif", "not-a-value"]) is True and bool({...}) is
            # True, so a list or a dict sent to a boolean setting was WRITTEN and reported ok. The
            # handler wrote something nobody asked for and said it worked. Only a real bool, or the
            # 0/1 a JSON caller may send for one, is accepted.
            if not isinstance(value, bool) and value not in (0, 1):
                raise MifOpError("'%s' on a %s constraint is a boolean - pass true or false, got "
                                 "%r. NOTHING was changed." % (key, con.type, value))
            staged.append((key, bool(value)))
        elif prop.type in ("INT", "FLOAT"):
            # finite_*, NOT a bare float()/int(). audit_unguarded_numbers caught this on the
            # first gate run and it is a real hazard here rather than a style rule: float("nan")
            # succeeds, Blender STORES the nan, and every later read of that constraint comes back
            # nan while the response agrees the call worked. int() has no 32-bit bound either, and
            # an out-of-range one raises a raw ValueError from inside the assignment instead of
            # this addon refusing with a sentence.
            if getattr(prop, "array_length", 0) > 1:
                staged.append((key, finite_floats(value, key)))
            elif prop.type == "INT":
                staged.append((key, finite_int(value, key)))
            else:
                staged.append((key, finite_float(value, key)))
        elif prop.type == "STRING":
            # str() TAKES ANYTHING TOO, and str({"a": 1}) is the text "{'a': 1}" - a value nobody
            # asked for, written and reported. Same shape as the boolean above, caught by the same
            # fuzz pass, guarded before it could be found the same way.
            if not isinstance(value, str):
                raise MifOpError("'%s' on a %s constraint is text, got %s. NOTHING was changed."
                                 % (key, con.type, type(value).__name__))
            staged.append((key, value))
        else:
            raise MifOpError("'%s' is a %s property, which this op cannot set. NOTHING was changed."
                             % (key, prop.type))

    for key, value in staged:
        setattr(con, key, value)

    # READ BACK from the constraint, never echoed from the request - the whole point of this op is
    # that a value which did not land must not report as though it did.
    applied = {}
    for key, _ in staged:
        got = getattr(con, key)
        if hasattr(got, "name"):
            got = got.name
        elif hasattr(got, "__len__") and not isinstance(got, str):
            got = [round(float(v), 6) for v in got]
        elif isinstance(got, float):
            got = round(got, 6)
        applied[key] = got

    return {"object": obj.name, "holder": label, "constraint": con.name, "type": con.type,
            "applied": applied, "appliedCount": len(applied),
            "isValid": bool(getattr(con, "is_valid", True)),
            "settableHere": writable,
            "note": "target, subtarget and influence belong to add_constraint - this op writes only "
                    "what is specific to a %s." % con.type}


def op_list_constraints(params):
    """Every constraint on an object or one of its pose bones, and whether each one is VALID.

    params:
      object (str, required)
      bone (str)    read the bone's stack instead of the object's
      limit (int)   how many rows to return. Unlimited by default; `count` is always the true
                    total and `truncated` says when rows were dropped.

    invalidCount is the field to read. A constraint whose target has been deleted stays in the
    stack, contributes nothing, and is indistinguishable from a working one everywhere except
    is_valid - Blender shows it red in the UI and reports it nowhere an API caller can reach.

    IT GROWS WITH THE STACK. Measured 2026-09-20 on 5.0.1: 60 COPY_LOCATION constraints on one
    object answered 11,577 bytes, about 190 bytes a row. A rig's control bone carries a handful,
    but a whole armature read bone by bone is the normal way to audit one, and Blender sets no
    ceiling on a stack. `limit` is the same optional, unlimited-by-default shape the other listings
    use.

    THE SUMMARY FIELDS ARE COMPUTED FROM THE FULL STACK, NOT THE PAGE. invalidCount, invalid and
    mutedCount are the reason this op exists - a caller handed the first ten rows must still learn
    that constraint 40 is broken, or the bound would have turned a diagnostic into a sample.
    """
    reject_unknown(params, {"object", "name", "bone", "limit", "rna"}, "list_constraints")
    obj = get_object(take(params, "object", "name", required=True))
    holder, label = _holder(obj, take(params, "bone", default=None, kind=str), "read")
    rows = [_constraint_row(c, obj) for c in holder.constraints]
    # rna true (2026-10-04): every readable RNA property of each constraint - its type-specific settings, the read side of
    # set_constraint, as list_modifiers {rna} is for modifiers.
    if take_bool(params, "rna", default=False):
        for row, c in zip(rows, holder.constraints):
            row["rna"] = rna_properties(c)
    invalid = [r["name"] for r in rows if not r["isValid"]]
    muted = sum(1 for r in rows if r["mute"])
    total = len(rows)
    rows, extra = limit_rows(rows, params, "constraint(s)", excludes=False)
    out = {}
    if getattr(obj, "override_library", None) is not None:
        out["overrideNote"] = ("'%s' is a library override: a constraint with fromLibrary true is the library's and "
                               "cannot be removed here (remove_constraint refuses it); one added in this file can."
                               % obj.name)
    return dict({
        "object": obj.name,
        "owner": label,
        "count": total,
        "constraints": rows,
        "invalidCount": len(invalid),
        "invalid": invalid,
        "mutedCount": muted,
    }, **extra, **out)


def op_add_constraint(params):
    """Add a constraint, and MEASURE that it actually moves the thing.

    params:
      object (str, required)
      type (str, required)     TRACK_TO | COPY_LOCATION | COPY_ROTATION | COPY_TRANSFORMS |
                               CHILD_OF | LIMIT_LOCATION | DAMPED_TRACK | IK | ... validated
                               against this Blender's own enum
      bone (str)               put it on this pose bone instead of the object
      target (str)             the object it follows
      subtarget (str)          a bone on an ARMATURE target, or a vertex group on a MESH or LATTICE target -
                               checked against the target and refused naming what it has
      influence (float)        0..1, default 1
      constraintName (str)     name the constraint. NOT `name` - see below.
      name (str)               an ALIAS FOR `object`, not for the constraint.

    THE PARAMETER THAT NAMES THE CONSTRAINT IS `constraintName`, AND THIS SAID `name`. Measured
    2026-09-04 against a live Blender, the documented spelling was unusable in every form: passing
    object and name together is refused as an alias conflict ("two names for the same parameter"),
    and passing name alone is read as the OBJECT, so it fails with "no object named 'MifConA'".
    A caller following the contract could not name a constraint at all - and the implementation
    below has always read `constraintName`.

    `name` stays an alias for `object` because that is what every op in this addon means by it and
    changing it here would make this the one that differs. The contract is what was wrong.

    THE POSTCONDITION IS MOVEMENT, MEASURED THROUGH THE DEPSGRAPH. A constraint does not touch
    obj.matrix_world, so reading the object's own transform reports no change for every constraint
    that works. This samples the EVALUATED world matrix before and after and reports the distance
    and angle the object actually moved.

    A zero delta is NOT reported as failure - a Copy Location onto something already in the right
    place legitimately moves nothing - but it IS reported, alongside the constraint's validity, so
    the difference between "nothing to do" and "pointing at a deleted object" is visible.
    """
    reject_unknown(params, {"object", "name", "type", "bone", "target", "subtarget",
                            "influence", "constraintName"}, "add_constraint")
    obj = get_object(take(params, "object", "name", required=True))
    holder, label = _holder(obj, take(params, "bone", default=None, kind=str), "added")

    kind = take(params, "type", default=None, kind=str)
    if not kind:
        raise MifOpError("'type' is required. NOTHING was added.")
    kind = str(kind).upper()
    valid = _valid_types()
    if valid and kind not in valid:
        raise MifOpError("unknown constraint type '%s'. Valid: %s. NOTHING was added."
                         % (kind, ", ".join(sorted(valid))))

    target_name = take(params, "target", default=None, kind=str)
    target = None
    if target_name:
        target = bpy.data.objects.get(str(target_name))
        if target is None:
            raise MifOpError("no target object named '%s'. NOTHING was added." % target_name)
        # Read BEFORE the self-target test - see audit_blender_shortcircuit_take. A non-string
        # `bone` was only refused when the target happened to be the object itself.
        bone_name = take(params, "bone", default=None, kind=str)
        if target is obj and not bone_name:
            raise MifOpError("'%s' cannot be constrained to itself. NOTHING was added." % obj.name)

    influence = take_float(params, "influence", default=None)
    if influence is not None and not (0.0 <= influence <= 1.0):
        raise MifOpError("influence must be between 0 and 1, got %g. NOTHING was added."
                         % influence)

    # constraintName AND subtarget, READ BEFORE constraints.new - they were read after it, so {constraintName: 5}
    # was refused with the constraint left on the object (measured on 5.0.1, 2026-09-21).
    try:
        named = take(params, "constraintName", default=None, kind=str)
        sub = take(params, "subtarget", default=None, kind=str)
    except MifOpError as exc:
        raise MifOpError("%s NOTHING was added." % str(exc).rstrip("."))

    # THE SUBTARGET IS CHECKED AGAINST THE TARGET (2026-10-05). It was written unvalidated, and on a MESH or LATTICE
    # target the same field is a vertex group on every build (properties_constraint.py): a misspelled group is stored,
    # the constraint stays valid and the object sits at the target's origin, which movedDistance cannot tell from a
    # success. A bone on an armature, a vertex group on a mesh or lattice; no other target type reads it.
    sub_kind = None
    if sub:
        if target is None:
            raise MifOpError("subtarget '%s' needs a target - it names a bone or a vertex group ON the target. NOTHING "
                             "was added." % sub)
        if target.type == "ARMATURE":
            names, sub_kind = [b.name for b in target.data.bones], "bone"
        elif target.type in ("MESH", "LATTICE"):
            names, sub_kind = [g.name for g in target.vertex_groups], "vertexGroup"
        else:
            raise MifOpError("subtarget names a bone on an ARMATURE target or a vertex group on a MESH or LATTICE "
                             "target, and '%s' is a %s - Blender would store it and read nothing. NOTHING was added."
                             % (target.name, target.type))
        if sub not in names:
            raise MifOpError("'%s' has no %s named '%s' (it has: %s). A wrong name is stored and the constraint "
                             "follows the target's origin, still reading valid. NOTHING was added."
                             % (target.name, "bone" if sub_kind == "bone" else "vertex group", sub,
                                ", ".join(names[:30]) or "none"))

    before = _evaluated_matrix(obj)

    con = holder.constraints.new(type=kind)
    if named:
        con.name = str(named)
    if target is not None:
        if not hasattr(con, "target"):
            holder.constraints.remove(con)
            raise MifOpError("a %s constraint takes no target, but one was given. NOTHING was "
                             "added." % kind)
        con.target = target
        if sub:
            con.subtarget = str(sub)
    if influence is not None:
        con.influence = influence

    bpy.context.view_layer.update()
    after = _evaluated_matrix(obj)
    moved = (after.to_translation() - before.to_translation()).length
    try:
        turned = before.to_quaternion().rotation_difference(after.to_quaternion()).angle
    except (AttributeError, ValueError):
        turned = 0.0

    row = _constraint_row(con, obj)
    if sub_kind:
        row["subtargetKind"] = sub_kind
    return {
        "object": obj.name,
        "owner": label,
        "constraint": row,
        # THE EVIDENCE. Measured through the depsgraph, because obj.matrix_world does not move when
        # a constraint is added and reading it would report every constraint as inert.
        "movedDistance": round(float(moved), 6),
        "turnedRadians": round(float(turned), 6),
        "hadEffect": moved > 1e-6 or turned > 1e-6,
        "effectNote": ("hadEffect false is not necessarily wrong - a Copy Location onto something "
                       "already in place moves nothing. Read it beside isValid: valid and inert is "
                       "'nothing to do', INVALID and inert is a target that does not resolve."),
        "constraintCount": len(holder.constraints),
    }


def op_remove_constraint(params):
    """Remove a constraint by name, and report where the thing went when it was taken off.

    params:
      object (str, required)
      constraintName (str, required)
      bone (str)

    Removing a constraint MOVES the object back to its own transform, and that movement is the
    proof the constraint was doing something. It is measured for the same reason adding one is.
    """
    reject_unknown(params, {"object", "name", "constraintName", "bone"}, "remove_constraint")
    obj = get_object(take(params, "object", "name", required=True))
    holder, label = _holder(obj, take(params, "bone", default=None, kind=str), "removed")
    want = take(params, "constraintName", default=None, kind=str)
    if not want:
        raise MifOpError("'constraintName' is required - list them with list_constraints. "
                         "NOTHING was removed.")
    con = holder.constraints.get(str(want))
    if con is None:
        known = [c.name for c in holder.constraints]
        raise MifOpError("no constraint named '%s' on %s. Present: %s. NOTHING was removed."
                         % (want, label, ", ".join(known) if known else "<none>"))

    row = _constraint_row(con, obj)
    before = _evaluated_matrix(obj)
    refuse_override_data(con, obj, "constraint")
    require_editable(obj, "remove a constraint from")
    count_before = len(holder.constraints)
    holder.constraints.remove(con)
    bpy.context.view_layer.update()
    after = _evaluated_matrix(obj)
    moved = (after.to_translation() - before.to_translation()).length

    return {
        "object": obj.name,
        "owner": label,
        "removed": row,
        "constraintCountBefore": count_before,
        "constraintCountAfter": len(holder.constraints),
        # COUNTED, not assumed. constraints.remove() returns None either way.
        "countsAgree": len(holder.constraints) == count_before - 1,
        "movedDistance": round(float(moved), 6),
        "wasDoingSomething": moved > 1e-6,
    }


_COPYCON_KEYS = {"object", "name", "from", "fromBone", "constraints", "to", "renameSubtarget"}
# Properties a copy must not write: identity and state Blender owns.
_COPY_SKIP = {"rna_type", "type", "is_valid", "active", "error_location", "error_rotation", "is_override_data",
              "show_expanded"}


def _rename_rules(raw):
    """renameSubtarget: {old: new} or [[old, new], ...] -> [(old, new)]."""
    if raw is None:
        return []
    pairs = list(raw.items()) if isinstance(raw, dict) else raw
    if not isinstance(pairs, (list, tuple)) or not pairs or any(
            not isinstance(p, (list, tuple)) or len(p) != 2 or any(not isinstance(s, str) or not s for s in p)
            for p in pairs):
        raise MifOpError("renameSubtarget is {old: new} or [[old, new], ...] - non-empty text on both sides; got %r. "
                         "NOTHING was copied." % (raw,))
    return [(a, b) for a, b in pairs]


def _derive_subtarget(src_bone, src_sub, dst_bone, rules):
    """The subtarget a copy on dst_bone gets: the rule that turns the source bone's name into the source's subtarget,
    applied to the destination bone's name (DF_arm -> FK_arm gives DF_leg -> FK_leg); kept as it is when no rule does
    (a fixed target such as root). Returns (subtarget, rule or None)."""
    for old, new in rules:
        if old in src_bone and src_bone.replace(old, new) == src_sub:
            return dst_bone.replace(old, new), (old, new)
    return src_sub, None


def op_copy_constraints(params):
    """Copy a bone's constraints onto other bones with every setting, each subtarget found by a name rule - see the MCP
    wrapper for the parameters."""
    reject_unknown(params, _COPYCON_KEYS, "copy_constraints")
    obj = get_object(take(params, "object", "name", required=True, kind=str))
    if obj.type != "ARMATURE" or obj.pose is None:
        raise MifOpError("'%s' is a %s; copy_constraints copies between pose bones of an ARMATURE. NOTHING was copied."
                         % (obj.name, obj.type))
    require_editable(obj, "constrain")
    src_name = take(params, "fromBone", "from", required=True, kind=str)
    src = obj.pose.bones.get(src_name)
    if src is None:
        raise MifOpError("'%s' has no bone '%s'. NOTHING was copied." % (obj.name, src_name))
    names = params.get("constraints")
    if names is None:
        cons = list(src.constraints)
    else:
        if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
            raise MifOpError("constraints is a non-empty list of constraint names on '%s'. NOTHING was copied." % src_name)
        missing = [n for n in names if src.constraints.get(n) is None]
        if missing:
            raise MifOpError("'%s' has no constraint(s) %s - it has %s. NOTHING was copied."
                             % (src_name, ", ".join(missing), ", ".join(c.name for c in src.constraints) or "none"))
        cons = [src.constraints[n] for n in names]
    if not cons:
        raise MifOpError("'%s' has no constraint to copy. NOTHING was copied." % src_name)
    to = params.get("to")
    if not isinstance(to, list) or not to or any(not isinstance(b, str) or not b for b in to):
        raise MifOpError("to is a non-empty list of bone names. NOTHING was copied.")
    if src_name in to:
        raise MifOpError("'%s' is the source; copying its constraints onto itself would double them. NOTHING was copied."
                         % src_name)
    absent = [b for b in to if obj.pose.bones.get(b) is None]
    if absent:
        raise MifOpError("'%s' has no bone(s) %s. NOTHING was copied." % (obj.name, ", ".join(absent[:20])))
    rules = _rename_rules(params.get("renameSubtarget"))
    # EVERY SUBTARGET RESOLVED BEFORE THE FIRST WRITE: a copy pointing at a bone that does not exist is the silent
    # failure list_constraints reports as invalid.
    plan, bad = [], []
    for b in to:
        for c in cons:
            subs = {}
            for attr in ("subtarget", "pole_subtarget"):
                val = getattr(c, attr, None)
                if not val:
                    continue
                new, rule = _derive_subtarget(src_name, val, b, rules)
                holder = getattr(c, "target" if attr == "subtarget" else "pole_target", None)
                if holder is not None and getattr(holder, "type", None) == "ARMATURE" \
                        and holder.data.bones.get(new) is None:
                    bad.append("%s.%s -> '%s' (not a bone of %s)" % (b, c.name, new, holder.name))
                subs[attr] = (new, rule)
            plan.append((b, c, subs))
    if bad:
        raise MifOpError("subtarget(s) that would not resolve: %s. NOTHING was copied." % "; ".join(bad[:10]))
    rows = []
    for b, c, subs in plan:
        pb = obj.pose.bones[b]
        new = pb.constraints.new(type=c.type)
        failed = []
        for prop in c.bl_rna.properties:
            ident = prop.identifier
            if ident in _COPY_SKIP or prop.is_readonly or prop.type == "COLLECTION":
                continue
            value = subs[ident][0] if ident in subs else getattr(c, ident)
            try:
                setattr(new, ident, value)
            except (AttributeError, TypeError, ValueError):
                failed.append(ident)
        rows.append({"bone": b, "constraint": new.name, "type": new.type, "from": c.name,
                     "subtarget": getattr(new, "subtarget", None) or None,
                     "renamedBy": ("%s -> %s" % subs["subtarget"][1]) if subs.get("subtarget", (0, None))[1] else None,
                     "isValid": bool(getattr(new, "is_valid", True)), "notCopied": failed or None})
    collections = sorted({p.identifier for c in cons for p in c.bl_rna.properties if p.type == "COLLECTION"
                          and p.identifier not in _COPY_SKIP})
    out = {"object": obj.name, "from": src_name, "constraintsPerBone": len(cons), "bones": len(to),
           "made": rows[:200], "madeCount": len(rows), "invalid": [r["bone"] + "." + r["constraint"] for r in rows
                                                                   if not r["isValid"]][:50]}
    if collections:
        out["collectionsNote"] = ("list settings (%s) are not copied - set them with set_constraint."
                                  % ", ".join(collections))
    return out


OPS = {
    "copy_constraints": op_copy_constraints,
    "list_constraints": op_list_constraints,
    "add_constraint": op_add_constraint,
    "set_constraint": op_set_constraint,
    "remove_constraint": op_remove_constraint,
}
