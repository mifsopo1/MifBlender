"""Shared helpers for MifBlender ops.

Everything in here runs on the MAIN thread (ops are only ever called from
server._execute, which is only ever called from the drain timer or inline in
background mode). bpy calls are therefore legal below this line.
"""

from __future__ import annotations

import array
import itertools
import math
import os
import re
import tempfile
import time

import bpy

# 1 Unreal unit == 1 cm, and Blender's FBX pipeline maps 1 Blender unit to 1 m
# in a cm-declared file. So a 1000 uu road is 10.0 Blender units.
# VERIFIED empirically on Blender 4.4.0: exporting with apply_unit_scale=True /
# apply_scale_options='FBX_SCALE_NONE' writes UnitScaleFactor=1.0 with the
# geometry pre-multiplied by 100, i.e. centimetre magnitudes in a centimetre
# file. See ops_mesh.FBX_EXPORT_ARGS.
UU_PER_BU = 100.0


class MifOpError(Exception):
    """A deliberate, actionable refusal. The message must name the fix.

    Raise this (not a bare Exception) whenever the caller can do something about
    it. server._execute turns it into {"ok": false, "error": ...} WITHOUT a
    traceback, because the message is the whole diagnosis.
    """


# ---------------------------------------------------------------------------
# The caller's deadline
# ---------------------------------------------------------------------------

# THE CALLER'S DEADLINE (1.4, FOUND 2026-11-07). The MCP server stamps each request with `deadline`, the wall-clock
# second its read gives up. A background Blender serves one connection at a time, so a request can wait in the accept
# backlog behind another client's open socket long past that - and then it ran: a bake_texture_set start whose caller
# had been told "timed out" 23 minutes earlier baked the session as the caller had changed it since. server._execute
# holds the request's deadline here for the length of the call, so an op that starts something lasting (a child
# process) can ask again just before it does.
CALL_DEADLINE = [None]


def caller_deadline():
    """The running request's deadline (epoch seconds), or None when its caller sent none."""
    return CALL_DEADLINE[0]


def caller_gave_up(now=None):
    """Seconds since the running request's caller stopped waiting, or None while it still waits (or sent no deadline)."""
    dl = CALL_DEADLINE[0]
    if dl is None:
        return None
    late = (time.time() if now is None else now) - dl
    return late if late > 0 else None


# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------

# clean_mesh openCollapsedUvs: a face's corners go out this far in UV units, about 0.4 of a texel on a 4096 texture, so a
# triangle of them has about 1e-8 of UV area: past NEAR_ZERO_UV_AREA and past mikktspace's zero.
_UV_OPEN_RADIUS = 1e-4
_FLT_MIN = 1.1754943508222875e-38      # mikktspace's NotZero() threshold, in float32


def _f32(x):
    return array.array("f", (x,))[0]


def _uv_collapsed(pts):
    """Does this face's UV ring hold a triangle of no area (zero in float32, mikktspace's own test)? Any three corners
    count, since an engine picks its own diagonal."""
    for i, j, k in itertools.combinations(range(len(pts)), 3):
        ax, ay = _f32(pts[j][0] - pts[i][0]), _f32(pts[j][1] - pts[i][1])
        bx, by = _f32(pts[k][0] - pts[i][0]), _f32(pts[k][1] - pts[i][1])
        if abs(_f32(_f32(ax * by) - _f32(ay * bx))) <= _FLT_MIN:
            return True
    return False


def _open_collapsed_uvs(bm, layer_name):
    """Each face with a triangle of no UV area gets its corners set on a small ring round where each one was. Corners
    are per face here, so no other face's UVs move. Returns how many faces."""
    uv = bm.loops.layers.uv.get(layer_name)
    opened = 0
    for f in bm.faces:
        ls = list(f.loops)
        pts = [(lp[uv].uv.x, lp[uv].uv.y) for lp in ls]
        if not _uv_collapsed(pts):
            continue
        for k, lp in enumerate(ls):
            a = 2.0 * math.pi * k / len(ls)
            lp[uv].uv = (pts[k][0] + _UV_OPEN_RADIUS * math.cos(a), pts[k][1] + _UV_OPEN_RADIUS * math.sin(a))
        opened += 1
    return opened


def open_collapsed_uvs_of(obj):
    """_open_collapsed_uvs on UV channel 0, the layer Unreal builds tangents from, for the ops whose new faces come out
    collapsed (a Solidify's rim, a Wireframe's struts, a decimate's survivors). 0 and no write when there is nothing to
    open, or no UV layer."""
    import bmesh   # noqa: PLC0415 - deferred, this module also loads under the offline stubs
    mesh = obj.data
    if not len(mesh.uv_layers):
        return 0
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        opened = _open_collapsed_uvs(bm, mesh.uv_layers[0].name)
        if opened:
            bm.to_mesh(mesh)
            mesh.update()
        return opened
    finally:
        bm.free()


def project_flat_uvs(obj, only_collapsed):
    """Box-project faces into UV channel 0 by each face's own dominant axis, a meter to the tile: every face when
    only_collapsed is false (a new layer, made when there was none), else only those whose UVs hold a triangle of no
    area. For a mesh converted from a type that has no real UVs: the font's fill comes out with v 0 on every corner,
    and a metaball's surface with no layer. Faces with UVs of their own keep them. Returns how many faces it wrote."""
    import bmesh   # noqa: PLC0415 - deferred, as above
    mesh = obj.data
    if not len(mesh.polygons):
        return 0
    if not len(mesh.uv_layers):
        mesh.uv_layers.new(name="UVMap")
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.normal_update()
        uv = bm.loops.layers.uv.get(mesh.uv_layers[0].name)
        wrote = 0
        for f in bm.faces:
            ls = list(f.loops)
            if only_collapsed and not _uv_collapsed([(lp[uv].uv.x, lp[uv].uv.y) for lp in ls]):
                continue
            n = f.normal
            ax = max(range(3), key=lambda k: abs(n[k]))
            a, b = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[ax]
            sign = 1.0 if n[ax] >= 0.0 else -1.0
            if ax == 1:
                sign = -sign
            for lp in ls:
                lp[uv].uv = (lp.vert.co[a] * sign, lp.vert.co[b])
            wrote += 1
        if wrote:
            bm.to_mesh(mesh)
            mesh.update()
        return wrote
    finally:
        bm.free()


def triangulate_ngons(bm, faces=None):
    """Split a bmesh's faces of more than four corners, among `faces` if given (BEAUTY), and no others. An engine splits
    an n-gon on import however it likes, so it is decided here, where gameReady can see it. Returns (how many, {new face:
    the face it came from}); UVs, materials and smoothing carry over, and the original face stays alive as one of its
    triangles."""
    import bmesh   # noqa: PLC0415 - deferred, as above
    ngons = [f for f in (bm.faces if faces is None else faces) if f.is_valid and len(f.verts) > 4]
    if not ngons:
        return 0, {}
    done = bmesh.ops.triangulate(bm, faces=ngons, quad_method="BEAUTY", ngon_method="BEAUTY")
    return len(ngons), dict(done.get("face_map") or {})


def drop_doubled_and_fin_faces(bm):
    """Mend the edges on three or more faces that an op's own welding or collapsing made. A face with every edge over two
    faces goes first: it is the wall an Array merged between two pieces, inside the solid, where nothing can see it.
    Then, on an edge still over two faces, a flap off the surface goes: the face with the most open edges, the smallest of
    equals. A face with no open edge is never taken for one, since that would open a closed surface. Returns how many
    faces went."""
    import bmesh   # noqa: PLC0415 - deferred, as above
    gone = 0
    walls = [f for f in bm.faces if all(len(e.link_faces) > 2 for e in f.edges)]
    if walls:
        gone += len(walls)
        bmesh.ops.delete(bm, geom=walls, context="FACES_ONLY")
    for _ in range(64):
        flaps = [f for e in bm.edges if len(e.link_faces) > 2 for f in e.link_faces
                 if any(len(x.link_faces) == 1 for x in f.edges)]
        if not flaps:
            break
        flap = max(flaps, key=lambda f: (sum(1 for x in f.edges if len(x.link_faces) == 1), -f.calc_area()))
        bmesh.ops.delete(bm, geom=[flap], context="FACES")
        gone += 1
    return gone


def jsonable(value, _depth=0):
    """Coerce bpy/mathutils values into something json.dumps will accept.

    A response that cannot be serialised is a silent hang from the caller's
    point of view, so this is defensive on purpose.

    APPLIED TO EVERY RESPONSE from 2026-09-04, not just to the `result` key. server._execute used to
    call this only on the rare non-dict path, so an op returning a dict - nearly all of them - went
    to json.dumps untouched. A NaN anywhere in one then reached the wire as bare `NaN`, which is not
    valid JSON: Python's json.loads accepts it and a strict parser rejects the entire frame.

    THE DEPTH CAP GUARDS AGAINST CYCLES, NOT AGAINST NESTING, and at 8 it was about to start
    truncating real answers now that every response passes through here - a repr() where the caller
    expected structure. Raised to 24: deep enough that no node tree, interface or modifier stack
    reaches it, still shallow enough that a cyclic structure dies quickly rather than eating the
    stack. The deepest structure this repo records is 4.
    """
    if _depth > 24:
        return repr(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v, _depth + 1) for v in value]
    # mathutils Vector / Euler / Matrix / Color are all iterable
    try:
        return [jsonable(v, _depth + 1) for v in value]
    except TypeError:
        pass
    for attr in ("name_full", "name"):
        if hasattr(value, attr):
            return getattr(value, attr)
    return repr(value)


def rnd(seq, places=6):
    return [round(float(v), places) for v in seq]


# ---------------------------------------------------------------------------
# Param plumbing
# ---------------------------------------------------------------------------

def take(params, *names, default=None, required=False, kind=None):
    """Read the first present key out of `names`. Aliases are first-class here
    because MifBridge's UE side accepts them too, and a caller should not have
    to remember which side it is talking to."""
    # TWO SPELLINGS OF ONE PARAMETER, TWO DIFFERENT VALUES, is a caller who believes something
    # untrue - not a preference to be resolved by argument order. Found on 2026-09-04 by asking
    # add_modifier for a modifier named MifSub and getting Subsurf: there `name` aliases the
    # OBJECT and `modifier` names the modifier, so "MifSub" was read as an object name, lost the
    # tie to "object", and was dropped in silence. The reasoning is take_bool's, thirty lines
    # below - a word it does not know is a typo, not a false, because "a false is a decision; a
    # typo is not".
    #
    # EQUAL VALUES PASS. Sending both spellings on purpose is exactly the use this function's
    # docstring says aliases exist for, and refusing it would punish the careful caller.
    present = [n for n in names if n in params and params[n] is not None]
    if len(present) > 1:
        first = params[present[0]]
        clashing = [n for n in present[1:] if params[n] != first]
        if clashing:
            # NO PROMISE ABOUT STATE. take() is a shared reader with hundreds of call sites and no
            # idea whether one of them has already written; a "NOTHING was changed" here is the
            # defect that took audit_mutate_then_deny from 0 findings to 103.
            raise MifOpError(
                "'%s' and '%s' are two names for the same parameter and were given different "
                "values (%r and %r). Send one, or send the same value for both - this op cannot "
                "tell which you meant."
                % (present[0], clashing[0], first, params[clashing[0]]))

    for name in names:
        if name in params and params[name] is not None:
            value = params[name]
            if kind is not None and not isinstance(value, kind):
                want = getattr(kind, "__name__", str(kind))
                raise MifOpError("'%s' must be %s, got %s"
                                 % (name, want, type(value).__name__))
            return value
    if required:
        raise MifOpError("'%s' is required" % names[0]
                         + (" (aliases: %s)" % ", ".join(names[1:]) if len(names) > 1 else ""))
    return default


_TRUE_WORDS = ("1", "true", "yes", "on")
_FALSE_WORDS = ("0", "false", "no", "off")


def take_bool(params, *names, default=False):
    value = take(params, *names, default=default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        # A WORD THIS DOES NOT KNOW IS A TYPO, NOT A FALSE. The test used to be
        # `word in _TRUE_WORDS`, so everything else - "ture", "flase", an empty string, a NUL -
        # came back False and the op carried on as though the caller had asked for it. Found by
        # sending sentinels to the parameters no payload covers: makeActive took a garbage string
        # and quietly meant "no". A false is a decision; a typo is not.
        if word in _FALSE_WORDS:
            return False
        raise MifOpError("'%s' must be a boolean - got %r. Accepted as text: %s for true, %s for "
                         "false. An unrecognised word is a typo rather than a no, so it is refused "
                         "instead of quietly meaning false."
                         % (names[0], value, "/".join(_TRUE_WORDS), "/".join(_FALSE_WORDS)))
    raise MifOpError("'%s' must be a boolean" % names[0])


# A SHARED READER MUST NOT PROMISE ABOUT STATE. These helpers refuse a bad value and say what is
# wrong with it - they do NOT end "NOTHING was changed", because they have no idea what their caller
# has already done. The same conclusion _socket_value reached earlier on 2026-09-04, and the reason
# it takes its tail from the caller: when take_float briefly did make that claim,
# audit_mutate_then_deny went from 0 findings to 103, every one of them an op that reads a parameter
# after a write - which is only a lie if the READER is the thing making the promise.


def take_float(params, *names, default=None, required=False):
    value = take(params, *names, default=default, required=required)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise MifOpError("'%s' must be a number, got %r" % (names[0], value))
    # NaN AND INFINITY ARE FLOATS, and every guard in this addon let them through. Python's json
    # module PARSES NaN, Infinity and -Infinity by default, so a caller can send one over the bridge
    # and float() is perfectly happy with it. Blender is too: a NaN location is accepted, reads back
    # as nan, and poisons everything that touches the object's bounds - the viewport frames nothing,
    # physics goes unstable, an exporter writes nan into the file - while every field in the
    # response agrees the call worked. There is no request a non-finite number is the answer to.
    if not math.isfinite(number):
        raise MifOpError("'%s' must be a finite number, got %r. NaN and Infinity are accepted by "
                         "Blender and poison everything that reads the object's bounds afterwards, "
                         "silently." % (names[0], value))
    return number


# Blender's integer RNA properties are 32-bit signed. Anything outside this cannot be stored, and
# what happened instead was a raw ValueError out of the assignment - see take_int.
INT32_MIN = -2147483648
INT32_MAX = 2147483647


def take_int(params, *names, default=None, required=False):
    value = take(params, *names, default=default, required=required)
    if value is None:
        return None
    # A FRACTION IS REFUSED, NOT TRUNCATED - finite_int's rule, which this reader never had. int(8.5) is 8, so
    # through the 60-odd sites that read an integer here a fraction was silently floored: measured on 5.0.1,
    # 2026-09-21, bake_texture {width: 8.5} baked 8 pixels wide, decimate_mesh {targetTris: 100.5} targeted 100,
    # face_info {limit: 2.5} returned 2 rows, and undo {steps: 1.5} undid one step and echoed askedSteps: 1 - the
    # request rewritten in the response. A whole-valued float (96.0) still passes.
    if isinstance(value, float) and math.isfinite(value) and not value.is_integer():
        raise MifOpError("'%s' must be a whole number, got %r - it is refused, not rounded or truncated."
                         % (names[0], value))
    try:
        number = int(value)
    except OverflowError:
        # int(float("inf")) raises OverflowError, which the old clause did not catch, so an infinite
        # frame number came back as a raw exception rather than a sentence.
        raise MifOpError("'%s' must be a finite integer, got %r." % (names[0], value))
    except (TypeError, ValueError):
        raise MifOpError("'%s' must be an integer, got %r" % (names[0], value))
    # RANGE-CHECKED, because Blender's int properties are 32-bit and the failure was ugly. Measured
    # on 5.0.1: set_frame_range{start: 2**40} and set_render_settings{resolutionX: 2**40} both came
    # back as a raw ValueError from deep inside the assignment - "bpy_struct: item.attr = val:
    # Scene.frame_start expected an int type" - escaping this addon's refusal contract, where every
    # other refusal is a sentence naming the fix.
    if not (INT32_MIN <= number <= INT32_MAX):
        raise MifOpError("'%s' must be between %d and %d - Blender stores integers in 32 bits and "
                         "cannot hold %r." % (names[0], INT32_MIN, INT32_MAX, value))
    return number



def finite_float(value, key):
    """One caller-supplied value as a finite float, or a refusal naming it.

    The scalar twin of finite_floats, for the conversion sites that never see take_float: a node
    socket default, a material input, a keyframe value, a UV coordinate. Each of those takes a value
    out of the caller's dict and calls float() on it, and float("nan") succeeds.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise MifOpError("'%s' must be a number, got %r." % (key, value))
    if not math.isfinite(number):
        raise MifOpError("'%s' must be a finite number, got %r. NaN and Infinity are accepted by "
                         "Blender and poison everything that reads it afterwards, silently."
                         % (key, value))
    return number


def finite_int(value, key):
    """One caller-supplied value as an int Blender can actually store, or a refusal naming it.

    The int twin of finite_float, for the conversion sites that never see take_int - create_primitive
    puts these straight into a bpy.ops operator's properties, and the operator converts them itself.
    segments: 2**40 came back as a raw ValueError from inside primitive_uv_sphere_add.

    A FRACTION IS REFUSED, NOT TRUNCATED (hand check, 2026-09-13): int(8.5) is 8, so evaluate_at_frame
    frames:[8.5] quietly sampled frame 8. Many callers pass take_float's result here, so a whole-valued float
    (96.0) must still pass; only a finite float with a fractional part is refused. NaN and infinity keep
    their own refusals below. set_keyframe already promises the same ("an INT refuses 2.5 rather than
    truncating it").
    """
    if isinstance(value, float) and math.isfinite(value) and not value.is_integer():
        # No closing full stop: callers wrap this message in their own sentence ("...: %s. NOTHING was read.").
        raise MifOpError("'%s' must be a whole number, got %r - it is refused, not rounded or truncated"
                         % (key, value))
    try:
        number = int(value)
    except OverflowError:
        raise MifOpError("'%s' must be a finite integer, got %r." % (key, value))
    except (TypeError, ValueError):
        raise MifOpError("'%s' must be an integer, got %r." % (key, value))
    if not (INT32_MIN <= number <= INT32_MAX):
        raise MifOpError("'%s' must be between %d and %d - Blender stores integers in 32 bits and "
                         "cannot hold %r." % (key, INT32_MIN, INT32_MAX, value))
    return number


def finite_floats(values, key):
    """Every element as a finite float, or a refusal naming the one that is not.

    THE VECTOR PARSERS BYPASSED take_float. They convert with a bare float(x), so the finiteness
    check added there did nothing for `location: [NaN, 0, 0]` - which Blender accepts, reads back as
    nan, and which poisons every later read of the object's bounds while the response agrees the
    call worked.

    Shared rather than repeated in each parser: the four differ legitimately (2D, 3D, defaults,
    refusal verb) and the VALIDATION is the part that was identical, exactly as check_axis_dict was.
    """
    out = []
    for i, raw in enumerate(values):
        try:
            number = float(raw)
        except (TypeError, ValueError):
            raise MifOpError("'%s'[%d] must be a number, got %r." % (key, i, raw))
        if not math.isfinite(number):
            raise MifOpError(
                "'%s'[%d] must be a finite number, got %r. NaN and Infinity are accepted by Blender "
                "and poison everything that reads the object's bounds afterwards, silently."
                % (key, i, raw))
        out.append(number)
    return out


def check_axis_dict(value, key, axes, tail="NOTHING was changed."):
    """A {x,y,z}-style dict names at least one real axis and nothing else, or it is refused.

    WHY THIS IS SHARED AND THE PARSERS ARE NOT. Six places in this addon read a vector from a dict
    and they legitimately differ - two axes for a node location, four for a quaternion, list or
    tuple returns, defaults that are sometimes zero and sometimes the object's current value.
    Forcing those into one function needs five parameters and is worse than the duplication.

    The VALIDATION is identical in all six, and on 2026-09-04 the same defect was in all six: a dict
    read with .get(axis, default) turned {"mif":"typo"} into the DEFAULT vector - the origin, or
    wherever the object already was - and reported success. In set_bone_pose a mistyped quaternion
    became a ZERO quaternion, which is not a rotation at all.

    A PARTIAL DICT STAYS LEGAL. {"z": 2} is a useful thing to write and the other axes keep their
    defaults; a dict that names NONE of them is a typo rather than a request.
    """
    unknown = sorted(set(value) - set(axes))
    if unknown or not (set(value) & set(axes)):
        raise MifOpError(
            "'%s' as an object takes %s - got %r. %s %s"
            % (key, "/".join(axes), value,
               ("Unrecognised: %s." % ", ".join(unknown)) if unknown
               else "It names none of them.", tail))


def write_rna_property(holder, key, value, tail, who, kind, deny, extra=""):
    """Write one RNA property on any Blender struct, chosen by the property's OWN declared type.

    MOVED HERE FROM ops_nodes._write_node_property on 2026-09-20, so that set_modifier could write a
    modifier's settings without a third copy of this dispatch. The node writer is now a wrapper that
    passes exactly the wording it always produced - `who` for the one sentence that names the holder
    ("node 'X' (ShaderNodeMath)"), `kind` for the "on a %s" phrases (the node's bl_idname), `deny` for
    the names that are not the caller's to set, and `extra` for a holder-specific pointer on where a
    denied name belongs.

    WIDENED IN THREE PLACES by the move, each of which was refused or raised before: a FLAG enum takes
    a list of identifiers (a string written to one raises TypeError, and that path had no handler, so
    it escaped raw); an INT array and a BOOLEAN array take a list - a Mirror modifier's use_axis is
    three booleans, and finite_int(list) or `list not in (0, 1)` refused every one.

    THE TAIL IS THE CALLER'S TO TELL: a shared refusal cannot know what its caller has already built
    (add_group_node has a node in the tree, set_node_property changed nothing), and a refusal that
    lies about the state it left is worse than one that says nothing about it.

    THE FOUR-KNOB PROBLEM it came from: add_group_node could set exactly `operation`, `dataType`,
    `domain` and `mode`, each uppercased as a string, while most of what makes a node do a particular
    thing is a property - a Math node's clamp, a Noise texture's dimensions, a Map Range's
    interpolation. DISPATCHED ON prop.type, so an enum is validated against ITS OWN items rather than
    uppercased and hoped for.

    Returns the value READ BACK - several properties clamp, and an enum set from a case-insensitive
    match is reported as the identifier Blender stored. A pointer comes back as the datablock's name.
    """
    def _num(convert, raw):
        # finite_float and friends word their own refusal and end without the caller's tail, so a NaN
        # refused here said nothing about what the call had left. Appended, once.
        try:
            return convert(raw, key)
        except MifOpError as exc:
            text = str(exc)
            raise MifOpError(text if (not tail or text.endswith(tail)) else "%s %s" % (text, tail))

    rna = holder.bl_rna.properties.get(key)
    if rna is None or key in deny or key.startswith("bl_"):
        # bl_* IS THE CLASS'S OWN METADATA - bl_description, bl_width_max and the rest are
        # RNA-writable and are not settings. Listing them buried the four or five names somebody
        # actually wants under a dozen that do nothing useful, which makes a refusal that names the
        # options no better than one that does not.
        writable = sorted(p.identifier for p in holder.bl_rna.properties
                          if not p.is_readonly and p.identifier not in deny
                          and not p.identifier.startswith("bl_"))
        raise MifOpError(
            "%s has no writable property '%s'.%s It has: %s. %s"
            % (who, key, extra, ", ".join(writable) or "(none)", tail))
    if rna.is_readonly:
        raise MifOpError("'%s' is read-only on a %s - it reports state rather than setting it. %s"
                         % (key, kind, tail))

    ptype = rna.type
    if ptype == "ENUM":
        valid = [i.identifier for i in rna.enum_items]
        if rna.is_enum_flag:
            # A FLAG ENUM TAKES A SET, and a string written to one raises. Accepted as a list of
            # identifiers, each matched case-insensitively against the property's own items.
            wanted = value if isinstance(value, (list, tuple, set)) else [value]
            picked = set()
            for item in wanted:
                match = [v for v in valid if v.upper() == str(item).upper()]
                if not match:
                    raise MifOpError("'%s' is not a valid %s for a %s on Blender %s. Available: %s. %s"
                                     % (item, key, kind, bpy.app.version_string, ", ".join(valid), tail))
                picked.add(match[0])
            setattr(holder, key, picked)
        else:
            want = str(value).upper()
            match = [v for v in valid if v.upper() == want]
            if match:
                setattr(holder, key, match[0])
            else:
                # ASSIGNED, AND BLENDER ANSWERS, when the class's own list does not have it. A DYNAMIC
                # enum has no useful list without its instance: a Render Layers node's `layer` is the
                # scene's view layers, and enum_items read off bl_rna said 'PLACEHOLDER' - so every real
                # layer name was refused. Found 2026-09-13 by the passes showcase. The TypeError Blender
                # raises names the members THIS holder offers, and the refusal quotes them.
                try:
                    setattr(holder, key, str(value))
                except TypeError as exc:
                    text = str(exc)
                    _, found, rest = text.partition("not found in (")
                    offered = ([m.strip().strip("'\"") for m in rest.rsplit(")", 1)[0].split(",") if m.strip()]
                               if found else valid)
                    raise MifOpError("'%s' is not a valid %s for a %s on Blender %s. Available: %s. %s"
                                     % (value, key, kind, bpy.app.version_string,
                                        ", ".join(offered) or text, tail))
    elif ptype == "BOOLEAN":
        # bool() ACCEPTS ANYTHING: bool({"mif":"bad"}) is True, so a caller who sent garbage got
        # use_clamp switched ON and a response saying it had been applied. Accepted-and-wrong is worse
        # than a raw exception, because the response agrees with them.
        if getattr(rna, "array_length", 0) > 1:
            if not isinstance(value, (list, tuple)) or not all(isinstance(v, bool) or v in (0, 1)
                                                              for v in value):
                raise MifOpError("'%s' on a %s is %d booleans, so it takes a list of true/false - got %r. %s"
                                 % (key, kind, rna.array_length, value, tail))
            setattr(holder, key, [bool(v) for v in value])
        else:
            if not isinstance(value, bool) and value not in (0, 1):
                raise MifOpError("'%s' on a %s is a boolean, so it takes true or false - got %r. %s"
                                 % (key, kind, value, tail))
            setattr(holder, key, bool(value))
            # A SETTER CAN DECLINE WITHOUT RAISING. Measured 2026-10-04 on 5.2.1: a cloth's point_cache.use_disk_cache
            # written true on an unsaved file reads back false - Blender keeps no disk cache without a file path - and
            # set_modifier answered ok with the false in `applied`. A boolean does not clamp, so a different read-back
            # is a refused write and is said as one.
            if bool(getattr(holder, key)) != bool(value):
                why = (" Blender keeps a disk cache only for a SAVED file, and this one has no path - save it with "
                       "save_file {repointSession: true} first." if key == "use_disk_cache" and not bpy.data.filepath
                       else "")
                raise MifOpError("Blender did not keep '%s' = %s on %s - it reads back %s.%s %s"
                                 % (key, bool(value), who, bool(getattr(holder, key)), why, tail))
    elif ptype == "INT":
        if getattr(rna, "array_length", 0) > 1:
            if not isinstance(value, (list, tuple)):
                raise MifOpError("'%s' on a %s is %d whole numbers, got %r. %s"
                                 % (key, kind, rna.array_length, value, tail))
            try:
                setattr(holder, key, [_num(finite_int, v) for v in value])
            except (TypeError, ValueError):
                raise MifOpError("'%s' on a %s takes %d whole numbers, got %r. %s"
                                 % (key, kind, rna.array_length, value, tail))
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise MifOpError("'%s' on a %s is a whole number, got %r. %s"
                                 % (key, kind, value, tail))
            try:
                setattr(holder, key, _num(finite_int, value))
            except (TypeError, ValueError):
                raise MifOpError("'%s' on a %s is a whole number, got %r. %s"
                                 % (key, kind, value, tail))
    elif ptype == "FLOAT":
        if isinstance(value, (list, tuple)):
            try:
                # finite_floats: the SCALAR branch below always had its guard and this one did not, which
                # is why {"color": [NaN, 0, 0]} through set_node_property returned ok:true with the nan
                # stored (2026-09-07).
                setattr(holder, key, _num(finite_floats, value))
            except (TypeError, ValueError):
                raise MifOpError("'%s' on a %s takes numbers, got %r. %s"
                                 % (key, kind, value, tail))
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise MifOpError("'%s' on a %s is a number, got %r. %s"
                                 % (key, kind, value, tail))
            try:
                setattr(holder, key, _num(finite_float, value))
            except (TypeError, ValueError):
                raise MifOpError("'%s' on a %s is a number, got %r. %s"
                                 % (key, kind, value, tail))
    elif ptype == "STRING":
        # str() ACCEPTS ANYTHING TOO, so a dict became the literal text "{'mif': 'bad'}" and was written
        # into the property without complaint. Same shape as the boolean above.
        if not isinstance(value, str):
            raise MifOpError("'%s' on a %s is text, got %r. %s"
                             % (key, kind, value, tail))
        setattr(holder, key, value)
    elif ptype == "POINTER":
        # A DATABLOCK BY NAME, resolved against the bpy.data collection whose element type is the
        # pointer's own fixed_type, and checked against that type before it is assigned. null CLEARS
        # the pointer: one that can be set and never unset is write-only state.
        fixed = getattr(rna, "fixed_type", None)
        if value is None:
            setattr(holder, key, None)
        else:
            if not isinstance(value, str) or not value:
                raise MifOpError("'%s' on a %s points at a %s - pass its NAME, or null to clear it; got %r. %s"
                                 % (key, kind, fixed.identifier if fixed else "datablock", value, tail))
            coll = None
            for prop in bpy.types.BlendData.bl_rna.properties:
                element = getattr(prop, "fixed_type", None)
                if prop.type == "COLLECTION" and element is not None and fixed is not None \
                        and fixed.identifier == element.identifier:
                    coll = getattr(bpy.data, prop.identifier, None)
                    break
            if coll is None:
                raise MifOpError("'%s' on a %s points at a %s, and no bpy.data collection holds that type, so "
                                 "there is nothing to resolve a name against. %s"
                                 % (key, kind, fixed.identifier if fixed else "?", tail))
            target = coll.get(value)
            if target is None:
                present = sorted(d.name for d in coll)[:25]
                raise MifOpError("'%s' on a %s points at a %s, and there is no %s named '%s'. Present: %s. %s"
                                 % (key, kind, fixed.identifier, fixed.identifier, value,
                                    ", ".join(present) or "(none)", tail))
            try:
                setattr(holder, key, target)
            except (TypeError, ValueError, AttributeError) as exc:
                # A POINTER WITH A POLL refuses a datablock of the right type but the wrong kind; Blender's
                # own message says which, so it is passed on rather than paraphrased.
                raise MifOpError("'%s' on a %s would not take '%s' (%s). %s"
                                 % (key, kind, value, exc, tail))
        got = getattr(holder, key)
        if (got is None) != (value is None) or (got is not None and got.name != value):
            # NOT the caller's tail: the assignment above has already run, so "NOTHING was changed"
            # would be the one false thing this message could say. A caller with a snapshot restores it.
            raise MifOpError("'%s' on a %s reads back %r after being given %r - the write ran and Blender did "
                             "not keep the value. The pointer is now what it reads back."
                             % (key, kind, getattr(got, "name", got), value))
        return getattr(got, "name", None)
    else:
        raise MifOpError("'%s' on a %s is a %s, which this op does not write. %s"
                         % (key, kind, ptype, tail))
    got = getattr(holder, key)
    if isinstance(got, (set, frozenset)):
        return sorted(got)
    return (list(got) if hasattr(got, "__len__") and not isinstance(got, str)
            else (round(float(got), 6) if isinstance(got, float) else got))


def _rna_json(value):
    """One RNA value as JSON - the formatting half of rna_properties. A datablock by name, a flag enum as a
    sorted list, any array (a vector, a colour, a matrix) as nested lists, a float rounded to 6 places."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, bpy.types.ID):
        return value.name
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    try:
        return [_rna_json(v) for v in value]
    except TypeError:
        return None


def _is_id_struct(struct):
    """Is this RNA struct a datablock type - does its base chain reach ID? A PointerProperty's fixed_type is the
    struct DEFINITION, not a Python class, so issubclass cannot ask it."""
    while struct is not None:
        if struct.identifier == "ID":
            return True
        struct = struct.base
    return False


def tag_fluid_domain(ds):
    """Tag a Mantaflow domain's object so the evaluated copy - the one that simulates - sees what was just written.

    cache_frame_start/end have setters and NO update callback (rna_fluid.cc, v4.4.0), so a write reached the original
    and not the copy that bakes: a domain made with cacheEnd 50, then written 240, simulated 50 frames (measured
    2026-09-21). Any write to a domain's range is followed by this."""
    ds.id_data.update_tag()


def rna_properties(holder, skip=frozenset()):
    """Every readable RNA property of `holder`, by identifier - the READ twin of write_rna_property.

    `skip` names properties that are never read at all - not read and dropped, NOT READ, because some getters crash
    Blender outright and no except clause can catch that. Measured 2026-09-21 on 4.4.0: FluidDomainSettings'
    density_grid on a domain with no simulation data is an access violation in manta_smoke_get_density, and it took
    the whole Blender down (list_modifiers {rna: true} on a fresh GAS domain). The grids are simulation data, not
    settings - a 64^3 domain's density alone is 262,144 floats - so a fluid's reader passes them here.

    WHY IT EXISTS: 2026-09-21 made add_modifier and set_modifier write ANY writable RNA property through
    write_rna_property, while the one reader of a modifier stack, list_modifiers, still decoded only the curated
    camelCase table - and nothing at all for a type outside it. Measured on 5.0.1 against d0fc6f33: a BEVEL given
    limit_method and angle_limit, a SUBSURF given quality and use_limit_surface, and a WAVE given height and width
    (a WAVE has no curated fields, so its row carried no settings at all) were all written, named in
    settingsApplied - and readable by no op afterwards. The read/write asymmetry _generated_fields' docstring calls
    a defect class, re-opened by the generic write.

    WHAT IS LEFT OUT, and why: `rna_type` (the struct describing itself); COLLECTION properties (a list of structs
    is another reader's job); and a POINTER to anything that is not a datablock - a nested settings struct such as
    a cloth's, which has its own op. A pointer to a datablock reads as its name, or None when unset. A property
    whose getter raises in this context reads as None rather than dropping the whole dump.
    """
    out = {}
    for prop in holder.bl_rna.properties:
        ident = prop.identifier
        if ident == "rna_type" or prop.type == "COLLECTION" or ident in skip:
            continue
        if prop.type == "POINTER" and not _is_id_struct(getattr(prop, "fixed_type", None)):
            continue
        try:
            out[ident] = _rna_json(getattr(holder, ident))
        except Exception:   # noqa: BLE001 - one context-dependent getter must not take the whole dump down
            out[ident] = None
    return out


def check_writable_dir(outdir):
    """Create `outdir` if it is missing and prove a file can be written in it. Raises OSError when it cannot.

    THE DIRECTORY IS CHECKED BY WRITING TO IT, because os.access reports the DACL on Windows and not the
    effective permission - it says yes for directories a write then fails on. That part is unchanged.

    WHAT CHANGED IS THE NAME. The two callers each wrote a FIXED one - export_scene `.mif_export_probe`,
    render_animation `.mif_render_probe` - and every Blender writing to that folder shared it. Measured
    2026-09-21: two Blenders (5.0.1 and 4.4.0) exporting 60 OBJs each into one writable folder at the same time had
    9 of the 120 calls refused "the output directory ... cannot be written" - WinError 5, 2 and 32, each probe
    opening, removing or finding the other's file - while 111 files were written beside them. And a file or folder
    that merely happened to carry the name made the check fail on every call.

    tempfile.TemporaryFile makes a file under a name nobody else holds (O_EXCL), and the OS deletes it when it
    is closed (O_TEMPORARY on Windows, unlinked at once elsewhere), so there is no remove step to race or to leave
    a probe behind. flush() inside the block surfaces a write that cannot land.
    """
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    with tempfile.TemporaryFile(prefix=".mif_probe_", dir=outdir) as fh:
        fh.write(b"x")
        fh.flush()


def check_output_path(raw, resolved, verb="written"):
    """Refuse an output path Blender cannot write, BEFORE the work that depends on it runs.

    Blender does not tell you until afterwards. A path containing a NUL or a control character
    collapses on the way to the filesystem: the render or bake or export RUNS, and then the save
    fails with a bare RuntimeError carrying a Python traceback - a raw exception escaping the op's
    contract, where every other refusal in this addon is a sentence.

    And where the format can be written to a relative path it does not fail at all. render_still
    silently produced a file called ".exr" - the extension alone - in the process's working
    directory, on every run of the version matrix, and it was noticed by a stray line in
    `git status` rather than by any check.

    SHARED, because the rule is identical at every op that writes a file and the failure is not
    local to any of them: set_render_settings STORES a path that render_still USES, so validating
    only at the point of use let one endpoint accept an unusable value, answer ok, and hand the
    failure to a different endpoint against a caller already told it worked.
    """
    if any(ord(ch) < 32 for ch in resolved):
        raise MifOpError("the output path contains a control character, which collapses to nothing "
                         "on the way to the filesystem - Blender does the work first and then fails "
                         "to save, or silently writes a file named after the extension alone in the "
                         "working directory. Got %r. NOTHING was %s." % (raw or resolved, verb))
    if not os.path.basename(resolved.rstrip("/\\")):
        raise MifOpError("the output path '%s' names a directory, not a file - Blender would write "
                         "a file called after the format's extension alone. Pass a full path "
                         "including a file name. NOTHING was %s." % (raw or resolved, verb))
    # every op that writes a caller's path comes through here (tools/audit_output_paths.py), so the registry needs no list
    # of writers: this only notes the path, and a record is made if the op ends ok with a new file there (artifacts.py)
    from . import artifacts
    artifacts.note_output(resolved)


def abs_output_path(raw):
    """The absolute path an image is saved to: Blender's '//' resolved against the open .blend, then anything still
    relative against the working directory - the path the answer reports.

    EVERY Image.save() THE ADD-ON MAKES GETS AN ABSOLUTE PATH. Measured on 5.0.1 (2026-10-18) with no .blend open: an
    image saved to '//x.png' or to 'x.png' wrote nothing ("Could not write image"), and the same image saved to an
    absolute path wrote it. bake_texture (and its multires bake), bake_hidden_face_map, fix_normal_map and the flipbook
    sheets passed the caller's path to Image.save() as given; they resolve it here now. save_image, spray_text_art and
    the preview encoder already resolved theirs the same way."""
    return os.path.abspath(bpy.path.abspath(str(raw)))


def file_before(path):
    """What a write to `path` would replace: None when no file is there, else {"bytes", "modified"} (UTC, ISO 8601).

    ONE RULE FOR EVERY EXPORT, decided 2026-09-28: an existing file is REPLACED by default - 1.0.0 did it, Blender's own
    exporters do it, and an agent re-exporting to one path in a loop would pay a refused round trip every time - but
    never silently. So this is taken BEFORE the write, while the old file can still be measured, and replaced_fields
    puts it in the answer; overwrite:false refuses the path by name before anything runs. The Unreal side answers with
    the same three fields (MifPrepareFileReplace in MifBridgeCommon.cpp)."""
    try:
        if not os.path.isfile(path):
            return None
        st = os.stat(path)
    except OSError:
        return None
    from .journal import iso           # the one UTC formatter the add-on writes times with
    return {"bytes": int(st.st_size), "modified": iso(st.st_mtime)}


def replaced_fields(before):
    """overwrote, and when it did, replacedBytes and replacedModified - what the write destroyed, from file_before."""
    if before is None:
        return {"overwrote": False}
    return {"overwrote": True, "replacedBytes": before["bytes"], "replacedModified": before["modified"]}


def reject_unknown(params, accepted, endpoint, notes=None):
    """Fail loudly on a key we do not understand, and say WHY when we know.

    Mirrors the UE side's RejectUnknownParams. Silently ignoring a misspelled
    param is how you get 'the op ran but did nothing' bug reports.

    `notes` IS THE HALF THE UE SIDE HAD AND THIS DID NOT, added 2026-09-07. Measured: 369 UE
    endpoints carry a keyed notes table - 738 entries - mapping a word a caller would plausibly
    guess to the reason it is not there. list_game_feature_plugins answers the key "activate" with
    "this endpoint is read-only; activating a game feature changes what is loaded in the running
    editor and the bridge does not do that", so a caller who guesses wrong is told the decision
    rather than only the accepted list. The addon had ZERO such tables. Its decisions are as
    considered - why add_fluid does not bake, why create_light_probe does not expose intensity - and
    they live in Python docstrings, which a caller holding a refusal cannot see.

    A NOTE IS NOT AN ALIAS. If a key should work, accept it; a note explains a key that deliberately
    does NOT exist, and the call still fails. Adding one is not a way to keep a caller going.

    The accepted list is still printed in full, because a note answers "why not this" and the list
    answers "then what".
    """
    unknown = [k for k in params if k not in accepted]
    if not unknown:
        return
    msg = ("%s: unknown param(s) %s. Accepted: %s"
           % (endpoint, ", ".join(sorted(unknown)), ", ".join(sorted(accepted))))
    if notes:
        # CASE-INSENSITIVE, because the guess is a guess. A caller reaching for "Bake" or "bakeNow"
        # has the same misunderstanding as one reaching for "bake", and answering only the exact
        # spelling would make the feature depend on getting the wrong word exactly right.
        lower = {str(k).lower(): v for k, v in notes.items()}
        said = []
        for k in sorted(unknown):
            why = lower.get(str(k).lower())
            if why:
                said.append("  %s: %s" % (k, why))
        if said:
            msg += "\n" + "\n".join(said)
    raise MifOpError(msg)


AXES = {"X": 0, "Y": 1, "Z": 2}


def axis_index(name, param="axis"):
    if not isinstance(name, str) or name.strip().upper() not in AXES:
        raise MifOpError("'%s' must be one of X, Y, Z (got %r)" % (param, name))
    return AXES[name.strip().upper()]


# ---------------------------------------------------------------------------
# ID naming
# ---------------------------------------------------------------------------

def assign_id_name(idblock, wanted, collection):
    """Name `idblock` `wanted`, taking a suffix if it is taken - THE SAME WAY ON EVERY BUILD.

    Returns the name it ended up with. `collection` is the bpy.data collection the block lives in
    (bpy.data.objects, bpy.data.meshes, ...), and is what decides whether a name is taken.

    THE REASON THIS EXISTS. Assigning an ID a name another ID holds does not always uniquify the
    block you assigned. Measured on all five builds 2026-09-14 (ledger B12):

        4.2.17 / 4.4.0 / 5.0.1 / 5.2.1  the ASSIGNEE always becomes '<name>.001'. Predictable.
        3.6.23                          whichever of the two sorts FIRST in the datablock list is
                                        the one renamed. So when the assignee's CURRENT name sorts
                                        after the requested one, the INCUMBENT is silently renamed
                                        instead and the assignee keeps the plain name.

    The 3.6 case defeats the standard guard, which is why it went unseen for so long: every op that
    sets nameWasSuffixed from `block.name != wanted` is asking about the block it just named, and in
    that case that block DID get the name it asked for. The check cannot fire, the caller is told all
    is well, and an object it created earlier has been renamed out from under it.

    IT ALSO HIDES BEHIND ITS OWN CONDITION, which is worth stating because it is what made this look
    like a per-KIND defect for an hour. create_primitive renames the object the operator just made,
    so the sort that decides the outcome is the OPERATOR'S DEFAULT NAME against the caller's. On 3.6,
    asking for "MifT_Clash" displaces through primitive_torus_add ("Torus" sorts after) and does not
    through primitive_cube_add ("Cube" sorts before) - same build, same call, same line of code,
    opposite outcomes. T4001 had collided deliberately since the op was written and never caught it,
    because it collided a cube. The torus row cost test_blender_creation one leaked object per run.

    ops_mesh.op_rename_object already carried this measurement and REFUSES by default, offering
    allowCollision for callers who want Blender's per-version outcome. That is right for a rename,
    where the caller named both sides. It is wrong for a create: the creation ops' documented
    contract is that a collision suffixes and carries on, T4001 collides deliberately and expects
    success, and create_camera / create_light / create_material already behave that way on every
    build because they name through .new(), which suffixes the NEW block. So this normalises 3.6
    ONTO that majority behaviour rather than refusing.

    THE SWAP USES BLENDER'S OWN SUFFIX, never one computed here. Blender's uniquifying rule has
    edges this module has no business restating - a 63-character limit below 5.0 among them - so on
    the 3.6 path the incumbent's post-clash name IS the answer, and it is handed back to the new
    block after the incumbent is made whole. The intermediate name is proven free before it is used,
    because parking on a taken one would displace a third block and rebuild the bug one step over.
    """
    wanted = str(wanted)
    incumbent = collection.get(wanted)
    if incumbent is idblock:
        return idblock.name
    idblock.name = wanted
    if incumbent is None or incumbent.name == wanted:
        # No clash, or Blender uniquified the ASSIGNEE - every build above 3.6, and 3.6 itself
        # whenever the sort fell the other way. That is the outcome the callers' own
        # nameWasSuffixed check already reports correctly, so there is nothing to undo.
        return idblock.name

    bumped = incumbent.name          # what BLENDER decided the loser should be called
    park = "_MifNameSwap"
    n = 0
    while collection.get(park) is not None:
        n += 1
        park = "_MifNameSwap%d" % n
    idblock.name = park              # frees `wanted`
    incumbent.name = wanted          # the object the caller never asked us to touch, made whole
    idblock.name = bumped            # and the new block takes the suffix Blender itself generated
    return idblock.name


# ---------------------------------------------------------------------------
# Object lookup / info
# ---------------------------------------------------------------------------

def get_object(name, want_mesh=False):
    if not isinstance(name, str) or not name:
        raise MifOpError("'object' is required (an object name; list them with list_objects)")
    obj = bpy.data.objects.get(name)
    if obj is None:
        known = [o.name for o in bpy.data.objects][:25]
        raise MifOpError("no object named '%s'. Present: %s%s"
                         % (name, ", ".join(known) if known else "<scene is empty>",
                            " ..." if len(bpy.data.objects) > 25 else ""))
    if want_mesh and obj.type != "MESH":
        raise MifOpError("object '%s' is a %s, not a MESH" % (name, obj.type))
    if want_mesh and obj.mode != "OBJECT":
        raise MifOpError("object '%s' is in %s mode. MifBlender edits mesh data "
                         "directly via bmesh and needs OBJECT mode -- leave edit mode "
                         "first." % (name, obj.mode))
    return obj


def shared_data_note(obj):
    """A response fragment naming the OTHER objects an edit to this mesh also changed.

    Alt+D makes a linked duplicate: two objects, one mesh datablock. It is how anyone lays out
    repeated geometry - a row of crates, a fence, a set of windows - so a real scene is full of
    them, and editing the mesh through one object changes every object that shares it.

    Measured on 5.0.1: clean_mesh and set_shading each changed a second object, and nothing in
    either response mentioned it - no field, no phrase.

    REPORTED, NOT REFUSED, and the difference from apply_transform is deliberate. That op refuses
    outright because applying a transform MOVES the other objects, which is never what was asked
    for - its message says "one of which you did not ask about". Editing the shared mesh is the
    opposite case: changing one crate to change all of them is the entire point of a linked
    duplicate, so refusing would remove the feature. What was missing is the caller knowing.
    """
    data = getattr(obj, "data", None)
    users = getattr(data, "users", 1) or 1
    if users <= 1:
        return {}
    others = sorted(o.name for o in bpy.data.objects
                    if o.data is data and o.name != obj.name)
    return {
        "meshSharedWith": others,
        "alsoChangedCount": len(others),
        "sharedNote": ("'%s' shares its mesh data with %d other object(s) - %s - so this edit "
                       "changed them too. That is what a linked duplicate is for; it is reported "
                       "because nothing else would tell you."
                       % (obj.name, len(others), ", ".join(others[:6]) or "(unnamed)")),
    }


def refuse_override_data(item, owner, kind, what="removed"):
    """Refuse removing a modifier or constraint that came from the LIBRARY on an override object.

    MEASURED 2026-10-04 on 3.6.23, 4.4.0 and 5.2.1: the RNA modifiers.remove of a library modifier on an override
    returns, the stack reads it gone, and after a save and reload it is back - the override cannot delete what the
    library defines, and only the operator's poll says so. is_override_data is True for those (and False for one added
    locally, which stays removable)."""
    ovr = getattr(owner, "override_library", None)
    if ovr is None:
        return
    from_lib = bool(getattr(item, "is_override_data", False))
    if not from_lib:
        # 3.6 READS is_override_data FALSE ON A LIBRARY CONSTRAINT (measured 2026-10-04: a library Copy Location on an
        # override answered False and was removed). So the reference is asked too: does the item's own path resolve on
        # the library's ID? A locally added item has a name the library does not.
        try:
            ovr.reference.path_resolve(item.path_from_id())
            from_lib = True
        except (ValueError, AttributeError, TypeError):
            from_lib = False
    if from_lib:
        src = getattr(getattr(ovr, "reference", None), "library", None)
        raise MifOpError("'%s' on '%s' is a %s that came from the library%s - a library override cannot remove it (it "
                         "reads gone and returns on the next reload). Disable it, or edit the source file. NOTHING "
                         "was %s." % (item.name, owner.name, kind,
                                      (" (%s)" % bpy.path.abspath(src.filepath)) if src is not None else "", what))


def require_editable(obj, what="changed"):
    """Refuse a write to a LIBRARY-LINKED datablock, which Blender can never save.

    Linked data belongs to another .blend and the local file may not modify it. Blender's RNA does
    not always stop you: measured on 5.0.1, transform_object moved a linked cube, reported the new
    location as fact, and the move is unsaveable and gone on the next reload. set_shading was worse
    in one way - it changed nothing at all and still answered ok.

    A LIBRARY OVERRIDE IS NOT LINKED DATA for this purpose. An override exists precisely so the
    local file can change selected properties, so obj.override_library being set means the write is
    legitimate. Checking only `.library` would refuse the case overrides were invented for.

    Nothing in this addon mentioned .library before 2026-09-04 - not once - and linking is how
    production files are assembled, so an agent driving a real scene meets this immediately.
    """
    lib = getattr(obj, "library", None)
    if lib is None or getattr(obj, "override_library", None) is not None:
        return
    raise MifOpError(
        "'%s' is LINKED from %s, so this file cannot %s it - Blender would either drop the change "
        "silently or lose it on the next reload. Make a library override for it, or open %s and "
        "edit it there. NOTHING was changed."
        % (getattr(obj, "name", "?"), getattr(lib, "filepath", "another .blend"), what,
           getattr(lib, "filepath", "the source file")))


def edit_mode_stale(obj):
    """A response fragment warning that this object's mesh data is not what the caller can see.

    WHY A READ NEEDS THIS AND A WRITE DOES NOT. A write in edit mode is refused outright by
    get_object(want_mesh=True), because the live geometry lives in the edit BMesh and anything
    written to mesh.polygons is discarded on the way out. A READ has the mirror problem and no
    guard: mesh.polygons still holds the state from the last time OBJECT mode was entered, so the
    answer is confidently wrong rather than lost.

    Measured on 5.0.1: delete a face in edit mode and face_info still reports 6 of them while the
    live mesh has 5. Nothing in the response said which number it was.

    NOT REFUSED, unlike the write path, and the difference is deliberate. This addon drives a LIVE
    editor where a person may well be in edit mode on something unrelated, and refusing every query
    for the duration would remove a capability to prevent a mistake the caller can now see. A read
    stays available and says what it is.
    """
    if getattr(obj, "mode", "OBJECT") == "OBJECT":
        return {}
    return {
        "editModeStale": True,
        "objectMode": obj.mode,
        "staleNote": ("'%s' is in %s mode, so these figures are the mesh as it was when OBJECT "
                      "mode was last left - Blender keeps live edits in a separate BMesh. Leave "
                      "edit mode and call again for the current state." % (obj.name, obj.mode)),
    }


def local_bounds(obj):
    """The LOCAL-space bounding box, read from the VERTEX DATA, not obj.bound_box.

    Local space rather than world: obj.dimensions folds in object scale, and a
    stray non-unit scale would mask geometry drift in a tiling assertion.

    Vertex data rather than obj.bound_box, because bound_box is a CACHE and it is
    stale exactly when this function matters most. MEASURED on Blender 4.4.0: a
    plane extruded via bmesh and written back with bm.to_mesh(mesh) +
    mesh.update() reports verts spanning z -0.15..0.0 while obj.bound_box still
    reports 0.0..0.0 and obj.dimensions still reports z=0. obj.update_tag() does
    NOT refresh it either; only bpy.context.view_layer.update() does.

    That is not a cosmetic difference. The round trip's X-length assert reads
    boundsLocalSizeUU straight out of object_info immediately after an edit, so a
    cached box would have it measuring the PRE-edit mesh and passing a sheared
    tile with a clean bill of health. Reading the verts cannot go stale.

    (The editing ops call view_layer.update() as well, so obj.bound_box and
    obj.dimensions are correct for every OTHER consumer. This function simply
    does not depend on their having done so.)

    VERIFIED identical to a refreshed obj.bound_box to 1e-6 on a 7-segment cone.
    """
    mesh = obj.data if obj.type == "MESH" else None
    count = len(mesh.vertices) if mesh is not None else 0
    if count:
        flat = [0.0] * (count * 3)
        mesh.vertices.foreach_get("co", flat)
        xs, ys, zs = flat[0::3], flat[1::3], flat[2::3]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    # No vertex data to read (a non-mesh object, or a mesh with no verts, whose
    # bound_box is all zeros anyway -- MEASURED).
    corners = [tuple(c) for c in obj.bound_box]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    zs = [c[2] for c in corners]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def evaluated_mesh_bounds(obj):
    """(min, max, vertex count) of the EVALUATED mesh, read from its vertex data - or None when nothing on the object
    can make it differ from the base mesh (no modifier enabled in the viewport, no shape keys).

    Blender's own obj.dimensions is not a measurement of that geometry. MEASURED 2026-09-16 on 3.6.23, 4.2.17, 4.4.0,
    5.0.1 and 5.2.1 (scratchpad measure_empty_eval_dims.py, measure_subsurf_wrapper_dims.py,
    measure_eval_bounds_read.py): a 2 BU cube whose Geometry Nodes group outputs nothing, or whose Mask modifier names
    an empty vertex group, evaluates to 0 vertices on every build, and obj.dimensions reads [0, 0, 0] on 3.6 but the
    BASE mesh's [2, 2, 2] from 4.2 on; a cube with a Subdivision Surface (levels 2) LAST in the stack reads the 1.679
    box on 3.6-4.4 and its 2.0 CAGE on 5.0 and 5.2, while the Workbench render is 1.68 wide on every build and a
    modifier after the subsurf brings 5.x back to 1.679. The evaluated object's own dimensions read the same wrong
    numbers. The evaluated vertices read the right size on every build in all of those cases, and in the controls (a
    shape key scaling x1.5 reads 3.0, an Array reads [4, 2, 2]).

    COST: an array.array buffer, not a list - 0.13-0.15 s for a Subsurf level 5 sphere of 1,589,250 evaluated vertices
    on every build, against 0.27-0.41 s for a list. On 5.x this read is what makes Blender subdivide a Subsurf last in
    the stack on the CPU, which is why a listing of many objects does not call it (see list_objects).
    """
    if obj.type != "MESH" or obj.data is None:
        return None
    if obj.data.shape_keys is None and not any(m.show_viewport for m in obj.modifiers):
        return None
    mesh = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    count = len(mesh.vertices)
    if not count:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0
    flat = array.array("f", bytes(12 * count))
    mesh.vertices.foreach_get("co", flat)
    xs, ys, zs = flat[0::3], flat[1::3], flat[2::3]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)), count


# INSTANCES ARE NOT PART OF ANY EVALUATED MESH, and every op that reads one is blind to them. MEASURED 2026-09-16 on
# 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1, identically, through the ops (scratchpad measure_gn_instances_reads.py,
# measure_instances_through_ops.py, measure_instance_sources_convert.py): a UV sphere (32x16) whose Geometry Nodes tree
# joins it with Instance on Points - a 0.1 cube on each of its 482 vertices, nothing realized - draws 482 depsgraph
# instances and evaluates to the sphere's 512 faces. select_faces, face_info, vertex_info and mesh_stats counted the
# sphere; closest_point_on_mesh and the object-form ray_cast answered z 1.0 on the sphere through the pole cube (top
# 1.05), which scene.ray_cast hits; objects_overlap called a probe cube cutting that pole cube DISJOINT and
# audit_placement listed no overlap; object_info's dimensionsBU read the sphere's 2.0. Realized, the same tree evaluates
# to 3404 faces and every read sees the cubes. Counting is cheap - 100,000 instances walked in 0.06 s.
def drawn_instances(obj, depsgraph=None):
    """How many depsgraph instances `obj` draws - Geometry Nodes instances, particles rendered as objects or collections,
    vertex and face instancing. None of them is part of its evaluated mesh. 0 without walking when nothing on the
    object can draw one."""
    can = (getattr(obj, "instance_type", "NONE") != "NONE"
           or any(m.show_viewport and m.type in ("NODES", "PARTICLE_SYSTEM") for m in getattr(obj, "modifiers", ())))
    if not can:
        return 0
    dg = depsgraph or bpy.context.evaluated_depsgraph_get()
    name = obj.name
    return sum(1 for inst in dg.object_instances
               if inst.is_instance and inst.parent is not None and inst.parent.original.name == name)


def instances_note(obj, what, depsgraph=None):
    """A sentence when `obj` draws instances that `what` (this op's answer, in words) cannot see, else None."""
    n = drawn_instances(obj, depsgraph)
    if not n:
        return None
    return ("'%s' also draws %d instance(s) - Geometry Nodes instances, particles or instancing - which are not part of "
            "its evaluated mesh, so they are missing from %s (measured on every build). A Realize Instances node at "
            "the end of a Geometry Nodes tree makes them geometry." % (obj.name, n, what))


def mesh_counts(obj):
    mesh = obj.data
    try:
        mesh.calc_loop_triangles()
        tris = len(mesh.loop_triangles)
    except Exception:
        tris = -1
    return {
        "verts": len(mesh.vertices),
        "edges": len(mesh.edges),
        "faces": len(mesh.polygons),
        "tris": tris,
    }


def bone_segment(obj, bone, armature, what):
    """(armature object, bone name, head, tail) in WORLD space from the bone's REST position - where a base mesh is.

    The armature is `armature` by name, else the mesh's first Armature modifier's object, else its armature parent.
    Shared by mesh_quality's boundary loops and bisect_plane's nearBone, so a bone is found one way. Refuses (what:
    'measured' or 'changed' ends the sentence) before anything is done."""
    if armature is not None:
        arm = bpy.data.objects.get(armature)
        if arm is None or arm.type != "ARMATURE":
            raise MifOpError("%r is not an armature object. NOTHING was %s." % (armature, what))
    else:
        mod = next((m for m in obj.modifiers if m.type == "ARMATURE" and m.object is not None), None)
        arm = mod.object if mod is not None else (obj.parent if obj.parent is not None
                                                  and obj.parent.type == "ARMATURE" else None)
        if arm is None:
            raise MifOpError("'%s' has no Armature modifier with an object and no armature parent - pass armature. "
                             "NOTHING was %s." % (obj.name, what))
    b = arm.data.bones.get(bone)
    if b is None:
        raise MifOpError("armature '%s' has no bone %r (it has %d). NOTHING was %s."
                         % (arm.name, bone, len(arm.data.bones), what))
    head, tail = arm.matrix_world @ b.head_local, arm.matrix_world @ b.tail_local
    if (tail - head).length < 1e-9:
        raise MifOpError("bone %r has zero length, so it has no axis. NOTHING was %s." % (bone, what))
    return arm, bone, head, tail


def mesh_shape(verts):
    """Every vertex position, rounded and sorted - what "the mesh is unchanged" has to compare.

    COUNTS ARE NOT A SHAPE. Half a cube with its cut capped is 8 verts, 12 edges and 6 faces, exactly
    the cube it was cut from, and a box cut down by a slab is the same again. Two postconditions
    compared counts alone: bisect_plane REFUSED a cut that had worked and said the plane had missed,
    and boolean_op reported changed:false about a cut that had happened. Found 2026-09-12 by
    audit_blender_inert_params, whose clearInner and clearOuter rows on the default cube were
    refused with the flag on, every run.

    Rounded to 1e-6 so a to_mesh round trip's float noise is not a change, and sorted so vertex
    ORDER is not one either. Takes bm.verts or mesh.vertices - both have .co. It walks every vertex,
    so ask it only once the counts already tie.
    """
    return sorted((round(v.co.x, 6), round(v.co.y, 6), round(v.co.z, 6)) for v in verts)


# THE SHADOW FLAG HAS MOVED BETWEEN VERSIONS, so it is looked up rather than assumed. Newest name
# first, exactly as op_add_particles handles show_instancer_for_render - which is in this addon
# because writing the old name raised AttributeError and took down a live build.
SHADOW_ATTRS = ("use_shadow", "use_shadow_jitter")


def shadow_attr(holder):
    """The name this Blender uses for the light's shadow toggle, or None if it has none."""
    for attr in SHADOW_ATTRS:
        if hasattr(holder, attr):
            return attr
    return None


def refuse_unsupported_shadow(params, verb):
    """Refuse `shadow` up front on a Blender that has no such property, rather than ignoring it.

    create_light used to accept the key and write it only `if hasattr(data, attr)`, so on a build
    where the property had moved the caller asked for shadows off, got shadows on, and was told
    nothing. A key this addon ACCEPTS and does not apply is the invoke_editor_tab shape, and the
    house rule is that it is refused rather than silently reinterpreted.

    Asked of the RNA CLASS, not an instance, so it can run before anything is created.
    """
    if "shadow" not in params:
        return
    # `in`, not hasattr: bl_rna.properties is a collection keyed by property name, so hasattr on it
    # asks whether the COLLECTION has an attribute of that name, which is always False and would
    # have made this guard fire on every Blender. Caught by writing it wrong first.
    #
    # AND THE CONCRETE TYPES, NOT ONLY THE BASE. Measured on 3.6.23: `use_shadow` is NOT on
    # bpy.types.Light there and IS on PointLight, SpotLight, SunLight and AreaLight - it moved onto
    # the base class in a later build. Asking only the base refused `shadow` on EVERY 3.6 light while
    # list_lights read the same property off the instance perfectly well and reported
    # shadowAttr "use_shadow", so the reader and the writer disagreed about whether the property
    # existed, and the writer's refusal said it did not. That took out fourteen assertions in
    # test_blender_scene on 3.6 and made it impossible to turn a light's shadow off on a supported
    # build. Ledger B9.
    #
    # The base is still checked first because on 4.x that is where it lives and one lookup settles
    # it; the subclasses are the 3.6 path. Either way this runs before anything is created, which is
    # the property this guard exists to keep.
    for holder in (bpy.types.Light, bpy.types.PointLight, bpy.types.SpotLight,
                   bpy.types.SunLight, bpy.types.AreaLight):
        if holder is None:
            continue
        if any(a in holder.bl_rna.properties for a in SHADOW_ATTRS):
            return
    raise MifOpError("this Blender's Light has no shadow toggle this op knows how to write (tried "
                     "%s), so `shadow` would have been silently ignored. Control shadows through "
                     "the render engine's own settings instead. NOTHING was %s."
                     % (", ".join(SHADOW_ATTRS), verb))


# THE INFLUENCE PROPERTIES, AND WHERE EACH ONE EXISTS. Two axes, because a property can be missing
# for two different reasons and they need different answers from the caller: this BUILD may not have
# it (transmission_factor is 4.2 and later) or this light TYPE may not (spread is AREA only, the
# cascades are SUN only). set_light_influence's refusal already distinguishes them; the reader below
# reports the same distinction rather than leaving an absent key ambiguous.
#
# HERE RATHER THAN IN ops_lightcam SINCE 2026-09-15, which is the same move the panorama fields got
# at the bottom of this file and for the same reason: light_readback reads them too, and ops_lightcam
# already imports this module. The point is that ONE map drives both the write and the read, so a
# property added to set_light_influence is reported by list_lights without anybody remembering to.
#     {jsonKey: (blenderAttr, light types or None for all, build availability)}
INFLUENCE_MAP = {
    "cutoffDistance":     ("cutoff_distance", None, "every build"),
    "useCustomDistance":  ("use_custom_distance", None, "every build"),
    "volumeFactor":       ("volume_factor", None, "every build"),
    "transmissionFactor": ("transmission_factor", None, "4.2 and later"),
    "spread":             ("spread", ("AREA",), "every build"),
    "softFalloff":        ("use_soft_falloff", ("POINT", "SPOT"), "4.2 and later"),
    "square":             ("use_square", ("SPOT",), "every build"),
    "showCone":           ("show_cone", ("SPOT",), "every build"),
    "cascadeCount":       ("shadow_cascade_count", ("SUN",), "every build"),
    "cascadeMaxDistance": ("shadow_cascade_max_distance", ("SUN",), "every build"),
    "cascadeExponent":    ("shadow_cascade_exponent", ("SUN",), "every build"),
    "cascadeFade":        ("shadow_cascade_fade", ("SUN",), "every build"),
}
INFLUENCE_BOOL = {"useCustomDistance", "softFalloff", "square", "showCone"}
INFLUENCE_INT = {"cascadeCount"}


def light_influence_readback(data):
    """The influence block off the datablock, plus what this light does not have and why.

    THE READ HALF THAT DID NOT EXIST. set_light_influence writes twelve properties and, until
    2026-09-15, every one of those attribute names appeared in ops_lightcam.py and nowhere else -
    inside that op's own map. The write was verified (it getattrs the stored value back and compares
    it against what was asked) but there was no SECOND way in: an agent that reconnects, or inherits
    a file somebody else built, could set a light's reach and never ask what it was.

    AN ABSENT KEY IS NOT A ZERO, so the omissions are named rather than left to inference. A caller
    seeing no `spread` cannot otherwise tell "this is a point light" from "nobody set it" from "this
    build is too old" - three states that look identical and want different responses.

    AND cutoffDistance IS REPORTED BESIDE ITS GATE, which is the whole reason the pair is in one
    block. Blender stores cutoff_distance whether or not use_custom_distance is on, so the number
    alone is a distance that may be applying to nothing - the same shape as physics_info reporting
    collisionMargin without use_margin, found the same day.
    """
    kind = str(data.type)
    have = data.bl_rna.properties.keys()
    block, missing = {}, {}
    # THE SAME TWO SENTENCES set_light_influence REFUSES WITH, deliberately. Its refusal says
    # "light TYPE, not this Blender" for one axis and names the build for the other, and a reader
    # who learns to tell them apart from a refusal should not have to learn a second vocabulary to
    # tell them apart from a listing. One contract, one wording, both directions.
    for key, (attr, types, build) in INFLUENCE_MAP.items():
        if types is not None and kind not in types:
            missing[key] = ("the light TYPE, not this Blender build: %s lights do not have it, "
                            "it is %s only" % (kind, "/".join(types)))
            continue
        if attr not in have:
            missing[key] = ("this Blender build, not the light TYPE: the property arrived in %s"
                            % build)
            continue
        value = getattr(data, attr)
        if key in INFLUENCE_BOOL:
            block[key] = bool(value)
        elif key in INFLUENCE_INT:
            block[key] = int(value)
        else:
            block[key] = round(float(value), 6)
    return block, missing


def light_readback(obj, data):
    """What the light IS, off the datablock. One reader for create, set and list.

    Written once on purpose. Three near-identical response builders would drift, and a response
    that disagrees with itself between the op that made a light and the op that lists it is the
    kind of thing nobody notices until a caller diffs them.
    """
    out = {
        "name": obj.name,
        "dataName": data.name,
        "type": data.type,
        "location": rnd(list(obj.matrix_world.to_translation())),
        "rotationEuler": rnd(list(obj.matrix_world.to_euler())),
        "energy": round(float(data.energy), 6),
        "color": rnd(list(data.color)),
        "diffuseFactor": round(float(getattr(data, "diffuse_factor", 1.0)), 6),
        "specularFactor": round(float(getattr(data, "specular_factor", 1.0)), 6),
    }
    if data.type in ("POINT", "SPOT"):
        out["shadowSoftSize"] = round(float(data.shadow_soft_size), 6)
    if data.type == "SPOT":
        out["spotSize"] = round(float(data.spot_size), 6)
        out["spotBlend"] = round(float(data.spot_blend), 6)
    if data.type == "AREA":
        out["size"] = round(float(data.size), 6)
        out["sizeY"] = round(float(getattr(data, "size_y", data.size)), 6)
        out["shape"] = data.shape
    if data.type == "SUN":
        out["angle"] = round(float(data.angle), 6)
    # COLOUR TEMPERATURE, reported wherever this Blender has it. create_light and set_light write it there
    # and, until 2026-09-13, nothing read it back - written and unverifiable, the same shape as
    # add_soft_body's goalSpring. Looked up on the RNA the way _check_light_temperature does, because
    # hasattr(bpy.types.Light, "use_temperature") is False even on the builds that have the property.
    if "use_temperature" in data.bl_rna.properties.keys():
        out["useTemperature"] = bool(data.use_temperature)
        out["temperature"] = round(float(data.temperature), 3)
    # Reported when this Blender has the property at all, through the same lookup the writers use,
    # so the read and the write can never disagree about which attribute the shadow flag lives on.
    _sh = shadow_attr(data)
    if _sh is not None:
        out["shadow"] = bool(getattr(data, _sh))
        out["shadowAttr"] = _sh
    # THE SECOND WAY IN. Every property set_light_influence writes is now readable through the op
    # that lists lights, which is what makes a scene able to check its claim through a DIFFERENT op
    # than the one that wrote it - the wave-11 bar, and the thing that was blocking a `falloff`
    # showcase from being worth filming.
    out["influence"], _inf_missing = light_influence_readback(data)
    if _inf_missing:
        out["influenceNotOnThisLight"] = _inf_missing
    # LIGHT LINKING, where this Blender has it (set_light_linking refuses by name where it does not). That op writes the
    # receiver and blocker collections and reported them in its own reply only: audit_write_only_state listed both as
    # set and never read (2026-09-21), so a light linked earlier, or by hand, could not say what it was limited to.
    linking = getattr(obj, "light_linking", None)
    if linking is not None:
        out["lightLinking"] = {
            "receiverCollection": getattr(linking.receiver_collection, "name", None),
            "blockerCollection": getattr(linking.blocker_collection, "name", None),
        }
    return out


# PANORAMA SETTINGS MOVED between versions: they were on camera.cycles in 3.x and are on the camera data itself in
# 4.x+. Both are tried, newest first, the same way the light shadow flag and the object ray-visibility flags are
# handled - a hardcoded single location is a silent no-op on half the Blenders this addon supports. HERE rather than
# in ops_lightcam since 2026-09-13: camera_readback reads them too, and ops_lightcam already imports this module.
PANO_FIELDS = (
    ("panoramaType", "panorama_type"),
    ("fisheyeFov", "fisheye_fov"),
    ("fisheyeLens", "fisheye_lens"),
    ("latitudeMin", "latitude_min"),
    ("latitudeMax", "latitude_max"),
    ("longitudeMin", "longitude_min"),
    ("longitudeMax", "longitude_max"),
)


def pano_holder(cam):
    """Where this Blender keeps the panorama settings, or None if it keeps them nowhere."""
    if hasattr(cam, "panorama_type"):
        return cam
    cyc = getattr(cam, "cycles", None)
    if cyc is not None and hasattr(cyc, "panorama_type"):
        return cyc
    return None


def camera_readback(obj, cam):
    """What the camera IS, off the datablock. One reader for object_info, set_camera and
    list_cameras - the same rule light_readback follows, and for the same reason: three response
    builders for one object type drift, and nobody notices until a caller diffs two of them.

    FOV is DERIVED here rather than left to the caller. It is the number that actually determines
    framing, it depends on sensor fit as well as focal length, and getting it wrong is the usual
    reason a camera "looks right in the numbers" and wrong in the render.
    """
    dof = getattr(cam, "dof", None)
    fit = getattr(cam, "sensor_fit", "AUTO")
    sw = float(cam.sensor_width)
    sh = float(getattr(cam, "sensor_height", sw))
    # AUTO fits to the LARGER render dimension, which is why a portrait render silently changes
    # which sensor dimension the focal length is measured against.
    r = bpy.context.scene.render
    rx, ry = int(r.resolution_x), int(r.resolution_y)
    if fit == "VERTICAL" or (fit == "AUTO" and ry > rx):
        fov_sensor = sh if fit == "VERTICAL" else sw
    else:
        fov_sensor = sw
    fov = 2.0 * math.atan(fov_sensor / (2.0 * float(cam.lens))) if cam.lens else 0.0
    out = {
        "type": cam.type,
        "lensMM": round(float(cam.lens), 6),
        "sensorWidthMM": round(sw, 6),
        "sensorHeightMM": round(sh, 6),
        "sensorFit": fit,
        "fovRadians": round(fov, 6),
        "fovDegrees": round(math.degrees(fov), 4),
        "clipStart": round(float(cam.clip_start), 6),
        "clipEnd": round(float(cam.clip_end), 6),
        "shiftX": round(float(cam.shift_x), 6),
        "shiftY": round(float(cam.shift_y), 6),
        "orthoScale": round(float(cam.ortho_scale), 6),
        "dofEnabled": bool(getattr(dof, "use_dof", False)),
        "dofDistance": round(float(getattr(dof, "focus_distance", 0.0)), 6),
        "dofFocusObject": (dof.focus_object.name
                           if dof is not None and getattr(dof, "focus_object", None) else None),
        "fStop": round(float(getattr(dof, "aperture_fstop", 0.0)), 6),
        # Whether THIS camera is the one a render will use. Obtainable nowhere else in the addon,
        # and the first thing anybody asks about a camera.
        "isSceneCamera": bpy.context.scene.camera is obj,
    }
    # A PANORAMIC CAMERA'S OWN SETTINGS, read where set_camera_panorama writes them and in the shape it reports them.
    # They were readable nowhere: list_cameras and object_info said type PANO and stopped, so what set_camera_panorama
    # stored rested on its own response and the hand check could only call it PARTIAL (2026-09-13).
    if cam.type == "PANO":
        holder = pano_holder(cam)
        pano = {"storedOn": None if holder is None else ("camera" if holder is cam else "camera.cycles")}
        for key, attr in PANO_FIELDS:
            if holder is not None and hasattr(holder, attr):
                v = getattr(holder, attr)
                pano[key] = v if isinstance(v, str) else round(float(v), 6)
        out["panorama"] = pano
    return out


def material_slot_names(obj):
    """An object's material slot names in order, or None when its type cannot hold a material.

    ONE ANSWER FOR THE WRITER AND EVERY READER. Slots were reported for MESH alone, by object_info
    and list_objects both, and set_material_slots refused everything else - while a text object, a
    curve, hair curves, a point cloud and a volume hold slots exactly as a mesh does, and set_text's
    and create_curves' own hints sent callers to set_material_slots for them. Found 2026-09-11 when
    the showcase could not colour its title. Grease pencil is left out: its materials need
    grease-pencil settings that set_material_slots does not create.
    """
    data = obj.data
    if data is None or not hasattr(data, "materials") or obj.type in ("GPENCIL", "GREASEPENCIL"):
        return None
    return [(s.material.name if s.material else None) for s in obj.material_slots]


# WHERE EACH VISIBILITY FLAG LIVES ON THIS BLENDER. Two places to look: the object itself, or its
# `cycles` sub-struct on builds that keep the ray flags there. A hardcoded single location is a
# silent no-op on half the builds this addon supports, which is the same reason the light shadow
# flag and the panorama fields are looked up rather than remembered.
#
# HERE RATHER THAN IN ops_scene SINCE 2026-09-15, third time this move has been made in this file
# and for the identical reason: object_info reads them too, and ops_scene already imports this
# module. ops_scene keeps its _VISIBILITY_KEYS literal - parity_check resolves accepted-key sets
# statically and is fail-closed - and keeps the import-time guard that compares the two.
#
# shadowCatcher WAS SPELT indirectOnly until 2026-09-28, and wrote Object.is_shadow_catcher under that name (the
# 2026-09-21 pack-rule review). Indirect Only is not an object setting on any build: it is LayerCollection.indirect_only,
# per collection per view layer, which set_collection_visibility writes. Measured on 5.2.1: set_object_visibility
# {indirectOnly: true} turned is_shadow_catcher True, and the object has no indirect_only attribute at all.
VIS_FLAGS = (
    ("hideViewport", ("hide_viewport",), None),
    ("hideRender", ("hide_render",), None),
    ("visibleCamera", ("visible_camera",), "camera"),
    ("visibleDiffuse", ("visible_diffuse",), "diffuse"),
    ("visibleGlossy", ("visible_glossy",), "glossy"),
    ("visibleTransmission", ("visible_transmission",), "transmission"),
    ("visibleVolumeScatter", ("visible_volume_scatter",), "scatter"),
    ("visibleShadow", ("visible_shadow",), "shadow"),
    ("holdout", ("is_holdout",), "is_holdout"),
    ("shadowCatcher", ("is_shadow_catcher",), "is_shadow_catcher"),
)


def vis_target(obj, attrs, cycles_attr):
    """(holder, attribute) for a visibility flag on this Blender, or None if it has none."""
    for a in attrs:
        if hasattr(obj, a):
            return obj, a
    cyc = getattr(obj, "cycles", None)
    if cycles_attr and cyc is not None and hasattr(cyc, cycles_attr):
        return cyc, cycles_attr
    return None


def visibility_readback(obj):
    """Every visibility flag this Blender exposes for the object, by our key names.

    THE SECOND WAY IN, from 2026-09-15. set_object_visibility writes eight ray flags and reads them
    back honestly - before and after, off the datablock - but this function was called from three
    lines and all three were inside that op. So an agent could set visibleGlossy and never ask what
    it was: it had to REMEMBER, and an agent's memory of a call it did not make is nothing. That is
    the same discoverability argument that put describe_op in this bridge.

    Stopping a softbox appearing as a white rectangle in every reflection is the most common
    adjustment in product lighting, and "is it off on this object already" was unanswerable.
    """
    out = {}
    for key, attrs, cyc in VIS_FLAGS:
        tgt = vis_target(obj, attrs, cyc)
        if tgt is not None:
            out[key] = bool(getattr(tgt[0], tgt[1]))
    # THE VIEW LAYER'S HIDE - the outliner's eye - is not a datablock flag but a method on the object for the active view
    # layer, so the ten flags above never carried it. Diagnosing a bake that refused 'No valid selected objects' needed a
    # raw-Python hide_get() (2026-10-04, the garment work): the high-poly source was hidden there, and hide_viewport was
    # False. visibleInViewLayer is Blender's own answer to "is it drawn": every hide flag and the collections above it.
    # An object outside the active view layer raises on both, and reads None.
    for key, method in (("hiddenInViewLayer", "hide_get"), ("visibleInViewLayer", "visible_get")):
        try:
            out[key] = bool(getattr(obj, method)())
        except (RuntimeError, AttributeError, TypeError):
            out[key] = None
    return out


def _is_identity(matrix, tol=1e-6):
    return all(abs(matrix[i][j] - (1.0 if i == j else 0.0)) <= tol for i in range(4) for j in range(4))


def _sharp_edge_count(mesh):
    """Edges marked sharp. use_edge_sharp exists on every supported build (3.6-5.2, read through to the sharp_edge
    attribute), and foreach_get reads it in C - a Python loop would cost a scan of every edge per object_info."""
    n = len(mesh.edges)
    if not n:
        return 0
    buf = [False] * n
    try:
        mesh.edges.foreach_get("use_edge_sharp", buf)
    except (AttributeError, TypeError, RuntimeError):
        return None
    return buf.count(True)


def children_by_parent():
    """{parent name: [child names, sorted]} for the whole file, in ONE pass - for callers that run object_info over many
    objects. Object.children is a Python property that scans bpy.data.objects on every read: measured 2026-09-21 at
    5,000 objects, 0.55 ms a call on 5.0.1 and 0.36 ms on 3.6.23, so a detail listing reading it per object would be
    quadratic - about 3 s at 5,000 objects and 45 s at 20,000."""
    out = {}
    for o in bpy.data.objects:
        if o.parent is not None:
            out.setdefault(o.parent.name, []).append(o.name)
    for names in out.values():
        names.sort()
    return out


def rotation_is_identity(obj):
    """Whether the rotation obj's rotation_mode USES is no rotation - an Euler order reads rotation_euler, QUATERNION
    rotation_quaternion (w of either sign with no xyz), AXIS_ANGLE an angle of 0."""
    mode = obj.rotation_mode
    if mode == "QUATERNION":
        q = obj.rotation_quaternion
        return abs(abs(q[0]) - 1.0) < 1e-6 and all(abs(v) < 1e-6 for v in q[1:])
    if mode == "AXIS_ANGLE":
        return abs(obj.rotation_axis_angle[0]) < 1e-6
    return all(abs(v) < 1e-6 for v in obj.rotation_euler)


# THE ROTATION WRITERS, ONE PLACE. An object evaluates rotation_euler ONLY when its rotation_mode is an Euler order; a
# QUATERNION or AXIS_ANGLE object ignores it entirely, and glTF import makes every object QUATERNION. transform_object
# learned that first and converted inline; set_light, set_camera and aim_object went on writing rotation_euler whatever
# the mode until 2026-09-28 - measured on 5.2.1 with a glTF spot light and camera: set_light {rotation} and set_camera
# {lookAt} answered ok with changedFields [] and the -Z axis where it was, and aim_object refused its own write 9.46 deg
# off (test_blender_review_defects RD116-RD119). So the three ops and transform_object write through these two.
#
# CONVERTED, NOT RE-MODED. The object's rotation_mode is kept and the rotation written into the property that mode
# reads. Switching the object to XYZ instead would stop its existing quaternion keys and drivers moving it, and change
# what an exporter writes - a side effect nobody asked for.
EULER_ORDERS = ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX")


def rotation_euler_values(obj):
    """obj's rotation as the three angles write_rotation takes: rotation_euler as it is in an Euler mode (in that
    order), else the rotation the mode uses - rotation_quaternion or rotation_axis_angle - as an XYZ Euler. It is the
    default for an axis a partial {x,y,z} leaves out."""
    import mathutils
    mode = obj.rotation_mode
    if mode in EULER_ORDERS:
        return tuple(obj.rotation_euler)
    if mode == "QUATERNION":
        q = obj.rotation_quaternion.copy()
    else:
        aa = obj.rotation_axis_angle
        q = mathutils.Quaternion((aa[1], aa[2], aa[3]), aa[0])
    return tuple(q.to_euler("XYZ"))


def write_rotation(obj, values, relative=False):
    """Write three angles (radians) as obj's rotation, into the property its rotation_mode USES; returns its name.

    transform_object's rule, for every op with a `rotation` {x,y,z}: an Euler mode takes the angles as they are, in its
    own order - what object_info's rotationEulerRad reports for it - and a QUATERNION or AXIS_ANGLE object takes them as
    an XYZ Euler, converted. `relative` adds them (Euler) or composes them onto the current rotation (the other two)."""
    import mathutils
    mode = obj.rotation_mode
    if mode in EULER_ORDERS:
        obj.rotation_euler = [a + b for a, b in zip(obj.rotation_euler, values)] if relative else values
        return "rotation_euler"
    q = mathutils.Euler(values, "XYZ").to_quaternion()
    if mode == "QUATERNION":
        obj.rotation_quaternion = (obj.rotation_quaternion @ q) if relative else q
        return "rotation_quaternion"
    if relative:
        cur = obj.rotation_axis_angle
        q = mathutils.Quaternion((cur[1], cur[2], cur[3]), cur[0]) @ q
    axis, angle = q.to_axis_angle()
    obj.rotation_axis_angle = (angle, axis[0], axis[1], axis[2])
    return "rotation_axis_angle"


def basis_space(obj):
    """The matrix obj's own location, rotation and scale are read in - matrix_world = basis_space(obj) @ matrix_basis:
    its parent's world matrix with the parent inverse, whatever the parent type. None when obj has no parent.

    Read off matrix_world, so take it with the depsgraph current and BEFORE this call writes the basis. A constraint on
    obj is folded in too, which makes it an approximation for a constrained object - aim_object's measurement is what
    says so."""
    if obj.parent is None:
        return None
    return obj.matrix_world @ obj.matrix_basis.inverted_safe()


def write_orientation(obj, euler_xyz, space=None):
    """Turn obj to the orientation the XYZ Euler `euler_xyz` describes - the look-at write - whatever its rotation_mode;
    returns the property written.

    AN ORIENTATION, NOT ANGLES: the look-at maths derives XYZ angles, and the same three numbers in a ZXY object's
    rotation_euler are a different orientation, so every mode but XYZ gets the orientation converted - rotation_euler in
    the object's own order, rotation_quaternion, or rotation_axis_angle. An XYZ object takes the angles exactly, as
    before.

    A WORLD ORIENTATION, when `space` is basis_space(obj): a child's rotation is read in its parent's space, so the
    orientation is taken out of that space first. Without it a child of a turned parent aimed about as far off as the
    parent was turned - measured on 5.2.1 (2026-09-28) under a rig turned 90 deg: aim_object refused its own write 84.26
    deg off, and set_camera lookAt answered ok 89.45 deg off (test_blender_review_defects RD120, RD121)."""
    mode = obj.rotation_mode
    if mode == "XYZ" and space is None:
        obj.rotation_euler = euler_xyz
        return "rotation_euler"
    import mathutils
    q = mathutils.Euler(euler_xyz, "XYZ").to_quaternion()
    if space is not None:
        q = space.to_quaternion().inverted() @ q
    if mode == "QUATERNION":
        obj.rotation_quaternion = q
        return "rotation_quaternion"
    if mode == "AXIS_ANGLE":
        axis, angle = q.to_axis_angle()
        obj.rotation_axis_angle = (angle, axis[0], axis[1], axis[2])
        return "rotation_axis_angle"
    obj.rotation_euler = q.to_euler(mode, obj.rotation_euler)
    return "rotation_euler"


def rotation_mode_note(obj, written):
    """A sentence when a rotation was written for an object that is not in XYZ mode, saying where it went - else None."""
    mode = obj.rotation_mode
    if mode == "XYZ":
        return None
    if mode in EULER_ORDERS:
        return ("this object's rotation_mode is %s, so the rotation was written to rotation_euler, which it reads in %s "
                "order." % (mode, mode))
    return ("this object's rotation_mode is %s, so the rotation was CONVERTED into %s - it ignores rotation_euler, and "
            "writing that would have turned nothing." % (mode, written))


def object_info(obj, with_counts=True, children=None):
    """The pre-image/post-image record. Every number a round-trip assertion
    might need, in both Blender units and Unreal units."""
    info = {
        "name": obj.name,
        "type": obj.type,
        "locationBU": rnd(obj.location),
        # WHERE THE OBJECT ACTUALLY IS, beside where its location property says. locationBU is the basis
        # value, and a parent, a constraint or a SIMULATION moves the object without touching it - a baked
        # rigid body reports its rest locationBU at every frame. Found 2026-09-16 writing
        # test_blender_rebake: nothing in this addon could say where a simulated body ended up, so a
        # caller could not check a bake through the bridge at all. Read off matrix_world, at the current
        # frame.
        "worldLocationBU": rnd(list(obj.matrix_world.to_translation())),
        "rotationEulerRad": rnd(obj.rotation_euler),
        # WHICH ROTATION THE OBJECT USES, and - when it is not an Euler order - the one it does use (2026-09-28, the
        # 2026-09-21 pack-rule review). rotationEulerRad is read whatever the mode, and a QUATERNION or AXIS_ANGLE object
        # ignores it entirely (the measurement is at ops_anim._EULER_PATHS); glTF import makes every object QUATERNION.
        # Measured on 5.2.1: a glTF cube turned 90 degrees about Z by transform_object read rotationEulerRad [0, 0, 0]
        # and, at the origin, isIdentityTransform true.
        "rotationMode": obj.rotation_mode,
        "scale": rnd(obj.scale),
        # WHICH DATA IT DRAWS (2026-10-26): linked duplicates share one mesh datablock, and a level hand-off sends each
        # datablock once - so the name and its user count are read here rather than guessed from object names
        "dataName": obj.data.name if getattr(obj, "data", None) is not None else None,
        "dataUsers": obj.data.users if getattr(obj, "data", None) is not None else None,
        "isIdentityTransform": (
            all(abs(v) < 1e-6 for v in obj.location)
            and rotation_is_identity(obj)
            and all(abs(v - 1.0) < 1e-6 for v in obj.scale)
        ),
        # TEN BOOLEANS, ON EVERY OBJECT RECORD, deliberately - they are cheap and they were
        # unreachable. This is also the pre-image/post-image record, so a write that changes an
        # object's visibility as a side effect now shows up in a round-trip diff instead of being
        # invisible to one. The cost is ten bools against a record that already carries three
        # vectors and the counts.
        "visibility": visibility_readback(obj),
        # THE HIERARCHY, which no reader reported. set_parent writes parent, parent_type, parent_bone and the parent
        # inverse and says what it did in its own reply, and audit_write_only_state listed parent_bone and
        # matrix_parent_inverse as set-and-never-read: a caller who had not just parented an object could not learn its
        # parent, its parent bone, or its children through the bridge (2026-09-21). inverseIsIdentity is the part an
        # exporter feels - Blender's own Ctrl+P stores the parent's inverse there, set_parent leaves identity, and a
        # child's local numbers mean different things under the two.
        "parent": ({"name": obj.parent.name, "type": obj.parent_type,
                    "bone": (obj.parent_bone or None) if obj.parent_type == "BONE" else None,
                    "inverseIsIdentity": _is_identity(obj.matrix_parent_inverse)}
                   if obj.parent is not None else None),
        # From the caller's children_by_parent() map when it lists many objects; one scan of the file otherwise.
        "children": (list(children.get(obj.name, ())) if children is not None
                     else sorted(c.name for c in obj.children)),
    }
    # THE ROTATION A NON-EULER MODE USES, only in that mode - [w, x, y, z], or [angle, x, y, z] in radians.
    if obj.rotation_mode == "QUATERNION":
        info["rotationQuaternion"] = rnd(obj.rotation_quaternion)
    elif obj.rotation_mode == "AXIS_ANGLE":
        info["rotationAxisAngle"] = rnd(obj.rotation_axis_angle)
    # BEFORE the per-type branches, which return early - so a text object's slots were never reported.
    slots = material_slot_names(obj)
    if slots is not None:
        info["materialSlots"] = slots
    # A FORCE FIELD IS NOT A TYPE, which is why this sits up here with the slots rather than in the
    # per-type branches below: obj.field can hang off an EMPTY, a mesh or a light alike, and the
    # branches return early. create_force_field reports all of this in its own response and until
    # 2026-09-15 nothing else did, so an agent could build a wind field and never ask what it built.
    #
    # useMaxDistance SITS BESIDE distanceMax because the number is ignored while the flag is off -
    # create_force_field's own comment says so where it writes the two together, and the pair is one
    # of the five audit_value_without_gate.py now scans for.
    fld = getattr(obj, "field", None)
    if fld is not None and getattr(fld, "type", "NONE") != "NONE":
        info["field"] = {
            "fieldType": fld.type,
            "shape": fld.shape,
            "strength": round(float(fld.strength), 6),
            "flow": round(float(fld.flow), 6),
            "noise": round(float(fld.noise), 6),
            "seed": int(fld.seed),
            "falloffPower": round(float(fld.falloff_power), 6),
            "useMaxDistance": bool(fld.use_max_distance),
            "distanceMax": round(float(fld.distance_max), 6),
        }
    # PER-TYPE DETAIL, because this early-returned for everything non-MESH until 2026-09-03 and this
    # is the addon's most-used read op. A light came back as a name, a type and a transform - none
    # of its energy, colour, cone or shadow - so nothing could diagnose a lighting rig it had not
    # just built. Six of the fourteen ops modules had no read op at all; this gate is why fixing
    # that anywhere else still left the general reader blind.
    #
    # The LIGHT branch calls the SAME light_readback that create_light, set_light and list_lights
    # use, rather than a fourth copy - which is the whole reason that reader moved into this module.
    if obj.type == "LIGHT" and obj.data is not None:
        info["light"] = light_readback(obj, obj.data)
        return info
    if obj.type == "CAMERA" and obj.data is not None:
        info["camera"] = camera_readback(obj, obj.data)
        return info
    if obj.type == "ARMATURE" and obj.data is not None:
        arm = obj.data
        info["armature"] = {
            "boneCount": len(arm.bones),
            "rootBones": [b.name for b in arm.bones if b.parent is None],
            "poseMode": obj.mode,
            "hasPose": obj.pose is not None and len(obj.pose.bones) > 0,
            # WRITTEN BY create_armature AND READ BY NOTHING ELSE until 2026-09-21 (audit_write_only_state): how the
            # bones draw, and whether they draw through the mesh they deform.
            "displayType": arm.display_type,
            "showInFront": bool(obj.show_in_front),
        }
        return info
    if obj.type in ("CURVE", "FONT", "SURFACE") and obj.data is not None:
        cu = obj.data
        # THE SIZE OF A TEXT OR A CURVE, which this row never carried - the second seat (mailbox id 57) had nothing in
        # the addon to confirm a title came out the size it was set to. MEASURED 2026-09-16 on 3.6.23, 4.2.17, 4.4.0,
        # 5.0.1 and 5.2.1: obj.dimensions of a FONT or CURVE object is a cache that only a depsgraph evaluation
        # refreshes. Read straight after the write it was stale on every build (extrude 0.3 and bevel 0.05 left z at
        # 0.0; 3.6 kept even the old width after a longer body), and a fresh curve read [5, 2, 2] - a placeholder box -
        # until evaluated; after view_layer.update() every build agreed ([3.5491, 0.4502, 0.7] for the text, [6.0,
        # 0.5, 0.5] for a 6 m curve bevelled 0.25). The EVALUATED object's own dimensions are no substitute: from 4.2
        # on they read z 0.0 for a text extruded 0.1 and [5, 2, 2] for a 3 m curve bevelled 0.25, after the update.
        bpy.context.view_layer.update()
        info["dimensionsBU"] = rnd(obj.dimensions)
        info["curve"] = {
            "splineCount": len(getattr(cu, "splines", []) or []),
            "dimensions": getattr(cu, "dimensions", None),
            "bevelDepth": round(float(getattr(cu, "bevel_depth", 0.0)), 6),
            "extrude": round(float(getattr(cu, "extrude", 0.0)), 6),
            "body": getattr(cu, "body", None) if obj.type == "FONT" else None,
        }
        # WHAT create_curve WRITES AND ONLY ITS OWN REPLY SAID, until 2026-09-21 (audit_write_only_state): whether a
        # spline closes, its handle types, and use_path - the flag a Follow Path constraint evaluates to nothing
        # without. Per spline, the first 20.
        if obj.type != "FONT":
            info["curve"]["usePath"] = bool(getattr(cu, "use_path", False))
            splines = list(getattr(cu, "splines", []) or [])
            rows = []
            for sp in splines[:20]:
                bezier = sp.type == "BEZIER"
                row = {"type": sp.type, "points": len(sp.bezier_points) if bezier else len(sp.points),
                       "cyclic": bool(sp.use_cyclic_u)}
                if bezier:
                    row["handleTypes"] = sorted({bp.handle_left_type for bp in sp.bezier_points}
                                                | {bp.handle_right_type for bp in sp.bezier_points})
                rows.append(row)
            info["curve"]["splines"] = rows
            if len(splines) > len(rows):
                info["curve"]["splinesTruncated"] = len(splines) - len(rows)
        return info
    if obj.type == "LATTICE" and obj.data is not None:
        # create_lattice's resolution, interpolation and use_outside, which no reader carried (audit_write_only_state,
        # 2026-09-21) - a lattice came back as a name and a transform.
        lat = obj.data
        info["lattice"] = {
            "pointsU": int(lat.points_u), "pointsV": int(lat.points_v), "pointsW": int(lat.points_w),
            "interpolation": {"u": lat.interpolation_type_u, "v": lat.interpolation_type_v,
                              "w": lat.interpolation_type_w},
            "useOutside": bool(lat.use_outside),
        }
        return info
    if obj.type == "EMPTY":
        info["empty"] = {
            "displayType": obj.empty_display_type,
            "displaySize": round(float(obj.empty_display_size), 6),
            # An empty that instances a collection is a whole scene branch, not a null.
            "instanceCollection": (obj.instance_collection.name
                                   if obj.instance_collection else None),
        }
        if obj.empty_display_type == "IMAGE":
            # create_empty's image backdrop (2026-10-05), read back off the object.
            img = obj.data if isinstance(obj.data, bpy.types.Image) else None
            info["empty"]["image"] = {
                "image": img.name if img is not None else None,
                "file": bpy.path.abspath(img.filepath) if img is not None and img.filepath else None,
                "size": list(img.size) if img is not None else None,
                "useAlpha": bool(obj.use_empty_image_alpha), "opacity": round(float(obj.color[3]), 6),
                "depth": obj.empty_image_depth, "side": obj.empty_image_side,
                "showPerspective": bool(obj.show_empty_image_perspective),
                "showOrthographic": bool(obj.show_empty_image_orthographic),
                "offset": [round(float(x), 6) for x in obj.empty_image_offset],
                "selectable": not obj.hide_select}
        return info
    # THE GEOMETRY-NODES TYPES, added 2026-09-07 alongside create_curves and create_point_cloud.
    # Without this they came back as a name, a type and a transform - the same blind-reader failure
    # the comment above describes for lights, one family later, and worse here: these object types
    # hold NOTHING of their own by design, so an empty one and a fully generated one were
    # indistinguishable through this op.
    #
    # BOTH COUNTS, AND THAT IS THE POINT. The base datablock of a working CURVES object is normally
    # empty - the geometry lives on the EVALUATED object, produced by the node group - so reporting
    # only the base count would say "0 points" about an object covered in them. Reporting only the
    # evaluated count would hide whether anything was ever stored. The pair separates "nothing has
    # filled this yet" from "the group generated it", which is the measurement that matters.
    if obj.type in ("CURVES", "POINTCLOUD", "VOLUME"):
        def _count(data):
            n = {}
            for attr in ("points", "curves", "grids"):
                holder = getattr(data, attr, None)
                if holder is not None:
                    try:
                        n[attr] = len(holder)
                    except TypeError:
                        pass
            return n
        gen = {}
        try:
            dg = bpy.context.evaluated_depsgraph_get()
            gen = _count(obj.evaluated_get(dg).data)
        except Exception:                                          # noqa: BLE001
            # NOT FATAL AND NOT SILENT. There is no depsgraph during some restricted contexts, and
            # a read op that raises is worse than one that says it could not look.
            gen = None
        info["generated"] = {
            "stored": _count(obj.data) if obj.data is not None else {},
            "evaluated": gen,
            "nodesModifiers": [m.name for m in obj.modifiers if m.type == "NODES"],
            "note": ("a %s is normally filled by a geometry node group, so `stored` is normally empty and "
                     "`evaluated` is what you see (guides grown by create_curves on a surface are stored). Both are "
                     "reported because an empty one and a generated one look identical otherwise."
                     % obj.type),
        }
        return info
    if obj.type != "MESH":
        return info

    bmin, bmax = local_bounds(obj)
    size = [bmax[i] - bmin[i] for i in range(3)]
    info.update({
        "dimensionsBU": rnd(obj.dimensions),
        "boundsLocalMinBU": rnd(bmin),
        "boundsLocalMaxBU": rnd(bmax),
        "boundsLocalSizeBU": rnd(size),
        "boundsLocalSizeUU": rnd([v * UU_PER_BU for v in size], 4),
        "uvLayers": [uv.name for uv in obj.data.uv_layers],
        "hasCustomSplitNormals": bool(obj.data.has_custom_normals),
        # WHAT set_shading's autoSmoothAngle LEAVES BEHIND, which no reader carried (audit_write_only_state, 2026-09-21).
        # Measured on 3.6.23, 4.2.17 and 5.0.1 the same day: on 3.6 it sets mesh.use_auto_smooth and auto_smooth_angle,
        # which nothing reported; on 4.2 shade_auto_smooth answered CANCELLED headless and set_shading marked 12 sharp
        # edges instead, which nothing counted; on 5.0 it added a Smooth by Angle modifier, which list_modifiers
        # already shows with its Angle input. So: the mesh's sharp edges on every build, and the mesh-level auto smooth
        # where the build still has it (up to 4.0) - null from 4.1, where it is a modifier.
        "sharpEdges": _sharp_edge_count(obj.data),
        "autoSmooth": ({"enabled": bool(obj.data.use_auto_smooth),
                        "angleDeg": round(math.degrees(obj.data.auto_smooth_angle), 4)}
                       if hasattr(obj.data, "use_auto_smooth") else None),
        # WHICH ARMATURE DEFORMS THIS MESH, if any - the pairing nothing else in this addon
        # reported before ops_rig.py existed. An ARMATURE modifier's .object is the actual rig at
        # DEFORM TIME; obj.parent can point at an armature too (the common "parent to armature"
        # workflow), but parenting alone does not deform anything without the modifier, so only the
        # modifier is reported here - that is the field ops_rig.list_bones' object name lines up
        # against, not a guess from the object hierarchy.
        "armatureModifier": next(
            (m.object.name for m in obj.modifiers if m.type == "ARMATURE" and m.object), None),
    })
    # TWO SIZES IN ONE REPLY, AND WHY. dimensionsBU is the EVALUATED size, so it includes every modifier enabled in the
    # viewport - measured off the evaluated vertices when a modifier or shape keys can change it (below), Blender's
    # obj.dimensions otherwise - while boundsLocal* and the vertex counts are read off the BASE mesh. The numbers that
    # follow are obj.dimensions as measured before that change.
    # MEASURED 2026-09-16 on 3.6.23, 4.2.17, 4.4.0, 5.0.1 and 5.2.1 through this function: a 2 BU cube with an Array
    # (count 2) reads dimensionsBU [4, 2, 2] beside boundsLocalSizeBU [2, 2, 2]; a scale of (2, 1, 1) keeps them
    # consistent ([4, 2, 2] against [2, 2, 2] x scale); an Array hidden in the viewport changes nothing; a Subdivision
    # Surface at level 2 reads 1.679 on 3.6-4.4 and 2.0 on 5.0/5.2. A LIQUID fluid domain on 4.2 and later is the stark
    # case - 8 base verts, a 2 BU container in boundsLocal*, and dimensionsBU [1.895, 1.895, 0.418], the simulated
    # water - which the second seat read as "dimensions describe the container" (mailbox id 53).
    # obj.dimensions is also a CACHE (see local_bounds), so a disagreement is re-read after one view_layer.update()
    # before anything is said about it.
    #
    # MEASURED, NOT BLENDER'S BOX, WHERE THE TWO CAN DIFFER. The version matrix at 0ef6f185 put this note on 3.6.23
    # only: its MifSpare carries a Geometry Nodes group that outputs nothing, obj.dimensions read [0, 0, 0] there and
    # the base [2, 2, 2] on 4.2 and later, so from 4.2 on the reply called an object with no geometry 2 BU wide and
    # said nothing. And a Subdivision Surface last in the stack reads its cage from 5.0 on (evaluated_mesh_bounds has
    # the numbers). So an object with a viewport modifier or shape keys gets dimensionsBU off its EVALUATED vertices,
    # and Blender's disagreeing number is kept beside it, named. Two false notes went with it, both MEASURED on all
    # five builds: this compared against obj.scale while Blender scales by the WORLD matrix, so a child of a (2, 1, 1)
    # parent or a Limit Scale constraint read [4, 2, 2] or [6, 2, 2] against [2, 2, 2] and was told "dimensionsBU is
    # the one to distrust" when it was right; and a shape key at 1.0 scaling x1.5 read 3.0 against the 2.0 basis with
    # no modifier and got the same wrong sentence (scratchpad measure_dims_scale_editmode.py, measure_eval_bounds_read.py).
    evaluated = evaluated_mesh_bounds(obj)
    world_scale = [abs(float(v)) for v in obj.matrix_world.to_scale()]    # after that evaluation, when there was one
    if evaluated is not None:
        emin, emax, ecount = evaluated
        blender_dims = rnd(obj.dimensions)          # after evaluated_mesh_bounds evaluated the depsgraph: not stale
        measured = rnd([(emax[i] - emin[i]) * world_scale[i] for i in range(3)])
        info["dimensionsBU"] = measured
        info["evaluatedVerts"] = ecount
        if any(abs(blender_dims[i] - measured[i]) > 1e-4 for i in range(3)):
            last = next((m for m in reversed(obj.modifiers) if m.show_viewport), None)
            if ecount == 0 and bpy.app.version >= (4, 2, 0):
                why = "from Blender 4.2 on, a stack that evaluates to no vertices reads as the BASE mesh's box"
            elif last is not None and last.type == "SUBSURF" and bpy.app.version >= (5, 0, 0):
                why = ("from Blender 5.0 on, a Subdivision Surface last in the stack reads as its CAGE's box, not the "
                       "subdivided surface")
            else:
                why = "it is a cached box and does not match the evaluated vertices"
            info["blenderDimensionsBU"] = blender_dims
            info["dimensionsNote"] = (
                "Blender's own obj.dimensions reads %s here: %s (measured 2026-09-16). dimensionsBU is measured off the "
                "evaluated vertices, with the modifiers' viewport settings." % (blender_dims, why))
        if ecount == 0 or any(abs(measured[i] - size[i] * world_scale[i]) > 1e-4 for i in range(3)):
            sources = [m.name for m in obj.modifiers if m.show_viewport]
            if obj.data.shape_keys is not None:
                sources.append("shape keys")
            info["boundsNote"] = (
                "dimensionsBU is the evaluated geometry's box and includes %s; boundsLocal* and the vertex counts "
                "describe the base mesh, so they differ here by design." % ", ".join(sources)
                + (" The evaluated mesh has NO vertices with the modifiers' viewport settings; instances a Geometry "
                   "Nodes modifier outputs are not measured." if ecount == 0 else ""))
    else:
        def _sizes_disagree(scale):
            return any(abs(float(obj.dimensions[i]) - size[i] * scale[i]) > 1e-4 for i in range(3))
        # obj.scale is current the moment it is written and matrix_world is not until an update, so a disagreement
        # with EITHER earns the update - the obj.scale one is the trigger this block always had, and dropping it would
        # let a stale matrix agree with a stale box. Only a disagreement with the world scale afterwards is reported.
        if _sizes_disagree([abs(float(v)) for v in obj.scale]) or _sizes_disagree(world_scale):
            bpy.context.view_layer.update()
            world_scale = [abs(float(v)) for v in obj.matrix_world.to_scale()]
            info["dimensionsBU"] = rnd(obj.dimensions)
            if _sizes_disagree(world_scale):
                info["boundsNote"] = (
                    "dimensionsBU and boundsLocal* disagree on an object with no modifier enabled in the viewport and "
                    "no shape keys - boundsLocal* is read off the vertices; dimensionsBU is Blender's bounding box and "
                    "is the one to distrust.")
    drawn = drawn_instances(obj)
    if drawn:
        info["drawnInstances"] = drawn
        info["instancesNote"] = instances_note(obj, "dimensionsBU, the bounds and the counts")
    if with_counts:
        info.update(mesh_counts(obj))
    return info


def remove_objects(objs):
    """Remove objects from the file and update the view layer before anything iterates it. Returns the names removed.

    EVERY OBJECT THE ADD-ON REMOVES GOES THROUGH HERE. Measured on 5.0.1 (2026-10-18): after bpy.data.objects.remove,
    view_layer.objects still yields an entry for the removed object - None - until view_layer.update(); after a
    collection unlink it still yields the unlinked object. scene.objects and context.selected_objects are already
    right. Between two calls something updates it, but not between two ops of one batch: delete_object then
    export_scene in one batch failed with "'NoneType' object has no attribute 'type'". One update after the removals
    costs one depsgraph pass, so a caller removing many passes them together."""
    removed = []
    for obj in list(objs):
        if obj is None:
            continue
        try:
            name = obj.name
        except ReferenceError:
            continue
        bpy.data.objects.remove(obj, do_unlink=True)
        removed.append(name)
    bpy.context.view_layer.update()
    return removed


def selection_snapshot():
    view_layer = bpy.context.view_layer
    return ([o.name for o in bpy.context.selected_objects],
            view_layer.objects.active.name if view_layer.objects.active else None)


def selection_restore(snapshot):
    names, active = snapshot
    # `names` is a LIST from the snapshot, so `obj.name in names` was a linear scan inside a loop
    # over every object in the view layer - the same quadratic shape measured in ops_scene's
    # list_objects. It only bites when a lot was selected, which is exactly the export path.
    names = set(names)
    try:
        for obj in bpy.context.view_layer.objects:
            obj.select_set(obj.name in names)
        bpy.context.view_layer.objects.active = bpy.data.objects.get(active) if active else None
    except Exception:
        pass


def select_only(objs):
    """Deselect everything, select `objs`, make the first one active.

    bpy.ops.object.select_all needs a context; in background mode with no
    window it can fail, so do it by hand over the view layer instead.
    """
    view_layer = bpy.context.view_layer
    for obj in view_layer.objects:
        obj.select_set(False)
    # One pass instead of one linear scan per object being selected. This is the export path -
    # export_mesh selects everything it is about to write - so `objs` is as large as the scene on
    # exactly the runs where it matters.
    vl_names = set(view_layer.objects.keys())
    first = None
    for obj in objs:
        if obj.name not in vl_names:
            raise MifOpError("object '%s' is not in the active view layer, so it cannot "
                             "be selected for export (is it in an excluded collection?)"
                             % obj.name)
        obj.select_set(True)
        first = first or obj
    view_layer.objects.active = first


# ---------------------------------------------------------------------------
# Bounding a listing, and excluding from it - ONE pair, because they cannot be separated
# ---------------------------------------------------------------------------

BOUND_KEYS = ("limit", "nameExcludes")


def exclude_rows(rows, params, name_of=lambda r: r.get("name", "")):
    """Drop rows whose name contains any of `nameExcludes` (a string or a list), case-insensitively.

    WHY THIS SHIPS WITH `limit` AND NOT AFTER IT. Every filter this add-on had was an INCLUDE, and with no cap that
    was only ergonomics: a caller got everything and filtered their own end, and could not miss a row. A cap changes
    that - the rows you did not want fill it and hide the ones you did, which is the saturation the UE side already
    has and the spec item says not to recreate here. So an exclusion lands with the first bound, not in a later pass.

    The add-on's own scratch convention is a Mif name prefix, so nameExcludes:["Mif"] is the common case.
    """
    raw = params.get("nameExcludes")
    if raw is None:
        return rows
    terms = [raw] if isinstance(raw, str) else list(raw)
    bad = [t for t in terms if not isinstance(t, str)]
    if bad:
        raise MifOpError("'nameExcludes' takes a string or a list of strings, got %r in the list." % (bad[0],))
    low = [t.lower() for t in terms if t]
    if not low:
        return rows
    return [r for r in rows if not any(t in str(name_of(r)).lower() for t in low)]


def limit_rows(rows, params, what, excludes=True):
    """(rows, extra fields) for a listing that can grow past what a caller wants to read.

    THE DECIDED SHAPE, taken from face_info (ops_query.py) rather than invented beside it: `limit` is optional and
    unlimited by default, so no caller that never passed it sees any change; the caller is told the number that
    MATCHED whatever the limit, because a caller deciding whether to page needs the true size; and `truncated` is
    reported whenever rows were dropped, because a caller handed 200 of 40 000 rows and not told has a sample they
    will treat as the set.

    `excludes` IS WHETHER THE CALLING OP ACCEPTS nameExcludes, because the note names it. It defaults True because
    every op that used this helper before 2026-09-20 does accept it - measured, not assumed: of the 27 ops that take
    a `limit`, 20 take nameExcludes as well, and the seven that do not are face_info, select_faces and vertex_info
    (which write their own truncation notes naming their own filters) plus the four listings bounded on that day.
    Pass excludes=False from an op that does not take it. A truncation note advising a parameter the op would refuse
    is the same defect class as a hint naming an endpoint that does not exist, and it is worth one keyword to keep
    the sentence true.
    """
    limit = params.get("limit")
    if limit is None:
        return rows, {}
    # finite_int, not int(). Measured on 5.0.1, 2026-09-21, through list_objects and list_materials: limit
    # Infinity was a RAW OverflowError (int() raises it and the except below did not name it), and 2.5 returned
    # 2 rows. Every one of the ops that bound a listing through this helper had both.
    try:
        limit = finite_int(limit, "limit")
    except MifOpError:
        raise MifOpError("'limit' must be a whole number of %s, got %r." % (what, limit))
    if limit < 0:
        raise MifOpError("'limit' cannot be negative, got %d." % limit)
    kept = rows[:limit]
    extra = {"truncated": len(kept) < len(rows), "returned": len(kept)}
    if extra["truncated"]:
        extra["truncationNote"] = (
            "%d %s matched and %d were returned - raise limit, narrow the request%s."
            % (len(rows), what, len(kept),
               ", or drop rows you do not want with nameExcludes" if excludes else ""))
    return kept, extra


# ---------------------------------------------------------------------------
# A frame change that would step a live simulation
# ---------------------------------------------------------------------------

# FOUND 2026-10-04 in a pose check on a working copy of a garment file: seven frame_set calls, and a leftover CLOTH
# modifier on an object hidden from the RENDER only stepped against the body's COLLISION; Blender asked for 58.7 GB "in
# collision array" and the editor died. A frame change evaluates every simulation the view layer's depsgraph holds, so
# an op that moves the frame to READ something can run a solver the caller never meant to start.
#
# WHAT THE DEPSGRAPH EVALUATES, measured 2026-10-05 on 4.2.23, 4.4.0, 5.0.1 and 5.2.2: an unbaked cloth stepped on an
# object hidden with hide_set (the eye) and on one with hide_render, and did not on one with hide_viewport (the screen)
# or with the modifier's show_viewport off. depsgraph.objects leaves the eye-hidden object out - it lists what DRAWS -
# so the set is read from depsgraph.ids. A new fluid domain is cache_type REPLAY, which runs the solver on a frame
# change (three of the four builds aborted in mantaflow on exit after three frames of it); MODULAR and ALL only read a
# bake. A Geometry Nodes simulation zone steps too, but no RNA on any of the four builds says whether its bake exists
# (NodesModifierBake has no baked flag), so those are listed and not refused.
def _node_group_has_simulation(group, seen=None):
    seen = set() if seen is None else seen
    if group is None or group.name_full in seen:
        return False
    seen.add(group.name_full)
    for node in group.nodes:
        if node.bl_idname == "GeometryNodeSimulationInput":
            return True
        if getattr(node, "node_tree", None) is not None and _node_group_has_simulation(node.node_tree, seen):
            return True
    return False


# WHAT A MANTAFLOW DOMAIN HAS SIMULATED, READ OFF ITS CACHE FOLDER. The second seat (mailbox id 151) found bake_physics
# listing only a liquid domain's "Liquid" particle cache - isBaked false, framesStored null - whether or not the fluid had
# simulated. MEASURED 2026-09-16 on 3.6.23, 4.2.17 and 5.2.1 (scratchpad measure_fluid_bake_all_readout.py): with the
# cache left at add_fluid's REPLAY, ptcache.bake_all over frames 1-40 took 2.4 s (0.8 s on 3.6) and left config/data (and
# mesh, where the domain makes one) frames 1-40 in the domain's cache directory; set to MODULAR it took 0.06 s and wrote
# nothing, and the domain evaluated at rest. The domain's is_cache_baking_* and cache_frame_pause_* properties read
# false/0 in every case, so the files are the evidence. Frame numbers come from the _NNNN before the extension - and a
# frame before 1, which a pre-rolled domain bakes, is written with its minus sign inside that field.
FLUID_FRAME = re.compile(r"_([0-9-]{2,})\.")


def fluid_frame_numbers(names):
    """Frame numbers from Mantaflow cache file names ("fluid_data_0012.vdb"), those before frame 1 included."""
    out = set()
    for name in names:
        m = FLUID_FRAME.search(name)
        if not m:
            continue
        text = m.group(1)
        digits = text.replace("-", "")
        if not digits.isdigit() or text.count("-") > 1:
            continue
        out.add(-int(digits) if "-" in text else int(digits))
    return out


def _replay_frames_held_through(ds):
    """The last frame f such that every frame from a fluid domain's cache start to f has data in its cache folder, or None
    when the folder holds not even the first. Frame numbers are the _NNNN before the extension, a minus sign inside it for
    a frame before 1 (fluid_frame_numbers)."""
    root = bpy.path.abspath(ds.cache_directory) if ds.cache_directory else ""
    path = os.path.join(root, "data") if root else ""
    if not path or not os.path.isdir(path):
        return None
    frames = fluid_frame_numbers(os.listdir(path))
    f = int(ds.cache_frame_start)
    while f in frames:
        f += 1
    return f - 1 if f > int(ds.cache_frame_start) else None


def _object_sim_rows(obj, add, colliders, gn_rows):
    for m in obj.modifiers:
        if not m.show_viewport:
            continue
        if m.type in ("CLOTH", "SOFT_BODY"):
            pc = m.point_cache
            if not pc.is_baked:
                add(obj.name, m.name, m.type, pc.frame_start,
                    "not baked: each frame past %d runs the solver" % pc.frame_start)
        elif m.type == "COLLISION":
            colliders.append(obj.name)
        elif m.type == "PARTICLE_SYSTEM":
            ps = m.particle_system
            st = ps.settings
            steps = ((st.type == "EMITTER" and st.physics_type != "NO")
                     or (st.type == "HAIR" and bool(getattr(ps, "use_hair_dynamics", False))))
            if steps and not ps.point_cache.is_baked:
                add(obj.name, m.name, "PARTICLE_SYSTEM", ps.point_cache.frame_start,
                    "%s particles, not baked: each frame past %d steps them"
                    % (st.type.lower(), ps.point_cache.frame_start))
        elif m.type == "FLUID" and m.fluid_type == "DOMAIN":
            ds = m.domain_settings
            if ds.cache_type == "REPLAY":
                # A REPLAY DOMAIN READS THE FRAMES ITS FOLDER HOLDS and simulates only the ones it lacks (the rule
                # ops_physics._bake_fluid_domains steps by). Counted as live whatever it held, a domain bake_physics had
                # just finished refused every later frame change - the refusal's own remedy, "bake them", could not
                # clear it (found 2026-10-10: a baked 12-frame liquid read on frame 1 only). Held through its cache end,
                # it steps nothing; held part way, it steps from the last frame held.
                held = _replay_frames_held_through(ds)
                if held is None or held < int(ds.cache_frame_end):
                    start = int(ds.cache_frame_start) if held is None else held
                    add(obj.name, m.name, "FLUID", start,
                        "a domain with cache type REPLAY runs the fluid solver on every frame change past the frames "
                        "its cache folder holds (%s; MODULAR or ALL only read a bake)"
                        % ("none held" if held is None else "held %d-%d" % (int(ds.cache_frame_start), held)))
        elif m.type == "DYNAMIC_PAINT" and getattr(m, "ui_type", "") == "CANVAS" and m.canvas_settings:
            for s in m.canvas_settings.canvas_surfaces:
                if s.surface_format != "IMAGE" and not s.point_cache.is_baked:
                    add(obj.name, m.name, "DYNAMIC_PAINT", s.point_cache.frame_start,
                        "canvas surface '%s', not baked: each frame past %d paints"
                        % (s.name, s.point_cache.frame_start))
        elif m.type == "NODES" and _node_group_has_simulation(getattr(m, "node_group", None)):
            gn_rows.append({"object": obj.name, "modifier": m.name, "type": "GEOMETRY_NODES_SIMULATION"})


def live_simulations(frames, scene=None):
    """What a frame change to any of `frames` would step, and the colliders it would step against.

    Returns (rows, colliders, gn_rows). A row is a simulation the depsgraph evaluates that has no bake to read and whose
    cache starts before the latest frame asked: {object, modifier, type, cacheFrameStart, why}. A frame at or before the
    cache's first frame resets it instead of stepping, so it is not counted."""
    sc = scene or bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    live_ids = {i.name_full for i in dg.ids if isinstance(i, bpy.types.Object)}
    top = max(int(f) for f in frames) if frames else None
    rows, colliders, gn_rows = [], [], []

    def add(obj_name, mod, kind, start, why):
        if top is not None and top > int(start):
            rows.append({"object": obj_name, "modifier": mod, "type": kind, "cacheFrameStart": int(start), "why": why})

    for obj in sc.objects:
        if obj.name_full in live_ids:
            _object_sim_rows(obj, add, colliders, gn_rows)
    rbw = getattr(sc, "rigidbody_world", None)
    if (rbw is not None and rbw.enabled and rbw.collection is not None
            and any(o.name_full in live_ids for o in rbw.collection.objects) and not rbw.point_cache.is_baked):
        add(None, "RigidBodyWorld", "RIGID_BODY", rbw.point_cache.frame_start,
            "the rigid body world, not baked: each frame past %d steps its %d bodies"
            % (rbw.point_cache.frame_start, len(rbw.collection.objects)))
    return rows, colliders, gn_rows


def simulation_guard(params, frames, what, tail="NOTHING was changed, and the frame did not move."):
    """Refuse a frame change that would step a live simulation unless allowSimulation is true. `frames` is every frame
    the op will visit (the frame it puts back included). Returns the reply block to merge ({} when nothing steps).

    NOT GUARDED, on purpose: bake_physics and step_simulation, whose job is to step; render_animation, which renders
    in a child process (a crash there does not take this Blender down, and playing the simulation is the render)."""
    allow = take_bool(params, "allowSimulation", default=False)
    frames = [int(f) for f in frames or [] if int(f) != bpy.context.scene.frame_current]
    if not frames:
        return {}
    rows, colliders, gn_rows = live_simulations(frames)
    if rows and not allow:
        raise MifOpError(
            "%s would move the scene to frame %d, and a frame change steps %d live simulation(s): %s%s. A step is not "
            "cheap - a cloth stepping against a collider has asked for 58.7 GB and taken the editor down. Bake them "
            "(bake_physics), take them out of the depsgraph (hide_viewport on the object, or the modifier's "
            "show_viewport off), or pass allowSimulation=true to step them on purpose. %s"
            % (what, max(int(f) for f in frames), len(rows),
               "; ".join("%s%s (%s): %s" % (r["object"] + " / " if r["object"] else "", r["modifier"], r["type"],
                                            r["why"]) for r in rows),
               (" - colliders they would step against: %s" % ", ".join(colliders)) if colliders else "", tail))
    out = {}
    if rows:
        out["simulationsStepped"] = rows
        if colliders:
            out["colliders"] = colliders
    if gn_rows:
        out["geometryNodesSimulations"] = gn_rows
        out["geometryNodesSimulationNote"] = (
            "a Geometry Nodes simulation zone steps on a frame change unless it is baked, and Blender exposes no baked "
            "flag to read, so these are listed and not refused.")
    return out


# ---------------------------------------------------------------------------
# An editor area to run an editor's operator in
# ---------------------------------------------------------------------------

def _first_window():
    wm = bpy.context.window_manager
    return bpy.context.window or (wm.windows[0] if wm is not None and wm.windows else None)


class editor_override:
    """`with editor_override("SEQUENCE_EDITOR", scene) as ctx:` - borrow the first area of the window's screen, turn it
    into that editor, and run operators inside it; the area's type (and whatever the caller keep()s) is put back after.

    WHY: the sequencer's meta_make and strip_modifier_move and the clip editor's track_markers, detect_features and
    solve_camera poll for their own editor, and nothing else reaches them. MEASURED 2026-10-06 on 4.2.23, 4.4.0, 5.0.1
    and 5.2.2 as the PyPI bpy module: the window manager has one window whose screen holds four areas, and an area
    switched to SEQUENCE_EDITOR runs sequencer.meta_make under temp_override(window, area, region) on every build.

    ON 5.0 AND LATER the sequencer acts on the WORKSPACE's sequencer_scene, not the window's scene, so while the
    override is held that is set to `scene` and put back after. `ctx` is the dict passed to temp_override, plus `space`.
    """

    def __init__(self, area_type, scene=None):
        self.area_type = area_type
        self.scene = scene or bpy.context.scene
        self._restore = []
        self._cm = None

    def __enter__(self):
        win = _first_window()
        if win is None or win.screen is None or not win.screen.areas:
            raise MifOpError("this Blender has no window with an editor area, and the %s operator this needs runs only "
                             "inside one." % self.area_type)
        area = win.screen.areas[0]
        self._restore.append((area, "type", area.type))
        area.type = self.area_type
        ws = win.workspace
        if ws is not None and "sequencer_scene" in ws.bl_rna.properties and ws.sequencer_scene != self.scene:
            self._restore.append((ws, "sequencer_scene", ws.sequencer_scene))
            ws.sequencer_scene = self.scene
        region = next((r for r in area.regions if r.type == "WINDOW"), area.regions[0])
        ctx = {"window": win, "screen": win.screen, "area": area, "region": region, "scene": self.scene}
        self._cm = bpy.context.temp_override(**ctx)
        self._cm.__enter__()
        ctx["space"] = area.spaces.active
        return ctx

    def keep(self, holder, attr):
        """Remember holder.attr now, to be put back when the override ends."""
        self._restore.append((holder, attr, getattr(holder, attr)))

    def __exit__(self, *exc):
        if self._cm is not None:
            self._cm.__exit__(*exc)
        for holder, attr, value in reversed(self._restore):
            try:
                setattr(holder, attr, value)
            except (AttributeError, TypeError, ReferenceError, RuntimeError):
                pass
        return False


class sequencer_scene_guard:
    """Hold the workspace's sequencer_scene on `scene` (5.0 and later) around a write that reads it.

    FOUND 2026-10-06 on 5.0.1 as the PyPI bpy module: Strip.modifiers.new() on a COLOR strip SEGFAULTS Blender when the
    workspace has no sequencer_scene (exit 139, the first modifier type tried), and runs when it is set to the strip's
    scene. 5.2.2 makes the modifier either way; 4.2 and 4.4 have no such property. So every modifier write is held
    inside this, on every build that has the property.
    """

    def __init__(self, scene):
        self.scene = scene
        self._ws = None
        self._old = None

    def __enter__(self):
        win = _first_window()
        ws = getattr(win, "workspace", None) if win is not None else getattr(bpy.context, "workspace", None)
        if ws is not None and "sequencer_scene" in ws.bl_rna.properties and ws.sequencer_scene != self.scene:
            self._ws, self._old = ws, ws.sequencer_scene
            ws.sequencer_scene = self.scene
        return self

    def __exit__(self, *exc):
        if self._ws is not None:
            try:
                self._ws.sequencer_scene = self._old
            except (AttributeError, TypeError, ReferenceError):
                pass
        return False
