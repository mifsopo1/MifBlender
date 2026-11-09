"""THE BULK PREVIEW, Blender's half (docs/15_SAFETY_GATE_AND_JOURNAL.md section 5, "The Blender half").

The Unreal plugin's handshake, with the same field names, for the add-on's bulk writes:

  1. PREVIEW - the op with dryRun:true. Its own plan resolves every target, reads every value before, predicts every value
     after, runs every refusal, and writes nothing. It answers with a previewId and the full change list.
  2. COMMIT - the same op with {previewId} and nothing else. It plans the stored request again; any difference from the
     preview refuses (drift[]) and writes nothing. Otherwise it runs the op's ordinary handler on the plan's commit
     request, reads every change back, and on any mismatch (or a handler that fails) runs the recorded inverse, plans a
     third time and says whether the file is back to exactly what the preview saw (rollbackComplete).
  3. A plain call (neither key) runs exactly as before.

THE STORE IS THE CHANGE CAP'S (cap._PREVIEWS, bpv-<pid>-<n> ids, 64 live, single use, bound to its op), so a preview the cap
forces and one a caller asks for are one kind of thing, committed one way.

EACH PLAN SITS BESIDE ITS OP and reuses that op's own code: an op module offers PREVIEW = {op: {plan, read, capture,
restore, noRollback?}}:
  plan(request)            -> {changes[], unchanged[], context[], response{}, commit{}} - writes nothing; a MifOpError is
                              the op's own refusal. A change is {kind, op (create|modify|delete), target, field?, label?,
                              before, after, _ref?}; keys starting with "_" are the plan's own and never leave the add-on.
  read(change, kept)       -> the value now, in `after`'s shape (None: absent)
  capture(plan)            -> what restore needs, taken just before the body runs
  restore(kept, plan)      -> puts it back; None for a change nothing brings back (noRollback says why)
  release(kept)            -> optional: frees what capture made (a mesh copy), whatever the outcome

A MESH OP PLANS ON A SCRATCH COPY (mesh_spec below, 2026-10-09). Most of the add-on's mesh writes - a bevel, a decimate,
an unwrap, a modifier applied, a garment pushed out of a body - compute their result in the same pass that writes it,
and several only through a Blender operator, which writes. Their plan runs the op's ORDINARY handler on a copy of the
object and its mesh linked into the scene for the call, reads the copy, and removes it and every data-block the run
made - the Unreal side's rule for a property ("imported into a scratch copy of the property, never the object"). The
copy against the object gives the changes: each moved vertex, or one geometry change when the topology changes, edge
marks, weights, vertex groups, UV layers, shape keys and the modifier stack. The inverse is a copy of the mesh taken
just before the commit, remapped back in, with the vertex groups and the modifier stack put back.

A BATCH PREVIEW (batch_spec) plans each op with its own plan, refuses an op with none (refused[], no previewId), an op
no inverse takes back, and two ops that change one target - batch runs its ops in order, so the later op's preview of
that target could never read back as shown. Its commit runs each op's handler and rolls back every op on a mismatch.

WHAT IS NOT BUILT HERE (said, so nobody reads its absence as a pass): the Unreal side's tripwire, which watches every
object modified while a plan runs, has no Blender equivalent - the suites prove each plan writes nothing instead; and
alsoChanged[] is not measured, so a commit does not carry it.
"""
import json
import os
import time

from . import cap
from .ops_common import MifOpError, finite_int, take_bool

INLINE = 200
DEFAULT_TTL = 600
MAX_TTL = 3600
PREVIEW_KEYS = ("dryRun", "previewId", "previewTtlSeconds")
# A suite hook, the twin of Unreal's mif.PreviewForceMismatchAt: the next commit treats change N's read-back as a
# mismatch, once. Set through run_python by test_blender_bulk_preview; nothing else touches it.
FORCE_MISMATCH_AT = None
_TOLERANCE = 1e-4
_JOURNAL = {}


def _modules():
    from . import ops_create, ops_file, ops_garment, ops_mesh, ops_rig, ops_rigify, ops_topology
    return (ops_create, ops_file, ops_garment, ops_mesh, ops_rig, ops_rigify, ops_topology)


def specs():
    """{op: its plan functions} - every op that takes the handshake."""
    out = {}
    for mod in _modules():
        out.update(getattr(mod, "PREVIEW", {}))
    return out


def spec(op):
    return specs().get(op)


# ------------------------------------------------------------------------------------------------------------ the keys
def mode(params):
    """None (a plain call), 'preview' or 'commit' - read strictly, before anything runs."""
    has_id = "previewId" in params
    try:
        dry = take_bool(params, "dryRun", default=False)
    except MifOpError as exc:
        raise MifOpError("%s Nothing was written." % exc)
    if dry and has_id:
        raise MifOpError("dryRun and previewId together ask for two different things: dryRun previews, previewId commits "
                         "a preview. Send one. Nothing was written.")
    if "previewTtlSeconds" in params and not dry:
        raise MifOpError("previewTtlSeconds belongs with a preview (dryRun:true) - it sets how long the preview can be "
                         "committed. Nothing was written.")
    if has_id:
        return "commit"
    return "preview" if dry else None


def _ttl(params):
    raw = params.get("previewTtlSeconds", DEFAULT_TTL)
    why = ("previewTtlSeconds must be a whole number of seconds from 1 to %d (default %d). Nothing was written."
           % (MAX_TTL, DEFAULT_TTL))
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise MifOpError(why)
    # finite_int, not int(): int(NaN) raised a ValueError and int(Infinity) an OverflowError out of the preview, where
    # every other refusal is a sentence.
    try:
        ttl = finite_int(raw, "previewTtlSeconds")
    except MifOpError:
        raise MifOpError(why) from None
    if not 1 <= ttl <= MAX_TTL:
        raise MifOpError(why)
    return ttl


def request_of(params):
    """The request a preview stores and a commit plans again: dryRun and the preview's own keys taken off."""
    return {k: v for k, v in params.items() if k not in PREVIEW_KEYS}


# ------------------------------------------------------------------------------------------------------------ lines
def _public(change, i=None):
    row = {k: v for k, v in change.items() if not k.startswith("_")}
    if i is not None:
        row["i"] = i
    row.setdefault("before", None)
    row.setdefault("after", None)
    return row


def _key(c):
    return "%s|%s|%s" % (c.get("kind"), c.get("target"), c.get("field") or "")


def _line(c):
    return json.dumps([c.get("op"), c.get("kind"), c.get("target"), c.get("field"), c.get("before"), c.get("after")],
                      sort_keys=True, default=str)


def lines(plan):
    return [_line(c) for c in plan["changes"]] + ["context|" + str(x) for x in plan.get("context") or []]


def _summary(changes, unchanged=0, refused=0):
    s = {"create": 0, "modify": 0, "delete": 0, "unchanged": unchanged, "refused": refused}
    for c in changes:
        s[c["op"] if c.get("op") in ("create", "delete") else "modify"] += 1
    return s


def _clip(value, cap_chars=300):
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= cap_chars else text[:cap_chars] + "... (%d characters)" % len(text)


def diff(previewed, fresh):
    """Line by line, what differs between the preview and a new plan of the same request - Unreal's DiffPlans."""
    drift = []
    if list(previewed.get("context") or []) != list(fresh.get("context") or []):
        drift.append({"change": "what the request resolves to is different",
                      "previewed": list(previewed.get("context") or []), "now": list(fresh.get("context") or [])})
    was = {_key(c): c for c in previewed["changes"]}
    now = {_key(c): c for c in fresh["changes"]}
    for i, c in enumerate(previewed["changes"]):
        n = now.get(_key(c))
        row = {"i": i, "target": c.get("target")}
        if c.get("field"):
            row["field"] = c["field"]
        if n is None:
            row["change"] = ("previewed, but the commit would no longer write it (it no longer resolves, no longer "
                             "matches the selector, or already holds the value)")
            drift.append(row)
        elif _line(n) != _line(c):
            row["previewed"] = {"before": c.get("before"), "after": c.get("after")}
            row["now"] = {"before": n.get("before"), "after": n.get("after")}
            drift.append(row)
    for c in fresh["changes"]:
        if _key(c) not in was:
            row = {"target": c.get("target"), "change": "the commit would now write this, and the preview did not show it"}
            if c.get("field"):
                row["field"] = c["field"]
            drift.append(row)
    if not drift and lines(previewed) != lines(fresh):
        drift.append({"change": "the plan's text differs in a way no single change names"})
    return drift


def _near(a, b):
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= _TOLERANCE
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_near(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_near(a[k], b[k]) for k in a)
    return a == b


# ------------------------------------------------------------------------------------------------------------ journal
def note_journal(mode_name, op, pid, outcome, changes):
    """The bulk fields of this call's end record (journal.call_end takes them): Unreal's NoteJournal."""
    targets = []
    for c in changes:
        if len(targets) >= 100:
            break
        if c.get("target") not in targets:
            targets.append(c.get("target"))
    _JOURNAL.clear()
    _JOURNAL.update({"mode": mode_name, "endpoint": op, "outcome": outcome, "changes": len(changes),
                     "targets": targets})
    if pid:
        _JOURNAL["previewId"] = pid


def take_journal():
    """The bulk record noted during the call that is ending, once; None when the call was not a preview or a commit."""
    if not _JOURNAL:
        return None
    out = dict(_JOURNAL)
    _JOURNAL.clear()
    return out


# ------------------------------------------------------------------------------------------------------------ preview
def preview_payload(op, params, plan_spec=None):
    """Plan `params` and store it: the preview fields (previewId among them when there is something to commit).
    Raises MifOpError, writing nothing, when the op refuses the request. Also what the change cap answers over the cap."""
    plan_spec = plan_spec or spec(op)
    ttl = _ttl(params)
    request = request_of(params)
    plan = plan_spec["plan"](request)
    changes = plan["changes"]
    unchanged = plan.get("unchanged") or []
    refused = list(plan.get("refused") or [])
    out = dict(plan.get("response") or {})            # the op's own dry-run answer: its fields stay
    op_note = out.pop("note", None)
    out.update(dryRun=True, nothingWritten=True, wouldCommit=bool(changes) and not refused,
               summary=_summary(changes, len(unchanged), len(refused)),
               changes=[_public(c, i) for i, c in enumerate(changes[:INLINE])],
               changesTotal=len(changes), changesShown=min(len(changes), INLINE),
               unchanged=[_public(c, i) for i, c in enumerate(unchanged[:INLINE])], unchangedTotal=len(unchanged),
               context=list(plan.get("context") or []), refused=refused)
    if plan_spec.get("noRollback"):
        out["rollbackPossible"] = False
        out["noRollback"] = plan_spec["noRollback"]
    if refused:
        # A BATCH WITH AN OP THAT CANNOT BE PLANNED (Unreal's rule): the rest is shown, and no id is issued - a commit
        # of part of a batch is not what was asked for.
        out["note"] = _with(op_note, "nothing was written, and no previewId is issued: %d op(s) cannot be previewed, "
                                     "named in refused[] with the reason. Take them out, or send them as a plain batch."
                            % len(refused))
        note_journal("preview", op, None, "refused", changes)
        return out
    if not changes:
        out["note"] = _with(op_note, "nothing was written, and there is nothing to commit: every target already holds its "
                                     "value after. No previewId is issued.")
        note_journal("preview", op, None, "nothing-to-commit", changes)
        return out
    evicted = cap._prune()
    pid = cap.new_id()
    now = time.monotonic()
    cap._PREVIEWS[pid] = {"op": op, "params": request, "plan": plan, "lines": lines(plan), "created": now,
                          "createdUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "expires": now + ttl,
                          "ttl": ttl, "spent": False}
    out.update(previewId=pid, expiresInSeconds=ttl,
               expiresAtUtc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + ttl)))
    if evicted:
        out["evicted"] = evicted[-1]
        out["warning"] = ("%d previews were live, so the oldest, %s, was dropped to make room for this one."
                          % (cap._MAX_PREVIEWS, evicted[-1]))
    out["note"] = _with(op_note, "nothing was written. Commit with {\"previewId\":\"%s\"} on %s, and nothing else, "
                                 "within %d s. The commit plans this request again and refuses if anything listed here "
                                 "has changed; list_previews pages the changes when there are more than %d."
                        % (pid, op, ttl, INLINE))
    note_journal("preview", op, pid, "previewed", changes)
    return out


def _with(op_note, note):
    """The op's own dry-run note first - callers read it - then the handshake's."""
    return "%s %s" % (op_note, note) if op_note else note


def answer_preview(op, params, plan_spec):
    out = preview_payload(op, params, plan_spec)
    out["ok"] = not out.get("refused")
    if out.get("refused"):
        out["error"] = out["note"]
    return out


# ------------------------------------------------------------------------------------------------------------ commit
def _refusal(pid, op, error, **extra):
    out = {"ok": False, "committed": False, "nothingWritten": True, "previewId": pid, "endpoint": op, "error": error}
    out.update(extra)
    return out


def answer_commit(op, params, fn, plan_spec):
    """{previewId} alone: plan again, compare, apply, read back, roll back on a mismatch. Returns the response.
    What capture made (a mesh copy) is released whatever the outcome."""
    holder = {}
    try:
        return _answer_commit(op, params, fn, plan_spec, holder)
    finally:
        if plan_spec.get("release") is not None and "kept" in holder:
            try:
                plan_spec["release"](holder["kept"])
            except Exception:                                      # noqa: BLE001 - a leftover copy, never the answer
                pass


def _answer_commit(op, params, fn, plan_spec, holder):
    pid, rec = cap.claim(op, params)          # refuses (raises) without spending when the id cannot commit here
    previewed = rec["plan"]
    request = rec["params"]

    def outcome(text):
        rec["outcome"] = text

    # ---- 1. plan again
    try:
        fresh = plan_spec["plan"](request)
    except MifOpError as exc:
        outcome("drift")
        note_journal("commit", op, pid, "drift", previewed["changes"])
        return _refusal(pid, op, "the file changed since preview %s: the same request no longer plans (%s). Nothing was "
                                 "written; the preview is spent. Preview again." % (pid, exc),
                        drift=[{"change": "the request no longer plans: %s" % exc}])
    if lines(fresh) != rec["lines"]:
        drift = diff(previewed, fresh)
        outcome("drift")
        note_journal("commit", op, pid, "drift", previewed["changes"])
        return _refusal(pid, op, "the file changed since preview %s: %d difference(s) between what was previewed and "
                                 "what the commit would write now, named in drift[]. Nothing was written; the preview "
                                 "is spent. Preview again." % (pid, len(drift)), drift=drift[:INLINE])

    # ---- 2. capture what undoing it needs, then run the ordinary handler on the plan's commit request
    restore = plan_spec.get("restore")
    kept = plan_spec["capture"](fresh) if plan_spec.get("capture") else None
    holder["kept"] = kept
    failed = None
    body = None
    try:
        body = fn(dict(fresh["commit"]))
    except MifOpError as exc:
        failed = "the handler refused: %s" % exc
    except Exception as exc:                                       # noqa: BLE001 - rolled back below, then reported
        failed = "the handler raised %s: %s" % (type(exc).__name__, exc)

    # ---- 3. read every change back
    global FORCE_MISMATCH_AT
    force, FORCE_MISMATCH_AT = FORCE_MISMATCH_AT, None
    rows, mismatches = [], []
    for i, c in enumerate(fresh["changes"]):
        row = _public(c, i)
        try:
            now = plan_spec["read"](c, kept)
            match = _near(now, c.get("after"))
            why = None if match else "previewed %s, read back %s" % (_clip(c.get("after")), _clip(now))
            row["readBack"] = now
        except Exception as exc:                                   # noqa: BLE001 - a read that fails is a mismatch
            match, why = False, "the read-back failed: %s: %s" % (type(exc).__name__, exc)
        if force == i and match:
            match, why = False, "forced by ops_preview.FORCE_MISMATCH_AT (a suite proving the rollback)"
        row["matches"] = match
        if not match:
            row["mismatch"] = why
            mismatches.append(row)
        if len(rows) < INLINE:
            rows.append(row)

    out = {"previewId": pid, "endpoint": op, "result": body, "changes": rows, "changesTotal": len(fresh["changes"]),
           "summary": _summary(fresh["changes"], len(fresh.get("unchanged") or []))}
    if failed is None and not mismatches:
        outcome("committed")
        note_journal("commit", op, pid, "committed", fresh["changes"])
        out.update(ok=True, committed=True, matchesPreview=True,
                   note="committed exactly what preview %s showed: %d change(s), every one read back and matched. %s"
                        % (pid, len(fresh["changes"]),
                           "One undo step takes it back." if restore else
                           "No inverse of this commit takes them back: %s." % plan_spec.get("noRollback")))
        return out

    out.update(ok=False, committed=False, matchesPreview=False)
    if mismatches:
        out["mismatches"] = mismatches[:INLINE]
    cause = failed or "%d change(s) read back different from the preview (mismatches[])" % len(mismatches)
    if restore is None:
        landed = len(fresh["changes"]) - min(len(fresh["changes"]), len(mismatches))
        outcome("partial")
        note_journal("commit", op, pid, "partial", fresh["changes"])
        out.update(rolledBack=False, rollbackPossible=False, partial=True, landed=landed,
                   error="the commit of %s did not write exactly what was previewed - %s. No inverse takes these "
                         "changes back (%s), so NOTHING WAS ROLLED BACK: %d of %d landed as previewed, and changes[] "
                         "marks each one. Preview again to see what is left."
                         % (pid, cause, plan_spec.get("noRollback"), landed, len(fresh["changes"])))
        return out

    # ---- 4. roll back, and prove it
    problems = []
    try:
        restore(kept, fresh)
    except Exception as exc:                                       # noqa: BLE001 - reported, never a clean rollback
        problems.append("the inverse raised %s: %s" % (type(exc).__name__, exc))
    try:
        third = plan_spec["plan"](request)
        back = lines(third) == rec["lines"]
        if not back:
            out["rollbackDifferences"] = diff(previewed, third)[:INLINE]
    except MifOpError as exc:
        back = False
        problems.append("the request no longer plans after the rollback: %s" % exc)
    if problems:
        out["rollbackProblems"] = problems
    complete = back and not problems
    out.update(rolledBack=True, rollbackComplete=complete)
    outcome("rolled-back" if complete else "rollback-incomplete")
    note_journal("commit", op, pid, "rolled-back" if complete else "rollback-incomplete", fresh["changes"])
    if complete:
        out["error"] = ("the commit of %s did not write what was previewed - %s - so it was ROLLED BACK, and planning the "
                        "request again gives the preview exactly: the file is as the preview found it. Nothing was kept."
                        % (pid, cause))
    else:
        out["undoKept"] = True
        out["error"] = ("the commit of %s did not write what was previewed - %s - and the rollback is INCOMPLETE: "
                        "rollbackDifferences and rollbackProblems name what is not back. The undo step is kept, so undo "
                        "takes the whole commit back." % (pid, cause))
    return out


# ------------------------------------------------------------------------------------------------------------ mesh ops
# THE SCRATCH PLAN (module docstring). SCRATCH is the copy's name for the length of one plan; an object already holding it
# refuses the plan rather than be renamed under somebody.
SCRATCH = "__mif_plan_scratch__"
_ID_KINDS = ("objects", "meshes", "materials", "images", "textures", "node_groups", "actions", "collections", "curves",
             "armatures", "lattices", "texts")
# Each edge mark: (the row's field name, the mesh attribute from 4.0, the MeshEdge property before it).
_MARKS = (("use_seam", None, "use_seam"), ("use_edge_sharp", None, "use_edge_sharp"),
          ("crease", "crease_edge", "crease"), ("bevel_weight", "bevel_weight_edge", "bevel_weight"))


def _r5(values):
    """Rounded to 5 places, -0.0 folded into 0.0 so a digest of the same mesh is the same text. Every caller hands it
    floats Blender gave back (a foreach_get buffer, a vertex's co), never a caller's value."""
    return [round(v, 5) + 0.0 for v in values]


def _digest(*parts):
    import hashlib
    return hashlib.sha1(json.dumps(parts, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _edge_marks(me, mark):
    _field, attr, legacy = next(m for m in _MARKS if m[0] == mark)
    n = len(me.edges)
    if attr is None:
        out = [False] * n
        me.edges.foreach_get(legacy, out)
        return [bool(x) for x in out]
    layer = me.attributes.get(attr)
    out = [0.0] * n
    try:
        if layer is not None:
            layer.data.foreach_get("value", out)
        else:
            me.edges.foreach_get(legacy, out)
    except (AttributeError, TypeError, RuntimeError):
        out = [0.0] * n                               # 4.0+ with no layer: every edge reads 0, as the op itself reads it
    return _r5(out)


def _weights(obj, i):
    names = {g.index: g.name for g in obj.vertex_groups}
    return {names[g.group]: round(g.weight, 5) + 0.0 for g in obj.data.vertices[i].groups
            if g.group in names and g.weight > 0.0}


def _uv(me, name):
    layer = me.uv_layers.get(name)
    if layer is None:
        return None
    flat = [0.0] * (len(layer.data) * 2)
    layer.data.foreach_get("uv", flat)
    return {"loops": len(layer.data), "digest": _digest(_r5(flat))}


def _shape_key(me, name):
    keys = me.shape_keys
    block = keys.key_blocks.get(name) if keys is not None else None
    if block is None:
        return None
    flat = [0.0] * (len(block.data) * 3)
    block.data.foreach_get("co", flat)
    return {"digest": _digest(_r5(flat))}


def _mesh_state(obj):
    """Everything a mesh op here can change on `obj`, in the shapes the rows carry - read the same way off the scratch
    copy (after) and the object (before, and the read-back)."""
    me = obj.data
    nv, ne, nf = len(me.vertices), len(me.edges), len(me.polygons)
    co = [0.0] * (nv * 3)
    me.vertices.foreach_get("co", co)
    co = _r5(co)
    ev = [0] * (ne * 2)
    me.edges.foreach_get("vertices", ev)
    lv = [0] * len(me.loops)
    me.loops.foreach_get("vertex_index", lv)
    ls, lt, mat = [0] * nf, [0] * nf, [0] * nf
    me.polygons.foreach_get("loop_start", ls)
    me.polygons.foreach_get("loop_total", lt)
    me.polygons.foreach_get("material_index", mat)
    names = {g.index: g.name for g in obj.vertex_groups}
    weights = [{names[g.group]: round(g.weight, 5) + 0.0 for g in v.groups if g.group in names and g.weight > 0.0}
               for v in me.vertices] if names else [{} for _ in range(nv)]
    return {"nv": nv, "ne": ne, "nf": nf, "co": [co[i:i + 3] for i in range(0, len(co), 3)],
            "topo": [ev, lv, ls, lt], "marks": {m[0]: _edge_marks(me, m[0]) for m in _MARKS}, "weights": weights,
            "mat": mat, "groups": [g.name for g in obj.vertex_groups],
            "uv": {layer.name: _uv(me, layer.name) for layer in me.uv_layers},
            "uvActive": me.uv_layers.active.name if me.uv_layers.active is not None else None,
            "keys": {b.name: _shape_key(me, b.name) for b in me.shape_keys.key_blocks} if me.shape_keys is not None else {},
            "modifiers": ["%s:%s" % (m.name, m.type) for m in obj.modifiers]}


def _canon(obj, state):
    """ORDER-FREE digests, for a mesh whose topology the op rewrote: the geometry (every vertex, edge with its marks,
    face with its material and weights, by position), each UV layer and each shape key, keyed by the vertices' positions
    rather than their indices. MEASURED 2026-10-09 on 5.2.1: extrude_skirt run twice on one mesh gave the same 28
    vertices in a different order each time - bmesh hands the new elements back in an order of its own - so a digest in
    index order called the same result a different one, and every commit drifted from its own preview."""
    me = obj.data
    pos = [tuple(c) for c in state["co"]]
    ev, lv, ls, lt = state["topo"]
    edges = sorted(tuple(sorted((pos[ev[2 * i]], pos[ev[2 * i + 1]]))) + tuple(state["marks"][m[0]][i] for m in _MARKS)
                   for i in range(state["ne"]))
    faces, turns = [], []
    for f in range(state["nf"]):
        loops = list(range(ls[f], ls[f] + lt[f]))
        at = [pos[lv[k]] for k in loops]
        s = at.index(min(at))                      # the same face from its lowest corner, its winding kept
        turn = loops[s:] + loops[:s]
        turns.append(turn)
        faces.append([[pos[lv[k]] for k in turn], state["mat"][f]])
    weights = sorted([pos[i], sorted(w.items())] for i, w in enumerate(state["weights"]))
    out = {"geometry": _digest(sorted(pos), edges, sorted(faces), weights), "uv": {}, "keys": {}}
    for layer in me.uv_layers:
        flat = [0.0] * (len(layer.data) * 2)
        layer.data.foreach_get("uv", flat)
        flat = _r5(flat)
        out["uv"][layer.name] = {"loops": len(layer.data), "digest": _digest(sorted(
            [[pos[lv[k]], flat[2 * k], flat[2 * k + 1]] for k in turn] for turn in turns))}
    if me.shape_keys is not None:
        for block in me.shape_keys.key_blocks:
            flat = [0.0] * (len(block.data) * 3)
            block.data.foreach_get("co", flat)
            flat = _r5(flat)
            out["keys"][block.name] = {"digest": _digest(sorted([pos[i], flat[3 * i:3 * i + 3]]
                                                                for i in range(len(pos))))}
    return out


def _geometry(state):
    return {"vertices": state["nv"], "edges": state["ne"], "faces": state["nf"], "digest": state["canon"]["geometry"]}


def _same_topology(b, a):
    return b["nv"] == a["nv"] and b["topo"] == a["topo"]


def _mesh_rows(name, b, a):
    """The changes from the object's state `b` to the scratch copy's `a` - per element while the topology holds, one
    geometry change when it does not (no vertex of the one corresponds to a vertex of the other), with the UV layers and
    shape keys then compared order-free (_canon, which both states carry in that case)."""
    rows = []

    def row(kind, field, before, after, **private):
        op = "create" if before is None else "delete" if after is None else "modify"
        r = {"kind": kind, "op": op, "target": name, "field": field, "before": before, "after": after}
        r.update({"_" + k: v for k, v in private.items()})
        rows.append(r)
    same = _same_topology(b, a)
    if not same:
        b = dict(b, uv=b["canon"]["uv"], keys=b["canon"]["keys"])
        a = dict(a, uv=a["canon"]["uv"], keys=a["canon"]["keys"])
    if same:
        for i, (x, y) in enumerate(zip(b["co"], a["co"])):
            if x != y:
                row("vertex", "vertices[%d].co" % i, x, y, index=i)
        for mark, _attr, _legacy in _MARKS:
            for i, (x, y) in enumerate(zip(b["marks"][mark], a["marks"][mark])):
                if x != y:
                    row("edge", "edges[%d].%s" % (i, mark), x, y, index=i, mark=mark)
        for i, (x, y) in enumerate(zip(b["weights"], a["weights"])):
            if x != y:
                row("weights", "vertices[%d].groups" % i, dict(sorted(x.items())), dict(sorted(y.items())), index=i)
        if b["mat"] != a["mat"]:
            row("faces", "polygons.material_index", {"digest": _digest(b["mat"])}, {"digest": _digest(a["mat"])})
    else:
        row("geometry", "geometry", _geometry(b), _geometry(a))
    for g in a["groups"]:
        if g not in b["groups"]:
            row("vertex_group", "vertex_groups['%s']" % g, None, "exists", name=g)
    for g in b["groups"]:
        if g not in a["groups"]:
            row("vertex_group", "vertex_groups['%s']" % g, "exists", None, name=g)
    if [g for g in a["groups"] if g in b["groups"]] != [g for g in b["groups"] if g in a["groups"]]:
        row("vertex_groups", "vertex_groups", b["groups"], a["groups"])
    for layer in list(b["uv"]) + [n for n in a["uv"] if n not in b["uv"]]:
        if b["uv"].get(layer) != a["uv"].get(layer):
            row("uv", "uv_layers['%s']" % layer, b["uv"].get(layer), a["uv"].get(layer), name=layer, canon=not same)
    if b["uvActive"] != a["uvActive"] and a["uvActive"] is not None:
        row("uv_active", "uv_layers.active", b["uvActive"], a["uvActive"])
    for key in list(b["keys"]) + [n for n in a["keys"] if n not in b["keys"]]:
        if b["keys"].get(key) != a["keys"].get(key):
            row("shape_key", "shape_keys['%s']" % key, b["keys"].get(key), a["keys"].get(key), name=key,
                canon=not same)
    if b["modifiers"] != a["modifiers"]:
        row("modifiers", "modifiers", b["modifiers"], a["modifiers"])
    return rows


def read_mesh(change, kept):
    """A mesh row's value on the object now, in its `after`'s shape. An edge mark is read for the whole mesh once per
    read-back (cached in `kept`), not once per edge."""
    import bpy
    obj = bpy.data.objects.get(change["target"])
    if obj is None or obj.type != "MESH":
        return None
    me, kind, i = obj.data, change["kind"], change.get("_index")
    if kind == "vertex":
        return _r5(me.vertices[i].co) if i < len(me.vertices) else None
    if kind == "edge":
        cache = kept.setdefault("_marks", {}) if isinstance(kept, dict) else {}
        marks = cache.get((obj.name, change["_mark"]))
        if marks is None:
            marks = cache[(obj.name, change["_mark"])] = _edge_marks(me, change["_mark"])
        return marks[i] if i < len(marks) else None
    if kind == "weights":
        return dict(sorted(_weights(obj, i).items())) if i < len(me.vertices) else None
    if kind == "faces":
        mat = [0] * len(me.polygons)
        me.polygons.foreach_get("material_index", mat)
        return {"digest": _digest(mat)}
    if kind in ("geometry", "uv", "shape_key") and (kind == "geometry" or change.get("_canon")):
        cache = kept.setdefault("_canon", {}) if isinstance(kept, dict) else {}
        state = cache.get(obj.name)
        if state is None:
            state = cache[obj.name] = _mesh_state(obj)
            state["canon"] = _canon(obj, state)
        if kind == "geometry":
            return _geometry(state)
        return state["canon"]["uv" if kind == "uv" else "keys"].get(change["_name"])
    if kind == "vertex_group":
        return "exists" if obj.vertex_groups.get(change["_name"]) is not None else None
    if kind == "vertex_groups":
        return [g.name for g in obj.vertex_groups]
    if kind == "uv":
        return _uv(me, change["_name"])
    if kind == "uv_active":
        return me.uv_layers.active.name if me.uv_layers.active is not None else None
    if kind == "shape_key":
        return _shape_key(me, change["_name"])
    if kind == "modifiers":
        return ["%s:%s" % (m.name, m.type) for m in obj.modifiers]
    raise MifOpError("no reader for a %s change" % kind)


def _ids():
    import bpy
    return {k: {i.as_pointer() for i in getattr(bpy.data, k)} for k in _ID_KINDS if hasattr(bpy.data, k)}


def scratch_run(obj, fn, request, keys):
    """Run the op's ordinary handler on a scratch copy of `obj` (its mesh copied too) and return (the object's
    _mesh_state, the copy's), each with its order-free digests when the topology changed. The copy, and every data-block
    the run made, is removed before this returns, and the selection, active object and frame are put back - whatever
    the run did. A refusal on the copy is the op's refusal, with the object's name in it."""
    import bpy
    from .ops_common import selection_restore, selection_snapshot
    if bpy.data.objects.get(SCRATCH) is not None:
        raise MifOpError("an object is named %s - the name a preview gives its scratch copy for the length of one plan. "
                         "Rename it. Nothing was written." % SCRATCH)
    scene = bpy.context.scene
    ids0, sel, frame = _ids(), selection_snapshot(), scene.frame_current
    scratch = obj.copy()
    scratch.data = obj.data.copy()
    scratch.name = SCRATCH
    scene.collection.objects.link(scratch)
    bpy.context.view_layer.update()
    run = dict(request)
    for k in keys:
        if k in run:
            run[k] = SCRATCH
    try:
        try:
            fn(run)
        except MifOpError as exc:
            raise MifOpError(str(exc).replace(SCRATCH, obj.name))
        except Exception as exc:                                   # noqa: BLE001 - said as the plan's failure
            raise MifOpError("planning it raised %s: %s. Nothing was written."
                             % (type(exc).__name__, str(exc).replace(SCRATCH, obj.name)))
        before, after = _mesh_state(obj), _mesh_state(scratch)
        if not _same_topology(before, after):
            before["canon"], after["canon"] = _canon(obj, before), _canon(scratch, after)
        return before, after
    finally:
        try:
            if bpy.context.view_layer.objects.active is scratch and scratch.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
        except (RuntimeError, ReferenceError):
            pass
        made = [i for k, ptrs in ((k, ids0.get(k, set())) for k in _ID_KINDS if hasattr(bpy.data, k))
                for i in getattr(bpy.data, k) if i.as_pointer() not in ptrs]
        if made:
            bpy.data.batch_remove(made)
            # BEFORE THE RESTORE, which iterates the view layer: a removed object is None there until an update, and one
            # that came first raises out of the restore, which swallows it, so nothing is put back (measured on 5.0.1).
            bpy.context.view_layer.update()
        selection_restore(sel)
        if scene.frame_current != frame:
            scene.frame_set(frame)
        bpy.context.view_layer.update()


def mesh_spec(fn, keys=("object", "name"), check=None):
    """The PREVIEW entry of a mesh op that plans on a scratch copy. `keys` are the request keys naming the object (the
    first present one is it; each is pointed at the copy for the run); check(request, dry-run response, obj) refuses
    what the copy cannot show - a refusal the real call would make and the copy would not."""
    def plan(request):
        import bpy
        resp = fn(dict(request, dryRun=True))                    # the op's own refusals and dry-run answer
        name = next((request[k] for k in keys if isinstance(request.get(k), str)), None)
        obj = bpy.data.objects.get(name) if name else None
        if obj is None or obj.type != "MESH":
            raise MifOpError("'%s' is not a mesh object. Nothing was written." % name)
        if check is not None:
            check(request, resp, obj)
        before, after = scratch_run(obj, fn, request, keys)
        context =["mesh %s: %d vertices, %d edges, %d faces" % (obj.data.name, before["nv"], before["ne"], before["nf"])]
        sharing = sorted(o.name for o in bpy.data.objects if o.data is obj.data and o is not obj)
        if sharing:
            context.append("the mesh is shared - %s change with it" % ", ".join(sharing))
        return {"changes": _mesh_rows(obj.name, before, after), "unchanged": [], "context": context,
                "commit": dict(request), "response": resp, "target": obj.name}
    return {"plan": plan, "read": read_mesh, "capture": capture_mesh, "restore": restore_mesh, "release": release_mesh}


def _modifier_state(mod):
    """A modifier's settings, to make it again: every writable RNA property, and its ID properties (a Geometry Nodes
    modifier's inputs)."""
    props = {}
    for p in mod.bl_rna.properties:
        if p.identifier in ("rna_type", "name", "type") or p.is_readonly or p.type == "COLLECTION":
            continue
        try:
            value = getattr(mod, p.identifier)
        except (AttributeError, TypeError):
            continue
        props[p.identifier] = value if p.type == "POINTER" or isinstance(value, (str, int, float, bool)) \
            else tuple(value)
    custom = {}
    try:
        names = list(mod.keys())
    except TypeError:                                   # most modifier types hold no ID properties, and say so this way
        names = []
    for k in names:
        v = mod[k]
        custom[k] = v.to_dict() if hasattr(v, "to_dict") else v.to_list() if hasattr(v, "to_list") else v
    return {"name": mod.name, "type": mod.type, "props": props, "custom": custom}


def _put_modifiers(obj, states):
    """The modifier stack back as `states` records it: ones it lacks removed, missing ones made with their settings,
    the order restored."""
    import bpy
    want = [s["name"] for s in states]
    for mod in [m for m in obj.modifiers if m.name not in want]:
        obj.modifiers.remove(mod)
    for s in states:
        if obj.modifiers.get(s["name"]) is not None:
            continue
        mod = obj.modifiers.new(s["name"], s["type"])
        mod.name = s["name"]
        for k, v in s["props"].items():
            try:
                setattr(mod, k, v)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                pass
        for k, v in s["custom"].items():
            try:
                mod[k] = v
            except (TypeError, ValueError):
                pass
    for index, name in enumerate(want):
        cur = obj.modifiers.find(name)
        if cur in (-1, index):
            continue
        if hasattr(obj.modifiers, "move"):
            obj.modifiers.move(cur, index)
        else:
            with bpy.context.temp_override(object=obj, active_object=obj):
                bpy.ops.object.modifier_move_to_index(modifier=name, index=index)


def capture_mesh(plan):
    """A copy of the mesh (its shape keys with it), the vertex group names and the modifier stack, just before the
    commit's body runs. The copy has no user until a rollback remaps the object back onto it; release_mesh removes it."""
    import bpy
    obj = bpy.data.objects[plan["target"]]
    return {"object": obj.name, "mesh": obj.data.copy(), "meshName": obj.data.name,
            "groups": [g.name for g in obj.vertex_groups],
            "modifiers": [_modifier_state(m) for m in obj.modifiers]}


def restore_mesh(kept, _plan):
    import bpy
    obj = bpy.data.objects.get(kept["object"])
    if obj is None:
        raise MifOpError("'%s' is gone, so its mesh cannot be put back" % kept["object"])
    copy, cur = kept["mesh"], obj.data
    if cur is not copy:
        cur.user_remap(copy)                    # every user of the written mesh - another object sharing it too
        if cur.users == 0:
            bpy.data.meshes.remove(cur)
        copy.name = kept["meshName"]
    for g in [g for g in obj.vertex_groups if g.name not in kept["groups"]]:
        obj.vertex_groups.remove(g)
    for name in kept["groups"]:
        if obj.vertex_groups.get(name) is None:
            obj.vertex_groups.new(name=name)
    if ["%s:%s" % (m.name, m.type) for m in obj.modifiers] != ["%s:%s" % (s["name"], s["type"])
                                                               for s in kept["modifiers"]]:
        _put_modifiers(obj, kept["modifiers"])
    obj.data.update()
    bpy.context.view_layer.update()


def release_mesh(kept):
    """The capture's copy, when nothing uses it - after a commit that stood, or a rollback that did not need it."""
    import bpy
    copy = kept.get("mesh") if kept else None
    try:
        if copy is not None and copy.users == 0:
            bpy.data.meshes.remove(copy)
    except ReferenceError:
        pass


# ------------------------------------------------------------------------------------------------------------ batch
_BATCH_NOTES = {
    "continueOnError": "a previewed batch commits every op or rolls every op back - there is no running on past one",
    "budgetSeconds": "a previewed batch is planned whole before it commits; send a plain batch for a budget",
    "stopOnBudget": "a previewed batch is planned whole before it commits; send a plain batch for a budget",
}


def is_bulk_batch(pid):
    """Whether previewId names a batch preview of this module (a plan) rather than one the change cap made (a target
    fingerprint, committed by cap.take_commit)."""
    rec = cap._PREVIEWS.get(pid) if isinstance(pid, str) else None
    return rec is not None and rec.get("plan") is not None


def batch_spec(table):
    """batch's PREVIEW entry, built per call on the op table. Each op is planned by its own plan against the file as it
    is; batch runs them in order, so two ops that change one target are refused (the later one's preview of it could
    never read back as shown), as is an op with no plan or none that an inverse takes back (refused[], no previewId)."""
    every = specs()

    def plan(request):
        from .ops_common import reject_unknown
        reject_unknown(request, ("ops",), "batch", notes=_BATCH_NOTES)
        ops = request.get("ops")
        if not isinstance(ops, list) or not ops:
            raise MifOpError("a batch preview needs 'ops': a non-empty list of {op, params}. Nothing was written.")
        steps, changes, unchanged, refused, context, owner = [], [], [], [], [], {}
        for i, entry in enumerate(ops):
            name = entry.get("op") or entry.get("endpoint") if isinstance(entry, dict) else None
            sub = (entry.get("params") or {}) if isinstance(entry, dict) else None
            if not name or not isinstance(sub, dict):
                raise MifOpError("ops[%d] must be {op, params}. Nothing was written." % i)
            if name not in table:
                raise MifOpError("ops[%d] names unknown op '%s'. Nothing was written." % (i, name))
            spec_i = every.get(name)
            if spec_i is None:
                refused.append({"index": i, "op": name, "reason": "%s takes no preview" % name})
                continue
            if spec_i.get("noRollback"):
                refused.append({"index": i, "op": name, "reason": "no inverse takes it back (%s), so it could not go "
                                                                 "back with the rest of the batch" % spec_i["noRollback"]})
                continue
            carried = [k for k in PREVIEW_KEYS if k in sub]
            if carried:
                raise MifOpError("ops[%d] (%s) carries %s - a batch preview previews every op itself, and its previewId "
                                 "commits them all. Nothing was written." % (i, name, ", ".join(carried)))
            try:
                p = spec_i["plan"](dict(sub))
            except MifOpError as exc:
                raise MifOpError("ops[%d] (%s) refused: %s" % (i, name, exc))
            for c in p["changes"]:
                t = c.get("target")
                if owner.get(t, i) != i:
                    raise MifOpError("ops[%d] (%s) and ops[%d] (%s) both change '%s' - batch runs its ops in order, so the "
                                     "later one's preview of it could never read back as shown. Preview them one at a "
                                     "time, or send a plain batch. Nothing was written."
                                     % (owner[t], ops[owner[t]].get("op"), i, name, t))
                owner[t] = i
            changes += [dict(c, step=i, endpoint=name, _step=i) for c in p["changes"]]
            unchanged += [dict(c, step=i, endpoint=name, _step=i) for c in p.get("unchanged") or []]
            context += ["ops[%d] %s: %s" % (i, name, x) for x in p.get("context") or []]
            steps.append({"index": i, "op": name, "plan": p})
        return {"changes": changes, "unchanged": unchanged, "context": context, "refused": refused, "steps": steps,
                "commit": {"ops": [{"op": s["op"], "params": s["plan"]["commit"]} for s in steps]},
                "response": {"steps": [{"index": s["index"], "op": s["op"], "response": s["plan"].get("response")}
                                       for s in steps]}}

    def read(change, kept):
        return every[change["endpoint"]]["read"](change, kept[change["_step"]]["kept"])

    def capture(plan_):
        return {s["index"]: {"op": s["op"], "kept": every[s["op"]]["capture"](s["plan"])
                             if every[s["op"]].get("capture") else None} for s in plan_["steps"]}

    def restore(kept, plan_):
        problems = []
        for s in reversed(plan_["steps"]):                  # the last op's change first, as an undo would
            try:
                every[s["op"]]["restore"](kept[s["index"]]["kept"], s["plan"])
            except Exception as exc:                        # noqa: BLE001 - every op is tried, then all are reported
                problems.append("ops[%d] %s: %s: %s" % (s["index"], s["op"], type(exc).__name__, exc))
        if problems:
            raise MifOpError("; ".join(problems))

    def release(kept):
        for k in (kept or {}).values():
            if every[k["op"]].get("release") is not None:
                every[k["op"]]["release"](k["kept"])

    return {"plan": plan, "read": read, "capture": capture, "restore": restore, "release": release}


def batch_runner(table):
    """The commit's body: each op's handler on its planned request, in order. A refusal stops it, and the commit rolls
    every op back."""
    def run(commit):
        results = []
        for i, entry in enumerate(commit["ops"]):
            try:
                results.append({"index": i, "op": entry["op"], "result": table[entry["op"]](dict(entry["params"]))})
            except MifOpError as exc:
                raise MifOpError("ops[%d] (%s): %s" % (i, entry["op"], exc))
        return {"results": results}
    return run


# ------------------------------------------------------------------------------------------------------------ listing
def op_list_previews(params):
    """The live previews of this Blender, or one preview's changes a page at a time - Unreal's list_previews.

    Without previewId: {pid, previews[{previewId, endpoint, createdUtc, expiresInSeconds, changes, unchanged, refused,
    spent, outcome?, expired}], count}. With one: {preview{...}, changes[page], offset, limit, changesTotal, context[],
    refused[]}. offset from 0, limit 1 to 1000 (default 200). A preview the change cap made for a batch lists with its
    targets as changes. Read-only."""
    from .ops_common import reject_unknown, take_int
    reject_unknown(params, ("previewId", "offset", "limit"), "list_previews",
                   notes={"id": "spelt previewId - the key every preview answers with",
                          "endpoint": "not a filter here - the listing names each preview's endpoint"})
    offset = take_int(params, "offset", default=0)
    limit = take_int(params, "limit", default=INLINE)
    if offset < 0 or not 1 <= limit <= 1000:
        raise MifOpError("offset must be a whole number from 0, and limit a whole number from 1 to 1000")
    cap._prune()
    now = time.monotonic()

    def preview_row(pid, r):
        plan = r.get("plan") or {}
        out = {"previewId": pid, "endpoint": r["op"], "createdUtc": r.get("createdUtc"),
               "expiresInSeconds": round(r["expires"] - now), "changes": len(plan.get("changes") or r.get("keys") or []),
               "unchanged": len(plan.get("unchanged") or []), "refused": 0, "spent": r["spent"],
               "expired": now > r["expires"]}
        if r["spent"]:
            out["outcome"] = r.get("outcome") or "committed"
        return out

    pid = params.get("previewId")
    if pid is None:
        rows = [preview_row(p, r) for p, r in sorted(cap._PREVIEWS.items(), key=lambda kv: -kv[1]["created"])]
        return {"pid": os.getpid(), "previews": rows, "count": len(rows)}
    rec = cap._PREVIEWS.get(pid)
    if rec is None:
        raise MifOpError("previewId %s is not held by this Blender (pid %d) - it was never made here, Blender restarted, "
                         "or it was dropped" % (pid, os.getpid()))
    plan = rec.get("plan")
    changes = ([_public(c, i) for i, c in enumerate(plan["changes"])] if plan
               else [{"i": i, "kind": "target", "op": "modify", "target": k} for i, k in enumerate(rec.get("keys") or [])])
    return {"pid": os.getpid(), "preview": preview_row(pid, rec), "changes": changes[offset:offset + limit], "offset": offset,
            "limit": limit, "changesTotal": len(changes), "context": list((plan or {}).get("context") or []),
            "refused": []}


OPS = {"list_previews": op_list_previews}
