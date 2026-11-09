"""What does this op take, and why is the key you guessed not on the list?

WHY THIS EXISTS, AND WHY IT IS THE LAST HALF OF THE NOTES WORK
==============================================================
reject_unknown gained a `notes` dict on 2026-09-07, so an addon op can answer the WORD a caller
guessed - add_modifier tells you that `apply` is not there because adding and applying are separate
on purpose, and points at apply_modifier. That closed most of the gap with the UE side.

Most, not all. UE serves the same table through describe_endpoint, so a caller can read it BEFORE
getting anything wrong; the addon could only answer someone who had already been refused. The
spec's own words for the remainder: "discovery BEFORE the call is not built... the addon has no
describe_op to serve them from". This is that op.

WHY IT READS THE SOURCE INSTEAD OF SHIPPING A GENERATED TABLE
============================================================
The UE side has to generate MifBridgeDescribe.cpp with harvest_param_table.py, because C++ cannot
look at its own call sites - and that generated file has to be kept honest by parity_check, which
has caught real drift more than once. Python does not have that problem. The accept-list is right
there in the .py file that is running, so this reads it with `ast` and cannot go stale by
construction. No generated artefact, no second source of truth, no drift check needed.

The one thing that would break it is renaming reject_unknown or calling it dynamically, and
parity_check already fails loudly on either.

RESOLVING THE ACCEPT-LIST
========================
`reject_unknown(params, accepted, endpoint, notes)` is called with `accepted` as either an inline
literal or a module-level constant (_RB_KEYS, _FLUID_KEYS...). A literal is read with
ast.literal_eval; a NAME is resolved by asking the already-imported module for that attribute,
which is exact rather than a second parse. That distinction has bitten this repo before: an earlier
measurement of bounded listings read inline literals only, could not see face_info's module-level
_FACE_KEYS, and reported 0 of 27 bounded when the true answer was 1.

AN OP WITH NO GUARD IS REPORTED AS SUCH, NOT AS ZERO KEYS
=========================================================
An op with no reject_unknown comes back with acceptedKeys: null and a `note` saying so, never with
an empty list - reporting "accepts nothing" for an op that simply has no guard is the same quiet
lie the notes exist to prevent. The UE side learned this about recipe_override_and_call_parent,
whose missing guard is deliberate and whose notes live on the op it delegates to.

AND AN OP WHOSE GUARD THIS READER CANNOT FOLLOW IS A THIRD ANSWER, not that one. Until 2026-09-15
both came back as a bare null, so "takes no parameters" and "has a list I could not parse" were
indistinguishable - and the two ops in that second state were set_camera and export_mesh, which
were at the time the two LARGEST unfilmed parameter sets in the corpus. The house rule is never
guess a shape, ask describe_op; on the two ops with the most shape to guess, following it returned
nothing. _resolve now reads `A | B` unions and set()/list() builders, which covers both, and where
it still cannot read an expression the note QUOTES the fragment it gave up on.

MEASURED 2026-09-08: ZERO of the 173 addon ops currently lack a guard, so THIS BRANCH DOES NOT
FIRE TODAY and is not covered by any test. It is kept because the alternative is worse - the first
op added without a guard would otherwise be reported as accepting nothing - but it is stated here
rather than left to look like tested behaviour. If it is still unreached when somebody next audits
dead defensive code, this paragraph is the evidence for deleting it.
"""
import ast
import inspect
import sys

from .ops_common import MifOpError, reject_unknown

_CACHE = {}


def _module_source(module):
    """Source text and parsed tree for a module, cached - parsing every ops module is not free."""
    key = getattr(module, "__name__", None) or repr(module)
    hit = _CACHE.get(key)
    if hit is None:
        try:
            src = inspect.getsource(module)
        except (OSError, TypeError) as exc:                        # pragma: no cover - packaged .pyc
            raise MifOpError(
                "cannot read the source of %s (%s), so its accepted keys cannot be reported. "
                "This happens when the addon is shipped without .py files." % (key, exc))
        hit = (src, ast.parse(src))
        _CACHE[key] = hit
    return hit


def _find_guard(tree, func_name):
    """The reject_unknown Call inside this function, or None.

    Nested function definitions are skipped: a helper defined inside a handler that guards its OWN
    params would otherwise be mistaken for the handler's guard. Same shape as the UE inserter
    picking the wrong definition when a handler was declared twice - worth avoiding by construction.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Name)
                        and inner.func.id == "reject_unknown"):
                    return inner
            return None
    return None


_SET_BUILDERS = {"set", "frozenset", "tuple", "list", "sorted"}


def _resolve(node, module):
    """A literal, a module-level name, a set()/list() of one, or a `|` union of those.

    THE UNION AND THE BUILDER CALL WERE ADDED 2026-09-15, and which two ops needed them is the
    argument for having them. Exactly two ops in the add-on answered acceptedKeys: null while HAVING
    a guard, and they were set_camera and export_mesh - at the time, the two largest unfilmed
    parameter sets in the whole corpus, 18 and 16, one on each seat's side of the showcase split.

    That is not a coincidence, and the direction runs one way: an op with a big accept-list is
    exactly the op whose author factored it into a constant plus a union -

        reject_unknown(params, _SET_CAMERA_KEYS | {"sensorHeight", "sensorFit"}, "set_camera")
        reject_unknown(params, set(_EXPORT_OVERRIDES) | {"object", "name", ...}, "export_mesh")

    - so this reader failed precisely where the list was longest and a caller needed it most. The
    house rule is never guess a shape, ask describe_op; following that rule on the two ops with the
    most shape to guess returned nothing at all. Found by the second seat while authoring the
    export round-trip clip, which is the sharpest possible way to find it: by obeying the rule.

    ALL-OR-NOTHING ON A UNION, DELIBERATELY, and it is the opposite call from the one made in
    showcase_coverage.py for what looks like the same problem. There, a partly-read call site is
    reported partly, because the consumer is a planner and a short list costs a clip filmed twice.
    Here the consumer is a CALLER deciding what to send, and a short accept-list tells it a
    parameter is not accepted when it is - so it silently does not send it. A partial set handed
    over as a complete one is the asset registry's failure mode, and the same answer applies: say
    you could not read it. The note names the unreadable fragment so a human can go and look.
    """
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        pass
    if isinstance(node, ast.Name):
        return getattr(module, node.id, None)
    # set(X) / frozenset(X) / tuple(X) / list(X) / sorted(X). set() over a DICT yields its keys,
    # which is what export_mesh means by set(_EXPORT_OVERRIDES) - the override table is keyed by
    # the parameter name.
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _SET_BUILDERS and len(node.args) == 1 and not node.keywords):
        inner = _resolve(node.args[0], module)
        if inner is None:
            return None
        try:
            return set(inner)
        except TypeError:
            return None
    # `A | B`, the union. ONLY BitOr: `-` and `&` NARROW an accept-list, and a reader that guessed
    # wrong about which way a narrowing went would report keys the guard actually rejects, which is
    # worse than reporting none. Neither appears in the add-on today; if one ever does, it should
    # be taught here deliberately rather than inherited from a generic operator walk.
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, right = _resolve(node.left, module), _resolve(node.right, module)
        if left is None or right is None:
            return None
        try:
            return set(left) | set(right)
        except TypeError:
            return None
    return None


def _unreadable_part(node, module):
    """Which fragment of an accept-list expression could not be read. For the note, not the answer."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        for side in (node.left, node.right):
            if _resolve(side, module) is None:
                return _unreadable_part(side, module)
    try:
        return ast.unparse(node)[:120]
    except (AttributeError, ValueError):       # ast.unparse is 3.9+; Blender ships newer, but a
        return type(node).__name__             # reader that crashes explaining itself is no help


# THE VALUE RULES, READ THE SAME WAY. acceptedKeys answers which WORDS an op takes; a caller with every word right still
# meets its value rules one refusal at a time - "surface takes 'keep' or 'disconnect'", "'density' cannot be negative".
# Each rule is written exactly once, as the text of a `raise MifOpError(...)`, so `refusals` reads those texts out of the
# running source: the format string as written, placeholders left in, because they are filled at call time. Helpers
# DEFINED IN THE SAME MODULE and called by name from the handler are followed one level, since that is where shared
# validation lives (set_material_volume's value checks are in _stage_volume_value). Helpers from other modules - take,
# get_object, reject_unknown - are not: their refusals are the same for every op and would bury the op's own. A
# MifOpError whose message is not a literal is COUNTED in refusalsUnreadable, never dropped, so a short list cannot pass
# for a complete one. Asked for by the second seat, which collected refusal vocabulary for describe_op by hand.
_REFUSAL_CAP = 100


def _message_text(node):
    """The literal text of a MifOpError argument - a string, "..." % args, "a" "b" concatenation, an f-string's literal
    parts, or "...".format(...) - or None when the message is built at runtime."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return _message_text(node.left)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _message_text(node.left), _message_text(node.right)
        return (left or "") + (right or "") if (left or right) else None
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        return _message_text(node.func.value)
    return None


def _refusals(tree, func_name):
    """(refusal texts, same-module helpers they came from, how many MifOpError messages were not literals)."""
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    root = funcs.get(func_name)
    if root is None:
        return [], [], 0
    texts, called = [], []
    unread = 0

    def scan(fn_node, collect_calls):
        nonlocal unread
        for inner in ast.walk(fn_node):
            if not isinstance(inner, ast.Call) or not isinstance(inner.func, ast.Name):
                continue
            if inner.func.id == "MifOpError":
                text = _message_text(inner.args[0]) if inner.args else None
                if text:
                    texts.append(" ".join(text.split()))
                else:
                    unread += 1
            elif collect_calls:
                called.append(inner.func.id)

    scan(root, True)
    followed = []
    for name in dict.fromkeys(called):
        helper = funcs.get(name)
        if helper is None or name == func_name or name.startswith("op_"):
            continue
        seen = len(texts) + unread
        scan(helper, False)
        if len(texts) + unread > seen:
            followed.append(name)
    return list(dict.fromkeys(texts)), followed, unread


def op_describe_op(params):
    """What an op accepts, and the recorded reason for keys it deliberately does not.

    Reads the accept-list and notes dict straight out of the running source with ast, so it cannot
    drift from the guard it describes. An op with no guard is reported as acceptedKeys: null with a
    reason, never as an empty list.
    """
    # ONE ACCEPTED KEY, AND `endpoint`/`name` GET NOTES INSTEAD OF BEING ALIASES. Both spellings are
    # plausible - `endpoint` is what the UE side's describe_endpoint calls it - so the temptation is
    # to accept all three. But this module's own reject_unknown docstring settles it: a note is NOT
    # an alias, because a note explains a key that deliberately does not exist rather than quietly
    # accepting it. Aliases also have a concrete cost here: param_reach reports every accepted key no
    # MCP tool sends, and three spellings for one value would have put two entries in a baseline
    # whose whole job is to stay near zero.
    reject_unknown(
        params, ("op",), "describe_op",
        notes={
            # Dogfooding: describe_op's own refusals carry notes, for the same reason every other
            # op's do. An op that explains other ops badly explains nothing.
            "endpoint": "spelt op here. The UE side calls it endpoint; this addon calls its units ops",
            "name": "spelt op - one op per call",
            "tool": "spelt op - one op per call",
            "ops": "one op per call; `ping` returns every registered op name",
            "all": "one op per call; `ping` returns every registered op name",
            "params": "describe_op REPORTS the accepted params; it does not take them",
            "verbose": "the whole record is returned every time - there is no short form",
        })

    name = params.get("op")
    if not isinstance(name, str) or not name.strip():
        raise MifOpError("op is required. `ping` lists every op.")
    name = name.strip()

    from . import server                                            # runtime: avoids a cycle
    table = server._op_table()                                      # noqa: SLF001 - same package
    fn = table.get(name)
    if fn is None:
        raise MifOpError("unknown op '%s'. `ping` returns the full list." % name)
    from . import extensions                                        # an op a separate add-on registered
    if extensions.is_extension(name) and getattr(fn, "__name__", "") == "ext_" + name:
        return extensions.describe(name)

    module = sys.modules.get(getattr(fn, "__module__", "") or "")
    if module is None:
        raise MifOpError("op '%s' resolves to a handler whose module is not imported" % name)

    _, tree = _module_source(module)
    call = _find_guard(tree, fn.__name__)

    out = {
        "ok": True,
        "op": name,
        "handler": fn.__name__,
        "module": getattr(module, "__name__", "?"),
    }
    # WHAT UNDO DOES WITH IT (1.3, docs/31_WORKFLOWS.md), from ops_undo's own sets - the ones server.py pushes steps by - so a
    # workflow's dry run can say before anything runs whether a step can be rolled back, without a second copy of the list.
    from . import ops_undo                                          # runtime: ops_undo imports bpy at module level
    if name in ops_undo.UNDO_OPS:
        out["undoClass"], out["undoNote"] = "undo", "moves through the undo stack itself; pushes no step"
    elif name in ops_undo.READ_ONLY_OPS:
        out["undoClass"], out["undoNote"] = "read", "changes nothing; pushes no undo step"
    elif name in ops_undo.RESETS_STACK:
        out["undoClass"], out["undoNote"] = "resets", "loads a file, which starts Blender's undo stack again - nothing before it can be undone"
    elif name in ops_undo.EXTERNAL_EFFECTS:
        out["undoClass"], out["undoNote"] = "external", ("writes outside the scene (a file on disk, a render, a generation): "
                                                         "undo takes back the scene, never the file")
    else:
        out["undoClass"], out["undoNote"] = "step", "pushes one undo step after it runs, success or refusal"
    # THE VALUE RULES - see _refusals. Placeholders (%s, %r, %d, {}) are filled when the op refuses.
    refusals, helpers, unreadable = _refusals(tree, fn.__name__)
    out["refusals"] = refusals[:_REFUSAL_CAP]
    if len(refusals) > _REFUSAL_CAP:
        out["refusalsTruncatedFrom"] = len(refusals)
    out["refusalsFromHelpers"] = helpers or None
    out["refusalsUnreadable"] = unreadable or None

    doc = inspect.getdoc(fn)
    if doc:
        # The FIRST paragraph. A whole handler docstring here is several screens of reasoning, and
        # the point of this op is an answer a caller can act on, not the essay behind it.
        out["summary"] = doc.split("\n\n")[0].strip()

    if call is None:
        out["acceptedKeys"] = None
        out["note"] = ("this op has no reject_unknown guard - it takes no parameters, or it "
                       "delegates to another op which owns the guard. Its accepted keys are not "
                       "reported here rather than reported as none.")
        return out

    out["guardLine"] = call.lineno
    accepted = _resolve(call.args[1] if len(call.args) > 1 else None, module)
    if isinstance(accepted, (set, frozenset, tuple, list)):
        out["acceptedKeys"] = sorted(str(k) for k in accepted)
    else:
        # NAME THE FRAGMENT. "not a literal or a module-level constant" is true and unactionable;
        # the caller cannot tell whether the op takes nothing, takes everything, or has a list this
        # reader could not follow. Quoting the expression turns it into a thing somebody can go and
        # look at, and is what makes a null here distinguishable from an op with no guard at all.
        out["acceptedKeys"] = None
        out["note"] = ("the accept-list could not be read statically - this op HAS a guard, so its "
                       "keys are unknown here rather than absent. Unreadable part: %s"
                       % _unreadable_part(call.args[1] if len(call.args) > 1 else None, module))

    notes = None
    if len(call.args) > 3:
        notes = _resolve(call.args[3], module)
    for kw in call.keywords:
        if kw.arg == "notes":
            notes = _resolve(kw.value, module)
    if isinstance(notes, dict):
        out["commonMistakes"] = [{"key": str(k), "hint": str(v)} for k, v in sorted(notes.items())]
    else:
        out["commonMistakes"] = []

    return out


OPS = {
    "describe_op": op_describe_op,
}
