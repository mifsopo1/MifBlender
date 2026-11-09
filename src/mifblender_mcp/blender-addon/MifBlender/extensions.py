"""Blender add-on extensions: ops a SEPARATE add-on registers into MifBlender's dispatcher (docs/32_EXTENSIONS.md section 8).

The Blender half of the extension API. An extension's server tools already reach every built-in op through api.blender;
a NEW op needs code inside Blender. Instead of forking MifBlender, a separate add-on registers its ops here, and they
are dispatched by the same _dispatch every built-in uses - the undo step pushed after each write, the change cap's
counting (cap.py), batch, describe_op and the call journal apply to them without a second copy of any of them.

THE CONTRACT, versioned like the native API (MifBridgeExtensionApi.h):

    import mifblender_api                        # published by MifBlender's register(); absent until it is enabled
    mifblender_api.register_ops("AcmeBlender", "acme_", [
        {"name": "acme_count_faces", "fn": op_count_faces, "keys": ["object"], "required": ["object"],
         "undo": "read", "summary": "Count an object's faces."},
    ], api_version=1)
    ...
    mifblender_api.unregister_ops("AcmeBlender")    # in the add-on's unregister()

  fn(params) -> dict         called with the request's params AFTER they were checked: no unknown key (reject_unknown's
                             own words), every required key present. Raise mifblender_api.MifOpError to refuse - the
                             message reaches the caller as error, without a traceback. Runs on Blender's main thread.
  keys / required            the accepted keys, and those that must be present
  capKeys                    keys whose values are the data-blocks the call changes: a string is one target, each entry
                             of a list one. Over the open file's change cap the call writes nothing and is refused, as
                             cap.py refuses a built-in that has no preview (bulk previews are not offered in v1)
  undo                       "step" (default: one undo step pushed after it runs, success or refusal), "read" (changes
                             nothing, pushes none), "external" (writes outside the scene - a file - which undo cannot
                             take back; the scene part is still a step). describe_op reports it as undoClass
  summary                    one sentence; describe_op answers it

WHAT REGISTRATION REFUSES, all or nothing (a half-registered provider is worse than none): an api version this MifBlender
does not speak (both numbers in the error), a provider that is not a plain name, a prefix outside the rule the server's
mifext.py uses ([a-z][a-z0-9]{1,15}_, not bl_ mif_ kr_ ue_ ext_), an op whose name lacks the prefix, a built-in op's name
or another provider's, a non-callable fn, a key named twice or reserved (readBack, op, previewId, dryRun), a required or
capKey that is not an accepted key, an unknown undo class, an unknown descriptor key.

NO bpy HERE: the module is pure so tools/test_wave3_w4e.py can drive it offline; the dispatcher integration is in
server.py (_op_table), cap.py (targets_of / gate), ops_undo.py (read ops push no step) and ops_describe.py.
"""
import re

from .ops_common import MifOpError, reject_unknown

API_VERSION = 1                     # the newest this MifBlender speaks
SUPPORTED = (1,)                    # every version it accepts; a new one ADDS to the descriptor, never changes v1
UNDO_CLASSES = ("step", "read", "external")
DESC_KEYS = ("name", "fn", "keys", "required", "capKeys", "undo", "summary")
RESERVED_KEYS = ("readBack", "op", "previewId", "dryRun")
RESERVED_PREFIXES = ("bl_", "mif_", "kr_", "ue_", "ext_")
_PREFIX = re.compile(r"^[a-z][a-z0-9]{1,15}_$")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_PROVIDER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{1,63}$")

_OPS = {}                           # op name -> record
_PROVIDERS = {}                     # provider -> {"prefix", "apiVersion", "ops": [names]}


class ExtensionError(Exception):
    """A registration refused - the message says which rule and what to change."""


def api_version():
    return API_VERSION


def _builtin_names():
    from . import server                                    # runtime: server imports this module
    return set(server._builtin_op_table())                  # noqa: SLF001 - same package


def register_ops(provider, prefix, ops, api_version=API_VERSION, builtins=None):
    """Register a provider's ops. Returns their names. Raises ExtensionError, registering nothing, on any problem.
    `builtins` is for the offline tests: the built-in op names to refuse clashes with (default: the live table)."""
    errs = []
    if not isinstance(api_version, int) or isinstance(api_version, bool) or api_version not in SUPPORTED:
        raise ExtensionError("extension api version %r is not one this MifBlender speaks (%s; newest %d) - the add-on "
                             "was written against a newer MifBlender: update MifBlender, or build the add-on against "
                             "version %d" % (api_version, ", ".join(map(str, SUPPORTED)), API_VERSION, API_VERSION))
    if not isinstance(provider, str) or not _PROVIDER.match(provider):
        errs.append("provider %r must be a plain name (a letter, then 1-63 letters, digits or _)" % (provider,))
    elif provider in _PROVIDERS:
        errs.append("provider %r is already registered - unregister_ops(%r) first (an add-on reload calls its "
                    "unregister)" % (provider, provider))
    if not isinstance(prefix, str) or not _PREFIX.match(prefix):
        errs.append("prefix %r must be 2-16 lowercase letters or digits ending in '_' (e.g. 'acme_')" % (prefix,))
        prefix = None
    elif prefix in RESERVED_PREFIXES:
        errs.append("prefix %r is reserved (%s)" % (prefix, ", ".join(RESERVED_PREFIXES)))
    if not isinstance(ops, (list, tuple)) or not ops:
        raise ExtensionError("; ".join(errs + ["ops must be a non-empty list of op descriptors"]))
    taken = set(builtins) if builtins is not None else _builtin_names()
    staged, seen = {}, set()
    for i, d in enumerate(ops):
        where = "ops[%d]" % i
        if not isinstance(d, dict):
            errs.append("%s is not a descriptor object" % where)
            continue
        name = d.get("name")
        if isinstance(name, str):
            where = "op %r" % name
        for k in sorted(set(d) - set(DESC_KEYS)):
            errs.append("%s: unknown descriptor key %r (known: %s)" % (where, k, ", ".join(DESC_KEYS)))
        if not isinstance(name, str) or not _NAME.match(name):
            errs.append("%s: name must be lowercase snake_case" % where)
            continue
        if prefix and not name.startswith(prefix):
            errs.append("%s: must start with the prefix %r" % (where, prefix))
        if name in taken:
            errs.append("%s is a built-in MifBlender op - built-ins win; rename it" % where)
        elif name in _OPS:
            errs.append("%s is already registered by %s" % (where, _OPS[name]["provider"]))
        elif name in seen:
            errs.append("%s is listed twice" % where)
        seen.add(name)
        if not callable(d.get("fn")):
            errs.append("%s: fn must be callable fn(params) -> dict" % where)
        keys = d.get("keys", [])
        if not isinstance(keys, (list, tuple)) or any(not isinstance(k, str) or not k for k in keys):
            errs.append("%s: keys must be a list of parameter names" % where)
            keys = []
        if len(set(keys)) != len(keys):
            errs.append("%s: keys names a parameter twice" % where)
        bad = sorted(set(keys) & set(RESERVED_KEYS))
        if bad:
            errs.append("%s: %s reserved by the dispatcher (readBack is the transport's, op is batch's routing key, "
                        "previewId and dryRun the bulk preview's, which v1 does not offer)" % (where, ", ".join(bad)))
        for field in ("required", "capKeys"):
            vals = d.get(field, [])
            if not isinstance(vals, (list, tuple)) or any(not isinstance(k, str) for k in vals):
                errs.append("%s: %s must be a list of parameter names" % (where, field))
                continue
            stray = sorted(set(vals) - set(keys))
            if stray:
                errs.append("%s: %s names %s, which keys does not accept" % (where, field, ", ".join(stray)))
        undo = d.get("undo", "step")
        if undo not in UNDO_CLASSES:
            errs.append("%s: undo %r must be one of %s" % (where, undo, ", ".join(UNDO_CLASSES)))
        if "summary" in d and not isinstance(d["summary"], str):
            errs.append("%s: summary is one sentence of text" % where)
        staged[name] = {"provider": provider, "apiVersion": api_version, "fn": d.get("fn"), "keys": tuple(keys),
                        "required": tuple(d.get("required") or ()), "capKeys": tuple(d.get("capKeys") or ()),
                        "undo": undo, "summary": (d.get("summary") or "").strip()}
    if errs:
        raise ExtensionError("%s registered nothing: %s" % (provider, "; ".join(errs)))
    _OPS.update(staged)
    _PROVIDERS[provider] = {"prefix": prefix, "apiVersion": api_version, "ops": sorted(staged)}
    return sorted(staged)


def unregister_ops(provider):
    """Remove every op a provider registered. Returns how many - 0 for a provider that registered none."""
    gone = (_PROVIDERS.pop(provider, None) or {}).get("ops") or []
    for name in gone:
        _OPS.pop(name, None)
    return len(gone)


def is_extension(name):
    return name in _OPS


def undo_class(name):
    rec = _OPS.get(name)
    return rec["undo"] if rec else None


def _wrapped(name):
    def handler(params):
        rec = _OPS.get(name)
        if rec is None:                                     # unregistered between the table read and the call
            raise MifOpError("op '%s' is no longer registered (its add-on was disabled)" % name)
        reject_unknown(params, set(rec["keys"]), name)
        missing = [k for k in rec["required"] if params.get(k) is None]
        if missing:
            raise MifOpError("%s: missing required param(s) %s (from extension %s)"
                             % (name, ", ".join(missing), rec["provider"]))
        result = rec["fn"](params)
        out = dict(result) if isinstance(result, dict) else ({"result": result} if result is not None else {})
        out["providedBy"] = rec["provider"]
        return out
    handler.__name__ = "ext_" + name
    handler.__doc__ = (_OPS.get(name) or {}).get("summary") or ""
    return handler


def ops():
    """{name: wrapped handler} for the dispatcher's table - added AFTER the built-ins, which registration keeps apart."""
    return {name: _wrapped(name) for name in _OPS}


def cap_targets(name, params):
    """The data-blocks one call would change, from its capKeys - "object:<name>" keys, as cap.py counts. None when the
    op declares no capKeys (it is not counted, as a built-in with no counter is not)."""
    rec = _OPS.get(name)
    if not rec or not rec["capKeys"]:
        return None
    keys = []
    for k in rec["capKeys"]:
        v = params.get(k)
        vals = v if isinstance(v, (list, tuple)) else ([v] if v is not None else [])
        keys += ["object:%s" % x for x in vals if isinstance(x, str) and x]
    return list(dict.fromkeys(keys))


def describe(name):
    """describe_op's answer for an extension op: its declared keys, required keys, cap keys and undo class."""
    rec = _OPS[name]
    notes = {"step": "pushes one undo step after it runs, success or refusal",
             "read": "changes nothing; pushes no undo step",
             "external": "writes outside the scene (a file): undo takes back the scene, never the file"}
    return {"ok": True, "op": name, "handler": "ext_" + name, "module": "extension:%s" % rec["provider"],
            "acceptedKeys": sorted(rec["keys"]), "requiredKeys": sorted(rec["required"]),
            "undoClass": rec["undo"], "undoNote": notes[rec["undo"]], "summary": rec["summary"] or None,
            "extension": {"provider": rec["provider"], "apiVersion": rec["apiVersion"],
                          "capKeys": list(rec["capKeys"]) or None},
            "refusals": [], "refusalsFromHelpers": None,
            "note": "an extension op: its keys are the ones it registered with (reject_unknown runs before it), and its "
                    "own refusals are its add-on's"}


def summary():
    """For ping: the api version and each provider's ops."""
    return {"version": API_VERSION, "supported": list(SUPPORTED),
            "providers": [{"provider": p, "prefix": r["prefix"], "apiVersion": r["apiVersion"], "ops": list(r["ops"])}
                          for p, r in sorted(_PROVIDERS.items())]}
