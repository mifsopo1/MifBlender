"""Every tool of Epic's UEFN MCP, with MifBridge's guarantees: the catalog reader, the strict validator, whole vectors,
the corrected access mode, and the generic read-back. Pure Python - the listener loads it by path inside UEFN, and the
MCP server and the offline tests import it on the host.

The catalog (mif_uefn/uefn_catalog.json) is generated from the toolsets' own schemas by mif_uefn/gen_catalog.py and
stamped with the UEFN build it was read from. What it adds to each tool, and why (each measured on UEFN
43.10):
  - mode: read or write. Epic marks many reads 'execute' (ListEventBindings, GetDeviceProperties, ...), and the write
    gate needs the truth: a name that only reads (Get, List, Find, Is, Has, Can, Describe, Read, Search, Capture, Trace,
    Count, Query) is read whatever its annotation says, unless the annotation is write;
  - getter: the Get<X> beside a Set<X> in the same toolset, whose arguments the setter's cover - used to read a write
    back, and to fill a partial vector from the current value;
  - deny: tools no MifBridge caller reaches through the generic route - publishing, Fab, account and revision control,
    editor exit, raw Python, and the session start/stop that uefn_session wraps with the client lifecycle and the
    anti-cheat lock.
Validation (validate): an unknown key is refused by name at every depth (Epic's tools drop it silently); a number is a
JSON number (never a string or a bool - Epic stores 'abc' as 0); enums, patterns and integer/number bounds are checked;
and a vector (x/y/z, pitch/yaw/roll, x/y, x/y/z/w, r/g/b/a) is either whole or filled from the current value, because
Epic fills a missing component with 0 and reports success.
"""
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG_FILE = os.path.join(os.path.dirname(HERE), "uefn_catalog.json")
VECTOR_SETS = [("x", "y", "z"), ("pitch", "yaw", "roll"), ("x", "y"), ("x", "y", "z", "w"), ("r", "g", "b", "a")]
READ_VERBS = ("Get", "List", "Find", "Is", "Has", "Can", "Describe", "Read", "Search", "Capture", "Trace", "Count",
              "Query", "get_", "list_", "find_", "is_", "has_", "describe_", "read_", "search_", "grep_", "query_",
              "count_")
DENY_TOOLSETS = {
    "ValkyrieEditor.ValkyrieRevisionControl": "revision control (check-in, sync) is the person's",
    "EmporiumToolset": "Fab listings and licences are the person's",
}
DENY_TOOLS = {
    "EditorToolset.EditorAppToolset.ExitEditor": "MifBridge never closes UEFN",
    "editor_toolset.toolsets.python_scripting.PythonScriptingTools.execute_python":
        "raw Python is MifBridge's transport, not a tool; ask for a typed tool",
    "editor_toolset.toolsets.programmatic.ProgrammaticToolset.execute_tool_script":
        "raw Python is MifBridge's transport, not a tool; ask for a typed tool",
    "ValkyrieToolset.ValkyriePythonToolset.EnablePythonForMainProject":
        "turning on project Python lets its startup scripts run; the person does it in UEFN",
    "ValkyrieFortniteToolset.SessionToolset.StartSession":
        "uefn_session start owns this: it records the client and waits for Easy Anti-Cheat",
    "ValkyrieFortniteToolset.SessionToolset.StopSession": "uefn_session end owns this: it closes only our client",
}
# Case-sensitive on the CamelCase tool name, so Prefab is not Fab.
DENY_WORDS = re.compile(r"Publish|(^|[a-z])Fab($|[A-Z])|Emporium|Licen[cs]e|Purchase|Account|Login|Logout")


class Invalid(Exception):
    pass


# ---------------------------------------------------------------- catalog
_CACHE = {}


def load(path=None):
    path = path or CATALOG_FILE
    if path not in _CACHE:
        with open(path, encoding="utf-8") as fh:
            _CACHE[path] = json.load(fh)
    return _CACHE[path]


# A running UEFN reports its build as "<engine>-<CL>+<branch>" (6.0.0-60458657+++Fortnite+Release-43.10); the catalog
# is stamped "<branch>-CL-<CL>" (++Fortnite+Release-43.10-CL-60458657). Both name the release and the changelist.
_RELEASE = re.compile(r"Release-(\d+(?:\.\d+)+)")
_CL_STAMP = re.compile(r"-CL-(\d+)")
_CL_ENGINE = re.compile(r"^\d+\.\d+\.\d+-(\d+)\+")


def build_of(text):
    """(release, changelist) read from a running engine version or a catalog stamp; None for a part not found."""
    text = text if isinstance(text, str) else ""
    rel = _RELEASE.search(text)
    cl = _CL_STAMP.search(text) or _CL_ENGINE.search(text)
    return (rel.group(1) if rel else None, cl.group(1) if cl else None)


def reachable(running_engine, path=None):
    """How many of Epic's UEFN tools uefn_call reaches, from the catalog actually loaded - the panel's EPIC TOOLS stat
    and the listener's ping. A number only when the catalog loaded AND is stamped with the build that is running:
    state 'ok' carries count (catalogued tools minus the denied ones); 'missing' (no catalog, or it does not read),
    'mismatch' (stamped for another build) and 'unknown' (the running build cannot be read) carry count None and say
    why, because a count from a catalog that is not this UEFN's is not what this UEFN reaches."""
    out = {"state": None, "count": None, "catalogued": None, "denied": None, "catalogBuild": None,
           "runningBuild": running_engine if isinstance(running_engine, str) else None, "why": None}
    try:
        cat = load(path)
        tools = cat["tools"]
        if not isinstance(tools, dict) or not tools:
            raise ValueError("it lists no tools")
    except (OSError, ValueError, KeyError, TypeError) as e:
        out.update(state="missing", why="the catalog %s did not load: %s" % (path or CATALOG_FILE, e))
        return out
    denied = sum(1 for r in tools.values() if isinstance(r, dict) and r.get("deny"))
    out.update(catalogued=len(tools), denied=denied, catalogBuild=cat.get("uefn"))
    have, want = build_of(out["catalogBuild"]), build_of(out["runningBuild"])
    if None in have:
        out.update(state="missing", why="the catalog carries no readable UEFN build stamp (%r)" % out["catalogBuild"])
    elif None in want:
        out.update(state="unknown", why="the running UEFN build cannot be read from %r" % out["runningBuild"])
    elif have != want:
        out.update(state="mismatch", why="the catalog is for UEFN %s CL %s and UEFN %s CL %s is running; regenerate "
                                         "it with gen_catalog.py --live" % (have + want))
    else:
        out.update(state="ok", count=len(tools) - denied)
    return out


def header_value(info):
    """The panel's EPIC TOOLS value: the count only for state 'ok', otherwise a word that is not a number."""
    info = info if isinstance(info, dict) else {}
    if info.get("state") == "ok" and isinstance(info.get("count"), int):
        return str(info["count"])
    return {"missing": "no catalog", "mismatch": "mismatch"}.get(info.get("state"), "unknown")


def corrected_mode(name, annotated):
    short = name.rsplit(".", 1)[-1]
    if annotated == "write":
        return "write"
    if annotated == "read" or short.startswith(READ_VERBS):
        return "read"
    return "write"


def deny_reason(full):
    toolset = full.rsplit(".", 1)[0]
    for ts, why in DENY_TOOLSETS.items():
        if toolset.startswith(ts):
            return why
    if full in DENY_TOOLS:
        return DENY_TOOLS[full]
    if DENY_WORDS.search(full.rsplit(".", 1)[-1]):
        return "publishing, Fab, licences and accounts are the person's"
    return None


def find_getter(full, tools):
    """Set<X> -> Get<X> in the same toolset when every required input of the getter is an input of the setter."""
    toolset, short = full.rsplit(".", 1)
    if not short.startswith("Set"):
        return None
    cand = "%s.Get%s" % (toolset, short[3:])
    g = tools.get(cand)
    if not g:
        return None
    s_props = set((tools[full].get("inputSchema") or {}).get("properties") or {})
    g_req = set((g.get("inputSchema") or {}).get("required") or [])
    return cand if g_req <= s_props else None


def _is_ref(schema):
    """An object reference in Epic's schemas: an object whose only property is refPath."""
    return isinstance(schema, dict) and set((schema.get("properties") or {})) == {"refPath"}


def ref_args(input_schema):
    """The top-level arguments that name an existing object (an actor, asset, device, entity, component, ...)."""
    return [k for k, v in ((input_schema or {}).get("properties") or {}).items()
            if _is_ref(v) or (isinstance(v, dict) and v.get("type") == "array" and _is_ref(v.get("items")))]


def returns_ref(output_schema):
    """Whether a tool's answer carries an object reference (what a create, place or import made)."""
    rv = ((output_schema or {}).get("properties") or {}).get("returnValue")
    seen, stack = 0, [rv]
    while stack and seen < 200:
        node = stack.pop()
        seen += 1
        if _is_ref(node):
            return True
        if isinstance(node, dict):
            stack.extend(v for k, v in node.items() if k in ("properties", "items") or isinstance(v, dict))
            if isinstance(node.get("properties"), dict):
                stack.extend(node["properties"].values())
    return False


def verify_route(row):
    """How a write is checked after it runs, best first:
      getter  - its own Get<X>, answer compared with what was sent;
      target  - the object(s) it names are read before and after through UEFN's property reader (ObjectToolset
                ListProperties/GetProperties) and the actor API: a sent value whose name is a property is compared,
                and every change is listed;
      result  - what it made (a reference in its answer) is read back: it exists, its class and properties;
      asset   - the content paths it names (assetPath, folderPath, ...) are checked before and after: exists or not;
      none    - nothing to read: the answer says unverified."""
    if row.get("mode") != "write":
        return None
    if row.get("getter"):
        return "getter"
    if ref_args(row.get("inputSchema")):
        return "target"
    if returns_ref(row.get("outputSchema")):
        return "result"
    if path_args(row.get("inputSchema")):
        return "asset"
    return "none"


PATH_ARG = re.compile(r"(?i)(^|[a-z])(path|paths|folder|folderpath|assetpath|destinationpath|sourcepath)$")


def path_args(input_schema):
    """String (or string-list) arguments that name a content path - an asset or a folder - by their name. A file on
    disk (sourceFile, filename) is not one."""
    out = []
    for k, v in ((input_schema or {}).get("properties") or {}).items():
        if not isinstance(v, dict) or not PATH_ARG.search(k) or "file" in k.lower():
            continue
        if v.get("type") == "string" or (v.get("type") == "array" and (v.get("items") or {}).get("type") == "string"):
            out.append(k)
    return out


def vector_keys(schema):
    props = set((schema or {}).get("properties") or {})
    for keys in VECTOR_SETS:
        if props == set(keys):
            return keys
    return None


# ---------------------------------------------------------------- validation
def _type_ok(v, t):
    if t == "object":
        return isinstance(v, dict)
    if t == "array":
        return isinstance(v, list)
    if t == "string":
        return isinstance(v, str)
    if t == "boolean":
        return isinstance(v, bool)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    if t == "null":
        return v is None
    return True


def validate(value, schema, where="args", partial_vectors=None):
    """Raise Invalid naming the first problem. partial_vectors, when a list, collects (where, missing keys) for vectors
    sent partial instead of refusing them - the caller fills them from the current value or refuses."""
    if not isinstance(schema, dict) or not schema:
        return
    if "oneOf" in schema or "anyOf" in schema:
        errs = []
        for sub in schema.get("oneOf") or schema.get("anyOf"):
            try:
                validate(value, sub, where, partial_vectors)
                return
            except Invalid as e:
                errs.append(str(e))
        raise Invalid("%s matches none of its allowed shapes (%s)" % (where, "; ".join(errs[:3])))
    t = schema.get("type")
    if isinstance(t, list):
        if not any(_type_ok(value, x) for x in t):
            raise Invalid("%s must be %s, got %s" % (where, " or ".join(t), _jtype(value)))
    elif isinstance(t, str):
        if value is None and schema.get("default", "nodefault") is None:
            return
        if not _type_ok(value, t):
            raise Invalid("%s must be %s, got %s%s" % (where, t, _jtype(value),
                                                         " (a number is sent as a JSON number, never a string)"
                                                         if t in ("number", "integer") and isinstance(value, str) else ""))
    if "enum" in schema and value not in schema["enum"]:
        raise Invalid("%s must be one of %s, got %r" % (where, ", ".join(map(repr, schema["enum"][:20])), value))
    if isinstance(value, str) and schema.get("pattern") and not re.search(schema["pattern"], value):
        raise Invalid("%s does not match %s" % (where, schema["pattern"]))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if lo is not None and value < lo or hi is not None and value > hi:
            raise Invalid("%s must be within %s..%s, got %s" % (where, lo, hi, value))
    if isinstance(value, dict) and (t == "object" or "properties" in schema):
        props = schema.get("properties") or {}
        extra = schema.get("additionalProperties")
        unknown = sorted(k for k in value if k not in props)
        if unknown and not isinstance(extra, dict) and extra is not True:
            raise Invalid("%s: unknown key %s (accepted: %s) - Epic's tool would drop it silently"
                          % (where, ", ".join(unknown), ", ".join(sorted(props)) or "none"))
        vec = vector_keys(schema)
        if vec:
            missing = [k for k in vec if k not in value]
            if missing:
                if partial_vectors is None:
                    raise Invalid("%s is missing %s: send every component (%s) - UEFN fills a missing one with 0 and "
                                  "reports success" % (where, ", ".join(missing), ", ".join(vec)))
                partial_vectors.append((where, missing))
        req = [k for k in schema.get("required") or [] if k not in value]
        if req and not vec:
            raise Invalid("%s is missing required %s" % (where, ", ".join(req)))
        for k, v in value.items():
            validate(v, props.get(k) if k in props else extra if isinstance(extra, dict) else {},
                     "%s.%s" % (where, k), partial_vectors)
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, v in enumerate(value):
            validate(v, schema["items"], "%s[%d]" % (where, i), partial_vectors)


def _jtype(v):
    return {bool: "a boolean", int: "an integer", float: "a number", str: "a string", list: "a list",
            dict: "an object", type(None): "null"}.get(type(v), type(v).__name__)


# A setter and its getter do not always name a field alike (measured, UEFN 43.10): SetActorTransform takes
# `translation` and GetActorTransform answers `location`; SetEntityTransform nests the vectors under `transform` and
# GetEntityTransform answers them at the top.
ALIASES = {"translation": "location", "location": "translation"}


def _lookup(current, path):
    """The value at path in the getter's answer, trying each key's alias and the path without its leading wrappers."""
    for start in range(len(path)):
        node = current
        for k in path[start:]:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(k, node.get(ALIASES.get(k, k)))
        if node is not None:
            return node
    return None


def fill_partial(args, partials, current):
    """Fill each partial vector at its path from `current` (the getter's answer). Returns the paths filled; raises
    Invalid when the current value does not hold that vector."""
    filled = []
    for where, missing in partials:
        path = where.split(".")[1:]
        node = args
        for k in path[:-1]:
            node = node[k]
        have = _lookup(current, path)
        if not isinstance(have, dict) or any(m not in have for m in missing):
            raise Invalid("%s is missing %s and its current value could not be read to fill it; send every component"
                          % (where, ", ".join(missing)))
        for m in missing:
            node[path[-1]][m] = have[m]
        filled.append("%s.%s" % (where, "/".join(missing)))
    return filled

def same(a, b, tol=1e-3):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol * max(1.0, abs(a))
    if isinstance(a, dict) and isinstance(b, dict):
        return all(k in b and same(v, b[k], tol) for k, v in a.items())
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, str) and isinstance(b, str):
        return a == b or a.lower() == b.lower()
    return a == b


def compare_readback(write_args, getter_schema, got):
    """(compared, mismatches): the write's values that the getter answers, compared. A scalar answer is compared with
    the one write argument the getter does not take; a dict answer key by key."""
    g_props = set((getter_schema or {}).get("properties") or {})
    sent = {k: v for k, v in write_args.items() if k not in g_props}
    if not sent:
        return False, []
    if isinstance(got, dict):
        pairs = [(k, _lookup(got, [k])) for k in sent]
        # a wrapper the getter does not repeat (SetEntityTransform's transform) is compared against the whole answer
        pairs = [(k, g if g is not None else (got if isinstance(sent[k], dict) and set(sent[k]) & set(got) else None))
                 for k, g in pairs]
        pairs = [(k, g) for k, g in pairs if g is not None]
        if not pairs:
            return False, []
        return True, ["%s want %r got %r" % (k, sent[k], g) for k, g in pairs if not same(sent[k], g)]
    if len(sent) == 1:
        (k, v), = sent.items()
        return True, [] if same(v, got) else ["%s want %r got %r" % (k, v, got)]
    return False, []
