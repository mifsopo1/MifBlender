"""Typed UEFN ops written as sequences of Epic tool calls, so they run on either transport (spec 1.5, 2026-11-25).

    ListenerCalls  each step is one `call` op through MifBridge's listener in UEFN, so the listener's gate, catalog
                   check, schema validation and confirm rule hold for every step (the default transport)
    EpicCalls      each step goes straight to Epic's UEFN MCP over HTTP (call_tool), for when the listener cannot be
                   installed (a UEFN release that blocks it, a project with Python off). The same guards run here on
                   the host: the catalog row, the deny list, the schema (unknown keys, JSON types, whole vectors), the
                   confirm rule, a write gate read from MIF_BRIDGE_WRITE_MODE in the MCP server's environment (read
                   unless the person set full or edit), and a drift check of every written tool's schema against
                   Epic's own describe_toolset answer. Slower: ~333 ms a call unfocused.

Every op takes a transport T first. T.call(full_tool, args, destroy=False) returns the tool's returnValue or raises
StepError; T.read refuses a tool the catalog does not hold as a read. A shape this file reads that is not in the
catalog's outputSchema is read tolerantly and handed back raw.

What runs over Epic's MCP and what does not (DIRECT, LISTENER_ONLY): the listener's actor read-back goes through the
unreal API (path, class, every level actor), which Epic's tools give only partly - FindActors answers labels and names,
not paths - so those answers say what they could not read (notRead). The imports (the animation clips go through the
editor's AssetImportTask), the capture (its file is confined inside UEFN), the session lifecycle (the client process
and Easy Anti-Cheat are watched from inside the editor) and the call log refuse by name over Epic's MCP.
"""
import json
import math
import os
import re
import time

from . import epic as _epic
from .listener import mifbridge_uefn_catalog as CAT

DEVICES = "ValkyrieFortniteToolset.DeviceToolset"
LEVEL = "EditorToolset.LevelToolset"
ACTOR = "EditorToolset.ActorToolset"
OBJECT = "EditorToolset.ObjectToolset"
VERSE = "VerseToolset.VerseToolset"
ENTITY = "ValkyrieToolset.EntityToolset"
PREFAB = "ValkyrieToolset.PrefabToolset"
ASSET = "EditorToolset.AssetToolset"
STATIC = "EditorToolset.StaticMeshToolset"
TEXTURE = "EditorToolset.TextureToolset"
LOGS = "ConsoleToolset.LogsToolset"
WRITE_MODES = ("read", "edit", "full")
TOL = 0.01


class StepError(Exception):
    """kind: refused (nothing was sent), gate (the write gate refused), drift (the catalog does not match the running
    UEFN for this tool), error (the tool raised), or a transport kind (down, busy, http, protocol, not_installed...)."""

    def __init__(self, kind, message, tool=None):
        super().__init__(message)
        self.kind = kind
        self.tool = tool

    def reply(self):
        key = "refused" if self.kind in ("refused", "gate", "drift") else "error"
        return {"ok": False, key: str(self), "kind": self.kind, "tool": self.tool}


def ref(path):
    return {"refPath": path}


def refpath(v):
    return v.get("refPath") if isinstance(v, dict) and isinstance(v.get("refPath"), str) else None


def _row(full):
    row = CAT.load()["tools"].get(full)
    if row is None:
        raise StepError("refused", "%s is not in the catalog for UEFN %s; nothing was sent" % (full, CAT.load().get(
            "uefn")), full)
    return row


def host_write_mode():
    """The Epic transport's gate, (mode, where from). There is no panel without the listener, so the person's setting is
    MIF_BRIDGE_WRITE_MODE in the MCP server's environment, read the way the listener reads it in UEFN's: full allows
    every write, edit allows every write that is not destructive, anything else (unset included) is read."""
    raw = (os.environ.get("MIF_BRIDGE_WRITE_MODE") or "").strip().lower()
    if raw in WRITE_MODES:
        return raw, "MIF_BRIDGE_WRITE_MODE=%s in the MCP server's environment" % raw
    return "read", "the safe default (MIF_BRIDGE_WRITE_MODE is %s in the MCP server's environment)" % (
        "%r" % raw if raw else "not set")


class _Calls:
    name = None

    def __init__(self):
        self.calls = 0
        self.writes = []
        self.ms = []

    def read(self, full, args):
        if _row(full)["mode"] != "read":
            raise StepError("refused", "%s is a write and this step only reads; nothing was sent" % full, full)
        return self.call(full, args)

    def _destroy_ok(self, full, row, destroy):
        if row.get("destructive") and not destroy:
            raise StepError("refused", "%s is destructive (%s) and this op did not ask for it; nothing was sent"
                            % (full, row["destructive"]), full)

    def content_root(self):
        return None


class ListenerCalls(_Calls):
    """Each step is the listener's `call` op: gated, catalog-checked and validated inside UEFN. The listener's own
    read-back of the step (verified, mismatch) is kept in self.verify; this op does its own read-back on top."""
    name = "listener"

    def __init__(self, backend):
        super().__init__()
        self.b = backend
        self.verify = []
        self._ping = None

    def ping(self):
        if self._ping is None:
            res = self.b.op("ping")
            self._ping = res.get("result") if res.get("ok") and isinstance(res.get("result"), dict) else {}
        return self._ping

    def write_mode(self):
        return self.ping().get("writeMode"), self.ping().get("writeModeFrom")

    def content_root(self):
        return self.ping().get("contentRoot")

    def call(self, full, args, destroy=False):
        row = _row(full)
        self._destroy_ok(full, row, destroy)
        a = {"tool": full, "args": args}
        if row.get("destructive"):
            a["confirm"] = row["name"]
        t0 = time.perf_counter()
        res = self.b.op("call", a)
        self.calls += 1
        self.ms.append(round((time.perf_counter() - t0) * 1000, 1))
        r = res.get("result")
        if res.get("refused") or not isinstance(r, dict):
            kind = "refused" if res.get("refused") else res.get("kind") or "error"
            raise StepError(kind, "%s: %s" % (full, res.get("refused") or res.get("error") or "no result"), full)
        if row["mode"] == "write":
            self.writes.append(full)
            self.verify.append({"tool": full, "verified": r.get("verified"), "mismatch": r.get("mismatch") or None})
        return r.get("value")

    def typed(self, op, args):
        """One of the listener's own typed ops (it reads the level through the unreal API): its flattened answer."""
        res = self.b.op(op, args)
        self.calls += 1
        if res.get("refused") or (not res.get("ok") and not isinstance(res.get("result"), dict)):
            raise StepError("refused" if res.get("refused") else res.get("kind") or "error",
                            "%s: %s" % (op, res.get("refused") or res.get("error")), op)
        return dict(res["result"], ok=res.get("ok"))


class EpicCalls(_Calls):
    """Each step straight to Epic's UEFN MCP, with the listener's guards run here first."""
    name = "epic"

    def __init__(self, mcp, timeout=120):
        super().__init__()
        self.m = mcp
        self.timeout = timeout
        self.schemas = {}           # toolset -> {full: inputSchema as Epic describes it now}
        self.drift_raw = {}

    def write_mode(self):
        return host_write_mode()

    def live_schemas(self, toolset):
        """Epic's describe_toolset answer for one toolset, read once a transport: the drift check's measurement."""
        if toolset not in self.schemas:
            from . import gen_catalog as G
            r = self.m.rpc("tools/call", {"name": "describe_toolset", "arguments": {"toolset_name": toolset}},
                           timeout=60) or {}
            try:
                doc = G._toolset_doc({"result": r})
            except ValueError:
                doc = {}
            self.schemas[toolset] = {t.get("name"): t.get("inputSchema") for t in doc.get("tools") or []
                                     if isinstance(t, dict)}
        return self.schemas[toolset]

    def drift(self, full, row):
        live = self.live_schemas(row["toolset"])
        if not live:
            return "describe_toolset answered no tools for %s" % row["toolset"]
        if full not in live:
            return "the running UEFN does not describe %s" % full
        if json.dumps(live[full], sort_keys=True) != json.dumps(row.get("inputSchema"), sort_keys=True):
            self.drift_raw[full] = live[full]
            return "%s's parameters differ from the catalog's" % full
        return None

    def call(self, full, args, destroy=False):
        row = _row(full)
        if row.get("deny"):
            raise StepError("refused", "%s is not reachable through MifBridge: %s" % (full, row["deny"]), full)
        self._destroy_ok(full, row, destroy)
        args = json.loads(json.dumps(args if args is not None else {}))
        try:
            CAT.validate(args, row.get("inputSchema") or {}, full)
        except CAT.Invalid as e:
            raise StepError("refused", "%s; nothing was sent" % e, full)
        if row["mode"] == "write":
            mode, came = host_write_mode()
            if mode == "read" or (mode == "edit" and row.get("destructive")):
                raise StepError("gate", "the Epic-MCP transport is in %s mode (%s): %s is a %swrite. The person sets "
                                "MIF_BRIDGE_WRITE_MODE=full (or edit) in the MCP server's environment; nothing was sent"
                                % (mode, came, full, "destructive " if mode == "edit" else ""), full)
            why = self.drift(full, row)
            if why:
                raise StepError("drift", "%s. Writes through a changed tool are refused until the catalog is "
                                "regenerated (gen_catalog.py --live); nothing was sent" % why, full)
        t0 = time.perf_counter()
        try:
            text, st = self.m.call(row["toolset"], row["name"], args, timeout=self.timeout)
        except _epic.EpicError as e:
            raise StepError("error" if e.kind == "tool" else e.kind, str(e), full)
        finally:
            self.calls += 1
            self.ms.append(round((time.perf_counter() - t0) * 1000, 1))
        if row["mode"] == "write":
            self.writes.append(full)
        if st is not None:
            return _epic.return_value(st)
        return text or None

    def content_root(self):
        roots = self.read(ASSET + ".GetAssetContentRoots", {"bWritable": True})
        roots = [r for r in roots if isinstance(r, str)] if isinstance(roots, list) else []
        return roots[0].rstrip("/") if len(roots) == 1 else None


# ---------------------------------------------------------------- transforms (the listener's rules, on the host)
XF_PARTS = (("location", ("x", "y", "z")), ("rotation", ("pitch", "yaw", "roll")), ("scale", ("x", "y", "z")))


def _num(v, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise StepError("refused", "%s must be a finite number, got %r" % (where, v))
    return float(v)


def whole_xform(x, where="xform"):
    """location, rotation and scale, every component, finite, no zero scale - UEFN fills a missing one with 0."""
    if not isinstance(x, dict):
        raise StepError("refused", "%s must be an object with location, rotation and scale" % where)
    names = [p for p, _ in XF_PARTS]
    missing, extra = [p for p in names if p not in x], sorted(k for k in x if k not in names)
    if missing or extra:
        raise StepError("refused", "%s needs location, rotation and scale (missing %s, unknown %s); every transform "
                        "is sent whole" % (where, missing or "none", extra or "none"))
    out = {}
    for part, comps in XF_PARTS:
        v = x[part]
        if not isinstance(v, dict) or set(v) != set(comps):
            raise StepError("refused", "%s.%s needs exactly %s; UEFN fills a missing component with 0, so a partial "
                            "vector is refused" % (where, part, ", ".join(comps)))
        out[part] = {c: _num(v[c], "%s.%s.%s" % (where, part, c)) for c in comps}
    if any(abs(out["scale"][c]) < 1e-6 for c in "xyz"):
        raise StepError("refused", "%s.scale has a zero component %s" % (where, out["scale"]))
    return out


def merge_xform(x, current, where="xform"):
    """A partial transform filled from the current one, part by part. Returns (whole, [what was filled])."""
    if not isinstance(x, dict) or not x:
        raise StepError("refused", "%s must be an object with any of location, rotation, scale" % where)
    extra = sorted(k for k in x if k not in dict(XF_PARTS))
    if extra:
        raise StepError("refused", "%s: unknown key %s (accepted: location, rotation, scale)" % (where, extra))
    cur = read_xform(current)
    if cur is None:
        raise StepError("refused", "the current transform could not be read to fill %s; send it whole" % where)
    out, filled = {}, []
    for part, comps in XF_PARTS:
        sent = x.get(part)
        if sent is None:
            out[part] = dict(cur[part])
            filled.append(part)
            continue
        if not isinstance(sent, dict) or not sent or any(k not in comps for k in sent):
            raise StepError("refused", "%s.%s must be an object with any of %s" % (where, part, ", ".join(comps)))
        out[part] = {}
        for c in comps:
            if c in sent:
                out[part][c] = sent[c]
            else:
                out[part][c] = cur[part][c]
                filled.append("%s.%s" % (part, c))
    return whole_xform(out, where), filled


def read_xform(v):
    """A transform answer (GetActorTransform, GetEntityTransform) read tolerantly: None when it does not hold one."""
    if not isinstance(v, dict):
        return None
    out = {}
    for part, comps in XF_PARTS:
        p = v.get(part, v.get("translation") if part == "location" else None)
        if not isinstance(p, dict) or any(not isinstance(p.get(c), (int, float)) or isinstance(p.get(c), bool)
                                          for c in comps):
            return None
        out[part] = {c: float(p[c]) for c in comps}
    return out


def _angle(a, b):
    d = (a - b) % 360.0
    return min(d, 360.0 - d)


def compare_xform(want, got_raw):
    got = read_xform(got_raw)
    if got is None:
        return ["the transform read back is not readable: %s" % json.dumps(got_raw, default=str)[:200]]
    bad = []
    for part in ("location", "scale"):
        for k, v in want[part].items():
            if abs(got[part][k] - v) > TOL * max(1.0, abs(v)):
                bad.append("%s.%s want %g got %g" % (part, k, v, got[part][k]))
    for k, v in want["rotation"].items():
        if _angle(got["rotation"][k], v) > 0.01:
            bad.append("rotation.%s want %g got %g" % (k, v, got["rotation"][k]))
    return bad


def _text(a, key, required=True):
    v = a.get(key)
    if v is None or v == "":
        if required:
            raise StepError("refused", "%s is required" % key)
        return None
    if not isinstance(v, str):
        raise StepError("refused", "%s must be a string, got %s" % (key, type(v).__name__))
    return v


def _json(v):
    """A JSON-encoded string answer decoded; anything else as it came."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _unref(v):
    if isinstance(v, dict) and set(v) == {"refPath"}:
        return v["refPath"]
    if isinstance(v, list):
        return [_unref(x) for x in v]
    return v


def same_value(want, got):
    return CAT.same(_unref(want), _unref(got))


def _same_number(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def _option_holds(row, value):
    got = row.get("currentValue")
    return got == value or str(row.get("currentLabel", "")).lower() == str(value).lower() or _same_number(got, value)


# ---------------------------------------------------------------- the typed ops over Epic's MCP (item 1)
# The argument keys of each op, the listener's ARGS for the same op exactly (test_uefn_steps_offline holds them equal
# by reading the listener's source), so a call refused there is refused here.
DIRECT_ARGS = {
    "find": {"label_prefix", "limit"}, "actor": {"path"}, "place_mesh": {"label", "asset", "xform"},
    "set_xform": {"path", "xform"}, "delete_actor": {"path"}, "place_device": {"label", "asset", "xform"},
    "get_options": {"device", "options"}, "set_option": {"device", "option", "value"},
    "get_verse_properties": {"device", "properties"}, "set_verse_property": {"device", "property", "value"},
    "bind_event": {"source", "event", "target", "function", "remove"}, "bindings": {"device"},
    "set_property": {"object", "property", "value"}, "verse_build": {"path", "content"},
    "find_entities": {"name_filter"}, "create_prefab": {"folder", "name", "entities"},
    "call": {"tool", "args", "confirm"},
}
LISTENER_ONLY = {
    "ping": "ping is the listener's own answer; uefn_status reports the Epic transport",
    "log": "the call log lives in the listener; over Epic's MCP nothing keeps one",
    "import_static": "the import reads its mesh back through the unreal API (class, LODs) inside UEFN",
    "import_skeletal": "the animation clips import through the editor's AssetImportTask, which only the unreal API in "
                       "UEFN reaches",
    "capture": "the capture's file is confined to MifBridge's output folder inside UEFN, and its camera check reads "
               "the editor there",
    "session_status": "the session ops watch the client process and Easy Anti-Cheat from inside the editor",
    "session_join": "the session ops watch the client process and Easy Anti-Cheat from inside the editor",
    "session_start": "StartSession is denied to the generic route; the listener owns the client it launches",
    "session_end": "StopSession is denied to the generic route; the listener closes only the client it launched",
    # uefn_refresh_catalog's in-process half: over Epic's MCP the refresh is gen_catalog.py --live's HTTP route instead.
    "describe_toolsets": "it reads toolset_registry inside UEFN; over Epic's MCP the catalog is rebuilt by gen_catalog.py "
                         "--live",
    "catalog_reload": "it reloads the catalog the listener holds; over Epic's MCP the host reads its catalog file itself",
}


def actor_read(T, path):
    if hasattr(T, "typed"):
        return T.typed("actor", {"path": path})
    xf = T.read(ACTOR + ".GetActorTransform", {"actor": ref(path)})
    label = T.read(ACTOR + ".GetLabel", {"actor": ref(path)})
    return {"path": path, "label": label, "class": None, "xform": read_xform(xf) or xf,
            "notRead": {"class": "Epic's actor tools answer no class for a path; the listener reads it"}}


def _d_find(T, a):
    prefix = _text(a, "label_prefix", required=False) or ""
    limit = a.get("limit", 200)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5000:
        raise StepError("refused", "limit must be an integer 1-5000")
    rows = T.read(LEVEL + ".FindActors", {"filter": {"name": prefix}} if prefix else {}) or []
    hits = sorted((r for r in rows if isinstance(r, dict) and str(r.get("label", "")).startswith(prefix)),
                  key=lambda r: (str(r.get("label")), str(r.get("name"))))
    out = [{"label": r.get("label"), "name": r.get("name"), "class": refpath(r.get("class")), "guid": r.get("guid")}
           for r in hits[:limit]]
    return {"count": len(hits), "returned": len(out), "truncated": len(hits) > len(out), "actors": out,
            "notRead": {"path": "FindActors answers labels and names, not object paths",
                        "xform": "read one actor's transform with uefn_get_actor and its path"}}


def _d_actor(T, a):
    return actor_read(T, _text(a, "path"))


def _set_xform(T, path, x):
    T.call(ACTOR + ".SetActorTransform", {"actor": ref(path), "translation": x["location"], "rotation": x["rotation"],
                                          "scale": x["scale"], "bWorldSpace": True})


def _settle(T, path, label, x):
    """Read back, re-send the whole transform once if it differs, judge the second read."""
    st = actor_read(T, path)
    first = compare_xform(x, st["xform"])
    if first:
        _set_xform(T, path, x)
        st = actor_read(T, path)
    bad = compare_xform(x, st["xform"])
    if st["label"] != label:
        bad.append("label want %r got %r" % (label, st["label"]))
    return st, first, bad


def _d_place_mesh(T, a):
    x = whole_xform(a.get("xform"))
    label, asset = _text(a, "label"), _text(a, "asset")
    path = refpath(T.call(LEVEL + ".AddToSceneFromAsset", {"assetPath": asset, "name": label, "xform": x}))
    if not path:
        raise StepError("error", "AddToSceneFromAsset returned no actor for %s" % asset)
    st, first, bad = _settle(T, path, label, x)
    return {"path": path, "readback": st, "mismatch": bad, "correctedAfterPlace": first or None,
            "leftInLevel": [path] if bad else None}


def _d_set_xform(T, a):
    path = _text(a, "path")
    cur = T.read(ACTOR + ".GetActorTransform", {"actor": ref(path)})
    x, filled = merge_xform(a.get("xform"), cur)
    _set_xform(T, path, x)
    st = actor_read(T, path)
    return {"path": path, "readback": st, "filledFromCurrent": filled or None, "mismatch": compare_xform(x, st["xform"])}


def _gone(T, full, args):
    """True when the read of something just removed raises (it is not there), False when it answers."""
    try:
        T.read(full, args)
        return False
    except StepError as e:
        if e.kind != "error":
            raise
        return True


def _d_delete_actor(T, a):
    path = _text(a, "path")
    label = T.read(ACTOR + ".GetLabel", {"actor": ref(path)})
    said = T.call(LEVEL + ".RemoveFromScene", {"actor": ref(path)}, destroy=True)
    gone = _gone(T, ACTOR + ".GetLabel", {"actor": ref(path)})
    return {"path": path, "label": label, "removedSaid": said, "inLevelAfter": not gone,
            "mismatch": [] if gone else ["actor still answers after RemoveFromScene"]}


def place_device(T, label, asset, x):
    """PlaceDevice, labelled and read back; the rotation re-sent when the device ignored it (measured: 6 of 8). Through
    the listener its own place_device runs, which adopts a device PlaceDevice raised for yet placed."""
    if hasattr(T, "typed"):
        res = T.typed("place_device", {"label": label, "asset": asset, "xform": x})
        if not res.get("path"):
            raise StepError("error", "place_device placed nothing for %s" % label)
        return res
    try:
        path = refpath(T.call(DEVICES + ".PlaceDevice", {"assetPath": ref(asset), "transform": x}))
        placed_error = None
    except StepError as e:
        if e.kind != "error":
            raise
        # MEASURED in UEFN: PlaceDevice raises for several devices and places them anyway. Over Epic's MCP the new
        # actor's path cannot be read from the level (FindActors answers names), so it is named, not adopted.
        rows = T.read(LEVEL + ".FindActors", {}) or []
        raise StepError("error", "PlaceDevice raised (%s); the level now holds %d actors - over this transport the "
                        "path of one it placed anyway cannot be read, so look for it in UEFN's outliner (%s)"
                        % (str(e)[:200], len(rows), asset))
    if not path:
        raise StepError("error", "PlaceDevice returned no actor for %s" % asset)
    T.call(ACTOR + ".SetLabel", {"actor": ref(path), "label": label})
    st, first, bad = _settle(T, path, label, x)
    return {"path": path, "readback": st, "mismatch": bad, "correctedAfterPlace": first or None,
            "placedDespiteError": placed_error, "leftInLevel": [path] if bad else None}


def _d_place_device(T, a):
    return place_device(T, _text(a, "label"), _text(a, "asset"), whole_xform(a.get("xform")))


def _d_get_options(T, a):
    dev = _text(a, "device")
    want = a.get("options")
    if want is not None and (not isinstance(want, list) or not all(isinstance(w, str) for w in want)):
        raise StepError("refused", "options must be a list of option names")
    opts = T.read(DEVICES + ".ListClassicDeviceOptions", {"devicePath": ref(dev)}) or []
    rows = [o for o in opts if not want or o.get("name") in want or o.get("displayName") in want]
    missing = [w for w in want or [] if not any(w in (o.get("name"), o.get("displayName")) for o in opts)]
    return {"device": dev, "options": rows, "notFound": missing or None}


def _option_row(T, dev, name):
    opts = T.read(DEVICES + ".ListClassicDeviceOptions", {"devicePath": ref(dev)}) or []
    return next((o for o in opts if isinstance(o, dict) and name in (o.get("name"), o.get("displayName"))), None)


def _d_set_option(T, a):
    dev, name = _text(a, "device"), _text(a, "option")
    value = a.get("value")
    if not isinstance(value, str):
        raise StepError("refused", "value must be a string (the option's value as its Details panel shows it)")
    T.call(DEVICES + ".SetClassicDeviceOption", {"devicePath": ref(dev), "optionName": name, "value": value})
    row = _option_row(T, dev, name)
    if row is None:
        raise StepError("error", "option %s is not listed on the device after the write" % name)
    return {"option": name, "readback": {"value": row.get("currentValue"), "label": row.get("currentLabel")},
            "mismatch": [] if _option_holds(row, value) else ["%s want %r got %r" % (name, value,
                                                                                    row.get("currentValue"))]}


def _verse_props(T, dev, names):
    return _json(T.read(DEVICES + ".GetDeviceProperties", {"device": ref(dev), "propertyNames": names}))


def _d_get_verse_properties(T, a):
    dev, props = _text(a, "device"), a.get("properties")
    if not isinstance(props, list) or not props or not all(isinstance(p, str) for p in props):
        raise StepError("refused", "properties must be a non-empty list of property names")
    return {"device": dev, "properties": _verse_props(T, dev, props)}


def _d_set_verse_property(T, a):
    dev, name = _text(a, "device"), _text(a, "property")
    if "value" not in a:
        raise StepError("refused", "value is required")
    T.call(DEVICES + ".SetDeviceProperty", {"device": ref(dev), "propertyName": name, "value": json.dumps(a["value"])})
    back = _verse_props(T, dev, [name])
    got = back.get(name) if isinstance(back, dict) else back
    return {"property": name, "readback": got,
            "mismatch": [] if same_value(a["value"], got) else ["%s want %r got %r" % (name, a["value"], got)]}


def binding_listed(rows, source, event, function):
    return any(isinstance(r, dict) and refpath(r.get("sourceDevicePath")) == source and r.get("sourceEvent") == event
               and r.get("targetFunction") == function for r in rows or [])


def _d_bind_event(T, a):
    for k in ("source", "event", "target", "function"):
        _text(a, k)
    remove = a.get("remove", False)
    if not isinstance(remove, bool):
        raise StepError("refused", "remove must be true or false")
    args = {"sourceDevicePath": ref(a["source"]), "sourceEvent": a["event"], "targetDevicePath": ref(a["target"]),
            "targetFunction": a["function"]}
    T.call(DEVICES + (".RemoveEventBinding" if remove else ".AddEventBinding"), args, destroy=remove)
    rows = T.read(DEVICES + ".ListEventBindings", {"devicePath": ref(a["target"])}) or []
    present = binding_listed(rows, a["source"], a["event"], a["function"])
    bad = [] if present != remove else ["binding %s on the target after the write" % ("still listed" if remove
                                                                                        else "not listed")]
    return {"removed" if remove else "bound": not bad, "targetBindings": rows, "mismatch": bad}


def _d_bindings(T, a):
    dev = _text(a, "device")
    return {"device": dev, "bindings": T.read(DEVICES + ".ListEventBindings", {"devicePath": ref(dev)}) or []}


def _allow_list(listed):
    return [ln.split(" ")[0].split("[")[0] for ln in str(listed).splitlines() if ln.strip()]


def _d_set_property(T, a):
    obj, name = _text(a, "object"), _text(a, "property")
    if "value" not in a:
        raise StepError("refused", "value is required")
    allowed = _allow_list(T.read(OBJECT + ".ListProperties", {"instance": ref(obj)}))
    key = name if name in allowed else name[:1].lower() + name[1:]
    if key not in allowed:
        raise StepError("refused", "%s is not on UEFN's allow-list for this object (not editable in UEFN's "
                        "interface, so it would fail validation). Editable here: %s" % (
                            name, ", ".join(n for n in allowed if not n.endswith("Component"))[:400]))
    T.call(OBJECT + ".SetProperties", {"instance": ref(obj), "values": json.dumps({key: a["value"]})})
    back = _json(T.read(OBJECT + ".GetProperties", {"instance": ref(obj), "properties": [key]}))
    got = back.get(key, back.get(name)) if isinstance(back, dict) else back
    return {"object": obj, "property": name, "readback": got,
            "mismatch": [] if same_value(a["value"], got) else ["%s want %r got %r" % (name, a["value"], got)]}


def _d_verse_build(T, a):
    path, content = _text(a, "path", required=False), a.get("content")
    if content is not None:
        if not isinstance(content, str):
            raise StepError("refused", "content must be a string")
        if not path or not path.endswith(".verse"):
            raise StepError("refused", "path must name a .verse file when content is given")
    elif path:
        raise StepError("refused", "path without content: pass content to write the file, or neither to build only")
    if content is not None:
        T.call(VERSE + ".WriteFile", {"path": path, "content": content, "bCreateIfMissing": True})
        back = T.read(VERSE + ".ReadFile", {"path": path})
        if isinstance(back, str) and back.rstrip("\n") != content.rstrip("\n"):
            return {"written": path, "mismatch": ["the file read back differs from the content sent"]}
    diags = T.call(VERSE + ".BuildAll", {})
    if not isinstance(diags, list):
        return {"written": path if content is not None else None, "clean": None, "errors": None, "diagnostics": None,
                "mismatch": ["BuildAll answered %s, not its list of diagnostics; the build's result is unknown"
                             % ("nothing" if diags is None else type(diags).__name__)]}
    out, cache = [], {}
    for d in diags:
        d = d if isinstance(d, dict) else {}
        fp, sp = d.get("filePath", ""), d.get("span") or {}
        line0 = sp.get("startLine", 0)
        if fp not in cache:
            try:
                cache[fp] = (T.read(VERSE + ".ReadFile", {"path": fp}) or "").splitlines()
            except StepError:
                cache[fp] = []
        src = cache[fp][line0] if 0 <= line0 < len(cache[fp]) else ""
        out.append({"severity": d.get("severity"), "code": d.get("code"), "file": fp, "line": line0 + 1,
                    "column": sp.get("startCharacter", 0) + 1, "message": (d.get("message") or "").strip(),
                    "source": src})
    errors = [d for d in out if d["severity"] == "Error"]
    return {"written": path if content is not None else None, "clean": not errors, "errors": len(errors),
            "diagnostics": out}


def _entity_rows(rows):
    return [{"entity": refpath(r.get("entity")), "name": r.get("displayName"), "class": refpath(r.get("entityClass"))}
            for r in rows or [] if isinstance(r, dict)]


def _d_find_entities(T, a):
    rows = _entity_rows(T.read(ENTITY + ".FindEntities", {"nameFilter": _text(a, "name_filter")}))
    return {"count": len(rows), "entities": rows}


def _d_create_prefab(T, a):
    folder, name = _text(a, "folder"), _text(a, "name")
    ents = a.get("entities")
    if ents is not None and (not isinstance(ents, list) or not ents or not all(isinstance(e, str) for e in ents)):
        raise StepError("refused", "entities must be a non-empty list of entity paths (uefn_find_entities lists them)")
    if ents:
        info = T.call(PREFAB + ".CreatePrefabFromEntities", {"entities": [ref(e) for e in ents], "folderPath": folder,
                                                             "assetName": name})
    else:
        info = T.call(PREFAB + ".CreatePrefab", {"folderPath": folder, "assetName": name})
    prefab = refpath((info or {}).get("prefab")) if isinstance(info, dict) else None
    if not prefab:
        raise StepError("error", "the prefab tool returned no prefab (answer %s)" % json.dumps(info, default=str)[:200])
    back = T.read(PREFAB + ".GetPrefab", {"prefab": ref(prefab)}) or {}
    bad = [] if refpath(back.get("prefab")) == prefab else ["GetPrefab does not find %s" % prefab]
    return {"prefab": prefab, "name": back.get("name"), "rootEntity": refpath(back.get("rootEntity")),
            "generatedClass": refpath(back.get("generatedClass")), "created": info, "saved": False, "mismatch": bad,
            "notRead": {"instances": "CreatePrefabFromEntities answers the prefab, not the instance that replaced the "
                                     "entities; uefn_prefab place makes an instance whose entity is answered"}
            if ents else None}


def _d_call(T, a):
    name = _text(a, "tool")
    args = a.get("args") if a.get("args") is not None else {}
    if not isinstance(args, dict):
        raise StepError("refused", "args must be an object")
    row = _row(name)
    short = row["name"]
    if row.get("destructive") and a.get("confirm") != short:
        raise StepError("refused", "%s needs confirm=%r: %s. Nothing was sent. Pass it only when the person asked for "
                        "exactly this" % (name, short, row["destructive"]), name)
    if a.get("confirm") is not None and not row.get("destructive"):
        raise StepError("refused", "confirm is only for destructive tools, and %s is not one; drop it" % name, name)
    args = json.loads(json.dumps(args))
    partial = []
    try:
        CAT.validate(args, row["inputSchema"], "args", partial_vectors=partial)
    except CAT.Invalid as e:
        raise StepError("refused", str(e), name)
    getter = row.get("getter")
    tools = CAT.load()["tools"]
    gschema = tools[getter]["inputSchema"] if getter else None
    gargs = {k: v for k, v in args.items() if k in ((gschema or {}).get("properties") or {})} if getter else None
    filled = []
    if partial:
        if not getter:
            raise StepError("refused", "%s is missing %s and %s has no getter to read the current value from; send "
                            "every component" % (partial[0][0], ", ".join(partial[0][1]), name), name)
        try:
            filled = CAT.fill_partial(args, partial, T.read(getter, gargs))
        except CAT.Invalid as e:
            raise StepError("refused", str(e), name)
    value = T.call(name, args, destroy=bool(row.get("destructive")))
    out = {"tool": name, "mode": row["mode"], "value": value, "filledFromCurrent": filled or None, "mismatch": []}
    if row["mode"] != "write":
        return out
    if getter:
        got = T.read(getter, gargs)
        compared, bad = CAT.compare_readback(args, gschema, got)
        out.update(readback=got, readbackThrough=getter, compared=compared, mismatch=bad,
                   verified=bool(compared) and not bad, verification="read back through %s" % getter)
    else:
        out.update(readback=None, verified=False,
                   verification="unverified: over Epic's MCP only a write's own getter reads it back (the listener's "
                                "object and content-path routes read the editor through the unreal API)")
    return out


DIRECT = {"find": _d_find, "actor": _d_actor, "place_mesh": _d_place_mesh, "set_xform": _d_set_xform,
          "delete_actor": _d_delete_actor, "place_device": _d_place_device, "get_options": _d_get_options,
          "set_option": _d_set_option, "get_verse_properties": _d_get_verse_properties,
          "set_verse_property": _d_set_verse_property, "bind_event": _d_bind_event, "bindings": _d_bindings,
          "set_property": _d_set_property, "verse_build": _d_verse_build, "find_entities": _d_find_entities,
          "create_prefab": _d_create_prefab, "call": _d_call}


def run_direct(T, op, args):
    """One listener op name, run as Epic tool calls: the same argument table, the same refusals."""
    if op in LISTENER_ONLY:
        raise StepError("refused", "%s needs MifBridge's listener in UEFN: %s. Nothing was sent; call uefn_install "
                        "(or set MIF_UEFN_TRANSPORT=listener)" % (op, LISTENER_ONLY[op]))
    if op not in DIRECT:
        raise StepError("refused", "unknown op %r" % op)
    unknown = sorted(set(args) - DIRECT_ARGS[op])
    if unknown:
        raise StepError("refused", "unknown argument %s for %s; accepted: %s" % (", ".join(unknown), op,
                                                                                 ", ".join(sorted(DIRECT_ARGS[op]))))
    return DIRECT[op](T, args)


# ---------------------------------------------------------------- devices from one description (item 2)
def device_catalog(T, name_filter="", device="", option="", label_filter=""):
    """The placeable devices (search), or one placed device's events, functions, options and Verse settings."""
    if device:
        out = {"device": device, "notRead": {}}
        try:
            b = T.read(DEVICES + ".GetBindingOptions", {"devicePath": ref(device)}) or {}
            out["events"], out["functions"] = b.get("sourceEvents") or [], b.get("targetFunctions") or []
        except StepError as e:
            out["events"] = out["functions"] = None
            out["notRead"]["bindingOptions"] = str(e)[:300]
        try:
            out["options"] = [{k: o.get(k) for k in ("name", "displayName", "currentValue", "currentLabel",
                                                     "allowedValues", "numAllowedValues", "minValue", "maxValue",
                                                     "maxLength", "bIsActive", "bHasNoOverride", "description")}
                              for o in T.read(DEVICES + ".ListClassicDeviceOptions", {"devicePath": ref(device)}) or []
                              if isinstance(o, dict)]
        except StepError as e:
            out["options"] = None
            out["notRead"]["classicOptions"] = str(e)[:300]
        try:
            schema = _json(T.read(DEVICES + ".ListDeviceProperties", {"device": ref(device)}))
            out["verseProperties"] = schema
        except StepError as e:
            out["verseProperties"] = None
            out["notRead"]["verseProperties"] = str(e)[:300]
        if option:
            out["optionValues"] = T.read(DEVICES + ".GetClassicDeviceOptionValues", {
                "devicePath": ref(device), "optionName": option, "labelFilter": label_filter or ""})
        out["notRead"] = out["notRead"] or None
        if out["events"] is None and out["options"] is None and out["verseProperties"] is None:
            out["mismatch"] = ["nothing about %s could be read: %s" % (device, out["notRead"])]
        return out
    rows = T.read(DEVICES + ".ListDeviceAssets", {"nameFilter": name_filter or ""}) or []
    found = [{"name": r.get("displayName"), "asset": refpath(r.get("assetPath")), "category": r.get("category"),
              "verse": r.get("bIsVerseDevice")} for r in rows if isinstance(r, dict)]
    out = {"count": len(found), "devices": found, "filter": name_filter or ""}
    if not found and not name_filter:
        out["mismatch"] = ["ListDeviceAssets listed no device at all: an empty catalog is not a search result"]
    return out


DEVICE_KEYS = {"label", "asset", "path", "xform", "options", "properties"}
WIRE_KEYS = {"source", "event", "target", "function"}


def parse_wire(w, i):
    if isinstance(w, str):
        m = re.fullmatch(r"\s*(.+)\.([A-Za-z_]\w*)\s*->\s*(.+)\.([A-Za-z_]\w*)\s*", w)
        if not m:
            raise StepError("refused", "wires[%d] %r is not 'source.Event -> target.Function'" % (i, w))
        return dict(zip(("source", "event", "target", "function"), m.groups()))
    if not isinstance(w, dict):
        raise StepError("refused", "wires[%d] must be 'source.Event -> target.Function' or an object" % i)
    bad = sorted(set(w) ^ WIRE_KEYS)
    if bad:
        raise StepError("refused", "wires[%d] needs exactly source, event, target, function (wrong: %s)" % (i, bad))
    for k in WIRE_KEYS:
        if not isinstance(w[k], str) or not w[k]:
            raise StepError("refused", "wires[%d].%s must be a non-empty string" % (i, k))
    return dict(w)


def plan_wiring(devices, wires):
    """The description checked for shape before anything is read: one row per device, the wires by label."""
    if not isinstance(devices, list) or not devices:
        raise StepError("refused", "devices must be a non-empty list")
    plan, seen = [], set()
    for i, d in enumerate(devices):
        if not isinstance(d, dict):
            raise StepError("refused", "devices[%d] must be an object" % i)
        unknown = sorted(set(d) - DEVICE_KEYS)
        if unknown:
            raise StepError("refused", "devices[%d]: unknown key %s (accepted: %s)" % (i, unknown,
                                                                                     ", ".join(sorted(DEVICE_KEYS))))
        label = d.get("label")
        if not isinstance(label, str) or not label.strip():
            raise StepError("refused", "devices[%d].label must be a non-empty string" % i)
        if label in seen:
            raise StepError("refused", "two devices are labelled %r; each label names one device" % label)
        seen.add(label)
        if bool(d.get("asset")) == bool(d.get("path")):
            raise StepError("refused", "devices[%d] (%s) needs asset (place a new one) or path (one already placed), "
                            "not both" % (i, label))
        row = {"label": label, "asset": d.get("asset"), "path": d.get("path"), "new": bool(d.get("asset"))}
        if row["new"]:
            row["xform"] = whole_xform(d.get("xform"), "devices[%d].xform" % i)
        elif d.get("xform") is not None:
            raise StepError("refused", "devices[%d] (%s) is already placed: move it with uefn_move_actor" % (i, label))
        opts = d.get("options") or {}
        if not isinstance(opts, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in opts.items()):
            raise StepError("refused", "devices[%d].options must map option names to string values (as the Details "
                            "panel shows them)" % i)
        props = d.get("properties") or {}
        if not isinstance(props, dict):
            raise StepError("refused", "devices[%d].properties must map Verse setting names to JSON values" % i)
        row["options"], row["properties"] = dict(opts), dict(props)
        plan.append(row)
    labels = {r["label"] for r in plan}
    parsed = [parse_wire(w, i) for i, w in enumerate(wires or [])]
    for i, w in enumerate(parsed):
        for side in ("source", "target"):
            if w[side] not in labels:
                raise StepError("refused", "wires[%d].%s %r is not a label in devices" % (i, side, w[side]))
    return plan, parsed


def _option_problem(row, value):
    if row is None:
        return "no such option"
    if row.get("bIsActive") is False:
        return "the option is not active with the device's other settings"
    allowed = row.get("allowedValues") or []
    if allowed:
        if not any(value == v.get("value") or value.lower() == str(v.get("label", "")).lower() for v in allowed
                   if isinstance(v, dict)):
            return "%r is not one of %s" % (value, ", ".join(repr(v.get("value")) for v in allowed[:12]
                                                             if isinstance(v, dict)))
        return None
    if value == "" and row.get("bHasNoOverride"):
        return None
    lo, hi = row.get("minValue") or "", row.get("maxValue") or ""
    if lo or hi:
        try:
            f = float(value)
        except ValueError:
            return "%r is not a number (the option takes %s..%s)" % (value, lo or "-inf", hi or "inf")
        if (lo and f < float(lo)) or (hi and f > float(hi)):
            return "%r is outside %s..%s" % (value, lo or "-inf", hi or "inf")
    if row.get("maxLength") and len(value) > int(row["maxLength"]):
        return "%r is longer than %s characters" % (value, row["maxLength"])
    if not allowed and (row.get("numAllowedValues") or 0) > 0:
        return None           # left out of the inline list; SetClassicDeviceOption refuses one it does not accept
    return None


def _verse_names(schema):
    props = schema.get("properties") if isinstance(schema, dict) else None
    return set(props) if isinstance(props, dict) else None


def _describe(T, path):
    """What a placed device offers, read for validation: events, functions, options, Verse setting names."""
    info = device_catalog(T, device=path)
    return {"events": set(info.get("events") or []), "functions": set(info.get("functions") or []),
            "options": {o["name"]: o for o in info.get("options") or [] if o.get("name")},
            "optionsByDisplay": {o["displayName"]: o for o in info.get("options") or [] if o.get("displayName")},
            "verse": _verse_names(info.get("verseProperties")), "raw": info}


def validate_against(plan, wires, described):
    """Every option, Verse setting and wire checked against what the devices say they offer."""
    bad = []
    for r in plan:
        d = described.get(r["label"])
        if d is None:
            continue
        for name, value in r["options"].items():
            why = _option_problem(d["options"].get(name) or d["optionsByDisplay"].get(name), value)
            if why:
                bad.append("%s option %s: %s" % (r["label"], name, why))
        if r["properties"] and d["verse"] is not None:
            for name in r["properties"]:
                if name not in d["verse"]:
                    bad.append("%s has no Verse setting %s (it has %s)" % (r["label"], name,
                                                                         ", ".join(sorted(d["verse"]))[:200]))
        elif r["properties"] and d["verse"] is None:
            bad.append("%s: its Verse settings could not be read, so %s cannot be checked" % (
                r["label"], ", ".join(r["properties"])))
    for w in wires:
        s, t = described.get(w["source"]), described.get(w["target"])
        if s is not None and w["event"] not in s["events"]:
            bad.append("%s has no event %s (it has %s)" % (w["source"], w["event"], ", ".join(sorted(s["events"]))[:200]))
        if t is not None and w["function"] not in t["functions"]:
            bad.append("%s has no function %s (it has %s)" % (w["target"], w["function"],
                                                              ", ".join(sorted(t["functions"]))[:200]))
    return bad


def wire_devices(T, devices, wires=None, dry_run=False):
    """Place, set and wire a set of devices from one description, then read all of it back as one table.

    Order: the description's shape, the catalog's assets, the level's labels and the already-placed devices'
    offers are checked first (nothing written); then the new devices are placed and their offers read; every option,
    Verse setting and wire is checked against them BEFORE the first option or wire is written, and on a refusal the
    devices this call placed are removed again. Then options, Verse settings, wires, and the read-back."""
    plan, wires = plan_wiring(devices, wires)
    catalog = {}
    if any(r["new"] for r in plan):
        catalog = {refpath(x.get("assetPath")): x.get("displayName")
                   for x in T.read(DEVICES + ".ListDeviceAssets", {"nameFilter": ""}) or [] if isinstance(x, dict)}
        if not catalog:
            raise StepError("refused", "ListDeviceAssets listed no device, so no asset can be checked; nothing placed")
    for r in plan:
        if r["new"] and r["asset"] not in catalog:
            word = r["asset"].rsplit("/", 1)[-1].split(".")[0].lower().replace("pid_device_", "")
            near = sorted(a for a in catalog if a and word and word in a.lower())[:5]
            raise StepError("refused", "%s: %s is not in the project's device catalog (ListDeviceAssets)%s; nothing "
                            "was placed" % (r["label"], r["asset"], "; near: " + ", ".join(near) if near else ""))
    taken = []
    for r in plan:
        if r["new"]:
            rows = T.read(LEVEL + ".FindActors", {"filter": {"name": r["label"]}}) or []
            if any(isinstance(x, dict) and x.get("label") == r["label"] for x in rows):
                taken.append(r["label"])
    if taken:
        raise StepError("refused", "the level already holds an actor labelled %s; a second one makes the read-back "
                        "ambiguous - pick new labels or name the placed one by path. Nothing was placed"
                        % ", ".join(taken))
    described = {r["label"]: _describe(T, r["path"]) for r in plan if not r["new"]}
    bad = validate_against(plan, wires, described)
    if bad:
        raise StepError("refused", "the description does not match what the placed devices offer: %s. Nothing was "
                        "written" % "; ".join(bad))
    order = ["place " + r["label"] for r in plan if r["new"]] + \
            ["option %s.%s" % (r["label"], k) for r in plan for k in r["options"]] + \
            ["property %s.%s" % (r["label"], k) for r in plan for k in r["properties"]] + \
            ["wire %s.%s -> %s.%s" % (w["source"], w["event"], w["target"], w["function"]) for w in wires]
    if dry_run:
        return {"dryRun": True, "order": order, "devices": [{k: r[k] for k in ("label", "asset", "path", "new")}
                                                           for r in plan], "wires": wires, "mismatch": []}
    placed, paths, place_rows = [], {r["label"]: r["path"] for r in plan if not r["new"]}, {}
    try:
        for r in plan:
            if r["new"]:
                res = place_device(T, r["label"], r["asset"], r["xform"])
                paths[r["label"]] = res["path"]
                placed.append(res["path"])
                place_rows[r["label"]] = res
        described.update({r["label"]: _describe(T, paths[r["label"]]) for r in plan if r["new"]})
        bad = validate_against(plan, wires, described)
        if bad:
            raise StepError("refused", "after placing, the devices do not offer what the description names: %s. No "
                            "option or wire was written" % "; ".join(bad))
    except StepError as e:
        removed = _remove_placed(T, placed)
        out = e.reply()
        out.update(inLevel=[p for p in placed if p not in removed], removedAgain=removed or None,
                   notInLevel=[r["label"] for r in plan if r["new"]], order=order)
        out["mismatch"] = [str(e)]
        return out
    for r in plan:
        for name, value in r["options"].items():
            T.call(DEVICES + ".SetClassicDeviceOption", {"devicePath": ref(paths[r["label"]]), "optionName": name,
                                                         "value": value})
        for name, value in r["properties"].items():
            T.call(DEVICES + ".SetDeviceProperty", {"device": ref(paths[r["label"]]), "propertyName": name,
                                                    "value": json.dumps(value)})
    for w in wires:
        T.call(DEVICES + ".AddEventBinding", {"sourceDevicePath": ref(paths[w["source"]]), "sourceEvent": w["event"],
                                              "targetDevicePath": ref(paths[w["target"]]),
                                              "targetFunction": w["function"]})
    return read_wiring(T, plan, wires, paths, place_rows, order)


def _remove_placed(T, placed):
    removed = []
    for p in placed:
        if hasattr(T, "typed"):
            try:
                if T.typed("delete_actor", {"path": p}).get("inLevelAfter") is False:
                    removed.append(p)
            except StepError:
                pass
            continue
        try:
            T.call(LEVEL + ".RemoveFromScene", {"actor": ref(p)}, destroy=True)
            if _gone(T, ACTOR + ".GetLabel", {"actor": ref(p)}):
                removed.append(p)
        except StepError:
            pass
    return removed


def read_wiring(T, plan, wires, paths, place_rows, order):
    """The whole description read back from the level: one row per device and one per wire."""
    rows, bad = [], []
    for r in plan:
        p = paths[r["label"]]
        row = {"label": r["label"], "path": p, "new": r["new"], "mismatch": []}
        st = actor_read(T, p)
        row["labelRead"], row["xform"] = st["label"], st["xform"]
        if st["label"] != r["label"]:
            row["mismatch"].append("label want %r got %r" % (r["label"], st["label"]))
        if r["new"]:
            row["mismatch"] += compare_xform(r["xform"], st["xform"])
            row["correctedAfterPlace"] = (place_rows.get(r["label"]) or {}).get("correctedAfterPlace")
        if r["options"]:
            listed = {o.get("name"): o for o in T.read(DEVICES + ".ListClassicDeviceOptions",
                                                       {"devicePath": ref(p)}) or [] if isinstance(o, dict)}
            listed.update({o.get("displayName"): o for o in list(listed.values())})
            row["options"] = {}
            for name, value in r["options"].items():
                o = listed.get(name)
                held = o is not None and _option_holds(o, value)
                row["options"][name] = {"want": value, "got": o.get("currentValue") if o else None, "ok": held}
                if not held:
                    row["mismatch"].append("option %s want %r got %r" % (name, value, o and o.get("currentValue")))
        if r["properties"]:
            back = _verse_props(T, p, list(r["properties"]))
            row["properties"] = {}
            for name, value in r["properties"].items():
                got = back.get(name) if isinstance(back, dict) else None
                held = same_value(value, got)
                row["properties"][name] = {"want": value, "got": got, "ok": held}
                if not held:
                    row["mismatch"].append("property %s want %r got %r" % (name, value, got))
        bad += ["%s: %s" % (r["label"], m) for m in row["mismatch"]]
        rows.append(row)
    wire_rows, by_target = [], {}
    for w in wires:
        tp = paths[w["target"]]
        if tp not in by_target:
            by_target[tp] = T.read(DEVICES + ".ListEventBindings", {"devicePath": ref(tp)}) or []
        listed = binding_listed(by_target[tp], paths[w["source"]], w["event"], w["function"])
        wire_rows.append(dict(w, listed=listed))
        if not listed:
            bad.append("wire %s.%s -> %s.%s is not listed on %s" % (w["source"], w["event"], w["target"],
                                                                     w["function"], w["target"]))
    return {"devices": rows, "wires": wire_rows, "order": order, "mismatch": bad,
            "inLevel": [paths[r["label"]] for r in plan], "notInLevel": [], "saved": False}


# ---------------------------------------------------------------- Scene Graph (item 3)
def _class_ref(T, given, tool, what):
    """A class by path ('/...') or by the friendly name its list answers; refused unsent when it is not listed."""
    if not isinstance(given, str) or not given:
        raise StepError("refused", "%s is required" % what)
    rows = [r for r in T.read(tool, {"nameFilter": "" if given.startswith("/") else given}) or []
            if isinstance(r, dict)]
    for r in rows:
        if refpath(r.get("classPath")) == given or r.get("className") == given:
            return refpath(r.get("classPath"))
    near = [r.get("className") for r in rows][:10]
    raise StepError("refused", "%s %r is not a class UEFN offers here (%s lists %s); nothing was sent" % (
        what, given, tool.rsplit(".", 1)[-1], ", ".join(map(str, near)) or "none matching"))


def _entity_info(v):
    v = v if isinstance(v, dict) else {}
    return {"entity": refpath(v.get("entity")), "name": v.get("displayName"), "class": refpath(v.get("entityClass"))}


def _children(T, parent):
    args = {"bRecursive": False}
    if parent:
        args["rootEntity"] = ref(parent)
    return {r["entity"] for r in _entity_rows(T.read(ENTITY + ".FindEntities", args))}


ENTITY_ACTIONS = ("classes", "get", "create", "rename", "parent", "transform", "delete")


def entity(T, action, entity="", entity_class="", name="", parent="", xform=None, new_name="", name_filter=""):
    act = (action or "").strip().lower()
    if act not in ENTITY_ACTIONS:
        raise StepError("refused", "action must be one of %s" % ", ".join(ENTITY_ACTIONS))
    if act == "classes":
        rows = T.read(ENTITY + ".ListEntityClasses", {"nameFilter": name_filter or ""}) or []
        return {"classes": [{"class": refpath(r.get("classPath")), "name": r.get("className"),
                             "prefab": r.get("bIsPrefab")} for r in rows if isinstance(r, dict)]}
    if act == "create":
        cls = _class_ref(T, entity_class, ENTITY + ".ListEntityClasses", "entity_class")
        x = whole_xform(xform)
        args = {"entityClass": ref(cls), "transform": x, "name": name or ""}
        if parent:
            args["parentEntity"] = ref(parent)
        info = _entity_info(T.call(ENTITY + ".CreateEntity", args))
        return dict(_entity_readback(T, info, x, parent or None), created=info)
    path = entity
    if not isinstance(path, str) or not path:
        raise StepError("refused", "entity (its path, from uefn_find_entities) is required")
    if act == "get":
        out = {"entity": path, "xform": T.read(ENTITY + ".GetEntityTransform", {"entity": ref(path)}),
               "components": _component_rows(T, path)}
        try:
            out["prefab"] = refpath((T.read(PREFAB + ".GetPrefabForEntity", {"entity": ref(path)}) or {}).get("prefab"))
        except StepError as e:
            if e.kind != "error":
                raise
            out["prefab"] = None          # "Raises if it comes from none"
        return out
    if act == "rename":
        if not isinstance(new_name, str) or not new_name:
            raise StepError("refused", "new_name is required")
        info = _entity_info(T.call(ENTITY + ".RenameEntity", {"entity": ref(path), "newName": new_name}))
        rows = _entity_rows(T.read(ENTITY + ".FindEntities", {"nameFilter": new_name}))
        mine = next((r for r in rows if r["entity"] == (info["entity"] or path)), None)
        bad = [] if mine and str(mine["name"]).startswith(new_name) else [
            "FindEntities does not list %s under the name %s" % (path, new_name)]
        return {"entity": info["entity"] or path, "name": mine and mine["name"], "renamed": info, "mismatch": bad}
    if act == "parent":
        before = T.read(ENTITY + ".GetEntityTransform", {"entity": ref(path)})
        args = {"entity": ref(path)}
        if parent:
            args["newParent"] = ref(parent)
        info = _entity_info(T.call(ENTITY + ".ReparentEntity", args))
        moved = info["entity"] or path
        bad = [] if moved in _children(T, parent or None) else [
            "%s is not listed under %s after the move" % (moved, parent or "its hierarchy's root")]
        keep = read_xform(before)
        if keep:
            bad += ["world transform kept: " + m for m in compare_xform(keep, T.read(ENTITY + ".GetEntityTransform",
                                                                                      {"entity": ref(moved)}))]
        return {"entity": moved, "parent": parent or None, "moved": info, "mismatch": bad}
    if act == "transform":
        cur = T.read(ENTITY + ".GetEntityTransform", {"entity": ref(path)})
        # UN5: SetEntityTransform resets an omitted rotation, so what is left out is merged from the current one.
        x, filled = merge_xform(xform, cur)
        T.call(ENTITY + ".SetEntityTransform", {"entity": ref(path), "transform": x})
        got = T.read(ENTITY + ".GetEntityTransform", {"entity": ref(path)})
        return {"entity": path, "xform": read_xform(got) or got, "filledFromCurrent": filled or None,
                "mismatch": compare_xform(x, got)}
    # delete
    T.call(ENTITY + ".DeleteEntity", {"entity": ref(path)}, destroy=True)
    gone = _gone(T, ENTITY + ".GetEntityTransform", {"entity": ref(path)})
    return {"entity": path, "deleted": gone, "mismatch": [] if gone else ["the entity still answers after DeleteEntity"]}


def _entity_readback(T, info, x, parent):
    path = info.get("entity")
    if not path:
        return {"mismatch": ["CreateEntity answered no entity"]}
    got = T.read(ENTITY + ".GetEntityTransform", {"entity": ref(path)})
    bad = compare_xform(x, got)
    if path not in _children(T, parent):
        bad.append("%s is not listed under %s" % (path, parent or "the level root"))
    return {"entity": path, "name": info.get("name"), "class": info.get("class"), "xform": read_xform(got) or got,
            "parent": parent, "mismatch": bad}


def _component_rows(T, ent):
    return [{"component": refpath(r.get("component")), "name": r.get("componentName"),
             "class": refpath(r.get("componentClass"))}
            for r in T.read(ENTITY + ".GetComponents", {"entity": ref(ent)}) or [] if isinstance(r, dict)]


def _component(T, ent, component):
    if isinstance(component, str) and component.startswith("/"):
        return component
    if not ent or not component:
        raise StepError("refused", "component needs its path, or entity and the component's Verse name")
    rows = _component_rows(T, ent)
    hit = [r for r in rows if r["name"] == component or (r["class"] or "").rsplit(".", 1)[-1] == component]
    if len(hit) != 1:
        raise StepError("refused", "%s has %s component named %s (it has %s)" % (
            ent, "no" if not hit else "more than one", component, ", ".join(str(r["name"]) for r in rows) or "none"))
    return hit[0]["component"]


COMPONENT_ACTIONS = ("classes", "list", "add", "remove", "get", "set")


def entity_components(T, action, entity="", component="", component_class="", property="", value=None,
                      name_filter="", with_properties=False, has_value=False):
    act = (action or "").strip().lower()
    component_class = component_class or ""
    if act not in COMPONENT_ACTIONS:
        raise StepError("refused", "action must be one of %s" % ", ".join(COMPONENT_ACTIONS))
    if act == "classes":
        rows = T.read(ENTITY + ".ListComponentClasses", {"nameFilter": name_filter or ""}) or []
        return {"classes": [{"class": refpath(r.get("classPath")), "name": r.get("className"),
                             "prefab": r.get("bIsPrefab")} for r in rows if isinstance(r, dict)]}
    if act == "list":
        if not entity:
            raise StepError("refused", "entity is required")
        rows = _component_rows(T, entity)
        if with_properties:
            for r in rows:
                r["properties"] = T.read(ENTITY + ".ListComponentProperties", {"component": ref(r["component"])})
        return {"entity": entity, "components": rows}
    if act in ("add", "remove"):
        if not entity:
            raise StepError("refused", "entity is required")
        cls = _class_ref(T, component_class, ENTITY + ".ListComponentClasses", "component_class") if act == "add" \
            else (component_class if component_class.startswith("/") else
                  next((r["class"] for r in _component_rows(T, entity) if r["name"] == component_class
                        or (r["class"] or "").rsplit(".", 1)[-1] == component_class), None))
        if not cls:
            raise StepError("refused", "%s has no component %r to remove" % (entity, component_class))
        if act == "add":
            made = T.call(ENTITY + ".AddComponent", {"entity": ref(entity), "componentClass": ref(cls)})
        else:
            made = T.call(ENTITY + ".RemoveComponent", {"entity": ref(entity), "componentClass": ref(cls)},
                          destroy=True)
        rows = _component_rows(T, entity)
        present = any(r["class"] == cls for r in rows)
        bad = [] if present == (act == "add") else ["%s %s listed on %s after the %s" % (
            cls, "not" if act == "add" else "still", entity, act)]
        return {"entity": entity, "componentClass": cls, "answer": made, "components": rows, "mismatch": bad}
    comp = _component(T, entity, component)
    if not isinstance(property, str) or not property:
        raise StepError("refused", "property (its friendly Verse name) is required")
    if act == "get":
        return {"component": comp, "property": property,
                "value": _json(T.read(ENTITY + ".GetComponentProperty", {"component": ref(comp),
                                                                         "propertyName": property}))}
    if not has_value:
        raise StepError("refused", "value is required")
    listed = [r for r in T.read(ENTITY + ".ListComponentProperties", {"component": ref(comp)}) or []
              if isinstance(r, dict)]
    row = next((r for r in listed if r.get("name") == property), None)
    if row is None:
        raise StepError("refused", "%s has no editable property %s (it has %s); nothing was sent" % (
            comp, property, ", ".join(str(r.get("name")) for r in listed)[:300] or "none"))
    if row.get("bReadOnly"):
        raise StepError("refused", "%s.%s is read-only%s; nothing was sent" % (
            comp, property, " (a transform is set on the entity: uefn_entity transform)" if "transform" in
            str(comp).lower() else ""))
    T.call(ENTITY + ".SetComponentProperty", {"component": ref(comp), "propertyName": property,
                                              "value": json.dumps(value)})
    got = _json(T.read(ENTITY + ".GetComponentProperty", {"component": ref(comp), "propertyName": property}))
    return {"component": comp, "property": property, "type": row.get("type"), "before": _json(row.get("value")),
            "readback": got, "mismatch": [] if same_value(value, got) else ["%s want %r got %r" % (property, value,
                                                                                                    got)]}


PREFAB_ACTIONS = ("get", "source", "place", "overrides", "reset", "break", "apply")
PREFAB_CONFIRM = {"reset": "it discards the instance's local changes: added entities and components are removed",
                  "break": "the instance stops following its prefab, and the entity is recreated",
                  "apply": "ApplyPrefabEdits recompiles the prefab into every placed instance and is not undoable"}


def _overrides(T, ent):
    return [{"object": refpath(r.get("object")), "property": r.get("propertyName"), "change": r.get("change"),
             "subObject": refpath(r.get("subObject"))}
            for r in T.read(PREFAB + ".GetPrefabOverrides", {"entity": ref(ent)}) or [] if isinstance(r, dict)]


def prefab(T, action, prefab="", entity="", xform=None, name="", parent="", confirm=""):
    act = (action or "").strip().lower()
    if act not in PREFAB_ACTIONS:
        raise StepError("refused", "action must be one of %s" % ", ".join(PREFAB_ACTIONS))
    if act in PREFAB_CONFIRM and confirm != act:
        raise StepError("refused", "uefn_prefab %s needs confirm=%r: %s. Nothing was sent. Pass it only when the "
                        "person asked for exactly this" % (act, act, PREFAB_CONFIRM[act]))
    if act in ("get", "place", "apply"):
        if not isinstance(prefab, str) or not prefab:
            raise StepError("refused", "prefab (the prefab asset's path) is required")
        info = T.read(PREFAB + ".GetPrefab", {"prefab": ref(prefab)}) or {}
        cls = refpath(info.get("generatedClass"))
        if act == "get":
            return {"prefab": refpath(info.get("prefab")), "name": info.get("name"), "generatedClass": cls,
                    "parentClass": refpath(info.get("parentClass")), "rootEntity": refpath(info.get("rootEntity"))}
        if act == "apply":
            T.call(PREFAB + ".ApplyPrefabEdits", {"prefab": ref(prefab)}, destroy=True)
            back = T.read(PREFAB + ".GetPrefab", {"prefab": ref(prefab)}) or {}
            return {"prefab": prefab, "applied": True, "stillThere": refpath(back.get("prefab")) == prefab,
                    "verified": False, "saved": False,
                    "verification": "nothing reads a recompile back; the prefab is read again and is still there",
                    "mismatch": [] if refpath(back.get("prefab")) == prefab else ["GetPrefab no longer finds it"]}
        if not cls:
            raise StepError("refused", "GetPrefab answered no generatedClass for %s, so no instance can be placed"
                            % prefab)
        x = whole_xform(xform)
        args = {"entityClass": ref(cls), "transform": x, "name": name or ""}
        if parent:
            args["parentEntity"] = ref(parent)
        made = _entity_info(T.call(ENTITY + ".CreateEntity", args))
        out = _entity_readback(T, made, x, parent or None)
        if made.get("entity"):
            src = refpath((T.read(PREFAB + ".GetPrefabForEntity", {"entity": ref(made["entity"])}) or {})
                          .get("prefab"))
            out["prefabOf"] = src
            if src != refpath(info.get("prefab")):
                out["mismatch"].append("the new entity reads back as from %s, not %s" % (src, prefab))
        return dict(out, prefab=prefab, created=made)
    if not isinstance(entity, str) or not entity:
        raise StepError("refused", "entity (the instance's root entity) is required")
    if act == "source":
        return {"entity": entity, "prefab": refpath((T.read(PREFAB + ".GetPrefabForEntity",
                                                            {"entity": ref(entity)}) or {}).get("prefab"))}
    if act == "overrides":
        rows = _overrides(T, entity)
        return {"entity": entity, "overrides": rows, "count": len(rows)}
    if act == "reset":
        before = _overrides(T, entity)
        T.call(PREFAB + ".ResetPrefabInstance", {"entity": ref(entity)}, destroy=True)
        after = _overrides(T, entity)
        return {"entity": entity, "overridesBefore": before, "overridesAfter": after,
                "mismatch": [] if not after else ["%d override(s) still listed after the reset" % len(after)]}
    made = _entity_info(T.call(PREFAB + ".BreakPrefabInstance", {"entity": ref(entity)}, destroy=True))
    bad = []
    if not made.get("entity"):
        bad.append("BreakPrefabInstance answered no entity")
    else:
        try:
            still = refpath((T.read(PREFAB + ".GetPrefabForEntity", {"entity": ref(made["entity"])}) or {})
                            .get("prefab"))
            bad.append("the plain entity still reads back as an instance of %s" % still)
        except StepError as e:
            if e.kind != "error":
                raise
    return {"entity": made.get("entity"), "replaced": entity, "broken": made, "mismatch": bad}


# ---------------------------------------------------------------- the island readiness report (item 4)
# Each source is Epic's page and its own sentence - the same ones MifBlender's uefn_check cites
# (tools/blender-addon/MifBlender/ops_handoff.py _UEFN_SOURCES, read 2026-11-16).
SOURCES = {
    "memory": ("Epic, Memory Management in UEFN",
               "dev.epicgames.com/documentation/en-us/fortnite/memory-management-in-unreal-editor-for-fortnite",
               "You cannot publish your island if it exceeds the 300 MiB publishing memory budget."),
    "simplify": ("Epic, Simplify Static Meshes in Fortnite",
                 "dev.epicgames.com/documentation/en-us/fortnite/simplify-static-meshes-in-fortnite",
                 "there's a rule that warns you if the mesh has an LOD0 and is over 30,000 vertices."),
    "singleLod": ("Epic, Simplify Static Meshes in Fortnite",
                  "dev.epicgames.com/documentation/en-us/fortnite/simplify-static-meshes-in-fortnite",
                  "The static mesh validation reports any static mesh that has 500 vertices or more with only a single "
                  "LOD"),
    "texturePow2": ("Epic, Creating Fortnite-Ready Assets in UEFN",
                    "dev.epicgames.com/documentation/fortnite/fortniteready-assets-best-practices-in-fortnite",
                    "Use power-of-2 scaling for texture maps (64, 128, 256, 512, 1024, 2048). This makes it possible "
                    "to do mipmapping."),
    "textureSize": ("Epic, Creating Fortnite-Ready Assets in UEFN",
                    "dev.epicgames.com/documentation/fortnite/fortniteready-assets-best-practices-in-fortnite",
                    "All texture maps should have a size/resolution of 2k or less."),
}
LOD0_VERTEX_WARN = 30000
SINGLE_LOD_VERTICES = 500
TEXTURE_MAX = 2048
VALIDATE_BATCH = 100


def _cite(key):
    t, u, s = SOURCES[key]
    return {"page": t, "url": "https://" + u, "says": s}


def _package(p):
    return p.split(".", 1)[0] if isinstance(p, str) else p


def _pow2(n):
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def island_check(T, folder="", max_assets=2000, max_meshes=300, max_textures=300):
    """One table before the person publishes: asset validation, LOD0 vertices, single-LOD meshes, texture streaming
    and size, the memory lines in the editor's log, and what only the editor decides (notChecked). Read only: every
    step goes through T.read, which refuses a write before it is sent."""
    for k, v in (("max_assets", max_assets), ("max_meshes", max_meshes), ("max_textures", max_textures)):
        if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 20000:
            raise StepError("refused", "%s must be an integer 1-20000" % k)
    root = (folder or "").rstrip("/") or T.content_root()
    if not root:
        raise StepError("refused", "the project's content root could not be read (GetAssetContentRoots answered no "
                        "single writable root); pass folder")
    t0 = time.perf_counter()
    checks = []

    def find(asset_type=None, limit=max_assets):
        f = {"folderPath": root, "bRecursive": True, "maxResults": limit}
        if asset_type:
            f["assetType"] = ref(asset_type)
        return [p for p in T.read(ASSET + ".FindAssets", {"filter": f}) or [] if isinstance(p, str)]

    assets = find()
    if not assets:
        checks.append({"check": "scan", "status": "fail", "count": 0, "detail": "FindAssets found no saved asset "
                       "under %s: a scan that found nothing checked nothing" % root})
    # 1. Epic's own validator, in batches; fewer checked than sent is a gap, never a pass.
    if assets:
        sent = sorted({_package(p) for p in assets})
        tot = {"numChecked": 0, "numValid": 0, "numInvalid": 0, "numWarnings": 0, "numSkipped": 0}
        errors, warnings, limit_hit, unread = [], [], False, []
        for i in range(0, len(sent), VALIDATE_BATCH):
            batch = sent[i:i + VALIDATE_BATCH]
            v = T.read(ASSET + ".ValidateAssets", {"assetPaths": batch})
            if not isinstance(v, dict) or isinstance(v.get("numChecked"), bool) or not isinstance(v.get("numChecked"),
                                                                                                  int):
                unread.append({"batch": i // VALIDATE_BATCH, "answer": json.dumps(v, default=str)[:300]})
                continue
            for k in tot:
                tot[k] += v.get(k) if isinstance(v.get(k), int) and not isinstance(v.get(k), bool) else 0
            errors += [e for e in v.get("errors") or [] if isinstance(e, dict)]
            warnings += [w for w in v.get("warnings") or [] if isinstance(w, dict)]
            limit_hit = limit_hit or bool(v.get("bAssetLimitReached"))
        status = "fail" if errors or tot["numInvalid"] or unread or tot["numChecked"] < len(sent) else \
            "warn" if warnings or limit_hit or tot["numSkipped"] else "pass"
        checks.append({"check": "assetValidation", "status": status, "sent": len(sent), **tot,
                       "assetLimitReached": limit_hit or None, "unreadBatches": unread or None,
                       "errors": errors[:50], "warnings": warnings[:50],
                       "assets": sorted({e.get("assetPath") for e in errors + warnings if e.get("assetPath")})[:50],
                       "source": {"tool": ASSET + ".ValidateAssets", "says": "Epic's own content validation"}})
        if len(assets) >= max_assets:
            checks.append({"check": "scanCapped", "status": "warn", "count": len(assets),
                           "detail": "FindAssets stopped at max_assets=%d; raise it to scan everything" % max_assets})
    # 2. static meshes: LOD0 vertices and single-LOD meshes.
    meshes = find("/Script/Engine.StaticMesh", max_meshes) if assets else []
    heavy, single, mesh_unread = [], [], []
    for m in meshes:
        try:
            v = T.read(STATIC + ".GetVertexCount", {"mesh": ref(m), "lodIndex": 0})
            lods = T.read(STATIC + ".GetLodCount", {"mesh": ref(m)})
            nanite = T.read(STATIC + ".IsNaniteMesh", {"mesh": ref(m)})
        except StepError as e:
            mesh_unread.append({"asset": m, "error": str(e)[:200]})
            continue
        if isinstance(v, int) and v > LOD0_VERTEX_WARN:
            heavy.append({"asset": m, "lod0Vertices": v, "lods": lods, "nanite": nanite})
        if isinstance(v, int) and v >= SINGLE_LOD_VERTICES and lods == 1:
            single.append({"asset": m, "lod0Vertices": v, "nanite": nanite})
    if assets:
        if not meshes:
            checks.append({"check": "lod0Vertices", "status": "notChecked", "count": 0,
                           "detail": "FindAssets with assetType /Script/Engine.StaticMesh found no mesh under %s" % root})
        else:
            for key, rows, name in (("simplify", heavy, "lod0Vertices"), ("singleLod", single, "singleLod")):
                checks.append({"check": name, "status": "warn" if rows else "pass", "checked": len(meshes) -
                               len(mesh_unread), "count": len(rows), "assets": rows[:50], "source": _cite(key),
                               "unread": mesh_unread[:20] or None, "capped": len(meshes) >= max_meshes or None})
    # 3. textures: power of two (else no mips, no streaming) and 2k.
    textures = find("/Script/Engine.Texture2D", max_textures) if assets else []
    npot, big, tex_unread = [], [], []
    for t in textures:
        try:
            s = T.read(TEXTURE + ".GetSize", {"texture": ref(t)})
        except StepError as e:
            tex_unread.append({"asset": t, "error": str(e)[:200]})
            continue
        w, h = (s.get("x"), s.get("y")) if isinstance(s, dict) else (None, None)
        if not (isinstance(w, int) and isinstance(h, int)):
            tex_unread.append({"asset": t, "answer": json.dumps(s, default=str)[:120]})
            continue
        if not (_pow2(w) and _pow2(h)):
            npot.append({"asset": t, "size": [w, h]})
        if max(w, h) > TEXTURE_MAX:
            big.append({"asset": t, "size": [w, h]})
    if assets:
        if not textures:
            checks.append({"check": "textureStreaming", "status": "notChecked", "count": 0,
                           "detail": "FindAssets with assetType /Script/Engine.Texture2D found no texture under %s"
                                     % root})
        else:
            checks.append({"check": "textureStreaming", "status": "warn" if npot else "pass", "count": len(npot),
                           "checked": len(textures) - len(tex_unread), "assets": npot[:50], "source": _cite(
                               "texturePow2"), "detail": "a texture that is not a power of two gets no mipmaps and "
                                                         "cannot stream", "unread": tex_unread[:20] or None})
            checks.append({"check": "textureSize", "status": "warn" if big else "pass", "count": len(big),
                           "checked": len(textures) - len(tex_unread), "assets": big[:50],
                           "source": _cite("textureSize")})
    # 4. memory: UEFN computes the figure on every cook; no tool answers it, so the log's lines are handed back raw.
    try:
        lines = T.read(LOGS + ".GetLogEntries", {"pattern": "(?i)memory", "maxEntries": 40, "maxCharsPerEntry": 400})
    except StepError as e:
        lines = {"error": str(e)[:200]}
    checks.append({"check": "publishingMemory", "status": "notChecked", "source": _cite("memory"),
                   "logLines": lines if isinstance(lines, (list, dict)) else None,
                   "detail": "UEFN calculates the publishing memory from cooked data and no tool answers it; the "
                             "editor log's lines that mention memory are returned as they are, for the person to read "
                             "(the line's own format is not measured yet)"})
    not_checked = [
        {"check": "heaviestAssetsByMemory", "why": "the per-asset memory breakdown comes from the cook, and no tool "
                                                   "answers it"},
        {"check": "experimentalFeatures", "why": "no Epic tool lists the project's experimental features; the "
                                                 "editor's own publish validation decides"},
        {"check": "propertyOverridesOffAllowList", "why": "no tool lists a level's property overrides against UEFN's "
                                                          "allow-list; uefn_set_property refuses one before it is "
                                                          "written"},
        {"check": "unsavedAssets", "why": "FindAssets sees only assets saved to disk"},
    ]
    fails = [c["check"] for c in checks if c["status"] == "fail"]
    return {"root": root, "assetsScanned": len(assets), "meshesChecked": len(meshes), "texturesChecked": len(textures),
            "checks": checks, "notChecked": not_checked, "passed": not fails, "failed": fails or None,
            "warned": [c["check"] for c in checks if c["status"] == "warn"] or None,
            "readOnly": True, "seconds": round(time.perf_counter() - t0, 2),
            # ok is about the REPORT: false when it could not measure (nothing scanned, validation unread or short);
            # the island's own verdict is passed.
            "mismatch": [c.get("detail") or "%s: Epic's validator did not check every asset sent" % c["check"]
                         for c in checks if c["status"] == "fail" and (c["check"] == "scan" or c.get("unreadBatches")
                                                                       or c.get("numChecked", 0) < c.get("sent", 0))]}
