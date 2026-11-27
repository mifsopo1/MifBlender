"""Core tier 0 left-overs (lane R2): attributes, tags, script create and patch - composites over the generic rbx_* tools.

    rbx_attributes     Instance:GetAttributes, SetAttribute (each read back through GetAttribute), and removal (SetAttribute
                       to nil), with the attribute types Roblox's attribute page lists
    rbx_tags           Instance:GetTags, AddTag and RemoveTag, each read back with HasTag, and GetTags read again at the end
    rbx_script_create  a Script, LocalScript or ModuleScript made with its RunContext and source, read back by sha256
    rbx_script_patch   a 1-based inclusive line range of a script replaced, guarded by expect_sha256, read back by sha256

Members (0.748 dump, every one security None): Instance.GetAttributes, Instance.GetAttribute, Instance.SetAttribute,
Instance.GetTags, Instance.AddTag, Instance.RemoveTag, Instance.HasTag, BaseScript.RunContext, Script.Source,
ModuleScript.Source. Writes go through rbx_call, rbx_create and rbx_script_write, so the plugin's mode gate, its undo
recording and the coverage record all see them. tools.py imports this module's rbx_* names with one line.

NEVER RAISE: every failure is {"ok": False, ...}.
"""
import builtins
import re
from typing import Any

__all__ = ["rbx_attributes", "rbx_tags", "rbx_script_create", "rbx_script_patch"]

# The types Roblox's attribute page lists (create.roblox.com/docs/studio/properties#instance-attributes, read 2026-11-19),
# by the codec's wire type names. Measured live on 0.748 against SetAttribute: each of these is taken
# and read back equal; a type not here is refused before it is sent.
ATTRIBUTE_TYPES = ("string", "boolean", "number", "UDim", "UDim2", "BrickColor", "Color3", "Vector2", "Vector3", "CFrame",
                   "NumberSequence", "ColorSequence", "NumberRange", "Rect", "Font", "EnumItem")
_WIRE_OK = {"UDim", "UDim2", "BrickColor", "Color3", "Vector2", "Vector3", "CFrame", "NumberSequence", "ColorSequence",
            "NumberRange", "Rect", "Font", "EnumItem"}
# Roblox: an attribute name is at most 100 characters of letters, digits and underscores, and may not start with RBX.
_NAME = re.compile(r"^[A-Za-z0-9_]{1,100}$")
SCRIPT_CLASSES = ("Script", "LocalScript", "ModuleScript")


_pyset = builtins.set   # rbx_attributes' parameter is named set, the verb a caller writes


def _T():
    from mif_roblox import tools
    return tools


def _V():
    from mif_roblox import values
    return values


def _name_ok(name):
    return isinstance(name, str) and bool(_NAME.match(name)) and not name.upper().startswith("RBX")


def _attr_wire(value, enums):
    """(wire, None) or (None, reason) for one attribute value: a plain JSON string, bool or number, or a typed value
    {"$t": Vector3, "v": [...]} (any of the attribute types)."""
    V = _V()
    if isinstance(value, (bool, str)) or V._num(value):
        return value, None
    if isinstance(value, dict) and isinstance(value.get("$t"), str):
        t = value["$t"]
        base = "EnumItem" if t == "EnumItem" or t.startswith("Enum.") else t
        if t in ("Instance", "Object"):
            return None, ("an instance is not one of Roblox's attribute types; measured on 0.748, SetAttribute takes one but "
                          "GetAttribute answers an InstanceHandle, not the instance (ledger RB805), so it is refused here")
        if base not in _WIRE_OK:
            return None, "an attribute cannot hold a %s: Roblox's attribute types are %s" % (t, ", ".join(ATTRIBUTE_TYPES))
        return V.encode("Variant", value, enums)
    return None, ("an attribute value is a string, true/false, a number, or a typed value such as "
                  "{\"$t\": \"Vector3\", \"v\": [1, 2, 3]}; got %r" % (value,))


def rbx_attributes(target: Any, set: dict = None, remove: list = None) -> dict:  # noqa: A002 - the verb a caller writes
    "Roblox Studio: an instance's attributes - read them all (Instance:GetAttributes), set some (SetAttribute, each read back through GetAttribute) and remove some (SetAttribute to nil, read back as absent), then GetAttributes again. A value is a string, true/false, a number, or a typed value such as {\"$t\":\"Vector3\",\"v\":[1,2,3]}, {\"$t\":\"Color3\",\"v\":[1,0,0]} or {\"$t\":\"EnumItem\",\"enum\":\"Material\",\"v\":\"Grass\"}; the types Roblox's attribute page lists (string, boolean, number, UDim, UDim2, BrickColor, Color3, Vector2, Vector3, CFrame, NumberSequence, ColorSequence, NumberRange, Rect, Font, EnumItem). A name is 1-100 letters, digits or underscores, not starting with RBX. Every write runs through rbx_call: the plugin's safety mode applies. With neither set nor remove, read only."
    T = _T()
    parts, err = T._target(target)
    if err:
        return err
    if set is not None and not isinstance(set, dict):
        return {"ok": False, "ran": False, "error": "set is an object {name: value}"}
    if remove is not None and not (isinstance(remove, list) and all(isinstance(n, str) for n in remove)):
        return {"ok": False, "ran": False, "error": "remove is a list of attribute names"}
    cat, err = T._catalog()
    if err:
        return err
    wires = {}
    for name, value in (set or {}).items():
        if not _name_ok(name):
            return {"ok": False, "ran": False, "refusedRule": "type", "attribute": name,
                    "error": "attribute name %r: 1-100 letters, digits or underscores, not starting with RBX" % (name,)}
        if value is None:
            return {"ok": False, "ran": False, "refusedRule": "type", "attribute": name,
                    "error": "a null value removes an attribute: put %r in remove instead" % name}
        w, why = _attr_wire(value, cat.enums)
        if why:
            return {"ok": False, "ran": False, "refusedRule": "type", "attribute": name, "error": why}
        wires[name] = value
    clash = [n for n in (remove or []) if n in wires]
    if clash:
        return {"ok": False, "ran": False, "error": "%s is both set and removed" % ", ".join(clash)}
    path = T.V.join_path(parts)
    results, ok = {}, True
    for name, value in wires.items():
        r = T.rbx_call(path, "SetAttribute", [name, value])
        results[name] = {k: r.get(k) for k in ("ok", "verified", "readBack", "error", "refusedRule") if r.get(k) is not None}
        ok = ok and bool(r.get("verified"))
        if r.get("refusedRule"):
            return dict(r, attribute=name, results=results)
    for name in remove or []:
        r = T.rbx_call(path, "SetAttribute", [name, None])
        results[name] = {k: r.get(k) for k in ("ok", "verified", "readBack", "error", "refusedRule") if r.get(k) is not None}
        ok = ok and bool(r.get("verified"))
        if r.get("refusedRule"):
            return dict(r, attribute=name, results=results)
    after = T.rbx_call(path, "GetAttributes", [])
    if not after.get("ok"):
        return after
    attrs = after.get("value") if isinstance(after.get("value"), dict) else {}
    out = {"ok": ok, "target": path, "attributes": attrs, "count": len(attrs)}
    if wires or remove:
        gone = [n for n in remove or [] if n in attrs]
        missing = [n for n in wires if n not in attrs]
        out.update(verified=ok and not gone and not missing, results=results)
        if gone or missing:
            out.update(ok=False, error="GetAttributes after the writes %s" % "; ".join(
                x for x in ("still holds %s" % gone if gone else "", "lacks %s" % missing if missing else "") if x))
        elif not ok:
            out["error"] = "a write did not read back: see results"
    return out


def rbx_tags(target: Any, add: list = None, remove: list = None) -> dict:
    "Roblox Studio: an instance's CollectionService tags - read them (Instance:GetTags), add some (AddTag, each read back with HasTag true) and remove some (RemoveTag, read back with HasTag false), then GetTags again. Every write runs through rbx_call: the plugin's safety mode applies. With neither add nor remove, read only."
    T = _T()
    parts, err = T._target(target)
    if err:
        return err
    for what, lst in (("add", add), ("remove", remove)):
        if lst is not None and not (isinstance(lst, list) and all(isinstance(n, str) and 0 < len(n) <= 100 for n in lst)):
            return {"ok": False, "ran": False, "error": "%s is a list of tag names (1-100 characters)" % what}
    clash = sorted(_pyset(add or []) & _pyset(remove or []))
    if clash:
        return {"ok": False, "ran": False, "error": "%s is both added and removed" % ", ".join(clash)}
    path = T.V.join_path(parts)
    results, ok = {}, True
    for verb, names in (("AddTag", add or []), ("RemoveTag", remove or [])):
        for n in names:
            r = T.rbx_call(path, verb, [n])
            results["%s %s" % (verb, n)] = {k: r.get(k) for k in ("ok", "verified", "readBack", "error", "refusedRule")
                                            if r.get(k) is not None}
            if r.get("refusedRule"):
                return dict(r, tag=n, results=results)
            ok = ok and bool(r.get("verified"))
    after = T.rbx_call(path, "GetTags", [])
    if not after.get("ok"):
        return after
    tags = sorted(after.get("value") or [])
    out = {"ok": ok, "target": path, "tags": tags, "count": len(tags)}
    if add or remove:
        wrong = [n for n in add or [] if n not in tags] + [n for n in remove or [] if n in tags]
        out.update(verified=ok and not wrong, results=results)
        if wrong:
            out.update(ok=False, error="GetTags after the writes disagrees for %s" % wrong)
        elif not ok:
            out["error"] = "a write did not read back: see results"
    return out


def rbx_script_create(class_name: str, parent: Any, name: str, source: str = "", run_context: str = None,
                      allow_syntax_errors: bool = False) -> dict:
    "Roblox Studio: make a Script, LocalScript or ModuleScript under parent with its source (through the script editor, as rbx_script_write) and, for a Script, its RunContext (Enum.RunContext: Legacy, Server, Client, Plugin; a LocalScript's stays Legacy, measured). Read back: the new script by path, its RunContext, and its source by sha256. An existing child of that name is refused, never overwritten. Source that does not compile is refused syntax before anything is made (as rbx_script_write; allow_syntax_errors true for a deliberately broken script). The plugin's safety mode applies."
    T = _T()
    if class_name not in SCRIPT_CLASSES:
        return {"ok": False, "ran": False, "error": "class_name is one of %s" % ", ".join(SCRIPT_CLASSES)}
    if not isinstance(name, str) or not name or "." in name:
        return {"ok": False, "ran": False, "error": "name is a non-empty name with no dot"}
    if not isinstance(source, str):
        return {"ok": False, "ran": False, "error": "source is a string"}
    if run_context is not None and class_name == "ModuleScript":
        return {"ok": False, "ran": False, "refusedRule": "type",
                "error": "a ModuleScript has no RunContext (BaseScript.RunContext is on Script and LocalScript)"}
    if run_context is not None and class_name == "LocalScript" and run_context != "Legacy":
        # Measured on 0.748: a LocalScript takes RunContext Client without an error and reads back Legacy.
        return {"ok": False, "ran": False, "refusedRule": "type",
                "error": "a LocalScript's RunContext stays Legacy (Studio drops any other value, measured on 0.748): use a "
                         "Script with run_context Client for a client script that runs where it is placed"}
    pparts, err = T._target(parent, "parent")
    if err:
        return err
    if not allow_syntax_errors:
        from mif_roblox import luau_syntax  # noqa: PLC0415
        bad = luau_syntax.refusal(T.V.join_path(pparts + [name]), source)
        if bad:
            return bad
    have = T._rbx("resolve", target=pparts + [name])
    if have.get("ok"):
        return {"ok": False, "ran": False, "refusedRule": "exists", "path": T.V.join_path(pparts + [name]),
                "error": "%s already exists (a %s): nothing made" % (T.V.join_path(pparts + [name]), have.get("className"))}
    props = {"RunContext": run_context} if run_context is not None else None
    made = T.rbx_create(class_name, T.V.join_path(pparts), name=name, props=props)
    if not made.get("ok"):
        return made
    path = made.get("path") or T.V.join_path(pparts + [name])
    out = {"path": path, "className": class_name, "created": {k: made.get(k) for k in ("ok", "verified", "after")}}
    w = T.rbx_script_write(path, source, allow_syntax_errors=bool(allow_syntax_errors))
    out["write"] = {k: w.get(k) for k in ("ok", "verified", "sha256", "sha256Sent", "error", "refusedRule") if k in w}
    if not w.get("ok"):
        # Create and write are two steps: a write that failed would leave an empty script behind. Take it out again,
        # and when that fails too, say plainly that it is still there.
        d = T.rbx_destroy(path)
        out.update(ok=False, verified=False, removed=bool(d.get("verified")))
        if d.get("verified"):
            out["error"] = "the source was not written (%s): the empty script made at %s was removed again" % (
                w.get("error") or "no reason given", path)
        else:
            out.update(leftEmpty=path, error="the source was not written (%s) and removing the script failed (%s): an "
                                             "empty %s was left at %s" % (w.get("error") or "no reason given",
                                                                          d.get("error") or "no reason given",
                                                                          class_name, path))
        return out
    rc = None
    if class_name != "ModuleScript":
        g = T.rbx_get_props(path, ["RunContext"])
        rc = (g.get("values") or {}).get("RunContext") if g.get("ok") else None
        out["runContext"] = rc
    good = bool(made.get("verified")) and bool(w.get("verified")) and (run_context is None or rc == run_context)
    out.update(ok=good, verified=good, sha256=w.get("sha256"))
    if not good:
        out["error"] = (w.get("error") or made.get("error") or
                        "RunContext reads back %s, not %s" % (rc, run_context))
    return out


# A Luau line ends at CR LF, LF or CR. Python's str.splitlines also breaks on \v, \f, \x1c-\x1e, \x85, U+2028 and U+2029,
# which Luau keeps inside a line, so a source holding one would be patched on the wrong lines. Each line keeps its ending.
_LINE = re.compile(r"[^\r\n]*(?:\r\n|\n|\r)|[^\r\n]+\Z")


def _lines(s):
    return _LINE.findall(s or "")


def rbx_script_patch(target: Any, start_line: int, end_line: int, text: str, expect_sha256: str,
                     allow_syntax_errors: bool = False) -> dict:
    "Roblox Studio: replace lines start_line..end_line (1-based, inclusive) of a script with text, through the script editor (ScriptEditorService:UpdateSourceAsync, as rbx_script_write). end_line = start_line - 1 inserts before start_line; start_line = line count + 1 appends. text is whole lines: a trailing newline is added when missing, and an empty text deletes the range. expect_sha256 is the sha256 rbx_script_read gave: a script that changed since is refused unsent. Read back: the whole source by sha256, equal to the patched text. A patch that leaves the script not compiling is refused syntax with the line (allow_syntax_errors true to write it anyway)."
    T = _T()
    for k, v in (("start_line", start_line), ("end_line", end_line)):
        if not isinstance(v, int) or isinstance(v, bool):
            return {"ok": False, "ran": False, "error": "%s is a whole number" % k}
    if not isinstance(text, str):
        return {"ok": False, "ran": False, "error": "text is a string"}
    if not (isinstance(expect_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expect_sha256)):
        return {"ok": False, "ran": False, "refusedRule": "type",
                "error": "expect_sha256 is the 64-hex sha256 rbx_script_read answered: a patch is never made blind"}
    cur = T.rbx_script_read(target)
    if not cur.get("ok"):
        return cur
    src = cur.get("source") or ""
    if cur.get("sha256") != expect_sha256:
        return {"ok": False, "ran": False, "refusedRule": "sha-mismatch", "sha256": cur.get("sha256"),
                "expected": expect_sha256, "error": "the script changed since it was read: nothing written. Read it again"}
    lines = _lines(src)
    n = len(lines)
    if not 1 <= start_line <= n + 1 or not start_line - 1 <= end_line <= n:
        return {"ok": False, "ran": False, "refusedRule": "range", "lineCount": n,
                "error": "lines %d..%d are outside a %d-line script (start 1..%d, end start-1..%d)"
                         % (start_line, end_line, n, n + 1, n)}
    if lines and not lines[-1].endswith(("\n", "\r")) and start_line == n + 1:
        lines[-1] += "\n"             # appending after a last line with no newline: end that line first
    new = text if not text or text.endswith("\n") else text + "\n"
    patched = "".join(lines[:start_line - 1]) + new + "".join(lines[end_line:])
    w = T.rbx_script_write(target, patched, expect_sha256=expect_sha256, allow_syntax_errors=bool(allow_syntax_errors))
    out = {k: w.get(k) for k in ("ok", "verified", "target", "className", "sha256Before", "sha256", "sha256Sent", "error",
                                 "refusedRule", "requestId", "ran", "line", "column", "message", "checkedBy") if k in w}
    out.update(replaced=[start_line, end_line], linesRemoved=max(0, end_line - start_line + 1),
               linesAdded=len(_lines(new)), lineCount=len(_lines(patched)))
    return out
