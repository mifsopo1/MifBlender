"""testSwitchesOff: the switches a place's self-test sets, read in the saved file that is about to ship.

Found 2026-11-26 on a game: a private publish went live with its self-test's other switches still set as Workspace
attributes (hazards off, enemy attacks off, the server's movement check off, a test rig on). The publish preflight's
selfTestOff reads one flag only, so it passed. A self-test run in the edit DataModel leaves what it set there, and a
save keeps it.

What this does, statically and on the file (nothing runs):
  1. test scripts: every script whose path names a test (a part matching self-test, or a part that is Test/Tests, ends
     in Test/Tests, or starts Test_/TestX), read from the saved file;
  2. setters: each `<receiver>:SetAttribute("Name", value)` with a literal name in them, by script and line; a value
     written literally false or nil is a reset, not a switch; a receiver this module can follow (workspace, Workspace,
     game.Workspace, game:GetService("Workspace") or another service, then .Child names) gives the instance's path,
     any other receiver (a local variable) is matched by the attribute's name on every instance;
  3. the file's instances (place_manifest rows: attributes decoded where they hold strings, booleans and numbers): a
     switch still holding a value other than false, 0, "" or nil fails, named with the script line that sets it; an
     instance whose attributes could not be decoded is unread for the switches that could be on it.
Caller switches (test_switches: 'Path@Attribute' or a bare attribute name) are checked the same way.
"""
import re

SERVICE_ALIASES = {"workspace": "Workspace", "Workspace": "Workspace"}
_TEST_PART = re.compile(r"(?i)self[_ ]?tests?")
_SET = re.compile(r"""([A-Za-z_][\w\.]*(?:\s*:\s*GetService\s*\(\s*["'][A-Za-z]+["']\s*\)(?:\.[A-Za-z_]\w*)*)?)"""
                  r"""\s*:\s*SetAttribute\s*\(\s*["']([^"'\n]+)["']\s*,\s*([^\n]*)""")
_GETSVC = re.compile(r"""^game\s*:\s*GetService\s*\(\s*["']([A-Za-z]+)["']\s*\)((?:\.[A-Za-z_]\w*)*)$""")
OFF_VALUES = (False, None, 0, 0.0, "")


def is_test_path(path):
    for part in path or ():
        p = str(part)
        if _TEST_PART.search(p) or p.lower() in ("test", "tests") or re.search(r"[a-z0-9_]Tests?$", p) \
                or re.match(r"Tests?(_|[A-Z])", p):
            return True
    return False


def _strip_comments(src):
    """Source with --[[ ]] / --[=[ ]=] blocks and -- line comments blanked, line breaks kept (so line numbers hold)."""
    out = re.sub(r"--\[(=*)\[.*?\]\1\]", lambda m: "\n" * m.group(0).count("\n"), src, flags=re.S)
    return re.sub(r"--[^\n]*", "", out)


def receiver_path(expr):
    """The instance path a SetAttribute receiver names, or None when it is not a path this module can follow."""
    e = re.sub(r"\s+", "", expr or "")
    m = _GETSVC.match(e)
    if m:
        return [m.group(1)] + [x for x in m.group(2).split(".") if x]
    parts = e.split(".")
    if parts[0] in SERVICE_ALIASES:
        return [SERVICE_ALIASES[parts[0]]] + parts[1:]
    if parts[0] == "game" and len(parts) > 1:
        return parts[1:]
    return None


def _literal_off(value_text):
    v = value_text.strip()
    v = v[:-1].strip() if v.endswith(")") else v
    return v in ("false", "nil")


def setters(sources):
    """[{script, line, path (None: by name), attribute}] of the switches the test scripts among sources set."""
    out = []
    for s in sources or []:
        if not isinstance(s, dict) or not is_test_path(s.get("path")):
            continue
        text = _strip_comments(str(s.get("source") or "").replace("\r\n", "\n"))
        for m in _SET.finditer(text):
            if _literal_off(m.group(3)):
                continue
            out.append({"script": ".".join(str(p) for p in s["path"]), "line": text.count("\n", 0, m.start()) + 1,
                        "path": receiver_path(m.group(1)), "attribute": m.group(2), "receiver": m.group(1).strip()})
    return out


def caller_switches(items):
    """test_switches entries -> setter rows (script 'named by the caller'); (rows, error)."""
    out = []
    for it in items or []:
        if not isinstance(it, str) or not it.strip():
            return None, "test_switches are 'Path@Attribute' or attribute names"
        if "@" in it:
            p, a = it.rsplit("@", 1)
            out.append({"script": "test_switches", "line": None, "path": [x for x in p.split(".") if x] or None,
                        "attribute": a, "receiver": p})
        else:
            out.append({"script": "test_switches", "line": None, "path": None, "attribute": it, "receiver": None})
    return out, None


def judge(rows, wanted):
    """(left [{attribute, on, value, setBy}], unread [{attribute, on, why}]) for the switches wanted against the file's
    manifest rows."""
    left, unread, seen = [], [], set()
    for w in wanted:
        cands = [r for r in rows if w["path"] is None or list(r["path"]) == list(w["path"])]
        for r in cands:
            on = ".".join(r["path"])
            key = (on, w["attribute"])
            if r.get("attributesAs") == "sha256":
                if w["path"] is not None and key not in seen:
                    seen.add(key)
                    unread.append({"attribute": w["attribute"], "on": on,
                                   "why": "its attributes hold a type this reader does not decode"})
                continue
            attrs = r.get("attributes") or {}
            if w["attribute"] in attrs and key not in seen:
                v = attrs[w["attribute"]]
                if v not in OFF_VALUES and v is not False:
                    seen.add(key)
                    left.append({"attribute": w["attribute"], "on": on, "value": v,
                                 "setBy": "%s:%s" % (w["script"], w["line"]) if w["line"] else w["script"]})
    return left, unread


def check(file, sources, test_switches=None, skip=None):
    """-> (status, read, why, step). skip: (path, attribute) the selfTestOff check already reads."""
    import os
    from mif_roblox import place_manifest as PM
    named, err = caller_switches(test_switches)
    if err:
        return "unread", None, err, "pass test_switches as a list of 'Path@Attribute' strings"
    found = setters(sources)
    wanted = [w for w in found + named if not (skip and w["path"] == skip[0] and w["attribute"] == skip[1])]
    tests = sorted({".".join(str(p) for p in s["path"]) for s in sources or [] if isinstance(s, dict)
                    and is_test_path(s.get("path"))})
    read = {"testScripts": tests[:50], "switches": [dict(w, path=".".join(w["path"]) if w["path"] else None)
                                                   for w in wanted][:100], "scriptsRead": len(sources or [])}
    if not sources:
        return "unread", read, "no scripts were read from the file, so no self-test was looked for", None
    if not wanted:
        return "pass", dict(read, note="no test script in the file sets an attribute with a literal name, and none "
                                       "was named in test_switches"), None, None
    try:
        with open(str(file), "rb") as fh:
            blob = fh.read()
        _fmt, rows, _und = PM.read(blob, os.path.splitext(str(file))[1].lower())
    except Exception as exc:  # noqa: BLE001 - an unreadable file is unread, never a pass
        return "unread", read, "the file's attributes could not be read (%s: %s)" % (type(exc).__name__, exc), None
    left, unread = judge(rows, wanted)
    read.update(left=left[:50], unreadOn=unread[:50])
    if left:
        return ("fail", read, "%d test switch(es) are still set in the file that would go live: %s" % (
            len(left), "; ".join("%s@%s = %r (set by %s)" % (x["on"], x["attribute"], x["value"], x["setBy"])
                                 for x in left[:5])),
                "clear each in Studio (SetAttribute(name, nil), or the self-test's own reset), save, and run the preflight "
                "again; a self-test run in the edit place (not a play test) leaves them there")
    if unread:
        return "unread", read, "the attributes of %s could not be decoded" % ", ".join(sorted({u["on"] for u in unread})), None
    return "pass", read, None, None
