"""Scripts (batch 4) and play-test diagnostics (batch 3): composites over the generic rbx_* tools (the internal notes sections 5.2
and 5.10, the internal notes "Scripts and diagnostics").

    rbx_script_readonly  an open tab's read-only reasons: list, add, remove (ScriptDocument:AddReadOnlyReasonsAsync), so a
                         read-only tab can be made and rbx_script_write's refusal proven without Team Create
    rbx_script_edit_range a range of an open tab replaced (ScriptDocument:EditTextAsync), read back by sha256
    rbx_device_state     the device simulator read through its own getters; orientation, density, scaling set and read back
    rbx_script_analyze   Luau type check and lint of the place's scripts by an analyzer on this machine (luau-analyze),
                         honoring Workspace.LuauTypeCheckMode; refused tool-missing when none is installed. sourcemap
                         types game, workspace and script from the place's tree and the catalog (luau_types.py) so a
                         wrong Instance member is found; format runs a formatter installed on this machine
    rbx_script_profile   ScriptProfilerService in a play server (rbx_play_probe kind profile): top functions
    rbx_cloud_luau       Open Cloud Luau Execution: a script run on Roblox's servers against a published place version,
                         with the person's own key; never inside the plugin

Every Studio member used here is the dump's (0.748), cited in the internal notes. Nothing here edits the plugin: each composite is
built from rbx_call, rbx_watch, rbx_run_mode, rbx_script_documents and the script_scan op, and every write is read back.
tools.py imports this module's rbx_* names into its own TOOLS with one line; this module reaches tools.py lazily (_T),
so importing either first works.

NEVER RAISE: every failure is {"ok": False, ...}.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

__all__ = ["rbx_script_readonly", "rbx_script_edit_range", "rbx_script_analyze", "rbx_script_profile", "rbx_device_state",
           "rbx_cloud_luau", "rbx_flagged"]


def _T():
    from mif_roblox import tools
    return tools


def _V():
    from mif_roblox import values
    return values


# ---------------------------------------------------------------- read-only tabs
SED = "ScriptEditorService"


def _doc_handle(T, path):
    """The open tab's ScriptDocument as a call target, or (None, the answer). Measured on 0.748 (RB200): a ScriptDocument
    is a child of ScriptEditorService named by the script's full name, so it crosses as the Instance path
    [ScriptEditorService, <full name>]; that name holds dots, so the target stays the two-name list.
    A handle (an object outside the place) is taken too."""
    r = T.rbx_call(SED, "FindScriptDocument", [path])
    if not r.get("ok"):
        return None, r
    v = r.get("value")
    if isinstance(v, dict) and v.get("handle"):
        return {"handle": v["handle"]}, None
    # join_path keeps a path as its list when a name holds a dot, which a document's name always does.
    if isinstance(v, list) and len(v) == 2 and v[0] == SED and isinstance(v[1], str) and v[1]:
        return list(v), None
    return None, {"ok": False, "ran": True, "refusedRule": "not-open",
                  "error": "%s is not open in a script editor tab (FindScriptDocument answered %r)" % (path, v)}


def _reasons(T, h):
    ro = T.rbx_call(h, "IsReadOnly")
    rs = T.rbx_call(h, "GetReadOnlyReasons")
    if not ro.get("ok") or not rs.get("ok"):
        bad = ro if not ro.get("ok") else rs
        return None, None, bad
    reasons = rs.get("value")
    return bool(ro.get("value")), (reasons if isinstance(reasons, list) else []), None


def rbx_script_readonly(target: Any, action: str = "list", reasons: list = None) -> dict:
    "Roblox Studio: a script's open editor tab and its read-only reasons. list reads IsReadOnly and GetReadOnlyReasons; add opens a tab if none is open (no dialog) and adds reasons through ScriptDocument:AddReadOnlyReasonsAsync; remove takes them off with RemoveReadOnlyReasonsAsync. Each add or remove is read back through GetReadOnlyReasons. A tab made read-only this way is refused by rbx_script_write with these reasons, which is how that refusal is proven without Team Create. It changes the person's editor, not the place; remove what you add."
    T = _T()
    if action not in ("list", "add", "remove"):
        return {"ok": False, "ran": False, "error": "action is list, add or remove"}
    parts, err = T._target(target)
    if err:
        return err
    path = _V().join_path(parts)
    if action != "list":
        if reasons is None:
            reasons = ["MifBridge"]
        if not isinstance(reasons, list) or not reasons or not all(isinstance(x, str) and 0 < len(x) <= 200 for x in reasons):
            return {"ok": False, "ran": False, "error": "reasons is a list of 1-200 character strings"}
    opened = False
    if action != "list":
        refusal, _ping = _may_write(T, parts)
        if refusal:
            return refusal
    if action == "add":
        st = T.rbx_script_documents(path)
        if not st.get("ok"):
            if st.get("refusedRule") != "not-open":
                return st
            op =T.rbx_script_documents(path, action="open")
            if not op.get("ok"):
                return op
            opened = True
    h, err = _doc_handle(T, path)
    if err:
        return err
    before_ro, before, err = _reasons(T, h)
    if err:
        return err
    out = {"ok": True, "target": path, "openedTab": opened, "readOnly": before_ro, "reasons": before}
    if action == "list":
        return out
    method = "AddReadOnlyReasonsAsync" if action == "add" else "RemoveReadOnlyReasonsAsync"
    r = T.rbx_call(h, method, [reasons], timeout=30)
    if not r.get("ok"):
        return dict(r, target=path, openedTab=opened)
    after_ro, after, err = _reasons(T, h)
    if err:
        return dict(err, ok=False, verified=False, target=path, error="the call ran; reading the reasons back failed: %s"
                    % err.get("error"))
    if action == "add":
        good = all(x in after for x in reasons) and after_ro
    else:
        good = not any(x in after for x in reasons)
    out.update(ok=good, verified=good, readOnly=after_ro, reasons=after, reasonsBefore=before, sent=reasons,
               member="ScriptDocument:%s" % method)
    if not good:
        out["error"] = "GetReadOnlyReasons reads %r after %s %r (IsReadOnly %s)" % (after, method, reasons, after_ro)
    elif action == "add":
        out["prove"] = "rbx_script_write(%r, ...) is now refused read-only-document with these reasons" % path
    return out


# ---------------------------------------------------------------- range edits in an open tab
SCRATCH = ["Workspace", "MifScratch"]
SCRATCH_PLACE = "MifScratch.rbxlx"   # the plugin's own constant: rbx_launch's scratch place, all of it scratch


def _may_write(T, parts):
    """A ScriptDocument is a handle, which the plugin's gate lets through in scratch mode: the scratch rule is applied
    here instead, by the script's own path. (None, ping) when allowed, else (refusal, ping)."""
    ping = T._rbx("ping", nonce="lane2")
    if not ping.get("ok"):
        return ping, ping
    mode = ping.get("mode")
    if mode == "full" or (mode == "scratch" and (ping.get("placeName") == SCRATCH_PLACE or parts[:2] == SCRATCH)):
        return None, ping
    rule = "read-mode" if mode == "read" else "scratch-path"
    return {"ok": False, "ran": False, "refusedRule": rule, "mode": mode,
            "error": "%s is a write to %s and the panel is in %s mode: nothing was done. Raise the mode in the MifBridge "
                     "panel; nothing over the bridge can" % ("an editor edit", _V().join_path(parts), mode)}, ping


def splice(text, start_line, start_char, end_line, end_char, new):
    """text with [start, end) replaced, positions 1-based lines and characters, end exclusive. None when out of range."""
    lines = text.split("\n")

    def off(line, char):
        if not 1 <= line <= len(lines) or not 1 <= char <= len(lines[line - 1]) + 1:
            return None
        return sum(len(x) + 1 for x in lines[:line - 1]) + char - 1
    a, b = off(start_line, start_char), off(end_line, end_char)
    if a is None or b is None or b < a:
        return None
    return text[:a] + new + text[b:]


def rbx_script_edit_range(target: Any, start_line: int, start_character: int, end_line: int, end_character: int,
                          text: str, expect_sha256: str = None) -> dict:
    "Roblox Studio: replace a range of a script in its open editor tab through ScriptDocument:EditTextAsync (a tab is opened first if none is, no dialog). Lines and characters are 1-based, the end exclusive: replacing line 3 whole is (3, 1, 3, len+1). Read back by sha256 of the tab's GetText against the edit applied here, and of the script's editor source. With expect_sha256 (of the tab's text from rbx_script_documents), refused unsent if the text changed. A read-only tab is refused with Studio's reasons. Outside Workspace.MifScratch it needs full mode or MifBridge's scratch place."
    T = _T()
    parts, err = T._target(target)
    if err:
        return err
    nums = (start_line, start_character, end_line, end_character)
    if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 1 for n in nums):
        return {"ok": False, "ran": False, "error": "lines and characters are whole numbers from 1"}
    if not isinstance(text, str):
        return {"ok": False, "ran": False, "error": "text is a string"}
    refusal, _ping = _may_write(T, parts)
    if refusal:
        return refusal
    path = _V().join_path(parts)
    st = T.rbx_script_documents(path)
    opened = False
    if not st.get("ok"):
        if st.get("refusedRule") != "not-open":
            return st
        st = T.rbx_script_documents(path, action="open")
        if not st.get("ok"):
            return st
        opened = True
    h, err = _doc_handle(T, path)
    if err:
        return err
    ro, reasons, err = _reasons(T, h)
    if err:
        return err
    if ro:
        return {"ok": False, "ran": False, "refusedRule": "read-only-document", "readOnlyReasons": reasons,
                "error": "%s is open read-only: nothing written" % path}
    g = T.rbx_call(h, "GetText")
    if not g.get("ok") or not isinstance(g.get("value"), str):
        return dict(g, ok=False, error="reading the tab's text failed: %s" % g.get("error"))
    before = g["value"]
    sha_before = T._sha(before)
    if expect_sha256 is not None and expect_sha256 != sha_before:
        return {"ok": False, "ran": False, "refusedRule": "sha-mismatch", "sha256": sha_before, "expected": expect_sha256,
                "error": "the tab's text changed since it was read: nothing written"}
    want = splice(before, start_line, start_character, end_line, end_character, text)
    if want is None:
        return {"ok": False, "ran": False, "refusedRule": "range", "lines": len(before.split("\n")),
                "error": "the range (%d,%d)-(%d,%d) is not inside the tab's text, or ends before it starts" % nums}
    r = T.rbx_call(h, "EditTextAsync", [text, start_line, start_character, end_line, end_character], timeout=30)
    if not r.get("ok"):
        return dict(r, target=path, openedTab=opened)
    g2 = T.rbx_call(h, "GetText")
    after = g2.get("value") if g2.get("ok") else None
    src = T.rbx_script_read(path)
    good_tab = T._sha(after) == T._sha(want)
    good_src = src.get("ok") and src.get("sha256") == T._sha(want)
    out = {"ok": bool(good_tab and good_src), "verified": bool(good_tab and good_src), "target": path, "openedTab": opened,
           "sha256Before": sha_before, "sha256Expected": T._sha(want), "sha256Tab": T._sha(after),
           "sha256Source": src.get("sha256"), "returned": r.get("values", r.get("value")),
           "member": "ScriptDocument:EditTextAsync"}
    if not good_tab:
        out["error"] = ("the tab reads back other than the edit applied as 1-based, end exclusive: the position rule is "
                        "measured, and this answer is the measurement")
        out["tabText"] = after if isinstance(after, str) and len(after) < 4000 else None
    elif not good_src:
        out["error"] = "the tab holds the edit but the script's editor source does not: %s" % (src.get("error") or "differs")
    return out


# ---------------------------------------------------------------- the device simulator, read back by its getters
DSS = "StudioDeviceSimulatorService"
_DEVICE_GETTERS = ("GetDeviceAsync", "GetResolutionAsync", "GetOrientationAsync", "GetPixelDensityAsync",
                   "GetScalingModeAsync")


def rbx_device_state(orientation: str = None, pixel_density: float = None, scaling_mode: str = None) -> dict:
    "Roblox Studio: the device simulator's state through its own getters - GetDeviceAsync, GetResolutionAsync, GetOrientationAsync, GetPixelDensityAsync, GetScalingModeAsync - and, when given, set orientation (Enum.ScreenOrientation), pixel_density or scaling_mode (Enum.DeviceSimulatorScalingMode), each read back through its getter. It changes the viewport, not the place; set a device first with rbx_device_sim (Studio refuses some sets without one)."
    T = _T()
    sets = [(m, g, v) for m, g, v in (("SetOrientationAsync", "GetOrientationAsync", orientation),
                                      ("SetPixelDensityAsync", "GetPixelDensityAsync", pixel_density),
                                      ("SetScalingModeAsync", "GetScalingModeAsync", scaling_mode)) if v is not None]
    # 72-10000 is Studio's own refusal text on 0.748 (RB202), not a guess.
    if pixel_density is not None and (not isinstance(pixel_density, (int, float)) or isinstance(pixel_density, bool)
                                      or not 72 <= pixel_density <= 10000):
        return {"ok": False, "ran": False, "error": "pixel_density is 72-10000 (Studio refuses anything else)"}
    out = {"ok": True, "set": {}, "state": {}}
    for setter, getter, value in sets:
        r = T.rbx_call(DSS, setter, [value], timeout=30)
        if not r.get("ok"):
            out["set"][setter] = {"ok": False, "error": r.get("error"), "refusedRule": r.get("refusedRule")}
            out["ok"] = False
            continue
        g = T.rbx_call(DSS, getter, timeout=30)
        got = g.get("value") if g.get("ok") else None
        if isinstance(value, str):
            same = _enum_name(got) == value.split(".")[-1]
        else:
            same = isinstance(got, (int, float)) and abs(got - value) <= 1e-3 * max(1.0, abs(value))
        out["set"][setter] = {"ok": same, "verified": same, "sent": value, "readBack": got,
                              "error": None if same else (g.get("error") or "%s reads %r" % (getter, got))}
        out["ok"] = out["ok"] and same
    for getter in _DEVICE_GETTERS:
        g = T.rbx_call(DSS, getter, timeout=30)
        out["state"][getter[3:-5]] = g.get("value") if g.get("ok") else {"error": g.get("error")}
    # With no device set, GetDeviceAsync answers "default" and the resolution, density and scaling getters refuse (RB202).
    res = out["state"].get("Resolution")
    out["deviceActive"] = out["state"].get("Device") not in (None, "default") and not (isinstance(res, dict) and "error" in res)
    if sets:
        out["verified"] = out["ok"]
    if not out["ok"]:
        out["error"] = "; ".join("%s: %s" % (k, v["error"]) for k, v in out["set"].items() if not v.get("ok"))
    return out


# ---------------------------------------------------------------- Luau analysis
# The analyzer is the person's own install. Found, in order: ANALYZER (a suite's
# command list), MIF_LUAU_ANALYZE (a path), luau-analyze on PATH. Nothing is downloaded or installed by the bridge.
ANALYZER = None
ANALYZE_TIMEOUT = 120
# Roblox's own globals (create.roblox.com/docs/reference/engine/globals/RobloxGlobals), its data types
# (/reference/engine/datatypes) and the task library: declared in .luaurc so a stock analyzer, which knows Luau's
# builtins only, does not report every one as an unknown global. They are declared untyped: luau-analyze takes no
# definitions file (0.748 measured), so in this pass a wrong member on an Instance is not caught (said in the answer).
# sourcemap true adds the typed pass (_typed_pass, luau_types.py), which binds typed values over these names.
ROBLOX_GLOBALS = (
    "delay", "DebuggerManager", "elapsedTime", "PluginManager", "printidentity", "settings", "spawn", "stats", "tick",
    "time", "typeof", "UserSettings", "version", "wait", "warn", "ypcall", "Enum", "game", "plugin", "shared", "script",
    "workspace", "task",
    "AdReward", "Axes", "BrickColor", "CatalogSearchParams", "CFrame", "Color3", "ColorSequence", "ColorSequenceKeypoint",
    "Content", "DateTime", "DockWidgetPluginGuiInfo", "Faces", "FloatCurveKey", "Font", "Instance", "NumberRange",
    "NumberSequence", "NumberSequenceKeypoint", "OverlapParams", "Path2DControlPoint", "PathWaypoint", "PhysicalProperties",
    "Random", "Ray", "RaycastParams", "Rect", "Region3", "Region3int16", "RotationCurveKey", "Secret", "SharedTable",
    "TweenInfo", "UDim", "UDim2", "ValueCurveKey", "Vector2", "Vector2int16", "Vector3", "Vector3int16")
# Workspace.LuauTypeCheckMode -> .luaurc languageMode. "Default" is Studio's choice, not named in the dump: it is read
# as nonstrict and the answer says modeAssumed.
MODES = {"Strict": "strict", "Nonstrict": "nonstrict", "NoCheck": "nocheck", "Default": "nonstrict"}
# Diagnostic lines in the three formatters luau-analyze has (default, --formatter=gnu, --formatter=plain). Which one the
# installed build prints is read from its output; a line none matches is returned raw, never dropped.
_DIAG = (
    re.compile(r"^(?P<file>[^\s(:][^(]*?)\((?P<line>\d+),(?P<col>\d+)\):\s*(?P<kind>\w+):\s*(?P<msg>.*)$"),
    re.compile(r"^(?P<file>.+?):(?P<line>\d+)\.(?P<col>\d+)-\d+\.\d+:\s*(?P<kind>\w+):\s*(?P<msg>.*)$"),
    re.compile(r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s*(?:\(\w\d*\)\s*)?(?P<kind>\w+):\s*(?P<msg>.*)$"),
)


def find_analyzer():
    if ANALYZER:
        return list(ANALYZER)
    env = os.environ.get("MIF_LUAU_ANALYZE")
    if env:
        return [env] if os.path.isfile(env) else None
    exe = shutil.which("luau-analyze")
    return [exe] if exe else None


# The formatter is the person's own install too, found the same way: FORMATTER (a suite's command list), MIF_STYLUA (a
# path), stylua on PATH. None was installed on the machine this was built on (2026-11-22), so no formatter's output is
# read at all: it formats copies of the scripts in a temporary folder and the copies are compared with the sources.
FORMATTER = None
# Typed-pass diagnostics that are the generated types' doing and not the script's (luau-analyze 0.748, new solver,
# measured 2026-11-22): indexing a typed instance with a string that is not a literal (workspace[name]).
NOT_FINDINGS = re.compile(r"^Cannot add indexer to table")
_MEMBER = re.compile(r"^(?:Key '(?P<a>[^']+)' not found in table|Cannot add property '(?P<b>[^']+)' to table) '(?P<t>[NUGS]\w*)'")
BASE_PREFIX = "do end "


def find_formatter():
    if FORMATTER:
        return list(FORMATTER)
    env = os.environ.get("MIF_STYLUA")
    if env:
        return [env] if os.path.isfile(env) else None
    exe = shutil.which("stylua")
    return [exe] if exe else None


def _run_tool(cmd, cwd, names):
    """(completed process, None) or (None, the failure) for one run of an installed tool over the files in cwd."""
    try:
        return subprocess.run(cmd + sorted(names), cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=ANALYZE_TIMEOUT,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)), None
    except subprocess.TimeoutExpired:
        return None, {"ok": False, "ran": True, "error": "%s ran past %ss" % (os.path.basename(cmd[-1]), ANALYZE_TIMEOUT)}
    except OSError as exc:
        return None, {"ok": False, "ran": False, "refusedRule": "tool-missing",
                      "error": "%s could not start: %s" % (os.path.basename(cmd[-1]), exc)}


def _place_tree(T):
    """The DataModel's node with each service's subtree, as the plugin's tree op answers them, or (None, failure).
    One read per service, so a large Workspace cannot use up the limit before the scripts' own services are read."""
    top = T._rbx("tree", root=[], depth=1, limit=10000)
    if not top.get("ok") or not isinstance(top.get("root"), dict):
        return None, top if not top.get("ok") else {"ok": False, "error": "the tree op answered no root: %r" % (top,)}
    root = dict(top["root"])
    kids = []
    for k in root.get("children") or []:
        if isinstance(k, dict) and isinstance(k.get("name"), str) and (k.get("childCount") or 0) > 0:
            sub = T._rbx("tree", root=[k["name"]], depth=20, limit=10000)
            if sub.get("ok") and isinstance(sub.get("root"), dict) and sub["root"].get("className") == k.get("className"):
                k = sub["root"]
        kids.append(k)
    root["children"] = kids
    return root, None


def _typed_pass(cmd, tmp, sources, files, place):
    """The strict pass with the place's types bound, less what the same strict pass reports without them. Answers
    (findings, info): each finding a diagnostic with source "roblox-types"; info counts the scripts the pass is proven
    on (its canary line was reported) and names the rest."""
    from mif_roblox import luau_types as LT
    typed, base = os.path.join(tmp, "typed"), os.path.join(tmp, "base")
    shift, last = {}, {}
    for d in (typed, base):
        os.mkdir(d)
        with open(os.path.join(d, ".luaurc"), "w", encoding="utf-8") as f:
            json.dump({"languageMode": "strict", "globals": list(ROBLOX_GLOBALS)}, f)
    for i, s in enumerate(sources):
        name = "s%05d.luau" % i
        text, bound = place.module(s["source"], place.script_node(s.get("path")))
        with open(os.path.join(typed, "t%05d.luau" % i), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        pre = LT.prefix("t%05d" % i, bound)
        shift[name] = len(pre)
        last[name] = s["source"].count("\n") + 1
        with open(os.path.join(typed, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(pre + s["source"] + LT.canary())
        with open(os.path.join(base, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(BASE_PREFIX + s["source"])
    same = {n: n for n in files}

    def run(folder, shifts):
        p, err = _run_tool(cmd, folder, files)
        if err:
            return None, None, err
        diags, raw = parse_diagnostics((p.stdout or "") + "\n" + (p.stderr or ""), same)
        out = []
        for d in diags:
            if d["line"] == 1:
                if d["column"] <= shifts[d["script"]]:
                    continue                      # the prefix's own line: an unused typed local, never the script's
                d = dict(d, column=d["column"] - shifts[d["script"]])
            out.append(d)
        return out, raw, None

    t_diags, t_raw, err = run(typed, shift)
    if err:
        return None, err
    b_diags, b_raw, err = run(base, {n: len(BASE_PREFIX) for n in files})
    if err:
        return None, err
    known = {(d["script"], d["line"], d["column"], d["kind"], d["message"]) for d in b_diags}
    proven = {d["script"] for d in t_diags if d["line"] > last[d["script"]]}
    findings = []
    for d in t_diags:
        if d["line"] > last[d["script"]] or NOT_FINDINGS.match(d["message"]) \
                or (d["script"], d["line"], d["column"], d["kind"], d["message"]) in known:
            continue
        m = _MEMBER.match(d["message"])
        msg = d["message"]
        if m:
            msg = "%s is not a member of %s nor a child it has in the place" % (m.group("a") or m.group("b"),
                                                                              place.describe(m.group("t")))
        findings.append(dict(d, script=files[d["script"]], message=msg[:600], source="roblox-types"))
    info = {"scriptsTyped": len(proven), "notTyped": [files[n] for n in sorted(files) if n not in proven][:50],
            "findings": len(findings)}
    extra = [x for x in t_raw if x not in b_raw]
    if extra:
        info["unparsed"] = extra[:50]
    return findings, info


def _format_pass(tmp, sources, files):
    """The format half: the formatter found on this machine run over copies, each changed block a finding."""
    import difflib
    cmd = find_formatter()
    if not cmd:
        return {"ok": False, "ran": False, "refusedRule": "tool-missing",
                "error": "no Luau formatter on this machine: install StyLua (github.com/JohnnyMorganz/StyLua releases) "
                         "and put stylua on PATH or set MIF_STYLUA to its path. The bridge installs nothing itself"}
    folder = os.path.join(tmp, "format")
    os.mkdir(folder)
    for i, s in enumerate(sources):
        with open(os.path.join(folder, "s%05d.luau" % i), "w", encoding="utf-8", newline="\n") as f:
            f.write(s["source"])
    p, err = _run_tool(cmd, folder, files)
    if err:
        return err
    findings = []
    for i, s in enumerate(sources):
        name = "s%05d.luau" % i
        with open(os.path.join(folder, name), encoding="utf-8", errors="replace", newline="") as f:
            after = f.read().replace("\r\n", "\n").split("\n")
        before = s["source"].replace("\r\n", "\n").split("\n")
        for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
            if tag != "equal":
                findings.append({"script": files[name], "line": i1 + 1, "lines": i2 - i1, "kind": "Format",
                                 "message": "the formatter would change %d line(s) here" % (i2 - i1) if i2 > i1
                                 else "the formatter would add lines here"})
    out = {"ok": True, "formatter": os.path.basename(cmd[-1]), "exitCode": p.returncode, "count": len(findings),
           "clean": not findings, "findings": findings[:2000]}
    said = [x.strip() for x in ((p.stdout or "") + "\n" + (p.stderr or "")).splitlines() if x.strip()]
    if p.returncode != 0:
        # a formatter that could not parse a script changes nothing and says why: never read as clean
        out.update(ok=False, clean=False, error="the formatter exited %s: %s" % (p.returncode, " | ".join(said[:5])))
    if said:
        out["output"] = said[:50]
    return out


def parse_diagnostics(text, files):
    """(diagnostics, unparsed lines). files maps the written file's base name to the script path."""
    diags, raw = [], []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        for rx in _DIAG:
            m = rx.match(s)
            if m:
                base = os.path.basename(m.group("file").strip())
                if base in files:
                    diags.append({"script": files[base], "line": int(m.group("line")), "column": int(m.group("col")),
                                  "kind": m.group("kind"), "message": m.group("msg").strip()})
                    break
        else:
            raw.append(s)
    diags.sort(key=lambda d: (d["script"], d["line"], d["column"]))
    return diags, raw


def _enum_name(v):
    if isinstance(v, dict):
        v = v.get("v") or v.get("name") or v.get("text")
    return str(v).split(".")[-1] if v is not None else None


def rbx_script_analyze(root: Any = "game", mode: str = None, max_scripts: int = 2000, sourcemap: bool = False,
                       format: bool = False) -> dict:  # noqa: A002 - the spec's own word for the formatter half
    "Roblox Studio: type check and lint every script under root with a Luau analyzer installed on this machine (luau-analyze, or MIF_LUAU_ANALYZE), each finding with its script path, line, column, kind and message. The mode is Workspace.LuauTypeCheckMode (Strict, Nonstrict, NoCheck; Default read as nonstrict) unless mode overrides it. Roblox's globals and data types are declared untyped, so Luau's own type errors are found but a wrong Instance member is not - unless sourcemap is true: the place's tree (names, classes, where each script sits) and Roblox's API from the catalog are then written as types the analyzer reads, and a second, strict pass adds what only they find, each marked source roblox-types: a member the class does not have or a child the place does not have on game, workspace, script and their children, on a service from GetService and on what Instance.new answers (both by literal name), a wrong number, string or boolean handed to an engine member. types says how many scripts that pass is proven on and names the rest; a value from FindFirstChild or WaitForChild, and a child made while the game runs, are not typed. A definitions file is not taken: luau-analyze has no option for one. format true also runs a Luau formatter installed on this machine (StyLua, or MIF_STYLUA) over copies of the scripts and answers each block it would change with its script and line; none installed answers tool-missing for that half. No analyzer installed answers tool-missing; nothing is installed for you. A place with no scripts fails. Read only."
    T = _T()
    if not isinstance(sourcemap, bool) or not isinstance(format, bool):
        return {"ok": False, "ran": False, "error": "sourcemap and format are true or false"}
    if mode is not None and mode not in MODES:
        return {"ok": False, "ran": False, "error": "mode is one of %s" % ", ".join(MODES)}
    if not isinstance(max_scripts, int) or isinstance(max_scripts, bool) or not 1 <= max_scripts <= 10000:
        return {"ok": False, "ran": False, "error": "max_scripts is 1-10000"}
    cmd = find_analyzer()
    if not cmd:
        return {"ok": False, "ran": False, "refusedRule": "tool-missing",
                "error": "no Luau analyzer on this machine: install luau-analyze (github.com/luau-lang/luau releases) and "
                         "put it on PATH or set MIF_LUAU_ANALYZE to its path. The bridge installs nothing itself "
                         "- whether to install one is the person's decision"}
    parts = [] if root in ("game", None) else _V().split_path(root)
    if parts is None:
        return T._target(root, "root")[1]
    assumed = False
    if mode is None:
        g = T.rbx_get_props("Workspace", ["LuauTypeCheckMode"])
        if not g.get("ok"):
            return g
        mode = _enum_name((g.get("values") or {}).get("LuauTypeCheckMode"))
        if mode not in MODES:
            return {"ok": False, "ran": True, "error": "Workspace.LuauTypeCheckMode read %r, not an item this tool maps" % mode}
        assumed = mode == "Default"
    r = T._rbx("script_scan", root=parts)
    if not r.get("ok"):
        return r
    sources = [s for s in r.get("sources") or [] if isinstance(s, dict) and isinstance(s.get("source"), str)]
    if not r.get("scanned") or not sources:
        return {"ok": False, "scanned": r.get("scanned", 0),
                "error": "no scripts under %s: nothing was analyzed" % (_V().join_path(parts) or "game")}
    sources = sources[:max_scripts]
    place = None
    if sourcemap:
        from mif_roblox import luau_types
        cat, err = T._catalog()
        if err:
            return err
        tree, err = _place_tree(T)
        if err:
            return err
        place = luau_types.Place(cat, tree)
    typed = fmt = None
    tmp = tempfile.mkdtemp(prefix="mif_luau_")
    try:
        files = {}
        for i, s in enumerate(sources):
            base = "s%05d.luau" % i
            files[base] = _V().join_path(s.get("path"))
            with open(os.path.join(tmp, base), "w", encoding="utf-8", newline="\n") as f:
                f.write(s["source"])
        with open(os.path.join(tmp, ".luaurc"), "w", encoding="utf-8") as f:
            json.dump({"languageMode": MODES[mode], "globals": list(ROBLOX_GLOBALS)}, f)
        t0 = time.time()
        try:
            p = subprocess.run(cmd + sorted(files), cwd=tmp, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=ANALYZE_TIMEOUT,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            return {"ok": False, "ran": True, "error": "the analyzer ran past %ss" % ANALYZE_TIMEOUT, "analyzer": cmd[-1]}
        except OSError as exc:
            return {"ok": False, "ran": False, "refusedRule": "tool-missing", "analyzer": cmd[-1],
                    "error": "the analyzer could not start: %s" % exc}
        diags, raw = parse_diagnostics((p.stdout or "") + "\n" + (p.stderr or ""), files)
        if place is not None:
            found, typed = _typed_pass(cmd, tmp, sources, files, place)
            if found is None:
                return dict(typed, analyzer=cmd[-1])
            diags = sorted(diags + found, key=lambda d: (d["script"], d["line"], d["column"]))
        if format:
            fmt = _format_pass(tmp, sources, files)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = {"ok": True, "analyzer": os.path.basename(cmd[-1]), "exitCode": p.returncode, "seconds": round(time.time() - t0, 2),
           "mode": mode, "languageMode": MODES[mode], "modeAssumed": assumed, "scanned": r.get("scanned"),
           "analyzed": len(sources), "truncated": bool(r.get("truncated")) or len(sources) < len(r.get("sources") or []),
           "count": len(diags), "clean": not diags, "diagnostics": diags,
           "byKind": {k: sum(1 for d in diags if d["kind"] == k) for k in sorted({d["kind"] for d in diags})},
           "robloxTypes": bool(typed and typed["scriptsTyped"])}
    if typed is not None:
        out["types"] = dict({"source": "catalog", "studio": (cat.studio or {}).get("version")}, **place.info(), **typed)
        if not typed["scriptsTyped"]:
            # an empty proof is a failure: the pass reported its own planted wrong member on no script
            out["types"]["error"] = ("the typed pass reported its canary on no script: this analyzer did not check "
                                     "the place's types, and the answer is the untyped one")
    else:
        out["typesNote"] = ("ran untyped: a wrong member of an Instance is not found. sourcemap true types game, "
                            "workspace and script from the place's tree and the catalog")
    if fmt is not None:
        out["format"] = fmt
    if MODES[mode] == "nonstrict":
        # Luau 0.748 measured (RB1403): nonstrict does not report a value that contradicts its own annotation
        # (local n: number = "x"); strict does. A script's own --!strict line still applies to it.
        out["modeNote"] = ("nonstrict: a value that contradicts its annotation is not reported (RB1403); mode: Strict, "
                           "or --!strict in a script, checks it")
    if raw:
        out["unparsed"] = raw[:200]
    if p.returncode == 1 and not diags:
        # Exit 1 means the analyzer found something; a format none of _DIAG reads must not answer clean.
        out.update(ok=False, clean=False, error="the analyzer exited 1 (findings): findings were printed and none was "
                                                "read: %s" % " | ".join(raw[:5]))
    elif p.returncode not in (0, 1) and not diags:
        out.update(ok=False, error="the analyzer exited %s with no finding read: %s" % (p.returncode, " | ".join(raw[:5])))
    return out


# ---------------------------------------------------------------- the script profiler
SPS = "ScriptProfilerService"


def _summary(obj, depth=0):
    """The shape of a decoded profile: keys, list lengths, scalar types."""
    if depth > 3:
        return "..."
    if isinstance(obj, dict):
        return {k: _summary(v, depth + 1) for k, v in list(obj.items())[:40]}
    if isinstance(obj, list):
        return {"list": len(obj), "first": _summary(obj[0], depth + 1) if obj else None}
    return type(obj).__name__


def source_kind(src):
    """Who a profile entry's Source belongs to, from the prefixes Studio 0.748 wrote: builtin_ and
    sabuiltin_ are Studio's own plugins, user_ the person's local plugins (MifBridge among them), [C] a native
    function, GC and "Script Context" the engine. Anything else - a dot path like Workspace.MifScratch.L2Busy - is the
    place's own script."""
    if not isinstance(src, str):
        return None
    if "builtin_" in src.split(".", 1)[0]:
        return "studio"
    if src.startswith("user_"):
        return "plugin"
    if src == "[C]":
        return "native"
    if src in ("GC",) or src.startswith("Script Context"):
        return "engine"
    return "place"


def top_functions(doc, n=10):
    """The profile's Functions by TotalDuration, largest first, and the place's own. Fields as Studio writes them (RB59,
    RB208): each entry has TotalDuration and Flags, and a Name, a Source and Line, or both. TotalDuration's unit is not
    documented and is passed on as written. An entry is the place's own when its Source is a place path, or it has only
    a Name and Flags 0 (the script's own chunk is named after the script). A local function inside a script was not
    named in the profile at 1000 Hz (RB208): the script is the finest unit seen."""
    fs = doc.get("Functions") if isinstance(doc, dict) else None
    if not isinstance(fs, list):
        return None, None
    rows = []
    for f in fs:
        if not isinstance(f, dict) or not isinstance(f.get("TotalDuration"), (int, float)):
            continue
        src = f.get("Source")
        kind = source_kind(src)
        if kind is None:
            kind = "place" if f.get("Flags") == 0 else "unknown"
        rows.append({"name": f.get("Name"), "source": src, "line": f.get("Line"), "flags": f.get("Flags"),
                     "totalDuration": f["TotalDuration"], "kind": kind})
    rows.sort(key=lambda r: -r["totalDuration"])
    user = [r for r in rows if r["kind"] == "place"]
    return rows[:n], user[:n]


def rbx_script_profile(seconds: float = 3, frequency: int = 1000, top: int = 10, timeout: float = 150,
                       folder: str = None) -> dict:
    "Roblox Studio: profile the place's scripts in a play session - rbx_play_probe kind profile, where the MifBridge plugin Studio loads into the play server runs ScriptProfilerService there (ServerStart at frequency Hz, seconds, ServerRequestData, OnNewData, ServerStop) - and answer the functions that took the most time, all and the place's own (user), each with its name or script source and line and Studio's TotalDuration. The whole JSON is saved to a file. Play runs on a copy of the place; nothing is saved. From the edit DataModel the profiler never answers (measured), which is why this runs a play test."
    T = _T()
    if not isinstance(seconds, (int, float)) or not 0.5 <= seconds <= 30:
        return {"ok": False, "ran": False, "error": "seconds is 0.5-30"}
    if not isinstance(frequency, int) or isinstance(frequency, bool) or not 1 <= frequency <= 10000:
        return {"ok": False, "ran": False, "error": "frequency is 1-10000 samples a second"}
    if not isinstance(top, int) or isinstance(top, bool) or not 1 <= top <= 100:
        return {"ok": False, "ran": False, "error": "top is 1-100"}
    if not isinstance(timeout, (int, float)) or not seconds + 30 <= timeout <= 300:
        return {"ok": False, "ran": False, "error": "timeout is at least seconds + 30 and at most 300"}
    r = T.rbx_play_probe("profile", seconds=seconds, value=frequency, timeout=timeout)
    text = r.get("profile")
    out = {"ok": False, "seconds": seconds, "frequency": frequency, "meta": r.get("profileMeta"),
           "testProcesses": r.get("testProcesses"), "requestId": r.get("requestId")}
    if not r.get("ok") or not isinstance(text, str) or not text:
        out.update(error=r.get("profileError") or r.get("error") or "the play probe returned no profile",
                   refusedRule=r.get("refusedRule"))
        return out
    try:
        doc = json.loads(text)
    except ValueError as exc:
        out.update(error="the profile is not JSON: %s" % str(exc)[:200], bytes=len(text))
        return out
    try:
        folder = folder or T.captures_dir()
        os.makedirs(folder, exist_ok=True)
        fname = os.path.join(folder, "profile_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
        with open(fname, "w", encoding="utf-8") as fh:
            fh.write(text)
        out["file"] = fname
    except OSError as exc:
        out["fileError"] = str(exc)
    allf, user = top_functions(doc, top)
    out.update(bytes=len(text), version=doc.get("Version") if isinstance(doc, dict) else None, shape=_summary(doc))
    if allf is None:
        out["error"] = "the profile has no Functions list (shape returned): its format changed"
        return out
    out.update(ok=True, top=allf, topUser=user, functions=len(doc.get("Functions") or []))
    return out


# ---------------------------------------------------------------- the panel's flagged rows
def rbx_flagged(limit: int = 50) -> dict:
    "Roblox Studio: the call rows the person flagged in the MifBridge panel, newest last, each with its note - read from the daemon's report file (<home>/flagged.jsonl, which the panel's flag button appends to through the daemon, since a plugin cannot write a file). Read only; the file is the record."
    T = _T()
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
        return {"ok": False, "ran": False, "error": "limit is 1-1000"}
    ready = T.ensure_daemon()
    if not ready.get("ok"):
        return ready
    try:
        _code, out, _b = T._http("GET", "/flagged?limit=%d" % limit)
    except (OSError, ValueError) as exc:
        return {"ok": False, "editorState": "daemonDown", "error": str(exc)}
    return out if isinstance(out, dict) else {"ok": False, "error": "the daemon answered %r" % (out,)}


# ---------------------------------------------------------------- Open Cloud Luau Execution
# create.roblox.com/docs/en-us/cloud/reference/features/luau-execution (read 2026-11-18): POST
# /cloud/v2/universes/{u}/places/{p}[/versions/{v}]/luau-execution-session-tasks with {script, timeout}; the task's
# `path` is polled with GET /cloud/v2/{path} until state COMPLETE, FAILED or CANCELLED; logs at {path}/logs. Scopes
# universe.place.luau-execution-session:write (create) and :read (get, logs). 5 creates a minute, 5-minute tasks, 4 MB
# scripts, ten unfinished tasks a place.
CLOUD_DONE = ("COMPLETE", "FAILED", "CANCELLED")
CLOUD_POLL_EVERY = 2.0  # get-task allows 200 a minute
MAX_SCRIPT = 4 * 1024 * 1024
# Decision 5: a write to live data is refused unless the call says allow_live_write. A cloud task
# runs against the published place's real data stores, so a script that names a live-data service or a write method is
# refused without it. Players:BanAsync and UnbanAsync change the experience's real ban list and Player:Kick removes a
# live player, so they are named too. A text guard against accidents, not a sandbox: the person's key is the authority.
LIVE_WRITE = re.compile(r"DataStoreService|MemoryStoreService|MessagingService|:\s*(?:SetAsync|UpdateAsync|RemoveAsync|"
                        r"IncrementAsync|PublishAsync|BanAsync|UnbanAsync|Kick)\b")
CLOUD_SCOPE_HINT = ("the key needs universe.place.luau-execution-session:write (create) and "
                    "universe.place.luau-execution-session:read (get, logs) for this experience")


def _oc():
    from mif_roblox import opencloud
    return opencloud


def _cloud_read(OC, key, task_path, poll_timeout, poll_every):
    """Poll one task to the end, then its logs. (dict) - never raises."""
    t0 = time.time()
    polls, task, waited = 0, None, 0.0
    while True:
        st, task, w = OC._call("GET", "/cloud/v2/" + task_path, key)
        waited += w
        polls += 1
        if st != 200 or not isinstance(task, dict):
            return {"ok": False, "ran": True, "outcomeUnknown": True, "taskPath": task_path, "polls": polls,
                    "status": st, "body": task,
                    "error": "reading %s answered %s%s" % (task_path, st, "; " + CLOUD_SCOPE_HINT if st in (401, 403) else "")}
        if task.get("state") in CLOUD_DONE or time.time() - t0 >= poll_timeout:
            break
        time.sleep(poll_every)
    out = {"taskPath": task_path, "state": task.get("state"), "polls": polls, "rateLimitWaitSeconds": waited,
           "results": (task.get("output") or {}).get("results"), "taskError": task.get("error"),
           "createTime": task.get("createTime"), "updateTime": task.get("updateTime")}
    if task.get("state") not in CLOUD_DONE:
        out.update(ok=False, ran=True, outcomeUnknown=True,
                   error="the task was %s after %ss; read it again with task_path" % (task.get("state"), poll_timeout))
        return out
    lst, logs, _w = OC._call("GET", "/cloud/v2/" + task_path + "/logs?view=STRUCTURED", key)
    messages = []
    if lst == 200 and isinstance(logs, dict):
        for page in logs.get("luauExecutionSessionTaskLogs") or []:
            for m in page.get("structuredMessages") or []:
                messages.append({"type": m.get("messageType"), "message": m.get("message"), "time": m.get("createTime")})
            if not page.get("structuredMessages"):
                messages.extend({"type": None, "message": x} for x in page.get("messages") or [])
        out["logsMore"] = bool(logs.get("nextPageToken"))
    else:
        out["logsError"] = {"status": lst, "body": logs}
    out["logs"] = messages
    out["ok"] = task.get("state") == "COMPLETE"
    out["ran"] = True
    if not out["ok"]:
        err = task.get("error") or {}
        out["error"] = "the task ended %s: %s %s" % (task.get("state"), err.get("code"), err.get("message") or "")
    return out


def rbx_cloud_luau(universe_id: int = None, place_id: int = None, script: str = None, version_id: int = None,
                   timeout_seconds: int = 60, confirm: bool = False, allow_live_write: bool = False,
                   task_path: str = None, poll_timeout: float = 330, allow_ids: list = None) -> dict:
    "Roblox Open Cloud: run a Luau script on Roblox's servers against a PUBLISHED place (its latest version, or version_id) with the person's own API key, and return the task's state, its return values (output.results), its error and its Output log. Not Studio, and never inside the plugin. confirm is required: it runs against the published experience and counts against Roblox's 5 creates a minute. A script naming DataStoreService, MemoryStoreService, MessagingService or a SetAsync/UpdateAsync/RemoveAsync/IncrementAsync/PublishAsync call is refused unless allow_live_write. The universe and place must be on the person's write-target list (roblox_write_targets.json) or named in allow_ids (refusedRule write-target): the key's scopes usually cover every experience. With task_path, reads an earlier task again instead of creating one. The key is never returned."
    OC = _oc()
    if not isinstance(poll_timeout, (int, float)) or not 5 <= poll_timeout <= 900:
        return {"ok": False, "ran": False, "error": "poll_timeout is 5-900 seconds"}
    if task_path is not None:
        if not isinstance(task_path, str) or not re.fullmatch(r"universes/\d+/places/\d+(/versions/\d+)?/"
                                                              r"luau-execution-sessions/[\w-]+/tasks/[\w-]+", task_path):
            return {"ok": False, "ran": False, "error": "task_path is the path a create answered: universes/<u>/places/"
                                                        "<p>/versions/<v>/luau-execution-sessions/<s>/tasks/<t>"}
        key, err = OC.read_key()
        if err:
            return err
        try:
            return OC.redact(_cloud_read(OC, key, task_path, poll_timeout, CLOUD_POLL_EVERY), key)
        except Exception as exc:  # noqa: BLE001 - never raise, never let a text carry the key
            return OC.redact({"ok": False, "outcomeUnknown": True, "error": "reading the task failed: %s" % OC.exc_text(exc, key)}, key)
    ids = {"universe_id": universe_id, "place_id": place_id}
    if version_id is not None:
        ids["version_id"] = version_id
    for k, v in ids.items():
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            return {"ok": False, "ran": False, "error": "%s is a positive whole number (the published experience's)" % k}
    if not isinstance(script, str) or not script.strip():
        return {"ok": False, "ran": False, "error": "script is the Luau source to run"}
    if len(script.encode("utf-8")) > MAX_SCRIPT:
        return {"ok": False, "ran": False, "refusedRule": "size", "error": "a script is at most 4 MB"}
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or not 1 <= timeout_seconds <= 300:
        return {"ok": False, "ran": False, "error": "timeout_seconds is 1-300 (Roblox's limit is 5 minutes)"}
    m = LIVE_WRITE.search(script)
    if m and allow_live_write is not True:
        return {"ok": False, "ran": False, "refusedRule": "live-write", "matched": m.group(0),
                "error": "the script names %r: a cloud task runs against the published experience's live data. Pass "
                         "allow_live_write true only when writing live data is intended" % m.group(0)}
    if confirm is not True:
        return {"ok": False, "ran": False, "refusedRule": "needs-confirm",
                "error": "this runs on Roblox's servers against the published place %s of experience %s; pass confirm true"
                         % (place_id, universe_id)}
    gate = OC.write_target_check(universe_id, place_id, allow_ids)
    if gate:
        return gate
    key, err = OC.read_key()
    if err:
        return err
    path = "/cloud/v2/universes/%d/places/%d" % (universe_id, place_id)
    if version_id is not None:
        path += "/versions/%d" % version_id
    path += "/luau-execution-session-tasks"
    try:
        body = json.dumps({"script": script, "timeout": "%ds" % timeout_seconds}).encode("utf-8")
        t0 = time.time()
        # retried on 429 only (_call): a create that got no answer or a 5xx may have started the script, and a second
        # send would run it twice against the published place (found 2026-11-22)
        st, created, waited = OC._call("POST", path, key, body, "application/json", timeout=60, tries=3)
        if OC.outcome_unknown(st):
            return OC.redact({"ok": False, "ran": None, "outcomeUnknown": True, "createStatus": st, "body": created,
                              "error": "POST %s answered %s (no answer, or a server error): the task may have been made "
                                       "and the script may be running. It was sent once and not again; read the place's "
                                       "Luau tasks before running it again" % (path, st or "nothing")}, key)
        if st != 200 or not isinstance(created, dict) or not created.get("path"):
            hint = "; " + CLOUD_SCOPE_HINT if st in (401, 403) else ""
            return OC.redact({"ok": False, "ran": st != 0, "createStatus": st, "body": created,
                              "error": "POST %s answered %s%s" % (path, st, hint)}, key)
        out = _cloud_read(OC, key, created["path"], poll_timeout, CLOUD_POLL_EVERY)
        out.update(createStatus=st, createRateLimitWaitSeconds=waited, seconds=round(time.time() - t0, 2),
                   universeId=universe_id, placeId=place_id, versionId=version_id)
        return OC.redact(out, key)
    except Exception as exc:  # noqa: BLE001
        return OC.redact({"ok": False, "ran": None, "outcomeUnknown": True, "error": "cloud luau failed: %s" % OC.exc_text(exc, key)}, key)
