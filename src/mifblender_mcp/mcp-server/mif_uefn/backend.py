"""What the uefn_* MCP tools do: status with setup guidance, install, one listener op with the ownership check, the
Blender hand-off. Transport and the TCP-table reader are injected, so everything here tests offline.
"""
import hashlib
import json
import os
import re
import secrets
import socket
import time

from . import epic as _epic
from .listener import mifbridge_uefn_catalog as CAT
from .listener import mifbridge_uefn_win32 as W32

HERE = os.path.dirname(os.path.abspath(__file__))
LISTENER_DIR = os.path.join(HERE, "listener")
LISTENER_FILE = os.path.join(LISTENER_DIR, "mifbridge_uefn_listener.py")
PANEL_FILE = os.path.join(LISTENER_DIR, "mifbridge_uefn_panel.py")
UEFN_IMAGE_PREFIX = "UnrealEditorFortnite"
PYTHON_TOOLSET = "editor_toolset.toolsets.python_scripting.PythonScriptingTools"
VALKYRIE_PYTHON = "ValkyrieToolset.ValkyriePythonToolset"
DEFAULT_LISTENER_PORT = 8930
DEFAULT_EPIC_PORT = 8000
# How long one op may take before the caller is told BUSY. A session start waits for the client's content (measured
# 4-5 minutes on a cold client); an import compiles shaders; the rest are editor reads and writes.
OP_TIMEOUTS = {"session_start": 1900.0, "session_end": 600.0, "import_static": 600.0, "import_skeletal": 900.0,
               "verse_build": 300.0, "capture": 120.0, "create_prefab": 180.0}
DEFAULT_OP_TIMEOUT = 120.0
# What uefn_describe_tool says each verify route does (mifbridge_uefn_catalog.verify_route decides the route).
VERIFY_TEXT = {
    "getter": "; the write is read back through its Get tool and compared",
    "target": "; the object it names is read before and after (UEFN's property reader and the actor API): a sent value "
              "that is a property is compared, every change is listed, and no change seen is answered verified: false",
    "result": "; what it made is read back (exists, class)",
    "asset": "; the content paths it names are checked before and after; only a change in existence is seen",
    "none": "; UEFN offers nothing to read its effect back through, so every answer says verified: false",
}

ENABLE_MCP = ("Turn on Epic's UEFN MCP server: in UEFN, Edit > Editor Preferences > Agentic Workflows > Model Context "
              "Protocol, then Start Server. Its port is the 'Server Port Number' there (default 8000); if it is not "
              "8000, set MIF_UEFN_MCP_PORT to it in the MCP server's environment.")
START_UEFN = ("Start UEFN (Unreal Editor for Fortnite) from the Epic Games Launcher and open the project. MifBridge "
              "never starts or closes UEFN: it needs the launcher's sign-in.")
ENABLE_PYTHON = ("Python is not enabled for this UEFN project, and MifBridge's listener runs in UEFN's Python. Enable "
                 "Python for the project in UEFN. Epic's MCP tool EnablePythonForMainProject (toolset "
                 "ValkyrieToolset.ValkyriePythonToolset) does the same, but it also lets the project's own Python "
                 "startup scripts run, so call it only when the person has asked for exactly that; no MifBridge tool "
                 "calls it.")


def _env_port(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        port = int(raw)
    except ValueError:
        return default
    return port if 1 <= port <= 65535 else default


def state_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    d = os.path.join(base, "MifBridge", "uefn")
    os.makedirs(d, exist_ok=True)
    return d


def token_file():
    return os.path.join(state_dir(), "token")


def read_or_make_token():
    path = token_file()
    try:
        with open(path, encoding="utf-8") as fh:
            tok = fh.read().strip()
        if re.fullmatch(r"[0-9a-f]{32,}", tok):
            return tok
    except OSError:
        pass
    tok = secrets.token_hex(16)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(tok)
    return tok


def file_sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha1(fh.read()).hexdigest()


class UefnError(Exception):
    def __init__(self, kind, message, **extra):
        super().__init__(message)
        self.kind = kind
        self.extra = extra

    def reply(self):
        return dict({"ok": False, "error": str(self), "kind": self.kind}, **self.extra)


class Backend:
    def __init__(self, listening_pid, image_of=W32.process_image, uefn_processes=None, epic_factory=None,
                 connect=None):
        self.listening_pid = listening_pid
        self.image_of = image_of
        self.uefn_processes = uefn_processes or (lambda: W32.processes(UEFN_IMAGE_PREFIX))
        self.epic_factory = epic_factory or (lambda port: _epic.EpicMCP(port))
        self.connect = connect or (lambda port, timeout: socket.create_connection(("127.0.0.1", port),
                                                                                    timeout=timeout))

    @property
    def port(self):
        return _env_port("MIF_UEFN_PORT", DEFAULT_LISTENER_PORT)

    @property
    def epic_port(self):
        return _env_port("MIF_UEFN_MCP_PORT", DEFAULT_EPIC_PORT)

    # ------------------------------------------------------------ ownership - a measurement, every call
    def _owner(self, port):
        try:
            pid = self.listening_pid(port)
        except Exception:  # noqa: BLE001 - not Windows, or the table could not be read
            pid = None
        image = self.image_of(pid) if pid else None
        return pid, image

    @staticmethod
    def _is_uefn(image):
        return bool(image) and os.path.basename(image).lower().startswith(UEFN_IMAGE_PREFIX.lower())

    def ownership(self):
        epid, eimg = self._owner(self.epic_port)
        lpid, limg = self._owner(self.port)
        return {"epicPort": self.epic_port, "epicPid": epid, "epicImage": eimg, "epicIsUefn": self._is_uefn(eimg),
                "listenerPort": self.port, "listenerPid": lpid, "listenerImage": limg,
                "listenerIsUefn": self._is_uefn(limg)}

    def _require_listener(self):
        own = self.ownership()
        if not own["listenerPid"]:
            raise UefnError("not_installed", "nothing listens on 127.0.0.1:%d - MifBridge's listener is not running "
                            "in UEFN (it goes with the editor process). Call uefn_install." % self.port,
                            ownership=own)
        if not own["listenerIsUefn"]:
            raise UefnError("port_taken", "127.0.0.1:%d is held by pid %s (%s), not by UEFN. Nothing was sent. Set "
                            "MIF_UEFN_PORT to a free port and call uefn_install." % (
                                self.port, own["listenerPid"], own["listenerImage"] or "image unreadable"),
                            ownership=own)
        return own

    # ------------------------------------------------------------ the listener
    def _send(self, op, args, timeout):
        token = read_or_make_token()
        line = (json.dumps({"token": token, "op": op, "args": args}) + "\n").encode("utf-8")
        t0 = time.perf_counter()
        try:
            s = self.connect(self.port, timeout)
        except (ConnectionRefusedError, OSError) as e:
            raise UefnError("down", "could not connect to the listener on 127.0.0.1:%d (%s). Call uefn_install."
                            % (self.port, e))
        try:
            s.settimeout(timeout)
            s.sendall(line)
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(1 << 16)
                if not chunk:
                    break
                buf += chunk
        except (socket.timeout, TimeoutError):
            raise UefnError("busy", "UEFN accepted %s and did not answer in %.0f s. UEFN may be hitching or hung (a "
                            "known UEFN issue with tool calls). Nothing was retried and the op may still complete: "
                            "read back before repeating it." % (op, timeout))
        finally:
            try:
                s.close()
            except OSError:
                pass
        try:
            res = json.loads(buf)
        except ValueError:
            raise UefnError("protocol", "the listener's reply to %s is not JSON (%d bytes)" % (op, len(buf)))
        res["clientMs"] = round((time.perf_counter() - t0) * 1000, 1)
        return res

    def op(self, op, args=None, timeout=None):
        """One listener op, after the ownership check; the reply's pid must be the port's owner."""
        try:
            own = self._require_listener()
            res = self._send(op, args or {}, timeout or OP_TIMEOUTS.get(op, DEFAULT_OP_TIMEOUT))
        except UefnError as e:
            return e.reply()
        if res.get("refused") == "bad token":
            return {"ok": False, "kind": "bad_token", "error": "the listener refused this server's token - it was "
                    "installed with another token file. Call uefn_install."}
        if res.get("pid") != own["listenerPid"]:
            return {"ok": False, "kind": "pid_mismatch", "error": "the reply came from pid %s but 127.0.0.1:%d is "
                    "owned by pid %s; the result is not trusted" % (res.get("pid"), self.port, own["listenerPid"])}
        return res

    def run(self, op, args=None):
        """op() with None-valued arguments left out and the reply flattened: {ok, <result fields>, refused?, error?,
        mismatch?, ms}. A refusal or error carries no result fields."""
        return flatten(self.op(op, {k: v for k, v in (args or {}).items() if v is not None}))

    # ------------------------------------------------------------ Epic's whole tool surface, through the catalog
    def find_catalog(self, query="", toolset="", mode="", limit=25):
        """Search the catalog (no UEFN needed): every word must appear in the tool's name, toolset or summary."""
        cat = CAT.load()
        words = [w for w in re.split(r"[\s_\-/,.;:]+", (query or "").lower())
                 if w and w not in ("uefn", "fortnite", "tool", "tools")]
        rows = []
        for full, r in cat["tools"].items():
            if r.get("deny") or (toolset and toolset.lower() not in r["toolset"].lower()) or (mode and r["mode"] != mode):
                continue
            hay = ("%s %s" % (full, r["summary"])).lower()
            if all(w in hay for w in words):
                score = sum(3 for w in words if w in r["name"].lower())
                rows.append((-score, full, r))
        rows.sort(key=lambda x: (x[0], x[1]))
        lim = max(1, min(int(limit or 25), 200))
        return {"ok": True, "uefn": cat.get("uefn"), "count": len(rows),
                "tools": [{"tool": full, "mode": r["mode"], "verify": r.get("verify"), "summary": r["summary"][:160]} for _s, full, r in rows[:lim]],
                "call": "uefn_call(tool, args); uefn_describe_tool(tool) gives its parameters"}

    def describe_catalog(self, name):
        cat = CAT.load()
        r = cat["tools"].get(name)
        if r is None:
            near = self.find_catalog(name.rsplit(".", 1)[-1], limit=8)["tools"]
            return {"ok": False, "error": "no UEFN tool %r in the catalog (UEFN %s)" % (name, cat.get("uefn")),
                    "didYouMean": [t["tool"] for t in near]}
        return {"ok": True, "tool": name, "toolset": r["toolset"], "mode": r["mode"], "epicAnnotated": r["annotated"],
                "readBackThrough": r.get("getter"), "verify": r.get("verify"), "deny": r.get("deny"), "description": r["description"],
                "parameters": r["inputSchema"], "uefn": cat.get("uefn"),
                "guarantees": "unknown keys refused at every depth, JSON types checked (no string for a number), "
                              "enums and bounds checked, vectors whole or filled from the getter's current value"
                              + VERIFY_TEXT.get(r.get("verify"), "")}

    def call_catalog(self, name, args):
        cat = CAT.load()
        r = cat["tools"].get(name)
        if r is not None and r.get("deny"):           # refused before anything is sent, UEFN or not
            return {"ok": False, "refused": "%s is not reachable through MifBridge: %s" % (name, r["deny"])}
        if r is not None:
            try:
                CAT.validate(json.loads(json.dumps(args if args is not None else {})), r["inputSchema"], "args",
                             partial_vectors=[] if r.get("getter") else None)
            except CAT.Invalid as e:
                return {"ok": False, "refused": str(e), "tool": name}
        return self.run("call", {"tool": name, "args": args if args is not None else {}})

    # ------------------------------------------------------------ status
    def status(self, log=0):
        own = self.ownership()
        procs = [{"pid": p, "image": n} for p, n, _ in self.uefn_processes()]
        out = {"ok": True, "preview": True, "ownership": own, "uefnProcesses": procs, "epicMcp": None,
               "pythonReady": None, "listener": None, "setup": []}
        if own["epicPid"] and own["epicIsUefn"]:
            try:
                m = self.epic_factory(self.epic_port)
                info = m.initialize()
                out["epicMcp"] = {"answering": True, "protocol": info.get("protocolVersion")}
                _t, st = m.call(VALKYRIE_PYTHON, "IsPythonReadyForMainProject", {}, timeout=30)
                out["pythonReady"] = bool(_epic.return_value(st))
            except _epic.EpicError as e:
                out["epicMcp"] = {"answering": False, "kind": e.kind, "error": str(e)}
        elif own["epicPid"]:
            out["epicMcp"] = {"answering": None, "error": "port %d is held by pid %s (%s), not UEFN" % (
                self.epic_port, own["epicPid"], own["epicImage"])}
        if not procs and not own["epicPid"] and not own["listenerPid"]:
            out["setup"].append(START_UEFN)
        elif not own["epicPid"] or not own["epicIsUefn"]:
            out["setup"].append(ENABLE_MCP)
        if out["pythonReady"] is False:
            out["setup"].append(ENABLE_PYTHON)
        if own["listenerPid"] and own["listenerIsUefn"]:
            ping = self.op("ping")
            if ping.get("ok"):
                r = ping["result"]
                stale = r.get("sourceSha") != file_sha(LISTENER_FILE)
                out["listener"] = {"running": True, "pid": r.get("pid"), "version": r.get("version"),
                                   "stale": stale, "writeMode": r.get("writeMode"), "level": r.get("level"),
                                   "contentRoot": r.get("contentRoot"), "engine": r.get("engine"),
                                   "panelOpen": r.get("panel"), "served": r.get("served"),
                                   "refused": r.get("refused"), "failed": r.get("failed"),
                                   "endpoints": len(r["ops"]) if isinstance(r.get("ops"), dict) else None,
                                   "epicTools": r.get("epicTools"), "pingMs": ping.get("ms"),
                                   "clientMs": ping.get("clientMs")}
                if stale:
                    out["setup"].append("The listener running in UEFN is not this server's copy (an older or newer "
                                        "MifBridge). Call uefn_install to refresh it.")
                if log:
                    lg = self.op("log", {"n": int(log)})
                    out["log"] = lg.get("result") if lg.get("ok") else lg
            else:
                out["listener"] = {"running": False, "error": ping.get("error") or ping.get("refused"),
                                   "kind": ping.get("kind")}
                out["setup"].append("The listener did not answer: call uefn_install.")
        elif own["listenerPid"]:
            out["listener"] = {"running": False, "error": "port %d is held by pid %s (%s)" % (
                self.port, own["listenerPid"], own["listenerImage"])}
            out["setup"].append("Port %d is taken by another program: set MIF_UEFN_PORT to a free port." % self.port)
        elif own["epicIsUefn"]:
            out["listener"] = {"running": False}
            out["setup"].append("MifBridge's listener is not running in UEFN (it is installed once per UEFN start): "
                                "call uefn_install.")
        out["ready"] = bool(out["listener"] and out["listener"].get("running") and not out["listener"].get("stale"))
        return out

    # ------------------------------------------------------------ install
    def install_script(self, open_panel):
        """The execute_python script. It names the token FILE, never the token."""
        return INSTALL_SCRIPT % {"listener": LISTENER_FILE.replace("\\", "/"), "panel": PANEL_FILE.replace("\\", "/"),
                                 "port": self.port, "token_file": token_file().replace("\\", "/"),
                                 "state_dir": state_dir().replace("\\", "/"), "open_panel": bool(open_panel)}

    def install(self, open_panel=True):
        own = self.ownership()
        if not own["epicPid"]:
            procs = self.uefn_processes()
            return {"ok": False, "kind": "epic_mcp_off" if procs else "uefn_closed",
                    "error": "nothing listens on Epic's UEFN MCP port %d" % self.epic_port,
                    "setup": [ENABLE_MCP if procs else START_UEFN]}
        if not own["epicIsUefn"]:
            return {"ok": False, "kind": "port_taken", "error": "port %d is held by pid %s (%s), not UEFN; nothing "
                    "was sent" % (self.epic_port, own["epicPid"], own["epicImage"])}
        if own["listenerPid"] and own["listenerPid"] != own["epicPid"]:
            return {"ok": False, "kind": "port_taken", "error": "port %d is held by pid %s (%s); set MIF_UEFN_PORT to "
                    "a free port" % (self.port, own["listenerPid"], own["listenerImage"])}
        read_or_make_token()
        m = self.epic_factory(self.epic_port)
        try:
            m.initialize()
            _t, st = m.call(VALKYRIE_PYTHON, "IsPythonReadyForMainProject", {}, timeout=30)
            if not _epic.return_value(st):
                return {"ok": False, "kind": "python_off", "error": "Python is not ready for this project",
                        "setup": [ENABLE_PYTHON]}
            text, st = m.call(PYTHON_TOOLSET, "execute_python",
                              {"script": {"content": self.install_script(open_panel), "language": "Python"}},
                              timeout=90)
        except _epic.EpicError as e:
            return {"ok": False, "kind": e.kind, "error": str(e)}
        printed = _epic.return_value(st) if st is not None else text
        found = re.search(r"MIFUEFN:(\{.*\})", printed or "")
        if not found:
            return {"ok": False, "kind": "install_failed", "error": "the install script printed no result",
                    "output": (printed or "")[-1500:]}
        got = json.loads(found.group(1))
        if got.get("error"):
            return {"ok": False, "kind": "install_failed", "error": got["error"]}
        # READ-BACK: the port's owner, the reply's pid and the source the listener loaded.
        after = self.ownership()
        ping = self.op("ping")
        bad = []
        if after["listenerPid"] != own["epicPid"]:
            bad.append("port %d is owned by pid %s after install, not UEFN's %s" % (self.port, after["listenerPid"],
                                                                                   own["epicPid"]))
        if not ping.get("ok"):
            bad.append("the listener does not answer: %s" % (ping.get("error") or ping.get("refused")))
        elif ping["result"].get("sourceSha") != file_sha(LISTENER_FILE):
            bad.append("UEFN loaded a different listener source than this server's copy")
        return {"ok": not bad, "pid": own["epicPid"], "port": self.port, "version": got.get("version"),
                "panel": got.get("panel"), "logCarried": got.get("carried"), "mismatch": bad,
                "level": (ping.get("result") or {}).get("level"),
                "contentRoot": (ping.get("result") or {}).get("contentRoot")}

    # ------------------------------------------------------------ the Blender hand-off
    def import_from_blender(self, check, folder, name="", skeleton="", animations=None, collision=""):
        plan = plan_blender_import(check, folder, name, skeleton, animations, collision)
        if not plan.get("ok"):
            return plan
        res = self.op(plan["op"], plan["args"])
        if not res.get("ok") and res.get("result") is None:
            return flatten(res)
        return flatten(compare_blender_import(plan, res))


def flatten(res):
    out = {"ok": bool(res.get("ok"))}
    r = res.get("result")
    if isinstance(r, dict):
        out.update(r)
    elif r is not None:
        out["result"] = r
    for k in ("refused", "error", "kind", "ms", "ownership", "setup"):
        if res.get(k) is not None:
            out[k] = res[k]
    return out


def plan_blender_import(check, folder, name="", skeleton="", animations=None, collision=""):
    """From bl_uefn_check's answer to the listener op that imports its file. Refuses a check that did not pass, wrote
    no file, or mixes skinned and static meshes (they import as different asset types)."""
    if not isinstance(check, dict):
        return {"ok": False, "error": "check must be bl_uefn_check's whole answer (an object)"}
    if check.get("passed") is not True:
        fails = [c.get("check") for c in check.get("checks") or [] if isinstance(c, dict)
                 and c.get("severity") == "fail"]
        return {"ok": False, "error": "bl_uefn_check did not pass (%s): fix those first, nothing was imported"
                % (", ".join(map(str, fails)) or "passed is not true")}
    exp = check.get("export") or {}
    path = exp.get("file") if isinstance(exp, dict) else None
    if not path:
        return {"ok": False, "error": "bl_uefn_check wrote no file - call it with file (.fbx, .glb or .gltf)"}
    if not os.path.isfile(path):
        return {"ok": False, "error": "the file bl_uefn_check wrote is not on this machine: %s" % path}
    meshes = [m for m in check.get("meshes") or [] if isinstance(m, dict)]
    if not meshes:
        return {"ok": False, "error": "check has no meshes list - pass bl_uefn_check's answer unchanged"}
    skinned = {bool(m.get("skinned")) for m in meshes}
    if len(skinned) > 1:
        return {"ok": False, "error": "the file mixes skinned and static meshes; UEFN imports them as different "
                "asset types - export them separately"}
    kind = "skeletal" if skinned.pop() else "static"
    asset = name or re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(os.path.basename(path))[0])
    if not re.fullmatch(r"[A-Za-z0-9_]+", asset):
        return {"ok": False, "error": "name must be English letters, digits and underscores"}
    lod0 = [(m.get("lods") or [{}])[0] for m in meshes]
    expect = {"meshes": len(meshes), "lod0Triangles": sum(int(r.get("triangles") or 0) for r in lod0),
              "lods": max(len(m.get("lods") or []) for m in meshes)}
    if kind == "static":
        if skeleton or animations:
            return {"ok": False, "error": "skeleton and animations are for a skinned mesh; this file is static"}
        args = {"file": path, "folder": folder, "name": asset}
        if collision:
            args["collision"] = collision
        return {"ok": True, "op": "import_static", "args": args, "kind": kind, "expect": expect}
    if collision:
        return {"ok": False, "error": "collision is a static mesh setting; this file is skinned"}
    args = {"file": path, "folder": folder, "name": asset}
    if skeleton:
        args["skeleton"] = skeleton
    # uefn_check writes each action that keys the rig into its own FBX beside the mesh (check["animations"], one row
    # per action); those are the clips to import unless the caller names others. A glTF carries them inside its one
    # file, which is not an animation-only FBX, so they are not listed here.
    from_check = [a.get("file") for a in check.get("animations") or [] if isinstance(a, dict)
                  and isinstance(a.get("file"), str) and a["file"].lower().endswith(".fbx")
                  and os.path.normcase(a["file"]) != os.path.normcase(path)]
    clips = list(animations) if animations else from_check
    if clips:
        args["animations"] = clips
    expect["animations"] = len(clips)
    plan = {"ok": True, "op": "import_skeletal", "args": args, "kind": kind, "expect": expect,
            "animationsFrom": "the caller" if animations else "uefn_check" if from_check else None}
    lods = (exp.get("skinnedLodFiles") if isinstance(exp, dict) else None) or None
    if lods:
        plan["notImported"] = {"skinnedLodFiles": lods, "why": "UEFN has no tool that imports a skeletal LOD from a "
                                                                "file; UEFN builds LODs itself"}
    return plan


def compare_blender_import(plan, res):
    """Read-back against what Blender wrote: the LOD count of the chain the check judged, and LOD0's triangles."""
    out = dict(res)
    r = res.get("result") or {}
    exp = plan["expect"]
    bad = list(r.get("mismatch") or [])
    if plan["kind"] == "static":
        if exp["lods"] > 1 and r.get("lods") != exp["lods"]:
            bad.append("LODs: Blender's chain has %d, UEFN holds %s" % (exp["lods"], r.get("lods")))
        tri = r.get("lod0Triangles")
        if exp["lod0Triangles"] and isinstance(tri, (int, float)) and \
                abs(tri - exp["lod0Triangles"]) > max(2, 0.005 * exp["lod0Triangles"]):
            bad.append("LOD0 triangles: Blender wrote %d, UEFN holds %d" % (exp["lod0Triangles"], tri))
    if plan["kind"] == "skeletal" and exp.get("animations") is not None:
        got_clips = [a for a in r.get("animations") or [] if a.get("asset")]
        if len(got_clips) != exp["animations"]:
            bad.append("animations: Blender wrote %d clip(s), UEFN holds %d" % (exp["animations"], len(got_clips)))
    if r:
        r = dict(r, mismatch=bad, expectedFromBlender=exp, assetKind=plan["kind"],
                 animationsFrom=plan.get("animationsFrom"), notImported=plan.get("notImported"))
        out["result"] = r
        out["ok"] = not bad and not res.get("refused") and not res.get("error")
    return out


INSTALL_SCRIPT = r'''import importlib.util
import json
import sys
MIF = {"listener": %(listener)r, "panel": %(panel)r, "port": %(port)d, "token_file": %(token_file)r,
       "state_dir": %(state_dir)r, "open_panel": %(open_panel)r}
out = {}
try:
    old = sys.modules.get("mifbridge_uefn_listener")
    carried = []
    if old is not None:
        await old.stop()
        carried = list(old.CALL_LOG)
    for _name in [n for n in sys.modules if n in ("mifbridge_uefn_win32", "mifbridge_uefn_catalog")]:
        sys.modules.pop(_name)
    spec = importlib.util.spec_from_file_location("mifbridge_uefn_listener", MIF["listener"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.CALL_LOG.extend(carried)
    sys.modules["mifbridge_uefn_listener"] = mod
    out = await mod.start(MIF["port"], MIF["token_file"], MIF["state_dir"])
    out["carried"] = len(carried)
    oldp = sys.modules.get("mifbridge_uefn_panel")
    was_open = bool(oldp and oldp.is_open())
    geom = oldp.STATE.get("geometry") if oldp else None
    if oldp is not None:
        oldp.close_panel()
    pspec = importlib.util.spec_from_file_location("mifbridge_uefn_panel", MIF["panel"])
    pmod = importlib.util.module_from_spec(pspec)
    pspec.loader.exec_module(pmod)
    sys.modules["mifbridge_uefn_panel"] = pmod
    if MIF["open_panel"] or was_open:
        out["panel"] = pmod.open_panel(geom)
    else:
        out["panel"] = None
except Exception as exc:
    out = {"error": "%%s: %%s" %% (type(exc).__name__, exc)}
print("MIFUEFN:" + json.dumps(out, default=str))
'''
