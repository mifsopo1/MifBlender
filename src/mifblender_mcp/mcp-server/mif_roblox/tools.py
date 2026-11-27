"""The Roblox backend's MCP side: the rbx_* tools, _rbx() - the one choke point to the daemon - and TOOLS and PACK.

A sibling of server.py, loaded in its main() and never at import, every tool held back, as the Unity backend is: an Unreal
or Blender session's tool list does not move. No import from server.py; server.py hands in its TCP-table reader
(LISTENING_PID), and a suite hands in a fake.

What happens on the MCP side, before anything reaches Studio:
- every class and member is checked against the catalog: a member a plugin cannot use is refused with its own security
  tag, a read-only one for a write, a NotCreatable class for a create, an unknown name with did-you-mean;
- every value is encoded by the property's catalog type, so a wrong shape is refused with the shape it needed;
- every write's answer is judged against what was sent: verified true only when the read-back holds it.

NEVER RAISE: every failure is {"ok": False, ...}.
"""
import functools
import hashlib
import http.client
import inspect
import json
import math
import os
import re
import socket
import sys
import threading
import time
import uuid
import zlib
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
from mif_roblox import catalog as C  # noqa: E402
from mif_roblox import daemon as D  # noqa: E402
from mif_roblox import values as V  # noqa: E402

PORT = int(os.environ.get("MIF_ROBLOX_PORT", D.DEFAULT_PORT))
# Whether the port was chosen. Unset means the default port, which every Studio without a MifBridgePort marker polls:
# a script that forgets it drives whichever Studio connects there (found 2026-11-23: one took the default port during
# another session's build and 14 of that build's steps landed wrong).
PORT_EXPLICIT = "MIF_ROBLOX_PORT" in os.environ
TIMEOUT = float(os.environ.get("MIF_ROBLOX_TIMEOUT", "60"))
STATUS_TIMEOUT = 3.0
PLUGIN_FILE = os.path.normpath(os.path.join(HERE, "..", "..", "roblox-plugin", "MifBridge.server.lua"))
# The ops the plugin's table must hold, the same names in the same order; the offline suite compares the two.
OPS = ("ping", "resolve", "find", "tree", "create", "clone", "destroy", "reparent", "get_props", "set_props",
       "selection_get", "selection_set", "undo_begin", "undo_end", "script_read", "script_write", "console",
       "panel_snapshot", "screenshot", "call", "watch_start", "watch_read", "query", "undo", "redo", "batch",
       "release_handle", "script_scan", "stats", "script_docs", "play_probe", "run_mode", "coverage_run",
       "debugger", "mesh_build", "set_mode", "camera", "step_physics", "voxels_read", "voxels_write",
       "read_many", "person_prompt", "module_require")
SCRIPT_BASE = "LuaSourceContainer"

# Injected by server.py (its _listening_pid) or a suite. A port's owner is a measurement; without it the daemon's own
# word is all there is, and the answer says so.
LISTENING_PID = None
_STATE = {"daemon": None, "studio": None, "warned": False}   # studio: the Studio pid rbx_launch confirmed for PORT


def setup_steps():
    return [
        "Install Roblox Studio and sign in.",
        "Copy %s into Studio's local plugins folder (Studio: Plugins tab > Plugins Folder), then restart Studio." % PLUGIN_FILE,
        "Open a place (a new Baseplate is fine for a first run).",
        "Press MifBridge > Connect in the Plugins tab. When Studio asks whether the plugin may reach 127.0.0.1, allow it.",
        "Port %d and token are the plugin's settings; they must match MIF_ROBLOX_PORT and MIF_ROBLOX_TOKEN here." % PORT,
        "Optional, beside MifBridge: Roblox's own Studio MCP server (execute_luau, input, save_place). Turn it on in "
        "Studio: Assistant > Manage MCP Servers > Enable Studio as MCP server, then add Roblox_Studio to your client "
        "(the MifBridge panel's Setup tab shows the entry; rbx_status says whether it is on).",
    ]


def _process_image(pid):
    try:
        from mif_uefn.listener import mifbridge_uefn_win32 as W
        return W.process_image(pid)
    except Exception:  # noqa: BLE001 - an image we cannot read is reported as unknown
        return None


# ---------------------------------------------------------------- the daemon
NO_PORT_ERRORS = (10048, 10055)    # WSAEADDRINUSE, WSAENOBUFS: this machine has no free ephemeral port to connect from
CONNECT_RETRY_PAUSE = 1.0


def _no_local_port(exc):
    return getattr(exc, "winerror", None) in NO_PORT_ERRORS


def _connect(timeout):
    """An HTTPConnection to the daemon, connected. A connect refused for want of a local port (about 25,000 sockets in
    TIME_WAIT on a busy machine, found 2026-11-22) sent nothing, so it is tried once more after a pause; a second such
    refusal is raised as it came. Any other error is raised at once."""
    for attempt in (1, 2):
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
        try:
            conn.connect()
            return conn
        except OSError as exc:
            conn.close()
            if attempt == 2 or not _no_local_port(exc):
                raise
            time.sleep(CONNECT_RETRY_PAUSE)


def _http(method, path, body=None, timeout=STATUS_TIMEOUT):
    conn = _connect(timeout)
    try:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"X-Mif-Token": D.token(), "Content-Type": "application/json"}
        conn.request(method, path, body=data, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, json.loads(raw.decode("utf-8") or "{}"), resp.getheader("X-Mif-Backend")
    finally:
        conn.close()


def _implicit_default():
    return PORT == D.DEFAULT_PORT and not PORT_EXPLICIT


def _port_warning():
    """The warning when this process takes the default port without MIF_ROBLOX_PORT, else None; printed to stderr once."""
    if not _implicit_default():
        return None
    msg = ("MIF_ROBLOX_PORT is not set, so this process (pid %d, %s) took the default port %d. Every Studio whose place "
           "has no ServerStorage.MifBridgePort polls it, so each call drives whichever Studio connects. A lane or a "
           "script sets MIF_ROBLOX_PORT to its own port; set it to %d to mean the default" % (
               os.getpid(), os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "?", PORT, PORT))
    if not _STATE["warned"]:
        _STATE["warned"] = True
        print("WARNING: " + msg, file=sys.stderr, flush=True)
    return msg


def _claim_refusal(st):
    """The refusal for a daemon this process may not use, else None. (1) This process launched its Studio (rbx_launch
    pinned its pid) and the plugin connected now is another Studio. (2) MIF_ROBLOX_PORT is unset and the default port is
    held by another tool's or server's daemon, which drives its own Studio: calls from here would reach that Studio too.
    A standalone daemon (daemon.py run on its own) drives nothing itself and is used as before."""
    plugin = st.get("plugin") or {}
    spid = plugin.get("studioPid") if plugin.get("connected") else None
    pinned = _STATE.get("studio")
    if pinned and isinstance(spid, int) and spid != pinned:
        return {"ok": False, "ran": False, "refusedRule": "other-studio", "launchedPid": pinned, "studioPid": spid,
                "placeName": plugin.get("placeName"), "port": PORT,
                "error": "the plugin on port %d answers for Studio pid %s (%s), not the Studio this process launched "
                         "(pid %s). Nothing was sent. rbx_launch again to take this Studio, or give each Studio its own "
                         "port" % (PORT, spid, plugin.get("placeName"), pinned)}
    owner = st.get("owner") or {}
    if _implicit_default() and st.get("pid") != os.getpid() and not owner.get("standalone"):
        return {"ok": False, "ran": False, "editorState": "portBusy", "refusedRule": "default-port-held",
                "pid": st.get("pid"), "image": _process_image(st.get("pid")), "ownerScript": owner.get("script"),
                "ownerCwd": owner.get("cwd"), "studioPid": spid, "placeName": plugin.get("placeName"), "port": PORT,
                "error": "MIF_ROBLOX_PORT is not set, and the default port %d is held by another process's daemon (pid "
                         "%s, %s)%s. Calls from here would drive that process's Studio%s. Nothing was sent. Set "
                         "MIF_ROBLOX_PORT to this session's own port, or to %d to share that daemon on purpose" % (
                             PORT, st.get("pid"), owner.get("script") or "script unknown",
                             "" if owner else " (a daemon too old to name its owner)",
                             " (pid %s, %s)" % (spid, plugin.get("placeName")) if spid else "", PORT)}
    return None


def _start_daemon(checked):
    _STATE["daemon"] = D.Daemon(PORT).start()
    out = {"ok": True, "started": True, "pid": os.getpid(), "port": PORT, "ownershipChecked": checked}
    warn = _port_warning()
    if warn:
        out["portWarning"] = warn
    return out


def _daemon_record_file(port=None):
    return os.path.join(D.home(), "places", str(PORT if port is None else port), "daemon.json")


DAEMON_MODULE = "mif_roblox.daemon"


def _detached_daemon_alive(rec):
    """True when rec's pid is still the daemon it recorded: same start time (to 0.5 s), and a command line that runs
    mif_roblox.daemon on rec's port. Anything else is not ours to stop."""
    from mif_roblox import launch as L
    pid = (rec or {}).get("pid")
    if not isinstance(pid, int):
        return False
    st = L.PROC_START(pid)
    if st is None or rec.get("start") is None or abs(st - rec["start"]) > 0.5:
        return False
    args = L.CMD_ARGS(pid) or []
    return DAEMON_MODULE in args and "--port" in args and str(rec.get("port")) in args


def read_detached_daemon(port=None):
    """The record of the detached daemon rbx_launch started for this port, with alive, or None."""
    try:
        with open(_daemon_record_file(port), encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return None
    return dict(rec, alive=_detached_daemon_alive(rec)) if isinstance(rec, dict) else None


def start_detached_daemon(wait=15.0):
    """The daemon for this port as a process of its own (python -m mif_roblox.daemon --port N, its own process, no window,
    its output in places/<port>/daemon.log), so it outlives the process that asked; its pid and start time are recorded
    beside the launch records and rbx_close ends it by that pid. Nothing is started when the port already has a
    listener. -> {ok, started, pid, port, log} or the refusal."""
    import subprocess
    from mif_roblox import launch as L
    owner = LISTENING_PID(PORT) if LISTENING_PID else None
    if owner is not None:
        rec = read_detached_daemon()
        return {"ok": True, "started": False, "pid": owner, "port": PORT,
                "detached": bool(rec and rec.get("alive") and rec.get("pid") == owner)}
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # tools/mcp-server
    log = os.path.join(D.home(), "places", str(PORT), "daemon.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    env = dict(os.environ, MIF_ROBLOX_PORT=str(PORT), PYTHONPATH=here + os.pathsep + os.environ.get("PYTHONPATH", ""))
    # CREATE_NO_WINDOW, never DETACHED_PROCESS (a detached process gives every console program it starts a window of its
    # own); CREATE_NEW_PROCESS_GROUP so the caller's Ctrl+C does not reach it. A Windows child outlives its parent.
    flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) | 0x00000200) if os.name == "nt" else 0
    with open(log, "ab") as fh:
        # window: none - flags above carry CREATE_NO_WINDOW; the daemon's output goes to its log
        proc = subprocess.Popen([sys.executable, "-m", DAEMON_MODULE, "--port", str(PORT)], cwd=here, env=env,
                                stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, creationflags=flags,
                                close_fds=True, start_new_session=os.name != "nt")
    deadline = time.time() + wait
    while time.time() < deadline:
        if LISTENING_PID and LISTENING_PID(PORT) == proc.pid:
            break
        if proc.poll() is not None:
            return {"ok": False, "ran": False, "error": "the detached daemon exited at once (code %s): see %s" % (
                proc.returncode, log)}
        time.sleep(0.2)
    else:
        return {"ok": False, "ran": False, "pid": proc.pid,
                "error": "the detached daemon (pid %d) did not listen on %d within %g s: see %s" % (proc.pid, PORT, wait, log)}
    rec = {"pid": proc.pid, "start": L.PROC_START(proc.pid), "port": PORT, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "by": os.getpid(), "log": log}
    with open(_daemon_record_file(), "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    return {"ok": True, "started": True, "pid": proc.pid, "port": PORT, "detached": True, "log": log}


def stop_detached_daemon():
    """End the detached daemon rbx_launch started for this port, by its recorded pid, only while that pid is still it
    (start time and command line); the record is removed. -> what was done."""
    from mif_roblox import launch as L
    rec = read_detached_daemon()
    if rec is None:
        return {"ended": False, "why": "no detached daemon was recorded for this port"}
    if not rec.get("alive"):
        try:
            os.remove(_daemon_record_file())
        except OSError:
            pass
        return {"ended": False, "pid": rec.get("pid"), "why": "the recorded daemon is gone (or its pid is another process)"}
    L.TERMINATE(rec["pid"])
    for _ in range(50):
        if not _detached_daemon_alive(rec):
            break
        time.sleep(0.1)
    gone = not _detached_daemon_alive(rec)
    if gone:
        try:
            os.remove(_daemon_record_file())
        except OSError:
            pass
    return {"ended": gone, "pid": rec["pid"], "why": None if gone else "the daemon did not end"}


def ensure_daemon():
    """{"ok": True, "started": bool, "pid": ...} once the port is a MifBridge Roblox daemon's this process may use, else
    the refusal (_claim_refusal says which daemons it may not)."""
    owner = LISTENING_PID(PORT) if LISTENING_PID else None
    if owner is None and LISTENING_PID is not None and os.environ.get("MIF_ROBLOX_DETACHED_DAEMON") == "1":
        # by option, the first call starts the daemon as a process of its own, which outlives this one
        got = start_detached_daemon()
        if not got.get("ok"):
            return got
        owner = got.get("pid")
    if owner is None and LISTENING_PID is not None:
        try:
            return _start_daemon(True)
        except OSError as exc:
            return {"ok": False, "editorState": "portBusy", "ran": False, "pid": getattr(exc, "pid", None),
                    "image": getattr(exc, "image", None),
                    "error": "port %d looked free but could not be bound: %s" % (PORT, exc)}
    try:
        code, st, backend = _http("GET", "/status")
    except (OSError, ValueError) as exc:
        if LISTENING_PID is None and _STATE["daemon"] is None:
            try:
                return _start_daemon(False)
            except OSError:
                pass
        return {"ok": False, "editorState": "portBusy" if owner else "daemonDown", "ran": False, "pid": owner,
                "image": _process_image(owner) if owner else None,
                "error": "port %d does not answer as a MifBridge Roblox daemon (%s)%s" % (
                    PORT, exc, "; it is held by pid %s" % owner if owner else "")}
    if backend != D.BACKEND or not isinstance(st, dict) or st.get("backend") != D.BACKEND:
        return {"ok": False, "editorState": "portBusy", "ran": False, "pid": owner, "image": _process_image(owner),
                "error": "port %d is held by something that is not a MifBridge Roblox daemon (pid %s, %s). Nothing was "
                         "sent. Free it or set MIF_ROBLOX_PORT" % (PORT, owner, _process_image(owner) or "image unknown")}
    if owner is not None and st.get("pid") != owner:
        return {"ok": False, "editorState": "portBusy", "ran": False, "pid": owner, "image": _process_image(owner),
                "claimedPid": st.get("pid"),
                "error": "port %d answers as a Roblox daemon with pid %s, but the TCP table says pid %s owns it: not "
                         "trusted, nothing sent" % (PORT, st.get("pid"), owner)}
    refused = _claim_refusal(st)
    if refused:
        return refused
    return {"ok": True, "started": False, "pid": st.get("pid"), "port": PORT, "ownershipChecked": owner is not None,
            "status": st}


_ANSWERED = threading.local()     # replies a daemon gave this thread's calls: the coverage record counts only those


def _answered():
    return getattr(_ANSWERED, "n", 0)


def _waited():
    """(seconds this thread's calls were held for an expected plugin, why), summed over its calls so far."""
    return getattr(_ANSWERED, "waited", (0.0, None))


def _ops():
    """This thread's plugin ops a daemon answered, in order: [(op, ok)]. A composite's proved list is checked against the
    slice its own call made (domain_release.keys_from), so it cannot claim a read-back it never asked for."""
    if not hasattr(_ANSWERED, "ops"):
        _ANSWERED.ops = []
    return _ANSWERED.ops


# Services that, once a plugin has made them (the plugin's path resolution makes a missing service with GetService),
# crash Studio at its next save: the auto-save thread reads a null pointer and Studio exits with whatever was unsaved.
# Measured on 0.748 (RB2412): one rbx_get_props of IncrementalPatchBuilder on a fresh place, nothing else, and the next
# auto-save five minutes later crashed Studio; a fresh Studio with nothing run saved fine. Any op that names one - a path
# rooted at it, or game:GetService of it - is refused before it is sent.
CRASHES_STUDIO_ON_SAVE = {"IncrementalPatchBuilder": "RB2412"}


def _save_crash(op, args):
    """The refusal for an op that would make or touch a service in CRASHES_STUDIO_ON_SAVE, else None."""
    def names(v):
        if isinstance(v, str):
            yield v
        elif isinstance(v, dict):
            for x in v.values():
                yield from names(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                yield from names(x)
    hit = None
    for key, v in args.items():
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], str) and v[0] in CRASHES_STUDIO_ON_SAVE and key != "args":
            hit = v[0]                                       # a path rooted at the service
        if key == "args" and op == "call" and args.get("method") in ("GetService", "FindService"):
            hit = hit or next((n for n in names(v) if n in CRASHES_STUDIO_ON_SAVE), None)
    if not hit:
        return None
    return {"ok": False, "ran": False, "refusedRule": "crashes-studio", "service": hit,
            "ledger": CRASHES_STUDIO_ON_SAVE[hit],
            "error": "%s is refused: once it exists in a place, Studio crashes at its next save (auto-save included) and "
                     "loses what was unsaved (%s). Nothing was sent" % (hit, CRASHES_STUDIO_ON_SAVE[hit])}


def _rbx(op, timeout=None, **args):
    """The one choke point: one op to the plugin through the daemon. Never raises."""
    crash = _save_crash(op, args)
    if crash:
        return crash
    ready = ensure_daemon()
    if not ready.get("ok"):
        return ready
    timeout = TIMEOUT if timeout is None else timeout
    body = {"op": op, "args": args, "timeout": timeout}
    if _OPEN_GROUPS and _OPEN_GROUPS[-1][1]:
        # this process holds the plugin's undo group: its calls carry the owner token, so they join it, and the plugin's
        # idle limit counts from them (plugin 0.17.23); another session's write is refused undo-group-held meanwhile
        body["undoOwner"] = _OPEN_GROUPS[-1][1]
    try:
        code, out, _b = _http("POST", "/call", body, timeout=timeout + 10)
    except socket.timeout:
        return {"ok": False, "outcomeUnknown": True, "error": "the daemon did not answer within %ss" % (timeout + 10)}
    except (OSError, ValueError) as exc:
        return {"ok": False, "editorState": "daemonDown", "error": "the daemon on port %d failed: %s" % (PORT, exc)}
    if code != 200 and isinstance(out, dict):
        out.setdefault("ok", False)
    if isinstance(out, dict) and isinstance(out.get("waitedSeconds"), (int, float)) and out.get("pluginExpected"):
        # the daemon held this call for a plugin it expected: the tool's own answer says so (_reports)
        _ANSWERED.waited = _waited()[0] + out["waitedSeconds"], out["pluginExpected"]
    if code == 200 and isinstance(out, dict):
        _ANSWERED.n = _answered() + 1
        ops = _ops()
        ops.append((op, out.get("ok") is not False))
        del ops[:-500]                     # a long session keeps only the recent ones; a call reads its own slice
    return out if isinstance(out, dict) else {"ok": False, "error": "the daemon answered %r" % (out,)}


def _catalog():
    try:
        return C.load(), None
    except (OSError, ValueError) as exc:
        return None, {"ok": False, "ran": False, "error": "no Roblox catalog (%s): run "
                      "tools/mcp-server/mif_roblox/gen_catalog.py" % exc}


PLUGIN_ROOT = "@plugin"
# The plugin's own objects by name under @plugin (plugin 0.17.13): the Plugin itself, its panel and its toolbar.
PLUGIN_ALIASES = ("widget", "toolbar", "button")


def _plugin_rooted(parts):
    return isinstance(parts, list) and bool(parts) and parts[0] == PLUGIN_ROOT


def _plugin_own(parts, what):
    return {"ok": False, "ran": False, "refusedRule": "plugin-own", "target": V.join_path(parts),
            "error": "%s is MifBridge's own plugin object (%s): reads only - a write would change MifBridge's own panel, "
                     "toolbar or settings. Nothing was sent" % (what, V.join_path(parts))}


# CSG on the Plugin (plugin 0.17.22, coverage ruling 7.2 #7): Plugin:Union, Negate, Separate and Intersect change no
# plugin object - they answer new parts made from the parts they are given - so rbx_call sends them on "@plugin" itself
# when every part in their list is a path under Workspace.MifScratch; the plugin's gate judges the same (pluginCsg).
# Every other write on @plugin is still refused plugin-own. Read back (_plugin_csg_readback): every part the call
# answered exists under the folder, and each input is gone or kept as PLUGIN_CSG_INPUTS says Studio leaves it.
PLUGIN_CSG = ("Union", "Negate", "Separate", "Intersect")
SCRATCH_PARTS = ["Workspace", "MifScratch"]
# What Studio does with the parts it was given, per member, as measured live (0.748, plugin 0.17.22, RB3069): every
# input is gone - Union and Intersect answer one new part, Separate the parts the union was made of, and Negate a new
# NegateOperation ("NegativePart") in place of each input. A member not listed is read back and reported, never judged.
PLUGIN_CSG_INPUTS = {"Union": "gone", "Negate": "gone", "Separate": "gone", "Intersect": "gone"}


def _plugin_csg(parts, method, args):
    """True when this rbx_call is one of PLUGIN_CSG on the Plugin itself with a non-empty list of paths, every one under
    Workspace.MifScratch (a handle, or anything outside the folder, is not: plugin-own refuses it as before)."""
    if parts != [PLUGIN_ROOT] or method not in PLUGIN_CSG or not isinstance(args, list) or not args:
        return False
    if not isinstance(args[0], list) or not args[0]:
        return False
    for x in args[0]:
        p = V.split_path(x) if isinstance(x, (str, list)) else None
        if not p or len(p) <= len(SCRATCH_PARTS) or p[:len(SCRATCH_PARTS)] != SCRATCH_PARTS:
            return False
    return True


def _plugin_csg_before(inputs):
    """Each input's ClassName before the call, by its joined path (None when it does not resolve)."""
    out = {}
    for x in inputs:
        p = V.split_path(x)
        r = _rbx("resolve", target=p)
        out[V.join_path(p)] = r.get("className") if r.get("ok") else None
    return out


def _plugin_csg_readback(method, before, returns):
    """(verified, readBack) for a PLUGIN_CSG call: each part it answered resolves under Workspace.MifScratch, and each
    input is kept (same path, same class, not one of the results), replaced (a result now holds its path) or gone
    (not-found); verified when there is at least one result, all of them are there, and - for a member PLUGIN_CSG_INPUTS
    lists - every input is in the state Studio leaves it."""
    made = returns[0] if returns else None
    made = [m for m in made] if isinstance(made, list) else ([] if made is None else [made])
    results, ok = [], bool(made)
    for m in made:
        p = V.split_path(m) if isinstance(m, str) else None
        r = _rbx("resolve", target=p) if p else {"ok": False, "error": "not a placed instance: %r" % (m,)}
        under = bool(p) and len(p) > len(SCRATCH_PARTS) and p[:len(SCRATCH_PARTS)] == SCRATCH_PARTS
        results.append({"path": m, "exists": bool(r.get("ok")), "className": r.get("className"), "underScratch": under,
                        "error": None if r.get("ok") else r.get("error")})
        ok = ok and bool(r.get("ok")) and under
    made_paths = {x["path"] for x in results}
    inputs = {}
    for path, cls in before.items():
        r = _rbx("resolve", target=V.split_path(path))
        if r.get("ok"):
            state = "replaced" if path in made_paths or r.get("className") != cls else "kept"
        elif r.get("refusedRule") == "not-found":
            state = "gone"
        else:
            state, ok = "unread: %s" % r.get("error"), False
        inputs[path] = state
    want = PLUGIN_CSG_INPUTS.get(method)
    back = {"results": results, "inputs": inputs, "inputsExpected": want}
    if want is not None:
        back["inputsAsStudio"] = all(v == want for v in inputs.values())
        ok = ok and back["inputsAsStudio"]
    return ok, back


def _target(target, what="target", game=False, plugin=False):
    """(parts, None) or (None, the refusal). game true (a read: rbx_get_props, rbx_call, rbx_watch) lets "game" itself
    through as [] (spec "game itself as a target"); plugin true lets a path rooted at @plugin through (spec "the plugin's
    own objects, read only"). Every other tool is refused both: an empty path, and plugin-own for @plugin."""
    parts = V.split_path(target)
    if parts is None or (not parts and not game):
        return None, {"ok": False, "ran": False, "refusedRule": "type",
                      "error": "%s is a dot path from game ('Workspace.Part') or a list of names, got %r%s" % (
                          what, target, "" if game else "; game itself is a target only for rbx_get_props, rbx_call "
                                                        "and rbx_watch")}
    if _plugin_rooted(parts) and not plugin:
        return None, _plugin_own(parts, what)
    return parts, None


def _resolve(parts):
    """The target's ClassName, read from Studio."""
    r = _rbx("resolve", target=parts)
    if not r.get("ok"):
        return None, r
    return r.get("className"), None


def _post_verdict(tool, out, ms=None, coverage=None):
    """Tell the plugin's panel how a call ended: verified or not after a read-back, or refused before it was sent. Best
    effort, short timeout, never fails the call: the panel is a view, the answer is the record. coverage, when this call
    added a member to the coverage record, is the panel's new CoverageRow (domain_release.panel_summary)."""
    if not isinstance(out, dict) or _STATE["daemon"] is None and LISTENING_PID is None:
        return
    body = {"tool": tool, "ok": bool(out.get("ok")), "verified": out.get("verified"),
            "refusedRule": out.get("refusedRule"), "error": out.get("error")}
    if coverage:
        body["coverage"] = coverage
    if ms is not None:
        body["ms"] = ms              # the whole call as the caller waited it, every read-back included
    refused = out.get("refused") if isinstance(out.get("refused"), dict) else {}
    if not body["refusedRule"] and refused:
        rules = sorted({str(v.get("refusedRule")) for v in refused.values() if isinstance(v, dict)})
        body["refusedRule"] = rules[0] if len(rules) == 1 else "several"
    if out.get("requestId"):
        body["requestId"] = out["requestId"]
    elif out.get("ran") is not False or out.get("editorState"):
        return                                   # neither judged nor refused here: the plugin's own row says it all
    try:
        _http("POST", "/verdict", body, timeout=1.0)
    except (OSError, ValueError):
        pass


def _reports(fn):
    """For a tool whose answer the panel's journal shows: post its verdict after it returns."""
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        t0 = time.monotonic()
        n0 = _answered()
        ops0 = len(_ops())
        w0 = _waited()[0]
        out = fn(*a, **kw)
        ms = int((time.monotonic() - t0) * 1000)
        w1, why = _waited()
        if isinstance(out, dict) and w1 > w0 and "waitedSeconds" not in out:
            out = dict(out, waitedSeconds=round(w1 - w0, 1), pluginExpected=why)   # held for its plugin first (L134)
        from mif_roblox import domain_release  # noqa: PLC0415 - lazy: the coverage record
        # answered: a daemon replied during this call - a suite that stands a lambda in for _rbx never counts
        added = domain_release.note(fn.__name__, a, kw, out, answered=_answered() > n0, ops=_ops()[ops0:])
        domain_release.wall_hint(fn.__name__, out)
        _post_verdict(fn.__name__, out, ms=ms, coverage=domain_release.panel_summary() if added else None)
        return out
    return wrapper


def _proved(cat, cls, names, how, op):
    """A composite's proved list: [{member: "Owner.Name", how, op}] for the members of cls
    that the plugin op `op` of this same call read back. note() records an entry only when that op was made and answered
    ok during the call, so a tool cannot claim a read-back it never asked for. Unknown names are left out."""
    out = []
    for n in names:
        owner, row = cat.find_member(cls, n)
        if row is not None:
            out.append({"member": "%s.%s" % (owner, n), "how": how, "op": op})
    return out


def _obj(x):
    """Studio's JSONEncode writes an empty table as [], so an empty map from the plugin arrives as a list."""
    return x if isinstance(x, dict) else {}


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if isinstance(text, str) else None


# ---------------------------------------------------------------- tools
def rbx_status(request_id: str = None) -> dict:
    "Roblox Studio: is the daemon up, is the MifBridge plugin connected, and which Studio and place - a nonce round trip through the plugin, the catalog's Studio stamp beside the plugin's, and the setup steps for whatever is missing. plugin.busy names the call the plugin is running now ({op, requestId, seconds, timeout}): a plugin running a long call (a play probe) does not poll and still reads connected, and a call sent meanwhile waits behind it (its answer says queuedBehind). plugin.requestStats (plugin 0.17.0) is the plugin's own HttpService use: requests in the last 60 s against its budget of 450, overruns Studio refused, and how often and how long it paced itself. builtInMcp says whether Roblox's own Studio MCP server is switched on (read from Studio's Assistant settings file) and the entry that adds it beside MifBridge. modeReport (plugin 0.17.4) says what the current safety mode refuses on THIS place - writes outside Workspace.MifScratch, scripts, terrain, client play probes, live data, uploads, Studio preferences - whether the place is bridge-launched, MifBridge's scratch place or published (a published scratch place loses its scratch treatment, RB1471), and for each refusal the one step that lifts it: rbx_set_mode when the person's policy lets the agent raise here, otherwise the person's click in the MifBridge panel. With request_id, what became of that call. playLock is the machine play-test lock read without joining it: held, the holder (pid, port, studioPid, nonce), heldSeconds, holderState and holderAlive, stale when a waiter would break it, and the waiters in arrival order with waitedSeconds and whether each is still live; nothing is written or removed. Read only."
    from mif_roblox import launch as L  # noqa: PLC0415 - the play lock is files on this machine, read only
    try:
        lock = L.play_lock_status()
    except OSError as exc:
        lock = {"error": "the play-test lock could not be read: %s" % exc}
    ready = ensure_daemon()
    if not ready.get("ok"):
        return dict(ready, setup=setup_steps(), playLock=lock)
    try:
        _c, st, _b = _http("GET", "/status")
    except (OSError, ValueError) as exc:
        return {"ok": False, "editorState": "daemonDown", "error": str(exc), "setup": setup_steps(), "playLock": lock}
    cat, _err = _catalog()
    out = {"ok": True, "daemon": {k: st.get(k) for k in ("pid", "port", "protocol", "hold", "queued", "upSeconds")},
           "daemonStarted": ready.get("started"), "ownershipChecked": ready.get("ownershipChecked"),
           "plugin": st.get("plugin"), "catalog": None, "playLock": lock}
    rec = read_detached_daemon()
    owner = st.get("owner") or {}
    out["daemon"]["host"] = ("a process of its own (pid %s, detached by rbx_launch)" % rec["pid"]
                             if rec and rec.get("alive") and rec.get("pid") == st.get("pid") else
                             "a standalone daemon (pid %s)" % st.get("pid") if owner.get("standalone") else
                             "inside pid %s (%s)" % (st.get("pid"), owner.get("script") or "a tool or server process"))
    from mif_roblox import domain_release as _rel  # noqa: PLC0415 - read from disk, never started or called
    _b = _rel.builtin_mcp()
    out["builtInMcp"] = {k: _b.get(k) for k in ("enabled", "launcher", "enable", "client", "note", "warnings")}
    if cat:
        out["catalog"] = {"studio": cat.studio, "reachable": cat.doc.get("counts", {}).get("reachable")}
        from mif_roblox import domain_release  # noqa: PLC0415 - the coverage record, read only
        out["coverage"] = domain_release.summary(cat)
    if request_id:
        try:
            out["request"] = _http("GET", "/request?id=%s" % request_id)[1]
        except (OSError, ValueError) as exc:
            out["request"] = {"ok": False, "error": str(exc)}
    plugin = st.get("plugin") or {}
    if not plugin.get("connected"):
        out.update(ok=False, editorState="noPlugin", setup=setup_steps(),
                   error="the daemon is up but no MifBridge plugin is polling it")
        return out
    nonce = uuid.uuid4().hex
    t0 = time.monotonic()
    ping = _rbx("ping", nonce=nonce, timeout=15)
    out["roundTripMs"] = int((time.monotonic() - t0) * 1000)
    out["ping"] = ping
    if not ping.get("ok") or ping.get("nonce") != nonce:
        out.update(ok=False, error="the plugin did not echo the nonce: %s" % (ping.get("error") or ping.get("nonce")))
        return out
    out["modeReport"] = mode_report(ping, D.read_policy())
    have, want = ping.get("studioVersion"), (cat.studio.get("version") if cat else None)
    out["catalogMatches"] = None if not (have and want) else have == want
    if out["catalogMatches"] is False:
        out["warning"] = ("the catalog is for Studio %s and this Studio is %s: a removed member answers member-missing, an "
                          "added one unknown-member until gen_catalog.py --studio-exe auto runs" % (want, have))
    return out


EDITOR_STATE = "editor state (selection_set, camera set, script tab open/close)"


def mode_report(ping, policy):
    """What the plugin's current mode refuses on this place, and the one step that lifts each refusal (spec: "Extend
    rbx_status: what the current mode refuses on this place"). Mirrors the plugin's gate (MifBridge.server.lua gate()):
    read refuses every write; scratch allows writes under Workspace.MifScratch, or anywhere in MifBridge's own scratch
    place (game.Name MifScratch.rbxlx), terrain only in the panel's terrain box, and never live data; full refuses only
    Studio's preferences (everywhere) and leaves uploads to their own confirm. Read from ping; nothing is sent."""
    mode = ping.get("mode")
    if mode not in D.MODES:
        return {"mode": mode, "error": "the plugin did not say its mode (a plugin before 0.11.0)"}
    known = "personClicked" in ping
    scratch_place = ping.get("scratchPlace") if "scratchPlace" in ping else ping.get("placeName") == "MifScratch.rbxlx"
    published = isinstance(ping.get("placeId"), int) and ping.get("placeId") > 0
    launched = bool(ping.get("launchMarked"))
    clicked = bool(ping.get("personClicked"))
    agent_may = bool(policy.get("agentMayRaiseMode")) and launched and not clicked
    if agent_may:
        raise_step = {"by": "agent", "step": "rbx_set_mode('full')"}
    else:
        why = ("the person's policy file does not let the agent raise the mode" if not policy.get("agentMayRaiseMode")
               else "this place is not one rbx_launch made (no ServerStorage.MifBridgeLaunched)" if not launched
               else "the person clicked a mode in the panel this session, and their click wins")
        raise_step = {"by": "person", "step": "in this Studio: Plugins > MifBridge, Safety mode > Full", "agentCannot": why}
    rows = []

    def row(kind, refused, why, lift=None):
        r = {"kind": kind, "refused": refused, "why": why}
        if refused and lift:
            r["lift"] = lift
        rows.append(r)

    if mode == "read":
        for kind in ("writes outside Workspace.MifScratch", "writes inside Workspace.MifScratch", "scripts", "terrain",
                     "play probes", "client play probes", "live data (data stores)", "uploads", "web requests (HttpService)"):
            row(kind, True, "read mode refuses every write", raise_step)
        # plugin 0.17.23 (found 2026-11-22): setting the selection, moving the editor camera and opening or closing a
        # script tab change no place content, and read mode refuses them all the same; reading them is a read
        row(EDITOR_STATE, True, "read mode refuses them since plugin 0.17.23: they change the person's editor, though "
            "not the place (reading them is allowed)", raise_step)
    elif mode == "scratch":
        whole = bool(scratch_place)
        row("writes outside Workspace.MifScratch", not whole,
            "MifBridge's own scratch place: all of it is scratch" if whole
            else "scratch mode writes only under Workspace.MifScratch on this place", raise_step)
        row("writes inside Workspace.MifScratch", False, "scratch mode allows them")
        row("scripts", not whole, "scripts follow the same path rule: allowed under Workspace.MifScratch" +
            ("" if whole else ", refused anywhere else on this place"), raise_step)
        box = bool(ping.get("terrainBox"))
        row("terrain", not whole, "MifBridge's own scratch place" if whole else (
            "only inside the terrain box %s" % ping.get("terrainBoxStuds") if box else
            "Terrain is one object, never under Workspace.MifScratch, and the panel's terrain box is off"),
            dict(raise_step, alternative="the person turns on the terrain box (Safety mode) for writes inside %s"
                 % ping.get("terrainBoxStuds")) if not box else raise_step)
        row("play probes", False, "the probe's own script goes into Workspace.MifScratch")
        row("client play probes", not whole, "allowed in MifBridge's own scratch place" if whole else
            "a client probe writes into StarterPlayer and ReplicatedStorage", raise_step)
        row("live data (data stores)", True, "a data store or memory store handle needs full mode on every place",
            raise_step)
        row("uploads", True, "an upload writes to the person's account and needs full mode", raise_step)
        row("web requests (HttpService)", not whole, "MifBridge's own scratch place: all of it is scratch" if whole else
            "HttpService's members other than its pure ones (JSONEncode...) are writes, and HttpService is outside "
            "Workspace.MifScratch", raise_step)
        row(EDITOR_STATE, False, "the person's editor, not the place: allowed outside read mode")
    else:
        row("writes outside Workspace.MifScratch", False, "full mode")
        row("scripts", False, "full mode")
        row("terrain", False, "full mode")
        row("client play probes", False, "full mode")
        row("live data (data stores)", False, "full mode (a data store write still needs its own confirm)")
        row("uploads", False, "full mode; each upload still needs confirm true")
        row("web requests (HttpService)", False, "full mode")
        row(EDITOR_STATE, False, "full mode")
    rows.append({"kind": "Studio preferences", "refused": True, "why": "refused in every mode on every place",
                 "lift": None})
    out = {"mode": mode, "modeBy": ping.get("modeBy"), "placeName": ping.get("placeName"), "placeId": ping.get("placeId"),
           "published": published, "bridgeLaunched": launched, "scratchPlace": bool(scratch_place),
           "personClicked": clicked if known else None, "agentMayRaise": agent_may if known else None,
           "refuses": [r["kind"] for r in rows if r["refused"]], "kinds": rows}
    if published and launched and not scratch_place:
        out["note"] = ("this bridge-launched place is published (PlaceId %s): its name is no longer MifScratch.rbxlx, so the "
                       "whole-place scratch treatment is off (RB1471)" % ping.get("placeId"))
    if not known:
        out["note"] = (out.get("note", "") + "; " if out.get("note") else "") + (
            "plugin %s does not report the person's click (0.17.4 does): agentMayRaise is unknown" % ping.get("plugin"))
    return out


def rbx_describe_member(class_name: str, member: str = None) -> dict:
    "Roblox Studio: what the catalog says about a class, or one member of it - kind, type, parameters, security, tags, whether a plugin can reach it and write it, and the superclass it comes from. From the catalog; Studio need not run. Ask this before naming a member: never guess one."
    cat, err = _catalog()
    if err:
        return err
    if class_name not in cat.classes:
        return cat.unknown_class(class_name)
    if member is None:
        return dict(cat.describe_class(class_name), ok=True, studio=cat.studio)
    owner, row = cat.find_member(class_name, member)
    if row is None:
        return cat.unknown_member(class_name, member)
    return dict(cat.describe_row(owner, member, row), ok=True, askedOn=class_name, studio=cat.studio)


def rbx_describe_class(class_name: str, kind: str = None, inherited: bool = True) -> dict:
    "Roblox Studio: every member of a class from the catalog, by kind (Property, Function, Event, Callback), each with its type or parameters, security, tags, whether a plugin can reach it and write it, and the class it comes from (inherited ones marked). Studio need not run."
    cat, err = _catalog()
    if err:
        return err
    if class_name not in cat.classes:
        return cat.unknown_class(class_name)
    if kind is not None and kind not in C.KINDS.values():
        return {"ok": False, "ran": False, "error": "kind is one of %s" % ", ".join(C.KINDS.values())}
    rows = cat.members_of(class_name)
    members = []
    for name, (owner, row) in sorted(rows.items()):
        if (not inherited and owner != class_name) or (kind and C.KINDS[row[0]] != kind):
            continue
        d = cat.describe_row(owner, name, row)
        d["inherited"] = owner != class_name
        if row[0] != "P" and any(p[1] == "Function" for p in row[5]):
            d["needsFunction"] = True
        members.append(d)
    own = [m for m in members if not m["inherited"]]
    return dict(cat.describe_class(class_name), ok=True, studio=cat.studio, members=members,
                ownCounted=sum(1 for m in own if m["counted"]), shown=len(members))


def rbx_describe_enum(enum_name: str) -> dict:
    "Roblox Studio: an enum's items from the catalog, and the properties that use it. Studio need not run."
    cat, err = _catalog()
    if err:
        return err
    if enum_name not in cat.enums:
        import difflib
        return {"ok": False, "ran": False, "refusedRule": "unknown-enum", "error": "no enum %r in the catalog" % enum_name,
                "didYouMean": difflib.get_close_matches(str(enum_name), list(cat.enums), n=5, cutoff=0.6)}
    used = sorted("%s.%s" % (cls, n) for cls, c in cat.classes.items() for n, row in c["m"].items()
                  if row[0] == "P" and row[1] == "Enum." + enum_name)
    return {"ok": True, "enum": enum_name, "items": list(cat.enums[enum_name]), "count": len(cat.enums[enum_name]),
            "usedByProperties": used[:200], "usedByCount": len(used), "studio": cat.studio}


def rbx_find(name: str = None, class_name: str = None, root: Any = "game", limit: int = 100) -> dict:
    "Roblox Studio: instances under root whose Name equals name and/or that are a class_name (IsA), as paths with their class. Read only."
    if name is None and class_name is None:
        return {"ok": False, "ran": False, "error": "give name, class_name, or both"}
    if class_name is not None:
        cat, err = _catalog()
        if err:
            return err
        if class_name not in cat.classes:
            return cat.unknown_class(class_name)
    parts = [] if root in ("game", None) else V.split_path(root)
    if parts is None:
        return _target(root, "root")[1]
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 5000:
        return {"ok": False, "ran": False, "error": "limit is 1-5000"}
    r = _rbx("find", root=parts, name=name, className=class_name, limit=limit)
    if r.get("ok"):
        r["matches"] = [dict(m, path=V.join_path(m.get("path"))) for m in r.get("matches") or []]
    return r


def rbx_tree(root: Any = "Workspace", depth: int = 2, limit: int = 500) -> dict:
    "Roblox Studio: the hierarchy under root to depth levels: name, class and child count per node. The answer is {ok, nodes, truncated, root}: the nodes are NOT at the top level - root is the node for root itself, {name, className, childCount, children}, and its children (each the same shape, nested to depth) are root.children; a node at the last level has childCount and no children key, and a node cut short by limit has truncated true. nodes counts every node returned, root included. A cleanup that removes stale children is checked by the count left (root.childCount on a second read), never by a top-level children, which is never there. Read only."
    parts = [] if root in ("game", None) else V.split_path(root)
    if parts is None:
        return _target(root, "root")[1]
    if not isinstance(depth, int) or isinstance(depth, bool) or not 0 <= depth <= 20:
        return {"ok": False, "ran": False, "error": "depth is 0-20"}
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 10000:
        return {"ok": False, "ran": False, "error": "limit is 1-10000"}
    return _rbx("tree", root=parts, depth=depth, limit=limit)


def _encode_props(cat, cls, props, write=True):
    """{name: wire} or a refusal naming every property that cannot be written and why. Nothing is sent on a refusal."""
    if not isinstance(props, dict) or not props:
        return None, {"ok": False, "ran": False, "error": "props is an object of property name to value"}
    wire, refused = {}, {}
    for name, value in props.items():
        bad = cat.write_refusal(cls, name) if write else cat.read_refusal(cls, name, kind="P")
        if bad:
            refused[name] = {k: bad[k] for k in ("refusedRule", "securityTag", "error", "didYouMean") if k in bad}
            continue
        w, why = V.encode(cat.property_type(cls, name), value, cat.enums)
        if why:
            refused[name] = {"refusedRule": "type", "type": cat.property_type(cls, name), "error": why}
            continue
        wire[name] = w
    if refused:
        return None, {"ok": False, "ran": False, "refused": refused, "className": cls,
                      "error": "%d of %d properties refused before anything was sent: %s"
                               % (len(refused), len(props), "; ".join("%s: %s" % (k, v["error"]) for k, v in refused.items()))}
    return wire, None


def _judge(sent, after):
    """Compare each sent wire value with its read-back. (verified, mismatches)."""
    bad = {}
    for name, w in sent.items():
        back = _obj(after).get(name, {"$t": "missing"})
        if not V.same(w, back):
            bad[name] = {"sent": V.decode(w), "readBack": V.decode(back)}
    return not bad, bad


@_reports
def rbx_get_props(target: Any, props: list) -> dict:
    "Roblox Studio: read properties of one instance (a path, or {handle} for an object a call returned), by name, typed by the catalog. target \"game\" is the DataModel itself (game.PlaceId); a path rooted at \"@plugin\" is MifBridge's own Plugin object, \"@plugin.widget\" its panel and \"@plugin.toolbar\" its toolbar (plugin 0.17.13; read only); \"@settings\" is Studio's settings() and \"@usersettings\" is UserSettings() (\"@usersettings.UserGameSettings\", plugin 0.17.17), both read only. A member a plugin cannot read is refused with its security tag before anything is sent. Read only."
    if isinstance(target, dict) and set(target) == {"handle"} and isinstance(target["handle"], str):
        parts = {"handle": target["handle"]}
    else:
        parts, err = _target(target, game=True, plugin=True)
        if err:
            return err
    if not isinstance(props, list) or not props or not all(isinstance(p, str) for p in props):
        return {"ok": False, "ran": False, "error": "props is a list of property names"}
    cat, err = _catalog()
    if err:
        return err
    cls, err = _resolve(parts)
    if err:
        return err
    refused = {}
    for p in props:
        bad = cat.read_refusal(cls, p, kind="P")
        if bad:
            refused[p] = {k: bad[k] for k in ("refusedRule", "securityTag", "error", "didYouMean") if k in bad}
    if refused:
        return {"ok": False, "ran": False, "className": cls, "refused": refused,
                "error": "refused before reading: %s" % "; ".join("%s: %s" % (k, v["error"]) for k, v in refused.items())}
    r = _rbx("get_props", target=parts, props=props)
    if r.get("ok"):
        raw = _obj(r.get("values"))
        r["values"] = {k: V.decode(v) for k, v in raw.items()}
        r.setdefault("className", cls)
        # a value the plugin could not encode came back as its text ({"$t", "str"}): read, but not decoded, so the
        # coverage record does not count it (spec "a read only when it decoded")
        undecoded = sorted(k for k, v in raw.items() if isinstance(v, dict) and "str" in v and "v" not in v)
        if undecoded:
            r["undecoded"] = undecoded
    return r


@_reports
def rbx_set_props(target: Any, props: dict) -> dict:
    "Roblox Studio: set properties of one instance in one undo step, then read each back. ok means Studio holds every value sent (verified: true); a clamp or rejection answers ok false with sent and readBack. Refused before sending: unreachable security, ReadOnly, an unknown name, a value of the wrong shape for the property's type."
    parts, err = _target(target)
    if err:
        return err
    cat, err = _catalog()
    if err:
        return err
    cls, err = _resolve(parts)
    if err:
        return err
    wire, err = _encode_props(cat, cls, props)
    if err:
        return err
    r = _rbx("set_props", target=parts, props=wire)
    if not r.get("ok"):
        return r
    verified, bad = _judge(wire, r.get("after"))
    before_w, after_w = _obj(r.get("before")), _obj(r.get("after"))
    out = {"ok": verified, "verified": verified, "className": cls, "target": V.join_path(parts),
           "before": {k: V.decode(v) for k, v in before_w.items()},
           "after": {k: V.decode(v) for k, v in after_w.items()}, "requestId": r.get("requestId"),
           # changed: Studio held something else before. A write of the value already there reads back equal too, so
           # the coverage record keeps "write" only for a changed property and "write-same" for the rest
           "changed": {k: k in before_w and not V.same(before_w[k], after_w[k]) for k in after_w}}
    if bad:
        out.update(mismatch=bad, error="Studio holds something else for %s" % ", ".join(sorted(bad)))
    return out


@_reports
def rbx_create(class_name: str, parent: Any, name: str = None, props: dict = None, tags: list = None,
               outside: bool = False) -> dict:
    "Roblox Studio: make an instance of class_name under parent, optionally named, with properties and CollectionService tags, in one undo step; Parent is set last. Read back: found by its path, its ClassName, every property sent and every tag. A NotCreatable or Service class is refused before sending. outside true (plugin 0.15.0) makes it with no parent at all and never parents it - for an object Studio refuses to parent, such as a GetTextBoundsParams - and answers a handle ({handle}) that rbx_call, rbx_get_props and rbx_release_handle take; parent is then ignored, and nothing enters the place or the undo history."
    if tags is not None and not (isinstance(tags, list) and all(isinstance(t, str) and t for t in tags) and len(tags) <= 50):
        return {"ok": False, "ran": False, "error": "tags is a list of up to 50 non-empty strings"}
    cat, err = _catalog()
    if err:
        return err
    bad = cat.create_refusal(class_name)
    if bad:
        return bad
    if not isinstance(outside, bool):
        return {"ok": False, "ran": False, "error": "outside is true or false"}
    if outside:
        if tags:
            return {"ok": False, "ran": False, "error": "an object made outside the place takes no tags"}
        pparts = None
    else:
        pparts, err = _target(parent, "parent")
        if err:
            return err
    if name is not None and (not isinstance(name, str) or not name):
        return {"ok": False, "ran": False, "error": "name is a non-empty string"}
    wire = {}
    if props:
        wire, err = _encode_props(cat, class_name, props)
        if err:
            return err
    if outside:
        r = _rbx("create", className=class_name, parent=None, outside=True, name=name, props=wire, tags=[])
        if not r.get("ok"):
            return r
        if r.get("outside") is not True:
            return dict(r, ok=False, verified=False, refusedRule="plugin-too-old",
                        error="the plugin made it in the place: outside needs plugin 0.15.0")
        ok_props, mism = _judge(wire, r.get("after"))
        good = ok_props and r.get("className") == class_name and r.get("parentIsNil") is True
        out = {"ok": good, "verified": good, "handle": r.get("handle"), "className": r.get("className"),
               "parentIsNil": r.get("parentIsNil"), "after": {k: V.decode(v) for k, v in _obj(r.get("after")).items()},
               "requestId": r.get("requestId")}
        if mism:
            out.update(mismatch=mism, error="properties differ: %s" % ", ".join(sorted(mism)))
        return out
    r = _rbx("create", className=class_name, parent=pparts, name=name, props=wire, tags=tags or [])
    if not r.get("ok"):
        return r
    problems = []
    if tags and not set(tags) <= set(r.get("tags") or []):
        problems.append("tags read back %s, sent %s" % (r.get("tags"), tags))
    if r.get("className") != class_name:
        problems.append("the new instance is a %s" % r.get("className"))
    path = r.get("path") or []
    if not same_parent(path[:-1], pparts):
        problems.append("it is at %s, not under %s" % (V.join_path(path), V.join_path(pparts)))
    if name is not None and (path[-1:] != [name]):
        problems.append("it is named %s" % (path[-1:] or ["?"])[0])
    ok_props, mism = _judge(wire, r.get("after"))
    out = {"ok": not problems and ok_props, "verified": not problems and ok_props, "path": V.join_path(path),
           "className": r.get("className"), "after": {k: V.decode(v) for k, v in _obj(r.get("after")).items()},
           "requestId": r.get("requestId")}
    if not problems and ok_props:
        # what the create op read back from Studio: the new instance's ClassName and every property sent, read equal.
        # Read, not written: a value sent may be the class's default, so this proves no write (spec S5's proved list)
        out["proved"] = _proved(cat, class_name, ["ClassName"] + sorted(wire), "read", "create")
    if mism:
        out["mismatch"] = mism
        problems.append("properties differ: %s" % ", ".join(sorted(mism)))
    if problems:
        out["error"] = "; ".join(problems)
    return out


@_reports
def rbx_clone(target: Any, parent: Any = None, name: str = None) -> dict:
    "Roblox Studio: copy an instance (Clone) under parent (default: the original's parent), optionally renamed, in one undo step. Read back: the copy found by path with the original's ClassName."
    parts, err = _target(target)
    if err:
        return err
    pparts = None
    if parent is not None:
        pparts, err = _target(parent, "parent")
        if err:
            return err
    r = _rbx("clone", target=parts, parent=pparts, name=name)
    if not r.get("ok"):
        return r
    want_parent = pparts if pparts is not None else parts[:-1]
    path = r.get("path") or []
    problems = []
    if not same_parent(path[:-1], want_parent):
        problems.append("the copy is at %s" % V.join_path(path))
    if r.get("className") != r.get("sourceClassName"):
        problems.append("the copy is a %s, the original a %s" % (r.get("className"), r.get("sourceClassName")))
    out = {"ok": not problems, "verified": not problems, "path": V.join_path(path), "className": r.get("className"),
           "requestId": r.get("requestId")}
    if problems:
        out["error"] = "; ".join(problems)
    return out


INDEXED = re.compile(r"^(.+)\[(\d+)\]$")


@_reports
def rbx_destroy(target: Any, all: bool = False) -> dict:  # noqa: A002 - the parameter is the word a caller writes
    "Roblox Studio: remove an instance in one undo step. Read back: nothing is left at its path under its parent. Plugin 0.16.0 and later remove it by Parent = nil inside the recording, so rbx_undo puts it back (removedBy, undoable); an older plugin calls Destroy, which no undo reverses. Several children with the same name: a path segment Name[n] reaches the nth of them (GetChildren order; plugin 0.16.3, and an ambiguous refusal lists each one's class, position and indexed path), and all=true removes every child of the parent named the last segment, in one undo step, read back as none left."
    parts, err = _target(target)
    if err:
        return err
    if len(parts) < 2:
        return {"ok": False, "ran": False, "refusedRule": "not-creatable", "error": "a service cannot be destroyed"}
    if all is not False and all is not True:
        return {"ok": False, "ran": False, "refusedRule": "type", "error": "all is true or false, got %r" % (all,)}
    if all:
        if INDEXED.match(parts[-1]):
            return {"ok": False, "ran": False, "refusedRule": "type",
                    "error": "all=true removes every child named the last segment; %s names one of them by index: drop "
                             "the [n], or drop all" % parts[-1]}
        r = _rbx("destroy", target=parts, all=True)
        if not r.get("ok"):
            return r
        after = _rbx("resolve", target=parts)
        gone = r.get("stillThere") is False and after.get("refusedRule") == "not-found"
        out = {"ok": gone, "verified": gone, "target": V.join_path(parts), "count": r.get("count"),
               "classNames": r.get("classNames"), "removedBy": r.get("removedBy"),
               "undoable": r.get("removedBy") == "Parent = nil", "requestId": r.get("requestId")}
        if r.get("count") is None:
            out.update(ok=False, verified=False, error="the plugin ignored all=true (it is older than 0.16.3): it removed "
                                                       "at most one; restart Studio on plugin 0.16.3 or later")
        elif not gone:
            out["error"] = "something named %s is still under its parent after removing %s: %s" % (
                parts[-1], r.get("count"), after.get("className") or after.get("error"))
        return out
    r = _rbx("destroy", target=parts)
    if not r.get("ok"):
        return r
    gone = r.get("stillThere") is False
    # Plugin 0.16.0 answers removedBy; an older plugin says nothing because it called Destroy, which no undo brings back
    # (RB712).
    removed_by = r.get("removedBy") or "Destroy"
    out = {"ok": gone, "verified": gone, "target": V.join_path(parts), "className": r.get("className"),
           "removedBy": removed_by, "undoable": removed_by == "Parent = nil", "requestId": r.get("requestId")}
    if not gone:
        out["error"] = "something is still at %s after Destroy" % V.join_path(parts)
    return out


@_reports
def rbx_reparent(target: Any, parent: Any) -> dict:
    "Roblox Studio: move an instance under a new parent in one undo step. Read back: its Parent is the new parent."
    parts, err = _target(target)
    if err:
        return err
    pparts, err = _target(parent, "parent")
    if err:
        return err
    r = _rbx("reparent", target=parts, parent=pparts)
    if not r.get("ok"):
        return r
    ok = r.get("parentPath") == pparts
    out = {"ok": ok, "verified": ok, "path": V.join_path(r.get("path")), "requestId": r.get("requestId")}
    if not ok:
        out["error"] = "its parent is %s, not %s" % (V.join_path(r.get("parentPath")), V.join_path(pparts))
    return out


@_reports
def rbx_selection(set: list = None) -> dict:  # noqa: A002 - the parameter is the verb a caller writes
    "Roblox Studio: the current selection as paths; with set, select exactly those instances and read the selection back."
    if set is None:
        r = _rbx("selection_get")
        if r.get("ok"):
            r["selection"] = [V.join_path(p) for p in r.get("selection") or []]
        return r
    if not isinstance(set, list):
        return {"ok": False, "ran": False, "error": "set is a list of paths ([] clears the selection)"}
    paths = []
    for t in set:
        parts, err = _target(t)
        if err:
            return err
        paths.append(parts)
    r = _rbx("selection_set", paths=paths)
    if not r.get("ok"):
        return r
    back = r.get("selection") or []
    ok = sorted(map(tuple, back)) == sorted(map(tuple, paths))
    out = {"ok": ok, "verified": ok, "selection": [V.join_path(p) for p in back], "requestId": r.get("requestId")}
    if not ok:
        out["error"] = "Studio's selection is not what was set"
    return out


def rbx_undo_group(action: str, name: str = None) -> dict:
    "Roblox Studio: begin a named undo group that every following write of this session joins, then commit it (one undo step) or cancel it (Studio reverts it). Without a group each write is its own undo step. The group has an owner (plugin 0.17.23): begin answers owner, a token this session's calls carry from then on; only they join the group and only its owner ends it (refusedRule undo-owner), and another session's write while it is open is refused undo-group-held, never folded in. A group with no call from its owner for idleLimit seconds (120) is committed, never cancelled, and the next reply says so (undoGroupCommitted). rbx_status's ping names the open group, its owner's first 8 characters and its age (undoGroup)."
    if action not in ("begin", "commit", "cancel"):
        return {"ok": False, "ran": False, "error": "action is begin, commit or cancel"}
    if action == "begin":
        if not isinstance(name, str) or not name:
            return {"ok": False, "ran": False, "error": "begin needs a name: it is what Studio's undo menu shows"}
        # the token is made here and sent, so a begin whose answer is lost still leaves this side able to end the group
        owner = uuid.uuid4().hex
        r = _rbx("undo_begin", name=name, owner=owner)
        if r.get("ok") or r.get("outcomeUnknown"):
            _OPEN_GROUPS.append((name, r.get("owner") or owner))
        return r
    owner = _OPEN_GROUPS.pop()[1] if _OPEN_GROUPS else None
    return _rbx("undo_end", operation=action, owner=owner)


# Undo groups this process began and has not ended, newest last, as (name, owner token). The plugin keeps one open group
# and runs every write inside it until undo_end: a group nobody ends took every later write, from any session (found
# 2026-11-22); since plugin 0.17.23 only calls carrying its owner join it, and an idle one is committed.
_OPEN_GROUPS = []


def begin_group(name):
    """rbx_undo_group begin for a composite. A begin whose answer was lost (outcomeUnknown) may have opened the group in
    Studio while the caller reads it as not open and never ends it: it is committed at once (nothing was written in it
    yet) and answered not ok, closedAfterUnknown saying whether the commit went through."""
    r = rbx_undo_group("begin", name=name)
    if not r.get("ok") and r.get("outcomeUnknown"):
        c = rbx_undo_group("commit")
        r = dict(r, closedAfterUnknown=bool(c.get("ok")))
    return r


def ends_group_on_raise(fn):
    """A composite that begins an undo group and may raise before it ends it: on an exception, every group it began and
    left open is cancelled (Studio reverts what it wrote in it), then the exception goes on."""
    @functools.wraps(fn)
    def run(*a, **k):
        depth = len(_OPEN_GROUPS)
        try:
            return fn(*a, **k)
        except BaseException:
            while len(_OPEN_GROUPS) > depth:
                rbx_undo_group("cancel")
            raise
    return run


def rbx_script_read(target: Any) -> dict:
    "Roblox Studio: a Script, LocalScript or ModuleScript's source as the script editor holds it, with its sha256 (pass that to rbx_script_write as expect_sha256). Read only."
    parts, err = _target(target)
    if err:
        return err
    r = _rbx("script_read", target=parts)
    if r.get("ok"):
        r["sha256"] = _sha(r.get("source"))
    return r


def _package_of(parts):
    """The package a script sits in: the parent of a PackageLink that is the path itself or one of its ancestors, as a
    path list. -> (package or None, failure or None). Every PackageLink in the place, page by page.

    Studio refuses a write to a package script (ScriptDocument reason RBXPackageScript) only while the script's tab is
    open; with it closed, UpdateSourceAsync wrote one and read it back (RB906). A package's edits belong to the package
    and are published by the person, so rbx_script_write and rbx_batch ask this before every write and rbx_collab_guard
    answers it. A query that fails is a failure, never "not in a package"."""
    root, _link, err = _package_link_of(parts)
    return root, err


def _package_link_of(parts):
    """_package_of, with the PackageLink's own path: (package, link, failure)."""
    page = 0
    while True:
        q = _rbx("query", root=[], className="PackageLink", page=page, pageSize=500)
        if not q.get("ok"):
            return None, None, dict(q, ok=False, ran=False,
                                    error="whether the script is inside a package could not be read: %s" % q.get("error"))
        for m in q.get("matches") or []:
            link = V.split_path(m.get("path"))
            root = link[:-1] if link else None
            if root and parts[:len(root)] == root:
                return root, link, None
        if not q.get("more"):
            return None, None, None
        page += 1


def _package_refusal(parts):
    """None when parts is in no package; else the refusal rbx_script_write and rbx_batch answer, nothing sent."""
    pkg, link, err = _package_link_of(parts)
    if err:
        return err
    if pkg is None:
        return None
    out = {"ok": False, "ran": False, "refusedRule": "read-only-document", "target": V.join_path(parts),
           "package": V.join_path(pkg), "packageLink": V.join_path(link),
           "error": "%s is inside the package %s: its edits belong to the package and are published by the person, so "
                    "nothing was written (Studio refuses it only while its tab is open, RBXPackageScript)"
                    % (V.join_path(parts), V.join_path(pkg))}
    # Studio's own reason, when the script's tab is open: one read of that tab (a closed tab answers not-open). Nothing
    # is written either way; the reason says why in Studio's words.
    doc = _rbx("script_docs", target=parts)
    out["openTab"] = bool(doc.get("ok"))
    if doc.get("ok"):
        out["readOnlyReasons"] = doc.get("readOnlyReasons") or []
        if out["readOnlyReasons"]:
            out["error"] += "; its open tab is read-only: %s" % ", ".join(str(x) for x in out["readOnlyReasons"])
    return out


@_reports
def rbx_script_write(target: Any, source: str, expect_sha256: str = None, allow_syntax_errors: bool = False) -> dict:
    "Roblox Studio: replace a script's whole source in one undo step, through the script editor so an open tab is not overwritten later. Read back by sha256. With expect_sha256, refused unsent when the script changed since it was read. A script inside a package (a PackageLink on it or an ancestor) is refused read-only-document, its tab open or not. Source that does not compile is refused syntax before anything is sent, naming the line, column and Luau's message (checkedBy: the Luau analyzer on this machine, else MifBridge's lexer pass, which catches a line break inside a string and an unfinished long string or comment); allow_syntax_errors true writes a deliberately broken script."
    parts, err = _target(target)
    if err:
        return err
    if not isinstance(source, str):
        return {"ok": False, "ran": False, "error": "source is a string"}
    if not isinstance(allow_syntax_errors, bool):
        return {"ok": False, "ran": False, "error": "allow_syntax_errors is true or false"}
    if not allow_syntax_errors:
        from mif_roblox import luau_syntax  # noqa: PLC0415
        bad = luau_syntax.refusal(V.join_path(parts), source)
        if bad:
            return bad
    cat, err = _catalog()
    if err:
        return err
    cur = _rbx("script_read", target=parts)
    if not cur.get("ok"):
        return cur
    cls = cur.get("className")
    if not cat.is_a(cls, SCRIPT_BASE):
        return {"ok": False, "ran": False, "refusedRule": "type", "error": "%s is a %s, not a script" % (V.join_path(parts), cls)}
    before = _sha(cur.get("source"))
    if expect_sha256 is not None and expect_sha256 != before:
        return {"ok": False, "ran": False, "refusedRule": "sha-mismatch", "sha256": before, "expected": expect_sha256,
                "error": "the script changed since it was read: nothing written. Read it again"}
    bad = _package_refusal(parts)
    if bad:
        return bad
    r = _rbx("script_write", target=parts, source=source)
    if not r.get("ok"):
        return r
    want, got = _sha(source), _sha(r.get("source"))
    tab = _sha(r.get("tabText")) if r.get("openTab") else None
    good = want == got and (tab is None or tab == want)
    out = {"ok": good, "verified": good, "target": V.join_path(parts), "className": cls,
           "sha256Before": before, "sha256": got, "sha256Sent": want, "openTab": bool(r.get("openTab")),
           "requestId": r.get("requestId")}
    if tab is not None:
        out["sha256Tab"] = tab
    if want != got:
        out["error"] = "the source read back after the write is not what was sent"
    elif tab is not None and tab != want:
        out["error"] = "the open editor tab shows something other than what was sent"
    return out


def _fetch_blob(blob_id, timeout=30.0):
    conn = _connect(timeout)
    try:
        conn.request("GET", "/blob?id=%s" % blob_id, headers={"X-Mif-Token": D.token()})
        resp = conn.getresponse()
        data = resp.read()
        return (data if resp.status == 200 else None), resp.status
    finally:
        conn.close()


def captures_dir():
    return os.environ.get("MIF_ROBLOX_CAPTURES") or os.path.join(D.home(), "captures")


def rbx_screenshot(camera: dict = None, max_width: int = 1280, folder: str = None, name: str = None,
                   request_permission: bool = False, route: str = "auto") -> dict:
    "Roblox Studio: capture the viewport to a PNG on this machine and say whether the frame shows anything (blank: too few colors, too little luminance spread, or 60% or more of it Studio's empty viewport color). camera {position, look_at, fov?} frames a view and the person's camera is put back after. route auto uses StudioCaptureService when Studio allows it and CaptureService plus an EditableImage otherwise (on 0.748 a local plugin is not allowed the first); studio forces the first, where request_permission true asks Studio's permission."
    args = {"maxWidth": max_width, "requestPermission": bool(request_permission), "route": route}
    if route not in ("auto", "studio", "capture-service"):
        return {"ok": False, "ran": False, "error": "route is auto, studio or capture-service"}
    if not isinstance(max_width, int) or isinstance(max_width, bool) or not 64 <= max_width <= 4096:
        return {"ok": False, "ran": False, "error": "max_width is 64-4096 pixels"}
    if camera is not None:
        look = camera.get("look_at", camera.get("lookAt")) if isinstance(camera, dict) else None
        extra = sorted(set(camera) - {"position", "look_at", "lookAt", "fov"}) if isinstance(camera, dict) else []
        if not isinstance(camera, dict) or extra or not _vec3(camera.get("position")) or not _vec3(look):
            return {"ok": False, "ran": False, "refusedRule": "type",
                    "error": "camera is {position: [x,y,z], look_at: [x,y,z], fov: degrees (optional)}%s"
                             % ("; unknown keys %s" % extra if extra else "")}
        args["camera"] = {"position": camera["position"], "lookAt": look}
        if camera.get("fov") is not None:
            if not isinstance(camera["fov"], (int, float)) or not 1 <= camera["fov"] <= 120:
                return {"ok": False, "ran": False, "error": "fov is 1-120 degrees"}
            args["camera"]["fov"] = camera["fov"]
    if name is not None and not (isinstance(name, str) and all(c.isalnum() or c in "-_" for c in name) and 0 < len(name) <= 64):
        return {"ok": False, "ran": False, "error": "name goes in a file name: letters, digits, - and _"}
    r = _rbx("screenshot", timeout=90, **args)
    if not r.get("ok"):
        return r
    return _save_capture(r, folder, name)


def _save_rgba(w, h, rgba, name, folder=None):
    """RGBA8 pixels read on this machine (a window's): written as a PNG, read back, judged blank or not."""
    from mif_roblox import png as P
    stats = P.frame_stats(int(w), int(h), rgba)
    folder = folder or captures_dir()
    os.makedirs(folder, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, "%s_%s.png" % (name, stamp))
    n = 1
    while os.path.exists(path):          # two captures in one second never overwrite each other
        n += 1
        path = os.path.join(folder, "%s_%s_%d.png" % (name, stamp, n))
    with open(path, "wb") as fh:
        fh.write(P.encode_rgba(int(w), int(h), rgba))
    back = P.header(open(path, "rb").read())
    problems = []
    if back is None or back[:2] != (int(w), int(h)):
        problems.append("the file read back as %s" % (back,))
    if stats["blank"]:
        problems.append(P.blank_reason(stats))
    out = {"ok": not problems, "verified": not problems, "file": path, "width": w, "height": h, "format": "RGBA8",
           "fileBytes": os.path.getsize(path), "frame": stats}
    if problems:
        out["error"] = "; ".join(problems)
    return out


def _save_capture(r, folder=None, name=None):
    """A frame the plugin sent as a blob (r: blobId, bytes, width, height, format): fetched from the daemon, written as a
    PNG, read back, and judged blank or not. Shared by rbx_screenshot and the play probe's client capture."""
    try:
        data, status = _fetch_blob(r.get("blobId"))
    except (OSError, ValueError) as exc:
        return dict(r, ok=False, error="the image did not come back from the daemon: %s" % exc)
    if data is None:
        return dict(r, ok=False, error="the daemon has no image %s (HTTP %s)" % (r.get("blobId"), status))
    w, h, fmt = r.get("width"), r.get("height"), r.get("format")
    problems = []
    if len(data) != r.get("bytes"):
        problems.append("%d bytes arrived, the plugin sent %s" % (len(data), r.get("bytes")))
    from mif_roblox import png as P
    try:
        if fmt == "PNG":
            hd = P.header(data)
            if hd is None or hd[:2] != (w, h):
                problems.append("the PNG says %s, the capture %sx%s" % (hd and hd[:2], w, h))
            pw, ph, rgba = P.decode_rgba(data)
            out_png = data
        elif fmt == "RGBA8":
            pw, ph, rgba = w, h, data
            out_png = P.encode_rgba(int(w), int(h), data)
        else:
            return dict(r, ok=False, error="Studio gave format %r, which this cannot read" % fmt)
    except (ValueError, zlib.error) as exc:
        return dict(r, ok=False, error="the image could not be read: %s" % exc)
    stats = P.frame_stats(int(pw), int(ph), rgba)
    if stats["blank"]:
        problems.append(P.blank_reason(stats))
    folder = folder or captures_dir()
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "%s_%s.png" % (name or "rbx", time.strftime("%Y%m%d_%H%M%S")))
    with open(path, "wb") as fh:
        fh.write(out_png)
    back = P.header(open(path, "rb").read())
    if back is None or back[:2] != (int(pw), int(ph)):
        problems.append("the file read back as %s" % (back,))
    out = {"ok": not problems, "verified": not problems, "file": path, "width": pw, "height": ph, "format": fmt,
           "originalSize": [r.get("originalWidth"), r.get("originalHeight")], "fileBytes": os.path.getsize(path),
           "frame": stats, "cameraRestored": r.get("cameraRestored"), "uiCaptureMode": r.get("uiCaptureMode"),
           "route": r.get("route"), "studioCanCapture": r.get("studioCanCapture"),
           "pluginSeconds": {k: r.get(k) for k in ("waitedSeconds", "captureSeconds", "scaleSeconds", "encodeSeconds",
                                                    "sendSeconds") if r.get(k) is not None},
           "scaledBy": r.get("scaledBy"), "encodedBy": r.get("encodedBy"),
           "requestId": r.get("requestId")}
    if problems:
        out["error"] = "; ".join(problems)
    return out


def _wait_plugin(seconds, since=None):
    """Poll the daemon's /status until a plugin is connected with a session other than `since` (the session before a
    launch, so a fresh Studio's hello is what ends the wait). Returns (status, waited seconds)."""
    t0 = time.time()
    st = {}
    while True:
        try:
            st = _http("GET", "/status")[1]
        except (OSError, ValueError):
            st = {}
        plugin = st.get("plugin") or {}
        if plugin.get("connected") and plugin.get("sessionId") != since:
            return st, round(time.time() - t0, 1)
        if time.time() - t0 >= seconds:
            return st, round(time.time() - t0, 1)
        time.sleep(0.5)


def _open_place_held(L, place_file, port, plugin):
    """None when the Studio connected to this daemon has place_file open (or the lane's marked copy of it), read from
    its command line's -localPlaceFile. Else the refusal: other-place-open naming what it has open, or place-unverified
    when that cannot be read. Nothing is launched or opened either way."""
    pid = plugin.get("studioPid")
    args = L.CMD_ARGS(pid) if pid else None
    have = L._place_arg(args) if args else None
    norm = lambda p: os.path.normcase(os.path.abspath(p))  # noqa: E731
    want = {norm(place_file), norm(os.path.join(D.home(), "places", str(int(port)), "copies",
                                               os.path.basename(place_file)))}
    if have is not None and norm(have) in want:
        return None
    base = {"ok": False, "ran": False, "launched": False, "studioPid": pid, "placeFile": place_file,
            "placeName": plugin.get("placeName")}
    if have is None:
        return dict(base, refusedRule="place-unverified",
                    error="a Studio%s is connected to port %d, and which place it has open could not be read from its "
                          "command line (%s), so it cannot be shown to be %s: nothing was opened or written. Call "
                          "rbx_launch without place_file to use the open place, or close that Studio first"
                          % (" (pid %s)" % pid if pid else "", int(port),
                             "no -localPlaceFile: opened from Roblox or with File > Open" if args else "unreadable",
                             place_file))
    return dict(base, refusedRule="other-place-open", openPlace=have,
                error="the Studio connected to port %d (pid %s) has %s open, not %s: nothing was opened or written. "
                      "Close it (rbx_close) and launch again, or call rbx_launch without place_file to keep using it"
                      % (int(port), pid, have, place_file))


def _records_busy(fn):
    """rbx_launch and rbx_close: a launch-record lock another process kept for all of its wait (launch.RecordsBusy)
    answers refusedRule records-busy instead of raising out of the tool (found 2026-11-22)."""
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        from mif_roblox import launch as L
        try:
            return fn(*a, **kw)
        except L.RecordsBusy as exc:
            out = {"ok": False, "refusedRule": "records-busy", "lockFile": exc.path, "waitedSeconds": exc.waited,
                   "error": "MifBridge's launch records (%s) stayed locked by another process for %g s, so %s could not "
                            "read or write them: try again" % (exc.path, exc.waited, fn.__name__)}
            if exc.spawnedPid is not None or exc.studioPids:
                out.update(ran=True, spawnedPid=exc.spawnedPid, studioPids=exc.studioPids,
                           warning="Studio was started (pid %s) but is not recorded as launched, so rbx_close will not "
                                   "know it; its provisional record keeps it from play-probe cleanup for %d s"
                                   % (exc.studioPids or exc.spawnedPid, L.PENDING_SECONDS))
            else:
                out["ran"] = False
            return out
    return wrapper


def _undo_detach_on_refusal(fn):
    """A launch that answers ok false ends the detached daemon it started itself (found live 2026-11-26: a launch
    refused copy-exists had already started one, which kept running with no Studio for it)."""
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        _STATE["detachedNow"] = False
        out = fn(*a, **kw)
        if isinstance(out, dict) and out.get("ok") is False and _STATE.get("detachedNow"):
            out = dict(out, daemonStopped=stop_detached_daemon())
        _STATE["detachedNow"] = False
        return out
    return wrapper


@_records_busy
@_undo_detach_on_refusal
def rbx_launch(place_file: str = None, wait_seconds: int = 120, baseplate: bool = True, port: int = None,
               open_original: bool = False, fresh: bool = False, detach_daemon: bool = False) -> dict:
    "Roblox Studio: make sure the daemon is up and a Studio is connected to it. On the default port a running Studio is used, never a second one launched. On another port (a lane: this MCP server started with MIF_ROBLOX_PORT) a Studio already connected to that port's daemon is used; otherwise one is launched on MifBridge's scratch place for that port, which carries ServerStorage.MifBridgePort so the plugin (0.10.0 or later) connects to this daemon and no other. With place_file, that place is opened instead - and a person's file (anything outside MifBridge's own places folder) opens as a COPY in places/<port>/copies on every port, the default one too, so no save from that Studio can land on it (portMarker names the copy and the original); open_original true opens the person's file itself, and only then can rbx_place_save write it, naming it in place_file. Every copy carries ServerStorage.MifBridgeLaunched, so under the person's policy the agent may raise the mode there (rbx_set_mode); an original opened with open_original never does (launched says what was written). On a lane port it must reach this port's daemon, so a file without ServerStorage.MifBridgePort = port is opened as a copy in places/<port>/copies with the marker written in (the person's file is never written; portMarker says which); a binary .rbxl without it is opened the same way, as a binary copy with the IntValue added to its ServerStorage (portMarker binary true), and one the bridge cannot mark (a zstd chunk with no zstd in this Python, or an IntValue property of a type it cannot extend) is refused no-port-marker naming why. A launch that would take Studio past its instance-count warning is refused studio-instance-limit, naming the running Studios by pid. A marked copy already in places/<port>/copies that differs from a fresh copy of place_file is never written over (refusedRule copy-exists, naming both files and their times): it is the file a lane Studio had open, so it may hold saved work; open the copy itself as place_file, or pass fresh true to replace it. With place_file, a Studio already connected to this port is used only when its command line shows that place (or its marked copy) open, else refused other-place-open (naming what it has open) or place-unverified. A new Studio process is recorded as this launch's only when its command line holds the place and it is no play-test window; notTaken lists the others and why. The scratch place opens on TextChatService. Before a launch, Studio's Auto Recovery files of the bridge's scratch place (MifScratch_AutoRecovery_*) are moved to AutoSaves/Archived as Studio's Ignore names them (never deleted, never another place's); if Studio's Auto Recovery File Found dialog still shows in the Studio this call started, on the scratch place it is answered Ignore on that pid only, and on any other place it is left for the person; recovery says what was moved and done. When the launch records' lock stays held by another process past its wait, it answers refusedRule records-busy (lockFile), with spawnedPid and a warning when Studio was already started. Studio reuses the person's own sign-in; nothing here touches credentials. detach_daemon true (or MIF_ROBLOX_DETACHED_DAEMON=1 for any first call) starts this port's daemon as a process of its own - python -m mif_roblox.daemon --port N, detached, no window, its output in places/<port>/daemon.log - so it outlives the calling process and the next process drives the same Studio with no gap; its pid and start time are recorded in places/<port>/daemon.json, rbx_status names the host (daemon.host), rbx_close ends it by that pid with its Studio (only while that pid's start time and command line still match), and a launch refused after starting one ends it again (daemonStopped). By default the daemon runs inside the calling process, as before."
    from mif_roblox import gen_catalog as G
    from mif_roblox import launch as L
    if not isinstance(wait_seconds, int) or isinstance(wait_seconds, bool) or not 5 <= wait_seconds <= 600:
        return {"ok": False, "ran": False, "error": "wait_seconds is 5-600"}
    if not isinstance(fresh, bool) or not isinstance(detach_daemon, bool):
        return {"ok": False, "ran": False, "error": "fresh and detach_daemon are true or false"}
    port = PORT if port is None else port
    if port != PORT:
        return {"ok": False, "ran": False, "refusedRule": "port",
                "error": "this MCP server's daemon is on port %d; a Studio for port %s needs a server started with "
                         "MIF_ROBLOX_PORT=%s" % (PORT, port, port)}
    if not isinstance(open_original, bool):
        return {"ok": False, "ran": False, "error": "open_original is true or false"}
    if place_file is not None and not (isinstance(place_file, str) and os.path.isfile(place_file)
                                       and place_file.lower().endswith((".rbxl", ".rbxlx"))):
        return {"ok": False, "ran": False, "error": "place_file is an existing .rbxl or .rbxlx file"}
    _STATE["studio"] = None             # a launch takes the Studio anew; it pins the one it confirms below
    detached = None
    if detach_daemon:
        detached = start_detached_daemon()
        if not detached.get("ok"):
            return detached
        _STATE["detachedNow"] = bool(detached.get("started"))
    ready = ensure_daemon()
    if not ready.get("ok"):
        return ready
    studios = L.running()
    out = {"daemon": {"pid": ready.get("pid"), "port": PORT, "started": ready.get("started")}}
    if detached is not None:
        out["daemon"].update(detached=bool(detached.get("detached")), detachedStarted=bool(detached.get("started")),
                             log=detached.get("log"))
    prior = None
    clearer, clear = None, {}
    connected = bool(((ready.get("status") or {}).get("plugin") or {}).get("connected"))
    # A running Studio is used only when it is connected to THIS daemon: with lanes, other Studios run on other ports,
    # and waiting for one of them to reach 8650 never ends (found 2026-11-19 opening a game's place beside two lane Studios).
    if studios and connected and place_file:
        # found 2026-11-22: with place_file, a connected Studio was used whatever it had open, so a session that asked
        # for one place wrote into another. The connected Studio's place is read from its command line and compared.
        held = _open_place_held(L, place_file, port, (ready.get("status") or {}).get("plugin") or {})
        if held:
            return dict(out, **held)
    if studios and connected:
        out.update(launched=False, studioPids=[p for p, _s in studios],
                   note="a Studio is already running: it is used, and no second one is launched" if port == D.DEFAULT_PORT
                   else "a Studio is already connected to this lane's daemon on port %d" % port)
    else:
        if studios:
            out["note"] = "Studios are running, none connected to port %d: a Studio of its own is launched for it" % port
        busy = L.instance_limit()
        if busy:
            return dict(out, **busy)
        exe = G.find_studio()
        if not exe:
            return dict(out, ok=False, ran=False, error="no Roblox Studio is installed under %LOCALAPPDATA%/Roblox/Versions")
        try:
            prior = (_http("GET", "/status")[1].get("plugin") or {}).get("sessionId")
        except (OSError, ValueError):
            prior = None
        place = None if place_file else L.scratch_place(port)
        if place_file:
            mk = L.marked_place(place_file, port, open_original=open_original is True, fresh=fresh)
            if mk.get("ran") is False:
                return dict(out, **mk)
            place = mk["place"]
            out["portMarker"] = mk
        # Studio's Auto Recovery dialog comes up for any MifScratch_AutoRecovery_* file, a crashed or killed bridge
        # Studio's or a live lane's (RB2720): the bridge's own are moved the way Studio's Ignore moves them, first.
        out["recovery"] = {"archivedBeforeLaunch": L.archive_recovery(L.SCRATCH_STEM)}
        try:
            rec = L.launch(exe, place, port=port)
        except L.RecordsBusy:
            raise                       # answered records-busy by _records_busy, saying whether Studio was started
        except OSError as exc:
            return dict(out, ok=False, error="Studio could not be started: %s" % exc)
        t_spawn = rec.pop("t0", None)
        out.update(launched=True, exe=exe, scratchPlace=place_file is None, **rec)
        if L.place_stem(place).lower() == L.SCRATCH_STEM.lower():
            # other lanes' Studios keep writing MifScratch files: keep the stem clear until after the new Studio looks
            clearer = threading.Thread(target=lambda: clear.update(moved=L.keep_clear(
                L.SCRATCH_STEM, (t_spawn or time.time()) + L.RECOVERY_CLEAR_SECONDS)), daemon=True)
            clearer.start()
        if not rec["studioPids"]:
            out["warning"] = ("no new RobloxStudioBeta process whose command line holds this place showed in the process "
                              "table within 8 s of the launch" + (" (notTaken says which others did)" if rec.get("notTaken") else ""))
    st, waited = _wait_plugin(wait_seconds, since=prior)
    plugin = st.get("plugin") or {}
    out.update(plugin=plugin, waitedSeconds=waited)
    if not plugin.get("connected"):
        return dict(out, ok=False, editorState="noPlugin", setup=setup_steps(),
                    error="no MifBridge plugin connected within %ds: Studio may still be loading, the plugin may not be "
                          "installed, or Connect automatically is off in its panel" % wait_seconds)
    # The Studio's PID, measured (spec "Extend rbx_launch: return the launched Studio's PID"): the daemon's TCP peer of
    # the plugin's hello, the launch record and the spawned process, each taken only when it is a running Studio whose
    # command line holds this place file. A Studio found only now is recorded, so rbx_close knows it.
    place_now = out.get("place") or ((L.ours(port) or [{}])[0].get("place"))
    cands = ([plugin.get("studioPid")] + list(out.get("studioPids") or []) + [out.get("spawnedPid")]
             + [r.get("pid") for r in L.ours(port)])
    pid, start, checks = L.confirm_pid(cands, place_now)
    out.update(pid=pid, pidChecks=checks)
    if pid is not None and (port is None or int(port) == PORT):
        _STATE["studio"] = pid          # this process's Studio: a daemon whose plugin answers for another is refused
    if pid is None:
        out["pidReason"] = ("no candidate is a running Studio whose command line holds %s" % place_now if checks
                            else "no candidate pid: neither the plugin's connection nor the process table named one")
    elif out.get("launched") and pid not in (out.get("studioPids") or []):
        out["adopted"] = L.adopt(pid, start, out.get("exe"), out.get("args"), place_now, port)
    if clearer is not None:
        clearer.join(L.RECOVERY_CLEAR_SECONDS + 5)
        out["recovery"]["archivedWhileStarting"] = clear.get("moved", [])
    if out.get("launched") and pid is not None:
        # a file written between the move and Studio's own check (another lane's Studio auto-saving) still raises the
        # dialog: on the scratch place it is answered Ignore on this Studio's pid only; on any other place it is left
        out["recovery"]["dialog"] = L.watch_recovery_dialog(pid, place_now, t_spawn or time.time())
    if out.get("launched") and out.get("scratchPlace") and baseplate:
        root = rbx_find(name="MifScratch", root="Workspace")
        if not root.get("matches"):
            rbx_create("Folder", "Workspace", name="MifScratch")
        bp = rbx_find(name="Baseplate", root="Workspace.MifScratch")
        if not bp.get("matches"):
            made = rbx_create("Part", "Workspace.MifScratch", name="Baseplate", props={
                "Anchored": True, "Size": [512, 1, 512], "CFrame": {"position": [0, -0.5, 0]}, "Material": "Plastic",
                "Color": [0.64, 0.64, 0.64]})
            out["baseplate"] = {k: made.get(k) for k in ("ok", "verified", "path", "error")}
    out["ok"] = True
    return out


@_reports
def rbx_set_mode(mode: str) -> dict:
    "Roblox Studio: change the plugin's safety mode (read, scratch, full). Lowering is always allowed. Raising needs the person's machine-local opt-in (%LOCALAPPDATA%/MifBridge/roblox_policy.json with agentMayRaiseMode true, scope bridge-launched - MifBridge never writes or ships it), a place rbx_launch made (ServerStorage.MifBridgeLaunched), and no mode click by the person this session; otherwise refused policy-off, not-bridge-launched or person-wins, saying why. The panel shows an agent-set mode in its own colour and its call log records every change; a click in the panel always wins. Read back: the mode the plugin reports afterwards."
    if mode not in D.MODES:
        return {"ok": False, "ran": False, "error": "mode is read, scratch or full"}
    pol = D.read_policy()
    before = _rbx("ping", nonce="set_mode")
    if not before.get("ok"):
        return before
    have = before.get("mode")
    rank = {m: i for i, m in enumerate(D.MODES)}
    if have in rank and rank[mode] > rank[have] and not pol["agentMayRaiseMode"]:
        return {"ok": False, "ran": False, "refusedRule": "policy-off", "mode": have, "policyFile": pol["file"],
                "policyFound": pol["found"],
                "error": "raising %s to %s needs the person's opt-in: %s %s. Only the person writes that file; the panel's "
                         "buttons always work" % (have, mode, pol["file"],
                                                  "does not allow it" if pol["found"] else "does not exist")}
    if before.get("modeBy") is None and rank.get(mode, 0) > rank.get(have, 0):
        return {"ok": False, "ran": False, "refusedRule": "plugin-too-old", "plugin": before.get("plugin"),
                "error": "plugin %s cannot have its mode set by the agent (0.11.0 can); it is in %s, and the person's "
                         "click changes it" % (before.get("plugin"), have)}
    r = _rbx("set_mode", mode=mode)
    if not r.get("ok"):
        return r
    after = _rbx("ping", nonce="set_mode-after")
    ok = after.get("mode") == mode
    return {"ok": ok, "verified": ok, "mode": after.get("mode"), "was": r.get("was"), "by": after.get("modeBy"),
            "launchMarked": after.get("launchMarked"), "requestId": r.get("requestId"),
            **({} if ok else {"error": "the plugin reports %s after set_mode %s" % (after.get("mode"), mode)})}


@_records_busy
def rbx_close(discard: bool = False) -> dict:
    "Roblox Studio: close the Studio that rbx_launch started - by its pid, never a Studio the person opened. On MifBridge's own scratch place (MifScratch.rbxlx in its places folder) it is ended without saving (nothing of the person's is in that file). The place is judged by what the Studio has open now, read from its plugin (placeName and PlaceId, openNow in the answer), not by the place it was launched on: another name, or a PlaceId above 0, is refused place-changed unless discard is true (a pid stop of a place opened from Roblox committed its unsaved edits, RB1968), and a scratch launch whose open place cannot be read (no plugin, or the plugin on this port answers for another Studio) is refused place-unread unless discard is true. On any other place, a marked copy of a person's place in places/<port>/copies included (it holds their content), it is refused unless discard is true: Studio's own close raises a Save changes? dialog even on an unchanged place (measured on 0.748), and the bridge never puts a dialog on the person's screen. Right before ending it, the Studio's command line is read again: one that does not hold the recorded place is refused not-ours-place and left running. When no launched Studio is found for this port it is refused not-ours, and the answer says why: records (each launch record read from recordFile, its pid, port and place, and taken or why not - not running, a reused pid, another port, still starting), droppedRecently (records dropped lately for this port: when, by which process, why) and running (every running Studio's pid, start and place from its command line). After its Studio exits, the Auto Recovery files of the place it had open, and no other place's, are moved to AutoSaves/Archived, so the next launch opens without Studio's recovery dialog: the scratch place's all, a marked copy's only those written while it ran, a place outside the places folder or one it only switched to none; nothing moves while another running Studio has, or may have, a place of that name open (other lanes' Studios on their own scratch places share the scratch name). recovery lists what moved and what was kept, and why. When the launch records' lock stays held by another process past its wait, it answers refusedRule records-busy (lockFile) instead of failing."
    from mif_roblox import launch as L
    if not isinstance(discard, bool):
        return {"ok": False, "ran": False, "error": "discard is true or false"}
    read = []
    mine = L.ours(PORT, explain=read)  # each lane closes only the Studio it launched for its own port
    if not mine:
        # the internal notes L140: say why - the records read for this port, the ones dropped lately, the running Studios' places
        return L.not_ours(PORT, read)
    if len(mine) > 1:
        return {"ok": False, "ran": False, "refusedRule": "ambiguous", "studioPids": [r["pid"] for r in mine],
                "error": "more than one launched Studio is recorded; close needs exactly one"}
    pid, recorded = mine[0]["pid"], mine[0].get("place")
    # What the Studio has open NOW, not what it was launched on (found 2026-11-22): a person may have opened another place
    # in it, and a pid stop of a place opened from Roblox committed its unsaved edits a minute later (RB1968).
    ping, now = _open_place_now(pid, recorded)
    if now.get("changed") and not discard:
        return {"ok": False, "ran": False, "refusedRule": "place-changed", "pid": pid, "place": recorded, "openNow": now,
                "error": "pid %s was launched on %s, but it has %s open now (PlaceId %s): nothing closed. That place may be "
                         "the person's; pass discard true to end it without saving, or let the person close it"
                         % (pid, recorded, now.get("placeName"), now.get("placeId"))}
    recorded_scratch = L.is_scratch_place(recorded)
    if recorded_scratch and not now.get("read") and not discard:
        return {"ok": False, "ran": False, "refusedRule": "place-unread", "pid": pid, "place": recorded, "openNow": now,
                "error": "pid %s was launched on MifBridge's scratch place, but what it has open now could not be read "
                         "(%s): nothing closed. Pass discard true to end it without saving, or let the person close it"
                         % (pid, now.get("why"))}
    scratch = recorded_scratch and not now.get("changed")
    if not discard and not scratch:
        # 2026-11-18, 0.748: WM_CLOSE on a freshly opened, unchanged file place (GetCanUndo false) still raised
        # "Save changes to MifScratch.rbxlx? Save / Don't Save / Cancel" on the person's screen. So there is no graceful
        # close: a place that is not ours is refused, whatever its undo state, until a person says discard.
        return {"ok": False, "ran": False, "refusedRule": "would-prompt", "pid": pid, "place": recorded,
                "canUndo": ping.get("canUndo"), "openNow": now,
                "error": "closing this Studio would put Studio's Save changes? dialog on the person's screen, and this place "
                         "is not MifBridge's scratch place: nothing closed. Pass discard true to end it without saving, "
                         "or let the person close it"}
    res = L.close(pid, True)
    if res.get("closed"):
        # ending Studio by pid leaves its Auto Recovery file, which would raise the dialog at the next launch (RB2724):
        # only the files of the place it had open move (a place it switched to is not known by file, so none do)
        res["recovery"] = L.archive_after_close(None if now.get("changed") else recorded, mine[0].get("start"))
    if res.get("closed") and read_detached_daemon() is not None:
        # the detached daemon rbx_launch started for this lane goes with its Studio
        res["daemon"] = stop_detached_daemon()
    return dict(res, ok=bool(res.get("closed")), pid=pid, place=recorded, openNow=now, discarded=True,
                scratchPlace=scratch, error=None if res.get("closed") else res.get("error") or "Studio is still running")


def _open_place_now(pid, recorded):
    """(ping, openNow) for rbx_close: the place the plugin in Studio pid has open now. openNow: read (the plugin connected
    to this daemon answered and is pid's), placeName, placeId, changed (read, and not the recorded file: another name, or
    a PlaceId above 0 - a local file opens with PlaceId 0), why (when not read)."""
    own = _own_studio_pid()
    if own != pid:
        return {}, {"read": False, "why": "the plugin on port %d answers for %s, not pid %s" % (
            PORT, "Studio pid %s" % own if own else "no known Studio", pid)}
    ping = _rbx("ping", nonce="close", timeout=15)
    name = ping.get("placeName") if ping.get("ok") else None
    if not isinstance(name, str):
        return ping, {"read": False, "why": ping.get("error") or "the plugin's ping named no place"}
    pid_now = ping.get("placeId")
    want = os.path.basename(recorded or "")
    changed = name.lower() != want.lower() or (isinstance(pid_now, int) and pid_now > 0)
    return ping, {"read": True, "placeName": name, "placeId": pid_now, "changed": changed, "recordedName": want}


# ---------------------------------------------------------------- batch 0: the generic layer
# A member is judged read-only BY NAME, the only signal the dump gives: these prefixes read; everything else may write and
# runs as one undo recording, gated by the safety mode like any write. Named, not guessed, in the internal notes.
READ_PREFIXES = ("Get", "Find", "Is", "Has", "Can", "Query", "Raycast", "Blockcast", "Spherecast", "Shapecast", "Read",
                 "Compute", "Check", "Count", "ArePartsTouching", "WaitFor", "Search", "IsA", "IsDescendantOf",
                 "IsAncestorOf", "ToWorld", "ToObject", "Sample", "Evaluate",
                 # makes an object only the plugin holds (a handle), not place data
                 "CreateEditable")
# A setter and the getter that proves it: (getter, args from the call's args, what the getter must answer).
PAIRED = {
    "SetAttribute": ("GetAttribute", lambda a: a[:1], lambda a: a[1] if len(a) > 1 else {"$t": "nil"}),
    "PivotTo": ("GetPivot", lambda a: [], lambda a: a[0]),
    "AddTag": ("HasTag", lambda a: a[:1], lambda a: True),
    "RemoveTag": ("HasTag", lambda a: a[:1], lambda a: False),
    "SetPrimaryPartCFrame": ("GetPrimaryPartCFrame", lambda a: [], lambda a: a[0]),
    "ScaleTo": ("GetScale", lambda a: [], lambda a: a[0]),
}


# Setters never paired by the derived rule below: live game data (decision 5), the plugin's own settings, the CoreGui,
# and anything whose getter answers a Tuple.
NO_DERIVED_PAIR = {"SetAsync", "SetSetting", "SetCore", "SetServerAttribute", "SetTeleportSetting", "SetSharedTable"}


def derived_pair(cat, cls, method):
    """Set<X>(k..., value) proved by Get<X>(k...) on the same class, when the catalog shows exactly that shape: the
    getter's parameter types are the setter's without its last, the getter answers one value (not a Tuple), and that
    value's type is the setter's last parameter's type (a getter's `?` allowed). The getter is named in the answer
    (pairedBy "catalog"), so a caller sees what proved the write. None when the catalog does not show that shape."""
    if not method.startswith("Set") or method in NO_DERIVED_PAIR:
        return None
    _o, srow = cat.find_member(cls, method)
    getter = "Get" + method[3:]
    _g, grow = cat.find_member(cls, getter)
    if srow is None or grow is None or grow[0] != "F" or srow[0] != "F":
        return None
    sp, gp = [p[1] for p in srow[5]], [p[1] for p in grow[5]]
    if not sp or gp != sp[:-1] or str(grow[1] or "").startswith("Tuple"):
        return None
    if str(grow[1]).rstrip("?") != str(sp[-1]).rstrip("?") or grow[4] and "Yields" in grow[4] and "Yields" not in srow[4]:
        return None
    n = len(gp)
    return (getter, lambda a, n=n: a[:n], lambda a, n=n: a[n] if len(a) > n else {"$t": "nil"}, "catalog")


# The proof table (spec "a proof table for writes that have no Get twin"): a write whose name has no Get<X> twin, keyed by
# the member's owner as the catalog gives it ("Owner:Method"), and the read that proves it:
#   ("getter", name, args_from_call, expected)   another member of the same target, called read-only after the write
#   ("property", name, expected_from(args, before))   a property of the target, read before and after the write
#   ("event", name, expected_args)   an event of the target, watched from before the call; one firing must carry them
#   ("return", None)   the call's own answer, judged by the row's test (what Instance:Clone or CreatePath made)
# A row may also carry (spec "test_roblox_live_cov_writes"):
#   "on": fn(wire_args, target_parts) -> path parts   read on another instance: one the arguments name (the item
#         Debris:AddItem was given), a service (LogService for TestService:Message), or the target's parent
#   "test": fn(decoded_read, ctx) -> True   judge the read instead of comparing it with the expected value; ctx holds the
#           decoded "args", "before" (a property's value, or a getter's answer with "before" true) and "returns"
#   "says": the test in words, answered beside the read;  "before": True   call a getter row before the call too
#   "argsFrom": "return"   a getter row's args fn takes (wire_args, wire_returns): the id EditableMesh:AddVertex answered
#   "wait": seconds an event row watches (default PROOF_EVENT_WAIT)
# Each row is added only after a live run showed the proof holds, and names its the internal notes. It is checked before PAIRED,
# whose rows are keyed by name alone (CollectionService:AddTag(instance, tag) is not Instance:AddTag(tag)). An offline test
# counts the rows, so the table cannot shrink unseen.
_AXIS = {"Top": 1, "Bottom": 1, "Left": 0, "Right": 0, "Front": 2, "Back": 2}


def _resized(a, before):
    """BasePart:Resize(normalId, delta): Size grows by delta along the face's axis, whichever side (0.748)."""
    v = list((before or {}).get("v") or [])
    axis = _AXIS.get(str((a[0] or {}).get("v") if isinstance(a[0], dict) else a[0]))
    if len(v) != 3 or axis is None or not _num_like(a[1]):
        return None
    v[axis] = v[axis] + a[1]
    return {"$t": "Vector3", "v": v}


def _num_like(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _leaf(p):
    return str(p).split(".")[-1]


def _leaves(xs):
    return [_leaf(x) for x in xs] if isinstance(xs, list) else []


def _gained(got, ctx):
    """The instance (or name) the call was given is in the list now, and was not before."""
    a = _leaf(ctx["args"][0])
    return a in _leaves(got) and a not in _leaves(ctx["before"])


def _lost(got, ctx):
    a = _leaf(ctx["args"][0])
    return a not in _leaves(got) and a in _leaves(ctx["before"])


def _gained_all(got, ctx):
    want = _leaves(ctx["args"][0])
    return bool(want) and all(w in _leaves(got) for w in want) and not all(w in _leaves(ctx["before"]) for w in want)


def _lost_all(got, ctx):
    want = _leaves(ctx["args"][0])
    return bool(want) and not any(w in _leaves(got) for w in want) and all(w in _leaves(ctx["before"]) for w in want)


def _one_more(got, ctx):
    return _num_like(got) and _num_like(ctx["before"]) and got == ctx["before"] + 1


def _n_less(got, ctx):
    return _num_like(got) and _num_like(ctx["before"]) and got == ctx["before"] - ctx["args"][1]


def _more(got, ctx):
    return _num_like(got) and _num_like(ctx["before"]) and got > ctx["before"]


def _longer(got, ctx):
    return isinstance(got, list) and isinstance(ctx["before"], list) and len(got) > len(ctx["before"])


def _shorter(got, ctx):
    return isinstance(got, list) and isinstance(ctx["before"], list) and len(got) < len(ctx["before"])


def _changed(got, ctx):
    return got is not None and got != ctx["before"]


def _is_arg0(got, ctx):
    return got is not None and _leaf(got) == _leaf(ctx["args"][0])


def _child(name):
    return lambda got, ctx: isinstance(got, str) and _leaf(got) == name


def _says_message(got, ctx):
    """A MessageOut firing whose message holds the text the call was given (TestService prefixes "TestService: ")."""
    texts = [a for a in ctx["args"] if isinstance(a, str) and a]
    return isinstance(got, list) and bool(got) and isinstance(got[0], str) and any(t in got[0] for t in texts)


def _made(cls):
    """A return-kind test: the call answered an object of class cls (a handle, alone or in a list or a dict)."""
    def test(got, ctx):
        found = []

        def walk(x):
            if isinstance(x, dict):
                if isinstance(x.get("handle"), str):
                    found.append(x.get("className"))
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(ctx["returns"])
        return cls in found
    return test


def _pixel(got, ctx, color):
    """An EditableImage test: the 1 x 1 ReadPixelsBuffer read is color (RGBA bytes, 2 of 255 slack)."""
    import base64
    raw = base64.b64decode((got or {}).get("base64") or "") if isinstance(got, dict) else b""
    want = [round(color[k] * 255) for k in "rgb"]
    return len(raw) >= 4 and all(abs(raw[i] - want[i]) <= 2 for i in range(3))


def _pixel_arg(i):
    return lambda got, ctx: isinstance(ctx["args"][i], dict) and _pixel(got, ctx, ctx["args"][i])


def _px1(a):
    return [a[0], {"$t": "Vector2", "v": [1, 1]}]


def _none(a):
    return []


def _pivot_by(got, ctx):
    d, b = ctx["args"][0], (ctx["before"] or {}).get("position")
    p = (got or {}).get("position")
    return isinstance(p, list) and isinstance(b, list) and all(
        abs(p[i] - (b[i] + d[k])) <= 1e-3 for i, k in enumerate("xyz"))


def _moved(got, ctx):
    return isinstance(got, dict) and isinstance(ctx["before"], dict) and got.get("position") != ctx["before"].get(
        "position")


def _G(name, test, says, args=_none, **kw):
    """A getter row judged by a test (a list read before and after, mostly)."""
    return dict({"by": ("getter", name, args, None), "test": test, "says": says, "before": True}, **kw)


def _P(name, test, says, **kw):
    return dict({"by": ("property", name, None), "test": test, "says": says}, **kw)


def _E(name, test, says, **kw):
    return dict({"by": ("event", name, None), "test": test, "says": says}, **kw)


def _R(cls):
    return {"by": ("return", None), "test": _made(cls), "says": "the call answered a %s" % cls}


def _ledger(rows, ledger):
    """The writes suite's rows: each names its the internal notes, and is soft - a proof that does not show answers verified
    false with ok kept, since composite tools (rbx_physics_step, rbx_input_fire) send these members too."""
    return {k: dict(v, ledger=ledger, soft=True) for k, v in rows.items()}


_LOG = {"on": lambda a, t: ["LogService"]}
_CELLS = _G("CountCells", _more, "Terrain:CountCells reads more cells after than before")
_GROUP_ON = ("getter", "IsCollisionGroupRegistered", lambda a: a[:1], lambda a: True)
_GROUP_OFF = ("getter", "IsCollisionGroupRegistered", lambda a: a[:1], lambda a: False)
_GROUP_NEW = ("getter", "IsCollisionGroupRegistered", lambda a: a[1:2], lambda a: True)
_GROUP_PAIR = ("getter", "CollisionGroupsAreCollidable", lambda a: a[:2], lambda a: a[2])
_ON_FIRST = {"on": lambda a, t: a[0]["v"][0]["path"]}
_FROM_RET = {"argsFrom": "return"}

PROOFS = {
    "CollectionService:AddTag": {"by": ("getter", "HasTag", lambda a: a[:2], lambda a: True), "ledger": "RB2242"},
    "CollectionService:RemoveTag": {"by": ("getter", "HasTag", lambda a: a[:2], lambda a: False), "ledger": "RB2242"},
    "BasePart:Resize": {"by": ("property", "Size", _resized), "ledger": "RB2244"},
    "BindableEvent:Fire": {"by": ("event", "Event", lambda a: list(a)), "ledger": "RB2245"},
}
# Spec "test_roblox_live_cov_writes": each row below ran live through rbx_call on MifBridge's scratch place, 0.748
#, and its the internal notes says what was read.
PROOFS.update(_ledger({     # animation data and curves (RB2520)
    "Keyframe:AddPose": _G("GetPoses", _gained, "GetPoses holds the pose now, not before"),
    "Keyframe:RemovePose": _G("GetPoses", _lost, "GetPoses held the pose before, not now"),
    "Keyframe:AddMarker": _G("GetMarkers", _gained, "GetMarkers holds the marker now, not before"),
    "Keyframe:RemoveMarker": _G("GetMarkers", _lost, "GetMarkers held the marker before, not now"),
    "KeyframeSequence:AddKeyframe": _G("GetKeyframes", _gained, "GetKeyframes holds the keyframe now, not before"),
    "KeyframeSequence:RemoveKeyframe": _G("GetKeyframes", _lost, "GetKeyframes held the keyframe before, not now"),
    "Pose:AddSubPose": _G("GetSubPoses", _gained, "GetSubPoses holds the pose now, not before"),
    "Pose:RemoveSubPose": _G("GetSubPoses", _lost, "GetSubPoses held the pose before, not now"),
    "AnimationNodeDefinition:AddInputPin": _G("GetOrderedInputPinNames", _gained, "the pin is named now, not before"),
    "AnimationNodeDefinition:RemoveInputPin": _G("GetOrderedInputPinNames", _lost, "the pin was named before, not now"),
    "FloatCurve:InsertKey": _P("Length", _one_more, "Length reads one more"),
    "FloatCurve:RemoveKeyAtIndex": _P("Length", _n_less, "Length reads count fewer"),
    "MarkerCurve:InsertMarkerAtTime": _P("Length", _one_more, "Length reads one more"),
    "MarkerCurve:RemoveMarkerAtIndex": _P("Length", _n_less, "Length reads count fewer"),
    "RotationCurve:InsertKey": _P("Length", _one_more, "Length reads one more"),
    "RotationCurve:RemoveKeyAtIndex": _P("Length", _n_less, "Length reads count fewer"),
    "EulerRotationCurve:X": {"by": ("getter", "FindFirstChild", lambda a: ["X"], None), "test": _child("X"),
                             "says": "the X child curve exists after the call"},
    "EulerRotationCurve:Y": {"by": ("getter", "FindFirstChild", lambda a: ["Y"], None), "test": _child("Y"),
                             "says": "the Y child curve exists after the call"},
    "EulerRotationCurve:Z": {"by": ("getter", "FindFirstChild", lambda a: ["Z"], None), "test": _child("Z"),
                             "says": "the Z child curve exists after the call"},
    "Vector3Curve:X": {"by": ("getter", "FindFirstChild", lambda a: ["X"], None), "test": _child("X"),
                       "says": "the X child curve exists after the call"},
    "Vector3Curve:Y": {"by": ("getter", "FindFirstChild", lambda a: ["Y"], None), "test": _child("Y"),
                       "says": "the Y child curve exists after the call"},
    "Vector3Curve:Z": {"by": ("getter", "FindFirstChild", lambda a: ["Z"], None), "test": _child("Z"),
                       "says": "the Z child curve exists after the call"},
}, "RB2520"))
PROOFS.update(_ledger({     # UI, style, localization, selection, avatar description (RB2521)
    "Path2D:InsertControlPoint": _G("GetControlPoints", _longer, "GetControlPoints is one longer"),
    "Path2D:RemoveControlPoint": _G("GetControlPoints", _shorter, "GetControlPoints is one shorter"),
    "Path2D:UpdateControlPoint": {"by": ("getter", "GetControlPoint", lambda a: a[:1], lambda a: a[1])},
    "UIPageLayout:JumpTo": _P("CurrentPage", _is_arg0, "CurrentPage is the page given"),
    "UIPageLayout:JumpToIndex": _P("CurrentPage", _changed, "CurrentPage moved"),
    "UIPageLayout:Next": _P("CurrentPage", _changed, "CurrentPage moved"),
    "UIPageLayout:Previous": _P("CurrentPage", _changed, "CurrentPage moved"),
    "StyleBase:InsertStyleRule": _G("GetStyleRules", _gained, "GetStyleRules holds the rule now, not before"),
    "StyleRule:SetPropertyTransition": _G("GetPropertyTransitions", lambda g, c: isinstance(g, dict) and c["args"][0] in g
                                          and c["args"][0] not in (c["before"] if isinstance(c["before"], dict) else {}),
                                          "GetPropertyTransitions names the property now, not before"),
    "LocalizationTable:SetEntries": {"by": ("getter", "GetEntries", _none, None), "says": "GetEntries holds the keys given",
                                     "test": lambda g, c: sorted(e.get("Key") for e in g) == sorted(
                                         e.get("Key") for e in c["args"][0])},
    "LocalizationTable:SetEntryValue": {"by": ("getter", "GetEntries", _none, None), "says": "the entry's value is set",
                                        "test": lambda g, c: any(e.get("Key") == c["args"][0] and (
                                            e.get("Values") or {}).get(c["args"][3]) == c["args"][4] for e in g)},
    "LocalizationTable:SetEntryExample": {"by": ("getter", "GetEntries", _none, None), "says": "the entry's example is set",
                                          "test": lambda g, c: any(e.get("Key") == c["args"][0] and e.get("Example")
                                                                   == c["args"][3] for e in g)},
    "LocalizationTable:SetEntryContext": {"by": ("getter", "GetEntries", _none, None), "says": "the entry's context is set",
                                          "test": lambda g, c: any(e.get("Key") == c["args"][0] and e.get("Context")
                                                                   == c["args"][3] for e in g)},
    "LocalizationTable:SetEntrySource": {"by": ("getter", "GetEntries", _none, None), "says": "the entry's source is set",
                                         "test": lambda g, c: any(e.get("Key") == c["args"][0] and e.get("Source")
                                                                  == c["args"][3] for e in g)},
    "LocalizationTable:SetEntryKey": {"by": ("getter", "GetEntries", _none, None), "says": "the entry has the new key only",
                                      "test": lambda g, c: any(e.get("Key") == c["args"][3] for e in g)
                                      and not any(e.get("Key") == c["args"][0] for e in g)},
    "LocalizationTable:RemoveEntryValue": {"by": ("getter", "GetEntries", _none, None), "before": True,
                                           "says": "the locale's value was there and is gone",
                                           "test": lambda g, c: any(e.get("Key") == c["args"][0] and c["args"][3] not in (
                                               e.get("Values") or {}) for e in g) and any(
                                               c["args"][3] in (e.get("Values") or {}) for e in c["before"])},
    "LocalizationTable:RemoveTargetLocale": {"by": ("getter", "GetEntries", _none, None), "before": True,
                                             "says": "no entry holds the locale now; one did before",
                                             "test": lambda g, c: all(c["args"][0] not in (e.get("Values") or {}) for e in g)
                                             and any(c["args"][0] in (e.get("Values") or {}) for e in c["before"])},
    "LocalizationTable:RemoveEntry": {"by": ("getter", "GetEntries", _none, None), "before": True,
                                      "says": "the key was there and is gone",
                                      "test": lambda g, c: not any(e.get("Key") == c["args"][0] for e in g)
                                      and any(e.get("Key") == c["args"][0] for e in c["before"])},
    "Selection:Add": _G("Get", _gained_all, "Selection:Get holds them now, not before"),
    "Selection:Remove": _G("Get", _lost_all, "Selection:Get held them before, not now"),
    "HumanoidDescription:AddEmote": _G("GetEmotes", lambda g, c: isinstance(g, dict) and c["args"][0] in g and c["args"][
        0] not in (c["before"] if isinstance(c["before"], dict) else {}), "GetEmotes names the emote now, not before"),
    "HumanoidDescription:RemoveEmote": _G("GetEmotes", lambda g, c: c["args"][0] not in (g if isinstance(g, dict) else {})
                                          and isinstance(c["before"], dict) and c["args"][0] in c["before"],
                                          "GetEmotes named the emote before, not now"),
}, "RB2521"))
PROOFS.update(_ledger({     # models, joints, physics in Edit, characters, seats, tools (RB2522)
    "Model:TranslateBy": _G("GetPivot", _pivot_by, "GetPivot moved by exactly the offset"),
    "Model:MoveTo": _G("GetPivot", _moved, "GetPivot moved"),
    "Motor:SetDesiredAngle": {"by": ("property", "DesiredAngle", lambda a, b: a[0])},
    "Workspace:JoinToOutsiders": _G("GetJoints", _longer, "the part's GetJoints is longer", **_ON_FIRST),
    "Workspace:UnjoinFromOutsiders": _G("GetJoints", _shorter, "the part's GetJoints is shorter", **_ON_FIRST),
    "WorldRoot:BulkMoveTo": _P("CFrame", lambda g, c: isinstance(g, dict) and all(abs(x - y) <= 1e-3 for x, y in zip(
        g.get("position") or [], c["args"][1][0]["position"])), "the part's CFrame is the one given", **_ON_FIRST),
    "WorldRoot:IKMoveTo": _P("CFrame", _changed, "the part's CFrame moved", on=lambda a, t: a[0]["path"]),
    "WorldRoot:StepPhysics": _P("Position", _changed, "the unanchored part moved in the step",
                                on=lambda a, t: a[1]["v"][0]["path"]),
    "WorldRoot:RegisterCollisionGroup": {"by": _GROUP_ON},
    "WorldRoot:RenameCollisionGroup": {"by": _GROUP_NEW},
    "WorldRoot:CollisionGroupSetCollidable": {"by": _GROUP_PAIR},
    "WorldRoot:UnregisterCollisionGroup": {"by": _GROUP_OFF},
    "PhysicsService:RegisterCollisionGroup": {"by": _GROUP_ON},
    "PhysicsService:RenameCollisionGroup": {"by": _GROUP_NEW},
    "PhysicsService:CollisionGroupSetCollidable": {"by": _GROUP_PAIR},
    "PhysicsService:UnregisterCollisionGroup": {"by": _GROUP_OFF},
    "BasePart:SetNetworkOwnershipAuto": {"by": ("getter", "GetNetworkOwnershipAuto", _none, lambda a: True)},
    "Camera:ZoomToExtents": _P("CFrame", _changed, "the camera's CFrame moved"),
    "Seat:Sit": _P("Occupant", _is_arg0, "Occupant is the humanoid given"),
    "VehicleSeat:Sit": _P("Occupant", _is_arg0, "Occupant is the humanoid given"),
    "Humanoid:TakeDamage": _P("Health", lambda g, c: _num_like(g) and abs(g - (c["before"] - c["args"][0])) < 1e-4,
                              "Health fell by the damage"),
    "Humanoid:MoveTo": {"by": ("property", "WalkToPoint", lambda a, b: a[0])},
    "Humanoid:Move": _P("MoveDirection", _changed, "MoveDirection changed"),
    "Humanoid:AddAccessory": _G("GetAccessories", _gained, "GetAccessories holds it now, not before"),
    "Humanoid:RemoveAccessories": _G("GetAccessories", lambda g, c: g == [] and bool(c["before"]),
                                     "GetAccessories held some before and none now"),
    "Humanoid:EquipTool": _E("Equipped", lambda g, c: True, "the tool's Equipped fired",
                             on=lambda a, t: a[0]["path"]),
    "Humanoid:UnequipTools": _G("FindFirstChildWhichIsA", lambda g, c: c["before"] is not None and g is None,
                                "the character held a Tool before and none now", args=lambda a: ["Tool"],
                                on=lambda a, t: t[:-1]),
    "Tool:Activate": {"by": ("event", "Activated", lambda a: [])},
    "Tool:Deactivate": {"by": ("event", "Deactivated", lambda a: [])},
    "InputBinding:Fire": _E("StateChanged", lambda g, c: isinstance(g, list) and bool(g) and g[0] == c["args"][0],
                            "the parent InputAction's StateChanged fired with the value given (a Scriptable binding)",
                            on=lambda a, t: t[:-1]),
    "Controller:BindButton": {"by": ("getter", "GetButton", lambda a: a[:1], None), "test": lambda g, c: isinstance(g, bool),
                              "says": "GetButton answers a bool (it raised 'Button has not been bound' before, RB2321)"},
}, "RB2522"))
PROOFS.update(_ledger({     # proved by what the call made (RB2523)
    "Instance:Clone": {"by": ("return", None), "says": "the call answered a copy of the target's class",
                       "test": lambda g, c: isinstance((c["returns"] or [None])[0], dict)
                       and isinstance(c["returns"][0].get("className"), str)},
    "PathfindingService:CreatePath": _R("Path"),
    "Animator:LoadAnimation": _R("AnimationTrack"),
    "BasePart:UnionAsync": _R("UnionOperation"),
    "BasePart:SubtractAsync": _R("UnionOperation"),
    "BasePart:IntersectAsync": _R("IntersectOperation"),
    "GeometryService:UnionAsync": _R("UnionOperation"),
    "GeometryService:SubtractAsync": _R("UnionOperation"),
    "GeometryService:IntersectAsync": _R("IntersectOperation"),
    "GeometryService:FragmentAsync": _R("MeshPart"),
    "GeometryService:SweepPartAsync": _R("MeshPart"),
    "GeometryService:GenerateFragmentSites": {"by": ("return", None), "says": "the call answered a list of Vector3 sites",
                                              "test": lambda g, c: isinstance((c["returns"] or [None])[0], list) and bool(
                                                  c["returns"][0]) and all(isinstance(x, dict) and "x" in x
                                                                           for x in c["returns"][0])},
    "ChatWindowConfiguration:DeriveNewMessageProperties": _R("ChatWindowMessageProperties"),
    "KeyframeSequenceProvider:RegisterActiveKeyframeSequence": {
        "by": ("return", None), "says": "the call answered an active:// id",
        "test": lambda g, c: str((c["returns"] or [""])[0]).startswith("active://")},
    "AnimationClipProvider:RegisterActiveAnimationClip": {
        "by": ("return", None), "says": "the call answered an active:// id",
        "test": lambda g, c: str((c["returns"] or [""])[0]).startswith("active://")},
}, "RB2523"))
PROOFS.update(_ledger({     # proved by an event (RB2524)
    "Debris:AddItem": {"by": ("event", "Destroying", lambda a: []), "on": lambda a, t: a[0]["path"], "wait": 4},
    "LogService:Info": _E("MessageOut", _says_message, "MessageOut carried the text"),
    "LogService:Warn": _E("MessageOut", _says_message, "MessageOut carried the text"),
    "LogService:Output": _E("MessageOut", _says_message, "MessageOut carried the text"),
    "LogService:Log": _E("MessageOut", _says_message, "MessageOut carried the text"),
    "Logger:Info": _E("MessageOut", _says_message, "LogService.MessageOut carried the text", **_LOG),
    "Logger:Warn": _E("MessageOut", _says_message, "LogService.MessageOut carried the text", **_LOG),
    "Logger:Error": _E("MessageOut", _says_message, "LogService.MessageOut carried the text", **_LOG),
    "Logger:Output": _E("MessageOut", _says_message, "LogService.MessageOut carried the text", **_LOG),
    "Logger:Log": _E("MessageOut", _says_message, "LogService.MessageOut carried the text", **_LOG),
    "TestService:Message": _E("MessageOut", _says_message, "LogService.MessageOut carried the text", **_LOG),
    "TestService:Warn": _E("MessageOut", _says_message, "LogService.MessageOut carried the text", **_LOG),
    "TestService:Error": _E("MessageOut", _says_message, "LogService.MessageOut carried the text", **_LOG),
    "TestService:Fail": _E("MessageOut", _says_message, "LogService.MessageOut carried the text", **_LOG),
    "FunctionalTest:Warn": _E("MessageOut", _says_message, "LogService.MessageOut carried the text", **_LOG),
    "FunctionalTest:Error": _E("MessageOut", _says_message, "LogService.MessageOut carried the text", **_LOG),
    "FunctionalTest:Failed": _E("MessageOut", _says_message, "LogService.MessageOut carried the text", **_LOG),
}, "RB2524"))
PROOFS.update(_ledger({     # media and motion, silent: volume 0, no asset, nothing wired (RB2525)
    "Sound:Play": {"by": ("property", "Playing", lambda a, b: True)},
    "Sound:Pause": {"by": ("property", "Playing", lambda a, b: False)},
    "Sound:Resume": {"by": ("property", "Playing", lambda a, b: True)},
    "Sound:Stop": {"by": ("property", "Playing", lambda a, b: False)},
    "AudioPlayer:Play": {"by": ("property", "IsPlaying", lambda a, b: True)},
    "AudioPlayer:Stop": {"by": ("property", "IsPlaying", lambda a, b: False)},
    "VideoFrame:Play": {"by": ("property", "Playing", lambda a, b: True)},
    "VideoFrame:Pause": {"by": ("property", "Playing", lambda a, b: False)},
    "SoundService:SetListener": _G("GetListener", lambda g, c: (g[0] if isinstance(g, list) else g) == c["args"][0] and (
        c["before"][0] if isinstance(c["before"], list) else c["before"]) != c["args"][0],
        "GetListener answers the type given, and answered another before"),
    "TweenBase:Play": {"by": ("property", "PlaybackState",
                              lambda a, b: {"$t": "EnumItem", "enum": "PlaybackState", "v": "Playing"})},
    "TweenBase:Pause": {"by": ("property", "PlaybackState",
                               lambda a, b: {"$t": "EnumItem", "enum": "PlaybackState", "v": "Paused"})},
    "AnimationTrack:Play": {"by": ("property", "IsPlaying", lambda a, b: True)},
    "AnimationTrack:Stop": {"by": ("property", "IsPlaying", lambda a, b: False)},
    "AnimationTrack:AdjustSpeed": {"by": ("property", "Speed", lambda a, b: a[0])},
    "AnimationTrack:AdjustWeight": {"by": ("property", "WeightTarget", lambda a, b: a[0])},
}, "RB2525"))
PROOFS.update(_ledger({     # EditableMesh and EditableImage, on the handles AssetService returns (RB2526)
    "EditableMesh:AddVertex": dict({"by": ("getter", "GetPosition", lambda a, r: r[:1], lambda a: a[0])}, **_FROM_RET),
    "EditableMesh:AddNormal": dict({"by": ("getter", "GetNormal", lambda a, r: r[:1], lambda a: a[0])}, **_FROM_RET),
    "EditableMesh:AddUV": dict({"by": ("getter", "GetUV", lambda a, r: r[:1], lambda a: a[0])}, **_FROM_RET),
    "EditableMesh:AddColor": dict({"by": ("getter", "GetColor", lambda a, r: r[:1], lambda a: a[0])}, **_FROM_RET),
    "EditableMesh:AddTriangle": dict({"by": ("getter", "GetFaceVertices", lambda a, r: r[:1], None),
                                      "test": lambda g, c: g == c["args"][:3], "says": "the face's vertices are the three given"},
                                     **_FROM_RET),
    "EditableMesh:AddFace": dict({"by": ("getter", "GetFaceVertices", lambda a, r: r[:1], None),
                                  "test": lambda g, c: g == c["args"], "says": "the face's vertices are the ones given"},
                                 **_FROM_RET),
    "EditableMesh:AddBone": dict({"by": ("getter", "GetBones", lambda a, r: [], None),
                                  "test": lambda g, c: isinstance(g, list) and c["returns"][0] in g,
                                  "says": "GetBones holds the id the call answered"}, **_FROM_RET),
    "EditableMesh:RemoveBone": {"by": ("getter", "GetBones", _none, None), "before": True, "says": "the bone left GetBones",
                                "test": lambda g, c: c["args"][0] not in g and c["args"][0] in c["before"]},
    "EditableMesh:RemoveFace": {"by": ("getter", "GetFaces", _none, None), "before": True, "says": "the face left GetFaces",
                                "test": lambda g, c: c["args"][0] not in g and c["args"][0] in c["before"]},
    "EditableMesh:Triangulate": _G("GetFaces", _longer, "a quad became two triangles: GetFaces is longer"),
    "EditableMesh:RemoveUnused": _G("GetVertices", lambda g, c: isinstance(c["returns"][0], list) and bool(
        c["returns"][0]) and not set(c["returns"][0]) & set(g) and _shorter(g, c),
        "the ids it answered left GetVertices"),
    "EditableMesh:Clear": _G("GetVertices", lambda g, c: g == [] and bool(c["before"]), "GetVertices is empty now"),
    "EditableMesh:BatchAdd": _G("GetVertices", lambda g, c: isinstance(g, list) and len(g) == len(c["before"]) + len(
        c["args"][1]), "GetVertices is longer by the values given"),
    "EditableMesh:BatchRemove": _G("GetFaces", _shorter, "GetFaces is shorter"),
    "EditableImage:WritePixelsBuffer": {"by": ("getter", "ReadPixelsBuffer", lambda a: a[:2], lambda a: a[2])},
    "EditableImage:DrawRectangle": {"by": ("getter", "ReadPixelsBuffer", _px1, None), "test": _pixel_arg(2),
                                    "says": "the pixel at the position given is the color given"},
    "EditableImage:DrawCircle": {"by": ("getter", "ReadPixelsBuffer", _px1, None), "test": _pixel_arg(2),
                                 "says": "the pixel at the center given is the color given"},
    "EditableImage:DrawLine": {"by": ("getter", "ReadPixelsBuffer", _px1, None), "test": _pixel_arg(2),
                               "says": "the pixel at the line's start is the color given"},
    "EditableImage:DrawImage": {"by": ("getter", "ReadPixelsBuffer", _px1, None), "before": True, "test": _changed,
                                "says": "the pixel at the position given changed"},
}, "RB2526"))
PROOFS.update(_ledger({     # Terrain (MifBridge's scratch place: terrain is scratch) and the undo history (RB2527)
    "Terrain:FillBlock": _CELLS,
    "Terrain:FillBall": _CELLS,
    "Terrain:FillCylinder": _CELLS,
    "Terrain:FillWedge": _CELLS,
    "Terrain:FillRegion": _CELLS,
    "Terrain:PasteRegion": _CELLS,
    "Terrain:WriteVoxels": _CELLS,
    "Terrain:WriteVoxelChannels": _CELLS,
    "Terrain:ReplaceMaterial": {"by": ("getter", "ReadVoxels", lambda a: a[:2], None),
                                "says": "ReadVoxels holds the new material and not the old",
                                "test": lambda g, c: c["args"][3] in str(g) and c["args"][2] not in str(g)},
    "Terrain:SetMaterialSlot": {"by": ("getter", "GetMaterialSlot", lambda a: a[:1], None),
                                "says": "GetMaterialSlot answers the material given",
                                "test": lambda g, c: isinstance(g, list) and g[0] == c["args"][1]},
    "Terrain:ResetMaterialSlot": _G("GetMaterialSlot", lambda g, c: g != c["before"] and c["before"] is not None,
                                    "GetMaterialSlot answered the slot before and answers nothing, or another, now",
                                    args=lambda a: a[:1]),
    "ChangeHistoryService:ResetWaypoints": _G("GetCanUndo", lambda g, c: (g[0] if isinstance(g, list) else g) is False
                                              and (c["before"][0] if isinstance(c["before"], list) else c["before"]) is True,
                                              "GetCanUndo was true and is false"),
}, "RB2527"))
PROOF_EVENT_WAIT = 2.0     # seconds an event proof reads the watch for its firing (Studio defers signals to a resume)


def _proof_on(row, parts, wire):
    """Where a proof row reads: the call's own target, or the instance its "on" names from the call's arguments (the
    item Debris:AddItem was given, the pose Keyframe:AddPose took). (parts, error)."""
    on = (row or {}).get("on")
    if on is None:
        return parts, None
    try:
        got = on(wire, parts)
    except (LookupError, TypeError, ValueError, AttributeError):
        got = None
    if isinstance(got, list) and got and all(isinstance(x, str) for x in got):
        return got, None
    return None, "the proof row names no instance to read from these arguments"


def _getter_read(name, at, gargs):
    g = _rbx("call", target=at, method=name, args=gargs, argc=len(gargs), readOnly=True)
    if not g.get("ok"):
        return None, g.get("error") or "the getter failed"
    rets = g.get("returns") or [{"$t": "nil"}]
    # a getter that answers several values (ChangeHistoryService:GetCanUndo's flag and step name) is read whole
    return (rets[0], None) if len(rets) == 1 else ({"$t": "Tuple", "v": list(rets)}, None)


def _prove_before(proof, parts, wire, row=None):
    """What the proof needs before the call: the property's value, a getter's answer (a row with before true), or a
    started watch. (state, error)."""
    kind, name = proof[0], proof[1]
    at, err = _proof_on(row, parts, wire)
    if err:
        return None, err
    if kind == "property":
        g = _rbx("get_props", target=at, props=[name])
        if not g.get("ok") or name not in _obj(g.get("values")):
            return None, "could not read %s before the call: %s" % (name, g.get("error"))
        return _obj(g.get("values"))[name], None
    if kind == "getter" and (row or {}).get("before"):
        got, err = _getter_read(name, at, proof[2](wire))
        return (got, None) if err is None else (None, "could not call %s before the call: %s" % (name, err))
    if kind == "event":
        w = _rbx("watch_start", target=at, event=name, property=None, attribute=None, seconds=PROOF_EVENT_WAIT + 28,
                 max=50)
        if not w.get("ok") or not isinstance(w.get("watchId"), str):
            return None, "could not watch %s before the call: %s" % (name, w.get("error"))
        return w["watchId"], None
    return None, None


def _proof_judge(row, want, got, ctx):
    """A row's own test (a function of the decoded read and {args, before, returns}) when it has one - a list that must
    hold the instance added, a count one higher - else the read must equal the expected value."""
    test = (row or {}).get("test")
    if test is None:
        return want is not None and V.same(want, got)
    try:
        return test(V.decode(got), ctx) is True
    except (LookupError, TypeError, ValueError, AttributeError, ZeroDivisionError):
        return False


def _prove_after(proof, parts, wire, before, row=None, returns=None):
    """(verified, readBack) for a proof table row after the call answered. returns is the call's own wire returns, which
    a getter row's args may take (argsFrom "return": EditableMesh:AddVertex's id read back by GetPosition)."""
    kind, name = proof[0], proof[1]
    row = row or {}
    ctx = {"args": [V.decode(x) for x in wire], "before": V.decode(before) if kind != "event" else None,
           "returns": [V.decode(x) for x in returns or []]}
    if kind == "return":
        # what the call made is its answer (Instance:Clone's copy, PathfindingService:CreatePath's Path): the row's test
        # judges that answer; a row of this kind always has one
        got = {"$t": "Array", "v": list(returns or [])}
        return row.get("test") is not None and _proof_judge(row, None, got, ctx), {"return": ctx["returns"],
                                                                                   "test": row.get("says")}
    at, err = _proof_on(row, parts, wire)
    if err:
        return False, {kind: name, "error": err}
    if kind == "getter":
        gargs = proof[2](wire, returns or []) if row.get("argsFrom") == "return" else proof[2](wire)
        got, err = _getter_read(name, at, gargs)
        if err is not None:
            return False, {"getter": name, "error": err}
        want = None if row.get("test") else proof[3](wire)
        back = {"getter": name, "value": V.decode(got)}
        back.update({"expected": V.decode(want)} if want is not None else {"test": row.get("says")})
        if row.get("before"):
            back["before"] = ctx["before"]
        return _proof_judge(row, want, got, ctx), back
    if kind == "property":
        want = None if row.get("test") else proof[2](wire, before)
        g = _rbx("get_props", target=at, props=[name])
        got = _obj(g.get("values")).get(name, {"$t": "missing"}) if g.get("ok") else {"$t": "missing"}
        ok = not (isinstance(got, dict) and got.get("$t") == "missing") and _proof_judge(row, want, got, ctx)
        back = {"property": name, "before": V.decode(before), "value": V.decode(got)}
        back.update({"expected": V.decode(want) if want is not None else None} if not row.get("test")
                    else {"test": row.get("says")})
        if ok and not row.get("test") and V.same(want, before):
            # the property already held what the write should leave (VideoPlayer:Pause on a player that never
            # started): read back equal, yet no proof the call did anything (the record's write-same rule)
            ok = False
            back["same"] = True
        return ok, back
    want = proof[2](wire) if proof[2] is not None else None
    deadline = time.monotonic() + float(row.get("wait") or PROOF_EVENT_WAIT)
    seen = []
    while True:
        last = time.monotonic() >= deadline
        r = _rbx("watch_read", watchId=before, stop=last)
        if r.get("ok"):
            seen = [e for e in (r.get("events") or []) if isinstance(e, dict)]   # a watch's list is whole on each read
        if row.get("test"):
            hit = next((e for e in seen if _proof_judge(row, None, {"$t": "Array", "v": list(e.get("args") or [])}, ctx)),
                       None)
        else:
            hit = next((e for e in seen if len(e.get("args") or []) == len(want)
                        and all(V.same(w, g) for w, g in zip(want, e.get("args") or []))), None)
        if hit is not None or last or not r.get("ok"):
            if hit is not None and not last:
                _rbx("watch_read", watchId=before, stop=True)
            return hit is not None, {"event": name, "fired": len(seen), "expectedArgs": [V.decode(x) for x in want]
                                     if want is not None else row.get("says"),
                                     "firings": [[V.decode(x) for x in e.get("args") or []] for e in seen[:5]]}
        time.sleep(0.1)


# Members that make an object outside the place and change nothing in it: allowed wherever a read is.
NON_PLACE_CALLS = {("TweenService", "Create"), ("Terrain", "CopyRegion")}

# Functions that only compute: they return a value and change nothing, but their names start with no read prefix, so the
# name rule ran them as gated writes that counted nothing. Named one by one by the class that declares them (never a
# wider prefix), each added only after a live run called it through the write path and Studio's undo history read the
# same before and after (ChangeHistoryService:GetCanUndo), with the internal notes that measured it. Measured and NOT pure
# (RB2200): Vector3Curve:X and EulerRotationCurve:Y make their child curve the first time (a new undo step, one child).
PURE_CALLS = {
    ("Camera", "WorldToViewportPoint"): "RB2200", ("Camera", "WorldToScreenPoint"): "RB2200",
    ("Camera", "ViewportPointToRay"): "RB2200", ("Camera", "ScreenPointToRay"): "RB2200",
    ("HttpService", "JSONEncode"): "RB2200", ("HttpService", "JSONDecode"): "RB2200",
    ("HttpService", "GenerateGUID"): "RB2200", ("HttpService", "UrlEncode"): "RB2200",
    ("EncodingService", "Base64Encode"): "RB2200", ("EncodingService", "Base64Decode"): "RB2200",
    ("EncodingService", "CompressBuffer"): "RB2200", ("PhysicsService", "CollisionGroupsAreCollidable"): "RB2200",
    ("TweenService", "SmoothDamp"): "RB2200", ("Workspace", "PGSIsEnabled"): "RB2200",
    # the voxel grid's arithmetic, a translation of text, a debug name for a mesh id
    ("Terrain", "CellCenterToWorld"): "RB2443", ("Terrain", "CellCornerToWorld"): "RB2443",
    ("Terrain", "WorldToCell"): "RB2443", ("Terrain", "WorldToCellPreferEmpty"): "RB2443",
    ("Terrain", "WorldToCellPreferSolid"): "RB2443", ("Translator", "Translate"): "RB2443",
    ("EditableMesh", "IdDebugString"): "RB2443",
    ("BasePart", "AngularAccelerationToTorque"): "RB2528", ("BasePart", "TorqueToAngularAcceleration"): "RB2528",
    ("WorldRoot", "CollisionGroupsAreCollidable"): "RB2528", ("StudioService", "GizmoRaycast"): "RB2528",
    ("Translator", "FormatByKey"): "RB2528", ("EncodingService", "DecompressBuffer"): "RB2528",
}


# Classes whose members read by name but reach outside the place (found 2026-11-22): HttpService:GetAsync makes a web
# request and GetSecret reads one of the experience's secrets, yet both start with Get. Every member of these is judged
# a write - full mode, gated like any write - unless PURE_CALLS names it after a live run (JSONEncode and its kin).
NOT_READ_CLASSES = ("HttpService",)


def _read_only(name, owner=None):
    """Whether a member reads, by its name; owner is the class the catalog says declares it."""
    if owner in NOT_READ_CLASSES:
        return (owner, name) in PURE_CALLS
    return name.startswith(READ_PREFIXES)


# Members that remove what they are called on, keyed by the class the catalog says owns them (so ParticleEmitter:Clear,
# which empties live particles, is not one). A call with no getter answers verified false, and a build that re-sent
# Terrain:Clear "to see the reply" cleared 607,370 cells of fresh islands: each of these is flagged before
# it is sent, sent only with confirm true, and answered with what it removed, counted before and after.
DESTRUCTIVE = {("Terrain", "Clear"): "cells", ("Instance", "Destroy"): "instance",
               ("Instance", "Remove"): "instance", ("Instance", "ClearAllChildren"): "children"}


def _destructive_count(parts, kind):
    """What a destructive member would remove now: {cells} for Terrain, else {children, descendants}; (None, err)."""
    if kind == "cells":
        r = _rbx("call", target=parts, method="CountCells", args=[], argc=0, readOnly=True)
        if not r.get("ok"):
            return None, r
        n = V.decode((r.get("returns") or [None])[0])
        return ({"cells": n}, None) if isinstance(n, (int, float)) and not isinstance(n, bool) else (
            None, {"ok": False, "error": "Terrain:CountCells answered %r" % (n,)})
    q = _rbx("query", root=parts, selector=None, tag=None, className="Instance", attribute=None, value=None, page=0,
             pageSize=1)
    if not q.get("ok") or not isinstance(q.get("total"), int):
        return None, q if not q.get("ok") else {"ok": False, "error": "the descendant count read %r" % (q.get("total"),)}
    t = _rbx("call", target=parts, method="GetChildren", args=[], argc=0, readOnly=True)
    if not t.get("ok"):
        return None, t
    kids = V.decode((t.get("returns") or [None])[0])
    return {"children": len(kids) if isinstance(kids, list) else None, "descendants": q["total"]}, None


def _destructive(parts, owner, method, kind, confirm, send):
    """A destructive member: a dry run by default (counts, nothing sent), or sent with confirm and answered with what
    it removed. send() makes the call and returns the plugin's reply."""
    what = {"cells": "every terrain cell (Terrain:Clear empties the whole Terrain)",
            "instance": "the instance and all its descendants",
            "children": "every child and all their descendants"}[kind]
    flag = {"member": "%s:%s" % (owner, method), "removes": what}
    if isinstance(parts, dict):
        return {"ok": False, "ran": False, "refusedRule": "destructive", "destructive": flag,
                "error": "%s:%s on a {handle} is not counted; call it on a path" % (owner, method)}
    before, err = _destructive_count(parts, kind)
    if err:
        return dict(err, ok=False, ran=False, destructive=flag,
                    error="%s:%s is destructive and what it would remove could not be counted, so nothing was sent: %s"
                          % (owner, method, err.get("error")))
    if kind == "instance":
        before["instances"] = before["descendants"] + 1
    if confirm is not True:
        # ok false, as every needs-confirm refusal: a script that checked only ok logged this dry run as done and built
        # on the terrain it believed cleared (found 2026-11-22)
        return {"ok": False, "ran": False, "sent": False, "dryRun": True, "verified": False, "refusedRule": "needs-confirm",
                "destructive": flag, "target": V.join_path(parts), "wouldRemove": before,
                "error": "%s:%s is destructive and was not sent: pass confirm true to send it (it would remove %s now)"
                         % (owner, method, before),
                "note": "these are the counts it would remove now. Never send it again to look at its reply: judge it by "
                        "these counts."}
    r = send()
    if not r.get("ok"):
        return dict(r, destructive=flag, before=before)
    if kind == "instance":
        gone = _rbx("resolve", target=parts)
        after = {"instances": 0} if gone.get("refusedRule") == "not-found" else None
        removed = {"instances": before["instances"]} if after else None
        good = after is not None
    else:
        after, err = _destructive_count(parts, kind)
        if err:
            return dict(err, ok=False, destructive=flag, before=before, outcomeUnknown=True,
                        error="%s:%s was sent; the count after failed: %s. Do not send it again" % (owner, method,
                                                                                                  err.get("error")))
        removed = {k: before[k] - after[k] for k in after if isinstance(before.get(k), (int, float))
                   and isinstance(after.get(k), (int, float))}
        good = all(v == 0 for v in after.values())
    out = {"ok": good, "verified": good, "sent": True, "destructive": flag, "target": V.join_path(parts),
           "before": before, "after": after, "removed": removed, "requestId": r.get("requestId"),
           "member": "%s:%s" % (owner, method)}
    if not good:
        out["error"] = "%s:%s was sent and the target still holds %s: do not send it again; read it" % (owner, method, after)
    return out


@_reports
def rbx_call(target: Any, method: str, args: list = None, timeout: float = 30, confirm: bool = False,
             allow_ids: list = None) -> dict:
    "Roblox Studio: call any member function a plugin can reach, by catalog name, with typed and checked arguments. target is a path, or {handle} for an object a call returned (an EditableImage). Members tagged Yields run with a timeout; several return values come back as a list in order. A setter with a paired getter (SetAttribute, PivotTo, AddTag, ScaleTo...) is read back through it; another write answers verified false with its return. A member that needs a Luau function is refused needs-function: the bridge sends no code. Destructive members - Terrain:Clear, Instance:Destroy, Instance:Remove, Instance:ClearAllChildren - are flagged (destructive) and by default only counted, nothing sent: refused needs-confirm (ok false, so a caller that checks only ok never takes it as done) with dryRun and wouldRemove (Terrain:CountCells cells, or children and descendants); with confirm true they are sent and answered with before, after and removed, verified when nothing is left (a no-getter call answers verified false, and re-sending Terrain:Clear to look at its reply wiped a freshly built map). target \"game\" is the DataModel (game:GetService, game:IsLoaded); a write on it runs only in full mode, as anything outside the scratch folder (refusedRule full-mode), even on MifBridge's own scratch place. A path rooted at \"@plugin\" (MifBridge's own Plugin, \"@plugin.widget\", \"@plugin.toolbar\"; plugin 0.17.13) takes reads only: a write is refused plugin-own, except Plugin:Union, Negate, Separate and Intersect on target \"@plugin\" whose list holds only parts under Workspace.MifScratch (plugin 0.17.22), read back as every part it answered existing under the folder and each input kept, replaced or gone as Studio leaves it. A member in PROOFS (a write with no Get twin) is proved by the read its row names - a getter of another name, a property, an event that must fire (on the target or on an instance the arguments name), or the object the call returned - and answers provedBy; a property that already held the value before the call proves nothing. A string return that is raw bytes, not text (EncodingService:ComputeStringHash's digest), comes back as {bytes, base64, length, hex} (plugin 0.17.16). GetService or FindService naming a service that crashes Studio at its next save (IncrementalPatchBuilder, RB2412) is refused crashes-studio, here and by the plugin itself. A member that changes a live data object (a GlobalDataStore or a MemoryStore queue, map or counter: anything but the plugin's live-data reads) writes the live game's data, so it runs only when the open place's experience (game.GameId) is on the person's write-target list (roblox_write_targets.json) or named in allow_ids, as rbx_datastore_write does (refusedRule write-target, nothing sent)."
    if isinstance(target, dict) and set(target) == {"handle"} and isinstance(target["handle"], str):
        parts = {"handle": target["handle"]}
    else:
        parts, err = _target(target, game=True, plugin=True)
        if err:
            return err
    if not isinstance(confirm, bool):
        return {"ok": False, "ran": False, "error": "confirm is true or false"}
    args = [] if args is None else args
    if not isinstance(args, list) or not isinstance(method, str) or not method:
        return {"ok": False, "ran": False, "error": "method is a name and args a list"}
    # A version of a Model made without IsPackage false turns it into a package for good, and a listed one stops being
    # purchasable (RB2008): refused unless the request parameters say IsPackage false.
    is_model = len(args) > 1 and str(args[1]).split(".")[-1] == "Model"
    if parts == ["AssetService"] and method == "CreateAssetVersionAsync" and is_model and not (
            len(args) > 3 and isinstance(args[3], dict) and args[3].get("IsPackage") is False):
        return {"ok": False, "ran": False, "refusedRule": "would-make-package", "member": "AssetService:CreateAssetVersionAsync",
                "error": "a Model version made without {IsPackage = false} makes the asset a package for good, and a listed "
                         "one is no longer purchasable (RB2008): pass IsPackage false in the request parameters (the 4th "
                         "argument), or save a new Model with rbx_save_to_roblox"}
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= 300:
        return {"ok": False, "ran": False, "error": "timeout is 0-300 seconds"}
    cat, err = _catalog()
    if err:
        return err
    cls, err = _resolve(parts)
    if err:
        return err
    bad = cat.read_refusal(cls, method, kind="F")
    if bad:
        return bad
    owner, row = cat.find_member(cls, method)
    params = row[5]
    if any(p[1] == "Function" for p in params):
        return {"ok": False, "ran": False, "refusedRule": "needs-function", "member": "%s:%s" % (owner, method),
                "error": "%s:%s takes a Luau function; the bridge sends no code" % (owner, method)}
    # A last parameter of type Tuple takes any number of values (BindableEvent:Fire(...)), each sent as a Variant.
    varargs = bool(params) and params[-1][1] == "Tuple"
    fixed = params[:-1] if varargs else params
    if len(args) > len(params) and not varargs:
        return {"ok": False, "ran": False, "refusedRule": "type", "params": [p[0] for p in params],
                "error": "%s:%s takes %d arguments, got %d" % (owner, method, len(params), len(args))}
    for p in fixed[len(args):]:
        if p[2] is None and not str(p[1]).endswith("?") and p[1] != "Function?":
            return {"ok": False, "ran": False, "refusedRule": "type", "params": [q[0] for q in params],
                    "error": "%s:%s needs %s (%s)" % (owner, method, p[0], p[1])}
    wire = []
    for i, value in enumerate(args):
        p = fixed[i] if i < len(fixed) else (params[-1][0], "Variant")
        w, why = V.encode(p[1], value, cat.enums)
        if why:
            return {"ok": False, "ran": False, "refusedRule": "type", "param": p[0], "type": p[1], "error": why}
        wire.append(w)
    read_only = _read_only(method, owner) or (cls, method) in NON_PLACE_CALLS or (owner, method) in PURE_CALLS
    csg = not read_only and _plugin_csg(parts, method, args)
    if not read_only and _plugin_rooted(parts) and not csg:
        return _plugin_own(parts, "%s:%s" % (owner, method))
    if not read_only and parts == []:
        # game itself is never under Workspace.MifScratch: a write on it is full mode's, as anything outside the scratch
        # folder - even on MifBridge's own scratch place, whose ids (SetPlaceId) decide its scratch treatment (RB1471)
        mode = _rbx("ping", nonce=uuid.uuid4().hex, timeout=15).get("mode")
        if mode != "full":
            return {"ok": False, "ran": False, "refusedRule": "full-mode", "mode": mode, "member": "%s:%s" % (owner, method),
                    "error": "%s:%s writes the DataModel itself, which is outside Workspace.MifScratch: it runs only in full "
                             "mode, and the panel is in %s mode. Nothing was sent" % (owner, method, mode)}
    live = next((c for c in LIVE_DATA_CLASSES if cat.is_a(cls, c)), None)
    if live and method not in LIVE_DATA_READS:
        # found 2026-11-22: in full mode a generic call on a data store handle (GetDataStore, then SetAsync) wrote live
        # data with no write-target check; only the typed tools checked the list
        from mif_roblox import opencloud as OC
        g = _rbx("get_props", target=[], props=["GameId"])
        gid = (g.get("values") or {}).get("GameId") if g.get("ok") else None
        gate = OC.write_target_check(gid if gid else None, False, allow_ids)
        if gate:
            return dict(gate, gameId=gid, liveData=live, member="%s:%s" % (owner, method))
    yields = "Yields" in row[4]
    kind = DESTRUCTIVE.get((owner, method))
    if kind:
        return _destructive(parts, owner, method, kind, confirm, lambda: _rbx(
            "call", timeout=timeout + 15, target=parts, method=method, args=wire, argc=len(wire), readOnly=False,
            yields=yields, callTimeout=timeout))
    csg_before = _plugin_csg_before(args[0]) if csg else None
    proof_row = None if read_only or csg else PROOFS.get("%s:%s" % (owner, method))
    proof = (proof_row or {}).get("by")
    proof_before, proof_unset = None, None
    if proof:
        proof_before, proof_unset = _prove_before(proof, parts, wire, proof_row)
        if proof_unset:
            proof = None          # the call still runs; it answers verified false and says why its proof could not be set
    r = _rbx("call", timeout=timeout + 15, target=parts, method=method, args=wire, argc=len(wire), readOnly=read_only,
             yields=yields, callTimeout=timeout)
    if not r.get("ok"):
        if proof and proof[0] == "event":
            _rbx("watch_read", watchId=proof_before, stop=True)
        return r
    returns = [V.decode(x) for x in r.get("returns") or []]
    out = {"ok": True, "member": "%s:%s" % (owner, method), "className": r.get("className"), "readOnly": read_only,
           "yields": yields, "returnType": row[1], "requestId": r.get("requestId")}
    if row[1] and str(row[1]).startswith("Tuple") or len(returns) > 1:
        out["values"] = returns
    else:
        out["value"] = returns[0] if returns else None
    if read_only:
        return out
    if csg:
        same, back = _plugin_csg_readback(method, csg_before, returns)
        out.update(verified=same, readBack=back)
        if not same:
            out.update(ok=False, error="%s:%s did not read back as Studio leaves it: %s" % (owner, method, back))
        return out
    if proof_unset:
        return dict(out, verified=False, note="the proof table's read for %s:%s could not be set before the call (%s): the "
                                              "return is all there is to read" % (owner, method, proof_unset))
    if proof:
        same, back = _prove_after(proof, parts, wire, proof_before, proof_row, r.get("returns"))
        out.update(verified=same, readBack=back, provedBy={"kind": proof[0], "member": proof[1],
                                                          "ledger": PROOFS["%s:%s" % (owner, method)].get("ledger")})
        if not same and proof_row.get("soft"):
            # the call ran and Studio answered ok; only its proof did not show (StepPhysics on an anchored part, an
            # InsertKey at a time that already had a key): verified false, never a failure a composite would cancel on
            out.update(note="the proof table's %s %s did not show what %s:%s should leave: %s" % (
                proof[0], proof[1], owner, method, back))
        elif not same:
            out.update(ok=False, error="the proof table's %s %s did not read back what %s:%s should leave: %s" % (
                proof[0], proof[1], owner, method, back))
        return out
    pair = PAIRED.get(method) or derived_pair(cat, cls, method)
    if not pair:
        return dict(out, verified=False, note="no paired getter is known for %s: the return is all there is to read" % method)
    getter, gargs, expect = pair[:3]
    if len(pair) > 3:
        out["pairedBy"] = pair[3]
    g = _rbx("call", target=parts, method=getter, args=gargs(wire), argc=len(gargs(wire)), readOnly=True)
    if not g.get("ok"):
        return dict(out, ok=False, verified=False, error="the paired getter %s failed: %s" % (getter, g.get("error")))
    got = (g.get("returns") or [{"$t": "nil"}])[0]
    want = expect(wire)
    same = V.same(want, got)
    out.update(verified=same, readBack={"getter": getter, "value": V.decode(got)})
    if not same:
        out.update(ok=False, error="%s reads back %s, not %s" % (getter, V.decode(got), V.decode(want)))
    return out


# The plugin's live data objects and the members of them that read and change nothing (its LIVE_DATA and LIVE_DATA_READS):
# judged by this list, never by a name's prefix - MemoryStoreQueue:ReadAsync is read-shaped and hides what it reads.
LIVE_DATA_CLASSES = ("GlobalDataStore", "MemoryStoreQueue", "MemoryStoreSortedMap", "MemoryStoreHashMap",
                     "MemoryStoreDistributedCounter")
LIVE_DATA_READS = frozenset(("GetAsync", "BatchGetAsync", "GetVersionAsync", "GetVersionAtTimeAsync", "ListKeysAsync",
                             "ListVersionsAsync", "GetSortedAsync", "GetSizeAsync", "GetRangeAsync", "ListItemsAsync",
                             "IsA", "GetFullName", "GetAttribute", "GetAttributes"))


# The event watches this process started: (port, watch_id) -> {"member": "Owner.Event", "t0": monotonic at start}. The
# plugin's read answer does not name its event, so the coverage record learns it from here; a plugin that restarted
# reuses its ids, so a read is tied to its start only when the plugin's elapsed time fits this process's own clock.
_WATCHES = {}
WATCH_CLOCK_SLACK_MS = 10000


def _watched(watch_id, r, stop):
    """The "Owner.Event" a read or stop answer saw fire, or None when this process did not start that watch on an event,
    or the plugin's elapsed time says the id names some other watch."""
    w = _WATCHES.get((PORT, watch_id))
    if w is None or not isinstance(r.get("elapsedMs"), (int, float)):
        return None
    mine = (time.monotonic() - w["t0"]) * 1000.0
    if stop or r.get("active") is False:
        _WATCHES.pop((PORT, watch_id), None)
    if not mine - WATCH_CLOCK_SLACK_MS <= r["elapsedMs"] <= mine + 1000:
        return None
    return w["member"]


@_reports
def rbx_watch(action: str, target: Any = None, event: str = None, property: str = None, attribute: str = None,  # noqa: A002
              seconds: float = 10, max_events: int = 1000, watch_id: str = None) -> dict:
    "Roblox Studio: watch an event on an instance for a bounded window (at most 30 s, 1,000 firings). action start subscribes to event (any event of the class), or to a property's or attribute's changed signal, and returns a watch_id at once (with member, the event's declaring class and name); read returns what fired so far, each with its time and decoded arguments; stop reads and ends it. A read or stop of an event watch this process started names it in watched, and one that saw it fire at least once adds that event to the coverage record (how event); a watch that saw nothing adds nothing. Read only."
    if action == "start":
        if isinstance(target, dict) and set(target) == {"handle"} and isinstance(target["handle"], str):
            parts = {"handle": target["handle"]}   # an object a call returned (lane 3's L3-4); the plugin resolves it
        else:
            parts, err = _target(target, game=True, plugin=True)   # game.Loaded; @plugin.toolbar's button Click
            if err:
                return err
        picks = [x for x in (event, property, attribute) if x is not None]
        if len(picks) != 1:
            return {"ok": False, "ran": False, "error": "give exactly one of event, property or attribute"}
        if not isinstance(seconds, (int, float)) or not 0 < seconds <= 30:
            return {"ok": False, "ran": False, "error": "seconds is above 0 and at most 30"}
        if not isinstance(max_events, int) or not 1 <= max_events <= 1000:
            return {"ok": False, "ran": False, "error": "max_events is 1-1000"}
        cat, err = _catalog()
        if err:
            return err
        cls, err = _resolve(parts)
        if err:
            return err
        if event is not None:
            bad = cat.read_refusal(cls, event, kind="E")
            if bad:
                return bad
            _o, row = cat.find_member(cls, event)
            if any(p[1] == "Function" for p in row[5]):
                return {"ok": False, "ran": False, "refusedRule": "needs-function", "error": "%s passes a function" % event}
        if property is not None:
            bad = cat.read_refusal(cls, property, kind="P")
            if bad:
                return bad
        t0 = time.monotonic()
        r = _rbx("watch_start", target=parts, event=event, property=property, attribute=attribute, seconds=seconds,
                 max=max_events)
        if r.get("ok") and event is not None and isinstance(r.get("watchId"), str):
            owner = cat.find_member(cls, event)[0]
            r["member"] = "%s.%s" % (owner, event)
            _WATCHES[(PORT, r["watchId"])] = {"member": r["member"], "t0": t0}
        return r
    if action in ("read", "stop"):
        if not isinstance(watch_id, str):
            return {"ok": False, "ran": False, "error": "read and stop need watch_id"}
        r = _rbx("watch_read", watchId=watch_id, stop=action == "stop")
        if r.get("ok"):
            r["events"] = [dict(e, args=[V.decode(x) for x in e.get("args") or []]) for e in r.get("events") or []]
            member = _watched(watch_id, r, action == "stop")
            if member:
                r["watched"] = member
        return r
    return {"ok": False, "ran": False, "error": "action is start, read or stop"}


def rbx_query(root: Any = "game", selector: str = None, tag: str = None, class_name: str = None, attribute: str = None,
              value: Any = None, page: int = 0, page_size: int = 200) -> dict:
    "Roblox Studio: find instances under root by a selector (Instance:QueryDescendants), a CollectionService tag, a class (IsA) and/or an attribute (and its value), sorted by full name and paged so pages never repeat. total is every match, not just this page (returned; nextPage while more). Each match carries handle ({handle}: a target rbx_get_props and the other tools take, unique even where names repeat) and ambiguous, true when its path names more than one instance - instances a play run builds share names, so read those by handle or indexedPath (Name[n] at the levels that repeat), never by path (plugin 0.17.15; an older plugin's matches have no handle, and only paths repeated on the page are flagged). A selector Studio rejects returns Studio's own error. Read only."
    parts = [] if root in ("game", None) else V.split_path(root)
    if parts is None:
        return _target(root, "root")[1]
    if not isinstance(page, int) or page < 0 or not isinstance(page_size, int) or not 1 <= page_size <= 500:
        return {"ok": False, "ran": False, "error": "page is 0 or more and page_size 1-500"}
    if class_name is not None:
        cat, err = _catalog()
        if err:
            return err
        if class_name not in cat.classes:
            return cat.unknown_class(class_name)
    wire_value = None
    if value is not None:
        if attribute is None:
            return {"ok": False, "ran": False, "error": "value filters an attribute: give attribute too"}
        wire_value, why = V.encode("Variant", value)
        if why:
            return {"ok": False, "ran": False, "refusedRule": "type", "error": why}
    r = _rbx("query", root=parts, selector=selector, tag=tag, className=class_name, attribute=attribute, value=wire_value,
             page=page, pageSize=page_size)
    if r.get("ok"):
        r["matches"] = query_matches(r.get("matches"))
        _query_page_facts(r, page)
    return r


def same_parent(read, asked):
    """The plugin's read-back path names the parent asked for: equal name for name, where an asked Name[n] (the nth
    same-named child, resolve's rule) matches the plain Name Studio reads back. Found live: a create under Site[2]
    read back Site and was failed though it was made where asked."""
    if len(read) != len(asked):
        return False
    for got, want in zip(read, asked):
        if got != want:
            m = re.match(r"^(.+)\[(\d+)\]$", str(want))
            if not m or got != m.group(1):
                return False
    return True


QUERY_HANDLE_PLUGIN = (0, 17, 15)       # the plugin whose query gives each match a handle and its ambiguity (N60)


def query_matches(raw):
    """The plugin's query matches for the caller: path as a dot path; handle as {"handle": id}, a target rbx_get_props
    and the other tools take, when the plugin gave one; ambiguous true when the path names more than one instance
    (indexedPath, Name[n] where siblings share a name, reaches this one)."""
    out = []
    for m in raw or []:
        if not isinstance(m, dict):
            continue
        row = dict(m, path=V.join_path(m.get("path")))
        if isinstance(m.get("handle"), str):
            row["handle"] = {"handle": m["handle"]}
        if m.get("indexedPath") is not None:
            row["indexedPath"] = V.join_path(m["indexedPath"])
        out.append(row)
    return out


def _query_page_facts(r, page):
    """total (every match, not just this page), returned, nextPage, ambiguous; a plugin without per-match handles is
    named, and two matches on this page with the same path are flagged ambiguous (a same-named sibling that did not
    match goes unseen then: only the plugin can tell)."""
    ms = r["matches"]
    r["returned"] = len(ms)
    if r.get("more"):
        r["nextPage"] = page + 1
    if ms and not all("handle" in m for m in ms):
        seen = {}
        for m in ms:
            seen[m["path"]] = seen.get(m["path"], 0) + 1
        for m in ms:
            if "ambiguous" not in m:
                m["ambiguous"] = seen[m["path"]] > 1
        r["ambiguousBy"] = "page"
        r["handlesWarning"] = ("this plugin gives matches no handles and judges no ambiguity: a path two matches on "
                               "this page share is flagged, a sibling that did not match is not seen. Plugin %s or later "
                               "gives each match a handle (%s)" % (".".join(map(str, QUERY_HANDLE_PLUGIN)),
                                                                   PLUGIN_RESTART_STEP))
    r["ambiguous"] = sum(1 for m in ms if m.get("ambiguous"))
    if r["ambiguous"]:
        r["ambiguousNote"] = ("%d match(es) on this page have a path that names more than one instance: read them by "
                              "handle (or indexedPath), never by path" % r["ambiguous"])


def rbx_undo() -> dict:
    "Roblox Studio: step Studio's undo once and name the step (OnUndo's waypoint). In scratch mode only a MifBridge recording may be undone: a person's step is redone at once and refused not-bridge-undo. Nothing to undo answers nothingToUndo."
    return _rbx("undo")


def rbx_redo() -> dict:
    "Roblox Studio: step Studio's redo once and name the step. In scratch mode only a MifBridge recording may be redone; another is undone again at once and refused not-bridge-undo."
    return _rbx("redo")


_BATCH_TOOLS = {"rbx_create": "create", "rbx_set_props": "set_props", "rbx_destroy": "destroy", "rbx_reparent": "reparent",
                "rbx_clone": "clone", "rbx_script_write": "script_write"}


@_reports
def rbx_batch(ops: list) -> dict:
    "Roblox Studio: several writes as one undo recording, all or nothing. ops is a list of {tool, args} with tool one of rbx_create, rbx_set_props, rbx_destroy, rbx_reparent, rbx_clone, rbx_script_write and args as that tool takes them. Every op is checked before the first runs; if one fails Studio cancels the recording, and the targets are read again to prove the place is as it was."
    if not isinstance(ops, list) or not 1 <= len(ops) <= 100:
        return {"ok": False, "ran": False, "error": "ops is a list of 1-100 {tool, args}"}
    cat, err = _catalog()
    if err:
        return err
    plan, before, judge = [], [], []
    for i, step in enumerate(ops):
        if not isinstance(step, dict) or step.get("tool") not in _BATCH_TOOLS or not isinstance(step.get("args"), dict):
            return {"ok": False, "ran": False, "failedAt": i, "error": "op %d is {tool, args}; tools: %s" % (i, ", ".join(_BATCH_TOOLS))}
        tool, a = step["tool"], dict(step["args"])
        op = _BATCH_TOOLS[tool]
        try:
            if op == "set_props":
                parts, err = _target(a.get("target"))
                if err:
                    return dict(err, failedAt=i)
                cls, err = _resolve(parts)
                if err:
                    return dict(err, failedAt=i)
                wire, err = _encode_props(cat, cls, a.get("props"))
                if err:
                    return dict(err, failedAt=i)
                got = _rbx("get_props", target=parts, props=list(wire))
                before.append(("props", parts, _obj(got.get("values"))))
                plan.append({"op": op, "args": {"target": parts, "props": wire}})
                judge.append(("set_props", wire))
            elif op == "create":
                bad = cat.create_refusal(a.get("class_name"))
                if bad:
                    return dict(bad, failedAt=i)
                pparts, err = _target(a.get("parent"), "parent")
                if err:
                    return dict(err, failedAt=i)
                wire = {}
                if a.get("props"):
                    wire, err = _encode_props(cat, a["class_name"], a["props"])
                    if err:
                        return dict(err, failedAt=i)
                tags = a.get("tags") or []
                if not (isinstance(tags, list) and all(isinstance(t, str) and t for t in tags)):
                    return {"ok": False, "ran": False, "failedAt": i, "error": "tags is a list of strings"}
                plan.append({"op": op, "args": {"className": a["class_name"], "parent": pparts, "name": a.get("name"),
                                                "props": wire, "tags": tags}})
                if a.get("name"):
                    before.append(("absent", pparts + [a["name"]], None))
                judge.append(("create", wire))
            elif op in ("destroy", "reparent", "clone", "script_write"):
                parts, err = _target(a.get("target"))
                if err:
                    return dict(err, failedAt=i)
                args = {"target": parts}
                if op in ("reparent", "clone") and a.get("parent") is not None:
                    args["parent"], err = _target(a["parent"], "parent")
                    if err:
                        return dict(err, failedAt=i)
                if op == "clone" and a.get("name") is not None:
                    args["name"] = a["name"]
                if op == "script_write":
                    if not isinstance(a.get("source"), str):
                        return {"ok": False, "ran": False, "failedAt": i, "error": "script_write needs source"}
                    if a.get("allow_syntax_errors") is not True:
                        from mif_roblox import luau_syntax  # noqa: PLC0415
                        bad = luau_syntax.refusal(V.join_path(parts), a["source"])
                        if bad:
                            return dict(bad, failedAt=i)
                    bad = _package_refusal(parts)
                    if bad:
                        return dict(bad, failedAt=i)
                    args["source"] = a["source"]
                if op in ("destroy", "reparent"):
                    # What the cancel must put back is what was there: a target that never existed stays absent.
                    exists = _rbx("resolve", target=parts).get("ok")
                    before.append(("present" if exists else "absent", parts, None))
                plan.append({"op": op, "args": args})
                judge.append((op, a.get("source")))
        except (TypeError, KeyError) as exc:
            return {"ok": False, "ran": False, "failedAt": i, "error": "op %d: %s" % (i, exc)}
    r = _rbx("batch", ops=plan, timeout=120)
    if not r.get("ok"):
        if not r.get("cancelled"):
            return r
        # PROVE the cancel: every target read again against what it was before the batch.
        problems = []
        for kind, parts, vals in before:
            if kind == "props":
                now = _rbx("get_props", target=parts, props=list(vals))
                for k, v in vals.items():
                    if not V.same(v, _obj(now.get("values")).get(k, {"$t": "missing"})):
                        problems.append("%s.%s is %s, was %s" % (V.join_path(parts), k,
                                                                  V.decode(_obj(now.get("values")).get(k)), V.decode(v)))
            elif kind == "absent":
                if _rbx("resolve", target=parts).get("ok"):
                    problems.append("%s exists after the cancel" % V.join_path(parts))
            elif kind == "present":
                if not _rbx("resolve", target=parts).get("ok"):
                    problems.append("%s is gone after the cancel" % V.join_path(parts))
        return dict(r, ok=False, restored=not problems, restoreProblems=problems,
                    error="op %s failed (%s); Studio cancelled the batch: %s" % (
                        r.get("failedAt"), r.get("error"),
                        "the place reads as it was" if not problems else "NOT restored: " + "; ".join(problems)))
    results, bad = r.get("results") or [], []
    for i, ((kind, data), res) in enumerate(zip(judge, results)):
        res = _obj(res)
        if kind in ("set_props", "create"):
            ok, mism = _judge(data, res.get("after"))
            if not ok:
                bad.append({"op": i, "mismatch": mism})
        elif kind == "destroy" and res.get("stillThere") is not False:
            bad.append({"op": i, "error": "still there"})
        elif kind == "script_write" and _sha(res.get("source")) != _sha(data):
            bad.append({"op": i, "error": "source read back differs"})
    return {"ok": not bad, "verified": not bad, "ops": len(plan), "problems": bad, "requestId": r.get("requestId"),
            "results": [{k: v for k, v in _obj(x).items() if k in ("path", "className", "parentPath", "stillThere")}
                        for x in results]}


# ---------------------------------------------------------------- batch 4: scripts, batch 3: diagnostics
def rbx_script_search(token: str, root: Any = "game", max_hits: int = 500) -> dict:
    "Roblox Studio: find a plain-text token in every script's editor source under root (Studio's own UI excluded), each hit with its script path, line, column and the line's text. A place with no scripts fails: nothing scanned is not nothing found. Read only."
    if not isinstance(token, str) or not token:
        return {"ok": False, "ran": False, "error": "token is a non-empty string"}
    if not isinstance(max_hits, int) or not 1 <= max_hits <= 5000:
        return {"ok": False, "ran": False, "error": "max_hits is 1-5000"}
    parts = [] if root in ("game", None) else V.split_path(root)
    if parts is None:
        return _target(root, "root")[1]
    r = _rbx("script_scan", root=parts, token=token, maxHits=max_hits)
    if not r.get("ok"):
        return r
    if not r.get("scanned"):
        return {"ok": False, "scanned": 0, "error": "no scripts under %s: nothing was searched" % (V.join_path(parts) or "game")}
    hits = [dict(h, path=V.join_path(h.get("path"))) for h in r.get("hits") or []]
    return {"ok": True, "scanned": r["scanned"], "count": len(hits), "hits": hits, "truncated": r.get("truncated", False)}


def rbx_require_graph(root: Any = "game") -> dict:
    "Roblox Studio: the require graph of every script under root - each static require resolved to a path, a dynamic one (an asset id, a computed path) listed unresolved, never guessed - and the cycles among modules. Read only."
    from mif_roblox import requires as R
    parts = [] if root in ("game", None) else V.split_path(root)
    if parts is None:
        return _target(root, "root")[1]
    r = _rbx("script_scan", root=parts)
    if not r.get("ok"):
        return r
    if not r.get("scanned"):
        return {"ok": False, "scanned": 0, "error": "no scripts under %s: nothing to graph" % (V.join_path(parts) or "game")}
    g = R.graph(r.get("sources") or [])
    return dict(g, ok=True, scanned=r["scanned"], truncated=r.get("truncated", False),
                missingTargets=[e for e in g["edges"] if not e["exists"]])


def rbx_stats() -> dict:
    "Roblox Studio: frame, physics, render and memory figures from the Stats service now: frame and step times, draw calls and triangles by layer, instance and primitive counts, network rates, and memory in MB per developer tag. Read only."
    return _rbx("stats")


def rbx_script_documents(target: Any = None, action: str = None) -> dict:
    "Roblox Studio: the script editor's open tabs (each with its script, line count, read-only state and Studio's reasons), or with target, that script's open tab and its text. action open opens a tab for the script (no dialog), close closes it. Edits go through rbx_script_write, which refuses a read-only tab with its reasons and reads an open tab back."
    if action not in (None, "open", "close"):
        return {"ok": False, "ran": False, "error": "action is open or close"}
    if target is None:
        if action:
            return {"ok": False, "ran": False, "error": "open and close need target"}
        r = _rbx("script_docs")
        if r.get("ok"):
            r["documents"] = [dict(d, path=V.join_path(d.get("path"))) for d in r.get("documents") or []]
        return r
    parts, err = _target(target)
    if err:
        return err
    r = _rbx("script_docs", target=parts, action=action, timeout=60)
    if r.get("ok") and "text" in r:
        r["path"] = V.join_path(r.get("path"))
        r["sha256"] = _sha(r.get("text"))
    return r


PROBE_KINDS = ("sample", "wait", "count", "log", "errors", "players", "tests", "profile", "ugc_validate", "coverage", "fire",
               "prompt", "screenshot", "save_place", "members")
MEMBERS_PLUGIN = (0, 17, 13)      # the plugin in the play server and clients runs a members plan


def _members_runs(kw, plan):
    """kind members: check the plan against the catalog, split it when it is larger than one session takes, run each
    part, and judge every answer (play_members.judge). (an answer or a refusal to return as it is, None), or (None, the
    checked steps of a plan that fits one session)."""
    from mif_roblox import play_members as PM
    cat, err = _catalog()
    if err:
        return err, None
    wire, bad = PM.check_plan(cat, plan, _resolve, _read_only)
    if bad:
        return bad, None
    runs = PM.split(wire)
    if len(runs) == 1:
        return None, wire
    parts = []
    for run in runs:
        idx = [s["i"] for s in run]
        sub = rbx_play_probe(**dict(kw, plan=[plan[i] for i in idx]))
        for key in ("steps", "watches", "causes"):
            for row in (sub.get("members") or {}).get(key) or []:
                if isinstance(row.get("i"), int) and row["i"] < len(idx):
                    row["i"] = idx[row["i"]]
        parts.append(sub)
    members = {k: [row for p in parts for row in (p.get("members") or {}).get(k) or []]
               for k in ("steps", "watches", "causes", "proved")}
    members["missing"] = sorted(i for i in range(len(plan)) if plan[i].get("wait") is None
                                and i not in {r.get("i") for k in ("steps", "watches", "causes") for r in members[k]})
    members["trusted"] = all((p.get("members") or {}).get("trusted") for p in parts)
    return {"ok": all(p.get("ok") for p in parts), "kind": "members", "split": len(runs), "members": members,
            "proved": members["proved"],
            "runs": [{k: p.get(k) for k in ("ok", "error", "run", "refusedRule")} for p in parts]}, None


@_reports
def rbx_play_probe(kind: str, target: Any = None, property: str = None, axis: str = None, seconds: float = 2,  # noqa: A002
                   interval: float = 0.1, op: str = None, value: Any = None, class_name: str = None,
                   run_mode: bool = False, timeout: float = 60, settle: float = 0, players: int = None,
                   clients: bool = False, target2: Any = None, minimize_windows: bool = True,
                   withhold_args: bool = False, fault_store: list = None, seed: dict = None, census: dict = None,
                   self_test: Any = None, collect: str = None, stop_leftover: bool = False, lock_wait: float = 600,
                   capture_width: int = 1280, confirm: bool = False, allow_ids: list = None, place_id: int = None,
                   frame_times: Any = None, client_windows: str = None, camera: dict = None,
                   plan: list = None, accept_runtime_state: bool = False) -> dict:
    "Roblox Studio: run a play session (run mode, or a multiplayer test with players 1-8) with MifBridge's fixed probe script and return what it measured: sample a target's property every interval for seconds (axis X, Y or Z for a Vector3), wait until it is lt, gt or eq a value, count a class under the target, log the Output, collect script errors (message, stack, script), line coverage of the target script, the players who joined (value: how many to wait for), or a script profile of the play server (value: frequency in Hz, default 1000; the MifBridge plugin 0.12.0 Studio loads into the play server runs ScriptProfilerService there and returns OnNewData's JSON). Plugin 0.15.0 adds: coverage (the plugin in the play server enables ScriptContext coverage on target, reads the hits, restarts the script - or target2, the Script that requires a ModuleScript - so it compiles with coverage on, waits seconds and reads them again), fire (InputBinding:Fire(value) on target, a Scriptable binding, in the running game; target2 is what to sample with property and axis, default @character, the first player's HumanoidRootPart; the action's GetState before, at and after, its Pressed/Released/StateChanged, Fire(false) after a true) and prompt (ProximityPrompt target held for its HoldDuration with InputHoldBegin/InputHoldEnd; PromptButtonHoldBegan, PromptButtonHoldEnded, Triggered and TriggerEnded counted with the player passed). settle waits before measuring. clients true adds a fixed client probe: each client reports its player name and its own Output (writes into StarterPlayer and ReplicatedStorage, so full mode or MifBridge's scratch place). Every kind also returns the Output and ScriptContext errors seen. The probe is written into Workspace.MifScratch and removed after; play runs on a copy of the place. Before a run, a client probe or remote an earlier probe left in the place (StarterPlayerScripts.MifClientProbe, ReplicatedStorage.MifProbeEvent or MifMembersRemote) refuses it probe-left, naming them; after it, anything the probe staged that is still in the edit place (Studio had another recording open, so the removal did not run) fails the answer probe-left with probeLeft, since a save keeps it and a publish ships it. minimize_windows (default true) minimizes each play-test window the probe opens, without taking focus, so a test stays off the person's screen; testWindows says which. To test a player leaving, have that client call StudioTestService:LeaveTest(): the server sees PlayerRemoving, the client's window ends by itself and the probe still reports. Player:Kick leaves a Disconnected dialog in the client's window, which no plugin can clear, and can end the play server before the probe reports. withhold_args true (plugin 0.16.4) starts the session with no instructions on purpose, to prove the probe answers play-no-args at once instead of hanging. When the session ends before the probe reports, or its result is lost (SendTestResult timed out), the play test's own Studio logs are read: the result itself when the server printed it (recovered true, plugin 0.16.6), else the Output, errors and client logs up to the end (endedEarly), with session saying when and how each process ended (a process still running is 'still running', never ended; cutAt is when another Studio's test ending cut the server's reply channel, apart from its own client leaving). A run whose reply was cut by another Studio's test while its own test processes still run is left to finish, not ended (measured: the server plays its window to the end, prints its result and exits by itself, RB2928): the probe waits for this run's result - posted by its play server to this daemon under resultBlob, or printed in its server's log - and for the processes to exit, up to timeout plus 30 s; the answer is recovered with resultFrom (daemon or log) and cut (waitedSeconds, exitedBySelf, boundSeconds); processes still there past that bound are ended by pid, and cut.stuckPastBound and cutWarning say so. fault_store (a list of events, [] for none) puts MifBridge's DataStore stand-in at Workspace.MifScratch.DataStoreStandIn for the run: an in-memory store that, on cue, fails, stalls (seconds, then fail optional) or holds or releases a foreign lock - each event {do: fail|stall|lock|unlock, op: UpdateAsync|SetAsync|GetAsync|RemoveAsync|IncrementAsync|*, store, key, nth (the first matching call it acts on), count (how many, 0 all after), after/before (a window in seconds from the start), seconds, field (the lock field, default Lock), session}; seed {store: {key: value}} fills it first. A place reaches the stand-in only through the seam rbx_script_scaffold's save template uses (dataStores(): the stand-in in Studio when it is there, else DataStoreService); a run where the place never asked for a store fails. census {module, function (default census), per: player (function(player) for each player) or all (function() once, {owner: items})} is read after the players join and again near the end: a list is item uids, a dict of numbers is counts; the answer lists each owner's items, what changed, and fails on a uid held twice. self_test names templates made by rbx_script_scaffold (kind save, action, prompt_once, economy, scenery or warps): their SelfTest scripts run during the probe (seconds 20 or more) and every case must pass. All of these are written under Workspace.MifScratch and removed after. seconds runs up to 1800 and timeout up to seconds + settle + 600: past the daemon's 600 s call the probe waits on the late reply and on the play server's own log, and collect ('last', or a run's nonce or requestId) reads an earlier run's result again when a call ended first (run says which run). One play test on the machine at a time (RB1015, RB1040): every probe takes the machine's play-test lock for its whole test, waiting up to lock_wait seconds in turn - waiters queue by arrival beside the lock and a gone or out-of-time waiter is skipped, so later waiters never starve an earlier one (RB2135) - and run.lock says who held it, how long this run waited and its queuePosition on arrival (its runClass: short for seconds 120 or less, else soak; a short run goes ahead of a soak queued before it, and a soak is passed by at most 3 short runs that came after it, so it cannot starve: wentAheadOf and passedBy name them) (every argument check runs before the lock is waited for, so a mistyped call is refused at once; a holder whose start time cannot be read and that is not certainly gone counts as holding it, named in lockUnverifiable, until its lock is older than the longest probe), and every run carries a nonce that its server and clients echo: a result or client report with another run's nonce is dropped and counted (foreignClientReports), never taken as this run's. A test server or client of this Studio still running from an earlier probe (a cancelled call) is refused leftover-test naming its pids instead of waited on; stop_leftover true ends them by pid first. Clients report their Output when the server's window ends (plugin 0.17.4), so a line printed late in a long window is kept; since plugin 0.17.8 the window starts only once every expected client probe said it runs (clientWait: a solo client DataModel is built after the server's, measured up to 37 s on a heavy place), and a run with no client report says why (clientsWarning, clientsMissing); with clients, a connected plugin before 0.17.8 (its own version, never guessed) adds pluginWarning naming it and the restart, since a running Studio keeps the plugin it loaded (RB1970). Each list is capped on its own - the server's Output keeps its first 1000 and last 1000 lines, each client's its first 500 and last 500 - and cuts gives every list's kept and total, so server lines never crowd client lines out; the server's logs come last in the answer. kind screenshot (plugin 0.17.5) captures a play-test client's own view: value seconds after the client starts (default 5), the MifBridge plugin in that client takes it with CaptureService and sends it to this daemon, and the answer carries screenshot (file, size, and the same blank-frame check as rbx_screenshot: too few colors, too little luminance spread, or 60% or more of the frame Studio's empty viewport color, an empty Studio window), capture_width wide at most; the client is a multiplayer test window at its own size, never the device simulator's device (RB1800), and deviceSimWarning says so when the simulator has one set; the client window is put behind the person's windows without focus instead of minimized, since a minimized window captures blank. kind save_place (plugin 0.17.9) saves the place to Roblox as a new version with AssetService:SavePlaceAsync from the play server (an Edit DataModel is refused, live 0.748), the probe itself left out (not Archivable). What it saves is the play server's DataModel as it runs, not the place as edited: a part a server Script made at run time was in the saved version (RB2901), and Run mode runs every server Script and physics first, so it is refused saves-runtime-state unless accept_runtime_state is true, and the answer then carries runtimeState saying so; the edit state goes up by saving the place file (rbx_place_save, or the person's File > Save) and rbx_place_publish of that file. It needs confirm true, full mode, and the open place and its experience on the person's write-target list (roblox_write_targets.json) or named in allow_ids (refusedRule write-target); it runs in Run mode with no players, and is read back against the place asset's revision (Open Cloud, the person's key): higher is ok, exactly one higher is verified, and revisionNote says when another save landed in between. A Studio that has the place open from Roblox holds it in a cloud edit session and the save fails PlaceOngoingTeamCreateSession (RB1967): open a local copy of the place file instead (rbx_launch place_file, no edit session) and name place_id - a local file has PlaceId 0, so the plugin first gives the open place that id and its experience (DataModel:SetPlaceId and SetUniverseId; the universe read from Roblox's public place-to-universe route) once both are on the write-target list and the machine play lock is held, and puts both back to 0 when the save fails (a PlaceId read that fails is refused place-ids-unread, never taken for a local copy), and the play server saves to them (live 0.748, RB2001). The hold outlives a Studio: a save within a minute of stopping a Studio that held the place (from Roblox, or a local copy given its id) fails the same way, and one about five minutes later saves (RB2002) - the answer says to wait. Team Create on does not block it. A place already open under another id is refused place-mismatch, never retargeted. Needs plugin 0.17.10 or later, whose play server checks its own PlaceId and GameId against the ones the gate checked and saves nothing on a mismatch. frame_times (plugin 0.17.12; true for the whole window, or [from, to] seconds after the window starts) times each client's frames: the play server says when its window starts and every client keeps RenderStepped's step from then over [from, to); frameTimes gives each client's frames, median, p95, max and mean ms and physics fps, Stats' RenderCPUFrameTime and RenderGPUFrameTime per frame (renderCpuMs, renderGpuMs: Roblox's own readings, which include the waits of a capped window), and windowState, the state its test window ran in as read through the run (behind, minimized, foreground, each held for 85% of the readings, or mixed). Clients and one player are implied. client_windows behind (the default with frame_times and kind screenshot) keeps each client window under the person's windows without focus; minimized (the default otherwise) minimizes it. Measured: a test window that is not the foreground window renders at a 66.7 ms cap whether minimized, kept behind or moved off-screen (RB2105, RB2130); only a window in the foreground ran faster (24.8 ms median), and the probe never takes the foreground from the person, so frame times compare runs in the same window state, and frameWarning says when the state was minimized, mixed or foreground. camera {position, look_at, fov?} (plugin 0.17.12, kind screenshot or frame_times) is held by the probe in each client - Scriptable, set again every frame after Roblox's camera scripts - from the client's start until it reports; camera says what the client read back (position, look vector, fov, viewport, where look_at fell on screen) and whether the window capture fell inside the hold, and a capture taken before it fails, saying to raise value. kind screenshot's window capture is cropped to the 3D viewport (the window holds Studio's ribbon and panes too, RB2108); capture says the window size and the viewport box. The window is proved to be this run's own client before it is kept (windowCheck): its owner read before and after PrintWindow is the client pid, the client's command line names this daemon's edit Studio (-editpid) and the test session (-playTestSessionGuid) of a server this run saw from that Studio, which is its -parentPid, and on a port with a Studio launched for it the connected Studio is that one; anything else is refused capture-not-this-run with no screenshot, each reason in windowCheck.why, and coveredBy lists the windows above it. A client report larger than one remote call carries (RB2073) is sent in pieces after a notice of its size (plugin 0.17.12) and joined by the server; one that arrives short is named in clientsWarning and partialReports, never dropped silently. kind members (plugin 0.17.13) runs plan, a list of typed steps checked against the catalog before the session starts and run as data by the MifBridge plugin in the play server and, for steps with on client, in the first player's client: {read: [properties], target}, {call: a read function, args, target}, {watch: event, target} (from that step to the end), {cause: key|fire|hold|move_to|change_state|damage|equip|touch|remote|leave, ...} and {wait: seconds}. A target is a path that exists in Edit or a play root - @player, @character, @playergui, @camera, @mouse (client) - with child names after it and class naming what the step expects there. The answer's members lists every step's result or exact error in plan order, each watch's firings, and each cause beside the watched events that fired in its window, each with firedBefore (its firings in the 2.5 s before the cause: a Humanoid regenerating health fires about once a second); a cause's expect names the Class.Event watches it should raise, and raisedNothing is whether none of them fired (without expect, whether nothing fired); a leaving client sends its answer before LeaveTest (plugin 0.17.14); proved lists what the run proved as play-read, play-call and play-event, which the coverage record takes only from this run's own reply (its nonce, not a log). A plan over 400 reads, calls, watches and causes runs in several sessions, its watches and causes kept together in the first. Player members are counted only this way (coverage plan decision 9). Plugin 0.17.19 adds causes mouse_button {button, at|position, seconds, drag}, mouse_move, mouse_delta, pointer_action {at|position, action: {Wheel|Pinch...}}, text_input, focus, make {class, parent, name, props, near}, unequip, activate, deactivate, chat, system_message, animate (its track is root @track), remote from the server (FireClient) or unreliable, and call (one row of play_members.PLAY_CALLS, with check {prop}|{call, args} read before and after); 0.17.20 adds set {target, props}. at is a GuiObject (its centre plus the GUI inset) or a BasePart (its viewport point); a point under the topbar or the chat window is refused hits CoreGUI. Root @input is the last InputObject a client watch saw; runtime true targets a path made in play; a Class argument may name a play root; early true arms a server watch before any player joins. A cause's own members are proved as play-call-verified when their event fired in its window and not in the 2.5 s before, or its check changed. Pass client_windows behind for pointer plans: a minimized client lays out a small view (RB2562)."
    if collect is not None:
        return _collect_run(collect)
    if fault_store is not None or seed is not None or census is not None or self_test is not None:
        from mif_roblox import domain_logic as DL
        kw = dict(kind=kind, target=target, property=property, axis=axis, seconds=seconds, interval=interval, op=op,
                  value=value, class_name=class_name, run_mode=run_mode, timeout=timeout, settle=settle, players=players,
                  clients=clients, target2=target2, minimize_windows=minimize_windows, withhold_args=withhold_args,
                  stop_leftover=stop_leftover, lock_wait=lock_wait, capture_width=capture_width, frame_times=frame_times,
                  client_windows=client_windows, camera=camera)
        return DL.play_with_logic(kw, fault_store=fault_store, seed=seed, census=census, self_test=self_test)
    if kind not in PROBE_KINDS:
        return {"ok": False, "ran": False, "error": "kind is one of %s" % ", ".join(PROBE_KINDS)}
    members_wire = None
    if kind == "members":
        from mif_roblox import play_members as PM
        kw = {k: v for k, v in locals().items() if k in inspect.signature(rbx_play_probe).parameters}
        done, members_wire = _members_runs(kw, plan)
        if done is not None:
            return done
        need = max(MEMBERS_PLUGIN, PM.needs_plugin(members_wire))
        have, older = _plugin_older(need)
        if older is not False:
            return {"ok": False, "ran": False, "refusedRule": "plugin-version", "plugin": have,
                    "error": "this members plan needs MifBridge plugin %s or later in this Studio (it runs %s): the plugin "
                             "in the play server and clients runs the plan%s" % (
                                 ".".join(map(str, need)), have, "" if need == MEMBERS_PLUGIN else
                                 " (the causes after the first ten, runtime targets, @input, @track, early watches, a "
                                 "server remote and the unreliable remote came in 0.17.19; set in 0.17.20)"),
                    "steps": [PLUGIN_RESTART_STEP]}
        clients = bool(clients) or PM.needs_clients(members_wire)
        players = players or 1
        # the window lasts at least what the plan's waits and causes take; the call waits for it (secondsUsed says so)
        win = PM.window_needed(members_wire)
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and seconds < win:
            seconds = win
        if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and isinstance(seconds, (int, float)) \
                and isinstance(settle, (int, float)) and timeout < seconds + settle + 60:
            timeout = seconds + settle + 60
    elif plan is not None:
        return {"ok": False, "ran": False, "error": "plan is for kind members"}
    save = None
    if kind == "save_place":
        held = _runtime_state_refusal(accept_runtime_state)
        if held:
            return held
        save, bad = _save_place_gate(confirm, allow_ids, place_id)
        if bad:
            return bad
        run_mode, players, clients = True, None, False
    if not isinstance(minimize_windows, bool):
        return {"ok": False, "ran": False, "error": "minimize_windows is true or false"}
    if not isinstance(withhold_args, bool):
        return {"ok": False, "ran": False, "error": "withhold_args is true or false"}
    if not isinstance(stop_leftover, bool):
        return {"ok": False, "ran": False, "error": "stop_leftover is true or false"}
    if not isinstance(lock_wait, (int, float)) or isinstance(lock_wait, bool) or not 0 <= lock_wait <= 3600:
        return {"ok": False, "ran": False, "error": "lock_wait is 0-3600 seconds"}
    if not isinstance(seconds, (int, float)) or not 0 < seconds <= PROBE_MAX_SECONDS \
            or not isinstance(interval, (int, float)) or not 0.02 <= interval <= 10:
        return {"ok": False, "ran": False, "error": "seconds is above 0 and at most %d; interval 0.02-10" % PROBE_MAX_SECONDS}
    if not isinstance(settle, (int, float)) or not 0 <= settle <= 30:
        return {"ok": False, "ran": False, "error": "settle is 0-30 seconds"}
    if not isinstance(timeout, (int, float)) or not seconds + settle < timeout <= seconds + settle + 600:
        return {"ok": False, "ran": False,
                "error": "timeout is more than seconds plus settle (%g) and at most 600 more" % (seconds + settle)}
    if players is not None and (not isinstance(players, int) or isinstance(players, bool) or not 1 <= players <= 8):
        return {"ok": False, "ran": False, "error": "players is 1-8"}
    frames, windows, cam, bad = _client_view_args(kind, frame_times, client_windows, camera, seconds, run_mode)
    if bad:
        return bad
    if frames is not None or cam is not None:
        clients, players = True, players or 1
    # every check that needs no play session runs here, before the machine's play lock is waited for (found 2026-11-22:
    # behind a default 600 s lock_wait a mistyped axis queued for minutes and was then refused)
    parts = None
    if kind not in ("log", "errors", "players", "tests", "profile", "screenshot", "save_place", "members"):
        parts, err = _target(target)
        if err:
            return err
    if kind in ("sample", "wait"):
        if not isinstance(property, str) or not property:
            return {"ok": False, "ran": False, "error": "%s needs property" % kind}
        cat, err = _catalog()
        if err:
            return err
        cls, err = _resolve(parts)
        if err:
            return err
        # "@Name" samples an attribute (GetAttribute) rather than a property; any attribute name may be asked for.
        bad = None if property.startswith("@") else cat.read_refusal(cls, property, kind="P")
        if bad:
            return bad
    if axis is not None and axis not in ("X", "Y", "Z"):
        return {"ok": False, "ran": False, "error": "axis is X, Y or Z"}
    if kind == "wait" and (op not in ("lt", "gt", "eq") or not isinstance(value, (int, float, str, bool))):
        return {"ok": False, "ran": False, "error": "wait needs op lt, gt or eq and a plain value"}
    parts2 = None
    if target2 is not None:
        parts2 = ["@character"] if target2 == "@character" else _target(target2, "target2")[0]
        if parts2 is None:
            return {"ok": False, "ran": False, "error": "target2 is a path, or \"@character\" for fire"}
    if kind in ("fire", "prompt"):
        want = "InputBinding" if kind == "fire" else "ProximityPrompt"
        cls, err = _resolve(parts)
        if err:
            return err
        if cls != want:
            return {"ok": False, "ran": False, "refusedRule": "wrong-class",
                    "error": "%s probes a %s; %s is a %s" % (kind, want, V.join_path(parts), cls)}
        if kind == "fire" and not isinstance(value, (bool, int, float)):
            return {"ok": False, "ran": False, "error": "fire needs value: true/false, or a number for an analog binding"}
    if kind == "screenshot":
        at = 5 if value is None else value
        if not isinstance(at, (int, float)) or isinstance(at, bool) or not 1 <= at <= 120:
            return {"ok": False, "ran": False, "error": "screenshot: value is when to capture, 1-120 seconds after the client starts"}
        if not isinstance(capture_width, int) or isinstance(capture_width, bool) or not 64 <= capture_width <= 4096:
            return {"ok": False, "ran": False, "error": "capture_width is 64-4096 pixels"}
        if seconds < at + 10:
            return {"ok": False, "ran": False, "error": "screenshot: seconds (%g) must leave the capture time and 10 s more (at "
                                                        "least %g)" % (seconds, at + 10)}
        value, clients = at, True
        players = players or 1
    if kind == "ugc_validate":
        cat, err = _catalog()
        if err:
            return err
        enum = "BodyPart" if class_name == "BodyPart" else "AccessoryType"
        if not isinstance(value, str) or value not in (cat.enums.get(enum) or []):
            return {"ok": False, "ran": False, "refusedRule": "type", "items": cat.enums.get(enum),
                    "error": "ugc_validate needs value: an Enum.%s item name (class_name BodyPart for a body part)" % enum}
    from mif_roblox import launch as L
    own = _own_studio_pid()
    def leftover_check():
        """None, or the refusal; stops this Studio's leftover test processes by pid when stop_leftover. A test whose
        probe still holds the machine lock is that probe's, not a leftover: then the lock is waited on instead."""
        left = L.leftover_tests(own)
        if not left:
            return None, None
        if not stop_leftover:
            return {"ok": False, "ran": False, "refusedRule": "leftover-test", "leftover": [p for p, _s in left],
                    "startedSecondsAgo": [round(time.time() - s, 1) if s else None for _p, s in left],
                    "error": "a play test of this Studio (pid %s) is still running from an earlier probe: pids %s. A new test "
                             "would wait on it. Nothing was started; stop_leftover true ends them by pid first"
                             % (own, ", ".join(str(p) for p, _s in left))}, None
        return None, {"found": [p for p, _s in left],
                      "stoppedByPid": [p for p, _s in sorted(left, key=lambda ps: -(ps[1] or 0)) if L.TERMINATE(p)]}
    holder = L.read_play_lock()
    leftover = None
    if not (L.holder_alive(holder) and holder.get("studioPid") == own):
        bad, leftover = leftover_check()
        if bad:
            return bad
    plugin_seen = _plugin_older(CLIENT_WAIT_PLUGIN) if clients or kind == "screenshot" else (None, None)
    nonce = uuid.uuid4().hex[:16]
    lock = L.take_play_lock(PORT, own, nonce, lock_wait, run_class=L.run_class(seconds))
    lock_taken = time.time()
    if not lock.get("ok"):
        return {"ok": False, "ran": False, "refusedRule": "play-lock-busy", "lock": lock,
                "error": "another play test held the machine's play-test lock for all of lock_wait (%gs): %s. Nothing was "
                         "started (one test at a time: RB1015, RB1040)" % (lock_wait, lock.get("heldBy"))}
    if leftover is None:
        # with the lock, any test of this Studio still running is a leftover (its probe let the lock go, or died)
        bad, leftover = leftover_check()
        if bad:
            L.release_play_lock(nonce)
            return bad
    try:
        stale = _probe_leftovers(server=False)
        if stale:
            # an earlier probe's client script or remote is still in the edit place: a save keeps it, and a publish
            # would ship a client script and a RemoteEvent nobody guards (found 2026-11-22)
            return {"ok": False, "ran": False, "refusedRule": "probe-left", "probeLeft": stale,
                    "error": "an earlier play probe left %s in the place: nothing was started. Remove them first "
                             "(rbx_destroy, full mode: they are outside Workspace.MifScratch)" % ", ".join(stale)}
        if save is not None:
            bad = _save_place_set_ids(save)
            if bad:
                return bad
        from mif_roblox import launch as L
        # A new place has no scratch folder and a plugin before 0.9.3 refuses the probe then; make it first (creating that
        # one folder is allowed in scratch mode too). Plugin 0.9.3 makes it itself.
        made_scratch = None
        if not _rbx("resolve", target=["Workspace", "MifScratch"]).get("ok"):
            mk = _rbx("create", parent=["Workspace"], className="Folder", name="MifScratch", props={}, tags=[])
            if not mk.get("ok"):
                return dict(mk, error="no Workspace.MifScratch and it could not be made: %s" % mk.get("error"))
            made_scratch = "Workspace.MifScratch"
        before = {p for p, _s in L.running()}
        t_start = time.time()
        keeper = (_hook_call(_WindowKeeper, L, before, t_start, own=own, behind_clients=windows == "behind",
                             capture_at=value if kind == "screenshot" else None) if minimize_windows else None)
        cut_waited = False
        # play_runs.json keeps what a queue replay needs (spec N86): run mode, players, the wait and the hold
        run_row = {"nonce": nonce, "port": PORT, "own": own, "tStart": t_start, "kind": kind, "seconds": seconds,
                   "settle": settle, "timeout": timeout, "requestId": None, "runMode": bool(run_mode),
                   "players": players, "clients": bool(clients), "runClass": lock.get("runClass"),
                   "waitedSeconds": lock.get("waitedSeconds"), "queuePosition": lock.get("queuePosition"),
                   "lockTakenAt": round(lock_taken, 2)}
        L.note_run(run_row)
        r = None
        try:
            call = min(timeout + 30, PROBE_CALL_CAP)
            r = _rbx("play_probe", timeout=call, kind=kind, path=parts, property=property, axis=axis, seconds=seconds,
                     interval=interval, cmp=op, value=value if kind != "players" else (value or players or 1),
                     className=class_name, runMode=bool(run_mode), callTimeout=timeout, settle=settle, players=players,
                     clients=bool(clients), path2=parts2, nonce=nonce, **({"withholdArgs": True} if withhold_args else {}),
                     **({"maxWidth": capture_width} if kind == "screenshot" else {}),
                     **({"frameFrom": frames[0], "frameTo": frames[1]} if frames is not None else {}),
                     **({"camera": cam} if cam is not None else {}),
                     **({"plan": members_wire, "membersRemote": PM.needs_remote(members_wire),
                         **({"membersUnreliable": True} if PM.needs_unreliable(members_wire) else {})}
                        if kind == "members" else {}),
                     **({"expectPlaceId": save["placeId"], "expectGameId": save["gameId"]} if save is not None else {}),
                     resultBlob=_result_blob_id(nonce))
            if r.get("requestId"):
                run_row["requestId"] = r.get("requestId")
                L.note_run(run_row)
            if r.get("outcomeUnknown") and r.get("requestId") and timeout + 30 > call:
                # the run outlives the daemon's call (RB1194): wait for the late reply, or the result in the server's log
                r = _await_late(L, r, own, t_start, nonce, t_start + timeout + 30)
            r = _match_nonce(r, nonce)
            if _cut_reply(r) and L.leftover_tests(own):
                # another Studio's test ended and took this run's reply channel with it; the run itself goes on (RB2928):
                # wait for its result and its exit instead of ending a healthy server (RB2927), up to the run's own bound
                r = _await_cut(L, r, own, t_start, nonce, t_start + timeout + CUT_GRACE)
                cut_waited = True
        finally:
            if keeper:
                keeper.stop()
            # after a cut wait, a process still there is past its bound: end it now (no second wait)
            procs = _hook_call(_end_test_processes, L, before, t_start,
                               wait=30.0 if r is not None and not cut_waited else 0.0, own=own)
            released = L.release_play_lock(nonce)
            t_end = time.time()
            L.note_run(dict(run_row, lockReleasedAt=round(t_end, 2), holdSeconds=round(t_end - lock_taken, 1),
                            ok=bool(r.get("ok")) if isinstance(r, dict) else None))
        if isinstance(r.get("cut"), dict) and procs.get("endedByPid"):
            r["cut"].update(stuckPastBound=True, endedByPid=procs["endedByPid"])
            r["cutWarning"] = ("the run's reply was cut by another Studio's test and its test processes %s were still "
                               "running %g s after the run's own deadline: they were ended by pid" % (
                                   procs["endedByPid"], CUT_GRACE))
        if r.get("ok") and kind == "errors":
            r["mapped"] = [dict(e, **_where(e)) for e in r.get("errors") or []]
        r["testProcesses"] = procs
        r["testWindows"] = keeper.report() if keeper else {"minimized": False}
        if not r.get("ok") and r.get("refusedRule") != "play-no-args" and (
                r.get("refusedRule") in ("play-failed", "timeout", "foreign-result") or r.get("outcomeUnknown")):
            r = _hook_call(_from_test_logs, L, r, own, t_start, nonce=nonce)
        r["run"] = {"nonce": nonce, "lock": {k: lock.get(k) for k in ("waitedSeconds", "heldBy", "broken",
                                                                      "queuePosition", "queuedAhead", "skipped",
                                                                      "runClass", "wentAheadOf", "passedBy")},
                    "lockReleased": released}
        r = _list_cuts(r, bool(clients))
        if clients and plugin_seen[1]:
            # a Studio keeps the plugin it loaded (RB1970): one before 0.17.8 starts the window without its clients and
            # keeps only what a client printed early, so its client lines can go missing (RB1850)
            r["pluginWarning"] = ("this Studio runs MifBridge plugin %s; client reports need %s or later to wait for a "
                                  "late client and keep what it printed late: %s" % (
                                      plugin_seen[0], ".".join(map(str, CLIENT_WAIT_PLUGIN)), PLUGIN_RESTART_STEP))
            if r.get("clientsWarning"):
                r["clientsWarning"] += "; " + r["pluginWarning"]
        if frames is not None:
            r = _frame_report(r, frames, keeper, windows)
        if kind == "screenshot":
            cap = next((c.get("capture") for c in r.get("clients") or [] if isinstance(c, dict) and c.get("capture")), None)
            win = keeper.shot if keeper else None
            owner = (_capture_owner_check(win, keeper, own, L, PORT)
                     if (not isinstance(cap, dict) or not cap.get("ok")) and win and win.get("image") else None)
            if owner is not None and not owner["ok"]:
                # never a picture of another window taken for this run's (found 2026-11-26)
                r.update(ok=False, refusedRule="capture-not-this-run", captureRoute="window", pluginCapture=cap,
                         windowCheck=owner, screenshot=None,
                         capture={"atSeconds": win.get("atSeconds"), "pid": win.get("pid")},
                         error="the captured window is not shown to be this run's play client: %s" % "; ".join(owner["why"]))
            elif owner is not None:
                # a play client's own CaptureService frame cannot become pixels there (RB1503): its window's are read,
                # cropped to the 3D viewport (the window holds Studio's ribbon and panes too, RB2108)
                r["windowCheck"] = owner
                w, h, rgba = win["image"]
                size = next((c.get("viewport") for c in r.get("clients") or [] if isinstance(c, dict)
                             and isinstance(c.get("viewport"), list)), None)
                crop, why = _viewport_box(win.get("rects"), size, w, h)
                if crop:
                    w, h, rgba = _crop_rgba(w, h, rgba, crop)
                shot = _save_rgba(w, h, rgba, "play_client")
                r.update(screenshot=shot, captureRoute="window", pluginCapture=cap,
                         capture={"atSeconds": win["atSeconds"], "captureSeconds": win["seconds"], "pid": win["pid"],
                                  "window": [win["image"][0], win["image"][1]], "viewport": crop,
                                  "clientViewport": size, "cropped": bool(crop)})
                r.update(ok=bool(shot.get("ok")), error=None if shot.get("ok") else "play client window: %s" % shot.get("error"))
                if not crop:
                    r["captureWarning"] = "the image is the whole Studio test window, not only its 3D viewport: %s" % why
            elif not isinstance(cap, dict):
                r.update(ok=False, error=r.get("error") or "no client reported a capture, and no client window was read")
            elif not cap.get("ok"):
                r.update(ok=False, capture=cap, error="the play client's capture failed: %s%s" % (
                    cap.get("error"), "" if win else "; no client window was read (minimize_windows false, or none seen)"))
            else:
                shot = _save_capture(cap, None, "play_client")
                r["screenshot"] = shot
                r["capture"] = {k: cap.get(k) for k in ("atSeconds", "captureSeconds", "scaledBy", "originalWidth",
                                                         "originalHeight")}
                if not shot.get("ok"):
                    r.update(ok=False, error="play client capture: %s" % shot.get("error"))
        if kind == "screenshot" or frames is not None:
            # a multiplayer test's client is its own window, never the simulated device (found 2026-11-23, RB1800)
            warn = _device_sim_warning()
            if warn:
                r["deviceSimWarning"] = warn
        if cam is not None:
            r = _camera_report(r, cam, keeper.shot if keeper and kind == "screenshot" else None)
        if save is not None:
            r = dict(_save_place_readback(r, save), runtimeState=RUNTIME_STATE_NOTE)
            save["savedOk"] = r.get("ok") is True
        if kind == "members":
            r = dict(_members_answer(r, members_wire, nonce), secondsUsed=seconds, timeoutUsed=timeout)
        r = _session_proved(r, kind, run_mode, players, nonce)
        if leftover:
            r["leftover"] = leftover
        if made_scratch:
            r["madeScratch"] = made_scratch
        back = _save_place_put_ids_back(save, save.get("savedOk")) if save is not None else None
        if back:
            r.update(back)
        return _probe_left_after(r)
    finally:
        if save is not None and save.get("idsSet") and "savedOk" not in save:
            _save_place_put_ids_back(save, False)    # an early return or a raise after the ids were set
        L.release_play_lock(nonce)     # every return after the lock was taken lets it go (a no-op once released)


# What a play probe stages in the edit place, by path and class (the plugin's play_probe op): the server probe, and with
# clients the client probe, its RemoteEvent and kind members' remote. The plugin removes them inside a recording whose
# result it does not read, so a busy recording leaves them in the place (found 2026-11-22).
PROBE_STAGED = ((("Workspace", "MifScratch", "MifPlayProbe"), "Script"),
                (("StarterPlayer", "StarterPlayerScripts", "MifClientProbe"), "LocalScript"),
                (("ReplicatedStorage", "MifProbeEvent"), "RemoteEvent"),
                (("ReplicatedStorage", "MifMembersRemote"), "RemoteEvent"),
                (("ReplicatedStorage", "MifMembersUnreliable"), "UnreliableRemoteEvent"))


def _probe_leftovers(server=True):
    """The staged probe objects present in the edit place now, as paths. server False leaves out the server probe, which
    the plugin replaces itself at the next probe."""
    left = []
    for parts, cls in PROBE_STAGED if server else PROBE_STAGED[1:]:
        got = _rbx("resolve", target=list(parts), timeout=15)
        if got.get("ok") and got.get("className") == cls:
            left.append(".".join(parts))
    return left


def _probe_left_after(r):
    """A probe's answer, failed probe-left when what it staged is still in the edit place after the run."""
    if not isinstance(r, dict):
        return r
    left = _probe_leftovers()
    if not left and r.get("probeRemoved") is not False:
        return r
    left = left or ["Workspace.MifScratch.MifPlayProbe"]
    note = ("the probe's own objects are still in the place after the run: %s (Studio had another recording open, so "
            "the removal did not run). A save keeps them and a publish ships them: remove them (rbx_destroy, full mode "
            "for those outside Workspace.MifScratch)" % ", ".join(left))
    return dict(r, ok=False, refusedRule=r.get("refusedRule") or "probe-left", probeLeft=left,
                error="%s; %s" % (r["error"], note) if r.get("error") else note)


TEST_RUN_PROPS = (("total", "Total"), ("passed", "Passed"), ("failed", "Failed"), ("skipped", "Skipped"),
                  ("status", "Status"), ("summary", "Summary"), ("duration", "Duration"))


def _session_proved(r, kind, run_mode, players, nonce):
    """What a play run itself proved, as composite rows on op play_probe in provedSession (COVERAGE_PLAN 7.2 #6):
    the StudioTestService call that started the session and EndTest that carried its result back - both only when the
    reply is this run's own (its nonce echoed, not recovered from a log) - and for kind tests TestService:RunTestsAsync
    with the TestRunResult properties it answered (rbx_test_run). A failed or foreign run adds nothing."""
    if not (isinstance(r, dict) and r.get("ok") and r.get("nonce") == nonce and r.get("nonceChecked") is True
            and not r.get("recovered")):
        return r
    cat, _err = _catalog()
    if cat is None:
        return r
    # the plugin's start(): players set -> ExecuteMultiplayerTestAsync, else runMode -> ExecuteRunModeAsync, else PlayMode
    start = ("ExecuteMultiplayerTestAsync" if players is not None else "ExecuteRunModeAsync" if run_mode
             else "ExecutePlayModeAsync")
    rows = _proved(cat, "StudioTestService", [start, "EndTest"], "call", "play_probe")
    tests = r.get("tests") if kind == "tests" and isinstance(r.get("tests"), dict) else None
    if tests is not None:
        rows += _proved(cat, "TestService", ["RunTestsAsync"], "call", "play_probe")
        rows += _proved(cat, "TestRunResult", [n for k, n in TEST_RUN_PROPS if tests.get(k) is not None], "read", "play_probe")
        if isinstance(tests.get("list"), list):
            rows += _proved(cat, "TestRunResult", ["GetTests"], "call", "play_probe")
    # its own key, beside a members run's play rows in proved; the server's logs stay last in the answer (_list_cuts)
    out = {k: v for k, v in r.items() if k != "logs"}
    out["provedSession"] = rows
    if "logs" in r:
        out["logs"] = r["logs"]
    return out


def _members_answer(r, wire, nonce):
    """A members run's answer: the server's and every client's step results judged as one (play_members.judge). What it
    proved may count only from this run's own reply: its nonce echoed, not recovered from a log."""
    from mif_roblox import play_members as PM
    cat, _err = _catalog()
    trusted = (isinstance(r, dict) and r.get("nonce") == nonce and r.get("nonceChecked") is True
               and not r.get("recovered") and cat is not None)
    judged = PM.judge(cat, wire, [r], trusted) if cat is not None else {"error": "no catalog"}
    out = dict(r)
    out["members"] = judged
    out["proved"] = judged.get("proved") or []
    out["clients"] = [{k: v for k, v in c.items() if k != "members"} if isinstance(c, dict) else c
                      for c in r.get("clients") or []] if "clients" in r else r.get("clients")
    if out.get("clients") is None:
        out.pop("clients", None)
    if judged.get("missing") and out.get("ok"):
        out["warning"] = "plan steps %s came back from no side of the test" % judged["missing"]
    # A side that planned steps and sent no members answer at all lost every one of them: that is a failed run, not a
    # warning. Live (RB2600): a server plan of 40 network reads outlasted seconds, the play server's answer was waited for
    # only 30 s past the window ("no MifBridge plugin answered in the play server within 93 s"), and the run answered ok
    # with 84 of its 132 steps missing.
    # A client that reported but whose plugin gave no members answer carries members {error} and nothing else (live
    # RB2604: "no members answer from the MifBridge plugin in this client"): that side answered nothing either.
    sides = PM.collect(r) if isinstance(r, dict) else []
    empty = {side: res.get("error") for side, res, _p in sides
             if res.get("error") and not any(res.get(k) for k in ("steps", "watches", "causes"))}
    answered = {side for side, _res, _p in sides} - set(empty)
    lost = sorted({s["on"] for s in wire or [] if s.get("op") != "wait"} - answered)
    if lost and out.get("ok"):
        n = len([s for s in wire if s.get("op") != "wait" and s.get("on") in lost])
        why = "; ".join(str(x) for x in [r.get("membersError")] + [empty.get(s) for s in lost] if x) or "no answer"
        out.update(ok=False, refusedRule="members-lost", lostSides=lost,
                   error="the play %s sent no members answer (%s): its %d planned steps are lost. The plugin there runs "
                         "the steps in order and the server's answer is waited for 30 s past the window: a plan whose "
                         "reads and calls take longer than seconds (a network call can take its full 10 s) needs a "
                         "larger seconds" % (" and ".join(lost), why, n))
    return out


PROBE_MAX_SECONDS = 1800


def _place_revision(place_id):
    """(revision id as int or None, error) of a place asset: GET /assets/v1/assets/{id} with the person's key."""
    from mif_roblox import opencloud as OC
    key, err = OC.read_key()
    if err:
        return None, err.get("error")
    st, a, _w = OC._call("GET", "/assets/v1/assets/%d" % place_id, key, tries=3)
    try:
        return int((a or {}).get("revisionId")), None
    except (TypeError, ValueError):
        return None, OC.redact("GET /assets/v1/assets/%s answered %s: %s" % (place_id, st, json.dumps(a)[:200]), key)


SAVE_PLACE_PLUGIN = (0, 17, 10)   # its play server checks its own ids against the gate's before SavePlaceAsync
CLIENT_WAIT_PLUGIN = (0, 17, 8)   # its play server waits for the clients and they keep what they print late (RB1850)
PLUGIN_RESTART_STEP = ("install the current plugin and restart this Studio: a running Studio does not reload a replaced "
                       "plugin (RB1970)")


def _version_tuple(v):
    try:
        return tuple(int(x) for x in str(v).split("."))
    except ValueError:
        return None


def _plugin_older(need, st=None):
    """(the connected plugin's version, whether it is older than need): from st, an rbx_status answer (plugin.plugin),
    or else the plugin's own ping, the version rbx_status shows as ping.plugin. older is None when the version cannot be
    read: it is never guessed."""
    if st is not None:
        have = ((st or {}).get("plugin") or {}).get("plugin")
    else:
        try:
            p = _rbx("ping", nonce=uuid.uuid4().hex, timeout=15)
        except Exception:  # noqa: BLE001 - an unreadable version is unknown, never a guessed one
            p = {}
        have = p.get("plugin") if isinstance(p, dict) and p.get("ok") else None
    have = have if isinstance(have, str) else None
    vt = _version_tuple(have) if have else None
    return have, (None if vt is None else vt < need)


RUNTIME_STATE_NOTE = ("this version is the play server's DataModel as it ran (RB2901): what server scripts made, moved or "
                      "removed by the save, and where physics left unanchored parts, are in it; the edit state is not")


def _runtime_state_refusal(accept):
    """None when a save_place probe may run (accept_runtime_state true), else the refusal saves-runtime-state. Measured by
    lane PS on 2026-11-22 (RB2901): SavePlaceAsync from the Run-mode play server saved a part a server Script made at run
    time into the version, so this route publishes the running game, not the place as edited."""
    if not isinstance(accept, bool):
        return {"ok": False, "ran": False, "error": "accept_runtime_state is true or false"}
    if accept:
        return None
    return {"ok": False, "ran": False, "refusedRule": "saves-runtime-state", "ledger": "RB2901",
            "error": "kind save_place saves the play server's DataModel as it runs, not the place as edited: measured, a "
                     "part a server Script made at run time was in the saved version (RB2901), and Run mode runs every "
                     "server Script and physics before the save. Nothing was saved. To publish the edit state, save the "
                     "place to its file (rbx_place_save, or the person's File > Save) and send that file with "
                     "rbx_place_publish; pass accept_runtime_state true only to save the running state on purpose"}


def _save_place_gate(confirm, allow_ids, place_id=None):
    """(what the read-back needs, None) or (None, the refusal) for a save_place probe: confirm, full mode, plugin 0.17.10,
    the open place and experience on the write-target list, and the place's revision before. A local copy (PlaceId 0)
    is first given place_id and its experience (SetPlaceId, SetUniverseId: no cloud edit session holds a local file,
    RB2001); a place open under another id is never retargeted."""
    from mif_roblox import opencloud as OC
    if confirm is not True:
        return None, {"ok": False, "ran": False, "refusedRule": "needs-confirm",
                      "error": "save_place saves the open place to Roblox as a new version: pass confirm true once the "
                               "person agreed"}
    if place_id is not None and not (isinstance(place_id, int) and not isinstance(place_id, bool) and place_id > 0):
        return None, {"ok": False, "ran": False, "refusedRule": "type", "error": "place_id is a published place's id"}
    mode, st = _plugin_mode()
    if mode != "full":
        return None, {"ok": False, "ran": False, "refusedRule": "full-mode", "mode": mode,
                      "error": "save_place writes a version of the place to Roblox: the MifBridge panel must be in full mode "
                               "(it is %s)" % mode}
    have, older = _plugin_older(SAVE_PLACE_PLUGIN, st)
    if older is not False:
        return None, {"ok": False, "ran": False, "refusedRule": "plugin-version", "plugin": have,
                      "error": "save_place needs MifBridge plugin %s or later (this Studio runs %s): its play server checks "
                               "the place it saves to" % (".".join(map(str, SAVE_PLACE_PLUGIN)), have),
                      "steps": [PLUGIN_RESTART_STEP]}

    pid, gid, unread = _place_ids()
    if unread:
        # a failed read is not "a local copy": taking it for one retargeted a published place (found 2026-11-22)
        return None, {"ok": False, "ran": False, "refusedRule": "place-ids-unread",
                      "error": "%s. save_place changed nothing: it never takes a failed read for a local copy" % unread}
    given = None
    if place_id is not None and pid is not None and pid != place_id:
        return None, {"ok": False, "ran": False, "refusedRule": "place-mismatch", "placeId": pid, "gameId": gid,
                      "error": "the open place is place %s, not %s: save_place never retargets a place open under another "
                               "id" % (pid, place_id)}
    if pid is None:
        if place_id is None:
            return None, {"ok": False, "ran": False, "refusedRule": "no-place-id", "placeId": pid, "gameId": gid,
                          "error": "the open place has no place id (a local file reads PlaceId 0): name the published place "
                                   "it is a copy of in place_id",
                          "steps": ["pass place_id, the published place this file is a copy of; it and its experience must "
                                    "be on the write-target list or in allow_ids"]}
        from mif_roblox import domain_store as DS
        uni, err = DS.universe_of_place(place_id)
        if err:
            return None, {"ok": False, "ran": False, "refusedRule": "place-universe", "placeId": place_id, "error": err}
        bad = OC.write_target_check(uni, place_id, allow_ids)
        if bad:
            return None, dict(bad, placeId=place_id, gameId=uni)
        # the ids are SET only once the machine play lock is held (_save_place_set_ids), and put back if the save fails
        given = {"placeId": place_id, "universeId": uni}
        pid, gid = place_id, uni
    else:
        bad = OC.write_target_check(gid, pid, allow_ids)
        if bad:
            return None, dict(bad, placeId=pid, gameId=gid)
    rev, err = _place_revision(pid)
    return {"placeId": pid, "gameId": gid, "revisionBefore": rev, "revisionBeforeError": err, "idsGiven": given,
            "opened": _opened_as((st.get("plugin") or {}).get("studioPid"))}, None


def _place_ids():
    """(PlaceId or None, GameId or None, error or None) of the open place. 0 (a local file) reads as None; a read that
    failed, or answered no PlaceId at all, is the error, never a local copy."""
    g = _rbx("get_props", target=[], props=["PlaceId", "GameId"])
    if not g.get("ok"):
        return None, None, "PlaceId and GameId could not be read: %s" % (g.get("error") or "no answer")
    vals = g.get("values") if isinstance(g.get("values"), dict) else {}
    out = []
    for k in ("PlaceId", "GameId"):
        v = vals.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v != int(v):
            return None, None, "the read of %s answered %r, not a whole number" % (k, v)
        out.append(int(v) or None)
    return out[0], out[1], None


def _set_place_ids(place_id, uni):
    """None, or the error: DataModel:SetPlaceId and SetUniverseId, then both read back."""
    for method, v in (("SetPlaceId", place_id), ("SetUniverseId", uni)):
        w, _why = V.encode("int64", v)
        r = _rbx("call", target=[], method=method, args=[w], argc=1, readOnly=False, yields=False, callTimeout=10)
        if not r.get("ok"):
            return "DataModel:%s(%s) failed: %s" % (method, v, r.get("error"))
    pid, gid, unread = _place_ids()
    if unread:
        return unread
    if (pid or 0, gid or 0) != (place_id or 0, uni or 0):
        return "SetPlaceId and SetUniverseId read back %s and %s, not %s and %s" % (pid, gid, place_id, uni)
    return None


def _save_place_set_ids(save):
    """None, or the refusal: gives a local copy the place's ids, under the machine play lock. Nothing to do when the
    open place already holds them."""
    given = save.get("idsGiven")
    if not given:
        return None
    err = _set_place_ids(given["placeId"], given["universeId"])
    if err:
        save["idsSet"] = True       # one of the two may have landed: the finally puts both back
        return {"ok": False, "ran": True, "refusedRule": "place-ids", "error": err}
    save["idsSet"] = True
    return None


def _save_place_put_ids_back(save, saved):
    """After a save_place probe: a local copy whose ids this call set gets 0 and 0 back unless the save succeeded (a
    saved copy now IS that place's file). Answers what it did, for the reply."""
    if not save or not save.get("idsSet") or saved:
        return None
    err = _set_place_ids(0, 0)
    return {"idsPutBack": err is None, "idsPutBackError": err}


def _opened_as(pid):
    """How the Studio saving was opened, from its command line: "file" (-task EditFile, a local file: no edit session),
    "cloud" (-task EditPlace, the place opened from Roblox) or None when unreadable."""
    from mif_roblox import launch as L
    cl = L.command_line(pid) if pid else None
    m = re.search(r"-task (EditFile|EditPlace)\b", cl or "")
    return {"EditFile": "file", "EditPlace": "cloud"}.get(m.group(1)) if m else None


def _save_place_readback(r, save):
    """The probe's savePlace answer checked against the place asset's revision: higher is ok, exactly one higher is
    verified, and not higher is not proven."""
    sp = r.get("savePlace") if isinstance(r.get("savePlace"), dict) else None
    rev, err = _place_revision(save["placeId"])
    out = dict(save, revisionAfter=rev, revisionAfterError=err)
    r["saved"] = out
    if sp is None:
        r.update(ok=False, error=r.get("error") or "the probe sent no savePlace answer")
    elif not sp.get("ok"):
        from mif_roblox import opencloud as OC
        held = OC.session_held(sp.get("error") or "")
        r.update(ok=False, refusedRule=r.get("refusedRule") or ("place-session-open" if held else "save-failed"),
                 error="SavePlaceAsync failed in the play server: %s" % sp.get("error"))
        if held and save.get("opened") == "file":
            # a local copy holds no edit session: a Studio that held the place's id and stopped minutes ago still does
            r["steps"] = ["a Studio that held this place (opened from Roblox, or a local copy given its id) stopped within "
                          "the last minutes, and its hold outlives it: the same save failed 31 s and 56 s after such a stop "
                          "and succeeded 5.6 minutes after (RB2002). Close any Studio holding the place, wait about five "
                          "minutes, then save again"]
        elif held:
            r["steps"] = ["the Studio running this test holds the place in a Team Create session (opened from Roblox): "
                          "close it, open a local copy of the place file (rbx_launch place_file) and save again naming "
                          "place_id (RB2001), about five minutes after this Studio closed (RB2002)"]
    else:
        before = save.get("revisionBefore")
        # ok on any rise; verified only on exactly one, as rbx_place_publish (a Team Create save can land in between,
        # and calling a good save failed invites a second one: found 2026-11-22)
        good = rev is not None and before is not None and rev > before
        one = good and rev == before + 1
        r.update(ok=good, verified=one)
        if good and not one:
            r["revisionNote"] = ("saved: revision %s is more than one above %s, so another save (Team Create) landed "
                                 "between the read and this one; do not save again" % (rev, before))
        if not good:
            r["error"] = "SavePlaceAsync answered, but the place's revision reads %s after %s" % (rev, before)
    return r


def _vec3(v):
    if isinstance(v, dict):
        v = [v.get("x", v.get("X")), v.get("y", v.get("Y")), v.get("z", v.get("Z"))]
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        return None
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v):
        return None
    return [float(x) for x in v]


def _client_view_args(kind, frame_times, client_windows, camera, seconds, run_mode):
    """(frames [from, to] or None, the client window mode, camera {position, lookAt, fov?} or None, refusal or None) for
    rbx_play_probe's frame_times, client_windows and camera."""
    def no(msg):
        return None, None, None, {"ok": False, "ran": False, "error": msg}
    frames = None
    if frame_times is not None and frame_times is not False:
        if kind in ("save_place", "players") or run_mode:
            return no("frame_times times play-test clients: not with kind %s or run_mode" % kind)
        if frame_times is True:
            frames = [0.0, float(seconds)]
        elif isinstance(frame_times, (list, tuple)) and len(frame_times) == 2 and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in frame_times):
            frames = [float(frame_times[0]), float(frame_times[1])]
        else:
            return no("frame_times is true (the whole window) or [from, to]: seconds after the window starts")
        if not 0 <= frames[0] < frames[1] <= seconds:
            return no("frame_times [from, to] must have 0 <= from < to <= seconds (%g)" % seconds)
    if client_windows is None:
        windows = "behind" if frames is not None or kind == "screenshot" else "minimized"
    elif client_windows in ("behind", "minimized"):
        windows = client_windows
    else:
        return no("client_windows is behind (under the person's windows, no focus) or minimized")
    if kind == "screenshot" and windows == "minimized":
        return no("kind screenshot keeps the client window behind: a minimized window captures blank (RB1037)")
    cam = None
    if camera is not None:
        if kind != "screenshot" and frames is None:
            return no("camera is held in the play client for kind screenshot or frame_times")
        if not isinstance(camera, dict):
            return no("camera is {position: [x, y, z], look_at: [x, y, z], fov?}")
        pos, look = _vec3(camera.get("position")), _vec3(camera.get("look_at", camera.get("lookAt")))
        fov = camera.get("fov")
        if pos is None or look is None:
            return no("camera needs position and look_at, each [x, y, z] of finite numbers")
        if math.dist(pos, look) < 1e-3:
            return no("camera position and look_at are the same point: no direction to look")
        if fov is not None and (not isinstance(fov, (int, float)) or isinstance(fov, bool) or not 1 <= fov <= 120):
            return no("camera fov is 1-120 degrees (Camera.FieldOfView, vertical)")
        unknown = sorted(set(camera) - {"position", "look_at", "lookAt", "fov"})
        if unknown:
            return no("camera takes position, look_at and fov; not %s" % ", ".join(unknown))
        cam = {"position": pos, "lookAt": look}
        if fov is not None:
            cam["fov"] = float(fov)
    return frames, windows, cam, None


def _frame_report(r, frames, keeper, windows):
    """frameTimes: each client's RenderStepped median, p95 and max ms over [from, to] of the window (plugin 0.17.12), with
    the window state its test window ran in (the keeper's readings: behind, minimized, foreground or mixed). A minimized
    test window renders at a cap (RB2105), so its numbers are the cap's, and the answer says so. A client with no frames
    fails the run with why."""
    states = keeper.client_states() if keeper and hasattr(keeper, "client_states") else {}
    names = sorted({v["state"] for v in states.values()})
    state = names[0] if len(names) == 1 else ("mixed" if names else ("not managed" if keeper is None else "unread"))
    rows, why = [], []
    for c in r.get("clients") or []:
        if not isinstance(c, dict):
            continue
        f = c.get("frames")
        player = c.get("player") or c.get("localPlayer")
        if not isinstance(f, dict):
            why.append("%s sent no frame times (a plugin before 0.17.12 in this Studio: restart it, RB1970)" % player)
            continue
        row = {"player": player, "from": f.get("from"), "to": f.get("to"), "frames": f.get("count"),
               "medianMs": _r2(f.get("medianMs")), "p95Ms": _r2(f.get("p95Ms")), "maxMs": _r2(f.get("maxMs")),
               "meanMs": _r2(f.get("meanMs")), "physicsFps": _r2(f.get("physicsFps")), "windowState": state}
        for key in ("renderCpu", "renderGpu"):
            if isinstance(f.get(key), dict):
                row[key + "Ms"] = {k[:-2]: _r2(v) for k, v in f[key].items() if k.endswith("Ms")}
        if f.get("dropped"):
            row["dropped"] = f["dropped"]
        if not f.get("windowStarted"):
            why.append("%s never heard the window start (the play server's plugin is older than 0.17.12)" % player)
        elif not f.get("count"):
            why.append("%s rendered no frame between %gs and %gs of the window" % (player, frames[0], frames[1]))
        rows.append(row)
    if not rows and not why:
        why.append("no client reported, so no frame was timed")
    out = dict(r, frameTimes=rows, frameWindow={"from": frames[0], "to": frames[1], "clientWindows": windows,
                                                "windowState": state,
                                                "states": {str(p): v for p, v in sorted(states.items())}})
    if state in ("behind", "minimized", "mixed", "foreground", "not managed", "unread"):
        out["frameWarning"] = {
            "behind": "the client windows were behind the person's windows, not in the foreground: such a test window "
                      "renders at a cap (66.7 ms median, the same as minimized or off-screen, RB2130), so compare these "
                      "numbers only with runs in the same window state; the probe never takes the foreground",
            "minimized": "the client windows were minimized: a minimized test window renders at a cap (66.7, 30.8 or 16.7 "
                         "ms median on one build, RB2105), so these are the cap's numbers; compare only with runs in the "
                         "same window state",
            "mixed": "the client windows changed state during the run (%s): the numbers mix window states" % states,
            "foreground": "the client window was the foreground window for the run: Studio opens a test window active, "
                          "lowering it does not take the foreground away, and nothing else took it (live 82 of 82 "
                          "readings, RB2130); a foreground window runs faster than one behind (24.5 vs 66.7 ms median), "
                          "so compare only runs in the same window state",
            "not managed": "minimize_windows false: the client windows were left as Studio opened them",
            "unread": "no client window's state could be read"}[state]
    if why:
        out.update(ok=False, error="; ".join(why) if not r.get("error") else "%s; %s" % (r.get("error"), "; ".join(why)))
    return out


def _r2(v):
    return round(v, 2) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def _camera_report(r, cam, shot):
    """camera: what the client held (read back: position, look vector, fov, viewport, where look_at fell on screen) and
    whether the window capture was taken while it was held (the client's wall-clock hold window against the capture's
    time). A capture taken before the camera was held, or a client that held nothing, fails the run."""
    reports = [c.get("camera") for c in r.get("clients") or [] if isinstance(c, dict) and c.get("camera") is not None]
    if not reports:
        why = "no client said it held the camera (a plugin before 0.17.12 in this Studio: restart it, RB1970)"
        return dict(r, ok=False, camera={"asked": cam, "held": False},
                    error="%s; %s" % (r["error"], why) if r.get("error") else why)
    got = reports[0] if isinstance(reports[0], dict) else {"error": str(reports[0])}
    out = {"asked": cam, "held": bool(got.get("frames")), "client": got}
    if got.get("error") or not got.get("frames"):
        return dict(r, ok=False, camera=out, error=r.get("error") or "the client did not hold the camera: %s"
                    % (got.get("error") or "no frame was drawn with it"))
    want = [b - a for a, b in zip(cam["position"], cam["lookAt"])]
    n = math.sqrt(sum(x * x for x in want)) or 1.0
    want = [x / n for x in want]
    lv = got.get("lookVector") or [0, 0, 0]
    out["positionError"] = round(math.dist(got.get("position") or [0, 0, 0], cam["position"]), 4)
    out["lookAngleDeg"] = round(math.degrees(math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(lv, want)))))), 3)
    if shot is not None and isinstance(got.get("heldFrom"), (int, float)):
        at_ms = shot.get("at", 0) * 1000.0
        out["captureInHold"] = got["heldFrom"] <= at_ms <= (got.get("heldTo") or 0) + 200
        out["captureAfterHoldStart"] = round((at_ms - got["heldFrom"]) / 1000.0, 2)
        if not out["captureInHold"]:
            return dict(r, ok=False, camera=out, error="the window was captured %.1f s %s the client held the camera: "
                        "raise value (the capture time)" % (abs(out["captureAfterHoldStart"]),
                                                            "before" if at_ms < got["heldFrom"] else "after"))
    return dict(r, camera=out)


def _viewport_box(rects, size, w, h):
    """([x, y, width, height] of the 3D viewport inside a captured client window, None) or (None, why). Studio's
    viewport is a native child window of the test window (Qt): the visible child whose size is the client's
    Camera.ViewportSize (within 2 px; measured 2140 x 1161 at 11, 215 of a 2576 x 1456 window, RB2131), else one of the
    same aspect (a scaled display). Several identical children stack there; any of them is the box."""
    if not isinstance(size, (list, tuple)) or len(size) != 2 or not all(isinstance(v, (int, float)) for v in size):
        return None, "the client did not report its viewport size (a plugin before 0.17.12)"
    if not isinstance(rects, list) or not rects:
        return None, "the window's child boxes were not read (%s)" % ((rects or {}).get("error") if isinstance(rects, dict)
                                                                     else "none")
    vw, vh = float(size[0]), float(size[1])
    inside = [b for b in rects if isinstance(b, dict) and b.get("w", 0) > 0 and b.get("h", 0) > 0
              and b.get("x", -1) >= 0 and b.get("y", -1) >= 0 and b["x"] + b["w"] <= w and b["y"] + b["h"] <= h]
    exact = [b for b in inside if abs(b["w"] - vw) <= 2 and abs(b["h"] - vh) <= 2]
    if exact:
        b = exact[0]
        return [b["x"], b["y"], b["w"], b["h"]], None
    if vh > 0:
        same = [b for b in inside if abs(b["w"] / b["h"] - vw / vh) <= 0.01 * vw / vh and b["w"] < w]
        if same:
            b = max(same, key=lambda b: b["w"] * b["h"])
            return [b["x"], b["y"], b["w"], b["h"]], None
    return None, "no child window of the test window is %g x %g (the client's viewport)" % (vw, vh)


def _crop_rgba(w, h, rgba, box):
    """The [x, y, width, height] box of a w x h RGBA image, clamped to it."""
    x, y, cw, ch = (int(v) for v in box)
    x, y = max(0, min(x, w - 1)), max(0, min(y, h - 1))
    cw, ch = max(1, min(cw, w - x)), max(1, min(ch, h - y))
    rows = [rgba[((y + j) * w + x) * 4:((y + j) * w + x + cw) * 4] for j in range(ch)]
    return cw, ch, b"".join(rows)


def _takes(fn, name):
    """Whether fn (a function, or a class by its __init__) takes the keyword `name`."""
    try:
        ps = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in ps.values()):
        return True
    return name in ps and ps[name].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def _hook_call(fn, *args, **optional):
    """Call a module-level hook of the play probe (_WindowKeeper, _end_test_processes, _from_test_logs) with the keywords
    it takes and no others. Callers replace these hooks with their own (a driver's own cleanup), written for the
    signature of their day: `own` arrived later as a 4th positional and broke every override that took wait there
    (TypeError: got multiple values for argument 'wait', found 2026-11-21). Newer arguments are keyword-only and passed
    only to a hook that names them, so an override written for an older signature is called the way it was written."""
    return fn(*args, **{k: v for k, v in optional.items() if _takes(fn, k)})
PROBE_CALL_CAP = 590            # the daemon answers one call for at most 600 s (D.MAX_TIMEOUT)


def _list_cuts(r, clients):
    """Each list of a probe's answer is capped on its own (plugin 0.17.8: the server's Output keeps its first 1000 and
    last 1000 lines, each client's its first 500 and last 500, errors 200, client reports 16), so a server that prints
    thousands of lines never crowds a client's lines out. cuts says, per list, what was kept of how many; the server's
    logs go last, after clients, so an answer printed short still shows the clients. A run with clients that came back
    with none says why (clientsWarning), never an empty list alone."""
    if not isinstance(r, dict):
        return r
    cuts = {}
    logs = r.get("logs")
    if isinstance(logs, list):
        total = r.get("logsTotal") if isinstance(r.get("logsTotal"), int) else len(logs)
        cuts["server"] = {"kept": len(logs), "total": total, "cut": total > len(logs)}
    errs = r.get("errors")
    if isinstance(errs, list):
        total = r.get("errorsTotal") if isinstance(r.get("errorsTotal"), int) else len(errs)
        cuts["errors"] = {"kept": len(errs), "total": total, "cut": total > len(errs)}
    reports = [c for c in r.get("clients") or [] if isinstance(c, dict)]
    if clients or reports:
        rows = []
        for c in reports:
            n = len(c.get("logs") or [])
            total = c.get("logsTotal") if isinstance(c.get("logsTotal"), int) else n
            rows.append({"player": c.get("player") or c.get("localPlayer"), "kept": n, "total": total, "cut": total > n})
        cuts["clients"] = rows
        sent = r.get("clientReportsTotal") if isinstance(r.get("clientReportsTotal"), int) else len(reports)
        cuts["clientReports"] = {"kept": len(reports), "total": sent, "cut": sent > len(reports)}
    if not cuts:
        return r
    out = {k: v for k, v in r.items() if k not in ("logs", "errors")}
    out["cuts"] = cuts
    partial = [c for c in reports if c.get("partial")]
    if partial:
        # plugin 0.17.12: a report sent in pieces that did not all arrive is named, with its size (RB2073)
        out["clientsWarning"] = "; ".join("%s: %s" % (c.get("player"), c.get("error") or "partial report")
                                          for c in partial)
        out["partialReports"] = [{k: c.get(k) for k in ("player", "chars", "pieces", "piecesGot", "notice")}
                                 for c in partial]
    if clients and not reports:
        why = ["no client reported"]
        wait = r.get("clientWait") or {}
        if wait.get("timedOut"):
            why.append("no client said it started within %ss of the server's start (clientWait)" % wait.get("limit"))
        missing = (r.get("clientsMissing") or {}).get("players") or []
        if missing:
            why.append("players with no report: %s" % ", ".join(str(m.get("player")) for m in missing))
        late = [m for m in missing if m.get("ready")]
        rw = r.get("reportWait") or {}
        if late and rw.get("limit") is not None:
            # plugin 0.17.12: the server waited reportWait for them after the window (RB2133)
            why.append("%s said it runs but sent no report within %ss of the window's end: its client was busy (a long "
                       "frame or a large report to build) or its probe script stopped; a longer timeout gives it more"
                       % (", ".join(str(m.get("player")) for m in late), round(rw["limit"], 1)))
        if "clientWait" not in r and "clientsMissing" not in r:
            why.append("the plugin that ran the test says nothing about its clients (before 0.17.8)")
        out["clientsWarning"] = "; ".join(why)
    if errs is not None:
        out["errors"] = errs
    if logs is not None:
        out["logs"] = logs
    return out


def _match_nonce(r, nonce):
    """A result is this run's only when it carries this run's nonce (plugin 0.17.4 echoes it). One with another nonce is
    another Studio's run handed back here (RB1015): dropped, and the logs are read for this run's own."""
    if not isinstance(r, dict):
        return {"ok": False, "error": "the plugin answered %r" % (r,)}
    if "nonce" in r and r.get("nonce") != nonce:
        return {"ok": False, "refusedRule": "foreign-result", "foreignNonce": r.get("nonce"),
                "requestId": r.get("requestId"), "ms": r.get("ms"),
                "error": "the result that came back carries another run's nonce (%s): it is not this run's and was dropped"
                         % r.get("nonce")}
    if r.get("kind") is not None or r.get("ok"):
        r["nonceChecked"] = "nonce" in r
    return r


def _late_reply(rid):
    try:
        st = _http("GET", "/request?id=%s" % rid)[1] or {}
    except (OSError, ValueError):
        return None
    if st.get("state") == "done" and isinstance(st.get("reply"), dict):
        return dict(st["reply"], requestId=rid, lateReply=True)
    return None


def _await_late(L, r, own, t_start, nonce, deadline, every=2.0):
    """The daemon gave up on the call but the plugin plays the run to its end (RB1194). Until deadline: the late reply
    (rbx_status request_id's route), or the result the server printed into its own log; stop early when the server's
    log shows it exited with no result."""
    rid = r.get("requestId")
    while time.time() < deadline:
        got = _late_reply(rid)
        if got is not None:
            return got
        logs = L.test_logs(own, t_start)
        server = next((x for x in logs if x["role"] == "server"), None)
        res = L.probe_result_from_log(server, nonce)
        if res is not None:
            return dict(res, ok=res.get("ok") is not False, recovered=True, recoveredFrom=server["file"],
                        requestId=rid, lostReply="the run outlived the daemon's call")
        if server and any(e["event"] == "normalExit" for e in server["events"]):
            return r
        time.sleep(every)
    return dict(r, error="no result within timeout: %s" % r.get("error"))


CUT_GRACE = 30.0    # a cut run's processes still there this long past its own deadline (timeout) are called stuck
EXIT_GRACE = 30.0   # once a cut run's result is in, how long its server has to exit by itself (measured 1.4 s, RB2928)


def _cut_reply(r):
    """The plugin's answer when the session ended without the probe's result: the reply channel was cut (another
    Studio's test ended, RB1015, RB2928) or the run really ended. Which one is told by whether this Studio's test
    processes still run."""
    return isinstance(r, dict) and not r.get("ok") and r.get("refusedRule") == "play-failed"


def _result_blob_id(nonce):
    """The blob id a play server posts its result under to this daemon (the route that is not a log line, N81)."""
    return "probe-result-%s" % nonce


def _result_from_daemon(nonce):
    """This run's result as its play server posted it to this daemon in parts (GET /blob), or None: none posted (a
    plugin without the route), not all parts in yet, or another run's nonce."""
    if not nonce:
        return None
    try:
        data, _status = _fetch_blob(_result_blob_id(nonce), timeout=5.0)
        res = json.loads(data.decode("utf-8")) if data else None
    except Exception:  # noqa: BLE001 - no daemon, no blob, or not JSON: no result by this route
        return None
    return res if isinstance(res, dict) and res.get("nonce") == nonce else None


def _await_cut(L, r, own, t_start, nonce, deadline, every=2.0):
    """A run whose reply was cut while its test processes still run (RB2928: the server plays its window to the end,
    prints its result, and exits by itself once SendTestResult times out). Wait for this nonce's result - posted to the
    daemon, or in the server's own log - and for the processes to exit, up to deadline; never end anything here. The
    answer carries cut: when the wait began, how long it took, whether the processes exited by themselves, and where
    the result came from."""
    t_cut = time.time()
    found = src = srv_file = None
    found_at = None
    exited = False

    def look():
        got = _result_from_daemon(nonce)
        if got is not None:
            return got, "daemon", None
        logs = L.test_logs(own, t_start)
        server = next((x for x in logs if x["role"] == "server"), None)
        res = L.probe_result_from_log(server, nonce)
        return (res, "log", server["file"]) if res is not None else (None, None, None)

    while True:
        if found is None:
            found, src, srv_file = look()
            found_at = time.time() if found is not None else None
        if not L.leftover_tests(own):
            exited = True
            break
        if time.time() >= deadline or (found_at is not None and time.time() >= found_at + EXIT_GRACE):
            break
        time.sleep(every)
    if found is None:
        found, src, srv_file = look()       # printed just before the processes went
    cut = {"waitStartedSeconds": round(t_cut - t_start, 1), "waitedSeconds": round(time.time() - t_cut, 1),
           "exitedBySelf": exited, "boundSeconds": round(max(0.0, deadline - t_cut), 1), "resultFrom": src,
           "lostReply": r.get("error")}
    if found is None:
        return dict(r, cut=cut)
    out = dict(found, ok=found.get("ok") is not False, recovered=True, resultFrom=src, lostReply=r.get("error"),
               cut=cut, nonceChecked=True, requestId=r.get("requestId"))
    if srv_file:
        out["recoveredFrom"] = srv_file
    return out


def _collect_run(which):
    """rbx_play_probe collect: an earlier run's result, by its nonce or requestId, or 'last' (this port's newest)."""
    from mif_roblox import launch as L
    if not isinstance(which, str) or not which:
        return {"ok": False, "ran": False, "error": "collect is 'last', or a run's nonce or requestId"}
    run = L.find_run(PORT, which)
    if run is None:
        return {"ok": False, "ran": False, "refusedRule": "no-run",
                "error": "no play probe run %r is recorded for port %d" % (which, PORT)}
    brief = {k: run.get(k) for k in ("nonce", "requestId", "kind", "seconds", "tStart")}
    if run.get("requestId"):
        got = _late_reply(run["requestId"])
        if got is not None:
            out = _match_nonce(got, run["nonce"])
            if out.get("ok"):
                return dict(out, collected=True, run=brief)
            # a late reply that is itself a failure (the session ended first) says less than the run's own log
    posted = _result_from_daemon(run["nonce"])
    if posted is not None:
        return dict(posted, ok=posted.get("ok") is not False, recovered=True, resultFrom="daemon", collected=True,
                    run=brief)
    logs = L.test_logs(run.get("own"), run.get("tStart") or 0)
    server = next((x for x in logs if x["role"] == "server"), None)
    res = L.probe_result_from_log(server, run["nonce"])
    if res is not None:
        return dict(res, ok=res.get("ok") is not False, recovered=True, recoveredFrom=server["file"], collected=True,
                    run=brief)
    if server and not any(e["event"] == "normalExit" for e in server["events"]):
        alive = [p for p, _s in L.leftover_tests(run.get("own"))]
        if alive:
            return {"ok": False, "running": True, "run": brief, "testProcesses": alive,
                    "error": "the run is still playing (its server has not printed a result): collect again later"}
    out = _from_test_logs(L, {"ok": False, "refusedRule": "play-failed",
                              "error": "no reply and no printed result for this run"}, run.get("own"),
                          run.get("tStart") or 0, run["nonce"])
    return dict(out, collected=True, run=brief)


def _from_test_logs(L, r, own, t_start, nonce=None):
    """A probe that got no result (the session ended first, or SendTestResult timed out: RB1015): read the play test's own
    Studio logs. When the server printed the probe's result (plugin 0.16.6), that result is the answer, marked recovered.
    Otherwise the answer keeps ok:false and carries the Output, errors and client logs gathered up to the end, and when
    and how each process ended (session)."""
    keep = {k: r[k] for k in ("requestId", "ms", "testProcesses", "testWindows", "withheldAs", "cut", "cutWarning")
            if k in r}
    posted = _result_from_daemon(nonce)
    if posted is not None:
        # the play server posted its result to this daemon (no log line limit, and Run mode too: N81)
        return dict(posted, **keep, ok=posted.get("ok") is not False, recovered=True, resultFrom="daemon",
                    lostReply=r.get("error"), nonceChecked=True)
    logs = L.test_logs(own, t_start)
    if not logs:
        return dict(r, session={"logsFound": 0, "why": "no play-test log of this Studio (-editpid %s) since the probe "
                                                       "started: run mode writes no separate log, or own is unknown" % own})
    alive = {p for p, _s in L.leftover_tests(own)} if own else set()
    rep = L.session_report(logs, t_start, alive=alive)
    server = next((x for x in logs if x["role"] == "server"), None)
    got = L.probe_result_from_log(server, nonce)
    brief = {"endOrder": rep["endOrder"], "server": {k: v for k, v in (rep["server"] or {}).items() if k != "output"},
             "clients": [{k: v for k, v in c.items() if k != "output"} for c in rep["clients"]]}
    if got is not None:
        out = dict(got, **keep)
        out.update(ok=got.get("ok") is not False, recovered=True, recoveredFrom=server["file"], lostReply=r.get("error"),
                   session=brief, resultFrom="log")
        return out
    srv = rep["server"] or {}
    out = dict(r)
    streamed = L.samples_from_log(server, nonce)
    if streamed:
        # a long sample probe printed its samples as it went (plugin 0.17.7): those up to the end are the answer's
        out["samples"], out["samplesFrom"] = streamed, "log"
    out.update(logs=srv.get("output") or [], errors=srv.get("errors") or [], session=brief,
               clients=[{"logs": c["output"], "file": c["file"]} for c in rep["clients"]],
               endedEarly=True,
               error="%s. The play test's own logs were read instead: logs, errors and clients hold what its scripts "
                     "printed up to the end, and session says when and how each process ended" % r.get("error"))
    return out


def _not_a_lane(L, p, s):
    """With several lanes, a Studio another lane launched during this probe is new too. A recorded launch is never a test
    window, and neither is a Studio that started inside a launch still in progress (its provisional record is written
    before the spawn, so no lane can see a starting Studio before its record)."""
    pids, windows = L.protected()
    if p in pids:
        return False
    if L.is_test_window(p) is False:
        return False                     # its command line says edit Studio (-localPlaceFile), not a play-test window
    return not (s is not None and any(a <= s <= b for a, b in windows))


def _capture_owner_check(win, keeper, own, L=None, port=None):
    """Is the window kind screenshot captured this run's own play client? Found 2026-11-26: with about 12 Studios open
    an answer carried another Studio's window with verified true, because a capture was judged only as a picture. Proved
    from what Windows and the processes' own command lines say (TEST_IDENTITY, measured on 0.748): the window's owner,
    read before and after PrintWindow, is the client pid; the client is a StartClient of this daemon's edit Studio
    (-editpid); a StartServer of the same edit Studio seen during this run shares its -playTestSessionGuid and is its
    -parentPid; and, when this daemon's port has a Studio launched for it, the connected Studio is that one. ->
    {ok, clientPid, serverPid, session, editPid, hwnd, coveredBy, why: [every reason it is not]}."""
    why = []
    pid = (win or {}).get("pid")
    w = (win or {}).get("window")
    if not isinstance(w, dict):
        why.append("the captured window's owner was not read, so it cannot be shown to be this run's client")
    else:
        if w.get("ownerBefore") != pid or w.get("ownerAfter") != pid:
            why.append("the captured window %s belonged to pid %s before PrintWindow and %s after, not the client %s"
                       % (w.get("hwnd"), w.get("ownerBefore"), w.get("ownerAfter"), pid))
        if w.get("printed") is False:
            why.append("PrintWindow failed on the client's window, so the bitmap is not its drawing")
    seen, now = (win or {}).get("startSeen"), (win or {}).get("startAtCapture")
    if isinstance(seen, (int, float)) and isinstance(now, (int, float)) and abs(now - seen) > 1.0:
        # found 2026-11-26: the wrong captures were a start page and an empty Place1, uncropped - a new Studio of another
        # lane given the client's pid after the client had gone
        why.append("pid %s is not the process first handled as the client: it started at %.1f, the client at %.1f (the "
                   "pid was reused)" % (pid, now, seen))
    elif (win or {}).get("startSeen", 0) is not None and "startAtCapture" in (win or {}) and now is None:
        why.append("pid %s was gone at the capture (no start time could be read)" % pid)
    ident = (win or {}).get("identity")
    server, guid = None, None
    if not isinstance(ident, dict):
        why.append("the client's command line could not be read, so its test session is unknown")
    else:
        guid = ident.get("guid")
        if ident.get("task") != "StartClient":
            why.append("pid %s is not a play-test client (-task %s)" % (pid, ident.get("task")))
        if not own or ident.get("editPid") != own:
            why.append("the client's edit Studio is pid %s, not this daemon's %s" % (ident.get("editPid"), own))
        for sp, si in sorted((getattr(keeper, "idents", None) or {}).items()):
            if isinstance(si, dict) and si.get("task") == "StartServer" and guid and si.get("guid") == guid \
                    and si.get("editPid") == own:
                server = sp
                break
        if server is None:
            why.append("no play server this run saw from edit Studio %s shares the client's test session %s"
                       % (own, guid))
        elif ident.get("parentPid") is not None and ident.get("parentPid") != server:
            why.append("the client's parent is pid %s, not this run's server %s" % (ident.get("parentPid"), server))
    if L is not None and port is not None and own and hasattr(L, "ours"):
        try:
            launched = [r.get("pid") for r in L.ours(port)]
        except Exception as e:  # noqa: BLE001 - an unreadable record is said, never taken as a match
            launched, why = None, why + ["the launch records could not be read: %s" % e]
        if launched and own not in launched:
            why.append("this daemon's connected Studio is pid %s, not the Studio launched for port %s (%s)"
                       % (own, port, ", ".join(str(x) for x in launched)))
    return {"ok": not why, "clientPid": pid, "serverPid": server, "session": guid, "editPid": own,
            "hwnd": (w or {}).get("hwnd") if isinstance(w, dict) else None,
            "coveredBy": (w or {}).get("coveredBy") if isinstance(w, dict) else None, "why": why}


def _own_studio_pid():
    """The pid of the edit Studio whose plugin is connected to THIS daemon (the daemon reads it from the plugin's TCP
    connection at hello), or None when it is not known."""
    try:
        pid = ((_http("GET", "/status")[1] or {}).get("plugin") or {}).get("studioPid")
    except (OSError, ValueError, AttributeError):
        return None
    return int(pid) if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 else None


def _mine(L, p, s, own):
    """A test window this probe may minimize or end: a play-test process of THIS daemon's edit Studio (its `-editpid` is
    `own`, or with no such flag its process tree leads to `own`), and not any lane's recorded or starting Studio. Another lane's test windows start during the probe
    too, and once all three earlier checks passed for them (found 2026-11-19: a lane's probes cut off another lane's
    multiplayer servers); -editpid and the parent chain are what tell them apart. With `own` unknown, nothing is."""
    return bool(own) and _not_a_lane(L, p, s) and L.belongs_to(p, own)


class _WindowKeeper:
    """While a probe runs, minimize each play-test window it opens, without activating it, so a test (and a dialog inside
    one, such as a kicked client's "Disconnected") stays off the person's screen. Only a process that started during this
    probe, descends from THIS daemon's edit Studio, is not a lane's recorded or starting Studio, and whose command line
    positively says play test (-playTestSessionGuid) is touched; an unreadable command line is left alone. Minimizing
    closes nothing."""

    def __init__(self, L, before, t_start, every=0.2, *, own=None, behind_clients=False, capture_at=None):
        self.L, self.before, self.t_start, self.own, self.every = L, before, t_start, own, every
        self.behind_clients, self.lowered = behind_clients, set()
        # a capture run: capture_at seconds after the client window is first seen, its pixels are read (PrintWindow)
        self.capture_at, self.seen, self.shot = capture_at, {}, None
        self.states = {}                 # client pid -> window-state counts (_sample)
        self.idents = {}                 # pid -> TEST_IDENTITY of every test process of this run (server and clients)
        self.starts = {}                 # pid -> its start time when first handled, to catch a reused pid at capture
        self.pids, self.count, self.errors = set(), 0, []
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                for p, s in self.L.running():
                    if p in self.before or (s is not None and s < self.t_start - 2) or not _mine(self.L, p, s, self.own):
                        continue
                    if self.L.is_test_window(p) is not True:
                        continue
                    if p not in self.idents and hasattr(self.L, "TEST_IDENTITY"):
                        self.idents[p] = self.L.TEST_IDENTITY(p)
                    self.starts.setdefault(p, s)
                    # a launch module without is_test_client (an older driver's fake) has no clients to tell apart
                    client = bool(getattr(self.L, "is_test_client", lambda _p: False)(p))
                    if self.behind_clients and client:
                        # a minimized window captures blank (RB1037) and renders at a capped rate (RB2105): behind the
                        # person's windows, without focus
                        if p not in self.lowered and self.L.LOWER_WINDOWS(p):
                            self.lowered.add(p)
                            self.seen.setdefault(p, time.time())
                        if self.capture_at is not None and self.shot is None and p in self.seen                                 and time.time() - self.seen[p] >= self.capture_at:
                            t0 = time.time()
                            # whose window it was is read with the pixels (found 2026-11-26: another Studio's window
                            # came back verified); a launch module without the checked route leaves it unproved
                            checked = getattr(self.L, "CAPTURE_WINDOW_CHECKED", None)
                            info = checked(p) if checked else None
                            got = (info or {}).get("image") if checked else self.L.CAPTURE_WINDOW(p)
                            rects = None
                            if hasattr(self.L, "CHILD_RECTS"):
                                try:
                                    rects = self.L.CHILD_RECTS(p)
                                except Exception as e:  # noqa: BLE001 - no boxes means no crop, and that is said
                                    rects = {"error": str(e)}
                            self.shot = {"pid": p, "seconds": round(time.time() - t0, 2), "at": t0,
                                         "atSeconds": round(time.time() - self.seen[p], 1), "image": got,
                                         "rects": rects,
                                         "window": {k: v for k, v in info.items() if k != "image"}
                                         if isinstance(info, dict) else None,
                                         "identity": (self.L.TEST_IDENTITY(p) if hasattr(self.L, "TEST_IDENTITY")
                                                      else None),
                                         "startSeen": self.starts.get(p),
                                         "startAtCapture": (self.L.PROC_START(p) if hasattr(self.L, "PROC_START")
                                                            else None)}
                        self._sample(p)
                        continue
                    n = self.L.MINIMIZE_WINDOWS(p)
                    if n:
                        self.pids.add(p)
                        self.count += n
                    if client and p in self.pids:
                        self._sample(p)
            except Exception as e:  # noqa: BLE001 - a failed minimize never ends the probe; it is reported
                if len(self.errors) < 5:
                    self.errors.append(str(e))
            self._stop.wait(self.every)

    def _sample(self, p):
        """One reading of a handled client window's state (WINDOW_STATE): how often it was minimized, in front, or
        neither (behind the person's windows), so frame times say which window state they ran in."""
        st = self.L.WINDOW_STATE(p) if hasattr(self.L, "WINDOW_STATE") else None
        if not isinstance(st, dict) or not st.get("windows"):
            return
        row = self.states.setdefault(p, {"samples": 0, "minimized": 0, "foreground": 0, "behind": 0})
        row["samples"] += 1
        # one state per reading: a minimized window does not draw, whether or not it is the foreground window
        row["minimized" if st.get("iconic") else "foreground" if st.get("foreground") else "behind"] += 1

    def stop(self):
        self._stop.set()
        self._th.join(5)

    def client_states(self):
        """{pid: {"state": behind | minimized | foreground | mixed, samples, minimized, foreground, behind}} of every
        client window handled. Each reading is one state (minimized, else the foreground window, else behind the
        person's windows); the window's state is the one that held for at least 85% of its readings, else mixed. A new
        test window is briefly the foreground window as Studio opens it (live: 7 of 76 and 8 of 71 readings), before it is lowered
        or minimized."""
        out = {}
        for p, row in self.states.items():
            n = row["samples"]
            top = max(("behind", "minimized", "foreground"), key=lambda k: row.get(k, 0))
            out[p] = dict(row, state=top if n and row.get(top, 0) >= 0.85 * n else "mixed")
        return out

    def report(self):
        out = {"minimized": True, "pids": sorted(self.pids), "windows": self.count, "errors": self.errors,
               "ownStudio": self.own}
        if self.behind_clients:
            out["loweredClients"] = sorted(self.lowered)
        if self.states:
            out["clientStates"] = {str(p): v for p, v in sorted(self.client_states().items())}
        if not self.own:
            out["warning"] = "this daemon's Studio pid is unknown, so no test window could be told apart from another lane's; none was minimized"
        return out


def _end_test_processes(L, before, t_start, wait=30.0, *, own=None):
    """A play or multiplayer test opens Studio processes of its own (a server and one per client). Studio ends them after
    EndTest; wait for that, then end only the ones that started during this probe, descend from THIS daemon's edit Studio
    (`own`) and are still there, by pid - never an edit Studio, and never another lane's test windows. With `own`
    unknown nothing is ended, and that is said."""
    if not own:
        return {"closedByStudioSeconds": None, "endedByPid": [], "editStudioUntouched": sorted(before), "ownStudio": None,
                "warning": "this daemon's Studio pid is unknown, so no leftover test window could be told apart from "
                           "another lane's; none was ended"}
    deadline = time.time() + wait
    t0 = time.time()
    leftover = []

    def not_a_lane(p, s):
        return _mine(L, p, s, own)

    while True:
        leftover = [(p, s) for p, s in L.running() if p not in before and (s is None or s >= t_start - 2) and not_a_lane(p, s)]
        if not leftover or time.time() >= deadline:
            break
        time.sleep(0.5)
    # Which leftovers are ours is decided once, on this snapshot: ending a test server first would orphan its clients,
    # whose parent chain then no longer leads to `own`. Newest first, so clients go before their server.
    ended, not_closed = [], []
    for p, s in sorted(leftover, key=lambda ps: -(ps[1] or 0)):
        if not _not_a_lane(L, p, s):
            continue
        if L.TERMINATE(p):
            ended.append(p)
        else:
            not_closed.append(p)
    out = {"closedByStudioSeconds": round(time.time() - t0, 1) if not leftover else None,
           "endedByPid": ended, "editStudioUntouched": sorted(before), "ownStudio": own}
    if not_closed:
        out["notClosed"] = not_closed
        out["warning"] = "test windows %s could not be ended and may still be on screen" % not_closed
    return out


# Two stack-frame spellings Roblox has used: "Script 'Workspace.X', Line 2" and "Workspace.X:2". Which one 0.748 writes is
# measured by the live errors probe; both are read, the first frame wins.
_STACK_LINES = (re.compile(r"([A-Za-z_][\w.]*), line (\d+)"), re.compile(r"Script '([^']+)', Line (\d+)"),
                re.compile(r"([A-Za-z_][\w.]*):(\d+)"))


def _where(err):
    """The script path and line of a ScriptContext error, from Studio's own stack text (then the message)."""
    for text in (err.get("stackTrace") or "", err.get("message") or ""):
        for rx in _STACK_LINES:
            m = rx.search(text)
            if m:
                return {"at": {"script": m.group(1), "line": int(m.group(2))}}
    return {"at": None}


_SCAFFOLD_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,40}$")


def rbx_script_scaffold(name: str, with_function: bool = False, kind: str = "pair", prompt: str = "favorite") -> dict:
    "Roblox Studio: scripts from MifBridge's fixed templates, one undo step, read back by path and sha256. kind pair (default): a server and client script pair joined by a RemoteEvent (and a RemoteFunction with with_function): ReplicatedStorage.<name>Event, ServerScriptService.<name>Server (RunContext Server) and StarterPlayer.StarterPlayerScripts.<name>Client (a LocalScript); the client fires hello, the server stores it on the remote's LastPayload attribute and acks; rbx_play_probe wait on '@LastPayload' proves the round trip. kind save: ServerScriptService.<name>Store, a ModuleScript of session-locked player saves (each load a session with its own id in the stored Lock; a save lands only over its own lock inside UpdateAsync and answers false otherwise; one write at a time per key, in order; a join waits LOCK_WAIT for another server's lock; a leave closes the profile at once, runs once, and keeps the profile on the server when every save fails, so a rejoin there takes it; flush for BindToClose waits for every save in flight; processReceipt keeps a grant and its receipt record together and answers PurchaseGranted only after the save holding them landed, NotProcessedYet for an absent player; a rewarded video ad's reward is one more receipt there, off until AD_PRODUCT (a developer product id) is set: adAvailable checks the daily cap and the cooldown, both counted in the saved record, then the AdService availability its client read and sent (GetAdAvailabilityNowAsync answers only a client, RB2947), showAd shows the ad from the server and pays nothing, and the receipt's grant and its count land in one save), with <name>Server wiring players, BindToClose, ProcessReceipt, the autosave and ReplicatedStorage.<name>AdRemote, which <name>AdClient asks before its button shows. Its test save slot is for the creator, whose account owns every pass: Store.setTestSlot(userId, on, fresh) writes only its own choice key, read once at the next join, and that session then uses a test key of its own (a new one for each fresh start) so the real save is never read or written, while Store.ownsPass answers false; the creator of a user-owned game, or TEST_SLOT_USERS, may use it. kind filtered_text: player-written text filtered on the server before anyone sees it or it is saved - <name>Text.submit(player, text) checks it is a string of 1 to MAX characters and the cooldown, filters it with the AUTHOR's user id (TextService:FilterStringAsync) and hands GetNonChatStringForBroadcastAsync's string to Text.apply; a filter that fails or errors refuses it, never shows it raw; <name>TextServer wires ReplicatedStorage.<name>TextRemote, and <name>TextClient sends a TextBox tagged <name>Input once on Enter, never per keystroke; its SelfTest runs a stand-in filter (Studio's own returns the text unchanged, RB3083), and rbx_text_filter_audit passes it. kind action: a server-checked action - the client names its target's id over ReplicatedStorage.<name>Remote, <name>Action checks the id names a live tagged target, its Health, reach plus lag allowance from the root to the target's surface, and the cooldown, spent on a miss too; one client hold loop for mouse, E, the gamepad's R2 and tagged buttons. With Action.GRID set (a table answering solid(x, y, z) over CELL-stud cells, for a game whose blocks are parts only where they touch air), the server also walks the line from the head to just inside the target in 0.2-stud steps and refuses a swing that meets another solid cell (no line of sight), or whose grid errors (line of sight unread). kind prompt_once: a favorite or notifications prompt (prompt) shown once per player when the server calls offer: never in the first 3 minutes or while a busy attribute is set, claimed and saved before the client is told, no API call when PlaceId is 0, GetFavoriteAsync in pcall. kind economy: <name>Clock, the one reader of game time (an offset a test moves, freeze for a soak, the daily reset hour and the week's first day), <name>Economy (income per second, timed boosts, offline pay at a share of the rate up to a cap, daily and weekly resets once per crossing; a tick pays income BEFORE the boosts count down, so a boost pays its whole length) and <name>EconomyServer ticking every player; its self-test is a time-warp soak: the clock held at a Sunday 20:00 UTC and stepped 10 s at a time for 8 h on a fake player, earnings against the closed form, each reset exactly once, the countdown-first order shown one tick short, offline pay and its cap. kind scenery: a tiled-texture scenery kit with no uploads - ReplicatedStorage.<name>Textures (pixel data as hex RRGGBBAA, which a build replaces), <name>Scenery (the tag convention: <name>Tex marks a part, <name>Tex=<id> its texture, <name>Top=<id> the top face's, <name>Tile=<studs> the tile; parse, decode, the average colour and an audit that fails a tagged id with no pixel data), <name>SceneryServer painting each part its texture's average colour and warning on missing ids, and <name>SceneryClient making one EditableImage per id and a Texture (Pixelated, StudsPerTileU/V the tile, ColorMapContent from Content.fromObject, RB2111) on every face but the bottom, reporting what it made over <name>SceneryReport. kind warps: rank-gated warps between separate islands - <name>Warps (AREAS and the rank each needs, islands under Workspace.<name>Islands each with an ArrivalPad; the server checks a real destination, a built area, the rank and a cooldown, then lands the character on the pad facing the area; islandGaps measures the sky between every pair of islands from their parts' extents against MIN_GAP 300), <name>WarpServer (the menu's RemoteFunction <name>WarpRemote and a ProximityPrompt on every part tagged <name>Portal with a WarpTo attribute) and <name>WarpClient (a menu that only colours what the player may not use). kind subscription: <name>Subscription, an entitlement that is Roblox's own answer - GetUserSubscriptionStatusAsync in pcall when a player joins, on Players.UserSubscriptionStatusChanged, after a purchase prompt closes and on a timer (RECHECK, or RETRY after a failed read, which is not subscribed); held for the session only and never saved; <name>SubscriptionServer wires it and the one RemoteFunction <name>SubscriptionRemote, over which <name>SubscriptionClient asks the server to prompt (never a subscriber, never after a failed read); SUBSCRIPTION_ID is a placeholder the person fills, and nothing is created or priced. The ad and subscription templates say in their comments which Roblox shapes they assume from documentation. kind analytics: <name>Analytics, the one server module that sends the game's events through AnalyticsService - an onboarding funnel from the ONBOARDING step list (LogOnboardingFunnelStepEvent, each step once per player and only as the next one), grant and spend, the currency's one source and sink functions, which run the caller's change first and send LogEconomyEvent with the ending balance only after it landed (transaction an Enum.AnalyticsEconomyTransactionType item), event for custom events looked up in the declared EVENTS table with up to 3 field values keyed Enum.AnalyticsCustomFieldKeys, and a budget of BUDGET_BASE + BUDGET_PER_PLAYER x players calls a 60 s window whose overflow is dropped and counted - with <name>AnalyticsServer sending the first step on join; its self-test puts a recorder in place of the service, so nothing is sent to Roblox. The limits it is written to (100 steps, 10 currency types, 100 custom event names, 3 custom fields, 120 + 20 x players requests a minute) are Roblox's documented ones as read on 2026-11-22, assumed and not measured; rbx_analytics_audit checks a game's scripts against the same. save, action, prompt_once, economy, scenery, warps, subscription and analytics each write a <name>...SelfTest script that does nothing unless MifBridge's probe marks the run; prove with rbx_play_probe self_test=<name> (the answer's prove line)."
    if not isinstance(name, str) or not _SCAFFOLD_NAME.match(name):
        return {"ok": False, "ran": False, "error": "name is letters, digits and _, starting with a letter"}
    if kind != "pair":
        from mif_roblox import domain_logic as DL
        if with_function:
            return {"ok": False, "ran": False, "error": "with_function is for kind pair"}
        return DL.scaffold(kind, name, prompt)
    server = ("-- " + name + "Server: written by MifBridge's scaffold\n"
              "local RS = game:GetService(\"ReplicatedStorage\")\n"
              "local remote = RS:WaitForChild(\"" + name + "Event\")\n"
              "remote.OnServerEvent:Connect(function(player, payload)\n"
              "\tremote:SetAttribute(\"LastPayload\", payload)\n"
              "\tremote:FireClient(player, \"ack:\" .. tostring(payload))\n"
              "end)\n")
    if with_function:
        server += ("local fn = RS:WaitForChild(\"" + name + "Function\")\n"
                   "fn.OnServerInvoke = function(player, x)\n\treturn x\nend\n")
    client = ("-- " + name + "Client: written by MifBridge's scaffold\n"
              "local RS = game:GetService(\"ReplicatedStorage\")\n"
              "local remote = RS:WaitForChild(\"" + name + "Event\")\n"
              "remote.OnClientEvent:Connect(function(msg)\n"
              "\tprint(\"" + name + " client got \" .. tostring(msg))\n"
              "end)\n"
              "remote:FireServer(\"hello\")\n")
    ops = [{"tool": "rbx_create", "args": {"class_name": "RemoteEvent", "parent": "ReplicatedStorage", "name": name + "Event"}},
           {"tool": "rbx_create", "args": {"class_name": "Script", "parent": "ServerScriptService", "name": name + "Server",
                                           "props": {"RunContext": "Server"}}},
           {"tool": "rbx_create", "args": {"class_name": "LocalScript", "parent": "StarterPlayer.StarterPlayerScripts",
                                           "name": name + "Client"}},
           {"tool": "rbx_script_write", "args": {"target": "ServerScriptService." + name + "Server", "source": server}},
           {"tool": "rbx_script_write", "args": {"target": "StarterPlayer.StarterPlayerScripts." + name + "Client",
                                                 "source": client}}]
    if with_function:
        ops.insert(1, {"tool": "rbx_create", "args": {"class_name": "RemoteFunction", "parent": "ReplicatedStorage",
                                                      "name": name + "Function"}})
    from mif_roblox import domain_logic as DL
    clash = DL.scaffold_clash(sys.modules[__name__], ops)
    if clash:
        return clash
    r = rbx_batch(ops)
    return dict(r, remote="ReplicatedStorage.%sEvent" % name, server="ServerScriptService.%sServer" % name,
                client="StarterPlayer.StarterPlayerScripts.%sClient" % name,
                prove="rbx_play_probe('wait', 'ReplicatedStorage.%sEvent', property='@LastPayload', op='eq', value='hello')" % name)


# A gated tool lists what it proved (COVERAGE_PLAN 7.2 #6, spec "gated tools list proved"): the RunService call each
# run_mode action makes, proven when RunState read before differs and read after is the state the call sets.
RUN_MODE_PROVES = {"run": ("Run", "Running"), "pause": ("Pause", "Paused"), "stop": ("Stop", "Stopped")}


def _run_mode_proved(cat, action, before, after):
    """rbx_run_mode's proved list: the reads it answered (RunState, IsRunning, IsRunMode), and the action's own RunService
    call as a write only when the state changed to the one it sets - a stop of a stopped session proves nothing."""
    names = [n for n, k in (("RunState", "runState"), ("IsRunning", "isRunning"), ("IsRunMode", "isRunMode"))
             if after.get(k) is not None]
    out = _proved(cat, "RunService", names[:1], "read", "run_mode") + _proved(cat, "RunService", names[1:], "call", "run_mode")
    call, state = RUN_MODE_PROVES.get(action, (None, None))
    if call and isinstance(before, dict) and before.get("runState") not in (None, state) and after.get("runState") == state:
        out += _proved(cat, "RunService", [call], "write", "run_mode")
    return out


# The roots whose attributes a Run-mode run is read around: a server Script's SetAttribute during the run stays in the
# edit place after Stop (RB3079: a game's self-test left its switches on Workspace that way, and the place went live
# with them). An instance's own attributes only; descendants are not walked.
RUN_ATTR_ROOTS = ("Workspace", "Lighting", "ReplicatedStorage", "ServerStorage", "ServerScriptService", "StarterGui",
                  "StarterPack", "StarterPlayer", "SoundService", "Teams")


# The roots whose top-level children a run is counted around (found 2026-11-26: a place shipped with 46 stacked copies
# of three folders its server scripts make at start, one per rbx_run_mode run, since Stop keeps what a run made).
RUN_CHILD_ROOTS = ("Workspace", "ReplicatedStorage", "ServerStorage")


def _read_root_children():
    """({root: {child name: how many}}, [roots that could not be listed])."""
    out, bad = {}, []
    for root in RUN_CHILD_ROOTS:
        t = _rbx("tree", root=[root], depth=1, limit=5000)
        node = (t or {}).get("root") if (t or {}).get("ok") else None
        kids = (node or {}).get("children")
        if not isinstance(kids, list) or (node or {}).get("truncated"):
            bad.append(root)
            continue
        counts = {}
        for k in kids:
            if isinstance(k, dict) and isinstance(k.get("name"), str):
                counts[k["name"]] = counts.get(k["name"], 0) + 1
        out[root] = counts
    return out, bad


def _run_attr_file():
    return os.path.join(D.home(), "places", str(PORT), "run_attributes.json")


def _read_root_attrs():
    """({root: {name: value}}, [roots that could not be read])."""
    out, bad = {}, []
    for root in RUN_ATTR_ROOTS:
        r = rbx_call(root, "GetAttributes", [])
        if r.get("ok") and isinstance(r.get("value"), dict):
            out[root] = r["value"]
        elif r.get("ok") and r.get("value") in (None, []):
            out[root] = {}
        else:
            bad.append(root)
    return out, bad


def _run_attr_snapshot():
    """At run: the roots' attributes, kept on disk under this port so a stop from another process puts them back."""
    snap, bad = _read_root_attrs()
    kids, kbad = _read_root_children()
    path = _run_attr_file()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"at": time.time(), "roots": snap, "unread": bad, "children": kids, "childrenUnread": kbad}, fh)
    except OSError as exc:
        return {"snapshot": False, "error": "the attribute snapshot was not written: %s" % exc}
    return {"snapshot": True, "roots": len(snap), "unread": bad}


def _run_added(before, remove):
    """The top-level children the run added under RUN_CHILD_ROOTS (a name counted more after than before), and with
    remove the added ones taken out - the last of each name in GetChildren order, where a run's new instances go - each
    read back. -> {added: [{under, name, count}], removed?, stillThere?, unread}."""
    now, bad = _read_root_children()
    added = []
    for root, was in (before or {}).items():
        cur = now.get(root)
        if cur is None:
            continue
        for name, n in sorted(cur.items()):
            extra = n - was.get(name, 0)
            if extra > 0:
                added.append({"under": root, "name": name, "count": extra, "before": was.get(name, 0)})
    out = {"added": added, "unread": bad}
    if not added:
        return out
    if not remove:
        out["note"] = ("instances the run's scripts made are left in the edit place, and a save keeps them: pass "
                       "remove_added true to take them out, or remove them by name")
        return out
    removed, still = [], []
    for a in added:
        total = a["before"] + a["count"]
        for k in range(total, a["before"], -1):
            d = rbx_destroy("%s.%s[%d]" % (a["under"], a["name"], k) if total > 1 else "%s.%s" % (a["under"], a["name"]))
            (removed if d.get("ok") else still).append({"under": a["under"], "name": a["name"], "index": k,
                                                        "error": None if d.get("ok") else d.get("error")})
    back, _b = _read_root_children()
    left = [a for a in added if (back.get(a["under"]) or {}).get(a["name"], 0) != a["before"]]
    out.update(removed=len(removed), stillThere=left + [s for s in still if s.get("error")])
    return out


def _run_attr_restore(restore, remove_added=False):
    """At stop: every attribute of the roots that the run changed, added or removed, put back (SetAttribute, then read
    back) unless restore is false, when they are only listed. -> the answer's attributes block."""
    path = _run_attr_file()
    try:
        with open(path, encoding="utf-8") as fh:
            saved = json.load(fh)
        snap = saved.get("roots") or {}
        kids_before = saved.get("children")
    except (OSError, ValueError):
        return {"snapshot": False, "note": "no attribute snapshot from this port's run (run was not started through "
                                           "rbx_run_mode, or it was started before plugin-side restore existed)"}
    now, bad = _read_root_attrs()
    changed = []
    for root, was in snap.items():
        cur = now.get(root)
        if cur is None:
            continue
        for k in sorted(set(was) | set(cur)):
            if json.dumps(was.get(k), sort_keys=True, default=str) != json.dumps(cur.get(k), sort_keys=True, default=str):
                changed.append({"on": root, "attribute": k, "was": was.get(k), "after": cur.get(k)})
    out = {"snapshot": True, "changed": changed, "unread": bad}
    if not restore:
        out["left"] = changed
        out["note"] = "restore false: the run's attribute changes were left in the edit place, and a save keeps them"
    else:
        for c in changed:
            w = rbx_call(c["on"], "SetAttribute", [c["attribute"], c["was"]])
            c["putBack"] = bool(w.get("ok"))
        back, _bad = _read_root_attrs()
        still = [c for c in changed if json.dumps((back.get(c["on"]) or {}).get(c["attribute"]), sort_keys=True,
                                                  default=str) != json.dumps(c["was"], sort_keys=True, default=str)]
        out.update(restored=len(changed) - len(still), stillDiffer=still)
    if isinstance(kids_before, dict):
        out["instances"] = _run_added(kids_before, remove_added)
    try:
        os.remove(path)
    except OSError:
        pass
    return out


@_reports
def rbx_run_mode(action: str, restore: bool = True, remove_added: bool = False) -> dict:
    "Roblox Studio: Run mode - physics in the edit DataModel with no players: run, pause, stop or state, answering RunService's RunState (before, read first, and after). Studio's Stop does not put the place back (measured), so run snapshots every unanchored part and stop restores them unless restore is false, saying how many. The server's Scripts run in the edit DataModel too, and an attribute one sets stays after Stop (RB3079: a self-test's switches were saved and went live that way), so run also reads the attributes of Workspace and the main services (RUN_ATTR_ROOTS: their own, not their descendants'), and stop puts back every one the run changed, added or removed and reads each back (attributes: changed, restored, stillDiffer); with restore false they are listed under attributes.left and kept. Instances the run's Scripts made stay in the edit place, and a save keeps them (a place shipped 46 stacked copies of folders its server makes at start, one per run): run counts the top-level children of Workspace, ReplicatedStorage and ServerStorage by name, and stop lists each name counted more under attributes.instances.added; remove_added true takes those out (the last of each name in GetChildren order) and reads the counts back. ANYTHING A RUN ADDS STAYS IN THE EDIT PLACE UNLESS REMOVED: by default it is left (a probe may read its results from what the run made), and stop then answers a top-level warning naming the count and the remove_added=True call, so ok true alone does not hide it. proved lists RunService's Run, Pause or Stop when RunState changed to the state it sets. Allowed in scratch mode only in MifBridge's own scratch place (or in full mode)."
    if action not in ("run", "pause", "stop", "state"):
        return {"ok": False, "ran": False, "error": "action is run, pause, stop or state"}
    if not isinstance(restore, bool) or not isinstance(remove_added, bool):
        return {"ok": False, "ran": False, "error": "restore and remove_added are true or false"}
    before = _rbx("run_mode", action="state", restore=restore) if action != "state" else None
    running = before is not None and before.get("ok") and before.get("runState") in ("Running", "Paused")
    snap = _run_attr_snapshot() if action == "run" and not running else None
    r = _rbx("run_mode", action=action, restore=restore)
    if snap is not None:
        r["attributes"] = snap
    if action == "stop" and r.get("ok"):
        r["attributes"] = _run_attr_restore(restore, remove_added)
        added = ((r["attributes"].get("instances") or {}).get("added") or []) if not remove_added else []
        if added:
            # top level, so a caller that only reads ok still sees it (the coordinator's ask, 2026-11-26)
            n = sum(a.get("count", 0) for a in added)
            r["warning"] = ("this run added %d top-level instance(s) that stay in the edit place, and a save keeps them: "
                            "%s. Call rbx_run_mode('stop', remove_added=True) on the next run to take them out, or "
                            "remove them by name" % (n, ", ".join("%s.%s x%d" % (a["under"], a["name"], a["count"])
                                                                   for a in added[:6])))
        if r["attributes"].get("stillDiffer"):
            r.update(ok=False, error="the run's attribute changes were not all put back: %s" % ", ".join(
                "%s@%s" % (c["on"], c["attribute"]) for c in r["attributes"]["stillDiffer"][:5]))
    if r.get("ok"):
        cat, _err = _catalog()
        if before is not None and before.get("ok"):
            r["before"] = before.get("runState")
        if cat is not None:
            r["proved"] = _run_mode_proved(cat, action, before, r)
    return r


def _coverage_proved(cat, r, target_path):
    """rbx_script_coverage's proved list (Edit route): GetCoverageStats answered, and EnableCoverage as a write when the
    stats hold the target with at least one line hit. Coverage data for a script exists only after EnableCoverage on
    it, so the hits are its effect read back; a run whose target shows no hit proves only the stats read."""
    out = _proved(cat, "ScriptContext", ["GetCoverageStats"], "call", "coverage_run")
    if any(isinstance(c, dict) and c.get("script") == target_path and c.get("hitPairs") for c in r.get("coverage") or []):
        out += _proved(cat, "ScriptContext", ["EnableCoverage"], "write", "coverage_run")
    return out


@_reports
def rbx_script_coverage(target: Any, seconds: float = 2, play: bool = False, restart: Any = None) -> dict:
    "Roblox Studio: line coverage of one script. Default: a Run mode session - ScriptContext:EnableCoverage on the target, Run for seconds, GetCoverageStats, Stop, unanchored parts put back, and the attributes of Workspace and the main services read around the run with every change put back and read back (attributes; RB3079). play true (plugin 0.15.0): inside a play test, where the MifBridge plugin in the play server enables coverage, reads the hits, restarts the script (or restart, the Script that requires a ModuleScript target) with Enabled false then true so it compiles with coverage on, waits seconds, and reads the hits again: both readings come back (before, after) with whether the restart happened. The plugin enables coverage itself because a play server script may not."
    parts, err = _target(target)
    if err:
        return err
    if not isinstance(seconds, (int, float)) or not 0 < seconds <= 60:
        return {"ok": False, "ran": False, "error": "seconds is above 0 and at most 60"}
    if play:
        r = rbx_play_probe("coverage", target=parts, seconds=seconds, timeout=seconds + 60, target2=restart)
        cov = r.get("coverage") if isinstance(r.get("coverage"), dict) else None
        if r.get("ok") and cov is None:
            return dict(r, ok=False, error=r.get("coverageError") or "the play server answered no coverage")
        if cov is not None:
            for k in ("before", "after"):
                cov[k] = [dict(c, script=V.join_path(c.get("script")) if isinstance(c.get("script"), list) else c.get("script"),
                               hits=V.decode(c.get("hits"))) for c in cov.get(k) or []]
            lines = sum(len(c.get("hitPairs") or []) for c in cov.get("after") or [])
            r["linesHit"] = lines
        return r
    # coverage_run is Run then Stop in the edit DataModel: the attributes its run's Scripts set stay (RB3079), so they
    # are read around it and put back like rbx_run_mode's
    snap = _run_attr_snapshot()
    r = _rbx("coverage_run", target=parts, seconds=seconds, timeout=seconds + 30)
    if snap.get("snapshot"):
        r["attributes"] = _run_attr_restore(True)
        if r["attributes"].get("stillDiffer"):
            r.update(ok=False, error="the run's attribute changes were not all put back: %s" % ", ".join(
                "%s@%s" % (c["on"], c["attribute"]) for c in r["attributes"]["stillDiffer"][:5]))
    if r.get("ok"):
        r["coverage"] = [dict(c, script=V.join_path(c.get("script")) if isinstance(c.get("script"), list) else c.get("script"),
                              hits=V.decode(c.get("hits"))) for c in r.get("coverage") or []]
        cat, _err = _catalog()
        if cat is not None:
            r["proved"] = _coverage_proved(cat, r, V.join_path(parts))
    return r


def rbx_test_run(seconds: float = 1, timeout: float = 120) -> dict:
    "Roblox Studio: run the place's own tests through TestService:RunTestsAsync inside a play session (it refuses edit mode), returning total, passed, failed, skipped, status, summary and the tests' names."
    return rbx_play_probe("tests", seconds=seconds, timeout=timeout)


# What the device simulator reaches (found 2026-11-23: a screenshot run set iphone_14 landscape, read back 844 x 390, and
# rbx_play_probe's client still rendered at its test window's 2140 x 1161). A solo play test's client is the simulated
# device (RB1800); a multiplayer test's client is its own window (RB1444, RB1800), and ExecuteMultiplayerTestAsync's only
# parameters are numPlayers and args (the API dump), so no plugin call gives that client a device.
DEVICE_SIM_CLIENT_VIEW = {
    "playProbeClientSized": False, "soloPlayClientSized": True,
    "clientViewNote": "the simulator sizes Studio's viewport and a solo play test's client (RB1800), not a multiplayer "
                      "test's client: rbx_play_probe's kind screenshot and frame_times run a multiplayer test, whose "
                      "client renders at its own window's size (RB1444, RB1800), and ExecuteMultiplayerTestAsync takes "
                      "no device"}


def _device_sim_warning():
    """deviceSimWarning for a multiplayer play run started while the simulator has a device: its client is not that
    device (RB1800). None when the simulator is off or its state cannot be read."""
    g = rbx_call("StudioDeviceSimulatorService", "GetDeviceAsync", timeout=30)
    dev = g.get("value") if g.get("ok") else None
    if not isinstance(dev, str) or not dev or dev == "default":
        return None
    return ("the device simulator is set to %s, but this run's client was a multiplayer test window at its own size, not "
            "that device (RB1444, RB1800): %s" % (dev, DEVICE_SIM_CLIENT_VIEW["clientViewNote"]))


def rbx_device_sim(action: str, device: str = None, width: int = None, height: int = None, verify: bool = True) -> dict:
    "Roblox Studio: the device simulator - list devices, set a device, set a resolution (a device must be set first: Studio refuses otherwise), or stop (restore). It changes Studio's own viewport, and a SOLO play test started under it runs its client as that device (RB1800); a multiplayer test's client is not sized: it is its own test window at that window's size (RB1444, RB1800; StudioTestService:ExecuteMultiplayerTestAsync takes only a player count and test args, so a plugin cannot give its client a device). rbx_play_probe always runs a multiplayer test for kind screenshot and frame_times, so its client view is never the simulated device: set and resolution answer playProbeClientSized false and soloPlayClientSized true with clientViewNote, and rbx_play_probe adds deviceSimWarning when it runs one under a set device. With verify, each is read back through the simulator's own getters: GetDeviceAsync for set and stop (stop reads 'default'), GetResolutionAsync for a resolution. Orientation, pixel density and scaling mode are rbx_device_state's."
    if action not in ("list", "set", "resolution", "stop"):
        return {"ok": False, "ran": False, "error": "action is list, set, resolution or stop"}
    svc = "StudioDeviceSimulatorService"
    if action == "list":
        return rbx_call(svc, "GetDeviceListAsync", timeout=30)
    if action == "stop":
        r = rbx_call(svc, "StopSimulationAsync", timeout=30)
    elif action == "set":
        if not isinstance(device, str) or not device:
            return {"ok": False, "ran": False, "error": "set needs device, an id from list"}
        r = rbx_call(svc, "SetDeviceAsync", [device], timeout=30)
    else:
        if not all(isinstance(x, int) and 100 <= x <= 8192 for x in (width, height)):
            return {"ok": False, "ran": False, "error": "resolution needs width and height, 100-8192"}
        r = rbx_call(svc, "SetResolutionAsync", [width, height], timeout=30)
    if not r.get("ok"):
        return r
    if action in ("set", "resolution"):
        r = dict(r, **DEVICE_SIM_CLIENT_VIEW)
    if not verify:
        return r
    if action == "resolution":
        g = rbx_call(svc, "GetResolutionAsync", timeout=30)
        got = g.get("value") if g.get("ok") else None
        good = isinstance(got, dict) and got.get("x") == width and got.get("y") == height
        want = {"x": width, "y": height}
    else:
        g = rbx_call(svc, "GetDeviceAsync", timeout=30)
        got = g.get("value") if g.get("ok") else None
        want = device if action == "set" else "default"
        good = got == want
    out = dict(r, verified=good, readBack={"getter": g.get("member"), "value": got})
    out.pop("note", None)                    # rbx_call's "no paired getter" no longer holds: this is the getter
    if not good:
        out.update(ok=False, error="%s reads %r, not %r%s" % (g.get("member") or "the getter", got, want,
                                                                "" if g.get("ok") else ": %s" % g.get("error")))
    return out


MAX_TRIANGLES = 20000      # Roblox's per-mesh limit for an imported mesh


@_reports
def rbx_mesh_preview(vertices: list, triangles: list, name: str = "MifMesh", parent: Any = "Workspace.MifScratch",
                     position: list = None, corner_uvs: list = None, corner_normals: list = None) -> dict:
    "Roblox Studio: show a mesh without uploading it - vertices (flat x,y,z list in studs) and triangles (flat 0-based vertex indices, 3 per face) become an EditableMesh and a MeshPart made from it (CreateMeshPartAsync with the mesh as Content), under parent. Read back: the face and vertex counts in Studio and the part's MeshSize against the vertices' own bounding box. At most 20,000 triangles, Roblox's limit for one mesh. Nothing is uploaded or saved; the mesh lives in this session."
    if not isinstance(vertices, list) or len(vertices) % 3 or not vertices or not all(V._num(x) for x in vertices):
        return {"ok": False, "ran": False, "error": "vertices is a flat list of finite numbers, x y z per vertex"}
    n = len(vertices) // 3
    if not isinstance(triangles, list) or len(triangles) % 3 or not triangles or not all(
            isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n for i in triangles):
        return {"ok": False, "ran": False, "error": "triangles is a flat list of 0-based vertex indices below %d, 3 per face" % n}
    if len(triangles) // 3 > MAX_TRIANGLES:
        return {"ok": False, "ran": False, "refusedRule": "budget",
                "error": "%d triangles is over Roblox's %d per mesh" % (len(triangles) // 3, MAX_TRIANGLES)}
    if not isinstance(name, str) or not name:
        return {"ok": False, "ran": False, "error": "name is a non-empty string"}
    pparts, err = _target(parent, "parent")
    if err:
        return err
    if position is not None and not _vec3(position):
        return {"ok": False, "ran": False, "error": "position is [x, y, z]"}
    xs, ys, zs = vertices[0::3], vertices[1::3], vertices[2::3]
    want = [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]
    # A MeshPart made from an EditableMesh keeps the vertices' offset from the part's origin (measured: a box built around
    # y=10 and placed at y=10 has its bounds centred at y=20). Send the vertices about their own centre, and place the
    # part at that centre unless a position is given - the mesh then sits where its vertices say.
    c = [(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (max(zs) + min(zs)) / 2]
    centred = [vertices[i] - c[i % 3] for i in range(len(vertices))]
    if position is None:
        position = c
    nf = len(triangles) // 3
    for nm, arr, per in (("corner_uvs", corner_uvs, 6), ("corner_normals", corner_normals, 9)):
        if arr is not None and (not isinstance(arr, list) or len(arr) != nf * per or not all(V._num(x) for x in arr)):
            return {"ok": False, "ran": False, "error": "%s is a flat list of %d numbers per triangle (%d for %d triangles)"
                    % (nm, per, nf * per, nf)}
    extra = {}
    if corner_uvs is not None:
        extra["cornerUVs"] = corner_uvs
    if corner_normals is not None:
        extra["cornerNormals"] = corner_normals
    r = _rbx("mesh_build", timeout=120, parent=pparts, name=name, vertices=centred, triangles=triangles,
             position=position, **extra)
    if not r.get("ok"):
        return r
    got = V.decode(r.get("meshSize")) or {}
    size = [got.get("x"), got.get("y"), got.get("z")]
    problems = []
    if r.get("faces") != len(triangles) // 3:
        problems.append("Studio holds %s faces, %d were sent" % (r.get("faces"), len(triangles) // 3))
    if not all(isinstance(g, (int, float)) and abs(g - w) <= max(1e-3, 1e-3 * w) for g, w in zip(size, want)):
        problems.append("MeshSize is %s, the vertices span %s" % (size, want))
    box = rbx_call(V.join_path(r.get("path")), "GetPivot")
    piv = (box.get("value") or {}).get("position") if isinstance(box.get("value"), dict) else None
    if isinstance(piv, list) and not all(abs(a - b) <= 1e-3 for a, b in zip(piv, position)):
        problems.append("the part is at %s, not %s" % (piv, position))
    out = {"ok": not problems, "verified": not problems, "path": V.join_path(r.get("path")), "faces": r.get("faces"),
           "vertices": r.get("vertices"), "meshSize": size, "expectedSize": want, "handle": r.get("handle"),
           # Whether a MeshPart made from an EditableMesh survives a place save is not measured (saves are not done
           # tonight): until it is, the answer says it may not.
           "persistent": False, "requestId": r.get("requestId")}
    if problems:
        out["error"] = "; ".join(problems)
    return out


def rbx_image_preview(file: str = None, width: int = None, height: int = None, rgba_base64: str = None,
                      verify: bool = True) -> dict:
    "Roblox Studio: an image shown without uploading - a PNG file on this machine (or raw RGBA8 base64 with width and height) written into an EditableImage the plugin holds, in row chunks of at most 1 MB, and with verify read back pixel for pixel. Returns the handle: set it as Content (e.g. rbx_set_props SurfaceAppearance.ColorMapContent {handle}). Nothing is uploaded."
    import base64 as _b64
    from mif_roblox import png as P
    if file is not None:
        try:
            with open(file, "rb") as fh:
                width, height, rgba = P.decode_rgba(fh.read())
        except (OSError, ValueError) as exc:
            return {"ok": False, "ran": False, "error": "the PNG could not be read: %s" % exc}
    else:
        if not isinstance(rgba_base64, str) or not all(isinstance(x, int) and x > 0 for x in (width, height)):
            return {"ok": False, "ran": False, "error": "give file, or rgba_base64 with width and height"}
        rgba = _b64.b64decode(rgba_base64)
        if len(rgba) != width * height * 4:
            return {"ok": False, "ran": False, "error": "%d bytes is not %d x %d x 4" % (len(rgba), width, height)}
    made = rbx_call("AssetService", "CreateEditableImage", [{"Size": {"$t": "Vector2", "v": [width, height]}}])
    handle = (made.get("value") or {}).get("handle") if isinstance(made.get("value"), dict) else None
    if not handle:
        return dict(made, ok=False, error="no EditableImage of %d x %d: %s" % (width, height, made.get("error")))
    stride = width * 4
    rows = max(1, (1 << 20) // stride)
    mismatched, chunks = [], 0
    for y in range(0, height, rows):
        h = min(rows, height - y)
        part = rgba[y * stride:(y + h) * stride]
        w = rbx_call({"handle": handle}, "WritePixelsBuffer", [[0, y], [width, h], _b64.b64encode(part).decode()])
        if not w.get("ok"):
            return dict(w, ok=False, handle=handle, error="writing rows %d-%d: %s" % (y, y + h - 1, w.get("error")))
        chunks += 1
        if verify:
            back = rbx_call({"handle": handle}, "ReadPixelsBuffer", [[0, y], [width, h]])
            got = (back.get("value") or {}).get("base64") if isinstance(back.get("value"), dict) else None
            if got is None or _b64.b64decode(got) != part:
                mismatched.append([y, y + h - 1])
    out = {"ok": not mismatched, "verified": bool(verify) and not mismatched, "handle": handle, "width": width,
           "height": height, "chunks": chunks, "bytes": len(rgba)}
    if mismatched:
        out["error"] = "rows read back different: %s" % mismatched
    return out


# Materials a MaterialVariant cannot take as BaseMaterial: each read back Plastic when set, measured on Studio 0.748 over
# all 45 Material items (RB2765 for Glass, RB2820 for the set); the other 40 read back as sent.
NO_VARIANT_BASES = ("Neon", "Glass", "ForceField", "Air", "Water")


def rbx_material_variant(name: str, base_material: str, props: dict = None, override: bool = False,
                         apply_to: list = None) -> dict:
    "Roblox Studio: a MaterialVariant in MaterialService on a base material (refused before anything is created for Neon, Glass, ForceField, Air and Water, which Studio reads back as Plastic: RB2765, RB2820 - the refusal lists the 40 bases that take a variant), with any of its properties (StudsPerTile, MaterialPattern, the four map Contents from uploaded asset uris or EditableImage handles; an rbxassetid that names a Decal - what an image upload answers - is swapped for the image inside it, since a Decal id draws nothing), optionally made the base material's override (MaterialService:SetBaseMaterialOverride) and applied to parts by name (BasePart.MaterialVariant). Read back: the variant's properties, GetMaterialVariant finding it, GetBaseMaterialOverride naming it, each part's MaterialVariant."
    if not isinstance(name, str) or not name or not isinstance(base_material, str):
        return {"ok": False, "ran": False, "error": "name and base_material (a Material item name) are required"}
    cat, err = _catalog()
    if err:
        return err
    if base_material not in (cat.enums.get("Material") or []):
        return {"ok": False, "ran": False, "refusedRule": "type", "error": "no Material %r" % base_material}
    if base_material in NO_VARIANT_BASES:
        takes = [m for m in cat.enums.get("Material") or [] if m not in NO_VARIANT_BASES]
        return {"ok": False, "ran": False, "refusedRule": "no-variant-base", "takesVariant": takes,
                "error": "a MaterialVariant cannot have BaseMaterial %s: Studio reads it back as Plastic, as it does %s "
                         "(RB2765, RB2820), and a part shows a variant only when its Material equals the variant's "
                         "BaseMaterial. Nothing was created: put the variant on a base that takes one (SmoothPlastic "
                         "carried a glass look's maps, RB2765) and give the part that Material. Bases that take a "
                         "variant: %s" % (base_material, ", ".join(m for m in NO_VARIANT_BASES if m != base_material),
                                          ", ".join(takes))}
    props, swaps, ierr = _image_props(props or {})
    if ierr:
        return {"ok": False, "ran": False, "error": ierr}
    made = rbx_create("MaterialVariant", "MaterialService", name=name,
                      props=dict(props or {}, BaseMaterial=base_material))
    if not made.get("ok"):
        return made
    out = {"variant": {k: made.get(k) for k in ("ok", "verified", "path", "after", "error")}}
    if swaps:
        out["decalSwaps"] = swaps
    problems = []
    found = rbx_call("MaterialService", "GetMaterialVariant", [base_material, name])
    if (found.get("value") or None) != "MaterialService." + name:
        problems.append("GetMaterialVariant answered %s" % found.get("value"))
    if override:
        rbx_call("MaterialService", "SetBaseMaterialOverride", [base_material, name])
        got = rbx_call("MaterialService", "GetBaseMaterialOverride", [base_material])
        out["override"] = got.get("value")
        if got.get("value") != name:
            problems.append("GetBaseMaterialOverride answered %r" % got.get("value"))
    for target in apply_to or []:
        s = rbx_set_props(target, {"Material": base_material, "MaterialVariant": name})
        out.setdefault("applied", {})[V.join_path(V.split_path(target))] = s.get("ok")
        if not s.get("ok"):
            problems.append("%s: %s" % (target, s.get("error")))
    out.update(ok=not problems, verified=not problems)
    if problems:
        out["error"] = "; ".join(problems)
    return out


def rbx_model_from_meshes(name: str, meshes: list, parent: Any = "Workspace.MifScratch") -> dict:
    "Roblox Studio: several meshes as one Model without uploading - meshes is a list of {name, vertices, triangles} with vertices in world space (studs, Roblox axes; Blender's X, Z, -Y). Each becomes an EditableMesh MeshPart placed at its own bounding-box centre under a Model whose WorldPivot is the meshes' combined centre. Read back: every part's faces and MeshSize, and Model:GetBoundingBox against the meshes' combined span within 0.01 stud."
    if not isinstance(meshes, list) or not meshes:
        return {"ok": False, "ran": False, "error": "meshes is a list of {name, vertices, triangles}"}
    pparts, err = _target(parent, "parent")
    if err:
        return err
    allx, ally, allz = [], [], []
    for m in meshes:
        if not isinstance(m, dict) or not isinstance(m.get("vertices"), list) or not m["vertices"]:
            return {"ok": False, "ran": False, "error": "every mesh needs name, vertices and triangles"}
        allx += m["vertices"][0::3]
        ally += m["vertices"][1::3]
        allz += m["vertices"][2::3]
    span = [max(allx) - min(allx), max(ally) - min(ally), max(allz) - min(allz)]
    centre = [(max(allx) + min(allx)) / 2, (max(ally) + min(ally)) / 2, (max(allz) + min(allz)) / 2]
    model = rbx_create("Model", pparts, name=name)
    if not model.get("ok"):
        return model
    mpath = model["path"]
    parts, problems = [], []
    for m in meshes:
        v = m["vertices"]
        xs, ys, zs = v[0::3], v[1::3], v[2::3]
        c = [(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (max(zs) + min(zs)) / 2]
        r = rbx_mesh_preview(v, m.get("triangles") or [], name=m.get("name") or "Mesh", parent=mpath, position=c)
        parts.append({k: r.get(k) for k in ("path", "faces", "meshSize", "ok", "error")})
        if not r.get("ok"):
            problems.append("%s: %s" % (m.get("name"), r.get("error")))
    piv = rbx_set_props(mpath, {"WorldPivot": {"position": centre}})
    if not piv.get("ok"):
        problems.append("WorldPivot: %s" % piv.get("error"))
    box = rbx_call(mpath, "GetBoundingBox")
    size = (box.get("values") or [None, {}])[1] or {}
    got = [size.get("x"), size.get("y"), size.get("z")]
    if not all(isinstance(g, (int, float)) and abs(g - w) <= 0.01 for g, w in zip(got, span)):
        problems.append("Model:GetBoundingBox is %s, the meshes span %s" % (got, span))
    out = {"ok": not problems, "verified": not problems, "model": mpath, "parts": parts, "boundingBox": got,
           "expectedSpan": span, "pivot": centre, "persistent": False}
    if problems:
        out["error"] = "; ".join(problems)
    return out


def rbx_debugger() -> dict:
    "Roblox Studio: whether this plugin can reach the script debugger at all (DebuggerManager is not a DataModel service), and if so whether debugging is enabled and how many debuggers exist. Read only; drives nothing."
    return _rbx("debugger")


def rbx_release_handle(handle: str) -> dict:
    "Roblox Studio: release an object handle (an EditableImage or EditableMesh the plugin holds); a later call using it is refused."
    if not isinstance(handle, str) or not handle:
        return {"ok": False, "ran": False, "error": "handle is the id a call returned"}
    # A plugin before 0.9.3 Destroys whatever the handle names, even an object since put into the place. Ask first.
    where = _rbx("resolve", target={"handle": handle})
    if where.get("ok") and where.get("path"):
        plugin = str(((rbx_status().get("plugin") or {}).get("plugin")) or "0")
        if tuple(int(x) for x in re.findall(r"\d+", plugin)[:3]) < (0, 9, 3):
            return {"ok": False, "ran": False, "refusedRule": "in-place", "path": V.join_path(where["path"]),
                    "error": "the object is in the place now (%s) and plugin %s would Destroy it on release; keep the "
                             "handle (plugin 0.9.3 forgets it instead)" % (V.join_path(where["path"]), plugin)}
    return _rbx("release_handle", handle=handle)


def rbx_console(limit: int = 100) -> dict:
    "Roblox Studio: the newest Output lines (LogService history) with their message type. Read only."
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 2000:
        return {"ok": False, "ran": False, "error": "limit is 1-2000"}
    return _rbx("console", limit=limit)


def _plugin_mode():
    st = rbx_status()
    return ((st.get("plugin") or {}).get("mode")), st


EMISSIVE_JPG_QUALITY = 97     # the quality the masks of RB2768 were saved at; a PNG can stay unusable as a map (RB1652)


def _emissive_check(file):
    """(scene or None, {materials, textures, warning} or None) for an .fbx whose materials feed a texture into an
    emissive property; (None, None) for any other file or one the reader cannot read."""
    if not (isinstance(file, str) and file.lower().endswith(".fbx") and os.path.isfile(file)):
        return None, None
    from mif_roblox import fbx_mesh as FM  # noqa: PLC0415 - the mesh pipeline's FBX reader
    try:
        sc = FM.load(file)
        rows = FM.emissive_textures(sc)
    except Exception:  # noqa: BLE001 - a file this reader cannot parse is sent unchecked (the upload says so), never a raise
        return None, None
    if not rows:
        return sc, None
    mats = sorted({r["material"] for r in rows})
    return sc, {"materials": mats, "textures": rows,
                "warning": "%d material(s) carry an emissive texture (%s): an FBX upload brings each back with "
                           "SurfaceAppearance.EmissiveMaskContent rbxassetid://6, an asset that does not exist, and play "
                           "clients draw those meshes untextured grey (RB2766). emissive_masks true also uploads each mask "
                           "as its own JPG image and answers the image id to set as EmissiveMaskContent (RB2768); "
                           "without it, clear the mask after placing (rbx_asset_place clear_broken_maps true)"
                           % (len(mats), ", ".join(mats))}


def _emissive_jpgs(sc, file, emis, display_name, given=None):
    """Write each emissive material's mask as a JPG beside the FBX (<stem>_<material>_emissive_mask.jpg, grey RGB,
    quality 97): from given[material] (a mask file the person's bake wrote) when named, else the FBX's own emissive
    texture. [{material, texture, from, jpg, displayName} or {..., error}]. Writes local files only."""
    from mif_roblox import fbx_mesh as FM  # noqa: PLC0415
    import io  # noqa: PLC0415
    stem = os.path.splitext(file)[0]
    out, done = [], set()
    for row in emis["textures"]:
        if row["material"] in done:
            continue
        done.add(row["material"])
        item = {"material": row["material"], "texture": row["texture"], "image": row["image"]}
        if given and row["material"] in given:
            item.update(image=given[row["material"]], source="given")
            with open(given[row["material"]], "rb") as fh:
                data = fh.read()
        else:
            item["source"] = "fbx"
            data = FM.emissive_image_bytes(sc, row)
        if data is None:
            out.append(dict(item, error="the emissive texture is not embedded and %s is not on this machine" % row["image"]))
            continue
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", row["material"]).strip("_") or "material"
        jpg = "%s_%s_emissive_mask.jpg" % (stem, safe)
        try:
            from PIL import Image  # noqa: PLC0415 - Pillow writes the JPG
            Image.open(io.BytesIO(data)).convert("L").convert("RGB").save(jpg, "JPEG", quality=EMISSIVE_JPG_QUALITY)
        except ImportError:
            out.append(dict(item, error="writing the JPG needs Pillow (pip install pillow)"))
            continue
        except (OSError, ValueError) as exc:
            out.append(dict(item, error="the emissive texture did not convert to a JPG: %s" % exc))
            continue
        out.append(dict(item, jpg=jpg, displayName=("%s %s emissive" % (display_name, safe))[:50]))
    return out


def rbx_asset_upload(file: str, display_name: str, confirm: bool = False, description: str = "", asset_type: str = None,
                     user_id: int = None, timeout: float = 300, retry_unknown: bool = False, probe: bool = False,
                     moderation_wait: float = 0, reuse: bool = True, privacy_on_since: str = None,
                     reupload: bool = False, accept_shrink: bool = False, emissive_masks: Any = False) -> dict:
    "Roblox: upload one file as a new asset through Open Cloud (POST /assets/v1/assets) with the person's own API key, read from their key file and never shown, then poll the operation to its end: asset id, moderation state and how long it took. An upload is permanent and moderated, so confirm must be true, and the panel must be in full mode. The creator is the signed-in Studio user (StudioService:GetUserId) unless user_id is given. Types by extension (Roblox's usage page): .fbx/.glb/.gltf as Model; .rbxm/.rbxmx as Model or, with asset_type Animation, an Animation; .png/.jpg/.bmp/.tga as Decal (or Image); .mp3/.ogg/.wav/.flac as Audio; .mp4/.mov as Video. 20 MB a call (Video 3.75 GB, streamed from disk). A Decal keeps its full resolution (a 4096 upload rendered at 4096 texels, measured). An upload Roblox charges Robux for (Video: 2000, measured) is refused by Roblox at the price 0 the bridge offers, and answered refusedRule costs-robux: MifBridge never pays. A Decal upload also answers imageId, the image asset inside it (InsertService:LoadAsset of the Decal): image Content properties draw that id, not the Decal's. Audio comes back moderation Reviewing and turns Approved minutes later: moderation_wait (0-900 s) re-reads GET /assets/v1/assets/{id} until it is no longer Reviewing. An operation that ends with no response is read again, then looked up by name in the creator's inventory (inventory.roblox.com, newest first, created after the upload began; a public read, the key is never sent there): found answers made yes with the id, listed twice without it answers made no; if the list cannot be read the outcome is unknown (made \"unknown\"), recorded in the MifBridge home's uploads.jsonl, and the same file and name are refused maybe-made until retry_unknown is true (after the person checked Creator Hub), so a retry never makes a silent duplicate; a log line that is not a record (a torn write) refuses every upload log-damaged naming the line, because these guards read that log. With probe true, the asset made is then loaded into Workspace.MifScratch by rbx_asset_probe, which reports each MeshPart's Size, MeshSize, PivotOffset and, for an .fbx, which file axis became each Roblox axis, and removes it (under probe; the upload's own ok stands either way). reuse (default true) answers, for a Model, every mesh and image id inside it as created by this upload or reused (made before the upload began, Open Cloud's creation time; an identical file re-uploaded gets a new Model over the OLD mesh, RB1030), and for a Decal its imageId the same way; a reused id keeps the privacy it had when made (Open Use if before the creator's Asset Privacy opt-in, RB1033): privacy_on_since, the ISO time they turned it on, names those as Open Use with the fix (an altered file, RB1032). The Model is loaded into Workspace.MifScratch for that read and removed; the upload's ok stands either way. The same content is not uploaded twice: an earlier upload logged as made, of the same file sha256 or - for an .fbx - the same mesh data (fbx_mesh.mesh_hash: every mesh's transform, positions, polygons, UVs, normals, materials and skin, and embedded images, so a re-export of an unchanged scene matches although Blender rewrote the bytes, RB1475) and the same asset type, by the same creator (a row logged before rows named the creator still matches, said in creatorUnknown), answers unchanged true with that asset id and sends nothing; reupload true sends it anyway. The earlier asset's moderation is read again first (GET /assets/v1/assets/{id}): one moderation has since rejected answers ok false, refusedRule prior-rejected, still unchanged and unsent, and a read that fails leaves the answer unchanged with moderationUnread saying why. An .fbx is read first, and one the reader cannot read (an ASCII FBX, a damaged file) is refused fbx-unread with the reason and sent nowhere - never uploaded unverified, the batch's rule too: a skinned file whose largest mesh is over 2048 studs at the import's scale (1 stud per file unit after node transforms, RB1700) is refused skinned-shrink, because the import shrinks the model to 2048 and leaves the bones at full size (RB1023, RB1472) - accept_shrink true sends it anyway; an unskinned one over 2048 is sent with a shrink warning. rbx_rig_check reads the bone frames after the import. An .fbx whose materials feed a texture into an emissive property (Blender's EmissiveColor) is named before anything is sent - in every answer, the needs-confirm refusal included, under emissive with a warning - because the upload brings each back with EmissiveMaskContent rbxassetid://6, an asset that does not exist, and play clients draw those meshes grey (RB2766). emissive_masks true writes each such material's mask as a JPG beside the FBX (<stem>_<material>_emissive_mask.jpg, grey RGB at quality 97, needs Pillow) from the FBX's own emissive texture (embedded, or the file it names); emissive_masks {material: image file} uses the mask the person's bake wrote for that material instead - better when the FBX's emissive texture is a colour image rather than the mask (measured: one lit 83% of its pixels where the baked mask lit 21%, RB2823) - and refuses a material that feeds no emissive texture. Then, after the Model upload, uploads each as its own Decal under the same confirm: the answer's emissiveMasks gives each material's imageId and the value to set as SurfaceAppearance.EmissiveMaskContent (rbxassetid://<imageId>, RB2768); without confirm it writes the JPGs and lists what it would send."
    given = None
    if isinstance(emissive_masks, dict):
        given = emissive_masks
        if not given or not all(isinstance(k, str) and isinstance(v, str) and os.path.isfile(v) for k, v in given.items()):
            return {"ok": False, "ran": False, "error": "emissive_masks is true, false, or {material: an image file of "
                                                         "its mask on this machine}"}
    elif not isinstance(emissive_masks, bool):
        return {"ok": False, "ran": False, "error": "emissive_masks is true, false, or {material: mask image file}"}
    sc, emis = _emissive_check(file)
    if given:
        unknown = sorted(set(given) - set((emis or {}).get("materials") or []))
        if unknown:
            return {"ok": False, "ran": False, "refusedRule": "unknown-material",
                    "error": "emissive_masks names %s, which feed no emissive texture in this FBX; the ones that do: %s"
                             % (", ".join(unknown), ", ".join((emis or {}).get("materials") or []) or "none")}
    plan = _emissive_jpgs(sc, file, emis, display_name if isinstance(display_name, str) else "", given) \
        if emissive_masks and emis else None
    out = _upload_one(file, display_name, confirm=confirm, description=description, asset_type=asset_type,
                      user_id=user_id, timeout=timeout, retry_unknown=retry_unknown, probe=probe,
                      moderation_wait=moderation_wait, reuse=reuse, privacy_on_since=privacy_on_since,
                      reupload=reupload, accept_shrink=accept_shrink)
    if emis is None:
        if emissive_masks and isinstance(file, str) and file.lower().endswith(".fbx"):
            out = dict(out, emissive={"materials": [], "note": "no material in this FBX feeds a texture into an emissive "
                                                               "property: no mask to upload"})
        return out
    out = dict(out, emissive={k: v for k, v in emis.items() if k != "textures"},
               emissiveTextures=[{k: v for k, v in r.items()} for r in emis["textures"]])
    if not emissive_masks:
        return out
    if confirm is not True or not (out.get("ok") and out.get("assetId")):
        why = ("nothing is sent without confirm" if confirm is not True else
               "the Model upload did not answer an asset id, so no mask was sent")
        return dict(out, emissiveMasks=[dict(p, sent=False) for p in plan],
                    emissiveMasksNote="the JPGs are written; %s. Each would go up as its own Decal and answer the image id "
                                      "to set as EmissiveMaskContent (RB2768)" % why)
    masks, failed = [], []
    for p in plan:
        if not p.get("jpg"):
            masks.append(dict(p, sent=False))
            failed.append("%s: %s" % (p["material"], p.get("error")))
            continue
        up = _upload_one(p["jpg"], p["displayName"], confirm=True, description=description, user_id=user_id,
                         timeout=timeout, retry_unknown=retry_unknown, reuse=reuse, privacy_on_since=privacy_on_since,
                         reupload=reupload)
        img = up.get("imageId")
        if up.get("ok") and up.get("assetId") and not img:
            # an unchanged mask answers its earlier Decal id: the image inside it is read the same way (RB1340)
            got, rep, ierr = _decal_image("rbxassetid://%s" % up["assetId"])
            img = int(got[len("rbxassetid://"):]) if not ierr and isinstance(got, str) and (rep or {}).get("image") else None
        row = dict(p, sent=up.get("ran") is not False, decalId=up.get("assetId"), imageId=img,
                   moderationState=up.get("moderationState"), unchanged=bool(up.get("unchanged")))
        if img:
            row["set"] = {"EmissiveMaskContent": "rbxassetid://%d" % img}
        else:
            failed.append("%s: %s" % (p["material"], up.get("error") or "no image id in the Decal"))
            row["error"] = up.get("error") or "no image id in the Decal"
        masks.append(row)
    out = dict(out, emissiveMasks=masks)
    if failed:
        out["emissiveMasksError"] = "; ".join(failed)
    return out


def _upload_one(file, display_name, confirm=False, description="", asset_type=None, user_id=None, timeout=300,
                retry_unknown=False, probe=False, moderation_wait=0, reuse=True, privacy_on_since=None, reupload=False,
                accept_shrink=False):
    """rbx_asset_upload's body for one file (the emissive masks of an FBX are each sent through it too)."""
    from mif_roblox import opencloud as OC
    if confirm is not True:
        return {"ok": False, "ran": False, "refusedRule": "needs-confirm",
                "error": "an upload makes a permanent, moderated asset on the person's account: pass confirm true once they agreed"}
    if not isinstance(file, str) or not os.path.isfile(file):
        return {"ok": False, "ran": False, "error": "file is a path to an existing file, got %r" % (file,)}
    if not isinstance(display_name, str) or not 1 <= len(display_name) <= 50:
        return {"ok": False, "ran": False, "error": "display_name is 1-50 characters"}
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 10 <= timeout <= 1800:
        return {"ok": False, "ran": False, "error": "timeout is 10-1800 seconds"}
    mode, st = _plugin_mode()
    if mode != "full":
        return {"ok": False, "ran": False, "refusedRule": "full-mode", "mode": mode,
                "error": "an upload writes to the person's account; the MifBridge panel must be in full mode (it is %s). "
                         "Only the person can raise it" % mode}
    if user_id is None:
        r = rbx_call("StudioService", "GetUserId")
        user_id = r.get("value") if r.get("ok") else None
        if not user_id:
            return {"ok": False, "ran": False, "error": "no signed-in Studio user to upload as: %s" % (r.get("error") or r.get("value"))}
    if not isinstance(user_id, int) or isinstance(user_id, bool) or user_id <= 0:
        return {"ok": False, "ran": False, "error": "user_id is a positive whole number"}
    if not isinstance(retry_unknown, bool):
        return {"ok": False, "ran": False, "error": "retry_unknown is true or false"}
    if not isinstance(probe, bool):
        return {"ok": False, "ran": False, "error": "probe is true or false"}
    if not isinstance(moderation_wait, (int, float)) or isinstance(moderation_wait, bool) or not 0 <= moderation_wait <= 900:
        return {"ok": False, "ran": False, "error": "moderation_wait is 0-900 seconds"}
    if not isinstance(reuse, bool):
        return {"ok": False, "ran": False, "error": "reuse is true or false"}
    from mif_roblox import domain_assets as A  # noqa: PLC0415 - the reuse check lives with the asset reads
    if privacy_on_since is not None and A._seconds(privacy_on_since) is None:
        return {"ok": False, "ran": False, "error": "privacy_on_since is an ISO time such as 2026-11-20T05:00:00Z"}
    if not isinstance(reupload, bool) or not isinstance(accept_shrink, bool):
        return {"ok": False, "ran": False, "error": "reupload and accept_shrink are true or false"}
    digest = OC.file_sha256(file)
    mesh_hash, size_check = None, None
    if file.lower().endswith(".fbx"):
        from mif_roblox import fbx_mesh as FM  # noqa: PLC0415 - the mesh pipeline's FBX reader
        try:
            scene = FM.load(file)
            mesh_hash, size_check = FM.mesh_hash(scene), FM.skinned_size_check(scene)
        except (OSError, ValueError) as exc:
            # ONE RULE WITH rbx_asset_upload_batch (decided 2026-11-23): a file this reader cannot read is skipped, never
            # uploaded unverified - an upload is a permanent asset, and the skinned-size and dedupe checks need the read.
            return {"ok": False, "ran": False, "skipped": True, "refusedRule": "fbx-unread", "file": file,
                    "why": "the FBX did not read: %s" % exc,
                    "error": "%s was not uploaded: the FBX did not read (%s), so its size and duplicates could not be "
                             "checked. Export it again as a binary FBX and upload that" % (os.path.basename(file), exc)}
        if size_check and size_check["skinned"] and size_check["overLimit"] and not accept_shrink:
            return {"ok": False, "ran": False, "refusedRule": "skinned-shrink", "skinnedSize": size_check,
                    "error": "a skinned FBX whose largest mesh (%s) is %.1f studs at the import's scale: Roblox shrinks the "
                             "whole model to 2048 (x%.4f) and leaves the bones at full size, so the skin breaks (RB1023, "
                             "RB1472). Nothing was sent: export it under 2048 studs and size it in game with "
                             "Model:ScaleTo, or pass accept_shrink true" % (size_check["largestMesh"],
                                                                           size_check["largestStuds"],
                                                                           size_check["shrinkFactor"])}
    damaged = OC.log_damage()
    if damaged:
        return damaged
    if not reupload:
        same = OC.prior_made(digest, mesh_hash, asset_type or OC.default_type(file), creator=user_id)
        if same:
            out = {"ok": True, "ran": False, "unchanged": True, "assetId": same.get("assetId"), "prior": same,
                   "matchedOn": "sha256" if same.get("sha256") == digest else "meshHash", "sha256": digest,
                   "creatorUnknown": same.get("creator") is None,
                   "meshHash": mesh_hash, "uploadsLog": OC.uploads_log(),
                   "note": "unchanged since asset %s (%s, %s): nothing was sent; reupload true sends it again" % (
                       same.get("assetId"), same.get("displayName"), same.get("at"))}
            return _prior_moderation(OC, out)
    prior = OC.prior_unknown(digest, display_name)
    if prior and not retry_unknown:
        return {"ok": False, "ran": False, "refusedRule": "maybe-made", "prior": prior, "uploadsLog": OC.uploads_log(),
                "error": "an earlier upload of this file as %r ended with no response, so it may have made an asset "
                         "(operation %s, %s): nothing sent. The person checks Creator Hub; retry_unknown true uploads "
                         "again" % (display_name, prior.get("operation"), prior.get("at"))}
    started = time.time()
    out = OC.upload(file, display_name, user_id, description=description, asset_type=asset_type, poll_timeout=timeout)
    OC.log_upload(digest, display_name, out, mesh_hash=mesh_hash, asset_type=asset_type or OC.default_type(file),
                  creator=user_id)
    if mesh_hash:
        out = dict(out, sha256=digest, meshHash=mesh_hash)
    if size_check and size_check["overLimit"]:
        out = dict(out, skinnedSize=size_check, shrinkWarning="the largest mesh is %.1f studs: the import shrinks the "
                   "whole model by x%.4f to 2048 (RB1023)" % (size_check["largestStuds"], size_check["shrinkFactor"]))
    if out.get("assetId") and (out.get("assetType") or asset_type) == "Decal":
        # An image upload answers the Decal's id; Content properties draw only the image inside it (RB1340).
        img, rep, ierr = _decal_image("rbxassetid://%s" % out["assetId"])
        out = dict(out, imageId=int(img[len("rbxassetid://"):]) if isinstance(img, str) and (rep or {}).get("image") else None,
                   imageResolve=rep if not ierr else dict(rep or {}, error=ierr))
    if moderation_wait and out.get("assetId") and out.get("moderationState") in (None, "Reviewing"):
        out = dict(out, moderation=_wait_moderation(OC, out["assetId"], moderation_wait))
        if out["moderation"].get("moderationState"):
            out["moderationState"] = out["moderation"]["moderationState"]
    made_type = out.get("assetType") or asset_type or OC.CONTENT.get(os.path.splitext(file)[1].lower(), (0, None))[1]
    if probe:
        out = dict(out, probe=_upload_probe(out, file, reuse=reuse and made_type == "Model", started=started,
                                            privacy_on_since=privacy_on_since))
        if "reuse" in out["probe"]:
            out["reuse"] = out["probe"]["reuse"]
    if reuse and out.get("assetId") and "reuse" not in out:
        out = dict(out, reuse=_upload_reuse(A, OC, out, made_type, started, privacy_on_since))
    return out


def _upload_reuse(A, OC, out, made_type, started, privacy_on_since):
    """Created or reused, for what an upload made: a Model's mesh and image ids (loaded and removed again), a Decal's
    image. Never raises; its answer never changes the upload's ok."""
    try:
        if made_type == "Model":
            r = A.probe(int(out["assetId"]), reuse=True, started=started, privacy_on_since=privacy_on_since, meshes=False)
            return r.get("reuse") or {"ok": False, "error": r.get("error") or "the Model did not load for the reuse check"}
        if made_type == "Decal" and out.get("imageId"):
            v = A.judge_reuse({}, {int(out["imageId"]): OC.asset_state(int(out["imageId"]))}, started=started,
                              privacy_on_since=privacy_on_since)
            v["ok"] = not v.get("unread") and not v.get("error")
            return v
        return {"ok": True, "checked": False, "note": "a %s upload holds no mesh or image id to check" % made_type}
    except Exception as exc:  # noqa: BLE001 - never raise
        return {"ok": False, "error": "the reuse check failed: %s" % OC.exc_text(exc)}


def _prior_moderation(OC, out):
    """An unchanged answer with the earlier asset's moderation read NOW (GET /assets/v1/assets/{id}): the log holds the
    state at upload time, and an asset moderation rejected later read as a good one (spec: prior_made never re-reads
    moderation). Rejected: ok false, refusedRule prior-rejected, still nothing sent and still unchanged, so a batch marks
    the entry rejected and goes on. A read that fails leaves the answer as it was, with moderationUnread saying why."""
    try:
        st = OC.asset_state(int(out["assetId"]))
    except (TypeError, ValueError, OSError) as exc:
        st = {"ok": False, "error": str(exc)}
    if not isinstance(st, dict) or not st.get("ok"):
        return dict(out, moderationState=None, moderationUnread=(st or {}).get("error") if isinstance(st, dict) else str(st))
    out = dict(out, moderationState=st.get("moderationState"))
    if st.get("moderationState") == "Rejected":
        out.update(ok=False, refusedRule="prior-rejected",
                   error="the same content was uploaded before as asset %s and moderation has since rejected it: nothing "
                         "was sent, and the same file would be rejected again. Change the content; reupload true sends "
                         "it anyway" % out["assetId"])
    return out


def _wait_moderation(OC, asset_id, seconds, every=10.0):
    """Re-read an asset's moderation until it is no longer Reviewing or seconds pass."""
    t0, reads, last = time.time(), 0, {}
    while True:
        last = OC.asset_state(int(asset_id))
        reads += 1
        if not last.get("ok") or last.get("moderationState") not in (None, "Reviewing") or time.time() - t0 + every > seconds:
            break
        time.sleep(every)
    return dict(last, reads=reads, waitedSeconds=round(time.time() - t0, 1))


def _upload_probe(out, file, reuse=False, started=None, privacy_on_since=None):
    """rbx_asset_probe on the asset an upload made; the file is passed when it is an .fbx, for the axis mapping. With
    reuse, the same load also answers the created-or-reused check against the upload's start."""
    aid = out.get("assetId") if isinstance(out, dict) else None
    try:
        aid = int(aid)
    except (TypeError, ValueError):
        return {"ok": False, "ran": False, "error": "the upload named no asset id to probe (%s)" % (out.get("error") if isinstance(out, dict) else out)}
    from mif_roblox import domain_assets  # noqa: PLC0415 - the probe lives with the asset reads
    fbx = file if str(file).lower().endswith(".fbx") else None
    if reuse:
        return domain_assets.probe(aid, file=fbx, reuse=True, started=started, privacy_on_since=privacy_on_since)
    return domain_assets.rbx_asset_probe(aid, file=fbx)


def rbx_asset_place(asset_id: int, parent: Any = "Workspace.MifScratch", mesh_part: bool = True, flatten: bool = False,
                    longest: float = None, clear_broken_maps: bool = False) -> dict:
    "Roblox Studio: load one of the person's own assets (InsertService:LoadAsset) under parent, default Workspace.MifScratch, and read it back: its path, class, bounding box (Model:GetBoundingBox) and each MeshPart's Size and MeshId. With mesh_part, also make a MeshPart from the first MeshId with AssetService:CreateMeshPartAsync under the same parent and read its Size. flatten true leaves Model > part(s) (LoadAsset gives Model > Model > MeshPart): every part moved up to the placed Model, the first named Body and made its PrimaryPart, every other child removed; longest scales the Model (Model:ScaleTo) so its longest side is that many studs; either one also sets the AssetId attribute to asset_id, so a build can skip an unchanged one. Read back in the normalized answer: children parts only, PrimaryPart Body, the extents' longest side, the attribute. Every SurfaceAppearance placed has its map Contents (ColorMapContent, NormalMapContent, RoughnessMapContent, MetalnessMapContent, EmissiveMaskContent) read: one naming rbxassetid://6 is the broken emissive mask an FBX with an emissive texture comes back with - play clients fail its TexturePack and draw the mesh grey while edit mode draws it fine (RB2766) - and is answered under brokenMaps with a warning; clear_broken_maps true sets each to an empty Content and reads it back empty (ok false if one does not clear, or if the maps could not be read). Full mode, or inside the scratch folder in scratch mode."
    if not isinstance(asset_id, int) or isinstance(asset_id, bool) or asset_id <= 0:
        return {"ok": False, "ran": False, "error": "asset_id is a positive whole number"}
    if not isinstance(flatten, bool) or (longest is not None and not (isinstance(longest, (int, float))
                                                                      and not isinstance(longest, bool) and 0 < longest <= 2048)):
        return {"ok": False, "ran": False, "error": "flatten is true or false; longest a size in studs above 0 and at most 2048"}
    if not isinstance(clear_broken_maps, bool):
        return {"ok": False, "ran": False, "error": "clear_broken_maps is true or false"}
    if (flatten or longest is not None) and mesh_part:
        mesh_part = False                    # the extra MeshPart would sit beside the normalized model as a second copy
    pparts, err = _target(parent, "parent")
    if err:
        return err
    loaded = rbx_call("InsertService", "LoadAsset", [asset_id], timeout=120)
    h = (loaded.get("value") or {}).get("handle") if isinstance(loaded.get("value"), dict) else None
    if not h:
        return {"ok": False, "ran": True, "step": "LoadAsset", "error": loaded.get("error") or "LoadAsset returned %r" % (loaded.get("value"),),
                "loaded": loaded}
    path, perr = _place_handle(h, pparts, "Asset%d" % asset_id)
    if perr:
        return {"ok": False, "ran": True, "step": "place", "handle": h, "error": perr}
    out = {"ok": True, "verified": True, "path": path, "assetId": asset_id}
    cls = _resolve(V.split_path(path))[0]
    out["className"] = cls
    if cls == "Model":
        bb = rbx_call(path, "GetBoundingBox")
        vals = bb.get("values") or []
        out["boundingBox"] = {"cframe": vals[0] if vals else None, "size": vals[1] if len(vals) > 1 else None}
    meshes = rbx_find(class_name="MeshPart", root=path, limit=50).get("matches") or []
    if cls == "MeshPart":
        meshes = [{"path": path}] + meshes
    out["meshParts"] = []
    for m in meshes:
        g = rbx_get_props(m["path"], ["Size", "MeshId", "TextureID"])
        out["meshParts"].append({"path": m["path"], "props": g.get("values") or g.get("props") or g})
    # No rbx_release_handle here: releasing a handle Destroys the object, and this one is now in the place (measured: an
    # earlier version released it and the placed model was gone right after its read-back).
    if mesh_part and out["meshParts"]:
        props = out["meshParts"][0]["props"]
        mesh_id = props.get("MeshId") if isinstance(props, dict) else None
        if mesh_id:
            made = rbx_call("AssetService", "CreateMeshPartAsync", [mesh_id], timeout=120)
            mh = (made.get("value") or {}).get("handle") if isinstance(made.get("value"), dict) else None
            if mh:
                mp, _e = _place_handle(mh, pparts, "MeshPart%d" % asset_id)
                g = rbx_get_props(mp, ["Size", "MeshId"]) if mp else {}
                out["createMeshPart"] = {"path": mp, "props": g.get("values") or g.get("props") or g}
            else:
                out["createMeshPart"] = {"error": made.get("error") or made.get("value")}
    # The last read-back: everything placed is still there after the tool's own steps.
    still = [p for p in [path] + [(out.get("createMeshPart") or {}).get("path")] if p]
    gone = [p for p in still if not rbx_get_props(p, ["Name"]).get("ok")]
    if gone:
        out.update(ok=False, verified=False, error="placed, then gone before the tool returned: %s" % ", ".join(gone))
        return out
    if flatten or longest is not None:
        from mif_roblox import domain_assets  # noqa: PLC0415 - the normalizer lives with the asset reads
        if cls != "Model":
            return dict(out, ok=False, verified=False, error="flatten and longest normalize a Model; asset %d loaded as a %s"
                        % (asset_id, cls))
        norm = domain_assets.flatten_normalize(path, asset_id, flatten=flatten, longest=longest)
        out["normalized"] = norm
        if not norm.get("ok"):
            out.update(ok=False, verified=False, error="placed, but normalizing failed: %s" % norm.get("error"))
            return out
    from mif_roblox import domain_assets  # noqa: PLC0415 - the map check lives with the asset reads
    bm = domain_assets.broken_maps(domain_assets._T(), path, clear=clear_broken_maps)
    out["brokenMaps"] = bm
    if bm.get("warning"):
        out["warning"] = bm["warning"]
    if not bm.get("ok"):
        out.update(ok=False, verified=False, error="placed, but the SurfaceAppearance map check: %s" % bm.get("error"))
    return out


def _child_names(parts):
    """({child names of parts}, None) or (None, the failure). Every child: the count is read first (depth 0) and the
    names with a limit above it, and an answer that holds fewer names than the count fails. A failed read is a
    failure, never an empty set, which would hand back names already in use (found 2026-11-22)."""
    head = _rbx("tree", root=parts, depth=0, limit=1)
    count = (head.get("root") or {}).get("childCount") if head.get("ok") else None
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        return None, dict(head, ok=False, error="could not read the children of %s to pick free names: %s"
                          % (V.path_text(parts), head.get("error") or "no child count"))
    t = _rbx("tree", root=parts, depth=1, limit=count + 2)
    node = (t.get("root") or {}) if t.get("ok") else {}
    kids, now = node.get("children"), node.get("childCount")
    if (not isinstance(kids, list) or node.get("truncated") or t.get("truncated")
            or not isinstance(now, int) or len(kids) < now):
        return None, dict(t, ok=False, error="could not see all %d children of %s to pick free names: %s"
                          % (count, V.path_text(parts), t.get("error") or "%s of them read" % (
                              len(kids) if isinstance(kids, list) else "none")))
    return {c.get("name") for c in kids if isinstance(c, dict)}, None


def _unique_name(parent_parts, base):
    """A child name not yet under the parent, so the placed object has a path of its own (two loads of one asset are both
    named Model, and a path that names two children is refused as ambiguous). Every child is read (_child_names); when
    they cannot all be read, the name gets a random suffix instead, never one that may be in use (until 2026-11-23 a
    failed or cut read was taken for an empty parent)."""
    taken, err = _child_names(parent_parts)
    if err:
        return "%s_%s" % (base, uuid.uuid4().hex[:8])
    name, n = base, 1
    while name in taken:
        n += 1
        name = "%s_%d" % (base, n)
    return name


def _place_handle(h, pparts, base):
    """Rename a loaded object (still outside the place) to a free name, then parent it. (path, error)."""
    name = _unique_name(pparts, base)
    nm = _rbx("set_props", target={"handle": h}, props={"Name": name})
    if not nm.get("ok"):
        return None, "renaming the loaded object failed: %s" % nm.get("error")
    moved = _rbx("reparent", target={"handle": h}, parent=pparts)
    if not moved.get("ok") or not same_parent(moved.get("parentPath") or [], pparts):
        return None, moved.get("error") or "not under %s" % V.join_path(pparts)
    return V.join_path(moved.get("path")), None


# the internal notes's figures, from Roblox's creator pages. rbx_ugc_rules compares Studio's own rules with these and says
# where they differ; Studio's numbers win in the checks.
UGC_PAGE_FIGURES = {
    "MeshRules.AccessoryMaxTriangles": 4000,
    "MeshRules.BodyPartMaxTriangles.Enum.AssetType.DynamicHead": 4000,
    "MeshRules.BodyPartMaxTriangles.Enum.AssetType.Torso": 1750,
    "MeshRules.BodyPartMaxTriangles.Enum.AssetType.LeftArm": 1248,
    "MeshRules.BodyPartMaxTriangles.Enum.AssetType.RightArm": 1248,
    "MeshRules.BodyPartMaxTriangles.Enum.AssetType.LeftLeg": 1248,
    "MeshRules.BodyPartMaxTriangles.Enum.AssetType.RightLeg": 1248,
    "TextureRules.AssetTextureLimits.Enum.AssetType.Hat.ColorMapSize": 2048,
    "AccessoryRules.Enum.AssetType.Hat.Attachments": [{"Name": "HatAttachment", "Offset": {"x": 0, "y": 0, "z": 0},
                                                        "Size": {"x": 3, "y": 4, "z": 3}}],
}
UGC_DIR = os.path.join(HERE, "ugc_rules")


def _flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, "%s%s." % (prefix, k)))
    else:
        out[prefix[:-1]] = d
    return out


@_reports
def rbx_ugc_rules(store: bool = True) -> dict:
    "Roblox Studio: Roblox's own avatar-item validation rules (AvatarCreationService:GetValidationRules) for this Studio build: triangle budgets, texture sizes, accessory attachments and bounds, body-part and full-body bounds. With store, kept as mif_roblox/ugc_rules/<studio version>.json; the answer lists what changed since the last stored build and where Studio's numbers differ from the creator pages' figures. Read only."
    r = rbx_call("AvatarCreationService", "GetValidationRules")
    if not r.get("ok"):
        return r
    rules = r.get("value") or {}
    st = rbx_status()
    ver = str(((st.get("plugin") or {}).get("studioVersion")) or "unknown")
    flat = _flatten(rules)
    differs = {k: {"pages": v, "studio": flat.get(k)} for k, v in UGC_PAGE_FIGURES.items() if flat.get(k) != v}
    changed, previous = None, None
    try:
        os.makedirs(UGC_DIR, exist_ok=True)
        from mif_roblox import ugc_check as _ugc  # noqa: PLC0415 - by build number: a string sort puts 0.99 after 0.1000
        stored = sorted((f for f in os.listdir(UGC_DIR) if f.endswith(".json") and f != ver + ".json"),
                        key=_ugc._version_key)
        if stored:
            previous = stored[-1][:-5]
            old = _flatten(json.load(open(os.path.join(UGC_DIR, stored[-1]), encoding="utf-8")))
            changed = {k: {"was": old.get(k), "now": flat.get(k)} for k in sorted(set(old) | set(flat)) if old.get(k) != flat.get(k)}
        if store:
            with open(os.path.join(UGC_DIR, ver + ".json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(rules, f, indent=1, sort_keys=True)
                f.write("\n")
    except (OSError, ValueError) as exc:
        return {"ok": False, "ran": True, "error": "rules read, but storing them failed: %s" % exc}
    mesh = rules.get("MeshRules") or {}
    tex = rules.get("TextureRules") or {}
    return {"ok": True, "studioVersion": ver, "sections": sorted(rules), "values": len(flat),
            "accessoryMaxTriangles": mesh.get("AccessoryMaxTriangles"), "bodyPartMaxTriangles": mesh.get("BodyPartMaxTriangles"),
            "meshPartBounds": [mesh.get("MeshPartBoundsMin"), mesh.get("MeshPartBoundsMax")],
            "maxTextureSize": tex.get("MaxTextureSize"), "accessoryTypes": sorted(rules.get("AccessoryRules") or {}),
            "differsFromPages": differs, "previousBuild": previous, "changedSincePrevious": changed,
            "stored": os.path.join(UGC_DIR, ver + ".json") if store else None}


def _cf_point(cf, local=(0, 0, 0)):
    """A decoded CFrame {position, rotation} applied to a local point: the world point."""
    p, r = cf["position"], cf["rotation"]
    x, y, z = local
    return [p[0] + r[0] * x + r[1] * y + r[2] * z, p[1] + r[3] * x + r[4] * y + r[5] * z, p[2] + r[6] * x + r[7] * y + r[8] * z]


def _cf_mul(a, b):
    """a * b for decoded CFrames."""
    ra, rb = a["rotation"], b["rotation"]
    rot = [sum(ra[i * 3 + k] * rb[k * 3 + j] for k in range(3)) for i in range(3) for j in range(3)]
    return {"position": _cf_point(a, b["position"]), "rotation": rot}


SCALE_PROPS = ("HeightScale", "WidthScale", "DepthScale", "HeadScale", "BodyTypeScale", "ProportionScale")


def rbx_mannequin(name: str = "Mannequin", parent: Any = "Workspace.MifScratch", rig_type: str = "R15",
                  scales: dict = None, position: list = None) -> dict:
    "Roblox Studio: a default avatar body to fit items on - a HumanoidDescription (with scales: HeightScale, WidthScale, DepthScale, HeadScale, BodyTypeScale, ProportionScale) under parent, then Players:CreateHumanoidModelFromDescriptionAsync with rig_type R15 or R6, the rig placed under parent and, with position, pivoted there. Read back: its path, and the attachments on it by name."
    if rig_type not in ("R15", "R6"):
        return {"ok": False, "ran": False, "error": "rig_type is R15 or R6"}
    if scales is not None and (not isinstance(scales, dict) or set(scales) - set(SCALE_PROPS)):
        return {"ok": False, "ran": False, "error": "scales takes %s" % ", ".join(SCALE_PROPS)}
    pparts, err = _target(parent, "parent")
    if err:
        return err
    dname = _unique_name(pparts, name + "Description")
    # An empty HumanoidDescription gives a black body (measured: every capture a silhouette), which hides clothing and
    # shading; the mannequin is a light grey unless the scales say otherwise.
    props = {k: [0.8, 0.8, 0.8] for k in ("HeadColor", "TorsoColor", "LeftArmColor", "RightArmColor", "LeftLegColor", "RightLegColor")}
    props.update(scales or {})
    d = rbx_create("HumanoidDescription", pparts, name=dname, props=props)
    if not d.get("ok"):
        return d
    m = rbx_call("Players", "CreateHumanoidModelFromDescriptionAsync", [d["path"], rig_type], timeout=120)
    val = m.get("value")
    h = val.get("handle") if isinstance(val, dict) else None
    rbx_destroy(d["path"])
    if not h:
        return {"ok": False, "ran": True, "step": "CreateHumanoidModelFromDescriptionAsync", "error": m.get("error") or "returned %r" % (val,)}
    path, perr = _place_handle(h, pparts, name)
    if perr:
        return {"ok": False, "ran": True, "step": "place", "error": perr}
    if position is not None:
        pv = rbx_call(path, "PivotTo", [{"position": list(position)}])
        if not pv.get("ok"):
            return dict(pv, step="PivotTo", path=path)
    atts = rbx_find(class_name="Attachment", root=path, limit=500).get("matches") or []
    return {"ok": True, "verified": True, "path": path, "rigType": rig_type, "scales": scales or {},
            "attachments": sorted({a["path"].rsplit(".", 1)[-1] for a in atts})}


AVATAR_BODIES = {
    "default": {},
    "slender": {"BodyTypeScale": 0, "ProportionScale": 1, "WidthScale": 0.75, "HeightScale": 1.05},
    "wide": {"WidthScale": 1, "DepthScale": 1, "BodyTypeScale": 1, "ProportionScale": 0, "HeightScale": 0.9},
    "tall": {"HeightScale": 1.05, "HeadScale": 1, "BodyTypeScale": 0.3},
}


@_reports
def rbx_avatar_preview(item: Any = None, bodies: dict = None, parent: Any = "Workspace.MifScratch", at: list = None,
                       max_width: int = 640) -> dict:
    "Roblox Studio: an avatar item on bodies of several scales - for each body (default: default, slender, wide, tall; or {name: scales}) a mannequin with those HumanoidDescription scales, a copy of item (an Accessory) added with Humanoid:AddAccessory, the applied description read back (Humanoid:GetAppliedDescription) and one viewport capture framed on it. The bodies stand in a row from at (default far from the place's own content). Read back: scales applied, the item on each body, each capture not blank."
    bodies = bodies or AVATAR_BODIES
    pparts, err = _target(parent, "parent")
    if err:
        return err
    base = at or [2000, 200, 2000]
    out, problems = {"ok": True, "bodies": {}}, []
    for i, (bname, scales) in enumerate(bodies.items()):
        pos = [base[0] + i * 14, base[1], base[2]]
        mq = rbx_mannequin(name="Body_" + bname, parent=parent, scales=scales, position=pos)
        row = {"mannequin": mq.get("path"), "ok": mq.get("ok")}
        if not mq.get("ok"):
            problems.append("%s: %s" % (bname, mq.get("error")))
            out["bodies"][bname] = dict(row, error=mq.get("error"))
            continue
        hum = (rbx_find(class_name="Humanoid", root=mq["path"], limit=1).get("matches") or [{}])[0].get("path")
        applied = rbx_call(hum, "GetAppliedDescription")
        ah = (applied.get("value") or {}).get("handle") if isinstance(applied.get("value"), dict) else None
        if ah:
            got = rbx_get_props({"handle": ah}, list(SCALE_PROPS)).get("values") or {}
            row["applied"] = got
            for k, v in (scales or {}).items():
                if not isinstance(got.get(k), (int, float)) or abs(got[k] - v) > 1e-4:
                    problems.append("%s: %s applied %s, asked %s" % (bname, k, got.get(k), v))
            rbx_release_handle(ah)
        if item is not None:
            # the copy goes beside the body, not into it: AddAccessory on an accessory already parented to the character
            # makes no joint (measured, 0.748)
            c = rbx_clone(item, parent=parent, name=_unique_name(pparts, "Item_" + bname))
            ipath = c.get("path")
            if c.get("ok"):
                # a copy of an item already worn carries its joint to the first body: drop it, AddAccessory makes its own
                for old in (rbx_find(class_name="RigidConstraint", root=ipath, limit=5).get("matches") or []) + \
                        (rbx_find(class_name="JointInstance", root=ipath, limit=5).get("matches") or []):
                    rbx_destroy(old["path"])
                add = rbx_call(hum, "AddAccessory", [ipath])
                ipath = mq["path"] + "." + ipath.rsplit(".", 1)[-1]
                row["item"] = ipath if add.get("ok") else {"error": add.get("error")}
                joints = rbx_find(class_name="RigidConstraint", root=ipath, limit=2).get("matches") or []
                if not add.get("ok") or not joints:
                    problems.append("%s: the item is not held on (%s)" % (bname, add.get("error") or "no joint"))
            else:
                problems.append("%s: copying the item failed: %s" % (bname, c.get("error")))
        shot = rbx_screenshot(camera={"position": [pos[0], pos[1] + 2.5, pos[2] - 7], "look_at": [pos[0], pos[1] + 1.5, pos[2]]},
                              max_width=max_width, name="avatar_%s" % bname)
        row["capture"] = {k: shot.get(k) for k in ("ok", "file", "blank", "showsSomething", "error") if k in shot}
        if not shot.get("ok") or shot.get("blank") is True:
            problems.append("%s: capture %s" % (bname, shot.get("error") or "blank"))
        out["bodies"][bname] = row
    if problems:
        out.update(ok=False, error="; ".join(problems))
    out["verified"] = out["ok"]
    return out


@_reports
def rbx_accessory_build(handle: Any = None, accessory_type: str = "Hat", name: str = "MifAccessory", size: list = None,
                        mannequin: Any = None, parent: Any = "Workspace.MifScratch") -> dict:
    "Roblox Studio: a rigid accessory fitted on a mannequin. handle is a MeshPart or Part to copy as the Handle (or a plain block of size, default 1.6 x 1 x 1.6); accessory_type a key of Roblox's AccessoryRules (Hat, FaceAccessory, BackAccessory...). The Handle gets the type's attachment, the accessory is checked against Studio's own rules (rbx_ugc_rules: triangle budget, the attachment's size box, rigid allowed) and refused budget when over, then Humanoid:AddAccessory on the mannequin (a fresh one when not given). Read back: the accessory under the rig, the weld, the two attachments at the same world point, and the handle's size inside the type's box."
    pparts, err = _target(parent, "parent")
    if err:
        return err
    rules = rbx_call("AvatarCreationService", "GetValidationRules")
    if not rules.get("ok"):
        return rules
    acc_rules = (rules.get("value") or {}).get("AccessoryRules") or {}
    key = "Enum.AssetType." + accessory_type
    rule = acc_rules.get(key)
    if not rule:
        return {"ok": False, "ran": False, "refusedRule": "type", "accessoryTypes": sorted(k.split(".")[-1] for k in acc_rules),
                "error": "%s is not an accessory type in Studio's rules" % accessory_type}
    if not rule.get("RigidAllowed"):
        return {"ok": False, "ran": False, "refusedRule": "layered-only",
                "error": "%s is layered clothing only in Roblox's rules; a rigid handle is not allowed" % accessory_type}
    att = (rule.get("Attachments") or [{}])[0]
    box = att.get("Size") or {}
    max_tris = ((rules.get("value") or {}).get("MeshRules") or {}).get("AccessoryMaxTriangles")
    problems, report = [], {"type": accessory_type, "attachment": att.get("Name"), "box": box, "maxTriangles": max_tris}
    aname = _unique_name(pparts, name)
    acc = rbx_create("Accessory", pparts, name=aname, props={"AccessoryType": "Hat" if accessory_type == "Hat" else accessory_type.replace("Accessory", "")})
    if not acc.get("ok"):
        acc = rbx_create("Accessory", pparts, name=aname)
        if not acc.get("ok"):
            return acc
        report["accessoryTypeNote"] = "AccessoryType left at its default: %s" % (acc.get("error") or "the enum has no matching item")
    apath = acc["path"]
    if handle is not None:
        c = rbx_clone(handle, parent=apath, name="Handle")
        if not c.get("ok"):
            return dict(c, step="copy handle")
        hpath = c["path"]
        # Roblox's validator wants Massless false on a Handle (measured, RB62)
        rbx_set_props(hpath, {"Anchored": False, "CanCollide": False, "Massless": False})
    else:
        s = size or [1.6, 1, 1.6]
        hc = rbx_create("Part", apath, name="Handle", props={"Size": s, "CanCollide": False, "Massless": False, "Anchored": False})
        if not hc.get("ok"):
            return dict(hc, step="handle")
        hpath = hc["path"]
    hp = rbx_get_props(hpath, ["Size"])
    hsize = (hp.get("values") or {}).get("Size") or {}
    report["handleSize"] = hsize
    if any((hsize.get(k) or 0) > (box.get(k) or 0) + 1e-6 for k in "xyz"):
        problems.append("the handle %s is outside the %s box %s" % (hsize, accessory_type, box))
    hcls = _resolve(V.split_path(hpath))[0]
    if hcls == "MeshPart":
        ins = rbx_mesh_inspect(hpath)
        report["triangles"] = ins.get("triangles")
        if isinstance(ins.get("triangles"), int) and max_tris and ins["triangles"] > max_tris:
            problems.append("%d triangles is over the accessory budget of %d" % (ins["triangles"], max_tris))
    ac = rbx_create("Attachment", hpath, name=att.get("Name") or "HatAttachment")
    if not ac.get("ok"):
        return dict(ac, step="attachment")
    if problems:
        return {"ok": False, "ran": True, "refusedRule": "budget", "accessory": apath, "report": report, "error": "; ".join(problems)}
    if mannequin is None:
        mq = rbx_mannequin(parent=parent)
        if not mq.get("ok"):
            return dict(mq, step="mannequin")
        mannequin = mq["path"]
    hum = rbx_find(class_name="Humanoid", root=mannequin, limit=2).get("matches") or []
    if not hum:
        return {"ok": False, "ran": True, "error": "%s has no Humanoid" % mannequin}
    add = rbx_call(hum[0]["path"], "AddAccessory", [apath])
    if not add.get("ok"):
        return dict(add, step="AddAccessory")
    new_path = mannequin + "." + aname
    out = {"ok": True, "accessory": new_path, "mannequin": mannequin, "report": report}
    if not rbx_get_props(new_path, ["Name"]).get("ok"):
        return dict(out, ok=False, verified=False, error="the accessory is not under the mannequin after AddAccessory")
    # What holds the accessory on is measured, not assumed: a Weld named AccessoryWeld, or a constraint.
    weld = []
    for cls in ("JointInstance", "RigidConstraint", "WeldConstraint"):
        weld += [dict(m, cls=cls) for m in rbx_find(class_name=cls, root=new_path, limit=5).get("matches") or []]
    out["weld"] = ("%s (%s)" % (weld[0]["path"], weld[0].get("className") or weld[0]["cls"])) if weld else None
    body_att = [a for a in rbx_find(name=att.get("Name"), class_name="Attachment", root=mannequin, limit=10).get("matches") or []
                if not a["path"].startswith(new_path + ".")]
    hand = new_path + ".Handle"
    try:
        hcf = (rbx_get_props(hand, ["CFrame"]).get("values") or {})["CFrame"]
        hat = (rbx_get_props(hand + "." + att["Name"], ["CFrame"]).get("values") or {})["CFrame"]
        bpart = body_att[0]["path"].rsplit(".", 1)[0]
        bcf = (rbx_get_props(bpart, ["CFrame"]).get("values") or {})["CFrame"]
        bat = (rbx_get_props(body_att[0]["path"], ["CFrame"]).get("values") or {})["CFrame"]
        a_world = _cf_mul(hcf, hat)["position"]
        b_world = _cf_mul(bcf, bat)["position"]
        gap = sum((x - y) ** 2 for x, y in zip(a_world, b_world)) ** 0.5
        out["attachmentGap"] = round(gap, 5)
        out["bodyAttachment"] = body_att[0]["path"]
    except (KeyError, IndexError, TypeError) as exc:
        out["attachmentGap"] = None
        out["gapError"] = "could not read the attachments' CFrames: %r" % (exc,)
    ok = bool(out["weld"]) and isinstance(out.get("attachmentGap"), float) and out["attachmentGap"] <= 0.01
    out.update(ok=ok, verified=ok)
    if not ok:
        out["error"] = "fitted, but the read-back disagrees: weld %s, attachment gap %s" % (out["weld"], out.get("attachmentGap"))
    return out


def _decal_image(v):
    """An rbxassetid that names a Decal is swapped for the image the Decal holds: a Decal id as a template draws nothing
   , and an Open Cloud image upload answers the Decal's id. InsertService:LoadAsset
    answers a Model holding one Decal for a Decal id and an empty Model for an image id. (value, report, error)."""
    try:
        aid = int(v[len("rbxassetid://"):])
    except ValueError:
        return None, None, "an rbxassetid is rbxassetid://<digits>, got %r" % (v,)
    r = rbx_call("InsertService", "LoadAsset", [aid], timeout=60)
    h = (r.get("value") or {}).get("handle") if isinstance(r.get("value"), dict) else None
    if not h:
        return v, {"decal": None, "loadAsset": r.get("error") or r.get("value")}, None
    try:
        kids = rbx_call({"handle": h}, "GetDescendants").get("value") or []
        decals = [k for k in kids if isinstance(k, dict) and k.get("className") == "Decal" and k.get("handle")]
        if not decals:
            return v, {"decal": None}, None
        tex = (rbx_get_props({"handle": decals[0]["handle"]}, ["TextureContent"]).get("values") or {}).get("TextureContent")
        if not (isinstance(tex, str) and tex.startswith("rbxassetid://")):
            return None, {"decal": aid, "texture": tex}, "asset %d is a Decal with no image id in it (%r)" % (aid, tex)
        return tex, {"decal": aid, "image": tex}, None
    finally:
        rbx_release_handle(h)


IMAGE_PROPS = ("ColorMapContent", "NormalMapContent", "MetalnessMapContent", "RoughnessMapContent", "EmissiveMaskContent",
               "TextureContent", "Texture", "ColorMap", "NormalMap", "MetalnessMap", "RoughnessMap", "ImageContent", "Image",
               "ShirtTemplateContent", "PantsTemplateContent", "ShirtTemplate", "PantsTemplate", "Graphic", "GraphicContent")


def _image_props(props):
    """props with every image Content given as an rbxassetid that names a Decal swapped for the image inside it
    (_decal_image, RB1340). (props, {prop: report} for each swap, error)."""
    if not isinstance(props, dict):
        return props, {}, None
    out, swaps = dict(props), {}
    for k, v in props.items():
        if k in IMAGE_PROPS and isinstance(v, str) and v.startswith("rbxassetid://"):
            img, rep, err = _decal_image(v)
            if err:
                return None, swaps, "%s: %s" % (k, err)
            if img != v:
                out[k] = img
                swaps[k] = rep
    return out, swaps, None


def _wait_fetch(content, seconds=30.0):
    """Poll ContentProvider:GetAssetFetchStatus for a template id until it ends. (status, seconds waited)."""
    t0 = time.time()
    status = None
    while time.time() - t0 < seconds:
        status = rbx_call("ContentProvider", "GetAssetFetchStatus", [content]).get("value")
        if status in ("Success", "Failure", "TimedOut"):
            break
        time.sleep(0.5)
    return status, round(time.time() - t0, 2)


# Where the clothing captures (camera 6 studs off, 1 above the pivot) put the torso and one leg, as fractions of the frame
PROBE_AT = {"Shirt": (0.5, 0.5), "Pants": (0.455, 0.75)}


def _box_mean(file, cx, cy, box=0.03):
    """Mean RGB of a small box around (cx, cy), fractions of a capture's width and height."""
    from mif_roblox import png as P
    w, h, rgba = P.decode_rgba(open(file, "rb").read())
    x0, x1, y0, y1 = int(w * (cx - box)), int(w * (cx + box)), int(h * (cy - box)), int(h * (cy + box))
    tot, n = [0, 0, 0], 0
    for y in range(y0, y1):
        row = y * w * 4
        for x in range(x0, x1):
            i = row + x * 4
            tot[0] += rgba[i]
            tot[1] += rgba[i + 1]
            tot[2] += rgba[i + 2]
            n += 1
    return [round(t / max(n, 1), 1) for t in tot]


DRAWN_DIFF = 12.0  # mean channel difference of the torso box, dressed against bare, above which the template is drawn


def _content_arg(v):
    """A clothing template as a property value: a PNG path becomes an EditableImage handle (nothing uploaded), a
    {handle} or an rbxassetid string goes as it is. (value, image report, error)."""
    if isinstance(v, dict) and set(v) == {"handle"}:
        return v, None, None
    if isinstance(v, str) and v.startswith("rbxassetid://"):
        return _decal_image(v)
    if isinstance(v, str) and os.path.isfile(v):
        img = rbx_image_preview(file=v)
        if not img.get("ok"):
            return None, img, "the image did not load: %s" % img.get("error")
        return {"handle": img["handle"]}, {k: img.get(k) for k in ("width", "height", "verified")}, None
    return None, None, "a template is a PNG path, {handle} or rbxassetid://..., got %r" % (v,)


@_reports
def rbx_classic_clothing(shirt: Any = None, pants: Any = None, color: list = None, mannequin: Any = None,
                         parent: Any = "Workspace.MifScratch", capture: bool = True) -> dict:
    "Roblox Studio: classic 2D clothing on a mannequin - shirt and pants templates (a PNG on this machine shown through an EditableImage without uploading, an EditableImage {handle}, or an rbxassetid) as Shirt.ShirtTemplateContent and Pants.PantsTemplateContent, Clothing.Color3 optional; read back on the body and captured front and back. An rbxassetid that names a Decal (what an image upload answers) is swapped for the image it holds, since a Decal id draws nothing; the template's fetch (ContentProvider:GetAssetFetchStatus) is waited for, and drawn means the front view with the clothing on differs from the same view with it set aside where it covers the body (torso for a shirt, a leg for pants). Roblox's template pages give the regions (128 x 128 front and back, 64 x 128 sides, 128 x 64 top and bottom, 64 x 64 limb ends); the overall image size is reported, never refused, because no page states one."
    if shirt is None and pants is None:
        return {"ok": False, "ran": False, "error": "give shirt, pants or both"}
    if mannequin is None:
        mq = rbx_mannequin(name="ClothingBody", parent=parent, position=[2100, 200, 2000])
        if not mq.get("ok"):
            return dict(mq, step="mannequin")
        mannequin = mq["path"]
    out, problems = {"ok": True, "mannequin": mannequin}, []
    made = {}
    for cls, prop, val in (("Shirt", "ShirtTemplateContent", shirt), ("Pants", "PantsTemplateContent", pants)):
        if val is None:
            continue
        content, img, err = _content_arg(val)
        if err:
            problems.append("%s: %s" % (cls, err))
            continue
        props = {prop: content}
        if color is not None:
            props["Color3"] = color
        mpparts = V.split_path(mannequin)
        c = rbx_create(cls, mpparts, name=_unique_name(mpparts, "Mif" + cls), props=props)
        out[cls.lower()] = {"path": c.get("path"), "ok": c.get("ok"), "image": img, "error": c.get("error")}
        if not c.get("ok"):
            problems.append("%s: %s" % (cls, c.get("error")))
        elif isinstance(content, dict):
            # Measured on 0.748: the property holds the EditableImage and reads it back, but the body does not show it.
            out[cls.lower()]["renders"] = False
            out.setdefault("notes", []).append(
                "%s: an EditableImage template is held by %s but Studio 0.748 does not draw it on the body (measured); an "
                "uploaded image (rbxassetid) is what shows" % (cls, prop))
        else:
            # The body draws the template only once Studio has fetched it: captured at once it showed the bare (even
            # black) body (measured, RB1340), so wait for the fetch before any capture.
            status, waited = _wait_fetch(content)
            out[cls.lower()].update(fetchStatus=status, fetchSeconds=waited)
            if status != "Success":
                problems.append("%s: Studio did not fetch %s (GetAssetFetchStatus %s after %ss)" % (cls, content, status, waited))
            made[cls] = c.get("path")
    if capture and not problems:
        time.sleep(1.5)  # the fetched template is composited onto the body over the next frames
        piv = rbx_call(mannequin, "GetPivot")
        p = ((piv.get("value") or {}).get("position")) or [2100, 200, 2000]
        cams = {"front": [p[0], p[1] + 1, p[2] - 6], "back": [p[0], p[1] + 1, p[2] + 6]}
        shots = {}
        for side, at in cams.items():
            s = rbx_screenshot(camera={"position": at, "look_at": [p[0], p[1], p[2]]}, max_width=640, name="clothing_" + side)
            shots[side] = {k: s.get(k) for k in ("ok", "file", "error")}
            if not s.get("ok"):
                problems.append("%s capture: %s" % (side, s.get("error")))
        out["captures"] = shots
        if made and not problems:
            # Seen on the body: the same front view with the clothing moved off the body, compared where it covers.
            # The bare view is taken after the dressed one, so the body itself has loaded in both.
            mparent = V.split_path(mannequin)[:-1] or ["Workspace"]
            aside = rbx_create("Folder", mparent, name=_unique_name(mparent, "MifClothingAside"))
            moved = {cls: rbx_reparent(cpath, aside.get("path")).get("ok") for cls, cpath in made.items()} if aside.get("ok") else {}
            time.sleep(1.0)
            bare = rbx_screenshot(camera={"position": cams["front"], "look_at": [p[0], p[1], p[2]]}, max_width=640,
                                  name="clothing_bare")
            for cls, cpath in made.items():
                back = rbx_reparent(aside.get("path", "") + "." + cpath.rsplit(".", 1)[-1], mannequin) if moved.get(cls) else None
                if not (back or {}).get("ok"):
                    problems.append("%s: not set aside or not put back on the body: %s" % (cls, (back or aside).get("error")))
            if aside.get("ok"):
                rbx_destroy(aside["path"])
            if not bare.get("ok"):
                problems.append("bare capture: %s" % bare.get("error"))
            else:
                out["captures"]["bare"] = {k: bare.get(k) for k in ("ok", "file", "error")}
                for cls in made:
                    cx, cy = PROBE_AT[cls]
                    a, b = _box_mean(shots["front"]["file"], cx, cy), _box_mean(bare["file"], cx, cy)
                    diff = round(sum(abs(x - y) for x, y in zip(a, b)) / 3, 1)
                    out[cls.lower()].update(dressedRGB=a, bareRGB=b, diff=diff, drawn=diff > DRAWN_DIFF)
                    if diff <= DRAWN_DIFF:
                        problems.append("%s: the front view with and without it differ by %s where it covers the body (needs "
                                        "> %s): the template is not drawn" % (cls, diff, DRAWN_DIFF))
    if problems:
        out.update(ok=False, error="; ".join(problems))
    # verified means the clothing is seen on the body; a template Studio does not draw cannot be
    out["verified"] = out["ok"] and not out.get("notes")
    return out


# the internal notes, from Roblox's fee and policy pages: upload fee and publishing advance in Robux. Kept as cited
# figures with their source, never as live prices: Roblox changes them, and the report says so.
LISTING_SOURCE = ("https://create.roblox.com/docs/art/marketplace/marketplace-fees-and-commissions and "
                  "https://create.roblox.com/docs/art/marketplace/marketplace-policy (read 2026-11-18)")
LAYERED_TYPES = {"JacketAccessory", "SweaterAccessory", "ShirtAccessory", "TShirtAccessory", "PantsAccessory",
                 "ShortsAccessory", "DressSkirtAccessory", "LeftShoeAccessory", "RightShoeAccessory"}
ADVANCE_1500 = {"Hat", "FaceAccessory", "DynamicHead"}
BODY_TYPES = {"Torso", "LeftArm", "RightArm", "LeftLeg", "RightLeg"}


def listing_report(item_type, checks, emissive=False, name=None):
    """The Marketplace hand-off for one item: what it is, whether every check passed, fees and account requirements.
    checks is a bl_roblox_ugc_check or rbx_accessory_build answer. Pure: nothing is sent anywhere."""
    if item_type in ("Shirt", "Pants", "TShirt"):
        advance, group = 10, "classic clothing"
    elif item_type in LAYERED_TYPES:
        advance, group = 600, "layered clothing"
    elif item_type in ADVANCE_1500:
        advance, group = 1500, "hats, faces and heads"
    elif item_type in BODY_TYPES:
        advance, group = 2500, "a body (per bundle, not per part)"
    else:
        advance, group = 1000, "hair and most rigid accessories"
    passed = bool(isinstance(checks, dict) and checks.get("ok") is not False and
                  (checks.get("pass") is True or checks.get("verified") is True))
    failed = (checks or {}).get("failed") or ([] if passed else [(checks or {}).get("error") or "no passing check given"])
    return {
        "item": name or (checks or {}).get("object") or (checks or {}).get("accessory"),
        "type": item_type, "ready": passed, "notReadyBecause": None if passed else failed,
        "checks": (checks or {}).get("checks") or {k: v for k, v in (checks or {}).items() if k in ("report", "weld", "attachmentGap")},
        "rulesBuild": (checks or {}).get("build"),
        "uploadFeeRobux": 500 if emissive else 80, "publishingAdvanceRobux": advance * (5 if emissive else 1),
        "advanceGroup": group + (" (emissive: five times)" if emissive else ""),
        "accountNeeds": ["ID verification (or a linked parent account that has it) to upload",
                         "2-step verification to publish for sale",
                         "Roblox Plus or Premium 1000/2200 on the creator's or group owner's account, also to stay on sale"],
        "feesNote": "figures as Roblox's pages gave them when read; Roblox changes them, so check before paying. The upload fee "
                    "is generally not refunded on rejection.",
        "source": LISTING_SOURCE,
        "bridgeDoes": "nothing on the Marketplace: it never uploads for sale, sets a price or touches Robux; the person "
                      "uploads and publishes in Studio and on the Creator Dashboard",
    }


@_reports
def rbx_listing_report(item_type: str, checks: dict, emissive: bool = False, name: str = None, folder: str = None) -> dict:
    "Roblox: the Marketplace hand-off for one avatar item - its type, whether every check passed (the answer of bl_roblox_ugc_check or rbx_accessory_build), the upload fee and publishing advance in Robux from Roblox's fee page as last read, and the account requirements (ID and 2-step verification, Plus or Premium). Written as JSON beside the captures. An item with a failed check is marked not ready, with why. It never uploads for sale, sets a price or touches Robux."
    if not isinstance(item_type, str) or not item_type:
        return {"ok": False, "ran": False, "error": "item_type is an asset type name (Hat, JacketAccessory, ...)"}
    rep = listing_report(item_type, checks, emissive=emissive, name=name)
    folder = folder or os.path.join(captures_dir(), "..", "listings")
    try:
        os.makedirs(folder, exist_ok=True)
        path = os.path.abspath(os.path.join(folder, "%s_%s.json" % (re.sub(r"[^\w-]", "_", str(rep["item"] or item_type)),
                                                                     time.strftime("%Y%m%d_%H%M%S"))))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    except OSError as exc:
        return {"ok": False, "ran": True, "error": "the report could not be written: %s" % exc, "report": rep}
    return {"ok": True, "file": path, "ready": rep["ready"], "report": rep}


@_reports
def rbx_surface_appearance(target: Any, color: Any = None, normal: Any = None, roughness: Any = None, metalness: Any = None,
                           alpha_mode: str = None, name: str = "MifLook", ready_wait: float = 45) -> dict:
    "Roblox Studio: PBR maps on a MeshPart - a SurfaceAppearance under target with ColorMap, NormalMap, RoughnessMap and MetalnessMap, each an rbxassetid (an uploaded image) or a PNG on this machine shown through an EditableImage without uploading; alpha_mode Overlay, Transparency or TintMask. An rbxassetid that names a Decal (what an image upload answers) is swapped for the image inside it. A new image draws plain grey until Studio has made its copy: each map's fetch status is waited for up to ready_wait seconds (default 45; measured ready in about 30 s, seen once over 20 minutes), and a map still loading, or Output saying a map is 'being generated', is reported under notReady; Output saying Studio was unable to generate the appearance turns ok false. Read back: every map held, and ContentProvider:GetAssetFetchStatus for each asset id."
    maps = {"Color": color, "Normal": normal, "Roughness": roughness, "Metalness": metalness}
    if not isinstance(ready_wait, (int, float)) or isinstance(ready_wait, bool) or not 0 <= ready_wait <= 600:
        return {"ok": False, "ran": False, "error": "ready_wait is 0-600 seconds"}
    if not any(v is not None for v in maps.values()):
        return {"ok": False, "ran": False, "error": "give at least one of color, normal, roughness, metalness"}
    parts, err = _target(target)
    if err:
        return err
    props, report, problems = {}, {}, []
    for k, v in maps.items():
        if v is None:
            continue
        content, img, cerr = _content_arg(v)
        if cerr:
            return {"ok": False, "ran": False, "error": "%s map: %s" % (k, cerr)}
        props[k + "MapContent"] = content
        report[k] = {"source": v if isinstance(v, str) else "handle", "image": img}
        if isinstance(content, str) and content != v:
            report[k]["content"] = content
    if alpha_mode is not None:
        props["AlphaMode"] = alpha_mode
    sa = rbx_create("SurfaceAppearance", parts, name=_unique_name(parts, name), props=props)
    if not sa.get("ok"):
        return sa
    # Fetch status None means not asked for yet: Studio fetches an image when something draws it (measured, RB65), so a
    # map is judged only on Failure or TimedOut; None and Loading are reported as they are.
    t0 = time.time()
    pending = {}
    for k, v in maps.items():
        v = props.get(k + "MapContent", v)  # the image a Decal id was swapped for
        if isinstance(v, str) and v.startswith("rbxassetid://"):
            pending[k] = v
    # A new image draws plain grey on a mesh until Studio has its SurfaceAppearance copy: measured 2026-11-20, fetch
    # status Loading and grey at 2 s, Success and drawn by 33 s (RB1312); a game lane saw one stay grey for 20+ minutes
    # with Output saying "Asset content representation is being generated". So each map is waited for,
    # up to ready_wait seconds, and Studio's own words are read from the Output.
    while True:
        for k, v in pending.items():
            st = rbx_call("ContentProvider", "GetAssetFetchStatus", [v])
            report[k]["fetchStatus"] = st.get("value") if st.get("ok") else {"error": st.get("error")}
        waiting = [k for k in pending if report[k].get("fetchStatus") in (None, "None", "Loading")]
        if not waiting or time.time() - t0 >= ready_wait:
            break
        time.sleep(1.0)
    for k, v in pending.items():
        if report[k]["fetchStatus"] in ("Failure", "TimedOut") or isinstance(report[k]["fetchStatus"], dict):
            problems.append("%s map %s fetch status %s" % (k, v, report[k]["fetchStatus"]))
    said = _generation_lines(t0 - 5)
    not_ready = [k for k in pending if report[k].get("fetchStatus") in (None, "None", "Loading")]
    if any("unable to generate" in ln.lower() for ln in said):
        problems.append("Studio dropped the appearance: %s" % "; ".join(said)[:300])
    out = {"ok": not problems, "verified": sa.get("verified") and not problems, "path": sa.get("path"), "maps": report,
           "readyWaitSeconds": round(time.time() - t0, 1)}
    if not_ready or said:
        out["notReady"] = {"maps": not_ready, "studioSays": said,
                           "note": "the appearance is held but may draw plain grey until Studio has made the map's copy; "
                                   "read it again later (rbx_surface_appearance's fetch status, or the Output)"}

    if problems:
        out["error"] = "; ".join(problems)
    return out


GENERATION_WORDS = ("being generated", "unable to generate", "asset type does not match")


def _generation_lines(since):
    """Studio's Output lines since a time (os time) that say a map is still being made or was dropped."""
    c = rbx_console(limit=300)
    rows = c.get("entries") if isinstance(c, dict) else None
    out = []
    for e in rows or []:
        msg = str((e or {}).get("message") or "")
        ts = (e or {}).get("timestamp")
        if any(w in msg.lower() for w in GENERATION_WORDS) and (not isinstance(ts, (int, float)) or ts >= since):
            out.append(msg[:300])
    return out


def _count(v):
    """A list's length as the plugin sent it: a truncated list carries its total."""
    if isinstance(v, dict) and "total" in v:
        return v["total"]
    return len(v) if isinstance(v, list) else None


@_reports
def rbx_insert_asset(asset_id: int, parent: Any = "Workspace.MifScratch", method: str = "LoadAsset", version_id: int = None) -> dict:
    "Roblox Studio: load one of the person's own assets under parent (default Workspace.MifScratch) with InsertService:LoadAsset, AssetService:LoadAssetAsync, or InsertService:LoadAssetVersion (version_id), and read back: its path and class, the asset's latest version (InsertService:GetLatestAssetVersionAsync) and ContentProvider:GetAssetFetchStatus for every MeshId and TextureID under it."
    if not isinstance(asset_id, int) or isinstance(asset_id, bool) or asset_id <= 0:
        return {"ok": False, "ran": False, "error": "asset_id is a positive whole number"}
    if method not in ("LoadAsset", "LoadAssetAsync", "LoadAssetVersion"):
        return {"ok": False, "ran": False, "error": "method is LoadAsset, LoadAssetAsync or LoadAssetVersion"}
    if method == "LoadAssetVersion" and not (isinstance(version_id, int) and version_id > 0):
        return {"ok": False, "ran": False, "error": "LoadAssetVersion needs version_id"}
    pparts, err = _target(parent, "parent")
    if err:
        return err
    svc = "AssetService" if method == "LoadAssetAsync" else "InsertService"
    loaded = rbx_call(svc, method, [version_id if method == "LoadAssetVersion" else asset_id], timeout=120)
    val = loaded.get("value")
    h = val.get("handle") if isinstance(val, dict) else None
    if not h:
        return {"ok": False, "ran": True, "step": method, "error": loaded.get("error") or "%s returned %r" % (method, val)}
    path, perr = _place_handle(h, pparts, "Asset%d" % asset_id)
    if perr:
        return {"ok": False, "ran": True, "step": "place", "handle": h, "error": perr}
    out = {"ok": True, "verified": True, "path": path, "className": val.get("className"), "method": method, "assetId": asset_id}
    latest = rbx_call("InsertService", "GetLatestAssetVersionAsync", [asset_id], timeout=60)
    out["latestVersion"] = latest.get("value") if latest.get("ok") else {"error": latest.get("error")}
    fetch = {}
    for cls, prop in (("MeshPart", "MeshId"), ("MeshPart", "TextureID"), ("Decal", "Texture"), ("SurfaceAppearance", "ColorMap")):
        for m in rbx_find(class_name=cls, root=path, limit=200).get("matches") or []:
            g = rbx_get_props(m["path"], [prop])
            cid = (g.get("values") or {}).get(prop) if g.get("ok") else None
            if isinstance(cid, str) and cid:
                st = rbx_call("ContentProvider", "GetAssetFetchStatus", [cid])
                fetch[cid] = st.get("value") if st.get("ok") else {"error": st.get("error")}
    out["fetchStatus"] = fetch
    if not rbx_get_props(path, ["Name"]).get("ok"):
        out.update(ok=False, verified=False, error="loaded, then gone before the tool returned")
    return out


@_reports
def rbx_mesh_inspect(target: Any, faces_of: int = 0) -> dict:
    "Roblox Studio: what a MeshPart's mesh holds, through AssetService:CreateEditableMeshAsync on its MeshContent: triangle, vertex, UV and normal counts, bones and FACS poses, the mesh size (EditableMesh:GetSize), and MeshPart:GetSurfaceSlotsAsync; with faces_of > 0, the vertex ids of that many faces. The EditableMesh is released after. Read only for the place."
    parts, err = _target(target)
    if err:
        return err
    g = rbx_get_props(target, ["MeshContent", "MeshId", "Size"])
    if not g.get("ok"):
        return g
    vals = g.get("values") or {}
    content = vals.get("MeshId") or vals.get("MeshContent")
    if not content:
        return {"ok": False, "ran": False, "error": "%s has no mesh content" % target}
    out = {"ok": True, "target": target, "meshId": vals.get("MeshId"), "partSize": vals.get("Size")}
    counts = _mesh_counts(content, faces_of)
    if not counts.get("ok"):
        return dict(counts, target=target)
    out.update({k: v for k, v in counts.items() if k != "ok"})
    slots = rbx_call(target, "GetSurfaceSlotsAsync", timeout=60)
    out["surfaceSlots"] = slots.get("value") if slots.get("ok") else {"error": slots.get("error")}
    return out


def _mesh_counts(content, faces_of=0):
    """Open mesh content as an EditableMesh and count it: triangles, vertices, UVs, normals, bones, FACS poses, size.
    The EditableMesh is outside the place and released after."""
    em = rbx_call("AssetService", "CreateEditableMeshAsync", [content], timeout=120)
    ev = em.get("value")
    h = ev.get("handle") if isinstance(ev, dict) else None
    if not h:
        return {"ok": False, "ran": True, "step": "CreateEditableMeshAsync", "content": content,
                "error": em.get("error") or "returned %r" % (ev,)}
    out = {"ok": True}
    try:
        for key, method in (("triangles", "GetFaces"), ("vertices", "GetVertices"), ("uvs", "GetUVs"), ("normals", "GetNormals"),
                            ("bones", "GetBones"), ("facsPoses", "GetFacsPoses")):
            r = rbx_call({"handle": h}, method)
            out[key] = _count(r.get("value")) if r.get("ok") else {"error": r.get("error")}
        sz = rbx_call({"handle": h}, "GetSize")
        out["meshSize"] = sz.get("value") if sz.get("ok") else {"error": sz.get("error")}
        if faces_of and isinstance(faces_of, int) and faces_of > 0:
            fr = rbx_call({"handle": h}, "GetFaces")
            ids = fr.get("value") if isinstance(fr.get("value"), list) else (fr.get("value") or {}).get("v", [])
            out["faces"] = {}
            for fid in (ids or [])[:min(faces_of, 50)]:
                fv = rbx_call({"handle": h}, "GetFaceVertices", [fid])
                out["faces"][str(fid)] = fv.get("value")
    finally:
        rbx_release_handle(h)  # the EditableMesh is outside the place: releasing destroys only it
    return out


@_reports
def rbx_cage_inspect(target: Any) -> dict:
    "Roblox Studio: the cages under target (a body, a body part, or layered clothing) read as meshes - for every WrapTarget and WrapLayer, AssetService:CreateEditableMeshAsync on its CageMeshContent (and a WrapLayer's ReferenceMeshContent): vertex, triangle and UV counts and size, with CageOrigin. A cage whose content will not open says so with Studio's error. Read only."
    parts, err = _target(target)
    if err:
        return err
    wraps = []
    for cls in ("WrapTarget", "WrapLayer"):
        wraps += [dict(m, cls=cls) for m in rbx_find(class_name=cls, root=target, limit=100).get("matches") or []]
    if not wraps:
        return {"ok": False, "ran": True, "error": "no WrapTarget or WrapLayer under %s" % target}
    out, failed = {"ok": True, "cages": {}}, []
    for w in wraps:
        props = ["CageMeshContent", "CageOrigin"] + (["ReferenceMeshContent", "Order"] if w["cls"] == "WrapLayer" else [])
        g = rbx_get_props(w["path"], props)
        vals = g.get("values") or {}
        row = {"class": w["cls"], "cageOrigin": (vals.get("CageOrigin") or {}).get("position")}
        for key, prop in (("cage", "CageMeshContent"), ("reference", "ReferenceMeshContent")):
            c = vals.get(prop)
            if not c:
                continue
            m = _mesh_counts(c)
            row[key] = {k: m.get(k) for k in ("vertices", "triangles", "uvs", "meshSize", "error") if m.get(k) is not None}
            row[key]["content"] = c
            if not m.get("ok"):
                failed.append("%s %s: %s" % (w["path"], prop, m.get("error")))
        out["cages"][w["path"]] = row
    if failed:
        out.update(ok=False, error="%d cage(s) did not open: %s" % (len(failed), "; ".join(failed[:5])))
    out["count"] = len(wraps)
    return out


from mif_roblox.domain_scripts import *  # noqa: E402,F401,F403 - lane 2: scripts and diagnostics
from mif_roblox.domain_release import *  # noqa: E402,F401,F403 - lane 2: release machinery, coverage
from mif_roblox.domain_world import *  # noqa: E402,F401,F403 - batches 6 and 7
from mif_roblox.domain_ui_physics import *  # noqa: E402,F401,F403 - lane 4: UI, physics, camera and input
from mif_roblox.domain_anim_data import *  # noqa: E402,F401,F403 - lane 5: animation, audio, data, editor
from mif_roblox.domain_core import *  # noqa: E402,F401,F403 - lane R2: attributes, tags, script create and patch
from mif_roblox.domain_geo import *  # noqa: E402,F401,F403 - lane R1: geometry; also gives rbx_reparent, rbx_set_props, rbx_destroy, rbx_release_handle their {handle} cases
from mif_roblox.domain_assets import *  # noqa: E402,F401,F403 - assets already on the account: the imported MeshPart's pivot and axis mapping
from mif_roblox.domain_avatar import *  # noqa: E402,F401,F403 - avatar bodies from the person's own parts: the R15 rig and Roblox's budgets
from mif_roblox.domain_controller import *  # noqa: E402,F401,F403 - controller support: a gamepad audit of the UI and actions
from mif_roblox.domain_logic import *  # noqa: E402,F401,F403 - game logic checks: saves, remotes, odds and config values
from mif_roblox.domain_store import *  # noqa: E402,F401,F403 - the experience on Roblox's side: settings and Creator Store listings
from mif_roblox.domain_showcase import *  # noqa: E402,F401,F403 - showing models: a cutout on a key colour, a gallery of a set
from mif_roblox.domain_kit import *  # noqa: E402,F401,F403 - modeled kits over a grey box and the mesh pipeline before an upload

TOOLS = {name: fn for name, fn in sorted(globals().items()) if name.startswith("rbx_") and callable(fn)}

# What the MCP server's pack machinery reads, as the Unity backend's PACK. Not in packs.json, not in the default list.
PACK = {
    "title": "Roblox Studio",
    "description": "Drive a live Roblox Studio through typed, read-back calls generated from Studio's own API dump: the "
                   "instance tree, create, clone, destroy, reparent, properties, selection, undo groups, script source and "
                   "the Output. Needs the MifBridge Studio plugin, connected to the local daemon.",
    "backend": "roblox",
    "tools": sorted(TOOLS),
    "rules": [
        {"rule": "Ask rbx_describe_member before naming a class or member; never guess one. A member a plugin cannot use is refused with its security tag, and there is no way around it.",
         "tools": ["rbx_describe_member"]},
        {"rule": "ok on a write means Studio was asked afterwards and holds the value (verified: true). Each write is one undo step; use rbx_undo_group to make several one.",
         "tools": ["rbx_set_props", "rbx_create", "rbx_undo_group"]},
        {"rule": "outcomeUnknown means the plugin took the call and did not answer in time: read the target before sending anything else.",
         "tools": ["rbx_status", "rbx_get_props"]},
        {"rule": "Read a script, then write it with expect_sha256 from that read, so a person's edit in between is never overwritten.",
         "tools": ["rbx_script_read", "rbx_script_write"]},
    ],
}
