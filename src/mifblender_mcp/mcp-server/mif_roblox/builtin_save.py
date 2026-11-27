"""Save the open place to its own local file through Roblox's built-in Studio MCP server, with no dialog (opt-in).

WHY THIS EXISTS. No member a plugin can reach writes a local place file without a dialog on 0.748: ExportPlace opened an
OBJ export dialog (RB43), SavePlaceAsync and DataModel:SavePlace write to Roblox. Roblox's own Studio MCP server has
save_place, which is Ctrl+S: a place opened from a .rbxl/.rbxlx is written back to that file, in 0.1 s, with no window
(RB1422). rbx_place_publish then sends that Studio-written file to a place (RB1969: a hand-written file is refused).

WHAT IT REVERSES. Until 2026-11-22 MifBridge only named that server (decision 9: the setup offer) and never called it.
This module calls exactly one of its tools, save_place, plus list_roblox_studios and inspect_instance to find the Studio,
and only when the caller passes use_builtin_mcp true. Nothing else of that server is ever called.

HOW THE STUDIO IS PICKED (measured, RB2900-RB2902): the server is a proxy (StudioMCP.exe, started by mcp.bat over stdio);
every Studio of the user whose "Enable Studio as MCP server" switch is on connects to it within about 40 s, and each call
takes a studio_id from list_roblox_studios, whose rows name the place file by its base name only. So the Studio is the
one row whose file is the open place's base name, and it is proved to be MifBridge's Studio before anything is saved:
MifBridge makes a Folder with a one-time name in ServerStorage, inspect_instance must find it in that Studio, the Folder
is removed and must be gone (inspect_instance again), and only then save_place runs. Two rows with the same base name
are refused (ambiguous) and neither is inspected: another session's Studio is never read or written.

The proxy is the caller's own process: closing its stdin ends it (RB1422); anything of it still alive after that is
ended by PID, only the PIDs this module started. Its auth token is never opened.

NEVER RAISE: every failure is a dict with ok False.
"""
import json
import os
import subprocess
import threading
import time

LIST_EVERY = 2.0        # seconds between list_roblox_studios polls while the Studios connect
SETTLE = 6.0            # a second list this long after the first match, so a late Studio with the same file is seen
CALL_WAIT = 60.0
# The only tools of that server MifBridge ever calls (decision 9 amended for place saving only, Mifsopo 2026-11-22).
# Any other name is refused in Client.tool before anything is written to the proxy.
ALLOWED_TOOLS = ("list_roblox_studios", "inspect_instance", "save_place")


def launcher():
    from mif_roblox.domain_release import ROBLOX_DIR  # one place names Roblox's folder
    p = os.path.join(ROBLOX_DIR, "mcp.bat")
    return p if os.path.isfile(p) else None


class Client:
    """A stdio JSON-RPC client of the built-in server: one proxy process for one save."""

    def __init__(self, bat):
        self.p = subprocess.Popen(["cmd.exe", "/c", bat], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.lines, self.errs, self.nid = [], [], 0
        self.lock = threading.Lock()
        for src, dst in ((self.p.stdout, self.lines), (self.p.stderr, self.errs)):
            threading.Thread(target=self._pump, args=(src, dst), daemon=True).start()
        self.pids = self._tree()

    def _pump(self, src, dst):
        try:
            for ln in iter(src.readline, b""):
                with self.lock:
                    dst.append(ln.decode("utf-8", "replace").rstrip())
        except (OSError, ValueError):
            return

    def _tree(self, wait=5.0):
        """Our cmd.exe and what it started (StudioMCP.exe), read once it appears: these are the only PIDs ever ended."""
        pids = [self.p.pid]
        try:
            import psutil
            t0 = time.time()
            while time.time() - t0 < wait:
                kids = [c.pid for c in psutil.Process(self.p.pid).children(recursive=True)]
                if kids:
                    return pids + kids
                time.sleep(0.2)
        except Exception:  # noqa: BLE001 - psutil missing or the process gone: the cmd.exe PID alone
            pass
        return pids

    def request(self, method, params=None, wait=CALL_WAIT):
        self.nid += 1
        msg = {"jsonrpc": "2.0", "id": self.nid, "method": method}
        if params is not None:
            msg["params"] = params
        try:
            self.p.stdin.write((json.dumps(msg) + "\n").encode())
            self.p.stdin.flush()
        except (OSError, ValueError) as exc:
            return {"error": {"message": "the proxy's stdin is closed: %r" % exc}}
        want, t0 = self.nid, time.time()
        while time.time() - t0 < wait:
            with self.lock:
                for i, ln in enumerate(self.lines):
                    try:
                        m = json.loads(ln)
                    except ValueError:
                        continue
                    if isinstance(m, dict) and m.get("id") == want:
                        del self.lines[i]
                        return m
            if self.p.poll() is not None:
                return {"error": {"message": "the proxy exited (code %s)" % self.p.returncode}}
            time.sleep(0.1)
        return {"error": {"message": "no answer to %s in %s s" % (method, wait)}}

    def notify(self, method):
        try:
            self.p.stdin.write((json.dumps({"jsonrpc": "2.0", "method": method}) + "\n").encode())
            self.p.stdin.flush()
        except (OSError, ValueError):
            pass

    def tool(self, name, args, wait=CALL_WAIT):
        refused = refuse_tool(name)
        if refused:
            return refused
        return self.request("tools/call", {"name": name, "arguments": args}, wait)

    def stderr_tail(self, n=8):
        with self.lock:
            return list(self.errs[-n:])

    def close(self):
        """Close stdin (the proxy exits on it, RB1422), then end any of our own PIDs still alive. Returns what was left.
        The tree is read again first: StudioMCP.exe can start after the first child of cmd.exe was seen (RB2902)."""
        try:
            import psutil
            kids = [c.pid for c in psutil.Process(self.p.pid).children(recursive=True)]
            self.pids = self.pids + [k for k in kids if k not in self.pids]
        except Exception:  # noqa: BLE001 - cmd.exe already gone: the PIDs read at start
            pass
        try:
            self.p.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        ended = []
        try:
            import psutil
            for pid in self.pids:
                try:
                    pr = psutil.Process(pid)
                    if pr.is_running():
                        pr.kill()
                        ended.append(pid)
                except psutil.NoSuchProcess:
                    continue
        except Exception:  # noqa: BLE001
            if self.p.poll() is None:
                self.p.kill()
                ended.append(self.p.pid)
        return {"proxyPids": self.pids, "endedByPid": ended}


def refuse_tool(name):
    """None for a tool MifBridge may call on the built-in server, else an error reply (sent nowhere)."""
    if name in ALLOWED_TOOLS:
        return None
    return {"error": {"message": "MifBridge calls only %s on Roblox's built-in Studio MCP server; %r is refused" % (
        ", ".join(ALLOWED_TOOLS), name)}, "refusedRule": "builtin-tool"}


def texts(reply):
    """The text parts of a tools/call reply, and whether it is an error."""
    res = (reply or {}).get("result") if isinstance(reply, dict) else None
    if not isinstance(res, dict):
        err = (reply or {}).get("error") if isinstance(reply, dict) else None
        return [str((err or {}).get("message") or err or "no result")], True
    parts = [c.get("text") for c in res.get("content") or [] if isinstance(c, dict) and isinstance(c.get("text"), str)]
    return parts, bool(res.get("isError"))


def last_json(parts):
    """The last text part that is a JSON object, or None."""
    for t in reversed(parts or []):
        try:
            v = json.loads(t)
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict):
            return v
    return None


def found(reply, name):
    """True when inspect_instance answered the instance named name, False when it answered an error, None if unclear."""
    parts, is_err = texts(reply)
    doc = last_json(parts)
    if not is_err and isinstance(doc, dict) and doc.get("name") == name:
        return True
    if is_err or any(str(t).lstrip().lower().startswith("error") for t in parts):
        return False
    return None


def studios(client):
    parts, is_err = texts(client.tool("list_roblox_studios", {}))
    doc = last_json(parts)
    rows = (doc or {}).get("studios") if not is_err else None
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else None


def pick(client, base, wait_seconds, clock=time.time, sleep=time.sleep):
    """(row, None, seen) for the one Studio whose file is base, or (None, refusal, seen). Polls while the Studios connect,
    then lists once more after SETTLE so a second Studio with the same file name is seen before choosing."""
    t0, seen = clock(), []
    while True:
        rows = studios(client)
        if rows is not None:
            seen = rows
            mine = [r for r in rows if r.get("file") == base]
            if mine:
                sleep(SETTLE)
                again = studios(client)
                if again is not None:
                    seen = again
                    mine = [r for r in again if r.get("file") == base]
                if len(mine) == 1:
                    return mine[0], None, seen
                if len(mine) > 1:
                    return None, {"refusedRule": "ambiguous", "error": "%d Studios have a place file named %s open; the "
                                  "built-in server names files by base name only, so neither is read or saved: give the "
                                  "place a file name no other open Studio has" % (len(mine), base)}, seen
        if clock() - t0 >= wait_seconds:
            return None, {"refusedRule": "not-listed", "error": "no Studio with %s open joined Roblox's built-in MCP server "
                          "in %s s (%d listed): its 'Enable Studio as MCP server' switch may be off, or this Studio has "
                          "not connected yet" % (base, wait_seconds, len(seen))}, seen
        sleep(LIST_EVERY)


def save(place_file, make_nonce, drop_nonce, wait_seconds=90, client_factory=None, bat=None):
    """Save the Studio MifBridge is connected to, whose open place is place_file, through the built-in server.

    make_nonce() -> (folder name, error): makes ServerStorage.<name> in MifBridge's Studio. drop_nonce(name) -> error or
    None: removes it. Returns {ok, studioId, saved: {status, location, filePath}, proof, proxy, ...}."""
    base = os.path.basename(place_file)
    bat = bat or launcher()
    if not bat:
        return {"ok": False, "ran": False, "refusedRule": "no-launcher",
                "error": "Roblox's built-in MCP launcher (%LOCALAPPDATA%/Roblox/mcp.bat, installed with Studio) is not there"}
    try:
        client = (client_factory or Client)(bat)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "ran": False, "error": "could not start the built-in server's proxy: %r" % exc}
    out = {"ok": False, "ran": False, "file": place_file}
    nonce = None
    try:
        init = client.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                             "clientInfo": {"name": "mifbridge-place-save", "version": "1"}})
        if not isinstance(init, dict) or "result" not in init:
            out["error"] = "the built-in server did not initialize: %s" % json.dumps(init)[:300]
            return out
        out["server"] = (init["result"].get("serverInfo") or {})
        client.notify("notifications/initialized")
        row, bad, seen = pick(client, base, wait_seconds)
        out["studiosListed"] = len(seen)
        if bad:
            out.update(bad)
            return out
        sid = row.get("id")
        out.update(studioId=sid, studioMode=row.get("mode"), persistedToCloud=row.get("persistedToCloud"))
        if row.get("mode") not in (None, "Edit"):
            out.update(refusedRule="busy", error="that Studio is in %s mode, not Edit: stop the play test first" % row.get("mode"))
            return out
        nonce, err = make_nonce()
        if err:
            out.update(error="could not make the identity folder in MifBridge's Studio: %s" % err)
            return out
        path = "ServerStorage.%s" % nonce
        seen_there = found(client.tool("inspect_instance", {"path": path, "studio_id": sid}), nonce)
        out["proof"] = {"folder": path, "seenBeforeSave": seen_there}
        derr = drop_nonce(nonce)
        nonce = None if not derr else nonce
        if seen_there is not True:
            out.update(refusedRule="not-this-studio", error="the Studio listed with %s did not show the folder MifBridge "
                       "made in its own Studio (%s): it is not MifBridge's Studio, so nothing was saved" % (base, path))
            return out
        if derr:
            out.update(error="the identity folder could not be removed (%s); nothing was saved" % derr)
            return out
        gone = found(client.tool("inspect_instance", {"path": path, "studio_id": sid}), path.split(".")[-1])
        out["proof"]["goneBeforeSave"] = gone is False
        if gone is not False:
            out.update(error="the identity folder still shows in that Studio after its removal; nothing was saved")
            return out
        t0 = time.time()
        reply = client.tool("save_place", {"studio_id": sid}, wait=120)
        out["saveSeconds"] = round(time.time() - t0, 2)
        out["ran"] = True
        parts, is_err = texts(reply)
        doc = last_json(parts) or {}
        out["saved"] = {k: doc.get(k) for k in ("status", "location", "filePath", "message")}
        if is_err or doc.get("status") != "Saved":
            out.update(error="save_place answered %s: %s" % (doc.get("status") or "an error", " ".join(parts)[:300]))
            if doc.get("status") == "NeedsUserAction":
                out["refusedRule"] = "needs-person"
            elif doc.get("status") == "Busy":
                out["refusedRule"] = "busy"
            return out
        fp = doc.get("filePath")
        if not (isinstance(fp, str) and os.path.normcase(os.path.abspath(fp)) == os.path.normcase(os.path.abspath(place_file))):
            out.update(error="save_place saved to %s, not %s" % (fp, place_file))
            return out
        out["ok"] = True
        return out
    except Exception as exc:  # noqa: BLE001
        out.update(ok=False, error="the save through the built-in server failed: %r" % exc)
        return out
    finally:
        if nonce:
            drop_nonce(nonce)
        try:
            out["stderr"] = client.stderr_tail()
        except Exception:  # noqa: BLE001
            pass
        try:
            out["proxy"] = client.close()
        except Exception as exc:  # noqa: BLE001
            out["proxy"] = {"error": repr(exc)}
