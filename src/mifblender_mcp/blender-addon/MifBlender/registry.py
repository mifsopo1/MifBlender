"""Where each running MifBlender listens - a small per-user file the MCP server reads to find it (2026-10-12).

WHY: a MifBlender listened on :8802 (its port preference) while the MCP server, which reads MIF_BLENDER_PORT once at its
start and otherwise tries 8792, answered "Blender backend unreachable" to every bl_* call - a Blender running and ready,
and no way to reach it without restarting the server. Each listener now writes one entry here when it binds and removes
it when it stops: port, pid, Blender version, add-on version, the open .blend, background or not. The MCP server's
bl_status lists the live ones (pid alive and a handshake answered) and switches to one when asked by port or pid; it never
picks one by itself when several are live.

THE FILE, beside the call journal (journal.py):
  Windows  %LOCALAPPDATA%\\MifBridge\\Blender\\listeners.json
  macOS    ~/Library/Logs/MifBridge/Blender/listeners.json
  other    $XDG_STATE_HOME/mifbridge/blender/listeners.json   (~/.local/state when XDG_STATE_HOME is unset)
  MIF_BLENDER_LISTENERS overrides it (the tests use that).
tools/mcp-server/server.py's _bl_listeners_path() spells the same rule - the two must agree (test_bl_listeners checks).

No bpy here: the module is pure so the tests can import it without Blender. A write is all-or-nothing (a temporary file
renamed over the old one), retried briefly when another process holds the file, and never raises - losing an entry costs
discovery, never the listener.
"""
from __future__ import annotations

import json
import os
import sys
import time


def listeners_path():
    override = (os.environ.get("MIF_BLENDER_LISTENERS") or "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        return os.path.join(base, "MifBridge", "Blender", "listeners.json")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Logs", "MifBridge", "Blender", "listeners.json")
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(home, ".local", "state")
    return os.path.join(base, "mifbridge", "blender", "listeners.json")


def read(path=None):
    """Every entry in the file, as written (live or not - the reader checks). [] when there is no file or it is unreadable."""
    p = path or listeners_path()
    try:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    rows = data.get("listeners") if isinstance(data, dict) else None
    return [r for r in rows or [] if isinstance(r, dict) and isinstance(r.get("port"), int)]


def _write(rows, path=None):
    p = path or listeners_path()
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    except OSError:
        return False
    tmp = "%s.%d.tmp" % (p, os.getpid())
    body = json.dumps({"version": 1, "listeners": rows}, indent=1, sort_keys=True)
    for _ in range(10):
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(body)
            os.replace(tmp, p)
            return True
        except OSError:
            time.sleep(0.05)   # another process has the file open for a moment (Windows refuses the rename then)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return False


def register(port, pid, **info):
    """Write this listener's entry, replacing any earlier one for the same port (a port has one listener)."""
    try:
        rows = [r for r in read() if r.get("port") != int(port) and r.get("pid") != int(pid)]
        entry = {"port": int(port), "pid": int(pid), "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        entry.update({k: v for k, v in info.items() if v is not None})
        rows.append(entry)
        return _write(rows)
    except Exception:    # discovery must never cost the listener
        return False


def unregister(port, pid):
    """Remove this listener's entry (only its own: same port AND pid)."""
    try:
        rows = read()
        keep = [r for r in rows if not (r.get("port") == int(port) and r.get("pid") == int(pid))]
        return _write(keep) if len(keep) != len(rows) else True
    except Exception:
        return False
