"""The call journal - every bridge call this Blender runs, on disk BEFORE it runs.

WHY IT EXISTS, 2026-09-21. A user asked for a log of the calls on the Blender side, like the UE side has,
because a model can end up using generic Blender access instead of the bridge. The UE plugin has kept a journal since an editor crash nobody could attribute
(Source/MifBridge/Private/MifBridgeJournal.cpp, PM-013). This is its Blender twin; where the two worlds differ, the
choice is written down here.

THE ONE PROPERTY THAT MATTERS is the UE file's: the `start` record has left this process before the op runs, so a
Blender that dies inside a call still names it. A `start` with no `end` is the call it died in, and a `session` with no
`shutdown` is a Blender that died rather than being closed. The absence is the finding.

RECORDS, one JSON object per line, with UE's field names where the two overlap (t, ev, ep, ok, ms, err):
  session   t ev pid proc port blender addon sourceHash background generative
                                                                the server started (or the journal was turned back
                                                                on); proc tells two Blenders with one pid apart
  start     t ev pid seq ep keys args generative [agent batch i]
                                                                before each op, and before each op inside a batch;
                                                                agent is the caller's self-reported name;
                                                                generative is whether the gen_* ops could run
                                                                when the call started - the allow_generative
                                                                preference and the open file's project mode
                                                                (ops_gen.generative_allowed) - so the file shows
                                                                a session ran non-generative
  end       t ev pid seq ep ok ms [err]                         after it answered; err is cut at 160 characters
  shutdown  t ev pid reason kind                                Blender exited, the add-on was unregistered, the
                                                                server restarted, or the journal was turned off -
                                                                kind exit, unregister, restart or journal-off

WHERE, and why not UE's answer. UE writes <project>/Saved/MifBridge/journal.jsonl. Blender has no project, and an
unsaved .blend has no folder at all. So there is one file per USER, the same one for every Blender version, because a
user with three Blenders should have one place to look:
  Windows  %LOCALAPPDATA%\\MifBridge\\Blender\\journal.jsonl
  macOS    ~/Library/Logs/MifBridge/Blender/journal.jsonl
  other    $XDG_STATE_HOME/mifbridge/blender/journal.jsonl   (~/.local/state when XDG_STATE_HOME is unset)
MIF_BLENDER_JOURNAL names another file. run_blender_suites points the Blenders it starts at a scratch one, so a
certification run does not bury the user's own sessions.

SEVERAL BLENDERS WRITE ONE FILE, so every record carries its pid and a per-process seq, and the file is opened for
APPEND ONLY at the OS level. MEASURED 2026-09-21 on Windows 11, four processes appending 20,000 lines each to one
file: open(path, "a") left 35,801 of the 80,000 lines whole, lost 44,202 and tore 1,939; os.open with O_APPEND lost
47,999 and tore 1,884. The C runtime implements append as a seek to the end and then a write, and two processes seek
to the same end. A handle opened with FILE_APPEND_DATA and no FILE_WRITE_DATA lost none and tore none - Windows then
puts each write at the true end itself. POSIX O_APPEND makes the same promise by specification.

WRITTEN, NOT FSYNCED. Each record is one os.write on an unbuffered descriptor, so it is out of this process before the
op runs and survives the process dying, which is the failure this is for. An fsync per record would also survive a
power cut, at a disk flush per call; UE's FArchive::Flush pays that and this does not.

BOUNDED. When a session opens a file past 16 MB it is first renamed journal.1.jsonl, replacing the one before, so the
two together stay under 32 MB. The UE journal has no bound; one project here holds 676,934 records.

THE RING. The last 64 calls are also kept in memory, whether or not the file is being written - the add-on preference
switches the disk half only, as UE's panel ring does. A call is in the ring from the moment it starts, with ok None
until it answers, and version() moves on every start and every finish, so a UI can poll it and redraw only on change.
The rows are the UI's contract: seq, t (epoch seconds), op, ok, ms, error, summary - plus keys, args and agent (the
caller's self-reported name from the frame, or None), and batch and i for an op run inside a batch.

WHAT IS NOT WRITTEN. A call's params are summarised, never copied: every key, and each value cut short - a string past
60 characters to its head and tail, a list to its length, an object to its keys. run_python's code goes in as its size
only, and a key that reads like a secret as "<redacted>". The journal is a trail, not a transcript.

NEVER RAISES. A journal that can fail a call is worse than none, so every entry point swallows its own errors and
reports the last one as diskError in call_log.

THREADS. Records are written on the main thread - server._execute runs there in both modes - and the socket threads
never call in here. The lock still guards every read and write of the ring, so a reader on any thread gets whole rows.
"""

import atexit
import collections
import json
import math
import os
import sys
import threading
import time

import bpy

RING_CAPACITY = 64
ROTATE_BYTES = 16 * 1024 * 1024
TAIL_BYTES = 512 * 1024              # how much of the file call_log {disk: true} reads back
ERR_CHARS = 160                      # UE's cut, for UE's reason: the full text is in the response the caller got
STR_CHARS = 60
SUMMARY_CHARS = 120
MAX_KEYS = 40
MAX_ARGS = 16
SECRET_WORDS = ("token", "password", "passwd", "secret", "apikey", "api_key", "credential", "authorization")

_LOCK = threading.RLock()
_RING = collections.deque(maxlen=RING_CAPACITY)
_OPEN = {}                           # seq -> its ring row, until the call answers
_S = {"fd": None, "path": None, "seq": 0, "version": 0, "port": None, "session": None, "error": None, "total": 0,
      "exitHook": False,
      # WHICH PROCESS WROTE A SESSION, beyond its pid - Windows hands pids out again: a five-build sweep on 2026-09-21
      # wrote 595 sessions from 516 pids. Random per import, so a reader can tell "the same Blender opened the file
      # again" from "a later Blender got the same pid". A module reload draws a new one, and is preceded by a shutdown.
      "proc": os.urandom(4).hex()}


def iso(epoch):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(epoch)) + ".%03dZ" % int((epoch % 1.0) * 1000)


def journal_path():
    """The file this Blender journals to: MIF_BLENDER_JOURNAL, or the per-user default for this platform."""
    override = (os.environ.get("MIF_BLENDER_JOURNAL") or "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        return os.path.join(base, "MifBridge", "Blender", "journal.jsonl")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Logs", "MifBridge", "Blender", "journal.jsonl")
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(home, ".local", "state")
    return os.path.join(base, "mifbridge", "blender", "journal.jsonl")


def previous_path(file_path):
    root, ext = os.path.splitext(file_path)
    return root + ".1" + ext


def disk_enabled():
    """The add-on preference, read where it is used. No preferences at all - a headless Blender that imported the
    package without enabling it - means the default, which is on."""
    try:
        addon = bpy.context.preferences.addons.get(__package__)
        prefs = addon.preferences if addon else None
    except Exception:                                                      # noqa: BLE001
        return True
    return bool(getattr(prefs, "journal_to_disk", True)) if prefs is not None else True


def _open_append(file_path):
    """A descriptor that can only APPEND - see the module docstring for the measurement behind this."""
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    if sys.platform != "win32":
        return os.open(file_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    import ctypes
    import msvcrt
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)   # a private instance: prototypes set here stay here
    create = kernel32.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                       wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    file_append_data, share_all, open_always, attr_normal = 0x0004, 0x0007, 4, 0x80
    # SHARE_ALL includes FILE_SHARE_DELETE, so a rename (rotation) or a user deleting the file works while it is open.
    handle = create(file_path, file_append_data, share_all, None, open_always, attr_normal, None)
    if handle is None or handle == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_last_error(), "cannot open the call journal for appending", file_path)
    try:
        return msvcrt.open_osfhandle(handle, os.O_APPEND)
    except OSError:
        kernel32.CloseHandle(handle)
        raise


def _raw_write(record):
    fd = _S["fd"]
    if fd is None:
        return False
    try:
        data = (json.dumps(record, separators=(",", ":"), ensure_ascii=False, default=str) + "\n").encode(
            "utf-8", "replace")
        if os.write(fd, data) != len(data):
            raise OSError("short write to the call journal")
        return True
    except (OSError, TypeError, ValueError) as exc:
        _S["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
        _close()
        return False


def _close(reason=None, kind=None):
    """Close the file; with a reason, say why first. kind is the machine-readable half: exit, unregister, restart or
    journal-off, so a reader does not have to match the sentence."""
    fd = _S["fd"]
    if fd is None:
        return
    if reason:
        record = {"t": iso(time.time()), "ev": "shutdown", "pid": os.getpid(), "reason": reason}
        if kind:
            record["kind"] = kind
        _raw_write(record)
    try:
        os.close(fd)
    except OSError:
        pass
    _S["fd"] = None


def _generative():
    """Whether the gen_* ops may run now - preference and file mode (ops_gen.generative_allowed) - or None when it
    cannot be read."""
    try:
        from .ops_gen import generative_allowed
        return generative_allowed()
    except Exception:                                                      # noqa: BLE001 - never fail a call
        return None


def _session_record():
    record = {"t": iso(time.time()), "ev": "session", "pid": os.getpid(), "proc": _S["proc"], "port": _S["port"],
              "blender": bpy.app.version_string, "background": bool(bpy.app.background),
              "generative": _generative()}
    try:
        from . import SOURCE_HASH_AT_IMPORT, bl_info
        record["addon"] = ".".join(str(n) for n in bl_info["version"])
        record["sourceHash"] = SOURCE_HASH_AT_IMPORT
    except Exception:                                                      # noqa: BLE001
        pass
    return record


def _ensure_open():
    """The descriptor, opening it - and writing a session header - if it is not open. None when it cannot be opened;
    the reason is kept for call_log and the next call tries again."""
    if _S["fd"] is not None:
        return _S["fd"]
    file_path = journal_path()
    try:
        try:
            if os.path.getsize(file_path) > ROTATE_BYTES:
                os.replace(file_path, previous_path(file_path))
        except OSError:
            pass                  # no file yet, or another program holds it without sharing a rename: keep appending
        fd = _open_append(file_path)
    except OSError as exc:
        _S["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
        return None
    _S["fd"], _S["path"], _S["error"] = fd, file_path, None
    if not _S["exitHook"]:
        atexit.register(_on_exit)
        _S["exitHook"] = True
    _S["session"] = _S["session"] or iso(time.time())
    _raw_write(_session_record())
    return _S["fd"]


def _name(ep):
    return str(ep)[:80] if ep else "(no endpoint)"


def _clip(text, cap):
    text = " ".join(str(text).split())
    return text if len(text) <= cap else text[:cap - 3] + "..."


def _short(value):
    """One parameter value, cut down to what a trail needs."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if abs(value) < 10 ** 15 else "<%d digits>" % len(str(abs(value)))
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else str(value)
    if isinstance(value, str):
        if len(value) <= STR_CHARS:
            return value
        return "%s...%s (%d chars)" % (value[:32], value[-20:], len(value))
    if isinstance(value, (list, tuple)):
        if len(value) <= 4 and all(v is None or isinstance(v, (bool, int, float)) for v in value):
            return [_short(v) for v in value]
        return "[%d items]" % len(value)
    if isinstance(value, dict):
        names = [str(k)[:24] for k in list(value)[:8]]
        more = len(value) - len(names)
        return "{%s%s}" % (", ".join(names), (", +%d more" % more) if more > 0 else "")
    return "<%s>" % type(value).__name__


def summarize(ep, params):
    """(keys, args) for a start record: every key, and a short form of the first MAX_ARGS values."""
    if not isinstance(params, dict):
        return [], ({"params": _short(params)} if params is not None else {})
    keys = sorted(str(k) for k in params)
    args = {}
    for key in keys[:MAX_ARGS]:
        value = params.get(key)
        lowered = key.lower()
        if any(word in lowered for word in SECRET_WORDS):
            args[key] = "<redacted>"
        elif ep == "run_python" and key in ("code", "script") and isinstance(value, str):
            args[key] = "<%d chars, %d lines - code is never journaled>" % (len(value), len(value.splitlines()))
        elif ep == "batch" and key == "ops" and isinstance(value, list):
            names = [str((e.get("op") or e.get("endpoint")) if isinstance(e, dict) else "?")[:40] for e in value]
            args[key] = names[:12] + (["+%d more" % (len(names) - 12)] if len(names) > 12 else [])
        else:
            args[key] = _short(value)
    if len(keys) > MAX_KEYS:
        keys = keys[:MAX_KEYS] + ["+%d more" % (len(keys) - MAX_KEYS)]
    return keys, args


def _summary_text(args):
    parts = []
    for key, value in args.items():
        shown = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        parts.append("%s=%s" % (key, shown))
    return _clip(", ".join(parts), SUMMARY_CHARS)


def current_seq():
    return _S["seq"]


def clean_agent(value):
    """The caller's self-reported name from the request frame's `agent` - the MCP server sends MIF_AGENT or the MCP
    client's own handshake name, the tag the UE side gets as X-Mif-Agent. Printable ASCII, 32 characters; None when
    absent. It is what the caller says about itself, not an identity."""
    if not isinstance(value, str):
        return None
    text = "".join(ch for ch in value if " " <= ch <= "~").strip()[:32]
    return text or None


def call_start(ep, params, batch=None, index=None, agent=None):
    """Record a call BEFORE it runs, and return its seq. Never raises. An op inside a batch takes the batch's agent."""
    try:
        with _LOCK:
            _S["seq"] += 1
            seq = _S["seq"]
            now = time.time()
            keys, args = summarize(ep, params)
            if agent is None and batch is not None:
                agent = (_OPEN.get(batch) or {}).get("agent")
            generative = _generative()
            row = {"seq": seq, "t": now, "op": _name(ep), "ok": None, "ms": None, "error": None,
                   "summary": _summary_text(args), "keys": keys, "args": args, "agent": agent, "generative": generative}
            record = {"t": iso(now), "ev": "start", "pid": os.getpid(), "seq": seq, "ep": row["op"], "keys": keys,
                      "args": args, "generative": generative}
            if agent:
                record["agent"] = agent
            if batch is not None:
                row["batch"] = record["batch"] = batch
                row["i"] = record["i"] = index
            _RING.append(row)
            _OPEN[seq] = row
            _S["total"] += 1
            _S["version"] += 1
            if disk_enabled():
                if _ensure_open() is not None:
                    _raw_write(record)
            else:
                # Turned off in the preferences since the last call. Said in the file, so this session's missing
                # shutdown is not read as a Blender that died.
                _close("the call journal was turned off in the add-on preferences", "journal-off")
            return seq
    except Exception as exc:                                               # noqa: BLE001
        _S["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
        return None


def call_end(seq, ep, ok, err, ms, bulk=None):
    """Close a call: its ring row, and the end record when the file is open. Never raises. `bulk` is a bulk preview's or
    commit's {mode, endpoint, previewId?, outcome, changes, targets[]} (ops_preview.py), the end record's "bulk" as on Unreal."""
    try:
        with _LOCK:
            row = _OPEN.pop(seq, None) if seq is not None else None
            ok = bool(ok)
            ms = round(float(ms), 2)
            error = None if ok else (_clip(err, ERR_CHARS) if err else "failed, with no error message")
            if row is not None:
                row.update(ok=ok, ms=ms, error=error)
            _S["version"] += 1
            record = {"t": iso(time.time()), "ev": "end", "pid": os.getpid(), "seq": seq, "ep": _name(ep), "ok": ok,
                      "ms": ms}
            if error is not None:
                record["err"] = error
            if bulk:
                record["bulk"] = bulk
                if row is not None:
                    row["bulk"] = bulk
            if _S["fd"] is not None:
                _raw_write(record)
    except Exception as exc:                                               # noqa: BLE001
        _S["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])


def recent(limit=50):
    """The last calls, NEWEST FIRST, as copies - a call still running has ok None. The UI's read."""
    with _LOCK:
        rows = [dict(r) for r in reversed(_RING)]
    return rows if limit is None else rows[:max(0, int(limit))]


def version():
    """Moves on every change to the ring - a call starting and a call finishing. No I/O: poll it."""
    return _S["version"]


def path():
    """The JSONL file, or None when the disk half is switched off in the preferences."""
    return (_S["path"] or journal_path()) if disk_enabled() else None


def status():
    return {"path": _S["path"] or journal_path(), "diskEnabled": disk_enabled(), "diskOpen": _S["fd"] is not None,
            "diskError": _S["error"], "pid": os.getpid(), "proc": _S["proc"], "sessionStarted": _S["session"],
            "totalCalls": _S["total"], "ringCapacity": RING_CAPACITY}


def open_session(port):
    """The server (re)started on `port`: the next record is a fresh session header carrying it."""
    try:
        with _LOCK:
            _S["port"] = port
            # A restart closes the session it replaces in words, so that one does not read as a Blender that died. A
            # first start has no session open, and writes nothing here.
            _close("the bridge server was restarted", "restart")
            if disk_enabled():
                _ensure_open()
    except Exception as exc:                                               # noqa: BLE001
        _S["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])


def on_disable():
    """The add-on is being unregistered: say so in the file, and take the exit hook with it.

    TWO CAUSES, ONE RECORD. Measured on 5.0.1 in a GUI Blender of my own, 2026-09-21: an ordinary File > Quit
    unregisters every add-on on the way out (Blender disables them all at exit), so this - not the exit hook - writes
    the shutdown record of a normal GUI quit. The reason names both, because this cannot tell them apart.
    """
    with _LOCK:
        _close("the add-on was unregistered - Blender was closed, or MifBlender was disabled", "unregister")
        if _S["exitHook"]:
            try:
                atexit.unregister(_on_exit)
            except Exception:                                              # noqa: BLE001
                pass
            _S["exitHook"] = False


def _on_exit():
    # NO bpy HERE. This runs while Python finalises inside Blender's exit, after much of Blender is already freed.
    with _LOCK:
        _close("Blender exited", "exit")


def read_disk():
    """(calls newest first, sessions newest first, info) from the last TAIL_BYTES of the file.

    EACH RECORD BELONGS TO THE LATEST SESSION HEADER WITH ITS PID, because a Blender writes its header before its first
    call and two live processes never share a pid. Starts pair with ends inside one session, by seq - never across a
    reused pid, where seq starts again at 1.

    HOW A SESSION ENDED. Every clean end writes a shutdown record first - Blender exiting, the add-on unregistering, a
    server restart, the journal switched off - with a kind for each. So a later header with the same pid and no
    shutdown between is a Blender that stopped without one - killed or crashed - whose pid came back to a later one:
    that session reads died, with the evidence. Windows reuses pids freely (595 sessions from 516 pids in one sweep,
    2026-09-21), which is why this used to read as "restarted", wrongly. Headers carry a per-process id (proc) since
    then: the same id again with no shutdown is the same Blender opening the file again after a write error - one
    session. A shutdown of kind restart reads restarted.

    The calls this process has in flight right now - the call_log that is reading, and a batch around it - are left
    out: they have no end yet only because they are still running here.
    """
    file_path = _S["path"] or journal_path()
    info = {"fileBytes": None, "readBytes": 0, "unreadableLines": 0}
    prev = previous_path(file_path)
    if os.path.isfile(prev):
        info["previousFile"] = prev
    try:
        size = os.path.getsize(file_path)
        with open(file_path, "rb") as fh:
            begin = max(0, size - TAIL_BYTES)
            fh.seek(begin)
            data = fh.read()
    except OSError as exc:
        info["readError"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
        return [], [], info
    info["fileBytes"], info["readBytes"] = size, len(data)
    lines = data.split(b"\n")
    if begin > 0 and lines:
        lines = lines[1:]                   # the read began mid-line
    me, my_proc = os.getpid(), _S["proc"]
    with _LOCK:
        in_flight = set(_OPEN)
    starts, order, sessions, latest = {}, [], [], {}
    for raw in lines:
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw.decode("utf-8"))
        except ValueError:
            info["unreadableLines"] += 1
            continue
        if not isinstance(rec, dict):
            info["unreadableLines"] += 1
            continue
        ev, pid = rec.get("ev"), rec.get("pid")
        if ev == "session":
            before, proc = latest.get(pid), rec.get("proc")
            if before is not None and not before["closed"]:
                if proc is not None and proc == before.get("proc"):
                    before["reopened"] = before.get("reopened", 0) + 1
                    continue
                before["died"] = True
                before["diedEvidence"] = ("no shutdown record, and a later session (%s) came up with the same pid - "
                                          "this Blender was killed or crashed" % rec.get("t"))
            session = {"pid": pid, "proc": proc, "t": rec.get("t"), "blender": rec.get("blender"),
                       "port": rec.get("port"), "background": rec.get("background"),
                       "generative": rec.get("generative"), "closed": False}
            sessions.append(session)
            latest[pid] = session
        elif ev == "shutdown":
            session = latest.get(pid)
            if session is not None:
                session["closed"] = True
                session["reason"] = rec.get("reason")
                kind = rec.get("kind")
                if kind:
                    session["kind"] = kind
                if kind == "restart" or (kind is None and "restarted" in str(rec.get("reason"))):
                    session["restarted"] = True
        elif ev in ("start", "end"):
            session = latest.get(pid)
            key = ((id(session) if session is not None else ("no session header in the window", pid)), rec.get("seq"))
            if ev == "start":
                row = {"pid": pid, "seq": rec.get("seq"), "t": rec.get("t"), "op": rec.get("ep"), "ended": False,
                       "ok": None, "ms": None, "error": None, "keys": rec.get("keys"), "args": rec.get("args"),
                       "agent": rec.get("agent"), "generative": rec.get("generative"),
                       "session": session["t"] if session is not None else None}
                if "batch" in rec:
                    row["batch"], row["i"] = rec.get("batch"), rec.get("i")
                starts[key] = row
                order.append((row, session))
            else:
                row = starts.get(key)
                if row is not None and not row["ended"]:
                    row.update(ended=True, ok=rec.get("ok"), ms=rec.get("ms"), error=rec.get("err"))
    mine = latest.get(me)
    if mine is not None and mine.get("proc") not in (None, my_proc):
        mine = None                         # the last header with this pid was another process's
    calls = []
    for row, session in reversed(order):
        if row["seq"] in in_flight and row["pid"] == me and (session is mine or session is None):
            continue
        if session is not None and session.get("died"):
            row["sessionDied"] = True
        calls.append(row)
    return calls, list(reversed(sessions)), info
