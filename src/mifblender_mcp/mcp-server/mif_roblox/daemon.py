"""The daemon the Roblox Studio plugin long-polls.

A Studio plugin cannot listen on a socket; it can make HTTP requests to 127.0.0.1 once the person allows it. So this
process listens, the plugin asks it for work (`/plugin/poll`, held open up to `hold` seconds), runs it and posts the reply
(`/plugin/reply`). The MCP side queues work with `POST /call` and waits for that reply. One FIFO queue, one plugin session.

    python tools/mcp-server/mif_roblox/daemon.py [--port 8650] [--hold 20]       # alone
    Daemon(port).start()                                                          # as a thread (tools.py does this)

RULES, each held by a dev test:
- loopback only; a request with an Origin header or a non-loopback Host is refused 403, before anything else is read;
- every route but GET /status needs X-Mif-Token (MIF_ROBLOX_TOKEN, default "dev");
- a call that finds no plugin is held while one is expected (a daemon up under 120 s with no hello yet, or a plugin
  that polled within 300 s whose Studio still runs), at most 90 s and never past its timeout, and its answer says
  waitedSeconds and pluginExpected; otherwise it fails at once, noPlugin;
- a call is handed to the plugin AT MOST ONCE. Timed out while queued: withdrawn, ran:false. Timed out after it was
  handed over: outcomeUnknown:true, and a late reply is kept for GET /request (KEEP_SECONDS from its timeout);
- a call is handed over only once the poll answer carrying it was written: until then it is "handing", and a write that
  fails puts it back at the front of the queue, never marked dispatched (found 2026-11-22: a gone poll connection left
  the caller waiting its whole timeout for "it may have run");
- a blob's parts are kept BLOB_PARTS_KEEP seconds, and a part whose count differs from its blob's first is refused;
- a Content-Length that is not a number, or is negative, is answered 400 before anything is read;
- a hello with another protocol number is refused, naming both;
- a plugin running a call it was handed (its caller still waiting) is connected though it does not poll, and /status
  names it busy {op, requestId, seconds, timeout}, and a call queued behind it says queuedBehind; a hello from another Studio pid while the session's Studio is connected is
  refused session-held, so a second Studio never takes a live session (found 2026-11-22);
- a poll or reply from a session this daemon did not open answers 409 rehello;
- a call's reply may ride on the plugin's next poll (hello answers replyOnPoll), and the poll's answer acknowledges it
  (replyAck); the plugin reports its own HttpService request rate in each poll (stats), which /status shows.
- the port is bound exclusively (no SO_REUSEADDR; SO_EXCLUSIVEADDRUSE on Windows): a second daemon on a held port fails
  'port in use' naming the holder's pid and image, instead of splitting the port.

Standard library only. Never raises out of a request: every failure is a JSON body with ok:false.
"""
import argparse
import base64
import binascii
import collections
import errno
import hmac
import json
import os
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PROTOCOL = 1
BACKEND = "roblox-daemon"
DEFAULT_PORT = 8650                 # measured free 2026-11-18: unity_port_probe 8600-8690, 91 of 91 clean
HOLD = float(os.environ.get("MIF_ROBLOX_HOLD", "20"))
GRACE = 10.0                        # a plugin is connected while it polled within hold + GRACE seconds
# A call that finds no plugin is held, not failed at once, while one is expected: the daemon started
# moments ago and no plugin has found it yet (ensure_daemon starts one inside each build or test process), or its plugin
# polled within EXPECT_SEEN seconds and its Studio still runs (a Studio busy after a run of play screenshots stopped
# polling for about a minute). Held at most PLUGIN_WAIT seconds, and never past the call's own timeout.
PLUGIN_WAIT = float(os.environ.get("MIF_ROBLOX_PLUGIN_WAIT", "90"))
EXPECT_UP = 120.0
EXPECT_SEEN = 300.0
KEEP_SECONDS = 600.0
MAX_BODY = 8 << 20
MAX_TIMEOUT = 600.0
BLOBS_KEPT = 8
BLOB_PARTS_KEEP = 300.0             # a blob still arriving after this long was abandoned (its sender gone): dropped
JOURNAL_ROLL = 16 << 20
LOOPBACK_HOSTS = ("127.0.0.1", "localhost")
DAEMON_PY = os.path.abspath(__file__)
SERVER_PY = os.path.join(os.path.dirname(os.path.dirname(DAEMON_PY)), "server.py")
MODES = ("read", "scratch", "full")


def catalog_stamp():
    """The Studio build the catalog was generated from, for the plugin's panel and rbx_status; None when unreadable."""
    try:
        here = os.path.dirname(os.path.dirname(DAEMON_PY))
        if here not in sys.path:
            sys.path.insert(0, here)
        from mif_roblox import catalog as C
        return dict(C.load().studio)
    except Exception:  # noqa: BLE001 - the panel shows unknown, never a guess
        return None


def token():
    return os.environ.get("MIF_ROBLOX_TOKEN", "dev")


def peer_pid(peer_port, server_port):
    """The pid of the process on the other end of a loopback TCP connection to this daemon: the row of the TCP table
    (owner-pid, all states) whose local port is peer_port and remote port server_port. None off Windows or when not
    found. A plugin cannot read its own process id; the connection it opens names it."""
    if os.name != "nt" or not isinstance(peer_port, int):
        return None
    try:
        import ctypes
        import socket as _s
        import struct
        iphlp = ctypes.WinDLL("iphlpapi")
        size = ctypes.c_ulong(0)
        iphlp.GetExtendedTcpTable(None, ctypes.byref(size), False, 2, 5, 0)       # AF_INET, TCP_TABLE_OWNER_PID_ALL
        buf = ctypes.create_string_buffer(size.value + 8192)
        if iphlp.GetExtendedTcpTable(buf, ctypes.byref(size), False, 2, 5, 0) != 0:
            return None
        n = struct.unpack_from("<I", buf, 0)[0]
        for i in range(n):
            _st, _la, lport, _ra, rport, pid = struct.unpack_from("<6I", buf, 4 + i * 24)
            if _s.ntohs(lport & 0xFFFF) == peer_port and _s.ntohs(rport & 0xFFFF) == server_port:
                return pid
    except (OSError, AttributeError, ValueError):
        return None
    return None


def listener_pid(port):
    """The pid listening on port (IPv4, any address), from the TCP table's listener rows; None off Windows or when
    nothing listens there. A refusal to bind names it, so a split port is never silent."""
    if os.name != "nt" or not isinstance(port, int):
        return None
    try:
        import ctypes
        import socket as _s
        import struct
        iphlp = ctypes.WinDLL("iphlpapi")
        size = ctypes.c_ulong(0)
        iphlp.GetExtendedTcpTable(None, ctypes.byref(size), False, 2, 3, 0)       # AF_INET, TCP_TABLE_OWNER_PID_LISTENER
        buf = ctypes.create_string_buffer(size.value + 8192)
        if iphlp.GetExtendedTcpTable(buf, ctypes.byref(size), False, 2, 3, 0) != 0:
            return None
        n = struct.unpack_from("<I", buf, 0)[0]
        for i in range(n):
            _st, _la, lport, _ra, _rp, pid = struct.unpack_from("<6I", buf, 4 + i * 24)
            if _s.ntohs(lport & 0xFFFF) == port:
                return pid
    except (OSError, AttributeError, ValueError):
        return None
    return None


def process_image(pid):
    """The full path of pid's executable (QueryFullProcessImageNameW), or None when it cannot be read."""
    if os.name != "nt" or not pid:
        return None
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.OpenProcess.restype = ctypes.c_void_p
        h = k.OpenProcess(0x1000, False, int(pid))                     # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return None
        try:
            size = ctypes.c_ulong(1024)
            buf = ctypes.create_unicode_buffer(1024)
            if not k.QueryFullProcessImageNameW(ctypes.c_void_p(h), 0, buf, ctypes.byref(size)):
                return None
            return buf.value
        finally:
            k.CloseHandle(ctypes.c_void_p(h))
    except (OSError, AttributeError):
        return None


class PortInUse(OSError):
    """The port is held by another socket. pid and image name the holder when the TCP table can tell."""

    def __init__(self, port, pid, image, cause):
        self.port, self.pid, self.image = port, pid, image
        super().__init__(errno.EADDRINUSE, "port in use: 127.0.0.1:%d is held by pid %s (%s); nothing was bound (%s)"
                         % (port, pid if pid is not None else "unknown", image or "image unknown", cause))


class _Server(ThreadingHTTPServer):
    """No address reuse. HTTPServer sets allow_reuse_address = 1, which on Windows is SO_REUSEADDR: a second bind on a
    port a live daemon holds then SUCCEEDS and the port is split between two processes. Here the bind
    asks for SO_EXCLUSIVEADDRUSE on Windows, so a second start fails instead."""
    allow_reuse_address = False
    allow_reuse_port = False
    daemon_threads = True

    def server_bind(self):
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def bind_server(port, handler):
    """A bound _Server on 127.0.0.1:port, or PortInUse naming the holder. Other OSErrors pass through."""
    try:
        return _Server(("127.0.0.1", port), handler)
    except OSError as exc:
        if exc.errno in (errno.EADDRINUSE, errno.EACCES) or getattr(exc, "winerror", None) in (10048, 10013):
            pid = listener_pid(port)
            raise PortInUse(port, pid, process_image(pid), exc) from exc
        raise


def pid_alive(pid):
    """True while pid is a running process (GetExitCodeProcess STILL_ACTIVE). True when it cannot be told (not Windows,
    no access): then the poll timeout decides, as before."""
    if os.name != "nt" or not pid:
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.OpenProcess.restype = ctypes.c_void_p
        h = k.OpenProcess(0x1000, False, int(pid))                     # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ctypes.GetLastError() == 5                            # access denied: it exists
        try:
            code = ctypes.c_ulong(0)
            if not k.GetExitCodeProcess(ctypes.c_void_p(h), ctypes.byref(code)):
                return True
            return code.value == 259                                     # STILL_ACTIVE
        finally:
            k.CloseHandle(ctypes.c_void_p(h))
    except (OSError, AttributeError):
        return True


def policy_file():
    """The person's machine-local opt-in for the agent raising the safety mode. Never shipped: a person writes it."""
    return os.environ.get("MIF_ROBLOX_POLICY_FILE") or os.path.join(
        os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "MifBridge", "roblox_policy.json")


def read_policy():
    """{"agentMayRaiseMode": bool, "scope", "grantedBy", "date", "file"}. Off unless the file says exactly
    agentMayRaiseMode true with scope bridge-launched; read at every set_mode, so deleting the file takes effect at once."""
    path = policy_file()
    out = {"agentMayRaiseMode": False, "scope": None, "file": path, "found": False}
    try:
        with open(path, encoding="utf-8") as fh:
            p = json.load(fh)
    except (OSError, ValueError):
        return out
    out["found"] = True
    if isinstance(p, dict):
        out.update(grantedBy=p.get("grantedBy"), date=p.get("date"), scope=p.get("scope"))
        out["agentMayRaiseMode"] = p.get("agentMayRaiseMode") is True and p.get("scope") == "bridge-launched"
    return out


def home():
    return os.environ.get("MIF_ROBLOX_HOME") or os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
                                                            "MifBridge", "Roblox")


class Record:
    __slots__ = ("id", "op", "args", "state", "session", "reply", "created", "dispatched", "done", "event", "deadline",
                 "owner")

    def __init__(self, op, args, owner=None):
        self.id = uuid.uuid4().hex[:16]
        self.op, self.args = op, args
        self.owner = owner               # the caller's undo-group owner token, handed to the plugin with the call
        self.state = "queued"
        self.session = self.reply = self.dispatched = self.done = None
        self.created = time.monotonic()
        self.deadline = None             # monotonic time the caller stops waiting (set when queued)
        self.event = threading.Event()


def _coverage_now():
    """The coverage record as the panel shows it (domain_release.panel_summary), or None. Never raises."""
    try:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if here not in sys.path:
            sys.path.insert(0, here)
        from mif_roblox import domain_release
        return _coverage_shape(domain_release.panel_summary())
    except Exception:  # noqa: BLE001 - the panel is a view
        return None


def _coverage_shape(c):
    """Only the panel's coverage keys, each of its own type; anything else is None."""
    if not isinstance(c, dict) or not isinstance(c.get("domains"), list):
        return None

    def num(x):
        return x if isinstance(x, int) and not isinstance(x, bool) else None
    rows = [[str(r[0])[:60], r[1], r[2]] for r in c["domains"][:40]
            if isinstance(r, list) and len(r) == 3 and num(r[1]) is not None and num(r[2]) is not None]
    return {"studio": str(c.get("studio"))[:40], "since": str(c.get("since"))[:40], "reached": num(c.get("reached")),
            "denominator": num(c.get("denominator")), "domains": rows}


class Daemon:
    def __init__(self, port=DEFAULT_PORT, hold=None, token_value=None, journal_dir=None, plugin_wait=None,
                 expect_up=None, expect_seen=None):
        self.port = port
        self.hold = HOLD if hold is None else float(hold)
        self.plugin_wait = PLUGIN_WAIT if plugin_wait is None else float(plugin_wait)
        self.expect_up = EXPECT_UP if expect_up is None else float(expect_up)
        self.expect_seen = EXPECT_SEEN if expect_seen is None else float(expect_seen)
        self.started_mono = time.monotonic()
        self.token = token_value if token_value is not None else token()
        self.journal_dir = journal_dir or home()
        self.queue = collections.deque()
        self.records = {}
        self.cond = threading.Condition()
        self.session = None              # {"id", "protocol", "plugin", "studioVersion", "placeId", "placeName", "since"}
        self.last_seen = None
        self.server = None
        self.thread = None
        self.started = time.time()
        self._jlock = threading.Lock()
        self.verdicts = collections.deque(maxlen=256)   # the MCP side's judgement of each write, for the plugin's panel
        self.blob_parts = {}                             # blobId -> {"parts", "got": {index: bytes}} while arriving
        self.blobs = {}                                  # blobId -> bytes, oldest first, at most BLOBS_KEPT

    # ---------------------------------------------------------------- lifecycle
    def start(self):
        """Bind and serve on a thread. Raises OSError when the port is taken: the caller decides what that means."""
        self.server = bind_server(self.port, _Handler)
        self.server.mif = self
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, name="mif-roblox-daemon", daemon=True)
        self.thread.start()
        # who holds the port, so a log names the owner later (found 2026-11-23: a process took the default port during
        # another session's build and was gone before anyone looked; the journal had its pid and nothing else)
        self.journal(dict({"ev": "daemon-start", "port": self.port, "pid": os.getpid(), "protocol": PROTOCOL},
                          **owner_info()))
        return self

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            with self.cond:
                self.cond.notify_all()

    # ---------------------------------------------------------------- state
    def busy(self):
        """{op, requestId, seconds} of the call this session's plugin is running now (dispatched, not answered, its caller
        still waiting), or None. The plugin's loop is one thread: while a call runs it does not poll."""
        sid = (self.session or {}).get("id")
        now = time.monotonic()
        for r in list(self.records.values()):
            if (r.state in ("handing", "dispatched") and sid and r.session == sid and r.dispatched is not None
                    and r.deadline is not None and now < r.deadline):
                return {"op": r.op, "requestId": r.id, "seconds": round(now - r.dispatched, 1),
                        "timeout": round(r.deadline - r.created, 1)}
        return None

    def connected(self):
        # The Studio process the plugin runs in, read from its connection at hello: gone means disconnected at once,
        # not after the poll timeout (a lane relaunching right after its Studio ended found a dead one "connected").
        pid = (self.session or {}).get("studioPid")
        if pid and not pid_alive(pid):
            return False
        if self.last_seen is not None and time.monotonic() - self.last_seen <= self.hold + GRACE:
            return True
        # A plugin running a long call (a play probe runs minutes) does not poll, and is still connected: read as
        # disconnected, a second call failed "no plugin" and rbx_launch started a second Studio (found 2026-11-22).
        return self.busy() is not None

    def expected(self):
        """Why a plugin is expected to poll soon, or None: "daemonNew" (this daemon started under expect_up seconds ago
        and no plugin has said hello yet) or "pluginRecent" (its plugin polled within expect_seen seconds and that
        plugin's Studio still runs). A Studio that has exited is not waited for."""
        now = time.monotonic()
        if self.session is None:
            return "daemonNew" if now - self.started_mono < self.expect_up else None
        pid = self.session.get("studioPid")
        if pid and not pid_alive(pid):
            return None
        if self.last_seen is not None and now - self.last_seen < self.expect_seen:
            return "pluginRecent"
        return None

    def status(self):
        s = self.session or {}
        with self.cond:
            queued = sum(1 for r in self.queue if r.state == "queued")
        return {"ok": True, "backend": BACKEND, "protocol": PROTOCOL, "pid": os.getpid(), "port": self.port,
                "hold": self.hold, "upSeconds": round(time.time() - self.started, 1), "queued": queued,
                "plugin": {"connected": self.connected(), "expected": None if self.connected() else self.expected(),
                           "busy": self.busy(),
                           "lastSeenAgeMs": None if self.last_seen is None else int((time.monotonic() - self.last_seen) * 1000),
                           "sessionId": s.get("id"), "protocol": s.get("protocol"), "plugin": s.get("plugin"),
                           "studioVersion": s.get("studioVersion"), "placeId": s.get("placeId"),
                           "placeName": s.get("placeName"), "mode": s.get("mode"),
                           "pluginPort": s.get("pluginPort"), "portFrom": s.get("portFrom"),
                           "studioPid": s.get("studioPid"), "requestStats": s.get("requestStats")},
                "catalog": catalog_stamp(), "policy": {k: v for k, v in read_policy().items() if k != "file"},
                "owner": owner_info()}

    def _prune(self):
        """Records that ended more than KEEP_SECONDS ago. A late one (its caller stopped waiting) counts from then: it was
        never pruned before, since only an answered record had an end time (found 2026-11-22)."""
        now = time.monotonic()
        for rid in [k for k, r in self.records.items() if r.done and now - r.done > KEEP_SECONDS]:
            del self.records[rid]
        for bid in [k for k, e in self.blob_parts.items() if now - e.get("t", now) > BLOB_PARTS_KEEP]:
            del self.blob_parts[bid]

    def handed(self, rid, delivered):
        """The poll answer carrying call rid was written (delivered True) or its write failed. Only a written one is
        dispatched; a failed one goes back to the front of the queue while its caller still waits, else is withdrawn."""
        now = time.monotonic()
        with self.cond:
            r = self.records.get(rid)
            if r is None:
                return
            if delivered:
                if r.state == "handing":
                    r.state = "dispatched"
                return
            if r.state == "handing" and r.deadline is not None and now < r.deadline:
                r.state, r.session, r.dispatched = "queued", None, None
                self.queue.appendleft(r)
                self.cond.notify_all()
                requeued = True
            else:
                if r.state in ("handing", "late"):
                    r.state, r.done = "withdrawn", now
                    r.event.set()
                requeued = False
        self.journal({"ev": "poll-write-failed", "rid": rid, "requeued": requeued})

    def journal(self, row):
        try:
            os.makedirs(self.journal_dir, exist_ok=True)
            path = os.path.join(self.journal_dir, "journal.jsonl")
            with self._jlock:
                if os.path.isfile(path) and os.path.getsize(path) > JOURNAL_ROLL:
                    os.replace(path, os.path.join(self.journal_dir, "journal.1.jsonl"))
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(dict({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid()}, **row)) + "\n")
        except OSError:
            pass                         # a journal that cannot be written never fails a call

    # ---------------------------------------------------------------- plugin routes
    def hello(self, body):
        proto = body.get("protocol")
        if proto != PROTOCOL:
            older = "the plugin" if isinstance(proto, int) and proto < PROTOCOL else "this daemon"
            return 200, {"ok": False, "refusedRule": "protocol", "protocol": PROTOCOL, "pluginProtocol": proto,
                         "error": "protocol mismatch: the plugin speaks %r, this daemon %d; %s is older - update it"
                                  % (proto, PROTOCOL, older)}
        new_pid = peer_pid(body.get("_peerPort"), self.port)
        with self.cond:
            old = self.session
            held = self._held(new_pid)
            if held:
                return 200, held
            self.session = {"id": uuid.uuid4().hex, "protocol": proto, "plugin": body.get("plugin"),
                            "studioVersion": body.get("studioVersion"), "placeId": body.get("placeId"),
                            "placeName": body.get("placeName"),
                            "pluginPort": body.get("port") if isinstance(body.get("port"), int) else None,
                            "studioPid": new_pid,
                            "portFrom": body.get("portFrom") if body.get("portFrom") in ("place", "setting") else None,
                            "mode": body.get("mode") if body.get("mode") in MODES else None, "since": time.time()}
            self.last_seen = time.monotonic()
            self.cond.notify_all()                  # a call held for a plugin (expected) goes ahead now
            # Calls handed to the previous session may have run there. A plugin from 0.15.0 carries in its hello the
            # replies it could not deliver (the reply POST failed, then it said hello again): such a call ends with its
            # own reply. Any other ends outcomeUnknown, never re-queued.
            carried = {}
            for u in body.get("undelivered") or []:
                if isinstance(u, dict) and isinstance(u.get("id"), str) and isinstance(u.get("reply"), dict):
                    carried[u["id"]] = u["reply"]
            recovered = 0
            for r in self.records.values():
                if r.id in carried and r.state in ("handing", "dispatched", "late", "orphaned"):
                    r.reply, r.state, r.done = dict(carried[r.id], deliveredOnHello=True), "done", time.monotonic()
                    r.event.set()
                    recovered += 1
                elif old and r.state in ("handing", "dispatched") and r.session == old["id"]:
                    r.state, r.done = "orphaned", time.monotonic()
                    r.event.set()
        self.journal({"ev": "hello", "session": self.session["id"], "studioVersion": body.get("studioVersion"),
                      "plugin": body.get("plugin"), "recovered": recovered})
        # replyOnPoll: this daemon takes a call's reply inside the next poll, so a call costs the plugin one request, not
        # two (Studio's HttpService budget, RB1001). A plugin that does not see it keeps posting /plugin/reply.
        return 200, {"ok": True, "sessionId": self.session["id"], "protocol": PROTOCOL, "hold": self.hold, "replyOnPoll": True,
                     "port": self.port, "catalog": catalog_stamp(), "server": SERVER_PY, "daemonScript": DAEMON_PY,
                     "coverage": _coverage_now()}

    def _held(self, new_pid):
        """The refusal session-held when a hello comes from another Studio while this session's Studio is alive and
        connected (polling, or running a call), else None. Found 2026-11-22: a second Studio's hello replaced the session
        and orphaned the first one's running call; the first then said hello again and the two took turns, each call
        landing in whichever polled last. A Studio whose pid cannot be read on either side is let in, as before."""
        old = self.session
        old_pid = (old or {}).get("studioPid")
        if not old or not old_pid or not new_pid or old_pid == new_pid or not self.connected():
            return None
        busy = self.busy()
        self.journal({"ev": "hello-refused", "session": old.get("id"), "holder": old_pid, "from": new_pid})
        return {"ok": False, "refusedRule": "session-held", "holderPid": old_pid, "holderPlace": old.get("placeName"),
                "studioPid": new_pid, "busy": busy,
                "error": "this daemon (port %d) is held by Studio pid %s (%s), which is %s; this Studio (pid %s) was not "
                         "let in, so calls keep reaching one place. It is let in once that Studio closes or stops "
                         "polling; or give this place its own port (ServerStorage.MifBridgePort)" % (
                             self.port, old_pid, old.get("placeName"), "running %s" % busy["op"] if busy else "polling",
                             new_pid)}

    def _session_ok(self, body):
        return self.session is not None and body.get("sessionId") == self.session["id"]

    STAT_KEYS = ("requests60", "requestsTotal", "overruns", "paced", "pacedSeconds", "budget", "repliesOnPoll")

    def poll(self, body):
        # Everything that reads or writes the session runs under the lock (found 2026-11-22: the check and the stats write
        # ran outside it, so a hello in between had one session's stats written into the next). cond's lock is an RLock.
        with self.cond:
            if not self._session_ok(body):
                return 409, {"ok": False, "rehello": True, "error": "unknown session: say hello again"}
            ack = None
            if isinstance(body.get("reply"), dict):
                # the previous call's reply, riding on this poll (replyOnPoll)
                rb = body["reply"]
                _code, ack = self.reply({"sessionId": body.get("sessionId"), "id": rb.get("id"), "reply": rb.get("reply")})
                ack = {"id": rb.get("id"), "ok": bool(ack.get("ok")), "error": ack.get("error")}
            stats = body.get("stats")
            if isinstance(stats, dict):
                self.session["requestStats"] = {k: stats[k] for k in self.STAT_KEYS if isinstance(stats.get(k), (int, float))
                                                and not isinstance(stats.get(k), bool)}
            deadline = time.monotonic() + self.hold
            self.last_seen = time.monotonic()
            self.cond.notify_all()                  # a call held for a plugin (expected) goes ahead now
            if body.get("mode") in MODES:
                self.session["mode"] = body["mode"]          # the panel's choice, reported; nothing here can set it
            while True:
                while self.queue and self.queue[0].state != "queued":
                    self.queue.popleft()
                if self.queue:
                    r = self.queue.popleft()
                    # "handing" until the handler has written this answer (handed): a write that fails requeues it
                    r.state, r.session, r.dispatched = "handing", self.session["id"], time.monotonic()
                    self.last_seen = time.monotonic()
                    call = {"id": r.id, "op": r.op, "args": r.args}
                    if r.owner:
                        call["undoOwner"] = r.owner
                    return 200, {"ok": True, "call": call, "verdicts": self._take_verdicts(), "replyAck": ack}
                left = deadline - time.monotonic()
                if left <= 0 or not self._session_ok(body) or self.verdicts:
                    self.last_seen = time.monotonic()
                    return 200, {"ok": True, "call": None, "verdicts": self._take_verdicts(), "replyAck": ack}
                self.cond.wait(min(left, 1.0))

    def blob(self, body):
        """One base64 part of a binary result (a screenshot): a plugin's only way out is JSON. Kept whole once every
        part is in, the newest BLOBS_KEPT only, for the MCP side to fetch with GET /blob."""
        with self.cond:
            if not self._session_ok(body):
                return 409, {"ok": False, "rehello": True, "error": "unknown session: say hello again"}
        bid, idx, parts, data = body.get("blobId"), body.get("index"), body.get("parts"), body.get("data")
        if not (isinstance(bid, str) and bid and isinstance(idx, int) and isinstance(parts, int) and 0 <= idx < parts <= 4096
                and isinstance(data, str)):
            return 200, {"ok": False, "error": "blob needs blobId, index, parts and base64 data"}
        try:
            chunk = base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error) as exc:
            return 200, {"ok": False, "error": "part %d is not base64: %s" % (idx, exc)}
        with self.cond:
            self._prune()                           # parts of a blob abandoned BLOB_PARTS_KEEP s ago go
            entry = self.blob_parts.setdefault(bid, {"parts": parts, "got": {}, "t": time.monotonic()})
            if entry["parts"] != parts:
                return 200, {"ok": False, "error": "part %d says %d parts, and blob %s began with %d: refused, the blob "
                                                   "keeps its first count" % (idx, parts, bid, entry["parts"])}
            entry["got"][idx] = chunk
            if len(entry["got"]) == entry["parts"]:
                self.blobs[bid] = b"".join(entry["got"][i] for i in range(entry["parts"]))
                del self.blob_parts[bid]
                while len(self.blobs) > BLOBS_KEPT:
                    self.blobs.pop(next(iter(self.blobs)))
        return 200, {"ok": True, "index": idx}

    def _take_verdicts(self):
        out = list(self.verdicts)
        self.verdicts.clear()
        return out

    def verdict(self, body):
        """The MCP side's judgement of a write it read back, handed to the plugin's panel with the next poll."""
        rid, tool = body.get("requestId"), body.get("tool")
        if not (isinstance(rid, str) and rid) and not (isinstance(tool, str) and tool):
            return 200, {"ok": False, "error": "verdict needs requestId, or tool for a call refused before it was sent"}
        row = {"id": rid if isinstance(rid, str) and rid else None, "tool": tool, "ok": bool(body.get("ok")),
               "verified": body.get("verified"), "refusedRule": body.get("refusedRule"),
               "error": None if body.get("error") is None else str(body["error"])[:300]}
        # What the caller waited, end to end on the MCP side: the panel's row shows it beside the op's own time.
        ms = body.get("ms")
        if isinstance(ms, (int, float)) and not isinstance(ms, bool) and 0 <= ms < 3.6e6:
            row["ms"] = int(ms)
        cov = _coverage_shape(body.get("coverage"))
        if cov is not None:
            row["coverage"] = cov                   # the panel's CoverageRow, when this call added a member
        with self.cond:
            self.verdicts.append(row)
            self.cond.notify_all()
        self.journal(dict(row, ev="verdict"))
        return 200, {"ok": True}

    FLAG_FILE = "flagged.jsonl"
    FLAG_KEYS = ("id", "time", "op", "tool", "rule", "verdict", "ms", "error", "target")

    def flag(self, body):
        """A row the person flagged in the panel, with their note: appended to <home>/flagged.jsonl, the report file the
        Unity panel's flag writes too (a plugin cannot write a file). Only the row's own fields and a 1,000-character
        note are kept; rbx_flagged reads them back for the agent."""
        with self.cond:
            if not self._session_ok(body):
                return 409, {"ok": False, "rehello": True, "error": "unknown session: say hello again"}
            sess = dict(self.session)
        row, note = body.get("row"), body.get("note")
        if not isinstance(row, dict) or not (note is None or isinstance(note, str)):
            return 200, {"ok": False, "error": "flag needs row (an object) and an optional note (a string)"}
        kept = {k: (str(row[k])[:300] if isinstance(row[k], str) else row[k]) for k in self.FLAG_KEYS
                if k in row and isinstance(row[k], (str, int, float, bool, type(None)))}
        entry = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "session": sess["id"],
                 "placeName": sess.get("placeName"), "row": kept, "note": (note or "")[:1000]}
        path = os.path.join(self.journal_dir, self.FLAG_FILE)
        try:
            os.makedirs(self.journal_dir, exist_ok=True)
            with self._jlock:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")
        except OSError as exc:
            return 200, {"ok": False, "error": "could not write %s: %s" % (path, exc)}
        return 200, {"ok": True, "file": path}

    def flagged(self, limit=100):
        path = os.path.join(self.journal_dir, self.FLAG_FILE)
        rows = []
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
        except OSError:
            pass
        return 200, {"ok": True, "file": path, "count": len(rows), "rows": rows[-limit:]}

    def reply(self, body):
        reply = body.get("reply")
        with self.cond:
            if not self._session_ok(body):
                return 409, {"ok": False, "rehello": True, "error": "unknown session: say hello again"}
            if not isinstance(reply, dict):
                return 200, {"ok": False, "error": "reply must be an object"}
            self.last_seen = time.monotonic()
            r = self.records.get(body.get("id"))
            # "handing": the plugin ran it and answered before the handler marked the write done
            if r is None or r.session != self.session["id"] or r.state not in ("handing", "dispatched", "late"):
                return 200, {"ok": False, "error": "no call %r is waiting on this session" % (body.get("id"),)}
            r.reply, r.done = reply, time.monotonic()
            r.state = "done"
            r.event.set()
        return 200, {"ok": True}

    # ---------------------------------------------------------------- MCP routes
    def call(self, body):
        op, args = body.get("op"), body.get("args") or {}
        timeout = body.get("timeout", 60)
        if not isinstance(op, str) or not op or not isinstance(args, dict):
            return 200, {"ok": False, "ran": False, "error": "call needs op (a string) and args (an object)"}
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= MAX_TIMEOUT:
            return 200, {"ok": False, "ran": False, "error": "timeout is seconds, above 0 and at most %d" % MAX_TIMEOUT}
        t_call = time.monotonic()
        waited, why = 0.0, None
        if not self.connected():
            # Held while a plugin is expected, never past the call's own timeout: what is left of it is
            # the plugin's to take the call in.
            why = self.expected()
            bound = min(self.plugin_wait, float(timeout)) if why else 0.0
            with self.cond:
                while not self.connected() and time.monotonic() - t_call < bound:
                    self.cond.wait(min(bound - (time.monotonic() - t_call), 0.5))
            waited = round(time.monotonic() - t_call, 1)
            if not self.connected():
                held = (" It was held %.1f s for a plugin that was expected (%s) and none came." % (waited, why)
                        if why else "")
                return 200, {"ok": False, "ran": False, "retryable": True, "editorState": "noPlugin",
                             "waitedSeconds": waited, "pluginExpected": why,
                             "error": "no Studio plugin has polled this daemon (port %d) in the last %d s: Studio is "
                                      "closed, the MifBridge plugin is not installed, Connect was not pressed, or "
                                      "127.0.0.1 was not allowed.%s" % (self.port, int(self.hold + GRACE), held)}
            timeout = max(float(timeout) - waited, 1.0)
        if op == "set_mode":
            # The daemon is the gate: it hands the plugin this session's id and the person's policy, read now. The plugin
            # still checks the place's launch marker and whether the person clicked, and lowers without any of it.
            pol = read_policy()
            args = dict(args, session=(self.session or {}).get("id"),
                        policy={"agentMayRaiseMode": pol["agentMayRaiseMode"], "scope": pol.get("scope")})
            self.journal({"ev": "set_mode", "mode": args.get("mode"), "policy": pol["agentMayRaiseMode"]})
        owner = body.get("undoOwner")
        # the caller's undo-group owner token (plugin 0.17.23): the plugin lets a write join its open group only with it
        r = Record(op, args, owner if isinstance(owner, str) and 0 < len(owner) <= 80 else None)
        r.deadline = time.monotonic() + float(timeout)
        behind = self.busy()                     # queued while the plugin runs another call: said in the answer
        with self.cond:
            self._prune()
            self.records[r.id] = r
            self.queue.append(r)
            self.cond.notify_all()
        t0 = time.monotonic()
        self.journal({"ev": "start", "rid": r.id, "op": op, "keys": sorted(args)})
        r.event.wait(timeout)
        with self.cond:
            ms = int((time.monotonic() - t0) * 1000)
            if r.state == "done":
                out = dict(r.reply)
                out.setdefault("ok", False)
            elif r.state == "queued":
                r.state, r.done = "withdrawn", time.monotonic()
                out = {"ok": False, "ran": False, "retryable": True,
                       "error": "the plugin did not take this call within %ss; it was withdrawn and will not run" % timeout}
            elif r.state == "orphaned":
                out = {"ok": False, "outcomeUnknown": True, "checkWith": "rbx_get_props or rbx_find",
                       "error": "the plugin said hello again while this call was with it: it may have run. Read the "
                                "target before sending anything else"}
            else:
                # its caller has stopped waiting: from now it is kept KEEP_SECONDS for a late reply, then pruned
                r.state, r.done = "late", time.monotonic()
                out = {"ok": False, "outcomeUnknown": True, "checkWith": "rbx_status request_id=%s" % r.id,
                       "error": "the plugin took this call and did not answer within %ss: it may have run. Read the "
                                "target, or ask for the late reply, before sending anything else" % timeout}
        out.update(requestId=r.id, ms=ms)
        if behind and behind.get("requestId") != r.id:
            out["queuedBehind"] = behind
        if why:
            # held for an expected plugin before it was queued; ms is the plugin's part, waitedSeconds the hold before it
            out.update(waitedSeconds=waited, pluginExpected=why)
        self.journal({"ev": "end", "rid": r.id, "op": op, "ok": bool(out.get("ok")), "ms": ms,
                      "err": None if out.get("ok") else str(out.get("error"))[:300]})
        return 200, out

    def request(self, rid):
        r = self.records.get(rid)
        if r is None:
            return 200, {"ok": True, "state": "unknown", "requestId": rid}
        out = {"ok": True, "state": r.state, "requestId": rid, "op": r.op}
        if r.reply is not None:
            out["reply"] = r.reply
        return 200, out


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "MifBridgeRoblox"

    def log_message(self, *a):
        pass                             # the journal is the record; stderr belongs to the MCP stdio host

    def _send(self, code, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.send_header("X-Mif-Backend", BACKEND)
        self.send_header("X-Mif-Protocol", str(PROTOCOL))
        self.end_headers()
        self.wfile.write(data)

    def _guard(self, need_token=True):
        if self.headers.get("Origin") is not None:
            return 403, {"ok": False, "refusedRule": "origin", "error": "a request with an Origin header is a web page: refused"}
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        if host not in LOOPBACK_HOSTS:
            return 403, {"ok": False, "refusedRule": "host", "error": "Host %r is not loopback: refused" % host}
        if need_token:
            got = self.headers.get("X-Mif-Token") or ""
            if not hmac.compare_digest(got.encode("utf-8"), self.server.mif.token.encode("utf-8")):
                return 403, {"ok": False, "refusedRule": "token",
                             "error": "X-Mif-Token is missing or wrong: set the same token in the plugin and MIF_ROBLOX_TOKEN"}
        return None

    def _body(self):
        # found 2026-11-22: a Content-Length that is not a number raised outside the route's try (the connection dropped
        # with no answer), and a negative one read until the client closed
        cl = (self.headers.get("Content-Length") or "0").strip()
        if not cl.isdigit():
            return None, (400, {"ok": False, "error": "Content-Length %r is not a whole number of bytes" % cl[:40]})
        n = int(cl)
        if n > MAX_BODY:
            return None, (413, {"ok": False, "error": "body over %d bytes" % MAX_BODY})
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except ValueError as exc:
            return None, (400, {"ok": False, "error": "body is not JSON: %s" % exc})
        if not isinstance(body, dict):
            return None, (400, {"ok": False, "error": "body must be a JSON object"})
        return body, None

    def do_GET(self):
        url = urlparse(self.path)
        refused = self._guard(need_token=url.path != "/status")
        if refused:
            return self._send(*refused)
        d = self.server.mif
        if url.path == "/status":
            return self._send(200, d.status())
        if url.path == "/request":
            return self._send(*d.request((parse_qs(url.query).get("id") or [""])[0]))
        if url.path == "/flagged":
            lim = (parse_qs(url.query).get("limit") or ["100"])[0]
            return self._send(*d.flagged(int(lim) if lim.isdigit() and 0 < int(lim) <= 1000 else 100))
        if url.path == "/blob":
            data = d.blobs.get((parse_qs(url.query).get("id") or [""])[0])
            if data is None:
                return self._send(404, {"ok": False, "error": "no such blob (it may have been dropped: the newest %d are kept)"
                                                              % BLOBS_KEPT})
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.send_header("X-Mif-Backend", BACKEND)
            self.end_headers()
            self.wfile.write(data)
            return None
        return self._send(404, {"ok": False, "error": "no route GET %s" % url.path})

    def do_POST(self):
        url = urlparse(self.path)
        refused = self._guard()
        if refused:
            return self._send(*refused)
        body, err = self._body()
        if err:
            return self._send(*err)
        d = self.server.mif
        route = {"/plugin/hello": d.hello, "/plugin/poll": d.poll, "/plugin/reply": d.reply, "/call": d.call,
                 "/verdict": d.verdict, "/plugin/blob": d.blob, "/plugin/flag": d.flag}.get(url.path)
        if route is None:
            return self._send(404, {"ok": False, "error": "no route POST %s" % url.path})
        if url.path == "/plugin/hello" and isinstance(body, dict):
            body["_peerPort"] = self.client_address[1]   # whose connection this is: read against the TCP table
        try:
            code, out = route(body)
        except Exception as exc:  # noqa: BLE001 - one bad request never takes the daemon down
            code, out = 500, {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        # A call rides on a poll's answer: it is dispatched only once that answer was written, and put back at the front
        # of the queue when the write fails (the plugin's connection is gone, so the call never reached it).
        handing = out.get("call", {}).get("id") if url.path == "/plugin/poll" and isinstance(out.get("call"), dict) else None
        try:
            self._send(code, out)
        except Exception:  # noqa: BLE001 - a write that failed: the connection is gone, nothing to answer on
            if handing:
                d.handed(handing, False)
            return None
        if handing:
            d.handed(handing, True)
        return None


STANDALONE = False     # True in a daemon run on its own (main below): it drives nothing itself, so a tool may use it


def owner_info():
    """The process that holds this daemon: its script, working folder, whether MIF_ROBLOX_PORT was set in it, and whether
    it is a standalone daemon (main) or one inside a tool or server process, which drives its Studio itself."""
    return {"script": os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else None, "cwd": os.getcwd(),
            "envPort": os.environ.get("MIF_ROBLOX_PORT"), "standalone": STANDALONE}


def main(argv):
    global STANDALONE
    STANDALONE = True
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("MIF_ROBLOX_PORT", DEFAULT_PORT)))
    ap.add_argument("--hold", type=float, default=None)
    a = ap.parse_args(argv)
    try:
        d = Daemon(a.port, hold=a.hold).start()
    except OSError as exc:
        print("cannot listen on 127.0.0.1:%d: %s" % (a.port, exc))
        return 1
    print("MifBridge Roblox daemon: pid %d, 127.0.0.1:%d, protocol %d, hold %ss" % (os.getpid(), d.port, PROTOCOL, d.hold))
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        d.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
