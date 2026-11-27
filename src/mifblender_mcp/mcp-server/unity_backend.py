"""The Unity backend's MCP side: _unity(), the one choke point, the classifier that tells reloading from dead, and the uy_* tools.

A sibling of server.py rather than part of it: server.py loads this in main(), never at import, and holds every tool back, so an Unreal or a Blender session's
tool list and footprint do not move. Stdlib only; no import from server.py, so the suites can use it alone.

NEVER RAISE AND NEVER HANG, the same contract as _post and _blender: every failure comes back as {"ok": False, ...} naming what to do, bounded by a deadline.

A WRITE IS NEVER SILENTLY REPEATED OR LOST. Every call carries a request id, and the editor runs an id at most once and keeps the reply across a reload. So when a connection is cut, this asks the new domain what
became of the id: `unknown` means the handler never started, so the same id goes again; anything else is answered from the record.
"""
import ctypes
import json
import os
import re
import socket
import struct
import sys
import threading
import time
import uuid

PROTOCOL = 1
PACKAGE_NAME = "com.mifbridge.editor"
MINIMUM_UNITY = "2022.3"    # the package's own "unity" field; test_offline holds the two equal. A patch below unityRelease is not refused here: nothing older has been run, and nothing older is known to fail
# THE VERSIONS THE SUITES RUN: a dev tool VERSIONS is the record, and test_offline holds this
# equal to it. A copy here because the shipped server is this folder alone, with no tools/ beside it. A project on any other version above
# the floor keeps working and is told so (untested_unity).
TESTED_UNITY = ("2022.3.62f3", "6000.0.23f1", "6000.0.31f1", "6000.3.6f1", "6000.5.0f1")
# THE TOKEN (FOUND 2026-11-22: it was "dev" whenever MIF_BRIDGE_TOKEN was unset): the environment's when set, else the token file the editor writes, read on each call (_raw). mif_credentials sits beside this file and is
# standard library too; a copy of this module without it falls back to the environment and "dev".
try:
    import mif_credentials as _CRED
except ImportError:                 # pragma: no cover - only a stripped copy of this file
    _CRED = None
TOKEN = os.environ.get("MIF_BRIDGE_TOKEN")     # None when unset; a set value, even "", is sent as it is
RELOAD_WAIT = float(os.environ.get("MIF_UNITY_RELOAD_WAIT", "60"))
CALL_TIMEOUT = float(os.environ.get("MIF_UNITY_TIMEOUT", "120"))
CONNECT_TIMEOUT = 5.0       # a closed port takes about 2.05 s to refuse on Windows; under 2 s a dead port reads as a timeout
STALLED_MS = 2000
KEEP_SECONDS = 600.0

_PIN = {"project": os.environ.get("MIF_UNITY_PROJECT") or None, "log": None}
_TIMED_OUT = {}             # (project, endpoint, body) -> (request id, when): a repeat of a timed-out call replays instead of running twice
_LOCK = threading.Lock()
NULL = object()             # a null that is meant: _unity drops None as "not given", and clearing an object reference is a real value


# -------------------------------------------------------------------------------------------------------------
# Processes. Ownership is a measurement: a pid in a file is a claim until this process table agrees.
# -------------------------------------------------------------------------------------------------------------
_IS_WIN = sys.platform == "win32"


def _pid_int(pid):
    """A pid read from a file is a claim (found 2026-11-22): a registry row with "pid": "abc" raised out of int() and out of every unpinned
    call. A pid that is not a positive whole number is no process: None."""
    if isinstance(pid, bool) or not (isinstance(pid, int) or (isinstance(pid, str) and pid.isdigit())):
        return None
    return int(pid) if int(pid) > 0 else None


def _win_handle(pid):
    pid = _pid_int(pid)
    if pid is None:
        return None
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    k = ctypes.windll.kernel32
    k.OpenProcess.restype = ctypes.c_void_p
    return k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)


def pid_alive(pid):
    if not pid:
        return False
    if _IS_WIN:
        k = ctypes.windll.kernel32
        h = _win_handle(pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        k.CloseHandle(h)
        return bool(ok) and code.value == 259     # STILL_ACTIVE
    if not _pid_int(pid):
        return False
    try:
        os.kill(_pid_int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def pid_image(pid):
    """The process's executable path, or None. On Windows this needs no rights over the process beyond limited query."""
    if _IS_WIN:
        k = ctypes.windll.kernel32
        h = _win_handle(pid)
        if not h:
            return None
        try:
            buf = ctypes.create_unicode_buffer(32768)
            size = ctypes.c_ulong(32768)
            k.QueryFullProcessImageNameW.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
            return buf.value if k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)) else None
        finally:
            k.CloseHandle.argtypes = [ctypes.c_void_p]
            k.CloseHandle(h)
    try:
        return os.readlink("/proc/%d/exe" % _pid_int(pid)) if _pid_int(pid) else None
    except (OSError, TypeError, ValueError):
        return None


def pid_cmdline(pid):
    """The command line, or None. Windows: NtQueryInformationProcess class 60, which needs no child process and so opens no window."""
    if _IS_WIN:
        h = _win_handle(pid)
        if not h:
            return None
        try:
            nt = ctypes.WinDLL("ntdll")
            size = ctypes.c_ulong(0)
            nt.NtQueryInformationProcess.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong)]
            nt.NtQueryInformationProcess(h, 60, None, 0, ctypes.byref(size))
            if not size.value:
                return None
            buf = ctypes.create_string_buffer(size.value)
            if nt.NtQueryInformationProcess(h, 60, buf, size, ctypes.byref(size)) != 0:
                return None

            class UNICODE_STRING(ctypes.Structure):
                _fields_ = [("Length", ctypes.c_ushort), ("MaximumLength", ctypes.c_ushort), ("Buffer", ctypes.c_void_p)]
            us = ctypes.cast(buf, ctypes.POINTER(UNICODE_STRING)).contents
            return ctypes.wstring_at(us.Buffer, us.Length // 2) if us.Buffer else None
        except OSError:
            return None
        finally:
            ctypes.windll.kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            ctypes.windll.kernel32.CloseHandle(h)
    try:
        if not _pid_int(pid):
            return None
        with open("/proc/%d/cmdline" % _pid_int(pid), "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except (OSError, TypeError, ValueError):
        return None


def _all_pids():
    if _IS_WIN:
        arr = (ctypes.c_ulong * 65536)()
        got = ctypes.c_ulong()
        if not ctypes.WinDLL("psapi").EnumProcesses(arr, ctypes.sizeof(arr), ctypes.byref(got)):
            return []
        return [arr[i] for i in range(got.value // ctypes.sizeof(ctypes.c_ulong))]
    try:
        return [int(n) for n in os.listdir("/proc") if n.isdigit()]
    except OSError:
        return []


def is_unity_pid(pid):
    image = pid_image(pid)
    return bool(image) and os.path.basename(image).lower() in ("unity.exe", "unity")


def _arg(cmdline, name):
    m = re.search(r'(?i)(?:^|\s)-%s\s+(?:"([^"]+)"|(\S+))' % re.escape(name), cmdline or "")
    return (m.group(1) or m.group(2)) if m else None


def _norm(path):
    return os.path.normcase(os.path.abspath(path)).replace("\\", "/") if path else path


def unity_processes():
    """Every main Unity editor process: [{pid, project, logFile, cmdline}]. Import workers carry -projectPath too, and -name or a worker flag tells them apart."""
    out = []
    for pid in _all_pids():
        if not is_unity_pid(pid):
            continue
        cmd = pid_cmdline(pid)
        if cmd is None or re.search(r"(?i)AssetImportWorker|-workerMode|-parentPid|-srvPort", cmd):
            continue
        if re.search(r"(?i)[\\/]Unity Hub[\\/]", cmd):
            continue    # the Hub's own unity.exe, always running beside a person's editors. It is no editor, and it used up an editor slot and made M100 uncheckable
        project = _arg(cmd, "projectPath") or _arg(cmd, "createProject")
        out.append({"pid": pid, "project": _norm(project) if project else None, "logFile": _arg(cmd, "logFile"), "cmdline": cmd})
    return out


def _port_holder(port):
    """Who is listening on a port, by measurement: {pid, image}, or None."""
    if not _IS_WIN:
        return None
    import subprocess
    try:
        text = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in text.splitlines():
        m = re.search(r":%d\s+\S+\s+LISTENING\s+(\d+)" % int(port), line)
        if m:
            pid = int(m.group(1))
            return {"pid": pid, "image": os.path.basename(pid_image(pid) or "?")}
    return None


# -------------------------------------------------------------------------------------------------------------
# What the outside can read: the instance registry and the files in the project's Library
# -------------------------------------------------------------------------------------------------------------
def registry_dir():
    over = os.environ.get("MIF_UNITY_REGISTRY")
    if over:
        return over
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "MifBridge", "Unity", "instances")


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def list_instances():
    """Live editors that have our server: one row per registry file whose pid is really a running Unity. A stale file is skipped: it proves nothing."""
    rows = []
    d = registry_dir()
    try:
        names = os.listdir(d)
    except OSError:
        return rows
    for name in sorted(names):
        if not name.endswith(".json"):
            continue
        row = _read_json(os.path.join(d, name))
        if not isinstance(row, dict) or not is_unity_pid(row.get("pid")):
            # A killed editor never removes its row, and these add up: 6 in one afternoon on one project. Only a row whose own pid is gone, and never one that could not be read this moment: rows are written atomically but not read so.
            if isinstance(row, dict) and row.get("pid") and not pid_alive(row["pid"]):
                try:
                    os.remove(os.path.join(d, name))
                except OSError:
                    pass
            continue
        cmd = pid_cmdline(row["pid"])
        project = _arg(cmd, "projectPath") if cmd else None
        if project and _norm(project) != _norm(row.get("project")):
            continue    # a reused pid: a Unity, but not this one
        rows.append({k: row.get(k) for k in ("pid", "port", "project", "unity", "bridge", "protocol", "state", "bindError")})
    return rows


def _project_files(project):
    state = os.path.join(project, "Library", "MifBridge")
    return _read_json(os.path.join(state, "status.json")), _read_json(os.path.join(state, "reloading"))


def _token_refused(kind, data, headers):
    """This bridge's own refusal of the token (its token gate runs before anything else, so nothing ran)."""
    return kind == "reply" and isinstance(data, dict) and "X-Mif-Token" in str(data.get("error")) and (headers or {}).get("X-Mif-Backend") == "unity"


_TOKEN_OK = {}      # port -> the token that bridge took last


def _token_source():
    """Where this client's token came from, in words, never the token: what a refusal names so a person can see which side to change."""
    if TOKEN is not None:
        return "MIF_BRIDGE_TOKEN in this client's environment"
    if _CRED is None:
        return "the old default 'dev'"
    _toks, source = _CRED.bridge_tokens()
    if source == "file":
        return "the token file %s" % _CRED.bridge_token_file()
    return "the old default 'dev' (no token file at %s)" % _CRED.bridge_token_file()


def current_token(port=None):
    """The token a call to `port` sends first: the one that bridge took last, else MIF_BRIDGE_TOKEN, else the token file's, else "dev"."""
    if port in _TOKEN_OK:
        return _TOKEN_OK[port]
    if TOKEN is not None:
        return TOKEN
    return _CRED.bridge_tokens()[0][0] if _CRED is not None else "dev"


def _raw(port, endpoint, body=None, rid=None, read_timeout=5.0, connect_timeout=CONNECT_TIMEOUT, expect_pid=None):
    """One HTTP exchange, sent only to a listener that is identified first. Returns (kind, data, headers): reply | empty | cut | refused | connect_timeout | read_timeout | bad_body | wrong_listener.

    IDENTIFY, THEN SEND (FOUND 2026-11-22: the token went to the port a status file named, and the pid was checked only on the reply). The process listening on the port must be `expect_pid` when the caller knows
    which editor it means, and must run as this user in every case (mif_credentials.listener_check); otherwise nothing is sent and the kind is wrong_listener, with the holder's pid.

    THE TOKEN: MIF_BRIDGE_TOKEN when set; else the token file the editor writes (mif_credentials), then "dev" for a package built before it. A refusal of the token runs nothing, so the next one is sent at once."""
    if _CRED is not None:
        holder = _CRED.listening_pid(int(port)) if expect_pid else None
        if expect_pid and holder and str(holder) != str(expect_pid):
            return "wrong_listener", {"pid": holder, "expected": expect_pid}, {}
        who = _CRED.listener_check(int(port))
        if not who.get("ok"):
            return "wrong_listener", {"pid": who.get("pid"), "reason": who.get("reason")}, {}
    tokens = [TOKEN] if TOKEN is not None else (_CRED.bridge_tokens()[0] if _CRED is not None else ["dev"])
    last = _TOKEN_OK.get(port)
    if last in tokens:
        tokens = [last] + [t for t in tokens if t != last]
    for i, tok in enumerate(tokens):
        out = _raw_once(port, endpoint, body, rid, read_timeout, connect_timeout, tok)
        if not _token_refused(*out):
            if out[0] == "reply":
                _TOKEN_OK[port] = tok
            return out
        if i + 1 == len(tokens):
            return out
    return out


def _raw_once(port, endpoint, body, rid, read_timeout, connect_timeout, token):
    """One HTTP exchange with one token. Returns (kind, data, headers): reply | empty | cut | refused | connect_timeout | read_timeout | bad_body.

    A SOCKET, NOT requests. One connection per call leaves a TIME_WAIT per call on whoever closes first, and at hundreds a second that runs out of ephemeral ports. Measured: 3,000 calls left 3,000, and a server
    that waits for the client's FIN and resets did not change it. SO_LINGER {1, 0} on close, once the reply is read in full by Content-Length, leaves none (the Blender client's fix). This also gives the classes above
    exactly, with no library's reading of its own exceptions."""
    payload = json.dumps(body or {}).encode("utf-8")
    lines = ["POST /%s HTTP/1.1" % endpoint, "Host: 127.0.0.1:%d" % int(port), "X-Mif-Token: %s" % token, "Content-Type: application/json", "Connection: close", "Content-Length: %d" % len(payload)]
    if rid:
        lines.append("X-Mif-Request-Id: %s" % rid)
    agent = os.environ.get("MIF_AGENT_TAG")
    if agent:
        lines.append("X-Mif-Agent: %s" % agent)
    request = ("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + payload
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(connect_timeout)
        try:
            sock.connect(("127.0.0.1", int(port)))
        except socket.timeout:
            return "connect_timeout", None, {}
        except OSError:
            return "refused", None, {}
        sock.settimeout(read_timeout)
        try:
            sock.sendall(request)
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    return ("cut" if buf else "empty"), {"status": None} if not buf else None, {}
                buf += chunk
            head, _, rest = buf.partition(b"\r\n\r\n")
            head_lines = head.decode("latin-1").split("\r\n")
            status = int(head_lines[0].split(" ")[1])
            headers = {}
            for ln in head_lines[1:]:
                k, _, v = ln.partition(":")
                headers[k.strip().title()] = v.strip()
            length = int(headers.get("Content-Length", -1))
            while length < 0 or len(rest) < length:
                chunk = sock.recv(65536)
                if not chunk:
                    if length < 0:
                        break
                    return "cut", None, {}
                rest += chunk
        except socket.timeout:
            return "read_timeout", None, {}
        except OSError:
            return "cut", None, {}
        except (ValueError, IndexError):
            return "bad_body", {"body": buf[:400].decode("latin-1")}, {}
        if not rest.strip():
            return "empty", {"status": status}, headers     # never from this server; from anything else it means the same
        try:
            return "reply", json.loads(rest.decode("utf-8")), headers
        except ValueError:
            return "bad_body", {"status": status, "body": rest[:400].decode("utf-8", "replace")}, headers
    finally:
        if sock is not None:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("hh" if _IS_WIN else "ii", 1, 0))
            except OSError:
                pass
            sock.close()


def _raw_watched(port, endpoint, body, rid, timeout, domain, expect_pid=None):
    """_raw, while watching for a domain that changed under the call. A socket left open by a dying domain would otherwise hold a caller to its full read timeout, with nobody left to answer. If the domain moved and
    the reply has not come, this is a cut connection: the record of the id decides what happened, and if the reply does get here later it is the same reply the record holds."""
    box = {}

    def run():
        box["r"] = _raw(port, endpoint, body, rid, read_timeout=timeout, expect_pid=expect_pid)
    t = threading.Thread(target=run, daemon=True, name="mif-unity-call")
    t.start()
    t.join(1.0)
    while t.is_alive():
        kind, s, _h = _raw(port, "status", read_timeout=2.0, connect_timeout=2.0)
        if kind == "reply" and isinstance(s, dict) and domain and s.get("domain") not in (None, domain) and s.get("state") != "reloading":
            t.join(0.3)
            if t.is_alive():
                return "cut", None, {}
        t.join(0.5)
    return box.get("r") or ("cut", None, {})


# Our own files are the UPM package (Packages/com.mifbridge.editor) or the Fab import.
ASSETS_FOLDER = "MifBridge"
_OWN_ERROR = re.compile(r"(?m)^(.*(?:" + re.escape(PACKAGE_NAME) + r"|Assets[\\/]" + ASSETS_FOLDER + r")[\\/][^(\r\n]*)\((\d+),(\d+)\):\s*error\s+(CS\d+):\s*(.*)$")


_ANY_ERROR = re.compile(r"(?m)^(\S[^(\r\n]*)\((\d+),(\d+)\):\s*error\s+(CS\d+):\s*(.*)$")


def own_assembly_errors(log_path):
    """Compile errors in our own package named by an editor log: [{file, line, column, code, message}]. Empty if none, or no log to read."""
    return _log_errors(log_path, _OWN_ERROR)


def project_errors(log_path):
    """Every compile error an editor log names, in any assembly, in the same shape."""
    return _log_errors(log_path, _ANY_ERROR)


def _installed_here(project):
    if os.path.isdir(os.path.join(project, "Packages", PACKAGE_NAME)) or os.path.isfile(os.path.join(project, "Assets", ASSETS_FOLDER, "Editor", "Core", "MifBridge.Editor.asmdef")):
        return True
    try:
        with open(os.path.join(project, "Packages", "manifest.json"), encoding="utf-8") as f:
            return PACKAGE_NAME in f.read()
    except OSError:
        return False


def _log_errors(log_path, pattern):
    if not log_path:
        return []
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4 * 1024 * 1024))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    seen, out = set(), []
    for m in pattern.finditer(text):
        key = m.groups()
        if key in seen:
            continue
        seen.add(key)
        out.append({"file": m.group(1).strip(), "line": int(m.group(2)), "column": int(m.group(3)), "code": m.group(4), "message": m.group(5).strip()})
    return out


def _default_log():
    base = os.environ.get("LOCALAPPDATA")
    return os.path.join(base, "Unity", "Editor", "Editor.log") if base else None


def project_unity(project):
    """The Unity version a project names in ProjectSettings/ProjectVersion.txt, or None. A claim until an editor answers, and the only thing to go on when none can."""
    try:
        with open(os.path.join(project, "ProjectSettings", "ProjectVersion.txt"), encoding="utf-8", errors="replace") as f:
            m = re.search(r"^m_EditorVersion:\s*(\S+)", f.read(), re.M)
    except OSError:
        return None
    return m.group(1) if m else None


def _older_than_supported(version):
    m = re.match(r"(\d+)\.(\d+)", version or "")
    if not m:
        return False
    return (int(m.group(1)), int(m.group(2))) < tuple(int(x) for x in MINIMUM_UNITY.split("."))


def untested_unity(version):
    """The notice for a Unity version above the floor that no suite has run, or None. It refuses nothing:
    the bridge keeps working, and the caller is told which side of the tested list it is on. None for a version below the floor too, which
    _refuse_old_unity answers."""
    if not version or version in TESTED_UNITY or _older_than_supported(version):
        return None
    return {"unity": version, "tested": list(TESTED_UNITY),
            "note": "Unity %s is not one of the versions MifBridge's suites run. The bridge keeps working on it; what differs from the tested versions has not been measured." % version}


def _refuse_old_unity(out, project):
    """A project on a Unity older than the package supports has no bridge to wait for, so a client is told that at once, and not "starting" until its wait runs out. The editor state stays what was measured."""
    version = project_unity(project)
    if _older_than_supported(version):
        out.update(refusedRule="unity-version", projectUnity=version, minimumUnity=MINIMUM_UNITY, retryable=False,
                   error="%s is on Unity %s, and the bridge needs %s or newer. Nothing can answer on this project until it is upgraded." % (project, version, MINIMUM_UNITY))
        out.pop("note", None)
    return out


def state_of(project, log_file=None):
    """Reloading, or dead? The editor-state decision table, from outside the editor. `editorState` is one of ready, compiling, importing, errors, playing, paused, changingPlay (the editor's own, when it
    answers), stalled, reloading, starting, stopped, serverNotLoaded, portBusy, wrongInstance, down."""
    project_norm = _norm(project)
    out = {"project": project}
    status_file, marker = _project_files(project)
    file_pid = (status_file or {}).get("pid")
    file_alive = bool(file_pid) and is_unity_pid(file_pid)
    port = (status_file or {}).get("port")
    if file_alive and (status_file or {}).get("state") == "stopped":
        out.update(pid=file_pid, port=port, editorState="stopped",
                   error="the bridge in this editor (pid %s) was stopped from its MifBridge panel, by a person or by its Start automatically setting. It stays stopped until someone presses Start there." % file_pid)
        return out
    if file_alive:
        out.update(pid=file_pid, port=port)
        kind, data, _h = _raw(port, "status", read_timeout=3.0, expect_pid=file_pid)
        # An editor that could not bind its explicit port says so in its own status file: the listener there is the program holding it,
        # so this is portBusy with the holder measured, not a wrong instance (FOUND 2026-11-26, test_multi M140). Nothing is sent either way.
        if kind == "wrong_listener" and (status_file or {}).get("bindError"):
            out.update(editorState="portBusy", holder=_port_holder(port), error=status_file["bindError"], retryable=False,
                       answeredBy={"pid": (data or {}).get("pid")})
            return out
        if kind == "wrong_listener":
            out.update(editorState="wrongInstance", refusedRule="listener-identity", holder=_port_holder(port), retryable=False,
                       error="port %s is held by pid %s, not this project's editor (pid %s), so nothing was sent to it, the token included. %s"
                             % (port, (data or {}).get("pid"), file_pid, (data or {}).get("reason") or ""))
            return out
        # The bridge's own refusal of this client's token. It is this editor answering, by its headers, and not somebody else on the port: a token saved in the panel is the usual way here.
        if kind == "reply" and isinstance(data, dict) and "X-Mif-Token" in str(data.get("error")) and (_h or {}).get("X-Mif-Backend") == "unity" and str((_h or {}).get("X-Mif-Instance")) == str(file_pid):
            out.update(editorState="wrongInstance", refusedRule="token", retryable=False,
                       tokenFrom=_token_source(),
                       error="the bridge on port %s refused this client's token, which came from %s. Set MIF_BRIDGE_TOKEN to the one saved in this project's panel, or clear it there; an editor "
                             "started with MIF_BRIDGE_TOKEN_FILE needs the client to name the same file (Set up writes it into the entry)." % (port, _token_source()))
            return out
        if kind == "reply" and isinstance(data, dict) and data.get("backend") == "unity":
            if (status_file or {}).get("bindError") and data.get("pid") != file_pid:
                out.update(editorState="portBusy", holder=_port_holder(port), error=status_file["bindError"], answeredBy={"pid": data.get("pid"), "project": data.get("project")})
                return out
            if data.get("pid") != file_pid or _norm(data.get("project")) != project_norm:
                out.update(editorState="wrongInstance", answeredBy={"pid": data.get("pid"), "project": data.get("project")},
                           error="port %s answers for another editor (pid %s, %s). Not driving it." % (port, data.get("pid"), data.get("project")))
                return out
            out.update(status=data, editorState="stalled" if (data.get("lastTickAgeMs") or 0) > STALLED_MS else data.get("state"), answered=True)
            return out
        if kind == "reply" or kind == "bad_body":
            holder = _port_holder(port)
            out.update(editorState="wrongInstance", holder=holder, error="port %s is answering, but not as a Unity bridge for this project (%s)." % (port, holder))
            return out
        if isinstance(marker, dict) and marker.get("pid") == file_pid:
            out.update(editorState="reloading", marker=marker, retryable=True)
            return out
        if (status_file or {}).get("bindError"):
            out.update(editorState="portBusy", holder=_port_holder(port), error=status_file["bindError"])
            return out
    procs = [p for p in unity_processes() if p["project"] == project_norm]
    if not procs:
        out.update(editorState="down", error="no Unity editor is running on %s." % project)
        if status_file or marker:
            out["staleFiles"] = True
            out["note"] = "status or marker files are there, left by a process that is gone. They prove nothing."
        # Plain batch exits 1 on a compile error before any of our code runs, so the log is all there is to say why. Only a log this client was told of: Editor.log may be another project's.
        errors = project_errors(log_file) if _installed_here(project) else []
        if errors:
            out.update(compileErrors=errors[:20], error="no Unity editor is running on %s. The last run's log names compile errors, which is why batch mode exits: %s:%d." % (project, errors[0]["file"], errors[0]["line"]))
        return _refuse_old_unity(out, project)
    pid = procs[0]["pid"]
    log = log_file or procs[0].get("logFile") or _default_log()
    out.update(pid=pid, logFile=log)
    # A log from before this process proves nothing: the last session's errors may be fixed, and this editor has not got as far as overwriting the file.
    errors = own_assembly_errors(log) if _log_is_this_runs(log, pid) else []
    if errors:
        out.update(editorState="serverNotLoaded", compileErrors=errors,
                   error="Unity (pid %d) is running on this project, but the bridge's own assembly did not compile, so nothing can answer. Fix %s:%d, or update the package." % (pid, errors[0]["file"], errors[0]["line"]))
        return out
    # Reload is all or nothing: while the project does not compile, an installed package stays unloaded in a live editor, and more waiting does nothing.
    errors = project_errors(log) if _log_is_this_runs(log, pid) and _installed_here(project) else []
    if errors:
        out.update(editorState="starting", blockedBy="project-compile-errors", compileErrors=errors[:20], retryable=False, sinceStartSeconds=_process_age(pid),
                   error="Unity (pid %d) has %s installed and cannot load it: the project does not compile. Fix %s:%d, and the bridge comes up by itself." % (pid, PACKAGE_NAME, errors[0]["file"], errors[0]["line"]))
        return out
    out.update(editorState="starting", retryable=True, sinceStartSeconds=_process_age(pid),
               note="the editor is up and the bridge has not answered yet: it is starting, or %s is not installed in this project." % PACKAGE_NAME)
    return _refuse_old_unity(out, project)


def _log_is_this_runs(log, pid):
    age = _process_age(pid)
    try:
        return age is None or os.path.getmtime(log) >= time.time() - age - 1.0
    except (OSError, TypeError):
        return False


def _process_age(pid):
    if not _IS_WIN:
        return None
    k = ctypes.windll.kernel32
    h = _win_handle(pid)
    if not h:
        return None
    try:
        times = [ctypes.c_ulonglong() for _ in range(4)]
        k.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_ulonglong)] * 4
        if not k.GetProcessTimes(h, *[ctypes.byref(t) for t in times]):
            return None
        now = (time.time() + 11644473600) * 1e7
        return round((now - times[0].value) / 1e7, 1)
    finally:
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        k.CloseHandle(h)


def _resolve(project=None):
    """Which editor this call means. Returns {project} or a typed failure: the same refusal whether no editor or several."""
    project = project or _PIN["project"]
    if project:
        return {"project": project}
    live = list_instances()
    if len(live) == 1:
        return {"project": live[0]["project"]}
    if len(live) > 1:
        return {"ok": False, "editorState": "selectInstance", "ran": False, "instances": live,
                "error": "%d Unity editors are running. Name one with uy_use(project), or start the server with MIF_UNITY_PROJECT." % len(live)}
    procs = [p for p in unity_processes() if p["project"]]
    if len(procs) == 1:
        return {"project": procs[0]["project"]}
    if len(procs) > 1:
        return {"ok": False, "editorState": "selectInstance", "ran": False, "instances": [{"pid": p["pid"], "project": p["project"]} for p in procs],
                "error": "%d Unity editors are running and none has the bridge answering. Name one with uy_use(project)." % len(procs)}
    return {"ok": False, "editorState": "noInstance", "ran": False,
            "error": "no Unity editor with the bridge is running. Open the project in Unity with %s installed, or set MIF_UNITY_PROJECT." % PACKAGE_NAME}


def _await_answer(project, deadline):
    """Wait until the project's editor answers, or it is clear it will not. Returns the last state_of."""
    while True:
        st = state_of(project, _PIN.get("log"))
        if st.get("answered"):
            return st
        if st.get("editorState") in ("down", "stopped", "serverNotLoaded", "portBusy", "wrongInstance") or st.get("refusedRule") or st.get("blockedBy"):
            return st
        if time.time() >= deadline:
            return st
        time.sleep(0.1)


def _failed_state(st, rid, endpoint, may_have_run):
    out = {"ok": False, "editorState": st.get("editorState"), "endpoint": endpoint, "requestId": rid,
           "error": st.get("error") or "the editor did not answer: %s." % st.get("editorState"), "retryable": st.get("editorState") in ("reloading", "starting") and not st.get("refusedRule") and not st.get("blockedBy")}
    for key in ("pid", "port", "compileErrors", "holder", "logFile", "note", "sinceStartSeconds", "answeredBy", "staleFiles", "refusedRule", "projectUnity", "minimumUnity", "blockedBy"):
        if key in st:
            out[key] = st[key]
    if may_have_run:
        out["outcomeUnknown"] = True
        out["retryable"] = False
        out["checkWith"] = "uy_request_status"
        out["error"] += " The call may have run. Ask uy_request_status with this requestId once the editor answers, or read the target."
    else:
        out["ran"] = False
    return out


def _unity(endpoint, /, _timeout=None, _project=None, **payload):
    """Every Unity call. Positional-only endpoint, like _post's and _blender's. None values are dropped, so the editor sees only what the caller gave."""
    body = {k: (None if v is NULL else v) for k, v in payload.items() if v is not None}
    where = _resolve(_project)
    if where.get("ok") is False:
        return dict(where, endpoint=endpoint)
    project = where["project"]
    key = (_norm(project), endpoint, json.dumps(body, sort_keys=True))
    with _LOCK:
        now = time.time()
        for k in [k for k, (_r, t) in _TIMED_OUT.items() if now - t > KEEP_SECONDS]:
            del _TIMED_OUT[k]
        kept = _TIMED_OUT.get(key)
    rid = kept[0] if kept else uuid.uuid4().hex
    began = time.time()
    deadline = began + RELOAD_WAIT
    timeout = CALL_TIMEOUT if _timeout is None else _timeout
    retries = 0
    may_have_run = False      # a request reached a socket and no reply said what became of it

    def done(data):
        with _LOCK:
            _TIMED_OUT.pop(key, None)
        if retries and isinstance(data, dict):
            data = dict(data, retried=retries, waitedMs=int((time.time() - began) * 1000))
        return data

    st = state_of(project, _PIN.get("log"))
    if not st.get("answered"):
        st = _await_answer(project, deadline)
        if not st.get("answered"):
            return _failed_state(st, rid, endpoint, False)
    while True:
        port = (st.get("status") or {}).get("port") or st.get("port")
        want_pid = (st.get("status") or {}).get("pid") or st.get("pid")
        kind, data, headers = _raw_watched(port, endpoint, body, rid, timeout, (st.get("status") or {}).get("domain"), expect_pid=want_pid)
        if kind == "wrong_listener":
            return {"ok": False, "editorState": "wrongInstance", "refusedRule": "listener-identity", "endpoint": endpoint, "requestId": rid, "ran": False,
                    "error": "port %s is held by pid %s, not the editor this call meant (pid %s), so nothing was sent to it, the token included." % (port, (data or {}).get("pid"), want_pid)}
        if kind == "reply":
            instance = headers.get("X-Mif-Instance")
            if headers.get("X-Mif-Backend") != "unity":
                return {"ok": False, "editorState": "wrongInstance", "endpoint": endpoint, "requestId": rid, "outcomeUnknown": True,
                        "error": "port %s answered, and not as a Unity bridge. Not trusting the reply." % port, "holder": _port_holder(port)}
            if not instance:
                # Every reply of this bridge names its pid (HttpLite.Write). One without it skipped the check below (found 2026-11-22).
                return {"ok": False, "editorState": "wrongInstance", "endpoint": endpoint, "requestId": rid, "outcomeUnknown": True,
                        "error": "port %s answered as a Unity bridge with no X-Mif-Instance, so whose editor it is cannot be checked. Not trusting the reply." % port, "holder": _port_holder(port)}
            if str(st.get("pid")) != instance and str((st.get("status") or {}).get("pid")) != instance:
                return {"ok": False, "editorState": "wrongInstance", "endpoint": endpoint, "requestId": rid, "outcomeUnknown": True,
                        "error": "port %s was answered by pid %s, not the editor this call meant." % (port, instance)}
            if isinstance(data, dict) and data.get("editorState") == "reloading" and data.get("ran") is False:
                may_have_run = False
            elif isinstance(data, dict) and data.get("requestRunning"):
                may_have_run = True     # it is on the main thread now: keep asking with the same id, which runs nothing
            else:
                return done(data)
            retries += 1
            if time.time() >= deadline:
                return dict(data, endpoint=endpoint, requestId=rid, waitedMs=int((time.time() - began) * 1000))
            time.sleep(0.1)
            st = _await_answer(project, deadline)
            if not st.get("answered"):
                return _failed_state(st, rid, endpoint, may_have_run)
            continue
        if kind == "read_timeout":
            # A timed-out call may be running or finished. Say what the editor is doing and keep the id, so the identical call replays.
            with _LOCK:
                _TIMED_OUT[key] = (rid, time.time())
            st = state_of(project, _PIN.get("log"))
            out = {"ok": False, "endpoint": endpoint, "requestId": rid, "outcomeUnknown": True, "retryable": False, "editorState": st.get("editorState") or "busy", "checkWith": "uy_request_status",
                   "error": "no reply in %s s. The editor is alive and its main thread is occupied, or the call is long. It may have run. Repeating this exact call is safe: it carries the same id. Do not send a changed call." % timeout}
            if st.get("status"):
                out["inFlight"] = st["status"].get("inFlight")
                out["lastTickAgeMs"] = st["status"].get("lastTickAgeMs")
            return out
        if kind == "bad_body":
            return {"ok": False, "endpoint": endpoint, "requestId": rid, "outcomeUnknown": True, "editorState": "wrongInstance", "error": "port %s replied with something that is not this bridge's JSON." % port, "body": data}
        # refused, connect_timeout: nothing was delivered. cut, empty: it may have been. In every case the record decides, because the id runs at most once.
        may_have_run = kind in ("cut", "empty")
        retries += 1
        if time.time() >= deadline:
            # A DEADLINE ON THE RE-SENDS TOO (found 2026-11-22): an editor that cut every reply while its record kept answering unknown was asked
            # again forever, since _await_answer answers at once for a live editor and only this loop could stop.
            return _failed_state(dict(st, error="%s: the editor cut the connection on each of %d tries in %d s, and its record never said the call ran." % (endpoint, retries, int(time.time() - began))),
                                 rid, endpoint, may_have_run)
        st = _await_answer(project, deadline)
        if not st.get("answered"):
            return _failed_state(st, rid, endpoint, may_have_run)
        port = (st.get("status") or {}).get("port") or st.get("port")
        k2, record, _h2 = _raw(port, "request_status", {"requestId": rid}, read_timeout=5.0,
                               expect_pid=(st.get("status") or {}).get("pid") or st.get("pid"))
        if k2 != "reply" or not isinstance(record, dict):
            if time.time() >= deadline:
                return _failed_state(dict(st, error="the editor stopped answering while its record of this call was being read."), rid, endpoint, may_have_run)
            time.sleep(0.1)
            continue
        state = record.get("state")
        if state == "interrupted":
            return done({"ok": False, "interrupted": True, "outcomeUnknown": True, "retryable": False, "endpoint": endpoint, "requestId": rid, "checkWith": "uy_request_status",
                         "error": "%s started and never finished: a reload or a death cut it. It may have written. Read the target; do not send a changed call." % endpoint})
        if state == "running":
            if time.time() >= deadline:
                return {"ok": False, "requestRunning": True, "retryable": True, "endpoint": endpoint, "requestId": rid, "checkWith": "uy_request_status", "error": "still running after the wait."}
            time.sleep(0.1)
        # unknown: the handler never started, so the same id goes again. done: the same id answers from the record and runs nothing.
        # PACED (FOUND 2026-11-22): a call the editor drops before recording it, every time, was resent with no pause; the deadline above ends it,
        # and this keeps the resends from spinning until then.
        may_have_run = False
        time.sleep(0.1)


# -------------------------------------------------------------------------------------------------------------
# Tools. Thin on purpose: the editor's table holds the truth, and a guessed key is refused there too.
# -------------------------------------------------------------------------------------------------------------
def uy_use(project: str) -> dict:
    "Pin this server's Unity calls to one project, by its folder. Needed when several Unity editors run. Read only."
    if not project or not os.path.isdir(os.path.join(project, "Assets")):
        return {"ok": False, "error": "'%s' is not a Unity project: it has no Assets folder." % project, "instances": list_instances()}
    _PIN["project"] = project
    return dict(state_of(project), ok=True, pinned=project)


def uy_instances() -> dict:
    "List running Unity editors: those with the bridge answering, and those without, by measured process. Read only."
    live = list_instances()
    known = {_norm(r["project"]) for r in live}
    others = [{"pid": p["pid"], "project": p["project"]} for p in unity_processes() if p["project"] and p["project"] not in known]
    return {"ok": True, "pinned": _PIN["project"], "instances": live, "noBridge": others, "count": len(live) + len(others)}


def uy_status() -> dict:
    "Say what the Unity editor is doing: ready, compiling, errors, playing, reloading, stalled, starting, down, or serverNotLoaded (a live editor whose bridge assembly failed to compile, with file and line). On a Unity version no suite runs it adds untestedUnity with the tested list. Read only."
    where = _resolve()
    if where.get("ok") is False:
        return where
    st = state_of(where["project"], _PIN.get("log"))
    out = dict(st, ok=bool(st.get("answered")), clientProtocol=PROTOCOL)
    theirs = (st.get("status") or {}).get("protocol")
    if theirs is not None and theirs != PROTOCOL:
        # The package is source inside each project, so each project can carry a different version.
        out["protocolMismatch"] = "this server speaks protocol %s and the package in this project speaks %s. Update the older one." % (PROTOCOL, theirs)
    # The editor's own version when it answers, else what the project names: a project not yet opened is still on its file's version.
    notice = untested_unity((st.get("status") or {}).get("unity") or project_unity(where["project"]))
    if notice:
        out["untestedUnity"] = notice
    return out


def uy_wait_idle(timeoutSeconds: float = 60, settleSeconds: float = 3.0) -> dict:
    "Wait until the Unity editor is settled: answering, same script domain, not compiling or importing, for settleSeconds. A first answer is not ready: startup can reload again. Read only."
    where = _resolve()
    if where.get("ok") is False:
        return where
    project, deadline = where["project"], time.time() + timeoutSeconds
    domain, since, last = None, None, None
    while time.time() < deadline:
        last = state_of(project, _PIN.get("log"))
        if last.get("editorState") in ("down", "serverNotLoaded", "portBusy", "wrongInstance") or last.get("refusedRule") or last.get("blockedBy"):
            return dict(last, ok=False)
        s = last.get("status") or {}
        idle = last.get("answered") and s.get("state") in ("ready", "errors", "playing", "paused") and not s.get("compiling") and not s.get("updating") and (s.get("lastTickAgeMs") or 0) < STALLED_MS
        if idle and s.get("domain") == domain:
            if time.time() - since >= settleSeconds:
                return {"ok": True, "editorState": s.get("state"), "domain": domain, "lastReloadMs": s.get("lastReloadMs"), "reloads": s.get("reloads")}
        else:
            domain, since = (s.get("domain"), time.time()) if idle else (None, None)
        time.sleep(0.1)
    return dict(last or {}, ok=False, timedOut=True, error="not settled after %s s." % timeoutSeconds)


def uy_self_audit(includeEndpointDetails: bool = None) -> dict:
    "The bridge inside the editor: its version, Unity version, write mode and scratch limits, journal path, and every endpoint's safety row. Read only."
    return _unity("self_audit", includeEndpointDetails=includeEndpointDetails)


def uy_describe_endpoint(endpoint: str) -> dict:
    "One editor endpoint's parameters with types, what it returns, and its safety row. Ask this rather than guess a shape. Read only."
    return _unity("describe_endpoint", endpoint=endpoint)


def uy_request_status(requestId: str, includeResult: bool = None) -> dict:
    "What became of a request id after a timeout or a cut connection: running, done, interrupted or unknown (never started). Read only."
    return _unity("request_status", requestId=requestId, includeResult=includeResult)


def uy_call_log(limit: int = None, disk: bool = None) -> dict:
    "The call journal: recent calls from memory, or with disk the file, with calls that never ended and sessions that never closed. Read only."
    return _unity("call_log", limit=limit, disk=disk)


def uy_compile_status(includeWarnings: bool = None, limit: int = None) -> dict:
    "Whether the Unity project compiles, with every error by assembly, file, line and column. The previous build keeps running while errors stand. Read only."
    return _unity("compile_status", includeWarnings=includeWarnings, limit=limit)


# An identical call after a timed-out one carries the same id and is answered from the record, which is the contract and right for a write. The next three ask the editor to look or act NOW, and an old answer
# does neither: a refresh that had timed out through a long import made the next refresh, for new files, a no-op that said ok. The replay also clears the kept id, so going again is a new call.
def uy_refresh() -> dict:
    "AssetDatabase.Refresh: make the editor look at disk after a file changed. It can start a compile and a domain reload; follow with uy_wait_idle and uy_compile_status."
    r = _unity("refresh")
    return _unity("refresh") if r.get("replayed") else r


def uy_request_compile() -> dict:
    "Compile the project's scripts again even if nothing changed. Follow with uy_wait_idle and uy_compile_status."
    r = _unity("request_compile")
    return _unity("request_compile") if r.get("replayed") else r


def uy_request_reload() -> dict:
    "Reload the script domain without compiling. The bridge is back in about a second; a call made meanwhile waits and retries."
    r = _unity("request_reload")
    return _unity("request_reload") if r.get("replayed") else r


def uy_play(action: str, timeoutSeconds: float = 30) -> dict:
    "Enter, exit, pause or unpause play mode, and wait for the outcome: the editor refuses play silently, so this reads what really happened. In scratch mode, entering needs every loaded scene untitled or under Assets/_Mif; scene writes stay refused during play."
    r = _unity("play", action=action)
    if not (isinstance(r, dict) and r.get("ok") and r.get("changed") and r.get("seq") is not None):
        return r
    where = _resolve()
    deadline = time.time() + timeoutSeconds
    while time.time() < deadline:
        st = state_of(where["project"], _PIN.get("log"))
        p = (st.get("status") or {}).get("play") or {}
        # The event can land before the phase does: entered with the phase still entering, with a reload to come. Settled means both.
        settled = p.get("result") == "refused" or (p.get("result") == "entered" and p.get("phase") in ("playing", "paused")) or (p.get("result") == "exited" and p.get("phase") == "edit")
        if p.get("seq") == r["seq"] and settled:
            ok = p.get("result") != "refused"
            out = dict(r, ok=ok, result=p.get("result"), phase=p.get("phase"), verified=ok)
            if not ok:
                out["error"] = "the editor refused to enter play mode: it left edit mode and came straight back. Read uy_compile_status."
            return out
        if st.get("editorState") in ("down", "serverNotLoaded"):
            return dict(r, ok=False, editorState=st.get("editorState"), error=st.get("error"))
        time.sleep(0.05)
    return dict(r, ok=False, timedOut=True, error="no outcome after %s s; read uy_status." % timeoutSeconds)


def uy_get_property(target: dict, path: str) -> dict:
    "Read one serialized property of any object, by serialized path, with its type. target is {path|id|asset|guid, component?, componentIndex?, scene?, subAsset?}."
    return _unity("get_property", target=target, path=path)


def uy_list_properties(target: dict, nameContains: str = None, includeValues: bool = None, limit: int = None, deep: bool = None) -> dict:
    "Every serialized property of an object, walked live, so the names are this Unity version's own."
    return _unity("list_properties", target=target, nameContains=nameContains, includeValues=includeValues, limit=limit, deep=deep)


def uy_set_property(target: dict, path: str, value) -> dict:
    "Write one serialized property: checked against its own type, read back after. ok means the editor holds the value. A clamp or a refused reference answers ok:false with what it holds."
    return _unity("set_property", target=target, path=path, value=NULL if value is None else value)


def uy_list_scenes() -> dict:
    "The open scenes: path, loaded, dirty, root count, and whether scratch mode may write to each. Read only."
    return _unity("list_scenes")


def uy_list_hierarchy(root: dict = None, scene: str = None, depth: int = None, nameContains: str = None, components: bool = None, limit: int = None, world: bool = None) -> dict:
    "The GameObjects of the loaded scenes, or under one object, to a depth, with a name filter. world adds each one's world position and its own Renderer's world bounds. Read only."
    return _unity("list_hierarchy", root=root, scene=scene, depth=depth, nameContains=nameContains, components=components, limit=limit, world=world)


def uy_get_object(target: dict) -> dict:
    "One GameObject: identity, active state, tag, layer, local and world transform, the world bounds of its own Renderer and Collider and of every drawn Renderer under it, components, children. Read only."
    return _unity("get_object", target=target)


def uy_scene_bounds(root: dict = None, scene: str = None, source: str = None, includeInactive: bool = None) -> dict:
    "The world box round what the loaded scenes, one scene or one root object's subtree draw (source renderers, the default) or collide with (source colliders), with the count, the skipped and the five largest contributors; includeInactive counts disabled ones and inactive objects too. Nothing to measure fails. Read only."
    return _unity("scene_bounds", root=root, scene=scene, source=source, includeInactive=includeInactive)


def uy_create_gameobject(name: str, parent: dict = None, primitive: str = None, position: list = None) -> dict:
    "Create a GameObject, empty or a primitive, at the scene root or under a parent. Undoable, read back by id."
    return _unity("create_gameobject", name=name, parent=parent, primitive=primitive, position=position)


def uy_delete_gameobject(target: dict) -> dict:
    "Delete a GameObject and its children. Undoable, verified gone."
    return _unity("delete_gameobject", target=target)


def uy_rename_gameobject(target: dict, name: str) -> dict:
    "Rename a GameObject. Undoable, read back."
    return _unity("rename_gameobject", target=target, name=name)


def uy_reparent_gameobject(target: dict, parent: dict = None, worldPositionStays: bool = None) -> dict:
    "Move a GameObject under a parent, or to the scene root with none. Undoable, read back; a cycle is refused."
    return _unity("reparent_gameobject", target=target, parent=parent, worldPositionStays=worldPositionStays)


def uy_set_active(target: dict, active: bool) -> dict:
    "Set a GameObject's own active flag. Undoable, read back; says when a parent keeps it inactive anyway."
    return _unity("set_active", target=target, active=active)


def uy_add_component(target: dict, type: str) -> dict:
    "Add a component by short or full type name. Undoable, read back; an abstract type, a duplicate Unity refuses or an unknown name is a typed failure."
    return _unity("add_component", target=target, type=type)


def uy_remove_component(target: dict) -> dict:
    "Remove one component. Undoable, verified gone: a component another one requires stays, and that reads as a failure."
    return _unity("remove_component", target=target)


def uy_find_assets(filter: str, folder: str = None, limit: int = None) -> dict:
    "Find assets with a filter such as 't:Material name', optionally under a folder. Read only."
    return _unity("find_assets", filter=filter, folder=folder, limit=limit)


def uy_create_folder(path: str) -> dict:
    "Create a folder and its parents. Asset writes go to disk at once; in scratch mode only under Assets/_Mif. Read back."
    return _unity("create_folder", path=path)


def uy_create_asset(path: str, type: str, shader: str = None) -> dict:
    "Create a Material, or a ScriptableObject by type name, at a path. Asset writes go to disk at once; in scratch mode only under Assets/_Mif. Read back from the asset database."
    return _unity("create_asset", path=path, type=type, shader=shader)


def uy_delete_asset(path: str) -> dict:
    "Delete an asset from disk. Not undoable. In scratch mode only under Assets/_Mif."
    return _unity("delete_asset", path=path)


def uy_save_scene(scene: str = None, path: str = None, asCopy: bool = None) -> dict:
    "Save a loaded scene to its path or a new one, and read it back. In scratch mode only under Assets/_Mif. asCopy writes a copy and leaves the open scene on its own path: the default outside full mode for a scene saved outside scratch."
    return _unity("save_scene", scene=scene, path=path, asCopy=asCopy)


def uy_new_scene(discardUnsaved: bool = None, defaults: bool = None) -> dict:
    "Replace the open scenes with a new untitled one. Refused when a dirty scene would be lost, unless discardUnsaved."
    return _unity("new_scene", discardUnsaved=discardUnsaved, defaults=defaults)


def uy_open_scene(path: str, discardUnsaved: bool = None) -> dict:
    "Open a scene alone, replacing the open ones. Refused when a dirty scene would be lost, unless discardUnsaved. In scratch mode only under Assets/_Mif."
    return _unity("open_scene", path=path, discardUnsaved=discardUnsaved)


# Materials, textures, lights, cameras, volumes, terrain, particles, animation. Each write is checked before it writes and read back after.
def uy_rp_info() -> dict:
    "Which render pipeline the project uses right now (URP, HDRP, Builtin), its asset, its default shader and materials, and which pipeline packages are installed. Read this before anything pipeline-sensitive. Read only."
    return _unity("rp_info")


def uy_urp_create_asset(path: str, rendererPath: str) -> dict:
    "Create a URP pipeline asset and its renderer data. It does not assign it: uy_rp_assign does. Refused when URP is not installed."
    return _unity("urp_create_asset", path=path, rendererPath=rendererPath)


def uy_rp_assign(target: dict = None, level=None, scope: str = None) -> dict:
    "Assign a pipeline asset, or Built-in with no target. scope current (default: the project default and the current quality level), level (one quality level, with level), default, or allLevels. Changes project settings, so scratch mode refuses it. Read back."
    return _unity("rp_assign", target=target, level=level, scope=scope)


def uy_quality_levels_get(platform: str = None, settings: bool = None) -> dict:
    "Every quality level: name, pipeline asset or inheriting the default, its rendering settings read from the serialized level, and which is current. Read only."
    return _unity("quality_levels_get", platform=platform, settings=settings)


def uy_quality_level_set(level, settings: dict = None, makeCurrent: bool = None, name: str = None) -> dict:
    "Change one quality level's settings (shadows, LOD bias, anti-aliasing, vSync, ...), rename it or make it current. Project settings: scratch mode refuses it. Saved, read back."
    return _unity("quality_level_set", level=level, settings=settings, makeCurrent=makeCurrent, name=name)


def uy_shader_props(shader: str = None, target: dict = None, includeHidden: bool = None, asset: dict = None) -> dict:
    "A shader's properties from the shader itself: names, types, ranges, defaults, its main color and main texture, its keywords, and whether it runs on this project's pipeline. Names differ by pipeline; ask rather than guess. Read only."
    return _unity("shader_props", shader=shader, target=target, includeHidden=includeHidden, asset=asset)


def uy_shader_errors(shader: str = None, target: dict = None, asset: dict = None) -> dict:
    "Whether a shader compiled: errors and warnings by platform and line, and whether a compile is still going. Read only."
    return _unity("shader_errors", shader=shader, target=target, asset=asset)


def uy_material_create(path: str, shader: str = None, parent: dict = None, properties: dict = None, keywords: list = None, shaderAsset: dict = None) -> dict:
    "Create a material with this pipeline's default shader or a named one, optionally a variant, with properties and keywords. A shader for another pipeline is refused, since it renders magenta with no message. Saved to disk and read back."
    return _unity("material_create", path=path, shader=shader, parent=parent, properties=properties, keywords=keywords, shaderAsset=shaderAsset)


def uy_material_get(target: dict, slot: int = None, nameContains: str = None, detail: bool = None) -> dict:
    "A material's shader, every property's value and type, keywords, render queue and variant parent, from an asset or a renderer's slot. Read only."
    return _unity("material_get", target=target, slot=slot, nameContains=nameContains, detail=detail)


def uy_material_set(target: dict, slot: int = None, properties: dict = None, enableKeywords: list = None, disableKeywords: list = None, renderQueue: int = None, passes: dict = None,
                    instancing: bool = None, giFlags: str = None, doubleSidedGI: bool = None, overrideTags: dict = None) -> dict:
    "Set material properties, keywords and queue. Names and types are checked against the shader first, a value outside a Range is refused, the shader's own validation runs after, as the inspector's would, and each value is read back. A material asset is saved to disk."
    return _unity("material_set", target=target, slot=slot, properties=properties, enableKeywords=enableKeywords, disableKeywords=disableKeywords, renderQueue=renderQueue, passes=passes,
                  instancing=instancing, giFlags=giFlags, doubleSidedGI=doubleSidedGI, overrideTags=overrideTags)


def uy_material_assign(target: dict, material: dict, slot: int = None) -> dict:
    "Put a material in a renderer's slot. A slot out of range and a material for another pipeline are refused. Undoable. Read back."
    return _unity("material_assign", target=target, material=material, slot=slot)


def uy_shader_graph_edit(asset: str = None, operation: str = None) -> dict:
    "Always refuses, with what cannot be done and what can. Shader Graph authoring has no public API in Unity. Materials and exposed properties of a graph's shader work through uy_material_* and uy_shader_props."
    return _unity("shader_graph_edit", asset=asset, operation=operation)


def uy_vfx_graph_edit(asset: str = None, operation: str = None) -> dict:
    "Always refuses, with what cannot be done and what can. VFX Graph authoring has no public API in Unity. uy_vfx_read reads a graph's exposed face; Particle Systems are fully public."
    return _unity("vfx_graph_edit", asset=asset, operation=operation)


def uy_graph_templates(kind: str, includeProject: bool = None) -> dict:
    "The template graphs a project can copy: Shader Graph's (none before 17.3, said so) or VFX Graph's, listed live; includeProject adds .vfx files marked as templates. Read only."
    return _unity("graph_templates", kind=kind, includeProject=includeProject)


def uy_shader_graph_create(path: str, template: str = None, source: dict = None) -> dict:
    "A .shadergraph or .shadersubgraph copied from a package template or a project graph. Read back: same bytes, and a graph loads as a Shader with no error."
    return _unity("shader_graph_create", path=path, template=template, source=source)


def uy_shader_graph_info(target: dict, usedBy: bool = None, limit: int = None) -> dict:
    "What a graph file compiled to: targets, passes and LightModes, keywords, errors, sub-graphs and HLSL it uses, and with usedBy what uses it. Read only."
    return _unity("shader_graph_info", target=target, usedBy=usedBy, limit=limit)


def uy_shader_info(shader: str = None, target: dict = None, asset: dict = None, passKeywords: bool = None, includes: bool = None) -> dict:
    "A shader's subshaders, passes (name, LightMode, stages, keywords), tags, render queue, custom editor and include files. Read only."
    return _unity("shader_info", shader=shader, target=target, asset=asset, passKeywords=passKeywords, includes=includes)


def uy_compute_shader_info(target: dict) -> dict:
    "A compute shader's kernels from its #pragma lines confirmed by HasKernel, thread group sizes (with a device), keywords and messages. Read only."
    return _unity("compute_shader_info", target=target)


def uy_shader_variant_compile(shaderPass, shader: str = None, target: dict = None, asset: dict = None, subshader: int = None, stage: str = None, keywords: list = None,
                              platform: str = None, buildTarget: str = None, preprocessOnly: bool = None) -> dict:
    "Compile one pass variant for a stage, keywords and platform (or only preprocess it): success, messages, constant buffers, texture bindings. Changes nothing saved."
    return _unity("shader_variant_compile", shader=shader, target=target, asset=asset, subshader=subshader, stage=stage, keywords=keywords, platform=platform,
                  buildTarget=buildTarget, preprocessOnly=preprocessOnly, shaderPass=shaderPass)


def uy_shader_variants_create(path: str, variants: list = None, fromMaterials: list = None) -> dict:
    "A ShaderVariantCollection with its first variants, given or from materials. Invalid variants refused with Unity's own message first. Read back."
    return _unity("shader_variants_create", path=path, variants=variants, fromMaterials=fromMaterials)


def uy_shader_variants_edit(target: dict, add: list = None, remove: list = None, clear: bool = None) -> dict:
    "Add, remove or clear variants in a ShaderVariantCollection. Invalid variants refused first. Read back."
    return _unity("shader_variants_edit", target=target, add=add, remove=remove, clear=clear)


def uy_shader_variants_get(target: dict, limit: int = None) -> dict:
    "What a ShaderVariantCollection holds: counts, and each variant from its serialized list when that agrees with the count. Read only."
    return _unity("shader_variants_get", target=target, limit=limit)


def uy_shader_variants_warmup(target: dict, maxVariants: int = None) -> dict:
    "Compile a collection's variants on this editor's graphics device now. Needs a graphics device. Changes nothing saved."
    return _unity("shader_variants_warmup", target=target, maxVariants=maxVariants)


def uy_material_variant_set(target: dict, parent=None, revert=None, lock: list = None, unlock: list = None, applyToParent: list = None) -> dict:
    "A material variant's parent, reverted overrides, locks, unlocks, and overrides applied to its parent. Loops and ancestor locks refused first. Saved. Read back per property."
    return _unity("material_variant_set", target=target, parent=parent, revert=revert, lock=lock, unlock=unlock, applyToParent=applyToParent)


def uy_texture_platform_set(target: dict, platform: str, clear: bool = None, maxTextureSize: int = None, format: str = None, compression: str = None, compressionQuality: int = None,
                            crunched: bool = None, resizeAlgorithm: str = None) -> dict:
    "A texture's per-platform import override (size, format checked for the platform first, compression, quality, crunch), or clear it. Reimported, read back."
    return _unity("texture_platform_set", target=target, platform=platform, clear=clear, maxTextureSize=maxTextureSize, format=format, compression=compression,
                  compressionQuality=compressionQuality, crunched=crunched, resizeAlgorithm=resizeAlgorithm)


def uy_render_texture_create(path: str, width: int, height: int, format: str = None, depthStencil: str = None, dimension: str = None, volumeDepth: int = None, antiAliasing: int = None,
                             mipmaps: bool = None, autoGenerateMips: bool = None, randomWrite: bool = None, custom: dict = None) -> dict:
    "A RenderTexture (or CustomRenderTexture) asset from a descriptor. A format the device cannot render to is refused when a device exists. Read back from the reloaded asset."
    return _unity("render_texture_create", path=path, width=width, height=height, format=format, depthStencil=depthStencil, dimension=dimension, volumeDepth=volumeDepth,
                  antiAliasing=antiAliasing, mipmaps=mipmaps, autoGenerateMips=autoGenerateMips, randomWrite=randomWrite, custom=custom)


def uy_texture_array_create(path: str, kind: str, sources: list, mipmaps: bool = None, linear: bool = None, wrapMode: str = None, filterMode: str = None) -> dict:
    "A Texture2DArray, Texture3D or Cubemap asset assembled from readable textures, slice by slice. Read back by checksum per slice."
    return _unity("texture_array_create", path=path, kind=kind, sources=sources, mipmaps=mipmaps, linear=linear, wrapMode=wrapMode, filterMode=filterMode)


def uy_shader_create(path: str, source: str, overwrite: bool = None) -> dict:
    "Write a .shader, .hlsl or .cginc from source and import it; a shader is read back with its compile messages and lines. A path write like any asset."
    return _unity("shader_create", path=path, source=source, overwrite=overwrite)


def uy_renderer_property_block_get(target: dict, slot: int = None) -> dict:
    "What a renderer's MaterialPropertyBlock overrides, probed by its material's property names. Read only."
    return _unity("renderer_property_block_get", target=target, slot=slot)


def uy_texture_info(target: dict) -> dict:
    "A texture as imported: size, format, mips, and every import setting uy_texture_import_set can change. Read only."
    return _unity("texture_info", target=target)


def uy_texture_import_set(target: dict, settings: dict) -> dict:
    "Change a texture's import settings: type, sRGB, mips, readable, max size, filter, wrap, compression and more. Checked first, then reimported, then read back from a fresh importer. Ask uy_describe_endpoint for the names."
    return _unity("texture_import_set", target=target, settings=settings)


def uy_texture_create(path: str, width: int, height: int, color: list = None, checkerColor: list = None, checkerSize: int = None) -> dict:
    "Write a PNG, solid or a checker, and import it. Read back from the file's bytes and the imported texture."
    return _unity("texture_create", path=path, width=width, height=height, color=color, checkerColor=checkerColor, checkerSize=checkerSize)


def uy_light_get(target: dict) -> dict:
    "A Light's settings, with its unit and the pipeline that gives the unit its meaning. Read only."
    return _unity("light_get", target=target)


def uy_light_create(name: str, parent: dict = None, position: list = None, rotation: list = None, settings: dict = None) -> dict:
    "Create a GameObject with a Light. Settings are checked before the object exists. Undoable. Read back. Ask uy_describe_endpoint for the setting names."
    return _unity("light_create", name=name, parent=parent, position=position, rotation=rotation, settings=settings)


def uy_light_set(target: dict, settings: dict) -> dict:
    "Change a Light. Checked before anything is written. What Unity clamps or snaps comes back as ok:false with the rows that differ. The reply carries the unit and the pipeline. Undoable."
    return _unity("light_set", target=target, settings=settings)


def uy_camera_get(target: dict) -> dict:
    "A Camera's settings. Read only."
    return _unity("camera_get", target=target)


def uy_camera_create(name: str, parent: dict = None, position: list = None, rotation: list = None, settings: dict = None) -> dict:
    "Create a GameObject with a Camera. Settings are checked before the object exists. Undoable. Read back."
    return _unity("camera_create", name=name, parent=parent, position=position, rotation=rotation, settings=settings)


def uy_camera_set(target: dict, settings: dict = None, pipelineSettings: dict = None) -> dict:
    "Change a Camera, and on URP its pipeline data (renderPostProcessing, antialiasing, rendererIndex, volumeLayerMask, volumeTrigger, cameraStack, ...). Checked before anything is written; a near plane at or past the far plane is refused. Undoable. Read back."
    return _unity("camera_set", target=target, settings=settings, pipelineSettings=pipelineSettings)


def uy_volume_profile_create(path: str) -> dict:
    "Create an empty Volume profile. URP and HDRP only: Built-in has no Volume system, and this refuses by name. Read back."
    return _unity("volume_profile_create", path=path)


def uy_volume_get(target: dict, component: str = None) -> dict:
    "A Volume profile's components, each parameter's value, type and override state, and the component types this pipeline offers. Read only."
    return _unity("volume_get", target=target, component=component)


def uy_volume_set(target: dict, component: str, values: dict = None, add: bool = None, active: bool = None, overrideAll: bool = None, clearOverrides: list = None) -> dict:
    "Set a Volume component's parameters, turning each override on, adding the component when asked (with no values too); overrideAll turns every override on or off, clearOverrides turns named ones off. Checked on a throwaway instance first. Read back."
    return _unity("volume_set", target=target, component=component, values=values, add=add, active=active, overrideAll=overrideAll, clearOverrides=clearOverrides)


def uy_bake_lighting(action: str, lightmapper: str = None) -> dict:
    "Start, cancel or clear a lightmap bake of the active scene (clear: its baked data and lighting data asset; clearCache: the GI cache, full mode only). A bake runs in the background: poll uy_bake_status. An untitled scene is refused, because a bake writes beside the scene."
    return _unity("bake_lighting", action=action, lightmapper=lightmapper)


def uy_bake_status() -> dict:
    "Whether a bake is running, and what the active scene's last bake left. Read only."
    return _unity("bake_status")


CAPTURE_SHADER_WAIT = 60


def uy_capture_camera(target: dict, name: str = None, width: int = None, height: int = None, allowBlank: bool = None) -> dict:
    "Render a camera to a PNG under Library/MifBridge/captures, with no window. A uniform image fails unless allowBlank. An editor started with -nographics refuses by name."
    # A compile still in flight draws a wrong picture, and its flag only falls on a tick after the handler ends, so the waiting is out here.
    start = time.time()
    while True:
        reply = _unity("capture_camera", target=target, name=name, width=width, height=height, allowBlank=allowBlank)
        if not (isinstance(reply, dict) and reply.get("shadersCompiling")) or time.time() - start > CAPTURE_SHADER_WAIT:
            break
        time.sleep(0.5)
    if isinstance(reply, dict) and time.time() - start > 1:
        reply["seconds"] = round(time.time() - start, 1)
    return reply


def uy_terrain_create(name: str, dataPath: str, heightmapResolution: int = None, size: list = None, alphamapResolution: int = None, parent: dict = None, position: list = None,
                      detailResolution: int = None, detailPerPatch: int = None, tiles: list = None, groupingID: int = None, copyLayersFrom: dict = None) -> dict:
    "Create a Terrain and its TerrainData asset, with this pipeline's terrain material; or tiles [nx, nz] of them under a dataPath folder, sharing a groupingID and connected. A resolution Unity would snap is refused. Read back."
    return _unity("terrain_create", name=name, dataPath=dataPath, heightmapResolution=heightmapResolution, size=size, alphamapResolution=alphamapResolution, parent=parent, position=position,
                  detailResolution=detailResolution, detailPerPatch=detailPerPatch, tiles=tiles, groupingID=groupingID, copyLayersFrom=copyLayersFrom)
# ---- 1.5 batch 4: render pipeline and lighting
def uy_urp_asset_get(target: dict = None) -> dict:
    "Every setting of a URP pipeline asset, typed, with whether each is written through its setter or its serialized field, and its renderers. Default: the current asset when it is URP. Read only."
    return _unity("urp_asset_get", target=target)


def uy_urp_asset_set(target: dict, settings: dict) -> dict:
    "Change URP asset settings by name (render scale, HDR, MSAA, upscaling, shadows, cascades, additional lights, soft shadows, light modes, SRP batcher, probe system, volume profile, defaultRendererIndex). Checked first, saved, read back."
    return _unity("urp_asset_set", target=target, settings=settings)


def uy_urp_renderer_get(target: dict = None, asset: dict = None, index: int = None, featureSettings: bool = None) -> dict:
    "One URP renderer: type, settings, renderer features (active or not), whether the feature list and map agree, and every feature type available. Read only."
    return _unity("urp_renderer_get", target=target, asset=asset, index=index, featureSettings=featureSettings)


def uy_urp_renderer_feature_add(target: dict, feature: str, name: str = None, active: bool = None, settings: dict = None) -> dict:
    "Add a renderer feature (FullScreenPassRendererFeature, RenderObjects, DecalRendererFeature, ...) to a URP renderer as URP's inspector does. DisallowMultiple and SupportedOnRenderer honored. Read back."
    return _unity("urp_renderer_feature_add", target=target, feature=feature, name=name, active=active, settings=settings)


def uy_urp_renderer_feature_set(target: dict, feature, active: bool = None, moveTo: int = None, settings: dict = None) -> dict:
    "Enable, disable, reorder or change the serialized settings of one renderer feature. Read back."
    return _unity("urp_renderer_feature_set", target=target, feature=feature, active=active, moveTo=moveTo, settings=settings)


def uy_urp_renderer_feature_remove(target: dict, feature) -> dict:
    "Remove a renderer feature and its sub-asset, list and map together. Read back."
    return _unity("urp_renderer_feature_remove", target=target, feature=feature)


def uy_urp_renderer_add(target: dict, renderer: dict, makeDefault: bool = None) -> dict:
    "Add a renderer data asset to a URP asset's renderer list, optionally as the default. Read back."
    return _unity("urp_renderer_add", target=target, renderer=renderer, makeDefault=makeDefault)


def uy_hdrp_create_asset(path: str) -> dict:
    "Create an HDRP pipeline asset; uy_rp_assign assigns it. Refused when HDRP is not installed. Read back."
    return _unity("hdrp_create_asset", path=path)


def uy_hdrp_asset_get(target: dict = None) -> dict:
    "An HDRP asset's RenderPipelineSettings by field name (supportSSR, supportSSAO, supportVolumetrics, ...), nested settings as objects, and its volume profile. Read only."
    return _unity("hdrp_asset_get", target=target)


def uy_hdrp_asset_set(target: dict, settings: dict = None, volumeProfile=None) -> dict:
    "Change top-level scalar fields of an HDRP asset's RenderPipelineSettings by name, and its volume profile. A nested struct is refused by name. Saved, read back."
    return _unity("hdrp_asset_set", target=target, settings=settings, volumeProfile=volumeProfile)


def uy_hdrp_camera_frame_settings_set(target: dict, fields: dict = None, custom: bool = None, get: list = None) -> dict:
    "Per-camera HDRP frame settings: {FrameSettingsField name: bool} with their override bits, or get:[names] to read. HDRP only. Undoable. Read back."
    return _unity("hdrp_camera_frame_settings_set", target=target, fields=fields, custom=custom, get=get)


def uy_volume_object_create(name: str, parent: dict = None, position: list = None, rotation: list = None, profile: dict = None, settings: dict = None) -> dict:
    "Create a GameObject with a Volume: global or local, priority, weight, blend distance, profile. URP and HDRP only. Undoable. Read back; a local volume with no collider says so."
    return _unity("volume_object_create", name=name, parent=parent, position=position, rotation=rotation, profile=profile, settings=settings)


def uy_volume_object_get(target: dict) -> dict:
    "A Volume component's mode, priority, weight, blend distance, profile, and whether the profile is shared or instanced. Never instantiates the profile. Read only."
    return _unity("volume_object_get", target=target)


def uy_volume_object_set(target: dict, settings: dict) -> dict:
    "Change a Volume component's isGlobal, priority, weight, blendDistance or profile (null clears). Undoable. Read back."
    return _unity("volume_object_set", target=target, settings=settings)


def uy_volume_component_remove(target: dict, component: str) -> dict:
    "Remove an override (Bloom, Tonemapping, ...) from a Volume profile, its sub-asset included. Read back: the profile no longer has it and the file holds no object of that type."
    return _unity("volume_component_remove", target=target, component=component)


def uy_volume_default_profile_set(profile: dict) -> dict:
    "Set the project's default Volume profile for URP or HDRP (Unity 6 and later), saved in the global settings asset. Project settings: scratch mode refuses it. Read back."
    return _unity("volume_default_profile_set", profile=profile)


def uy_graphics_settings_get() -> dict:
    "Graphics settings that change how lights and sorting behave: linear intensity, color temperature, SRP batching, transparency sort, light probe outside-hull strategy, camera-relative culling. Read only."
    return _unity("graphics_settings_get")


def uy_graphics_settings_set(settings: dict) -> dict:
    "Change those graphics settings, saved at once. Project settings: scratch mode refuses it. Read back."
    return _unity("graphics_settings_set", settings=settings)


def uy_lighting_settings_get(target: dict = None, scene: str = None) -> dict:
    "Every field of a LightingSettings asset, or of the active (or a loaded) scene's; a scene with none says it bakes with Unity's defaults. Read only."
    return _unity("lighting_settings_get", target=target, scene=scene)


def uy_lighting_settings_create(path: str, fields: dict = None, assignToActiveScene: bool = None) -> dict:
    "Create a LightingSettings asset (.lighting) with chosen fields, optionally assigned to the active scene. Checked first, read back."
    return _unity("lighting_settings_create", path=path, fields=fields, assignToActiveScene=assignToActiveScene)


def uy_lighting_settings_set(fields: dict, target: dict = None) -> dict:
    "Change fields of a LightingSettings asset, default the active scene's. Checked first, saved, read back."
    return _unity("lighting_settings_set", target=target, fields=fields)


def uy_render_settings_get() -> dict:
    "The active scene's environment: ambient, skybox, sun, fog, reflections. Read only."
    return _unity("render_settings_get")


def uy_render_settings_set(settings: dict) -> dict:
    "Change the active scene's ambient, skybox, sun, fog and reflection settings. A non-skybox shader, a sun that is not directional and a negative fog density are refused. Undoable. Read back."
    return _unity("render_settings_set", settings=settings)


def uy_static_flags_set(targets: list, set: list = None, clear: list = None, children: bool = None) -> dict:
    "Set or clear static editor flags (ContributeGI, OccluderStatic, OccludeeStatic, BatchingStatic, ReflectionProbeStatic) on objects, optionally their children. Undoable. Read back."
    return _unity("static_flags_set", targets=targets, set=set, clear=clear, children=children)


def uy_static_flags_get(targets: list, children: bool = None) -> dict:
    "Objects' static editor flags by name, and isStatic. Read only."
    return _unity("static_flags_get", targets=targets, children=children)


def uy_lightmap_report(renderers: bool = None, limit: int = None, allowEmpty: bool = None) -> dict:
    "What the last bake produced: lightmaps and texel count, lightmaps mode, light probes, each renderer's lightmap index and scale-offset, each light's baking output. Read only."
    return _unity("lightmap_report", renderers=renderers, limit=limit, allowEmpty=allowEmpty)


def uy_renderer_gi_set(target: dict, settings: dict = None, get: bool = None) -> dict:
    "How a mesh renderer takes part in GI: receiveGI, scaleInLightmap, stitchLightmapSeams, probe usage. get:true reads. Undoable. Read back."
    return _unity("renderer_gi_set", target=target, settings=settings, get=get)


def uy_light_probe_group_set(target: dict = None, name: str = None, parent: dict = None, position: list = None, positions: list = None, grid: dict = None, dering: bool = None,
                             allowDuplicates: bool = None) -> dict:
    "Create or change a Light Probe Group: positions (local) or a grid {min, max, step}, dering. Duplicates and more than 10000 probes refused. Undoable. Read back."
    return _unity("light_probe_group_set", target=target, name=name, parent=parent, position=position, positions=positions, grid=grid, dering=dering, allowDuplicates=allowDuplicates)


def uy_light_probe_group_get(target: dict, world: bool = None) -> dict:
    "A Light Probe Group's positions, count, dering and bounds. Read only."
    return _unity("light_probe_group_get", target=target, world=world)


def uy_light_probes_sample(positions: list, direction: list = None) -> dict:
    "The baked light at points from the scene's light probes: SH coefficients and a color in a direction. No baked probes is refused. Read only."
    return _unity("light_probes_sample", positions=positions, direction=direction)


def uy_reflection_probe_set(target: dict = None, name: str = None, parent: dict = None, position: list = None, settings: dict = None, get: bool = None) -> dict:
    "Create or change a Reflection Probe: mode, resolution (power of two), size, center, box projection, importance, intensity, clipping, custom cubemap. get:true reads. Undoable. Read back."
    return _unity("reflection_probe_set", target=target, name=name, parent=parent, position=position, settings=settings, get=get)


def uy_reflection_probe_bake(target: dict, path: str, allowBlank: bool = None) -> dict:
    "Bake one reflection probe to a cubemap file (.exr HDR, .png LDR); each face's mean color is read back. Needs a graphics device."
    return _unity("reflection_probe_bake", target=target, path=path, allowBlank=allowBlank)


def uy_probe_volume_set(target: dict = None, name: str = None, parent: dict = None, position: list = None, settings: dict = None, get: bool = None) -> dict:
    "Create or change an Adaptive Probe Volume (mode Global/Scene/Local, size, subdivision overrides, fill empty spaces). URP and HDRP; says when the URP asset's probe system is off. Undoable. Read back."
    return _unity("probe_volume_set", target=target, name=name, parent=parent, position=position, settings=settings, get=get)


def uy_probe_volume_baking_set(path: str = None, target: dict = None, addActiveScene: bool = None, removeScene: str = None, scenario: str = None, settings: dict = None) -> dict:
    "Create or change an APV baking set (Unity 6+): add the active scene, remove a scene, add a scenario, probe spacing, simplification, sky occlusion. Read back, with hasBakedData."
    return _unity("probe_volume_baking_set", path=path, target=target, addActiveScene=addActiveScene, removeScene=removeScene, scenario=scenario, settings=settings)


def uy_lppv_set(target: dict = None, name: str = None, parent: dict = None, position: list = None, settings: dict = None, get: bool = None) -> dict:
    "Create or change a Light Probe Proxy Volume (Built-in only; URP and HDRP refuse by name). Undoable. Read back."
    return _unity("lppv_set", target=target, name=name, parent=parent, position=position, settings=settings, get=get)


def uy_lod_group_set(target: dict, add: bool = None, lods: list = None, fadeMode: str = None, animateCrossFading: bool = None, size: float = None, localReferencePoint: list = None,
                     recalculateBounds: bool = None) -> dict:
    "Create or change a LOD Group: levels [{height, renderers, fadeTransitionWidth}] from LOD0 down, fade mode, size, reference point. Heights must decrease; renderers must be under the target. Undoable. Read back."
    return _unity("lod_group_set", target=target, add=add, lods=lods, fadeMode=fadeMode, animateCrossFading=animateCrossFading, size=size, localReferencePoint=localReferencePoint, recalculateBounds=recalculateBounds)


def uy_lod_group_get(target: dict) -> dict:
    "A LOD Group's levels with renderers, heights and triangle counts, fade mode, size, the LOD bias, and renderers missing. Read only."
    return _unity("lod_group_get", target=target)


def uy_occlusion_bake(action: str, settings: dict = None) -> dict:
    "Start, cancel or clear an occlusion culling bake (start runs in the background: poll uy_occlusion_status). Nothing flagged Occluder or Occludee is refused."
    return _unity("occlusion_bake", action=action, settings=settings)


def uy_occlusion_status() -> dict:
    "Whether an occlusion bake runs, the occlusion data size, the bake settings, and how many objects are flagged. Read only."
    return _unity("occlusion_status")


def uy_occlusion_area_set(target: dict = None, name: str = None, parent: dict = None, position: list = None, center: list = None, size: list = None, open: bool = None) -> dict:
    "Create or change an Occlusion Area (center, size), or open or close an Occlusion Portal. Undoable. Read back."
    return _unity("occlusion_area_set", target=target, name=name, parent=parent, position=position, center=center, size=size, open=open)


def uy_lightmap_uv_generate(target: dict, params: dict = None) -> dict:
    "Generate lightmap UVs (UV2) for a mesh .asset the project owns; a model's mesh is refused (use set_importer generateSecondaryUV). Read back: one UV2 per vertex, all in [0,1]."
    return _unity("lightmap_uv_generate", target=target, params=params)


def uy_mesh_lod_generate(target: dict, limit: int = None, discardOddLevels: bool = None) -> dict:
    "Generate Mesh LODs inside a mesh .asset (Unity 6.3 and later; older editors refuse by name). Read back level by level."
    return _unity("mesh_lod_generate", target=target, limit=limit, discardOddLevels=discardOddLevels)


def uy_mesh_lod_info(target: dict) -> dict:
    "A mesh's LOD levels (Unity 6.3 and later), and a renderer's forced level and bias. Read only."
    return _unity("mesh_lod_info", target=target)



def uy_terrain_info(target: dict, sections: list = None) -> dict:
    "A terrain: resolutions, size, height statistics, layers with coverage, trees, material; sections (default all): settings, neighbors, holes, trees, details, collider. Read only."
    return _unity("terrain_info", target=target, sections=sections)


def uy_terrain_set_heights(target: dict, x: int = None, y: int = None, heights: list = None, fill: float = None, width: int = None, height: int = None) -> dict:
    "Write heights, 0 to 1 of the terrain's height, from rows or a fill. Bounds and values are checked first. Undoable. Read back through GetHeights."
    return _unity("terrain_set_heights", target=target, x=x, y=y, heights=heights, fill=fill, width=width, height=height)


def uy_terrain_add_layer(target: dict, layer: dict = None, texture: dict = None, layerPath: str = None, tileSize: list = None) -> dict:
    "Add a paint layer, an existing TerrainLayer or a new one from a texture. Read back."
    return _unity("terrain_add_layer", target=target, layer=layer, texture=texture, layerPath=layerPath, tileSize=tileSize)


def uy_terrain_paint_layer(target: dict, layer, weight: float, x: int = None, y: int = None, width: int = None, height: int = None) -> dict:
    "Paint one layer's weight over a block, scaling the others so each texel still sums to 1. Undoable. Read back through GetAlphamaps."
    return _unity("terrain_paint_layer", target=target, layer=layer, weight=weight, x=x, y=y, width=width, height=height)


def uy_terrain_sample(target: dict, points: list, space: str = None, layers: bool = None, details: bool = None, collider: bool = None) -> dict:
    "Sample a terrain at [x,z] points (world or normalized): height, normal, steepness, hole, layer weights, detail values, and the TerrainCollider's height with collider. Read only."
    return _unity("terrain_sample", target=target, points=points, space=space, layers=layers, details=details, collider=collider)


def uy_terrain_heights_get(target: dict, x: int = None, y: int = None, width: int = None, height: int = None, stride: int = None, grid: dict = None, world: bool = None) -> dict:
    "Raw heights over a block of the heightmap (every stride-th), or interpolated on a normalized grid {u0, v0, count, step}; 0..1 or world y. At most 65536 values. Read only."
    return _unity("terrain_heights_get", target=target, x=x, y=y, width=width, height=height, stride=stride, grid=grid, world=world)


def uy_terrain_alphamap_get(target: dict, x: int = None, y: int = None, width: int = None, height: int = None, layer=None, stride: int = None) -> dict:
    "Paint-layer weights over a block of the alphamap, one layer or all, with the worst texel sum error. At most 65536 values. Read only."
    return _unity("terrain_alphamap_get", target=target, x=x, y=y, width=width, height=height, layer=layer, stride=stride)


def uy_terrain_trees_place(target: dict, prototype, trees: list = None, scatter: dict = None, keepExisting: bool = None, snapToHeightmap: bool = None) -> dict:
    "Place trees of one prototype (an index, or a prefab added when missing), listed or scattered with a seed, in normalized terrain space. Read back."
    return _unity("terrain_trees_place", target=target, prototype=prototype, trees=trees, scatter=scatter, keepExisting=keepExisting, snapToHeightmap=snapToHeightmap)


def uy_terrain_trees_get(target: dict, prototype: int = None, rect: list = None, offset: int = None, limit: int = None, countsOnly: bool = None) -> dict:
    "A terrain's tree prototypes with counts, and its instances filtered by prototype and normalized rect, paged. Read only."
    return _unity("terrain_trees_get", target=target, prototype=prototype, rect=rect, offset=offset, limit=limit, countsOnly=countsOnly)


def uy_terrain_trees_remove(target: dict, prototype: int = None, rect: list = None, circle: dict = None, indices: list = None, all: bool = None, removePrototype: bool = None) -> dict:
    "Remove trees by prototype, area or index (all selectors must match), or all; removePrototype drops it and renumbers the rest. Read back."
    return _unity("terrain_trees_remove", target=target, prototype=prototype, rect=rect, circle=circle, indices=indices, all=all, removePrototype=removePrototype)


def uy_terrain_detail_prototypes_set(target: dict, action: str, index: int = None, mesh: dict = None, texture: dict = None, values: dict = None) -> dict:
    "Add, change or remove a detail prototype (grass texture or detail mesh prefab) and its values by API name; one Unity's Validate rejects is refused with its message. Read back."
    return _unity("terrain_detail_prototypes_set", target=target, action=action, index=index, mesh=mesh, texture=texture, values=values)


def uy_terrain_detail_paint(target: dict, layer: int, x: int = None, y: int = None, fill: float = None, width: int = None, height: int = None, values: list = None, normalized: bool = None) -> dict:
    "Set one detail layer's density over a block, by fill or rows, bounded by maxDetailScatterPerRes (or 0..1 normalized). Undoable. Read back cell by cell."
    return _unity("terrain_detail_paint", target=target, layer=layer, x=x, y=y, fill=fill, width=width, height=height, values=values, normalized=normalized)


def uy_terrain_detail_info(target: dict, patch: list = None, sample: int = None) -> dict:
    "Detail resolution, patches, scatter mode, waving grass, each prototype's coverage; with patch [px, pz], the instances each layer draws there. Read only."
    return _unity("terrain_detail_info", target=target, patch=patch, sample=sample)


def uy_terrain_heights_import(target: dict, texture: dict = None, raw: str = None, file: str = None, depth: int = None, byteOrder: str = None, resolution: int = None,
                              channel: str = None, flipVertically: bool = None, scale: float = None, offset: float = None) -> dict:
    "Set the whole heightmap from a readable texture, a PNG/JPG or an 8/16-bit RAW inside the project, resampled to the terrain. Read back through GetHeights."
    return _unity("terrain_heights_import", target=target, texture=texture, raw=raw, file=file, depth=depth, byteOrder=byteOrder, resolution=resolution,
                  channel=channel, flipVertically=flipVertically, scale=scale, offset=offset)


def uy_terrain_heights_export(target: dict, path: str, depth: int = None, byteOrder: str = None, flipVertically: bool = None, overwrite: bool = None) -> dict:
    "Write the heightmap to a 16-bit (default) or 8-bit .raw/.r16 under Assets/, as the editor's Export Raw does. Read back from disk."
    return _unity("terrain_heights_export", target=target, path=path, depth=depth, byteOrder=byteOrder, flipVertically=flipVertically, overwrite=overwrite)


def uy_terrain_holes_set(target: dict, x: int = None, y: int = None, fill: bool = None, width: int = None, height: int = None, rows: list = None, circle: dict = None) -> dict:
    "Cut or fill holes (true solid, false a hole) over a block, by fill, rows, or a normalized circle {u, v, radius, solid?}. Bounds checked first. Read back through GetHoles."
    return _unity("terrain_holes_set", target=target, x=x, y=y, fill=fill, width=width, height=height, rows=rows, circle=circle)


def uy_terrain_settings_set(target: dict, values: dict) -> dict:
    "Set the Terrain's render, LOD, tree and detail distance values and material (undoable), and waving grass on its data, by API name. Read back through each getter."
    return _unity("terrain_settings_set", target=target, values=values)


def uy_terrain_layer_set(target: dict, values: dict, layer=None) -> dict:
    "Set a TerrainLayer's textures, tiling and surface values by API name: the layer asset, or a terrain and its layer. A texture not imported as a normal map is refused as one. Read back."
    return _unity("terrain_layer_set", target=target, values=values, layer=layer)


def uy_terrain_neighbors_set(mode: str, targets: list, groupingID: int = None, links: list = None) -> dict:
    "Connect terrains: auto sets groupingID and allowAutoConnect on each and auto-connects them; explicit sets links [{target, left?, top?, right?, bottom?}] (not saved). Read back, checked against the placement map."
    return _unity("terrain_neighbors_set", mode=mode, targets=targets, groupingID=groupingID, links=links)


def uy_terrain_resolution_set(target: dict, heightmapResolution: int = None, alphamapResolution: int = None, baseMapResolution: int = None, detailResolution: int = None,
                              detailPerPatch: int = None, detailScatterMode: str = None, size: list = None, keepContent: bool = None) -> dict:
    "Change heightmap, alphamap, base map or detail resolution, scatter mode or world size; with keepContent (default) heights, weights and holes are resampled. Read back."
    return _unity("terrain_resolution_set", target=target, heightmapResolution=heightmapResolution, alphamapResolution=alphamapResolution, baseMapResolution=baseMapResolution,
                  detailResolution=detailResolution, detailPerPatch=detailPerPatch, detailScatterMode=detailScatterMode, size=size, keepContent=keepContent)


def uy_terrain_layer_remove(target: dict, layer, giveTo=None) -> dict:
    "Remove a paint layer and give its weight to one layer, or spread it over the rest in proportion; every texel still sums to 1. Read back."
    return _unity("terrain_layer_remove", target=target, layer=layer, giveTo=giveTo)


def uy_terrain_height_range_fit(target: dict, min: float = None, max: float = None, padBottom: float = None, padTop: float = None) -> dict:
    "Fit the terrain's height range (size.y) to its content, or to a world min and max, moving the object so no point moves in the world. Read back at nine points."
    return _unity("terrain_height_range_fit", target=target, min=min, max=max, padBottom=padBottom, padTop=padTop)


def uy_particle_create(name: str, parent: dict = None, position: list = None, rotation: list = None, material: dict = None) -> dict:
    "Create a Particle System with this pipeline's particle material or a named one; another pipeline's is refused. Undoable. Read back."
    return _unity("particle_create", name=name, parent=parent, position=position, rotation=rotation, material=material)


def uy_particle_module_get(target: dict, module: str = None, limit: int = None, links: bool = None) -> dict:
    "A Particle System's modules with their enabled flags, or every serialized value inside one, by the path uy_particle_module_set takes. Read only."
    return _unity("particle_module_get", target=target, module=module, limit=limit, links=links)


def uy_particle_module_set(target: dict, values: dict) -> dict:
    "Write module values by serialized path, as one undo. Every path and type is checked first. Read back. uy_particle_simulate then shows what they do."
    return _unity("particle_module_set", target=target, values=values)


def uy_particle_simulate(target: dict, seconds: float = None, restart: bool = None, withChildren: bool = None, sample: int = None, allowEmpty: bool = None, times: list = None, sampleEach: int = None, keep: bool = None) -> dict:
    "Scrub a Particle System to a time and read its particles: count, bounds, a sample. With times, a series of count, size, color and age per time. No particles is a failure unless allowEmpty. Changes nothing saved and leaves the system as it found it (leftAsFound), unless keep leaves it scrubbed for a capture. A series is capped at 1800 simulated seconds in all."
    return _unity("particle_simulate", target=target, seconds=seconds, restart=restart, withChildren=withChildren, sample=sample, allowEmpty=allowEmpty, times=times, sampleEach=sampleEach, keep=keep)


def uy_particle_bursts_set(target: dict, bursts: list) -> dict:
    "Replace a system's bursts: time, count (number or minMaxCurve), cycles, interval, probability; an empty list clears. Every burst checked first. Undoable. Read back."
    return _unity("particle_bursts_set", target=target, bursts=bursts)


def uy_particle_subemitter_set(target: dict, add: list = None, set: list = None, remove: list = None, enable: bool = None) -> dict:
    "Add (an existing system or a created child), change or remove sub-emitters. A loop is refused; a non-child is written with a warning. Undoable. Read back."
    return _unity("particle_subemitter_set", target=target, add=add, set=set, remove=remove, enable=enable)


def uy_particle_attach(target: dict, kind: str, add: list = None, remove: list = None, enableModule: bool = None) -> dict:
    "Attach or detach collision planes, trigger colliders or force-field influences on a system, turning the module on. Undoable. Read back by identity."
    return _unity("particle_attach", target=target, kind=kind, add=add, remove=remove, enableModule=enableModule)


def uy_particle_renderer_set(target: dict, renderMode: str = None, material: dict = None, trailMaterial: dict = None, meshes: list = None, sortMode: str = None, alignment: str = None,
                             lengthScale: float = None, velocityScale: float = None, minParticleSize: float = None, maxParticleSize: float = None, gpuInstancing: bool = None,
                             vertexStreams: list = None) -> dict:
    "A particle renderer's mode, materials (another pipeline's refused), meshes, sorting, scales, size limits, instancing and vertex streams. Mesh mode with no mesh refused. Undoable. Read back."
    return _unity("particle_renderer_set", target=target, renderMode=renderMode, material=material, trailMaterial=trailMaterial, meshes=meshes, sortMode=sortMode, alignment=alignment,
                  lengthScale=lengthScale, velocityScale=velocityScale, minParticleSize=minParticleSize, maxParticleSize=maxParticleSize, gpuInstancing=gpuInstancing, vertexStreams=vertexStreams)


def uy_particle_forcefield_create(name: str = None, parent: dict = None, position: list = None, shape: str = None, startRange: float = None, endRange: float = None, length: float = None,
                                  gravityFocus: float = None, gravity=None, drag=None, rotationSpeed=None, rotationAttraction=None, directionX=None, directionY=None, directionZ=None,
                                  vectorField: dict = None, attachTo: list = None) -> dict:
    "Create a ParticleSystemForceField with shape, ranges and forces (numbers or minMaxCurves), optionally attached to systems as an influence. Undoable. Read back."
    return _unity("particle_forcefield_create", name=name, parent=parent, position=position, shape=shape, startRange=startRange, endRange=endRange, length=length, gravityFocus=gravityFocus,
                  gravity=gravity, drag=drag, rotationSpeed=rotationSpeed, rotationAttraction=rotationAttraction, directionX=directionX, directionY=directionY, directionZ=directionZ,
                  vectorField=vectorField, attachTo=attachTo)


def uy_particle_bake_mesh(target: dict, path: str, seconds: float = None, trails: bool = None, camera: dict = None, useTransform: bool = None) -> dict:
    "Bake a system's particles (or trails) into a Mesh asset, optionally simulating to a time first. No particles refused; camera-facing modes need a camera. Read back from the asset."
    return _unity("particle_bake_mesh", target=target, path=path, seconds=seconds, trails=trails, camera=camera, useTransform=useTransform)


def uy_particle_events_get(target: dict, kind: str = None, hit: dict = None, triggerType: str = None, collect: bool = None) -> dict:
    "In play mode: a system's collision events with one object, or its trigger particles. The last simulation step's; collect gathers every frame's until play ends. Changes nothing saved."
    return _unity("particle_events_get", target=target, kind=kind, hit=hit, triggerType=triggerType, collect=collect)


def uy_particle_record(target: dict, seconds: float, everyFrames: int = None) -> dict:
    "Record particle counts in play mode, every system under the target, to a file kept across leaving play mode. Request: poll uy_particle_record_get. Refused outside play mode."
    return _unity("particle_record", target=target, seconds=seconds, everyFrames=everyFrames)


def uy_particle_record_get(recordId: str, limit: int = None) -> dict:
    "What a particle_record took: counts per system over time, the peak, the last sample, done or cut short. Read only."
    return _unity("particle_record_get", recordId=recordId, limit=limit)


def uy_vfx_create(path: str, template: str = None, source: dict = None) -> dict:
    "A .vfx copied from a VFX Graph package template (listed live, names differ by version) or a project .vfx. Built-in and no package refused by name. Read back: exposed properties and events."
    return _unity("vfx_create", path=path, template=template, source=source)


def uy_vfx_place(asset: dict, name: str = None, parent: dict = None, position: list = None, initialEvent: str = None, seed: int = None) -> dict:
    "A GameObject with a VisualEffect playing a .vfx, with its initial event and seed. Undoable. Read back, with the exposed properties vfx_set takes."
    return _unity("vfx_place", asset=asset, name=name, parent=parent, position=position, initialEvent=initialEvent, seed=seed)


def uy_vfx_set(target: dict, values: dict = None, reset: list = None, settings: dict = None) -> dict:
    "A VisualEffect's exposed-property overrides (type-checked against the asset), resets and settings, written to its property sheet as the inspector does. One undo. Read back from the sheet and the component."
    return _unity("vfx_set", target=target, values=values, reset=reset, settings=settings)


def uy_vfx_control(target: dict, action: str, event: str = None, attributes: dict = None, step: float = None, steps: int = None) -> dict:
    "Drive a VisualEffect: send an event with a typed payload, play, stop, reinit or step the simulation; answers the spawn systems' state. Needs a graphics device."
    return _unity("vfx_control", target=target, action=action, event=event, attributes=attributes, step=step, steps=steps)


def uy_vfx_watch(target: dict, seconds: float, sendFirst: str = None, allowEmpty: bool = None) -> dict:
    "Watch a VisualEffect's alive particles per system for at least 1 s (the count updates about once a second). Request: poll uy_vfx_watch_get. Needs a graphics device."
    return _unity("vfx_watch", target=target, seconds=seconds, sendFirst=sendFirst, allowEmpty=allowEmpty)


def uy_vfx_watch_get(watchId: str, limit: int = None) -> dict:
    "What a vfx_watch took: alive particles per system over time, peak and final. A watch that stayed at zero fails unless it asked allowEmpty. Read only."
    return _unity("vfx_watch_get", watchId=watchId, limit=limit)


def uy_vfx_template_set(target: dict, name: str = None, category: str = None, description: str = None) -> dict:
    "Mark a project .vfx as a VFX Graph template with a name, category and description (VFX Graph 17, Unity 6; 2022.3 refuses needs-version). Read back."
    return _unity("vfx_template_set", target=target, name=name, category=category, description=description)


def uy_vfx_read(target: dict, values: bool = None) -> dict:
    "What VFX Graph makes public: exposed properties and events of a VisualEffect or a .vfx asset, and a live one's alive particles. Built-in refuses by name. Read only."
    return _unity("vfx_read", target=target, values=values)


def uy_anim_controller_build(path: str, states: list = None, parameters: list = None, transitions: list = None, layers: list = None) -> dict:
    "Create an Animator Controller from one description: parameters, and one layer (states, transitions) or several (layers: weight, Override or Additive, AvatarMask, synced with override motions, IK pass), sub-state machines, blend trees of all five types nested, and state, Any, Entry, Exit and sub-machine transitions by state path (Sub/State, Sub/ for a machine). Everything named is checked before the asset exists. Reloaded and compared field by field, with no unreferenced sub-asset left."
    return _unity("anim_controller_build", path=path, states=states, parameters=parameters, transitions=transitions, layers=layers)


def uy_anim_controller_get(target: dict, layer=None, positions: bool = None) -> dict:
    "An Animator Controller as data: every layer's settings, states by path with motions and blend trees, sub-state machines, every transition kind, a synced layer's overrides, and any sub-asset the graph does not reach (orphans). layer picks one by index or name. Read only."
    return _unity("anim_controller_get", target=target, layer=layer, positions=positions)


def uy_anim_layer_set(target: dict, add: list = None, set: list = None, remove: list = None) -> dict:
    "Add, change and remove layers of an existing Animator Controller: name, weight, Override or Additive, AvatarMask (null clears), syncedTo (null unsyncs), IK pass. A taken name, removing a layer another follows, and syncing a layer with states are refused first. Undoable. Read back, nothing left orphaned in the file."
    return _unity("anim_layer_set", target=target, add=add, set=set, remove=remove)


def uy_anim_state_set(target: dict, layer=None, addMachines: list = None, add: list = None, set: list = None, remove: list = None, defaultState: str = None, createMachines: bool = None, cascade: bool = None) -> dict:
    "Add, change, rename and remove states and sub-state machines of one controller layer by path (Combat/Attack; a machine to remove ends in /). Removing what a transition leads to is refused unless cascade. Undoable. Read back."
    return _unity("anim_state_set", target=target, layer=layer, addMachines=addMachines, add=add, set=set, remove=remove, defaultState=defaultState, createMachines=createMachines, cascade=cascade)


def uy_anim_transition_set(target: dict, layer=None, add: list = None, set: list = None, remove: list = None) -> dict:
    "Add, change and remove transitions of every kind in one controller layer (state, Any, Entry, Sub/Entry, Sub/, to a state, Sub/ or Exit), with conditions and timing; several between the same ends are picked by index. Undoable. Read back."
    return _unity("anim_transition_set", target=target, layer=layer, add=add, set=set, remove=remove)


def uy_anim_blendtree_set(target: dict, state: str, tree: dict, layer=None) -> dict:
    "Put a blend tree of any type, nested allowed, on an existing state by path, or as a synced layer's override of it. The tree replaced is destroyed with its nested trees. Undoable. Read back."
    return _unity("anim_blendtree_set", target=target, layer=layer, state=state, tree=tree)


def uy_animator_state(target: dict, layers: list = None, parameters: bool = None) -> dict:
    "An Animator in play mode: each layer's current and next state by path (never a guessed name), normalized time, transition, clip weights and weight, and parameter values. Outside play mode refuses not-playing with what edit time can read. Read only."
    return _unity("animator_state", target=target, layers=layers, parameters=parameters)


def uy_animator_drive(target: dict, set: dict = None, trigger: list = None, resetTrigger: list = None, play: dict = None, crossFade: dict = None, layerWeight: dict = None, speed: float = None) -> dict:
    "In play mode, drive an Animator: set parameters, fire or reset triggers, play or cross-fade to a state by path, set a layer weight or the speed. Play-mode state only. A curve-driven parameter is refused. Read back; poll uy_animator_state for the state it moves to."
    return _unity("animator_drive", target=target, set=set, trigger=trigger, resetTrigger=resetTrigger, play=play, crossFade=crossFade, layerWeight=layerWeight, speed=speed)


def uy_model_rig_get(path: str, mapping: bool = None, paths: bool = None) -> dict:
    "A model's rig import: animation type, avatar setup, the avatar with isValid and isHuman, the human bone mapping and the required bones it misses, takes and clips. Read only."
    return _unity("model_rig_get", path=path, mapping=mapping, paths=paths)


def uy_model_rig_set(path: str, animationType: str = None, avatarSetup: str = None, sourceAvatar: dict = None, mapping: list = None, optimizeGameObjects: bool = None, rootMotionNode: str = None,
                     muscles: dict = None, exposedTransforms: list = None) -> dict:
    "Set a model's rig: None, Legacy, Generic or Human; avatar from this model or copied; a human bone mapping or Unity's auto-mapping; Human muscles (twists, stretches, feetSpacing 0..1, translationDoF); exposedTransforms kept when optimizeGameObjects strips the rest; rootMotionNode. A mapping missing a required bone is refused before the import. Reimported; the avatar read back."
    return _unity("model_rig_set", path=path, animationType=animationType, avatarSetup=avatarSetup, sourceAvatar=sourceAvatar, mapping=mapping, optimizeGameObjects=optimizeGameObjects, rootMotionNode=rootMotionNode,
                  muscles=muscles, exposedTransforms=exposedTransforms)


def uy_human_bones(muscles: bool = None) -> dict:
    "Unity's humanoid tables: each human bone's name, whether it is required, and its parent; with muscles, the muscle names and ranges. The names model_rig_set's mapping takes. Read only."
    return _unity("human_bones", muscles=muscles)


def uy_model_clips_set(path: str, clips: list) -> dict:
    "Split a model's takes into named clips: frames, loop, root locks, mirror, offsets, events (time 0..1 of the clip), mask (an AvatarMask, \"fromModel\" or null), additiveReferenceFrame; [] goes back to Unity's defaults. Reimported, each clip's length checked against its frames."
    return _unity("model_clips_set", path=path, clips=clips)


def uy_avatar_mask_create(path: str, bodyParts: dict = None, root: dict = None, transforms: dict = None) -> dict:
    "Create an AvatarMask: humanoid body parts on or off, and a hierarchy's transforms (a scene object or a model) each on or off. Read back."
    return _unity("avatar_mask_create", path=path, bodyParts=bodyParts, root=root, transforms=transforms)


def uy_avatar_mask_set(target: dict, bodyParts: dict = None, transforms: dict = None, removePaths: list = None) -> dict:
    "Change an AvatarMask's body parts and transform flags, or remove paths. Undoable. Read back."
    return _unity("avatar_mask_set", target=target, bodyParts=bodyParts, transforms=transforms, removePaths=removePaths)


def uy_override_controller_create(path: str, controller: dict, overrides: dict = None) -> dict:
    "Create an Animator Override Controller on a base, clips replaced by the original's name. Read back through GetOverrides."
    return _unity("override_controller_create", path=path, controller=controller, overrides=overrides)


def uy_override_controller_set(target: dict, overrides: dict = None, controller: dict = None) -> dict:
    "Change an override controller's overrides (null clears one) or its base: overrides the new base cannot take are dropped and named. Undoable. Read back."
    return _unity("override_controller_set", target=target, overrides=overrides, controller=controller)


def uy_anim_behaviour_set(target: dict, on: str, layer=None, add: list = None, set: list = None, remove: list = None) -> dict:
    "Add, change and remove StateMachineBehaviours on a state (path) or a state machine (Sub/, or / for the root), with field values as set_property writes them. Refused on compile errors. Undoable. Read back."
    return _unity("anim_behaviour_set", target=target, layer=layer, on=on, add=add, set=set, remove=remove)


def uy_constraint_set(target: dict, sources: list = None, weight: float = None, active: bool = None, locked: bool = None, offset: list = None, axes: dict = None, atRest: list = None,
                      offsets: list = None, translationAxes: dict = None, rotationAxes: dict = None, translationAtRest: list = None, rotationAtRest: list = None, aimVector: list = None,
                      upVector: list = None, worldUp: str = None, worldUpObject: dict = None, worldUpVector: list = None, roll: float = None, useUpObject: bool = None) -> dict:
    "Set an Aim, Parent, Position, Rotation, Scale or LookAt constraint: sources and weights, weight, active, locked, offsets, axes, rest values, up vectors. A key the type lacks is refused by name. Undoable. Read back."
    return _unity("constraint_set", target=target, sources=sources, weight=weight, active=active, locked=locked, offset=offset, axes=axes, atRest=atRest, offsets=offsets,
                  translationAxes=translationAxes, rotationAxes=rotationAxes, translationAtRest=translationAtRest, rotationAtRest=rotationAtRest, aimVector=aimVector, upVector=upVector,
                  worldUp=worldUp, worldUpObject=worldUpObject, worldUpVector=worldUpVector, roll=roll, useUpObject=useUpObject)


def uy_constraint_get(target: dict, recursive: bool = None, limit: int = None) -> dict:
    "Every constraint on a GameObject or under it: sources, weights, offsets, axes, state and the object's pose. Read only."
    return _unity("constraint_get", target=target, recursive=recursive, limit=limit)


def uy_clip_sample(target: dict, clip: dict, times: list, bindings: list = None, transforms: list = None) -> dict:
    "Sample a clip on a GameObject at times in edit mode and read each binding's value and named transforms' poses; then prove the object back as it was. Runs alone."
    return _unity("clip_sample", target=target, clip=clip, times=times, bindings=bindings, transforms=transforms)


def uy_avatar_build(path: str, root: dict, kind: str, mapping: list = None, rootMotionNode: str = None) -> dict:
    "Build an Avatar from a scene hierarchy, Human from a mapping or Generic, saved only when Unity says it is valid. Read back."
    return _unity("avatar_build", path=path, root=root, kind=kind, mapping=mapping, rootMotionNode=rootMotionNode)


def uy_clip_record(action: str, target: dict = None, components: list = None, recursive: bool = None, path: str = None, fps: float = None, maxSeconds: float = None) -> dict:
    "Record a GameObject over play-mode frames into a clip: start (maxSeconds caps it, default 600), status (poll), stop with a path; a failed save keeps the take. Read back."
    return _unity("clip_record", action=action, target=target, components=components, recursive=recursive, path=path, fps=fps, maxSeconds=maxSeconds)


def uy_spline_create(name: str, parent: dict = None, position: list = None, rotation: list = None, knots: list = None, points: list = None, mode: str = None, shape: dict = None,
                     fit: dict = None, closed: bool = None) -> dict:
    "Create a GameObject with a SplineContainer and one spline: from knots, from points in one tangent mode, from a SplineFactory shape, or fitted to points. One source. Needs com.unity.splines. Undoable. Read back."
    return _unity("spline_create", name=name, parent=parent, position=position, rotation=rotation, knots=knots, points=points, mode=mode, shape=shape, fit=fit, closed=closed)


def uy_spline_get(target: dict, spline: int = None, world: bool = None) -> dict:
    "A SplineContainer's splines: knots with tangents, rotation, mode and tension, lengths, bounds, data keys, and knot links. Read only."
    return _unity("spline_get", target=target, spline=spline, world=world)


def uy_spline_knots_set(target: dict, ops: list, spline: int = None) -> dict:
    "Edit a spline's knots by ops in order: insert, set, remove, mode, tension, closed, clear. All checked before the first runs. Undoable. Read back."
    return _unity("spline_knots_set", target=target, spline=spline, ops=ops)


def uy_spline_evaluate(target: dict, spline: int = None, at: list = None, count: int = None, unit: str = None, fields: list = None, nearest: list = None) -> dict:
    "Sample a spline in world space at times (Normalized, Distance or Knot units): position, tangent, up, curvature; and the nearest point to world points. Read only."
    return _unity("spline_evaluate", target=target, spline=spline, at=at, count=count, unit=unit, fields=fields, nearest=nearest)


def uy_spline_extrude(target: dict, container: dict = None, values: dict = None, meshPath: str = None, material: dict = None) -> dict:
    "Add or configure a SplineExtrude: a tube along a spline, built into a saved Mesh asset at meshPath (required the first time: the package's own mesh is not saved). Read back against the package's vertex count."
    return _unity("spline_extrude", target=target, container=container, values=values, meshPath=meshPath, material=material)


def uy_spline_instantiate(target: dict, container: dict = None, items: list = None, values: dict = None, bake: dict = None) -> dict:
    "Add or configure a SplineInstantiate (prefabs along a spline), count its instances, and optionally bake them into real prefab instances a scene saves. Undoable. Read back."
    return _unity("spline_instantiate", target=target, container=container, items=items, values=values, bake=bake)


def uy_spline_topology_edit(target: dict, op: str, knots: list = None, points: list = None, mode: str = None, closed: bool = None, spline: int = None, a: list = None,
                            b: list = None, main: list = None, other: list = None, at: list = None) -> dict:
    "Change a SplineContainer's topology: addSpline, removeSpline, link or unlink knots ([spline, knot]), join at end knots, split at a knot, reverse. Undoable. Read back."
    return _unity("spline_topology_edit", target=target, op=op, knots=knots, points=points, mode=mode, closed=closed, spline=spline, a=a, b=b, main=main, other=other, at=at)


def uy_spline_data_set(target: dict, key: str, spline: int = None, points: list = None, unit: str = None, default: float = None, remove: bool = None) -> dict:
    "Set or remove keyed float data along a spline: points [t, value] in Normalized, Distance or Knot units, and a default. Undoable. Read back."
    return _unity("spline_data_set", target=target, spline=spline, key=key, points=points, unit=unit, default=default, remove=remove)


def uy_spline_animate_set(target: dict, container: dict = None, values: dict = None, probe: list = None) -> dict:
    "Add or configure a SplineAnimate (an object moved along a spline) and probe the position it gives at normalized times, restored after. Undoable. Read back."
    return _unity("spline_animate_set", target=target, container=container, values=values, probe=probe)


def uy_pb_shape_create(shape: str, name: str = None, parent: dict = None, position: list = None, rotation: list = None, pivot: str = None, material: dict = None, size: list = None,
                       steps: int = None, buildSides: bool = None, stairWidth: float = None, innerRadius: float = None, circumference: float = None, height: float = None,
                       radius: float = None, axisDivisions: int = None, heightCuts: int = None, smoothing: int = None, width: float = None, widthCuts: int = None, axis: str = None,
                       totalWidth: float = None, totalHeight: float = None, ledgeHeight: float = None, legWidth: float = None, depth: float = None, thickness: float = None,
                       subdivAxis: int = None, subdivHeight: int = None, angle: float = None, radialCuts: int = None, insideFaces: bool = None, outsideFaces: bool = None,
                       frontFaces: bool = None, backFaces: bool = None, endCaps: bool = None, subdivisions: int = None, rows: int = None, columns: int = None,
                       outerRadius: float = None, smooth: bool = None, horizontalCircumference: float = None, verticalCircumference: float = None) -> dict:
    "Create a ProBuilder shape (cube, stair, curvedStair, prism, cylinder, plane, door, pipe, cone, arch, icosphere, torus) with its generator's parameters; another shape's parameter is refused. Needs com.unity.probuilder. Undoable. Read back."
    return _unity("pb_shape_create", shape=shape, name=name, parent=parent, position=position, rotation=rotation, pivot=pivot, material=material, size=size, steps=steps,
                  buildSides=buildSides, stairWidth=stairWidth, innerRadius=innerRadius, circumference=circumference, height=height, radius=radius, axisDivisions=axisDivisions,
                  heightCuts=heightCuts, smoothing=smoothing, width=width, widthCuts=widthCuts, axis=axis, totalWidth=totalWidth, totalHeight=totalHeight, ledgeHeight=ledgeHeight,
                  legWidth=legWidth, depth=depth, thickness=thickness, subdivAxis=subdivAxis, subdivHeight=subdivHeight, angle=angle, radialCuts=radialCuts, insideFaces=insideFaces,
                  outsideFaces=outsideFaces, frontFaces=frontFaces, backFaces=backFaces, endCaps=endCaps, subdivisions=subdivisions, rows=rows, columns=columns,
                  outerRadius=outerRadius, smooth=smooth, horizontalCircumference=horizontalCircumference, verticalCircumference=verticalCircumference)


def uy_pb_polyshape_create(name: str, points: list, extrude: float = None, flipNormals: bool = None, holes: list = None, parent: dict = None, position: list = None,
                           rotation: list = None, material: dict = None) -> dict:
    "Create a ProBuilder mesh from a polygon outline extruded upward, with optional holes. Undoable. Read back."
    return _unity("pb_polyshape_create", name=name, points=points, extrude=extrude, flipNormals=flipNormals, holes=holes, parent=parent, position=position, rotation=rotation,
                  material=material)


def uy_pb_mesh_info(target: dict, faces: bool = None, positions: bool = None, world: bool = None, limit: int = None) -> dict:
    "A ProBuilder mesh as data: counts, sync state, submeshes, and each face's vertices, edges, center, normal, smoothing group and UV settings. Indexes here are what the editing tools take. Read only."
    return _unity("pb_mesh_info", target=target, faces=faces, positions=positions, world=world, limit=limit)


def uy_pb_extrude(target: dict, distance: float, faces: list = None, edges: list = None, method: str = None, asGroup: bool = None, manifold: bool = None) -> dict:
    "Extrude ProBuilder faces (by index) or open edges ([a,b]) by a distance. Undoable. Read back."
    return _unity("pb_extrude", target=target, faces=faces, edges=edges, distance=distance, method=method, asGroup=asGroup, manifold=manifold)


def uy_pb_bevel(target: dict, amount: float, edges: list = None, facesPerimeter: list = None, all: bool = None) -> dict:
    "Bevel ProBuilder edges by an amount: given edges, faces' perimeters, or all. Undoable. Read back."
    return _unity("pb_bevel", target=target, edges=edges, facesPerimeter=facesPerimeter, all=all, amount=amount)


def uy_pb_subdivide(target: dict, faces: list = None, iterations: int = None, edges: list = None, edgeCuts: int = None) -> dict:
    "Subdivide ProBuilder faces (1..4 passes) or connect edges. Undoable. Read back."
    return _unity("pb_subdivide", target=target, faces=faces, iterations=iterations, edges=edges, edgeCuts=edgeCuts)


def uy_pb_faces_edit(target: dict, op: str, faces: list = None, deleteSource: bool = None, a: list = None, b: list = None) -> dict:
    "Edit ProBuilder faces: delete, merge, flip, conformNormals, triangulate, quadrangulate, detach, duplicateFlip, bridge two open edges, clean. Undoable. Read back."
    return _unity("pb_faces_edit", target=target, op=op, faces=faces, deleteSource=deleteSource, a=a, b=b)


def uy_pb_vertices_edit(target: dict, op: str, vertices: list = None, faces: list = None, edges: list = None, offset: list = None, space: str = None, radius: float = None,
                        collapseToFirst: bool = None, world: list = None) -> dict:
    "Edit ProBuilder vertices: translate, weld, merge, centerPivot, setPivot, freezeScale. Undoable. Read back."
    return _unity("pb_vertices_edit", target=target, op=op, vertices=vertices, faces=faces, edges=edges, offset=offset, space=space, radius=radius,
                  collapseToFirst=collapseToFirst, world=world)


def uy_pb_smoothing_set(target: dict, faces: list = None, angle: float = None, group: int = None) -> dict:
    "Set ProBuilder faces' smoothing groups by angle or to one group (0 hard). Undoable. Read back."
    return _unity("pb_smoothing_set", target=target, faces=faces, angle=angle, group=group)


def uy_pb_merge_objects(targets: list, into: dict = None, deleteSources: bool = None) -> dict:
    "Combine several ProBuilder meshes into one; the others are deleted unless asked not to. Undoable. Read back."
    return _unity("pb_merge_objects", targets=targets, into=into, deleteSources=deleteSources)


def uy_pb_uv_set(target: dict, faces: list = None, uv: dict = None, textureGroup: int = None, manualUV: bool = None, lightmapUVs: dict = None) -> dict:
    "Set ProBuilder faces' auto UV settings, texture group or manual UV, and build lightmap UVs. Read back with each face's UV extent."
    return _unity("pb_uv_set", target=target, faces=faces, uv=uv, textureGroup=textureGroup, manualUV=manualUV, lightmapUVs=lightmapUVs)


def uy_pb_material_set(target: dict, faces: list = None, material: dict = None, color: list = None) -> dict:
    "Give ProBuilder faces a material (its own submesh) or a vertex color. Undoable. Read back."
    return _unity("pb_material_set", target=target, faces=faces, material=material, color=color)


def uy_pb_mesh_export(target: dict, meshPath: str, optimize: bool = None, lightmapUVs: bool = None, strip: bool = None, overwrite: bool = None) -> dict:
    "Save a ProBuilder mesh's compiled Mesh as a .asset under scratch; strip swaps the ProBuilderMesh for a plain MeshFilter on the asset. Writes to disk. Read back."
    return _unity("pb_mesh_export", target=target, meshPath=meshPath, optimize=optimize, lightmapUVs=lightmapUVs, strip=strip, overwrite=overwrite)


def uy_pb_mesh_import(target: dict, quads: bool = None, smoothing: bool = None, smoothingAngle: float = None) -> dict:
    "Convert a GameObject's regular mesh into an editable ProBuilder mesh, as quads or triangles. The source mesh is left as it was. Undoable. Read back."
    return _unity("pb_mesh_import", target=target, quads=quads, smoothing=smoothing, smoothingAngle=smoothingAngle)


def uy_anim_params_set(target: dict, add: list = None, remove: list = None) -> dict:
    "Add or remove Animator parameters. A duplicate and removing one a transition still reads are refused first. Undoable. Read back."
    return _unity("anim_params_set", target=target, add=add, remove=remove)


def uy_clip_create(path: str, frameRate: float = None, loop: bool = None) -> dict:
    "Create an empty animation clip. Read back."
    return _unity("clip_create", path=path, frameRate=frameRate, loop=loop)


def uy_clip_curves_set(target: dict, curves: list, root: dict = None, ensureQuaternionContinuity: bool = None) -> dict:
    "Set curves on a clip: float keys as [t, v] or {t, v, in, out, inWeight, outWeight, weighted, left, right, broken}, or object-reference curves {objects: [{t, value}]} for sprite or material swaps. Pass root: a curve on a property that does not exist is stored without complaint and animates nothing, and root lets it be checked. ensureQuaternionContinuity flips each rotation key that points away from the one before and says how many (flippedKeys). Undoable. Read back."
    return _unity("clip_curves_set", target=target, curves=curves, root=root, ensureQuaternionContinuity=ensureQuaternionContinuity)


def uy_clip_events_set(target: dict, events: list, receiver: dict = None, allowPastEnd: bool = None) -> dict:
    "Replace a clip's animation events: time, function, float/int/string/object argument, requireReceiver. receiver checks each function against a GameObject's or a prefab asset's methods first. An imported clip is refused (model_clips_set). Undoable. Read back."
    return _unity("clip_events_set", target=target, events=events, receiver=receiver, allowPastEnd=allowPastEnd)


def uy_clip_settings_set(target: dict, settings: dict) -> dict:
    "A clip's own settings by the inspector's words: loopTime, loopPose, bakeRotation, bakeY, bakeXZ, keepOriginal*, heightFromFeet, rotationOffset, level, cycleOffset, mirror, startTime, stopTime, additiveReference {clip, time} or null. An imported clip is refused. Undoable. Read back."
    return _unity("clip_settings_set", target=target, settings=settings)


def uy_clip_info(target: dict, keys: bool = None, binding: str = None, maxKeys: int = None) -> dict:
    "A clip: length, frame rate, loop, every curve binding with its keys, its events with their arguments, and its settings. Read only."
    return _unity("clip_info", target=target, keys=keys, binding=binding, maxKeys=maxKeys)


def uy_scene_stats(scene: str = None, top: int = None, allowEmpty: bool = None) -> dict:
    "Counts from walking the loaded scenes: objects, renderers, triangles, materials, lights, cameras, missing scripts, components by type. An empty scene fails unless allowEmpty. Read only."
    return _unity("scene_stats", scene=scene, top=top, allowEmpty=allowEmpty)


def uy_project_report(includePackages: bool = None) -> dict:
    "One project at a glance: Unity version, render pipeline, color space, build target, packages, asset counts, build scenes, compile state. Read only."
    return _unity("project_report", includePackages=includePackages)


def uy_ui_canvas_create(name: str, renderMode: str = None, referenceResolution: list = None, eventSystem: bool = None, parent: dict = None, inputModule: str = None) -> dict:
    "Create a uGUI Canvas with a scaler and a raycaster, and an EventSystem when asked and the scene has none, with the input module the project's input handling needs (inputModule auto, standalone or inputSystem). Refused without the uGUI package. Undoable. Read back."
    return _unity("ui_canvas_create", name=name, renderMode=renderMode, referenceResolution=referenceResolution, eventSystem=eventSystem, parent=parent, inputModule=inputModule)


def uy_ui_element_create(kind: str, name: str, parent: dict, anchorMin: list = None, anchorMax: list = None, pivot: list = None, anchoredPosition: list = None, sizeDelta: list = None, text: str = None, color: list = None,
                         sprites: bool = None) -> dict:
    "Create a uGUI element under a Canvas: Panel, Image, Text, Button, Toggle, Slider, Scrollbar, ScrollView, RawImage, TMPText, TMPButton, TMPInputField, TMPDropdown, through Unity's own control factories. A TMP kind without TMP's essentials is refused first. Undoable. Read back."
    return _unity("ui_element_create", kind=kind, name=name, parent=parent, anchorMin=anchorMin, anchorMax=anchorMax, pivot=pivot, anchoredPosition=anchoredPosition, sizeDelta=sizeDelta, text=text, color=color,
                  sprites=sprites)


def uy_ui_layout_report(target: dict, limit: int = None, layoutSizes: bool = None) -> dict:
    "Lay a Canvas out (layout groups first) and read every element's rectangle: no size, outside its parent, overlaps, what drives it, TMP text that overflows or is cut, and with layoutSizes its min, preferred and flexible sizes. Read only."
    return _unity("ui_layout_report", target=target, limit=limit, layoutSizes=layoutSizes)


def uy_uitk_validate(path: str) -> dict:
    "Check a UXML, USS or TSS file you wrote: reimport it, read the importer's own error flags and messages, clone a UXML tree, count it and check the style sheets and templates it uses."
    return _unity("uitk_validate", path=path)


def _laid_out(call, seconds=3.0):
    """UI Toolkit lays a panel out on the editor's next tick, so a read just after a change can answer not-laid-out: ask again for a few seconds."""
    end = time.time() + seconds
    while True:
        r = call()
        if not (isinstance(r, dict) and r.get("refusedRule") == "not-laid-out") or time.time() > end:
            return r
        time.sleep(0.2)


def uy_uitk_write(path: str, text: str, overwrite: bool = None) -> dict:
    "Write a UXML, USS or TSS file under the path rule, import it at once and read the asset back with the importer's own error flags; a UXML is cloned and counted. A file with errors stays on disk and answers ok:false with them."
    return _unity("uitk_write", path=path, text=text, overwrite=overwrite)


def uy_uitk_panel_settings_create(path: str, theme: str = None, scaleMode: str = None, referenceResolution: list = None, screenMatchMode: str = None, match: float = None, scale: float = None,
                                  referenceDpi: float = None, fallbackDpi: float = None, sortingOrder: float = None, targetDisplay: int = None, clearColor: bool = None, colorClearValue: list = None,
                                  targetTexture: str = None) -> dict:
    "Create a PanelSettings asset: scale mode, reference resolution, match, scale, DPI, sorting, target display, clear settings, theme (.tss) and target texture. Read back from a fresh load."
    return _unity("uitk_panel_settings_create", path=path, theme=theme, scaleMode=scaleMode, referenceResolution=referenceResolution, screenMatchMode=screenMatchMode, match=match, scale=scale,
                  referenceDpi=referenceDpi, fallbackDpi=fallbackDpi, sortingOrder=sortingOrder, targetDisplay=targetDisplay, clearColor=clearColor, colorClearValue=colorClearValue, targetTexture=targetTexture)


def uy_uitk_panel_settings_set(target: dict, theme: str = None, scaleMode: str = None, referenceResolution: list = None, screenMatchMode: str = None, match: float = None, scale: float = None,
                               referenceDpi: float = None, fallbackDpi: float = None, sortingOrder: float = None, targetDisplay: int = None, clearColor: bool = None, colorClearValue: list = None,
                               targetTexture: str = None, referenceSpritePixelsPerUnit: float = None, bindingLogLevel: str = None, save: bool = None) -> dict:
    "Change an existing PanelSettings asset's scaling, theme, sorting, clear and binding settings. Undoable in memory; save writes it to disk. Read back."
    return _unity("uitk_panel_settings_set", target=target, theme=theme, scaleMode=scaleMode, referenceResolution=referenceResolution, screenMatchMode=screenMatchMode, match=match, scale=scale,
                  referenceDpi=referenceDpi, fallbackDpi=fallbackDpi, sortingOrder=sortingOrder, targetDisplay=targetDisplay, clearColor=clearColor, colorClearValue=colorClearValue,
                  targetTexture=targetTexture, referenceSpritePixelsPerUnit=referenceSpritePixelsPerUnit, bindingLogLevel=bindingLogLevel, save=save)


def uy_uitk_document_create(panelSettings: str, name: str = None, target: dict = None, parent: dict = None, visualTreeAsset: str = None, sortingOrder: float = None) -> dict:
    "Add a UIDocument with a PanelSettings and a UXML tree, on a new GameObject (name) or an existing one (target), and read its live root back. Undoable."
    return _unity("uitk_document_create", panelSettings=panelSettings, name=name, target=target, parent=parent, visualTreeAsset=visualTreeAsset, sortingOrder=sortingOrder)


def uy_uitk_document_set(target: dict, visualTreeAsset: str = None, panelSettings: str = None, sortingOrder: float = None, position: str = None, worldSpaceSizeMode: str = None,
                         worldSpaceSize: list = None) -> dict:
    "Swap a UIDocument's tree or panel settings, change its sorting order, and on 6000.3+ its world-space position mode and size. The live root is read back after a swap. Undoable."
    return _unity("uitk_document_set", target=target, visualTreeAsset=visualTreeAsset, panelSettings=panelSettings, sortingOrder=sortingOrder, position=position,
                  worldSpaceSizeMode=worldSpaceSizeMode, worldSpaceSize=worldSpaceSize)


def uy_uitk_tree_read(target: dict = None, asset: str = None, name: str = None, className: str = None, type: str = None, depth: int = None, limit: int = None) -> dict:
    "Read a UI Toolkit element tree, a live UIDocument's or a UXML cloned detached: names, types, classes, text, style sheets and UXML-authored bindings (6000.0+). A query matching nothing fails. Read only."
    return _unity("uitk_tree_read", target=target, asset=asset, name=name, className=className, type=type, depth=depth, limit=limit)


def uy_uitk_layout_report(target: dict, name: str = None, limit: int = None) -> dict:
    "A live UIDocument's element rectangles in panel space, with no-size, outside-parent, overlap and elided-text flags. Asks again for a few seconds while the panel is not laid out yet. Read only."
    return _laid_out(lambda: _unity("uitk_layout_report", target=target, name=name, limit=limit))


def uy_uitk_pick(target: dict, points: list) -> dict:
    "Which element of a live UIDocument's panel a pointer at each panel position reaches, by the panel's own pick (picking-mode and what is on top count). Read only."
    return _laid_out(lambda: _unity("uitk_pick", target=target, points=points))


def uy_uitk_data_source_set(target: dict, name: str, source: dict, dataSourcePath: str = None) -> dict:
    "Give a live element a data source and source path, read back. Live only, never saved: the document's next rebuild drops it. 6000.0+. Read the bound value with uy_uitk_tree_read."
    return _unity("uitk_data_source_set", target=target, name=name, source=source, dataSourcePath=dataSourcePath)


def uy_ui_layout_set(target: dict, component: str, values: dict, replaceGroup: bool = None) -> dict:
    "Add or change a Horizontal/Vertical/Grid layout group, ContentSizeFitter, LayoutElement or AspectRatioFitter by typed fields, re-run layout and read the child rects back. Undoable."
    return _unity("ui_layout_set", target=target, component=component, values=values, replaceGroup=replaceGroup)


def uy_ui_rect_set(target: dict, preset: str = None, anchorMin: list = None, anchorMax: list = None, pivot: list = None, anchoredPosition: list = None, sizeDelta: list = None,
                   offsetMin: list = None, offsetMax: list = None, width: float = None, height: float = None, inset: dict = None) -> dict:
    "Set a RectTransform by anchors, pivot, position, size, offsets, an anchor preset or an edge inset; read back with its canvas rect. A field a layout drives is refused, naming the driver. Undoable."
    return _unity("ui_rect_set", target=target, preset=preset, anchorMin=anchorMin, anchorMax=anchorMax, pivot=pivot, anchoredPosition=anchoredPosition, sizeDelta=sizeDelta,
                  offsetMin=offsetMin, offsetMax=offsetMax, width=width, height=height, inset=inset)


def uy_ui_text_set(target: dict, text: str = None, font: str = None, fontSize: float = None, autoSize: bool = None, fontSizeMin: float = None, fontSizeMax: float = None, fontStyle: list = None,
                   alignment: str = None, wrapping: str = None, overflow: str = None, color: list = None, characterSpacing: float = None, lineSpacing: float = None, margin: list = None,
                   richText: bool = None) -> dict:
    "Set a TextMeshPro or legacy Text's string, font, size, auto-size, style, alignment, wrapping, overflow, color, spacing; read back, then answer whether it fits. No TMP read without its essentials. Undoable."
    return _unity("ui_text_set", target=target, text=text, font=font, fontSize=fontSize, autoSize=autoSize, fontSizeMin=fontSizeMin, fontSizeMax=fontSizeMax, fontStyle=fontStyle,
                  alignment=alignment, wrapping=wrapping, overflow=overflow, color=color, characterSpacing=characterSpacing, lineSpacing=lineSpacing, margin=margin, richText=richText)


def uy_tmp_status() -> dict:
    "Whether TextMeshPro can be used: package, settings asset, default font, SDF and bitmap shaders, read without the importer window ever opening. The poll for uy_tmp_essentials_import. Read only."
    return _unity("tmp_status")


def uy_tmp_essentials_import(examples: bool = None, waitSeconds: float = 60) -> dict:
    "Import TextMeshPro's essential resources without a dialog (full mode only), then poll tmp_status until they are in, up to waitSeconds (0: do not wait)."
    r = _unity("tmp_essentials_import", examples=examples)
    if not (isinstance(r, dict) and r.get("ok")) or not waitSeconds:
        return r
    end, status = time.time() + waitSeconds, {}
    while time.time() < end:
        status = _unity("tmp_status")
        if isinstance(status, dict) and status.get("ready"):
            break
        time.sleep(0.5)
    r["status"] = status
    r["ready"] = bool(isinstance(status, dict) and status.get("ready"))
    return r


def uy_ui_event_wire(source: dict, event: str, target: dict = None, method: str = None, argument: dict = None, state: str = None, remove: int = None) -> dict:
    "Add a persistent listener to a UnityEvent (onClick, onValueChanged or any UnityEvent field), resolved first by UnityEvent's own lookup, with a fixed argument or the event's value; or remove one by index. Undoable. Read back."
    return _unity("ui_event_wire", source=source, event=event, target=target, method=method, argument=argument, state=state, remove=remove)


def uy_ui_events_get(target: dict, event: str = None) -> dict:
    "Every persistent listener on a GameObject's UnityEvent fields with target, method, state and whether it still resolves (a renamed method reads false). Read only."
    return _unity("ui_events_get", target=target, event=event)


def uy_ui_canvas_set(target: dict, canvas: dict = None, scaler: dict = None, group: dict = None) -> dict:
    "Change a Canvas, its CanvasScaler and CanvasGroup by typed fields: render mode, camera, sorting, scale mode, reference resolution, match, alpha. A key in the wrong group is refused. Undoable."
    return _unity("ui_canvas_set", target=target, canvas=canvas, scaler=scaler, group=group)


def uy_ui_graphic_set(target: dict, sprite: str = None, subAsset: str = None, type: str = None, fillMethod: str = None, fillAmount: float = None, fillOrigin: int = None, fillClockwise: bool = None,
                      fillCenter: bool = None, preserveAspect: bool = None, useSpriteMesh: bool = None, pixelsPerUnitMultiplier: float = None, nativeSize: bool = None, texture: str = None,
                      color: list = None, raycastTarget: bool = None) -> dict:
    "Set an Image's sprite, type and fill, a RawImage's texture, any Graphic's color and raycast target, and optionally its native size. Undoable. Read back."
    return _unity("ui_graphic_set", target=target, sprite=sprite, subAsset=subAsset, type=type, fillMethod=fillMethod, fillAmount=fillAmount, fillOrigin=fillOrigin, fillClockwise=fillClockwise,
                  fillCenter=fillCenter, preserveAspect=preserveAspect, useSpriteMesh=useSpriteMesh, pixelsPerUnitMultiplier=pixelsPerUnitMultiplier, nativeSize=nativeSize, texture=texture,
                  color=color, raycastTarget=raycastTarget)


def uy_ui_selectable_set(target: dict, interactable: bool = None, transition: str = None, colors: dict = None, targetGraphic: dict = None, navigation: dict = None) -> dict:
    "Set a Selectable's interactable state, transition, colors, target graphic and navigation; answers the neighbours navigation resolves to. Directions need Explicit. Undoable. Read back."
    return _unity("ui_selectable_set", target=target, interactable=interactable, transition=transition, colors=colors, targetGraphic=targetGraphic, navigation=navigation)


def uy_ui_event_system_set(target: dict = None, firstSelected: dict = None, sendNavigationEvents: bool = None, pixelDragThreshold: int = None, inputModule: str = None, actions: str = None,
                           defaultActions: bool = None) -> dict:
    "Set the EventSystem's first selected object, navigation events and drag threshold; swap its input module (auto, standalone, inputSystem) or give the Input System module its UI actions. Undoable."
    return _unity("ui_event_system_set", target=target, firstSelected=firstSelected, sendNavigationEvents=sendNavigationEvents, pixelDragThreshold=pixelDragThreshold, inputModule=inputModule,
                  actions=actions, defaultActions=defaultActions)


def uy_ui_navigation_report(target: dict) -> dict:
    "Every Selectable under a Canvas with its resolved up/down/left/right, dead ends, and those no path reaches from the first selected one. Read only."
    return _unity("ui_navigation_report", target=target)


def uy_tmp_font_asset_create(font: str, path: str, samplingPointSize: int = None, padding: int = None, renderMode: str = None, atlasSize: list = None, population: str = None,
                             multiAtlas: bool = None, characters: str = None) -> dict:
    "Make a TextMeshPro font asset from a .ttf/.otf with its atlas and material inside it, characters baked; read back from a fresh load with the characters held and missing."
    return _unity("tmp_font_asset_create", font=font, path=path, samplingPointSize=samplingPointSize, padding=padding, renderMode=renderMode, atlasSize=atlasSize, population=population,
                  multiAtlas=multiAtlas, characters=characters)


def uy_ui_scroll_set(target: dict, content: dict = None, viewport: dict = None, horizontalScrollbar: dict = None, verticalScrollbar: dict = None, horizontal: bool = None, vertical: bool = None,
                     inertia: bool = None, movementType: str = None, elasticity: float = None, decelerationRate: float = None, scrollSensitivity: float = None,
                     horizontalScrollbarVisibility: str = None, verticalScrollbarVisibility: str = None, normalizedPosition: list = None, mask: str = None, showMaskGraphic: bool = None,
                     maskPadding: list = None, softness: list = None) -> dict:
    "Configure a ScrollRect: content, viewport, axes, movement, scrollbars, position and the viewport mask; lays the content out and answers whether each axis scrolls. Undoable. Read back."
    return _unity("ui_scroll_set", target=target, content=content, viewport=viewport, horizontalScrollbar=horizontalScrollbar, verticalScrollbar=verticalScrollbar, horizontal=horizontal,
                  vertical=vertical, inertia=inertia, movementType=movementType, elasticity=elasticity, decelerationRate=decelerationRate, scrollSensitivity=scrollSensitivity,
                  horizontalScrollbarVisibility=horizontalScrollbarVisibility, verticalScrollbarVisibility=verticalScrollbarVisibility, normalizedPosition=normalizedPosition, mask=mask,
                  showMaskGraphic=showMaskGraphic, maskPadding=maskPadding, softness=softness)


def uy_ui_hit_test(points: list = None, targets: list = None) -> dict:
    "In play mode, which uGUI objects a pointer at each screen point hits in raycast order; for elements, whether each one's centre reaches it. Read only."
    return _unity("ui_hit_test", points=points, targets=targets)


def uy_build_target_info() -> dict:
    "The active build target, whether this install can build it, and which standalone and mobile targets it can. Read only."
    return _unity("build_target_info")


def uy_build_scenes_get() -> dict:
    "The scenes in the build list, each with whether it is enabled and whether its file exists. Read only."
    return _unity("build_scenes_get")


def uy_build_scenes_set(scenes: list) -> dict:
    "Replace the build's scene list. A scene with no file is refused first. Read back. A project setting, so scratch mode refuses it."
    return _unity("build_scenes_set", scenes=scenes)


def uy_build_player(output: str, development: bool = None, options: list = None, profile: str = None, extraDefines: list = None, subtarget: str = None, timeoutSeconds: float = 3600) -> dict:
    "Build the player for the active target from the build list, or from a BuildProfile, and read BuildReport. options: allow-listed BuildOptions names; extraDefines and subtarget without profile. ok means Succeeded, the output exists and the report holds the options asked. Holds the editor until it ends. Scratch mode refuses it."
    return _unity("build_player", _timeout=timeoutSeconds, output=output, development=development, options=options, profile=profile, extraDefines=extraDefines, subtarget=subtarget)


def uy_build_report(detail: str = None, limit: int = None, source: str = None) -> dict:
    "The last uy_build_player's report, kept on disk; or with detail (files, steps, assets, stripping, scenes) the editor's latest BuildReport, matched to the kept one by guid. source bundles: the last uy_asset_bundles_build. Read only."
    return _unity("build_report", detail=detail, limit=limit, source=source)


def uy_player_settings_get(target: str = None, buildTarget: str = None, keys: list = None) -> dict:
    "PlayerSettings as typed values for one NamedBuildTarget (Standalone, Server, Android, ...): identity, version, backend, API level, IL2CPP, stripping, color space, batching, screen, input handling. Read only."
    return _unity("player_settings_get", target=target, buildTarget=buildTarget, keys=keys)


def uy_player_settings_set(values: dict, target: str = None, buildTarget: str = None, save: bool = None) -> dict:
    "Write PlayerSettings values for one NamedBuildTarget, each checked first, read back through its getter, saved at once and checked in ProjectSettings.asset. activeInputHandler answers restartRequired. Scratch mode refuses it."
    return _unity("player_settings_set", values=values, target=target, buildTarget=buildTarget, save=save)


def uy_scripting_defines_set(add: list = None, remove: list = None, target: str = None) -> dict:
    "Add or remove scripting define symbols for one NamedBuildTarget. Read back. For the active target Unity recompiles after the reply: follow with uy_wait_idle. Scratch mode refuses it."
    return _unity("scripting_defines_set", add=add, remove=remove, target=target)


def uy_build_settings_get() -> dict:
    "EditorUserBuildSettings: development and debugging flags, standalone subtarget, the active target's build location. Read only."
    return _unity("build_settings_get")


def uy_build_settings_set(values: dict) -> dict:
    "Set EditorUserBuildSettings values (development, allowDebugging, connectProfiler, deepProfiling, waitForManagedDebugger, standaloneSubtarget, buildLocation). Read back. Scratch mode refuses it."
    return _unity("build_settings_set", values=values)


def uy_build_target_switch(target: str, timeoutSeconds: float = 1800) -> dict:
    "Switch the active build target; holds the editor while assets reimport. Then waits for the reload and reads uy_build_target_info as afterReload. Scratch mode refuses it."
    r = _unity("build_target_switch", _timeout=timeoutSeconds, target=target)
    if not (isinstance(r, dict) and r.get("ok") and r.get("changed")):
        return r
    w = uy_wait_idle(timeoutSeconds)
    info = uy_build_target_info()
    return dict(r, settled=w.get("ok"), afterReload=info, verified=bool(r.get("verified") and info.get("target") == r.get("target")))


def uy_build_profile_list(path: str = None) -> dict:
    "The project's build profiles with platform, scenes, override, defines and the scenes each builds, and which is active (6000.0 and newer). Read only."
    return _unity("build_profile_list", path=path)


def uy_build_profile_set(path: str, scenes: list = None, overrideGlobalScenes: bool = None, defines: list = None) -> dict:
    "Set a build profile's scenes, scene override and defines, saved at once and read back. Scratch refuses the active profile."
    return _unity("build_profile_set", path=path, scenes=scenes, overrideGlobalScenes=overrideGlobalScenes, defines=defines)


def uy_build_profile_set_active(path: str = None) -> dict:
    "Make a build profile active, or with no path the classic platform. Read back. Scratch mode refuses it."
    return _unity("build_profile_set_active", path=path)


def uy_build_profile_create(platform: str, name: str, timeoutSeconds: float = 60) -> dict:
    "Create a build profile for an installed platform GUID (uy_build_target_info platforms, 6000.5), then poll uy_build_profile_list until it is listed. Scratch mode refuses it."
    r = _unity("build_profile_create", platform=platform, name=name)
    if not (isinstance(r, dict) and r.get("ok") and r.get("pending")):
        return r
    end = time.time() + timeoutSeconds
    while time.time() < end:
        lst = uy_build_profile_list()
        row = next((p for p in lst.get("profiles") or [] if p.get("path") == r.get("path")), None)
        if row:
            return dict(r, pending=False, ready=True, profile=row)
        time.sleep(0.5)
    return dict(r, ok=False, timedOut=True, error="the profile was not listed after %s s." % timeoutSeconds)


def uy_player_splash_set(values: dict) -> dict:
    "Set the splash screen: show, showUnityLogo, logoStyle, backgroundColor, background sprite, overlayOpacity, animation, drawMode, blurBackground, logos [{sprite, duration}]. Read back, saved at once. Scratch mode refuses it."
    return _unity("player_splash_set", values=values)


def uy_player_icons_set(icons: list, target: str = None, kind: str = None) -> dict:
    "Set application icons for one NamedBuildTarget (default Unknown, the default icon): Texture2D paths, one per size slot. Read back, saved at once. Scratch mode refuses it."
    return _unity("player_icons_set", icons=icons, target=target, kind=kind)


def uy_asset_bundle_names(bundle: str = None, recursive: bool = None) -> dict:
    "The project's AssetBundle names with their assets and dependencies, and unused names. No bundle is count 0. Read only."
    return _unity("asset_bundle_names", bundle=bundle, recursive=recursive)


def uy_asset_bundle_name_set(path: str, bundle: str, variant: str = None) -> dict:
    "Assign an asset or folder to an AssetBundle and variant (an empty bundle clears it), written to its .meta. Read back."
    return _unity("asset_bundle_name_set", path=path, bundle=bundle, variant=variant)


def uy_asset_bundles_build(output: str, target: str = None, bundles: list = None, options: list = None, timeoutSeconds: float = 3600) -> dict:
    "Build AssetBundles into a folder outside Assets; the manifest, each file's CRC and hash read back. Holds the editor. Scratch mode refuses it."
    return _unity("asset_bundles_build", _timeout=timeoutSeconds, output=output, target=target, bundles=bundles, options=options)


def uy_test_list(mode: str = None, refresh: bool = None, limit: int = None, timeoutSeconds: float = 60) -> dict:
    "The tests the runner finds, by full name. Refused by name without com.unity.test-framework. An empty list says so. Read only."
    end = time.time() + timeoutSeconds
    while True:
        r = _unity("test_list", mode=mode, refresh=refresh, limit=limit)
        refresh = None
        if not (isinstance(r, dict) and r.get("pending")):
            return r
        if time.time() >= end:
            return dict(r, ok=False, timedOut=True, error="the test runner gave no list in %s s." % timeoutSeconds)
        time.sleep(0.2)


def uy_test_run(mode: str = None, names: list = None, assemblies: list = None) -> dict:
    "Start a test run in the background; uy_test_results says how it went. Tests are arbitrary code in the editor, so scratch mode refuses it."
    return _unity("test_run", mode=mode, names=names, assemblies=assemblies)


def uy_test_results() -> dict:
    "The last test run's state and, once done, counts, each failure with message and stack, each test's status. Kept on disk, so it outlives a reload. Read only."
    return _unity("test_results")
def uy_find_gameobjects(name: str = None, nameContains: str = None, tag: str = None, layer: str = None, component: str = None, scene: str = None, activeOnly: bool = None, limit: int = None) -> dict:
    "Find GameObjects in the loaded scenes by name, tag, layer or component, at any depth. No match is a failure that says how many it looked at. Read only."
    return _unity("find_gameobjects", name=name, nameContains=nameContains, tag=tag, layer=layer, component=component, scene=scene, activeOnly=activeOnly, limit=limit)


def uy_duplicate_gameobject(target: dict, name: str = None, parent: dict = None) -> dict:
    "Duplicate a GameObject with its children; a prefab instance's root stays an instance with its overrides. Undoable, read back against the original."
    return _unity("duplicate_gameobject", target=target, name=name, parent=parent)


def uy_set_transform(target: dict, space: str = None, position: list = None, rotation: list = None, scale: list = None) -> dict:
    "Set position, rotation (Euler degrees) and scale, local or world. Undoable, read back: a parent that cannot hold the value fails."
    return _unity("set_transform", target=target, space=space, position=position, rotation=rotation, scale=scale)


def uy_set_object_flags(target: dict, tag: str = None, layer: str = None, isStatic: bool = None, recursive: bool = None) -> dict:
    "Set a GameObject's tag, layer and static flag, optionally on its children. An unknown tag or layer is refused. Undoable, read back."
    return _unity("set_object_flags", target=target, tag=tag, layer=layer, isStatic=isStatic, recursive=recursive)


def uy_set_sibling_index(target: dict, index: int) -> dict:
    "Move a GameObject among its siblings. An index out of range is refused. Undoable, read back."
    return _unity("set_sibling_index", target=target, index=index)


def uy_list_tags_layers(sorting: bool = None) -> dict:
    "The project's tags, named layers and (unless sorting is false) sorting layers in drawing order, each with its id. Read only."
    return _unity("list_tags_layers", sorting=sorting)


def uy_tags_layers_set(addTags: list = None, removeTags: list = None, layers: dict = None) -> dict:
    "Add or remove tags, and name or clear user layers 8-31 ({'8': 'Water2', '9': ''}). Built-in tags and layers, duplicates and a tag still on a loaded object are refused before anything changes. Read back, saved to disk at once. A project setting, so scratch mode refuses it."
    return _unity("tags_layers_set", addTags=addTags, removeTags=removeTags, layers=layers)


def uy_sorting_layer_set(layers: list, rename: dict = None) -> dict:
    "Set the sorting layers: every name in drawing order, Default among them, with rename {'old': 'new'} applied first. Each existing layer keeps its id, so its renderers stay on it. Dropping one a loaded renderer uses is refused. Read back, saved at once. A project setting, so scratch mode refuses it."
    return _unity("sorting_layer_set", layers=layers, rename=rename)


def uy_layer_collision_get(dimension: str, layers: list = None) -> dict:
    "Which layers collide for 3d (Physics) or 2d (Physics2D): every pair among the named layers, or the ones given. Refused by name when that physics module is off. Read only."
    return _unity("layer_collision_get", dimension=dimension, layers=layers)


def uy_layer_collision_set(dimension: str, pairs: list) -> dict:
    "Set whether layer pairs collide, 3d or 2d: [{a, b, collide}], names or indexes. Read back both ways and from the settings asset, saved at once. A project setting, so scratch mode refuses it; play mode too."
    return _unity("layer_collision_set", dimension=dimension, pairs=pairs)


def uy_collider_fit(target: dict, shape: str, source: str = None, includeChildren: bool = None, padding: float = None, convex: bool = None, isTrigger: bool = None,
                    reuse: bool = None) -> dict:
    "Add or resize a box, sphere, capsule or mesh collider to fit what the object renders (every Renderer under it, or its MeshFilter mesh), padding in meters. Undoable. Read back, then from Collider.bounds after a physics sync."
    return _unity("collider_fit", target=target, shape=shape, source=source, includeChildren=includeChildren, padding=padding, convex=convex, isTrigger=isTrigger, reuse=reuse)


def uy_rigidbody_set(target: dict, values: dict, add: bool = None) -> dict:
    "Add or set a Rigidbody by its Unity 6 member names on every version (mass, linearDamping, angularDamping, useGravity, isKinematic, interpolation, collisionDetectionMode, constraints, centerOfMass, inertiaTensor, includeLayers...). drag and velocity are refused with the new name. Undoable. Read back."
    return _unity("rigidbody_set", target=target, values=values, add=add)


def uy_physics_material_create(path: str, dim: str = None, dynamicFriction: float = None, staticFriction: float = None, friction: float = None, bounciness: float = None,
                               frictionCombine: str = None, bounceCombine: str = None, assignTo: list = None, overwrite: bool = None) -> dict:
    "Create a 3d or 2d physics material asset (friction, bounciness, combine modes) and optionally assign it to colliders. Read back from the asset and each collider."
    return _unity("physics_material_create", path=path, dim=dim, dynamicFriction=dynamicFriction, staticFriction=staticFriction, friction=friction, bounciness=bounciness,
                  frictionCombine=frictionCombine, bounceCombine=bounceCombine, assignTo=assignTo, overwrite=overwrite)


def uy_physics_raycast(origin: list, direction: list, shape: str = None, radius: float = None, halfExtents: list = None, rotation: list = None, point2: list = None,
                       maxDistance: float = None, layerMask=None, triggers: str = None, all: bool = None, scene: str = None) -> dict:
    "Cast a ray, sphere, box or capsule into the 3D colliders in edit mode, transforms synced first; the nearest hit, or all nearest first: object, point, normal, distance. No hit is hit:false. Read only."
    return _unity("physics_raycast", origin=origin, direction=direction, shape=shape, radius=radius, halfExtents=halfExtents, rotation=rotation, point2=point2,
                  maxDistance=maxDistance, layerMask=layerMask, triggers=triggers, all=all, scene=scene)


def uy_physics_overlap(shape: str, center: list = None, point0: list = None, point1: list = None, halfExtents: list = None, rotation: list = None, radius: float = None,
                       layerMask=None, triggers: str = None, against: dict = None, closestPoint: list = None) -> dict:
    "The 3D colliders inside a box, sphere or capsule in edit mode; with against, each one's penetration against that collider; with closestPoint, the nearest point on each. Read only."
    return _unity("physics_overlap", shape=shape, center=center, point0=point0, point1=point1, halfExtents=halfExtents, rotation=rotation, radius=radius,
                  layerMask=layerMask, triggers=triggers, against=against, closestPoint=closestPoint)


def uy_physics2d_query(kind: str, origin: list = None, direction: list = None, distance: float = None, radius: float = None, size: list = None, angle: float = None,
                       ray: dict = None, a: dict = None, b: dict = None, layerMask=None, triggers: bool = None) -> dict:
    "2D physics in edit mode, transforms synced first: raycast, circlecast, boxcast, rayIntersection, overlapPoint, overlapCircle, overlapBox, or the distance between colliders a and b. Read only."
    return _unity("physics2d_query", kind=kind, origin=origin, direction=direction, distance=distance, radius=radius, size=size, angle=angle, ray=ray, a=a, b=b,
                  layerMask=layerMask, triggers=triggers)


def uy_physics_simulate(seconds: float, bodies: list = None, scene: str = None, step: float = None, untilSleep: bool = None, keep: bool = None, sample: int = None) -> dict:
    "Step 3D physics in edit mode and measure where the bodies end up (start, end, displacement, velocity, sleeping, hinge angle). Simulation mode set to Script for the call and put back; everything moved is put back unless keep:true (then one undo step). Nothing moving fails. Runs alone."
    return _unity("physics_simulate", bodies=bodies, scene=scene, seconds=seconds, step=step, untilSleep=untilSleep, keep=keep, sample=sample)


def uy_physics2d_simulate(seconds: float, bodies: list = None, scene: str = None, step: float = None, untilSleep: bool = None, keep: bool = None, sample: int = None,
                          layers: int = None) -> dict:
    "Step 2D physics in edit mode and measure where the bodies end up, as uy_physics_simulate: mode set to Script and put back, everything put back unless keep:true. layers is Unity 6 only. Runs alone."
    return _unity("physics2d_simulate", bodies=bodies, scene=scene, seconds=seconds, step=step, untilSleep=untilSleep, keep=keep, sample=sample, layers=layers)


def uy_joint_set(target: dict, type: str, add: bool = None, connectedBody=None, anchor: list = None, connectedAnchor: list = None, axis: list = None,
                 autoConfigureConnectedAnchor: bool = None, breakForce=None, breakTorque=None, enableCollision: bool = None, hinge: dict = None, spring: dict = None,
                 character: dict = None, configurable: dict = None) -> dict:
    "Add or configure a fixed, hinge, spring, character or configurable joint: connected body (a target, or 'world' to clear it), anchors, axis, break limits ('infinity'), and the type's block of limits, springs, motors, drives. Undoable. Read back field by field."
    return _unity("joint_set", target=target, type=type, add=add, connectedBody=NULL if connectedBody == "world" else connectedBody, anchor=anchor, connectedAnchor=connectedAnchor, axis=axis,
                  autoConfigureConnectedAnchor=autoConfigureConnectedAnchor, breakForce=breakForce, breakTorque=breakTorque, enableCollision=enableCollision, hinge=hinge,
                  spring=spring, character=character, configurable=configurable)


def uy_character_controller_set(target: dict, values: dict = None, add: bool = None, fitToRenderers: bool = None, probe: dict = None) -> dict:
    "Add or set a CharacterController (radius, height, center, slopeLimit, stepOffset, skinWidth...), or fit it to the renderers; probe {motion} moves it once in edit mode and puts it back. Undoable. Read back."
    return _unity("character_controller_set", target=target, values=values, add=add, fitToRenderers=fitToRenderers, probe=probe)


def uy_rigidbody2d_set(target: dict, values: dict, add: bool = None) -> dict:
    "Add or set a Rigidbody2D by its Unity 6 member names on every version (bodyType, mass, gravityScale, linearDamping, constraints, sleepMode, sharedMaterial...). drag and isKinematic are refused with the name to use. Undoable. Read back."
    return _unity("rigidbody2d_set", target=target, values=values, add=add)


def uy_collider2d_set(target: dict, shape: str, reuse: bool = None, fit: bool = None, offset: list = None, size: list = None, radius: float = None, direction: str = None,
                      paths: list = None, sides: int = None, points: list = None, edgeRadius: float = None, isTrigger: bool = None, compositeOperation: str = None,
                      usedByComposite=None) -> dict:
    "Add or shape a box, circle, capsule, polygon or edge collider 2D, or fit one to the object's renderer; polygon paths and edge points read back point by point. usedByComposite is refused: use compositeOperation. Undoable."
    return _unity("collider2d_set", target=target, shape=shape, reuse=reuse, fit=fit, offset=offset, size=size, radius=radius, direction=direction, paths=paths, sides=sides,
                  points=points, edgeRadius=edgeRadius, isTrigger=isTrigger, compositeOperation=compositeOperation, usedByComposite=usedByComposite)


def uy_joint2d_set(target: dict, type: str, add: bool = None, connectedBody=None, anchor: list = None, connectedAnchor: list = None,
                   autoConfigureConnectedAnchor: bool = None, enableCollision: bool = None, breakForce=None, breakTorque=None, hinge: dict = None, distance: dict = None,
                   spring: dict = None, fixed: dict = None, slider: dict = None, wheel: dict = None) -> dict:
    "Add or configure a 2D hinge, distance, spring, fixed, slider or wheel joint: connected body (a target, or 'world' to clear it), anchors, break limits and the type's block. Undoable. Read back field by field."
    return _unity("joint2d_set", target=target, type=type, add=add, connectedBody=NULL if connectedBody == "world" else connectedBody, anchor=anchor, connectedAnchor=connectedAnchor,
                  autoConfigureConnectedAnchor=autoConfigureConnectedAnchor, enableCollision=enableCollision, breakForce=breakForce, breakTorque=breakTorque, hinge=hinge,
                  distance=distance, spring=spring, fixed=fixed, slider=slider, wheel=wheel)


def uy_composite_collider2d_build(target: dict, geometryType: str = None, generationType: str = None, vertexDistance: float = None, merge: list = None) -> dict:
    "Merge an object's 2D colliders (its children's and a tilemap's too) into a CompositeCollider2D and read the generated outline: paths and points. Adds a Static Rigidbody2D when missing. Nothing to merge fails. Undoable."
    return _unity("composite_collider2d_build", target=target, geometryType=geometryType, generationType=generationType, vertexDistance=vertexDistance, merge=merge)


def uy_physics_settings_get(dim: str = None, matrix: bool = None, namedOnly: bool = None) -> dict:
    "The 3D and 2D physics settings (gravity, contact offset, solver iterations, simulation mode...) and both layer matrices as the pairs that do not collide. A module that is off answers {module: off}. Read only."
    return _unity("physics_settings_get", dim=dim, matrix=matrix, namedOnly=namedOnly)


def uy_physics_settings_set(values3d: dict = None, values2d: dict = None) -> dict:
    "Write 3D and 2D physics settings by their runtime names; read back through the runtime getter and the settings asset, saved at once. A project setting, so scratch mode refuses it; play mode too. The layer matrix is uy_layer_collision_set's."
    return _unity("physics_settings_set", values3d=values3d, values2d=values2d)


def uy_mesh_info(target: dict, samples: int = None, uvMetrics: bool = None) -> dict:
    "A mesh's facts from a mesh asset or an object with a MeshFilter or SkinnedMeshRenderer: counts per submesh and topology, vertex attributes, UV channels, blend shapes, bind poses, bones, materials beside submeshes. Read only."
    return _unity("mesh_info", target=target, samples=samples, uvMetrics=uvMetrics)


def uy_mesh_create(path: str, vertices: list, submeshes: list, normals: list = None, recalculateNormals: bool = None, tangents: str = None, uv0: list = None,
                   uv1: list = None, uv2: list = None, uv3: list = None, colors: list = None, indexFormat: str = None, assignTo: dict = None) -> dict:
    "Build a Mesh asset from arrays (vertices, indices per submesh with topology, normals, UVs, colors), every index checked first, read back element by element; optionally put on an object's MeshFilter."
    return _unity("mesh_create", path=path, vertices=vertices, submeshes=submeshes, normals=normals, recalculateNormals=recalculateNormals, tangents=tangents, uv0=uv0, uv1=uv1,
                  uv2=uv2, uv3=uv3, colors=colors, indexFormat=indexFormat, assignTo=assignTo)


def uy_mesh_optimize(target: dict, op: str, compression: str = None) -> dict:
    "Reorder a Mesh asset's buffers for the GPU cache (optimize, indices, vertices) or set its compression; counts and bounds compared before and after. A model's mesh is refused: use uy_model_import_set."
    return _unity("mesh_optimize", target=target, op=op, compression=compression)


def uy_skinned_mesh_info(target: dict, bake: bool = None) -> dict:
    "A SkinnedMeshRenderer's bones as scene paths, whether they match the bind poses, the root bone, influences, unweighted vertices, blend shape weights; bake answers a posed snapshot's bounds. Read only."
    return _unity("skinned_mesh_info", target=target, bake=bake)


def uy_skinned_blendshape_set(target: dict, weights: dict) -> dict:
    "Set blend shape weights on a SkinnedMeshRenderer by shape name. An unknown name is refused with the list. Undoable. Read back."
    return _unity("skinned_blendshape_set", target=target, weights=weights)


def uy_model_info(target: dict, keys: list = None) -> dict:
    "A model file's import settings by ModelImporter property name, sub-assets by type, transform paths, material slots, the remap table and the embedded materials. Read only."
    return _unity("model_info", target=target, keys=keys)


def uy_model_import_set(target: dict, settings: dict) -> dict:
    "Set a model's import settings by ModelImporter property name (globalScale, isReadable, meshCompression, generateSecondaryUV, importTangents, addCollider, materialImportMode...), reimport, read back from the importer and from the imported meshes. Rig and animation keys belong to uy_model_rig_set and uy_model_clips_set."
    return _unity("model_import_set", target=target, settings=settings)


def uy_model_materials_remap(target: dict, map: dict = None, search: dict = None) -> dict:
    "Point a model's material slots at project materials (map {slot: {asset}}, None removes a remap), or search and remap by name; reimport, read back the table and the imported renderers' materials."
    return _unity("model_materials_remap", target=target, map=map, search=search)


def uy_model_extract(target: dict, folder: str, materials: bool = None, textures: bool = None) -> dict:
    "Extract a model's embedded materials (and textures) into a scratch folder as editable files; the slots are remapped to them. Nothing to extract fails."
    return _unity("model_extract", target=target, folder=folder, materials=materials, textures=textures)

def uy_undo(steps: int = None) -> dict:
    "Undo the editor's last undo groups, one step at a time, each named. In scratch mode only the bridge's own groups: a person's step is redone back at once and refused. Nothing to undo answers nothingToUndo."
    return _unity("undo", steps=steps)


def uy_redo(steps: int = None) -> dict:
    "Redo the editor's last undone groups, each named. In scratch mode only the bridge's own groups: a person's step is undone back at once and refused. Nothing to redo answers nothingToRedo."
    return _unity("redo", steps=steps)


def uy_undo_state(limit: int = None) -> dict:
    "The current undo group and the undo groups this bridge opened in this domain, newest first. Whether a person's step sits above them cannot be read. Read only."
    return _unity("undo_state", limit=limit)


def uy_save_open_scenes(includeClean: bool = None, skipRefused: bool = None) -> dict:
    "Save every loaded dirty scene in one call. Untitled scenes are always refused (use uy_save_scene with a path); one scene the gate refuses refuses the call unless skipRefused. Read back: clean, file newer."
    return _unity("save_open_scenes", includeClean=includeClean, skipRefused=skipRefused)


def uy_describe_type(type: str, depth: int = None, limit: int = None) -> dict:
    "A component's or ScriptableObject's serialized fields and their default values, from a hidden throwaway that is destroyed before the reply. Read only."
    return _unity("describe_type", type=type, depth=depth, limit=limit)


def uy_diff_vs_default(target: dict, limit: int = None) -> dict:
    "Which of a component's or ScriptableObject's serialized values differ from a fresh instance of its type. Zero differences is count 0 with how many it compared. Read only."
    return _unity("diff_vs_default", target=target, limit=limit)


def uy_package_info(name: str = None, direct: bool = None) -> dict:
    "The project's registered packages, or one by name: version, source, direct or not, resolved path and whether it exists, registry, git, errors, dependencies and the versions the registry knows (fromRegistry false when it knows none). Read only."
    return _unity("package_info", name=name, direct=direct)


def uy_package_search(query: str = None, offline: bool = None, searchId: str = None) -> dict:
    "Search the package registry for a name or id, or everything with no query. The first call answers pending with a searchId: poll with uy_package_search(searchId=...) until pending is false. A reload drops it (lost:true); a registry failure answers its errorCode; nothing found fails. Changes no project data."
    return _unity("package_search", query=query, offline=offline, searchId=searchId)


def uy_preset_create(source: dict, path: str, exclude: list = None, overwrite: bool = None) -> dict:
    "Save an object's settings (a component, an asset, or an importer as {asset, importer: True}) as a .preset asset under Assets/_Mif in scratch; overwrite updates an existing one. Read back from the file with DataEquals."
    return _unity("preset_create", source=source, path=path, exclude=exclude, overwrite=overwrite)


def uy_preset_apply(preset: str, target: dict, properties: list = None) -> dict:
    "Apply a .preset to an object, whole or only the named property paths (a parent path takes its children). A preset of another type refuses preset-type. Undoable for a scene object; an importer target ({asset, importer: True}) is reimported. Read back."
    return _unity("preset_apply", preset=preset, target=target, properties=properties)


def uy_preset_info(preset: str, filter: str = None, limit: int = None) -> dict:
    "What a .preset holds: target type, whether it can be a default, excluded paths, and each property path with its value and object reference. Read only."
    return _unity("preset_info", preset=preset, filter=filter, limit=limit)


def uy_preset_defaults_get(type: str = None, target: dict = None) -> dict:
    "The Preset Manager's default presets: every type with defaults, one type's (Light), or those a given object would start from. Read only."
    return _unity("preset_defaults_get", type=type, target=target)


def uy_preset_defaults_set(type: str, defaults: list) -> dict:
    "Replace one type's default presets: [{preset, filter, enabled}], [] clears. A preset of another type is refused before anything changes. Read back row by row, saved at once. A project setting, so scratch mode refuses it."
    return _unity("preset_defaults_set", type=type, defaults=defaults)


def uy_search(query: str, providers: list = None, limit: int = None) -> dict:
    "Run a Unity Search query (t:Material, t:Light, ...) through chosen providers (ids from uy_search_providers; default the active ones). Each item: provider, id, label, description, score and the object as a target. A list the search had not finished says partial. Nothing found fails. Read only."
    return _unity("search", query=query, providers=providers, limit=limit)


def uy_search_providers() -> dict:
    "The Unity Search providers this editor has: id, name, filter prefix, active, explicit-only. Read only."
    return _unity("search_providers")


def uy_memory_stats(target: dict = None) -> dict:
    "The editor's memory (the game's in play mode) by Unity's own counters, in bytes; with target, that object's runtime size. The graphics driver figure is null with no graphics device. Read only."
    return _unity("memory_stats", target=target)


def uy_profiler_counters(category: str = None, filter: str = None, limit: int = None) -> dict:
    "The profiler counters this editor can record now: name, category (Render, Memory, Scripts, ...) and unit. Use a name from here in uy_profiler_record. Read only."
    return _unity("profiler_counters", category=category, filter=filter, limit=limit)


def uy_profiler_record(action: str, counters: list = None, frames: int = None) -> dict:
    "Record counters over frames: action start with counters [{category, name}] and frames, then read (count, last, min, max, mean per counter), then stop. A reload drops the recorders (read answers lost); enter play mode first, then start. Changes no project data."
    return _unity("profiler_record", action=action, counters=counters, frames=frames)


def uy_frame_timing(frames: int = None) -> dict:
    "CPU and GPU frame times from FrameTimingManager over the latest frames. Off answers enabled false and the setting that turns it on; no graphics device is refused. Read only."
    return _unity("frame_timing", frames=frames)


def uy_memory_snapshot(name: str = None, snapshotId: str = None) -> dict:
    "Take a memory snapshot file for the Memory Profiler package, under Library/MifBridge/snapshots/<name>.snap. Answers pending with a snapshotId: poll with uy_memory_snapshot(snapshotId=...). Read back: success, file, size."
    return _unity("memory_snapshot", name=name, snapshotId=snapshotId)


def uy_assembly_list(kind: str = None, name: str = None, script: str = None) -> dict:
    "The project's compiled assemblies (Editor, Player or PlayerWithoutTestAssemblies) with asmdef, references, flags; name gives one with its sources and defines; script says which assembly a .cs file compiles into. Read only."
    return _unity("assembly_list", kind=kind, name=name, script=script)


def uy_code_optimization_set(mode: str) -> dict:
    "Switch the editor session between Debug and Release code optimization. Unity recompiles and reloads; the reply comes before the reload. Refused while compiling or in play mode."
    return _unity("code_optimization_set", mode=mode)


def uy_vcs_status(paths: list = None) -> dict:
    "Whether Unity's version control is on and connected, and whether paths are open for edit, from the cached status. With it off every path is editable. Read only."
    return _unity("vcs_status", paths=paths)


def uy_editor_settings_get() -> dict:
    "The project's editor settings: asset serialization, enter play mode options from the serialized fields the engine reads, domainReloadOnPlay as uy_play computes it, line endings, naming and others. Read only."
    return _unity("editor_settings_get")


def uy_editor_settings_set(values: dict, confirmReserialize: bool = None) -> dict:
    "Set editor settings, e.g. {'enterPlayModeOptionsEnabled': True, 'enterPlayModeOptions': ['DisableDomainReload']} to skip the domain reload on play. serializationMode rewrites every asset and needs confirmReserialize. Read back, saved at once. A project setting, so scratch mode refuses it; play mode too."
    return _unity("editor_settings_set", values=values, confirmReserialize=confirmReserialize)


def uy_package_add(packages: list, allowGit: bool = None, remove: list = None) -> dict:
    "Off by default (MIF_BRIDGE_ALLOW_PACKAGES=1 at editor launch, and full mode). Add registry packages by name or name@version, optionally removing others. Answers pending: the add compiles and reloads, so poll uy_package_request."
    return _unity("package_add", packages=packages, allowGit=allowGit, remove=remove)


def uy_package_remove(name: str) -> dict:
    "Off by default, as uy_package_add. Remove a direct dependency; this bridge itself is refused. Answers pending: poll uy_package_request."
    return _unity("package_remove", name=name)


def uy_package_request() -> dict:
    "What the last package add or remove did, across its reload: state (running, registered, removed, failed, lost), added, removed, changed, and each asked package's version now. Read only."
    return _unity("package_request")
def uy_input_handling_get() -> dict:
    "Active Input Handling (InputManager, InputSystem or Both), which backend this editor session runs, restartPending when they differ, the Input System package, project-wide actions, the Build Profile caveat and the UI module needed. Read only."
    return _unity("input_handling_get")


def uy_input_axes_get(name: str = None) -> dict:
    "The legacy Input Manager's axes as its settings page shows them; name returns every axis of that name. Says when nothing reads them at runtime. Read only."
    return _unity("input_axes_get", name=name)

def uy_input_binding_check(paths: list = None, composites: list = None, interactions: str = None, processors: str = None) -> dict:
    "Check Input System binding paths against the registered device layouts (each component walked), and composites, interactions and processors by name, with didYouMean. A usage or wildcard is checked:false. Read only."
    return _unity("input_binding_check", paths=paths, composites=composites, interactions=interactions, processors=processors)


def uy_input_actions_create(path: str, maps: list, controlSchemes: list = None, overwrite: bool = None) -> dict:
    "Create an .inputactions asset in one call: maps, actions, bindings, composites, control schemes. Every path checked first (bad-binding with didYouMean, nothing written). Read back from the import."
    return _unity("input_actions_create", path=path, maps=maps, controlSchemes=controlSchemes, overwrite=overwrite)


def uy_input_actions_get(asset: dict = None, projectWide: bool = None, map: str = None, check: bool = None) -> dict:
    "Read an .inputactions asset or the project-wide actions: maps, actions, bindings with composite parts grouped, control schemes; paths checked, orphan groups and unused schemes flagged. Read only."
    return _unity("input_actions_get", asset=asset, projectWide=projectWide, map=map, check=check)


def uy_input_actions_edit(asset: dict, ops: list) -> dict:
    "Change an .inputactions asset all-or-nothing: add or remove maps, actions, bindings, composites and schemes, rename actions, change a binding. The first failing op answers failedOp with the file untouched. Read back."
    return _unity("input_actions_edit", asset=asset, ops=ops)


def uy_input_player_input_set(target: dict, actions: dict = None, defaultMap: str = None, defaultScheme: str = None, neverAutoSwitch: bool = None, behavior: str = None, camera: dict = None, uiInputModule: dict = None) -> dict:
    "Add or configure a PlayerInput: actions asset, default map and scheme (checked against it), auto-switch, behavior (InvokeUnityEvents gets one event slot per action), camera, UI module. Undoable. Read back."
    return _unity("input_player_input_set", target=target, actions=actions, defaultMap=defaultMap, defaultScheme=defaultScheme, neverAutoSwitch=neverAutoSwitch, behavior=behavior, camera=camera, uiInputModule=uiInputModule)


def uy_input_handling_set(value: str) -> dict:
    "Set Active Input Handling to InputManager, InputSystem or Both and save it. Takes effect after an editor restart, which this does not do (restartRequired). Gated. Read back from the file."
    return _unity("input_handling_set", value=value)


def uy_input_project_actions_set(asset: dict) -> dict:
    "Make an .inputactions asset the project-wide actions, or clear them with NULL. Read back from InputSystem.actions and the EditorBuildSettings entry. Gated."
    return _unity("input_project_actions_set", asset=asset)


def uy_input_settings_get() -> dict:
    "The Input System's settings: update mode, dead zones, press point, background behavior, supported devices, and the settings asset or inMemoryDefault. Read only."
    return _unity("input_settings_get")


def uy_input_settings_set(updateMode: str = None, defaultDeadzoneMin: float = None, defaultDeadzoneMax: float = None, defaultButtonPressPoint: float = None, backgroundBehavior: str = None, editorInputBehaviorInPlayMode: str = None, supportedDevices: list = None) -> dict:
    "Set the Input System's settings on the project's settings asset and save at once; an in-memory default is refused no-settings-asset. Gated. Read back."
    return _unity("input_settings_set", updateMode=updateMode, defaultDeadzoneMin=defaultDeadzoneMin, defaultDeadzoneMax=defaultDeadzoneMax, defaultButtonPressPoint=defaultButtonPressPoint,
                  backgroundBehavior=backgroundBehavior, editorInputBehaviorInPlayMode=editorInputBehaviorInPlayMode, supportedDevices=supportedDevices)

def uy_cinemachine_camera_create(name: str = None, parent: dict = None, position: list = None, rotation: list = None, follow: dict = None, lookAt: dict = None, priority: int = None, outputChannel: int = None,
                                 lens: dict = None, body: str = None, bodySettings: dict = None, aim: str = None, aimSettings: dict = None, evaluate: bool = None) -> dict:
    "Create a Cinemachine 3 camera with follow and look-at targets, priority, lens, a body and an aim component, solved once in the call (the pose Cinemachine computes). 2.x is refused package-version. Undoable. Read back."
    return _unity("cinemachine_camera_create", name=name, parent=parent, position=position, rotation=rotation, follow=follow, lookAt=lookAt, priority=priority, outputChannel=outputChannel,
                  lens=lens, body=body, bodySettings=bodySettings, aim=aim, aimSettings=aimSettings, evaluate=evaluate)


def uy_cinemachine_brain_set(target: dict = None, defaultBlend: dict = None, updateMethod: str = None, blendUpdateMethod: str = None, channelMask: int = None, ignoreTimeScale: bool = None, worldUpOverride: dict = None) -> dict:
    "Add or set the CinemachineBrain on a Camera (default the MainCamera): blend, update methods, channel mask; answers the cameras by priority and the active one. Undoable. Read back."
    return _unity("cinemachine_brain_set", target=target, defaultBlend=defaultBlend, updateMethod=updateMethod, blendUpdateMethod=blendUpdateMethod, channelMask=channelMask,
                  ignoreTimeScale=ignoreTimeScale, worldUpOverride=worldUpOverride)


def uy_cinemachine_evaluate(target: dict = None, steps: int = None, deltaTime: float = None, reset: bool = None, keep: bool = None) -> dict:
    "Drive a CinemachineBrain by manual updates in edit mode and read the output camera's pose, then restore its update method and every moved transform (unless keep). Refused in play mode."
    return _unity("cinemachine_evaluate", target=target, steps=steps, deltaTime=deltaTime, reset=reset, keep=keep)


def uy_recorder_settings_create(path: str, recorders: list, range: dict = None, frameRate: float = None, playback: str = None, capFrameRate: bool = None, exitPlayMode: bool = None) -> dict:
    "Create a Recorder settings file with movie and image-sequence recorders (input, size, format, output under Assets/_Mif or Recordings/) and a range. Works without graphics. Read back from the file."
    return _unity("recorder_settings_create", path=path, recorders=recorders, range=range, frameRate=frameRate, playback=playback, capFrameRate=capFrameRate, exitPlayMode=exitPlayMode)


def uy_recorder_run(action: str = None, settings: str = None, timeoutSeconds: float = None, exitPlayMode: bool = None) -> dict:
    "Record with a Recorder settings file: start enters play mode and records, status polls the phase and outputs. Needs a graphics device (refused no-graphics under -nographics). Gated."
    return _unity("recorder_run", action=action, settings=settings, timeoutSeconds=timeoutSeconds, exitPlayMode=exitPlayMode)


def uy_timeline_build(path: str, frameRate: float = None, duration: float = None, tracks: list = None, markers: list = None) -> dict:
    "Create a Timeline asset in one call: frame rate, fixed or clip-based duration, tracks nested in groups, clips (start, duration, clip-in, speed, ease, blend) and signal markers. All checked before the asset exists. Needs com.unity.timeline. Read back from the reimported file."
    return _unity("timeline_build", path=path, frameRate=frameRate, duration=duration, tracks=tracks, markers=markers)


def uy_timeline_get(target: dict, clips: bool = None, markers: bool = None) -> dict:
    "A Timeline asset as data: frame rate, duration, the track tree with paths, types, binding types, clips and markers. Needs com.unity.timeline. Read only."
    return _unity("timeline_get", target=target, clips=clips, markers=markers)


def uy_signal_asset_create(path: str) -> dict:
    "Create a SignalAsset (.signal), the event a Timeline signal emitter raises. Needs com.unity.timeline. Read back."
    return _unity("signal_asset_create", path=path)


def uy_timeline_track_set(target: dict, add: list = None, set: list = None, remove: list = None, cascade: bool = None) -> dict:
    "Add, change (name, mute, lock, post-playback, avatar mask, offsets) and remove tracks on a timeline. A taken name, a Group with tracks (without cascade) and a locked track are refused first. Read back from the file."
    return _unity("timeline_track_set", target=target, add=add, set=set, remove=remove, cascade=cascade)


def uy_timeline_clip_set(target: dict, track: str, add: list = None, set: list = None, remove: list = None) -> dict:
    "Add, change and remove clips on one timeline track. A value the clip's ClipCaps does not allow is refused clip-caps before the write; a shared name is refused ambiguous. Read back from the file."
    return _unity("timeline_clip_set", target=target, track=track, add=add, set=set, remove=remove)


def uy_timeline_marker_set(target: dict, track: str = None, add: list = None, set: list = None, remove: list = None) -> dict:
    "Add, move and remove markers (signal emitters by default) on a track or the timeline's own marker track, indexed in time order. Read back from the file."
    return _unity("timeline_marker_set", target=target, track=track, add=add, set=set, remove=remove)


def uy_director_bind(target: dict, timeline: dict = None, bindings: dict = None, controlSources: dict = None, wrap: str = None, updateMode: str = None, playOnAwake: bool = None, initialTime: float = None) -> dict:
    "Set a PlayableDirector's timeline, bind tracks by path to GameObjects (the component each track needs is picked), resolve control clips' sources, and set wrap, update mode, play on awake and initial time. Undoable. Read back."
    return _unity("director_bind", target=target, timeline=timeline, bindings=bindings, controlSources=controlSources, wrap=wrap, updateMode=updateMode, playOnAwake=playOnAwake, initialTime=initialTime)


def uy_director_get(target: dict) -> dict:
    "A PlayableDirector: timeline, play state, time, duration, wrap and update modes, and each track's binding with the unbound ones. Read only."
    return _unity("director_get", target=target)


def uy_timeline_evaluate(target: dict, times: list, read: list = None) -> dict:
    "Evaluate a director's timeline at times in edit mode, read the bound objects (pose, active, properties), then put them back and prove it. Refused in play mode."
    return _unity("timeline_evaluate", target=target, times=times, read=read)


def uy_director_play(target: dict, action: str = None, time: float = None) -> dict:
    "In play mode: play, pause, resume or stop a PlayableDirector and set its time. Refused not-playing in edit mode. Read back."
    return _unity("director_play", target=target, action=action, time=time)


def uy_signal_receiver_set(target: dict, reactions: list = None, remove: list = None) -> dict:
    "Make a SignalReceiver call public void methods with no parameters when a signal fires, as persistent listeners; remove clears a signal. Undoable. Read back."
    return _unity("signal_receiver_set", target=target, reactions=reactions, remove=remove)


def uy_list_components(target: dict) -> dict:
    "A GameObject's components in order, with enabled state and missing scripts. Read only."
    return _unity("list_components", target=target)


def uy_find_types(nameContains: str, kind: str = None, includeAbstract: bool = None, limit: int = None) -> dict:
    "Component or ScriptableObject types by name, as compiled now: discover a type before uy_add_component or uy_create_asset. No match is a failure. Read only."
    return _unity("find_types", nameContains=nameContains, kind=kind, includeAbstract=includeAbstract, limit=limit)


def uy_copy_component_values(source: dict, target: dict, addIfMissing: bool = None) -> dict:
    "Copy every serialized value between components of the same type, adding it if asked. Undoable, each value read back. A Transform is refused."
    return _unity("copy_component_values", source=source, target=target, addIfMissing=addIfMissing)


def uy_wire_reference(target: dict, path: str, reference: dict) -> dict:
    "Point an object-reference field at an object, taking the component the field wants from a GameObject. A wrong type, a scene object into an asset, or a cross-scene reference is refused. Read back."
    return _unity("wire_reference", target=target, path=path, reference=reference)


def uy_create_prefab(target: dict, path: str, connect: bool = None, overwrite: bool = None) -> dict:
    "Save a scene GameObject as a prefab asset, by default making it an instance. On disk at once; in scratch mode only under Assets/_Mif. Read back against the source."
    return _unity("create_prefab", target=target, path=path, connect=connect, overwrite=overwrite)


def uy_instantiate_prefab(asset: str, parent: dict = None, name: str = None, position: list = None, rotation: list = None, ground: bool = None) -> dict:
    "Place a prefab asset or a model file in the active scene or under a parent as a real instance; ground puts its bounds' bottom at y 0. Undoable, read back: the link names that asset; renderers, triangles and world bounds."
    return _unity("instantiate_prefab", asset=asset, parent=parent, name=name, position=position, rotation=rotation, ground=ground)


def uy_prefab_info(target: dict) -> dict:
    "What an object is to the prefab system: asset type, instance status, source, a variant's base, overrides. Read only."
    return _unity("prefab_info", target=target)


def uy_prefab_overrides(target: dict, limit: int = None, includeDefault: bool = None) -> dict:
    "An instance's differences from its prefab: changed properties, added and removed components, added children. Read only."
    return _unity("prefab_overrides", target=target, limit=limit, includeDefault=includeDefault)


def uy_apply_prefab(target: dict) -> dict:
    "Write an instance's overrides into its prefab asset, on disk at once. Scene and asset both follow the path rule. Read back: no overrides left and the asset matches."
    return _unity("apply_prefab", target=target)


def uy_revert_prefab(target: dict) -> dict:
    "Drop an instance's overrides so it matches its prefab again. Undoable, read back."
    return _unity("revert_prefab", target=target)


def uy_create_prefab_variant(source: str, path: str) -> dict:
    "Make a prefab variant of a prefab asset, touching no open scene. In scratch mode only under Assets/_Mif. Read back: it is a variant, of that base."
    return _unity("create_prefab_variant", source=source, path=path)


def uy_unpack_prefab(target: dict, mode: str = None) -> dict:
    "Turn a prefab instance into plain objects: root, or completely with nested prefabs. Undoable, read back."
    return _unity("unpack_prefab", target=target, mode=mode)


def uy_edit_prefab(asset: str, ops: list) -> dict:
    "Edit a prefab asset all or nothing: ops apply to a private copy, the file is written only when all pass, then reopened and compared. ops: set_property, add_component, remove_component, create_child, delete_child, rename, each with a path relative to the root, / for the root. uy_describe_endpoint has each op's keys."
    return _unity("edit_prefab", asset=asset, ops=ops)


def uy_asset_info(path: str = None, guid: str = None) -> dict:
    "One asset by path or guid: guid, type, sub-assets, labels, importer, size, unsaved changes. Read only."
    return _unity("asset_info", path=path, guid=guid)


def uy_list_folder(path: str, recursive: bool = None, limit: int = None) -> dict:
    "What a folder holds. An existing folder with nothing in it fails with empty:true, a missing one with notFound. Read only."
    return _unity("list_folder", path=path, recursive=recursive, limit=limit)


def uy_asset_dependencies(path: str, recursive: bool = None, limit: int = None) -> dict:
    "What an asset uses, direct or the whole chain, itself included as self. Read only."
    return _unity("asset_dependencies", path=path, recursive=recursive, limit=limit)


def uy_asset_referencers(path: str, folder: str = None, limit: int = None, maxScan: int = None) -> dict:
    "What uses an asset, by scanning every asset under a folder, holding the main thread meanwhile. Zero rows with scanned above zero is the answer: nothing uses it. Read only."
    # 8,435 assets took over 120 s cold and 25 s warm on a spinning disk: the default 120 s wait would read a working scan as a failure.
    return _unity("asset_referencers", _timeout=600, path=path, folder=folder, limit=limit, maxScan=maxScan)


def uy_move_asset(source: str, destination: str) -> dict:
    "Move or rename an asset or folder; its guid and every reference survive. On disk at once; both ends follow the path rule. Read back."
    return _unity("move_asset", source=source, destination=destination)


def uy_copy_asset(source: str, destination: str) -> dict:
    "Copy an asset to a new path with a new guid. The source may be anywhere; the destination follows the path rule. Read back."
    return _unity("copy_asset", source=source, destination=destination)


def uy_set_asset_labels(path: str, labels: list) -> dict:
    "Replace an asset's labels; [] clears them. In scratch mode only under Assets/_Mif. Read back."
    return _unity("set_asset_labels", path=path, labels=labels)


def uy_get_importer(path: str, nameContains: str = None, limit: int = None) -> dict:
    "An asset's importer and its serialized settings, walked live, so the names are this version's own. Read only."
    return _unity("get_importer", path=path, nameContains=nameContains, limit=limit)


def uy_set_importer(path: str, values: dict) -> dict:
    "Write importer settings and reimport: {serialized path: value}, each checked against its type before any is written, and a fresh importer compared after."
    return _unity("set_importer", path=path, values=values)


def uy_reimport_asset(path: str) -> dict:
    "Import an asset again now, synchronously. Its source file does not change. Read back: it still loads."
    return _unity("reimport_asset", path=path)


def uy_save_asset(path: str) -> dict:
    "Write one asset's unsaved changes to disk, such as after uy_set_property on a material (uy_material_create and uy_material_set save their own). A person's other unsaved assets stay theirs. Read back."
    return _unity("save_asset", path=path)


def uy_open_scene_additive(path: str) -> dict:
    "Open a scene beside the ones already open. In scratch mode only under Assets/_Mif. Read back."
    return _unity("open_scene_additive", path=path)


def uy_close_scene(scene: str, discardUnsaved: bool = None) -> dict:
    "Close one open scene. The last scene, and a dirty scene without discardUnsaved, are refused. Read back."
    return _unity("close_scene", scene=scene, discardUnsaved=discardUnsaved)


def uy_set_active_scene(scene: str) -> dict:
    "Choose which open scene new objects land in. Saves nothing. Read back."
    return _unity("set_active_scene", scene=scene)


def uy_console_read(type: str = None, contains: str = None, sinceSeq: int = None, limit: int = None, includeStack: bool = None) -> dict:
    "Console messages since the bridge started, surviving reloads. Pass the last reply's lastSeq as sinceSeq to read what is new. No match is an answer; a console nobody captures fails. Read only."
    return _unity("console_read", type=type, contains=contains, sinceSeq=sinceSeq, limit=limit, includeStack=includeStack)


def uy_console_clear(editorConsole: bool = None) -> dict:
    "Empty the bridge's console record, and the Console window when asked. A compile error stays in the window while it stands. Read back."
    return _unity("console_clear", editorConsole=editorConsole)


def uy_editor_info() -> dict:
    "The editor and project now: versions, paths, active build target, render pipeline, color space, play and compile state, open and dirty scenes. Read only."
    return _unity("editor_info")


def uy_get_selection() -> dict:
    "What is selected in the editor. Nothing selected is an answer: count 0. Read only."
    return _unity("get_selection")


def uy_set_selection(targets: list) -> dict:
    "Select scene objects or assets; [] clears it. Each target is resolved before the selection changes. Saves nothing. Read back."
    return _unity("set_selection", targets=targets)


def uy_batch(calls: list, onError: str = None) -> dict:
    "Many calls, one hop, one undo group, one request id, each checked before the first runs: [{endpoint, args}], endpoint names as in uy_describe_endpoint. A later call can use an earlier reply: '$0.object.id'. onError rollback (default, undoable calls only), stop or continue. ok means every call was ok."
    return _unity("batch", calls=calls, onError=onError)


def uy_batch_read(calls: list, stopOnError: bool = None) -> dict:
    "Many read calls in one round trip: [{endpoint, args}]. Only read rows are admitted, so it works in read mode. ok means every call was ok."
    return _unity("batch_read", calls=calls, stopOnError=stopOnError)


def uy_run_csharp(code: str) -> dict:
    "Off by default: needs MIF_BRIDGE_ALLOW_RUN_CSHARP=1 at editor launch and write mode full. Compiles a method body and runs it on the editor's main thread."
    return _unity("run_csharp", code=code)


# ------------------------------------------------------------------ navigation, audio, video, 2D (1.5 batch 8)
def uy_navmesh_info(bounds: dict = None, includeTriangles: bool = None, areaNames: bool = None) -> dict:
    "The loaded navmesh as numbers: triangles, area in square meters, bounds, area by nav area, and the NavMeshSurfaces. No navmesh fails empty:true. Read only."
    return _unity("navmesh_info", bounds=bounds, includeTriangles=includeTriangles, areaNames=areaNames)


def uy_navmesh_path(start: list, end: list, agentType=None, areaMask=None, areaCosts: dict = None, snap: float = None) -> dict:
    "A path on the loaded navmesh: status, corners, length, for an agent type, area mask and per-call area costs. Read only."
    return _unity("navmesh_path", start=start, end=end, agentType=agentType, areaMask=areaMask, areaCosts=areaCosts, snap=snap)


def uy_navmesh_sample(points: list = None, rays: list = None, query: str = None, maxDistance: float = None, agentType=None, areaMask=None) -> dict:
    "Nearest navmesh point or edge for up to 1024 points, or a navmesh raycast per ray. Read only."
    return _unity("navmesh_sample", points=points, rays=rays, query=query, maxDistance=maxDistance, agentType=agentType, areaMask=areaMask)


def uy_navmesh_agent_types() -> dict:
    "Every agent type's build settings and every named nav area with its cost. Read only."
    return _unity("navmesh_agent_types")


def uy_navmesh_bake(target: dict, dataPath: str, mode: str = None, overwrite: bool = None) -> dict:
    "Bake a NavMeshSurface (AI Navigation package), save its NavMeshData where asked, read the navmesh back. sync, or async with navmesh_bake_status."
    return _unity("navmesh_bake", target=target, dataPath=dataPath, mode=mode, overwrite=overwrite)


def uy_navmesh_bake_status(bakeId: str = None) -> dict:
    "Poll async navmesh bakes: progress, and the finished bake's read-back. Read only."
    return _unity("navmesh_bake_status", bakeId=bakeId)


def uy_navmesh_clear(target: dict, deleteAsset: bool = None) -> dict:
    "Remove a NavMeshSurface's navmesh and delete its data asset. Read back: no triangles left in its bounds."
    return _unity("navmesh_clear", target=target, deleteAsset=deleteAsset)


def uy_navmesh_surface_set(target: dict, values: dict = None, create: bool = None) -> dict:
    "Add or configure a NavMeshSurface: agent type, what it collects, bounds, layers, overrides. Undoable. Read back, with Unity's ValidationReport."
    return _unity("navmesh_surface_set", target=target, values=values, create=create)


def uy_nav_component_set(target: dict, kind: str, values: dict, create: bool = None) -> dict:
    "Add or configure a NavMeshAgent, NavMeshObstacle, NavMeshModifier, NavMeshModifierVolume or NavMeshLink by API name. A link is proven by a path across it. Undoable."
    return _unity("nav_component_set", target=target, kind=kind, values=values, create=create)


def uy_nav_component_get(target: dict, kind: str = None) -> dict:
    "Read an object's navigation components, and an agent's live path state in play mode. Read only."
    return _unity("nav_component_get", target=target, kind=kind)


def uy_navmesh_agent_move(target: dict, destination: list, warpTo: list = None) -> dict:
    "Play mode: warp a NavMeshAgent and send it to a destination; poll nav_component_get for the movement. Refuses needs-play-mode and agent-off-navmesh."
    return _unity("navmesh_agent_move", target=target, destination=destination, warpTo=warpTo)


def uy_navmesh_agent_type_set(action: str, agentType=None, name: str = None, values: dict = None) -> dict:
    "Add, remove or edit an agent type in the project's navigation settings. Gated: scratch refuses it. Saved at once, read back."
    return _unity("navmesh_agent_type_set", action=action, agentType=agentType, name=name, values=values)


def uy_audio_source_set(target: dict, add: bool = None, clip: dict = None, resource: dict = None, values: dict = None, curves: dict = None, mixerGroup: dict = None,
                        clearMixerGroup: bool = None) -> dict:
    "Add or set an AudioSource: clip, members by API name, rolloff and spatial curves, mixer group output. Undoable. Read back, curves key by key."
    return _unity("audio_source_set", target=target, add=add, clip=clip, resource=resource, values=values, curves=curves, mixerGroup=mixerGroup, clearMixerGroup=clearMixerGroup)


def uy_audio_clip_info(target: dict, platforms: list = None) -> dict:
    "An AudioClip's length, samples, channels, frequency and its importer's default and per-platform sample settings. Read only."
    return _unity("audio_clip_info", target=target, platforms=platforms)


def uy_audio_import_set(target: dict, forceToMono: bool = None, ambisonic: bool = None, loadInBackground: bool = None, default: dict = None, overrides: dict = None) -> dict:
    "Set an audio file's import settings, default and per platform (null clears one); reimport and read back from a fresh importer and the clip."
    return _unity("audio_import_set", target=target, forceToMono=forceToMono, ambisonic=ambisonic, loadInBackground=loadInBackground, default=default, overrides=overrides)


def uy_audio_clip_create(path: str, seconds: float = None, frequency: int = None, channels: int = None, wave: str = None, toneHz: float = None, amplitude: float = None,
                         overwrite: bool = None) -> dict:
    "Write a 16-bit PCM WAV (silence, sine or noise) and import it. Read back from the RIFF header and the clip."
    return _unity("audio_clip_create", path=path, seconds=seconds, frequency=frequency, channels=channels, wave=wave, toneHz=toneHz, amplitude=amplitude, overwrite=overwrite)


def uy_audio_mixer_get(target: dict, includeSources: bool = None) -> dict:
    "A mixer's groups with their paths, snapshots, exposed parameters and output. Read only."
    return _unity("audio_mixer_get", target=target, includeSources=includeSources)


def uy_audio_mixer_param(target: dict, op: str, name: str = None, value: float = None, snapshot: str = None, seconds: float = None) -> dict:
    "Play mode, audio on: get, set or clear an exposed parameter on the running mixer, or transition to a snapshot. Refuses audio-disabled and needs-play-mode by name."
    return _unity("audio_mixer_param", target=target, op=op, name=name, value=value, snapshot=snapshot, seconds=seconds)


def uy_audio_reverb_zone_set(target: dict, values: dict, add: bool = None) -> dict:
    "Add or set an AudioReverbZone: distances, a preset or room values. Undoable. Read back."
    return _unity("audio_reverb_zone_set", target=target, values=values, add=add)


def uy_audio_listener_report(scene: str = None) -> dict:
    "The loaded scenes' AudioListeners and the problem they make (none or several), and the audio settings. Read only."
    return _unity("audio_listener_report", scene=scene)


def uy_video_player_set(target: dict, add: bool = None, source: str = None, clip: dict = None, url: str = None, renderMode: str = None, targetCamera: dict = None,
                        targetCameraAlpha: float = None, targetTexture: dict = None, targetMaterialRenderer: dict = None, targetMaterialProperty: str = None,
                        aspectRatio: str = None, isLooping: bool = None, playOnAwake: bool = None, waitForFirstFrame: bool = None, skipOnDrop: bool = None,
                        playbackSpeed: float = None, timeUpdateMode: str = None, audioOutputMode: str = None, audioSource: dict = None) -> dict:
    "Add or set a VideoPlayer: source, clip or URL, render mode and its target, loop, speed, audio route. Keys the render mode does not use are refused. Undoable. Read back."
    return _unity("video_player_set", target=target, add=add, source=source, clip=clip, url=url, renderMode=renderMode, targetCamera=targetCamera,
                  targetCameraAlpha=targetCameraAlpha, targetTexture=targetTexture, targetMaterialRenderer=targetMaterialRenderer,
                  targetMaterialProperty=targetMaterialProperty, aspectRatio=aspectRatio, isLooping=isLooping, playOnAwake=playOnAwake,
                  waitForFirstFrame=waitForFirstFrame, skipOnDrop=skipOnDrop, playbackSpeed=playbackSpeed, timeUpdateMode=timeUpdateMode,
                  audioOutputMode=audioOutputMode, audioSource=audioSource)


def uy_video_clip_info(target: dict, platforms: list = None) -> dict:
    "A VideoClip's size, frame rate, frame count, length and audio tracks, and its importer's transcode settings. Read only."
    return _unity("video_clip_info", target=target, platforms=platforms)


def uy_video_import_set(target: dict, importAudio: bool = None, keepAlpha: bool = None, flipVertical: bool = None, sRGBClip: bool = None, deinterlaceMode: str = None,
                        default: dict = None, targets: dict = None, maxSeconds: float = None, allowLong: bool = None) -> dict:
    "Set a video's import options and transcode settings, default and per platform; reimport and read back. A long transcode needs allowLong."
    return _unity("video_import_set", target=target, importAudio=importAudio, keepAlpha=keepAlpha, flipVertical=flipVertical, sRGBClip=sRGBClip,
                  deinterlaceMode=deinterlaceMode, default=default, targets=targets, maxSeconds=maxSeconds, allowLong=allowLong)


def uy_video_clip_create(path: str, width: int = None, height: int = None, fps: int = None, frames: int = None, pattern: str = None, color: list = None) -> dict:
    "Encode a short test video (.mp4 or .webm) with Unity's MediaEncoder and import it. Read back: frame count, size, frame rate."
    return _unity("video_clip_create", path=path, width=width, height=height, fps=fps, frames=frames, pattern=pattern, color=color)


def uy_video_player_probe(target: dict, frame: int = None) -> dict:
    "Prove a clip decodes: prepare (call again until prepared), step to a frame, read its center color, then stop the player. A URL source needs full mode; a frame past the clip is refused. Needs a graphics device; refuses no-graphics."
    return _unity("video_player_probe", target=target, frame=frame)


def uy_sprite_slice(target: dict, mode: str, cellSize: list = None, offset: list = None, padding: list = None, columns: int = None, rows: int = None, rects: list = None,
                    namePrefix: str = None, pivot: list = None, keepIds: bool = None, setMultiple: bool = None) -> dict:
    "Cut a texture into named sprites (grid, cells or explicit rects) through the Sprite Editor's data provider; names keep their spriteID. Reimport and read back."
    return _unity("sprite_slice", target=target, mode=mode, cellSize=cellSize, offset=offset, padding=padding, columns=columns, rows=rows, rects=rects, namePrefix=namePrefix,
                  pivot=pivot, keepIds=keepIds, setMultiple=setMultiple)


def uy_sprite_info(target: dict) -> dict:
    "A sprite, or every sprite of a texture: rect, pivot, ppu, border, bounds, physics shapes, packed. Read only."
    return _unity("sprite_info", target=target)


def uy_sprite_renderer_set(target: dict, add: bool = None, sprite: dict = None, color: list = None, flipX: bool = None, flipY: bool = None, drawMode: str = None,
                           size: list = None, tileMode: str = None, maskInteraction: str = None, sortingLayer: str = None, sortingOrder: int = None,
                           sortingGroup: bool = None) -> dict:
    "Add or set a SpriteRenderer: sprite, color, flips, draw mode and size, sorting layer by name and order (or on a SortingGroup). Undoable. Read back."
    return _unity("sprite_renderer_set", target=target, add=add, sprite=sprite, color=color, flipX=flipX, flipY=flipY, drawMode=drawMode, size=size, tileMode=tileMode,
                  maskInteraction=maskInteraction, sortingLayer=sortingLayer, sortingOrder=sortingOrder, sortingGroup=sortingGroup)


def uy_sprite_atlas_create(path: str, packables: list, padding: int = None, enableRotation: bool = None, enableTightPacking: bool = None, filterMode: str = None,
                           sRGB: bool = None, includeInBuild: bool = None, pack: bool = None, buildTarget: str = None) -> dict:
    "Create a Sprite Atlas (V1 or V2, as the packer mode needs), add packables, set packing, pack and read back what packed."
    return _unity("sprite_atlas_create", path=path, packables=packables, padding=padding, enableRotation=enableRotation, enableTightPacking=enableTightPacking,
                  filterMode=filterMode, sRGB=sRGB, includeInBuild=includeInBuild, pack=pack, buildTarget=buildTarget)


def uy_tile_create(folder: str, sprites: list = None, fromTexture: dict = None, color: list = None, colliderType: str = None, flags: list = None, nameFormat: str = None,
                   overwrite: bool = None) -> dict:
    "Create Tile assets from sprites, one per sprite, with color, collider type and flags. Read back."
    return _unity("tile_create", folder=folder, sprites=sprites, fromTexture=fromTexture, color=color, colliderType=colliderType, flags=flags, nameFormat=nameFormat,
                  overwrite=overwrite)


def uy_tilemap_create(name: str = None, parent: dict = None, cellSize: list = None, cellGap: list = None, cellLayout: str = None, cellSwizzle: str = None,
                      tilemapName: str = None, mode: str = None, sortOrder: str = None, sortingLayer: str = None, sortingOrder: int = None, collider: bool = None,
                      composite: bool = None) -> dict:
    "Create a Grid with a Tilemap child: cell size and layout, renderer mode and sorting, optional collider and composite. Undoable. Read back."
    return _unity("tilemap_create", name=name, parent=parent, cellSize=cellSize, cellGap=cellGap, cellLayout=cellLayout, cellSwizzle=cellSwizzle, tilemapName=tilemapName,
                  mode=mode, sortOrder=sortOrder, sortingLayer=sortingLayer, sortingOrder=sortingOrder, collider=collider, composite=composite)


def uy_tilemap_set_tiles(target: dict, ops: list, compress: bool = None) -> dict:
    "Paint a tilemap with ops (set, block, box, flood, swap, clear, color, flags, rotate); every written cell read back. Undoable."
    return _unity("tilemap_set_tiles", target=target, ops=ops, compress=compress)


def uy_tilemap_get(target: dict, rect: dict = None, format: str = None, worldPoint: list = None, limit: int = None) -> dict:
    "Read a tilemap: bounds, used tiles, cells as rows with a legend or as a list, or the cell under a world point. Read only."
    return _unity("tilemap_get", target=target, rect=rect, format=format, worldPoint=worldPoint, limit=limit)


def uy_tilemap_renderer_set(target: dict, values: dict) -> dict:
    "Set a TilemapRenderer: mode, sort order, chunk size, culling, mask interaction, sorting layer by name. Undoable. Read back."
    return _unity("tilemap_renderer_set", target=target, values=values)


# THE ONE CALL THAT REACHES BOTH BACKENDS. server.py hands this module its Blender transport in _load_unity, since this file
# imports nothing from server.py; a suite sets it the same way.
BLENDER = None
SEND_IMPORT_WAIT = float(os.environ.get("MIF_UNITY_IMPORT_WAIT", "1800"))     # 1,200 textures held a real project's main thread 185 s, past the 120 s a call waits
_SEND_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def mif_send_to_unity(objects: list, name: str = None, textures: bool = True) -> dict:
    "Blender to Unity in one call: the named Blender objects are exported with profile unity into Assets/_Mif/<name>/ of the pinned project, with their textures and .meta files, the editor refreshes, and the import is read back: root rotation 0 and scale 1, and the mesh's height against Blender's. Blender and Unity must share a disk."
    tail = "NOTHING was written."
    if BLENDER is None:
        return {"ok": False, "error": "no Blender transport here: this runs from the MCP server. %s" % tail}
    if not isinstance(objects, list) or not objects or not all(isinstance(o, str) for o in objects):
        return {"ok": False, "error": "objects is a list of Blender object names. %s" % tail}
    name = name or objects[0]
    if not _SEND_NAME.match(name):
        return {"ok": False, "error": "name goes in a folder and a file name: letters, digits, _ and -, got %r. %s" % (name, tail)}
    where = _resolve()
    if where.get("ok") is False:
        return where
    # THE EDITOR'S GATE MAKES THE FOLDER (found 2026-11-22): this made it itself, so a send wrote in read mode and through a link. Nothing is
    # written here, by this module or by Blender, until send_prepare answers; its refusal is the gate's, passed on as it came.
    subs = ["Textures"] if textures else None
    prep = _unity("send_prepare", folder="Assets/_Mif/" + name, subfolders=subs)
    if prep.get("outcomeUnknown"):
        # A main thread held past the call's wait (a long import): wait it out, then the identical call, which the editor answers from its record.
        uy_wait_idle(timeoutSeconds=SEND_IMPORT_WAIT)
        prep = _unity("send_prepare", folder="Assets/_Mif/" + name, subfolders=subs)
    if prep.get("ok") is False:
        return dict(prep, step="send_prepare", error="%s %s" % (prep.get("error"), tail))
    folder = os.path.join(where["project"], "Assets", "_Mif", name)
    fbx = os.path.join(folder, name + ".fbx")
    sent = BLENDER("export_mesh", objects=objects, file=fbx, profile="unity")
    if not sent.get("ok"):
        return {"ok": False, "step": "bl_export_mesh", "error": sent.get("error")}
    steps = ["bl_export_mesh"]
    out = {"ok": True, "project": where["project"], "folder": folder, "file": fbx, "asset": "Assets/_Mif/%s/%s.fbx" % (name, name),
           "exported": [o.get("name") for o in sent.get("exported") or []], "axis": sent.get("axis")}
    for k in ("hullsLeftOut", "lodRenamedForFile", "lodWarning", "rigNote", "transformsKept"):
        if sent.get(k) is not None:
            out[k] = sent[k]
    if textures:
        maps = os.path.join(folder, "Textures")
        tex = BLENDER("export_material_textures", objects=out["exported"], folder=maps, profile="unity")
        if tex.get("ok"):
            steps.append("bl_export_material_textures")
            out["materials"] = tex.get("materials")
            out["texturesWritten"] = tex.get("texturesWritten")
        elif "use no material" in str(tex.get("error")):
            out["texturesWritten"] = 0
        else:
            return dict(out, ok=False, step="bl_export_material_textures", error=tex.get("error"),
                        leftBehind=fbx, steps=steps)
    r = uy_refresh()
    steps.append("uy_refresh")
    # A refresh that outlives the call's own wait is an import still running, and it is what was asked for. The read-back below decides; the wait below is what ends it.
    if r.get("ok") is False and not (r.get("outcomeUnknown") and r.get("editorState") == "stalled"):
        return dict(out, ok=False, step="uy_refresh", error=r.get("error"), leftBehind=folder, steps=steps)
    idle = uy_wait_idle(timeoutSeconds=SEND_IMPORT_WAIT)
    steps.append("uy_wait_idle")
    if idle.get("ok") is False:
        return dict(out, ok=False, step="uy_wait_idle", error=idle.get("error"), leftBehind=folder, steps=steps)
    # READ BACK, never taken from the export's reply: what the importer made is the claim.
    target = {"asset": out["asset"], "component": "Transform"}
    rot, scale = uy_get_property(target, "m_LocalRotation"), uy_get_property(target, "m_LocalScale")
    back = {"rotation": rot.get("value"), "scale": scale.get("value")}
    problems = []
    if rot.get("ok") is False or scale.get("ok") is False:
        problems.append("the imported root could not be read: %s" % (rot.get("error") or scale.get("error")))
    else:
        q, s = back["rotation"] or {}, back["scale"] or {}
        if any(abs(float(q.get(k, 9))) > 1e-3 for k in "xyz") or abs(abs(float(q.get("w", 9))) - 1) > 1e-3:
            problems.append("the root's rotation is %s, not 0" % (q,))
        if any(abs(float(s.get(k, 9)) - 1) > 1e-4 for k in "xyz"):
            problems.append("the root's scale is %s, not 1" % (s,))
    height = sent.get("writtenHeight") or {}
    row = next((o for o in sent.get("exported") or [] if o.get("name") == height.get("object")), None)
    if len(out["exported"]) == 1 and row and row.get("isIdentityTransform"):
        box = uy_get_property({"asset": out["asset"], "subAsset": row["name"], "subAssetType": "Mesh"}, "m_LocalAABB.m_Extent")
        extent = box.get("value") if box.get("ok") is not False else None
        back["meshExtent"] = extent
        if extent is None or abs(2 * float(extent.get("y", 0)) - float(height.get("heightBU", 0))) > 1e-3:
            problems.append("the mesh's height in Unity is not Blender's %s: extent %s (%s)"
                            % (height.get("heightBU"), extent, box.get("error")))
    out.update(steps=steps, readBack=back, verified=not problems)
    if problems:
        return dict(out, ok=False, error="; ".join(problems), leftBehind=folder)
    return out


TOOLS = {name: fn for name, fn in sorted(globals().items()) if name.startswith("uy_") and callable(fn)}
TOOLS["mif_send_to_unity"] = mif_send_to_unity

# What the MCP server's own pack machinery reads. Not in packs.json: that file is generated from server.py and stays Unreal and Blender. Not in the default list.
PACK = {
    "title": "Unity",
    "description": "Drive a live Unity editor through typed, read-back calls: generic properties through SerializedObject, the hierarchy, components, prefabs, the asset database, scenes, the console, compile errors, play mode, batches, materials and textures by render pipeline, lights, cameras, Volumes, terrain, particles, Animator controllers and clips, uGUI, builds and tests. Needs the com.mifbridge.editor package in the project.",
    "backend": "unity",
    "tools": sorted(TOOLS),
    "rules": [
        {"rule": "Ask uy_describe_endpoint or uy_list_properties for a name; never guess one. Serialized names are not API and change between Unity versions (m_ShadowRadius became m_ShapeRadius on 6000.5).", "tools": ["uy_describe_endpoint", "uy_list_properties"]},
        {"rule": "ok on a write means the editor holds the value: it was read back. Read before and after only when you need the old value.", "tools": ["uy_set_property"]},
        {"rule": "In scratch mode, the default, writes go under Assets/_Mif or an untitled scene. A real scene is refused even in memory. A person raises the mode in the panel; nothing over the bridge can.", "tools": ["uy_save_scene", "uy_create_asset", "uy_set_property"]},
        {"rule": "A changed script reaches the editor only through uy_refresh. Then uy_wait_idle and uy_compile_status: refresh returning does not mean compiled, and the old build keeps running while errors stand.", "tools": ["uy_refresh", "uy_compile_status", "uy_wait_idle"]},
        {"rule": "Read the pipeline with uy_rp_info before anything pipeline-sensitive; never assume URP. Property names differ by pipeline, so ask uy_shader_props: it names the main color and main texture from the shader itself.", "tools": ["uy_rp_info", "uy_shader_props", "uy_material_create", "uy_material_set"]},
        {"rule": "Shader Graph and VFX Graph have no public authoring API, so uy_shader_graph_edit and uy_vfx_graph_edit refuse and say what can be done instead. Do not edit those files by hand: use a graph's exposed properties, or write a .shader or a Particle System.", "tools": ["uy_shader_graph_edit", "uy_vfx_graph_edit", "uy_vfx_read"]},
        {"rule": "A curve on a property that does not exist is stored and animates nothing: pass root to uy_clip_curves_set so each binding is checked. uy_particle_simulate shows what module values do; uy_capture_camera shows what a scene looks like when the editor has graphics.", "tools": ["uy_clip_curves_set", "uy_particle_simulate", "uy_capture_camera"]},
        {"rule": "Use uy_batch for several writes: one undo group, checked before the first runs, and a repeat of its request id never runs a call twice. Use uy_batch_read to read many things in one round trip.", "tools": ["uy_batch", "uy_batch_read"]},
        {"rule": "A prefab asset is changed by uy_edit_prefab, or through an instance and uy_apply_prefab. Never edit prefab or scene YAML by hand.", "tools": ["uy_edit_prefab", "uy_apply_prefab", "uy_prefab_overrides"]},
        {"rule": "uy_set_property on an asset changes it in memory; uy_save_asset writes it to disk.", "tools": ["uy_set_property", "uy_save_asset"]},
        {"rule": "A reload is not a failure: calls wait and retry by themselves, and the same request id never runs twice. outcomeUnknown means read the target; do not send a changed call.", "tools": ["uy_request_status"]},
    ],
}
