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
TOKEN = os.environ.get("MIF_BRIDGE_TOKEN", "dev")
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


def _win_handle(pid):
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    k = ctypes.windll.kernel32
    k.OpenProcess.restype = ctypes.c_void_p
    return k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))


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
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
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
        return os.readlink("/proc/%d/exe" % int(pid))
    except OSError:
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
        with open("/proc/%d/cmdline" % int(pid), "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
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


def _raw(port, endpoint, body=None, rid=None, read_timeout=5.0, connect_timeout=CONNECT_TIMEOUT):
    """One HTTP exchange. Returns (kind, data, headers): reply | empty | cut | refused | connect_timeout | read_timeout | bad_body.

    A SOCKET, NOT requests. One connection per call leaves a TIME_WAIT per call on whoever closes first, and at hundreds a second that runs out of ephemeral ports. Measured: 3,000 calls left 3,000, and a server
    that waits for the client's FIN and resets did not change it. SO_LINGER {1, 0} on close, once the reply is read in full by Content-Length, leaves none (the Blender client's fix). This also gives the classes above
    exactly, with no library's reading of its own exceptions."""
    payload = json.dumps(body or {}).encode("utf-8")
    lines = ["POST /%s HTTP/1.1" % endpoint, "Host: 127.0.0.1:%d" % int(port), "X-Mif-Token: %s" % TOKEN, "Content-Type: application/json", "Connection: close", "Content-Length: %d" % len(payload)]
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


def _raw_watched(port, endpoint, body, rid, timeout, domain):
    """_raw, while watching for a domain that changed under the call. A socket left open by a dying domain would otherwise hold a caller to its full read timeout, with nobody left to answer. If the domain moved and
    the reply has not come, this is a cut connection: the record of the id decides what happened, and if the reply does get here later it is the same reply the record holds."""
    box = {}

    def run():
        box["r"] = _raw(port, endpoint, body, rid, read_timeout=timeout)
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


_OWN_ERROR = re.compile(r"(?m)^(.*" + re.escape(PACKAGE_NAME) + r"[\\/][^(\r\n]*)\((\d+),(\d+)\):\s*error\s+(CS\d+):\s*(.*)$")


_ANY_ERROR = re.compile(r"(?m)^(\S[^(\r\n]*)\((\d+),(\d+)\):\s*error\s+(CS\d+):\s*(.*)$")


def own_assembly_errors(log_path):
    """Compile errors in our own package named by an editor log: [{file, line, column, code, message}]. Empty if none, or no log to read."""
    return _log_errors(log_path, _OWN_ERROR)


def project_errors(log_path):
    """Every compile error an editor log names, in any assembly, in the same shape."""
    return _log_errors(log_path, _ANY_ERROR)


def _installed_here(project):
    if os.path.isdir(os.path.join(project, "Packages", PACKAGE_NAME)):
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
        kind, data, _h = _raw(port, "status", read_timeout=3.0)
        # The bridge's own refusal of this client's token. It is this editor answering, by its headers, and not somebody else on the port: a token saved in the panel is the usual way here.
        if kind == "reply" and isinstance(data, dict) and "X-Mif-Token" in str(data.get("error")) and (_h or {}).get("X-Mif-Backend") == "unity" and str((_h or {}).get("X-Mif-Instance")) == str(file_pid):
            out.update(editorState="wrongInstance", refusedRule="token", retryable=False,
                       error="the bridge on port %s refused this client's token. Set MIF_BRIDGE_TOKEN to the one saved in this project's panel, or clear it there." % port)
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
        kind, data, headers = _raw_watched(port, endpoint, body, rid, timeout, (st.get("status") or {}).get("domain"))
        if kind == "reply":
            instance = headers.get("X-Mif-Instance")
            if headers.get("X-Mif-Backend") != "unity":
                return {"ok": False, "editorState": "wrongInstance", "endpoint": endpoint, "requestId": rid, "outcomeUnknown": True,
                        "error": "port %s answered, and not as a Unity bridge. Not trusting the reply." % port, "holder": _port_holder(port)}
            if instance and str(st.get("pid")) != instance and str((st.get("status") or {}).get("pid")) != instance:
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
        st = _await_answer(project, deadline)
        if not st.get("answered"):
            return _failed_state(st, rid, endpoint, may_have_run)
        port = (st.get("status") or {}).get("port") or st.get("port")
        k2, record, _h2 = _raw(port, "request_status", {"requestId": rid}, read_timeout=5.0)
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
        may_have_run = False


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
    "Say what the Unity editor is doing: ready, compiling, errors, playing, reloading, stalled, starting, down, or serverNotLoaded (a live editor whose bridge assembly failed to compile, with file and line). Read only."
    where = _resolve()
    if where.get("ok") is False:
        return where
    st = state_of(where["project"], _PIN.get("log"))
    out = dict(st, ok=bool(st.get("answered")), clientProtocol=PROTOCOL)
    theirs = (st.get("status") or {}).get("protocol")
    if theirs is not None and theirs != PROTOCOL:
        # The package is source inside each project, so each project can carry a different version.
        out["protocolMismatch"] = "this server speaks protocol %s and the package in this project speaks %s. Update the older one." % (PROTOCOL, theirs)
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
    "Enter, exit, pause or unpause play mode, and wait for the outcome: the editor refuses play silently, so this reads what really happened. Refused in scratch mode."
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


def uy_list_hierarchy(root: dict = None, scene: str = None, depth: int = None, nameContains: str = None, components: bool = None, limit: int = None) -> dict:
    "The GameObjects of the loaded scenes, or under one object, to a depth, with a name filter. Read only."
    return _unity("list_hierarchy", root=root, scene=scene, depth=depth, nameContains=nameContains, components=components, limit=limit)


def uy_get_object(target: dict) -> dict:
    "One GameObject: identity, active state, tag, layer, local transform, components, children. Read only."
    return _unity("get_object", target=target)


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
    "Create a folder and its parents. Asset writes go to disk at once; in scratch mode only under Assets/_Mif."
    return _unity("create_folder", path=path)


def uy_create_asset(path: str, type: str, shader: str = None) -> dict:
    "Create a Material, or a ScriptableObject by type name, at a path. Asset writes go to disk at once; in scratch mode only under Assets/_Mif."
    return _unity("create_asset", path=path, type=type, shader=shader)


def uy_delete_asset(path: str) -> dict:
    "Delete an asset from disk. Not undoable. In scratch mode only under Assets/_Mif."
    return _unity("delete_asset", path=path)


def uy_save_scene(scene: str = None, path: str = None) -> dict:
    "Save a loaded scene to its path or a new one, and read it back. In scratch mode only under Assets/_Mif."
    return _unity("save_scene", scene=scene, path=path)


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


def uy_rp_assign(target: dict = None) -> dict:
    "Make a pipeline asset the project's default, or go back to Built-in with no target. Changes project settings, so scratch mode refuses it. Read back from currentRenderPipeline."
    return _unity("rp_assign", target=target)


def uy_shader_props(shader: str = None, target: dict = None, includeHidden: bool = None) -> dict:
    "A shader's properties from the shader itself: names, types, ranges, defaults, its main color and main texture, its keywords, and whether it runs on this project's pipeline. Names differ by pipeline; ask rather than guess. Read only."
    return _unity("shader_props", shader=shader, target=target, includeHidden=includeHidden)


def uy_shader_errors(shader: str = None, target: dict = None) -> dict:
    "Whether a shader compiled: errors and warnings by platform and line, and whether a compile is still going. Read only."
    return _unity("shader_errors", shader=shader, target=target)


def uy_material_create(path: str, shader: str = None, parent: dict = None, properties: dict = None, keywords: list = None) -> dict:
    "Create a material with this pipeline's default shader or a named one, optionally a variant, with properties and keywords. A shader for another pipeline is refused, since it renders magenta with no message. Read back."
    return _unity("material_create", path=path, shader=shader, parent=parent, properties=properties, keywords=keywords)


def uy_material_get(target: dict, slot: int = None, nameContains: str = None) -> dict:
    "A material's shader, every property's value and type, keywords, render queue and variant parent, from an asset or a renderer's slot. Read only."
    return _unity("material_get", target=target, slot=slot, nameContains=nameContains)


def uy_material_set(target: dict, slot: int = None, properties: dict = None, enableKeywords: list = None, disableKeywords: list = None, renderQueue: int = None) -> dict:
    "Set material properties, keywords and queue. Names and types are checked against the shader first, a value outside a Range is refused, the shader's own validation runs after, as the inspector's would, and each value is read back."
    return _unity("material_set", target=target, slot=slot, properties=properties, enableKeywords=enableKeywords, disableKeywords=disableKeywords, renderQueue=renderQueue)


def uy_material_assign(target: dict, material: dict, slot: int = None) -> dict:
    "Put a material in a renderer's slot. A slot out of range and a material for another pipeline are refused. Undoable. Read back."
    return _unity("material_assign", target=target, material=material, slot=slot)


def uy_shader_graph_edit(asset: str = None, operation: str = None) -> dict:
    "Always refuses, with what cannot be done and what can. Shader Graph authoring has no public API in Unity. Materials and exposed properties of a graph's shader work through uy_material_* and uy_shader_props."
    return _unity("shader_graph_edit", asset=asset, operation=operation)


def uy_vfx_graph_edit(asset: str = None, operation: str = None) -> dict:
    "Always refuses, with what cannot be done and what can. VFX Graph authoring has no public API in Unity. uy_vfx_read reads a graph's exposed face; Particle Systems are fully public."
    return _unity("vfx_graph_edit", asset=asset, operation=operation)


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


def uy_camera_set(target: dict, settings: dict) -> dict:
    "Change a Camera. Checked before anything is written; a near plane at or past the far plane is refused. Undoable. Read back."
    return _unity("camera_set", target=target, settings=settings)


def uy_volume_profile_create(path: str) -> dict:
    "Create an empty Volume profile. URP and HDRP only: Built-in has no Volume system, and this refuses by name. Read back."
    return _unity("volume_profile_create", path=path)


def uy_volume_get(target: dict, component: str = None) -> dict:
    "A Volume profile's components, each parameter's value, type and override state, and the component types this pipeline offers. Read only."
    return _unity("volume_get", target=target, component=component)


def uy_volume_set(target: dict, component: str, values: dict, add: bool = None, active: bool = None) -> dict:
    "Set a Volume component's parameters, turning each override on, adding the component when asked. Names and types are checked on a throwaway instance first. Read back."
    return _unity("volume_set", target=target, component=component, values=values, add=add, active=active)


def uy_bake_lighting(action: str, lightmapper: str = None) -> dict:
    "Start or cancel a lightmap bake of the active scene. It runs in the background: poll uy_bake_status. An untitled scene is refused, because a bake writes beside the scene."
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


def uy_terrain_create(name: str, dataPath: str, heightmapResolution: int = None, size: list = None, alphamapResolution: int = None, parent: dict = None, position: list = None) -> dict:
    "Create a Terrain and its TerrainData asset, with this pipeline's terrain material. A resolution Unity would snap is refused. Read back."
    return _unity("terrain_create", name=name, dataPath=dataPath, heightmapResolution=heightmapResolution, size=size, alphamapResolution=alphamapResolution, parent=parent, position=position)


def uy_terrain_info(target: dict) -> dict:
    "A terrain: resolutions, size, height statistics, layers with coverage, trees, material. Read only."
    return _unity("terrain_info", target=target)


def uy_terrain_set_heights(target: dict, x: int = None, y: int = None, heights: list = None, fill: float = None, width: int = None, height: int = None) -> dict:
    "Write heights, 0 to 1 of the terrain's height, from rows or a fill. Bounds and values are checked first. Undoable. Read back through GetHeights."
    return _unity("terrain_set_heights", target=target, x=x, y=y, heights=heights, fill=fill, width=width, height=height)


def uy_terrain_add_layer(target: dict, layer: dict = None, texture: dict = None, layerPath: str = None, tileSize: list = None) -> dict:
    "Add a paint layer, an existing TerrainLayer or a new one from a texture. Read back."
    return _unity("terrain_add_layer", target=target, layer=layer, texture=texture, layerPath=layerPath, tileSize=tileSize)


def uy_terrain_paint_layer(target: dict, layer, weight: float, x: int = None, y: int = None, width: int = None, height: int = None) -> dict:
    "Paint one layer's weight over a block, scaling the others so each texel still sums to 1. Undoable. Read back through GetAlphamaps."
    return _unity("terrain_paint_layer", target=target, layer=layer, weight=weight, x=x, y=y, width=width, height=height)


def uy_particle_create(name: str, parent: dict = None, position: list = None, rotation: list = None, material: dict = None) -> dict:
    "Create a Particle System with this pipeline's particle material or a named one; another pipeline's is refused. Undoable. Read back."
    return _unity("particle_create", name=name, parent=parent, position=position, rotation=rotation, material=material)


def uy_particle_module_get(target: dict, module: str = None, limit: int = None) -> dict:
    "A Particle System's modules with their enabled flags, or every serialized value inside one, by the path uy_particle_module_set takes. Read only."
    return _unity("particle_module_get", target=target, module=module, limit=limit)


def uy_particle_module_set(target: dict, values: dict) -> dict:
    "Write module values by serialized path, as one undo. Every path and type is checked first. Read back. uy_particle_simulate then shows what they do."
    return _unity("particle_module_set", target=target, values=values)


def uy_particle_simulate(target: dict, seconds: float, restart: bool = None, withChildren: bool = None, sample: int = None, allowEmpty: bool = None) -> dict:
    "Scrub a Particle System to a time and read its particles: count, bounds, a sample. No particles is a failure unless allowEmpty. Changes nothing saved."
    return _unity("particle_simulate", target=target, seconds=seconds, restart=restart, withChildren=withChildren, sample=sample, allowEmpty=allowEmpty)


def uy_vfx_read(target: dict) -> dict:
    "What VFX Graph makes public: exposed properties and events of a VisualEffect or a .vfx asset, and a live one's alive particles. Built-in refuses by name. Read only."
    return _unity("vfx_read", target=target)


def uy_anim_controller_build(path: str, states: list, parameters: list = None, transitions: list = None) -> dict:
    "Create an Animator Controller from one description: parameters, states with clips or a 1D blend tree, transitions with conditions. Everything named is checked before the asset exists. Reloaded and compared."
    return _unity("anim_controller_build", path=path, states=states, parameters=parameters, transitions=transitions)


def uy_anim_controller_get(target: dict) -> dict:
    "An Animator Controller as data: parameters, states, motions, transitions, conditions. Read only."
    return _unity("anim_controller_get", target=target)


def uy_anim_params_set(target: dict, add: list = None, remove: list = None) -> dict:
    "Add or remove Animator parameters. A duplicate and removing one a transition still reads are refused first. Undoable. Read back."
    return _unity("anim_params_set", target=target, add=add, remove=remove)


def uy_clip_create(path: str, frameRate: float = None, loop: bool = None) -> dict:
    "Create an empty animation clip. Read back."
    return _unity("clip_create", path=path, frameRate=frameRate, loop=loop)


def uy_clip_curves_set(target: dict, curves: list, root: dict = None) -> dict:
    "Set float curves on a clip. Pass root: a curve on a property that does not exist is stored without complaint and animates nothing, and root lets it be checked. Undoable. Read back."
    return _unity("clip_curves_set", target=target, curves=curves, root=root)


def uy_clip_info(target: dict) -> dict:
    "A clip: length, frame rate, loop, every curve binding with its keys, its events. Read only."
    return _unity("clip_info", target=target)


def uy_scene_stats(scene: str = None, top: int = None, allowEmpty: bool = None) -> dict:
    "Counts from walking the loaded scenes: objects, renderers, triangles, materials, lights, cameras, missing scripts, components by type. An empty scene fails unless allowEmpty. Read only."
    return _unity("scene_stats", scene=scene, top=top, allowEmpty=allowEmpty)


def uy_project_report(includePackages: bool = None) -> dict:
    "One project at a glance: Unity version, render pipeline, color space, build target, packages, asset counts, build scenes, compile state. Read only."
    return _unity("project_report", includePackages=includePackages)


def uy_ui_canvas_create(name: str, renderMode: str = None, referenceResolution: list = None, eventSystem: bool = None, parent: dict = None) -> dict:
    "Create a uGUI Canvas with a scaler and a raycaster, and an EventSystem when asked and the scene has none. Refused without the uGUI package. Undoable. Read back."
    return _unity("ui_canvas_create", name=name, renderMode=renderMode, referenceResolution=referenceResolution, eventSystem=eventSystem, parent=parent)


def uy_ui_element_create(kind: str, name: str, parent: dict, anchorMin: list = None, anchorMax: list = None, pivot: list = None, anchoredPosition: list = None, sizeDelta: list = None, text: str = None, color: list = None) -> dict:
    "Create a uGUI Panel, Image, Text or Button under a Canvas, with its rect. A parent outside a Canvas is refused, since it would never draw. Undoable. Read back."
    return _unity("ui_element_create", kind=kind, name=name, parent=parent, anchorMin=anchorMin, anchorMax=anchorMax, pivot=pivot, anchoredPosition=anchoredPosition, sizeDelta=sizeDelta, text=text, color=color)


def uy_ui_layout_report(target: dict, limit: int = None) -> dict:
    "Lay a Canvas out and read every element's rectangle, flagging no size, outside its parent, and overlapping a sibling. No window needed. An empty canvas fails. Read only."
    return _unity("ui_layout_report", target=target, limit=limit)


def uy_uitk_validate(path: str) -> dict:
    "Check a UXML or USS file you wrote: reimport it, catch what the importer says, clone a UXML tree and count what came out. Unity logs a bad file and still hands back an asset, so this is the read-back for text authoring."
    return _unity("uitk_validate", path=path)


def uy_build_target_info() -> dict:
    "The active build target, whether this install can build it, and which standalone and mobile targets it can. Read only."
    return _unity("build_target_info")


def uy_build_scenes_get() -> dict:
    "The scenes in the build list, each with whether it is enabled and whether its file exists. Read only."
    return _unity("build_scenes_get")


def uy_build_scenes_set(scenes: list) -> dict:
    "Replace the build's scene list. A scene with no file is refused first. Read back. A project setting, so scratch mode refuses it."
    return _unity("build_scenes_set", scenes=scenes)


def uy_build_player(output: str, development: bool = None, timeoutSeconds: float = 3600) -> dict:
    "Build the player for the active target from the build list and read BuildReport. ok means Succeeded and the output exists. Holds the editor until it ends. Scratch mode refuses it."
    return _unity("build_player", _timeout=timeoutSeconds, output=output, development=development)


def uy_build_report() -> dict:
    "The last uy_build_player's report, kept on disk so it outlives a reload and a restart. Read only."
    return _unity("build_report")


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


def uy_list_tags_layers() -> dict:
    "The project's tags and named layers. Read only."
    return _unity("list_tags_layers")


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


def uy_instantiate_prefab(asset: str, parent: dict = None, name: str = None, position: list = None) -> dict:
    "Place a prefab asset in the active scene or under a parent as a real instance. Undoable, read back: the link names that asset."
    return _unity("instantiate_prefab", asset=asset, parent=parent, name=name, position=position)


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
    "Write one asset's unsaved changes to disk, such as after uy_set_property on a material. A person's other unsaved assets stay theirs. Read back."
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
    folder = os.path.join(where["project"], "Assets", "_Mif", name)
    os.makedirs(folder, exist_ok=True)
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
        os.makedirs(maps, exist_ok=True)
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
