"""Child processes the bridge starts: no window, a memory ceiling, and a cap on how many heavy jobs run at once.

WHY (filed from use, 2026-10-18). Background Blender renders and analysis jobs started beside a working editor, several
at a time, used up a 64 GB machine and took it down. What fixed it was a watchdog that ended a job's whole process tree
past a memory limit, and one heavy job at a time - peaks after that were 0.5-1.8 GB. Separately, a background job must
never open a console window on the person's desktop.

THREE THINGS, one module so every spawn path takes them the same way:

  hidden(kwargs)   Popen keyword arguments for a child that opens no window. On Windows that is CREATE_NO_WINDOW: the
                   child gets a console that has no window, and every process it starts inherits that console, so a
                   compiler or a pip under it shows nothing either. DETACHED_PROCESS is dropped when it is there: the
                   two cannot be combined (CreateProcess ignores CREATE_NO_WINDOW beside it), and a detached console
                   program that starts another console program gives that one a window of its own - which is the case
                   Unreal's own ExecProcess moved to CREATE_NO_WINDOW for (WindowsPlatformProcess.cpp, 5.8 :1154).
  slot(kind)       one of MIF_MAX_CHILD_JOBS (default 1) machine-wide slots for a heavy job. The count is kept in files
                   under the temp folder, so the MCP server, the Blender add-on and a second server all share it; a
                   slot whose owner process has gone is freed by the next caller. No slot: the caller refuses, naming
                   what holds them.
  watch(...)       a watchdog thread that reads the job's process TREE's memory every half second and ends the whole
                   tree past the limit (MIF_CHILD_JOB_MEMORY_GB, default half of physical memory), then frees the slot.
                   Memory is private commit on Windows (what ran the machine out) and resident size elsewhere.

KEPT IN STEP: tools/blender-addon/MifBlender/childproc.py is a byte copy of this file - the add-on ships and loads on
its own, without the server - and tools/test_child_processes.py fails when the two differ. Edit this one, copy it there.
Standard library only.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time

IS_WINDOWS = sys.platform.startswith("win")
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if IS_WINDOWS else 0
_DETACHED = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) if IS_WINDOWS else 0
_NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200) if IS_WINDOWS else 0

# where the slots are kept; MIF_CHILD_JOBS_DIR moves it (a test uses its own)
JOBS_DIR = os.environ.get("MIF_CHILD_JOBS_DIR") or os.path.join(tempfile.gettempdir(), "MifBridge", "child_jobs")


def hidden(kwargs=None, new_group=False):
    """`kwargs` (Popen / subprocess.run keyword arguments) with what keeps the child's window off the screen: on Windows
    CREATE_NO_WINDOW merged into creationflags (DETACHED_PROCESS taken out, see the module note), CREATE_NEW_PROCESS_GROUP
    added with new_group; elsewhere `kwargs` unchanged, since no platform but Windows gives a child a window of its own."""
    kw = dict(kwargs or {})
    if IS_WINDOWS:
        flags = (int(kw.get("creationflags") or 0) | CREATE_NO_WINDOW | (_NEW_GROUP if new_group else 0)) & ~_DETACHED
        kw["creationflags"] = flags
    return kw


# ---- processes: alive, the tree, its memory, ending it --------------------------------------------------------------

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _PROCESS_VM_READ = 0x0010
    _PROCESS_TERMINATE = 0x0001
    _STILL_ACTIVE = 259
    _TH32CS_SNAPPROCESS = 0x00000002

    class _PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", wintypes.WCHAR * 260)]

    class _PMC_EX(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    _k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    _psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC_EX), wintypes.DWORD]


def alive(pid):
    """True while process `pid` runs. False for a pid that is not a positive integer."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if IS_WINDOWS:
        h = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        try:
            code = wintypes.DWORD()
            return bool(_k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == _STILL_ACTIVE
        finally:
            _k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _parents():
    """{pid: parent pid} for every process, or {} when the table cannot be read."""
    out = {}
    if IS_WINDOWS:
        snap = _k32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
        if not snap or snap == wintypes.HANDLE(-1).value:
            return out
        try:
            e = _PROCESSENTRY32W()
            e.dwSize = ctypes.sizeof(e)
            ok = _k32.Process32FirstW(snap, ctypes.byref(e))
            while ok:
                out[int(e.th32ProcessID)] = int(e.th32ParentProcessID)
                ok = _k32.Process32NextW(snap, ctypes.byref(e))
        finally:
            _k32.CloseHandle(snap)
        return out
    if os.path.isdir("/proc"):
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open("/proc/%s/stat" % d, "rb") as fh:
                    stat = fh.read().decode("utf-8", "replace")
                out[int(d)] = int(stat.rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
    return out


def tree(pid):
    """`pid` and every process descended from it, parents first. A process whose parent ended is no longer found
    under it - which is why the watchdog reads the tree every half second rather than once."""
    pid = int(pid)
    kids = {}
    for child, parent in _parents().items():
        if child != parent:
            kids.setdefault(parent, []).append(child)
    out, todo, seen = [], [pid], set()
    while todo:
        p = todo.pop(0)
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        todo.extend(kids.get(p, []))
    return out


def memory_bytes(pid):
    """One process's memory in bytes: private commit on Windows, resident size elsewhere. None when unreadable."""
    if IS_WINDOWS:
        h = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION | _PROCESS_VM_READ, False, int(pid))
        if not h:
            h = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return None
        try:
            c = _PMC_EX()
            c.cb = ctypes.sizeof(c)
            if not _psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
                return None
            return int(c.PrivateUsage)
        finally:
            _k32.CloseHandle(h)
    try:
        with open("/proc/%d/status" % int(pid), encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def tree_memory_bytes(pid):
    """(bytes over the whole tree, how many processes were counted)."""
    total, n = 0, 0
    for p in tree(pid):
        b = memory_bytes(p)
        if b is not None:
            total += b
            n += 1
    return total, n


def physical_memory_bytes():
    """Total physical memory in bytes, or None."""
    if IS_WINDOWS:
        m = _MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(m)
        if _k32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return int(m.ullTotalPhys)
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def kill_tree(pid):
    """End `pid` and everything under it, children first, by PID only. Returns the pids it ended."""
    pids = tree(pid)
    ended = []
    for p in reversed(pids):
        if IS_WINDOWS:
            h = _k32.OpenProcess(_PROCESS_TERMINATE, False, p)
            if h:
                try:
                    if _k32.TerminateProcess(h, 1):
                        ended.append(p)
                finally:
                    _k32.CloseHandle(h)
        else:
            try:
                os.kill(p, 9)
                ended.append(p)
            except OSError:
                pass
    return ended


# ---- the limits -----------------------------------------------------------------------------------------------------

def max_jobs():
    """How many heavy jobs may run at once machine-wide: MIF_MAX_CHILD_JOBS, default 1, never below 1."""
    try:
        return max(1, int(os.environ.get("MIF_MAX_CHILD_JOBS", "1")))
    except ValueError:
        return 1


def memory_limit_bytes(gb=None):
    """The per-job memory ceiling in bytes: `gb` when given, else MIF_CHILD_JOB_MEMORY_GB, else half of physical
    memory (8 GB when that cannot be read). 0 or less means no ceiling."""
    ceiling = gb
    if ceiling is None:
        env = os.environ.get("MIF_CHILD_JOB_MEMORY_GB")
        if env:
            try:
                ceiling = float(env)
            except ValueError:
                ceiling = None
            # "inf" or "nan" in the environment is no setting at all, as an unparsable one is
            if ceiling is not None and not math.isfinite(ceiling):
                ceiling = None
    if ceiling is not None:
        # int(inf * 2**30) is an OverflowError and nan > 0 is False (no ceiling): a non-finite ceiling from a caller is
        # refused here rather than either. Every op validates its own first; this is the last line.
        ceiling = float(ceiling)
        if not math.isfinite(ceiling):
            raise ValueError("a memory ceiling of %r GB is not a finite number" % (gb,))
        return int(ceiling * (1 << 30)) if ceiling > 0 else 0
    phys = physical_memory_bytes()
    return int(phys // 2) if phys else 8 << 30


# ---- slots: one heavy job at a time, machine-wide ------------------------------------------------------------------

_LOCK = threading.Lock()


def _registry_lock(timeout=5.0):
    """An exclusive lock file for the slot folder, or None when it could not be had in `timeout` seconds - a lock
    left by a process that died is taken over after 30 seconds."""
    os.makedirs(JOBS_DIR, exist_ok=True)
    path = os.path.join(JOBS_DIR, ".lock")
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return path
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(path) > 30:
                    os.remove(path)
                    continue
            except OSError:
                continue
            if time.monotonic() > deadline:
                return None
            time.sleep(0.05)


def running():
    """The slots held now, as {id, kind, label, owner, pid, started}; a slot whose holder is gone is freed here."""
    rows = []
    try:
        names = os.listdir(JOBS_DIR)
    except OSError:
        return rows
    for name in names:
        if not name.endswith(".json"):
            continue
        path = os.path.join(JOBS_DIR, name)
        try:
            with open(path, encoding="utf-8") as fh:
                row = json.load(fh)
        except (OSError, ValueError):
            continue
        # a slot lives while its job runs - or, before the job has a pid, while the process that took it runs
        holder = row.get("pid") or row.get("owner")
        if not alive(holder):
            try:
                os.remove(path)
            except OSError:
                pass
            continue
        rows.append(row)
    return rows


def acquire(kind, label=""):
    """Take a heavy-job slot: (slot id, None) or (None, the jobs holding every slot)."""
    with _LOCK:
        lock = _registry_lock()
        if lock is None:
            return None, [{"kind": "?", "label": "the slot registry %s is locked by another process" % JOBS_DIR}]
        try:
            held = running()
            if len(held) >= max_jobs():
                return None, held
            sid = "%d_%d_%d" % (os.getpid(), int(time.time() * 1000), threading.get_ident() % 100000)
            row = {"id": sid, "kind": kind, "label": label, "owner": os.getpid(), "pid": None,
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with open(os.path.join(JOBS_DIR, sid + ".json"), "w", encoding="utf-8") as fh:
                json.dump(row, fh)
            return sid, None
        finally:
            try:
                os.remove(lock)
            except OSError:
                pass


def attach(sid, pid):
    """Record the job's pid on its slot, so the slot lives exactly as long as the job."""
    path = os.path.join(JOBS_DIR, "%s.json" % sid)
    try:
        with open(path, encoding="utf-8") as fh:
            row = json.load(fh)
        row["pid"] = int(pid)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(row, fh)
    except (OSError, ValueError):
        pass


def release(sid):
    if sid:
        try:
            os.remove(os.path.join(JOBS_DIR, "%s.json" % sid))
        except OSError:
            pass


def busy_message(kind, held):
    """The refusal when every slot is taken: what holds them and how to allow more."""
    names = ", ".join("%s%s (pid %s, since %s)" % (r.get("kind"), (" " + r["label"]) if r.get("label") else "",
                                                   r.get("pid") or r.get("owner"), r.get("started"))
                      for r in held) or "?"
    return ("%s not started: %d heavy job(s) already run and the limit is %d at a time (MIF_MAX_CHILD_JOBS) - %s. Wait "
            "for it to end, or stop it, then try again." % (kind, len(held), max_jobs(), names))


# ---- the watchdog ----------------------------------------------------------------------------------------------------

class Watch:
    """A job under watch: its peak tree memory, and whether the watchdog ended it."""

    def __init__(self, proc, sid, limit, kind):
        self.proc, self.sid, self.limit, self.kind = proc, sid, limit, kind
        self.peak = 0
        self.killed = None          # {"bytes", "limit", "pids"} once ended past the limit
        self.done = threading.Event()

    def report(self):
        out = {"memoryLimitBytes": self.limit or None, "peakTreeMemoryBytes": self.peak,
               "memoryMeasure": "private commit" if IS_WINDOWS else "resident size"}
        if self.killed:
            out["killedForMemory"] = self.killed
        return out


class PidProc:
    """A process known only by pid (an asyncio child, one started elsewhere) in the shape watch() reads."""

    def __init__(self, pid):
        self.pid = int(pid)

    def poll(self):
        return None if alive(self.pid) else 0


def watch(proc, sid=None, limit=None, kind="job", interval=0.5):
    """Start the watchdog on a started Popen-like object (anything with .pid and .poll()). Ends the tree past `limit`
    bytes (None: memory_limit_bytes(); 0: no ceiling, the peak is still read) and frees slot `sid` when the job ends."""
    lim = memory_limit_bytes() if limit is None else int(limit)
    w = Watch(proc, sid, lim, kind)
    if sid:
        attach(sid, proc.pid)

    def run():
        try:
            while proc.poll() is None:
                used, _n = tree_memory_bytes(proc.pid)
                w.peak = max(w.peak, used)
                if lim > 0 and used > lim:
                    pids = kill_tree(proc.pid)
                    w.killed = {"bytes": used, "limit": lim, "pids": pids,
                                "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                    break
                time.sleep(interval)
        finally:
            release(sid)
            w.done.set()

    threading.Thread(target=run, name="mif-childjob-%s" % kind, daemon=True).start()
    return w


def start(args, kind, label="", limit_gb=None, **popen_kwargs):
    """Start a heavy job: a slot, the child with no window, the watchdog. (Popen, Watch, None) or (None, None, the
    refusal) when every slot is taken or the program could not start (the slot is freed again then)."""
    sid, held = acquire(kind, label)
    if sid is None:
        return None, None, busy_message(kind, held)
    try:
        proc = subprocess.Popen(args, **hidden(popen_kwargs))
    except OSError as exc:
        release(sid)
        return None, None, "%s could not start %s: %s" % (kind, args[0] if isinstance(args, (list, tuple)) else args, exc)
    return proc, watch(proc, sid, None if limit_gb is None else memory_limit_bytes(limit_gb), kind), None


def run(args, kind, label="", limit_gb=None, timeout=None, **popen_kwargs):
    """subprocess.run's shape for a heavy job: (CompletedProcess or None, the watch report, the refusal or None). The
    output is read through pipes (stdout and stderr captured as text unless the caller says otherwise); past `timeout`
    seconds the tree is ended and the report says timedOut."""
    popen_kwargs.setdefault("stdin", subprocess.DEVNULL)
    popen_kwargs.setdefault("stdout", subprocess.PIPE)
    popen_kwargs.setdefault("stderr", subprocess.PIPE)
    popen_kwargs.setdefault("text", True)
    proc, w, refusal = start(args, kind, label, limit_gb, **popen_kwargs)
    if proc is None:
        return None, None, refusal
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_tree(proc.pid)
        out, err = proc.communicate()
    w.done.wait(5)
    rep = w.report()
    if timed_out:
        rep["timedOut"] = True
    return subprocess.CompletedProcess(args, proc.returncode, out, err), rep, None
