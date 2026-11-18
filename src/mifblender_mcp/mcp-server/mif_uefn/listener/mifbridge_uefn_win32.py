"""Windows process and window helpers for the UEFN backend - one copy, used on both sides.

The MCP server imports it as mif_uefn.listener.mifbridge_uefn_win32 (process image of a port's owner); the in-UEFN listener
loads the same file by path (the Fortnite client processes a session launched, and closing them). Standard library and
ctypes only: no child process, so nothing here can open a window on the person's desktop.

Why not Get-Process: the prototype listed clients through a hidden PowerShell and had to filter HasExited, because an
exited client's helper stays in the process list while another process holds a handle to it (measured 2026-11-16).
Here a process counts only while GetExitCodeProcess answers STILL_ACTIVE.
"""
import ctypes
import os
import re
from ctypes import wintypes

STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TH32CS_SNAPPROCESS = 0x00000002
WM_CLOSE = 0x0010
# FILETIME counts 100 ns ticks from 1601-01-01; the Unix epoch is this many ticks later.
EPOCH_AS_FILETIME = 116444736000000000


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


def available():
    return os.name == "nt"


def _k32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    k.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    k.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                             ctypes.POINTER(wintypes.DWORD)]
    k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    k.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    return k


def _open(k, pid, access=PROCESS_QUERY_LIMITED_INFORMATION):
    h = k.OpenProcess(access, False, int(pid))
    return h or None


def process_image(pid):
    """The full path of pid's executable, or None when it cannot be read (gone, or no access)."""
    if not available() or not pid:
        return None
    k = _k32()
    h = _open(k, pid)
    if not h:
        return None
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(1024)
        if not k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return None
        return buf.value
    finally:
        k.CloseHandle(h)


def is_alive(pid):
    """True while pid exists and has not exited (an exited process held open by a handle is NOT alive)."""
    if not available() or not pid:
        return False
    k = _k32()
    h = _open(k, pid)
    if not h:
        return False
    try:
        code = wintypes.DWORD(0)
        return bool(k.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == STILL_ACTIVE
    finally:
        k.CloseHandle(h)


def _start_epoch(k, h):
    c, e, kt, ut = (wintypes.FILETIME() for _ in range(4))
    if not k.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(kt), ctypes.byref(ut)):
        return None
    ticks = (c.dwHighDateTime << 32) | c.dwLowDateTime
    return (ticks - EPOCH_AS_FILETIME) / 1e7


def processes(prefix):
    """[(pid, exe name, start time as Unix seconds)] of LIVE processes whose exe name starts with prefix
    (case-blind). An entry whose start time cannot be read gets None."""
    if not available():
        return []
    k = _k32()
    snap = k.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snap or snap == wintypes.HANDLE(-1).value:
        return []
    rows = []
    try:
        e = _PROCESSENTRY32W()
        e.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
        ok = k.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            name = e.szExeFile
            if name.lower().startswith(prefix.lower()):
                pid = int(e.th32ProcessID)
                h = _open(k, pid)
                if h:
                    try:
                        code = wintypes.DWORD(0)
                        if k.GetExitCodeProcess(h, ctypes.byref(code)) and code.value == STILL_ACTIVE:
                            rows.append((pid, name, _start_epoch(k, h)))
                    finally:
                        k.CloseHandle(h)
            ok = k.Process32NextW(snap, ctypes.byref(e))
    finally:
        k.CloseHandle(snap)
    return rows


SERVICE_STATES = {1: "stopped", 2: "start_pending", 3: "stop_pending", 4: "running", 5: "continue_pending",
                  6: "pause_pending", 7: "paused"}


class _SERVICE_STATUS(ctypes.Structure):
    _fields_ = [("dwServiceType", wintypes.DWORD), ("dwCurrentState", wintypes.DWORD),
                ("dwControlsAccepted", wintypes.DWORD), ("dwWin32ExitCode", wintypes.DWORD),
                ("dwServiceSpecificExitCode", wintypes.DWORD), ("dwCheckPoint", wintypes.DWORD),
                ("dwWaitHint", wintypes.DWORD)]


def service_state(name):
    """A Windows service's state ('stopped', 'running', 'stop_pending', ...), 'missing' when it is not installed, or
    None when it cannot be read. Read-only: SC_MANAGER_CONNECT and SERVICE_QUERY_STATUS, which need no elevation."""
    if not available():
        return None
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    adv.OpenSCManagerW.restype = wintypes.HANDLE
    adv.OpenSCManagerW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    adv.OpenServiceW.restype = wintypes.HANDLE
    adv.OpenServiceW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, wintypes.DWORD]
    adv.QueryServiceStatus.argtypes = [wintypes.HANDLE, ctypes.POINTER(_SERVICE_STATUS)]
    adv.CloseServiceHandle.argtypes = [wintypes.HANDLE]
    scm = adv.OpenSCManagerW(None, None, 0x0001)          # SC_MANAGER_CONNECT
    if not scm:
        return None
    try:
        svc = adv.OpenServiceW(scm, name, 0x0004)          # SERVICE_QUERY_STATUS
        if not svc:
            return "missing" if ctypes.get_last_error() == 1060 else None   # ERROR_SERVICE_DOES_NOT_EXIST
        try:
            st = _SERVICE_STATUS()
            if not adv.QueryServiceStatus(svc, ctypes.byref(st)):
                return None
            return SERVICE_STATES.get(st.dwCurrentState, str(st.dwCurrentState))
        finally:
            adv.CloseServiceHandle(svc)
    finally:
        adv.CloseServiceHandle(scm)


# THE ANTI-CHEAT LOCK (Mifsopo, 2026-11-16: "make sure the bridge doesnt close EAC as a lock in the bridge"). A session
# end that terminated Easy Anti-Cheat's helper put EAC error 30005 on his screen. So the ONE process-closing helper this
# backend has refuses, before it acts, any process whose image is an anti-cheat - and any process whose image it cannot
# read, since then it cannot tell. There is no flag, no parameter and no second helper: nothing in the UEFN backend
# closes or kills a process by any other route (this module has no terminate, by design).
ANTI_CHEAT = re.compile(r"(?i)(easyanticheat|anticheat|anti[-_ ]cheat|(^|[-_.])eac([-_.]|$)|_eac|(^|[-_.])be([-_.]|$)|beservice|battleye|"
                        r"^be(daisy|client)|^vgc|^vgk|vanguard|faceit|xigncode|nprotect|gameguard|^xhunter|^ace-|"
                        r"^sguard|^punkbuster|^pnkbstr|^equ8|^mhyprot|^hoyokprotect)")


class AntiCheatLock(Exception):
    """Raised instead of closing a process the lock protects. Carries pid and image."""

    def __init__(self, pid, image, why):
        super().__init__("REFUSED by the anti-cheat lock: pid %s (%s) %s. MifBridge never closes or kills an "
                         "anti-cheat process, by pid or by name, gracefully or forced; it exits on its own when its game "
                         "does." % (pid, image or "image unreadable", why))
        self.pid = pid
        self.image = image


def is_anti_cheat(image):
    """True when an executable name or path is an anti-cheat's (Easy Anti-Cheat, BattlEye, Vanguard, and the like)."""
    return bool(image) and bool(ANTI_CHEAT.search(os.path.basename(str(image)).rsplit(".", 1)[0]))


def guard_close(pid, image=None):
    """The lock itself, called before anything is closed: returns the image, or raises AntiCheatLock."""
    image = image or process_image(pid)
    if not image:
        raise AntiCheatLock(pid, None, "could not be identified, so it cannot be shown not to be an anti-cheat")
    if is_anti_cheat(image):
        raise AntiCheatLock(pid, os.path.basename(image), "is an anti-cheat process")
    return image


def close_windows(pid, image=None):
    """Post WM_CLOSE to every visible top-level window of pid. Returns how many were asked. Raises AntiCheatLock for
    an anti-cheat process, or one whose image cannot be read, before anything is posted."""
    if not available():
        return 0
    guard_close(pid, image)
    user32 = ctypes.WinDLL("user32")
    hits = []
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _lp):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            hits.append(hwnd)
        return True
    user32.EnumWindows(proto(cb), 0)
    for h in hits:
        user32.PostMessageW(h, WM_CLOSE, 0, 0)
    return len(hits)
