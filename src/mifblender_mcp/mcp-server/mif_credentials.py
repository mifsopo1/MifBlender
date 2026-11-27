"""Tokens and listener identity for the editor bridges: one place for what the Unreal, Unity and Blender clients share.

WHY (FOUND 2026-11-22 in a code audit). The Unreal and Unity bridges accepted the published token "dev" whenever
MIF_BRIDGE_TOKEN was unset, so any local process could reach exec_console or run_csharp; the Blender add-on had been
moved off "dev" in 1.4 (MifBlender/token_store.py) and the other two never followed. And every client sent its token to
whatever held the port it was told, so a process that bound the port first - another account's, on a shared machine -
collected the secret and then drove the real editor with it.

THE TOKEN FILE for the editor bridges extends the add-on's rule with a file of its own: MIF_BRIDGE_TOKEN_FILE, else
~/.mifbridge/bridge_token, one line, the token. An editor makes it on its first start (MifBridge.cpp's StartupModule,
the Unity package's Server.cs), with owner-only permissions where the OS has them; clients only read it. A
valid token is at least 16 printable ASCII characters (0x21-0x7E); anything else counts as missing and is replaced. The
Blender file stays separate (~/.mifbridge/blender_token), so a Blender port is never offered the editors' secret.

IDENTIFY BEFORE SENDING. listener_check(port) reads the PID that owns the listening socket from the OS TCP table and
refuses one that does not run as this user. The same user is the line that matters: a process of this user can read the
token file anyway, so naming its executable proves nothing more, while another account's process is exactly the one that
must never be handed the secret. Outside Windows the table is not read here: the check answers unverified and allows.

Standard library only, and no import from server.py, so unity_backend.py and the free MifBlender package can carry it.
"""
import os
import socket
import struct
import sys

LEGACY_TOKEN = "dev"
BRIDGE_TOKEN_ENV = "MIF_BRIDGE_TOKEN"
BRIDGE_TOKEN_FILE_ENV = "MIF_BRIDGE_TOKEN_FILE"
_MIN_LEN = 16
_IS_WIN = sys.platform == "win32"


def valid(tok):
    return isinstance(tok, str) and len(tok) >= _MIN_LEN and all(" " < c <= "~" for c in tok)


def bridge_token_file():
    raw = (os.environ.get(BRIDGE_TOKEN_FILE_ENV) or "").strip()
    if raw:
        return os.path.abspath(os.path.expanduser(raw))
    return os.path.join(os.path.expanduser("~"), ".mifbridge", "bridge_token")


def read_token(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            tok = fh.read().strip()
    except (OSError, UnicodeDecodeError):
        return None
    return tok if valid(tok) else None


def bridge_tokens():
    """(tokens to try in order, source). MIF_BRIDGE_TOKEN when set (an explicit "dev" too: it was set on purpose) and
    nothing else; otherwise the token file's, then "dev" for an editor built before the token file, whose refusal runs
    nothing. Read on every call, because an editor started later makes the file then.

    A CLIENT READS THE FILE AND NEVER MAKES IT. Only an editor built with the token file makes it, so a file that exists
    means such an editor ran; a file a client made would hold a token no editor still on "dev" knows, and every simple
    tool that sends the file's token without a fallback would be locked out of it."""
    env = (os.environ.get(BRIDGE_TOKEN_ENV) or "").strip()
    if env:
        return [env], "environment"
    tok = read_token(bridge_token_file())
    return ([tok, LEGACY_TOKEN], "file") if tok else ([LEGACY_TOKEN], "default")


def bridge_token():
    """The first token bridge_tokens would send, for a child process's environment."""
    return bridge_tokens()[0][0]


# ----------------------------------------------------------------------------------------------------- the listener


def listening_pid(port):
    """The PID listening on TCP <port> (IPv4 then IPv6), or None when nothing listens or the table cannot be read.
    Windows only; elsewhere None."""
    if not _IS_WIN:
        return None
    import ctypes
    from ctypes import wintypes
    try:
        iphlp = ctypes.WinDLL("iphlpapi")
    except OSError:
        return None
    for family, row_size, port_off, pid_off in ((2, 24, 8, 20), (23, 56, 20, 52)):   # AF_INET, AF_INET6 owner-pid rows
        size = wintypes.DWORD(0)
        iphlp.GetExtendedTcpTable(None, ctypes.byref(size), False, family, 3, 0)   # TCP_TABLE_OWNER_PID_LISTENER
        buf = ctypes.create_string_buffer(size.value + 4096)
        if iphlp.GetExtendedTcpTable(buf, ctypes.byref(size), False, family, 3, 0) != 0:
            continue
        count = struct.unpack_from("<I", buf, 0)[0]
        for i in range(count):
            base = 4 + i * row_size
            lport = struct.unpack_from("<I", buf, base + port_off)[0]
            if socket.ntohs(lport & 0xFFFF) == port:
                return struct.unpack_from("<I", buf, base + pid_off)[0]
    return None


def _token_user_sid(handle):
    """The SID bytes of a process or thread token's user, or None."""
    import ctypes
    from ctypes import wintypes
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    adv.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                        ctypes.POINTER(wintypes.DWORD)]
    adv.GetLengthSid.argtypes = [ctypes.c_void_p]
    need = wintypes.DWORD(0)
    adv.GetTokenInformation(handle, 1, None, 0, ctypes.byref(need))   # TokenUser
    if not need.value:
        return None
    buf = ctypes.create_string_buffer(need.value)
    if not adv.GetTokenInformation(handle, 1, buf, need, ctypes.byref(need)):
        return None
    psid = ctypes.c_void_p.from_buffer(buf).value                      # TOKEN_USER.User.Sid
    return ctypes.string_at(psid, adv.GetLengthSid(ctypes.c_void_p(psid)))


def _process_sid(pid):
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    k.OpenProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    adv.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    h = k.OpenProcess(0x1000, False, int(pid))                          # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    try:
        tok = wintypes.HANDLE()
        if not adv.OpenProcessToken(h, 0x0008, ctypes.byref(tok)):     # TOKEN_QUERY
            return None
        try:
            return _token_user_sid(tok)
        finally:
            k.CloseHandle(tok)
    finally:
        k.CloseHandle(h)


def process_image(pid):
    """The full path of pid's executable, or None."""
    if not _IS_WIN or not pid:
        return None
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32")
    k.OpenProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                             ctypes.POINTER(wintypes.DWORD)]
    h = k.OpenProcess(0x1000, False, int(pid))
    if not h:
        return None
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(1024)
        return buf.value if k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)) else None
    finally:
        k.CloseHandle(h)


_OWN_SID = {}


def _own_sid():
    if "sid" not in _OWN_SID:
        try:
            _OWN_SID["sid"] = _process_sid(os.getpid())
        except Exception:  # noqa: BLE001 - an OS call that failed: unknown, which refuses below
            _OWN_SID["sid"] = None
    return _OWN_SID["sid"]


def listener_check(port, host="127.0.0.1"):
    """{"ok", "pid", "image", "verified", "reason"} for the process listening on host:port, before a token goes to it.

    ok False only when the listener is measured and is NOT this user's (or its owner cannot be read, which is what
    another account's process looks like to a normal user). Nothing listening: ok True, pid None - the connection then
    fails on its own and nothing is sent. Not Windows, or a host that is not this machine: verified False, ok True."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        return {"ok": True, "pid": None, "image": None, "verified": False, "reason": "not a loopback host"}
    if not _IS_WIN:
        return {"ok": True, "pid": None, "image": None, "verified": False, "reason": "the TCP table is read on Windows only"}
    try:
        pid = listening_pid(int(port))
    except Exception:  # noqa: BLE001 - a table that cannot be read is unverified, not a refusal
        return {"ok": True, "pid": None, "image": None, "verified": False, "reason": "the TCP table could not be read"}
    if not pid:
        return {"ok": True, "pid": None, "image": None, "verified": True, "reason": "nothing listens there"}
    try:
        theirs, mine = _process_sid(pid), _own_sid()
    except Exception:  # noqa: BLE001
        theirs, mine = None, _own_sid()
    image = None
    try:
        image = process_image(pid)
    except Exception:  # noqa: BLE001
        pass
    if mine is not None and theirs == mine:
        return {"ok": True, "pid": pid, "image": image, "verified": True, "reason": "runs as this user"}
    return {"ok": False, "pid": pid, "image": image, "verified": True,
            "reason": ("PID %s%s listening on port %s does not run as this user%s, so no token was sent to it"
                       % (pid, " (%s)" % os.path.basename(image) if image else "", port,
                          "" if theirs is not None else " (its owner cannot be read from this account)"))}
