"""Launching and closing Roblox Studio for rbx_launch and rbx_close.

THE RULES, each held by a dev test:
- a Studio that is already running is used, never a second one launched;
- a launch is recorded (pid, process start time, exe, arguments) in <home>/launched.json, and only a process matching a
  record - same pid AND same start time, image RobloxStudioBeta.exe - is ever closed; a Studio the person opened is not;
- closing is by that pid only, never by image name.

The command line `-task EditFile -localPlaceFile <file>` is read from the 0.748 executable's own strings (`-task`,
`EditFile`, `-localPlaceFile` are all present in RobloxStudioBeta.exe); that Studio opens the file with it is measured by
the first live rbx_launch, not assumed. The hooks below (PROCESSES, SPAWN, TERMINATE, CLOSE_WINDOWS, MINIMIZE_WINDOWS,
IS_ALIVE, PARENTS) are what
a suite replaces.
"""
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
from mif_roblox import daemon as D  # noqa: E402

STUDIO_IMAGE = "robloxstudiobeta.exe"
START_SLACK = 2.0                   # seconds a recorded start time may differ from the process table's
# An empty place: just the root element, with no guessed property serialization. Studio fills in its default services.
EMPTY_PLACE = '<roblox version="4">\n</roblox>\n'


def _w32():
    from mif_uefn.listener import mifbridge_uefn_win32 as W
    return W


def _processes(prefix):
    return _w32().processes(prefix)


def _terminate(pid):
    """TerminateProcess on pid. Only ever called with a pid this module launched and re-matched."""
    import ctypes
    k = ctypes.windll.kernel32
    k.OpenProcess.restype = ctypes.c_void_p
    h = k.OpenProcess(0x0001, False, int(pid))          # PROCESS_TERMINATE
    if not h:
        return False
    try:
        return bool(k.TerminateProcess(ctypes.c_void_p(h), 1))
    finally:
        k.CloseHandle(ctypes.c_void_p(h))


PROCESSES = _processes
SPAWN = subprocess.Popen
TERMINATE = _terminate
CLOSE_WINDOWS = lambda pid: _w32().close_windows(pid)  # noqa: E731 - a hook a suite replaces
MINIMIZE_WINDOWS = lambda pid: _w32().minimize_windows(pid)  # noqa: E731
IS_ALIVE = lambda pid: _w32().is_alive(pid)  # noqa: E731


def _lower_windows(pid):
    """Put every visible top-level window of pid at the bottom of the z-order without activating it (SetWindowPos
    HWND_BOTTOM, SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE), restoring a minimized one without activation first
    (SW_SHOWNOACTIVATE): it renders, behind the person's windows, and takes no focus. Returns how many. Closes nothing."""
    import ctypes
    from ctypes import wintypes
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
        if user32.IsIconic(h):
            user32.ShowWindow(h, 4)                       # SW_SHOWNOACTIVATE
        user32.SetWindowPos(h, wintypes.HWND(1), 0, 0, 0, 0, 0x0010 | 0x0002 | 0x0001)
    return len(hits)


LOWER_WINDOWS = _lower_windows


def _user32():
    """A user32 of our own with the argument types set: an HWND above 2**31 passed as a bare int overflows (seen live:
    "argument 1: OverflowError: int too long to convert")."""
    import ctypes
    from ctypes import wintypes
    u = ctypes.WinDLL("user32")
    u.GetForegroundWindow.restype = wintypes.HWND
    u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    for name in ("IsWindowVisible", "IsIconic"):
        getattr(u, name).argtypes = [wintypes.HWND]
    u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    u.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    u.EnumChildWindows.argtypes = [wintypes.HWND, ctypes.c_void_p, wintypes.LPARAM]
    return u


def _window_state(pid):
    """What pid's visible top-level windows are doing right now: {"windows": how many, "iconic": any minimized,
    "foreground": one of them is the foreground window}. Reads only (IsWindowVisible, IsIconic, GetForegroundWindow)."""
    import ctypes
    from ctypes import wintypes
    user32 = _user32()
    fg = user32.GetForegroundWindow()
    hits = []
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _lp):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            hits.append(hwnd)
        return True
    user32.EnumWindows(proto(cb), 0)
    return {"windows": len(hits), "iconic": any(bool(user32.IsIconic(h)) for h in hits),
            "foreground": bool(fg) and any(h == fg for h in hits)}


WINDOW_STATE = _window_state


def _capture_window(pid):
    """The pixels of pid's largest visible top-level window, as (width, height, RGBA bytes), read with PrintWindow
    (PW_RENDERFULLCONTENT) into a 32-bit DIB: the window draws itself into our bitmap, so it need not be on top and
    nothing is put on the screen. None when pid has no visible window. A play client's own CaptureService frame cannot be
    turned into pixels there (RB1503), so this is the play-test capture route."""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32")
    wins = []
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _lp):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            rc = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rc))
            wins.append(((rc.right - rc.left) * (rc.bottom - rc.top), hwnd, rc.right - rc.left, rc.bottom - rc.top))
        return True
    user32.EnumWindows(proto(cb), 0)
    if not wins:
        return None
    _area, hwnd, w, h = max(wins, key=lambda x: x[0])
    if w <= 0 or h <= 0:
        return None
    return _print_hwnd(hwnd, w, h)


def _print_hwnd(hwnd, w, h, printer=None):
    """(w, h, RGBA bytes) of one window drawn by itself with PrintWindow (PW_RENDERFULLCONTENT) into a 32-bit DIB: nothing
    goes on the screen and the window need not be on top."""
    import ctypes
    from ctypes import wintypes
    user32, gdi32 = ctypes.WinDLL("user32"), ctypes.WinDLL("gdi32")
    user32.GetDC.restype = wintypes.HDC
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateDIBSection.restype = wintypes.HBITMAP
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    # handles are pointer-sized: without argtypes ctypes passes them as C ints, and a handle past 2**31 overflows
    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p),
                                       wintypes.HANDLE, wintypes.DWORD]
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]

    class BIH(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long), ("biHeight", ctypes.c_long),
                    ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long), ("biYPelsPerMeter", ctypes.c_long),
                    ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]
    screen = user32.GetDC(None)
    mem = gdi32.CreateCompatibleDC(screen)
    bih = BIH(ctypes.sizeof(BIH), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
    bits = ctypes.c_void_p()
    bmp = gdi32.CreateDIBSection(mem, ctypes.byref(bih), 0, ctypes.byref(bits), None, 0)
    old = gdi32.SelectObject(mem, bmp)
    try:
        (printer or user32.PrintWindow)(wintypes.HWND(hwnd), mem, 2)          # PW_RENDERFULLCONTENT
        raw = ctypes.string_at(bits, w * h * 4)
    finally:
        gdi32.SelectObject(mem, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem)
        user32.ReleaseDC(None, screen)
    rgba = bytearray(raw)
    rgba[0::4], rgba[2::4] = raw[2::4], raw[0::4]              # BGRA -> RGBA
    rgba[3::4] = b"\xff" * (w * h)
    return w, h, bytes(rgba)


CAPTURE_WINDOW = _capture_window


def _capture_window_checked(pid):
    """CAPTURE_WINDOW with the proof of whose window it was (found 2026-11-26: with about 12 Studios open, a play
    probe's capture came back as another Studio's window, judged only as a picture). The window is chosen as
    CAPTURE_WINDOW chooses it; its owner is read with GetWindowThreadProcessId before PrintWindow and again after, and
    the windows above it in z-order that overlap it are listed (a capture of a window kept behind others must be its
    own drawing, never what covers it). -> {"image": (w, h, RGBA) or None, "hwnd", "ownerBefore", "ownerAfter",
    "printed": PrintWindow's answer, "rect": [l, t, r, b], "coveredBy": [{"pid", "studio", "rect"}]}, or None when pid
    has no visible window. Reads only."""
    import ctypes
    from ctypes import wintypes
    user32 = _user32()
    user32.EnumWindows.argtypes = [ctypes.c_void_p, wintypes.LPARAM]
    user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    order, mine = [], []
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def owner_of(hwnd):
        o = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(o))
        return o.value

    def cb(hwnd, _lp):
        if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            rc = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rc))
            row = (hwnd, owner_of(hwnd), (rc.left, rc.top, rc.right, rc.bottom))
            order.append(row)                         # EnumWindows walks top-level windows from the top of the z-order
            if row[1] == pid:
                mine.append(row)
        return True
    keep = proto(cb)
    user32.EnumWindows(ctypes.cast(keep, ctypes.c_void_p), 0)
    if not mine:
        return None
    hwnd, _o, rc = max(mine, key=lambda r: (r[2][2] - r[2][0]) * (r[2][3] - r[2][1]))
    w, h = rc[2] - rc[0], rc[3] - rc[1]
    if w <= 0 or h <= 0:
        return None
    studios = {p_ for p_, _s in running()}
    covered = []
    for hw, ow, r in order:
        if hw == hwnd:
            break
        if ow != pid and r[0] < rc[2] and r[2] > rc[0] and r[1] < rc[3] and r[3] > rc[1]:
            covered.append({"pid": ow, "studio": ow in studios, "rect": list(r)})
    before = owner_of(hwnd)
    printed = {"ok": None}

    def printer(hw, mem, flags):
        printed["ok"] = bool(user32.PrintWindow(hw, mem, flags))
        return printed["ok"]
    image = _print_hwnd(hwnd, w, h, printer=printer)
    after = owner_of(hwnd)
    return {"image": image if printed["ok"] else None, "hwnd": int(hwnd), "ownerBefore": before, "ownerAfter": after,
            "printed": printed["ok"], "rect": list(rc), "coveredBy": covered[:8]}


CAPTURE_WINDOW_CHECKED = _capture_window_checked


def test_identity(pid):
    """What a play-test process's command line says it is, measured on 0.748 (2026-11-26): a server carries
    `-task StartServer -instanceId StudioServer`, `-parentPid` its edit Studio; a client `-task StartClient -instanceId
    StudioPlayer_<n>`, `-parentPid` its server; both carry the edit Studio's `-editpid`, one `-edittargetid` and one
    `-playTestSessionGuid`, and `-localProjectFile` the place the test runs. -> {task, instanceId, guid, editPid,
    parentPid, editTarget, projectFile}, or None when the command line cannot be read."""
    try:
        import psutil
        args = psutil.Process(int(pid)).cmdline()
    except Exception:  # noqa: BLE001 - unreadable is "unknown", never a guess
        return None
    flags = {}
    for i, a in enumerate(args[:-1]):
        if a.startswith("-"):
            flags.setdefault(a.lower(), args[i + 1])

    def num(k):
        try:
            return int(flags.get(k))
        except (TypeError, ValueError):
            return None
    return {"task": flags.get("-task"), "instanceId": flags.get("-instanceid"), "guid": flags.get("-playtestsessionguid"),
            "editPid": num("-editpid"), "parentPid": num("-parentpid"), "editTarget": flags.get("-edittargetid"),
            "projectFile": flags.get("-localprojectfile")}


TEST_IDENTITY = test_identity


def _child_rects(pid):
    """Every visible child window of pid's largest visible top-level window (the one CAPTURE_WINDOW reads), as
    {"x", "y", "w", "h", "class"} relative to that window's top-left corner, so a box found here is a box of the capture.
    Reads only."""
    import ctypes
    from ctypes import wintypes
    user32 = _user32()
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    tops = []

    def cb(hwnd, _lp):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            rc = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rc))
            tops.append(((rc.right - rc.left) * (rc.bottom - rc.top), hwnd, rc.left, rc.top))
        return True
    user32.EnumWindows(proto(cb), 0)
    if not tops:
        return []
    _a, top, ox, oy = max(tops, key=lambda x: x[0])
    out = []

    def kb(hwnd, _lp):
        if user32.IsWindowVisible(hwnd):
            rc = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rc))
            name = ctypes.create_unicode_buffer(128)
            user32.GetClassNameW(hwnd, name, 128)
            out.append({"x": rc.left - ox, "y": rc.top - oy, "w": rc.right - rc.left, "h": rc.bottom - rc.top,
                        "class": name.value})
        return True
    cb_kids = proto(kb)
    user32.EnumChildWindows(top, ctypes.cast(cb_kids, ctypes.c_void_p), 0)
    return out


CHILD_RECTS = _child_rects
PARENTS = lambda: _w32().parent_pids("RobloxStudio")  # noqa: E731 - {pid: parent pid} of Studio processes


def record_path():
    return os.path.join(D.home(), "launched.json")


def running():
    """[(pid, start)] of live Roblox Studio processes, whoever started them."""
    return [(pid, start) for pid, name, start in PROCESSES("RobloxStudio") if str(name).lower() == STUDIO_IMAGE]


# Studio counts its running processes at start (play-test servers and clients included) and, above a limit, stops
# before loading anything to show an instance-count warning that waits for a click (0.748, RB1410: an edit launch
# counting 12 sat on it; launches counting 10 or fewer never did; 11 is unmeasured, so it is refused too).
STUDIO_INSTANCE_LIMIT = 11


def instance_limit():
    """None when one more Studio may start, else the refusal (nothing is launched)."""
    pids = sorted(p for p, _s in running())
    n = len(pids)
    if n + 1 < STUDIO_INSTANCE_LIMIT:
        return None
    return {"ok": False, "ran": False, "refusedRule": "studio-instance-limit", "running": n, "pids": pids,
            "error": "%d Roblox Studio processes are running (play tests count), pids %s; one more would count %d, and "
                     "Studio opens a warning that waits for a click at that count (RB1410). Close a Studio, or end a play "
                     "test, first" % (n, ", ".join(str(p) for p in pids), n + 1)}


# A place file with no TextChatService item opens with ChatVersion LegacyChatService (RB1191), and no script can change
# ChatVersion (its write is RobloxScriptSecurity, RB1034), so the scratch place names it: Enum.ChatVersion.TextChatService
# is 1 in the API dump.
TEXT_CHAT_ITEM = ('<Item class="TextChatService" referent="RBX3"><Properties><string name="Name">TextChatService</string>'
                  '<token name="ChatVersion">1</token></Properties></Item>\n')


def port_place(port=None):
    """An empty place MifBridge makes: ServerStorage.MifBridgeLaunched (a BoolValue, true), the marker that lets the
    agent raise the safety mode here under the person's policy (plugin 0.11.0), and with a port other than the default,
    ServerStorage.MifBridgePort = port, which the plugin (0.10.0) reads to pick its daemon; and TextChatService with
    ChatVersion TextChatService, so the place opens on the chat a new place has. Only the root, two services and their
    values: Studio fills in everything else."""
    port_item = ('<Item class="IntValue" referent="RBX2"><Properties><string name="Name">MifBridgePort</string>'
                 '<int64 name="Value">%d</int64></Properties></Item>\n' % int(port)) if port is not None and int(port) != D.DEFAULT_PORT else ""
    return ('<roblox version="4">\n'
            '<Item class="ServerStorage" referent="RBX0"><Properties><string name="Name">ServerStorage</string></Properties>\n'
            '<Item class="BoolValue" referent="RBX1"><Properties><string name="Name">MifBridgeLaunched</string>'
            '<bool name="Value">true</bool></Properties></Item>\n'
            + port_item +
            '</Item>\n' + TEXT_CHAT_ITEM + '</roblox>\n')


PORT_MARKER_REF = "RBXMIFBRIDGEPORT"


def _port_marker(text):
    """(referent, value) of ServerStorage's direct IntValue child named MifBridgePort in an rbxlx text, (None, None) when
    there is none; raises ValueError when the file is not a place XML."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError("not a readable place XML: %s" % exc) from exc
    if root.tag != "roblox":
        raise ValueError("not a place XML (its root is <%s>)" % root.tag)
    for ss in root.findall("Item"):
        if ss.get("class") != "ServerStorage":
            continue
        for it in ss.findall("Item"):
            props = it.find("Properties")
            name = props.find("string[@name='Name']") if props is not None else None
            if it.get("class") == "IntValue" and name is not None and name.text == "MifBridgePort":
                val = props.find("int64[@name='Value']")
                try:
                    return it.get("referent"), int(val.text) if val is not None else None
                except (TypeError, ValueError):
                    return it.get("referent"), None
    return None, None


LAUNCH_ITEM = ('<Item class="BoolValue" referent="RBXMIFBRIDGELAUNCHED"><Properties><string name="Name">'
               'MifBridgeLaunched</string><bool name="Value">true</bool></Properties></Item>')


def _launched_text(text):
    """True when an rbxlx text's ServerStorage has a BoolValue MifBridgeLaunched holding true."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(text)
    for ss in root.findall("Item"):
        if ss.get("class") != "ServerStorage":
            continue
        for it in ss.findall("Item"):
            props = it.find("Properties")
            name = props.find("string[@name='Name']") if props is not None else None
            if it.get("class") == "BoolValue" and name is not None and name.text == "MifBridgeLaunched":
                val = props.find("bool[@name='Value']")
                return val is not None and (val.text or "").strip() == "true"
    return False


def mark_launched_text(text):
    """(text, how) with ServerStorage.MifBridgeLaunched = true in an rbxlx place, the marker that lets the agent raise the
    safety mode under the person's policy (plugin 0.11.0). Mifsopo, 2026-11-22: rbx_launch's copies count as
    bridge-launched; an original opened with open_original and a cloud place never get it. ValueError when the text is
    not a place XML or the result does not read back."""
    _port_marker(text)                       # raises ValueError when it is not a place XML
    if _launched_text(text):
        return text, "the copy already carries MifBridgeLaunched = true"
    import re as _re
    m = _re.search(r'<Item class="BoolValue" referent="([^"]*)">\s*<Properties>\s*<string name="Name">MifBridgeLaunched'
                   r'</string>.*?</Item>', text, _re.S)
    if m:
        body = _re.sub(r'<bool name="Value">\s*false\s*</bool>', '<bool name="Value">true</bool>', m.group(0), count=1)
        out, how = text[:m.start()] + body + text[m.end():], "MifBridgeLaunched set true in the copy"
    else:
        m = _re.search(r'<Item class="ServerStorage" referent="[^"]*">\s*<Properties>.*?</Properties>', text, _re.S)
        if m:
            out = text[:m.end()] + "\n" + LAUNCH_ITEM + text[m.end():]
        else:
            end = text.rindex("</roblox>")
            out = (text[:end] + '<Item class="ServerStorage" referent="RBXMIFBRIDGESS"><Properties><string name="Name">'
                   'ServerStorage</string></Properties>\n' + LAUNCH_ITEM + '\n</Item>\n' + text[end:])
        how = "MifBridgeLaunched = true written into the copy's ServerStorage"
    if not _launched_text(out):
        raise ValueError("the copy does not read back ServerStorage.MifBridgeLaunched = true")
    return out, how


def _launched_bytes(data, place_file):
    """(bytes, how) of a copy's content with the bridge-launched marker, for an .rbxlx or a binary .rbxl."""
    if place_file.lower().endswith(".rbxlx"):
        text, how = mark_launched_text(data.decode("utf-8"))
        return text.encode("utf-8"), how
    from mif_roblox import rbxl as RB
    return RB.mark_launched(data)


def _own_copy_launched(path):
    """A copy MifBridge made earlier (places/<port>/copies) opened again as itself gets the marker in place: the file is
    MifBridge's, and a copy made before 2026-11-22 lacks it. None, or a refusal."""
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        out, how = _launched_bytes(data, path)
    except Exception as exc:  # noqa: BLE001 - every unreadable copy is one refusal that says why
        return {"ok": False, "ran": False, "refusedRule": "no-launch-marker",
                "error": "%s could not be marked bridge-launched (%s): nothing was launched" % (path, exc)}
    if out != data:
        with open(path, "wb") as fh:
            fh.write(out)
    return None


def is_copy_place(path):
    """True for a copy rbx_launch made of a person's place (places/<port>/copies/): MifBridge's file, named as theirs."""
    return bool(path) and in_places_folder(path) and os.path.basename(os.path.dirname(os.path.abspath(path))) == "copies"


def _plain_copy(place_file, port, fresh=False):
    """A byte-for-byte copy of a person's place in places/<port>/copies/ (the default port needs no marker)."""
    folder = os.path.join(D.home(), "places", str(port), "copies")
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, os.path.basename(place_file))
    try:
        with open(place_file, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        return {"ok": False, "ran": False, "refusedRule": "copy-failed",
                "error": "%s could not be read to copy it (%s): nothing was launched" % (place_file, exc)}
    try:
        data, launched_how = _launched_bytes(data, place_file)
    except Exception as exc:  # noqa: BLE001 - a copy that cannot carry the marker is refused, never opened unmarked
        return {"ok": False, "ran": False, "refusedRule": "no-launch-marker",
                "error": "%s could not be copied with ServerStorage.MifBridgeLaunched (%s): nothing was launched"
                         % (place_file, exc)}
    held = _copy_held(dest, data, place_file, fresh)
    if held:
        return held
    try:
        with open(dest, "wb") as fh:
            fh.write(data)
    except OSError as exc:
        return {"ok": False, "ran": False, "refusedRule": "copy-failed",
                "error": "%s could not be copied into %s (%s): nothing was launched, and the person's file is never "
                         "opened in its place" % (place_file, folder, exc)}
    return {"place": dest, "copied": True, "original": place_file, "launched": launched_how,
            "marker": "no port marker needed here; opened as a copy so no save lands on the person's file "
                      "(open_original true opens the file itself)"}


def marked_place(place_file, port, open_original=False, fresh=False):
    """The place a lane Studio on `port` may open for place_file. A Studio reaches a lane daemon only through
    ServerStorage.MifBridgePort; a place without it connects to the default port's daemon, the person's (found 2026-11-20:
    a copied place did exactly that). So:
    - on the default port, or when the file already carries MifBridgePort = port, the file itself (nothing written);
    - an .rbxlx without it (or with another port): a COPY in places/<port>/copies/ with the marker written into its
      ServerStorage - the person's file is never written;
    - an .rbxl (binary) without it: a binary COPY there with the marker added by mif_roblox.rbxl; a
      binary place it cannot mark is refused, saying why and how to open it.
    A person's file (anything outside MifBridge's places folder) is never the file a Studio opens unless open_original
    is true: on the default port it opens as a plain copy in places/<port>/copies/ too, so no later save (rbx_place_save,
    a person's Ctrl+S in that Studio) can land on it. Found 2026-11-22: rbx_launch(place_file=<the person's .rbxl>) on
    8650 opened the file itself ("default port: no marker needed"), and a save from that Studio would have written it.
    A copy already there that differs from a fresh one is never written over (refusedRule copy-exists): it is what the
    lane's Studio had open, so it may hold saved work. fresh true replaces it.
    Returns {"place", "copied", "marker"} or a refusal."""
    port = int(port)
    own = in_places_folder(place_file)
    if is_copy_place(place_file):
        bad = _own_copy_launched(place_file)
        if bad:
            return bad
    if port == D.DEFAULT_PORT:
        if own or open_original:
            return {"place": place_file, "copied": False, "marker": "default port: no marker needed",
                    "original": None if own else place_file, "openedOriginal": not own}
        return _plain_copy(place_file, port, fresh)
    if not place_file.lower().endswith(".rbxlx"):
        return _marked_binary(place_file, port, open_original, fresh)
    try:
        # newline="": a CRLF place keeps its CRs in the copy, so its scripts hash as the original's (found 2026-11-22)
        with open(place_file, encoding="utf-8", newline="") as fh:
            text = fh.read()
        ref, have = _port_marker(text)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return {"ok": False, "ran": False, "refusedRule": "no-port-marker",
                "error": "%s could not be read to check its port marker (%s): nothing was launched" % (place_file, exc)}
    if have == port and (own or open_original):
        return {"place": place_file, "copied": False, "marker": "the file already carries MifBridgePort = %d" % port}
    if have == port:
        return _plain_copy(place_file, port, fresh)
    item = ('<Item class="IntValue" referent="%s"><Properties><string name="Name">MifBridgePort</string>'
            '<int64 name="Value">%d</int64></Properties></Item>' % (PORT_MARKER_REF, port))
    if ref is not None:
        # the file's own marker names another port: rewrite its Value inside that one item (found by referent)
        m = re.search(r'<Item class="IntValue" referent="%s">.*?</Item>' % re.escape(ref), text, re.S)
        if not m or not re.search(r'<int64 name="Value">-?\d+</int64>', m.group(0)):
            return {"ok": False, "ran": False, "refusedRule": "no-port-marker",
                    "error": "%s has a MifBridgePort the bridge could not rewrite: nothing was launched" % place_file}
        body = re.sub(r'<int64 name="Value">-?\d+</int64>', '<int64 name="Value">%d</int64>' % port, m.group(0), count=1)
        out = text[:m.start()] + body + text[m.end():]
        how = "MifBridgePort %s rewritten to %d in the copy" % (have, port)
    else:
        m = re.search(r'<Item class="ServerStorage" referent="[^"]*">\s*<Properties>.*?</Properties>', text, re.S)
        if m:
            out = text[:m.end()] + "\n" + item + text[m.end():]
        else:
            end = text.rindex("</roblox>")
            out = (text[:end] + '<Item class="ServerStorage" referent="RBXMIFBRIDGESS"><Properties><string name="Name">'
                   'ServerStorage</string></Properties>\n' + item + '\n</Item>\n' + text[end:])
        how = "MifBridgePort = %d written into the copy's ServerStorage" % port
    try:
        out, launched_how = mark_launched_text(out)
        if _port_marker(out)[1] != port:
            raise ValueError("the copy does not read back MifBridgePort = %d" % port)
    except ValueError as exc:
        return {"ok": False, "ran": False, "refusedRule": "no-port-marker",
                "error": "%s: the marked copy failed its read-back (%s); nothing was launched" % (place_file, exc)}
    folder = os.path.join(D.home(), "places", str(port), "copies")
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, os.path.basename(place_file))
    if os.path.normcase(os.path.abspath(dest)) == os.path.normcase(os.path.abspath(place_file)):
        return {"ok": False, "ran": False, "refusedRule": "no-port-marker",
                "error": "%s is already the bridge's copy and still has no marker for port %d: nothing was launched"
                         % (place_file, port)}
    held = _copy_held(dest, out.encode("utf-8"), place_file, fresh)
    if held:
        return held
    with open(dest, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    return {"place": dest, "copied": True, "original": place_file, "marker": how, "launched": launched_how}


def _copy_held(dest, data, place_file, fresh):
    """None when the marked copy may be written at dest: nothing there, the same bytes there (written again, unchanged),
    or fresh true. Else the refusal copy-exists. The copy is the file the lane's Studio has open, so a save in that
    Studio lands in it; writing a fresh copy over it lost that work (found 2026-11-22), and two sources with one file
    name from different folders collided the same way."""
    if fresh is True or not os.path.isfile(dest):
        return None
    try:
        with open(dest, "rb") as fh:
            have = fh.read()
    except OSError as exc:
        return {"ok": False, "ran": False, "refusedRule": "copy-exists", "copy": dest,
                "error": "a copy already exists at %s and could not be read (%s): nothing was launched" % (dest, exc)}
    # text copies are written with the platform's newline, so compare without CRs
    if have.replace(b"\r\n", b"\n") == data.replace(b"\r\n", b"\n"):
        return None
    stamp = lambda p: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))  # noqa: E731
    return {"ok": False, "ran": False, "refusedRule": "copy-exists", "copy": dest, "copyModified": stamp(dest),
            "original": place_file, "originalModified": stamp(place_file),
            "error": "a marked copy already exists at %s (modified %s) and differs from a fresh copy of %s (modified %s): "
                     "it may hold work saved in a lane Studio, so it was not replaced and nothing was launched. Open the "
                     "copy itself (place_file the copy: it already carries this port's marker), or pass fresh true to "
                     "replace it with a new copy of the original" % (dest, stamp(dest), place_file, stamp(place_file))}


def _marked_binary(place_file, port, open_original=False, fresh=False):
    """marked_place for a binary .rbxl: the file itself when it carries MifBridgePort = port (and is MifBridge's own, or
    open_original), else a binary copy in places/<port>/copies/ with the marker added (or the file's own rewritten),
    read back before it is used."""
    from mif_roblox import rbxl as RB
    try:
        with open(place_file, "rb") as fh:
            blob = fh.read()
        if RB.read_marker(blob) == port and not (in_places_folder(place_file) or open_original):
            return _plain_copy(place_file, port, fresh)
        if RB.read_marker(blob) == port:
            return {"place": place_file, "copied": False, "binary": True,
                    "marker": "the file already carries MifBridgePort = %d" % port}
        out, how = RB.mark(blob, port)
        out, launched_how = RB.mark_launched(out)
    except (OSError, RB.BinaryPlaceError) as exc:
        return {"ok": False, "ran": False, "refusedRule": "no-port-marker",
                "error": "%s is a binary place the bridge could not mark with ServerStorage.MifBridgePort = %d (%s); without "
                         "it this Studio would connect to the default port's daemon (the person's). Nothing was launched. "
                         "Save it as .rbxlx, or add an IntValue MifBridgePort = %d under ServerStorage and save, then launch "
                         "again" % (place_file, port, exc, port)}
    folder = os.path.join(D.home(), "places", str(port), "copies")
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, os.path.basename(place_file))
    if os.path.normcase(os.path.abspath(dest)) == os.path.normcase(os.path.abspath(place_file)):
        return {"ok": False, "ran": False, "refusedRule": "no-port-marker",
                "error": "%s is already the bridge's copy and still has no marker for port %d: nothing was launched"
                         % (place_file, port)}
    held = _copy_held(dest, out, place_file, fresh)
    if held:
        return held
    with open(dest, "wb") as fh:
        fh.write(out)
    with open(dest, "rb") as fh:
        back = RB.read_marker(fh.read())
    if back != port:
        return {"ok": False, "ran": False, "refusedRule": "no-port-marker",
                "error": "%s: the marked binary copy read back MifBridgePort = %s, not %d; nothing was launched"
                         % (place_file, back, port)}
    return {"place": dest, "copied": True, "binary": True, "original": place_file, "marker": how,
            "launched": launched_how}


def scratch_place(port=None):
    """The empty place file rbx_launch opens when given none: ours, so a save there never touches a person's place.
    With a port other than the default, a place of its own in places/<port>/, carrying that port for the plugin. The file
    keeps the name MifScratch.rbxlx, which the plugin treats as all-scratch."""
    folder = os.path.join(D.home(), "places")
    if port is not None and int(port) != D.DEFAULT_PORT:
        folder = os.path.join(folder, str(int(port)))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "MifScratch.rbxlx")
    want = port_place(port)
    if not os.path.isfile(path) or open(path, encoding="utf-8").read() != want:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(want)
    return path


def in_places_folder(path):
    """True for a file under MifBridge's own places folder: a scratch place, or a marked copy of a person's place
    (places/<port>/copies/), which holds the person's content."""
    if not path:
        return False
    folder = os.path.normcase(os.path.abspath(os.path.join(D.home(), "places")))
    return os.path.normcase(os.path.abspath(path)).startswith(folder + os.sep)


def is_scratch_place(path):
    """True for MifBridge's own scratch place: under its places folder AND named MifScratch (the name the plugin treats
    as all-scratch). Nothing of the person's is in it. A marked copy in places/<port>/copies is NOT scratch: it is a copy
    of the person's place, and closing it without saving needs discard (found 2026-11-22: it counted as scratch, so
    rbx_close ended an hour of work in a copy without asking)."""
    return in_places_folder(path) and place_stem(path).lower() == SCRATCH_STEM.lower() \
        and os.path.basename(os.path.dirname(os.path.abspath(path))).lower() != "copies"


def _read_records():
    try:
        with open(record_path(), encoding="utf-8") as fh:
            rows = json.load(fh)
        return rows if isinstance(rows, list) else []
    except (OSError, ValueError):
        return []


def _write_records(rows):
    os.makedirs(os.path.dirname(record_path()), exist_ok=True)
    tmp = record_path() + ".tmp"
    from mif_roblox import durable
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
        durable.sync(fh)
    os.replace(tmp, record_path())


RECORD_LOCK_WAIT = 10.0     # seconds a _RecordLock waits for another process's hold before RecordsBusy


class RecordsBusy(OSError):
    """The launch-record lock (or another _RecordLock file) stayed held by another process for all of its wait. An OSError,
    so a caller that caught OSError before still does; rbx_launch and rbx_close answer it as refusedRule records-busy.
    spawnedPid / studioPids are set when a launch had already started Studio before the lock was refused."""

    def __init__(self, path, waited):
        super().__init__("the record lock %s stayed held by another process for %g s" % (path, waited))
        self.path, self.waited = path, waited
        self.spawnedPid, self.studioPids = None, []


class _RecordLock:
    """Several lanes share launched.json: every read-modify-write holds this lock (a byte lock on launched.lock). path
    names another lock file for another shared record (the coverage record, domain_release). The OS drops the lock when
    its process dies, so a killed holder never leaves it stuck."""

    def __init__(self, path=None, wait=None):
        self.path, self.wait = path, RECORD_LOCK_WAIT if wait is None else wait

    def __enter__(self):
        path = self.path or record_path()[:-5] + ".lock"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, "a+b")
        deadline = time.time() + self.wait
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    self.fh.seek(0)
                    msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.time() > deadline:
                    self.fh.close()
                    raise RecordsBusy(path, self.wait) from None
                time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        try:
            if os.name == "nt":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self.fh.close()
        return False


RecordLock = _RecordLock    # public: the coverage record (domain_release) holds the same kind of lock


PENDING_SECONDS = 60.0  # a launch's provisional record holds this long, or until its Studio pid is found


def launch(exe, place, settle=8.0, port=None):
    """Start Studio on place and record every Studio process that started after the call (Studio may hand off to a
    second process; the record holds what the table shows). A provisional record is written BEFORE the spawn, so from
    the first instant every lane's play-probe cleanup knows a Studio is starting here and leaves it alone."""
    before = {pid for pid, _s in running()}
    t0 = time.time()
    port = int(port) if port is not None else D.DEFAULT_PORT
    token = "%d-%d" % (os.getpid(), int(t0 * 1000))
    with _RecordLock():
        _write_records(_read_records() + [{"pending": token, "t0": t0, "port": port, "place": place,
                                           "lanePid": os.getpid(), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}])
    args = [exe, "-task", "EditFile", "-localPlaceFile", place]
    try:
        # window: Roblox Studio is a GUI application the person asked to have launched; its window is the point.
        proc = SPAWN(args, creationflags=getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                     close_fds=True)
    except OSError:
        with _RecordLock():
            _write_records([r for r in _read_records() if r.get("pending") != token])
        raise
    # Ownership is measured, never inferred from timing (found 2026-11-22: every Studio that appeared in this window was
    # recorded, so another lane's launch, the person's own Studio or a play-test server could be closed as this lane's).
    # A new process is this launch's only when its command line holds this place and it is not a play-test window.
    found, skipped, unread = [], {}, set()
    deadline = time.time() + settle
    while time.time() < deadline:
        for pid, start in running():
            if pid in before or not (start is None or start >= t0 - START_SLACK) or pid in skipped:
                continue
            why = _launch_mismatch(pid, place)
            if why is None:
                found.append((pid, start))
            elif why != UNREAD:
                skipped[pid] = why               # a definite mismatch; an unreadable one is read again next round
            else:
                unread.add(pid)
        if found:
            break
        time.sleep(0.25)
    unread -= {p for p, _s in found} | set(skipped)
    rows = [{"pid": pid, "start": start, "spawnedPid": getattr(proc, "pid", None), "exe": exe, "args": args[1:],
             "place": place, "port": port, "at": time.strftime("%Y-%m-%dT%H:%M:%S")} for pid, start in found]
    try:
        with _RecordLock():
            rest = [r for r in _read_records() if r.get("pending") != token]
            # no pid seen yet: keep the provisional record until it expires, so the Studio is still never taken for a test
            # window
            _write_records(rest + (rows or [dict(r) for r in _read_records() if r.get("pending") == token]))
    except RecordsBusy as exc:
        exc.spawnedPid, exc.studioPids = getattr(proc, "pid", None), [r["pid"] for r in rows]
        raise
    out = {"spawnedPid": getattr(proc, "pid", None), "studioPids": [r["pid"] for r in rows], "args": args[1:],
           "place": place, "t0": t0}
    if skipped or unread:
        out["notTaken"] = ([{"pid": p, "why": w} for p, w in sorted(skipped.items())]
                           + [{"pid": p, "why": UNREAD + ": not recorded (ownership is read from it)"}
                              for p in sorted(unread)])
    return out


UNREAD = "its command line could not be read"


def _launch_mismatch(pid, place):
    """None when the new Studio process pid is this launch's: its command line holds place and it is no play-test
    window. Else why not (UNREAD when the command line cannot be read: then it is not recorded)."""
    args = CMD_ARGS(pid)
    if args is None:
        return UNREAD
    cmd = " ".join(str(a) for a in args)
    if "-playTestSessionGuid" in cmd:
        return "a play-test window (-playTestSessionGuid), not an edit Studio"
    want = os.path.normcase(os.path.abspath(place)) if place else None
    if want and want not in os.path.normcase(cmd.replace("/", "\\")):
        return "its command line does not hold this launch's place file (another Studio started in the window)"
    return None


def command_line(pid):
    """The process's command line as one string, or None when it cannot be read (psutil missing, access denied)."""
    try:
        import psutil
        return " ".join(psutil.Process(int(pid)).cmdline())
    except Exception:  # noqa: BLE001 - unreadable is "unknown", never a guess
        return None


def confirm_pid(candidates, place):
    """The first candidate that is a running Studio (image RobloxStudioBeta.exe) whose command line holds place. Returns
    (pid, start, checks): checks lists each candidate looked at and why it was or was not taken, so a null pid says why."""
    live = dict(running())
    checks, seen = [], set()
    want = os.path.normcase(os.path.abspath(place)) if place else None
    for pid in candidates:
        if not pid or pid in seen:
            continue
        seen.add(pid)
        if pid not in live:
            checks.append({"pid": pid, "taken": False, "why": "not a running RobloxStudioBeta.exe"})
            continue
        cmd = command_line(pid)
        if cmd is None:
            checks.append({"pid": pid, "taken": False, "why": "its command line could not be read"})
            continue
        if want and want not in os.path.normcase(cmd.replace("/", "\\")):
            checks.append({"pid": pid, "taken": False, "why": "its command line does not hold the place file"})
            continue
        checks.append({"pid": pid, "taken": True, "why": "a running Studio whose command line holds the place file"})
        return pid, live[pid], checks
    return None, None, checks


def adopt(pid, start, exe, args, place, port):
    """Record a Studio found after launch()'s settle window (its provisional record would expire): rbx_close and every
    lane's play-probe cleanup then know it as launched. Never records a pid twice."""
    with _RecordLock():
        rows = _read_records()
        if any(r.get("pid") == pid for r in rows):
            return False
        rows.append({"pid": pid, "start": start, "spawnedPid": None, "exe": exe, "args": args, "place": place,
                     "port": int(port), "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "adopted": True})
        _write_records(rows)
    return True


def is_test_window(pid):
    """True / False when the command line can be read: Studio's play-test processes carry `-instanceId StudioServer` or
    `StudioPlayer_<n>` and `-playTestSessionGuid <guid>` (measured on 0.748); an edit Studio carries -localPlaceFile.
    None when it cannot be read (psutil missing, access denied): then the records alone decide."""
    try:
        import psutil
        cmd = " ".join(psutil.Process(int(pid)).cmdline())
    except Exception:  # noqa: BLE001 - unreadable is "unknown", never a guess
        return None
    return "-playTestSessionGuid" in cmd


def edit_pid(pid):
    """The edit Studio a play-test process belongs to, from its own command line: Studio 0.748 starts every test server
    and client with `-editpid <edit Studio pid>`, and it stays true after the
    server that spawned a client has exited. None when there is no such flag or the command line cannot be read."""
    try:
        import psutil
        args = psutil.Process(int(pid)).cmdline()
    except Exception:  # noqa: BLE001 - unreadable is "unknown", never a guess
        return None
    for i, a in enumerate(args[:-1]):
        if a.lower() == "-editpid":
            try:
                return int(args[i + 1])
            except ValueError:
                return None
    return None


EDIT_PID = edit_pid


def is_test_client(pid):
    """True when pid is a play-test client (-task StartClient), False otherwise or unreadable."""
    try:
        import psutil
        return "StartClient" in psutil.Process(int(pid)).cmdline()
    except Exception:  # noqa: BLE001
        return False


def belongs_to(pid, root):
    """True when the play-test process pid belongs to the edit Studio root: by its own `-editpid` when it has one (an
    orphaned client whose server already exited included), else by its parent chain (descends_from)."""
    if not root or not pid or pid == root:
        return False
    e = EDIT_PID(pid)
    if e is not None:
        return e == int(root)
    return descends_from(pid, root)


def descends_from(pid, root, hops=4):
    """True when pid is a play-test process of the edit Studio `root`: a chain of live Studio processes, each started no
    earlier than its parent, leads from pid up to root. Measured on 0.748: a multiplayer test's server is a child of the
    edit Studio that started it and each client a child of that server, so another Studio's test windows never lead to
    root. A dead or reused parent pid ends the walk with False."""
    if not root or not pid or pid == root:
        return False
    par, starts = PARENTS(), dict(running())
    cur = pid
    for _ in range(hops):
        pp = par.get(cur)
        if pp is None or pp not in starts:
            return False
        sc, sp = starts.get(cur), starts.get(pp)
        if sc is not None and sp is not None and sp > sc + START_SLACK:
            return False                     # the parent pid was reused by a later process: not the real parent
        if pp == root:
            return True
        cur = pp
    return False


def protected(now=None):
    """What no lane's cleanup may end: the pids of recorded launches, and the windows [t0 - slack, t0 + PENDING_SECONDS]
    of launches still starting. (pids, windows)."""
    now = time.time() if now is None else now
    pids, windows = set(), []
    for r in _read_records():
        if r.get("pending"):
            if now - r.get("t0", 0) <= PENDING_SECONDS:
                windows.append((r["t0"] - START_SLACK, r["t0"] + PENDING_SECONDS))
        elif r.get("pid"):
            pids.add(r["pid"])
    return pids, windows


def gone_path():
    return os.path.join(D.home(), "launched_gone.jsonl")


def _note_gone(rows):
    """Append the records ours() dropped, with why and which process dropped them, so a later not-ours answer can say
    what became of a launch record. Kept to the newest GONE_KEPT rows. Never raises."""
    if not rows:
        return
    try:
        path = gone_path()
        old = []
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                old = fh.read().splitlines()[-(GONE_KEPT - len(rows)):]
        from mif_roblox import durable
        with open(path + ".tmp", "w", encoding="utf-8") as fh:
            fh.write("\n".join(old + [json.dumps(r) for r in rows]) + "\n")
            durable.sync(fh)
        os.replace(path + ".tmp", path)
    except OSError:
        pass


GONE_KEPT = 200


def recent_gone(port=None, limit=10):
    """The newest dropped launch records (from _note_gone), for port when given."""
    try:
        with open(gone_path(), encoding="utf-8") as fh:
            rows = [json.loads(x) for x in fh if x.strip()]
    except (OSError, ValueError):
        return []
    if port is not None:
        rows = [r for r in rows if int((r.get("record") or {}).get("port") or D.DEFAULT_PORT) == int(port)]
    return rows[-limit:]


def ours(port=None, explain=None):
    """Records whose process is still the one launched: same pid, same start time, a Studio image. With port, only the
    Studios launched for that daemon port (each lane closes its own). Provisional records of launches still starting are
    kept until they expire, and never returned. explain, a list, gets one row per record read - pid, port, place, and
    taken or why not - so a caller can say why no Studio was recognized. A dropped record is noted in launched_gone.jsonl."""
    keep, pending, gone = [], [], []
    now = time.time()

    def why(r, text, taken=False):
        if explain is not None:
            explain.append({"pid": r.get("pid"), "port": r.get("port"), "place": r.get("place"), "at": r.get("at"),
                            "start": r.get("start"), "pending": r.get("pending") is not None, "taken": taken,
                            "why": text})
    with _RecordLock():
        # the process table is read inside the lock (found 2026-11-22: read before it, a Studio another lane recorded
        # between the two was not in the table and its record was dropped as not running)
        live = dict(running())
        for r in _read_records():
            if r.get("pending"):
                if now - r.get("t0", 0) <= PENDING_SECONDS:
                    pending.append(r)
                    why(r, "a launch still starting (no Studio pid seen yet)")
                else:
                    why(r, "a launch that never showed a Studio pid within %d s: dropped" % PENDING_SECONDS)
                continue
            start = live.get(r.get("pid"), "gone")
            if start == "gone":
                gone.append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "byPid": os.getpid(), "record": r,
                             "why": "pid %s is not a running RobloxStudioBeta.exe" % r.get("pid")})
                why(r, gone[-1]["why"] + ": dropped")
                continue
            if r.get("start") is not None and start is not None and abs(start - r["start"]) > START_SLACK:
                # the pid was reused by another Studio: not ours
                gone.append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "byPid": os.getpid(), "record": r,
                             "why": "pid %s is a Studio started at %.1f, not the launch's %.1f (the pid was reused)"
                                    % (r.get("pid"), start, r["start"])})
                why(r, gone[-1]["why"] + ": dropped")
                continue
            keep.append(r)
            mine = port is None or int(r.get("port") or D.DEFAULT_PORT) == int(port)
            why(r, "a running Studio this record launched" + ("" if mine else ", for port %s, not port %s"
                                                               % (r.get("port") or D.DEFAULT_PORT, port)), taken=mine)
        _write_records(keep + pending)
    _note_gone(gone)
    if port is not None:
        return [r for r in keep if int(r.get("port") or D.DEFAULT_PORT) == int(port)]
    return keep


def not_ours(port, explain):
    """rbx_close's refusal when ours(port) found nothing: the launch records it read (explain) and why each was not
    taken, the records dropped lately for this port, and every running Studio with its place, so a caller can tell a
    record that is gone from a Studio it never launched."""
    studios = []
    for pid, start in running():
        args = CMD_ARGS(pid)
        studios.append({"pid": pid, "start": start, "place": _place_arg(args) if args else None,
                        "testWindow": None if args is None else "-playTestSessionGuid" in args,
                        "commandLine": "read" if args is not None else "unreadable"})
    mine = [e for e in explain if int(e.get("port") or D.DEFAULT_PORT) == int(port)]
    others = [e for e in explain if e not in mine]
    gone = recent_gone(port)
    parts = ["%d launch record(s) for port %d in %s" % (len(mine), port, record_path())]
    parts += ["pid %s (%s): %s" % (e.get("pid"), e.get("place"), e.get("why")) for e in mine]
    if others:
        parts.append("%d record(s) for other ports (%s)" % (len(others), ", ".join(
            "pid %s on %s" % (e.get("pid"), e.get("port")) for e in others)))
    if gone:
        g = gone[-1]
        parts.append("the newest record dropped for this port: pid %s at %s by process %s, %s" % (
            (g.get("record") or {}).get("pid"), g.get("at"), g.get("byPid"), g.get("why")))
    edit = [s for s in studios if s["testWindow"] is not True]
    parts.append("%d Studio(s) running: %s" % (len(studios), ", ".join(
        "pid %s on %s" % (s["pid"], s["place"] or "a place unknown") for s in edit) or "none but test windows"))
    return {"ok": False, "ran": False, "refusedRule": "not-ours", "port": port, "recordFile": record_path(),
            "records": explain, "droppedRecently": gone, "running": studios,
            "error": "no Studio launched by MifBridge for port %d is running; a Studio the person opened is never closed "
                     "from here. %s" % (port, "; ".join(parts))}


def _cmd_args(pid):
    try:
        import psutil
        return psutil.Process(int(pid)).cmdline()
    except Exception:  # noqa: BLE001 - unreadable is "unknown", never a guess
        return None


CMD_ARGS = _cmd_args


def close(pid, discard, wait=20.0):
    """Close one recorded Studio by pid: TerminateProcess with discard. rbx_close always passes discard: WM_CLOSE raises
    Studio's save dialog on 0.748 even for an unchanged place, so the bridge does not use it; the branch stays for a
    caller that a person is watching. Returns {closed, how, stillRunning}."""
    rec = next((r for r in ours() if r.get("pid") == pid), None)
    if rec is None:
        return {"closed": False, "error": "pid %s is not a Studio MifBridge launched" % pid}
    # measured again right before ending it: a record written by an older launch() could name a Studio that only
    # started in its window (found 2026-11-22). A command line that reads and does not hold the record's place is
    # another Studio, never ended from here.
    why = _launch_mismatch(pid, rec.get("place"))
    if why and why != UNREAD:
        return {"closed": False, "refusedRule": "not-ours-place", "pid": pid, "place": rec.get("place"),
                "error": "pid %s is recorded as launched for %s, but %s: nothing was ended" % (pid, rec.get("place"), why)}
    how = "terminate" if discard else "close"
    if discard:
        TERMINATE(pid)
    else:
        CLOSE_WINDOWS(pid)
    deadline = time.time() + wait
    while time.time() < deadline and IS_ALIVE(pid):
        time.sleep(0.25)
    alive = IS_ALIVE(pid)
    out = {"closed": not alive, "how": how, "stillRunning": alive}
    if not alive:
        try:
            with _RecordLock():
                _write_records([r for r in _read_records() if r.get("pid") != pid])
        except RecordsBusy as exc:
            # the Studio is already ended: say the record stays (the next ours() drops it as not running), never raise
            out["recordKept"] = str(exc)
    return out

# ---------------------------------------------------------------- Studio's Auto Recovery files (spec "Extend rbx_launch
# and rbx_close: no Auto Recovery dialog from a bridge Studio"). Measured on 0.748 (RB2720-RB2729):
# - Studio auto-saves an open, changed place every N seconds (the person's setting, never touched here) to
#   AutoSaves/<stem>_AutoRecovery_<n>.rbxl: stem is the place FILE's name without its extension - no folder - and n the
#   first free index, so lane Studios on different ports share one name space.
# - At start, a Studio opening <stem> that finds any <stem>_AutoRecovery_*.rbxl shows a modal "Auto Recovery File Found"
#   (Review / Ignore), a live Studio's file included. It does not hold the plugin off: the plugin connects and works.
# - Ignore moves every such file of that stem into AutoSaves/Archived/<name>_<yyyymmdd_hhmmss>.rbxl (local time); an
#   archived file never brings the dialog back. Ending Studio by pid (rbx_close) leaves its file behind.
# So the bridge moves what is certainly its own, the way Ignore does, and never deletes. Its own is the scratch stem:
# MifScratch is the bridge's file name (the plugin's all-scratch rule needs that exact name), nobody's place. A marked
# copy keeps the person's file name, so its stem is also the person's: its files move only at close, only those written
# while that Studio ran, and only when every other running edit Studio's place file is known and has another name.
SCRATCH_STEM = "MifScratch"
RECOVERY_WATCH_SECONDS = 15.0   # Studio raised the dialog 6-8 s after it started (RB2721): watch until this long after spawn
RECOVERY_CLEAR_SECONDS = 12.0   # Studio looked for recovery files 5.8-8.6 s after it started (RB2721): keep the scratch stem
                                # clear until this long after spawn, since other lanes' Studios keep auto-saving (RB2728)
_ARCHIVE_STAMP = "%Y%m%d_%H%M%S"


def recovery_dir():
    """Studio's AutoSaves folder; MIF_ROBLOX_AUTOSAVES moves it, for a suite (never the real one in a test)."""
    return os.environ.get("MIF_ROBLOX_AUTOSAVES") or os.path.join(
        os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "Roblox", "RobloxStudio", "AutoSaves")


def place_stem(place):
    """The name Studio gives a place's recovery files: the file name without its extension."""
    return os.path.splitext(os.path.basename(place or ""))[0]


def recovery_files(stem, folder=None):
    """Names of stem's live recovery files (<stem>_AutoRecovery_<n>.rbxl, case-insensitive), not the Archived ones."""
    folder = folder or recovery_dir()
    if not stem:
        return []
    pat = re.compile(r"^%s_AutoRecovery_\d+\.rbxl$" % re.escape(stem), re.I)
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(n for n in names if pat.match(n) and os.path.isfile(os.path.join(folder, n)))


def archive_recovery(stem, since=None, now=None):
    """Move stem's recovery files into Archived/, named as Studio's Ignore names them (<name>_<yyyymmdd_hhmmss>.rbxl, a
    _<k> added only if that name is taken). Never deletes, never overwrites. since (epoch seconds): only files last
    written at or after it. Returns {stem, folder, archived: [[from, to]], kept: [[name, why]], failed: [[name, error]]}."""
    folder = recovery_dir()
    arch = os.path.join(folder, "Archived")
    stamp = time.strftime(_ARCHIVE_STAMP, time.localtime(time.time() if now is None else now))
    out = {"stem": stem, "folder": folder, "archived": [], "kept": [], "failed": []}
    for name in recovery_files(stem, folder):
        src = os.path.join(folder, name)
        try:
            written = os.path.getmtime(src)
        except OSError:
            continue                                         # gone already (another lane, or Studio's own Ignore)
        if since is not None and written < since - START_SLACK:
            out["kept"].append([name, "written before this Studio started"])
            continue
        base = name[:-len(".rbxl")]
        dest, k = os.path.join(arch, "%s_%s.rbxl" % (base, stamp)), 1
        while os.path.exists(dest):
            dest, k = os.path.join(arch, "%s_%s_%d.rbxl" % (base, stamp, k)), k + 1
        try:
            os.makedirs(arch, exist_ok=True)
            os.rename(src, dest)                             # a rename refuses an existing target: nothing overwritten
            out["archived"].append([name, os.path.basename(dest)])
        except OSError as exc:
            out["failed"].append([name, str(exc)])
    return out


def keep_clear(stem, until, every=0.25):
    """Move stem's recovery files again and again until `until` (epoch s): a running Studio of another lane writes a new
    one every few minutes, and one landing between rbx_launch's first move and the new Studio's own look raised the
    dialog in both Studios of a two-port relaunch (RB2728). Returns [[from, to]] of everything it moved."""
    moved = []
    while True:
        moved += archive_recovery(stem)["archived"]
        if time.time() >= until:
            return moved
        time.sleep(every)


def _place_arg(cmd_args):
    for i, a in enumerate(cmd_args[:-1]):
        if a.lower() == "-localplacefile":
            return cmd_args[i + 1]
    return None


def other_place_stems(exclude_pid):
    """(stems, unknown): the place-file stems every other running edit Studio has open, read from -localPlaceFile, and
    the pids whose place cannot be known (command line unreadable, or opened from Roblox with no local file). Play-test
    processes (-playTestSessionGuid) are not edit Studios and are skipped."""
    stems, unknown = set(), []
    for pid, _start in running():
        if pid == exclude_pid:
            continue
        args = CMD_ARGS(pid)
        if args is None:                                     # unreadable is "unknown", never a guess
            unknown.append(pid)
            continue
        if "-playTestSessionGuid" in args:
            continue
        place = _place_arg(args)
        if place is None:
            unknown.append(pid)
        else:
            stems.add(place_stem(place).lower())
    return stems, unknown


OTHER_PLACE_STEMS = other_place_stems


def archive_after_close(place, start):
    """rbx_close, after its Studio has exited: move the recovery files of the place that Studio had open, and no other
    stem's (found 2026-11-22: a close of a Studio on a copy also moved MifScratch files while other lanes' Studios on
    their own scratch places ran). place is the place it had open (rbx_close passes None when that is not known, and
    nothing moves). The scratch place's stem moves whole; a marked copy's (a place under the bridge's places folder with
    another name) only files written since `start`; a place outside the places folder never. Nothing moves while another
    running Studio has, or may have, a place of that name open (its command line unreadable or holding no file): the
    scratch stem is every lane's, and the next launch moves what is left. Returns a list of one archive_recovery result,
    or one row saying why nothing moved (skipped)."""
    stem = place_stem(place)
    none = {"stem": stem or None, "archived": [], "kept": [], "failed": []}
    if not place or not stem:
        return [dict(none, skipped="the place this Studio had open is not known: no recovery file was moved")]
    scratch = is_scratch_place(place)
    if not scratch and not in_places_folder(place):
        return [dict(none, skipped="%s is not in MifBridge's places folder: its recovery files are the person's" % place)]
    stems, unknown = OTHER_PLACE_STEMS(None)
    if stem.lower() in stems or unknown:
        why = "another running Studio has, or may have, a place named %s open (pids unknown: %s)" % (stem, unknown)
        return [dict(none, kept=[[n, why] for n in recovery_files(stem)])]
    return [archive_recovery(stem, since=None if scratch else start)]


def _visible_titled(pid, title):
    """Visible top-level windows of pid whose title is exactly `title`."""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32")
    hits = []
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _lp):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            if buf.value == title:
                hits.append(int(hwnd or 0))
        return True
    user32.EnumWindows(proto(cb), 0)
    return hits


# UI Automation names the dialog by its Qt object name (RB2722): a QDialog, a child of Studio's main window in the UIA
# tree, whose AutomationId holds the built-in Dialog plugin's "_PluginId:AutoRecovery" and "_ItemId:RecoveryPrompt". Read
# through Windows PowerShell's UIAutomationClient, started with no window; only each top window and its children are
# searched (a whole-subtree walk of Studio takes most of a minute).
_UIA_SCRIPT = r"""
param([int]$ProcId)
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
$A = [System.Windows.Automation.AutomationElement]
$c = New-Object System.Windows.Automation.PropertyCondition($A::ProcessIdProperty, $ProcId)
foreach ($t in $A::RootElement.FindAll([System.Windows.Automation.TreeScope]::Children, $c)) {
  foreach ($k in $t.FindAll([System.Windows.Automation.TreeScope]::Element -bor [System.Windows.Automation.TreeScope]::Children, [System.Windows.Automation.Condition]::TrueCondition)) {
    $id = $k.Current.AutomationId
    if ($id -like "*_PluginId:AutoRecovery*" -and $id -like "*_ItemId:RecoveryPrompt*") { "HWND " + $k.Current.NativeWindowHandle }
  }
}
"""


def _uia_recovery_hwnds(pid):
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "mifbridge_uia_recovery_%d.ps1" % os.getpid())
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_UIA_SCRIPT)
    try:
        p = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", path,
                            "-ProcId", str(int(pid))], capture_output=True, text=True, timeout=60,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(x.split()[1]) for x in p.stdout.splitlines() if x.startswith("HWND ") and x.split()[1].isdigit()]


def find_recovery_dialog(pid):
    """The hwnd of the Auto Recovery dialog in Studio pid, or None: a visible top-level window of that pid titled
    "RobloxStudio" (cheap, every poll) that UI Automation then names as the AutoRecovery RecoveryPrompt."""
    cands = _visible_titled(pid, "RobloxStudio")
    if not cands:
        return None
    named = _uia_recovery_hwnds(pid)
    hit = [h for h in cands if h in named]
    return hit[0] if hit else None


FIND_RECOVERY_DIALOG = find_recovery_dialog


def ignore_point(w, h, rgba):
    """Where the dialog's Ignore button is, in its content's pixels: the plain (not blue) button to the right of the blue
    Review button on the same row (0.748's layout, RB2723). Returns ((x, y), None) or (None, why). Never a blue pixel:
    no Review button found, or nothing plain right of it, is a refusal, not a guess."""
    if not w or not h or len(rgba) < w * h * 4:
        return None, "no pixels"

    def px(x, y):
        i = (y * w + x) * 4
        return rgba[i], rgba[i + 1], rgba[i + 2]

    def blue(c):
        return c[2] >= 180 and c[2] - c[0] >= 100 and c[2] - c[1] >= 60

    bg = px(2, 2)
    bx = [(x, y) for y in range(h // 2, h) for x in range(w) if blue(px(x, y))]
    if len(bx) < 100:
        return None, "no blue Review button in the dialog's lower half"
    x1 = max(x for x, _y in bx)
    y0, y1 = min(y for _x, y in bx), max(y for _x, y in bx)
    rows = range(y0, y1 + 1)

    def plain(c):
        return not blue(c) and sum(abs(a - b) for a, b in zip(c, bg)) > 24

    run = []
    for x in range(x1 + 1, w):
        if sum(1 for y in rows if plain(px(x, y))) >= 0.6 * len(rows):
            run.append(x)
        elif run and len(run) >= 20:
            break
        else:
            run = []
    if len(run) < 20:
        return None, "no button right of Review"
    pt = ((run[0] + run[-1]) // 2, (y0 + y1) // 2)
    if blue(px(*pt)):
        return None, "the point found is blue"
    return pt, None


def answer_ignore(hwnd, wait=5.0, draw_wait=8.0):
    """Press Ignore in the Auto Recovery dialog hwnd: its content window (a native Qt child) is drawn with PrintWindow,
    ignore_point finds the button, and a click is POSTED to that child window - no mouse move, no focus change. Returns
    {answered, point, why?}; answered only when the dialog is gone afterwards."""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32")
    kids = []
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumChildWindows(wintypes.HWND(hwnd), proto(lambda c, _lp: kids.append(int(c or 0)) or True), 0)
    if not kids:
        return {"answered": False, "why": "the dialog has no content window"}
    child = kids[0]
    rc = wintypes.RECT()
    user32.GetClientRect(wintypes.HWND(child), ctypes.byref(rc))
    w, h = rc.right - rc.left, rc.bottom - rc.top
    if w <= 0 or h <= 0:
        return {"answered": False, "why": "the content window has no size"}
    # the window shows before its content is drawn (RB2723: found at once, it read as plain background), so read it
    # again until the buttons are there
    pt, why, tries, until = None, None, 0, time.time() + draw_wait
    while True:
        tries += 1
        pt, why = ignore_point(*_print_hwnd(child, w, h))
        if pt is not None or time.time() >= until:
            break
        time.sleep(0.5)
    if pt is None:
        return {"answered": False, "why": "%s (%d reads over %ss)" % (why, tries, draw_wait)}
    lp = (pt[1] << 16) | (pt[0] & 0xFFFF)
    for msg, wp in ((0x0200, 0), (0x0201, 1), (0x0202, 0)):     # WM_MOUSEMOVE, WM_LBUTTONDOWN, WM_LBUTTONUP
        user32.PostMessageW(wintypes.HWND(child), msg, wp, lp)
        time.sleep(0.06)
    deadline = time.time() + wait
    while time.time() < deadline and user32.IsWindowVisible(wintypes.HWND(hwnd)):
        time.sleep(0.2)
    gone = not user32.IsWindowVisible(wintypes.HWND(hwnd))
    return {"answered": gone, "point": list(pt), "reads": tries, **({} if gone else {"why": "the dialog was still showing %ss after the "
                                                                         "click" % wait})}


ANSWER_IGNORE = answer_ignore


def watch_recovery_dialog(pid, place, t0, until=None):
    """After rbx_launch: watch Studio pid (the one this launch started, confirmed by its command line) until
    t0 + RECOVERY_WATCH_SECONDS for the Auto Recovery dialog. On the scratch place, answer Ignore (never Review) and say
    which files that moved; on any other place leave it for the person, saying why. Returns {shown, ...}."""
    until = t0 + RECOVERY_WATCH_SECONDS if until is None else until
    hwnd = None
    while True:
        hwnd = FIND_RECOVERY_DIALOG(pid)
        if hwnd or time.time() >= until:
            break
        time.sleep(0.5)
    if not hwnd:
        return {"shown": False, "watchedUntil": round(until - t0, 1)}
    stem = place_stem(place)
    if stem.lower() != SCRATCH_STEM.lower():
        return {"shown": True, "answered": False, "pid": pid, "files": recovery_files(stem),
                "why": "the dialog names %s's recovery files, which are not provably MifBridge's (a copy keeps the "
                       "person's file name): left for the person" % stem}
    before = recovery_files(stem)
    r = ANSWER_IGNORE(hwnd)
    return dict(r, shown=True, pid=pid, button="Ignore", filesBefore=before, filesAfter=recovery_files(stem))


# ---------------------------------------------------------------- a play test's own Studio logs (spec: keep what was
# measured when the play session ends early). Every Studio process writes %LOCALAPPDATA%/Roblox/logs/<ver>_<UTC>_Studio_
# <id>_last.log; its first lines hold its command line, and a test server or client of ours carries -task StartServer or
# StartClient with -editpid <our edit Studio> (RB1017). A script's print, warn and error are [FLog::CreatorOutput],
# [FLog::CreatorWarning] "Warning: ..." and [FLog::CreatorError] "Error: ..." (measured on 0.748, RB1131).
LOG_DIR = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "Roblox", "logs")
_LOG_LINE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z,[\d.]+,[0-9a-fA-F]+,\d+(?:,(\w+))? \[(\w+)::(\w+)\] ?(.*)$")
_STAMPED = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d")
_CREATOR = {"CreatorOutput": "MessageOutput", "CreatorWarning": "MessageWarning", "CreatorError": "MessageError",
            "CreatorInfo": "MessageInfo"}
_EVENTS = (("joined", re.compile(r"^Player (-?\d+) added, there are now (\d+) players")),
           ("left", re.compile(r"^Player (-?\d+) leaving, (\d+) remain")),
           ("resultTimedOut", re.compile(r"Timed out in reply with name SendTestResult")),
           ("resultSent", re.compile(r"Finished processing reply with name SendTestResult")),
           ("kicked", re.compile(r"[Kk]ick")),
           ("normalExit", re.compile(r"About to normally exit the main event loop")))


def _utc(stamp):
    try:
        import calendar
        whole, _dot, frac = stamp.partition(".")
        return calendar.timegm(time.strptime(whole, "%Y-%m-%dT%H:%M:%S")) + (float("0." + frac) if frac else 0.0)
    except (ValueError, OverflowError):
        return None


def read_test_log(path, max_lines=400):
    """One Studio log, read: its role (server, client or None), -editpid, the script output lines (message, type, t),
    the session events and its first and last timestamps (UTC seconds). Never raises; None when unreadable."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(32 << 20)
    except OSError:
        return None
    text = raw.decode("utf-8", "replace")
    head = text[:6000]
    task = re.search(r"-task (StartServer|StartClient|EditFile|\w+)", head)
    ep = re.search(r"-editpid (\d+)", head)
    out = {"file": path, "role": {"StartServer": "server", "StartClient": "client"}.get(task.group(1)) if task else None,
           "editPid": int(ep.group(1)) if ep else None, "pid": None, "first": None, "last": None, "output": [],
           "events": [], "truncated": False}
    last_creator = None
    for line in text.splitlines():
        m = _LOG_LINE.match(line)
        if not m:
            if _STAMPED.match(line):
                last_creator = None                            # a stamped line of another shape ([telemetryLog] ...)
            elif last_creator is not None and line.strip() and len(last_creator["message"]) < 4000:
                last_creator["message"] += "\n" + line       # a stack trace's continuation lines
            continue
        stamp, _level, _ns, chan, msg = m.groups()
        t = _utc(stamp)
        out["first"] = out["first"] if out["first"] is not None else t
        out["last"] = t
        last_creator = None
        if chan in _CREATOR:
            kind = _CREATOR[chan]
            if chan == "CreatorOutput" and msg.startswith("Info: "):
                kind = "MessageInfo"                           # a stack trace's Begin/Script/End lines
            for pre in ("Warning: ", "Error: ", "Info: "):
                if kind != "MessageOutput" and msg.startswith(pre):
                    msg = msg[len(pre):]
            if len(out["output"]) < max_lines or msg.startswith(("MIFPROBE_RESULT ", "MIFPROBE_SAMPLE ")):
                last_creator = {"message": msg, "type": kind, "utc": t}
                out["output"].append(last_creator)
            else:
                out["truncated"] = True
            continue
        if chan == "UIThreadNotifier":
            # 0.748 writes "...,Info [FLog::UIThreadNotifier] connection has been disconnected": the name is the channel,
            # not part of the message (RB2925). Its constructor line names this process's own pid.
            pm = re.match(r"Constructing UIThreadNotifier for process '(\d+)'", msg)
            if pm and out["pid"] is None:
                out["pid"] = int(pm.group(1))
            elif msg.startswith("connection has been disconnected"):
                out["events"].append({"event": "uiDisconnected", "utc": t, "line": msg[:300]})
            continue
        for name, rx in _EVENTS:
            if name == "kicked" and chan not in ("Output",):
                continue
            em = rx.search(msg)
            if em:
                out["events"].append({"event": name, "utc": t, "line": msg[:300]})
                break
    return out


def test_logs(own, since, log_dir=None, max_lines=400):
    """The logs of THIS edit Studio's play-test processes (server and clients, -editpid own) started since `since` (UTC
    seconds), each read by read_test_log, oldest first; [] when own is unknown, never another Studio's."""
    if not own:
        return []
    d = log_dir or LOG_DIR
    try:
        names = [n for n in os.listdir(d) if n.endswith(".log") and "_Studio_" in n]
    except OSError:
        return []
    out = []
    for n in names:
        p = os.path.join(d, n)
        try:
            if os.path.getmtime(p) < since - 5:
                continue
        except OSError:
            continue
        r = read_test_log(p, max_lines=max_lines)
        if r and r["editPid"] == int(own) and r["role"] in ("server", "client") and (r["first"] or 0) >= since - 5:
            out.append(r)
    return sorted(out, key=lambda r: r["first"] or 0)


def probe_result_from_log(server, nonce=None):
    """The probe's result as its server printed it (MIFPROBE_RESULT i/n parts, plugin 0.16.6), or None. With nonce, only
    a result that carries that nonce (plugin 0.17.4), or none at all (an older plugin), is taken."""
    parts = {}
    total = None
    for o in (server or {}).get("output") or []:
        m = re.match(r"^MIFPROBE_RESULT (\d+)/(\d+) (.*)$", o.get("message") or "", re.S)
        if m:
            parts[int(m.group(1))] = m.group(3)
            total = int(m.group(2))
    if not total or len(parts) != total:
        return None
    try:
        res = json.loads("".join(parts[i] for i in range(1, total + 1)))
    except ValueError:
        return None
    # a result with no nonce is from a plugin before 0.17.4 (taken, as before); one with another nonce never is
    if not isinstance(res, dict) or (nonce is not None and res.get("nonce") not in (nonce, None)):
        return None
    return res


def samples_from_log(server, nonce):
    """The samples a long sample probe printed as it went (MIFPROBE_SAMPLE <nonce> <json>, plugin 0.17.7), this run's
    only, in order; [] when none."""
    out = []
    pre = "MIFPROBE_SAMPLE %s " % nonce
    for o in (server or {}).get("output") or []:
        m = o.get("message") or ""
        if nonce and m.startswith(pre):
            try:
                row = json.loads(m[len(pre):])
            except ValueError:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


# ---------------------------------------------------------------- one play test on the machine at a time (spec: "one
# play test per machine at a time, and results matched to the session that asked"). One Studio's test ending cuts
# another's off from its result (RB1015, RB1040), so every MifBridge play probe on the machine takes this lock for the
# whole test. It is a file, not a per-daemon thing: every lane and every game session's daemon shares it. It lives in
# the machine's MifBridge folder whatever MIF_ROBLOX_HOME says (a lane's own home would make a lock nobody else sees);
# MIF_ROBLOX_PLAYLOCK moves it, for a suite. The holder is a process (pid and start time): a lock whose holder is gone,
# or whose pid now names another process, is broken and the break is reported.
def play_lock_path():
    return os.environ.get("MIF_ROBLOX_PLAYLOCK") or os.path.join(
        os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "MifBridge", "Roblox", "playtest.lock")


def _proc_start(pid):
    """pid's start time (Unix seconds, to 0.1 s), or None when it cannot be read (gone, or no access). psutil when it is
    installed; else Win32 GetProcessTimes, since psutil is not in requirements.txt (found 2026-11-22: without it every
    holder read as gone and every waiter broke every held lock)."""
    try:
        import psutil
        return round(psutil.Process(int(pid)).create_time(), 1)
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - gone or unreadable
        return None
    if os.name != "nt":
        return None
    try:
        W = _w32()
        k = W._k32()
        h = W._open(k, pid)
        if not h:
            return None
        try:
            import ctypes
            from ctypes import wintypes
            code = wintypes.DWORD(0)
            if not k.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != W.STILL_ACTIVE:
                return None
            st = W._start_epoch(k, h)
            return None if st is None else round(st, 1)
        finally:
            k.CloseHandle(h)
    except Exception:  # noqa: BLE001 - unreadable is "unknown", never a guess
        return None


PROC_START = _proc_start


def _proc_gone(pid):
    """True only when pid certainly names no running process: psutil says so, or Win32 OpenProcess refuses the pid as
    invalid (ERROR_INVALID_PARAMETER), or the process has exited. Access denied, or anything else, is not "gone"."""
    try:
        import psutil
        return not psutil.pid_exists(int(pid))
    except ImportError:
        pass
    except Exception:  # noqa: BLE001
        return False
    try:
        if os.name != "nt":
            try:
                os.kill(int(pid), 0)
                return False
            except ProcessLookupError:
                return True
            except OSError:
                return False
        import ctypes
        from ctypes import wintypes
        W = _w32()
        k = W._k32()
        h = k.OpenProcess(W.PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return ctypes.get_last_error() == 87            # ERROR_INVALID_PARAMETER: no process has this pid
        try:
            code = wintypes.DWORD(0)
            return bool(k.GetExitCodeProcess(h, ctypes.byref(code))) and code.value != W.STILL_ACTIVE
        finally:
            k.CloseHandle(h)
    except Exception:  # noqa: BLE001
        return False


PROC_GONE = _proc_gone

# A holder whose start time cannot be read and that is not certainly gone is taken as alive until its lock is older than
# the longest probe can hold it: seconds 1800 + settle 30 + 600 more timeout + the cut wait 30 + the 30 s end wait, and a
# minute of slack (tools.PROBE_MAX_SECONDS, CUT_GRACE; test_roblox_play RQ710 keeps them in step).
PLAY_LOCK_UNVERIFIED_SECONDS = 1800 + 30 + 600 + 30 + 30 + 60


def _holder_state(h, now=None):
    """"alive"; "dead"; "unverified" (start unreadable, pid not certainly gone, lock younger than
    PLAY_LOCK_UNVERIFIED_SECONDS: counted alive); or "expired" (the same, but older: counted gone)."""
    if not isinstance(h, dict) or not isinstance(h.get("pid"), int):
        return "dead"
    st = PROC_START(h["pid"])
    if st is None:
        if PROC_GONE(h["pid"]):
            return "dead"
        since = h.get("since")
        now = time.time() if now is None else now
        if isinstance(since, (int, float)) and not isinstance(since, bool) and now - since < PLAY_LOCK_UNVERIFIED_SECONDS:
            return "unverified"
        return "expired"
    return "alive" if h.get("pidStart") is None or abs(st - h["pidStart"]) <= START_SLACK else "dead"


def _holder_alive(h):
    return _holder_state(h) in ("alive", "unverified")


holder_alive = _holder_alive    # public: rbx_play_probe asks whether the lock's holder still runs


def read_play_lock():
    try:
        with open(play_lock_path(), encoding="utf-8") as fh:
            h = json.load(fh)
        return h if isinstance(h, dict) else {}
    except (OSError, ValueError):
        return None


# The waiters' queue (spec "the machine play-test lock serves waiters in turn"): polling the lock alone let a waiter that
# came first wait its whole lock_wait while later ones took the lock eight times (RB2135). Each waiter now writes one
# file into a folder beside the lock, named by its arrival time, and tries the lock only when no live waiter came before
# it. The lock file itself is unchanged, so a daemon running the older code still sees a held lock and waits; it does not
# read the queue, so it may still take a free lock out of turn. A waiter whose process is gone, or whose own lock_wait
# ran out QUEUE_GRACE seconds ago, is skipped and its file removed.
QUEUE_GRACE = 5.0


def play_queue_path():
    return play_lock_path() + ".queue"


def _queue_entries():
    """[(file name, entry)] in arrival order; an unreadable file is (name, None)."""
    q = play_queue_path()
    try:
        names = sorted(n for n in os.listdir(q) if n.endswith(".json"))
    except OSError:
        return []
    out = []
    for n in names:
        try:
            with open(os.path.join(q, n), encoding="utf-8") as fh:
                e = json.load(fh)
            out.append((n, e if isinstance(e, dict) else None))
        except (OSError, ValueError):
            out.append((n, None))
    return out


def _waiter_live(name, e, now):
    """A queued waiter still waiting: its process runs (pid and start time) and its lock_wait has not run out. An
    unreadable file counts as live for 10 s from its write (a write in progress), then not."""
    if e is None:
        try:
            return now - os.path.getmtime(os.path.join(play_queue_path(), name)) < 10
        except OSError:
            return False
    if not _holder_alive(e):
        return False
    until = e.get("until")
    return not isinstance(until, (int, float)) or now <= until + QUEUE_GRACE


# Short runs before soaks (spec "short runs go before soaks in the play-test queue, while one lock remains"): a waiter's
# file carries its class - "short" (a probe of SHORT_RUN_SECONDS or less) or "soak". A short run goes ahead of a soak
# that came before it, but no soak is passed by more than SOAK_PASS_BOUND short runs that came after it, so a soak
# cannot starve (RB2135). Each short run that took the lock ahead of a waiting soak leaves a marker file of its own,
# <soak's file stem>.pass.<its file stem>, which counts the soak's passes (every file has one writer); a soak removes its
# markers when it leaves, and a dead soak's go with its file. A waiter with no class (older code) is a short run, so an
# older waiter keeps strict arrival order among the others.
SHORT_RUN_SECONDS = 120
SOAK_PASS_BOUND = 3
PASS_MARK = ".pass."


def run_class(seconds):
    """"short" for a run of SHORT_RUN_SECONDS or less, else "soak"."""
    try:
        return "soak" if float(seconds) > SHORT_RUN_SECONDS else "short"
    except (TypeError, ValueError):
        return "short"


def _class_of(e):
    return "soak" if isinstance(e, dict) and e.get("class") == "soak" else "short"


def _passes(name):
    """The markers of the short runs that went ahead of the soak queued as `name`."""
    stem = name[:-len(".json")] + PASS_MARK
    try:
        return sorted(n for n in os.listdir(play_queue_path()) if n.startswith(stem))
    except OSError:
        return []


def _drop_passes(name):
    for n in _passes(name):
        try:
            os.remove(os.path.join(play_queue_path(), n))
        except OSError:
            pass


def _live_queue(skipped):
    """[(name, entry)] of the live waiters in arrival order; dead or expired ones (and a dead soak's pass markers) are
    removed and added to skipped."""
    now, out = time.time(), []
    for n, e in _queue_entries():
        if _waiter_live(n, e, now):
            out.append((n, e))
            continue
        try:
            os.remove(os.path.join(play_queue_path(), n))
            skipped.append({k: (e or {}).get(k) for k in ("pid", "port", "studioPid", "nonce", "at")})
        except OSError:
            pass
        _drop_passes(n)
    return out


def _soak_rank_ahead(q, s, j):
    """True when the soak at q[s] goes before the short run at q[j] (s < j): the soak was passed SOAK_PASS_BOUND times
    already, counting its markers and the live short runs queued between the two."""
    between = sum(1 for n, e in q[s + 1:j] if _class_of(e) == "short")
    return len(_passes(q[s][0])) + between >= SOAK_PASS_BOUND


def _ahead(mine, skipped, my_class="short"):
    """The live waiters that go before `mine` (a file name): every earlier one, except that a short run is not held by
    an earlier soak that may still be passed, and a soak also waits for later short runs that may pass it."""
    q = _live_queue(skipped)
    names = [n for n, _e in q]
    i = names.index(mine) if mine in names else len(names)
    out = []
    for k, (n, e) in enumerate(q):
        if k == i:
            continue
        if k < i:
            if my_class == "short" and _class_of(e) == "soak" and i < len(q) and not _soak_rank_ahead(q, k, i):
                continue
        elif not (my_class == "soak" and _class_of(e) == "short" and not _soak_rank_ahead(q, i, k)):
            continue
        out.append({k2: (e or {}).get(k2) for k2 in ("pid", "port", "studioPid", "at")}
                   | {"runClass": _class_of(e)})
    return out


def _mark_passes(mine, me):
    """After a short run took the lock: a marker in each live soak queued before it, naming this run. -> [soak nonces]."""
    passed = []
    stem = mine[:-len(".json")]
    for n, e in _live_queue([]):
        if n >= mine:
            break
        if _class_of(e) != "soak":
            continue
        try:
            with open(os.path.join(play_queue_path(), n[:-len(".json")] + PASS_MARK + stem), "w",
                      encoding="utf-8") as fh:
                json.dump({k: me.get(k) for k in ("pid", "port", "studioPid", "nonce", "at")}, fh)
            passed.append((e or {}).get("nonce"))
        except OSError:
            pass
    return passed


HALF_WRITTEN_GRACE = 10.0     # seconds an unparsable lock file may sit unchanged before it counts as a dead holder's


def _half_written_stale(path):
    """True when the lock file exists, does not parse, and was last written HALF_WRITTEN_GRACE seconds ago or more: a
    holder writes its record right after creating the file, so a file that stays unparsable that long was left by one
    that died in between (found 2026-11-22 after a power loss: a 0-byte lock would have blocked every play test)."""
    try:
        return time.time() - os.path.getmtime(path) >= HALF_WRITTEN_GRACE
    except OSError:
        return False


def take_play_lock(port, studio, nonce, wait, every=1.0, run_class="short"):
    """Take the machine's play-test lock, waiting up to `wait` seconds, in turn: a waiter tries the lock only when no
    live waiter queued before it. Returns {"ok", "waitedSeconds", "heldBy" (the holders seen while waiting), "broken"
    (stale holders removed), "queuePosition" (1 = first in line when this waiter arrived), "queuedAhead" (the waiters
    before it then), "skipped" (queued waiters that were gone or out of time)}; ok False when the lock or the line
    ahead kept it for all of wait."""
    path = play_lock_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    me = {"pid": os.getpid(), "pidStart": PROC_START(os.getpid()), "port": port, "studioPid": studio, "nonce": nonce,
          "since": time.time(), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t0 = time.time()
    seen, broken, skipped = [], [], []
    qdir = play_queue_path()
    os.makedirs(qdir, exist_ok=True)
    mine = "%020d-%08d-%s.json" % (time.time_ns(), os.getpid(), re.sub(r"[^0-9A-Za-z_-]", "_", str(nonce))[:40])
    cls = "soak" if run_class == "soak" else "short"
    entry = dict(me, arrived=t0, until=t0 + float(wait), **{"class": cls})
    tmp = os.path.join(qdir, mine + ".tmp")
    from mif_roblox import durable
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(entry, fh)
        durable.sync(fh)
    os.replace(tmp, os.path.join(qdir, mine))
    first_ahead = _ahead(mine, skipped, cls)
    out = {"queuePosition": len(first_ahead) + 1, "queuedAhead": first_ahead, "runClass": cls}
    try:
        got = dict(_take_in_turn(path, me, mine, wait, every, t0, seen, broken, skipped, first_ahead, cls), **out)
        if got.get("ok"):
            if cls == "short":
                got["wentAheadOf"] = _mark_passes(mine, me)
            else:
                passes = []
                for n in _passes(mine):
                    try:
                        with open(os.path.join(qdir, n), encoding="utf-8") as fh:
                            passes.append(json.load(fh).get("nonce"))
                    except (OSError, ValueError):
                        passes.append(None)
                got["passedBy"] = passes
        return got
    finally:
        try:
            os.remove(os.path.join(qdir, mine))
        except OSError:
            pass
        if cls == "soak":
            _drop_passes(mine)


def _read_lock_at(p):
    try:
        with open(p, encoding="utf-8") as fh:
            h = json.load(fh)
        return h if isinstance(h, dict) else {}
    except (OSError, ValueError):
        return None


def _put_back(aside, path):
    """Move a lock moved aside back to path, never over a lock taken there meanwhile. True when it is back."""
    try:
        if os.name == "nt":
            os.rename(aside, path)                   # refuses an existing target on Windows
        else:
            os.link(aside, path)                     # refuses an existing target; rename would replace it
            os.remove(aside)
        return True
    except OSError:
        return False


def _break_stale(path, h, unverifiable=False):
    """Remove the lock at path only if it is still the stale holder h's: move it aside under a name of this process's
    own (a rename either moves the whole file or nothing), read it there, delete it when it is h, else put it back. h None
    is a half-written file: deleted when it still does not parse.
    Returns the broken row, or None when nothing was broken."""
    aside = "%s.break-%d-%d" % (path, os.getpid(), time.time_ns())
    try:
        os.rename(path, aside)
    except OSError:
        return None                                  # gone already (another waiter broke it) or held open
    got = _read_lock_at(aside)
    if got == h:
        try:
            os.remove(aside)
        except OSError:
            pass
        row = {k: (h or {}).get(k) for k in ("pid", "port", "studioPid", "nonce", "at")}
        if h is None:
            row["halfWritten"] = True
        if unverifiable:
            since = h.get("since")
            row.update(unverifiable=True, ageSeconds=round(time.time() - since, 1) if isinstance(since, (int, float)) else None)
        return row
    if not _put_back(aside, path):
        # a third waiter took the free lock in the instant the file was aside: the moved holder is live and now has no
        # file. Leave its file where it is, named, rather than overwrite the new holder's.
        return dict({k: (got or {}).get(k) for k in ("pid", "port", "studioPid", "nonce", "at")}, displaced=True,
                    file=aside)
    return None


def _take_in_turn(path, me, mine, wait, every, t0, seen, broken, skipped, ahead, cls="short"):
    lock_unverifiable = []

    def done(ok):
        out = {"ok": ok, "waitedSeconds": round(time.time() - t0, 1), "heldBy": seen, "broken": broken,
               "skipped": skipped, "lock": path}
        if lock_unverifiable:
            out["lockUnverifiable"] = lock_unverifiable
        return out
    while True:
        if ahead:
            for w in ahead:
                brief = dict(w, queued=True)
                if brief not in seen:
                    seen.append(brief)
            if time.time() - t0 >= wait:
                return done(False)
            time.sleep(every)
            ahead = _ahead(mine, skipped, cls)
            continue
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            from mif_roblox import durable
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(me, fh)
                durable.sync(fh)             # a lock left empty by a power loss is broken only after HALF_WRITTEN_GRACE
            return done(True)
        except FileExistsError:
            pass
        h = read_play_lock()
        if h is None and _half_written_stale(path):
            # a lock file that does not parse and has not changed for HALF_WRITTEN_GRACE seconds: a holder that died between
            # creating the file and writing it (a crash or power loss) left it, and nothing else would ever remove it
            # (moved aside and read again first, like a stale holder's: a holder may have written it meanwhile)
            row = _break_stale(path, None)
            if row is not None:
                broken.append(row)
            continue
        state = _holder_state(h) if h is not None else "alive"
        if state in ("dead", "expired"):
            # stale: its holder is gone. Another waiter may have broken it and taken the lock since it was read, so the
            # file is first moved aside under a name of this waiter's own, read there, and deleted only when it is still
            # the stale holder's; anything else is put back (found 2026-11-22: read, read again, remove let a live lock
            # that replaced it between the second read and the remove be deleted).
            row = _break_stale(path, h, unverifiable=state == "expired")
            if row is not None:
                broken.append(row)
            continue
        brief = {k: (h or {}).get(k) for k in ("pid", "port", "studioPid", "at")}
        if state == "unverified":
            brief["unverified"] = True
            unverified = dict(brief, nonce=h.get("nonce"), since=h.get("since"))
            if unverified not in lock_unverifiable:
                lock_unverifiable.append(unverified)
        if brief not in seen:
            seen.append(brief)
        if time.time() - t0 >= wait:
            return done(False)
        time.sleep(every)
        ahead = _ahead(mine, skipped, cls)


def play_lock_status(now=None):
    """The machine play-test lock as it stands, read only: nothing is written, broken or removed, and no queue file is
    made (rbx_status, spec "read without joining"; take_play_lock's own reads remove dead waiters, this one does not).
    {held, holder {pid, port, studioPid, nonce, at}, heldSeconds, holderState (alive, unverified, dead, expired,
    halfWritten), holderAlive, stale (a holder a waiter would break), waiters [in arrival order: pid, port, studioPid,
    nonce, at, waitedSeconds, live], liveWaiters, lock, queue}."""
    now = time.time() if now is None else now
    path = play_lock_path()
    out = {"lock": path, "queue": play_queue_path(), "held": False, "holder": None, "heldSeconds": None,
           "holderState": None, "holderAlive": None, "stale": False}
    if os.path.exists(path):
        h = read_play_lock()
        out["held"] = True
        if h is None:
            state = "halfWritten"
            out["stale"] = _half_written_stale(path)
        else:
            state = _holder_state(h, now)
            out["holder"] = {k: h.get(k) for k in ("pid", "port", "studioPid", "nonce", "at")}
            since = h.get("since")
            if isinstance(since, (int, float)) and not isinstance(since, bool):
                out["heldSeconds"] = round(now - since, 1)
            out["stale"] = state in ("dead", "expired")
        out.update(holderState=state, holderAlive=state in ("alive", "unverified"))
    waiters = []
    for n, e in _queue_entries():
        live = _waiter_live(n, e, now)
        row = {k: (e or {}).get(k) for k in ("pid", "port", "studioPid", "nonce", "at")}
        row["runClass"] = _class_of(e)
        if row["runClass"] == "soak":
            row["passedBy"] = len(_passes(n))
        arrived = (e or {}).get("arrived")
        row["waitedSeconds"] = (round(now - arrived, 1) if isinstance(arrived, (int, float))
                                and not isinstance(arrived, bool) else None)
        row["live"] = live
        if e is None:
            row["unreadable"] = n
        waiters.append(row)
    out["waiters"] = waiters
    out["liveWaiters"] = sum(1 for w in waiters if w["live"])
    return out


def release_play_lock(nonce):
    """Remove the lock only when it is this run's (same nonce and pid): never another run's."""
    h = read_play_lock()
    if isinstance(h, dict) and h.get("nonce") == nonce and h.get("pid") == os.getpid():
        try:
            os.remove(play_lock_path())
            return True
        except OSError:
            return False
    return False


def leftover_tests(own):
    """Play-test processes of THIS edit Studio still running (its -editpid, or its process tree): a test server or client
    an earlier probe left (a cancelled call, a killed client). [(pid, start)]."""
    if not own:
        return []
    return [(p, s) for p, s in running() if p != own and is_test_window(p) is not False and belongs_to(p, own)]


# The runs this machine's probes started, so a reply that outlived its call can be collected later (rbx_play_probe
# collect). Newest last, at most RUNS_KEEP; in the daemon's home (per lane), keyed by port. Each row also keeps runMode,
# players, runClass, waitedSeconds, queuePosition, and once the lock is let go holdSeconds, so a queue can be replayed
# from the file (spec N86; 40 rows were one busy hour, RB2926).
RUNS_KEEP = 400


def runs_path():
    return os.path.join(D.home(), "play_runs.json")


def note_run(row):
    with _RecordLock():
        try:
            with open(runs_path(), encoding="utf-8") as fh:
                rows = json.load(fh)
        except (OSError, ValueError):
            rows = []
        rows = [r for r in rows if isinstance(r, dict) and r.get("nonce") != row.get("nonce")] + [row]
        tmp = runs_path() + ".tmp"
        from mif_roblox import durable
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rows[-RUNS_KEEP:], fh, indent=1)
            durable.sync(fh)
        os.replace(tmp, runs_path())


def find_run(port, which="last"):
    try:
        with open(runs_path(), encoding="utf-8") as fh:
            rows = [r for r in json.load(fh) if isinstance(r, dict) and int(r.get("port") or 0) == int(port)]
    except (OSError, ValueError):
        return None
    if not rows:
        return None
    if which == "last":
        return rows[-1]
    return next((r for r in reversed(rows) if r.get("nonce") == which or r.get("requestId") == which), None)


def session_report(logs, t_start, alive=None):
    """What the play test's own logs say, for a probe that got no result: the server's and each client's script output
    (seconds since the probe started), errors, who joined and left, and when and how each process ended - in the order
    they ended. alive (pids still running): a log whose process still runs is "still running", not ended - its log was
    only read early (RB2929: a cut run's answer said ended while both processes ran 90 s more). A disconnect in the server
    that is not its own client ending is the cut (cutAt, RB2928)."""
    def rel(t):
        return None if t is None else round(t - t_start, 2)
    alive = set(alive or ())
    rep = {"server": None, "clients": [], "endOrder": []}
    client_ends = [lg["last"] for lg in logs if lg["role"] == "client" and lg.get("last") is not None
                   and lg.get("pid") not in alive]
    client_ends += [e["utc"] for lg in logs if lg["role"] == "server" for e in lg["events"] if e["event"] == "left"]
    for i, lg in enumerate(logs):
        ev = [dict(e, t=rel(e.pop("utc"))) for e in (dict(x) for x in lg["events"])]
        names = {e["event"] for e in ev}
        running = lg.get("pid") is not None and lg.get("pid") in alive
        how = ("still running (pid %s): its log was read before it ended" % lg["pid"] if running
               else "normal exit" if "normalExit" in names else "no exit line: the process was ended, or crashed")
        row = {"file": lg["file"], "started": rel(lg["first"]), "ended": None if running else rel(lg["last"]),
               "lastLine": rel(lg["last"]), "endedHow": how, "running": running, "pid": lg.get("pid"),
               "events": ev, "truncated": lg["truncated"],
               "output": [{"message": o["message"], "type": o["type"], "t": rel(o["utc"])} for o in lg["output"]
                          if not o["message"].startswith(("MIFPROBE_RESULT ", "MIFPROBE_SAMPLE "))]}
        row["errors"] = [o for o in row["output"] if o["type"] == "MessageError"]
        if lg["role"] == "server" and rep["server"] is None:
            # the cut: a disconnect in the server more than 2 s from any of its own clients ending or leaving (an own
            # client's leave logs the same line and costs nothing, RB2922)
            # and before the server's own end (its reply, or the reply's time-out, then its own shutdown's disconnect)
            own_end = min((e["utc"] for e in lg["events"] if e["event"] in ("resultSent", "resultTimedOut", "normalExit")),
                          default=None)
            cuts = [e["utc"] for e in lg["events"] if e["event"] == "uiDisconnected"
                    and (own_end is None or e["utc"] < own_end)
                    and not any(abs(e["utc"] - c) <= 2.0 for c in client_ends)]
            if cuts:
                row["cutAt"] = rel(cuts[0])
            rep["server"] = row
        else:
            row["role"] = lg["role"]
            rep["clients"].append(row)
        rep["endOrder"].append({"role": lg["role"] if lg["role"] == "server" else "client %d" % len(rep["clients"]),
                                "ended": row["ended"], "endedHow": row["endedHow"]})
    rep["endOrder"].sort(key=lambda r: r["ended"] if r["ended"] is not None else 1e9)
    return rep
