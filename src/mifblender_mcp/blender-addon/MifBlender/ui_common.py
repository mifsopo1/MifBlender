"""What the sidebar panel and the viewport HUD share: the palette, the colour chips, op families, the call data, and
the health checks - computed OFF the draw path.

WHY THIS SHAPE, 2026-09-21. The sidebar and HUD were chosen from four mocked directions: health checks on top,
calls grouped by family below, the HUD as a toggle, "as polished as physically possible without GPU lag". Blender
redraws a panel on every mouse move over it, so nothing in a draw() may cost more than reading a dict: every check
that touches the disk or a socket runs from a timer here and the draw code reads its cached answer.

THE UI READS, IT NEVER STEERS. Like the UE panel (MifBridgePanel.cpp - "this must never become load-bearing"), the
server and the ops do not depend on anything in the ui_* modules. Delete them and the bridge is unchanged.

THE CALLS COME FROM journal.py (the call journal, built alongside), through three functions this module asks for by
name: journal.recent(limit), journal.version(), journal.path(). If the journal is absent or raises, the UI shows its
empty state rather than failing - a panel that throws draws nothing at all.
"""
import os
import socket
import tempfile
import threading
import time

import bpy

# ---------------------------------------------------------------- the UE panel's palette (MifBridgePanel.cpp, MifStyle)
HEX = {
    "purple": "8B5CF6", "purpleSoft": "C4B5FD", "railL": "2A1A52", "railR": "6D28F5",
    "read": "7C5CFF", "write": "FF8A3D", "refused": "FF4D6A", "ok": "34D399", "warn": "FBBF24", "running": "4FDD8C",
    "ground": "141118", "plate": "1C1824", "plateHi": "221A38", "line": "2F2839",
    "ink": "EFEAF6", "ink2": "A297B5", "ink3": "6F6580",
    "violet": "7B34D6", "teal": "1D8F86", "amber": "C2782A", "rose": "B53A63", "slate": "4B4568",
    "blue": "4FADF5", "stopped": "F1666D",
}


def rgba(key, alpha=1.0):
    h = HEX[key]
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0, alpha)


# ---------------------------------------------------------------- op families: the module each op lives in
FAMILY_OF_MODULE = {
    "ops_scene": "Scene", "ops_mesh": "Mesh", "ops_gen": "Generate", "ops_rig": "Rig & Modifiers",
    "ops_create": "Create", "ops_material": "Materials", "ops_lightcam": "Lights & Cameras",
    "ops_anim": "Animation", "ops_render": "Render", "ops_world": "World", "ops_physics": "Physics",
    "ops_particles": "Particles", "ops_nodes": "Nodes", "ops_viewport": "Viewport", "ops_file": "File",
    "ops_constraint": "Constraints", "ops_collection": "Collections", "ops_viewlayer": "View Layers",
    "ops_io": "Import & Export", "ops_query": "Query", "ops_describe": "Describe", "ops_undo": "Undo",
    "ops_audio": "Audio", "ops_tree": "Create",
    "ops_nature": "Create",
    "ops_kit": "Create",
    "ops_topology": "Mesh",
    "ops_image": "Materials",
    "ops_uv": "Mesh",
    "ops_pose": "Animation",
    "ops_rigify": "Rig & Modifiers",
    "ops_motion": "Animation",
    "ops_garment": "Garments",
    "ops_body": "Garments",
    "ops_video": "Video",
}
FAMILY_COLOR = {  # the UE plates' five-colour cycle, assigned so neighbours in the list tend to differ
    "Scene": "slate", "Mesh": "violet", "Generate": "teal", "Rig & Modifiers": "teal", "Create": "violet",
    "Materials": "rose", "Lights & Cameras": "amber", "Animation": "amber", "Render": "violet", "World": "teal",
    "Physics": "amber", "Particles": "rose", "Nodes": "teal", "Viewport": "slate", "File": "slate",
    "Constraints": "rose", "Collections": "slate", "View Layers": "slate", "Import & Export": "teal",
    "Query": "slate", "Describe": "slate", "Undo": "slate", "Audio": "rose", "Garments": "rose", "Video": "amber", "Batch": "violet", "Other": "slate",
}
_FAMILY_CACHE = {}


def family_of(op):
    fam = _FAMILY_CACHE.get(op)
    if fam is None:
        fam = "Batch" if op == "batch" else "Other"
        try:
            from . import server
            fn = server._op_table().get(op) if fam == "Other" else None
            if fn is not None:
                fam = FAMILY_OF_MODULE.get(fn.__module__.rsplit(".", 1)[-1], "Other")
        except Exception:   # noqa: BLE001 - a UI label must never raise
            pass
        _FAMILY_CACHE[op] = fam
    return fam


def is_read_only(op):
    try:
        from .ops_undo import READ_ONLY_OPS
        return op in READ_ONLY_OPS
    except Exception:   # noqa: BLE001
        return False


def kind_of(call):
    """read / write / refused / running - the colour a call is drawn in."""
    ok = call.get("ok")
    if ok is None:
        return "running"
    if ok is False:
        return "refused"
    return "read" if is_read_only(call.get("op", "")) else "write"


SLOW_MS = 1000.0


def ms_text(ms):
    if ms is None:
        return "..."
    return ("%.2f s" % (ms / 1000.0)) if ms >= 1000.0 else ("%d ms" % int(round(ms)))


def ago_text(t):
    if not t:
        return ""
    d = max(0.0, time.time() - float(t))
    if d < 60:
        return "%ds ago" % int(d)
    if d < 3600:
        return "%dm ago" % int(d // 60)
    return "%dh ago" % int(d // 3600)


# ---------------------------------------------------------------- colour chips as custom icons
_PREVIEWS = None
_CHIP_KEYS = ("read", "write", "refused", "running", "ok", "warn", "purple", "stopped",
              "violet", "teal", "amber", "rose", "slate")


def _write_chip_png(path, rgb, size=32, radius=8):
    import struct
    import zlib
    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            dx = max(radius - x, x - (size - 1 - radius), 0)
            dy = max(radius - y, y - (size - 1 - radius), 0)
            d2 = dx * dx + dy * dy
            a = 255 if d2 <= (radius - 1) ** 2 else (128 if d2 <= radius * radius else 0)   # a soft edge
            row += bytes((rgb[0], rgb[1], rgb[2], a))
        rows.append(bytes(row))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(png)


def chip(key):
    """icon_value for a colour chip, or 0 (no icon) before register or in a Blender without previews."""
    try:
        return _PREVIEWS[key].icon_id
    except Exception:   # noqa: BLE001
        return 0


def _load_chips():
    global _PREVIEWS
    import bpy.utils.previews
    folder = tempfile.mkdtemp(prefix="mifblender_icons_")
    _PREVIEWS = bpy.utils.previews.new()
    for key in _CHIP_KEYS:
        h = HEX[key]
        path = os.path.join(folder, "%s.png" % key)
        _write_chip_png(path, (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)))
        _PREVIEWS.load(key, path, "IMAGE")


def _free_chips():
    global _PREVIEWS
    if _PREVIEWS is not None:
        import bpy.utils.previews
        try:
            bpy.utils.previews.remove(_PREVIEWS)
        except Exception:   # noqa: BLE001
            pass
        _PREVIEWS = None


# ---------------------------------------------------------------- the calls, cached per journal version
def _journal():
    try:
        from . import journal
        return journal
    except Exception:   # noqa: BLE001 - the journal not being there yet is the empty state, not an error
        return None


def journal_version():
    j = _journal()
    try:
        return int(j.version()) if j is not None else 0
    except Exception:   # noqa: BLE001
        return 0


def journal_path():
    j = _journal()
    try:
        return j.path() if j is not None else None
    except Exception:   # noqa: BLE001
        return None


_CALLS = {"version": None, "calls": [], "status": {}}


def recent_calls(limit=None):
    """The journal's ring, newest first, fetched once per journal version. The whole ring is fetched (64 rows) and
    sliced here, so every caller shares one cached copy whatever limit it asks for."""
    v = journal_version()
    if v != _CALLS["version"]:
        j = _journal()
        try:
            _CALLS["calls"] = list(j.recent(None)) if j is not None else []
        except Exception:   # noqa: BLE001
            _CALLS["calls"] = []
        try:
            _CALLS["status"] = dict(j.status()) if j is not None else {}
        except Exception:   # noqa: BLE001
            _CALLS["status"] = {}
        _CALLS["version"] = v
    calls = _CALLS["calls"]
    return calls if limit is None else calls[:limit]


def journal_status():
    """journal.status() as of the last journal version the UI saw: path, diskEnabled, diskError, totalCalls..."""
    recent_calls()
    return _CALLS["status"]


def session_stats(calls):
    """The numbers the sidebar and the HUD show. A batch whose ops are in the list is counted through its ops, not
    twice; `total` is every call this Blender has journaled, `window` how many of them the ring still holds."""
    parents = {c.get("batch") for c in calls if c.get("batch") is not None}
    leaf = [c for c in calls if c.get("seq") not in parents]
    done = [c for c in leaf if c.get("ms") is not None]
    slow = max(done, key=lambda c: c.get("ms") or 0.0) if done else None
    total = journal_status().get("totalCalls")
    agents = {}
    for c in leaf:
        if c.get("agent"):
            agents[c["agent"]] = agents.get(c["agent"], 0) + 1
    return {"calls": len(leaf), "total": total if isinstance(total, int) else len(calls), "window": len(calls),
            "agents": sorted(agents.items(), key=lambda kv: (-kv[1], kv[0])),
            "errors": sum(1 for c in leaf if c.get("ok") is False),
            "writes": sum(1 for c in leaf if c.get("ok") and not is_read_only(c.get("op", ""))),
            "slowest": slow}


# ---------------------------------------------------------------- health checks, from a timer, never from draw()
# Other Blender MCP add-ons, by the module names they install under, and the port the popular one listens on.
OTHER_MCP_MODULES = ("blender_mcp", "blendermcp", "blender-mcp", "mcp_blender", "addon_blender_mcp")
OTHER_MCP_PORT = 9876
CHECKS = {"at": 0.0, "slow_at": 0.0, "rows": [], "install": None, "other_addons": [], "port_9876": None}


def _addon():
    """The MifBlender package module itself - where _server, source_hash and SOURCE_HASH_AT_IMPORT live."""
    import sys
    return sys.modules.get(__package__)


def _prefs():
    try:
        addon = bpy.context.preferences.addons.get(__package__)
        return addon.preferences if addon else None
    except Exception:   # noqa: BLE001
        return None


def other_mcp_addons(names, lookup=None):
    """The add-ons in `names` that are another Blender MCP - by module name, or by what the add-on calls itself.

    BY ITS OWN NAME TOO, 2026-09-21: one popular Blender MCP add-on ships as one file, addon.py, so it installs as a module
    called "addon", which no module-name list would match - found in this machine's Blender 5.0 add-ons folder.
    Its bl_info names it "Blender MCP". lookup(name) returns the module; sys.modules.get by default."""
    import sys
    lookup = lookup or sys.modules.get
    found = []
    for name in names:
        if name == __package__:
            continue
        info = getattr(lookup(name), "bl_info", None) or {}
        label = str(info.get("name", "") if isinstance(info, dict) else "")
        if any(m in name.lower() for m in OTHER_MCP_MODULES) or "mcp" in label.lower():
            found.append(label or name)
    return found


def _probe_port(port, timeout=0.05):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex(("127.0.0.1", int(port))) == 0
    except OSError:
        return False
    finally:
        s.close()


_PROBE = {"thread": None}


def _probe_async(port):
    """Probe `port` on a daemon thread. A connect to a closed port on Windows waits out its whole timeout - measured
    74 ms for the slow checks with the probe inline, most of it this - and on the main thread that is a stutter
    every 30 s. The answer lands in CHECKS, and the next pass of the checks reads it."""
    t = _PROBE["thread"]
    if t is not None and t.is_alive():
        return

    def run():
        CHECKS["port_9876"] = _probe_port(port, timeout=0.5)
    t = threading.Thread(target=run, name="MifBlender-port-probe", daemon=True)
    _PROBE["thread"] = t
    t.start()


def health_checks(refresh_slow=False):
    """The checks the sidebar and the HUD show - ONE list for both, so they cannot disagree. (No op returns it: ping
    already reports the install check, the source hashes and runPythonAllowed from the same sources.)

    Each row: {"id", "status": ok | warn | fail | info, "text", "detail", "fix"}. refresh_slow re-reads what touches
    the disk or a socket (the install manifest, a port probe); otherwise their last answer is reused.
    """
    pkg = _addon()
    settings = _prefs()
    rows = []
    server = getattr(pkg, "_server", None) if pkg else None
    running = bool(server is not None and getattr(server, "running", False))
    port = getattr(server, "port", None) or (settings.port if settings else None)
    rows.append({"id": "server", "status": "ok" if running else "fail",
                 "text": ("Listening on :%s" % port) if running else "Server stopped",
                 "detail": ("pid %d, %d client(s)" % (os.getpid(), getattr(server, "_clients", 0))) if running
                 else "Start it here, or turn on auto-start in the preferences.", "fix": None if running else "start"})
    token = (server.token if running else (settings.token if settings else "")) or ""
    if not token:
        rows.append({"id": "token", "status": "fail", "text": "No token: any local process can drive this",
                     "detail": "Set a token in the add-on preferences and the same one on the MCP server.",
                     "fix": "prefs"})
    elif token == "dev":
        rows.append({"id": "token", "status": "info", "text": "Default token · loopback only",
                     "detail": "The token is MifBridge's shared default, 'dev': fine on a machine only you use. If other "
                               "people's code runs here, set your own in the preferences and the same one as "
                               "MIF_BLENDER_TOKEN on the MCP server.", "fix": "prefs"})
    else:
        rows.append({"id": "token", "status": "ok", "text": "Own token · loopback only",
                     "detail": "127.0.0.1 only, and every request must carry your token.", "fix": None})
    allow = bool(settings.allow_run_python) if settings else True
    rows.append({"id": "run_python", "status": "info",
                 "text": "run_python allowed" if allow else "run_python blocked",
                 "detail": "The escape hatch that runs arbitrary Python. Untick it in the preferences if you do not "
                           "need it." if allow else "Scripts sent by a client are refused.", "fix": "prefs"})
    # THE GENERATIVE MODE, BOTH SWITCHES THIS SIDE HOLDS (ops_gen.py): the preference, for every file, and this file's
    # project mode, the user's choice. A project whose mode is not chosen works as generative and says so.
    from . import ops_gen
    by, mode = ops_gen.off_by(), ops_gen.file_mode()
    if by == "preference":
        text = "Non-generative: gen_* off (preference)"
        detail = ("The gen_* ops are refused in every file and nothing contacts ComfyUI. Tick 'Allow generative "
                  "tools (ComfyUI)' in the preferences to allow them.")
    elif mode == "non-generative":
        text = "Non-generative project: gen_* off"
        detail = ("This file's project mode is non-generative: the gen_* ops are refused and nothing contacts "
                  "ComfyUI. Its user chose it; the assistant records a change with mif_project_mode.")
    elif mode == "generative":
        text = "Generative project: gen_* on"
        detail = ("The gen_* ops may run Flux and Hunyuan3D-2 through your ComfyUI. MIF_GENERATIVE=off on the MCP "
                  "server, the preference, or a non-generative mode turns them off.")
    else:
        text = "Generative: project mode not chosen"
        detail = ("This file works as generative until its mode is chosen; the assistant asks you on its first call "
                  "and records it in the file (save it to keep it).")
    rows.append({"id": "generative", "status": "info", "text": text, "detail": detail, "fix": "prefs"})
    now = time.time()
    if refresh_slow or now - CHECKS["slow_at"] > 30.0 or CHECKS["install"] is None:
        CHECKS["slow_at"] = now
        try:
            from . import install_check
            CHECKS["install"] = install_check.check(os.path.dirname(os.path.abspath(__file__)))
        except Exception as exc:   # noqa: BLE001
            CHECKS["install"] = {"error": str(exc)}
        try:
            found = other_mcp_addons(list(bpy.context.preferences.addons.keys()))
        except Exception:   # noqa: BLE001
            found = []
        CHECKS["other_addons"] = found
        if port and int(port) == OTHER_MCP_PORT:
            CHECKS["port_9876"] = None          # that is this server's own port
        else:
            _probe_async(OTHER_MCP_PORT)
    # install_check.check's own shape: duplicates, manifestPresent, missing / changed / unexpected, manifestError, note.
    inst = CHECKS["install"] or {}
    dups = inst.get("duplicates") or []
    if inst.get("error"):
        rows.append({"id": "install", "status": "warn", "text": "Install check failed", "detail": inst["error"],
                     "fix": None})
    elif dups:
        rows.append({"id": "install", "status": "fail", "text": "%d duplicate MifBlender folder(s)" % len(dups),
                     "detail": inst.get("note") or ", ".join(dups[:3]), "fix": "folder"})
    elif inst.get("manifestError"):
        rows.append({"id": "install", "status": "warn", "text": "Release manifest unreadable",
                     "detail": inst.get("note") or inst["manifestError"], "fix": "folder"})
    elif not inst.get("manifestPresent"):
        rows.append({"id": "install", "status": "info", "text": "Source checkout (no release manifest)",
                     "detail": "Installed from a folder, not a release zip, so there is nothing to compare against.",
                     "fix": None})
    else:
        bad = sum(len(inst.get(k) or []) for k in ("missing", "changed", "unexpected"))
        rows.append({"id": "install", "status": "ok" if not bad else "fail",
                     "text": "Install matches its release" if not bad else "Install differs from its release (%d)" % bad,
                     "detail": inst.get("note") or "Every packaged file has the bytes the release shipped.",
                     "fix": None if not bad else "folder"})
    stale = _sources_changed(pkg)
    rows.append({"id": "freshness", "status": "warn" if stale else "ok",
                 "text": "Code changed since Blender started" if stale else "Running the code on disk",
                 "detail": "Restart Blender to load it." if stale else "No edits since this Blender imported it.",
                 "fix": None})
    comp = CHECKS["other_addons"]
    p9876 = CHECKS["port_9876"]
    if comp or p9876:
        what = (", ".join(comp) if comp else "") + ((" · " if comp else "") + ":%d answers" % OTHER_MCP_PORT
                                                    if p9876 else "")
        rows.append({"id": "other_mcp", "status": "warn", "text": "Another Blender MCP is running",
                     "detail": "%s - an AI can end up working through it instead of MifBlender." % what,
                     "fix": "addons"})
    else:
        rows.append({"id": "other_mcp", "status": "ok", "text": "No other Blender MCP found",
                     "detail": "No known MCP add-on enabled, nothing on :%d." % OTHER_MCP_PORT, "fix": None})
    jp = journal_path()
    disk_error = journal_status().get("diskError")
    if jp and disk_error:
        rows.append({"id": "journal", "status": "warn", "text": "Call journal cannot write",
                     "detail": "%s - calls are still listed here, but not kept on disk. File: %s" % (disk_error, jp),
                     "fix": "journal"})
    else:
        rows.append({"id": "journal", "status": "ok" if jp else "info",
                     "text": "Call journal on" if jp else "Call journal off (list only)",
                     "detail": jp or "The last calls are still listed here; turn the journal on in the preferences to "
                                     "keep every call on disk, where a crash cannot take it.",
                     "fix": "journal" if jp else "prefs"})
    CHECKS["rows"] = rows
    CHECKS["at"] = now
    return rows


_SOURCES = {"sig": None, "stale": False}


def _sources_changed(pkg):
    """pkg.source_hash() differs from the hash taken at import - but hashing every module every five seconds on the
    main thread is a hitch, so the hash is recomputed only when a size or mtime in the folder moved."""
    if pkg is None:
        return False
    try:
        folder = pkg.PACKAGE_DIR
        sig = tuple(sorted((e.name, e.stat().st_size, e.stat().st_mtime_ns) for e in os.scandir(folder)
                           if e.name.endswith(".py")))
    except Exception:   # noqa: BLE001
        return _SOURCES["stale"]
    if sig != _SOURCES["sig"]:
        _SOURCES["sig"] = sig
        try:
            _SOURCES["stale"] = pkg.source_hash() != pkg.SOURCE_HASH_AT_IMPORT
        except Exception:   # noqa: BLE001
            _SOURCES["stale"] = False
    return _SOURCES["stale"]


def cached_checks():
    return CHECKS["rows"] or health_checks()


# ---------------------------------------------------------------- redraw only when something changed
_TICK = {"version": None, "checked": 0.0, "fade_until": 0.0, "changed_at": 0.0, "idle": None}


def touch_activity():
    """Count now as activity - the HUD being switched on, or a setting changed - so it opens rather than showing a
    collapsed strip to someone who just asked to see it."""
    _TICK["changed_at"] = time.time()


def hud_idle(settings=None):
    """True when the HUD should be the collapsed strip: auto-collapse is on and no call has started or finished for
    hud_idle_seconds. The next call opens it again."""
    s = settings if settings is not None else _prefs()
    if s is None or not getattr(s, "hud_auto_collapse", True):
        return False
    return time.time() - _TICK["changed_at"] > float(getattr(s, "hud_idle_seconds", 10))


def tag_redraw_views():
    wm = bpy.context.window_manager
    for win in getattr(wm, "windows", []):
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _tick():
    """0.25 s: redraw the 3D views when the journal moved (and while the newest HUD card fades in); every 5 s, re-run
    the cheap checks and redraw if a status changed. Costs a dict read when nothing happened."""
    try:
        now = time.time()
        v = journal_version()
        redraw = False
        if v != _TICK["version"]:
            if _TICK["version"] is not None:
                _TICK["fade_until"] = now + 1.2
                _TICK["changed_at"] = now
            _TICK["version"] = v
            redraw = True
        idle = hud_idle()
        if idle != _TICK["idle"]:          # the HUD collapses on a deadline, not on an event: redraw when it passes
            _TICK["idle"] = idle
            redraw = True
        if now < _TICK["fade_until"]:
            redraw = True
        if now - _TICK["checked"] > 5.0:
            before = [(r["id"], r["status"], r["text"]) for r in CHECKS["rows"]]
            health_checks()
            _TICK["checked"] = now
            redraw = redraw or before != [(r["id"], r["status"], r["text"]) for r in CHECKS["rows"]]
        if redraw:
            tag_redraw_views()
    except Exception:   # noqa: BLE001 - a timer that raises is unregistered by Blender for good
        pass
    return 0.04 if time.time() < _TICK["fade_until"] else 0.25


def fade_amount():
    """1.0 right after the newest call landed, falling to 0.0 over 1.2 s - the HUD's highlight."""
    left = _TICK["fade_until"] - time.time()
    return max(0.0, min(1.0, left / 1.2))


def register():
    _load_chips()
    if not bpy.app.background and not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=0.5, persistent=True)


def unregister():
    try:
        if bpy.app.timers.is_registered(_tick):
            bpy.app.timers.unregister(_tick)
    except Exception:   # noqa: BLE001
        pass
    _free_chips()
