"""MifBlender -- the Blender backend of MifBridge.

MifBridge is one MCP server with two backends: an Unreal Engine 5.3-5.8 C++ plugin
over HTTP on 127.0.0.1:8791, and this Blender addon over a framed-JSON TCP
socket on 127.0.0.1:8792. Same tool namespace, same {ok:true,...} /
{ok:false,error:"..."} response shape on both sides.

  https://mifshowcase.store  (MifBlender add-on: MIT licence, (c) 2026 Mifsopo "Mif" and contributors)

This add-on carries its own MIT licence - see LICENSE beside this file. Since 2026-09-17 the
add-on is licensed this way because Blender's licence treats add-ons that use its Python API as
needing a GPL-compatible licence when published. The free MCP server has its own license.

The socket framing and the main-thread-marshalling pattern are adapted from
blender-mcp (github.com/MCPBlender/blender-mcp), (c) 2025 Siddharth Ahuja,
MIT licence; its full notice is reproduced in LICENSE beside this file.

Install: Edit > Preferences > Add-ons > Install..., pick the zip (or this
folder's parent), enable "MifBlender". The server auto-starts. Status and
controls live in the 3D viewport N-panel under the "MifBridge" tab.
"""

bl_info = {
    "name": "MifBlender (MifBridge backend)",
    "author": "Mifsopo",
    "version": (1, 3, 1),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar (N) > MifBridge",
    "description": "Framed-JSON socket backend so the MifBridge MCP server can drive "
                   "Blender alongside Unreal.",
    "warning": "Exposes a local socket that can import/export files and optionally run "
               "Python. Loopback only, token gated.",
    "category": "Import-Export",
}

# The 4.4 floor is now CONSERVATIVE rather than unverified. Measured 2026-08-27 with
# tools/blender_probe.py and tools/test_blender_ops.py, on four real installs:
#
#   version     imports  registers  33 ops  FBX kwargs  op suite
#   3.6.23        yes      yes        yes    all present   12/12
#   4.2.17 LTS    yes      yes        yes    all present   12/12
#   4.4.0         yes      yes        yes    all present   12/12
#   5.0.1         yes      yes        yes    all present   12/12
#
# The worry this comment used to carry was real but did not bite: the exporter's
# properties DO move between releases -- use_ascii is gone in 4.4 -- but that is one
# this addon never passes. All 17 kwargs in FBX_EXPORT_ARGS, all 3 in FBX_IMPORT_ARGS,
# the 4 enum values (FBX_SCALE_NONE / FACE / SRGB / AUTO) and all 6 bmesh.ops are still
# real on 5.0.1.
#
# Blender 5.0 still ships 25 legacy bl_info addons of its own and still has
# addon_utils.enable, so the extensions manifest is NOT required. Determined by running
# it, not by reading release notes.
#
# The floor stays at 4.4 anyway, and that is a decision rather than an oversight: the op
# SUITE covers set_material_slots and the read ops around it, not the FBX mesh round
# trip. The kwargs are proven present on 3.6; the round trip through them is not proven
# on 3.6. Lower the floor when something exercises it.

import importlib
import os
import sys

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import AddonPreferences, Operator

# Reload submodules cleanly when the addon is re-enabled or hot-reloaded, otherwise Blender keeps
# the stale bytecode and edits appear not to apply.
#
# READ FROM DISK, because the hand-kept tuple this replaces had rotted and the failure it caused is
# the one the comment above describes. It named 17 of the 23 modules. The six it had missed -
# ops_collection, ops_create, ops_io, ops_material, ops_query, ops_viewlayer - carry 40 ops between
# them, and every edit to any of them survived a hot-reload as old bytecode. ops_material is all ten
# material ops; ops_create is eleven of the creators. Whoever added those six modules had no reason
# to know a list somewhere else needed the name too, which is the whole problem with the list.
#
# ORDER IS DELIBERATE, not alphabetical: framing and ops_common are imported BY the op modules, and
# server is imported by everything, so they reload at the ends. The middle is sorted so the sequence
# is the same on every machine and every run.
def _submodules():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        found = sorted(f[:-3] for f in os.listdir(here)
                       if f.endswith(".py") and f != "__init__.py")
    except OSError:
        return ()
    head = [m for m in ("framing", "ops_common") if m in found]
    tail = [m for m in ("server",) if m in found]
    return tuple(head + [m for m in found if m not in head and m not in tail] + tail)


_SUBMODULES = _submodules()
for _name in _SUBMODULES:
    _full = "%s.%s" % (__name__, _name)
    if _full in sys.modules:
        importlib.reload(sys.modules[_full])


# ---------------------------------------------------------------------------
# WHICH COPY OF THIS ADDON AM I RUNNING?
#
# Blender loads the addon from its scripts/addons directory, NOT from wherever you are editing it.
# A running Blender keeps the modules it imported at startup. Those two facts together mean an
# answer from this server can come from code that is days older than the file open in your editor,
# and nothing in the protocol said so: ping reported an addonVersion, which is a hand-written tuple
# that does not change when a module does, and an op list, which only differs when an op is ADDED.
# A changed BEHAVIOUR was invisible.
#
# Measured 2026-09-07: four installs on this machine, every one of the 24 modules stale, three of
# them six days behind the repo. Work had been verified against the wrong code and looked fine.
#
# So ping now answers it directly, in three parts, because they fail differently:
#   addonPath           WHERE the running package was loaded from. Compare it with your checkout.
#   sourceHash          what is on disk at that path RIGHT NOW.
#   sourceHashAtImport  what was there when this process imported it.
# The first catches the wrong copy; the last two differing catches the right copy edited since
# startup, which needs a restart rather than a sync.
# ---------------------------------------------------------------------------

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def source_hash(pkg_dir=None):
    """sha256 over this package's .py sources, or None if they cannot be read.

    Hashes the FILENAME as well as the bytes, so that adding or removing a module changes the
    result even in the unlikely case the remaining bytes are unchanged. Returns None rather than
    raising: an unreadable directory is a worse answer than no answer, but it must not stop ping,
    which is the op people reach for when something is already wrong.
    """
    import hashlib
    d = pkg_dir or PACKAGE_DIR
    try:
        names = sorted(f for f in os.listdir(d) if f.endswith(".py"))
        h = hashlib.sha256()
        for f in names:
            h.update(f.encode("utf-8"))
            with open(os.path.join(d, f), "rb") as fh:
                h.update(fh.read())
        return h.hexdigest()[:16]
    except OSError:
        return None


SOURCE_HASH_AT_IMPORT = source_hash()

from . import server as mif_server  # noqa: E402
from . import journal as mif_journal  # noqa: E402

# Module-level server handle -- do NOT hang this on bpy.types (breaks theme presets).
_server = None


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------

def _hud_changed(self, context):
    from . import ui_hud  # local: the preferences class is built before the UI modules register
    ui_hud.on_setting_changed()


class MifBlenderPreferences(AddonPreferences):
    bl_idname = __name__

    port: IntProperty(
        name="Port",
        description="TCP port on 127.0.0.1. The Unreal side of MifBridge uses 8791; "
                    "the unrelated third-party blender-mcp addon uses 9876",
        default=mif_server.DEFAULT_PORT, min=1024, max=65535,
    )
    token: StringProperty(
        name="Token",
        description="Shared secret. Must match MIF_BLENDER_TOKEN on the MCP server "
                    "(which itself defaults to MIF_BRIDGE_TOKEN). Leaving it EMPTY "
                    "disables authentication entirely",
        default="dev", subtype="PASSWORD",
    )
    auto_start: BoolProperty(
        name="Start server automatically",
        description="Start listening as soon as the addon is enabled and on Blender "
                    "launch. Ignored in background mode, where starting would block "
                    "startup forever",
        default=True,
    )
    # OPT-IN (artifacts.py, docs/35_ARTIFACTS.md): off, the add-on keeps no list of what it wrote
    record_artifacts: BoolProperty(
        name="Record artifacts",
        description="Keep a list of the files MifBlender's ops write, beside the call journal, so artifact_inventory can "
                    "later tell a tool's output from a file nobody can account for. It lists and never removes",
        default=False,
    )
    allow_run_python: BoolProperty(
        name="Allow run_python (arbitrary code execution)",
        description="Lets the run_python op exec arbitrary Python inside Blender with "
                    "your user's privileges. The socket is loopback-only and token "
                    "gated, but turn this off if you do not need the escape hatch",
        default=True,
    )
    # GENERATIVE BY DEFAULT, A SWITCH YOU CAN SEE (ops_gen.py, 2026-10-01). The gen_* ops are the only ones that
    # run a trained model; off here, each is refused before it reads a parameter or contacts ComfyUI, in every file.
    # Each file has its own mode as well (ops_gen.project_mode), and the MCP server its own switch (MIF_GENERATIVE=off).
    allow_generative: BoolProperty(
        name="Allow generative tools (ComfyUI)",
        description="Lets the gen_* ops run generative models (Flux, Hunyuan3D-2) through a ComfyUI you run "
                    "yourself, in files whose project mode allows it. Off, this Blender is non-generative: those "
                    "ops are refused and nothing contacts ComfyUI",
        default=True,
    )
    job_timeout: IntProperty(
        name="Main-thread job timeout (s)",
        description="How long a request waits for Blender's main thread before it is "
                    "answered with an error. Long bakes and modal operators block the "
                    "main thread. KEEP THIS BELOW the MCP server's work timeout "
                    "(MIF_BLENDER_TIMEOUT, default 180s): whichever end gives up first "
                    "owns the failure, and it should be this one, so the socket carries "
                    "a real error instead of the MCP abandoning a job Blender then goes "
                    "on running against a scene the caller believes untouched",
        default=int(mif_server.DEFAULT_JOB_TIMEOUT), min=5, max=86400,
    )
    verbose: BoolProperty(
        name="Log to console",
        description="Print connection and error lines to Blender's system console",
        default=True,
    )
    # THE CALL JOURNAL'S DISK HALF (journal.py). On by default, as the UE journal is: a crash journal you must turn on
    # before the crash is off when it matters. Off keeps the in-memory list of recent calls.
    journal_to_disk: BoolProperty(
        name="Write the call journal to disk",
        description="Record every bridge call in a per-user file, written before the call runs so a crash still "
                    "names it. Off keeps only the in-memory list of recent calls",
        default=True,
    )

    # THE VIEWPORT HUD (ui_hud.py). Off by default: an overlay nobody asked for is clutter. The sidebar's HUD button
    # and the Overlays popover both flip hud_enabled; the rest are how it looks.
    hud_enabled: BoolProperty(
        name="Viewport HUD",
        description="Draw the bridge's last calls over the 3D viewport",
        default=False, update=_hud_changed,
    )
    hud_corner: EnumProperty(
        name="Corner",
        description="Which corner of the 3D viewport the HUD sits in",
        items=[("TOP_LEFT", "Top left", "Under the view's own text"),
               ("TOP_RIGHT", "Top right", "Beside the navigation gizmo"),
               ("BOTTOM_LEFT", "Bottom left", "Where the Adjust Last Operation panel also appears"),
               ("BOTTOM_RIGHT", "Bottom right", "The emptiest corner of a default layout")],
        default="TOP_LEFT", update=_hud_changed,
    )
    hud_max_cards: IntProperty(
        name="Calls shown",
        description="The most calls the HUD lists; fewer when the view is too short to hold them",
        default=3, min=1, max=20, update=_hud_changed,
    )
    hud_auto_collapse: BoolProperty(
        name="Collapse when idle",
        description="Shrink the HUD to a one-line strip when no call has run for a while; the next call opens it "
                    "again. It covers the viewport only while the AI is working",
        default=True, update=_hud_changed,
    )
    hud_idle_seconds: IntProperty(
        name="Idle after (s)",
        description="How long without a call before the HUD collapses",
        default=10, min=2, max=600, update=_hud_changed,
    )
    hud_opacity: FloatProperty(
        name="Opacity",
        description="How opaque the HUD's background is. The text stays fully opaque",
        default=0.92, min=0.3, max=1.0, subtype="FACTOR", update=_hud_changed,
    )
    hud_show_summary: BoolProperty(
        name="Show what each call sent",
        description="A second line on each call: its family and parameters, or why it was refused",
        default=True, update=_hud_changed,
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.prop(self, "port")
        col.prop(self, "token")
        col.prop(self, "job_timeout")
        layout.prop(self, "auto_start")
        layout.prop(self, "verbose")
        layout.prop(self, "journal_to_disk")
        if self.journal_to_disk:
            layout.label(text="Call journal: %s" % mif_journal.journal_path())
        layout.prop(self, "record_artifacts")

        box = layout.box()
        box.prop(self, "allow_run_python")
        if self.allow_run_python:
            box.label(text="run_python can execute any code the MCP client sends.",
                      icon="ERROR")
        box.prop(self, "allow_generative")
        box.label(text="gen_* ops may run generative models where the file's project mode allows it."
                  if self.allow_generative else "Non-generative: the gen_* ops are refused in every file.",
                  icon="INFO")
        if not self.token:
            layout.label(text="Token is empty: any local process can drive Blender.",
                         icon="ERROR")

        box = layout.box()
        box.prop(self, "hud_enabled")
        col = box.column()
        col.active = self.hud_enabled
        col.prop(self, "hud_corner")
        col.prop(self, "hud_max_cards")
        col.prop(self, "hud_auto_collapse")
        sub = col.row()
        sub.active = self.hud_auto_collapse
        sub.prop(self, "hud_idle_seconds")
        col.prop(self, "hud_opacity")
        col.prop(self, "hud_show_summary")

        row = layout.row(align=True)
        row.operator(MIFBLENDER_OT_start.bl_idname, icon="PLAY")
        row.operator(MIFBLENDER_OT_stop.bl_idname, icon="PAUSE")
        layout.label(text=_status_line())


def prefs(context=None):
    addon = (context or bpy.context).preferences.addons.get(__name__)
    return addon.preferences if addon else None


# ---------------------------------------------------------------------------
# Server control
# ---------------------------------------------------------------------------

def _status_line():
    if _server is None or not _server.running:
        return "MifBlender: stopped"
    # The pid is on the status line because it is the ONLY thing that
    # distinguishes two Blender windows from the outside. ping reports the pid
    # of whichever process actually owns the port; if it does not match the one
    # shown here, this Blender is not the one your edits are landing in.
    # SAID bl_status UNTIL 2026-09-04 - an endpoint that does not exist. A tool
    # built on that advice would have asked for it, caught the error and gone
    # quiet, which is worse than no advice at all.
    return "MifBlender: listening on %s:%d, pid %d (%d client%s)" % (
        _server.host, _server.port, os.getpid(), _server._clients,
        "" if _server._clients == 1 else "s")


def start_server(port=None, token=None, job_timeout=None, verbose=None):
    """Start (or restart) the socket server. Safe to call twice."""
    global _server
    settings = prefs()
    if _server is not None and _server.running:
        _server.stop()
    _server = mif_server.MifBlenderServer(
        port=port if port is not None else getattr(settings, "port", mif_server.DEFAULT_PORT),
        token=token if token is not None else getattr(settings, "token", "dev"),
        job_timeout=(job_timeout if job_timeout is not None
                     else getattr(settings, "job_timeout",
                                  mif_server.DEFAULT_JOB_TIMEOUT)),
        verbose=verbose if verbose is not None else getattr(settings, "verbose", True),
    )
    mif_server.ensure_timer()
    return _server.start()


def stop_server():
    global _server
    if _server is not None:
        _server.stop()
    _server = None


def serve_forever(port=None, token=None, job_timeout=None):
    """Headless entry point:  blender -b --python-expr "import MifBlender; MifBlender.serve_forever()"

    BLOCKS. In background mode there is no event loop, so the drain timer never
    fires and the server has to own the main thread. This is also exactly why
    the addon does not auto-start in background: register() would never return.
    """
    if not bpy.app.background:
        raise RuntimeError("serve_forever() is for `blender -b`. In the GUI the server "
                           "already runs on its own thread -- use start_server().")
    start_server(port=port, token=token, job_timeout=job_timeout)


class MIFBLENDER_OT_start(Operator):
    bl_idname = "mifblender.start_server"
    bl_label = "Start MifBlender Server"
    bl_description = "Begin listening for MifBridge MCP connections"

    def execute(self, context):
        if start_server():
            self.report({"INFO"}, _status_line())
            return {"FINISHED"}
        self.report({"ERROR"},
                    "could not bind the port -- see the system console for why")
        return {"CANCELLED"}


class MIFBLENDER_OT_stop(Operator):
    bl_idname = "mifblender.stop_server"
    bl_label = "Stop MifBlender Server"
    bl_description = "Close the listening socket"

    def execute(self, context):
        stop_server()
        self.report({"INFO"}, "MifBlender: stopped")
        return {"FINISHED"}


# THE SIDEBAR PANEL AND THE VIEWPORT HUD live in ui_panel.py and ui_hud.py, with what they share in ui_common.py
# (2026-09-21). They only read: nothing in the server or the ops imports them, so the bridge works without them.
_CLASSES = (
    MifBlenderPreferences,
    MIFBLENDER_OT_start,
    MIFBLENDER_OT_stop,
)
_UI_MODULES = ("ui_common", "ui_panel", "ui_hud")


def _ui_modules():
    return [importlib.import_module("%s.%s" % (__name__, name)) for name in _UI_MODULES]


def _deferred_autostart():
    """One-shot, main thread. Deferred so preferences are fully constructed
    before we read them -- reading them inside register() is a coin flip on a
    fresh enable."""
    settings = prefs()
    if settings is None or settings.auto_start:
        start_server()
    return None  # one-shot: do not re-arm


def register():
    # THE EXTENSION API (extensions.py, docs/32_EXTENSIONS.md section 8), under ONE fixed name: this add-on's own module
    # name is "MifBlender" when installed as a legacy add-on and "bl_ext.<repo>.MifBlender" as a 4.2+ extension, so a
    # separate add-on could not import it by a name it can know. First, so it is there even in background mode.
    from . import extensions as _ext
    sys.modules["mifblender_api"] = _ext
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    for module in _ui_modules():
        module.register()

    if bpy.app.background:
        # No event loop: the drain timer would never fire, and auto-starting
        # would block register() forever inside accept(). Headless callers must
        # call MifBlender.serve_forever() themselves.
        print("[MifBlender] background mode: not auto-starting. "
              "Call MifBlender.serve_forever() to serve on this thread.")
        return

    mif_server.ensure_timer()
    bpy.app.timers.register(_deferred_autostart, first_interval=0.1)


def unregister():
    from . import extensions as _ext
    if sys.modules.get("mifblender_api") is _ext:
        del sys.modules["mifblender_api"]
    stop_server()
    try:
        mif_journal.on_disable()      # a shutdown record, so a disabled add-on does not read as a Blender that died
    except Exception:
        pass
    try:
        if bpy.app.timers.is_registered(_deferred_autostart):
            bpy.app.timers.unregister(_deferred_autostart)
    except Exception:
        pass
    mif_server.remove_timer()
    for module in reversed(_ui_modules()):
        try:
            module.unregister()
        except Exception:
            pass
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
