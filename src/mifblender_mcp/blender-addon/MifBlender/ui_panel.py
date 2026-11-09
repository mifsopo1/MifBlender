"""The MifBlender sidebar (View3D > N > MifBridge): health on top, the calls grouped by family below, and everything else
the bridge knows about itself one fold away.

THE ORDER WAS CHOSEN 2026-09-21, from four mocked directions: "C on top, B below, D toggle". So the first thing on
screen answers "is this Blender set up right" (C), the second "what has the AI been doing here" (B), and the HUD (D,
ui_hud.py) is a button. Everything else - the session, the install, the settings - is a folded section, because the
sidebar is narrow and the brief was as much information as possible WITHOUT losing the two answers above the fold.

EVERY ROW IS CHEAP TO DRAW. Blender redraws a panel on every mouse move over it. Rows read ui_common's caches (the
checks run from a timer, the calls are cached per journal version) and never touch the disk or a socket. A row that
can say more than fits is a button that opens a popup with the rest - the full error, the full path, the fix.

Blender 3.6-4.0 has no collapsible layout panels (UILayout.panel arrived in 4.1), so there each section is a plain
box, always open. Everything else draws the same from 3.6 to 5.2.
"""
import os
import textwrap
import time

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty
from bpy.types import Operator, Panel

from . import ui_common as U

STATUS_ICON = {"ok": "CHECKMARK", "warn": "ERROR", "fail": "CANCEL", "info": "INFO"}


# ---------------------------------------------------------------- small helpers
def _section(layout, idname, title, default_closed=False, icon="NONE", right=None):
    """A collapsible section on 4.1+, a titled box before it. Returns the body layout, or None when folded."""
    if hasattr(layout, "panel"):
        header, body = layout.panel(idname, default_closed=default_closed)
        row = header.row(align=True)
        row.label(text=title, icon=icon)
        if right:
            sub = row.row(align=True)
            sub.alignment = "RIGHT"
            sub.label(text=right)
        return body
    box = layout.box()
    row = box.row(align=True)
    row.label(text=title, icon=icon)
    if right:
        sub = row.row(align=True)
        sub.alignment = "RIGHT"
        sub.label(text=right)
    return box


def _wrapped(layout, text, width=46, dim=False):
    col = layout.column(align=True)
    col.enabled = not dim
    for line in textwrap.wrap(str(text or ""), width=width) or [""]:
        col.label(text=line)


def _kv(layout, key, value, alert=False):
    split = layout.split(factor=0.42, align=True)
    k = split.row()
    k.enabled = False
    k.label(text=key)
    v = split.row()
    v.alert = alert
    v.label(text=str(value))


def _find_call(seq):
    for c in U.recent_calls():
        if c.get("seq") == seq:
            return c
    return None


# ---------------------------------------------------------------- operators behind the rows
class MIFBLENDER_OT_call_details(Operator):
    bl_idname = "mifblender.call_details"
    bl_label = "MifBlender call"
    bl_description = "What this call sent, how long it took, and what came back"
    bl_options = {"INTERNAL"}
    seq: IntProperty(default=-1)

    @classmethod
    def description(cls, context, properties):
        c = _find_call(properties.seq)
        if c is None:
            return cls.bl_description
        lines = ["%s - %s, %s" % (c.get("op", "?"), U.family_of(c.get("op", "")), U.ms_text(c.get("ms")))]
        if c.get("agent"):
            lines.append("Caller: " + c["agent"])
        if c.get("summary"):
            lines.append("Sent: " + c["summary"])
        if c.get("error"):
            lines.append("Refused: " + c["error"])
        lines.append("Click for the whole call")
        return "\n".join(lines)

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=460)

    def execute(self, context):
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        c = _find_call(self.seq)
        if c is None:
            layout.label(text="That call has left the journal's window.", icon="INFO")
            return
        kind = U.kind_of(c)
        row = layout.row()
        row.label(text=c.get("op", "?"), icon_value=U.chip(kind))
        right = row.row()
        right.alignment = "RIGHT"
        right.alert = kind == "refused" or (c.get("ms") or 0) >= U.SLOW_MS
        right.label(text=U.ms_text(c.get("ms")))
        col = layout.column(align=True)
        _kv(col, "Family", U.family_of(c.get("op", "")))
        _kv(col, "Caller", c.get("agent") or "not named - the client sent no agent")
        _kv(col, "Kind", {"read": "read - changes nothing", "write": "write - can change the scene",
                          "refused": "refused", "running": "still running"}[kind])
        _kv(col, "When", "%s (%s)" % (time.strftime("%H:%M:%S", time.localtime(c.get("t") or 0)),
                                      U.ago_text(c.get("t"))))
        if c.get("summary"):
            layout.separator()
            layout.label(text="Sent", icon="EXPORT")
            _wrapped(layout, c["summary"], dim=True)
        if c.get("error"):
            layout.separator()
            box = layout.box()
            box.alert = True
            box.label(text="Refused", icon="ERROR")
            _wrapped(box, c["error"])
            op = layout.operator(MIFBLENDER_OT_copy.bl_idname, text="Copy the error", icon="COPYDOWN")
            op.text = c["error"]


class MIFBLENDER_OT_check_details(Operator):
    bl_idname = "mifblender.check_details"
    bl_label = "MifBlender check"
    bl_description = "What this check found and what to do about it"
    bl_options = {"INTERNAL"}
    check_id: StringProperty(default="")

    @classmethod
    def description(cls, context, properties):
        row = next((r for r in U.cached_checks() if r["id"] == properties.check_id), None)
        return (row.get("detail") or row["text"]) if row else cls.bl_description

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=440)

    def execute(self, context):
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        row = next((r for r in U.cached_checks() if r["id"] == self.check_id), None)
        if row is None:
            layout.label(text="No such check.", icon="INFO")
            return
        head = layout.row()
        head.alert = row["status"] in ("fail", "warn")
        head.label(text=row["text"], icon=STATUS_ICON[row["status"]])
        _wrapped(layout, row.get("detail") or "", dim=False)
        if row.get("fix"):
            _fix_button(layout, row["fix"], big=True)


class MIFBLENDER_OT_check_fix(Operator):
    bl_idname = "mifblender.check_fix"
    bl_label = "Fix"
    bl_description = "Do the thing this check needs"
    bl_options = {"INTERNAL"}
    action: StringProperty(default="")

    def execute(self, context):
        pkg = U._addon()
        if self.action == "start" and pkg is not None:
            pkg.start_server()
        elif self.action == "prefs":
            try:
                bpy.ops.preferences.addon_show(module=__package__)
            except Exception:   # noqa: BLE001
                bpy.ops.screen.userpref_show()
        elif self.action == "addons":
            bpy.ops.screen.userpref_show()
            context.preferences.active_section = "ADDONS"
        elif self.action == "folder":
            bpy.ops.wm.path_open(filepath=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        elif self.action == "journal":
            path = U.journal_path()
            if path:
                bpy.ops.wm.path_open(filepath=os.path.dirname(path))
        U.health_checks(refresh_slow=True)
        U.tag_redraw_views()
        return {"FINISHED"}


_FIX_LABEL = {"start": ("Start", "PLAY"), "prefs": ("Prefs", "PREFERENCES"), "addons": ("Add-ons", "PLUGIN"),
              "folder": ("Folder", "FILE_FOLDER"), "journal": ("Open", "FILE_TEXT")}


def _fix_button(layout, action, big=False):
    text, icon = _FIX_LABEL.get(action, ("Fix", "TOOL_SETTINGS"))
    op = layout.operator(MIFBLENDER_OT_check_fix.bl_idname, text=text, icon=icon)
    op.action = action


class MIFBLENDER_OT_recheck(Operator):
    bl_idname = "mifblender.recheck"
    bl_label = "Re-run checks"
    bl_description = "Run every health check again now - the install files and the port probe included"

    def execute(self, context):
        U.health_checks(refresh_slow=True)
        U.tag_redraw_views()
        return {"FINISHED"}


class MIFBLENDER_OT_copy(Operator):
    bl_idname = "mifblender.copy_text"
    bl_label = "Copy"
    bl_description = "Copy this to the clipboard"
    bl_options = {"INTERNAL"}
    text: StringProperty(default="")

    def execute(self, context):
        context.window_manager.clipboard = self.text
        self.report({"INFO"}, "Copied")
        return {"FINISHED"}


class MIFBLENDER_OT_open_path(Operator):
    bl_idname = "mifblender.open_path"
    bl_label = "Open"
    bl_description = "Open this in the file browser"
    bl_options = {"INTERNAL"}
    path: StringProperty(default="")

    def execute(self, context):
        if self.path:
            bpy.ops.wm.path_open(filepath=self.path)
        return {"FINISHED"}


# ---------------------------------------------------------------- the panel
_FILTERS = [("ALL", "All", "Every call"), ("WRITES", "Writes", "Calls that can change the scene"),
            ("ERRORS", "Refused", "Calls that were refused")]
_OPS_BY_FAMILY = {"counts": None}


def _ops_by_family():
    if _OPS_BY_FAMILY["counts"] is None:
        counts = {}
        try:
            from . import server
            for name in server._op_table():
                fam = U.family_of(name)
                counts[fam] = counts.get(fam, 0) + 1
        except Exception:   # noqa: BLE001
            pass
        _OPS_BY_FAMILY["counts"] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return _OPS_BY_FAMILY["counts"]


def _filtered(calls, wm):
    mode = getattr(wm, "mifblender_filter", "ALL")
    needle = (getattr(wm, "mifblender_search", "") or "").lower().strip()
    out = []
    for c in calls:
        kind = U.kind_of(c)
        if mode == "WRITES" and kind not in ("write", "running"):
            continue
        if mode == "ERRORS" and kind != "refused":
            continue
        if needle and needle not in c.get("op", "").lower() and needle not in (c.get("summary") or "").lower():
            continue
        out.append(c)
    return out


def _chars(context):
    """About how many characters of label text fit across this region - so a line is cut where it ends, rather than
    Blender clipping it out of the middle."""
    ui = context.preferences.system.ui_scale
    return max(20, int((context.region.width - 44 * ui) / (6.3 * ui)))


def _call_row(layout, c, show_summary, chars=40):
    kind = U.kind_of(c)
    ms = c.get("ms")
    split = layout.split(factor=0.64, align=True)
    name = c.get("op", "?")
    if c.get("batch") is not None:
        name = "↳ %d · %s" % ((c.get("i") or 0) + 1, name)
    left = split.row(align=True)
    left.alignment = "LEFT"          # an unembossed button centres its text; a LEFT row sizes it to the text instead
    op = left.operator(MIFBLENDER_OT_call_details.bl_idname, text=name, icon_value=U.chip(kind), emboss=False)
    op.seq = int(c.get("seq", -1))
    right = split.row(align=True)
    right.alignment = "RIGHT"
    right.alert = kind == "refused" or (ms or 0) >= U.SLOW_MS
    right.label(text=U.ms_text(ms), icon={"refused": "ERROR", "running": "TIME"}.get(kind, "CHECKMARK"))
    if show_summary and (c.get("error") or c.get("summary")):
        text = c.get("error") or c.get("summary")
        lines = textwrap.wrap(text, max(12, chars - 4)) or [""]
        keep = 2 if kind == "refused" else 1          # why it was refused is worth a second line
        if len(lines) > keep:
            lines = lines[:keep]
            lines[-1] = lines[-1][:max(1, chars - 6)].rstrip() + "…"
        col = layout.column(align=True)
        for line in lines:
            sub = col.row()
            sub.scale_y = 0.72
            sub.alert = kind == "refused"
            sub.enabled = kind == "refused"
            sub.label(text="    " + line)


class VIEW3D_PT_mifblender(Panel):
    bl_label = "MifBlender"
    bl_idname = "VIEW3D_PT_mifblender"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MifBridge"

    def draw_header(self, context):
        self.layout.label(text="", icon_value=U.chip("purple"))

    def draw_header_preset(self, context):
        pkg = U._addon()
        server = getattr(pkg, "_server", None) if pkg else None
        running = bool(server is not None and server.running)
        row = self.layout.row(align=True)
        row.label(text=(":%d" % server.port) if running else "stopped",
                  icon_value=U.chip("ok" if running else "stopped"))

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        pkg = U._addon()
        settings = U._prefs()
        server = getattr(pkg, "_server", None) if pkg else None
        running = bool(server is not None and server.running)
        calls = U.recent_calls()
        stats = U.session_stats(calls)
        checks = U.cached_checks()

        # ---- the hero: is it listening, and the start / stop that answers "then make it"
        hero = layout.row(align=True)
        hero.scale_y = 1.5
        hero.label(text=("Listening on :%d" % server.port) if running else "Server stopped",
                   icon_value=U.chip("ok" if running else "stopped"))
        if running:
            hero.operator("mifblender.stop_server", text="", icon="PAUSE")
        else:
            hero.operator("mifblender.start_server", text="Start", icon="PLAY")
        sub = layout.row()
        sub.enabled = False
        clients = getattr(server, "_clients", 0) if running else 0
        last = "no calls yet"
        if calls and calls[0].get("t"):
            last = "last call %s" % time.strftime("%H:%M:%S", time.localtime(calls[0]["t"]))
        sub.label(text="pid %d · %d client%s · %s" % (os.getpid(), clients, "" if clients == 1 else "s", last))

        # ---- the numbers
        grid = layout.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
        slow = stats["slowest"]
        for title, value, icon, alert in (
                ("CALLS", str(stats["total"]), "CONSOLE", False),
                ("REFUSED", str(stats["errors"]), "ERROR" if stats["errors"] else "CHECKMARK", stats["errors"] > 0),
                ("WRITES", str(stats["writes"]), "GREASEPENCIL", False),
                ("SLOWEST", U.ms_text(slow.get("ms")) if slow else "-", "TIME",
                 bool(slow and (slow.get("ms") or 0) >= U.SLOW_MS))):
            box = grid.box()
            col = box.column(align=True)
            head = col.row()
            head.enabled = False
            head.label(text=title, icon=icon)
            val = col.row()
            val.scale_y = 1.25
            val.alert = alert
            val.label(text=value)
        notes = []
        if slow:
            notes.append("slowest: %s" % slow.get("op", "?"))
        if stats["total"] > stats["window"]:
            notes.append("counts from the last %d" % stats["window"])
        if stats["agents"]:
            col = layout.column(align=True)
            for i, (agent, n) in enumerate(stats["agents"][:3]):
                split = col.split(factor=0.8, align=True)
                split.enabled = False
                split.label(text=agent, icon="USER" if i == 0 else "BLANK1")
                cnt = split.row()
                cnt.alignment = "RIGHT"
                cnt.label(text=str(n))
        if notes:
            note = layout.row()
            note.enabled = False
            note.label(text="  ·  ".join(notes))

        # ---- C: health
        worst = "ok"
        for r in checks:
            if r["status"] == "fail" or (r["status"] == "warn" and worst == "ok"):
                worst = r["status"]
        n_ok = sum(1 for r in checks if r["status"] == "ok")
        box = layout.box()
        head = box.row(align=True)
        head.label(text="Health", icon=STATUS_ICON[worst] if worst != "ok" else "CHECKBOX_HLT")
        tail = head.row(align=True)
        tail.alignment = "RIGHT"
        n_bad = sum(1 for r in checks if r["status"] in ("fail", "warn"))
        tail.label(text=("all clear · %d ok" % n_ok) if not n_bad else
                   ("%d need%s a look" % (n_bad, "s" if n_bad == 1 else "")))
        tail.operator(MIFBLENDER_OT_recheck.bl_idname, text="", icon="FILE_REFRESH", emboss=False)
        col = box.column(align=True)
        for r in checks:
            if r["id"] == "server":
                continue    # the hero says it
            row = col.row(align=True)
            if r.get("fix") and r["status"] in ("fail", "warn"):
                split = row.split(factor=0.74, align=True)
                left = split.row(align=True)
                left.alignment = "LEFT"
                left.alert = True
                op = left.operator(MIFBLENDER_OT_check_details.bl_idname, text=r["text"],
                                   icon=STATUS_ICON[r["status"]], emboss=False)
                op.check_id = r["id"]
                _fix_button(split, r["fix"])
            else:
                row.alignment = "LEFT"
                row.alert = r["status"] in ("fail", "warn")
                op = row.operator(MIFBLENDER_OT_check_details.bl_idname, text=r["text"],
                                  icon=STATUS_ICON[r["status"]], emboss=False)
                op.check_id = r["id"]

        # ---- B: activity, grouped by family
        shown = _filtered(calls, wm)
        body = _section(layout, "mifblender_activity", "Activity", default_closed=False, icon="SORTTIME",
                        right="%d call%s" % (len(calls), "" if len(calls) == 1 else "s"))
        if body is not None:
            row = body.row(align=True)
            row.prop(wm, "mifblender_filter", expand=True)
            row = body.row(align=True)
            row.prop(wm, "mifblender_search", text="", icon="VIEWZOOM")
            row.prop(wm, "mifblender_show_summary", text="", icon="TEXT")
            if not calls:
                empty = body.box()
                empty.label(text="No calls yet", icon="INFO")
                _wrapped(empty, "Every call an AI makes through MifBlender lands here as it happens, grouped by "
                                "what it works on." if U.journal_version() or U._journal() else
                                "The call journal is not loaded in this build of the add-on.", dim=True)
            elif not shown:
                body.label(text="Nothing matches the filter.", icon="FILTER")
            else:
                order, groups = [], {}
                for c in shown:
                    fam = U.family_of(c.get("op", ""))
                    if fam not in groups:
                        order.append(fam)
                        groups[fam] = []
                    groups[fam].append(c)
                show_summary = getattr(wm, "mifblender_show_summary", True)
                chars = _chars(context)
                for fam in order:
                    fcalls = groups[fam]
                    slow_n = sum(1 for c in fcalls if (c.get("ms") or 0) >= U.SLOW_MS)
                    ref_n = sum(1 for c in fcalls if c.get("ok") is False)
                    total = sum((c.get("ms") or 0.0) for c in fcalls)
                    idname = "mifblender_fam_" + "".join(ch if ch.isalnum() else "_" for ch in fam.lower())
                    if hasattr(body, "panel"):
                        header, plate = body.panel(idname, default_closed=False)
                    else:
                        plate = body.box()
                        header = plate.row()
                    split = header.split(factor=0.6)
                    split.label(text=fam.upper(), icon_value=U.chip(U.FAMILY_COLOR.get(fam, "slate")))
                    tail = split.row(align=True)
                    tail.alignment = "RIGHT"
                    tail.alert = bool(slow_n or ref_n)
                    tail.label(text="%d · %s" % (len(fcalls), U.ms_text(total)),
                               icon="ERROR" if (slow_n or ref_n) else "NONE")
                    if plate is None:
                        continue
                    col = plate.column(align=True)
                    for c in fcalls[:6]:
                        _call_row(col, c, show_summary, chars)
                    if len(fcalls) > 6:
                        more = col.row()
                        more.enabled = False
                        more.label(text="      + %d more" % (len(fcalls) - 6))

        # the two ways out of the panel: the journal file, and the HUD over the viewport
        jp = U.journal_path()
        row = layout.row(align=True)
        row.scale_y = 1.2
        sub = row.row(align=True)
        sub.enabled = bool(jp)
        op = sub.operator(MIFBLENDER_OT_open_path.bl_idname, text="Journal", icon="FILE_TEXT")
        op.path = os.path.dirname(jp) if jp else ""
        on = bool(settings and getattr(settings, "hud_enabled", False))
        row.operator("mifblender.toggle_hud", text="Viewport HUD", icon="HIDE_OFF" if on else "HIDE_ON", depress=on)

        # ---- the rest, folded
        body = _section(layout, "mifblender_session", "Session", default_closed=True, icon="FILE_BLEND")
        if body is not None:
            col = body.column(align=True)
            blend = bpy.data.filepath
            _kv(col, "File", os.path.basename(blend) if blend else "unsaved")
            _kv(col, "Saved", "no - unsaved changes" if bpy.data.is_dirty else "yes", alert=bpy.data.is_dirty)
            try:
                from . import ops_undo
                _kv(col, "Bridge writes", "%d since load/save" % ops_undo.bridge_writes())
                st = ops_undo._STATE
                _kv(col, "Undo steps", "%d (at %d)" % (len(st.get("journal") or []), st.get("cursor") or 0))
            except Exception:   # noqa: BLE001
                pass
            sc = context.scene
            _kv(col, "Scene", "%s · %d objects" % (sc.name, len(sc.objects)))
            engine = sc.render.engine.replace("BLENDER_", "").replace("_", " ").title()
            _kv(col, "Render", "%s · frames %d-%d" % (engine, sc.frame_start, sc.frame_end))

        body = _section(layout, "mifblender_addon", "Add-on", default_closed=True, icon="PLUGIN")
        if body is not None:
            col = body.column(align=True)
            try:
                ver = ".".join(str(x) for x in pkg.bl_info.get("version", ()))
            except Exception:   # noqa: BLE001
                ver = "?"
            from . import server as mif_server
            _kv(col, "Version", "%s · protocol v%s" % (ver, getattr(mif_server, "PROTOCOL_VERSION", "?")))
            _kv(col, "Ops", "%d" % len(mif_server.op_names()))
            try:
                now_h, at_h = pkg.source_hash(), pkg.SOURCE_HASH_AT_IMPORT
                _kv(col, "Code", now_h or "?", alert=now_h != at_h)
                if now_h != at_h:
                    _kv(col, "Loaded", at_h or "?", alert=True)
            except Exception:   # noqa: BLE001
                pass
            folder = os.path.dirname(os.path.abspath(__file__))
            row = body.row(align=True)
            op = row.operator(MIFBLENDER_OT_open_path.bl_idname, text="Folder", icon="FILE_FOLDER")
            op.path = folder
            op = row.operator(MIFBLENDER_OT_copy.bl_idname, text="Copy path", icon="COPYDOWN")
            op.text = folder
            fams = _ops_by_family()
            if fams:
                sub = body.column(align=True)
                sub.enabled = False
                line = []
                for fam, n in fams:
                    line.append("%s %d" % (fam, n))
                _wrapped(sub, " · ".join(line), width=44, dim=True)

        body = _section(layout, "mifblender_settings", "Settings", default_closed=True, icon="PREFERENCES")
        if body is not None and settings is not None:
            col = body.column(align=True)
            col.prop(settings, "port")
            col.prop(settings, "token")
            col.prop(settings, "job_timeout")
            body.prop(settings, "auto_start")
            body.prop(settings, "allow_run_python")
            if hasattr(settings, "allow_generative"):
                body.prop(settings, "allow_generative")
            body.prop(settings, "verbose")
            hud = body.box()
            hud.label(text="Viewport HUD", icon="OVERLAY")
            for name in ("hud_corner", "hud_max_cards", "hud_auto_collapse", "hud_idle_seconds", "hud_opacity",
                         "hud_show_summary"):
                if hasattr(settings, name):
                    hud.prop(settings, name)
            note = body.row()
            note.enabled = False
            note.label(text="Port and token apply when the server restarts.")


CLASSES = (MIFBLENDER_OT_call_details, MIFBLENDER_OT_check_details, MIFBLENDER_OT_check_fix, MIFBLENDER_OT_recheck,
           MIFBLENDER_OT_copy, MIFBLENDER_OT_open_path, VIEW3D_PT_mifblender)


def register():
    bpy.types.WindowManager.mifblender_filter = EnumProperty(name="Show", items=_FILTERS, default="ALL")
    bpy.types.WindowManager.mifblender_search = StringProperty(name="Search", default="",
                                                               description="Only calls whose op or parameters "
                                                                           "contain this")
    bpy.types.WindowManager.mifblender_show_summary = BoolProperty(name="Details", default=True,
                                                                   description="Show what each call sent")
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:   # noqa: BLE001
            pass
    for name in ("mifblender_filter", "mifblender_search", "mifblender_show_summary"):
        try:
            delattr(bpy.types.WindowManager, name)
        except Exception:   # noqa: BLE001
            pass
