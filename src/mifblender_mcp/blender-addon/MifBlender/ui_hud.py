"""The viewport HUD: the bridge's last calls drawn over the 3D view, toggled from the sidebar or the Overlays popover.

WHY IT IS BUILT THIS WAY, 2026-09-21. It was picked from four mocked directions ("D toggle") with the bar set: "as
polished as physically possible without jaking it GPU lag". Those two pull against each other, so the design is the
answer to both:

  - ONE BATCH. Every shape - the shadow, the panel, the header gradient, the cards, their rails - is one triangle list
    drawn with one shader call. Rounded corners and anti-aliased edges are geometry (a 1 px fringe that fades to
    transparent), not extra passes.
  - BUILT ONLY WHEN SOMETHING CHANGED. The geometry and every string, already measured and cut to fit, are cached per
    region under a key: the journal's version, the region size, the UI scale, the preferences, the server state and
    the health checks' statuses. A frame where nothing changed draws the cached batch and the cached strings - no
    Python geometry, no text measuring.
  - THE ONE ANIMATION IS SMALL. When a call lands its card glows for 1.2 s; the glow is a separate few-dozen-vertex
    batch rebuilt only while it fades. ui_common's timer asks for redraws during the fade and never otherwise.

It draws only in the main region of each 3D view (not the side panes of a quad view), respects the viewport's
Overlays switch like Blender's own overlays, keeps clear of the toolbar and sidebar when they overlap the view, and
steps aside entirely when the view is too small to hold it.

The builtin shader names lost their 2D_ prefix in Blender 3.4-4.0 and blf.size lost its dpi argument in 4.0; both
are handled, so this draws the same from 3.6 to 5.2. In background mode nothing draws, and no shader is ever created.
"""
import math
import os

import blf
import bpy
import gpu
from bpy.types import Operator
from gpu_extras.batch import batch_for_shader

from . import ui_common as U

FONT = 0
_HANDLE = {"draw": None, "overlay": False, "broken": None}
_SHADER = {"smooth": None, "failed": False}
_CACHE = {}          # region pointer -> {"key", "batch", "texts", "cards", "newest"}


# ---------------------------------------------------------------- geometry: rounded rectangles with an AA fringe
class _Geo:
    """Positions, colours and triangle indices for one SMOOTH_COLOR batch."""

    def __init__(self):
        self.pos, self.col, self.idx = [], [], []

    def rrect(self, x, y, w, h, r, c0, c1=None, fringe=1.0, corners=(True, True, True, True)):
        """A rounded rectangle from (x, y), w by h, corner radius r, coloured c0 -> c1 left to right.

        fringe is the width over which the edge fades to transparent, centred on the true edge: 1 px is an
        anti-aliased edge, ten is a soft shadow. corners picks which corners are rounded (bl, br, tr, tl).
        """
        if w <= 0 or h <= 0:
            return
        c1 = c1 or c0
        r = max(0.0, min(r, w / 2.0, h / 2.0))
        half = fringe / 2.0
        segs = max(2, min(10, int(r * 0.7)))
        centres = ((x + r, y + r, math.pi), (x + w - r, y + r, 1.5 * math.pi),
                   (x + w - r, y + h - r, 0.0), (x + r, y + h - r, 0.5 * math.pi))
        sharp = ((x, y, -1, -1), (x + w, y, 1, -1), (x + w, y + h, 1, 1), (x, y + h, -1, 1))
        edge = []                       # (px, py, nx, ny): a point on the true edge and its outward normal
        for k in range(4):
            if corners[k] and r > 0.5:
                cx, cy, a0 = centres[k]
                for s in range(segs + 1):
                    a = a0 + (0.5 * math.pi) * s / segs
                    nx, ny = math.cos(a), math.sin(a)
                    edge.append((cx + nx * r, cy + ny * r, nx, ny))
            else:
                px, py, sx, sy = sharp[k]
                edge.append((px, py, float(sx), float(sy)))   # a mitred corner: 1 px out in both axes

        def colour(px, alpha_scale):
            t = 0.0 if w <= 0 else min(1.0, max(0.0, (px - x) / w))
            return (c0[0] + (c1[0] - c0[0]) * t, c0[1] + (c1[1] - c0[1]) * t, c0[2] + (c1[2] - c0[2]) * t,
                    (c0[3] + (c1[3] - c0[3]) * t) * alpha_scale)

        base = len(self.pos)
        n = len(edge)
        self.pos.append((x + w / 2.0, y + h / 2.0))
        self.col.append(colour(x + w / 2.0, 1.0))
        for px, py, nx, ny in edge:                               # inner ring: solid, half a fringe inside the edge
            ix, iy = px - nx * half, py - ny * half
            self.pos.append((ix, iy))
            self.col.append(colour(ix, 1.0))
        for px, py, nx, ny in edge:                               # outer ring: transparent, half a fringe outside
            ox, oy = px + nx * half, py + ny * half
            self.pos.append((ox, oy))
            self.col.append(colour(ox, 0.0))
        for i in range(n):
            j = (i + 1) % n
            self.idx.append((base, base + 1 + i, base + 1 + j))
            self.idx.append((base + 1 + i, base + 1 + n + i, base + 1 + n + j))
            self.idx.append((base + 1 + i, base + 1 + n + j, base + 1 + j))

    def ring(self, x, y, w, h, r, colour, width):
        """A soft ring hugging a rounded rectangle from the outside: full colour at its edge, gone `width` px out."""
        r = max(0.0, min(r, w / 2.0, h / 2.0))
        segs = max(2, min(10, int(r * 0.7)))
        centres = ((x + r, y + r, math.pi), (x + w - r, y + r, 1.5 * math.pi),
                   (x + w - r, y + h - r, 0.0), (x + r, y + h - r, 0.5 * math.pi))
        edge = []
        for cx, cy, a0 in centres:
            for s in range(segs + 1):
                a = a0 + (0.5 * math.pi) * s / segs
                edge.append((cx + math.cos(a) * r, cy + math.sin(a) * r, math.cos(a), math.sin(a)))
        base = len(self.pos)
        n = len(edge)
        faded = (colour[0], colour[1], colour[2], 0.0)
        for px, py, nx, ny in edge:
            self.pos.append((px - nx * 0.5, py - ny * 0.5))
            self.col.append(colour)
        for px, py, nx, ny in edge:
            self.pos.append((px + nx * width, py + ny * width))
            self.col.append(faded)
        for i in range(n):
            j = (i + 1) % n
            self.idx.append((base + i, base + n + i, base + n + j))
            self.idx.append((base + i, base + n + j, base + j))

    def batch(self, shader):
        if not self.idx:
            return None
        return batch_for_shader(shader, "TRIS", {"pos": self.pos, "color": self.col}, indices=self.idx)


def _shader():
    if _SHADER["smooth"] is None and not _SHADER["failed"]:
        for name in ("SMOOTH_COLOR", "2D_SMOOTH_COLOR"):
            try:
                _SHADER["smooth"] = gpu.shader.from_builtin(name)
                break
            except Exception:   # noqa: BLE001 - the other spelling, on the other side of 4.0
                continue
        _SHADER["failed"] = _SHADER["smooth"] is None
    return _SHADER["smooth"]


# ---------------------------------------------------------------- text: measured once, when the layout is built
def _size(px):
    try:
        blf.size(FONT, px)
    except TypeError:          # 3.6: size(fontid, size, dpi)
        blf.size(FONT, int(round(px)), 72)


def _width(text, px):
    _size(px)
    return blf.dimensions(FONT, text)[0]


def _fit(text, px, room):
    """text, cut with an ellipsis to fit `room` pixels at size px."""
    text = str(text or "")
    if room <= 0:
        return ""
    if _width(text, px) <= room:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _width(text[:mid].rstrip() + "…", px) <= room:
            lo = mid
        else:
            hi = mid - 1
    return (text[:lo].rstrip() + "…") if lo else ""


def _mix(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))


def _bright(key, amount=0.35):
    return _mix(U.rgba(key), (1.0, 1.0, 1.0, 1.0), amount)


# ---------------------------------------------------------------- where the HUD may sit in this region
def _insets(area, region):
    """(left, right, top) pixels of this region covered by an overlapping toolbar, sidebar or header. Blender 5.x
    draws the headers over the view too, so the view's own text starts below them - and so must the HUD."""
    left = right = top = 0
    for r in area.regions:
        if r.width <= 1 or r.height <= 1 or r.as_pointer() == region.as_pointer():
            continue
        if r.type == "UI" and region.x <= r.x < region.x + region.width:
            right = max(right, region.x + region.width - r.x)
        elif r.type == "TOOLS" and r.x <= region.x < r.x + r.width:
            left = max(left, r.x + r.width - region.x)
        elif r.type in ("HEADER", "TOOL_HEADER") and region.y <= r.y < region.y + region.height:
            top = max(top, region.y + region.height - r.y)
    return left, right, top


def _settings_key(s):
    if s is None:
        return None
    return (getattr(s, "hud_corner", "TOP_LEFT"), int(getattr(s, "hud_max_cards", 6)),
            round(float(getattr(s, "hud_opacity", 0.92)), 3), bool(getattr(s, "hud_show_summary", True)),
            bool(getattr(s, "allow_run_python", True)))


# ---------------------------------------------------------------- the layout
def _build(context, region, settings, key):
    ui = context.preferences.system.ui_scale
    area = context.area
    space = context.space_data
    corner, max_cards, opacity, show_summary, run_py = key[3]
    left_in, right_in, top_in = key[5]
    pkg = U._addon()
    server = getattr(pkg, "_server", None) if pkg else None
    running = bool(server is not None and server.running)
    calls = U.recent_calls()
    stats = U.session_stats(calls)
    checks = [r for r in U.cached_checks() if r["status"] in ("fail", "warn") and r["id"] != "server"]

    def px(v):
        return v * ui

    margin, pad = px(14), px(12)
    W = min(px(348), region.width - left_in - right_in - 2 * margin)
    if W < px(230):
        return None
    if key[12]:
        return _build_strip(context, region, key, stats, checks, running, px, margin, pad, left_in, right_in, top_in)
    head_h = px(84)
    card_h = px(44) if show_summary else px(28)
    gap = px(5)
    alert_h = px(17) * min(2, len(checks))
    foot_h = px(22)

    # How tall may it be? Below the view's own text at the top left, clear of the navigation gizmo at the top right.
    top_clear = top_in + px(12)
    if corner == "TOP_LEFT":
        overlay = getattr(space, "overlay", None)
        lines = (2 if getattr(overlay, "show_text", True) else 0) + (9 if getattr(overlay, "show_stats", False) else 0)
        top_clear = top_in + px(16) + lines * px(17)
    usable = region.height - top_clear - px(16)
    fixed = head_h + alert_h + foot_h + pad
    if usable < fixed + px(30):
        return None
    fits = int((usable - fixed) // (card_h + gap))
    n_cards = max(0, min(max_cards, len(calls), fits))
    H = fixed + (n_cards * (card_h + gap) if calls else px(30))
    gizmo_w = px(96) if (corner == "TOP_RIGHT" and key[11]) else 0

    if corner.endswith("LEFT"):
        x0 = left_in + margin
    else:
        x0 = region.width - right_in - margin - gizmo_w - W
    if corner.startswith("TOP"):
        y0 = region.height - top_clear - H
    else:
        y0 = px(16)
    x0, y0 = int(x0), int(y0)

    geo = _Geo()
    texts = []

    def text(x, y, s, size, colour, right=False, bold=False, room=None):
        s = _fit(s, size, room) if room is not None else str(s)
        if not s:
            return
        if right:
            x -= _width(s, size)
        texts.append((x, y, s, size, colour, bold))

    bg_a = 0.90 * opacity
    rad = px(9)
    # the shadow, the panel, the header gradient with its top corners only rounded
    geo.rrect(x0 - px(2), y0 - px(5), W + px(4), H + px(4), rad + px(4), (0.0, 0.0, 0.0, 0.40 * opacity),
              fringe=px(14))
    geo.rrect(x0, y0, W, H, rad, U.rgba("ground", bg_a))
    geo.rrect(x0, y0 + H - head_h, W, head_h, rad, U.rgba("railL", 0.97 * opacity), U.rgba("railR", 0.97 * opacity),
              corners=(False, False, True, True))
    geo.rrect(x0, y0 + H - head_h, W, px(1), 0, U.rgba("purpleSoft", 0.35 * opacity))    # a hairline under it

    top = y0 + H
    # header line 1: the name, and the status pill
    text(x0 + pad, top - px(24), "MIFBLENDER", px(15), U.rgba("ink"), bold=True)
    pill_key = "ok" if running else "stopped"
    pill_text = "LIVE" if running else "STOPPED"
    pw = _width(pill_text, px(9.5)) + px(22)
    pill_x, pill_y, pill_h = x0 + W - pad - pw, top - px(28), px(17)
    geo.rrect(pill_x, pill_y, pw, pill_h, pill_h / 2.0, U.rgba(pill_key, 0.20 * max(opacity, 0.6)))
    geo.rrect(pill_x + px(6), pill_y + pill_h / 2.0 - px(3), px(6), px(6), px(3), U.rgba(pill_key))
    text(pill_x + px(15), pill_y + px(5), pill_text, px(9.5), _bright(pill_key, 0.25))
    # header line 2: where, who, and what it allows
    where = (":%d" % server.port) if running else "not listening"
    line2 = "%s  ·  pid %d  ·  run_python %s  ·  Blender %s" % (
        where, os.getpid(), "on" if run_py else "off", bpy.app.version_string.split(" ")[0])
    text(x0 + pad, top - px(44), line2, px(9.5), U.rgba("purpleSoft"), room=W - 2 * pad)
    # header line 3: the numbers
    slow = stats["slowest"]
    cols = (("CALLS", str(stats["total"]), U.rgba("ink")),
            ("REFUSED", str(stats["errors"]), U.rgba("refused") if stats["errors"] else U.rgba("ink")),
            ("WRITES", str(stats["writes"]), U.rgba("ink")),
            ("SLOWEST", U.ms_text(slow.get("ms")) if slow else "-",
             U.rgba("warn") if slow and (slow.get("ms") or 0) >= U.SLOW_MS else U.rgba("ink")))
    cw = (W - 2 * pad) / 4.0
    for i, (label, value, colour) in enumerate(cols):
        cx = x0 + pad + i * cw
        text(cx, top - px(62), label, px(7.5), _mix(U.rgba("purpleSoft"), U.rgba("ink2"), 0.5))
        text(cx, top - px(78), value, px(12.5), colour, room=cw - px(4))

    y = top - head_h
    # health that needs attention, straight under the header
    for r in checks[:2]:
        y -= px(17)
        tone = "refused" if r["status"] == "fail" else "warn"
        geo.rrect(x0 + pad, y + px(5), px(5), px(5), px(2.5), U.rgba(tone))
        text(x0 + pad + px(11), y + px(3), r["text"], px(9.5), _bright(tone, 0.15), room=W - 2 * pad - px(11))

    # the cards
    cards = []
    y -= gap + (0 if checks else px(2))
    card_r = px(6)
    for c in calls[:n_cards]:
        kind = U.kind_of(c)
        sub = c.get("batch") is not None
        indent = px(12) if sub else 0
        cx0, cw_ = x0 + pad / 2.0 + indent, W - pad - indent
        y -= card_h
        geo.rrect(cx0, y, cw_, card_h, card_r, U.rgba("plate", 0.96 * max(opacity, 0.5)))
        geo.rrect(cx0 + px(5), y + px(6), px(3), card_h - px(12), px(1.5), U.rgba(kind))
        ms = c.get("ms")
        slow_call = (ms or 0) >= U.SLOW_MS
        ms_col = U.rgba("refused") if kind == "refused" else (U.rgba("warn") if slow_call else U.rgba("ink2"))
        ms_s = U.ms_text(ms)
        ms_w = _width(ms_s, px(10.5))
        tx = cx0 + px(15)
        line1_y = y + card_h - (px(18) if show_summary else px(18.5))
        text(cx0 + cw_ - px(10), line1_y, ms_s, px(10.5), ms_col, right=True)
        label = ("%s  %d · %s" % ("↳", (c.get("i") or 0) + 1, c.get("op", "?"))) if sub else c.get("op", "?")
        room = cw_ - px(15) - ms_w - px(18)
        text(tx, line1_y, label, px(12), U.rgba("ink"), room=room)
        agent = c.get("agent") if not sub else None        # a batch's ops share its caller; the batch row says it
        if agent:
            used = _width(_fit(label, px(12), room), px(12)) + px(8)
            text(tx + used, line1_y + px(1), agent, px(9), U.rgba("ink3"), room=room - used)
        if show_summary:
            fam = U.family_of(c.get("op", ""))
            fam_s = fam.upper()
            fam_px = px(8)
            fam_w = _width(fam_s, fam_px)
            text(tx, y + px(9), fam_s, fam_px, _bright(U.FAMILY_COLOR.get(fam, "slate"), 0.45))
            detail = c.get("error") if kind == "refused" else (c.get("summary") or "")
            text(tx + fam_w + px(7), y + px(9), detail, px(9.5),
                 _bright("refused", 0.1) if kind == "refused" else U.rgba("ink3"),
                 room=cw_ - px(15) - fam_w - px(17))
        cards.append((cx0, y, cw_, card_h, card_r))
        y -= gap
    # the footer: what was left out, or the empty state
    if not calls:
        y -= px(24)
        text(x0 + pad, y + px(6), "No calls yet. Every call an AI makes through MifBlender lands here.", px(9.5),
             U.rgba("ink2"), room=W - 2 * pad)
    more = len(calls) - n_cards
    foot = ("+%d earlier  ·  " % more if more > 0 else "") + "N panel › MifBridge for details"
    text(x0 + pad, y0 + px(8), foot, px(8.5), U.rgba("ink3"), room=W - 2 * pad)

    shader = _shader()
    return {"key": key, "batch": geo.batch(shader) if shader else None, "texts": texts, "cards": cards,
            "newest": cards[0] if cards else None, "ui": ui}


def _build_strip(context, region, key, stats, checks, running, px, margin, pad, left_in, right_in, top_in):
    """IDLE: the header alone, one line high - the name, how many calls, and whether it is listening. A call opens the
    full HUD again; nothing is lost, it is only out of the way while nothing is happening."""
    corner, opacity = key[3][0], key[3][2]
    space = context.space_data
    W, H = px(292), px(30)
    top_clear = top_in + px(12)
    if corner == "TOP_LEFT":
        overlay = getattr(space, "overlay", None)
        lines = (2 if getattr(overlay, "show_text", True) else 0) + (9 if getattr(overlay, "show_stats", False) else 0)
        top_clear = top_in + px(16) + lines * px(17)
    if region.height - top_clear < H + px(16):
        return None
    gizmo_w = px(96) if (corner == "TOP_RIGHT" and key[11]) else 0
    x0 = left_in + margin if corner.endswith("LEFT") else region.width - right_in - margin - gizmo_w - W
    y0 = region.height - top_clear - H if corner.startswith("TOP") else px(16)
    x0, y0 = int(x0), int(y0)
    geo = _Geo()
    texts = []
    geo.rrect(x0 - px(2), y0 - px(4), W + px(4), H + px(3), px(11), (0.0, 0.0, 0.0, 0.35 * opacity), fringe=px(12))
    geo.rrect(x0, y0, W, H, px(8), U.rgba("railL", 0.95 * opacity), U.rgba("railR", 0.95 * opacity))
    base = y0 + px(10)
    texts.append((x0 + pad, base, "MIFBLENDER", px(11.5), U.rgba("ink"), True))
    title_w = _width("MIFBLENDER", px(11.5)) + px(10)
    pill_key = "ok" if running else "stopped"
    pill_text = "LIVE" if running else "STOPPED"
    pw = _width(pill_text, px(8.5)) + px(20)
    pill_x, pill_h = x0 + W - px(8) - pw, px(16)
    pill_y = y0 + (H - pill_h) / 2.0
    geo.rrect(pill_x, pill_y, pw, pill_h, pill_h / 2.0, U.rgba(pill_key, 0.22))
    geo.rrect(pill_x + px(6), pill_y + pill_h / 2.0 - px(2.5), px(5), px(5), px(2.5), U.rgba(pill_key))
    texts.append((pill_x + px(14), pill_y + px(4.5), pill_text, px(8.5), _bright(pill_key, 0.25), False))
    if stats["total"]:
        middle = "%d call%s" % (stats["total"], "" if stats["total"] == 1 else "s")
        if stats["errors"]:
            middle += "  ·  %d refused" % stats["errors"]
    else:
        middle = "no calls yet"
    room = pill_x - (x0 + pad + title_w) - px(8)
    if checks:
        tone = "refused" if any(r["status"] == "fail" for r in checks) else "warn"
        geo.rrect(pill_x - px(12), y0 + H / 2.0 - px(3), px(6), px(6), px(3), U.rgba(tone))
        room -= px(12)
    s = _fit(middle, px(9.5), room)
    if s:
        texts.append((x0 + pad + title_w, base + px(0.5), s, px(9.5), U.rgba("purpleSoft"), False))
    shader = _shader()
    return {"key": key, "batch": geo.batch(shader) if shader else None, "texts": texts, "cards": [],
            "newest": None, "ui": context.preferences.system.ui_scale}


# ---------------------------------------------------------------- the draw callback
def _draw():
    context = bpy.context
    settings = U._prefs()
    if settings is None or not getattr(settings, "hud_enabled", False) or _HANDLE["broken"]:
        return
    space, region, area = context.space_data, context.region, context.area
    if space is None or region is None or area is None:
        return
    overlay = getattr(space, "overlay", None)
    if overlay is not None and not overlay.show_overlays:
        return
    try:
        if context.region_data is not None and space.region_3d is not None and \
                context.region_data.as_pointer() != space.region_3d.as_pointer():
            return                   # a locked pane of a quad view; the HUD belongs to the main one
    except Exception:   # noqa: BLE001
        pass
    shader = _shader()
    if shader is None:
        return
    try:
        pkg = U._addon()
        server = getattr(pkg, "_server", None) if pkg else None
        running = bool(server is not None and server.running)
        # EVERYTHING THE LAYOUT DEPENDS ON. _build reads key[3] (the settings) and key[5] (the insets) back out.
        key = (U.journal_version(), region.width, region.height,
               _settings_key(settings) or ("TOP_LEFT", 6, 0.92, True, True),
               round(context.preferences.system.ui_scale, 3), _insets(area, region), running,
               getattr(server, "port", None), tuple((r["id"], r["status"]) for r in U.cached_checks()),
               bool(getattr(overlay, "show_text", True)), bool(getattr(overlay, "show_stats", False)),
               bool(getattr(space, "show_gizmo", True)), U.hud_idle(settings))
        ptr = region.as_pointer()
        entry = _CACHE.get(ptr)
        if entry is None or entry["key"] != key:
            if len(_CACHE) > 16:
                _CACHE.clear()
            entry = _build(context, region, settings, key) or {"key": key, "batch": None, "texts": [],
                                                                  "cards": [], "newest": None}
            _CACHE[ptr] = entry
        if entry["batch"] is None and not entry["texts"]:
            return
        gpu.state.blend_set("ALPHA")
        if entry["batch"] is not None:
            shader.bind()
            entry["batch"].draw(shader)
        fade = U.fade_amount()
        if fade > 0.0 and entry["newest"] is not None:
            x, y, w, h, r = entry["newest"]
            geo = _Geo()
            ui = entry.get("ui", 1.0)
            geo.ring(x, y, w, h, r, U.rgba("purple", 0.85 * fade), 7.0 * ui)
            geo.rrect(x, y, w, h, r, U.rgba("purple", 0.16 * fade))
            glow = geo.batch(shader)
            if glow is not None:
                glow.draw(shader)
        for x, y, s, size, colour, bold in entry["texts"]:
            _size(size)
            blf.color(FONT, *colour)
            blf.position(FONT, x, y, 0)
            blf.draw(FONT, s)
            if bold:
                blf.position(FONT, x + 0.7, y, 0)
                blf.draw(FONT, s)
        gpu.state.blend_set("NONE")
    except Exception as exc:   # noqa: BLE001 - a draw callback that raises would print it every frame
        _CACHE.clear()
        try:
            gpu.state.blend_set("NONE")
        except Exception:   # noqa: BLE001
            pass
        _HANDLE["broken"] = "%s: %s" % (type(exc).__name__, exc)
        print("[MifBlender] the viewport HUD stopped drawing: %s. Toggle it to try again." % _HANDLE["broken"])


# ---------------------------------------------------------------- the toggle, and its line in the Overlays popover
class MIFBLENDER_OT_toggle_hud(Operator):
    bl_idname = "mifblender.toggle_hud"
    bl_label = "MifBlender HUD"
    bl_description = "Show or hide the bridge's last calls over the 3D viewport"

    def execute(self, context):
        settings = U._prefs()
        if settings is None:
            self.report({"WARNING"}, "MifBlender's preferences are not available")
            return {"CANCELLED"}
        settings.hud_enabled = not settings.hud_enabled
        _HANDLE["broken"] = None
        U.touch_activity()                 # switched on, it opens - not a collapsed strip
        _CACHE.clear()
        U.tag_redraw_views()
        return {"FINISHED"}


def _overlay_line(self, context):
    settings = U._prefs()
    if settings is None:
        return
    layout = self.layout
    layout.separator()
    row = layout.row(align=True)
    row.prop(settings, "hud_enabled", text="MifBlender call HUD")
    sub = row.row(align=True)
    sub.active = settings.hud_enabled
    sub.prop(settings, "hud_corner", text="")


def on_setting_changed(_self=None, _context=None):
    """The preferences' update callback: drop the cached layouts, forget a failure, and redraw."""
    _HANDLE["broken"] = None
    U.touch_activity()
    _CACHE.clear()
    try:
        U.tag_redraw_views()
    except Exception:   # noqa: BLE001 - called while preferences load, before there is a window
        pass


CLASSES = (MIFBLENDER_OT_toggle_hud,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    if _HANDLE["draw"] is None:
        _HANDLE["draw"] = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_PIXEL")
    panel = getattr(bpy.types, "VIEW3D_PT_overlay", None)
    if panel is not None and not _HANDLE["overlay"]:
        panel.append(_overlay_line)
        _HANDLE["overlay"] = True


def unregister():
    if _HANDLE["overlay"]:
        panel = getattr(bpy.types, "VIEW3D_PT_overlay", None)
        try:
            panel.remove(_overlay_line)
        except Exception:   # noqa: BLE001
            pass
        _HANDLE["overlay"] = False
    if _HANDLE["draw"] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_HANDLE["draw"], "WINDOW")
        except Exception:   # noqa: BLE001
            pass
        _HANDLE["draw"] = None
    _CACHE.clear()
    _SHADER["smooth"] = None
    _SHADER["failed"] = False
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:   # noqa: BLE001
            pass
